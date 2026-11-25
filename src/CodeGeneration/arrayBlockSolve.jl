#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF AGPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GNU AGPL
* VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the OSMC (Open Source Modelica Consortium)
* Public License (OSMC-PL) are obtained from OSMC, either from the above
* address, from the URLs:
* http://www.openmodelica.org or
* https://github.com/OpenModelica/ or
* http://www.ida.liu.se/projects/OpenModelica,
* and in the OpenModelica distribution.
*
* GNU AGPL version 3 is obtained from:
* https://www.gnu.org/licenses/licenses.html#GPL
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#

#=
Large algebraic loops of an array model (blocks of more than NEWTON_BLOCK_SIZE equation instances: loops that couple
the elements, MSL EngineV6's cylinders through the crankshaft), torn (`_tearBlock`), solved inside its equations, so the
solver integrates the states only: Newton on the residuals in the iteration variables, the Jacobian by colored finite
differences and its factorization kept from call to call (a simplified Newton, refreshed where it would not converge
in time), from the last solution.
=#

const _SPARSE = ModelingToolkit.Symbolics.SparseArrays
const _LINALG = ModelingToolkit.Symbolics.SparseArrays.LinearAlgebra
const _UMFPACK_LU = _SPARSE.UMFPACK.UmfpackLU{Float64, Int}

"Blocks of up to this many iteration variables are factored dense (LAPACK, in place), larger ones sparse (UMFPACK)."
const DENSE_BLOCK_SIZE = 128

"Singular values of a dense block's Jacobian below this share of the largest are zero (`_blockJacobian!`)."
const RANK_TOLERANCE = 1.0e-6

"""
The most column colors of a linear loop's Jacobian for its direct solve in every call (`_blockLinear!`): a Jacobian
costs one evaluation per color, Newton with a kept one about eight in all (MSL EngineV6's loops of two unknowns).
"""
const LINEAR_DIRECT_COLORS = 4

"""
    columnColors(S) -> Vector{Int}

A greedy coloring of the columns of the sparse pattern `S`: columns that share no row get one color, so one residual
evaluation perturbing all of a color's columns gives each of their Jacobian columns (Curtis, Powell, Reid).
"""
function columnColors(S::_SPARSE.SparseMatrixCSC{Float64, Int})::Vector{Int}
  local colors::Vector{Int} = zeros(Int, size(S, 2))
  local used::Vector{BitVector} = BitVector[]
  for j in 1:size(S, 2)
    local rows::Vector{Int} = S.rowval[S.colptr[j]:(S.colptr[j + 1] - 1)]
    local c::Union{Nothing, Int} = findfirst(u::BitVector -> !any(r::Int -> u[r], rows), used)
    if c === nothing
      push!(used, falses(size(S, 1)))
      c = length(used)
    end
    for r::Int in rows
      used[c][r] = true
    end
    colors[j] = c
  end
  return colors
end

"""
    BlockSolver(pattern, colors, guess, linear = false; nominal = ones(length(guess)))

The solver of one large algebraic loop of an array model (`blockSolve!`): the columns of each color of its residuals'
sparsity in its iteration variables, the Jacobian on that sparsity and its factorization, kept between calls (dense
up to `DENSE_BLOCK_SIZE` variables: the factors in place with their pivots; else sparse), the last solution (the next
call's guess; the start values until a solve succeeded), the largest of its residuals (`rsolved`; -1 before the
first) and counters (residual evaluations, Jacobians, failed solves).
`rankDeficient`: the dense Jacobian was singular and `factors` holds its pseudo-inverse (`_blockJacobian!`).
`linear`: the residuals are affine in the iteration variables (`_affineIn` of the explicit solutions and residuals).
`held`: of a linear loop, the discrete state at its last Jacobian (`_holdDiscretes!`), which its kept factors are for.
`nominal`: the iteration variables' nominal values (1 without the attribute), the scale of the differences, the steps
and the varied starts below 1 in size: as in OpenModelica's solvers, max(nominal, |x|) where the scale was max(1, |x|).
"""
mutable struct BlockSolver
  const groups::Vector{Vector{Int}}
  const J::_SPARSE.SparseMatrixCSC{Float64, Int}
  const dense::Bool
  const linear::Bool
  const factors::Matrix{Float64}
  const pivots::Vector{_LINALG.BlasInt}
  F::Union{Nothing, _UMFPACK_LU}
  factored::Bool
  rankDeficient::Bool
  const y::Vector{Float64}
  const x::Vector{Float64}
  const r::Vector{Float64}
  const rp::Vector{Float64}
  const xp::Vector{Float64}
  const dx::Vector{Float64}
  const h::Vector{Float64}
  const held::Vector{Float64}
  const nominal::Vector{Float64}
  solved::Bool
  rsolved::Float64
  residuals::Int
  jacobians::Int
  failures::Int
end

"""
A copy of a solver for a copied problem (`deepcopy(prob.p)`): its buffers and dense factors copied, a sparse
factorization not (the copy factors again: a copied UMFPACK object shared its C memory with the original's).
"""
function Base.deepcopy_internal(s::BlockSolver, dict::IdDict)::BlockSolver
  haskey(dict, s) && return dict[s]
  local c::BlockSolver = BlockSolver(Vector{Int}[copy(g) for g in s.groups], copy(s.J), s.dense, s.linear, copy(s.factors), copy(s.pivots),
                                     nothing, s.dense && s.factored, s.rankDeficient, copy(s.y), copy(s.x), copy(s.r),
                                     copy(s.rp), copy(s.xp),
                                     copy(s.dx), copy(s.h), copy(s.held), copy(s.nominal), s.solved, s.rsolved,
                                     s.residuals, s.jacobians,
                                     s.failures)
  dict[s] = c
  return c
end

function BlockSolver(pattern::_SPARSE.SparseMatrixCSC{Float64, Int}, colors::Vector{Int}, guess::Vector{Float64},
                     linear::Bool = false; nominal::Vector{Float64} = ones(length(guess)))::BlockSolver
  local n::Int = length(guess)
  length(nominal) == n || throw(DimensionMismatch("$(length(nominal)) nominal values for $(n) unknowns"))
  local groups::Vector{Vector{Int}} = Vector{Int}[Int[] for _ in 1:maximum(colors; init = 0)]
  foreach(j::Int -> push!(groups[colors[j]], j), 1:n)
  local dense::Bool = n <= DENSE_BLOCK_SIZE
  return BlockSolver(groups, copy(pattern), dense, linear, zeros(dense ? n : 0, dense ? n : 0), zeros(_LINALG.BlasInt, dense ? n : 0),
                     nothing, false, false, copy(guess), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
                     Float64[], copy(nominal), false, -1.0, 0, 0, 0)
end

"A solve without parameters (the solver's own tests): its residuals read no discrete state (`_holdDiscretes!`)."
_holdDiscretes!(::BlockSolver, ::Nothing)::Nothing = nothing

"A solve without parameters: kept factors are for its matrix (`_sameDiscretes`)."
_sameDiscretes(::BlockSolver, ::Nothing)::Bool = true

"""
    BlockResidual(blocks!, k)

The residual function of a model's `k`-th large algebraic loop: `blocks!(k, res, y, du, a, u, p, t, z, mode, fire)`,
the model's one function that calls its loops' residual functions (`BLOCKS!`, a branch per loop). `blockSolve!` and
`_passBlock!` are specialized on the residual's type: one type per model, not one per loop (MSL EngineV6, 13 loops:
0.225 s of compile each).
"""
struct BlockResidual{F <: Function} <: Function
  blocks!::F
  k::Int
end

@inline (r::BlockResidual)(res::AbstractVector, y::AbstractVector, du::AbstractVector, a::AbstractVector, u::AbstractVector, p,
                           t::Real, z::CrossingOut, mode::Bool, fire::Bool)::Nothing =
  r.blocks!(r.k, res, y, du, a, u, p, t, z, mode, fire)

"""
    blockSolve!(s, residual!, du, a, u, p, t, z, mode, fire)

Solve a large algebraic loop of an array model where the equations reach it: `residual!(res, y, du, a, u, p, t, z,
mode, fire)` places the iteration variables `y` where the equations read them (du, a), computes the loop's other
unknowns from them and evaluates its residuals. Newton from the last solution with the kept factorization, else with
a Jacobian in every step (also from the start values), then with differences over sqrt(eps) of each unknown's own
size, over a step of the unknowns' size, with shortened steps, and for a loop never solved from its start values
varied; the relations keep their values during the solve, then the residuals are evaluated once more at the solution
with the caller's `z` and `mode` (crossing functions, relation values). Where no solve converges, the unknowns are
NaN (the solver rejects the step; an initialization fails). An iterate whose last Jacobian was singular is no
solution (`rankDeficient`: its steps were least-norm steps, small where the residuals are not), but where all else
fails one whose residuals are zero to round-off (`_consistentIterate`: a loop singular in its mode and consistent).
A linear loop whose Jacobian costs at most `LINEAR_DIRECT_COLORS` evaluations is solved directly first
(`_blockLinear!`).
"""
function blockSolve!(s::BlockSolver, residual!::R, du::AbstractVector, a::AbstractVector, u::AbstractVector, p, t::Real,
                     z::CrossingOut, mode::Bool, fire::Bool)::Nothing where {R <: Function}
  eltype(a) === Float64 ||
    error("array model: an algebraic loop solved in the equations needs Float64 values (a solver with finite differences)")
  #= of concrete type: an abstract `Tuple` boxed it (208 B per call) and made each residual call a dynamic one =#
  local args::Tuple{typeof(du), typeof(a), typeof(u), typeof(p), typeof(t), CrossingOut, Bool, Bool} =
    (du, a, u, p, t, NO_CROSSINGS, false, fire)
  s.x .= s.y
  local ok::Bool = s.linear && length(s.groups) <= LINEAR_DIRECT_COLORS && _blockAttempt!(s, residual!, args, :linear)
  if !(ok && !s.rankDeficient)
    s.x .= s.y
    ok = _blockAttempt!(s, residual!, args, s.solved ? :kept : :full)
  end
  if !(ok && !s.rankDeficient) && s.solved
    s.x .= s.y
    ok = _blockAttempt!(s, residual!, args, :full)
  end
  if !(ok && !s.rankDeficient)
    #= unknowns far below 1 whose standard differences, over at least sqrt(eps), cross a kink of the residuals
       (`_differenceStep`; MSL DifferentialAmplifier's op-amp limiter): differences over their own size =#
    s.x .= s.y
    ok = _blockAttempt!(s, residual!, args, :fine)
  end
  if !(ok && !s.rankDeficient)
    #= a loop linear in its unknowns and badly conditioned (ideal switches: both thyristors of a commutating pair
       conducting for a pass of the event iteration, 1e-5 ohm against 1e5): the forward differences' round-off,
       1e-8 of the Jacobian, times the condition leaves Newton's method no contraction. Differences over a step of
       the unknowns' size are exact for a linear loop (`coarse`) =#
    s.x .= s.y
    ok = _blockAttempt!(s, residual!, args, :coarse)
  end
  if !(ok && !s.rankDeficient)
    #= a loop whose full Newton steps overshoot (a pipe's flow, the root of the pressure difference, at zero flow:
       each step lands as far on the other side): steps shortened until the residuals fall (`_blockDamped!`) =#
    s.x .= s.y
    ok = _blockAttempt!(s, residual!, args, :damped)
  end
  if !(ok && !s.rankDeficient) && !s.solved
    #= start values at which the residuals are stationary in an unknown (a MOSFET's current at a drain-source
       voltage of zero is even in it: the Jacobian's column is zero and no step leaves the point; MSL Spice3 Nand,
       its output node at 0 V): from the start values varied by 0.1 of their size, as OpenModelica's "vary the
       initial values" (nonlinearSolverHybrd.c: x += 0.1 nominal), with shortened steps =#
    s.x .= s.y .+ 0.1 .* max.(s.nominal, abs.(s.y))
    ok = _blockAttempt!(s, residual!, args, :damped)
  end
  if !(ok && !s.rankDeficient)
    #= a loop singular in its present mode but consistent (MSL OpenTank's ports above the level, both closed:
       `m_flow = 0` from either side, the pressures free up to the pipe's equation): an iterate of least-norm steps
       whose residuals are zero to round-off, as OpenModelica's solver accepts it (MSL TanksWithOverflow's overflow
       pipe at the start); one whose residuals stay (rows without a derivative) is none =#
    for kind in (:full, :damped)
      s.x .= s.y
      ok = _blockAttempt!(s, residual!, args, kind) && (!s.rankDeficient || _consistentIterate(s))
      ok && break
    end
    ok && (s.rankDeficient = false)
  end
  ok = ok && !s.rankDeficient
  if ok
    s.y .= s.x
    s.solved = true
    s.rsolved = maximum(abs, s.r; init = 0.0)
  else
    s.failures += 1
    fill!(s.x, NaN)
  end
  #= the last evaluation was at the solution unless crossing functions or relation values are asked for =#
  if ok
    (_noCrossings(z) && !mode) || residual!(s.r, s.x, du, a, u, p, t, z, mode, fire)
  else
    #= of a loop that did not solve at its NaN unknowns, for what the equations after it read; a function's assert
       there (`x0 < x1` of NaN: MSL Fluid's regFun3 in a pipe's friction) is no other failure than the loop's =#
    try
      residual!(s.r, s.x, du, a, u, p, t, z, mode, fire)
    catch err
      err isa _OMBACKEND.CodeGeneration._RHS_STEP_ERRORS || rethrow()
      fill!(s.r, NaN)
    end
  end
  return nothing
end

"""
    _consistentIterate(s) -> Bool

Whether the residuals `s.r` at the iterate `s.x` of the block's Jacobian `s.J` are zero to round-off
(`_consistentResiduals`).
"""
_consistentIterate(s::BlockSolver)::Bool = _consistentResiduals(s.r, s.J, s.x, s.nominal)

"""
    _consistentResiduals(r, J, x, nominal) -> Bool

Whether the residuals `r` at the iterate `x` are zero to round-off: each within 1e-8 of what the row's terms move by
over the unknowns' size (its Jacobian row's entries times the unknowns, at least their nominal values); a row without
a derivative has to be zero. The test of an iterate of least-norm steps (a singular Jacobian), whose step goes to zero
where the residuals are only stationary: `blockSolve!`'s (`_consistentIterate`) and `loopFamilySolve!`'s.
"""
function _consistentResiduals(r::Vector{Float64}, J::_SPARSE.SparseMatrixCSC{Float64, Int}, x::Vector{Float64},
                              nominal::AbstractVector{Float64})::Bool
  local scale::Vector{Float64} = zeros(length(r))
  for j in axes(J, 2), k in _SPARSE.nzrange(J, j)
    scale[J.rowval[k]] += abs(J.nzval[k]) * max(nominal[j], abs(x[j]))
  end
  return all(i::Int -> isfinite(r[i]) && abs(r[i]) <= 1.0e-8 * scale[i], eachindex(r))
end

"`_consistentResiduals` of a dense Jacobian (a loop family's, `loopFamilySolve!`)."
function _consistentResiduals(r::Vector{Float64}, J::Matrix{Float64}, x::Vector{Float64},
                              nominal::AbstractVector{Float64})::Bool
  for i in eachindex(r)
    local scale::Float64 = 0.0
    for j in axes(J, 2)
      scale += abs(J[i, j]) * max(nominal[j], abs(x[j]))
    end
    (isfinite(r[i]) && abs(r[i]) <= 1.0e-8 * scale) || return false
  end
  return true
end

"""
    _blockAttempt!(s, residual!, args, kind) -> Bool

One attempt of `blockSolve!`: the direct solve of a linear loop (`kind` `:linear`, `_blockLinear!`), Newton with the
kept factorization (`:kept`), with a Jacobian in every step
(`:full`), that with differences over the unknowns' size (`:coarse`) or over sqrt(eps) of each one's own size
(`:fine`, `_differenceStep`), or with shortened steps (`:damped`,
`_blockDamped!`). An error of the residuals at an iterate or at a point the Jacobian's
differences perturb (a domain error: `cos` of an infinite angle, a root of a negative number; a function's assert:
the scalar path's `_RHS_STEP_ERRORS`) is a failed attempt, kept as the right-hand side's last error
(`StepRejectingRHS`): thrown on, it ended a run from the check of the asserts after a step, which no wrapper guards
(MSL MultiBody UniversalConstraint).
"""
function _blockAttempt!(s::BlockSolver, residual!::R, args::Tuple, kind::Symbol)::Bool where {R <: Function}
  try
    kind === :damped && return _blockDamped!(s, residual!, args)
    kind === :linear && return _blockLinear!(s, residual!, args)
    return _blockNewton!(s, residual!, args, kind !== :kept;
                         differences = kind in (:coarse, :fine) ? kind : :standard)
  catch err
    err isa _OMBACKEND.CodeGeneration._RHS_STEP_ERRORS || rethrow()
    _OMBACKEND.CodeGeneration.LAST_RHS_ERROR[] = err
    return false
  end
end

"""
    _blockNewton!(s, residual!, args, full) -> Bool

Newton on the block's residuals from `s.x` (in place), the last evaluation at the returned iterate. `full`: a Jacobian
in every step (from the start values, or where the kept one failed); otherwise the kept factorization while the steps
contract fast enough to converge within the iterations left (at least twofold; as many iterations as a Jacobian costs
evaluations, its colors, and ten: EngineV6's linear loop of 90 variables, 47 colors, refreshed its Jacobian in half the
calls with ten), else a Jacobian at the current iterate. Within 1e-10 of their values (every step component, relative,
absolute near zero) the steps go on while they contract at least twofold, to round-off (1e-15): a solve's last digits
then do not depend on its guess (a relation on a loop's unknown flipped back and forth between the passes of an event
iteration and the checks after the steps, its hysteresis 1e-12 at a tolerance of 1e-8, below the loop's 1e-10). The
guess is kept when its first step is at round-off already, or within 1e-8 with its residuals no larger than at the
solution last accepted (`rsolved`: a solve from its own solution returns it); the steps of a Jacobian of this call that stop
contracting end within 1e-5 (EngineV6's forces of 1e9 cancel to zero only that far).
"""
function _blockNewton!(s::BlockSolver, residual!::R, args::Tuple, full::Bool;
                       differences::Symbol = :standard)::Bool where {R <: Function}
  local x::Vector{Float64} = s.x; local r::Vector{Float64} = s.r; local dx::Vector{Float64} = s.dx
  residual!(r, x, args...)
  s.residuals += 1
  local fresh::Bool = false
  if full || !s.factored
    _blockJacobian!(s, residual!, args; differences) || return false
    fresh = true
  end
  local maxiters::Int = full ? 50 : 10 + length(s.groups)
  local previous::Float64 = Inf
  for iteration in 1:maxiters
    all(isfinite, r) || return false
    _blockStep!(s)
    local step::Float64 = _relativeStep(x, dx, s.nominal)
    #= EngineV6's piston speed at top dead center, zero to round-off, flipped a relation in every pass of the event
       iteration while each solve stepped again. A badly conditioned loop's step at its solution is round-off times
       the condition, beyond 1e-15 of the values: a guess whose residuals are no larger than those at the solution
       last accepted (`rsolved`, at the same guess the same ones) is kept too. MSL ThyristorBridge2mPulse_DC_Drive, a
       commutating pair of thyristors (1e-5 ohm against 1e5): residuals of 2e-15 to 6e-15 of their terms, steps of
       1e-10 back and forth, a thyristor's `s < 0` at s = +-1.1e-10 flipped in every pass. Only where the step is
       within 1e-8 of the values (`step <= 100`): DC_Drive's are up to 10; a guess far from the solution whose
       residuals are small for the loop's scale was kept as well (MSL ThyristorBridge2mPulse_R: steps up to 7e14, the
       mean values off from the first period) =#
    if iteration == 1 && (step <= 1.0e-5 || (step <= 100.0 && maximum(abs, r; init = 0.0) <= s.rsolved))
      #= a Jacobian of this call evaluated last at its perturbed points (the equations' du and a there): back at the
         iterate =#
      if fresh
        residual!(r, x, args...)
        s.residuals += 1
      end
      return true
    end
    x .-= dx
    all(isfinite, x) || return false
    residual!(r, x, args...)
    s.residuals += 1
    local rate::Float64 = step / previous
    previous = step
    (step <= 1.0e-5 || (step <= 1.0 && rate > 0.5)) && return all(isfinite, r)
    step <= 1.0 && continue
    (rate >= 1.0 && fresh && step <= 1.0e5) && return all(isfinite, r)
    if full
      _blockJacobian!(s, residual!, args; differences) || return false
    elseif rate > 0.5 || step * rate^(maxiters - iteration) > 1.0
      #= diverging with a Jacobian of this call: the full Newton's; slow with one: on with it =#
      (fresh && rate >= 1.0) && return false
      if !fresh
        _blockJacobian!(s, residual!, args; differences) || return false
        fresh = true
        previous = Inf
      end
    end
  end
  #= the iterations used up while polishing: within 1e-10 =#
  return previous <= 1.0 && all(isfinite, r)
end

"""
    _blockLinear!(s, residual!, args) -> Bool

The solve of a loop whose residuals are affine in its unknowns (`s.linear`) from `s.x`. With kept factors made at the
present discrete state (`_sameDiscretes`) first one step with them: where the step after it is within 1e-10 of the
values the matrix has not moved (a circuit of constant resistances between switchings: two evaluations). A switching
changes the matrix, and that second step, of the old factors, measures the old one: MSL SMPM_Braking's diode bridge
at the start, its currents zero to round-off, took the kept factors' solution under each new set of relations, whose
round-off signs set the relations again (six relations alternating between two sets). Else the Jacobian at the iterate by differences over the unknowns'
size (exact for affine residuals, `:coarse`), one step with its factors, and steps of iterative refinement with them
while they are beyond 1e-10 of the values (at most two). False for a singular Jacobian, residuals not finite or
refinement that does not get within 1e-10 (the general attempts follow); the last evaluation is at the returned
iterate. A matrix that moves with the states (MSL EngineV6's loops of two unknowns) costs the Jacobian's colors and
three evaluations, where Newton with a kept Jacobian needed eight to nine.
"""
function _blockLinear!(s::BlockSolver, residual!::R, args::Tuple)::Bool where {R <: Function}
  local x::Vector{Float64} = s.x; local r::Vector{Float64} = s.r; local dx::Vector{Float64} = s.dx
  residual!(r, x, args...)
  s.residuals += 1
  all(isfinite, r) || return false
  if s.factored && !s.rankDeficient && _sameDiscretes(s, args[4])
    _blockStep!(s)
    x .-= dx
    all(isfinite, x) || return false
    residual!(r, x, args...)
    s.residuals += 1
    all(isfinite, r) || return false
    _blockStep!(s)
    _relativeStep(x, dx, s.nominal) <= 1.0 && return true
  end
  (_blockJacobian!(s, residual!, args; differences = :coarse) && !s.rankDeficient) || return false
  for iteration in 1:3
    _blockStep!(s)
    #= a refinement step within 1e-10: the solution as the last step left it =#
    iteration > 1 && _relativeStep(x, dx, s.nominal) <= 1.0 && return true
    x .-= dx
    all(isfinite, x) || return false
    residual!(r, x, args...)
    s.residuals += 1
    all(isfinite, r) || return false
  end
  #= refinement that does not converge (a badly conditioned loop): the general attempts =#
  return false
end

"""
The largest component of the step `dx` at `x` in units of 1e-10 of the values (relative; absolute near zero, in units
of the unknowns' nominal values).
"""
function _relativeStep(x::Vector{Float64}, dx::Vector{Float64}, nominal::Vector{Float64})::Float64
  local step::Float64 = 0.0
  for i in eachindex(x)
    step = max(step, abs(dx[i]) / (1.0e-10 * (nominal[i] + abs(x[i]))))
  end
  return step
end

"""
    _blockDamped!(s, residual!, args) -> Bool

Newton on the block's residuals from `s.x` with a Jacobian in every step and each step shortened (halved, at most
twelve times) until the sum of the squared residuals falls: the Newton direction is one of descent for it. Converged
where the full step is within 1e-10 of the values (as `_blockNewton!`); false where no shortened step lowers the
residuals, a Jacobian fails, or after 100 steps. The last evaluation is at the returned iterate.
"""
function _blockDamped!(s::BlockSolver, residual!::R, args::Tuple)::Bool where {R <: Function}
  local x::Vector{Float64} = s.x; local r::Vector{Float64} = s.r; local dx::Vector{Float64} = s.dx
  local x0::Vector{Float64} = s.xp
  all(isfinite, x) || return false
  residual!(r, x, args...)
  s.residuals += 1
  for _ in 1:100
    all(isfinite, r) || return false
    _blockJacobian!(s, residual!, args) || return false
    _blockStep!(s)
    all(isfinite, dx) || return false
    local step::Float64 = _relativeStep(x, dx, s.nominal)
    if step <= 1.0
      x .-= dx
      residual!(r, x, args...)
      s.residuals += 1
      return all(isfinite, r)
    end
    local before::Float64 = sum(abs2, r)
    local scale::Float64 = 1.0
    local lower::Bool = false
    x0 .= x
    for _ in 1:13
      x .= x0 .- scale .* dx
      residual!(r, x, args...)
      s.residuals += 1
      if all(isfinite, r) && sum(abs2, r) < (1.0 - 1.0e-4 * scale) * before
        lower = true
        break
      end
      scale /= 2.0
    end
    lower || return false
  end
  return false
end

"The Newton step `s.dx` for the residuals `s.r` with the kept factorization (the least-norm one with a pseudo-inverse)."
function _blockStep!(s::BlockSolver)::Nothing
  if s.rankDeficient
    _LINALG.mul!(s.dx, s.factors, s.r)
  elseif s.dense
    s.dx .= s.r
    _LINALG.LAPACK.getrs!('N', s.factors, s.pivots, s.dx)
  else
    _LINALG.ldiv!(s.dx, s.F::_UMFPACK_LU, s.r)
  end
  return nothing
end

"""
    _differenceStep(x, differences, nominal) -> Float64

The step of a forward difference in an unknown of value `x`: `:standard` sqrt(eps) of its size, at least its nominal
value `nominal` (1 without the attribute); `:coarse` its size, at least `nominal`; `:fine` sqrt(eps) of its own size
(of `nominal` at zero), for an unknown far below 1 whose standard step crosses a kink of the residuals (MSL
DifferentialAmplifier: a current of 1.6 mA reaches its op-amp's limiter with a gain of 1e9, the standard step of
1.5e-8 A moves its input by 15 V, beyond the 9.7 V to the supply).
"""
function _differenceStep(x::Float64, differences::Symbol, nominal::Float64)::Float64
  differences === :fine && return sqrt(eps(Float64)) * (iszero(x) ? nominal : abs(x))
  return (differences === :coarse ? 1.0 : sqrt(eps(Float64))) * max(nominal, abs(x))
end

"""
    _blockJacobian!(s, residual!, args; differences) -> Bool

The block's Jacobian at `s.x` (residuals `s.r` there) by forward differences (steps `_differenceStep`), one evaluation
per color, factored
(dense in place; sparse with the kept factorization's symbolic analysis); false for one not finite or a singular
sparse one (no factorization kept: the next solve starts with a Jacobian). A singular dense one gives its
pseudo-inverse for least-norm steps (`rankDeficient`; singular values below `RANK_TOLERANCE` of the largest count as
zero, the forward differences' accuracy), not kept either: products of unknowns that start at zero have no
derivative there (MSL QuasiStatic SinglePhase Rectifier's power balance `v * conj(i)`: two zero rows at the start
values, the loop never solved and its 83 variables NaN through a run that ended in Success), and the least-norm step
of the other rows leaves that point.
"""
function _blockJacobian!(s::BlockSolver, residual!::R, args::Tuple;
                         differences::Symbol = :standard)::Bool where {R <: Function}
  local x::Vector{Float64} = s.x; local xp::Vector{Float64} = s.xp; local h::Vector{Float64} = s.h
  local J::_SPARSE.SparseMatrixCSC{Float64, Int} = s.J
  for cols in s.groups
    xp .= x
    for j in cols
      h[j] = _differenceStep(x[j], differences, s.nominal[j])
      xp[j] += h[j]
    end
    residual!(s.rp, xp, args...)
    s.residuals += 1
    for j in cols, k in _SPARSE.nzrange(J, j)
      J.nzval[k] = (s.rp[J.rowval[k]] - s.r[J.rowval[k]]) / h[j]
    end
  end
  s.jacobians += 1
  s.factored = false
  s.rankDeficient = false
  s.linear && _holdDiscretes!(s, args[4])
  all(isfinite, J.nzval) || return false
  if s.dense
    fill!(s.factors, 0.0)
    for j in 1:size(J, 2), k in _SPARSE.nzrange(J, j)
      s.factors[J.rowval[k], j] = J.nzval[k]
    end
    s.factored = _LINALG.LAPACK.getrf!(s.factors, s.pivots; check = false)[3] == 0
    if !s.factored
      s.factors .= _LINALG.pinv(Matrix(J); rtol = RANK_TOLERANCE)
      s.rankDeficient = true
      return true
    end
  else
    local F::_UMFPACK_LU = s.F === nothing ? _LINALG.lu(J; check = false) : _LINALG.lu!(s.F::_UMFPACK_LU, J; check = false)
    s.factored = _LINALG.issuccess(F)
    s.F = s.factored ? F : nothing
  end
  return s.factored
end
