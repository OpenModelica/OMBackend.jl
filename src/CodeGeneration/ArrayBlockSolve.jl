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
    foreach(r::Int -> (used[c][r] = true), rows)
    colors[j] = c
  end
  return colors
end

"""
    BlockSolver(pattern, colors, guess)

The solver of one large algebraic loop of an array model (`blockSolve!`): the columns of each color of its residuals'
sparsity in its iteration variables, the Jacobian on that sparsity and its factorization, kept between calls (dense
up to `DENSE_BLOCK_SIZE` variables: the factors in place with their pivots; else sparse), the last solution (the next
call's guess; the start values until a solve succeeded) and counters (residual evaluations, Jacobians, failed solves).
"""
mutable struct BlockSolver
  const groups::Vector{Vector{Int}}
  const J::_SPARSE.SparseMatrixCSC{Float64, Int}
  const dense::Bool
  const factors::Matrix{Float64}
  const pivots::Vector{_LINALG.BlasInt}
  F::Union{Nothing, _UMFPACK_LU}
  factored::Bool
  const y::Vector{Float64}
  const x::Vector{Float64}
  const r::Vector{Float64}
  const rp::Vector{Float64}
  const xp::Vector{Float64}
  const dx::Vector{Float64}
  const h::Vector{Float64}
  solved::Bool
  residuals::Int
  jacobians::Int
  failures::Int
end

function BlockSolver(pattern::_SPARSE.SparseMatrixCSC{Float64, Int}, colors::Vector{Int}, guess::Vector{Float64})::BlockSolver
  local n::Int = length(guess)
  local groups::Vector{Vector{Int}} = Vector{Int}[Int[] for _ in 1:maximum(colors; init = 0)]
  foreach(j::Int -> push!(groups[colors[j]], j), 1:n)
  local dense::Bool = n <= DENSE_BLOCK_SIZE
  return BlockSolver(groups, copy(pattern), dense, zeros(dense ? n : 0, dense ? n : 0), zeros(_LINALG.BlasInt, dense ? n : 0),
                     nothing, false, copy(guess), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), false, 0, 0, 0)
end

"""
    blockSolve!(s, residual!, du, a, u, p, t, z, mode, fire)

Solve a large algebraic loop of an array model where the equations reach it: `residual!(res, y, du, a, u, p, t, z,
mode, fire)` places the iteration variables `y` where the equations read them (du, a), computes the loop's other
unknowns from them and evaluates its residuals. Newton from the last solution with the kept factorization, else with
a Jacobian in every step (also from the start values); the relations keep their values during the solve, then the
residuals are evaluated once more at the solution with the caller's `z` and `mode` (crossing functions, relation
values). Where no solve converges, the unknowns are NaN (the solver rejects the step; an initialization fails).
"""
function blockSolve!(s::BlockSolver, residual!::R, du::AbstractVector, a::AbstractVector, u::AbstractVector, p, t::Real,
                     z, mode::Bool, fire::Bool)::Nothing where {R <: Function}
  eltype(a) === Float64 ||
    error("array model: an algebraic loop solved in the equations needs Float64 values (a solver with finite differences)")
  local args::Tuple = (du, a, u, p, t, nothing, false, fire)
  s.x .= s.y
  local ok::Bool = _blockNewton!(s, residual!, args, !s.solved)
  if !ok && s.solved
    s.x .= s.y
    ok = _blockNewton!(s, residual!, args, true)
  end
  if ok
    s.y .= s.x
    s.solved = true
  else
    s.failures += 1
    fill!(s.x, NaN)
  end
  #= the last evaluation was at the solution unless crossing functions or relation values are asked for =#
  (ok && z === nothing && !mode) || residual!(s.r, s.x, du, a, u, p, t, z, mode, fire)
  return nothing
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
guess is kept when its first step is at round-off already; the steps of a Jacobian of this call that stop contracting
end within 1e-5 (EngineV6's forces of 1e9 cancel to zero only that far).
"""
function _blockNewton!(s::BlockSolver, residual!::R, args::Tuple, full::Bool)::Bool where {R <: Function}
  local x::Vector{Float64} = s.x; local r::Vector{Float64} = s.r; local dx::Vector{Float64} = s.dx
  residual!(r, x, args...)
  s.residuals += 1
  local fresh::Bool = false
  if full || !s.factored
    _blockJacobian!(s, residual!, args) || return false
    fresh = true
  end
  local maxiters::Int = full ? 50 : 10 + length(s.groups)
  local previous::Float64 = Inf
  for iteration in 1:maxiters
    all(isfinite, r) || return false
    _blockStep!(s)
    local step::Float64 = 0.0
    for i in eachindex(x)
      step = max(step, abs(dx[i]) / (1.0e-10 * (1.0 + abs(x[i]))))
    end
    #= EngineV6's piston speed at top dead center, zero to round-off, flipped a relation in every pass of the event
       iteration while each solve stepped again =#
    if iteration == 1 && step <= 1.0e-5
      #= a Jacobian of this call evaluated last at its perturbed points (the equations' du and a there): back at the
         iterate =#
      if fresh
        residual!(r, x, args...)
        s.residuals += 1
      end
      return true
    end
    x .-= dx
    residual!(r, x, args...)
    s.residuals += 1
    local rate::Float64 = step / previous
    previous = step
    (step <= 1.0e-5 || (step <= 1.0 && rate > 0.5)) && return all(isfinite, r)
    step <= 1.0 && continue
    (rate >= 1.0 && fresh && step <= 1.0e5) && return all(isfinite, r)
    if full
      _blockJacobian!(s, residual!, args) || return false
    elseif rate > 0.5 || step * rate^(maxiters - iteration) > 1.0
      #= diverging with a Jacobian of this call: the full Newton's; slow with one: on with it =#
      (fresh && rate >= 1.0) && return false
      if !fresh
        _blockJacobian!(s, residual!, args) || return false
        fresh = true
        previous = Inf
      end
    end
  end
  #= the iterations used up while polishing: within 1e-10 =#
  return previous <= 1.0 && all(isfinite, r)
end

"The Newton step `s.dx` for the residuals `s.r` with the kept factorization."
function _blockStep!(s::BlockSolver)::Nothing
  if s.dense
    s.dx .= s.r
    _LINALG.LAPACK.getrs!('N', s.factors, s.pivots, s.dx)
  else
    _LINALG.ldiv!(s.dx, s.F::_UMFPACK_LU, s.r)
  end
  return nothing
end

"""
    _blockJacobian!(s, residual!, args) -> Bool

The block's Jacobian at `s.x` (residuals `s.r` there) by forward differences, one evaluation per color, factored
(dense in place; sparse with the kept factorization's symbolic analysis); false for a singular one or one not finite
(no factorization kept: the next solve starts with a Jacobian).
"""
function _blockJacobian!(s::BlockSolver, residual!::R, args::Tuple)::Bool where {R <: Function}
  local x::Vector{Float64} = s.x; local xp::Vector{Float64} = s.xp; local h::Vector{Float64} = s.h
  local J::_SPARSE.SparseMatrixCSC{Float64, Int} = s.J
  for cols in s.groups
    xp .= x
    for j in cols
      h[j] = sqrt(eps(Float64)) * max(1.0, abs(x[j]))
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
  all(isfinite, J.nzval) || return false
  if s.dense
    fill!(s.factors, 0.0)
    for j in 1:size(J, 2), k in _SPARSE.nzrange(J, j)
      s.factors[J.rowval[k], j] = J.nzval[k]
    end
    s.factored = _LINALG.LAPACK.getrf!(s.factors, s.pivots; check = false)[3] == 0
  else
    local F::_UMFPACK_LU = s.F === nothing ? _LINALG.lu(J; check = false) : _LINALG.lu!(s.F::_UMFPACK_LU, J; check = false)
    s.factored = _LINALG.issuccess(F)
    s.F = s.factored ? F : nothing
  end
  return s.factored
end
