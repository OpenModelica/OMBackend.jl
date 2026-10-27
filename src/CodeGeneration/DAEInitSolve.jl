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
  Two-phase Newton solver for DAE consistent-IC.
  Shared by DirectRHS and MTK pre-solve paths.
  Phase 1 fixes differential states and solves algebraic residuals; phase 2
  frees all unknowns to handle rank-deficient kinematic loops.
=#

"""
Residual vector of one init iterate: dynamics rows (du minus targets) plus
the optional extra rows from signal-valued initialization equations.
"""
function _initResidualVec(du, u, eq_idx, targets, extraRes)
  local base = du[eq_idx] .- targets
  extraRes === nothing && return base
  return vcat(base, extraRes(du, u))
end

"""
Forward-difference Jacobian of the init residual at u0 over the var_idx
columns. `res` is the residual already evaluated at u0.
"""
function _fdInitResidualJacobian(rhsFunc, p_vec, u0, eq_idx, targets, extraRes, var_idx, res;
                                 eps_fd=1e-7)
  local J = zeros(length(res), length(var_idx))
  local du_pert = similar(u0)
  for (jcol, jstate) in enumerate(var_idx)
    local u_pert = copy(u0)
    #= Relative to the value: 1e-7 vanished in 8.1e9 (Buildings' PowerLinearized, T4 = T^4),
       a zero column, and Newton never moved. =#
    u_pert[jstate] += eps_fd * max(1.0, abs(u0[jstate]))
    local h = u_pert[jstate] - u0[jstate]
    rhsFunc(du_pert, u_pert, p_vec, 0.0)
    J[:, jcol] = (_initResidualVec(du_pert, u_pert, eq_idx, targets, extraRes) .- res) ./ h
  end
  return J
end

"""
Underdetermined-init completion: a converged root whose Jacobian has a
nontrivial null space means the equations leave spare degrees of freedom.
Tool convention is to complete such a system by fixing selected variables
at their start values. Pin the algebraic unknowns with the largest
null-space projections at their entry values and re-solve the remainder
from the root; keep the completed root only when the constrained solve
converges.
"""
function _completeUnderdeterminedInit!(u0, rhsFunc, p_vec, eq_idx, var_idx, algCandidates, u0_entry;
                                       targets=zeros(Float64, length(eq_idx)), tol=1e-10,
                                       extraRes=nothing, maxiter=200, restoreIdx=Int[])
  OMBackend.envSwitch("OMBACKEND_INIT_COMPLETE") || return false
  isempty(var_idx) && return false
  #= Only algebraic unknowns are completion candidates; with none, the
     Jacobian and SVD below cannot produce a pick. =#
  isempty(algCandidates) && return false
  local traceInit = OMBackend.envSwitch("OMBACKEND_INIT_TRACE")
  local du = similar(u0)
  local freeVars = collect(var_idx)
  local algSet = OrderedSet(algCandidates)
  local changed = false
  #= Pins a prior relaxed phase may have moved are restored together with the
     completion picks: from a near-root start the constrained solve converges
     where the same restore from a far point did not. Drop the restore on
     failure rather than the whole completion. =#
  local pendingRestore = collect(restoreIdx)
  #= Each accepted round shrinks the free set; the cap bounds the cost when a
     round keeps exposing a smaller residual null space. =#
  local maxRounds = 6
  for round in 1:maxRounds
    rhsFunc(du, u0, p_vec, 0.0)
    local res = _initResidualVec(du, u0, eq_idx, targets, extraRes)
    all(isfinite, res) || return changed
    local nRows = length(res)
    local J = _fdInitResidualJacobian(rhsFunc, p_vec, u0, eq_idx, targets, extraRes, freeVars, res)
    all(isfinite, J) || return changed
    local rowNorm = [max(maximum(abs, @view J[i, :]), 1.0) for i in 1:nRows]
    local F = LinearAlgebra.svd(J ./ rowNorm; full=true)
    local smax = isempty(F.S) ? 0.0 : F.S[1]
    local rank = count(s -> s > 1e-8 * smax, F.S)
    local nullDim = length(freeVars) - rank
    nullDim <= 0 && return changed
    local score = Dict{Int, Float64}()
    for (jcol, jstate) in enumerate(freeVars)
      jstate in algSet || continue
      local s2 = 0.0
      for d in 1:nullDim
        s2 += F.V[jcol, end - d + 1]^2
      end
      s2 > 1e-6 && (score[jstate] = sqrt(s2))
    end
    isempty(score) && return changed
    local picks = sort!(collect(keys(score)); by = j -> -score[j])
    picks = picks[1:min(nullDim, length(picks))]
    traceInit && println("[initcomplete] round ", round, " null dim ", nullDim,
                         ", pinning ", length(picks), " algebraic var(s) at entry values")
    local pickSet = OrderedSet(picks)
    local newFree = [j for j in freeVars if !(j in pickSet)]
    local solved = false
    local u_try = similar(u0)
    #= First attempt re-imposes the pending restore; only if that fails to
       converge is the round retried without it. =#
    for withRestore in (true, false)
      !withRestore && isempty(pendingRestore) && continue
      copyto!(u_try, u0)
      if withRestore
        for j in pendingRestore
          u_try[j] = u0_entry[j]
        end
      end
      for j in picks
        u_try[j] = u0_entry[j]
      end
      if _solveDAEPhase!(u_try, rhsFunc, p_vec, eq_idx, newFree;
                         targets=targets, maxiter=maxiter, tol=tol, extraRes=extraRes,
                         phaseLabel=string("complete-r", round, withRestore ? "" : "-norestore"))
        copyto!(u0, u_try)
        changed = true
        freeVars = newFree
        withRestore && (pendingRestore = Int[])
        solved = true
        break
      end
    end
    if !solved
      traceInit && println("[initcomplete] constrained re-solve failed; keeping prior root")
      return changed
    end
  end
  return changed
end

"""
A free unknown guessed at exactly 0 can make a residual row non-finite
at the entry (MSL QS FluxTubes GeneralLeakage: `0 = -7e-6 + 0.3/G_m`, G_m
without a start value), and then no phase can start. Those guesses become
1: all together, else one at a time where it lowers the number of
non-finite rows. Only at such an entry, which no phase could solve from.
Whether a guess changed.
"""
function _nudgeZeroGuesses!(u0, rhsFunc, p_vec, eq_idx, eq_target, extra_residuals, fixed)
  local du = similar(u0)
  local nonFinite = u -> (rhsFunc(du, u, p_vec, 0.0);
                          count(!isfinite, _initResidualVec(du, u, eq_idx, eq_target, extra_residuals)))
  local candidates = [i for i in eachindex(u0) if iszero(u0[i]) && !(i in fixed)]
  isempty(candidates) && return false
  local bad = nonFinite(u0)
  local u = copy(u0)
  u[candidates] .= 1.0
  if nonFinite(u) == 0
    copyto!(u0, u)
    return true
  end
  local changed = false
  for i in candidates
    u0[i] = 1.0
    local b = nonFinite(u0)
    if b < bad
      bad = b
      changed = true
      bad == 0 && break
    else
      u0[i] = 0.0
    end
  end
  return changed
end

"""
    _solveCausalRows!(u0, rhsFunc, p_vec, eq_idx, eq_target, unknowns, sparsity; tol)

The initialization's one-unknown blocks of a square system (as many rows as `unknowns`), before
the simultaneous solve (OpenModelica's causal order): a row that reads one unknown not yet
solved (by the Jacobian's structure `sparsity`) and is affine in it (probed at three values) is
solved for it, the unknowns so solved known for the rows after, until no row is left of that
kind. Such a row is that unknown's in every matching, so its value holds at every solution with
the others' values, and the solve starts from it: from the media's start values (MSL Fluid's d =
150, p = 5e6 against 1000 and 1.1e5, the tank's mass 0) MeasuringTemperature's tanks were a
simultaneous Newton of 78 rows that converged or not on rounding. A value is kept only where the
row's residual at it is zero to the tolerance; a probe that fails (outside a medium's range)
leaves the row. Returns how many unknowns were solved.
"""
function _solveCausalRows!(u0::Vector{Float64}, rhsFunc::F, p_vec::Vector{Float64}, eq_idx::Vector{Int},
                           eq_target::Vector{Float64}, unknowns::Vector{Int},
                           sparsity::Symbolics.SparseArrays.SparseMatrixCSC{Float64, Int}; tol::Float64)::Int where {F}
  local SA = Symbolics.SparseArrays
  local rowsOf = permutedims(sparsity)   # column k: the columns row k reads
  local open = BitSet(unknowns)
  local du = similar(u0)
  local residualAt = (w::Int, j::Int, x::Float64) -> begin
    u0[j] = x
    rhsFunc(du, u0, p_vec, 0.0)
    du[eq_idx[w]] - eq_target[w]
  end
  local done = falses(length(eq_idx))
  local solved = 0
  local progressed = true
  while progressed
    progressed = false
    for w in eachindex(eq_idx)
      done[w] && continue
      local k = eq_idx[w]
      k <= size(rowsOf, 2) || continue
      local reads = Int[c for c in SA.rowvals(rowsOf)[SA.nzrange(rowsOf, k)] if c in open]
      length(reads) == 1 || continue
      local j = only(reads)
      local x0 = u0[j]
      local h = 0.5 * max(1.0, abs(x0))
      local accepted = try
        local (f0, f1, f2) = (residualAt(w, j, x0), residualAt(w, j, x0 + h), residualAt(w, j, x0 + 2h))
        local (s1, s2) = (f1 - f0, f2 - f1)
        local scale = max(1.0, abs(f0), abs(f1), abs(f2))
        local affine = all(isfinite, (f0, f1, f2)) && !iszero(s1) &&
                       abs(s1 - s2) <= 1.0e-8 * max(abs(s1), abs(s2)) + 1.0e-12 * scale
        local x = affine ? x0 - f0 * h / s1 : NaN
        isfinite(x) && abs(residualAt(w, j, x)) <= max(tol, 1.0e-10 * scale)
      catch e
        OMBackend._fallback(e, :solveCausalRows)
        false
      end
      if accepted
        delete!(open, j)
        done[w] = true
        solved += 1
        progressed = true
      else
        u0[j] = x0
      end
    end
  end
  return solved
end

"""
    ErasedRHS(f)

A right-hand side `f(du, u, p, t)` behind an abstract field: the initialization's phases are compiled once for every
model (their argument of one type), not again for each generated right-hand side (a RuntimeGeneratedFunction's type is
its code's hash). MSL CoupledClutches compiled the initialization's body 1.36 s of its 2.3 s translate. A call is one
dynamic dispatch (~0.1 µs; a right-hand side takes 10-140 µs).
"""
struct ErasedRHS <: Function
  f::Function
end

(r::ErasedRHS)(du::AbstractVector, u::AbstractVector, p, t::Real) = (r.f(du, u, p, t); nothing)

"""
    ErasedResiduals(f)

Extra initialization residuals `f(du, u) -> Vector` behind an abstract field, for the same reason as `ErasedRHS`.
"""
struct ErasedResiduals <: Function
  f::Function
end

(r::ErasedResiduals)(du::AbstractVector, u::AbstractVector) = r.f(du, u)

_erasedRHS(f::ErasedRHS)::ErasedRHS = f
_erasedRHS(f::Function)::ErasedRHS = ErasedRHS(f)
_erasedResiduals(::Nothing)::Nothing = nothing
_erasedResiduals(f::ErasedResiduals)::ErasedResiduals = f
_erasedResiduals(f::Function)::ErasedResiduals = ErasedResiduals(f)

"""
    _solveCausalBlocks!(u0, rhsFunc, p_vec, eq_idx, eq_target, extra_residuals, unknowns, sparsity; tol, maxiter) -> Int

The initialization's square system (as many rows as `unknowns`: the right-hand side's rows `eq_idx`,
then the initialization equations' `extra_residuals`) in causal order, as OpenModelica's
block-lower-triangular form: each row matched to an unknown (a maximum matching by the Jacobian's
structure `sparsity`; the initialization equations' rows by finite differences, `_initIncidence`), the
strongly connected blocks of the matched rows' dependencies solved one after
the other, in dependency order, each by Newton over its own unknowns (`_solveDAEPhase!`) with the
blocks before it known. A block that does not converge ends the pass: the simultaneous phases go on
from the blocks solved before it, whose values hold at every solution with theirs. Nothing where the
system is one block or has no perfect matching. MSL EngineV6's slider-cranks: a loop's positions are a
block of their own; solved together with the gas forces (polynomials of the piston position, far
outside their range at the start guesses), the phases stalled at residuals of 35 and the free
initialization moved the fixed load speed (test/initializationCausalRowsTests.jl). Returns how many unknowns were
solved.
"""
function _solveCausalBlocks!(u0::Vector{Float64}, rhsFunc::F, p_vec::Vector{Float64}, eq_idx::Vector{Int},
                             eq_target::Vector{Float64}, extra_residuals::Union{Nothing, ErasedResiduals},
                             unknowns::Vector{Int}, sparsity::Symbolics.SparseArrays.SparseMatrixCSC{Float64, Int};
                             tol::Float64, maxiter::Int)::Int where {F}
  local SA = Symbolics.SparseArrays
  local nEq::Int = length(eq_idx)
  local reads::Vector{Vector{Int}} = if extra_residuals === nothing
    local rowsOf::SA.SparseMatrixCSC{Float64, Int} = permutedims(sparsity)   # column k: the columns row k reads
    local position::Dict{Int, Int} = Dict{Int, Int}(j => k for (k, j) in enumerate(unknowns))
    Vector{Int}[k <= size(rowsOf, 2) ? Int[position[c] for c in SA.rowvals(rowsOf)[SA.nzrange(rowsOf, k)] if haskey(position, c)] :
                Int[] for k in eq_idx]
  else
    _initIncidence(rhsFunc, p_vec, u0, eq_idx, eq_target, extra_residuals, unknowns, sparsity)
  end
  length(reads) == length(unknowns) || return 0
  local assign::Vector{Int} = _maximumMatching(reads, length(unknowns))   # assign[c]: the row of unknown c
  all(!iszero, assign) || return 0
  local columnOf::Vector{Int} = zeros(Int, length(reads))
  foreach(c::Int -> (columnOf[assign[c]] = c), eachindex(assign))
  #= an edge from the row that determines an unknown to each row that reads it =#
  local g::Graphs.SimpleDiGraph{Int} = Graphs.SimpleDiGraph(length(reads))
  for (w, cs) in enumerate(reads), c in cs
    assign[c] == w || Graphs.add_edge!(g, assign[c], w)
  end
  local blocks::Vector{Vector{Int}} = Graphs.strongly_connected_components(g)
  length(blocks) > 1 || return 0
  local order::Vector{Int} = Graphs.topological_sort_by_dfs(Graphs.condensation(g, blocks))
  local traceInit::Bool = OMBackend.envSwitch("OMBACKEND_INIT_TRACE")
  local solved::Int = 0
  for b in order
    local rows::Vector{Int} = sort(blocks[b])
    local vars::Vector{Int} = Int[unknowns[columnOf[w]] for w in rows]
    local eqRows::Vector{Int} = Int[w for w in rows if w <= nEq]
    #= the block's own initialization-equation rows =#
    local extraRows::Vector{Int} = Int[w - nEq for w in rows if w > nEq]
    local blockExtra::Union{Nothing, ErasedResiduals} = isempty(extraRows) ? nothing :
      ErasedResiduals((du::AbstractVector, u::AbstractVector) -> extra_residuals(du, u)[extraRows])
    local trial::Vector{Float64} = copy(u0)
    local converged::Bool = _solveDAEPhase!(trial, rhsFunc, p_vec, eq_idx[eqRows], vars; targets = eq_target[eqRows],
                                            maxiter = maxiter, tol = tol, extraRes = blockExtra, phaseLabel = "block")
    if !converged
      traceInit && println("[initblocks] block of ", length(rows), " rows did not converge; ", solved, " of ",
                           length(unknowns), " unknowns solved block by block")
      break
    end
    copyto!(u0, trial)
    solved += length(vars)
  end
  return solved
end

"""
    _solveDAEInitialization!(u0, rhsAt0, p_vec, mm; extra_residuals, kwargs...) -> u0

The initialization solve (`_solveDAEInitializationErased!`) with the right-hand side and the extra residuals
type-erased (`ErasedRHS`, `ErasedResiduals`): only this forwarding method is compiled per model.
"""
function _solveDAEInitialization!(u0, @nospecialize(rhsAt0::Function), p_vec, mm; @nospecialize(extra_residuals = nothing), kwargs...)
  return _solveDAEInitializationErased!(u0, _erasedRHS(rhsAt0), p_vec, mm; extra_residuals = _erasedResiduals(extra_residuals),
                                        kwargs...)
end

"""
`converged` is set false when no phase converged (the result is then the
best effort the warning or the error below reports).
"""
function _solveDAEInitializationErased!(u0, rhsAt0::ErasedRHS, p_vec, mm; maxiter=200, tol=1e-10, failure_threshold=20.0, pinned=Int[], derivative_targets=Pair{Int, Float64}[], eqLabels=nothing, extra_residuals::Union{Nothing, ErasedResiduals}=nothing, discrete_pinned=Int[], warm::Bool=false,
                                  converged::Base.RefValue{Bool}=Ref(true), t0::Float64=0.0,
                                  sparsity::Union{Nothing, Symbolics.SparseArrays.SparseMatrixCSC{Float64, Int}}=nothing,
                                  stateOrder::Vector{Int}=Int[])
  #= The residuals at the start time t0: the phases below evaluate at their time 0. Assigned
     once (closures below capture it: a reassigned argument would be boxed). =#
  local rhsFunc = t0 == 0.0 ? rhsAt0 : ErasedRHS((du, u, p, t) -> rhsAt0(du, u, p, t + t0))
  local n = length(u0)
  local nMM = size(mm, 1)
  local nSafe = min(n, nMM)
  local alg_idx = [i for i in 1:nSafe if mm[i,i] == 0]
  local der_idx = Int[]
  local der_target = Float64[]
  for (idx, target) in derivative_targets
    1 <= idx <= nSafe || continue
    mm[idx, idx] == 0 && continue
    push!(der_idx, idx)
    push!(der_target, mm[idx, idx] * target)
  end
  local eq_idx = vcat(alg_idx, der_idx)
  local eq_target = vcat(zeros(Float64, length(alg_idx)), der_target)
  if isempty(eq_idx) && extra_residuals === nothing
    return u0
  end
  local du = similar(u0)
  rhsFunc(du, u0, p_vec, 0.0)
  local init_res_vec = _initResidualVec(du, u0, eq_idx, eq_target, extra_residuals)
  if !all(isfinite, init_res_vec) &&
     _nudgeZeroGuesses!(u0, rhsFunc, p_vec, eq_idx, eq_target, extra_residuals, union(pinned, discrete_pinned))
    rhsFunc(du, u0, p_vec, 0.0)
    init_res_vec = _initResidualVec(du, u0, eq_idx, eq_target, extra_residuals)
  end
  local init_res = isempty(init_res_vec) ? 0.0 : maximum(abs, init_res_vec)
  if init_res < tol
    return u0
  end
  if OMBackend.envSwitch("OMBACKEND_INIT_TRACE") && eqLabels !== nothing
    println("[initentry] ", length(u0), " unknowns, u0 = ", first(u0, 20), length(u0) > 20 ? " ..." : "",
            ", pinned ", pinned, ", discrete pinned ", discrete_pinned)
    for (rowk, k) in enumerate(eq_idx)
      local kind = rowk <= length(alg_idx) ? "alg" : "der"
      println("[initrow] ", rowk, " (", kind, " eq ", k, ") ",
              first(string(k <= length(eqLabels) ? eqLabels[k] : "?"), 110))
    end
    extra_residuals === nothing ||
      println("[initrow] rows above ", length(eq_idx), " are initialization-eq extras")
  end
  #= Phase 1 var set: algebraic vars that are NOT pinned by a fixed=true init eq.
     Honouring pins prevents the solver from collapsing `sd1.s_rel = 1` to the
     trivial alg-residual root (-1.5 from m1.s = m2.s = 0 default geometry). =#
  local pinnedSet = OrderedSet(pinned)
  local discretePinnedSet = OrderedSet(discrete_pinned)
  local u0_atEntry = copy(u0)
  local algCandidates = [i for i in alg_idx if !(i in pinnedSet) && !(i in discretePinnedSet)]
  #= Sub-tolerance drift from an entry value is solver noise, not a solved
     value; restoring it exactly keeps relation kinks (e.g. `initial() and
     w < 0`) from firing on a signed numerical zero. =#
  local snapEntryNoise! = () -> begin
    for i in 1:n
      if u0[i] != u0_atEntry[i] && abs(u0[i] - u0_atEntry[i]) < 1e-12 * max(1.0, abs(u0_atEntry[i]))
        u0[i] = u0_atEntry[i]
      end
    end
  end
  local completeInit! = varSet -> begin
    _completeUnderdeterminedInit!(
      u0, rhsFunc, p_vec, eq_idx, varSet, algCandidates, u0_atEntry;
      targets=eq_target, tol=tol, extraRes=extra_residuals, maxiter=maxiter,
      restoreIdx=vcat(pinned, discrete_pinned))
    snapEntryNoise!()
  end
  #= `warm`: u0 is a consistent state for nearby parameter values (a
     re-initialization for other tunable parameter values, DAE_REINIT). A
     plain min-norm Newton over the free variables converges from there in a
     few steps; the cold-start phases below would first spend hundreds of
     iterations on sets that cannot converge (algebraic-only, anchored). =#
  if warm
    local warmVars = [i for i in 1:n if !(i in pinnedSet) && !(i in discretePinnedSet)]
    local u0_warm = copy(u0)
    if !isempty(warmVars) && _solveDAEPhase!(u0_warm, rhsFunc, p_vec, eq_idx, warmVars;
                                             targets=eq_target, maxiter=20, tol=tol,
                                             extraRes=extra_residuals, phaseLabel="warm")
      copyto!(u0, u0_warm)
      completeInit!(warmVars)
      return u0
    end
  end
  #= The one-unknown rows first (`sparsity` the right-hand side's Jacobian structure), over the
     unknowns the latched phase solves (all but the pins and the latches): only where those are
     as many as the rows. A row that reads one unknown is that unknown's in every matching of a
     square system; in phase 1's (the differential states held, more rows than unknowns) it is
     not: MeasuringTemperature's `m = V*d` with the mass m, a state, held at its start 0 gave V
     = 0. Not for a warm start, which is a solution for nearby values already. =#
  local causalVars = Int[i for i in 1:n if !(i in pinnedSet) && !(i in discretePinnedSet)]
  if sparsity !== nothing && extra_residuals === nothing && !isempty(causalVars) &&
     length(causalVars) == length(eq_idx)
    local nCausal = _solveCausalRows!(u0, rhsFunc, p_vec, eq_idx, eq_target, causalVars, sparsity; tol = tol)
    OMBackend.envSwitch("OMBACKEND_INIT_TRACE") && println("[initcausal] ", nCausal, " of ", length(causalVars),
                                                           " unknowns from one-unknown rows")
  end
  #= then the blocks in causal order, the initialization equations' rows among them: each a smaller,
     better-posed Newton than the whole =#
  if sparsity !== nothing && !isempty(causalVars) && length(causalVars) == length(init_res_vec)
    #= an exception at a point only the blocks' Newton or their incidence probes reach (a function's domain, an
       assert): on to the phases below, u0 as after the last converged block =#
    local nBlocks::Int = try
      _solveCausalBlocks!(u0, rhsFunc, p_vec, eq_idx, eq_target, extra_residuals, causalVars, sparsity;
                          tol = tol, maxiter = min(maxiter, 50))
    catch e
      OMBackend._fallback(e, :solveCausalBlocks)
      0
    end
    OMBackend.envSwitch("OMBACKEND_INIT_TRACE") && println("[initblocks] ", nBlocks, " of ", length(causalVars),
                                                           " unknowns solved block by block")
  end
  local alg_unpinned = [i for i in alg_idx if !(i in pinnedSet)]
  local u0_phase1 = copy(u0)
  if !isempty(alg_unpinned) && _solveDAEPhase!(u0_phase1, rhsFunc, p_vec, eq_idx, alg_unpinned;
                     targets=eq_target, maxiter=min(maxiter, 50), tol=tol,
                     extraRes=extra_residuals, phaseLabel="p1-alg")
    copyto!(u0, u0_phase1)
    completeInit!(alg_unpinned)
    return u0
  end
  #= Phase 1b: an underdetermined initialization completed as OpenModelica and Dymola complete it. With
     fewer rows than unknowns (the latches held), the unfixed states the rows leave free are fixed at their
     start values (_initStatesToFix, a matching: never an algebraic unknown) and the square remainder is
     solved. Phase 1 holds every state, which an initialization equation on a state contradicts; phase 2
     frees them all, and its least squares ends anywhere on the solutions near the start values, its
     completion pinning algebraic unknowns (`x + y = 5` from x(start = 1), y(start = 2): x = 2, y = 3, where
     OpenModelica fixes x: x = 1, y = 4; test/Models/InitFixStates.mo). =#
  local fixVars = [i for i in 1:n if !(i in pinnedSet) && !(i in discretePinnedSet)]
  #= the states in the model's order (`stateOrder`, its rank of each unknown): the first ones are fixed =#
  local fixStateCols = sort!(Int[k for (k, i) in enumerate(fixVars) if i <= nSafe && mm[i, i] != 0];
                             by = k -> fixVars[k] <= length(stateOrder) ? stateOrder[fixVars[k]] : typemax(Int))
  if OMBackend.envSwitch("OMBACKEND_INIT_FIX_STATES") && 0 < length(fixVars) - length(init_res_vec) < length(fixStateCols)
    local fixed = _initStatesToFix(_initIncidence(rhsFunc, p_vec, u0, eq_idx, eq_target, extra_residuals, fixVars, sparsity),
                                   length(fixVars), fixStateCols)
    if fixed !== nothing
      local fixedSet = Set{Int}(fixVars[k] for k in fixed)
      local vars1b = Int[i for i in fixVars if !(i in fixedSet)]
      OMBackend.envSwitch("OMBACKEND_INIT_TRACE") &&
        println("[initfix] ", length(fixed), " of ", length(fixStateCols), " unfixed states fixed at their start values")
      #= square: Newton; anchored only where it does not converge (the anchor rows pull off the root) =#
      local u0_1b = copy(u0)
      local solved1b = _solveDAEPhase!(u0_1b, rhsFunc, p_vec, eq_idx, vars1b; targets=eq_target, maxiter=min(maxiter, 50),
                                       tol=tol, extraRes=extra_residuals, phaseLabel="p1b-fixed")
      if !solved1b
        copyto!(u0_1b, u0)
        solved1b = _solveDAEPhaseAnchored!(u0_1b, rhsFunc, p_vec, eq_idx, vars1b,
                                           OMBackend.envSwitch("OMBACKEND_INIT_ANCHOR") ? copy(u0) : nothing;
                                           targets=eq_target, maxiter=maxiter, tol=tol,
                                           extraRes=extra_residuals, phaseLabel="p1b-fixed")
      end
      if solved1b
        copyto!(u0, u0_1b)
        completeInit!(vars1b)
        return u0
      end
    end
  end
  #= Phase 2: free all unpinned vars (alg + diff without fixed=true) to allow
     algebraic eqs to be satisfied by adjusting differential vars. Pinned vars
     stay at user-requested values. Anchored to the entry guesses so an
     underdetermined manifold resolves to the nearest root. =#
  local u0_guess = OMBackend.envSwitch("OMBACKEND_INIT_ANCHOR") ? copy(u0) : nothing
  local all_unpinned = [i for i in 1:n if !(i in pinnedSet)]
  #= Latched phase: the discrete latches held at their init values along with
     the user pins. Their defining rows evaluate locally constant, so the
     active branch is fixed and the landscape near the entry point is smooth;
     leaving the latches free makes those rows relation cliffs the solver
     keeps tripping over. =#
  local latched_unpinned = [i for i in all_unpinned if !(i in discretePinnedSet)]
  if !isempty(discrete_pinned) && !isempty(latched_unpinned)
    local u0_latched = copy(u0)
    if _solveDAEPhaseAnchored!(u0_latched, rhsFunc, p_vec, eq_idx, latched_unpinned, u0_guess;
                               targets=eq_target, maxiter=maxiter, tol=tol,
                               extraRes=extra_residuals, phaseLabel="p2-latched")
      copyto!(u0, u0_latched)
      completeInit!(latched_unpinned)
      return u0
    end
  end
  local u0_phase2 = copy(u0)
  if !isempty(all_unpinned) && _solveDAEPhaseAnchored!(u0_phase2, rhsFunc, p_vec, eq_idx, all_unpinned, u0_guess;
                                                       targets=eq_target, maxiter=maxiter, tol=tol,
                                                       extraRes=extra_residuals, phaseLabel="p2")
    copyto!(u0, u0_phase2)
    completeInit!(latched_unpinned)
    return u0
  end
  #= Phase 3 escape hatch: free EVERY var, including pinned. Some kinematic
     loops (PersonalityAspects, multibody overconstrained connectors) are
     not consistent with all fixed=true starts simultaneously and need the
     solver to relax pins to find any consistent root. Pre-pinned-fix
     behavior. Reach here only when both unpinned phases failed. =#
  local all_idx = collect(1:n)
  local u0_entry = copy(u0)
  local phase3_ok = _solveDAEPhaseAnchored!(u0, rhsFunc, p_vec, eq_idx, all_idx, u0_guess;
                                            targets=eq_target, maxiter=maxiter, tol=tol,
                                            extraRes=extra_residuals, phaseLabel="p3")
  #= Phase 3 relaxed the pins to find a root. Re-impose the user-constrained
     values on top of that root and re-solve only the free remainder: from a
     near-root start the constrained solve converges where the same pinned
     phase diverged from far away. The discrete latches are pinned here too:
     left free they are a continuous relaxation of step functions whose
     defining rows are cliffs; held at their init values those rows evaluate
     locally constant and the remaining system is smooth. Keeps the free
     root when the constrained polish cannot converge. =#
  if phase3_ok && !(isempty(pinned) && isempty(discrete_pinned)) &&
     OMBackend.envSwitch("OMBACKEND_INIT_REPIN")
    local repinVars = latched_unpinned
    if !isempty(repinVars)
      local u0_repin = copy(u0)
      for i in pinned
        u0_repin[i] = u0_entry[i]
      end
      for i in discrete_pinned
        u0_repin[i] = u0_entry[i]
      end
      if _solveDAEPhase!(u0_repin, rhsFunc, p_vec, eq_idx, repinVars;
                         targets=eq_target, maxiter=maxiter, tol=tol,
                         extraRes=extra_residuals, phaseLabel="repin")
        copyto!(u0, u0_repin)
      end
    end
  end
  phase3_ok && completeInit!(latched_unpinned)
  phase3_ok || snapEntryNoise!()
  if !phase3_ok
    converged[] = false
    rhsFunc(du, u0, p_vec, 0.0)
    local resids = abs.(_initResidualVec(du, u0, eq_idx, eq_target, extra_residuals))
    local final_res = maximum(resids)
    local rowsDetail = rows -> join((begin
        local label
        if w <= length(eq_idx)
          local k = eq_idx[w]
          label = string("eq[", k, "]",
                         eqLabels === nothing || k > length(eqLabels) ? "" :
                         string(" :: ", first(string(eqLabels[k]), 160)))
        else
          label = string("initialization eq row ", w - length(eq_idx))
        end
        string(label, " residual ", round(resids[w], sigdigits = 4))
      end for w in rows), "\n  ")
    if !isfinite(final_res)
      local bad = findall(!isfinite, resids)
      @error "DAE init: residual is non-finite ($final_res); ICs unverified, integrator may NaN. Rows:\n  " *
             rowsDetail(bad[1:min(5, length(bad))])
    elseif final_res >= failure_threshold
      local order = sortperm(resids; rev = true)
      error("DAE init: residual $(round(final_res, sigdigits=4)) exceeds threshold $(failure_threshold); refusing inconsistent ICs. Worst:\n  $(rowsDetail(order[1:min(5, length(order))]))")
    else
      @warn "DAE init: did not fully converge (residual $(round(final_res, sigdigits=4)) < threshold $(failure_threshold)); proceeding."
    end
  end
  return u0
end

"""
    _initStatesToFix(incidence, nVars, stateCols) -> Union{Nothing, Vector{Int}}

The unfixed states an underdetermined initialization fixes at their start values, as OpenModelica
(Initialization.fixInitialSystem; Ochel and Bachmann, EOOLT 2013) and Dymola ("Assuming fixed start
value for the continuous states") do: one artificial row per missing equation, which reads the unfixed
states `stateCols`, and the rows (`incidence[r]`, the unknowns `1:nVars` row `r` reads) matched to the
unknowns; the states the artificial rows are matched to. The artificial rows first: they take the first
states, which a row takes from them only where it has no other unknown, so the earlier states are the
ones fixed (OpenModelica's choice in `x + y = 5` for x(start = 1), y(start = 2): x = 1, y = 4). An
algebraic unknown is never fixed: `nothing` where no matching covers every unknown (the rows singular
without fixing an algebraic unknown), and for an overdetermined system.
"""
function _initStatesToFix(incidence::Vector{Vector{Int}}, nVars::Int, stateCols::Vector{Int})::Union{Nothing, Vector{Int}}
  local nMissing = nVars - length(incidence)
  nMissing < 0 && return nothing
  nMissing == 0 && return Int[]
  isempty(stateCols) && return nothing
  local rows = vcat([stateCols for _ in 1:nMissing], incidence)
  local assign = _maximumMatching(rows, nVars)
  any(==(0), assign) && return nothing
  return sort!([c for c in 1:nVars if assign[c] <= nMissing])
end

"""
    _maximumMatching(rows, n) -> Vector{Int}

A maximum matching of `rows` (each the columns `1:n` it reads) to the columns: `assign[c]`, the row
column `c` is matched to, 0 for none (Kuhn's augmenting paths, rows in order). The search keeps its
path on explicit stacks: a recursive one can overflow the stack on a large system, which ends the
process on this platform.
"""
function _maximumMatching(rows::Vector{Vector{Int}}, n::Int)::Vector{Int}
  local assign = zeros(Int, n)
  local visited = falses(n)
  local rowStack = Int[]
  local posStack = Int[]
  local colStack = Int[]
  for r in eachindex(rows)
    local direct = findfirst(c -> assign[c] == 0, rows[r])
    if direct !== nothing
      assign[rows[r][direct]] = r
      continue
    end
    fill!(visited, false)
    empty!(rowStack); empty!(posStack); empty!(colStack)
    push!(rowStack, r); push!(posStack, 0)
    while !isempty(rowStack)
      local row = rowStack[end]
      local pos = posStack[end] + 1
      if pos > length(rows[row])
        #= exhausted: back to the row whose column led here =#
        pop!(rowStack); pop!(posStack)
        isempty(colStack) || pop!(colStack)
        continue
      end
      posStack[end] = pos
      local c = rows[row][pos]
      visited[c] && continue
      visited[c] = true
      push!(colStack, c)
      if assign[c] == 0
        #= an augmenting path: each row on it takes the column it chose =#
        for d in eachindex(colStack)
          assign[colStack[d]] = rowStack[d]
        end
        break
      end
      push!(rowStack, assign[c]); push!(posStack, 0)
    end
  end
  return assign
end

"""
    _initIncidence(rhsFunc, p_vec, u0, eq_idx, targets, extraRes, vars, sparsity) -> Vector{Vector{Int}}

The unknowns (positions in `vars`) each initialization row reads: a right-hand side's row (`eq_idx`)
by the right-hand side's Jacobian structure `sparsity` when known, the initialization-equation rows
(`extraRes`) and the rest by finite differences, at the entry point and at a point near it (a factor
that is 0 there, a current at its start value 0, hides an occurrence).
"""
function _initIncidence(rhsFunc::F, p_vec::Vector{Float64}, u0::Vector{Float64}, eq_idx::Vector{Int},
                        targets::Vector{Float64}, extraRes, vars::Vector{Int},
                        sparsity::Union{Nothing, Symbolics.SparseArrays.SparseMatrixCSC{Float64, Int}})::Vector{Vector{Int}} where {F}
  local du = similar(u0)
  rhsFunc(du, u0, p_vec, 0.0)
  local res = _initResidualVec(du, u0, eq_idx, targets, extraRes)
  local J = _fdInitResidualJacobian(rhsFunc, p_vec, u0, eq_idx, targets, extraRes, vars, res)
  #= a deterministic point near the entry (the golden-ratio sequence) =#
  local near = copy(u0)
  for (k, j) in enumerate(vars)
    near[j] += 1e-3 * max(1.0, abs(u0[j])) * (2 * mod(k * 0.6180339887498949, 1.0) - 1)
  end
  rhsFunc(du, near, p_vec, 0.0)
  local resNear = _initResidualVec(du, near, eq_idx, targets, extraRes)
  local Jnear = all(isfinite, resNear) ?
    _fdInitResidualJacobian(rhsFunc, p_vec, near, eq_idx, targets, extraRes, vars, resNear) : nothing
  local position = Dict{Int, Int}(j => k for (k, j) in enumerate(vars))
  local structure = sparsity === nothing ? nothing : permutedims(sparsity)   # column k: the columns row k reads
  local incidence = Vector{Vector{Int}}(undef, length(res))
  for r in eachindex(res)
    if structure !== nothing && r <= length(eq_idx) && eq_idx[r] <= size(structure, 2)
      incidence[r] = sort!(Int[position[j] for j in Symbolics.SparseArrays.rowvals(structure)[Symbolics.SparseArrays.nzrange(structure, eq_idx[r])]
                              if haskey(position, j)])
    else
      incidence[r] = Int[k for k in eachindex(vars)
                         if J[r, k] != 0 || !isfinite(J[r, k]) || (Jnear !== nothing && Jnear[r, k] != 0)]
    end
  end
  return incidence
end

"""
Anchored attempt first (when a guess vector is provided), then an
unanchored polish continuing from the stalled endpoint: the anchor term
biases the stationary point off the residual zero when the root lies far
from the guesses, stalling short of tolerance, but branch selection is
already done at the stalled endpoint.
"""
function _solveDAEPhaseAnchored!(u0, rhsFunc, p_vec, eq_idx, var_idx, anchorVals;
                                 targets=zeros(Float64, length(eq_idx)), maxiter=50, tol=1e-10,
                                 extraRes=nothing, phaseLabel::String="")
  local entry = anchorVals === nothing ? nothing : copy(u0)
  if _solveDAEPhase!(u0, rhsFunc, p_vec, eq_idx, var_idx;
                     targets=targets, maxiter=maxiter, tol=tol, anchorVals=anchorVals,
                     extraRes=extraRes, phaseLabel=string(phaseLabel, "-anchored"))
    return true
  end
  anchorVals === nothing && return false
  #= The anchored attempt can end worse than it started; polish from the
     better of its endpoint and the phase entry point. =#
  local du = similar(u0)
  rhsFunc(du, u0, p_vec, 0.0)
  local endNorm = maximum(abs, _initResidualVec(du, u0, eq_idx, targets, extraRes))
  rhsFunc(du, entry, p_vec, 0.0)
  local entryNorm = maximum(abs, _initResidualVec(du, entry, eq_idx, targets, extraRes))
  if !isfinite(endNorm) || entryNorm < endNorm
    copyto!(u0, entry)
  end
  return _solveDAEPhase!(u0, rhsFunc, p_vec, eq_idx, var_idx;
                         targets=targets, maxiter=maxiter, tol=tol, extraRes=extraRes,
                         phaseLabel=string(phaseLabel, "-polish"))
end

"""
The Newton step of an init phase: the minimum-norm step, pinv's, which
also serves the underdetermined and anchored phases. Where pinv's rank
cutoff (eps * min(m, n) * the largest singular value) drops a direction of
an unanchored Jacobian with at least as many rows as columns, and the
Jacobian is regular once its rows and columns are scaled, the step is that
scaled matrix's Newton step (for more rows than columns, its row-weighted
least-squares step) when it descends. A badly scaled Jacobian is not a
singular one: an ideal diode in the wrong mode at the start has s = -8e5
against Ron = 1e-5; the dropped direction is the one that corrects s, the
step moved the if-equation's coefficient instead, the phase crept above
its tolerance, and a later phase moved the states to fit the wrong mode (a
capacitor's start value). Elsewhere the step is pinv's, bit for bit: a V6
engine cylinder's piston reaches its stroke limit to 1e-12, and a last-bit
change of the initial crank angle trips that assert.
"""
function _newtonStep(Js, ress, unanchored::Bool)
  if unanchored && size(Js, 1) >= size(Js, 2) && !isempty(Js) && !LinearAlgebra.isdiag(Js)
    local S = LinearAlgebra.svdvals(Js)
    if S[end] <= eps(Float64) * minimum(size(Js)) * S[1]
      local rowScale = [(m = maximum(abs, @view Js[i, :]); m > floatmin(Float64) ? 1 / m : 1.0) for i in 1:size(Js, 1)]
      local Jr = rowScale .* Js
      local colScale = [(m = maximum(abs, @view Jr[:, j]); m > floatmin(Float64) ? 1 / m : 1.0) for j in 1:size(Js, 2)]
      local Fc = LinearAlgebra.svd(Jr .* colScale')
      if Fc.S[end] > 1e-10 * Fc.S[1]
        local delta = colScale .* (Fc \ (rowScale .* ress))
        #= A descent direction for the phase's objective (ress' Js delta > 0): a
           row-weighted least-squares step of an inconsistent system may not be. =#
        LinearAlgebra.dot(ress, Js * delta) > 0 && return delta
      end
    end
  end
  return LinearAlgebra.pinv(Js) * ress
end

"""
    stationaryPhase(norms) -> Bool

Whether a phase's iterate has stopped moving: its last `INIT_STATIONARY_ITERATIONS` equilibrated
residual norms (`norms`) agree to 1e-8 (they wander at round-off, 1e-11 to 3e-9, where the
iterate has stalled). The remaining iterations would only repeat the Jacobian and the line
search: MSL PrismaticConstraint's anchored phase held 0.0083998691 for 195 of its 200, Fluid's
PumpingSystem spent 500 s of its translation so (IF97 calls that throw at every trial).
"""
function stationaryPhase(norms::Vector{Float64})::Bool
  length(norms) >= INIT_STATIONARY_ITERATIONS || return false
  local last = @view norms[end - INIT_STATIONARY_ITERATIONS + 1:end]
  local (lo, hi) = extrema(last)
  return hi - lo <= 1.0e-8 * max(hi, eps(Float64))
end

"The iterations over which `stationaryPhase` compares the norms."
const INIT_STATIONARY_ITERATIONS = 10

"""
    crawlingPhase(norms) -> Bool

Whether a phase has stopped making progress: its best equilibrated norm (`norms`) of the last
`INIT_CRAWL_ITERATIONS` iterations is less than `INIT_CRAWL_FACTOR` below the best before them.
Newton near a root gains orders of magnitude per iteration; an anchored phase wandering between
1.5e5 and 1.7e5 (Fluid's PumpingSystem, 200 iterations from each of five entries, IF97 calls
throwing at every trial) does not reach 1e-10 in the iterations left.
"""
function crawlingPhase(norms::Vector{Float64})::Bool
  local n = length(norms)
  n > INIT_CRAWL_ITERATIONS || return false
  local before = minimum(@view norms[1:(n - INIT_CRAWL_ITERATIONS)])
  local recent = minimum(@view norms[(n - INIT_CRAWL_ITERATIONS + 1):n])
  return recent > (1 - INIT_CRAWL_FACTOR) * before
end

"The window and the relative gain of `crawlingPhase`."
const INIT_CRAWL_ITERATIONS = 50
const INIT_CRAWL_FACTOR = 0.01

"""
The unknowns of a stalled phase at exactly 0 whose Jacobian column is 0, not nudged yet
(_solveDAEPhase!): at a flat point of every row that reads them Newton cannot move them, and the
other unknowns take the rows' mismatch. MSL MomentumBalanceFittings: suddenExpansion2.m_flow from
its start 0, where the quadratic pressure losses are flat; the two pressures of the loop took a
third of its 10 kPa each, and the phases stalled there. Nudged to 1, as a zero guess at a
non-finite entry (_nudgeZeroGuesses!).
"""
_flatZeroUnknowns(J::AbstractMatrix{Float64}, u0::AbstractVector{Float64}, var_idx::AbstractVector{Int}, nudged::Set{Int})::Vector{Int} =
  Int[var_idx[k] for k in eachindex(var_idx) if iszero(u0[var_idx[k]]) && !(var_idx[k] in nudged) && all(iszero, @view J[:, k])]

function _solveDAEPhase!(u0, rhsFunc, p_vec, eq_idx, var_idx; targets=zeros(Float64, length(eq_idx)), maxiter=50, tol=1e-10, anchorVals=nothing, anchorWeight=1e-2, extraRes=nothing, phaseLabel::String="")
  local nVar = length(var_idx)
  local du = similar(u0)
  local anchorRows = anchorVals === nothing ? nothing :
    Matrix(LinearAlgebra.Diagonal(fill(anchorWeight, nVar)))
  local traceInit = OMBackend.envSwitch("OMBACKEND_INIT_TRACE")
  #= Fixed per-phase row equilibration, from the ENTRY Jacobian: symbolic
     elimination can emit rows whose constant coefficients reach 1e40+, so
     raw residual units make both the tolerance and any line-search measure
     meaningless. Dividing each row by its entry gradient magnitude (floored
     at 1 so weak rows are never inflated) puts residuals in solve-variable
     units, comparable across rows AND across iterates. Refreshed once, after
     the first step, for the rows a far entry guess scaled 1e3 times too high. =#
  local rowNorm = Float64[]
  local lastNres = Float64[]
  local phiEntry = Inf
  #= the equilibrated norms of the iterations: a phase whose iterate no longer moves (`stationaryPhase`) stops =#
  local norms = Float64[]
  local stationary = false
  local nudged = Set{Int}()
  for iter in 1:maxiter
    rhsFunc(du, u0, p_vec, 0.0)
    local res = _initResidualVec(du, u0, eq_idx, targets, extraRes)
    local nRows = length(res)
    if !all(isfinite, res)
      traceInit && println("[initphase ", phaseLabel, "] FAIL nonfinite residual")
      return false
    end
    #= Raw convergence is stricter than equilibrated convergence (rowNorm >= 1),
       so this early-out is safe and avoids the entry-Jacobian cost. =#
    if maximum(abs, res) < tol
      return true
    end
    local J = _fdInitResidualJacobian(rhsFunc, p_vec, u0, eq_idx, targets, extraRes, var_idx, res)
    if any(!isfinite, J)
      traceInit && println("[initphase ", phaseLabel, "] FAIL nonfinite Jacobian")
      return false
    end
    if iter == 1
      rowNorm = OMBackend.envSwitch("OMBACKEND_INIT_ROWSCALE") ?
        [max(maximum(abs, @view J[i, :]), 1.0) for i in 1:nRows] : ones(nRows)
    elseif iter == 2 && OMBackend.envSwitch("OMBACKEND_INIT_ROWSCALE")
      #= A row whose entry scale came from a far guess (MSL ThreeSprings: a line force's direction
         r/s_small = -2e9 where its length's start is 0) is equilibrated again after the first
         step, where its scale is 1e3 times smaller: frozen, it stays weighted out and the phase
         stalls at a floor above the tolerance. The objective's entry value with it. =#
      local rescaled = false
      for i in 1:nRows
        local current = max(maximum(abs, @view J[i, :]), 1.0)
        rowNorm[i] > 1.0e3 * current && ((rowNorm[i], rescaled) = (current, true))
      end
      rescaled && (phiEntry = Inf)
    end
    local norm_res = maximum(abs, res ./ rowNorm)
    traceInit && (lastNres = res ./ rowNorm)
    traceInit && println("[initphase ", phaseLabel, "] iter=", iter,
                         " norm=", round(norm_res, sigdigits = 4),
                         " rows=", nRows, " vars=", nVar)
    if traceInit && iter == 1
      local nres = res ./ rowNorm
      local order = sortperm(abs.(nres); rev = true)
      println("[initphase ", phaseLabel, "] worst rows (equilibrated): ",
              join((string(order[k], "=>", round(nres[order[k]], sigdigits = 3))
                    for k in 1:min(4, length(order))), " "))
    end
    if norm_res < tol
      return true
    end
    push!(norms, norm_res)
    if stationaryPhase(norms) || crawlingPhase(norms)
      local flat = _flatZeroUnknowns(J, u0, var_idx, nudged)
      if !isempty(flat)
        traceInit && println("[initphase ", phaseLabel, "] flat at 0, nudged to 1: ", flat)
        foreach(j -> (u0[j] = 1.0; push!(nudged, j)), flat)
        empty!(norms)
        continue
      end
      stationary = true
      break
    end
    local Js = J ./ rowNorm
    local ress = res ./ rowNorm
    #= Guess anchoring: an underdetermined init manifold (e.g. a NoInit PI
       state) otherwise lets Newton converge to an arbitrary far root; weak
       anchor rows pull the null-space components toward the start values,
       matching the pick-the-root-nearest-the-guess tool convention. =#
    if anchorRows !== nothing
      local ares = [anchorWeight * (u0[jstate] - anchorVals[jstate]) for jstate in var_idx]
      Js = vcat(Js, anchorRows)
      ress = vcat(ress, ares)
    end
    local delta = _newtonStep(Js, ress, anchorRows === nothing)
    #= Converged at the round-off floor: the full Newton step moves the
       variables by round-off only, so the residual cannot shrink further. An
       ideal diode conducting V/Ron = 1e7 A leaves ~1e-9 in its rows (the MSL
       MultiPhase Rectifier stalled at 1.863e-9 for 20 iterations). Only with
       a residual at round-off relative to the variables (a singular
       Jacobian's least-squares step can be tiny away from a root too), and
       not in an anchored phase (its least-squares stationary point has a
       zero step with the anchors pulling the residual off zero). =#
    local uMax = max(1.0, maximum(j -> abs(u0[j]), var_idx; init = 0.0))
    if anchorRows === nothing && norm_res <= 1.0e3 * eps(Float64) * uMax &&
       maximum(abs, delta; init = 0.0) <= 64 * eps(Float64) * uMax
      traceInit && println("[initphase ", phaseLabel, "] converged at the round-off floor, norm=",
                           round(norm_res, sigdigits = 4))
      return true
    end
    #= The acceptance measure must match the objective the direction
       minimizes: the equilibrated least-squares norm. Raw max-norm
       acceptance on mixed-scale systems rejects every step (one huge-
       coefficient row dominates any movement), forcing the blind fallback
       to catapult. The FIXED rowNorm keeps the measure comparable across
       iterates. =#
    local scaledNorm = function (resVec, uVec)
      local s = 0.0
      for i in 1:nRows
        s += (resVec[i] / rowNorm[i])^2
      end
      if anchorRows !== nothing
        for jstate in var_idx
          s += (anchorWeight * (uVec[jstate] - anchorVals[jstate]))^2
        end
      end
      return sqrt(s)
    end
    local phi0 = scaledNorm(res, u0)
    isinf(phiEntry) && (phiEntry = phi0)
    #= Backtracking line search: a fixed step-length clamp starves quasi-linear
       systems whose solution components are large (e.g. di/dt = V/L). Take the
       full Newton step when it reduces the objective; halve only when it does not. =#
    local alpha = 1.0
    local accepted = false
    local u_trial = similar(u0)
    local du_trial = similar(u0)
    while alpha >= 1.0 / 1024
      copyto!(u_trial, u0)
      for (jcol, jstate) in enumerate(var_idx)
        u_trial[jstate] -= alpha * delta[jcol]
      end
      rhsFunc(du_trial, u_trial, p_vec, 0.0)
      local trial_res = _initResidualVec(du_trial, u_trial, eq_idx, targets, extraRes)
      local trial_raw = maximum(abs, trial_res)
      local phi_t = scaledNorm(trial_res, u_trial)
      if (isfinite(trial_raw) && trial_raw < tol) ||
         (isfinite(phi_t) && phi_t < phi0)
        copyto!(u0, u_trial)
        accepted = true
        break
      end
      alpha /= 2
    end
    if !accepted
      #= Non-monotone fallback: piecewise residuals (ifelse branches) stall a
         monotone search at kink-local minima; a bounded blind step can cross.
         The step is taken only while the scaled objective stays within a
         fixed factor of the PHASE-ENTRY objective: a per-iteration bound
         would compound into a catapult, no bound at all freezes the phase at
         a blown-up iterate with a vanished Newton direction. =#
      local alphaFB = min(1.0, 10.0 / max(1.0, LinearAlgebra.norm(delta)))
      copyto!(u_trial, u0)
      for (jcol, jstate) in enumerate(var_idx)
        u_trial[jstate] -= alphaFB * delta[jcol]
      end
      rhsFunc(du_trial, u_trial, p_vec, 0.0)
      local fb_res = _initResidualVec(du_trial, u_trial, eq_idx, targets, extraRes)
      local phi_fb = scaledNorm(fb_res, u_trial)
      if isfinite(phi_fb) && phi_fb <= 1.0e3 * phiEntry
        copyto!(u0, u_trial)
      else
        traceInit && println("[initphase ", phaseLabel, "] FAIL blind step rejected, scaled norm ",
                             round(phi_fb, sigdigits = 4), " vs entry ", round(phiEntry, sigdigits = 4))
        return false
      end
    end
  end
  if traceInit
    local msg = "[initphase " * phaseLabel * "] FAIL " *
                (stationary ? "no progress (stationaryPhase, crawlingPhase)" : "maxiter exhausted")
    if !isempty(lastNres)
      local order = sortperm(abs.(lastNres); rev = true)
      msg *= ", worst rows: " * join((string(order[k], "=>", round(lastNres[order[k]], sigdigits = 3))
                                      for k in 1:min(4, length(order))), " ")
    end
    println(msg)
  end
  return false
end
