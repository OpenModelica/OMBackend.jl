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

#= Residual vector of one init iterate: dynamics rows (du minus targets) plus
   the optional extra rows from signal-valued initialization equations. =#
function _initResidualVec(du, u, eq_idx, targets, extraRes)
  local base = du[eq_idx] .- targets
  extraRes === nothing && return base
  return vcat(base, extraRes(du, u))
end

#= Forward-difference Jacobian of the init residual at u0 over the var_idx
   columns. `res` is the residual already evaluated at u0. =#
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

#= Underdetermined-init completion: a converged root whose Jacobian has a
   nontrivial null space means the equations leave spare degrees of freedom.
   Tool convention is to complete such a system by fixing selected variables
   at their start values. Pin the algebraic unknowns with the largest
   null-space projections at their entry values and re-solve the remainder
   from the root; keep the completed root only when the constrained solve
   converges. =#
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

#= A free unknown guessed at exactly 0 can make a residual row non-finite
   at the entry (MSL QS FluxTubes GeneralLeakage: `0 = -7e-6 + 0.3/G_m`, G_m
   without a start value), and then no phase can start. Those guesses become
   1: all together, else one at a time where it lowers the number of
   non-finite rows. Only at such an entry, which no phase could solve from.
   Whether a guess changed. =#
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

#= `converged` is set false when no phase converged (the result is then the
   best effort the warning or the error below reports). =#
function _solveDAEInitialization!(u0, rhsFunc, p_vec, mm; maxiter=200, tol=1e-10, failure_threshold=20.0, pinned=Int[], derivative_targets=Pair{Int, Float64}[], eqLabels=nothing, extra_residuals=nothing, discrete_pinned=Int[], warm::Bool=false,
                                  converged::Base.RefValue{Bool}=Ref(true), t0::Float64=0.0)
  #= The residuals at the start time t0: the phases below evaluate at their time 0. =#
  t0 == 0.0 || (rhsFunc = let f = rhsFunc; (du, u, p, t) -> f(du, u, p, t + t0); end)
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
  local alg_unpinned = [i for i in alg_idx if !(i in pinnedSet)]
  local u0_phase1 = copy(u0)
  if !isempty(alg_unpinned) && _solveDAEPhase!(u0_phase1, rhsFunc, p_vec, eq_idx, alg_unpinned;
                     targets=eq_target, maxiter=min(maxiter, 50), tol=tol,
                     extraRes=extra_residuals, phaseLabel="p1-alg")
    copyto!(u0, u0_phase1)
    completeInit!(alg_unpinned)
    return u0
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

#= Anchored attempt first (when a guess vector is provided), then an
   unanchored polish continuing from the stalled endpoint: the anchor term
   biases the stationary point off the residual zero when the root lies far
   from the guesses, stalling short of tolerance, but branch selection is
   already done at the stalled endpoint. =#
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

#= The Newton step of an init phase: the minimum-norm step, pinv's, which
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
   change of the initial crank angle trips that assert. =#
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
     units, comparable across rows AND across iterates. =#
  local rowNorm = Float64[]
  local lastNres = Float64[]
  local phiEntry = Inf
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
    iter == 1 && (phiEntry = phi0)
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
    local msg = "[initphase " * phaseLabel * "] FAIL maxiter exhausted"
    if !isempty(lastNres)
      local order = sortperm(abs.(lastNres); rev = true)
      msg *= ", worst rows: " * join((string(order[k], "=>", round(lastNres[order[k]], sigdigits = 3))
                                      for k in 1:min(4, length(order))), " ")
    end
    println(msg)
  end
  return false
end
