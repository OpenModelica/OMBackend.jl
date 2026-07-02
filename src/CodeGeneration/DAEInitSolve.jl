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
    u_pert[jstate] += eps_fd
    rhsFunc(du_pert, u_pert, p_vec, 0.0)
    J[:, jcol] = (_initResidualVec(du_pert, u_pert, eq_idx, targets, extraRes) .- res) ./ eps_fd
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
  get(ENV, "OMBACKEND_INIT_COMPLETE", "true") == "true" || return false
  isempty(var_idx) && return false
  #= Only algebraic unknowns are completion candidates; with none, the
     Jacobian and SVD below cannot produce a pick. =#
  isempty(algCandidates) && return false
  local traceInit = get(ENV, "OMBACKEND_INIT_TRACE", "") == "true"
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

function _solveDAEInitialization!(u0, rhsFunc, p_vec, mm; maxiter=200, tol=1e-10, failure_threshold=20.0, pinned=Int[], derivative_targets=Pair{Int, Float64}[], eqLabels=nothing, extra_residuals=nothing, discrete_pinned=Int[])
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
  local init_res = isempty(init_res_vec) ? 0.0 : maximum(abs, init_res_vec)
  if init_res < tol
    return u0
  end
  if get(ENV, "OMBACKEND_INIT_TRACE", "") == "true" && eqLabels !== nothing
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
  local u0_guess = get(ENV, "OMBACKEND_INIT_ANCHOR", "true") == "true" ? copy(u0) : nothing
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
     get(ENV, "OMBACKEND_INIT_REPIN", "true") == "true"
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
    rhsFunc(du, u0, p_vec, 0.0)
    local resids = abs.(_initResidualVec(du, u0, eq_idx, eq_target, extra_residuals))
    local final_res = maximum(resids)
    if !isfinite(final_res)
      @error "DAE init: residual is non-finite ($final_res); ICs unverified, integrator may NaN."
    elseif final_res >= failure_threshold
      local order = sortperm(resids; rev = true)
      local worst = order[1:min(5, length(order))]
      local detail = join((begin
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
      end for w in worst), "\n  ")
      error("DAE init: residual $(round(final_res, sigdigits=4)) exceeds threshold $(failure_threshold); refusing inconsistent ICs. Worst:\n  $(detail)")
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

function _solveDAEPhase!(u0, rhsFunc, p_vec, eq_idx, var_idx; targets=zeros(Float64, length(eq_idx)), maxiter=50, tol=1e-10, anchorVals=nothing, anchorWeight=1e-2, extraRes=nothing, phaseLabel::String="")
  local nVar = length(var_idx)
  local du = similar(u0)
  local anchorRows = anchorVals === nothing ? nothing :
    Matrix(LinearAlgebra.Diagonal(fill(anchorWeight, nVar)))
  local traceInit = get(ENV, "OMBACKEND_INIT_TRACE", "") == "true"
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
      rowNorm = get(ENV, "OMBACKEND_INIT_ROWSCALE", "true") == "true" ?
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
    local delta = LinearAlgebra.pinv(Js) * ress
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
