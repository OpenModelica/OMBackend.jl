#=
  Two-phase Newton solver for DAE consistent-IC.
  Shared by DirectRHS and MTK pre-solve paths.
  Phase 1 fixes differential states and solves algebraic residuals; phase 2
  frees all unknowns to handle rank-deficient kinematic loops.
=#

function _solveDAEInitialization!(u0, rhsFunc, p_vec, mm; maxiter=200, tol=1e-10, failure_threshold=1e-1)
  local n = length(u0)
  local nMM = size(mm, 1)
  local nSafe = min(n, nMM)
  local alg_idx = [i for i in 1:nSafe if mm[i,i] == 0]
  if isempty(alg_idx)
    return u0
  end
  local du = similar(u0)
  rhsFunc(du, u0, p_vec, 0.0)
  local init_res = maximum(abs, du[alg_idx])
  if init_res < tol
    return u0
  end
  local u0_phase1 = copy(u0)
  if _solveDAEPhase!(u0_phase1, rhsFunc, p_vec, alg_idx, alg_idx;
                     maxiter=min(maxiter, 50), tol=tol)
    copyto!(u0, u0_phase1)
    return u0
  end
  local all_idx = collect(1:n)
  local phase2_ok = _solveDAEPhase!(u0, rhsFunc, p_vec, alg_idx, all_idx;
                                    maxiter=maxiter, tol=tol)
  if !phase2_ok
    rhsFunc(du, u0, p_vec, 0.0)
    local final_res = maximum(abs, du[alg_idx])
    if !isfinite(final_res)
      @error "DAE init: residual is non-finite ($final_res); ICs unverified, integrator may NaN."
    elseif final_res >= failure_threshold
      error("DAE init: residual $(round(final_res, sigdigits=4)) exceeds threshold $(failure_threshold); refusing inconsistent ICs.")
    else
      @warn "DAE init: did not fully converge (residual $(round(final_res, sigdigits=4)) < threshold $(failure_threshold)); proceeding."
    end
  end
  return u0
end

function _solveDAEPhase!(u0, rhsFunc, p_vec, eq_idx, var_idx; maxiter=50, tol=1e-10)
  local nEq = length(eq_idx)
  local nVar = length(var_idx)
  local du = similar(u0)
  local eps_fd = 1e-7
  for iter in 1:maxiter
    rhsFunc(du, u0, p_vec, 0.0)
    local res = du[eq_idx]
    local norm_res = maximum(abs, res)
    if !isfinite(norm_res)
      return false
    end
    if norm_res < tol
      return true
    end
    local J = zeros(nEq, nVar)
    local du_pert = similar(u0)
    for (jcol, jstate) in enumerate(var_idx)
      local u_pert = copy(u0)
      u_pert[jstate] += eps_fd
      rhsFunc(du_pert, u_pert, p_vec, 0.0)
      J[:, jcol] = (du_pert[eq_idx] .- res) ./ eps_fd
    end
    if any(!isfinite, J)
      return false
    end
    local delta = LinearAlgebra.pinv(J) * res
    local alpha = min(1.0, 10.0 / max(1.0, LinearAlgebra.norm(delta)))
    for (jcol, jstate) in enumerate(var_idx)
      u0[jstate] -= alpha * delta[jcol]
    end
  end
  return false
end
