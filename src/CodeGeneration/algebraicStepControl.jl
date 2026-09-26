#= The algebraic unknowns of a mass-matrix DAE solved with a Rodas method.

   A stiffly accurate Rosenbrock step (Rodas5P) satisfies the algebraic
   equations only in their linearization, and only at its end, so its
   embedded error estimate is zero for the algebraic unknowns:
   - It does not see their variation between the step ends. An algebraic
     unknown driven by time grows steps by 10 each (x' = 0, 0 = y - 100 sin t:
     6 steps over [0, 10], a dense output wrong by 200). Events on such a
     variable are located on that dense output: a friction element held by a
     time-varying torque missed a short excess over its breakaway limit, or
     found it late.
   - At the step end they satisfy nonlinear algebraic equations only to the
     local error. OpenModelica computes algebraic variables from the states at
     every step; here an assert such as the MSL GasForce's `s_rel <= L + 1e-12`
     saw a piston 6.6e-9 past its stroke.

   So after each accepted step:
   1. The dense output of the algebraic unknowns at three points of the step
      is compared with the algebraic equations there: one Newton correction
      on the algebraic rows (differential unknowns kept) is the error of the
      interpolant there. Scaled by the tolerances like the solver's own error,
      the largest caps the next step as a step-size controller would. The
      accepted step stays; what follows is kept accurate (steps are not
      rejected, so an error within one step is not bounded, only the next
      steps are).
   2. The algebraic unknowns at the step end are projected onto their
      equations (Newton with the Jacobian of step 1). A Rodas step starts from
      the state alone (no FSAL derivative, no history), so this needs no
      restart.
   Other methods control the algebraic error themselves (BDF through its
   predictor) or keep state a changed step end would not match, so only Rodas
   methods use it. Its work arrays belong to one solve at a time: parallel
   solves of one build need their own callbacks (as the event callbacks do). =#

"""
    AlgebraicStepControl

The DiscreteCallback condition that does the work after each accepted step
(see the top of the file), with its work arrays. `isStepControl` tells it
from event callbacks.
"""
mutable struct AlgebraicStepControl
  rows::Vector{Int}          # the algebraic rows, which are also the algebraic unknowns
  active::Bool               # this solve: a Rodas method, adaptive, the mass-matrix form
  u::Any                     # a state inside the step or at its end
  f0::Any                    # f at u
  f1::Any                    # f with one algebraic unknown perturbed
  g::Any                     # f[rows]
  jac::Any                   # the full Jacobian when the problem has one (f.jac), else nothing
  J::Any                     # d f[rows] / d u[rows] at the step's midpoint
  lu::Any                    # its factorization, or nothing when singular
end

AlgebraicStepControl(rows::Vector{Int}) =
  AlgebraicStepControl(rows, false, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

"""
    isStepControl(callback) -> Bool

Whether a callback is the algebraic step control of a mass-matrix build
(`withAlgebraicStepControl`), not an event callback.
"""
isStepControl(@nospecialize(cb)) = cb isa DiffEqBase.DiscreteCallback && cb.condition isa AlgebraicStepControl

#= Exponent of the step-size rule: the dense output's error is taken to grow
   like dt^4. Limits of the factor on the next step, as a step-size
   controller's. Where in the step the dense output is checked (its error
   peaks near 0.8 of a Rodas5P step, not at the midpoint). The projection's
   Newton iterations (with the check's Jacobian, so they converge linearly)
   stop below this fraction of the tolerance, near a nonlinear solver's
   accuracy: a relation or assert at a threshold must not see the step's
   error; or where the correction is round-off. =#
const _ALGEBRAIC_ERROR_ORDER = 4
const _ALGEBRAIC_STEP_FACTORS = (0.2, 10.0)
const _ALGEBRAIC_CHECK_POINTS = (0.2, 0.5, 0.8)
const _PROJECTION_TOLERANCE = 1.0e-6
const _PROJECTION_ITERATIONS = 8

#= The algebraic rows of a diagonal mass matrix, or none (an ODE, a
   DAEFunction, a mass matrix that mixes rows). =#
function _algebraicRows(f)
  hasproperty(f, :mass_matrix) || return Int[]
  local mm = f.mass_matrix
  mm isa LinearAlgebra.UniformScaling && return Int[]
  LinearAlgebra.isdiag(mm) || return Int[]
  return Int[i for i in axes(mm, 1) if iszero(mm[i, i])]
end

#= A Rodas method (OrdinaryDiffEqRosenbrock's Rodas3P ... Rodas5Pr): one
   step, no FSAL derivative. By name: the Rosenbrock methods differ in FSAL. =#
_isRodas(alg) = startswith(string(nameof(typeof(alg))), "Rodas")

#= At the start of a solve: whether it applies (a Rodas method with adaptive
   steps on the mass-matrix form: a DAE solver gets the residual form), and
   work arrays of the state's element type (dual numbers under ForwardDiff). =#
function _startStepControl!(c::AlgebraicStepControl, integrator)
  local f = hasproperty(integrator, :f) ? integrator.f : nothing
  c.active = f !== nothing && hasproperty(integrator, :alg) && _isRodas(integrator.alg) &&
             integrator.opts.adaptive && _algebraicRows(f) == c.rows
  c.active || return nothing
  local u = integrator.u
  c.u = similar(u); c.f0 = similar(u); c.f1 = similar(u)
  c.g = similar(u, length(c.rows))
  c.J = similar(u, length(c.rows), length(c.rows))
  local jac = hasproperty(f, :jac) ? f.jac : nothing
  c.jac = jac === nothing ? nothing :
          (hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing ? similar(f.jac_prototype, eltype(u)) :
                                                                          similar(u, length(u), length(u)))
  return nothing
end

#= The tolerance-scaled RMS norm of a correction of the algebraic unknowns at u. =#
function _scaledNorm(c::AlgebraicStepControl, integrator, correction, u)
  local abstol = integrator.opts.abstol
  local reltol = integrator.opts.reltol
  local total = zero(real(eltype(u)))
  for (i, k) in enumerate(c.rows)
    local scale = (abstol isa Number ? abstol : abstol[k]) + (reltol isa Number ? reltol : reltol[k]) * abs(u[k])
    total += (correction[i] / scale)^2
  end
  return sqrt(total / length(c.rows))
end

#= f[rows] at (c.u, t) into c.g. =#
function _algebraicResidual!(c::AlgebraicStepControl, integrator, t)
  integrator.f(c.f0, c.u, integrator.p, t)
  for (i, r) in enumerate(c.rows)
    c.g[i] = c.f0[r]
  end
  return c.g
end

#= d f[rows] / d u[rows] at (c.u, t) into c.J: from the problem's Jacobian
   when it has one, else by forward differences; factorized into c.lu. =#
function _algebraicJacobian!(c::AlgebraicStepControl, integrator, t)
  local f = integrator.f
  local p = integrator.p
  if c.jac !== nothing
    f.jac(c.jac, c.u, p, t)
    for (j, k) in enumerate(c.rows), (i, r) in enumerate(c.rows)
      c.J[i, j] = c.jac[r, k]
    end
  else
    f(c.f0, c.u, p, t)
    for (j, k) in enumerate(c.rows)
      local uk = c.u[k]
      local h = sqrt(eps(Float64)) * max(1.0, abs(uk))
      c.u[k] = uk + h
      f(c.f1, c.u, p, t)
      c.u[k] = uk
      for (i, r) in enumerate(c.rows)
        c.J[i, j] = (c.f1[r] - c.f0[r]) / h
      end
    end
  end
  local lu = LinearAlgebra.lu(c.J; check = false)
  c.lu = LinearAlgebra.issuccess(lu) ? lu : nothing
  return c.lu
end

#= The largest error of the dense output of the algebraic unknowns at the
   check points of the last step, in the solver's error norm (1 is the
   tolerance). Leaves the factorized Jacobian (at the midpoint) in c.lu. =#
function _algebraicError!(c::AlgebraicStepControl, integrator)
  c.lu = nothing
  local t0 = integrator.tprev
  local dt = integrator.t - t0
  #= A step of no length (at a tstop) has no interpolation of its own. =#
  abs(dt) <= 8 * eps(max(abs(t0), 1.0)) && return 0.0
  integrator(c.u, t0 + dt / 2)
  _algebraicJacobian!(c, integrator, t0 + dt / 2) === nothing && return 0.0   # singular: nothing to judge
  local err = zero(real(eltype(c.u)))
  for θ in _ALGEBRAIC_CHECK_POINTS
    local tθ = t0 + θ * dt
    integrator(c.u, tθ)
    err = max(err, _scaledNorm(c, integrator, c.lu \ _algebraicResidual!(c, integrator, tθ), c.u))
  end
  return err
end

#= Newton on the algebraic rows at the step end, differential unknowns kept.
   Kept only if it converges; the state is left as it was otherwise. =#
function _projectStepEnd!(c::AlgebraicStepControl, integrator)
  c.lu === nothing && return nothing
  local u = integrator.u
  copyto!(c.u, u)
  for _ in 1:_PROJECTION_ITERATIONS
    local correction = try
      c.lu \ _algebraicResidual!(c, integrator, integrator.t)
    catch
      return nothing
    end
    local roundoff = true
    for (i, k) in enumerate(c.rows)
      c.u[k] -= correction[i]
      roundoff &= abs(correction[i]) <= 16 * eps(max(abs(c.u[k]), 1.0))
    end
    local step = _scaledNorm(c, integrator, correction, c.u)
    isfinite(step) || return nothing
    if step < _PROJECTION_TOLERANCE || roundoff
      for k in c.rows
        u[k] = c.u[k]
      end
      return nothing
    end
  end
  return nothing
end

#= The condition of the callback: the work, before the integrator saves the
   step (a DiscreteCallback's affect would run after the save, which would
   keep the unprojected step end). It never fires. Not after a continuous
   event: the event iteration solves the algebraic unknowns then. =#
function (c::AlgebraicStepControl)(u, t, integrator)
  (c.active && !_continuousEventFired(integrator)) || return false
  local dt = integrator.t - integrator.tprev
  local err = _algebraicError!(c, integrator)
  if isfinite(err) && err > 0
    local (lo, hi) = _ALGEBRAIC_STEP_FACTORS
    local factor = clamp(0.9 * err^(-1 / _ALGEBRAIC_ERROR_ORDER), lo, hi)
    local cap = abs(dt) * factor
    cap < abs(integrator.dtpropose) && ModelingToolkit.SciMLBase.set_proposed_dt!(integrator, sign(dt) * cap)
  end
  _projectStepEnd!(c, integrator)
  return false
end

"""
    withAlgebraicStepControl(callbacks, problem) -> callbacks

For a mass-matrix DAE, add a DiscreteCallback that after each accepted step
of a Rodas method keeps the algebraic unknowns accurate (see the top of the
file): their dense output caps the next step, and the step end is projected
onto the algebraic equations. It is emitted first among the model's discrete
callbacks, so it sees the step as the solver took it. An ODE is returned
unchanged; other methods and DAE solvers (the residual form) do not use it.
"""
function withAlgebraicStepControl(callbacks, problem)
  local rows = _algebraicRows(problem.f)
  isempty(rows) && return callbacks
  local c = AlgebraicStepControl(rows)
  local cb = DiffEqBase.DiscreteCallback(c, integrator -> nothing;
                                         initialize = (cb, u, t, integrator) -> _startStepControl!(c, integrator),
                                         save_positions = (false, false),
                                         initializealg = ModelingToolkit.SciMLBase.NoInit())
  return callbacks === nothing ? DiffEqBase.CallbackSet(cb) : DiffEqBase.CallbackSet(cb, callbacks)
end
