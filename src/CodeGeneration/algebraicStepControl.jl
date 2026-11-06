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
   3. Before a step is accepted (the solve's `isoutofdomain`,
      AlgebraicDomainCheck): a step end where the algebraic equations do not
      evaluate to finite numbers is rejected, and the solver retries with a
      smaller step. The error estimate and step 2 cannot catch it: an
      unknown that ran off to 1e176 sets its own error scale, and no Newton
      recovers from there. Only that: a step end merely far from the
      equations stands, as one past an event whose re-solve fixes it.
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
  work::Any                  # this solve's AlgebraicWork (the state's element type), or nothing
  capDt::Float64             # the step and its error when the current run of caps began (0: no run)
  capErr::Float64
  floorErr::Float64          # an error that did not shrink with the step: errors up to twice it do not cap
  lastDt::Float64            # the last capped step and its error (0: the last step was not capped)
  lastErr::Float64
  model::Any                 # this solve's right-hand side and parameters (f, p), for AlgebraicDomainCheck (it
                             # gets only the state); not the integrator, which holds the solution: kept in the
                             # model module's problem, every model's last solution stayed (MSL Magnetic
                             # ThreePhaseTransformerWithRectifier: 3.3 GB after its run)
end

AlgebraicStepControl(rows::Vector{Int}) = AlgebraicStepControl(rows, false, nothing, 0.0, 0.0, 0.0, 0.0, 0.0, nothing)

#= A solve's work arrays, concretely typed: the work after a step runs behind
   one dynamic dispatch (_control!). The fields were `Any` (a dispatch per
   access), the factorization and each solve allocated, and a sparse
   Jacobian's algebraic block was read entry by entry with a search each
   (item 3 of the consolidation: 77 % of the MSL AIMC_DOL's solve). =#
struct AlgebraicWork{V<:AbstractVector, G<:AbstractVector, M<:AbstractMatrix, JC, JV}
  u::V                       # a state inside the step or at its end
  f0::V                      # f at u
  f1::V                      # f with one algebraic unknown perturbed
  g::G                       # f[rows]
  correction::G              # a Newton correction of the algebraic unknowns
  J::M                       # d f[rows] / d u[rows] at the step's midpoint, factorized in place
  jac::JC                    # the full Jacobian when the problem has one (f.jac), else nothing
  jacValues::JV              # its stored values (a sparse Jacobian's nonzeros, a dense one's elements)
  pick::Vector{Int}          # where each entry of J (column-major) is in jacValues; 0: not stored (zero)
end

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

"""
The algebraic rows of a diagonal mass matrix, or none (an ODE, a
DAEFunction, a mass matrix that mixes rows).
"""
function _algebraicRows(f)
  hasproperty(f, :mass_matrix) || return Int[]
  local mm = f.mass_matrix
  mm isa LinearAlgebra.UniformScaling && return Int[]
  LinearAlgebra.isdiag(mm) || return Int[]
  return Int[i for i in axes(mm, 1) if iszero(mm[i, i])]
end

"""
A Rodas method: a Rosenbrock method that starts each step from the state
alone, no FSAL derivative (OrdinaryDiffEqCore's `isfsal` trait: Rodas3P ...
Rodas6P; not Rosenbrock23, Rodas3 or Rodas3d, whose steps reuse the last
derivative).
"""
_isRodas(alg) = alg isa OrdinaryDiffEq.OrdinaryDiffEqCore.OrdinaryDiffEqRosenbrockAdaptiveAlgorithm &&
                !OrdinaryDiffEq.OrdinaryDiffEqCore.isfsal(alg)

const _SparseArrays = Symbolics.SparseArrays

"""
Where the entries of jac[rows, rows] are stored (column-major over J):
an index into a sparse Jacobian's nonzeros (0 where it stores none), or a
dense one's linear index.
"""
function _jacobianPick(jac, rows::Vector{Int})::Vector{Int}
  local pick = Vector{Int}(undef, length(rows)^2)
  local n = 0
  for k in rows, r in rows
    n += 1
    pick[n] = _storedIndex(jac, r, k)
  end
  return pick
end
_storedIndex(jac::AbstractMatrix, r::Int, k::Int) = LinearIndices(jac)[r, k]
function _storedIndex(jac::_SparseArrays.AbstractSparseMatrixCSC, r::Int, k::Int)
  local rv = _SparseArrays.rowvals(jac)
  for idx in _SparseArrays.nzrange(jac, k)
    rv[idx] == r && return idx
  end
  return 0
end
_storedValues(jac::_SparseArrays.AbstractSparseMatrixCSC) = _SparseArrays.nonzeros(jac)
_storedValues(jac::AbstractMatrix) = vec(jac)

"""
At the start of a solve: whether it applies (a Rodas method with adaptive
steps on the mass-matrix form: a DAE solver gets the residual form), and
work arrays of the state's element type (dual numbers under ForwardDiff).
"""
function _startStepControl!(c::AlgebraicStepControl, integrator)
  local f = hasproperty(integrator, :f) ? integrator.f : nothing
  c.active = f !== nothing && hasproperty(integrator, :alg) && _isRodas(integrator.alg) &&
             integrator.opts.adaptive && _algebraicRows(f) == c.rows
  c.active || return nothing
  c.capDt = 0.0; c.capErr = 0.0; c.floorErr = 0.0; c.lastDt = 0.0; c.lastErr = 0.0
  c.model = (integrator.f, integrator.p)
  local u = integrator.u
  local n = length(c.rows)
  local fjac = hasproperty(f, :jac) ? f.jac : nothing
  local jac = fjac === nothing ? nothing :
              (hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing ? similar(f.jac_prototype, eltype(u)) :
                                                                              similar(u, length(u), length(u)))
  c.work = AlgebraicWork(similar(u), similar(u), similar(u), similar(u, n), similar(u, n), similar(u, n, n),
                         jac, jac === nothing ? nothing : _storedValues(jac),
                         jac === nothing ? Int[] : _jacobianPick(jac, c.rows))
  return nothing
end

"""
The tolerance-scaled RMS norm of a correction of the algebraic unknowns at u.
"""
function _scaledNorm(rows::Vector{Int}, integrator, correction, u)
  local abstol = integrator.opts.abstol
  local reltol = integrator.opts.reltol
  local total = zero(real(eltype(u)))
  for (i, k) in enumerate(rows)
    local scale = (abstol isa Number ? abstol : abstol[k]) + (reltol isa Number ? reltol : reltol[k]) * abs(u[k])
    total += (correction[i] / scale)^2
  end
  return sqrt(total / length(rows))
end

"""
f[rows] at (w.u, t) into w.g.
"""
_algebraicResidual!(w::AlgebraicWork, rows::Vector{Int}, integrator, t) = _algebraicResidualOf!(w, rows, integrator.f, integrator.p, t)

"""
f[rows] at (w.u, t), with the right-hand side `f` and parameters `p`, into w.g.
"""
function _algebraicResidualOf!(w::AlgebraicWork, rows::Vector{Int}, f, p, t)
  f(w.f0, w.u, p, t)
  for (i, r) in enumerate(rows)
    w.g[i] = w.f0[r]
  end
  return w.g
end

"""
d f[rows] / d u[rows] at (w.u, t) into w.J: from the problem's Jacobian
when it has one, else by forward differences; factorized in place. The
factorization, or nothing when singular.
"""
function _algebraicJacobian!(w::AlgebraicWork, rows::Vector{Int}, integrator, t)
  local f = integrator.f
  local p = integrator.p
  if w.jac !== nothing
    f.jac(w.jac, w.u, p, t)
    local vals = w.jacValues
    for (n, idx) in enumerate(w.pick)
      w.J[n] = idx == 0 ? zero(eltype(w.J)) : vals[idx]
    end
  else
    f(w.f0, w.u, p, t)
    for (j, k) in enumerate(rows)
      local uk = w.u[k]
      local h = sqrt(eps(Float64)) * max(1.0, abs(uk))
      w.u[k] = uk + h
      f(w.f1, w.u, p, t)
      w.u[k] = uk
      for (i, r) in enumerate(rows)
        w.J[i, j] = (w.f1[r] - w.f0[r]) / h
      end
    end
  end
  local F = LinearAlgebra.lu!(w.J; check = false)
  return LinearAlgebra.issuccess(F) ? F : nothing
end

"""
The largest error of the dense output of the algebraic unknowns at the
check points of the last step, in the solver's error norm (1 is the
tolerance), and the factorized Jacobian at the midpoint (nothing when
singular or when the step has no length).
"""
function _algebraicError!(w::AlgebraicWork, rows::Vector{Int}, integrator)
  local t0 = integrator.tprev
  local dt = integrator.t - t0
  local zeroErr = zero(real(eltype(w.u)))
  #= A step of no length (at a tstop) has no interpolation of its own. =#
  abs(dt) <= 8 * eps(max(abs(t0), 1.0)) && return (zeroErr, nothing)
  integrator(w.u, t0 + dt / 2)
  local F = _algebraicJacobian!(w, rows, integrator, t0 + dt / 2)
  F === nothing && return (zeroErr, nothing)   # singular: nothing to judge
  local err = zeroErr
  for θ in _ALGEBRAIC_CHECK_POINTS
    local tθ = t0 + θ * dt
    integrator(w.u, tθ)
    LinearAlgebra.ldiv!(w.correction, F, _algebraicResidual!(w, rows, integrator, tθ))
    err = max(err, _scaledNorm(rows, integrator, w.correction, w.u))
  end
  return (err, F)
end

"""
Newton on the algebraic rows at the step end, differential unknowns kept.
Kept only if it converges; the state is left as it was otherwise.
"""
function _projectStepEnd!(w::AlgebraicWork, rows::Vector{Int}, F, integrator)
  F === nothing && return nothing
  local u = integrator.u
  copyto!(w.u, u)
  for _ in 1:_PROJECTION_ITERATIONS
    try
      LinearAlgebra.ldiv!(w.correction, F, _algebraicResidual!(w, rows, integrator, integrator.t))
    catch err
      #= The model's residual threw (a singular factorization is `nothing`): no projection. =#
      OMBackend._fallback(err, :algebraicProjection)
      return nothing
    end
    local correction = w.correction
    local roundoff = true
    for (i, k) in enumerate(rows)
      w.u[k] -= correction[i]
      roundoff &= abs(correction[i]) <= 16 * eps(max(abs(w.u[k]), 1.0))
    end
    local step = _scaledNorm(rows, integrator, correction, w.u)
    isfinite(step) || return nothing
    if step < _PROJECTION_TOLERANCE || roundoff
      for k in rows
        u[k] = w.u[k]
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
  _control!(c, c.work, integrator)
  return false
end

function _control!(c::AlgebraicStepControl, w::AlgebraicWork, integrator)
  local dt = integrator.t - integrator.tprev
  local (err, F) = _algebraicError!(w, c.rows, integrator)
  if isfinite(err) && err > 0
    #= An error floor: since a run of caps began the step has at least halved
       and the error has not (it falls like dt^4 where the step causes it). It
       does not come from the step (the MSL SMPM_Braking's dummy derivative
       row: 1.7e-5 at every step size), and capping on it would shrink the
       step until maxiters. =#
    c.capErr > 0 && abs(dt) <= 0.5 * c.capDt && err >= 0.5 * c.capErr && (c.floorErr = max(c.floorErr, err))
    #= Nor one that falls slower than dt^2 from the last capped step (MSL R134a2 with p and h its volumes'
       states: the derivative unknown der(p), ~1e7 and fast, fell like dt; capped from its first error of 2.3e5,
       the steps went to 1e-12 and stayed). =#
    c.lastDt > 0 && abs(dt) <= 0.5 * c.lastDt && err > c.lastErr * (abs(dt) / c.lastDt)^2 && (c.floorErr = max(c.floorErr, err))
    local capped = false
    if err > 2 * c.floorErr
      local (lo, hi) = _ALGEBRAIC_STEP_FACTORS
      local factor = clamp(0.9 * err^(-1 / _ALGEBRAIC_ERROR_ORDER), lo, hi)
      local cap = abs(dt) * factor
      if cap < abs(integrator.dtpropose)
        ModelingToolkit.SciMLBase.set_proposed_dt!(integrator, sign(dt) * cap)
        capped = true
      end
    end
    if !capped
      c.capDt = 0.0; c.capErr = 0.0
    elseif c.capErr == 0
      c.capDt = abs(dt); c.capErr = err
    end
    (c.lastDt, c.lastErr) = capped ? (abs(dt), err) : (0.0, 0.0)
  end
  _projectStepEnd!(w, c.rows, F, integrator)
  return nothing
end

"""
    AlgebraicDomainCheck(control)

The solve's `isoutofdomain` with the algebraic step control `control` (step 3
at the top of the file): whether the proposed step end `u` at `t` is rejected.
"""
struct AlgebraicDomainCheck
  control::AlgebraicStepControl
end

function (d::AlgebraicDomainCheck)(u, p, t)
  local c = d.control
  c.active || return false
  return _outOfDomain(c, c.work, c.model, u, t)
end

"""
Whether the algebraic equations at the step end u at t do not evaluate to finite numbers (or
throw): the step is rejected. MSL Spice3 Inverter: no capacitance at the output node, so its
voltage is algebraic, and at the start both MOSFETs sit at the edge of conduction (the residual
~ k (v^2 - D^2), its slope gmin alone, 2e-12). The first Rodas step's stage went to 1e4 V, the
exponentials to 1e176 and the residual to Inf, and the step was accepted. A step end that is only
far from the equations stands: a stricter check (its residual against the step's start values')
held MSL TwoMass before its time event at 0.7 s, where the step past the event leaves the old
branch's equations and the event's re-solve fixes the unknowns.
"""
function _outOfDomain(c::AlgebraicStepControl, w::AlgebraicWork, model::Tuple{Any, Any}, u, t)::Bool
  copyto!(w.u, u)
  local g = try
    _algebraicResidualOf!(w, c.rows, model[1], model[2], t)
  catch err
    #= The model's residual threw at the step end (a domain error of a run-off value). =#
    OMBackend._fallback(err, :algebraicDomain)
    return true
  end
  return !all(isfinite, g)
end

"""
The algebraic step control of `callbacks`, or nothing.
"""
function _stepControl(callbacks)
  local i = findfirst(isStepControl, _callbackList(callbacks))
  return i === nothing ? nothing : _callbackList(callbacks)[i].condition
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
