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

#= The asserts of the array models (ArrayPreserveDE) during a solve: one method for every model, over its parameters `p` and
   the model's code they hold (`p.code`, ArrayModelCode; its own `checkAsserts` evaluates them). =#

"""
    _assertFailed(p, k, instance, t, message) -> Bool

The assert `k` (`asserts`) of the instance `instance` (an algorithm's: `(k, column)`) violated at `t`: an error
(`ModelicaAssertionError`), a warning once per instance (`AssertionLevel.warning`). While `p.state.probe`
(`_violated!` looks for a violation without reporting it) only recorded.
"""
function _assertFailed(p::ArrayModelParameters, k::Int, instance::Union{Int, Tuple{Int, Int}}, t::Real,
                       message::AbstractString)::Bool
  local (condition::String, _, warning::Bool) = p.code.asserts[k]
  if p.state.probe
    (warning && instance in p.state.warned) || (p.state.violated = true)
    return true
  end
  warning || throw(ModelicaAssertionError(Float64(t), message, condition))
  instance in p.state.warned ||
    (push!(p.state.warned, instance); @warn string("Assertion violated at time ", t, ": ", message) condition)
  return true
end

"Whether an assert is violated at (u, t) (a warning reported already does not count)."
function _violated!(u::AbstractVector, p::ArrayModelParameters, t::Real)::Bool
  p.state.probe = true; p.state.violated = false
  try
    p.code.checkAsserts(u, p, t)
  finally
    p.state.probe = false
  end
  return p.state.violated
end

"""
    _assertStep(u, t, integrator) -> Bool

After a step: an assert violated at its end is reported at the first violating time in the step (bisection on the
step's interpolation), as the MTK path reports the crossing; within the step with its discrete state (the step start's:
an event that ended it changed them) and the state just before an event that ended it (a reinit moved it); else the
event's, at it. A violation at a crossing of the asserts' relations (`hasCrossingAsserts`) is decided halfway to the
next check, not at the step that ends at it (`_assertPending!`).
"""
function _assertStep(u::AbstractVector, t::Real, integrator::DiffEqBase.DEIntegrator)::Bool
  local p::ArrayModelParameters = integrator.p
  local c::ArrayModelCode = p.code
  try
    if c.hasCrossingAsserts
      _assertPending!(integrator)
      p.state.pending[1] == t && return false
    end
    _violated!(u, p, t) || return false
    local stepStart::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}} = p.state.stepStart
    local uLeft::AbstractVector = p.state.leftLimit[1] == t ? p.state.leftLimit[2] : u
    local lo::Float64 = integrator.tprev; local hi::Float64 = t
    _withDiscreteState(() -> _violated!(uLeft, p, t), p, stepStart) || return (c.checkAsserts(u, p, t); false)
    _withDiscreteState(p, stepStart) do
      if hi > lo
        for _ in 1:100
          local mid::Float64 = (lo + hi) / 2
          _violated!(integrator(mid), p, mid) ? (hi = mid) : (lo = mid)
          hi - lo <= 1.0e-12 * max(1.0, abs(hi)) && break
        end
      end
      c.checkAsserts(hi == t ? uLeft : integrator(hi), p, hi)
    end
    return false
  finally
    p.state.stepStart = _discreteState(p)
  end
end

"""
    _assertPending!(integrator)

A crossing of the asserts' relations (its time and state): decided at the next check, halfway to it, as the scalarized
path: a relation that touches its boundary (a piston at its top dead center) is past it by rounding there and holds
again; one that crosses fails halfway too, reported at the crossing, or after it where the relation is at its boundary
there (`x <= lim` at x = lim holds) at the first violation, by bisection on the step's interpolation. Within the step
its discrete state is the step start's (an event that ended it changed them).
"""
function _assertPending!(integrator::DiffEqBase.DEIntegrator)::Nothing
  local p::ArrayModelParameters = integrator.p
  local c::ArrayModelCode = p.code
  local (τp::Float64, up::Vector{Float64}) = p.state.pending
  (isnan(τp) || integrator.t <= τp) && return nothing
  p.state.pending = (NaN, Float64[])
  local τm::Float64 = (τp + integrator.t) / 2
  local within::Bool = integrator.tprev <= τm
  local decide::Function = () -> begin
    local (um::AbstractVector, tm::Float64) =
      within ? (integrator(τm), τm) : (copy(integrator.u), Float64(integrator.t))
    _violated!(um, p, tm) || return nothing
    _violated!(up, p, τp) && return (c.checkAsserts(up, p, τp); nothing)
    (within && integrator.tprev <= τp) || return (c.checkAsserts(um, p, tm); nothing)
    local lo::Float64 = τp; local hi::Float64 = tm
    for _ in 1:100
      local mid::Float64 = (lo + hi) / 2
      _violated!(integrator(mid), p, mid) ? (hi = mid) : (lo = mid)
      hi - lo <= 1.0e-12 * max(1.0, abs(hi)) && break
    end
    c.checkAsserts(integrator(hi), p, hi)
    return nothing
  end
  within ? _withDiscreteState(decide, p, p.state.stepStart) : decide()
  return nothing
end

"A crossing of the asserts' relations: pending (`_assertPending!`), decided at the next check."
function affectAssertCrossing!(integrator::DiffEqBase.DEIntegrator, ::AbstractVector{Int8})::Nothing
  _assertPending!(integrator)
  isnan(integrator.p.state.pending[1]) && (integrator.p.state.pending = (Float64(integrator.t), copy(integrator.u)))
  DiffEqBase.derivative_discontinuity!(integrator, false)
  return nothing
end

"At the end, a violation still pending is decided on the final state."
function _assertPendingAtEnd!(integrator::DiffEqBase.DEIntegrator)::Nothing
  local p::ArrayModelParameters = integrator.p
  local (τp::Float64, up::Vector{Float64}) = p.state.pending
  isnan(τp) && return nothing
  p.state.pending = (NaN, Float64[])
  _violated!(integrator.u, p, integrator.t) || return nothing
  _violated!(up, p, τp) ? p.code.checkAsserts(up, p, τp) : p.code.checkAsserts(integrator.u, p, integrator.t)
  return nothing
end

"The names the model's module binds to the functions of this file."
const _ASSERT_FUNCTIONS = (:_assertFailed, :_violated!, :_assertStep, :_assertPending!, :affectAssertCrossing!,
                           :_assertPendingAtEnd!)
