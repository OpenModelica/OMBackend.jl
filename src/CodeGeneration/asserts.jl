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

#= Modelica asserts of equation sections and algorithms (outside when-clauses).

   Checked like OpenModelica does: after initialization, after every accepted
   step and where a relation of the condition crosses (a relation is an event,
   MLS 3.8.4), not inside the right-hand side, where rejected trial steps may
   leave the region the assert guards. An assert that reads only parameters is
   checked once, after initialization. AssertionLevel.error stops the
   simulation with a ModelicaAssertionError; AssertionLevel.warning warns once
   each time the condition becomes false. =#

"""
    ModelicaAssertionError(time, message, condition)

Thrown when a Modelica `assert` of level `AssertionLevel.error` (the default)
is violated during a simulation.
"""
struct ModelicaAssertionError <: Exception
  time::Float64
  message::String
  condition::String
end

Base.showerror(io::IO, e::ModelicaAssertionError) =
  print(io, "ModelicaAssertionError: assertion violated at time ", e.time, ": ", e.message, "\n  condition: ", e.condition)

#= One generated assert: the variables it reads (local name => variable name),
   the condition and message as functions of those values and the integrator,
   and a crossing function per relation of the condition (`a - b` for `a < b`,
   1.0 where the condition does not reach it; a tuple). =#
struct ModelicaAssert{O <: NamedTuple, C, M, Z}
  observed::O
  condition::C
  message::M
  warning::Bool
  timeVarying::Bool
  text::String
  crossings::Z
end

#= The asserts as one DiscreteCallback, added to `callbacks`. The variables are
   read through the problem's symbolic index (states, observed and eliminated
   variables alike); an assert on a variable the simulation does not keep is
   reported and left out. =#
function withAssertCallback(callbacks, problem, asserts::Vector)
  local checks = Tuple{ModelicaAssert, Any}[]
  local byName = _variablesByName(problem)
  for a in asserts
    local absent = [n for n in values(a.observed) if !haskey(byName, n)]
    if !isempty(absent)
      @warn "[asserts] $(a.text) reads a variable the simulation does not keep; it is not checked" absent
      continue
    end
    local getter = isempty(a.observed) ? nothing :
      ModelingToolkit.SymbolicIndexingInterface.getu(problem, [byName[n] for n in values(a.observed)])
    push!(checks, (a, getter))
  end
  isempty(checks) && return callbacks
  local violated = falses(length(checks))
  local report = function (a, getter, integrator, τ)
    local src = τ == integrator.t ? integrator : _stateAt(integrator, τ)
    local msg = string(a.message(_assertValues(a, getter, src), (t = τ,)))
    a.warning || throw(ModelicaAssertionError(τ, msg, a.text))
    @warn "Assertion violated at time $(τ): $(msg)" condition = a.text
  end
  #= A failure just past a crossing, decided at the next check: a relation that
     touches its boundary (a piston at its top dead center, s_rel = L, MSL
     Engine1b_analytic) is past it by rounding there and holds again; one that
     crosses fails halfway to the next check too (reported at the crossing). =#
  local pending = fill(NaN, length(checks))
  #= At a step end (`all`: after the initialization, every assert) or at a
     crossing (the solver's state just past it). A failure at a step end whose
     relations are crossings came with the step's event: there. =#
  local checkOne! = function (integrator, i, all::Bool, atCrossing::Bool)
    local (a, getter) = checks[i]
    local τp = pending[i]
    if !isnan(τp) && integrator.t > τp
      pending[i] = NaN
      local τm = (τp + integrator.t) / 2
      local (src, at) = integrator.tprev <= τm ? (_stateAt(integrator, τm), (t = τm,)) : (integrator, integrator)
      if !a.condition(_assertValues(a, getter, src), at) && !(a.warning && violated[i])
        report(a, getter, integrator, τp)
        violated[i] = true
        return nothing
      end
    end
    if a.condition(_assertValues(a, getter, integrator), integrator)
      violated[i] = false
    elseif !all && (atCrossing || integrator.t == τp)
      pending[i] = integrator.t
    elseif !(a.warning && violated[i])
      local τ = all || !isempty(a.crossings) ? integrator.t : _violationTime(a, getter, integrator)
      report(a, getter, integrator, τ)
      violated[i] = true
    end
    nothing
  end
  local check! = function (integrator, all::Bool)
    for (i, (a, _)) in enumerate(checks)
      (all || a.timeVarying) && checkOne!(integrator, i, all, false)
    end
    return false
  end
  local cb = DiffEqBase.DiscreteCallback((u, t, integrator) -> check!(integrator, false), integrator -> nothing;
                                         initialize = function (c, u, t, integrator)
                                           fill!(violated, false)
                                           fill!(pending, NaN)
                                           check!(integrator, true)
                                           nothing
                                         end,
                                         #= A failure still pending at the end: decided on the final state. =#
                                         finalize = function (c, u, t, integrator)
                                           for (i, (a, getter)) in enumerate(checks)
                                             isnan(pending[i]) && continue
                                             a.condition(_assertValues(a, getter, integrator), integrator) ||
                                               report(a, getter, integrator, pending[i])
                                             pending[i] = NaN
                                           end
                                           nothing
                                         end,
                                         save_positions = (false, false))
  #= The relations of the time-varying asserts as crossings, without an
     effect on the state: a violation that started and ended within a step was
     missed (x in [0.4, 0.5) between the events at 0.2 and 0.5). The solver
     finds them on the step's own path, before an event changes the state. =#
  local crossing = Tuple{Int, Int}[(i, k) for (i, (a, _)) in enumerate(checks) if a.timeVarying for k in eachindex(a.crossings)]
  isempty(crossing) && return DiffEqBase.CallbackSet(callbacks, cb)
  local crossingCondition = function (out, u, t, integrator)
    local src = ModelingToolkit.SymbolicIndexingInterface.ProblemState(; u = u, p = integrator.p, t = t)
    local j = 0
    for (a, getter) in checks
      (a.timeVarying && !isempty(a.crossings)) || continue
      local values = _assertValues(a, getter, src)
      for f in a.crossings
        out[j += 1] = f(values, (t = t,))
      end
    end
    nothing
  end
  #= The crossing's index, or (other DiffEqBase versions) the directions of
     every crossing at this time, 0 for the others. =#
  local crossingAffect! = function (integrator, which)
    local fired = which isa Integer ? [which] : [j for (j, d) in enumerate(which) if d != 0]
    foreach(i -> checkOne!(integrator, i, false, true), unique(first(crossing[j]) for j in fired))
    DiffEqBase.u_modified!(integrator, false)
  end
  local zc = DiffEqBase.VectorContinuousCallback(crossingCondition, crossingAffect!, length(crossing);
                                                 rootfind = ModelingToolkit.SciMLBase.RightRootFind,
                                                 save_positions = (false, false))
  return DiffEqBase.CallbackSet(callbacks, zc, cb)
end

#= The problem's variables by name: states, observed (eliminated and alias
   variables) and parameters. =#
function _variablesByName(problem)
  local byName = Dict{Symbol, Any}()
  local sys = hasproperty(problem.f, :sys) ? problem.f.sys : nothing
  sys === nothing && return byName
  for v in Iterators.flatten((ModelingToolkit.parameters(sys), (eq.lhs for eq in ModelingToolkit.observed(sys)),
                              ModelingToolkit.unknowns(sys)))
    byName[Symbol(ModelingToolkit.getname(v))] = v
  end
  return byName
end

_assertValues(a::ModelicaAssert, getter, src) =
  getter === nothing ? NamedTuple() : NamedTuple{keys(a.observed)}(Tuple(getter(src)))

#= The solution at τ within the last step, as a value provider for the getters. =#
_stateAt(integrator, τ) =
  ModelingToolkit.SymbolicIndexingInterface.ProblemState(; u = integrator(τ), p = integrator.p, t = τ)

#= The first time in (tprev, t] at which the condition fails, by bisection on
   the step's interpolation: the check runs at step ends and a step can be long
   (der(x) = 1 is solved in one step). The step end if the interpolation does
   not bracket the change. =#
function _violationTime(a::ModelicaAssert, getter, integrator)
  local lo = integrator.tprev; local hi = integrator.t
  hi > lo || return hi
  local holds(τ) = a.condition(_assertValues(a, getter, _stateAt(integrator, τ)), (t = τ,))
  try
    holds(lo) || return hi
    for _ in 1:80
      local mid = (lo + hi) / 2
      (mid <= lo || mid >= hi) && break
      holds(mid) ? (lo = mid) : (hi = mid)
    end
  catch err
    #= The interpolation or the condition failed inside the step: its end. =#
    OMBackend._fallback(err, :assertViolationTime)
    return integrator.t
  end
  return hi
end
