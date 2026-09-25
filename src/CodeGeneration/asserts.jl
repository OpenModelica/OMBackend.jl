#= Modelica asserts of equation sections and algorithms (outside when-clauses).

   Checked like OpenModelica does: after initialization and after every
   accepted step, not inside the right-hand side, where rejected trial steps
   may leave the region the assert guards. An assert that reads only
   parameters is checked once, after initialization. AssertionLevel.error
   stops the simulation with a ModelicaAssertionError; AssertionLevel.warning
   warns once each time the condition becomes false. =#

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
   the condition and message as functions of those values and the integrator. =#
struct ModelicaAssert{O <: NamedTuple, C, M}
  observed::O
  condition::C
  message::M
  warning::Bool
  timeVarying::Bool
  text::String
end

#= The asserts as one DiscreteCallback, added to `callbacks`. The variables are
   read through the problem's symbolic index (states, observed and eliminated
   variables alike); an assert on a variable the simulation does not keep is
   reported and left out. =#
function withAssertCallback(callbacks, problem, asserts::Vector)
  local checks = Tuple{ModelicaAssert, Any}[]
  local byName = _variablesByName(problem)
  for a in asserts
    local getter = try
      isempty(a.observed) ? nothing :
        ModelingToolkit.SymbolicIndexingInterface.getu(problem, [byName[n] for n in values(a.observed)])
    catch err
      @warn "[asserts] $(a.text) reads a variable the simulation does not keep; it is not checked" exception = err
      continue
    end
    push!(checks, (a, getter))
  end
  isempty(checks) && return callbacks
  local violated = falses(length(checks))
  local check! = function (integrator, all::Bool)
    for (i, (a, getter)) in enumerate(checks)
      (all || a.timeVarying) || continue
      if a.condition(_assertValues(a, getter, integrator), integrator)
        violated[i] = false
      elseif !(a.warning && violated[i])
        #= Where in the step it failed, and the message there. =#
        local τ = all ? integrator.t : _violationTime(a, getter, integrator)
        local src = τ == integrator.t ? integrator : _stateAt(integrator, τ)
        local msg = string(a.message(_assertValues(a, getter, src), (t = τ,)))
        a.warning || throw(ModelicaAssertionError(τ, msg, a.text))
        @warn "Assertion violated at time $(τ): $(msg)" condition = a.text
        violated[i] = true
      end
    end
    return false
  end
  local cb = DiffEqBase.DiscreteCallback((u, t, integrator) -> check!(integrator, false), integrator -> nothing;
                                         initialize = function (c, u, t, integrator)
                                           fill!(violated, false)
                                           check!(integrator, true)
                                           nothing
                                         end,
                                         save_positions = (false, false))
  return DiffEqBase.CallbackSet(callbacks, cb)
end

#= The problem's variables by name: states, observed (eliminated and alias
   variables) and parameters. =#
function _variablesByName(problem)
  local byName = Dict{Symbol, Any}()
  local sys = try
    problem.f.sys
  catch
    return byName
  end
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
  catch
    return integrator.t
  end
  return hi
end
