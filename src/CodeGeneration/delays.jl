#=
delay(expr, delayTime[, delayMax]) (MLS 3.7.4.1): the value of expr delayTime ago; up to the start plus
delayTime, its value at the start.

Each delay() call of a model keeps a history of its argument (DelayHistory): its value at every step
end, and a second point at the same time where an event changed it (the right limit). The equations
read the history through an opaque term (DelayedValueFn): linear between the points; within the
current step, towards the argument's present value (so delay(x, 0) is x); the argument itself before
the history has a point (the initialization). A jump of the argument is a jump of the delayed value
delayTime later: a time event there (withDelayEvents), which runs the event iteration where the model
has one. Only during that event are the delayed values read as right limits (the value after the
jump); otherwise as left limits, so a delay of a delayed value records the jump as a jump again.

Not handled: a jump caused inside a continuous callback's affect is recorded after it (without its
left limit); for a variable delay time, the delayed jump's event is at the jump's time plus the delay
time at the jump.
=#

#= A delay() call as OMFrontend passes it on: OpenModelica.Internal.delay2/delay3 (or a bare delay). =#
_isDelayCall(@nospecialize(path)) = string(path) in ("OpenModelica.Internal.delay2", "OpenModelica.Internal.delay3", "delay")

mutable struct DelayHistory
  t::Vector{Float64}
  v::Vector{Float64}
  due::Vector{Float64}        # the times of delayed jumps to come: a jump's time + delayTime
end
DelayHistory() = DelayHistory(Float64[], Float64[], Float64[])

#= A model's histories, by the model's name (a structural mode's own), and whether they are read as
   right limits (during a delayed jump's event). The terms and the callbacks name them rather than hold
   them: a build's parameters are deep-copied (iMTKGen's pristine parameters), and a copied history is
   one the recorders never fill. =#
const DELAY_HISTORIES = Dict{Symbol, Vector{DelayHistory}}()
const DELAY_RIGHT_LIMIT = Dict{Symbol, Bool}()

function delayHistories!(model::Symbol, n::Integer)
  DELAY_HISTORIES[model] = [DelayHistory() for _ in 1:n]
  DELAY_RIGHT_LIMIT[model] = false
  return nothing
end

"""
    clearDelayHistories!()

Empty every model's delay histories: before a solve, so the initialization reads the arguments
themselves, not the previous solve's start values.
"""
function clearDelayHistories!()
  for hs in values(DELAY_HISTORIES), h in hs
    empty!(h.t); empty!(h.v); empty!(h.due)
  end
  foreach(m -> DELAY_RIGHT_LIMIT[m] = false, collect(keys(DELAY_RIGHT_LIMIT)))
  return nothing
end

#= The value at s = t - delayTime, as a left limit or (`right`) a right limit on a recorded jump; `x` is
   the argument's value at the present time t. =#
function _delayedValue(h::DelayHistory, s::Float64, right::Bool, t::Float64, x::Float64)
  local n = length(h.t)
  s <= h.t[1] && return h.v[1]
  local tol = 1e-12 * max(1.0, abs(s))
  #= Past the last point: within the current step, towards the present value. Not on the last point
     itself: a jump recorded there (a delay of a delayed value, read by its recorder) has its limits. =#
  if s > h.t[n] + tol
    t <= h.t[n] && return h.v[n]
    return h.v[n] + (s - h.t[n]) / (t - h.t[n]) * (x - h.v[n])
  end
  #= The last point at or before s: on a jump (a repeated time) its last point for the right limit, the
     point before it for the left one, up to round-off. =#
  local k = clamp(searchsortedlast(h.t, right ? s + tol : s - tol), 1, n)
  k == n && return h.v[n]
  local t1 = h.t[k]
  local t2 = h.t[k + 1]
  return h.v[k] + (s - t1) / (t2 - t1) * (h.v[k + 1] - h.v[k])
end

#= Named functor (not a closure): ModelingToolkit rebuilds terms and re-infers their type, as for
   ConstTableLookupFn. Arguments: time, the delay time, the argument's present value. =#
struct DelayedValueFn
  model::Symbol
  k::Int
end
SymbolicUtils.promote_symtype(::DelayedValueFn, args...) = Real
SymbolicUtils.promote_shape(::DelayedValueFn, szs::SymbolicUtils.ShapeT...) = SymbolicUtils.ShapeVecT()

function (f::DelayedValueFn)(t, delayTime, x)
  local hs = get(DELAY_HISTORIES, f.model, nothing)
  (hs === nothing || f.k > length(hs) || isempty(hs[f.k].t)) && return x
  local tt = Float64(_primalValue(t))
  return _delayedValue(hs[f.k], tt - Float64(_primalValue(delayTime)), get(DELAY_RIGHT_LIMIT, f.model, false),
                       tt, Float64(_primalValue(x)))
end

"""
    delayLookup(model, k, t, delayTime, x)

The term for `delay(x, delayTime)`: `x`'s value `delayTime` before `t`, from `model`'s history `k`.
"""
delayLookup(model::Symbol, k::Integer, t, delayTime, x) =
  hasSymbolicArgs(t, delayTime, x) ? makeSymbolicTerm(DelayedValueFn(model, k), Any[t, delayTime, x]) :
                                     DelayedValueFn(model, k)(t, delayTime, x)

#= The delayed jumps' time events: a discrete callback at each due time, among the other callbacks, so
   the event iteration (withRelationRefresh) runs after it; from its affect until the recorder after the
   event, the delayed values are right limits. Due times already passed are dropped. =#
function withDelayEvents(callbacks, model::Symbol)
  local due = function (u, t, integrator)
    local tol = 1e-12 * max(1.0, abs(t))
    local hit = false
    for h in get(DELAY_HISTORIES, model, DelayHistory[]), k in length(h.due):-1:1
      local d = h.due[k]
      abs(d - t) <= tol && (hit = true)
      d <= t + tol && deleteat!(h.due, k)
    end
    return hit
  end
  local affect! = function (integrator)
    DELAY_RIGHT_LIMIT[model] = true
    _derivativeDiscontinuity!(integrator, true)
    return nothing
  end
  local cb = DiffEqBase.DiscreteCallback(due, affect!; save_positions = (false, false))
  return callbacks === nothing ? DiffEqBase.CallbackSet(cb) : DiffEqBase.CallbackSet(callbacks, cb)
end

"""
    withDelayRecords(callbacks, problem, model, args, delayTimes) -> callbacks

`model`'s histories of its delay() arguments `args` (symbolic, in `problem`'s system): created here, then
filled by two callbacks that only read (their conditions do the work and never fire, so no step is a
discontinuity): the first records the step end as the solver took it, the last one the values after the
event callbacks. A value that changed at an event is a jump: recorded as the right limit, and due again
`delayTime` later within the time span (a time event, withDelayEvents).
"""
function withDelayRecords(callbacks, problem, model::Symbol, args::Vector, delayTimes::Vector)
  delayHistories!(model, length(args))
  isempty(args) && return callbacks
  local sys = problem.f.sys
  local argumentValues = ModelingToolkit.build_explicit_observed_function(sys, args)
  local delayTimeValues = ModelingToolkit.build_explicit_observed_function(sys, delayTimes)
  local now(integrator) = Base.invokelatest(argumentValues, integrator.u, integrator.p, integrator.t)
  #= The start values once every other callback's initialization (the clusters' start bodies) has run:
     the last callback's. =#
  local start! = function (c, u, t, integrator)
    local v = Base.invokelatest(argumentValues, u, integrator.p, t)
    for (k, h) in enumerate(DELAY_HISTORIES[model])
      empty!(h.t); empty!(h.v); empty!(h.due)
      push!(h.t, t); push!(h.v, Float64(v[k]))
    end
    DELAY_RIGHT_LIMIT[model] = false
    return nothing
  end
  local stepEnd = function (u, t, integrator)
    local v = now(integrator)
    for (k, h) in enumerate(DELAY_HISTORIES[model])
      t > h.t[end] && (push!(h.t, t); push!(h.v, Float64(v[k])))
    end
    return false
  end
  local afterEvents = function (u, t, integrator)
    local v = now(integrator)
    local T = nothing
    for (k, h) in enumerate(DELAY_HISTORIES[model])
      local x = Float64(v[k])
      #= Round-off from an event's algebraic re-solve is no jump. =#
      (t == h.t[end] && abs(x - h.v[end]) <= 1e-6 * max(1.0, abs(x))) && continue
      push!(h.t, t); push!(h.v, x)
      T === nothing && (T = Base.invokelatest(delayTimeValues, u, integrator.p, t))
      local tDue = t + Float64(T[k])
      #= The solve's end, not the build's: a cached build is made for a short span and solved for the
         model's (MSL Digital FullAdder: jumps due after the build's end were dropped, its gates fell
         back to 'U'). =#
      (tDue > t && integrator.tdir * (last(integrator.sol.prob.tspan) - tDue) >= 0) || continue
      push!(h.due, tDue)
      ModelingToolkit.SciMLBase.add_tstop!(integrator, tDue)
    end
    DELAY_RIGHT_LIMIT[model] = false
    return false
  end
  local never! = integrator -> nothing
  local cbStepEnd = DiffEqBase.DiscreteCallback(stepEnd, never!; save_positions = (false, false))
  local cbAfterEvents = DiffEqBase.DiscreteCallback(afterEvents, never!; initialize = start!,
                                                    save_positions = (false, false))
  return callbacks === nothing ? DiffEqBase.CallbackSet(cbStepEnd, cbAfterEvents) :
                                 DiffEqBase.CallbackSet(cbStepEnd, callbacks, cbAfterEvents)
end
