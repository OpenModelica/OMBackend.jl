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

A jump caused inside a continuous callback's affect: its left limit is recorded where the callback moves
the integrator to the event (the first re-solve of the instant, before the affect). Not handled: for a
variable delay time, the delayed jump's event is at the jump's time plus the delay time at the jump.
=#

"""
A delay() call as OMFrontend passes it on: OpenModelica.Internal.delay2/delay3 (or a bare delay).
"""
_isDelayCall(e::SC.Exp)::Bool = e isa SC.CALL && e.name in (:OpenModelica_Internal_delay2, :OpenModelica_Internal_delay3, :delay)

mutable struct DelayHistory
  t::Vector{Float64}
  v::Vector{Float64}
  due::Vector{Float64}        # the times of delayed jumps to come: a jump's time + delayTime
end
DelayHistory() = DelayHistory(Float64[], Float64[], Float64[])

"""
    DelayState(n)

A solve's delay histories of one model (a structural mode's own), one per delay() call (the array code: per
instance of a call), whether they are read as right limits (during a delayed jump's event), and the present values
of the calls' arguments and delay times, which the recorders fill before they record (`now`, `times`).
"""
mutable struct DelayState
  histories::Vector{DelayHistory}
  rightLimit::Bool
  now::Vector{Float64}
  times::Vector{Float64}
end
DelayState(n::Integer) = DelayState([DelayHistory() for _ in 1:n], false, zeros(n), zeros(n))

"The key of the delay states in a task's local storage."
const DELAY_STATE_KEY = :omjl_delay_states

"""
    delayStates()

The delay states of the solves of the current task, by model name. The terms and the callbacks
name a model's state rather than hold it: a build's parameters are deep-copied (iMTKGen's
pristine parameters), and a copied history is one the recorders never fill. Per task: an
integrator evaluates its right-hand side and runs its callbacks in the task that solves, so two
solves of one model at once (threads, an ensemble's trajectories) each have their own, and one
solve's start does not empty another's.
"""
delayStates()::Dict{Symbol, DelayState} =
  get!(() -> Dict{Symbol, DelayState}(), task_local_storage(), DELAY_STATE_KEY)::Dict{Symbol, DelayState}

"The delay state of `model` in the current task; `nothing` before its solve's start."
delayState(model::Symbol)::Union{Nothing, DelayState} = get(delayStates(), model, nothing)

"A new delay state of `model` with `n` empty histories, the current task's."
delayHistories!(model::Symbol, n::Integer)::DelayState = delayStates()[model] = DelayState(n)

"""
    clearDelayHistories!()

The current task's delay states dropped: before a solve, so the initialization reads the arguments
themselves, not the previous solve's start values (a history is created at the solve's start).
"""
function clearDelayHistories!()::Nothing
  empty!(delayStates())
  return nothing
end

"""
The value at s = t - delayTime, as a left limit or (`right`) a right limit on a recorded jump; `x` is
the argument's value at the present time t.
"""
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

#= The histories' use and upkeep, shared by the scalarized path's callbacks (below) and the array code's
   (ArrayODEGeneration.jl): each reads and fills a `DelayState`; they differ in where the state is kept and in how
   the present values of the arguments are computed. =#

"""
    delayedValue(s, k, t, delayTime, x) -> Float64

`delay(x, delayTime)` at the time `t` from the history `k` of `s`: the argument's value `delayTime` ago
(`_delayedValue`); `x`, its present value, before the history has a point (the initialization).
"""
function delayedValue(s::DelayState, k::Int, t::Float64, delayTime::Float64, x::Float64)::Float64
  (k > length(s.histories) || isempty(s.histories[k].t)) && return x
  return _delayedValue(s.histories[k], t - delayTime, s.rightLimit, t, x)
end

"The histories of `s` emptied: before a solve's start, so its initialization reads the arguments themselves."
function emptyDelayHistories!(s::DelayState)::Nothing
  for h in s.histories
    empty!(h.t); empty!(h.v); empty!(h.due)
  end
  s.rightLimit = false
  return nothing
end

"The first point of each history of `s`: the present values (`s.now`) at the start `t`."
function startDelayHistories!(s::DelayState, t::Float64)::Nothing
  emptyDelayHistories!(s)
  for (k, h) in enumerate(s.histories)
    push!(h.t, t); push!(h.v, s.now[k])
  end
  return nothing
end

"A step's end (or an event's left limit) at `t`: the present values (`s.now`), where `t` is past the last point."
function recordDelayStepEnd!(s::DelayState, t::Float64)::Nothing
  for (k, h) in enumerate(s.histories)
    (!isempty(h.t) && t > h.t[end]) && (push!(h.t, t); push!(h.v, s.now[k]))
  end
  return nothing
end

"""
    recordDelayJumps!(times!, s, t, integrator)

After the events at `t`: a present value (`s.now`) that differs from the last point at `t` is a jump, recorded as the
right limit and due again its delay time later (`s.times`, filled by `times!()` when the first jump is found): a
stop time of the solve within its time span (`delayJumpsDue!`). Round-off from an event's algebraic re-solve is no
jump. The histories are read as left limits again.
"""
function recordDelayJumps!(times!::F, s::DelayState, t::Float64, integrator)::Nothing where {F}
  local filled::Bool = false
  for (k, h) in enumerate(s.histories)
    isempty(h.t) && continue
    local x::Float64 = s.now[k]
    (t == h.t[end] && abs(x - h.v[end]) <= 1e-6 * max(1.0, abs(x))) && continue
    push!(h.t, t); push!(h.v, x)
    filled || (times!(); filled = true)
    local tDue::Float64 = t + s.times[k]
    #= The solve's end, not the build's: a cached build is made for a short span and solved for the
       model's (MSL Digital FullAdder: jumps due after the build's end were dropped, its gates fell
       back to 'U'). =#
    (tDue > t && integrator.tdir * (last(integrator.sol.prob.tspan) - tDue) >= 0) || continue
    push!(h.due, tDue)
    ModelingToolkit.SciMLBase.add_tstop!(integrator, tDue)
  end
  s.rightLimit = false
  return nothing
end

"Whether a delayed jump of `s` is due at `t`; the due times at or before `t` are dropped."
function delayJumpsDue!(s::DelayState, t::Float64)::Bool
  local tol::Float64 = 1e-12 * max(1.0, abs(t))
  local hit::Bool = false
  for h in s.histories, k in length(h.due):-1:1
    local d::Float64 = h.due[k]
    abs(d - t) <= tol && (hit = true)
    d <= t + tol && deleteat!(h.due, k)
  end
  return hit
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
  local s = delayState(f.model)
  (s === nothing || f.k > length(s.histories) || isempty(s.histories[f.k].t)) && return x
  return delayedValue(s, f.k, Float64(_primalValue(t)), Float64(_primalValue(delayTime)), Float64(_primalValue(x)))
end

"""
    delayLookup(model, k, t, delayTime, x)

The term for `delay(x, delayTime)`: `x`'s value `delayTime` before `t`, from `model`'s history `k`.
"""
delayLookup(model::Symbol, k::Integer, t, delayTime, x) =
  hasSymbolicArgs(t, delayTime, x) ? makeSymbolicTerm(DelayedValueFn(model, k), Any[t, delayTime, x]) :
                                     DelayedValueFn(model, k)(t, delayTime, x)

"""
The delayed jumps' time events: a discrete callback at each due time, among the other callbacks, so
the event iteration (withRelationRefresh) runs after it; from its affect until the recorder after the
event, the delayed values are right limits. Due times already passed are dropped.
"""
function withDelayEvents(callbacks, model::Symbol)
  local due = function (u, t, integrator)
    local s = delayState(model)
    return s !== nothing && delayJumpsDue!(s, Float64(t))
  end
  local affect! = function (integrator)
    local s = delayState(model)
    s === nothing || (s.rightLimit = true)
    _derivativeDiscontinuity!(integrator, true)
    return nothing
  end
  local cb = DiffEqBase.DiscreteCallback(due, affect!; save_positions = (false, false))
  return _joinCallbacks(callbacks, cb)
end

"""
    withDelayRecords(callbacks, problem, model, args, delayTimes) -> callbacks

`model`'s histories of its delay() arguments `args` (symbolic, in `problem`'s system): created at the
start of each solve (in its task: `delayStates`), then filled by two callbacks that only read (their conditions do the work and never fire, so no step is a
discontinuity): the first records the step end as the solver took it, the last one the values after the
event callbacks. A value that changed at an event is a jump: recorded as the right limit, and due again
`delayTime` later within the time span (a time event, withDelayEvents).
"""
function withDelayRecords(callbacks, problem, model::Symbol, args::Vector, delayTimes::Vector)
  isempty(args) && return callbacks
  local n = length(args)
  local sys = problem.f.sys
  local argumentValues = ModelingToolkit.build_explicit_observed_function(sys, args)
  local delayTimeValues = ModelingToolkit.build_explicit_observed_function(sys, delayTimes)
  local now(integrator) = Base.invokelatest(argumentValues, integrator.u, integrator.p, integrator.t)
  #= The start values once every other callback's initialization (the clusters' start bodies) has run:
     the last callback's. =#
  local stepEnd = function (u, t, integrator)
    local s = delayState(model)
    s === nothing && return false
    s.now .= now(integrator)
    recordDelayStepEnd!(s, Float64(t))
    return false
  end
  local start! = function (c, u, t, integrator)
    local s = delayHistories!(model, n)
    s.now .= Base.invokelatest(argumentValues, u, integrator.p, t)
    startDelayHistories!(s, Float64(t))
    #= The left limits at a branch event, before its affect (eventInstant.jl beginInstant!). =#
    atInstantStart!(integrator, i -> (stepEnd(i.u, i.t, i); nothing))
    return nothing
  end
  local afterEvents = function (u, t, integrator)
    local s = delayState(model)
    s === nothing && return false
    s.now .= now(integrator)
    recordDelayJumps!(() -> (s.times .= Base.invokelatest(delayTimeValues, u, integrator.p, t); nothing), s, Float64(t),
                      integrator)
    return false
  end
  local never! = integrator -> nothing
  local cbStepEnd = DiffEqBase.DiscreteCallback(stepEnd, never!; save_positions = (false, false))
  local cbAfterEvents = DiffEqBase.DiscreteCallback(afterEvents, never!; initialize = start!,
                                                    save_positions = (false, false))
  return _joinCallbacks(cbStepEnd, callbacks, cbAfterEvents)
end
