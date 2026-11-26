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

#= The event iteration and the event callbacks of the array models (ArrayPreserveDE): one method for every model, over its
   parameters `p` and the model's code they hold (`p.code`, ArrayModelCode). They were generated into each model's
   module, and compiled again with each (BouncingBall: 0.3 s of its first simulate). The module binds their names to
   them (`_eventFunctionNames`). =#

"""
    _settle!(du, a, u, p, t, fire, searched = false) -> Bool

One sweep at (u, t): passes with the same pre() until nothing changes (a fired body runs in each: with pre() fixed it
gives the same values). Returns whether they settled. Relations that keep changing (a mixed system that cycles: a
bridge's commutation): the combination of their values that gives itself back (`_searchRelations!`), then the sweeps
from it.
"""
function _settle!(du::AbstractVector, a::AbstractVector, u::AbstractVector, p::ArrayModelParameters, t::Real,
                  fire::Bool, searched::Bool = false)::Bool
  local c::ArrayModelCode = p.code
  p.upre .= u; p.ureinit .= u
  empty!(p.state.whenMemo)
  local rel0::Vector{Bool} = copy(p.rel); local d0::Vector{Float64} = copy(p.d)
  local flips::BitVector = falses(length(rel0))
  for pass::Int in 1:50
    local relBefore::Vector{Bool} = copy(p.rel); local dBefore::Vector{Float64} = copy(p.d)
    local rBefore::Vector{Float64} = copy(p.ureinit); local wBefore::Vector{Bool} = copy(p.wnew)
    local eBefore::Vector{Float64} = copy(p.evv)
    c.equations!(du, a, u, p, t, NO_CROSSINGS, true, fire)
    c.whens!(du, a, u, p, t, NO_CROSSINGS, true, fire)
    (relBefore == p.rel && dBefore == p.d && rBefore == p.ureinit && wBefore == p.wnew && eBefore == p.evv) &&
      return true
    pass > 25 && (flips .|= relBefore .!= p.rel)
    pass > 46 && _traceSettlePass(t, pass, (relBefore, dBefore, eBefore, wBefore, rBefore),
                                  (p.rel, p.d, p.evv, p.wnew, p.ureinit))
  end
  searched && return false
  _searchRelations!(c.equations!, c.whens!, du, a, u, p, t, fire, rel0, d0, flips, c.relationInstance,
                    c.relationSlot) || return false
  return _settle!(du, a, u, p, t, fire, true)
end

"""
    startSolution!(u, p, t, fire) -> Bool

The initialization's solution at (u, t): the relations, the when-conditions and the discrete variables the equations
compute, with pre() the start values (OpenModelica: pre(v) = v.start for the discrete variables). No when fires on its
condition there, the conditions' values become pre() of the conditions; with fire (initial() true) the bodies of the
whens initial() turns true run. reinit() has no effect.
"""
function startSolution!(u::AbstractVector, p::ArrayModelParameters, t::Real, fire::Bool)::Bool
  local du::AbstractVector = similar(u); local a::AbstractVector = similar(u, length(p.abuf))
  _settle!(du, a, u, p, t, fire) || return false
  p.ureinit .= u
  p.wcond .= p.wnew
  return true
end

"""
    eventIteration!(u, p, t, fire, sampleInstant = false; left = true) -> Bool

Event iteration at (u, t), in sweeps as OpenModelica's updateDiscreteSystem (and the MTK path, relationRefresh.jl):
within a sweep pre() keeps the values from before it, and the relations, equations and fired when bodies are evaluated
again until they settle (an equation reading a variable a when body just set: ch = change(k)); a sweep that changed a
discrete value starts the next one from the new values, until one changes nothing. At a sample instant the samples are
true for the first pass only (OpenModelica's updateDiscreteSystem: functionDAE once, the samples off, then iterated).
reinit() takes effect after the sweep (at the initialization, as OpenModelica, it has none).
"""
function eventIteration!(u::AbstractVector, p::ArrayModelParameters, t::Real, fire::Bool, sampleInstant::Bool = false;
                         left::Bool = true)::Bool
  local c::ArrayModelCode = p.code
  local du::AbstractVector = similar(u); local a::AbstractVector = similar(u, length(p.abuf))
  #= the algebraic variables' left limit: the equations with the relations and discrete values of before the event
     (`left`; a crossing computed it already, before its relations changed: `affectCrossing!`) =#
  if c.usesLeft && left
    c.equations!(du, a, u, p, t, NO_CROSSINGS, false)
    p.aleft .= a
  end
  for _ in 1:50
    p.dpre .= p.d
    _settle!(du, a, u, p, t, fire) || return false
    sampleInstant && fill!(p.sampleActive, false)
    local moved::Bool = !p.initPhase[1] && p.ureinit != u
    moved && (u .= p.ureinit)
    local switched::Bool = p.wnew != p.wcond
    p.wcond .= p.wnew
    (moved || switched || p.d != p.dpre) || return true
    #= pre() in the next sweep: this one's values =#
    c.usesLeft && (p.aleft .= a)
  end
  return false
end

"""
    _structuralChange!(p, k, value) -> Nothing

OMJL's structural change `k` (the module's STRUCTURAL_CHANGES) reached in a when body, with its new value `value` (a
recompilation's; 0 for an explicit transition): the run ends after the event (`_terminateReached!`), and the model's
modes go on from there (`simulateModes`). The first change of the event is the one taken.
"""
function _structuralChange!(p::ArrayModelParameters, k::Int, value::Real)::Nothing
  first(p.state.structural) == 0 && (p.state.structural = (k, Float64(value)))
  return nothing
end

"""
    _terminateReached!(integrator) -> Bool

After an event: a structural change reached (`_structuralChange!`) ends the mode's run (Success: the model goes on in
the next mode), a terminate() reached (its message, p.state.terminated) the run. Whether the run goes on.
"""
function _terminateReached!(integrator::DiffEqBase.DEIntegrator)::Bool
  if first(integrator.p.state.structural) != 0
    DiffEqBase.terminate!(integrator, DiffEqBase.SciMLBase.ReturnCode.Success)
    return false
  end
  local message::Union{Nothing, String} = integrator.p.state.terminated
  message === nothing && return true
  integrator.p.state.terminated = nothing
  @info "Modelica terminate() reached" message time = integrator.t
  DiffEqBase.terminate!(integrator)
  return false
end

"""
    _eventAt!(integrator, sampleInstant = false; left = true) -> Bool

The event iteration at an event during the solve: a model that does not settle (relations switching back and forth)
stops the simulation, as OpenModelica. After an event that changed the equations the next step starts small
(`_restartStep!`): a relation's crossing changed them; a sample instant did where a relation, a discrete variable or the
state changed. Whether the run goes on: not after a terminate() or a structural change (`_terminateReached!`), nor where
the event iteration did not settle.
"""
function _eventAt!(integrator::DiffEqBase.DEIntegrator, sampleInstant::Bool = false;
                  left::Bool = true)::Bool
  local p::ArrayModelParameters = integrator.p
  #= the delay() arguments just before the event: their left limits (a jump is recorded after it, `_delayJumps!`) =#
  p.code.nDelays == 0 || _delayLeft!(integrator)
  #= the state just before the event (a reinit moves it): the asserts' left limit =#
  p.state.leftLimit = (Float64(integrator.t), copy(integrator.u))
  local relBefore::Vector{Bool} = copy(p.rel); local dBefore::Vector{Float64} = copy(p.d)
  if eventIteration!(integrator.u, p, integrator.t, true, sampleInstant; left)
    (!sampleInstant || relBefore != p.rel || dBefore != p.d || p.state.leftLimit[2] != integrator.u) &&
      _restartStep!(integrator)
    return (p.code.hasTerminate || p.code.hasStructural) ? _terminateReached!(integrator) : true
  end
  @error string("[events] the event iteration did not settle at t = ", integrator.t,
                " (relations switching back and forth: a chattering model); the simulation stops")
  DiffEqBase.terminate!(integrator, DiffEqBase.SciMLBase.ReturnCode.Failure)
  return false
end

"""
    _terminalEvent!(integrator)

The terminal event (MLS 8.6: terminal() is true at the end of a successful analysis): an event iteration at the final
time with terminal() true, in which the when-branches with it among their triggers run (MSL's ContinuousSignalExtrema,
`when {u <= x, u >= x, terminal()}`; an FFT's `when terminal()`); the discrete values it gives logged at that time.
"""
function _terminalEvent!(integrator::DiffEqBase.DEIntegrator)::Nothing
  local p::ArrayModelParameters = integrator.p
  p.state.terminal = true
  try
    eventIteration!(integrator.u, p, integrator.t, true) ||
      @error string("[events] the terminal event did not settle at t = ", integrator.t)
    _logDiscretes!(p, integrator.t)
  finally
    p.state.terminal = false
  end
  return nothing
end

"The relations' and when-conditions' crossing functions at (u, t) into `out` (the relations' callback's condition)."
function conditions!(out::AbstractVector, u::AbstractVector, t::Real,
                     integrator::DiffEqBase.DEIntegrator)::Nothing
  local p::ArrayModelParameters = integrator.p
  local c::ArrayModelCode = p.code
  local du::AbstractVector = similar(u)
  c.crossingsOnly && (c.crossings!(du, p.abuf, u, p, t, CrossingOut(out), false); return nothing)
  local a::AbstractVector = similar(u, length(p.abuf))
  c.equations!(du, a, u, p, t, CrossingOut(out), false)
  c.whens!(du, a, u, p, t, CrossingOut(out), false, false)
  return nothing
end

"At the start of a solve: the relations' hysteresis for its tolerance and span."
function _startHysteresis!(::DiffEqBase.SciMLBase.DECallback, ::AbstractVector, ::Real,
                          integrator::DiffEqBase.DEIntegrator)::Nothing
  integrator.p.hyst[1] = _hysteresis(integrator)
  return nothing
end

"""
    logStepRoots!(integrator)

The loops' roots at the end of an accepted step, logged (_logRoots!). After an event at that time p holds the roots of
the event iteration's passes (the states before the reinit with the new discrete values): the right-hand side at the
state the next step starts from gives the roots the run goes on with.
"""
function logStepRoots!(integrator::DiffEqBase.DEIntegrator)::Nothing
  local p::ArrayModelParameters = integrator.p
  local t::Float64 = integrator.t
  if p.state.leftLimit[1] == t
    local u::AbstractVector = integrator.u
    p.code.equations!(similar(u), similar(u, length(p.abuf)), u, p, t, NO_CROSSINGS, false)
  end
  _logRoots!(p, t)
  return nothing
end

"The discrete values, the relations and the event-generating functions' values at `t`, logged where they changed."
function _logDiscretes!(p::ArrayModelParameters, t::Real)::Nothing
  local entry::Vector{Float64} = vcat(p.d, Float64.(p.rel), p.evv)
  (isempty(p.dlog) || p.dlog[end][2] != entry) && push!(p.dlog, (Float64(t), entry))
  return nothing
end

"""
    _atSampleInstant(p, t, tspan) -> Bool

Whether t is the instant of the sample tick still to come (`sampleStops[2]`). A relation's change there belongs to the
tick's event (`affectSample!` takes the relations literal at the instant), which follows at this time: handled first as
an event of its own, the tick read the values of after the change (a clocked `sample(u)` at the instant of u's step
takes u's left limit, MLS 16.5.1: MSL Clocked BooleanSignals.Sample1). At: within 1e-12 of the run's time scale (its
span, or the time where larger), not of a second: a pulse edge 1e-13 s after a tick is no part of the tick (MSL Spice3
Inverter, a run of 1e-11 s: its sources' edges were dropped). Not a tick that ran (`sampleStops[1]`): it took the
relations at its time, and a threshold an ulp after it is a crossing of its own (MSL SawTooth's
`time >= (pre(count) + 1)*period + startTime` at 1251 * 25e-6, an ulp after the tick of a sample of the same period:
dropped, the reset waited for the end of the step after it, 1e-7 s, and ChopperBuckBoost_DutyCycle's switching every
other period with it).
"""
function _atSampleInstant(p::ArrayModelParameters, t::Real, tspan::Tuple{Real, Real})::Bool
  p.code.nSamples > 0 || return false
  local stops::Vector{Float64} = p.state.sampleStops
  local span::Float64 = abs(tspan[2] - tspan[1])
  local w::Float64 = 1.0e-12 * (isfinite(span) ? max(span, abs(t)) : 1 + abs(t))
  return length(stops) >= 2 && abs(stops[2] - t) <= w
end

"""
    affectCrossing!(integrator, events)

A crossing: relations and when-conditions first take their values at the left limit (the state just before the event),
so a when fires exactly when its condition rises across the event; a relation whose crossing the root finder found
(`events`: +1 its crossing function rose past zero, -1 it fell) changes where that is its change (a true one's rises, a
false one's falls) and its value at the left limit did not change already; then the event iteration at the event.
Re-evaluated there, a hair off its bound, the relation kept its value where an algebraic loop solved inside the
equations gave its last digits from another guess, and the same crossing was found again in every step (MSL EngineV6's
piston speed at dead center). A time event (a relation of time and parameters crossed): the event iteration with the
relations literal at the instant (H = 0), as at a sample instant: one on its boundary there changes with it (a brake's
normal force ramped to 0 at the time event: f <= 0 holds). The left limit lies beyond the hysteresis band (2H): a
relation of time is located at its band's edge, zc = -eps with eps = H*(1 + |t|), and at H = 1e-10 (reltol 1e-6) a left
limit 1e-10*(1 + |t|) back put it exactly on its threshold, where the exact-zero rule made it true already: the when on
it took its rising edge before the event and never fired (MSL 4.1 Trapezoid's `time >= (pre(count) + 1)*period +
startTime`: OpAmps.Differentiator's source stayed in its first period).
"""
function affectCrossing!(integrator::DiffEqBase.DEIntegrator, events::AbstractVector{Int8})::Nothing
  local t::Float64 = integrator.t
  local p::ArrayModelParameters = integrator.p
  local c::ArrayModelCode = p.code
  if _atSampleInstant(p, t, integrator.sol.prob.tspan)
    DiffEqBase.derivative_discontinuity!(integrator, false)
    return nothing
  end
  #= the delay() arguments' left limits, before the relations take their new values =#
  c.nDelays == 0 || _delayLeft!(integrator)
  local tl::Float64 = max(integrator.tprev, t - max(1e-10, 2 * p.hyst[1]) * (1 + abs(t)))
  local ul::AbstractVector = tl < t ? integrator(tl) : copy(integrator.uprev)
  local du::AbstractVector = similar(ul); local a::AbstractVector = similar(ul, length(p.abuf))
  local before::Vector{Bool} = copy(p.rel)
  c.equations!(du, a, ul, p, tl, NO_CROSSINGS, true)
  c.whens!(du, a, ul, p, tl, NO_CROSSINGS, true, false)
  p.wcond .= p.wnew
  #= the algebraic variables' left limit at t, with the relations of before the crossing: pre(u) of
     u = if time < T then ... is u before the switch (MSL DemonstrateContinuousSignalExtrema's pulse edges) =#
  if c.usesLeft
    c.equations!(du, a, integrator.u, p, t, NO_CROSSINGS, false)
    p.aleft .= a
  end
  local timed::Bool = false
  for k::Int in eachindex(events)
    (events[k] != 0 && c.relationSlot[k] && p.rel[k] == before[k] && p.rel[k] == (events[k] > 0)) &&
      (p.rel[k] = !p.rel[k])
    timed |= events[k] != 0 && c.timeSlot[k]
  end
  local hyst::Float64 = p.hyst[1]
  timed && (p.hyst[1] = 0.0)
  try
    _eventAt!(integrator; left = false)
  finally
    p.hyst[1] = hyst
  end
  _logDiscretes!(p, t)
  DiffEqBase.derivative_discontinuity!(integrator, true)
  return nothing
end

"""
    affectSample!(integrator)

A sample instant: the samples due now are true for the first pass of the event iteration, false in the passes after it
(OpenModelica's): what reads sample() other than as a when's rising edge takes its fall at the instant
(`y = not u and pre(u)` of `u = sample(...)`, CDL FallingEdge: y rises and falls there; `k = if u then pre(k) + 1 else
pre(k)` counts once), and the when-conditions take that value too, so the next instant is a rising edge. The relations
literal at the instant (H = 0): one whose crossing function is 0 there (a threshold the tick reaches,
`time >= pulseStart + 0.2` at 1) changes with it, as OpenModelica takes that state event together with the time event.
"""
function affectSample!(integrator::DiffEqBase.DEIntegrator)::Nothing
  local t::Float64 = integrator.t; local p::ArrayModelParameters = integrator.p
  p.code.nDelays == 0 || _delayLeft!(integrator)
  local tspan::Tuple{Real, Real} = integrator.sol.prob.tspan
  local starts::Vector{Float64} = p.state.sampleStart
  local intervals::Vector{Float64} = p.state.sampleInterval
  local stops::Vector{Float64} = p.state.sampleStops
  local next::Float64 = _nextSampleStop(p, t, tspan)
  for id::Int in eachindex(starts)
    local k::Float64 = round((t - starts[id]) / intervals[id])
    local (ts::Float64, w::Float64) = _sampleTick(starts[id], intervals[id], k, tspan)
    #= active at the stop its tick joined (_nextSampleStop): at or after it within its round-off, before the next =#
    p.sampleActive[id] = k >= 0 && 0.0 <= ts - t <= w && ts < next
  end
  local hyst::Float64 = p.hyst[1]
  p.hyst[1] = 0.0
  local goesOn::Bool = false
  try
    goesOn = _eventAt!(integrator, true)
  finally
    p.hyst[1] = hyst
  end
  fill!(p.sampleActive, false)
  _logDiscretes!(p, t)
  DiffEqBase.derivative_discontinuity!(integrator, true)
  #= the stop after this one; none after a terminate() or a structural change at the instant (a stop added after
     terminate! went on to it) =#
  stops[1] = t
  stops[2] = next
  (goesOn && next < tspan[2]) && DiffEqBase.add_tstop!(integrator, next)
  return nothing
end

"""
The round-off of a tick st + k dt, relative to the largest magnitude in it (omc's SAMPLE_EPS 1e-14): ticks within it of
each other or of the start or stop time are one instant.
"""
const SAMPLE_EPS = 1.0e-14

"""
    _sampleTick(st, dt, k, tspan) -> (tick, roundoff)

Tick k of sample(st, dt) and its round-off w (SAMPLE_EPS of the largest magnitude in st + k dt): at the start or stop
time when within w of it (sample(-0.9, 0.3): 0).
"""
function _sampleTick(st::Float64, dt::Float64, k::Float64, tspan::Tuple{Real, Real})::Tuple{Float64, Float64}
  local tk::Float64 = st + k * dt
  local w::Float64 = SAMPLE_EPS * max(1.0, abs(st), abs(k * dt), abs(tk))
  local ts::Float64 = abs(tk - tspan[1]) <= w ? Float64(tspan[1]) : abs(tk - tspan[2]) <= w ? Float64(tspan[2]) : tk
  return (ts, w)
end

"""
    _nextSampleStop(p, after, tspan) -> Float64

The stop of p's samples after the stop `after` (NaN: the first one, at or after the start time); Inf where no tick is
left in tspan. Ticks within round-off of each other are one instant, a tick joining the stop before it (3 * 0.2 and
2 * 0.3; sample(-100, 0.1) and sample(0, 0.1) near 0): affectSample! activates each sample at the stop its tick joined.
One stop at a time, each from the samples' ticks next to the last: a table of the ticks over tspan took all memory where
the period is far below the span (MSL Spice3 Inverter: V_pulse's sample(TD, PER) of period 2e-12 s, its problem built
over 1 s for the start check, 5e11 ticks).
"""
function _nextSampleStop(p::ArrayModelParameters, after::Float64, tspan::Tuple{Real, Real})::Float64
  local next::Float64 = Inf
  local initial::Bool = isnan(after)
  local from::Float64 = initial ? Float64(tspan[1]) : after
  for (st::Float64, dt::Float64) in zip(p.state.sampleStart, p.state.sampleInterval)
    (isfinite(st) && isfinite(dt)) || error("array model: sample start and interval must be finite")
    dt > 0 || error("array model: sample interval must be positive")
    local k::Float64 = max(0.0, ceil((from - st) / dt) - 1)
    while true
      local (ts::Float64, w::Float64) = _sampleTick(st, dt, k, tspan)
      ts > tspan[2] && break
      if initial ? ts >= tspan[1] : ts - after > w
        next = min(next, ts)
        break
      end
      k + 1 > k || error(string("array model: sample interval ", dt, " below the round-off of the time ", from))
      k += 1
    end
  end
  return next
end

"""
    _sampleStart!(cb, u, t, integrator)

The samples' stops as a callback: the first stop at the start of a solve (a tick at the start time handled there; not
for a mode entered at a structural change, `ArrayRuntimeState.entered`: the mode before took that instant's ticks), then
each from the one before (affectSample!), a solver stop where within tspan.
"""
function _sampleStart!(::DiffEqBase.SciMLBase.DECallback, ::AbstractVector, t::Real,
                       integrator::DiffEqBase.DEIntegrator)::Nothing
  local stops::Vector{Float64} = integrator.p.state.sampleStops
  local tspan::Tuple{Real, Real} = integrator.sol.prob.tspan
  resize!(stops, 2)
  stops[1] = -Inf
  stops[2] = _nextSampleStop(integrator.p, integrator.p.state.entered ? Float64(tspan[1]) : NaN, tspan)
  if stops[2] == t
    affectSample!(integrator)
  elseif stops[2] < tspan[2]
    DiffEqBase.add_tstop!(integrator, stops[2])
  end
  return nothing
end

"Whether a sample stop is due at t (the samples' callback's condition)."
_sampleDue(::AbstractVector, t::Real, integrator::DiffEqBase.DEIntegrator)::Bool =
  t == integrator.p.state.sampleStops[2]

"""
    affect!(integrator)

An event the relations' check after a step found (`relationsChanged`): the event iteration, the discrete values logged.
"""
function affect!(integrator::DiffEqBase.DEIntegrator)::Nothing
  if _atSampleInstant(integrator.p, integrator.t, integrator.sol.prob.tspan)
    DiffEqBase.derivative_discontinuity!(integrator, false)
    return nothing
  end
  _eventAt!(integrator)
  _logDiscretes!(integrator.p, integrator.t)
  DiffEqBase.derivative_discontinuity!(integrator, true)
  return nothing
end

"""
    relationsChanged(u, t, integrator) -> Bool

After a step: whether a relation has another value than the stored one. A crossing right after an event (the ball
leaving the floor it was reinitialized just below) has no sign change the root finder sees; it is caught here, at the
end of the step. The discrete values are restored too: this evaluation runs the lifted clusters of discrete equations
(and computes the discrete unknowns) with the relations as they are now; left in place, pre() at the event that follows
was the value after it (`when sample() then bs := pre(b)` of `b = time >= 0.2` at the tick at 0.2).
"""
function relationsChanged(u::AbstractVector, t::Real, integrator::DiffEqBase.DEIntegrator)::Bool
  local p::ArrayModelParameters = integrator.p
  local c::ArrayModelCode = p.code
  local rel0::Vector{Bool} = copy(p.rel); local w0::Vector{Bool} = copy(p.wcond)
  local e0::Vector{Float64} = copy(p.evv)
  local d0::Vector{Float64} = copy(p.d)
  local du::AbstractVector = similar(u)
  if c.crossingsOnly
    c.crossings!(du, p.abuf, u, p, t, NO_CROSSINGS, true)
  else
    local a::AbstractVector = similar(u, length(p.abuf))
    c.equations!(du, a, u, p, t, NO_CROSSINGS, true)
    c.whens!(du, a, u, p, t, NO_CROSSINGS, true, false)
  end
  local changed::Bool = rel0 != p.rel || e0 != p.evv
  p.rel .= rel0; p.wcond .= w0; p.evv .= e0; p.d .= d0
  return changed
end

"The delay() arguments at the integrator's point as left limits: a step's end, the state before an event."
function _delayLeft!(integrator::DiffEqBase.DEIntegrator)::Bool
  integrator.p.code.delayNow!(integrator.u, integrator.p, integrator.t)
  recordDelayStepEnd!(integrator.p.state.delays, Float64(integrator.t))
  return false
end

"After the events of a step: an argument that changed at its end jumped (due again its delay time later)."
function _delayJumps!(integrator::DiffEqBase.DEIntegrator)::Bool
  integrator.p.code.delayNow!(integrator.u, integrator.p, integrator.t)
  recordDelayJumps!(() -> nothing, integrator.p.state.delays, Float64(integrator.t), integrator)
  return false
end

"At a solve's start, after the other callbacks' initializations: each history's first point."
function _delayStart!(integrator::DiffEqBase.DEIntegrator)::Nothing
  integrator.p.code.delayNow!(integrator.u, integrator.p, integrator.t)
  startDelayHistories!(integrator.p.state.delays, Float64(integrator.t))
  return nothing
end

"""
    affectDelay!(integrator)

A delayed jump is due (delays.jl): the event iteration with the delayed values as right limits, until the recorder
after the events (`_delayJumps!`).
"""
function affectDelay!(integrator::DiffEqBase.DEIntegrator)::Nothing
  integrator.p.state.delays.rightLimit = true
  _eventAt!(integrator)
  _logDiscretes!(integrator.p, integrator.t)
  DiffEqBase.derivative_discontinuity!(integrator, true)
  return nothing
end

"""
The names the model's module binds to the functions of this file: its own code and the callbacks it builds call them
by these names.
"""
const _EVENT_FUNCTIONS = (:_settle!, :startSolution!, :eventIteration!, :_terminateReached!, :_eventAt!,
                          :_terminalEvent!, :conditions!, :_startHysteresis!, :logStepRoots!, :_logDiscretes!,
                          :_atSampleInstant, :affectCrossing!, :affectSample!, :_sampleTick, :_nextSampleStop,
                          :_sampleStart!, :_sampleDue, :affect!, :relationsChanged, :_delayLeft!, :_delayJumps!,
                          :_delayStart!, :affectDelay!)
