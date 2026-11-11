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

#= The event instant: the state before it, the reinit()s at it, and where it
   begins (`beginInstant!`).

   pre() at an event instant (MLS 3.7.5, 8.6): the values before it. Every
   affect at that time and the first sweep of the event iteration
   (relationRefresh.jl) read them:
   OpenModelica's first iteration evaluates its sorted equations with one
   pre(), so a when that another when's change triggers at the same instant
   fires in it with the values before the instant (`when sample(0.5, 1) then
   n = pre(n) + 1` and `when n > 0 then m = pre(n) * 10`: m = 0; each affect's
   own snapshot gave 10). A later sweep is a later iteration: pre() is the
   state before it. Per integrator, recorded by the first affect at a new time
   and dropped when the iteration settles. A reinit() takes effect after the
   first iteration (OpenModelica, needToIterate): in the first sweep the
   relations see the reinitialized states at their values before the instant,
   and a when on such a relation fires in the next sweep, where pre() is the
   state after the reinit (`when sample(0.5, 1) then reinit(x, 0)` and `when
   x < 0.1 then y = pre(x)`: y = 0); a when on a discrete the same body sets
   fires in the first, with pre() before the instant (y = pre(x) = 1.5). =#
mutable struct _InstantPre
  t::Float64
  u::Any
  reinitialized::Vector{Int}   # the states a reinit() wrote at this instant
  held::Vector{Int}            # the held discretes among the states (`holdDiscretes!`)
  moving::Bool                 # the integrator is being moved to an event's time (`HeldBack`)
end

_InstantPre() = _InstantPre(NaN, nothing, Int[], Int[], false)

const _INSTANT_PRE = WeakKeyDict{Any, _InstantPre}()

"""
    holdDiscretes!(integrator, held)

The states `held` are held discretes: unknowns with der(v) = 0, which only an
affect writes. Registered at the start of a solve (`withHeldDiscretes`).
"""
function holdDiscretes!(integrator, held::Vector{Int})::Nothing
  (ismutable(integrator) && hasproperty(integrator, :uprev) && integrator.u isa Vector{Float64}) || return nothing
  local h::_InstantPre = get!(_InstantPre, _INSTANT_PRE, integrator)
  h.held = held
  h.moving = false
  return nothing
end

"""
The record of `integrator` if it has held discretes, or nothing.
"""
function _heldRecord(integrator)::Union{Nothing, _InstantPre}
  ismutable(integrator) || return nothing
  local h::Union{Nothing, _InstantPre} = get(_INSTANT_PRE, integrator, nothing)
  return (h === nothing || isempty(h.held)) ? nothing : h
end

"""
The held discretes back at their values at the step's start, where the
integrator was moved to an event's time inside the step (`HeldBack`).

The move interpolates every state; the interpolant of a constant is not the
constant: `(1 - Θ)*v + Θ*v` is off by an ulp for some Θ, and the value stays
(15.0 became 15.000000000000002 at an event of another variable). A comparison
for equality reads it: MSL's CombiTimeTable returns the value before a time
event while `nextTimeEvent == preNextTimeEvent && t >= nextTimeEvent`, so with
nextTimeEvent an ulp later the table's right side had a point discontinuity at
the event's time, exactly where a step cut to it evaluates its stages (MSL
4.1.0 Fluid.Examples.TraceSubstances.RoomCO2WithControls: a wrong dense output
of that step). Nothing writes a held discrete inside a step, so its value at
an event is the one at the step's start.
"""
function _heldBack!(h::_InstantPre, integrator)::Nothing
  local u::Vector{Float64} = integrator.u
  local uprev::Vector{Float64} = integrator.uprev
  for k::Int in h.held
    u[k] = uprev[k]
  end
  return nothing
end

"""
    instantPre(integrator)

The state before the current event instant, for pre() in an affect: recorded
by the first affect at a new time. Read only, but for the discrete clusters'
members, put back to their held values (`_putBack!`).
"""
function instantPre(integrator)
  ismutable(integrator) || return copy(integrator.u)
  local h::_InstantPre = get!(_InstantPre, _INSTANT_PRE, integrator)
  if h.t != integrator.t
    h.t = integrator.t
    h.u = copy(integrator.u)
    empty!(h.reinitialized)
  end
  return h.u
end

"""
    noteReinit!(integrator, k)

A reinit() of the state `k` at this instant, before it writes it (see `instantPre`).
"""
function noteReinit!(integrator, k::Int)
  ismutable(integrator) || return nothing
  instantPre(integrator)
  push!(_INSTANT_PRE[integrator].reinitialized, k)
  return nothing
end

"""
The states a reinit() wrote at this instant after the first `since` (a
sweep hides those of the instant until it began: a reinit() in its own
bodies is seen in the next sweep).
"""
function _reinitializedSince(integrator, since::Int)::Vector{Int}
  ismutable(integrator) || return Int[]
  local h = get(_INSTANT_PRE, integrator, nothing)
  (h === nothing || h.t != integrator.t || length(h.reinitialized) <= since) && return Int[]
  return unique(h.reinitialized[(since + 1):end])
end

_reinitializedCount(integrator) =
  ismutable(integrator) ? (local h = get(_INSTANT_PRE, integrator, nothing); h === nothing ? 0 : length(h.reinitialized)) : 0

"""
`f()` with the states `hidden` at their values in `pre` (the reinitialized
ones in the first sweep), put back after.
"""
function _hidingReinit(f, integrator, pre, hidden::Vector{Int})
  isempty(hidden) && return f()
  local now = integrator.u[hidden]
  integrator.u[hidden] = pre[hidden]
  try
    return f()
  finally
    integrator.u[hidden] = now
  end
end

function _endInstant!(integrator)
  ismutable(integrator) || return nothing
  local h = get(_INSTANT_PRE, integrator, nothing)
  h === nothing || (h.t = NaN; h.u = nothing)
  return nothing
end

"""
The holder's state if an affect recorded it at the start of this solve. A
holder of an earlier solve from the same time (reinit!) is not read: it
was dropped when that solve's iteration ended, or its time is another.
"""
function _instantStateAtStart(integrator)
  ismutable(integrator) || return nothing
  local h = get(_INSTANT_PRE, integrator, nothing)
  return h === nothing || h.t != integrator.t ? nothing : h.u
end

#= Per integrator, what runs where an instant begins (delays.jl: the delay() arguments' left limits). =#
const _AT_INSTANT_START = WeakKeyDict{Any, Any}()

"""
    atInstantStart!(integrator, record!)

`record!(integrator)` runs where an event instant begins (`beginInstant!`), at its left limit, before any
affect at that time. One per integrator; a new solve's replaces it.
"""
function atInstantStart!(integrator, record!)
  ismutable(integrator) && (_AT_INSTANT_START[integrator] = record!)
  return nothing
end

"""
    beginInstant!(integrator)

An event instant begins where a branch event's callback moves the integrator to the event's time, before
its affect (the first re-solve at that time, eventReinit.jl): the state there is the left limit. The
pre-state is recorded there (`instantPre`): recorded later, after the re-solve, pre() of an algebraic
unknown read its value on the new branch (MSL 4 DemonstrateContinuousSignalExtrema's `max({pre(y_max), u,
pre(u)})` missed the peak before a pulse's step). Then what `atInstantStart!` registered runs: without
the left limit, delay() of a pulse ramped down over the pulse (MSL 4 CompareLosslessLines).
"""
function beginInstant!(integrator)
  ismutable(integrator) || return nothing
  local h = get(_INSTANT_PRE, integrator, nothing)
  local fresh = h === nothing || h.t != integrator.t
  instantPre(integrator)
  if fresh
    local record! = get(_AT_INSTANT_START, integrator, nothing)
    record! === nothing || record!(integrator)
  end
  return nothing
end
