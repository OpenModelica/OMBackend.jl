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
end

const _INSTANT_PRE = WeakKeyDict{Any, _InstantPre}()

"""
    instantPre(integrator)

The state before the current event instant, for pre() in an affect: recorded
by the first affect at a new time. Read only.
"""
function instantPre(integrator)
  ismutable(integrator) || return copy(integrator.u)
  local h = get!(() -> _InstantPre(NaN, nothing, Int[]), _INSTANT_PRE, integrator)
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
