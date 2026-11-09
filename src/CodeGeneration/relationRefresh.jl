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

#= Relations with a hysteresis (MLS 8.5, OpenModelica's relationhysteresis).
   A relation's value is buffered (an if-equation branch's ifCond, or the
   buffer of a when on a relation), changes only at events, and its crossing
   function is shifted by eps = H * scale relative to that value.

   Event iteration (MLS 8.6): the continuous callbacks locate events; the
   events are handled after the step, all in one place. An event can move a
   relation's operand without that relation's crossing function crossing
   zero (a reinit, a branch switch that changes an algebraic variable, a
   complementary relation on the same zero set): the jump happens between
   two evaluations. So after every step each buffered relation is checked
   against the state with OpenModelica's rule (a TRUE relation stays TRUE
   while zc <= eps, a FALSE one becomes TRUE when zc <= -eps), and where one
   disagrees, or where a continuous event fired in the step, the event is
   handled as OpenModelica's updateDiscreteSystem. The algebraic unknowns
   are solved first (an if-equation's branch switch leaves them stale), then
   in sweeps:
   1. every relation is evaluated on the same state;
   2. the whens on a relation that became true run their bodies, in order,
      with pre() read from the state before the sweep; each discrete cluster
      (discreteClusters.jl) is solved as a mixed system with the pre()
      values of the sweep; then the whens on discrete conditions (a changed
      discrete, an algorithm section's inputs), each checked just before it
      runs, in the order of the model; where those change the state, the
      relations and the bodies again, with the same pre(), until they do not;
   3. the algebraic unknowns are solved again, and the next sweep starts,
      until nothing changes. A chain through algebraic variables takes one
      sweep per link; a model that does not settle (chattering) is stopped
      with an error, as OpenModelica does.

   The instant's pre-state: eventInstant.jl; the algebraic re-solve at an
   event (EventReinit): eventReinit.jl. =#

"""
The hysteresis H for a relative tolerance and a solve's time span, as
OpenModelica's tolZC = 1e-4 * min(stepSize, tolerance), its output step
the span over 500 intervals (its default): an absolute 1e-4 * reltol is
1e-7 at the default tolerance, the whole span of the MSL Spice3 examples,
and no relation on time switched before their stop time (a pulse source
stayed at 0 V). The if-equation relations read it from their parameter
(setZCHysteresis! sets it per solve), the whens from the integrator.
"""
_hysteresis(reltol, span::Real = Inf) =
  1.0e-4 * max(min(Float64(first(reltol)), abs(Float64(span)) / 500), 1.0e-12)
_timeSpan(tspan) = Float64(last(tspan)) - Float64(first(tspan))
_hysteresisFromTolerance(integrator) = _hysteresis(integrator.opts.reltol, _timeSpan(integrator.sol.prob.tspan))

"""
A relation's value by the hysteresis rule, from its buffered value `old`:
true when zc <= 0 (true while zc <= eps, becomes true at zc <= -eps).
"""
_hysteresisRule(zc, eps, old::Bool) = old ? zc <= eps : zc <= -eps

"""
Whether a crossing function `zc` is on its relation's boundary to rounding
(a ramp's end, `1 + (-1)` or `1 - 1.0000000000000009`), far inside the
hysteresis band.
"""
_onBoundary(zc, scale) = abs(zc) <= 1.0e-12 * max(abs(scale), 1.0)

"""
Sweeps before the event iteration gives up: at least 20, and 3 per
relation or when (a chain through algebraic variables takes one per
link).
"""
_eventIterationLimit(n::Integer) = max(20, 3 * n)

#= The if-equation relations: their buffers (the ifCond parameters), and
   one function that reads every one's crossing function and scale (one
   observed function per value recomputed the observed chain each and cost
   three dynamic calls a relation: MSL BatchPlant compiled 82 of them). =#
struct IfRelations
  buffers::Any                         # integrator or problem -> the ifConds
  setBuffer::Vector{Any}               # per relation: (integrator, value) -> nothing
  values!::Any                         # (out, u, p, t): relation k's crossing function out[2k - 1], its scale out[2k]
  zs::Vector{Float64}                  # the output of values!
  hysteresis::Any                      # integrator -> H
  compiled::Vector{Float64}            # the ifConds the initialization used
  timed::Vector{Bool}                  # per relation: its condition reads time (`_bandEdgeRule`)
end

Base.length(r::IfRelations) = length(r.timed)

"""
The relations `entries` = `(ifCond, crossing function, scale, reads time)` of
`problem`; nothing when there are none or they cannot be read.
"""
function _ifRelations(problem, hSym::Symbol, entries::Vector)
  isempty(entries) && return nothing
  local SII = ModelingToolkit.SymbolicIndexingInterface
  local syms = Symbol[e[1] for e in entries]
  try
    local buffers = SII.getp(problem, syms)
    local reads::Vector{Any} = Any[x for e in entries for x in (e[2], e[3])]
    return IfRelations(buffers, Any[SII.setp(problem, s) for s in syms],
                       first(_buildObservedFunction(problem.f.sys, reads; return_inplace = Val(true))),
                       zeros(length(reads)), SII.getp(problem, hSym), _bufferValues(buffers, problem),
                       Bool[length(e) >= 4 && e[4] === true for e in entries])
  catch err
    OMBackend._fallback(err, :ifRelationsFromProblem; impact = :result)
    @warn "[events] the if-equation relations cannot be read from the problem; they are not iterated" exception = err
    return nothing
  end
end

"""
A copy: setting a buffer changes the parameters the getter reads.
"""
_bufferValues(buffers, integrator) = Float64[v for v in buffers(integrator)]

"""
The relations' crossing functions and scales at the integrator's state and parameters (the buffers): relation k's
are `zs[2k - 1]` and `zs[2k]`.
"""
function _ifValues!(r::IfRelations, integrator)::Vector{Float64}
  r.values!(r.zs, integrator.u, integrator.p, integrator.t)
  return r.zs
end

"""
Relation k by the hysteresis rule from `old`, its crossing function and scale from `zs` (`_ifValues!`), `H` the
hysteresis.
"""
function _ruled(r::IfRelations, zs::Vector{Float64}, H::Float64, k::Int, old::Bool)::Bool
  local scale::Float64 = zs[2k]
  local eps::Float64 = H * scale
  local zc::Float64 = zs[2k - 1]
  return r.timed[k] ? _bandEdgeRule(zc, eps, scale, old) : _hysteresisRule(zc, eps, old)
end

"""
    _bandEdgeRule(zc, eps, scale, old) -> Bool

The hysteresis rule (`_hysteresisRule`) with the band's edge, to rounding (`_onBoundary`), as crossed: for a
relation that reads time. Its callback locates a crossing where its shifted function is zero, the band's edge;
a relation of the same crossing function (a twin) is at its own edge there, to rounding, and its callback does
not fire: OrdinaryDiffEq takes a component near zero at the last event as crossed. By the band it stayed: MSL
Blocks.Sources.Trapezoid without width, `time < T_start + T_rising` and `time < T_start + T_width`, held
SignalPWM's triangle at its peak for the rest of a period (HBridge_DC_Drive missed two switchings). Not for a
relation of the state: two equal pipes' flows at their laminar limit (MSL FluidHeatFlow TwoMass) are crossed
by their own callbacks, and taken as crossed in the sweep the run became Unstable.
"""
function _bandEdgeRule(zc::Real, eps::Real, scale::Real, old::Bool)::Bool
  #= only an edge apart from the relation's own boundary: at zc = 0 (a time event exactly on its threshold,
     MSL Spice3's pulse sources) the literal decides, `<` or `>=`, which the band keeps and this rule does not
     know (flipped there, the start's event iteration ended the solve: CompareLosslessLines, Spice3 Inverter) =#
  local atEdge = _onBoundary(old ? zc - eps : zc + eps, scale) && !_onBoundary(zc, scale)
  return atEdge ? !old : _hysteresisRule(zc, eps, old)
end

function _inconsistent(r::IfRelations, integrator)
  local values = r.buffers(integrator)
  local zs::Vector{Float64} = _ifValues!(r, integrator)
  local H::Float64 = r.hysteresis(integrator)
  return any(k -> _ruled(r, zs, H, k, values[k] > 0.5) != (values[k] > 0.5), eachindex(values))
end

"""
Phase 1 of a sweep for the if-equation relations: each buffer from the
state. Whether one changed.
"""
function _update!(r::IfRelations, integrator)
  local changed = false
  local H::Float64 = r.hysteresis(integrator)
  local zs::Vector{Float64} = _ifValues!(r, integrator)
  for (k, value) in enumerate(_bufferValues(r.buffers, integrator))
    local old = value > 0.5
    local new = _ruled(r, zs, H, k, old)
    new == old && continue
    r.setBuffer[k](integrator, new ? 1.0 : 0.0)
    #= the later relations read this buffer's new value (a crossing function can read it through the parameters) =#
    _ifValues!(r, integrator)
    changed = true
  end
  return changed
end

"""
The if-equation relations switched back and forth in the event iteration:
a comparator with positive feedback (MSL SignalGenerator) has its linear
branch's solution at the unstable midpoint, where the relation flips back,
and the saturated one it left. OpenModelica solves the loop with the live
if-expression and lands on the other saturated side. Here the combinations
of the literals, the fewest changes first (at most `maxRelations`
relations), until one holds at its own solution: the algebraic unknowns
solved for it, every relation evaluates to its literal there (with
`clear`, beyond its hysteresis band too). Whether one was found; the
literals and the state are then that solution's, else as before. A failed
solve's return code is put back.
"""
function _consistentIfRelations!(r::IfRelations, integrator, reinit; maxRelations::Int = 10, clear::Bool = false)
  local base = _bufferValues(r.buffers, integrator)
  local n = length(base)
  (n == 0 || n > maxRelations) && return false
  local u0 = copy(integrator.u)
  local retcode = integrator.sol.retcode
  local setAll! = lits -> foreach(k -> r.setBuffer[k](integrator, lits[k] ? 1.0 : 0.0), 1:n)
  local current = Bool[v > 0.5 for v in base]
  local H::Float64 = r.hysteresis(integrator)
  local clearly = (zs, k, lit) -> _ruled(r, zs, H, k, !lit) == lit
  #= With `clear`, a combination that holds clearly where it is stays: the
     instant was only iterated again (several callbacks at one time). =#
  if clear
    local zsNow::Vector{Float64} = _ifValues!(r, integrator)
    all(k -> clearly(zsNow, k, current[k]), 1:n) && return false
  end
  local zsStart::Vector{Float64} = copy(_ifValues!(r, integrator))
  local zc0 = Float64[zsStart[2k - 1] for k in 1:n]
  #= Every relation holds. With `clear`, beyond its hysteresis band too, but
     for one the solve does not move (its crossing as at the start: an
     independent relation at its threshold, which holds either way), which
     keeps its literal. =#
  local holds = cand -> begin
    local zs::Vector{Float64} = _ifValues!(r, integrator)
    all(k -> _ruled(r, zs, H, k, cand[k]) == cand[k], 1:n) || return false
    clear || return true
    all(1:n) do k
      local static = isapprox(zs[2k - 1], zc0[k]; rtol = 1e-12, atol = 1e-14)
      static ? cand[k] == current[k] : clearly(zs, k, cand[k])
    end
  end
  #= Bit k of `flips` changes relation k; 0 is the combination that did not settle. =#
  for flips in sort!(collect(1:(2^n - 1)); by = count_ones)
    local cand = Bool[xor(current[k], isodd(flips >> (k - 1))) for k in 1:n]
    setAll!(cand)
    copyto!(integrator.u, u0)
    if _resolveAlgebraics!(integrator, reinit)
      holds(cand) && return true
    else
      _restoreRetcode!(integrator, retcode)
    end
  end
  setAll!(current)
  copyto!(integrator.u, u0)
  return false
end

#= When-equations on a single relation (codeGen.jl `_emitRelationWhen`).
   The relation keeps its value between events in a buffer (MLS 8.5), set
   literally at the start; its crossing function is shifted by the same
   hysteresis as the if-equation branches, relative to that value, so a
   relation that starts at its threshold becomes true right after the start
   and one that crosses back is seen. The when body runs only when the
   relation becomes true (MLS 8.3.5). The operands are read by name, so they
   may be unknowns, observed variables or parameters. =#

"""
    variableName(v) -> Union{Nothing, String}

The name of the symbolic variable `v` as SimCode names it (`opener[3]_tSwitch` for `var"opener[3]_tSwitch"(t)`, `R`
for a parameter), from the symbol's own name, not from its printing; `nothing` for anything else (a derivative
`Differential(t)(x)`, a sum, a number).
"""
function variableName(@nospecialize(v))::Union{Nothing, String}
  local x = Symbolics.unwrap(v)
  x isa SymbolicUtils.BasicSymbolic || return nothing
  SymbolicUtils.issym(x) && return String(nameof(x))
  (SymbolicUtils.iscall(x) && SymbolicUtils.issym(SymbolicUtils.operation(x))) || return nothing
  return String(nameof(SymbolicUtils.operation(x)))
end

"""
    _variableNames(xs) -> OrderedSet{String}

The names (`variableName`) of the variables among `xs`, in order; anything else (a derivative, a number) has none.
"""
function _variableNames(xs)::OrderedSet{String}
  local out::OrderedSet{String} = OrderedSet{String}()
  for x in xs
    local name::Union{Nothing, String} = variableName(x)
    name === nothing || push!(out, name)
  end
  return out
end

"""
    mtkDerivativeVariables(sys) -> Set{Any}

The variables MTK created for derivatives (`xˍt`), unwrapped: the ones its schedule substitutes for derivatives
(`dummy_sub`: `D(x) => xˍt`) that the generated model did not declare (no `SimCodeKind` of their own: `xˍt` keeps
`x`'s, `simCodeKind`). The schedule substitutes the model's own variables too (`D(body.r_0) => body.v_0`, MSL
HeatLosses): taken as MTK's, the bodies' velocities were given MTK's derivative starts and the solve ended in NaN at
its first step.
"""
function mtkDerivativeVariables(sys::ModelingToolkit.AbstractSystem)::Set{Any}
  ModelingToolkit.get_schedule(sys) === nothing && return Set{Any}()
  return Set{Any}(x for x in (Symbolics.unwrap(v) for v in values(ModelingToolkit.get_schedule(sys).dummy_sub))
                  if simCodeKind(x) === nothing)
end

"""
    isIndependentVariable(v) -> Bool

Whether `v` is the system's independent variable (time): the one plain symbol that is no parameter (an unknown is a
call `x(t)`, a parameter carries the parameter metadata).
"""
function isIndependentVariable(@nospecialize(v))::Bool
  local x = Symbolics.unwrap(v)
  return x isa SymbolicUtils.BasicSymbolic && SymbolicUtils.issym(x) && !ModelingToolkit.isparameter(x)
end

"""
    namedValueFunctions(sys, names) -> Tuple or nothing

For each variable name (as the Modelica cref prints), a function `(u, p, t)`
that returns its value: an unknown, an observed variable or a parameter; a
tuple, so a call of all of them is typed (`_callEvent`). Nothing when the
system lacks one of them (a variable-structure model runs the callbacks of
every mode; a mode's callbacks see the other modes' systems).
"""
function namedValueFunctions(sys, names::Vector{String})::Union{Nothing, Tuple}
  sys === nothing && return nothing
  local byName::Dict{String, Any} = _variablesByPlainName(sys)
  all(n -> haskey(byName, n), names) || return nothing
  return Tuple(ModelingToolkit.build_explicit_observed_function(sys, byName[n]) for n in names)
end

"""
    eventFunction(code, mod) -> RuntimeGeneratedFunction

An event's code, `(t, values...) -> value` (the crossing function of a continuous when, `WhenCondition`; a when on a
relation's crossing function and scale, `relationWhenCallback`), compiled: a RuntimeGeneratedFunction of
CodeGeneration whose names that are not its own are read from the model's module `mod` (`_modelNamesFrom`), its
calls of Modelica functions through their wrappers. As a closure of the model's module it ran interpreted (the module
is compile=min, `_moduleCompilerOptions`): MSL SoftStarter's two continuous whens 1.5-1.9 us a condition, its two whens
on a relation 0.8-1.1 us, each evaluated about ten times a step. A RuntimeGeneratedFunction is called in any world, and
one code is compiled once.
"""
function eventFunction(code::Expr, mod::Module)::RuntimeGeneratedFunctions.RuntimeGeneratedFunction
  return RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, _modelNamesFrom(code, mod))
end

"""
`ev(t, values...)` for the values the value functions `fns` (namedValueFunctions) give at (u, p, t): the call behind
the barrier, typed by the functions' tuple.
"""
_callEvent(ev::Function, fns::Tuple, u::AbstractVector{Float64}, p, t::Float64) = ev(t, map(f -> f(u, p, t), fns)...)

"""
    WhenCondition(names, eval)

The condition of a continuous when (codeGen.jl `_continuousWhen`): its crossing function, `eval(t, values...)`
(eventFunction) of the values of `names`, which the system's value functions read (namedValueFunctions), resolved once
per system. Where the system lacks a name (another mode's system, in a variable-structure model) 1.0: no crossing.
"""
mutable struct WhenCondition{E}
  names::Vector{String}
  eval::E
  sys::Any                   # the system `fns` were built for (:unresolved at first)
  fns::Union{Nothing, Tuple} # the value functions, or nothing (another mode's system)
end

WhenCondition(names::Vector{String}, eval::RuntimeGeneratedFunctions.RuntimeGeneratedFunction) =
  WhenCondition(names, eval, :unresolved, nothing)

function (c::WhenCondition)(u::AbstractVector{Float64}, t::Float64, integrator)::Float64
  local sys::Union{Nothing, ModelingToolkit.AbstractSystem} = hasproperty(integrator.f, :sys) ? integrator.f.sys : nothing
  sys === c.sys || (c.sys = sys; c.fns = namedValueFunctions(sys, c.names))
  local fns::Union{Nothing, Tuple} = c.fns
  fns === nothing && return 1.0
  return _callEvent(c.eval, fns, u, integrator.p, t)
end

"""
The system's unknowns, observed variables and parameters by name (`variableName`).
"""
function _variablesByPlainName(sys)::Dict{String, Any}
  local byName = Dict{String, Any}()
  for v in Iterators.flatten((ModelingToolkit.unknowns(sys), (eq.lhs for eq in ModelingToolkit.observed(sys)),
                              ModelingToolkit.parameters(sys)))
    local name::Union{Nothing, String} = variableName(v)
    name === nothing || (byName[name] = v)
  end
  return byName
end

"""
    RelationWhenAffect

A when on one relation: its buffer, the condition's value when the whens
last fired (for the edge), the functions that read its operands (per
system), and the body. Its continuous callback only locates the event; the
event iteration (`withRelationRefresh`) updates the buffer and fires the
body, with the other relations.
"""
mutable struct RelationWhenAffect{E, B}
  names::Vector{String}
  eval::E                    # (t, values...) -> (zc, scale) (eventFunction); true when zc < 0 (strict) or zc <= 0
  strict::Bool
  body!::B                   # (integrator, pre) -> nothing; pre(v) read from the vector `pre`
  sys::Any                   # the system `fns` were built for (:unresolved at first)
  fns::Any                   # the value functions (a Tuple), or nothing (another mode's system)
  rel::Bool                  # the relation's buffer
  last::Bool                 # the condition when the whens last fired
  located::Bool              # its crossing function located the event being handled
end

"""
The value functions for the integrator's system; nothing while it lacks
the names (another mode). Resolved once per system (`a.sys`, initially
:unresolved); becoming active, the relation starts literal and does not
fire (a when true at the start is not an edge).
"""
function _whenFunctions(a::RelationWhenAffect, integrator)
  local sys = hasproperty(integrator.f, :sys) ? integrator.f.sys : nothing
  a.sys === sys && return a.fns
  a.sys = sys
  a.fns = namedValueFunctions(sys, a.names)
  if a.fns !== nothing
    a.rel = _whenLiteral(a, integrator.u, integrator.t, integrator)
    a.last = a.rel
  end
  return a.fns
end

"""
The crossing function and the scale at (u, t). `eval` is an eventFunction and the value functions are MTK's, both
RuntimeGeneratedFunctions: called in any world, also where a variable-structure model compiles them after the solve
started (`body!`, a closure of the model's module, is called in the latest world).
"""
_whenValues(a::RelationWhenAffect, u::AbstractVector{Float64}, t::Float64, integrator)::Tuple{Float64, Float64} =
  _callEvent(a.eval, a.fns, u, integrator.p, t)

"The when's relation at the start: literal, a tie to rounding as the exact tie (as `_tieLiteral`)."
function _whenLiteral(a::RelationWhenAffect, u, t, integrator)
  local (zc, scale) = _whenValues(a, u, t, integrator)
  _onBoundary(zc, scale) && return !a.strict
  return a.strict ? zc < 0 : zc <= 0
end

"""
The relation's value by the hysteresis rule at the integrator's state.
"""
function _whenRuled(a::RelationWhenAffect, integrator)
  local (zc, scale) = _whenValues(a, integrator.u, integrator.t, integrator)
  return _hysteresisRule(zc, _hysteresisFromTolerance(integrator) * scale, a.rel)
end

function _whenInconsistent(a::RelationWhenAffect, integrator)
  _whenFunctions(a, integrator) === nothing && return false
  return _whenRuled(a, integrator) != a.rel
end

"""
Phase 1 of a sweep: the buffer from the state. Whether it changed. A false
`<=` or `>=` exactly on its boundary at the event holds (as a cluster's,
discreteClusters.jl `_update!`). A relation whose own crossing function
located the event changes where the root finder left it on the shifted
boundary (`_locatedOnBoundary`), though the rule does not see it past.
"""
function _whenRelationUpdate!(a::RelationWhenAffect, integrator)
  _whenFunctions(a, integrator) === nothing && return false
  local (zc, scale) = _whenValues(a, integrator.u, integrator.t, integrator)
  local eps = _hysteresisFromTolerance(integrator) * scale
  local new = _hysteresisRule(zc, eps, a.rel)
  (!new && !a.rel && !a.strict && _onBoundary(zc, scale)) && (new = true)
  (new == a.rel && a.located && _locatedOnBoundary(zc, eps, a.rel)) && (new = !a.rel)
  a.located = false
  new == a.rel && return false
  a.rel = new
  return true
end

"""
Whether the shifted crossing function `zc + eps (1 - 2 rel)` (the callback's condition) is
within 1 % of the hysteresis `eps` of zero: where the root finder left the event it
located, a hair before the boundary. The rule kept the relation, nothing changed, and the
root was found again at the next step: MSL Machines IMC_DCBraking with preferred states, a
dynamic state set's switch at +2.3e-14 for 200 000 steps of 4e-16 s (MaxIters).
"""
_locatedOnBoundary(zc, eps, rel::Bool)::Bool = abs(zc + eps * (1 - 2 * rel)) <= 1.0e-2 * abs(eps)

"""
Phase 2: the body, when the condition became true since the whens last
fired; `pre` is the state before the sweep's bodies. Whether it fired.
"""
function _whenFire!(a::RelationWhenAffect, integrator, pre)
  a.fns === nothing && return false
  local fire = a.rel && !a.last
  a.last = a.rel
  fire && Base.invokelatest(a.body!, integrator, pre)
  return fire
end

#= The crossing only locates the event: it is handled after the step, with
   every other relation (the event iteration). The state is not changed; the
   relation records that it located it (_whenRelationUpdate!). =#
(a::RelationWhenAffect)(integrator) = (a.located = true; _derivativeDiscontinuity!(integrator, false); nothing)

"""
    relationWhenCallback(names, eval, strict, body!) -> ContinuousCallback

A when-equation on one relation. `eval(t, values...)` returns `(zc, scale)`
for the values of `names`: the relation is true when zc < 0 (`strict`) or
zc <= 0, and `scale` sizes the hysteresis. `body!(integrator, pre)` is the
when body (`pre(v)` read from `pre`).
"""
function relationWhenCallback(names::Vector{String}, eval, strict::Bool, body!)
  local a = RelationWhenAffect(names, eval, strict, body!, :unresolved, nothing, false, false, false)
  local condition = function (u, t, integrator)
    _whenFunctions(a, integrator) === nothing && return 1.0
    local (zc, scale) = _whenValues(a, u, t, integrator)
    return zc + _hysteresisFromTolerance(integrator) * scale * (1 - 2 * a.rel)
  end
  #= A new solve: the relation starts literal again. The value functions are
     kept for the same system (they were built again at every solve: one
     generated function per name, per when, as OMSurrogates solves a build
     over and over). =#
  local initialize = function (c, u, t, integrator)
    a.located = false
    local sys = hasproperty(integrator.f, :sys) ? integrator.f.sys : nothing
    if a.sys === sys && a.fns !== nothing
      a.rel = _whenLiteral(a, integrator.u, integrator.t, integrator)
      a.last = a.rel
    else
      a.sys = :unresolved; a.fns = nothing
      _whenFunctions(a, integrator)
    end
    return nothing
  end
  #= RightRootFind: the event lands just past the root, where the rule sees
     the crossing. It saves the left limit; the event iteration saves the
     right one and solves the algebraic variables (none here). The initialize
     function leaves the modified flag alone (the default one clears what
     another callback set). =#
  return DiffEqBase.ContinuousCallback(condition, a, a;
                                       initialize = initialize, rootfind = ModelingToolkit.SciMLBase.RightRootFind,
                                       initializealg = ModelingToolkit.SciMLBase.NoInit(),
                                       save_positions = (true, false))
end

"""
    DiscreteWhenAffect

A when on a discrete condition (a changed discrete, a Boolean, an algorithm
section's inputs): its condition and body, as a DiscreteCallback's. Where a
model has buffered relations, the event iteration (`withRelationRefresh`)
runs it, after the whens on a relation; otherwise its DiscreteCallback does.
"""
struct DiscreteWhenAffect{C, A, I}
  condition::C               # (u, t, integrator[, follow::Bool]) -> Bool: whether it fires; where it does
                             # not and `follow`, its change()/edge() memory takes the values
  affect!::A                 # (integrator[, pre]) -> nothing; pre(v) read from `pre` (default: u on entry)
  initialize!::I             # (u, t, integrator) -> nothing: its state (an edge latch) at the start
  reads::Vector{String}      # the names its condition and body read at their current values (not under pre())
  writes::Vector{String}     # the names its body assigns (`_sweepOrder`)
end

(d::DiscreteWhenAffect)(integrator) = d.affect!(integrator)

"""
    TieredCondition(interp, compiled, shared, nodes)

A discrete when's condition (codeGen.jl `_tieredCondition`), evaluated after every step: its closure of the model's
module `interp`, which runs interpreted (the module is compile=min), until it has run about as long as compiling its
code of `nodes` nodes takes (CompileTier, COMPILE_AFTER_NS_PER_NODE), then the same code `compiled` (eventFunction),
the closure's shared state (`shared`: its caches, edge latch and change() memory, which its affect and initialization
change) its arguments. Interpreted, each of MSL SoftStarter's nine took 2.4-2.9 us a step. Not parametric: its method
and the functions given it compile once, not once per condition (55 in MSL BatchPlant); the compiled call is behind a
barrier (`_compiledCondition`).
"""
struct TieredCondition <: Function
  interp::Function
  compiled::RuntimeGeneratedFunctions.RuntimeGeneratedFunction
  shared::Tuple
  tier::CompileTier
end

TieredCondition(interp::Function, compiled::RuntimeGeneratedFunctions.RuntimeGeneratedFunction, shared::Tuple, nodes::Int) =
  TieredCondition(interp, compiled, shared, CompileTier(:condition, 0, COMPILE_AFTER_NS_PER_NODE[] * max(nodes, 1), false))

function (c::TieredCondition)(x::AbstractVector, t::Real, integrator, follow::Bool = true)::Bool
  local tier::CompileTier = c.tier
  tier.compiled && return _compiledCondition(c.compiled, c.shared, x, t, integrator, follow)
  local t0::UInt64 = time_ns()
  local r::Bool = c.interp(x, t, integrator, follow)
  tier.interpreted += time_ns() - t0
  tier.interpreted >= tier.after && (tier.compiled = true)
  return r
end

"A tiered condition's compiled code called with its shared state: typed by the code and the state."
_compiledCondition(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction, shared::Tuple, x::AbstractVector, t::Real,
                   integrator, follow::Bool)::Bool = f(x, t, integrator, follow, shared...)

const _NO_WHEN_INITIALIZE = (u, t, integrator) -> nothing

"""
    discreteWhenCallback(condition, affect!, initialize! = _NO_WHEN_INITIALIZE; reads, writes) -> DiscreteCallback

A when on a discrete condition, checked after every step. `initialize!` runs at
the start of every solve (a reinit! included). Its change()/edge() memory
follows the values only after a step no other callback changed: an earlier
callback in the same step can change a discrete without solving the algebraic
unknowns again, and `edge(b) and c` would lose b's edge on a stale c (it fires
a step later instead). `reads` and `writes`: the names it reads at their
current values and those it assigns, which place it in the event iteration's
sweep (`_sweepOrder`).
"""
discreteWhenCallback(condition, affect!, initialize! = _NO_WHEN_INITIALIZE;
                     reads::Vector{String} = String[], writes::Vector{String} = String[]) =
  DiffEqBase.DiscreteCallback((u, t, integrator) -> condition(u, t, integrator, !_changedThisStep(integrator)),
                              DiscreteWhenAffect(condition, affect!, initialize!, reads, writes);
                              initialize = (c, u, t, integrator) -> begin
                                initialize!(u, t, integrator)
                                _derivativeDiscontinuity!(integrator, false)
                                nothing
                              end,
                              save_positions = (true, true))

#= What the event iteration updates. `reinit` solves the algebraic
   unknowns (nothing: EventReinit). =#
struct EventIteration
  ifRelations::Union{Nothing, IfRelations}
  relationWhens::Vector{RelationWhenAffect}
  clusters::Vector{DiscreteCluster}
  discreteWhens::Vector{DiscreteWhenAffect}
  #= A sweep's pass over the clusters (i) and the discrete whens (-j), in the order of what they read (`_sweepOrder`). =#
  order::Vector{Int}
  limit::Int
  reinit::Any
  #= The last instant iterated and how often in a row (the integrator stuck
     at one time: `_iterateInstant!`). =#
  instant::Base.RefValue{Tuple{Float64, Int}}
  #= The instant of the last degenerate restart and how many in a row (`_restartStepSize!`). =#
  stall::Base.RefValue{Tuple{Float64, Int}}
end

"""
Whether a callback changed the state in the step that just ended: a
continuous event, or an earlier discrete callback (the integrator's flag,
`derivative_discontinuity` since SciMLBase 3, `u_modified` before).
"""
_changedThisStep(integrator) =
  _continuousEventFired(integrator) ||
  (hasproperty(integrator, :derivative_discontinuity) && integrator.derivative_discontinuity) ||
  (hasproperty(integrator, :u_modified) && integrator.u_modified)

"""
Whether the when fires; where it does not, its change()/edge() memory
takes the values.
"""
_holds(d::DiscreteWhenAffect, integrator) = d.condition(integrator.u, integrator.t, integrator)

"""
Within a sweep: the change()/edge() memory stays (the condition's
`_follow` false). It follows the values once the iteration settles.
"""
_holdsInSweep(d::DiscreteWhenAffect, integrator) = d.condition(integrator.u, integrator.t, integrator, false)

_follow!(d::DiscreteWhenAffect, integrator) = (_holds(d, integrator); nothing)

"""
Whether a continuous callback fired in the step that just ended: the
integrators of OrdinaryDiffEqCore and Sundials record it in
`event_last_time` (0 when none) before they run the discrete callbacks.
"""
_continuousEventFired(integrator) =
  hasproperty(integrator, :event_last_time) && integrator.event_last_time != 0

"""
After a step: whether a continuous event fired, a relation disagrees with
the state, or a discrete when's condition holds.
"""
function _needsIteration(e::EventIteration, integrator)
  _continuousEventFired(integrator) && return true
  e.ifRelations !== nothing && _inconsistent(e.ifRelations, integrator) && return true
  any(a -> _whenInconsistent(a, integrator), e.relationWhens) && return true
  any(c -> _inconsistent(c, integrator), e.clusters) && return true
  return any(d -> _holds(d, integrator), e.discreteWhens)
end

"""
Phases 1 and 2 of a sweep (see the top of the file); the first of an
instant reads pre() from the state before it (instantPre). Whether
anything changed.
"""
function _sweep!(e::EventIteration, integrator, firstSweep::Bool)
  local pre = firstSweep ? instantPre(integrator) : copy(integrator.u)
  local relPre = [copy(c.rel) for c in e.clusters]
  #= The first sweep hides every reinit() of the instant, a later one those of its own bodies. =#
  local since = firstSweep ? 0 : _reinitializedCount(integrator)
  #= 1. Every relation on this state. =#
  local changed = _hidingReinit(integrator, pre, _reinitializedSince(integrator, since)) do
    local ch = e.ifRelations !== nothing && _update!(e.ifRelations, integrator)
    for a in e.relationWhens
      _whenRelationUpdate!(a, integrator) && (ch = true)
    end
    for c in e.clusters
      _update!(c, integrator) && (ch = true)
    end
    ch
  end
  #= 2. The bodies, with pre() from the state before them. A relation when's
     discretes, or a cluster that no re-solve moves (`coupled` false, a pulse
     switching a circuit), can move the algebraic unknowns: they are solved
     again before a coupled cluster or a discrete when reads them in this
     sweep. The clusters and the discrete whens run in the order of what they
     read (`_sweepOrder`), and a cluster's relations are evaluated on the state
     its body reads: where something before it in the pass changed the state
     (a cluster, a when, a solve of the algebraic unknowns), again before it
     runs, as OpenModelica evaluates a relation where its equation is, in the
     sorted order. MSL StateGraph ControlledTanks: an inner transition of the
     composite step `makeProduct` deactivated its step in one iteration and
     activated the next one in the following, its `integer()` of the active
     steps (`$EVT_k`, discrete.jl `stepWhen`) read 0 in between; evaluated at
     the start of the sweep, its relation `q >= pre(k) + 1` read 0 and was true,
     the steps' cluster before it put q back at -1, and the when fired on that
     relation with q = pre(k): k = pre(k) - 1, two active steps, and the
     composite step never became inactive when suspended. Only until the
     cluster changed its members in this sweep: a relation reads pre(k) as k's
     value in the state, which is pre(k) only before the body writes k (MSL
     StateGraph ShowExceptions: refreshed in a later pass with k written, `q <
     pre(k)` read the new k, and the when fired again on it). =#
  local clusterPre::Vector{AbstractVector} = AbstractVector[_preValues(c, c.values(pre, integrator.p, integrator.t)) for c in e.clusters]
  #= How often the state changed in this sweep, per cluster at which count its relations were evaluated, and whether
     it changed its members in this sweep. =#
  local moves::Int = 0
  local evaluatedAt::Vector{Int} = zeros(Int, length(e.clusters))
  local wrote::Vector{Bool} = fill(false, length(e.clusters))
  for pass in 1:e.limit
    local stale::Bool = false       # the algebraic unknowns, for a coupled cluster or a discrete when next
    local whenStale::Bool = false   # the same after a discrete when, for a coupled cluster next
    local fired::Bool = false
    local passStart::AbstractVector = copy(integrator.u)
    local relationFired::Bool = false
    for a in e.relationWhens
      _whenFire!(a, integrator, pre) && (changed = true; stale = true; relationFired = true; moves += 1)
    end
    for item in e.order
      if item > 0
        local c::DiscreteCluster = e.clusters[item]
        c.coupled && (stale || whenStale) && (_staleResolve!(integrator, e.reinit); stale = whenStale = false; moves += 1)
        if !wrote[item] && evaluatedAt[item] != moves
          _hidingReinit(() -> _update!(c, integrator), integrator, pre, _reinitializedSince(integrator, since))
          evaluatedAt[item] = moves
        end
        _solveMixedSystem!(c, integrator, clusterPre[item], relPre[item], e.reinit) || continue
        changed = true
        wrote[item] = true
        moves += 1
        c.coupled || (stale = true)
      else
        local d::DiscreteWhenAffect = e.discreteWhens[-item]
        stale && (_staleResolve!(integrator, e.reinit); stale = whenStale = false; moves += 1)
        _holdsInSweep(d, integrator) || continue
        local before::AbstractVector = copy(integrator.u)
        d.affect!(integrator, pre)
        changed = true
        isequal(integrator.u, before) || (fired = whenStale = true; moves += 1)
      end
    end
    (fired || (relationFired && !isequal(integrator.u, passStart))) || break
    #= A discrete when (or a when on a relation) changed a discrete: the relations and the
       bodies read the new value in this same iteration (MLS 8.6: its equations hold
       together, with the same pre(); `when x > 0.5 then d = 1` and `when x + d > 1.2
       then y = pre(d)`: y = 0, not the 1 of the next sweep). StateGraph's transition has `when enableFire then t_start = time` and
       `fire = enableFire and time >= t_start + waitTime`: fire is false where the timer
       starts; with the old t_start it fired at once. The algebraic unknowns follow first.
       A when with edge() or an edge latch does not fire again: its affect took the
       values. A when that fired in an earlier pass on a value this pass corrects is not
       undone: the order of a pass (`_sweepOrder`) runs what it reads before it. =#
    _staleResolve!(integrator, e.reinit) || break
    _hidingReinit(integrator, pre, _reinitializedSince(integrator, since)) do
      e.ifRelations !== nothing && _update!(e.ifRelations, integrator)
      foreach(a -> _whenRelationUpdate!(a, integrator), e.relationWhens)
      foreach(c -> _update!(c, integrator), e.clusters)
    end
    moves += 1
    fill!(evaluatedAt, moves)
  end
  return changed
end

"""
The algebraic unknowns solved again within a sweep, for the bodies that
read them next. A failure is not the event's (phase 3 solves again): the
return code is put back, or its InitialFailure made every later solve
report failure; and so is the state, which the failed solve left at its
last iterate.
"""
function _staleResolve!(integrator, reinit)::Bool
  local retcode = integrator.sol.retcode
  local before::AbstractVector = copy(integrator.u)
  _resolveAlgebraics!(integrator, reinit) && return true
  _restoreRetcode!(integrator, retcode)
  copyto!(integrator.u, before)
  return false
end

#= After an event that changed the equations, the next step starts small, as
   after a when: the derivatives can jump by orders of magnitude (an ideal
   switch opening onto an inductor), and a step sized before the event
   crossed the fast transient at once, its interpolation far off inside it
   (SwitchWithArc: -4.5 A for 0.0005 A). Without algebraic rows,
   OrdinaryDiffEq's estimate (auto_dt_reset!); with them its estimate is a
   fixed 1e-6 s, so 1e-6 of the time span instead (as its DAE problems
   start), if smaller than the current step. Not for a DAE problem: DFBDF
   restarted from a smaller step ended in NaN (DAEIfReinit). =#
"""
OrdinaryDiffEq's initial step estimate after an event (auto_dt_reset!). It reads the next
tstop: at an event on the last one there is none left (MSL 4 RobotR3: a BoundsError in
first_tstop), and no step to size either.
"""
function _autoDtReset!(integrator)
  local tstops = integrator.opts.tstops
  isempty(tstops) && return nothing
  ModelingToolkit.SciMLBase.auto_dt_reset!(integrator)
  return nothing
end

"""
The length of an event's instant at `t`: crossings closer than this coincide (`_TIME_EVENT_AHEAD`).
"""
_instantLength(t::Real) = _TIME_EVENT_AHEAD * (1 + abs(t))

"""
Steps of at most this many ulps of t are degenerate (`_restartStepSize!`).
"""
const _DEGENERATE_STEP_ULPS = 8

_restartStepSize!(integrator, stall::Base.RefValue{Tuple{Float64, Int}}) = nothing

"""
    _restartStepSize!(integrator, stall)

The next step after an event that changed the equations (see above). A degenerate step (`_DEGENERATE_STEP_ULPS`)
that comes again within the same instant (`_instantLength`, at most the restart's cap; `stall` holds the instant's
start and the count) restarts at the instant's length: one an ulp long, its Rosenbrock matrix's algebraic rows were
singular to rounding and their values off by more than the hysteresis, so the callback located the next crossing
an ulp later, and the step after that again (MSL ThyristorBridge2Pulse_DC_Drive: steps of 1 to 5 ulps where the
diagonal thyristors commutate, 1.5 million of them over 7e-11 s; SimpleTriacCircuit: zero steps at one time; both
MaxIters). A single one stays (MSL Spice3 Inverter, a span of 1e-11 s, is Unstable with its zero steps restarted),
and so do longer short steps (SMPM_Braking's diode commutates in steps of 19 to 51 ulps and leaves them 4e-12 s on;
restarted at an instant's length, its crossing was located 5.7e-15 s into every step).
"""
function _restartStepSize!(integrator::OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator,
                           stall::Base.RefValue{Tuple{Float64, Int}})
  local prob = integrator.sol.prob
  prob isa ModelingToolkit.SciMLBase.AbstractODEProblem || return nothing
  local mm = integrator.f.mass_matrix
  if mm isa LinearAlgebra.UniformScaling || all(!iszero, LinearAlgebra.diag(mm))
    _autoDtReset!(integrator)
    return nothing
  end
  local span = abs(prob.tspan[2] - prob.tspan[1])
  local cap = 1.0e-6 * (isfinite(span) ? span : 1.0)
  local dt = min(abs(integrator.dt), cap)
  local t = Float64(integrator.t)
  if dt <= _DEGENERATE_STEP_ULPS * eps(t)
    local (start, count) = stall[]
    local window = min(_instantLength(start), cap)
    stall[] = start <= t <= start + window ? (start, count + 1) : (t, 1)
    last(stall[]) > 1 && (dt = min(_instantLength(t), cap))
  end
  integrator.dt = integrator.tdir * dt
  integrator.dtpropose = integrator.dt
  return nothing
end

#= An affect after which the event iteration runs: a table or time when
   (PresetTimeCallback) sets a discrete that the clusters read, and nothing
   else makes the iteration run at that instant (a JK flip-flop's K from a
   table at t = 22 reached the latches only at the next clock edge, 25). The
   iteration comes last in the callback set, so it runs in the same step. =#
struct _PendingAffect{A}
  affect!::A
  pending::Base.RefValue{Bool}
end

(a::_PendingAffect)(integrator) = (a.affect!(integrator); a.pending[] = true; nothing)

_markingPending(cb::DiffEqBase.DiscreteCallback, pending::Base.RefValue{Bool}) =
  DiffEqBase.DiscreteCallback(cb.condition, _PendingAffect(cb.affect!, pending), cb.initialize, cb.finalize,
                              cb.save_positions, cb.initializealg, cb.saved_clock_partitions,
                              cb.initialize_save_discretes)

"""
`resolve`: another callback fired (_PendingAffect); the algebraic
unknowns follow it first, as after a continuous event.
"""
_terminated(integrator) = integrator.sol.retcode == ModelingToolkit.SciMLBase.ReturnCode.Terminated

function _iterate!(e::EventIteration, integrator, resolve::Bool = false)
  #= The left limit, before the algebraic unknowns follow another callback's
     change, with the clusters' members at their held values (`_putBack!`);
     whether the callbacks before the iteration changed the state. =#
  try
    local pre = instantPre(integrator)
    foreach(c -> _putBack!(c, integrator, pre), e.clusters)
    _iterateInstant!(e, integrator, resolve, !isequal(pre, integrator.u))
  finally
    foreach(c -> _hold!(c, integrator), e.clusters)
    _endInstant!(integrator)
  end
  return nothing
end

function _iterateInstant!(e::EventIteration, integrator, resolve::Bool, affected::Bool; retried::Bool = false)
  #= The selections of the if-relations' literals below only where nothing
     else runs in the iteration: a when or a cluster body fired at every pass
     of a cycle and every repeat of the instant (a when on the comparator's
     relation counted 347 switches, OpenModelica 20). Such a model stops as
     before. =#
  local r = isempty(e.relationWhens) && isempty(e.discreteWhens) && isempty(e.clusters) ? e.ifRelations : nothing
  #= The algebraic unknowns at the event in the modes from before it. Where they have no solution, the modes the
     relations switch to can have one (an arc's current caught past its arc-out window, MSL OpenerWithArc: `v =
     V*sign(i)` has none there, the arc-off `i = Goff*v` has; where the steps land relative to the window moved with
     OrdinaryDiffEqCore 4.19): the return code and the state are put back (`_staleResolve!`) and the sweep reads the
     relations at the located crossing. The event fails if none changes, or if the solve in the new modes fails too. =#
  local unsolved::Bool = (resolve || _continuousEventFired(integrator)) && !_staleResolve!(integrator, e.reinit)
  #= The same instant again and again (a comparator with positive feedback at
     its fold, MSL SignalGenerator's fifth switch: the linear branch settled
     where it meets the saturated one, and its crossing fired at once): the
     if-relations' combination that holds clearly, beyond the hysteresis. =#
  if !retried && r !== nothing
    local (lastT, count) = e.instant[]
    e.instant[] = integrator.t == lastT ? (lastT, count + 1) : (integrator.t, 1)
    last(e.instant[]) >= 3 && _consistentIfRelations!(r, integrator, e.reinit; clear = true) &&
      return _iterateInstant!(e, integrator, false, affected; retried = true)
  end
  local forced = 0
  for n in 1:e.limit
    local changed = _sweep!(e, integrator, n == 1)
    #= A body ran terminate(): the solve ends at this event (no tstops left to
       restart a step against). =#
    _terminated(integrator) && return nothing
    #= The first sweep read pre() from before the instant, and a callback before
       it changed the state: the next iteration reads pre() from the state now
       (OpenModelica iterates while a value differs from its pre()). A gate's
       `y = pre(auxiliary_n)` moves one gate per iteration (MSL Digital). =#
    if !changed && n == 1 && affected
      forced = 1
      continue
    end
    if !changed && unsolved
      _restoreRetcode!(integrator, ModelingToolkit.SciMLBase.ReturnCode.InitialFailure)
      @error "[events] the algebraic variables could not be solved at the event at t = $(integrator.t)"
      #= The located crossings belong to this event. =#
      foreach(c -> fill!(c.crossed, 0), e.clusters)
      return nothing
    end
    if !changed
      n - forced > 1 && _restartStepSize!(integrator, e.stall)
      #= Settled: the discrete whens' change()/edge() memory takes the
         values (pre() at the next event); none of them holds here. =#
      foreach(d -> _follow!(d, integrator), e.discreteWhens)
      return nothing
    end
    #= Phase 3: the branches or the state changed; the algebraic unknowns follow. =#
    if !_resolveAlgebraics!(integrator, e.reinit)
      @error "[events] the algebraic variables could not be solved after the event at t = $(integrator.t)"
      return nothing
    end
    unsolved = false
  end
  #= The if-equation relations may switch back and forth where the loop has
     a solution on another branch (a comparator with positive feedback): one
     that holds at its own solution, then the iteration from there. Every
     relation may change: the other side of a comparator (its upper limit)
     did not move in the cycle. =#
  !retried && r !== nothing && _consistentIfRelations!(r, integrator, e.reinit) &&
    return _iterateInstant!(e, integrator, false, affected; retried = true)
  @error "[events] the event iteration did not settle in $(e.limit) sweeps at t = $(integrator.t) " *
         "(relations switching back and forth: a chattering model); the simulation stops"
  ModelingToolkit.SciMLBase.terminate!(integrator, ModelingToolkit.SciMLBase.ReturnCode.Failure)
  return nothing
end

"""
At the start, after the event callbacks' initialize functions set every
relation to its literal value at the solved initial state: the discrete
clusters set theirs and run their initial bodies. If one changed a
discrete, or an if-equation relation differs from the value the
initialization used, the algebraic unknowns are solved again (the states
kept). Then the event iteration runs at the start, pre() the initial values
(OpenModelica's storePreValues and updateDiscreteSystem at the end of the
initialization).
"""
function _initialize!(e::EventIteration, integrator)
  #= An affect that ran at the start before this (a sample(0, ...) tick:
     PeriodicCallback's initial_affect) recorded the state before it. The
     discrete whens' state (an edge latch) is that state's, and the iteration
     runs at the start: a when the tick triggers fires there (`when sample(0,
     0.5) then n = pre(n) + 1` and `when n > 0 then m = pre(n) * 10`: m = 0 from
     the start, OpenModelica; the latch took n = 1, and the when never fired). =#
  local before = _instantStateAtStart(integrator)
  #= Only the values the tick changed: the clusters initialize theirs below
     (a pulse true from the start is no edge). =#
  local moved = before === nothing ? Int[] : Int[k for k in eachindex(before) if !isequal(before[k], integrator.u[k])]
  #= A new solve (a re-simulation reuses the callbacks): no instant repeated yet, no degenerate restart. =#
  e.instant[] = (NaN, 0)
  e.stall[] = (NaN, 0)
  _initializeRelations!(e, integrator)
  local start = integrator.u
  if !isempty(moved)
    start = copy(integrator.u)
    start[moved] = before[moved]
    _INSTANT_PRE[integrator].u = start
  end
  foreach(d -> d.initialize!(start, integrator.t, integrator), e.discreteWhens)
  #= The initialization solved the discretes with pre() at the start values
     (`mode(start = Unknown, fixed = true)`); the iteration reads pre() from the
     initial values: the friction's `startBackward = pre(mode) == Stuck and sa <
     -f0_max` holds at the start (MSL LossyGearDemo2's bearing breaks away at
     t = 0, as in OpenModelica; it stayed locked). Nothing changes in most
     models: one sweep. =#
  _iterate!(e, integrator, !isempty(moved))
  return nothing
end

function _initializeRelations!(e::EventIteration, integrator)
  local changed = false
  for c in e.clusters
    _initialize!(c, integrator) && (changed = true)
  end
  local r = e.ifRelations
  r !== nothing && _bufferValues(r.buffers, integrator) != r.compiled && (changed = true)
  changed && !_startResolve!(e, integrator) && return nothing
  #= Then, on the solved state, a relation it puts on its other side takes
     that value now, not after the first step: an `initial()` condition,
     false after the initialization (its first branch was integrated for a
     step). =#
  if r !== nothing && _update!(r, integrator)
    _startResolve!(e, integrator) || return nothing
    changed = true
  end
  changed || return nothing
  #= The solve moves operands: the cluster relations literal again on the
     solved state, or they would make an event at the first step. =#
  foreach(c -> _literalBuffers!(c, integrator), e.clusters)
  return nothing
end

function _startResolve!(e::EventIteration, integrator)
  _resolveAlgebraics!(integrator, e.reinit) && return true
  @error "[events] the algebraic variables could not be solved at the start (t = $(integrator.t))"
  return false
end

"""
The callbacks that stay, and the whens the iteration takes over: those on
a relation and the discrete clusters (their continuous callbacks stay, to
locate the events) and those on a discrete condition (their callbacks go).
"""
function _splitCallbacks(callbacks)
  local kept = Any[]
  local relationWhens = RelationWhenAffect[]
  local clusters = DiscreteCluster[]
  local discreteWhens = DiscreteWhenAffect[]
  if callbacks isa DiffEqBase.CallbackSet
    for cb in callbacks.continuous_callbacks
      cb.affect! isa RelationWhenAffect && push!(relationWhens, cb.affect!)
      cb.affect! isa ClusterCrossings && push!(clusters, cb.affect!.cluster)
      push!(kept, cb)
    end
    for cb in callbacks.discrete_callbacks
      cb.affect! isa DiscreteWhenAffect ? push!(discreteWhens, cb.affect!) : push!(kept, cb)
    end
  elseif callbacks !== nothing
    push!(kept, callbacks)
  end
  return (kept, relationWhens, clusters, discreteWhens)
end

"""
A mass-matrix problem with algebraic unknowns and continuous events of its
own (MTK's: an if-equation's pure time events, NoInit): the event iteration
solves its algebraic unknowns again after such an event too, after the
step-end recorder took the left limit (withDelayRecords). Without the
iteration nothing did, but the merged continuous callback's re-solve with
the integrator's algorithm (DirectRHSGeneration.jl `_mergeableReinit`),
inside the callback: there delay() recorded no jump, and where that re-solve
is gone, the unknowns kept their old values until the next step's end (a
`time > 1.5` branch read by delay() recorded a ramp over the step).
"""
function _eventsOnAlgebraicUnknowns(problem)
  isempty(_algebraicRows(problem.f)) && return false
  local own = get(problem.kwargs, :callback, nothing)
  return own isa DiffEqBase.CallbackSet && !isempty(own.continuous_callbacks)
end

"""
    _StateDependencies(sys, algebraic)

What the positions of a problem's state are computed from, for the order of a sweep (`_sweepOrder`): the position of
each unknown of `sys`, its observed equations, and per algebraic unknown (`algebraic`: the algebraic rows) the other
positions its algebraic equations read: those of its connected part of the algebraic system, which is solved
together. A system whose equations are not one per position has no algebraic part here.
"""
struct _StateDependencies
  position::Dict{Any, Int}             # an unknown => its position in the state
  observed::Dict{Any, Any}             # an observed variable => its right-hand side
  solvedFrom::Dict{Int, Vector{Int}}   # an algebraic unknown's position => the other positions its solve reads
end

function _StateDependencies(sys, algebraic::Vector{Int})
  local unknowns::Vector{Any} = Any[Symbolics.unwrap(x) for x in ModelingToolkit.get_unknowns(sys)]
  local d::_StateDependencies = _StateDependencies(Dict{Any, Int}(x => i for (i, x) in enumerate(unknowns)),
                                                   Dict{Any, Any}(Symbolics.unwrap(eq.lhs) => Symbolics.unwrap(eq.rhs)
                                                                  for eq in ModelingToolkit.observed(sys)),
                                                   Dict{Int, Vector{Int}}())
  local eqs::AbstractVector = ModelingToolkit.equations(sys)
  (isempty(algebraic) || length(eqs) != length(unknowns)) && return d
  local isAlgebraic::Set{Int} = Set{Int}(algebraic)
  local root::Dict{Int, Int} = Dict{Int, Int}(k => k for k in algebraic)
  local find(k::Int)::Int = (while root[k] != k; k = root[k]; end; k)
  local rowReads::Dict{Int, Set{Int}} = Dict{Int, Set{Int}}()
  for r in algebraic
    local reads::Set{Int} = _stateReads(d, Any[eqs[r].lhs, eqs[r].rhs])
    rowReads[r] = reads
    local mine::Vector{Int} = Int[k for k in reads if k in isAlgebraic]
    for k in mine
      root[find(k)] = find(first(mine))
    end
  end
  local outside::Dict{Int, Set{Int}} = Dict{Int, Set{Int}}()
  for r in algebraic
    local mine::Vector{Int} = Int[k for k in rowReads[r] if k in isAlgebraic]
    isempty(mine) && continue
    union!(get!(Set{Int}, outside, find(first(mine))), (k for k in rowReads[r] if !(k in isAlgebraic)))
  end
  for k in algebraic
    d.solvedFrom[k] = collect(get(outside, find(k), Set{Int}()))
  end
  return d
end

"""
The positions of the state that the expressions `exprs` read: their unknowns, through the observed equations, and
for an algebraic unknown what its solve reads (`_StateDependencies`).
"""
function _stateReads(d::_StateDependencies, exprs::Vector{Any})::Set{Int}
  local out::Set{Int} = Set{Int}()
  local seen::Set{Any} = Set{Any}()
  local pending::Vector{Any} = Any[]
  for e in exprs
    append!(pending, Symbolics.get_variables(Symbolics.unwrap(e)))
  end
  while !isempty(pending)
    local v = pop!(pending)
    v in seen && continue
    push!(seen, v)
    local k::Union{Nothing, Int} = get(d.position, v, nothing)
    if k !== nothing
      push!(out, k)
      union!(out, get(d.solvedFrom, k, Int[]))
    elseif haskey(d.observed, v)
      append!(pending, Symbolics.get_variables(d.observed[v]))
    end
  end
  return out
end

"""
    _sweepOrder(clusters, discreteWhens, problem) -> Vector{Int}

The order of a sweep's pass over the discrete clusters (`i`) and the whens on discrete conditions (`-j`): the order
before (the clusters, then the whens), with what an item reads moved before it (MLS 8.6: the equations of an
iteration hold together with the same pre(), and OpenModelica evaluates them sorted). A cluster reads its operands and
relations, not its pre() values, through the observed equations and the algebraic block; a when its condition and
body, not under pre(). Two whens that share a variable (one assigns what the other reads or assigns) keep the
model's order: the whens of an algorithm section run as its statements do. Items that read each other (a cycle: the
phases of a circuit, a transition and its timer's when) keep the order before, and the passes settle them as before;
a group of them moves as a whole, in that order. Only what an item reads moves: a free item does not run ahead (MSL
StateGraph ShowExceptions: the composite steps' `integer()` counts ran before the steps they count).

MSL StateGraph ControlledTanks: a RadioButtonSource's `when pre(reset) then on := false` resets `start.on` in the
iteration in which the stop button's step becomes active. The cluster of the steps and transitions ran first, read
the old `on` and fired the transition out of that step, and `when fire` took its edge before the next pass corrected
it: the controller left the suspended step at once and never resumed the filling.
"""
function _sweepOrder(clusters::Vector{DiscreteCluster}, whens::Vector{DiscreteWhenAffect}, problem)::Vector{Int}
  local items::Vector{Int} = vcat(collect(eachindex(clusters)), Int[-j for j in eachindex(whens)])
  local sys = hasproperty(problem.f, :sys) ? problem.f.sys : nothing
  (sys === nothing || length(items) <= 1 || !applicable(getStatesAsSymbols, problem.f)) && return items
  local deps::_StateDependencies = _StateDependencies(sys, _algebraicRows(problem.f))
  local byName::Dict{String, Int} = Dict{String, Int}(string(s) => i for (i, s) in enumerate(getStatesAsSymbols(problem.f)))
  local positions(names::Vector{String})::Vector{Int} = Int[byName[n] for n in names if haskey(byName, n)]
  local reads::Vector{Set{Int}} = Vector{Set{Int}}(undef, length(items))
  local writes::Vector{Set{Int}} = Vector{Set{Int}}(undef, length(items))
  for (k, item) in enumerate(items)
    if item > 0
      local c::DiscreteCluster = clusters[item]
      reads[k] = _stateReads(deps, Any[c.reads[1:c.nOperands]; c.reads[(c.nOperands + c.nPre + 1):end]])
      writes[k] = Set{Int}(i for i in c.memberIndex if i != 0)
    else
      local w::DiscreteWhenAffect = whens[-item]
      local read::Vector{Int} = positions(w.reads)
      reads[k] = Set{Int}(read)
      foreach(i -> union!(reads[k], get(deps.solvedFrom, i, Int[])), read)
      writes[k] = Set{Int}(positions(w.writes))
    end
  end
  local n::Int = length(items)
  #= before[b]: the items that run before b, in the order before =#
  local before::Vector{Vector{Int}} = [Int[] for _ in 1:n]
  for b in 1:n, a in 1:n
    a == b && continue
    local runsBefore::Bool = items[a] < 0 && items[b] < 0 ?
      a < b && !(isdisjoint(writes[a], reads[b]) && isdisjoint(writes[a], writes[b]) && isdisjoint(reads[a], writes[b])) :
      !isdisjoint(writes[a], reads[b])
    runsBefore && push!(before[b], a)
  end
  local group::Vector{Int} = _cycleGroups(before)
  local placed::Vector{Bool} = fill(false, n)
  local order::Vector{Int} = Int[]
  local place!(g::Int)::Nothing = begin
    local members::Vector{Int} = Int[k for k in 1:n if group[k] == g && !placed[k]]
    isempty(members) && return nothing
    foreach(k -> placed[k] = true, members)
    for k in members, a in before[k]
      group[a] == g || placed[a] || place!(group[a])
    end
    append!(order, items[members])
    return nothing
  end
  foreach(k -> place!(group[k]), 1:n)
  return order
end

"""
    _cycleGroups(before) -> Vector{Int}

Per item the group of items that run before each other in a cycle (a strongly connected component of the graph
`before[b]`: the items that run before b), numbered from 1; an item outside every cycle is a group of its own.
"""
function _cycleGroups(before::Vector{Vector{Int}})::Vector{Int}
  local n::Int = length(before)
  #= reaches[a, b]: a runs (transitively) before b =#
  local reaches::BitMatrix = falses(n, n)
  for b in 1:n
    local pending::Vector{Int} = copy(before[b])
    while !isempty(pending)
      local a::Int = pop!(pending)
      reaches[a, b] && continue
      reaches[a, b] = true
      append!(pending, before[a])
    end
  end
  local group::Vector{Int} = zeros(Int, n)
  local count::Int = 0
  for a in 1:n
    group[a] == 0 || continue
    count += 1
    group[a] = count
    for b in (a + 1):n
      group[b] == 0 && reaches[a, b] && reaches[b, a] && (group[b] = count)
    end
  end
  return group
end

"""
    withRelationRefresh(callbacks, problem, hSym, entries) -> callbacks

Add the event iteration over the buffered relations: the if-equation
relations `entries` = `(ifCond, crossing function, scale)`, and the whens on
a relation and the discrete clusters among `callbacks` (their affects are
`RelationWhenAffect`s and `ClusterCrossings`). A DiscreteCallback checks them
after every step and, where one disagrees with the state, a continuous event
fired or another discrete callback fired (a table's time event), iterates as
described above. `hSym` is the hysteresis parameter H.

The whens on discrete conditions among `callbacks` (`DiscreteWhenAffect`s)
move into the iteration: a relation's when body can change the discrete
they watch, and their bodies can move a relation, within one event.
"""
function withRelationRefresh(callbacks, problem, hSym::Symbol, entries::Vector)
  local (kept, relationWhens, clusters, discreteWhens) = _splitCallbacks(callbacks)
  #= A when whose operands the model's own system lacks never fired, without
     a word. (At run time the other modes of a variable-structure model run
     with systems that lack them: there such a when waits, _whenFunctions.) =#
  local ownSys = hasproperty(problem.f, :sys) ? problem.f.sys : nothing
  if ownSys !== nothing && !isempty(relationWhens)
    local byName = _variablesByPlainName(ownSys)
    for a in relationWhens
      local absent = filter(n -> !haskey(byName, n), a.names)
      isempty(absent) || OMBackend.unsupported("a when on a relation whose operands the system does not keep", join(absent, ", "))
    end
  end
  local ifRelations = _ifRelations(problem, hSym, entries)
  #= Discrete whens alone are iterated too: one can read an algebraic unknown that
     another callback in the same step made stale (a source switching a logic
     gate's inputs, MSL Digital Adder4), and only the iteration solves them again
     before it reads them. =#
  ifRelations === nothing && isempty(relationWhens) && isempty(clusters) && isempty(discreteWhens) &&
    !_eventsOnAlgebraicUnknowns(problem) && return callbacks
  local n = (ifRelations === nothing ? 0 : length(ifRelations)) + length(relationWhens) + length(discreteWhens) +
            sum((length(c.rel) + length(c.members) for c in clusters); init = 0)
  local reinit = any(c -> c.table, clusters) ? tableClusterInitAlg() : nothing
  local e = EventIteration(ifRelations, relationWhens, clusters, discreteWhens,
                           _sweepOrder(clusters, discreteWhens, problem), _eventIterationLimit(n), reinit,
                           Ref((NaN, 0)), Ref((NaN, 0)))
  local pending = Ref(false)
  kept = Any[cb isa DiffEqBase.DiscreteCallback ? _markingPending(cb, pending) : cb for cb in kept]
  local cb = DiffEqBase.DiscreteCallback((u, t, integrator) -> pending[] || _needsIteration(e, integrator),
                                         integrator -> (local p = pending[]; pending[] = false; _iterate!(e, integrator, p));
                                         #= A solve that ended in an exception can have left
                                            `pending` set: a spurious iteration in the next. =#
                                         initialize = (c, u, t, integrator) -> (pending[] = false; _initialize!(e, integrator)),
                                         #= The state after the iteration, at the event's time
                                            (the when callbacks saved the left limit). The
                                            iteration solves the algebraic variables itself. =#
                                         save_positions = (false, true),
                                         initializealg = ModelingToolkit.SciMLBase.NoInit())
  return _joinCallbacks(kept..., cb)
end

"""
    setZCHysteresis!(problem, hSym, reltol)

Set the hysteresis parameter H of the event crossing functions for a solve
with relative tolerance `reltol` over the problem's time span (`_hysteresis`,
OpenModelica's tolZC). Nothing for a problem without it.
"""
function setZCHysteresis!(problem, hSym::Symbol, reltol)
  local SII = ModelingToolkit.SymbolicIndexingInterface
  local present = try
    SII.parameter_index(problem, hSym) !== nothing
  catch err
    #= A problem without symbolic indexing (no system). =#
    OMBackend._fallback(err, :zcHysteresisIndex)
    false
  end
  present || return problem
  SII.setp(problem, hSym)(problem, _hysteresis(reltol, _timeSpan(problem.tspan)))
  return problem
end

"""
Callbacks that change parameters at initialization (the literal relation
values) make the integrator save t0 twice; interpolating at t0 then uses
the zero-length interval and returns NaN. The solution starts with the
post-initialization values, as a Modelica result does.
"""
function dropPreInitializationPoint!(sol)
  (length(sol.t) >= 2 && sol.t[1] == sol.t[2]) || return sol
  local n = length(sol.t)
  deleteat!(sol.t, 1)
  length(sol.u) == n && deleteat!(sol.u, 1)
  local k = hasproperty(sol, :k) ? sol.k : nothing
  (k isa AbstractVector && length(k) == n) && deleteat!(k, 1)
  local interp = hasproperty(sol, :interp) ? sol.interp : nothing
  if interp !== nothing && hasproperty(interp, :alg_choice)
    local ac = interp.alg_choice
    (ac isa AbstractVector && length(ac) == n) && deleteat!(ac, 1)
  end
  return sol
end
