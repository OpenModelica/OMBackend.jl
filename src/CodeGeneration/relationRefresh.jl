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
      with an error, as OpenModelica does. =#

#= A getter for a crossing function or scale; a constant (not symbolic) is a
   Real that is not a Symbolics.Num. Not SII.getu: it takes an expression over
   observed variables for a parameter-only one (GetParameterObserved), and the
   generated function then reads the unknowns as undefined globals (the V6
   cylinder rig, `2*v_rel` with v_rel observed). A single variable works, and
   so does building the observed function directly. =#
function _valueGetter(problem, @nospecialize(ex))
  local v = Symbolics.unwrap(ex)
  (v isa Real && !(v isa Symbolics.Num)) && return (integrator -> Float64(v))
  local f = _buildObservedFunction(problem.f.sys, v)
  return integrator -> f(integrator.u, integrator.p, integrator.t)
end

#= The hysteresis H for a relative tolerance and a solve's time span, as
   OpenModelica's tolZC = 1e-4 * min(stepSize, tolerance), its output step
   the span over 500 intervals (its default): an absolute 1e-4 * reltol is
   1e-7 at the default tolerance, the whole span of the MSL Spice3 examples,
   and no relation on time switched before their stop time (a pulse source
   stayed at 0 V). The if-equation relations read it from their parameter
   (setZCHysteresis! sets it per solve), the whens from the integrator. =#
_hysteresis(reltol, span::Real = Inf) =
  1.0e-4 * max(min(Float64(first(reltol)), abs(Float64(span)) / 500), 1.0e-12)
_timeSpan(tspan) = Float64(last(tspan)) - Float64(first(tspan))
_hysteresisFromTolerance(integrator) = _hysteresis(integrator.opts.reltol, _timeSpan(integrator.sol.prob.tspan))

#= A relation's value by the hysteresis rule, from its buffered value `old`:
   true when zc <= 0 (true while zc <= eps, becomes true at zc <= -eps). =#
_hysteresisRule(zc, eps, old::Bool) = old ? zc <= eps : zc <= -eps

#= Sweeps before the event iteration gives up: at least 20, and 3 per
   relation or when (a chain through algebraic variables takes one per
   link). =#
_eventIterationLimit(n::Integer) = max(20, 3 * n)

#= The if-equation relations: their buffers (the ifCond parameters), and
   functions that read each one's crossing function and scale. =#
struct IfRelations
  buffers::Any                         # integrator or problem -> the ifConds
  setBuffer::Vector{Any}               # per relation: (integrator, value) -> nothing
  crossing::Vector{Any}                # per relation: integrator -> zc
  scale::Vector{Any}                   # per relation: integrator -> scale
  hysteresis::Any                      # integrator -> H
  compiled::Vector{Float64}            # the ifConds the initialization used
end

Base.length(r::IfRelations) = length(r.crossing)

#= The relations `entries` = `(ifCond, crossing function, scale)` of
   `problem`; nothing when there are none or they cannot be read. =#
function _ifRelations(problem, hSym::Symbol, entries::Vector)
  isempty(entries) && return nothing
  local SII = ModelingToolkit.SymbolicIndexingInterface
  local syms = Symbol[e[1] for e in entries]
  try
    local buffers = SII.getp(problem, syms)
    return IfRelations(buffers, Any[SII.setp(problem, s) for s in syms],
                       Any[_valueGetter(problem, e[2]) for e in entries],
                       Any[_valueGetter(problem, e[3]) for e in entries],
                       SII.getp(problem, hSym), _bufferValues(buffers, problem))
  catch err
    OMBackend._fallback(err, :ifRelationsFromProblem; impact = :result)
    @warn "[events] the if-equation relations cannot be read from the problem; they are not iterated" exception = err
    return nothing
  end
end

#= A copy: setting a buffer changes the parameters the getter reads. =#
_bufferValues(buffers, integrator) = Float64[v for v in buffers(integrator)]

function _ruled(r::IfRelations, integrator, k::Int, old::Bool)
  local eps = r.hysteresis(integrator) * r.scale[k](integrator)
  return _hysteresisRule(r.crossing[k](integrator), eps, old)
end

function _inconsistent(r::IfRelations, integrator)
  local values = r.buffers(integrator)
  return any(k -> _ruled(r, integrator, k, values[k] > 0.5) != (values[k] > 0.5), eachindex(values))
end

#= Phase 1 of a sweep for the if-equation relations: each buffer from the
   state. Whether one changed. =#
function _update!(r::IfRelations, integrator)
  local changed = false
  for (k, value) in enumerate(_bufferValues(r.buffers, integrator))
    local old = value > 0.5
    local new = _ruled(r, integrator, k, old)
    new == old && continue
    r.setBuffer[k](integrator, new ? 1.0 : 0.0)
    changed = true
  end
  return changed
end

#= When-equations on a single relation (codeGen.jl `_emitRelationWhen`).
   The relation keeps its value between events in a buffer (MLS 8.5), set
   literally at the start; its crossing function is shifted by the same
   hysteresis as the if-equation branches, relative to that value, so a
   relation that starts at its threshold becomes true right after the start
   and one that crosses back is seen. The when body runs only when the
   relation becomes true (MLS 8.3.5). The operands are read by name, so they
   may be unknowns, observed variables or parameters. =#

_plainVariableName(@nospecialize(v)) = replace(string(v), "(t)" => "", "var\"" => "", "\"" => "")

"""
    namedValueFunctions(sys, names) -> Vector or nothing

For each variable name (as the Modelica cref prints), a function `(u, p, t)`
that returns its value: an unknown, an observed variable or a parameter.
Nothing when the system lacks one of them (a variable-structure model runs
the callbacks of every mode; a mode's callbacks see the other modes'
systems).
"""
function namedValueFunctions(sys, names::Vector{String})
  sys === nothing && return nothing
  local byName = Dict{String, Any}()
  for v in ModelingToolkit.unknowns(sys)
    byName[_plainVariableName(v)] = v
  end
  for eq in ModelingToolkit.observed(sys)
    byName[_plainVariableName(eq.lhs)] = eq.lhs
  end
  for p in ModelingToolkit.parameters(sys)
    byName[_plainVariableName(p)] = p
  end
  all(n -> haskey(byName, n), names) || return nothing
  return [ModelingToolkit.build_explicit_observed_function(sys, byName[n]) for n in names]
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
  eval::E                    # (t, values...) -> (zc, scale); true when zc < 0 (strict) or zc <= 0
  strict::Bool
  body!::B                   # (integrator, pre) -> nothing; pre(v) read from the vector `pre`
  sys::Any                   # the system `fns` were built for (:unresolved at first)
  fns::Any                   # the value functions, or nothing (another mode's system)
  rel::Bool                  # the relation's buffer
  last::Bool                 # the condition when the whens last fired
end

#= The value functions for the integrator's system; nothing while it lacks
   the names (another mode). Resolved once per system (`a.sys`, initially
   :unresolved); becoming active, the relation starts literal and does not
   fire (a when true at the start is not an edge). =#
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

#= `eval` and `body!` are generated code, which a variable-structure model
   may compile while it runs (after the solve started): call them in the
   latest world. =#
function _whenValues(a::RelationWhenAffect, u, t, integrator)
  local fns = a.fns
  local p = integrator.p
  return Base.invokelatest(a.eval, t, (f(u, p, t) for f in fns)...)
end

function _whenLiteral(a::RelationWhenAffect, u, t, integrator)
  local (zc, _) = _whenValues(a, u, t, integrator)
  return a.strict ? zc < 0 : zc <= 0
end

#= The relation's value by the hysteresis rule at the integrator's state. =#
function _whenRuled(a::RelationWhenAffect, integrator)
  local (zc, scale) = _whenValues(a, integrator.u, integrator.t, integrator)
  return _hysteresisRule(zc, _hysteresisFromTolerance(integrator) * scale, a.rel)
end

function _whenInconsistent(a::RelationWhenAffect, integrator)
  _whenFunctions(a, integrator) === nothing && return false
  return _whenRuled(a, integrator) != a.rel
end

#= Phase 1 of a sweep: the buffer from the state. Whether it changed. =#
function _whenRelationUpdate!(a::RelationWhenAffect, integrator)
  _whenFunctions(a, integrator) === nothing && return false
  local new = _whenRuled(a, integrator)
  new == a.rel && return false
  a.rel = new
  return true
end

#= Phase 2: the body, when the condition became true since the whens last
   fired; `pre` is the state before the sweep's bodies. Whether it fired. =#
function _whenFire!(a::RelationWhenAffect, integrator, pre)
  a.fns === nothing && return false
  local fire = a.rel && !a.last
  a.last = a.rel
  fire && Base.invokelatest(a.body!, integrator, pre)
  return fire
end

#= The crossing only locates the event: it is handled after the step, with
   every other relation (the event iteration). The state is not changed. =#
(a::RelationWhenAffect)(integrator) = (_derivativeDiscontinuity!(integrator, false); nothing)

"""
    relationWhenCallback(names, eval, strict, body!) -> ContinuousCallback

A when-equation on one relation. `eval(t, values...)` returns `(zc, scale)`
for the values of `names`: the relation is true when zc < 0 (`strict`) or
zc <= 0, and `scale` sizes the hysteresis. `body!(integrator, pre)` is the
when body (`pre(v)` read from `pre`).
"""
function relationWhenCallback(names::Vector{String}, eval, strict::Bool, body!)
  local a = RelationWhenAffect(names, eval, strict, body!, :unresolved, nothing, false, false)
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
end

(d::DiscreteWhenAffect)(integrator) = d.affect!(integrator)

const _NO_WHEN_INITIALIZE = (u, t, integrator) -> nothing

"""
    discreteWhenCallback(condition, affect!, initialize! = _NO_WHEN_INITIALIZE) -> DiscreteCallback

A when on a discrete condition, checked after every step. `initialize!` runs at
the start of every solve (a reinit! included). Its change()/edge() memory
follows the values only after a step no other callback changed: an earlier
callback in the same step can change a discrete without solving the algebraic
unknowns again, and `edge(b) and c` would lose b's edge on a stale c (it fires
a step later instead).
"""
discreteWhenCallback(condition, affect!, initialize! = _NO_WHEN_INITIALIZE) =
  DiffEqBase.DiscreteCallback((u, t, integrator) -> condition(u, t, integrator, !_changedThisStep(integrator)),
                              DiscreteWhenAffect(condition, affect!, initialize!);
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
  limit::Int
  reinit::Any
end

#= Whether a callback changed the state in the step that just ended: a
   continuous event, or an earlier discrete callback (the integrator's flag,
   `derivative_discontinuity` since SciMLBase 3, `u_modified` before). =#
_changedThisStep(integrator) =
  _continuousEventFired(integrator) ||
  (hasproperty(integrator, :derivative_discontinuity) && integrator.derivative_discontinuity) ||
  (hasproperty(integrator, :u_modified) && integrator.u_modified)

#= Whether the when fires; where it does not, its change()/edge() memory
   takes the values. =#
_holds(d::DiscreteWhenAffect, integrator) = d.condition(integrator.u, integrator.t, integrator)

#= Within a sweep: the change()/edge() memory stays (the condition's
   `_follow` false). It follows the values once the iteration settles. =#
_holdsInSweep(d::DiscreteWhenAffect, integrator) = d.condition(integrator.u, integrator.t, integrator, false)

_follow!(d::DiscreteWhenAffect, integrator) = (_holds(d, integrator); nothing)

#= Whether a continuous callback fired in the step that just ended: the
   integrators of OrdinaryDiffEqCore and Sundials record it in
   `event_last_time` (0 when none) before they run the discrete callbacks. =#
_continuousEventFired(integrator) =
  hasproperty(integrator, :event_last_time) && integrator.event_last_time != 0

#= After a step: whether a continuous event fired, a relation disagrees with
   the state, or a discrete when's condition holds. =#
function _needsIteration(e::EventIteration, integrator)
  _continuousEventFired(integrator) && return true
  e.ifRelations !== nothing && _inconsistent(e.ifRelations, integrator) && return true
  any(a -> _whenInconsistent(a, integrator), e.relationWhens) && return true
  any(c -> _inconsistent(c, integrator), e.clusters) && return true
  return any(d -> _holds(d, integrator), e.discreteWhens)
end

#= Phases 1 and 2 of a sweep (see the top of the file). Whether anything
   changed. =#
function _sweep!(e::EventIteration, integrator)
  #= 1. Every relation on this state. =#
  local changed = e.ifRelations !== nothing && _update!(e.ifRelations, integrator)
  for a in e.relationWhens
    _whenRelationUpdate!(a, integrator) && (changed = true)
  end
  local relPre = [copy(c.rel) for c in e.clusters]
  for c in e.clusters
    _update!(c, integrator) && (changed = true)
  end
  #= 2. The bodies, with pre() from the state before them. A relation when's
     discretes, or a cluster that no re-solve moves (`coupled` false, a pulse
     switching a circuit), can move the algebraic unknowns: they are solved
     again before a coupled cluster or a discrete when reads them in this
     sweep. =#
  local pre = copy(integrator.u)
  local clusterPre = [_preValues(c, _readValues(c, integrator)) for c in e.clusters]
  for pass in 1:e.limit
    local stale = false
    for a in e.relationWhens
      _whenFire!(a, integrator, pre) && (changed = true; stale = true)
    end
    for (i, c) in enumerate(e.clusters)
      c.coupled && stale && (_staleResolve!(integrator, e.reinit); stale = false)
      _solveMixedSystem!(c, integrator, clusterPre[i], relPre[i], e.reinit) || continue
      changed = true
      c.coupled || (stale = true)
    end
    stale && !isempty(e.discreteWhens) && _staleResolve!(integrator, e.reinit)
    local before = copy(integrator.u)
    local fired = false
    for d in e.discreteWhens
      _holdsInSweep(d, integrator) || continue
      d.affect!(integrator, pre)
      changed = fired = true
    end
    (fired && !isequal(integrator.u, before)) || break
    #= A discrete when changed a discrete: the relations and the bodies read the new
       value in this same iteration (MLS 8.6: its equations hold together, with the same
       pre()). StateGraph's transition has `when enableFire then t_start = time` and
       `fire = enableFire and time >= t_start + waitTime`: fire is false where the timer
       starts; with the old t_start it fired at once. The algebraic unknowns follow first.
       A when with edge() or an edge latch does not fire again: its affect took the
       values. A when that fired in an earlier pass on a value this pass corrects is not
       undone. =#
    _staleResolve!(integrator, e.reinit) || break
    e.ifRelations !== nothing && _update!(e.ifRelations, integrator)
    foreach(a -> _whenRelationUpdate!(a, integrator), e.relationWhens)
    foreach(c -> _update!(c, integrator), e.clusters)
  end
  return changed
end

#= The algebraic unknowns solved again within a sweep, for the bodies that
   read them next. A failure is not the event's (phase 3 solves again): the
   return code is put back, or its InitialFailure made every later solve
   report failure. =#
function _staleResolve!(integrator, reinit)
  local retcode = integrator.sol.retcode
  _resolveAlgebraics!(integrator, reinit) && return true
  _restoreRetcode!(integrator, retcode)
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
_restartStepSize!(integrator) = nothing
function _restartStepSize!(integrator::OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator)
  local prob = integrator.sol.prob
  prob isa ModelingToolkit.SciMLBase.AbstractODEProblem || return nothing
  local mm = integrator.f.mass_matrix
  if mm isa LinearAlgebra.UniformScaling || all(!iszero, LinearAlgebra.diag(mm))
    ModelingToolkit.SciMLBase.auto_dt_reset!(integrator)
    return nothing
  end
  local span = abs(prob.tspan[2] - prob.tspan[1])
  local dt = min(abs(integrator.dt), 1.0e-6 * (isfinite(span) ? span : 1.0))
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

#= `resolve`: another callback fired (_PendingAffect); the algebraic
   unknowns follow it first, as after a continuous event. =#
function _iterate!(e::EventIteration, integrator, resolve::Bool = false)
  if (resolve || _continuousEventFired(integrator)) && !_resolveAlgebraics!(integrator, e.reinit)
    @error "[events] the algebraic variables could not be solved at the event at t = $(integrator.t)"
    #= The located crossings belong to this event. =#
    foreach(c -> fill!(c.crossed, false), e.clusters)
    return nothing
  end
  for n in 1:e.limit
    if !_sweep!(e, integrator)
      n > 1 && _restartStepSize!(integrator)
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
  end
  @error "[events] the event iteration did not settle in $(e.limit) sweeps at t = $(integrator.t) " *
         "(relations switching back and forth: a chattering model); the simulation stops"
  ModelingToolkit.SciMLBase.terminate!(integrator, ModelingToolkit.SciMLBase.ReturnCode.Failure)
  return nothing
end

#= At the start, after the event callbacks' initialize functions set every
   relation to its literal value at the solved initial state: the discrete
   clusters set theirs and run their initial bodies. If one changed a
   discrete, or an if-equation relation differs from the value the
   initialization used, the algebraic unknowns are solved again (the states
   kept). =#
function _initialize!(e::EventIteration, integrator)
  _initializeRelations!(e, integrator)
  #= The discrete whens' state (an edge latch) from the initialized state. =#
  foreach(d -> d.initialize!(integrator.u, integrator.t, integrator), e.discreteWhens)
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

#= The callbacks that stay, and the whens the iteration takes over: those on
   a relation and the discrete clusters (their continuous callbacks stay, to
   locate the events) and those on a discrete condition (their callbacks go). =#
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
  local ifRelations = _ifRelations(problem, hSym, entries)
  #= Discrete whens alone are iterated too: one can read an algebraic unknown that
     another callback in the same step made stale (a source switching a logic
     gate's inputs, MSL Digital Adder4), and only the iteration solves them again
     before it reads them. =#
  ifRelations === nothing && isempty(relationWhens) && isempty(clusters) && isempty(discreteWhens) && return callbacks
  local n = (ifRelations === nothing ? 0 : length(ifRelations)) + length(relationWhens) + length(discreteWhens) +
            sum((length(c.rel) + length(c.members) for c in clusters); init = 0)
  local reinit = any(c -> c.table, clusters) ? tableClusterInitAlg() : nothing
  local e = EventIteration(ifRelations, relationWhens, clusters, discreteWhens, _eventIterationLimit(n), reinit)
  local pending = Ref(false)
  kept = Any[cb isa DiffEqBase.DiscreteCallback ? _markingPending(cb, pending) : cb for cb in kept]
  local cb = DiffEqBase.DiscreteCallback((u, t, integrator) -> pending[] || _needsIteration(e, integrator),
                                         integrator -> (local p = pending[]; pending[] = false; _iterate!(e, integrator, p));
                                         initialize = (c, u, t, integrator) -> _initialize!(e, integrator),
                                         #= The state after the iteration, at the event's time
                                            (the when callbacks saved the left limit). The
                                            iteration solves the algebraic variables itself. =#
                                         save_positions = (false, true),
                                         initializealg = ModelingToolkit.SciMLBase.NoInit())
  return DiffEqBase.CallbackSet(kept..., cb)
end

#= Integer, Boolean and enumeration variables hold integral values. They
   are unknowns with der(v) = 0, and a Rosenbrock or BDF step can leave
   round-off in them (pivoting in the linear solve mixes rows):
   3 becomes 3.0000000000000004, and change(v) or v == 3 would see it. =#

"""
    withIntegralDiscretes(callbacks, problem, names) -> callbacks

Add a DiscreteCallback ahead of the other discrete callbacks (before the event
iteration reads them), that rounds the unknowns `names` back to integers
after a step that left round-off in them.
"""
function withIntegralDiscretes(callbacks, problem, names::Vector{String})
  local states = getStatesAsSymbols(problem.f)
  local idx = Int[k for k in indexin(Symbol.(names), states) if k !== nothing]
  isempty(idx) && return callbacks
  local drifted = (u, t, integrator) -> any(k -> u[k] != round(u[k]), idx)
  local round! = function (integrator)
    for k in idx
      integrator.u[k] = round(integrator.u[k])
    end
    return nothing
  end
  local cb = DiffEqBase.DiscreteCallback(drifted, round!; save_positions = (false, false),
                                         initializealg = ModelingToolkit.SciMLBase.NoInit())
  return callbacks === nothing ? DiffEqBase.CallbackSet(cb) : DiffEqBase.CallbackSet(cb, callbacks)
end

#= The re-solve of the algebraic unknowns at an event: BrownFullBasicInit
   at its own tolerance (1e-10 on the residual: relations are decided on the
   solved values; a looser one skips the solve where a switched diode's
   equations barely change the residual, and its s keeps the wrong sign).
   Where that is out of reach (large values, a switched Jacobian near
   singular: a thyristor bridge's commutation stops at 4e-9 with values of
   1e6), the result stands if it lowered the residual to within the solve's
   abstol, as the initialization's (defaultInitializeKwargs). A type,
   because a callback's reinitializealg is fixed when the code is
   generated. =#
struct EventReinit <: ModelingToolkit.SciMLBase.DAEInitializationAlgorithm end

#= SciMLBase's fallback takes any algorithm for any integrator,
   OrdinaryDiffEqCore's any for its own. =#
DiffEqBase.initialize_dae!(integrator::ModelingToolkit.SciMLBase.DEIntegrator, ::EventReinit) =
  _eventReinit!(integrator)
DiffEqBase.initialize_dae!(integrator::OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator, ::EventReinit) =
  _eventReinit!(integrator)

#= `alg`: BrownFullBasicInit at its own tolerance, or the table-cluster
   models' (tableClusterInitAlg); the same fallback for both (the MSL QS
   IMC_Transformer's switching event at t = 2 stopped at a residual within
   the solve's abstol but above the table algorithm's 1e-8). =#
function _eventReinit!(integrator, alg = DiffEqBase.BrownFullBasicInit())
  local InitialFailure = ModelingToolkit.SciMLBase.ReturnCode.InitialFailure
  local before = integrator.sol.retcode
  local u0 = copy(integrator.u)
  try
    DiffEqBase.initialize_dae!(integrator, alg)
  catch err
    OMBackend._fallback(err, :eventInitializeDae)
    #= A method of the default polyalgorithm can throw (NonlinearSolve's
       MoreTrustRegion descent, `restructure(x, nothing)`, on the MSL Spice3
       Inverter's switching MOSFETs): Newton with finite differences instead,
       from the state before; a failure if that throws too. =#
    copyto!(integrator.u, u0)
    try
      DiffEqBase.initialize_dae!(integrator, tableClusterInitAlg())
    catch err2
      OMBackend._fallback(err2, :eventInitializeDaeNewton; impact = :result)
      copyto!(integrator.u, u0)
      _restoreRetcode!(integrator, InitialFailure)
    end
  end
  (integrator.sol.retcode == InitialFailure && before != InitialFailure) || return nothing
  local residual = _maxAlgebraicResidual(integrator, integrator.u)
  ((residual <= _solveAbstol(integrator) || _atRoundoffFloor(integrator, integrator.u)) &&
   residual < _maxAlgebraicResidual(integrator, u0)) || return nothing
  @debug "[events] algebraic re-solve kept at the solve's abstol" t = integrator.t residual
  return _restoreRetcode!(integrator, before)
end

#= Whether every algebraic row's residual at u is within the solve's abstol
   plus its round-off floor, 1e3 eps times the size of its terms
   (sum_j |J_ij u_j|, from the problem's Jacobian; false without one). A
   switched ideal element's rows carry terms of 1e11 (the MSL QS
   IMC_Transformer's commuting switch: s = -9.1e5 against Goff = 1e-5), and
   their residual stays near 1e-5 whatever the solver does. =#
function _atRoundoffFloor(integrator, u)
  local f = integrator.f
  (hasproperty(f, :jac) && f.jac !== nothing && hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing) ||
    return false
  local rows = _algebraicRows(f)
  (isempty(rows) || !ModelingToolkit.SciMLBase.isinplace(f)) && return false
  local J = similar(f.jac_prototype, eltype(u))
  local r = similar(u)
  try
    f.jac(J, u, integrator.p, integrator.t)
    f(r, u, integrator.p, integrator.t)
  catch e
    OMBackend._fallback(e, :roundoffFloorJacobian)
    return false
  end
  local scale = abs.(J) * abs.(u)
  local abstol = _solveAbstol(integrator)
  return all(k -> abs(r[k]) <= abstol + 1.0e3 * eps(Float64) * scale[k], rows)
end

#= The solve's absolute tolerance, the smallest of a per-component one. =#
_solveAbstol(integrator) = (local a = integrator.opts.abstol; a isa Number ? a : minimum(a))

#= The largest residual of the algebraic equations of a mass-matrix ODE at
   u (Inf for other problems). =#
function _maxAlgebraicResidual(integrator, u)
  local f = integrator.f
  local rows = _algebraicRows(f)
  (isempty(rows) || !ModelingToolkit.SciMLBase.isinplace(f)) && return Inf
  local r = similar(u)
  f(r, u, integrator.p, integrator.t)
  return maximum(k -> abs(r[k]), rows)
end

#= Solve the algebraic unknowns again with `alg` (nothing: EventReinit),
   the differential states kept (a pure ODE has none). A DAEFunction (DFBDF,
   IDA) has no mass matrix; its algebraic variables are the ones
   `differential_vars` excludes. Whether it succeeded. =#
function _resolveAlgebraics!(integrator, alg = nothing)
  local f = integrator.f
  if hasproperty(f, :mass_matrix)
    local mm = f.mass_matrix
    (mm isa LinearAlgebra.UniformScaling || all(!iszero, LinearAlgebra.diag(mm))) && return true
  end
  alg === nothing ? DiffEqBase.initialize_dae!(integrator, EventReinit()) : _eventReinit!(integrator, alg)
  _derivativeDiscontinuity!(integrator, true)
  return integrator.sol.retcode != ModelingToolkit.SciMLBase.ReturnCode.InitialFailure
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

#= Callbacks that change parameters at initialization (the literal relation
   values) make the integrator save t0 twice; interpolating at t0 then uses
   the zero-length interval and returns NaN. The solution starts with the
   post-initialization values, as a Modelica result does. =#
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
