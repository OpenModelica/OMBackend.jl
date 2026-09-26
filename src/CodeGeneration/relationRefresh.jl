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
   disagrees the event is handled in sweeps, as OpenModelica's
   updateDiscreteSystem:
   1. every relation is evaluated on the same state;
   2. the whens on a relation that became true run their bodies, in order,
      with pre() read from the state before the sweep; then the whens on
      discrete conditions (a changed discrete, an algorithm section's
      inputs), each checked just before it runs, in the order of the model;
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
  local f = ModelingToolkit.build_explicit_observed_function(problem.f.sys, v)
  return integrator -> f(integrator.u, integrator.p, integrator.t)
end

#= The hysteresis H for a relative tolerance (OpenModelica's tolZC). The
   if-equation relations read it from their parameter (setZCHysteresis!
   sets it per solve), the whens from the integrator's tolerance. =#
_hysteresis(reltol) = 1.0e-4 * max(Float64(first(reltol)), 1.0e-12)
_hysteresisFromTolerance(integrator) = _hysteresis(integrator.opts.reltol)

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
   every other relation (the event iteration). =#
(a::RelationWhenAffect)(integrator) = nothing

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
  local initialize = function (c, u, t, integrator)
    a.sys = :unresolved; a.fns = nothing
    _whenFunctions(a, integrator)
    return nothing
  end
  #= RightRootFind: the event lands just past the root, where the rule sees
     the crossing. It saves the left limit; the event iteration saves the
     right one. The initialize function leaves the modified flag alone (the
     default one clears what another callback set). =#
  return DiffEqBase.ContinuousCallback(condition, a, a;
                                       initialize = initialize, rootfind = ModelingToolkit.SciMLBase.RightRootFind,
                                       save_positions = (true, false))
end

"""
    DiscreteWhenAffect

A when on a discrete condition (a changed discrete, a Boolean, an algorithm
section's inputs): its condition and body, as a DiscreteCallback's. Where a
model has buffered relations, the event iteration (`withRelationRefresh`)
runs it, after the whens on a relation; otherwise its DiscreteCallback does.
"""
struct DiscreteWhenAffect{C, A}
  condition::C               # (u, t, integrator) -> Bool
  affect!::A                 # integrator -> nothing
end

(d::DiscreteWhenAffect)(integrator) = d.affect!(integrator)

"""
    discreteWhenCallback(condition, affect!) -> DiscreteCallback

A when on a discrete condition, checked after every step.
"""
discreteWhenCallback(condition, affect!) =
  DiffEqBase.DiscreteCallback(condition, DiscreteWhenAffect(condition, affect!); save_positions = (true, true))

#= What the event iteration updates. =#
struct EventIteration
  ifRelations::Union{Nothing, IfRelations}
  relationWhens::Vector{RelationWhenAffect}
  discreteWhens::Vector{DiscreteWhenAffect}
  limit::Int
end

_holds(d::DiscreteWhenAffect, integrator) = d.condition(integrator.u, integrator.t, integrator)

#= After a step: whether a relation disagrees with the state or a discrete
   when's condition holds. =#
function _needsIteration(e::EventIteration, integrator)
  e.ifRelations !== nothing && _inconsistent(e.ifRelations, integrator) && return true
  any(a -> _whenInconsistent(a, integrator), e.relationWhens) && return true
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
  #= 2. The bodies, with pre() from the state before them. =#
  local pre = copy(integrator.u)
  for a in e.relationWhens
    _whenFire!(a, integrator, pre) && (changed = true)
  end
  for d in e.discreteWhens
    _holds(d, integrator) || continue
    d.affect!(integrator)
    changed = true
  end
  return changed
end

function _iterate!(e::EventIteration, integrator)
  for _ in 1:e.limit
    _sweep!(e, integrator) || return nothing
    #= Phase 3: the branches or the state changed; the algebraic unknowns follow. =#
    if !_resolveAlgebraics!(integrator)
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
   relation to its literal value at the solved initial state: if an
   if-equation relation differs from the value the initialization used, the
   algebraic unknowns are solved again with it (the states kept). =#
function _initialize!(e::EventIteration, integrator)
  local r = e.ifRelations
  r === nothing && return nothing
  _bufferValues(r.buffers, integrator) != r.compiled && _resolveAlgebraics!(integrator)
  return nothing
end

#= The callbacks that stay, and the whens the iteration takes over: those on
   a relation (their continuous callbacks stay, to locate the events) and
   those on a discrete condition (their callbacks go). =#
function _splitCallbacks(callbacks)
  local kept = Any[]
  local relationWhens = RelationWhenAffect[]
  local discreteWhens = DiscreteWhenAffect[]
  if callbacks isa DiffEqBase.CallbackSet
    for cb in callbacks.continuous_callbacks
      cb.affect! isa RelationWhenAffect && push!(relationWhens, cb.affect!)
      push!(kept, cb)
    end
    for cb in callbacks.discrete_callbacks
      cb.affect! isa DiscreteWhenAffect ? push!(discreteWhens, cb.affect!) : push!(kept, cb)
    end
  elseif callbacks !== nothing
    push!(kept, callbacks)
  end
  return (kept, relationWhens, discreteWhens)
end

"""
    withRelationRefresh(callbacks, problem, hSym, entries) -> callbacks

Add the event iteration over the buffered relations: the if-equation
relations `entries` = `(ifCond, crossing function, scale)` and the whens on
a relation among `callbacks` (their affects are `RelationWhenAffect`s). A
DiscreteCallback checks them after every step and, where one disagrees with
the state, iterates as described above. `hSym` is the hysteresis parameter H.

The whens on discrete conditions among `callbacks` (`DiscreteWhenAffect`s)
move into the iteration: a relation's when body can change the discrete
they watch, and their bodies can move a relation, within one event.
"""
function withRelationRefresh(callbacks, problem, hSym::Symbol, entries::Vector)
  local (kept, relationWhens, discreteWhens) = _splitCallbacks(callbacks)
  local ifRelations = _ifRelations(problem, hSym, entries)
  ifRelations === nothing && isempty(relationWhens) && return callbacks
  local n = (ifRelations === nothing ? 0 : length(ifRelations)) + length(relationWhens) + length(discreteWhens)
  local e = EventIteration(ifRelations, relationWhens, discreteWhens, _eventIterationLimit(n))
  local cb = DiffEqBase.DiscreteCallback((u, t, integrator) -> _needsIteration(e, integrator),
                                         integrator -> _iterate!(e, integrator);
                                         initialize = (c, u, t, integrator) -> _initialize!(e, integrator),
                                         #= The state after the iteration, at the event's time
                                            (the when callbacks saved the left limit). The
                                            iteration solves the algebraic variables itself. =#
                                         save_positions = (false, true),
                                         initializealg = ModelingToolkit.SciMLBase.NoInit())
  return DiffEqBase.CallbackSet(kept..., cb)
end

#= Solve the algebraic unknowns again, the differential states kept (a
   pure ODE has none). A DAEFunction (DFBDF, IDA) has no mass matrix; its
   algebraic variables are the ones `differential_vars` excludes. Whether it
   succeeded. =#
function _resolveAlgebraics!(integrator)
  local f = integrator.f
  if hasproperty(f, :mass_matrix)
    local mm = f.mass_matrix
    (mm isa LinearAlgebra.UniformScaling || all(!iszero, LinearAlgebra.diag(mm))) && return true
  end
  DiffEqBase.initialize_dae!(integrator, OMBackend.OrdinaryDiffEq.BrownFullBasicInit())
  DiffEqBase.u_modified!(integrator, true)
  return integrator.sol.retcode != ModelingToolkit.SciMLBase.ReturnCode.InitialFailure
end

"""
    setZCHysteresis!(problem, hSym, reltol)

Set the hysteresis parameter H of the event crossing functions for a solve
with relative tolerance `reltol`: H = 1e-4 * reltol (OpenModelica's tolZC).
Nothing for a problem without it.
"""
function setZCHysteresis!(problem, hSym::Symbol, reltol)
  local SII = ModelingToolkit.SymbolicIndexingInterface
  local present = try
    SII.parameter_index(problem, hSym) !== nothing
  catch
    false
  end
  present || return problem
  SII.setp(problem, hSym)(problem, _hysteresis(reltol))
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
