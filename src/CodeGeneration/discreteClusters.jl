#= Discrete clusters: a when-equation over the relations of coupled
   discretes, lifted from their discrete equations (PartialFriction's free,
   startForward, startBackward, locked and mode; a digital gate's logic
   values), or a user when on `change`/`edge` of relations in such a model.

   Its relations are buffered with a hysteresis, like the if-equation
   relations and the whens on a relation (relationRefresh.jl). Its
   continuous callback only locates their crossings; the event iteration
   evaluates the body. There each pass solves the cluster as a mixed system,
   as OpenModelica solves discrete equations together with the algebraic
   loop they are in (PartialFriction's startForward and locked together with
   `sa`): with pre() fixed, the body gives the discretes, the algebraic
   variables are solved again for them, the relations are evaluated again,
   and so on until the discretes stay. A friction element that locks from
   sliding therefore decides a breakaway on its stuck torque, not on the
   sliding one.

   A cluster reads the problem's system once (`_bind!`); a variable-structure
   model, whose system changes during the solve, is not supported here. =#

"""
    DiscreteCluster(names, members, reads, nOperands, nPre, strict, exact, body, atStart, table)

A when-equation evaluated by the event iteration (see the top of the file).

- `members`: the discretes the body assigns, in order (symbolic), named `names`.
- `reads`: symbolic expressions the cluster reads: the body's operands
  (`nOperands`), the discretes it reads through pre() (`nPre`), then for each
  relation its crossing function and its scale. A relation is true when its
  crossing function is negative (`strict`: `<`, `>`) or not positive.
- `exact`: per relation, whether it has no hysteresis: no continuous-time
  operand (time, parameters, discretes only), or `==`/`<>`.
- `body(integrator, operands, pre, rel, relPre, initial)`: the new values of
  the members, or `nothing` when the when does not fire. `rel` are the
  relation values, `relPre` their values before the pass.
- `atStart`: the body at the start: 0 not run, 1 with initial() false, 2 with
  initial() true.
- `table`: the model's algebraic rows read constant tables. A flag of the
  model, the same for all its clusters: the start body runs with initial()
  true, and the algebraic variables are solved with Newton on finite
  differences (`tableClusterInitAlg`).
"""
mutable struct DiscreteCluster
  names::Vector{String}
  members::Vector{Any}
  reads::Vector{Any}
  nOperands::Int
  nPre::Int
  strict::Vector{Bool}
  exact::Vector{Bool}
  body::Any
  atStart::Int
  table::Bool
  values::Any                # (u, p, t) -> all reads, for the problem's system
  crossings!::Any            # (out, u, p, t): per relation its crossing function and scale
  zs::Vector{Float64}        # the output of crossings!
  memberIndex::Vector{Int}   # each member's position in u; 0 when it is not an unknown
  rel::Vector{Bool}          # the relation buffers
  crossed::Vector{Bool}      # the relations whose crossing the callback located, until the next update
end

DiscreteCluster(names, members, reads, nOperands, nPre, strict, exact, body, atStart, table) =
  DiscreteCluster(names, members, reads, nOperands, nPre, strict, exact, body, atStart, table,
                  nothing, nothing, zeros(2 * length(strict)), Int[], fill(false, length(strict)),
                  fill(false, length(strict)))

#= SciMLBase 3 renamed u_modified! (now deprecated) to derivative_discontinuity!. =#
const _derivativeDiscontinuity! = isdefined(ModelingToolkit.SciMLBase, :derivative_discontinuity!) ?
  ModelingToolkit.SciMLBase.derivative_discontinuity! : ModelingToolkit.SciMLBase.u_modified!

#= Every evaluation of a cluster's body is printed with
   OMBACKEND_EVENT_TRACE=true (read when an event is handled), or when this
   is set to true. =#
const _EVENT_TRACE = Ref{Union{Nothing, Bool}}(nothing)
_eventTrace() = something(_EVENT_TRACE[], get(ENV, "OMBACKEND_EVENT_TRACE", "false") == "true")

_readValues(c::DiscreteCluster, integrator) = c.values(integrator.u, integrator.p, integrator.t)
_operands(c::DiscreteCluster, v) = view(v, 1:c.nOperands)
_preValues(c::DiscreteCluster, v) = v[(c.nOperands + 1):(c.nOperands + c.nPre)]

#= The crossing functions and scales at (u, t): relation k's are zs[2k - 1] and zs[2k]. =#
function _crossingValues!(c::DiscreteCluster, u, p, t)
  c.crossings!(c.zs, u, p, t)
  return c.zs
end
_crossingValues!(c::DiscreteCluster, integrator) = _crossingValues!(c, integrator.u, integrator.p, integrator.t)

_literal(zs, k::Int, strict::Bool) = strict ? zs[2k - 1] < 0 : zs[2k - 1] <= 0

#= Relation k by the hysteresis rule from `old`; an exact one literal. =#
function _ruled(c::DiscreteCluster, zs, k::Int, H, old::Bool)
  c.exact[k] && return _literal(zs, k, c.strict[k])
  return _hysteresisRule(zs[2k - 1], H * zs[2k], old)
end

#= The functions for the problem's system. =#
function _bind!(c::DiscreteCluster, problem)
  local SII = ModelingToolkit.SymbolicIndexingInterface
  local sys = problem.f.sys
  c.values = ModelingToolkit.build_explicit_observed_function(sys, c.reads)
  c.crossings! = first(ModelingToolkit.build_explicit_observed_function(sys, c.reads[(c.nOperands + c.nPre + 1):end];
                                                                        return_inplace = Val(true)))
  c.memberIndex = Int[something(SII.variable_index(sys, m), 0) for m in c.members]
  for (name, k) in zip(c.names, c.memberIndex)
    k == 0 && @warn "[events] the discrete $name is not an unknown of the system; its when cannot change it"
  end
  return c
end

#= The crossing functions, shifted by the hysteresis relative to the buffers.
   Called at every condition evaluation: the typed loop is behind a barrier. =#
function _crossings!(out, c::DiscreteCluster, u, t, integrator)
  _shiftCrossings!(out, _crossingValues!(c, u, integrator.p, t), c.rel, c.exact, _hysteresisFromTolerance(integrator))
  return nothing
end

function _shiftCrossings!(out, zs::Vector{Float64}, rel::Vector{Bool}, exact::Vector{Bool}, H::Float64)
  for k in eachindex(rel)
    out[k] = exact[k] ? zs[2k - 1] : zs[2k - 1] + H * zs[2k] * (1 - 2 * rel[k])
  end
  return nothing
end

#= Whether a relation disagrees with the state by the rule from `reference`
   (the buffers, or the values at the start of a pass). =#
function _inconsistent(c::DiscreteCluster, integrator, reference::Vector{Bool} = c.rel)
  local zs = _crossingValues!(c, integrator)
  local H = _hysteresisFromTolerance(integrator)
  return any(k -> _ruled(c, zs, k, H, reference[k]) != c.rel[k], eachindex(c.rel))
end

#= Phase 1 of a sweep: the buffers from the state. Whether one changed. A
   relation whose crossing the callback located changes, as in OpenModelica:
   its shifted crossing function is zero where the rule flips it, and the
   algebraic solve at the event can leave it just inside the hysteresis band
   (an ideal thyristor's firing, found only at the end of the next step). A
   state clearly on the old side flips it back in the next sweep. =#
function _update!(c::DiscreteCluster, integrator)
  local zs = _crossingValues!(c, integrator)
  local H = _hysteresisFromTolerance(integrator)
  local changed = false
  for k in eachindex(c.rel)
    local new = c.crossed[k] ? !c.rel[k] : _ruled(c, zs, k, H, c.rel[k])
    c.crossed[k] = false
    new == c.rel[k] && continue
    c.rel[k] = new
    changed = true
  end
  return changed
end

#= Every relation literal on the state (MLS 8.5, as at initialization). =#
function _literalBuffers!(c::DiscreteCluster, integrator)
  local zs = _crossingValues!(c, integrator)
  for k in eachindex(c.rel)
    c.rel[k] = _literal(zs, k, c.strict[k])
  end
  fill!(c.crossed, false)
  return nothing
end

#= Write the body's values. Whether a member changed. =#
function _write!(c::DiscreteCluster, integrator, values)
  values === nothing && return false
  local changed = false
  for (i, k) in enumerate(c.memberIndex)
    (k == 0 || integrator.u[k] == values[i]) && continue
    integrator.u[k] = values[i]
    changed = true
  end
  changed && _derivativeDiscontinuity!(integrator, true)
  return changed
end

#= `body` is generated code, compiled after OMBackend: call it in the latest world. =#
_runBody(c::DiscreteCluster, integrator, v, pre, relPre, initial::Bool) =
  Base.invokelatest(c.body, integrator, _operands(c, v), pre, c.rel, relPre, initial)

#= Passes of the mixed system before a cluster gives up (the event iteration
   goes on with the last values), and the most relation groups a search may
   vary. =#
const _MIXED_SYSTEM_LIMIT = 10
const _SEARCH_LIMIT = 8

#= The values of the members that are unknowns: in the state, or of the body. =#
_memberValues(c::DiscreteCluster, integrator) = Float64[integrator.u[k] for k in c.memberIndex if k != 0]
_memberValues(c::DiscreteCluster, values::Tuple) =
  Float64[values[i] for i in eachindex(c.memberIndex) if c.memberIndex[i] != 0]

#= Whether the body's values equal the state's (members that are unknowns). =#
_settled(c::DiscreteCluster, integrator, values) =
  values === nothing || all(i -> c.memberIndex[i] == 0 || integrator.u[c.memberIndex[i]] == values[i],
                            eachindex(c.memberIndex))

function _traceBody(c::DiscreteCluster, integrator, pre, relPre, values, v)
  println("[events] t=", integrator.t, " ", first(c.names), " pre=", pre, " rel=", Int.(c.rel),
          " relPre=", Int.(relPre), " -> ", values, " reads=", round.(v; sigdigits = 6))
end

#= A failed algebraic solve leaves the solution's return code at
   InitialFailure, and later solves report the same even when they succeed:
   put back the code from before a trial that failed. =#
_restoreRetcode!(integrator, retcode) =
  (integrator.sol = ModelingToolkit.SciMLBase.solution_new_retcode(integrator.sol, retcode); nothing)

"""
    _solveMixedSystem!(c, integrator, pre, relPre, reinit) -> Bool

Phase 2 of a sweep for a cluster: its mixed system, with pre() fixed at
`pre` (the values at the start of the phase). The body gives the discretes;
the algebraic variables are solved again for them and the relations
evaluated again, until the discretes stay. Whether a discrete changed.

Where this cycles (PartialFriction locking from a slide: the forward slide
gives the torque of a backward breakaway and the other way round, and the
stuck state is never tried), the relations that flipped are searched
(`_search!`), as OpenModelica's nonlinear solver finds the consistent branch
of the residual that contains the discrete equations.
"""
function _solveMixedSystem!(c::DiscreteCluster, integrator, pre, relPre, reinit)
  local changed = false
  local relStart = copy(c.rel)
  local flipped = fill(false, length(c.rel))
  local tried = Vector{Vector{Float64}}()
  for _ in 1:_MIXED_SYSTEM_LIMIT
    local v = _readValues(c, integrator)
    local values = _runBody(c, integrator, v, pre, relPre, false)
    _eventTrace() && _traceBody(c, integrator, pre, relPre, values, v)
    _settled(c, integrator, values) && return changed
    if _memberValues(c, values) in tried
      return _search!(c, integrator, pre, relPre, reinit, relStart, flipped) || changed
    end
    push!(tried, _memberValues(c, integrator))
    _write!(c, integrator, values)
    changed = true
    _resolveAlgebraics!(integrator, reinit) || return changed
    local before = copy(c.rel)
    _update!(c, integrator)
    flipped .|= before .!= c.rel
  end
  @warn "[events] the discretes $(join(c.names, ", ")) did not settle at t = $(integrator.t); " *
        "the event iteration goes on with their last values" _id = Symbol(:mixed_, first(c.names)) maxlog = 1
  return changed
end

"""
    _search!(c, integrator, pre, relPre, reinit, relStart, flipped) -> Bool

The values of the relations `flipped` for which the mixed system is
consistent: with them the body gives discretes whose solution gives the same
relation values back, by the hysteresis rule from `relStart` (the values at
the start of the pass, fixed for the search as OpenModelica keeps its stored
relations). Relations whose crossing function and scale are equal (twins,
such as `sa > tau0_max` and `sa > tau0` with peak = 1) vary together.
Candidates are tried from the same state, fewest changes from `relStart`
first. Whether a discrete changed; when none is consistent the state is put
back, and a model that keeps cycling stops at the event iteration's limit.
"""
function _search!(c::DiscreteCluster, integrator, pre, relPre, reinit, relStart, flipped)
  local zs = copy(_crossingValues!(c, integrator))
  local groups = Vector{Vector{Int}}()
  for k in findall(flipped)
    local j = findfirst(g -> zs[2first(g) - 1] == zs[2k - 1] && zs[2first(g)] == zs[2k], groups)
    j === nothing ? push!(groups, [k]) : push!(groups[j], k)
  end
  if length(groups) > _SEARCH_LIMIT
    @warn "[events] the discretes $(join(c.names, ", ")) cycle at t = $(integrator.t) over " *
          "$(length(groups)) relations, too many to search" _id = Symbol(:searchsize_, first(c.names)) maxlog = 1
    return false
  end
  local u0 = copy(integrator.u)
  local rel0 = copy(c.rel)
  local retcode = integrator.sol.retcode
  local bit(m, j) = (m >> (j - 1)) & 1 == 1
  local distance(m) = count(j -> bit(m, j) != relStart[first(groups[j])], eachindex(groups))
  for m in sort(0:(2^length(groups) - 1); by = m -> (distance(m), m))
    copyto!(integrator.u, u0)
    c.rel .= rel0
    for (j, g) in enumerate(groups), k in g
      c.rel[k] = bit(m, j)
    end
    local v = _readValues(c, integrator)
    local values = _runBody(c, integrator, v, pre, relPre, false)
    _eventTrace() && _traceBody(c, integrator, pre, relPre, values, v)
    _write!(c, integrator, values)
    if !_resolveAlgebraics!(integrator, reinit)
      _restoreRetcode!(integrator, retcode)
      continue
    end
    _inconsistent(c, integrator, relStart) && continue
    _settled(c, integrator, _runBody(c, integrator, _readValues(c, integrator), pre, relPre, false)) || continue
    return true
  end
  copyto!(integrator.u, u0)
  c.rel .= rel0
  _derivativeDiscontinuity!(integrator, true)
  @warn "[events] the discretes $(join(c.names, ", ")) have no consistent values at t = $(integrator.t)" _id = Symbol(:search_, first(c.names)) maxlog = 1
  return false
end

#= At the start, on the solved initial state: every relation literal, then
   the body where the when has an initial() term. Whether a member changed. =#
function _initialize!(c::DiscreteCluster, integrator)
  _literalBuffers!(c, integrator)
  c.atStart == 0 && return false
  local v = _readValues(c, integrator)
  return _write!(c, integrator, _runBody(c, integrator, v, _preValues(c, v), copy(c.rel), c.atStart == 2))
end

"""
    ClusterCrossings

The affect of a cluster's continuous callback: it only locates the event;
the event iteration handles it (and solves the algebraic variables there).
"""
struct ClusterCrossings
  cluster::DiscreteCluster
end

#= `events`: per relation 0 (no crossing) or the crossing's direction (±1). =#
function (a::ClusterCrossings)(integrator, events)
  for (k, d) in pairs(events)
    d == 0 || (a.cluster.crossed[k] = true)
  end
  _derivativeDiscontinuity!(integrator, false)
  return nothing
end

function _clusterCallback(c::DiscreteCluster)
  local condition = (out, u, t, integrator) -> _crossings!(out, c, u, t, integrator)
  #= As relationWhenCallback: RightRootFind, the left limit saved here and
     the right one by the event iteration, no algebraic solve here (the event
     iteration does it), and an initialize function that leaves the modified
     flag alone. The event iteration sets the buffers. =#
  return DiffEqBase.VectorContinuousCallback(condition, ClusterCrossings(c), length(c.rel);
                                             initialize = (cb, u, t, integrator) -> nothing,
                                             initializealg = ModelingToolkit.SciMLBase.NoInit(),
                                             rootfind = ModelingToolkit.SciMLBase.RightRootFind,
                                             save_positions = (true, false))
end

"""
    withDiscreteClusters(callbacks, problem, clusters) -> callbacks

Add a continuous callback per discrete cluster that locates the crossings of
its relations; the event iteration (`withRelationRefresh`, emitted after
this) evaluates them. A cluster whose values cannot be read from the
problem is an error: its discretes would never change.
"""
function withDiscreteClusters(callbacks, problem, clusters::Vector)
  isempty(clusters) && return callbacks
  local added = Any[]
  for c in clusters
    try
      _bind!(c, problem)
    catch err
      error("[events] the discretes $(join(c.names, ", ")) cannot be read from the problem: " *
            sprint(showerror, err))
    end
    push!(added, _clusterCallback(c))
  end
  return callbacks === nothing ? DiffEqBase.CallbackSet(added...) : DiffEqBase.CallbackSet(callbacks, added...)
end
