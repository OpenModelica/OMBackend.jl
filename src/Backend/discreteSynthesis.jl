#= BDAECreate: discrete equations lifted into when-clusters (MLS 8.6: a discrete
   variable changes only at events), with alias read-through and initial pre() values. =#

Base.@nospecializeinfer function _pushDiscreteCref!(out::Vector{DAE.ComponentRef},
                                                    seen::OrderedSet{String},
                                                    blocked::OrderedSet{String},
                                                    @nospecialize(cref))
  cref isa DAE.ComponentRef || return nothing
  _isTimeCref(cref) && return nothing
  local key = string(cref)
  key in blocked && return nothing
  local ty = _crefType(cref)
  ty === nothing && return nothing
  _isDiscreteDAEType(ty) || return nothing
  key in seen && return nothing
  push!(seen, key)
  push!(out, cref)
  return nothing
end

# Collect discrete RHS crefs into `out` (deduped via `seen`, skipping `blocked`
# reduction/for iterators). Typed functor replacing the threaded ctx tuple.
struct DiscreteRhsCrefVisitor
  out::Vector{DAE.ComponentRef}
  seen::OrderedSet{String}
  blocked::OrderedSet{String}
end
Base.@nospecializeinfer function (v::DiscreteRhsCrefVisitor)(@nospecialize(exp), arg::Nothing)
  @match exp begin
    DAE.CREF(cr, _) => _pushDiscreteCref!(v.out, v.seen, v.blocked, cr)
    DAE.REDUCTION(_, _, iters) => _collectReductionIterNames!(v.blocked, iters)
    _ => nothing
  end
  return (exp, true, arg)
end

Base.@nospecializeinfer function _collectReductionIterNames!(blocked::OrderedSet{String}, @nospecialize(iters))
  for it in iters
    @match it begin
      DAE.REDUCTIONITER(id, _, _, _) => push!(blocked, id)
      _ => nothing
    end
  end
  return nothing
end

Base.@nospecializeinfer function _makeChangeCall(@nospecialize(cref))
  local ty = _crefType(cref)
  local callArg = DAE.CREF(cref, ty === nothing ? DAE.T_REAL_DEFAULT : ty)
  return DAE.CALL(Absyn.IDENT("change"),
                  MetaModelica.list(callArg),
                  DAE.callAttrBuiltinBool)
end

#= §17.4.4 equation-section lift. A discrete (Bool/Int/enum) variable defined by
   `lhs = relexpr`, where `relexpr` is a discrete-time expression (relations,
   pre/initial/change, logical ops over discrete/param/const operands), is held
   constant between events and recomputed only at zero-crossings of the relations
   in its RHS. Such an equation is replaced by a paired INITIAL_WHEN_EQUATION
   (t=0 value) + runtime WHEN_EQUATION triggered by `change(rel1) or … or change(relK)`.
   This is the equation-section analogue of `synthesizeWhenEquationsFromRegularAlgorithms`,
   except the trigger is built over the RELATIONS (the event sources) rather than over
   discrete RHS crefs — `change(w_rel <= 0)` fires only at the sign flip, whereas
   `change(w_rel)` would over-trigger every step. =#

#= Discrete-time predicate: an expression with no continuous-Real dependence
   OUTSIDE a relation. Relations are event sources, so their (possibly continuous)
   operands are allowed. =#
Base.@nospecializeinfer function _isDiscreteTimeExp(@nospecialize(exp), paramOrConstNames::OrderedSet{String})::Bool
  @match exp begin
    DAE.RELATION(__) => true
    DAE.LBINARY(e1, _, e2) => _isDiscreteTimeExp(e1, paramOrConstNames) && _isDiscreteTimeExp(e2, paramOrConstNames)
    DAE.LUNARY(_, e1) => _isDiscreteTimeExp(e1, paramOrConstNames)
    DAE.BCONST(__) => true
    DAE.ICONST(__) => true
    DAE.RCONST(__) => true
    DAE.SCONST(__) => true
    DAE.ENUM_LITERAL(__) => true
    DAE.CALL(Absyn.IDENT(n), _, _) => (n in ("pre", "initial", "change", "edge", "sample", "noEvent"))
    DAE.IFEXP(c, t, f) => _isDiscreteTimeExp(c, paramOrConstNames) &&
                          _isDiscreteTimeExp(t, paramOrConstNames) &&
                          _isDiscreteTimeExp(f, paramOrConstNames)
    DAE.CREF(cr, ty) => ((string(cr) in paramOrConstNames) ? true : !_isContinuousRealType(ty))
    _ => false
  end
end

#= Collect the (structurally deduplicated) RELATION subtrees of `exp`. =#
Base.@nospecializeinfer function _collectRelationsInExp(@nospecialize(exp))::Vector{DAE.Exp}
  local rels = DAE.Exp[]
  local seen = OrderedSet{String}()
  function visit(@nospecialize(e), arg)
    if e isa DAE.RELATION
      local key = string(e)
      if !(key in seen)
        push!(seen, key)
        push!(rels, e)
      end
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(exp, visit, nothing)
  return rels
end

Base.@nospecializeinfer function _makeChangeCallExp(@nospecialize(relExp))
  return DAE.CALL(Absyn.IDENT("change"), MetaModelica.list(relExp), DAE.callAttrBuiltinBool)
end

#= True if a relation has at least one continuous-Real operand, i.e. a genuine
   zero-crossing event source. A relation over only discrete/param/const operands
   (e.g. `pre(mode) == Stuck`) has no smooth crossing and must NOT become an
   event — it is re-evaluated inside the affect from pre-event state instead. =#
Base.@nospecializeinfer function _relationHasContinuousOperand(@nospecialize(rel::DAE.Exp), paramOrConstNames::OrderedSet{String})::Bool
  local found = false
  function visit(@nospecialize(e), arg)
    if e isa DAE.CREF && !found
      if !(string(e.componentRef) in paramOrConstNames) && _isContinuousRealType(e.ty)
        found = true
      end
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(rel, visit, nothing)
  return found
end

Base.@nospecializeinfer function _buildChangeOrConditionFromExps(rels::Vector{DAE.Exp})
  isempty(rels) && return DAE.BCONST(false)
  local acc = _makeChangeCallExp(rels[1])
  for i in 2:length(rels)
    acc = DAE.LBINARY(acc, DAE.OR(DAE.T_BOOL_DEFAULT), _makeChangeCallExp(rels[i]))
  end
  return acc
end

#= The variables a lifted body reads through pre(), edge() or change(), once each. =#
function _preReadCrefs(body::Vector{Tuple{DAE.Exp, DAE.Exp, Any}})::Vector{DAE.Exp}
  local found = DAE.Exp[]
  local seen = OrderedSet{String}()
  function visit(@nospecialize(e), arg)
    if e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name in ("pre", "edge", "change")
      local a = listHead(e.expLst)
      if a isa DAE.CREF && !(string(a.componentRef) in seen)
        push!(seen, string(a.componentRef))
        push!(found, a)
      end
    end
    return (e, arg)
  end
  for (_, r, _) in body
    Util.traverseExpBottomUp(r, visit, nothing)
  end
  return found
end

#= Whether a variable the group reads through pre() reads a member outside
   pre(): its own definition (a member), or another equation it is in
   (`otherRefs()`: per name, the names each other equation reads outside
   pre(); names under `canon`, their alias set's). A member that reads only
   its own pre() (y = (s or pre(y)) and not r, s and r from whens) does not
   count: the lifted when fires on the change of pre(y) alone and would miss
   s; unlifted, the equation is solved at every event. =#
function _preReadsCloseLoop(held::Vector{DAE.Exp}, members::OrderedSet{String},
                            candByName::Dict{String, Any}, otherRefs::Function, canon::Function)::Bool
  for h in held
    local n = canon(string(h.componentRef))
    if n in members
      isempty(_candRefsOutsidePre(candByName[n].rhs, members)) || return true
    else
      any(refs -> any(m -> m in refs, members), get(otherRefs(), n, OrderedSet{String}[])) && return true
    end
  end
  return false
end

#= Per name, the sets of names the equations that read it outside pre()
   read outside pre(), all under `canon`. =#
function _refsOutsidePreIndex(eqs::AbstractVector{BDAE.Equation}, canon::Function)::Dict{String, Vector{OrderedSet{String}}}
  local index = Dict{String, Vector{OrderedSet{String}}}()
  for eq in eqs
    local refs = OrderedSet{String}()
    local visit = function (@nospecialize(e), arg)
      @match e begin
        DAE.CALL(Absyn.IDENT("pre"), _, _) => (e, false, arg)
        DAE.CREF(cr, _) => (push!(refs, canon(string(cr))); (e, true, arg))
        _ => (e, true, arg)
      end
    end
    BDAEUtil.traverseEquationExpressions(eq, visit, nothing)
    for r in refs
      push!(get!(index, r, OrderedSet{String}[]), refs)
    end
  end
  return index
end

#= The condition of a lifted cluster without relations: `change(pre(n1)) or
   change(pre(n2)) ...` over the values it follows. Not valid Modelica (the
   argument of change() is a variable), so no user when has it; the code
   generation takes it for a cluster with no relation (_collectChangeRelations!)
   and never evaluates it (the event iteration runs the body at every pass). =#
function _buildChangeOfPreCondition(held::Vector{DAE.Exp})
  local preOf(c) = DAE.CALL(Absyn.IDENT("pre"), MetaModelica.list(c), DAE.callAttrBuiltinBool)
  local acc = _makeChangeCallExp(preOf(held[1]))
  for i in 2:length(held)
    acc = DAE.LBINARY(acc, DAE.OR(DAE.T_BOOL_DEFAULT), _makeChangeCallExp(preOf(held[i])))
  end
  return acc
end

#= A discrete (Bool/Int/enum) equation `lhs = rhs` whose RHS is a discrete-time
   expression. Returns `(name, lhs, rhs, src, eq)` or `nothing`. Unlike the
   when-emit step this does NOT require the RHS to contain a relation: a
   no-relation member like `locked = pre(stuck) and not startForward` qualifies
   as a candidate so it can join a coupled cluster (it is event-driven through
   its siblings' relations). =#
Base.@nospecializeinfer function _discreteBoolCandidate(@nospecialize(eq::BDAE.Equation),
                                                        paramOrConstNames::OrderedSet{String})
  eq isa BDAE.EQUATION || return nothing
  local lhs = eq.lhs
  lhs isa DAE.CREF || return nothing
  local lhsDiscrete = _isDiscreteDAEType(lhs.ty)
  if !lhsDiscrete
    local ct = _crefType(lhs.componentRef)
    lhsDiscrete = ct !== nothing && _isDiscreteDAEType(ct)
  end
  lhsDiscrete || return nothing
  local lhsName = string(lhs.componentRef)
  (lhsName == "time" || lhsName in paramOrConstNames) && return nothing
  _isDiscreteTimeExp(eq.rhs, paramOrConstNames) || return nothing
  return (name = lhsName, lhs = lhs, rhs = eq.rhs, src = eq.source, eq = eq)
end

#= The rhs of an alias equation: a discrete variable, not a parameter. =#
_isDiscreteAlias(@nospecialize(rhs), paramOrConstNames::OrderedSet{String})::Bool =
  rhs isa DAE.CREF && !(string(rhs.componentRef) in paramOrConstNames)

#= The alias sets of the discrete aliases `a = b`, in the aliases' order. =#
function _aliasSets(aliases::Vector{Tuple{String, String}})::Vector{Vector{String}}
  local names = OrderedSet{String}()
  local adj = Dict{String, OrderedSet{String}}()
  for (a, b) in aliases
    push!(names, a); push!(names, b)
    push!(get!(adj, a, OrderedSet{String}()), b)
    push!(get!(adj, b, OrderedSet{String}()), a)
  end
  return _connectedComponents(collect(names), adj)
end

#= The initial equations that fix a variable or a pre() value to a literal or
   a parameter (`pre(y) = pre_y_start`, the MSL Hysteresis and Pre blocks):
   (values = name => value, pre = name => value for `pre(name) = value`). =#
function _initialConstants(initialEquations::Vector{BDAE.Equation}, paramOrConstNames::OrderedSet{String})
  local values = OrderedDict{String, DAE.Exp}()
  local pre = OrderedDict{String, DAE.Exp}()
  isValue(e) = e isa DAE.BCONST || e isa DAE.ICONST || e isa DAE.RCONST || e isa DAE.ENUM_LITERAL ||
               (e isa DAE.CREF && string(e.componentRef) in paramOrConstNames)
  for eq in initialEquations
    eq isa BDAE.EQUATION || continue
    for (a, b) in ((eq.lhs, eq.rhs), (eq.rhs, eq.lhs))
      isValue(b) || continue
      if a isa DAE.CREF
        values[string(a.componentRef)] = b
      else
        local x = _preArgument(a)
        x === nothing || (pre[x] = b)
      end
    end
  end
  return (values = values, pre = pre)
end

#= The name `n` of `pre(n)`, else nothing. =#
Base.@nospecializeinfer function _preArgument(@nospecialize(e))::Union{Nothing, String}
  @match e begin
    DAE.CALL(Absyn.IDENT("pre"), args, _) => (local a = listHead(args); a isa DAE.CREF ? string(a.componentRef) : nothing)
    _ => nothing
  end
end

#= name => the name of its alias set's definition, for the sets with exactly
   one definition among the candidates (the definition maps to itself). =#
function _aliasDefinitions(candByName::Dict{String, Any}, aliasSets::Vector{Vector{String}})::Dict{String, String}
  local definitionOf = Dict{String, String}()
  for set in aliasSets
    local defs = filter(n -> haskey(candByName, n), set)
    length(defs) == 1 || continue
    foreach(n -> definitionOf[n] = only(defs), set)
  end
  return definitionOf
end

#= Rewrite each candidate's rhs to read the definition of an alias set in
   place of its aliases (pre() included: an alias holds at every instant), so
   the clusters and their bodies do not depend on how the connects are
   oriented. A set without exactly one definition among the candidates is
   left as it is. =#
function _readThroughAliases!(candByName::Dict{String, Any}, definitionOf::Dict{String, String})
  isempty(definitionOf) && return candByName
  function subst(@nospecialize(e), arg)
    e isa DAE.CREF || return (e, arg)
    local n = string(e.componentRef)
    local d = get(definitionOf, n, n)
    return (d == n ? e : DAE.CREF(candByName[d].lhs.componentRef, e.ty), arg)
  end
  for n in collect(keys(candByName))
    local c = candByName[n]
    candByName[n] = merge(c, (rhs = first(Util.traverseExpBottomUp(c.rhs, subst, nothing)),))
  end
  return candByName
end

#= The start lookup that folds pre() at initialization, and the names whose
   pre() the initial equations fix (folded even outside the cluster). On the
   rhs read through the aliases (_readThroughAliases!), so on the
   definitions' names: a start only an alias carries is the definition's;
   `pre(n) = v`; and pre(n) of a candidate `m = pre(n)` whose value an
   initial equation fixes, directly or through an alias (the StateGraph
   InitialStep: `active = true` with `active = localActive` and `localActive =
   pre(newActive)`, newActive not lifted, starts the step active; newActive's
   start alone left every step inactive). =#
function _initialPreValues(startLookup::Dict{String, DAE.Exp}, candNames::Vector{String},
                           candByName::Dict{String, Any}, aliasSets::Vector{Vector{String}},
                           definitionOf::Dict{String, String}, initialConstants)
  local fixed = OrderedSet{String}()
  (isempty(aliasSets) && isempty(initialConstants.values) && isempty(initialConstants.pre)) && return (startLookup, fixed)
  local lookup = copy(startLookup)
  local valueOf = Dict{String, DAE.Exp}()
  for set in aliasSets
    local d = get(definitionOf, first(set), nothing)
    if d !== nothing && !haskey(startLookup, d)
      local i = findfirst(n -> haskey(startLookup, n), set)
      i === nothing || (lookup[d] = startLookup[set[i]])
    end
    local j = findfirst(n -> haskey(initialConstants.values, n), set)
    j === nothing || foreach(n -> valueOf[n] = initialConstants.values[set[j]], set)
  end
  for (x, v) in initialConstants.pre
    local d = get(definitionOf, x, x)
    lookup[d] = v
    push!(fixed, d)
  end
  for n in candNames
    local x = _preArgument(candByName[n].rhs)
    x === nothing && continue
    local v = get(initialConstants.values, n, get(valueOf, n, nothing))
    v === nothing && continue
    lookup[x] = v
    push!(fixed, x)
  end
  return (lookup, fixed)
end

#= Candidate cref names referenced anywhere in `exp` (including inside pre()).
   Used to connect a cluster: two candidates are coupled if either references
   the other. =#
Base.@nospecializeinfer function _candRefsAnywhere(@nospecialize(exp::DAE.Exp), restrict::OrderedSet{String})::OrderedSet{String}
  local found = OrderedSet{String}()
  function visit(@nospecialize(e), arg)
    if e isa DAE.CREF
      local nm = string(e.componentRef)
      (nm in restrict) && push!(found, nm)
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(exp, visit, nothing)
  return found
end

#= Candidate cref names referenced OUTSIDE any pre(): the topological-order
   edges. References inside pre() are loop-breakers (read the pre-event value)
   and impose no order. =#
Base.@nospecializeinfer function _candRefsOutsidePre(@nospecialize(exp::DAE.Exp), restrict::OrderedSet{String})::OrderedSet{String}
  local found = OrderedSet{String}()
  function f(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("pre"), _, _) => (e, false, arg)
      DAE.CREF(cr, _) => begin
        (string(cr) in restrict) && push!(found, string(cr))
        (e, true, arg)
      end
      _ => (e, true, arg)
    end
  end
  Util.traverseExpTopDown(exp, f, nothing)
  return found
end

#= Replace every bare (outside-pre) occurrence of a sibling cref with its
   already-inlined RHS. pre(sibling) is left untouched so it keeps reading the
   pre-event held value. =#
Base.@nospecializeinfer function _inlineSiblingsOutsidePre(@nospecialize(exp::DAE.Exp), subst::Dict{String, DAE.Exp})
  function f(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("pre"), _, _) => (e, false, arg)
      DAE.CREF(cr, _) => begin
        local nm = string(cr)
        haskey(subst, nm) ? (subst[nm], false, arg) : (e, true, arg)
      end
      _ => (e, true, arg)
    end
  end
  return first(Util.traverseExpTopDown(exp, f, nothing))
end

#= Start-attribute expression for each variable that has one (Bool/Int/Real/enum).
   Used to fold `pre(member)` at initialization: Modelica §8.6.2 — before the
   first event `pre(v)` is `v.start`. =#
Base.@nospecializeinfer function _discreteStartExpLookup(variables::Vector{BDAE.VAR},
                                   varNames::Vector{String} = String[string(v.varName) for v in variables])::Dict{String, DAE.Exp}
  local d = Dict{String, DAE.Exp}()
  for (i, var) in enumerate(variables)
    local s = @match var.values begin
      SOME(va) => @match va begin
        DAE.VAR_ATTR_BOOL(start = SOME(e)) => e
        DAE.VAR_ATTR_INT(start = SOME(e)) => e
        DAE.VAR_ATTR_REAL(start = SOME(e)) => e
        DAE.VAR_ATTR_ENUMERATION(start = SOME(e)) => e
        _ => nothing
      end
      _ => nothing
    end
    s === nothing && continue
    d[varNames[i]] = s
  end
  return d
end

#= Fold `pre(n)` for n in `members` (a lifted cluster's discretes, and the
   names whose pre() the initial equations fix) to its value in `startLookup`
   for the INITIAL_WHEN body. The default false matches the Boolean start
   default; pre() of other names is left untouched. =#
Base.@nospecializeinfer function _foldPreOfMembers(@nospecialize(exp::DAE.Exp), members::OrderedSet{String},
                                                   startLookup::Dict{String, DAE.Exp})
  function f(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("pre"), args, _) => begin
        local inner = listHead(args)
        if inner isa DAE.CREF && (string(inner.componentRef) in members)
          (get(startLookup, string(inner.componentRef), DAE.BCONST(false)), false, arg)
        else
          (e, false, arg)
        end
      end
      _ => (e, true, arg)
    end
  end
  return first(Util.traverseExpTopDown(exp, f, nothing))
end

#= Connected components over the undirected coupling graph. =#
function _connectedComponents(names::Vector{String}, adj::Dict{String, OrderedSet{String}})::Vector{Vector{String}}
  local seen = OrderedSet{String}()
  local comps = Vector{String}[]
  for start in names
    start in seen && continue
    local comp = String[]
    local stack = String[start]
    push!(seen, start)
    while !isempty(stack)
      local n = pop!(stack)
      push!(comp, n)
      for m in adj[n]
        if !(m in seen)
          push!(seen, m); push!(stack, m)
        end
      end
    end
    push!(comps, comp)
  end
  return comps
end

#= Topological order of a cluster by non-pre dependency edges (dep before the
   member that references it outside pre). Returns the ordered names, or
   `nothing` if a non-pre cycle survives (the cluster is then left unlifted). =#
function _topoOrderCluster(cluster::Vector{String}, candByName::Dict{String, Any})::Union{Vector{String}, Nothing}
  local clusterSet = OrderedSet(cluster)
  local indeg = Dict{String, Int}(n => 0 for n in cluster)
  local succ = Dict{String, Vector{String}}(n => String[] for n in cluster)
  for n in cluster
    for d in _candRefsOutsidePre(candByName[n].rhs, clusterSet)
      if d != n && d in clusterSet
        push!(succ[d], n); indeg[n] += 1
      end
    end
  end
  local q = sort!([n for n in cluster if indeg[n] == 0])
  local order = String[]
  while !isempty(q)
    local n = popfirst!(q)
    push!(order, n)
    for m in succ[n]
      indeg[m] -= 1
      indeg[m] == 0 && push!(q, m)
    end
    sort!(q)
  end
  return length(order) == length(cluster) ? order : nothing
end

#= Emit one cluster of coupled discrete-Boolean equations as a single ordered
   when-cluster: topologically sort, inline non-pre sibling references so each
   ASSIGN RHS is pure in pre-event state + relations, and emit one
   INITIAL_WHEN + one runtime WHEN (triggered by `change()` over the UNION of the
   cluster's relations) whose bodies are the ordered ASSIGN list. =#
function _emitDiscreteCluster!(out::Vector{BDAE.Equation}, cluster::Vector{String},
                               candByName::Dict{String, Any}, liftedLhs::OrderedSet{String},
                               startLookup::Dict{String, DAE.Exp},
                               paramOrConstNames::OrderedSet{String},
                               initialPre::OrderedSet{String},
                               otherRefs::Function, canon::Function)
  local order = _topoOrderCluster(cluster, candByName)
  if order === nothing
    @warn "[BDAE: lifter] cyclic discrete cluster left unlifted" cluster
    for n in cluster; push!(out, candByName[n].eq); end
    return
  end
  local members = OrderedSet(cluster)
  local subst = Dict{String, DAE.Exp}()
  local body = Tuple{DAE.Exp, DAE.Exp, Any}[]
  for n in order
    local c = candByName[n]
    local inlined = isempty(subst) ? c.rhs : _inlineSiblingsOutsidePre(c.rhs, subst)
    subst[n] = inlined
    push!(body, (c.lhs, inlined, c.src))
  end
  #= Trigger only on relations with a continuous operand (real zero-crossings).
     All-discrete relations such as `pre(mode) == Stuck` are kept in the affect
     RHS but excluded as event sources. =#
  local rels = DAE.Exp[]
  local seen = OrderedSet{String}()
  for (_, r, _) in body
    for rel in _collectRelationsInExp(r)
      local k = string(rel)
      if !(k in seen) && _relationHasContinuousOperand(rel, paramOrConstNames)
        push!(seen, k); push!(rels, rel)
      end
    end
  end
  #= No continuous event source: the equations stay residuals, except a
     latch. A residual reads pre(n) as n; where n reads the group back that
     is a Boolean loop: StateGraph's `localActive = pre(newActive)` of a step
     read only through a Parallel (anyTrue and allTrue are no candidates)
     became `localActive = newActive`, which stayed set by a transition whose
     firing the same event took back (ExecutionPaths: step2 never became
     active). Such a group is lifted: pre() changes only at events, where it
     takes the values of the previous pass (MLS 8.6), and the event iteration
     evaluates a lifted cluster at every pass of every event, so it needs no
     relation of its own; its condition `change(pre(n)) or ...` names the
     values it follows (_buildChangeOfPreCondition). A group whose pre()
     reads do not come back (MSL Digital's gates, `y = pre(auxiliary_n)`,
     auxiliary_n from the inputs) stays residual: lifted, it changed only
     when an event iteration ran, and the gates stayed at 'U'. =#
  local held = isempty(rels) ? _preReadCrefs(body) : DAE.Exp[]
  if isempty(rels) && (isempty(held) || !_preReadsCloseLoop(held, members, candByName, otherRefs, canon))
    for n in cluster; push!(out, candByName[n].eq); end
    return
  end
  local bareInitial = DAE.CALL(Absyn.IDENT("initial"), MetaModelica.list(), DAE.callAttrBuiltinBool)
  local changeCond = isempty(rels) ? _buildChangeOfPreCondition(held) : _buildChangeOrConditionFromExps(rels)
  local src0 = body[1][3]
  #= INITIAL body: pre(member) ≡ member.start (no held value exists yet), and
     pre(n) that the initial equations fix. The runtime body keeps pre() — it
     resolves to the affect-entry held value. =#
  local folded = isempty(initialPre) ? members : union(members, initialPre)
  local initAssigns = [BDAE.ASSIGN(lhs, _replaceInitialCall(_foldPreOfMembers(r, folded, startLookup), true), s)
                       for (lhs, r, s) in body]
  local runAssigns  = [BDAE.ASSIGN(lhs, _replaceInitialCall(r, false), s) for (lhs, r, s) in body]
  push!(out, BDAE.INITIAL_WHEN_EQUATION(
    1,
    BDAE.WHEN_STMTS(bareInitial, MetaModelica.list(initAssigns...), NONE()),
    src0,
    BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
  ))
  push!(out, BDAE.WHEN_EQUATION(
    1,
    BDAE.WHEN_STMTS(changeCond, MetaModelica.list(runAssigns...), NONE()),
    src0,
    BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
  ))
  for (lhs, _, _) in body; push!(liftedLhs, string(lhs.componentRef)); end
  return
end

"""
    synthesizeWhenEquationsFromDiscreteEquations(equations, paramOrConstNames, startLookup; initialConstants) -> (Vector{BDAE.Equation}, OrderedSet{String})

Replace qualifying discrete-Boolean/Integer equation-section definitions with
paired INITIAL_WHEN + runtime WHEN equations. Mutually-referencing definitions
(e.g. the Coulomb-friction `{startForward, locked, stuck}` FSM) are grouped into
one ordered cluster so a no-relation member is still event-driven through its
siblings' relations, and the cluster recomputes in topological order on any
member relation crossing. Self-gating: equations that do not qualify are returned
unchanged, so models with no discrete-time defining equations pay only a single
linear scan.
`startLookup` gives the start values pre() takes at initialization, and
`initialConstants` (`_initialConstants`) the values the initial equations fix.
"""
function synthesizeWhenEquationsFromDiscreteEquations(equations::Vector{BDAE.Equation},
                                                      paramOrConstNames::OrderedSet{String} = OrderedSet{String}(),
                                                      startLookup::Dict{String, DAE.Exp} = Dict{String, DAE.Exp}();
                                                      initialConstants = (values = OrderedDict{String, DAE.Exp}(),
                                                                          pre = OrderedDict{String, DAE.Exp}()))
  local out = BDAE.Equation[]
  local cands = Any[]
  #= An alias (`a = b` between discrete variables, a connect in either
     orientation) defines neither side: it stays an equation, for alias
     elimination, and the definitions read through it (_readThroughAliases!).
     Lifting it made the clusters depend on how the connects are oriented (a
     Greater block's `y = u1 > u2` next to `y = fire` was left in the
     continuous equations, its crossing without an event). =#
  local aliases = Tuple{String, String}[]
  for eq in equations
    local c = _discreteBoolCandidate(eq, paramOrConstNames)
    if c === nothing
      push!(out, eq)
    elseif _isDiscreteAlias(c.rhs, paramOrConstNames)
      push!(out, eq)
      push!(aliases, (c.name, string(c.rhs.componentRef)))
    else
      push!(cands, c)
    end
  end
  isempty(cands) && return (equations, OrderedSet{String}())
  #= An LHS with more than one definition is not a discrete definition;
     lifting it through a name-keyed map would silently drop all but one of
     its equations. Keep such equations unchanged. =#
  local lhsCount = Dict{String, Int}()
  for c in cands
    lhsCount[c.name] = get(lhsCount, c.name, 0) + 1
  end
  local candByName = Dict{String, Any}()
  local candNames = String[]
  for c in cands
    if lhsCount[c.name] > 1
      push!(out, c.eq)
    else
      candByName[c.name] = c
      push!(candNames, c.name)
    end
  end
  isempty(candNames) && return (equations, OrderedSet{String}())
  local aliasSets = _aliasSets(aliases)
  local definitionOf = _aliasDefinitions(candByName, aliasSets)
  _readThroughAliases!(candByName, definitionOf)
  local (initialLookup, initialPre) = _initialPreValues(startLookup, candNames, candByName, aliasSets,
                                                        definitionOf, initialConstants)
  local candSet = OrderedSet(candNames)
  local adj = Dict{String, OrderedSet{String}}(n => OrderedSet{String}() for n in candNames)
  for n in candNames
    for r in _candRefsAnywhere(candByName[n].rhs, candSet)
      if r != n
        push!(adj[n], r); push!(adj[r], n)
      end
    end
  end
  local liftedLhs = OrderedSet{String}()
  #= An alias set's name: its definition, else a fixed member. =#
  local repOf = Dict{String, String}()
  for set in aliasSets, n in set
    repOf[n] = get(definitionOf, n, first(set))
  end
  local canon = n -> get(repOf, n, n)
  #= The equations that are not candidates (the first `nOthers` of `out`, which
     only grows), indexed when a group without relations first asks
     (_preReadsCloseLoop). =#
  local nOthers = length(out)
  local index = Ref{Union{Nothing, Dict{String, Vector{OrderedSet{String}}}}}(nothing)
  local otherRefs = () -> (index[] === nothing && (index[] = _refsOutsidePreIndex(view(out, 1:nOthers), canon)); index[])
  for cluster in _connectedComponents(candNames, adj)
    _emitDiscreteCluster!(out, cluster, candByName, liftedLhs, initialLookup, paramOrConstNames, initialPre,
                          otherRefs, canon)
  end
  return (out, liftedLhs)
end
