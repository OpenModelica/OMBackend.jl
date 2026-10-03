#= MTK code generation: the relations of whens and discrete clusters (change conditions,
   relations read through, zero sets and their pins). =#

#= A synthesized cluster's body can read a Boolean defined by a relation elsewhere
   (an ideal thyristor's `fire`, the output of a comparison block). The body reads
   the relation instead (outside pre()), and the relation joins the cluster's
   event sources: the cluster is evaluated again when it changes, as every
   discrete equation is at every event (MLS 8.6), with the relation pinned to its
   value after the crossing (at the crossing itself `fire` can still read false).
   Returns the relations and the body. =#
function _inlineReadBooleanRelations(rels::Vector{DAE.Exp}, assigns::Vector{Tuple{Symbol,Any,Bool}},
                                     relationOf::Dict{String, DAE.Exp}, simCode)
  local members = Set{String}(string(d) for (d, _, _) in assigns)
  local seen = Set{String}(string(r) for r in rels)
  local subst = Dict{String, DAE.Exp}()
  local rels2 = copy(rels)
  for (_, rhs, _) in assigns, name in _crefNamesOutsidePre(rhs)
    (name in members || haskey(subst, name)) && continue
    local rel = get(relationOf, name, nothing)
    #= A relation of discretes only changes at events the cluster already sees
       (the lifter does not make such relations event sources either). =#
    (rel === nothing || _withoutContinuousOperand(rel, simCode)) && continue
    subst[name] = rel
    string(rel) in seen || (push!(seen, string(rel)); push!(rels2, rel))
  end
  isempty(subst) && return (rels, assigns)
  return (rels2, Tuple{Symbol,Any,Bool}[(d, Backend.BDAECreate._inlineSiblingsOutsidePre(rhs, subst), isInt)
                                        for (d, rhs, isInt) in assigns])
end

#= The names of the crefs an expression reads outside pre(). =#
function _crefNamesOutsidePre(@nospecialize(exp::DAE.Exp))::Set{String}
  local names = Set{String}()
  function f(@nospecialize(e), acc)
    @match e begin
      DAE.CALL(Absyn.IDENT("pre"), _, _) => (e, false, acc)
      DAE.CREF(cr, _) => (push!(acc, string(cr)); (e, false, acc))
      _ => (e, true, acc)
    end
  end
  Util.traverseExpTopDown(exp, f, names)
  return names
end

#= Identify a synthesised discrete-Boolean when (from
   `synthesizeWhenEquationsFromDiscreteEquations`): its condition is `change(rel)`
   or an OR-chain of `change(rel)` over relations. Returns the relation list
   (DAE side) or `nothing`. Such whens are routed to MTK events (not the legacy
   CallbackSet) so the relation operands resolve as MTK observed variables. =#
#= Resolve a Boolean condition variable to its defining relation: a residual
   `0 ~ v - REL` (e.g. `above = x > 0.1`). Returns the relation as a DAE.Exp, or
   nothing. Lets `change(b)`/`edge(b)` over an observed Boolean route to an MTK
   SymbolicContinuousCallback (which reads observed vars + root-finds) instead of
   the legacy CallbackSet (which cannot read the observed `b`). =#
function _condVarRelation(crefName::AbstractString, simCode)
  for req in simCode.residualEquations
    local r = _definingRelation(req)
    r === nothing || r[1] != crefName || return r[2]
  end
  return nothing
end

#= (name, relation) of a residual `0 ~ v - REL`, or nothing. =#
function _definingRelation(@nospecialize(req))
  req isa SimulationCode.RESIDUAL_EQUATION || return nothing
  local b = req.exp
  (b isa SimulationCode.BINARY && b.op === SimulationCode.OP_SUB &&
   b.exp1 isa SimulationCode.EXP_CREF && b.exp2 isa SimulationCode.RELATION) || return nothing
  return (string(SimulationCode.toDAEExp(b.exp1).componentRef), SimulationCode.toDAEExp(b.exp2))
end

#= Every Boolean defined by a relation (_condVarRelation), by name. =#
function _condVarRelations(simCode)::Dict{String, DAE.Exp}
  local out = Dict{String, DAE.Exp}()
  for req in simCode.residualEquations
    local r = _definingRelation(req)
    r === nothing || get!(out, r[1], r[2])
  end
  return out
end

#= Collect the zero-crossing relations of a `change(...)`/`edge(...)` condition
   (or an OR-chain of them). The argument may be a relation directly or a
   Boolean variable defined by a relation (resolved via `_condVarRelation`). =#
function _collectChangeRelations!(rels::Vector{DAE.Exp}, @nospecialize(e), simCode)::Bool
  @match e begin
    DAE.CALL(Absyn.IDENT("change"), args, _) || DAE.CALL(Absyn.IDENT("edge"), args, _) => begin
      local inner = listHead(args)
      if inner isa DAE.RELATION
        push!(rels, inner); true
      elseif inner isa DAE.CREF
        local rel = _condVarRelation(string(inner.componentRef), simCode)
        rel === nothing ? false : (push!(rels, rel); true)
      else
        #= A lifted cluster without relations: no crossing to add. =#
        _isChangeOfPre(e)
      end
    end
    DAE.LBINARY(e1, DAE.OR(__), e2) =>
      (_collectChangeRelations!(rels, e1, simCode) && _collectChangeRelations!(rels, e2, simCode))
    _ => false
  end
end

#= `change(pre(v))`: a term of the condition of a cluster the lifter made
   without relations (BDAECreate._buildChangeOfPreCondition); invalid in
   Modelica, so no user when has it. =#
_isChangeOfPre(@nospecialize(e))::Bool =
  e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name == "change" && _isPreRead(listHead(e.expLst))

#= pre(v), also behind the sign a negated alias substitution puts in front. =#
Base.@nospecializeinfer function _isPreRead(@nospecialize(a))::Bool
  @match a begin
    DAE.CALL(Absyn.IDENT("pre"), _, _) => true
    DAE.UNARY(_, x) || DAE.LUNARY(_, x) => _isPreRead(x)
    _ => false
  end
end

#= The relations of a lifted when's condition (empty for a cluster without
   relations, which only the event iteration evaluates), or nothing. =#
function _extractChangeRelations(@nospecialize(cond), simCode)
  local dcond = cond isa SimulationCode.Exp ? SimulationCode.toDAEExp(cond) : cond
  local rels = DAE.Exp[]
  local ok = _collectChangeRelations!(rels, dcond, simCode)
  return ok ? rels : nothing
end

#= True when the when condition marks a synthesized (lifter) when: a literal
   `initial()` term or a `change()`/`edge()` call. Such whens carry the
   implied §17.4.4 initial() term and run their body at t0; user whens with
   bare relation conditions must not. =#
function _condHasInitial(@nospecialize(e))::Bool
  local d = e isa SimulationCode.Exp ? SimulationCode.toDAEExp(e) : e
  @match d begin
    DAE.CALL(Absyn.IDENT("initial"), _, _) => true
    DAE.CALL(Absyn.IDENT("change"), _, _) => true
    DAE.CALL(Absyn.IDENT("edge"), _, _) => true
    DAE.LBINARY(e1, _, e2) => (_condHasInitial(e1) || _condHasInitial(e2))
    DAE.LUNARY(_, e1) => _condHasInitial(e1)
    _ => false
  end
end

#= True when the when-condition is `edge(...)` (or an OR-chain of `edge`), which
   fires on the RISING transition only (false->true), unlike `change` (both). =#
function _isEdgeWhenCondition(@nospecialize(e))::Bool
  local d = e isa SimulationCode.Exp ? SimulationCode.toDAEExp(e) : e
  @match d begin
    DAE.CALL(Absyn.IDENT("edge"), _, _) => true
    DAE.LBINARY(e1, DAE.OR(__), e2) => (_isEdgeWhenCondition(e1) && _isEdgeWhenCondition(e2))
    _ => false
  end
end

#= Replace every structural occurrence of each relation in `pins` (keys:
   string(relation)) with its pinned value. =#
function _substRelations(@nospecialize(exp), pins::Dict{String,Bool})
  function repl(e::DAE.Exp, arg)
    local v = e isa DAE.RELATION ? get(pins, string(e), nothing) : nothing
    v === nothing ? (e, arg) : (DAE.BCONST(v), arg)
  end
  return first(Util.traverseExpBottomUp(exp, repl, nothing))
end

#= The zero set of an ordering relation: its two operands in a canonical order,
   and whether the relation holds when the first canonical operand is the
   larger one. `a <= b`, `a > b`, `b >= a`, `b < a` all switch where a - b
   crosses zero. `nothing` for relations that are not orderings (==, <>). =#
function _relationZeroSet(@nospecialize(rel))::Union{Nothing, Tuple{String, String, Bool}}
  rel isa DAE.RELATION || return nothing
  local greater = @match rel.operator begin
    DAE.GREATER(__) => true
    DAE.GREATEREQ(__) => true
    DAE.LESS(__) => false
    DAE.LESSEQ(__) => false
    _ => nothing
  end
  greater === nothing && return nothing
  local s1 = string(rel.exp1)
  local s2 = string(rel.exp2)
  return s1 <= s2 ? (s1, s2, greater) : (s2, s1, !greater)
end

#= Group relations that share a zero set, in first-occurrence order. =#
function _groupRelationsByZeroSet(rels)::Vector{Vector{Any}}
  local groups = Vector{Vector{Any}}()
  local groupOf = Dict{Tuple{String,String},Int}()
  for r in rels
    local zs = _relationZeroSet(r)
    if zs === nothing
      push!(groups, Any[r])
      continue
    end
    local key = (zs[1], zs[2])
    if haskey(groupOf, key)
      push!(groups[groupOf[key]], r)
    else
      push!(groups, Any[r])
      groupOf[key] = length(groups)
    end
  end
  return groups
end

#= Values of every relation in a zero-set group once `rel` has value `val`. =#
function _zeroSetPins(group, @nospecialize(rel), val::Bool)::Dict{String,Bool}
  local pins = Dict{String,Bool}(string(rel) => val)
  local zs = _relationZeroSet(rel)
  zs === nothing && return pins
  local firstIsLarger = (val == zs[3])
  for r in group
    local rzs = _relationZeroSet(r)
    rzs === nothing || (pins[string(r)] = (firstIsLarger == rzs[3]))
  end
  return pins
end
