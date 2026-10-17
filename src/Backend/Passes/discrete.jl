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

#= Discrete-time equations as when-equations (the previous backend's discreteSynthesis.jl):
   a discrete variable changes only at events (MLS 8.6). =#

#= ==================================================================================== =#
#= Booleans for the relations of compound when-conditions                              =#
#= ==================================================================================== =#

"""
    booleanizeWhenConditions!(model)

A when-condition that combines relations (`x > 0.3 and y > 0.5`, `x < 0.2 or x > 0.6`, `not
x > 0.5`) fires when the combination becomes true, its relations read at their crossings
(MLS 8.5). The callbacks have a crossing for one relation: each relation of such a condition
with a continuous operand becomes a Boolean variable defined by it (`__whenConditionK = x >
0.3`, an equation `liftDiscreteEquations!` lifts), and the when is on Booleans, fired on an
edge. A single relation keeps its own lowering. Left alone: conditions with `sample()` or
`terminal()`, and relations calling `pre()`, `edge()` or `change()` (self-scheduling time
whens). A vector of triggers `{c1, c2}` fires at each element's rising edge: an element that
is not a Boolean variable, a constant, `initial()`, `sample()` or a reading of `pre()`, `edge()`,
`change()` becomes one defined by it.
"""
function booleanizeWhenConditions!(model::IR.Model)::IR.Model
  for p in model.continuous
    booleanizeWhenConditions!(model.variables, p)
  end
  for mode in model.modes
    booleanizeWhenConditions!(mode.variables, mode.system)
  end
  return model
end

function booleanizeWhenConditions!(variables::IR.VariableSet, partition::IR.Partition)::Nothing
  isempty(variables) && return nothing
  any(eq -> eq isa IR.WHEN_EQUATION && any(needsConditionBooleans, eq.branches), partition.equations) || return nothing
  local known = knownNames(variables)
  local count = Ref(0)
  local define = e -> begin
    local name = :_
    while true
      count[] += 1
      name = Symbol("__whenCondition", count[])
      haskey(variables, name) || break
    end
    push!(variables, IR.Variable(name, IR.T_BOOLEAN(), IR.DISCRETE(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                                 false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :booleanizeWhenConditions))
    local lhs = IR.CREF(IR.ComponentRef(name, IR.T_BOOLEAN()), IR.T_BOOLEAN())
    push!(partition.equations, IR.SCALAR_EQUATION(lhs, e, IR.T_BOOLEAN(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    return lhs
  end
  local rewrite
  rewrite = e -> begin
    e isa IR.LBINARY && return IR.LBINARY(rewrite(e.exp1), e.op, rewrite(e.exp2))
    e isa IR.LUNARY && return IR.LUNARY(e.op, rewrite(e.exp))
    (e isa IR.RELATION && hasContinuousOperand(e, known) && !callsAnyOf(e, (:pre, :edge, :change))) && return define(e)
    return e
  end
  local rewriteBranch = b -> begin
    if isVectorCondition(b)
      return IR.WhenBranch(IR.Expression[isTriggerAtom(t) ? t : define(t) for t in b.triggers], b.atInitial, b.atTerminal, b.body)
    end
    return isCompoundCondition(b) ? IR.WhenBranch(IR.Expression[rewrite(b.triggers[1])], false, false, b.body) : b
  end
  for eq in partition.equations
    eq isa IR.WHEN_EQUATION || continue
    #= the equations `define` appends are no when-equations =#
    eq.branches = IR.WhenBranch[rewriteBranch(b) for b in eq.branches]
  end
  return nothing
end

"A when-branch whose condition is a vector `{c1, c2, ...}` (`initial()` and `terminal()` count)."
isVectorCondition(b::IR.WhenBranch)::Bool = length(b.triggers) + b.atInitial + b.atTerminal > 1

"Whether a branch's condition gets Boolean variables: a compound one, or a vector with an element that is not a trigger."
needsConditionBooleans(b::IR.WhenBranch)::Bool =
  isVectorCondition(b) ? !all(isTriggerAtom, b.triggers) : isCompoundCondition(b)

"A condition combining relations with and, or, not, without `sample()` or `terminal()`."
function isCompoundCondition(b::IR.WhenBranch)::Bool
  (length(b.triggers) == 1 && !b.atTerminal && !b.atInitial) || return false
  local c = b.triggers[1]
  callsAnyOf(c, (:sample, :terminal)) && return false
  return c isa IR.LBINARY || c isa IR.LUNARY
end

"""
An element of a vector of triggers kept as it is: a Boolean variable, a constant, `sample()`,
and an expression reading `pre()`, `edge()` or `change()` (the MSL tables' self-scheduling
`time >= pre(nextTimeEvent)`).
"""
isTriggerAtom(e::IR.Expression)::Bool =
  e isa IR.CREF || e isa IR.BOOLEAN || isBuiltinCallOf(e, (:initial, :sample, :terminal)) ||
  callsAnyOf(e, (:pre, :edge, :change))

"Whether `e` calls one of the built-ins `names`."
callsAnyOf(e::IR.Expression, names::Tuple{Vararg{Symbol}})::Bool = IR.anyExp(x -> x isa IR.CALL && x.name in names, e)

#= ==================================================================================== =#
#= Discrete equations as when-clusters                                                  =#
#= ==================================================================================== =#

"""
    Candidate(name, lhs, rhs, source, equation)

A discrete equation `lhs = rhs` (Boolean, Integer, enumeration) whose right-hand side is a
discrete-time expression, with the name of its variable.
"""
struct Candidate
  name::Symbol
  lhs::IR.CREF
  rhs::IR.Expression
  source::IR.Source
  equation::IR.Equation
end

withRhs(c::Candidate, rhs::IR.Expression)::Candidate = Candidate(c.name, c.lhs, rhs, c.source, c.equation)

"""
    liftDiscreteEquations!(model)

The discrete definitions of the equations (MLS 17.4.4: a discrete variable defined by
`v = e`, `e` a discrete-time expression, keeps its value between events) as an initial
when-equation (its value at the initialization) and a when-equation on `change()` of the
relations with a continuous operand in `e`, after the other equations. Definitions that
reference each other (the Coulomb friction's `{startForward, locked, stuck}`) are one cluster,
in topological order, its siblings inlined (a definition without relations is event-driven
through its siblings' relations). Left as equations: aliases `a = b` (they define neither
side; the definitions read through them), sample triggers only when-equations read, variables
with more than one definition, cyclic clusters, clusters without an event source (but a
latch on `pre()` or one with `edge()`/`change()`), the definitions of clocked partitions'
variables.
"""
function liftDiscreteEquations!(model::IR.Model)::IR.Model
  for p in model.continuous
    liftDiscreteEquations!(model.variables, p)
  end
  for mode in model.modes
    liftDiscreteEquations!(mode.variables, mode.system)
  end
  return model
end

function liftDiscreteEquations!(variables::IR.VariableSet, partition::IR.Partition)::Nothing
  (isempty(partition.equations) || isempty(variables)) && return nothing
  local known = knownNames(variables)
  local starts = startValues(variables)
  local constants = initialConstants(partition.initialEquations, known)
  partition.equations = first(liftDiscreteEquations(partition.equations, known, starts, constants, variables))
  return nothing
end

"""
Discrete-time predicate: no continuous Real dependence outside a relation (relations are event
sources, their operands may be continuous).
"""
function isDiscreteTime(e::IR.Expression, known::Set{Symbol})::Bool
  e isa IR.RELATION && return true
  e isa IR.LBINARY && return isDiscreteTime(e.exp1, known) && isDiscreteTime(e.exp2, known)
  e isa IR.LUNARY && return isDiscreteTime(e.exp, known)
  e isa Union{IR.BOOLEAN, IR.INTEGER, IR.REAL, IR.STRING, IR.ENUM_LITERAL} && return true
  e isa IR.CALL && return e.name in (:pre, :initial, :change, :edge, :sample, :noEvent)
  e isa IR.IF_EXPRESSION && return isDiscreteTime(e.condition, known) && isDiscreteTime(e.thenExp, known) &&
                                   isDiscreteTime(e.elseExp, known)
  e isa IR.CREF && return e.cref.name in known || !isRealType(e.ty)
  return false
end

"A discrete definition `v = e` (an equation not classified yet), or `nothing`."
function candidateOf(eq::IR.Equation, known::Set{Symbol})::Union{Nothing, Candidate}
  (eq isa IR.SCALAR_EQUATION && eq.attributes.kind === IR.EQ_UNKNOWN) || return nothing
  local lhs = eq.lhs
  (lhs isa IR.CREF && (isDiscreteType(lhs.ty) || isDiscreteType(lhs.cref.parts[end].ty))) || return nothing
  local name = lhs.cref.name
  (name === :time || name in known) && return nothing
  isDiscreteTime(eq.rhs, known) || return nothing
  return Candidate(name, lhs, eq.rhs, eq.source, eq)
end

"""
`sample(...)`, or a conjunction with one `sample()` conjunct and no other `sample()`
(Buildings' plotters: `sampleTrigger = active and sample(t0, samplePeriod)`).
"""
function isSampleTrigger(e::IR.Expression)::Bool
  local conjuncts = IR.Expression[]
  local collect!
  collect! = x -> (x isa IR.LBINARY && x.op === IR.AND) ? (collect!(x.exp1); collect!(x.exp2)) : push!(conjuncts, x)
  collect!(e)
  local isSample = x -> IR.isBuiltinCall(x, :sample)
  return count(isSample, conjuncts) == 1 && !any(c -> !isSample(c) && callsAnyOf(c, (:sample,)), conjuncts)
end

"Per name, how many equations other than when-equations mention it."
function readsOutsideWhens(equations::AbstractVector{IR.Equation})::Dict{Symbol, Int}
  local counts = Dict{Symbol, Int}()
  for eq in equations
    eq isa IR.WHEN_EQUATION && continue
    local refs = Set{Symbol}()
    IR.foreachExp(e -> e isa IR.CREF && push!(refs, e.cref.name), eq)
    foreach(r -> counts[r] = get(counts, r, 0) + 1, refs)
  end
  return counts
end

"The start values of the variables that have one: `pre(v)` before the first event (MLS 8.6.2)."
function startValues(variables::IR.VariableSet)::Dict{Symbol, IR.Expression}
  local out = Dict{Symbol, IR.Expression}()
  for v in variables
    v.attributes.start === nothing && continue
    IR.elementType(v.ty) isa Union{IR.T_BOOLEAN, IR.T_INTEGER, IR.T_REAL, IR.T_ENUMERATION} || continue
    out[v.name] = v.attributes.start
  end
  return out
end

"The initial equations' constants (`initialConstants`): values and `pre()` values by name, each in their order."
const InitialConstants = NamedTuple{(:values, :valueOrder, :pre, :preOrder),
                                    Tuple{Dict{Symbol, IR.Expression}, Vector{Symbol}, Dict{Symbol, IR.Expression}, Vector{Symbol}}}

"""
    initialConstants(initialEquations, known)

The initial equations that fix a variable or a `pre()` value to a literal or a parameter
(`pre(y) = pre_y_start`, the MSL Hysteresis and Pre blocks): `values` (name => value) and `pre`
(name => value for `pre(name) = value`).
"""
function initialConstants(initialEquations::Vector{IR.Equation}, known::Set{Symbol})::InitialConstants
  local values = Dict{Symbol, IR.Expression}()
  local valueOrder = Symbol[]
  local pre = Dict{Symbol, IR.Expression}()
  local preOrder = Symbol[]
  local isValue = e -> e isa Union{IR.BOOLEAN, IR.INTEGER, IR.REAL, IR.ENUM_LITERAL} || (e isa IR.CREF && e.cref.name in known)
  for eq in initialEquations
    eq isa IR.SCALAR_EQUATION || continue
    for (a, b) in ((eq.lhs, eq.rhs), (eq.rhs, eq.lhs))
      isValue(b) || continue
      if a isa IR.CREF
        haskey(values, a.cref.name) || push!(valueOrder, a.cref.name)
        values[a.cref.name] = b
      else
        local x = preArgument(a)
        x === nothing && continue
        haskey(pre, x) || push!(preOrder, x)
        pre[x] = b
      end
    end
  end
  return (values = values, valueOrder = valueOrder, pre = pre, preOrder = preOrder)
end

"The name `n` of `pre(n)`, else `nothing`."
preArgument(e::IR.Expression)::Union{Nothing, Symbol} =
  (IR.isBuiltinCall(e, :pre) && e.arguments[1] isa IR.CREF) ? e.arguments[1].cref.name : nothing

"The alias sets of the discrete aliases `a = b`, in the aliases' order."
function aliasSets(aliases::Vector{Tuple{Symbol, Symbol}})::Vector{Vector{Symbol}}
  local names = Symbol[]
  local adjacent = Dict{Symbol, Vector{Symbol}}()
  local add = (a, b) -> (local l = get!(adjacent, a, Symbol[]); b in l || push!(l, b))
  for (a, b) in aliases
    a in names || push!(names, a)
    b in names || push!(names, b)
    add(a, b); add(b, a)
  end
  return connectedComponents(names, adjacent)
end

"The connected components over the undirected graph `adjacent`, in the order of `names`."
function connectedComponents(names::Vector{Symbol}, adjacent::Dict{Symbol, Vector{Symbol}})::Vector{Vector{Symbol}}
  local seen = Set{Symbol}()
  local components = Vector{Symbol}[]
  for start in names
    start in seen && continue
    local component = Symbol[]
    local stack = Symbol[start]
    push!(seen, start)
    while !isempty(stack)
      local n = pop!(stack)
      push!(component, n)
      for m in get(adjacent, n, Symbol[])
        m in seen || (push!(seen, m); push!(stack, m))
      end
    end
    push!(components, component)
  end
  return components
end

"name => the name of its alias set's definition, for the sets with exactly one definition among the candidates."
function aliasDefinitions(candidates::Dict{Symbol, Candidate}, sets::Vector{Vector{Symbol}})::Dict{Symbol, Symbol}
  local definitionOf = Dict{Symbol, Symbol}()
  for set in sets
    local defs = filter(n -> haskey(candidates, n), set)
    length(defs) == 1 || continue
    foreach(n -> definitionOf[n] = only(defs), set)
  end
  return definitionOf
end

"""
Each candidate's right-hand side reading the definition of an alias set in place of its
aliases (`pre()` included: an alias holds at every instant), so the clusters do not depend on
how the connects are oriented.
"""
function readThroughAliases!(candidates::Dict{Symbol, Candidate}, definitionOf::Dict{Symbol, Symbol})::Nothing
  isempty(definitionOf) && return nothing
  local subst = e -> begin
    e isa IR.CREF || return e
    local d = get(definitionOf, e.cref.name, e.cref.name)
    return d === e.cref.name ? e : IR.CREF(candidates[d].lhs.cref, e.ty)
  end
  for n in collect(keys(candidates))
    candidates[n] = withRhs(candidates[n], IR.mapExp(subst, candidates[n].rhs))
  end
  return nothing
end

"""
The start lookup that folds `pre()` at the initialization, and the names whose `pre()` the
initial equations fix: on the definitions' names (a start only an alias carries is the
definition's); `pre(n) = v`; and `pre(n)` of a candidate `m = pre(n)` whose value an initial
equation fixes, directly or through an alias (the StateGraph InitialStep).
"""
function initialPreValues(starts::Dict{Symbol, IR.Expression}, names::Vector{Symbol}, candidates::Dict{Symbol, Candidate},
                          sets::Vector{Vector{Symbol}}, definitionOf::Dict{Symbol, Symbol}, constants::InitialConstants)
  local fixed = Symbol[]
  (isempty(sets) && isempty(constants.values) && isempty(constants.pre)) && return (starts, fixed)
  local lookup = copy(starts)
  local valueOf = Dict{Symbol, IR.Expression}()
  for set in sets
    local d = get(definitionOf, first(set), nothing)
    if d !== nothing && !haskey(starts, d)
      local i = findfirst(n -> haskey(starts, n), set)
      i === nothing || (lookup[d] = starts[set[i]])
    end
    local j = findfirst(n -> haskey(constants.values, n), set)
    j === nothing || foreach(n -> valueOf[n] = constants.values[set[j]], set)
  end
  for x in constants.preOrder
    local d = get(definitionOf, x, x)
    lookup[d] = constants.pre[x]
    d in fixed || push!(fixed, d)
  end
  for n in names
    local x = preArgument(candidates[n].rhs)
    x === nothing && continue
    local v = get(constants.values, n, get(valueOf, n, nothing))
    v === nothing && continue
    lookup[x] = v
    x in fixed || push!(fixed, x)
  end
  return (lookup, fixed)
end

"The names of `restrict` referenced anywhere in `e` (`pre()` included), in order."
function namesReferenced(e::IR.Expression, restrict::Set{Symbol})::Vector{Symbol}
  local out = Symbol[]
  local visit
  visit = x -> (foreach(visit, IR.children(x)); x isa IR.CREF && x.cref.name in restrict && !(x.cref.name in out) && push!(out, x.cref.name))
  visit(e)
  return out
end

"The names of `restrict` referenced outside any `pre()`: the topological order's edges."
function namesReferencedOutsidePre(e::IR.Expression, restrict::Set{Symbol})::Vector{Symbol}
  local out = Symbol[]
  local visit
  visit = x -> begin
    IR.isBuiltinCall(x, :pre) && return
    x isa IR.CREF && x.cref.name in restrict && !(x.cref.name in out) && push!(out, x.cref.name)
    foreach(visit, IR.children(x))
  end
  visit(e)
  return out
end

"""
`e` with each sibling read outside `pre()`, `edge()` and `change()` replaced by its inlined
right-hand side: `pre(sibling)` keeps reading the value before the event, and `edge()` and
`change()` read `pre()` of their argument, which must stay a variable.
"""
inlineSiblings(e::IR.Expression, subst::Dict{Symbol, IR.Expression})::IR.Expression =
  IR.mapExpTopDown(e) do x
    isBuiltinCallOf(x, (:pre, :edge, :change)) && return (x, false)
    (x isa IR.CREF && haskey(subst, x.cref.name)) && return (subst[x.cref.name], false)
    return (x, true)
  end

"""
`e` with `pre(n)` for `n` in `members` replaced by its value at the initialization (for one
without a start value, its type's default: `false`, `0`, the first literal).
"""
foldPreOfMembers(e::IR.Expression, members::Set{Symbol}, starts::Dict{Symbol, IR.Expression})::IR.Expression =
  IR.mapExpTopDown(e) do x
    if IR.isBuiltinCall(x, :pre)
      local a = x.arguments[1]
      (a isa IR.CREF && a.cref.name in members) && return (get(() -> defaultStart(a.ty), starts, a.cref.name), false)
      return (x, false)
    end
    return (x, true)
  end

"The start value of a discrete variable without one (MLS 4.8.1): `false`, `0`, the enumeration's first literal, `0.0`."
function defaultStart(ty::IR.ModelicaType)::IR.Expression
  local t = IR.elementType(ty)
  t isa IR.T_INTEGER && return IR.INTEGER(0)
  t isa IR.T_ENUMERATION && return IR.dimensionIndex(IR.DIM_ENUM(t), 1)
  t isa IR.T_REAL && return IR.REAL(0.0)
  return IR.BOOLEAN(false)
end

"""
The topological order of a cluster by its references outside `pre()` (a member after the
members it reads), the names ready at each step in the previous backend's name order;
`nothing` for a cycle (the cluster is left as equations).
"""
function topologicalOrder(cluster::Vector{Symbol}, candidates::Dict{Symbol, Candidate})::Union{Nothing, Vector{Symbol}}
  local members = Set{Symbol}(cluster)
  local indegree = Dict{Symbol, Int}(n => 0 for n in cluster)
  local successors = Dict{Symbol, Vector{Symbol}}(n => Symbol[] for n in cluster)
  for n in cluster, d in namesReferencedOutsidePre(candidates[n].rhs, members)
    d === n && continue
    push!(successors[d], n)
    indegree[n] += 1
  end
  local byName = names -> sort!(names; by = n -> legacyName(candidates[n].lhs.cref))
  local queue = byName([n for n in cluster if indegree[n] == 0])
  local order = Symbol[]
  while !isempty(queue)
    local n = popfirst!(queue)
    push!(order, n)
    for m in successors[n]
      indegree[m] -= 1
      indegree[m] == 0 && push!(queue, m)
    end
    byName(queue)
  end
  return length(order) == length(cluster) ? order : nothing
end

"The variables a body reads through `pre()`, `edge()` or `change()`, once each."
function preReads(body::Vector{Tuple{IR.CREF, IR.Expression, IR.Source}})::Vector{IR.CREF}
  local found = IR.CREF[]
  local visit
  visit = x -> begin
    foreach(visit, IR.children(x))
    if isBuiltinCallOf(x, (:pre, :edge, :change)) && x.arguments[1] isa IR.CREF
      local a = x.arguments[1]
      any(f -> f.cref.name === a.cref.name, found) || push!(found, a)
    end
  end
  foreach(b -> visit(b[2]), body)
  return found
end

"""
Whether a variable the group reads through `pre()` reads a member outside `pre()`: its own
definition (a member), or another equation it is in (`otherRefs()`: per name, the names each
other equation reads outside `pre()`, under `canon`).
"""
function preReadsCloseLoop(held::Vector{IR.CREF}, members::Set{Symbol}, candidates::Dict{Symbol, Candidate},
                           otherRefs::F, canon::G)::Bool where {F, G}
  for h in held
    local n = canon(h.cref.name)
    if n in members
      isempty(namesReferencedOutsidePre(candidates[n].rhs, members)) || return true
    else
      any(refs -> any(m -> m in refs, members), get(otherRefs(), n, Set{Symbol}[])) && return true
    end
  end
  return false
end

"Per name, the sets of names the equations that read it outside `pre()` read outside `pre()`, under `canon`."
function refsOutsidePreIndex(equations::AbstractVector{IR.Equation}, canon::F)::Dict{Symbol, Vector{Set{Symbol}}} where {F}
  local index = Dict{Symbol, Vector{Set{Symbol}}}()
  for eq in equations
    local refs = Set{Symbol}()
    local visit
    visit = x -> begin
      IR.isBuiltinCall(x, :pre) && return
      x isa IR.CREF && push!(refs, canon(x.cref.name))
      foreach(visit, IR.children(x))
    end
    IR.foreachRootExp(visit, eq)
    for r in refs
      push!(get!(index, r, Set{Symbol}[]), refs)
    end
  end
  return index
end

"`change(c1) or change(c2) or ...` over the relations."
changeCondition(relations::Vector{IR.Expression})::IR.Expression =
  isempty(relations) ? IR.BOOLEAN(false) : foldl((a, b) -> IR.LBINARY(a, IR.OR, b), IR.Expression[changeCall(r) for r in relations])

"""
The condition of a lifted cluster without relations: `change(pre(n1)) or ...` over the values
it follows. Not Modelica (the argument of change() is a variable): the code generation takes
it for a cluster with no relation and never evaluates it (the event iteration runs the body at
every pass).
"""
function changeOfPreCondition(held::Vector{IR.CREF})::IR.Expression
  isempty(held) && unsupported("a discrete cluster without relations or variables it follows", "")
  return foldl((a, b) -> IR.LBINARY(a, IR.OR, b),
               IR.Expression[changeCall(builtinCall(:pre, IR.Expression[h], IR.T_BOOLEAN())) for h in held])
end

"""
One cluster of coupled discrete equations as a when-cluster: in topological order, the
siblings read outside `pre()` inlined, an initial when-equation (`pre(member)` its start value,
`initial()` true) and a when-equation on `change()` of the cluster's relations with a
continuous operand (`initial()` false). A cluster without an event source stays equations,
but one with `edge()` or `change()` and a latch whose `pre()` reads come back (StateGraph's
`localActive = pre(newActive)` through a Parallel): the event iteration evaluates a lifted
cluster at every pass of every event.
"""
function emitCluster!(out::Vector{IR.Equation}, cluster::Vector{Symbol}, candidates::Dict{Symbol, Candidate},
                      lifted::Vector{Symbol}, starts::Dict{Symbol, IR.Expression}, known::Set{Symbol},
                      initialPre::Vector{Symbol}, otherRefs::F, canon::G)::Nothing where {F, G}
  local order = topologicalOrder(cluster, candidates)
  if order === nothing
    @warn "[OMJLIR: liftDiscreteEquations] cyclic discrete cluster left unlifted" cluster
    foreach(n -> push!(out, candidates[n].equation), cluster)
    return nothing
  end
  local members = Set{Symbol}(cluster)
  local subst = Dict{Symbol, IR.Expression}()
  local body = Tuple{IR.CREF, IR.Expression, IR.Source}[]
  for n in order
    local c = candidates[n]
    local inlined = isempty(subst) ? c.rhs : inlineSiblings(c.rhs, subst)
    subst[n] = inlined
    push!(body, (c.lhs, inlined, c.source))
  end
  local relations = IR.Expression[]
  for (_, r, _) in body, rel in relationsIn(r)
    (rel in relations || !hasContinuousOperand(rel, known)) && continue
    push!(relations, rel)
  end
  local held = isempty(relations) ? preReads(body) : IR.CREF[]
  local eventCalls = isempty(relations) && any(b -> callsAnyOf(b[2], (:edge, :change)), body)
  if isempty(relations) && !eventCalls && (isempty(held) || !preReadsCloseLoop(held, members, candidates, otherRefs, canon))
    foreach(n -> push!(out, candidates[n].equation), cluster)
    return nothing
  end
  local condition = isempty(relations) ? changeOfPreCondition(held) : changeCondition(relations)
  local folded = isempty(initialPre) ? members : union(members, initialPre)
  local initBody = IR.Statement[IR.ASSIGNMENT(lhs, replaceInitial(foldPreOfMembers(r, folded, starts), true), s) for (lhs, r, s) in body]
  local runBody = IR.Statement[IR.ASSIGNMENT(lhs, replaceInitial(r, false), s) for (lhs, r, s) in body]
  local source = body[1][3]
  push!(out, initialWhen(initBody, source))
  push!(out, IR.WHEN_EQUATION([IR.WhenBranch(IR.Expression[condition], false, false, runBody)], IR.REGULAR_WHEN, source,
                              IR.DEFAULT_ATTRIBUTES))
  foreach(b -> push!(lifted, b[1].cref.name), body)
  return nothing
end

"""
    liftDiscreteEquations(equations, known, starts, constants, variables)

The equations with the discrete definitions lifted (`liftDiscreteEquations!`): the others in
their order, then the clusters. With the names lifted.
"""
function liftDiscreteEquations(equations::Vector{IR.Equation}, known::Set{Symbol}, starts::Dict{Symbol, IR.Expression},
                               constants::InitialConstants, variables::IR.VariableSet)::Tuple{Vector{IR.Equation}, Vector{Symbol}}
  local out = IR.Equation[]
  local candidates = Candidate[]
  local aliases = Tuple{Symbol, Symbol}[]
  local reads = Ref{Union{Nothing, Dict{Symbol, Int}}}(nothing)
  local readCount = name -> get(reads[] === nothing ? (reads[] = readsOutsideWhens(equations)) : reads[], name, 0)
  for eq in equations
    local c = candidateOf(eq, known)
    if c === nothing
      push!(out, eq)
    elseif isSampleTrigger(c.rhs) && readCount(c.name) <= 1
      #= A sample trigger only when-equations read: it stays, and the SimCode lowering puts
         it into the whens (lifted, it missed every tick). =#
      push!(out, eq)
    elseif c.rhs isa IR.CREF && !(c.rhs.cref.name in known)
      push!(out, eq)
      push!(aliases, (c.name, c.rhs.cref.name))
    else
      push!(candidates, c)
    end
  end
  isempty(candidates) && return (equations, Symbol[])
  #= A clocked partition's definition stays for the clocked lowering: lifted, MSL Clocked's
     BooleanSignals.Sample1 sampled `sample(step.y, Clock())` at the step's change only. =#
  local clocked = clockedVariables(variables, equations)
  if !isempty(clocked)
    foreach(c -> c.name in clocked && push!(out, c.equation), candidates)
    filter!(c -> !(c.name in clocked), candidates)
    isempty(candidates) && return (equations, Symbol[])
  end
  #= A variable with more than one definition is not a discrete definition. =#
  local definitions = Dict{Symbol, Int}()
  foreach(c -> definitions[c.name] = get(definitions, c.name, 0) + 1, candidates)
  local byName = Dict{Symbol, Candidate}()
  local names = Symbol[]
  for c in candidates
    if definitions[c.name] > 1
      push!(out, c.equation)
    else
      byName[c.name] = c
      push!(names, c.name)
    end
  end
  isempty(names) && return (equations, Symbol[])
  local sets = aliasSets(aliases)
  local definitionOf = aliasDefinitions(byName, sets)
  readThroughAliases!(byName, definitionOf)
  local (initialLookup, initialPre) = initialPreValues(starts, names, byName, sets, definitionOf, constants)
  local nameSet = Set{Symbol}(names)
  local adjacent = Dict{Symbol, Vector{Symbol}}(n => Symbol[] for n in names)
  for n in names, r in namesReferenced(byName[n].rhs, nameSet)
    r === n && continue
    r in adjacent[n] || push!(adjacent[n], r)
    n in adjacent[r] || push!(adjacent[r], n)
  end
  local lifted = Symbol[]
  local representative = Dict{Symbol, Symbol}()
  for set in sets, n in set
    representative[n] = get(definitionOf, n, first(set))
  end
  local canon = n -> get(representative, n, n)
  local others = length(out)
  local index = Ref{Union{Nothing, Dict{Symbol, Vector{Set{Symbol}}}}}(nothing)
  local otherRefs = () -> (index[] === nothing && (index[] = refsOutsidePreIndex(view(out, 1:others), canon)); index[])
  for cluster in connectedComponents(names, adjacent)
    emitCluster!(out, cluster, byName, lifted, initialLookup, known, initialPre, otherRefs, canon)
  end
  return (out, lifted)
end
