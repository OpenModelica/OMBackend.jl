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
#= Aliases read through one representative before index reduction, with arrays kept (`substituteAliases!` of the
   scalarized route in array form): whole arrays whose elements are aliases of another array's, index by index. =#

"""
    ArrayPair(negated, conflicting, indices, equations)

The alias equations of an element view between the elements of two arrays at the same index (`a[i] = b[i]`;
`negated`: `a[i] = -b[i]`): the element indices they cover, the view's equations (by position); `conflicting` where
two of them differ in sign.
"""
mutable struct ArrayPair
  negated::Bool
  conflicting::Bool
  const indices::Set{Vector{Int}}
  const equations::Vector{Int}
end

"The model's equation an element equation of a view comes from: its position and its path in for-equations' bodies."
const EquationLeaf = Tuple{Int, Vector{Int}}

"""
    arrayAliasPairs(view, blocked) -> Dict{Tuple{Symbol, Symbol}, ArrayPair}

The pairs of arrays (a scalar an array of no dimensions; by name, the lesser first) whose elements the element view
`view` makes aliases index by index with one sign, every element covered (`aliasEquation` of the continuous
variables), none of them `blocked`; only those whose alias equations make up whole equations of the model
(`EquationLeaf`: an equation, a body of a for-equation), so that these stay as written while the others read the
representatives.
"""
function arrayAliasPairs(view::Scalarized, blocked::Set{Symbol})::Dict{Tuple{Symbol, Symbol}, ArrayPair}
  local E::Elements = view.elements
  local variables::IR.VariableSet = view.model.variables
  local partition::IR.Partition = view.model.continuous[1]
  local discretes::Set{Symbol} = discreteTargets(partition)
  local isCandidate = (name::Symbol) -> begin
    local v::Union{Nothing, IR.Variable} = get(variables, name, nothing)
    v !== nothing && v.ty isa IR.T_REAL && v.kind isa Union{IR.STATE, IR.ALGEBRAIC, IR.ALGEBRAIC_DER} &&
      !(name in discretes) && v.direction !== IR.INPUT
  end
  local noConstant = (::IR.Expression) -> false
  local pairs::Dict{Tuple{Symbol, Symbol}, ArrayPair} = Dict{Tuple{Symbol, Symbol}, ArrayPair}()
  for (j::Int, eq::IR.Equation) in enumerate(partition.equations)
    local a::Union{Nothing, Tuple} = aliasEquation(eq, isCandidate, noConstant)
    (a === nothing || length(a) != 3) && continue
    local (A::Symbol, ia::Vector{Int}) = get(E.array, a[1], (a[1], Int[]))
    local (B::Symbol, ib::Vector{Int}) = get(E.array, a[2], (a[2], Int[]))
    (ia == ib && A != B && !(A in blocked) && !(B in blocked)) || continue
    local pair::ArrayPair = get!(() -> ArrayPair(a[3], false, Set{Vector{Int}}(), Int[]), pairs, A < B ? (A, B) : (B, A))
    pair.negated == a[3] || (pair.conflicting = true)
    push!(pair.indices, ia)
    push!(pair.equations, j)
  end
  local sizes = (A::Symbol) -> get(E.sizes, A, Int[])
  filter!(((k, p),) -> !p.conflicting && sizes(k[1]) == sizes(k[2]) && length(p.indices) == prod(sizes(k[1])), pairs)
  #= the pairs whose alias equations are whole equations of the model: a pair that shares one with another equation
     (an array equation of more than the pair's elements, a for-equation's body of another equation too) is left,
     until all are whole =#
  local leaf = (j::Int) -> (view.origins[j].equation, view.origins[j].path)
  local leafSize::Dict{EquationLeaf, Int} = Dict{EquationLeaf, Int}()
  foreach(o -> (leafSize[(o.equation, o.path)] = get(leafSize, (o.equation, o.path), 0) + 1), view.origins)
  while true
    local inPairs::Dict{EquationLeaf, Int} = Dict{EquationLeaf, Int}()
    for p in values(pairs), j in p.equations
      inPairs[leaf(j)] = get(inPairs, leaf(j), 0) + 1
    end
    local before::Int = length(pairs)
    filter!(((_, p),) -> all(j -> inPairs[leaf(j)] == leafSize[leaf(j)], p.equations), pairs)
    length(pairs) == before && return pairs
  end
end

"""
    aliasReference(cref, variables) -> Tuple{Symbol, Vector{Subscript}}

The variable a reference reads and the subscripts it reads it by: a variable of the reference's own name (a scalar,
an element `R.T[1,1]` of a component not rolled) whole; else its array by the reference's subscripts (`allSubscripts`;
an element, a slice), or a name of no variable (a record) without.
"""
function aliasReference(cref::IR.ComponentRef, variables::IR.VariableSet)::Tuple{Symbol, Vector{IR.Subscript}}
  (haskey(variables, cref.name) || !IR.hasSubscripts(cref)) && return (cref.name, IR.Subscript[])
  return (unsubscriptedName(cref), allSubscripts(cref))
end

"""
    leadingSubscripts(cref) -> Bool

Whether the subscripts of `cref` are those of its array's leading dimensions, part by part: each part of dimensions
subscripted in all of them up to the first not subscripted, none after (a rolled instance's `G[\$c].body.w_a` of the
array `G.body.w_a`).
"""
function leadingSubscripts(cref::IR.ComponentRef)::Bool
  local open::Bool = true
  for p in cref.parts
    local n::Int = length(IR.dimensions(IR.unsubscriptedPart(p).ty))
    local s::Int = length(IR.partSubscripts(p))
    n == 0 && continue
    if s == 0
      open = false
    elseif !open || s != n
      return false
    end
  end
  return true
end

"""
    leadingCref(v, subscripts) -> Union{Nothing, ComponentRef}

The reference to the slice of the array variable `v` by the subscripts of its leading dimensions (`leadingSubscripts`),
part by part as `distributedCref` places all; `nothing` where they end inside a part's dimensions or `v` is a
derivative array.
"""
function leadingCref(v::IR.Variable, subscripts::Vector{IR.Subscript})::Union{Nothing, IR.ComponentRef}
  IR.isDerivativeCref(v.cref) && return nothing
  local parts::Vector{IR.CrefPart} = IR.CrefPart[]
  local k::Int = 0
  for p in v.cref.parts
    local n::Int = length(IR.dimensions(p.ty))
    if n == 0 || k == length(subscripts)
      push!(parts, p)
    elseif k + n <= length(subscripts)
      push!(parts, IR.CrefPart(p.name, subscripts[k + 1:k + n], p.ty))
      k += n
    else
      return nothing
    end
  end
  return k == length(subscripts) ? IR.ComponentRef(parts) : nothing
end

"""
    arrayAliasBlocked(model, E) -> Union{Nothing, Set{Symbol}}

The arrays of an arrays-kept model that `substituteArrayAliases!` leaves as written for how the partition reads them:
by a slice other than of its leading dimensions (`leadingSubscripts`), or as part of a whole record or array around
them (a record argument `G[\$c].frame_a.R` of rolled components: a name of no variable, the variables under it); a
derivative array's slice blocks its base array. `nothing` where a reference has a slice part (`T_SLICE`, a split's):
none substituted.
"""
function arrayAliasBlocked(model::IR.Model, E::Elements)::Union{Nothing, Set{Symbol}}
  local variables::IR.VariableSet = model.variables
  local partition::IR.Partition = model.continuous[1]
  local antiderivative::Dict{Symbol, Symbol} = antiderivatives(variables)
  local blocked::Set{Symbol} = Set{Symbol}()
  local resolve::Resolver = Resolver(Symbol[v.name for v in variables])
  local sliced::Bool = false
  local scan = (@nospecialize(e::IR.Expression)) -> begin
    IR.foreachExp(e) do @nospecialize(x::IR.Expression)
      x isa IR.CREF || return nothing
      any(p -> p.ty isa IR.T_SLICE, x.cref.parts) && (sliced = true; return nothing)
      local (name::Symbol, subscripts::Vector{IR.Subscript}) = aliasReference(x.cref, variables)
      if haskey(variables, name)
        local n::Int = length(subscripts)
        local whole::Bool = n == 0 || n == length(get(E.sizes, name, Int[]))
        (whole || (leadingSubscripts(x.cref) && !haskey(antiderivative, name))) ||
          push!(blocked, baseVariable(antiderivative, name))
      else
        union!(blocked, resolve(name))
      end
      nothing
    end
  end
  for eq in partition.equations
    IR.foreachRootExp(scan, eq)
  end
  for eq in partition.initialEquations
    IR.foreachRootExp(scan, eq)
  end
  return sliced ? nothing : blocked
end

"""
    rewriteAliasLeaves(eq, at, kept, substitute, reads) -> Equation

The model's equation `eq` (at `at`: its position, its path in for-equations' bodies) with `substitute` mapped over a
copy of it where it `reads` a substituted array, except for the bodies `kept` (the alias equations) and for when- and
assert-equations; `eq` itself where it changes nothing.
"""
function rewriteAliasLeaves(eq::IR.Equation, at::EquationLeaf, kept::Set{EquationLeaf}, substitute::F, reads::G)::IR.Equation where {F, G}
  (at in kept || eq isa Union{IR.WHEN_EQUATION, IR.ASSERT_EQUATION} || !reads(eq)) && return eq
  eq isa IR.FOR_EQUATION || return IR.mapEquation!(substitute, deepcopy(eq))
  local body::Vector{IR.Equation} = IR.Equation[rewriteAliasLeaves(b, (at[1], vcat(at[2], k)), kept, substitute, reads)
                                               for (k, b) in enumerate(eq.body)]
  return IR.FOR_EQUATION(eq.iterator, body, eq.source, eq.attributes)
end

"""
    substituteArrayAliases!(model, view) -> AliasUndo

`substituteAliases!` of an arrays-kept model in array form, before its index reduction (`reduceIndexArrays!`): the
arrays whose elements are aliases of another array's, index by index with one sign (`arrayAliasPairs` of its element
view `view`: a connection's `frame_a.R.T = frame_b.R.T` of rolled components), read through one representative array
in the partition's equations and initial equations, so that index reduction differentiates neither the alias
equations nor their derivatives. The representative of a set as in `substituteAliases!` (a state with a fixed start,
a state, StateSelect, a start, the declaration order), the substituted arrays' start values merged onto it
(`mergeAliasStarts!`); an array is left where an element of it is (`aliasBlocked`:
written by a when or an algorithm, read through a derivative by a when, read whole; an input), where it is read by a
slice or within a whole record (`arrayAliasBlocked`), or where its derivatives go further than the representative's;
a set with two fixed starts too. The alias equations stay as written (each array is still computed); the
states substituted become algebraic, their derivative arrays removed. Returns what to restore (`undoAliases!`).
"""
function substituteArrayAliases!(model::IR.Model, view::Scalarized)::AliasUndo
  local variables::IR.VariableSet = model.variables
  local partition::IR.Partition = model.continuous[1]
  local E::Elements = view.elements
  local none::AliasUndo = AliasUndo(partition.equations, partition.initialEquations, Pair{Symbol, IR.VariableKind}[], IR.Variable[], 0,
                                    Pair{Symbol, IR.VariableAttributes}[])
  local readBlocked::Union{Nothing, Set{Symbol}} = arrayAliasBlocked(model, E)
  readBlocked === nothing && return none
  local blocked::Set{Symbol} = Set{Symbol}(first(get(E.array, n, (n, Int[])))
                                           for n in aliasBlocked(view.model.variables, view.model.continuous[1]))
  union!(blocked, readBlocked)
  local pairs::Dict{Tuple{Symbol, Symbol}, ArrayPair} = arrayAliasPairs(view, blocked)
  isempty(pairs) && return none
  #= the sets of arrays, with signs =#
  local sets::AliasSets = AliasSets()
  local kept::Set{EquationLeaf} = Set{EquationLeaf}()
  for ((a, b), p) in pairs
    joinAliases!(sets, a, b, p.negated, view.model.continuous[1].equations[first(p.equations)])
    foreach(j -> push!(kept, (view.origins[j].equation, view.origins[j].path)), p.equations)
  end
  local members::Dict{Symbol, Vector{AliasSign}} = Dict{Symbol, Vector{AliasSign}}()
  for x in collect(keys(sets.parent))
    local (root::Symbol, negated::Bool) = aliasRoot!(sets, x)
    push!(get!(() -> AliasSign[], members, root), AliasSign(x, negated))
  end
  #= an array state of any fixed start (all elements, or an expression) counts as fixed =#
  local fixedState = (name::Symbol) -> begin
    local v::IR.Variable = variables[name]
    local fixed = v.attributes.fixed
    v.kind isa IR.STATE && fixed !== nothing && !(fixed isa IR.BOOLEAN && !fixed.value)
  end
  local position::Dict{Symbol, Int} = Dict{Symbol, Int}(v.name => i for (i, v) in enumerate(variables))
  local priority = (name::Symbol) -> begin
    local v::IR.Variable = variables[name]
    (fixedState(name) ? 1000 : 0) + (v.kind isa IR.STATE ? 100 : 0) +
      10 * Int(something(v.attributes.stateSelect, IR.DEFAULT)) + (v.attributes.start === nothing ? 0 : 1)
  end
  local replacements::Dict{Symbol, AliasSign} = Dict{Symbol, AliasSign}()
  local substituted::Vector{Symbol} = Symbol[]
  local states::Vector{Symbol} = Symbol[]
  local changedAttributes::Vector{Pair{Symbol, IR.VariableAttributes}} = Pair{Symbol, IR.VariableAttributes}[]
  for (root, set) in members
    #= a set of two fixed starts (of states or not) is left as written =#
    (root in sets.conflicting || count(m -> fixedStart(variables[m.name]), set) > 1) && continue
    local best::AliasSign = argmax(m -> (priority(m.name), -position[m.name]), set)
    local bestChain::Vector{Symbol} = derivativeChain(variables, best.name)
    local merged::Vector{AliasSign} = AliasSign[]
    for m in set
      m.name == best.name && continue
      local chain::Vector{Symbol} = derivativeChain(variables, m.name)
      length(chain) <= length(bestChain) || continue
      #= m = ±root, best = ±root: m = (m.negated xor best.negated) best =#
      local n::Bool = xor(m.negated, best.negated)
      replacements[m.name] = AliasSign(best.name, n)
      foreach(((k, d),) -> replacements[d] = AliasSign(bestChain[k], n), enumerate(chain))
      push!(substituted, m.name)
      push!(merged, AliasSign(m.name, n))
      isempty(chain) || push!(states, m.name)
    end
    isempty(merged) ||
      mergeAliasStarts!(variables, best.name, sort!(merged; by = m::AliasSign -> (-priority(m.name), position[m.name])),
                        changedAttributes)
  end
  isempty(substituted) && return none
  local undo::AliasUndo = AliasUndo(partition.equations, partition.initialEquations,
                                    Pair{Symbol, IR.VariableKind}[n => variables[n].kind for n in states],
                                    IR.Variable[variables[d] for n in states for d in derivativeChain(variables, n)],
                                    sum(n -> prod(get(E.sizes, n, Int[])), substituted), changedAttributes)
  #= a reference to a member: the representative's by the same subscripts (an element, a slice of the leading
     dimensions, the whole array) =#
  local substitute = (@nospecialize(x::IR.Expression)) -> begin
    x isa IR.CREF || return x
    local (name::Symbol, subscripts::Vector{IR.Subscript}) = aliasReference(x.cref, variables)
    local r::Union{Nothing, AliasSign} = get(replacements, name, nothing)
    r === nothing && return x
    local target::IR.Variable = variables[r.name]
    local cref::IR.ComponentRef =
      isempty(subscripts) ? target.cref :
      length(subscripts) == length(get(E.sizes, name, Int[])) ? distributedCref(target, subscripts) :
      something(leadingCref(target, subscripts), x.cref)
    cref === x.cref && throw(NotHandled("the slice $(x.cref.name) of $(name) read as one of $(r.name)"))
    local e::IR.CREF = IR.CREF(cref, x.ty)
    return r.negated ? IR.UNARY(IR.UMINUS, e, x.ty) : e
  end
  local reads = (eq::IR.Equation) -> begin
    local found::Bool = false
    IR.foreachRootExp(eq) do @nospecialize(e::IR.Expression)
      found || (found = IR.anyExp((@nospecialize(x::IR.Expression)) ->
                                    x isa IR.CREF && haskey(replacements, first(aliasReference(x.cref, variables))), e))
      nothing
    end
    found
  end
  partition.equations = IR.Equation[rewriteAliasLeaves(eq, (i, Int[]), kept, substitute, reads)
                                    for (i, eq) in enumerate(partition.equations)]
  local noLeaves::Set{EquationLeaf} = Set{EquationLeaf}()
  partition.initialEquations = IR.Equation[rewriteAliasLeaves(eq, (0, Int[]), noLeaves, substitute, reads)
                                           for eq in partition.initialEquations]
  local removed::Set{Symbol} = Set{Symbol}(v.name for v in undo.removed)
  foreach(n -> (variables[n].kind = IR.ALGEBRAIC()), states)
  filter!(v -> !(v.name in removed), variables)
  return undo
end

"""
    arrayAliasPlan(model, view, cache) -> Tuple{Scalarized, Union{Nothing, Plan}}

`aliasPlan` of an arrays-kept model that needs index reduction (`differentiates` of its element view `view`): planned
with the aliases read through their representatives (`substituteArrayAliases!`), which the model keeps, its new
element view and plan. Where that leaves nothing to reduce (the aliases were the constraint: a capacitor loop's
`v1 = v2`) or is not handled, the model as written, `view` and `nothing` (the caller plans it as written).
"""
function arrayAliasPlan(model::IR.Model, view::Scalarized, cache::ScalarCache)::Tuple{Scalarized, Union{Nothing, Plan}}
  local order::Vector{Symbol} = Symbol[v.name for v in model.variables]
  #= a reference it does not rewrite (a slice of the representative that ends inside a part's dimensions) throws
     before the model is changed =#
  local aliases::AliasUndo = try
    substituteArrayAliases!(model, view)
  catch err
    err isa NotHandled || rethrow()
    return (view, nothing)
  end
  aliases.substituted > 0 || return (view, nothing)
  local substitutedView::Union{Nothing, Scalarized} = nothing
  local p::Union{Nothing, Plan} = try
    substitutedView = scalarizeArrays(model; cache)
    plan(substitutedView.model, substitutedView.model.continuous[1]; indexOne = true, initialDerivatives = true)
  catch err
    err isa NotHandled || rethrow()
    nothing
  end
  (p !== nothing && !isempty(p.differentiated)) && return (something(substitutedView), p)
  undoArrayAliases!(model, aliases, order)
  return (view, nothing)
end

"""
    undoArrayAliases!(model, undo, order) -> Nothing

`undoAliases!` of `substituteArrayAliases!`, the variables in their order before (`order`): the plans and the
commit read the variables in the model's order.
"""
function undoArrayAliases!(model::IR.Model, undo::AliasUndo, order::Vector{Symbol})::Nothing
  undoAliases!(model, model.continuous[1], undo)
  local byName::Dict{Symbol, IR.Variable} = Dict{Symbol, IR.Variable}(v.name => v for v in model.variables)
  empty!(model.variables)
  foreach(n -> push!(model.variables, byName[n]), order)
  return nothing
end
