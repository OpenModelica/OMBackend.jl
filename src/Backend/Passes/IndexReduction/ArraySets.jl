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
#= Dynamic state sets with arrays kept: the element view's sets of the elements of the same arrays (a pendulum per
   element) as one family, committed in OpenModelica's form per element with the element's subscripts as
   iterators (`x[i, j] = sum(A[i, j, c] * V_c[i])`, the switching when-equation in the for-equation). =#

"""
    SetLevel(variables, derivatives)

A level of a family of dynamic state sets: the arrays of the candidates' variables `V` and of the candidates
(their derivatives), in the candidates' order.
"""
struct SetLevel
  variables::Vector{Symbol}
  derivatives::Vector{Symbol}
end

"""
    SetFamily(levels, sizes, sets, jacobian)

The element view's dynamic state sets of the same arrays (`levels`: the first level's, then the linked ones),
one per element of arrays of dimensions `sizes`: `sets[k]` the set of the `k`-th element (row-major), `jacobian`
the pivots in array form (`setIterators` for the element's subscripts), equal to each set's at its element.
"""
struct SetFamily
  levels::Vector{SetLevel}
  sizes::Vector{Int}
  sets::Vector{DynamicSet}
  jacobian::Vector{IR.Expression}
end

"The iterators of a for-equation over the elements of arrays of `sizes`: `\$i1`, `\$i2`, ..."
setIterators(sizes::Vector{Int})::Vector{Symbol} = Symbol[Symbol("\$i", d) for d in eachindex(sizes)]

"The subscripts by the iterators over the elements of arrays of `sizes` (`setIterators`): `[\$i1, \$i2, ...]`."
iteratorSubscripts(sizes::Vector{Int})::Vector{IR.Subscript} =
  IR.Subscript[IR.INDEX(IR.CREF(IR.ComponentRef(it, IR.T_INTEGER()), IR.T_INTEGER())) for it in setIterators(sizes)]

"""
    nestedFor(body, sizes) -> FOR_EQUATION

The equations `body` for every element of arrays of `sizes`: nested for-equations over the dimensions, the first the
outermost, by the iterators `setIterators(sizes)`.
"""
function nestedFor(body::Vector{IR.Equation}, sizes::Vector{Int})::IR.FOR_EQUATION
  local iterators = setIterators(sizes)
  local eq = IR.FOR_EQUATION(IR.Iterator(iterators[end], integerRange(sizes[end])), body, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
  for d in reverse(eachindex(sizes))[2:end]
    eq = IR.FOR_EQUATION(IR.Iterator(iterators[d], integerRange(sizes[d])), IR.Equation[eq], IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
  end
  return eq
end

"The range `1:n`."
integerRange(n::Int)::IR.Expression = IR.RANGE(IR.INTEGER(1), nothing, IR.INTEGER(n), IR.T_ARRAY(IR.T_INTEGER(), IR.Dimension[IR.DIM_SIZE(n)]))

"""
    setLevels(set, s, E) -> Union{Nothing, Tuple{Vector{SetLevel}, Vector{Int}}}

The levels of a dynamic state set whose candidates are elements of arrays, all at the same subscripts, and those
subscripts; `nothing` for a set of scalars. NotHandled for one of elements at different subscripts or of
scalars and elements together.
"""
function setLevels(set::DynamicSet, s::Structure, E::Elements)::Union{Nothing, Tuple{Vector{SetLevel}, Vector{Int}}}
  local levels = SetLevel[]
  local index = nothing
  local scalars = 0
  local elementCount = 0
  for candidates in vcat([set.candidates], set.linked)
    local V = Symbol[]
    local D = Symbol[]
    for c in candidates
      for (name, out) in ((s.variables[s.antiderivative[c]], V), (s.variables[c], D))
        local element = get(E.array, name, nothing)
        if element === nothing
          scalars += 1
          continue
        end
        elementCount += 1
        index === nothing && (index = element[2])
        element[2] == index || throw(NotHandled("a dynamic state set of the elements of arrays at different subscripts"))
        push!(out, element[1])
      end
    end
    push!(levels, SetLevel(V, D))
  end
  elementCount == 0 && return nothing
  scalars == 0 || throw(NotHandled("a dynamic state set of scalars and elements of arrays"))
  return (levels, index)
end

"""
    familyPivots(sets, indices, sizes, E, variables) -> Vector{IR.Expression}

The pivots of a family's sets (`sets[k]` the set of the element of subscripts `indices[k]`, of arrays of `sizes`) in
array form, each candidate's: rolled from the first set's (`rolledPivot`) where that is every set's at its element
(structurally); else a table by element (`pivotTable`: the pivots differ by values the plan folds, a parameter
array's elements).
"""
function familyPivots(sets::Vector{DynamicSet}, indices::Vector{Vector{Int}}, sizes::Vector{Int}, E::Elements,
                      variables::IR.VariableSet)::Vector{IR.Expression}
  local iterators = setIterators(sizes)
  local subscripts = iteratorSubscripts(sizes)
  local values = Dict{Symbol, Int}()
  local out = IR.Expression[]
  for c in axes(sets[1].jacobian, 2)
    local rolled = rolledPivot(sets[1].jacobian[1, c], indices[1], subscripts, E, variables)
    local same = all(zip(sets, indices)) do (set, index)
      foreach(((it, k),) -> (values[it] = k), zip(iterators, index))
      foldTree(scalarExpression(rolled, values, E)) == foldTree(set.jacobian[1, c])
    end
    push!(out, same ? rolled : pivotTable(IR.Expression[set.jacobian[1, c] for set in sets], sizes, subscripts, E, variables))
  end
  return out
end

"""
    rolledPivot(pivot, index, subscripts, E, variables) -> Expression

A pivot of the set of the element `index` in array form: each reference to an element at `index` by its array
subscripted by the iterators (`subscripts`), another element's as a reference into its array.
"""
function rolledPivot(pivot::IR.Expression, index::Vector{Int}, subscripts::Vector{IR.Subscript}, E::Elements,
                     variables::IR.VariableSet)::IR.Expression
  return IR.mapExp(pivot) do x
    (x isa IR.CREF && haskey(E.array, x.cref.name)) || return x
    local (array, at) = E.array[x.cref.name]
    at == index || return IR.CREF(arrayElementRef(x.cref.name, E, variables), x.ty)
    return IR.CREF(distributedCref(variables[array], subscripts), x.ty)
  end
end

"""
    pivotTable(pivots, sizes, subscripts, E, variables) -> Expression

The pivots of a family's sets by element (row-major) as one expression of the iterators: the array literal of them
(references into the arrays) subscripted by the iterators (`subscripts`), each element's its own.
"""
function pivotTable(pivots::Vector{IR.Expression}, sizes::Vector{Int}, subscripts::Vector{IR.Subscript}, E::Elements,
                    variables::IR.VariableSet)::IR.Expression
  local table = nestedArray(sizes, IR.T_REAL(), IR.Expression[arrayReferences(p, E, variables) for p in pivots])
  return IR.SUBSCRIPTED(table, subscripts, IR.T_REAL())
end

"""
    setFamilies(p, E, variables) -> Tuple{Vector{SetFamily}, Vector{DynamicSet}}

The plan's dynamic state sets of elements of arrays as families (`SetFamily`), one set per element of the same
arrays, and the sets of scalars. NotHandled for sets of some elements of arrays only (slicing).
"""
function setFamilies(p::Plan, E::Elements, variables::IR.VariableSet)::Tuple{Vector{SetFamily}, Vector{DynamicSet}}
  local scalarSets = DynamicSet[]
  #= the sets by their levels' arrays (names: a level has no equality of its own) =#
  local byArrays = Dict{Vector{Vector{Symbol}}, Tuple{Vector{SetLevel}, Vector{Tuple{Vector{Int}, DynamicSet}}}}()
  for set in p.sets
    local found = setLevels(set, p.structure, E)
    found === nothing && (push!(scalarSets, set); continue)
    local (levels, index) = found
    local key = Vector{Symbol}[vcat(l.variables, l.derivatives) for l in levels]
    push!(last(get!(() -> (levels, Tuple{Vector{Int}, DynamicSet}[]), byArrays, key)), (index, set))
  end
  local families = SetFamily[]
  for (_, (levels, members)) in sort!(collect(byArrays); by = first)
    local sizes = E.sizes[levels[1].variables[1]]
    all(l -> all(a -> E.sizes[a] == sizes, vcat(l.variables, l.derivatives)), levels) ||
      throw(NotHandled("a dynamic state set of arrays of different dimensions"))
    #= one set per element, each in its element's place =#
    local slots = Vector{Union{Nothing, DynamicSet}}(nothing, prod(sizes))
    for (index, set) in members
      local k = rowMajorIndex(index, sizes)
      slots[k] === nothing || throw(NotHandled("two dynamic state sets of the element $(index) of $(levels[1].variables)"))
      slots[k] = set
    end
    any(isnothing, slots) &&
      throw(NotHandled("dynamic state sets of $(length(members)) of the $(prod(sizes)) elements of $(levels[1].variables) (slicing)"))
    local sets = DynamicSet[something(x) for x in slots]
    all(set -> size(set.jacobian, 1) == 1, sets) || throw(NotHandled("a dynamic state set of several rows"))
    push!(families, SetFamily(levels, sizes, sets, familyPivots(sets, rowMajor(sizes), sizes, E, variables)))
  end
  return (families, scalarSets)
end

"""
    distributedCref(v, subscripts) -> ComponentRef

The reference to the element of the array variable `v` by `subscripts` (literals or expressions), on the parts
whose types are arrays (`c[i].T` of the field `c.T` of a record array); of a derivative array on its last part
(`IR.withLastSubscripts`: `\$DER.c.T[i]`).
"""
function distributedCref(v::IR.Variable, subscripts::Vector{IR.Subscript})::IR.ComponentRef
  if IR.isDerivativeCref(v.cref)
    local n::Int = length(IR.dimensions(v.ty))
    n == length(subscripts) || throw(NotHandled("the array $(v.name), its reference of $(n) dimensions, subscripted by $(length(subscripts))"))
    return IR.withLastSubscripts(v.cref, subscripts)
  end
  local parts = IR.CrefPart[]
  local k = 0
  for p in v.cref.parts
    local n = length(IR.dimensions(p.ty))
    (n > 0 && !isempty(p.subscripts)) && throw(NotHandled("the array $(v.name), a part of its reference subscripted"))
    k + n <= length(subscripts) || throw(NotHandled("the array $(v.name), its reference of more dimensions than $(length(subscripts))"))
    push!(parts, IR.CrefPart(p.name, subscripts[k + 1:k + n], p.ty))
    k += n
  end
  k == length(subscripts) || throw(NotHandled("the array $(v.name), its reference of $(k) dimensions, subscripted by $(length(subscripts))"))
  return IR.ComponentRef(parts)
end

"""
    nestedArray(sizes, ty, leaves) -> Expression

The array literal of dimensions `sizes` of the elements `leaves` (row-major: the last subscript fastest), nested by
dimension, of element type `ty`.
"""
function nestedArray(sizes::Vector{Int}, ty::IR.ModelicaType, leaves::Vector{IR.Expression})::IR.Expression
  length(leaves) == prod(sizes) || error("nestedArray: $(length(leaves)) leaves for dimensions $(sizes)")
  return nestedArray(sizes, ty, leaves, 1, 0)
end

function nestedArray(sizes::Vector{Int}, ty::IR.ModelicaType, leaves::Vector{IR.Expression}, d::Int, offset::Int)::IR.Expression
  d > length(sizes) && return leaves[offset + 1]
  local stride = prod(@view(sizes[d + 1:end]); init = 1)
  return IR.ARRAY(IR.Expression[nestedArray(sizes, ty, leaves, d + 1, offset + (k - 1) * stride) for k in 1:sizes[d]],
                  IR.T_ARRAY(ty, IR.Dimension[IR.DIM_SIZE(n) for n in @view(sizes[d:end])]))
end

"""
    familyParts(f, k, s, E, variables, viewVariables) -> Tuple{Vector{IR.Variable}, Vector{IR.Equation}}

The family of dynamic state sets `f` as the `k`-th state set of the model, `dynamicSetParts` per element in array
form: the selection matrices `\$STATESETk.A[i.., j, c]` (Integer, the start values' choice of each element), the
states `\$STATESETk.x[i.., j]` of each level (`dx` the next) and their derivatives, and a for-equation over the
elements of `x = A * V`, `der(x) = A * der(V)` and the when-equation switching the dummy where its pivot grows past
twice the dummy's. The states' start values guess the variables the start values select (`viewVariables`: the
element view's).
"""
function familyParts(f::SetFamily, k::Int, s::Structure, E::Elements, variables::IR.VariableSet,
                     viewVariables::IR.VariableSet)::Tuple{Vector{IR.Variable}, Vector{IR.Equation}}
  local n = length(f.levels[1].variables)
  local sizes = f.sizes
  local at = iteratorSubscripts(sizes)
  local literal = (j::Int...) -> IR.Subscript[IR.INDEX(IR.INTEGER(x)) for x in j]
  #= the candidate the j-th state selects where candidate d is the dummy =#
  local selects = (d::Int, j::Int) -> j < d ? j : j + 1
  local starts = Int[findfirst(==(only(set.chosen)), set.candidates) for set in f.sets]
  local new = IR.Variable[]
  #= A: each element's (n - 1) x n selection by its start, one block per start shared =#
  local blocks = Dict{Int, Vector{IR.Expression}}()
  local block = (start::Int) -> get!(blocks, start) do
    IR.Expression[IR.INTEGER(c == selects(start, j) ? 1 : 0) for j in 1:n-1 for c in 1:n]
  end
  local aType = IR.T_ARRAY(IR.T_INTEGER(), IR.Dimension[IR.DIM_SIZE(m) for m in vcat(sizes, n - 1, n)])
  local leaves = IR.Expression[]
  foreach(start -> append!(leaves, block(start)), starts)
  local aStart = nestedArray(vcat(sizes, n - 1, n), IR.T_INTEGER(), leaves)
  local A::IR.Variable = IR.Variable(stateSetCref(k, :A, IR.Subscript[], aType), aType, IR.DISCRETE(), nothing,
                        IR.VariableAttributes(aStart, IR.BOOLEAN(true), nothing, nothing, nothing, nothing, nothing, nothing, nothing),
                        IR.NO_DIRECTION, true, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction)
  push!(new, A)
  local aRef::Function = (j::Int, c::Int) -> IR.CREF(IR.withLastSubscripts(A.cref, vcat(at, literal(j, c))), IR.T_INTEGER())
  local elementOf = (array::Symbol) -> IR.CREF(distributedCref(variables[array], at), IR.T_REAL())
  local selection = (j::Int, arrays::Vector{Symbol}) -> foldl(plus, IR.Expression[times(aRef(j, c), elementOf(arrays[c])) for c in 1:n])
  local body = IR.Equation[]
  local states = IR.Variable[]
  local stateType = IR.T_ARRAY(IR.T_REAL(), IR.Dimension[IR.DIM_SIZE(m) for m in vcat(sizes, n - 1)])
  for (l, level) in enumerate(f.levels)
    local cref::IR.ComponentRef = stateSetCref(k, Symbol("d"^(l - 1), "x"), IR.Subscript[], stateType)
    local name::Symbol = cref.name
    #= the start values' guess: the variable each element's state selects at the start =#
    local guesses = IR.Expression[]
    for (set, start) in zip(f.sets, starts), j in 1:n-1
      local candidates = l == 1 ? set.candidates : set.linked[l - 1]
      local v = get(viewVariables, s.variables[s.antiderivative[candidates[selects(start, j)]]], nothing)
      push!(guesses, (v === nothing || v.attributes.start === nothing) ? IR.REAL(0.0) : arrayReferences(v.attributes.start, E, variables))
    end
    local derivativeRef::IR.ComponentRef = IR.derivativeCref(cref)
    local x::IR.Variable = IR.Variable(cref, stateType, IR.STATE(1, derivativeRef.name, true), nothing,
                          IR.VariableAttributes(nestedArray(vcat(sizes, n - 1), IR.T_REAL(), guesses), nothing, nothing, nothing,
                                                nothing, nothing, nothing, nothing, nothing),
                          IR.NO_DIRECTION, true, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction)
    local derivative::IR.Variable = IR.Variable(derivativeRef, stateType, IR.STATE_DER(name), nothing,
                                   IR.NO_ATTRIBUTES, IR.NO_DIRECTION, true, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction)
    push!(new, x, derivative)
    push!(states, x)
    for j in 1:n-1
      local ref::Function = (v::IR.Variable) -> IR.CREF(IR.withLastSubscripts(v.cref, vcat(at, literal(j))), IR.T_REAL())
      push!(body, IR.SCALAR_EQUATION(ref(x), selection(j, level.variables), IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
      push!(body, IR.SCALAR_EQUATION(ref(derivative), selection(j, level.derivatives), IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    end
  end
  #= the dummy's pivot: each candidate's weighted by 1 - sum(A[i.., :, c]), 1 for the candidate no state selects =#
  local magnitude = IR.Expression[builtin(:abs, f.jacobian[c]) for c in 1:n]
  local unselected = c::Int -> foldl(minus, IR.Expression[aRef(j, c) for j in 1:n-1]; init = ONE)
  local pivot = foldl(plus, IR.Expression[times(unselected(c), magnitude[c]) for c in 1:n])
  local branches = IR.WhenBranch[]
  for d in 1:n
    local trigger = IR.RELATION(magnitude[d], IR.GREATER, times(IR.REAL(2.0), pivot), 0)
    local statements = IR.Statement[IR.ASSIGNMENT(aRef(j, c), IR.INTEGER(c == selects(d, j) ? 1 : 0), IR.NO_SOURCE)
                                    for j in 1:n-1 for c in 1:n]
    for (level, x) in zip(f.levels, states), j in 1:n-1
      local state::IR.ComponentRef = IR.withLastSubscripts(x.cref, vcat(at, literal(j)))
      push!(statements, IR.REINIT(state, elementOf(level.variables[selects(d, j)]), IR.NO_SOURCE))
    end
    push!(branches, IR.WhenBranch(IR.Expression[trigger], false, false, statements))
  end
  push!(body, IR.WHEN_EQUATION(branches, IR.REGULAR_WHEN, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
  return (new, IR.Equation[nestedFor(body, sizes)])
end
