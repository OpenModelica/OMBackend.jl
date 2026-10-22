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
#= The plan made on the element view of an arrays-kept model (Arrays.jl), committed in array form: the arrays
   whose elements the plan treats alike keep one kind, a for-equation whose iterations it differentiates alike is
   differentiated once, symbolically, its iterator kept (Otter and Elmqvist, Modelica Conference 2017, Theorem 2:
   under full incidence every element of an array equation is differentiated equally). Where the plan treats the
   elements of one array differently (a boundary element, a mixed choice of states), NotHandled: slicing the array
   is the next step. =#

"""
    ArrayCommit

What the array form of a plan adds to the model before it is changed: the new variables (derivative arrays, the
states' derivative arrays `\$STATEDER.x` and scalar ones, the dynamic state sets'), the new kind of each variable
whose kind changes, the equations to append (the differentiated ones, the states' links, the sets'), and the number
of dynamic state sets (a family over the elements of arrays counted once).
"""
struct ArrayCommit
  variables::Vector{IR.Variable}
  kinds::Dict{Symbol, IR.VariableKind}
  equations::Vector{IR.Equation}
  sets::Int
end

"""
    plannedKinds(s, dummies) -> Dict{Symbol, VariableKind}

The kind of each structure variable of the element view whose kind the plan changes (`plannedKind`), the selected
states linked to their derivatives as `linkStateDerivative!` links them: a state's derivative `\$STATEDER.x`
(STATE_DER), the derivative variable it read algebraic unless it is a state itself.
"""
function plannedKinds(s::Structure, dummies::Set{Int})::Dict{Symbol, IR.VariableKind}
  local kinds = Dict{Symbol, IR.VariableKind}()
  for v in eachindex(s.variables)
    local kind = plannedKind(s, dummies, v)
    kind === nothing || (kinds[s.variables[v]] = kind)
  end
  for v in selectedStates(s, dummies)
    local (state, derivative) = (s.variables[v], s.variables[s.derivative[v]])
    local link = stateDerivativeName(state)
    kinds[state] = IR.STATE(1, link, true)
    kinds[link] = IR.STATE_DER(state)
    get(kinds, derivative, nothing) isa IR.STATE || (kinds[derivative] = IR.ALGEBRAIC())
  end
  return kinds
end

"""
    arrayElementRef(name, E, variables) -> ComponentRef

The element variable `name` of the element view as a reference into its array in the arrays-kept model
(`x[2]`, `c[2].T`; `\$DER.x[2]` as the element 2 of the array `\$DER.x`); a scalar variable by its own reference.
"""
function arrayElementRef(name::Symbol, E::Elements, variables::IR.VariableSet)::IR.ComponentRef
  local element = get(E.array, name, nothing)
  if element === nothing
    local v = get(variables, name, nothing)
    return v === nothing ? IR.ComponentRef(name, IR.T_REAL()) : v.cref
  end
  local cref = E.crefs[name]
  IR.hasSubscripts(cref) && return cref
  #= a backend-made array, its elements named by flat names: the element of its one part =#
  local (array, index) = element
  local part = variables[array].cref.parts[1]
  return IR.ComponentRef(IR.CrefPart[IR.CrefPart(part.name, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index], part.ty)])
end

"A reference to an element variable of the element view as a reference into its array (`arrayElementRef`); another node as it is."
arrayReference(x::IR.Expression, E::Elements, variables::IR.VariableSet)::IR.Expression =
  (x isa IR.CREF && haskey(E.array, x.cref.name)) ? IR.CREF(arrayElementRef(x.cref.name, E, variables), x.ty) : x

"`e` (of the element view) with its element variables as references into their arrays (`arrayReference`)."
arrayReferences(e::IR.Expression, E::Elements, variables::IR.VariableSet)::IR.Expression =
  IR.mapExp(x -> arrayReference(x, E, variables), e)

"""
    arrayEquation(eq, E, variables) -> Equation

The equation `eq` of the element view as an equation of the arrays-kept model (`arrayReferences`), a new equation:
a scalar equation's sides mapped, another kind copied and mapped.
"""
arrayEquation(eq::IR.SCALAR_EQUATION, E::Elements, variables::IR.VariableSet)::IR.Equation =
  IR.SCALAR_EQUATION(arrayReferences(eq.lhs, E, variables), arrayReferences(eq.rhs, E, variables), eq.ty, eq.source, eq.attributes)
arrayEquation(eq::IR.Equation, E::Elements, variables::IR.VariableSet)::IR.Equation =
  IR.mapEquation!(x -> arrayReference(x, E, variables), deepcopy(eq))

"""
    canonicalReferences(e, E, variables) -> Expression

`e` (of the arrays-kept model) with each reference to an element by literal subscripts as `arrayElementRef`
names it (`c.p.T[2]`, as the passes subscript a field of a record array, is `c[2].p.T`): the names the element
view's structure knows.
"""
function canonicalReferences(e::IR.Expression, E::Elements, variables::IR.VariableSet)::IR.Expression
  return IR.mapExp(e) do x
    (x isa IR.CREF && IR.hasSubscripts(x.cref)) || return x
    local element = elementReference(x, E)
    element === x && return x
    return arrayReferences(element, E, variables)
  end
end

"""
    uniformKind(array, kinds, E) -> VariableKind

The kind of the array `array` whose elements have the kinds `kinds` (in its elements' order): one kind of the same
type whose variables are the elements of one array at the same subscripts (`\$DER.x[k]` for each `x[k]`:
`\$DER.x`). NotHandled where the elements differ.
"""
function uniformKind(array::Symbol, kinds::Vector{IR.VariableKind}, E::Elements)::IR.VariableKind
  local K = typeof(kinds[1])
  all(k -> k isa K, kinds) || throw(NotHandled("an array whose elements the plan gives different kinds ($(array): $(unique(typeof.(kinds))))"))
  K === IR.ALGEBRAIC && return IR.ALGEBRAIC()
  local named = Symbol[kindVariable(k) for k in kinds]
  local of = get(E.array, named[1], nothing)
  (of !== nothing && E.elements[of[1]] == named) ||
    throw(NotHandled("an array whose elements' $(nameof(K)) variables are not one array's elements in order ($(array))"))
  if K === IR.STATE
    all(k -> k.index == kinds[1].index && k.natural == kinds[1].natural, kinds) ||
      throw(NotHandled("an array of states of different indices ($(array))"))
    return IR.STATE(kinds[1].index, of[1], kinds[1].natural)
  end
  return K(of[1])
end

"The variable a kind names: a state's derivative, a derivative's state, a dummy's counterpart."
kindVariable(k::IR.STATE)::Symbol = k.derivative
kindVariable(k::IR.STATE_DER)::Symbol = k.state
kindVariable(k::IR.DUMMY_DER)::Symbol = k.dummyState
kindVariable(k::IR.DUMMY_STATE)::Symbol = k.dummyDerivative

"""
    newArrays(names, s, E, model) -> Vector{IR.Variable}

The variables the plan adds (`names`: the derivatives Pantelides' algorithm made) in array form: the derivatives
of all the elements of an array as one derivative array (`\$DER.\$DER.x`, of the array's dimensions; lower orders
first, so that the derivative of a new array is an array too), a scalar's derivative as a scalar. NotHandled for
the derivatives of some elements of an array only.
"""
function newArrays(names::Vector{Symbol}, s::Structure, E::Elements, model::IR.Model)::Vector{IR.Variable}
  local out = IR.Variable[]
  local arrayOf = copy(E.array)
  local elementsOf = copy(E.elements)
  local dimensionsOf = Dict{Symbol, Vector{IR.Dimension}}(v.name => v.ty.dimensions for v in model.variables if v.ty isa IR.T_ARRAY)
  local byOrder = Dict{Int, Vector{Symbol}}()
  foreach(n -> push!(get!(() -> Symbol[], byOrder, derivativeOrder(s, s.index[n])), n), names)
  for order in sort!(collect(keys(byOrder)))
    local byArray = Dict{Symbol, Vector{Symbol}}()
    for name in byOrder[order]
      local element = get(arrayOf, s.variables[s.antiderivative[s.index[name]]], nothing)
      if element === nothing
        push!(out, IR.Variable(name, IR.T_REAL(), IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                               false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
      else
        push!(get!(() -> Symbol[], byArray, element[1]), name)
      end
    end
    for (array, derived) in sort!(collect(byArray); by = first)
      local expected = Symbol[IR.derivativeName(e) for e in elementsOf[array]]
      Set(derived) == Set(expected) ||
        throw(NotHandled("the derivatives of $(length(derived)) of the $(length(expected)) elements of the array $(array) (slicing)"))
      local name = IR.derivativeName(array)
      haskey(model.variables, name) && throw(NotHandled("a derivative array $(name) that exists"))
      push!(out, IR.Variable(name, IR.T_ARRAY(IR.T_REAL(), dimensionsOf[array]), IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES,
                             IR.NO_DIRECTION, false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
      dimensionsOf[name] = dimensionsOf[array]
      elementsOf[name] = expected
      for (e, d) in zip(elementsOf[array], expected)
        arrayOf[d] = (name, arrayOf[e][2])
      end
    end
  end
  return out
end

"""
    arrayDerivatives(s, E) -> Dict{Symbol, Symbol}

The arrays whose elements' derivatives (in the structure) are the elements of one array, in order: array =>
derivative array (`x` => `\$DER.x`), how a for-equation's `x[i]` is differentiated.
"""
function arrayDerivatives(s::Structure, E::Elements)::Dict{Symbol, Symbol}
  local out = Dict{Symbol, Symbol}()
  for (array, names) in E.elements
    all(n -> haskey(s.index, n) && s.derivative[s.index[n]] != 0, names) || continue
    local derivatives = Symbol[s.variables[s.derivative[s.index[n]]] for n in names]
    local of = get(E.array, derivatives[1], nothing)
    (of !== nothing && E.elements[of[1]] == derivatives) && (out[array] = of[1])
  end
  return out
end

"""
    arraySeeds(s, derivatives, E, variables, functions) -> Seeds

The time derivative in array form: of an array variable the derivative array (`derivatives`; `x[i]` is then
`\$DER.x[i]`, `arrayElementDerivative`), of an element or a scalar the structure's derivative as a reference into its
array; NotHandled for an array of continuous variables whose elements' derivatives are not one array.
"""
function arraySeeds(s::Structure, derivatives::Dict{Symbol, Symbol}, E::Elements, variables::IR.VariableSet,
                    functions::Vector{IR.ModelicaFunction})::Seeds
  local variable = (name::Symbol) -> begin
    local members = get(E.elements, name, nothing)
    if members !== nothing
      local derivative = get(derivatives, name, nothing)
      derivative === nothing || return IR.CREF(variables[derivative].cref, variables[derivative].ty)
      any(n -> haskey(s.index, n), members) &&
        throw(NotHandled("the derivative of the array $(name), its elements' derivatives not one array"))
      return nothing
    end
    local v = get(s.index, name, 0)
    v == 0 && return nothing
    local d = s.derivative[v]
    d == 0 && throw(NotHandled("the derivative of $(name), which Pantelides' algorithm gave none"))
    return IR.CREF(arrayElementRef(s.variables[d], E, variables), IR.T_REAL())
  end
  return Seeds(ArrayVariables(variable), 1.0, Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in functions))
end

"The equation at `path` in the bodies of the nested for-equation `eq`, and the iterators on the way (outermost first)."
function bodyAt(eq::IR.FOR_EQUATION, path::Vector{Int})::Tuple{IR.Equation, Vector{Symbol}}
  local iterators = Symbol[eq.iterator.name]
  local current::IR.Equation = eq.body[path[1]]
  for k in path[2:end]
    current isa IR.FOR_EQUATION || throw(NotHandled("an equation path through a $(nameof(typeof(current)))"))
    push!(iterators, current.iterator.name)
    current = current.body[k]
  end
  return (current, iterators)
end

"""
    differentiatedForEquation(eq, chains, level) -> Union{Nothing, FOR_EQUATION}

The `level`-th derivative of the for-equation `eq` in array form: its nesting kept, each body equation by its
derivative of that order (`chains[path]`, the derivatives of the body equation at `path`), the bodies with none
left out; `nothing` where no body equation is differentiated that often.
"""
function differentiatedForEquation(eq::IR.FOR_EQUATION, chains::Dict{Vector{Int}, Vector{IR.Equation}}, level::Int,
                                   prefix::Vector{Int} = Int[])::Union{Nothing, IR.FOR_EQUATION}
  local body = IR.Equation[]
  for (k, e) in enumerate(eq.body)
    local path = vcat(prefix, k)
    if e isa IR.FOR_EQUATION
      local inner = differentiatedForEquation(e, chains, level, path)
      inner === nothing || push!(body, inner)
    else
      local chain = get(chains, path, nothing)
      (chain !== nothing && length(chain) >= level) && push!(body, chain[level])
    end
  end
  isempty(body) && return nothing
  return IR.FOR_EQUATION(eq.iterator, body, eq.source, IR.EquationAttributes(eq.attributes.kind, true, nothing))
end

"The scalar equation `eq` of the arrays-kept model at the iterator `values`, in the element view's form (`equationExpression`)."
elementEquation(eq::IR.SCALAR_EQUATION, values::Dict{Symbol, Int}, E::Elements)::IR.SCALAR_EQUATION =
  IR.SCALAR_EQUATION(equationExpression(eq.lhs, values, E), equationExpression(eq.rhs, values, E), eq.ty, eq.source, eq.attributes)

"Whether two scalar equations are one, their sides' arithmetic folded (`foldTree`), structurally."
sameEquation(a::IR.SCALAR_EQUATION, b::IR.SCALAR_EQUATION)::Bool =
  foldTree(a.lhs) == foldTree(b.lhs) && foldTree(a.rhs) == foldTree(b.rhs)
sameEquation(::IR.Equation, ::IR.Equation)::Bool = false

"An equation as one line of text, for a message."
equationLine(eq::IR.Equation)::String = replace(string(eq), r"\s*\n\s*" => " ")

"""
    forDerivatives(eq, rows, view, p, seeds, planSeeds, E, variables, functions) -> Dict{Vector{Int}, Vector{IR.Equation}}

The derivatives in array form of the body equations of the for-equation `eq` that the plan differentiates
(`rows`: the element view's equations of `eq`): each body equation (by its path) differentiated symbolically as
often as the plan differentiates each of its iterations, its iterators kept; checked against the derivative of
every iteration (its element view's equation differentiated by the plan's structure, `planSeeds`, as written: the
parameters symbolic, which the plan's own equations read as their values). NotHandled where the iterations are
differentiated unequally (slicing) or the array form's derivative of one is not its own.
"""
function forDerivatives(eq::IR.FOR_EQUATION, rows::Vector{Int}, view::Scalarized, p::Plan, seeds::Seeds, planSeeds::Seeds,
                        E::Elements, variables::IR.VariableSet,
                        functions::Dict{Symbol, IR.ModelicaFunction})::Dict{Vector{Int}, Vector{IR.Equation}}
  local byPath = Dict{Vector{Int}, Vector{Int}}()
  foreach(r -> push!(get!(() -> Int[], byPath, view.origins[r].path), r), rows)
  local viewEquations = view.model.continuous[1].equations
  local chains = Dict{Vector{Int}, Vector{IR.Equation}}()
  for (path, pathRows) in byPath
    local count = length(get(p.differentiated, pathRows[1], IR.Equation[]))
    all(r -> length(get(p.differentiated, r, IR.Equation[])) == count, pathRows) ||
      throw(NotHandled("a for-equation whose iterations are differentiated unequally (slicing)"))
    count == 0 && continue
    local (body, iterators) = bodyAt(eq, path)
    body isa IR.SCALAR_EQUATION || throw(NotHandled("differentiating a $(nameof(typeof(body))) in a for-equation"))
    local current::IR.Equation = inlineEquation(IR.SCALAR_EQUATION(canonicalReferences(body.lhs, E, variables),
                                                                   canonicalReferences(body.rhs, E, variables),
                                                                   body.ty, body.source, body.attributes), functions)
    local chain = IR.SCALAR_EQUATION[]
    for _ in 1:count
      current = differentiateEquation(current, seeds)
      push!(chain, current)
    end
    local values = Dict{Symbol, Int}()
    for r in pathRows
      foreach(((it, k),) -> (values[it] = k), zip(iterators, view.origins[r].iteration))
      local own::IR.Equation = inlineEquation(viewEquations[r], functions)
      for derivative in chain
        own = differentiateEquation(own, planSeeds)
        local mine = elementEquation(derivative, values, E)
        sameEquation(mine, own) ||
          throw(NotHandled("a for-equation whose derivative in array form is not the element's at $(view.origins[r].iteration): $(equationLine(mine)) vs $(equationLine(own))"))
      end
    end
    chains[path] = chain
  end
  return chains
end

"""
    linkEquation(link, derivative, sizes) -> Equation

`link = derivative` for every element of the arrays (of `sizes`): nested for-equations over the dimensions,
iterators `\$i1`, `\$i2`, ... (`linkStateDerivative!`'s equation in array form).
"""
function linkEquation(link::IR.Variable, derivative::IR.Variable, sizes::Vector{Int})::IR.Equation
  local subscripts = iteratorSubscripts(sizes)
  local element = v::IR.Variable -> IR.CREF(distributedCref(v, subscripts), IR.T_REAL())
  return nestedFor(IR.Equation[IR.SCALAR_EQUATION(element(link), element(derivative), IR.T_REAL(), IR.NO_SOURCE,
                                                  IR.DEFAULT_ATTRIBUTES)], sizes)
end

"A variable the array form of a plan adds, of no attributes."
newVariable(name::Symbol, ty::IR.ModelicaType)::IR.Variable =
  IR.Variable(name, ty, IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION, false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction)

"""
    stateLinks(s, dummies, E, derivatives, variables) -> Tuple{Vector{IR.Variable}, Vector{IR.Equation}}

The selected states' derivatives as `linkStateDerivative!` links them, in array form: the states of all the
elements of an array by one array `\$STATEDER.x` and a for-equation (`linkEquation`), a scalar state by a variable
and an equation; the new variables pushed into `variables` too. NotHandled for some elements of an array states
(slicing), and for states whose derivatives are not one array (`derivatives`).
"""
function stateLinks(s::Structure, dummies::Set{Int}, E::Elements, derivatives::Dict{Symbol, Symbol},
                    variables::IR.VariableSet)::Tuple{Vector{IR.Variable}, Vector{IR.Equation}}
  local byArray = Dict{Symbol, Vector{Symbol}}()
  local scalars = Symbol[]
  for v in selectedStates(s, dummies)
    local element = get(E.array, s.variables[v], nothing)
    element === nothing ? push!(scalars, s.variables[v]) : push!(get!(() -> Symbol[], byArray, element[1]), s.variables[v])
  end
  local links = IR.Variable[]
  local equations = IR.Equation[]
  for (array, selected) in sort!(collect(byArray); by = first)
    length(selected) == length(E.elements[array]) ||
      throw(NotHandled("$(length(selected)) of the $(length(E.elements[array])) elements of the array $(array) states (slicing)"))
    local derivative = get(derivatives, array, nothing)
    derivative === nothing && throw(NotHandled("the states of the array $(array), their derivatives not one array"))
    local link = newVariable(stateDerivativeName(array), variables[array].ty)
    push!(links, link)
    push!(variables, link)
    push!(equations, linkEquation(link, variables[derivative], E.sizes[array]))
  end
  for x in scalars
    local derivative = variables[s.variables[s.derivative[s.index[x]]]]
    local link = newVariable(stateDerivativeName(x), IR.T_REAL())
    push!(links, link)
    push!(variables, link)
    push!(equations, IR.SCALAR_EQUATION(IR.CREF(link.cref, IR.T_REAL()), IR.CREF(derivative.cref, IR.T_REAL()), IR.T_REAL(),
                                        IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
  end
  return (links, equations)
end

"""
    arrayKinds(kinds, E, variables) -> Dict{Symbol, VariableKind}

The kinds of the variables (`variables`, arrays and scalars) whose elements' or own kinds the plan changes
(`kinds`, by the element view's names): an array's from its elements' (`uniformKind`), an array whose elements are
variables of their own (`elements`) as a scalar. NotHandled for an array some of whose elements change their kind
(slicing).
"""
function arrayKinds(kinds::Dict{Symbol, IR.VariableKind}, E::Elements, variables::IR.VariableSet)::Dict{Symbol, IR.VariableKind}
  local out = Dict{Symbol, IR.VariableKind}()
  for v in variables
    local members = get(E.elements, v.name, nothing)
    if members !== nothing
      local changed = count(n -> haskey(kinds, n), members)
      changed == 0 && continue
      changed == length(members) ||
        throw(NotHandled("$(changed) of the $(length(members)) elements of the array $(v.name) change their kind (slicing)"))
      out[v.name] = uniformKind(v.name, IR.VariableKind[kinds[n] for n in members], E)
    elseif haskey(kinds, v.name)
      out[v.name] = kinds[v.name]
    end
  end
  return out
end

"""
    differentiatedArrayEquations(partition, view, p, E, variables, derivatives, functions) -> Vector{IR.Equation}

The equations the plan differentiates, their derivatives in array form, by equation of the partition: a
for-equation's for-equations of its body's derivatives (`forDerivatives`), another's the plan's derivatives of its
elements (an array equation the passes split, element by element; `arrayEquation`).
"""
function differentiatedArrayEquations(partition::IR.Partition, view::Scalarized, p::Plan, E::Elements, variables::IR.VariableSet,
                                      derivatives::Dict{Symbol, Symbol}, functions::Vector{IR.ModelicaFunction})::Vector{IR.Equation}
  local rowsOf = Dict{Int, Vector{Int}}()
  for (r, origin) in enumerate(view.origins)
    push!(get!(() -> Int[], rowsOf, origin.equation), r)
  end
  local functionsByName = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in functions)
  local seeds = arraySeeds(p.structure, derivatives, E, variables, functions)
  local planSeeds = derivativeSeeds(p.structure, functions)
  local out = IR.Equation[]
  for i in sort!(unique(Int[view.origins[r].equation for r in keys(p.differentiated)]))
    local eq = partition.equations[i]
    if eq isa IR.FOR_EQUATION
      local chains = forDerivatives(eq, rowsOf[i], view, p, seeds, planSeeds, E, variables, functionsByName)
      for level in 1:maximum(length, values(chains); init = 0)
        local d = differentiatedForEquation(eq, chains, level)
        d === nothing || push!(out, d)
      end
    else
      for r in rowsOf[i], derivative in get(p.differentiated, r, IR.Equation[])
        push!(out, arrayEquation(derivative, E, variables))
      end
    end
  end
  return out
end

"""
    stateSets(model, p, E, variables, viewVariables) -> Tuple{Vector{IR.Variable}, Vector{IR.Equation}, Int}

The plan's dynamic state sets: of scalars as `commit!` makes them (`dynamicSetParts`), of the elements of arrays by
family (`setFamilies`, `familyParts`); their variables, equations and number.
"""
function stateSets(model::IR.Model, p::Plan, E::Elements, variables::IR.VariableSet,
                   viewVariables::IR.VariableSet)::Tuple{Vector{IR.Variable}, Vector{IR.Equation}, Int}
  local newVariables = IR.Variable[]
  local equations = IR.Equation[]
  local (families, scalarSets) = setFamilies(p, E, variables)
  for (k, set) in enumerate(scalarSets)
    local (vs, eqs) = dynamicSetParts(model, p.structure, set, k)
    append!(newVariables, vs)
    #= its pivots read the element view's variables: references into their arrays =#
    append!(equations, IR.Equation[arrayEquation(eq, E, variables) for eq in eqs])
  end
  for (f, family) in enumerate(families)
    local (vs, eqs) = familyParts(family, length(scalarSets) + f, p.structure, E, variables, viewVariables)
    append!(newVariables, vs)
    append!(equations, eqs)
  end
  return (newVariables, equations, length(scalarSets) + length(families))
end

"""
    arrayCommit(model, partition, view, p) -> ArrayCommit

The plan `p`, made on the element view `view` of the arrays-kept `model`, in array form (`ArrayCommit`), the model
not changed: the new derivative variables (`newArrays`), the kinds (`plannedKinds`, an array's from its elements',
`arrayKinds`), the differentiated equations (`differentiatedArrayEquations`), the selected states' derivatives
(`stateLinks`), the dynamic state sets (`stateSets`). NotHandled where the plan treats an array's elements
differently.
"""
function arrayCommit(model::IR.Model, partition::IR.Partition, view::Scalarized, p::Plan)::ArrayCommit
  local s = p.structure
  local added = newArrays(Symbol[n for n in s.variables if !haskey(view.model.variables, n)], s, view.elements, model)
  #= all variables, the model's and the new ones (not yet in the model), and their arrays' elements =#
  local variables = IR.VariableSet(vcat(collect(model.variables), added))
  local E = extendedElements(view.elements, added, variables)
  local derivatives = arrayDerivatives(s, E)
  local (links, linkEquations) = stateLinks(s, p.dummies, E, derivatives, variables)
  E = extendedElements(E, links, variables)
  local kinds = arrayKinds(plannedKinds(s, p.dummies), E, variables)
  local equations = differentiatedArrayEquations(partition, view, p, E, variables, derivatives, model.functions)
  append!(equations, linkEquations)
  local (setVariables, setEquations, sets) = stateSets(model, p, E, variables, view.model.variables)
  append!(equations, setEquations)
  return ArrayCommit(vcat(added, links, setVariables), kinds, equations, sets)
end

"Whether the equation reads der() of an expression (a call introduceDerivatives left)."
function readsDerCall(eq::IR.Equation)::Bool
  local found = Ref(false)
  IR.foreachExp(e -> (IR.isBuiltinCall(e, :der) && (found[] = true); nothing), eq)
  return found[]
end

"""
    commitArrays!(model, partition, c)

The array form of a plan applied (`arrayCommit`): the new variables added, the kinds set, the equations appended.
"""
function commitArrays!(model::IR.Model, partition::IR.Partition, c::ArrayCommit)::Nothing
  foreach(v -> push!(model.variables, v), c.variables)
  for (name, kind) in c.kinds
    model.variables[name].kind = kind
  end
  append!(partition.equations, c.equations)
  return nothing
end

"""
    reduceIndexArrays!(model) -> Bool

Index reduction of an arrays-kept model in array form: planned on its element view (`scalarizeArrays`, `knownPlan`),
committed in array form (`arrayCommit`, `commitArrays!`); true where reduced. False where the model has no
continuous arrays or needs no reduction. NotHandled, the model unchanged, where the array form does not take the
plan (der() of an expression, an array's elements treated differently: the caller then takes the scalarized path).
"""
function reduceIndexArrays!(model::IR.Model)::Bool
  hasContinuousArrays(model) || return false
  any(readsDerCall, model.continuous[1].equations) && throw(NotHandled("der() of an expression with arrays kept"))
  local view = scalarizeArrays(model)
  local partition = view.model.continuous[1]
  local written = plan(view.model, partition)
  written === nothing && return false
  #= the array form of the plan as written first: a plan it does not take is refused before the simulation code's
     values are computed for the plan again (`knownPlan`) =#
  arrayCommit(model, model.continuous[1], view, written)
  local known = KNOWN_VALUES[] === nothing ? Dict{Symbol, IR.Expression}() : KNOWN_VALUES[](view.model)
  local p = isempty(known) ? written : plan(view.model, partition; known)
  p === nothing && return false
  local c = arrayCommit(model, model.continuous[1], view, p)
  commitArrays!(model, model.continuous[1], c)
  #= the derivative unknowns solved for, as on the scalarized path (`reduceIndex!`): left algebraic unknowns, MSL
     R134a2's der(p) of its volumes jumped at the dew line and the steps fell to 1e-16 =#
  local solved = envSwitch("OMBACKEND_SOLVE_DERIVATIVES") ? solveDerivatives!(model, model.continuous[1]) : Symbol[]
  local states = sort!([String(n) for (n, k) in c.kinds if k isa IR.STATE])
  local stateText = isempty(states) ? "" : ", states $(join(first(states, 8), ", "))"
  local setText = c.sets == 0 ? "" : ", $(c.sets) dynamic state set$(c.sets == 1 ? "" : "s")"
  local solvedText = isempty(solved) ? "" : "; $(length(solved)) derivatives solved for"
  @info "[index reduction] $(model.name): in array form, $(length(c.equations)) equations added$(stateText)$(setText)$(solvedText)"
  return true
end
