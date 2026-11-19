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
#= The plan made on the element view of an arrays-kept model (arrays.jl), committed in array form: the arrays
   whose elements the plan treats alike keep one kind, a for-equation whose iterations it differentiates alike is
   differentiated once, symbolically, its iterator kept (Otter and Elmqvist, Modelica Conference 2017, Theorem 2:
   under full incidence every element of an array equation is differentiated equally). Where the plan treats the
   elements of one array differently (a boundary element, a mixed choice of states), NotHandled: slicing the array
   is the next step. =#

"""
    ArrayCommit

What the array form of a plan adds to the model before it is changed: the new variables (derivative arrays, the
states' derivative arrays `\$STATEDER.x` and scalar ones, the dynamic state sets'), the new kind of each variable
whose kind changes, the equations to append (the differentiated ones, the states' links, the sets'), the number
of dynamic state sets (a family over the elements of arrays counted once), and the partition's equations with der()
of an expression expanded (`derivativesExpanded`), in their order.
"""
struct ArrayCommit
  variables::Vector{IR.Variable}
  kinds::Dict{Symbol, IR.VariableKind}
  equations::Vector{IR.Equation}
  sets::Int
  expanded::Vector{IR.Equation}
end

"""
    plannedKinds(s, dummies) -> Dict{Symbol, VariableKind}

The kind of each structure variable of the element view whose kind the plan changes (`plannedKind`), the selected
states linked to their derivatives as `linkStateDerivative!` links them: a state's derivative `\$STATEDER.x`
(STATE_DER), the derivative variable it read an algebraic unknown (ALGEBRAIC_DER) unless it is a state itself.
"""
function plannedKinds(s::Structure, dummies::Set{Int})::Dict{Symbol, IR.VariableKind}
  local kinds = Dict{Symbol, IR.VariableKind}()
  for v in eachindex(s.variables)
    local kind = plannedKind(s, dummies, v)
    kind === nothing || (kinds[s.variables[v]] = kind)
  end
  for v in selectedStates(s, dummies)
    local (state, derivative) = (s.variables[v], s.variables[s.derivative[v]])
    local link::Symbol = IR.stateDerivativeCref(s.crefs[v]).name
    kinds[state] = IR.STATE(1, link, true)
    kinds[link] = IR.STATE_DER(state)
    get(kinds, derivative, nothing) isa IR.STATE || (kinds[derivative] = IR.ALGEBRAIC_DER(state))
  end
  return kinds
end

"""
    arrayElementRef(name, E, variables) -> ComponentRef

The element variable `name` of the element view as a reference into its array in the arrays-kept model
(`x[2]`, `c[2].T`; `\$DER.x[2]` as the element 2 of the derivative array `\$DER.x`, `IR.withLastSubscripts`); a scalar
variable by its own reference; NotHandled for a name of neither.
"""
function arrayElementRef(name::Symbol, E::Elements, variables::IR.VariableSet)::IR.ComponentRef
  local element = get(E.array, name, nothing)
  if element === nothing
    local v::Union{Nothing, IR.Variable} = get(variables, name, nothing)
    v === nothing && throw(NotHandled("the element view's variable $(name), of no array and not of the model"))
    return v.cref
  end
  local (array::Symbol, index::Vector{Int}) = element
  local arrayVariable::Union{Nothing, IR.Variable} = get(variables, array, nothing)
  (arrayVariable !== nothing && IR.isDerivativeCref(arrayVariable.cref)) &&
    return IR.withLastSubscripts(arrayVariable.cref, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index])
  local cref = E.crefs[name]
  IR.hasSubscripts(cref) && return cref
  #= another backend-made array of one part, its elements named by flat names: the element of its one part =#
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
    #= an element of an array split along its inner dimensions: the slice's (`slicedCref`) =#
    if !haskey(variables, unsubscriptedName(x.cref))
      local sliced::Union{Nothing, IR.ComponentRef} = slicedCref(x.cref, (n::Symbol) -> haskey(variables, n))
      sliced === nothing || (x = IR.CREF(sliced, x.ty))
    end
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
  all(k -> k isa K, kinds) ||
    throw(NotHandled("an array whose elements the plan gives different kinds ($(array): $(unique(typeof.(kinds))))", Symbol[array], Int[]))
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
kindVariable(k::IR.ALGEBRAIC_DER)::Symbol = k.state
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
  #= a new derivative array's array of the model (to split for slicing: `x` of `\$DER.x`), and its reference =#
  local rootOf::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}()
  local crefOf::Dict{Symbol, IR.ComponentRef} = Dict{Symbol, IR.ComponentRef}()
  #= every array derived for some of its elements only, collected: split together =#
  local partial::Vector{Symbol} = Symbol[]
  local firstPartial::String = ""
  local byOrder = Dict{Int, Vector{Symbol}}()
  foreach(n -> push!(get!(() -> Symbol[], byOrder, derivativeOrder(s, s.index[n])), n), names)
  for order in sort!(collect(keys(byOrder)))
    local byArray = Dict{Symbol, Vector{Symbol}}()
    for name in byOrder[order]
      local element = get(arrayOf, s.variables[s.antiderivative[s.index[name]]], nothing)
      if element === nothing
        push!(out, IR.Variable(s.crefs[s.index[name]], IR.T_REAL(), IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                               false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
      else
        push!(get!(() -> Symbol[], byArray, element[1]), name)
      end
    end
    for (array, derived) in sort!(collect(byArray); by = first)
      #= each element's derivative among those derived (the structure's antiderivative of each), in the elements' order =#
      local derivativeOf::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}(s.variables[s.antiderivative[s.index[d]]] => d for d in derived)
      if length(derived) != length(elementsOf[array]) || !all(e::Symbol -> haskey(derivativeOf, e), elementsOf[array])
        isempty(partial) && (firstPartial = "the derivatives of $(length(derived)) of the $(length(elementsOf[array])) elements of the array $(array)")
        push!(partial, get(rootOf, array, array))
        continue
      end
      local expected::Vector{Symbol} = Symbol[derivativeOf[e] for e in elementsOf[array]]
      local cref::IR.ComponentRef = IR.derivativeCref(haskey(crefOf, array) ? crefOf[array] : model.variables[array].cref)
      local name::Symbol = cref.name
      haskey(model.variables, name) && throw(NotHandled("a derivative array $(name) that exists"))
      crefOf[name] = cref
      push!(out, IR.Variable(cref, IR.T_ARRAY(IR.T_REAL(), dimensionsOf[array]), IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES,
                             IR.NO_DIRECTION, false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
      dimensionsOf[name] = dimensionsOf[array]
      elementsOf[name] = expected
      rootOf[name] = get(rootOf, array, array)
      for (e, d) in zip(elementsOf[array], expected)
        arrayOf[d] = (name, arrayOf[e][2])
      end
    end
  end
  isempty(partial) || throw(NotHandled("$(firstPartial)$(length(partial) > 1 ? " and $(length(partial) - 1) more arrays" : "") (slicing)",
                                       unique(partial), Int[]))
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
    arraySeeds(s, derivatives, E, variables, functions, constants) -> Seeds

The time derivative in array form: of an array variable the derivative array (`derivatives`; `x[i]` is then
`\$DER.x[i]`, `arrayElementDerivative`), of an element or a scalar the structure's derivative as a reference into its
array, of one the plan's structure read as a constant (`constants`) zero; NotHandled for an array of continuous
variables whose elements' derivatives are not one array.
"""
function arraySeeds(s::Structure, derivatives::Dict{Symbol, Symbol}, E::Elements, variables::IR.VariableSet,
                    functions::Vector{IR.ModelicaFunction}, constants::Set{Symbol})::Seeds
  local variable = (name::Symbol) -> begin
    local members = get(E.elements, name, nothing)
    if members !== nothing
      local derivative = get(derivatives, name, nothing)
      derivative === nothing || return IR.CREF(variables[derivative].cref, variables[derivative].ty)
      any(n -> haskey(s.index, n), members) &&
        throw(NotHandled("the derivative of the array $(name), its elements' derivatives not one array", Symbol[name], Int[]))
      return nothing
    end
    local v = get(s.index, name, 0)
    v == 0 && return nothing
    local d = s.derivative[v]
    (d == 0 && name in constants) && return ZERO
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
  return IR.FOR_EQUATION(eq.iterator, body, eq.source, IR.EquationAttributes(eq.attributes.kind, true, nothing, false))
end

"The scalar equation `eq` of the arrays-kept model at the iterator `values`, in the element view's form (`equationExpression`)."
elementEquation(eq::IR.SCALAR_EQUATION, values::Dict{Symbol, Int}, E::Elements)::IR.SCALAR_EQUATION =
  IR.SCALAR_EQUATION(equationExpression(eq.lhs, values, E), equationExpression(eq.rhs, values, E), eq.ty, eq.source, eq.attributes)

"""
The if-equation `eq` of the arrays-kept model at the iterator `values`, as the element view writes one
(`scalarEquations!`): a branch whose condition folds to false left out, one that folds to true the else of those
before it; the else's one equation alone where no branch is left.
"""
function elementEquation(eq::IR.IF_EQUATION, values::Dict{Symbol, Int}, E::Elements)::IR.Equation
  local branches::Vector{IR.EquationBranch} = IR.EquationBranch[]
  local elseEquations::Vector{IR.Equation} = eq.elseEquations
  for b::IR.EquationBranch in eq.branches
    local condition::IR.Expression = foldTree(equationExpression(b.condition, values, E))
    local taken::Union{Nothing, Bool} = literalCondition(condition)
    taken === false && continue
    taken === true && (elseEquations = b.equations; break)
    push!(branches, IR.EquationBranch(condition, IR.Equation[elementEquation(e, values, E) for e in b.equations]))
  end
  local rest::Vector{IR.Equation} = IR.Equation[elementEquation(e, values, E) for e in elseEquations]
  (isempty(branches) && length(rest) == 1) && return rest[1]
  return IR.IF_EQUATION(branches, rest, eq.source, eq.attributes)
end

"""
Whether two scalar equations are one, their sides' arithmetic folded (`foldTree`): structurally, or up to the order
and the grouping of their sums' terms and the order of their products' factors (`equationForm`: the derivative of a
scalar product written `a * b` in a for-equation's body and term by term in the element view).
"""
function sameEquation(a::IR.SCALAR_EQUATION, b::IR.SCALAR_EQUATION)::Bool
  local (al::IR.Expression, ar::IR.Expression) = (foldTree(a.lhs), foldTree(a.rhs))
  local (bl::IR.Expression, br::IR.Expression) = (foldTree(b.lhs), foldTree(b.rhs))
  (al == bl && ar == br) && return true
  return equationForm(IR.SCALAR_EQUATION(al, ar, a.ty, a.source, a.attributes)) ==
         equationForm(IR.SCALAR_EQUATION(bl, br, b.ty, b.source, b.attributes))
end

"An equation's line with its sides folded as `sameEquation` compares them (`foldTree`); another equation's as it is."
foldedLine(eq::IR.SCALAR_EQUATION)::String =
  equationLine(IR.SCALAR_EQUATION(foldTree(eq.lhs), foldTree(eq.rhs), eq.ty, eq.source, eq.attributes))
foldedLine(eq::IR.Equation)::String = equationLine(eq)

"""
Where the texts `a` and `b` first differ, 100 characters of each around it (`a ... vs b ...`): a message's part for
two equations of thousands of characters (a medium's derivative: 98000 each, the whole of both in a refusal).
"""
function firstDifference(a::String, b::String)::String
  local (x::Vector{Char}, y::Vector{Char}) = (collect(a), collect(b))
  local shorter::Int = min(length(x), length(y))
  local k::Int = something(findfirst(i::Int -> x[i] != y[i], 1:shorter), shorter + 1)
  local around = (t::Vector{Char}) -> String(t[max(1, k - 40):min(length(t), k + 60)])
  return string("at character ", k, " of ", length(x), " and ", length(y), ": ", around(x), " vs ", around(y))
end

"Whether `name` is a Real parameter or constant of `variables` whose value `values` knows as zero."
function isKnownZero(name::Symbol, variables::IR.VariableSet, values::StartValues)::Bool
  local v = get(variables, name, nothing)
  (v === nothing || !(v.kind isa Union{IR.PARAMETER, IR.CONSTANT}) || !(v.ty isa IR.T_REAL)) && return false
  local x = values(name)
  return x !== nothing && iszero(x)
end

"""
    withoutKnownZeros(eq, variables, values, E) -> IR.SCALAR_EQUATION

The scalar equation `eq` with each reference to a Real parameter whose value is known as zero before the simulation
(`values`: the parameters', by the element view's `variables`) as that zero, its arithmetic folded: the terms the
plan's structure does not read (`withKnownValues`; a prismatic joint's `e = {0, -1, 0}` leaves `R.T[1, i] * e[1] * s`
out), whose variables Pantelides' algorithm gave no derivatives. A reference is a scalar's, or an element of an
array of `E` at literal subscripts (`cylinder.e[1]`, the array's reference subscripted).
"""
function withoutKnownZeros(eq::IR.SCALAR_EQUATION, variables::IR.VariableSet, values::StartValues,
                           E::Elements)::IR.SCALAR_EQUATION
  local zeroed = (@nospecialize(e::IR.Expression)) -> begin
    if e isa IR.CREF
      return isKnownZero(e.cref.name, variables, values) ? ZERO : e
    elseif e isa IR.SUBSCRIPTED && e.exp isa IR.CREF && !IR.hasSubscripts(e.exp.cref)
      local array::Symbol = e.exp.cref.name
      local sizes::Union{Nothing, Vector{Int}} = get(E.sizes, array, nothing)
      (sizes === nothing || length(sizes) != length(e.subscripts) || !haskey(E.elements, array)) && return e
      local index::Vector{Int} = Int[]
      for sub::IR.Subscript in e.subscripts
        local k::Union{Nothing, Int} = IR.literalIndex(sub)
        (k === nothing || !(1 <= k <= sizes[length(index) + 1])) && return e
        push!(index, k)
      end
      return isKnownZero(elementName(array, index, E), variables, values) ? ZERO : e
    end
    return e
  end
  return IR.SCALAR_EQUATION(foldTree(IR.mapExp(zeroed, eq.lhs)), foldTree(IR.mapExp(zeroed, eq.rhs)), eq.ty, eq.source,
                            eq.attributes)
end
"Two if-equations: the same conditions (folded) in order and their branches' equations the same, one by one."
function sameEquation(a::IR.IF_EQUATION, b::IR.IF_EQUATION)::Bool
  (length(a.branches) == length(b.branches) && length(a.elseEquations) == length(b.elseEquations)) || return false
  for (x::IR.EquationBranch, y::IR.EquationBranch) in zip(a.branches, b.branches)
    (foldTree(x.condition) == foldTree(y.condition) && length(x.equations) == length(y.equations)) || return false
    all(((e, f),) -> sameEquation(e, f), zip(x.equations, y.equations)) || return false
  end
  return all(((e, f),) -> sameEquation(e, f), zip(a.elseEquations, b.elseEquations))
end

"An if-equation without the terms of parameters known as zero, in each branch's scalar equations."
function withoutKnownZeros(eq::IR.IF_EQUATION, variables::IR.VariableSet, values::StartValues,
                           E::Elements)::IR.IF_EQUATION
  local folded = (equations::Vector{IR.Equation}) ->
    IR.Equation[e isa IR.SCALAR_EQUATION ? withoutKnownZeros(e, variables, values, E) : e for e in equations]
  return IR.IF_EQUATION(IR.EquationBranch[IR.EquationBranch(b.condition, folded(b.equations)) for b in eq.branches],
                        folded(eq.elseEquations), eq.source, eq.attributes)
end

"Another equation as it is."
withoutKnownZeros(eq::IR.Equation, ::IR.VariableSet, ::StartValues, ::Elements)::IR.Equation = eq

sameEquation(::IR.Equation, ::IR.Equation)::Bool = false

"An equation as one line of text, for a message."
equationLine(eq::IR.Equation)::String = replace(string(eq), r"\s*\n\s*" => " ")

"""
The names of the variables (no parameters or constants) the equation `eq` reads (a for-equation: its body), an
array's for a read of an element of it (`variableNameOf`).
"""
function variableReads(eq::IR.Equation, variables::IR.VariableSet)::Set{Symbol}
  local names::Set{Symbol} = Set{Symbol}()
  local visit = (@nospecialize(e::IR.Expression)) -> begin
    if e isa IR.CREF
      local v = get(variables, variableNameOf(e.cref, variables), nothing)
      (v === nothing || v.kind isa Union{IR.PARAMETER, IR.CONSTANT}) || push!(names, v.name)
    end
    nothing
  end
  IR.foreachExp(visit, eq)
  return names
end

"""
    dropKnownZeroTerms!(model, view) -> Bool

The scalar equations of the model's continuous partition as the plan's structure reads them (`withKnownValues`):
without the terms a Real parameter known as zero before the simulation takes out (`withoutKnownZeros`), where that
takes a variable out of the equation. Left in, the committed model kept a variable the plan did not count on: MSL
FundamentalWave's converter at orientation 0, `-v = N.re * der(Phi.re) + N.im * der(Phi.im)` with `N.im = 0`, kept
`der(Phi.im)` as an unknown whose only coefficient is zero (the array code: structurally singular). An equation whose
variables the zeros leave in place stays as written, its parameters symbolic. A scalar state whose derivative no
equation reads after that is no state (`demoteDeadStates!`). Whether an equation changed.
"""
function dropKnownZeroTerms!(model::IR.Model, view::Scalarized)::Bool
  local variables::IR.VariableSet = view.model.variables
  local values::StartValues = StartValues(variables, view.model.functions; parametersOnly = true)
  #= a zero of some elements only (a polyphase converter's `N[k].im` at the phase of orientation 0) that takes a
     derivative out of those iterations: the state of that element is none while its siblings are, so the arrays are
     split and their equations unrolled; the terms then go out of the scalar equations =#
  local sliced::Vector{Symbol} = derivativesZeroedInArrays(model, view, values)
  if !isempty(sliced)
    splitArrays!(model, view, sliced, Int[])
    view = scalarizeArrays(model)
    variables = view.model.variables
    values = StartValues(variables, view.model.functions; parametersOnly = true)
  end
  local equations::Vector{IR.Equation} = model.continuous[1].equations
  local dropped::Set{Symbol} = Set{Symbol}()
  for (k::Int, eq::IR.Equation) in enumerate(equations)
    eq isa IR.SCALAR_EQUATION || continue
    local folded::IR.SCALAR_EQUATION = withoutKnownZeros(eq, variables, values, view.elements)
    local before::Set{Symbol} = variableReads(eq, model.variables)
    local after::Set{Symbol} = variableReads(folded, model.variables)
    (length(after) < length(before) && issubset(after, before)) || continue
    equations[k] = folded
    union!(dropped, setdiff(before, after))
  end
  #= the variables the equations give as zero (the folds above made some: `V_m = 0` of a core without conductance),
     as the plan's structure knows them (`withKnownValues`), out of the equations where that takes another variable
     out (`lossPower = V_m * der(Phi)`: the plan counts on no der(Phi) there) =#
  dropZeroVariableTerms!(equations, model.variables, dropped, Set{Symbol}(), Set{Int}())
  isempty(dropped) || demoteDeadStates!(model, dropped)
  return !isempty(dropped) || !isempty(sliced)
end

"""
    dropZeroVariableTerms!(equations, variables, dropped, known, defining) -> Nothing

The scalar equations and the for-equations (their bodies) without the reads of the variables that are zero
(`zeroVariables`: those of `known`, and those the scalar equations give), where that takes another variable out of
the equation; the variables taken out added to `dropped`. The equations of `defining` (by index: those that define a
variable of `known`) stay as written.
"""
function dropZeroVariableTerms!(equations::Vector{IR.Equation}, variables::IR.VariableSet, dropped::Set{Symbol},
                                known::Set{Symbol}, defining::Set{Int})::Nothing
  local (givenZero::Set{Symbol}, giving::Dict{Int, Symbol}) = zeroVariables(equations, variables, known, defining)
  isempty(givenZero) && return nothing
  for (k::Int, eq::IR.Equation) in enumerate(equations)
    (eq isa Union{IR.SCALAR_EQUATION, IR.FOR_EQUATION} && !(k in defining)) || continue
    local folded::IR.Equation = withoutZeroVariables(eq, variables, givenZero, get(giving, k, nothing))
    local out::Set{Symbol} = setdiff(variableReads(eq, variables), variableReads(folded, variables))
    issubset(out, givenZero) && continue
    equations[k] = folded
    union!(dropped, out)
  end
  return nothing
end

"""
    dropFunctionZeroTerms!(model, view, known) -> Bool

`dropKnownZeroTerms!` for the zeros among the simulation code's values `known` (`KNOWN_VALUES`, of the element view
`view`) of the variables a function call defines (`functionDefinedValues`: a record function's output its
interpretation finds constant), which the plan made with them reads as literals (`withKnownValues`) in every
equation but the defining one: MSL Spice3's MOSFET without capacitances,
`icGS = cc.cGS * (der(G.v) - der(Sinternal))` with `cc.cGS = mosCalcNoBypassCode(...)` zero. The plan then
differentiates nothing and has no states; left in, the committed model kept the gate's and the internal nodes'
voltages as states without an equation for a derivative (the array code: structurally singular,
`0 = S.v - Sinternal` reads no unknown). An array is zero where all its elements are (the MOSFETs rolled into one
class, their equations for-equations: Spice3's inverter chains); one of some zero elements only stays as written.
Whether an equation changed.
"""
function dropFunctionZeroTerms!(model::IR.Model, view::Scalarized, known::Dict{Symbol, IR.Expression})::Bool
  local constants::Dict{Symbol, Tuple{IR.Expression, Int}} = functionDefinedValues(view.model.continuous[1], known)
  local zeroElements::Set{Symbol} = Set{Symbol}()
  for (name::Symbol, c::Tuple{IR.Expression, Int}) in constants
    isZero(c[1]) && push!(zeroElements, name)
  end
  isempty(zeroElements) && return false
  #= the model's variables that are zero: a scalar, an array of zero elements =#
  local zero::Set{Symbol} = Set{Symbol}()
  for v::IR.Variable in model.variables
    local elements::Union{Nothing, Vector{Symbol}} = get(view.elements.elements, v.name, nothing)
    (elements === nothing ? v.name in zeroElements : all(in(zeroElements), elements)) && push!(zero, v.name)
  end
  isempty(zero) && return false
  #= the defining equations, by the model's equations the view's rows are of =#
  local defining::Set{Int} = Set{Int}(view.origins[c[2]].equation for c::Tuple{IR.Expression, Int} in values(constants))
  local dropped::Set{Symbol} = Set{Symbol}()
  dropZeroVariableTerms!(model.continuous[1].equations, model.variables, dropped, zero, defining)
  isempty(dropped) && return false
  demoteDeadStates!(model, dropped)
  return true
end

"""
    zeroVariable(eq, variables) -> Union{Nothing, Symbol}

The Real algebraic scalar variable the scalar equation `eq` gives as zero: `x = 0`, `0 = x` (the plan's
`setsConstant`), also negated or times a literal other than zero (`1.5707963267948966 * V_m.re = 0`); nothing for
any other equation.
"""
function zeroVariable(eq::IR.SCALAR_EQUATION, variables::IR.VariableSet)::Union{Nothing, Symbol}
  (isZero(eq.rhs) || isZero(eq.lhs)) || return nothing
  local side::IR.Expression = isZero(eq.rhs) ? eq.lhs : eq.rhs
  while true
    if side isa IR.UNARY && side.op === IR.UMINUS
      side = side.exp
    elseif side isa IR.BINARY && side.op === IR.MUL && isLiteral(side.exp1) && !isZero(side.exp1)
      side = side.exp2
    elseif side isa IR.BINARY && side.op === IR.MUL && isLiteral(side.exp2) && !isZero(side.exp2)
      side = side.exp1
    else
      break
    end
  end
  (side isa IR.CREF && !IR.hasSubscripts(side.cref)) || return nothing
  local v = get(variables, side.cref.name, nothing)
  (v !== nothing && v.kind isa Union{IR.ALGEBRAIC, IR.ALGEBRAIC_DER} && v.ty isa IR.T_REAL) || return nothing
  return v.name
end

"""
    zeroVariables(equations, variables, known, defining) -> (Set{Symbol}, Dict{Int, Symbol})

The variables the scalar equations give as zero, and for each equation that gives one (by its index) the variable:
by an equation of its own (`zeroVariable`), or by one that is such an equation once the others are zero (`y = x`,
`y = x * w`), until no more is found. `known`: variables that are zero by other means (a function's constant
output), in the result too; the equations of `defining` (those that define them) give none.
"""
function zeroVariables(equations::Vector{IR.Equation}, variables::IR.VariableSet, known::Set{Symbol},
                       defining::Set{Int})::Tuple{Set{Symbol}, Dict{Int, Symbol}}
  local givenZero::Set{Symbol} = copy(known)
  local giving::Dict{Int, Symbol} = Dict{Int, Symbol}()
  local grew::Bool = true
  while grew
    grew = false
    for (k::Int, eq::IR.Equation) in enumerate(equations)
      (eq isa IR.SCALAR_EQUATION && !haskey(giving, k) && !(k in defining)) || continue
      local folded::IR.SCALAR_EQUATION =
        isempty(givenZero) ? eq : withoutZeroVariables(eq, variables, givenZero, nothing)
      local name::Union{Nothing, Symbol} = zeroVariable(folded, variables)
      (name === nothing || name in givenZero) && continue
      push!(givenZero, name)
      giving[k] = name
      grew = true
    end
  end
  return (givenZero, giving)
end

"""
The scalar equation `eq` with each read of a variable of `givenZero` (of an element of one that is an array) as
that zero, its arithmetic folded; `own`: the variable this equation gives, which stays.
"""
function withoutZeroVariables(eq::IR.SCALAR_EQUATION, variables::IR.VariableSet, givenZero::Set{Symbol},
                              own::Union{Nothing, Symbol})::IR.SCALAR_EQUATION
  local zeroed = (@nospecialize(e::IR.Expression)) -> begin
    e isa IR.CREF || return e
    local name::Symbol = variableNameOf(e.cref, variables)
    return (name !== own && name in givenZero) ? ZERO : e
  end
  return IR.SCALAR_EQUATION(foldTree(IR.mapExp(zeroed, eq.lhs)), foldTree(IR.mapExp(zeroed, eq.rhs)), eq.ty, eq.source,
                            eq.attributes)
end

"A for-equation with its body's scalar equations (those of a for-equation in it too) so."
function withoutZeroVariables(eq::IR.FOR_EQUATION, variables::IR.VariableSet, givenZero::Set{Symbol},
                              own::Union{Nothing, Symbol})::IR.FOR_EQUATION
  local body::Vector{IR.Equation} = IR.Equation[]
  for b::IR.Equation in eq.body
    local inBody::Bool = b isa Union{IR.SCALAR_EQUATION, IR.FOR_EQUATION}
    push!(body, inBody ? withoutZeroVariables(b, variables, givenZero, own) : b)
  end
  return IR.FOR_EQUATION(eq.iterator, body, eq.source, eq.attributes)
end

"""
    derivativesZeroedInArrays(model, view, values) -> Vector{Symbol}

The arrays of derivative variables an element of which a parameter known as zero takes out of an iteration of a
for-equation or array equation (`withoutKnownZeros` of the element view's equation reads it no more): in those
iterations the plan's structure has no such derivative, in the others it has (`dropKnownZeroTerms!` splits them).
"""
function derivativesZeroedInArrays(model::IR.Model, view::Scalarized, values::StartValues)::Vector{Symbol}
  local variables::IR.VariableSet = view.model.variables
  local equations::Vector{IR.Equation} = model.continuous[1].equations
  local rows::Vector{IR.Equation} = view.model.continuous[1].equations
  local arrayOf::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}()
  for (array::Symbol, elements::Vector{Symbol}) in view.elements.elements, element::Symbol in elements
    arrayOf[element] = array
  end
  local arrays::Set{Symbol} = Set{Symbol}()
  for (r::Int, row::IR.Equation) in enumerate(rows)
    (row isa IR.SCALAR_EQUATION && !(equations[view.origins[r].equation] isa IR.SCALAR_EQUATION)) || continue
    local folded::IR.SCALAR_EQUATION = withoutKnownZeros(row, variables, values, view.elements)
    local before::Set{Symbol} = variableReads(row, variables)
    local after::Set{Symbol} = variableReads(folded, variables)
    (length(after) < length(before) && issubset(after, before)) || continue
    for name::Symbol in setdiff(before, after)
      (variables[name].kind isa IR.STATE_DER && haskey(arrayOf, name)) && push!(arrays, arrayOf[name])
    end
  end
  return sort!(collect(arrays))
end

"""
    demoteDeadStates!(model, candidates)

The Real states (scalars and arrays) whose derivative variable (of `candidates`: the variables a fold took out of an
equation) no equation of the model reads any more: algebraic variables, their derivative variables gone, as the
plan's structure reads them (`Structure`) and SimCode's `demoteDeadStates` makes them on the scalarized path. Left a
state, the array code kept it as given and its derivative as an unknown without an equation.
"""
function demoteDeadStates!(model::IR.Model, candidates::Set{Symbol})::Nothing
  local read::Set{Symbol} = Set{Symbol}()
  local visit = (@nospecialize(e::IR.Expression)) ->
    (e isa IR.CREF && push!(read, variableNameOf(e.cref, model.variables)); nothing)
  for partition::IR.Partition in model.continuous
    foreach((eq::IR.Equation) -> IR.foreachExp(visit, eq), partition.equations)
    foreach((eq::IR.Equation) -> IR.foreachExp(visit, eq), partition.initialEquations)
  end
  local dead::Set{Symbol} = Set{Symbol}()
  for v::IR.Variable in model.variables
    (v.kind isa IR.STATE && IR.elementType(v.ty) isa IR.T_REAL && v.kind.derivative !== nothing) || continue
    local derivative::Symbol = v.kind.derivative
    (derivative in candidates && !(derivative in read) && haskey(model.variables, derivative)) || continue
    v.kind = IR.ALGEBRAIC()
    push!(dead, derivative)
  end
  isempty(dead) || filter!((v::IR.Variable) -> !(v.name in dead), model.variables)
  return nothing
end

"""
    preparedBody(body, E, variables, elementVariables, functions, known) -> Equation

The body equation of a for-equation as `forDerivatives` differentiates it: its references canonical
(`canonicalReferences`), its calls inlined (the references an inlined body made, a record argument's fields, canonical
too), without the terms of parameters known as zero (`withoutKnownZeros`). A scalar equation; or an if-equation of
one scalar equation per branch, its conditions' references canonical: an if-expression the passes lifted
(`ifEq_tmp = ...` in every branch), in a for-equation where its components were rolled (MSL Spice3's `V_pulse`,
Blocks' `Ramp` in two sources of one class: `if time < startTime then 0 else ...`, a voltage or current the
constraint differentiates). NotHandled for another body.
"""
function preparedBody(body::IR.SCALAR_EQUATION, E::Elements, variables::IR.VariableSet,
                      elementVariables::IR.VariableSet, functions::Dict{Symbol, IR.ModelicaFunction},
                      known::StartValues)::IR.Equation
  local canonical = (@nospecialize(e::IR.Expression)) -> canonicalReferences(e, E, variables)
  local inlined::IR.Equation =
    inlineEquation(IR.SCALAR_EQUATION(canonical(body.lhs), canonical(body.rhs), body.ty, body.source, body.attributes),
                   functions)
  local again::IR.SCALAR_EQUATION = IR.SCALAR_EQUATION(canonical(inlined.lhs), canonical(inlined.rhs), inlined.ty,
                                                       inlined.source, inlined.attributes)
  return withoutKnownZeros(again, elementVariables, known, E)
end

function preparedBody(body::IR.IF_EQUATION, E::Elements, variables::IR.VariableSet, elementVariables::IR.VariableSet,
                      functions::Dict{Symbol, IR.ModelicaFunction}, known::StartValues)::IR.Equation
  local branches::Vector{IR.EquationBranch} = IR.EquationBranch[]
  for b::IR.EquationBranch in body.branches
    push!(branches, IR.EquationBranch(canonicalReferences(b.condition, E, variables),
                                      preparedBranch(b.equations, E, variables, elementVariables, functions, known)))
  end
  return IR.IF_EQUATION(branches, preparedBranch(body.elseEquations, E, variables, elementVariables, functions, known),
                        body.source, body.attributes)
end

preparedBody(body::IR.Equation, ::Elements, ::IR.VariableSet, ::IR.VariableSet, ::Dict{Symbol, IR.ModelicaFunction},
             ::StartValues)::IR.Equation =
  throw(NotHandled("differentiating a $(nameof(typeof(body))) in a for-equation"))

"The one scalar equation of an if-equation's branch, prepared (`preparedBody`); NotHandled for another branch."
function preparedBranch(equations::Vector{IR.Equation}, E::Elements, variables::IR.VariableSet,
                        elementVariables::IR.VariableSet, functions::Dict{Symbol, IR.ModelicaFunction},
                        known::StartValues)::Vector{IR.Equation}
  (length(equations) == 1 && equations[1] isa IR.SCALAR_EQUATION) ||
    throw(NotHandled("differentiating an if-equation in a for-equation whose branch is not one scalar equation"))
  return IR.Equation[preparedBody(equations[1], E, variables, elementVariables, functions, known)]
end

"""
    forDerivatives(eq, rows, view, p, seeds, planSeeds, E, variables, functions, known)
      -> Dict{Vector{Int}, Vector{IR.Equation}}

The derivatives in array form of the body equations of the for-equation `eq` that the plan differentiates
(`rows`: the element view's equations of `eq`): each body equation (by its path) differentiated symbolically as
often as the plan differentiates each of its iterations, its iterators kept; checked against the derivative of
every iteration (its element view's equation differentiated by the plan's structure, `planSeeds`, as written: the
parameters symbolic, which the plan's own equations read as their values; those known as zero, `known`, are zero in
both: `withoutKnownZeros`). A body that is an if-equation of one equation per branch (a lifted if-expression) is
differentiated branch by branch, as one at the top level (`preparedBody`). NotHandled where the iterations are
differentiated unequally (slicing) or the array form's derivative of one is not its own.
"""
function forDerivatives(eq::IR.FOR_EQUATION, rows::Vector{Int}, view::Scalarized, p::Plan, seeds::Seeds, planSeeds::Seeds,
                        E::Elements, variables::IR.VariableSet,
                        functions::Dict{Symbol, IR.ModelicaFunction},
                        known::StartValues)::Dict{Vector{Int}, Vector{IR.Equation}}
  local elementVariables::IR.VariableSet = view.model.variables
  local byPath = Dict{Vector{Int}, Vector{Int}}()
  foreach(r -> push!(get!(() -> Int[], byPath, view.origins[r].path), r), rows)
  local viewEquations = view.model.continuous[1].equations
  local chains = Dict{Vector{Int}, Vector{IR.Equation}}()
  for (path, pathRows) in byPath
    local count = length(get(p.differentiated, pathRows[1], IR.Equation[]))
    all(r -> length(get(p.differentiated, r, IR.Equation[])) == count, pathRows) ||
      throw(NotHandled("a for-equation whose iterations are differentiated unequally (slicing)", Symbol[],
                       Int[view.origins[pathRows[1]].equation]))
    count == 0 && continue
    local (body, iterators) = bodyAt(eq, path)
    local current::IR.Equation = preparedBody(body, E, variables, elementVariables, functions, known)
    local chain = IR.Equation[]
    for _ in 1:count
      current = differentiateEquation(current, seeds)
      push!(chain, current)
    end
    local values = Dict{Symbol, Int}()
    for r in pathRows
      foreach(((it, k),) -> (values[it] = k), zip(iterators, view.origins[r].iteration))
      local own::IR.Equation = inlineEquation(viewEquations[r], functions)
      own = withoutKnownZeros(own, elementVariables, known, E)
      for derivative in chain
        own = differentiateEquation(own, planSeeds)
        #= an iteration's own zeros (`p[i]` at this i) =#
        local mine = withoutKnownZeros(elementEquation(derivative, values, E), elementVariables, known, E)
        sameEquation(mine, own) ||
          throw(NotHandled("a for-equation whose derivative in array form is not the element's at " *
                           "$(view.origins[r].iteration): $(firstDifference(foldedLine(mine), foldedLine(own)))"))
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
                                                  IR.DERIVATIVE_LINK_ATTRIBUTES)], sizes)
end

"A variable the array form of a plan adds, of no attributes."
newVariable(cref::IR.ComponentRef, ty::IR.ModelicaType)::IR.Variable =
  IR.Variable(cref, ty, IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION, false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction)

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
      throw(NotHandled("$(length(selected)) of the $(length(E.elements[array])) elements of the array $(array) states (slicing)",
                       Symbol[array], Int[]))
    local derivative = get(derivatives, array, nothing)
    derivative === nothing && throw(NotHandled("the states of the array $(array), their derivatives not one array"))
    local link::IR.Variable = newVariable(IR.stateDerivativeCref(variables[array].cref), variables[array].ty)
    push!(links, link)
    push!(variables, link)
    push!(equations, linkEquation(link, variables[derivative], E.sizes[array]))
  end
  for x in scalars
    local derivative = variables[s.variables[s.derivative[s.index[x]]]]
    local link::IR.Variable = newVariable(IR.stateDerivativeCref(variables[x].cref), IR.T_REAL())
    push!(links, link)
    push!(variables, link)
    push!(equations, IR.SCALAR_EQUATION(IR.CREF(link.cref, IR.T_REAL()), IR.CREF(derivative.cref, IR.T_REAL()), IR.T_REAL(),
                                        IR.NO_SOURCE, IR.DERIVATIVE_LINK_ATTRIBUTES))
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
        throw(NotHandled("$(changed) of the $(length(members)) elements of the array $(v.name) change their kind (slicing)",
                         Symbol[v.name], Int[]))
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
  local seeds = arraySeeds(p.structure, derivatives, E, variables, functions, p.constants)
  local planSeeds = derivativeSeeds(p.structure, functions; constants = p.constants)
  local known::StartValues = StartValues(view.model.variables, view.model.functions; parametersOnly = true)
  local out = IR.Equation[]
  for i in sort!(unique(Int[view.origins[r].equation for r in keys(p.differentiated)]))
    local eq = partition.equations[i]
    if eq isa IR.FOR_EQUATION
      local chains = forDerivatives(eq, rowsOf[i], view, p, seeds, planSeeds, E, variables, functionsByName, known)
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
    #= on the element view's variables: a state's guess is the start value of the element it selects =#
    local (vs, eqs) = dynamicSetParts(viewVariables, p.structure, set, k, p.carried.values)
    for v in vs
      local guess::Union{Nothing, IR.Expression} = v.attributes.start
      guess === nothing && continue
      v.attributes = IR.VariableAttributes(arrayReferences(guess, E, variables), v.attributes.fixed, nothing, nothing, nothing,
                                           nothing, nothing, nothing, nothing)
    end
    append!(newVariables, vs)
    #= its pivots read the element view's variables: references into their arrays =#
    append!(equations, IR.Equation[arrayEquation(eq, E, variables) for eq in eqs])
  end
  for (f, family) in enumerate(families)
    local (vs, eqs) = familyParts(family, length(scalarSets) + f, p.structure, E, variables, viewVariables,
                                  p.carried.values)
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
  #= der() of an expression by the chain rule in array form, as the plan read it (`expandDerivatives!` of `plan` and
     `commit!`), before the equations are differentiated =#
  local seeds::Seeds = arraySeeds(s, derivatives, E, variables, model.functions, p.constants)
  local expanded::Vector{IR.Equation} =
    IR.Equation[readsDerCall(eq) ? derivativesExpanded(eq, seeds) : eq for eq in partition.equations]
  local expandedPartition::IR.Partition =
    IR.Partition(partition.index, partition.association, partition.unknowns, expanded, partition.initialEquations)
  local (links, linkEquations) = stateLinks(s, p.dummies, E, derivatives, variables)
  E = extendedElements(E, links, variables)
  local kinds = arrayKinds(plannedKinds(s, p.dummies), E, variables)
  local equations = differentiatedArrayEquations(expandedPartition, view, p, E, variables, derivatives, model.functions)
  append!(equations, linkEquations)
  local (setVariables, setEquations, sets) = stateSets(model, p, E, variables, view.model.variables)
  append!(equations, setEquations)
  return ArrayCommit(vcat(added, links, setVariables), kinds, equations, sets, expanded)
end

"Whether the equation reads der() of an expression (a call introduceDerivatives left)."
function readsDerCall(eq::IR.Equation)::Bool
  local found = Ref(false)
  IR.foreachExp(e -> (IR.isBuiltinCall(e, :der) && (found[] = true); nothing), eq)
  return found[]
end

"""
A copy of the equation `eq` with der() of an expression replaced by its time derivative in array form (`seeds`:
`arraySeeds`), as `expandDerivatives!` replaces it on the element view: MSL FluxTubes' Tellinen hysteresis
`dHyst = der(hystR - mu0 * Hstat)` is `dHyst = \$DER.hystR - mu0 * \$DER.Hstat`.
"""
function derivativesExpanded(eq::IR.Equation, seeds::Seeds)::IR.Equation
  local expand::Function = (@nospecialize(e::IR.Expression)) ->
    IR.isBuiltinCall(e, :der) ? differentiate(only((e::IR.CALL).arguments), seeds) : e
  return IR.mapEquation!(expand, deepcopy(eq))
end

"""
    commitArrays!(model, partition, c)

The array form of a plan applied (`arrayCommit`): the new variables added, the kinds set, the equations appended.
"""
function commitArrays!(model::IR.Model, partition::IR.Partition, c::ArrayCommit)::Nothing
  partition.equations = c.expanded
  foreach(v -> push!(model.variables, v), c.variables)
  for (name, kind) in c.kinds
    model.variables[name].kind = kind
  end
  append!(partition.equations, c.equations)
  partition.initialEquations =
    IR.Equation[initialDummyDerivatives(eq, model.variables) for eq in partition.initialEquations]
  return nothing
end

"""
The initial equation `eq` with der() of a dummy state (DUMMY_STATE: index reduction's, or one of the variables the
initial equations read der() of, `initialDerivatives!`) its derivative variable: `der(v) = 0` reads `\$DER.v`, an
element `der(x[2])` the element `\$DER.x[2]`; `eq` itself where it reads none.
"""
function initialDummyDerivatives(eq::IR.Equation, variables::IR.VariableSet)::IR.Equation
  local isDummy::Function = (@nospecialize(x::IR.Expression)) -> begin
    IR.isBuiltinCall(x, :der) || return false
    local a::IR.Expression = only((x::IR.CALL).arguments)
    a isa IR.CREF || return false
    #= a variable of its own (`x[2]` of a model of scalars), else the element of an array =#
    local v::Union{Nothing, IR.Variable} = get(variables, a.cref.name, nothing)
    v === nothing && (v = get(variables, IR.withoutSubscripts(a.cref).name, nothing))
    return v !== nothing && v.kind isa IR.DUMMY_STATE
  end
  local found::Bool = false
  IR.foreachRootExp(eq) do @nospecialize(e::IR.Expression)
    found || (found = IR.anyExp(isDummy, e))
    nothing
  end
  found || return eq
  local replace::Function = (@nospecialize(x::IR.Expression)) ->
    isDummy(x) ? IR.CREF(IR.derivativeCref(only((x::IR.CALL).arguments).cref), IR.T_REAL()) : x
  return IR.mapEquation!(replace, deepcopy(eq))
end

"""
    exposeArrayLinearConstraints!(model, view) -> Bool

`exposeLinearConstraints!` for an arrays-kept model: the hidden linear constraints of its element view `view`
(`linearConstraints`) written into `model`, each in place of the equation it combines, where that one is a scalar
equation of `model` (the constraint's elements as references into their arrays: `arrayEquation`). Whether one was
written. A star point of a polyphase winding that is not grounded: its current balance and a zero-sequence inductor's
`i0 = sum(i) / m` give `i0 = 0`; as written, of index 1 with i0 a state, the algebraic loop of the winding's voltages
was singular (MSL FundamentalWave's polyphase machines: "the initialization cannot hold"). A constraint in place of
an element of a for- or array equation is left: one element of an array equation cannot be replaced in array form.
"""
function exposeArrayLinearConstraints!(model::IR.Model, view::Scalarized)::Bool
  envSwitch("OMBACKEND_LINEAR_CONSTRAINTS") || return false
  local equations::Vector{IR.Equation} = model.continuous[1].equations
  local written::Bool = false
  for (i::Int, constraint::IR.SCALAR_EQUATION) in linearConstraints(view.model, view.model.continuous[1])
    local origin::EquationOrigin = view.origins[i]
    (isempty(origin.path) && isempty(origin.element) && equations[origin.equation] isa IR.SCALAR_EQUATION) || continue
    equations[origin.equation] = arrayEquation(constraint, view.elements, model.variables)
    written = true
  end
  return written
end

"""
    reduceIndexArrays!(model) -> Bool

Index reduction of an arrays-kept model in array form: planned on its element view (`scalarizeArrays`, `knownPlan`),
committed in array form (`arrayCommit`, `commitArrays!`); true where reduced. A model of index 1 is committed with
the plan that differentiates nothing (its states, their derivative variables and links in array form, as the array
code reads them): false. With `arrays` (the default) false, the model not changed, where it has no continuous
arrays; without, a model of scalars is reduced the same way (a scalar is the array form's variable of no
dimensions: MSL Rotational ElasticBearing, of index 3 and no array, for the array code). An array whose elements the
plan treats differently (slicing) is split into its elements first (`splitArrays!`), and the plan committed again.
NotHandled where the array form does not take the plan (a split down to no array equations, der() of an expression
whose derivative it does not write: the caller takes the scalarized path and discards the model, which a split may
have changed).
"""
function reduceIndexArrays!(model::IR.Model; arrays::Bool = true)::Bool
  (arrays && !hasContinuousArrays(model)) && return false
  #= the element equations of the equations the steps below keep (`ScalarCache`): each step replaces the equations
     it changes; the views share the others' (MSL EngineV6: 5 splits, re-scalarized in 0.27 s instead of 1.0 s) =#
  local cache::ScalarCache = ScalarCache()
  local view::Scalarized = scalarizeArrays(model; cache)
  #= the equations as the plan's structure reads them (the terms of parameters known as zero) =#
  dropKnownZeroTerms!(model, view) && (view = scalarizeArrays(model; cache))
  exposeArrayLinearConstraints!(model, view) && (view = scalarizeArrays(model; cache))
  local partition::IR.Partition = view.model.continuous[1]
  #= the arrays of aliases read through representatives (`arrayAliasPlan`) where the plan differentiates anything
     (`differentiates`): MSL EngineV6 differentiates 4974 element rows instead of 7132, translated and first simulated
     5 s faster. The plan as written only where that does not apply (EngineV6: a plan fewer, 0.4 s). =#
  local aliased::Union{Nothing, Plan} = nothing
  if envSwitch("OMBACKEND_ALIAS_SUBSTITUTION") && differentiates(view.model, partition; initialDerivatives = true)
    (view, aliased) = arrayAliasPlan(model, view, cache)
    partition = view.model.continuous[1]
  end
  local written::Plan = aliased !== nothing ? aliased : plan(view.model, partition; indexOne = true, initialDerivatives = true)
  #= the array form of the plan as written first: a plan it does not take is refused before the simulation code's
     values are computed for the plan again (`knownPlan`). Arrays whose elements it treats differently (slicing) are
     split into their elements, until it takes it. The element view stays (the splits keep the model's meaning), so
     its plan is kept while splitting and made again on the final view, which the array form must take too (its
     variables' order may decide differently): MSL EngineV6, 5 rounds, planned each round, 3.6 s of 7.6. The plan made
     again is the one committed (`knownPlan`, the simulation code's values), with its commit (EngineV6: a plan and a
     commit fewer, 0.9 s). =#
  local splits::Int = 0
  local stale::Bool = false
  local final::Union{Nothing, Tuple{Scalarized, Plan, ArrayCommit}} = nothing
  local arrayElements::Int = sum((length(view.elements.elements[v.name]) for v in model.variables
                                  if isContinuousArray(v) && haskey(view.elements.elements, v.name)); init = 0)
  while true
    try
      arrayCommit(model, model.continuous[1], view, written)
      stale || break
      (view, written) = knownPlan(model, view, nothing)
      partition = view.model.continuous[1]
      stale = false
      final = (view, written, arrayCommit(model, model.continuous[1], view, written))
      break
    catch err
      (err isa NotHandled && needsSplit(err) && splits < 256) || rethrow()
      splitArrays!(model, view, err.arrays, err.equations; cache)
      splits += 1
      #= split down to no array equations, or most of the arrays' elements split: little the array form keeps, the
         scalarized path is the one (MSL EngineV6: its frames' vectors and rotations, 10877 elements split) =#
      local kept::Int = sum((length(view.elements.elements[v.name]) for v in model.variables
                             if isContinuousArray(v) && haskey(view.elements.elements, v.name)); init = 0)
      (hasArrayEquations(model) && 2 * kept >= arrayElements) ||
        throw(NotHandled("slicing that splits most of the arrays ($(arrayElements - kept) of $(arrayElements) elements, " *
                         "starting at $(err.reason))"))
      view = scalarizeArrays(model; cache)
      partition = view.model.continuous[1]
      stale = true
    end
  end
  final === nothing && return commitArrayPlan!(model, view, written)
  return commitPlan!(model, final...)
end

"""
    commitIndexOne!(model) -> Bool

A model without continuous arrays (on the arrays-kept route, for the array code) of index 1 committed as
`reduceIndexArrays!` commits one: the plan that differentiates nothing, its states, their derivative variables and
links. False, the model not changed, where its index is above one (the scalarized path reduces it).
"""
function commitIndexOne!(model::IR.Model)::Bool
  local view::Scalarized = scalarizeArrays(model)
  local written::Plan = plan(view.model, view.model.continuous[1]; indexOne = true, initialDerivatives = true)
  isempty(written.differentiated) || return false
  commitArrayPlan!(model, view, written)
  return true
end

"""
    applyCarriedArrays!(variables, E, carried) -> Nothing

`applyCarried!` for the arrays that declare no start: a state array with elements of a carried start, and an array
whose elements all have an alias that declares one, take the array of the carried values as their start (0 for an
element without one).
"""
function applyCarriedArrays!(variables::IR.VariableSet, E::Elements, carried::CarriedStarts)::Nothing
  for v in variables
    (v.ty isa IR.T_ARRAY && v.attributes.start === nothing && haskey(E.elements, v.name)) || continue
    local names::Vector{Symbol} = E.elements[v.name]
    (v.kind isa IR.STATE || all(in(carried.aliased), names)) || continue
    any(n::Symbol -> haskey(carried.values, n), names) || continue
    local leaves::Vector{IR.Expression} = IR.Expression[IR.REAL(get(carried.values, n, 0.0)) for n in names]
    v.attributes = withStart(v.attributes, nestedArray(E.sizes[v.name], IR.T_REAL(), leaves))
  end
  return nothing
end

"Whether index reduction reduced `model`: it has dummy derivatives (one for each differentiated equation)."
indexReduced(model::IR.Model)::Bool = any(v::IR.Variable -> v.kind isa Union{IR.DUMMY_DER, IR.DUMMY_STATE}, model.variables)

"""
    knownPlan(model, view, written) -> Tuple{Scalarized, Plan}

The plan of the element view `view` of `model` made with the simulation code's values (`KNOWN_VALUES`), on the model
as the plan with these values reads it: without the terms a function's zero output takes out (`model` changed, its
view made again). Without such values `written`, or where that is `nothing` the plan made again.
"""
function knownPlan(model::IR.Model, view::Scalarized, written::Union{Nothing, Plan})::Tuple{Scalarized, Plan}
  local known::Dict{Symbol, IR.Expression} =
    KNOWN_VALUES[] === nothing ? Dict{Symbol, IR.Expression}() : KNOWN_VALUES[](view.model)
  if isempty(known)
    written === nothing || return (view, written)
    return (view, plan(view.model, view.model.continuous[1]; indexOne = true, initialDerivatives = true)::Plan)
  end
  local v::Scalarized = dropFunctionZeroTerms!(model, view, known) ? scalarizeArrays(model) : view
  return (v, plan(v.model, v.model.continuous[1]; known, indexOne = true, initialDerivatives = true)::Plan)
end

"""
    commitArrayPlan!(model, view, written) -> Bool

The plan `written`, made on the element view `view` of `model` and taken by its array form, made again with the
simulation code's values (`knownPlan`) and committed (`commitPlan!`); whether it reduced the index.
"""
function commitArrayPlan!(model::IR.Model, view::Scalarized, written::Plan)::Bool
  local (v::Scalarized, p::Plan) = knownPlan(model, view, written)
  return commitPlan!(model, v, p, arrayCommit(model, model.continuous[1], v, p))
end

"""
    commitPlan!(model, view, p, c) -> Bool

The plan `p` of the element view `view` of `model` committed in array form (`c`, its `arrayCommit`; `commitArrays!`),
the derivative unknowns solved for; whether it reduced the index (differentiated equations).
"""
function commitPlan!(model::IR.Model, view::Scalarized, p::Plan, c::ArrayCommit)::Bool
  local reduced::Bool = !isempty(p.differentiated)
  commitArrays!(model, model.continuous[1], c)
  applyCarried!(model.variables, p.carried)
  applyCarriedArrays!(model.variables, view.elements, p.carried)
  #= the derivative unknowns solved for, as on the scalarized path (`reduceIndex!`): left algebraic unknowns, MSL
     R134a2's der(p) of its volumes jumped at the dew line and the steps fell to 1e-16 =#
  #= a coefficient of parameters stays an expression (the array code's parameters can be overridden: `c * der(x) = 1`
     with c overridden kept c's translated value) =#
  local solved = envSwitch("OMBACKEND_SOLVE_DERIVATIVES") ?
    solveDerivatives!(model, model.continuous[1]; literalCoefficients = false) : Symbol[]
  local states = sort!([String(n) for (n, k) in c.kinds if k isa IR.STATE])
  local stateText = isempty(states) ? "" : ", states $(join(first(states, 8), ", "))"
  local setText = c.sets == 0 ? "" : ", $(c.sets) dynamic state set$(c.sets == 1 ? "" : "s")"
  local solvedText = isempty(solved) ? "" : "; $(length(solved)) derivatives solved for"
  @info "[index reduction] $(model.name): $(reduced ? "" : "of index 1, ")in array form, $(length(c.equations)) equations added$(stateText)$(setText)$(solvedText)"
  return reduced
end
