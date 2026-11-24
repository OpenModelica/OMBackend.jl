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
#= Arrays kept (the frontend's no-scalarize mode): the element view of a model's arrays, on which index reduction
   plans as on a scalarized model (the scalar graph of Pseudo Array Causalization: Abdelhak, Casella, Bachmann,
   Modelica Conference 2023), and the same scalarization of the reduced model for the code generators, which take
   scalars. =#

"""
    Elements

The element variables of the array variables of a model (`elements(variables)`): `elements[a]` the elements of
the array `a` in row-major order (the last subscript fastest, as the frontend scalarizes: `x[1,1]`, `x[1,2]`,
...), `sizes[a]` its dimensions, `array[e]` the array of the element `e` and its subscripts, `crefs[e]` the
element's reference as the scalarized model names it (`c[2].p.T` of the array `c.p.T`, whose dimension is the
record array's; `\$DER.c[2].T` of the backend's `\$DER.c.T`). `passesFollow`: the view is of a model as lowered, for
the scalar passes (`scalarizeArrays(model; passesFollow = true)`), and keeps what they take as the frontend's
scalarization does: a reference by a subscript known at run time only (`chooseRuntimeIndices!`), a record equation
of a call (`fieldEquations`). The view the array form's index reduction plans on does not. `structural[p]` the value
of the structural Integer parameter `p` (`structuralValues`): a range over it (`1:N`) has its translated value.
"""
struct Elements
  elements::Dict{Symbol, Vector{Symbol}}
  sizes::Dict{Symbol, Vector{Int}}
  array::Dict{Symbol, Tuple{Symbol, Vector{Int}}}
  crefs::Dict{Symbol, IR.ComponentRef}
  structural::Dict{Symbol, Int}
  passesFollow::Bool
end

"""
    EquationOrigin(equation, path, iteration, element)

Where an equation of a scalarized partition comes from: the `equation`-th equation of the partition with arrays,
inside it the positions `path` in the bodies of nested for-equations (outermost first; empty outside one), at the
iterator values `iteration`; `element` the element of an array equation (empty for a scalar one). The equations
of an if-equation's branch taken whatever the time have the if-equation's origin.
"""
struct EquationOrigin
  equation::Int
  path::Vector{Int}
  iteration::Vector{Int}
  element::Vector{Int}
end

"The origin of a scalar equation outside a for-equation, the `i`-th of its partition."
EquationOrigin(i::Int) = EquationOrigin(i, Int[], Int[], Int[])

"""
    Scalarized(model, origins, elements)

A model with arrays as its scalarized form (`scalarizeArrays`): the model of the element variables and the
partition's equations one per element (for-equations unrolled, array equations per element), `origins[j]` the
origin of the `j`-th equation of its continuous partition, `elements` the arrays' elements.
"""
struct Scalarized
  model::IR.Model
  origins::Vector{EquationOrigin}
  elements::Elements
end

"No iterator values: an expression outside a for-equation."
const NO_ITERATORS = Dict{Symbol, Int}()

"No names bound by for-statements (`scalarExpression`'s `bound`); never changed."
const NO_BOUND_NAMES = Set{Symbol}()

"Whether `x` is a reference to the iterator of an enclosing for-statement, one of `bound` (one part, no subscripts)."
isBoundIterator(x::IR.CREF, bound::Set{Symbol})::Bool =
  !isempty(bound) && length(x.cref.parts) == 1 && isempty(x.cref.parts[1].subscripts) && x.cref.parts[1].name in bound

"The dimensions of an array type; NotHandled for a dimension of a size not known (`what` names it)."
function knownSizes(@nospecialize(ty::IR.ModelicaType), what::String)::Vector{Int}
  local sizes = Int[]
  for d in IR.dimensions(ty)
    local n = IR.dimensionSize(d)
    n === nothing && throw(NotHandled("$(what) of a dimension whose size is not known"))
    push!(sizes, n)
  end
  return sizes
end

"The position of the subscripts `index` in an array of `sizes`, row-major."
function rowMajorIndex(index::Vector{Int}, sizes::Vector{Int})::Int
  local k = 0
  for (i, n) in zip(index, sizes)
    k = k * n + (i - 1)
  end
  return k + 1
end

"""
    derivedFrom(v, variables) -> Union{Nothing, Tuple{Vector{IR.CrefPart}, Symbol}}

Of a derivative array (`\$DER.c.T`, `\$STATEDER.x`: `IR.isDerivativeCref`), its operator part and the array it is the
derivative of (`\$DER`, `c.T`: `IR.operand`); of one whose operand is no variable (a derivative the array code's form
dropped, `Passes.dropUnreadDerivatives!`), the operator parts, outermost first, down to the first operand that is
(`\$DER`, `\$DER`, `x` of `\$DER.\$DER.x`); `nothing` for another variable.
"""
function derivedFrom(v::IR.Variable, variables::IR.VariableSet)::Union{Nothing, Tuple{Vector{IR.CrefPart}, Symbol}}
  local operators::Vector{IR.CrefPart} = IR.CrefPart[]
  local cref::IR.ComponentRef = v.cref
  while IR.isDerivativeCref(cref)
    push!(operators, cref.parts[1])
    cref = IR.operand(cref)::IR.ComponentRef
    local of::Union{Nothing, IR.Variable} = get(variables, cref.name, nothing)
    of === nothing && continue
    return of.ty isa IR.T_ARRAY ? (operators, of.name) : nothing
  end
  return nothing
end

"""
    modelElementCref(v, index) -> ComponentRef

The reference of the element `index` of the array variable `v` as the scalarized model names it: the subscripts
on the parts whose types are arrays (`c[2].p.T` of `c.p.T`, a field of the record array `c`), each the index of its
dimension (`IR.dimensionIndex`: the literal of an enumeration's, `t[Logic.'0']`).
"""
function modelElementCref(v::IR.Variable, index::Vector{Int})::IR.ComponentRef
  local dims::Vector{IR.Dimension} = IR.dimensions(v.ty)
  return distributedCref(v, IR.Subscript[IR.INDEX(IR.dimensionIndex(dims[d], k)) for (d, k) in enumerate(index)])
end

"The elements `crefs` (row-major) of the array `v` of `sizes` into `E`."
function addElements!(E::Elements, v::IR.Variable, sizes::Vector{Int}, crefs::Vector{IR.ComponentRef})::Nothing
  local names = Vector{Symbol}(undef, length(crefs))
  for (k, (index, cref)) in enumerate(zip(rowMajor(sizes), crefs))
    haskey(E.array, cref.name) && throw(NotHandled("the element $(cref.name) of two arrays"))
    names[k] = cref.name
    E.array[cref.name] = (v.name, index)
    E.crefs[cref.name] = cref
  end
  E.elements[v.name] = names
  E.sizes[v.name] = sizes
  return nothing
end

"""
    elements(variables) -> Elements

The element variables of the array variables of `variables`, by their scalarized names (`Elements`): a model
array's by its reference (`modelElementCref`), a derivative array's (`\$DER.x`) by its operator part before the
elements of the array it is the derivative of (`\$DER.x[2]`, `\$DER.c[2].T`), after that array's. Not an array whose elements are variables
already (a record field array of a parameter, `expandRecordFieldArrays`). `passesFollow` as `Elements` has it.
"""
function elements(variables::IR.VariableSet; passesFollow::Bool = false)::Elements
  local E = Elements(Dict{Symbol, Vector{Symbol}}(), Dict{Symbol, Vector{Int}}(), Dict{Symbol, Tuple{Symbol, Vector{Int}}}(),
                     Dict{Symbol, IR.ComponentRef}(), structuralValues(variables), passesFollow)
  local derived::Vector{Tuple{IR.Variable, Vector{IR.CrefPart}, Symbol}} = Tuple{IR.Variable, Vector{IR.CrefPart}, Symbol}[]
  #= the arrays of rolled components first (their elements may be another array's, kept beside them: rollComponents!) =#
  local rolledFirst::Vector{IR.Variable} = IR.Variable[v for v in variables if IR.isRolled(v.cref.parts[1].ty)]
  append!(rolledFirst, IR.Variable[v for v in variables if !IR.isRolled(v.cref.parts[1].ty)])
  for v in rolledFirst
    v.ty isa IR.T_ARRAY || continue
    local from = derivedFrom(v, variables)
    if from === nothing
      local sizes = knownSizes(v.ty, "the array $(v.name)")
      local crefs = IR.ComponentRef[modelElementCref(v, index) for index in rowMajor(sizes)]
      #= a record field array of a parameter or a constant: its elements are variables already, the array kept beside
         them for a call's whole argument (expandRecordFieldArrays), as in the scalarized model; or elements of a rolled
         array (an instance's array kept beside it) =#
      local existing = count(c -> haskey(variables, c.name) || haskey(E.array, c.name), crefs)
      existing == length(crefs) && continue
      existing == 0 || throw(NotHandled("the array $(v.name), $(existing) of its $(length(crefs)) elements variables of their own"))
      addElements!(E, v, sizes, crefs)
    else
      push!(derived, (v, from[1], from[2]))
    end
  end
  addDerivedArrays!(E, derived)
  return E
end

"""
    structuralValues(variables) -> Dict{Symbol, Int}

The values of the structural Integer parameters of `variables` (`IR.PARAMETER(true)`: the frontend evaluated them,
the generated code holds their translated values) whose bindings fold to an Integer, also through other structural
parameters (`M = 2 * N`).
"""
function structuralValues(variables::IR.VariableSet)::Dict{Symbol, Int}
  local values::Dict{Symbol, Int} = Dict{Symbol, Int}()
  local todo::Vector{IR.Variable} = IR.Variable[v for v in variables
                                               if v.kind isa IR.PARAMETER && v.kind.structural && v.binding !== nothing]
  local progress::Bool = true
  while progress && !isempty(todo)
    progress = false
    for k in length(todo):-1:1
      local value::Union{Nothing, Int} = foldInteger(withStructural(todo[k].binding, values))
      value === nothing && continue
      values[todo[k].name] = value
      deleteat!(todo, k)
      progress = true
    end
  end
  return values
end

"`e` with the references to the structural parameters of `values` (`structuralValues`) replaced by their values."
function withStructural(e::IR.Expression, values::Dict{Symbol, Int})::IR.Expression
  isempty(values) && return e
  local replace = (@nospecialize(x::IR.Expression)) -> begin
    (x isa IR.CREF && !IR.hasSubscripts(x.cref) && haskey(values, x.cref.name)) || return x
    return IR.INTEGER(values[x.cref.name])
  end
  return IR.mapExp(replace, e)
end

"""
    iterationRange(range, values, E) -> Union{Nothing, StepRange{Int, Int}}

The values of a loop's `range` at the iterators `values`: literal once the iterators and the structural parameters
(`E.structural`) are replaced (`1:N`); `nothing` where it is not.
"""
iterationRange(range::IR.Expression, values::Dict{Symbol, Int}, E::Elements)::Union{Nothing, StepRange{Int, Int}} =
  literalRange(withStructural(scalarExpression(range, values, E), E.structural))

"""
    addDerivedElements!(E, v, operators, of)

The elements of the derivative array `v` (`\$DER.x`) into `E`: its operator parts `operators` (`derivedFrom`) before
the references of the elements of the array `of` it is the derivative of (`\$DER.x[2]`), which `E` has.
"""
function addDerivedElements!(E::Elements, v::IR.Variable, operators::Vector{IR.CrefPart}, of::Symbol)::Nothing
  local sizes = knownSizes(v.ty, "the array $(v.name)")
  sizes == E.sizes[of] || throw(NotHandled("the array $(v.name) of other dimensions than $(of)'s"))
  local operated = (c::IR.ComponentRef) -> foldr(IR.withOperator, operators; init = c)
  addElements!(E, v, sizes, IR.ComponentRef[operated(E.crefs[e]) for e in E.elements[of]])
  return nothing
end

"The derivative arrays `derived` (each with its operator part and the array it is the derivative of) into `E`, each after the array it is the derivative of (`\$DER.\$DER.x` after `\$DER.x`)."
function addDerivedArrays!(E::Elements, derived::Vector{Tuple{IR.Variable, Vector{IR.CrefPart}, Symbol}})::Nothing
  while !isempty(derived)
    local ready = findfirst(d -> haskey(E.elements, d[3]), derived)
    ready === nothing && throw(NotHandled("the arrays $(join((d[1].name for d in derived), ", ")), made of arrays not known"))
    local (v::IR.Variable, operators::Vector{IR.CrefPart}, of::Symbol) = popat!(derived, ready)
    addDerivedElements!(E, v, operators, of)
  end
  return nothing
end

"""
    extendedElements(E, added, variables) -> Elements

`E` with the elements of the backend-made arrays `added` (of `variables`, which has them and the arrays they are
made of), a new `Elements`; `E` not changed. NotHandled for an added array that is no backend-made one.
"""
function extendedElements(E::Elements, added::Vector{IR.Variable}, variables::IR.VariableSet)::Elements
  local out = Elements(copy(E.elements), copy(E.sizes), copy(E.array), copy(E.crefs), E.structural, E.passesFollow)
  local derived::Vector{Tuple{IR.Variable, Vector{IR.CrefPart}, Symbol}} = Tuple{IR.Variable, Vector{IR.CrefPart}, Symbol}[]
  for v in added
    v.ty isa IR.T_ARRAY || continue
    local from = derivedFrom(v, variables)
    from === nothing && throw(NotHandled("the new array $(v.name), not made of an array"))
    push!(derived, (v, from[1], from[2]))
  end
  addDerivedArrays!(out, derived)
  return out
end

"""
    selectedIndices(s, n) -> Union{Nothing, Int, Vector{Int}}

The indices a subscript selects of a dimension of size `n`: one (`x[2]`; `x[1 + 1]`, `x[(1:2)[1]]`, `foldInteger`),
several (`x[2:3]`, `x[{1, 3}]`, `x[:]`); `nothing` where they are not literals (an iterator, a parameter).
"""
function selectedIndices(s::IR.INDEX, ::Int)::Union{Nothing, Int, Vector{Int}}
  local e::IR.Expression = s.exp
  e isa IR.ENUM_LITERAL && return e.index   # a dimension of an enumeration: the literal's position
  e isa IR.BOOLEAN && return e.value ? 2 : 1
  return foldInteger(e)
end
selectedIndices(::IR.WHOLE, n::Int)::Union{Nothing, Int, Vector{Int}} = collect(1:n)
function selectedIndices(s::IR.SLICE, ::Int)::Union{Nothing, Int, Vector{Int}}
  local range = literalRange(s.exp)
  range === nothing || return collect(range)
  s.exp isa IR.ARRAY && all(e -> e isa IR.INTEGER, s.exp.elements) && return Int[e.value for e in s.exp.elements]
  return nothing
end

"""
    selectedElements(array, index, selection, E, element) -> Expression

The elements of `array` the per-dimension `selection` (an index or indices, `selectedIndices`) picks, after the
subscripts `index` so far: the reference to an element variable, an array literal of them (nested by the
dimensions selected by several indices).
"""
function selectedElements(array::Symbol, index::Vector{Int}, selection::Vector{Union{Int, Vector{Int}}}, E::Elements,
                          @nospecialize(element::IR.ModelicaType))::IR.Expression
  local d = length(index) + 1
  if d > length(selection)
    local sizes = E.sizes[array]
    all(k -> 1 <= index[k] <= sizes[k], eachindex(index)) ||
      throw(NotHandled("the element $(array)$(index) outside the array's dimensions $(sizes)"))
    return IR.CREF(E.crefs[E.elements[array][rowMajorIndex(index, sizes)]], element)
  end
  local picked = selection[d]
  picked isa Int && return selectedElements(array, push!(copy(index), picked), selection, E, element)
  local dims = IR.Dimension[IR.DIM_SIZE(length(s)) for s in selection[d:end] if s isa Vector{Int}]
  return IR.ARRAY(IR.Expression[selectedElements(array, push!(copy(index), k), selection, E, element) for k in picked],
                  IR.T_ARRAY(element, dims))
end

"""
    elementReference(e, E) -> Expression

A reference to an array variable by its element variables: an element by its variable (`x[2]`; `c.p.T[2]`, as
the passes subscript a field of a record array, is `c[2].p.T`), a whole array or a part or slice of one (`x`,
`R[1]`, `x[2:3]`) by the array literal of them. A subscript that is not literal (an iterator of an algorithm's
loop) leaves the reference as it is; `danglingReference` finds those where they may not stay.
"""
function elementReference(e::IR.CREF, E::Elements)::IR.Expression
  local array = IR.hasSubscripts(e.cref) ? unsubscriptedName(e.cref) : e.cref.name
  local sizes = get(E.sizes, array, nothing)
  #= an array without elements has no variables: the empty array literal, as the frontend's scalarization writes it =#
  (sizes === nothing && isEmptyArrayType(e.ty)) && return IR.ARRAY(IR.Expression[], e.ty)
  sizes === nothing && return recordReference(e)
  local selection::Union{Nothing, Vector{Union{Int, Vector{Int}}}} = referenceSelection(e.cref, sizes)
  selection === nothing && return e
  return selectedElements(array, Int[], selection, E, IR.elementType(e.ty))
end

"Whether `ty` is an array type with a dimension of size 0."
isEmptyArrayType(ty::IR.ModelicaType)::Bool = ty isa IR.T_ARRAY && any(d -> IR.dimensionSize(d) == 0, ty.dimensions)

"""
    withoutEmptyTargets(e) -> Expression

A tuple of targets (the left-hand side of an equation or an assignment of a call with several outputs) with each
array without elements discarded (`_`), as the frontend's scalarization writes it; another expression as it is.
"""
function withoutEmptyTargets(e::IR.Expression)::IR.Expression
  (e isa IR.TUPLE && any(isEmptyArray, e.elements)) || return e
  return IR.TUPLE(IR.Expression[isEmptyArray(x) ? IR.WILD() : x for x in e.elements], e.ty)
end

isEmptyArray(e::IR.Expression)::Bool = e isa IR.ARRAY && isempty(e.elements)

"""
    referenceSelection(cref, sizes) -> Union{Nothing, Vector{Union{Int, Vector{Int}}}}

What the subscripts of `cref` (of all its parts, the leading dimensions: `allSubscripts`) select of an array of
`sizes`, per dimension (`selectedIndices`; a dimension without a subscript whole); `nothing` where one is not
literal.
"""
function referenceSelection(cref::IR.ComponentRef, sizes::Vector{Int})::Union{Nothing, Vector{Union{Int, Vector{Int}}}}
  local subscripts::Vector{IR.Subscript} = allSubscripts(cref)
  length(subscripts) > length(sizes) &&
    throw(NotHandled("the reference $(cref.name) of more subscripts than its array's dimensions"))
  local selection = Vector{Union{Int, Vector{Int}}}(undef, length(sizes))
  for d in eachindex(sizes)
    local picked::Union{Nothing, Int, Vector{Int}} =
      d <= length(subscripts) ? selectedIndices(subscripts[d], sizes[d]) : collect(1:sizes[d])
    picked === nothing && return nothing
    selection[d] = picked
  end
  return selection
end

"""
    recordSizes(cref) -> Union{Nothing, Vector{Int}}

The dimensions the types of the parts of `cref` have, in order (`[3]` of `s.pin.i`, `pin[3]`): those of the array
of records a reference to a record names. `nothing` where one is not known, or a part is not a plain array (a slice,
heterogeneous components).
"""
function recordSizes(cref::IR.ComponentRef)::Union{Nothing, Vector{Int}}
  local sizes::Vector{Int} = Int[]
  for p in cref.parts
    (p.ty isa IR.T_SLICE || p.ty isa IR.T_COMPONENTS) && return nothing
    for d in IR.dimensions(p.ty)
      local n::Union{Nothing, Int} = IR.dimensionSize(d)
      n === nothing && return nothing
      push!(sizes, n)
    end
  end
  return sizes
end

"""
The reference to the element `index` of the array of records `cref` names: the subscripts on the parts whose types
are arrays (`s.pin[2].i`).
"""
function recordElementCref(cref::IR.ComponentRef, index::Vector{Int})::IR.ComponentRef
  local parts::Vector{IR.CrefPart} = IR.CrefPart[]
  local k::Int = 0
  for p in cref.parts
    local n::Int = length(IR.dimensions(p.ty))
    push!(parts, IR.CrefPart(p.name, IR.Subscript[IR.INDEX(IR.INTEGER(i)) for i in index[k + 1:k + n]], p.ty))
    k += n
  end
  return IR.ComponentRef(parts)
end

"""
    recordReference(e) -> Expression

A reference to a record of an array (`s.pin.i` of `pin[3]`, `i` of `Complex i[m]`; the view has the arrays of its
scalar fields, `s.pin.i.re`, not one of the record): an element by its subscripts on the parts whose types are
arrays (`s.pin[2].i`, whose fields are the element variables), a whole array or a slice by the array literal of
them. Another reference, and one by a subscript that is not literal, as it is.
"""
function recordReference(e::IR.CREF)::IR.Expression
  local element::IR.ModelicaType = IR.elementType(e.ty)
  element isa IR.T_RECORD || return e
  local sizes::Union{Nothing, Vector{Int}} = recordSizes(e.cref)
  (sizes === nothing || isempty(sizes)) && return e
  local selection::Union{Nothing, Vector{Union{Int, Vector{Int}}}} = referenceSelection(e.cref, sizes)
  selection === nothing && return e
  return selectedRecords(e.cref, Int[], selection, sizes, element)
end

"The records of the array `cref` names that `selection` picks after `index`: `selectedElements` of record references."
function selectedRecords(cref::IR.ComponentRef, index::Vector{Int}, selection::Vector{Union{Int, Vector{Int}}},
                         sizes::Vector{Int}, element::IR.T_RECORD)::IR.Expression
  local d::Int = length(index) + 1
  if d > length(selection)
    all(k -> 1 <= index[k] <= sizes[k], eachindex(index)) ||
      throw(NotHandled("the element $(cref.name)$(index) outside the array's dimensions $(sizes)"))
    return IR.CREF(recordElementCref(cref, index), element)
  end
  local picked::Union{Int, Vector{Int}} = selection[d]
  picked isa Int && return selectedRecords(cref, push!(copy(index), picked), selection, sizes, element)
  local dims = IR.Dimension[IR.DIM_SIZE(length(s)) for s in selection[d:end] if s isa Vector{Int}]
  local elements::Vector{IR.Expression} =
    IR.Expression[selectedRecords(cref, push!(copy(index), k), selection, sizes, element) for k in picked]
  return IR.ARRAY(elements, IR.T_ARRAY(element, dims))
end

"""
    danglingReference(e, E) -> Union{Nothing, Symbol}

A reference in `e` to an array that the scalarized model has as elements only, by a subscript that is not literal
(`x[n]`, `n` a parameter): an equation may not keep one (an algorithm's loop may: `x[i]`). `nothing` if none.
"""
function danglingReference(e::IR.Expression, E::Elements)::Union{Nothing, Symbol}
  local found = Ref{Union{Nothing, Symbol}}(nothing)
  IR.foreachExp(e) do x
    (found[] === nothing && x isa IR.CREF && IR.hasSubscripts(x.cref) && IR.literalIndices(allSubscripts(x.cref)) === nothing &&
     haskey(E.sizes, unsubscriptedName(x.cref))) && (found[] = x.cref.name)
    nothing
  end
  return found[]
end

"""
    constructorElement(e, k) -> Union{Nothing, Expression}

The `k`-th element of an array constructor whose last iterator is over a literal range (`{f(i) for i in 1:n}`): its
body at that iterator's `k`-th value; of several iterators, the constructor of the others over that body (the last
iterator is the first dimension, MLS 10.4.1.2: `{f(i, j) for j in 1:2, i in 1:3}[1]` is `{f(1, j) for j in 1:2}`, a
row of a parameter matrix's binding, CDL's TimeTable). `nothing` for another constructor.
"""
function constructorElement(e::IR.ARRAY_CONSTRUCTOR, k::Int)::Union{Nothing, IR.Expression}
  isempty(e.iterators) && return nothing
  local values::Union{Nothing, StepRange{Int, Int}} = literalRange(e.iterators[end].range)
  (values === nothing || !(1 <= k <= length(values))) && return nothing
  local body::IR.Expression = withIterator(e.body, e.iterators[end].name, values[k])
  length(e.iterators) == 1 && return body
  length(e.ty.dimensions) == length(e.iterators) || return nothing
  return IR.ARRAY_CONSTRUCTOR(body, e.iterators[1:end - 1], IR.T_ARRAY(e.ty.elementType, e.ty.dimensions[2:end]))
end

"""
    literalRange(e) -> Union{Nothing, StepRange{Int, Int}}

The values of a range of Integer literals (`1:n - 1` once `n` is folded; `1:n` with a parameter `n` is not);
`nothing` for another expression. `rangeValues` without the vector.
"""
function literalRange(e::IR.Expression)::Union{Nothing, StepRange{Int, Int}}
  e isa IR.RANGE || return nothing
  local start = foldInteger(e.start)
  local stop = foldInteger(e.stop)
  local step = e.step === nothing ? 1 : foldInteger(e.step)
  (start === nothing || stop === nothing || step === nothing || step == 0) && return nothing
  return start:step:stop
end

"""
    pickElement(e, subscripts, ty, E) -> Expression

`e[subscripts]` for an expression `e` of the scalarized model and literal subscripts: picked out of an array value
(`(2 * {a, b})[1]` is `2 * a`, `selectElement`; `{f(i) for i in 1:n}[2]` is `f(2)`, a nested constructor by each
subscript in turn; `cos({a, b})[2]` is `cos(b)`, `der({a, b})[2]` is `der(b)`, `(not {a, b})[2]` is `not b`: an
element-wise operator, `isElementwise`, also where it is left inside the picked element); subscripted where it is not.
"""
function pickElement(@nospecialize(e::IR.Expression), subscripts::Vector{IR.Subscript}, @nospecialize(ty::IR.ModelicaType),
                     E::Elements)::IR.Expression
  local index::Union{Nothing, Vector{Int}} = IR.literalIndices(subscripts)
  if e isa IR.ARRAY_CONSTRUCTOR && index !== nothing
    local body = constructorElement(e, index[1])
    if body !== nothing
      local element = foldTree(scalarExpression(body, NO_ITERATORS, E))
      return length(subscripts) == 1 ? element : pickElement(element, subscripts[2:end], ty, E)
    end
  end
  if e isa IR.CALL && isElementwise(e)
    return IR.CALL(e.name, IR.Expression[pickElement(e.arguments[1], subscripts, ty, E)], ty, e.attributes)
  elseif e isa IR.LUNARY && isElementwise(e)
    return IR.LUNARY(e.op, pickElement(e.exp, subscripts, ty, E))
  elseif e isa IR.LBINARY && isElementwise(e)
    return IR.LBINARY(logicalOperand(e.exp1, subscripts, ty, E), e.op, logicalOperand(e.exp2, subscripts, ty, E))
  end
  local picked::IR.Expression = selectElement(e, subscripts, ty)
  #= an element-wise operator left subscripted inside the element (`(a + der(b))[2]`: `a[2] + der(b)[2]`) =#
  if IR.anyExp(isSubscriptedElementwise, picked)
    picked = IR.mapExp(x -> isSubscriptedElementwise(x) ? pickElement(x.exp, x.subscripts, x.ty, E) : x, picked)
  end
  return foldTree(picked)
end

"An operand of an element-wise `and` or `or`: its element where it is an array, the scalar otherwise."
logicalOperand(e::IR.Expression, subscripts::Vector{IR.Subscript}, ty::IR.ModelicaType, E::Elements)::IR.Expression =
  isArrayValued(e) ? pickElement(e, subscripts, ty, E) : e

"""
The built-in functions and operators of one argument applied element-wise to an array (MLS 12.4.6; `der`, `pre`,
`previous` and `noEvent` of an array are those of its elements, MLS 3.7.4).
"""
const ELEMENTWISE_BUILTINS = (:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :exp, :log, :log10, :sqrt,
                              :abs, :sign, :floor, :ceil, :der, :pre, :previous, :noEvent)

"""
    isElementwise(e) -> Bool

Whether `e` is an operator applied to an array element by element, so that an element of it is the operator of the
operand's element: a built-in of `ELEMENTWISE_BUILTINS` of one array argument, `not`, `and`, `or` of arrays.
"""
isElementwise(e::IR.CALL)::Bool =
  e.attributes.builtin && e.name in ELEMENTWISE_BUILTINS && length(e.arguments) == 1 && isArrayValued(e.arguments[1])
isElementwise(e::IR.LUNARY)::Bool = isArrayValued(e.exp)
isElementwise(e::IR.LBINARY)::Bool = isArrayValued(e.exp1) || isArrayValued(e.exp2)
isElementwise(::IR.Expression)::Bool = false

"Whether `e` is an array (a logical operator has no type of its own: by its operands)."
isArrayValued(e::IR.LUNARY)::Bool = isArrayValued(e.exp)
isArrayValued(e::IR.LBINARY)::Bool = isArrayValued(e.exp1) || isArrayValued(e.exp2)
isArrayValued(e::IR.Expression)::Bool = IR.typeOf(e) isa IR.T_ARRAY

"Whether `x` is an element-wise operator subscripted by literals (`pickElement` takes the operand's element)."
isSubscriptedElementwise(x::IR.Expression)::Bool =
  x isa IR.SUBSCRIPTED && isElementwise(x.exp) && IR.literalIndices(x.subscripts) !== nothing

"""
    selectedParts(e) -> Expression

`e` (its iterators replaced) with what its literals select taken before the rest is mapped: an element of an array
literal (a table of pivots, a binding), an element of a constructor (`{f(i) for i in 1:1000}[100]`: its body there,
the others not made), the branch of an if-expression whose condition is of literals (`if j == {1, 2, 3}[i] then a
else b` at an `i` and a `j`, as the frontend's scalarization takes it; the other branch may read an element its
array does not have, `x[i - 1]` at `i = 1`).
"""
selectedParts(e::IR.Expression)::IR.Expression = IR.mapExpTopDown(x -> (selectedPart(x), true), e)

"The part of `x` its literals select (`selectedParts`), in turn (an element that is an if-expression); `x` without."
function selectedPart(@nospecialize(x::IR.Expression))::IR.Expression
  while true
    local taken::IR.Expression = x
    if x isa IR.IF_EXPRESSION
      local known::Union{Nothing, Bool} = literalCondition(foldTree(selectedParts(x.condition)))
      known === nothing || (taken = known ? x.thenExp : x.elseExp)
    elseif x isa IR.SUBSCRIPTED && IR.literalIndices(x.subscripts) !== nothing
      if x.exp isa IR.ARRAY
        taken = selectElement(x.exp, x.subscripts, x.ty)
        #= an element the literal does not have: subscripted again, as it was =#
        (taken isa IR.SUBSCRIPTED && taken.exp === x.exp) && return x
      elseif x.exp isa IR.ARRAY_CONSTRUCTOR
        local body::Union{Nothing, IR.Expression} = constructorElement(x.exp, IR.literalIndex(x.subscripts[1]))
        body === nothing ||
          (taken = length(x.subscripts) == 1 ? body : IR.SUBSCRIPTED(body, x.subscripts[2:end], x.ty))
      end
    end
    taken === x && return x
    x = taken
  end
end

"""
    scalarExpression(e, values, E) -> Expression

`e` in the scalarized model: the iterators of `values` replaced by their values (`withIterators`), references to
arrays by their element variables (`elementReference`), an element picked out of an array value where its
subscripts are literals (`pickElement`). What literals select is taken first (`selectedParts`). A name in `bound` is
the iterator of a for-statement around `e`, which shadows a variable of that name (MLS 11.2.2.1): it stays.
"""
function scalarExpression(e::IR.Expression, values::Dict{Symbol, Int}, E::Elements;
                          bound::Set{Symbol} = NO_BOUND_NAMES)::IR.Expression
  local pruned::IR.Expression = selectedParts(withIterators(e, values))
  return IR.mapExp(pruned) do x
    x isa IR.CREF && return isBoundIterator(x, bound) ? x : elementReference(x, E)
    (x isa IR.SUBSCRIPTED && IR.literalIndices(x.subscripts) !== nothing) && return pickElement(x.exp, x.subscripts, x.ty, E)
    (x isa IR.BINARY && x.op == IR.SCALAR_PRODUCT) && return expandedScalarProduct(x, E)
    x isa IR.ARRAY_CONSTRUCTOR && return expandedConstructor(x, E)
    return x
  end
end

"""
    expandedConstructor(e, E) -> Expression

An array constructor of one iterator over a literal range as the array literal of its body at each of the
iterator's values, each scalarized (`{f(x[i]) for i in 1:3}` is `{f(x[1]), f(x[2]), f(x[3])}`, as the frontend's
scalarization writes it: the references by the iterator are to element variables); as it is for another one (a
range that is not constant, several iterators).
"""
function expandedConstructor(e::IR.ARRAY_CONSTRUCTOR, E::Elements)::IR.Expression
  length(e.iterators) == 1 || return e
  local range::Union{Nothing, StepRange{Int, Int}} = literalRange(e.iterators[1].range)
  range === nothing && return e
  local name::Symbol = e.iterators[1].name
  local elements::Vector{IR.Expression} =
    IR.Expression[scalarExpression(withIterator(e.body, name, k), NO_ITERATORS, E) for k in range]
  return IR.ARRAY(elements, e.ty)
end

"""
    expandedScalarProduct(e, E) -> Expression

A scalar product of two vectors of a known size in the scalarized model: the sum of the products of their elements
(`pickElement`), as the frontend's scalarization writes it (MultiBody's `tau = -frame_b.t * e`); as it is where the
size of neither is known (one's may be an expression of parameters, `bb[2:na] * x`: the other's holds).
"""
function expandedScalarProduct(e::IR.BINARY, E::Elements)::IR.Expression
  local ty1::IR.ModelicaType = IR.typeOf(e.exp1)
  local size::Union{Nothing, Int} = vectorSize(ty1)
  size === nothing && (size = vectorSize(IR.typeOf(e.exp2)))
  size === nothing && return e
  local element::IR.ModelicaType = IR.elementType(ty1)
  local terms::Vector{IR.Expression} = IR.Expression[]
  for k in 1:size
    local index::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(IR.INTEGER(k))]
    push!(terms, times(pickElement(e.exp1, index, element, E), pickElement(e.exp2, index, element, E)))
  end
  isempty(terms) && return zeroOf(e)
  return foldTree(foldl(plus, terms))
end

"The size of a vector type where it is a literal; `nothing` for another type."
function vectorSize(ty::IR.ModelicaType)::Union{Nothing, Int}
  (ty isa IR.T_ARRAY && length(ty.dimensions) == 1) || return nothing
  return IR.dimensionSize(ty.dimensions[1])
end

"""
    equationExpression(e, values, E) -> Expression

`scalarExpression` of an expression of an equation, which may not keep a reference to an array by a subscript
that is not literal (`danglingReference`): NotHandled for one, unless the scalar passes follow (`E.passesFollow`).
"""
function equationExpression(e::IR.Expression, values::Dict{Symbol, Int}, E::Elements)::IR.Expression
  local scalar = scalarExpression(e, values, E)
  E.passesFollow && return scalar
  local dangling = danglingReference(scalar, E)
  dangling === nothing || throw(NotHandled("the reference $(dangling) to an array by a subscript that is not constant"))
  return scalar
end

"""
    ElementValue

An attribute or binding of an array variable, for its elements (`elementValue`, `valueAt`): `EachValue` the same
for every element, `ArrayValue` an array value picked per element (of a component array, per element of the field
in each component), `ComponentValue` a value reading the fields of the element's own component.
"""
abstract type ElementValue end

"The same value for every element (`each start = 0`, `each start = sum(m)`), scalarized once."
struct EachValue <: ElementValue
  value::IR.Expression
end

"""
    ArrayValue(value, ty, lead)

An array value, scalarized once (`scalarExpression`), its element picked per element: of the variable's dimensions
(`lead` 0), or of its trailing ones, the same for each component of a component array (`b = {1, 2}` of `b[2]` in
`f[3]`, as the frontend keeps the class's value; `lead` the component's dimensions before it, 1).
"""
struct ArrayValue <: ElementValue
  value::IR.Expression
  ty::IR.T_ARRAY
  lead::Int
end

"""
    ComponentValue(value, cref, ty, lead)

A value of a variable `cref` in a component array that reads fields of its own component (`p.L * cos(p.angle)` of
`p.x`, `bb[1] / a` of `d`: the class's value, its references typed as in the class, as the frontend keeps it for
`p[3]`): for each element the fields of the element's component (`componentElement`), then, of an array value (`ty`
of the field's own dimensions, after the `lead` of its components), the element's.
"""
struct ComponentValue <: ElementValue
  value::IR.Expression
  cref::IR.ComponentRef
  ty::IR.ModelicaType
  lead::Int
end

elementValue(::Nothing, ::IR.Variable, ::Elements)::Nothing = nothing

"""
    elementValue(a, v, E) -> ElementValue

The attribute or binding `a` of the array variable `v` for its elements: one reading fields of `v`'s own component
(`isComponentField`) a `ComponentValue`; another array value, of `v`'s dimensions or of its trailing ones (a field
array of a component array, the value of each component), an `ArrayValue`; another scalar one an `EachValue` (read
as written). A dimension of the value without a literal size (`fill(0.0, n)` typed `Real[n]`) is the variable's.
NotHandled for an array value of other dimensions.
"""
function elementValue(@nospecialize(a::IR.Expression), v::IR.Variable, E::Elements)::ElementValue
  local sizes = E.sizes[v.name]
  local ty = IR.typeOf(a)
  local dims::Vector{Int} = ty isa IR.T_ARRAY ? Int[something(IR.dimensionSize(d), -1) for d in IR.dimensions(ty)] : Int[]
  local lead::Int = length(sizes) - length(dims)
  if ty isa IR.T_ARRAY
    (lead >= 0 && all(k -> dims[k] == -1 || dims[k] == sizes[lead + k], eachindex(dims))) ||
      throw(NotHandled("an attribute of $(v.name) of dimensions $(dims), the variable's $(sizes)"))
  end
  IR.anyExp(x -> isComponentField(x, v.cref, sizes, E), a) && return ComponentValue(a, v.cref, ty, lead)
  local value::IR.Expression = scalarExpression(a, NO_ITERATORS, E)
  return ty isa IR.T_ARRAY ? ArrayValue(value, ty, lead) : EachValue(value)
end

"""
    componentDimensions(x, E) -> Int

The leading dimensions of the array `x` references that its type does not have and its subscripts do not index:
those of the component arrays it is a field of, in a value kept as the class has it (`p.L` typed `Real`, `p.w[1]`
of `Real w[2]`, in a value of a field of `p[3]`: 1). 0 for a reference typed with all its dimensions (`s` typed
`Real[3]` in `p(k = s)`), and for one to no array.
"""
function componentDimensions(x::IR.CREF, E::Elements)::Int
  local full::Union{Nothing, Vector{Int}} =
    get(E.sizes, IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name, nothing)
  full === nothing && IR.elementType(x.ty) isa IR.T_RECORD && (full = recordSizes(x.cref))
  full === nothing && return 0
  return length(full) - count(s -> s isa IR.INDEX, allSubscripts(x.cref)) - length(IR.dimensions(x.ty))
end

"""
    componentCref(cref, own, index, missing) -> ComponentRef

The reference `cref` of a value of the element `index` of the variable `own`, with the element's component named:
the component arrays both are fields of (the parts of the same names from the first, before `own`'s last), where
`cref` does not subscript them, subscripted by the element's indices of them (`p[2].L` for `p.L` in the value of
`p.x[2]`), up to `missing` dimensions (`componentDimensions`); the parts' types `own`'s (a value's reference may
lack those of the parts before its component). `cref` itself where it names none.
"""
componentCref(cref::IR.ComponentRef, own::IR.ComponentRef, index::Vector{Int}, missing::Int)::IR.ComponentRef =
  componentCref(cref, own, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index], missing)

"`componentCref` by the subscripts `index` of the element (the iterators of a for-equation over the elements)."
function componentCref(cref::IR.ComponentRef, own::IR.ComponentRef, index::Vector{IR.Subscript},
                       missing::Int)::IR.ComponentRef
  local parts::Vector{IR.CrefPart} = cref.parts
  local at::Int = 0
  for k in 1:min(length(cref.parts), length(own.parts) - 1)
    local p::IR.CrefPart = cref.parts[k]
    local q::IR.CrefPart = own.parts[k]
    p.name === q.name || break
    local n::Int = q.ty isa IR.T_ARRAY ? length(IR.dimensions(q.ty)) : 0
    if 0 < n <= missing && isempty(p.subscripts)
      parts === cref.parts && (parts = copy(cref.parts))
      parts[k] = IR.CrefPart(p.name, index[at + 1:at + n], q.ty)
      missing -= n
    end
    at += n
  end
  return parts === cref.parts ? cref : IR.ComponentRef(parts)
end

"""
Whether `x` reads a field of the component an element of the variable `own` (of dimensions `sizes`) is in: a
reference typed without its components' dimensions (`componentDimensions`) that shares them with `own`
(`componentCref`).
"""
function isComponentField(x::IR.Expression, own::IR.ComponentRef, sizes::Vector{Int}, E::Elements)::Bool
  x isa IR.CREF || return false
  local missing::Int = componentDimensions(x, E)
  return missing > 0 && componentCref(x.cref, own, sizes, missing) !== x.cref
end

"""
    valueAt(v, index, E) -> Union{Nothing, Expression}

The value for the element `index` (`ElementValue`); `nothing` for no value. NotHandled for one the element view
cannot scalarize (a reference by a subscript that is not literal, an element-wise built-in of an array left).
"""
valueAt(::Nothing, ::Vector{Int}, ::Elements)::Nothing = nothing
valueAt(v::EachValue, ::Vector{Int}, E::Elements)::IR.Expression = checkedValue(v.value, E)
valueAt(v::ArrayValue, index::Vector{Int}, E::Elements)::IR.Expression =
  checkedValue(pickElement(v.value, IR.Subscript[IR.INDEX(IR.INTEGER(index[k])) for k in (v.lead + 1):length(index)],
                           IR.elementType(v.ty), E), E)
valueAt(v::ComponentValue, index::Vector{Int}, E::Elements)::IR.Expression = checkedValue(componentElement(v, index, E), E)

"`value`, scalarized for an element; NotHandled where it is not (`valueAt`)."
function checkedValue(@nospecialize(value::IR.Expression), E::Elements)::IR.Expression
  local dangling = E.passesFollow ? nothing : danglingReference(value, E)
  dangling === nothing || throw(NotHandled("an attribute reading $(dangling) by a subscript that is not constant"))
  IR.anyExp(x -> x isa IR.CALL && isElementwise(x), value) &&
    throw(NotHandled("an attribute of an element-wise built-in of an array ($(IR.toString(value)))"))
  return value
end

"""
    componentElement(v, index, E) -> Expression

The component value `v` for the element `index`: each field of its component by that of the element's component
(`componentCref`), scalarized (`scalarExpression`), and of an array value the element's (`pickElement`).
"""
function componentElement(v::ComponentValue, index::Vector{Int}, E::Elements)::IR.Expression
  local named::IR.Expression = IR.mapExp(v.value) do x
    x isa IR.CREF || return x
    local missing::Int = componentDimensions(x, E)
    missing > 0 || return x
    return IR.CREF(componentCref(x.cref, v.cref, index, missing), x.ty)
  end
  local value::IR.Expression = scalarExpression(named, NO_ITERATORS, E)
  v.lead == length(index) && return value
  return pickElement(value, IR.Subscript[IR.INDEX(IR.INTEGER(index[k])) for k in (v.lead + 1):length(index)],
                     IR.elementType(v.ty), E)
end

"The element of the array `name` at `index` where `name` is an array of `E`; NotHandled for a scalar."
function elementName(name::Symbol, index::Vector{Int}, E::Elements)::Symbol
  haskey(E.elements, name) || throw(NotHandled("an array variable whose kind names the scalar $(name)"))
  return E.elements[name][rowMajorIndex(index, E.sizes[name])]
end

"""
    elementKind(kind, index, E) -> VariableKind

The kind of the element `index` of an array of kind `kind`: the variables it names (a state's derivative, a
derivative's state) by their elements at `index`.
"""
elementKind(kind::IR.VariableKind, index::Vector{Int}, E::Elements)::IR.VariableKind =
  IR.mapKindNames((x::Symbol) -> elementName(x, index, E), kind)

"An optional expression scalarized (`scalarExpression`)."
scalarOptional(::Nothing, ::Elements)::Nothing = nothing
scalarOptional(e::IR.Expression, E::Elements)::IR.Expression = scalarExpression(e, NO_ITERATORS, E)

"""
    scalarVariables(variables, E) -> VariableSet

The variables with each array variable as its elements, in its place: of the element's type, kind
(`elementKind`), binding and attributes (`ElementValue`, scalarized once per array); the scalar variables, and the
arrays whose elements are variables already, as new variables of the same fields (their bindings and attributes
reading arrays by the elements).
"""
function scalarVariables(variables::IR.VariableSet, E::Elements)::IR.VariableSet
  local out = IR.VariableSet()
  for v in variables
    local a = v.attributes
    #= a scalar, or an array whose elements are variables already (`elements`): itself =#
    if !haskey(E.elements, v.name)
      local attributes = IR.VariableAttributes(scalarOptional(a.start, E), scalarOptional(a.fixed, E), scalarOptional(a.nominal, E),
                                               scalarOptional(a.min, E), scalarOptional(a.max, E), a.unit, a.displayUnit,
                                               a.quantity, a.stateSelect)
      push!(out, IR.Variable(v.cref, v.ty, v.kind, scalarOptional(v.binding, E), attributes, v.direction, v.isProtected,
                             v.connector, v.source, v.origin))
      continue
    end
    local element = IR.elementType(v.ty)
    #= external objects (a table of each of an array of components: the array code makes one per element,
       `_objectElementRead`); clocks: variables of the scalar passes only (the array form has no element of them) =#
    element isa Union{IR.ScalarType, IR.T_STRING} ||
      element isa IR.T_EXTERNAL_OBJECT || (E.passesFollow && element isa IR.T_CLOCK) ||
      throw(NotHandled("the array $(v.name) of $(element) elements"))
    local (binding, start, fixed, nominal, low, high) =
      (elementValue(v.binding, v, E), elementValue(a.start, v, E), elementValue(a.fixed, v, E), elementValue(a.nominal, v, E),
       elementValue(a.min, v, E), elementValue(a.max, v, E))
    for name in E.elements[v.name]
      local index = E.array[name][2]
      local attributes = IR.VariableAttributes(valueAt(start, index, E), valueAt(fixed, index, E), valueAt(nominal, index, E),
                                               valueAt(low, index, E), valueAt(high, index, E), a.unit, a.displayUnit,
                                               a.quantity, a.stateSelect)
      push!(out, IR.Variable(E.crefs[name], element, elementKind(v.kind, index, E), valueAt(binding, index, E), attributes,
                             v.direction, v.isProtected, v.connector, v.source, v.origin))
    end
  end
  return out
end

"An equation's attributes `a` without the link to its derivative (an equation of the array model)."
unlinked(a::IR.EquationAttributes)::IR.EquationAttributes = IR.EquationAttributes(a.kind, a.differentiated, nothing, a.derivativeLink)

"""
    scalarEquations!(out, origins, eq, origin, values, E)

The equation `eq` (at the iterator `values`) as scalar equations into `out`, each with its `origin` into
`origins`: a for-equation unrolled over its literal range, an if-equation of literal conditions as its branch
taken, an array equation one per element; references by the element variables (`equationExpression`; statements
by `scalarExpression`). NotHandled for a for-equation over a range that is not constant and a record equation.
"""
function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.FOR_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local range::Union{Nothing, StepRange{Int, Int}} = iterationRange(eq.iterator.range, values, E)
  range === nothing && throw(NotHandled("a for-equation over a range that is not constant ($(IR.toString(eq.iterator.range)))"))
  local inner = copy(values)
  local paths = Vector{Int}[vcat(origin.path, b) for b in eachindex(eq.body)]
  for k in range
    inner[eq.iterator.name] = k
    local iteration = vcat(origin.iteration, k)
    for (b, body) in enumerate(eq.body)
      scalarEquations!(out, origins, body, EquationOrigin(origin.equation, paths[b], iteration, Int[]), inner, E)
    end
  end
  return nothing
end

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.SCALAR_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local lhs::IR.Expression = withoutEmptyTargets(equationExpression(eq.lhs, values, E))
  push!(out, IR.SCALAR_EQUATION(lhs, equationExpression(eq.rhs, values, E), eq.ty, eq.source, unlinked(eq.attributes)))
  push!(origins, origin)
  return nothing
end

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.ARRAY_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local (lhs, rhs) = (equationExpression(eq.lhs, values, E), equationExpression(eq.rhs, values, E))
  #= As lowered, an array equation is `rhs = lhs` of its source (LowerFlatModel's `lowerEquation`); its elements for
     the scalar passes are `lhs[i] = rhs[i]` again, as the frontend's scalarization writes them: the initialization
     takes `pre(x) = value` as the start of `pre(x)` in that orientation only (MSL Noise's `pre(state) =
     initialState(...)`: with the sides swapped the model did not build, "DAE init: residual 2e9") =#
  E.passesFollow && ((lhs, rhs) = (rhs, lhs))
  local element = IR.elementType(eq.ty)
  local indices::Vector{Vector{Int}} = rowMajor(knownSizes(eq.ty, "an array equation"))
  #= and in the frontend's order, the last element first: the order of the equations decides the alias roots and the
     states the passes choose (MSL MultiBody PrismaticConstraint: other states, and a build the frontend's order
     does not need) =#
  E.passesFollow && reverse!(indices)
  for index in indices
    local subscripts = IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index]
    local (l, r) = (pickElement(lhs, subscripts, element, E), pickElement(rhs, subscripts, element, E))
    local at = EquationOrigin(origin.equation, origin.path, origin.iteration, index)
    if element isa IR.T_RECORD   # an element of an array of records: the record equation of the two elements
      local fieldwise::IR.RECORD_EQUATION =
        IR.RECORD_EQUATION(l, r, element, length(element.fields), eq.source, eq.attributes)
      scalarEquations!(out, origins, fieldwise, at, NO_ITERATORS, E)
      continue
    end
    push!(out, IR.SCALAR_EQUATION(l, r, element, eq.source, unlinked(eq.attributes)))
    push!(origins, at)
  end
  return nothing
end

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.IF_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local branches = IR.EquationBranch[]
  local elseEquations = eq.elseEquations
  for b in eq.branches
    local condition = foldTree(equationExpression(b.condition, values, E))
    local known = literalCondition(condition)
    known === false && continue
    #= a branch taken whatever the time ends the if-equation: the else of the branches before it =#
    known === true && (elseEquations = b.equations; break)
    push!(branches, IR.EquationBranch(condition, scalarBranch(b.equations, origin, values, E)))
  end
  if isempty(branches)
    foreach(e -> scalarEquations!(out, origins, e, origin, values, E), elseEquations)
  else
    push!(out, IR.IF_EQUATION(branches, scalarBranch(elseEquations, origin, values, E), eq.source, unlinked(eq.attributes)))
    push!(origins, origin)
  end
  return nothing
end

"The equations of a branch of an if-equation the scalarized model keeps, scalarized."
function scalarBranch(equations::Vector{IR.Equation}, origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Vector{IR.Equation}
  local out = IR.Equation[]
  local origins = EquationOrigin[]
  foreach(e -> scalarEquations!(out, origins, e, origin, values, E), equations)
  return out
end

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.WHEN_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local branches = IR.WhenBranch[IR.WhenBranch(elementTriggers(IR.Expression[equationExpression(t, values, E)
                                                                             for t in b.triggers]),
                                               b.atInitial, b.atTerminal,
                                               elementAssignments(scalarStatements(b.body, values, E), E))
                                 for b in eq.branches]
  push!(out, IR.WHEN_EQUATION(branches, eq.kind, eq.source, unlinked(eq.attributes)))
  push!(origins, origin)
  return nothing
end

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.ALGORITHM,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local loop::Union{Nothing, IR.FOR_STATEMENT} = E.passesFollow ? independentLoop(eq.statements) : nothing
  local range::Union{Nothing, StepRange{Int, Int}} =
    loop === nothing ? nothing : iterationRange(loop.iterator.range, values, E)
  if loop !== nothing && range !== nothing
    #= one algorithm per iteration, as the frontend's scalarization has one per component =#
    local inner::Dict{Symbol, Int} = copy(values)
    for k in range
      inner[loop.iterator.name] = k
      scalarEquations!(out, origins, IR.ALGORITHM(loop.body, assignedCrefs(loop.body), eq.source, eq.attributes),
                       EquationOrigin(origin.equation, origin.path, vcat(origin.iteration, k), Int[]), inner, E)
    end
    return nothing
  end
  local outputs = IR.ComponentRef[]
  for c in eq.outputs
    pushReferences!(outputs, elementReference(IR.CREF(withIteratorCref(c, values), IR.T_UNKNOWN()), E))
  end
  local statements::Vector{IR.Statement} = IR.Statement[wholeTargets(s, x, values) for (s, x) in
                                                        zip(eq.statements, scalarStatements(eq.statements, values, E))]
  push!(out, IR.ALGORITHM(statements, outputs, eq.source, unlinked(eq.attributes)))
  push!(origins, origin)
  return nothing
end

"The references of `e` (a reference, or an array literal of them, `elementReference`) into `out`."
pushReferences!(out::Vector{IR.ComponentRef}, e::IR.CREF)::Nothing = (push!(out, e.cref); nothing)
pushReferences!(out::Vector{IR.ComponentRef}, e::IR.ARRAY)::Nothing = (foreach(x -> pushReferences!(out, x), e.elements); nothing)
pushReferences!(::Vector{IR.ComponentRef}, e::IR.Expression)::Nothing =
  throw(NotHandled("an algorithm output $(IR.toString(e))"))

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.ASSERT_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  push!(out, IR.ASSERT_EQUATION(equationExpression(eq.condition, values, E), equationExpression(eq.message, values, E),
                                equationExpression(eq.level, values, E), eq.source))
  push!(origins, origin)
  return nothing
end

"""
A record equation, where the scalar passes follow (`E.passesFollow`; NotHandled otherwise): by its fields where its
right-hand side is not a call (`recordEquations`, as the frontend's scalarization writes `a = b` of two records),
each scalarized in turn; one of a call (its result assigned whole; the branch of an if-expression its iterators
select, `selectedParts`), or that has no fields to go by, a record equation of the scalarized sides. A constant
`Complex(re, im)` constructor call is the record value (`constantRecord`).
"""
function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.RECORD_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  E.passesFollow || throw(NotHandled("a RECORD_EQUATION with arrays kept"))
  local lhs::IR.Expression = constantRecord(withIterators(eq.lhs, values))
  local rhs::IR.Expression = constantRecord(selectedParts(withIterators(eq.rhs, values)))
  if !(rhs isa IR.CALL)
    local fields::Vector{IR.Equation} = recordEquations(lhs, rhs, eq.source)
    if !(length(fields) == 1 && fields[1] isa IR.RECORD_EQUATION)
      foreach(f -> scalarEquations!(out, origins, f, origin, NO_ITERATORS, E), fields)
      return nothing
    end
  end
  push!(out, IR.RECORD_EQUATION(equationExpression(lhs, NO_ITERATORS, E), equationExpression(rhs, NO_ITERATORS, E),
                                eq.ty, eq.size, eq.source, unlinked(eq.attributes)))
  push!(origins, origin)
  return nothing
end

scalarEquations!(::Vector{IR.Equation}, ::Vector{EquationOrigin}, eq::IR.Equation, ::EquationOrigin, ::Dict{Symbol, Int},
                 ::Elements)::Nothing = throw(NotHandled("a $(nameof(typeof(eq))) with arrays kept"))

"""
    independentLoop(statements) -> Union{Nothing, IR.FOR_STATEMENT}

The for-statement an algorithm consists of, where its iterations are independent of each other: the algorithm is
then one algorithm per iteration (with arrays kept the frontend writes the algorithm section of a component array
as a loop over the components; scalarized it has one algorithm per component). Independent: every variable the
body assigns or reinitializes is subscripted by the loop's iterator in one of its dimensions, the same in every
assignment and in every reference of the body, so that an iteration reads and writes its own elements only.
`nothing` for another algorithm (several statements, a target that is not a reference, a variable assigned whole).
"""
function independentLoop(statements::Vector{IR.Statement})::Union{Nothing, IR.FOR_STATEMENT}
  (length(statements) == 1 && statements[1] isa IR.FOR_STATEMENT) || return nothing
  local loop::IR.FOR_STATEMENT = statements[1]
  local name::Symbol = loop.iterator.name
  local targets::Vector{IR.ComponentRef} = IR.ComponentRef[]
  collectTargets!(targets, loop.body) || return nothing
  local at::Dict{Symbol, Int} = Dict{Symbol, Int}()   # an assigned array => the dimension the iterator subscripts
  for target in targets
    local dimension::Union{Nothing, Int} = iteratorDimension(target, name)
    dimension === nothing && return nothing
    get!(at, unsubscriptedName(target), dimension) == dimension || return nothing
  end
  isempty(at) && return nothing
  local independent::Base.RefValue{Bool} = Ref(true)
  for s in loop.body
    IR.foreachExp(s) do x::IR.Expression
      x isa IR.CREF || return nothing
      local array::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
      local dimension::Union{Nothing, Int} = get(at, array, nothing)
      dimension === nothing && return nothing
      iteratorDimension(x.cref, name) == dimension || (independent[] = false)
      return nothing
    end
  end
  return independent[] ? loop : nothing
end

"The first of the dimensions of `cref` (`allSubscripts`) subscripted by the iterator `name`; `nothing` without one."
function iteratorDimension(cref::IR.ComponentRef, name::Symbol)::Union{Nothing, Int}
  for (d, s) in enumerate(allSubscripts(cref))
    local index::Union{Nothing, IR.Expression} = s isa IR.INDEX ? s.exp : nothing
    (index isa IR.CREF && index.cref.name === name && !IR.hasSubscripts(index.cref)) && return d
  end
  return nothing
end

"""
    collectTargets!(out, statements) -> Bool

The references `statements` assign or reinitialize, pushed onto `out`; false where a target is not a reference
(or a tuple of references and discarded outputs).
"""
collectTargets!(out::Vector{IR.ComponentRef}, statements::Vector{IR.Statement})::Bool =
  all(s -> collectTargets!(out, s), statements)
function collectTargets!(out::Vector{IR.ComponentRef}, s::IR.ASSIGNMENT)::Bool
  for target in (s.lhs isa IR.TUPLE ? s.lhs.elements : IR.Expression[s.lhs])
    target isa IR.WILD && continue
    target isa IR.CREF || return false
    push!(out, target.cref)
  end
  return true
end
collectTargets!(out::Vector{IR.ComponentRef}, s::IR.REINIT)::Bool = (push!(out, s.state); true)
collectTargets!(out::Vector{IR.ComponentRef}, s::IR.IF_STATEMENT)::Bool =
  all(b -> collectTargets!(out, b.body), s.branches) && collectTargets!(out, s.elseBody)
collectTargets!(out::Vector{IR.ComponentRef}, s::IR.WHEN_STATEMENT)::Bool =
  all(b -> collectTargets!(out, b.body), s.branches)
collectTargets!(out::Vector{IR.ComponentRef}, s::Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT})::Bool =
  collectTargets!(out, s.body)
collectTargets!(::Vector{IR.ComponentRef}, ::IR.Statement)::Bool = true

"`cref` with the iterators of `values` replaced by their values, its subscripts folded."
withIteratorCref(cref::IR.ComponentRef, values::Dict{Symbol, Int})::IR.ComponentRef =
  withIterators(IR.CREF(cref, IR.T_UNKNOWN()), values).cref

"""
    scalarStatements(body, values, E) -> Vector{Statement}

Statements with their root expressions scalarized (`scalarExpression`; a loop's `x[i]` kept), a `reinit`'s state
by its element variable. In a for-statement's body its iterator is bound (`bound`): `for i in 1:3 loop s := s + i`
in a model with an array `i` read the array (`s := s + {i[1], i[2], i[3]}`, the model refused).
"""
function scalarStatements(body::Vector{IR.Statement}, values::Dict{Symbol, Int}, E::Elements)::Vector{IR.Statement}
  return IR.Statement[elementTriggers(scalarReinit(s, values, E)) for s in rootsScalarized(body, values, E, NO_BOUND_NAMES)]
end

"""
`body` with the root expressions of its statements scalarized (`scalarExpression`, as `mapRoots` applies it), also in
nested statements; in a for-statement's body with its iterator among the names `bound`.
"""
function rootsScalarized(body::Vector{IR.Statement}, values::Dict{Symbol, Int}, E::Elements,
                         bound::Set{Symbol})::Vector{IR.Statement}
  #= a tuple as a root expression is the targets of an assignment =#
  local g::Function = (e::IR.Expression) -> withoutEmptyTargets(scalarExpression(e, values, E; bound))
  return IR.Statement[rootsScalarized(g, s, values, E, bound) for s in body]
end
rootsScalarized(g::Function, s::IR.FOR_STATEMENT, values::Dict{Symbol, Int}, E::Elements, bound::Set{Symbol})::IR.Statement =
  IR.FOR_STATEMENT(IR.Iterator(s.iterator.name, g(s.iterator.range)),
                   rootsScalarized(s.body, values, E, union(bound, (s.iterator.name,))), s.source)
rootsScalarized(g::Function, s::IR.IF_STATEMENT, values::Dict{Symbol, Int}, E::Elements, bound::Set{Symbol})::IR.Statement =
  IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(g(b.condition), rootsScalarized(b.body, values, E, bound))
                                     for b in s.branches],
                  rootsScalarized(s.elseBody, values, E, bound), s.source)
rootsScalarized(g::Function, s::IR.WHILE_STATEMENT, values::Dict{Symbol, Int}, E::Elements, bound::Set{Symbol})::IR.Statement =
  IR.WHILE_STATEMENT(g(s.condition), rootsScalarized(s.body, values, E, bound), s.source)
rootsScalarized(g::Function, s::IR.WHEN_STATEMENT, values::Dict{Symbol, Int}, E::Elements, bound::Set{Symbol})::IR.Statement =
  IR.WHEN_STATEMENT(IR.WhenBranch[IR.WhenBranch(IR.Expression[g(t) for t in b.triggers], b.atInitial, b.atTerminal,
                                                rootsScalarized(b.body, values, E, bound)) for b in s.branches],
                    s.source)
rootsScalarized(g::Function, s::IR.Statement, ::Dict{Symbol, Int}, ::Elements, ::Set{Symbol})::IR.Statement = mapRoots(g, s)

"""
    elementTriggers(triggers) -> Vector{Expression}

The triggers of a scalarized when-branch with each trigger that is an array (`when b`, `b` a Boolean array: `{b[1],
b[2]}` scalarized) as its elements, each a trigger: the when fires at each element's rising edge (MLS 8.3.5.1), as the
lowering splits a vector condition (`whenBranch`) and the frontend's scalarization writes it. Kept as one trigger, a
one-element array `{pre_reset[1]}` (MSL RadioButtonSource's `when pre_reset`) was no Boolean the MTK route's events
read: the event iteration did not settle at the initialization.
"""
function elementTriggers(triggers::Vector{IR.Expression})::Vector{IR.Expression}
  local out::Vector{IR.Expression} = IR.Expression[]
  foreach(t -> elementTriggers!(out, t), triggers)
  return out
end
elementTriggers!(out::Vector{IR.Expression}, t::IR.ARRAY)::Nothing = (foreach(e -> elementTriggers!(out, e), t.elements); nothing)
elementTriggers!(out::Vector{IR.Expression}, @nospecialize(t::IR.Expression))::Nothing = (push!(out, t); nothing)
"A when-statement with its branches' triggers as `elementTriggers`; another statement as it is."
elementTriggers(s::IR.WHEN_STATEMENT)::IR.Statement =
  IR.WHEN_STATEMENT(IR.WhenBranch[IR.WhenBranch(elementTriggers(b.triggers), b.atInitial, b.atTerminal, b.body)
                                  for b in s.branches], s.source)
elementTriggers(s::IR.Statement)::IR.Statement = s

"""
    elementAssignments(body, E) -> Vector{Statement}

The statements of a when-equation with each assignment of an array (`d = 2 * pre(d)`, arrays kept: `{d[1], d[2]} :=
2 * pre({d[1], d[2]})` in the scalarized model) as one assignment per element (`pickElement` of both sides), as the
frontend's scalarization writes the equation; in the branches of an if too. An array of records and a tuple of
targets stay. Not for an algorithm's statements: those run in order, and an array's assignment there reads its whole
right-hand side before an element is assigned (`wholeTargets`).
"""
function elementAssignments(body::Vector{IR.Statement}, E::Elements)::Vector{IR.Statement}
  local out::Vector{IR.Statement} = IR.Statement[]
  foreach(s -> elementAssignments!(out, s, E), body)
  return out
end
function elementAssignments!(out::Vector{IR.Statement}, s::IR.ASSIGNMENT, E::Elements)::Nothing
  local ty::IR.ModelicaType = s.lhs isa IR.ARRAY ? IR.typeOf(s.lhs) : IR.T_UNKNOWN()
  local element::IR.ModelicaType = ty isa IR.T_ARRAY ? IR.elementType(ty) : ty
  if !(ty isa IR.T_ARRAY) || element isa IR.T_RECORD
    push!(out, s)
    return nothing
  end
  for index in rowMajor(knownSizes(ty, "an array assigned in a when-equation"))
    local subscripts::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index]
    push!(out, IR.ASSIGNMENT(pickElement(s.lhs, subscripts, element, E), pickElement(s.rhs, subscripts, element, E),
                             s.source))
  end
  return nothing
end
function elementAssignments!(out::Vector{IR.Statement}, s::IR.IF_STATEMENT, E::Elements)::Nothing
  local branches::Vector{IR.StatementBranch} =
    IR.StatementBranch[IR.StatementBranch(b.condition, elementAssignments(b.body, E)) for b in s.branches]
  push!(out, IR.IF_STATEMENT(branches, elementAssignments(s.elseBody, E), s.source))
  return nothing
end
elementAssignments!(out::Vector{IR.Statement}, s::IR.Statement, ::Elements)::Nothing = (push!(out, s); nothing)

"""
    wholeTargets(original, scalar, values) -> Statement

The statement `scalar` of an algorithm (`scalarStatements` of `original`) with each target that is an array as a
whole as the reference it was (`s`, not `{s[1], s[2]}`; a loop's iterators at their values), also in a tuple of
targets and in nested statements: the form the frontend's scalarization gives an algorithm, which the passes on
algorithms take (`Passes.tupleAssignElements`: a target that is not a variable is not supported; MSL realFFT's
`(info, Ai, ...) := realFFT(...)`, Xorshift64star's `(r, state) := random(pre(state))`).
"""
function wholeTargets(original::IR.ASSIGNMENT, scalar::IR.ASSIGNMENT, values::Dict{Symbol, Int})::IR.Statement
  local lhs::IR.Expression = scalar.lhs
  if original.lhs isa IR.TUPLE && lhs isa IR.TUPLE && length(original.lhs.elements) == length(lhs.elements)
    lhs = IR.TUPLE(IR.Expression[wholeTarget(o, x, values) for (o, x) in zip(original.lhs.elements, lhs.elements)],
                   lhs.ty)
  else
    lhs = wholeTarget(original.lhs, lhs, values)
  end
  return IR.ASSIGNMENT(lhs, scalar.rhs, scalar.source)
end
"The target `scalar` of `original`: the reference `original` where its scalarized form is its array of elements."
wholeTarget(original::IR.Expression, scalar::IR.Expression, values::Dict{Symbol, Int})::IR.Expression =
  (original isa IR.CREF && scalar isa IR.ARRAY) ? withIterators(original, values) : scalar
wholeTargets(original::IR.IF_STATEMENT, scalar::IR.IF_STATEMENT, values::Dict{Symbol, Int})::IR.Statement =
  IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(x.condition, wholeTargets(o.body, x.body, values))
                                     for (o, x) in zip(original.branches, scalar.branches)],
                  wholeTargets(original.elseBody, scalar.elseBody, values), scalar.source)
wholeTargets(original::IR.FOR_STATEMENT, scalar::IR.FOR_STATEMENT, values::Dict{Symbol, Int})::IR.Statement =
  IR.FOR_STATEMENT(scalar.iterator, wholeTargets(original.body, scalar.body, values), scalar.source)
wholeTargets(original::IR.WHILE_STATEMENT, scalar::IR.WHILE_STATEMENT, values::Dict{Symbol, Int})::IR.Statement =
  IR.WHILE_STATEMENT(scalar.condition, wholeTargets(original.body, scalar.body, values), scalar.source)
wholeTargets(original::IR.WHEN_STATEMENT, scalar::IR.WHEN_STATEMENT, values::Dict{Symbol, Int})::IR.Statement =
  IR.WHEN_STATEMENT(IR.WhenBranch[IR.WhenBranch(x.triggers, x.atInitial, x.atTerminal,
                                                wholeTargets(o.body, x.body, values))
                                  for (o, x) in zip(original.branches, scalar.branches)], scalar.source)
wholeTargets(::IR.Statement, scalar::IR.Statement, ::Dict{Symbol, Int})::IR.Statement = scalar
wholeTargets(original::Vector{IR.Statement}, scalar::Vector{IR.Statement},
             values::Dict{Symbol, Int})::Vector{IR.Statement} =
  IR.Statement[wholeTargets(o, x, values) for (o, x) in zip(original, scalar)]

"A `reinit`'s state by its element variable; another statement (its nested ones too) as it is."
function scalarReinit(s::IR.REINIT, values::Dict{Symbol, Int}, E::Elements)::IR.Statement
  local state = elementReference(IR.CREF(withIteratorCref(s.state, values), IR.T_REAL()), E)
  state isa IR.CREF || throw(NotHandled("a reinit of a whole array ($(s.state.name))"))
  return IR.REINIT(state.cref, s.value, s.source)
end
scalarReinit(s::IR.IF_STATEMENT, values::Dict{Symbol, Int}, E::Elements)::IR.Statement =
  IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(b.condition, IR.Statement[scalarReinit(x, values, E) for x in b.body]) for b in s.branches],
                  IR.Statement[scalarReinit(x, values, E) for x in s.elseBody], s.source)
scalarReinit(s::IR.Statement, ::Dict{Symbol, Int}, ::Elements)::IR.Statement = s

"""
The element equations of equations of the array form and their origins (relative to the equation's own index), by
the equation (its identity), for the next element view of the same model (`scalarizeArrays(model; cache)`): an
entry holds while the arrays its equation reads keep their elements. Splitting arrays (`splitArrays!`) replaces the
equations that read them and keeps the others, whose element equations are then the same (MSL EngineV6 split its
arrays in 28 rounds, each view of 18321 element equations scalarized again: 7.4 s of its 11.6 s solve).
"""
const ScalarCache = IdDict{IR.Equation, Tuple{Vector{IR.Equation}, Vector{EquationOrigin}}}

"""
    scalarizeArrays(model; passesFollow = false, cache = nothing) -> Scalarized

`model` with its arrays as their elements (`Scalarized`): its variables (`scalarVariables`), the equations and
initial equations of its continuous partition (`scalarEquations!`). The model is not changed. NotHandled for a
model with several continuous partitions, clocked partitions or structural modes, and for what
`scalarEquations!` does not handle. `passesFollow`: `model` is as lowered and the scalar passes follow (`Elements`).
`cache`: the element equations of earlier views of the model taken and kept (`ScalarCache`); the views share them.
"""
function scalarizeArrays(model::IR.Model; passesFollow::Bool = false,
                         cache::Union{Nothing, ScalarCache} = nothing)::Scalarized
  (length(model.continuous) == 1 && isempty(model.clocked) && isempty(model.modes)) ||
    throw(NotHandled("arrays in a model of several partitions or structural modes"))
  local E::Elements = elements(model.variables; passesFollow)
  local (partition::IR.Partition, origins::Vector{EquationOrigin}) = scalarPartition(model.continuous[1], E, cache)
  local scalar::IR.Model = IR.Model(model.name, scalarVariables(model.variables, E), IR.Partition[partition], model.clocked,
                                    model.baseClocks, model.functions, model.initialMode, model.modes, model.transitions,
                                    model.frontendModel)
  return Scalarized(scalar, origins, E)
end

"""
    scalarPartition(p, E, cache) -> (partition, origins)

The partition `p` with its equations and initial equations as their elements (`scalarEquationsOf!`) and its array
unknowns as their elements (`E`), and the origins of its element equations.
"""
function scalarPartition(p::IR.Partition, E::Elements,
                         cache::Union{Nothing, ScalarCache})::Tuple{IR.Partition, Vector{EquationOrigin}}
  local equations::Vector{IR.Equation} = IR.Equation[]
  local origins::Vector{EquationOrigin} = EquationOrigin[]
  for (i, eq) in enumerate(p.equations)
    scalarEquationsOf!(equations, origins, eq, i, E, cache)
  end
  local initialEquations::Vector{IR.Equation} = IR.Equation[]
  local initialOrigins::Vector{EquationOrigin} = EquationOrigin[]
  for (i, eq) in enumerate(p.initialEquations)
    scalarEquationsOf!(initialEquations, initialOrigins, eq, i, E, cache)
  end
  local unknowns::Vector{Symbol} = Symbol[]
  for n in p.unknowns
    haskey(E.elements, n) ? append!(unknowns, E.elements[n]) : push!(unknowns, n)
  end
  return (IR.Partition(p.index, p.association, unknowns, equations, initialEquations), origins)
end

"""
    scalarizeModel(model) -> IR.Model

`model` as lowered with its arrays as their elements, for the scalar passes (`scalarizeArrays` with `passesFollow`)
in every part of it: its continuous and clocked partitions, and each structural mode with its own variables. The
backend's scalarization (`backendScalarizedModel`) of a model the array form does not take. The model is not
changed.
"""
function scalarizeModel(model::IR.Model)::IR.Model
  local E::Elements = elements(model.variables; passesFollow = true)
  return IR.Model(model.name, scalarVariables(model.variables, E),
                  IR.Partition[first(scalarPartition(p, E, nothing)) for p in model.continuous],
                  IR.Partition[first(scalarPartition(p, E, nothing)) for p in model.clocked], model.baseClocks,
                  model.functions, model.initialMode, IR.StructuralMode[scalarMode(m) for m in model.modes],
                  model.transitions, model.frontendModel)
end

"The structural mode `m` with its variables and its system as their elements (`scalarizeModel`)."
function scalarMode(m::IR.StructuralMode)::IR.StructuralMode
  local E::Elements = elements(m.variables; passesFollow = true)
  return IR.StructuralMode(m.name, scalarVariables(m.variables, E), first(scalarPartition(m.system, E, nothing)))
end

"""
    scalarEquationsOf!(out, origins, eq, i, E, cache)

The element equations of the `i`-th equation `eq` (`scalarEquations!`) onto `out` and their origins onto `origins`;
taken from `cache` where it has them (`ScalarCache`), else put there.
"""
function scalarEquationsOf!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, @nospecialize(eq::IR.Equation), i::Int, E::Elements,
                            cache::Union{Nothing, ScalarCache})::Nothing
  if cache === nothing
    scalarEquations!(out, origins, eq, EquationOrigin(i), NO_ITERATORS, E)
    return nothing
  end
  local entry::Union{Nothing, Tuple{Vector{IR.Equation}, Vector{EquationOrigin}}} = get(cache, eq, nothing)
  if entry === nothing
    local mine::Vector{IR.Equation} = IR.Equation[]
    local where::Vector{EquationOrigin} = EquationOrigin[]
    scalarEquations!(mine, where, eq, EquationOrigin(i), NO_ITERATORS, E)
    entry = cache[eq] = (mine, where)
  end
  append!(out, entry[1])
  for o::EquationOrigin in entry[2]
    push!(origins, EquationOrigin(i, o.path, o.iteration, o.element))
  end
  return nothing
end

"Whether `model` has a continuous-time array variable (not a parameter's or a constant's)."
hasContinuousArrays(model::IR.Model)::Bool = any(isContinuousArray, model.variables)

"""
    hasArrayEquations(model) -> Bool

Whether a continuous partition of `model` has a for-equation or an array equation reading a continuous array: what
the array form keeps (a model without one reduces as its scalarized form).
"""
function hasArrayEquations(model::IR.Model)::Bool
  local found = Ref(false)
  for p in model.continuous, eq in p.equations
    eq isa Union{IR.FOR_EQUATION, IR.ARRAY_EQUATION} || continue
    IR.foreachRootExp(e -> (found[] || (found[] = readsContinuousArray(e, model.variables)); nothing), eq)
    found[] && return true
  end
  return false
end

"""
    bindingEquation(v, variables) -> Union{Nothing, FOR_EQUATION}

The declaration binding of the continuous Real array `v` (arrays kept) as a for-equation over its elements
(`nestedFor`): `for \$i1 in 1:3 loop c[\$i1].b = 2 * c[\$i1].a; end for` of `Real b = 2 * a` in `C c[3]`. The
frontend keeps the binding of a field of a component array as the class has it, its references typed without the
components' dimensions (`c.a` typed `Real`): those are the fields of the element's component (`componentCref`). An
array value of the field's own dimensions gives its element (`elementAt`). `nothing` where a dimension has no
literal size, or the value's dimensions are not the variable's trailing ones.
"""
function bindingEquation(v::IR.Variable, variables::IR.VariableSet)::Union{Nothing, IR.FOR_EQUATION}
  local binding::Union{Nothing, IR.Expression} = v.binding
  (binding === nothing || !(v.ty isa IR.T_ARRAY)) && return nothing
  local sizes::Vector{Int} = Int[]
  for d in IR.dimensions(v.ty)
    local n::Union{Nothing, Int} = IR.dimensionSize(d)
    n === nothing && return nothing
    push!(sizes, n)
  end
  local ty::IR.ModelicaType = IR.typeOf(binding)
  local own::Int = ty isa IR.T_ARRAY ? length(IR.dimensions(ty)) : 0
  local lead::Int = length(sizes) - own
  lead >= 0 || return nothing
  local subscripts::Vector{IR.Subscript} = iteratorSubscripts(sizes)
  local named::IR.Expression = IR.mapExp(binding) do @nospecialize(x::IR.Expression)
    x isa IR.CREF || return x
    local missing::Int = missingComponentDimensions(x, variables)
    missing > 0 || return x
    return IR.CREF(componentCref(x.cref, v.cref, subscripts, missing), x.ty)
  end
  local element::IR.ModelicaType = IR.elementType(v.ty)
  local value::IR.Expression = own == 0 ? named : elementAt(named, subscripts[lead + 1:end], element, variables)
  local lhs::IR.Expression = IR.CREF(distributedCref(v, subscripts), element)
  return nestedFor(IR.Equation[IR.SCALAR_EQUATION(lhs, value, element, v.source, IR.DEFAULT_ATTRIBUTES)], sizes)
end

"""
    discreteBindingEquations(v, variables) -> Vector{Equation}

The declaration binding of the Boolean, Integer or enumeration array `v` (arrays kept: a field of an array of
records, `states.phase` of `ThermodynamicState states[1] = {medium.state}`) as one equation per element,
`v[k] = e[k]` (`bindingEquation` with its iterations written out, an element of an array literal picked): the
passes of discrete equations take scalar ones. Empty where `bindingEquation` gives none. Without them the elements
were discrete unknowns no equation gives (MSL Fluid PumpingSystem: 314 equations for 316 unknowns).
"""
function discreteBindingEquations(v::IR.Variable, variables::IR.VariableSet)::Vector{IR.Equation}
  local rolled::Union{Nothing, IR.FOR_EQUATION} = bindingEquation(v, variables)
  rolled === nothing && return IR.Equation[]
  local iterators::Set{Symbol} = Set{Symbol}()
  local loop::IR.Equation = rolled
  while loop isa IR.FOR_EQUATION
    push!(iterators, loop.iterator.name)
    loop = loop.body[1]
  end
  local picked = (@nospecialize(x::IR.Expression)) ->
    x isa IR.SUBSCRIPTED ? subscriptedReference(foldTree(selectElement(x.exp, x.subscripts, x.ty))) : x
  local out::Vector{IR.Equation} = IR.Equation[]
  for eq::IR.Equation in unrollIterators(rolled, iterators)
    push!(out, eq isa IR.SCALAR_EQUATION ?
                 IR.SCALAR_EQUATION(eq.lhs, IR.mapExp(picked, eq.rhs), eq.ty, eq.source, eq.attributes) : eq)
  end
  return out
end

"""
`componentDimensions` of the reference `x` in a model with its arrays kept: the leading dimensions of the array
variable it names that its type does not have and its subscripts do not index; 0 for a reference to no array.
"""
function missingComponentDimensions(x::IR.CREF, variables::IR.VariableSet)::Int
  local name::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
  local v::Union{Nothing, IR.Variable} = get(variables, name, nothing)
  #= a record is no variable: its dimensions are its first Real field's (`c.V_m` of a component array of Complex
     values: `c.V_m.re`) =#
  local ty::IR.ModelicaType = IR.elementType(x.ty)
  while v === nothing && ty isa IR.T_RECORD && !isempty(ty.fields)
    name = Symbol(name, ".", ty.fields[1].name)
    v = get(variables, name, nothing)
    ty = IR.elementType(ty.fields[1].ty)
  end
  (v === nothing || !(v.ty isa IR.T_ARRAY)) && return 0
  return length(IR.dimensions(v.ty)) - count(s -> s isa IR.INDEX, allSubscripts(x.cref)) - length(IR.dimensions(x.ty))
end

"""
    rollArrayEquations!(model) -> Model

The array equations of `model`'s continuous partitions (arrays kept) that read a continuous array as for-equations
over their elements (`nestedFor`), before `expandEquations` would split them into one equation per element:
`phi2 = phi1` is `for \$i1 in 1:n loop phi2[\$i1] = phi1[\$i1]; end for`, a side that is no variable subscripted
(`(2 * der(w))[\$i1]`). Array equations of unknown dimensions and those of no continuous array stay.
"""
function rollArrayEquations!(model::IR.Model)::IR.Model
  for p in model.continuous
    local equations::Vector{IR.Equation} = IR.Equation[]
    for eq in p.equations
      local rolled::IR.Equation = rolledEquation(eq, model.variables)
      local elements::Union{Nothing, Vector{IR.Equation}} =
        rolled === eq && eq isa IR.ARRAY_EQUATION ? recordElementEquations(eq, model.variables) : nothing
      elements === nothing ? push!(equations, rolled) : append!(equations, elements)
    end
    p.equations = equations
  end
  return model
end

"""
The element of the array of records `e` at `subscripts`: of a reference the reference subscripted (an array of
records is no variable, `elementAt` reads a variable's dimensions; its fields' arrays are), of another expression
`elementAt`.
"""
function recordElementAt(e::IR.Expression, subscripts::Vector{IR.Subscript}, element::IR.ModelicaType,
                         variables::IR.VariableSet)::IR.Expression
  e isa IR.CREF && return IR.CREF(sliceElementCref(e.cref, subscripts), element)
  return elementAt(e, subscripts, element, variables)
end

"""
    sliceElementCref(cref, subscripts) -> ComponentRef

The element `subscripts` of the array the reference `cref` names, where its parts select whole dimensions or ranges
of them: each subscript in place of the reference's next slice, in order (`mediums[:].state` at i is
`mediums[i].state`; `statesFM[2:3]` at i is `statesFM[1 + i]`, the range's member: `sliceMember`), those left after
its last part's own (`IR.appendSubscripts`). MSL Fluid's pipes, `mediums[:].state = statesFM[2:3]`: with the
subscript appended the body read `mediums[:].state[i].p` ("of more subscripts than its array's dimensions").
"""
function sliceElementCref(cref::IR.ComponentRef, subscripts::Vector{IR.Subscript})::IR.ComponentRef
  local next::Int = 0
  local parts::Vector{IR.CrefPart} = IR.CrefPart[]
  for p::IR.CrefPart in cref.parts
    local own::Vector{IR.Subscript} = IR.Subscript[]
    for s::IR.Subscript in p.subscripts
      local slice::Bool = s isa Union{IR.WHOLE, IR.SLICE} && next < length(subscripts)
      push!(own, slice ? sliceMember(s, subscripts[next += 1]) : s)
    end
    push!(parts, own == p.subscripts ? p : IR.CrefPart(p.name, own, p.ty))
  end
  local resolved::IR.ComponentRef = IR.ComponentRef(parts)
  return next == length(subscripts) ? resolved : IR.appendSubscripts(resolved, subscripts[(next + 1):end])
end

"The member `at` of the dimension a slice subscript selects: of the whole dimension `at` itself."
sliceMember(::IR.WHOLE, at::IR.Subscript)::IR.Subscript = at

"""
Of a range of Integer literals its member: `first + (at - 1) * step`, a literal for a literal `at` (`2:3` at 2 is 3,
at `i` it is `1 + i`). NotHandled for a slice that is no such range (an array of indices).
"""
function sliceMember(s::IR.SLICE, at::IR.Subscript)::IR.Subscript
  local range::Union{Nothing, StepRange{Int, Int}} = literalRange(foldTree(s.exp))
  (range === nothing || !(at isa IR.INDEX)) && throw(NotHandled("an element of a slice that is no range of literals"))
  local k::Union{Nothing, Int} = IR.literalIndex(at)
  k === nothing || return IR.INDEX(IR.INTEGER(range[k]))
  local offset::Int = first(range) - step(range)
  local scaled::IR.Expression =
    step(range) == 1 ? at.exp : IR.BINARY(IR.INTEGER(step(range)), IR.MUL, at.exp, IR.T_INTEGER())
  return IR.INDEX(offset == 0 ? scaled : IR.BINARY(IR.INTEGER(offset), IR.ADD, scaled, IR.T_INTEGER()))
end

"""
    recordElementEquations(eq, variables) -> Union{Nothing, Vector{Equation}}

The array equation `eq` of records that is no for-equation over its elements (`elementLoop`: a side is the array of
its elements' values written out, `{i_p[1] + i_n[1], i_p[2] + i_n[2], ...}`, which has no element by an iterator)
as the equations of each element's fields (`bodyRecordEquations`); `nothing` for another equation, an unknown size,
or where a record equation is left. `expandEquations` wrote one equation of a record per element, one row for its
two unknowns (MSL QuasiStatic Polyphase: 18 unknowns over the rows).
"""
function recordElementEquations(eq::IR.ARRAY_EQUATION, variables::IR.VariableSet)::Union{Nothing, Vector{IR.Equation}}
  local element::IR.ModelicaType = IR.elementType(eq.ty)
  element isa IR.T_RECORD || return nothing
  local sizes::Vector{Int} = Int[]
  for d in eq.ty.dimensions
    local n::Union{Nothing, Int} = IR.dimensionSize(d)
    n === nothing && return nothing
    push!(sizes, n)
  end
  local out::Vector{IR.Equation} = IR.Equation[]
  for index in rowMajor(sizes)
    local subscripts::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index]
    local elements::IR.RECORD_EQUATION =
      IR.RECORD_EQUATION(recordElementAt(eq.lhs, subscripts, element, variables),
                         recordElementAt(eq.rhs, subscripts, element, variables), element, length(element.fields),
                         eq.source, eq.attributes)
    local fields::Vector{IR.Equation} = bodyRecordEquations(IR.Equation[elements])
    any(f -> f isa IR.RECORD_EQUATION, fields) && return nothing
    append!(out, fields)
  end
  return out
end

rolledEquation(eq::IR.Equation, ::IR.VariableSet, ::Int = 0)::IR.Equation = eq

"""
The array equations in the body of a for-equation rolled likewise (`for \$i1 loop {a[\$i1], b[\$i1]} = c[\$i1].w`,
an array field of a component array), by iterators after the enclosing ones (`used`, `iteratorIndex`).
"""
function rolledEquation(eq::IR.FOR_EQUATION, variables::IR.VariableSet, used::Int = 0)::IR.Equation
  local inner::Int = max(used, iteratorIndex(eq.iterator.name))
  local body::Vector{IR.Equation} = IR.Equation[rolledEquation(b, variables, inner) for b in eq.body]
  all(k -> body[k] === eq.body[k], eachindex(body)) && return eq
  return IR.FOR_EQUATION(eq.iterator, body, eq.source, eq.attributes)
end

function rolledEquation(eq::IR.ARRAY_EQUATION, variables::IR.VariableSet, used::Int = 0)::IR.Equation
  #= an array of records has no variable of its own (its fields' arrays are the variables) =#
  (IR.elementType(eq.ty) isa IR.T_RECORD || readsContinuousArray(eq.lhs, variables) ||
   readsContinuousArray(eq.rhs, variables)) || return eq
  return elementLoop(eq, variables, used)
end

"""
    elementLoop(eq, variables, used) -> Equation

The array equation `eq` as a for-equation over its elements (`nestedFor`, `lhs[\$i1] = rhs[\$i1]` by `elementAt`),
by iterators after the enclosing ones (`used`); of an unknown size as it is. Of records: the elements' record
equation by its fields in the body (`bodyRecordEquations`: a Complex one by its parts, `i_p[\$i1].re + i_n[\$i1].re
= 0.0`), as it is where that leaves a record equation (a function's record result).
"""
function elementLoop(eq::IR.ARRAY_EQUATION, variables::IR.VariableSet, used::Int = 0)::IR.Equation
  local sizes::Vector{Int} = Int[]
  for d in eq.ty.dimensions
    local n::Union{Nothing, Int} = IR.dimensionSize(d)
    n === nothing && return eq
    push!(sizes, n)
  end
  local subscripts::Vector{IR.Subscript} = iteratorSubscripts(sizes, used)
  local element::IR.ModelicaType = IR.elementType(eq.ty)
  if element isa IR.T_RECORD
    local elements::IR.RECORD_EQUATION =
      IR.RECORD_EQUATION(recordElementAt(eq.lhs, subscripts, element, variables),
                         recordElementAt(eq.rhs, subscripts, element, variables), element, length(element.fields),
                         eq.source, eq.attributes)
    local fields::Vector{IR.Equation} = bodyRecordEquations(IR.Equation[elements])
    any(f -> f isa IR.RECORD_EQUATION, fields) && return eq
    return nestedFor(fields, sizes, used)
  end
  local body::IR.SCALAR_EQUATION =
    IR.SCALAR_EQUATION(elementAt(eq.lhs, subscripts, element, variables),
                       elementAt(eq.rhs, subscripts, element, variables), element, eq.source, eq.attributes)
  return nestedFor(IR.Equation[body], sizes, used)
end

"""
An if-equation whose branches all hold array equations of one size (an array-valued if-expression lifted to its
variable, `liftIfExpressions!`: `if c then tmp = a else tmp = b`) as a for-equation over the elements with the
if-equation of their elements in its body: `for i in 1:n loop if c then tmp[i] = a[i] else ... end for`.
"""
function rolledEquation(eq::IR.IF_EQUATION, variables::IR.VariableSet, used::Int = 0)::IR.Equation
  local bodies::Vector{Vector{IR.Equation}} =
    Vector{IR.Equation}[[b.equations for b in eq.branches]; [eq.elseEquations]]
  local lead::Union{Nothing, IR.Equation} = isempty(bodies[1]) ? nothing : bodies[1][1]
  lead isa IR.ARRAY_EQUATION || return eq
  local sizes::Vector{Int} = Int[]
  for d in lead.ty.dimensions
    local n::Union{Nothing, Int} = IR.dimensionSize(d)
    n === nothing && return eq
    push!(sizes, n)
  end
  local alike = (b::IR.Equation) -> b isa IR.ARRAY_EQUATION && b.ty.dimensions == lead.ty.dimensions
  all(body -> !isempty(body) && all(alike, body), bodies) || return eq
  local subscripts::Vector{IR.Subscript} = iteratorSubscripts(sizes, used)
  local element::IR.ModelicaType = IR.elementType(lead.ty)
  local elements = (body::Vector{IR.Equation}) ->
    IR.Equation[IR.SCALAR_EQUATION(elementAt(b.lhs, subscripts, element, variables),
                                   elementAt(b.rhs, subscripts, element, variables), element, b.source, b.attributes)
                for b in body]
  local rolled::IR.IF_EQUATION =
    IR.IF_EQUATION(IR.EquationBranch[IR.EquationBranch(b.condition, elements(b.equations)) for b in eq.branches],
                   elements(eq.elseEquations), eq.source, eq.attributes)
  return nestedFor(IR.Equation[rolled], sizes, used)
end

"""
    splitStaticIfs!(model) -> Model

The for-equations of `model`'s continuous partitions (arrays kept) with an if-equation in their body on the iterator
alone (its conditions fold to literals at each of its values: `if i == 1 then ... else ... end if`): that if-equation
as for-equations over the runs of values that take one branch, each with the branch's equations (`for i in 1:1`,
`for i in 2:n`); the body's other equations stay in the for-equation. Index reduction and the derivatives' solve in
array form read them as any for-equation's body.
"""
function splitStaticIfs!(model::IR.Model)::IR.Model
  for p in model.continuous
    p.equations = IR.Equation[x for eq in p.equations for x in staticIfsSplit(eq)]
  end
  return model
end

"The equation `eq` with its static if-equations split (`splitStaticIfs!`): itself, or the for-equations it becomes."
function staticIfsSplit(eq::IR.Equation)::Vector{IR.Equation}
  eq isa IR.FOR_EQUATION || return IR.Equation[eq]
  local range::Union{Nothing, StepRange{Int, Int}} = literalRange(foldTree(eq.iterator.range))
  range === nothing && return IR.Equation[eq]
  local kept::Vector{IR.Equation} = IR.Equation[]
  local split::Vector{IR.Equation} = IR.Equation[]
  for b in eq.body
    local taken::Union{Nothing, Vector{Int}} = b isa IR.IF_EQUATION ? staticBranches(b, eq.iterator.name, range) : nothing
    if taken === nothing
      push!(kept, b)
      continue
    end
    local ifEq::IR.IF_EQUATION = b
    for k in sort!(unique(taken))
      local equations::Vector{IR.Equation} = k == 0 ? ifEq.elseEquations : ifEq.branches[k].equations
      isempty(equations) && continue
      for run in valueRuns(Int[v for (v, t) in zip(range, taken) if t == k])
        local runRange::IR.RANGE = IR.RANGE(IR.INTEGER(first(run)), nothing, IR.INTEGER(last(run)),
                                            IR.T_ARRAY(IR.T_INTEGER(), IR.Dimension[IR.DIM_SIZE(length(run))]))
        push!(split, IR.FOR_EQUATION(IR.Iterator(eq.iterator.name, runRange), deepcopy(equations), eq.source, eq.attributes))
      end
    end
  end
  isempty(split) && return IR.Equation[eq]
  isempty(kept) || pushfirst!(split, IR.FOR_EQUATION(eq.iterator, kept, eq.source, eq.attributes))
  return split
end

"""
The branch an if-equation takes at each value of the iterator `name` in `range` (its position; 0: the else), where its
conditions are of literals there (`literalCondition`); `nothing` where one is not.
"""
function staticBranches(eq::IR.IF_EQUATION, name::Symbol, range::StepRange{Int, Int})::Union{Nothing, Vector{Int}}
  local taken::Vector{Int} = Int[]
  for v in range
    local branch::Int = 0
    for (k, b) in enumerate(eq.branches)
      local c::Union{Nothing, Bool} = literalCondition(foldTree(withIterator(b.condition, name, v)))
      c === nothing && return nothing
      if c
        branch = k
        break
      end
    end
    push!(taken, branch)
  end
  return taken
end

"The ascending `values` as runs of consecutive integers (`[1, 2, 3, 5]`: `1:3`, `5:5`)."
function valueRuns(values::Vector{Int})::Vector{UnitRange{Int}}
  local runs::Vector{UnitRange{Int}} = UnitRange{Int}[]
  for v in values
    (!isempty(runs) && last(runs[end]) == v - 1) ? (runs[end] = first(runs[end]):v) : push!(runs, v:v)
  end
  return runs
end

"Whether `e` reads a continuous array variable (a state, a derivative, an algebraic array), whole or by an element."
readsContinuousArray(e::IR.Expression, variables::IR.VariableSet)::Bool =
  IR.anyExp(x -> x isa IR.CREF && isContinuousArray(get(variables, unsubscriptedName(x.cref), nothing)), e)

isContinuousArray(::Nothing)::Bool = false
isContinuousArray(v::IR.Variable)::Bool = v.ty isa IR.T_ARRAY && v.kind isa IR.ContinuousKind

"""
    elementAt(e, subscripts, element, variables) -> Expression

The element of the array expression `e` at `subscripts` (the iterators of a rolled array equation), distributed over
element-wise arithmetic (`selectElement`): a whole array variable's element reference (`distributedCref`), an array
constructor's body at the iterators (`constructorAt`: `{sin(time) * i for i in 1:n}` is `sin(time) * \$i1`),
another expression subscripted.
"""
function elementAt(@nospecialize(e::IR.Expression), subscripts::Vector{IR.Subscript},
                   @nospecialize(element::IR.ModelicaType),
                   variables::IR.VariableSet)::IR.Expression
  return IR.mapExp(selectElement(e, subscripts, element)) do @nospecialize(x::IR.Expression)
    (x isa IR.SUBSCRIPTED && x.subscripts == subscripts) || return x
    if x.exp isa IR.CREF && !IR.hasSubscripts(x.exp.cref)
      local v = get(variables, x.exp.cref.name, nothing)
      (v !== nothing && v.ty isa IR.T_ARRAY && length(IR.dimensions(v.ty)) == length(subscripts)) &&
        return IR.CREF(distributedCref(v, subscripts), x.ty)
    elseif x.exp isa IR.CREF
      #= an array field of an element of a component array (`c[\$i1].w`, in a for-equation over the components) =#
      local own::Vector{IR.Subscript} = allSubscripts(x.exp.cref)
      local w = get(variables, unsubscriptedName(x.exp.cref), nothing)
      (w !== nothing && w.ty isa IR.T_ARRAY && all(s -> s isa IR.INDEX, own) &&
       length(IR.dimensions(w.ty)) == length(own) + length(subscripts)) &&
        return IR.CREF(distributedCref(w, vcat(own, subscripts)), x.ty)
    elseif x.exp isa IR.ARRAY_CONSTRUCTOR && length(subscripts) == 1
      local body = constructorAt(x.exp, only(subscripts))
      body === nothing || return body
    end
    return x
  end
end

"""
    constructorAt(e, s) -> Union{Nothing, Expression}

The element of an array constructor of one iterator over a literal range (`{f(i) for i in a:b}`) at the subscript `s`
(an iterator of a rolled equation): its body, the iterator by its value there (`a + s - 1`); `nothing` for another
constructor.
"""
function constructorAt(e::IR.ARRAY_CONSTRUCTOR, s::IR.Subscript)::Union{Nothing, IR.Expression}
  (length(e.iterators) == 1 && s isa IR.INDEX) || return nothing
  local range = literalRange(e.iterators[1].range)
  range === nothing && return nothing
  local value = (first(range) == 1 && step(range) == 1) ? s.exp :
    IR.BINARY(IR.INTEGER(first(range) - step(range)), IR.ADD, IR.BINARY(IR.INTEGER(step(range)), IR.MUL, s.exp, IR.T_INTEGER()), IR.T_INTEGER())
  local name = e.iterators[1].name
  return IR.mapExp((@nospecialize(x::IR.Expression)) ->
                     (x isa IR.CREF && length(x.cref.parts) == 1 && !IR.hasSubscripts(x.cref) &&
                      x.cref.name === name) ? value : x,
                   e.body)
end
