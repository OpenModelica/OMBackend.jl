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
record array's; `\$DER.c[2].T` of the backend's `\$DER.c.T`).
"""
struct Elements
  elements::Dict{Symbol, Vector{Symbol}}
  sizes::Dict{Symbol, Vector{Int}}
  array::Dict{Symbol, Tuple{Symbol, Vector{Int}}}
  crefs::Dict{Symbol, IR.ComponentRef}
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

"The dimensions of an array type; NotHandled for a dimension of a size not known (`what` names it)."
function knownSizes(ty::IR.ModelicaType, what::String)::Vector{Int}
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

"`\$DER.` and the variable a backend-made variable (`\$DER.x`, `\$STATEDER.x`) is made of."
const DERIVED_NAME = r"^(\$[A-Z]+\.)(.+)$"

"""
    derivedFrom(v, variables) -> Union{Nothing, Tuple{String, Symbol}}

Of a backend-made array of one part (`\$DER.c.T`), its prefix and the array it is made of (`\$DER.`, `c.T`);
`nothing` for another variable.
"""
function derivedFrom(v::IR.Variable, variables::IR.VariableSet)::Union{Nothing, Tuple{String, Symbol}}
  length(v.cref.parts) == 1 || return nothing
  local m = match(DERIVED_NAME, String(v.name))
  m === nothing && return nothing
  local of = get(variables, Symbol(m.captures[2]), nothing)
  (of !== nothing && of.ty isa IR.T_ARRAY) || return nothing
  return (String(m.captures[1]), of.name)
end

"""
    modelElementCref(v, index) -> ComponentRef

The reference of the element `index` of the array variable `v` as the scalarized model names it: the subscripts
on the parts whose types are arrays (`c[2].p.T` of `c.p.T`, a field of the record array `c`).
"""
function modelElementCref(v::IR.Variable, index::Vector{Int})::IR.ComponentRef
  return distributedCref(v, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index])
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
array's by its reference (`modelElementCref`), a backend-made one's (`\$DER.x`) by its prefix and the elements of
the array it is made of (`\$DER.x[2]`, of one part), after that array's. Not an array whose elements are variables
already (a record field array of a parameter, `expandRecordFieldArrays`).
"""
function elements(variables::IR.VariableSet)::Elements
  local E = Elements(Dict{Symbol, Vector{Symbol}}(), Dict{Symbol, Vector{Int}}(), Dict{Symbol, Tuple{Symbol, Vector{Int}}}(),
                     Dict{Symbol, IR.ComponentRef}())
  local derived = Tuple{IR.Variable, String, Symbol}[]
  #= the arrays of rolled components first (their elements may be another array's, kept beside them: rollComponents!) =#
  local rolledFirst::Vector{IR.Variable} = IR.Variable[v for v in variables if v.cref.parts[1].ty isa IR.T_COMPONENTS]
  append!(rolledFirst, IR.Variable[v for v in variables if !(v.cref.parts[1].ty isa IR.T_COMPONENTS)])
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
    addDerivedElements!(E, v, prefix, of)

The elements of the backend-made array `v` (of one part, `\$DER.x`) into `E`: named by `prefix` and the elements of
the array `of` it is made of (`\$DER.x[2]`), which `E` has.
"""
function addDerivedElements!(E::Elements, v::IR.Variable, prefix::String, of::Symbol)::Nothing
  local sizes = knownSizes(v.ty, "the array $(v.name)")
  sizes == E.sizes[of] || throw(NotHandled("the array $(v.name) of other dimensions than $(of)'s"))
  local element = IR.elementType(v.ty)
  addElements!(E, v, sizes, IR.ComponentRef[IR.ComponentRef(Symbol(prefix, e), element) for e in E.elements[of]])
  return nothing
end

"The backend-made arrays `derived` (each with its prefix and the array it is made of) into `E`, each after the array it is made of (`\$DER.\$DER.x` after `\$DER.x`)."
function addDerivedArrays!(E::Elements, derived::Vector{Tuple{IR.Variable, String, Symbol}})::Nothing
  while !isempty(derived)
    local ready = findfirst(d -> haskey(E.elements, d[3]), derived)
    ready === nothing && throw(NotHandled("the arrays $(join((d[1].name for d in derived), ", ")), made of arrays not known"))
    local (v, prefix, of) = popat!(derived, ready)
    addDerivedElements!(E, v, prefix, of)
  end
  return nothing
end

"""
    extendedElements(E, added, variables) -> Elements

`E` with the elements of the backend-made arrays `added` (of `variables`, which has them and the arrays they are
made of), a new `Elements`; `E` not changed. NotHandled for an added array that is no backend-made one.
"""
function extendedElements(E::Elements, added::Vector{IR.Variable}, variables::IR.VariableSet)::Elements
  local out = Elements(copy(E.elements), copy(E.sizes), copy(E.array), copy(E.crefs))
  local derived = Tuple{IR.Variable, String, Symbol}[]
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

The indices a subscript selects of a dimension of size `n`: one (`x[2]`), several (`x[2:3]`, `x[{1, 3}]`, `x[:]`);
`nothing` where they are not literals (an iterator, a parameter).
"""
function selectedIndices(s::IR.INDEX, ::Int)::Union{Nothing, Int, Vector{Int}}
  s.exp isa IR.INTEGER || return nothing
  return s.exp.value
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
                          element::IR.ModelicaType)::IR.Expression
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
  sizes === nothing && return e
  local subscripts = allSubscripts(e.cref)
  length(subscripts) > length(sizes) && throw(NotHandled("the reference $(e.cref.name) of more subscripts than its array's dimensions"))
  local selection = Vector{Union{Int, Vector{Int}}}(undef, length(sizes))
  for d in eachindex(sizes)
    local picked = d <= length(subscripts) ? selectedIndices(subscripts[d], sizes[d]) : collect(1:sizes[d])
    picked === nothing && return e
    selection[d] = picked
  end
  return selectedElements(array, Int[], selection, E, IR.elementType(e.ty))
end

"""
    danglingReference(e, E) -> Union{Nothing, Symbol}

A reference in `e` to an array that the scalarized model has as elements only, by a subscript that is not literal
(`x[n]`, `n` a parameter): an equation may not keep one (an algorithm's loop may: `x[i]`). `nothing` if none.
"""
function danglingReference(e::IR.Expression, E::Elements)::Union{Nothing, Symbol}
  local found = Ref{Union{Nothing, Symbol}}(nothing)
  IR.foreachExp(e) do x
    (found[] === nothing && x isa IR.CREF && IR.hasSubscripts(x.cref) && literalIndices(allSubscripts(x.cref)) === nothing &&
     haskey(E.sizes, unsubscriptedName(x.cref))) && (found[] = x.cref.name)
    nothing
  end
  return found[]
end

"""
    constructorElement(e, k) -> Union{Nothing, Expression}

The `k`-th element of an array constructor of one iterator over a literal range (`{f(i) for i in 1:n}`): its
body at the iterator's `k`-th value; `nothing` for another constructor.
"""
function constructorElement(e::IR.ARRAY_CONSTRUCTOR, k::Int)::Union{Nothing, IR.Expression}
  length(e.iterators) == 1 || return nothing
  local values = literalRange(e.iterators[1].range)
  (values === nothing || !(1 <= k <= length(values))) && return nothing
  return withIterator(e.body, e.iterators[1].name, values[k])
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
subscript in turn; `cos({a, b})[2]` is `cos(b)`, an element-wise built-in); subscripted where it is not.
"""
function pickElement(e::IR.Expression, subscripts::Vector{IR.Subscript}, ty::IR.ModelicaType, E::Elements)::IR.Expression
  local index = literalIndices(subscripts)
  if e isa IR.ARRAY_CONSTRUCTOR && index !== nothing
    local body = constructorElement(e, index[1])
    if body !== nothing
      local element = foldTree(scalarExpression(body, NO_ITERATORS, E))
      return length(subscripts) == 1 ? element : pickElement(element, subscripts[2:end], ty, E)
    end
  end
  if e isa IR.CALL && e.attributes.builtin && e.name in ELEMENTWISE_BUILTINS && length(e.arguments) == 1 &&
     IR.typeOf(e.arguments[1]) isa IR.T_ARRAY
    return IR.CALL(e.name, IR.Expression[pickElement(e.arguments[1], subscripts, ty, E)], ty, e.attributes)
  end
  return foldTree(selectElement(e, subscripts, ty))
end

"The built-in functions of one argument applied element-wise to an array (MLS 12.4.6)."
const ELEMENTWISE_BUILTINS = (:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :exp, :log, :log10, :sqrt,
                              :abs, :sign, :floor, :ceil)

"""
    scalarExpression(e, values, E) -> Expression

`e` in the scalarized model: the iterators of `values` replaced by their values (`withIterators`), references to
arrays by their element variables (`elementReference`), an element picked out of an array value where its
subscripts are literals (`pickElement`).
"""
function scalarExpression(e::IR.Expression, values::Dict{Symbol, Int}, E::Elements)::IR.Expression
  #= an element of an array literal (a table of pivots, a binding) picked before its other elements are mapped =#
  local pruned = IR.mapExpTopDown(withIterators(e, values)) do x
    (x isa IR.SUBSCRIPTED && x.exp isa IR.ARRAY && literalIndices(x.subscripts) !== nothing) || return (x, true)
    return (selectElement(x.exp, x.subscripts, x.ty), true)
  end
  return IR.mapExp(pruned) do x
    x isa IR.CREF && return elementReference(x, E)
    (x isa IR.SUBSCRIPTED && literalIndices(x.subscripts) !== nothing) && return pickElement(x.exp, x.subscripts, x.ty, E)
    (x isa IR.BINARY && x.op == IR.SCALAR_PRODUCT) && return expandedScalarProduct(x, E)
    return x
  end
end

"""
    expandedScalarProduct(e, E) -> Expression

A scalar product of two vectors of a known size in the scalarized model: the sum of the products of their elements
(`pickElement`), as the frontend's scalarization writes it (MultiBody's `tau = -frame_b.t * e`); as it is where the
size is not known.
"""
function expandedScalarProduct(e::IR.BINARY, E::Elements)::IR.Expression
  local ty1::IR.ModelicaType = IR.typeOf(e.exp1)
  (ty1 isa IR.T_ARRAY && length(ty1.dimensions) == 1 && ty1.dimensions[1] isa IR.DIM_SIZE) || return e
  local element::IR.ModelicaType = IR.elementType(ty1)
  local terms::Vector{IR.Expression} = IR.Expression[]
  for k in 1:ty1.dimensions[1].size
    local index::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(IR.INTEGER(k))]
    push!(terms, times(pickElement(e.exp1, index, element, E), pickElement(e.exp2, index, element, E)))
  end
  isempty(terms) && return zeroOf(e)
  return foldTree(foldl(plus, terms))
end

"""
    equationExpression(e, values, E) -> Expression

`scalarExpression` of an expression of an equation, which may not keep a reference to an array by a subscript
that is not literal (`danglingReference`): NotHandled for one.
"""
function equationExpression(e::IR.Expression, values::Dict{Symbol, Int}, E::Elements)::IR.Expression
  local scalar = scalarExpression(e, values, E)
  local dangling = danglingReference(scalar, E)
  dangling === nothing || throw(NotHandled("the reference $(dangling) to an array by a subscript that is not constant"))
  return scalar
end

"""
    ElementValue

An attribute or binding of an array variable, for its elements (`elementValue`, `valueAt`): `EachValue` the same
for every element, `ArrayValue` an array value picked per element, `ComponentValue` a component array's value of its
fields per element.
"""
abstract type ElementValue end

"The same value for every element (`each start = 0`, `each start = sum(m)`), scalarized once."
struct EachValue <: ElementValue
  value::IR.Expression
end

"An array value of the variable's dimensions, scalarized once (`scalarExpression`), its element picked per element."
struct ArrayValue <: ElementValue
  value::IR.Expression
  ty::IR.T_ARRAY
end

"""
    ComponentValue(value, prefix, sizes)

A value of a field of a component array written over the component's fields (`p.L * cos(p.angle)` of `p.x`, as the
frontend keeps it for `p[3]`): for each element its fields' elements (`componentElement`); `prefix` the component's
parts, `sizes` its dimensions.
"""
struct ComponentValue <: ElementValue
  value::IR.Expression
  prefix::Vector{Symbol}
  sizes::Vector{Int}
end

elementValue(::Nothing, ::IR.Variable, ::Elements)::Nothing = nothing

"""
    elementValue(a, v, E) -> ElementValue

The attribute or binding `a` of the array variable `v` for its elements: an array value of `v`'s dimensions an
`ArrayValue`; a scalar one reading whole fields of `v`'s own component array a `ComponentValue`; another scalar one
an `EachValue` (read as written). NotHandled for an array value of other dimensions, and for a component value that
reduces a field (`sum(p.w)`).
"""
function elementValue(a::IR.Expression, v::IR.Variable, E::Elements)::ElementValue
  local sizes = E.sizes[v.name]
  local ty = IR.typeOf(a)
  if ty isa IR.T_ARRAY
    local dims = Int[something(IR.dimensionSize(d), -1) for d in IR.dimensions(ty)]
    dims == sizes || throw(NotHandled("an attribute of $(v.name) of dimensions $(dims), the variable's $(sizes)"))
    return ArrayValue(scalarExpression(a, NO_ITERATORS, E), ty)
  end
  local prefix = componentPrefix(v.cref)
  (prefix !== nothing && IR.anyExp(x -> isComponentField(x, prefix, sizes, E), a)) ||
    return EachValue(scalarExpression(a, NO_ITERATORS, E))
  IR.anyExp(x -> x isa IR.CALL && x.attributes.builtin && x.name in REDUCING_BUILTINS, a) &&
    throw(NotHandled("an attribute of the component array field $(v.name) that reduces an array ($(IR.toString(a)))"))
  return ComponentValue(a, prefix, sizes)
end

"The built-in functions that reduce an array or read its shape: a component value reading fields through one is not componentwise."
const REDUCING_BUILTINS = (:sum, :product, :min, :max, :size, :ndims, :scalar, :vector, :matrix, :transpose, :cross, :skew,
                           :identity, :diagonal, :fill, :zeros, :ones, :linspace, :cat, :array)

"""
    componentPrefix(cref) -> Union{Nothing, Vector{Symbol}}

Of a field of a component array (the dimensions on a part before the last: `p.x` of `p[3]`), the names of the parts
up to the component's (`[:p]`); `nothing` for another reference (an array of one part, a field array of a scalar
component).
"""
function componentPrefix(cref::IR.ComponentRef)::Union{Nothing, Vector{Symbol}}
  local parts = cref.parts
  local k = findlast(p -> !isempty(IR.dimensions(p.ty)), parts[1:end-1])
  k === nothing && return nothing
  return Symbol[p.name for p in parts[1:k]]
end

"Whether `x` is a whole reference to a field of the component array `prefix` of dimensions `sizes`."
isComponentField(x::IR.Expression, prefix::Vector{Symbol}, sizes::Vector{Int}, E::Elements)::Bool =
  x isa IR.CREF && !IR.hasSubscripts(x.cref) && get(E.sizes, x.cref.name, nothing) == sizes && componentPrefix(x.cref) == prefix

"""
    valueAt(v, index, E) -> Union{Nothing, Expression}

The value for the element `index` (`ElementValue`); `nothing` for no value. NotHandled for one the element view
cannot scalarize (a reference by a subscript that is not literal, an element-wise built-in of an array left).
"""
valueAt(::Nothing, ::Vector{Int}, ::Elements)::Nothing = nothing
valueAt(v::EachValue, ::Vector{Int}, E::Elements)::IR.Expression = checkedValue(v.value, E)
valueAt(v::ArrayValue, index::Vector{Int}, E::Elements)::IR.Expression =
  checkedValue(pickElement(v.value, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index], IR.elementType(v.ty), E), E)
valueAt(v::ComponentValue, index::Vector{Int}, E::Elements)::IR.Expression = checkedValue(componentElement(v, index, E), E)

"`value`, scalarized for an element; NotHandled where it is not (`valueAt`)."
function checkedValue(value::IR.Expression, E::Elements)::IR.Expression
  local dangling = danglingReference(value, E)
  dangling === nothing || throw(NotHandled("an attribute reading $(dangling) by a subscript that is not constant"))
  IR.anyExp(x -> x isa IR.CALL && x.attributes.builtin && x.name in ELEMENTWISE_BUILTINS &&
                 any(a -> IR.typeOf(a) isa IR.T_ARRAY, x.arguments), value) &&
    throw(NotHandled("an attribute of an element-wise built-in of an array ($(IR.toString(value)))"))
  return value
end

"""
    componentElement(v, index, E) -> Expression

The component value `v` for the element `index`: each whole field of its component array by its element at `index`
(`isComponentField`), then scalarized (`scalarExpression`). NotHandled for a whole array left (another array, a field
of another size).
"""
function componentElement(v::ComponentValue, index::Vector{Int}, E::Elements)::IR.Expression
  local selection = Union{Int, Vector{Int}}[k for k in index]
  local picked = IR.mapExp(v.value) do x
    isComponentField(x, v.prefix, v.sizes, E) || return x
    return selectedElements(x.cref.name, Int[], selection, E, IR.elementType(x.ty))
  end
  IR.anyExp(x -> x isa IR.CREF && !IR.hasSubscripts(x.cref) && haskey(E.sizes, x.cref.name), picked) &&
    throw(NotHandled("a component array's attribute reading another whole array ($(IR.toString(v.value)))"))
  return scalarExpression(picked, NO_ITERATORS, E)
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
elementKind(kind::IR.STATE, index::Vector{Int}, E::Elements)::IR.VariableKind =
  IR.STATE(kind.index, kind.derivative === nothing ? nothing : elementName(kind.derivative, index, E), kind.natural)
elementKind(kind::IR.STATE_DER, index::Vector{Int}, E::Elements)::IR.VariableKind = IR.STATE_DER(elementName(kind.state, index, E))
elementKind(kind::IR.DUMMY_DER, index::Vector{Int}, E::Elements)::IR.VariableKind = IR.DUMMY_DER(elementName(kind.dummyState, index, E))
elementKind(kind::IR.DUMMY_STATE, index::Vector{Int}, E::Elements)::IR.VariableKind =
  IR.DUMMY_STATE(elementName(kind.dummyDerivative, index, E))
elementKind(kind::IR.CLOCKED_STATE, index::Vector{Int}, E::Elements)::IR.VariableKind =
  IR.CLOCKED_STATE(elementName(kind.previous, index, E), kind.isStartFixed)
elementKind(kind::IR.VariableKind, ::Vector{Int}, ::Elements)::IR.VariableKind = kind

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
    element isa Union{IR.T_REAL, IR.T_INTEGER, IR.T_BOOLEAN, IR.T_ENUMERATION, IR.T_STRING} ||
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
unlinked(a::IR.EquationAttributes)::IR.EquationAttributes = IR.EquationAttributes(a.kind, a.differentiated, nothing)

"""
    scalarEquations!(out, origins, eq, origin, values, E)

The equation `eq` (at the iterator `values`) as scalar equations into `out`, each with its `origin` into
`origins`: a for-equation unrolled over its literal range, an if-equation of literal conditions as its branch
taken, an array equation one per element; references by the element variables (`equationExpression`; statements
by `scalarExpression`). NotHandled for a for-equation over a range that is not constant and a record equation.
"""
function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.FOR_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local range = literalRange(scalarExpression(eq.iterator.range, values, E))
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
  push!(out, IR.SCALAR_EQUATION(equationExpression(eq.lhs, values, E), equationExpression(eq.rhs, values, E), eq.ty,
                                eq.source, unlinked(eq.attributes)))
  push!(origins, origin)
  return nothing
end

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.ARRAY_EQUATION,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local (lhs, rhs) = (equationExpression(eq.lhs, values, E), equationExpression(eq.rhs, values, E))
  local element = IR.elementType(eq.ty)
  for index in rowMajor(knownSizes(eq.ty, "an array equation"))
    local subscripts = IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index]
    push!(out, IR.SCALAR_EQUATION(pickElement(lhs, subscripts, element, E), pickElement(rhs, subscripts, element, E), element,
                                  eq.source, unlinked(eq.attributes)))
    push!(origins, EquationOrigin(origin.equation, origin.path, origin.iteration, index))
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
  local branches = IR.WhenBranch[IR.WhenBranch(IR.Expression[equationExpression(t, values, E) for t in b.triggers], b.atInitial,
                                               b.atTerminal, scalarStatements(b.body, values, E)) for b in eq.branches]
  push!(out, IR.WHEN_EQUATION(branches, eq.kind, eq.source, unlinked(eq.attributes)))
  push!(origins, origin)
  return nothing
end

function scalarEquations!(out::Vector{IR.Equation}, origins::Vector{EquationOrigin}, eq::IR.ALGORITHM,
                          origin::EquationOrigin, values::Dict{Symbol, Int}, E::Elements)::Nothing
  local outputs = IR.ComponentRef[]
  for c in eq.outputs
    pushReferences!(outputs, elementReference(IR.CREF(withIteratorCref(c, values), IR.T_UNKNOWN()), E))
  end
  push!(out, IR.ALGORITHM(scalarStatements(eq.statements, values, E), outputs, eq.source, unlinked(eq.attributes)))
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

scalarEquations!(::Vector{IR.Equation}, ::Vector{EquationOrigin}, eq::IR.Equation, ::EquationOrigin, ::Dict{Symbol, Int},
                 ::Elements)::Nothing = throw(NotHandled("a $(nameof(typeof(eq))) with arrays kept"))

"`cref` with the iterators of `values` replaced by their values, its subscripts folded."
withIteratorCref(cref::IR.ComponentRef, values::Dict{Symbol, Int})::IR.ComponentRef =
  withIterators(IR.CREF(cref, IR.T_UNKNOWN()), values).cref

"""
    scalarStatements(body, values, E) -> Vector{Statement}

Statements with their root expressions scalarized (`scalarExpression`; a loop's `x[i]` kept), a `reinit`'s state
by its element variable.
"""
function scalarStatements(body::Vector{IR.Statement}, values::Dict{Symbol, Int}, E::Elements)::Vector{IR.Statement}
  local g = e::IR.Expression -> scalarExpression(e, values, E)
  return IR.Statement[scalarReinit(mapRoots(g, s), values, E) for s in body]
end

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
    scalarizeArrays(model) -> Scalarized

`model` with its arrays as their elements (`Scalarized`): its variables (`scalarVariables`), the equations and
initial equations of its continuous partition (`scalarEquations!`). The model is not changed. NotHandled for a
model with several continuous partitions, clocked partitions or structural modes, and for what
`scalarEquations!` does not handle.
"""
function scalarizeArrays(model::IR.Model)::Scalarized
  (length(model.continuous) == 1 && isempty(model.clocked) && isempty(model.modes)) ||
    throw(NotHandled("arrays in a model of several partitions or structural modes"))
  local E = elements(model.variables)
  local variables = scalarVariables(model.variables, E)
  local p = model.continuous[1]
  local equations = IR.Equation[]
  local origins = EquationOrigin[]
  for (i, eq) in enumerate(p.equations)
    scalarEquations!(equations, origins, eq, EquationOrigin(i), NO_ITERATORS, E)
  end
  local initialEquations = IR.Equation[]
  for (i, eq) in enumerate(p.initialEquations)
    scalarEquations!(initialEquations, EquationOrigin[], eq, EquationOrigin(i), NO_ITERATORS, E)
  end
  local unknowns = Symbol[]
  for n in p.unknowns
    haskey(E.elements, n) ? append!(unknowns, E.elements[n]) : push!(unknowns, n)
  end
  local partition = IR.Partition(p.index, p.association, unknowns, equations, initialEquations)
  local scalar = IR.Model(model.name, variables, IR.Partition[partition], model.clocked, model.baseClocks, model.functions,
                          model.initialMode, model.modes, model.transitions, model.frontendModel)
  return Scalarized(scalar, origins, E)
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
    rollArrayEquations!(model) -> Model

The array equations of `model`'s continuous partitions (arrays kept) that read a continuous array as for-equations
over their elements (`nestedFor`), before `expandEquations` would split them into one equation per element:
`phi2 = phi1` is `for \$i1 in 1:n loop phi2[\$i1] = phi1[\$i1]; end for`, a side that is no variable subscripted
(`(2 * der(w))[\$i1]`). Array equations of unknown dimensions and those of no continuous array stay.
"""
function rollArrayEquations!(model::IR.Model)::IR.Model
  for p in model.continuous
    p.equations = IR.Equation[rolledEquation(eq, model.variables) for eq in p.equations]
  end
  return model
end

rolledEquation(eq::IR.Equation, ::IR.VariableSet)::IR.Equation = eq

function rolledEquation(eq::IR.ARRAY_EQUATION, variables::IR.VariableSet)::IR.Equation
  local sizes = Int[]
  for d in eq.ty.dimensions
    local n = IR.dimensionSize(d)
    n === nothing && return eq
    push!(sizes, n)
  end
  (readsContinuousArray(eq.lhs, variables) || readsContinuousArray(eq.rhs, variables)) || return eq
  local subscripts = iteratorSubscripts(sizes)
  local element = IR.elementType(eq.ty)
  local body = IR.SCALAR_EQUATION(elementAt(eq.lhs, subscripts, element, variables), elementAt(eq.rhs, subscripts, element, variables),
                                  element, eq.source, eq.attributes)
  return nestedFor(IR.Equation[body], sizes)
end

"Whether `e` reads a continuous array variable (a state, a derivative, an algebraic array), whole or by an element."
readsContinuousArray(e::IR.Expression, variables::IR.VariableSet)::Bool =
  IR.anyExp(x -> x isa IR.CREF && isContinuousArray(get(variables, unsubscriptedName(x.cref), nothing)), e)

isContinuousArray(::Nothing)::Bool = false
isContinuousArray(v::IR.Variable)::Bool = v.ty isa IR.T_ARRAY && v.kind isa Union{IR.STATE, IR.STATE_DER, IR.ALGEBRAIC}

"""
    elementAt(e, subscripts, element, variables) -> Expression

The element of the array expression `e` at `subscripts` (the iterators of a rolled array equation), distributed over
element-wise arithmetic (`selectElement`): a whole array variable's element reference (`distributedCref`), an array
constructor's body at the iterators (`constructorAt`: `{sin(time) * i for i in 1:n}` is `sin(time) * \$i1`),
another expression subscripted.
"""
function elementAt(e::IR.Expression, subscripts::Vector{IR.Subscript}, element::IR.ModelicaType,
                   variables::IR.VariableSet)::IR.Expression
  return IR.mapExp(selectElement(e, subscripts, element)) do x
    (x isa IR.SUBSCRIPTED && x.subscripts == subscripts) || return x
    if x.exp isa IR.CREF && !IR.hasSubscripts(x.exp.cref)
      local v = get(variables, x.exp.cref.name, nothing)
      (v !== nothing && v.ty isa IR.T_ARRAY && length(IR.dimensions(v.ty)) == length(subscripts)) &&
        return IR.CREF(distributedCref(v, subscripts), x.ty)
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
  return IR.mapExp(x -> (x isa IR.CREF && length(x.cref.parts) == 1 && !IR.hasSubscripts(x.cref) && x.cref.name === name) ? value : x,
                   e.body)
end
