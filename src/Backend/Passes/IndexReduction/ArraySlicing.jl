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
#= Slicing in the array form of a plan: an array whose elements the plan treats differently (a boundary element, the
   planar loops' positions of which two of three are differentiated) is split into its element variables, with the
   arrays derived from it or it from (`$DER.x` of x: one family), and the for-equations that read it unrolled; the
   other arrays and equations stay as they are, and the plan, made on the same element view, is committed again
   (`reduceIndexArrays!`). The element variables are the element view's (`scalarizeArrays`): their names and
   attributes those of the scalarized model. =#

"""
    familyBase(name, variables) -> Symbol

The array a backend-made array is made of, through its prefixes (`x` of `\$DER.\$DER.x`); the array itself for
another.
"""
function familyBase(name::Symbol, variables::IR.VariableSet)::Symbol
  local current::Symbol = name
  while true
    local v::Union{Nothing, IR.Variable} = get(variables, current, nothing)
    v === nothing && return current
    local from::Union{Nothing, Tuple{Vector{IR.CrefPart}, Symbol}} = derivedFrom(v, variables)
    from === nothing && return current
    current = from[2]
  end
end

"""
    readsArrays(eq, arrays, containers) -> Bool

Whether the equation `eq` reads one of the arrays `arrays` (by any subscripts, or whole), or a whole record or array
that contains one (`containers`, `arrayContainers`: a record argument `G[\$c].frame.R` whose field `R.T` is split).
"""
function readsArrays(eq::IR.Equation, arrays::Set{Symbol}, containers::Set{Symbol} = Set{Symbol}())::Bool
  local found::Bool = false
  IR.foreachExp(eq) do x::IR.Expression
    found && return nothing
    x isa IR.CREF || return nothing
    local name::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
    (name in arrays || (x.ty isa Union{IR.T_RECORD, IR.T_ARRAY} && name in containers)) && (found = true)
    nothing
  end
  return found
end

"The names that contain one of the arrays `arrays` (their prefixes up to a `.`: the records and components around them)."
function arrayContainers(arrays::Set{Symbol})::Set{Symbol}
  local out::Set{Symbol} = Set{Symbol}()
  for a in arrays
    local name::String = String(a)
    local depth::Int = 0
    for (i, c) in pairs(name)
      c == '[' && (depth += 1)
      c == ']' && (depth -= 1)
      (c == '.' && depth == 0) && push!(out, Symbol(name[1:prevind(name, i)]))
    end
  end
  return out
end

"""
    splitReference(x, E, split, variables) -> Expression

A reference of an unrolled equation (in the element view's form, `scalarEquations!`) in the split model's form: an
element of a split array its element variable's (the element view's reference), an element of another array a
reference into that array (`arrayReference`); a whole split array (a call's argument) the array of its elements.
"""
function splitReference(x::IR.Expression, E::Elements, split::Set{Symbol}, variables::IR.VariableSet)::IR.Expression
  x isa IR.CREF || return x
  local element::Union{Nothing, Tuple{Symbol, Vector{Int}}} = get(E.array, x.cref.name, nothing)
  element === nothing && return wholeSplitArray(x, E, split)
  element[1] in split && return IR.CREF(E.crefs[x.cref.name], x.ty)
  return arrayReference(x, E, variables)
end

"""
    wholeSplitArray(x, E, split) -> Expression

A reference to a whole split array of one dimension as the array of its element variables; another as it is.
NotHandled for a whole split array of more dimensions.
"""
function wholeSplitArray(x::IR.CREF, E::Elements, split::Set{Symbol})::IR.Expression
  (!IR.hasSubscripts(x.cref) && x.cref.name in split) || return x
  local sizes::Vector{Int} = E.sizes[x.cref.name]
  length(sizes) == 1 || throw(NotHandled("the whole split array $(x.cref.name) of $(length(sizes)) dimensions"))
  local element::IR.ModelicaType = IR.elementType(x.ty)
  return IR.ARRAY(IR.Expression[IR.CREF(E.crefs[n], element) for n in E.elements[x.cref.name]],
                  IR.T_ARRAY(element, IR.Dimension[IR.DIM_SIZE(sizes[1])]))
end

"""
    splitEquations(equations, E, split, unroll, variables) -> Vector{IR.Equation}

The equations of a partition with the arrays `split` split: an equation that reads one or a record around one (or
whose index is in `unroll`, a for-equation differentiated unequally) unrolled into the element view's equations
(`scalarEquations!`), its references then in the split model's form (`splitReference`); another as it is.
"""
function splitEquations(equations::Vector{IR.Equation}, E::Elements, split::Set{Symbol}, unroll::Set{Int},
                        variables::IR.VariableSet)::Vector{IR.Equation}
  local out::Vector{IR.Equation} = IR.Equation[]
  local containers::Set{Symbol} = arrayContainers(split)
  for (i, eq) in enumerate(equations)
    if !(i in unroll) && !readsArrays(eq, split, containers)
      push!(out, eq)
      continue
    end
    local unrolled::Vector{IR.Equation} = IR.Equation[]
    scalarEquations!(unrolled, EquationOrigin[], eq, EquationOrigin(i), NO_ITERATORS, E)
    foreach(u::IR.Equation -> push!(out, IR.mapEquation!(x::IR.Expression -> splitReference(x, E, split, variables), u)), unrolled)
  end
  return out
end

"""
    innerSplittable(model, name) -> Bool

Whether the array `name` of the model is one of rolled components (its first part rolled, `IR.isRolled`) of more
dimensions than the instances'.
"""
function innerSplittable(model::IR.Model, name::Symbol)::Bool
  local v::Union{Nothing, IR.Variable} = get(model.variables, name, nothing)
  v === nothing && return false
  return IR.isRolled(v.cref.parts[1].ty) && length(IR.dimensions(v.ty)) > 1
end

"""
    sliceCref(cref, t) -> ComponentRef

The reference of the slice `t` (the inner subscripts) of the array of reference `cref`: its last part of type
`T_SLICE`, named with them (`G.b.T[1,2]`), as an element of the instance is (`cylinder3.b.T[1,2]`); of a derivative
(`\$DER.G.b.T`) the derivative of the slice's.
"""
sliceCref(cref::IR.ComponentRef, t::Vector{Int})::IR.ComponentRef =
  IR.ComponentRef(IR.CrefPart[cref.parts[1:end - 1]; IR.CrefPart(cref.parts[end].name, IR.Subscript[], IR.T_SLICE(t, cref.parts[end].ty))])

"""
    splitInner!(model, bases)

The rolled arrays `bases`, with the arrays derived from them (`\$DER.`), split along their inner dimensions: one array
over the instances per inner subscript (`G.b.T[1,2]`, its elements `cylinder3.b.T[1,2]` as before), its kind, binding
and attributes the slice's; the references with literal inner subscripts the slices' elements, a reference by the
instance only the array literal of the slices; the for-equations whose iterators subscript an inner dimension unrolled
over them (`unrollIterators`). The model changed in place.
"""
function splitInner!(model::IR.Model, bases::Set{Symbol})::Nothing
  #= each family member's slices: name => inner sizes =#
  local family::Dict{Symbol, Vector{Int}} = Dict{Symbol, Vector{Int}}()
  for v in model.variables
    familyBase(v.name, model.variables) in bases || continue
    family[v.name] = knownSizes(v.ty, "the array $(v.name)")[2:end]
  end
  #= the members' references, their slices' named by them (`sliceCref`) =#
  local crefOf::Dict{Symbol, IR.ComponentRef} = Dict{Symbol, IR.ComponentRef}(v.name => v.cref for v in model.variables if haskey(family, v.name))
  local sliceOf::Function = (name::Symbol, t::Vector{Int}) -> sliceCref(crefOf[name], t).name
  #= the variables: each member by its slices, in its place =#
  local out::Vector{IR.Variable} = IR.Variable[]
  for v in model.variables
    if !haskey(family, v.name)
      push!(out, v)
      continue
    end
    local sizes::Vector{Int} = family[v.name]
    local n::Int = knownSizes(v.ty, "the array $(v.name)")[1]
    local element::IR.ModelicaType = IR.elementType(v.ty)
    for t in rowMajor(sizes)
      local cref::IR.ComponentRef = sliceCref(v.cref, t)
      local slice::Function = (x::Union{Nothing, IR.Expression}) -> sliceValue(x, n, t)
      local a::IR.VariableAttributes = v.attributes
      local attributes::IR.VariableAttributes = IR.VariableAttributes(slice(a.start), slice(a.fixed), slice(a.nominal), slice(a.min),
                                                                     slice(a.max), slice(a.unit), slice(a.displayUnit), slice(a.quantity),
                                                                     a.stateSelect)
      push!(out, IR.Variable(cref, IR.T_ARRAY(element, IR.Dimension[IR.DIM_SIZE(n)]), slicedKind(v.kind, family, t, sliceOf),
                             slice(v.binding), attributes, v.direction, v.isProtected, v.connector, v.source, v.origin))
    end
  end
  empty!(model.variables)
  foreach(v::IR.Variable -> push!(model.variables, v), out)
  local partition::IR.Partition = model.continuous[1]
  partition.equations = IR.Equation[slicedEquation(eq, family, model.variables) for eq in partition.equations]
  partition.initialEquations = IR.Equation[slicedEquation(eq, family, model.variables) for eq in partition.initialEquations]
  local unknowns::Vector{Symbol} = Symbol[]
  for u in partition.unknowns
    haskey(family, u) ? append!(unknowns, Symbol[sliceOf(u, t) for t in rowMajor(family[u])]) : push!(unknowns, u)
  end
  empty!(partition.unknowns); append!(partition.unknowns, unknowns)
  return nothing
end

"""
    slicedCref(cref, isVariable) -> Union{Nothing, IR.ComponentRef}

A reference to an element of an array split along its inner dimensions (`splitInner!`) as the reference to the
slice's element: `G[\$c].f.R.T[2, 1]` (an inlined body's field) is `G[\$c].f.R.T[2,1]` (the last part's literal
subscripts its slice's, `T_SLICE`), a derivative's likewise (`\$DER.G.f.R.T[\$c, 2, 1]` is `\$DER.G.f.R.T[2,1][\$c]`, on its last part).
`nothing` where no such slice is a variable (`isVariable` of its name).
"""
function slicedCref(cref::IR.ComponentRef, isVariable::Function)::Union{Nothing, IR.ComponentRef}
  local last::IR.CrefPart = cref.parts[end]
  for j in 0:length(last.subscripts) - 1
    local textual::Vector{IR.Subscript} = last.subscripts[j + 1:end]
    local inner::Union{Nothing, Vector{Int}} = IR.literalIndices(textual)
    inner === nothing && continue
    local candidate::IR.ComponentRef = IR.ComponentRef(IR.CrefPart[cref.parts[1:end - 1]; IR.CrefPart(last.name, last.subscripts[1:j], IR.T_SLICE(inner, last.ty))])
    isVariable(unsubscriptedName(candidate)) && return candidate
  end
  return nothing
end

"""
    sliceValue(x, n, t) -> Union{Nothing, IR.Expression}

A binding or attribute of a rolled array as its slice `t`'s: a scalar (each element's) as it is, an array literal by
its instances' elements at `t`, another by the instances' elements selected (`selectElement`).
"""
function sliceValue(x::Union{Nothing, IR.Expression}, n::Int, t::Vector{Int})::Union{Nothing, IR.Expression}
  x === nothing && return nothing
  local ty::IR.ModelicaType = IR.typeOf(x)
  ty isa IR.T_ARRAY || return x
  local element::IR.ModelicaType = IR.elementType(ty)
  local inner::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(IR.INTEGER(i)) for i in t]
  local values::Vector{IR.Expression} = IR.Expression[foldTree(selectElement(x, IR.Subscript[IR.INDEX(IR.INTEGER(k)); inner], element)) for k in 1:n]
  return IR.ARRAY(values, IR.T_ARRAY(element, IR.Dimension[IR.DIM_SIZE(n)]))
end

"The kind of a slice: the array's, the variable it names (a state's derivative, a derivative's state) that one's slice."
function slicedKind(kind::IR.VariableKind, family::Dict{Symbol, Vector{Int}}, t::Vector{Int}, sliceOf::Function)::IR.VariableKind
  local named::Function = (x::Symbol) -> haskey(family, x) ? sliceOf(x, t) : throw(NotHandled("a slice whose kind names $(x), not split with it"))
  #= a clocked state's previous value is not split with it =#
  kind isa IR.CLOCKED_STATE && return kind
  return IR.mapKindNames(named, kind)
end

"""
    slicedEquation(eq, family, variables) -> IR.Equation

An equation reading the split arrays `family` by their slices (`slicedReference`); its for-equations whose iterators
subscript an inner dimension unrolled first (`unrollIterators`).
"""
function slicedEquation(eq::IR.Equation, family::Dict{Symbol, Vector{Int}},
                        variables::IR.VariableSet)::IR.Equation
  local iterators::Set{Symbol} = Set{Symbol}()
  local reads::Bool = false
  IR.foreachExp(eq) do x::IR.Expression
    x isa IR.CREF || return nothing
    local name::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
    haskey(family, name) || return nothing
    reads = true
    local subscripts::Vector{IR.Subscript} = allSubscripts(x.cref)
    for s in subscripts[min(2, end + 1):end]
      IR.literalIndex(s) !== nothing && continue
      IR.foreachExp(y::IR.Expression -> ((y isa IR.CREF && length(y.cref.parts) == 1 && !IR.hasSubscripts(y.cref)) && push!(iterators, y.cref.name); nothing),
                    (s::IR.INDEX).exp)
    end
    nothing
  end
  reads || return eq
  local unrolled::Vector{IR.Equation} = isempty(iterators) ? IR.Equation[eq] : unrollIterators(eq, iterators)
  local mapped::Vector{IR.Equation} = IR.Equation[IR.mapEquation!(x::IR.Expression -> slicedReference(x, family), deepcopy(u)) for u in unrolled]
  length(mapped) == 1 && return mapped[1]
  #= several equations from one: kept together in a for-equation of one iteration =#
  return IR.FOR_EQUATION(IR.Iterator(Symbol("\$u"), IR.RANGE(IR.INTEGER(1), nothing, IR.INTEGER(1), IR.T_ARRAY(IR.T_INTEGER(), IR.Dimension[IR.DIM_SIZE(1)]))),
                         mapped, mapped[1].source, IR.DEFAULT_ATTRIBUTES)
end

"""
    unrollIterators(eq, iterators) -> Vector{IR.Equation}

The equation with its for-equations over `iterators` unrolled (their bodies once per value, `withIterators`), the
others kept around their bodies.
"""
function unrollIterators(eq::IR.Equation, iterators::Set{Symbol})::Vector{IR.Equation}
  eq isa IR.FOR_EQUATION || return IR.Equation[eq]
  local body::Vector{IR.Equation} = IR.Equation[b for x in eq.body for b in unrollIterators(x, iterators)]
  if eq.iterator.name in iterators
    local range::Union{Nothing, StepRange{Int, Int}} = literalRange(foldTree(eq.iterator.range))
    range === nothing && throw(NotHandled("a for-equation over $(eq.iterator.name) of a range not known, to unroll"))
    return IR.Equation[IR.mapEquation!(x::IR.Expression -> withIterators(x, Dict{Symbol, Int}(eq.iterator.name => k)), deepcopy(b))
                       for k in range for b in body]
  end
  return IR.Equation[IR.FOR_EQUATION(eq.iterator, body, eq.source, eq.attributes)]
end

"""
    slicedReference(x, family) -> IR.Expression

A reference to a split array by its slices: with literal inner subscripts the slice's element (the instance's
subscript kept), by the instance's subscript only the array literal of the slices' elements (picked where literal
subscripts follow); NotHandled for another (a whole array, an inner subscript not literal).
"""
function slicedReference(x::IR.Expression, family::Dict{Symbol, Vector{Int}})::IR.Expression
  #= the array literal of slices a part became, at literal subscripts (an unrolled `R.T[2, \$i1]`): the element =#
  (x isa IR.SUBSCRIPTED && x.exp isa IR.ARRAY && IR.literalIndices(x.subscripts) !== nothing) &&
    return foldTree(selectElement(x.exp, x.subscripts, x.ty))
  x isa IR.CREF || return x
  local name::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
  haskey(family, name) || return x
  local sizes::Vector{Int} = family[name]
  local subscripts::Vector{IR.Subscript} = allSubscripts(x.cref)
  isempty(subscripts) && throw(NotHandled("the whole array $(name), split along its inner dimensions"))
  local instance::IR.Subscript = subscripts[1]
  local element::IR.ModelicaType = IR.elementType(x.ty)
  local sliceRef::Function = (t::Vector{Int}) -> begin
    local last::IR.CrefPart = IR.unsubscriptedPart(x.cref.parts[end])
    local slice::IR.CrefPart = IR.CrefPart(last.name, IR.Subscript[], IR.T_SLICE(t, last.ty))
    #= a derivative's slice (one of one part too) subscripted on its last part (`IR.withLastSubscripts`), another's on
       its first (the instance) =#
    local cref::IR.ComponentRef = (IR.isDerivativeCref(x.cref) || length(x.cref.parts) == 1) ?
      IR.ComponentRef(IR.CrefPart[x.cref.parts[1:end - 1]; IR.CrefPart(slice.name, IR.Subscript[instance], slice.ty)]) :
      IR.ComponentRef(IR.CrefPart[IR.withSubscripts(x.cref.parts[1], IR.Subscript[instance]); x.cref.parts[2:end - 1]; slice])
    IR.CREF(cref, element)
  end
  if length(subscripts) == 1 + length(sizes)
    local t::Union{Nothing, Vector{Int}} = IR.literalIndices(subscripts[2:end])
    t === nothing && throw(NotHandled("the split array $(name) by an inner subscript not literal"))
    return sliceRef(t)
  end
  length(subscripts) == 1 || throw(NotHandled("the split array $(name) partly subscripted"))
  return nestedArray(sizes, element, IR.Expression[sliceRef(t) for t in rowMajor(sizes)])
end

"""
    splitArrays!(model, view, arrays, equations)

The arrays `arrays` of the arrays-kept `model` split into their element variables (the element view `view`'s), with
the arrays of their families (`familyBase`), and the equations that read them unrolled (`splitEquations`), with the
for-equations `equations` (indices in the partition); the initial equations likewise. The model changed in place.
"""
function splitArrays!(model::IR.Model, view::Scalarized, arrays::Vector{Symbol}, equations::Vector{Int})::Nothing
  local E::Elements = view.elements
  local bases::Set{Symbol} = Set{Symbol}(familyBase(a, model.variables) for a in arrays)
  #= a rolled array of more dimensions than the instances' (rolled components): along its inner dimensions only =#
  local inner::Set{Symbol} = Set{Symbol}(b for b in bases if innerSplittable(model, b))
  if !isempty(inner)
    splitInner!(model, inner)
    setdiff!(bases, inner)
    (isempty(bases) && isempty(equations)) && return nothing
    view = scalarizeArrays(model)
    E = view.elements
  end
  local split::Set{Symbol} = Set{Symbol}(n for n in keys(E.elements) if familyBase(n, model.variables) in bases)
  local partition::IR.Partition = model.continuous[1]
  local newEquations::Vector{IR.Equation} = splitEquations(partition.equations, E, split, Set{Int}(equations), model.variables)
  local newInitial::Vector{IR.Equation} = splitEquations(partition.initialEquations, E, split, Set{Int}(), model.variables)
  empty!(partition.equations); append!(partition.equations, newEquations)
  empty!(partition.initialEquations); append!(partition.initialEquations, newInitial)
  #= the variables: each split array by its elements, in its place =#
  local kept::Vector{IR.Variable} = IR.Variable[]
  for v in model.variables
    if v.name in split
      foreach(n::Symbol -> push!(kept, view.model.variables[n]), E.elements[v.name])
    else
      push!(kept, v)
    end
  end
  empty!(model.variables)
  foreach(v::IR.Variable -> push!(model.variables, v), kept)
  local unknowns::Vector{Symbol} = Symbol[]
  for n in partition.unknowns
    n in split ? append!(unknowns, E.elements[n]) : push!(unknowns, n)
  end
  empty!(partition.unknowns); append!(partition.unknowns, unknowns)
  return nothing
end
