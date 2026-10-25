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
    local from::Union{Nothing, Tuple{String, Symbol}} = derivedFrom(v, variables)
    from === nothing && return current
    current = from[2]
  end
end

"""
    readsArrays(eq, arrays) -> Bool

Whether the equation `eq` reads one of the arrays `arrays` (by any subscripts, or whole).
"""
function readsArrays(eq::IR.Equation, arrays::Set{Symbol})::Bool
  local found::Bool = false
  IR.foreachExp(eq) do x::IR.Expression
    found && return nothing
    x isa IR.CREF || return nothing
    local name::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
    name in arrays && (found = true)
    nothing
  end
  return found
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

The equations of a partition with the arrays `split` split: an equation that reads one (or whose index is in
`unroll`, a for-equation differentiated unequally) unrolled into the element view's equations (`scalarEquations!`),
its references then in the split model's form (`splitReference`); another as it is.
"""
function splitEquations(equations::Vector{IR.Equation}, E::Elements, split::Set{Symbol}, unroll::Set{Int},
                        variables::IR.VariableSet)::Vector{IR.Equation}
  local out::Vector{IR.Equation} = IR.Equation[]
  for (i, eq) in enumerate(equations)
    if !(i in unroll) && !readsArrays(eq, split)
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
    splitArrays!(model, view, arrays, equations)

The arrays `arrays` of the arrays-kept `model` split into their element variables (the element view `view`'s), with
the arrays of their families (`familyBase`), and the equations that read them unrolled (`splitEquations`), with the
for-equations `equations` (indices in the partition); the initial equations likewise. The model changed in place.
"""
function splitArrays!(model::IR.Model, view::Scalarized, arrays::Vector{Symbol}, equations::Vector{Int})::Nothing
  local E::Elements = view.elements
  local bases::Set{Symbol} = Set{Symbol}(familyBase(a, model.variables) for a in arrays)
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
