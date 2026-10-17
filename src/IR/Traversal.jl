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

"""
    Traversal

The traversal of OMBackend's representations (OMJLIR, SimCode), written once: a node is
traversed field by field, in declaration order, and what a field's value is decides what happens
to it. Each representation says that per type, by methods of `traverseField`:
- a child (an expression, a statement, an equation) is handed to the traversal's
  `recur(child, arg) -> (newChild, newArg)`;
- a container (an iterator, a branch, a subscript, ...) holds children: it is traversed by
  `traverseFields` in turn;
- anything else is a leaf, kept (the default).
Every traversal of a representation (its children, one-level maps, maps with an accumulator,
substitution in statements and equations) follows from those few methods, so a new node type
needs none of its own.
"""
module Traversal

export traverseFields, traverseField, traverseElements

"""
    traverseFields(x, recur, arg) -> (x2, arg2)

`x` with each of its fields traversed by `traverseField(value, recur, arg)`, in declaration
order, `arg` threaded through. An immutable `x` (a value) is rebuilt by its default constructor
if a field changed, else returned itself (`===`); a mutable `x` (an entity, OMJLIR's
equations) has its changed fields set in place and is returned itself.
"""
@generated function traverseFields(x::T, recur::R, arg::A) where {T, R, A}
  local n = fieldcount(T)
  local values = [Symbol(:v, i) for i in 1:n]
  local traversals = [:(($(values[i]), a) = traverseField(getfield(x, $i), recur, a)) for i in 1:n]
  local unchanged = [:($(values[i]) === getfield(x, $i)) for i in 1:n]
  local result = if ismutabletype(T)
    quote
      $((:($(unchanged[i]) || setfield!(x, $i, $(values[i]))) for i in 1:n)...)
      x
    end
  else
    :($(foldl((l, r) -> :($l && $r), unchanged; init = true)) ? x : $T($(values...)))
  end
  return quote
    a = arg
    $(traversals...)
    return ($result, a)
  end
end

"""
    traverseField(value, recur, arg) -> (value2, arg2)

A field's value traversed (see `Traversal`): a leaf by default, kept. The representations add
methods for their children (`recur(value, arg)`), their containers (`traverseFields`) and
vectors of those (`traverseElements`).
"""
traverseField(value::Any, recur::R, arg::A) where {R, A} = (value, arg)

"""
    traverseElements(xs, recur, arg) -> (xs2, arg2)

The elements of `xs` traversed by `traverseField` in order, `arg` threaded through; `xs` itself
if none changed, else a new vector of the same element type.
"""
function traverseElements(xs::Vector{T}, recur::R, arg::A) where {T, R, A}
  local out::Union{Nothing, Vector{T}} = nothing
  local a = arg
  for i in eachindex(xs)
    local (y, a2) = traverseField(xs[i], recur, a)
    a = a2
    if out === nothing && y !== xs[i]
      out = similar(xs)
      copyto!(out, 1, xs, 1, i - 1)
    end
    out === nothing || (out[i] = y)
  end
  return (out === nothing ? xs : out, a)
end

end # Traversal
