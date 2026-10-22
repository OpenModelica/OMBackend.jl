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

#= The elements and fields of OMJLIR expressions: an element or a field picked from a record, an
   array literal or a built-in's result (`selectElement`), a subscripted expression where it cannot
   be; the references to an array's elements; the shape of an expression; arrays of zeros. =#

function zeroArray(sizes::Vector{Int}, element::IR.ModelicaType)::IR.Expression
  isempty(sizes) && return ZERO
  local inner = sizes[2:end]
  local ty = IR.T_ARRAY(element, IR.Dimension[IR.DIM_SIZE(n) for n in sizes])
  return IR.ARRAY(IR.Expression[zeroArray(inner, element) for _ in 1:sizes[1]], ty)
end

"""
    shapeOf(e) -> Union{Nothing, Vector{Int}}

The dimensions of `e` from its operands where its type does not give them (an inlined body's
types are its function's, `Real[size(e, 1), 1]`): an array literal's, a reference's, an
element-wise operation's, a matrix product's, promote's, transpose's, identity's; `Int[]` for a
scalar; `nothing` where not known.
"""
function shapeOf(e::IR.Expression)::Union{Nothing, Vector{Int}}
  local ty = IR.typeOf(e)
  ty isa IR.T_ARRAY || return Int[]
  local typed = Int[]
  for d in IR.dimensions(ty)
    local n = IR.dimensionSize(d)
    n === nothing && (typed = nothing; break)
    push!(typed, n)
  end
  typed === nothing || return typed
  if e isa IR.ARRAY
    isempty(e.elements) && return nothing
    local inner = shapeOf(e.elements[1])
    return inner === nothing ? nothing : vcat(length(e.elements), inner)
  elseif e isa IR.BINARY
    local (a, b) = (shapeOf(e.exp1), shapeOf(e.exp2))
    (a === nothing || b === nothing) && return nothing
    e.op === IR.MATRIX_PRODUCT && return (length(a) == 2 && length(b) == 2) ? [a[1], b[2]] : nothing
    e.op === IR.MUL_MATRIX_VECTOR && return length(a) == 2 ? [a[1]] : nothing
    e.op === IR.MUL_VECTOR_MATRIX && return length(b) == 2 ? [b[2]] : nothing
    return isempty(a) ? b : a
  elseif e isa IR.UNARY
    return shapeOf(e.exp)
  elseif e isa IR.CALL && e.attributes.builtin
    local args = e.arguments
    if e.name === :promote && length(args) == 2 && args[2] isa IR.INTEGER
      local a = shapeOf(args[1])
      a === nothing && return nothing
      return vcat(a, ones(Int, max(0, args[2].value - length(a))))
    elseif e.name === :transpose && length(args) == 1
      local a = shapeOf(args[1])
      return (a === nothing || length(a) != 2) ? nothing : [a[2], a[1]]
    elseif e.name === :identity && length(args) == 1 && args[1] isa IR.INTEGER
      return [args[1].value, args[1].value]
    elseif e.name in (:skew,)
      return [3, 3]
    end
  end
  return nothing
end

"""
    zeroOf(e) -> Union{Nothing, Expression}

The zero of `e`'s shape (`shapeOf`): `0.0` or an array of zeros; `nothing` where the shape is
not known (the caller keeps its expression).
"""
function zeroOf(e::IR.Expression)::Union{Nothing, IR.Expression}
  local shape = shapeOf(e)
  shape === nothing && return nothing
  return zeroArray(shape, IR.T_REAL())
end

"""
    selectElement(e, subscripts, ty) -> Expression

`e[subscripts]`, picked where `e` is a record or an array literal and the subscripts are
literals (the frontend reads a field of a record-valued call by its position, `f(x)[1]`; the
derivative of the call is a record literal); a subscripted expression otherwise.
"""
function selectElement(e::IR.Expression, subscripts::Vector{IR.Subscript}, ty::IR.ModelicaType)::IR.Expression
  isempty(subscripts) && return e
  local first = subscripts[1]
  if first isa IR.INDEX && first.exp isa IR.INTEGER
    local k = first.exp.value
    local record = asRecord(e)
    if record !== nothing && length(subscripts) == 1 && 1 <= k <= length(record.elements)
      return record.elements[k]
    end
    if e isa IR.ARRAY && 1 <= k <= length(e.elements)
      return selectElement(e.elements[k], subscripts[2:end], ty)
    end
  end
  local distributed = ty isa IR.T_REAL ? scalarElement(e, subscripts, ty) : nothing
  distributed === nothing || return distributed
  local picked = IR.SUBSCRIPTED(e, subscripts, ty)
  local z = isZeroExp(e) ? zeroOf(picked) : nothing
  return z === nothing ? picked : z
end

"""
    scalarElement(e, subscripts, ty) -> Union{Nothing, Expression}

A scalar element of element-wise or array-scalar arithmetic (`({0, 0, 1} * w)[3]` is `1 * w`,
folded: `w`), the operands' elements picked (of `transpose(A)[i, j]`, `A[j, i]`; of an
if-expression, each branch's; of a cast, its operand's), or of a matrix-matrix, matrix-vector or
vector-matrix product, the sum over the inner index (`(R_rel.T * {{0, 0, 0}, ..., {0, 0, 1}})[1, 1]`,
a MultiBody frame's partial derivative); `nothing` for another expression.
"""
function scalarElement(e::IR.Expression, subscripts::Vector{IR.Subscript}, ty::IR.ModelicaType)::Union{Nothing, IR.Expression}
  local pick = x::IR.Expression -> selectElement(x, subscripts, ty)
  if e isa IR.BINARY
    e.op === IR.MUL_ARRAY_SCALAR && return times(pick(e.exp1), e.exp2)
    e.op === IR.MUL_SCALAR_ARRAY && return times(e.exp1, pick(e.exp2))
    e.op === IR.DIV_ARRAY_SCALAR && return divide(pick(e.exp1), e.exp2)
    e.op === IR.ADD_ARRAY_SCALAR && return plus(pick(e.exp1), e.exp2)
    e.op === IR.SUB_ARRAY_SCALAR && return minus(pick(e.exp1), e.exp2)
    e.op === IR.ADD_SCALAR_ARRAY && return plus(e.exp1, pick(e.exp2))
    e.op === IR.SUB_SCALAR_ARRAY && return minus(e.exp1, pick(e.exp2))
    e.op in (IR.ADD, IR.ADD_EW) && return plus(pick(e.exp1), pick(e.exp2))
    e.op in (IR.SUB, IR.SUB_EW) && return minus(pick(e.exp1), pick(e.exp2))
    e.op === IR.MUL_EW && return times(pick(e.exp1), pick(e.exp2))
    e.op === IR.DIV_EW && return divide(pick(e.exp1), pick(e.exp2))
    e.op in (IR.MATRIX_PRODUCT, IR.MUL_MATRIX_VECTOR, IR.MUL_VECTOR_MATRIX) && return productElement(e, subscripts, ty)
  elseif e isa IR.UNARY && e.op === IR.UMINUS
    return negate(pick(e.exp))
  elseif e isa IR.CALL && e.attributes.builtin && e.name === :transpose && length(subscripts) == 2
    return selectElement(only(e.arguments), IR.Subscript[subscripts[2], subscripts[1]], ty)
  elseif e isa IR.IF_EXPRESSION
    return ifThenElse(e.condition, pick(e.thenExp), pick(e.elseExp), ty)
  elseif e isa IR.CAST
    return pick(e.exp)
  elseif e isa IR.CALL && e.attributes.builtin
    return builtinElement(e, subscripts, ty)
  end
  return nothing
end

"""
    builtinElement(e, subscripts, ty) -> Union{Nothing, Expression}

A scalar element of an array built-in of literal subscripts: `identity(n)[i, j]` 1 or 0,
`promote(a, n)[i, 1]` `a[i]` (the added dimensions of size 1), `fill(x, ...)[...]` x,
`cat(d, a, b, ...)` the element of the argument whose span along d holds the index;
`nothing` for another call.
"""
function builtinElement(e::IR.CALL, subscripts::Vector{IR.Subscript}, ty::IR.ModelicaType)::Union{Nothing, IR.Expression}
  local index = Int[]
  for sub in subscripts
    (sub isa IR.INDEX && sub.exp isa IR.INTEGER) || return nothing
    push!(index, sub.exp.value)
  end
  if e.name === :identity && length(index) == 2
    return index[1] == index[2] ? ONE : ZERO
  elseif e.name === :promote && length(e.arguments) == 2
    local k = length(IR.dimensions(IR.typeOf(e.arguments[1])))
    (k <= length(index) && all(==(1), index[k + 1:end])) || return nothing
    return selectElement(e.arguments[1], subscripts[1:k], ty)
  elseif e.name === :fill && length(index) == length(e.arguments) - 1
    return e.arguments[1]
  elseif e.name === :cat && e.arguments[1] isa IR.INTEGER
    #= the argument whose span along the dimension holds the index, the index shifted into it =#
    local d = e.arguments[1].value
    1 <= d <= length(index) || return nothing
    local offset = 0
    for a in e.arguments[2:end]
      local dims = IR.dimensions(IR.typeOf(a))
      local n = length(dims) >= d ? IR.dimensionSize(dims[d]) : nothing
      n === nothing && return nothing
      if index[d] <= offset + n
        local shifted = copy(subscripts)
        shifted[d] = IR.INDEX(IR.INTEGER(index[d] - offset))
        return selectElement(a, shifted, ty)
      end
      offset += n
    end
  end
  return nothing
end

"""
    productElement(e, subscripts, ty) -> Union{Nothing, Expression}

The scalar element `subscripts` of the matrix or matrix-vector product `e`: the sum over the
inner index k of the left operand's element (row, k) times the right's (k, column); `nothing`
where the inner dimension is not known or the subscripts do not pick a scalar.
"""
function productElement(e::IR.BINARY, subscripts::Vector{IR.Subscript}, ty::IR.ModelicaType)::Union{Nothing, IR.Expression}
  local dims = IR.dimensions(IR.typeOf(e.exp1))
  local n = isempty(dims) ? nothing : IR.dimensionSize(last(dims))
  n === nothing && return nothing
  local (left, right) = if e.op === IR.MATRIX_PRODUCT
    length(subscripts) == 2 || return nothing
    (subscripts[1:1], subscripts[2:2])
  else
    length(subscripts) == 1 || return nothing
    e.op === IR.MUL_MATRIX_VECTOR ? (subscripts, IR.Subscript[]) : (IR.Subscript[], subscripts)
  end
  local sum = ZERO
  for k in 1:n
    local inner = IR.Subscript[IR.INDEX(IR.INTEGER(k))]
    sum = plus(sum, times(selectElement(e.exp1, vcat(left, inner), ty), selectElement(e.exp2, vcat(inner, right), ty)))
  end
  return sum
end

"""
    expandWhole(e) -> Expression

A reference to a whole array or record (an input's field `R.T` in an inlined body) as the array
or record literal of its elements' references (`{{R.T[1,1], ...}, ...}`): in a scalarized model
only the elements are variables, as the frontend writes them (`{e[1], e[2], e[3]}`). Other
expressions as they are; an array of unknown size stays a reference.
"""
function expandWhole(e::IR.Expression)::IR.Expression
  e isa IR.CREF || return e
  local ty = e.ty
  if ty isa IR.T_RECORD
    return IR.RECORD(IR.Expression[expandWhole(fieldOf(e, f.name, f.ty)) for f in ty.fields], ty)
  elseif ty isa IR.T_ARRAY
    local sizes = Int[]
    for d in IR.dimensions(ty)
      local n = IR.dimensionSize(d)
      n === nothing && return e
      push!(sizes, n)
    end
    return elementReferences(e, sizes, Int[], IR.elementType(ty))
  end
  return e
end

"The references to the elements of the array `e` (of `sizes`) under the subscripts `index` so far: an array literal of them, nested by dimension."
function elementReferences(e::IR.CREF, sizes::Vector{Int}, index::Vector{Int}, element::IR.ModelicaType)::IR.Expression
  if length(index) == length(sizes)
    return expandWhole(subscripted(e, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index], element))
  end
  local rest = sizes[length(index) + 1:end]
  local ty = IR.T_ARRAY(element, IR.Dimension[IR.DIM_SIZE(n) for n in rest])
  return IR.ARRAY(IR.Expression[elementReferences(e, sizes, vcat(index, k), element) for k in 1:rest[1]], ty)
end

"`e[subscripts]`: a reference with the subscripts appended (its flat name), else a subscripted expression."
function subscripted(e::IR.Expression, subscripts::Vector{IR.Subscript}, ty::IR.ModelicaType)::IR.Expression
  e isa IR.ARRAY && return selectElement(e, subscripts, ty)
  if e isa IR.CREF && all(sub -> sub isa IR.INDEX && sub.exp isa IR.INTEGER, subscripts)
    local parts = copy(e.cref.parts)
    local last = parts[end]
    parts[end] = IR.CrefPart(last.name, vcat(last.subscripts, subscripts), ty)
    return IR.CREF(IR.ComponentRef(parts), ty)
  end
  return IR.SUBSCRIPTED(e, subscripts, ty)
end

"The field `name` of a record expression: of a reference, the reference to it; of a record or a constructor call, its element; of a call, the call's field."
function fieldOf(e::IR.Expression, name::Symbol, ty::IR.ModelicaType)::Union{Nothing, IR.Expression}
  if e isa IR.CREF
    local parts = copy(e.cref.parts)
    push!(parts, IR.CrefPart(name, IR.Subscript[], ty))
    return IR.CREF(IR.ComponentRef(parts), ty)
  end
  local rty = IR.typeOf(e)
  rty isa IR.T_RECORD || return nothing
  local index = findfirst(fl -> fl.name === name, rty.fields)
  index === nothing && return nothing
  local record = asRecord(e)
  record === nothing || return record.elements[index]
  return IR.RECORD_ELEMENT(e, index, name, ty)
end

"""
    asRecord(e) -> Union{Nothing, RECORD}

`e` as a record expression: itself, or a call of a record's constructor with every field given
(positionally, in the record's field order); `nothing` otherwise.
"""
function asRecord(e::IR.Expression)::Union{Nothing, IR.RECORD}
  e isa IR.RECORD && return e
  (e isa IR.CALL && e.ty isa IR.T_RECORD && e.name === e.ty.name && length(e.arguments) == length(e.ty.fields)) || return nothing
  return IR.RECORD(copy(e.arguments), e.ty)
end
