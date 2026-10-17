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
    decomposeRecordEquations!(model)

The record equations of the continuous-time partitions (and the structural modes), equations
and initial equations, as the equations of their fields (the previous backend's
decomposeComplexEquation), in place: `lhs = rhs` of a record type is one equation per field,
an array field an array equation, a record field decomposed in turn. A record equation whose
right-hand side is a call stays one (the call's result is assigned whole), as one whose left-hand
side is neither a reference nor a record value.
"""
function decomposeRecordEquations!(model::IR.Model)::IR.Model
  for p in Iterators.flatten((model.continuous, (m.system for m in model.modes)))
    p.equations = decomposeRecordEquations(p.equations)
    p.initialEquations = decomposeRecordEquations(p.initialEquations)
  end
  return model
end

function decomposeRecordEquations(equations::Vector{IR.Equation})::Vector{IR.Equation}
  any(eq -> eq isa IR.RECORD_EQUATION, equations) || return equations
  local out = IR.Equation[]
  for eq in equations
    if eq isa IR.RECORD_EQUATION && !(eq.rhs isa IR.CALL)
      append!(out, recordEquations(eq.lhs, eq.rhs, eq.source))
    else
      push!(out, eq)
    end
  end
  return out
end

"""
    recordEquations(lhs, rhs, source)

`lhs = rhs` of a record type as the equations of its fields. The record type is the
left-hand side's or the right-hand side's (of a reference, a call, a record value); without
one, or with a left-hand side that is not a reference or a record value, the record equation
itself.
"""
function recordEquations(lhs::IR.Expression, rhs::IR.Expression, source::IR.Source)::Vector{IR.Equation}
  local ty = something(recordTypeOf(lhs), recordTypeOf(rhs), Some(nothing))
  ty === nothing && return IR.Equation[IR.RECORD_EQUATION(lhs, rhs, IR.typeOf(lhs), 1, source, IR.DEFAULT_ATTRIBUTES)]
  (lhs isa IR.CREF || lhs isa IR.RECORD) ||
    return IR.Equation[IR.RECORD_EQUATION(lhs, rhs, ty, length(ty.fields), source, IR.DEFAULT_ATTRIBUTES)]
  local out = IR.Equation[]
  for field in ty.fields
    local l = fieldExpression(lhs, field.name, field.ty)
    local r = fieldExpression(rhs, field.name, field.ty)
    if field.ty isa IR.T_ARRAY
      push!(out, IR.ARRAY_EQUATION(l, r, field.ty, source, IR.DEFAULT_ATTRIBUTES))
    elseif field.ty isa IR.T_RECORD
      append!(out, recordEquations(l, r, source))
    else
      push!(out, IR.SCALAR_EQUATION(l, r, field.ty, source, IR.DEFAULT_ATTRIBUTES))
    end
  end
  return out
end

"The record type of a reference, a call or a record value; `nothing` for another expression."
function recordTypeOf(e::IR.Expression)::Union{Nothing, IR.T_RECORD}
  (e isa IR.CREF || e isa IR.CALL || e isa IR.RECORD) || return nothing
  local ty = IR.typeOf(e)
  return ty isa IR.T_RECORD ? ty : nothing
end

const COMPLEX_CONJ = (Symbol("Modelica.ComplexMath.conj"), :Modelica_ComplexMath_conj)
const COMPLEX_FROM_REAL = (Symbol("Complex.'constructor'.fromReal"), Symbol("Complex_'constructor'_fromReal"),
                           :Complex_constructor_fromReal)

"""
    fieldExpression(e, name, ty)

The field `name` (of type `ty`) of the record expression `e`: of a reference, the reference
with the field as a part; of a record value, its element; of an if-expression, the field of
each branch; of `Modelica.ComplexMath.conj(c)`, `c.re` and `-c.im`; of an inlined
`Complex(re, im)` (its constructor `fromReal`), the argument.
"""
function fieldExpression(e::IR.Expression, name::Symbol, ty::IR.ModelicaType)::IR.Expression
  if e isa IR.CREF
    return IR.CREF(IR.ComponentRef(IR.CrefPart[e.cref.parts..., IR.CrefPart(name, IR.Subscript[], ty)]), ty)
  elseif e isa IR.RECORD
    local k = findfirst(f -> f.name === name, e.ty.fields)
    k === nothing && unsupported("a record value without the field $(name)", IR.toString(e))
    return e.elements[k]
  elseif e isa IR.IF_EXPRESSION
    return IR.IF_EXPRESSION(e.condition, fieldExpression(e.thenExp, name, ty), fieldExpression(e.elseExp, name, ty), ty)
  elseif e isa IR.CALL && e.name in COMPLEX_CONJ && !isempty(e.arguments)
    name === :re && return fieldExpression(e.arguments[1], :re, ty)
    name === :im && return IR.UNARY(IR.UMINUS, fieldExpression(e.arguments[1], :im, ty), ty)
    unsupported("the field $(name) of conj(), not re or im", IR.toString(e))
  elseif e isa IR.CALL && e.name in COMPLEX_FROM_REAL
    name === :re && return isempty(e.arguments) ? e : e.arguments[1]
    name === :im && return length(e.arguments) >= 2 ? e.arguments[2] : IR.REAL(0.0)
    unsupported("the field $(name) of Complex(), not re or im", IR.toString(e))
  end
  unsupported("the field $(name) of this record expression ($(nameof(typeof(e))))", IR.toString(e))
end
