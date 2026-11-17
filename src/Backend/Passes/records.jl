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
side is neither a reference nor a record value. A constant `Complex(re, im)` constructor call is
the record value (`constantRecord`).
"""
function decomposeRecordEquations!(model::IR.Model)::IR.Model
  for p in Iterators.flatten((model.continuous, (m.system for m in model.modes)))
    p.equations = decomposeRecordEquations(p.equations)
    p.initialEquations = decomposeRecordEquations(p.initialEquations)
  end
  return model
end

function decomposeRecordEquations(equations::Vector{IR.Equation})::Vector{IR.Equation}
  any(eq -> eq isa IR.RECORD_EQUATION || eq isa IR.FOR_EQUATION || isRecordArrayEquation(eq), equations) ||
    return equations
  local out = IR.Equation[]
  for eq in equations
    #= an equation of two arrays of records is lowered as a record equation (of the array type): an array equation =#
    (eq isa IR.RECORD_EQUATION && eq.ty isa IR.T_ARRAY && IR.elementType(eq.ty) isa IR.T_RECORD) &&
      (eq = IR.ARRAY_EQUATION(eq.lhs, eq.rhs, eq.ty, eq.source, eq.attributes))
    if eq isa IR.RECORD_EQUATION
      local (lhs, rhs) = (constantRecord(eq.lhs), constantRecord(eq.rhs))
      if !(rhs isa IR.CALL)
        append!(out, recordEquations(lhs, rhs, eq.source))
        continue
      end
    elseif eq isa IR.FOR_EQUATION
      push!(out, IR.FOR_EQUATION(eq.iterator, bodyRecordEquations(eq.body), eq.source, eq.attributes))
      continue
    elseif eq isa IR.ARRAY_EQUATION
      local fields::Union{Nothing, Vector{IR.Equation}} = recordArrayEquations(eq)
      if fields !== nothing
        append!(out, fields)
        continue
      end
    end
    push!(out, eq)
  end
  return out
end

"Whether `eq` is an array equation of records (arrays kept: `c.Phi = c.port_p.Phi` of a component array)."
isRecordArrayEquation(eq::IR.Equation)::Bool = eq isa IR.ARRAY_EQUATION && IR.elementType(eq.ty) isa IR.T_RECORD

"""
    recordArrayEquations(eq) -> Union{Nothing, Vector{Equation}}

The equation `eq` of two arrays of records, both references, as the array equations of their fields (`c.Phi.re =
c.port_p.Phi.re`: the variables of an array of records are its fields' arrays); `nothing` for another equation, or a
record with an array or record field. Left whole, `expandEquations` wrote one equation of a record per element,
which the structure reads as one row for the two unknowns (MSL FundamentalWave: 12 unknowns over the rows).
"""
function recordArrayEquations(eq::IR.ARRAY_EQUATION)::Union{Nothing, Vector{IR.Equation}}
  local element::IR.ModelicaType = IR.elementType(eq.ty)
  (element isa IR.T_RECORD && eq.lhs isa IR.CREF && eq.rhs isa IR.CREF) || return nothing
  any(f -> f.ty isa Union{IR.T_RECORD, IR.T_ARRAY}, element.fields) && return nothing
  local out::Vector{IR.Equation} = IR.Equation[]
  for field in element.fields
    local ty::IR.ModelicaType = IR.T_ARRAY(field.ty, eq.ty.dimensions)
    push!(out, IR.ARRAY_EQUATION(fieldExpression(eq.lhs, field.name, ty), fieldExpression(eq.rhs, field.name, ty), ty,
                                 eq.source, eq.attributes))
  end
  return out
end

"""
    bodyRecordEquations(body) -> Vector{Equation}

The equations of a for-equation's body (arrays kept: a source loop, a component array's equations) with its record
equations as the equations of their fields, in nested for- and if-equations too: a Complex one by its real and
imaginary parts (`complexEquations`), another whose right-hand side is a reference or a record value by
`recordEquations`; one of another value (a call's result) stays. At the top level a record equation of a call
stays whole for `expandEquations`, which writes `r.f = call[k]`; in a body nothing splits it later, and the array
form of index reduction has no record equations (MSL QuasiStatic and FundamentalWave: `i[j] = if j == k then
-pin_p.i else Complex(0)`, `V_m = (2.0/pi)*N*i`).
"""
function bodyRecordEquations(body::Vector{IR.Equation})::Vector{IR.Equation}
  local out::Vector{IR.Equation} = IR.Equation[]
  for eq in body
    if eq isa IR.RECORD_EQUATION
      local parts::Union{Nothing, Vector{IR.Equation}} = complexEquations(eq)
      local (lhs::IR.Expression, rhs::IR.Expression) = (constantRecord(eq.lhs), constantRecord(eq.rhs))
      if parts !== nothing
        append!(out, parts)
      elseif rhs isa Union{IR.CREF, IR.RECORD}
        append!(out, recordEquations(lhs, rhs, eq.source))
      else
        push!(out, eq)
      end
    elseif eq isa IR.FOR_EQUATION
      push!(out, IR.FOR_EQUATION(eq.iterator, bodyRecordEquations(eq.body), eq.source, eq.attributes))
    elseif eq isa IR.IF_EQUATION
      local branches::Vector{IR.EquationBranch} =
        IR.EquationBranch[IR.EquationBranch(b.condition, bodyRecordEquations(b.equations)) for b in eq.branches]
      push!(out, IR.IF_EQUATION(branches, bodyRecordEquations(eq.elseEquations), eq.source, eq.attributes))
    elseif eq isa IR.ARRAY_EQUATION
      local fields::Union{Nothing, Vector{IR.Equation}} = recordArrayEquations(eq)
      fields === nothing ? push!(out, eq) : append!(out, fields)
    else
      push!(out, eq)
    end
  end
  return out
end

"""
The Complex equation `eq` as the equations of its real and imaginary parts; `nothing` where a side has no parts
(`complexParts`).
"""
function complexEquations(eq::IR.RECORD_EQUATION)::Union{Nothing, Vector{IR.Equation}}
  local l::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(eq.lhs)
  l === nothing && return nothing
  local r::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(eq.rhs)
  r === nothing && return nothing
  return IR.Equation[IR.SCALAR_EQUATION(l[1], r[1], IR.T_REAL(), eq.source, eq.attributes),
                     IR.SCALAR_EQUATION(l[2], r[2], IR.T_REAL(), eq.source, eq.attributes)]
end

"A record type of the fields `re` and `im`: Complex, or a type that extends it."
isComplexType(ty::IR.ModelicaType)::Bool =
  ty isa IR.T_RECORD && length(ty.fields) == 2 && ty.fields[1].name === :re && ty.fields[2].name === :im

"""
Whether the call `e` is an operation of Complex values the parts of whose result `complexParts` writes: a function of
Complex or of Modelica.ComplexMath, or an operator of a type that extends Complex, with a Complex result.
"""
function isComplexOperation(e::IR.CALL)::Bool
  isComplexType(e.ty) || return false
  local name::String = String(e.name)
  return name == "Complex" || startswith(name, "Complex.") || startswith(name, "Complex_") ||
         startswith(name, "Modelica.ComplexMath.") ||
         startswith(name, "Modelica_ComplexMath_") || occursin(r"'(\*|\+|-|/|constructor)'", name)
end

"""
The operation of an operator record's function: the last part of its name (`Complex.'*'.multiply`: `multiply`;
`Complex.'+'`: `'+'`).
"""
complexOperation(name::Symbol)::String = String(last(split(String(name), r"[._]")))

"""
    lowerComplexProjections!(model) -> Model

The Real projections of Complex values in the equations of `model`'s continuous partitions as arithmetic on the
parts (`complexProjection`), for a model with its arrays kept: the array code has no Complex values, and a
projection's argument is a reference to a record (`abs_V_m = Modelica.ComplexMath.'abs'(V_m)`: "unknown variable
converter_s[1].V_m"). The scalarized model's are lowered on its simulation code
(`SimCode.lowerComplexOperatorRecords`).
"""
function lowerComplexProjections!(model::IR.Model)::IR.Model
  for p in model.continuous
    foreach(eq -> mapRoots!(withComplexProjections, eq), p.equations)
    foreach(eq -> mapRoots!(withComplexProjections, eq), p.initialEquations)
  end
  return model
end

"`e` with its projections of Complex values lowered (`complexProjection`), top-down."
function withComplexProjections(@nospecialize(e::IR.Expression))::IR.Expression
  local p::Union{Nothing, IR.Expression} = complexProjection(e)
  return IR.mapChildren(withComplexProjections, p === nothing ? e : p)
end

"""
    complexProjection(e) -> Union{Nothing, Expression}

The Real value of a projection of a Complex value by its parts (`complexParts`): a field `re` or `im` of an
operation's result; `Modelica.ComplexMath.real` and `imag`; the absolute value (`'abs'`, `abs`:
`(re^2 + im^2)^0.5`, as MSL); the argument (`arg(c)`, `arg(c, phi0)` in `(phi0 - pi, phi0 + pi]`: MSL's
`atan3`). `nothing` for another expression.
"""
function complexProjection(@nospecialize(e::IR.Expression))::Union{Nothing, IR.Expression}
  if e isa IR.RECORD_ELEMENT
    (e.field in (:re, :im) && e.exp isa Union{IR.CALL, IR.IF_EXPRESSION, IR.RECORD}) || return nothing
    local parts::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e.exp)
    return parts === nothing ? nothing : e.field === :re ? parts[1] : parts[2]
  elseif e isa Union{IR.TUPLE_ELEMENT, IR.SUBSCRIPTED}
    #= a Complex operation's result by position, the form the record equations give `y = a * b` of Complex values
       (`y.re = Complex.'*'.multiply(a, b)[1]`). Not a function of a library whose body decides its value (the scalar
       product of Complex vectors conjugates its first argument in MSL 4, not in MSL 3): the array code calls it =#
    (e.exp isa IR.CALL && isComplexOperation(e.exp)) || return nothing
    local position::Union{Nothing, Int} = e isa IR.TUPLE_ELEMENT ? e.index :
      length(e.subscripts) == 1 ? IR.literalIndex(e.subscripts[1]) : nothing
    (position === nothing || !(position in (1, 2))) && return nothing
    local byPosition::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e.exp)
    return byPosition === nothing ? nothing : byPosition[position]
  end
  (e isa IR.CALL && !isempty(e.arguments) && !isComplexType(e.ty)) || return nothing
  local name::String = String(e.name)
  (startswith(name, "Modelica.ComplexMath.") || startswith(name, "Modelica_ComplexMath_")) || return nothing
  local op::String = complexOperation(e.name)
  op in ("'abs'", "abs", "arg", "real", "imag") || return nothing
  local c::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e.arguments[1])
  c === nothing && return nothing
  op == "real" && return c[1]
  op == "imag" && return c[2]
  if op != "arg"
    local squares::IR.Expression = realBinary(realBinary(c[1], IR.POW, IR.REAL(2.0)), IR.ADD,
                                              realBinary(c[2], IR.POW, IR.REAL(2.0)))
    return realBinary(squares, IR.POW, IR.REAL(0.5))
  end
  local w::IR.Expression = IR.CALL(:atan2, IR.Expression[c[2], c[1]], IR.T_REAL(), IR.BUILTIN_CALL)
  local phi0::Union{Nothing, IR.Expression} = length(e.arguments) >= 2 ? e.arguments[2] : nothing
  (phi0 === nothing || (phi0 isa Union{IR.REAL, IR.INTEGER} && iszero(phi0.value))) && return w
  #= MSL Math.atan3: w + 2pi * integer((pi + phi0 - w) / (2pi)) =#
  local turns::IR.Expression =
    IR.CALL(:integer, IR.Expression[realBinary(realBinary(IR.REAL(Float64(pi)), IR.ADD, realBinary(phi0, IR.SUB, w)),
                                               IR.DIV, IR.REAL(2pi))], IR.T_INTEGER(), IR.BUILTIN_CALL)
  return realBinary(w, IR.ADD, realBinary(IR.REAL(2pi), IR.MUL, turns))
end

realNegation(v::IR.Expression)::IR.Expression = IR.UNARY(IR.UMINUS, v, IR.T_REAL())
realBinary(l::IR.Expression, op::IR.Op, r::IR.Expression)::IR.Expression = IR.BINARY(l, op, r, IR.T_REAL())

"""
    complexParts(e) -> Union{Nothing, Tuple{Expression, Expression}}

The real and imaginary parts of the Complex value `e`: of a reference its fields, of a record value its elements, of
an if-expression those of its branches, of an operator record's call (`fromReal`, `conj`, negation, `+`, `-`, `*`,
`/`, `exp`) Real arithmetic on its operands' parts, as SimCode's `complexParts` writes them for the scalarized
model; `nothing` for another expression (a function's Complex result).
"""
function complexParts(e::IR.Expression)::Union{Nothing, NTuple{2, IR.Expression}}
  local real::IR.ModelicaType = IR.T_REAL()
  if e isa IR.CREF
    isComplexType(IR.typeOf(e)) || return nothing
    return (fieldExpression(e, :re, real), fieldExpression(e, :im, real))
  elseif e isa IR.RECORD
    (isComplexType(e.ty) && length(e.elements) == 2) || return nothing
    return (e.elements[1], e.elements[2])
  elseif e isa IR.IF_EXPRESSION
    local a::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e.thenExp)
    a === nothing && return nothing
    local b::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e.elseExp)
    b === nothing && return nothing
    return (IR.IF_EXPRESSION(e.condition, a[1], b[1], real), IR.IF_EXPRESSION(e.condition, a[2], b[2], real))
  end
  #= the constructor of a record type that extends Complex, `Modelica.Units.SI.ComplexVoltage(re, im)` =#
  (e isa IR.CALL && isComplexType(e.ty) && length(e.arguments) == 2 && e.name === (e.ty::IR.T_RECORD).name) &&
    return (e.arguments[1], e.arguments[2])
  (e isa IR.CALL && isComplexOperation(e)) || return nothing
  local op::String = complexOperation(e.name)
  local n::Int = length(e.arguments)
  (op == "fromReal" && 1 <= n <= 2) && return (e.arguments[1], n == 2 ? e.arguments[2] : IR.REAL(0.0))
  #= the record's constructor, `Complex(re, im)` =#
  (e.name === :Complex && n == 2) && return (e.arguments[1], e.arguments[2])
  (1 <= n <= 2) || return nothing
  local x::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e.arguments[1])
  x === nothing && return nothing
  if n == 1
    op == "conj" && return (x[1], realNegation(x[2]))
    op in ("negate", "'-'") && return (realNegation(x[1]), realNegation(x[2]))
    if op == "exp"
      local magnitude::IR.Expression = IR.CALL(:exp, IR.Expression[x[1]], real, IR.BUILTIN_CALL)
      return (realBinary(magnitude, IR.MUL, IR.CALL(:cos, IR.Expression[x[2]], real, IR.BUILTIN_CALL)),
              realBinary(magnitude, IR.MUL, IR.CALL(:sin, IR.Expression[x[2]], real, IR.BUILTIN_CALL)))
    end
    return nothing
  end
  local y::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e.arguments[2])
  y === nothing && return nothing
  op in ("add", "'+'") && return (realBinary(x[1], IR.ADD, y[1]), realBinary(x[2], IR.ADD, y[2]))
  op in ("subtract", "'-'") && return (realBinary(x[1], IR.SUB, y[1]), realBinary(x[2], IR.SUB, y[2]))
  op in ("multiply", "'*'") &&
    return (realBinary(realBinary(x[1], IR.MUL, y[1]), IR.SUB, realBinary(x[2], IR.MUL, y[2])),
            realBinary(realBinary(x[1], IR.MUL, y[2]), IR.ADD, realBinary(x[2], IR.MUL, y[1])))
  if op in ("divide", "'/'")
    local den::IR.Expression = realBinary(realBinary(y[1], IR.MUL, y[1]), IR.ADD, realBinary(y[2], IR.MUL, y[2]))
    return (realBinary(realBinary(realBinary(x[1], IR.MUL, y[1]), IR.ADD, realBinary(x[2], IR.MUL, y[2])), IR.DIV, den),
            realBinary(realBinary(realBinary(x[2], IR.MUL, y[1]), IR.SUB, realBinary(x[1], IR.MUL, y[2])), IR.DIV, den))
  end
  return nothing
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
    constantRecord(e)

A call of Complex's constructor with literal arguments as the record value `Complex(re, im)`, as the frontend's
constant evaluation writes it where it scalarizes; any other expression itself.
"""
function constantRecord(e::IR.Expression)::IR.Expression
  (e isa IR.CALL && e.name in COMPLEX_FROM_REAL && e.ty isa IR.T_RECORD && length(e.ty.fields) == 2) || return e
  (1 <= length(e.arguments) <= 2 && all(a -> a isa IR.REAL || a isa IR.INTEGER, e.arguments)) || return e
  local re::IR.Expression = IR.REAL(Float64(e.arguments[1].value))
  local im::IR.Expression = length(e.arguments) == 2 ? IR.REAL(Float64(e.arguments[2].value)) : IR.REAL(0.0)
  return IR.RECORD(IR.Expression[re, im], e.ty)
end

"""
    fieldExpression(e, name, ty)

The field `name` (of type `ty`) of the record expression `e`: of a reference, the reference
with the field as a part; of a record value, its element; of an if-expression, the field of
each branch; of `Modelica.ComplexMath.conj(c)`, `c.re` and `-c.im`; of an inlined
`Complex(re, im)` (its constructor `fromReal`), the argument; of another operation of Complex values, its part
(`complexParts`).
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
  elseif e isa IR.CALL && name in (:re, :im)
    #= of another Complex operation, its part (`(-i0).re` in a branch of an if-expression) =#
    local parts::Union{Nothing, NTuple{2, IR.Expression}} = complexParts(e)
    parts === nothing || return name === :re ? parts[1] : parts[2]
  end
  unsupported("the field $(name) of this record expression ($(nameof(typeof(e))))", IR.toString(e))
end
