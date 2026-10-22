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

#= OMJLIR expressions built simplified: the constructors drop zero and one terms and fold
   literals (`0 * w` is `0`, `if c then e else e` is `e`), the literal tests, and `foldScalar`,
   which rebuilds an arithmetic node with them. Differentiation builds its derivatives with them
   (IndexReduction/Differentiate.jl). =#

const ZERO = IR.REAL(0.0)

const ONE = IR.REAL(1.0)

isZero(e::IR.Expression)::Bool = (e isa IR.REAL && e.value == 0.0) || (e isa IR.INTEGER && e.value == 0)

isOne(e::IR.Expression)::Bool = (e isa IR.REAL && e.value == 1.0) || (e isa IR.INTEGER && e.value == 1)

"Whether `e` is a Real or Integer literal."
isLiteral(e::IR.Expression)::Bool = e isa Union{IR.REAL, IR.INTEGER}

literalValue(e::IR.Expression)::Float64 = Float64(e.value)

"`a + b`, a zero term dropped, literals folded (a derivative is mostly zeros)."
function plus(a::IR.Expression, b::IR.Expression)::IR.Expression
  isZero(a) && return b
  isZero(b) && return a
  isLiteral(a) && isLiteral(b) && return real(literalValue(a) + literalValue(b))
  return IR.BINARY(a, IR.ADD, b, IR.T_REAL())
end

"`a - b`, a zero term dropped (`0 - b` is `-b`), literals folded."
function minus(a::IR.Expression, b::IR.Expression)::IR.Expression
  isZero(b) && return a
  isZero(a) && return negate(b)
  isLiteral(a) && isLiteral(b) && return real(literalValue(a) - literalValue(b))
  return IR.BINARY(a, IR.SUB, b, IR.T_REAL())
end

"`a * b`: 0 where a factor is 0, a factor 1 dropped, literals folded."
function times(a::IR.Expression, b::IR.Expression)::IR.Expression
  (isZero(a) || isZero(b)) && return ZERO
  isOne(a) && return b
  isOne(b) && return a
  isLiteral(a) && isLiteral(b) && return real(literalValue(a) * literalValue(b))
  return IR.BINARY(a, IR.MUL, b, IR.T_REAL())
end

"`a / b`: 0 where `a` is 0, `a` where `b` is 1."
function divide(a::IR.Expression, b::IR.Expression)::IR.Expression
  isZero(a) && return ZERO
  isOne(b) && return a
  return IR.BINARY(a, IR.DIV, b, IR.T_REAL())
end

"`-a`: a double negation dropped, a literal folded."
function negate(a::IR.Expression)::IR.Expression
  isZero(a) && return ZERO
  a isa IR.UNARY && a.op === IR.UMINUS && return a.exp
  isLiteral(a) && return real(-literalValue(a))
  return IR.UNARY(IR.UMINUS, a, IR.T_REAL())
end

"`a ^ b`: `a` where `b` is 1, 1 where `b` is 0."
function power(a::IR.Expression, b::IR.Expression)::IR.Expression
  isOne(b) && return a
  isZero(b) && return ONE
  return IR.BINARY(a, IR.POW, b, IR.T_REAL())
end

"`if c then a else b`, or the branch where both are the same (one node, or literals of one value: the derivative of each branch of a piecewise function by a variable neither reads)."
function ifThenElse(c::IR.Expression, a::IR.Expression, b::IR.Expression, ty::IR.ModelicaType)::IR.Expression
  (a === b || (isLiteral(a) && isLiteral(b) && literalValue(a) == literalValue(b))) && return a
  local known = literalCondition(c)
  known === nothing || return known ? a : b
  return IR.IF_EXPRESSION(c, a, b, ty)
end

"""
    literalCondition(c) -> Union{Nothing, Bool}

The value of a condition of literals (`true`, `3 == 1`: an inlined function's `if axis == 1`
for a literal axis); `nothing` for one that reads anything else.
"""
function literalCondition(c::IR.Expression)::Union{Nothing, Bool}
  c isa IR.BOOLEAN && return c.value
  (c isa IR.RELATION && isLiteral(c.exp1) && isLiteral(c.exp2)) || return nothing
  local (a, b) = (literalValue(c.exp1), literalValue(c.exp2))
  c.op === IR.EQUAL && return a == b
  c.op === IR.NEQUAL && return a != b
  c.op === IR.LESS && return a < b
  c.op === IR.LESSEQ && return a <= b
  c.op === IR.GREATER && return a > b
  c.op === IR.GREATEREQ && return a >= b
  return nothing
end

real(x::Float64)::IR.Expression = IR.REAL(x)

"""
    foldScalar(x) -> Expression

A scalar arithmetic node (`ADD`, `SUB`, `MUL`, `DIV`, `POW`, unary minus of a Real) rebuilt with
the constructors above, its zero and one terms dropped and its literals folded (`0 * w` is `0`);
an if-expression of equal branches the branch (`ifThenElse`); another node as it is.
"""
function foldScalar(x::IR.Expression)::IR.Expression
  if x isa IR.BINARY && x.ty isa IR.T_REAL
    x.op === IR.ADD && return plus(x.exp1, x.exp2)
    x.op === IR.SUB && return minus(x.exp1, x.exp2)
    x.op === IR.MUL && return times(x.exp1, x.exp2)
    x.op === IR.DIV && return divide(x.exp1, x.exp2)
    x.op === IR.POW && return power(x.exp1, x.exp2)
  elseif x isa IR.UNARY && x.op === IR.UMINUS && x.ty isa IR.T_REAL
    return negate(x.exp)
  elseif x isa IR.IF_EXPRESSION
    return ifThenElse(x.condition, x.thenExp, x.elseExp, x.ty)
  end
  return x
end

"Whether `e` is zero: a zero literal, or an array of zeros."
isZeroExp(e::IR.Expression)::Bool = isZero(e) || (e isa IR.ARRAY && all(isZeroExp, e.elements))
