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

#= The Modelica text of SimCode expressions: an assert's condition in its message, the
   warnings, reading the simulation code. Parentheses where the operators' precedence needs
   them. =#

const OPERATOR_TEXT = Dict{OpKind, String}(
  OP_ADD => "+", OP_ADD_SCALAR_ARRAY => "+", OP_ADD_ARRAY_SCALAR => "+", OP_ADD_EW => ".+",
  OP_SUB => "-", OP_SUB_SCALAR_ARRAY => "-", OP_SUB_ARRAY_SCALAR => "-", OP_SUB_EW => ".-",
  OP_MUL => "*", OP_MUL_SCALAR_ARRAY => "*", OP_MUL_ARRAY_SCALAR => "*", OP_MUL_VECTOR_MATRIX => "*",
  OP_MUL_MATRIX_VECTOR => "*", OP_SCALAR_PRODUCT => "*", OP_MATRIX_PRODUCT => "*", OP_MUL_EW => ".*",
  OP_DIV => "/", OP_DIV_ARRAY_SCALAR => "/", OP_DIV_SCALAR_ARRAY => "/", OP_DIV_EW => "./",
  OP_POW => "^", OP_POW_MATRIX => "^", OP_POW_SCALAR_ARRAY => "^", OP_POW_ARRAY_SCALAR => "^", OP_POW_EW => ".^",
  OP_UMINUS => "-", OP_AND => "and", OP_OR => "or", OP_NOT => "not",
  OP_LESS => "<", OP_LESSEQ => "<=", OP_GREATER => ">", OP_GREATEREQ => ">=", OP_EQUAL => "==", OP_NEQUAL => "<>")

"How tightly an expression binds: an operand of a tighter operator is parenthesized."
function precedence(e::Exp)::Int
  if e isa BINARY
    local t = OPERATOR_TEXT[e.op]
    t in ("+", "-", ".+", ".-") && return 5
    t in ("^", ".^") && return 8
    return 6
  end
  e isa UNARY && return 7
  e isa RELATION && return 4
  e isa LUNARY && return 3
  e isa LBINARY && return e.op === OP_AND ? 2 : 1
  e isa IFEXP && return 0
  return 10
end

"""
An operand, parenthesized where it binds less tightly than `p` (or as tightly, `strict`).
"""
function _printOperand(io::IO, e::Exp, p::Int; strict::Bool = false)
  local q = precedence(e)
  local paren = q < p || (strict && q == p)
  paren && print(io, "(")
  show(io, e)
  paren && print(io, ")")
end

function _printJoined(io::IO, xs::AbstractVector, sep::String = ", ")
  for (i, x) in enumerate(xs)
    i > 1 && print(io, sep)
    show(io, x)
  end
end

Base.show(io::IO, e::Union{ICONST, RCONST, BCONST}) = print(io, e.value)
Base.show(io::IO, e::SCONST) = print(io, repr(e.value))
Base.show(io::IO, e::ENUM_LITERAL) = print(io, e.name)
Base.show(io::IO, e::EXP_CREF) = print(io, crefName(e.cref))
Base.show(io::IO, e::DER) = print(io, "der(", crefName(e.cref), ")")
Base.show(io::IO, e::PRE) = print(io, "pre(", crefName(e.cref), ")")
Base.show(io::IO, ::WILD) = print(io, "_")
Base.show(io::IO, ::WHOLE_DIMENSION) = print(io, ":")

function Base.show(io::IO, e::Union{BINARY, LBINARY, RELATION})
  local p = precedence(e)
  _printOperand(io, e.exp1, p)
  print(io, " ", OPERATOR_TEXT[e.op], " ")
  #= `a - (b - c)`, `a / (b * c)`; `^` is not associative =#
  _printOperand(io, e.exp2, p; strict = !(e.op in (OP_ADD, OP_MUL, OP_AND, OP_OR)))
end

Base.show(io::IO, e::UNARY) = (print(io, "-"); _printOperand(io, e.exp, 7; strict = true))
Base.show(io::IO, e::LUNARY) = (print(io, "not "); _printOperand(io, e.exp, 3))

function Base.show(io::IO, e::IFEXP)
  print(io, "if ")
  show(io, e.cond)
  print(io, " then ")
  show(io, e.thenExp)
  print(io, " else ")
  show(io, e.elseExp)
end

Base.show(io::IO, e::ARRAY_EXP) = (print(io, "{"); _printJoined(io, e.elements); print(io, "}"))

function Base.show(io::IO, e::RANGE)
  _printOperand(io, e.start, 10)
  e.step === nothing || (print(io, ":"); _printOperand(io, e.step, 10))
  print(io, ":")
  _printOperand(io, e.stop, 10)
end

Base.show(io::IO, e::ASUB) = (_printOperand(io, e.exp, 10); print(io, "["); _printJoined(io, e.subs); print(io, "]"))
Base.show(io::IO, e::TSUB) = (_printOperand(io, e.exp, 10); print(io, "[", e.index, "]"))
Base.show(io::IO, e::RSUB) = (_printOperand(io, e.exp, 10); print(io, ".", e.fieldName))
Base.show(io::IO, e::CALL) = (print(io, e.name, "("); _printJoined(io, e.args); print(io, ")"))
Base.show(io::IO, e::CAST) = show(io, e.exp)
Base.show(io::IO, e::RECORD) = (print(io, e.name, "("); _printJoined(io, e.exps); print(io, ")"))
Base.show(io::IO, e::TUPLE) = (print(io, "("); _printJoined(io, e.elements); print(io, ")"))
Base.show(io::IO, it::Iterator) = (print(io, it.name, " in "); show(io, it.range))

function Base.show(io::IO, e::REDUCTION)
  print(io, e.reducer === :array ? "{" : string(e.reducer, "("))
  show(io, e.body)
  print(io, " for ")
  _printJoined(io, e.iterators)
  print(io, e.reducer === :array ? "}" : ")")
end

function Base.show(io::IO, e::PARTEVALFUNCTION)
  print(io, "function ", e.name, "(")
  for (i, (n, a)) in enumerate(zip(e.argNames, e.args))
    i > 1 && print(io, ", ")
    print(io, n, " = ")
    show(io, a)
  end
  print(io, ")")
end
