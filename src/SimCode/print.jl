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

#= Statements and equations: one line each, Modelica's form (the reasons a code generator refuses one name it so). =#

"The statements or equations `body`, joined by `; `."
function _printBody(io::IO, body::AbstractVector)
  for (k, x) in enumerate(body)
    k > 1 && print(io, "; ")
    show(io, x)
  end
end

"Branches `head c then body`, the later ones `elsehead c then body`."
function _printBranches(io::IO, head::String, conditions::AbstractVector, bodies::AbstractVector)
  for (k, (c, b)) in enumerate(zip(conditions, bodies))
    print(io, k == 1 ? head : string(" else", head), " ")
    show(io, c)
    print(io, " then ")
    _printBody(io, b)
  end
end

Base.show(io::IO, s::ASSIGN) = (show(io, s.left); print(io, " := "); show(io, s.right))

function Base.show(io::IO, s::IF)
  _printBranches(io, "if", Exp[b.condition for b in s.branches], Vector{Statement}[b.body for b in s.branches])
  isempty(s.elseBody) || (print(io, " else "); _printBody(io, s.elseBody))
  print(io, " end if")
end

Base.show(io::IO, s::FOR) = (print(io, "for "); show(io, s.iterator); print(io, " loop "); _printBody(io, s.body); print(io, " end for"))
Base.show(io::IO, s::WHILE) =
  (print(io, "while "); show(io, s.condition); print(io, " loop "); _printBody(io, s.body); print(io, " end while"))

function Base.show(io::IO, s::WHEN)
  _printBranches(io, "when", Exp[b.condition for b in s.branches], Vector{Statement}[b.body for b in s.branches])
  print(io, " end when")
end

Base.show(io::IO, s::ASSERT) = (print(io, "assert("); _printJoined(io, Exp[s.condition, s.message, s.level]); print(io, ")"))
Base.show(io::IO, s::TERMINATE) = (print(io, "terminate("); show(io, s.message); print(io, ")"))

function Base.show(io::IO, s::REINIT)
  print(io, "reinit(", crefName(s.stateVar))
  isempty(s.subscripts) || (print(io, "["); _printJoined(io, s.subscripts); print(io, "]"))
  print(io, ", ")
  show(io, s.value)
  print(io, ")")
end

Base.show(io::IO, s::NORETCALL) = show(io, s.exp)
Base.show(io::IO, ::RETURN) = print(io, "return")
Base.show(io::IO, ::BREAK) = print(io, "break")

Base.show(io::IO, eq::RESIDUAL_EQUATION) = (print(io, "0 = "); show(io, eq.exp))
Base.show(io::IO, eq::EQUATION) = (show(io, eq.lhs); print(io, " = "); show(io, eq.rhs))
Base.show(io::IO, eq::ARRAY_EQUATION) = (show(io, eq.left); print(io, " = "); show(io, eq.right))
Base.show(io::IO, eq::FOR_EQUATION) =
  (print(io, "for "); show(io, eq.iterator); print(io, " loop "); _printBody(io, eq.body); print(io, " end for"))
Base.show(io::IO, eq::ALGORITHM) = (print(io, "algorithm "); _printBody(io, eq.statements))

"A when-clause and its elsewhen parts."
function _printWhen(io::IO, w::WHEN_STMTS)
  local conditions::Vector{Exp} = Exp[]
  local bodies::Vector{Vector{Statement}} = Vector{Statement}[]
  local part::Union{Nothing, WHEN_STMTS} = w
  while part !== nothing
    push!(conditions, part.condition)
    push!(bodies, part.body)
    part = part.elsewhenPart
  end
  _printBranches(io, "when", conditions, bodies)
  print(io, " end when")
end

Base.show(io::IO, eq::WHEN_EQUATION) = _printWhen(io, eq.whenEquation)
Base.show(io::IO, eq::INITIAL_WHEN_EQUATION) = (print(io, "initial "); _printWhen(io, eq.whenEquation))

function Base.show(io::IO, eq::INLINE_IF_EQUATION)
  _printBranches(io, "if", eq.conditions, eq.branchesTrue)
  isempty(eq.branchElse) || (print(io, " else "); _printBody(io, eq.branchElse))
  print(io, " end if")
end

Base.show(io::IO, eq::ASSERT_EQUATION) = (print(io, "assert("); _printJoined(io, Exp[eq.condition, eq.message, eq.level]); print(io, ")"))
Base.show(io::IO, eq::UNLOWERED_EQUATION) = print(io, eq.what, ": ", eq.construct)
