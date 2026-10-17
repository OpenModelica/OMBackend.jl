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

#= Compile-time evaluation of SimCode expressions: of literals alone, or with the parameters'
   bindings; and the if-expressions whose conditions so evaluate, resolved. =#

"""
    resolveIfExpInBindings!(simCode)

The parameters' and array parameters' bindings with the if-expressions whose conditions
evaluate at compile time resolved (`resolveConstantIfExp`).
"""
function resolveIfExpInBindings!(simCode::SIM_CODE)::SIM_CODE
  local ht = simCode.stringToSimVarHT
  for (name, (index, sv)) in ht
    local k = sv.varKind
    (k isa Union{PARAMETER, ARRAY_PARAMETER} && k.bindExp !== nothing) || continue
    local b = resolveConstantIfExp(k.bindExp, simCode)
    b === k.bindExp || (ht[name] = (index, withKind(sv, withBinding(k, b))))
  end
  return simCode
end

"""
    resolveConstantIfExp(e[, simCode])

`e` with each if-expression whose condition evaluates (`tryEvalCondition`; with `simCode`, the
parameters' bindings read) as its branch; another kept, its parts resolved in turn. Without
`simCode` only through arithmetic, calls and arrays; with it through relations, logical
operators, casts and subscripts too. An unchanged expression is returned itself.
"""
function resolveConstantIfExp(e::Exp)::Exp
  if e isa IFEXP
    local c = e.cond isa BCONST ? e.cond.value : tryEvalCondition(e.cond)
    c === true && return resolveConstantIfExp(e.thenExp)
    c === false && return resolveConstantIfExp(e.elseExp)
    return mapChildren(resolveConstantIfExp, e)
  end
  e isa Union{BINARY, UNARY, CALL, ARRAY_EXP} && return mapChildren(resolveConstantIfExp, e)
  return e
end

function resolveConstantIfExp(e::Exp, simCode::SIM_CODE)::Exp
  if e isa IFEXP
    local c = tryEvalCondition(e.cond, simCode)
    c === true && return resolveConstantIfExp(e.thenExp, simCode)
    c === false && return resolveConstantIfExp(e.elseExp, simCode)
  end
  e isa Union{IFEXP, BINARY, UNARY, LBINARY, LUNARY, RELATION, CAST, CALL, ARRAY_EXP, ASUB} ||
    return e
  return mapChildren(x -> resolveConstantIfExp(x, simCode), e)
end

"""
    tryEvalCondition(c[, simCode])

The value of a Boolean condition of literals (and, with `simCode`, parameters): a literal, a
relation of numbers, and/or/not, noEvent(); `nothing` if it does not evaluate. With
`simCode`, `and`/`or` short-circuit on a known operand, and if-expressions and casts
evaluate.
"""
function tryEvalCondition(c::Exp)::Union{Bool, Nothing}
  c isa BCONST && return c.value
  isBuiltin(c, :noEvent) && return length(c.args) == 1 ? tryEvalCondition(c.args[1]) : nothing
  if c isa RELATION
    local (v1, v2) = (tryEvalNumeric(c.exp1), tryEvalNumeric(c.exp2))
    (v1 === nothing || v2 === nothing) && return nothing
    return compareValues(v1, c.op, v2)
  elseif c isa LBINARY && c.op in (OP_AND, OP_OR)
    local (r1, r2) = (tryEvalCondition(c.exp1), tryEvalCondition(c.exp2))
    (r1 === nothing || r2 === nothing) && return nothing
    return c.op === OP_AND ? r1 && r2 : r1 || r2
  elseif c isa LUNARY && c.op === OP_NOT
    local r = tryEvalCondition(c.exp)
    return r === nothing ? nothing : !r
  end
  return nothing
end

tryEvalCondition(c::Exp, simCode::SIM_CODE)::Union{Bool, Nothing} = tryEvalCondition(c, simCode, OrderedSet{String}())

function tryEvalCondition(c::Exp, simCode::SIM_CODE, seen::OrderedSet{String})::Union{Bool, Nothing}
  c isa BCONST && return c.value
  if c isa EXP_CREF
    local v = tryEvalScalar(c, simCode, seen)
    return v isa Bool ? v : nothing
  elseif c isa IFEXP
    local cv = tryEvalCondition(c.cond, simCode, seen)
    return cv === true ? tryEvalCondition(c.thenExp, simCode, seen) :
           cv === false ? tryEvalCondition(c.elseExp, simCode, seen) : nothing
  elseif isBuiltin(c, :noEvent)
    return length(c.args) == 1 ? tryEvalCondition(c.args[1], simCode, seen) : nothing
  elseif c isa RELATION
    return compareScalarValues(tryEvalScalar(c.exp1, simCode, seen), c.op, tryEvalScalar(c.exp2, simCode, seen))
  elseif c isa LBINARY && c.op === OP_AND
    local r1 = tryEvalCondition(c.exp1, simCode, seen)
    r1 === false && return false
    local r2 = tryEvalCondition(c.exp2, simCode, seen)
    r2 === false && return false
    return r1 === true && r2 === true ? true : nothing
  elseif c isa LBINARY && c.op === OP_OR
    local r1 = tryEvalCondition(c.exp1, simCode, seen)
    r1 === true && return true
    local r2 = tryEvalCondition(c.exp2, simCode, seen)
    r2 === true && return true
    return r1 === false && r2 === false ? false : nothing
  elseif c isa LUNARY && c.op === OP_NOT
    local r = tryEvalCondition(c.exp, simCode, seen)
    return r === nothing ? nothing : !r
  elseif c isa CAST
    return tryEvalCondition(c.exp, simCode, seen)
  end
  return nothing
end

"Whether `e` is a call of the built-in `name`."
isBuiltin(e::Exp, name::Symbol)::Bool = e isa CALL && e.kind === CALL_BUILTIN && e.name === name

"""
    tryEvalNumeric(e[, simCode])

The value of a numeric expression of literals (and, with `simCode`, parameters): `+ - * /`
(and, with `simCode`, `^` where real), unary minus, abs(), noEvent() (with `simCode` also
Integer(), casts, enumeration literals as their indices); `nothing` if it does not evaluate.
"""
function tryEvalNumeric(e::Exp)::Union{Float64, Nothing}
  e isa RCONST && return e.value
  e isa ICONST && return Float64(e.value)
  if e isa UNARY && e.op === OP_UMINUS
    local v = tryEvalNumeric(e.exp)
    return v === nothing ? nothing : -v
  elseif isBuiltin(e, :abs) || isBuiltin(e, :noEvent)
    length(e.args) == 1 || return nothing
    local v = tryEvalNumeric(e.args[1])
    return v === nothing ? nothing : e.name === :abs ? abs(v) : v
  elseif e isa BINARY && e.op in (OP_ADD, OP_SUB, OP_MUL, OP_DIV)
    local (v1, v2) = (tryEvalNumeric(e.exp1), tryEvalNumeric(e.exp2))
    (v1 === nothing || v2 === nothing) && return nothing
    return arithmetic(v1, e.op, v2)
  end
  return nothing
end

tryEvalNumeric(e::Exp, simCode::SIM_CODE)::Union{Float64, Nothing} = tryEvalNumeric(e, simCode, OrderedSet{String}())

function tryEvalNumeric(e::Exp, simCode::SIM_CODE, seen::OrderedSet{String})::Union{Float64, Nothing}
  e isa RCONST && return e.value
  e isa ICONST && return Float64(e.value)
  e isa ENUM_LITERAL && return Float64(e.index)
  if e isa EXP_CREF
    return withParameterBinding(b -> tryEvalNumeric(b, simCode, seen), e, simCode, seen)
  elseif e isa UNARY && e.op === OP_UMINUS
    local v = tryEvalNumeric(e.exp, simCode, seen)
    return v === nothing ? nothing : -v
  elseif isBuiltin(e, :abs) || isBuiltin(e, :noEvent) || isBuiltin(e, :Integer)
    length(e.args) == 1 || return nothing
    local v = tryEvalNumeric(e.args[1], simCode, seen)
    return v === nothing ? nothing : e.name === :abs ? abs(v) : v
  elseif e isa BINARY && e.op in (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW)
    local (v1, v2) = (tryEvalNumeric(e.exp1, simCode, seen), tryEvalNumeric(e.exp2, simCode, seen))
    (v1 === nothing || v2 === nothing) && return nothing
    if e.op === OP_POW
      #= Buildings' Movers: kRes = dpMax/V_flow_max*delta^2/10 =#
      return (v1 > 0.0 || isinteger(v2) && (v1 != 0.0 || v2 >= 0.0)) ? v1 ^ v2 : nothing
    end
    return arithmetic(v1, e.op, v2)
  elseif e isa CAST
    return tryEvalNumeric(e.exp, simCode, seen)
  end
  return nothing
end

"`v1 op v2` for `+ - * /` (a division by zero does not evaluate, another operator neither)."
function arithmetic(v1::Float64, op::OpKind, v2::Float64)::Union{Float64, Nothing}
  op === OP_ADD && return v1 + v2
  op === OP_SUB && return v1 - v2
  op === OP_MUL && return v1 * v2
  op === OP_DIV && return v2 == 0.0 ? nothing : v1 / v2
  return nothing
end

"""
    tryEvalScalar(e, simCode[, seen])

The value of a scalar expression (a Boolean, a String, an Integer or a Real: a literal, a
parameter's binding, noEvent(), Integer(), a cast, a condition, an if-expression on one,
else `tryEvalNumeric`'s); `nothing` if it does not evaluate. `seen`: the parameters being
evaluated (a cycle does not evaluate).
"""
tryEvalScalar(e::Exp, simCode::SIM_CODE) = tryEvalScalar(e, simCode, OrderedSet{String}())

function tryEvalScalar(e::Exp, simCode::SIM_CODE, seen::OrderedSet{String})
  e isa Union{BCONST, SCONST, ICONST, RCONST} && return e.value
  e isa ENUM_LITERAL && return e.index
  if e isa EXP_CREF
    return withParameterBinding(b -> tryEvalScalar(b, simCode, seen), e, simCode, seen)
  elseif isBuiltin(e, :noEvent)
    return length(e.args) == 1 ? tryEvalScalar(e.args[1], simCode, seen) : nothing
  elseif isBuiltin(e, :Integer)
    length(e.args) == 1 || return nothing
    local v = tryEvalScalar(e.args[1], simCode, seen)
    return v isa Number && isinteger(v) ? Int(v) : nothing
  elseif e isa CAST
    return tryEvalScalar(e.exp, simCode, seen)
  elseif e isa Union{RELATION, LBINARY, LUNARY}
    #= a Boolean bound to a comparison (Buildings' Movers: haveVMax = abs(...) < eps) =#
    return tryEvalCondition(e, simCode, seen)
  elseif e isa IFEXP
    local c = tryEvalCondition(e.cond, simCode, seen)
    return c === true ? tryEvalScalar(e.thenExp, simCode, seen) : c === false ? tryEvalScalar(e.elseExp, simCode, seen) : nothing
  end
  return tryEvalNumeric(e, simCode, seen)
end

"""
    withParameterBinding(f, e, simCode, seen)

`f` of the binding of the parameter `e` references (a parameter, an array parameter or a
String with one; not a tunable one, not one being evaluated), with it marked as being
evaluated; `nothing` for another reference.
"""
function withParameterBinding(f::F, e::EXP_CREF, simCode::SIM_CODE, seen::OrderedSet{String}) where {F}
  local bound = boundParameterExpression(e, simCode, seen)
  bound === nothing && return nothing
  local (name, b) = bound
  push!(seen, name)
  try
    return f(b)
  finally
    delete!(seen, name)
  end
end

"The name and binding of the parameter `e` references; `nothing` for another expression."
function boundParameterExpression(e::Exp, simCode::SIM_CODE, seen::OrderedSet{String})::Union{Nothing, Tuple{String, Exp}}
  local name = crefNameOf(e)
  (name === nothing || name in seen || isTunableParameter(name)) && return nothing
  local entry = get(simCode.stringToSimVarHT, name, nothing)
  entry === nothing && return nothing
  local k = last(entry).varKind
  (k isa Union{PARAMETER, ARRAY_PARAMETER, STRING} && k.bindExp !== nothing) || return nothing
  return (name, k.bindExp)
end

"`v1 op v2` for a relational `op` of two numbers; `==`/`<>` of other values; `nothing` if either is unknown."
function compareScalarValues(v1::Union{Nothing, Bool, Real, String}, op::OpKind, v2::Union{Nothing, Bool, Real, String})::Union{Bool, Nothing}
  (v1 === nothing || v2 === nothing) && return nothing
  (v1 isa Number && v2 isa Number) && return compareValues(Float64(v1), op, Float64(v2))
  op === OP_EQUAL && return v1 == v2
  op === OP_NEQUAL && return v1 != v2
  return nothing
end

function compareValues(v1::Float64, op::OpKind, v2::Float64)::Union{Bool, Nothing}
  op === OP_LESS && return v1 < v2
  op === OP_LESSEQ && return v1 <= v2
  op === OP_GREATER && return v1 > v2
  op === OP_GREATEREQ && return v1 >= v2
  op === OP_EQUAL && return v1 == v2
  op === OP_NEQUAL && return v1 != v2
  return nothing
end
