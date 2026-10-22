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

#=
  The lowering of SimCode expressions to the ModelingToolkit model (`expToJuliaExpMTK`, one
  method per expression), of conditions to zero-crossing functions and callback conditions, and
  the values the code generation needs at the start of the simulation (the t0 value map, the
  initial branches of if-equations).
=#
module MTK_CodeGenerationUtil

import DataStructures
using DataStructures: OrderedSet
import MacroTools
using ModelingToolkit
using LinearAlgebra

import ...SimCode
import ...OMBackend
import ..AlgorithmicCodeGeneration
import ..CodeGenerationUtil
using ..CodeGenerationUtil
import ..CodeGeneration: lowerKnownSymbolicFunctionCall
import ..MTKDump: dumpPreStructuralSimplifyExpr

import ...@BACKEND_LOGGING
import ...COMPONENT_SEPARATOR

import Symbolics
import Symbolics.RuntimeGeneratedFunctions
import SymbolicUtils
import OMRuntimeExternalC

const SC = SimCode

#= ==================================================================================== =#
#= Conditions                                                                           =#
#= ==================================================================================== =#

"""
    expToJuliaBoolMTK(cond, simCode; cachedChange = false)

A condition of a DiscreteCallback as a Julia Boolean: relations and `&&`, `||`, `!` as such
(`expToJuliaExpMTK` encodes them arithmetically, for Symbolics in the equations); `change(x)`
as `x != pre(x)` and `edge(b)` as `b and not pre(b)`, from the integrator's previous state (the
values the callback cached before its sweep with `cachedChange`); `initial()` and
`terminal()` false: the callback runs neither in the initialization (the initial algorithms
do) nor at the end.
"""
function expToJuliaBoolMTK(cond::SC.Exp, simCode::SC.SIM_CODE; cachedChange::Bool = false)
  if cond isa SC.RELATION
    local op = opKindToJuliaOperator(cond.op)
    return :($op($(expToJuliaExpMTK(cond.exp1, simCode)), $(expToJuliaExpMTK(cond.exp2, simCode))))
  elseif cond isa SC.LBINARY
    local a = _boolOperandMTK(cond.exp1, simCode, cachedChange)
    local b = _boolOperandMTK(cond.exp2, simCode, cachedChange)
    return cond.op === SC.OP_AND ? :($a && $b) : :($a || $b)
  elseif cond isa SC.LUNARY
    return :(!($(_boolOperandMTK(cond.exp, simCode, cachedChange))))
  elseif cond isa SC.CALL && cond.kind === SC.CALL_BUILTIN
    if length(cond.args) == 1 && cond.name in (:noEvent, :change, :edge)
      local arg = only(cond.args)
      cond.name === :noEvent && return expToJuliaBoolMTK(arg, simCode; cachedChange)
      local curr = expToJuliaBoolMTK(arg, simCode; cachedChange)
      local prev = _preValueLookup(arg, simCode; cachedChange)
      #= `!= 0`: a Boolean discrete is a Float64 (0.0/1.0) in the state vector =#
      return cond.name === :change ? :(($(curr)) != ($(prev))) : :((($(curr)) != 0) && !((($(prev)) != 0)))
    end
    cond.name in (:initial, :terminal) && return false
  end
  return expToJuliaExpMTK(cond, simCode)
end

"""
An operand of not/and/or: a Boolean variable (or its pre()) is a Float64 (0.0/1.0) in the
state vector, and `!active` threw a MethodError (`when not active`, the StateGraph steps).
"""
function _boolOperandMTK(e::SC.Exp, simCode::SC.SIM_CODE, cachedChange::Bool)
  local x = expToJuliaBoolMTK(e, simCode; cachedChange)
  local isBool = e isa Union{SC.RELATION, SC.LBINARY, SC.LUNARY, SC.BCONST} ||
                 (e isa SC.CALL && e.kind === SC.CALL_BUILTIN && e.name in (:change, :edge, :initial, :terminal))
  return isBool ? x : :($x != 0)
end

"""
The value of a reference before the event, in a discrete callback: from the integrator's
previous state (`lookuptableStates` the callback's table of the states' indices), read
through the integrator (the affect binds the model's variables by name: one named `x`
replaced the state vector `x`, a BoundsError). A parameter's, a constant's: its value.
"""
function _preValueLookup(arg::SC.Exp, simCode::SC.SIM_CODE; cachedChange::Bool = false)
  arg isa SC.EXP_CREF || return expToJuliaExpMTK(arg, simCode)
  local entry = get(simCode.stringToSimVarHT, SC.crefName(arg.cref), nothing)
  (entry === nothing || SC.isParameter(last(entry))) && return expToJuliaExpMTK(arg, simCode)
  local name = last(entry).name
  cachedChange && return :(get(_changePreValues, Symbol($(name)), integrator.u[lookuptableStates[Symbol($(name))]]))
  return :(integrator.uprev[lookuptableStates[Symbol($(name))]])
end

"""
    transformToMTKContinuousCondition(cond, simCode; atInitial = false, normalized = false)

The zero-crossing function of a condition: negative where the condition holds. A relation
`a < b` as `a - b`; `==`, `<>`, a Boolean variable or call `b` as `0.5 - b`; `or`, `and`, `not`
as `min`, `max`, `-`; `initial()` a constant, negative at the initialization (`atInitial`, the
condition's initial value) and positive after it (the event iteration's start re-evaluation
reads the runtime one: MSL FluxTubes' Tellinen hysteresis stayed in its `if initial()` branch).

`normalized`: each relation's difference divided by its own scale, `1 + max(|a|, |b|)`
(`_operandScaleExpr`), so that where min/max combine several relations a hysteresis is
relative to each one's operands (OpenModelica's per-relation hysteresis).
"""
function transformToMTKContinuousCondition(cond::SC.Exp, simCode::SC.SIM_CODE;
                                           atInitial::Bool = false, normalized::Bool = false)
  local difference = (a::SC.Exp, b::SC.Exp) -> begin
    local ja = expToJuliaExpMTK(a, simCode); local jb = expToJuliaExpMTK(b, simCode)
    normalized ? :(($(ja) - $(jb)) / $(_operandScaleExpr(Any[ja, jb]))) : :($(ja) - $(jb))
  end
  local sub = c::SC.Exp -> transformToMTKContinuousCondition(c, simCode; atInitial, normalized)
  if cond isa SC.RELATION
    cond.op in (SC.OP_LESS, SC.OP_LESSEQ) && return difference(cond.exp1, cond.exp2)
    cond.op in (SC.OP_GREATER, SC.OP_GREATEREQ) && return difference(cond.exp2, cond.exp1)
    #= `==`, `<>`: true is 1, false 0; negative when true (`b - 0.5` inverted the polarity:
       `nperiod == 0` with nperiod = -1 read true inside a min). =#
    return :(0.5 - $(expToJuliaExpMTK(cond, simCode)))
  elseif cond isa SC.LBINARY
    return cond.op === SC.OP_OR ? :(min($(sub(cond.exp1)), $(sub(cond.exp2)))) :
                                  :(max($(sub(cond.exp1)), $(sub(cond.exp2))))
  elseif cond isa SC.LUNARY
    return :(-($(sub(cond.exp))))
  elseif SC.isBuiltin(cond, :noEvent)
    length(cond.args) == 1 || OMBackend.unsupported("noEvent with several arguments in a condition", cond)
    return sub(only(cond.args))
  elseif SC.isBuiltin(cond, :initial)
    return atInitial ? :(-1) : :(1)
  elseif cond isa Union{SC.EXP_CREF, SC.PRE, SC.CALL}
    return :(0.5 - $(expToJuliaExpMTK(cond, simCode)))
  end
  OMBackend.unsupported("condition expression", cond)
end

"The zero-crossing equation `zc ~ 0` of a condition (`transformToMTKContinuousCondition`)."
transformToMTKContinuousConditionEquation(cond::SC.Exp, simCode::SC.SIM_CODE; atInitial::Bool = false)::Expr =
  :($(transformToMTKContinuousCondition(cond, simCode; atInitial)) ~ 0)

"""
True if the condition holds at its zero-crossing boundary (zc == 0): the normalized
crossing function cannot tell >= from > there, the relation's operator decides.
Conservative: false for anything but a (negated, noEvent-wrapped) relation.
"""
function condClosedAtBoundary(cond::SC.Exp)::Bool
  cond isa SC.RELATION && return cond.op in (SC.OP_LESSEQ, SC.OP_GREATEREQ, SC.OP_EQUAL)
  (cond isa SC.LUNARY && cond.exp isa SC.RELATION) && return cond.exp.op in (SC.OP_LESS, SC.OP_GREATER)
  (SC.isBuiltin(cond, :noEvent) && length(cond.args) == 1) && return condClosedAtBoundary(only(cond.args))
  return false
end

#= Relation semantics for events (MLS 8.5, OpenModelica's relationhysteresis): a relation
   keeps its value during integration and changes only at events; its crossing function has
   a hysteresis relative to that value, so it is never zero after the initialization or after
   an event (FMI 3.0 3.1.1). =#

"The relations of a condition that generate events (not inside noEvent)."
function _eventRelations(cond::SC.Exp)::Vector{SC.RELATION}
  local out = SC.RELATION[]
  local visit = function (e::SC.Exp)
    if SC.isBuiltin(e, :noEvent)
      return
    elseif e isa SC.RELATION
      push!(out, e)
      return
    end
    foreach(visit, SC.children(e))
  end
  visit(cond)
  return out
end

"""
The scale of a condition's operands, `1 + max(|a|, |b|, ...)`, for the hysteresis width
`H * scale` (OpenModelica: max(|a|,|b|) + nominal).
"""
function _conditionScaleExpr(cond::SC.Exp, simCode::SC.SIM_CODE)
  local terms = Any[]
  for rel in _eventRelations(cond)
    push!(terms, expToJuliaExpMTK(rel.exp1, simCode), expToJuliaExpMTK(rel.exp2, simCode))
  end
  isempty(terms) && return 1.0
  return _operandScaleExpr(terms)
end

"""
`1 + max(|x|, ...)` of lowered operands. An infinite operand (the MSL Spice3 V_pulse's
default pulse width and period, inf, in `time >= T0 + Tfalling`) is left out: it made the
hysteresis infinite.
"""
function _operandScaleExpr(xs::Vector{Any})
  local terms = Any[:(ModelingToolkit.ifelse(abs($(x)) < 1.0e300, abs($(x)), 0.0)) for x in xs]
  return length(terms) == 1 ? :(1.0 + $(terms[1])) : :(1.0 + max($(terms...)))
end

"""
A literal: a constant, its negation, an array of them.
"""
function _isLiteralExp(e::SC.Exp)::Bool
  e isa Union{SC.ICONST, SC.RCONST, SC.SCONST, SC.BCONST, SC.ENUM_LITERAL} && return true
  e isa SC.UNARY && return _isLiteralExp(e.exp)
  e isa SC.ARRAY_EXP && return all(_isLiteralExp, e.elements)
  return false
end

"""
The literal value of a condition (as at the initialization), as a Julia Bool expression
over `observed.zcK`: each relation from its own crossing function, `<`/`>` strict and
`<=`/`>=` closed, and/or/not as such. `obsKws` collects the observed crossing functions.
Nothing for a condition of another shape.
"""
function _literalConditionExpr(cond::SC.Exp, simCode::SC.SIM_CODE, obsKws::Vector{Expr})
  local rec = e::SC.Exp -> _literalConditionExpr(e, simCode, obsKws)
  local observe = function (zc)
    local name = Symbol("zc", length(obsKws) + 1)
    push!(obsKws, Expr(:kw, name, zc))
    return :(observed.$(name))
  end
  if cond isa SC.RELATION
    local zc = observe(transformToMTKContinuousCondition(cond, simCode))
    #= == and <> cross as 0.5 - (a == b): negative exactly when it holds =#
    return cond.op in (SC.OP_LESSEQ, SC.OP_GREATEREQ) ? :($(zc) <= 0) : :($(zc) < 0)
  elseif cond isa SC.LBINARY
    local l = rec(cond.exp1); local r = rec(cond.exp2)
    (l === nothing || r === nothing) && return nothing
    return cond.op === SC.OP_AND ? :($(l) && $(r)) : :($(l) || $(r))
  elseif cond isa SC.LUNARY
    local inner = rec(cond.exp)
    return inner === nothing ? nothing : :(!$(inner))
  elseif SC.isBuiltin(cond, :noEvent)
    return length(cond.args) == 1 ? rec(only(cond.args)) : nothing
  elseif cond isa SC.EXP_CREF
    return :($(observe(expToJuliaExpMTK(cond, simCode))) > 0.5)
  elseif cond isa SC.BCONST
    return cond.value
  end
  return nothing
end

#= ==================================================================================== =#
#= State of the lowering                                                                =#
#= ==================================================================================== =#

#= The delay() calls of the model being generated (delays.jl), as (key, argument, delay time)
   with the argument and the delay time as Julia expressions; a call's position is the index
   of its history, and an equal call shares it. With DELAY_MODEL, the name the histories are
   kept under. Both reset at the start of each model's (or structural mode's) generation. =#
const DELAY_CALLS = Tuple{String, Any, Any}[]
const DELAY_MODEL = Ref{Symbol}(:none)

function _delayIndex!(key::String, x, delayTime)::Int
  local k = findfirst(c -> first(c) == key, DELAY_CALLS)
  k === nothing || return k
  push!(DELAY_CALLS, (key, x, delayTime))
  return length(DELAY_CALLS)
end

#= The argument of a when body the event iteration runs (codeGen.jl: the relation whens of
   `_emitRelationWhen`, and the discrete-when and elsewhen-arm affects): the state before the
   current sweep, indexed by `lookuptableStates`. =#
const PRE_SNAPSHOT = :__prevals
#= The state index a when assignment writes through (`integrator.u[i]`): a reserved name, the
   body binds the model's variables by their own names (a discrete `idx` indexing a table,
   BuildingsRepro.ParameterArrayByDiscreteIndex). =#
const STATE_INDEX = :__stateIdx
#= The homotopy parameter λ (homotopy(actual, simplified), MLS 3.7.4.4): 1 in a simulation;
   the initialization of a model with homotopy() goes from 0 (the simplified expressions) to 1
   (`_homotopyContinuation`), as OpenModelica's default does. =#
const HOMOTOPY_LAMBDA = :_homotopyLambda
#= Whether homotopy() is lowered as the blend in λ: in the continuous equations and the
   initialization equations only (ODE_MODE_MTK_MODEL_GENERATION), where λ is a parameter and
   the continuation acts. Elsewhere (when bodies and conditions, relations, if-equation
   branches, bindings, observed equations) the actual expression: the blend read an undefined
   λ there (an UndefVarError at the first event), and λ is 1 after the initialization. =#
const HOMOTOPY_BLEND = Base.ScopedValues.ScopedValue(false)

#= Where `pre(v)` of a variable is read while a when body is lowered: false, from `v` itself;
   true (within `with(PRE_FROM_SNAPSHOT => true)`), from `PRE_SNAPSHOT`, so a body sees the
   pre() values even when an earlier body of the same sweep assigned the variable. =#
const PRE_FROM_SNAPSHOT = Base.ScopedValues.ScopedValue(false)

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

"""
    expToJuliaExpMTK(e, simCode)

The Julia expression of the SimCode expression `e` in the ModelingToolkit model: a reference
by its variable's symbol, `time` as `t`, `der(x)` as `D(x)`; the Boolean operators
arithmetically (`a and b` as `a * b`: neither `&&` nor `&` works on Symbolics' mix of Bool, Num
and BasicSymbolic); an if-expression of Real branches as `c * a + (1 - c) * b`; a call of the
model's functions with its records' fields as arguments, of a built-in by its implementation.
"""
function expToJuliaExpMTK end

expToJuliaExpMTK(e::Union{SC.BCONST, SC.ICONST, SC.RCONST, SC.SCONST}, ::SC.SIM_CODE)::Expr = quote $(e.value) end
expToJuliaExpMTK(::SC.WILD, ::SC.SIM_CODE)::Expr = quote _ end
"An enumeration literal is its index."
expToJuliaExpMTK(e::SC.ENUM_LITERAL, ::SC.SIM_CODE)::Expr = quote $(e.index) end

function expToJuliaExpMTK(e::SC.EXP_CREF, simCode::SC.SIM_CODE)::Expr
  (e.cref.sym === :time && e.ty isa SC.TYPE_REAL) && return quote t end
  e.ty isa SC.TYPE_ARRAY && return _wholeArrayExpr(e, simCode)
  if !isempty(e.cref.subs)
    local element = _arrayElementExpr(String(e.cref.sym), Any[s for s in e.cref.subs], simCode)
    element === nothing || return element
  end
  return _variableExpr(SC.crefName(e.cref), simCode)
end

"`der(x)`: `D(x)`."
expToJuliaExpMTK(e::SC.DER, ::SC.SIM_CODE)::Expr = quote D($(Symbol(SC.crefName(e.cref)))) end

"""
`pre(x)`: in a when body lowered with PRE_FROM_SNAPSHOT, the value of `x` before the event
iteration's sweep; else `x` (a callback reads its arguments before the affect changes them).
"""
function expToJuliaExpMTK(e::SC.PRE, simCode::SC.SIM_CODE)::Expr
  local name = SC.crefName(e.cref)
  local entry = get(simCode.stringToSimVarHT, name, nothing)
  if PRE_FROM_SNAPSHOT[] && entry !== nothing && !SC.isParameter(last(entry))
    return :($(PRE_SNAPSHOT)[lookuptableStates[$(QuoteNode(Symbol(name)))]])
  end
  return quote $(Symbol(name)) end
end

function expToJuliaExpMTK(e::SC.BINARY, simCode::SC.SIM_CODE)::Expr
  local op = opKindToJuliaOperator(e.op)
  return :($op($(expToJuliaExpMTK(e.exp1, simCode)), $(expToJuliaExpMTK(e.exp2, simCode))))
end

expToJuliaExpMTK(e::SC.UNARY, simCode::SC.SIM_CODE)::Expr =
  :($(opKindToJuliaOperator(e.op))($(expToJuliaExpMTK(e.exp, simCode))))

"`not a`: `1 - a`."
expToJuliaExpMTK(e::SC.LUNARY, simCode::SC.SIM_CODE)::Expr = :(1 - $(expToJuliaExpMTK(e.exp, simCode)))

"""
`a and b`: `a * b`; `a or b`: `a + b - a * b`, the operands bound once (spliced twice, a
left-nested chain of ors grew exponentially: Digital RAM and table when-conditions).
"""
function expToJuliaExpMTK(e::SC.LBINARY, simCode::SC.SIM_CODE)::Expr
  local a = expToJuliaExpMTK(e.exp1, simCode)
  local b = expToJuliaExpMTK(e.exp2, simCode)
  e.op === SC.OP_AND && return :($(a) * $(b))
  return :(let _a = $(a), _b = $(b); _a + _b - _a * _b end)
end

function expToJuliaExpMTK(e::SC.RELATION, simCode::SC.SIM_CODE)::Expr
  local op = opKindToJuliaOperator(e.op)
  return quote ($op($(expToJuliaExpMTK(e.exp1, simCode)), $(expToJuliaExpMTK(e.exp2, simCode)))) end
end

"""
`if c then a else b`: of a literal condition its branch; of Real branches `c * a + (1 - c) * b`
(ModelingToolkit.ifelse has dispatch problems with BasicSymbolic and Num); else
`ifelse(c, a, b)` (a String, a tuple: CombiTables' `if useFile then fileName else "NoName"`).
"""
function expToJuliaExpMTK(e::SC.IFEXP, simCode::SC.SIM_CODE)::Expr
  if e.cond isa SC.BCONST
    return quote
      $(LineNumberNode(@__LINE__, "evaluated if expr"))
      $(expToJuliaExpMTK(e.cond.value ? e.thenExp : e.elseExp, simCode))
    end
  end
  local c = expToJuliaExpMTK(e.cond, simCode)
  local a = _guardNonIntegerPowerBasesForEagerBranch(expToJuliaExpMTK(e.thenExp, simCode))
  local b = _guardNonIntegerPowerBasesForEagerBranch(expToJuliaExpMTK(e.elseExp, simCode))
  (_ifexpBranchIsNonReal(e.thenExp) || _ifexpBranchIsNonReal(e.elseExp)) && return :(ifelse(Bool($(c)), $(a), $(b)))
  return :($(c) * $(a) + (1 - $(c)) * $(b))
end

"""
    _ifexpBranchIsNonReal(e)

Whether a branch of an if-expression is clearly no Real number, so the arithmetic
`c * a + (1 - c) * b` does not apply: a String (literal, variable, call), a tuple (a guarded
call of several outputs of a sub-clocked partition: `if active then random(pre(state)) else
(pre(y), pre(state))`), an if-expression of such a branch.
"""
function _ifexpBranchIsNonReal(e::SC.Exp)::Bool
  e isa Union{SC.SCONST, SC.TUPLE} && return true
  e isa SC.EXP_CREF && return e.ty isa SC.TYPE_STRING
  e isa SC.CALL && return e.ty isa Union{SC.TYPE_STRING, SC.TYPE_TUPLE}
  e isa SC.IFEXP && return _ifexpBranchIsNonReal(e.thenExp) || _ifexpBranchIsNonReal(e.elseExp)
  return false
end

"""
The condition of a noEvent if-expression as ifelse takes it: a Bool, or a symbolic Bool. A
Boolean the backend holds as a Real (0 or 1: MSL ElastoGap's `contact`) is true above 1/2.
"""
_noEventCondition(c::Bool)::Bool = c
_noEventCondition(c::Number)::Bool = !iszero(c)
_noEventCondition(c::Symbolics.Num) = SymbolicUtils.symtype(Symbolics.unwrap(c)) <: Bool ? c : c > 0.5
_noEventCondition(c::SymbolicUtils.BasicSymbolic) = SymbolicUtils.symtype(c) <: Bool ? c : c > 0.5

expToJuliaExpMTK(e::SC.CAST, simCode::SC.SIM_CODE)::Expr = quote $(generateCastExpressionMTK(e.ty, e.exp, simCode)) end

"""
    generateCastExpressionMTK(ty, exp, simCode)

A conversion to `ty`: to Real `float(x)`, to a Real array `float.(x)`, to an Integer array
`Int.(round.(x))`.
"""
function generateCastExpressionMTK(ty::SC.SType, exp::SC.Exp, simCode::SC.SIM_CODE)::Expr
  local x = expToJuliaExpMTK(exp, simCode)
  ty isa SC.TYPE_REAL && return quote float($(x)) end
  (ty isa SC.TYPE_ARRAY && ty.elementType isa SC.TYPE_REAL) && return quote float.($(x)) end
  (ty isa SC.TYPE_ARRAY && ty.elementType isa SC.TYPE_INTEGER) && return quote Int.(round.($(x))) end
  OMBackend.unsupported("cast to $(ty)", exp)
end

expToJuliaExpMTK(e::SC.TUPLE, simCode::SC.SIM_CODE)::Expr = Expr(:tuple, (expToJuliaExpMTK(x, simCode) for x in e.elements)...)

"""
Output `k` of a call: `tupleElementCall(f, k, args...)`, of an array output of known size
`tupleArrayElementCall(f, k, dims, args...)` (indexing the call, `Num(term)[1]`, is a no-op in
Symbolics); of another tuple the element.
"""
function expToJuliaExpMTK(e::SC.TSUB, simCode::SC.SIM_CODE)::Expr
  e.exp isa SC.CALL || return :($(expToJuliaExpMTK(e.exp, simCode))[$(e.index)])
  local call = e.exp
  local fn = QuoteNode(call.name)
  local args = _modelicaFunctionCallArgs(call.args, simCode)
  if e.ty isa SC.TYPE_ARRAY && !isempty(e.ty.dims) && all(>=(0), e.ty.dims)
    return :(OMBackend.CodeGeneration.tupleArrayElementCall($fn, $(e.index), $(Tuple(e.ty.dims)), $(args...)))
  end
  return :(OMBackend.CodeGeneration.tupleElementCall($fn, $(e.index), $(args...)))
end

"""
A field of a record value: `re`, `im` by their projections (a Julia Complex, a record's
NamedTuple, a Symbolics Num of a Complex term: Magnetic.FundamentalWave's
`Complex.'*'.multiply(c1, c2).re`); another by name, or by position in a tuple of a function's
outputs.
"""
function expToJuliaExpMTK(e::SC.RSUB, simCode::SC.SIM_CODE)::Expr
  local inner = expToJuliaExpMTK(e.exp, simCode)
  e.fieldName == "re" && return :(OMBackend.CodeGeneration._recordFieldRe($inner))
  e.fieldName == "im" && return :(OMBackend.CodeGeneration._recordFieldIm($inner))
  local ix = AlgorithmicCodeGeneration._positionalFieldIndex(e.exp, e.index)
  return :(OMBackend.CodeGeneration._recordField($inner, $(QuoteNode(Symbol(e.fieldName))), $(ix)))
end

"""
A record value: Modelica's `Complex(re, im)` as Julia's; another as a NamedTuple of its
fields, each wrapped as a Num (a bare BasicSymbolic field broke the arithmetic on a parameter
record passed to a function: Spice3's Mosfet, the ideal gases' DataRecord).
"""
function expToJuliaExpMTK(e::SC.RECORD, simCode::SC.SIM_CODE)::Expr
  if e.name === :Complex && length(e.exps) == 2
    return quote Complex($(expToJuliaExpMTK(e.exps[1], simCode)), $(expToJuliaExpMTK(e.exps[2], simCode))) end
  end
  local fields = Expr[:(Symbolics.wrap($(expToJuliaExpMTK(x, simCode)))) for x in e.exps]
  length(e.fieldNames) == length(fields) || return Expr(:tuple, fields...)
  return Expr(:tuple, Expr(:parameters, (Expr(:(=), Symbol(n), f) for (n, f) in zip(e.fieldNames, fields))...))
end

expToJuliaExpMTK(e::SC.ARRAY_EXP, simCode::SC.SIM_CODE)::Expr = handleArrayExp(e, simCode)

"""
A function argument with bound arguments: a closure calling the function's wrapper (symbolic
arguments make a term, numeric ones a value).
"""
function expToJuliaExpMTK(e::SC.PARTEVALFUNCTION, simCode::SC.SIM_CODE)::Expr
  local callee = Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(e.name))
  return AlgorithmicCodeGeneration._partialApplicationExpr(callee, Any[expToJuliaExpMTK(x, simCode) for x in e.args], e)
end

"""
A reduction: `sum`, `product`, `min`, `max` of a generator; an array constructor
`{e for i in u, j in v}` as a comprehension, its last iterator the first dimension
(MLS 10.4.1.2; a Julia comprehension's first iterator is).
"""
function expToJuliaExpMTK(e::SC.REDUCTION, simCode::SC.SIM_CODE)::Expr
  local body = expToJuliaExpMTK(e.body, simCode)
  local iterators = Expr[Expr(:(=), it.name, expToJuliaExpMTK(it.range, simCode)) for it in e.iterators]
  e.reducer === :array && return Expr(:comprehension, body, reverse(iterators)...)
  local f = get(REDUCERS_MTK, e.reducer, nothing)
  f === nothing && OMBackend.unsupported("the reduction $(e.reducer)", e)
  return :($(f)($(Expr(:generator, body, iterators...))))
end

const REDUCERS_MTK = Dict{Symbol, Symbol}(:sum => :sum, :product => :prod, :min => :minimum, :max => :maximum)

function expToJuliaExpMTK(e::SC.RANGE, simCode::SC.SIM_CODE)::Expr
  local start = expToJuliaExpMTK(e.start, simCode)
  local stop = expToJuliaExpMTK(e.stop, simCode)
  e.step === nothing && return :($start:$stop)
  return :($start:$(expToJuliaExpMTK(e.step, simCode)):$stop)
end

function expToJuliaExpMTK(e::SC.CALL, simCode::SC.SIM_CODE)::Expr
  e.kind === SC.CALL_BUILTIN && return _builtinCallMTK(e, simCode)
  return _modelicaFunctionCallExpr(e, simCode)
end

expToJuliaExpMTK(e::SC.Exp, ::SC.SIM_CODE)::Expr = OMBackend.unsupported("expression", e)

#= ==================================================================================== =#
#= References                                                                           =#
#= ==================================================================================== =#

"""
    _variableExpr(name, simCode)

The variable `name`: its symbol in the table; an eliminated alias's representative (the
substitution missed the reference); an element by a variable subscript from its array's table
(`_dynamicSubscriptLookup`); else the name itself.
"""
function _variableExpr(name::String, simCode::SC.SIM_CODE)::Expr
  local entry = get(simCode.stringToSimVarHT, name, nothing)
  entry === nothing || return quote $(Symbol(last(entry).name)) end
  local (aliasResolved, aliasExpr) = resolveAliasedCref(name, simCode)
  if aliasResolved
    @warn "expToJuliaExpMTK: resolved alias-eliminated cref via fallback" name
    return aliasExpr
  end
  local dynamic = _dynamicSubscriptLookup(name, simCode)
  dynamic === nothing || return dynamic
  return quote $(Symbol(name)) end
end

"""
    _wholeArrayExpr(e, simCode)

A whole array by its name, `x`, when its elements are variables (its last element is); a zero
size one as an empty array of its shape (MSL Fluid's Xi_outflow[Medium.nXi], nXi = 0 for a
single substance); else the array of its elements (MSL Engine1b_analytic's frame_im.R.T passed
to selectBranch in an initial equation).
"""
function _wholeArrayExpr(e::SC.EXP_CREF, simCode::SC.SIM_CODE)::Expr
  local ty = e.ty::SC.TYPE_ARRAY
  any(==(0), ty.dims) && return _emptyArrayExpr(ty)
  any(<(0), ty.dims) && OMBackend.unsupported("array dimension", e)
  local name = SC.crefName(e.cref)
  local lastElement = name * join("[$(d)]" for d in ty.dims)
  haskey(simCode.stringToSimVarHT, lastElement) && return quote $(Symbol(name)) end
  local (aliasResolved, aliasExpr) = resolveAliasedCref(lastElement, simCode)
  if aliasResolved
    @warn "expToJuliaExpMTK: resolved alias-eliminated array variable via fallback" lastElement
    return aliasExpr
  end
  length(ty.dims) <= 2 || OMBackend.unsupported("an array of more than two dimensions without variables", name)
  local element = idx::Vector{Int} -> expToJuliaExpMTK(SC.EXP_CREF(SC.SimCref(Symbol(name), idx), ty.elementType), simCode)
  length(ty.dims) == 1 && return Expr(:vect, (element([i]) for i in 1:ty.dims[1])...)
  return Expr(:vcat, (Expr(:row, (element([i, j]) for j in 1:ty.dims[2])...) for i in 1:ty.dims[1])...)
end

function _emptyArrayExpr(ty::SC.TYPE_ARRAY)::Expr
  local elementType = ty.elementType isa SC.TYPE_INTEGER ? :Int : ty.elementType isa SC.TYPE_BOOL ? :Bool : :Float64
  return :(zeros($elementType, $((max(d, 0) for d in ty.dims)...)))
end

"""
    _arrayElementExpr(base, subs, simCode)

An element of the array variable `base` (one kept as an array) by its lowered subscripts:
with literal subscripts, the element of its binding (an array literal) or its scalar variable
(`R_T[1][2]`: a record's field array has both, the parent's binding is empty where equations
give the values); else `base[subs...]`. `nothing` for a `base` that is no array variable.
"""
function _arrayElementExpr(base::String, subs::Vector{Any}, simCode::SC.SIM_CODE)::Union{Nothing, Expr}
  local ht = simCode.stringToSimVarHT
  local entry = get(ht, base, nothing)
  (entry !== nothing && last(entry).varKind isa Union{SC.ARRAY, SC.ARRAY_PARAMETER}) || return nothing
  if all(s -> s isa Integer, subs)
    local binding = last(entry).varKind.bindExp
    if binding isa SC.ARRAY_EXP
      local element = binding
      for i in subs
        element isa SC.ARRAY_EXP || OMBackend.unsupported("a binding row that is not an array literal", element)
        element = element.elements[i]
      end
      return expToJuliaExpMTK(element, simCode)
    end
    local scalar = get(ht, base * join("[$(i)]" for i in subs), nothing)
    scalar === nothing || return quote $(Symbol(last(scalar).name)) end
  end
  return quote
    $(LineNumberNode(@__LINE__, "Array subscript: $base"))
    $(Symbol(base))[$(subs...)]
  end
end

"A subscript: a literal as its value, `:` as `:`, another lowered."
function _subscriptExpr(s::SC.Exp, simCode::SC.SIM_CODE)
  s isa SC.ICONST && return s.value
  s isa SC.WHOLE_DIMENSION && return :(:)
  return expToJuliaExpMTK(s, simCode)
end

"""
A subscript as the flattened names write it (`2`, `idx`); `?` for another expression.
"""
_subscriptText(s::SC.Exp)::String = s isa SC.ICONST ? string(s.value) : s isa SC.EXP_CREF ? SC.crefName(s.cref) : "?"

"""
`exp[subs...]`. With literal subscripts: an element of an array output of a call
(`tupleArrayElementAt`), the scalar variable of an array's element (a record's field array,
`frame_b.R.T[1, 1]`), an output of a function's call (`tupleElementCall`); else the indexing.
With a subscript that is not a literal: an element of an array known at the build
(`_dynamicSubscriptLookup`), of variables (`_variableSelectionExpr`), of a constant table
(`constTableLookup`, symbolic indices: Digital's `Buf3sTable[strength, UX01Conv[enable], ...]`);
else the indexing.
"""
function expToJuliaExpMTK(e::SC.ASUB, simCode::SC.SIM_CODE)::Expr
  local subs = Any[_subscriptExpr(s, simCode) for s in e.subs]
  all(s -> s isa Integer, subs) && return _literalSubscriptExpr(e, Int[s for s in subs], simCode)
  local inner = e.exp
  if inner isa SC.EXP_CREF
    local table = _dynamicSubscriptLookup(SC.crefName(inner.cref) * join("[" * _subscriptText(s) * "]" for s in e.subs), simCode)
    table === nothing || return table
  end
  if inner isa SC.ARRAY_EXP && length(subs) == 1
    local selection = _variableSelectionExpr(inner.elements, only(subs), simCode)
    selection === nothing || return selection
  end
  local innerCode = expToJuliaExpMTK(inner, simCode)
  local entry = inner isa SC.EXP_CREF ? get(simCode.stringToSimVarHT, SC.crefName(inner.cref), nothing) : nothing
  if inner isa SC.ARRAY_EXP || (entry !== nothing && last(entry).varKind isa SC.DATA_STRUCTURE)
    return quote OMBackend.CodeGeneration.constTableLookup($(innerCode), $(subs...)) end
  end
  return quote $(innerCode)[$(subs...)] end
end

function _literalSubscriptExpr(e::SC.ASUB, subs::Vector{Int}, simCode::SC.SIM_CODE)::Expr
  local inner = e.exp
  local output = _functionOutputOf(inner, simCode)
  if output !== nothing
    local (call, k) = output
    return :(OMBackend.CodeGeneration.tupleArrayElementAt($(QuoteNode(call.name)), $k, $(Tuple(subs)),
                                                         $(_modelicaFunctionCallArgs(call.args, simCode)...)))
  end
  if inner isa SC.EXP_CREF && isempty(inner.cref.subs)
    local entry = get(simCode.stringToSimVarHT, SC.crefName(inner.cref) * join("[$(i)]" for i in subs), nothing)
    entry === nothing || return quote $(Symbol(last(entry).name)) end
  end
  #= An element of a function's result (an output of several, an element of an array output:
     Xorshift128plus.initialState, Integer[4]): a term of its own for symbolic arguments (a
     parameter binding: MSL PlanarLoops_analytic's normalizeWithAssert(n)[2]: an array whose
     size follows the arguments is one opaque term). With literal arguments the indexing stays:
     the result is numbers, evaluated once for all its elements (_parameterBindingExpr). =#
  if length(subs) == 1 && _isModelFunctionCall(inner, simCode) &&
     !(inner.ty isa SC.TYPE_ARRAY && all(_isLiteralExp, inner.args))
    return :(OMBackend.CodeGeneration.tupleElementCall($(QuoteNode(inner.name)), $(only(subs)),
                                                      $(_modelicaFunctionCallArgs(inner.args, simCode)...)))
  end
  return quote $(expToJuliaExpMTK(inner, simCode))[$(subs...)] end
end

"Whether `e` calls one of the model's functions (SIM_CODE.functions)."
_isModelFunctionCall(e::SC.Exp, simCode::SC.SIM_CODE)::Bool =
  e isa SC.CALL && e.kind !== SC.CALL_BUILTIN && _isSimCodeFunctionName(String(e.name), simCode)

"""
Output `k` of a call of one of the model's functions, as `f(...)[k]` or `TSUB(f(...), k)`:
`(call, k)`, else nothing. Its elements are indexed through tupleArrayElementAt (plain
indexing of tupleElementCall's scalar Num fails).
"""
function _functionOutputOf(e::SC.Exp, simCode::SC.SIM_CODE)::Union{Nothing, Tuple{SC.CALL, Int}}
  if e isa SC.ASUB && _isModelFunctionCall(e.exp, simCode) && length(e.subs) == 1 && only(e.subs) isa SC.ICONST
    return (e.exp, only(e.subs).value)
  elseif e isa SC.TSUB && _isModelFunctionCall(e.exp, simCode)
    return (e.exp, e.index)
  end
  return nothing
end

"""
    _variableSelectionExpr(elements, index, simCode)

`{x1, x2, ...}[i]` of variables by a subscript `i` that is not a literal (SimCode's reference
by a variable subscript of a part before the last: Buildings' DX coils' `sta[i].nomVal`): the
lookup of the table of their values when all are known at the build; else the variable `i`
selects, `ifelse(i == 1, x1, ifelse(i == 2, x2, ...))`. Nothing for other elements.
"""
function _variableSelectionExpr(elements::Vector{SC.Exp}, index, simCode::SC.SIM_CODE)::Union{Nothing, Expr}
  isempty(elements) && return nothing
  all(x -> x isa SC.EXP_CREF && !(x.ty isa SC.TYPE_ARRAY), elements) || return nothing
  local names = String[SC.crefName(x.cref) for x in elements]
  local table = _knownValues(names, simCode)
  table === nothing || return :(OMBackend.CodeGeneration.constTableLookup($(table), $(index)))
  local ht = simCode.stringToSimVarHT
  all(n -> haskey(ht, n), names) || return nothing
  return quote $(_selectionChain(Symbol[Symbol(last(ht[n]).name) for n in names], index)) end
end

"`ifelse(i == 1, x1, ifelse(i == 2, x2, ... xn))`."
function _selectionChain(symbols::Vector{Symbol}, index)
  local chain = symbols[end]
  for k in (length(symbols) - 1):-1:1
    chain = :(ifelse($(index) == $(k), $(symbols[k]), $(chain)))
  end
  return chain
end

"""
    _dynamicSubscriptLookup(name, simCode)

`base[idx][2]`, a flattened name whose subscript is a variable (the array crefs the backend
flattened: CDL's Integer TimeTable `y[:] = val[idx, :]`, idx a discrete; its elements kept): the
lookup of the table of base's elements, all known at the build (the dimensions from the names
in the table), indexed at run time; for elements not known at the build (`fixed = false`
parameters the initialization computes, variables: Buildings' DX coils' `uacp[stage].UAcp`),
one subscript a variable, the element the index selects among their variables; nothing
otherwise.
"""
function _dynamicSubscriptLookup(name::String, simCode::SC.SIM_CODE)::Union{Expr, Nothing}
  #= a record array's field after the subscripts (`uacp[stage]_UAcp`): literal subscripts only there =#
  local m = match(r"^([^\[\]]+)((?:\[[^\[\]]+\])+)((?:[^\[\]]|\[[0-9]+\])*)$", name)
  m === nothing && return nothing
  local base = String(m.captures[1])
  local suffix = String(m.captures[3])
  local subs = String[String(x.captures[1]) for x in eachmatch(r"\[([^\[\]]+)\]", m.captures[2])]
  all(s -> tryparse(Int, s) !== nothing, subs) && return nothing
  local ht = simCode.stringToSimVarHT
  local index = Any[]
  for s in subs
    local i = tryparse(Int, s)
    if i !== nothing
      push!(index, i)
    else
      local entry = get(ht, s, nothing)
      entry === nothing && return nothing
      push!(index, Symbol(last(entry).name))
    end
  end
  #= the extent of each dimension: the elements along it from 1 =#
  local key = (d::Int, k::Int) -> base * join("[" * string(j == d ? k : 1) * "]" for j in eachindex(subs)) * suffix
  local dims = Int[]
  for d in eachindex(subs)
    local n = 0
    while haskey(ht, key(d, n + 1))
      n += 1
    end
    n == 0 && return nothing
    push!(dims, n)
  end
  local names = map(I -> base * join("[" * string(i) * "]" for i in Tuple(I)) * suffix, CartesianIndices(Tuple(dims)))
  local table = _knownValues(names, simCode)
  table === nothing || return :(OMBackend.CodeGeneration.constTableLookup($(table), $(index...)))
  local dynamic = findall(s -> tryparse(Int, s) === nothing, subs)
  length(dynamic) == 1 || return nothing
  local d = only(dynamic)
  local symbols = Symbol[]
  for k in 1:dims[d]
    local entry = get(ht, base * join("[" * (j == d ? string(k) : subs[j]) * "]" for j in eachindex(subs)) * suffix, nothing)
    entry === nothing && return nothing
    push!(symbols, Symbol(last(entry).name))
  end
  return quote $(_selectionChain(symbols, index[d])) end
end

"""
The values of the parameters `names` (an array of them), all known at the build, or nothing.
"""
function _knownValues(names::AbstractArray{String}, simCode::SC.SIM_CODE)::Union{Array{Float64}, Nothing}
  local ht = simCode.stringToSimVarHT
  local table = Array{Float64}(undef, size(names)...)
  for I in eachindex(names)
    local entry = get(ht, names[I], nothing)
    entry === nothing && return nothing
    local kind = last(entry).varKind
    (kind isa SC.PARAMETER && kind.bindExp !== nothing) || return nothing
    local b = kind.bindExp
    local v = SC.tryEvalNumeric(b, simCode)
    #= `integer(table[...][i, j])`: evaluated as the initial values are =#
    v === nothing && (v = _t0Number(() -> evalConstant(b, simCode)))
    v === nothing && return nothing
    table[I] = v
  end
  return table
end

#= ==================================================================================== =#
#= Arrays                                                                               =#
#= ==================================================================================== =#

"""
An expression of literals only (no variable, no call).
"""
function _isConstantExp(e::SC.Exp)::Bool
  e isa Union{SC.RCONST, SC.ICONST, SC.BCONST, SC.SCONST, SC.ENUM_LITERAL} && return true
  e isa SC.UNARY && return _isConstantExp(e.exp)
  e isa SC.BINARY && return _isConstantExp(e.exp1) && _isConstantExp(e.exp2)
  e isa SC.ARRAY_EXP && return all(_isConstantExp, e.elements)
  return false
end

"""
    handleArrayExp(e, simCode)

An array literal: of constants, the array evaluated at the build (rows stacked along the first
dimension: a three-dimensional literal's rows are matrices, Buildings' Borefields
TemporalSuperposition); with variables, a vector, a matrix by `hvcat` of its rows' elements.
"""
function handleArrayExp(e::SC.ARRAY_EXP, simCode::SC.SIM_CODE)::Expr
  local dims = e.ty isa SC.TYPE_ARRAY ? e.ty.dims : Int[length(e.elements)]
  local steps = first(dims)
  steps >= 0 || OMBackend.unsupported("an array literal of a size not known", e)
  local elemExprs = Expr[expToJuliaExpMTK(e.elements[i], simCode) for i in 1:steps]
  local values = nothing
  if all(_isConstantExp, e.elements)
    values = try
      Any[eval(x) for x in elemExprs]
    catch err
      OMBackend._fallback(err, :constantArrayElements; expect = Union{UndefVarError, MethodError})
      nothing
    end
  end
  if values !== nothing
    length(dims) >= 2 || return quote $[values...] end
    local arr = ndims(first(values)) >= 2 ? stack(values; dims = 1) : Matrix(transpose(stack(values)))
    return quote $(arr) end
  end
  length(dims) >= 2 || return quote [$(elemExprs...)] end
  local rows = e.elements[1:steps]
  if all(r -> r isa SC.ARRAY_EXP, rows) && steps > 0
    local nCols = length(first(rows).elements)
    if nCols > 0
      local scalars = Expr[expToJuliaExpMTK(x, simCode) for r in rows for x in r.elements]
      return quote hvcat($(ntuple(_ -> nCols, steps)), $(scalars...)) end
    end
  end
  return quote
    let rows = [$(elemExprs...)]
      ndims(first(rows)) >= 2 ? stack(rows; dims = 1) : Matrix(transpose(stack(rows)))
    end
  end
end

#= ==================================================================================== =#
#= Calls                                                                                =#
#= ==================================================================================== =#

"""
    _builtinCallMTK(e, simCode)

A call of a built-in: `der` and `pre` of an expression, `noEvent` of a Real if-expression as
`ifelse` (only the branch the condition chooses: the arithmetic if-expression turned a NaN of
the other branch into the result, MSL ContinuousMean's `noEvent(if time >= t_0 + t_eps then
(u - mu)/(time - t_0) else 0)`), `delay` from the model's history (delays.jl), `initial()` 0
in the equations (the initialization has its own), `homotopy` as the blend in λ where the
continuation acts (`HOMOTOPY_BLEND`), `Integer` of an enumeration the identity (it is its
index already), `sample` in a when body whether the instant is a tick (`if sampleTrigger` at
`when {occ, sampleTrigger}`); another by its implementation (MODELICA_BUILTIN_FUNCTIONS) or its
name.
"""
function _builtinCallMTK(e::SC.CALL, simCode::SC.SIM_CODE)::Expr
  local lower = x::SC.Exp -> expToJuliaExpMTK(x, simCode)
  local name = e.name
  if name === :der
    return _derivativeExpr(first(e.args), simCode)
  elseif name === :pre
    return _preExpr(first(e.args), simCode)
  elseif name === :noEvent && length(e.args) == 1 && _isRealIfExp(only(e.args))
    local ie = only(e.args)::SC.IFEXP
    return :(ifelse(OMBackend.CodeGeneration.MTK_CodeGenerationUtil._noEventCondition($(lower(ie.cond))), $(lower(ie.thenExp)), $(lower(ie.elseExp))))
  elseif name === :delay
    return _delayExpr(e.args, simCode)
  elseif name === :initial
    return quote 0 end
  elseif name === :homotopy
    local actual = lower(e.args[1])
    HOMOTOPY_BLEND[] || return actual
    return :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.modelica_homotopy($(actual), $(lower(e.args[2])), $(HOMOTOPY_LAMBDA)))
  elseif name === :Integer
    return lower(e.args[1])
  elseif name === :sample && PRE_FROM_SNAPSHOT[]
    return :(OMBackend.CodeGeneration._isSampleInstant(integrator.t, $(lower(e.args[1])), $(lower(e.args[2]))))
  elseif name === :String
    #= a String parameter's binding (createStringParameterAssignments): the argument types
       from the expressions, not from the values =#
    return AlgorithmicCodeGeneration.modelicaStringCall(e.args, lower)
  end
  local args = map(lower, e.args)
  local builtin = get(AlgorithmicCodeGeneration.MODELICA_BUILTIN_FUNCTIONS, String(name), nothing)
  builtin === nothing && return quote $(name)($(args...)) end
  local callee = Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(builtin))
  return quote $(callee)($(args...)) end
end

_isRealIfExp(e::SC.Exp)::Bool =
  e isa SC.IFEXP && !(e.cond isa SC.BCONST) && !_ifexpBranchIsNonReal(e.thenExp) && !_ifexpBranchIsNonReal(e.elseExp)

"""
    _derivativeExpr(arg, simCode)

`der` of an expression the passes left (an initial equation `der(w) = 0` whose `w` was replaced
by its definition, differentiated later): of an array its elements' (MultiBody's `der({c1,
c2})`), of a literal 0 (a parameter replaced by its value), of `-x` `-der(x)`, of a variable
`D(x)`, else `D(e)`.
"""
function _derivativeExpr(arg::SC.Exp, simCode::SC.SIM_CODE)::Expr
  arg isa SC.ARRAY_EXP && return Expr(:vect, (_derivativeExpr(x, simCode) for x in arg.elements)...)
  arg isa SC.RCONST && return quote 0.0 end
  arg isa SC.ICONST && return quote 0 end
  arg isa SC.BCONST && return quote false end
  (arg isa SC.UNARY && arg.op === SC.OP_UMINUS) && return :(-($(_derivativeExpr(arg.exp, simCode))))
  arg isa SC.EXP_CREF && return expToJuliaExpMTK(SC.DER(arg.cref), simCode)
  return quote D($(expToJuliaExpMTK(arg, simCode))) end
end

"""
    _preExpr(arg, simCode)

`pre` of an expression the passes left (a variable replaced by its definition): of an array its
elements', of a literal the literal, else the expression of the references' pre values.
"""
function _preExpr(arg::SC.Exp, simCode::SC.SIM_CODE)::Expr
  arg isa SC.ARRAY_EXP && return Expr(:vect, (_preExpr(x, simCode) for x in arg.elements)...)
  arg isa Union{SC.RCONST, SC.ICONST, SC.BCONST} && return quote $(arg.value) end
  local pre = SC.mapExp(x -> (x isa SC.EXP_CREF && x.cref.sym !== :time) ? SC.PRE(x.cref, x.ty) : x, arg)
  return expToJuliaExpMTK(pre, simCode)
end

"`delay(x, T[, Tmax])`: x's value T ago, from the model's history of x (delays.jl)."
function _delayExpr(args::Vector{SC.Exp}, simCode::SC.SIM_CODE)::Expr
  local x = expToJuliaExpMTK(args[1], simCode)
  local T = expToJuliaExpMTK(args[2], simCode)
  local k = _delayIndex!(string(args[1]) * "|" * string(args[2]), x, T)
  return :(OMBackend.CodeGeneration.delayLookup($(QuoteNode(DELAY_MODEL[])), $k, t, $T, $x))
end

"""
    _modelicaFunctionCallExpr(e, simCode)

A call of a function: OMFrontend's delay functions (`OpenModelica.Internal.delay2/3`, as
OpenModelica: functions whose body was the identity) from the model's history; a Boolean
reduction with a symbolic form (lowerKnownSymbolicFunctionCall); a Julia mirror of a Modelica
function (MODELICA_BUILTIN_FUNCTIONS: Modelica.Math.Vectors.length); Modelica.Utilities by
OMRuntimeExternalC; another by its name, its record arguments by their fields.
"""
function _modelicaFunctionCallExpr(e::SC.CALL, simCode::SC.SIM_CODE)::Expr
  local name = String(e.name)
  name in ("OpenModelica_Internal_delay2", "OpenModelica_Internal_delay3") && return _delayExpr(e.args, simCode)
  local lowered = lowerKnownSymbolicFunctionCall(name, e.args, simCode)
  lowered === nothing || return lowered
  local args = _modelicaFunctionCallArgs(e.args, simCode)
  local builtinSym = get(AlgorithmicCodeGeneration.MODELICA_BUILTIN_FUNCTIONS, name, nothing)
  if builtinSym !== nothing
    local builtinCallee = Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)),
                                        QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(builtinSym))
    return Expr(:call, builtinCallee, args...)
  end
  local runtimeName = get(AlgorithmicCodeGeneration.MODELICA_UTILITIES_TO_RUNTIME_C, name, nothing)
  local callee = runtimeName === nothing ? e.name :
    :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.RuntimeCCall($(Expr(:., :OMRuntimeExternalC, QuoteNode(runtimeName)))))
  return Expr(:call, callee, args...)
end

"""
    _modelicaFunctionCallArgs(args, simCode)

The arguments of a call of a function: a record by its fields (`flattenRecordCallArg`), an
element of an array of records by its fields' elements, a Complex-returning call by its
outputs (`_expandComplexReturnArg`), another lowered.
"""
function _modelicaFunctionCallArgs(args::Vector{SC.Exp}, simCode::SC.SIM_CODE)::Vector{Any}
  local out = Any[]
  for arg in args
    local fields = flattenRecordCallArg(arg, simCode)
    isempty(fields) || (append!(out, fields); continue)
    local elementFields = _recordArrayElementCallArgs(arg, simCode)
    isempty(elementFields) || (append!(out, elementFields); continue)
    local extracts = _expandComplexReturnArg(arg, simCode)
    extracts === nothing || (append!(out, extracts); continue)
    push!(out, expToJuliaExpMTK(arg, simCode))
  end
  return out
end

"""
An element of a named array of records by subscripts that are not literals (`data[i]` in a
reduction: the ideal gases' `h_T(data[i], T, ...)`, MSL Media mixtures): that element of
each field array, `data_MM[i]` (the module-level arrays of _recordArrayFieldArrays). Empty
otherwise.
"""
function _recordArrayElementCallArgs(arg::SC.Exp, simCode::SC.SIM_CODE)::Vector{Any}
  (arg isa SC.ASUB && arg.exp isa SC.EXP_CREF) || return Any[]
  local elemTy = arg.exp.ty
  local nDims = 0
  while elemTy isa SC.TYPE_ARRAY
    nDims += length(elemTy.dims)
    elemTy = elemTy.elementType
  end
  (elemTy isa SC.TYPE_COMPLEX && elemTy.isRecord) || return Any[]
  (nDims == 0 || length(arg.subs) == nDims) || return Any[]
  local base = SC.crefName(arg.exp.cref)
  occursin('[', base) && return Any[]
  any(s -> s isa Union{SC.RANGE, SC.ARRAY_EXP, SC.WHOLE_DIMENSION}, arg.subs) && return Any[]
  local idx = Any[expToJuliaExpMTK(s, simCode) for s in arg.subs]
  return Any[Expr(:ref, Symbol(base * COMPONENT_SEPARATOR * f), idx...) for f in elemTy.fieldNames]
end

"""
A Complex-returning argument (a call of a record output, a record constructor) as its
fields' values for an enclosing call: `tupleElementCall(:f, k, args...)` for k = 1..nFields
of a call, the field expressions of a record literal; nothing for another argument (or a
record of fewer than two fields).
"""
function _expandComplexReturnArg(arg::SC.Exp, simCode::SC.SIM_CODE)::Union{Nothing, Vector{Any}}
  if arg isa SC.CALL && arg.ty isa SC.TYPE_COMPLEX
    local nFields = length(arg.ty.fieldNames)
    nFields >= 2 || return nothing
    local innerArgs = Any[]
    for inner in arg.args
      local flat = flattenRecordCallArg(inner, simCode)
      isempty(flat) || (append!(innerArgs, flat); continue)
      local nested = _expandComplexReturnArg(inner, simCode)
      nested === nothing || (append!(innerArgs, nested); continue)
      push!(innerArgs, expToJuliaExpMTK(inner, simCode))
    end
    return Any[:(OMBackend.CodeGeneration.tupleElementCall($(QuoteNode(arg.name)), $k, $(innerArgs...))) for k in 1:nFields]
  elseif arg isa SC.RECORD && arg.ty isa SC.TYPE_COMPLEX
    length(arg.exps) >= 2 || return nothing
    return Any[expToJuliaExpMTK(x, simCode) for x in arg.exps]
  end
  return nothing
end

# TODO: unify cref resolution into one function consulting both the SimCode (state /
# numeric-param lookup tables) and the module-level bindings (String parameters, data
# structures), so callers need not special-case the latter.

"""
In a callback's condition or affect: a state from `x`, a parameter from the integrator's
parameters. A `p` in scope was the callback set's, the event parameters as symbols (the
condition a Num: Buildings' weather data readers, `canRepeatWeatherFile - (modTimAux >
tNext)`), where the condition did not bind its own.
"""
function getIdxForLookupMTK(crefAsStr::String, simCode::SC.SIM_CODE)
  crefAsStr == "time" && return :t
  local simVar = last(simCode.stringToSimVarHT[crefAsStr])
  if !SC.isParameter(simVar)
    return Expr(:call, getindex, :x, Expr(:call, :getindex, :lookuptableStates, :(Symbol($(crefAsStr)))))
  end
  return Expr(:call, getindex, :(integrator.p), Expr(:call, :getindex, :lookuptableParams, :(Symbol($(crefAsStr)))))
end

#= ==================================================================================== =#
#= Values at the build                                                                  =#
#= ==================================================================================== =#

"""
    evalExpression(e, simCode; keepTunable = false)

`e` evaluated at the build: its parameters replaced by their bindings, recursively (a
parameter alias `actualGlobalSeed = globalSeed_seed` left the bare name otherwise), lowered
and evaluated; the lowered expression where it reads a variable or a parameter without a
binding. With `keepTunable`, tunable parameters stay references (_substituteBoundParameters).
"""
function evalExpression(e::SC.Exp, simCode::SC.SIM_CODE; keepTunable::Bool = false)::Expr
  local shouldEval = Ref(true)
  local substituted = _substituteBoundParameters(e, simCode; shouldEval, keepTunable)
  local jl = expToJuliaExpMTK(substituted, simCode)
  shouldEval[] || return quote $(jl) end
  local value = try
    eval(jl)
  catch err
    #= left unevaluated: it reads a name that is not a constant here =#
    OMBackend._fallback(err, :evalExpression; expect = Union{UndefVarError, MethodError})
    jl
  end
  return quote $(value) end
end

"""
Whether `e` reads a tunable parameter (withTunableParameters), directly or through the
bindings of the parameters it reads.
"""
function _readsTunableParameter(e::SC.Exp, simCode::SC.SIM_CODE)::Bool
  isempty(OMBackend.TUNABLE_PARAMETERS[]) && return false
  local resolved = _substituteBoundParameters(e, simCode; keepTunable = true)
  return any(OMBackend.isTunableParameter, SC.crefNames!(OrderedSet{String}(), resolved))
end

"""
    _evalNumeric(e, valMap)

A scalar expression evaluated numerically at t0 from `valMap` (the parameters' values and the
variables' values so far): the arithmetic, relations, Boolean operators, `integer`, `floor`,
`ceil`, `abs`, `sqrt`, `min`, `max`, `smooth`, `noEvent`; nothing for anything else.
"""
function _evalNumeric(e::SC.Exp, valMap::Dict{Symbol, Float64})::Union{Float64, Nothing}
  e isa Union{SC.RCONST, SC.ICONST} && return Float64(e.value)
  e isa SC.BCONST && return e.value ? 1.0 : 0.0
  if e isa Union{SC.EXP_CREF, SC.ASUB}
    local name = SC.scalarReferenceName(e)
    name === nothing && return nothing
    return name == "time" ? 0.0 : get(valMap, Symbol(name), nothing)
  end
  e isa SC.CAST && return _evalNumeric(e.exp, valMap)
  if e isa SC.UNARY
    local a = _evalNumeric(e.exp, valMap)
    return a === nothing ? nothing : -a
  elseif e isa SC.LUNARY
    local a = _evalNumeric(e.exp, valMap)
    return a === nothing ? nothing : (a == 0.0 ? 1.0 : 0.0)
  elseif e isa SC.IFEXP
    local c = _evalNumeric(e.cond, valMap)
    return c === nothing ? nothing : _evalNumeric(c != 0.0 ? e.thenExp : e.elseExp, valMap)
  elseif e isa Union{SC.BINARY, SC.RELATION, SC.LBINARY}
    local a = _evalNumeric(e.exp1, valMap); a === nothing && return nothing
    local b = _evalNumeric(e.exp2, valMap); b === nothing && return nothing
    if e isa SC.RELATION
      local holds = SC.compareValues(a, e.op, b)
      return holds === nothing ? nothing : holds ? 1.0 : 0.0
    end
    e isa SC.LBINARY && return (e.op === SC.OP_AND ? (a != 0.0 && b != 0.0) : (a != 0.0 || b != 0.0)) ? 1.0 : 0.0
    e.op === SC.OP_ADD && return a + b
    e.op === SC.OP_SUB && return a - b
    e.op === SC.OP_MUL && return a * b
    e.op === SC.OP_DIV && return b == 0.0 ? nothing : a / b
    e.op === SC.OP_POW && return (a < 0.0 && !isinteger(b)) ? nothing : a ^ b
    return nothing
  elseif e isa SC.CALL && e.kind === SC.CALL_BUILTIN && !isempty(e.args)
    e.name === :smooth && return length(e.args) == 2 ? _evalNumeric(e.args[2], valMap) : nothing
    local a = _evalNumeric(e.args[1], valMap); a === nothing && return nothing
    if e.name in (:min, :max) && length(e.args) == 2
      local b = _evalNumeric(e.args[2], valMap); b === nothing && return nothing
      return e.name === :min ? min(a, b) : max(a, b)
    end
    e.name === :integer && return isfinite(a) ? floor(a) : nothing
    e.name === :floor && return floor(a)
    e.name === :ceil && return ceil(a)
    e.name === :abs && return abs(a)
    e.name === :sqrt && return a < 0.0 ? nothing : sqrt(a)
    e.name in (:noEvent, :float, :Real, :Integer) && return a
  end
  return nothing
end

"""
A statement with the loop iterator `name` replaced by `value` (an inner loop of the same
iterator keeps its own).
"""
function _withIteratorValue(s::SC.Statement, name::Symbol, value::Int)::SC.Statement
  local sub = e::SC.Exp -> SC.mapExp(x -> (x isa SC.EXP_CREF && x.cref.sym === name && isempty(x.cref.subs)) ? SC.ICONST(value) : x, e)
  local body = b::Vector{SC.Statement} -> SC.Statement[_withIteratorValue(x, name, value) for x in b]
  if s isa SC.FOR
    s.iterator.name === name && return s
    return SC.FOR(SC.Iterator(s.iterator.name, sub(s.iterator.range)), body(s.body), s.source)
  elseif s isa SC.IF
    return SC.IF(SC.Branch[SC.Branch(sub(b.condition), body(b.body)) for b in s.branches], body(s.elseBody), s.source)
  elseif s isa SC.ASSIGN
    return SC.ASSIGN(sub(s.left), sub(s.right), s.source)
  end
  return s
end

"""
    _execInitAlgStmts!(valMap, statements, written)

The statements of an initial algorithm interpreted at t0 into `valMap`: assignments of a
variable, if/elseif/else, for-loops over constant integer ranges; another statement, an
expression `_evalNumeric` does not evaluate, is skipped (its names keep their values).
`written` (when given) collects the names an assignment wrote.
"""
function _execInitAlgStmts!(valMap::Dict{Symbol, Float64}, statements::Vector{SC.Statement},
                            written::Union{Nothing, Set{Symbol}})::Nothing
  foreach(s -> _execInitAlgStmt!(valMap, s, written), statements)
  return nothing
end

function _execInitAlgStmt!(valMap::Dict{Symbol, Float64}, s::SC.ASSIGN, written::Union{Nothing, Set{Symbol}})::Nothing
  local name = SC.scalarReferenceName(s.left)
  name === nothing && return nothing
  local v = _evalNumeric(s.right, valMap)
  v === nothing && return nothing
  valMap[Symbol(name)] = v
  written === nothing || push!(written, Symbol(name))
  return nothing
end

function _execInitAlgStmt!(valMap::Dict{Symbol, Float64}, s::SC.IF, written::Union{Nothing, Set{Symbol}})::Nothing
  for b in s.branches
    local c = _evalNumeric(b.condition, valMap)
    c === nothing && return nothing
    c != 0.0 && return _execInitAlgStmts!(valMap, b.body, written)
  end
  return _execInitAlgStmts!(valMap, s.elseBody, written)
end

function _execInitAlgStmt!(valMap::Dict{Symbol, Float64}, s::SC.FOR, written::Union{Nothing, Set{Symbol}})::Nothing
  local range = s.iterator.range
  range isa SC.RANGE || return nothing
  local startV = _evalNumeric(range.start, valMap)
  local stopV = _evalNumeric(range.stop, valMap)
  local stepV = range.step === nothing ? 1.0 : something(_evalNumeric(range.step, valMap), NaN)
  #= Int() of a bound beyond Int64 throws an InexactError =#
  (startV !== nothing && stopV !== nothing && isfinite(stepV) && stepV != 0.0 &&
   isinteger(startV) && isinteger(stopV) && isinteger(stepV) &&
   max(abs(startV), abs(stopV), abs(stepV)) < 2.0^62) || return nothing
  for iv in Int(startV):Int(stepV):Int(stopV), b in s.body
    _execInitAlgStmt!(valMap, _withIteratorValue(b, s.iterator.name, iv), written)
  end
  return nothing
end

_execInitAlgStmt!(::Dict{Symbol, Float64}, ::SC.Statement, ::Union{Nothing, Set{Symbol}})::Nothing = nothing

"""
    _seedInitialAlgValues!(valMap, simCode, written = nothing)

`valMap` with the values the initial algorithm sections (`section`: an `initial algorithm`, an
algorithm's `when initial()`) assign at t0 (trapezoid `count`, `T_start`), in statement order.
Without them a discrete set imperatively in an `initial algorithm` (no `start` attribute) was
0.0 where the initial branches of if-equations are chosen: the wrong branch. Not the `initial()`
arms of when-equations and discrete clusters: the DAE initialization starts their discretes at
their start values and the clusters settle them after it, so the branches must read those (as
the old pipeline did: only its sections had DAE statements); seeded, MSL 4.1.0 IMC_YD's main
switches started open in the branches and closed in the initialization, which then failed and
moved the logical delays' tSwitch.
"""
function _seedInitialAlgValues!(valMap::Dict{Symbol, Float64}, simCode::SC.SIM_CODE,
                                written::Union{Nothing, Set{Symbol}} = nothing)::Dict{Symbol, Float64}
  for ia in simCode.initialAlgorithms
    ia.section && _execInitAlgStmts!(valMap, ia.statements, written)
  end
  return valMap
end

#= ==================================================================================== =#
#= The t0 value map                                                                     =#
#= ==================================================================================== =#

#= One-entry memo: the map is requested once per if-equation branch but only depends on the
   simCode object, and the forward-evaluation sweep is not free. Fields: (the simCode, valMap,
   explicit, trusted-derived pairs). The simCode is kept and compared by identity: an
   `objectid` comes from its fields' addresses, which the GC gives a later model's. =#
const _T0_MAP_CACHE = Ref{Union{Nothing, Tuple{SC.SIM_CODE, Dict{Symbol, Float64}, Set{Symbol}, Vector{Pair{Symbol, Float64}}}}}(nothing)

"Whether the t0 map memo is that of `simCode`."
_t0MapCachedFor(simCode::SC.SIM_CODE)::Bool = (local c = _T0_MAP_CACHE[]; c !== nothing && c[1] === simCode)

"""
The number `f()` (evalConstant, evalSimCodeParameter) gives, or nothing when it is not a
constant. A BINARY/LBINARY comes back as evalExpression's `:block` around its value (around
the Expr itself when it reads a variable).
"""
function _t0Number(f::F)::Union{Float64, Nothing} where {F}
  local raw = try
    f()
  catch err
    OMBackend._fallback(err, :t0Number; only = OMBackend.UnsupportedLowering)
    return nothing
  end
  while raw isa Expr && raw.head === :block
    local body = filter(a -> !(a isa LineNumberNode), raw.args)
    length(body) == 1 || return nothing
    raw = body[1]
  end
  return raw isa Real ? Float64(raw) : nothing
end

"""
The forms evalConstant evaluates: a literal, or a BINARY/LBINARY through evalExpression.
The t0 map asks only for those (anything else is not a constant, which is the rule, not a
failure).
"""
_isEvaluableConstant(e::Union{Nothing, SC.Exp})::Bool =
  e isa Union{SC.BCONST, SC.ICONST, SC.RCONST, SC.SCONST, SC.BINARY, SC.LBINARY}

"""
`sv`'s start value as a number, or nothing.
"""
function _startNumber(sv::SC.SIMVAR, simCode::SC.SIM_CODE)::Union{Float64, Nothing}
  local start = sv.attributes.start
  return _isEvaluableConstant(start) ? _t0Number(() -> evalConstant(start, simCode)) : nothing
end

_isFixedStart(sv::SC.SIMVAR)::Bool = sv.attributes.fixed == SC.BCONST(true)

"""
    _buildT0ValueMapAndExplicit(simCode)

The values at t0 by symbol: the parameters' (binding, else start), the states' and algebraic
variables' starts (0.0 without one), the initial algorithms' results, extended through the
explicit residuals (`_forwardEvalT0!`); and the symbols whose value has an actual source
(`explicit`, not the 0.0 default). The consequences of the values safe for hard decisions
(parameters, fixed starts, the initial algorithms' results) override guess-grade starts first,
so a branch never reads a value the fixed initial configuration contradicts.
"""
function _buildT0ValueMapAndExplicit(simCode::SC.SIM_CODE)::Tuple{Dict{Symbol, Float64}, Set{Symbol}}
  if _t0MapCachedFor(simCode)
    local cached = _T0_MAP_CACHE[]
    return (copy(cached[2]), copy(cached[3]))
  end
  local valMap = Dict{Symbol, Float64}()
  local explicit = Set{Symbol}()
  local trusted = Set{Symbol}()
  for (key, (_, sv)) in simCode.stringToSimVarHT
    local sym = Symbol(key)
    if sv.varKind isa SC.PARAMETER
      local bind = sv.varKind.bindExp
      local pval = _isEvaluableConstant(bind) ? _t0Number(() -> evalConstant(bind, simCode)) : nothing
      pval === nothing && (pval = _startNumber(sv, simCode))
      if pval !== nothing
        valMap[sym] = pval
        push!(explicit, sym)
        push!(trusted, sym)
      end
    elseif SC.isStateOrAlgebraic(sv)
      local sval = _startNumber(sv, simCode)
      if sval !== nothing
        push!(explicit, sym)
        _isFixedStart(sv) && push!(trusted, sym)
      end
      valMap[sym] = something(sval, 0.0)
    end
  end
  #= the names the initial algorithms assign (an assignment equal to the start value is an
     initial algorithm's result too) =#
  local written = Set{Symbol}()
  _seedInitialAlgValues!(valMap, simCode, written)
  union!(explicit, written)
  union!(trusted, written)
  local trustedBase = copy(trusted)
  local trustedMap = Dict{Symbol, Float64}(k => valMap[k] for k in trusted if haskey(valMap, k))
  _forwardEvalT0!(trustedMap, trusted, simCode)
  local trustedDerived = Pair{Symbol, Float64}[k => trustedMap[k] for k in trusted
                                               if !(k in trustedBase) && haskey(trustedMap, k)]
  for (k, v) in trustedMap
    valMap[k] = v
    push!(explicit, k)
  end
  _forwardEvalT0!(valMap, explicit, simCode)
  _T0_MAP_CACHE[] = (simCode, copy(valMap), copy(explicit), trustedDerived)
  return (valMap, explicit)
end

"""
The values derived from the trusted ones (deterministic consequences of parameters and
fixed starts), for overriding guess-grade initial values at the build.
"""
function buildT0TrustedDerivedPairs(simCode::SC.SIM_CODE)::Vector{Pair{Symbol, Float64}}
  _t0MapCachedFor(simCode) || _buildT0ValueMapAndExplicit(simCode)
  return copy(_T0_MAP_CACHE[][4])
end

_buildT0ValueMap(simCode::SC.SIM_CODE)::Dict{Symbol, Float64} = first(_buildT0ValueMapAndExplicit(simCode))

"""
True when every variable-position Symbol of the expression is explicit. Call heads and
qualified references are code, not variables.
"""
function _exprSymbolsExplicit(@nospecialize(e), explicit::Set{Symbol})::Bool
  e isa Symbol && return e in explicit
  e isa Expr || return true
  e.head == :. && return true
  for a in (e.head == :call ? e.args[2:end] : e.args)
    a isa LineNumberNode && continue
    _exprSymbolsExplicit(a, explicit) || return false
  end
  return true
end

"""
Whole Float64 literals in index position fault plain numeric evaluation (the lowering
emits `x[3.0]`): Int literals.
"""
_intifyFloatIndices(@nospecialize(x)) = x
function _intifyFloatIndices(x::Expr)
  if x.head == :ref && length(x.args) >= 2
    return Expr(x.head, _intifyFloatIndices(x.args[1]),
                (i isa Float64 && isinteger(i) ? Int(i) : _intifyFloatIndices(i) for i in x.args[2:end])...)
  end
  return Expr(x.head, (_intifyFloatIndices(a) for a in x.args)...)
end

"""
    evalCausalRHSAtT0(rhsExpr, valMap, explicit = Set{Symbol}())

A causalized right-hand side evaluated at the t0 value map; nothing where it is not
statically evaluable or reads a variable without an explicit value (a seed contaminated by
0.0 defaults can pull the consistent-initial-conditions solve to a wrong root).
"""
function evalCausalRHSAtT0(@nospecialize(rhsExpr), valMap::Dict{Symbol, Float64},
                           explicit::Set{Symbol} = Set{Symbol}())::Union{Float64, Nothing}
  ccall(:jl_generating_output, Cint, ()) != 0 && return nothing
  _exprSymbolsExplicit(rhsExpr, explicit) || return nothing
  local numExpr = _intifyFloatIndices(_substituteExprValues(rhsExpr, valMap))
  try
    #= generated Modelica function bindings live in the parent CodeGeneration module =#
    local result = Core.eval(parentmodule(@__MODULE__), numExpr)
    local value = result isa Number ? Float64(result) : Base.invokelatest(SymbolicUtils.unwrap, result)
    value isa Number || return nothing
    return isfinite(value) ? Float64(value) : nothing
  catch err
    #= a name not resolved yet, a Modelica function outside its domain =#
    OMBackend._fallback(err, :evalCausalRHSAtT0; expect = Union{UndefVarError, MethodError})
    return nothing
  end
end

_t0ContainsDer(@nospecialize(x)) = false
_t0ContainsDer(x::Expr) = (x.head == :call && !isempty(x.args) && x.args[1] in (:der, :D)) || any(_t0ContainsDer, x.args)

"""
The explicit residuals `0 = x - rhs` evaluated forward at t0, extending the value map
through chains the start attributes do not cover (frame positions, guarded line-force
lengths); with the residuals of the if-equations' initial branches, chosen again as the map
grows. Operands must be explicit already, so defaulted values never contaminate a
filled-in entry.
"""
function _forwardEvalT0!(valMap::Dict{Symbol, Float64}, explicit::Set{Symbol}, simCode::SC.SIM_CODE)::Nothing
  ccall(:jl_generating_output, Cint, ()) != 0 && return nothing
  local lowered = Dict{SC.Exp, Any}()
  local pair = function (e::SC.Exp)
    (e isa SC.BINARY && e.op === SC.OP_SUB && e.exp1 isa SC.EXP_CREF) || return nothing
    local target = Symbol(SC.crefName(e.exp1.cref))
    target in explicit && return nothing
    local rhs = get!(lowered, e.exp2) do
      try
        expToJuliaExpMTK(e.exp2, simCode)
      catch err
        OMBackend._fallback(err, :forwardEvalT0Lowering; only = OMBackend.UnsupportedLowering)
        nothing
      end
    end
    (rhs === nothing || _t0ContainsDer(rhs)) && return nothing
    return (target, rhs)
  end
  for _ in 1:3
    local pairs = Tuple{Symbol, Any}[]
    for eq in simCode.residualEquations
      local p = pair(eq.exp)
      p === nothing || push!(pairs, p)
    end
    local env = Dict{String, Float64}(string(k) => valMap[k] for k in explicit if haskey(valMap, k))
    for ifEq in simCode.ifEquations
      local branch = SC.activeInitialBranch(ifEq, env)
      branch === nothing && continue
      for r in branch.residualEquations
        local p = pair(r.exp)
        p === nothing || push!(pairs, p)
      end
    end
    local progressed = false
    for _ in 1:20
      local round = false
      for (target, rhs) in pairs
        target in explicit && continue
        local v = evalCausalRHSAtT0(rhs, valMap, explicit)
        v === nothing && continue
        valMap[target] = v
        push!(explicit, target)
        round = true
      end
      round || break
      progressed = true
    end
    progressed || break
  end
  return nothing
end

"""
    evalInitialCondition(mtkCond, simCode; closedBoundary = false, extraVals = nothing)

Whether a condition is false at t0: its zero-crossing function (`zc ~ 0`) evaluated with the
t0 value map (and `extraVals`, the relays' t0 values: an operand that is an if-equation's
target defaulted to 0 and chose the wrong initial branch) is positive, at zero `closedBoundary`
decides. The caller inverts: `ifCond = !evalInitialCondition(...)`. True where it cannot be
evaluated.
"""
function evalInitialCondition(mtkCond::Expr, simCode::SC.SIM_CODE; closedBoundary::Bool = false,
                              extraVals::Union{Nothing, AbstractDict{Symbol, Float64}} = nothing)::Bool
  #= during the precompile output eval would mutate this closed module's bindings =#
  ccall(:jl_generating_output, Cint, ()) != 0 && return true
  local valMap = _buildT0ValueMap(simCode)
  extraVals === nothing || merge!(valMap, extraVals)
  local numExpr = _substituteExprValues(_extractZeroCrossingLHS(mtkCond), valMap)
  local value = try
    _numberOf(eval(numExpr))
  catch err
    #= a name without a t0 value, a Modelica function outside its domain =#
    OMBackend._fallback(err, :evalInitialCondition; expect = Union{UndefVarError, MethodError}, impact = :result)
    @warn "evalInitialCondition: failed to evaluate, defaulting to true" exception=(err, catch_backtrace())
    return true
  end
  return closedBoundary ? value > 0.0 : value >= 0.0
end

"""
An evaluated condition value as a Float64 (a symbolic constant unwrapped; 0.0 otherwise).
"""
function _numberOf(@nospecialize(result))::Float64
  result isa Number && return Float64(result)
  local unwrapped = Base.invokelatest(SymbolicUtils.unwrap, result)
  unwrapped isa Number && return Float64(unwrapped)
  local valued = Base.invokelatest(Symbolics.value, result)
  return Float64(valued isa Number ? valued : 0.0)
end

"The left side of a zero-crossing equation `lhs ~ 0`; another expression itself."
function _extractZeroCrossingLHS(expr::Expr)
  (expr.head == :call && length(expr.args) == 3 && expr.args[1] == :~) && return expr.args[2]
  return expr
end

#= ==================================================================================== =#
#= Tuple equations                                                                      =#
#= ==================================================================================== =#

"""
    _tupleEquationTargets(lhs, rhs)

The targets of a tuple equation `(t1, t2, ...) = f(...)` of a pure call: per variable of the
left side, (its reference, the call's output or output element it equals, that one's type). A
wildcard has none; an array of references takes the output's elements (MSL
Blocks.Continuous.Filter: `({r[1], r[2]}, _, _, _) = roots.lowPass(cr, ...)`). `nothing` for
another left side.
"""
function _tupleEquationTargets(lhs::SC.Exp, rhs::SC.Exp)::Union{Nothing, Vector{Tuple{SC.EXP_CREF, SC.Exp, SC.SType}}}
  (lhs isa SC.TUPLE && rhs isa SC.CALL && rhs.kind !== SC.CALL_IMPURE_FUNCTION && rhs.ty isa SC.TYPE_TUPLE) || return nothing
  local types = rhs.ty.types
  length(types) >= length(lhs.elements) || return nothing
  local out = Tuple{SC.EXP_CREF, SC.Exp, SC.SType}[]
  for (k, t) in enumerate(lhs.elements)
    if t isa SC.WILD
      continue
    elseif t isa SC.EXP_CREF
      push!(out, (t, SC.TSUB(rhs, k, types[k]), types[k]))
    elseif t isa SC.ARRAY_EXP && types[k] isa SC.TYPE_ARRAY && all(e -> e isa SC.EXP_CREF, t.elements)
      for (j, e) in enumerate(t.elements)
        push!(out, (e, SC.ASUB(SC.TSUB(rhs, k, types[k]), SC.Exp[SC.ICONST(j)]), types[k].elementType))
      end
    else
      return nothing
    end
  end
  return out
end

"""
`e` with `time` read as 0.0, the start time of the build.
"""
_timeAtBuildStart(e::SC.Exp)::SC.Exp =
  SC.mapExp(x -> (x isa SC.EXP_CREF && x.cref.sym === :time && isempty(x.cref.subs)) ? SC.RCONST(0.0) : x, e)

"""
`(p1, p2, ...) = f(...)`, every free parameter of the equation alone on the left and none in
the call (Buildings' borehole resistances: `(x, Rgb, Rgg, RCondGro) =
internalResistancesOneUTube(...)`, in HexInternalElement with states, in its validations
without): each p_k bound to the call's k-th output (or output element:
_tupleEquationTargets), its value at the build where the call interprets with the
parameters' values, else the output's expression (its symbolic term did not fold with a
String argument, the instance name). Not an impure f (one call for each output).
"""
function _bindTupleOfFreeParameters!(lhs::SC.Exp, rhs::SC.Exp, freeParams::Vector{String}, simCode::SC.SIM_CODE)::Bool
  local targets = _tupleEquationTargets(lhs, rhs)
  targets === nothing && return false
  local binds = Tuple{String, SC.Exp, SC.SType}[(SC.crefName(t.cref), e, ty) for (t, e, ty) in targets]
  local names = String[first(b) for b in binds]
  (!isempty(names) && allunique(names) && issetequal(names, freeParams)) || return false
  any(c -> c in names, SC.allCrefNames(rhs)) && return false
  local ht = simCode.stringToSimVarHT
  for (name, element, ty) in binds
    local (idx, sv) = ht[name]
    local value = ty isa SC.TYPE_REAL ? SC.valueAtBuildStart(element, simCode) : nothing
    ht[name] = (idx, SC.withKind(sv, SC.PARAMETER(value === nothing ? element : SC.RCONST(value))))
  end
  return true
end

"""
The names of the references of `e` (not der() or pre()), in bottom-up order.
"""
function _referenceNames(e::SC.Exp)::Vector{String}
  local out = String[]
  SC.mapExp(x -> (x isa SC.EXP_CREF && push!(out, SC.crefName(x.cref)); x), e)
  return out
end

"""
Whether `e` reads the variable `name` (of Integer type, with `integer`).
"""
_readsName(e::SC.Exp, name::String; integer::Bool = false)::Bool =
  SC.anyExp(x -> x isa SC.EXP_CREF && SC.crefName(x.cref) == name && (!integer || x.ty isa SC.TYPE_INTEGER), e)

"""
`name` bound to `binding` in the table (a free parameter the build solved).
"""
function _bindParameter!(simCode::SC.SIM_CODE, name::String, binding::SC.Exp)::Nothing
  local (idx, sv) = simCode.stringToSimVarHT[name]
  simCode.stringToSimVarHT[name] = (idx, SC.withKind(sv, SC.PARAMETER(binding)))
  return nothing
end

"""
    solveParametricInitialEquations!(simCode)

The initial equations that only involve parameters (no states or algebraic variables) solved
numerically for their free parameters (`fixed = false`, no binding): the table bound to the
solved values, to a fixed point (each pass may bind a free parameter another equation reads;
at most one pass per equation finishes a chain in the worst order), then the coupled ones
(`_solveCoupledParametricInitialEquations!`). At the build the start time is 0 (`t0 = time`,
MSL Blocks.Math.Mean; simulateFromBuild refuses another start time for such a model,
_startTimeGuard).
"""
function solveParametricInitialEquations!(simCode::SC.SIM_CODE)::Nothing
  local ht = simCode.stringToSimVarHT
  local solvedValueExp = (x::Float64, asInteger::Bool) -> asInteger && isfinite(x) ? SC.ICONST(Int(round(x))) : SC.RCONST(x)
  local nEq = length(simCode.initialEquations)
  local solvedThisPass = true
  local pass = 0
  while solvedThisPass && pass <= nEq
    solvedThisPass = false
    pass += 1
    local solvedNames = String[]
    for ieq in simCode.initialEquations
      isParametricOnlyEquation(ieq, simCode) || continue
      local (ieqLhs, ieqRhs) = map(_timeAtBuildStart, equationSides(ieq))
      local freeParams = unique(String[n for side in (ieqLhs, ieqRhs) for n in _referenceNames(side)
                                       if (local entry = get(ht, n, nothing); entry !== nothing &&
                                           SC.isParameter(last(entry)) && !SC.hasBindingExp(last(entry)))])
      if _bindTupleOfFreeParameters!(ieqLhs, ieqRhs, freeParams, simCode)
        append!(solvedNames, freeParams)
        solvedThisPass = true
        continue
      end
      length(freeParams) == 1 || continue
      local freeName = only(freeParams)
      #= A tunable parameter is set at run time: solved here, the free one kept its compiled
         value in every run (`q = 2 * p`). Alone on a side it is bound to the other side (`p =
         q`, `2 * p = q`), and follows p as a bound parameter does; otherwise it stays unbound
         and the initialization computes it for the run's values (`q * q = p`). As an unbound
         parameter alone on the right it was assigned nowhere: q = 0. =#
      if any(side -> _readsTunableParameter(side, simCode), (ieqLhs, ieqRhs))
        local isFree = e::SC.Exp -> e isa SC.EXP_CREF && SC.crefName(e.cref) == freeName
        local other = isFree(ieqLhs) ? ieqRhs : isFree(ieqRhs) ? ieqLhs : nothing
        if other !== nothing && !_readsName(other, freeName)
          _bindParameter!(simCode, freeName, other)
          push!(solvedNames, freeName)
          solvedThisPass = true
        end
        continue
      end
      local guess = something(_startNumber(last(ht[freeName]), simCode), 0.1)
      #= The residual LHS - RHS = 0, the bound parameters replaced recursively (an alias chain
         actualGlobalSeed = globalSeed_seed, globalSeed_seed solved by an earlier initial
         equation of the fixed point), the free parameter the scalar Newton variable. =#
      local skipFree = OrderedSet{String}([freeName])
      local lhsEvalOk = Ref(true)
      local rhsEvalOk = Ref(true)
      local lhsSubst = _substituteBoundParameters(ieqLhs, simCode; skipNames = skipFree, shouldEval = lhsEvalOk)
      local rhsSubst = _substituteBoundParameters(ieqRhs, simCode; skipNames = skipFree, shouldEval = rhsEvalOk)
      #= The free parameter on the left (`globalSeed_seed = automaticGlobalSeed(0.0)`): the
         sides swapped, so the left is the constants' side (evaluating the bare free name was an
         UndefVarError). =#
      if _readsName(lhsSubst, freeName) && !_readsName(rhsSubst, freeName)
        (lhsSubst, rhsSubst) = (rhsSubst, lhsSubst)
        (lhsEvalOk, rhsEvalOk) = (rhsEvalOk, lhsEvalOk)
      end
      local freeIsInteger = _readsName(ieqLhs, freeName; integer = true) || _readsName(ieqRhs, freeName; integer = true)
      #= A reference substitution left unresolved: the fixed point tries again. =#
      lhsEvalOk[] || continue
      #= Interpreted first: it calls the model's functions, which the generated code does not
         have yet (CDL's TimeTable: t0 = round(integer(time/timeRange)*timeRange, 6), read by
         its table's startTime at module level). =#
      local lhsVal = SC.valueAtBuildStart(lhsSubst, simCode)
      if lhsVal === nothing
        lhsVal = try
          local raw = eval(expToJuliaExpMTK(lhsSubst, simCode))
          raw isa Symbolics.Num ? Float64(Symbolics.unwrap(raw)) : Float64(raw)
        catch err
          OMBackend._fallback(err, :parametricInitLhs; expect = Union{UndefVarError, MethodError}, impact = :result)
          @warn "[SIMCODE: solveParametricInitialEquations] could not evaluate LHS" freeName err
          continue
        end
      end
      #= A free parameter that aliases a bound parameter or a literal (`globalSeed_seed =
         globalSeed_fixedSeed`): no Newton (it may keep the start guess where the generated
         residual does not depend on its argument: world age, module bindings). =#
      if rhsSubst isa SC.EXP_CREF && _readsName(rhsSubst, freeName)
        _bindParameter!(simCode, freeName, solvedValueExp(lhsVal, freeIsInteger))
        push!(solvedNames, freeName)
        solvedThisPass = true
        continue
      end
      rhsEvalOk[] || continue
      local freeSymbol = Symbol(freeName)
      local rhsJl = expToJuliaExpMTK(rhsSubst, simCode)
      #= f(x) = lhsVal - rhs(x) =#
      local residualFn = try
        eval(Expr(:->, freeSymbol, Expr(:call, :-, lhsVal, rhsJl)))
      catch e
        OMBackend._fallback(e, :parametricInitResidual; expect = Union{UndefVarError, MethodError}, impact = :result)
        @warn "[SIMCODE: solveParametricInitialEquations] could not build residual" freeName e
        continue
      end
      #= Newton-Raphson (invokelatest: world age). A call of the residual can throw where the
         right side reads a parameter whose binding the frontend could not fold (neither a value
         nor the free one: an UndefVarError); without the catch the translation was aborted. =#
      local x = guess
      local newtonOk = true
      local lastStep = Inf
      try
        for _ in 1:100
          local fx = Base.invokelatest(residualFn, x)
          abs(fx) < 1e-10 && break
          local h = max(abs(x) * 1e-8, 1e-12)
          local dfx = (Base.invokelatest(residualFn, x + h) - Base.invokelatest(residualFn, x - h)) / (2h)
          abs(dfx) < 1e-15 && break
          lastStep = fx / dfx
          x -= lastStep
        end
      catch err
        OMBackend._fallback(err, :parametricInitNewton; expect = Union{UndefVarError, MethodError}, impact = :result)
        @warn "[SIMCODE: solveParametricInitialEquations] residual call threw, skipping" freeName err
        newtonOk = false
      end
      #= Converged: a small residual, or a small last step (the rounding of `0 = 1e9*G - 2e9` is
         about 4e-7). Otherwise the parameter stays unbound, for the initialization
         (generateInitialEquationsAsConstraints takes an equation reading one). =#
      newtonOk = newtonOk && let fx = try Base.invokelatest(residualFn, x) catch; NaN end
        isfinite(x) && isfinite(fx) &&
          (abs(fx) <= 1e-8 * max(1.0, abs(lhsVal)) || abs(lastStep) <= 1e-10 * max(1.0, abs(x)))
      end
      newtonOk || continue
      @debug "[SIMCODE: solveParametricInitialEquations] solved $freeName = $x (from initial equation)"
      _bindParameter!(simCode, freeName, solvedValueExp(x, freeIsInteger))
      push!(solvedNames, freeName)
      solvedThisPass = true
    end
    isempty(solvedNames) ||
      @debug "[SIMCODE: solveParametricInitialEquations] pass $pass solved $(length(solvedNames)) parameter(s)" solvedNames
  end
  _solveCoupledParametricInitialEquations!(simCode)
  return nothing
end

"""
The parametric initial equations left, of several free parameters each, solved together
where they are as many as their free parameters (MSL 4 SMPM_ResistiveBraking: the
steady-state currents idq_sr from two coupled equations; one at a time they were not
solved, and the free parameters kept their start values, 0). Not where one reads a tunable
parameter (its run's value is not known here) or the solve does not converge: those stay
unbound, as before.
"""
function _solveCoupledParametricInitialEquations!(simCode::SC.SIM_CODE)::Nothing
  local ht = simCode.stringToSimVarHT
  local isFreeParam = key::String -> begin
    local entry = get(ht, key, nothing)
    entry !== nothing && SC.isParameter(last(entry)) && !SC.hasBindingExp(last(entry))
  end
  local sides = Tuple{SC.Exp, SC.Exp}[]
  local free = OrderedSet{String}()
  for ieq in simCode.initialEquations
    isParametricOnlyEquation(ieq, simCode) || continue
    local (lhs, rhs) = map(_timeAtBuildStart, equationSides(ieq))
    local names = OrderedSet{String}(n for side in (lhs, rhs) for n in _referenceNames(side) if isFreeParam(n))
    length(names) >= 2 || continue
    any(side -> _readsTunableParameter(side, simCode), (lhs, rhs)) && return nothing
    push!(sides, (lhs, rhs)); union!(free, names)
  end
  (isempty(sides) || length(sides) != length(free)) && return nothing
  local names = collect(free)
  local residuals = Any[]
  for (lhs, rhs) in sides
    local ok = Ref(true)
    local l = _substituteBoundParameters(lhs, simCode; skipNames = free, shouldEval = ok)
    local r = _substituteBoundParameters(rhs, simCode; skipNames = free, shouldEval = ok)
    ok[] || return nothing
    push!(residuals, :($(expToJuliaExpMTK(l, simCode)) - $(expToJuliaExpMTK(r, simCode))))
  end
  local residualFn = try
    eval(Expr(:->, Expr(:tuple, Symbol.(names)...), Expr(:vect, residuals...)))
  catch err
    OMBackend._fallback(err, :coupledParametricInitResidual; expect = Union{UndefVarError, MethodError}, impact = :result)
    return nothing
  end
  local f = x -> Float64.(Base.invokelatest(residualFn, x...))
  local x = Float64[something(_startNumber(last(ht[n]), simCode), 0.0) for n in names]
  local n = length(x)
  local converged = false
  try
    for _ in 1:50
      local fx = f(x)
      all(isfinite, fx) || break
      if maximum(abs, fx) <= 1e-10 * max(1.0, maximum(abs, x))
        converged = true
        break
      end
      local J = Matrix{Float64}(undef, n, n)
      for k in 1:n
        local h = max(abs(x[k]) * 1e-7, 1e-9)
        local xp = copy(x); xp[k] += h
        local xm = copy(x); xm[k] -= h
        J[:, k] = (f(xp) .- f(xm)) ./ (2h)
      end
      local step = J \ fx
      all(isfinite, step) || break
      x .-= step
    end
  catch err
    OMBackend._fallback(err, :coupledParametricInitNewton; expect = Union{UndefVarError, MethodError, LinearAlgebra.SingularException}, impact = :result)
    return nothing
  end
  converged || return nothing
  foreach(((k, name),) -> _bindParameter!(simCode, name, SC.RCONST(x[k])), enumerate(names))
  @debug "[SIMCODE: solveParametricInitialEquations] solved $(n) coupled free parameters" names x
  return nothing
end

#= ==================================================================================== =#
#= If-equations                                                                         =#
#= ==================================================================================== =#

"""
True when every variable reference in `condition` is a DISCRETE simvar or a PARAMETER (no
STATE / ALG_VARIABLE / `time`): the if-expression gates directly on the held value, the
discrete's own update event localises the switch, so no ifCond relay is needed. A condition
with initial() is a relation (its crossing function a constant of either sign), not a gate
on the runtime value, where initial() is false (MSL LimIntegrator's `initial() and not
limitsAtInit` never held).
"""
function _ifConditionAllDiscreteOrParameter(condition::SC.Exp, simCode::SC.SIM_CODE)::Bool
  _hasInitialCall(condition) && return false
  local refs = SC.crefNames!(OrderedSet{String}(), condition)
  isempty(refs) && return false
  local ht = simCode.stringToSimVarHT
  for name in refs
    name == "time" && return false
    local entry = get(ht, name, nothing)
    entry === nothing && return false
    last(entry).varKind isa Union{SC.DISCRETE, SC.PARAMETER} || return false
  end
  return true
end

_hasInitialCall(condition::SC.Exp)::Bool = SC.anyExp(x -> SC.isBuiltin(x, :initial), condition)

"""
    _ifConditionLiteral(condition, simCode)

True when an if-equation's branch condition is evaluated as written in the residuals: its
relations are all inside noEvent() (MLS 3.7.4: no events), the references outside them
discretes or parameters (their own events localise a change), and it has no initial(), pre()
or other event operator. An ifCond relay with hysteresis kept MSL FluidHeatFlow IdealPump's
`noEvent(abs(w) < small)` false after the speed returned to zero: `abs(w) - small` never falls
below the hysteresis, and the else branch divided by the zero speed.
"""
function _ifConditionLiteral(condition::SC.Exp, simCode::SC.SIM_CODE)::Bool
  SC.anyExp(x -> SC.isBuiltin(x, :noEvent), condition) || return false
  isempty(_eventRelations(condition)) || return false
  local eventOperator = SC.anyExp(condition) do x
    x isa SC.PRE || (x isa SC.CALL && x.kind === SC.CALL_BUILTIN && x.name in (:initial, :terminal, :sample, :edge, :change, :delay))
  end
  eventOperator && return false
  local outside = OrderedSet{String}()
  local visit = function (e::SC.Exp)
    SC.isBuiltin(e, :noEvent) && return
    e isa SC.EXP_CREF && push!(outside, SC.crefName(e.cref))
    foreach(visit, SC.children(e))
  end
  visit(condition)
  local ht = simCode.stringToSimVarHT
  return all(n -> haskey(ht, n) && last(ht[n]).varKind isa Union{SC.DISCRETE, SC.PARAMETER}, outside)
end

"""
True when every non-else branch condition of an if-equation is discrete/parameter, so the
residual gates directly on the boolean condition instead of emitting a continuous ifCond
relay callback.
"""
function _allBranchConditionsDiscrete(branches::AbstractVector{<:SC.BRANCH}, simCode::SC.SIM_CODE)::Bool
  return all(b -> b.targets == -1 || _ifConditionAllDiscreteOrParameter(b.condition, simCode), branches)
end

"""
Residuals pair across branches by the variable their causalized form defines, with the
positional index only as a fallback: source-order differences between sibling branches
(hoisted nested ifs, lifted if-expressions) made pure positional pairing relay the wrong
variable.
"""
function _causalLhsKey(@nospecialize(lhsExpr))::String
  local e = lhsExpr isa Expr ? Base.remove_linenums!(copy(lhsExpr)) : lhsExpr
  while e isa Expr && e.head === :block && length(e.args) == 1
    e = e.args[1]
  end
  return string(e)
end

function _branchResidualForLhs(branch::SC.BRANCH, lhsKey::String, fallbackIdx::Int, simCode::SC.SIM_CODE)::SC.RESIDUAL_EQUATION
  for r in branch.residualEquations
    _causalLhsKey(last(deCausalize(r, simCode))) == lhsKey && return r
  end
  return branch.residualEquations[min(fallbackIdx, length(branch.residualEquations))]
end

"""
    generateIfExpressions(branches, target, resEqIdx, identifier, simCode; subIdentifier = 1,
                          lhsKey = nothing, residualForm = false)

The value of an if-equation's residual `resEqIdx` as nested `ifelse` over its branches from
`target`: per branch its right-hand side (causal form; its residual with `residualForm`, where
the branches define different variables), gated by the branch's condition where every
condition is discrete or a parameter (its own update event localises the step: no relay, and
no start value of a relay that picks the wrong branch), as written where its relations are all
inside noEvent() (`_ifConditionLiteral`), else by its `ifCond` relay parameter
(`ifCond<identifier><subIdentifier> == 1`: a discrete parameter the solver never perturbs).
For

```modelica
if uMax < time then der(y) = uMax; elseif uMin < time then der(y) = uMin; else der(y) = u; end if;
```

`D(y) ~ ifelse(ifCond11 == 1, uMax, ifelse(ifCond12 == 1, uMin, u))`.
"""
function generateIfExpressions(branches::AbstractVector{<:SC.BRANCH}, target::Int, resEqIdx::Int, identifier::Int,
                               simCode::SC.SIM_CODE; subIdentifier::Int = 1,
                               lhsKey::Union{String, Nothing} = nothing, residualForm::Bool = false)
  local branch = branches[target]
  local selEq = (lhsKey === nothing || residualForm) ?
                branch.residualEquations[min(resEqIdx, length(branch.residualEquations))] :
                _branchResidualForLhs(branch, lhsKey, resEqIdx, simCode)
  local value = _guardNonIntegerPowerBasesForEagerBranch(
    residualForm ? expToJuliaExpMTK(selEq.exp, simCode) : first(deCausalize(selEq, simCode)))
  branch.targets == -1 && return :($(value))
  local cond = _ifConditionAllDiscreteOrParameter(branch.condition, simCode) ?
    :($(expToJuliaExpMTK(branch.condition, simCode)) > 0.5) :
    _ifConditionLiteral(branch.condition, simCode) ?
    :(OMBackend.CodeGeneration.MTK_CodeGenerationUtil._noEventCondition($(expToJuliaExpMTK(branch.condition, simCode)))) :
    :($(Symbol(string("ifCond", identifier, subIdentifier))) == 1)
  quote
    ModelingToolkit.ifelse($(cond),
                           $(value),
                           $(generateIfExpressions(branches, branch.targets, resEqIdx, identifier, simCode;
                                                   subIdentifier = subIdentifier + 1, lhsKey, residualForm)))
  end
end

"""
    deCausalize(eq, simCode)

The sides of a residual `a - b` (or `a + b`, ...) as `(b, a)`: the value and the variable of
the causal form the code generation gave the residuals of if-equations' branches. TODO: this
assumes residuals in that form.
"""
function deCausalize(eq::SC.RESIDUAL_EQUATION, simCode::SC.SIM_CODE)::Tuple{Any, Any}
  local e = eq.exp
  e isa SC.BINARY || OMBackend.unsupported("equation", eq)
  return (expToJuliaExpMTK(e.exp2, simCode), expToJuliaExpMTK(e.exp1, simCode))
end

#= ==================================================================================== =#
#= Eager branches                                                                       =#
#= ==================================================================================== =#

function _singleBlockPayload(@nospecialize(e))
  while e isa Expr && e.head === :block
    local payload = Any[a for a in e.args if !(a isa LineNumberNode)]
    length(payload) == 1 || return e
    e = payload[1]
  end
  return e
end

function _isNonIntegerJuliaConstant(@nospecialize(e))::Bool
  local v = _singleBlockPayload(e)
  v isa Integer && return false
  return v isa Real && !isinteger(v)
end

"""
ModelingToolkit.ifelse and the arithmetic if-expression evaluate both branches. Guard only
those eager branch payloads against one-sided event-localization noise around a relation
boundary; ordinary powers keep their native domain semantics.
"""
function _guardNonIntegerPowerBasesForEagerBranch(@nospecialize(e))
  e isa Expr || return e
  local args = Any[_guardNonIntegerPowerBasesForEagerBranch(a) for a in e.args]
  if e.head === :call && length(args) == 3 && args[1] == :^ && _isNonIntegerJuliaConstant(args[3])
    return :(max($(args[2]), 0.0) ^ $(args[3]))
  end
  return Expr(e.head, args...)
end

#= ==================================================================================== =#
#= The system                                                                           =#
#= ==================================================================================== =#

"""
    performStructuralSimplify(; observedFilter = nothing, split = !DIRECT_RHS_GENERATION[], indexReduced = false)

The structural simplification of `firstOrderSystem` into `reducedSystem` (tearing only for a
system OMBackend reduced, `indexReduced`); with an observed
filter, the observed equations kept those whose variable a pattern matches, with the observed
variables they read and the system's and the initialization equations read. With
ENABLE_BACKEND_LOGGING, the system before is dumped to `backend/codeGen/preStructuralSimplify.log`
(the path taken at codegen time, in the model's run directory).
"""
function performStructuralSimplify(; observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                                   split::Bool = !OMBackend.DIRECT_RHS_GENERATION[], indexReduced::Bool = false)::Expr
  local dumpExpr = :(nothing)
  @BACKEND_LOGGING dumpExpr = dumpPreStructuralSimplifyExpr(
      OMBackend.logPath("backend/codeGen", "preStructuralSimplify.log"))
  #= a system OMBackend reduced (SIM_CODE.indexReduced) is of index 1 with its states chosen: MTK only tears =#
  local simplifyExpr = indexReduced ?
    quote
      $dumpExpr
      reducedSystem = OMBackend.CodeGeneration.structural_simplify(firstOrderSystem; simplify = true, allow_parameter=true, split = $(split), fully_determined = false)
    end :
    quote
      $dumpExpr
      reducedSystem = OMBackend.CodeGeneration.structural_simplify(firstOrderSystem; simplify = true, allow_parameter=true, split = $(split))
    end
  (observedFilter === nothing || isempty(observedFilter)) && return simplifyExpr
  local patternStrings = observedFilter isa Vector{Regex} ? [p.pattern for p in observedFilter] : observedFilter
  return quote
    $simplifyExpr
    local _obsPatterns = [Regex(p) for p in $(patternStrings)]
    local _allObs = ModelingToolkit.observed(reducedSystem)
    local _nBefore = length(_allObs)
    #= Keep matched observed equations AND the transitive closure of observed
       variables their RHS references; dropping a dependency leaves it
       referenced-but-undefined in the residual/init build_function. =#
    local _obsByLhs = Dict(string(eq.lhs) => eq for eq in _allObs)
    local _keepNames = OrderedSet{String}()
    local _stack = String[string(eq.lhs) for eq in _allObs
                          if any(p -> occursin(p, string(eq.lhs)), _obsPatterns)]
    #= Also seed from observed vars referenced by the system equations and the
       initialization equations: the residual / init build_function references them
       (the initialization rows are inlined to the states through the observed
       equations: MSL IMC_YD's rotorCage.electroMagneticConverter.V_m = Complex(0, 0)),
       so their defs must survive. =#
    for _seq in Iterators.flatten((ModelingToolkit.equations(reducedSystem),
                                   ModelingToolkit.initialization_equations(reducedSystem)))
      for _side in (_seq.lhs, _seq.rhs)
        for _v in Symbolics.get_variables(_side)
          local _vn = string(_v)
          haskey(_obsByLhs, _vn) && push!(_stack, _vn)
        end
      end
    end
    while !isempty(_stack)
      local _nm = pop!(_stack)
      (_nm in _keepNames) && continue
      push!(_keepNames, _nm)
      local _eq = get(_obsByLhs, _nm, nothing)
      _eq === nothing && continue
      for _v in Symbolics.get_variables(_eq.rhs)
        local _vn = string(_v)
        (haskey(_obsByLhs, _vn) && !(_vn in _keepNames)) && push!(_stack, _vn)
      end
    end
    local _filteredObs = filter(eq -> string(eq.lhs) in _keepNames, _allObs)
    if length(_filteredObs) < _nBefore
      @debug "[MTK GEN: observed] observedFilter: kept $(length(_filteredObs)) of $(_nBefore) MTK observed equations"
      reducedSystem = Setfield.set(reducedSystem, Setfield.PropertyLens{:observed}(), _filteredObs)
    end
  end
end

"""
    odeSystemWithEvents(hasEvents, modelName; hasObserved = false)

The ODESystem's constructor: with the if-equations' events (SymbolicContinuousCallback with
discrete parameters: the ifCond relays live in the parameter vector) but then no observed
equations (MTK's complete(sys) injects them into every callback's AffectSystem, which fails
tearing where they introduce variables the callback cannot solve: they are added to the
reduced system after structural_simplify, MTK_CodeGeneration.jl); the initial equations as
`initialization_eqs` (constraints that hold at t = 0: `PID.gainPID.y = 0`, not only guesses).
"""
function odeSystemWithEvents(hasEvents::Bool, modelName::String; hasObserved::Bool = false)::Expr
  if hasEvents
    return :(System(eqs, t, vars, parameters;
                       name=:($(Symbol($modelName))),
                       continuous_events = events, guesses = initialValues,
                       initialization_eqs = initialConstraintEqs))
  elseif hasObserved
    return :(System(eqs, t, vars, parameters;
                       name=:($(Symbol($modelName))), guesses = initialValues,
                       observed = observedEqs,
                       initialization_eqs = initialConstraintEqs))
  end
  return :(System(eqs, t, vars, parameters;
                     name=:($(Symbol($modelName))), guesses = initialValues,
                     initialization_eqs = initialConstraintEqs))
end

using ExportAll
@exportAll()

end #= module MTK_CodeGenerationUtil =#
