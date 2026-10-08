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
  MTK / Symbolics-specific lowering helpers carved out of CodeGenerationUtil.
  Generic predicates, parameter folding and Expr utilities still live in
  CodeGenerationUtil; this module is the home for code that is coupled to
  ModelingToolkit / Symbolics / SymbolicUtils.
=#
module MTK_CodeGenerationUtil

import DataStructures
using DataStructures: OrderedSet
import MacroTools
using MetaModelica
using MetaModelica: Cons
using Setfield
using DocStringExtensions
using ModelingToolkit
using LinearAlgebra

using ...FrontendUtil
import ...FrontendUtil.Util
using ...Backend
import ...Backend.BDAE
using ...SimulationCode

import ...OMBackend
import ..AlgorithmicCodeGeneration
import ..CodeGenerationUtil
using ..CodeGenerationUtil
import ..CodeGeneration: lowerKnownSymbolicFunctionCall
import ..MTKDump: dumpPreStructuralSimplifyExpr

import ...@BACKEND_LOGGING
import ...COMPONENT_SEPARATOR

import Absyn
import DAE
import MetaGraphs
import OMFrontend
import OMParser
import Symbolics
import Symbolics.RuntimeGeneratedFunctions
import SymbolicUtils
import OMRuntimeExternalC

#= Bool-context lowering for DiscreteCallback condition functions: emits real
   `&&` / `||` / `!` where `expToJuliaExpMTK` uses arithmetic encoding for
   compatibility with Symbolics.jl in residual contexts. =#
function expToJuliaBoolMTK(@nospecialize(cond::DAE.Exp), simCode; cachedChange::Bool = false)
  @match cond begin
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      local lhs = expToJuliaExpMTK(e1, simCode)
      local rhs = expToJuliaExpMTK(e2, simCode)
      local opSym = DAE_OP_toJuliaOperator(op)
      :($opSym($lhs, $rhs))
    end
    DAE.LBINARY(exp1 = e1, operator = DAE.AND(__), exp2 = e2) =>
      :($(_boolOperandMTK(e1, simCode, cachedChange)) && $(_boolOperandMTK(e2, simCode, cachedChange)))
    DAE.LBINARY(exp1 = e1, operator = DAE.OR(__), exp2 = e2) =>
      :($(_boolOperandMTK(e1, simCode, cachedChange)) || $(_boolOperandMTK(e2, simCode, cachedChange)))
    DAE.LUNARY(operator = DAE.NOT(__), exp = e1) =>
      :(!($(_boolOperandMTK(e1, simCode, cachedChange))))
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        expToJuliaBoolMTK(innerArgs[1], simCode; cachedChange = cachedChange)
      else
        expToJuliaExpMTK(cond, simCode)
      end
    end
    #= Modelica `change(x)` ≡ `pre(x) != x`. Used as a trigger condition in
       discrete callbacks synthesised from non-when algorithm bodies whose
       LHS is discrete-time. The DiscreteCallback environment exposes the
       previous step's state vector as `integrator.uprev` and the current
       one as `x` (via the cache lookup the surrounding affect generator
       builds), so the runtime test is a straightforward index comparison. =#
    DAE.CALL(Absyn.IDENT("change"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        local curr = expToJuliaBoolMTK(innerArgs[1], simCode; cachedChange = cachedChange)
        local prev = _preValueLookup(innerArgs[1], simCode; cachedChange = cachedChange)
        :(($(curr)) != ($(prev)))
      else
        expToJuliaExpMTK(cond, simCode)
      end
    end
    #= Modelica `edge(b)` ≡ `b and not pre(b)` — the rising-edge detector. Uses
       the same previous-value machinery as `change`; the `!= 0` normalises a
       Boolean discrete stored as a Float64 (0.0/1.0) so `&&`/`!` stay boolean. =#
    DAE.CALL(Absyn.IDENT("edge"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        local curr = expToJuliaBoolMTK(innerArgs[1], simCode; cachedChange = cachedChange)
        local prev = _preValueLookup(innerArgs[1], simCode; cachedChange = cachedChange)
        :((($(curr)) != 0) && !((($(prev)) != 0)))
      else
        expToJuliaExpMTK(cond, simCode)
      end
    end
    #= `initial()` is true exactly during the initialisation phase. The
       DiscreteCallback we use for synthesised when-equations does not run
       during MTK's InitializationProblem — the body's init-pass fires
       through the `__runInitialAlgorithm!` path the BDAECreate lifter sets
       up. So `initial()` as a runtime check is always false here. =#
    DAE.CALL(Absyn.IDENT("initial"), _, _) => :(false)
    #= `terminal()` fires only at end-of-simulation; the per-step DiscreteCallback has no finalize hook, so it never triggers here. =#
    DAE.CALL(Absyn.IDENT("terminal"), _, _) => :(false)
    _ => expToJuliaExpMTK(cond, simCode)
  end
end

#= An operand of not/and/or: a Boolean variable (or its pre()) is stored as a
   Float64 (0.0/1.0) in the state vector, and `!active` threw a MethodError
   (`when not active`, the StateGraph steps). =#
function _boolOperandMTK(@nospecialize(e::DAE.Exp), simCode, cachedChange::Bool)
  local x = expToJuliaBoolMTK(e, simCode; cachedChange = cachedChange)
  local isBool = e isa DAE.RELATION || e isa DAE.LBINARY || e isa DAE.LUNARY || e isa DAE.BCONST ||
                 (e isa DAE.CALL && string(e.path) in ("change", "edge", "initial", "terminal"))
  return isBool ? x : :($x != 0)
end

#= Compile-time helper: emit the Julia expression that reads the previous
   value of a CREF from the discrete callback's `integrator.uprev` /
   parameter table. Falls back to `expToJuliaExpMTK` (which gives the
   current-value lookup) for non-CREF arguments — `pre(constant)` and
   `pre(parameter)` are the same as the current value. =#
function _preValueLookup(@nospecialize(arg::DAE.Exp), simCode; cachedChange::Bool = false)
  @match arg begin
    DAE.CREF(componentRef = cr) => begin
      local crefStr = string(arg)
      local ht = simCode.stringToSimVarHT
      if !haskey(ht, crefStr)
        return expToJuliaExpMTK(arg, simCode)
      end
      local (_, sv) = ht[crefStr]
      if SimulationCode.isParameter(sv)
        return expToJuliaExpMTK(arg, simCode)
      end
      #= States and discretes both live on the integrator's state vector;
         the surrounding callback codegen has populated `lookuptableStates`
         with `Symbol(name) => index`. Read through the integrator: the
         affect binds the model's variables by name, and one named `x`
         replaced the state vector `x` (a BoundsError). =#
      if cachedChange
        return :(get(_changePreValues,
                     Symbol($(string(sv.name))),
                     integrator.u[lookuptableStates[Symbol($(string(sv.name)))]]))
      end
      :(integrator.uprev[lookuptableStates[Symbol($(string(sv.name)))]])
    end
    _ => expToJuliaExpMTK(arg, simCode)
  end
end

"""
Transforms a DAE Condition into a MTK continuous condition.
"""
function transformToMTKContinuousCondition(cond, simCode; atInitial::Bool = false)
  # @match patterns are DAE.* only; convert SIM-side conditions at entry.
  if cond isa SimulationCode.Exp
    cond = SimulationCode.toDAEExp(cond)
  end
  res = @match cond begin
    DAE.RELATION(e1, DAE.LESS(__), e2) => begin
      :($(expToJuliaExpMTK(e1, simCode)) - $(expToJuliaExpMTK(e2, simCode)))
    end
    DAE.RELATION(e1, DAE.LESSEQ(__), e2) => begin
      :($(expToJuliaExpMTK(e1, simCode)) - $(expToJuliaExpMTK(e2, simCode)))
    end
    DAE.RELATION(e1, DAE.GREATER(__), e2) => begin
      :($(expToJuliaExpMTK(e2, simCode)) - $(expToJuliaExpMTK(e1, simCode)))
    end
    DAE.RELATION(e1, DAE.GREATEREQ(__), e2) => begin
      :($(expToJuliaExpMTK(e2, simCode)) - $(expToJuliaExpMTK(e1, simCode)))
    end
    #= Boolean-valued sub-conditions: encode `b` (true=1, false=0) such that the
       result is *negative* when the original condition is TRUE — the convention
       used by min/max for OR/AND composition and by evalInitialCondition. The
       previous `b - 0.5` form had inverted polarity. =#
    DAE.RELATION(e1, DAE.EQUAL(__), e2) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)))
    end
    DAE.RELATION(e1, DAE.NEQUAL(__), e2) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)))
    end
    DAE.CREF(__) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)))
    end
    DAE.LBINARY(e1, DAE.OR(__), e2) => begin
      :(min($(transformToMTKContinuousCondition(e1, simCode; atInitial = atInitial)),
            $(transformToMTKContinuousCondition(e2, simCode; atInitial = atInitial))))
    end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      :(max($(transformToMTKContinuousCondition(e1, simCode; atInitial = atInitial)),
            $(transformToMTKContinuousCondition(e2, simCode; atInitial = atInitial))))
    end
    #= Logical NOT: negate the inner condition =#
    DAE.LUNARY(DAE.NOT(__), e) => begin
      :(-($(transformToMTKContinuousCondition(e, simCode; atInitial = atInitial))))
    end
    #= Strip noEvent wrapper and recurse =#
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        transformToMTKContinuousCondition(innerArgs[1], simCode; atInitial = atInitial)
      else
        OMBackend.unsupported("noEvent with several arguments in a condition", cond)
      end
    end
    #= initial() is true during the initialization (`atInitial`: the
       condition's initial value) and false after it: a constant crossing
       function, negative (true) or positive (false). The event iteration's
       start re-evaluation reads the runtime one (MSL FluxTubes' Tellinen
       hysteresis stayed in its `if initial()` branch, k = 0.01, dHyst = 0). =#
    DAE.CALL(Absyn.IDENT("initial"), _, _) => begin
      atInitial ? :(-1) : :(1)
    end
    #= General function call as boolean condition: same polarity rule. =#
    DAE.CALL(__) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)))
    end
    _ => begin
      OMBackend.unsupported("condition expression", cond)
    end
  end
  return res
end

"""
Transforms a DAE Condition into a MTK continuous condition equation.
"""
function transformToMTKContinuousConditionEquation(cond, simCode; atInitial::Bool = false)
  # @match patterns are DAE.* only; convert SIM-side conditions at entry.
  if cond isa SimulationCode.Exp
    cond = SimulationCode.toDAEExp(cond)
  end
  res = @match cond begin
    DAE.RELATION(e1, DAE.LESS(__), e2) => begin
      :($(expToJuliaExpMTK(e1, simCode)) - $(expToJuliaExpMTK(e2, simCode)) ~ 0)
    end
    DAE.RELATION(e1, DAE.LESSEQ(__), e2) => begin
      :($(expToJuliaExpMTK(e1, simCode)) - $(expToJuliaExpMTK(e2, simCode)) ~ 0)
    end
    DAE.RELATION(e1, DAE.GREATER(__), e2) => begin
      :($(expToJuliaExpMTK(e2, simCode)) - $(expToJuliaExpMTK(e1, simCode)) ~ 0)
    end
    DAE.RELATION(e1, DAE.GREATEREQ(__), e2) => begin
      :($(expToJuliaExpMTK(e2, simCode)) - $(expToJuliaExpMTK(e1, simCode)) ~ 0)
    end
    #= Equality / inequality / boolean CREF / general boolean CALL all encode a
       Bool-valued condition. Convention: the zero-crossing function is negative
       when the original condition is TRUE, positive when FALSE (see the OR / AND
       composition with min / max, and `evalInitialCondition`). A Bool b
       cast to Float64 maps `true=>1.0`, `false=>0.0`, so the encoding must be
       `0.5 - b`, not `b - 0.5` — the latter inverts polarity, e.g. `nperiod == 0`
       with nperiod = -1 (false) produces `0 - 0.5 = -0.5` and is mis-read as TRUE
       inside `min(...)`, flipping the whole disjunction. =#
    DAE.RELATION(e1, DAE.EQUAL(__), e2) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)) ~ 0)
    end
    DAE.RELATION(e1, DAE.NEQUAL(__), e2) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)) ~ 0)
    end
    #= Boolean variable used directly as condition: same polarity rule. =#
    DAE.CREF(__) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)) ~ 0)
    end
    DAE.LBINARY(e1, DAE.OR(__), e2) => begin
      :(min($(transformToMTKContinuousCondition(e1, simCode; atInitial = atInitial)),
            $(transformToMTKContinuousCondition(e2, simCode; atInitial = atInitial))) ~ 0)
    end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      :(max($(transformToMTKContinuousCondition(e1, simCode; atInitial = atInitial)),
            $(transformToMTKContinuousCondition(e2, simCode; atInitial = atInitial))) ~ 0)
    end
    #= Logical NOT: negate the inner condition =#
    DAE.LUNARY(DAE.NOT(__), e) => begin
      :(-($(transformToMTKContinuousCondition(e, simCode; atInitial = atInitial))) ~ 0)
    end
    #= Strip noEvent wrapper and recurse =#
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        transformToMTKContinuousConditionEquation(innerArgs[1], simCode; atInitial = atInitial)
      else
        OMBackend.unsupported("noEvent with several arguments in a condition", cond)
      end
    end
    #= initial(): see transformToMTKContinuousCondition. =#
    DAE.CALL(Absyn.IDENT("initial"), _, _) => begin
      atInitial ? :(-1 ~ 0) : :(1 ~ 0)
    end
    #= General function call as boolean condition: same polarity rule as above. =#
    DAE.CALL(__) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)) ~ 0)
    end
    _ => begin
      OMBackend.unsupported("condition expression", cond)
    end
  end
  return res
end


#= The delay() calls of the model being generated (delays.jl), as (key, argument, delay time) with the
   argument and the delay time as Julia expressions; a call's position is the index of its history, and
   an equal call shares it. With DELAY_MODEL, the name the histories are kept under. Both reset at the
   start of each model's (or structural mode's) generation. =#
const DELAY_CALLS = Tuple{String, Any, Any}[]
const DELAY_MODEL = Ref{Symbol}(:none)

function _delayIndex!(key::String, x, delayTime)::Int
  local k = findfirst(c -> first(c) == key, DELAY_CALLS)
  k === nothing || return k
  push!(DELAY_CALLS, (key, x, delayTime))
  return length(DELAY_CALLS)
end

#= The argument of a when body the event iteration runs (codeGen.jl: the
   relation whens of `_emitRelationWhen`, and the discrete-when and
   elsewhen-arm affects): the state before the current sweep, indexed by
   `lookuptableStates`. =#
const PRE_SNAPSHOT = :__prevals
#= The homotopy parameter λ (homotopy(actual, simplified), MLS 3.7.4.4):
   1 in a simulation; the initialization of a model with homotopy() goes from
   0 (the simplified expressions) to 1 (`_homotopyContinuation`), as
   OpenModelica's default does. =#
const HOMOTOPY_LAMBDA = :_homotopyLambda
#= Whether homotopy() is lowered as the blend in λ: in the continuous
   equations and the initialization equations only (ODE_MODE_MTK_MODEL_GENERATION),
   where λ is a parameter and the continuation acts. Elsewhere (when bodies and
   conditions, relations, if-equation branches, bindings, observed equations)
   the actual expression: the blend read an undefined λ there (an
   UndefVarError at the first event), and λ is 1 after the initialization. =#
const HOMOTOPY_BLEND = Base.ScopedValues.ScopedValue(false)

#= Where `pre(v)` of a variable is read while a when body is lowered: false,
   from `v` itself; true (within `with(PRE_FROM_SNAPSHOT => true)`), from
   `PRE_SNAPSHOT`, so a body sees the pre() values even when an earlier body
   of the same sweep assigned the variable. =#
const PRE_FROM_SNAPSHOT = Base.ScopedValues.ScopedValue(false)

"""
  TODO: Keeping it simple for now, we assume we only have one argument in the call..
  Also the der as symbol is really ugly..
"""
function DAECallExpressionToMTKCallExpression(pathStr::String, expLst::List,
                                              simCode::SimulationCode.SimCode, ht; varPrefix=varPrefix, varSuffix = varSuffix, derAsSymbol=false)::Expr
  @match pathStr begin
    "der" => begin
      local arg = listHead(expLst)
      @match arg begin
        #= Scalarize der({c1, c2, ...}) into [der(c1), der(c2), ...].
           MSL MultiBody (Body.Q, frame_a.R_T rows etc.) emits a DAE.ARRAY
           of element CREFs that survives frontend scalarization. Without
           this arm DAE_identifierToString throws on the DAE.ARRAY. =#
        DAE.ARRAY(_, _, array) => begin
          local elemExprs = map(array) do e
            DAECallExpressionToMTKCallExpression("der", Cons(e, MetaModelica.nil), simCode, ht;
              varPrefix=varPrefix, varSuffix=varSuffix, derAsSymbol=derAsSymbol)
          end
          Expr(:vect, elemExprs...)
        end
        #= der(literal) ≡ 0. Reachable when an upstream parameter-eval pass
           (solveParametricInitialEquations, foldParameterClosure) substituted
           a parameter cref with its default constant before residual rewriting.
           See SimCodeCheck rule_no_literal_in_der_pre for early diagnostic. =#
        DAE.RCONST(_) => quote 0.0 end
        DAE.ICONST(_) => quote 0 end
        DAE.BCONST(_) => quote false end
        #= der(-x) = -der(x): the name was taken through the minus (x grew for
           der(-x) = 1). =#
        DAE.UNARY(DAE.UMINUS(__), inner) => begin
          local d = DAECallExpressionToMTKCallExpression("der", Cons(inner, MetaModelica.nil), simCode, ht;
                                                         varPrefix=varPrefix, varSuffix=varSuffix, derAsSymbol=derAsSymbol)
          :(-($(d)))
        end
        #= der of an expression: an initial equation der(w) = 0 whose w the
           backend replaced by its definition. Differentiated later (the init
           solve's derivative rows on observed expressions). =#
        _ where (!(arg isa DAE.CREF || arg isa DAE.UNARY) && !derAsSymbol) => begin
          local inner = expToJuliaExpMTK(arg, simCode; varPrefix = varPrefix, varSuffix = varSuffix)
          quote
            D($(inner))
          end
        end
        _ => begin
          varName = SimulationCode.DAE_identifierToString(arg)
          if derAsSymbol
            quote
              $(Symbol("der_$(varName)"))
            end
          else
            quote
              D($(Symbol(varName)))
            end
          end
        end
      end
    end
    "pre" => begin
      local arg = listHead(expLst)
      @match arg begin
        DAE.ARRAY(_, _, array) => begin
          local elemExprs = map(array) do e
            DAECallExpressionToMTKCallExpression("pre", Cons(e, MetaModelica.nil), simCode, ht;
              varPrefix=varPrefix, varSuffix=varSuffix, derAsSymbol=derAsSymbol)
          end
          Expr(:vect, elemExprs...)
        end
        #= pre(literal) ≡ literal — pre-of-constant is the constant itself. =#
        DAE.RCONST(r) => quote $r end
        DAE.ICONST(i) => quote $i end
        DAE.BCONST(b) => quote $b end
        _ => begin
          varName = SimulationCode.DAE_identifierToString(arg)
          local entry = get(simCode.stringToSimVarHT, varName, nothing)
          if PRE_FROM_SNAPSHOT[] && entry !== nothing && !SimulationCode.isParameter(last(entry))
            :($(PRE_SNAPSHOT)[lookuptableStates[$(QuoteNode(Symbol(varName)))]])
          else
            quote
              $(Symbol(varName))
            end
          end
        end
      end
    end
    #= delay(x, T[, Tmax]): x's value T ago, from the model's history of x (delays.jl). =#
    "delay" => begin
      local args = collect(expLst)
      local x = expToJuliaExpMTK(args[1], simCode; varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derAsSymbol)
      local T = expToJuliaExpMTK(args[2], simCode; varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derAsSymbol)
      local k = _delayIndex!(string(args[1]) * "|" * string(args[2]), x, T)
      :(OMBackend.CodeGeneration.delayLookup($(QuoteNode(DELAY_MODEL[])), $k, t, $T, $x))
    end
    "initial" => begin
      #= Modelica initial() is true only during initialization (handled separately by MTK).
         In continuous equations it is always false (0 in arithmetic boolean context). =#
      quote
        0
      end
    end
    #= Modelica Integer(enum) returns the 1-based index of an enum literal. Our
       codegen already lowers enum CREFs to integer indices and ENUM_LITERAL to
       its `index` field, so the cast is the identity at the Julia level. Without
       this arm the splice emits `Integer(::Num)` which has no method. =#
    #= homotopy(actual, simplified): a blend in the homotopy parameter λ (1
       in a simulation), for the initialization's continuation from the
       simplified expressions (MLS 3.7.4.4; OpenModelica's default for a
       model with homotopy). It was the actual expression only: the
       initialization found a root the simplified ones do not lead to (CubicRoot:
       x = -0.347, OpenModelica 1.879). =#
    "homotopy" => begin
      local lower = e -> expToJuliaExpMTK(e, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derAsSymbol)
      local a = lower(listHead(expLst))
      HOMOTOPY_BLEND[] ?
        :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.modelica_homotopy($(a), $(lower(listHead(listRest(expLst)))), $(HOMOTOPY_LAMBDA))) :
        a
    end
    "Integer" => begin
      expToJuliaExpMTK(listHead(expLst), simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derAsSymbol)
    end
    #= A String parameter's binding (createStringParameterAssignments): the
       argument types from the expressions, not from the values. =#
    "String" => AlgorithmicCodeGeneration.modelicaStringCall(collect(expLst), x -> expToJuliaExpMTK(x, simCode))
    _  =>  begin
      argPart = tuple(map((x) -> expToJuliaExpMTK(x, simCode), expLst)...)
      #= Check if this is a Modelica built-in with a dedicated Julia implementation =#
      builtinSym = get(AlgorithmicCodeGeneration.MODELICA_BUILTIN_FUNCTIONS, pathStr, nothing)
      if builtinSym !== nothing
        qualifiedName = Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(builtinSym))
        quote
          $(qualifiedName)($(argPart...))
        end
      else
        funcName = Symbol(pathStr)
        quote
          $(funcName)($(argPart...))
        end
      end
    end
  end
end

"""
  _ifexpBranchIsNonReal(e::DAE.Exp) -> Bool

Conservative type sniff used by the IFEXP lowering: returns `true` when the
expression's runtime value is clearly not a Real number, so the arithmetic
`cond*then + (1-cond)*else` encoding cannot apply.

Detects:
- `DAE.SCONST` literals — String values
- `DAE.CREF` whose `ty` is `DAE.T_STRING` — String parameters / variables
- `DAE.CALL` whose `attr.ty` is `DAE.T_STRING` — String-returning functions
- Nested `DAE.IFEXP` — recurse into both branches

Used to decide whether to emit `ifelse(cond, then, else)` (works for any type)
or the MTK-friendlier arithmetic form (Real only).
"""
Base.@nospecializeinfer function _ifexpBranchIsNonReal(@nospecialize(e::DAE.Exp))::Bool
  @match e begin
    DAE.SCONST(_) => true
    DAE.CREF(_, ty) => ty isa DAE.T_STRING
    DAE.CALL(_, _, attrs) => attrs.ty isa DAE.T_STRING
    DAE.IFEXP(_, t, el) => _ifexpBranchIsNonReal(t) || _ifexpBranchIsNonReal(el)
    _ => false
  end
end

#= SimCode-Exp entry: codegen consumes `SimulationCode.Exp`
   directly. The body below mirrors the `::DAE.Exp` version's dispatch
   shape but operates on SIM Exp variants:

   - Trivial leaves (ICONST / RCONST / BCONST / SCONST / ENUM_LITERAL)
     are emitted natively — no DAE round-trip.
   - Algebraic / logical / relational composites (BINARY / UNARY /
     LBINARY / LUNARY / RELATION / IFEXP / CAST / TUPLE) recurse into
     this same SIM-Exp entry. The `OpKind` enum maps to the existing
     `DAE_OP_toJuliaOperator` helper by reconstructing a minimal
     `DAE.Operator` value (the operator-side type is unused by the
     helper for the common cases).
   - Complex shapes that drive the bulk of the DAE.Exp version
     (EXP_CREF, CALL, ASUB, ARRAY_EXP, RECORD, TSUB) fall through to the
     `::DAE.Exp` emitter via `toDAEExp` for now. Migrating each is a
     mechanical mirror of the matching `DAE.X` arm in the long
     function below; do it variant-by-variant so each landing is
     small and testable. =#
"""
    expToJuliaExpMTK(exp::SimulationCode.Exp, simCode; varPrefix="", varSuffix="", derSymbol=false)::Expr

SIM-Exp method. Emits an MTK `Expr` for a `SimulationCode.Exp`: scalars and
operators directly, complex shapes (EXP_CREF/CALL/ASUB/ARRAY_EXP/RECORD/TSUB)
by delegating to the `::DAE.Exp` method below via `toDAEExp`. `varPrefix`/
`varSuffix` affix emitted cref names; `derSymbol` emits derivatives as a symbol.
"""
expToJuliaExpMTK(exp::SimulationCode.BCONST, simCode::SimulationCode.SIM_CODE;
                 varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr = quote $(exp.value) end
expToJuliaExpMTK(exp::SimulationCode.ICONST, simCode::SimulationCode.SIM_CODE;
                 varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr = quote $(exp.value) end
expToJuliaExpMTK(exp::SimulationCode.RCONST, simCode::SimulationCode.SIM_CODE;
                 varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr = quote $(exp.value) end
expToJuliaExpMTK(exp::SimulationCode.SCONST, simCode::SimulationCode.SIM_CODE;
                 varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr = quote $(exp.value) end
expToJuliaExpMTK(exp::SimulationCode.WILD, simCode::SimulationCode.SIM_CODE;
                 varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr = quote _ end

function expToJuliaExpMTK(exp::SimulationCode.ENUM_LITERAL, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  return quote $(exp.index) end
end

function expToJuliaExpMTK(exp::SimulationCode.EXP_CREF, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  #= SimCref is flat post-Causalize.flattenArrayCrefs, so the CREF_QUAL hierarchy
     that the DAE arm handles cannot reach here. Handle the cheap scalar / "time" /
     simple subscript cases natively; delegate T_ARRAY-typed and harder subscript
     cases to the DAE.Exp emitter. =#
  if exp.cref.sym === :time && exp.ty isa SimulationCode.TYPE_REAL
    return quote t end
  end
  if exp.ty isa SimulationCode.TYPE_ARRAY
    return expToJuliaExpMTK(SimulationCode.toDAEExp(exp), simCode;
                            varSuffix = varSuffix, varPrefix = varPrefix,
                            derSymbol = derSymbol)
  end
  local hashTable = simCode.stringToSimVarHT
  local nameStr = string(exp.cref.sym)
  local lookUpStr = isempty(exp.cref.subs) ?
    nameStr :
    string(nameStr, "[", join(exp.cref.subs, ","), "]")
  local htEntry = get(hashTable, lookUpStr, nothing)
  if htEntry !== nothing
    return quote $(Symbol(htEntry[2].name)) end
  end
  local (aliasResolved, aliasExpr) = resolveAliasedCref(lookUpStr, simCode, hashTable,
    varPrefix = varPrefix, varSuffix = varSuffix)
  if aliasResolved
    @warn "expToJuliaExpMTK[SIM]: resolved alias-eliminated cref via fallback" lookUpStr
    return aliasExpr
  end
  return quote $(Symbol(string(varPrefix, lookUpStr, varSuffix))) end
end

function expToJuliaExpMTK(exp::SimulationCode.BINARY, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local lhs = expToJuliaExpMTK(exp.exp1, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local rhs = expToJuliaExpMTK(exp.exp2, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local opSym = opKindToJuliaOperator(exp.op)
  return :( $opSym($(lhs), $(rhs)) )
end

function expToJuliaExpMTK(exp::SimulationCode.UNARY, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local operand = expToJuliaExpMTK(exp.exp, simCode;
                                    varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local opSym = opKindToJuliaOperator(exp.op)
  return :( $opSym($(operand)) )
end

function expToJuliaExpMTK(exp::SimulationCode.LUNARY, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local operand = expToJuliaExpMTK(exp.exp, simCode;
                                    varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  if exp.op === SimulationCode.OP_NOT
    return :( 1 - $(operand) )
  end
  local opSym = opKindToJuliaOperator(exp.op)
  return :( $opSym($(operand)) )
end

function expToJuliaExpMTK(exp::SimulationCode.LBINARY, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local lhs = expToJuliaExpMTK(exp.exp1, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local rhs = expToJuliaExpMTK(exp.exp2, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  if exp.op === SimulationCode.OP_OR
    return :( $(lhs) + $(rhs) - $(lhs) * $(rhs) )
  elseif exp.op === SimulationCode.OP_AND
    return :( $(lhs) * $(rhs) )
  end
  local opSym = opKindToJuliaOperator(exp.op)
  return :( $opSym($(lhs), $(rhs)) )
end

function expToJuliaExpMTK(exp::SimulationCode.RELATION, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local lhs = expToJuliaExpMTK(exp.exp1, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local rhs = expToJuliaExpMTK(exp.exp2, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local opSym = opKindToJuliaOperator(exp.op)
  return quote ($opSym($(lhs), $(rhs))) end
end

function expToJuliaExpMTK(exp::SimulationCode.IFEXP, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  if exp.cond isa SimulationCode.BCONST
    local branch = exp.cond.value ? exp.thenExp : exp.elseExp
    local e = expToJuliaExpMTK(branch, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
    return quote
      $(LineNumberNode(@__LINE__, "evaluated if expr"))
      $(e)
    end
  end
  local condJL = expToJuliaExpMTK(exp.cond, simCode;
                                   varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local thenJL = _guardNonIntegerPowerBasesForEagerBranch(
    expToJuliaExpMTK(exp.thenExp, simCode;
                     varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol))
  local elseJL = _guardNonIntegerPowerBasesForEagerBranch(
    expToJuliaExpMTK(exp.elseExp, simCode;
                     varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol))
  #= Same Real-vs-non-Real branch-typing rule as the DAE.Exp version
     (see comment above the matching `DAE.IFEXP` arm). =#
  if _ifexpBranchIsNonReal(SimulationCode.toDAEExp(exp.thenExp)) ||
     _ifexpBranchIsNonReal(SimulationCode.toDAEExp(exp.elseExp))
    return :(ifelse(Bool($(condJL)), $(thenJL), $(elseJL)))
  else
    return :( $(condJL) * $(thenJL) + (1 - $(condJL)) * $(elseJL) )
  end
end

function expToJuliaExpMTK(exp::SimulationCode.CAST, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  return quote
    $(generateCastExpressionMTK(SimulationCode.toDAEType(exp.ty), SimulationCode.toDAEExp(exp.exp), simCode, varPrefix))
  end
end

function expToJuliaExpMTK(exp::SimulationCode.TUPLE, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local elemExprs = Expr[expToJuliaExpMTK(e, simCode;
                                            varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
                          for e in exp.PR]
  return Expr(:tuple, elemExprs...)
end

function expToJuliaExpMTK(exp::SimulationCode.TSUB, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  #= TSUB over a CALL needs the Modelica-function dispatch in the DAE arm
     (tupleElementCall / tupleArrayElementCall); only the non-CALL form
     lowers cleanly to plain indexing. =#
  if exp.exp isa SimulationCode.CALL
    return expToJuliaExpMTK(SimulationCode.toDAEExp(exp), simCode;
                            varSuffix = varSuffix, varPrefix = varPrefix,
                            derSymbol = derSymbol)
  end
  local tupleExpr = expToJuliaExpMTK(exp.exp, simCode;
                                      varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  return :($tupleExpr[$(exp.index)])
end

function expToJuliaExpMTK(exp::SimulationCode.RSUB, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local innerJL = expToJuliaExpMTK(exp.exp, simCode;
                                    varPrefix = varPrefix, varSuffix = varSuffix,
                                    derSymbol = derSymbol)
  if exp.fieldName == "re"
    return :(OMBackend.CodeGeneration._recordFieldRe($innerJL))
  elseif exp.fieldName == "im"
    return :(OMBackend.CodeGeneration._recordFieldIm($innerJL))
  end
  local ix = AlgorithmicCodeGeneration._positionalFieldIndex(SimulationCode.toDAEExp(exp.exp), exp.index)
  return :(OMBackend.CodeGeneration._recordField($innerJL, $(QuoteNode(Symbol(exp.fieldName))), $(ix)))
end

function expToJuliaExpMTK(exp::SimulationCode.ARRAY_EXP, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  return handleArrayExp(SimulationCode.toDAEExp(exp), simCode)
end

function expToJuliaExpMTK(exp::SimulationCode.RECORD, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  #= Mirror the DAE.RECORD arms: Modelica `Complex(re, im)` becomes Julia's
     `Complex(re, im)`; any other record lowers to a `Symbolics.wrap`-wrapped
     NamedTuple keyed by field name. =#
  if exp.path isa Absyn.IDENT && exp.path.name == "Complex" && length(exp.exps) == 2
    local reExpr = expToJuliaExpMTK(exp.exps[1], simCode;
                                     varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
    local imExpr = expToJuliaExpMTK(exp.exps[2], simCode;
                                     varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
    return quote Complex($reExpr, $imExpr) end
  end
  local wrappedElems = Expr[:(Symbolics.wrap($(expToJuliaExpMTK(e, simCode;
                                                                  varPrefix = varPrefix,
                                                                  varSuffix = varSuffix,
                                                                  derSymbol = derSymbol))))
                             for e in exp.exps]
  if length(exp.fieldNames) == length(wrappedElems)
    local names = Symbol[Symbol(n) for n in exp.fieldNames]
    local pairs = Expr[Expr(:(=), names[i], wrappedElems[i]) for i in eachindex(names)]
    return Expr(:tuple, Expr(:parameters, pairs...))
  end
  return Expr(:tuple, wrappedElems...)
end

#= Fallback: CALL and any variant without a SIM-native arm route through the
   DAE.Exp emitter (~600 lines of HT lookup, alias resolution, builtin/external
   function dispatch, array-binding unfolding). Each can be migrated incrementally
   to its own SIM-native method above. =#
Base.@nospecializeinfer function expToJuliaExpMTK(@nospecialize(exp::SimulationCode.Exp),
                          simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "",
                          derSymbol::Bool = false)::Expr
  return expToJuliaExpMTK(SimulationCode.toDAEExp(exp), simCode;
                          varSuffix = varSuffix, varPrefix = varPrefix,
                          derSymbol = derSymbol)
end

"""
    expToJuliaExpMTK(exp::DAE.Exp, simCode; varPrefix="", varSuffix="", derSymbol=false)::Expr

DAE-Exp method (the bulk emitter). Converts a `DAE.Exp` into an MTK `Expr`. The
`SimulationCode.Exp` method above dispatches here via `toDAEExp` for shapes it
does not handle natively. `varPrefix`/`varSuffix` affix emitted cref names;
`derSymbol` emits derivatives as a symbol.
"""
function expToJuliaExpMTK(@nospecialize(exp::DAE.Exp),
                          simCode::SimulationCode.SIM_CODE;
                          varSuffix="",
                          varPrefix="",
                          derSymbol::Bool=false)::Expr
  hashTable = simCode.stringToSimVarHT
  local expr::Expr = begin
    local int::Int64
    local real::Float64
    local bool::Bool
    local tmpStr::String
    local cr::DAE.ComponentRef
    local e1::DAE.Exp
    local e2::DAE.Exp
    local e3::DAE.Exp
    local expl::List{DAE.Exp}
    local lstexpl::List{List{DAE.Exp}}
    @match exp begin
      DAE.BCONST(bool) => quote $bool end
      DAE.ICONST(int) => quote $int end
      DAE.RCONST(real) => quote $real end
      DAE.SCONST(tmpStr) => quote $tmpStr end
      DAE.CREF(DAE.CREF_IDENT("time", DAE.T_REAL(_)), _) => begin
        quote
          t
        end
      end
      #=
      Qualified path to a variable of type array.
      See array access below.
      Note that the array is added as <name>[<size>] in the HT during the simcode phase.
      Hence, the dimensionality must be added before lookup in the ht.
      =#
      #= A zero-size array has no variables (MSL Fluid's Xi_outflow[Medium.nXi],
         nXi = 0 for a single substance, passed to setState_phX): an empty array
         of its shape. =#
      DAE.CREF(cr, DAE.T_ARRAY(ty, dims)) where _isZeroSizeArray(dims) => _emptyArrayExpr(ty, dims)
      DAE.CREF(cr, DAE.T_ARRAY(ty, dims)) => begin
        lookUpStr = string(exp)
        arrName = string(exp)
        #= To make sure the variable is indexed =#
        for d in dims
          local i = @match d begin
            DAE.DIM_INTEGER(n) => n
            _ => OMBackend.unsupported("array dimension", d)
          end
          lookUpStr *= string("[", i, "]")
        end
        local arrEntry = get(hashTable, lookUpStr, nothing)
        if arrEntry !== nothing
          hashTable[arrName] = arrEntry
          expr = quote $(Symbol(arrName)) end
          expr
        else
          local (aliasResolved, aliasExpr) = resolveAliasedCref(lookUpStr, simCode, hashTable,
            varPrefix=varPrefix, varSuffix=varSuffix)
          if aliasResolved
            @warn "expToJuliaExpMTK: resolved alias-eliminated T_ARRAY variable via fallback" lookUpStr
            aliasExpr
          else
            #= Not a variable of its own (its elements were eliminated): the array
               of its elements, each lowered as a subscripted reference. The bare
               name was undefined (MSL Engine1b_analytic's frame_im.R.T passed to
               selectBranch in an initial equation). =#
            local sizes = Int[d.integer for d in dims]   #= DIM_INTEGER: checked above =#
            length(sizes) <= 2 || OMBackend.unsupported("an array of more than two dimensions without variables", arrName)
            local elem = idx -> expToJuliaExpMTK(
              DAE.CREF(DAE.CREF_IDENT(arrName, ty, MetaModelica.list((DAE.INDEX(DAE.ICONST(i)) for i in idx)...)), ty),
              simCode; varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
            length(sizes) == 1 ? Expr(:vect, (elem((i,)) for i in 1:sizes[1])...) :
              Expr(:vcat, (Expr(:row, (elem((i, j)) for j in 1:sizes[2])...) for i in 1:sizes[1])...)
          end
        end
      end
      #=
      This is an array access. Note the difference to the case above,
      that is a component of type array.
      In the case above we do not lookup the subscript whereas here it is subscripted.
      =#
      DAE.CREF(DAE.CREF_IDENT(ident, identType, subscriptLst), _) where !isempty(subscriptLst) => begin
        local varName = SimulationCode.string(ident)
        #= First try to handle as subscripted array with binding expression =#
        local cref = DAE.CREF_IDENT(ident, identType, subscriptLst)
        (success, arrayExpr) = tryHandleSubscriptedArrayCref(cref, hashTable, simCode,
          varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
        if success
          arrayExpr
        else
          #= Fallback: look up expanded variable name or generate runtime subscript =#
          local allConstant = true
          local lookUpStr = ""
          for s in subscriptLst
            @match s begin
              DAE.INDEX(DAE.ICONST(i)) => begin
                lookUpStr *= string("[", i, "]")
              end
              _ => begin
                allConstant = false
              end
            end
          end
          if allConstant
            local fullKey = string(varName, lookUpStr)
            local htEntry = get(hashTable, fullKey, nothing)
            if htEntry !== nothing
              quote
                $(LineNumberNode(@__LINE__, "$varName array"))
                $(Symbol(htEntry[2].name))
              end
            else
              #= Variable was eliminated by alias elimination. Resolve via aliasMap. =#
              local (aliasResolved, aliasExpr) = resolveAliasedCref(fullKey, simCode, hashTable,
                varPrefix=varPrefix, varSuffix=varSuffix)
              if aliasResolved
                @warn "expToJuliaExpMTK: resolved alias-eliminated variable via fallback" fullKey
                aliasExpr
              else
                #= Variable not in hash table (may have been eliminated), using direct reference =#
                quote $(Symbol(string(varPrefix, fullKey, varSuffix))) end
              end
            end
          else
            #= Variable subscripts: generate runtime indexing =#
            local subExprs = map(subscriptLst) do sub
              @match sub begin
                DAE.INDEX(idxExp) => expToJuliaExpMTK(idxExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
                DAE.SLICE(idxExp) => expToJuliaExpMTK(idxExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
                DAE.WHOLEDIM(__) => :(:)
                _ => OMBackend.unsupported("subscript", sub)
              end
            end
            local baseSymbol = Symbol(varPrefix, varName, varSuffix)
            Expr(:ref, baseSymbol, subExprs...)
          end
        end
      end
      #=
      Note in some cases we still retain information that something is a part of a complex component.
      In this case we reference a component that in turn is a part of a record.
      =#
      DAE.CREF(DAE.CREF_QUAL(componentRef = componentRef,
                             ident = ident,
                             subscriptLst = subscriptLst,
                             identType = identType), ty) where {
                                 FrontendUtil.Util.finalCrefIsArray(componentRef)
                             } =>
      begin
        local cr = DAE.CREF_QUAL(ident, identType, subscriptLst, componentRef)
        varName = SimulationCode.DAE_identifierToString(cr)
        #= Workaround =#
        local fcr = FrontendUtil.Util.getFinalCref(componentRef)
        local fcrs = FrontendUtil.Util.getAllCrefsAsVector(cr)
        local subscripts = fcr.subscriptLst
        @assign fcr.subscriptLst = MetaModelica.nil
        local lookupStrPrefix = reduce((x,y) -> string(x, COMPONENT_SEPARATOR, y), map(string, fcrs[1:end-1]))
        local lookupStr = string(lookupStrPrefix, COMPONENT_SEPARATOR, SimulationCode.DAE_identifierToString(fcr))

        local lookupEntry = get(hashTable, lookupStr, nothing)
        if lookupEntry === nothing
          #= Base name not found. The backend scalarizes arrays into individual elements,
             so try looking up the element name with subscripts (e.g., "var[1]"). =#
          local allConstSubs = true
          local subSuffix = ""
          for sub in subscripts
            @match sub begin
              DAE.INDEX(DAE.ICONST(i)) => begin
                subSuffix = string(subSuffix, "[", i, "]")
              end
              _ => begin
                allConstSubs = false
                break
              end
            end
          end
          local elemKey = string(lookupStr, subSuffix)
          local elemLookup = allConstSubs ? get(hashTable, elemKey, nothing) : nothing
          if elemLookup !== nothing
            local elemName = elemLookup[2].name
            quote $(Symbol(string(varPrefix, elemName, varSuffix))) end
          else
            #= Check if alias-eliminated =#
            local (aliasRes2, aliasEx2) = resolveAliasedCref(elemKey, simCode, hashTable,
              varPrefix=varPrefix, varSuffix=varSuffix)
            if aliasRes2
              @warn "expToJuliaExpMTK: resolved alias-eliminated CREF_QUAL element via fallback" elemKey
              aliasEx2
            else
              local ss = subscriptsToExpr(subscripts, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
              local refExpr = makeRefExpr(Symbol(string(varPrefix, lookupStr, varSuffix)), ss)
              quote
                $(LineNumberNode(@__LINE__, "Array access to missing var: $lookupStr"))
                $(refExpr)
              end
            end
          end
        else
          local indexAndVar = lookupEntry
          local varKind = indexAndVar[2].varKind
          @match varKind begin
            (SimulationCode.ARRAY(_, SOME(bindRaw && SimulationCode.ARRAY_EXP(__))) ||
             SimulationCode.ARRAY_PARAMETER(_, SOME(bindRaw && SimulationCode.ARRAY_EXP(__)))) => begin
              local bindArray = SimulationCode.toDAEExp(bindRaw)
              local subIndices = Int[]
              local allConstant = true
              for sub in subscripts
                @match sub begin
                  DAE.INDEX(DAE.ICONST(i)) => push!(subIndices, i)
                  _ => begin allConstant = false; break end
                end
              end
              if allConstant && !isempty(subIndices)
                #= Evaluate binding expression at compile time =#
                local element = if length(subIndices) == 1
                  listGet(bindArray.array, first(subIndices))
                else
                  local current = bindArray
                  for idx in subIndices
                    current isa DAE.ARRAY || OMBackend.unsupported("a binding row that is not an array literal", current)
                    current = listGet(current.array, idx)
                  end
                  current
                end
                return expToJuliaExpMTK(element, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
              end
            end
            _ => ()
          end
          #= Fallback: generate array indexing =#
          local vRef = string(varPrefix, indexAndVar[2].name, varSuffix)
          local ss = subscriptsToExpr(subscripts, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
          local refExpr = makeRefExpr(Symbol(vRef), ss)
          quote
            $(LineNumberNode(@__LINE__, "Array access to: $vRef"))
            $(refExpr)
          end
        end
      end

      DAE.CREF(cr, _)  => begin
        varName = SimulationCode.DAE_identifierToString(cr)
        if !haskey(hashTable, varName)
          #= Try to handle as subscripted array access (e.g., R_w[1] where R_w is an ARRAY) =#
          (success, arrayExpr) = tryHandleSubscriptedArrayCref(cr, hashTable, simCode,
            varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
          if success
            arrayExpr
          else
            #= Check if alias-eliminated =#
            local (aliasRes, aliasEx) = resolveAliasedCref(varName, simCode, hashTable,
              varPrefix=varPrefix, varSuffix=varSuffix)
            if aliasRes
              @warn "expToJuliaExpMTK: resolved alias-eliminated bare CREF via fallback" varName
              aliasEx
            else
              #= Variable not in hash table, using direct reference =#
              quote $(Symbol(string(varPrefix, varName, varSuffix))) end
            end
          end
        else
          indexAndVar = hashTable[varName]
          varKind::SimulationCode.SimVarType = indexAndVar[2].varKind
          @match varKind begin
            SimulationCode.INPUT(__) => OMBackend.unsupported("INPUT variable", varName)
            SimulationCode.STATE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName state"))
              $(Symbol(string(varPrefix, indexAndVar[2].name, varSuffix)))
            end
            SimulationCode.PARAMETER(__) => quote
              $(LineNumberNode(@__LINE__, "$varName parameter"))
              $(Symbol(string(varPrefix, indexAndVar[2].name, varSuffix)))
            end
            SimulationCode.ALG_VARIABLE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, algebraic"))
              $(Symbol(string(varPrefix, indexAndVar[2].name, varSuffix)))
            end
            SimulationCode.DISCRETE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, discrete"))
              $(Symbol(string(varPrefix, indexAndVar[2].name, varSuffix)))
            end
            SimulationCode.DATA_STRUCTURE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, datastructure variable"))
              $(Symbol(string(varPrefix, indexAndVar[2].name, varSuffix)))
            end
            SimulationCode.STRING(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, datastructure variable"))
              $(Symbol(string(varPrefix, indexAndVar[2].name, varSuffix)))
            end
            _ => OMBackend.unsupported("variable kind $(nameof(typeof(varKind)))", varName)
          end
        end
      end
      DAE.UNARY(operator = op, exp = e1) => begin
        o = DAE_OP_toJuliaOperator(op)
        quote
          $(o)($(expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, varSuffix = varSuffix)))
        end
      end
      DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
        local rhs = expToJuliaExpMTK(e2, simCode, varPrefix=varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
        local opSym = DAE_OP_toJuliaOperator(op)
        #= Matrix multiplication: operands are always proper Matrix (from hvcat in
           equation codegen) or symbolic Num (where ensureMatrix is a no-op).
           Function impl params are pre-converted by generateArrayConversions. =#
        :($opSym($(lhs), $(rhs)))
      end
      DAE.LUNARY(operator = op, exp = e1)  => begin
        local operand = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
        @match op begin
          DAE.NOT(__) => :(1 - $(operand))
          _ => begin
            local opSym = DAE_OP_toJuliaOperator(op)
            :($opSym($(operand)))
          end
        end
      end
      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
        local rhs = expToJuliaExpMTK(e2, simCode, varPrefix=varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
        #= Use arithmetic for boolean ops: AND = a*b, OR = a+b-a*b, on 0/1 values.
           Neither short-circuit (&&/||) nor bitwise (|/&) work reliably with Symbolics.jl
           due to type mismatches between Bool, Num, and BasicSymbolic{Real}. =#
        @match op begin
          #= Bind both operands once: `a + b - a*b` splices `lhs`/`rhs` twice, so a
             left-nested OR chain duplicates the accumulator at every step and the
             generated Expr grows exponentially (Digital RAM/table when-conditions). =#
          DAE.OR(__) => :(let _a = $(lhs), _b = $(rhs); _a + _b - _a * _b end)
          DAE.AND(__) => :($(lhs) * $(rhs))
          _ => begin
            local opSym = DAE_OP_toJuliaOperator(op)
            :($opSym($(lhs), $(rhs)))
          end
        end
      end
      DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, varSuffix = varSuffix,derSymbol = derSymbol)
        local rhs = expToJuliaExpMTK(e2, simCode,varPrefix=varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
        local op = DAE_OP_toJuliaOperator(op)
        quote
          ($op($(lhs), $(rhs)))
        end
      end
      DAE.IFEXP(DAE.BCONST(false), e2, e3) => begin
        local e = expToJuliaExpMTK(e3, simCode)
        quote
          $(LineNumberNode(@__LINE__, "evaluated if expr: $(string(exp))"))
          $(e)
        end
      end
      DAE.IFEXP(DAE.BCONST(true), e2, e3) => begin
        local e = expToJuliaExpMTK(e2, simCode)
        quote
          $(LineNumberNode(@__LINE__, "evaluated if expr: $(string(exp))"))
          $(e)
        end
      end
      #=
      In the other case, see if the condition can be evaluated into a constant.
      If that is the case the expression can be resolved.
      =#
      DAE.IFEXP(expCond, expThen, expElse) => begin
        local condJL = expToJuliaExpMTK(expCond, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
        local thenJL = _guardNonIntegerPowerBasesForEagerBranch(
          expToJuliaExpMTK(expThen, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol))
        local elseJL = _guardNonIntegerPowerBasesForEagerBranch(
          expToJuliaExpMTK(expElse, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol))
        #= For Real branches use arithmetic ifelse: cond*then + (1-cond)*else.
           This avoids type dispatch issues with ModelingToolkit.ifelse on
           BasicSymbolic{Real} vs Num. For String / Boolean / Integer branches
           the arithmetic encoding is invalid (you cannot multiply a String by
           a number), so fall back to Julia's `ifelse(cond, then, else)`.
           Surfaces on every CombiTable / CombiTimeTable model where the
           constructor's `fileName` argument is wrapped in
           `if useFile and ... then fileName else "NoName"`. =#
        if _ifexpBranchIsNonReal(expThen) || _ifexpBranchIsNonReal(expElse)
          :(ifelse(Bool($(condJL)), $(thenJL), $(elseJL)))
        else
          :($(condJL) * $(thenJL) + (1 - $(condJL)) * $(elseJL))
        end
      end
      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = explst)  => begin
        #= Route through _modelicaFunctionCallExpr (which consults the runtime
           registry) when the name is either a simcode function OR a registered
           OMRuntimeExternalC runtime function (e.g. the impure-RNG family). The
           registry check matters for callers that run before simCode.functions
           is populated (solveParametricInitialEquations!), where the bare-symbol
           fallback would emit an unbound name that fails at eval. =#
        if _isSimCodeFunctionName(tmpStr, simCode) ||
           haskey(AlgorithmicCodeGeneration.MODELICA_UTILITIES_TO_RUNTIME_C, OMBackend.canonicalName(tmpStr))
          _modelicaFunctionCallExpr(tmpStr, explst, simCode, hashTable;
                                    varPrefix = varPrefix,
                                    varSuffix = varSuffix,
                                    derSymbol = derSymbol)
        else
          #Call as symbol is really ugly.. please fix me :(
          DAECallExpressionToMTKCallExpression(tmpStr, explst, simCode, hashTable; varPrefix=varPrefix, varSuffix = varSuffix, derAsSymbol=derSymbol)
        end
      end
      DAE.CALL(path, expLst) => begin
        _modelicaFunctionCallExpr(path, expLst, simCode, hashTable;
                                  varPrefix = varPrefix,
                                  varSuffix = varSuffix,
                                  derSymbol = derSymbol)
      end
      DAE.CAST(ty, exp)  => begin
        quote
          $(generateCastExpressionMTK(ty, exp, simCode, varPrefix))
        end
      end
      #= For enumeration we just take the value of the index. =#
      DAE.ENUM_LITERAL(path, index) => begin
        quote
          $(LineNumberNode(@__LINE__, "$(string(path)) ENUM"))
          $(index)
        end
      end
      DAE.ARRAY(DAE.T_ARRAY(DAE.T_REAL(MetaModelica.Nil(__)), dims), scalar, arr) => begin
        handleArrayExp(exp, simCode)
      end
      DAE.ARRAY(DAE.T_ARRAY(DAE.T_INTEGER(MetaModelica.Nil(__)), dims), scalar, arr) => begin
        handleArrayExp(exp, simCode)
      end
      DAE.ARRAY(DAE.T_ARRAY(_, _), _, _) => begin
        handleArrayExp(exp, simCode)
      end
      #= Handle array subscripting: expr[subscripts] =#
      DAE.ASUB(innerExp, subscripts) => begin
        #= Convert subscripts to Julia indices =#
        local subExprs = map(subscripts) do sub
          @match sub begin
            DAE.ICONST(i) => i
            DAE.INDEX(DAE.ICONST(i)) => i
            #= Non-constant subscript: lower its inner index expression. =#
            DAE.INDEX(e) => expToJuliaExpMTK(e, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
            DAE.SLICE(e) => expToJuliaExpMTK(e, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
            DAE.WHOLE_NONEXP(e) => expToJuliaExpMTK(e, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
            _ => expToJuliaExpMTK(sub, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
          end
        end
        local allConstSubs = all(s -> s isa Integer, subExprs)
        #= When the inner expression is a bare CREF (no subscripts on the CREF itself)
           and all ASUB subscripts are constant, try scalarized variable lookup.
           This handles record field array equations like frame_b.R.T[1,1] = frame_a.R.T[1,1]
           where the frontend flattened to ASUB(CREF("R_T"), [1,1]) instead of CREF("R_T", subs=[1,1]). =#
        if allConstSubs
          #= First, detect nested ASUB(ASUB(CALL(qualified_path, args), [tupleIx]), [subExprs...])
             where the inner ASUB extracts a tuple element that is an array.
             This covers BOTH 1D access [i] and multi-D access [i, j, ...].
             Plain indexing on tupleElementCall fails because it returns a scalar Num. =#
          local _nestedTupleArrResult = @match innerExp begin
            DAE.ASUB(DAE.CALL(path, expLst), innerSubs) where {_isSimCodeFunctionPath(path, simCode) && length(innerSubs) == 1} => begin
              local innerSubExpr = first(innerSubs)
              local tupleIx = @match innerSubExpr begin
                DAE.ICONST(i) => i
                DAE.INDEX(DAE.ICONST(i)) => i
                _ => nothing
              end
              if tupleIx === nothing
                nothing
              else
                local callFuncName2 = Symbol(OMBackend.canonicalName(string(path)))
                local fnQuote2 = QuoteNode(callFuncName2)
                local callArgs2 = _modelicaFunctionCallArgs(expLst, simCode, hashTable;
                                                            varPrefix = varPrefix,
                                                            varSuffix = varSuffix,
                                                            derSymbol = derSymbol)
                local arrIdxTuple = Tuple(Int[Int(s) for s in subExprs])
                :(OMBackend.CodeGeneration.tupleArrayElementAt($fnQuote2, $tupleIx, $arrIdxTuple, $(callArgs2...)))
              end
            end
            #= Same pattern but with TSUB instead of inner ASUB for tuple extraction.
               Handles ASUB(TSUB(CALL(func, args), tupleIx), [arraySubscripts]). =#
            DAE.TSUB(DAE.CALL(path, expLst), tupleIx, _) where {_isSimCodeFunctionPath(path, simCode)} => begin
              local callFuncName3 = Symbol(OMBackend.canonicalName(string(path)))
              local fnQuote3 = QuoteNode(callFuncName3)
              local callArgs3 = _modelicaFunctionCallArgs(expLst, simCode, hashTable;
                                                          varPrefix = varPrefix,
                                                          varSuffix = varSuffix,
                                                          derSymbol = derSymbol)
              local arrIdxTuple3 = Tuple(Int[Int(s) for s in subExprs])
              :(OMBackend.CodeGeneration.tupleArrayElementAt($fnQuote3, $tupleIx, $arrIdxTuple3, $(callArgs3...)))
            end
            _ => nothing
          end
          local scalarizedResult = @match innerExp begin
            DAE.CREF(DAE.CREF_IDENT(ident, _, sLst), _) where {isempty(sLst)} => begin
              local lookUpStr = string(ident) * join(("[" * string(s) * "]" for s in subExprs))
              local entry = get(hashTable, lookUpStr, nothing)
              if entry !== nothing
                quote $(Symbol(string(varPrefix, entry[2].name, varSuffix))) end
              else
                nothing
              end
            end
            _ => nothing
          end
          if _nestedTupleArrResult !== nothing
            _nestedTupleArrResult
          elseif scalarizedResult !== nothing
            scalarizedResult
          elseif length(subExprs) == 1 && first(subExprs) isa Integer
            #= Check for multi-output Modelica function call: ASUB(CALL(qualified_path, args), [ix]).
               Array-returning functions are not multi-output tuple calls; they
               must use normal indexing on the returned array. This matters for
               Modelica.Math.Random.Generators.Xorshift128plus.initialState,
               whose return type is Integer[4]. =#
            local _asubCallResult = @match innerExp begin
              DAE.CALL(path, expLst, DAE.CALL_ATTR(ty=DAE.T_ARRAY(__))) where {_isSimCodeFunctionPath(path, simCode)} => begin
                nothing
              end
              DAE.CALL(path, expLst) where {_isSimCodeFunctionPath(path, simCode)} => begin
                local callFuncName = Symbol(OMBackend.canonicalName(string(path)))
                local fnQuote = QuoteNode(callFuncName)
                local callArgs = _modelicaFunctionCallArgs(expLst, simCode, hashTable;
                                                           varPrefix = varPrefix,
                                                           varSuffix = varSuffix,
                                                           derSymbol = derSymbol)
                local ix = first(subExprs)
                :(OMBackend.CodeGeneration.tupleElementCall($fnQuote, $ix, $(callArgs...)))
              end
              _ => nothing
            end
            if _asubCallResult !== nothing
              _asubCallResult
            else
              local innerCode = expToJuliaExpMTK(innerExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
              quote $(innerCode)[$(first(subExprs))] end
            end
          else
            local innerCode = expToJuliaExpMTK(innerExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
            if length(subExprs) == 1
              quote $(innerCode)[$(first(subExprs))] end
            else
              quote $(innerCode)[$(subExprs...)] end
            end
          end
        else
          #= Symbolic indexing into a literal constant array: MTK rejects
             `arr[Num, Num]` because Num isn't a valid array index. Emit a
             call to OMBackend.CodeGeneration.constTableLookup, which is
             Symbolic-aware: it returns the literal element for numeric args
             and an opaque Symbolics Term for symbolic args, so MTK
             structural-simplify treats the whole lookup as a black box. =#
          local innerCode = expToJuliaExpMTK(innerExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
          local _isArrayLiteral = (innerExp isa DAE.ARRAY)
          #= Detect a CREF into a constant `DATA_STRUCTURE` table (e.g.
             `Modelica.Electrical.Digital`'s `Buf3sTable[strength,
             UX01Conv[enable], UX01Conv[NotTable[x]]]`). Those are flattened
             as module-level `Matrix{Int}` constants; with Symbolic Num
             indices the plain `arr[i,j]` form throws
             `ArgumentError: invalid index ... of type SymbolicUtils.BasicSymbolicImpl`.
             Routing through `constTableLookup` works for both numeric and
             symbolic indices. =#
          local _isConstTableCref = (innerExp isa DAE.CREF) && let
            local _crefName = SimulationCode.DAE_identifierToString(innerExp.componentRef)
            haskey(hashTable, _crefName) &&
              hashTable[_crefName][2].varKind isa SimulationCode.DATA_STRUCTURE
          end
          if _isArrayLiteral || _isConstTableCref
            quote
              OMBackend.CodeGeneration.constTableLookup($(innerCode), $(subExprs...))
            end
          elseif length(subExprs) == 1
            #= Function call results are proper Matrix (impl bodies use ensureArray
               for array construction, generateArrayConversions for params).
               Symbolic Num handles subscripting directly. =#
            quote
              $(innerCode)[$(first(subExprs))]
            end
          else
            #= Multi-dim non-literal non-const-table-CREF with symbolic indices.
               This shape is rare and most likely a bug upstream — we have no
               way to dispatch to a Symbolic-aware Matrix indexer without a
               concrete table to look at. Keep the plain form so the runtime
               error points at the offending expression. =#
            quote
              $(innerCode)[$(subExprs...)]
            end
          end
        end
      end
      DAE.REDUCTION(reductionInfo, bodyExp, iterators) => begin
        #= Handle array comprehensions and reductions =#
        local bodyExpr = expToJuliaExpMTK(bodyExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
        #= Build iterator expressions =#
        local iterExprs = Expr[]
        for iter in iterators
          @match iter begin
            DAE.REDUCTIONITER(id, rangeExp, guardExp, _) => begin
              guardExp === nothing || OMBackend.unsupported("a reduction iterator with a guard", exp)
              local rangeExpr = expToJuliaExpMTK(rangeExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
              push!(iterExprs, Expr(:(=), Symbol(id), rangeExpr))
            end
          end
        end
        #= Handle different reduction types =#
        @match reductionInfo.path begin
          Absyn.IDENT("array") => begin
            #= {e for i in u, j in v}: the last iterator is the first dimension
               (MLS 10.4.1.2); a Julia comprehension's first iterator is. =#
            Expr(:comprehension, bodyExpr, reverse(iterExprs)...)
          end
          Absyn.IDENT("sum") => begin
            #= Sum reduction: sum(expr for i in range) =#
            local genExpr = Expr(:generator, bodyExpr, iterExprs...)
            :(sum($genExpr))
          end
          Absyn.IDENT("product") => begin
            #= Product reduction =#
            local genExpr = Expr(:generator, bodyExpr, iterExprs...)
            :(prod($genExpr))
          end
          Absyn.IDENT("min") => begin
            local genExpr = Expr(:generator, bodyExpr, iterExprs...)
            :(minimum($genExpr))
          end
          Absyn.IDENT("max") => begin
            local genExpr = Expr(:generator, bodyExpr, iterExprs...)
            :(maximum($genExpr))
          end
          _ => OMBackend.unsupported("the reduction $(reductionInfo.path)", exp)
        end
      end
      DAE.RANGE(_, startExp, NONE(), stopExp) => begin
        #= Range expression: start:stop =#
        local startExpr = expToJuliaExpMTK(startExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
        local stopExpr = expToJuliaExpMTK(stopExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
        :($startExpr:$stopExpr)
      end
      DAE.RANGE(_, startExp, SOME(stepExp), stopExp) => begin
        #= Range expression: start:step:stop =#
        local startExpr = expToJuliaExpMTK(startExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
        local stepExpr = expToJuliaExpMTK(stepExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
        local stopExpr = expToJuliaExpMTK(stopExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
        :($startExpr:$stepExpr:$stopExpr)
      end
      DAE.TSUB(tupleExp, ix, tsubTy) => begin
        #= Tuple subscript: extract element ix from a tuple-returning expression.
           For function calls, use tupleElementCall for scalar elements or
           tupleArrayElementCall for array elements. Direct indexing
           (expr[ix]) fails because Num(scalar_term)[1] is a no-op in Symbolics. =#
        if tupleExp isa DAE.CALL
          local callFuncName = Symbol(OMBackend.canonicalName(string(tupleExp.path)))
          local fnQuote = QuoteNode(callFuncName)
          local args = _modelicaFunctionCallArgs(tupleExp.expLst, simCode, hashTable;
                                                 varPrefix = varPrefix,
                                                 varSuffix = varSuffix,
                                                 derSymbol = derSymbol)
          #= Check if the tuple element type is an array with known dimensions =#
          local tsubArrayDims = nothing
          if tsubTy isa DAE.T_ARRAY
            local intDims = Int[]
            local allKnown = true
            for d in tsubTy.dims
              if d isa DAE.DIM_INTEGER
                push!(intDims, d.integer)
              else
                allKnown = false
                break
              end
            end
            if allKnown && !isempty(intDims)
              tsubArrayDims = Tuple(intDims)
            end
          end
          if tsubArrayDims !== nothing
            :(OMBackend.CodeGeneration.tupleArrayElementCall($fnQuote, $ix, $tsubArrayDims, $(args...)))
          else
            :(OMBackend.CodeGeneration.tupleElementCall($fnQuote, $ix, $(args...)))
          end
        else
          local tupleExpr = expToJuliaExpMTK(tupleExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
          :($tupleExpr[$ix])
        end
      end
      #=
      Record constructor special case: Modelica `Complex(re, im)`.

      The Modelica `Complex` operator record is inlined from
      `Modelica.ComplexMath.j` (the imaginary unit) and similar constants
      into backend IR as DAE.RECORD(IDENT("Complex"), [re, im], ...).
      Lower it to Julia's built-in `Complex(re, im)` so downstream
      arithmetic works with the standard Julia/Symbolics complex support.

      Non-Complex record constructors deliberately fall through to the
      generic `_ => throw(...)` below. A permissive tuple fallback has
      been tried and regressed models whose call-argument handling
      expects a CREF-stringifiable expression (FilterWithDifferentiation
      hit `DAE_identifierToString: DAE.ARRAY` via
      DAECallExpressionToMTKCallExpression once a non-Complex RECORD got
      lowered to a tuple). If we encounter a non-Complex record
      constructor we want the compile-time error, not a silent
      type-mismatch downstream.
      =#
      DAE.RECORD(Absyn.IDENT("Complex"), expl, _, _) where length(expl) == 2 => begin
        local reExpr = expToJuliaExpMTK(listGet(expl, 1), simCode,
                                        varPrefix=varPrefix,
                                        varSuffix=varSuffix,
                                        derSymbol=derSymbol)
        local imExpr = expToJuliaExpMTK(listGet(expl, 2), simCode,
                                        varPrefix=varPrefix,
                                        varSuffix=varSuffix,
                                        derSymbol=derSymbol)
        quote Complex($reExpr, $imExpr) end
      end
      #=
      Generic record constructor fallback. Lowers any DAE.RECORD to a
      Julia `NamedTuple` keyed by the Modelica field names. Used by
      Spice3.Internal.Mosfet.Mosfet and Media.IdealGases.DataRecord
      parameter records.

      Historically a plain tuple fallback regressed FilterWithDifferentiation
      because downstream `DAECallExpressionToMTKCallExpression` der/pre arms
      did not handle DAE.ARRAY-of-CREFs — those arms now scalarize, so this
      permissive arm is safe again.
      =#
      DAE.RECORD(_, expl, fieldNames, _) => begin
        local elemExprs = [expToJuliaExpMTK(e, simCode;
                                             varPrefix=varPrefix,
                                             varSuffix=varSuffix,
                                             derSymbol=derSymbol)
                           for e in expl]
        local names = [Symbol(n) for n in fieldNames]
        #= Wrap each field with `Symbolics.wrap` so the resulting NamedTuple
           field type is `Num` (or stays `Number` for plain literals).
           Without the wrap, fields can hold a bare `BasicSymbolic{Real}`
           which is rejected by `SymbolicUtils._numeric_or_arrnumeric_symtype`
           and shows up as `MethodError: -(::SymReal, ::SymReal)` when MTK
           applies arithmetic to a Modelica function-call argument that
           was passed a parameter record (Spice3.Internal.Mosfet.Mosfet,
           Media.IdealGases.DataRecord). Verified harmless on
           Magnetic.FundamentalWave — those use the `Complex(re, im)`
           special-case constructor handled at the previous arm, not this
           generic record fallback. =#
        local wrappedElems = [:(Symbolics.wrap($(e))) for e in elemExprs]
        if length(names) == length(wrappedElems)
          local pairs = [Expr(:(=), names[i], wrappedElems[i]) for i in eachindex(names)]
          Expr(:tuple, Expr(:parameters, pairs...))
        else
          #= Safety: fall back to positional tuple if field-name list is mismatched. =#
          Expr(:tuple, wrappedElems...)
        end
      end
      #=
        Record-field subscript: extract one named field from a record-valued
        sub-expression. Surfaces on Magnetic.FundamentalWave / QuasiStatic
        models where Modelica `Complex.*.multiply(c1,c2).re` reaches the
        backend with an unevaluated outer `*` (the OMC inliner left it
        intact). We lower this to `getproperty(inner, :fieldName)`, which
        works for:
          • Julia `Base.Complex` (has `re`/`im` fields directly)
          • NamedTuple lowering of MOS / Medium parameter records (line 1480)
          • Any user-defined Julia struct with the named field
        For the canonical `re` / `im` fields we additionally hand off to
        `real` / `imag` when the inner is a Symbolics `Num` so the
        symbolic engine sees the structural complex projection rather
        than a plain `getproperty` call. =#
      DAE.RSUB(exp = innerExp, ix = ix, fieldName = fname) => begin
        local innerJL = expToJuliaExpMTK(innerExp, simCode;
                                          varPrefix=varPrefix,
                                          varSuffix=varSuffix,
                                          derSymbol=derSymbol)
        if fname == "re"
          :(OMBackend.CodeGeneration._recordFieldRe($innerJL))
        elseif fname == "im"
          :(OMBackend.CodeGeneration._recordFieldIm($innerJL))
        else
          :(OMBackend.CodeGeneration._recordField($innerJL, $(QuoteNode(Symbol(fname))),
                                                  $(AlgorithmicCodeGeneration._positionalFieldIndex(innerExp, ix))))
        end
      end
    _ => OMBackend.unsupported("expression", exp)
    end
  end
  return expr
end

_isZeroSizeArray(dims)::Bool = any(d -> d isa DAE.DIM_INTEGER && d.integer == 0, dims)

function _emptyArrayExpr(ty::DAE.Type, dims)::Expr
  local elType = ty isa DAE.T_INTEGER ? :Int : ty isa DAE.T_BOOL ? :Bool : :Float64
  local sizes = [d isa DAE.DIM_INTEGER ? Int(d.integer) : 0 for d in dims]
  return :(zeros($elType, $(sizes...)))
end

"""
  Try to handle a CREF that references a subscripted array parameter.
  Returns (success::Bool, expr::Expr).
  If the base array has a binding expression and subscripts are constant,
  evaluates at compile time. Otherwise generates symbol reference.
"""
function tryHandleSubscriptedArrayCref(cr::DAE.ComponentRef, hashTable, simCode;
                                        varPrefix="", varSuffix="", derSymbol=false)
  local subscripts = FrontendUtil.Util.getSubscriptsFromCref(cr)
  local baseName = FrontendUtil.Util.getBaseNameWithoutSubscripts(cr)

  if isempty(subscripts)
    return (false, :())
  end
  local baseVar = get(hashTable, baseName, nothing)
  if baseVar === nothing
    return (false, :())
  end
  if !(baseVar[2].varKind isa SimulationCode.ARRAY || baseVar[2].varKind isa SimulationCode.ARRAY_PARAMETER)
    return (false, :())
  end

  local arrayKind = baseVar[2].varKind
  local subExprs = map(subscripts) do sub
    @match sub begin
      DAE.INDEX(DAE.ICONST(i)) => i
      DAE.ICONST(i) => i
      #= A DAE.Subscript is lowered through its expression (it was passed whole: a MethodError). =#
      DAE.INDEX(idxExp) || DAE.SLICE(idxExp) => expToJuliaExpMTK(idxExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
      DAE.WHOLEDIM(__) => :(:)
      _ => sub isa DAE.Exp ? expToJuliaExpMTK(sub, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol) :
                             OMBackend.unsupported("subscript", sub)
    end
  end

  #= Check if we have a binding expression and all subscripts are constant integers =#
  local allConstantSubscripts = all(s -> s isa Integer, subExprs)

  if allConstantSubscripts
    @match arrayKind begin
      (SimulationCode.ARRAY(_, SOME(bindRaw && SimulationCode.ARRAY_EXP(__))) ||
       SimulationCode.ARRAY_PARAMETER(_, SOME(bindRaw && SimulationCode.ARRAY_EXP(__)))) => begin
        local bindArray = SimulationCode.toDAEExp(bindRaw)
        #= Extract element from the binding expression =#
        local element = if length(subExprs) == 1
          listGet(bindArray.array, first(subExprs))
        else
          #= Multi-dimensional array: navigate nested structure =#
          local current = bindArray
          for idx in subExprs
            current isa DAE.ARRAY || OMBackend.unsupported("a binding row that is not an array literal", current)
            current = listGet(current.array, idx)
          end
          current
        end
        #= Convert the extracted element to a Julia expression =#
        local constExpr = expToJuliaExpMTK(element, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
        return (true, constExpr)
      end
      _ => () #= Fall through to check scalarized variable or generate symbol reference =#
    end
  end

  #= When subscripts are constant, check if a scalarized element variable exists in
     the hash table (e.g. "R_T[1][2]" for subscripts [1,2]). Record field arrays
     create both a parent ARRAY variable and individual scalar element variables.
     The parent has no binding when values come from equations, so we must look up the
     scalarized name instead of generating runtime indexing on the parent. =#
  if allConstantSubscripts
    local scalarLookup = baseName * join(("[" * string(s) * "]" for s in subExprs))
    local scalarEntry = get(hashTable, scalarLookup, nothing)
    if scalarEntry !== nothing
      local scalarName = scalarEntry[2].name
      return (true, quote $(Symbol(string(varPrefix, scalarName, varSuffix))) end)
    end
  end

  #= Fallback: generate symbol reference for dynamic access =#
  local expr = if length(subExprs) == 1
    quote
      $(LineNumberNode(@__LINE__, "Array subscript: $baseName"))
      $(Symbol(string(varPrefix, baseName, varSuffix)))[$(first(subExprs))]
    end
  else
    quote
      $(LineNumberNode(@__LINE__, "Array subscript: $baseName"))
      $(Symbol(string(varPrefix, baseName, varSuffix)))[$(subExprs...)]
    end
  end

  return (true, expr)
end

"""
  Generate code for array expressions.
  For arrays with constants, evaluates at codegen time.
  For arrays with CREFs (variable references), generates code with symbolic expressions.
"""
function handleArrayExp(exp::DAE.ARRAY, simCode)
  local steps = listHead(exp.ty.dims)
  local dimSize = length(exp.ty.dims)
  @assert(steps isa DAE.DIM_INTEGER, "Only integer dimensions are currently supported. Type was : $(typeof(steps))")
  steps = steps.integer
  #= Determine element type from DAE type =#
  local elemType = @match exp.ty begin
    DAE.T_ARRAY(ty = DAE.T_REAL(__)) => Float64
    DAE.T_ARRAY(ty = DAE.T_INTEGER(__)) => Int
    _ => Float64  #= Default to Float64 =#
  end
  #= Check if all elements are constant (no variable references) at the DAE level =#
  local canEval = all(FrontendUtil.Util.isConstantExp, exp.array)
  #= Generate expressions for each element =#
  local elemExprs = [expToJuliaExpMTK(listGet(exp.array, i), simCode) for i in 1:steps]
  local arrJL = if canEval
    try
      [eval(expr) for expr in elemExprs]
    catch err
      OMBackend._fallback(err, :constantArrayElements; expect = Union{UndefVarError, MethodError})
      canEval = false
      []
    end
  else
    []
  end
  if canEval
    #= All elements are constants, return pre-computed array =#
    if dimSize >= 2
      #= rows of rows: stacked along the first dimension (a three-dimensional literal's
         rows are matrices, and transpose has no method for the stack: Buildings'
         Borefields TemporalSuperposition) =#
      arr = ndims(first(arrJL)) >= 2 ? stack(arrJL; dims = 1) : Matrix(transpose(stack(arrJL)))
      quote
        $(arr)
      end
    else
      quote
        $[arrJL...]
      end
    end
  else
    #= Contains CREFs, generate array with symbolic element expressions =#
    if dimSize >= 2
      #= For 2D arrays, generate hvcat for direct matrix construction =#
      local allScalars = Expr[]
      local nCols = 0
      local flattenable = true
      for i in 1:steps
        local rowDaeExp = listGet(exp.array, i)
        if @match rowDaeExp begin
          DAE.ARRAY(__) => true
          _ => false
        end
          local rowLen = listHead(rowDaeExp.ty.dims).integer
          if nCols == 0
            nCols = rowLen
          end
          for j in 1:rowLen
            push!(allScalars, expToJuliaExpMTK(listGet(rowDaeExp.array, j), simCode))
          end
        else
          flattenable = false
          break
        end
      end
      if flattenable && nCols > 0
        local colCounts = ntuple(_ -> nCols, steps)
        quote
          hvcat($(colCounts), $(allScalars...))
        end
      else
        #= Fallback: rows are not plain arrays =#
        quote
          let rows = [$(elemExprs...)]
            ndims(first(rows)) >= 2 ? stack(rows; dims = 1) : Matrix(transpose(stack(rows)))
          end
        end
      end
    else
      quote
        [$(elemExprs...)]
      end
    end
  end
end

"""
  If the system needs to conduct index reduction make sure to inform MTK.
(We avoid structural simplification for now since that might interfere with some other algorithms)
"""
function performStructuralSimplify(simplify; observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                                   split::Bool = !OMBackend.DIRECT_RHS_GENERATION[])::Expr
  #= Dump the pre-simplify ODESystem (equations + unknowns) so structural-balance
     debugging does not require re-running the model with extra instrumentation.
     Written to `backend/codeGen/preStructuralSimplify.log` next to the existing
     codegen logs. Only emitted when `ENABLE_BACKEND_LOGGING=true` was set at
     OMBackend load time; @BACKEND_LOGGING is a compile-time NOP otherwise so
     normal runs pay zero cost.
     Capture the absolute log path at codegen time so the dump lands in the
     same per-model run directory as the existing BDAE/simCode logs. The
     `logRunDir` stack is active during translate; by simulate time the model
     dir would no longer be on the stack and the dump would land in the
     session root. =#
  local dumpExpr = :(nothing)
  @BACKEND_LOGGING dumpExpr = dumpPreStructuralSimplifyExpr(
      OMBackend.logPath("backend/codeGen", "preStructuralSimplify.log"))
  local simplifyExpr = quote
    $dumpExpr
    reducedSystem = OMBackend.CodeGeneration.structural_simplify(firstOrderSystem; simplify = true, allow_parameter=true, split = $(split))
  end
  if observedFilter === nothing || isempty(observedFilter)
    return simplifyExpr
  end
  #= Embed the observed filter patterns into the generated code.
     After structural_simplify, filter the observed equations to keep only
     those whose LHS variable name matches at least one pattern. =#
  local patternStrings = if observedFilter isa Vector{Regex}
    [p.pattern for p in observedFilter]
  else
    observedFilter
  end
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
    #= Also seed from observed vars referenced by the system equations: the
       residual / init build_function references them, so their defs must survive. =#
    for _seq in ModelingToolkit.equations(reducedSystem)
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
  Generates different constructors for the ODESystem depending on given parameters.
  If-equation events use SymbolicContinuousCallback with discrete_parameters,
  so ifCond variables live in the parameter vector (not ODE state).
"""
function odeSystemWithEvents(hasEvents, modelName; hasObserved = false)
  #= When the model has events (if-equations), do NOT pass observed equations here.
     MTK's complete(sys) injects observed(sys) into every callback's AffectSystem
     (abstractsystem.jl:651), which causes tearing failures when observed equations
     introduce variables the callback sub-system cannot solve.
     Observed equations are injected into the reduced system AFTER structural_simplify
     returns, so callbacks never see them (see MTK_CodeGeneration.jl). =#
  #= `initial_eqs` is passed as `initialization_eqs` kwarg only when non-empty.
     These are constraints from the Modelica `initial equation` block that MUST
     hold at t=0 (e.g. `PID.gainPID.y = 0` for InitialOutput init of a PID).
     Without this, the constraints are passed only as `guesses` (Pair form),
     which MTK treats as starting points the solver may ignore. =#
  if hasEvents
    :(ODESystem(eqs, t, vars, parameters;
              name=:($(Symbol($modelName))),
              continuous_events = events, guesses = initialValues,
              initialization_eqs = initialConstraintEqs))
  elseif hasObserved
    :(ODESystem(eqs, t, vars, parameters;
              name=:($(Symbol($modelName))), guesses = initialValues,
              observed = observedEqs,
              initialization_eqs = initialConstraintEqs))
  else
    :(ODESystem(eqs, t, vars, parameters;
              name=:($(Symbol($modelName))), guesses = initialValues,
              initialization_eqs = initialConstraintEqs))
  end
end

"""
  Decide the iv of the condition (whether the zero-crossing function is at zero at t=0).
  Returns true if the zero-crossing expression evaluates to zero at t=0.
  Returns false if it evaluates to a nonzero value (guard is active or inactive).
  When simCode is provided, substitutes parameter values and state variable start values.
"""
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

#= ModelingToolkit.ifelse and the arithmetic if-expression lowering evaluate
   both branches. Guard only those eager branch payloads against one-sided
   event-localization noise around a relation boundary; ordinary powers keep
   their native domain semantics. =#
function _guardNonIntegerPowerBasesForEagerBranch(@nospecialize(e))
  if e isa Expr
    local args = Any[_guardNonIntegerPowerBasesForEagerBranch(a) for a in e.args]
    if e.head === :call && length(args) == 3 && args[1] == :^ &&
       _isNonIntegerJuliaConstant(args[3])
      return :(max($(args[2]), 0.0) ^ $(args[3]))
    end
    return Expr(e.head, args...)
  end
  return e
end

#= Numerically evaluate a scalar DAE.Exp at t0 against `valMap` (params +
   already-seeded vars). Returns nothing for anything outside the small
   arithmetic/`integer` subset, so callers fall back safely. =#
function _evalDAENumeric(@nospecialize(e), valMap::Dict{Symbol, Float64})::Union{Float64, Nothing}
  @match e begin
    DAE.RCONST(__) => Float64(e.real)
    DAE.ICONST(__) => Float64(e.integer)
    DAE.BCONST(__) => e.bool ? 1.0 : 0.0
    DAE.CREF(__) => begin
      local nm = SimulationCode.DAE_identifierToString(e.componentRef)
      nm == "time" ? 0.0 : get(valMap, Symbol(nm), nothing)
    end
    DAE.UNARY(operator = DAE.UMINUS(__)) => begin
      local a = _evalDAENumeric(e.exp, valMap); a === nothing ? nothing : -a
    end
    DAE.BINARY(__) => begin
      local a = _evalDAENumeric(e.exp1, valMap); a === nothing && return nothing
      local b = _evalDAENumeric(e.exp2, valMap); b === nothing && return nothing
      @match e.operator begin
        DAE.ADD(__) => a + b
        DAE.SUB(__) => a - b
        DAE.MUL(__) => a * b
        DAE.DIV(__) => b == 0.0 ? nothing : a / b
        DAE.POW(__) => (a < 0.0 && !isinteger(b)) ? nothing : a ^ b
        _ => nothing
      end
    end
    DAE.CAST(__) => _evalDAENumeric(e.exp, valMap)
    DAE.IFEXP(__) => begin
      local c = _evalDAENumeric(e.expCond, valMap); c === nothing && return nothing
      _evalDAENumeric(c != 0.0 ? e.expThen : e.expElse, valMap)
    end
    DAE.RELATION(__) => begin
      local a = _evalDAENumeric(e.exp1, valMap); a === nothing && return nothing
      local b = _evalDAENumeric(e.exp2, valMap); b === nothing && return nothing
      @match e.operator begin
        DAE.LESS(__) => a < b ? 1.0 : 0.0
        DAE.LESSEQ(__) => a <= b ? 1.0 : 0.0
        DAE.GREATER(__) => a > b ? 1.0 : 0.0
        DAE.GREATEREQ(__) => a >= b ? 1.0 : 0.0
        DAE.EQUAL(__) => a == b ? 1.0 : 0.0
        DAE.NEQUAL(__) => a != b ? 1.0 : 0.0
        _ => nothing
      end
    end
    DAE.LUNARY(DAE.NOT(__), _) => begin
      local a = _evalDAENumeric(e.exp, valMap); a === nothing && return nothing
      a == 0.0 ? 1.0 : 0.0
    end
    DAE.LBINARY(__) => begin
      local a = _evalDAENumeric(e.exp1, valMap); a === nothing && return nothing
      local b = _evalDAENumeric(e.exp2, valMap); b === nothing && return nothing
      @match e.operator begin
        DAE.AND(__) => (a != 0.0 && b != 0.0) ? 1.0 : 0.0
        DAE.OR(__) => (a != 0.0 || b != 0.0) ? 1.0 : 0.0
        _ => nothing
      end
    end
    DAE.CALL(path = Absyn.IDENT(fn)) => begin
      local argv = collect(e.expLst)
      isempty(argv) && return nothing
      #= smooth(k, expr) evaluates to its second argument. =#
      if fn == "smooth"
        return length(argv) == 2 ? _evalDAENumeric(argv[2], valMap) : nothing
      end
      local a = _evalDAENumeric(argv[1], valMap); a === nothing && return nothing
      if (fn == "min" || fn == "max") && length(argv) == 2
        local b = _evalDAENumeric(argv[2], valMap); b === nothing && return nothing
        return fn == "min" ? min(a, b) : max(a, b)
      end
      fn == "integer" ? (isfinite(a) ? floor(a) : nothing) :
      fn == "floor"   ? floor(a) :
      fn == "ceil"    ? ceil(a) :
      fn == "abs"     ? abs(a) :
      fn == "sqrt"    ? (a < 0.0 ? nothing : sqrt(a)) :
      fn == "noEvent" ? a :
      (fn == "float" || fn == "Real" || fn == "Integer") ? a : nothing
    end
    _ => nothing
  end
end

#= Substitute a loop iterator with a constant integer inside an expression,
   including in component-reference subscripts. =#
function _substIterT0(@nospecialize(e::DAE.Exp), iterId::String, value::Int)::DAE.Exp
  local repl = function (ex::DAE.Exp, arg)
    local res = ex
    @match ex begin
      DAE.CREF(DAE.CREF_IDENT(id, _, _), _) where (id == iterId) => begin
        res = DAE.ICONST(value)
        ()
      end
      _ => ()
    end
    return (res, true, arg)
  end
  return first(Util.traverseExpTopDown(e, repl, 0))
end

function _substIterStmt(@nospecialize(stmt), iterId::String, value::Int)
  @match stmt begin
    DAE.STMT_ASSIGN(__) =>
      DAE.STMT_ASSIGN(stmt.type_,
                      _substIterT0(stmt.exp1, iterId, value),
                      _substIterT0(stmt.exp, iterId, value),
                      stmt.source)
    DAE.STMT_IF(__) =>
      DAE.STMT_IF(_substIterT0(stmt.exp, iterId, value),
                  list((_substIterStmt(s, iterId, value) for s in stmt.statementLst)...),
                  _substIterElse(stmt.else_, iterId, value),
                  stmt.source)
    DAE.STMT_FOR(__) => begin
      #= An inner loop shadowing the same iterator keeps its own binding. =#
      if stmt.iter == iterId
        stmt
      else
        DAE.STMT_FOR(stmt.type_, stmt.iterIsArray, stmt.iter, stmt.index,
                     _substIterT0(stmt.range, iterId, value),
                     list((_substIterStmt(s, iterId, value) for s in stmt.statementLst)...),
                     stmt.source)
      end
    end
    _ => stmt
  end
end

function _substIterElse(@nospecialize(els), iterId::String, value::Int)
  @match els begin
    DAE.ELSE(__) => DAE.ELSE(list((_substIterStmt(s, iterId, value) for s in els.statementLst)...))
    DAE.ELSEIF(__) => DAE.ELSEIF(_substIterT0(els.exp, iterId, value),
                                 list((_substIterStmt(s, iterId, value) for s in els.statementLst)...),
                                 _substIterElse(els.else_, iterId, value))
    _ => els
  end
end

#= `written` (when given) collects the names an assignment wrote. =#
function _execInitAlgStmts!(valMap::Dict{Symbol, Float64}, stmts; written = nothing)::Nothing
  for stmt in stmts
    _execInitAlgStmt!(valMap, stmt; written)
  end
  return nothing
end

function _execInitAlgElse!(valMap::Dict{Symbol, Float64}, @nospecialize(els); written = nothing)::Nothing
  @match els begin
    DAE.ELSE(__) => _execInitAlgStmts!(valMap, els.statementLst; written)
    DAE.ELSEIF(__) => begin
      local c = _evalDAENumeric(els.exp, valMap)
      if c !== nothing
        c != 0.0 ? _execInitAlgStmts!(valMap, els.statementLst; written) :
                   _execInitAlgElse!(valMap, els.else_; written)
      end
      ()
    end
    _ => ()
  end
  return nothing
end

function _execInitAlgStmt!(valMap::Dict{Symbol, Float64}, @nospecialize(stmt); written = nothing)::Nothing
  @match stmt begin
    DAE.STMT_ASSIGN(__) => begin
      local nm = stmt.exp1 isa DAE.CREF ? SimulationCode.DAE_identifierToString(stmt.exp1) : nothing
      if nm !== nothing
        local v = _evalDAENumeric(stmt.exp, valMap)
        if v !== nothing
          valMap[Symbol(nm)] = v
          written === nothing || push!(written, Symbol(nm))
        end
      end
      ()
    end
    DAE.STMT_IF(__) => begin
      local c = _evalDAENumeric(stmt.exp, valMap)
      if c !== nothing
        if c != 0.0
          _execInitAlgStmts!(valMap, stmt.statementLst; written)
        else
          _execInitAlgElse!(valMap, stmt.else_; written)
        end
      end
      ()
    end
    DAE.STMT_FOR(__) => begin
      @match stmt.range begin
        DAE.RANGE(__) => begin
          local startV = _evalDAENumeric(stmt.range.start, valMap)
          local stopV = _evalDAENumeric(stmt.range.stop, valMap)
          local stepV = 1.0
          @match stmt.range.step begin
            SOME(se) => begin
              local sv = _evalDAENumeric(se, valMap)
              stepV = sv === nothing ? NaN : sv
              ()
            end
            _ => ()
          end
          #= Int() of a bound beyond Int64 throws an InexactError (review, 2026-09-29). =#
          if startV !== nothing && stopV !== nothing && isfinite(stepV) && stepV != 0.0 &&
             isinteger(startV) && isinteger(stopV) && isinteger(stepV) &&
             max(abs(startV), abs(stopV), abs(stepV)) < 2.0^62
            for iv in Int(startV):Int(stepV):Int(stopV)
              for s in stmt.statementLst
                _execInitAlgStmt!(valMap, _substIterStmt(s, stmt.iter, iv); written)
              end
            end
          end
          ()
        end
        _ => ()
      end
      ()
    end
    _ => ()
  end
  return nothing
end

#= Seed valMap with initial-algorithm-assigned values (e.g. trapezoid `count`,
   `T_start`) evaluated at t0, in statement order. Interprets assignments,
   if/elseif/else and for-loops over constant integer ranges; anything else
   is skipped and the affected names keep their defaults. Without this,
   discrete states set imperatively in an `initial algorithm` (no `start`
   attribute) default to 0.0 when evaluating if-equation initial branches,
   picking the wrong branch. =#
function _seedInitialAlgValues!(valMap::Dict{Symbol, Float64}, simCode; written = nothing)
  for ia in simCode.initialAlgorithms
    _execInitAlgStmts!(valMap, ia.daeStatements; written)
  end
  return valMap
end

#= True if the condition is TRUE at its zero-crossing boundary (zc == 0).
   The normalized crossing function cannot distinguish >= from > there; the
   original relational operator decides. Conservative: false for anything
   but a (possibly negated / noEvent-wrapped) plain relation. =#
function condClosedAtBoundary(cond)::Bool
  if cond isa SimulationCode.Exp
    cond = SimulationCode.toDAEExp(cond)
  end
  @match cond begin
    DAE.RELATION(_, DAE.LESSEQ(__), _) => true
    DAE.RELATION(_, DAE.GREATEREQ(__), _) => true
    DAE.RELATION(_, DAE.EQUAL(__), _) => true
    DAE.LUNARY(DAE.NOT(__), DAE.RELATION(_, DAE.LESS(__), _)) => true
    DAE.LUNARY(DAE.NOT(__), DAE.RELATION(_, DAE.GREATER(__), _)) => true
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local args = collect(lst)
      length(args) == 1 ? condClosedAtBoundary(args[1]) : false
    end
    _ => false
  end
end

#= Relation semantics for events (MLS 8.5, OpenModelica's relationhysteresis):
   a relation keeps its value during integration and changes only at events;
   its crossing function has a hysteresis relative to that value, so it is
   never zero after initialization or after an event (FMI 3.0 3.1.1). =#

#= The relations of a condition that generate events (not inside noEvent). =#
Base.@nospecializeinfer function _eventRelations(@nospecialize(cond))::Vector{DAE.Exp}
  cond isa SimulationCode.Exp && (cond = SimulationCode.toDAEExp(cond))
  local out = DAE.Exp[]
  local visit = function (e, arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("noEvent"), _, _) => return (e, false, arg)
      DAE.RELATION(__) => begin
        push!(out, e)
        return (e, false, arg)
      end
      _ => return (e, true, arg)
    end
  end
  Util.traverseExpTopDown(cond, visit, nothing)
  return out
end

#= The scale of a condition's operands, `1 + max(|a|, |b|, ...)`, for the
   hysteresis width `H * scale` (OpenModelica: max(|a|,|b|) + nominal). =#
function _conditionScaleExpr(@nospecialize(cond), simCode)
  local terms = Any[]
  #= An infinite operand (the MSL Spice3 V_pulse's default pulse width and
     period, inf, in `time >= T0 + Tfalling`) is left out: it made the
     hysteresis infinite. =#
  for rel in _eventRelations(cond)
    for x in (expToJuliaExpMTK(rel.exp1, simCode), expToJuliaExpMTK(rel.exp2, simCode))
      push!(terms, :(ModelingToolkit.ifelse(abs($(x)) < 1.0e300, abs($(x)), 0.0)))
    end
  end
  isempty(terms) && return 1.0
  return length(terms) == 1 ? :(1.0 + $(terms[1])) : :(1.0 + max($(terms...)))
end

#= The literal value of a condition (as at initialization), as a Julia Bool
   expression over `observed.zcK`: each relation from its own crossing
   function, `<`/`>` strict and `<=`/`>=` closed, and/or/not as such.
   `obsKws` collects the observed crossing functions. Nothing if the
   condition has a shape this does not cover. =#
Base.@nospecializeinfer function _literalConditionExpr(@nospecialize(cond), simCode, obsKws::Vector{Expr})
  cond isa SimulationCode.Exp && (cond = SimulationCode.toDAEExp(cond))
  local rec(@nospecialize e) = _literalConditionExpr(e, simCode, obsKws)
  local observe = function (zc)
    local name = Symbol("zc", length(obsKws) + 1)
    push!(obsKws, Expr(:kw, name, zc))
    return :(observed.$(name))
  end
  return @match cond begin
    DAE.RELATION(_, op, _) => begin
      local zc = observe(transformToMTKContinuousCondition(cond, simCode))
      (op isa DAE.LESSEQ || op isa DAE.GREATEREQ) ? :($(zc) <= 0) :
        #= == and <> cross as 0.5 - (a == b): negative exactly when it holds. =#
        (op isa DAE.LESS || op isa DAE.GREATER || op isa DAE.EQUAL || op isa DAE.NEQUAL) ? :($(zc) < 0) : nothing
    end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      local l = rec(e1); local r = rec(e2)
      (l === nothing || r === nothing) ? nothing : :($(l) && $(r))
    end
    DAE.LBINARY(e1, DAE.OR(__), e2) => begin
      local l = rec(e1); local r = rec(e2)
      (l === nothing || r === nothing) ? nothing : :($(l) || $(r))
    end
    DAE.LUNARY(DAE.NOT(__), e) => begin
      local inner = rec(e)
      inner === nothing ? nothing : :(!$(inner))
    end
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local args = collect(lst)
      length(args) == 1 ? rec(args[1]) : nothing
    end
    DAE.CREF(__) => :($(observe(expToJuliaExpMTK(cond, simCode))) > 0.5)
    DAE.BCONST(b) => b
    _ => nothing
  end
end

#= Symbol -> value map at t0: parameter values plus state/algebraic start
   attributes (default 0.0), overridden by early init-algorithm results.
   `explicit` collects the symbols whose value came from an actual source
   (parameter, explicit start attribute, init-algorithm result) rather than
   the 0.0 default. =#
#= One-entry memo: the map is requested once per if-equation branch but only
   depends on the simCode object, and the forward-evaluation sweep is not free.
   Fields: (simCode id, valMap, explicit, trusted-derived pairs). =#
const _T0_MAP_CACHE = Ref{Tuple{UInt, Dict{Symbol, Float64}, Set{Symbol}, Vector{Pair{Symbol, Float64}}}}(
  (UInt(0), Dict{Symbol, Float64}(), Set{Symbol}(), Pair{Symbol, Float64}[]))

#= The number `f()` (evalDAEConstant, evalSimCodeParameter) gives, or nothing
   when it is not a constant. A BINARY/LBINARY comes back as evalDAE_Expression's
   `:block` around its evaluated value (around the Expr itself when it reads a
   variable): `Float64` of the block was a MethodError, swallowed, so such a
   start value was lost (error policy, stage 4). =#
function _t0Number(f)::Union{Float64, Nothing}
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

#= The forms evalDAEConstant evaluates: a literal, or a BINARY/LBINARY through
   evalDAE_Expression. The t0 map asks only for those (anything else is not a
   constant, which is the rule, not a failure). =#
_isEvaluableConstant(@nospecialize(e))::Bool =
  e isa Union{DAE.BCONST, DAE.ICONST, DAE.RCONST, DAE.SCONST, DAE.BINARY, DAE.LBINARY,
              SimulationCode.BCONST, SimulationCode.ICONST, SimulationCode.RCONST, SimulationCode.SCONST,
              SimulationCode.BINARY, SimulationCode.LBINARY}

#= `sv`'s start value as a number, or nothing. =#
function _startNumber(sv, simCode)::Union{Float64, Nothing}
  local startExp = @match sv.attributes begin
    SOME(attr) where hasproperty(attr, :start) => @match attr.start begin
      SOME(e) => e
      _ => nothing
    end
    _ => nothing
  end
  return _isEvaluableConstant(startExp) ? _t0Number(() -> evalDAEConstant(startExp, simCode)) : nothing
end

_isFixedStart(sv)::Bool = @match sv.attributes begin
  SOME(attr) where hasproperty(attr, :fixed) => @match attr.fixed begin
    SOME(DAE.BCONST(true)) => true
    _ => false
  end
  _ => false
end

function _buildT0ValueMapAndExplicit(simCode)::Tuple{Dict{Symbol, Float64}, Set{Symbol}}
  local cached = _T0_MAP_CACHE[]
  if cached[1] === objectid(simCode)
    return (copy(cached[2]), copy(cached[3]))
  end
  local valMap = Dict{Symbol, Float64}()
  local explicit = Set{Symbol}()
  #= Values safe for hard structural decisions (branch selection): parameters,
     fixed=true starts, init-algorithm results. Non-fixed starts are guesses
     and may contradict the actual initial configuration. =#
  local trusted = Set{Symbol}()
  local ht = simCode.stringToSimVarHT
  for (key, (_, sv)) in ht
    local sym = Symbol(key)
    if sv.varKind isa SimulationCode.PARAMETER
      #= The binding's value, else the start value. =#
      local bind = @match sv.varKind begin
        SimulationCode.PARAMETER(SOME(b)) => b
        _ => nothing
      end
      local pval = _isEvaluableConstant(bind) ? _t0Number(() -> evalDAEConstant(bind, simCode)) : nothing
      pval === nothing && (pval = _startNumber(sv, simCode))
      if pval !== nothing
        valMap[sym] = pval
        push!(explicit, sym)
        push!(trusted, sym)
      end
    elseif SimulationCode.isStateOrAlgebraic(sv)
      local sval = _startNumber(sv, simCode)
      if sval !== nothing
        push!(explicit, sym)
        _isFixedStart(sv) && push!(trusted, sym)
      end
      valMap[sym] = something(sval, 0.0)
    end
  end
  #= The names the initial algorithms assign (by name: an assignment equal to
     the start value is an initial-algorithm result too). =#
  local written = Set{Symbol}()
  _seedInitialAlgValues!(valMap, simCode; written)
  union!(explicit, written)
  union!(trusted, written)
  #= Trusted-first sweep: deterministic consequences of trusted data override
     guess-grade start values in the map, so branch decisions never read a
     value contradicted by the fixed initial configuration. =#
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
  _T0_MAP_CACHE[] = (objectid(simCode), copy(valMap), copy(explicit), trustedDerived)
  return (valMap, explicit)
end

#= Trusted-derived t0 values (deterministic consequences of parameters and
   fixed=true starts). Used to override guess-grade init values at build time. =#
function buildT0TrustedDerivedPairs(simCode)::Vector{Pair{Symbol, Float64}}
  local cached = _T0_MAP_CACHE[]
  if cached[1] === objectid(simCode)
    return copy(cached[4])
  end
  _buildT0ValueMapAndExplicit(simCode)
  return copy(_T0_MAP_CACHE[][4])
end

_buildT0ValueMap(simCode)::Dict{Symbol, Float64} = first(_buildT0ValueMapAndExplicit(simCode))

#= True when every variable-position Symbol of the expression is explicit.
   Call heads and qualified references are code, not variables. =#
function _exprSymbolsExplicit(e, explicit::Set{Symbol})::Bool
  if e isa Symbol
    return e in explicit
  elseif e isa Expr
    e.head == :. && return true
    local args = e.head == :call ? e.args[2:end] : e.args
    for a in args
      a isa LineNumberNode && continue
      _exprSymbolsExplicit(a, explicit) || return false
    end
    return true
  end
  return true
end

#= Whole Float64 literals in index position fault plain numeric evaluation
   (the lowering emits `x[3.0]`); convert them to Int literals. =#
_intifyFloatIndices(x) = x
function _intifyFloatIndices(x::Expr)
  local newArgs = if x.head == :ref && length(x.args) >= 2
    vcat(Any[_intifyFloatIndices(x.args[1])],
         Any[(i isa Float64 && isinteger(i)) ? Int(i) : _intifyFloatIndices(i) for i in x.args[2:end]])
  else
    Any[_intifyFloatIndices(a) for a in x.args]
  end
  return Expr(x.head, newArgs...)
end

#= Evaluate a causalized branch RHS at the t0 value map. Returns nothing when
   the expression is not statically evaluable, or when it references any
   non-explicit (0.0-defaulted) variable: a seed contaminated by defaults can
   pull the consistent-IC solve to a wrong root. =#
function evalCausalRHSAtT0(rhsExpr, valMap::Dict{Symbol, Float64},
                           explicit::Set{Symbol} = Set{Symbol}())::Union{Float64, Nothing}
  if ccall(:jl_generating_output, Cint, ()) != 0
    return nothing
  end
  _exprSymbolsExplicit(rhsExpr, explicit) || return nothing
  local numExpr = _intifyFloatIndices(_substituteExprValues(rhsExpr, valMap))
  try
    #= Generated Modelica function bindings live in the parent CodeGeneration
       module (Phase A evals them there), not in this submodule. =#
    local result = Core.eval(parentmodule(@__MODULE__), numExpr)
    local numResult = if result isa Number
      Float64(result)
    else
      local unwrapped = Base.invokelatest(SymbolicUtils.unwrap, result)
      if unwrapped isa Number
        Float64(unwrapped)
      else
        return nothing
      end
    end
    return isfinite(numResult) ? numResult : nothing
  catch err
    #= A name not resolved yet, a Modelica function outside its domain. =#
    OMBackend._fallback(err, :evalCausalRHSAtT0; expect = Union{UndefVarError, MethodError})
    return nothing
  end
end

_t0ContainsDer(x) = false
function _t0ContainsDer(x::Expr)
  if x.head == :call && !isempty(x.args) && (x.args[1] === :der || x.args[1] === :D)
    return true
  end
  return any(_t0ContainsDer, x.args)
end

#= Forward-evaluate explicitly causal residuals (`0 = cref - rhs`) at t0 to
   extend the value map through chains the start attributes do not cover
   (frame positions, guarded line-force lengths). Operands must already be
   explicit, so defaulted values never contaminate a filled-in entry. =#
function _forwardEvalT0!(valMap::Dict{Symbol, Float64}, explicit::Set{Symbol}, simCode)
  if ccall(:jl_generating_output, Cint, ()) != 0
    return
  end
  local loweredCache = IdDict{Any, Any}()
  local mkPair = function (rawExp)
    local dae = rawExp isa DAE.Exp ? rawExp : SimulationCode.toDAEExp(rawExp)
    local arm = @match dae begin
      DAE.BINARY(DAE.CREF(cr, _), DAE.SUB(__), rhs) => (cr, rhs)
      _ => nothing
    end
    arm === nothing && return nothing
    local (cr, rhsDAE) = arm
    local tgt = Symbol(SimulationCode.DAE_identifierToString(cr))
    (tgt in explicit) && return nothing
    local rhsExpr = get!(loweredCache, rhsDAE) do
      try
        expToJuliaExpMTK(rhsDAE, simCode)
      catch err
        OMBackend._fallback(err, :forwardEvalT0Lowering; only = OMBackend.UnsupportedLowering)
        :__t0_lower_failed
      end
    end
    rhsExpr === :__t0_lower_failed && return nothing
    _t0ContainsDer(rhsExpr) && return nothing
    return (tgt, rhsExpr)
  end
  #= Outer rounds re-select if-equation branches as the value map grows: a
     branch whose condition depends on derived values only becomes decidable
     after the residual sweep has filled them. =#
  for _outer in 1:3
    local pairs = Tuple{Symbol, Any}[]
    for eq in simCode.residualEquations
      local p = mkPair(eq.exp)
      p === nothing || push!(pairs, p)
    end
    local envStr = Dict{String, Float64}(string(k) => valMap[k] for k in explicit if haskey(valMap, k))
    for ifEq in simCode.ifEquations
      local br = SimulationCode._selectActiveInitBranch(ifEq, envStr)
      br === nothing && continue
      for req in br.residualEquations
        local p = mkPair(req.exp)
        p === nothing || push!(pairs, p)
      end
    end
    local outerProgressed = false
    local progressed = true
    local rounds = 0
    while progressed && rounds < 20
      progressed = false
      rounds += 1
      for (tgt, rhsExpr) in pairs
        (tgt in explicit) && continue
        local v = evalCausalRHSAtT0(rhsExpr, valMap, explicit)
        v === nothing && continue
        valMap[tgt] = v
        push!(explicit, tgt)
        progressed = true
        outerProgressed = true
      end
    end
    outerProgressed || break
  end
  return
end

function evalInitialCondition(mtkCond, simCode; closedBoundary::Bool = false, extraVals = nothing)
  #= Skip during precompile output: `eval(...)` below would mutate this
     closed module's bindings and Julia rejects that. The runtime path is
     unaffected. Fallback `true` matches the existing catch arm. =#
  if ccall(:jl_generating_output, Cint, ()) != 0
    return true
  end
  #= Evaluate zero-crossing function at t=0 to determine initial condition.
     Works at the Expr level: substitutes all variable/parameter references
     in the mtkCond Expr (form :(lhs ~ 0)) with numeric values, then evals
     the result. Returns true when the zero-crossing function is non-negative
     at t=0 (condition FALSE), false when negative (condition TRUE).
     The caller inverts: ifCond = !(evalInitialCondition(...)). =#
  local valMap = _buildT0ValueMap(simCode)
  #= Relay-computed t0 values: condition operands that are themselves
     if-equation targets would otherwise default to 0 here and select the
     wrong initial branch. =#
  if extraVals !== nothing
    for (k, v) in extraVals
      valMap[k] = v
    end
  end
  local numExpr = _substituteExprValues(_extractZeroCrossingLHS(mtkCond), valMap)
  local numResult = try
    _numberOf(eval(numExpr))
  catch e
    #= A name without a t0 value, a Modelica function outside its domain. =#
    OMBackend._fallback(e, :evalInitialCondition; expect = Union{UndefVarError, MethodError}, impact = :result)
    @warn "evalInitialCondition: failed to evaluate, defaulting to true" exception=(e, catch_backtrace())
    return true
  end
  #= The zero-crossing function is negative when the condition is TRUE,
     positive when FALSE. Return true when condition is FALSE (positive),
     because the caller inverts: ifCond = !(evalInitialCondition(...)).
     At zc == 0 the original operator decides (closedBoundary). =#
  return closedBoundary ? numResult > 0.0 : numResult >= 0.0
end

#= An evaluated condition value as a Float64 (a symbolic constant unwrapped; 0.0 otherwise). =#
function _numberOf(result)::Float64
  result isa Number && return Float64(result)
  local unwrapped = Base.invokelatest(SymbolicUtils.unwrap, result)
  unwrapped isa Number && return Float64(unwrapped)
  local valued = Base.invokelatest(Symbolics.value, result)
  return Float64(valued isa Number ? valued : 0.0)
end

"""
Extract the LHS from a zero-crossing equation Expr of the form :(lhs ~ 0).
Handles nested forms like :(min(a, b) ~ 0).
"""
function _extractZeroCrossingLHS(expr::Expr)
  if expr.head == :call && length(expr.args) == 3 && expr.args[1] == :~
    return expr.args[2]
  end
  return expr
end

"""
  Generates code for DAE cast expressions for MTK code.
"""
function generateCastExpressionMTK(@nospecialize(ty::DAE.Type), @nospecialize(exp::DAE.Exp),
                                   simCode, varPrefix = "", varSuffix = "")
  expr = @match ty, exp begin
    (DAE.T_REAL(__), DAE.ICONST(__)) => begin
      quote
        float($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix, varSuffix = varSuffix,)))
      end
    end
    (DAE.T_REAL(__), DAE.CREF(cref)) where typeof(cref.identType) === DAE.T_INTEGER => begin
      quote
        float($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix, varSuffix = varSuffix,)))
      end
    end
    #= Conversion to a float, other alternatives. =#
    (DAE.T_REAL(__), _) => begin
      quote
        float($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix, varSuffix = varSuffix,)))
      end
    end
    #= Conversion of array to real array (broadcast float) =#
    (DAE.T_ARRAY(DAE.T_REAL(__), _), _) => begin
      quote
        float.($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix, varSuffix = varSuffix,)))
      end
    end
    #= Conversion to integer array =#
    (DAE.T_ARRAY(DAE.T_INTEGER(__), _), _) => begin
      quote
        Int.(round.($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix, varSuffix = varSuffix,))))
      end
    end
    _ => OMBackend.unsupported("cast to $(ty)", exp)
  end
  return expr
end

# TODO: unify cref resolution into one function consulting both the SimCode
# (state / numeric-param lookup tables) and the module-level bindings (String
# parameters, data structures), so callers need not special-case the latter.
getIdxForLookupMTK(x::Union{DAE.ComponentRef, DAE.CREF}, simCode) = getIdxForLookupMTK(string(x), simCode)

function getIdxForLookupMTK(crefAsStr::String, simCode)
  if crefAsStr == "time"
    return :t
  end
  @match _, simVar = simCode.stringToSimVarHT[crefAsStr]
  if !(SimulationCode.isParameter(simVar))
    Expr(:call, getindex, :x, Expr(:call, :getindex, :lookuptableStates, :(Symbol($(crefAsStr)))))
  else
    Expr(:call, getindex, :p, Expr(:call, :getindex, :lookuptableParams, :(Symbol($(crefAsStr)))))
  end
end


#= Helpers that depend on the MTK lowering (expToJuliaExpMTK / ModelingToolkit.ifelse / Symbolics) =#

_isSimCodeFunctionPath(path::Absyn.Path, simCode)::Bool = _isSimCodeFunctionName(string(path), simCode)

#= An element of a named array of records by subscripts that are not literals (`data[i]` in a
   reduction: the ideal gases' `h_T(data[i], T, ...)`, MSL Media mixtures): that element of
   each field array, `data_MM[i]` (the module-level arrays of _recordArrayFieldArrays). With
   the subscript dropped (before toSimExp kept it), the whole field arrays were passed. Empty
   otherwise. =#
function _recordArrayElementCallArgs(arg::DAE.Exp, simCode::SimulationCode.SIM_CODE;
                                     varPrefix::String = "", varSuffix::String = "", derSymbol::Bool = false)::Vector{Any}
  (arg isa DAE.ASUB && arg.exp isa DAE.CREF) || return Any[]
  local elemTy = arg.exp.ty
  local nDims = 0
  while elemTy isa DAE.T_ARRAY
    nDims += length(collect(elemTy.dims))
    elemTy = elemTy.ty
  end
  (elemTy isa DAE.T_COMPLEX && elemTy.complexClassType isa DAE.ClassInf.RECORD) || return Any[]
  local subs = collect(arg.sub)
  #= the base's type is the array's, or the element's (toSimExp of a subscripted cref types
     the base by its identifier's type) =#
  (nDims == 0 || length(subs) == nDims) || return Any[]
  local base = SimulationCode.string(arg.exp.componentRef)
  occursin('[', base) && return Any[]
  #= an ASUB's subscripts: expressions, or INDEX subscripts (a slice is not an element) =#
  local idx = Any[]
  for sub in subs
    local e = sub isa DAE.INDEX ? sub.exp : sub
    (e isa DAE.Exp && !(e isa DAE.RANGE || e isa DAE.ARRAY)) || return Any[]
    push!(idx, expToJuliaExpMTK(e, simCode; varPrefix, varSuffix, derSymbol))
  end
  return Any[Expr(:ref, Symbol(base * COMPONENT_SEPARATOR * f.name), idx...) for f in elemTy.varLst]
end

function _modelicaFunctionCallArgs(expLst,
                                   simCode,
                                   hashTable;
                                   varPrefix = "",
                                   varSuffix = "",
                                   derSymbol = false)
  local args::Vector{Any} = Any[]
  for arg in expLst
    local flattenedArgs::Vector{Symbol} = flattenRecordCallArg(arg, simCode, hashTable;
                                                               varPrefix = varPrefix,
                                                               varSuffix = varSuffix)
    if !isempty(flattenedArgs)
      append!(args, flattenedArgs)
      continue
    end
    local elementFields = _recordArrayElementCallArgs(arg, simCode; varPrefix, varSuffix, derSymbol)
    if !isempty(elementFields)
      append!(args, elementFields)
      continue
    end

    local extracts = _expandComplexReturnArg(arg, simCode, hashTable;
                                             varPrefix = varPrefix,
                                             varSuffix = varSuffix,
                                             derSymbol = derSymbol)
    if extracts !== nothing
      append!(args, extracts)
      continue
    end

    push!(args, expToJuliaExpMTK(arg, simCode;
                                 varPrefix = varPrefix,
                                 varSuffix = varSuffix,
                                 derSymbol = derSymbol))
  end
  return args
end

function subscriptsToExpr(subscripts, simCode; varPrefix="", varSuffix="", derSymbol=:der)
  local exprs = map(subscripts) do sub
    @match sub begin
      DAE.INDEX(DAE.ICONST(i)) => i
      DAE.ICONST(i) => i
      DAE.INDEX(idxExp) => expToJuliaExpMTK(idxExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
      DAE.SLICE(idxExp) => expToJuliaExpMTK(idxExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
      DAE.WHOLEDIM(__) => :(:)
      _ => Meta.parse(string(sub))  #= Fallback for unknown subscript types =#
    end
  end
  if length(exprs) == 1
    return first(exprs)
  else
    return Expr(:tuple, exprs...)
  end
end

function evalDAE_Expression(expr, simCode; keepTunable::Bool = false)::Expr
  local shouldEval = Ref(true)
  #= Replaces all known bound parameters in the DAE expression. This must be
     recursive: parameter aliases such as `actualGlobalSeed = globalSeed_seed`
     otherwise leave a bare `globalSeed_seed` in the generated Julia expression
     even after `globalSeed_seed` itself has been solved from an initial equation.
     With `keepTunable`, tunable parameters stay references (_substituteBoundParameters). =#
  local daeExp = _substituteBoundParameters(expr, simCode; shouldEval=shouldEval, keepTunable=keepTunable)
  local jlExpr = expToJuliaExpMTK(daeExp, simCode)
  local evaluatedJLExpr = if shouldEval[]
    try
      eval(jlExpr)
    catch err
      #= Left unevaluated: it reads a name that is not a constant here. =#
      OMBackend._fallback(err, :evalDAE_Expression; expect = Union{UndefVarError, MethodError})
      jlExpr
    end
  else
    jlExpr
  end
  return quote $(evaluatedJLExpr) end
end

#= Whether `exp` reads a tunable parameter (withTunableParameters), directly or
   through the bindings of the parameters it reads. =#
function _readsTunableParameter(exp, simCode)::Bool
  isempty(OMBackend.TUNABLE_PARAMETERS[]) && return false
  local resolved = _substituteBoundParameters(exp, simCode; keepTunable = true)
  return any(c -> OMBackend.isTunableParameter(string(c)), Util.getAllCrefs(resolved))
end

#= `e` with `time` read as 0.0, the start time of the build. =#
_timeAtBuildStart(@nospecialize(e)) = first(Util.traverseExpBottomUp(e, (x, acc) ->
  (x isa DAE.CREF && x.componentRef isa DAE.CREF_IDENT && x.componentRef.ident == "time" ? DAE.RCONST(0.0) : x, acc), nothing))

"""
    solveParametricInitialEquations!(simCode)

For initial equations that only involve parameters (no states/algebraics),
solve numerically for parameters with `fixed=false` (no binding).
Updates the simCode hash table with the solved binding values.
"""
function solveParametricInitialEquations!(simCode::SimulationCode.SimCode)
  ht = simCode.stringToSimVarHT
  local function containsCref(exp, name::String)::Bool
    local found = Ref(false)
    function visit(e, acc)
      if !found[] && Util.isCref(e) && string(e) == name
        found[] = true
      end
      (e, true, acc)
    end
    Util.traverseExpBottomUp(exp, visit, 0)
    return found[]
  end
  local function containsIntegerCref(exp, name::String)::Bool
    local found = Ref(false)
    function visit(e, acc)
      if !found[] && Util.isCref(e) && string(e) == name
        @match e begin
          DAE.CREF(_, DAE.T_INTEGER(__)) => begin
            found[] = true
            nothing
          end
          _ => nothing
        end
      end
      (e, true, acc)
    end
    Util.traverseExpBottomUp(exp, visit, 0)
    return found[]
  end
  local function solvedValueExp(x, asInteger::Bool)
    if asInteger && isfinite(x)
      return DAE.ICONST(Int(round(x)))
    end
    return DAE.RCONST(x)
  end
  #= Iterate to fixed-point: each pass may bind a free parameter that another
     equation depends on. Cap iterations to (length+1) so a chain of N equations
     finishes even with the worst-case ordering. =#
  local nEq = length(simCode.initialEquations)
  local solvedThisPass = true
  local pass = 0
  while solvedThisPass && pass <= nEq
    solvedThisPass = false
    pass += 1
    local solvedNames = String[]
  for ieq in simCode.initialEquations
    if !isParametricOnlyEquation(ieq, simCode)
      continue
    end
    #= Find free parameters (no binding) in this equation =#
    freeParams = String[]
    function findFree(exp, acc)
      if Util.isCref(exp)
        local key = string(exp)
        local entry = get(ht, key, nothing)
        if entry !== nothing
          local sv = last(entry)
          if !SimulationCode.hasBindingExp(sv) && SimulationCode.isParameter(sv)
            push!(freeParams, key)
          end
        end
      end
      (exp, true, acc)
    end
    #= At the build the start time is 0 (`t0 = time`, MSL Blocks.Math.Mean;
       simulateFromBuild refuses another start time for such a model,
       _startTimeGuard): `time` could not be evaluated, and the parameter kept
       its start value. =#
    local ieqLhs, ieqRhs = map(_timeAtBuildStart, equationSides(ieq))
    Util.traverseExpBottomUp(ieqLhs, findFree, 0)
    Util.traverseExpBottomUp(ieqRhs, findFree, 0)
    unique!(freeParams)
    if length(freeParams) != 1
      continue
    end
    freeName = freeParams[1]
    #= A tunable parameter is set at run time: solved here, the free one kept
       its compiled value in every run (`q = 2 * p`). Alone on a side it is
       bound to the other side (`p = q`, `2 * p = q`), and follows p as a
       bound parameter does; otherwise it stays unbound and the
       initialization computes it for the run's values (`q * q = p`). As an
       unbound parameter alone on the right it was assigned nowhere: q = 0. =#
    if any(side -> _readsTunableParameter(side, simCode), (ieqLhs, ieqRhs))
      local isFree = e -> e isa DAE.CREF && string(e) == freeName
      local other = isFree(ieqLhs) ? ieqRhs : isFree(ieqRhs) ? ieqLhs : nothing
      if other !== nothing && !containsCref(other, freeName)
        local (idx, oldSV) = ht[freeName]
        ht[freeName] = (idx, SimulationCode.SIMVAR(oldSV.name, oldSV.index,
                                                   SimulationCode.PARAMETER(SOME(SimulationCode.toSimExp(other))), oldSV.attributes))
        push!(solvedNames, freeName)
        solvedThisPass = true
      end
      continue
    end
    #= Get initial guess from start attribute =#
    local (_, freeSV) = ht[freeName]
    local guess = something(_startNumber(freeSV, simCode), 0.1)
    #= Build residual: LHS - RHS = 0.
       Replace all bound params recursively, leave the free param as the scalar
       Newton variable. This handles alias chains like
       actualGlobalSeed = globalSeed_seed, where globalSeed_seed was solved by
       an earlier initial equation in the same fixed-point loop. =#
    local skipFree = OrderedSet{String}([freeName])
    local lhsEvalOk = Ref(true)
    local rhsEvalOk = Ref(true)
    local lhsSubst = _substituteBoundParameters(ieqLhs, simCode;
                                                skipNames=skipFree,
                                                shouldEval=lhsEvalOk)
    local rhsSubst = _substituteBoundParameters(ieqRhs, simCode;
                                                skipNames=skipFree,
                                                shouldEval=rhsEvalOk)
    #= If the free parameter sits on the LHS (e.g. `globalSeed_seed = automaticGlobalSeed(0.0)`),
       swap the sides so eval(lhsJl) is on the constants side and the freeName lives in the
       residual function we Newton-solve. Without this swap eval(lhsJl) tries to evaluate the
       bare freeName cref and trips an UndefVarError. =#
    if containsCref(lhsSubst, freeName) && !containsCref(rhsSubst, freeName)
      lhsSubst, rhsSubst = rhsSubst, lhsSubst
      lhsEvalOk, rhsEvalOk = rhsEvalOk, lhsEvalOk
    end
    local freeIsInteger = containsIntegerCref(ieqLhs, freeName) || containsIntegerCref(ieqRhs, freeName)
    #= Defer if substitution left an unresolved CREF; the fixed-point loop will retry. =#
    if !lhsEvalOk[]
      continue
    end
    local lhsJl = expToJuliaExpMTK(lhsSubst, simCode)
    local lhsVal = try
      local raw = eval(lhsJl)
      raw isa Symbolics.Num ? Float64(Symbolics.unwrap(raw)) : Float64(raw)
    catch err
      OMBackend._fallback(err, :parametricInitLhs; expect = Union{UndefVarError, MethodError}, impact = :result)
      @warn "[SIMCODE: solveParametricInitialEquations] could not evaluate LHS" freeName err
      continue
    end
    #= Common case: a free parameter aliases a bound parameter/literal directly,
       e.g. `globalSeed_seed = globalSeed_fixedSeed`. Avoid Newton here; it
       may keep the start guess if the generated residual fails to depend on the
       argument due to world-age/module binding subtleties. =#
    if containsCref(rhsSubst, freeName) && rhsSubst isa DAE.CREF
      local (idx, oldSV) = ht[freeName]
      local newSV = SimulationCode.SIMVAR(oldSV.name, oldSV.index,
        SimulationCode.PARAMETER(SOME(SimulationCode.toSimExp(solvedValueExp(lhsVal, freeIsInteger)))), oldSV.attributes)
      ht[freeName] = (idx, newSV)
      push!(solvedNames, freeName)
      solvedThisPass = true
      continue
    end
    if !rhsEvalOk[]
      continue
    end
    local freeSymbol = Symbol(freeName)
    local rhsJl = expToJuliaExpMTK(rhsSubst, simCode)
    #= Create residual function: f(x) = lhsVal - rhs(x) =#
    local residualFn = try
      eval(Expr(:->, freeSymbol, Expr(:call, :-, lhsVal, rhsJl)))
    catch e
      OMBackend._fallback(e, :parametricInitResidual; expect = Union{UndefVarError, MethodError}, impact = :result)
      @warn "[SIMCODE: solveParametricInitialEquations] could not build residual" freeName e
      continue
    end
    #= Newton-Raphson solver (use invokelatest to avoid world-age issues).
       Wrapped in try/catch because rhsJl may reference parameters that have
       a binding the front-end could not fold (so they are not in paramValues
       and not freeName either). Invoking residualFn on such an expression
       throws UndefVarError; without this catch the exception propagates out
       of solveParametricInitialEquations and aborts translate. =#
    local x = guess
    local eps = 1e-10
    local maxIter = 100
    local newtonOk = true
    local lastStep = Inf
    try
      for _ in 1:maxIter
        local fx = Base.invokelatest(residualFn, x)
        if abs(fx) < eps
          break
        end
        local h = max(abs(x) * 1e-8, 1e-12)
        local dfx = (Base.invokelatest(residualFn, x + h) - Base.invokelatest(residualFn, x - h)) / (2h)
        if abs(dfx) < 1e-15
          break
        end
        lastStep = fx / dfx
        x -= lastStep
      end
    catch err
      OMBackend._fallback(err, :parametricInitNewton; expect = Union{UndefVarError, MethodError}, impact = :result)
      @warn "[SIMCODE: solveParametricInitialEquations] residual call threw, skipping" freeName err
      newtonOk = false
    end
    #= Converged: a small residual, or a small last step (the rounding of
       `0 = 1e9*G - 2e9` is about 4e-7). Otherwise the parameter stays unbound,
       for the initialization (generateInitialEquationsAsConstraints takes an
       equation reading one). =#
    newtonOk = newtonOk && let fx = try Base.invokelatest(residualFn, x) catch; NaN end
      isfinite(x) && isfinite(fx) &&
        (abs(fx) <= 1e-8 * max(1.0, abs(lhsVal)) || abs(lastStep) <= 1e-10 * max(1.0, abs(x)))
    end
    if !newtonOk
      continue
    end
    @debug "[SIMCODE: solveParametricInitialEquations] solved $freeName = $x (from initial equation)"
    #= Update the simCode hash table with the solved value =#
    local (idx, oldSV) = ht[freeName]
    local newSV = SimulationCode.SIMVAR(oldSV.name, oldSV.index,
      SimulationCode.PARAMETER(SOME(SimulationCode.toSimExp(solvedValueExp(x, freeIsInteger)))), oldSV.attributes)
    ht[freeName] = (idx, newSV)
    push!(solvedNames, freeName)
    solvedThisPass = true
  end
  if pass == 1 && !isempty(solvedNames)
    @debug "[SIMCODE: solveParametricInitialEquations] pass $pass solved $(length(solvedNames)) parameter(s)" solvedNames
  elseif !isempty(solvedNames)
    @debug "[SIMCODE: solveParametricInitialEquations] pass $pass solved $(length(solvedNames)) more parameter(s)" solvedNames
  end
  end #= while fixed-point =#
end

#= True when every variable reference in `condition` is a DISCRETE simvar, a
   PARAMETER, or a constant (no STATE / ALG_VARIABLE / `time`). For such conditions
   the if-expression can gate directly on the held discrete value; the discrete's
   own update event localises the switch, so no ifCond relay is needed. =#
function _ifConditionAllDiscreteOrParameter(@nospecialize(condition), simCode)::Bool
  #= initial() is true during the initialization only: a condition with it is
     a relation (its crossing function a constant of either sign), not a gate
     on the runtime value, where initial() is false (MSL LimIntegrator's
     `initial() and not limitsAtInit` never held). =#
  _hasInitialCall(condition) && return false
  local refs::OrderedSet{String} = OrderedSet{String}()
  SimulationCode.collectCrefNames!(refs, condition)
  isempty(refs) && return false
  local ht = simCode.stringToSimVarHT
  for name in refs
    name == "time" && return false
    local entry = get(ht, name, nothing)
    entry === nothing && return false
    local kind = entry[2].varKind
    if !(kind isa SimulationCode.DISCRETE || kind isa SimulationCode.PARAMETER)
      return false
    end
  end
  return true
end

function _hasInitialCall(@nospecialize(condition))::Bool
  local d = condition isa SimulationCode.Exp ? SimulationCode.toDAEExp(condition) : condition
  local found = Ref(false)
  Util.traverseExpBottomUp(d, (x, acc) -> begin
    x isa DAE.CALL && x.path isa Absyn.IDENT && x.path.name == "initial" && (found[] = true)
    (x, true, acc)
  end, 0)
  return found[]
end

"True when every non-else branch condition of an if-equation is discrete/parameter,
 so the residual gates directly on the boolean condition instead of emitting a
 continuous ifCond relay callback."
function _allBranchConditionsDiscrete(branches, simCode)::Bool
  for b in branches
    b.targets == -1 && continue
    _ifConditionAllDiscreteOrParameter(b.condition, simCode) || return false
  end
  return true
end

#= Residuals pair across branches by the variable their causalized form
   defines, with the positional index only as a fallback: source-order
   differences between sibling branches (hoisted nested ifs, lifted
   if-expressions) make pure positional pairing relay the wrong variable. =#
function _causalLhsKey(@nospecialize(lhsExpr))::String
  local e = lhsExpr isa Expr ? Base.remove_linenums!(copy(lhsExpr)) : lhsExpr
  while e isa Expr && e.head === :block && length(e.args) == 1
    e = e.args[1]
  end
  return string(e)
end

function _branchResidualForLhs(branch, lhsKey::String, fallbackIdx::Int, simCode)
  for r in branch.residualEquations
    if _causalLhsKey(last(deCausalize(r, simCode))) == lhsKey
      return r
    end
  end
  return branch.residualEquations[min(fallbackIdx, length(branch.residualEquations))]
end

"""
  Generates an if-expression equation and add it to the continuous part of the system.
Assume single equations in each if-branch for now.
An assertion error should have been thrown earlier before reaching this function.

The sub identifier is used to for the different branches of a single if-equation.
Hence for the model:

```modelica
model IfEquationDer
  parameter Real u = 4;
  parameter Real uMax = 10;
  parameter Real uMin = 2;
  Real y;
equation
  if uMax < time then
    der(y) = uMax;
  elseif uMin < time then
    der(y) = uMin;
  else
    der(y) = u;
  end if;
end IfEquationDer;
```

The if expression:
```
D(y) ~ ifelse(ifCond11 == true, uMin, ifelse(ifCond12 == true, uMax, u))
```
will be generated along with variables for the sub branches.

"""
function generateIfExpressions(branches,
                               target::Int,
                               resEqIdx::Int,
                               identifier::Int,
                               simCode;
                               subIdentifier::Int = 1,
                               lhsKey::Union{String, Nothing} = nothing,
                               residualForm::Bool = false)
  local branch = branches[target]
  local selEq = (lhsKey === nothing || residualForm) ?
                branch.residualEquations[min(resEqIdx, length(branch.residualEquations))] :
                _branchResidualForLhs(branch, lhsKey, resEqIdx, simCode)
  #= The branch's value: its right-hand side (causal form), or its residual
     when the branches define different variables. =#
  local value = _guardNonIntegerPowerBasesForEagerBranch(
    residualForm ? expToJuliaExpMTK(SimulationCode.toDAEExp(selEq.exp), simCode) :
                   first(deCausalize(selEq, simCode)))
  if branch.targets == -1
    return :($(value))
  end
  #= When every branch condition is discrete/parameter (e.g. an event-held Boolean),
     gate directly on the condition value: its own update event localises the step,
     so no `ifCond` relay parameter or continuous callback is needed (and the relay's
     start-attribute initial value, which can pick the wrong branch, is avoided). =#
  local cond = if _ifConditionAllDiscreteOrParameter(branch.condition, simCode)
    #= Per branch: an elseif chain may mix a discrete condition (changed by a
       when, never a crossing of its own) with relations. =#
    :( $(expToJuliaExpMTK(SimulationCode.toDAEExp(branch.condition), simCode)) > 0.5 )
  else
    #= ifCond variables are discrete parameters (not ODE unknowns), so the solver
       never perturbs them during Jacobian computation. Exact comparison is safe. =#
    :( $(Symbol(string("ifCond", identifier, subIdentifier))) == 1 )
  end
  quote
    ModelingToolkit.ifelse($(cond),
                           $(value),
                           $(generateIfExpressions(branches,
                                                   branches[target].targets,
                                                   resEqIdx,
                                                   identifier,
                                                   simCode;
                                                   subIdentifier = subIdentifier + 1,
                                                   lhsKey = lhsKey,
                                                   residualForm = residualForm)))
  end
end

#= TODO.
  We currently assume residuals that we have made causal
  and that the original equations are written in a certain form.
=#
function deCausalize(eq, simCode)
  local expDAE = SimulationCode.toDAEExp(eq.exp)
  @match expDAE begin
    DAE.BINARY(DAE.RCONST(0.0), _, exp2) => begin
      (:($(expToJuliaExpMTK(exp2, simCode))), :($(expToJuliaExpMTK(expDAE.exp1, simCode))))
    end
    DAE.BINARY(exp1, _, DAE.RCONST(0.0)) => begin
      (:($(expToJuliaExpMTK(expDAE.exp2, simCode))), :($(expToJuliaExpMTK(exp1, simCode))))
    end
    DAE.BINARY(exp1, _, exp2) => begin
      (:($(expToJuliaExpMTK(exp2, simCode))), :($(expToJuliaExpMTK(exp1, simCode))))
    end
    _ => begin
      OMBackend.unsupported("equation", eq)
    end
  end
end

#= If `arg` is a Complex-returning expression (CALL with T_COMPLEX return,
   or RECORD constructor), produce a Vector{Expr} of per-field scalar extracts
   suitable for splicing into an enclosing call's argument list. Returns
   `nothing` for anything else so callers can fall back to the default path.

   For a CALL whose return is T_COMPLEX(varLst=[re, im]), this emits
   `tupleElementCall(:funcName, k, ...inner-scalar-args...)` for k = 1..nFields.
   For a RECORD literal it just splices the field expressions. =#
function _expandComplexReturnArg(arg::DAE.Exp, simCode, hashTable;
                                  varPrefix::String="", varSuffix::String="",
                                  derSymbol::Bool=false)
  @match arg begin
    DAE.CALL(path, innerExpLst, DAE.CALL_ATTR(ty=DAE.T_COMPLEX(varLst=varLst))) => begin
      local nFields = length(collect(varLst))
      nFields >= 2 || return nothing
      local fnName = Symbol(string(path))
      local fnQuote = QuoteNode(fnName)
      local innerArgs = Any[]
      for inner in innerExpLst
        local flat = flattenRecordCallArg(inner, simCode, hashTable; varPrefix=varPrefix, varSuffix=varSuffix)
        if !isempty(flat)
          append!(innerArgs, flat)
          continue
        end
        local nested = _expandComplexReturnArg(inner, simCode, hashTable;
                                                varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
        if nested !== nothing
          append!(innerArgs, nested)
          continue
        end
        push!(innerArgs, expToJuliaExpMTK(inner, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol))
      end
      return Any[:(OMBackend.CodeGeneration.tupleElementCall($fnQuote, $k, $(innerArgs...))) for k in 1:nFields]
    end
    DAE.RECORD(_, expl, _, DAE.T_COMPLEX(__)) => begin
      local fieldExprs = [expToJuliaExpMTK(e, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
                          for e in expl]
      length(fieldExprs) >= 2 || return nothing
      return Any[fieldExprs...]
    end
    _ => return nothing
  end
end

function _modelicaFunctionCallExpr(path,
                                   expLst,
                                   simCode,
                                   hashTable;
                                   varPrefix = "",
                                   varSuffix = "",
                                   derSymbol = false)
  local normalizedFuncName = OMBackend.canonicalName(string(path))
  #= OMFrontend lowers delay(x, T[, Tmax]) to OpenModelica.Internal.delay2/delay3, as OpenModelica: a
     generated function whose body was the identity. The history-based term (delays.jl). =#
  if normalizedFuncName in ("OpenModelica_Internal_delay2", "OpenModelica_Internal_delay3")
    return DAECallExpressionToMTKCallExpression("delay", expLst isa List ? expLst : MetaModelica.list(expLst...), simCode, hashTable;
                                                varPrefix = varPrefix, varSuffix = varSuffix, derAsSymbol = derSymbol)
  end
  local lowered = lowerKnownSymbolicFunctionCall(normalizedFuncName, expLst, simCode, hashTable;
                                                varPrefix = varPrefix,
                                                varSuffix = varSuffix,
                                                derSymbol = derSymbol)
  lowered !== nothing && return lowered
  #= Qualified MSL paths (e.g. Modelica.Math.Vectors.length) only reach this
     branch because the bare-name dispatcher fires for Absyn.IDENT calls.
     Reuse MODELICA_BUILTIN_FUNCTIONS so a registered Julia mirror resolves
     the call instead of emitting an unresolved Symbol that fails at eval. =#
  local builtinSym = get(AlgorithmicCodeGeneration.MODELICA_BUILTIN_FUNCTIONS,
                         normalizedFuncName, nothing)
  if builtinSym !== nothing
    local builtinCallee = Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)),
                                        QuoteNode(:AlgorithmicCodeGeneration)),
                               QuoteNode(builtinSym))
    local builtinExpr = Expr(:call, builtinCallee)
    append!(builtinExpr.args, _modelicaFunctionCallArgs(expLst, simCode, hashTable;
                                                       varPrefix = varPrefix,
                                                       varSuffix = varSuffix,
                                                       derSymbol = derSymbol))
    return builtinExpr
  end
  local runtimeName = get(AlgorithmicCodeGeneration.MODELICA_UTILITIES_TO_RUNTIME_C,
                          normalizedFuncName, nothing)
  local callee = if runtimeName !== nothing
    :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.RuntimeCCall($(Expr(:., :OMRuntimeExternalC, QuoteNode(runtimeName)))))
  else
    Symbol(normalizedFuncName)
  end
  local expr = Expr(:call, callee)
  append!(expr.args, _modelicaFunctionCallArgs(expLst, simCode, hashTable;
                                               varPrefix = varPrefix,
                                               varSuffix = varSuffix,
                                               derSymbol = derSymbol))
  return expr
end

using ExportAll
@exportAll()

end #= module MTK_CodeGenerationUtil =#
