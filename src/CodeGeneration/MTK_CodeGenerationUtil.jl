#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* This program is distributed WITHOUT ANY WARRANTY. See the OSMC Public License
* for details.
=#

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
      :($(expToJuliaBoolMTK(e1, simCode; cachedChange = cachedChange)) && $(expToJuliaBoolMTK(e2, simCode; cachedChange = cachedChange)))
    DAE.LBINARY(exp1 = e1, operator = DAE.OR(__), exp2 = e2) =>
      :($(expToJuliaBoolMTK(e1, simCode; cachedChange = cachedChange)) || $(expToJuliaBoolMTK(e2, simCode; cachedChange = cachedChange)))
    DAE.LUNARY(operator = DAE.NOT(__), exp = e1) =>
      :(!($(expToJuliaBoolMTK(e1, simCode; cachedChange = cachedChange))))
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
         with `Symbol(name) => index`. =#
      if cachedChange
        return :(get(_changePreValues,
                     Symbol($(string(sv.name))),
                     x[lookuptableStates[Symbol($(string(sv.name)))]]))
      end
      :(integrator.uprev[lookuptableStates[Symbol($(string(sv.name)))]])
    end
    _ => expToJuliaExpMTK(arg, simCode)
  end
end

"""
Transforms a DAE Condition into a MTK continuous condition.
"""
function transformToMTKContinousCondition(cond, simCode)
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
      :(min($(transformToMTKContinousCondition(e1, simCode)),
            $(transformToMTKContinousCondition(e2, simCode))))
    end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      :(max($(transformToMTKContinousCondition(e1, simCode)),
            $(transformToMTKContinousCondition(e2, simCode))))
    end
    #= Logical NOT: negate the inner condition =#
    DAE.LUNARY(DAE.NOT(__), e) => begin
      :(-($(transformToMTKContinousCondition(e, simCode))))
    end
    #= Strip noEvent wrapper and recurse =#
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        transformToMTKContinousCondition(innerArgs[1], simCode)
      else
        throw("noEvent with multiple arguments not supported in condition: " * string(cond))
      end
    end
    #= initial() is true only during initialization (handled by MTK InitializationProblem).
       In continuous equations it never triggers, so return constant negative. =#
    DAE.CALL(Absyn.IDENT("initial"), _, _) => begin
      :(-1)
    end
    #= General function call as boolean condition: same polarity rule. =#
    DAE.CALL(__) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)))
    end
    _ => begin
      throw("Unsupported condition expression in IF_EQUATION: " * string(cond))
    end
  end
  return res
end

"""
Transforms a DAE Condition into a MTK continuous condition equation.
"""
function transformToMTKContinousConditionEquation(cond, simCode)
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
      :(min($(transformToMTKContinousCondition(e1, simCode)),
            $(transformToMTKContinousCondition(e2, simCode))) ~ 0)
    end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      :(max($(transformToMTKContinousCondition(e1, simCode)),
            $(transformToMTKContinousCondition(e2, simCode))) ~ 0)
    end
    #= Logical NOT: negate the inner condition =#
    DAE.LUNARY(DAE.NOT(__), e) => begin
      :(-($(transformToMTKContinousCondition(e, simCode))) ~ 0)
    end
    #= Strip noEvent wrapper and recurse =#
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        transformToMTKContinousConditionEquation(innerArgs[1], simCode)
      else
        throw("noEvent with multiple arguments not supported in condition: " * string(cond))
      end
    end
    #= initial() is true only during initialization (handled by MTK InitializationProblem).
       In continuous equations it never triggers, so return constant negative equation. =#
    DAE.CALL(Absyn.IDENT("initial"), _, _) => begin
      :(-1 ~ 0)
    end
    #= General function call as boolean condition: same polarity rule as above. =#
    DAE.CALL(__) => begin
      :(0.5 - $(expToJuliaExpMTK(cond, simCode)) ~ 0)
    end
    _ => begin
      throw("Unsupported condition expression in IF_EQUATION: " * string(cond))
    end
  end
  return res
end


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
          quote
            $(Symbol(varName))
          end
        end
      end
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
    "Integer" => begin
      expToJuliaExpMTK(listHead(expLst), simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derAsSymbol)
    end
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
Transforms:
  <name>[<index>] -> <name>_index
"""
function arrayToSymbolicVariable(arrayRepr::Expr)::Expr
  _iterativePostwalk(arrayRepr) do x
    MacroTools.@capture(x, T_[index_]) || return x
    local newVar = Symbol("$(T)_$(index)")
    return newVar
  end
end

"""
Transforms:
  <name>_index -> <name>[index]
Uses direct Expr construction instead of string interpolation + Meta.parse.
"""
const pattern = r".*_[0-9]+"
function symbolicVariableToArrayRef(e::Expr)::Expr
  _iterativePostwalk(e) do x
    x isa Symbol || return x
    local sstr = String(x)
    match(pattern, sstr) === nothing && return x
    local parts = split(sstr, "_")
    return Expr(:ref, Symbol(parts[1]), parse(Int, parts[2]))
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
    DAE.CALL(_, _, attrs) => begin
      try
        attrs.ty isa DAE.T_STRING
      catch
        false
      end
    end
    DAE.IFEXP(_, t, el) => _ifexpBranchIsNonReal(t) || _ifexpBranchIsNonReal(el)
    _ => false
  end
end

#= SimCode-Exp entry (Phase 4b): codegen consumes `SimulationCode.Exp`
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
  return quote
    $(LineNumberNode(@__LINE__, "$(string(exp.path)) ENUM"))
    $(exp.index)
  end
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
    return quote
      $(LineNumberNode(@__LINE__, "SIM cref: $lookUpStr"))
      $(Symbol(htEntry[2].name))
    end
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
  local opSym = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(exp.op))
  return :( $opSym($(lhs), $(rhs)) )
end

function expToJuliaExpMTK(exp::SimulationCode.UNARY, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local operand = expToJuliaExpMTK(exp.exp, simCode;
                                    varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local opSym = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(exp.op))
  return :( $opSym($(operand)) )
end

function expToJuliaExpMTK(exp::SimulationCode.LUNARY, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local operand = expToJuliaExpMTK(exp.exp, simCode;
                                    varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  if exp.op === SimulationCode.OP_NOT
    return :( 1 - $(operand) )
  end
  local opSym = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(exp.op))
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
  local opSym = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(exp.op))
  return :( $opSym($(lhs), $(rhs)) )
end

function expToJuliaExpMTK(exp::SimulationCode.RELATION, simCode::SimulationCode.SIM_CODE;
                          varSuffix = "", varPrefix = "", derSymbol::Bool = false)::Expr
  local lhs = expToJuliaExpMTK(exp.exp1, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local rhs = expToJuliaExpMTK(exp.exp2, simCode;
                                varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local opSym = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(exp.op))
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
  local thenJL = expToJuliaExpMTK(exp.thenExp, simCode;
                                   varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
  local elseJL = expToJuliaExpMTK(exp.elseExp, simCode;
                                   varPrefix = varPrefix, varSuffix = varSuffix, derSymbol = derSymbol)
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
  return :(getproperty($innerJL, $(QuoteNode(Symbol(exp.fieldName)))))
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
      DAE.CREF(cr, DAE.T_ARRAY(ty, dims)) => begin
        lookUpStr = string(exp)
        arrName = string(exp)
        #= To make sure the variable is indexed =#
        for d in dims
          @match DAE.DIM_INTEGER(i) = d
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
            #= Variable not in hash table (may have been eliminated), using direct reference =#
            quote $(Symbol(string(varPrefix, arrName, varSuffix))) end
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
                _ => throw("Unsupported subscript: $sub")
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
                    @match DAE.ARRAY(__) = current
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
            SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
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
            SimulationCode.OCC_VARIABLE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, occ variable"))
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
            _ => begin
              @error "Unsupported varKind: $(varKind)"
              fail()
            end
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
        local thenJL = expToJuliaExpMTK(expThen, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
        local elseJL = expToJuliaExpMTK(expElse, simCode; varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
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
              local rangeExpr = expToJuliaExpMTK(rangeExp, simCode, varPrefix=varPrefix, varSuffix=varSuffix)
              push!(iterExprs, Expr(:(=), Symbol(id), rangeExpr))
            end
          end
        end
        #= Handle different reduction types =#
        @match reductionInfo.path begin
          Absyn.IDENT("array") => begin
            #= Array comprehension: [expr for i in range] =#
            Expr(:comprehension, bodyExpr, iterExprs...)
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
          _ => begin
            #= Default: treat as array comprehension =#
            Expr(:comprehension, bodyExpr, iterExprs...)
          end
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
      DAE.RSUB(exp = innerExp, fieldName = fname) => begin
        local innerJL = expToJuliaExpMTK(innerExp, simCode;
                                          varPrefix=varPrefix,
                                          varSuffix=varSuffix,
                                          derSymbol=derSymbol)
        if fname == "re"
          :(OMBackend.CodeGeneration._recordFieldRe($innerJL))
        elseif fname == "im"
          :(OMBackend.CodeGeneration._recordFieldIm($innerJL))
        else
          :(getproperty($innerJL, $(QuoteNode(Symbol(fname)))))
        end
      end
    _ =>  throw(ErrorException("$exp not yet supported"))
    end
  end
  return expr
end

"""
Extract integer array dimensions from a DAE.Type, or nothing if not an array type.
Used by the TSUB handler to detect when a tuple element is an array.
"""
function _extractTsubArrayDims(ty)
  @match ty begin
    DAE.T_ARRAY(_, dims) => begin
      local intDims = Int[]
      for d in dims
        @match d begin
          DAE.DIM_INTEGER(n) => push!(intDims, n)
          _ => return nothing
        end
      end
      return Tuple(intDims)
    end
    _ => return nothing
  end
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
      _ => expToJuliaExpMTK(sub, simCode, varPrefix=varPrefix, varSuffix=varSuffix, derSymbol=derSymbol)
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
            @match DAE.ARRAY(__) = current
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
    catch
      canEval = false
      []
    end
  else
    []
  end
  if canEval
    #= All elements are constants, return pre-computed array =#
    if dimSize >= 2
      arr = Matrix(transpose(stack(arrJL)))
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
          Matrix(transpose(stack([$(elemExprs...)])))
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
#= Numerically evaluate a scalar DAE.Exp at t0 against `valMap` (params +
   already-seeded vars). Returns nothing for anything outside the small
   arithmetic/`integer` subset, so callers fall back safely. =#
function _evalDAENumeric(@nospecialize(e), valMap::Dict{Symbol, Float64})::Union{Float64, Nothing}
  @match e begin
    DAE.RCONST(__) => Float64(e.real)
    DAE.ICONST(__) => Float64(e.integer)
    DAE.BCONST(__) => e.bool ? 1.0 : 0.0
    DAE.CREF(__) => begin
      local nm = try SimulationCode.DAE_identifierToString(e.componentRef) catch; return nothing end
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
        DAE.POW(__) => a ^ b
        _ => nothing
      end
    end
    DAE.CAST(__) => _evalDAENumeric(e.exp, valMap)
    DAE.CALL(path = Absyn.IDENT(fn)) => begin
      local argv = collect(e.expLst)
      isempty(argv) && return nothing
      local a = _evalDAENumeric(argv[1], valMap); a === nothing && return nothing
      fn == "integer" ? Float64(floor(Int, a)) :
      fn == "floor"   ? floor(a) :
      fn == "ceil"    ? ceil(a) :
      fn == "abs"     ? abs(a) :
      (fn == "float" || fn == "Real" || fn == "Integer") ? a : nothing
    end
    _ => nothing
  end
end

#= Seed valMap with initial-algorithm-assigned values (e.g. trapezoid `count`,
   `T_start`) evaluated at t0, in statement order. Without this, discrete states
   set imperatively in an `initial algorithm` (no `start` attribute) default to
   0.0 when evaluating if-equation initial branches, picking the wrong branch. =#
function _seedInitialAlgValues!(valMap::Dict{Symbol, Float64}, simCode)
  for ia in simCode.initialAlgorithms
    for stmt in ia.daeStatements
      @match stmt begin
        DAE.STMT_ASSIGN(__) => begin
          local nm = try SimulationCode.DAE_identifierToString(stmt.exp1) catch; nothing end
          nm === nothing && continue
          local v = _evalDAENumeric(stmt.exp, valMap)
          v !== nothing && (valMap[Symbol(nm)] = v)
        end
        _ => nothing
      end
    end
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

#= Symbol -> value map at t0: parameter values plus state/algebraic start
   attributes (default 0.0), overridden by early init-algorithm results. =#
function _buildT0ValueMap(simCode)::Dict{Symbol, Float64}
  local valMap = Dict{Symbol, Float64}()
  local ht = simCode.stringToSimVarHT
  for (key, (_, sv)) in ht
    local sym = Symbol(key)
    if sv.varKind isa SimulationCode.PARAMETER
      local pval = try
        local raw = evalSimCodeParameter(sv, simCode)
        if raw isa Expr
          #= evalDAE_Expression wraps its result in a :block Expr; evaluate
             once to unwrap before the numeric coercion. =#
          raw = Base.invokelatest(eval, raw)
        end
        Float64(raw)
      catch
        try
          @match SOME(attr) = sv.attributes
          @match SOME(startExp) = attr.start
          Float64(evalDAEConstant(startExp, simCode))
        catch
          nothing
        end
      end
      if pval !== nothing
        valMap[sym] = pval
      end
    elseif SimulationCode.isStateOrAlgebraic(sv)
      local sval = 0.0
      try
        @match SOME(attr) = sv.attributes
        @match SOME(startExp) = attr.start
        sval = Float64(evalDAEConstant(startExp, simCode))
      catch
      end
      valMap[sym] = sval
    end
  end
  _seedInitialAlgValues!(valMap, simCode)
  return valMap
end

#= Evaluate a causalized branch RHS at the t0 value map. Returns nothing when
   the expression is not statically evaluable. =#
function evalCausalRHSAtT0(rhsExpr, valMap::Dict{Symbol, Float64})::Union{Float64, Nothing}
  if ccall(:jl_generating_output, Cint, ()) != 0
    return nothing
  end
  try
    local numExpr = _substituteExprValues(rhsExpr, valMap)
    local result = eval(numExpr)
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
  catch
    return nothing
  end
end

function evalInitialCondition(mtkCond, simCode = nothing; closedBoundary::Bool = false)
  #= Skip during precompile output: `eval(...)` below would mutate this
     closed module's bindings and Julia rejects that. The runtime path is
     unaffected. Fallback `true` matches the existing catch arm. =#
  if ccall(:jl_generating_output, Cint, ()) != 0
    return true
  end
  #= Evaluate zero-crossing function at t=0 to determine initial condition.
     Works at the Expr level: walks the mtkCond Expr tree, substitutes all
     variable/parameter references with numeric values, then evals the result.
     Returns true when the zero-crossing function is non-negative at t=0
     (condition FALSE), false when negative (condition TRUE).
     The caller inverts: ifCond = !(evalInitialCondition(...)). =#
  try
    if simCode === nothing
      #= No simCode: fall back to old behavior =#
      local mtkCondE = Base.invokelatest(eval, mtkCond)
      local lhs = mtkCondE.lhs
      local tSym = ModelingToolkit.t_nounits
      local v = Base.invokelatest(substitute, lhs, Dict(tSym => 0.0))
      local numV = try Float64(v) catch; nothing end
      if numV !== nothing
        return closedBoundary ? numV > 0.0 : numV >= 0.0
      end
      return (v == 0) != false
    end
    local valMap = _buildT0ValueMap(simCode)
    #= Extract LHS from the mtkCond Expr (form: :(lhs ~ 0)) =#
    local lhsExpr = _extractZeroCrossingLHS(mtkCond)
    #= Substitute all variable references with numeric values =#
    local numExpr = _substituteExprValues(lhsExpr, valMap)
    local result = eval(numExpr)
    local numResult = if result isa Number
      Float64(result)
    else
      local unwrapped = Base.invokelatest(SymbolicUtils.unwrap, result)
      if unwrapped isa Number
        Float64(unwrapped)
      else
        local valued = try Base.invokelatest(Symbolics.value, result) catch; result end
        Float64(valued isa Number ? valued : 0.0)
      end
    end
    #= The zero-crossing function is negative when the condition is TRUE,
       positive when FALSE. Return true when condition is FALSE (positive),
       because the caller inverts: ifCond = !(evalInitialCondition(...)).
       At zc == 0 the original operator decides (closedBoundary). =#
    return closedBoundary ? numResult > 0.0 : numResult >= 0.0
  catch e
    @warn "evalInitialCondition: failed to evaluate, defaulting to true" exception=(e, catch_backtrace())
    return true
  end
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
    _ => throw("Cast $ty: for exp: $exp not yet supported in codegen!")
  end
  return expr
end

# TODO: unify cref resolution into one function consulting both the SimCode
# (state / numeric-param lookup tables) and the module-level bindings (String
# parameters, data structures), so callers need not special-case the latter.
function getIdxForLookupMTK(x::Union{DAE.ComponentRef, DAE.CREF}, simCode)
  local crefAsStr = string(x)
  if crefAsStr == "time"
    return :t
  end
  @match _, simVar = simCode.stringToSimVarHT[crefAsStr]
  if !(SimulationCode.isParameter(simVar))
    Expr(:call, getindex, :x, Expr(:call, :getindex, :lookuptableStates, :(Symbol($(string(x))))))
  else
    Expr(:call, getindex, :p, Expr(:call, :getindex, :lookuptableParams, :(Symbol($(string(x))))))
  end
end


#= Helpers that depend on the MTK lowering (expToJuliaExpMTK / ModelingToolkit.ifelse / Symbolics) =#

_isSimCodeFunctionPath(path::Absyn.Path, simCode)::Bool = _isSimCodeFunctionName(string(path), simCode)

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

function evalDAE_Expression(expr, simCode)::Expr
  local shouldEval = Ref(true)
  #= Replaces all known bound parameters in the DAE expression. This must be
     recursive: parameter aliases such as `actualGlobalSeed = globalSeed_seed`
     otherwise leave a bare `globalSeed_seed` in the generated Julia expression
     even after `globalSeed_seed` itself has been solved from an initial equation. =#
  local daeExp = _substituteBoundParameters(expr, simCode; shouldEval=shouldEval)
  local jlExpr = expToJuliaExpMTK(daeExp, simCode)
  local evaluatedJLExpr = if shouldEval[]
    try
      eval(jlExpr)
    catch
      jlExpr
    end
  else
    jlExpr
  end
  return quote $(evaluatedJLExpr) end
end

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
    local ieqLhs, ieqRhs = equationSides(ieq)
    Util.traverseExpBottomUp(ieqLhs, findFree, 0)
    Util.traverseExpBottomUp(ieqRhs, findFree, 0)
    unique!(freeParams)
    if length(freeParams) != 1
      continue
    end
    freeName = freeParams[1]
    #= Get initial guess from start attribute =#
    local (_, freeSV) = ht[freeName]
    local guess = 0.1
    @match freeSV.attributes begin
      SOME(attr) => begin
        @match attr.start begin
          SOME(startExp) => begin
            try
              guess = Float64(evalDAEConstant(startExp, simCode))
            catch
            end
          end
          _ => nothing
        end
      end
      _ => nothing
    end
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
        x -= fx / dfx
      end
    catch err
      @warn "[SIMCODE: solveParametricInitialEquations] residual call threw, skipping" freeName err
      newtonOk = false
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
  local refs::OrderedSet{String} = OrderedSet{String}()
  try
    SimulationCode.collectCrefNames!(refs, condition)
  catch
    return false
  end
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
                               subIdentifier::Int = 1)
  local branch = branches[target]
  if branch.targets == -1
    return :($(first(deCausalize(branch.residualEquations[resEqIdx], simCode))))
  end
  #= When every branch condition is discrete/parameter (e.g. an event-held Boolean),
     gate directly on the condition value: its own update event localises the step,
     so no `ifCond` relay parameter or continuous callback is needed (and the relay's
     start-attribute initial value, which can pick the wrong branch, is avoided). =#
  local cond = if _allBranchConditionsDiscrete(branches, simCode)
    :( $(expToJuliaExpMTK(SimulationCode.toDAEExp(branch.condition), simCode)) > 0.5 )
  else
    #= ifCond variables are discrete parameters (not ODE unknowns), so the solver
       never perturbs them during Jacobian computation. Exact comparison is safe. =#
    :( $(Symbol(string("ifCond", identifier, subIdentifier))) == 1 )
  end
  local rhs = first(deCausalize(branch.residualEquations[resEqIdx], simCode))
  quote
    ModelingToolkit.ifelse($(cond),
                           $(rhs),
                           $(generateIfExpressions(branches,
                                                   branches[target].targets,
                                                   resEqIdx,
                                                   identifier,
                                                   simCode;
                                                   subIdentifier = subIdentifier + 1)))
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
      throw("Unsupported equation:" * string(eq))
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
    Expr(:., :OMRuntimeExternalC, QuoteNode(runtimeName))
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
