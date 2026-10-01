#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
=#

#=
# Author: John Tinnerholm (johti17)
=#
module CodeGenerationUtil

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


"""
Return string containing the OSMC copyright stuff.
"""
function copyrightString()

  strOut = string("#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
=#")
  return strOut
end


"""
johti17
    Transform a condition into a zero crossing function.
    For instance y < 10 -> y - 10
TODO:
    Assumes a Real-Expression.
    Also assume that relation expressions are written as <= That is we go from positive
    to negative..
    Fix this.
"""
#= integer(X) is a step (often CAST to real); a continuous zero-crossing on it never
   triggers, so `when integer(X) ⋛ Y` freezes (e.g. Modelica.Blocks.Sources.Pulse).
   Unwrap an optional CAST and return the integer() argument, else nothing. =#
function _zcIntArg(@nospecialize(e::DAE.Exp))
  local inner = @match e begin
    DAE.CAST(exp = c) => c
    _ => e
  end
  @match inner begin
    DAE.CALL(path = Absyn.IDENT("integer"), expLst = args) => listHead(args)
    _ => nothing
  end
end

#= Rewrite a relation with an integer() operand to a smooth zero-crossing on X:
   floor(X) ⋛ Y ⟺ X ⋛ Y±1 for integer-valued Y. Returns the zero-crossing or nothing. =#
function _zcRewriteIntegerRel(@nospecialize(cond::DAE.Exp))
  @match cond begin
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      local x1 = _zcIntArg(e1)
      local x2 = _zcIntArg(e2)
      (x1 === nothing && x2 === nothing) && return nothing
      local one = DAE.RCONST(1.0)
      local S = DAE.SUB(DAE.T_REAL_DEFAULT)
      local A = DAE.ADD(DAE.T_REAL_DEFAULT)
      if x1 !== nothing
        @match op begin
          DAE.GREATER(__)   => DAE.BINARY(DAE.BINARY(e2, A, one), S, x1)
          DAE.GREATEREQ(__) => DAE.BINARY(e2, S, x1)
          DAE.LESS(__)      => DAE.BINARY(x1, S, e2)
          DAE.LESSEQ(__)    => DAE.BINARY(x1, S, DAE.BINARY(e2, A, one))
          _ => nothing
        end
      else
        @match op begin
          DAE.LESS(__)      => DAE.BINARY(DAE.BINARY(e1, A, one), S, x2)
          DAE.LESSEQ(__)    => DAE.BINARY(e1, S, x2)
          DAE.GREATER(__)   => DAE.BINARY(x2, S, e1)
          DAE.GREATEREQ(__) => DAE.BINARY(x2, S, DAE.BINARY(e1, A, one))
          _ => nothing
        end
      end
    end
    _ => nothing
  end
end

function transformToZeroCrossingCondition(@nospecialize(conditonalExpression::DAE.Exp))::DAE.Exp
  local _intRW = _zcRewriteIntegerRel(conditonalExpression)
  _intRW !== nothing && return _intRW
  #= Build a Real-valued zero-crossing function f such that the Modelica
     condition becomes true exactly when f goes from positive to negative
     (the SciML `affect!` direction). For `<` / `<=` we use `e1 - e2` so the
     condition becomes true when e1 falls below e2. For `>` / `>=` we swap to
     `e2 - e1` so the same affect! direction matches an upcrossing of e1
     above e2. This lets the caller pass `affect_neg! = nothing` and avoid
     double-firing when e1 oscillates around e2 (classic bouncing-ball Zeno
     pattern). =#
  res = @match conditonalExpression begin
    DAE.RELATION(exp1 = e1, operator = DAE.LESS(__), exp2 = e2) => begin
      DAE.BINARY(e1, DAE.SUB(DAE.T_REAL_DEFAULT), e2)
    end
    DAE.RELATION(exp1 = e1, operator = DAE.LESSEQ(__), exp2 = e2) => begin
      DAE.BINARY(e1, DAE.SUB(DAE.T_REAL_DEFAULT), e2)
    end
    DAE.RELATION(exp1 = e1, operator = DAE.GREATER(__), exp2 = e2) => begin
      DAE.BINARY(e2, DAE.SUB(DAE.T_REAL_DEFAULT), e1)
    end
    DAE.RELATION(exp1 = e1, operator = DAE.GREATEREQ(__), exp2 = e2) => begin
      DAE.BINARY(e2, DAE.SUB(DAE.T_REAL_DEFAULT), e1)
    end
    #= `change(rel)` fires when the relation flips: its zero-crossing is the
       relation's own zero-crossing function. Unwrap and recurse. =#
    DAE.CALL(path = Absyn.IDENT("change"), expLst = args) => begin
      transformToZeroCrossingCondition(listHead(args))
    end
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      DAE.BINARY(e1, DAE.SUB(DAE.T_REAL_DEFAULT), e2)
    end
    DAE.LUNARY(operator = op, exp = e1)  => begin
      DAE.UNARY(DAE.SUB(DAE.T_REAL_DEFAULT), e1)
    end
    DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      DAE.BINARY(e1, DAE.SUB(DAE.T_REAL_DEFAULT), e2)
    end
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      DAE.BINARY(e1, DAE.SUB(DAE.T_REAL_DEFAULT), e2)
    end
    _ => begin
      conditonalExpression
    end
  end
  return res
end



function DAE_OP_toJuliaOperator(@nospecialize(op::DAE.Operator))
    return @match op begin
      DAE.ADD() => :+
      DAE.SUB() => :-
      DAE.MUL() => :*
      DAE.DIV() => :/
      DAE.POW() => :^
      DAE.UMINUS() =>  :-
      DAE.UMINUS_ARR() => :-
      DAE.ADD_ARR() => :+
      DAE.SUB_ARR() => :-
      #= Element-wise, broadcast (the same on scalars): `*` and `/` were a matrix
         product and a right division on matrices (function bodies), and an
         array with a scalar or a scalar with an array had no method. =#
      DAE.MUL_ARR() => Symbol(".*")
      DAE.DIV_ARR() => Symbol("./")
      DAE.MUL_ARRAY_SCALAR() => :*
      DAE.ADD_ARRAY_SCALAR() => Symbol(".+")
      DAE.SUB_SCALAR_ARRAY() => Symbol(".-")
      #= Julia's `*` has no vector-vector method; also right for scalars. =#
      DAE.MUL_SCALAR_PRODUCT() => :(OMBackend.CodeGeneration.vectorDot)
      DAE.MUL_MATRIX_PRODUCT() => :*
      DAE.DIV_ARRAY_SCALAR() => :/
      DAE.DIV_SCALAR_ARRAY() => Symbol("./")
      DAE.POW_ARRAY_SCALAR() => Symbol(".^")
      DAE.POW_SCALAR_ARRAY() => Symbol(".^")
      DAE.POW_ARR() => :^
      DAE.POW_ARR2() => Symbol(".^")
      DAE.AND() => :(&)
      DAE.OR() => :(||)
      DAE.NOT() => :(!)
      DAE.LESS() => :(<)
      DAE.LESSEQ() => :(<=)
      DAE.GREATER() => :(>)
      DAE.GREATEREQ() => :(>=)
      DAE.EQUAL() => :(==)
      DAE.NEQUAL() => :(!=)
      DAE.USERDEFINED() => OMBackend.unsupported("operator", op)
      _ => OMBackend.unsupported("operator", op)
    end
end

#= Direct SimCode OpKind -> Julia operator Symbol, equivalent to
   DAE_OP_toJuliaOperator(toDAEOperator(k)) but without the throwaway
   DAE.Operator allocation per operator node in codegen. =#
function opKindToJuliaOperator(k::SimulationCode.OpKind)::Union{Symbol, Expr}
  if k === SimulationCode.OP_ADD;          :+
  elseif k === SimulationCode.OP_SUB;      :-
  elseif k === SimulationCode.OP_MUL;      :*
  elseif k === SimulationCode.OP_DIV;      :/
  elseif k === SimulationCode.OP_POW;      :^
  elseif k === SimulationCode.OP_UMINUS;   :-
  elseif k === SimulationCode.OP_AND;      :(&)
  elseif k === SimulationCode.OP_OR;       :(||)
  elseif k === SimulationCode.OP_NOT;      :(!)
  elseif k === SimulationCode.OP_LESS;     :(<)
  elseif k === SimulationCode.OP_LESSEQ;   :(<=)
  elseif k === SimulationCode.OP_GREATER;  :(>)
  elseif k === SimulationCode.OP_GREATEREQ; :(>=)
  elseif k === SimulationCode.OP_EQUAL;    :(==)
  elseif k === SimulationCode.OP_NEQUAL;   :(!=)
  elseif k === SimulationCode.OP_DOT;      :(OMBackend.CodeGeneration.vectorDot)
  else error("opKindToJuliaOperator: unhandled OpKind $k")
  end
end


"
  TODO: Keeping it simple for now, we assume we only have one argument in the call
  We handle derivatives separately
"
function DAECallExpressionToJuliaCallExpression(pathStr::String, expLst::List, simCode, ht; varPrefix=varPrefix)::Expr
  @match pathStr begin
    "der" => begin
      local arg = listHead(expLst)
      @match arg begin
        DAE.ARRAY(_, _, array) => begin
          local elemExprs = map(array) do e
            DAECallExpressionToJuliaCallExpression("der", Cons(e, MetaModelica.nil), simCode, ht; varPrefix=varPrefix)
          end
          Expr(:vect, elemExprs...)
        end
        #= der(literal) ≡ 0. See companion arm in DAECallExpressionToMTKCallExpression. =#
        DAE.RCONST(_) => quote 0.0 end
        DAE.ICONST(_) => quote 0 end
        DAE.BCONST(_) => quote false end
        _ => begin
          varName = SimulationCode.DAE_identifierToString(arg)
          (index, _) = ht[varName]
          quote
            dx[$(index)] #= der($varName) =#
          end
        end
      end
    end
    "pre" => begin
      local arg = listHead(expLst)
      @match arg begin
        DAE.ARRAY(_, _, array) => begin
          local elemExprs = map(array) do e
            DAECallExpressionToJuliaCallExpression("pre", Cons(e, MetaModelica.nil), simCode, ht; varPrefix=varPrefix)
          end
          Expr(:vect, elemExprs...)
        end
        #= pre(literal) ≡ literal. =#
        DAE.RCONST(r) => quote $r end
        DAE.ICONST(i) => quote $i end
        DAE.BCONST(b) => quote $b end
        _ => begin
          varName = SimulationCode.DAE_identifierToString(arg)
          (index, _) = ht[varName]
          indexForVar = ht[varName][1]
          quote
            (integrator.u[$(indexForVar)])
          end
        end
      end
    end
    #= See sibling arm in DAECallExpressionToMTKCallExpression — Integer(enum)
       collapses to identity since enums are already integer-indexed in codegen. =#
    "Integer" => begin
      OMBackend.CodeGeneration.expToJuliaExp(listHead(expLst), simCode, varPrefix=varPrefix)
    end
    #= Values read from the integrator are Float64s: String takes their types
       from the expressions. =#
    "String" => AlgorithmicCodeGeneration.modelicaStringCall(
      collect(expLst), x -> OMBackend.CodeGeneration.expToJuliaExp(x, simCode, varPrefix=varPrefix))
    _  =>  begin
      argPart = tuple(map((x) -> OMBackend.CodeGeneration.expToJuliaExp(x, simCode, varPrefix=varPrefix), expLst)...)
      #= Mirror DAECallExpressionToMTKCallExpression: route Modelica built-ins
         (sample, noEvent, smooth, ...) to their qualified
         OMBackend.CodeGeneration.AlgorithmicCodeGeneration stubs. Without
         this, sample() inside the body of a when-statement or any non-MTK
         code path emits a bare `sample(...)` Julia call which the per-model
         module cannot resolve. =#
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
  Get the name of the active model.
  The default is the original model.
"""
function getActiveModel(simCode)
  local activeModelName = simCode.activeModel
  for sm in simCode.subModels
    if sm.name == activeModelName
      return sm
    end
  end
  return activeModelName
end

function _isSimCodeFunctionName(name::AbstractString, simCode)::Bool
  local canonical = OMBackend.canonicalName(name)
  for f in simCode.functions
    if f.name == canonical
      return true
    end
  end
  return false
end


"""
  Removes all comments from a given exp
"""
function stripComments(ex::Expr)::Expr
  return Base.remove_linenums!(ex)
end

#= Iterative equivalent of `MacroTools.postwalk`. Visits each subexpression
   bottom-up, replacing each node with the result of `f(rebuilt_node)`.
   Used in place of the recursive walker for huge generated function-body
   ASTs (e.g. wrappers for Modelica.Utilities.Strings.scanToken family),
   where the recursive form blows past Julia's runtime stack guard. =#
function _iterativePostwalk(f, root)
  isa(root, Expr) || return f(root)
  local todo = Any[(root, false)]
  local newMap = Base.IdDict{Any,Any}()
  #= Dedup by object identity: generated ASTs are DAGs with shared sub-Expr
     nodes, so expanding every reference re-walks shared subtrees and blows up
     exponentially. Each unique node is expanded and rebuilt once; `f` is a pure
     function of the node, so all references resolve to the same cached result. =#
  local expanded = Base.IdSet{Any}()
  while !isempty(todo)
    local (node, processedChildren) = pop!(todo)
    if !(node isa Expr)
      haskey(newMap, node) || (newMap[node] = f(node))
      continue
    end
    if !processedChildren
      node in expanded && continue
      push!(expanded, node)
      push!(todo, (node, true))
      for a in node.args
        push!(todo, (a, false))
      end
    else
      #= Lazy rebuild: only allocate a new args vector + Expr when a child
         actually changed; otherwise pass the original node to `f`. `f` is a
         pure function of structure, so a structurally identical node yields
         the same result -> emitted code is unchanged. =#
      local changed = false
      for a in node.args
        if get(newMap, a, a) !== a
          changed = true
          break
        end
      end
      local rebuilt = changed ? Expr(node.head, Any[get(newMap, a, a) for a in node.args]...) : node
      newMap[node] = f(rebuilt)
    end
  end
  return get(newMap, root, root)
end

"""
 Removes all redundant blocks from a generated expression
"""
function stripBeginBlocks(e)::Expr
  _iterativePostwalk(MacroTools.unblock, e)
end

"""
Convert a MetaModelica list of DAE subscripts directly to Julia Expr form
without going through string conversion + Meta.parse.
Returns an integer for single constant subscripts, or a tuple Expr for multiple.
"""
#= Build an array indexing Expr from a symbol and subscript expression.
   subscriptExpr is the return value of subscriptsToExpr:
   single value (int/Expr) or Expr(:tuple, ...) for multi-dimensional. =#
function makeRefExpr(sym, subscriptExpr)
  if subscriptExpr isa Expr && subscriptExpr.head == :tuple
    return Expr(:ref, sym, subscriptExpr.args...)
  else
    return Expr(:ref, sym, subscriptExpr)
  end
end

function isCycleInSCCs(sccs)
  for sc in sccs
    if length(sc) > 1
      return true
    end
  end
  return false
end

"""
    resolveAliasedCref(fullName, simCode, hashTable; varPrefix, varSuffix)

Fallback for code generation when a variable was eliminated by alias elimination
but a reference to it survived in the equation expressions (missed by substitution).
Returns `(true, expr)` if the variable was found in `simCode.aliasMap`, or `(false, nothing)`.
"""
function resolveAliasedCref(fullName::String, simCode, hashTable; varPrefix="", varSuffix="")
  for entry in simCode.aliasMap
    if entry.eliminatedName == fullName
      local repEntry = get(hashTable, entry.representativeName, nothing)
      if repEntry !== nothing
        local sym = Symbol(varPrefix, repEntry[2].name, varSuffix)
        if entry.negated
          return (true, :(-$(sym)))
        else
          return (true, quote $(sym) end)
        end
      end
    end
  end
  return (false, nothing)
end

"""
  Helper used by the lowered DAE.RSUB(`re`) form. Falls back through:
    Base.Complex — `.re`
    NamedTuple / structs with `:re` — `getproperty`
    Symbolics.Num wrapping a Complex term — `real(num)`
  Annotated as `Base.Complex` because the module rebinds the bare name
  `Complex` to a Modelica function wrapper via
  `createModelicaFunctionWrapper(:Complex, …)`, which would otherwise
  shadow the type and turn this method definition into "invalid type
  for argument".
"""
@inline _recordFieldRe(x::Base.Complex) = x.re
@inline _recordFieldRe(x::Tuple) = x[1]
@inline _recordFieldRe(x) = hasproperty(x, :re) ? getproperty(x, :re) : real(x)

#= A record's field `name`, the `ix`th: by position in a tuple (a function returning a
   record returns its fields in order: the MSL Media `setState_psX(...).T`), by name
   otherwise (a record constructor's NamedTuple). =#
@inline _recordField(x::Tuple, name::Symbol, ix::Integer) = ix >= 1 ? x[ix] : getproperty(x, name)
@inline _recordField(x, name::Symbol, ::Integer) = getproperty(x, name)

"""
  Companion to `_recordFieldRe` for the imaginary part (DAE.RSUB(`im`)).
  Same `Base.Complex` reason: avoid the module-scope shadow of `Complex`.
"""
@inline _recordFieldIm(x::Base.Complex) = x.im
@inline _recordFieldIm(x::Tuple) = x[2]
@inline _recordFieldIm(x) = hasproperty(x, :im) ? getproperty(x, :im) : imag(x)


"""
  Given the variable idx and simCode statically decide if this variable is
  the LHS target of any when-equation `ASSIGN` / `REINIT` statement
  (including elsewhen branches).

  Such variables receive their state updates from discrete callbacks at
  event time, so they should be routed into the discrete bucket rather than
  the continuous residual set during MTK code generation. This prevents the
  continuous integrator from synthesising a competing dynamic for a
  variable that is only meaningful at event boundaries.
"""
function involvedInEvent(idx, simCode)
  local targetName = nothing
  for (k, (i, sv)) in simCode.stringToSimVarHT
    if i == idx
      targetName = k
      break
    end
  end
  targetName === nothing && return false

  for weq in simCode.whenEquations
    local wEqInner = weq.whenEquation
    if _whenStmtLstTargets(wEqInner.whenStmtLst, targetName)
      return true
    end
    #= elsewhenPart carries two storage shapes: BDAE wraps with
       SOME(WHEN_EQUATION(WHEN_STMTS)), SimCode stores the bare WHEN_STMTS. =#
    local ew = wEqInner.elsewhenPart
    while ew !== nothing
      local nested = ew isa SOME ? ew.data : ew
      local inner = hasproperty(nested, :whenEquation) ? nested.whenEquation : nested
      if _whenStmtLstTargets(inner.whenStmtLst, targetName)
        return true
      end
      ew = inner.elsewhenPart
    end
  end
  return false
end

"""
  Walk a `List{BDAE.WhenOperator}` and return true if any `ASSIGN`'s
  `left` cref-name or any `REINIT`'s `stateVar` cref-name matches
  `targetName` (the simCode `stringToSimVarHT` key).
"""
function _whenStmtLstTargets(stmtLst, targetName::String)::Bool
  for stmt in stmtLst
    local lhsName = if stmt isa BDAE.ASSIGN || stmt isa SimulationCode.ASSIGN
      SimulationCode.string(_asDAE(stmt.left))
    elseif stmt isa BDAE.REINIT || stmt isa SimulationCode.REINIT
      SimulationCode.string(stmt.stateVar)
    else
      ""
    end
    if lhsName == targetName
      return true
    end
  end
  return false
end

"""
  This function evaluates a single DAE-constant:{Bool, Integer, Real, String}.
  If the argument  to this function is not a constant it throws an error.
"""
function evalDAEConstant(daeConstant::DAE.Exp, simCode)
  @match daeConstant begin
    DAE.BCONST(bool) => bool
    DAE.ICONST(int) => int
    DAE.RCONST(real) => real
    DAE.SCONST(tmpStr) => tmpStr
    #= Try to evaluate the expression: operators and calls of constants (MSL
       Fluid's tanks: a parameter bound to max(1.0, 1e-15)). =#
    DAE.BINARY(__) || DAE.LBINARY(__) || DAE.UNARY(__) || DAE.CALL(__) => begin
      OMBackend.CodeGeneration.evalDAE_Expression(daeConstant, simCode)
    end
    _ => OMBackend.unsupported("not a constant", daeConstant)
  end
end

function evalDAEConstant(daeConstant::DAE.Exp)
  @match daeConstant begin
    DAE.BCONST(bool) => bool
    DAE.ICONST(int) => int
    DAE.RCONST(real) => real
    DAE.SCONST(tmpStr) => tmpStr
    #= Try to evaluate the expression =#
    _ => OMBackend.unsupported("not a constant", daeConstant)
  end
end

#= SIM-native: literal fast-path returns the value without toDAEExp; computed
   constants (BINARY/LBINARY) still route through the DAE constant evaluator. =#
function evalDAEConstant(c::SimulationCode.Exp, simCode)
  @match c begin
    SimulationCode.BCONST(b) => b
    SimulationCode.ICONST(i) => i
    SimulationCode.RCONST(r) => r
    SimulationCode.SCONST(s) => s
    SimulationCode.BINARY(__) || SimulationCode.LBINARY(__) || SimulationCode.UNARY(__) || SimulationCode.CALL(__) =>
      OMBackend.CodeGeneration.evalDAE_Expression(SimulationCode.toDAEExp(c), simCode)
    _ => OMBackend.unsupported("not a constant", SimulationCode.toDAEExp(c))
  end
end
function evalDAEConstant(c::SimulationCode.Exp)
  @match c begin
    SimulationCode.BCONST(b) => b
    SimulationCode.ICONST(i) => i
    SimulationCode.RCONST(r) => r
    SimulationCode.SCONST(s) => s
    _ => OMBackend.unsupported("not a constant", SimulationCode.toDAEExp(c))
  end
end


"""
  Evaluates a simulation code parameter's binding (evalDAEConstant).
  UnsupportedLowering when `v` is not a bound parameter or the binding not a constant.
"""
function evalSimCodeParameter(v::V, simCode) where V
  return @match v.varKind begin
    SimulationCode.PARAMETER(SOME(bindExp)) => evalDAEConstant(bindExp, simCode)
    _ => OMBackend.unsupported("a constant value of a variable other than a bound parameter", v.name)
  end
end

"""
 Evalutates the components in a DAE expression (Currently if the components are parameters)
"""
function _substituteBoundParameters(exp, simCode;
                                    skipNames::OrderedSet{String}=OrderedSet{String}(),
                                    shouldEval::Base.RefValue{Bool}=Ref(true),
                                    seen::OrderedSet{String}=OrderedSet{String}())
  function replaceParameterVariable(exp, ht)
    if Util.isCref(exp)
      local key = string(exp)
      if key in skipNames
        return (exp, true, ht)
      end
      local entry = get(simCode.stringToSimVarHT, key, nothing)
      if entry === nothing
        shouldEval[] = false
        return (exp, true, ht)
      end
      local simVar = last(entry)
      if SimulationCode.isStateOrAlgebraic(simVar)
        shouldEval[] = false
      else
        local bindExp = @match simVar.varKind begin
          SimulationCode.PARAMETER(SOME(be)) => SimulationCode.toDAEExp(be)
          SimulationCode.ARRAY_PARAMETER(_, SOME(be)) => SimulationCode.toDAEExp(be)
          _ => nothing
        end
        if bindExp !== nothing
          if key in seen
            shouldEval[] = false
            return (exp, true, ht)
          end
          push!(seen, key)
          local resolved = _substituteBoundParameters(bindExp, simCode;
                                                      skipNames=skipNames,
                                                      shouldEval=shouldEval,
                                                      seen=seen)
          delete!(seen, key)
          return (resolved, true, ht)
        else
          #= Parameter without binding (fixed=false, determined by an initial
             equation not solved yet). Leave the CREF in place and avoid eval. =#
          shouldEval[] = false
        end
      end
    end
    (exp, true, ht)
  end
  return first(Util.traverseExpBottomUp(exp, replaceParameterVariable, 0))
end

"""
    equationSides(eq) -> Tuple{DAE.Exp, DAE.Exp}

Return (lhs, rhs) for any BDAE equation shape.

- `BDAE.EQUATION`           has `.lhs` / `.rhs`
- `BDAE.COMPLEX_EQUATION`   has `.left` / `.right`   (record-to-record equality)
- `BDAE.ARRAY_EQUATION`     has `.left` / `.right`   (pre-scalarized array equality)
- `BDAE.RESIDUAL_EQUATION`  has `.exp`, already in LHS−RHS=0 form → return `(eq.exp, RCONST(0.0))`

Any other shape throws; the caller should have screened those out.
"""
function equationSides(eq)::Tuple{DAE.Exp, DAE.Exp}
  if eq isa BDAE.EQUATION || eq isa SimulationCode.EQUATION
    return (SimulationCode.toDAEExp(eq.lhs), SimulationCode.toDAEExp(eq.rhs))
  elseif eq isa BDAE.COMPLEX_EQUATION
    return (eq.left, eq.right)
  elseif eq isa BDAE.ARRAY_EQUATION || eq isa SimulationCode.ARRAY_EQUATION
    return (SimulationCode.toDAEExp(eq.left), SimulationCode.toDAEExp(eq.right))
  elseif eq isa BDAE.RESIDUAL_EQUATION || eq isa SimulationCode.RESIDUAL_EQUATION
    return (SimulationCode.toDAEExp(eq.exp), DAE.RCONST(0.0))
  end
  error("equationSides: no (lhs, rhs) for $(typeof(eq)); " *
        "caller should filter to EQUATION / COMPLEX_EQUATION / ARRAY_EQUATION / RESIDUAL_EQUATION.")
end

#= The equations `equationSides` takes. =#
hasEquationSides(eq)::Bool =
  eq isa Union{BDAE.EQUATION, SimulationCode.EQUATION, BDAE.COMPLEX_EQUATION, BDAE.ARRAY_EQUATION,
               SimulationCode.ARRAY_EQUATION, BDAE.RESIDUAL_EQUATION, SimulationCode.RESIDUAL_EQUATION}

"""
    isParametricOnlyEquation(eq, simCode) -> Bool

Check if an initial equation only involves parameters (no state/algebraic variables).
Such equations determine parameter values and should not be treated as initial conditions.
"""
function isParametricOnlyEquation(eq, simCode::SimulationCode.SimCode)::Bool
  ht = simCode.stringToSimVarHT
  hasStateOrAlg = Ref(false)
  function checker(exp, acc)
    if Util.isCref(exp)
      local key = string(exp)
      local entry = get(ht, key, nothing)
      if entry !== nothing
        local sv = last(entry)
        if SimulationCode.isStateOrAlgebraic(sv) || SimulationCode.isDiscrete(sv)
          hasStateOrAlg[] = true
        end
      end
    end
    (exp, true, acc)
  end
  lhs, rhs = equationSides(eq)
  Util.traverseExpBottomUp(lhs, checker, 0)
  if !hasStateOrAlg[]
    Util.traverseExpBottomUp(rhs, checker, 0)
  end
  return !hasStateOrAlg[]
end

function _substituteExprValues(expr, valMap::Dict{Symbol, Float64})
  if expr isa Number
    return Float64(expr)
  end
  if expr isa Symbol
    if expr == :t
      return 0.0
    end
    if haskey(valMap, expr)
      return valMap[expr]
    end
    #= Defensive fallback: a missing symbol would hit `eval` at module
       scope and raise `UndefVarError`. Returning 0.0 (which is `false`
       for boolean thresholds) keeps initial-condition evaluation
       robust. The outer try/catch in `evalInitialCondition` already
       defaults to `true` on any remaining failure, so this is defense
       in depth for fold-promoted parameters whose valMap entry was
       skipped. =#
    return 0.0
  end
  if !(expr isa Expr)
    return expr
  end
  if expr.head == :call
    local fname = expr.args[1]
    #= Pattern: varName(t) -> substitute with value from valMap =#
    if fname isa Symbol && length(expr.args) == 2 && expr.args[2] == :t
      if haskey(valMap, fname)
        return valMap[fname]
      end
      #= `varName(t)` where `varName` is not in valMap: recursion below
         would rebuild `Expr(:call, :varName, 0.0)` and hit
         `UndefVarError` on final eval. Defensive fallback to 0.0, same
         policy as the bare-symbol branch above. =#
      return 0.0
    end
    #= Recurse into call arguments =#
    local newArgs = Any[fname]
    for i in 2:length(expr.args)
      push!(newArgs, _substituteExprValues(expr.args[i], valMap))
    end
    return Expr(:call, newArgs...)
  end
  #= Broadcast call `f.(args)`: the callee position is code, not a variable. =#
  if expr.head == :. && length(expr.args) == 2 && expr.args[2] isa Expr && expr.args[2].head == :tuple
    return Expr(:., expr.args[1], _substituteExprValues(expr.args[2], valMap))
  end
  #= Qualified name `A.b`: not a variable reference. =#
  if expr.head == :. && length(expr.args) == 2 && expr.args[2] isa QuoteNode
    return expr
  end
  #= Generic recursion for other Expr types =#
  local newExpr = Expr(expr.head)
  for arg in expr.args
    push!(newExpr.args, _substituteExprValues(arg, valMap))
  end
  return newExpr
end

function isIntOrBool(@nospecialize(exp::DAE.Exp))
  @match exp begin
    DAE.BCONST(__) => true
    DAE.ICONST(__) => true
    DAE.CREF(componentRef, DAE.T_INTEGER(__) || DAE.T_BOOL(__)) => true
    _ => false
  end
end
isIntOrBool(exp::SimulationCode.Exp)::Bool =
  exp isa SimulationCode.ICONST || exp isa SimulationCode.BCONST ||
  (exp isa SimulationCode.EXP_CREF &&
   (exp.ty isa SimulationCode.TYPE_INTEGER || exp.ty isa SimulationCode.TYPE_BOOL))

"""
`writeEqsToFile(elems::Vector{Expr}, filename)`
  Used for logging.
"""
function writeEqsToFile(elems::Vector{Expr}, filename)
  local buffer = IOBuffer()
  try
    for e in elems
      e = stripComments(e)
      e = stripBeginBlocks(e)
      local eStr = string(e)
      eqStr = replace(eStr, "~" => "=")
      eqStr = replace(eqStr, "(t)" => "")
      eqStr = replace(eqStr, "Differential" => "der")
      println(buffer, eqStr)
    end
  catch e
    @error string("Failed writing the model to the file:",  filename)
    rethrow()
  end
  println(buffer, "------------------------------------")
  println(buffer, "Statistics:")
  println(buffer, "Number of items:" * string(length(elems)))
  println(buffer, "------------------------------------")
  write(filename, String(take!(buffer)))
end

"""
  Returns true if there is no discrete variables in the condition.
"""
function isContinuousCondition(cond::DAE.Exp, simCode)
  local allCrefs = Util.getAllCrefs(cond)
  local isContinuousCond = false
  if isone(length(allCrefs)) && string(first(allCrefs)) == "time"
    isContinuousCond = true
  else
    for cref in allCrefs
      local ht = simCode.stringToSimVarHT
      local crefName = string(cref)
      if crefName == "time"
        #= A relation against `time` (e.g. `time >= t_next`) is a time/state
           event, i.e. a zero-crossing of `time - rhs`, even when the other
           operand is a discrete variable. Treat it as continuous so it is
           emitted as an edge-triggered ContinuousCallback rather than a
           level-triggered discrete callback that re-fires every step. =#
        isContinuousCond = true
        continue
      end
      if !haskey(ht, crefName)
        isContinuousCond = true
        continue
      end
      local var = last(ht[crefName])
      #= Only a genuinely continuous variable makes the condition continuous.
         Parameters are constant and discrete variables change only at events,
         so a condition over only those (e.g. `(tLH>0 or tHL>0) and change(x)`)
         is an event-driven discrete callback, not a zero-crossing. =#
      isContinuousCond = isContinuousCond ||
        (!(SimulationCode.isDiscrete(var)) && !(SimulationCode.isParameter(var)))
    end
  end
  return isContinuousCond
end

"""
Replace a variable with an optional naming convention specified by the prefix and suffix arguments
"""
function replaceVars(sym::Symbol; kwargs...)
  if Base.isoperator(sym)
    sym
  elseif sym === :t
    sym
  else
    local vStr = string(sym)
    #= If the symbol is not a simulation variable (e.g. a function or module name),
       pass it through unchanged rather than crashing with a KeyError. =#
    if !haskey(kwargs[:ht], vStr)
      return sym
    end
    #= Lookup the variable. =#
    local htEntry = kwargs[:ht][vStr]
    local sVar = last(htEntry)
    local isStateOrAlg = SimulationCode.isStateOrAlgebraic(sVar)
    if isStateOrAlg
      #= useIndexedU: for ContinuousCallback conditions the trial state is passed as u,
         so generate u[idx] instead of integrator[:sym] to correctly detect sign changes. =#
      if get(kwargs, :useIndexedU, false)
        local idx = first(htEntry)
        #= useMTKIdx: use runtime MTK variable ordering instead of BDAE index.
           MTK may reorder unknowns (e.g., algebraic vars before differential),
           so _mtk_idx (built from unknowns(reducedSystem)) gives the correct u position. =#
        if get(kwargs, :useMTKIdx, false)
          :(u[get(_mtk_idx, $vStr, $idx)])
        else
          :(u[$idx])
        end
      else
        :($(Meta.parse(string(kwargs[:integratorCref],
                              kwargs[:prefix],
                              ":$sym",
                              kwargs[:suffix]))))
      end
    else #Parameter or Discrete.
      :($(Meta.parse(string(kwargs[:integratorCref],
                            ".ps",
                            kwargs[:prefix],
                            ":$sym",
                            kwargs[:suffix]))))
    end
  end
end
function replaceVars(expr::Expr; kwargs...)
  Expr(expr.head,
       map((x) -> replaceVars(x; kwargs...), expr.args)...)
end
replaceVars(x; kwargs...) = x


"""
  Utility function to check if a given expression contains a specific datatype.
"""
function exprContainsDatatype(ex, datatype)
  if ex isa Symbol
    return false
  elseif ex isa datatype
    return true
  elseif ex isa Expr
    return any(arg -> exprContainsDatatype(arg, datatype), ex.args)
  else
    return false
  end
end

"""
  Flatten a record argument in a function call into its constituent fields.
  Returns a vector of Symbols for the flattened fields, or empty vector if not a record.
"""
function flattenRecordCallArg(arg::DAE.Exp, simCode, hashTable; varPrefix::String="", varSuffix::String="")::Vector{Symbol}
  @match arg begin
    #= Original pattern: T_COMPLEX wrapping a RECORD class. =#
    DAE.CREF(cr, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _)) => begin
      local baseName::String = SimulationCode.string(cr)
      local flattenedExprs::Vector{Symbol} = Symbol[]
      for field in varLst
        @match field begin
          DAE.TYPES_VAR(fieldName, _, _, _, _) => begin
            local flatName::String = baseName * COMPONENT_SEPARATOR * fieldName
            #= Check if the flattened variable exists in the hash table =#
            if haskey(hashTable, flatName)
              push!(flattenedExprs, Symbol(varPrefix * flatName * varSuffix))
            else
              push!(flattenedExprs, Symbol(flatName))
            end
          end
          _ => nothing
        end
      end
      return flattenedExprs
    end
    #= Connector-typed Complex CREFs (e.g. `Modelica.ComplexBlocks.Interfaces.ComplexInput
       transferFunction_u`) carry a T_COMPLEX with `ClassInf.CONNECTOR` rather than
       `ClassInf.RECORD`. Codegen needs the same scalarization treatment — without it,
       a bare `transferFunction_u` symbol leaks into the generated module and triggers
       `UndefVarError` at MTK eval. Same flattening, different ClassInf. =#
    DAE.CREF(cr, DAE.T_COMPLEX(_, varLst, _)) => begin
      local baseName::String = SimulationCode.string(cr)
      local flattenedExprs::Vector{Symbol} = Symbol[]
      for field in varLst
        @match field begin
          DAE.TYPES_VAR(fieldName, _, _, _, _) => begin
            local flatName::String = baseName * COMPONENT_SEPARATOR * fieldName
            if haskey(hashTable, flatName)
              push!(flattenedExprs, Symbol(varPrefix * flatName * varSuffix))
            else
              push!(flattenedExprs, Symbol(flatName))
            end
          end
          _ => nothing
        end
      end
      #= Only treat as flattened if at least one scalar sibling exists in HT.
         Otherwise fall through to the default codegen path so we don't emit
         bogus references to fields that were never produced. =#
      if any(s -> haskey(hashTable, String(s)), flattenedExprs)
        return flattenedExprs
      end
      return Symbol[]
    end
    _ => return Symbol[]
  end
end

"""
  Extract common hvcat subexpressions from a list of equation Exprs.
  Returns (modifiedEquations, preambleAssignments) where preamble contains
  local variable assignments for deduplicated hvcat calls.
"""
function extractCommonHvcats(equations::Vector)
  local allHvcats = Dict{String, Int}()
  local hvcatExprs = Dict{String, Expr}()
  for eq in equations
    local found = Dict{String, Bool}()
    _collectHvcats!(eq, allHvcats, hvcatExprs, found)
  end
  local duplicates = filter(p -> p.second >= 2, allHvcats)
  if isempty(duplicates)
    return equations, Expr[]
  end
  local replacements = Dict{String, Symbol}()
  local preamble = Expr[]
  local idx = 0
  for (key, _) in duplicates
    local varName = Symbol("_hv_", idx)
    replacements[key] = varName
    push!(preamble, :(local $varName = $(hvcatExprs[key])))
    idx += 1
  end
  local modifiedEqs = [_replaceHvcats(deepcopy(eq), replacements) for eq in equations]
  return modifiedEqs, preamble
end

function _collectHvcats!(expr, counts::Dict{String,Int}, exprs::Dict{String,Expr}, found::Dict{String,Bool})
  if !(expr isa Expr)
    return
  end
  if expr.head == :call && length(expr.args) >= 1 && expr.args[1] == :hvcat
    local key = string(expr)
    if !haskey(found, key)
      found[key] = true
      counts[key] = get(counts, key, 0) + 1
      if !haskey(exprs, key)
        exprs[key] = expr
      end
    end
  end
  for arg in expr.args
    _collectHvcats!(arg, counts, exprs, found)
  end
end

function _replaceHvcats(expr, replacements::Dict{String,Symbol})
  if !(expr isa Expr)
    return expr
  end
  if expr.head == :call && length(expr.args) >= 1 && expr.args[1] == :hvcat
    local key = string(expr)
    if haskey(replacements, key)
      return replacements[key]
    end
  end
  for i in eachindex(expr.args)
    expr.args[i] = _replaceHvcats(expr.args[i], replacements)
  end
  return expr
end

#= ===================================================================
   Utilities migrated out of MTK_CodeGeneration.jl. These are pure
   predicates / collectors / Expr transforms — no MTK-specific codegen.
   =================================================================== =#

_isCodegenNameChar(c::Char)::Bool =
  isletter(c) || isdigit(c) || c == '_' || c == 'ˍ' || c == '[' || c == ']' || c == '"'

function equationMentionsVariableName(eqStr::AbstractString, varName::AbstractString)::Bool
  isempty(varName) && return false
  local startIdx = firstindex(eqStr)
  while true
    local match = findnext(varName, eqStr, startIdx)
    match === nothing && return false
    local firstIdx = first(match)
    local lastIdx = last(match)
    local beforeOk = firstIdx == firstindex(eqStr) || !_isCodegenNameChar(eqStr[prevind(eqStr, firstIdx)])
    local afterOk = lastIdx == lastindex(eqStr) || !_isCodegenNameChar(eqStr[nextind(eqStr, lastIdx)])
    beforeOk && afterOk && return true
    startIdx = nextind(eqStr, firstIdx)
  end
end

function containsDerCall(@nospecialize(exp::DAE.Exp))::Bool
  @match exp begin
    DAE.CALL(Absyn.IDENT("der"), _) => true
    DAE.UNARY(_, e) => containsDerCall(e)
    DAE.BINARY(e1, _, e2) => containsDerCall(e1) || containsDerCall(e2)
    DAE.LUNARY(_, e) => containsDerCall(e)
    DAE.LBINARY(e1, _, e2) => containsDerCall(e1) || containsDerCall(e2)
    DAE.RELATION(e1, _, e2) => containsDerCall(e1) || containsDerCall(e2)
    DAE.IFEXP(c, t, e) => containsDerCall(c) || containsDerCall(t) || containsDerCall(e)
    DAE.CALL(_, explst) => any(containsDerCall, explst)
    DAE.CAST(_, e) => containsDerCall(e)
    DAE.ASUB(e, _) => containsDerCall(e)
    DAE.TSUB(e, _, _) => containsDerCall(e)
    _ => false
  end
end

"""
    startValueVariableNames(simCode) -> Vector{String}

The variables whose start values go into the initial-value map: the flagged
irreducibles, then every state that is not among them. A state keeps its
start value whether or not it is irreducible (a state MTK may reduce still
needs one, and a structural transition model has no initialization problem
to find it).
"""
function startValueVariableNames(simCode::SimulationCode.SIM_CODE)::Vector{String}
  local names = String[vn for vn in simCode.irreducibleVariables]
  local seen = Set{String}(names)
  for (name, (_, simVar)) in simCode.stringToSimVarHT
    simVar.varKind isa SimulationCode.STATE || continue
    name in seen && continue
    push!(names, name)
    push!(seen, name)
  end
  return names
end

function hasExplicitStartValue(vars::Vector, simCode::SimulationCode.SIM_CODE)::Bool
  local ht::Dict = simCode.stringToSimVarHT
  for var in vars
    local entry = get(ht, var, nothing)
    if entry === nothing
      continue
    end
    (_, simVar) = entry
    local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
    @match optAttributes begin
      SOME(attributes) => begin
        @match attributes.start begin
          SOME(_) => return true
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  return false
end

function fixedStartVarNames(vars::Vector, simCode::SimulationCode.SIM_CODE)::Vector{String}
  local result::Vector{String} = String[]
  if isempty(vars)
    return result
  end
  local ht::Dict = simCode.stringToSimVarHT
  for var in vars
    haskey(ht, var) || continue
    (_, simVar) = ht[var]
    local matched = @match simVar.attributes begin
      SOME(attributes) => @match (attributes.start, attributes.fixed) begin
        (SOME(_), SOME(DAE.BCONST(true))) => true
        _ => false
      end
      _ => false
    end
    matched && push!(result, simVar.name)
  end
  return result
end

function _isLiteralBind(exp::DAE.Exp)::Bool
  @match exp begin
    DAE.RCONST(__) => true
    DAE.ICONST(__) => true
    DAE.BCONST(__) => true
    DAE.SCONST(__) => true
    DAE.ENUM_LITERAL(__) => true
    _ => false
  end
end

#= SimCode-native bindings (post-bindExp migration) reach the literal check too. =#
_isLiteralBind(exp::SimulationCode.Exp)::Bool =
  exp isa SimulationCode.RCONST || exp isa SimulationCode.ICONST ||
  exp isa SimulationCode.BCONST || exp isa SimulationCode.SCONST ||
  exp isa SimulationCode.ENUM_LITERAL

function isArrayType(v::DAE.VAR)::Bool
  local crefType = @match v.componentRef begin
    DAE.CREF_IDENT(_, identType, _) => identType
    DAE.CREF_QUAL(_, identType, _, _) => identType
    _ => v.ty
  end
  @match crefType begin
    DAE.T_ARRAY(__) => true
    _ => false
  end
end

function collectCalledFunctionNames!(names::OrderedSet{String}, @nospecialize(exp::DAE.Exp))
  @match exp begin
    DAE.CALL(path = path, expLst = explst) => begin
      push!(names, string(path))
      for arg in explst
        collectCalledFunctionNames!(names, arg)
      end
    end
    _ => begin
      Util.traverseExpTopDown(exp,
                              (e, acc) -> begin
                                if e isa DAE.CALL
                                  push!(acc, string(e.path))
                                end
                                (e, true, acc)
                              end,
                              names)
    end
  end
  return names
end

function collectCalledFunctionNames!(names::OrderedSet{String},
                                     eq::Union{BDAE.RESIDUAL_EQUATION, SimulationCode.RESIDUAL_EQUATION})
  collectCalledFunctionNames!(names, SimulationCode.toDAEExp(eq.exp))
end

function collectCalledFunctionNames!(names::OrderedSet{String}, eq::BDAE.EQUATION)
  collectCalledFunctionNames!(names, eq.lhs)
  collectCalledFunctionNames!(names, eq.rhs)
end

Base.@nospecializeinfer function collectCalledFunctionNames!(names::OrderedSet{String}, @nospecialize(eq::SimulationCode.EQUATION))
  collectCalledFunctionNames!(names, SimulationCode.toDAEExp(eq.lhs))
  collectCalledFunctionNames!(names, SimulationCode.toDAEExp(eq.rhs))
end

function collectCalledFunctionNames!(names::OrderedSet{String}, eq::BDAE.ARRAY_EQUATION)
  collectCalledFunctionNames!(names, eq.left)
  collectCalledFunctionNames!(names, eq.right)
end

Base.@nospecializeinfer function collectCalledFunctionNames!(names::OrderedSet{String}, @nospecialize(eq::SimulationCode.ARRAY_EQUATION))
  collectCalledFunctionNames!(names, SimulationCode.toDAEExp(eq.left))
  collectCalledFunctionNames!(names, SimulationCode.toDAEExp(eq.right))
end

function collectCalledFunctionNames!(names::OrderedSet{String}, eq::BDAE.SOLVED_EQUATION)
  collectCalledFunctionNames!(names, eq.exp)
end

function collectCalledFunctionNames!(names::OrderedSet{String}, eq::BDAE.COMPLEX_EQUATION)
  collectCalledFunctionNames!(names, eq.left)
  collectCalledFunctionNames!(names, eq.right)
end

"""
    getRHSVariables(op) -> Vector{DAE.ComponentRef}

Unified WhenOperator RHS-cref collector that accepts both BDAE-side and
SimCode-side operator records. BDAE variants delegate to
`Backend.BDAEUtil.getRHSVariables`.
"""
# Helper: SIM-side WhenOperators carry ::Exp; Util.getAllCrefs is DAE.Exp-typed.
_asDAE(e::DAE.Exp) = e
_asDAE(e::SimulationCode.Exp) = SimulationCode.toDAEExp(e)

function getRHSVariables(op::Union{BDAE.ASSIGN, SimulationCode.ASSIGN})::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(_asDAE(op.right)))
end

function getRHSVariables(op::Union{BDAE.REINIT, SimulationCode.REINIT})::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(_asDAE(op.value)))
end

function getRHSVariables(op::Union{BDAE.NORETCALL, SimulationCode.NORETCALL})::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(_asDAE(op.exp)))
end

function getRHSVariables(op::Union{BDAE.ASSERT, SimulationCode.ASSERT})::Vector{DAE.ComponentRef}
  return vcat(listArray(Util.getAllCrefs(_asDAE(op.condition))),
              listArray(Util.getAllCrefs(_asDAE(op.message))),
              listArray(Util.getAllCrefs(_asDAE(op.level))))
end

function getRHSVariables(op::Union{BDAE.TERMINATE, SimulationCode.TERMINATE})::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(_asDAE(op.message)))
end

function getRHSVariables(::Union{BDAE.RECOMPILATION, SimulationCode.RECOMPILATION})::Vector{DAE.ComponentRef}
  return DAE.ComponentRef[]
end

function getRHSVariables(::Union{BDAE.AGENTIC_RECOMPILATION, SimulationCode.AGENTIC_RECOMPILATION})::Vector{DAE.ComponentRef}
  return DAE.ComponentRef[]
end

function collectCalledFunctionNames!(names::OrderedSet{String}, stmt::Union{BDAE.ASSIGN, SimulationCode.ASSIGN})
  collectCalledFunctionNames!(names, _asDAE(stmt.left))
  collectCalledFunctionNames!(names, _asDAE(stmt.right))
end

function collectCalledFunctionNames!(names::OrderedSet{String}, stmt::Union{BDAE.REINIT, SimulationCode.REINIT})
  collectCalledFunctionNames!(names, stmt.stateVar)
  collectCalledFunctionNames!(names, _asDAE(stmt.value))
end

function collectCalledFunctionNames!(names::OrderedSet{String}, stmt::Union{BDAE.NORETCALL, SimulationCode.NORETCALL})
  collectCalledFunctionNames!(names, _asDAE(stmt.exp))
end

# SIM-Exp passthrough so `branch.condition` (now ::Exp) routes through the DAE.Exp visitor.
function collectCalledFunctionNames!(names::OrderedSet{String}, e::SimulationCode.Exp)
  return collectCalledFunctionNames!(names, SimulationCode.toDAEExp(e))
end

function collectCalledFunctionNames!(names::OrderedSet{String}, ::Any)
  return names
end

function collectCalledFunctionNames!(names::OrderedSet{String}, simCode::SimulationCode.SIM_CODE)
  for eq in simCode.residualEquations
    collectCalledFunctionNames!(names, eq)
  end
  for eq in simCode.initialEquations
    collectCalledFunctionNames!(names, eq)
  end
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCalledFunctionNames!(names, branch.condition)
      for eq in branch.residualEquations
        collectCalledFunctionNames!(names, eq)
      end
    end
  end
  for whenEq in simCode.whenEquations
    collectCalledFunctionNames!(names, SimulationCode.toDAEExp(whenEq.whenEquation.condition))
    for stmt in whenEq.whenEquation.whenStmtLst
      collectCalledFunctionNames!(names, stmt)
    end
  end
  return names
end

function _resolveModelicaCallTargets(expr)
  if !(expr isa Expr)
    return expr
  end
  if expr.head === :call && length(expr.args) >= 2 &&
     expr.args[1] == :(Base.invokelatest) && expr.args[2] isa Symbol
    local funcSym = expr.args[2]
    local registry = :(OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[$(QuoteNode(funcSym))])
    local rest = Any[_resolveModelicaCallTargets(a) for a in expr.args[3:end]]
    return Expr(:call, expr.args[1], registry, rest...)
  end
  return Expr(expr.head, Any[_resolveModelicaCallTargets(a) for a in expr.args]...)
end

function _literalRefName(expr::Expr)
  expr.head === :ref || return nothing
  isempty(expr.args) && return nothing
  expr.args[1] isa Symbol || return nothing
  local parts = String[string(expr.args[1])]
  for idx in expr.args[2:end]
    local v = if idx isa Integer
      idx
    elseif idx isa QuoteNode && idx.value isa Integer
      idx.value
    else
      return nothing
    end
    push!(parts, string("[", v, "]"))
  end
  return join(parts)
end

function _renameAlgIdentifiers(expr, names::OrderedSet{String}, prefix::String)
  if expr isa Symbol
    local s = string(expr)
    s == "time" && return expr
    return s in names ? Symbol(prefix * s) : expr
  elseif expr isa Expr
    #= The growth guard of an array whose elements are locals of their own
       here (x[1] := 1.0 is var"_alg_x[1]" = 1.0): no array to grow. =#
    if expr.head === :call && length(expr.args) >= 2 && expr.args[2] isa Symbol &&
       endswith(string(expr.args[1]), "ensureAlgArrayLength!") && !(string(expr.args[2]) in names)
      return nothing
    end
    if expr.head === :ref
      local refName = _literalRefName(expr)
      if refName !== nothing && refName in names
        return Symbol(prefix * refName)
      end
    end
    if expr.head === :call && length(expr.args) >= 1
      local newArgs = Any[expr.args[1]]
      for i in 2:length(expr.args)
        push!(newArgs, _renameAlgIdentifiers(expr.args[i], names, prefix))
      end
      return Expr(:call, newArgs...)
    else
      return Expr(expr.head, map(a -> _renameAlgIdentifiers(a, names, prefix), expr.args)...)
    end
  end
  return expr
end

function _renameAlgIdentifiers(expr, names::OrderedSet{String})
  return _renameAlgIdentifiers(expr, names, "_alg_")
end

function _readStartAttributeAsLiteral(sv)::Float64
  local attrs = sv.attributes
  @match attrs begin
    SOME(a) => begin
      @match a.start begin
        SOME(DAE.RCONST(r)) => Float64(r)
        SOME(DAE.ICONST(i)) => Float64(i)
        SOME(DAE.BCONST(b)) => (b ? 1.0 : 0.0)
        _ => 0.0
      end
    end
    _ => 0.0
  end
end

function _collectInitAlgLhsRhsCrefsDAE!(lhs::OrderedSet{String}, rhs::OrderedSet{String}, stmt)
  local pushCrefsFrom = function(exp)
    exp === nothing && return
    for c in Util.getAllCrefs(exp); push!(rhs, string(c)); end
  end
  local crefName = function(e)
    @match e begin
      DAE.CREF(cr, _) => string(cr)
      _ => ""
    end
  end
  @match stmt begin
    DAE.STMT_ASSIGN(_, e1, e, _) => begin
      local n = crefName(e1); !isempty(n) && push!(lhs, n)
      pushCrefsFrom(e)
    end
    DAE.STMT_TUPLE_ASSIGN(_, lhsList, e, _) => begin
      for lhsExp in lhsList
        local n = crefName(lhsExp); !isempty(n) && push!(lhs, n)
      end
      pushCrefsFrom(e)
    end
    DAE.STMT_ASSIGN_ARR(_, e1, e, _) => begin
      local n = crefName(e1); !isempty(n) && push!(lhs, n)
      pushCrefsFrom(e)
    end
    DAE.STMT_NORETCALL(e, _) => pushCrefsFrom(e)
    DAE.STMT_ASSERT(c, m, l, _) => begin
      pushCrefsFrom(c); pushCrefsFrom(m); pushCrefsFrom(l)
    end
    DAE.STMT_TERMINATE(m, _) => pushCrefsFrom(m)
    DAE.STMT_REINIT(v, val, _) => begin
      pushCrefsFrom(v); pushCrefsFrom(val)
    end
    DAE.STMT_IF(cond, body, else_, _) => begin
      pushCrefsFrom(cond)
      for s in body; _collectInitAlgLhsRhsCrefsDAE!(lhs, rhs, s); end
      _collectInitAlgLhsRhsCrefsDAEElse!(lhs, rhs, else_)
    end
    DAE.STMT_FOR(_, _, iter, _, range, body, _) => begin
      pushCrefsFrom(range)
      push!(lhs, iter)
      for s in body; _collectInitAlgLhsRhsCrefsDAE!(lhs, rhs, s); end
    end
    DAE.STMT_PARFOR(_, _, iter, _, range, body, _, _) => begin
      pushCrefsFrom(range)
      push!(lhs, iter)
      for s in body; _collectInitAlgLhsRhsCrefsDAE!(lhs, rhs, s); end
    end
    DAE.STMT_WHILE(cond, body, _) => begin
      pushCrefsFrom(cond)
      for s in body; _collectInitAlgLhsRhsCrefsDAE!(lhs, rhs, s); end
    end
    _ => nothing
  end
end

function _collectInitAlgLhsRhsCrefsDAEElse!(lhs::OrderedSet{String}, rhs::OrderedSet{String}, else_)
  @match else_ begin
    DAE.ELSE(stmts) => for s in stmts; _collectInitAlgLhsRhsCrefsDAE!(lhs, rhs, s); end
    DAE.ELSEIF(cond, stmts, rest) => begin
      for c in Util.getAllCrefs(cond); push!(rhs, string(c)); end
      for s in stmts; _collectInitAlgLhsRhsCrefsDAE!(lhs, rhs, s); end
      _collectInitAlgLhsRhsCrefsDAEElse!(lhs, rhs, rest)
    end
    _ => nothing
  end
end

function _collectInitAlgLhsRhsCrefs!(lhs::OrderedSet{String}, rhs::OrderedSet{String}, op)
  # Use DAE_identifierToString → canonicalName so cref strings match stringToSimVarHT key format
  # (per-dim brackets `mem[1][1]`, not comma-join `mem[1, 1]`).
  local pushCrefsFrom = function(exp)
    exp === nothing && return
    for c in Util.getAllCrefs(exp)
      push!(rhs, SimulationCode.DAE_identifierToString(c))
    end
  end
  if op isa BDAE.ASSIGN || op isa SimulationCode.ASSIGN
    # SimulationCode.ASSIGN.left is ::Exp (SIM.EXP_CREF) post-migration; convert to DAE for the match.
    local leftDAE = op isa SimulationCode.ASSIGN ? SimulationCode.toDAEExp(op.left) : op.left
    @match leftDAE begin
      DAE.CREF(cr, _) => push!(lhs, SimulationCode.DAE_identifierToString(cr))
      _ => nothing
    end
    pushCrefsFrom(op.right)
  elseif op isa BDAE.REINIT || op isa SimulationCode.REINIT
    push!(lhs, SimulationCode.DAE_identifierToString(op.stateVar))
    pushCrefsFrom(op.value)
  elseif op isa BDAE.NORETCALL || op isa SimulationCode.NORETCALL
    pushCrefsFrom(op.exp)
  elseif op isa BDAE.ASSERT || op isa SimulationCode.ASSERT
    pushCrefsFrom(op.condition); pushCrefsFrom(op.message); pushCrefsFrom(op.level)
  elseif op isa BDAE.TERMINATE || op isa SimulationCode.TERMINATE
    pushCrefsFrom(op.message)
  end
end

using ExportAll
@exportAll()

end #= module CodeGenerationUtil =#
