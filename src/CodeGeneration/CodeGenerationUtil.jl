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
import MacroTools


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
function transformToZeroCrossingCondition(@nospecialize(conditonalExpression::DAE.Exp))::DAE.Exp
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
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      DAE.BINARY(lhs, DAE.SUB(DAE.T_REAL_DEFAULT), rhs)
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

"""
Transforms a DAE Condition into a MTK continous condition.
"""
function transformToMTKContinousCondition(cond, simCode)
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
    #= Equality/inequality: discrete boolean comparison, treat as boolean zero-crossing =#
    DAE.RELATION(e1, DAE.EQUAL(__), e2) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5)
    end
    DAE.RELATION(e1, DAE.NEQUAL(__), e2) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5)
    end
    #= Boolean variable: convert to zero-crossing (var - 0.5) =#
    DAE.CREF(__) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5)
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
    #= General function call as boolean condition: treat as boolean zero-crossing =#
    DAE.CALL(__) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5)
    end
    _ => begin
      throw("Unsupported condition expression in IF_EQUATION: " * string(cond))
    end
  end
  return res
end

"""
Transforms a DAE Condition into a MTK continous condition equation.
"""
function transformToMTKContinousConditionEquation(cond, simCode)
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
    #= Equality/inequality: discrete boolean comparison, treat as boolean zero-crossing =#
    DAE.RELATION(e1, DAE.EQUAL(__), e2) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5 ~ 0)
    end
    DAE.RELATION(e1, DAE.NEQUAL(__), e2) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5 ~ 0)
    end
    #= Boolean variable: convert to zero-crossing equation (var - 0.5 ~ 0) =#
    DAE.CREF(__) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5 ~ 0)
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
    #= General function call as boolean condition: treat as boolean zero-crossing =#
    DAE.CALL(__) => begin
      :($(expToJuliaExpMTK(cond, simCode)) - 0.5 ~ 0)
    end
    _ => begin
      throw("Unsupported condition expression in IF_EQUATION: " * string(cond))
    end
  end
  return res
end


"""
  Flattens a vector of expressions.
"""
function flattenExprs(eqs::Vector{Expr})
  quote
    $(eqs...)
  end
end


"""
 Convert DAE.Exp into a Julia string.
"""
function expToJL(exp::DAE.Exp, simCode::SimulationCode.SIM_CODE; varPrefix="x")::String
  hashTable = simCode.stringToSimVarHT
  str = begin
    local int::ModelicaInteger
    local real::ModelicaReal
    local bool::Bool
    local tmpStr::String
    local cr::DAE.ComponentRef
    local e1::DAE.Exp
    local e2::DAE.Exp
    local e3::DAE.Exp
    local expl::List{DAE.Exp}
    local lstexpl::List{List{DAE.Exp}}
    @match exp begin
      DAE.ICONST(int) => string(int)
      DAE.RCONST(real)  => string(real)
      DAE.SCONST(tmpStr)  => (tmpStr)
      DAE.BCONST(bool)  => string(bool)
      DAE.ENUM_LITERAL((Absyn.IDENT(str), int))  => str + "()" + string(int) + ")"
      DAE.CREF(cr, _)  => begin
        varName = SimulationCode.string(cr)
        builtin = if varName == "time"
          true
        else
          false
        end
        if ! builtin
          #= If we refer to time, we simply return t instead of a concrete variable =#
          indexAndVar = hashTable[varName]
          varKind::SimulationCode.SimVarType = indexAndVar[2].varKind
          @match varKind begin
            SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
            SimulationCode.STATE(__) => "$varPrefix[$(indexAndVar[1])] #= $varName =#"
            SimulationCode.PARAMETER(__) => "p[$(indexAndVar[1])] #= $varName =#"
            SimulationCode.ALG_VARIABLE(__) => "$varPrefix[$(indexAndVar[1])] #= $varName =#"
            SimulationCode.STATE_DERIVATIVE(__) => "dx[$(indexAndVar[1])] #= der($varName) =#"
          end
        else #= Currently only time is a builtin variabe. Time is represented as t in the generated code =#
          "t"
        end
      end
      DAE.UNARY(operator = op, exp = e1) => begin
        ("(" + SimulationCode.string(op) + " " + expToJL(e1, simCode) + ")")
      end
      DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        (expToJL(e1, simCode, varPrefix=varPrefix) + " " + SimulationCode.string(op) + " " + expToJL(e2, simCode, varPrefix=varPrefix))
      end
      DAE.LUNARY(operator = op, exp = e1)  => begin
        ("(" + SimulationCode.string(op) + " " + expToJL(e1, simCode, varPrefix=varPrefix) + ")")
      end
      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        (expToJL(e1, simCode, varPrefix=varPrefix) + " " + SimulationCode.string(op) + " " + expToJL(e2, simCode, varPrefix=varPrefix))
      end
      DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
        (expToJL(e1, simCode, varPrefix=varPrefix) + " " + SimulationCode.string(op) + " " + expToJL(e2, simCode,varPrefix=varPrefix))
      end
      #=TODO?=#
      DAE.IFEXP(expCond = e1, expThen = e2, expElse = e3) => begin
        "if" + expToJL(e1, simCode, varPrefix=varPrefix) + "\n" + expToJL(e2, simCode,varPrefix=varPrefix) + "else\n" + expToJL(e3, simCode,varPrefix=varPrefix) + "\nend"
      end
      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = expl)  => begin
        #=
          TODO: Keeping it simple for now, we assume we only have one argument in the call
          We handle derivitives seperatly
        =#
        varName = SimulationCode.DAE_identifierToString(listHead(expl))
        (index, type) = hashTable[varName]
        @match tmpStr begin
        "der" => "dx[$index]  #= der($varName) =#"
          "pre" => begin
            indexForVar = hashTable[varName][1]
            "(integrator.u[$(indexForVar)])"
          end
          "edge" =>  begin
             indexForVar = hashTable[varName][1]
             string(tuple(map((x) -> expToJL(x, simCode, varPrefix=varPrefix), expl)...)...) + " && ! integrator.uprev[$(indexForVar)]"
          end
          _  =>  begin
            tmpStr *= string(tuple(map((x) -> expToJL(x, simCode, varPrefix=varPrefix), expl)...)...)
          end
        end
      end
      DAE.CAST(exp = e1)  => begin
         expToJL(e1, simCode)
      end
      DAE.ARRAY(DAE.T_ARRAY(DAE.T_BOOL(__)), scalar, array) => begin
        local arrayExp = "#= Array exp=# reduce(|, ["
        for e in array
          arrayExp *= expToJL(e, simCode) + ","
        end
        arrayExp *= "])"
      end
      _ =>  throw(ErrorException("$exp not yet supported"))
    end
  end
  return "(" + str + ")"
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
      DAE.MUL_ARR() => :*
      DAE.DIV_ARR() => :/
      DAE.MUL_ARRAY_SCALAR() => :*
      DAE.ADD_ARRAY_SCALAR() => :+
      DAE.SUB_SCALAR_ARRAY() =>  :-
      DAE.MUL_SCALAR_PRODUCT() => :*
      DAE.MUL_MATRIX_PRODUCT() => :*
      DAE.DIV_ARRAY_SCALAR() => :/
      DAE.DIV_SCALAR_ARRAY() => :/
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
      DAE.USERDEFINED() => throw("Unknown operator: Userdefined")
      _ => throw("Unknown operator")
    end
end


"
  TODO: Keeping it simple for now, we assume we only have one argument in the call
  We handle derivitives seperatly
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
      expToJuliaExp(listHead(expLst), simCode, varPrefix=varPrefix)
    end
    _  =>  begin
      argPart = tuple(map((x) -> expToJuliaExp(x, simCode, varPrefix=varPrefix), expLst)...)
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

function _isSimCodeFunctionName(name::AbstractString, simCode)::Bool
  local canonical = OMBackend.canonicalName(name)
  for f in simCode.functions
    if f.name == canonical
      return true
    end
  end
  return false
end

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

function _modelicaFunctionCallExpr(path,
                                   expLst,
                                   simCode,
                                   hashTable;
                                   varPrefix = "",
                                   varSuffix = "",
                                   derSymbol = false)
  local normalizedFuncName = OMBackend.canonicalName(string(path))
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


"""
  Removes all comments from a given exp
"""
function stripComments(ex::Expr)::Expr
  return Base.remove_linenums!(ex)
end

"""
Transforms:
  <name>[<index>] -> <name>_index
"""
function arrayToSymbolicVariable(arrayRepr::Expr)::Expr
  MacroTools.postwalk(arrayRepr) do x
    MacroTools.@capture(x, T_[index_]) || return let
      x
    end
    return let
      local newVarStr::String = "$(T)_$(index)"
      local newVar = Symbol(newVarStr)
      return newVar
    end
  end
end

"""
 Removes all redudant blocks from a generated expression
"""
function stripBeginBlocks(e)::Expr
  MacroTools.postwalk(e) do x
    return MacroTools.unblock(x)
  end
end

"""
Transforms:
  <name>_index -> <name>[index]
Uses direct Expr construction instead of string interpolation + Meta.parse.
"""
const pattern = r".*_[0-9]+"
function symbolicVariableToArrayRef(e::Expr)::Expr
  MacroTools.postwalk(e) do x
    x isa Symbol || return x
    local sstr = String(x)
    match(pattern, sstr) === nothing && return x
    local parts = split(sstr, "_")
    return Expr(:ref, Symbol(parts[1]), parse(Int, parts[2]))
  end
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

"""
  Utility function, traverses a DAE exp. Variables are saved in the supplied variables array
  (Note that variables here refers to parameters as well)
"""
function getVariablesInDAE_Exp(@nospecialize(exp::DAE.Exp), simCode::SimulationCode.SIM_CODE, variables::Set)
  local  hashTable = simCode.stringToSimVarHT
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
    #= These are not varaiables, so we simply return what we have collected thus far. =#
    DAE.BCONST(bool) => variables
    DAE.ICONST(int) => variables
    DAE.RCONST(real) => variables
    DAE.SCONST(tmpStr) => variables
    DAE.CREF(DAE.CREF_IDENT("time", __), _) => begin
      push!(variables, Symbol("t"))
    end
    DAE.CREF(cr, _)  => begin
      varName = SimulationCode.string(cr)
      indexAndVar = hashTable[varName]
      push!(variables, Symbol(varName))
      varKind::SimulationCode.SimVarType = indexAndVar[2].varKind
      @match varKind begin
        SimulationCode.STATE(__) || SimulationCode.PARAMETER(__) || SimulationCode.ALG_VARIABLE(__) => begin
          push!(variables, Symbol(varName))
        end
        _ => begin
          @error "Unsupported varKind: $(varKind)"
          throw()
        end
      end
    end
    DAE.UNARY(operator = op, exp = e1) => begin
      getVariablesInDAE_Exp(e1, simCode, variables)
    end
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) || DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) || DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      getVariablesInDAE_Exp(e1, simCode, variables)
      getVariablesInDAE_Exp(e2, simCode, variables)
    end
    DAE.LUNARY(operator = op, exp = e1)  => begin
      getVariablesInDAE_Exp(e1, simCode, variables)
    end
    DAE.IFEXP(expCond = e1, expThen = e2, expElse = e3) => begin
      throw(ErrorException("If expressions not allowed in backend code"))
    end
    #= Should not introduce anything new..  I am a idiot - John 2021=#
    DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = explst)  => begin
      #TODO only assumes one argument
      getVariablesInDAE_Exp(listHead(explst), simCode, variables)
    end
    DAE.CAST(ty, exp)  => begin
      getVariablesInDAE_Exp(exp, simCode, variables)
    end
    _ =>  throw(ErrorException("$exp not yet supported"))
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

function getCycleInSCCs(sccs)
  for sc in sccs
    if length(sc) > 1
      return sc
    end
  end
  return []
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
function _ifexpBranchIsNonReal(e::DAE.Exp)::Bool
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

"""
Converts a DAE expression into a MTK expression.
varPrefix and varSuffix can be used to provide a prefix and a suffix to component reference.
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
            (SimulationCode.ARRAY(_, SOME(bindArray && DAE.ARRAY(__))) ||
             SimulationCode.ARRAY_PARAMETER(_, SOME(bindArray && DAE.ARRAY(__)))) => begin
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
          DAE.OR(__) => :($(lhs) + $(rhs) - $(lhs) * $(rhs))
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
        if _isSimCodeFunctionName(tmpStr, simCode)
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
               Qualified paths (not Absyn.IDENT) are user-defined Modelica functions that may return
               records (e.g., Complex with fields re, im). In symbolic mode, the wrapper returns a
               single Num, so plain [ix] fails. Use tupleElementCall for dual numeric/symbolic dispatch. =#
            local _asubCallResult = @match innerExp begin
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
          if _isArrayLiteral
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

"""
  Companion to `_recordFieldRe` for the imaginary part (DAE.RSUB(`im`)).
  Same `Base.Complex` reason: avoid the module-scope shadow of `Complex`.
"""
@inline _recordFieldIm(x::Base.Complex) = x.im
@inline _recordFieldIm(x::Tuple) = x[2]
@inline _recordFieldIm(x) = hasproperty(x, :im) ? getproperty(x, :im) : imag(x)

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
      (SimulationCode.ARRAY(_, SOME(bindArray && DAE.ARRAY(__))) ||
       SimulationCode.ARRAY_PARAMETER(_, SOME(bindArray && DAE.ARRAY(__)))) => begin
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
  local arrJL = canEval ? [eval(expr) for expr in elemExprs] : []
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
(We avoid structurally simplify for now since that might interfer with some other algorithms)
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
  @BACKEND_LOGGING dumpExpr = let _logPath = OMBackend.logPath("backend/codeGen", "preStructuralSimplify.log")
    quote
      try
        local _buffer = IOBuffer()
        local _eqs = ModelingToolkit.equations(firstOrderSystem)
        local _unks = ModelingToolkit.unknowns(firstOrderSystem)
        println(_buffer, "Pre-structural-simplify dump")
        println(_buffer, "============================")
        println(_buffer, "equations: ", length(_eqs))
        println(_buffer, "unknowns:  ", length(_unks))
        println(_buffer, "")
        println(_buffer, "Equations:")
        println(_buffer, "----------")
        for (_i, _e) in enumerate(_eqs)
          println(_buffer, "[", _i, "] ", _e)
        end
        println(_buffer, "")
        println(_buffer, "Unknowns:")
        println(_buffer, "---------")
        for (_i, _u) in enumerate(_unks)
          println(_buffer, "[", _i, "] ", _u)
        end
        write($(_logPath), String(take!(_buffer)))
      catch _err
        @warn "[preStructuralSimplify dump] failed" exception=_err
      end
    end
  end
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
    local _filteredObs = filter(_allObs) do eq
      local _varName = string(eq.lhs)
      any(p -> occursin(p, _varName), _obsPatterns)
    end
    if length(_filteredObs) < _nBefore
      @info "observedFilter: kept $(length(_filteredObs)) of $(_nBefore) MTK observed equations"
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
    local ew = wEqInner.elsewhenPart
    while ew !== nothing
      local nested = ew.data
      if _whenStmtLstTargets(nested.whenEquation.whenStmtLst, targetName)
        return true
      end
      ew = nested.whenEquation.elsewhenPart
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
    local lhsName = @match stmt begin
      BDAE.ASSIGN(left = lhs) => begin
        try
          SimulationCode.string(lhs)
        catch
          ""
        end
      end
      BDAE.REINIT(stateVar = sv) => begin
        try
          SimulationCode.string(sv)
        catch
          ""
        end
      end
      _ => ""
    end
    if lhsName == targetName
      return true
    end
  end
  return false
end

"""
  This functions evaluate a single DAE-constant:{Bool, Integer, Real, String}.
  If the argument  to this function is not a constant it throws an error.
"""
function evalDAEConstant(daeConstant::DAE.Exp, simCode)
  @match daeConstant begin
    DAE.BCONST(bool) => bool
    DAE.ICONST(int) => int
    DAE.RCONST(real) => real
    DAE.SCONST(tmpStr) => tmpStr
    #= Try to evaluate the expression =#
    DAE.BINARY(__) => begin
      evalDAE_Expression(daeConstant, simCode)
    end
    DAE.LBINARY(__) => begin
      evalDAE_Expression(daeConstant, simCode)
    end
    _ => begin
      local str = string(daeConstant)
      throw("$(str) is not a constant")
    end
  end
end

function evalDAEConstant(daeConstant::DAE.Exp)
  @match daeConstant begin
    DAE.BCONST(bool) => bool
    DAE.ICONST(int) => int
    DAE.RCONST(real) => real
    DAE.SCONST(tmpStr) => tmpStr
    #= Try to evaluate the expression =#
    _ => begin
      local str = string(daeConstant)
      throw("$(str) is not a constant")
    end
  end
end


"""
  Evaluates a simulation code parameter.
  Fails if the function is not a parameter.
"""
function evalSimCodeParameter(v::V, simCode) where V
  @match SimulationCode.SIMVAR(name, _, SimulationCode.PARAMETER(SOME(bindExp)), _) = v
  local val = evalDAEConstant(bindExp, simCode)
  return val
end

"""
 Evalutates the components in a DAE expression (Currently if the components are parameters)
"""
function evalDAE_Expression(expr, simCode)::Expr
  local shouldEval = true
  function replaceParameterVariable(exp, ht)
    if Util.isCref(exp)
      local simVar = last(simCode.stringToSimVarHT[string(exp)])
      if SimulationCode.isStateOrAlgebraic(simVar)
        shouldEval = false
      else
        #= Try to get binding expression for parameter substitution =#
        local bindExp = @match simVar.varKind begin
          SimulationCode.PARAMETER(SOME(be)) => be
          _ => nothing
        end
        if bindExp !== nothing
          return (bindExp, true, ht)
        else
          #= Parameter without binding (fixed=false, determined by initial equation).
             Leave CREF in place like a state/algebraic variable. =#
          shouldEval = false
        end
      end
    end
    (exp, true, ht)
  end
  local a = 0
  #= Replaces all known variables in the daeExp =#
  local daeExp = first(Util.traverseExpBottomUp(expr, replaceParameterVariable, 0))
  local jlExpr = expToJuliaExpMTK(daeExp, simCode)
  local evaluatedJLExpr = if shouldEval eval(jlExpr) else jlExpr end
  return quote $(evaluatedJLExpr) end
end

"""
    equationSides(eq) -> Tuple{DAE.Exp, DAE.Exp}

Return (lhs, rhs) for any BDAE equation shape.

- `BDAE.EQUATION`           has `.lhs` / `.rhs`
- `BDAE.COMPLEX_EQUATION`   has `.left` / `.right`   (record-to-record equality)
- `BDAE.ARRAY_EQUATION`     has `.left` / `.right`   (pre-scalarised array equality)
- `BDAE.RESIDUAL_EQUATION`  has `.exp`, already in LHS−RHS=0 form → return `(eq.exp, RCONST(0.0))`

Any other shape throws; the caller should have screened those out.
"""
function equationSides(eq)::Tuple{DAE.Exp, DAE.Exp}
  if eq isa BDAE.EQUATION
    return (eq.lhs, eq.rhs)
  elseif eq isa BDAE.COMPLEX_EQUATION
    return (eq.left, eq.right)
  elseif eq isa BDAE.ARRAY_EQUATION
    return (eq.left, eq.right)
  elseif eq isa BDAE.RESIDUAL_EQUATION
    return (eq.exp, DAE.RCONST(0.0))
  end
  error("equationSides: no (lhs, rhs) for $(typeof(eq)); " *
        "caller should filter to EQUATION / COMPLEX_EQUATION / ARRAY_EQUATION / RESIDUAL_EQUATION.")
end

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

"""
    solveParametricInitialEquations!(simCode)

For initial equations that only involve parameters (no states/algebraics),
solve numerically for parameters with `fixed=false` (no binding).
Updates the simCode hash table with the solved binding values.
"""
function solveParametricInitialEquations!(simCode::SimulationCode.SimCode)
  ht = simCode.stringToSimVarHT
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
    #= Build a parameter value map for all bound parameters =#
    paramValues = Dict{String, Float64}()
    for (key, (_, sv)) in ht
      if sv.varKind isa SimulationCode.PARAMETER && SimulationCode.hasBindingExp(sv)
        try
          paramValues[key] = Float64(evalSimCodeParameter(sv, simCode))
        catch
        end
      end
    end
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
    #= Build residual: LHS - RHS = 0 =#
    #= Replace all bound params with their values, leave free param as variable =#
    function substituteParams(exp, acc)
      if Util.isCref(exp)
        local key = string(exp)
        if key == freeName
          return (exp, true, acc)
        end
        local val = get(paramValues, key, nothing)
        if val !== nothing
          return (DAE.RCONST(val), true, acc)
        end
      end
      (exp, true, acc)
    end
    local lhsSubst = first(Util.traverseExpBottomUp(ieqLhs, substituteParams, 0))
    local rhsSubst = first(Util.traverseExpBottomUp(ieqRhs, substituteParams, 0))
    #= If the free parameter sits on the LHS (e.g. `globalSeed_seed = automaticGlobalSeed(0.0)`),
       swap the sides so eval(lhsJl) is on the constants side and the freeName lives in the
       residual function we Newton-solve. Without this swap eval(lhsJl) tries to evaluate the
       bare freeName cref and trips an UndefVarError. =#
    local function _containsCref(exp, name::String)::Bool
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
    if _containsCref(lhsSubst, freeName) && !_containsCref(rhsSubst, freeName)
      lhsSubst, rhsSubst = rhsSubst, lhsSubst
    end
    #= Evaluate LHS (should be all constants) =#
    local lhsJl = expToJuliaExpMTK(lhsSubst, simCode)
    local lhsVal = try
      Float64(eval(lhsJl))
    catch err
      @warn "[SIMCODE: solveParametricInitialEquations] could not evaluate LHS" freeName err
      continue
    end
    #= Build a Julia function for RHS with the free param as argument =#
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
    @info "[SIMCODE: solveParametricInitialEquations] solved $freeName = $x (from initial equation)"
    #= Update the simCode hash table with the solved value =#
    local (idx, oldSV) = ht[freeName]
    local newSV = SimulationCode.SIMVAR(oldSV.name, oldSV.index,
      SimulationCode.PARAMETER(SOME(DAE.RCONST(x))), oldSV.attributes)
    ht[freeName] = (idx, newSV)
    push!(solvedNames, freeName)
    solvedThisPass = true
  end
  if pass == 1 && !isempty(solvedNames)
    @info "[SIMCODE: solveParametricInitialEquations] pass $pass solved $(length(solvedNames)) parameter(s)" solvedNames
  elseif !isempty(solvedNames)
    @info "[SIMCODE: solveParametricInitialEquations] pass $pass solved $(length(solvedNames)) more parameter(s)" solvedNames
  end
  end #= while fixed-point =#
end

"""
  Decide the iv of the condition (whether the zero-crossing function is at zero at t=0).
  Returns true if the zero-crossing expression evaluates to zero at t=0.
  Returns false if it evaluates to a nonzero value (guard is active or inactive).
  When simCode is provided, substitutes parameter values and state variable start values.
"""
function evalInitialCondition(mtkCond, simCode = nothing)
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
        return numV >= 0.0
      end
      return (v == 0) != false
    end
    #= Build symbol -> value map from simCode =#
    local valMap = Dict{Symbol, Float64}()
    local ht = simCode.stringToSimVarHT
    for (key, (_, sv)) in ht
      local sym = Symbol(key)
      if sv.varKind isa SimulationCode.PARAMETER
        local pval = try
          local raw = evalSimCodeParameter(sv, simCode)
          if raw isa Expr
            #= `evalDAE_Expression` wraps its result in `quote $(val) end`
               (a `:block` Expr). That wrapper breaks `Float64(::Expr)`,
               which is the critical failure path for fold-promoted
               parameters whose bindExp is a BINARY/LBINARY/RELATION.
               Evaluate once to unwrap before the numeric coercion. =#
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
    #= Extract LHS from the mtkCond Expr (form: :(lhs ~ 0)) =#
    local lhsExpr = _extractZeroCrossingLHS(mtkCond)
    #= Substitute all variable references with numeric values =#
    local numExpr = _substituteExprValues(lhsExpr, valMap)
    #= Evaluate the resulting numeric expression =#
    local result = Base.invokelatest(eval, numExpr)
    local numResult = Float64(result)
    #= The zero-crossing function is negative when the condition is TRUE,
       positive when FALSE. Return true when condition is FALSE (positive),
       because the caller inverts: ifCond = !(evalInitialCondition(...)). =#
    return numResult >= 0.0
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
Walk an Expr tree and substitute variable references with numeric values.
- `sym(t)` calls (MTK variable references) -> look up sym in valMap
- bare Symbols -> look up in valMap if present
- :t -> 0.0
"""
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
  #= Generic recursion for other Expr types =#
  local newExpr = Expr(expr.head)
  for arg in expr.args
    push!(newExpr.args, _substituteExprValues(arg, valMap))
  end
  return newExpr
end

"""
  Generates an if-expression equation and add it to the continous part of the system.
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
                               subIdentifier::Int = identifier)
  local branch = branches[target]
  if branch.targets == -1
    return :($(first(deCausalize(branch.residualEquations[resEqIdx], simCode))))
  end
  #= Otherwise generate code for the other part =#
  #= ifCond variables are discrete parameters (not ODE unknowns), so the solver
     never perturbs them during Jacobian computation. Exact comparison is safe. =#
  local cond = :( $(Symbol(string("ifCond", identifier, subIdentifier))) == 1 )
  local rhs = first(deCausalize(branch.residualEquations[resEqIdx], simCode))
  quote
    ModelingToolkit.ifelse($(cond),
                           $(rhs),
                           $(generateIfExpressions(branches,
                                                   branches[target].targets,
                                                   resEqIdx,
                                                   identifier,
                                                   simCode;
                                                   subIdentifier = identifier + 1)))
  end
end

#= TODO.
  We currently assume residuals that we have made causal
  and that the original equations are written in a certain form.
=#
function deCausalize(eq, simCode)
  @match eq.exp begin
    DAE.BINARY(DAE.RCONST(0.0), _, exp2) => begin
      (:($(expToJuliaExpMTK(exp2, simCode))), :($(expToJuliaExpMTK(eq.exp.exp1, simCode))))
    end
    DAE.BINARY(exp1, _, DAE.RCONST(0.0)) => begin
      (:($(expToJuliaExpMTK(eq.exp.exp2, simCode))), :($(expToJuliaExpMTK(exp1, simCode))))
    end
    DAE.BINARY(exp1, _, exp2) => begin
      (:($(expToJuliaExpMTK(exp2, simCode))), :($(expToJuliaExpMTK(exp1, simCode))))
    end
    _ => begin
      throw("Unsupported equation:" * string(eq))
    end
  end
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

function isIntOrBool(@nospecialize(exp::DAE.Exp))
  @match exp begin
    DAE.BCONST(__) => true
    DAE.ICONST(__) => true
    DAE.CREF(componentRef, DAE.T_INTEGER(__) || DAE.T_BOOL(__)) => true
    _ => false
  end
end

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
    throw(e)
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
function isContinousCondition(cond::DAE.Exp, simCode)
  local allCrefs = Util.getAllCrefs(cond)
  local isContinuousCond = false
  if isone(length(allCrefs)) && string(first(allCrefs)) == "time"
    isContinuousCond = true
  else
    for cref in allCrefs
      local ht = simCode.stringToSimVarHT
      local crefName = string(cref)
      if crefName == "time"
        continue
      end
      if !haskey(ht, crefName)
        isContinuousCond = true
        continue
      end
      local var = last(ht[crefName])
      #=
      If one variable in the condition is continuous treat it as a continuous  callback
      =#
      isContinuousCond = isContinuousCond || !(SimulationCode.isDiscrete(var))
    end
  end
  return isContinuousCond
end

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

"""
  Flatten a record argument in a function call into its constituent fields.
  Returns a vector of Symbols for the flattened fields, or empty vector if not a record.
"""
function flattenRecordCallArg(arg::DAE.Exp, simCode, hashTable; varPrefix::String="", varSuffix::String="")::Vector{Symbol}
  @match arg begin
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
