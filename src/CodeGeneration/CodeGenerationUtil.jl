#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
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
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
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
  res = @match conditonalExpression begin
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

function transformToMTKConditionEquation(cond::DAE.Exp, simCode)
  res = @match cond begin
    DAE.RELATION(e1, DAE.LESS(__), e2) => begin
      :($(expToJuliaExpMTK(e1, simCode)) < $(expToJuliaExpMTK(e2, simCode)))
    end
    #= TODO: Is this correct? =#
    DAE.RELATION(e1, DAE.GREATER(__), e2) => begin
      :($(expToJuliaExpMTK(e1, simCode)) > $(expToJuliaExpMTK(e2, simCode)))
    end
    _ => begin
      throw("Operator: " * "'" * string(cond.operator) * "' in: " * string(cond) * " is not supported")
    end
  end
  return res
end

function transformToMTKContinousConditionEquation(cond::DAE.Exp, simCode)
  res = @match cond begin
    DAE.RELATION(e1, DAE.LESS(__), e2) => begin
      :($(expToJuliaExpMTK(e1, simCode)) - $(expToJuliaExpMTK(e2, simCode)) ~ 0)
    end
    _ => begin
      throw("Operator: " * "'" * string(cond.operator) * "' in: " * string(cond) * " is not supported")
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
Returns:
stateVariables, algebraicVariables, stateVariablesLoop, algebraicVariablesLoop
"""
function separateVariables(simCode)::Tuple
  local stringToSimVarHT = simCode.stringToSimVarHT
  local parameters::Vector = []
  local stateDerivatives::Vector = []
  local stateVariables::Vector = []
  local algebraicVariables::Vector = []
  local discreteVariables::Vector = []
  #= Loop arrays=#
  local stateDerivativesLoop::Vector = []
  local stateVariablesLoop::Vector = []
  local algebraicVariablesLoop::Vector = []
  local discreteVariablesLoop::Vector = []
  #= Separate the variables =#
  for varName in keys(stringToSimVarHT)
    (idx, var) = stringToSimVarHT[varName]
    if simCode.matchOrder[idx] in loop
      local varType = var.varKind
      @match varType  begin
        SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
        SimulationCode.STATE(__) => push!(stateVariablesLoop, varName)
        SimulationCode.PARAMETER(__) => push!(parameters, varName)
        SimulationCode.ALG_VARIABLE(__) => begin
          if idx in simCode.matchOrder
            push!(algebraicVariablesLoop, varName)
          else #= We have a variable that is not contained in continious system =#
            #= Treat discrete variables separate =#
            push!(discreteVariablesLoop, varName)
          end
        end
        #=TODO: Do I need to modify this?=#
        SimulationCode.STATE_DERIVATIVE(__) => push!(stateDerivativesLoop, varName)
      end
    else #= Someplace else=#
      local varType = var.varKind
      @match varType  begin
        SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
        SimulationCode.STATE(__) => push!(stateVariables, varName)
        SimulationCode.PARAMETER(__) => push!(parameters, varName)
        SimulationCode.ALG_VARIABLE(__) => begin
          if idx in simCode.matchOrder
            push!(algebraicVariables, varName)
          else #= We have a variable that is not contained in continious system =#
            #= Treat discrete variables separate =#
            push!(discreteVariables, varName)
          end
        end
        #=TODO: Do I need to modify this?=#
        SimulationCode.STATE_DERIVATIVE(__) => push!(stateDerivatives, varName)
      end
    end
  end
  return (stateVariables, algebraicVariables, stateVariablesLoop, algebraicVariablesLoop)
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
        varName = SimulationCode.string(listHead(expl))
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
      DAE.POW_ARRAY_SCALAR() => :^
      DAE.POW_SCALAR_ARRAY() => :^
      DAE.POW_ARR() => :^
      DAE.POW_ARR2() => :^
      DAE.AND() => :(&)
      DAE.OR() => :(||)
      DAE.NOT() => :(!)
      DAE.LESS() => :(<)
      DAE.LESSEQ() => :(<=)
      DAE.GREATER() => :(>)
      DAE.GREATEREQ() => :(>=)
      DAE.EQUAL() => :(=)
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
      varName = SimulationCode.string(listHead(expLst))
      (index, _) = ht[varName]
      quote
        dx[$(index)] #= der($varName) =#
      end
    end
    "pre" => begin
      varName = SimulationCode.string(listHead(expLst))
      (index, _) = ht[varName]
      indexForVar = ht[varName][1]
      quote
        (integrator.u[$(indexForVar)])
      end
    end
    _  =>  begin
      funcName = Symbol(pathStr)
      argPart = tuple(map((x) -> expToJuliaExp(x, simCode, varPrefix=varPrefix), expLst)...)
      quote
        $(funcName)($(argPart...))
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
                                              simCode::SimulationCode.SimCode, ht; varPrefix=varPrefix, derAsSymbol=false)::Expr
  @match pathStr begin
    "der" => begin
      varName = SimulationCode.string(listHead(expLst))
      if derAsSymbol
        quote
          $(Symbol("der_$(varName)"))
        end
      else
        quote
          der($(Symbol(varName)))
        end
      end
    end
    _  =>  begin
      funcName = Symbol(pathStr)
      argPart = tuple(map((x) -> expToJuliaExpMTK(x, simCode), expLst)...)
      quote
        $(funcName)($(argPart...))
      end
    end
  end
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
Transforms:s
  <name>_index -> <name>[index]
"""
const pattern = r".*_[0-9]+"
function symbolicVariableToArrayRef(e::Expr)::Expr
  MacroTools.postwalk(e) do x
      x isa Symbol || return let
        x
    end
    return let
      if match(pattern, "$x") == nothing
        return x
      end
      local splitted = split("$x", "_")
      local res = Meta.parse("$(splitted[1])[$(splitted[2])]")
      return res
    end
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
    DAE.CREF(cr, _) where SimulationCode.string(cr)=="time" => begin
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
  Converts a DAE expression into a MTK expression.
"""
function expToJuliaExpMTK(@nospecialize(exp::DAE.Exp),
                          simCode::SimulationCode.SIM_CODE,
                          varSuffix="";
                          varPrefix="x", derSymbol::Bool=false)::Expr
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
      DAE.CREF(DAE.CREF_IDENT("time", DAE.T_REAL(nil)), _) => begin
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
        @info "Lookup string:" * lookUpStr
        @info "Replaced with:" * arrName
        indexAndVar = hashTable[lookUpStr]
        hashTable[arrName] = hashTable[lookUpStr]
        expr = quote $(Symbol(arrName)) end
        #fail()
        expr
      end
      #=
      This is an array acess. Note the difference to the case above,
      that is a component of type array.
      In the case above we do not lookup the subscript wheras here it is subscripted.
      =#
      DAE.CREF(DAE.CREF_IDENT(ident, identType, subscriptLst), _) where !isempty(subscriptLst) => begin
        local varName = SimulationCode.string(ident)
        local lookUpStr = ""
        for s in subscriptLst
          @match DAE.INDEX(DAE.ICONST(i)) = s
          lookUpStr *= string("[", i, "]")
        end
        indexAndVar = hashTable[string(varName, lookUpStr)]
        quote
          $(LineNumberNode(@__LINE__, "$varName array"))
          $(Symbol(indexAndVar[2].name))
        end
      end
      DAE.CREF(cr, _)  => begin
        varName = SimulationCode.string(cr)
        indexAndVar = hashTable[varName]
        varKind::SimulationCode.SimVarType = indexAndVar[2].varKind
        @match varKind begin
          SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
          SimulationCode.STATE(__) => quote
            $(LineNumberNode(@__LINE__, "$varName state"))
            $(Symbol(indexAndVar[2].name))
          end
          SimulationCode.PARAMETER(__) => quote
            $(LineNumberNode(@__LINE__, "$varName parameter"))
            $(Symbol(indexAndVar[2].name))
          end
          SimulationCode.ALG_VARIABLE(__) => quote
            $(LineNumberNode(@__LINE__, "$varName, algebraic"))
            $(Symbol(indexAndVar[2].name))
          end
          SimulationCode.DISCRETE(__) => quote
            $(LineNumberNode(@__LINE__, "$varName, discrete"))
            $(Symbol(indexAndVar[2].name))
          end
          SimulationCode.OCC_VARIABLE(__) => quote
            $(LineNumberNode(@__LINE__, "$varName, occ variable"))
            $(Symbol(indexAndVar[2].name))
          end
          SimulationCode.DATA_STRUCTURE(__) => quote
            $(LineNumberNode(@__LINE__, "$varName, datastructure variable"))
            $(Symbol(indexAndVar[2].name))
          end
          _ => begin
            @error "Unsupported varKind: $(varKind)"
            fail()
          end
        end
      end
      DAE.UNARY(operator = op, exp = e1) => begin
        o = DAE_OP_toJuliaOperator(op)
        quote
          $(o)($(expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix)))
        end
      end
      DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, derSymbol = derSymbol)
        local rhs = expToJuliaExpMTK(e2, simCode, varPrefix=varPrefix, derSymbol = derSymbol)
        local op = DAE_OP_toJuliaOperator(op)
        :($op($(lhs), $(rhs)))
      end
      DAE.LUNARY(operator = op, exp = e1)  => begin
        local operand = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, derSymbol = derSymbol)
        local op = DAE_OP_toJuliaOperator(op)
        :($op($(op)))
      end
      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix, derSymbol = derSymbol)
        local rhs = expToJuliaExpMTK(e2, simCode, varPrefix=varPrefix, derSymbol = derSymbol)
        local op = DAE_OP_toJuliaOperator(op)
        quote
          ($op($(lhs), $(rhs)))
        end
      end
      DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpMTK(e1, simCode, varPrefix=varPrefix,derSymbol = derSymbol)
        local rhs = expToJuliaExpMTK(e2, simCode,varPrefix=varPrefix, derSymbol = derSymbol)
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
        try
        #   expr = evalDAEConstant(expCond, simCode)
        #   local expThenJL = expToJuliaExpMTK(expThen, simCode)
          local expElseJL = expToJuliaExpMTK(expElse, simCode)
          quote
            $(expElseJL)
          end
        catch e
          throw(ErrorException("If expressions with variable conditions not allowed in backend code.\n Expression was:\t $(string(exp))"))
        end
      end
      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = explst)  => begin
        #Call as symbol is really ugly.. please fix me :(
        DAECallExpressionToMTKCallExpression(tmpStr, explst, simCode, hashTable; varPrefix=varPrefix, derAsSymbol=derSymbol)
      end
      DAE.CALL(path, expLst) => begin
        local expr = Expr(:call, Symbol(string(path)))
        local args = map(expLst) do arg
          expToJuliaExpMTK(arg, simCode, varPrefix=varPrefix,derSymbol = derSymbol)
        end
        expr.args = vcat(expr.args, args)
        expr
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
      DAE.ARRAY(DAE.T_ARRAY(DAE.T_REAL(Nil(__)), dims), scalar, arr) => begin
        handleArrayExp(exp, simCode)
      end
      DAE.ARRAY(DAE.T_ARRAY(DAE.T_INTEGER(Nil(__)), dims), scalar, arr) => begin
        handleArrayExp(exp, simCode)
      end
    _ =>  throw(ErrorException("$exp not yet supported"))
    end
  end
  return expr
end

"""
  Generate code for array expressions.
This assumes scalarization have been sucessful and that this function does not contain anycrefs
"""
function handleArrayExp(exp::DAE.ARRAY, simCode)
  local arrJL = [] #Type yet unknown
  local steps = listHead(exp.ty.dims)
  local dimSize = length(exp.ty.dims)
  @assert(steps isa DAE.DIM_INTEGER, "Only integer dimensions are currently supported. Type was : $(typeof(steps))")
  steps = steps.integer
  for i in 1:steps
    push!(arrJL, eval(expToJuliaExpMTK(listGet(exp.array, i), simCode)))
  end
  #= Assuming it does not contains crefs =#
  if dimSize >= 2
    arr = transpose(stack(eval(arrJL)))
    quote
      $(arr)
    end
  else
    quote
      $[arrJL...]
    end
  end
end

"""
  If the system needs to conduct index reduction make sure to inform MTK.
(We avoid structurally simplify for now since that might interfer with some other algorithms)
"""
function performStructuralSimplify(simplify)::Expr
  if (simplify)
    #:(reducedSystem = ModelingToolkit.dae_index_lowering(firstOrderSystem))
    #TODO: Report issue when variables are removed from events
    #ModelingToolkit.dae_index_lowering(firstOrderSystem)
    #:(reducedSystem = ModelingToolkit.structural_simplify(firstOrderSystem; simplify = true, allow_parameter=true))
    :(reducedSystem = OMBackend.CodeGeneration.structural_simplify(firstOrderSystem; simplify = true, allow_parameter=true))
  else
    :(reducedSystem = OMBackend.CodeGeneration.structural_simplify(firstOrderSystem; simplify = true, allow_parameter=true))
    #:(reducedSystem = OMBackend.CodeGeneration.structural_simplify(firstOrderSystem; simplify = true, allow_parameter=true))
    #:(reducedSystem = firstOrderSystem)
  end
end

"""
  Generates different constructors for the ODESystem depending on given parameters.
TODO:
Having them as discrete events are currently a workaround...
They should be added as continuous events for continous variables.
TODO:
Clearer seperation of discrete and non discrete if equations
"""
function odeSystemWithEvents(hasEvents, modelName)
  if hasEvents
    :(ODESystem(eqs, t, vars, parameters;
              name=:($(Symbol($modelName))),
              continuous_events = events, guesses = initialValues))
  else
    :(ODESystem(eqs, t, vars, parameters;
              name=:($(Symbol($modelName))), guesses = initialValues))
  end
end


"""
  Given the variable idx and simCode statically decide if this variable is involved in some event.
(TODO: Unused)
"""
function involvedInEvent(idx, simCode)
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
      if ! SimulationCode.isStateOrAlgebraic(simVar)
        @match SimulationCode.SIMVAR(name, _, SimulationCode.PARAMETER(SOME(bindExp)), _) = simVar
        return (bindExp, true, ht)
      else
        @warn "States and Algebraic variables in initial equations is not supported"
        shouldEval = false
        #fail()
        #return (expToJuliaExpMTK(), )
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
  Decide the iv of the condition.
  This currently assumes that the simulation starts at 0.0 (which might not be the case).
  This function needs to be improved so that it also evaluates static parameters.
"""
function evalInitialCondition(mtkCond)
  try
    local mtkCondE = eval(mtkCond)
    local v = substitute(mtkCondE.lhs, t => 0.0)
    local ivCond = (v == 0)
    if ivCond == false
      return ivCond
    end
    if length(v.val.arguments) > 1
      return true
    end
    return ivCond
  catch #= Unable to evaluate at this point in time. =#
    return true
  end
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
  local cond = :( $(Symbol(string("ifCond", identifier, subIdentifier))) == true )
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
function generateCastExpressionMTK(@nospecialize(ty::DAE.Type), @nospecialize(exp::DAE.Exp), simCode, varPrefix)
  return @match ty, exp begin
    (DAE.T_REAL(__), DAE.ICONST(__)) => begin
      quote
        float($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix)))
      end
    end
    (DAE.T_REAL(__), DAE.CREF(cref)) where typeof(cref.identType) === DAE.T_INTEGER => begin
      quote
        float($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix)))
      end
    end
    #= Conversion to a float other alternatives, =#
    (DAE.T_REAL(__), _) => begin
      quote
        float($(expToJuliaExpMTK(exp, simCode, varPrefix=varPrefix)))
      end
    end
    _ => throw("Cast $ty: for exp: $exp not yet supported in codegen!")
  end
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
      local var = last(ht[crefName])
      #= If one variable in the condition is continuous treat it as a conditinous callback =#
      isContinuousCond = isContinuousCond || !(SimulationCode.isDiscrete(var))
    end
  end
  return isContinuousCond
end
