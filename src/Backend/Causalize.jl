#= /*
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

module Causalize

using Setfield
using ExportAll
using MetaModelica

import Absyn
import ..BDAE
import ..BDAEUtil
import ..BackendEquation
import ..@BACKEND_LOGGING
import DAE


"""
    Variable can be: Variable, Discrete, Constant and Parameters
    From this create Algebraic and State Variables (Known variable)
    Traverse all equations and locate the variables that are derived.
    These we mark as states
"""
function detectStates(dae::BDAE.BACKEND_DAE)
  BDAEUtil.mapEqSystems(dae, detectStatesEqSystem)
end

"""
This function detects and removes unused parameters a modeler might have introduced by mistake.
This reduces the explosion of parameters common for instance when using large matrices.
author:johti17
"""
function detectUnusedParametersAndConstants(bdae::BDAE.BACKEND_DAE)
  #= Those that are to be kept have been temporary marked DUMMY_STATE =#
  bdae = BDAEUtil.mapEqSystems(bdae, detectParamsEqSystem)
  @assert length(bdae.eqs) == 1 "Eq systems larger than 1 not supported"
  local sys = first(bdae.eqs)
  #= Remove parameters of all types, but leave complex types alone. =#
  newOrderedVars = filter((x) -> (x.varKind !== BDAE.PARAM() || x.varType isa DAE.T_COMPLEX), sys.orderedVars)
  for (i, v) in enumerate(newOrderedVars)
    if v.varKind === BDAE.DUMMY_STATE()
      tv = newOrderedVars[i]
      @assign tv.varKind = BDAE.PARAM()
      newOrderedVars[i] = tv
    end
  end
  #  println("New ordered vars")
  #  println(map(string, newOrderedVars))
  @assign first(bdae.eqs).orderedVars = newOrderedVars
  return bdae
end


"""
  Replaces all if expressions with a temporary variable.
  Adds an equation assigning this variable to the set of equations.
"""
function detectIfExpressions(dae::BDAE.BACKEND_DAE)
  BDAEUtil.mapEqSystems(dae, detectIfEquationsEqSystem)
end

function detectAndReplaceArrayVariables(dae::BDAE.BDAE.BACKEND_DAE, expandedVariables::Array)
  BDAEUtil.mapEqSystems(dae, replaceArrayVariables, expandedVariables)
end

"""
    kabdelhak:
    Detects all states in the system by looking for component references in
    der() calls.
    Updates all variables with those component references to
    varKind BDAE.STATE()
"""
function detectStatesEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  syst = begin
    local vars::Vector{BDAE.VAR}
    local eqs::Vector{BDAE.Equation}
    local stateCrefs = Dict{DAE.ComponentRef, Bool}()
    @match syst begin
      BDAE.EQSYSTEM(name, vars, eqs, simpleEqs, initialEqs) => begin
        for eq in eqs
          (_, stateCrefs) = BDAEUtil.traverseEquationExpressions(eq, detectStateExpression, stateCrefs)
        end
        #= Do replacements for stateCrefs =#
        @assign syst.orderedVars = updateStates(vars, stateCrefs)
        syst
      end
    end
  end
  return syst
end

"""
 Detect parameters used in the equations.
Save those variables in a HT.
"""
function detectParamsEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  local pars = filter((x) -> x.varKind === BDAE.PARAM(), syst.orderedVars)
  local parStrs = Set(map((x) -> string(x.varName), pars))
  local parStrs2 = map((x) -> string(x.varName) * "|" * string(x.varType), pars)
  local buffer = IOBuffer()
  @BACKEND_LOGGING write("allpars.log", String(take!(buffer)))

  function detectParamExpression(exp::DAE.Exp, paramCrefs::Dict{DAE.ComponentRef, Bool})
    local cont::Bool
    local outCrefs = paramCrefs
    (outCrefs, cont) = begin
      local param::DAE.ComponentRef
      @match exp begin
        #= Ignore complex components =#
        DAE.CREF(c, DAE.T_COMPLEX(__)) => begin
          outCrefs[exp.componentRef] = true
          (outCrefs, true)
        end
        DAE.CREF(__) => begin
          local cand = string(exp.componentRef)
          if (cand in parStrs)
            #println("Located param in the variables:" * cand)
            #println(exp)
            outCrefs[exp.componentRef] = true
          end
          (outCrefs, true)
        end
        DAE.IFEXP(cond, expThen, expElse) => begin
          #fail()
          (outCrefs, true)
        end
        _ => begin
          (outCrefs, true)
        end
      end
    end
    return (exp, cont, outCrefs)
  end

  function updateParams(vars::Vector, paramCrefs::Dict{DAE.ComponentRef, Bool})
    local varArr::Vector{BDAE.VAR} = vars
    for i in 1:arrayLength(varArr)
      varArr[i] = begin
        local cref::DAE.ComponentRef
        local var::BDAE.Var
        @match varArr[i] begin
          var && BDAE.VAR(varName = cref) where (haskey(paramCrefs, cref)) => begin
            @assign var.varKind = BDAE.DUMMY_STATE()#= In the meantime. Mark it for keeping=#
            var
          end
          _ => begin
            varArr[i]
          end
        end
      end
      vars = varArr
    end
    return vars
  end

  syst = begin
    local vars::BDAE.Variables
    local eqs::Array
    local paramCrefs = Dict{DAE.ComponentRef, Bool}()
    @match syst begin
      BDAE.EQSYSTEM(name, vars, eqs, simpleEqs, initialEqs) => begin
        for eq in eqs
          (_, paramCrefs) = BDAEUtil.traverseEquationExpressions(eq, detectParamExpression, paramCrefs)
        end
        #= Go through the initial equations =#
        for ieq in initialEqs
          (_, paramCrefs) = BDAEUtil.traverseEquationExpressions(ieq, detectParamExpression, paramCrefs)
        end
        #= Do replacements for paramCrefs =#
        @assign syst.orderedVars = updateParams(vars, paramCrefs)
        syst
      end
    end
  end
  return syst
end

"""
johti17:
  Detects if-equations.
  Returns new temporary variables and an array of equations
"""
function detectIfEquationsEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  syst = begin
    local vars::BDAE.Variables
    local eqs::Array
    #= Tick is used to keep track of generated if-equations =#
    local tick::Ref{Int} = 0
    local tmpVarToElement = Dict{BDAE.VAR, BDAE.IF_EQUATION}()
    local tmpVarToElementAndTick = (tmpVarToElement, tick)
    @match syst begin
      BDAE.EQSYSTEM(__) => begin
        for i in 1:length(syst.orderedEqs)
          local eq = syst.orderedEqs[i]
          (eq2, _) = BDAEUtil.traverseEquationExpressions(eq, replaceIfExpressionWithTmpVar,
                                                                  tmpVarToElementAndTick)
          if ! (eq === eq2)
            @assign syst.orderedEqs[i] = eq2
          end
          tick.x += 1
        end

        #= Append the new variables to the list of variables =#
        local newVariables = collect(keys(tmpVarToElement))
        local newEquations = collect(values(tmpVarToElement))
        @assign syst.orderedEqs = vcat(syst.orderedEqs, newEquations)
        @assign syst.orderedVars = vcat(syst.orderedVars, newVariables)
        syst
      end
    end
  end
  return syst
end

"""
  Detects if expression.
  We replace the if expression with our temporary variable.
  These variables are assigned in newly created if equations that we add to the tmpVarToElement::Dict.
  We create the mapping:
  tmpVar -> equation it is assigned in
"""
function replaceIfExpressionWithTmpVar(exp::DAE.Exp, tmpVarToElementAndTick::Tuple{Dict, Ref{Int}})
  (newExp, cont, tmpVarToElementAndTick) = begin
    local tmpVarToElement::Dict = first(tmpVarToElementAndTick)
    local tick::Ref{Int} = last(tmpVarToElementAndTick)
    #= NOTE: All these temporary variables are asumed to be REAL numbers for now =#
    local varType = DAE.T_REAL_DEFAULT
    local varName = string("ifEq_tmp", tick.x)
    local var::DAE.ComponentRef = DAE.CREF_IDENT(varName, varType, nil)
    local emptySource = DAE.emptyElementSource
    local attr = BDAE.EQ_ATTR_DEFAULT_UNKNOWN
    @match exp begin
      DAE.IFEXP(cond, expThen, expElse) => begin
        local varAsCREF::DAE.CREF = DAE.CREF(var, varType)
        local backendVar = BDAE.VAR(DAE.CREF_IDENT(varName, DAE.T_UNKNOWN_DEFAULT, nil),
                                    BDAE.VARIABLE(), varType)
        tmpVarToElement[backendVar] = BDAE.IF_EQUATION(list(cond),
                                                       list(list(BDAE.EQUATION(varAsCREF, expThen, emptySource, attr))),
                                                       list(BDAE.EQUATION(varAsCREF, expElse, emptySource, attr)),
                                                       emptySource,
                                                       BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
        (varAsCREF, true, tmpVarToElementAndTick)
      end
      _ => begin
        (exp, true, tmpVarToElementAndTick)
      end
    end
  end
  #= Note we replace the if expression with our temporary variable =#
  return (newExp, cont, tmpVarToElementAndTick)
end

"""
    kabdelhak:
    Detects if a given expression is a der() call and adds the corresponding
    cref to a hashmap
"""
function detectStateExpression(exp::DAE.Exp, stateCrefs::Dict{DAE.ComponentRef, Bool})
  local cont::Bool
  local outCrefs = stateCrefs
  (outCrefs, cont) = begin
    local state::DAE.ComponentRef
    @match exp begin
      DAE.CALL(Absyn.IDENT("der"), DAE.CREF(state) <| _ ) => begin
        #= Add state with boolean value that does not matter,
        it is later only BDAE.BACKEND_DAE(eqs = eqs) checked if it exists at all  =#
        outCrefs[state] = true
        (outCrefs, true)
      end
      _ => begin
        (outCrefs, true)
      end
    end
  end
  return (exp, cont, outCrefs)
end

"""
  Detects if an expression is of type array.
  If it is the case return that expression,
  else if it is not the case return the original exp.
"""
function detectArrayExpression(exp::DAE.Exp, arrayCrefs::Dict{String, Bool})
  local cont::Bool
  local outCrefs = arrayCrefs
  local newExp::DAE.Exp
  @debug "detectArrayExpression for $(exp)"
  (newExp, outCrefs, cont) = begin
    @match exp begin
      DAE.CREF(matchedComponentRef, ty) where typeof(matchedComponentRef.identType) == DAE.T_ARRAY => begin
        newExp = if haskey(arrayCrefs, matchedComponentRef.ident)
          local subscriptStr::String = BDAEUtil.getSubscriptAsUnicodeString(matchedComponentRef.subscriptLst)
          local newName::String = matchedComponentRef.ident + subscriptStr
          local newCref = DAE.CREF_IDENT(newName, ty, nil)
          @debug "Replaced the expression. New type is $(ty)"
          DAE.CREF(newCref, ty)
        else
          @debug "Did not replace the expression"
          exp
        end
        (newExp, outCrefs, true)
      end
      DAE.CALL(Absyn.IDENT("sum"),
               DAE.CREF(matchedComponentRef, ty) <| _ ) where typeof(matchedComponentRef.identType) == DAE.T_ARRAY => begin
                 @debug "We are in sum"
                 newExp = if haskey(arrayCrefs, matchedComponentRef.ident)
                   local subscriptStr = BDAEUtil.getSubscriptAsUnicodeString(matchedComponentRef.subscriptLst)
                   local newName = matchedComponentRef.ident + subscriptStr
                   local newCref = DAE.CREF_IDENT(newName, ty, nil)
                   @debug "Replaced the call "
                   dimLen = listLength(ty.dims)
                   @assert(dimLen == 1)
                   len = listHead(ty.dims).integer
                   arrType = ty.ty
                   #=Add adds=#
                   ident = BDAEUtil.getIntAsUnicodeSubscript(1)
                   local tmp = DAE.CREF(DAE.CREF_IDENT("x$(ident)", arrType, nil), arrType)
                   i = 2
                   for _ in 2:len
                     ident = BDAEUtil.getIntAsUnicodeSubscript(i)
                     tmp = DAE.BINARY(DAE.CREF(DAE.CREF_IDENT("x$(ident)", arrType, nil), arrType), DAE.ADD(arrType), tmp)
                     i += 1
                   end
                   #DAE.CALL(exp.path, list(DAE.CREF(newCref, ty)), exp.attr)
                   tmp
                 else
                   @debug "Did not replace the call expression"
                   exp
                 end
                 (newExp, outCrefs, true)
               end
      DAE.CALL(Absyn.IDENT("product"),
               DAE.CREF(matchedComponentRef, ty) <| _ ) where typeof(matchedComponentRef.identType) == DAE.T_ARRAY => begin
                 @debug "We are in sum"
                 newExp = if haskey(arrayCrefs, matchedComponentRef.ident)
                   local subscriptStr = BDAEUtil.getSubscriptAsUnicodeString(matchedComponentRef.subscriptLst)
                   local newName = matchedComponentRef.ident + subscriptStr
                   local newCref = DAE.CREF_IDENT(newName, ty, nil)
                   @debug "Replaced the call "
                   dimLen = listLength(ty.dims)
                   @assert(dimLen == 1)
                   len = listHead(ty.dims).integer
                   arrType = ty.ty
                   #=Add adds=#
                   ident = BDAEUtil.getIntAsUnicodeSubscript(1)
                   local tmp = DAE.CREF(DAE.CREF_IDENT("x$(ident)", arrType, nil), arrType)
                   i = 2
                   for _ in 2:len
                     ident = BDAEUtil.getIntAsUnicodeSubscript(i)
                     tmp = DAE.BINARY(DAE.CREF(DAE.CREF_IDENT("x$(ident)", arrType, nil), arrType), DAE.MUL(arrType), tmp)
                     i += 1
                   end
                   #DAE.CALL(exp.path, list(DAE.CREF(newCref, ty)), exp.attr)
                   tmp
                 else
                   @debug "Did not replace the call expression"
                   exp
                 end
                 (newExp, outCrefs, true)
               end
      DAE.CALL(Absyn.IDENT("exp"),
               DAE.CREF(matchedComponentRef, ty) <| _ ) where typeof(matchedComponentRef.identType) == DAE.T_ARRAY => begin
                 fail()
                 newExp = if haskey(arrayCrefs, matchedComponentRef.ident)
                   local subscriptStr = BDAEUtil.getSubscriptAsUnicodeString(matchedComponentRef.subscriptLst)
                   local newName = matchedComponentRef.ident + subscriptStr
                   local newCref = DAE.CREF_IDENT(newName, ty, nil)
                   @debug "Replaced the call expression"
                   DAE.CALL(exp.path, list(DAE.CREF(newCref, ty)), exp.attr)
                 else
                   @debug "Did not replace the call expression"
                   exp
                 end
                 (newExp, outCrefs, true)
               end
      #= Any other call that contains an array should be replaced. =#
      DAE.CALL(Absyn.IDENT(__),
               DAE.CREF(matchedComponentRef, ty) <| _ ) where typeof(matchedComponentRef.identType) == DAE.T_ARRAY => begin
                 newExp = if haskey(arrayCrefs, matchedComponentRef.ident)
                   local subscriptStr = BDAEUtil.getSubscriptAsUnicodeString(matchedComponentRef.subscriptLst)
                   local newName = matchedComponentRef.ident + subscriptStr
                   local newCref = DAE.CREF_IDENT(newName, ty, nil)
                   @debug "Replaced the call expression"
                   DAE.CALL(exp.path, list(DAE.CREF(newCref, ty)), exp.attr)
                 else
                   @debug "Did not replace the call expression"
                   exp
                 end
                 (newExp, outCrefs, true)
               end
      _ => begin
        (exp, outCrefs, true)
      end
    end
  end
  return (newExp, cont, outCrefs)
end

"""
  kabdelhak:
    Traverses all variables and uses a hashmap to determine if a variable needs
    to be updated to be a BDAE.STATE()
"""
function updateStates(vars::Vector, stateCrefs::Dict{DAE.ComponentRef, Bool})
  local varArr::Vector{BDAE.VAR} = vars
  for i in 1:arrayLength(varArr)
    varArr[i] = begin
      local cref::DAE.ComponentRef
      local var::BDAE.Var
      @match varArr[i] begin
        var && BDAE.VAR(varName = cref) where (haskey(stateCrefs, cref)) => begin
          @assign var.varKind = BDAE.STATE(0, NONE(), true)
          var
        end
        _ => begin
          varArr[i]
        end
      end
    end
    vars = varArr
  end
  return vars
end


"""
  Author: johti17

"""
function updateArrayCrefs(vars::BDAE.Variables, arrayCrefs::Dict{DAE.ComponentRef, Bool})
  vars = begin
    @match vars begin
      BDAE.VARIABLES(varArr) => begin
        for i in 1:arrayLength(varArr)
          varArr[i] = begin
            local cref::DAE.ComponentRef
            local var::BDAE.Var
            @match varArr[i] begin
              var && BDAE.VAR(varName = cref) where (haskey(arrayCrefs, cref)) => begin
                var
              end
              _ => begin
                varArr[i]
              end
            end
          end
        end
        @assign vars.varArr = varArr
        (vars)
      end
    end
  end
end

"""
    kabdelhak:
    Residualize every equation in each system of the dae by subtracting the rhs
    from the lhs.
    (daeMode)
"""
function residualizeEveryEquation(dae::BDAE.BACKEND_DAE)
  return BDAEUtil.mapEqSystems(dae, makeResidualEquations)
end

"""
    kabdelhak:
    Traverser for daeMode() to map all equations of an equation system
"""
function makeResidualEquations(syst::BDAE.EQSYSTEM)
  syst = BDAEUtil.mapEqSystemEquations(syst, BackendEquation.makeResidualEquation)
end

"""
johti17:
  Expand variables in arrays.
 if x = [x₁, x₂, x₃, x₄] is an array of 4 elements it is replaced by
  x₁, x₂, x₃, x₄.
 The new name for each component is <variable-name>_<index>
"""
function expandArrayVariables(bDAE::BDAE.BACKEND_DAE)::Tuple{BDAE.BACKEND_DAE, Array}
  local systems = bDAE.eqs
  local expandedVars = []
  local newVars = []
  for system in systems
    local orderedVars = system.orderedVars
    local indexOfExpandedVariables = []
    for v in orderedVars
      if typeof(v.varType) == DAE.T_ARRAY
        local dims = v.varType.dims
        local dimIndices = BDAEUtil.getSubscriptAsIntArray(dims)
        #= We know how many variables we are supposed to generate now. =#
        local etype = v.varType.ty
        local varPrototype = v
        local newVarNames = []
        for r in dimIndices
          for i in 1:r
            local newVarName = string(v.varName)
            newVarName *= BDAEUtil.getIntAsUnicodeSubscript(i)
            push!(newVarNames, newVarName)
          end
        end
        for vName in newVarNames
          nV = BDAE.VAR(DAE.CREF_IDENT(vName, etype, nil),
                            varPrototype.varKind,
                            varPrototype.varDirection,
                            etype,
                            varPrototype.bindExp,
                            varPrototype.arryDim,
                            varPrototype.source,
                            varPrototype.values,
                            varPrototype.tearingSelectOption,
                            varPrototype.connectorType,
                            varPrototype.unreplaceable
                            )
          push!(newVars, nV)
        end
        push!(expandedVars, v)
        #= Expanding v=#
        @debug "Expanding:" v
        @debug "Expanded into: $(newVarNames)"
      end
    end
  end
  #= Expanded variables =#
  #= TODO: Consider if there are more than one system =#
  @debug "Expanded variables" string(expandedVars)
  newOrderedVars = setdiff(bDAE.eqs[1].orderedVars, expandedVars)
  newOrderedVars = vcat(newOrderedVars, newVars)
  @assign bDAE.eqs[1].orderedVars = newOrderedVars
  return (bDAE, expandedVars)
end

  @exportAll()
end
