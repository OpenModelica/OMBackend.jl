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

"""
  Creates the structural callbacks
"""
function createStructuralCallbacks(simCode::SC.SIM_CODE, structuralTransitions::Vector{SC.StructuralTransition})
  local structuralCallbacks = Expr[]
  local idx = 1
  for structuralTransisiton in structuralTransitions
    push!(structuralCallbacks, createStructuralCallback(simCode, structuralTransisiton, idx))
    idx += 1
  end
  return structuralCallbacks
end

"""
  Creates a single structural callback for an explicit transition.
"""
function createStructuralCallback(simCode::SC.SIM_CODE, structuralTransition::SC.EXPLICIT_STRUCTURAL_TRANSITION, idx::Int)
  local callbackName = createCallbackName(structuralTransition, 0)
  local transitionCondition = structuralTransition.transitionCondition
  if isContinuousCondition(transitionCondition, simCode)
    local cond = transformToZeroCrossingCondition(transitionCondition)
    quote
      function $(Symbol(callbackName))(destinationSystem, callbacks)
        #= Represents a structural change. =#
        local structuralChange = OMBackend.Runtime.StructuralChange($(structuralTransition.toState), false, destinationSystem, callbacks)
        #= The affect simply activates the structural callback informing us to generate code for a new system =#
        function affect!(integrator)
          structuralChange.structureChanged = true
        end
        function condition(x, t, integrator)
          return $(expToJuliaExp(cond, simCode))
        end
        local cb = ContinuousCallback(condition, affect!)
        return (cb, structuralChange)
      end
    end
  else
    quote
      function $(Symbol(callbackName))(destinationSystem, callbacks)
        #= Represents a structural change. =#
        local structuralChange = OMBackend.Runtime.StructuralChange($(structuralTransition.toState), false, destinationSystem, callbacks)
        #= The affect simply activates the structural callback informing us to generate code for a new system =#
        function affect!(integrator)
          #@info "Potential structural change triggered at  callback:" * $(callbackName) * " at $(integrator.t)"
          structuralChange.structureChanged = true
        end
        function condition(x, t, integrator)
          return $(expToJuliaExp(transitionCondition, simCode))
        end
        local cb = DiscreteCallback(condition, affect!)
        return (cb, structuralChange)
      end
    end
  end
end

"""
  Creates an implicit structural callback where the final state is unknown.
  These structural callback can only occur as a part of a when equation.
  The when equation might also make other changes to the variables before recompilation.
TODO:
Also make sure to create possible other elements in the structural when equation
"""
function createStructuralCallback(simCode::SC.SIM_CODE, structuralTransition::SC.IMPLICIT_STRUCTURAL_TRANSITION, idx::Int)
  local callbackName = createCallbackName(structuralTransition, idx)
  local whenCondition = structuralTransition.whenEquation.condition
  local (whenOperators, recompilationDirective) = createStructuralWhenStatements(structuralTransition.whenEquation.body, simCode)
  local affect::Expr = quote
    function affect!(integrator)
      $(whenOperators...)
      structuralChange.structureChanged = true
      integrator.just_hit_tstop = true
      structuralChange.timeAtChange = integrator.t
      structuralChange.solutionAtChange = deepcopy(integrator.sol)
      terminate!(integrator, ReturnCode.Success)
    end
  end
  local callback = if SC.isBuiltin(whenCondition, :sample)
    #= The interval: a literal is a constant of the code (it cannot change between runs); a
       parameter is read at run time from the `pars` the callback factory takes, so a
       simulation's parameter values (`simulate(tspan, parameters = ...)`) hold, its value
       here the fallback where `pars` does not have it (an alias eliminated, renamed). =#
    local interval = whenCondition.args[2]
    local Δt_code = if interval isa Union{SC.RCONST, SC.ICONST}
      :($(Float64(interval.value)))
    elseif interval isa SC.EXP_CREF
      local cname = SC.crefName(interval.cref)
      local fallbackVal = evalSimCodeParameter(last(simCode.stringToSimVarHT[cname]), simCode)
      quote
        let _resolved = nothing
          for (_k, _v) in pars
            if string(_k) == $(cname)
              _resolved = _v
              break
            end
          end
          isnothing(_resolved) ? $(fallbackVal) : _resolved
        end
      end
    else
      :($(evalConstant(interval, simCode)))
    end
    quote
      $(affect)
      local _Δt = $(Δt_code)
      local cb = PeriodicCallback(affect!, _Δt)
    end
  elseif isContinuousCondition(whenCondition, simCode)
    local zeroCrossingCond = transformToZeroCrossingCondition(whenCondition)
    quote
      $(affect)
      function condition(u, t, integrator)
        return $(replaceVars(expToJuliaExpMTK(zeroCrossingCond, simCode);
                             integratorCref = "integrator",
                             prefix = "[",
                             suffix = "]",
                             ht = simCode.stringToSimVarHT,
                             useIndexedU = true,
                             useMTKIdx = true))
      end
      local cb = ContinuousCallback(condition, affect!)
    end
  else
    quote
      $(affect)
      function condition(u, t, integrator)
        return $(expToJuliaExpMTK(whenCondition, simCode))
      end
      local cb = ContinuousCallback(condition, affect!)
    end
  end
  #=
    Construct the specified structural change.
    Dispatch on whether this is a standard recompilation or an agentic one.
  =#
  local metaModel = simCode.frontendModel
  metaModel === nothing && unsupported("a structural change of a model without its frontend model", simCode.name)
  local structuralCallback = if recompilationDirective isa SC.AGENTIC_RECOMPILATION
    #= Agentic recompilation: new values come from the external agent at runtime =#
    local componentsToChange = String[SC.crefName(c) for c in recompilationDirective.componentsToChange]
    local promptVal = recompilationDirective.prompt
    local initEqVal = recompilationDirective.initialEquations
    quote
      function $(Symbol(callbackName))(reducedSystem, pars)
        #= Build a name→MTK-index map so condition functions use the correct u[i].
           MTK may reorder unknowns relative to the SimCode state indices. =#
        local _mtk_idx = Dict{String, Int}(
          split(string(v), "(")[1] => i
          for (i, v) in enumerate(ModelingToolkit.unknowns(reducedSystem)))
        stringToSimVarHT = $(simCode.stringToSimVarHT)
        local structuralChange = OMBackend.Runtime.StructuralChangeAgenticRecompilation(
          $(simCode.name),
          false,
          $(metaModel),
          $(componentsToChange),
          stringToSimVarHT,
          0.0,
          Float64[],
          $(promptVal),
          $(initEqVal))
        $(callback)
        return (cb, structuralChange)
      end
    end
  else
    #= Standard recompilation: the new value is a Modelica expression; a parameter's is its
       binding, another evaluated with the component's current value =#
    local componentToModify = SC.crefName(recompilationDirective.componentToChange)
    local newValueExp = recompilationDirective.newValue
    local newValue::Expr = if newValueExp isa SC.EXP_CREF
      local variableSpec = last(simCode.stringToSimVarHT[SC.crefName(newValueExp.cref)])
      (variableSpec.varKind isa SC.PARAMETER && variableSpec.varKind.bindExp !== nothing) ||
        unsupported("a structural change to a variable other than a bound parameter", variableSpec.name)
      expToJuliaExpMTK(variableSpec.varKind.bindExp, simCode)
    else
      quote
        local variableSpec = last(stringToSimVarHT[$(componentToModify)])
        $(Symbol(componentToModify)) = OMBackend.CodeGeneration.evalConstant(variableSpec.varKind.bindExp)
        $(expToJuliaExpMTK(newValueExp, simCode))
      end
    end
    local modification = quote
      ($(componentToModify), $("$(newValue)"))
    end
    quote
      function $(Symbol(callbackName))(reducedSystem, pars)
        local _mtk_idx = Dict{String, Int}(
          split(string(v), "(")[1] => i
          for (i, v) in enumerate(ModelingToolkit.unknowns(reducedSystem)))
        stringToSimVarHT = $(simCode.stringToSimVarHT)
        local modification::Tuple{String, String} = $(modification)
        local structuralChange = OMBackend.Runtime.StructuralChangeRecompilation($(simCode.name),
                                                                                 false,
                                                                                 $(metaModel),
                                                                                 modification,
                                                                                 stringToSimVarHT,
                                                                                 0.0,
                                                                                 Float64[])
        $(callback)
        return (cb, structuralChange)
      end
    end
  end
  return structuralCallback
end

"""
  Creates the supermodel that composes one or more submodels.
  This is to allow the model to modify itself.
"""
function createStructuralAssignments(simCode::SC.SIM_CODE, structuralTransitions::Vector{SC.StructuralTransition})
  local structuralAssignments = Expr[]
  for (idx, transition) in enumerate(structuralTransitions)
    push!(structuralAssignments, transition isa SC.EXPLICIT_STRUCTURAL_TRANSITION ?
      createStructuralAssignment(simCode, transition) : createStructuralAssignment(simCode, transition, idx))
  end
  result = quote
    structuralCallbacks = OMBackend.Runtime.AbstractStructuralChange[]
    callbackSet = Any[]
    $(structuralAssignments...)
  end
  return result
end

"""
  This function creates a structural assignment.
  That is the constructor for a structural callback guiding structural change.
"""
function createStructuralAssignment(simCode::SC.SIM_CODE, structuralTransition::SC.EXPLICIT_STRUCTURAL_TRANSITION)
  local callbackName = createCallbackName(structuralTransition)
  local toState = structuralTransition.toState
  local toStateProblem = Symbol(toState * "Problem")
  local toStateModel = Symbol(toState * "Model")
  local integratorCallbackName = structuralTransition.fromState * structuralTransition.toState * "_CALLBACK"
  local structuralChangeStructure = structuralTransition.fromState * structuralTransition.toState * "_STRUCTURAL_CHANGE"
  quote
    ($(toStateProblem), callbacks, _, _, _, _) = ($(toStateModel))(tspan)
    ($(Symbol(integratorCallbackName)), $(Symbol(structuralChangeStructure))) = $(Symbol(callbackName))($(toStateProblem), callbacks)
    push!(structuralCallbacks, $(Symbol(structuralChangeStructure )))
    push!(callbackSet, ($(Symbol(integratorCallbackName))))
  end
end

"""
  Creates a structural assignment for an implicit structural transition.
  These are numbered from 1->N
"""
function createStructuralAssignment(simCode::SC.SIM_CODE, structuralTransition::SC.IMPLICIT_STRUCTURAL_TRANSITION, idx::Int)
  local callbackName = createCallbackName(structuralTransition, idx)
  local integratorCallbackName = string(callbackName, "_CALLBACK")
  local structuralChangeStructure = string(callbackName, "_STRUCTURAL_CHANGE")
  quote
    ($(Symbol(integratorCallbackName)), $(Symbol(structuralChangeStructure))) = $(Symbol(callbackName))(reducedSystem, pars)
    push!(structuralCallbacks, $(Symbol(structuralChangeStructure)))
    push!(callbackSet, ($(Symbol(integratorCallbackName))))
  end
end

function createCallbackName(structuralTransisiton::SC.EXPLICIT_STRUCTURAL_TRANSITION, idx::Int = 0)
  return "structuralCallback" * structuralTransisiton.fromState * structuralTransisiton.toState
end

"""
  Creates a structural callback for the when equation.
  The name is up to change.
"""
function createCallbackName(structuralTransisiton::SC.IMPLICIT_STRUCTURAL_TRANSITION, idx::Int)
  return string("structuralCallbackWhenEquation", idx)
end

"""
  Creates the variables that are shared between the structural modes of a model.
"""
function createCommonVariables(commonVariables::Vector{String})
  commonVariablesExpr = Expr[]
  for variable in commonVariables
    res = :(
      push!(commonVariables, $variable)
    )
    push!(commonVariablesExpr, res)
  end
  return quote
    commonVariables = String[]
    $(commonVariablesExpr...)
  end
end

"""
  Generates statements for the structural when equation construct.
  This function returns a tuple where the first part is a vector of statements occurring in the when equation.
  The second part is the recompilation statement itself that specifies what structural changes are to occur.
  This last part is then used by the runtime to just-in-time compile the model when the event occurs.
"""
function createStructuralWhenStatements(whenStatements::Vector{SC.Statement}, simCode::SC.SIM_CODE)
  local res = Expr[]
  local recompilationOperator = nothing
  for wStmt in whenStatements
    if wStmt isa SC.ASSIGN
      push!(res, :($(expToJuliaExp(wStmt.left, simCode; varPrefix = "integrator.u")) = $(expToJuliaExp(wStmt.right, simCode))))
    elseif wStmt isa Union{SC.RECOMPILATION, SC.AGENTIC_RECOMPILATION}
      recompilationOperator = wStmt
    elseif wStmt isa SC.REINIT
      unsupported("reinit in a structural when-equation", wStmt)
    else
      unsupported("statement in a when equation", wStmt)
    end
  end
  recompilationOperator === nothing && unsupported("a structural when-equation without a recompilation", whenStatements)
  return (res, recompilationOperator)
end
