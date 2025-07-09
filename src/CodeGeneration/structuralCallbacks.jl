#=
# This file is part of OpenModelica.
#
# Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
# c/o Linköpings universitet, Department of Computer and Information Science,
# SE-58183 Linköping, Sweden.
#
# All rights reserved.
#
# THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
# THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
# ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
# RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
# ACCORDING TO RECIPIENTS CHOICE.
#
# The OpenModelica software and the Open Source Modelica
# Consortium (OSMC) Public License (OSMC-PL) are obtained
# from OSMC, either from the above address,
# from the URLs: http:www.ida.liu.se/projects/OpenModelica or
# http:www.openmodelica.org, and in the OpenModelica distribution.
# GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
#
# This program is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of  MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
# IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
#
# See the full OSMC Public License conditions for more details.
#
  Author: John Tinnerholm, john.tinnerholm@liu.se
=#

"""
  Creates the structural callbacks
"""
function createStructuralCallbacks(simCode, structuralTransitions::Vector{ST}) where {ST}
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
function createStructuralCallback(simCode, simCodeStructuralTransition::SimulationCode.EXPLICIT_STRUCTURAL_TRANSISTION, idx)
  local structuralTransition = simCodeStructuralTransition.structuralTransition
  local callbackName = createCallbackName(structuralTransition, 0)

  if isContinousCondition(structuralTransition.transistionCondition, simCode)
    local cond = transformToZeroCrossingCondition(structuralTransition.transistionCondition)
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
          @info "Potential structural change triggered at  callback:" * $(callbackName) * " at $(integrator.t)"
          structuralChange.structureChanged = true
        end
        function condition(x, t, integrator)
          return $(expToJuliaExp(structuralTransition.transistionCondition, simCode))
        end
        local cb = DiscreteCallback(condition, affect!)
        return (cb, structuralChange)
      end
    end
  end
end

"""
  For dynamic overconstrained connectors.
"""
function createStructuralCallback(simCode,
                                  simCodeStructuralTransition::SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION,
                                  idx)
  local structuralTransition = simCodeStructuralTransition.structuralDOCC_equation
  local callbackName = createCallbackName(structuralTransition, idx)
  (equationsToAddOnTrue, cond) = extractTransitionEquationBody(structuralTransition)
  #=
  The system structure is changed when the new equations are added.
    In this case we know how the equations look like.
    On true the structure of the system is the one with the equations of the branch active.
    On false that same system shall change slightly.
  =#
  @match SOME(flatModel) = simCode.flatModel
  unresolvedFlatModel = createNewFlatModel(flatModel)
  #= Note in this way we can't print the flat model since we have some circular references... =#
  global FLAT_MODEL = unresolvedFlatModel #= To be referenced as OM.OMBackend.CodeGeneration.FLAT_MODEL =#
  global OLD_FLAT_MODEL = flatModel
  quote
    import OMBackend.CodeGeneration
    function $(Symbol(callbackName))()
      #= Represent structural change. =#
      local stringToSimVarHT = $(simCode.stringToSimVarHT)
      local structuralChange = OMBackend.Runtime.StructuralChangeDynamicConnection($(flatModel.name),
                                                                                   false,
                                                                                   OMBackend.CodeGeneration.FLAT_MODEL,
                                                                                   $(idx), #= Assumes specific ordering =#
                                                                                   stringToSimVarHT,
                                                                                   $(flatModel.active_DOCC_Equations[idx]))
      #= The affect simply activates the structural callback informing us to generate code for a new system =#
      $(createAffectCondPairForDOCC(cond, idx, flatModel.active_DOCC_Equations, simCode))
      local cb = DiscreteCallback(condition,
                                  affect!;
                                  save_positions=(true, true))
      return (cb, structuralChange)
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
function createStructuralCallback(simCode,
                                  simCodeStructuralTransition::SimulationCode.IMPLICIT_STRUCTURAL_TRANSISTION,
                                  idx)
  local structuralTransition = simCodeStructuralTransition.structuralWhenEquation
  local callbackName = createCallbackName(structuralTransition, idx)
  local whenCondition = structuralTransition.whenEquation.condition
  local stmtLst = structuralTransition.whenEquation.whenStmtLst
  local stringToSimVarHT = simCode.stringToSimVarHT
  (whenOperators, recompilationDirective) = createStructuralWhenStatements(stmtLst, simCode)
  local affect::Expr = quote
    function affect!(integrator)
      #= Expand the when operators =#
      $(whenOperators...)
      #println("Callback triggered at $(integrator.t)")
      #println("tprev at $(integrator.tprev)")
      structuralChange.structureChanged = true
      integrator.just_hit_tstop = true
      structuralChange.timeAtChange = integrator.t
      structuralChange.solutionAtChange = integrator.sol
      terminate!(integrator, ReturnCode.Success)
    end
  end
  callback = @match whenCondition begin
    DAE.CALL(Absyn.IDENT("sample"), args, attrs) => begin
      @match start <| interval <| tail = args
      quote
        $(affect)
        Δt = $(expToJuliaExpMTK(interval, simCode))
        local cb = PeriodicCallback(affect!, Δt)
      end
    end
    _ #=Continuous  or discrete =# => begin
      if isContinousCondition(whenCondition, simCode)
        local zeroCrossingCond = transformToZeroCrossingCondition(whenCondition)
        quote
          $(affect)
          function condition(u, t, integrator)
            return $(expToJuliaExpMTK(zeroCrossingCond, simCode))
          end
          local cb = ContinuousCallback(condition, affect!)
        end
      else #= Discrete =#
        quote
          $(affect)
          function condition(u, t, integrator)
            return $(expToJuliaExpMTK(whenCondition, simCode))
          end
          local cb = ContinuousCallback(condition, affect!)
        end

      end
    end
  end
  #=
    Construct the specified structural change as a pair.
    For now assume that we only change some parameter.
  =#
  local componentToModify = string(recompilationDirective.componentToChange)
  local newValue::Expr = if typeof(recompilationDirective.newValue) === DAE.CREF
    #= The type we are modifying is either a parameter or a constant =#
    local variableSpec = last(simCode.stringToSimVarHT[string(recompilationDirective.newValue)])
    @match SimulationCode.SIMVAR(name, index, SimulationCode.PARAMETER(SOME(bindExp)), _) =  variableSpec
    #= Above might throw. In that case it is an error.=#
    expToJuliaExpMTK(bindExp, simCode)
  else
    newValue = expToJuliaExpMTK(recompilationDirective.newValue, simCode)
    #= Get the component we are currently modifying =#
    evalExpr = quote
        variableSpec = last(stringToSimVarHT[$(componentToModify)])
        @match SimulationCode.SIMVAR(name, index, SimulationCode.PARAMETER(SOME(bindExp)), _) =  variableSpec
        parameterVal = OMBackend.CodeGeneration.evalDAEConstant(bindExp)
        $(Symbol(componentToModify)) = parameterVal
        $(newValue)
      end
    #expToJuliaExpMTK(evalExpr, simCode)
    evalExpr
  end
  modification = quote
    ($(componentToModify), $("$(newValue)"))
  end
  @match SOME(metaModel) = simCode.metaModel
  structuralCallback = quote
    function $(Symbol(callbackName))()
      #= The recompilation directive. =#
      stringToSimVarHT = $(simCode.stringToSimVarHT)
      #=
      We quote the modification.
      This is to evaluate it in the correct context later when recompiling.
      =#
      local modification::Tuple{String, String} = $(modification)
      #= Represent structural change. =#
      #=NOTE: For the implicit callbacks the model is assumed to be the same=#
      local structuralChange = OMBackend.Runtime.StructuralChangeRecompilation($(simCode.name),
                                                                               false,
                                                                               $(metaModel),
                                                                               modification,
                                                                               stringToSimVarHT,
                                                                               0.0,
                                                                               Float64[])
      #=
        TODO, here we need to change the code generation
        to encompass other types of callbacks not based on zero crossing functions...
      =#
      #= The affect simply activates the structural callback informing us to generate code for a new system =#
      $(callback)
      return (cb, structuralChange)
    end
  end
  return structuralCallback
end

"""
  Creates the supermodel that composes one or more submodels.
  This is to allow the model to modify itself.
"""
function createStructuralAssignments(simCode, structuralTransitions::Vector{ST}) where {ST}
  local structuralAssignments = Expr[]
  local idx = 1
  for structuralTransisiton in structuralTransitions
    @match structuralTransisiton begin
      SimulationCode.EXPLICIT_STRUCTURAL_TRANSISTION(__) => begin
        push!(structuralAssignments, createStructuralAssignment(simCode, structuralTransisiton))
      end
      SimulationCode.IMPLICIT_STRUCTURAL_TRANSISTION(__) => begin
        push!(structuralAssignments, createStructuralAssignment(simCode, structuralTransisiton, idx))
      end
      SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION(__) => begin
        push!(structuralAssignments, createStructuralAssignment(simCode, structuralTransisiton, idx))
      end
    end
    idx += 1
  end
  result = quote
    structuralCallbacks = OMBackend.Runtime.AbstractStructuralChange[]
    callbackSet = []
    $(structuralAssignments...)
  end
  return result
end

"""
  This function creates a structural assignment.
  That is the constructor for a structural callback guiding structural change.
"""
function createStructuralAssignment(simCode, simCodeStructuralTransition::SimulationCode.EXPLICIT_STRUCTURAL_TRANSISTION)
  local structuralTransition = simCodeStructuralTransition.structuralTransition
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
  Creates a structural assignment for an implicit structural transisiton.
  These are numbered from 1->N
"""
function createStructuralAssignment(simCode, simCodeStructuralTransition::SimulationCode.IMPLICIT_STRUCTURAL_TRANSISTION, idx::Int)
  local structuralTransition = simCodeStructuralTransition.structuralWhenEquation
  local callbackName = createCallbackName(structuralTransition, idx)
  local integratorCallbackName = string(callbackName, "_CALLBACK")
  local structuralChangeStructure = string(callbackName, "_STRUCTURAL_CHANGE")
  quote
    ($(Symbol(integratorCallbackName)), $(Symbol(structuralChangeStructure))) = $(Symbol(callbackName))()
    push!(structuralCallbacks, $(Symbol(structuralChangeStructure)))
    push!(callbackSet, ($(Symbol(integratorCallbackName))))
  end
end

function createStructuralAssignment(simCode, simCodeStructuralTransition::SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION, idx::Int)
  local structuralTransition = simCodeStructuralTransition.structuralDOCC_equation
  local callbackName = createCallbackName(structuralTransition, idx)
  local integratorCallbackName = string(callbackName,  "_CALLBACK")
  local structuralChangeStructure = string(callbackName, "_STRUCTURAL_CHANGE")
  quote
    ($(Symbol(integratorCallbackName)), $(Symbol(structuralChangeStructure))) = $(Symbol(callbackName))()
    push!(structuralCallbacks, $(Symbol(structuralChangeStructure)))
    push!(callbackSet, ($(Symbol(integratorCallbackName))))
  end
end

function createCallbackName(structuralTransisiton::BDAE.STRUCTURAL_TRANSISTION, idx = 0)
  return "structuralCallback" * structuralTransisiton.fromState * structuralTransisiton.toState
end

"""
  Creates a structural callback for the when equation.
  The name is up to change.
"""
function createCallbackName(structuralTransisiton::BDAE.STRUCTURAL_WHEN_EQUATION, idx::Int)
  return string("structuralCallbackWhenEquation", idx)
end

function createCallbackName(structuralTransisiton::BDAE.STRUCTURAL_IF_EQUATION, idx::Int)
  return string("structuralCallbackDynamicConnectEquation", idx)
end

"""
  Creates the variables that are shared between the structural modes of a model.
"""
function createCommonVariables(commonVariables)
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
  This function returns a tuple where the first part is a vector of statements occuring in the when equation.
  The second part is the recompilation statement itself that specifies what structural changes are to occur.
  This last part is then used by the runtime to just-in-time compile the model when the event occurs.
"""
function createStructuralWhenStatements(@nospecialize(whenStatements::List{BDAE.WhenOperator}),
                                        simCode::SimulationCode.SIM_CODE)
  local res::Vector{Expr} = Expr[]
  local recompilationOperator
  for wStmt in  whenStatements
    @match wStmt begin
      BDAE.ASSIGN(__) => begin
        exp1 = expToJuliaExp(wStmt.left, simCode, varPrefix="integrator.u")
        exp2 = expToJuliaExp(wStmt.right, simCode)
        push!(res, :($(exp1) = $(exp2)))
      end
      BDAE.RECOMPILATION(__) => begin
        recompilationOperator = wStmt
      end
      BDAE.REINIT(__) => begin
        throw("Reinit is not allowed in a structural when equation")
      end
      _ => begin
        throw("Unsupported statement in the when equation:" * string(wStmt))
      end
    end
  end
  return (res, recompilationOperator)
end

"""
  Returns the equations and the condition
"""
function extractTransitionEquationBody(structuralTransition)
  local ifEquation = structuralTransition.ifEquation
  local structuralTransisitonAsDAE = listHead(OMFrontend.Frontend.convertEquation(ifEquation, nil))
  #= For now assumed to only allow a single statement. No else. =#
  @assert length(structuralTransisitonAsDAE.condition1) == 1
  @assert length(ifEquation.branches) == 1
  local cond = listHead(structuralTransisitonAsDAE.condition1)
  local branch = listHead(ifEquation.branches)
  local bodyEquations = branch.body
  return (bodyEquations, cond)
end


"""
  Creates a flat model without the connectors expanded.
  The equations in this model does not include active DOCC equations.
"""
function createNewFlatModel(flatModel)
  local newFlatModel = OMFrontend.Frontend.FLAT_MODEL(flatModel.name,
                                                  flatModel.variables,
                                                  flatModel.unresolvedConnectEquations, #Why is this in two places?
                                                  flatModel.initialEquations,
                                                  flatModel.algorithms,
                                                  flatModel.initialAlgorithms,
                                                  nil,
                                                  NONE(),
                                                  flatModel.DOCC_equations,
                                                  flatModel.unresolvedConnectEquations,
                                                  flatModel.active_DOCC_Equations,
                                                  flatModel.comment)
  return newFlatModel
end

"""
  If equation start as active it should be removed and the condition should be reverted.
  If we start without the equation  equations for DOCC should be added.
"""
function createAffectCondPairForDOCC(cond,
                                     idx::Int,
                                     active_DOCC_Equations::Vector{Bool},
                                     simCode)
  affectCondPair = if ! active_DOCC_Equations[idx]
    quote
      function affect!(integrator)
        #@info "Structural callback at:" integrator.t
        #@info "Structural callback Δt" integrator.dt
        local t = integrator.t
        local x = integrator.u
        structuralChange.structureChanged = true
        $(expToJuliaExp(cond, simCode)) = false
        auto_dt_reset!(integrator)
        add_tstop!(integrator, integrator.t #= Some small number =#)
        #set_!(integrator, t)
      end
      function condition(x, t, integrator)
        return Bool($(expToJuliaExp(cond, simCode)))
      end
    end
  else #= The equation is active at the start =#
    quote
      function affect!(integrator)
        #@info "Structural callback at:" integrator.t
        #@info "Structural callback Δt" integrator.dt
        local t = integrator.t
        local x = integrator.u
        structuralChange.structureChanged = true
        $(expToJuliaExp(cond, simCode)) = true
        #= Stop integration =#
        terminate!(integrator)
      end
      function condition(x, t, integrator)
        return Bool($(expToJuliaExp(cond, simCode))) == false
      end
    end
  end
  return affectCondPair
end
