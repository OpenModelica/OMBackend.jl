#=
# This file is part of OpenModelica.
#
# Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
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
  local structuralTransition = simCodeStructuralTransition
  local callbackName = createCallbackName(structuralTransition, 0)
  # SIM.Exp -> DAE.Exp at the boundary; DAE-typed helpers below.
  local conditionDAE = SimulationCode.toDAEExp(structuralTransition.transistionCondition)

  if isContinousCondition(conditionDAE, simCode)
    local cond = transformToZeroCrossingCondition(conditionDAE)
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
          return $(expToJuliaExp(conditionDAE, simCode))
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
  local structuralTransition = simCodeStructuralTransition
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
  #= Note: the flat model has circular references, so we cannot print it.
     We embed it directly into the generated callback via $-interpolation
     so the value rides into the StructuralChangeDynamicConnection.flatModel
     field at codegen time. The runtime then reads it from the struct field
     instead of from a module-scope global. =#
  quote
    function $(Symbol(callbackName))(reducedSystem)
      #= Represent structural change. =#
      local stringToSimVarHT = $(simCode.stringToSimVarHT)
      local structuralChange = OMBackend.Runtime.StructuralChangeDynamicConnection($(flatModel.name),
                                                                                   false,
                                                                                   $(unresolvedFlatModel),
                                                                                   $(idx), #= Assumes specific ordering =#
                                                                                   stringToSimVarHT,
                                                                                   $(flatModel.active_DOCC_Equations[idx]),
                                                                                   0.0,
                                                                                   nothing)
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
  local structuralTransition = simCodeStructuralTransition
  local callbackName = createCallbackName(structuralTransition, idx)
  local whenCondition = SimulationCode.toDAEExp(structuralTransition.whenEquation.condition)
  local stmtLst = structuralTransition.whenEquation.whenStmtLst
  local stringToSimVarHT = simCode.stringToSimVarHT
  (whenOperators, recompilationDirective) = createStructuralWhenStatements(stmtLst, simCode)
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
  callback = @match whenCondition begin
    DAE.CALL(Absyn.IDENT("sample"), args, attrs) => begin
      @match start <| interval <| tail = args
      #= Resolve the interval at callback construction time. Literal RCONST/ICONST are
         baked as compile-time constants (they cannot change between runs). Parameter
         CREFs are resolved at runtime from the `pars` dict threaded into the callback
         factory, so per-simulation parameter overrides (`simulate(tspan, parameters=...)`)
         are honored. Falls back to the simCode-resolved value if the parameter is not
         present in `pars` (e.g. alias-eliminated or renamed). =#
      local Δt_code = @match interval begin
        DAE.RCONST(r) => :( $(r) )
        DAE.ICONST(i) => :( $(Float64(i)) )
        DAE.CREF(cr, _) => begin
          local cname = SimulationCode.DAE_identifierToString(cr)
          local fallbackVal = begin
            local sv = last(simCode.stringToSimVarHT[cname])
            evalSimCodeParameter(sv, simCode)
          end
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
        end
        _ => :( $(evalDAEConstant(interval, simCode)) )
      end
      quote
        $(affect)
        local _Δt = $(Δt_code)
        local cb = PeriodicCallback(affect!, _Δt)
      end
    end
    _ #=Continuous or discrete =# => begin
      if isContinousCondition(whenCondition, simCode)
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
    Construct the specified structural change.
    Dispatch on whether this is a standard recompilation or an agentic one.
  =#
  @match SOME(metaModel) = simCode.metaModel
  structuralCallback = if typeof(recompilationDirective) === BDAE.AGENTIC_RECOMPILATION ||
                          typeof(recompilationDirective) === SimulationCode.AGENTIC_RECOMPILATION
    #= Agentic recompilation: new values come from the external agent at runtime =#
    local componentsToChange = [string(c) for c in recompilationDirective.componentsToChange]
    local promptVal = if isSome(recompilationDirective.prompt)
      @match SOME(s) = recompilationDirective.prompt
      s
    else
      nothing
    end
    local initEqVal = if isSome(recompilationDirective.initialEquations)
      @match SOME(s) = recompilationDirective.initialEquations
      s
    else
      nothing
    end
    quote
      function $(Symbol(callbackName))(reducedSystem, pars)
        #= Build a name→MTK-index map so condition functions use the correct u[i].
           MTK may reorder unknowns relative to the BDAE state indices. =#
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
    #= Standard recompilation: new value is determined by a Modelica expression =#
    local componentToModify = string(recompilationDirective.componentToChange)
    # RECOMPILATION.newValue is ::Exp post-migration; collapse to DAE.Exp for both
    # the cref-as-string lookup and the codegen call.
    local newValueDAE = SimulationCode.toDAEExp(recompilationDirective.newValue)
    local newValue::Expr = if typeof(newValueDAE) === DAE.CREF
      local variableSpec = last(simCode.stringToSimVarHT[string(newValueDAE)])
      @match SimulationCode.SIMVAR(name, index, SimulationCode.PARAMETER(SOME(bindExp)), _) = variableSpec
      expToJuliaExpMTK(bindExp, simCode)
    else
      newValue = expToJuliaExpMTK(newValueDAE, simCode)
      evalExpr = quote
          variableSpec = last(stringToSimVarHT[$(componentToModify)])
          @match SimulationCode.SIMVAR(name, index, SimulationCode.PARAMETER(SOME(bindExp)), _) = variableSpec
          parameterVal = OMBackend.CodeGeneration.evalDAEConstant(bindExp)
          $(Symbol(componentToModify)) = parameterVal
          $(newValue)
        end
      evalExpr
    end
    modification = quote
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
  local structuralTransition = simCodeStructuralTransition
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
function createStructuralAssignment(simCode, simCodeStructuralTransition::SimulationCode.IMPLICIT_STRUCTURAL_TRANSISTION, idx::Int)
  local structuralTransition = simCodeStructuralTransition
  local callbackName = createCallbackName(structuralTransition, idx)
  local integratorCallbackName = string(callbackName, "_CALLBACK")
  local structuralChangeStructure = string(callbackName, "_STRUCTURAL_CHANGE")
  quote
    ($(Symbol(integratorCallbackName)), $(Symbol(structuralChangeStructure))) = $(Symbol(callbackName))(reducedSystem, pars)
    push!(structuralCallbacks, $(Symbol(structuralChangeStructure)))
    push!(callbackSet, ($(Symbol(integratorCallbackName))))
  end
end

function createStructuralAssignment(simCode, simCodeStructuralTransition::SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION, idx::Int)
  local structuralTransition = simCodeStructuralTransition
  local callbackName = createCallbackName(structuralTransition, idx)
  local integratorCallbackName = string(callbackName,  "_CALLBACK")
  local structuralChangeStructure = string(callbackName, "_STRUCTURAL_CHANGE")
  quote
    ($(Symbol(integratorCallbackName)), $(Symbol(structuralChangeStructure))) = $(Symbol(callbackName))(reducedSystem)
    push!(structuralCallbacks, $(Symbol(structuralChangeStructure)))
    push!(callbackSet, ($(Symbol(integratorCallbackName))))
  end
end

function createCallbackName(structuralTransisiton::SimulationCode.EXPLICIT_STRUCTURAL_TRANSISTION, idx = 0)
  return "structuralCallback" * structuralTransisiton.fromState * structuralTransisiton.toState
end

"""
  Creates a structural callback for the when equation.
  The name is up to change.
"""
function createCallbackName(structuralTransisiton::SimulationCode.IMPLICIT_STRUCTURAL_TRANSISTION, idx::Int)
  return string("structuralCallbackWhenEquation", idx)
end

function createCallbackName(structuralTransisiton::SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION, idx::Int)
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
  This function returns a tuple where the first part is a vector of statements occurring in the when equation.
  The second part is the recompilation statement itself that specifies what structural changes are to occur.
  This last part is then used by the runtime to just-in-time compile the model when the event occurs.
"""
function createStructuralWhenStatements(whenStatements,
                                        simCode::SimulationCode.SIM_CODE)
  local res::Vector{Expr} = Expr[]
  local recompilationOperator
  for wStmt in whenStatements
    if wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
      exp1 = expToJuliaExp(wStmt.left, simCode, varPrefix="integrator.u")
      exp2 = expToJuliaExp(wStmt.right, simCode)
      push!(res, :($(exp1) = $(exp2)))
    elseif wStmt isa BDAE.RECOMPILATION || wStmt isa SimulationCode.RECOMPILATION
      recompilationOperator = wStmt
    elseif wStmt isa BDAE.AGENTIC_RECOMPILATION || wStmt isa SimulationCode.AGENTIC_RECOMPILATION
      recompilationOperator = wStmt
    elseif wStmt isa BDAE.REINIT || wStmt isa SimulationCode.REINIT
      throw("Reinit is not allowed in a structural when equation")
    else
      throw("Unsupported statement in the when equation:" * string(wStmt))
    end
  end
  return (res, recompilationOperator)
end

"""
  Returns the equations and the condition
"""
function extractTransitionEquationBody(structuralTransition)
  local ifEquation = structuralTransition.ifEquation
  local structuralTransisitonAsDAE = listHead(OMFrontend.Frontend.convertEquation(ifEquation, MetaModelica.nil))
  #= For now assumed to only allow a single statement. No else. =#
  @assert length(structuralTransisitonAsDAE.condition1) == 1
  @assert length(ifEquation.branches) == 1
  local cond = listHead(structuralTransisitonAsDAE.condition1)
  local branch = first(ifEquation.branches)
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
                                                  MetaModelica.nil,
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
  #= Extract component references from condition for MTK-aware runtime lookup.
     After MTK structural_simplify, hardcoded x[N] indices are invalid. =#
  local condCrefs = filter(c -> string(c) != "time", listArray(Util.getAllCrefs(cond)))
  local condCrefSymbols = map(c -> Symbol(string(c)), condCrefs)
  #= Build condition-variable assignment expressions: set each cref to a new boolean value =#
  local condSetExprs = map(condCrefs) do c
    local cStr = string(c)
    local entry = get(simCode.stringToSimVarHT, cStr, nothing)
    if entry !== nothing && !SimulationCode.isParameter(entry[2])
      local sym = Symbol(cStr)
      (falseExpr = :(x[lookuptableStates[$(QuoteNode(sym))]] = false),
       trueExpr  = :(x[lookuptableStates[$(QuoteNode(sym))]] = true))
    else
      (falseExpr = :(), trueExpr = :())
    end
  end
  affectCondPair = if ! active_DOCC_Equations[idx]
    quote
      function affect!(integrator)
        local t = integrator.t
        local x = integrator.u
        local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
        local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
        local lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
        local lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
        structuralChange.structureChanged = true
        $(map(e -> e.falseExpr, condSetExprs)...)
        auto_dt_reset!(integrator)
        add_tstop!(integrator, integrator.t)
      end
      function condition(x, t, integrator)
        local xs = $(condCrefSymbols)
        local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
        if !isempty(xs) && all(isnothing, indices)
          return false
        end
        local lookuptableStates = Dict((xs) .=> indices)
        local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
        local lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
        $(map(c -> Expr(Symbol("="),
                        Symbol(string(c)),
                        getIdxForLookupMTK(c, simCode)),
              Util.getAllCrefs(cond))...)
        return $(expToJuliaBoolMTK(cond, simCode))
      end
    end
  else #= The equation is active at the start =#
    quote
      function affect!(integrator)
        local t = integrator.t
        local x = integrator.u
        local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
        local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
        local lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
        local lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
        structuralChange.structureChanged = true
        $(map(e -> e.trueExpr, condSetExprs)...)
        #= Save state for the solve-loop recompilation path =#
        integrator.just_hit_tstop = true
        structuralChange.timeAtChange = integrator.t
        structuralChange.solutionAtChange = deepcopy(integrator.sol)
        #= Stop integration for reconfiguration =#
        terminate!(integrator)
      end
      function condition(x, t, integrator)
        local xs = $(condCrefSymbols)
        local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
        if !isempty(xs) && all(isnothing, indices)
          return false
        end
        local lookuptableStates = Dict((xs) .=> indices)
        local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
        local lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
        $(map(c -> Expr(Symbol("="),
                        Symbol(string(c)),
                        getIdxForLookupMTK(c, simCode)),
              Util.getAllCrefs(cond))...)
        return !($(expToJuliaBoolMTK(cond, simCode)))
      end
    end
  end
  return affectCondPair
end
