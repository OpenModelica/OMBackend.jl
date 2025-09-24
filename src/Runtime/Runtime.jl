#=
  Simulation runtime implemented based on the integrator interface
=#
module Runtime
include("RuntimeUtil.jl")
import .RuntimeUtil
import Absyn
import SCode
import OMBackend
import OMBackend.SimulationCode
import OMBackend.CodeGeneration
import ..@BACKEND_LOGGING
import OMFrontend
import DAE

import ModelingToolkit
import ModelingToolkit.IfElse

using DataStructures
using ModelingToolkit
using DifferentialEquations
using MetaModelica

abstract type AbstractOMSolution end

"""
    Wrapper structure with the intention to wrap solutions produced by the compiler suite.
    Not currently in use.
"""
struct OMSolution{T1, T2} <: AbstractOMSolution
  "Solution given by DifferentialEquations.jl"
  diffEqSol::T1
  "Various metadata for the specific model"
  idxToName::T2
end

"""
  Wrapper object for equation based models that contain several solutions
"""
struct OMSolutions{T1, T2} <: AbstractOMSolution
  "Set of solutions given by DifferentialEquations.jl"
  diffEqSol::Vector{T1}
  "Various metadata for the specific model"
  idxToName::T2
end


abstract type AbstractStructuralChange end

"""
  Wrapper callback for a static structural change.
  That is the model
  we are simulating changes during the simulation but the future model can be predicted statically
"""
mutable struct StructuralChange{SYS} <: AbstractStructuralChange
  "The name of the next mode"
  name::String
  "Indicates if the structure has changed"
  structureChanged::Bool
  "The system we are switching to."
  system::SYS
  systemSpecificCallbacks
end

"""
  Wrapper callback for structural change that triggers a recompilation.
"""
mutable struct StructuralChangeRecompilation{MOD <: Tuple} <: AbstractStructuralChange
  "The name of the next mode"
  name::String
  "Indicates if the structure has changed"
  structureChanged::Bool
  "The meta model. That is a SCode representation of the model itself"
  metaModel::SCode.CLASS
  "The modification to be applied during recompilation"
  modification::MOD
  """
    The symbol table for the old model.
    This is used to map indices of variables when the structure of the model changes
  """
  stringToSimVarHT
  timeAtChange::Float64
  solutionAtChange
end

"""
 Wrapper callback for a dynamic connection reconfiguration
"""
mutable struct StructuralChangeDynamicConnection <: AbstractStructuralChange
  "The name of the next mode"
  name::String
  "Indicates if the structure has changed"
  structureChanged::Bool
  "The meta model. A flat representation of the model itself."
  #= Would it be better to modify the SCode instead? Less code to change?=#
  flatModel::OMFrontend.Frontend.FLAT_MODEL
  "The index of the specific dynamic connection equation."
  index::Int
  """
    The symbol table for the old model.
    This is used to map indices of variables when the structure of the model changes
  """
  stringToSimVarHT
  """
    If equations are to be added or removed.
  """
  activeEquations::Bool
end

mutable struct OM_ProblemStructural{T0 <: String, T1, T2, T3}
  "The name of the active mode"
  activeModeName::T0
  "The problem we are currently solving"
  problem::T1
  "The set of structural callbacks"
  structuralCallbacks::T2
  """ The parameter of the model """
  pars
  """ Variables that all modes have in common """
  commonVariables::T3
  "Topmost variables of the model "
  topVariables::Vector{Symbol}
  "The callback conditions "
  callbackConditions
end

mutable struct OM_ProblemRecompilation{T0 <: String, T1, T2, T3}
  "The name of the active mode"
  activeModeName::T0
  "The problem we are currently solving"
  problem::T1
  "The set of structural callbacks"
  structuralCallbacks::T2
  "The set of callback conditons"
  callbackConditions::T3
end

mutable struct OM_Problem{T0, T1}
  problem::T0
end

#=
The current scheme for structural change.
Callbacks are created in the model that encompasses the two submodels.

These callbacks contains a boolean field that indicate if the structure has changed.
It also contains a field indicating what system we should switch to.

During solving, this field is set by the callback.
In the solver loop we iterate through these callbacks.

If a structural change was detected we act on it and change to the system pointed to by the callback.
Depending on the encompassing system we either just in time recompile the new system or we switch to the new.
Saving our current time step and reinitialize our new changed system
.
We should also statically detect if VSS simulation is needed since it is more resource heavy than regular simulation
=# ≈

"""
  Custom solver function for Modelica code with structuralCallbacks to monitor the solving process
  (Using the integrator interface) from DifferentialEquations.jl
"""
function solve(omProblem::OM_ProblemStructural, tspan, alg; kwargs...)
  local problem = omProblem.problem
  local oldSystem = problem
  local structuralCallbacks = omProblem.structuralCallbacks
  local commonVariableSet = omProblem.commonVariables
  local symsOfInitialMode = getSyms(problem)
  local activeModeName = omProblem.activeModeName
  #= Create integrator =#
  integrator = init(problem, alg; kwargs...)
  add_tstop!(integrator, tspan[2])
  local oldSols = []
  #= Run the integrator=#
  @label START_OF_INTEGRATION
  for i in integrator
    @BACKEND_LOGGING @info "u values at Δt $(integrator.dt) & t = $(integrator.t)" integrator.u
    #= Check structural callbacks in order =#
    @BACKEND_LOGGING @info "Stepping at:" i.t
    retCode = check_error(integrator)
    for cb in structuralCallbacks
      if cb.structureChanged && cb.name != activeModeName
        #@info "Structure changed at $(i.t) transition to $(cb.name) => $(cb.structureChanged)"
        #= Find the correct variables and map them between the two models =#
        local newSystem = cb.system
        indicesOfCommonVariables = getIndicesOfCommonVariables(getSyms(newSystem)
                                                               ,getSyms(oldSystem)
                                                               ,omProblem.topVariables
                                                               ,commonVariableSet
                                                               ;destinationPrefix = cb.name
                                                               ,srcPrefix = activeModeName)

        newU0 = Float64[newSystem[sym] for sym in getSyms(newSystem)]
        @BACKEND_LOGGING @info "new initial values" newU0
        @BACKEND_LOGGING @info "Common vs" indicesOfCommonVariables
        for oldIdx in 1:min(length(getSyms(oldSystem)), length(getSyms(newSystem)))
          local idx = indicesOfCommonVariables[oldIdx]
          if idx != 0
            newU0[idx] = integrator.u[oldIdx]
          end
        end
        @BACKEND_LOGGING @info "New u0:" newU0
        #= Save the old solution together with the name and the mode that was active =#
        push!(oldSols, integrator.sol)
        #= Now we have the start values for the next part of the system=#
        local newF = newSystem.f
        newProbTest = ModelingToolkit.ODEProblem(
          newF,
          newU0,
          tspan,
          newSystem.p,
          callback = CallbackSet(cb.systemSpecificCallbacks, omProblem.callbackConditions...)
        )
        integrator = init(newProbTest,
                          alg;
                          force_dtmin = true,
                          u0 = newU0,
                          kwargs...)
        #=
          Reset with the new values of u0
          and set the active mode to the mode we are currently using.
        =#
        activeModeName = cb.name
        reinit!(integrator, newU0; t0 = i.t, reset_dt = true)
        for cb in structuralCallbacks
          cb.structureChanged = false
        end
        oldSystem = newSystem
        @goto START_OF_INTEGRATION
      end
    end
  end
  #= The solution of the integration procedure =#
  local solution = integrator.sol
  push!(oldSols, solution)
  return oldSols
end

#= Enable this switch to allow DOCC without uncessary recompilation. =#
global SHOULD_DO_REINITIALIZATION = false

"""
  Custom solver function for Modelica code with structuralCallbacks to monitor the solving process
  (Using the integrator interface) from DifferentialEquations.jl
"""
function solve(omProblem::OM_ProblemRecompilation, tspan::Tuple, alg; kwargs...)
  local problem = omProblem.problem
  local structuralCallbacks = omProblem.structuralCallbacks
  local callbackConditions = omProblem.callbackConditions
  local activeModeName = omProblem.activeModeName
  local integrator = init(problem, alg)
  local solutions = []
  #println("Start integration")
  #= Run the integrator=#
  @label START_OF_INTEGRATION
  while true
    local i = integrator
    local old_t = i.t
    #= Check structural callbacks in order =#
    retCode = check_error(integrator)
    for j in 1:length(structuralCallbacks)
      local cb = structuralCallbacks[j]
      #@info("Structure Changed? $(cb.structureChanged)")
      if cb.structureChanged && i.t <= tspan[2]
        #@info "Recompilation directive triggered at:" i.t
        #@info "Δt is:" i.t - i.dt
        local newU0
        #@info "Test syms before recompilation call..." getSyms(problem)
        (newProblem, newSymbolTable, finalInitialValues, initialValues, reducedSystem, specialCase) = recompilation(cb.name,
                                                                                                                    cb,
                                                                                                                    integrator.u,
                                                                                                                    tspan,
                                                                                                                    callbackConditions)
        #= Assuming the indices are the same (Which is not always true) =#
          local symsOfOldProblem = getSyms(problem)
          local symsOfNewProlem = getSyms(newProblem)
          #@info "U0 before variables are set:"
          #@info integrator.u
          #@info i.u
          newU0 = RuntimeUtil.createNewU0(symsOfOldProblem,
                                          symsOfNewProlem,
                                          initialValues,
                                          integrator,
                                          specialCase)
        #=
        TODO:
            Also add the continuous events here
          TODO: If there are discrete events these should be evaluated before proceeding
          local discrete_events = reducedSystem.discrete_events
          @info "discrete_events:" discrete_events
          =#
          # # TMP for System 10 With optimization
          #= Now we have the start values for the next part of the system=#
          integrator = init(newProblem,
                            alg,
                            force_dtmin = true,
                            kwargs...)
          reinit!(integrator,
                  newU0;
                  t0 = i.t,
                  tf = tspan[2],
                  reset_dt = true)
        #=
          Reset with the new values of u0
          and set the active mode to the mode we are currently using.
        =#
        #= Point the problem to the new problem =#
        #= ! This runs for both routines. That is initialization and recompilation !=#
        problem = newProblem
        #= Note that the structural event might be triggered again here (We kill it later) =#
        #@info "Stepping... at $(integrator.t)"
        Base.invokelatest(step!, integrator, integrator.dt, true)
        #@info "After step"
        #@info cb.structureChanged
        #@info "Time after step" integrator.t
        #@info "Time after step" i.t
        #=
        We reset the structural change pointer again here just to make sure
        that we do not trigger the structural callback again.
        Note,currently this will trigger the callback setting the boolean cb variable to true again.
        However, we reset twice s.t it will not go into this conditional again for the same callback.
        =#
        integrator.just_hit_tstop = false
        cb.structureChanged = false
        #=
        TODO:
        Make a PR for the Julia guys to provide an option to step without triggering callbacks.
        This so that we can avoid the hack above.
        =#
        #@info "Reset!"
        @goto START_OF_INTEGRATION
      end
    end #= Structural callback handling =#
    #= invoke latest to avoid world age problems =#
    # @info "i.t + i.dt" i.t + i.dt
    # @info "integrator.t + integrator.dt" integrator.t + integrator.dt
    # @info "integrator.t + integrator.dt" integrator.t + integrator.dtpropose
    if integrator.t + integrator.dtpropose >= tspan[2]
      #@info integrator.t
      #= Calculate the length of the final step=#
      #@info "integrator.t + integrator.dtpropose >= tspan[2]!\n tspan[2] - i.t =  " tspan[2] - i.t
      #@info "integrator.dtpropose" integrator.dtpropose
      # @info "tspan[2]" tspan[2]
      # @info "i.t" i.t
      #local finalStep = tspan[2] - i.t
      #= Use a very small timestep =#
      integrator.dt = 1e-6
      integrator.dtpropose = 1e-6
      #@info "integrator.dt" integrator.dt
      #@info "integrator.dtpropose" integrator.dtpropose
      Base.invokelatest(step!, integrator, integrator.dt, true)#Base.invokelatest(step!, integrator) #step!(integrator, tspan[2])
      # @info "i.t" integrator.t
      #break #= Restarts  integration =#
    else
      Base.invokelatest(step!, integrator, integrator.dt, false)
    end
    #=
    If a structural callback was triggered at the last integration step this boolean variable is true.
    =#
    local timeBeforeCallbackWasApplied::Float64 = 0
    local hitTstop = false
    if i.just_hit_tstop == true #&& RuntimeUtil.isReturnCodeSuccess(i)
      local solAtChange
      #= Find the callback that was triggered =#
      local timeAtChange::Vector{Float64} = Float64[]
      local uAtChange#::Vector{Vector{Float64}}
      for cb in structuralCallbacks
        if cb.structureChanged
          timeBeforeCallbackWasApplied = cb.timeAtChange
          #=
          Save the old solution
          Resize the solution to the time t before the change.
          =#
          #@info cb.solutionAtChange.t
          local stopIdx = findlast((x) -> x == timeBeforeCallbackWasApplied, cb.solutionAtChange.t)
          @assert stopIdx !== nothing "Invalid callback occured during simulation"
          #@info "stopIdx" stopIdx
          solAtChange = cb.solutionAtChange #Used for error checking
          local modifiedSol = deepcopy(cb.solutionAtChange)
          #@info "modifiedSol" modifiedSol
          #@info "modifiedSol.t"  modifiedSol.t
          resize!(modifiedSol.t, stopIdx)
          resize!(modifiedSol.u, stopIdx)
          ##@info "After resize..."
          #@info "modifiedSol" modifiedSol
          #@info "modifiedSol.t"  modifiedSol.t
          #= Assign the adjusted solution vector. =#
          uAtChange = modifiedSol.u
          timeAtChange = modifiedSol.t
          #= The modified solution is the solution before we start the next part of the solution process. =#
          #@info "Saving solution..."
          push!(solutions, modifiedSol)
          hitTstop = true
        end
      end
      #@assert !(isempty(solAtChange)) "Invalid structural change occured"
      #@info "Reinit"
      #@info "tprev" timeAtChange
      #@info "uAtChange" uAtChange
      #@info typeof(uAtChange)
      if hitTstop
        reinit!(i,
                last(uAtChange);
                t0 = timeBeforeCallbackWasApplied,
                tf = tspan[2],
                reset_dt = true,
                erase_sol = true,
                reinit_callbacks = false)
        i.just_hit_tstop = false
      elseif integrator.t >= last(tspan) #= Hack to handle the special case where we are slightly above tstop=#
        sol = integrator.sol
        idx = findall(t -> t <= tspan[2], sol.t)
        @assign sol.t = sol.t[idx]
        @assign sol.u = sol.u[idx]
        push!(solutions, sol)
        return solutions
      end
    end
  end
  push!(solutions, integrator.sol)
  return solutions
end

"""
  Recompile the meta model with some component changed.
  Returns a tuple of a new problem together with a new symbol table.
inputs:
  Name of the active model
  The structural callback causing the recompilation
  The current u values of the integrator
  The provided timespan
  The callback conditions (This to make sure that the new model have the same callbacks)

  The boolean sc (Special case indicates if the variables can be assumed to be unchanged or not)
"""
function recompilation(activeModeName,
                       structuralCallback::StructuralChangeRecompilation,
                       integrator_u,
                       tspan,
                       callbackConditions)::Tuple
  #=  Recompilation =#
  #= Have the SCode =#
  #= 1) Fetch the parameter from the structural callback =#
  local metaModel = structuralCallback.metaModel
  local modification = structuralCallback.modification
  local inProgram = MetaModelica.list(metaModel)
  local elementToChange = first(modification)
  #= Get the symbol table, using this we evaluate the new value based on the value of some parameter. =#
  local newValueExpr = Meta.parse(last(modification))
  newValueExpr = quote
    stringToSimVarHT = $(structuralCallback.stringToSimVarHT)
    $(newValueExpr)
  end
  local newValue = eval(newValueExpr)
  #=  2) Change the parameters in the SCode via API (As specified by the modification)=#
  #=  2.1 Change the parameter so that it is the same as the modifcation. =#
  newProgram = MetaModelica.list(RuntimeUtil.setElementInSCodeProgram!(activeModeName, elementToChange, newValue, metaModel))
  local classToInstantiate = activeModeName
  #= 3) Call the frontend + the backend + JIT compile Julia code in memory =#
  local flatModelica = first(OMFrontend.instantiateSCodeToFM(classToInstantiate, newProgram))
  #= 3.1 Run the backend=#
  (resultingModel, simulationCode) = runBackend(flatModelica, classToInstantiate)
  #= 4.0 Revaulate the model=#
  local modelName = string(replace(activeModeName, "." => "__"), "Model")
  @eval $(resultingModel)
  modelCall = quote
    $(Symbol(modelName))($(tspan))
  end
  (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars) = @eval $(modelCall)
  #= Reconstruct the composite problem to keep the callbacks =#
  compositeProblem = ModelingToolkit.ODEProblem(
    reducedSystem,
    finalInitialValues,
    tspan,
    pars,
    #=
      TODO currently only handles a single structural callback.
    =#
    callback = CallbackSet(callbackConditions, callbacks)
  )
  #=4) Changed System=#
  #= 4.1 Update the structural callback with the new situation =#
  @match SOME(newMetaModel) = simulationCode.metaModel
  structuralCallback.metaModel = newMetaModel
  structuralCallback.stringToSimVarHT = simulationCode.stringToSimVarHT
  #= 4.2) Assign this system to newSystem. =#
  return (compositeProblem, simulationCode.stringToSimVarHT,finalInitialValues,  initialValues, reducedSystem, false)
end

"""
  Structural callback for dynamic connection handling.
  Returns (problem, symbol table, initial values, sc).
The boolean sc (Special case indicates if the variables can be assumed to be unchanged or not).
For the DOCC systems this can be assumed to be true.
"""
function recompilation(activeModeName,
                       structuralCallback::StructuralChangeDynamicConnection,
                       integrator_u,
                       tspan,
                       callbackConditions)
  #@info "recompilation"
  #= Fetch the model that we were generating from memory. =#
  local flatModel = OMBackend.CodeGeneration.FLAT_MODEL
  local unresolvedConnectEquations = flatModel.unresolvedConnectEquations
  #= Get the relevant equation =#
  local indexOfEquation = structuralCallback.index
  local equationIf = MetaModelica.listGet(flatModel.DOCC_equations, indexOfEquation)
  @assert length(equationIf.branches) == 1
  #println("Assertions fullfilled")
  if ! structuralCallback.activeEquations
    #println("First branch")
    equationsToAdd = first(equationIf.branches).body
    newFlatModel = RuntimeUtil.createNewFlatModel(flatModel, unresolvedConnectEquations, equationsToAdd)
  else
    newFlatModel = RuntimeUtil.createNewFlatModel(flatModel, indexOfEquation, unresolvedConnectEquations)
  end
  (resultingModel, simulationCode) = runBackend(newFlatModel, activeModeName)
  #println("New model generated")
  local model = replace(activeModeName, "." => "__")
  local modelName = string(model, "Model")
  #local result = OMBackend.modelToString(model; MTK = true, keepComments = false, keepBeginBlocks = false)
  #println("We have a new model!\n");
  resultingModel = OMBackend.CodeGeneration.stripComments(resultingModel)
  resultingModel = OMBackend.CodeGeneration.stripBeginBlocks(resultingModel)
  result = "$resultingModel"
  @eval $(resultingModel)
  @BACKEND_LOGGING OMBackend.writeStringToFile(string("modfied", modelName * ".jl"), result)
  local newParsedModel = Meta.parse(result)
  @eval $(newParsedModel)
  local modelCall = quote
    $(Symbol(modelName))($(tspan))
  end
  (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars) = @eval $(modelCall)
  #= Update meta model somehow=#
  return (problem,
          simulationCode.stringToSimVarHT,
          problem.u0,
          true, #= Returns true to indicate special case... =#
          reducedSystem)
end

"""
  This function returns the root indices of the OCC graph.
"""
function returnRootIndices(activeModeName,
                structuralCallback::StructuralChangeDynamicConnection,
                integrator_u,
                tspan,
                callbackConditions)
  #@info "Calling returnRootIndices"
  local flatModel = OMBackend.CodeGeneration.FLAT_MODEL
  (variablestoReset, rootSources) = RuntimeUtil.resolveDOOCConnections(flatModel, flatModel.name)
  #@info "variablestoReset" variablestoReset
  #@info rootSources
  local rootVariables = keys(variablestoReset)
  #@info "Root variables" rootVariables
  local ht = structuralCallback.stringToSimVarHT
  rootIndices = Int[]
  variablesToSet = []
  variablesToSetIdx = Vector{Int}[]
  for v in rootVariables
    indexOfRoot = first(ht[v])
    #@info "Index of $(v) was:" indexOfRoot
    push!(rootIndices, indexOfRoot)
    push!(variablesToSet, values(variablestoReset[v]))
  end
  for variables in variablesToSet
    tmp = Int[]
    for v in variables
      idx = first(ht[v])
      push!(tmp, idx)
    end
    push!(variablesToSetIdx, tmp)
  end
  return (rootIndices, variablesToSetIdx, rootSources, variablestoReset)
end

"""
 Runs the backend.
  Translates the flat model to Flat Modelica.
  Generates the simulation code.
  Creates the new model.
Returns, a tuple of the new model and the simulation code of this model.
"""
function runBackend(flatModelica, classToInstantiate)
  local bdae = OMBackend.lower(flatModelica)
  local simulationCode = OMBackend.generateSimulationCode(bdae; mode = OMBackend.MTK_MODE)
  local newModel = OMBackend.CodeGeneration.ODE_MODE_MTK_MODEL_GENERATION(simulationCode, classToInstantiate, [])
  return (newModel, simulationCode)
end

"""
  Solving procedure without structural callbacks.
"""
function solve(omProblem::OM_Problem, tspan, alg; kwargs...)
  local problem = omProblem.problem
  #= Create integrator =#
  integrator = init(problem, alg, stop_at_next_tstop = true, kwargs...)
  add_tstop!(integrator, tspan[2])
  for i in integrator
  end
  #= Return the final solution =#
  return integrator.sol
end

"""
    Fetches the symbolic variables from a problem.
"""
function getSyms(problem::ODEProblem)::Vector{Symbol}
  return [state.f.name for state in ModelingToolkit.get_unknowns(problem.f.sys)]
end

"""
  Fetches  the symbolic variables from a solution
"""
function getSymsFromSolution(sol)::Vector{Symbol}
  getSyms(sol.f.prob)
end


"""
  Get a vector of indices of the variables between syms1 and syms2.
  The destination mode might have a prefix for it's symbols.
  The prefix string is used to give the common variables the correct prefix
  between transistions.
"""
function getIndicesOfCommonVariables(syms1::Vector{Symbol} # New system
                                     ,syms2::Vector{Symbol} # old system
                                     ,topVariables::Vector{Symbol}
                                     ,inCommonVariables::Vector{String}
                                     ;destinationPrefix::String = ""
                                     ,srcPrefix::String = "")
  #= The common variables have the name without the prefix of the destination system =#
  #@info "syms2 (Old System)" syms2
  #@info "syms1 (New System)" syms1
  local newSyms = Symbol[]
  for name in syms2
    if name in topVariables
      push!(newSyms, name)
    else
      push!(newSyms, Symbol(replace(string(name), srcPrefix => destinationPrefix)))
    end
  end
  local indicesOfCommonVariables = Int[]
  local idxDict1 = DataStructures.OrderedDict()
  local idxDict2 = DataStructures.OrderedDict()
  for (i, sym) in enumerate(syms1)
    idxDict1[sym] = i
  end
  for (i, sym) in enumerate(newSyms)
    idxDict2[sym] = i
  end
  local commonVariables = ∩(keys(idxDict1), keys(idxDict1))
  #@info "Common variables", commonVariables
  (smallestKeyset, dict) = if length(keys(idxDict1)) < length(keys(idxDict2))
    keys(idxDict1), idxDict2
  else
    keys(idxDict2), idxDict1
  end
  for key in smallestKeyset
    if haskey(dict, key)
      push!(indicesOfCommonVariables, dict[key])
    else
      push!(indicesOfCommonVariables, 0)
    end
  end
  return indicesOfCommonVariables
end

end
