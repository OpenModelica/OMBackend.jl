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
Simulation runtime implemented based on the integrator interface from DifferentialEquations.jl.
This file contains the runtime for structural change handling.
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
import ..@VSS_DEBUG
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
  Full simulation context passed to the agent at a decision point.
  Gives the agent everything needed to reason about the model state.
"""
struct AgentContext
  "Continuous state variables: name → current value"
  stateVariables::Dict{String,Float64}
  "Current parameter values: name → value"
  parameters::Dict{String,Any}
  "Current values of the structural parameters up for decision"
  currentValues::Dict{String,Any}
  "Full simulation time span"
  tspan::Tuple{Float64,Float64}
  "Natural-language hint from the reconfigure block prompt clause (nothing if absent)"
  prompt::Union{String, Nothing}
  "Serialized constraint equations from the initial equation block (nothing if absent)"
  initialEquations::Union{String, Nothing}
end

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
  Wrapper callback for agentic recompilation.
  The new parameter values are determined at runtime by an external agent
  that receives the model metamodel and current simulation state.
"""
mutable struct StructuralChangeAgenticRecompilation <: AbstractStructuralChange
  "The name of the model being simulated"
  name::String
  "Indicates if the structure has changed"
  structureChanged::Bool
  "SCode representation of the model — sent to the agent as context"
  metaModel::SCode.CLASS
  "Names of the parameters the agent may modify"
  componentsToChange::Vector{String}
  "Symbol table used to map variable names to indices"
  stringToSimVarHT
  timeAtChange::Float64
  solutionAtChange
  "Natural-language hint from the reconfigure block prompt clause"
  prompt::Union{String, Nothing}
  "Serialized constraint equations from the initial equation block (nothing if absent)"
  initialEquations::Union{String, Nothing}
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
  "Time at which the structural change was triggered"
  timeAtChange::Float64
  "Saved solution at the time of structural change"
  solutionAtChange
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
        @VSS_DEBUG @info "Structure changed at $(i.t) transition to $(cb.name) => $(cb.structureChanged)"
        #= Find the correct variables and map them between the two models =#
        local newSystem = cb.system
        indicesOfCommonVariables = getIndicesOfCommonVariables(getSyms(newSystem)
                                                               ,getSyms(oldSystem)
                                                               ,omProblem.topVariables
                                                               ,commonVariableSet
                                                               ;destinationPrefix = cb.name
                                                               ,srcPrefix = activeModeName)

        local newSyms = getSyms(newSystem)
        local oldSyms = getSyms(oldSystem)
        newU0 = Float64[newSystem[sym] for sym in newSyms]
        @BACKEND_LOGGING @info "new initial values" newU0
        @BACKEND_LOGGING @info "Common vs" indicesOfCommonVariables
        #= Map old states to matching new states (by common variable name) =#
        for oldIdx in 1:min(length(oldSyms), length(newSyms))
          local idx = indicesOfCommonVariables[oldIdx]
          if idx != 0
            newU0[idx] = integrator.u[oldIdx]
          end
        end
        #=
          For any new state that was not filled from an old state, fall back
          to the old system's observed variables. In VSS mode eliminateNonDynamic
          is skipped so algebraic variables (e.g. pendulum_vx) remain accessible
          as observed in the old system and must be carried across the transition.
        =#
        local oldSysObj = oldSystem.f.sys
        for (newIdx, newSym) in enumerate(newSyms)
          local newSymStr = string(newSym)
          local oldSymName = if newSym in omProblem.topVariables
            newSym
          else
            Symbol(replace(newSymStr, cb.name => activeModeName))
          end
          if oldSymName in oldSyms
            continue
          end
          try
            local oldVar = getproperty(oldSysObj, oldSymName)
            local val = integrator[oldVar]
            if val isa Number && isfinite(val)
              newU0[newIdx] = val
              @BACKEND_LOGGING @info "Pulled observed value for $newSym from old $oldSymName" val
            end
          catch e
            @BACKEND_LOGGING @info "Failed to pull observed for $newSym" e
          end
        end
        @BACKEND_LOGGING @info "New u0:" newU0
        #= Save the old solution together with the name and the mode that was active =#
        push!(oldSols, integrator.sol)
        #= Now we have the start values for the next part of the system=#
        local newF = newSystem.f
        #=
          If the new mode is a DAE (singular mass matrix or registered guesses)
          rebuild the ODEProblem with `build_initializeprob = true` and the
          inherited values as a Dict. MTK then runs its initializer at t=i.t
          so the state entering the new mode satisfies that mode's algebraic
          constraints. On any failure we fall through to the legacy vector-u0
          path so currently passing VSS tests cannot regress.
        =#
        local dispatchedToDAEInit = false
        if _isDAETransferTarget(newSystem)
          try
            local u0Dict = _buildTransferU0Dict(newSystem, oldSystem, integrator,
                                                cb.name, activeModeName,
                                                omProblem.topVariables)
            local daeProb = ModelingToolkit.ODEProblem(
              newSystem.f.sys,
              u0Dict,
              (i.t, tspan[2]),
              newSystem.p;
              build_initializeprob = true,
              warn_initialize_determined = false,
              callback = CallbackSet(cb.systemSpecificCallbacks, omProblem.callbackConditions...)
            )
            integrator = init(daeProb, alg; force_dtmin = true, kwargs...)
            dispatchedToDAEInit = true
            @VSS_DEBUG @info "VSS transfer: re-ran MTK DAE init" n_inherited=length(u0Dict) u_after=integrator.u
          catch err
            @VSS_DEBUG @info "VSS transfer DAE-init path failed; falling back to vector u0" err
            dispatchedToDAEInit = false
          end
        end
        if !dispatchedToDAEInit
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
          reinit!(integrator, newU0; t0 = i.t, reset_dt = true)
        end
        #=
          Set the active mode to the mode we are currently using.
        =#
        activeModeName = cb.name
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
  Fill in initial conditions for new state variables that were observed (algebraic)
  in the old segment. Mutates newU0 in-place.

  When a structural change increases the number of unknowns (e.g. a clutch disengages
  and a previously algebraic shaft velocity becomes an independent ODE state), the
  standard createNewU0 only maps old-state -> new-state by name. Variables that were
  not states in the old segment default to their model start= values, ignoring the
  algebraic value they had at the transition instant.

  This function evaluates such variables as observed quantities from the old solution
  (using MTK symbolic indexing) and writes the correct value into newU0.
"""
function _fill_observed_u0!(newU0::Vector{Float64},
                             symsOfOldProblem::Vector{Symbol},
                             newProblem,
                             solutionAtChange)
  local oldSys = solutionAtChange.prob.f.sys
  for (i, sym) in enumerate(ModelingToolkit.get_unknowns(newProblem.f.sys))
    local varName = sym.f.name
    if !(varName in symsOfOldProblem)
      try
        local oldSym = getproperty(oldSys, varName)
        newU0[i] = last(solutionAtChange[oldSym])
      catch
        #= Variable not observable in old system; keep the start= value. =#
      end
    end
  end
end

"""
  Returns `true` when the new-mode ODEProblem needs MTK's DAE initializer
  at transition time (singular mass matrix or registered guesses).
  Defensive: any introspection failure returns `false` so we stay on the
  legacy vector-u0 path.
"""
function _isDAETransferTarget(newSystem)::Bool
  try
    local sys = newSystem.f.sys
    if !isempty(ModelingToolkit.guesses(sys))
      return true
    end
    local mm = newSystem.f.mass_matrix
    if mm === nothing
      return false
    end
    if mm isa UniformScaling
      return false
    end
    local d = [mm[k, k] for k in 1:size(mm, 1)]
    return any(iszero, d)
  catch
    return false
  end
end

"""
  Build a Dict mapping MTK symbolic variables of the new mode to inherited
  numeric values pulled from the old mode's integrator (state first, then
  observed). Unresolvable symbols are skipped silently; the new system's
  own defaults/guesses stay in effect for those unknowns.
"""
function _buildTransferU0Dict(newSystem, oldSystem, integrator,
                              newModeName::String, oldModeName::String,
                              topVariables::Vector{Symbol})
  local newSys = newSystem.f.sys
  local oldSys = oldSystem.f.sys
  local newSyms = getSyms(newSystem)
  local oldSyms = getSyms(oldSystem)
  local u0Dict = Dict{Any, Float64}()
  local oldIdxMap = Dict{Symbol, Int}()
  for (i, s) in enumerate(oldSyms)
    oldIdxMap[s] = i
  end
  for newSym in newSyms
    local oldSymName = newSym in topVariables ? newSym :
                       Symbol(replace(string(newSym), newModeName => oldModeName))
    local val = nothing
    if haskey(oldIdxMap, oldSymName)
      local v = integrator.u[oldIdxMap[oldSymName]]
      if v isa Number && isfinite(v)
        val = Float64(v)
      end
    else
      try
        local oldVar = getproperty(oldSys, oldSymName)
        local v = integrator[oldVar]
        if v isa Number && isfinite(v)
          val = Float64(v)
        end
      catch
        val = nothing
      end
    end
    if val !== nothing
      try
        local newVar = getproperty(newSys, newSym)
        u0Dict[newVar] = val
      catch
        #= Cannot resolve symbol on new sys — let the new system's
           own defaults handle it. =#
      end
    end
  end
  return u0Dict
end

"""
  Custom solver function for Modelica code with structuralCallbacks to monitor the solving process
  (Using the integrator interface) from DifferentialEquations.jl
"""
function solve(omProblem::OM_ProblemRecompilation, tspan::Tuple, alg; kwargs...)
  local problem = omProblem.problem
  local structuralCallbacks = omProblem.structuralCallbacks
  local callbackConditions = omProblem.callbackConditions
  local activeModeName = omProblem.activeModeName
  local integrator = init(problem, alg; kwargs...)
  local solutions = []
  local tmpSolAtChange
  #= Run the integrator=#
  @label START_OF_INTEGRATION
  while true
    local i = integrator
    local old_t = i.t
    #= Check structural callbacks in order =#
    retCode = check_error(integrator)
    for j in 1:length(structuralCallbacks)
      local cb = structuralCallbacks[j]
      @VSS_DEBUG @info "Structure Changed? $(cb.structureChanged)"
      if cb.structureChanged && i.t <= tspan[2]
        @VSS_DEBUG @info "Recompilation directive triggered at:" i.t "Δt is:" (i.t - i.dt)
        @VSS_DEBUG @info "[solve OM_ProblemRecompilation] calling recompilation" t=i.t cbName=cb.name
        local _t_recomp = time()
        local newU0
        @VSS_DEBUG @info "Syms before recompilation:" getSyms(problem) integrator.u
        (newProblem, newSymbolTable, finalInitialValues, initialValues, reducedSystem, specialCase) = recompilation(cb.name,
                                                                                                                    cb,
                                                                                                                    integrator,
                                                                                                                    tspan,
                                                                                                                    callbackConditions)
        @VSS_DEBUG @info "[solve OM_ProblemRecompilation] recompilation returned" elapsed_s=round(time()-_t_recomp, digits=2)
        local symsOfOldProblem = getSyms(problem)
        local symsOfNewProblem = getSyms(newProblem)
        @VSS_DEBUG begin
          @info "Old u" integrator.u
          @info "initialValues" initialValues
          @info "finalInitialValues" finalInitialValues
          @info "symsOfOldProblem" symsOfOldProblem
          @info "symsOfNewProblem" symsOfNewProblem
        end
        #= Use newProblem.u0 as the base vector: its ordering matches
           symsOfNewProblem (MTK unknowns order), whereas `initialValues`
           returned by the generated model can be ordered differently
           (e.g. fixed=true phi/phid values landing at vy/vx slots when old
           and new modes share no state name). createNewU0 then only
           overrides entries whose suffix matched an old state. =#
        local baseU0 = collect(Float64, newProblem.u0)
        newU0 = RuntimeUtil.createNewU0(symsOfOldProblem,
                                        symsOfNewProblem,
                                        baseU0,
                                        last(cb.solutionAtChange.u),
                                        specialCase)
        @VSS_DEBUG @info "[solve OM_ProblemRecompilation] newU0 from createNewU0" newU0 baseU0 symsOfNewProblem symsOfOldProblem oldFinal=last(cb.solutionAtChange.u)
        #= Fill in observed variables: new state vars absent from old state vector
           are recovered from the old solution's algebraic (observed) equations. =#
        _fill_observed_u0!(newU0, symsOfOldProblem, newProblem, cb.solutionAtChange)
        @VSS_DEBUG @info "The new u0" newU0
        @VSS_DEBUG @info "[solve OM_ProblemRecompilation] newU0 after _fill_observed_u0" newU0 anyNaN=any(isnan, newU0) elapsed_s=round(time()-_t_recomp, digits=2)
        #=
        TODO:
            Also add the continuous events here
          TODO: If there are discrete events these should be evaluated before proceeding
          local discrete_events = reducedSystem.discrete_events
          @info "discrete_events:" discrete_events
          =#
          # # TMP for System 10 With optimization
          #= Now we have the start values for the next part of the system=#
          local _t_init = time()
          integrator = init(newProblem,
                            alg,
                            force_dtmin = true;
                            kwargs...)
          @VSS_DEBUG @info "[solve OM_ProblemRecompilation] init(newProblem) done" elapsed_s=round(time()-_t_init, digits=2)
          local _t_reinit = time()
          reinit!(integrator,
                  newU0;
                  t0 = i.t,
                  tf = tspan[2],
                  reset_dt = true,
                  kwargs...)
          @VSS_DEBUG @info "[solve OM_ProblemRecompilation] reinit! done" elapsed_s=round(time()-_t_reinit, digits=2)
        #=
          Reset with the new values of u0
          and set the active mode to the mode we are currently using.
        =#
        #= Point the problem to the new problem =#
        #= ! This runs for both routines. That is initialization and recompilation !=#
        problem = newProblem
        #= Note that the structural event might be triggered again here (We kill it later) =#
        integrator.dt = min(integrator.dt, 1e-6)
        local _t_step = time()
        Base.invokelatest(step!, integrator, integrator.dt, true)
        @VSS_DEBUG @info "[solve OM_ProblemRecompilation] first step! after reinit done" t=integrator.t dt=integrator.dt dtpropose=integrator.dtpropose u=integrator.u anyNaN=any(isnan, integrator.u) elapsed_s=round(time()-_t_step, digits=2)
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
        @goto START_OF_INTEGRATION
      end
    end #= Structural callback handling =#
    #= invoke latest to avoid world age problems =#
    @VSS_DEBUG @info "integrator step" (integrator.t + integrator.dt) (integrator.t + integrator.dtpropose)
    if integrator.t + integrator.dtpropose >= tspan[2]
      @VSS_DEBUG @info "Timestep exceeds simulation, adjusting" integrator.t integrator.u (integrator.t + integrator.dtpropose)
      integrator.dt = 1e-6
      integrator.dtpropose = 1e-6
      @VSS_DEBUG @info "After adjustment" integrator.dt integrator.dtpropose integrator.u
      Base.invokelatest(step!, integrator, integrator.dt, true)
    else
      Base.invokelatest(step!, integrator, integrator.dt, false)
    end
    #=
    If a structural callback was triggered at the last integration step this boolean variable is true.
    =#
    local timeBeforeCallbackWasApplied::Float64 = 0
    local hitTstop = false
    if i.just_hit_tstop == true&& RuntimeUtil.isReturnCodeSuccess(i)
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
          @VSS_DEBUG @info "solutionAtChange.t" cb.solutionAtChange.t
          local stopIdx = findlast((x) -> x == timeBeforeCallbackWasApplied, cb.solutionAtChange.t)
          @assert stopIdx !== nothing "Invalid callback occured during simulation"
          @VSS_DEBUG @info "stopIdx" stopIdx
          solAtChange = cb.solutionAtChange #Used for error checking
          local modifiedSol = deepcopy(cb.solutionAtChange)
          resize!(modifiedSol.t, stopIdx)
          resize!(modifiedSol.u, stopIdx)
          @VSS_DEBUG @info "modifiedSol after resize" modifiedSol.u modifiedSol.t
          #= Assign the adjusted solution vector. =#
          uAtChange = modifiedSol.u
          timeAtChange = modifiedSol.t
          tmpSolAtChange = uAtChange
          #= The modified solution is the solution before we start the next part of the solution process. =#
          @VSS_DEBUG @info "Saving solution..."
          push!(solutions, modifiedSol)
          hitTstop = true
        end
      end
      if hitTstop
        reinit!(i,
                last(uAtChange);
                t0 = timeBeforeCallbackWasApplied,
                tf = tspan[2],
                reset_dt = true,
                erase_sol = true,
                reinit_callbacks = false)
        i.just_hit_tstop = false
        @VSS_DEBUG begin
          @info "uAtChange" uAtChange
          @info "activeModeName" activeModeName
          @info "timeBeforeCallbackWasApplied" timeBeforeCallbackWasApplied
          @info "integrator.u" integrator.u
          @info "integrator.t" integrator.t
          @info "i.u" i.u
          @info "i.t" i.t
        end
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
  Recompile the metamodel with components changed.
  Returns a tuple of a new problem together with a new symbol table.
inputs:
  Name of the active model
  The structural callback causing the recompilation
  The current u values of the integrator
  The provided timespan
  The callback conditions (This to make sure that the new model have the same callbacks)
"""
function recompilation(activeModeName,
                       structuralCallback::StructuralChangeRecompilation,
                       integrator,
                       tspan,
                       callbackConditions)::Tuple
  #=  Recompilation =#
  @VSS_DEBUG @info "[recompilation] ENTER" activeModeName t=integrator.t
  local _t_start = time()
  local integrator_u = integrator.u
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
  @VSS_DEBUG @info "[recompilation] step 1/7: parsed modification" elementToChange newValue elapsed_s=round(time()-_t_start, digits=2)
  #=  2) Change the parameters in the SCode via API (As specified by the modification)=#
  #=  2.1 Change the parameter so that it is the same as the modifcation. =#
  newProgram = MetaModelica.list(RuntimeUtil.setElementInSCodeProgram!(activeModeName,
                                                                       elementToChange,
                                                                       newValue, metaModel))
  @VSS_DEBUG @info "[recompilation] step 2/7: SCode mutated" elapsed_s=round(time()-_t_start, digits=2)
  local classToInstantiate = activeModeName
  #= 3) Call the frontend + the backend + JIT compile Julia code in memory =#
  local flatModelica = first(OMFrontend.instantiateSCodeToFM(classToInstantiate, newProgram))
  @VSS_DEBUG @info "[recompilation] step 3/7: frontend (instantiateSCodeToFM) done" elapsed_s=round(time()-_t_start, digits=2)
  #= 3.1 Run the backend =#
  local simulationCode = translateToSimCode(flatModelica, activeModeName)
  @VSS_DEBUG @info "[recompilation] step 4/7: backend (translateToSimCode) done" elapsed_s=round(time()-_t_start, digits=2)
  #= 3.2 Adjust variables and special parameters =#
  RuntimeUtil.updateInitialConditions!(simulationCode, integrator)
  local resultingModel = translateToMTK(simulationCode, activeModeName)
  @VSS_DEBUG @info "[recompilation] step 5/7: MTK codegen (translateToMTK) done" elapsed_s=round(time()-_t_start, digits=2)
  #= 4.0 Revaulate the model=#
  local modelName = string(replace(activeModeName, "." => "__"), "Model")
  @eval $(resultingModel)
  @VSS_DEBUG @info "[recompilation] step 6/7: module @eval done" modelName elapsed_s=round(time()-_t_start, digits=2)
  modelCall = quote
    $(Symbol(modelName))($(tspan))
  end
  (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars) = @eval $(modelCall)
  @VSS_DEBUG @info "[recompilation] step 7/7: modelCall @eval done (submodel ODEProblem built)" elapsed_s=round(time()-_t_start, digits=2)
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
  @VSS_DEBUG @info "[recompilation] composite ODEProblem rebuilt" elapsed_s=round(time()-_t_start, digits=2)
  #=4) Changed System=#
  #= 4.1 Update the structural callback with the new situation =#
  @match SOME(newMetaModel) = simulationCode.metaModel
  structuralCallback.metaModel = newMetaModel
  structuralCallback.stringToSimVarHT = simulationCode.stringToSimVarHT
  #= 4.2) Assign this system to newSystem. =#
  return (compositeProblem, simulationCode.stringToSimVarHT,finalInitialValues, initialValues, reducedSystem, false)
end

"""
  Agentic recompilation: queries an external agent server for new parameter
  values, then delegates to the standard recompilation path.

  The agent receives:
    - the model metamodel (SCode) serialised as a string
    - the names of the parameters that may be changed
    - the current simulation state (variable name → value)

  The agent returns a vector of new values, one per component in
  structuralCallback.componentsToChange.
"""
function recompilation(activeModeName,
                       structuralCallback::StructuralChangeAgenticRecompilation,
                       integrator,
                       tspan,
                       callbackConditions)::Tuple
  @info "[RECOMPILATION] Agentic recompilation triggered at t=$(integrator.t), components=$(structuralCallback.componentsToChange)"
  #= Build full agent context using MTK symbolic indexing for correctness =#
  local solAtChange = structuralCallback.solutionAtChange
  local sysAtChange = solAtChange.prob.f.sys
  #= State variable values at the change instant =#
  local stateVars = Dict{String,Float64}()
  for sym in ModelingToolkit.get_unknowns(sysAtChange)
    try
      stateVars[string(sym.f.name)] = last(solAtChange[sym])
    catch
    end
  end
  #= Current values of the specific parameters the agent may change =#
  local currentValues = Dict{String,Any}()
  for p in structuralCallback.componentsToChange
    try
      local sym = getproperty(sysAtChange, Symbol(p))
      if ModelingToolkit.isparameter(sym)
        currentValues[p] = ModelingToolkit.SymbolicIndexingInterface.getp(sysAtChange, sym)(solAtChange.prob)
      else
        currentValues[p] = last(solAtChange[sym])
      end
    catch
      currentValues[p] = nothing
    end
  end
  #= Full parameter snapshot of the system, distinct from the parameters
     exposed to the agent via componentsToChange. Gives the agent the
     full context it needs to reason about which values to return. =#
  local parameters = Dict{String,Any}()
  for p in ModelingToolkit.parameters(sysAtChange)
    try
      local pname = string(ModelingToolkit.getname(p))
      parameters[pname] = ModelingToolkit.SymbolicIndexingInterface.getp(sysAtChange, p)(solAtChange.prob)
    catch
    end
  end
  local agentCtx = AgentContext(stateVars, parameters, currentValues, tspan,
                                structuralCallback.prompt,
                                structuralCallback.initialEquations)
  #= Serialise metamodel and query agent =#
  local metaStr = string(structuralCallback.metaModel)
  local newValues = queryAgent(structuralCallback.componentsToChange,
                               agentCtx,
                               metaStr,
                               integrator.t)
  #= Validate agent response: length must match componentsToChange.
     A short/long return would otherwise silently drop or truncate
     modifications via zip() below. =#
  if length(newValues) != length(structuralCallback.componentsToChange)
    error("Agent returned $(length(newValues)) values for $(length(structuralCallback.componentsToChange)) components; expected equal length. components=$(structuralCallback.componentsToChange), returned=$(newValues)")
  end
  #= Apply each modification in sequence via the standard recompilation path =#
  local problem = nothing
  local ht_new = structuralCallback.stringToSimVarHT
  local finalInitialValues = integrator.u
  local initialValues = integrator.u
  local reducedSystem = nothing
  for (componentName, newValue) in zip(structuralCallback.componentsToChange, newValues)
    scr = StructuralChangeRecompilation(activeModeName,
                                        false,
                                        structuralCallback.metaModel,
                                        (componentName, string(newValue)),
                                        structuralCallback.stringToSimVarHT,
                                        integrator.t,
                                        Float64[])
    (problem, ht_new, finalInitialValues, initialValues, reducedSystem, _) =
      recompilation(activeModeName, scr, integrator, tspan, callbackConditions)
    structuralCallback.metaModel = scr.metaModel
    structuralCallback.stringToSimVarHT = scr.stringToSimVarHT
  end
  return (problem, ht_new, finalInitialValues, initialValues, reducedSystem, false)
end

"""
  The global agent callback used by agentic_recompilation.

  By default this is a no-op that returns `nothing` for every parameter,
  leaving the current value unchanged.  Replace it with a real agent
  (rule-based, LLM, HTTP, etc.) before simulating:

    OMBackend.Runtime.AGENT_CALLBACK[] = (params, state, metamodel, t) -> ...

  For HTTP-based agents set the environment variable AGENTIC_MODELICA_URL and
  use an external server compatible with MockAgentServer.jl.
"""
const AGENT_CALLBACK = Ref{Function}(
  (componentsToChange, context, metamodel, t) -> [nothing for _ in componentsToChange]
)

"""
  Query the agent for new parameter values.
  Dispatches to AGENT_CALLBACK[](componentsToChange, context::AgentContext, metamodel, t).
"""
function queryAgent(componentsToChange::Vector{String},
                    context::AgentContext,
                    metamodel::String,
                    t::Float64)::Vector{Any}
  try
    return AGENT_CALLBACK[](componentsToChange, context, metamodel, t)
  catch e
    @warn "agentic_recompilation: agent callback failed, keeping current values." exception=e
    return [nothing for _ in componentsToChange]
  end
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
  @VSS_DEBUG @info "[recompilation DOCC] ENTER" activeModeName
  local _t_start = time()
  #= Fetch the model that we were generating from memory. =#
  local flatModel = OMBackend.CodeGeneration.FLAT_MODEL
  local unresolvedConnectEquations = flatModel.unresolvedConnectEquations
  #= Get the relevant equation =#
  local indexOfEquation = structuralCallback.index
  local equationIf = MetaModelica.listGet(flatModel.DOCC_equations, indexOfEquation)
  @assert length(equationIf.branches) == 1
  if ! structuralCallback.activeEquations
    equationsToAdd = first(equationIf.branches).body
    newFlatModel = RuntimeUtil.createNewFlatModel(flatModel, unresolvedConnectEquations, equationsToAdd)
  else
    newFlatModel = RuntimeUtil.createNewFlatModel(flatModel, indexOfEquation, unresolvedConnectEquations)
  end
  @VSS_DEBUG @info "[recompilation DOCC] step 1/5: new flat model built" elapsed_s=round(time()-_t_start, digits=2)
  local simulationCode = translateToSimCode(newFlatModel, activeModeName)
  @VSS_DEBUG @info "[recompilation DOCC] step 2/5: backend (translateToSimCode) done" elapsed_s=round(time()-_t_start, digits=2)
  local resultingModel = translateToMTK(simulationCode, activeModeName)
  @VSS_DEBUG @info "[recompilation DOCC] step 3/5: MTK codegen (translateToMTK) done" elapsed_s=round(time()-_t_start, digits=2)
  #println("New model generated")
  local model = replace(activeModeName, "." => "__")
  local modelName = string(model, "Model")
  #local result = OMBackend.modelToString(model; MTK = true, keepComments = false, keepBeginBlocks = false)
  #println("We have a new model!\n");
  resultingModel = OMBackend.CodeGeneration.stripComments(resultingModel)
  resultingModel = OMBackend.CodeGeneration.stripBeginBlocks(resultingModel)
  #= Ensure Modelica function wrapper bindings are available in Runtime scope.
     The wrappers are created in CodeGeneration (mtkExternals.jl:317) but the
     recompilation eval runs here in Runtime. =#
  for (_fn, _w) in OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS
    @eval $_fn = OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS[$(QuoteNode(_fn))]
  end
  @eval $(resultingModel)
  @VSS_DEBUG @info "[recompilation DOCC] step 4/5: module @eval done" modelName elapsed_s=round(time()-_t_start, digits=2)
  @BACKEND_LOGGING OMBackend.writeStringToFile(string("modfied", modelName * ".jl"), "$resultingModel")
  local modelCall = quote
    $(Symbol(modelName))($(tspan))
  end
  (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars) = @eval $(modelCall)
  @VSS_DEBUG @info "[recompilation DOCC] step 5/5: modelCall @eval done (submodel ODEProblem built)" elapsed_s=round(time()-_t_start, digits=2)
  #= Reconstruct composite problem: use 3-arg ODEProblem with complete u0
     (including algebraic unknowns) and skip the initialization solver.
     buildDefaultGuesses fills algebraic unknowns not in finalInitialValues with 0.0. =#
  local _recompGuesses = Base.invokelatest(
    OMBackend.CodeGeneration.buildDefaultGuesses, reducedSystem, finalInitialValues, initialValues)
  compositeProblem = ModelingToolkit.ODEProblem(
    reducedSystem,
    merge(Dict(finalInitialValues), _recompGuesses, pars),
    tspan;
    callback = CallbackSet(callbackConditions, callbacks),
    warn_initialize_determined = false,
    build_initializeprob = false,
  )
  @VSS_DEBUG @info "[recompilation DOCC] composite ODEProblem rebuilt" elapsed_s=round(time()-_t_start, digits=2)
  #= Update the structural callback's symbol table for the new system =#
  structuralCallback.stringToSimVarHT = simulationCode.stringToSimVarHT
  return (compositeProblem,
          simulationCode.stringToSimVarHT,
          finalInitialValues,
          initialValues,
          reducedSystem,
          true) #= specialCase=true: variables can be assumed unchanged for DOCC =#
end

"""
  This function returns the root indices of the OCC graph.
"""
function returnRootIndices(activeModeName,
                structuralCallback::StructuralChangeDynamicConnection,
                integrator_u,
                tspan,
                callbackConditions)
  local flatModel = OMBackend.CodeGeneration.FLAT_MODEL
  (variablestoReset, rootSources) = RuntimeUtil.resolveDOOCConnections(flatModel, flatModel.name)
  local rootVariables = keys(variablestoReset)
  local ht = structuralCallback.stringToSimVarHT
  rootIndices = Int[]
  variablesToSet = []
  variablesToSetIdx = Vector{Int}[]
  for v in rootVariables
    indexOfRoot = first(ht[v])
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
  local newModel = OMBackend.CodeGeneration.ODE_MODE_MTK_MODEL_GENERATION(simulationCode, classToInstantiate, []; useDirectRHS = false)
  return (newModel, simulationCode)
end


function translateToSimCode(flatModelica, classToInstantiate)
  local bdae = OMBackend.lower(flatModelica)
  local simulationCode = OMBackend.generateSimulationCode(bdae; mode = OMBackend.MTK_MODE)
  return simulationCode
end

function translateToMTK(simulationCode, classToInstantiate)
  local newModel = OMBackend.CodeGeneration.ODE_MODE_MTK_MODEL_GENERATION(simulationCode, classToInstantiate, []; useDirectRHS = false)
  return newModel
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
  return Symbol[state.f.name for state in ModelingToolkit.get_unknowns(problem.f.sys)]
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
  (smallestKeyset, dict) = if length(keys(idxDict1)) < length(keys(idxDict2))
    keys(idxDict1), idxDict2
  else
    keys(idxDict2), idxDict1
  end
  for key in smallestKeyset
    local val = get(dict, key, 0)
    push!(indicesOfCommonVariables, val)
  end
  return indicesOfCommonVariables
end

end
