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

#=
Simulation runtime implemented based on the integrator interface from DifferentialEquations.jl.
This file contains the runtime for structural change handling.
=#
module Runtime
include("RuntimeUtil.jl")
import .RuntimeUtil
import OMBackend
import OMBackend.SimCode
import ..@BACKEND_LOGGING
import ..@VSS_DEBUG

import ModelingToolkit
import Setfield

using DataStructures
using ModelingToolkit
using DifferentialEquations
#= SciMLBase's integrators, return codes, NoInit and remake (as ModelingToolkit has them) =#
import ModelingToolkit.SciMLBase: DEIntegrator, ReturnCode, NoInit, remake

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
  Wrapper callback for structural change that triggers a recompilation. `M`: the type of the frontend's model of the
  model (OMFrontend's SCode class), which the recompilation changes and instantiates again (`recompiledSimCode`).
"""
mutable struct StructuralChangeRecompilation{MOD <: Tuple, M} <: AbstractStructuralChange
  "The name of the next mode"
  name::String
  "Indicates if the structure has changed"
  structureChanged::Bool
  "The meta model: the frontend's representation of the model itself (OMFrontend: its SCode class)"
  metaModel::M
  """
    The modification to be applied during recompilation: the element changed and the function of the symbol table
    (`stringToSimVarHT`, the current mode's) that gives its new value (generated with the model, structuralCallbacks.jl)
  """
  modification::MOD
  """
    The symbol table for the old model.
    This is used to map indices of variables when the structure of the model changes
  """
  stringToSimVarHT
  timeAtChange::Float64
  solutionAtChange
  "The frontend's model the simulation started with, before any change"
  origin::M
  """
    The changes made to `origin` so far, each element's last value (`string`), by element: with `origin` the key of
    the mode they give (RECOMPILED_MODES). The frontend sets an element's binding to the value, so the model of a
    change depends only on each element's last value.
  """
  applied::Vector{Pair{String, String}}
end

"""
A structural change by recompilation of the frontend's model `metaModel`, before any change (as the generated callbacks
create it): its `origin` the model itself, nothing `applied`.
"""
StructuralChangeRecompilation(name::String, structureChanged::Bool, metaModel::M, modification::MOD, stringToSimVarHT,
                              timeAtChange::Float64, solutionAtChange) where {MOD <: Tuple, M} =
  StructuralChangeRecompilation{MOD, M}(name, structureChanged, metaModel, modification, stringToSimVarHT, timeAtChange,
                                        solutionAtChange, metaModel, Pair{String, String}[])

"""
  Wrapper callback for agentic recompilation.
  The new parameter values are determined at runtime by an external agent
  that receives the model metamodel and current simulation state. `M` as `StructuralChangeRecompilation`'s.
"""
mutable struct StructuralChangeAgenticRecompilation{M} <: AbstractStructuralChange
  "The name of the model being simulated"
  name::String
  "Indicates if the structure has changed"
  structureChanged::Bool
  "The frontend's representation of the model (OMFrontend: its SCode class), sent to the agent as context"
  metaModel::M
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
A mode compiled at a recompilation (`recompilation`), kept for the same changes of the same frontend model in the same
build (RECOMPILED_MODES): a model that makes a change again (a sampled change, a second simulate) switches without
compiling the mode again. Its problem before any solve, with its parameters as built (a solve's events change the
problem's); the frontend's model after the change and the mode's symbol table; the symbol table before the change's
start values and the start values (`RuntimeUtil.switchStartValues`), which a later change must give the same
(`_sameStarts`) except those of the problem's unknowns the build took as their u0 (set in u0) and those of its observed
variables (computed from the unknowns; the solve does not initialize: NoInit).
"""
struct RecompiledMode
  problem::ODEProblem
  "`problem.p` as built: a solve changes the problem's (discrete variables)"
  parameters::Any
  "The frontend's model after the change (OMFrontend: an SCode class)"
  frontendModel::Any
  stringToSimVarHT::RuntimeUtil.SymbolTable
  "The symbol table before the change's start values (`RuntimeUtil.updateInitialConditions!`)"
  originalHT::RuntimeUtil.SymbolTable
  starts::Dict{String, Tuple{Float64, Bool}}
  "The index in u0 of each unknown whose u0 the build took from its start value"
  startsInU0::Dict{String, Int}
  "The names of the problem's observed variables"
  observed::Set{String}
end

"""
The key of a recompiled mode: the owner model's name and module (its build), the objectid of the frontend's model the
simulation started with and the changes made to it (`StructuralChangeRecompilation.applied`).
"""
const RecompiledModeKey = Tuple{String, Module, UInt, Vector{Pair{String, String}}}

"""
The recompiled modes (`RecompiledMode`) by their key (`RecompiledModeKey`). A model's are dropped with its build
(`OMBackend.forgetModel`; a new build's first mode: `_rememberMode!`) and by `OMBackend.clearCaches!`.
"""
const RECOMPILED_MODES = Dict{RecompiledModeKey, RecompiledMode}()

"""
    forgetRecompiledModes!(owner::String) -> Nothing

Drops the recompiled modes of the model `owner` (canonical name).
"""
function forgetRecompiledModes!(owner::String)::Nothing
  filter!(entry -> first(first(entry)) != owner, RECOMPILED_MODES)
  return nothing
end

"""
    _rememberMode!(key::RecompiledModeKey, mode::RecompiledMode) -> Nothing

Keeps `mode` under `key`, after dropping the modes of the same model in another build (an earlier one's module).
"""
function _rememberMode!(key::RecompiledModeKey, mode::RecompiledMode)::Nothing
  local (owner::String, ownerModule::Module) = (key[1], key[2])
  filter!(entry -> !(first(entry)[1] == owner && first(entry)[2] !== ownerModule), RECOMPILED_MODES)
  RECOMPILED_MODES[key] = mode
  return nothing
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
=#

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
  local oldSols = Any[]
  #= Run the integrator=#
  @label START_OF_INTEGRATION
  for i in integrator
    @BACKEND_LOGGING @info "u values at Δt $(integrator.dt) & t = $(integrator.t)" integrator.u
    #= Check structural callbacks in order =#
    @BACKEND_LOGGING @info "Stepping at:" i.t
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
        #= Map old states to matching new states (by common variable name).
           AUDIT (ombackend-bug-audit-2026-06-05 #8): getIndicesOfCommonVariables
           returns values ordered by whichever keyset is SMALLER. When the NEW
           system has FEWER states than OLD, indicesOfCommonVariables[k] is the
           OLD index of the k-th NEW sym, but this loop treats the position as an
           OLD index and the value as a NEW index -> the two index spaces are
           swapped, scattering old state values into the wrong new slots (or a
           BoundsError). Correct only on the OLD<=NEW (state-adding) branch.
           Narrow VSS (state-reducing structural change) path. Fix: make
           getIndicesOfCommonVariables return a canonical old->new mapping
           independent of which keyset is smaller. =#
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
            OMBackend._fallback(e, :vssPullObserved; impact = :result)
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
            OMBackend._fallback(err, :vssTransferDAEInit; impact = :result)
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
      catch err
        #= Variable not observable in old system; keep the start= value. =#
        OMBackend._fallback(err, :vssOldValue; impact = :result)
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
  catch err
    OMBackend._fallback(err, :vssMassMatrix)
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
      catch err
        OMBackend._fallback(err, :vssOldObserved; impact = :result)
        val = nothing
      end
    end
    if val !== nothing
      try
        local newVar = getproperty(newSys, newSym)
        u0Dict[newVar] = val
      catch err
        #= Cannot resolve symbol on new sys — let the new system's
           own defaults handle it. =#
        OMBackend._fallback(err, :vssNewSymbol; impact = :result)
      end
    end
  end
  return u0Dict
end

"""
    solve(omProblem::OM_ProblemRecompilation, tspan::Tuple, alg; kwargs...) -> Vector{Any}

Solves a model whose structural changes recompile it: integrates the active mode until a structural change
(`_integrateSegment!`), switches to the mode of the change (`_switchMode`) and continues, to the end of `tspan`. The
solutions of the modes, in order.
"""
function solve(omProblem::OM_ProblemRecompilation, tspan::Tuple, alg; kwargs...)::Vector{Any}
  local problem::ODEProblem = omProblem.problem
  local integrator::DEIntegrator = init(problem, alg; kwargs...)
  local solutions::Vector{Any} = Any[]
  while true
    #= In the latest world: a recompiled mode's functions are newer than this solve. Once per mode, not per step: a
       function barrier, its steps compiled for the mode's integrator. =#
    local change::Union{Nothing, AbstractStructuralChange} =
      Base.invokelatest(_integrateSegment!, integrator, omProblem.structuralCallbacks, tspan, solutions)
    change === nothing && return solutions
    (problem, integrator) = _switchMode(change, problem, integrator, tspan, omProblem.callbackConditions, alg; kwargs...)
  end
end

"""
    _integrateSegment!(integrator::DEIntegrator, structuralCallbacks::Vector{AbstractStructuralChange}, tspan::Tuple,
                       solutions::Vector{Any}) -> Union{Nothing, AbstractStructuralChange}

Steps the active mode's `integrator` until one of the structural callbacks has a change pending (returned; the mode's
solution up to the change, its `solutionAtChange`, pushed to `solutions`) or to the end of `tspan` (nothing; the mode's
solution up to the end pushed). An integration that fails, is terminated or stops advancing ends there (nothing; its
solution pushed). A step's return code is Success (or Default) while the integration goes on.
"""
function _integrateSegment!(integrator::DEIntegrator, structuralCallbacks::Vector{AbstractStructuralChange},
                            tspan::Tuple, solutions::Vector{Any})::Union{Nothing, AbstractStructuralChange}
  local tEnd::Float64 = last(tspan)
  while true
    for cb in structuralCallbacks
      (cb.structureChanged && integrator.t <= tEnd) && return cb
    end
    local tBefore::Float64 = integrator.t
    #= the last step: at least 1e-6, stopping at the end =#
    if integrator.t + integrator.dtpropose >= tEnd
      @VSS_DEBUG @info "Timestep exceeds simulation, adjusting" integrator.t (integrator.t + integrator.dtpropose)
      integrator.dt = 1e-6
      integrator.dtpropose = 1e-6
      step!(integrator, integrator.dt, true)
    else
      step!(integrator, integrator.dt, false)
    end
    local retcode::ReturnCode.T = integrator.sol.retcode
    local going::Bool = retcode == ReturnCode.Default || retcode == ReturnCode.Success
    #= A structural callback's affect stops the integration at its change (terminate!, just_hit_tstop): the mode's
       solution up to the change, the integrator back at the change; the change is made at the loop's start. =#
    if going && integrator.just_hit_tstop
      local changed::Union{Nothing, AbstractStructuralChange} = nothing
      for cb in structuralCallbacks
        cb.structureChanged || continue
        _truncateAtChange!(cb)
        push!(solutions, cb.solutionAtChange)
        changed = cb
      end
      if changed !== nothing
        reinit!(integrator, last(changed.solutionAtChange.u); t0 = changed.timeAtChange, tf = tEnd, reset_dt = true,
                erase_sol = true, reinit_callbacks = false)
        integrator.just_hit_tstop = false
        continue
      end
    end
    (going && tBefore < integrator.t < tEnd) && continue
    if going && integrator.t >= tEnd
      #= the last step may end slightly past the end =#
      local solution = integrator.sol
      local idx::Vector{Int} = findall(t -> t <= tEnd, solution.t)
      solution = Setfield.@set solution.t = solution.t[idx]
      solution = Setfield.@set solution.u = solution.u[idx]
      push!(solutions, solution)
    else
      @VSS_DEBUG @info "[solve OM_ProblemRecompilation] integration ended" retcode t = integrator.t
      push!(solutions, integrator.sol)
    end
    return nothing
  end
end

"""
    _truncateAtChange!(change::AbstractStructuralChange) -> Nothing

The solution up to a structural change (`solutionAtChange`, the copy the callback's affect took) cut at the time of the
change.
"""
function _truncateAtChange!(change::AbstractStructuralChange)::Nothing
  local solution = change.solutionAtChange
  local stopIdx::Union{Nothing, Int} = findlast(==(change.timeAtChange), solution.t)
  stopIdx === nothing && error("Invalid callback occurred during simulation: no solution point at the change (t = $(change.timeAtChange))")
  resize!(solution.t, stopIdx)
  resize!(solution.u, stopIdx)
  return nothing
end

"""
    _switchMode(change::AbstractStructuralChange, problem::ODEProblem, integrator::DEIntegrator, tspan::Tuple,
                callbackConditions::CallbackSet, alg; kwargs...) -> Tuple{ODEProblem, DEIntegrator}

The problem of the mode the structural change `change` switches to (`recompilation`) and its integrator at the change,
from the active mode's `problem` and `integrator` (back at the change): the new problem's unknowns at the values of the
old states of the same names (`RuntimeUtil.createNewU0`), others from the old observed variables
(`_fill_observed_u0!`). One step is taken, and the change reset (the structural event triggers again there).
"""
function _switchMode(change::AbstractStructuralChange, problem::ODEProblem, integrator::DEIntegrator, tspan::Tuple,
                     callbackConditions::CallbackSet, alg; kwargs...)::Tuple{ODEProblem, DEIntegrator}
  local started::Float64 = time()
  local tChange::Float64 = integrator.t
  @VSS_DEBUG @info "[solve OM_ProblemRecompilation] calling recompilation" t = tChange cbName = change.name
  local newProblem::ODEProblem = recompilation(change.name, change, integrator, tspan, callbackConditions)
  @VSS_DEBUG @info "[solve OM_ProblemRecompilation] recompilation returned" elapsed_s = round(time() - started, digits = 2)
  local oldSyms::Vector{Symbol} = getSyms(problem)
  #= newProblem.u0 as the base: its order is the new unknowns' (getSyms); createNewU0 sets the entries of the old
     states of the same names =#
  local newU0::Vector{Float64} = RuntimeUtil.createNewU0(oldSyms, getSyms(newProblem), collect(Float64, newProblem.u0),
                                                         last(change.solutionAtChange.u))
  #= new unknowns that are not old states: from the old solution's observed variables =#
  _fill_observed_u0!(newU0, oldSyms, newProblem, change.solutionAtChange)
  @VSS_DEBUG @info "[solve OM_ProblemRecompilation] new u0" newU0 anyNaN = any(isnan, newU0)
  #= In the latest world, as the steps: init runs the new mode's callbacks' initialization (a table's time events). =#
  local newIntegrator::DEIntegrator = Base.invokelatest(init, newProblem, alg; force_dtmin = true, kwargs...)
  Base.invokelatest(reinit!, newIntegrator, newU0; t0 = tChange, tf = last(tspan), reset_dt = true, kwargs...)
  #= The structural event may trigger again in this step: reset after it. =#
  newIntegrator.dt = min(newIntegrator.dt, 1e-6)
  Base.invokelatest(step!, newIntegrator, newIntegrator.dt, true)
  newIntegrator.just_hit_tstop = false
  change.structureChanged = false
  @VSS_DEBUG @info "[solve OM_ProblemRecompilation] switched" t = newIntegrator.t elapsed_s = round(time() - started, digits = 2)
  return (newProblem, newIntegrator)
end

"""
    recompilation(activeModeName::String, structuralCallback::StructuralChangeRecompilation, integrator::DEIntegrator,
                  tspan::Tuple, callbackConditions::CallbackSet) -> ODEProblem

The problem of the mode a structural change switches to from the active mode `activeModeName`, at the change
(`integrator`): the frontend's model with the change's element set to its new value (the `modification`'s function of
the active mode's symbol table), instantiated and compiled again with its variables starting from the values at the
change (`_compiledMode`), or the mode compiled before for the same changes and start values (RECOMPILED_MODES,
`_sameStarts`). The mode's problem for `tspan` with the composite model's callbacks `callbackConditions` ahead of its
own, not initialized by the solve (NoInit: `_switchMode` sets the state at the change). The structural callback
continues from the mode: its frontend model, symbol table and changes.
"""
function recompilation(activeModeName::String, structuralCallback::StructuralChangeRecompilation,
                       integrator::DEIntegrator, tspan::Tuple, callbackConditions::CallbackSet)::ODEProblem
  local runId::String = OMBackend.createLogRunId(activeModeName; suffix = "recompilation")
  return OMBackend.withLogRunDir(runId) do
    local started::Float64 = time()
    local elementToChange::String = first(structuralCallback.modification)
    local newValueOf::Function = last(structuralCallback.modification)
    local newValue = Base.invokelatest(newValueOf, structuralCallback.stringToSimVarHT)
    newValue isa Real ||
      error("Only primitive values {Integer, Boolean, Real} are currently supported in a Recompilation call, not $(repr(newValue))")
    local owner::String = OMBackend.canonicalName(activeModeName)
    local ownerModule::Module = Base.invokelatest(getglobal, OMBackend, Symbol(owner))
    local applied::Vector{Pair{String, String}} = _withChange(structuralCallback.applied, elementToChange, newValue)
    local key::RecompiledModeKey = (owner, ownerModule, objectid(structuralCallback.origin), applied)
    local values::Dict{String, Float64} = RuntimeUtil.createLookupTable(integrator)
    local cached::Union{Nothing, RecompiledMode} = get(RECOMPILED_MODES, key, nothing)
    local starts::Dict{String, Tuple{Float64, Bool}} =
      cached === nothing ? Dict{String, Tuple{Float64, Bool}}() : RuntimeUtil.switchStartValues(cached.originalHT, values)
    local reuse::Bool = cached !== nothing && _sameStarts(cached, starts)
    @VSS_DEBUG @info "[RECOMPILATION]" activeModeName elementToChange newValue t = integrator.t compiledBefore = reuse
    local mode::RecompiledMode = if reuse
      cached
    else
      _compiledMode(structuralCallback.metaModel, activeModeName, owner, ownerModule, elementToChange, newValue, values,
                    tspan, started)
    end
    reuse || _rememberMode!(key, mode)
    local callback::CallbackSet =
      OMBackend.CodeGeneration._joinCallbacks(callbackConditions, get(mode.problem.kwargs, :callback, nothing))
    #= NoInit: the solve sets the state at the change (createNewU0); an initialization would undo reinits of the
       mode's events. A mode compiled before: its u0 with this change's start values, its parameters as built. =#
    local problem::ODEProblem = if reuse
      remake(mode.problem; u0 = _u0WithStarts(mode, starts), p = deepcopy(mode.parameters), tspan, callback,
             initializealg = NoInit(), lazy_initialization = true)
    else
      remake(mode.problem; callback, initializealg = NoInit(), lazy_initialization = true)
    end
    structuralCallback.metaModel = mode.frontendModel
    structuralCallback.stringToSimVarHT = mode.stringToSimVarHT
    structuralCallback.applied = applied
    @VSS_DEBUG @info "[RECOMPILATION] done" elapsed_s = round(time() - started, digits = 2)
    return problem
  end
end

"""
    _withChange(applied::Vector{Pair{String, String}}, element::String, value::Real) -> Vector{Pair{String, String}}

The changes `applied` with the element `element` set to `value`: each element's last value, sorted by element.
"""
function _withChange(applied::Vector{Pair{String, String}}, element::String, value::Real)::Vector{Pair{String, String}}
  local changes::Vector{Pair{String, String}} = filter(change -> first(change) != element, applied)
  push!(changes, element => string(value))
  return sort!(changes; by = first)
end

"""
    _compiledMode(metaModel, activeModeName::String, owner::String, ownerModule::Module, elementToChange::String,
                  newValue::Real, values::Dict{String, Float64}, tspan::Tuple, started::Float64) -> RecompiledMode

The mode of the frontend's model `metaModel` with the element `elementToChange` of the active mode `activeModeName` set
to `newValue` (`recompiledSimCode`), its variables starting from the values at the change `values`
(`RuntimeUtil.switchStartValues`), compiled into the module of the model `owner` (`ownerModule`: its build owns the
tables) and its problem built for `tspan`. `started`: the recompilation's start, for the VSS_DEBUG timings.
"""
function _compiledMode(metaModel, activeModeName::String, owner::String, ownerModule::Module, elementToChange::String,
                       newValue::Real, values::Dict{String, Float64}, tspan::Tuple, started::Float64)::RecompiledMode
  local simulationCode::SimCode.SIM_CODE = recompiledSimCode(metaModel, activeModeName, elementToChange, newValue; started)
  @VSS_DEBUG @info "[RECOMPILATION] frontend and backend done" elapsed_s = round(time() - started, digits = 2)
  local originalHT::RuntimeUtil.SymbolTable = copy(simulationCode.stringToSimVarHT)
  local starts::Dict{String, Tuple{Float64, Bool}} = RuntimeUtil.switchStartValues(originalHT, values)
  RuntimeUtil.updateInitialConditions!(simulationCode, starts)
  Core.eval(ownerModule, translateToMTK(simulationCode, owner))
  @VSS_DEBUG @info "[RECOMPILATION] code generated and evaluated" elapsed_s = round(time() - started, digits = 2)
  local recompiled::Module = Base.invokelatest(getglobal, ownerModule, :Recompiled)
  local modelFunction::Function = Base.invokelatest(getglobal, recompiled, Symbol(owner, "Model"))
  local problem::ODEProblem = first(Base.invokelatest(modelFunction, tspan))
  @VSS_DEBUG @info "[RECOMPILATION] problem built" elapsed_s = round(time() - started, digits = 2)
  local unknowns::Vector{String} = String[string(u.f.name) for u in ModelingToolkit.unknowns(problem.f.sys)]
  local startsInU0::Dict{String, Int} = Dict{String, Int}()
  for (i, name) in enumerate(unknowns)
    local start::Union{Nothing, Tuple{Float64, Bool}} = get(starts, name, nothing)
    (start !== nothing && isequal(problem.u0[i], first(start))) && (startsInU0[name] = i)
  end
  local observed::Set{String} = Set{String}(string(eq.lhs.f.name) for eq in ModelingToolkit.observed(problem.f.sys))
  return RecompiledMode(problem, deepcopy(problem.p), simulationCode.frontendModel, simulationCode.stringToSimVarHT,
                        originalHT, starts, startsInU0, observed)
end

"""
    _sameStarts(mode::RecompiledMode, starts::Dict{String, Tuple{Float64, Bool}}) -> Bool

Whether a change that gives the recompiled model the start values `starts` (`RuntimeUtil.switchStartValues`) can take
the mode compiled before: the same variables get start values, by the same names (`updateInitialConditions!` sets
their other attributes by it), and the same values except where the solve does not read them (the unknowns whose start
the build took as their u0, set by `_u0WithStarts`; the observed variables).
"""
function _sameStarts(mode::RecompiledMode, starts::Dict{String, Tuple{Float64, Bool}})::Bool
  length(starts) == length(mode.starts) || return false
  for (name, (value, byLocalName)) in starts
    local before::Union{Nothing, Tuple{Float64, Bool}} = get(mode.starts, name, nothing)
    (before !== nothing && last(before) == byLocalName) || return false
    (haskey(mode.startsInU0, name) || name in mode.observed || isequal(first(before), value)) || return false
  end
  return true
end

"""
    _u0WithStarts(mode::RecompiledMode, starts::Dict{String, Tuple{Float64, Bool}}) -> Vector{Float64}

The u0 of the mode compiled before for a change that gives the start values `starts` (`_sameStarts`): its own, with
the new start values of the unknowns whose start the build took as their u0.
"""
function _u0WithStarts(mode::RecompiledMode, starts::Dict{String, Tuple{Float64, Bool}})::Vector{Float64}
  local u0::Vector{Float64} = collect(Float64, mode.problem.u0)
  for (name, i) in mode.startsInU0
    u0[i] = first(starts[name])
  end
  return u0
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
function recompilation(activeModeName::String, structuralCallback::StructuralChangeAgenticRecompilation,
                       integrator::DEIntegrator, tspan::Tuple, callbackConditions::CallbackSet)::ODEProblem
  local runId = OMBackend.createLogRunId(activeModeName; suffix = "agentic_recompilation")
  return OMBackend.withLogRunDir(runId) do
    @info "[RECOMPILATION] Agentic recompilation triggered at t=$(integrator.t), components=$(structuralCallback.componentsToChange)"
    #= Build full agent context using MTK symbolic indexing for correctness =#
    local solAtChange = structuralCallback.solutionAtChange
    local sysAtChange = solAtChange.prob.f.sys
    #= State variable values at the change instant =#
    local stateVars = Dict{String,Float64}()
    for sym in ModelingToolkit.get_unknowns(sysAtChange)
      try
        stateVars[string(ModelingToolkit.getname(sym))] = last(solAtChange[sym])
      catch err
        OMBackend._fallback(err, :agentStateSnapshot; impact = :result)
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
      catch err
        OMBackend._fallback(err, :agentCurrentValues; impact = :result)
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
      catch err
        OMBackend._fallback(err, :agentParameterSnapshot; impact = :result)
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
    local problem::Union{Nothing, ODEProblem} = nothing
    for (componentName, newValue) in zip(structuralCallback.componentsToChange, newValues)
      #= the agent's value, whatever the symbol table =#
      local newValueOf::Function = let value = newValue
        (_::RuntimeUtil.SymbolTable) -> value
      end
      local scr::StructuralChangeRecompilation = StructuralChangeRecompilation(activeModeName, false,
                                                                               structuralCallback.metaModel,
                                                                               (componentName, newValueOf),
                                                                               structuralCallback.stringToSimVarHT,
                                                                               integrator.t, Float64[])
      problem = recompilation(activeModeName, scr, integrator, tspan, callbackConditions)
      structuralCallback.metaModel = scr.metaModel
      structuralCallback.stringToSimVarHT = scr.stringToSimVarHT
    end
    problem === nothing && error("agentic_recompilation without parameters to change ($(activeModeName))")
    return problem
  end
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
    recompiledSimCode(metaModel, activeModeName::String, elementToChange::String, newValue::Real; started::Float64)
      -> SimCode.SIM_CODE

The simulation code of a model recompiled at a structural change: the frontend's model `metaModel` with the element
`elementToChange` of the active mode `activeModeName` set to `newValue`, instantiated again and through the backend, as
a translate's. Defined by the frontend's extension (OMBackendOMFrontendExt: an SCode class); `started`, the
recompilation's start for its VSS_DEBUG timings.
"""
function recompiledSimCode end

"""
    translateToMTK(simulationCode::SIM_CODE, owner::String) -> Expr

The module of a recompiled model (CodeGeneration.recompiledModelModule): its tables are owned by
the build of the model `owner`.
"""
function translateToMTK(simulationCode::OMBackend.SimCode.SIM_CODE, owner::String)::Expr
  local checkResult = OMBackend.SimCode.check(simulationCode)
  local canonicalErrors = filter(v -> v.rule === :canonical_cref_names && v.severity === :error,
                                 checkResult.violations)
  if !isempty(canonicalErrors)
    local details = join([string(v.where, ": ", v.detail) for v in canonicalErrors], "\n")
    error("SimCode still contains non-canonical names before runtime code generation:\n" * details)
  end
  return OMBackend.CodeGeneration.recompiledModelModule(simulationCode, owner)
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
  Get a vector of indices of the variables between syms1 and syms2.
  The destination mode might have a prefix for it's symbols.
  The prefix string is used to give the common variables the correct prefix
  between transitions.
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
