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

#= iMTKGen.jl — in-backend MTK path: reuses generateMTKCode and the generated
   module's `simulateFromBuild`. Builds + runs structural_simplify in the backend
   at translate time and caches the raw build tuple; simulate remakes the
   problem's tspan and delegates the post-build solve to `simulateFromBuild`, so
   iMTK is "MTK with the build cached" — same correctness, same code path.
   Selected via IMTK_MODE (see backendAPI.jl). =#
module IMTKGen

import ..CodeGeneration
import ..SimulationCode

#= Build tspan is arbitrary; simulateIMTK remakes the cached problem for its tspan. =#
const IMTK_BUILD_TSPAN = (0.0, 1.0)

#= Optional: dump the post-simplify System to backend/imtk/ when enabled. =#
const DUMP_ENABLED = Ref(false)

#= Per-model in-backend artifacts. `BUILT[cname]` is the raw 9-tuple returned
   by the generated `<name>Model(tspan)`: (problem, callbacks, ivs, _ivs_all,
   reducedSystem, tspan, pars, vars, irreducibleSyms). =#
const BUILT           = Dict{String, Tuple}()
const REDUCED_SYSTEMS = Dict{String, Any}()
const DUMP_PATHS      = Dict{String, String}()
#= Pristine parameter snapshot per build: event affects mutate the problem's
   shared parameter vector in place, so cached re-solves must restore it. =#
const PRISTINE_P      = Dict{String, Any}()
#= Debug hook: when OMJL_STASH_MODELCODE is set, stash the generated model Expr
   and skip Core.eval. Lets a caller inspect a model that OOMs at eval/simplify. =#
const LAST_MODELCODE  = Ref{Any}(nothing)

@inline _OMBackend() = parentmodule(CodeGeneration)

"""
    generateIMTKCode(simCode) -> (modelName, modelCode::Expr)

Build the module via `generateMTKCode`, then construct the System and run
`structural_simplify` in the backend, caching the build for reuse at simulate.
"""
function generateIMTKCode(simCode::SimulationCode.SIM_CODE)
  local (modelName, modelCode) = CodeGeneration.generateMTKCode(simCode)
  #= Only the standard PROGRAM_GENERATION path emits `simulateFromBuild` and
     returns the 9-tuple shape iMTK's cache assumes. Structural transitions,
     sub-models, and the flat-model path use MODEL_GENERATION's simpler
     simulate; skip _buildAndCache so iMTK falls through cleanly to the module
     `simulate` instead of warning loudly for every such model. Mirrors the
     condition in ODE_MODE_MTK (MTK_CodeGeneration.jl:415). =#
  if ccall(:jl_generating_output, Cint, ()) != 0
    #= Precompile/image generation: Core.eval'ing the model module into this
       closed backend module is rejected; skip the build+eval (codegen warmed). =#
  elseif !SimulationCode.hasStructuralTransitions(simCode) &&
         !SimulationCode.hasSubModels(simCode) &&
         !SimulationCode.hasFlatModel(simCode)
    _buildAndCache(modelName, modelCode)
  else
    @info "[IMTK GEN] structural / sub-model / flat-model path; build-cache skipped (iMTK delegates to MTK simulate)" model = modelName
  end
  return (modelName, modelCode)
end

#= Eval the module, invoke `<name>Model(IMTK_BUILD_TSPAN)` (runs structural_simplify),
   and cache the resulting 9-tuple. The post-simplify System (element 5) is also
   stashed for the optional dump and external inspection via `reducedSystem`. =#
function _buildAndCache(modelName::String, modelCode::Expr)
  local OMB = _OMBackend()
  local cname = OMB.canonicalName(modelName)
  try
    if get(ENV, "OMJL_STASH_MODELCODE", "") != ""
      LAST_MODELCODE[] = modelCode
      @info "[IMTK GEN] modelCode stashed; skipping Core.eval (OMJL_STASH_MODELCODE)" model = modelName
      return
    end
    if get(ENV, "OMJL_DUMP_IMTK_SRC", "") != ""
      try
        write("/tmp/imtk_$(cname).jl", string(modelCode))
      catch
      end
    end
    Core.eval(OMB, modelCode)
    local res = Base.invokelatest() do
      local mod = getfield(OMB, Symbol(modelName))
      local modelFn = getfield(mod, Symbol(string(modelName, "Model")))
      modelFn(IMTK_BUILD_TSPAN)
    end
    BUILT[cname] = res
    if res isa Tuple && length(res) >= 5
      REDUCED_SYSTEMS[cname] = res[5]
    end
    try
      PRISTINE_P[cname] = deepcopy(res[1].p)
    catch
      delete!(PRISTINE_P, cname)
    end
    @info "[IMTK GEN] structural_simplify ran in backend; build cached" model = modelName
    DUMP_ENABLED[] && _dumpReduced(OMB, modelName, cname)
  catch e
    @warn "[IMTK GEN] in-backend build / structural_simplify failed" model = modelName exception = e
  end
  return nothing
end

function _dumpReduced(OMB, modelName::String, cname::String)
  haskey(REDUCED_SYSTEMS, cname) || return nothing
  try
    local path = OMB.logPath("backend/imtk", string(modelName, "_reducedSystem.txt"))
    write(path, sprint(show, MIME("text/plain"), REDUCED_SYSTEMS[cname]))
    DUMP_PATHS[cname] = path
    @info "[IMTK GEN] dumped post-simplify system" model = modelName path = path
  catch e
    @warn "[IMTK GEN] reduced-system dump failed" model = modelName exception = e
  end
  return nothing
end

"Return the in-backend post-simplify `System` for an iMTK-translated model."
function reducedSystem(modelName::String)
  local OMB = _OMBackend()
  local cname = OMB.canonicalName(modelName)
  haskey(REDUCED_SYSTEMS, cname) && return REDUCED_SYSTEMS[cname]
  error("No iMTK reduced system for $(modelName); translate with mode = IMTK_MODE first.")
end

"""
    simulateIMTK(modelName, tspan, solver; kwargs...)

Reuse the build cached at translate time: remake the cached problem for `tspan`,
patch it into a rebuilt tuple, and call the model module's `simulateFromBuild`.
That delegate is the exact same post-build pipeline MTK-mode runs, so behavior
matches MTK except for skipping the rerun of `<name>Model(tspan)`. Falls back to
the module's own `simulate` on cache miss / unexpected failure.
"""
function simulateIMTK(modelName::String, tspan, solver; kwargs...)
  local OMB = _OMBackend()
  local cname = OMB.canonicalName(modelName)
  #= Structural/sub-model/flat-model iMTK builds skip _buildAndCache and are not
     eval'd at translate; eval on first simulate if absent (mirrors MTK_MODE). =#
  if !isdefined(OMB, Symbol(cname))
    Core.eval(OMB, OMB.getCompiledModel(cname))
  end
  if haskey(BUILT, cname)
    try
      local cached = BUILT[cname]
      local prob   = OMB.Runtime.ModelingToolkit.SciMLBase.remake(cached[1]; tspan = tspan)
      #= Restore the build-time parameter values: a previous run's affects may
         have mutated the shared vector (ifCond toggles persist otherwise). =#
      if haskey(PRISTINE_P, cname)
        prob = OMB.Runtime.ModelingToolkit.SciMLBase.remake(prob; p = deepcopy(PRISTINE_P[cname]))
      end
      local rebuilt = (prob, cached[2], cached[3], cached[4], cached[5],
                       tspan, cached[7], cached[8], cached[9])
      #= Route through `mod.simulate(...; cached_build = rebuilt)` using the same
         closure form as the MTK path, so the body executes inside the model module
         and `global LATEST_REDUCED_SYSTEM = …` / `global LATEST_PROBLEM = …` are
         visible to subsequent introspection on the module. =#
      return Base.invokelatest() do
        getfield(OMB, Symbol(cname)).simulate(tspan, solver; cached_build = rebuilt, kwargs...)
      end
    catch e
      #= A user interrupt must propagate, not trigger a retry of the same solve. =#
      e isa InterruptException && rethrow()
      @warn "[IMTK] cached-build solve failed; falling back to module simulate" model = modelName exception = e
    end
  end
  return Base.invokelatest() do
    getfield(OMB, Symbol(cname)).simulate(tspan, solver; kwargs...)
  end
end

end #= module IMTKGen =#
