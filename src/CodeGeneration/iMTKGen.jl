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
#= Hash of (modelCode, build-affecting flags) for each cached build. A repeat
   translate (e.g. overwriteCache) whose regenerated code and flags are
   unchanged reuses the live module + cached build instead of re-eval'ing, which
   would otherwise force redundant recompilation of the generated
   `simulateFromBuild` / `simulate` methods on the next solve. =#
const BUILT_HASH      = Dict{String, UInt64}()
const REDUCED_SYSTEMS = Dict{String, Any}()
const DUMP_PATHS      = Dict{String, String}()
#= Pristine parameter snapshot per build: event affects mutate the problem's
   shared parameter vector in place, so cached re-solves must restore it. =#
const PRISTINE_P      = Dict{String, Any}()
#= Debug hook: when OMJL_STASH_MODELCODE is set, stash the generated model Expr
   and skip Core.eval. Lets a caller inspect a model that OOMs at eval/simplify. =#
const LAST_MODELCODE  = Ref{Any}(nothing)
#= The tunable parameters (canonical names, `TUNABLE_PARAMETERS` at translate
   time) each cached build was compiled with: only these can be set per run.
   Being a parameter of the problem is not enough: a parameter referenced by a
   start attribute stays one, but its value is compiled into the equations. =#
const TUNABLE_SETS    = Dict{String, Set{String}}()

#= Forget a model's cached build (a failed or skipped build, a translate in
   another mode): an older build must not answer for the current one. =#
function forgetBuild(cname::String)
  for cache in (BUILT, BUILT_HASH, REDUCED_SYSTEMS, PRISTINE_P, TUNABLE_SETS)
    delete!(cache, cname)
  end
  return nothing
end

@inline _OMBackend() = parentmodule(CodeGeneration)

"""
    generateIMTKCode(simCode) -> (modelName, modelCode::Expr)

Build the module via `generateMTKCode`, then construct the System and run
`structural_simplify` in the backend, caching the build for reuse at simulate.
"""
function generateIMTKCode(simCode::SimulationCode.SIM_CODE)
  local (modelName, modelCode) = CodeGeneration.generateMTKCode(simCode)
  local cname = _OMBackend().canonicalName(modelName)
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
    TUNABLE_SETS[cname] = copy(_OMBackend().TUNABLE_PARAMETERS[])
    _buildAndCache(modelName, modelCode)
  else
    forgetBuild(cname)
    @info "[IMTK GEN] structural / sub-model / flat-model path; build-cache skipped (iMTK delegates to MTK simulate)" model = modelName
  end
  return (modelName, modelCode)
end

#= Eval the module, invoke `<name>Model(IMTK_BUILD_TSPAN)` (runs structural_simplify),
   and cache the resulting 9-tuple. The post-simplify System (element 5) is also
   stashed for the optional dump and external inspection via `reducedSystem`. =#
function _buildAndCache(modelName::String, modelCode::Expr; overwriteCache::Bool = false)
  local OMB = _OMBackend()
  local cname = OMB.canonicalName(modelName)
  #= Reuse path: identical regenerated code + build flags, a live module and a
     cached build mean the compiled methods and the problem are still valid.
     Re-eval'ing identical code would only invalidate `simulateFromBuild` and
     friends, forcing a full recompile on the next solve. Flags read at build
     time (not codegen) are folded into the hash so a flag flip still rebuilds.
     `overwriteCache` bypasses this reuse check to force a fresh rebuild. =#
  local buildHash = hash((modelCode, OMB.DIRECT_RHS_GENERATION[],
                          OMB.DIRECT_JAC_GENERATION[], OMB.DIRECT_RHS_TYPE_ERASE[]))
  if !overwriteCache && get(BUILT_HASH, cname, UInt64(0)) == buildHash &&
     haskey(BUILT, cname) && isdefined(OMB, Symbol(cname))
    @info "[IMTK GEN] regenerated code unchanged; reusing compiled module + cached build" model = modelName
    return nothing
  end
  try
    if get(ENV, "OMJL_STASH_MODELCODE", "") != ""
      LAST_MODELCODE[] = modelCode
      forgetBuild(cname)
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
    BUILT_HASH[cname] = buildHash
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
    #= No stale build: a previous build of this model (other tunable
       parameters, an older version of it) must not answer for this one. =#
    forgetBuild(cname)
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

`parameters` (name => value pairs) sets tunable parameters
(`withTunableParameters`) for this run, on top of the pristine values; it
needs the cached build and never falls back.
"""
function simulateIMTK(modelName::String, tspan, solver; parameters = nothing, kwargs...)
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
      if parameters !== nothing
        prob = _setParameterValues(prob, parameters, modelName)
        #= A DAE's consistent initial state depends on the parameters: solve it
           again for these values (the build solved it for the compiled ones). =#
        local reinit = get(OMB.CodeGeneration.DAE_REINIT, OMB.Runtime.ModelingToolkit.SciMLBase.unwrapped_f(prob.f.f), nothing)
        reinit === nothing || (prob = OMB.Runtime.ModelingToolkit.SciMLBase.remake(prob; u0 = reinit(prob.p)))
      end
      #= Route through `mod.simulate(...; cached_build = rebuilt)` using the same
         closure form as the MTK path, so the body executes inside the model module
         and `global LATEST_REDUCED_SYSTEM = …` / `global LATEST_PROBLEM = …` are
         visible to subsequent introspection on the module.
         Collapse the continuous callbacks HERE, inside invokelatest (a settled world,
         the build call having returned), then remake: the FunctionWrappers built around
         the RGF-backed MTK callbacks now resolve to the live methods (build-time collapse
         captured a stale world -> wrong events), and the collapsed prob has the
         model-independent erased type whose `solve` is baked into the image. =#
      return Base.invokelatest() do
        local SB = OMB.Runtime.ModelingToolkit.SciMLBase
        local p = prob
        if OMB.DIRECT_RHS_TYPE_ERASE[]
          local cb = get(p.kwargs, :callback, nothing)
          cb === nothing ||
            (p = SB.remake(p; callback = OMB.CodeGeneration._eraseContinuousCallbacks(cb)))
        end
        local rebuilt = (p, cached[2], cached[3], cached[4], cached[5],
                         tspan, cached[7], cached[8], cached[9])
        getfield(OMB, Symbol(cname)).simulate(tspan, solver; cached_build = rebuilt, kwargs...)
      end
    catch e
      #= A user interrupt must propagate, not trigger a retry of the same solve;
         the fallback cannot apply `parameters`. =#
      (e isa InterruptException || parameters !== nothing) && rethrow()
      @warn "[IMTK] cached-build solve failed; falling back to module simulate" model = modelName exception = e
    end
  end
  parameters === nothing ||
    error("simulating $(modelName) with `parameters` needs its cached build (translate it in IMTK mode)")
  return Base.invokelatest() do
    getfield(OMB, Symbol(cname)).simulate(tspan, solver; kwargs...)
  end
end

#= A copy of `prob` with the tunable parameters `parameters` (name => value;
   Modelica or flattened names, array elements as `A[2][1]` or `A[2,1]`) set. =#
function _setParameterValues(prob, parameters, modelName)
  local OMB = _OMBackend()
  local MTK = OMB.Runtime.ModelingToolkit
  local out = MTK.SciMLBase.remake(prob; p = copy(prob.p))
  local tunable = get(TUNABLE_SETS, OMB.canonicalName(modelName), Set{String}())
  for (name, value) in parameters
    local cn = OMB.canonicalName(OMB._elementSubscripts(string(name)))
    local sym = Symbol(cn)
    (OMB.isTunableParameter(cn, tunable) && MTK.is_parameter(out, sym)) ||
      throw(ArgumentError("$(name) is not a parameter of the compiled $(modelName); compile it inside " *
                          "OMBackend.withTunableParameters to change it without recompiling"))
    MTK.setp(out, sym)(out, value)
  end
  return out
end

end #= module IMTKGen =#
