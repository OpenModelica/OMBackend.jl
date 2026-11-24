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

#= The extension's package image: OMJL's translate over small models of every route (`_precompileWarmupTranslate`).
   OMBackend is closed while the image is generated (an evaluation into it is rejected: `evaluationRejected`): the
   models are translated, not built or simulated; OMBackend's own image has the solves (`_precompileWarmup`,
   `_precompileWarmupDirectRHS`). =#
using PrecompileTools: @setup_workload, @compile_workload

const ARRAYS_KEPT_WORKLOAD = joinpath(@__DIR__, "Precompile", "ArraysKept.mo")
include_dependency(ARRAYS_KEPT_WORKLOAD)
const TRANSLATE_WORKLOAD = joinpath(@__DIR__, "Precompile", "Translate.mo")
include_dependency(TRANSLATE_WORKLOAD)

"""
    _precompileTranslateAndSimulate(scode, name::String, arrayCode::Bool) -> Nothing

The model `name` of `scode` translated as OMJL's `translate` does (`_precompileTranslate`), and with `simulate`
simulated as OMJL's `simulate` does: the build and `OMBackend.simulate` with OMJL's keyword arguments, so that the
keyword calls compile for a session's argument types. `simulate` translates again (the cached build reused). Simulated
with the default solver and with the DAE fallback (`defaultSolverFor`'s choice for a purely algebraic system or
discrete algebraic rows), so that both solvers' integrators over the erased problem types compile. `arrayCode`: whether
the model takes the array code (an error otherwise: the workload would compile less than it says).
"""
function _precompileTranslateAndSimulate(scode::OMFrontend.SCode.Program, name::String, arrayCode::Bool;
                                         simulate::Bool)::Nothing
  local mode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[]
  _precompileTranslate(scode, name, mode)
  (canonicalName(name) in ARRAY_PRESERVE_DE_MODELS) == arrayCode ||
    error("$(name): $(arrayCode ? "no array code" : "array code") ($(get(ARRAY_PATH_REASONS, canonicalName(name), "-")))")
  simulate || return nothing
  _precompileTranslate(scode, name, mode)
  for solver in (defaultSolver(), daeFallbackSolver())
    OMBackend.simulate(name; mode, tspan = (0.0, 1.0), solver = solver, overwriteCache = false)
  end
  return nothing
end

"""
OMJL's `translate` of the model `name` of `scode`: the frontend's flatten with the arrays kept and
`translate(fm; scalarize = false)` (the array code, else the scalarized reduced model, else the kept arrays scalarized by the
backend); an error where none takes the model (OMJL's `ARRAY_PRESERVING_CODEGEN_NOT_SUPPORTED`: no second flatten).
"""
function _precompileTranslate(scode::OMFrontend.SCode.Program, name::String,
                              mode::OMBackend.BackendMode)::Union{Nothing, Tuple{String, Expr}}
  local (fm::OMFrontend.Frontend.FLAT_MODEL, cache) = OMFrontend.instantiateSCodeToFM(name, scode; scalarize = false)
  local reduced::Union{Nothing, Tuple{String, Expr}} =
    translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), mode, scalarize = false,
              warnMissingStartValues = nothing, eliminateNonDynamic = true, observedFilter = nothing, checkSimCode = true)
  reduced === nothing || return reduced
  error("$(name): array preserving codegen not supported ($(OMBackend.scalarizedFallbackReason(canonicalName(name))))")
end

"""
The translate and simulate of OMJL (`_precompileTranslateAndSimulate`) over the workload's models, so that their code
compiles into the extension's package image, not at each process's first model: `Precompile/Translate.mo`'s models
of scalars in the array code (Drive: rolled inertias the plan treats differently, taken as declared; Rectifier: the
array code's loop of scalars; Pendulum: its index reduced in the array form; Thermostat: the array code with events)
and `Precompile/ArraysKept.mo`'s of the array code (`Chain`: MultiBody's frames through functions, slicing; `Volumes`: a semiLinear chain, a free parameter, a when-equation; `LoopEvents`: a torn loop, events,
an assert). A fresh process's first translate compiled 3-18 s of OMBackend's, MTK's and Symbolics' code before (MSL
4.1.0 Rotational First 12.9 s, MultiBody DoublePendulum 17.9 s; EngineV6 2371 methods for reducedArrayModel); the
registries are restored afterwards (`OMBackend._registries`). `simulate`: also build and simulate them, where an
evaluation into OMBackend is allowed (a session; not while the extension's image is generated: OMBackend is closed
then, `evaluationRejected`, and the models are only translated). Each model is a step of the display on `io`
(`OMBackend._precompileStep`). Returns the number of models that failed.
"""
function _precompileWarmupTranslate(io::IOContext = IOContext(devnull);
                                    simulate::Bool = !OMBackend.evaluationRejected(OMBackend))::Int
  local registries::Vector{Pair{Any, Any}} = OMBackend._registries()
  local failed::Int = 0
  try
    local n::Int = sum(length(last(w)) for w in PRECOMPILE_TRANSLATE_MODELS)
    local i::Int = 0
    for (file, package, models) in PRECOMPILE_TRANSLATE_MODELS
      local scode::OMFrontend.SCode.Program = OMFrontend.translateToSCode(OMFrontend.parseFile(file))
      for (model, arrayCode) in models
        i += 1
        OMBackend._precompileStep(io, string(model, " (", basename(file), ")"); prefix = OMBackend._precompileBar(i, n),
                                  quiet = true) do
          _precompileTranslateAndSimulate(scode, package * "." * model, arrayCode; simulate)
        end || (failed += 1)
      end
    end
  finally
    OMBackend._restoreRegistries!(registries)
  end
  return failed
end

"""
The models of the translate workload (`_precompileWarmupTranslate`): a file, its package, and its models with
whether each takes the array code.
"""
const PRECOMPILE_TRANSLATE_MODELS = ((TRANSLATE_WORKLOAD, "OMBackendTranslateWorkload",
                                      ("Drive" => true, "Thermostat" => true, "Rectifier" => true, "Pendulum" => true)),
                                     (ARRAYS_KEPT_WORKLOAD, "OMBackendWorkload",
                                      ("Chain" => true, "Volumes" => true, "LoopEvents" => true)))

"""The time the extension's display started (`time_ns`), 0 when its workload did not run."""
const PRECOMPILE_START = Base.RefValue{UInt64}(0)

@setup_workload begin
  @compile_workload begin
    #= Escape hatch for fast dev precompiles; a workload failure must never break loading, but must be seen (a ✗
       line and the error). Of the log messages outside the steps (the parse of the workload's files) only the
       warnings show. =#
    if !OMBackend.envSwitch("OMBACKEND_NO_PRECOMPILE_WORKLOAD")
      local io::IOContext = OMBackend._precompileIO()
      PRECOMPILE_START[] = OMBackend._precompileHeader(io, "OMBackendOMFrontendExt",
                                                       "Modelica ▸ OMFrontend ▸ OMJLIR ▸ Passes ▸ SimCode ▸ Code generation")
      try
        Base.CoreLogging.with_logger(() -> _precompileWarmupTranslate(io),
                                     Base.CoreLogging.SimpleLogger(stderr, Base.CoreLogging.Warn))
      catch err
        @warn "[OMBackend] precompile workload: the translate workload failed" exception = (err, catch_backtrace())
      end
    end
  end
end

"""
    _precompileSignatures(file::String) -> Tuple{Int, Int}

The `precompile` statements of `file`, each evaluated on its own: a statement naming a function or type that was
renamed or removed is skipped (counted in one line), not an error that fails the extension's image and with it
OM.jl's load. Returns the number of statements and the number skipped.
"""
function _precompileSignatures(file::String)::Tuple{Int, Int}
  include_dependency(file)
  local total::Int = 0
  local stale::Int = 0
  for statement in Meta.parseall(read(file, String); filename = file).args
    statement isa Expr || continue
    total += 1
    try
      Core.eval(@__MODULE__, statement)
    catch err
      stale += 1
      @debug "[OMBackend] stale precompile signature skipped" statement exception = err
    end
  end
  return (total, stale)
end

#= what the workloads do not reach, from a trace (precompileSignatures.jl): with the workload's switch, as the
   workloads; its line and the total close the display, a hint to regenerate the file follows it. =#
if !OMBackend.envSwitch("OMBACKEND_NO_PRECOMPILE_WORKLOAD")
  let file::String = joinpath(@__DIR__, "precompileSignatures.jl"), io::IOContext = OMBackend._precompileIO(),
      t0::UInt64 = time_ns(), (total, stale) = _precompileSignatures(file)
    if PRECOMPILE_START[] > 0
      OMBackend._precompileLine(io, string("Traced precompile signatures (", stale > 0 ? "$(stale) of " : "", total,
                                           stale > 0 ? " stale)" : ")"), time_ns() - t0, false)
      OMBackend._precompileDone(io, PRECOMPILE_START[])
    end
    stale > 0 && @info "[OMBackend] precompile signatures: $(stale) of $(total) no longer resolve; regenerate $(basename(file))"
  end
end
