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
simulated as OMJL's `simulate` does: the build and `simulateModel` with OMJL's keyword arguments, so that the
keyword calls compile for a session's argument types. `simulate` translates again (the cached build reused). Simulated
with the default solver and with the DAE fallback (`defaultSolverFor`'s choice for a purely algebraic system or
discrete algebraic rows), so that both solvers' integrators over the erased problem types compile. `arrayCode`: whether
the model takes the array code (an error otherwise: the workload would compile less than it says).
"""
function _precompileTranslateAndSimulate(scode::OMFrontend.SCode.Program, name::String, arrayCode::Bool;
                                         simulate::Bool)::Nothing
  local mode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[]
  _precompileTranslate(scode, name, mode)
  (canonicalName(name) in ARRAY_ODE_MODELS) == arrayCode ||
    error("$(name): $(arrayCode ? "no array code" : "array code") ($(get(ARRAY_PATH_REASONS, canonicalName(name), "-")))")
  simulate || return nothing
  _precompileTranslate(scode, name, mode)
  for solver in (defaultSolver(), daeFallbackSolver())
    simulateModel(name; MODE = mode, tspan = (0.0, 1.0), solver = solver, overwriteCache = false)
  end
  return nothing
end

"""
OMJL's `translate` of the model `name` of `scode`: the frontend's flatten with the arrays kept and
`translateReducedArrays` (the array code, else the scalarized reduced model), else the scalarizing flatten and
`translate`.
"""
function _precompileTranslate(scode::OMFrontend.SCode.Program, name::String,
                              mode::OMBackend.BackendMode)::Union{Nothing, Tuple{String, Expr}}
  local (fm::OMFrontend.Frontend.FlatModel, cache) = OMFrontend.instantiateSCodeToFM(name, scode; scalarize = false)
  local reduced::Union{Nothing, Tuple{String, Expr}} =
    translateReducedArrays(fm; functionList = OMFrontend.cacheToFunctionList(cache), BackendMode = mode,
                           warnMissingStartValues = nothing, eliminateNonDynamic = true, observedFilter = nothing,
                           checkSimCode = true)
  reduced === nothing || return reduced
  (fm, cache) = OMFrontend.instantiateSCodeToFM(name, scode; scalarize = true)
  return translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), BackendMode = mode,
                   warnMissingStartValues = nothing, eliminateNonDynamic = true, observedFilter = nothing,
                   checkSimCode = true, scalarized = true)
end

"""
The translate and simulate of OMJL (`_precompileTranslateAndSimulate`) over the workload's models, so that their code
compiles into the extension's package image, not at each process's first model: `Precompile/Translate.mo`'s models of the
scalarized paths (Drive, Pendulum: the scalarizing flatten; Rectifier: the reduced model scalarized; Thermostat: the
array code with events) and `Precompile/ArraysKept.mo`'s of the array code (`Chain`: MultiBody's frames through
functions, slicing; `Volumes`: a semiLinear chain, a free parameter, a when-equation; `LoopEvents`: a torn loop, events,
an assert). A fresh process's first translate compiled 3-18 s of OMBackend's, MTK's and Symbolics' code before (MSL
4.1.0 Rotational First 12.9 s, MultiBody DoublePendulum 17.9 s; EngineV6 2371 methods for reducedArrayModel); the
registries are restored afterwards (`OMBackend._registries`). `simulate`: also build and simulate them, where an
evaluation into OMBackend is allowed (a session; not while the extension's image is generated: OMBackend is closed
then, `evaluationRejected`, and the models are only translated).
"""
function _precompileWarmupTranslate(; simulate::Bool = !OMBackend.evaluationRejected(OMBackend))::Nothing
  local registries::Vector{Pair{Any, Any}} = OMBackend._registries()
  try
    for (file, package, models) in ((TRANSLATE_WORKLOAD, "OMBackendTranslateWorkload",
                                     ("Drive" => false, "Thermostat" => true, "Rectifier" => false, "Pendulum" => false)),
                                    (ARRAYS_KEPT_WORKLOAD, "OMBackendWorkload",
                                     ("Chain" => true, "Volumes" => true, "LoopEvents" => true)))
      local scode::OMFrontend.SCode.Program = OMFrontend.translateToSCode(OMFrontend.parseFile(file))
      for (model, arrayCode) in models
        _precompileTranslateAndSimulate(scode, package * "." * model, arrayCode; simulate)
      end
    end
  finally
    OMBackend._restoreRegistries!(registries)
  end
  return nothing
end

@setup_workload begin
  @compile_workload begin
    #= Escape hatch for fast dev precompiles; a workload failure must never break loading, but must be seen. =#
    if !OMBackend.envSwitch("OMBACKEND_NO_PRECOMPILE_WORKLOAD")
      try
        Base.CoreLogging.with_logger(_precompileWarmupTranslate, Base.CoreLogging.NullLogger())
      catch err
        @warn "[OMBackend] translate precompile warmup skipped" exception = (err, catch_backtrace())
      end
    end
  end
end
