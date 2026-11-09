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

#= The package image's workloads: OM.jl's translate and simulate over small models of every route
   (`_precompileWarmupTranslate`), the DirectRHS solve of a pure ODE (`_precompileWarmupDirectRHS`),
   and the MTK System-form solve of the structural (VSS) paths (`_precompileWarmup`). Type erasure (the right-hand side and Jacobian behind FunctionWrappers, the
   callback sets as vectors: `_eraseCallbackSet`) makes the problems of every model of a route one
   type, so the solves compiled here are the ones a session's models run; a process's first model
   compiles its own generated code. =#
using PrecompileTools: @setup_workload, @compile_workload

"""
Small index-1 system whose algebraic row folds away, leaving a pure-ODE
reduced system (UniformScaling mass matrix) — the most common DirectRHS class.
"""
function _precompileBuildReduced()
  @independent_variables t
  D = Differential(t)
  @variables x(t) y(t) z(t)
  @parameters a
  eqs = [D(x) ~ -a * x + y,
         D(y) ~ x - y,
         0 ~ z - (x + y)]
  sys = System(eqs, t, [x, y, z], [a];
                  name = :OMBackendPrecompileWarmup,
                  guesses = [z => 0.0],
                  initialization_eqs = ModelingToolkit.Equation[])
  local reduced = CodeGeneration.structural_simplify(sys; simplify = true,
                                                      allow_parameter = true, split = false)
  return (reduced, [x => 1.0, y => 0.0], Dict(a => 1.0))
end

"""
Mirrors the generated XModel build for the standard MTK System-form path
(structural transitions / non-DirectRHS modes).
"""
function _precompileWarmup()
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local prob = ODEProblem(reduced, merge(Dict(ivs), pars), (0.0, 1.0);
                          warn_initialize_determined = false,
                          build_initializeprob = true, fully_determined = false)
  prob = ModelingToolkit.SciMLBase.remake(prob; tspan = (0.0, 2.0))
  solve(prob, defaultSolver())
  solve(prob, daeFallbackSolver())
  return nothing
end

"""
    _precompileWarmupDirectRHS(eventful::Bool) -> Nothing

The type-erased DirectRHS solve of a pure ODE (`_precompileBuildReduced`: a UniformScaling mass matrix), as a model's
build solves it (`_precompileSolveAsModels`), without callbacks or (`eventful`) with one continuous callback whose
condition crosses zero in the solve. None of the translate workload's scalarized models is a pure ODE: without these,
MSL Thermal Motor's first simulate compiled 1.4 s more (2026-11-07).
"""
function _precompileWarmupDirectRHS(eventful::Bool)::Nothing
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local callbacks::ModelingToolkit.SciMLBase.CallbackSet = eventful ?
    CodeGeneration._joinCallbacks(ModelingToolkit.SciMLBase.ContinuousCallback(_precompileCBCondition, _precompileCBAffect!)) :
    CodeGeneration._joinCallbacks()
  _precompileSolveAsModels(CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), callbacks), callbacks)
  return nothing
end

"The condition of `_precompileWarmupDirectRHS`'s callback: it crosses zero in the solve."
_precompileCBCondition(u::Vector{Float64}, t::Float64, integrator)::Float64 = u[1] - 0.5

"The affect of `_precompileWarmupDirectRHS`'s callback."
_precompileCBAffect!(integrator)::Nothing = nothing

"""
    _precompileSolveAsModels(prob, buildCallbacks)

The solves of a model's build of `prob` from `buildCallbacks` (the generated build and `simulateFromBuild`, and
simulateIMTK): its step control for a DAE (withAlgebraicStepControl, withProblemCallbacks: the solve's
`isoutofdomain`), the callbacks erased (`_eraseCallbackSet`), the solver `defaultSolverFor` picks from the default solver
and from the DAE fallback, the default initialization and tolerances: the integrator types a model's solve has, compiled
into the image. Solved without them, every process's first model compiled `solve!` again (2026-11-04).
"""
function _precompileSolveAsModels(prob::ModelingToolkit.SciMLBase.AbstractODEProblem,
                                  buildCallbacks::ModelingToolkit.SciMLBase.CallbackSet)::Nothing
  local callbacks::ModelingToolkit.SciMLBase.CallbackSet = CodeGeneration.withAlgebraicStepControl(buildCallbacks, prob)
  prob = CodeGeneration.withProblemCallbacks(prob, buildCallbacks, callbacks)
  prob = ModelingToolkit.SciMLBase.remake(prob; callback = CodeGeneration._eraseCallbackSet(get(prob.kwargs, :callback, nothing)))
  local initKw = CodeGeneration.defaultInitializeKwargs(prob, (;), false)
  local tolKw = CodeGeneration.defaultToleranceKwargs(prob, (;))
  for solver in (defaultSolver(), daeFallbackSolver())
    solve(prob, CodeGeneration.defaultSolverFor(solver, prob, prob.f.sys, Set{Symbol}(), false); initKw..., tolKw...)
  end
  return nothing
end

const ARRAYS_KEPT_WORKLOAD = joinpath(@__DIR__, "Precompile", "ArraysKept.mo")
include_dependency(ARRAYS_KEPT_WORKLOAD)
const TRANSLATE_WORKLOAD = joinpath(@__DIR__, "Precompile", "Translate.mo")
include_dependency(TRANSLATE_WORKLOAD)

"""
    _precompileTranslateAndSimulate(scode, name::String, arrayCode::Bool) -> Nothing

The model `name` of `scode` translated and simulated as OM.jl's `translate` and `simulate` do
(`_precompileTranslate`); the build (the model's module evaluated into OMBackend, open while its own image is
generated: `evaluationRejected`) and `simulateModel` with OM.jl's keyword arguments, so that the keyword calls compile
for a session's argument types. `simulate` translates again (the cached build reused). Simulated with the default
solver and with the DAE fallback (`defaultSolverFor`'s choice for a purely algebraic system or discrete algebraic
rows), so that both solvers' integrators over the erased problem types compile into the image. `arrayCode`: whether
the model takes the array code (an error otherwise: the workload would compile less than it says).
"""
function _precompileTranslateAndSimulate(scode::OMFrontend.SCode.Program, name::String, arrayCode::Bool)::Nothing
  local mode::BackendMode = DEFAULT_BACKEND_MODE[]
  _precompileTranslate(scode, name, mode)
  (canonicalName(name) in ARRAY_ODE_MODELS) == arrayCode ||
    error("$(name): $(arrayCode ? "no array code" : "array code") ($(get(ARRAY_PATH_REASONS, canonicalName(name), "-")))")
  _precompileTranslate(scode, name, mode)
  for solver in (defaultSolver(), daeFallbackSolver())
    simulateModel(name; MODE = mode, tspan = (0.0, 1.0), solver = solver, overwriteCache = false)
  end
  return nothing
end

"""
OM.jl's `translate` of the model `name` of `scode`: the frontend's flatten with the arrays kept and
`translateReducedArrays` (the array code, else the scalarized reduced model), else the scalarizing flatten and
`translate`.
"""
function _precompileTranslate(scode::OMFrontend.SCode.Program, name::String, mode::BackendMode)::Union{Nothing, Tuple{String, Expr}}
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
The registries of OMBackend's modules (their mutable Dict and Set constants and their Refs), with copies of their
contents: the translate workload restores them (`_restoreRegistries!`), so that its models' builds, generated code,
function implementations and run directories are not kept in the image (its models' modules are).
"""
function _registries(mod::Module = @__MODULE__, found::Vector{Pair{Any, Any}} = Pair{Any, Any}[])::Vector{Pair{Any, Any}}
  for name in names(mod; all = true)
    (isdefined(mod, name) && isconst(mod, name)) || continue
    local value::Any = getglobal(mod, name)
    if value isa Module
      (value !== mod && parentmodule(value) === mod) && _registries(value, found)
    elseif value isa Union{AbstractDict, AbstractSet} && ismutable(value)
      push!(found, value => copy(value))
    elseif value isa Base.RefValue && isassigned(value)
      push!(found, value => value[])
    end
  end
  return found
end

"The registries `_registries` copied, with the contents they had."
function _restoreRegistries!(registries::Vector{Pair{Any, Any}})::Nothing
  for (registry, contents) in registries
    if registry isa AbstractDict
      empty!(registry)
      merge!(registry, contents)
    elseif registry isa AbstractSet
      empty!(registry)
      union!(registry, contents)
    else
      registry[] = contents
    end
  end
  return nothing
end

"""
The translate and simulate of OM.jl (`_precompileTranslateAndSimulate`) over the workload's models, so that their code
compiles into the package image, not at each process's first model: `Precompile/Translate.mo`'s models of the
scalarized paths (Drive, Pendulum: the scalarizing flatten; Rectifier: the reduced model scalarized; Thermostat: the
array code with events) and `Precompile/ArraysKept.mo`'s of the array code (`Chain`: MultiBody's frames through
functions, slicing; `Volumes`: a semiLinear chain, a free parameter, a when-equation; `LoopEvents`: a torn loop, events,
an assert). A fresh process's first translate compiled 3-18 s of OMBackend's, MTK's and Symbolics' code before (MSL
4.1.0 Rotational First 12.9 s, MultiBody DoublePendulum 17.9 s; EngineV6 2371 methods for reducedArrayModel); the
registries are restored afterwards (`_registries`).
"""
function _precompileWarmupTranslate()::Nothing
  local registries::Vector{Pair{Any, Any}} = _registries()
  try
    for (file, package, models) in ((TRANSLATE_WORKLOAD, "OMBackendTranslateWorkload",
                                     ("Drive" => false, "Thermostat" => true, "Rectifier" => false, "Pendulum" => false)),
                                    (ARRAYS_KEPT_WORKLOAD, "OMBackendWorkload",
                                     ("Chain" => true, "Volumes" => true, "LoopEvents" => true)))
      local scode::OMFrontend.SCode.Program = OMFrontend.translateToSCode(OMFrontend.parseFile(file))
      for (model, arrayCode) in models
        _precompileTranslateAndSimulate(scode, package * "." * model, arrayCode)
      end
    end
  finally
    _restoreRegistries!(registries)
  end
  return nothing
end

@setup_workload begin
  @compile_workload begin
    #= Escape hatch for fast dev precompiles. A workload failure must never break
       loading, but must be seen: a rename or a broken path otherwise leaves the
       workload silently empty (efficiency review 2026-09-29). =#
    if !envSwitch("OMBACKEND_NO_PRECOMPILE_WORKLOAD")
      try
        _precompileWarmup()
      catch err
        @warn "[OMBackend] precompile warmup skipped" exception = (err, catch_backtrace())
      end
      for eventful in (false, true)
        try
          _precompileWarmupDirectRHS(eventful)
        catch err
          @warn "[OMBackend] DirectRHS precompile warmup skipped" eventful exception = (err, catch_backtrace())
        end
      end
      try
        Base.CoreLogging.with_logger(_precompileWarmupTranslate, Base.CoreLogging.NullLogger())
      catch err
        @warn "[OMBackend] translate precompile warmup skipped" exception = (err, catch_backtrace())
      end
    end
  end
end
