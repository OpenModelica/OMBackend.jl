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

#= Warm the shared MTK + OrdinaryDiffEq method instances that every generated
   model module invokes at build/solve time (structural_simplify, the init-solve
   ODEProblem, and the default (Rodas5P) and FBDF mass-matrix solves). The per-model module is
   eval'd after package load and cannot be cached into the image; these shared
   callees can, which is where the first-call latency actually lives.

   The DirectRHS workload below is the load-bearing one: because type erasure
   makes every DirectRHS problem share one concrete `ODEProblem` type (the
   runtime-generated RHS/jac are hidden behind FunctionWrappers), `solve` on that
   type is a single specialization that can be baked into the package image here.
   A real model that produces the same erased type then reuses it from its first
   solve in a fresh session — paid once at build, not once per model. =#
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
Bakes the type-erased DirectRHS solve into the image. Builds the problem
through the real `buildDirectRHSProblem` (so the type matches what models
produce) and solves it as a model's build does (`_precompileSolveAsModels`).
"""
function _precompileWarmupDirectRHS()
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local callbacks = CodeGeneration._joinCallbacks()
  _precompileSolveAsModels(CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), callbacks), callbacks)
  return nothing
end

"""
    _precompileSolveAsModels(prob, buildCallbacks)

The solves of a model's build of `prob` from `buildCallbacks` (the generated build and `simulateFromBuild`, and
simulateIMTK): its step control for a DAE (withAlgebraicStepControl, withProblemCallbacks: the solve's
`isoutofdomain`), the callback set collapsed and erased, the solver `defaultSolverFor` picks from the default solver
and from the DAE fallback, the default initialization and tolerances: the integrator types a model's solve has, compiled
into the image. Solved without them, every process's first model compiled `solve!` again (2026-11-04).
"""
function _precompileSolveAsModels(prob, buildCallbacks)::Nothing
  local callbacks = CodeGeneration.withAlgebraicStepControl(buildCallbacks, prob)
  prob = _precompileCollapseCallback(CodeGeneration.withProblemCallbacks(prob, buildCallbacks, callbacks))
  local initKw = CodeGeneration.defaultInitializeKwargs(prob, (;), false)
  local tolKw = CodeGeneration.defaultToleranceKwargs(prob, (;))
  for solver in (defaultSolver(), daeFallbackSolver())
    solve(prob, CodeGeneration.defaultSolverFor(solver, prob, prob.f.sys, Set{Symbol}(), false); initKw..., tolKw...)
  end
  return nothing
end

"""
Plain OMBackend-parented callback functions for the event-ful DirectRHS bake.
They live in OMBackend so the callback-erasure whitelist guard collapses them
exactly as it does the generated when-equation callbacks. The condition crosses
zero during the warmup solve so the event-handling path is compiled too.
"""
function _precompileCBCondition(u::Vector{Float64}, t::Float64, integrator)::Float64
  return u[1] - 0.5
end

function _precompileCBAffect!(integrator)::Nothing
  return nothing
end

"""
simulateIMTK's solve-time collapse and erasure of the callbacks, so the baked solve is over the callback type the
runtime produces.
"""
function _precompileCollapseCallback(prob)
  local cb = get(prob.kwargs, :callback, nothing)
  cb === nothing || (cb = CodeGeneration._eraseContinuousCallbacks(cb))
  return ModelingToolkit.SciMLBase.remake(prob; callback = CodeGeneration._eraseCallbackSet(cb))
end

"""
Bakes the type-erased event-ful DirectRHS solve into the image: a pure-ODE problem carrying one continuous callback,
which collapses to the model-independent `VectorContinuousCallback` of the legacy when-equation class; the event
handling of the erased set compiled too (the condition crosses zero in the solve).
"""
function _precompileWarmupDirectRHSEventful()::Nothing
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local SB = ModelingToolkit.SciMLBase
  local cbset = CodeGeneration._joinCallbacks(SB.ContinuousCallback(_precompileCBCondition, _precompileCBAffect!))
  _precompileSolveAsModels(CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), cbset), cbset)
  return nothing
end

"""
Small index-1 DAE (a retained nonlinear algebraic constraint keeps a non-identity
mass matrix) with a continuous event. The pure-ODE warmups above never exercise the
DAE mass-matrix solve (sparse W assembly / reinit) nor the MTK `process_events`
callback path that the chua-class models pay for at first run, so this bakes both.
"""
function _precompileBuildReducedEventfulDAE()
  @independent_variables t
  D = Differential(t)
  @variables x(t) y(t)
  @parameters a
  local eqs = [D(x) ~ -a * x + y, 0 ~ y^3 + y - x]
  local ev = ModelingToolkit.SymbolicContinuousCallback([x ~ 0.5] => [x ~ 0.4];
               reinitializealg = ModelingToolkit.SciMLBase.NoInit())
  local sys = System(eqs, t, [x, y], [a]; name = :OMBackendPrecompileEventfulDAE,
                        continuous_events = [ev], guesses = [y => 0.0],
                        initialization_eqs = ModelingToolkit.Equation[])
  local reduced = CodeGeneration.structural_simplify(sys; simplify = true,
                                                      allow_parameter = true, split = false)
  return (reduced, [x => 1.0, y => 0.0], Dict(a => 1.0))
end

"""
Bakes the DAE-with-continuous-callback DirectRHS solve (chua class: non-identity
mass matrix + an MTK process_events callback) into the image.
"""
function _precompileWarmupMTKCallbackDAE()::Nothing
  local (reduced, ivs, pars) = _precompileBuildReducedEventfulDAE()
  local callbacks = CodeGeneration._joinCallbacks()
  #= With the initialization a model's solve uses (defaultInitializeKwargs, in
     _precompileSolveAsModels): the default only checks u0, and this system's start
     values leave a residual (4.5e-6 > abstol), so the workload failed and warmed nothing. =#
  _precompileSolveAsModels(CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), callbacks), callbacks)
  return nothing
end

"""
Bakes the array path's solve into the image (CodeGeneration.ArrayODEGen modules: an RHS
the solve wraps, a sparse Jacobian pattern, an ArraySystem, an ArrayModelParameters).
Every array model without events shares these types, so its first solve in a fresh
session reuses this one.
"""
function _precompileArrayRHS!(du, u, p, t)
  du .= -u
  return nothing
end

"The warmup's time gradient (`_precompileWarmupArrayPath`): its right-hand side does not read t."
function _precompileArrayTgrad!(dT, u, p, t)
  fill!(dT, 0.0)
  return nothing
end

function _precompileWarmupArrayPath()::Nothing
  local AG = CodeGeneration.ArrayODEGen
  local p = AG.ArrayModelParameters(Float64[], Bool[], Float64[], Float64[], Bool[], Bool[], [false], [false], zeros(2),
                                    zeros(2), Bool[], Tuple{Float64, Vector{Float64}}[], Float64[], [0.0], [1.0], Float64[],
                                    AG.BlockSolver[], AG.ArrayRuntimeState(0, String[]), OMBackend)
  local sys = AG.ArraySystem(Dict{Symbol, Int}(), Symbol[], Dict{Symbol, Tuple{Symbol, Int}}(), OMBackend)
  local jac = CodeGeneration.Symbolics.SparseArrays.sparse([1, 2], [1, 2], ones(2), 2, 2)
  #= the time gradient wrapped as the array code wraps it (one type for every array model) =#
  local tgrad = CodeGeneration.DiffEqBase.wrapfun_jac_iip(_precompileArrayTgrad!, ([1.0, 2.0], [1.0, 2.0], p, 0.0))
  local f = ODEFunction(_precompileArrayRHS!; jac_prototype = jac, tgrad = tgrad, sys = sys)
  solve(ODEProblem(f, [1.0, 2.0], (0.0, 1.0), p), defaultSolver(); callback = nothing)
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
for a session's argument types. `simulate` translates again (the cached build reused). `arrayCode`: whether the model
takes the array code (an error otherwise: the workload would compile less than it says).
"""
function _precompileTranslateAndSimulate(scode::OMFrontend.SCode.Program, name::String, arrayCode::Bool)::Nothing
  local mode::BackendMode = DEFAULT_BACKEND_MODE[]
  _precompileTranslate(scode, name, mode)
  (canonicalName(name) in ARRAY_ODE_MODELS) == arrayCode ||
    error("$(name): $(arrayCode ? "no array code" : "array code") ($(get(ARRAY_PATH_REASONS, canonicalName(name), "-")))")
  _precompileTranslate(scode, name, mode)
  simulateModel(name; MODE = mode, tspan = (0.0, 1.0), solver = defaultSolver(), overwriteCache = false)
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
      try
        _precompileWarmupDirectRHS()
      catch err
        @warn "[OMBackend] DirectRHS precompile warmup skipped" exception = (err, catch_backtrace())
      end
      try
        _precompileWarmupDirectRHSEventful()
      catch err
        @warn "[OMBackend] event-ful DirectRHS precompile warmup skipped" exception = (err, catch_backtrace())
      end
      try
        _precompileWarmupMTKCallbackDAE()
      catch err
        @warn "[OMBackend] MTK-callback DAE precompile warmup skipped" exception = (err, catch_backtrace())
      end
      try
        _precompileWarmupArrayPath()
      catch err
        @warn "[OMBackend] array-path precompile warmup skipped" exception = (err, catch_backtrace())
      end
      try
        Base.CoreLogging.with_logger(_precompileWarmupTranslate, Base.CoreLogging.NullLogger())
      catch err
        @warn "[OMBackend] translate precompile warmup skipped" exception = (err, catch_backtrace())
      end
    end
  end
end
