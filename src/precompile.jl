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
produce) and solves with the two default solvers.
"""
function _precompileWarmupDirectRHS()
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local callbacks = ModelingToolkit.SciMLBase.CallbackSet()
  local prob = CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), callbacks)
  solve(prob, defaultSolver())
  solve(prob, daeFallbackSolver())
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

#= Builds a pure-ODE DirectRHS problem carrying one continuous callback, so the
   erased callback collapses to the single model-independent
   `VectorContinuousCallback` that the legacy when-equation class produces. Kept
   separate from the solve so the baked type can be checked against a real model. =#
# Mirror simulateIMTK's solve-time collapse so the baked solve is over the same erased
# callback type the runtime produces.
function _precompileCollapseCallback(prob)
  local cb = get(prob.kwargs, :callback, nothing)
  cb === nothing && return prob
  return ModelingToolkit.SciMLBase.remake(prob;
            callback = CodeGeneration._eraseContinuousCallbacks(cb))
end

function _precompileBuildDirectRHSEventfulProblem()
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local SB = ModelingToolkit.SciMLBase
  local cbset = SB.CallbackSet(SB.ContinuousCallback(_precompileCBCondition,
                                                     _precompileCBAffect!))
  local prob = CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), cbset)
  return _precompileCollapseCallback(prob)
end

"""
Bakes the type-erased event-ful DirectRHS solve into the image. With RHS/jac
erasure the `F` param already matches the pure-ODE warmup; the only remaining
model-specific axis is the callback, now collapsed to a model-independent type.
Solving here compiles `solve(::ErasedEventFulType, ::alg)` once for the whole
legacy when-equation class.
"""
function _precompileWarmupDirectRHSEventful()::Nothing
  local prob = _precompileBuildDirectRHSEventfulProblem()
  solve(prob, defaultSolver())
  solve(prob, daeFallbackSolver())
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
  local prob = CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0),
                                                    ModelingToolkit.SciMLBase.CallbackSet())
  prob = _precompileCollapseCallback(prob)
  #= With the initialization a model's solve uses (defaultInitializeKwargs): the
     default only checks u0, and this system's start values leave a residual
     (4.5e-6 > abstol), so the workload failed and warmed nothing. =#
  local initKw = CodeGeneration.defaultInitializeKwargs(prob, (;), false)
  solve(prob, defaultSolver(); initKw...)
  solve(prob, daeFallbackSolver(); initKw...)
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
                                    AG.BlockSolver[], OMBackend)
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

"""
The arrays-kept paths over the models of `Precompile/ArraysKept.mo`, as `translate` takes them, up to the generated
module's code (a module is not evaluated while the package image is built): the frontend's flatten with the arrays
kept, lowering and the passes, component rolling, index reduction in array form with its splits, the derivatives
solved in array form and the array code after index reduction (`Chain`: MultiBody's frames through functions, slicing;
`Volumes`: a semiLinear chain, a free parameter, a when-equation), and the array code of a model without index
reduction (`LoopEvents`: a torn loop, events, a when-equation, an assert). Their code then compiles into the package image, not at each process's first
such model: MSL EngineV6's first build compiled 2371 methods (18.7 s) for reducedArrayModel and 594 (3.8 s) for its
array code, of OMBackend's own code (`Traversal.traverseField`'s specializations by node and pass: 1073). Each model
must take its path (an error otherwise: the workload would compile less than it says). The registries those paths
write are cleared of the workload's models.
"""
function _precompileWarmupArraysKept()::Nothing
  local scode = OMFrontend.translateToSCode(OMFrontend.parseFile(ARRAYS_KEPT_WORKLOAD))
  for model in ("Chain", "Volumes", "LoopEvents")
    local name::String = "OMBackendWorkload." * model
    local cn::String = canonicalName(name)
    try
      local (fm::OMFrontend.Frontend.FlatModel, cache) = OMFrontend.instantiateSCodeToFM(name, scode; scalarize = false)
      local functions::FunctionList = OMFrontend.cacheToFunctionList(cache)
      local code::Tuple = if model == "LoopEvents"
        CodeGeneration.ArrayODEGen.generateArrayODECode(fm, cn; functions)
      else
        local reduced::Union{Nothing, OMJLIR.Model} = reducedArrayModel(fm; functionList = functions)
        reduced === nothing && error("$(name) not reduced in array form: $(scalarizedFallbackReason(cn))")
        local (solved::OMJLIR.Model, _) = Passes.IndexReduction.solvedInArrays(reduced)
        CodeGeneration.ArrayODEGen.generateArrayODECode(solved, cn; functions)
      end
      code[1] === nothing && error("$(name): no array code ($(code[2]))")
    finally
      delete!(ARRAY_PATH_REASONS, cn)
      delete!(ARRAY_REDUCED_MODELS, cn)
      delete!(ARRAY_ODE_MODELS, cn)
    end
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
        Base.CoreLogging.with_logger(_precompileWarmupArraysKept, Base.CoreLogging.NullLogger())
      catch err
        @warn "[OMBackend] arrays-kept precompile warmup skipped" exception = (err, catch_backtrace())
      end
    end
  end
end
