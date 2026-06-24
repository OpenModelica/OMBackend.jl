#= Warm the shared MTK + OrdinaryDiffEq method instances that every generated
   model module invokes at build/solve time (structural_simplify, the init-solve
   ODEProblem, and the Rodas5/FBDF mass-matrix solve). The per-model module is
   eval'd after package load and cannot be cached into the image; these shared
   callees can, which is where the first-call latency actually lives.

   The DirectRHS workload below is the load-bearing one: because type erasure
   makes every DirectRHS problem share one concrete `ODEProblem` type (the
   runtime-generated RHS/jac are hidden behind FunctionWrappers), `solve` on that
   type is a single specialization that can be baked into the package image here.
   A real model that produces the same erased type then reuses it from its first
   solve in a fresh session — paid once at build, not once per model. =#
using PrecompileTools: @setup_workload, @compile_workload

#= Small index-1 system whose algebraic row folds away, leaving a pure-ODE
   reduced system (UniformScaling mass matrix) — the most common DirectRHS class. =#
function _precompileBuildReduced()
  @independent_variables t
  D = Differential(t)
  @variables x(t) y(t) z(t)
  @parameters a
  eqs = [D(x) ~ -a * x + y,
         D(y) ~ x - y,
         0 ~ z - (x + y)]
  sys = ODESystem(eqs, t, [x, y, z], [a];
                  name = :OMBackendPrecompileWarmup,
                  guesses = [z => 0.0],
                  initialization_eqs = ModelingToolkit.Equation[])
  local reduced = CodeGeneration.structural_simplify(sys; simplify = true,
                                                      allow_parameter = true, split = false)
  return (reduced, [x => 1.0, y => 0.0], Dict(a => 1.0))
end

#= Mirrors the generated XModel build for the standard MTK System-form path
   (structural transitions / non-DirectRHS modes). =#
function _precompileWarmup()
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local prob = ODEProblem(reduced, merge(Dict(ivs), pars), (0.0, 1.0);
                          warn_initialize_determined = false,
                          build_initializeprob = true, fully_determined = false)
  prob = ModelingToolkit.SciMLBase.remake(prob; tspan = (0.0, 2.0))
  solve(prob, Rodas5(autodiff = false))
  solve(prob, FBDF(autodiff = false))
  return nothing
end

#= Bakes the type-erased DirectRHS solve into the image. Builds the problem
   through the real `buildDirectRHSProblem` (so the type matches what models
   produce) and solves with the two default solvers. =#
function _precompileWarmupDirectRHS()
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local callbacks = ModelingToolkit.SciMLBase.CallbackSet()
  local prob = CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), callbacks)
  solve(prob, Rodas5(autodiff = false))
  solve(prob, FBDF(autodiff = false))
  return nothing
end

#= Plain OMBackend-parented callback functions for the event-ful DirectRHS bake.
   They live in OMBackend so the callback-erasure whitelist guard collapses them
   exactly as it does the generated when-equation callbacks. The condition crosses
   zero during the warmup solve so the event-handling path is compiled too. =#
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

#= Bakes the type-erased event-ful DirectRHS solve into the image. With RHS/jac
   erasure the `F` param already matches the pure-ODE warmup; the only remaining
   model-specific axis is the callback, now collapsed to a model-independent type.
   Solving here compiles `solve(::ErasedEventFulType, ::alg)` once for the whole
   legacy when-equation class. =#
function _precompileWarmupDirectRHSEventful()::Nothing
  local prob = _precompileBuildDirectRHSEventfulProblem()
  solve(prob, Rodas5(autodiff = false))
  solve(prob, FBDF(autodiff = false))
  return nothing
end

#= Small index-1 DAE (a retained nonlinear algebraic constraint keeps a non-identity
   mass matrix) with a continuous event. The pure-ODE warmups above never exercise the
   DAE mass-matrix solve (sparse W assembly / reinit) nor the MTK `process_events`
   callback path that the chua-class models pay for at first run, so this bakes both. =#
function _precompileBuildReducedEventfulDAE()
  @independent_variables t
  D = Differential(t)
  @variables x(t) y(t)
  @parameters a
  local eqs = [D(x) ~ -a * x + y, 0 ~ y^3 + y - x]
  local ev = ModelingToolkit.SymbolicContinuousCallback([x ~ 0.5] => [x ~ 0.4];
               reinitializealg = ModelingToolkit.SciMLBase.NoInit())
  local sys = ODESystem(eqs, t, [x, y], [a]; name = :OMBackendPrecompileEventfulDAE,
                        continuous_events = [ev], guesses = [y => 0.0],
                        initialization_eqs = ModelingToolkit.Equation[])
  local reduced = CodeGeneration.structural_simplify(sys; simplify = true,
                                                      allow_parameter = true, split = false)
  return (reduced, [x => 1.0, y => 0.0], Dict(a => 1.0))
end

#= Bakes the DAE-with-continuous-callback DirectRHS solve (chua class: non-identity
   mass matrix + an MTK process_events callback) into the image. =#
function _precompileWarmupMTKCallbackDAE()::Nothing
  local (reduced, ivs, pars) = _precompileBuildReducedEventfulDAE()
  local prob = CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0),
                                                    ModelingToolkit.SciMLBase.CallbackSet())
  prob = _precompileCollapseCallback(prob)
  solve(prob, Rodas5(autodiff = false))
  solve(prob, FBDF(autodiff = false))
  return nothing
end

@setup_workload begin
  @compile_workload begin
    #= Escape hatch for fast dev precompiles; a workload failure must never break
       loading, so each is demoted to debug. =#
    if get(ENV, "OMBACKEND_NO_PRECOMPILE_WORKLOAD", "") == ""
      try
        _precompileWarmup()
      catch err
        @debug "[OMBackend] precompile warmup skipped" exception = err
      end
      try
        _precompileWarmupDirectRHS()
      catch err
        @debug "[OMBackend] DirectRHS precompile warmup skipped" exception = err
      end
      try
        _precompileWarmupDirectRHSEventful()
      catch err
        @debug "[OMBackend] event-ful DirectRHS precompile warmup skipped" exception = err
      end
      try
        _precompileWarmupMTKCallbackDAE()
      catch err
        @debug "[OMBackend] MTK-callback DAE precompile warmup skipped" exception = err
      end
    end
  end
end
