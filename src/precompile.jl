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
    end
  end
end
