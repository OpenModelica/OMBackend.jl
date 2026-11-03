#= Array-preserving code generation (translate with scalarized = false): models from the
   frontend's no-scalarize mode keep their loops in the generated code; results equal the
   scalarizing path's. =#
const ARRAY_PATH_FILE = joinpath(@__DIR__, "Models", "ArrayPath.mo")
const _ARRAY_PATH_SCODE = Ref{Any}(nothing)

function _arrayPathTranslate(model::String; scalarized::Bool)
  _ARRAY_PATH_SCODE[] === nothing &&
    (_ARRAY_PATH_SCODE[] = OMFrontend.translateToSCode(OMFrontend.parseFile(ARRAY_PATH_FILE)))
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, _ARRAY_PATH_SCODE[]; scalarize = scalarized)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarized = scalarized)
  return OMBackend.canonicalName(model)
end

_arrayPathTaken(name) = name in OMBackend.ARRAY_ODE_MODELS
_values(sol, names) = [OMBackend.getVariableValues(sol, n)[end] for n in names]

function _bothWays(model, names; tspan = (0.0, 1.0), kwargs...)
  local name = _arrayPathTranslate(model; scalarized = true)
  local s::Vector{Float64} = _values(OMBackend.simulateModel(name; tspan = tspan, kwargs...), names)
  _arrayPathTranslate(model; scalarized = false)
  @test _arrayPathTaken(name)
  local sol::OMBackend.DifferentialEquations.SciMLBase.AbstractODESolution = OMBackend.simulateModel(name; tspan = tspan, kwargs...)
  @test sol.retcode == ReturnCode.Success
  return (s, _values(sol, names), sol)
end

@testset "Array path" begin
  @testset "Decay, n = 1000: loops kept" begin
    local name = _arrayPathTranslate("ArrayPath.Decay"; scalarized = false)
    @test _arrayPathTaken(name)
    local stats = OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[]
    @test stats.states == 1000 && stats.batches == 1
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    @test maximum(abs(sol.u[end][i] - exp(-i)) for i in 1:1000) < 1e-6
    @test OMBackend.getVariableValues(sol, "x[2]")[end] ≈ exp(-2) atol = 1e-6
  end

  @testset "Chain: connections, algebraic variables, as scalarized" begin
    local (s, a, sol) = _bothWays("ArrayPath.Chain", ["c[1].T", "c[5].T", "c[10].T", "c[2].p.Q", "g[1].a.T"])
    @test maximum(abs.(s .- a)) < 1e-6
    local loops = OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches
    #= the code does not grow with n =#
    _arrayPathTranslate("ArrayPath.Chain1000"; scalarized = false)
    @test OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches == loops
    #= symbolic indexing: states and observed (algebraic) variables =#
    @test sol[Symbol("c[1].T")][end] ≈ a[1]
    @test sol(1.0; idxs = Symbol("c[2].p.Q")) ≈ a[4] atol = 1e-8
  end

  @testset "Parameters: overrides and dependent bindings" begin
    local name = _arrayPathTranslate("ArrayPath.ParamDep"; scalarized = false)
    @test _arrayPathTaken(name)
    local x = OMBackend.simulateModel(name; tspan = (0.0, 1.0)).u[end]
    @test x ≈ exp.(-[2.0, 4.0, 6.0]) rtol = 1e-4
    #= b = 3a and k = {a, 2a, b} follow a =#
    x = OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("a" => 1.0)).u[end]
    @test x ≈ exp.(-[1.0, 2.0, 3.0]) rtol = 1e-4
    x = OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("k[2]" => 0.5)).u[end]
    @test x[2] ≈ exp(-0.5) rtol = 1e-4
    #= an element override with an override of what the array reads: the array's rule, then the element (k[1] kept
       its translated 2 before) =#
    x = OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("a" => 1.0, "k[2]" => 0.5)).u[end]
    @test x ≈ exp.(-[1.0, 0.5, 3.0]) rtol = 1e-4
    #= a Real array whose rule gives Integers keeps its type (an element override of 0.5 threw InexactError) =#
    local ib = _arrayPathTranslate("ArrayPath.IntBoundRealArray"; scalarized = false)
    @test _arrayPathTaken(ib)
    local xi = OMBackend.resimulateModel(ib; tspan = (0.0, 1.0), parameters = Dict{String, Any}("m" => 3, "k[2]" => 0.5))
    @test OMBackend.getVariableValues(xi, "x")[end] ≈ 3.5 rtol = 1e-6
    #= parameter names that differ in . and _ only are their own parameters (one symbol: x(1) = 66, omc 26) =#
    local nc = _arrayPathTranslate("ArrayPath.ParamNameClash"; scalarized = false)
    @test _arrayPathTaken(nc)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(nc; tspan = (0.0, 1.0)), "x")[end] ≈ 26.0 rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(nc; tspan = (0.0, 1.0), parameters = Dict{String, Any}("c_k" => 1.0)), "x")[end] ≈ 13.0 rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(nc; tspan = (0.0, 1.0), parameters = Dict{String, Any}("c.k" => 5.0)), "x")[end] ≈ 25.0 rtol = 1e-6
    #= a binding that references another parameter follows its overrides (it was that one's value: gain = 5 gave 5532,
       omc 5535) =#
    local pr = _arrayPathTranslate("ArrayPath.ParamReference"; scalarized = false)
    @test _arrayPathTaken(pr)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(pr; tspan = (0.0, 1.0)), "x")[end] ≈ 5232.0 rtol = 1e-6
    local po = Dict{String, Any}("gain" => 5.0, "p1" => 4.0, "a" => [4.0, 6.0])
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(pr; tspan = (0.0, 1.0), parameters = po), "x")[end] ≈ 5 + 40 + 500 + 6000 rtol = 1e-6
    #= a parameter coefficient of der() follows its override (the array commit folded it: c = 4 kept 2) =#
    local dc = _arrayPathTranslate("ArrayPath.DerCoefficient"; scalarized = false)
    @test _arrayPathTaken(dc)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(dc; tspan = (0.0, 1.0)), "x")[end] ≈ 0.5 rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(dc; tspan = (0.0, 1.0), parameters = Dict{String, Any}("c" => 4.0)), "x")[end] ≈ 0.25 rtol = 1e-6
    #= a coefficient that is a call of a parameter: a parameter of its own (computed at each start, not in each right-hand
       side), which follows the override =#
    local dk = _arrayPathTranslate("ArrayPath.DerCoefficientOfCall"; scalarized = false)
    @test _arrayPathTaken(dk)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(dk; tspan = (0.0, 1.0)), "x")[end] ≈ 1 / sin(0.5) rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(dk; tspan = (0.0, 1.0), parameters = Dict{String, Any}("p" => 1.0)), "x")[end] ≈ 1 / sin(1.0) rtol = 1e-6
    #= starts that read parameters follow their overrides (they kept the translated values: 4) =#
    local sp = _arrayPathTranslate("ArrayPath.StartOfParameters"; scalarized = false)
    @test _arrayPathTaken(sp)
    local sps = OMBackend.resimulateModel(sp; tspan = (0.0, 1.0), parameters = Dict{String, Any}("p1" => 10.0))
    @test [OMBackend.getVariableValues(sps, v)[end] for v in ("y", "w")] ≈ [11.0, 11.0] rtol = 1e-6
    #= a fixed start that reads a parameter the initialization computes: y = q there (it took q's start, 1); the array
       code declines it (its initial algorithms run after the initialization), the scalar path's value omc's =#
    local sf = _arrayPathTranslate("ArrayPath.StartOfFreeParameter"; scalarized = false)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(sf; tspan = (0.0, 1.0)), "y")[1] ≈ 3.0 rtol = 1e-6
    #= a parameter a structural one's binding reads is structural too: its override is refused (it ran with the
       translated if-equation) =#
    local sd = _arrayPathTranslate("ArrayPath.StructuralDependency"; scalarized = false)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(sd; tspan = (0.0, 1.0)), "x")[end] ≈ 2.0 rtol = 1e-6
    @test_throws ErrorException OMBackend.resimulateModel(sd; tspan = (0.0, 1.0), parameters = Dict{String, Any}("R0" => -3.0))
    #= a call equation under an if-equation's branch: made where the branch holds (a TypeError: its condition was
       lifted into a Real) =#
    local cb = _arrayPathTranslate("ArrayPath.CallInIfBranch"; scalarized = false)
    @test _arrayPathTaken(cb)
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(cb; tspan = (0.0, 1.0), parameters = Dict{String, Any}("lim" => 2.0)), "x")[end] ≈ 1.0 rtol = 1e-6
    @test_throws Exception OMBackend.resimulateModel(cb; tspan = (0.0, 1.0))
    #= a sample interval that is not finite is refused (sampleTimes did not end) =#
    local se = _arrayPathTranslate("ArrayPath.SampleEquation"; scalarized = false)
    @test_throws ErrorException OMBackend.resimulateModel(se; tspan = (0.0, 1.0), parameters = Dict{String, Any}("samplePeriod" => Inf))
    @test_throws Exception OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("n" => 4))
  end

  @testset "ARRAY_PATH_FULL = false: events go the ModelingToolkit path" begin
    OMBackend.ARRAY_PATH_FULL[] = false
    try
      @test !_arrayPathTaken(_arrayPathTranslate("ArrayPath.BouncingBalls"; scalarized = false))
    finally
      OMBackend.ARRAY_PATH_FULL[] = true
    end
  end

  @testset "Events: when-equations in a loop, relations" begin
    local names = ["h[1]", "v[1]", "h[5]", "v[5]", "bounces[1]", "bounces[5]"]
    local (s, a, _) = _bothWays("ArrayPath.BouncingBalls", names; tspan = (0.0, 1.5))
    @test maximum(abs.(s .- a)) < 1e-4
    @test a[5] == 3.0
    local (s2, a2, _) = _bothWays("ArrayPath.Rectifier", ["vc[1]", "vc[3]"]; tspan = (0.0, 2.0))
    @test maximum(abs.(s2 .- a2)) < 1e-3
  end

  @testset "Asserts" begin
    local name = _arrayPathTranslate("ArrayPath.Guarded"; scalarized = false)
    @test _arrayPathTaken(name)
    @test_throws OMBackend.CodeGeneration.ModelicaAssertionError OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test OMBackend.simulateModel(name; tspan = (0.0, 0.4)).retcode == ReturnCode.Success
  end

  @testset "Initialization, sample, static if, discrete equations, algorithms" begin
    for (m, names, T) in (("ArrayPath.InitSteady", ["x", "y"], 1.0),
                          ("ArrayPath.SampleZOH", ["x", "u", "k"], 1.0),
                          ("ArrayPath.StaticIf", ["x[1]", "x[5]", "s", "sl"], 2.0),
                          ("ArrayPath.DiscreteEq", ["x", "m", "z[1]", "z[2]"], 2.0),
                          ("ArrayPath.AlgInitial", ["x", "t0", "z", "n"], 1.0))
      local (s, a, _) = _bothWays(m, names; tspan = (0.0, T))
      @test maximum(abs.(s .- a)) < 1e-5
    end
  end

  @testset "Event functions, vector when-conditions, fixed = false parameters, initial algorithms" begin
    #= at 0.95: floor(2.375), mod(0.95, 0.3), div(2.85, 1), integer(4x) rose at 0.25, 0.5, 0.75
       (not against the scalarized path: its when on integer(4x) > pre(n) fails, a BoundsError) =#
    local name = _arrayPathTranslate("ArrayPath.EventFunctions"; scalarized = false)
    @test _arrayPathTaken(name)
    local ev = OMBackend.simulateModel(name; tspan = (0.0, 0.95))
    @test _values(ev, ["y[1]", "y[2]", "y[3]", "n"]) ≈ [2.0, 0.05, 2.0, 3.0] atol = 1e-6
    #= the events: floor(2.5x) steps at 0.4 and 0.8, n at 0.25, 0.5, 0.75 =#
    @test [ev(t; idxs = Symbol("y[1]")) for t in (0.39, 0.41, 0.79, 0.81)] == [0.0, 1.0, 1.0, 2.0]
    @test [ev(t; idxs = :n) for t in (0.24, 0.26, 0.74, 0.76)] == [0.0, 1.0, 2.0, 3.0]
    #= initial() fires at the start, then each element when it becomes true =#
    local (_, v, _) = _bothWays("ArrayPath.VectorWhen", ["n", "tl"])
    @test v ≈ [3.0, 0.6] atol = 1e-6
    #= t0 = 0, k = x(0) / 2 = 1 =#
    local (s3, a3, _) = _bothWays("ArrayPath.FreeParam", ["x"])
    @test a3[1] ≈ 2 * exp(-1.0) rtol = 1e-5
    @test maximum(abs.(s3 .- a3)) < 1e-5
    #= a double exponential's free parameters (MSL LightningImpulse): the physical root, eta = 0.9511..., tau1 = 470 us,
       tau2 = 4.06 us, not tau1 = tau2 with eta = 0 (undamped Newton from the starts) =#
    local impulse = _arrayPathTranslate("ArrayPath.DoubleExpImpulse"; scalarized = false)
    @test _arrayPathTaken(impulse)
    local (eta, tau1, tau2) = (0.9511248490912196, 4.7010661173524463e-4, 4.0639516102066676e-6)
    local ys = _values(OMBackend.simulateModel(impulse; tspan = (0.0, 1e-3), reltol = 1e-10, abstol = 1e-12), ["y[1]", "y[2]", "x[1]"])
    @test ys ≈ [fill((exp(-1e-3 / tau1) - exp(-1e-3 / tau2)) / eta, 2); (tau1 * (1 - exp(-1e-3 / tau1)) - tau2 * (1 - exp(-1e-3 / tau2))) / eta] rtol = 1e-8
    local (s4, a4, _) = _bothWays("ArrayPath.InitAlgPulse", ["count", "T_start", "y"])
    @test maximum(abs.(s4 .- a4)) < 1e-6
    @test a4[1:2] ≈ [6.0, 0.9] atol = 1e-6
  end

  @testset "If-equations on variable conditions, asserts of initial equations" begin
    local (s, v, sol) = _bothWays("ArrayPath.IfDynamic", ["y", "z", "w"])
    @test maximum(abs.(s .- v)) < 1e-6
    @test [sol(t; idxs = :y) for t in (0.1, 0.3, 0.7)] == [2.0, 3.0, 1.0]
    local name = _arrayPathTranslate("ArrayPath.InitialAssert"; scalarized = false)
    @test _arrayPathTaken(name)
    local err = try
      OMBackend.simulateModel(name; tspan = (0.0, 1.0)); nothing
    catch e
      e
    end
    @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && err.time == 0.0
    @test startswith(err.message, "k must be positive, k = -1")
  end

  @testset "Modelica functions" begin
    #= the model's functions as Julia functions of the module (the MTK path's code
       generation); (p, q) = f(x) as an algorithm node (the MTK path fails on it) =#
    local name = _arrayPathTranslate("ArrayPath.Functions"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    local x = exp(-1.0)
    @test _values(sol, ["x", "y", "z", "p", "q"]) ≈ [x, 2x, 3x + 1, x^2, -x] rtol = 1e-5
    #= v' = -2 v / |v|: the direction stays, |v| = sqrt(14) - 2t =#
    local r = sqrt(14.0) - 2.0
    @test _values(sol, ["v[1]", "v[3]", "n", "w[2]"]) ≈ [r / sqrt(14.0), 3r / sqrt(14.0), r, 4r / sqrt(14.0)] rtol = 1e-5
  end

  @testset "elsewhen, calls without a result, asserts in when bodies" begin
    local name = _arrayPathTranslate("ArrayPath.ElseWhen"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = @test_logs (:warn, r"n reached 11") match_mode = :any OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    #= x > 0.3: the first branch; x > 0.6 (the first condition still true): the second =#
    @test [sol(t; idxs = :n) for t in (0.2, 0.4, 0.7)] == [0.0, 1.0, 11.0]
    @test sol(1.0; idxs = :m) == 1.0
  end

    @testset "An assert's message and time" begin
    local name = _arrayPathTranslate("ArrayPath.AssertMessage"; scalarized = false)
    @test _arrayPathTaken(name)
    local err = try
      OMBackend.simulateModel(name; tspan = (0.0, 1.0)); nothing
    catch e
      e
    end
    @test err isa OMBackend.CodeGeneration.ModelicaAssertionError
    @test err.time ≈ 0.5 atol = 1e-6
    @test startswith(err.message, "x reached 0.5")
  end

  #= blockSolve! from a guess at the solution: a fresh Jacobian (its evaluations at perturbed points), its first step
     at round-off; the equations' a was left at the last color's perturbed point (1.5e-8 relative) =#
  @testset "A latch on pre(), a when-assigned Real, initial assignments" begin
    #= active = nmax > pre(n) (a cluster on change(pre(n))): n rises at 0 and 0.25, then active is false: x[i] = i / 4 =#
    local (s::Vector{Float64}, a::Vector{Float64}, _) = _bothWays("ArrayPath.PreLatch", ["n", "x[1]", "x[2]"])
    @test a ≈ [2.0, 0.25, 0.5] atol = 1e-6
    @test maximum(abs.(s .- a)) < 1e-6
    #= T0 = 0.1, 0.3, 0.5, 0.7, 0.9 from the samples, d = 3 + 5, p1 = 2, p2 = 3 (the parameters not in the results):
       x[i] = p2 t + i (the integral of T0) =#
    local (s2::Vector{Float64}, a2::Vector{Float64}, _) = _bothWays("ArrayPath.ImplicitDiscrete", ["T0", "d", "x[1]", "x[2]"])
    local held::Float64 = 0.1 * 0.2 + 0.3 * 0.2 + 0.5 * 0.2 + 0.7 * 0.2 + 0.9 * 0.1
    @test a2 ≈ [0.9, 8.0, 3.0 + held, 3.0 + 2 * held] atol = 1e-6
    @test maximum(abs.(s2 .- a2)) < 1e-6
  end

  @testset "A block solve's equations at its solution after a fresh Jacobian" begin
    local AG::Module = OMBackend.CodeGeneration.ArrayODEGen
    local SA::Module = OMBackend.CodeGeneration.Symbolics.SparseArrays
    local c::Vector{Float64} = [1.0, 2.0]
    local S::SA.SparseMatrixCSC{Float64, Int} = SA.sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
    local s::AG.BlockSolver = AG.BlockSolver(S, AG.columnColors(S), copy(c))
    local residual!(r, y, du, a, u, p, t, z, mode, fire) = (a .= y; r .= y .- c; nothing)
    local (du::Vector{Float64}, a::Vector{Float64}) = (zeros(2), zeros(2))
    AG.blockSolve!(s, residual!, du, a, Float64[], nothing, 0.0, nothing, false, false)
    @test s.jacobians == 1
    @test a == c
  end

  @testset "Algebraic loops: per element by Newton, through the elements as a DAE" begin
    local tight::NamedTuple = (reltol = 1e-10, abstol = 1e-10)
    #= an equation nonlinear in its unknown; a nonlinear loop of two per element =#
    local (s::Vector{Float64}, a::Vector{Float64}, _) = _bothWays("ArrayPath.Cubic", ["x[1]", "y[2]", "x[3]"]; tight...)
    @test isapprox(a, s; atol = 1e-9)
    (s, a, _) = _bothWays("ArrayPath.LocalLoops", ["x[1]", "y[2]", "w[4]"]; tight...)
    @test isapprox(a, s; atol = 1e-9)
    #= one solve for the family of blocks: the code does not grow with n =#
    local batches::Int = OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches
    @test _arrayPathTaken(_arrayPathTranslate("ArrayPath.LocalLoops1000"; scalarized = false))
    @test OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches == batches
    #= a tridiagonal loop through all the elements: solved in the right-hand side (blockSolve!), the solver sees the
       states only =#
    local (s2::Vector{Float64}, a2::Vector{Float64}, sol::OMBackend.DifferentialEquations.SciMLBase.AbstractODESolution) = _bothWays("ArrayPath.CoupledLoop", ["x[1]", "v[2]", "x[20]", "v[10]"]; tight...)
    @test isapprox(a2, s2; atol = 1e-9)
    @test length(sol.u[1]) == 20
    #= with an event in the loop (its unknowns solved again in the event iteration), a when-equation, an assert and an
       initial equation (x[N] free) =#
    local (s3::Vector{Float64}, a3::Vector{Float64}, _) = _bothWays("ArrayPath.CoupledLoopEvents", ["x[1]", "v[2]", "x[20]", "v[10]", "crossings"]; tight...)
    @test isapprox(a3, s3; atol = 1e-8)
    @test a3[5] == 1.0
    #= nonlinear (the scalarized model's initialization fails here): its residuals at the end =#
    local name::String = _arrayPathTranslate("ArrayPath.CoupledNonlinear"; scalarized = false)
    @test _arrayPathTaken(name)
    sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0), tight...)
    @test sol.retcode == ReturnCode.Success
    local val::Function = (n::String) -> OMBackend.getVariableValues(sol, n)[end]
    local (x::Vector{Float64}, v::Vector{Float64}, w::Vector{Float64}) = ([val("x[$i]") for i in 1:20], [val("v[$i]") for i in 1:20], [val("w[$i]") for i in 1:20])
    local residuals::Vector{Float64} = [2v[1] - v[2] + 0.1v[1]^3 - x[1]; [-v[i - 1] + 3v[i] - v[i + 1] + 0.1v[i]^3 - x[i] for i in 2:19];
                       -v[19] + 2v[20] + 0.1v[20]^3 - x[20]]
    @test maximum(abs, residuals) < 1e-10
    @test w ≈ v .- x atol = 1e-12
  end

  @testset "Outside the scope: scalarized as before" begin
    #= a loop of a model without arrays (the scalar path solves it); a loop through a relation (events in a loop) =#
    for model in ("ArrayPath.AlgebraicLoop", "ArrayPath.LoopWithEvent")
      local name::String = _arrayPathTranslate(model; scalarized = false)
      @test !_arrayPathTaken(name)
      local sol::OMBackend.DifferentialEquations.SciMLBase.AbstractODESolution = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
      @test sol.retcode == ReturnCode.Success
    end
  end

  @testset "a when algorithm reading its target, reinit of elements, enumeration dimensions" begin
    #= (s, y) := step(s) at 0.1, 0.35, 0.6, 0.85: s from pre(s) at each run (the event iteration reran the body with
       the last pass's s and never settled) =#
    local (s, a, _) = _bothWays("ArrayPath.WhenReadsTarget", ["s", "y"])
    @test a == [4.0, 30.0] && s == a
    #= v[i] reset at 1.5 (rates i), v[3] set to -1 at 0.3 (a parameter's subscript, which the scalarized path's when
       code does not take): [1.2, 0.9, 0.2] at 1.2 =#
    local name = _arrayPathTranslate("ArrayPath.ReinitElements"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.2))
    @test sol.retcode == ReturnCode.Success
    @test isapprox(_values(sol, ["v[1]", "v[2]", "v[3]"]), [1.2, 0.9, 0.2]; atol = 1e-6)
    #= the record's fields and the scalar by their positions in the result, 1, 2, 3 (1, 3, 4 before: out of the
       tuple's range, an illegal instruction under @inbounds); the last tick at 0.5 =#
    local (sq, aq, _) = _bothWays("ArrayPath.RecordTupleWhen", ["r.a", "r.b", "q"]; tspan = (0.0, 0.55))
    @test isapprox(aq, [0.5, 1.0, 1.5]; atol = 1e-12) && isapprox(sq, aq; atol = 1e-12)
    #= a regular algorithm reading its target before assigning it: n := pre(n) + u at each change of u (the event
       iteration did not settle: n grew by u at every pass); OpenModelica 1.27.1: u = 4, n = 10, m = 20 at 1 =#
    local (sn, an, _) = _bothWays("ArrayPath.RegularAlgReads", ["u", "n", "m"])
    @test an == [4.0, 10.0, 20.0] && sn == an
    #= a when algorithm's targets start from pre(): edge() of one before its assignment is false (cnt = 0), a whole
       array read of element targets their pre() values (s = 4, 6); OpenModelica 1.27.1's values =#
    local (sg, ag, _) = _bothWays("ArrayPath.EdgeOfTarget", ["cnt"])
    @test ag == [0.0] && sg == ag
    local (ss, as, solS) = _bothWays("ArrayPath.SumOfTargets", ["s"])
    @test as == [6.0] && ss == as && OMBackend.getVariableValues(solS, "s")[findlast(<=(0.5), solS.t)] == 4.0
    #= a read through a variable subscript (x[i], i = 2) before the assignment of x[2]: x[2] from pre(x[2]) (did not
       settle: x[2] grew by 10 at every pass); OpenModelica 1.27.1: 40 at 1 =#
    local (sv, av, _) = _bothWays("ArrayPath.VarSubRead", ["x[1]", "x[2]"])
    @test av == [1.0, 40.0] && sv == av
    #= targets through a variable index of a record array (the scalarized path refuses them): an element assigned
       before its guarded assignment through s[k] keeps its value there; an element read before its assignment through
       s[k] starts from pre(); OpenModelica 1.27.1's values =#
    local at = (name::String, var::String, ts::Vector{Float64}) -> begin
      local sol = OMBackend.simulateModel(_arrayPathTranslate(name; scalarized = false); tspan = (0.0, 1.0))
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t), sol.t)] for t in ts]
    end
    @test at("ArrayPath.AssignedThenGuarded", "s[2].a", [0.2, 0.4, 0.7, 0.9]) == [5.0, 5.0, 7.0, 7.0]
    @test at("ArrayPath.ReadThenIndexed", "s[2].a", [0.2, 0.9]) == [1.0, 4.0]
    @test at("ArrayPath.ReadThenIndexed", "y", [0.9]) == [3.0]
    #= pre() of an element by an index assigned after (the index read now); a ring buffer read, written and advanced
       through its index; an element target by an index that changes between the passes: every element it may assign
       starts from pre() =#
    @test at("ArrayPath.PreIndexedRecord", "y", [0.2, 0.4, 0.7, 0.9]) == [10.0, 20.0, 10.0, 20.0]
    @test at("ArrayPath.RingRead", "y", [0.2, 0.4, 0.7, 0.9]) ≈ [0.0, 0.0, 0.1, 0.35]
    @test at("ArrayPath.StaleElement", "s[2].a", [0.2, 0.9]) == [0.0, 0.0] && at("ArrayPath.StaleElement", "s[1].a", [0.9]) ≈ [0.85]
    @test [at("ArrayPath.RecBig", "s[$i].a", [1.0])[1] for i in 1:3] ≈ [0.85, 0.35, 0.6]
    #= a tuple equation with a record target and fewer targets than outputs: q its position 3 (r.a, r.b, q) =#
    local fewer = _arrayPathTranslate("ArrayPath.TupleFewerTargets"; scalarized = true)
    @test _values(OMBackend.simulateModel(fewer; tspan = (0.0, 1.0)), ["q"]) ≈ [3.0]
    #= discrete variables equations compute, no events: evaluated at the start (they kept their starts: x = 0) =#
    local (sd, ad, _) = _bothWays("ArrayPath.DiscreteNoEvents", ["x"])
    @test isapprox(ad, [4.0]; atol = 1e-9) && isapprox(sd, ad; atol = 1e-9)
    #= a table over an enumeration's dimensions (its values per element: a KeyError before): T[o, o] = o, the 4th =#
    local (st, at, _) = _bothWays("ArrayPath.EnumTable", ["a[2]"])
    @test at == [4.0] && st == at
  end

  @testset "discrete equations at every event, time events, sample() in equations, tuples in whens, parameters" begin
    local arrayRun = (name::String; kw...) -> begin
      local n = _arrayPathTranslate(name; scalarized = false)
      @test _arrayPathTaken(n)
      local sol = OMBackend.simulateModel(n; tspan = (0.0, 1.0), kw...)
      @test sol.retcode == ReturnCode.Success
      return sol
    end
    local at = (sol, var::String, ts::Vector{Float64}) -> begin
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t), sol.t)] for t in ts]
    end
    #= a lifted discrete cluster runs at every pass of every event and of the start's iteration: pre(mode) the initial
       value 0 after the initialization, so started and mode 1 (the cluster fired only on its relation x < 0.5: mode
       0, x(1) = 0); OpenModelica 1.27.1's values =#
    local pai = arrayRun("ArrayPath.PreAfterInit")
    @test at(pai, "mode", [0.0, 0.5]) == [1.0, 1.0]
    @test OMBackend.getVariableValues(pai, "x")[end] ≈ 1.0 atol = 1e-6
    #= a time event (time < 0.5) at its exact time, the relations literal there: f <= 0 at f = 0 holds (with the
       events' hysteresis free stayed false: x(1) = 0); OpenModelica 1.27.1's values =#
    local eb = arrayRun("ArrayPath.ExactBoundaryAtEvent")
    @test at(eb, "free", [0.45, 0.6]) == [0.0, 1.0]
    @test OMBackend.getVariableValues(eb, "x")[end] ≈ 0.5 atol = 1e-6
    #= sample() in an equation, its start from an initial equation: the ticks at 0, 0.2, ... (refused before);
       OpenModelica 1.27.1's values =#
    @test at(arrayRun("ArrayPath.SampleEquation"), "y", [0.1, 0.3, 0.5, 0.9]) ≈ [0.0, 0.0, 0.2, 0.6] atol = 1e-9
    #= a tuple with an array output in a when-equation, each output at its position in the result: ten Fibonacci steps =#
    local tw = arrayRun("ArrayPath.TupleWhen")
    @test _values(tw, ["r", "s[1]", "s[2]"]) ≈ [5.5, 89.0, 55.0]
    @test at(tw, "r", [0.1, 0.32]) ≈ [0.1, 0.2]
    #= an array parameter bound to a call of literals, a String parameter as an assert's message =#
    @test _values(arrayRun("ArrayPath.ParameterOfArrayCall"), ["y"]) ≈ [3.0] atol = 1e-9
    @test _values(arrayRun("ArrayPath.StringParameterMessage"), ["x"]) ≈ [exp(-1)] rtol = 1e-4
    #= through a reference to another String parameter (kept by the arrays-kept passes: it was declined) =#
    @test _values(arrayRun("ArrayPath.StringParameterReference"), ["x"]) ≈ [exp(-1)] rtol = 1e-4
    #= sample() read other than as a when's rising edge: the samples' fall at the instant in an event iteration of its
       own (a single pass absorbed the rise of y = not u and pre(u): y stayed true, n = 0); OpenModelica 1.27.1's values =#
    local fe = arrayRun("ArrayPath.FallingEdgeOfSample")
    @test at(fe, "y", [0.2, 0.5]) == [0.0, 0.0] && at(fe, "n", [0.2, 0.5, 1.0]) == [1.0, 2.0, 4.0]
    @test at(arrayRun("ArrayPath.TickNot"), "n", [0.1, 0.3, 0.6, 1.0]) == [1.0, 2.0, 3.0, 5.0]
    #= the samples true for the first pass of the instant's event iteration only (OpenModelica's updateDiscreteSystem):
       an iteration with them true until it settled counted twice (change(u)), never settled (k), or read pre() of
       what the instant set (b); OpenModelica 1.27.1's values =#
    local sf = arrayRun("ArrayPath.SampleFirstPass")
    @test at(sf, "c", [0.2, 0.5, 1.0]) == [1.0, 2.0, 4.0] && at(sf, "k", [0.2, 0.5, 1.0]) == [1.0, 2.0, 4.0]
    @test at(sf, "a", [0.2, 1.0]) == [1.0, 4.0] && at(sf, "b", [0.2, 0.5, 1.0]) == [0.0, 1.0, 3.0]
    #= ticks of two samples equal up to round-off are one instant (two stops fired both samples twice) =#
    local ts = arrayRun("ArrayPath.TwoSampleTimes")
    @test at(ts, "a", [0.5, 0.7, 1.0]) == [3.0, 4.0, 6.0] && at(ts, "b", [0.5, 0.7, 1.0]) == [2.0, 3.0, 4.0]
    #= ticks apart by more than round-off are instants of their own (a 1e-9 window took 5e-10 apart as one: b read
       pre(a) before a's tick), and a tick near two stops fires once (it fired at both); OpenModelica 1.27.1's values =#
    local nt = arrayRun("ArrayPath.NearTicks")
    @test at(nt, "b", [0.05, 0.2, 0.5, 1.0]) == [0.0, 1.0, 2.0, 4.0] && at(nt, "c", [0.05, 0.2, 0.5, 1.0]) == [0.0, 1.0, 2.0, 4.0]
    local ct = arrayRun("ArrayPath.ChainTicks")
    @test all(v -> at(ct, v, [0.2, 1.0]) == [1.0, 4.0], ["a", "b", "c"])
    #= a tick 5e-10 after the start or the stop time is not at it (a 1e-9 snap fired them there: n(1) = 5, k(1) = 4) =#
    local sn = arrayRun("ArrayPath.TicksNearStartAndStop")
    @test at(sn, "n", [0.0, 0.1, 1.0]) == [0.0, 1.0, 4.0] && at(sn, "k", [1.0]) == [3.0]
    @test at(sn, "tn", [0.1]) ≈ [5e-10] atol = 1e-15
    #= a tick's round-off is relative to the magnitudes in it (st + k dt): -100 + 1003 * 0.1 and 0.3 one instant (two
       with a window relative to t: w = 5 at 0.35) =#
    @test at(arrayRun("ArrayPath.FarStartTicks"), "w", [0.05, 0.35, 0.85, 1.0]) == [1.0, 4.0, 9.0, 11.0]
  end

  @testset "terminate() in if-branches, outside when-equations, in if-equations of when-equations" begin
    #= each on the scalarized path and on the array path =#
    local bothRuns = (name::String) -> begin
      local n = _arrayPathTranslate(name; scalarized = true)
      local s = OMBackend.simulateModel(n; tspan = (0.0, 1.0))
      _arrayPathTranslate(name; scalarized = false)
      @test _arrayPathTaken(n)
      return [s, OMBackend.simulateModel(n; tspan = (0.0, 1.0))]
    end
    #= at the crossing of the branch's condition (OpenModelica 1.27.1: at its output step after it, 0.752); at the
       initialization where it holds at the start, as OpenModelica; in an else-branch's if-equation (OpenModelica
       ignores it) =#
    for (model, stop) in (("TerminateInBranch", 0.75), ("TerminateAtStart", 0.0), ("TerminateInElse", 0.6), ("TerminateNow", 0.0))
      for sol in bothRuns("ArrayPath." * model)
        @test sol.retcode == ReturnCode.Terminated
        @test sol.t[end] ≈ stop atol = 1e-6
      end
    end
    #= in an if-equation of a when-equation (OpenModelica fails to build it): the fourth tick, 0.35 =#
    for sol in bothRuns("ArrayPath.TerminateInWhenIf")
      @test sol.retcode == ReturnCode.Terminated
      @test sol.t[end] ≈ 0.35 atol = 1e-9
      @test OMBackend.getVariableValues(sol, "n")[end] == 4.0
    end
  end

  @testset "asserts: violations between steps by their relations' crossings, a relation touching its boundary" begin
    #= a violation for x in (0.7, 0.9) within one step of der(x) = 1, by a plain assert and by one in an elseif branch
       (`x <= lim`: at its boundary at the crossing, violated after it); OpenModelica 1.27.1 fails at 0.701 =#
    for model in ("AssertWindow", "BranchAssertWindow"), scalarized in (true, false)
      local n = _arrayPathTranslate("ArrayPath." * model; scalarized)
      scalarized || @test _arrayPathTaken(n)
      local err = try
        OMBackend.simulateModel(n; tspan = (0.0, 1.0))
        nothing
      catch e
        e
      end
      @test err isa OMBackend.CodeGeneration.ModelicaAssertionError
      @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && abs(err.time - 0.7) < 1e-6
    end
    #= x reaches 1 at 0.5 and falls again: x <= 1 holds =#
    for scalarized in (true, false)
      local n = _arrayPathTranslate("ArrayPath.AssertTouch"; scalarized)
      @test OMBackend.simulateModel(n; tspan = (0.0, 1.0)).retcode == ReturnCode.Success
    end
  end

  @testset "overrides of parameters read only by starts" begin
    local x0 = (name::String, var::String, parameters::Dict{String, Any}) ->
      OMBackend.getVariableValues(OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters), var)[1]
    #= sum(q) in a start (q was no parameter of the array code: the sum of an array literal did not translate); omc
       6, with -override=q[1]=10 15 =#
    local su = _arrayPathTranslate("ArrayPath.StartOfSum"; scalarized = false)
    @test _arrayPathTaken(su)
    @test x0(su, "x", Dict{String, Any}()) == 6.0
    @test x0(su, "x", Dict{String, Any}("q[1]" => 10.0)) == 15.0
    @test x0(su, "x", Dict{String, Any}("q" => [10.0, 2.0, 3.0])) == 15.0
    #= the starts of rolled components, c[i].x(start = c[i].s) with c.s = q (c[1].s was no variable of the array
       code); omc 4, 5, 6 =#
    local sc = _arrayPathTranslate("ArrayPath.StartOfComponents"; scalarized = false)
    @test _arrayPathTaken(sc)
    @test [x0(sc, "c[$(i)].x", Dict{String, Any}("q" => [4.0, 5.0, 6.0])) for i in 1:3] == [4.0, 5.0, 6.0]
    @test [x0(sc, "c[$(i)].x", Dict{String, Any}("c[2].s" => 9.0)) for i in 1:3] == [1.0, 9.0, 3.0]
  end

  @testset "self-reading algorithms, when-equation order, starts, asserts at events, terminate(), overrides" begin
    #= each on the scalarized path and on the array path (taken where `taken`) =#
    local run = (model::String, scalarized::Bool; taken::Bool = true, parameters = Dict{String, Any}()) -> begin
      local n = _arrayPathTranslate("ArrayPath." * model; scalarized)
      (scalarized || !taken) || @test _arrayPathTaken(n)
      return isempty(parameters) ? OMBackend.simulateModel(n; tspan = (0.0, 1.0)) :
                                   OMBackend.resimulateModel(n; tspan = (0.0, 1.0), parameters)
    end
    local failure = (model::String, scalarized::Bool) -> try
      run(model, scalarized)
      nothing
    catch e
      e
    end
    local at = (sol, var::String, ts::Vector{Float64}) -> begin
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t + 1e-12), sol.t)] for t in ts]
    end
    for scalarized in (true, false)
      #= a whole-array assignment that reads its own target: its values first, into temporaries (MLS 11.2.1; the
         elements in order read the new x[1]: {2, 2}); OpenModelica 1.27.1's values =#
      local sw = run("SwapInWhen", scalarized)
      @test at(sw, "x[1]", [0.05, 0.2, 0.4]) == [1.0, 2.0, 1.0]
      @test at(sw, "x[2]", [0.05, 0.2, 0.4]) == [2.0, 1.0, 2.0]
      local rw = run("RotationInWhen", scalarized)
      @test at(rw, "x[1]", [0.2, 0.4, 0.95]) ≈ [0.0, -1.0, 1.0] atol = 1e-12
      @test at(rw, "x[2]", [0.2, 0.4, 0.95]) ≈ [1.0, 0.0, 0.0] atol = 1e-12
      @test _values(run("SwapRegular", scalarized), ["y[1]", "y[2]"]) ≈ [2.0, 1.0] rtol = 1e-9
      #= an if lifted on its own counted x[k] and x[1] apart (x[1] = 5 at the tick); omc 3 =#
      @test at(run("Alias", scalarized; taken = false), "x[1]", [0.05, 0.2, 1.0]) == [3.0, 3.0, 3.0]
      #= the equations of a when-equation in an order that solves them: b reads the new a; terminate() after n's
         equation (it stopped at the fourth tick) =#
      local wo = run("WhenOrder", scalarized)
      @test at(wo, "a", [0.2, 1.0]) == [2.0, 10.0] && at(wo, "b", [0.2, 1.0]) == [3.0, 11.0]
      local wb = run("WhenIfBeforeAssign", scalarized)
      @test wb.retcode == ReturnCode.Terminated
      @test wb.t[end] ≈ 0.25 atol = 1e-9
      @test OMBackend.getVariableValues(wb, "n")[end] == 3.0
      #= starts of reductions the evaluation does not take: placeholders were 0 without overrides =#
      @test _values(run("StartSumNested", scalarized), ["x"]) ≈ [11.0] rtol = 1e-9
      @test _values(run("StartSumMatrix", scalarized), ["x"]) ≈ [11.0] rtol = 1e-9
      @test at(run("DiscreteStartSum", scalarized), "d", [0.2, 1.0]) == [10.0, 11.0]
      @test OMBackend.getVariableValues(run("GuessStartSum", scalarized), "x")[1] ≈ -10.0 rtol = 1e-6
      #= asserts decided with the discrete values within the step, not those of the event that ended it; omc: the
         first holds, the second fails at 0.6; an if-expression condition; a value set at a sample tick =#
      @test run("AssertAfterEvent", scalarized).retcode == ReturnCode.Success
      for (model, time) in (("AssertFailAtEvent", 0.6), ("AssertIfExp", 0.8), ("AssertSampleCount", 0.25), ("AlgIfAssert", 0.7))
        local err = failure(model, scalarized)
        @test err isa OMBackend.CodeGeneration.ModelicaAssertionError
        @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && abs(err.time - time) < 1e-6
      end
      #= terminate() under noEvent(), in if-equations of an elsewhen, of a for-equation, in algorithm sections =#
      for (model, stop) in (("TerminateNoEvent", 0.5), ("TermElsewhen", 0.6), ("ForBranch", 0.3), ("AlgTerminate", 0.5),
                            ("AlgTerminateReal", 0.5))
        local sol = run(model, scalarized)
        @test sol.retcode == ReturnCode.Terminated
        @test sol.t[end] ≈ stop atol = 1e-6
      end
      #= max of a literal of two arrays: over their four elements =#
      @test _values(run("MaxOfArrays", scalarized), ["y"]) ≈ [2.0] rtol = 1e-9
    end
    #= the array path: a der() coefficient of a parameter with fixed = false stays an expression (its parameter was a
       binding of a free parameter: declined); a discrete start of one after the initialization =#
    @test _values(run("CoefFreeParam", false), ["x"]) ≈ [1 / sin(0.5)] rtol = 1e-6
    @test at(run("DiscreteStartFree", false), "d", [0.2, 1.0]) == [2.0, 3.0]
    #= overrides: of a String parameter (OpenModelica's -override=s=xyz), of a value of another kind =#
    local sp = _arrayPathTranslate("ArrayPath.StringParam"; scalarized = false)
    @test _arrayPathTaken(sp)
    local se = try
      OMBackend.resimulateModel(sp; tspan = (0.0, 1.0), parameters = Dict{String, Any}("s" => "xyz"))
      nothing
    catch e
      e
    end
    @test se isa OMBackend.CodeGeneration.ModelicaAssertionError && occursin("xyz", se.message)
    @test_throws ArgumentError OMBackend.resimulateModel(sp; tspan = (0.0, 1.0), parameters = Dict{String, Any}("s" => 3.0))
    local ps = _arrayPathTranslate("ArrayPath.ParamSample"; scalarized = false)
    @test_throws ArgumentError OMBackend.resimulateModel(ps; tspan = (0.0, 1.0), parameters = Dict{String, Any}("T" => "abc"))
    #= two problems of one model, both built before either is solved: each its own samples (they shared them) =#
    local mod::Module = getfield(OMBackend, Symbol(ps))
    local modelFn = getfield(mod, Symbol(ps, "Model"))
    local (p1, c1) = Base.invokelatest(modelFn, (0.0, 1.0); parameters = Dict{String, Any}("T" => 0.25))
    local (p2, c2) = Base.invokelatest(modelFn, (0.0, 1.0); parameters = Dict{String, Any}("T" => 0.1))
    local DE = OMBackend.DifferentialEquations
    #= the generated module's functions in the latest world (it was evaluated in this test) =#
    @test OMBackend.getVariableValues(Base.invokelatest(DE.solve, p1, DE.Rodas5P(); callback = c1), "n")[end] == 5.0
    @test OMBackend.getVariableValues(Base.invokelatest(DE.solve, p2, DE.Rodas5P(); callback = c2), "n")[end] == 11.0
    #= a terminate() at the initialization: the problem of getMTKProblem ends at its start, on both paths =#
    for scalarized in (true, false)
      local n = _arrayPathTranslate("ArrayPath.TerminateAtStart"; scalarized)
      local (prob, cbs) = OMBackend.getMTKProblem(n)
      @test prob.tspan == (0.0, 0.0)
      @test Base.invokelatest(DE.solve, prob, DE.Rodas5P(); callback = cbs).t[end] == 0.0
    end
    #= a copied problem (an ensemble's trajectories, `remake` with deepcopy(p)): the module shared, the run's state its own =#
    local (p3, c3) = Base.invokelatest(modelFn, (0.0, 1.0); parameters = Dict{String, Any}("T" => 0.25))
    local q3 = deepcopy(p3.p)
    @test q3.model === p3.p.model && q3.state !== p3.p.state
    local ens = DE.EnsembleProblem(p3; prob_func = (pr, ctx) -> DE.remake(pr; p = deepcopy(pr.p)))
    local es = Base.invokelatest(DE.solve, ens, DE.Rodas5P(), DE.EnsembleSerial(); trajectories = 2, callback = c3)
    @test [OMBackend.getVariableValues(s, "n")[end] for s in es.u] == [5.0, 5.0]
  end

  @testset "event functions, self-reading assignments, when-assigned records and enumerations, run-time subscripts" begin
    #= review 20's models, each on the scalarized path and on the array path (taken where `taken`); OpenModelica
       1.27.1's values (or the analytic ones) =#
    local run = (model::String, scalarized::Bool; taken::Bool = true, stop::Float64 = 1.0) -> begin
      local n = _arrayPathTranslate("ArrayPath." * model; scalarized)
      (scalarized || !taken) || @test _arrayPathTaken(n)
      return OMBackend.simulateModel(n; tspan = (0.0, stop))
    end
    local failure = (model::String, scalarized::Bool) -> try
      run(model, scalarized)
      nothing
    catch e
      e
    end
    local at = (sol, var::String, ts::Vector{Float64}) -> begin
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t + 1e-12), sol.t)] for t in ts]
    end
    local T5 = [0.05, 0.2, 0.4, 0.7, 1.0]
    local TI = [0.04, 0.12, 0.57, 0.98]
    for scalarized in (true, false)
      #= a regular Real swap and a when swap in one section; elements that read each other (no cycle by element) =#
      local ms = run("MixedSection", scalarized)
      @test _values(ms, ["y[1]", "y[2]"]) ≈ [2.0, 1.0] rtol = 1e-9
      @test at(ms, "z[1]", [0.05, 0.2, 0.4, 1.0]) == [1.0, 2.0, 1.0, 1.0]
      @test at(run("ElemCycle", scalarized), "x[2]", [0.1, 0.2, 1.0]) == [3.0, 5.0, 21.0]
      #= if-statements whose branches assign what their conditions read: the conditions' values before the branches =#
      local ig = run("IfGuardReassign", scalarized)
      @test _values(ig, ["m", "n"]) == [2.0, 0.0]
      local is = run("IfGuardSwap", scalarized)
      @test at(is, "x[1]", T5) == [2.0, 1.0, 1.0, 1.0, 1.0] && at(is, "x[2]", T5) == [1.0, 2.0, 2.0, 2.0, 2.0]
      @test _values(run("IfGuardRegular", scalarized), ["y", "z"]) == [20.0, 1.0]
      local id = run("IfGuardDiscreteRegular", scalarized)
      @test at(id, "m", [0.3, 0.8]) == [3.0, 7.0] && at(id, "n", [0.3, 0.8]) == [0.0, 2.0]
      #= asserts and terminate() of algorithm sections at their point (after reassignments, in for loops), asserts after
         an event's reinit and discrete changes =#
      @test run("AlgAssertTopReassigned", scalarized).retcode == ReturnCode.Success
      for (model, time) in (("AlgAssertReassigned", 0.7), ("AlgAssertFor", 0.4), ("AssertBoolAlg", 0.6), ("AssertReinit", 0.5),
                            ("AssertReinitDiscrete", 0.5))
        local err = failure(model, scalarized)
        @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && abs(err.time - time) < 1e-5
      end
      local tf = run("AlgTermFor", scalarized)
      @test tf.retcode == ReturnCode.Terminated && abs(tf.t[end] - 0.3) < 1e-5
      #= the guard reads the discrete n (its value inlined, the operand landed on the threshold: never fired) =#
      local td = run("TermDiscreteGuard", scalarized)
      @test td.retcode == ReturnCode.Terminated && abs(td.t[end] - 0.5) < 1e-5
      #= ... also where the section assigns it again after the guard (its intermediate value: an operand that jumped onto
         the threshold of a relation; one built from relations and parameters only has no crossing, held between events) =#
      local tb = run("TermBeforeLastAssignment", scalarized)
      @test tb.retcode == ReturnCode.Terminated && abs(tb.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(tb, "n")[end] == 6.0
      #= ... against a Real literal or parameter, and of Real values (omc's hysteresis does not fire the last one, MLS
         11.1.2 says it terminates) =#
      for model in ("TermRealThreshold", "TermParamThreshold", "TermRealBranches")
        local tr = run(model, scalarized)
        @test tr.retcode == ReturnCode.Terminated && abs(tr.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(tr, "n")[end] == 6.0
      end
      #= an opaque function's Integer result (a continuous argument too) landing on its threshold, through arithmetic
         and a sign change: its crossing relation takes the exact zero (`_relationAt`). The scalarized route's relation
         callbacks miss these (as they miss `(if x > 0.5 then 5 else 0) >= 5`: older, open) =#
      for model in ("OpaqueIntegerLanding", "OpaqueSumParam", "OpaqueNegated")
        scalarized || @test at(run(model, scalarized), "y", [0.4, 0.75, 1.0]) == [0.0, 1.0, 1.0]
      end
      #= a relation whose operand lands exactly on its threshold at an event (an opaque function's value, a promoted
         Integer, abs() of it) is true there (`_relationAt`: the band kept it false forever); a relation of a noEvent'd
         condition's if-expression or of an Integer function of a continuous argument keeps its crossing (omc fires
         neither of those two). Operands held between events (built from relations, parameters and discretes) compare
         literally: a rounded sum `0.1 + 0.2 >= 0.3` and a strict relation on a plateau of exactly its threshold =#
      scalarized || @test at(run("RoundedWhen", scalarized), "hits", [0.4, 0.75, 1.0]) == [0.0, 1.0, 1.0]
      scalarized || @test at(run("StrictLanding", scalarized), "hits", [0.2, 0.3, 0.6, 0.8, 1.0]) == [0.0, 1.0, 1.0, 2.0, 2.0]
      #= a 64-bit Integer intermediate: the relation's difference of two native Integers must not wrap around (a false guard
         turned true before the event); the value is exact in Float64 =#
      scalarized || @test (local wi = run("WideIntegerIntermediate", scalarized); wi.retcode == ReturnCode.Terminated && abs(wi.t[end] - 0.5) < 1e-5)
      local ri = run("RoundedIntermediate", scalarized)
      @test ri.retcode == ReturnCode.Terminated && abs(ri.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(ri, "n")[end] == 2.0
      scalarized || @test OMBackend.getVariableValues(run("IfNoEventCondition", scalarized), "hits")[end] == 1.0
      scalarized || @test OMBackend.getVariableValues(run("IntegerFunctionNeedsCrossing", scalarized), "hits")[end] == 1.0
      local ta = run("TermOpaqueIntegerAbs", scalarized)
      @test ta.retcode == ReturnCode.Terminated && abs(ta.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(ta, "n")[end] == 6.0
      #= ... also promoted to Real by the frontend's cast in an intermediate-value guard (round 14's symbolic run) =#
      local to = run("TermOpaqueInteger", scalarized)
      @test to.retcode == ReturnCode.Terminated && abs(to.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(to, "n")[end] == 6.0
      #= an elseif's condition (a snapshot of the conditions before the branches) and a branch's partial function are
         evaluated only where their guards hold: sqrt of a negative d =#
      local ie = run("InactiveElseif", scalarized)
      @test at(ie, "d", [0.05, 0.95]) == [-1.0, -10.0] && at(ie, "n", [0.05, 0.95]) == [0.0, 9.0]
      @test at(run("GuardedSqrt", scalarized), "y", [0.05, 0.95]) == [0.0, 9.0]
      #= initial arms: run once at the initialization (a count, swaps); a swap in an initial algorithm of states =#
      @test at(run("InitArmCount", scalarized), "n", [0.0, 1.0]) == [1.0, 5.0]
      @test at(run("InitialArmSwap", scalarized), "x[1]", [0.0, 0.05, 0.2, 1.0]) == [2.0, 2.0, 1.0, 2.0]
      @test _values(run("InitWhenSwap", scalarized), ["x[1]", "x[2]"]) == [2.0, 1.0]
      @test _values(run("InitAlgSwap", scalarized; taken = false), ["x[1]", "x[2]"]) == [2.0, 1.0]
      #= event functions of continuous arguments: events at each crossing, in equations and in algorithm sections
         (an algorithm's Integer swap, one argument through an earlier assignment) =#
      @test at(run("EqInteger", scalarized), "n", TI) == [0.0, 1.0, 5.0, 9.0]
      @test at(run("EqIntegerReal", scalarized), "y", TI) == [0.0, 1.0, 5.0, 9.0]
      local a2 = run("AlgIntegerTwo", scalarized)
      @test at(a2, "n", TI) == [0.0, 1.0, 5.0, 9.0] && at(a2, "m", TI) == [1.0, 2.0, 6.0, 10.0]
      @test at(run("AlgIntegerReadsAssigned", scalarized), "n", TI) == [0.0, 2.0, 11.0, 19.0]
      local mi = run("MixedRealInt", scalarized)
      @test at(mi, "n[1]", TI) == [0.0, 2.0, 11.0, 19.0] && at(mi, "n[2]", TI) == [0.0, 1.0, 5.0, 9.0]
      #= a step variable of an argument through an if-equation (the scalarized route's step body read its own triggers,
         pinned after the crossing: 4); one of an inner step (a chain of whens: unbalanced) =#
      local si = run("StepThroughIfEq", scalarized; stop = 10.0)
      local (su, sy) = (OMBackend.getVariableValues(si, "u"), OMBackend.getVariableValues(si, "y"))
      @test all(((a, b),) -> b == (a > 0 ? floor(a + 0.5) : ceil(a - 0.5)) || abs(abs(a) - round(abs(a) - 0.5) - 0.5) < 1e-6,
                zip(su, sy))
      #= a whole period: the falling crossings (4: 2) and the negative half (6.3: -2, 8: -3), omc =#
      @test at(si, "y", [1.0, 2.5, 4.0, 6.3, 8.0]) == [2.0, 3.0, 2.0, -2.0, -3.0]
      local sc = run("StepChain", scalarized)
      @test at(sc, "n", [0.3, 0.5, 0.9]) == [1.0, 2.0, 4.0] && at(sc, "cnt", [0.3, 0.5, 0.9]) == [1.0, 2.0, 4.0]
      @test run("StepOfDer", scalarized).retcode == ReturnCode.Success
      #= ceil, div, mod, rem of an oscillating argument, both signs: each saved point off a jump against the formula =#
      local q = (x::Float64) -> 3.7x
      local saved = (sol, var::String, f::Function, g::Function) -> begin
        local xs::Vector{Float64} = OMBackend.getVariableValues(sol, "x")
        local ys::Vector{Float64} = OMBackend.getVariableValues(sol, var)
        return maximum(abs(y - f(x)) for (x, y) in zip(xs, ys) if abs(g(x) - round(g(x))) > 1e-5)
      end
      local ef = run("EventFns", scalarized)
      @test saved(ef, "c", x -> ceil(q(x)), q) < 1e-9
      @test saved(ef, "d", x -> trunc(q(x) / 1.3), x -> q(x) / 1.3) < 1e-9
      @test saved(ef, "m", x -> q(x) - floor(q(x) / 1.3) * 1.3, x -> q(x) / 1.3) < 1e-9
      @test saved(ef, "r", x -> q(x) - trunc(q(x) / 1.3) * 1.3, x -> q(x) / 1.3) < 1e-9
      local ae = run("AlgEventFns", scalarized)
      @test saved(ae, "i", x -> ceil(q(x)), q) < 1e-9
      @test saved(ae, "j", x -> trunc(q(x) / 1.3), x -> q(x) / 1.3) < 1e-9
      #= swaps by an index vector, a record by its constructor or a function, an enumeration array; starts of
         enumerations; a target by a run-time subscript =#
      @test at(run("IndexSwap", scalarized), "x[1]", T5) == [1.0, 2.0, 1.0, 2.0, 1.0]
      @test at(run("RecordSwapCtor", scalarized), "r.a", T5) == [1.0, 2.0, 1.0, 2.0, 1.0]
      @test at(run("RecordSwapFunc", scalarized; taken = false), "r.a", T5) == [1.0, 2.0, 1.0, 2.0, 1.0]
      @test at(run("EnumSwap", scalarized; taken = false), "i1", T5) == [1.0, 3.0, 1.0, 3.0, 1.0]
      local es = run("EnumStart", scalarized; taken = false)
      @test at(es, "e1", [0.1, 0.7]) == [1.0, 3.0] && at(es, "e2", [0.1, 0.7]) == [2.0, 1.0]
      local vs = run("VarSubscript", scalarized)
      @test at(vs, "x[1]", T5) == [1.0, 12.0, 12.0, 32.0, 32.0] && at(vs, "x[2]", T5) == [2.0, 2.0, 22.0, 22.0, 42.0]
    end
  end
end