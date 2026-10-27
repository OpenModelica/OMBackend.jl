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
end
