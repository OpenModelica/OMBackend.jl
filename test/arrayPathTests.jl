#= Array-preserving code generation (translate with scalarized = false): models from the
   frontend's no-scalarize mode keep their loops in the generated code; results equal the
   scalarizing path's. =#
const ARRAY_PATH_FILE = joinpath(@__DIR__, "models", "ArrayPath.mo")
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

function _bothWays(model, names; tspan = (0.0, 1.0))
  local name = _arrayPathTranslate(model; scalarized = true)
  local s = _values(OMBackend.simulateModel(name; tspan = tspan), names)
  _arrayPathTranslate(model; scalarized = false)
  @test _arrayPathTaken(name)
  local sol = OMBackend.simulateModel(name; tspan = tspan)
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

  @testset "Outside the scope: scalarized as before" begin
    local name = _arrayPathTranslate("ArrayPath.AlgebraicLoop"; scalarized = false)
    @test !_arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
  end
end
