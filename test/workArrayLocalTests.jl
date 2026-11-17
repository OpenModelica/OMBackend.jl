#= A function's work array used only by literal subscripts (Models/WorkArrayLocals.mo, as MSL IF97's protected
   `Real o[20]`): one scalar local per element (_scalarizeLiteralIndexedLocals!), no array allocated and grown at each
   call. MSL BatchPlant_StandardWater's right-hand side allocated 147 KB a call in its IF97 calls. An array also used by
   a computed subscript stays an array. =#
using Test
import OMBackend
import OMFrontend

@testset "Work arrays by literal subscripts as scalars" begin
  local CG = OMBackend.CodeGeneration
  local file = joinpath(@__DIR__, "Models", "WorkArrayLocals.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("WorkArrayLocals.M", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("WorkArrayLocals.M"); tspan = (0.0, 1.0))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= f(x) = 2 (x^2 + 1) - x^2 = x^2 + 2 at x = 1, 2; g(x) = 3 x =#
  @test OMBackend.getVariableValues(sol, "y")[1] ≈ 3.0
  @test OMBackend.getVariableValues(sol, "y")[end] ≈ 6.0
  @test OMBackend.getVariableValues(sol, "z")[end] ≈ 3.0
  local code(name::Symbol)::String = string(CG.COMPILED_BODIES[name].code)
  @test !occursin("zeros(", code(:WorkArrayLocals_f)) && !occursin("ensureAlgArrayLength!", code(:WorkArrayLocals_f))
  @test occursin("zeros(", code(:WorkArrayLocals_g))
  local body = CG.MODELICA_FUNCTION_IMPLS[:WorkArrayLocals_f].f
  local alloc = () -> (body((2.0,)); @allocated body((2.0,)))
  @test Base.invokelatest(alloc) == 0
end
