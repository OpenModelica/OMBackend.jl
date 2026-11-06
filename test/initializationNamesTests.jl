#= The initialization matches parameters and pins to the variables by their names (`variableName`), not their printing:
   an array element's symbol prints as `var"k[1]"`, SimCode names it `k[1]` (Models/InitializationNames.mo). =#
using Test
import OMBackend
import OMFrontend

@testset "The initialization's variables by name" begin
  local file::String = joinpath(@__DIR__, "Models", "InitializationNames.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("InitializationNames.FreeArrayParameter",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("InitializationNames.FreeArrayParameter");
                                      tspan = (0.0, 1.0), reltol = 1e-8, abstol = 1e-8)
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= k = {3, 4} from y = k .* x = {3, 8} at x = {1, 2}: x = {exp(-3 t), 2 exp(-4 t)} =#
  @test OMBackend.getVariableValues(sol, "x[1]")[end] ≈ exp(-3) rtol = 1e-6
  @test OMBackend.getVariableValues(sol, "x[2]")[end] ≈ 2 * exp(-4) rtol = 1e-6
end

@testset "The initialization's relations by their parameters' names" begin
  local file::String = joinpath(@__DIR__, "Models", "InitializationNames.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("InitializationNames.RelationAtStart",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("InitializationNames.RelationAtStart"); tspan = (0.0, 1.0))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  @test OMBackend.getVariableValues(sol, "y")[1] ≈ 1.0 atol = 1e-6
end

#= mergeSoftGuesses takes names: the if-equations' relay guesses (Strings), the trusted t0 values (Symbols). =#
@testset "Guesses merged by name" begin
  local MTK::Module = OMBackend.ModelingToolkit
  local CG::Module = OMBackend.CodeGeneration
  local t = MTK.t_nounits
  local (x, s) = OMBackend.ModelingToolkit.@variables x(t) s(t)
  local sys = MTK.mtkcompile(MTK.System([MTK.D_nounits(x) ~ -x, 0 ~ s^2 - x], t, [x, s], []; name = :g,
                                        guesses = [s => 0.0]))
  local guessOf(sys, name::String) = CG._constantNumber(only(v for (k, v) in MTK.guesses(sys) if CG.variableName(k) == name))
  @test guessOf(CG.mergeSoftGuesses(sys, Pair{Any, Any}["s" => 0.5]), "s") == 0.5
  @test guessOf(CG.mergeSoftGuesses(sys, Pair{Any, Any}[:s => 0.25]), "s") == 0.25
  #= not over a value that is no default 0, unless forced =#
  local seeded = CG.mergeSoftGuesses(sys, Pair{Any, Any}["s" => 0.5])
  @test guessOf(CG.mergeSoftGuesses(seeded, Pair{Any, Any}["s" => 2.0]), "s") == 0.5
  @test guessOf(CG.mergeSoftGuesses(seeded, Pair{Any, Any}[:s => 2.0]; force = true), "s") == 2.0
end

