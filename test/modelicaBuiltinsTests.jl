#= The Modelica builtins of generated code (src/CodeGeneration/modelicaBuiltins.jl). =#
using Test
import OMBackend

@testset "Modelica builtins" begin
  local String_ = OMBackend.CodeGeneration.AlgorithmicCodeGeneration.modelica_String
  @testset "String(x, minimumLength, leftJustified) of Integer and Boolean values" begin
    @test String_(3, 0, true) == "3"
    @test String_(3, 4, false) == "   3"
    @test String_(3, 4) == "3   "
    @test String_(true, 6, true) == "true  "
    @test String_(2.5) == "2.5"
  end
end
