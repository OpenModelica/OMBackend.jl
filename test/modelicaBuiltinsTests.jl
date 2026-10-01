#= The Modelica builtins of generated code (src/CodeGeneration/modelicaBuiltins.jl). =#
using Test
import OMBackend

@testset "Modelica builtins" begin
  local String_ = OMBackend.CodeGeneration.AlgorithmicCodeGeneration.modelica_String
  #= As OpenModelica formats them (its runtime's snprintf); the frontend fills in
     the default arguments. =#
  @testset "String(x, minimumLength, leftJustified) of Integer and Boolean values" begin
    @test String_(3, 0, true) == "3"
    @test String_(3, 4, false) == "   3"
    @test String_(3, 4, true) == "3   "
    @test String_(true, 6, true) == "true  "
    @test String_(false, 0, true) == "false"
  end
  @testset "String(r, significantDigits, minimumLength, leftJustified)" begin
    @test String_(2.5, 6, 0, true) == "2.5"
    @test String_(1 / 3, 6, 0, true) == "0.333333"
    @test String_(0.7, 3, 8, false) == "     0.7"
    @test String_(1e-5, 6, 8, true) == "1e-05   "
    @test String_(1e20, 6, 0, true) == "1e+20"
  end
  @testset "String(r, format)" begin
    @test String_(0.7, "8.3f") == "   0.700"
    @test String_(12345.678, "e") == "1.234568e+04"
    @test String_(0.5, "-6.2g") == "0.5   "
    @test_throws ArgumentError String_(3.0, "d")
    @test_throws ArgumentError String_(3.0, "s")
  end
end
