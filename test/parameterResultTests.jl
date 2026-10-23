#= A model without continuous equations keeps its parameters, its result (SimCode.eliminateDeadParameters:
   MSL readRealParameterModel's reference compares J, phi_rel0 and w_rel0, and the solution had none), and an
   external C function's String output that the call passes (a const char**) comes back (ModelicaStrings_scanIdentifier
   and scanString in MSL Strings.scanToken: a "" buffer took no method). Models/ParameterResults.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "Parameters as a model's result" begin
  local file = joinpath(@__DIR__, "Models", "ParameterResults.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ParameterResults.OnlyParameters", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  local functions = OMFrontend.cacheToFunctionList(cache)
  local sc = OMBackend.mtkSimCode(OMBackend.omjlirModel(fm; functionList = functions))
  @test haskey(sc.stringToSimVarHT, "c") && haskey(sc.stringToSimVarHT, "d")
  OMBackend.translate(fm; functionList = functions)
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ParameterResults.OnlyParameters"); tspan = (0.0, 1.0))
  #= "myVar = 42" from 1: the identifier myVar, the index after it 6 =#
  @test sol(0.0; idxs = :c) == 6.0
  @test sol(1.0; idxs = :d) == 12.0
end
