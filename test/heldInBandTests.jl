#= A cluster's mixed system with no consistent mode while a relation is held by its hysteresis band: the search
   (CodeGeneration._search!) varies that relation too, and its literal value counts as consistent. A level rising to
   a port with Modelica.Fluid.Vessels' port logic: s's band edge is located first, regularFlow = level >= h stays
   false in its band, and neither inFlow mode is consistent; the event iteration failed at t = 0.1 (MSL
   TanksWithOverflow at its overflow port after the connection sets' order changed). Analytic values.
   Models/HeldInBand.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "A relation held by its band is searched" begin
  local file = joinpath(@__DIR__, "Models", "HeldInBand.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("HeldInBand.Port",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("HeldInBand.Port"); tspan = (0.0, 1.0))
  @test string(sol.retcode) == "Success"
  @test sol(0.05; idxs = :level) ≈ 5.95 atol = 1.0e-6
  @test sol(1.0; idxs = :level) ≈ 6.45 atol = 1.0e-6
  @test [sol(t; idxs = :regularFlow) for t in (0.05, 0.5)] == [0.0, 1.0]
  @test sol(0.5; idxs = :m) == -0.5
end
