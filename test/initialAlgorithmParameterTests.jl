#= A parameter without a binding that an initial algorithm assigns (Models/InitialAlgorithms.mo):
   one of the initialization's assigned parameters (ASSIGNED_PARAMETERS), as one an initial equation
   assigns. Left out, it kept its 0 against the algorithm's row: MSL Fluid RoomCO2's tracer index
   moved the volume's fixed T to 347 K, DCPowerSupply's vLim, iLim and cp stayed 0. =#
using Test
import OMBackend
import OMFrontend

@testset "Parameters an initial algorithm assigns" begin
  local file = joinpath(@__DIR__, "Models", "InitialAlgorithms.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("InitialAlgorithms.AssignedParameter",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("InitialAlgorithms.AssignedParameter"); tspan = (0.0, 1.0),
                                      reltol = 1e-8, abstol = 1e-8)
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= ind = 2, k = 4: x = exp(-4 t), y + y^3 = 4 at the start =#
  @test OMBackend.getVariableValues(sol, "x")[end] ≈ exp(-4) rtol = 1e-6
  local y0 = OMBackend.getVariableValues(sol, "y")[1]
  @test y0 + y0^3 ≈ 4 rtol = 1e-6
end
