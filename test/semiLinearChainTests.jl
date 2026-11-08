#= A chain of semiLinear equations through a component without storage (MLS 3.7.2.5, SimCode.simplifySemiLinear):
   its enthalpy is determined at zero flow only by the chain rule. With the enthalpy from a temperature, h = cp*T, the
   chain's node at the component is an expression after the alias elimination; not taken as a node, the chain broke
   in two, T kept its start value, and the pipe after it never cooled (MSL FluidHeatFlow OneMass: T_q drifted to 2e6
   and the solve became Unstable at the flow's start). Analytic values. Models/SemiLinearChains.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "semiLinear chains through a component without storage" begin
  local file = joinpath(@__DIR__, "Models", "SemiLinearChains.mo")
  local scode = OMFrontend.translateToSCode(OMFrontend.parseFile(file))
  #= the pipe's enthalpy after the flow starts at t = 0.2: hAmb + (hPipe0 - hAmb)*exp(-(t - 0.2)^2/(2C)) =#
  local hPipe = 293.15 + 20 * exp(-0.8^2 / 0.2)
  for m in ("ZeroFlowPump", "ZeroFlowPumpT")
    local (fm, cache) = OMFrontend.instantiateSCodeToFM("SemiLinearChains." * m, scode)
    OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("SemiLinearChains." * m); tspan = (0.0, 1.0),
                                        reltol = 1.0e-8, abstol = 1.0e-10)
    @test string(sol.retcode) == "Success"
    @test sol(0.1; idxs = :hPump) ≈ 293.15
    @test sol(1.0; idxs = :hPipe) ≈ hPipe rtol = 1.0e-6
    m == "ZeroFlowPumpT" && @test sol(0.1; idxs = :TPump) ≈ 293.15 / 4.2
  end
end
