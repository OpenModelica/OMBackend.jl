#= A clocked discretized continuous-time partition (a solver clock, MLS 16.8.2) discretized by the clocked lowering
   (Passes.discretizedPartition): ImplicitEuler x(ti) = x(ti-1) + T der(x)(ti), ExplicitEuler with der(x)(ti-1), the
   start at the first tick; omc 1.27.1's values. It was refused (MSL ClockedWithDiscretizedContinuousController).
   Models/SolverClocks.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "Solver clocks" begin
  local file = joinpath(@__DIR__, "Models", "SolverClocks.mo")
  local values = (("SolverClocks.ImplicitIntegrator", [0.0, 0.11, 0.23, 0.36]), ("SolverClocks.ExplicitDecay", [2.0, 1.9, 1.81, 1.729]))
  for (name, expected) in values
    local (fm, cache) = OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
    OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    local sol = OMBackend.simulateModel(OMBackend.canonicalName(name); tspan = (0.0, 0.35))
    @test string(sol.retcode) == "Success"
    #= between the ticks: the held value of the last one =#
    @test [sol(t; idxs = :y) for t in (0.05, 0.15, 0.25, 0.35)] ≈ expected atol = 1e-10
  end
end
