#= MSL 4.1 Trapezoid's period counter, `when time >= (pre(count) + 1)*period + startTime`, on the array code's path
   (ArrayODEGen's affectCrossing!): its relation is located at its hysteresis band's edge, and at reltol 1e-6 the event's
   left limit (1e-10*(1 + |t|) back) lay exactly on its threshold, where the exact-zero rule made it true already; the
   when took its rising edge before the event and never fired (MSL OpAmps.Differentiator's source stayed in its first
   period). The left limit lies beyond the band now. The counter steps at t = 0.065 + k*0.1. Models/PeriodicCounters.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "A period counter on the array code's path" begin
  local file = joinpath(@__DIR__, "Models", "PeriodicCounters.mo")
  local model = "PeriodicCounters.TrapezoidDifferentiator"
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, OMFrontend.translateToSCode(OMFrontend.parseFile(file));
                                                      scalarize = false)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarized = false)
  local name = OMBackend.canonicalName(model)
  @test name in OMBackend.ARRAY_ODE_MODELS
  local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0), reltol = 1.0e-6)
  @test string(sol.retcode) == "Success"
  @test [sol(t; idxs = :signalSource_count) for t in (0.06, 0.07, 0.17, 0.97)] == [0.0, 1.0, 2.0, 10.0]
  @test sol.prob.p.d[1] ≈ 0.965 atol = 1.0e-8
  #= the capacitor follows the trapezoid: 5 at its top, -5 at its bottom =#
  @test [sol(t; idxs = :vc) for t in (0.05, 0.1, 0.15, 0.2)] ≈ [-5.0, 5.0, -5.0, 5.0] atol = 1.0e-2
end
