#= sample() read by a discrete cluster: true at its ticks (_affectCall), which run the event iteration (_withTicks);
   it was refused in the cluster's body (MSL BooleanNetwork1's RSFlipFlop, reset by a SampleTrigger). An initial
   algorithm section of assignments reading a variable the initialization determines runs after the initialization
   (_lateSections); the early pass refused it (MSL MathBoolean.OnDelay). omc 1.27.1's values.
   Models/SampleClusters.mo. =#
using Test
import OMBackend
import OMFrontend

function _sampleClusterSolution(name::String, stop::Float64)
  local file = joinpath(@__DIR__, "Models", "SampleClusters.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  return OMBackend.simulateModel(OMBackend.canonicalName(name); tspan = (0.0, stop))
end

@testset "sample() in a discrete cluster; a late initial section" begin
  local flipFlop = _sampleClusterSolution("SampleClusters.FlipFlop", 1.2)
  @test string(flipFlop.retcode) == "Success"
  @test [flipFlop(t; idxs = :Q) for t in (0.1, 0.35, 0.6, 0.9, 1.1)] == [1.0, 0.0, 1.0, 0.0, 1.0]
  local late = _sampleClusterSolution("SampleClusters.LateSection", 1.0)
  @test string(late.retcode) == "Success"
  @test late(0.5; idxs = :delaySignal) == 1.0
  @test late(0.5; idxs = :t_next) == -1.0
  @test late(0.5; idxs = :y) == 1.0
end
