#= The band's edge as crossed for if-equation relations that read time (CodeGeneration._bandEdgeRule). A relation's
   callback locates its crossing where the shifted function is zero, the band's edge; a twin (MSL Trapezoid without
   width: `time < T_start + T_rising`, `time < T_start + T_width`) is at its own edge there, to rounding, and its
   callback does not fire, so the band kept it: SignalPWM's triangle held at its peak, HBridge_DC_Drive missed two
   switchings. The values are that event's (t = 3.0555 + 4.06e-10, H = 1e-10, scale |t| + 1). Relations of the state
   keep the band (MSL FluidHeatFlow TwoMass's two equal pipes). =#
using Test
import OMBackend

@testset "Band edge of time relations" begin
  local CG = OMBackend.CodeGeneration
  local scale = 4.0555
  local eps = 1.0e-10 * scale
  #= true at its edge to rounding: crossed (the hysteresis rule kept it) =#
  @test CG._hysteresisRule(4.0554981595164463e-10, eps, true)
  @test !CG._bandEdgeRule(4.0554981595164463e-10, eps, scale, true)
  #= inside the band, clearly: kept =#
  @test CG._bandEdgeRule(1.0e-10, eps, scale, true)
  @test !CG._bandEdgeRule(-1.0e-10, eps, scale, false)
  #= false at its edge from the other side: crossed; beyond: as the rule =#
  @test CG._bandEdgeRule(-eps + 1.0e-17, eps, scale, false)
  @test CG._bandEdgeRule(-2eps, eps, scale, false)
  @test !CG._bandEdgeRule(2eps, eps, scale, true)
  #= no band (eps = 0) or zc on its own boundary: the band's rule, not a flip (the literal is `<` or `>=`) =#
  @test CG._bandEdgeRule(0.0, 0.0, 1.0, true) == CG._hysteresisRule(0.0, 0.0, true)
  @test CG._bandEdgeRule(1.0e-17, 0.0, 1.0, false) == CG._hysteresisRule(1.0e-17, 0.0, false)
end
