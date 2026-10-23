#= A discrete cluster's relation whose crossing function is exactly zero takes its literal value
   (CodeGeneration._update!): the band kept `x < 0` true on x = 0 after the reinit, and the latch never turned off
   (MSL ThyristorBridge2mPulse_DC_Drive's fire_n read 0 > 0 after its timer's reset). A degenerate step after an
   event that comes again at one instant restarts at an instant's length (CodeGeneration._restartStepSize!; MSL
   ThyristorBridge2Pulse_DC_Drive crawled in 2e-16 s steps). omc 1.27.1's values. Models/ExactZeroRelations.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "Exact zero relations; the step after an event" begin
  local file = joinpath(@__DIR__, "Models", "ExactZeroRelations.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ExactZeroRelations.Latch",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ExactZeroRelations.Latch"); tspan = (0.0, 1.0))
  @test string(sol.retcode) == "Success"
  @test [sol(t; idxs = :off) for t in (0.2, 0.6, 0.8, 0.95)] == [1.0, 1.0, 0.0, 0.0]
  @test sol(0.8; idxs = :x) == 0.0
  @test OMBackend.CodeGeneration._instantLength(0.19) ≈ 1.19e-10
end

#= The exact-zero literal skips a relation the callback located at the same instant (c045c5e): MSL Rotational
   HeatLosses' locked brake had `w < 0` located at the band's edge, then w exactly 0 in the new mode's solve, and the
   literal flipped it back at every event, 2e-12 s apart (MaxIters). One strict relation of a cluster, its crossing
   function set by hand, on a stand-in integrator. =#
@testset "Exact zero after a located crossing" begin
  local CG = OMBackend.CodeGeneration
  local zc = Ref(0.0)
  local c = CG.DiscreteCluster(["off"], Any[], Any[], 0, 0, [true], [false], nothing, 0, false)
  c.crossings! = (out, u, p, t) -> (out[1] = zc[]; out[2] = 1.0; nothing)
  local at(t) = (u = Float64[], p = nothing, t = t, opts = (reltol = 1.0e-6,), sol = (prob = (tspan = (0.0, 1.0),),))
  #= located at the band's edge: flipped, as the callback says =#
  zc[] = -1.04e-10
  c.crossed[1] = true
  CG._update!(c, at(0.5))
  @test c.rel[1]
  #= the new mode's solve puts the crossing function at exactly 0, at the same instant: kept =#
  zc[] = 0.0
  @test !CG._update!(c, at(0.5))
  @test c.rel[1]
  #= exactly 0 at another instant, not located: the literal (`0 < 0` is false) =#
  @test CG._update!(c, at(0.75))
  @test !c.rel[1]
end
