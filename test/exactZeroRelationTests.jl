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

#= Nor the rounding rule (`_onBoundary`: a false `<=` on its boundary to rounding holds): an open switch's arc-out
   relation `abs(i) <= abs(v)*Goff` (MSL OpenerWithArc, CloserWithArc) compares the two sides of `i = Goff*v`, which
   the integrator holds only to its error. Located at the band's edge it flipped false, the event's solve put it back
   on its boundary, and the rule flipped it true again at the same instant (MSL IMC_YDarc). =#
@testset "The rounding rule after a located crossing" begin
  local CG::Module = OMBackend.CodeGeneration
  local zc::Base.RefValue{Float64} = Ref(0.0)
  local c::OMBackend.CodeGeneration.DiscreteCluster = CG.DiscreteCluster(["arcOff"], Any[], Any[], 0, 0, [false], [false], nothing, 0, false)
  c.crossings! = (out, u, p, t) -> (out[1] = zc[]; out[2] = 1.0; nothing)
  local at(t) = (u = Float64[], p = nothing, t = t, opts = (reltol = 1.0e-6,), sol = (prob = (tspan = (0.0, 1.0),),))
  c.rel[1] = true
  #= located at the band's edge: flipped false =#
  zc[] = 1.04e-10
  c.crossed[1] = true
  CG._update!(c, at(0.5))
  @test !c.rel[1]
  #= back on its boundary to rounding at the same instant: kept =#
  zc[] = 1.0e-16
  @test !CG._update!(c, at(0.5))
  @test !c.rel[1]
  #= on its boundary at another instant, not located: it holds =#
  @test CG._update!(c, at(0.75))
  @test c.rel[1]
end

#= The array code's relation (`ArrayODEGen._relationAt`, mode = an event pass): a false relation becomes true at an exact
   zero where it holds (an operand that jumped onto its threshold: `(if x > 0.5 then 5 else 0) >= 5.0`), a strict one and a
   true one are not changed there (a true one is not turned off at an exact zero: chatter), a landing inside the
   hysteresis band but not exact stays as it was (an operand held between events compares literally by construction,
   `Rel` is made only for continuous ones). =#
@testset "Array relations at an exact zero" begin
  local relAt = OMBackend.CodeGeneration.ArrayODEGen._relationAt
  local H = 1.0e-6
  local run = (old::Bool, op, x::Real, y::Real; mode::Bool = true, H::Float64 = H) -> begin
    local p = (rel = [old],)
    local z = [0.0]
    local v = relAt(z, p, mode, 1, x, y, op, H)
    return (v, p.rel[1], z[1])
  end
  #= exact zero, positive H: false stays false for a strict relation, becomes true for a closed one; true stays true =#
  @test run(false, >=, 5.0, 5.0)[1]
  @test !run(false, >, 5.0, 5.0)[1]
  @test run(false, <=, 5.0, 5.0)[1]
  @test !run(false, <, 5.0, 5.0)[1]
  @test all(op -> run(true, op, 5.0, 5.0)[1], (>=, >, <=, <))
  #= its crossing function is shifted away from the new value (z uses the old one) =#
  local (v, r, z) = run(false, >=, 5.0, 5.0)
  @test v && r && z == H * 6.0
  #= clearly beyond the band: as before; clearly short of it too =#
  @test run(false, >=, 6.0, 5.0)[1] && !run(false, >=, 4.0, 5.0)[1]
  @test !run(true, >=, 4.0, 5.0)[1] && run(true, >=, 5.0 + 1.0e-9, 5.0)[1]
  #= inside the band, not exact: unchanged (false stays false, true stays true) =#
  @test !run(false, >=, 5.0 + 1.0e-9, 5.0)[1] && !run(false, >=, 5.0 - 1.0e-9, 5.0)[1]
  @test run(true, >=, 5.0 - 1.0e-9, 5.0)[1]
  #= not in an event pass: the stored value =#
  @test !run(false, >=, 5.0, 5.0; mode = false)[1] && run(true, >=, 4.0, 5.0; mode = false)[1]
  #= native Integer operands (an opaque function's result): the difference of 6e18 and -6e18 does not wrap around =#
  local big = 6_000_000_000_000_000_000
  @test !run(false, >=, -big, 4_000_000_000_000_000_000)[1] && run(false, >=, big, 4_000_000_000_000_000_000)[1]
  @test run(false, >=, 5, 5)[1] && !run(false, >, 5, 5)[1]
  #= the smallest Integer on both sides: the width is not negative (abs(typemin(Int)) stays negative) =#
  local lo = typemin(Int)
  @test run(false, <=, lo, lo)[1] && run(true, <=, lo, lo)[1] && !run(false, <, lo, lo)[1]
  @test run(false, <=, lo, lo)[3] == H * (1 + 2.0^63)
  #= at the initialization (H = 0): literal =#
  @test run(false, >=, 5.0, 5.0; H = 0.0)[1] && !run(false, >, 5.0, 5.0; H = 0.0)[1] && !run(true, >, 5.0, 5.0; H = 0.0)[1]
end
