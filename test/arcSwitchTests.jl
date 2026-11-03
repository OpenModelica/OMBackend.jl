#= A failed algebraic re-solve inside a cluster's mixed system left the return code at InitialFailure, and every
   later re-solve of the instant reported failure (Models/ArcSwitchYD.mo, MSL 4.1.0's SwitchYDwithArc). When the Y
   openers open, the closers' cluster reads its delay's `time >= tSwitch + delayTime` with the tSwitch from before the
   instant: all closers close with the arcs burning, a re-solve fails, and the sweep stopped at that failure before
   the delay's when gave the new tSwitch. The closers stay open until 45 ms.
   The arcs go out where their current passes zero: phase 3 (5 A at the opening) at 25.1028 ms, then phases 1 and 2
   together at 28.5941 ms. The arc-out relation `abs(i) <= abs(v)*Goff` holds only for |i| below ~1e-4 A: caught past
   that window (OrdinaryDiffEqCore 4.19), the re-solve at the event in the arc's mode had no solution and ended the
   solve; stepped over, phase 3's arc burned on reversed and its winding carried -74 A at 28 ms. The reference values
   are the circuit's equations written out per mode (each arc going out a smooth crossing; Rodas5P and FBDF at reltol
   1e-8 and 1e-10 agree to the digits below); omc 1.27.1 agrees on 25.1028 ms and fails its NLS at 28.594 ms. =#
using Test
import OMBackend
import OMFrontend

@testset "Arcs and closers at a Y-D switch: a failed re-solve within the event iteration" begin
  local file::String = joinpath(@__DIR__, "Models", "ArcSwitchYD.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ArcSwitchYD.Switching",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ArcSwitchYD.Switching"); tspan = (0.0, 0.06))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= a variable's value at t, interpolated (the saved points are sparse) =#
  local value(name::String, t::Float64)::Float64 = sol(t; idxs = OMBackend.canonicalSymbol(name))
  #= open (Goff): milliamperes until the delay closes them, then the windings' currents =#
  @test maximum(k::Int -> abs(value("closer[$k].i", 0.03)), 1:3) < 0.01
  @test isapprox([value("closer[$k].i", 0.05) for k in 1:3], [-10.696, 213.24, -202.55]; rtol = 1.0e-4)
  @test isapprox([value("i[$k]", 0.028) for k in 1:3], [29.49484, -29.49234, -0.002509956]; rtol = 1.0e-4)
  #= the arcs going out: the first time after the opening each opener's `arcOff` holds =#
  local arcOffAt(s, k::Int)::Float64 = begin
    local q::Vector{Float64} = OMBackend.getVariableValues(s, "opener[$k].arcOff")
    s.t[findfirst(j::Int -> s.t[j] > 0.025 && q[j] > 0.5, eachindex(s.t))]
  end
  @test isapprox([arcOffAt(sol, k) for k in 1:3], [0.028594113, 0.028594051, 0.025102795]; atol = 1.0e-6)
  #= a second solve of the cached build takes the same path: the event iteration's memory of degenerate step restarts
     (`stall`) was kept from the first, and the arcs of phases 1 and 2 went out at 29.288 ms =#
  local again = OMBackend.simulateModel(OMBackend.canonicalName("ArcSwitchYD.Switching"); tspan = (0.0, 0.06))
  @test [arcOffAt(again, k) for k in 1:3] == [arcOffAt(sol, k) for k in 1:3]
end
