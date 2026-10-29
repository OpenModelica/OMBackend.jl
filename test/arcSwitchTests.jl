#= A failed algebraic re-solve inside a cluster's mixed system left the return code at InitialFailure, and every
   later re-solve of the instant reported failure (Models/ArcSwitchYD.mo, MSL 4.1.0's SwitchYDwithArc). When the Y
   openers open, the closers' cluster reads its delay's `time >= tSwitch + delayTime` with the tSwitch from before the
   instant: all closers close with the arcs burning, a re-solve fails, and the sweep stopped at that failure before
   the delay's when gave the new tSwitch. The closers stay open until 45 ms.
   The arcs quench where their current passes zero: phase 3 (5 A at the opening) at 25.1028 ms, then phases 1 and 2
   together at 28.5941 ms. The quench relation `abs(i) <= abs(v)*Goff` holds only for |i| below ~1e-4 A: caught past
   that window (OrdinaryDiffEqCore 4.19), the re-solve at the event in the arc's mode had no solution and ended the
   solve; stepped over, phase 3's arc burned on reversed and its winding carried -74 A at 28 ms. The reference values
   are the circuit's equations written out per mode (each arc's quench a smooth crossing; Rodas5P and FBDF at reltol
   1e-8 and 1e-10 agree to the digits below); omc 1.27.1 agrees on 25.1028 ms and fails its NLS at 28.594 ms. =#
using Test
import OMBackend
import OMFrontend

@testset "Arcs and closers at a Y-D switch: a failed re-solve within the event iteration" begin
  local file = joinpath(@__DIR__, "Models", "ArcSwitchYD.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ArcSwitchYD.Switching",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ArcSwitchYD.Switching"); tspan = (0.0, 0.06))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  local closerCurrents(t::Float64)::Vector{Float64} =
    [OMBackend.getVariableValues(sol, "closer[$k].i")[argmin(abs.(sol.t .- t))] for k in 1:3]
  #= open (Goff): milliamperes until the delay closes them, then the windings' currents =#
  local (open, closed) = (closerCurrents(0.03), closerCurrents(0.05))
  @test maximum(abs, open) < 0.01
  @test maximum(abs, closed) > 10
  #= the quenches: events (a repeated time) at the reference's instants =#
  local events::Vector{Float64} = unique(Float64[sol.t[k] for k in 2:length(sol.t) if sol.t[k] == sol.t[k - 1]])
  @test any(t::Float64 -> abs(t - 0.025102795) < 1.0e-6, events)
  @test count(t::Float64 -> abs(t - 0.0285941) < 1.0e-6, events) == 2
  #= the winding currents at 28 ms, interpolated (the saved points are sparse) =#
  local named::Dict{String, Any} =
    Dict{String, Any}(string(x) => x for x in OMBackend.CodeGeneration.ModelingToolkit.unknowns(sol.prob.f.sys))
  local winding(k::Int)::Float64 = sol(0.028; idxs = named["var\"i[$k]\"(t)"])
  @test isapprox([winding(k) for k in 1:3], [29.49484, -29.49234, -0.002509956]; rtol = 1.0e-4)
  #= a second solve of the cached build takes the same path: the event iteration's memory of degenerate step restarts
     (`stall`) was kept from the first, and the arcs of phases 1 and 2 quenched at 29.288 ms =#
  local again = OMBackend.simulateModel(OMBackend.canonicalName("ArcSwitchYD.Switching"); tspan = (0.0, 0.06))
  @test unique(Float64[again.t[k] for k in 2:length(again.t) if again.t[k] == again.t[k - 1]]) == events
end
