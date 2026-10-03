#= The merge of the continuous callbacks into one (DirectRHSGeneration.jl
   `_eraseContinuousCallbacks`) keeps their re-initialization. It created the merged
   callback with `initializealg = nothing` for components that asked NoInit (MTK's
   events): OrdinaryDiffEqCore then re-solved the DAE with the integrator's algorithm
   at every event, and MSL DifferenceAmplifier stopped with InitialFailure at its
   ramp's end (2026-10-02). A component's own: EventReinit, NoInit or nothing. =#
using Test
import OMBackend

@testset "Merged continuous callbacks keep their re-initialization" begin
  local SB = OMBackend.ModelingToolkit.SciMLBase
  local CG = OMBackend.CodeGeneration
  local cc(ia) = SB.ContinuousCallback((u, t, integ) -> t - 0.5, integ -> nothing; initializealg = ia)
  local merged(cbs...) = CG._eraseContinuousCallbacks(SB.CallbackSet(cbs...))
  local noInit = merged(cc(SB.NoInit()), cc(SB.NoInit()))
  @test length(noInit.continuous_callbacks) == 1
  @test only(noInit.continuous_callbacks).initializealg isa SB.NoInit
  #= A branch event's re-solve (EventReinit: the relations of states). =#
  local reinit = merged(cc(CG.EventReinit()), cc(CG.EventReinit()))
  @test length(reinit.continuous_callbacks) == 1
  @test only(reinit.continuous_callbacks).initializealg isa CG.EventReinit
  local default = merged(cc(nothing), cc(nothing))
  @test length(default.continuous_callbacks) == 1
  @test only(default.continuous_callbacks).initializealg === nothing
  #= Components that disagree are not merged: each keeps its own. =#
  local mixed = merged(cc(SB.NoInit()), cc(nothing))
  @test length(mixed.continuous_callbacks) == 2
  @test [typeof(c.initializealg) for c in mixed.continuous_callbacks] == [SB.NoInit, Nothing]
end
