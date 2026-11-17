#= The merge of the continuous callbacks into one (directRHSGeneration.jl
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

#= The array path's callbacks behind one wrapper type (directRHSGeneration.jl `_erasedCallbacks`): the solver's
   callback code compiled once, not per model (first simulate 0.3-0.8 s less for typical MSL models, 2026-11-16). =#
@testset "Erased callbacks: one type for any condition and affect, the same calls" begin
  local SB = OMBackend.ModelingToolkit.SciMLBase
  local CG = OMBackend.CodeGeneration
  local hits::Vector{Float64} = Float64[]
  local d1 = SB.DiscreteCallback((u, t, integ) -> t > 0.5, integ -> push!(hits, 1.0))
  local d2 = SB.DiscreteCallback((u, t, integ) -> u[1] < 0.0, integ -> push!(hits, 2.0))
  local (e1, e2) = (CG._erasedCallback(d1), CG._erasedCallback(d2))
  @test typeof(d1) != typeof(d2) && typeof(e1) == typeof(e2)
  @test e1.condition([1.0], 0.75, nothing) === true && e2.condition([1.0], 0.75, nothing) === false
  e1.affect!(nothing); e2.affect!(nothing)
  @test hits == [1.0, 2.0]
  #= the default hooks kept as they are (the solver tests for them by identity), others erased =#
  @test e1.initialize === SB.INITIALIZE_DEFAULT && e1.finalize === SB.FINALIZE_DEFAULT
  local d3 = SB.DiscreteCallback((u, t, integ) -> true, integ -> nothing; initialize = (c, u, t, integ) -> push!(hits, 3.0))
  local e3 = CG._erasedCallback(d3)
  e3.initialize(e3, [1.0], 0.0, nothing)
  @test e3.initialize isa CG._ErasedCall && hits[end] == 3.0
  local v = SB.VectorContinuousCallback((out, u, t, integ) -> (out .= t .- (0.25, 0.5)), (integ, i) -> push!(hits, i), 2;
                                        initializealg = SB.NoInit())
  local ev = CG._erasedCallback(v)
  local out::Vector{Float64} = zeros(2)
  ev.condition(out, [0.0], 1.0, nothing)
  ev.affect!(nothing, 2)
  @test out == [0.75, 0.5] && hits[end] == 2.0
  @test ev.len == 2 && ev.initializealg isa SB.NoInit && ev.rootfind == v.rootfind
end
