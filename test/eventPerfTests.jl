#= The event iteration's compiled parts and the translate's repeated work (Models/EventPerf.mo):
   - A discrete cluster's body (an ideal diode's, MSL IdealDiode's equations) is a TieredBody: its closure runs
     interpreted until its tier, then the same code compiled, with the same solution (MSL HBridge_DC_Drive's
     bodies took 1 us a call interpreted, 2.8 million calls).
   - The event Newton factorizes the algebraic block in place, UMFPACK's symbolic analysis kept while the block's
     exact zeros stay: the factors equal a fresh `lu`'s (the diode's block changes its zeros along the solve).
   - The time-varying asserts' crossings as one compiled function: the same values as the asserts' closures
     (MSL BranchingDynamicPipes' 141 crossings took 75 us an evaluation interpreted).
   - An eager symbolic evaluation that threw a TypeError is not tried again with the same arguments (MSL
     BatchPlant's translate threw 4918 times); a Modelica function's body an earlier build compiled is called
     compiled at once, and its implementation's type does not depend on the body's. =#
using Test
import OMBackend
import OMFrontend
import SparseArrays
import LinearAlgebra

const EP_SOLUTION = OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.AbstractODESolution
const EP_FILE = joinpath(@__DIR__, "Models", "EventPerf.mo")

"Translate the model `name` of Models/EventPerf.mo."
function _epTranslate(name::String)::Nothing
  local (fm::OMFrontend.Frontend.FlatModel, cache::OMFrontend.Frontend.FunctionTreeImpl.Tree) =
    OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(OMFrontend.parseFile(EP_FILE)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  return nothing
end

"The event iteration of the problem `sol` was solved with (`withRelationRefresh`'s callback holds it)."
function _epEventIteration(sol::EP_SOLUTION)::OMBackend.CodeGeneration.EventIteration
  local c = only(filter(c -> hasfield(typeof(c.condition), :e), collect(sol.prob.kwargs[:callback].discrete_callbacks)))
  return getfield(c.condition, :e)
end

"The continuous callbacks of the problem `sol` was solved with, those a merged callback holds included."
function _epContinuousCallbacks(sol::EP_SOLUTION)::Vector{Any}
  local out = Any[]
  for c in sol.prob.kwargs[:callback].continuous_callbacks
    c.condition isa OMBackend.CodeGeneration.DiffEqBase.FunctionWrapper ? append!(out, c.condition.obj[].subs) : push!(out, c)
  end
  return out
end

@testset "Event iteration compiled parts" begin
  local CG = OMBackend.CodeGeneration
  @testset "Cluster body tiered" begin
    _epTranslate("EventPerf.HalfWave")
    local sol::EP_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName("EventPerf.HalfWave"); tspan = (0.0, 0.1))
    @test string(sol.retcode) == "Success"
    @test maximum(OMBackend.getVariableValues(sol, "vC")) ≈ 1.0 atol = 1e-3
    local e::CG.EventIteration = _epEventIteration(sol)
    @test length(e.clusters) == 1 && only(e.clusters).body isa CG.TieredBody
    foreach(c -> c.body.tier.compiled = true, e.clusters)
    local compiled::EP_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName("EventPerf.HalfWave"); tspan = (0.0, 0.1))
    @test all(c -> c.body.tier.compiled, _epEventIteration(compiled).clusters)
    @test compiled.t == sol.t && compiled.u == sol.u
    #= the event Newton's factors of the algebraic block against a fresh lu, at states along the solve =#
    local f = sol.prob.f
    local w::CG._EventNewtonWork = CG._EventNewtonWork(f, length(first(sol.u)))
    local zeroCounts::Set{Int} = Set{Int}()
    for k in 1:10:length(sol.t)
      local J = similar(f.jac_prototype, Float64)
      f.jac(J, sol.u[k], sol.prob.p, sol.t[k])
      copyto!(SparseArrays.nonzeros(w.jac), SparseArrays.nonzeros(J))
      local F = CG._factorBlock!(w)
      local G = LinearAlgebra.lu(J[w.rows, w.rows])
      @test F.L == G.L && F.U == G.U && F.p == G.p && F.q == G.q && F.Rs == G.Rs
      push!(zeroCounts, count(iszero, SparseArrays.nonzeros(J[w.rows, w.rows])))
    end
    @test length(zeroCounts) > 1
  end
  @testset "Assert crossings compiled" begin
    _epTranslate("EventPerf.Asserts")
    local sol::EP_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName("EventPerf.Asserts"); tspan = (0.0, 1.0))
    @test string(sol.retcode) == "Success"
    local zc = only(filter(c -> hasfield(typeof(c.condition), :compiled), _epContinuousCallbacks(sol)))
    local cond = zc.condition
    @test cond.compiled !== nothing && cond.compiled[2].count == zc.len == 5
    local SII = CG.ModelingToolkit.SymbolicIndexingInterface
    for k in eachindex(sol.t)
      local src = SII.ProblemState(; u = sol.u[k], p = sol.prob.p, t = sol.t[k])
      local (closures, compiled) = (zeros(zc.len), zeros(zc.len))
      Base.invokelatest(CG._assertCrossings!, closures, cond.crossingFns, cond.crossing, cond.checks, src, sol.t[k])
      Base.invokelatest(CG._compiledAssertCrossings!, cond.compiled[2].f, compiled, cond.compiled[1](src), sol.t[k])
      @test compiled == closures
    end
  end
  @testset "Eager evaluation failures; known bodies" begin
    #= a build: a cached one would not evaluate the equations again =#
    OMBackend.clearCaches!()
    OMBackend.resetFallbacks!()
    _epTranslate("EventPerf.Eager")
    @test only(f[2] for f in OMBackend.fallbackSummary() if f[1] === :eagerFunctionCall) == 1
    @test length(CG._eagerFailures(:EventPerf_clipped).args) == 1
    local sol::EP_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName("EventPerf.Eager"); tspan = (0.0, 2.0))
    @test OMBackend.getVariableValues(sol, "y1")[end] == 0.0 && OMBackend.getVariableValues(sol, "y2")[end] == 1.0
    OMBackend.clearCaches!()
    _epTranslate("EventPerf.Eager")
    local impl = CG.MODELICA_FUNCTION_IMPLS[:EventPerf_clipped]
    @test impl isa CG.CompiledModelicaFunction{1} && isconcretetype(typeof(impl))
    @test impl.tier.compiled
  end
end
