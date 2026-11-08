#= When conditions compiled (relationRefresh.jl eventFunction): a continuous when's crossing function is a
   WhenCondition, a when on one relation's crossing function and scale an eventFunction of its RelationWhenAffect,
   both RuntimeGeneratedFunctions of CodeGeneration that read the values by the system's value functions. As closures
   of the compile=min model module they ran interpreted: MSL SoftStarter's four 0.8-1.9 us a condition, 0.06 us
   compiled, its simulate 106.9 -> 64.3 s with the same steps. Models/CompiledConditions.mo: a when with an elsewhen
   (its first arm a continuous when, the elsewhen a when on one relation) whose condition calls a function, and a
   when on a relation with a parameter; s = time, so each when records the time its condition becomes true. A discrete
   when's condition (TieredCondition) runs interpreted until it has run about as long as compiling it takes, then
   compiled: the same values either way. =#
using Test
import OMBackend
import OMFrontend

const CC_SOLUTION = OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.AbstractODESolution

"""
The continuous callbacks of the problem `sol` was solved with, those a merged VectorContinuousCallback holds
(`_eraseContinuousCallbacks`) included.
"""
function _ccContinuousCallbacks(sol::CC_SOLUTION)::Vector{Any}
  local out = Any[]
  for c in sol.prob.kwargs[:callback].continuous_callbacks
    local merged = c.condition isa OMBackend.CodeGeneration.DiffEqBase.FunctionWrapper
    merged ? append!(out, c.condition.obj[].subs) : push!(out, c)
  end
  return out
end

"""
The conditions of the discrete whens of the problem `sol` was solved with: the event iteration's
(`withRelationRefresh`'s callback holds them) and those that are callbacks of their own.
"""
function _ccDiscreteConditions(sol::CC_SOLUTION)::Vector{Any}
  local out = Any[]
  for c in sol.prob.kwargs[:callback].discrete_callbacks
    c.affect! isa OMBackend.CodeGeneration.DiscreteWhenAffect && push!(out, c.affect!.condition)
    hasfield(typeof(c.condition), :e) && append!(out, [d.condition for d in getfield(c.condition, :e).discreteWhens])
  end
  return out
end

"The final value of the variable `v` of `sol`."
_ccFinal(sol::CC_SOLUTION, v::String)::Float64 = OMBackend.getVariableValues(sol, v)[end]

@testset "When conditions compiled" begin
  local CG = OMBackend.CodeGeneration
  local file::String = joinpath(@__DIR__, "Models", "CompiledConditions.mo")
  local (fm::OMFrontend.Frontend.FlatModel, cache::OMFrontend.Frontend.FunctionTreeImpl.Tree) =
    OMFrontend.instantiateSCodeToFM("CompiledConditions.Whens", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol::CC_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName("CompiledConditions.Whens"); tspan = (0.0, 1.0))
  @test string(sol.retcode) == "Success"
  local callbacks::Vector{Any} = _ccContinuousCallbacks(sol)
  local whens::Vector{Any} = filter(c -> c.condition isa CG.WhenCondition, callbacks)
  local relations::Vector{Any} = filter(c -> c.affect! isa CG.RelationWhenAffect, callbacks)
  @test length(whens) == 1 && length(relations) == 2
  @test all(c -> c.condition.eval isa CG.RuntimeGeneratedFunctions.RuntimeGeneratedFunction, whens)
  @test all(c -> c.affect!.eval isa CG.RuntimeGeneratedFunctions.RuntimeGeneratedFunction, relations)
  #= each condition's value functions resolved for the problem's system, a tuple =#
  @test all(c -> c.condition.fns isa Tuple && c.condition.sys === sol.prob.f.sys, whens)
  @test _ccFinal(sol, "sA") ≈ 0.3 atol = 1e-6
  @test _ccFinal(sol, "sB") ≈ 0.8 atol = 1e-6
  @test _ccFinal(sol, "sK") ≈ 0.6 atol = 1e-6
  @test _ccFinal(sol, "n") == 1
  @test _ccFinal(sol, "sE") ≈ 0.45 atol = 1e-6
  #= the discrete when's condition tiered; its compiled code (the tier passed) gives the same solution =#
  local tiered::Vector{Any} = filter(c -> c isa CG.TieredCondition, _ccDiscreteConditions(sol))
  @test !isempty(tiered) && all(c -> c.compiled isa CG.RuntimeGeneratedFunctions.RuntimeGeneratedFunction, tiered)
  foreach(c -> c.tier.compiled = true, tiered)
  local compiled::CC_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName("CompiledConditions.Whens"); tspan = (0.0, 1.0))
  @test all(c -> c.tier.compiled, filter(c -> c isa CG.TieredCondition, _ccDiscreteConditions(compiled)))
  @test compiled.t == sol.t && compiled.u == sol.u
end
