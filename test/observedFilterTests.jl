#= With an observed filter, the observed closure (performStructuralSimplify) kept a variable ModelingToolkit made
   observed only for the filter's patterns and the system's reads, not for the alias and eliminated variables'
   observations injected after it (Models/ObservedFilter.mo): an alias read an undefined name (MSL IMC_DCBraking's
   imc.is, SMPM_NoLoad's toSpacePhasor.y), nor for the events' conditions and affects (MSL FullRobot: its path
   planning relations read pathPlanning_path_sd_max_inv, and its build failed). =#
using Test
import OMBackend
import OMFrontend

@testset "An observed filter keeps what the injected observations read" begin
  local file = joinpath(@__DIR__, "Models", "ObservedFilter.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ObservedFilter.AliasOfObserved",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), observedFilter = ["^z\$"])
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ObservedFilter.AliasOfObserved"); tspan = (0.0, 1.0))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  @test OMBackend.getVariableValues(sol, "z")[end] ≈ 2 * exp(-1.0) + sin(exp(-1.0)) atol = 1e-5
end

@testset "An observed filter keeps what the events read" begin
  local file = joinpath(@__DIR__, "Models", "ObservedFilter.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ObservedFilter.RelationOfObserved",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), observedFilter = ["^y\$"])
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ObservedFilter.RelationOfObserved"); tspan = (0.0, 1.0))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= the crossing of time = 0.3 + 0.1 exp(-time), at 0.36913 =#
  local tEvent = sol.t[findfirst(iszero, diff(sol.t))]
  @test tEvent ≈ 0.3 + 0.1 * exp(-tEvent) atol = 1e-6
  @test sol(0.2; idxs = :y) == 0
  @test sol(0.5; idxs = :y) == 1
end
