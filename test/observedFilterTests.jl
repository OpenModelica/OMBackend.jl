#= With an observed filter, the observed closure (performStructuralSimplify) kept a variable ModelingToolkit made
   observed only for the filter's patterns and the system's reads, not for the alias and eliminated variables'
   observations injected after it, which a model with if-equation events adds after the structural simplification
   (Models/ObservedFilter.mo): an alias read an undefined name (MSL IMC_DCBraking's
   imc.is, SMPM_NoLoad's toSpacePhasor.y), nor for the events' conditions and affects (MSL FullRobot: its path
   planning relations read pathPlanning_path_sd_max_inv, and its build failed). =#
using Test
import OMBackend
import OMFrontend

@testset "An observed filter keeps what the injected observations read" begin
  local file::String = joinpath(@__DIR__, "Models", "ObservedFilter.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ObservedFilter.AliasOfObserved",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), observedFilter = ["^u\$"])
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ObservedFilter.AliasOfObserved"); tspan = (0.0, 1.0))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  @test OMBackend.getVariableValues(sol, "u")[end] ≈ 2 * exp(-1.0) + sin(exp(-1.0)) atol = 1e-5
end

@testset "An observed filter keeps what the events read" begin
  local file::String = joinpath(@__DIR__, "Models", "ObservedFilter.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ObservedFilter.RelationOfObserved",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), observedFilter = ["^y\$"])
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("ObservedFilter.RelationOfObserved"); tspan = (0.0, 1.0))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= the crossing of time = 0.3 + 0.1 exp(-time), at 0.36913 =#
  local tEvent::Float64 = sol.t[findfirst(iszero, diff(sol.t))]
  @test tEvent ≈ 0.3 + 0.1 * exp(-tEvent) atol = 1e-6
  @test sol(0.2; idxs = :y) == 0
  @test sol(0.5; idxs = :y) == 1
end

#= An equation affect (ModelingToolkit's AffectSystem: `m ~ Pre(m) + w`) reads the observed w: the observed filter keeps
   it (eventReadVariables). =#
@testset "The variables an equation affect reads" begin
  local MTK::Module = OMBackend.ModelingToolkit
  local t = MTK.t_nounits
  local (x, w) = OMBackend.ModelingToolkit.@variables x(t) w(t)
  local (m,) = OMBackend.ModelingToolkit.@discretes m(t) = 0.0
  local ev = MTK.SymbolicContinuousCallback([x ~ 0.5] => [m ~ MTK.Pre(m) + w]; discrete_parameters = [m])
  local sys = MTK.mtkcompile(MTK.System([MTK.D_nounits(x) ~ -x, w ~ 2x], t, [x, w], [m]; name = :s,
                                        continuous_events = [ev]))
  local CG::Module = OMBackend.CodeGeneration
  @test "w" in Set(CG.variableName(v) for v in CG.MTK_CodeGenerationUtil.eventReadVariables(sys))
end

