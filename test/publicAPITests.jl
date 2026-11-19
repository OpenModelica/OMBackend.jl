#= The public API (src/OMBackend.jl, docs/src/api.md): what holds for all of 1.x. A name added or removed here is an
   API change: a minor release adds one, only a major release removes one. =#

"The public names of OMBackend 1.x."
const PUBLIC_API_NAMES = Symbol[
  #= translate =#
  :translate, :translateReducedArrays, :generateTargetCode, :knownValues, :BackendMode, :IMTK_MODE, :MTK_MODE,
  :EliminationOptions, :explainIndexReduction, :scalarizedFallbackReason, :canonicalName,
  #= simulate =#
  :simulateModel, :resimulateModel, :getMTKProblem, :defaultSolver, :daeFallbackSolver, :getVariableValues,
  :eventCallbacks, :isEventCallback, :isStepControl, :ModelicaAssertionError, :plot,
  #= tunable parameters (OMSurrogates) =#
  :withTunableParameters, :isTunable, :pristineParameters,
  #= inspect and write models =#
  :availableModels, :printModel, :modelToString, :writeModelToFile, :dumpGeneratedCode,
  #= caches =#
  :clearCaches!, :forgetModel]

@testset "The public API" begin
  local public::Vector{Symbol} = filter(!=(:OMBackend), names(OMBackend))
  @test sort(public) == sort(PUBLIC_API_NAMES)
  @testset "$n is documented" for n in PUBLIC_API_NAMES
    @test Base.ispublic(OMBackend, n)
    @test Docs.hasdoc(OMBackend, n)
  end
  @test filter(n -> Base.isexported(OMBackend, n), names(OMBackend)) == [:OMBackend]
  local apiPage::String = read(joinpath(@__DIR__, "..", "docs", "src", "api.md"), String)
  @test all(n -> occursin("OMBackend.$n\n", apiPage), PUBLIC_API_NAMES)
end
