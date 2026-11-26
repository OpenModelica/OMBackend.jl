#= The public API (src/OMBackend.jl, docs/src/api.md): what holds for all of 1.x. A name added or removed here is an
   API change: a minor release adds one, only a major release removes one (but 1.3.0 removed printModel, modelToString
   and writeModelToFile, which exportModel replaces). =#

"The public names of OMBackend 1.x."
const PUBLIC_API_NAMES = Symbol[
  #= translate =#
  :translate, :BackendMode, :IMTK_MODE, :MTK_MODE, :EliminationOptions, :explainIndexReduction,
  :scalarizedFallbackReason, :canonicalName,
  #= simulate =#
  :simulate, :resimulate, :getMTKProblem, :defaultSolver, :daeFallbackSolver, :variableValues,
  :eventCallbacks, :isEventCallback, :isStepControl, :ModelicaAssertionError, :plot,
  #= tunable parameters (OMSurrogates) =#
  :withTunableParameters, :isTunable, :pristineParameters, :buildId,
  #= inspect and export models =#
  :availableModels, :exportModel, :ExportFormat, :JuliaSource, :Executable,
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
  #= CRLF on a Windows checkout =#
  local apiPage::String = replace(read(joinpath(@__DIR__, "..", "docs", "src", "api.md"), String), "\r\n" => "\n")
  @test all(n -> occursin("OMBackend.$n\n", apiPage), PUBLIC_API_NAMES)
end

#= The names of OMBackend 0.x that 1.0 renamed: they still work in 1.x, with a deprecation warning. =#
@testset "Deprecated names" begin
  @test all(n -> isdefined(OMBackend, n) && !Base.ispublic(OMBackend, n),
            [:simulateModel, :resimulateModel, :getVariableValues, :translateReducedArrays])
  @test OMBackend._deprecatedKeyword(:f, :old, nothing, :new, 1) == 1
  @test (@test_deprecated OMBackend._deprecatedKeyword(:f, :old, 2, :new, 1)) == 2
end
