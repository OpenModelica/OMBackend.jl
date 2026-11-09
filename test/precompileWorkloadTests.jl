#= The translate precompile workload (src/precompile.jl, src/Precompile/Translate.mo and ArraysKept.mo): each of its
   models is translated and simulated as OM.jl does and takes the path it is there to compile (the workload throws
   otherwise, which the precompile turns into a warning only), and it leaves OMBackend's registries as it found them. =#
@testset "Precompile workload: translate and simulate" begin
  local CG = OMBackend.CodeGeneration
  local registered()::Vector{String} = String[collect(OMBackend.ARRAY_ODE_MODELS); collect(OMBackend.ARRAY_REDUCED_MODELS);
                                              collect(keys(OMBackend.ARRAY_PATH_REASONS)); collect(keys(OMBackend.COMPILED_MODELS_MTK));
                                              collect(keys(OMBackend.IMTKGen.BUILT)); string.(keys(CG.MODELICA_FUNCTION_IMPLS));
                                              string.(keys(CG.COMPILED_BODIES))]
  local before::Vector{String} = registered()
  local registries::Vector{Pair{Any, Any}} = OMBackend._registries()
  @test (Base.CoreLogging.with_logger(OMBackend._precompileWarmupTranslate, Base.CoreLogging.NullLogger()); true)
  @test isempty(setdiff(registered(), before))
  @test all(((registry, contents),) -> registry isa Base.RefValue ? registry[] === contents : length(registry) == length(contents),
            registries)
end
