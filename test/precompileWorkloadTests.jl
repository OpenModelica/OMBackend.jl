#= The precompile workloads: OMBackend's OMJLIR models (`_precompileWarmupIR`, src/precompile.jl) and the translate
   workload of the OMFrontend extension (its precompile.jl, Precompile/Translate.mo and ArraysKept.mo). Each of their
   models is translated and simulated as OMJL does and takes the path it is there to compile (the workload throws
   otherwise, which the precompile turns into a warning only), and they leave OMBackend's registries as they found them. =#
@testset "Precompile workload: translate and simulate" begin
  local CG = OMBackend.CodeGeneration
  local registered()::Vector{String} = String[collect(OMBackend.ARRAY_ODE_MODELS); collect(OMBackend.ARRAY_REDUCED_MODELS);
                                              collect(keys(OMBackend.ARRAY_PATH_REASONS)); collect(keys(OMBackend.COMPILED_MODELS_MTK));
                                              collect(keys(OMBackend.IMTKGen.BUILT)); string.(keys(CG.MODELICA_FUNCTION_IMPLS));
                                              string.(keys(CG.COMPILED_BODIES))]
  local before::Vector{String} = registered()
  local registries::Vector{Pair{Any, Any}} = OMBackend._registries()
  local ext::Module = Base.get_extension(OMBackend, :OMBackendOMFrontendExt)
  @test (Base.CoreLogging.with_logger(OMBackend._precompileWarmupIR, Base.CoreLogging.NullLogger()); true)
  @test (Base.CoreLogging.with_logger(() -> ext._precompileWarmupTranslate(; simulate = true), Base.CoreLogging.NullLogger()); true)
  @test isempty(setdiff(registered(), before))
  @test all(((registry, contents),) -> registry isa Base.RefValue ? registry[] === contents : length(registry) == length(contents),
            registries)
end
