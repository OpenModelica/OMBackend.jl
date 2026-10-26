#= The arrays-kept precompile workload (src/precompile.jl, src/Precompile/ArraysKept.mo): each of its models takes
   the path it is there to compile (the workload throws otherwise, which the precompile turns into a warning only),
   and it leaves the registries clear of them. =#
@testset "Precompile workload: arrays kept" begin
  @test (Base.CoreLogging.with_logger(OMBackend._precompileWarmupArraysKept, Base.CoreLogging.NullLogger()); true)
  local registered::Vector{String} = String[collect(OMBackend.ARRAY_ODE_MODELS); collect(OMBackend.ARRAY_REDUCED_MODELS);
                                            collect(keys(OMBackend.ARRAY_PATH_REASONS))]
  @test !any(n::String -> startswith(n, "OMBackendWorkload"), registered)
end
