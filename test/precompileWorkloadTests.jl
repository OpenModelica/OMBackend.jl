#= The precompile workloads: OMBackend's OMJLIR models (`_precompileWarmupIR`, src/precompile.jl) and the translate
   workload of the OMFrontend extension (its precompile.jl, Precompile/Translate.mo and ArraysKept.mo). Each of their
   models is translated and simulated as OMJL does and takes the path it is there to compile (its step fails otherwise:
   a ✗ line and a warning in the precompile, a failure counted here), and they leave OMBackend's registries as they
   found them. =#
@testset "Precompile workload: translate and simulate" begin
  local CG = OMBackend.CodeGeneration
  local registered()::Vector{String} = String[collect(OMBackend.ARRAY_PRESERVE_DE_MODELS); collect(OMBackend.ARRAY_REDUCED_MODELS);
                                              collect(OMBackend.BACKEND_SCALARIZED_MODELS);
                                              collect(keys(OMBackend.ARRAY_PATH_REASONS)); collect(keys(OMBackend.COMPILED_MODELS_MTK));
                                              collect(keys(OMBackend.IMTKGen.BUILT)); string.(keys(CG.MODELICA_FUNCTION_IMPLS));
                                              string.(keys(CG.COMPILED_BODIES))]
  local before::Vector{String} = registered()
  local registries::Vector{Pair{Any, Any}} = OMBackend._registries()
  local ext::Module = Base.get_extension(OMBackend, :OMBackendOMFrontendExt)
  @test Base.CoreLogging.with_logger(OMBackend._precompileWarmupIR, Base.CoreLogging.NullLogger()) == 0
  @test Base.CoreLogging.with_logger(() -> ext._precompileWarmupTranslate(; simulate = true), Base.CoreLogging.NullLogger()) == 0
  @test isempty(setdiff(registered(), before))
  @test all(((registry, contents),) -> registry isa Base.RefValue ? registry[] === contents : length(registry) == length(contents),
            registries)
end

#= The precompile display (src/precompile.jl): a line per step, what a step prints and logs below a warning dropped,
   a failing step a ✗ line, its error a warning, and the workload goes on. =#
@testset "Precompile display: a line per step" begin
  local buffer::IOBuffer = IOBuffer()
  local io::IOContext = IOContext(buffer, :color => false)
  local noisyStep = () -> (println("printed"); println(stderr, "printed"); @info "logged"; nothing)
  local printed::String = mktemp() do path::String, file::IOStream
    redirect_stdio(stdout = file, stderr = file) do
      @test OMBackend._precompileStep(noisyStep, io, "A step")
    end
    flush(file)
    read(path, String)
  end
  @test isempty(printed)
  @test !Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
    OMBackend._precompileStep(() -> error("a failure"), io, "A failing step"; prefix = OMBackend._precompileBar(2, 4))
  end
  local lines::Vector{String} = split(String(take!(buffer)), '\n'; keepempty = false)
  @test length(lines) == 2
  @test startswith(lines[1], "  ✓ A step")
  @test startswith(lines[2], "  ✗ ━━━━━━━━──────── 2/4  A failing step") && endswith(lines[2], "failed")
end
