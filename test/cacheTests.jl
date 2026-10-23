#= The model cache (OMBackend.Cache): the models kept oldest first, each with its size on the heap; over the limit
   (6 GiB, OMBACKEND_CACHE_LIMIT_GIB) the oldest are forgotten until the rest fits, the newest kept. The suite's own
   cached models are put back afterwards. =#
using Test
import OMBackend
import OMFrontend

@testset "Model cache" begin
  local C = OMBackend.Cache
  local savedModels = copy(C.MODELS)
  local savedSizes = copy(C.SIZES)
  try
    C.clear!()
    @test C.limitBytes() == 6 * 2^30
    withenv("OMBACKEND_CACHE_LIMIT_GIB" => "0.5") do
      @test C.limitBytes() == 2^29
    end
    #= oldest first; storing one again makes it the newest =#
    local forgotten = String[]
    local forget = name -> push!(forgotten, name)
    withenv("OMBACKEND_CACHE_LIMIT_GIB" => string(10 / 2^30)) do
      @test isempty(C.store!("a", 4; forget))
      @test isempty(C.store!("b", 4; forget))
      @test isempty(C.store!("a", 4; forget))
      @test C.MODELS == ["b", "a"]
      #= 11 bytes, over 10: b, the oldest, goes; a and c fit (at exactly the limit nothing goes) =#
      @test C.store!("c", 3; forget) == ["b"]
      @test C.MODELS == ["a", "c"] && C.totalBytes() == 7
      #= one larger than the limit stays alone =#
      @test C.store!("d", 20; forget) == ["a", "c"]
      @test C.MODELS == ["d"]
      @test forgotten == ["b", "a", "c"]
    end
    C.remove!("d")
    @test isempty(C.MODELS) && isempty(C.SIZES)
    #= a module is not counted (a build's closures reach the generated model module) =#
    @test C.sizeOf((1, OMBackend)) < 1000
    #= nor the compiled code a GlobalRef's binding leads to (a RuntimeGeneratedFunction's body: 1.7 GiB in 150 s) =#
    @test C.sizeOf(Expr(:call, GlobalRef(Base, :sin), 1.0)) < 10_000
    #= end to end: two models translated under a limit of one byte, the first forgotten =#
    local translateFile(name, file) = begin
      local (fm, cache) = OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
      OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    end
    withenv("OMBACKEND_CACHE_LIMIT_GIB" => string(1 / 2^30)) do
      translateFile("ExactZeroRelations.Latch", joinpath(@__DIR__, "Models", "ExactZeroRelations.mo"))
      @test C.MODELS == ["ExactZeroRelations_Latch"] && C.SIZES["ExactZeroRelations_Latch"] > 0
      translateFile("SampleClusters.FlipFlop", joinpath(@__DIR__, "Models", "SampleClusters.mo"))
      @test C.MODELS == ["SampleClusters_FlipFlop"]
      @test !haskey(OMBackend.COMPILED_MODELS_MTK, "ExactZeroRelations_Latch")
      @test haskey(OMBackend.COMPILED_MODELS_MTK, "SampleClusters_FlipFlop")
    end
    local sol = OMBackend.simulateModel("SampleClusters_FlipFlop"; tspan = (0.0, 1.2))
    @test string(sol.retcode) == "Success"
    #= a forgotten model is no translated array model either (a simulate by name looked for its code) =#
    push!(OMBackend.ARRAY_ODE_MODELS, "Forgotten_ArrayModel")
    OMBackend.forgetModel("Forgotten_ArrayModel")
    @test !("Forgotten_ArrayModel" in OMBackend.ARRAY_ODE_MODELS)
  finally
    C.clear!()
    append!(C.MODELS, savedModels)
    merge!(C.SIZES, savedSizes)
  end
end
