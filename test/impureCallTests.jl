#= Impure calls on the array ODE path (Models/ImpureCalls.mo, as MSL's GlobalSeed and
   ImpureGenerator): the seed's initial equation `id = initializeImpureRandom(seed)` is assigned
   once, as an initial algorithm's, and a when body's call of parameters is made once per event.
   The event iteration's passes rerun a fired body: each pass drew a new value and the initial
   events of ImpureGenerator never settled. The generator is seeded here first, so a second
   draw shows as wrong values (an unseeded id is a ModelicaError, which aborts the process). =#
using Test
import OMBackend
import OMFrontend

const IMPURE_RX = OMBackend.CodeGeneration.AlgorithmicCodeGeneration.OMRuntimeExternalC

"""
The first `n` draws after seeding the generator as `ImpureCalls.initializeImpureRandom(7)` does.
"""
function _impureDraws(n::Int)::Vector{Float64}
  IMPURE_RX.ModelicaRandom_setInternalState_xorshift1024star(Cint[7 + 7919 * i for i in 1:33], 33, 715827883)
  return [IMPURE_RX.ModelicaRandom_impureRandom_xorshift1024star(715827883) for _ in 1:n]
end

@testset "Impure calls on the array path" begin
  local file = joinpath(@__DIR__, "Models", "ImpureCalls.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ImpureCalls.ImpureSample",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)); scalarize = false)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarized = false)
  local name = OMBackend.canonicalName("ImpureCalls.ImpureSample")
  @test name in OMBackend.ARRAY_ODE_MODELS
  local expected = _impureDraws(6)
  _impureDraws(0)
  local sol = OMBackend.simulateModel(name; tspan = (0.0, 0.55))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= id is assigned, no unknown of the initialization's Newton (INIT_PARAMS of the model's module,
     evaluated by the simulation) =#
  @test isempty(getfield(OMBackend, Symbol(name)).INIT_PARAMS)
  #= one draw at each sample, 0, 0.1, ..., 0.5 =#
  @test [sol(t; idxs = :r) for t in (0.05, 0.15, 0.25, 0.35, 0.45, 0.52)] == expected
end
