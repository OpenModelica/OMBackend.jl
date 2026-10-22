#= Observed equations, each solved for its variable, need an order (Models/ObservedCycles.mo): a loop of
   explicit definitions or of output-only variables, eliminated together, was ModelingToolkit's "The
   equations have at least one cycle" (OM.jl InitialEquationTests IEQ54 and IEQ55, once smooth(0, ...) was
   noEvent). Their variables stay unknowns (SimCode.cyclicNames). =#
using Test
import OMBackend
import OMFrontend

const OBSERVED_CYCLES_FILE = joinpath(@__DIR__, "Models", "ObservedCycles.mo")

"The flat model of `ObservedCycles.<name>` and its functions."
function _ocFlat(name::String)::Tuple
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ObservedCycles." * name, OMFrontend.translateToSCode(OMFrontend.parseFile(OBSERVED_CYCLES_FILE)))
  return (fm, OMFrontend.cacheToFunctionList(cache))
end

@testset "Observed equations without a cycle" begin
  local SC = OMBackend.SimCode
  local deps = SC.OrderedDict{String, SC.OrderedSet{String}}("u" => SC.OrderedSet(["y", "x"]), "y" => SC.OrderedSet(["u"]),
                                                             "z" => SC.OrderedSet(["u"]), "w" => SC.OrderedSet(["w"]))
  @test SC.cyclicNames(deps) == SC.OrderedSet(["u", "y"])
  for (name, elimination, expected) in (("Folded", nothing, 1.0), ("OutputOnly", true, 0.0))
    local (fm, functions) = _ocFlat(name)
    local sc = OMBackend.mtkSimCode(OMBackend.omjlirModel(fm; functionList = functions); eliminateNonDynamic = elimination)
    @test !("u" in sc.eliminatedVariables) && !("y" in sc.eliminatedVariables)
    OMBackend.translate(fm; functionList = functions, eliminateNonDynamic = elimination)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("ObservedCycles." * name); tspan = (0.0, 1.0))
    @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
    @test OMBackend.getVariableValues(sol, "y")[end] ≈ expected atol = 1e-6
  end
end
