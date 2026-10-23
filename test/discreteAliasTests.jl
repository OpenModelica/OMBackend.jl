#= A pure alias of two held discretes, both read in when conditions (MSL RadioButtonSource's reset inputs): the demotion
   makes one algebraic, and the relay elimination does not keep that one as the representative when no other equation
   reads it (eliminateIfEqRelays' `avoid`). It kept it, it then had no equation, and `pre(shut.reset[2]) = false` read
   no variable of the system (MSL ControlledTanks, Fluid and StateGraph). omc 1.27.1's values.
   Models/DiscreteAliases.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "Aliases of held discretes" begin
  local file = joinpath(@__DIR__, "Models", "DiscreteAliases.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("DiscreteAliases.Three", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("DiscreteAliases.Three"); tspan = (0.0, 4.0))
  @test string(sol.retcode) == "Success"
  @test [sol(t; idxs = v) for (v, t) in ((:start_on, 1.5), (:start_on, 2.5), (:stop_on, 2.5), (:stop_on, 3.5), (:shut_on, 3.5), (:start_on, 3.5))] ==
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
end
