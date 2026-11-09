#= The event iteration's sweep in the order of what its parts read (relationRefresh.jl `_sweepOrder`), and a discrete
   cluster's relations evaluated where it runs, on the state its body reads (`_sweep!`). MSL StateGraph ControlledTanks
   lost both: its stop button's `when pre(reset) then on := false` reset `start.on` after the steps' cluster had read
   the old value and fired the resume transition, and the composite step's `integer()` of its active steps took a
   relation evaluated on a count of the iteration before (two active steps after a hand-over): the composite step never
   became inactive when suspended, and tank 1 was not filled again after t = 15. Models/EventOrder.mo, with
   OpenModelica's values. =#
using Test
import OMBackend
import OMFrontend

const EO_SOLUTION = OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.AbstractODESolution
const EO_FILE = joinpath(@__DIR__, "Models", "EventOrder.mo")

"Translate the model `name` of Models/EventOrder.mo (the MTK code's path) and simulate it over [0, 2]."
function _eoSimulate(name::String)::EO_SOLUTION
  local (fm::OMFrontend.Frontend.FlatModel, cache::OMFrontend.Frontend.FunctionTreeImpl.Tree) =
    OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(OMFrontend.parseFile(EO_FILE)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  return OMBackend.simulateModel(OMBackend.canonicalName(name); tspan = (0.0, 2.0), reltol = 1.0e-6)
end

@testset "Event iteration order" begin
  @testset "Radio buttons resetting each other" begin
    local sol::EO_SOLUTION = _eoSimulate("EventOrder.RadioButtons")
    @test string(sol.retcode) == "Success"
    @test sol(0.75; idxs = :startOn) == 1.0
    #= start reset as s2 becomes active: the transition out of s2 does not fire =#
    @test [sol(1.5; idxs = v) for v in (:startOn, :stopOn, :s2Active, :t4Fired)] == [0.0, 1.0, 1.0, 0.0]
    @test sol(2.0; idxs = :x) ≈ 1.0 atol = 1.0e-6
  end
  @testset "integer() of a count of steps" begin
    local sol::EO_SOLUTION = _eoSimulate("EventOrder.StepCount")
    @test string(sol.retcode) == "Success"
    @test [sol(t; idxs = :n) for t in (0.5, 1.5)] == [1.0, 1.0]
    @test [sol(1.5; idxs = v) for v in (:q, :aActive, :bActive)] == [1.0, 0.0, 1.0]
    @test sol(2.0; idxs = :y) ≈ 2.0 atol = 1.0e-6
  end
end
