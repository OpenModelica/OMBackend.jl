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
  local (fm::OMFrontend.Frontend.FLAT_MODEL, cache::OMFrontend.Frontend.FunctionTreeImpl.Tree) =
    OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(OMFrontend.parseFile(EO_FILE)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  return OMBackend.simulate(OMBackend.canonicalName(name); tspan = (0.0, 2.0), reltol = 1.0e-6)
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

@testset "A when condition's change() of a variable named x (the MTK code)" begin
  #= The condition's lookups bind the model's variables by name; a variable `x` replaced the condition's argument `x`
     (the state vector), and the change() memory read the variable: a BoundsError on a Float64 (MSL
     InertialDelaySensitive, ArrayPath.DelayCell). The memory reads the state vector bound before the lookups. =#
  local program = OMFrontend.OpenModelicaParser.Modelica"""
    model ChangeOfX
      Real z(start = 0, fixed = true);
      Integer x = if time < 0.3 then 2 else 3;
      discrete Integer n(start = 0, fixed = true);
      discrete Real tChange(start = -1, fixed = true);
    equation
      der(z) = 1;
    algorithm
      when {initial(), change(x) and not initial()} then
        n := pre(n) + 1;
        tChange := time;
      end when;
    end ChangeOfX;"""
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ChangeOfX", OMFrontend.translateToSCode(program))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local name::String = OMBackend.canonicalName("ChangeOfX")
  try
    local sol = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1.0e-6)
    @test sol.retcode == ReturnCode.Success
    @test sol(1.0; idxs = :n) == 2.0
    @test sol(1.0; idxs = :tChange) ≈ 0.3 atol = 1.0e-6
  finally
    OMBackend.forgetModel(name)
  end
end
