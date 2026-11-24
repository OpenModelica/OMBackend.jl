#= Author: John Tinnerholm
   OMBackend.plot of several solutions (OMBackendPlotsExt): all of them, on a new plot. Each later solution was drawn
   as `plot!(previous)`, without the solution (JETLS lowering/unused-local, 2026-11-24): only the first was plotted,
   onto the current plot. =#
using Test
import OMBackend
import OMFrontend
import Plots

const var"@Modelica_str" = OMFrontend.OpenModelicaParser.var"@Modelica_str"

@testset "plot: every solution of a vector, on a new plot" begin
  local program = Modelica"""
    model PlotDecay
      Real x(start = 1.0, fixed = true);
    equation
      der(x) = -x;
    end PlotDecay;
  """
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("PlotDecay", OMFrontend.translateToSCode(program))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sols = [OMBackend.simulate("PlotDecay"; tspan = (0.0, stop)) for stop in (0.5, 1.0, 1.5)]
  local perSolution::Int = length(Plots.plot(sols[1]).series_list)
  local current = Plots.plot([0.0, 1.0], [1.0, 0.0])
  @test length(OMBackend.plot(sols).series_list) == 3 * perSolution
  @test length(current.series_list) == 1
end
