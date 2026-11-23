#= A simulation from another start time than its build's (translate builds at 0): built at that start once and cached,
   so the next run from it reuses the build and `parameters` apply (OMBackend.simulate after translate, OMCompiler.simulate;
   OM.simulate builds at its startTime at translate). der(x) = -k*x from x = 1 at t0: x(t0 + 1) = exp(-k). =#
using Test
import OMBackend
import OMFrontend

@testset "Build at the simulation's start" begin
  local program = OMFrontend.OpenModelicaParser.Modelica"""
    model BuildStart
      parameter Real k = 1;
      Real x(start = 1, fixed = true);
    equation
      der(x) = -k * x;
    end BuildStart;"""
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("BuildStart", OMFrontend.translateToSCode(program))
  OMBackend.withTunableParameters(["k"]) do
    OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  end
  local BUILT = OMBackend.IMTKGen.BUILT
  local start(name::String)::Float64 = BUILT[name][6][1]
  local xEnd(sol)::Float64 = OMBackend.variableValues(sol, "x")[end]
  local tol::NamedTuple = (; reltol = 1e-10, abstol = 1e-12)
  @test start("BuildStart") == 0.0
  @test isapprox(xEnd(OMBackend.simulate("BuildStart"; tspan = (1.0, 2.0), tol...)), exp(-1.0); rtol = 1e-7)
  @test start("BuildStart") == 1.0
  #= the next run from 1 reuses that build; `parameters` from it (an error before) =#
  local built = BUILT["BuildStart"]
  @test isapprox(xEnd(OMBackend.simulate("BuildStart"; tspan = (1.0, 2.0), tol...)), exp(-1.0); rtol = 1e-7)
  @test BUILT["BuildStart"] === built
  @test isapprox(xEnd(OMBackend.simulate("BuildStart"; tspan = (1.0, 2.0), parameters = Dict("k" => 2.0), tol...)),
                 exp(-2.0); rtol = 1e-7)
  @test BUILT["BuildStart"] === built
  #= back to 0, and the compiled k after the run with k = 2 =#
  @test isapprox(xEnd(OMBackend.simulate("BuildStart"; tspan = (0.0, 1.0), tol...)), exp(-1.0); rtol = 1e-7)
  @test start("BuildStart") == 0.0
end
