#= Parametric initial equations that call a function of the model, solved at the build
   (solveParametricInitialEquations!). The function wrappers are bound in CodeGeneration, not in
   MTK_CodeGenerationUtil: evaluated there, the call was an UndefVarError and the free parameter
   kept its start value (MSL Fluid HeatExchangerSimulation's dp_nominal from
   `sum(WallFriction.pressureLoss_m_flow(...))`, HeatingSystem's valve Av from
   `valveCharacteristic(opening_nominal)`, FluxTubes ThreePhaseTransformerWithRectifier's
   filter coefficients from `CriticalDamping(order)`). =#
using Test
import OMBackend
import OMFrontend

const var"@Modelica_str" = OMFrontend.OpenModelicaParser.var"@Modelica_str"

const PARAMETRIC_INIT_FUNCTIONS = Modelica"""
package ParametricInitFunctions
  function twice
    input Real u[:];
    output Real y[size(u, 1)];
  algorithm
    y := 2 * u;
  end twice;
  function square
    input Real u;
    output Real y;
  algorithm
    y := u * u;
  end square;
  model FunctionOfFree
    parameter Real a[3] = {1, 2, 3};
    parameter Real p(start = 1, fixed = false) "sum(twice(a)) = 12: an array-valued call";
    parameter Real q(start = 1, fixed = false) "square(q) = 4: Newton on a call of the free one";
    Real x(start = 1, fixed = true);
    Real z(start = 1, fixed = true);
  initial equation
    p = sum(twice(a));
    square(q) = 4;
  equation
    der(x) = -p * x;
    der(z) = -q * z;
  end FunctionOfFree;
end ParametricInitFunctions;
"""

@testset "Parametric initial equations calling the model's functions" begin
  local scode = OMFrontend.translateToSCode(PARAMETRIC_INIT_FUNCTIONS)
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ParametricInitFunctions.FunctionOfFree", scode; scalarize = true)
  OMBackend.resetFallbacks!()
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sites::Vector{Symbol} = Symbol[first(r) for r in OMBackend.fallbackSummary()]
  @test !(:parametricInitLhs in sites)
  @test !(:parametricInitNewton in sites)
  local sol = OMBackend.simulate(OMBackend.canonicalName("ParametricInitFunctions.FunctionOfFree"); tspan = (0.0, 0.1),
                                 reltol = 1e-8, abstol = 1e-8)
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= p = 12, q = 2: x = exp(-12 t), z = exp(-2 t) =#
  @test OMBackend.variableValues(sol, "x")[end] ≈ exp(-1.2) rtol = 1e-6
  @test OMBackend.variableValues(sol, "z")[end] ≈ exp(-0.2) rtol = 1e-6
end
