#= Variable structure systems on the arrays-kept route (OM's translate: `translateReducedArrays`): the backend
   scalarizes the models the array form does not take, also their structural modes and a for-equation over a
   structural parameter (`1:N`, N the target of a recompilation). Analytic values. =#

"The model `model` of the Modelica text `program` with its arrays kept, translated as OM's translate does; its name."
function _vssTranslate(program, model::String)::String
  local scode = OMFrontend.translateToSCode(program)
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, scode; scalarize = false)
  @test OMBackend.translateReducedArrays(fm; functionList = OMFrontend.cacheToFunctionList(cache)) !== nothing
  return OMBackend.canonicalName(model)
end

"The value of `name` at the end of the solution `sol` of a mode."
_vssValue(sol, name::String)::Float64 = OMBackend.getVariableValues(sol, name)[end]

@testset "Variable structure systems, scalarized by the backend" begin
  @testset "Structural modes with arrays: the state carried over" begin
    local program = OMFrontend.OMParser.Modelica"""
      model VSSModes
        model A
          Real x[2](start = {1.0, 2.0});
        equation
          for i in 1:2 loop
            der(x[i]) = -x[i];
          end for;
        end A;
        model B
          Real x[2](start = {1.0, 2.0});
        equation
          der(x) = {1.0, 2.0};
        end B;
        structuralmode A a;
        structuralmode B b;
      equation
        initialStructuralState(a);
        structuralTransition(a, b, time >= 0.5);
      end VSSModes;
      """
    local name::String = _vssTranslate(program, "VSSModes")
    @test name in OMBackend.BACKEND_SCALARIZED_MODELS
    OMBackend.resetFallbacks!()
    local sols = OMBackend.simulateModel(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
    @test length(sols) == 2
    #= b's problem rebuilt from its system at the transition (MTK's initialization with the carried values): not the
       vector u0 fallback, whose problem starts at 0 =#
    @test !haskey(OMBackend.FALLBACKS, :vssTransferDAEInit)
    @test sols[2].prob.tspan == (0.5, 1.0)
    @test _vssValue(sols[1], "a_x[2]") ≈ 2 * exp(-0.5) atol = 1e-8
    #= x(0.5) carried into b, then x(1) = x(0.5) + {1, 2} * 0.5 =#
    @test _vssValue(sols[2], "b_x[1]") ≈ exp(-0.5) + 0.5 atol = 1e-8
    @test _vssValue(sols[2], "b_x[2]") ≈ 2 * exp(-0.5) + 1.0 atol = 1e-8
  end

  @testset "Recompilation of an array size: a for-equation over 1:N" begin
    local program = OMFrontend.OMParser.Modelica"""
      model VSSGrow
        parameter Integer N = 2;
        Real x[N](start = {i for i in 1:N});
      equation
        for i in 1:N loop
          der(x[i]) = -x[i];
        end for;
        when time > 0.5 then
          recompilation(N, 3);
        end when;
      end VSSGrow;
      """
    local name::String = _vssTranslate(program, "VSSGrow")
    @test name in OMBackend.BACKEND_SCALARIZED_MODELS
    #= solver options reach the recompiled mode's integrator (reinit! takes none of them) =#
    local sols = OMBackend.simulateModel(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
    @test length(sols) == 2
    @test length(sols[1].u[end]) == 2 && length(sols[2].u[end]) == 3
    #= x[1], x[2] carried over; x[3] from its start value 3 at the recompilation =#
    @test _vssValue(sols[2], "x[1]") ≈ exp(-1.0) atol = 1e-8
    @test _vssValue(sols[2], "x[2]") ≈ 2 * exp(-1.0) atol = 1e-8
    @test _vssValue(sols[2], "x[3]") ≈ 3 * exp(-0.5) atol = 1e-8
  end
end
