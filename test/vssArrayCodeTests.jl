#= Variable structure systems on the arrays-kept route (OM's translate: `translate(fm; scalarize = false)`): array code,
   one array module per structural mode (OMJL's `structuralmode`, the modes a recompilation gives), the values at each
   change carried into the next mode by name. A small model per structural feature, analytic values. =#

"The model `model` of the Modelica text `program` with its arrays kept, translated as OM's translate does; its name."
function _vssTranslate(program, model::String)::String
  local scode = OMFrontend.translateToSCode(program)
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, scode; scalarize = false)
  @test OMBackend.translate(fm; scalarize = false, functionList = OMFrontend.cacheToFunctionList(cache)) !== nothing
  return OMBackend.canonicalName(model)
end

"The value of `name` at the end of the solution `sol` of a mode."
_vssValue(sol, name::String)::Float64 = OMBackend.variableValues(sol, name)[end]

@testset "Variable structure systems, array code" begin
  @testset "Structural modes with arrays: the state carried over" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
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
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    local sols = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
    @test length(sols) == 2
    #= b from the change, located by the crossing of time - 0.5 (just past it), to the end =#
    @test sols[2].prob.tspan[1] ≈ 0.5 atol = 1e-12
    @test sols[2].prob.tspan[2] == 1.0
    @test _vssValue(sols[1], "a_x[2]") ≈ 2 * exp(-0.5) atol = 1e-8
    #= x(0.5) carried into b, then x(1) = x(0.5) + {1, 2} * 0.5 =#
    @test _vssValue(sols[2], "b_x[1]") ≈ exp(-0.5) + 0.5 atol = 1e-8
    @test _vssValue(sols[2], "b_x[2]") ≈ 2 * exp(-0.5) + 1.0 atol = 1e-8
    #= the scalarizing route (ModelingToolkit): the frontend left the modes' variables arrays and their equations the
       elements' (UndefVarError der(a_x)[2]); the backend scalarizes the modes (`scalarizeModes`) =#
    local scode = OMFrontend.translateToSCode(program)
    local (fm, cache) = OMFrontend.instantiateSCodeToFM("VSSModes", scode)
    OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    @test !(name in OMBackend.ARRAY_PRESERVE_DE_MODELS)
    local mtk = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
    @test length(mtk) == 2
    @test _vssValue(mtk[2], "b_x[1]") ≈ exp(-0.5) + 0.5 atol = 1e-8
    @test _vssValue(mtk[2], "b_x[2]") ≈ 2 * exp(-0.5) + 1.0 atol = 1e-8
  end

  @testset "A transition on a state; a variable outside the modes; a mode's parameter" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model VSSCarry
        model A
          parameter Real r = 1.0;
          Real x(start = 1.0);
        equation
          der(x) = -r * x;
        end A;
        model B
          Real x(start = 0.0);
        equation
          der(x) = -2.0 * x;
        end B;
        structuralmode A a;
        structuralmode B b;
        Real y(start = 0.0);
      equation
        der(y) = 1.0;
        initialStructuralState(a);
        structuralTransition(a, b, a.x < 0.5);
      end VSSCarry;
      """
    local name::String = _vssTranslate(program, "VSSCarry")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    @test OMBackend.isTunable(name, ["a.r"])
    @test !OMBackend.isTunable(name, ["c"])
    @test_throws ArgumentError OMBackend.simulate(name; tspan = (0.0, 1.0), parameters = Dict("c" => 1.0))
    #= a.r = 2: x = exp(-2t) crosses 0.5 at ln(2)/2; b.x from 0.5 there, y through both modes =#
    local sols = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10,
                                    parameters = Dict("a.r" => 2.0))
    local t1::Float64 = log(2) / 2
    @test length(sols) == 2
    @test sols[2].prob.tspan[1] ≈ t1 atol = 1e-9
    @test _vssValue(sols[2], "b.x") ≈ 0.5 * exp(-2 * (1 - t1)) atol = 1e-8
    @test _vssValue(sols[2], "y") ≈ 1.0 atol = 1e-8
  end

  @testset "A transition on a discrete variable: at its sample instant, the tick once" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model VSSDiscrete
        model A
          Real x(start = 1.0);
        equation
          der(x) = -x;
        end A;
        model B
          Real x(start = 0.0);
        equation
          der(x) = 1.0;
        end B;
        structuralmode A a;
        structuralmode B b;
        Integer k(start = 0);
      equation
        when sample(0.0, 0.25) then
          k = pre(k) + 1;
        end when;
        initialStructuralState(a);
        structuralTransition(a, b, k >= 3);
      end VSSDiscrete;
      """
    local name::String = _vssTranslate(program, "VSSDiscrete")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    #= ticks at 0, 0.25, 0.5: k = 3 at 0.5, the switch there; b ticks at 0.75 only =#
    local sols = OMBackend.simulate(name; tspan = (0.0, 0.9), reltol = 1e-10, abstol = 1e-10)
    @test length(sols) == 2
    @test sols[2].prob.tspan[1] ≈ 0.5 atol = 1e-12
    @test _vssValue(sols[1], "k") == 3.0
    @test _vssValue(sols[2], "k") == 4.0
    @test _vssValue(sols[2], "b.x") ≈ exp(-0.5) + 0.4 atol = 1e-8
  end

  @testset "Recompilation of an array size: a for-equation over 1:N" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
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
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    @test !OMBackend.isTunable(name, ["N"])
    for run in 1:2
      local sols = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
      @test length(sols) == 2
      @test length(sols[1].u[end]) == 2 && length(sols[2].u[end]) == 3
      #= x[1], x[2] carried over; x[3] from its start value 3 at the recompilation =#
      @test _vssValue(sols[2], "x[1]") ≈ exp(-1.0) atol = 1e-8
      @test _vssValue(sols[2], "x[2]") ≈ 2 * exp(-1.0) atol = 1e-8
      @test _vssValue(sols[2], "x[3]") ≈ 3 * exp(-0.5) atol = 1e-8
    end
    #= the second run took the mode the first one compiled =#
    @test length(getfield(OMBackend, Symbol(name)).RECOMPILED) == 1
  end

  @testset "Recompilations of a Boolean and a Real parameter, one after the other" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model VSSRetune
        parameter Boolean fast = false;
        parameter Real k = 1.0;
        Real x(start = 1.0);
      equation
        der(x) = if fast then -2.0 * k * x else -k * x;
        when time > 0.25 then
          recompilation(fast, true);
        end when;
        when time > 0.5 then
          recompilation(k, 0.5);
        end when;
      end VSSRetune;
      """
    local name::String = _vssTranslate(program, "VSSRetune")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    #= rate 1 to 0.25, 2 to 0.5 (fast), 2 * 0.5 after (fast kept, k = 0.5) =#
    local sols = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
    @test length(sols) == 3
    @test _vssValue(sols[1], "x") ≈ exp(-0.25) atol = 1e-8
    @test _vssValue(sols[2], "x") ≈ exp(-0.75) atol = 1e-8
    @test _vssValue(sols[3], "x") ≈ exp(-1.25) atol = 1e-8
  end

  @testset "A terminate() at a sample instant ends the run there" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model TerminateAtSample
        Real x(start = 1.0);
        Integer k(start = 0);
      equation
        der(x) = -x;
        when sample(0.0, 0.25) then
          k = pre(k) + 1;
        end when;
        when k >= 3 then
          terminate("three ticks");
        end when;
      end TerminateAtSample;
      """
    local name::String = _vssTranslate(program, "TerminateAtSample")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    local sol = OMBackend.simulate(name; tspan = (0.0, 0.9), reltol = 1e-10, abstol = 1e-10)
    @test sol.t[end] ≈ 0.5 atol = 1e-12
    @test _vssValue(sol, "k") == 3.0
  end

  #= An event at the start instant in the solve's initialization (a sample's first tick, `_sampleStart!`) that ends
     the run: terminate! left the integrator no stop, and its first step size read one (a BoundsError). =#
  @testset "A terminate() at the first sample instant ends the run at the start" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model TerminateAtFirstSample
        Real x(start = 1.0);
        Integer k(start = 0);
      equation
        der(x) = -x;
        when sample(0.0, 0.25) then
          k = pre(k) + 1;
        end when;
        when k >= 1 then
          terminate("the first tick");
        end when;
      end TerminateAtFirstSample;
      """
    local name::String = _vssTranslate(program, "TerminateAtFirstSample")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    local sol = OMBackend.simulate(name; tspan = (0.0, 0.9), reltol = 1e-10, abstol = 1e-10)
    @test sol.retcode == ReturnCode.Terminated
    @test sol.t[end] == 0.0
    @test _vssValue(sol, "k") == 1.0 && _vssValue(sol, "x") == 1.0
  end

  @testset "A transition at the first sample instant: the next mode from the start" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model VSSFirstTick
        model A
          Real x(start = 1.0);
        equation
          der(x) = -x;
        end A;
        model B
          Real x(start = 0.0);
        equation
          der(x) = 1.0;
        end B;
        structuralmode A a;
        structuralmode B b;
        Integer k(start = 0);
      equation
        when sample(0.0, 0.25) then
          k = pre(k) + 1;
        end when;
        initialStructuralState(a);
        structuralTransition(a, b, k >= 1);
      end VSSFirstTick;
      """
    local name::String = _vssTranslate(program, "VSSFirstTick")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    local sols = OMBackend.simulate(name; tspan = (0.0, 0.9), reltol = 1e-10, abstol = 1e-10)
    @test length(sols) == 2
    @test sols[1].t[end] == 0.0 && _vssValue(sols[1], "k") == 1.0
    #= b.x from a.x = 1 at 0; the ticks at 0.25, 0.5, 0.75 in b =#
    @test _vssValue(sols[2], "b.x") ≈ 1.9 atol = 1e-8
    @test _vssValue(sols[2], "k") == 4.0
  end

  #= A recompilation in a model of structural modes recompiles all its modes (a family, `simulateModes`): the run goes
     on in the mode of the same name, a later transition to a mode of the recompiled family. =#
  @testset "Structural modes and a recompilation in one model" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model VSSModesGrow
        parameter Integer N = 2;
        model A
          parameter Integer n;
          Real x[n](each start = 1.0);
        equation
          der(x) = -x;
        end A;
        model B
          parameter Integer n;
          Real x[n](each start = 0.0);
        equation
          der(x) = ones(n);
        end B;
        structuralmode A a(n = N);
        structuralmode B b(n = N);
      equation
        initialStructuralState(a);
        structuralTransition(a, b, time >= 0.5);
        when time > 0.25 then
          recompilation(N, 3);
        end when;
      end VSSModesGrow;
      """
    local name::String = _vssTranslate(program, "VSSModesGrow")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    local sols = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
    #= a with N = 2 to 0.25, a with N = 3 (a.x[3] from its start 1) to 0.5, b with N = 3 =#
    @test length(sols) == 3
    @test [length(s.u[end]) for s in sols] == [2, 3, 3]
    @test sols[3].u[end] ≈ [exp(-0.5) + 0.5, exp(-0.5) + 0.5, exp(-0.25) + 0.5] atol = 1e-8
    #= one family recompiled, its b the mode the transition took =#
    local m::Module = getfield(OMBackend, Symbol(name))
    @test length(m.RECOMPILED) == 1
    @test sols[3].prob.p.model === only(values(m.RECOMPILED))["b"]
  end

  #= A structural change the initialization reaches (`when initial()`): kept in the start state (`ArrayStartState`),
     which the solve restores (it was reset there and lost), and the mode's run ends at its start. =#
  @testset "A recompilation at the initialization" begin
    local program = OMFrontend.OpenModelicaParser.Modelica"""
      model VSSInitialGrow
        parameter Integer N = 2;
        Real x[N](start = {i for i in 1:N});
      equation
        for i in 1:N loop
          der(x[i]) = -x[i];
        end for;
        when initial() then
          recompilation(N, 3);
        end when;
      end VSSInitialGrow;
      """
    local name::String = _vssTranslate(program, "VSSInitialGrow")
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    local sols = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-10)
    @test length(sols) == 2
    @test sols[1].t[end] == 0.0 && sols[1].retcode == ReturnCode.Success
    @test length(sols[2].u[end]) == 3
    @test _vssValue(sols[2], "x[3]") ≈ 3 * exp(-1.0) atol = 1e-8
  end
end
