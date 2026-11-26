#= Events of the array code (Author: John Tinnerholm): coincident ones, the step after one, their buffers. Relations crossing within an event's instant
   after it change in that event (ArrayPreserveDE._coincidentCrossings): the root finder reports only the crossings at
   its earliest root, and coincident ones, found 7e-13 s apart at reltol 1e-6, were an event each (MSL EngineV6's
   cylinders at dead center: 651 events, 326 merged). After an event whose right-hand side changes slowly over the step
   proposed before it, that step is kept (ArrayPreserveDE._slowAfterEvent; EngineV6: 4876 steps to 2951). The event
   code evaluates the equations into the parameters' buffers (ArrayPreserveDE._eventBuffers; EngineV6: 168 to 45 MiB
   in its event steps). =#
using Test
import OMBackend
import OMFrontend

@testset "Coincident crossings are one event (array code)" begin
  #= 2 x > 0 and x > 0 cross zero together; their hysteresis bands put their roots 5e-11 s apart at reltol 1e-6
     (2 x at H / 2, x at H). x > 1e-6 crosses 1e-6 s later: an event of its own. =#
  local program = OMFrontend.OpenModelicaParser.Modelica"""
    model CoincidentCrossings
      Real x(start = -0.5, fixed = true);
      Boolean b1 = 2 * x > 0;
      Boolean b2 = x > 0;
      Boolean b3 = x > 1e-6;
      discrete Integer c2(start = 0, fixed = true);
      discrete Integer c3(start = 0, fixed = true);
    equation
      der(x) = 1;
      when b1 then
        c2 = if b2 then 1 else 2;
        c3 = if b3 then 1 else 2;
      end when;
    end CoincidentCrossings;"""
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("CoincidentCrossings", OMFrontend.translateToSCode(program);
                                                      scalarize = false)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarize = false)
  local name::String = OMBackend.canonicalName("CoincidentCrossings")
  try
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    local sol = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-6, abstol = 1e-6)
    @test sol.retcode == ReturnCode.Success
    #= the when on b1 sees b2 true at its event, b3 not yet =#
    @test OMBackend.variableValues(sol, "c2")[end] == 1
    @test OMBackend.variableValues(sol, "c3")[end] == 2
    @test OMBackend.variableValues(sol, "b3")[end] == true
    #= two events after the start's: b1 with b2 at 0.5, b3 at 0.5 + 1e-6 =#
    local ts::Vector{Float64} = sol.t
    local events::Vector{Float64} = [ts[i] for i in 1:length(ts) - 1 if ts[i] == ts[i + 1] && ts[i] > 0]
    @test length(events) == 2
    @test isapprox(events[1], 0.5; atol = 1e-9) && isapprox(events[2], 0.5 + 1e-6; atol = 1e-9)
  finally
    OMBackend.forgetModel(name)
  end
end

@testset "The step after an event that changes the right-hand side slowly is kept (array code)" begin
  local APDE = OMBackend.CodeGeneration.ArrayPreserveDE
  local ODE = OMBackend.OrdinaryDiffEq
  local rodas = ODE.Rodas5P(autodiff = OMBackend.CodeGeneration.ADTypes.AutoFiniteDiff())
  #= x' = 1 from 0 (x = t), an event at x = 0.5 inside a first step of 0.9, after which x' is
     - :smooth, still 1: the step of 0.9 kept;
     - :jump, (0.6 - x) / 1e-4: the derivative jumps from 1 to 1000 (a transient of 1e-4 s);
     - :transient, 1 + (2 t - 0.5 - x) / 1e-4: continuous at the event (1), x then follows a ramp of slope 2 within
       1e-4 s (`ArrayPath.HighGainKink`'s case, found by the probe).
     The values at 0.75: 0.75, 0.6, and the ramp 1.0 less the lag 1e-4. =#
  for (kind::Symbol, small::Bool, x075::Float64) in ((:smooth, false, 0.75), (:jump, true, 0.6),
                                                     (:transient, true, 1.0 - 1e-4))
    local mode::Base.RefValue{Symbol} = Base.RefValue{Symbol}(:before)
    local rhs!(du, u, p, t) = (du[1] = mode[] === :jump ? (0.6 - u[1]) / 1e-4 :
                                       mode[] === :transient ? 1.0 + (2 * t - 0.5 - u[1]) / 1e-4 : 1.0; nothing)
    local proposed::Vector{Float64} = Float64[]
    local affect!(integ) = (mode[] = kind; APDE._restartStep!(integ); push!(proposed, abs(integ.dtpropose)); nothing)
    local event = ODE.ContinuousCallback((u, t, integ) -> u[1] - 0.5, affect!)
    local sol = ODE.solve(ODE.ODEProblem(rhs!, [0.0], (0.0, 1.0)), rodas; callback = event, dt = 0.9, reltol = 1e-8,
                          abstol = 1e-10)
    @test sol.retcode == ReturnCode.Success
    @test length(proposed) == 1
    @test (proposed[1] <= 1e-6) == small
    @test sol(0.75)[1] ≈ x075 atol = 1e-8
  end
end

"The bytes one evaluation of the relations' and when-conditions' crossing functions allocates at the integrator's point."
function conditionBytes(integ::OMBackend.CodeGeneration.DiffEqBase.DEIntegrator, out::Vector{Float64})::Int
  local APDE::Module = OMBackend.CodeGeneration.ArrayPreserveDE
  APDE.conditions!(out, integ.u, integ.t, integ)
  return @allocated APDE.conditions!(out, integ.u, integ.t, integ)
end

@testset "The event code evaluates the equations into the parameters' buffers (array code)" begin
  #= a when-equation: the conditions evaluate all the equations (not only the relations' rows), into buffers of the
     200 algebraic variables (a fresh pair at each evaluation: 1.6 KB) =#
  local program = OMFrontend.OpenModelicaParser.Modelica"""
    model EventBuffers
      parameter Integer n = 200;
      Real x(start = -0.5, fixed = true);
      Real y[n];
      discrete Integer k(start = 0, fixed = true);
    equation
      der(x) = 1;
      for i in 1:n loop
        y[i] = i * sin(x);
      end for;
      when y[n] > 0 then
        k = pre(k) + 1;
      end when;
    end EventBuffers;"""
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("EventBuffers", OMFrontend.translateToSCode(program); scalarize = false)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarize = false)
  local name::String = OMBackend.canonicalName("EventBuffers")
  try
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    #= the simulation first: it evaluates the model's module =#
    local sol = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-6, abstol = 1e-6)
    @test sol.retcode == ReturnCode.Success
    @test OMBackend.variableValues(sol, "k")[end] == 1
    local mod::Module = OMBackend._modelModule(Symbol(name))
    local (prob, cbs) = Base.invokelatest(getglobal(mod, Symbol(name, "Model")), (0.0, 1.0))
    local integ = Base.invokelatest(OMBackend.DifferentialEquations.init, prob, OMBackend.defaultSolver(); callback = cbs)
    Base.invokelatest(OMBackend.DifferentialEquations.step!, integ)
    local p = integ.p
    @test !p.code.crossingsOnly && length(p.abuf) >= 200
    @test Base.invokelatest(conditionBytes, integ, zeros(length(p.rel) + length(p.wnew))) < 8 * length(p.abuf)
  finally
    OMBackend.forgetModel(name)
  end
end
