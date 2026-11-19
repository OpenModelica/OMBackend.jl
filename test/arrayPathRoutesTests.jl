#= The array path tests that build their models on both routes, the scalarizing one (`scalarized = true`) and the
   arrays-kept one, each against the same values: on the routes of `ROUTES` (runtests.jl; CI runs each route in a
   shard of its own). Helpers: arrayPathHelpers.jl. =#

"The model `ArrayPath.<model>` translated on its route and simulated on (0, stop); taken: it must take the array path."
function _routeRun(model::String, scalarized::Bool; taken::Bool = true, stop::Float64 = 1.0,
                   parameters::Dict{String, Any} = Dict{String, Any}())
  local n::String = _arrayPathTranslate("ArrayPath." * model; scalarized)
  (scalarized || !taken) || @test _arrayPathTaken(n)
  return isempty(parameters) ? OMBackend.simulateModel(n; tspan = (0.0, stop)) :
                               OMBackend.resimulateModel(n; tspan = (0.0, stop), parameters)
end

"The exception of `_routeRun(model, scalarized)`, or `nothing`."
function _routeFailure(model::String, scalarized::Bool)
  try
    _routeRun(model, scalarized)
    return nothing
  catch e
    return e
  end
end

"The values of `var` in `sol` at the times `ts` (the last saved point at or before each)."
function _at(sol, var::String, ts::Vector{Float64})::Vector{Float64}
  local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
  return [v[findlast(<=(t + 1e-12), sol.t)] for t in ts]
end

@testset "Array path: both routes" begin
  @testset "terminate() in if-branches, outside when-equations, in if-equations of when-equations" begin
    #= each on the routes of ROUTES =#
    local runs = (model::String) -> [_routeRun(model, scalarized) for scalarized in ROUTES]
    #= at the crossing of the branch's condition (OpenModelica 1.27.1: at its output step after it, 0.752); at the
       initialization where it holds at the start, as OpenModelica; in an else-branch's if-equation (OpenModelica
       ignores it) =#
    for (model, stop) in (("TerminateInBranch", 0.75), ("TerminateAtStart", 0.0), ("TerminateInElse", 0.6), ("TerminateNow", 0.0))
      for sol in runs(model)
        @test sol.retcode == ReturnCode.Terminated
        @test sol.t[end] ≈ stop atol = 1e-6
      end
    end
    #= in an if-equation of a when-equation (OpenModelica fails to build it): the fourth tick, 0.35 =#
    for sol in runs("TerminateInWhenIf")
      @test sol.retcode == ReturnCode.Terminated
      @test sol.t[end] ≈ 0.35 atol = 1e-9
      @test OMBackend.getVariableValues(sol, "n")[end] == 4.0
    end
    local DE = OMBackend.DifferentialEquations
    #= a terminate() at the initialization: the problem of getMTKProblem ends at its start, on both paths =#
    for scalarized in ROUTES
      local n = _arrayPathTranslate("ArrayPath.TerminateAtStart"; scalarized)
      local (prob, cbs) = OMBackend.getMTKProblem(n)
      @test prob.tspan == (0.0, 0.0)
      @test Base.invokelatest(DE.solve, prob, DE.Rodas5P(); callback = cbs).t[end] == 0.0
    end
  end

  @testset "asserts: violations between steps by their relations' crossings, a relation touching its boundary" begin
    #= a violation for x in (0.7, 0.9) within one step of der(x) = 1, by a plain assert and by one in an elseif branch
       (`x <= lim`: at its boundary at the crossing, violated after it); OpenModelica 1.27.1 fails at 0.701 =#
    for model in ("AssertWindow", "BranchAssertWindow"), scalarized in ROUTES
      local n = _arrayPathTranslate("ArrayPath." * model; scalarized)
      scalarized || @test _arrayPathTaken(n)
      local err = try
        OMBackend.simulateModel(n; tspan = (0.0, 1.0))
        nothing
      catch e
        e
      end
      @test err isa OMBackend.CodeGeneration.ModelicaAssertionError
      @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && abs(err.time - 0.7) < 1e-6
    end
    #= x reaches 1 at 0.5 and falls again: x <= 1 holds =#
    for scalarized in ROUTES
      local n = _arrayPathTranslate("ArrayPath.AssertTouch"; scalarized)
      @test OMBackend.simulateModel(n; tspan = (0.0, 1.0)).retcode == ReturnCode.Success
    end
  end

  @testset "self-reading algorithms, when-equation order, starts, asserts at events, terminate()" begin
    #= each on the routes of ROUTES (taken where `taken`) =#
    for scalarized in ROUTES
      #= a whole-array assignment that reads its own target: its values first, into temporaries (MLS 11.2.1; the
         elements in order read the new x[1]: {2, 2}); OpenModelica 1.27.1's values =#
      local sw = _routeRun("SwapInWhen", scalarized)
      @test _at(sw, "x[1]", [0.05, 0.2, 0.4]) == [1.0, 2.0, 1.0]
      @test _at(sw, "x[2]", [0.05, 0.2, 0.4]) == [2.0, 1.0, 2.0]
      local rw = _routeRun("RotationInWhen", scalarized)
      @test _at(rw, "x[1]", [0.2, 0.4, 0.95]) ≈ [0.0, -1.0, 1.0] atol = 1e-12
      @test _at(rw, "x[2]", [0.2, 0.4, 0.95]) ≈ [1.0, 0.0, 0.0] atol = 1e-12
      @test _values(_routeRun("SwapRegular", scalarized), ["y[1]", "y[2]"]) ≈ [2.0, 1.0] rtol = 1e-9
      #= an if lifted on its own counted x[k] and x[1] apart (x[1] = 5 at the tick); omc 3 =#
      @test _at(_routeRun("Alias", scalarized; taken = false), "x[1]", [0.05, 0.2, 1.0]) == [3.0, 3.0, 3.0]
      #= the equations of a when-equation in an order that solves them: b reads the new a; terminate() after n's
         equation (it stopped at the fourth tick) =#
      local wo = _routeRun("WhenOrder", scalarized)
      @test _at(wo, "a", [0.2, 1.0]) == [2.0, 10.0] && _at(wo, "b", [0.2, 1.0]) == [3.0, 11.0]
      local wb = _routeRun("WhenIfBeforeAssign", scalarized)
      @test wb.retcode == ReturnCode.Terminated
      @test wb.t[end] ≈ 0.25 atol = 1e-9
      @test OMBackend.getVariableValues(wb, "n")[end] == 3.0
      #= starts of reductions the evaluation does not take: placeholders were 0 without overrides =#
      @test _values(_routeRun("StartSumNested", scalarized), ["x"]) ≈ [11.0] rtol = 1e-9
      @test _values(_routeRun("StartSumMatrix", scalarized), ["x"]) ≈ [11.0] rtol = 1e-9
      @test _at(_routeRun("DiscreteStartSum", scalarized), "d", [0.2, 1.0]) == [10.0, 11.0]
      @test OMBackend.getVariableValues(_routeRun("GuessStartSum", scalarized), "x")[1] ≈ -10.0 rtol = 1e-6
      #= asserts decided with the discrete values within the step, not those of the event that ended it; omc: the
         first holds, the second fails at 0.6; an if-expression condition; a value set at a sample tick =#
      @test _routeRun("AssertAfterEvent", scalarized).retcode == ReturnCode.Success
      for (model, time) in (("AssertFailAtEvent", 0.6), ("AssertIfExp", 0.8), ("AssertSampleCount", 0.25), ("AlgIfAssert", 0.7))
        local err = _routeFailure(model, scalarized)
        @test err isa OMBackend.CodeGeneration.ModelicaAssertionError
        @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && abs(err.time - time) < 1e-6
      end
      #= terminate() under noEvent(), in if-equations of an elsewhen, of a for-equation, in algorithm sections =#
      for (model, stop) in (("TerminateNoEvent", 0.5), ("TermElsewhen", 0.6), ("ForBranch", 0.3), ("AlgTerminate", 0.5),
                            ("AlgTerminateReal", 0.5))
        local sol = _routeRun(model, scalarized)
        @test sol.retcode == ReturnCode.Terminated
        @test sol.t[end] ≈ stop atol = 1e-6
      end
      #= max of a literal of two arrays: over their four elements =#
      @test _values(_routeRun("MaxOfArrays", scalarized), ["y"]) ≈ [2.0] rtol = 1e-9
    end
  end

  @testset "event functions, self-reading assignments, when-assigned records and enumerations, run-time subscripts" begin
    #= review 20's models, each on the scalarized path and on the array path (taken where `taken`); OpenModelica
       1.27.1's values (or the analytic ones) =#
    local T5 = [0.05, 0.2, 0.4, 0.7, 1.0]
    local TI = [0.04, 0.12, 0.57, 0.98]
    for scalarized in ROUTES
      #= a regular Real swap and a when swap in one section; elements that read each other (no cycle by element) =#
      local ms = _routeRun("MixedSection", scalarized)
      @test _values(ms, ["y[1]", "y[2]"]) ≈ [2.0, 1.0] rtol = 1e-9
      @test _at(ms, "z[1]", [0.05, 0.2, 0.4, 1.0]) == [1.0, 2.0, 1.0, 1.0]
      @test _at(_routeRun("ElemCycle", scalarized), "x[2]", [0.1, 0.2, 1.0]) == [3.0, 5.0, 21.0]
      #= if-statements whose branches assign what their conditions read: the conditions' values before the branches =#
      local ig = _routeRun("IfGuardReassign", scalarized)
      @test _values(ig, ["m", "n"]) == [2.0, 0.0]
      local is = _routeRun("IfGuardSwap", scalarized)
      @test _at(is, "x[1]", T5) == [2.0, 1.0, 1.0, 1.0, 1.0] && _at(is, "x[2]", T5) == [1.0, 2.0, 2.0, 2.0, 2.0]
      @test _values(_routeRun("IfGuardRegular", scalarized), ["y", "z"]) == [20.0, 1.0]
      local id = _routeRun("IfGuardDiscreteRegular", scalarized)
      @test _at(id, "m", [0.3, 0.8]) == [3.0, 7.0] && _at(id, "n", [0.3, 0.8]) == [0.0, 2.0]
      #= asserts and terminate() of algorithm sections at their point (after reassignments, in for loops), asserts after
         an event's reinit and discrete changes =#
      @test _routeRun("AlgAssertTopReassigned", scalarized).retcode == ReturnCode.Success
      for (model, time) in (("AlgAssertReassigned", 0.7), ("AlgAssertFor", 0.4), ("AssertBoolAlg", 0.6), ("AssertReinit", 0.5),
                            ("AssertReinitDiscrete", 0.5))
        local err = _routeFailure(model, scalarized)
        @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && abs(err.time - time) < 1e-5
      end
      local tf = _routeRun("AlgTermFor", scalarized)
      @test tf.retcode == ReturnCode.Terminated && abs(tf.t[end] - 0.3) < 1e-5
      #= the guard reads the discrete n (its value inlined, the operand landed on the threshold: never fired) =#
      local td = _routeRun("TermDiscreteGuard", scalarized)
      @test td.retcode == ReturnCode.Terminated && abs(td.t[end] - 0.5) < 1e-5
      #= ... also where the section assigns it again after the guard (its intermediate value: an operand that jumped onto
         the threshold of a relation; one built from relations and parameters only has no crossing, held between events) =#
      local tb = _routeRun("TermBeforeLastAssignment", scalarized)
      @test tb.retcode == ReturnCode.Terminated && abs(tb.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(tb, "n")[end] == 6.0
      #= ... against a Real literal or parameter, and of Real values (omc's hysteresis does not fire the last one, MLS
         11.1.2 says it terminates) =#
      for model in ("TermRealThreshold", "TermParamThreshold", "TermRealBranches")
        local tr = _routeRun(model, scalarized)
        @test tr.retcode == ReturnCode.Terminated && abs(tr.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(tr, "n")[end] == 6.0
      end
      #= an opaque function's Integer result (a continuous argument too) landing on its threshold, through arithmetic
         and a sign change: its crossing relation takes the exact zero (`_relationAt`). The scalarized route's relation
         callbacks miss these (as they miss `(if x > 0.5 then 5 else 0) >= 5`: older, open) =#
      for model in ("OpaqueIntegerLanding", "OpaqueSumParam", "OpaqueNegated")
        scalarized || @test _at(_routeRun(model, scalarized), "y", [0.4, 0.75, 1.0]) == [0.0, 1.0, 1.0]
      end
      #= a relation whose operand lands exactly on its threshold at an event (an opaque function's value, a promoted
         Integer, abs() of it) is true there (`_relationAt`: the band kept it false forever); a relation of a noEvent'd
         condition's if-expression or of an Integer function of a continuous argument keeps its crossing (omc fires
         neither of those two). Operands held between events (built from relations, parameters and discretes) compare
         literally: a rounded sum `0.1 + 0.2 >= 0.3` and a strict relation on a plateau of exactly its threshold =#
      scalarized || @test _at(_routeRun("RoundedWhen", scalarized), "hits", [0.4, 0.75, 1.0]) == [0.0, 1.0, 1.0]
      scalarized || @test _at(_routeRun("StrictLanding", scalarized), "hits", [0.2, 0.3, 0.6, 0.8, 1.0]) == [0.0, 1.0, 1.0, 2.0, 2.0]
      #= a 64-bit Integer intermediate: the relation's difference of two native Integers must not wrap around (a false guard
         turned true before the event); the value is exact in Float64 =#
      scalarized || @test (local wi = _routeRun("WideIntegerIntermediate", scalarized); wi.retcode == ReturnCode.Terminated && abs(wi.t[end] - 0.5) < 1e-5)
      #= a relation `abs(x) <= b` with a small b holds only in a narrow window around x = 0, which a step crosses with its
         crossing function positive at both ends: the zero of x is watched too (`_absProbes`, `_withAbsProbe`) =#
      @test _at(_routeRun("AbsWindow", scalarized; stop = 0.05), "n", [0.004, 0.006, 0.016, 0.05]) == [0.0, 1.0, 2.0, 5.0]
      local ri = _routeRun("RoundedIntermediate", scalarized)
      @test ri.retcode == ReturnCode.Terminated && abs(ri.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(ri, "n")[end] == 2.0
      scalarized || @test OMBackend.getVariableValues(_routeRun("IfNoEventCondition", scalarized), "hits")[end] == 1.0
      scalarized || @test OMBackend.getVariableValues(_routeRun("IntegerFunctionNeedsCrossing", scalarized), "hits")[end] == 1.0
      local ta = _routeRun("TermOpaqueIntegerAbs", scalarized)
      @test ta.retcode == ReturnCode.Terminated && abs(ta.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(ta, "n")[end] == 6.0
      #= ... also promoted to Real by the frontend's cast in an intermediate-value guard (round 14's symbolic run) =#
      local to = _routeRun("TermOpaqueInteger", scalarized)
      @test to.retcode == ReturnCode.Terminated && abs(to.t[end] - 0.5) < 1e-5 && OMBackend.getVariableValues(to, "n")[end] == 6.0
      #= an elseif's condition (a snapshot of the conditions before the branches) and a branch's partial function are
         evaluated only where their guards hold: sqrt of a negative d =#
      local ie = _routeRun("InactiveElseif", scalarized)
      @test _at(ie, "d", [0.05, 0.95]) == [-1.0, -10.0] && _at(ie, "n", [0.05, 0.95]) == [0.0, 9.0]
      @test _at(_routeRun("GuardedSqrt", scalarized), "y", [0.05, 0.95]) == [0.0, 9.0]
      #= initial arms: run once at the initialization (a count, swaps); a swap in an initial algorithm of states =#
      @test _at(_routeRun("InitArmCount", scalarized), "n", [0.0, 1.0]) == [1.0, 5.0]
      @test _at(_routeRun("InitialArmSwap", scalarized), "x[1]", [0.0, 0.05, 0.2, 1.0]) == [2.0, 2.0, 1.0, 2.0]
      @test _values(_routeRun("InitWhenSwap", scalarized), ["x[1]", "x[2]"]) == [2.0, 1.0]
      @test _values(_routeRun("InitAlgSwap", scalarized; taken = false), ["x[1]", "x[2]"]) == [2.0, 1.0]
      #= event functions of continuous arguments: events at each crossing, in equations and in algorithm sections
         (an algorithm's Integer swap, one argument through an earlier assignment) =#
      @test _at(_routeRun("EqInteger", scalarized), "n", TI) == [0.0, 1.0, 5.0, 9.0]
      @test _at(_routeRun("EqIntegerReal", scalarized), "y", TI) == [0.0, 1.0, 5.0, 9.0]
      local a2 = _routeRun("AlgIntegerTwo", scalarized)
      @test _at(a2, "n", TI) == [0.0, 1.0, 5.0, 9.0] && _at(a2, "m", TI) == [1.0, 2.0, 6.0, 10.0]
      @test _at(_routeRun("AlgIntegerReadsAssigned", scalarized), "n", TI) == [0.0, 2.0, 11.0, 19.0]
      local mi = _routeRun("MixedRealInt", scalarized)
      @test _at(mi, "n[1]", TI) == [0.0, 2.0, 11.0, 19.0] && _at(mi, "n[2]", TI) == [0.0, 1.0, 5.0, 9.0]
      #= a step variable of an argument through an if-equation (the scalarized route's step body read its own triggers,
         pinned after the crossing: 4); one of an inner step (a chain of whens: unbalanced) =#
      local si = _routeRun("StepThroughIfEq", scalarized; stop = 10.0)
      local (su, sy) = (OMBackend.getVariableValues(si, "u"), OMBackend.getVariableValues(si, "y"))
      @test all(((a, b),) -> b == (a > 0 ? floor(a + 0.5) : ceil(a - 0.5)) || abs(abs(a) - round(abs(a) - 0.5) - 0.5) < 1e-6,
                zip(su, sy))
      #= a whole period: the falling crossings (4: 2) and the negative half (6.3: -2, 8: -3), omc =#
      @test _at(si, "y", [1.0, 2.5, 4.0, 6.3, 8.0]) == [2.0, 3.0, 2.0, -2.0, -3.0]
      local sc = _routeRun("StepChain", scalarized)
      @test _at(sc, "n", [0.3, 0.5, 0.9]) == [1.0, 2.0, 4.0] && _at(sc, "cnt", [0.3, 0.5, 0.9]) == [1.0, 2.0, 4.0]
      @test _routeRun("StepOfDer", scalarized).retcode == ReturnCode.Success
      #= ceil, div, mod, rem of an oscillating argument, both signs: each saved point off a jump against the formula =#
      local q = (x::Float64) -> 3.7x
      local saved = (sol, var::String, f::Function, g::Function) -> begin
        local xs::Vector{Float64} = OMBackend.getVariableValues(sol, "x")
        local ys::Vector{Float64} = OMBackend.getVariableValues(sol, var)
        return maximum(abs(y - f(x)) for (x, y) in zip(xs, ys) if abs(g(x) - round(g(x))) > 1e-5)
      end
      local ef = _routeRun("EventFns", scalarized)
      @test saved(ef, "c", x -> ceil(q(x)), q) < 1e-9
      @test saved(ef, "d", x -> trunc(q(x) / 1.3), x -> q(x) / 1.3) < 1e-9
      @test saved(ef, "m", x -> q(x) - floor(q(x) / 1.3) * 1.3, x -> q(x) / 1.3) < 1e-9
      @test saved(ef, "r", x -> q(x) - trunc(q(x) / 1.3) * 1.3, x -> q(x) / 1.3) < 1e-9
      local ae = _routeRun("AlgEventFns", scalarized)
      @test saved(ae, "i", x -> ceil(q(x)), q) < 1e-9
      @test saved(ae, "j", x -> trunc(q(x) / 1.3), x -> q(x) / 1.3) < 1e-9
      #= swaps by an index vector, a record by its constructor or a function, an enumeration array; starts of
         enumerations; a target by a run-time subscript =#
      @test _at(_routeRun("IndexSwap", scalarized), "x[1]", T5) == [1.0, 2.0, 1.0, 2.0, 1.0]
      @test _at(_routeRun("RecordSwapCtor", scalarized), "r.a", T5) == [1.0, 2.0, 1.0, 2.0, 1.0]
      @test _at(_routeRun("RecordSwapFunc", scalarized; taken = false), "r.a", T5) == [1.0, 2.0, 1.0, 2.0, 1.0]
      @test _at(_routeRun("EnumSwap", scalarized; taken = false), "i1", T5) == [1.0, 3.0, 1.0, 3.0, 1.0]
      local es = _routeRun("EnumStart", scalarized; taken = false)
      @test _at(es, "e1", [0.1, 0.7]) == [1.0, 3.0] && _at(es, "e2", [0.1, 0.7]) == [2.0, 1.0]
      local vs = _routeRun("VarSubscript", scalarized)
      @test _at(vs, "x[1]", T5) == [1.0, 12.0, 12.0, 32.0, 32.0] && _at(vs, "x[2]", T5) == [2.0, 2.0, 22.0, 22.0, 42.0]
    end
  end

  @testset "a slide's coefficients zero at some angle, enumeration starts" begin
    #= the slide's acceleration has a coefficient in each of the three force equations, R.T[2, i], every one 0 at
       some angle (R.T[2, 1] at the start): not divided by, an unknown of the solver's linear system; to t = 4, past
       the zeros of all of them =#
    for scalarized in ROUTES
      @test _values(_routeRun("SlidingMassAligned", scalarized; stop = 4.0), ["s"]) ≈
            [9.81 / 2 * cos(4) + (1 - 9.81 / 2) * cosh(4)] rtol = 1e-4
    end
    #= an enumeration without a start starts at its first literal (MLS 4.9.5; omc 1.27.1: 0, an assertion) =#
    for scalarized in ROUTES
      local ed = _routeRun("EnumDefaultStart", scalarized)
      @test _at(ed, "i[1]", [0.1, 0.7]) == [10.0, 30.0] && _at(ed, "i[2]", [0.1, 0.7]) == [10.0, 20.0]
    end
  end

  @testset "A row of a parameter matrix by a discrete index" begin
    local program = Modelica"""
      package ByIndex
        model TableRow
          parameter Real table[3, 2] = {{1, 4}, {2, 2}, {3, 7}};
          final parameter Integer val[3, 2] = integer(table + fill(1, 3, 2) * 1e-37);
          Integer idx(start = 1, fixed = true);
          discrete Integer y[2];
        initial equation
          y[:] = val[idx, :];
        equation
          when sample(0.3, 0.3) then
            idx = if pre(idx) < 3 then pre(idx) + 1 else 1;
            y[:] = val[idx, :];
          end when;
        end TableRow;
      end ByIndex;
      """
    #= `val[idx, :]`: the array index reduction declines a subscript that is not constant; the backend
       scalarization selects its elements by the discrete idx (`selectElement`), it read val as a whole
       (UndefVarError). idx: 1, 2 at 0.3, 3 at 0.6, 1 at 0.9 (CDL's Integer TimeTable) =#
    for scalarized in ROUTES
      local rows::String = _translateProgram(program, "ByIndex.TableRow"; scalarized)
      local sol = OMBackend.simulateModel(rows; tspan = (0.0, 1.0))
      @test sol.retcode == ReturnCode.Success
      @test [OMBackend.getVariableValues(sol, "y[2]")[1]; _values(sol, ["y[1]", "y[2]"])] == [4.0, 1.0, 4.0]
    end
  end

  #= the events among a model's callbacks (isEventCallback): an assert and a run's bookkeeping are none (OM.eventCount
     counted the assert check on the scalarizing route, the start state, first step and assert callbacks on the
     arrays-kept one: a UDE over the model refused); a relation is one =#
  @testset "Event callbacks: asserts and bookkeeping are no events" begin
    local program = Modelica"""
      package EventKinds
        model Asserted
          Real x[2](each start = 1, each fixed = true);
        equation
          der(x) = -{1, 2} .* x;
          assert(x[1] > 0.1, "x[1] too small");
        end Asserted;
        model Switched
          Real x[2](each start = 1, each fixed = true);
          Real y;
        equation
          der(x) = -{1, 2} .* x;
          y = if x[1] > 0.5 then 1 else 0;
        end Switched;
      end EventKinds;
      """
    local events = cb -> cb === nothing ? 0 :
      cb isa OMBackend.CodeGeneration.DiffEqBase.CallbackSet ?
        count(OMBackend.isEventCallback, cb.continuous_callbacks) + count(OMBackend.isEventCallback, cb.discrete_callbacks) :
        Int(OMBackend.isEventCallback(cb))
    for scalarized in ROUTES
      local asserted::String = _translateProgram(program, "EventKinds.Asserted"; scalarized)
      local sol = OMBackend.simulateModel(asserted; tspan = (0.0, 1.0))
      @test sol.retcode == ReturnCode.Success
      @test OMBackend.eventCallbacks(asserted) !== nothing
      @test events(OMBackend.eventCallbacks(asserted)) == 0
      @test events(get(sol.prob.kwargs, :callback, nothing)) == 0
      local switched::String = _translateProgram(program, "EventKinds.Switched"; scalarized)
      @test events(OMBackend.eventCallbacks(switched)) > 0
      @test _values(OMBackend.simulateModel(switched; tspan = (0.0, 1.0)), ["y"]) == [0.0]
    end
  end

  #= a tuple equation with a record target and fewer targets than outputs: q its position 3 (r.a, r.b, q); the
     scalarizing route =#
  if true in ROUTES
    @testset "A tuple with fewer targets than outputs" begin
      local fewer = _arrayPathTranslate("ArrayPath.TupleFewerTargets"; scalarized = true)
      @test _values(OMBackend.simulateModel(fewer; tspan = (0.0, 1.0)), ["q"]) ≈ [3.0]
    end
  end
end
