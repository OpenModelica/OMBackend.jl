#= Array-preserving code generation (translate with scalarized = false): models from the
   frontend's no-scalarize mode keep their loops in the generated code; results equal the
   scalarizing path's. =#

#= helpers: arrayPathHelpers.jl (runtests.jl includes it) =#

include("arrayPathReferences.jl")

"The key of a comparison in `SCALARIZED_REFERENCES`: the model, the names, the interval and the solver options."
function _referenceKey(model::String, names::Vector{String}, tspan::Tuple{Float64, Float64}, kwargs)::String
  return string(model, " | ", join(names, ", "), " | ", tspan, " | ", join(sort!(["$k = $v" for (k, v) in kwargs]), ", "))
end

"""
The scalarizing path's values of `names` at the end of `tspan`, as recorded in `SCALARIZED_REFERENCES`
(arrayPathReferences.jl), so that each model is built once, on the array path. A comparison not recorded builds the
scalarizing path (`build()`), fails, and prints the entry to add.
"""
function _scalarizedReference(build::F, model::String, names::Vector{String}, tspan, kwargs)::Vector{Float64} where {F <: Function}
  local key::String = _referenceKey(model, names, (Float64(tspan[1]), Float64(tspan[2])), kwargs)
  haskey(SCALARIZED_REFERENCES, key) && return SCALARIZED_REFERENCES[key]
  local s::Vector{Float64} = build()
  @error "No recorded scalarized values; add to arrayPathReferences.jl:\n  $(repr(key)) => $(repr(s)),"
  @test haskey(SCALARIZED_REFERENCES, key)
  return s
end

function _bothWays(model, names; tspan = (0.0, 1.0), kwargs...)
  local name = OMBackend.canonicalName(model)
  local s::Vector{Float64} = _scalarizedReference(model, names, tspan, kwargs) do
    _arrayPathTranslate(model; scalarized = true)
    return _values(OMBackend.simulateModel(name; tspan = tspan, kwargs...), names)
  end
  _arrayPathTranslate(model; scalarized = false)
  @test _arrayPathTaken(name)
  local sol::OMBackend.DifferentialEquations.SciMLBase.AbstractODESolution = OMBackend.simulateModel(name; tspan = tspan, kwargs...)
  @test sol.retcode == ReturnCode.Success
  return (s, _values(sol, names), sol)
end

"""
`_bothWays` of the model `model` of the Modelica text `program`: the values of `names` at the end of `tspan` on the
scalarizing path and on the array path (which the model must take), and the array path's solution.
"""
function _bothWaysOf(program, model::String, names::Vector{String}; tspan = (0.0, 1.0), kwargs...)
  local name::String = OMBackend.canonicalName(model)
  local s::Vector{Float64} = _scalarizedReference(model, names, tspan, kwargs) do
    _translateProgram(program, model; scalarized = true)
    return _values(OMBackend.simulateModel(name; tspan = tspan, kwargs...), names)
  end
  _translateProgram(program, model; scalarized = false)
  @test _arrayPathTaken(name)
  local sol = OMBackend.simulateModel(name; tspan = tspan, kwargs...)
  @test sol.retcode == ReturnCode.Success
  return (s, _values(sol, names), sol)
end

@testset "Array path" begin
  @testset "Decay, n = 1000: loops kept" begin
    local name = _arrayPathTranslate("ArrayPath.Decay"; scalarized = false)
    @test _arrayPathTaken(name)
    local stats = OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[]
    @test stats.states == 1000 && stats.batches == 1
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    @test maximum(abs(sol.u[end][i] - exp(-i)) for i in 1:1000) < 1e-6
    @test OMBackend.getVariableValues(sol, "x[2]")[end] ≈ exp(-2) atol = 1e-6
  end

  @testset "Chain: connections, algebraic variables, as scalarized" begin
    local (s, a, sol) = _bothWays("ArrayPath.Chain", ["c[1].T", "c[5].T", "c[10].T", "c[2].p.Q", "g[1].a.T"])
    @test maximum(abs.(s .- a)) < 1e-6
    local loops = OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches
    #= the code does not grow with n =#
    _arrayPathTranslate("ArrayPath.Chain1000"; scalarized = false)
    @test OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches == loops
    #= symbolic indexing: states and observed (algebraic) variables =#
    @test sol[Symbol("c[1].T")][end] ≈ a[1]
    @test sol(1.0; idxs = Symbol("c[2].p.Q")) ≈ a[4] atol = 1e-8
  end

  @testset "Parameters: overrides and dependent bindings" begin
    local name = _arrayPathTranslate("ArrayPath.ParamDep"; scalarized = false)
    @test _arrayPathTaken(name)
    local x = OMBackend.simulateModel(name; tspan = (0.0, 1.0)).u[end]
    @test x ≈ exp.(-[2.0, 4.0, 6.0]) rtol = 1e-4
    #= b = 3a and k = {a, 2a, b} follow a =#
    x = OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("a" => 1.0)).u[end]
    @test x ≈ exp.(-[1.0, 2.0, 3.0]) rtol = 1e-4
    x = OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("k[2]" => 0.5)).u[end]
    @test x[2] ≈ exp(-0.5) rtol = 1e-4
    #= an element override with an override of what the array reads: the array's rule, then the element (k[1] kept
       its translated 2 before) =#
    x = OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("a" => 1.0, "k[2]" => 0.5)).u[end]
    @test x ≈ exp.(-[1.0, 0.5, 3.0]) rtol = 1e-4
    #= a Real array whose rule gives Integers keeps its type (an element override of 0.5 threw InexactError) =#
    local ib = _arrayPathTranslate("ArrayPath.IntBoundRealArray"; scalarized = false)
    @test _arrayPathTaken(ib)
    local xi = OMBackend.resimulateModel(ib; tspan = (0.0, 1.0), parameters = Dict{String, Any}("m" => 3, "k[2]" => 0.5))
    @test OMBackend.getVariableValues(xi, "x")[end] ≈ 3.5 rtol = 1e-6
    #= parameter names that differ in . and _ only are their own parameters (one symbol: x(1) = 66, omc 26) =#
    local nc = _arrayPathTranslate("ArrayPath.ParamNameClash"; scalarized = false)
    @test _arrayPathTaken(nc)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(nc; tspan = (0.0, 1.0)), "x")[end] ≈ 26.0 rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(nc; tspan = (0.0, 1.0), parameters = Dict{String, Any}("c_k" => 1.0)), "x")[end] ≈ 13.0 rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(nc; tspan = (0.0, 1.0), parameters = Dict{String, Any}("c.k" => 5.0)), "x")[end] ≈ 25.0 rtol = 1e-6
    #= a binding that references another parameter follows its overrides (it was that one's value: gain = 5 gave 5532,
       omc 5535) =#
    local pr = _arrayPathTranslate("ArrayPath.ParamReference"; scalarized = false)
    @test _arrayPathTaken(pr)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(pr; tspan = (0.0, 1.0)), "x")[end] ≈ 5232.0 rtol = 1e-6
    local po = Dict{String, Any}("gain" => 5.0, "p1" => 4.0, "a" => [4.0, 6.0])
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(pr; tspan = (0.0, 1.0), parameters = po), "x")[end] ≈ 5 + 40 + 500 + 6000 rtol = 1e-6
    #= a parameter coefficient of der() follows its override (the array commit folded it: c = 4 kept 2) =#
    local dc = _arrayPathTranslate("ArrayPath.DerCoefficient"; scalarized = false)
    @test _arrayPathTaken(dc)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(dc; tspan = (0.0, 1.0)), "x")[end] ≈ 0.5 rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(dc; tspan = (0.0, 1.0), parameters = Dict{String, Any}("c" => 4.0)), "x")[end] ≈ 0.25 rtol = 1e-6
    #= a coefficient that is a call of a parameter: a parameter of its own (computed at each start, not in each right-hand
       side), which follows the override =#
    local dk = _arrayPathTranslate("ArrayPath.DerCoefficientOfCall"; scalarized = false)
    @test _arrayPathTaken(dk)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(dk; tspan = (0.0, 1.0)), "x")[end] ≈ 1 / sin(0.5) rtol = 1e-6
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(dk; tspan = (0.0, 1.0), parameters = Dict{String, Any}("p" => 1.0)), "x")[end] ≈ 1 / sin(1.0) rtol = 1e-6
    #= starts that read parameters follow their overrides (they kept the translated values: 4) =#
    local sp = _arrayPathTranslate("ArrayPath.StartOfParameters"; scalarized = false)
    @test _arrayPathTaken(sp)
    local sps = OMBackend.resimulateModel(sp; tspan = (0.0, 1.0), parameters = Dict{String, Any}("p1" => 10.0))
    @test [OMBackend.getVariableValues(sps, v)[end] for v in ("y", "w")] ≈ [11.0, 11.0] rtol = 1e-6
    #= a fixed start that reads a parameter the initialization computes: y = q there (it took q's start, 1); the array
       code declines it (its initial algorithms run after the initialization), the scalar path's value omc's =#
    local sf = _arrayPathTranslate("ArrayPath.StartOfFreeParameter"; scalarized = false)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(sf; tspan = (0.0, 1.0)), "y")[1] ≈ 3.0 rtol = 1e-6
    #= a parameter a structural one's binding reads is structural too: its override is refused (it ran with the
       translated if-equation) =#
    local sd = _arrayPathTranslate("ArrayPath.StructuralDependency"; scalarized = false)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(sd; tspan = (0.0, 1.0)), "x")[end] ≈ 2.0 rtol = 1e-6
    @test_throws ErrorException OMBackend.resimulateModel(sd; tspan = (0.0, 1.0), parameters = Dict{String, Any}("R0" => -3.0))
    #= a call equation under an if-equation's branch: made where the branch holds (a TypeError: its condition was
       lifted into a Real) =#
    local cb = _arrayPathTranslate("ArrayPath.CallInIfBranch"; scalarized = false)
    @test _arrayPathTaken(cb)
    @test OMBackend.getVariableValues(OMBackend.resimulateModel(cb; tspan = (0.0, 1.0), parameters = Dict{String, Any}("lim" => 2.0)), "x")[end] ≈ 1.0 rtol = 1e-6
    @test_throws Exception OMBackend.resimulateModel(cb; tspan = (0.0, 1.0))
    #= a sample interval that is not finite is refused (its ticks did not end) =#
    local se = _arrayPathTranslate("ArrayPath.SampleEquation"; scalarized = false)
    @test_throws ErrorException OMBackend.resimulateModel(se; tspan = (0.0, 1.0), parameters = Dict{String, Any}("samplePeriod" => Inf))
    @test_throws Exception OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters = Dict("n" => 4))
  end

  @testset "ARRAY_PATH_FULL = false: events go the ModelingToolkit path" begin
    OMBackend.ARRAY_PATH_FULL[] = false
    try
      @test !_arrayPathTaken(_arrayPathTranslate("ArrayPath.BouncingBalls"; scalarized = false))
    finally
      OMBackend.ARRAY_PATH_FULL[] = true
    end
  end

  @testset "Events: when-equations in a loop, relations" begin
    local names = ["h[1]", "v[1]", "h[5]", "v[5]", "bounces[1]", "bounces[5]"]
    local (s, a, _) = _bothWays("ArrayPath.BouncingBalls", names; tspan = (0.0, 1.5))
    @test maximum(abs.(s .- a)) < 1e-4
    @test a[5] == 3.0
    local (s2, a2, _) = _bothWays("ArrayPath.Rectifier", ["vc[1]", "vc[3]"]; tspan = (0.0, 2.0))
    @test maximum(abs.(s2 .- a2)) < 1e-3
  end

  @testset "Asserts" begin
    local name = _arrayPathTranslate("ArrayPath.Guarded"; scalarized = false)
    @test _arrayPathTaken(name)
    @test_throws OMBackend.CodeGeneration.ModelicaAssertionError OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test OMBackend.simulateModel(name; tspan = (0.0, 0.4)).retcode == ReturnCode.Success
  end

  @testset "Initial array equations: for-equations over their elements (rollLeftArrayEquations!)" begin
    #= of references, of an inlined call against a literal (MSL Body: zeros(3) = equalityConstraint(Q, Q_start)), of a
       call that is not inlined (its result subscripted per element); q(1) = q0 exp(-1) =#
    for m in ("ArrayPath.ArrayInitialRefs", "ArrayPath.ArrayInitialCall", "ArrayPath.ArrayInitialLoopCall")
      local (s, a, _) = _bothWays(m, ["q[1]", "q[2]"]; reltol = 1e-9, abstol = 1e-9)
      @test maximum(abs.(a .- [exp(-1), 2 * exp(-1)])) < 1e-6
      @test maximum(abs.(s .- a)) < 1e-6
    end
    #= of a parameter array with fixed = false and of a discrete array: assignments of the initialization, an element
       at a time (no residual solves them); q' = -p q, d doubled at 0.5 =#
    local (s, a, sol) = _bothWays("ArrayPath.ArrayInitialFree", ["q[1]", "q[2]", "d[1]", "d[2]"]; reltol = 1e-9, abstol = 1e-9)
    @test maximum(abs.(a .- [exp(-1), exp(-2), 6, 8])) < 1e-6 && maximum(abs.(s .- a)) < 1e-6
    @test OMBackend.getVariableValues(sol, "d[2]")[1] == 4
  end

  @testset "The step after an event, a singular Jacobian at the start values, a coefficient of zero" begin
    local tight::NamedTuple = (reltol = 1e-9, abstol = 1e-9)
    #= y follows u with Td = 1e-4, u stops rising at 0.2: a = der(y) = 20 exp(-(t - 0.2) / Td). The step sized before
       the event crossed the transient at once: a(0.2003) read 18.9 (`_restartStep!`) =#
    local step::String = _arrayPathTranslate("ArrayPath.StepAfterEvent"; scalarized = false)
    @test _arrayPathTaken(step)
    local s1 = OMBackend.simulateModel(step; tspan = (0.0, 0.3), reltol = 1e-8, abstol = 1e-10)
    @test s1.retcode == ReturnCode.Success
    @test [s1(t; idxs = :a) for t in (0.2003, 0.2005, 0.21)] ≈ [20 * exp(-3), 20 * exp(-5), 0.0] atol = 1e-3
    #= a state following its input through a gain of 15000 into a limit, no event at the limit: the solver's own
       estimate of the first step after the slope's change is 0.01 s, the output at its limit after 2e-6 s; x stayed
       on the input, 5.32 at 0.11 (the values: this model at a tolerance of 1e-10) =#
    local kink::String = _arrayPathTranslate("ArrayPath.HighGainKink"; scalarized = false)
    @test _arrayPathTaken(kink)
    local s2 = OMBackend.simulateModel(kink; tspan = (0.0, 0.11), reltol = 1e-6, abstol = 1e-8)
    @test s2.retcode == ReturnCode.Success
    @test [s2(t; idxs = :x) for t in (0.101, 0.105, 0.11)] ≈ [0.6116052, 1.8580219, 3.4969888] atol = 1e-4
    #= (v[k] + i[k])^3 = P in a loop through the elements, the start values zero: its rows of the Jacobian are zero
       there. The loop was never solved, its unknowns NaN and the run a Success (`_blockJacobian!`: a least-norm step;
       `_checkBlocksSolved`) =#
    local i0::Float64 = 1 / 1.05
    local (s, a, sol) = _bothWays("ArrayPath.SingularStart", ["v[1]", "i[1]", "v[5]", "x"]; tight...)
    @test a ≈ [2 - i0, i0, 2 - i0, i0 * (1 - exp(-1))] atol = 1e-7
    @test maximum(abs.(s .- a)) < 1e-7
    @test all(b -> b.solved && b.failures == 0, sol.prob.p.blocks) && !isempty(sol.prob.p.blocks)
    #= i[k] * R[k] = v[k] - vi[k] with R = {0, 0, 1}: not solved for i[k] where R[k] is 0 (0 / 0 from the first
       evaluation on; `InstanceTable.vanishing`) =#
    local (sz, az, _) = _bothWays("ArrayPath.ZeroCoefficient", ["v[1]", "v[2]", "v[3]"]; tight...)
    @test az ≈ [exp(-0.5), exp(-0.5), exp(-1 / 3)] atol = 1e-7
    @test maximum(abs.(sz .- az)) < 1e-7
  end

  @testset "Initialization, sample, static if, discrete equations, algorithms" begin
    for (m, names, T) in (("ArrayPath.InitSteady", ["x", "y"], 1.0),
                          ("ArrayPath.SampleZOH", ["x", "u", "k"], 1.0),
                          ("ArrayPath.StaticIf", ["x[1]", "x[5]", "s", "sl"], 2.0),
                          ("ArrayPath.DiscreteEq", ["x", "m", "z[1]", "z[2]"], 2.0),
                          ("ArrayPath.AlgInitial", ["x", "t0", "z", "n"], 1.0))
      local (s, a, _) = _bothWays(m, names; tspan = (0.0, T))
      @test maximum(abs.(s .- a)) < 1e-5
    end
  end

  @testset "Event functions, vector when-conditions, fixed = false parameters, initial algorithms" begin
    #= at 0.95: floor(2.375), mod(0.95, 0.3), div(2.85, 1), integer(4x) rose at 0.25, 0.5, 0.75
       (not against the scalarized path: its when on integer(4x) > pre(n) fails, a BoundsError) =#
    local name = _arrayPathTranslate("ArrayPath.EventFunctions"; scalarized = false)
    @test _arrayPathTaken(name)
    local ev = OMBackend.simulateModel(name; tspan = (0.0, 0.95))
    @test _values(ev, ["y[1]", "y[2]", "y[3]", "n"]) ≈ [2.0, 0.05, 2.0, 3.0] atol = 1e-6
    #= the events: floor(2.5x) steps at 0.4 and 0.8, n at 0.25, 0.5, 0.75 =#
    @test [ev(t; idxs = Symbol("y[1]")) for t in (0.39, 0.41, 0.79, 0.81)] == [0.0, 1.0, 1.0, 2.0]
    @test [ev(t; idxs = :n) for t in (0.24, 0.26, 0.74, 0.76)] == [0.0, 1.0, 2.0, 3.0]
    #= initial() fires at the start, then each element when it becomes true =#
    local (_, v, _) = _bothWays("ArrayPath.VectorWhen", ["n", "tl"])
    @test v ≈ [3.0, 0.6] atol = 1e-6
    #= t0 = 0, k = x(0) / 2 = 1 =#
    local (s3, a3, _) = _bothWays("ArrayPath.FreeParam", ["x"])
    @test a3[1] ≈ 2 * exp(-1.0) rtol = 1e-5
    @test maximum(abs.(s3 .- a3)) < 1e-5
    #= a double exponential's free parameters (MSL LightningImpulse): the physical root, eta = 0.9511..., tau1 = 470 us,
       tau2 = 4.06 us, not tau1 = tau2 with eta = 0 (undamped Newton from the starts) =#
    local impulse = _arrayPathTranslate("ArrayPath.DoubleExpImpulse"; scalarized = false)
    @test _arrayPathTaken(impulse)
    local (eta, tau1, tau2) = (0.9511248490912196, 4.7010661173524463e-4, 4.0639516102066676e-6)
    local ys = _values(OMBackend.simulateModel(impulse; tspan = (0.0, 1e-3), reltol = 1e-10, abstol = 1e-12), ["y[1]", "y[2]", "x[1]"])
    @test ys ≈ [fill((exp(-1e-3 / tau1) - exp(-1e-3 / tau2)) / eta, 2); (tau1 * (1 - exp(-1e-3 / tau1)) - tau2 * (1 - exp(-1e-3 / tau2))) / eta] rtol = 1e-8
    local (s4, a4, _) = _bothWays("ArrayPath.InitAlgPulse", ["count", "T_start", "y"])
    @test maximum(abs.(s4 .- a4)) < 1e-6
    @test a4[1:2] ≈ [6.0, 0.9] atol = 1e-6
  end

  @testset "If-equations on variable conditions, asserts of initial equations" begin
    local (s, v, sol) = _bothWays("ArrayPath.IfDynamic", ["y", "z", "w"])
    @test maximum(abs.(s .- v)) < 1e-6
    @test [sol(t; idxs = :y) for t in (0.1, 0.3, 0.7)] == [2.0, 3.0, 1.0]
    local name = _arrayPathTranslate("ArrayPath.InitialAssert"; scalarized = false)
    @test _arrayPathTaken(name)
    local err = try
      OMBackend.simulateModel(name; tspan = (0.0, 1.0)); nothing
    catch e
      e
    end
    @test err isa OMBackend.CodeGeneration.ModelicaAssertionError && err.time == 0.0
    @test startswith(err.message, "k must be positive, k = -1")
  end

  @testset "Modelica functions" begin
    #= the model's functions as Julia functions of the module (the MTK path's code
       generation); (p, q) = f(x) as an algorithm node (the MTK path fails on it) =#
    local name = _arrayPathTranslate("ArrayPath.Functions"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    local x = exp(-1.0)
    @test _values(sol, ["x", "y", "z", "p", "q"]) ≈ [x, 2x, 3x + 1, x^2, -x] rtol = 1e-5
    #= v' = -2 v / |v|: the direction stays, |v| = sqrt(14) - 2t =#
    local r = sqrt(14.0) - 2.0
    @test _values(sol, ["v[1]", "v[3]", "n", "w[2]"]) ≈ [r / sqrt(14.0), 3r / sqrt(14.0), r, 4r / sqrt(14.0)] rtol = 1e-5
  end

  @testset "elsewhen, calls without a result, asserts in when bodies" begin
    local name = _arrayPathTranslate("ArrayPath.ElseWhen"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = @test_logs (:warn, r"n reached 11") match_mode = :any OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    #= x > 0.3: the first branch; x > 0.6 (the first condition still true): the second =#
    @test [sol(t; idxs = :n) for t in (0.2, 0.4, 0.7)] == [0.0, 1.0, 11.0]
    @test sol(1.0; idxs = :m) == 1.0
  end

    @testset "An assert's message and time" begin
    local name = _arrayPathTranslate("ArrayPath.AssertMessage"; scalarized = false)
    @test _arrayPathTaken(name)
    local err = try
      OMBackend.simulateModel(name; tspan = (0.0, 1.0)); nothing
    catch e
      e
    end
    @test err isa OMBackend.CodeGeneration.ModelicaAssertionError
    @test err.time ≈ 0.5 atol = 1e-6
    @test startswith(err.message, "x reached 0.5")
  end

  #= blockSolve! from a guess at the solution: a fresh Jacobian (its evaluations at perturbed points), its first step
     at round-off; the equations' a was left at the last color's perturbed point (1.5e-8 relative) =#
  @testset "A latch on pre(), a when-assigned Real, initial assignments" begin
    #= active = nmax > pre(n) (a cluster on change(pre(n))): n rises at 0 and 0.25, then active is false: x[i] = i / 4 =#
    local (s::Vector{Float64}, a::Vector{Float64}, _) = _bothWays("ArrayPath.PreLatch", ["n", "x[1]", "x[2]"])
    @test a ≈ [2.0, 0.25, 0.5] atol = 1e-6
    @test maximum(abs.(s .- a)) < 1e-6
    #= T0 = 0.1, 0.3, 0.5, 0.7, 0.9 from the samples, d = 3 + 5, p1 = 2, p2 = 3 (the parameters not in the results):
       x[i] = p2 t + i (the integral of T0) =#
    local (s2::Vector{Float64}, a2::Vector{Float64}, _) = _bothWays("ArrayPath.ImplicitDiscrete", ["T0", "d", "x[1]", "x[2]"])
    local held::Float64 = 0.1 * 0.2 + 0.3 * 0.2 + 0.5 * 0.2 + 0.7 * 0.2 + 0.9 * 0.1
    @test a2 ≈ [0.9, 8.0, 3.0 + held, 3.0 + 2 * held] atol = 1e-6
    @test maximum(abs.(s2 .- a2)) < 1e-6
  end

  @testset "A block solve's equations at its solution after a fresh Jacobian" begin
    local AG::Module = OMBackend.CodeGeneration.ArrayODEGen
    local SA::Module = OMBackend.CodeGeneration.Symbolics.SparseArrays
    local c::Vector{Float64} = [1.0, 2.0]
    local S::SA.SparseMatrixCSC{Float64, Int} = SA.sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
    local s::AG.BlockSolver = AG.BlockSolver(S, AG.columnColors(S), copy(c))
    local residual!(r, y, du, a, u, p, t, z, mode, fire) = (a .= y; r .= y .- c; nothing)
    local (du::Vector{Float64}, a::Vector{Float64}) = (zeros(2), zeros(2))
    AG.blockSolve!(s, residual!, du, a, Float64[], nothing, 0.0, nothing, false, false)
    @test s.jacobians == 1
    @test a == c
  end

  #= a solve from its own solution returns it: a loop whose residual at the solution is round-off of a term of 300
     (an ulp of 5.7e-14) against a slope of 1e-5, its Newton step 6e-9 of noise. Each solve stepped again, x moved
     between -6e-10 and 5e-9: a relation on it flipped in every pass of an event iteration (MSL
     ThyristorBridge2mPulse_DC_Drive's commutating thyristors) =#
  @testset "A block solve from its own solution keeps it" begin
    local AG::Module = OMBackend.CodeGeneration.ArrayODEGen
    local SA::Module = OMBackend.CodeGeneration.Symbolics.SparseArrays
    local S::SA.SparseMatrixCSC{Float64, Int} = SA.sparse(ones(1, 1))
    local s::AG.BlockSolver = AG.BlockSolver(S, AG.columnColors(S), [0.0])
    local residual!(r, y, du, a, u, p, t, z, mode, fire) = (a .= y; r[1] = (300.0 + 1.0e-5 * y[1]) - 300.0 - 3.0e-14;
                                                            nothing)
    local (du::Vector{Float64}, a::Vector{Float64}) = (zeros(1), zeros(1))
    local xs::Vector{Float64} = Float64[]
    for _ in 1:4
      AG.blockSolve!(s, residual!, du, a, Float64[], nothing, 0.0, nothing, false, false)
      push!(xs, a[1])
    end
    @test abs(xs[1]) < 1.0e-8
    @test all(==(xs[1]), xs)
    #= a guess far from the solution (a step of 1e-7, beyond 1e-8) whose residuals are no larger than at the last
       solution (a loop of a small scale, `rsolved` 1e-6 here) is solved from, not kept (MSL ThyristorBridge2mPulse_R:
       steps up to 7e14 kept, its mean values off) =#
    local target::Base.RefValue{Float64} = Base.RefValue{Float64}(0.0)
    local shifted!(r, y, du, a, u, p, t, z, mode, fire) = (a .= y; r[1] = y[1] - target[]; nothing)
    local f::AG.BlockSolver = AG.BlockSolver(S, AG.columnColors(S), [0.0])
    AG.blockSolve!(f, shifted!, du, a, Float64[], nothing, 0.0, nothing, false, false)
    f.rsolved = 1.0e-6
    target[] = 1.0e-7
    AG.blockSolve!(f, shifted!, du, a, Float64[], nothing, 0.0, nothing, false, false)
    @test a[1] == 1.0e-7
  end

  @testset "Affine in a loop's unknowns (_affineIn)" begin
    local D::Module = OMBackend.SimCode.Domains
    local (x::D.Ref, y::D.Ref, k::D.Ref) = (D.Ref("x", D.IR[], false), D.Ref("y", D.IR[], false), D.Ref("k", D.IR[], false))
    local reads::Function = (r::D.Ref) -> r.name in ("x", "y")
    @test D._affineIn(D.Op(:+, D.IR[D.Op(:*, D.IR[k, x]), D.Op(:neg, D.IR[y])]), reads) == (true, true)
    @test D._affineIn(D.Op(:*, D.IR[x, y]), reads) == (false, true)
    @test D._affineIn(D.Op(:/, D.IR[x, k]), reads) == (true, true)
    @test D._affineIn(D.Op(:/, D.IR[k, x]), reads) == (false, true)
    @test D._affineIn(D.Op(:sin, D.IR[k]), reads) == (true, false)
    @test D._affineIn(D.Op(:sin, D.IR[x]), reads) == (false, true)
    #= a relation holds its value during a solve: an if on it is affine where its branches are =#
    @test D._affineIn(D.Op(:if, D.IR[D.Rel(:>, x, D.Lit(0.0), 1), x, D.Op(:neg, D.IR[y])]), reads) == (true, true)
    #= an unknown as a subscript =#
    @test D._affineIn(D.Ref("k", D.IR[y], false), reads) == (false, true)
  end

  @testset "A linear block solved directly: one Jacobian per call" begin
    local AG::Module = OMBackend.CodeGeneration.ArrayODEGen
    local SA::Module = OMBackend.CodeGeneration.Symbolics.SparseArrays
    local S::SA.SparseMatrixCSC{Float64, Int} = SA.sparse(ones(2, 2))
    local s::AG.BlockSolver = AG.BlockSolver(S, AG.columnColors(S), [0.0, 0.0], true)
    local A::Base.RefValue{Matrix{Float64}} = Base.RefValue{Matrix{Float64}}(zeros(2, 2))
    local b::Vector{Float64} = [1.0, 2.0]
    local residual!(r, y, du, a, u, p, t, z, mode, fire) = (a .= y; r .= A[] * y .- b; nothing)
    local (du::Vector{Float64}, a::Vector{Float64}) = (zeros(2), zeros(2))
    #= a matrix that moves between calls (a mechanism's, with its angles) =#
    for M in ([2.0 1.0; 1.0 3.0], [2.5 1.0; 0.5 3.0], [1.0e-3 1.0; 1.0 2.0])
      A[] = M
      local before::Int = s.residuals
      AG.blockSolve!(s, residual!, du, a, Float64[], nothing, 0.0, nothing, false, false)
      @test a ≈ M \ b rtol = 1e-12
      #= the residuals at the guess, at the kept factors' step, two colors, at the step =#
      @test s.residuals - before <= 5
    end
    @test s.jacobians == 3
    #= a matrix that has not moved (a circuit between switchings): the kept factors' step, no Jacobian =#
    local before::Int = s.residuals
    AG.blockSolve!(s, residual!, du, a, Float64[], nothing, 0.0, nothing, false, false)
    @test s.residuals - before == 2
    @test s.jacobians == 3
  end

  #= a linear block whose matrix changes with its relations: kept factors only for the relations they were made at.
     The bridge's currents are zero at the start, its loop's solution zero to round-off; the kept factors' solution
     under each new set of relations set them again from round-off signs, and the start did not settle (MSL
     SMPM_Braking, its machine's EMFs into a diode bridge) =#
  @testset "A switched linear block: kept factors for their relations only" begin
    local bridge = Modelica"""
      model SwitchedBridge "three EMFs behind inductances into six ideal diodes onto a resistor"
        parameter Real Ron = 1e-4;
        parameter Real Goff = 1e-4;
        parameter Real L = 1e-3;
        parameter Real Rs = 0.05;
        parameter Real Rg = 1e6;
        parameter Real R = 1;
        parameter Real E = 100;
        parameter Real w = 314.1592653589793;
        Real i[3](each start = 0, each fixed = true);
        Real e[3];
        Real vs;
        Real vac[3](each start = 10);
        Real vp(start = 50);
        Real s[6](each start = 0.5);
        Real vd[6];
        Real id[6];
        Boolean off[6];
      equation
        for k in 1:3 loop
          e[k] = E * sin(w * time - 2 * 3.141592653589793 * (k - 1) / 3 + 0.3);
          L * der(i[k]) = e[k] + vs - vac[k] - Rs * i[k];
          vd[k] = vac[k] - vp;
          vd[3 + k] = -vac[k];
          i[k] = id[k] - id[3 + k];
        end for;
        sum(i) = -vs / Rg;
        for k in 1:6 loop
          off[k] = s[k] < 0;
          vd[k] = s[k] * (if off[k] then 1 else Ron);
          id[k] = s[k] * (if off[k] then Goff else 1);
        end for;
        id[1] + id[2] + id[3] = vp / R;
      end SwitchedBridge;
      """
    local (s::Vector{Float64}, a::Vector{Float64}, sol) = _bothWaysOf(bridge, "SwitchedBridge", ["i[1]", "vp"];
                                                                       tspan = (0.0, 0.04))
    @test only(sol.prob.p.blocks).linear
    @test maximum(abs.(s .- a) ./ (1.0 .+ abs.(s))) < 1e-4
  end

  @testset "A small linear loop solved directly" begin
    local loop = Modelica"""
      model LinearLoop "i1 = (x + 2 sin(time)) / 7"
        Real x(start = 1.0, fixed = true);
        Real i1;
        Real i2;
      equation
        i1 + 2.0 * i2 = x;
        3.0 * i1 - i2 = sin(time);
        der(x) = -i1;
      end LinearLoop;
      """
    local traced::String = withenv("OMBACKEND_ARRAY_TRACE" => "true") do
      mktemp() do path::String, io::IO
        redirect_stdio(stdout = io) do
          _translateProgram(loop, "LinearLoop"; scalarized = false)
        end
        flush(io)
        read(path, String)
      end
    end
    @test occursin("loop family 1: 2 unknowns, 1 members, linear", traced)
    local (ss, as, _) = _bothWaysOf(loop, "LinearLoop", ["x", "i1"]; reltol = 1e-10, abstol = 1e-12)
    #= x' + x/7 = -2/7 sin(t): x = 0.72 exp(-t/7) - 0.04 sin(t) + 0.28 cos(t) =#
    local x1::Float64 = 0.72 * exp(-1 / 7) - 0.04 * sin(1.0) + 0.28 * cos(1.0)
    @test as ≈ [x1, (x1 + 2 * sin(1.0)) / 7] rtol = 1e-8
    @test maximum(abs.(ss .- as)) < 1e-8
  end

  @testset "Time-invariant rows: in invariants! only, per solve; the callbacks' types erased" begin
    #= n and c read parameters only (SimCode Domains: invariant rows): invariants! computes them once per solve,
       equations! (the relation's event, the observed values) calls it and holds no copy. MSL EngineV6: such rows were
       28 % of equations!. =#
    local rows = Modelica"""
      model InvariantRows "x = exp(-c t), c = k n[3] = 2 k / 3; y = n[1] = 1/3 while x > 0.5, 2 n[2] = 4/3 after"
        parameter Real k = 2;
        parameter Real axis[3] = {1, 2, 2};
        Real n[3] = axis / sqrt(axis * axis);
        Real c = k * n[3];
        Real x(start = 1, fixed = true);
        Real y;
      equation
        der(x) = -c * x;
        y = if x > 0.5 then n[1] else 2.0 * n[2];
      end InvariantRows;
      """
    local (ss, as, _) = _bothWaysOf(rows, "InvariantRows", ["x", "y", "c", "n[3]"]; reltol = 1e-10, abstol = 1e-12)
    @test as ≈ [exp(-4 / 3), 4 / 3, 4 / 3, 2 / 3] rtol = 1e-8
    @test maximum(abs.(ss .- as)) < 1e-8
    local name::String = OMBackend.canonicalName("InvariantRows")
    #= n's row once in the module: in invariants!, not in equations! =#
    @test count("sqrt(", string(OMBackend.COMPILED_MODELS_DEJL[name][1])) == 1
    #= another solve, another k: its own invariant values (x(1) = exp(-2/3) > 0.5: y stays n[1]) =#
    local again = OMBackend.resimulateModel(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-12,
                                            parameters = Dict{String, Any}("k" => 1.0))
    @test _values(again, ["x", "y", "c"]) ≈ [exp(-2 / 3), 1 / 3, 2 / 3] rtol = 1e-8
    #= the solver's callbacks: conditions and affects behind one wrapper type (`_erasedCallbacks`), nothing of the
       model in their types, so the solver's callback code is compiled once for every model =#
    local (_, cbs) = OMBackend.getMTKProblem(name)
    local callbacks::Vector{Any} = Any[cbs.continuous_callbacks..., cbs.discrete_callbacks...]
    @test !isempty(callbacks)
    @test all(c -> c.condition isa OMBackend.CodeGeneration._ErasedCall && c.affect! isa OMBackend.CodeGeneration._ErasedCall,
              callbacks)
    @test !any(c -> occursin(name, string(typeof(c))), callbacks)
  end

  @testset "The time derivative's step does not grow with time" begin
    #= tgrad! by central differences over cbrt(eps) |t|: at t = 29 a step of 1.8e-4 s, 5e-4 off on 50 Hz; a stiff
       state driven by it took 5695 steps in the last second against 1905 in the first (MSL Electrical
       SMEE_Generator: 618938 steps for 30 s, now 1910) =#
    local hum = Modelica"""
      model StiffHum "a stiff filter of a 50 Hz source over 30 s"
        Real x(start = 0.0, fixed = true);
      equation
        der(x) = 1000.0 * (sin(314.1592653589793 * time) - x);
      end StiffHum;
      """
    local name::String = _translateProgram(hum, "StiffHum"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 30.0), reltol = 1e-6)
    @test sol.retcode == ReturnCode.Success
    @test count(t::Float64 -> t > 29.0, sol.t) < 1.2 * count(t::Float64 -> t < 1.0, sol.t)
  end

  @testset "A time relation an ulp after a sample tick: an event of its own" begin
    #= MSL SawTooth's reset, `time >= (pre(count) + 1) * period`, at 6 * 0.1 = 0.6000000000000001; the tick of
       sample(0.1, 0.1) at 0.1 + 5 * 0.1 = 0.6. The tick took the relation false; its crossing an ulp later was dropped
       as part of the tick, and the reset waited for the end of the step after it (MSL ChopperBuckBoost_DutyCycle's
       switching 1e-7 s late every other period, through the sawtooth) =#
    local saw = Modelica"""
      model SawToothBehindTick "T_start = 0.6 from the reset at 0.6000000000000001"
        parameter Real period = 0.1;
        discrete Real T_start(start = 0, fixed = true);
        discrete Integer count(start = 0, fixed = true);
        discrete Real ticks(start = 0, fixed = true);
        Real x(start = 0, fixed = true);
      equation
        when time >= (pre(count) + 1) * period then
          count = pre(count) + 1;
          T_start = time;
        end when;
        when sample(period, period) then
          ticks = pre(ticks) + 1;
        end when;
        der(x) = (time - T_start) / period;
      end SawToothBehindTick;
      """
    local (s::Vector{Float64}, a::Vector{Float64}, _) =
      _bothWaysOf(saw, "SawToothBehindTick", ["T_start", "count", "ticks"]; tspan = (0.0, 0.65))
    @test a[2:3] == [6.0, 6.0]
    @test abs(a[1] - 0.6) < 1.0e-12
    @test maximum(abs.(s .- a)) < 1.0e-12
  end

  @testset "Algebraic loops: per element by Newton, through the elements as a DAE" begin
    local tight::NamedTuple = (reltol = 1e-10, abstol = 1e-10)
    #= an equation nonlinear in its unknown; a nonlinear loop of two per element =#
    local (s::Vector{Float64}, a::Vector{Float64}, _) = _bothWays("ArrayPath.Cubic", ["x[1]", "y[2]", "x[3]"]; tight...)
    @test isapprox(a, s; atol = 1e-9)
    (s, a, _) = _bothWays("ArrayPath.LocalLoops", ["x[1]", "y[2]", "w[4]"]; tight...)
    @test isapprox(a, s; atol = 1e-9)
    #= one solve for the family of blocks: the code does not grow with n =#
    local batches::Int = OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches
    @test _arrayPathTaken(_arrayPathTranslate("ArrayPath.LocalLoops1000"; scalarized = false))
    @test OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].batches == batches
    #= a tridiagonal loop through all the elements: solved in the right-hand side (blockSolve!), the solver sees the
       states only =#
    local (s2::Vector{Float64}, a2::Vector{Float64}, sol::OMBackend.DifferentialEquations.SciMLBase.AbstractODESolution) = _bothWays("ArrayPath.CoupledLoop", ["x[1]", "v[2]", "x[20]", "v[10]"]; tight...)
    @test isapprox(a2, s2; atol = 1e-9)
    @test length(sol.u[1]) == 20
    #= with an event in the loop (its unknowns solved again in the event iteration), a when-equation, an assert and an
       initial equation (x[N] free) =#
    local (s3::Vector{Float64}, a3::Vector{Float64}, _) = _bothWays("ArrayPath.CoupledLoopEvents", ["x[1]", "v[2]", "x[20]", "v[10]", "crossings"]; tight...)
    @test isapprox(a3, s3; atol = 1e-8)
    @test a3[5] == 1.0
    #= nonlinear (the scalarized model's initialization fails here): its residuals at the end =#
    local name::String = _arrayPathTranslate("ArrayPath.CoupledNonlinear"; scalarized = false)
    @test _arrayPathTaken(name)
    sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0), tight...)
    @test sol.retcode == ReturnCode.Success
    local val::Function = (n::String) -> OMBackend.getVariableValues(sol, n)[end]
    local (x::Vector{Float64}, v::Vector{Float64}, w::Vector{Float64}) = ([val("x[$i]") for i in 1:20], [val("v[$i]") for i in 1:20], [val("w[$i]") for i in 1:20])
    local residuals::Vector{Float64} = [2v[1] - v[2] + 0.1v[1]^3 - x[1]; [-v[i - 1] + 3v[i] - v[i + 1] + 0.1v[i]^3 - x[i] for i in 2:19];
                       -v[19] + 2v[20] + 0.1v[20]^3 - x[20]]
    @test maximum(abs, residuals) < 1e-10
    @test w ≈ v .- x atol = 1e-12
  end

  @testset "Models of scalars: algebraic loops, index above one, a clock; a loop through a relation" begin
    local tight::NamedTuple = (reltol = 1e-9, abstol = 1e-9)
    #= a model without arrays is the array code's (its variables arrays of no dimensions): a loop of scalars
       (y = 0.75 x, z = 0.25 x, x' = -0.75 x), a nonlinear one, an index of two reduced in the array form
       ((m1 + m2) x1'' = -k x1). Declined before by policy ("algebraic loop in a model without arrays", "no
       for-equations or array equations over continuous arrays, of index above one": 69 MSL 4.1.0 models) =#
    local (s, a, _) = _bothWays("ArrayPath.AlgebraicLoop", ["x", "y", "z"]; tight...)
    @test a ≈ exp(-0.75) .* [1.0, 0.75, 0.25] atol = 1e-7
    @test maximum(abs.(s .- a)) < 1e-7
    local (sn, an, _) = _bothWays("ArrayPath.ScalarLoop", ["x", "y", "z"]; tight...)
    @test an[2] + an[3]^3 ≈ an[1] atol = 1e-9
    @test an[2] - an[3] ≈ 0.5 * an[1] atol = 1e-9
    @test maximum(abs.(sn .- an)) < 1e-7
    local (s2, a2, _) = _bothWays("ArrayPath.ScalarIndexTwo", ["x1", "x2"]; tight...)
    @test a2 ≈ [cos(sqrt(10 / 3)), cos(sqrt(10 / 3))] atol = 1e-7
    @test maximum(abs.(s2 .- a2)) < 1e-7
    #= a clocked partition of index 1 (lowered to a when-equation on sample()): ticks at 0, 0.1, 0.2 before 0.25.
       Declined before ("index 1 with clocked partitions": 57 MSL 4.1.0 models) =#
    local (sc, ac, _) = _bothWays("ArrayPath.ClockedSampleHold", ["xs", "xd", "y"]; tspan = (0.0, 0.25), tight...)
    @test ac ≈ [exp(-0.2), exp(-0.1), exp(-0.1)] atol = 1e-7
    @test maximum(abs.(sc .- ac)) < 1e-7
    #= a loop through a relation (events in a loop), scalarized before ("relations in an algebraic loop"): y = 0.75 x
       above x = 0.5, 0.6 x below; x = 0.5 at t1 = ln(2) / 0.75, x(1) = 0.5 exp(-0.6 (1 - t1)) =#
    local (se, ae, _) = _bothWays("ArrayPath.LoopWithEvent", ["x[1]", "y[1]"]; tight...)
    @test ae ≈ [1.0, 0.6] .* (0.5 * exp(-0.6 * (1 - log(2) / 0.75))) atol = 1e-6
    @test maximum(abs.(se .- ae)) < 1e-6
  end

  @testset "Complex values: record equations in for bodies, array equations of records" begin
    #= a record equation in a for body is the equations of its fields (`bodyRecordEquations`), a Complex one its real
       and imaginary parts with the operator calls taken apart (`complexParts`); an array equation of Complex values
       a for-equation of the parts (`elementLoop`), by its fields' arrays (`recordArrayEquations`) or an equation per
       element (`recordElementEquations`). The array form of index reduction refused each ("a RECORD_EQUATION with
       arrays kept", 49 MSL 4.1.0 models); `(-i0).re` in a branch of an if-expression failed on every route =#
    local tight::NamedTuple = (reltol = 1e-9, abstol = 1e-9)
    for (model, name, expected) in (("ArrayPath.RecordForBody", "x[1]", exp(-1.5)),
                                    ("ArrayPath.RecordForSum", "x[2]", exp(-2.0)),
                                    ("ArrayPath.RecordForIf", "x", exp(-1.0)),
                                    ("ArrayPath.RecordComponents", "x", exp(-1.75)),
                                    ("ArrayPath.ComplexArrays", "x[3]", exp(-1.0)))
      local (s, a, _) = _bothWays(model, [name]; tight...)
      @test a[1] ≈ expected atol = 1e-7
      @test s[1] ≈ expected atol = 1e-7
    end
    #= the parts of Complex operations, and the projections =#
    local I::Module = OMBackend.OMJLIR
    local complex::I.ModelicaType = I.T_RECORD(:Complex, I.Field[I.Field(:re, I.T_REAL()), I.Field(:im, I.T_REAL())])
    local z::I.Expression = I.CREF(I.ComponentRef(:z, complex), complex)
    local negated::I.Expression = I.CALL(Symbol("Complex.'-'.negate"), I.Expression[z], complex, I.FUNCTION_CALL)
    local parts = OMBackend.Passes.complexParts(negated)
    @test parts !== nothing && I.toString(parts[1]) == "(-z.re)" && I.toString(parts[2]) == "(-z.im)"
    local magnitude::I.Expression =
      I.CALL(Symbol("Modelica.ComplexMath.'abs'"), I.Expression[z], I.T_REAL(), I.FUNCTION_CALL)
    @test I.toString(OMBackend.Passes.complexProjection(magnitude)) == "(((z.re ^ 2.0) + (z.im ^ 2.0)) ^ 0.5)"
    @test OMBackend.Passes.complexParts(I.CALL(:f, I.Expression[z], complex, I.FUNCTION_CALL)) === nothing
  end

  @testset "Derivatives solved with the value of another put in; the sum of an element-wise power" begin
    #= der(is) and der(psi) read each other: the element view solves one from its equation with the other's value put
       in (`solveThroughSolved!`), the array form from the body with the other's template put in
       (`withTemplateValues`). Refused before ("derivatives solved through the values of others": 57 MSL 4.1.0 models
       behind it, the Machines among them) =#
    local tight::NamedTuple = (reltol = 1e-9, abstol = 1e-9)
    local a::Float64 = 2 / 1.1
    local (s, v, _) = _bothWays("ArrayPath.DerivativeTwoCycle", ["is[1]", "psi[2]"]; tight...)
    @test v[1] ≈ (a * sin(1.0) - cos(1.0) + exp(-a)) / (1.1 * (1 + a^2)) atol = 1e-7
    @test maximum(abs.(s .- v)) < 1e-7
    #= sum(i .^ 2 / m): an element of an element-wise power is the element's power (`scalarElement`) =#
    local (sp, vp, _) = _bothWays("ArrayPath.ElementwisePowerSum", ["x"]; tight...)
    @test vp[1] ≈ exp(-sqrt(14 / 3)) atol = 1e-7
    @test sp[1] ≈ vp[1] atol = 1e-7
  end

  @testset "Rolled components: reinit of their states; initial equations that assign discrete elements, pre()" begin
    #= two instances of MSL's Mean rolled into arrays: a reinit's state is a reference `IR.mapEquation!` keeps, the
       rolling rewrote the expressions only ("unknown variable meanCurrent.x", `rolledReinits`); their initial
       equations `y = y0` are a for-equation over the instances that no residual solves ("initialization
       over-determined": unrolled, `rollLeftArrayEquations!`) =#
    local (s, a, _) = _bothWays("ArrayPath.RolledReinit", ["m1.y", "m2.y", "m1.x", "m2.x"]; tspan = (0.0, 0.95),
                                reltol = 1e-9, abstol = 1e-9)
    @test a ≈ [1.0, 0.85, 0.15, 0.04625] atol = 1e-7
    @test maximum(abs.(s .- a)) < 1e-7
    #= `pre(e) = 0.5` gives a when-assigned variable its value before the first event (`_addInitialEquations!`), of a
       component array's elements too (MSL Timer's `pre(entryTime) = 0`) =#
    for (model, names) in (("ArrayPath.PreStartCell", ["e"]), ("ArrayPath.PreStarts", ["c[1].e", "c[2].e"]))
      local (sp, ap, _) = _bothWays(model, names)
      @test all(==(1.5), ap) && sp == ap
    end
    #= `pre(y) = y0` of a variable an equation computes is pre() at the initialization (`SPreStart`), not an
       assignment of y: the equation gives y from it (MSL Hysteresis, MultiSwitch) =#
    local (sh, ah, _) = _bothWays("ArrayPath.PreStartParameter", ["x[1]", "x[2]"])
    @test isapprox(ah[1], 1.5; atol = 1e-6) && isapprox(ah[2], -0.5; atol = 1e-6)
    @test maximum(abs.(sh .- ah)) < 1e-6
  end

  @testset "External objects: a table read by the equations and by a parameter's binding" begin
    #= the object is a constant of the model's module, made once per build by its constructor's call on the
       parameters' translated values (`_objectRead`, `_objectDefinitions`); a function takes it as an argument (it
       was "a record argument") =#
    local (st, at, _) = _bothWays("ArrayPath.TableDrive", ["x[1]", "x[2]", "z"])
    @test isapprox(at, [1.0, 2.0, 1.0]; atol = 1e-6)
    @test maximum(abs.(st .- at)) < 1e-6
    #= components that hold external objects are not rolled into arrays: rolled, the array form refused the array of
       their objects and the model left the array code =#
    local (sr, ar, _) = _bothWays("ArrayPath.RolledTables", ["a.x", "b.x"])
    @test isapprox(ar, [1.0, 2.0]; atol = 1e-6)
    @test maximum(abs.(sr .- ar)) < 1e-6
  end

  @testset "A model without continuous unknowns: its parameters in the solution" begin
    #= no equation reads J or k: on array code they were not in the parameter layout and not in the solution (MSL
       readRealParameterModel: J, phi_rel0, w_rel0 read from a file, "not in the solution"); the scalar route keeps
       them where the model has no continuous equations (`eliminateDeadParameters`) =#
    local unread = Modelica"""
      model UnreadParameters
        parameter Real J = 2;
        parameter Real k = 3 * J;
        discrete Integer n(start = 0, fixed = true);
      equation
        when initial() then
          n = 1;
        end when;
      end UnreadParameters;
      """
    local name::String = _translateProgram(unread, "UnreadParameters"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    @test _values(sol, ["J", "k"]) == [2.0, 6.0]
  end

  @testset "Rolled components the plan treats differently: taken as declared" begin
    #= three inertias rolled into arrays, a gear between the first two: one of them is no state, and the plan split
       most of the rolled arrays ("slicing that splits most of the arrays"); the array form refused the model, and the
       backend's scalarization of the rolled form did not build (OMBackend's precompile workload Drive) =#
    local geared = Modelica"""
      model GearedInertias
        connector Flange
          Real phi;
          flow Real tau;
        end Flange;
        model Inertia
          parameter Real J = 1;
          Flange a;
          Flange b;
          Real phi(start = 0, fixed = true);
          Real w(start = 0, fixed = true);
        equation
          phi = a.phi;
          phi = b.phi;
          der(phi) = w;
          J * der(w) = a.tau + b.tau;
        end Inertia;
        model IdealGear
          parameter Real ratio = 2;
          Flange a;
          Flange b;
        equation
          a.phi = ratio * b.phi;
          0 = ratio * a.tau + b.tau;
        end IdealGear;
        model Spring
          parameter Real c = 100;
          Flange a;
          Flange b;
        equation
          b.tau = c * (b.phi - a.phi);
          a.tau = -b.tau;
        end Spring;
        Inertia j1(J = 0.1);
        IdealGear gear;
        Inertia j2(J = 2, phi(fixed = false), w(fixed = false));
        Spring spring;
        Inertia j3(phi(start = 0.1));
      equation
        connect(j1.b, gear.a);
        connect(gear.b, j2.a);
        connect(j2.b, spring.a);
        connect(spring.b, j3.a);
      end GearedInertias;
      """
    local (sg::Vector{Float64}, ag::Vector{Float64}, _) =
      _bothWaysOf(geared, "GearedInertias", ["j1.phi", "j2.w", "j3.phi"]; reltol = 1e-8, abstol = 1e-8)
    @test maximum(abs.(sg .- ag)) < 1e-6
  end

  @testset "The first step where OrdinaryDiffEq's probe for it leaves the right-hand side's domain" begin
    #= its estimate was NaN and the run ended at the start (DtNaN): `_finiteFirstStep!` =#
    local (so, ao, _) = _bothWays("ArrayPath.OverflowingProbe", ["v"])
    @test isapprox(ao, [0.025 * log(1e3 / 0.9e-14)]; rtol = 1e-6)
    @test maximum(abs.(so .- ao)) < 1e-6
  end

  @testset "Small loops: a solve that does not converge is made again, damped, from the last solution and the starts" begin
    #= from the solution on the saturated piece Newton's method cycles between two points that are no solutions; the
       last iterate went into the equations as the solution (MSL OpAmps.VoltageFollower: +15 V for -5 V) =#
    local (ss, as, _) = _bothWays("ArrayPath.SaturatedLoop", ["x", "y"])
    @test isapprox(as, [7.5 + 0.5 * 1000 / 1001, 1000 / 1001]; atol = 1e-5)
    @test maximum(abs.(ss .- as)) < 1e-5
    #= a Jacobian singular at the start values (a product of two unknowns, both 0 there): the least-squares step of
       the smallest norm in the damped attempts; it was NaN (MSL Analog.Examples.Resistor: its assert failed at 0) =#
    local (sb, ab, _) = _bothWays("ArrayPath.BilinearStart", ["x", "R"])
    @test isapprox(ab[1], sin(1) / 10; rtol = 1e-4)
    @test maximum(abs.(sb .- ab)) < 1e-6
    #= a loop that divides by one of its unknowns is torn (`_dividesByBlockUnknowns`): the generated Newton started vt
       at 0 (no start value), exp(v / vt) was NaN (MSL HeatingRectifier's diode, heated by its own losses) =#
    local heated = Modelica"""
model DivideInLoop
  parameter Real R = 10, Gth = 0.1;
  Real T(start = 300, fixed = true);
  Real Tp, vt, i, v, Q;
equation
  der(T) = Q - 0.1 * (T - 300);
  Tp = T + Q / Gth;
  Q = v * i;
  vt = 8.617e-5 * Tp;
  i = 1e-9 * (exp(v / vt) - 1);
  1 = v + R * i;
end DivideInLoop;
"""
    local (sd, ad, _) = _bothWaysOf(heated, "DivideInLoop", ["v", "vt", "T"])
    @test all(isfinite, ad) && maximum(abs.(sd .- ad)) < 1e-8
  end

  @testset "A parameter known as zero: its terms out of the equations, a state whose derivative they held is none" begin
    #= the plan's structure reads the equations with the known values (`withKnownValues`); the committed model kept
       `Nim * der(Phiim)` and with it Phiim as a state and der(Phiim) as an unknown whose only coefficient is zero:
       structurally singular in the array code (MSL FundamentalWave SinglePhaseInductance). `dropKnownZeroTerms!` =#
    local (sz, az, _) = _bothWays("ArrayPath.ZeroOrientation", ["Phire", "Phiim"])
    @test isapprox(az, [exp(-1), 0.0]; atol = 1e-6)
    @test maximum(abs.(sz .- az)) < 1e-6
    #= a zero of one element only (three converters at 0, 2 pi / 3, 4 pi / 3: `Nim[1] = 0`): der(Phiim[1]) leaves
       the first iteration, the arrays are split and their equations unrolled (`derivativesZeroedInArrays`), Phiim[1]
       is no state. The scalarized route does not build this model (a structural imbalance of its reduced system) =#
    local phases::String = _arrayPathTranslate("ArrayPath.ZeroOrientationPhases"; scalarized = false)
    @test _arrayPathTaken(phases)
    local solPhases = OMBackend.simulateModel(phases; tspan = (0.0, 1.0))
    @test solPhases.retcode == ReturnCode.Success
    local angles::Vector{Float64} = [0.0, 2pi / 3, 4pi / 3]
    @test isapprox(_values(solPhases, ["Phire[1]", "Phire[2]", "Phire[3]"]), cos.(angles) .* exp(-1); atol = 1e-6)
    @test isapprox(_values(solPhases, ["Phiim[1]", "Phiim[2]", "Phiim[3]"]), sin.(angles) .* exp(-1); atol = 1e-6)
    #= a variable an equation gives as zero once the parameter's zero is out (`Vm = 0` of a core without
       conductance) takes der(PhiCore) out of the loss, its last read: PhiCore is no state. Left a state beside the
       winding's flux it equals, the array code had two states tied by an equation (the MSL machines: structurally
       singular) =#
    local (sc, ac, _) = _bothWays("ArrayPath.ZeroConductanceCore", ["Phi", "PhiCore", "Vm", "loss"])
    @test isapprox(ac, [(1 - cos(1.0)) / 2, (1 - cos(1.0)) / 2, 0.0, 0.0]; atol = 1e-6)
    @test maximum(abs.(sc .- ac)) < 1e-6
  end

  @testset "An array of records set by an array of calls: an equation per field of each element" begin
    #= `states = {flowState(...), flowState(...)}` was one equation per element, of the record: one row for the
       element's two fields (MSL Fluid's flow models: fewer rows than unknowns). `recordElementEquations` =#
    local (sr, ar, _) = _bothWays("ArrayPath.RecordArrayElements", ["x", "y", "states[1].p", "states[2].T"])
    @test isapprox(ar, [exp(-1), 5 * exp(-1), exp(-1), 4 * exp(-1)]; atol = 1e-6)
    @test maximum(abs.(sr .- ar)) < 1e-6
  end

  @testset "Rolled components: equalities of discrete variables as written, a parameter of the sibling's" begin
    #= the two bodies are rolled into arrays over the instances. The equalities of a body's Integer colors
       (`cylinderColor = sphereColor`, `sphere.color = sphereColor`) were chained by the names with the Real ones,
       `cylinderColor = sphere.color`: an Integer given by a Real, a row without an unknown (MSL MultiBody
       PointGravity: 880 rows over 874 unknowns). `equalityPair` takes equalities of Reals only. The masses,
       `body2.m = body1.m`, are the array `m = {2, m[1]}`: its second element reads its first
       (`_selfReadsResolved`; was "cyclic parameter bindings", MSL MultiBody's Constraints examples) =#
    local (sb, ab, _) = _bothWays("ArrayPath.RolledBodies",
                                  ["body1.v", "body2.v", "body1.sphere.brightness", "body2.cylinder.brightness"])
    @test isapprox(ab, [exp(-0.5), exp(-0.5), 383.0, 383.0]; atol = 1e-6)
    @test maximum(abs.(sb .- ab)) < 1e-6
  end

  @testset "A discrete unknown in an algebraic loop: the loop cut at it, the event iteration settles it" begin
    #= an ideal diode's `off = s < 0` is in the loop of s, v and i. Its equation reads the continuous variables
       through a relation only, so it is constant between events: the loop is solved for the continuous unknowns with
       its last value, its own instance after the loop, and the event iteration's sweeps repeat until it stays
       (`_scheduleWithLoops`). The matching leaves `off` to that equation (`_elements`: an equation of discrete
       time is not solved for a continuous unknown) =#
    local (sd, ad, _) = _bothWays("ArrayPath.IdealDiodeLoop", ["q"])
    @test isapprox(ad[1], (1 - 1e-5) / (pi * 1.00001); atol = 1e-6)
    @test abs(sd[1] - ad[1]) < 1e-6
    #= three thyristors to one load: the loop of all is a large block, cut at each `off`; a commutation passes
       through both conducting =#
    local (st, at, _) = _bothWays("ArrayPath.ThyristorCommutation", ["q"])
    @test isapprox(at[1], 3 * sqrt(3) / (2 * pi) * cosd(20); atol = 1e-3)
    @test abs(st[1] - at[1]) < 1e-4
  end

  @testset "Lifted clusters of discrete equations: their initialized values, a mixed system's search" begin
    #= the cluster's initial when-equation (initial() true) gives its variables their initialized values; where the
       cluster runs in the start's iteration pre() of them is that value, not the start (`initialAlgorithms!`): MSL
       Rotational.Examples.Friction's clutch at a relative speed of -10 was Stuck, its mode's start Unknown =#
    local (si, ai, _) = _bothWays("ArrayPath.InitialCluster", ["x"]; tspan = (0.0, 0.5))
    @test isapprox(ai[1], -0.5; atol = 1e-6) && isapprox(si[1], -0.5; atol = 1e-6)
    #= a friction element's equations: sa, a and f in a loop with startForward and locked =#
    local (ss, as, solStick) = _bothWays("ArrayPath.StickSlip", ["v"]; tspan = (0.0, 2.0))
    @test isapprox(as[1], 2 * (cos(asin(0.75)) - cos(2)) - (2 - asin(0.75)); atol = 1e-5)
    @test abs(ss[1] - as[1]) < 1e-3
    #= relations that cycle in the event iteration's sweeps (MSL Rotational SimpleGearShift at its shift: "the event
       iteration did not settle"): the combination that gives itself back is searched for, fewest changes first
       (`_searchRelations!`). Here two relations: (true, false) and (false, true) give each other, (false, false)
       gives (true, false), (true, true) itself =#
    local generation::Module = OMBackend.CodeGeneration.ArrayODEGen
    local p = solStick.prob.p
    fill!(p.rel, false)
    p.rel[1] = true
    local flips::BitVector = falses(length(p.rel))
    flips[1:2] .= true
    local next = Dict((true, true) => (true, true), (true, false) => (false, true), (false, true) => (true, false),
                      (false, false) => (true, false))
    local equations! = (du, a, u, q, t, z, mode, fire) -> begin
      if mode
        local r::Tuple{Bool, Bool} = next[(a[1] >= 2.0, isodd(round(Int, a[1])))]
        q.rel[1] = r[1]
        q.rel[2] = r[2]
      end
      a[1] = 2 * q.rel[1] + q.rel[2]
      nothing
    end
    local whens! = (du, a, u, q, t, z, mode, fire) -> nothing
    local a::Vector{Float64} = zeros(length(p.abuf))
    a[1] = 2.0
    #= each relation its own instance: no other relations to search with =#
    @test generation._searchRelations!(equations!, whens!, zeros(1), a, zeros(1), p, 0.0, true, copy(p.rel), copy(p.d),
                                       flips, collect(1:length(p.rel)), trues(length(p.rel)))
    @test p.rel[1] && p.rel[2]
  end

  @testset "If-equations whose branches set different variables: a residual per equation" begin
    #= `if cv then u = Vend else -i = -I`: the branches assign different variables, one by a left side that is no
       reference. Each equation of the if-equation is a class `0 = if c then l1 - r1 else l2 - r2` (`_ifResidual`),
       here in a loop with the circuit's equation; it was refused ("branches give different variables": MSL
       CCCV_Cell, SwitchWithArc, DemoPowerSupply) =#
    local (sr, ar, _) = _bothWays("ArrayPath.IfBranchResiduals", ["vc[1]", "vc[2]"]; tspan = (0.0, 3.0))
    @test isapprox(ar, [2 - exp(-2), 3 - exp(-1)]; atol = 1e-5)
    @test maximum(abs.(sr .- ar)) < 1e-5
  end

  @testset "Vectors of records as function arguments, a reduction over an array constructor" begin
    #= a vector of Complex values has no variable of its own: its fields' vectors are the variables, and a generated
       function takes it by those (`_recordVectorFields`, the elements' fields of an array literal of them). The
       function's own body decides the value (MSL 4's Complex scalar product conjugates its first argument): y = 2 x.
       It was "unknown variable k" (MSL QuasiStatic power and RMS sensors) =#
    local (sv, av, _) = _bothWays("ArrayPath.ComplexVectorArguments", ["y.re", "y.im", "r"])
    @test isapprox(av, [2 * exp(-1), 0.0, exp(-1)]; atol = 1e-6)
    @test maximum(abs.(sv .- av)) < 1e-9
  end

  @testset "Clocked sample(): the left limit at a tick, a hold sampled back into its partition" begin
    #= `sample(u, clock)` takes u's left limit (MLS 16.5.1): pre() of u, of an algebraic variable its value before the
       event's sweep (`p.aleft`); a relation's change at a sample instant belongs to the tick's event
       (`_atSampleInstant`), and the after-step probe leaves the discrete values as they were (`relationsChanged`).
       u and b step at 0.2, a tick: sampled at the tick at 0.3; n counts the ticks through a hold (MSL Clocked
       Sample1, Sample2, Sample3: 999x off omc, the event iteration not settling). The scalarized route reads the
       sampled variable itself (`IR.SAMPLED_READ`): pre() of an Integer the equations compute had no slot in its
       sample callback (MSL Clocked IntegerSignals.Hold: KeyError) =#
    for (stop, expected) in ((0.25, [0.0, 0.0, 3.0, 2.0]), (0.35, [1.0, 1.0, 4.0, 3.0]))
      local (sl, al, _) = _bothWays("ArrayPath.ClockedLeftLimit", ["y", "by", "nh", "kh"]; tspan = (0.0, stop))
      @test al == expected
      @test sl == expected
    end
  end

  @testset "A large loop that is linear and badly conditioned: differences over a step of the unknowns' size" begin
    #= forward differences carry the residuals' round-off into the Jacobian, 1e-8 of it; at a condition of 1e10
       Newton's method does not contract with it and the loop's unknowns were NaN (MSL ThyristorBridge2mPulse_R: both
       thyristors of a commutating pair conducting for a pass of the event iteration; DtNaN). `blockSolve!` then
       takes differences over a step of the unknowns' size, exact for a linear loop =#
    local generation::Module = OMBackend.CodeGeneration.ArrayODEGen
    local A::Matrix{Float64} = [1.0 1.0; 1.0 1.0 + 1e-10]
    local b::Vector{Float64} = A * [1.0, 2.0]
    local solver = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [0.0, 0.0])
    local residual! = (res, y, du, a, u, p, t, z, mode, fire) -> (res .= A * y .- b; a .= y; nothing)
    generation.blockSolve!(solver, residual!, zeros(1), zeros(2), zeros(1), nothing, 0.0, nothing, false, false)
    @test solver.failures == 0
    @test isapprox(solver.x, [1.0, 2.0]; rtol = 1e-5)
  end

  @testset "A large loop whose Newton steps overshoot, or whose residuals throw" begin
    #= the root of a difference at zero (a pipe's flow at equal pressures): each full Newton step lands as far on
       the other side, and the loop's unknowns were NaN (MSL Fluid EmptyTanks did not start). `blockSolve!` then
       shortens the steps until the residuals fall (`_blockDamped!`) =#
    local generation::Module = OMBackend.CodeGeneration.ArrayODEGen
    local root = (x::Float64) -> sign(x) * sqrt(abs(x))
    local residual! = (res, y, du, a, u, p, t, z, mode, fire) ->
      (res[1] = root(y[1] - y[2]); res[2] = y[1] + y[2] - 3.0; a .= y; nothing)
    local arguments::Tuple = (zeros(1), zeros(2), zeros(1), nothing, 0.0, nothing, false, false)
    local plain = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [3.0, 0.0])
    plain.x .= plain.y
    @test !generation._blockAttempt!(plain, residual!, arguments, :full)
    local solver = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [3.0, 0.0])
    generation.blockSolve!(solver, residual!, zeros(1), zeros(2), zeros(1), nothing, 0.0, nothing, false, false)
    @test solver.failures == 0
    @test isapprox(solver.x, [1.5, 1.5]; atol = 1e-8)
    #= residuals that throw at every iterate (the cosine of an infinite angle; from the varied start values too): a
       failed solve, the unknowns NaN, no exception (thrown on from the asserts' check after a step it ended MSL
       MultiBody UniversalConstraint's run) =#
    local throwing! = (res, y, du, a, u, p, t, z, mode, fire) ->
      (res[1] = cos(1.0 / (y[1] - y[1])); res[2] = y[2]; nothing)
    local thrown = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [1.0, 1.0])
    generation.blockSolve!(thrown, throwing!, zeros(1), zeros(2), zeros(1), nothing, 0.0, nothing, false, false)
    @test thrown.failures == 1
    @test all(isnan, thrown.x)
    #= start values at which a residual is stationary in its unknown (a MOSFET's current, even in the drain-source
       voltage at zero: the Jacobian's column is zero, the least-norm step is none; MSL Spice3 Nand did not start):
       solved from the start values varied by 0.1, as OpenModelica varies them (the positive root) =#
    local even! = (res, y, du, a, u, p, t, z, mode, fire) ->
      (res[1] = 1.0e-12 * y[2]; res[2] = 1.5e-5 * (y[1]^2 - 25.0); a .= y; nothing)
    local still = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [0.0, 0.0])
    still.x .= still.y
    @test !(generation._blockAttempt!(still, even!, arguments, :full) && !still.rankDeficient)
    local varied = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [0.0, 0.0])
    generation.blockSolve!(varied, even!, zeros(1), zeros(2), zeros(1), nothing, 0.0, nothing, false, false)
    @test varied.failures == 0
    @test isapprox(varied.x, [5.0, 0.0]; atol = 1e-8)
  end

  @testset "A large loop through a limiter of high gain: differences over the unknowns' own size" begin
    #= a current of 1.6 mA into a limiter of gain 1e9 (an op-amp's V0 times a resistance), its solution 0.1 below
       the limit of 10: the standard difference's step, sqrt(eps) of at least 1, moves the limiter's input by 15 and
       crosses the limit, its derivative 2/3 of the true one. Newton's steps then leave the linear window, and the
       damped ones fall short (MSL DifferentialAmplifier Unstable at 8 ms). `blockSolve!` then takes differences
       over sqrt(eps) of each unknown's own size (`_differenceStep`) =#
    local generation::Module = OMBackend.CodeGeneration.ArrayODEGen
    local limiter! = (res, y, du, a, u, p, t, z, mode, fire) ->
      (res[1] = clamp(1.0e9 * (y[1] - 1.6e-3), -10.0, 10.0) - y[2]; res[2] = y[2] - 9.9; a .= y; nothing)
    local arguments::Tuple = (zeros(1), zeros(2), zeros(1), nothing, 0.0, nothing, false, false)
    for kind in (:full, :coarse, :damped)
      local plain = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [1.6e-3, 0.0])
      plain.x .= plain.y
      @test !(generation._blockAttempt!(plain, limiter!, arguments, kind) && !plain.rankDeficient)
    end
    local solver = generation.BlockSolver(generation._SPARSE.sparse(ones(2, 2)), [1, 2], [1.6e-3, 0.0])
    generation.blockSolve!(solver, limiter!, zeros(1), zeros(2), zeros(1), nothing, 0.0, nothing, false, false)
    @test solver.failures == 0
    @test isapprox(solver.x, [1.6e-3 + 9.9e-9, 9.9]; rtol = 1e-12)
  end

  @testset "A function inlined into an equation: its relations generate no events" begin
    #= `dp = squareBothWays(q, 1, 4)`, inlined `if q >= 0 then q^2 else -4 q^2`, in the loop of q: as an event's
       relation its branch was held during the loop's solve (the residual kinked at zero flow: MSL AST_BatchPlant
       TwoTanks Unstable where the levels meet, EmptyTanks 30200 steps) and a small loop refused it. A function's
       relations are no events (MLS 3.7.5): `IndexReduction.withoutEvents` =#
    local (si, ai, _) = _bothWays("ArrayPath.InlinedRelation", ["h[1]", "h[2]", "q", "dp"]; reltol = 1e-8, abstol = 1e-10)
    local d::Float64 = (sqrt(2) - 1)^2
    @test isapprox(ai, [1 + d / 2, 1 - d / 2, sqrt(2) - 1, d]; atol = 1e-6)
    @test maximum(abs.(si .- ai)) < 1e-6
  end

  @testset "pre() of array elements in an initial for-equation; a field of an if-expression of records" begin
    #= `pre(above[i]) = x0 >= height[i]` in an initial for-equation was an initial equation without an unknown
       ("initialization over-determined": six MSL AST_BatchPlant tank models); it gives pre() of each element at the
       initialization, as the scalar equation's does (`_initialPreStart`). At the start x = 0.9 is inside the band of
       above[1], which keeps pre()'s true: count 11. (The scalarized route starts with 10: not compared there) =#
    local startName::String = _arrayPathTranslate("ArrayPath.PreStartInLoop"; scalarized = false)
    @test _arrayPathTaken(startName)
    @test _values(OMBackend.simulateModel(startName; tspan = (0.0, 0.01)), ["count"]) == [11.0]
    local (sp, ap, _) = _bothWays("ArrayPath.PreStartInLoop", ["x", "count"])
    @test isapprox(ap, [0.5 + 0.4 * cos(1.0), 10.0]; atol = 1e-6)
    @test maximum(abs.(sp .- ap)) < 1e-6
    #= `w = {Complex(x, 0) * quarterTurn(1), ...}`: the inlined function's value is an if-expression of records,
       whose field is the if-expression of its branches' fields (MSL ComplexBlocks ShowTransferFunction's powerOfJ) =#
    local (sq, aq, _) = _bothWays("ArrayPath.RecordIfField", ["x", "w[1].re", "w[1].im", "w[2].re", "w[2].im"])
    @test isapprox(aq, [exp(-1), 0.0, exp(-1), -exp(-1), 0.0]; atol = 1e-6)
    @test maximum(abs.(sq .- aq)) < 1e-6
  end

  @testset "An array of components with algorithm sections; an enumeration's literal as a subscript" begin
    #= with the arrays kept the sections of `SampleCounter c[2]` come as one for-statement around the when-statement;
       none of the lowerings of an algorithm took it and the counters were left without an equation ("1 equations for
       3 unknowns"; MSL Digital's InertialDelaySensitive). `whenLoopSections` writes the iterations as sections =#
    local (sa, aa, _) = _bothWays("ArrayPath.ComponentAlgorithms", ["c[1].n", "c[2].n", "x"]; tspan = (0.0, 0.95))
    @test aa[1:2] == [3.0, 9.0]
    @test isapprox(aa[3], exp(-0.95); atol = 1e-5)
    @test maximum(abs.(sa .- aa)) < 1e-5
    #= `gain[Logic3.High]`, an array over an enumeration at a literal, is the element at the literal's position
       (`IR.literalIndex`; was "a subscript that is not constant": MSL Digital's ResetMap[Logic.'U']). The lifted
       when-equation of k has `change(tHigh[1] > 0)` among its triggers, of parameters: no trigger (`_addWhen!`;
       the Digital registers on array code have 0 variables off omc, the scalarized route 12 of 24) =#
    local (se, ae, _) = _bothWays("ArrayPath.EnumSubscriptCluster", ["y", "k", "x"])
    @test isapprox(ae, [2 * exp(-1) + 2, 2.0, exp(-1)]; atol = 1e-5)
    @test maximum(abs.(se .- ae)) < 1e-5
  end

  @testset "Initial values of discrete variables: the variable on the right, an initial if-equation" begin
    #= `y0 = last` and `if x > level then mode = 2; else mode = 1; end if` give a discrete variable its initial value;
       taken as initial equations to solve they had no free state ("initialization over-determined": MSL Mean's
       `y0 = y_last`, SoftStarter's modeOfOperation). Initial assignments (`_addInitialEquations!`) =#
    local reversed = Modelica"""
      model ReversedInitialAssignment "last = 3 until the sample at 0.5 adds x there"
        parameter Real y0 = 3;
        discrete Real last;
        Real x(start = 1, fixed = true);
        Real y;
      initial equation
        y0 = last;
      equation
        der(x) = -x;
        when sample(0.5, 0.5) then
          last = pre(last) + x;
        end when;
        y = last;
      end ReversedInitialAssignment;
      """
    for (stop, expected) in ((0.4, [3.0, 3.0]), (0.9, [3 + exp(-0.5), 3 + exp(-0.5)]))
      local (sr, ar, _) = _bothWaysOf(reversed, "ReversedInitialAssignment", ["y", "last"]; tspan = (0.0, stop))
      @test isapprox(ar, expected; atol = 1e-5)
      @test maximum(abs.(sr .- ar)) < 1e-5
    end
    local branches = Modelica"""
      model InitialIfAssignment "mode = 2 from the start (x = 1 > level), 3 once x = exp(-t) is below 0.4, at 0.916"
        parameter Real level = 0.4;
        Real x(start = 1, fixed = true);
        discrete Integer mode;
        Real y;
      initial equation
        if x > level then
          mode = 2;
        else
          mode = 1;
        end if;
      equation
        der(x) = -x;
        when x < level then
          mode = 3;
        end when;
        y = mode * x;
      end InitialIfAssignment;
      """
    for (stop, expected) in ((0.5, [2 * exp(-0.5), 2.0]), (1.0, [3 * exp(-1), 3.0]))
      local (sf, af, _) = _bothWaysOf(branches, "InitialIfAssignment", ["y", "mode"]; tspan = (0.0, stop))
      @test isapprox(af, expected; atol = 1e-5)
      @test maximum(abs.(sf .- af)) < 1e-5
    end
  end

  @testset "A String parameter given by a string function of constants" begin
    #= `c1 = Modelica.Utilities.Strings.substring(group, 1, 1)` is "D", read by the binding of k: the array code had
       the values of String parameters bound to literals only ("the value of parameter transformerData.V1ph: computed
       at the start, and its binding has no array code": MSL Machines' nine transformer examples). The runtime
       library's function is called at the translation (`_stringCallValue`) =#
    local strings = Modelica"""
      package Modelica
        package Utilities
          package Strings
            function substring
              input String string;
              input Integer startIndex;
              input Integer endIndex;
              output String result;
              external "C" result = ModelicaStrings_substring(string, startIndex, endIndex);
            end substring;
          end Strings;
        end Utilities;
      end Modelica;

      model StringFunctionParameter "k = 2 for the group D: x = exp(-2 t)"
        parameter String group = "Dy01";
        final parameter String c1 = Modelica.Utilities.Strings.substring(group, 1, 1);
        final parameter Real k = 2 / (if c1 == "D" then 1 else 2);
        Real x(start = 1, fixed = true);
      equation
        der(x) = -k * x;
      end StringFunctionParameter;
      """
    local (ss, as, _) = _bothWaysOf(strings, "StringFunctionParameter", ["x"])
    @test isapprox(as, [exp(-2)]; atol = 1e-5)
    @test maximum(abs.(ss .- as)) < 1e-5
  end

  @testset "An Integer field of an array of records with a declaration binding" begin
    #= `reader(states = {source, source})` gives the fields of `states` their bindings; the Real one became a
       for-equation, the Integer one (`states.phase = {source.phase, source.phase}`) no equation: its elements were
       discrete unknowns nothing gives ("314 equations for 316 unknowns": MSL Fluid PumpingSystem's
       `heatTransfer.states.phase`). An equation per element (`IndexReduction.discreteBindingEquations`) =#
    local phases = Modelica"""
      record PhaseState
        Integer phase;
        Real p;
      end PhaseState;

      model Reader
        input PhaseState states[2];
        Real y;
      equation
        y = states[1].phase * states[2].p;
      end Reader;

      model DiscreteArrayBinding "phase = 2 while x = exp(-t) > 0.5, then 1; y = phase x"
        Real x(start = 1, fixed = true);
        PhaseState source;
        Reader reader(states = {source, source});
      equation
        der(x) = -x;
        source.phase = if x > 0.5 then 2 else 1;
        source.p = x;
      end DiscreteArrayBinding;
      """
    for (stop, expected) in ((0.5, [2 * exp(-0.5)]), (1.0, [exp(-1)]))
      local (sd, ad, _) = _bothWaysOf(phases, "DiscreteArrayBinding", ["reader.y"]; tspan = (0.0, stop))
      @test isapprox(ad, expected; atol = 1e-5)
      @test maximum(abs.(sd .- ad)) < 1e-5
    end
  end

  @testset "delay(): histories per instance, jumps as events, the scalarized path's history functions" begin
    #= the array code refused delay(). It keeps a history per instance of a call in its run state and reads and
       fills it with the functions of delays.jl the scalarized path uses (`delayedValue`, `recordDelayStepEnd!`,
       `recordDelayJumps!`, `delayJumpsDue!`): MSL's lossless lines and Digital's transport delays =#
    local delays = Modelica"""
      model DelayedDecay "y = x(t - 0.25): exp(-0.75) at t = 1, the start value 1 until 0.25"
        Real x(start = 1, fixed = true);
        Real y;
      equation
        der(x) = -x;
        y = delay(x, 0.25);
      end DelayedDecay;

      model DelayedStep "u steps at 0.3, y = delay(u, 0.2) at 0.5: x, the integral of y, is 0.5 at t = 1"
        Real u;
        Real y;
        Real x(start = 0, fixed = true);
      equation
        u = if time > 0.3 then 1 else 0;
        y = delay(u, 0.2);
        der(x) = y;
      end DelayedStep;

      model DelayedPair "each end the other's, delayed: at the start a = 0.5 b, b = 1 + 0.5 a, so a = 2/3, b = 4/3"
        parameter Real T[2] = {0.1, 0.2};
        Real a[2];
        Real b[2];
        Real x(start = 0, fixed = true);
      equation
        for k in 1:2 loop
          a[k] = sin(time) + 0.5 * delay(b[k], T[k]);
          b[k] = cos(time) + 0.5 * delay(a[k], T[k]);
        end for;
        der(x) = a[1] + b[2];
      end DelayedPair;

      model DelayedTable "a table read at a delayed logic value: 30 until the step at 0.3 arrives at 0.5, then 40"
        parameter Real table[5] = {10, 20, 30, 40, 50};
        Real u;
        Integer k;
        Real y;
      equation
        u = if time > 0.3 then 4 else 3;
        k = integer(delay(u, 0.2));
        y = table[k];
      end DelayedTable;
      """
    #= between the steps the history is linear: the solver's steps at this tolerance leave 1.2e-4 =#
    local (sd, ad, _) = _bothWaysOf(delays, "DelayedDecay", ["x", "y"]; reltol = 1e-8, abstol = 1e-10)
    @test isapprox(ad, [exp(-1), exp(-0.75)]; atol = 2e-4)
    @test maximum(abs.(sd .- ad)) < 1e-6
    local (_, early, _) = _bothWaysOf(delays, "DelayedDecay", ["y"]; tspan = (0.0, 0.2))
    @test early == [1.0]
    #= a jump of the argument is a jump of the delayed value its delay time later: an event there, the left limit
       recorded before the relations take their new values =#
    for (stop, expected) in ((0.45, [1.0, 0.0, 0.0]), (1.0, [1.0, 1.0, 0.5]))
      local (ss, as, _) = _bothWaysOf(delays, "DelayedStep", ["u", "y", "x"]; tspan = (0.0, stop), reltol = 1e-8,
                                      abstol = 1e-10)
      @test isapprox(as, expected; atol = 1e-6)
      @test maximum(abs.(ss .- as)) < 1e-6
    end
    #= at the initialization delay(x) is x: the argument is read at the present time, the two ends a loop there =#
    local (_, start, _) = _bothWaysOf(delays, "DelayedPair", ["a[1]", "b[1]", "a[2]", "b[2]"]; tspan = (0.0, 1e-6))
    @test isapprox(start, [2 / 3, 4 / 3, 2 / 3, 4 / 3]; atol = 1e-4)
    local (sp, ap, _) = _bothWaysOf(delays, "DelayedPair", ["a[1]", "b[2]", "x"]; reltol = 1e-8, abstol = 1e-10)
    @test maximum(abs.(sp .- ap)) < 2e-4
    #= integer() of a delayed value that jumps exactly onto the next Integer takes it (`_evfn`), and its value is a
       table's subscript (MSL Digital's delays: 0 variables off omc; before, the logic value stayed). The scalarized
       path keeps 3 after 0.5 for an Integer variable defined so (its cluster has no trigger at the delayed jump):
       compared before the jump only =#
    for (stop, expected) in ((0.45, [3.0, 30.0]), (1.0, [4.0, 40.0]))
      local (st, at, _) = _bothWaysOf(delays, "DelayedTable", ["k", "y"]; tspan = (0.0, stop))
      @test at == expected
      stop < 0.5 && @test st == expected
    end
  end

  @testset "Function arguments and results: arrays as wholes, the first of several outputs, free parameters" begin
    #= `y = f(x)` of a function of two outputs takes the first, an array, by its elements (`SC.TSUB` under a
       subscript); an array variable or a sum of arrays as the argument is the array of its elements
       (`_wholeArrayIRx`, `_elementIRx`): MSL Machines' `i_ = ToSpacePhasor(plug_p.pin.i)`, `ToSpacePhasor(v_p - v_n)` =#
    local names::Vector{String} = ["y[1]", "y[2]", "w[1]", "w[2]"]
    local (sf, af, _) = _bothWays("ArrayPath.FirstArrayOutput", names)
    local y::Vector{Float64} = [exp(-1) - 2 * exp(-2), 2 * exp(-2) - 4 * exp(-3)]
    @test isapprox(af, [y; 2 .* y]; atol = 1e-6)
    @test maximum(abs.(sf .- af)) < 1e-9
    #= an initial tuple equation whose targets are parameters with fixed = false is an initial algorithm =#
    local (sp, ap, _) = _bothWays("ArrayPath.FreeParameterTuple", ["x[1]", "x[2]"])
    @test isapprox(ap, [3 - 2 * exp(-1), 1.5 - 0.5 * exp(-2)]; atol = 1e-6)
    @test maximum(abs.(sp .- ap)) < 1e-6
    #= ... that runs before the initial system where its parameters are read there (der(x) = 0): after it, the steady
       state was solved with a = b = 0 and x stayed at 0 =#
    local (ss, as, _) = _bothWays("ArrayPath.FreeParameterTupleSteady", ["x[1]", "x[2]"])
    @test isapprox(as, [3.0, 1.5]; atol = 1e-6)
    @test maximum(abs.(ss .- as)) < 1e-6
    #= initial values of discrete variables from parameters (an initial algorithm each) the initial system reads: they
       ran after it, and mu = y read r's start, 0 =#
    local (sd, ad, _) = _bothWays("ArrayPath.InitialDiscreteChain", ["mu"])
    @test isapprox(ad, [0.72]; atol = 1e-9)
    @test maximum(abs.(sd .- ad)) < 1e-9
  end

  @testset "pre() of an algebraic variable at a time event: its value before the relation switches" begin
    #= a crossing took the relations' new values before the left limit was computed: pre(u) was u after the
       step and the maxima missed it (y_max -1.4, z_max 2.2) =#
    local (sl, al, _) = _bothWays("ArrayPath.LeftLimitAtTimeEvent", ["y_max", "z_max"])
    @test isapprox(al, [1.6, 2.8]; atol = 1e-6)
    @test maximum(abs.(sl .- al)) < 1e-6
  end

  @testset "Scalarized by the backend: array targets of when-equations and of an algorithm's tuple" begin
    #= models the array code does not take stay in the backend (its element view, no second flatten): `d = 2 * pre(d)`
       in a when-equation is an assignment per element (`elementAssignments`), in the branches of an if too; the array
       target of an algorithm's tuple stays its reference (`wholeTargets`). The element view gave both the array of
       the elements as the target, which the when's and the tuple's translation refused. An initial array equation
       `pre(s) = seedPair(seed)` has its elements in that orientation (as lowered it is `rhs = lhs`): with
       `seedPair(seed)[k] = pre(s[k])` the scalarized model did not build. The two tuple models are the array
       code's since it takes pre() of an array as a function's argument, the two when models since it takes an
       array assigned as a whole in a when-equation and in an if-equation in one (`_whenStatements!`): the backend's
       scalarization of the when models checked with the array code off (`whenModel`); the same values =#
    for (model, names, T, expected, whenModel) in
        (("ArrayPath.WhenArrayTarget", ["d[1]", "d[2]"], 1.0, [6.0, 8.0], true),
         ("ArrayPath.WhenIfArrayTarget", ["d[1]", "d[2]"], 1.0, [8.0, 10.0], true),
         ("ArrayPath.TupleArrayTarget", ["r", "s[1]", "s[2]"], 0.9, [5.0, 3.0, 3.0], false),
         ("ArrayPath.PreArrayStart", ["r", "s[1]", "s[2]"], 0.9, [2.000000005e9, 2.000000003e9, 3.0], false))
      #= the route of OM's translate (`translateReducedArrays`: `nothing` is its caller's
         `ARRAY_PRESERVING_CODEGEN_NOT_SUPPORTED`) =#
      local name::String = _arrayPathTranslate(model; scalarized = false)
      local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, _ARRAY_PATH_SCODE[]; scalarize = false)
      @test OMBackend.translateReducedArrays(fm; functionList = OMFrontend.cacheToFunctionList(cache)) !== nothing
      @test _arrayPathTaken(name)
      if whenModel
        OMBackend.ARRAY_ODE_GENERATION[] = false
        try
          @test OMBackend.translateReducedArrays(fm; functionList = OMFrontend.cacheToFunctionList(cache)) !== nothing
          @test name in OMBackend.BACKEND_SCALARIZED_MODELS
          @test _values(OMBackend.simulateModel(name; tspan = (0.0, T)), names) ≈ expected atol = 1e-9
          #= switched off: `nothing` (the caller's error), the reason recorded =#
          withenv("OMBACKEND_BACKEND_SCALARIZATION" => "false") do
            @test OMBackend.translateReducedArrays(fm; functionList = OMFrontend.cacheToFunctionList(cache)) === nothing
            @test !(name in OMBackend.BACKEND_SCALARIZED_MODELS)
            @test occursin("backend scalarization: switched off", OMBackend.scalarizedFallbackReason(name))
          end
        finally
          OMBackend.ARRAY_ODE_GENERATION[] = true
        end
      end
      _arrayPathTranslate(model; scalarized = false)
      local sol::OMBackend.DifferentialEquations.SciMLBase.AbstractODESolution =
        OMBackend.simulateModel(name; tspan = (0.0, T))
      @test sol.retcode == ReturnCode.Success
      @test _values(sol, names) ≈ expected atol = 1e-9
    end
  end

  @testset "Scalarized by the backend: reductions of no elements, elements of a row of a table" begin
    local I::Module = OMBackend.OMJLIR
    #= sum and product of no elements are 0 and 1 (a tank's sum(topPorts.m_flow) without top ports: with the arrays
       kept `sum({})` reached the generated code, MethodError zero(Any); MSL AST_BatchPlant's tank tests) =#
    @test OMBackend.Passes.foldReduction(:sum, I.Expression[], I.T_REAL(), I.BUILTIN_CALL) === I.REAL(0.0)
    @test OMBackend.Passes.foldReduction(:product, I.Expression[], I.T_INTEGER(), I.BUILTIN_CALL) === I.INTEGER(1)
    @test OMBackend.Passes.foldReduction(:max, I.Expression[], I.T_REAL(), I.BUILTIN_CALL) === nothing
    #= fire[ka + 1, :][2] is fire[ka + 1, 2] (a row of SVPWM's constant Boolean table by a discrete index: left
       nested, the row's lookup was one term in the generated code, its second element a BoundsError) =#
    local ka::I.Expression = I.CREF(I.ComponentRef(:ka, I.T_INTEGER()), I.T_INTEGER())
    local table::I.Expression = I.CREF(I.ComponentRef(:fire, I.T_BOOLEAN()), I.T_BOOLEAN())
    local row::I.Expression = I.SUBSCRIPTED(table, I.Subscript[I.INDEX(ka), I.WHOLE()], I.T_BOOLEAN())
    local picked::I.Expression =
      OMBackend.Passes.Evaluation.selectElement(row, I.Subscript[I.INDEX(I.INTEGER(2))], I.T_BOOLEAN())
    @test picked isa I.SUBSCRIPTED && picked.exp === table && length(picked.subscripts) == 2
    @test picked.subscripts[1] isa I.INDEX && picked.subscripts[1].exp === ka
    @test picked.subscripts[2] isa I.INDEX && picked.subscripts[2].exp === I.INTEGER(2)
  end

  @testset "a when algorithm reading its target, reinit of elements, enumeration dimensions" begin
    #= (s, y) := step(s) at 0.1, 0.35, 0.6, 0.85: s from pre(s) at each run (the event iteration reran the body with
       the last pass's s and never settled) =#
    local (s, a, _) = _bothWays("ArrayPath.WhenReadsTarget", ["s", "y"])
    @test a == [4.0, 30.0] && s == a
    #= v[i] reset at 1.5 (rates i), v[3] set to -1 at 0.3 (a parameter's subscript, which the scalarized path's when
       code does not take): [1.2, 0.9, 0.2] at 1.2 =#
    local name = _arrayPathTranslate("ArrayPath.ReinitElements"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.2))
    @test sol.retcode == ReturnCode.Success
    @test isapprox(_values(sol, ["v[1]", "v[2]", "v[3]"]), [1.2, 0.9, 0.2]; atol = 1e-6)
    #= the record's fields and the scalar by their positions in the result, 1, 2, 3 (1, 3, 4 before: out of the
       tuple's range, an illegal instruction under @inbounds); the last tick at 0.5 =#
    local (sq, aq, _) = _bothWays("ArrayPath.RecordTupleWhen", ["r.a", "r.b", "q"]; tspan = (0.0, 0.55))
    @test isapprox(aq, [0.5, 1.0, 1.5]; atol = 1e-12) && isapprox(sq, aq; atol = 1e-12)
    #= a regular algorithm reading its target before assigning it: n := pre(n) + u at each change of u (the event
       iteration did not settle: n grew by u at every pass); OpenModelica 1.27.1: u = 4, n = 10, m = 20 at 1 =#
    local (sn, an, _) = _bothWays("ArrayPath.RegularAlgReads", ["u", "n", "m"])
    @test an == [4.0, 10.0, 20.0] && sn == an
    #= a when algorithm's targets start from pre(): edge() of one before its assignment is false (cnt = 0), a whole
       array read of element targets their pre() values (s = 4, 6); OpenModelica 1.27.1's values =#
    local (sg, ag, _) = _bothWays("ArrayPath.EdgeOfTarget", ["cnt"])
    @test ag == [0.0] && sg == ag
    local (ss, as, solS) = _bothWays("ArrayPath.SumOfTargets", ["s"])
    @test as == [6.0] && ss == as && OMBackend.getVariableValues(solS, "s")[findlast(<=(0.5), solS.t)] == 4.0
    #= a read through a variable subscript (x[i], i = 2) before the assignment of x[2]: x[2] from pre(x[2]) (did not
       settle: x[2] grew by 10 at every pass); OpenModelica 1.27.1: 40 at 1 =#
    local (sv, av, _) = _bothWays("ArrayPath.VarSubRead", ["x[1]", "x[2]"])
    @test av == [1.0, 40.0] && sv == av
    #= targets through a variable index of a record array (the scalarized path refuses them): an element assigned
       before its guarded assignment through s[k] keeps its value there; an element read before its assignment through
       s[k] starts from pre(); OpenModelica 1.27.1's values =#
    local at = (name::String, var::String, ts::Vector{Float64}) -> begin
      local sol = OMBackend.simulateModel(_arrayPathTranslate(name; scalarized = false); tspan = (0.0, 1.0))
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t), sol.t)] for t in ts]
    end
    @test at("ArrayPath.AssignedThenGuarded", "s[2].a", [0.2, 0.4, 0.7, 0.9]) == [5.0, 5.0, 7.0, 7.0]
    @test at("ArrayPath.ReadThenIndexed", "s[2].a", [0.2, 0.9]) == [1.0, 4.0]
    @test at("ArrayPath.ReadThenIndexed", "y", [0.9]) == [3.0]
    #= pre() of an element by an index assigned after (the index read now); a ring buffer read, written and advanced
       through its index; an element target by an index that changes between the passes: every element it may assign
       starts from pre() =#
    @test at("ArrayPath.PreIndexedRecord", "y", [0.2, 0.4, 0.7, 0.9]) == [10.0, 20.0, 10.0, 20.0]
    @test at("ArrayPath.RingRead", "y", [0.2, 0.4, 0.7, 0.9]) ≈ [0.0, 0.0, 0.1, 0.35]
    @test at("ArrayPath.StaleElement", "s[2].a", [0.2, 0.9]) == [0.0, 0.0] && at("ArrayPath.StaleElement", "s[1].a", [0.9]) ≈ [0.85]
    @test [at("ArrayPath.RecBig", "s[$i].a", [1.0])[1] for i in 1:3] ≈ [0.85, 0.35, 0.6]
    #= discrete variables equations compute, no events: evaluated at the start (they kept their starts: x = 0) =#
    local (sd, ad, _) = _bothWays("ArrayPath.DiscreteNoEvents", ["x"])
    @test isapprox(ad, [4.0]; atol = 1e-9) && isapprox(sd, ad; atol = 1e-9)
    #= a table over an enumeration's dimensions (its values per element: a KeyError before): T[o, o] = o, the 4th =#
    local (st, at, _) = _bothWays("ArrayPath.EnumTable", ["a[2]"])
    @test at == [4.0] && st == at
  end

  @testset "discrete equations at every event, time events, sample() in equations, tuples in whens, parameters" begin
    local arrayRun = (name::String; kw...) -> begin
      local n = _arrayPathTranslate(name; scalarized = false)
      @test _arrayPathTaken(n)
      local sol = OMBackend.simulateModel(n; tspan = (0.0, 1.0), kw...)
      @test sol.retcode == ReturnCode.Success
      return sol
    end
    local at = (sol, var::String, ts::Vector{Float64}) -> begin
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t), sol.t)] for t in ts]
    end
    #= a lifted discrete cluster runs at every pass of every event and of the start's iteration: pre(mode) the initial
       value 0 after the initialization, so started and mode 1 (the cluster fired only on its relation x < 0.5: mode
       0, x(1) = 0); OpenModelica 1.27.1's values =#
    local pai = arrayRun("ArrayPath.PreAfterInit")
    @test at(pai, "mode", [0.0, 0.5]) == [1.0, 1.0]
    @test OMBackend.getVariableValues(pai, "x")[end] ≈ 1.0 atol = 1e-6
    #= a time event (time < 0.5) at its exact time, the relations literal there: f <= 0 at f = 0 holds (with the
       events' hysteresis free stayed false: x(1) = 0); OpenModelica 1.27.1's values =#
    local eb = arrayRun("ArrayPath.ExactBoundaryAtEvent")
    @test at(eb, "free", [0.45, 0.6]) == [0.0, 1.0]
    @test OMBackend.getVariableValues(eb, "x")[end] ≈ 0.5 atol = 1e-6
    #= sample() in an equation, its start from an initial equation: the ticks at 0, 0.2, ... (refused before);
       OpenModelica 1.27.1's values =#
    @test at(arrayRun("ArrayPath.SampleEquation"), "y", [0.1, 0.3, 0.5, 0.9]) ≈ [0.0, 0.0, 0.2, 0.6] atol = 1e-9
    #= a tuple with an array output in a when-equation, each output at its position in the result: ten Fibonacci steps =#
    local tw = arrayRun("ArrayPath.TupleWhen")
    @test _values(tw, ["r", "s[1]", "s[2]"]) ≈ [5.5, 89.0, 55.0]
    @test at(tw, "r", [0.1, 0.32]) ≈ [0.1, 0.2]
    #= an array parameter bound to a call of literals, a String parameter as an assert's message =#
    @test _values(arrayRun("ArrayPath.ParameterOfArrayCall"), ["y"]) ≈ [3.0] atol = 1e-9
    @test _values(arrayRun("ArrayPath.StringParameterMessage"), ["x"]) ≈ [exp(-1)] rtol = 1e-4
    #= through a reference to another String parameter (kept by the arrays-kept passes: it was declined) =#
    @test _values(arrayRun("ArrayPath.StringParameterReference"), ["x"]) ≈ [exp(-1)] rtol = 1e-4
    #= sample() read other than as a when's rising edge: the samples' fall at the instant in an event iteration of its
       own (a single pass absorbed the rise of y = not u and pre(u): y stayed true, n = 0); OpenModelica 1.27.1's values =#
    local fe = arrayRun("ArrayPath.FallingEdgeOfSample")
    @test at(fe, "y", [0.2, 0.5]) == [0.0, 0.0] && at(fe, "n", [0.2, 0.5, 1.0]) == [1.0, 2.0, 4.0]
    @test at(arrayRun("ArrayPath.TickNot"), "n", [0.1, 0.3, 0.6, 1.0]) == [1.0, 2.0, 3.0, 5.0]
    #= the samples true for the first pass of the instant's event iteration only (OpenModelica's updateDiscreteSystem):
       an iteration with them true until it settled counted twice (change(u)), never settled (k), or read pre() of
       what the instant set (b); OpenModelica 1.27.1's values =#
    local sf = arrayRun("ArrayPath.SampleFirstPass")
    @test at(sf, "c", [0.2, 0.5, 1.0]) == [1.0, 2.0, 4.0] && at(sf, "k", [0.2, 0.5, 1.0]) == [1.0, 2.0, 4.0]
    @test at(sf, "a", [0.2, 1.0]) == [1.0, 4.0] && at(sf, "b", [0.2, 0.5, 1.0]) == [0.0, 1.0, 3.0]
    #= ticks of two samples equal up to round-off are one instant (two stops fired both samples twice) =#
    local ts = arrayRun("ArrayPath.TwoSampleTimes")
    @test at(ts, "a", [0.5, 0.7, 1.0]) == [3.0, 4.0, 6.0] && at(ts, "b", [0.5, 0.7, 1.0]) == [2.0, 3.0, 4.0]
    #= ticks apart by more than round-off are instants of their own (a 1e-9 window took 5e-10 apart as one: b read
       pre(a) before a's tick), and a tick near two stops fires once (it fired at both); OpenModelica 1.27.1's values =#
    local nt = arrayRun("ArrayPath.NearTicks")
    @test at(nt, "b", [0.05, 0.2, 0.5, 1.0]) == [0.0, 1.0, 2.0, 4.0] && at(nt, "c", [0.05, 0.2, 0.5, 1.0]) == [0.0, 1.0, 2.0, 4.0]
    local ct = arrayRun("ArrayPath.ChainTicks")
    @test all(v -> at(ct, v, [0.2, 1.0]) == [1.0, 4.0], ["a", "b", "c"])
    #= a tick 5e-10 after the start or the stop time is not at it (a 1e-9 snap fired them there: n(1) = 5, k(1) = 4) =#
    local sn = arrayRun("ArrayPath.TicksNearStartAndStop")
    @test at(sn, "n", [0.0, 0.1, 1.0]) == [0.0, 1.0, 4.0] && at(sn, "k", [1.0]) == [3.0]
    @test at(sn, "tn", [0.1]) ≈ [5e-10] atol = 1e-15
    #= a tick's round-off is relative to the magnitudes in it (st + k dt): -100 + 1003 * 0.1 and 0.3 one instant (two
       with a window relative to t: w = 5 at 0.35) =#
    @test at(arrayRun("ArrayPath.FarStartTicks"), "w", [0.05, 0.35, 0.85, 1.0]) == [1.0, 4.0, 9.0, 11.0]
  end

  @testset "overrides of parameters read only by starts" begin
    local x0 = (name::String, var::String, parameters::Dict{String, Any}) ->
      OMBackend.getVariableValues(OMBackend.resimulateModel(name; tspan = (0.0, 1.0), parameters), var)[1]
    #= sum(q) in a start (q was no parameter of the array code: the sum of an array literal did not translate); omc
       6, with -override=q[1]=10 15 =#
    local su = _arrayPathTranslate("ArrayPath.StartOfSum"; scalarized = false)
    @test _arrayPathTaken(su)
    @test x0(su, "x", Dict{String, Any}()) == 6.0
    @test x0(su, "x", Dict{String, Any}("q[1]" => 10.0)) == 15.0
    @test x0(su, "x", Dict{String, Any}("q" => [10.0, 2.0, 3.0])) == 15.0
    #= the starts of rolled components, c[i].x(start = c[i].s) with c.s = q (c[1].s was no variable of the array
       code); omc 4, 5, 6 =#
    local sc = _arrayPathTranslate("ArrayPath.StartOfComponents"; scalarized = false)
    @test _arrayPathTaken(sc)
    @test [x0(sc, "c[$(i)].x", Dict{String, Any}("q" => [4.0, 5.0, 6.0])) for i in 1:3] == [4.0, 5.0, 6.0]
    @test [x0(sc, "c[$(i)].x", Dict{String, Any}("c[2].s" => 9.0)) for i in 1:3] == [1.0, 9.0, 3.0]
  end

  @testset "der() coefficients of free parameters, overrides, two problems of one model" begin
    #= the arrays-kept route (both routes: arrayPathRoutesTests.jl) =#
    local run = (model::String, scalarized::Bool; taken::Bool = true, parameters = Dict{String, Any}()) -> begin
      local n = _arrayPathTranslate("ArrayPath." * model; scalarized)
      (scalarized || !taken) || @test _arrayPathTaken(n)
      return isempty(parameters) ? OMBackend.simulateModel(n; tspan = (0.0, 1.0)) :
                                   OMBackend.resimulateModel(n; tspan = (0.0, 1.0), parameters)
    end
    local at = (sol, var::String, ts::Vector{Float64}) -> begin
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t + 1e-12), sol.t)] for t in ts]
    end
    #= the array path: a der() coefficient of a parameter with fixed = false stays an expression (its parameter was a
       binding of a free parameter: declined); a discrete start of one after the initialization =#
    @test _values(run("CoefFreeParam", false), ["x"]) ≈ [1 / sin(0.5)] rtol = 1e-6
    @test at(run("DiscreteStartFree", false), "d", [0.2, 1.0]) == [2.0, 3.0]
    #= overrides: of a String parameter (OpenModelica's -override=s=xyz), of a value of another kind =#
    local sp = _arrayPathTranslate("ArrayPath.StringParam"; scalarized = false)
    @test _arrayPathTaken(sp)
    local se = try
      OMBackend.resimulateModel(sp; tspan = (0.0, 1.0), parameters = Dict{String, Any}("s" => "xyz"))
      nothing
    catch e
      e
    end
    @test se isa OMBackend.CodeGeneration.ModelicaAssertionError && occursin("xyz", se.message)
    @test_throws ArgumentError OMBackend.resimulateModel(sp; tspan = (0.0, 1.0), parameters = Dict{String, Any}("s" => 3.0))
    local ps = _arrayPathTranslate("ArrayPath.ParamSample"; scalarized = false)
    @test_throws ArgumentError OMBackend.resimulateModel(ps; tspan = (0.0, 1.0), parameters = Dict{String, Any}("T" => "abc"))
    #= two problems of one model, both built before either is solved: each its own samples (they shared them) =#
    local mod::Module = getfield(OMBackend, Symbol(ps))
    local modelFn = getfield(mod, Symbol(ps, "Model"))
    local (p1, c1) = Base.invokelatest(modelFn, (0.0, 1.0); parameters = Dict{String, Any}("T" => 0.25))
    local (p2, c2) = Base.invokelatest(modelFn, (0.0, 1.0); parameters = Dict{String, Any}("T" => 0.1))
    local DE = OMBackend.DifferentialEquations
    #= the generated module's functions in the latest world (it was evaluated in this test) =#
    @test OMBackend.getVariableValues(Base.invokelatest(DE.solve, p1, DE.Rodas5P(); callback = c1), "n")[end] == 5.0
    @test OMBackend.getVariableValues(Base.invokelatest(DE.solve, p2, DE.Rodas5P(); callback = c2), "n")[end] == 11.0
    #= a copied problem (an ensemble's trajectories, `remake` with deepcopy(p)): the module shared, the run's state its own =#
    local (p3, c3) = Base.invokelatest(modelFn, (0.0, 1.0); parameters = Dict{String, Any}("T" => 0.25))
    local q3 = deepcopy(p3.p)
    @test q3.model === p3.p.model && q3.state !== p3.p.state
    local ens = DE.EnsembleProblem(p3; prob_func = (pr, ctx) -> DE.remake(pr; p = deepcopy(pr.p)))
    local es = Base.invokelatest(DE.solve, ens, DE.Rodas5P(), DE.EnsembleSerial(); trajectories = 2, callback = c3)
    @test [OMBackend.getVariableValues(s, "n")[end] for s in es.u] == [5.0, 5.0]
  end

  @testset "slices of component arrays, a start solution that fires whens, checked subscripts" begin
    local run = (model::String, scalarized::Bool; stop::Float64 = 1.0) -> begin
      local n = _arrayPathTranslate("ArrayPath." * model; scalarized)
      scalarized || @test _arrayPathTaken(n)
      return OMBackend.simulateModel(n; tspan = (0.0, stop))
    end
    local at = (sol, var::String, ts::Vector{Float64}) -> begin
      local v::Vector{Float64} = OMBackend.getVariableValues(sol, var)
      return [v[findlast(<=(t + 1e-12), sol.t)] for t in ts]
    end
    #= a field of a connector array and of a model array through a whole slice (ac.pin[:].i, c[:].x) =#
    local R = [1.0, 2.0, 4.0]
    @test _values(run("SliceConnectors", false), ["load.q[1]", "load.q[2]", "load.q[3]"]) ≈
      [(cos(k) - cos(k + 1)) / R[k] for k in 1:3] atol = 1e-6
    @test _values(run("SliceCells", false), ["s"]) ≈ [2 * sum((1 - exp(-k)) / k for k in 1:3)] atol = 1e-6
    #= the binding of a field of a model array is an equation over the elements (it stays on the merged variable) =#
    local bc = run("BindCells", false)
    @test _values(bc, ["c[2].a", "c[3].b", "y[1]"]) ≈ [exp(-4), 6 * exp(-6), 3 * exp(-2)] atol = 1e-6
    #= a modifier over a model array on an array field is each element's value (a constructor, a matrix); an array
       field bound in the class is an array equation in the for-equation over the elements, rolled =#
    @test _values(run("ModCells", false), ["c[1].x", "c[2].x", "c[3].x"]) ≈ exp.(-[3.0, 5.0, 7.0]) atol = 1e-6
    @test _values(run("ModCellsMatrix", false), ["c[1].x", "c[3].x"]) ≈ exp.(-[3.0, 7.0]) atol = 1e-6
    @test _values(run("WCells", false), ["c[2].a", "c[3].w[2]"]) ≈ [exp(-2), 2 * exp(-2)] atol = 1e-6
    #= a record with array fields in a model array (c.R.T is Real[3, 2, 2]) stays an array of the component =#
    @test _values(run("TurnCells", false), ["c[1].x", "c[3].x"]) ≈ [exp(-2), exp(-2)] atol = 1e-6
    #= elements of smooth() and vector(cat()); an if-equation on a function of parameters; an array-valued
       if-expression is an array variable, its if-equation a for-equation over the elements =#
    @test _values(run("ElementForms", false), ["x", "y", "e[2]"]) ≈ [exp(-1), 2 * exp(-1), -exp(-1)] atol = 1e-6
    #= a fixed start on a dummy state (x, y the state): the initial equation x = start gives the state's value =#
    @test _values(run("FixedDummy", false), ["y[1]", "y[2]", "z[2]"]) ≈
          [2 * exp(-1), 4 * exp(-1), 4 * exp(-1)] atol = 1e-5
    #= index 3 through an inlined function of a record's matrix: R.T[1, i] and R.T[3, i] are read times parameters
       known as zero (e = {0, 1, 0}), so Pantelides' algorithm gave them no derivatives; the for-equation's derivative
       reads none of them =#
    @test _values(run("SlidingMass", false), ["s"]) ≈
          [9.81 / 2 * (cos(1.5) + sin(0.5) * sinh(1)) + (1 - 9.81 / 2 * cos(0.5)) * cosh(1)] atol = 1e-5
    #= an if-equation in a branch of an if-equation on a variable condition: one class `y[i] = if ... (if ...)` =#
    @test _values(run("NestedIf", false), ["y[1]", "y[2]"]) ≈ [2 * exp(-1), 4 * exp(-1)] atol = 1e-5
    #= pre() is the start at the initialization and no when fires on its condition there; the event iteration after it
       fires them: the change of x from its start (omc). Initial sections of two instances, merged to arrays, run in
       the order of what they read through the equation between them (d2.x = d1.y). The scalarized route misses the
       first change (open) =#
    local T = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 1.0]
    local dp = run("DelayPair", false)
    @test at(dp, "d1.y", T) == [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    @test at(dp, "d2.y", T) == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    #= a run-time subscript outside its array is an error (omc: an assertion), not a read outside it =#
    local err = try
      run("SubscriptOutside", false)
      nothing
    catch e
      e
    end
    @test err isa OMBackend.CodeGeneration.ArrayODEGen.ModelicaSubscriptError && err.index == 3 && err.size == 2
  end

  @testset "the plan's folds in the committed model, if-equations in for-equations, time events, sample stops" begin
    #= an if-equation in a for-equation that index reduction differentiates: the ramp's if-expression, lifted to an
       if-equation (`ifEq_tmp = ...` per branch), in the for-equation over the two masses rolled into one class; the
       position constraint is differentiated twice, branch by branch (it was refused: "differentiating a IF_EQUATION
       in a for-equation"; MSL Spice3's V_pulse sources). Each mass follows its ramp: v its slope, f zero. =#
    local ramps = Modelica"""
      model DrivenMass "a mass whose position a ramp gives: v and f are the ramp's derivatives"
        parameter Real startTime = 0.5;
        parameter Real height = 1;
        parameter Real duration = 1;
        parameter Real m = 1;
        Real x;
        Real v;
        Real f;
      equation
        x = if time < startTime then 0
            else if time < startTime + duration then (time - startTime) * height / duration else height;
        der(x) = v;
        m * der(v) = f;
      end DrivenMass;

      model RolledRampConstraint
        DrivenMass a(startTime = 0.5);
        DrivenMass b(startTime = 1);
      end RolledRampConstraint;
      """
    for (stop, expected) in ((1.2, [0.7, 1.0, 0.2, 1.0, 0.0]), (1.8, [1.0, 0.0, 0.8, 1.0, 0.0]))
      local (sr, ar, _) = _bothWaysOf(ramps, "RolledRampConstraint", ["a.x", "a.v", "b.x", "b.v", "b.f"];
                                      tspan = (0.0, stop))
      @test ar ≈ expected atol = 1e-9
      @test sr ≈ expected atol = 1e-9
    end
    #= a function's output that is zero for every value of its arguments (the simulation code's interpretation: a
       junction without area has no capacitance) takes der(v) out of the plan's structure: v is no state. The
       committed model kept the term and the state (MSL Spice3's MOSFET capacitances: "no unknown left for
       0 = S.v - Sinternal"). Alone, and two of one class rolled (the zero an array's, the equation a for-equation).
       v = g u / (1 + g). =#
    local junctions = Modelica"""
      function junctionCapacitance "none for a junction without area (not inlined: an algorithm with an if)"
        input Real v;
        input Real area;
        output Real c;
      algorithm
        if area > 0 then
          c := area * (1 + v * v);
        else
          c := 0;
        end if;
      end junctionCapacitance;

      model Junction "a node behind a conductance: its voltage a state only where the junction has a capacitance"
        parameter Real area = 0;
        parameter Real g = 2;
        Real u;
        Real v;
        Real c;
        Real ic;
      equation
        c = junctionCapacitance(v, area);
        ic = c * der(v);
        g * (u - v) = ic + v;
      end Junction;

      model ZeroJunction
        Junction a;
      equation
        a.u = sin(time);
      end ZeroJunction;

      model ZeroJunctions
        Junction a(g = 2);
        Junction b(g = 4);
      equation
        a.u = sin(time);
        b.u = cos(time);
      end ZeroJunctions;
      """
    local (sj, aj, _) = _bothWaysOf(junctions, "ZeroJunction", ["a.v", "a.ic"])
    @test aj ≈ [2 * sin(1) / 3, 0.0] atol = 1e-9
    @test sj ≈ aj atol = 1e-9
    local (sjs, ajs, _) = _bothWaysOf(junctions, "ZeroJunctions", ["a.v", "b.v", "b.ic"])
    @test ajs ≈ [2 * sin(1) / 3, 4 * cos(1) / 5, 0.0] atol = 1e-9
    @test sjs ≈ ajs atol = 1e-9
    #= a relation of time and a discrete variable is a time event at its exact time (OpenModelica's
       timeEventTrigger), as one of time and parameters: with a state event's hysteresis, 1e-10 (1 + |time|), the
       pulse of 0.5 ps never ended within this run of 10 ps (x = 1e-11; MSL Spice3 Inverter's sources). The first
       step given: left to the solver it is cut to the first stop, 2 ps, and OrdinaryDiffEqCore 4.19.1 interpolates an
       event in such a step with the step proposed before the cut (x = 0.1 ps at the first edge; fixed in 4.20,
       SciML/OrdinaryDiffEq.jl#4664). The samples' stops come one at a time: the 1001 of the span, each from the one
       before, not a table over the span the problem is built with. =#
    local pulses = Modelica"""
      model PicoPulse "5 pulses of 0.5 ps, each from the time a sample sets"
        parameter Real period = 2e-12;
        parameter Real width = 0.5e-12;
        discrete Real T0(start = 0, fixed = true);
        Real y;
        Real x(start = 0, fixed = true);
      equation
        when sample(0, period) then
          T0 = time;
        end when;
        y = if time < T0 + width then 1 else 0;
        der(x) = y;
      end PicoPulse;

      model Ticks "1001 ticks in 1 s"
        Integer n(start = 0, fixed = true);
        Real x(start = 1, fixed = true);
      equation
        der(x) = -x;
        when sample(0, 1e-3) then
          n = pre(n) + 1;
        end when;
      end Ticks;
      """
    local pico::String = _translateProgram(pulses, "PicoPulse"; scalarized = false)
    @test _arrayPathTaken(pico)
    local ps = OMBackend.simulateModel(pico; tspan = (0.0, 1.0e-11), dt = 1.0e-15)
    @test ps.retcode == ReturnCode.Success
    @test OMBackend.getVariableValues(ps, "x")[end] ≈ 2.5e-12 rtol = 1e-9
    local ticks::String = _translateProgram(pulses, "Ticks"; scalarized = false)
    @test _arrayPathTaken(ticks)
    local (tp, tc) = OMBackend.getMTKProblem(ticks)
    @test tp.tspan == (0.0, 1.0)
    local DE = OMBackend.DifferentialEquations
    local tsol = Base.invokelatest(DE.solve, tp, DE.Rodas5P(); callback = tc)
    @test OMBackend.getVariableValues(tsol, "n")[end] == 1001.0
    @test length(tp.p.state.sampleStops) == 2
  end

  @testset "inStream of three or more connectors; an if-expression on a for-equation's iterator" begin
    #= a stream connection of four connectors: the frontend writes the mixing enthalpy with OMCPositiveMax(flow, eps)
       (NFConnectEquations.makePositiveMaxCall), max in the array code as in the scalar code (it was refused:
       "function OMCPositiveMax"; MSL Fluid DrumBoiler, ThreeTanks). 0.4 at h = 10 and 0.2 at h = 4 flow into the
       junction: h = 8 there, to the eps of 1e-7 =#
    local streams = Modelica"""
      connector FluidPort
        Real p;
        flow Real m_flow;
        stream Real h_outflow;
      end FluidPort;

      model Boundary
        parameter Real p0 = 1;
        parameter Real h0 = 0;
        FluidPort port;
      equation
        port.p = p0;
        port.h_outflow = h0;
      end Boundary;

      model Resistance
        FluidPort a;
        FluidPort b;
      equation
        a.m_flow + b.m_flow = 0;
        a.m_flow = a.p - b.p;
        a.h_outflow = inStream(b.h_outflow);
        b.h_outflow = inStream(a.h_outflow);
      end Resistance;

      model Probe "no flow: the mixing enthalpy of what flows into the junction"
        FluidPort port;
        Real h;
      equation
        port.m_flow = 0;
        port.h_outflow = 0;
        h = inStream(port.h_outflow);
      end Probe;

      model Junction
        Boundary hot(p0 = 2, h0 = 10);
        Boundary cold(p0 = 1.8, h0 = 4);
        Boundary sink(p0 = 1, h0 = 0);
        Resistance r1;
        Resistance r2;
        Resistance r3;
        Probe probe;
        Real x(start = 0, fixed = true);
      equation
        connect(hot.port, r1.a);
        connect(cold.port, r2.a);
        connect(r1.b, r3.a);
        connect(r2.b, r3.a);
        connect(probe.port, r3.a);
        connect(r3.b, sink.port);
        der(x) = probe.h;
      end Junction;
      """
    local (sj, aj, _) = _bothWaysOf(streams, "Junction", ["probe.h", "x", "r3.a.m_flow"])
    @test aj ≈ [8.0, 8.0, 0.6] atol = 1e-5
    @test sj ≈ aj atol = 1e-8
    #= `x[i] = if i == 1 then u else 0.5 * x[i - 1]`: each instance its branch, the class split by the condition on
       the iterator (`_iteratorIf`). As one class the branch not taken was read at every instance: "subscript 0 out of
       1:3 on x" (MSL StateGraph's steps, `outPort[i].available = if i == 1 then active else outPort[i - 1]...`) =#
    local chains = Modelica"""
      model ChainedHalves "x = {u, u / 2, u / 4} with u = 1 + time"
        parameter Integer n = 3;
        Real u;
        Real x[n];
        Real s(start = 0, fixed = true);
      equation
        u = 1 + time;
        for i in 1:n loop
          x[i] = if i == 1 then u else 0.5 * x[i - 1];
        end for;
        der(s) = x[n];
      end ChainedHalves;
      """
    local (sc, ac, _) = _bothWaysOf(chains, "ChainedHalves", ["x[1]", "x[3]", "s"])
    @test ac ≈ [2.0, 0.5, 0.375] atol = 1e-6
    @test sc ≈ ac atol = 1e-6
  end

  @testset "an initial equation on a discrete variable an equation computes" begin
    #= `initial equation active = initiallyActive` with `active = localActive`, `localActive = pre(newActive)` and a
       connection's `active = mirror`: the initial equation gives pre(newActive), `pre(newActive) = pre(localActive)`
       then pre(localActive) (`_throughDefinitions`). As an assignment of `active` it took `active` out of the
       unknowns: "8 equations for 7 unknowns" (MSL StateGraph's steps, FirstExample). Active until `set` at 0.5. =#
    local latches = Modelica"""
      model Latch "a step as StateGraph's: active from pre(newActive), left when set"
        parameter Boolean initiallyActive = true;
        Boolean localActive;
        Boolean newActive;
        Boolean active;
        Boolean mirror;
        Boolean set;
        Real x(start = 0, fixed = true);
      initial equation
        pre(newActive) = pre(localActive);
        active = initiallyActive;
      equation
        localActive = pre(newActive);
        active = localActive;
        active = mirror;
        set = time > 0.5;
        newActive = localActive and not set;
        der(x) = if mirror then 1 else 0;
      end Latch;
      """
    local (sl, al, _) = _bothWaysOf(latches, "Latch", ["x", "active"])
    @test al ≈ [0.5, 0.0] atol = 1e-6
    @test sl ≈ al atol = 1e-6
    local latch::String = _translateProgram(latches, "Latch"; scalarized = false)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(latch; tspan = (0.0, 0.25)), "active")[end] == 1.0
    local inactive = OMBackend.resimulateModel(latch; tspan = (0.0, 1.0),
                                               parameters = Dict{String, Any}("initiallyActive" => false))
    @test OMBackend.getVariableValues(inactive, "x")[end] ≈ 0.0 atol = 1e-9
  end

  @testset "a when-equation on an array: a vector of conditions, change() of an array" begin
    #= `when b then` with b a Boolean array is a when on the vector of its elements (MLS 8.3.5): it fires where one
       becomes true; `change(k)` of an array where an element changes (at 1/3, 1/2, 2/3).
       It was refused: "the array b as a whole in an equation of scalars" (MSL ControlledTanks, StateGraph and Fluid:
       `when pre_reset then on = false`) =#
    local vectors = Modelica"""
      model ArrayWhen "n counts the elements of b that became true, m the instants an element of k changed"
        Boolean b[2];
        Integer k[2];
        Integer n(start = 0, fixed = true);
        Integer m(start = 0, fixed = true);
        Real x(start = 0, fixed = true);
      equation
        b[1] = time > 0.25;
        b[2] = time > 0.75;
        k[1] = integer(2 * time);
        k[2] = integer(3 * time);
        when b then
          n = pre(n) + 1;
        end when;
        when change(k) then
          m = pre(m) + 1;
        end when;
        der(x) = n;
      end ArrayWhen;
      """
    local (sv, av, _) = _bothWaysOf(vectors, "ArrayWhen", ["n", "m", "x"]; tspan = (0.0, 0.9))
    @test av ≈ [2.0, 3.0, 0.8] atol = 1e-6
    @test sv ≈ av atol = 1e-6
  end

  @testset "the elements of a record's array field: each its own start" begin
    #= the passes write a record's array field as its elements (`recordFieldArrayElements`); each took the array's
       start, `{0.25, -0.75}` on a scalar: "the start of s.X[1] does not evaluate" (MSL Media's MoistAir,
       `state.X(start = reference_X)`). Each element its own value: the root of X[i]^2 = c its start's sign chooses =#
    local fields = Modelica"""
      record Composition
        Real X[2];
      end Composition;

      model FieldStarts "X = {0.25, -0.75} sqrt(1 + time)"
        Composition s(X(start = {0.25, -0.75}));
        Real x(start = 1, fixed = true);
      equation
        der(x) = -x;
        s.X[1] * s.X[1] = 0.0625 * (1 + time);
        s.X[2] * s.X[2] = 0.5625 * (1 + time);
      end FieldStarts;
      """
    local (sf, af, _) = _bothWaysOf(fields, "FieldStarts", ["s.X[1]", "s.X[2]"])
    @test af ≈ [0.25 * sqrt(2), -0.75 * sqrt(2)] rtol = 1e-6
    @test sf ≈ af rtol = 1e-6
  end

  @testset "pre() of an array as an argument, an array output into a row, an element of a negated array" begin
    #= `(r, state) = fibStep(pre(state))` in a when-equation: pre() of the array as the argument (the array of pre()
       of its elements), the array output into the array (MSL's random number generators: "the array state64 as a
       whole in an equation of scalars"); of two instances rolled into one class the argument and the target are a
       row of the rolled array, `state[\$c]` ("partial subscripts on uniformNoise1|uniformNoise2.state"). Fibonacci
       numbers: after 9 ticks r = 55 and the state {34, 55} from {0, 1}, r = 110 and {68, 110} from {0, 2} =#
    local steps = Modelica"""
      function fibStep
        input Integer s[2];
        output Real r;
        output Integer t[2];
      algorithm
        r := s[1] + s[2];
        t := {s[2], s[1] + s[2]};
      end fibStep;

      model Fib
        parameter Integer seed = 1;
        discrete Real r(start = 0, fixed = true);
        Integer state[2](start = {0, seed}, each fixed = true);
      equation
        when sample(0.1, 0.1) then
          (r, state) = fibStep(pre(state));
        end when;
      end Fib;

      model Fibs
        Fib a(seed = 1);
        Fib b(seed = 2);
      end Fibs;
      """
    local (s1, a1, _) = _bothWaysOf(steps, "Fib", ["r", "state[2]"]; tspan = (0.0, 0.95))
    @test a1 == [55.0, 55.0]
    @test s1 == a1
    local (s2, a2, _) = _bothWaysOf(steps, "Fibs", ["a.r", "b.r", "b.state[1]"]; tspan = (0.0, 0.95))
    @test a2 == [55.0, 110.0, 68.0]
    @test s2 == a2
    #= `n = not b` of Boolean arrays, element by element: `(not b)[i]` is the negation of the element (it was
       refused: "an element of not pwm.svPWM.fire_p", MSL PowerConverters' space vector PWM) =#
    local negations = Modelica"""
      model NotArray "n[1] until 0.25, n[2] until 0.75"
        Boolean b[2];
        Boolean n[2];
        Real x(start = 0, fixed = true);
      equation
        b[1] = time > 0.25;
        b[2] = time > 0.75;
        n = not b;
        der(x) = (if n[1] then 1 else 0) + (if n[2] then 2 else 0);
      end NotArray;
      """
    local (sn, an, _) = _bothWaysOf(negations, "NotArray", ["x", "n[2]"])
    @test an ≈ [1.75, 0.0] atol = 1e-6
    @test sn ≈ an atol = 1e-6
  end

  @testset "terminal(): the when-branches of the terminal event" begin
    #= terminal() is true at the terminal event, after the solve (MLS 8.6): the when-branches with it among their
       triggers run once at the final time, their values the last of the result (`_terminalEvent!`). It was refused:
       "terminal() in a when-condition" (MSL's ContinuousSignalExtrema, `when {u <= x, u >= x, terminal()}`).
       n counts the crossing at 0.5 and the end; `last` is the final time =#
    local endings = Modelica"""
      model TerminalCount
        Real x(start = 0, fixed = true);
        discrete Real last(start = -1, fixed = true);
        Integer n(start = 0, fixed = true);
      equation
        der(x) = 1;
        when {x > 0.5, terminal()} then
          n = pre(n) + 1;
        end when;
        when terminal() then
          last = time;
        end when;
      end TerminalCount;
      """
    local (st, at, sol) = _bothWaysOf(endings, "TerminalCount", ["n", "last", "x"]; tspan = (0.0, 0.8))
    @test at ≈ [2.0, 0.8, 0.8] atol = 1e-9
    @test st ≈ at atol = 1e-9
    #= not before the end: at 0.6 the crossing only =#
    local n::Vector{Float64} = OMBackend.getVariableValues(sol, "n")
    @test n[findlast(<=(0.6), sol.t)] == 1.0
  end

  @testset "the inner split of rolled arrays whose inner dimension is a middle part's" begin
    #= two trunks rolled into one class, each an array of stages: the stages' variables are arrays of the trunks'
       dimension and the stages' (`a|b.s.v`, [2, 3]), the stages' on the part `s`. The first stage's voltage is
       given, so of the three it alone is no state: the array is split along the stages (`splitInner!`). The slice
       is named on the part that carries the dimension, `a|b.s[1].v`, and its derivative after it,
       `\$DER.a|b.s[1].v`; with the inner subscript on the last part the part `s` kept its dimension ("its reference
       of more dimensions than 1": MSL's lines, plugs and terminal boxes). v2 = u (t - 1 + exp(-t)),
       v3 = u (t - 2 + (t + 2) exp(-t)) =#
    local trunks = Modelica"""
      model Stage "a resistor into a capacitor to ground"
        Real vin;
        Real v(start = 0);
        Real i;
      equation
        i = vin - v;
        der(v) = i;
      end Stage;

      model LedTrunk "three stages in a row; the first one's voltage is given"
        parameter Real u = 1;
        Stage s[3];
      initial equation
        s[2].v = 0;
        s[3].v = 0;
      equation
        s[1].v = u * time;
        s[2].vin = s[1].v;
        s[3].vin = s[2].v;
      end LedTrunk;

      model LedTrunks
        LedTrunk a(u = 1);
        LedTrunk b(u = 2);
      end LedTrunks;
      """
    local names::Vector{String} = ["a.s[2].v", "a.s[3].v", "b.s[2].v", "b.s[3].v", "b.s[1].vin"]
    local (ss, as, _) = _bothWaysOf(trunks, "LedTrunks", names; reltol = 1e-9, abstol = 1e-11)
    local (v2, v3) = (exp(-1), 3 * exp(-1) - 1)
    @test as ≈ [v2, v3, 2 * v2, 2 * v3, 4.0] atol = 1e-7
    @test ss ≈ as atol = 1e-7
  end

  @testset "an element of a reference whose parts select slices" begin
    #= a record array equation between slices, `mediums[:].state = statesFM[2:3]` (MSL Fluid's pipes), is written per
       element; the element i of each side is its slice's member, `mediums[i].state` and `statesFM[1 + i]`
       (`IndexReduction.sliceElementCref`). With the subscript appended the body read `mediums[:].state[i].p`:
       "the reference ... of more subscripts than its array's dimensions" =#
    local R::Module = OMBackend.Passes.IndexReduction
    local I::Module = OMBackend.OMJLIR
    local vector = I.T_ARRAY(I.T_REAL(), I.Dimension[I.DIM_SIZE(4)])
    local whole = I.ComponentRef(I.CrefPart[I.CrefPart(:mediums, I.Subscript[I.WHOLE()], vector),
                                            I.CrefPart(:state, I.Subscript[], I.T_REAL())])
    local range = I.RANGE(I.INTEGER(2), nothing, I.INTEGER(3), I.T_ARRAY(I.T_INTEGER(), I.Dimension[I.DIM_SIZE(2)]))
    local ranged = I.ComponentRef(I.CrefPart[I.CrefPart(:statesFM, I.Subscript[I.SLICE(range)], vector)])
    local at = (k::Int) -> I.Subscript[I.INDEX(I.INTEGER(k))]
    local iterator = I.Subscript[I.INDEX(I.CREF(I.ComponentRef(I.CrefPart[I.CrefPart(:i, I.Subscript[], I.T_INTEGER())]),
                                                I.T_INTEGER()))]
    @test R.sliceElementCref(whole, at(2)).name === Symbol("mediums[2].state")
    @test R.sliceElementCref(ranged, at(2)).name === Symbol("statesFM[3]")
    @test R.sliceElementCref(whole, iterator).name === Symbol("mediums[i].state")
    @test occursin(r"statesFM\[\(?1 \+ i\)?\]", String(R.sliceElementCref(ranged, iterator).name))
    #= a reference without slices: the subscripts after its last part's own, as before =#
    local plain = I.ComponentRef(I.CrefPart[I.CrefPart(:x, I.Subscript[], vector)])
    @test R.sliceElementCref(plain, at(3)).name === :x_3 || R.sliceElementCref(plain, at(3)).name === Symbol("x[3]")
  end

  @testset "The Jacobian by differences of a step per state" begin
    #= a pressure of 1e5 beside fractions of 0.01 whose logarithm a derivative takes: the solver's differences in a
       sparsity pattern took one step for all states from the norm of the state vector (0.61 here), the fraction
       went negative and `log` rejected every first step (MSL Media ReferenceAir MoistAir1 on the array code:
       "DomainError with -0.155"). The array code's own differences take a step per state (`ColoredJacobian`) =#
    local scaled = Modelica"""
      model ScaledStates
        Real p(start = 1.0e5, fixed = true);
        Real x[2](each start = 0.01, each fixed = true);
      equation
        der(p) = 1.0e5 - p + 10.0 * log(x[1]);
        for i in 1:2 loop
          der(x[i]) = (0.02 - x[i]) * (p / 1.0e5);
        end for;
      end ScaledStates;
      """
    local (ss, as, _) = _bothWaysOf(scaled, "ScaledStates", ["p", "x[1]", "x[2]"])
    @test isapprox(as, ss; rtol = 1e-5)
    @test 0.016 < as[2] < 0.017
    local G = OMBackend.CodeGeneration.ArrayODEGen
    #= columns that share no row in one group, greedy in the columns' order =#
    local band = G._SPARSE.sparse([1, 2, 2, 3, 3, 4], [1, 1, 2, 2, 3, 4], ones(6), 4, 4)
    @test G.columnGroups(band) == [[1, 3, 4], [2]]
    #= J = [log(u2) u1 / u2; 1 2 u2], each column by its state's own step =#
    local rhs! = function (du, u, p, t)
      du[1] = u[1] * log(u[2])
      du[2] = u[1] + u[2]^2
      return nothing
    end
    local dense = G._SPARSE.sparse([1, 2, 1, 2], [1, 1, 2, 2], ones(4), 2, 2)
    local jac = G.ColoredJacobian(G.StepRejectingRHS(rhs!), dense)
    local J = copy(dense)
    jac(J, [1.0e5, 0.01], nothing, 0.0)
    @test isapprox(Matrix(J), [log(0.01) 1.0e7; 1.0 0.02]; rtol = 1e-5)
    #= a state at the edge of the domain (its step below it leaves it): the difference to the point itself =#
    jac(J, [1.0e5, 1.0e-7], nothing, 0.0)
    @test all(isfinite, J.nzval)
    @test J[1, 2] > 1.0e9
    OMBackend.CodeGeneration._resetRHSError!()
  end

  @testset "A function with bound inputs as an argument" begin
    #= `twice(function line(k = k, w = w), 2.0)`: the array code had no form for a function with inputs bound
       ("expression PARTEVALFUNCTION": MSL Media moist air's T_phX, `solveOneNonlinearEquation(function
       f_nonlinear(p = p, h = h, X = X), ...)`, in the pipes of MSL Fluid). A closure of the free inputs
       (`_partialApplicationIRx`); the bound vector a slice by a structural range =#
    local curves = Modelica"""
      package Shapes
        partial function Curve
          input Real u;
          output Real y;
        end Curve;

        function line
          extends Curve;
          input Real k;
          input Real w[2];
        algorithm
          y := k * u + w[1] - w[2];
        end line;

        function twice
          input Curve f;
          input Real u;
          output Real y;
        protected
          Real once;
        algorithm
          once := f(u);
          y := f(once);
        end twice;

        function curved "as moist air's T_phX: one assignment, inlined into the equation by the arrays-kept passes"
          input Real k;
          input Real w[2];
          output Real y;
        algorithm
          y := twice(function line(k = k, w = w), 2.0);
        end curved;
      end Shapes;

      model BoundCurve "y[i] = k (2 k + c) + c, k = x[i] = exp(-i t), c = x[1] - x[2]"
        parameter Integer n = 2;
        Real x[3](each start = 1.0, each fixed = true);
        Real y[3];
      equation
        for i in 1:3 loop
          der(x[i]) = -i * x[i];
          y[i] = Shapes.curved(x[i], x[1:n]);
        end for;
      end BoundCurve;
      """
    local (sb, ab, _) = _bothWaysOf(curves, "BoundCurve", ["y[1]", "y[2]", "y[3]"]; reltol = 1e-8, abstol = 1e-10)
    local c::Float64 = exp(-1) - exp(-2)
    @test isapprox(ab, [exp(-i) * (2 * exp(-i) + c) + c for i in 1:3]; atol = 1e-6)
    @test maximum(abs.(sb .- ab)) < 1e-6
  end

  @testset "The array field of a record array of one record" begin
    #= `reader(states = {state})`: the field's elements were read as `{state.X}[1][i]`, the element of a picked
       element, and `state.X` is no variable (its elements are): "unknown variable volume.medium.state.X" (MSL
       Fluid's `heatTransfer.states = {medium.state}` of a volume of moist air). The element form picks through both
       subscripts, and a for-equation left reading `state.X[i]` is unrolled =#
    local records = Modelica"""
      record Mixture
        Real p;
        Real X[2];
      end Mixture;

      model Reader
        input Mixture states[1];
        Real weighted[2];
      equation
        for i in 1:2 loop
          weighted[i] = i * states[1].X[i];
        end for;
      end Reader;

      model OneRecordArray "weighted = {x, 2 (1 - x)}, x = exp(-t)"
        Mixture state;
        Reader reader(states = {state});
        Real x(start = 1.0, fixed = true);
      equation
        der(x) = -x;
        state.p = 1.0e5 * x;
        state.X = {x, 1.0 - x};
      end OneRecordArray;
      """
    local (sr, ar, _) = _bothWaysOf(records, "OneRecordArray", ["reader.weighted[1]", "reader.weighted[2]"])
    @test isapprox(ar, [exp(-1), 2 * (1 - exp(-1))]; atol = 1e-5)
    @test maximum(abs.(sr .- ar)) < 1e-5
  end

  @testset "A String parameter of rolled components, an index an initial algorithm gives" begin
    #= two probes of one class are rolled: their String parameter is an array read by element in the initial
       algorithm ("parameter first|second.substance of a type the array path does not pass"), and `ind`, fixed =
       false, is the frontend's structural value in the equations ("with fixed = false used structurally"): MSL
       Fluid's trace substance sensors in RoomCO2 =#
    local probes = Modelica"""
      function sameText
        input String a;
        input String b;
        output Boolean same;
      algorithm
        same := a == b;
      end sameText;

      model Probe
        parameter String substance = "b";
        input Real cs[1];
        Real c;
      protected
        parameter Integer ind(fixed = false);
      initial algorithm
        ind := -1;
        for i in 1:1 loop
          if sameText("b", substance) then
            ind := i;
          end if;
        end for;
        assert(ind > 0, "Substance '" + substance + "' is not present");
      equation
        c = cs[ind];
      end Probe;

      model TwoProbes "c = x and 2 x, x = exp(-t)"
        Real x(start = 1.0, fixed = true);
        Probe near(cs = {x});
        Probe far(cs = {2.0 * x});
      equation
        der(x) = -x;
      end TwoProbes;
      """
    local (sp, ap, _) = _bothWaysOf(probes, "TwoProbes", ["near.c", "far.c"])
    @test isapprox(ap, [exp(-1), 2 * exp(-1)]; atol = 1e-5)
    @test maximum(abs.(sp .- ap)) < 1e-5
  end

  @testset "Initial equations beyond the free states" begin
    #= inductors in series by pairs, each with `i = 0` at the start: one current is a state, the other's initial
       equation has no unknown left ("initialization over-determined (an initial equation without a free state)":
       the four lines in series of MSL's CompareLineTrunks). Out of the square system and checked at its solution, as
       OpenModelica keeps a model whose redundant initial equations are consistent =#
    local series = Modelica"""
      model SeriesInductors "ia[k] = ib[k] = k (1 - exp(-t / 3))"
        parameter Real L = 1.0;
        Real ia[2];
        Real ib[2];
        Real va[2];
        Real vb[2];
      equation
        for k in 1:2 loop
          L * der(ia[k]) = va[k];
          2.0 * L * der(ib[k]) = vb[k];
          ia[k] = ib[k];
          va[k] + vb[k] = k - ia[k];
        end for;
      initial equation
        for k in 1:2 loop
          ia[k] = 0.0;
          ib[k] = 0.0;
        end for;
      end SeriesInductors;
      """
    local (ss, as, _) = _bothWaysOf(series, "SeriesInductors", ["ia[1]", "ib[2]", "vb[2]"];
                                    reltol = 1e-8, abstol = 1e-10)
    @test isapprox(as, [1 - exp(-1 / 3), 2 * (1 - exp(-1 / 3)), 4 * exp(-1 / 3) / 3]; atol = 1e-6)
    @test maximum(abs.(ss .- as)) < 1e-6
    #= one that the others' solution does not satisfy ends the start with its equation =#
    local clash = Modelica"""
      model ClashingStarts
        parameter Real L = 1.0;
        Real ia[2];
        Real ib[2];
        Real va[2];
        Real vb[2];
      equation
        for k in 1:2 loop
          L * der(ia[k]) = va[k];
          2.0 * L * der(ib[k]) = vb[k];
          ia[k] = ib[k];
          va[k] + vb[k] = k - ia[k];
        end for;
      initial equation
        for k in 1:2 loop
          ia[k] = 0.0;
          ib[k] = 1.0;
        end for;
      end ClashingStarts;
      """
    local name::String = _translateProgram(clash, "ClashingStarts"; scalarized = false)
    if _arrayPathTaken(name)
      @test_throws Exception OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    else
      @test occursin("the initialization is over-determined: ia[k] = 0.0", OMBackend.scalarizedFallbackReason(name))
    end
  end

  @testset "An if-expression on the iterator of an array constructor" begin
    #= `{x[i] / (if i == 2 then 3 else 2) for i in 2:4}`: the if-expression was lifted to an if-equation of its own,
       which read `i` outside the constructor ("unknown variable i": the work of the pressure in MSL Fluid's pipes,
       `Wb_flows[2:n - 1]`). It stays inline, as one that reads a for-equation's iterator =#
    local weights = Modelica"""
      model WeightedConstructor "y = {x[2] / 3, x[3] / 2, x[4] / 2}, x[k] = exp(-k t)"
        Real x[4](each start = 1.0, each fixed = true);
        Real y[3];
      equation
        for k in 1:4 loop
          der(x[k]) = -k * x[k];
        end for;
        y = {x[i] / (if i == 2 then 3 else 2) for i in 2:4};
      end WeightedConstructor;
      """
    local (sw, aw, _) = _bothWaysOf(weights, "WeightedConstructor", ["y[1]", "y[2]", "y[3]"];
                                    reltol = 1e-8, abstol = 1e-10)
    @test isapprox(aw, [exp(-2) / 3, exp(-3) / 2, exp(-4) / 2]; atol = 1e-6)
    @test maximum(abs.(sw .- aw)) < 1e-6
  end

  @testset "A relation of time in a small algebraic loop" begin
    #= an if-equation on `time < 0.5` among the equations of a loop of two unknowns: the generated Newton of a loop
       family has no relations ("relations in an algebraic loop: 0 = if ... (time < 0.02)": the lightning current's
       source in MSL's LightningSegmentedTransmissionLine). Such a loop is solved as a large one (`blockSolve!`,
       which evaluates relations); a relation of time has one value while the loop is solved =#
    local sources = Modelica"""
      model TimeRelationLoop "a[k] = s / 1.25, s = 1 before 0.5 s and 3 after: x[k] = 1.6 k at 1 s"
        Real x[2](each start = 0.0, each fixed = true);
        Real a[2];
        Real b[2];
      equation
        for k in 1:2 loop
          der(x[k]) = k * a[k];
          if time < 0.5 then
            a[k] + 0.5 * b[k] = 1.0;
          else
            a[k] + 0.5 * b[k] = 3.0;
          end if;
          b[k] = 0.5 * a[k] + 0.0 * a[k] ^ 3;
        end for;
      end TimeRelationLoop;
      """
    local (st, at, _) = _bothWaysOf(sources, "TimeRelationLoop", ["x[1]", "x[2]", "a[2]", "b[1]"];
                                    reltol = 1e-8, abstol = 1e-10)
    @test isapprox(at, [1.6, 3.2, 2.4, 1.2]; atol = 1e-6)
    @test maximum(abs.(st .- at)) < 1e-6
  end

  @testset "Folds: the size and the scalar product of vector literals" begin
    #= the element view of a for-equation differentiated has `{dh_dX} * {der_X}` and `if size({Xi[1]}, 1) == 2 then
       ... else ...` where the array form has the terms and the branch ("a for-equation whose derivative in array
       form is not the element's": MSL Fluid's ducts of moist air, RoomCO2WithControls); `sameEquation` compares
       them folded =#
    local I = OMBackend.OMJLIR
    local R = OMBackend.Passes.IndexReduction
    local real = (name::Symbol) -> I.CREF(I.ComponentRef(I.CrefPart[I.CrefPart(name, I.Subscript[], I.T_REAL())]),
                                          I.T_REAL())
    local vector = I.T_ARRAY(I.T_REAL(), I.Dimension[I.DIM_SIZE(2)])
    local a = I.ARRAY(I.Expression[real(:a), real(:b)], vector)
    local b = I.ARRAY(I.Expression[real(:c), real(:d)], vector)
    local product = R.foldTree(I.BINARY(a, I.SCALAR_PRODUCT, b, I.T_REAL()))
    @test product isa I.BINARY && product.op === I.ADD
    @test product.exp1 == I.BINARY(real(:a), I.MUL, real(:c), I.T_REAL())
    @test product.exp2 == I.BINARY(real(:b), I.MUL, real(:d), I.T_REAL())
    @test R.foldTree(I.SIZE(a, I.INTEGER(1))) == I.INTEGER(2)
    local chosen = (n::Int) -> R.foldTree(I.IF_EXPRESSION(I.RELATION(I.SIZE(a, I.INTEGER(1)), I.EQUAL, I.INTEGER(n), 0),
                                                           real(:a), real(:b), I.T_REAL()))
    @test chosen(2) == real(:a)
    @test chosen(3) == real(:b)
    #= a size along another dimension, or of what is no literal, stays =#
    @test R.foldTree(I.SIZE(a, I.INTEGER(2))) isa I.SIZE
    @test R.foldTree(I.SIZE(real(:a), I.INTEGER(1))) isa I.SIZE
    #= an array literal subscripted by Integer literals: its element (`literalElement`; the derivative of a constant
       rotation matrix written out per element in array form, `{{0.0, 0.0}, {0.0, 0.0}}[1, 2] * w`, then `0 * w`),
       fewer subscripts an inner literal; a subscript out of range, or not a literal, stays =#
    local matrixType = I.T_ARRAY(I.T_REAL(), I.Dimension[I.DIM_SIZE(2), I.DIM_SIZE(2)])
    local matrix = I.ARRAY(I.Expression[a, b], matrixType)
    local at = (e::I.Expression, k::Vector{I.Expression}) -> I.SUBSCRIPTED(e, I.Subscript[I.INDEX(x) for x in k], I.T_REAL())
    @test R.foldTree(at(matrix, I.Expression[I.INTEGER(2), I.INTEGER(1)])) == real(:c)
    @test R.foldTree(at(matrix, I.Expression[I.INTEGER(1)])) == a
    local zeroMatrix = I.ARRAY(I.Expression[I.ARRAY(I.Expression[I.REAL(0.0), I.REAL(0.0)], vector) for _ in 1:2], matrixType)
    @test R.foldTree(I.BINARY(at(zeroMatrix, I.Expression[I.INTEGER(1), I.INTEGER(2)]), I.MUL, real(:w), I.T_REAL())) == I.REAL(0.0)
    @test R.foldTree(at(matrix, I.Expression[I.INTEGER(3), I.INTEGER(1)])) isa I.SUBSCRIPTED
    @test R.foldTree(at(matrix, I.Expression[real(:i)])) isa I.SUBSCRIPTED
  end

  @testset "Tearing: the cascade of a candidate iteration variable" begin
    #= instances e1 (v1, v2), e2 (v2, v3), e3 (v1, v2, v3): v1 known, e1 solves v2, then e2 v3: three unknowns
       (`_tearCascade`, `_tearBlock`'s lookahead); where e1 solves nothing, v1 alone. The state passed in is not
       changed =#
    local D = OMBackend.SimCode.Domains
    local reads::Vector{Vector{Tuple{Int, Int}}} = [[(1, 1), (2, 1)], [(2, 1), (3, 1)], [(1, 1), (2, 1), (3, 1)]]
    local readers::Vector{Vector{Int}} = [[1, 3], [1, 2, 3], [2, 3]]
    local known::BitVector = falses(3); local used::BitVector = falses(3); local open::Vector{Int} = [2, 2, 3]
    @test D._tearCascade(1, known, used, open, reads, readers, (e::Int, v::Int, access::Int) -> true) == 3
    @test D._tearCascade(1, known, used, open, reads, readers, (e::Int, v::Int, access::Int) -> e != 1) == 1
    @test known == falses(3) && used == falses(3) && open == [2, 2, 3]
  end

  @testset "Tearing: greedily and by cascades; a block piecewise in its unknowns greedily" begin
    #= the instances above torn greedily (`_tearOrder`): v2 unlocks e1 and e2, the most; e1 solves v1, e2 v3, e3 is
       the residual. By cascades all three reach three unknowns: v2, the first, the same =#
    local D = OMBackend.SimCode.Domains
    local reads::Vector{Vector{Tuple{Int, Int}}} = [[(1, 1), (2, 1)], [(2, 1), (3, 1)], [(1, 1), (2, 1), (3, 1)]]
    local readers::Vector{Vector{Int}} = [[1, 3], [1, 2, 3], [2, 3]]
    local all3::Function = (e::Int, v::Int, access::Int) -> true
    local (order, tears, used) = D._tearOrder(reads, readers, all3, 1)
    @test order == [(1, 1, 1), (2, 3, 1)] && tears == [2] && used == BitVector([true, true, false])
    @test D._tearOrder(reads, readers, all3, 3)[2] == [2]
    #= piecewise in the unknowns (x), with which `_tearBlock` tears greedily only (MSL DifferentialAmplifier's
       op-amp limiter): an if-expression whose condition reads x under noEvent, abs of x. Not piecewise in them: a
       smooth function, an if-expression on a mode held through the solve (a thyristor's `off`), a relation with
       events (held), abs of what is no unknown =#
    local x::D.Ref = D.Ref("x", D.IR[], false); local y::D.Ref = D.Ref("y", D.IR[], false)
    local off::D.Ref = D.Ref("off", D.IR[], false)
    local readsX::Function = (r::D.Ref) -> r.name == "x"
    @test D._piecewiseInUnknowns(D.Op(:if, D.IR[D.Op(:>, D.IR[x, D.Lit(0.0)]), x, y]), readsX)
    @test D._piecewiseInUnknowns(D.Op(:+, D.IR[y, D.Op(:abs, D.IR[x])]), readsX)
    @test !D._piecewiseInUnknowns(D.Op(:*, D.IR[x, D.Op(:sin, D.IR[y])]), readsX)
    @test !D._piecewiseInUnknowns(D.Op(:if, D.IR[off, D.Op(:*, D.IR[x, D.Lit(1.0e-5)]), x]), readsX)
    @test !D._piecewiseInUnknowns(D.Op(:if, D.IR[D.Rel(:<, x, D.Lit(0.0), 1), x, y]), readsX)
    @test !D._piecewiseInUnknowns(D.Op(:abs, D.IR[y]), readsX)
  end

  @testset "actualStream without events" begin
    #= `smooth(0, m * (if m > 0 then a else b))` (actualStream, both forms): OpenModelica makes no event in it. The
       array code computes flows from interpolated states; at the flow's hysteresis their round-off (1e-10 kg/s)
       switched the relation at every step (MSL Fluid's RoomCO2WithControls after its fan started at 32400 s). The
       condition is a `noEvent` on the arrays-kept route: no relations in the array code =#
    local streams = Modelica"""
      model ReversingStreams "the enthalpy flows of flows that reverse"
        Real x[2](each start = 1.0, each fixed = true);
        Real m[2];
        Real H[2];
        Real G[2];
      equation
        for k in 1:2 loop
          m[k] = cos(6.283185307179586 * k * time);
          H[k] = smooth(0, m[k] * (if m[k] > 0 then 2.0 else x[k]));
          G[k] = smooth(0, if m[k] >= 0.0 then m[k] * x[k] else 0.0);
          der(x[k]) = H[k] + G[k] - x[k];
        end for;
      end ReversingStreams;
      """
    #= reltol 1e-9: the array route crosses the 6 kinks in t without events (smooth(0, ...)); at reltol 1e-8 its error
       there depends on the step path, 2e-8 to 1.1e-6 against an exact reference (4.3e-10 at reltol 1e-9) =#
    local (ss, as, _) = _bothWaysOf(streams, "ReversingStreams", ["x[1]", "x[2]"]; reltol = 1e-9, abstol = 1e-11)
    @test OMBackend.CodeGeneration.ArrayODEGen.LAST_STATS[].relations == 0
    @test maximum(abs.(ss .- as)) < 1e-6
  end

  @testset "The Jacobian's step at the scale of a pressure difference" begin
    #= pressures of 1e5 whose flow saturates beyond 0.01 Pa of their difference: a step of 6e-6 relative (0.6 Pa)
       sees the flow's slope as 1.7e3 for 1e5, Newton left the solution (m = -1000 for 1). The step 1.5e-8 relative
       (1.5e-3 Pa) sees it (MSL Fluid's RoomCO2WithControls: 40000 steps for 2 s after its fan started) =#
    local lines = Modelica"""
      model StiffLines "two pressures of 1e5 per line"
        Real p[2](each start = 1.0e5, each fixed = true);
        Real q[2](each start = 1.0e5, each fixed = true);
        Real m[2];
      equation
        for k in 1:2 loop
          m[k] = 1.0e3 * tanh(1.0e2 * (p[k] - q[k]));
          der(p[k]) = k - m[k];
          der(q[k]) = m[k] - k;
        end for;
      end StiffLines;
      """
    local (ss, as, _) = _bothWaysOf(lines, "StiffLines", ["m[1]", "m[2]", "p[2]"])
    @test isapprox(as, [1.0, 2.0, 1.0e5]; rtol = 1e-5)
    @test isapprox(as, ss; rtol = 1e-6)
  end

  @testset "A table's output read after the solve" begin
    #= an MSL table keeps the event interval of its last getNextTimeEvent call: after the solve the last one, and a
       read of the solution gave the last segment's value at every time (MSL Fluid's RoomCO2WithControls: the number
       of people 0 all day, 42 of 606 variables off omc). A read makes the run's calls again up to its time =#
    local name = _arrayPathTranslate("ArrayPath.TableSegments"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    @test isapprox(_values(sol, ["x[1]", "x[2]"]), [3.8, 7.6]; rtol = 1e-6)
    @test [sol(t; idxs = :y) for t in (0.2, 0.45, 0.8, 0.1)] == [0.0, 10.0, 2.0, 0.0]
    #= at the events the left and the right value =#
    local y::Vector{Float64} = OMBackend.getVariableValues(sol, "y")
    local steps::Vector{Tuple{Float64, Float64, Float64}} =
      [(sol.t[i], y[i - 1], y[i]) for i in 2:length(sol.t) if sol.t[i] == sol.t[i - 1] && sol.t[i] > 0]
    @test [(round(t; digits = 6), l, r) for (t, l, r) in steps] == [(0.3, 0.0, 10.0), (0.6, 10.0, 2.0)]
  end

  @testset "The array code's abstol: states far below 1e-6" begin
    #= the array code controls the error of its states only: a state of 1e-8 under the solvers' abstol of 1e-6 was
       free to be 100% off (MSL Fluid's RoomCO2WithControls: the ducts' trace substance 1 to 5% off omc; IMC_DOL's
       currents before its switch closes). The default is 1e-10; a caller's abstol is taken =#
    local small = Modelica"""
      model SmallStates "x[k] = 1e-8 k exp(-k (t + (1 - cos(20 t)) / 20))"
        Real x[2](start = {1.0e-8, 2.0e-8}, each fixed = true);
      equation
        for k in 1:2 loop
          der(x[k]) = -k * (1 + sin(20 * time)) * x[k];
        end for;
      end SmallStates;
      """
    local name::String = _translateProgram(small, "SmallStates"; scalarized = false)
    @test _arrayPathTaken(name)
    local exact::Vector{Float64} = [k * 1.0e-8 * exp(-k * (1 + (1 - cos(20.0)) / 20)) for k in 1:2]
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    @test isapprox(_values(sol, ["x[1]", "x[2]"]), exact; rtol = 0.02)
    #= 1% at 1e-10 (the states 1e-8); at 1e-6 x[2] was 1.2e-8 for 2.6e-9. A caller's abstol is the solver's =#
    local loose = OMBackend.simulateModel(name; tspan = (0.0, 1.0), abstol = 1e-6)
    @test length(loose.t) < length(sol.t)
  end

  @testset "A parameter bound to one with fixed = false" begin
    #= `k = 3 s`, s (fixed = false) from an initial equation: k is computed by the initialization too (OpenModelica's
       initial system). The array code refused ("a parameter binding depends on a parameter with fixed = false":
       MSL's noise blocks, `actualGlobalSeed = if useGlobalSeed then globalSeed.seed else 0`) =#
    local free = Modelica"""
      model FreeDependent "s = 2, k = 3 s, n = 5, m = n if use: x[i] = (i k + m) t"
        parameter Boolean use = true;
        parameter Real s(fixed = false, start = 0.5);
        parameter Real k = 3 * s;
        parameter Integer n(fixed = false, start = 0);
        parameter Integer m = if use then n else 0;
        Real x[2](each start = 0, each fixed = true);
      initial equation
        s = 2;
        n = 5;
      equation
        for i in 1:2 loop
          der(x[i]) = i * k + m;
        end for;
      end FreeDependent;
      """
    local (sf, af, _) = _bothWaysOf(free, "FreeDependent", ["x[1]", "x[2]"])
    @test isapprox(af, [11.0, 17.0]; rtol = 1e-8)
    @test isapprox(sf, af; rtol = 1e-8)
  end

  @testset "The initial system solved in its blocks" begin
    #= m and H the states, without start values (0); level and T fixed: level = 0.5 gives m, then T = T0 gives H.
       Solved as one system from m = H = 0, T = H / (4 m) was NaN there: "the initialization cannot hold" (MSL
       FluidHeatFlow TestOpenTank). OpenModelica sorts its initial system into blocks =#
    local sorted = Modelica"""
      model SortedInit "m = exp(-t / 10), level = m / 2, T = H / (4 m) = 313.15"
        Real m;
        Real H;
        Real level(start = 0.5, fixed = true);
        Real T(start = 313.15, fixed = true);
        Real x[2](each start = 0, each fixed = true);
      equation
        level = m / 2;
        T = H / (4 * m);
        der(m) = -0.1 * m;
        der(H) = -0.1 * H;
        for i in 1:2 loop
          der(x[i]) = i * T;
        end for;
      end SortedInit;
      """
    local (ss, as, _) = _bothWaysOf(sorted, "SortedInit", ["level", "T", "x[2]"])
    @test isapprox(as, [0.5 * exp(-0.1), 313.15, 2 * 313.15]; rtol = 1e-6)
    @test isapprox(ss, as; rtol = 1e-6)
  end

  @testset "A hidden linear constraint through sum() of an array" begin
    #= a zero-sequence inductor (`3 i0 = sum(i)`, v0 = L0 der(i0)) on a star point that is not grounded (sum(i) = 0):
       i0 = 0, no state. The element view writes `sum(i)` as a call the linear rows did not read: of index 1 with i0 a
       state, the loop of the voltages singular (MSL FundamentalWave's polyphase machines). OpenModelica differentiates
       the constraint =#
    local zero = Modelica"""
      model ZeroSequence "u = k t, vs = mean(u) = 2 t, i = (u - vs) / R: x = {-1/2, 0}"
        parameter Real R = 1;
        parameter Real L0 = 0.01;
        Real i[3];
        Real i0;
        Real v0;
        Real vs;
        Real x[2](each start = 0, each fixed = true);
      equation
        for k in 1:3 loop
          R * i[k] + v0 = k * time - vs;
        end for;
        3 * i0 = sum(i);
        v0 = L0 * der(i0);
        sum(i) = 0;
        for k in 1:2 loop
          der(x[k]) = i[k];
        end for;
      end ZeroSequence;
      """
    local (sz, az, _) = _bothWaysOf(zero, "ZeroSequence", ["x[1]", "x[2]", "i0"])
    @test isapprox(az, [-0.5, 0.0, 0.0]; atol = 1e-6)
    @test isapprox(sz, az; atol = 1e-6)
  end

  @testset "A fixed = false parameter found through the equations" begin
    #= a = 0 at the start fixes c (a = 10 - k x[1], k = c): an initial equation of an algebraic variable whose free
       parameter only the equations read. The initial matching followed the algebraic variable to the states only:
       `a = 0` counted as over-determined (MSL InitSpringConstant: rev.a = 0 fixes the spring's c) =#
    local free = Modelica"""
      model FreeThroughEquations "k = c = 20 makes a = 0: x stays at 0.5"
        parameter Real c(fixed = false, start = 1);
        parameter Real k = c;
        Real a;
        Real x[2](each start = 0.5, each fixed = true);
      initial equation
        a = 0;
      equation
        a = 10 - k * x[1];
        for i in 1:2 loop
          der(x[i]) = i * a;
        end for;
      end FreeThroughEquations;
      """
    local (sf, af, _) = _bothWaysOf(free, "FreeThroughEquations", ["x[1]", "x[2]", "a"])
    @test isapprox(af, [0.5, 0.5, 0.0]; atol = 1e-8)
    @test isapprox(sf, af; atol = 1e-8)
  end

  @testset "An if-equation on a parameter with fixed = false" begin
    #= `if big then`, big from an initial algorithm: its value is the initialization's, so the condition is evaluated
       when the model runs. The array code evaluated it at the translation (a parameter's condition) and refused:
       "parameter big with fixed = false used structurally" (MSL DCPowerSupply's `if cp then`) =#
    local cond = Modelica"""
      model FreeCondition "big = P0 > 1 = true: y = 2, x = {2, 4} at t = 1"
        parameter Real P0 = 2;
        parameter Boolean big(fixed = false);
        Real y;
        Real x[2](each start = 0, each fixed = true);
      initial algorithm
        big := P0 > 1;
      equation
        if big then
          y = 2;
        else
          y = 1;
        end if;
        for i in 1:2 loop
          der(x[i]) = i * y;
        end for;
      end FreeCondition;
      """
    local (sc, ac, _) = _bothWaysOf(cond, "FreeCondition", ["x[1]", "x[2]", "y"])
    @test isapprox(ac, [2.0, 4.0, 2.0]; rtol = 1e-8)
    @test isapprox(sc, ac; rtol = 1e-8)
  end

  @testset "An initial equation reads a parameter an initial assignment computes" begin
    #= `c = 2*p0` is an initial assignment of parameters (an initial algorithm), `x[1] = c` an initial equation that
       reads c: the assignment runs before the initial system. It ran after it, so the array code refused: "initial
       equation reads c, which an initial algorithm assigns" (MSL Engine1b_analytic's `prismatic.length`, bound to
       `rod1.rodLength(fixed = false)`, read by `positiveBranch = selectBranch(length, ...)`) =#
    local assign = Modelica"""
      model EarlyAssign "c = 4: x = 4 exp(-i t)"
        parameter Real p0 = 2;
        parameter Real c(fixed = false);
        Real x[2](each start = 0);
      initial equation
        c = 2*p0;
        x[1] = c;
        x[2] = c;
      equation
        for i in 1:2 loop
          der(x[i]) = -i*x[i];
        end for;
      end EarlyAssign;
      """
    local (sa, aa, _) = _bothWaysOf(assign, "EarlyAssign", ["x[1]", "x[2]"])
    @test isapprox(aa, 4 .* exp.(-[1.0, 2.0]); rtol = 1e-5)
    @test isapprox(sa, aa; rtol = 1e-5)
    #= Engine1b's form: a Boolean parameter with fixed = false from a function of the assigned parameter (through a
       binding) and a variable: pos = 6 > 2, x = exp(-i t) =#
    local branch = Modelica"""
      model EarlyBranch "L = 6 > y = 2: pos, x = exp(-i t)"
        function pick
          input Real L;
          input Real y;
          output Boolean pos;
        algorithm
          pos := L > y;
        end pick;
        parameter Real r = 3;
        parameter Real L(fixed = false, start = 1);
        parameter Real len = L;
        parameter Boolean pos(fixed = false);
        Real x[2](each start = 1, each fixed = true);
        Real y;
      initial equation
        L = 2*r;
        pos = pick(len, y);
      equation
        y = x[1] + x[2];
        for i in 1:2 loop
          der(x[i]) = if pos then -i*x[i] else i*x[i];
        end for;
      end EarlyBranch;
      """
    local (sb, ab, _) = _bothWaysOf(branch, "EarlyBranch", ["x[1]", "x[2]"])
    @test isapprox(ab, exp.(-[1.0, 2.0]); rtol = 1e-5)
    @test isapprox(sb, ab; rtol = 1e-5)
  end

  @testset "A parameter with fixed = false as a divisor in a torn loop" begin
    #= a Boolean with fixed = false in a large algebraic loop (more than NEWTON_BLOCK_SIZE instances, torn), d[i] in
       both branches as Engine1b's `distance = -k1 + (if positiveBranch then k2 else -k2)`: solving for d[i] divides by
       `if pos then 1 else -1`, which the tearing evaluated at the translation with pos at its start, 0.0 (TypeError:
       non-boolean used in boolean context). pos = true: d = x / 1.5, der(x) = -x =#
    local torn = Modelica"""
      model TornBranch "pos = x > 0: d = x / 1.5, x = exp(-t)"
        parameter Boolean pos(fixed = false);
        Real x(start = 1, fixed = true);
        Real s[5];
        Real d[5];
      initial equation
        pos = x > 0;
      equation
        for i in 1:5 loop
          x = s[i] + (if pos then d[i] else -d[i]);
          s[i] = 0.1 * sum(d);
        end for;
        der(x) = -1.5 * d[1];
      end TornBranch;
      """
    local (st, at, _) = _bothWaysOf(torn, "TornBranch", ["x", "d[1]"])
    @test isapprox(at, [exp(-1.0), exp(-1.0) / 1.5]; rtol = 1e-5)
    @test isapprox(st, at; rtol = 1e-5)
  end

  @testset "A loop singular in its mode but consistent" begin
    #= MSL OpenTank's ports above the level, both closed (TanksWithOverflow's overflow pipe at the start): `m = 0`
       from either end, the pressures free up to the pipe's equations. The loop's Jacobian is singular; the solver
       refused the iterate of least-norm steps, so the initialization failed ("the algebraic loop 1 of 14 unknowns
       has no solution from its start values"). Its residuals are zero to round-off: accepted, as OpenModelica's
       solver does. Open from t = 10: m = -0.1 / sqrt(0.1 + 1e-6), x(20) = 10 m; 14 unknowns: a large block. =#
    local program = Modelica"""
      model ModeSingular "closed ends until t = 10: m = 0; then m = -0.1 / sqrt(0.1 + 1e-6)"
        parameter Real p0 = 1;
        parameter Real G = 1;
        Real p[11](each start = 1);
        Real m;
        Real sA;
        Real sB;
        Real x(start = 0, fixed = true);
        Real y(start = 0, fixed = true);
        Boolean opened(start = false);
      equation
        der(y) = 1;
        opened = y > 10;
        for i in 1:10 loop
          m = (p[i] - p[i + 1] - G / 10) / sqrt(abs(p[i] - p[i + 1] - G / 10) + 1e-6);
        end for;
        if opened then
          p[1] = p0;
          sA = m;
        else
          m = 0;
          sA = p[1] - p0;
        end if;
        if opened then
          p[11] = p0;
          sB = -m;
        else
          -m = 0;
          sB = p[11] - p0;
        end if;
        der(x) = m;
      end ModeSingular;
      """
    local name::String = _translateProgram(program, "ModeSingular"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 20.0))
    @test sol.retcode == ReturnCode.Success
    local x::Vector{Float64} = OMBackend.getVariableValues(sol, "x")
    local dp::Vector{Float64} = OMBackend.getVariableValues(sol, "p[1]") .- OMBackend.getVariableValues(sol, "p[11]")
    local mOpen::Float64 = -0.1 / sqrt(0.1 + 1e-6)
    @test isapprox(x[end], 10 * mOpen; rtol = 1e-6)
    @test isapprox(dp[1], 1.0; rtol = 1e-8)
    @test isapprox(OMBackend.getVariableValues(sol, "m")[end], mOpen; rtol = 1e-8)
  end

  @testset "Two relations of one threshold within their hysteresis bands" begin
    #= MSL OpenTank's port reached by the level (TanksWithOverflow at 2025 s): `regular = x >= 6`, and the port's
       `s > 0` with s = 0.48 (x - 6) while closed. At the event of `s > 0` (its band H), x - 6 = 2.1 H, within the band
       of `x >= 6` (7 H): `inFlow` true gives s = -0.16, false again gives s > 0, every pass; the run stopped
       ("the event iteration did not settle"). The relations within their bands are searched too: regular.
       x(10) = 6 + 5 * 0.1 =#
    local program = Modelica"""
      model BandedSwitch "x rises at 0.2 to 6 (t = 5), then at 0.1: x(10) = 6.5"
        Real x(start = 5, fixed = true);
        Real s(start = 20);
        Boolean regular;
        Boolean inFlow(start = false);
      equation
        regular = x >= 6;
        inFlow = not regular and s > 0;
        if regular then
          s = x - 6;
        elseif inFlow then
          s = -0.16;
        else
          s = 0.48 * (x - 6);
        end if;
        der(x) = if regular then 0.1 else 0.2;
      end BandedSwitch;
      """
    local name::String = _translateProgram(program, "BandedSwitch"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 10.0))
    @test sol.retcode == ReturnCode.Success
    @test isapprox(OMBackend.getVariableValues(sol, "x")[end], 6.5; rtol = 1e-6)
  end

  @testset "A clocked partition of arrays read by their elements" begin
    #= MSL Clocked's AssignClockVectorized, SampleVectorizedAndClocked, Sample2, UpSample with the arrays kept: an
       element reference `y[2]` (no variable of that name) did not tie its equation to the array's partition ("a
       clocked partition whose clock cannot be inferred"); the elements' equations `u[1] = ...`, `u[2] = ...` were
       matched to whole variables ("a clocked equation that is not an assignment of one of its partition's
       variables"); the when body's `y := u` and `ys := pre({time, 2 time})` assigned arrays as a whole ("the array
       y as a whole in an equation of scalars"). Ticks at 0, 0.1, ..., 0.9: y[1] = 10, y[2] = a = 20, ys[2] = 1.8,
       w = 21.8 (OpenModelica 1.27.1's values) =#
    local program = Modelica"""
      model ClockedVector "a clocked when-clause over arrays, elements read by equations; Clock(0.1)"
        Clock c = Clock(0.1);
        Real u[2];
        Real y[2](each start = 0);
        Real ys[2];
        Real a;
        Real w;
      equation
        when c then
          y = u;
        end when;
        u[1] = previous(y[1]) + 1;
        u[2] = 2 * u[1];
        a = y[2];
        ys = sample({time, 2 * time}, c);
        w = hold(a) + hold(ys[2]);
      end ClockedVector;
      """
    local name::String = _translateProgram(program, "ClockedVector"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 0.95))
    @test sol.retcode == ReturnCode.Success
    local at = (v::String) -> OMBackend.getVariableValues(sol, v)[end]
    @test isapprox([at("y[1]"), at("y[2]"), at("a"), at("ys[2]"), at("w")], [10.0, 20.0, 20.0, 1.8, 21.8]; rtol = 1e-9)
  end

  @testset "When-equations assigning arrays: if-equations, if-expressions of arrays, slices of pre(), tuples" begin
    #= Refused for array code ("a when-statement ASSIGN", "a when-statement assigning the array ... the value ...",
       "a when-statement assigning (r, s)"): an if-equation in a when-equation (MSL SignalExtrema, PreisachEverett's
       `aSav = cat(1, {alpha}, pre(aSav[1:end - 1]))`; an array assigned there is discrete, `whenAssignedDiscrete!`),
       a clocked buffer's if-expression of arrays with a slice of previous() (MSL Clocked's FIR filters), a function of
       two outputs in a subsampled partition (`(r, s) := if tick then f(...) else (pre(r), pre(s))`, MSL Clocked's
       Noise). Ticks at 0.1, ..., 0.9 (the sample()), 0, 0.1, ..., 0.9 (Clock(0.1)), 0, 0.2, ..., 0.8 (subSample 2) =#
    local program = Modelica"""
      model ShiftOnTicks "an if-equation in a when-equation shifts an array"
        Real buf[3](each start = 0, each fixed = true);
        Integer k(start = 0, fixed = true);
        Real tk(start = 0, fixed = true);
      equation
        when sample(0.1, 0.1) then
          k = pre(k) + 1;
          if pre(k) >= 2 then
            buf = cat(1, {k}, pre(buf[1:2]));
            tk = time;
          else
            buf = pre(buf);
            tk = pre(tk);
          end if;
        end when;
      end ShiftOnTicks;

      model ClockedShift "a clocked buffer: an if-expression of arrays, a slice of previous()"
        Real u = sample(time, Clock(0.1));
        Real b[3](each start = 0);
        Boolean first(start = true);
        Real y = hold(b[3]);
      equation
        first = false;
        b = if previous(first) then {u, u, u} else cat(1, {u}, previous(b[1:2]));
      end ClockedShift;

      model SubsampledTuple "a function of two outputs in a subsampled partition"
        function twice
          input Real s;
          output Real r;
          output Real sNext;
        algorithm
          r := 2 * s;
          sNext := s + 1;
        end twice;
        Real v = sample(time, Clock(0.1));
        Real vs = subSample(v, 2);
        Real s(start = 1);
        Real r(start = 0);
        Real y = hold(r);
      equation
        (r, s) = twice(previous(s) + vs);
      end SubsampledTuple;
      """
    #= buf shifted at the ticks from the third on: {9, 8, 7} after the ninth =#
    local (ss, as, _) = _bothWaysOf(program, "ShiftOnTicks", ["k", "tk", "buf[1]", "buf[2]", "buf[3]"];
                                    tspan = (0.0, 0.95))
    @test as ≈ [9.0, 0.9, 9.0, 8.0, 7.0] atol = 1e-9
    @test ss ≈ as atol = 1e-9
    #= b = {u, u, u} at the first tick, then u in front: {0.9, 0.8, 0.7} after the tick at 0.9 =#
    local (sb, ab, _) = _bothWaysOf(program, "ClockedShift", ["b[1]", "b[2]", "b[3]", "y"]; tspan = (0.0, 0.95))
    @test ab ≈ [0.9, 0.8, 0.7, 0.7] atol = 1e-9
    @test sb ≈ ab atol = 1e-9
    #= r = 2 (s + v), s + v + 1 the next s: (1, 0) -> r = 2, s = 2; (2, 0.2) -> 4.4, 3.2; (3.2, 0.4) -> 7.2, 4.6;
       (4.6, 0.6) -> 10.4, 6.2; (6.2, 0.8) -> 14, 8 =#
    local (st, at, _) = _bothWaysOf(program, "SubsampledTuple", ["r", "s", "y"]; tspan = (0.0, 0.95))
    @test at ≈ [14.0, 8.0, 14.0] atol = 1e-9
    @test st ≈ at atol = 1e-9
  end

  @testset "Initial equations on derivatives of variables that are no states" begin
    #= Refused for array code ("der of non-state ..."): an initial equation der(v) = c of an algebraic v. MSL
       Rotational's InitializeFlange sets `der(flange.phi) = w_start` on a flange whose angle is an alias of a state
       (here negated too); FundamentalWave's IMC_Initialize starts steady, `der(airGap.V_mrr.re) = 0` of an algebraic
       variable. v's equation is differentiated down to the states (`initialDerivatives!`), the initial equation
       reads the derivative. x'' = 1 - x, x(0) = 0.5: der(-x) = -2 gives x'(0) = 2; der(x + 2 x') = 0 gives
       x'(0) = 2 (x(0) - 1) = -1. x = 1 - 0.5 cos t + x'(0) sin t =#
    local program = Modelica"""
      model AliasStart "der() of a negated alias of a state in an initial equation"
        Real x(start = 0.5, fixed = true);
        Real w;
        Real flangePhi;
      initial equation
        der(flangePhi) = -2;
      equation
        flangePhi = -x;
        der(x) = w;
        der(w) = 1 - x;
      end AliasStart;

      model SteadyAlgebraicStart "der() of an algebraic variable in an initial equation: a steady start"
        Real x(start = 0.5, fixed = true);
        Real w;
        Real y;
      initial equation
        der(y) = 0;
      equation
        y = x + 2 * w;
        der(x) = w;
        der(w) = 1 - x;
      end SteadyAlgebraicStart;
      """
    for (model, w0) in (("AliasStart", 2.0), ("SteadyAlgebraicStart", -1.0))
      local (s, a, _) = _bothWaysOf(program, model, ["x", "w"])
      @test a ≈ [1 - 0.5 * cos(1.0) + w0 * sin(1.0), 0.5 * sin(1.0) + w0 * cos(1.0)] rtol = 1e-5
      @test s ≈ a rtol = 1e-5
    end
  end

  @testset "A friction element sticking again: one slot per relation" begin
    #= MSL Rotational's Clutch (PartialFriction) in MSL's Friction example: the discrete cluster inlines startForward
       and startBackward into locked and mode, so `sa < -tau0max` stood three times. Each copy had its own relation
       slot; the copy in locked, skipped by a short-circuit while the clutch slid backward, kept `true` from the
       break-away at 0.0068. At the sticking (0.109) locked stayed false in every combination the event iteration
       tried: "did not settle", the simulation stopped. Equal relations of a template now share their slot.
       omc: w3(1) = 94.73865, w1(1) = 95.29922. =#
    local program = Modelica"""
      model ClutchSticking "MSL Clutch's friction equations between two inertias, a spring-damper load, a sine torque"
        parameter Real J3 = 1;
        parameter Real J2 = 0.05;
        parameter Real J1 = 1;
        parameter Real c = 160;
        parameter Real d = 1;
        parameter Real tau0max = 80;
        Real phi3(start = 0, fixed = true);
        Real w3(start = 100, fixed = true);
        Real phiRel(start = 0, fixed = true, stateSelect = StateSelect.prefer);
        Real wRel(start = -10, fixed = true, stateSelect = StateSelect.prefer);
        Real phi1(start = 0, fixed = true);
        Real w1(start = 90, fixed = true);
        Real a3;
        Real aRel;
        Real sa;
        Real tau;
        Real tauSpring;
        Boolean startForward(start = false, fixed = true);
        Boolean startBackward(start = false, fixed = true);
        Boolean locked(start = false, fixed = true);
        Integer mode(start = 3, fixed = true);
      equation
        der(phi3) = w3;
        der(w3) = a3;
        der(phiRel) = wRel;
        der(wRel) = aRel;
        der(phi1) = w1;
        J3 * a3 = 200 * sin(100 * time) + tau;
        J2 * (a3 + aRel) = -tau - tauSpring;
        tauSpring = c * (phi3 + phiRel - phi1) + d * (w3 + wRel - w1);
        J1 * der(w1) = tauSpring;
        startForward = pre(mode) == 0 and (sa > tau0max or pre(startForward) and sa > tau0max)
                       or pre(mode) == -1 and wRel > 1e10 or initial() and wRel > 0;
        startBackward = pre(mode) == 0 and (sa < -tau0max or pre(startBackward) and sa < -tau0max)
                        or pre(mode) == 1 and wRel < -1e10 or initial() and wRel < 0;
        locked = not (pre(mode) == 1 or startForward or pre(mode) == -1 or startBackward);
        aRel = if locked then 0 else if startForward then sa - tau0max else if startBackward then sa + tau0max
               else if pre(mode) == 1 then sa - tau0max else sa + tau0max;
        tau = if locked then sa else if startForward then tau0max else if startBackward then -tau0max
              else if pre(mode) == 1 then tau0max else -tau0max;
        mode = if (pre(mode) == 1 or startForward) and wRel > 0 then 1
               else if (pre(mode) == -1 or startBackward) and wRel < 0 then -1 else 0;
      end ClutchSticking;
      """
    local (s, a, _) = _bothWaysOf(program, "ClutchSticking", ["w3", "w1"]; reltol = 1e-6)
    @test a ≈ [94.73865, 95.29922] rtol = 1e-5
    @test s ≈ a rtol = 1e-5
  end

  @testset "A locked friction at the start: the loops in the initial system" begin
    #= MSL RobotR3's bearing friction at the start: locked (pre(mode) = 3), so `a = 0` whatever sa is, and the load
       tauL is held by the motor's current i, by the friction torque tau = sa, or shared. omc solves the friction's
       equation for sa alone after der(w) = 0 fixed a: its residual is zero at sa's start, sa = 0 stays, the motor
       holds the load (i = x = 2) and the shaft rests. A solve of the states alone kept i at its start and the loop
       gave sa = -2 < -tau0max: the shaft slid backward. omc: i = x = 2, sa = 0 at 0 and at 0.2. =#
    local program = Modelica"""
      model LockedLoad "a locked friction at the start under a constant load a motor with a PI controller can hold"
        parameter Real J = 1;
        parameter Real tauL = 2;
        parameter Real tau0max = 0.5;
        parameter Real L = 0.1;
        parameter Real R = 1;
        parameter Real k = 1;
        parameter Real kp = 1;
        parameter Real T = 0.1;
        Real phi(start = 0, fixed = true);
        Real w(start = 0, fixed = true);
        Real i(start = 0);
        Real x(start = 0);
        Real v;
        Real a;
        Real sa;
        Real tau;
        Boolean startForward(start = false, fixed = true);
        Boolean startBackward(start = false, fixed = true);
        Boolean locked(start = false, fixed = true);
        Integer mode(start = 3, fixed = true);
      initial equation
        der(w) = 0;
        der(i) = 0;
      equation
        der(phi) = w;
        der(w) = a;
        J * a = k * i - tau - tauL;
        L * der(i) = v - R * i - k * w;
        der(x) = -w / T;
        v = kp * (x - w);
        startForward = pre(mode) == 0 and (sa > tau0max or pre(startForward) and sa > tau0max)
                       or pre(mode) == -1 and w > 1e10 or initial() and w > 0;
        startBackward = pre(mode) == 0 and (sa < -tau0max or pre(startBackward) and sa < -tau0max)
                        or pre(mode) == 1 and w < -1e10 or initial() and w < 0;
        locked = not (pre(mode) == 1 or startForward or pre(mode) == -1 or startBackward);
        a = if locked then 0 else if startForward then sa - tau0max else if startBackward then sa + tau0max
            else if pre(mode) == 1 then sa - tau0max else sa + tau0max;
        tau = if locked then sa else if startForward then tau0max else if startBackward then -tau0max
              else if pre(mode) == 1 then tau0max else -tau0max;
        mode = if (pre(mode) == 1 or startForward) and w > 0 then 1
               else if (pre(mode) == -1 or startBackward) and w < 0 then -1 else 0;
      end LockedLoad;
      """
    local (s, a, sol) = _bothWaysOf(program, "LockedLoad", ["i", "x", "sa", "w"]; tspan = (0.0, 0.2))
    @test a ≈ [2.0, 2.0, 0.0, 0.0] atol = 1e-8
    @test s ≈ a atol = 1e-8
    @test OMBackend.getVariableValues(sol, "i")[1] ≈ 2.0 atol = 1e-8
    @test OMBackend.getVariableValues(sol, "sa")[1] ≈ 0.0 atol = 1e-8
  end

  @testset "Rotated constant currents through an inductance" begin
    #= MSL's SMPM_CurrentSource and SMR_CurrentSource: dqToThreePhase's rotator turns the constant references
       u = {id, iq} by the rotor angle, the result prescribes the stator currents. The plan's structure reads u as
       constants (`u[1] = id` folded), so Pantelides' algorithm gave no derivative of u[1], and the array form refused
       the differentiated rotator ("index reduction does not handle the derivative of u[1]"). A constant's derivative
       is zero. v = L di/dt with i = R(t) u: v = L (-id sin t - iq cos t, id cos t - iq sin t). =#
    local program = Modelica"""
      model RotatedCurrent "constant currents rotated by phi = time, prescribed through an inductance"
        parameter Real L = 0.5;
        parameter Real id = 2;
        parameter Real iq = 3;
        Real phi(start = 0, fixed = true);
        Real u[2];
        Real i[2];
        Real v[2];
      equation
        der(phi) = 1;
        u[1] = id;
        u[2] = iq;
        i = {{cos(phi), -sin(phi)}, {sin(phi), cos(phi)}} * u;
        v = L * der(i);
      end RotatedCurrent;
      """
    local name::String = _translateProgram(program, "RotatedCurrent"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    local at = (v::String) -> OMBackend.getVariableValues(sol, v)[end]
    local expected::Vector{Float64} = [0.5 * (-2 * sin(1.0) - 3 * cos(1.0)), 0.5 * (2 * cos(1.0) - 3 * sin(1.0))]
    @test isapprox([at("v[1]"), at("v[2]")], expected; rtol = 1e-5)
  end

  @testset "An if-equation on a Boolean variable with a for-equation in a branch" begin
    #= MSL's LossyGear (LossyGearDemo1, LossyGearDemo3, ActuatorWithNoise): `if ideal then interpolation_result =
       [1, 1, 0, 0]; ... else for i in 1:4 loop interpolation_result[1, i] = if smoothness == ... then
       getTable1DValueNoDer(tableID, i, ...) ...`, `ideal` a Boolean variable. Three refusals: the if-expression on the
       smoothness, lifted out of its loop, read `i` there; the Integer matrix literal's element `cat(2, {{1}}, {{1}})[1,
       1]` was not picked ("function cat"); the branches, a for-equation and `{{1, 1}}[1, 1] = r[1, 1]`, were no
       `v = e` ("if-equation on a variable condition: branch equation for i in 1:2 loop ..."). a = 2.5 x,
       der(x) = -2.5 x^2: x(1) = 1 / 3.5 =#
    local program = Modelica"""
      function isOne "whether every element of the first row but the first is one"
        input Real t[:, :];
        output Boolean result = true;
      algorithm
        for j in 2:size(t, 2) loop
          result := result and abs(t[1, j] - 1) < 1e-15;
        end for;
      end isOne;

      model IdealOrTable "LossyGear's ideal or table: an if-equation on a Boolean variable"
        parameter Real table[1, 3] = [0, 0.5, 2];
        parameter Integer smoothness = 1;
        Boolean ideal;
        Real r[1, 2];
        Real a;
        Real x(start = 1, fixed = true);
      equation
        ideal = isOne(table);
        if ideal then
          r = [1, 1];
          a = 1;
        else
          for i in 1:2 loop
            r[1, i] = if smoothness == 1 then table[1, i + 1] * x else i * x;
          end for;
          a = r[1, 1] + r[1, 2];
        end if;
        der(x) = -a * x;
      end IdealOrTable;
      """
    local name::String = _translateProgram(program, "IdealOrTable"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    local at = (v::String) -> OMBackend.getVariableValues(sol, v)[end]
    @test isapprox([at("x"), at("r[1,2]"), at("a")], [1 / 3.5, 2 / 3.5, 2.5 / 3.5]; rtol = 1e-5)
  end

  @testset "der() of an expression next to an array" begin
    #= MSL FluxTubes' Tellinen hysteresis, `dHyst = der(hystR - mu0 * Hstat)` in an if-equation: the plan on the
       element view expands der() of an expression, the array form refused it ("der() of an expression with arrays
       kept"). y = 3 h: der(y - h) = 2 der(h), der(y - 2 h) = der(h); h(1) = 1 - exp(-1), h(3) = 1 - exp(-4) =#
    local program = Modelica"""
      model DerOfExpression "der() of an expression in an if-equation, an array beside it"
        Real x[2](each start = 1, each fixed = true);
        Real h(start = 0, fixed = true);
        Real y;
        Real d;
      equation
        der(x) = -x;
        y = 3 * h;
        if time < 2 then
          d = der(y - h);
        else
          d = der(y - 2 * h);
        end if;
        d = 2 * (1 - h);
      end DerOfExpression;
      """
    local name::String = _translateProgram(program, "DerOfExpression"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 3.0))
    @test sol.retcode == ReturnCode.Success
    @test isapprox([sol(1.0; idxs = :h), sol(3.0; idxs = :h), sol(3.0; idxs = Symbol("x[2]"))],
                   [1 - exp(-1.0), 1 - exp(-4.0), exp(-3.0)]; rtol = 1e-4)
  end
  @testset "Comparators with positive feedback in an algebraic loop" begin
    #= MSL SignalGenerator's IdealizedOpAmpLimited: the limiter `if V0*e > 1 then 1 elseif V0*e < -1 then -1 else
       V0*e` of the output in a loop with its input e = v/2 - x. The array code refused relations in an algebraic
       loop; homotopy() kept only its actual expression, and the loop solved from the start values went to the
       unstable linear root (v = 0, all values 0; x[2] grew away from v = 1/2); at the fold, the lower limit and the
       linear branch each gave the other: the jump to the upper limit is the other condition of the if-expression.
       From v = -1 (homotopy's simplified expression): x = x0 - t until e = -1/V0, i.e. x = -0.499, then a triangle
       between -0.499 and 0.499 at slope 1 =#
    local program = Modelica"""
      model Comparators "relaxation oscillators: comparators with positive feedback, homotopy() at the start"
        parameter Real V0 = 1000 "the comparators' gain";
        Real x[2](start = {0, 0.25}, each fixed = true);
        Real e[2];
        Real v[2];
      equation
        der(x) = v;
        e = 0.5 * v - x;
        for i in 1:2 loop
          v[i] = homotopy(actual = if V0 * e[i] > 1 then 1 elseif V0 * e[i] < -1 then -1 else V0 * e[i],
                          simplified = -1);
        end for;
      end Comparators;
      """
    local name::String = _translateProgram(program, "Comparators"; scalarized = false)
    @test _arrayPathTaken(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 2.0))
    @test sol.retcode == ReturnCode.Success
    @test isapprox([sol(t; idxs = Symbol("x[1]")) for t in (0.25, 1.0, 2.0)], [-0.25, 0.002, -0.004]; atol = 1e-4)
    @test isapprox([sol(t; idxs = Symbol("x[2]")) for t in (0.5, 1.0)], [-0.25, -0.248]; atol = 1e-4)
    @test [sol(t; idxs = Symbol("v[1]")) for t in (0.25, 1.0, 2.0)] ≈ [-1.0, 1.0, -1.0]
  end
  @testset "A homotopy() continuation that throws: the loops' guesses restored" begin
    #= The continuation (`homotopyStart!`) fails at λ = 0 (sqrt of a negative number); the initialization from the
       start values follows, with the loops' guesses from before it. The guesses were restored by a closure with a
       destructured argument, which the generated module called with the pair as the function ("objects of type
       Tuple{Matrix{Float64}, Matrix{Float64}} are not callable"): the model did not start and lost the array code
       (MSL BranchingDynamicPipes). =#
    local program = Modelica"""
      model HomotopyThrows "a homotopy() whose simplified expression throws at the start of the continuation"
        parameter Real V0 = 1000;
        Real x[2](start = {0, 0.25}, each fixed = true);
        Real e[2];
        Real v[2];
      equation
        der(x) = v;
        e = 0.5 * v - x;
        for i in 1:2 loop
          v[i] = homotopy(actual = if V0 * e[i] > 1 then 1 elseif V0 * e[i] < -1 then -1 else V0 * e[i],
                          simplified = sqrt(e[i] - 10));
        end for;
      end HomotopyThrows;
      """
    delete!(OMBackend.FALLBACKS, :arrayHomotopyContinuation)
    local name::String = _translateProgram(program, "HomotopyThrows"; scalarized = false)
    @test _arrayPathTaken(name)
    @test OMBackend.FALLBACKS[:arrayHomotopyContinuation].exceptionType == "DomainError"
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    @test sol.retcode == ReturnCode.Success
    @test sol.u[1] == [0.0, 0.25]
  end
  @testset "A constant array of records read through inlined functions" begin
    #= MSL Media's `data` (MixtureGases): cp_X, inlined by the frontend after its constants were evaluated, kept
       `{cp_T(data[i], T) for i in 1:nX}` with the package constants `data` and `nX`; the index reduction refused the
       array of records (and its scalarization with it, the model had no array preserving code). With the body's
       constants evaluated, cp_T inlined after index reduction reads `{Data(...), Data(...)}[1].Tlimit`, a field the
       array code took only of a record value. =#
    local program = Modelica"""
      package RecordData
        record Data
          Real MM;
          Real cp;
          Real a[2];
          Real Tlimit = 1000;
        end Data;
        package Gases
          constant Data N2(MM = 0.028, cp = 1040.0, a = {1.0, 2.0});
          constant Data O2(MM = 0.032, cp = 918.0, a = {3.0, 4.0});
        end Gases;
        function cp_T "one statement: inlined after index reduction"
          input Data data;
          input Real T;
          output Real cp;
        algorithm
          cp := smooth(0, if T < data.Tlimit then data.cp + data.a[1] * T else data.cp);
          annotation(Inline = false, smoothOrder = 2);
        end cp_T;
        package Medium
          constant Integer nX = 2;
          constant Data data[nX] = {Gases.N2, Gases.O2};
          function cp_X "one statement: inlined by the frontend"
            input Real T;
            input Real X[nX];
            output Real cp;
          algorithm
            cp := {cp_T(data[i], T) for i in 1:nX} * X;
          end cp_X;
        end Medium;
        model Mixture
          Real T(start = 300, fixed = true);
          Real X[2] = {0.7, 0.3};
          Real cp = Medium.cp_X(T, X);
        equation
          der(T) = 1;
        end Mixture;
      end RecordData;
      """
    #= cp = 0.7 (1040 + T) + 0.3 (918 + 3 T) = 1003.4 + 1.6 T, T = 301 at the end =#
    local (s, a, _) = _bothWaysOf(program, "RecordData.Mixture", ["cp"])
    @test a ≈ [1003.4 + 1.6 * 301] rtol = 1e-9
    @test s ≈ a rtol = 1e-9
  end

  @testset "A call returning an array of records as an argument: all its fields" begin
    #= rotateAll returns its records' fields' arrays (y_a, y_b); as dot's argument the array code passed y_a only, a
       MethodError at the first right-hand side (OM.jl's ComplexLoweringTests.ArrayRecordOutputs) =#
    local program = Modelica"""
      package RecordArrayCall
        record R
          Real a;
          Real b;
        end R;
        function rotateAll
          input R u[:];
          input Real phi;
          output R y[size(u, 1)];
        algorithm
          for i in 1:size(u, 1) loop
            y[i] := R(u[i].a * cos(phi) - u[i].b * sin(phi), u[i].a * sin(phi) + u[i].b * cos(phi));
          end for;
        end rotateAll;
        function dot
          input R x[:];
          input R y[:];
          output Real s = 0;
        algorithm
          for i in 1:size(x, 1) loop
            s := s + x[i].a * y[i].a + x[i].b * y[i].b;
          end for;
        end dot;
        model M
          R u[2] = {R(1, 0), R(0, 2)};
          Real d = dot(u, rotateAll(u, time));
          Real x(start = 0, fixed = true);
        equation
          der(x) = d;
        end M;
      end RecordArrayCall;
      """
    #= d = |u|^2 cos(t) = 5 cos(t), x(1) = 5 sin(1) =#
    local (s, a, _) = _bothWaysOf(program, "RecordArrayCall.M", ["d", "x"]; reltol = 1e-8, abstol = 1e-10)
    @test a ≈ [5 * cos(1.0), 5 * sin(1.0)] rtol = 1e-6
    @test s ≈ a rtol = 1e-6
  end

  @testset "Initial arms of lifted clusters before the equations that read them" begin
    #= b and a are lifted clusters, initialized by their initial arms; the equations ran before them and read
       andT[0, 0] with b's and a's starts (a ModelicaSubscriptError: OM.jl's EventSemantics.TwoTablesSameIndices) =#
    local program = Modelica"""
      model TwoTables
        constant Integer andT[4, 4] = [1, 1, 3, 1; 1, 2, 3, 2; 3, 3, 3, 3; 1, 2, 3, 4];
        constant Integer xorT[4, 4] = [1, 1, 1, 1; 1, 2, 2, 2; 1, 2, 3, 4; 1, 2, 4, 3];
        Integer b = if time < 0.5 then 3 else 4;
        Integer a = if time < 0.25 then 1 else 4;
        Integer yAnd;
        Integer yXor;
      equation
        yAnd = andT[b, a];
        yXor = xorT[b, a];
      end TwoTables;
      """
    local (s, a, sol) = _bothWaysOf(program, "TwoTables", ["yAnd", "yXor"])
    @test a == [4.0, 3.0] && s == a
    @test [OMBackend.getVariableValues(sol, "yXor")[1]] == [1.0]
  end

  @testset "Events: a short window, an event function at its interval's end, a DAE solver" begin
    local program = Modelica"""
      package SmallEvents
        model ShortWindow
          Real x(start = 0, fixed = true);
          Integer n(start = 0, fixed = true);
        equation
          der(x) = 1;
          when 81 * sin(time) > 80 then
            n = pre(n) + 1;
          end when;
        end ShortWindow;
        model ModDay
          Real DNTime;
          Real awake;
        equation
          if DNTime > 24 then
            DNTime = 0;
          else
            DNTime = mod(time, 24);
          end if;
          awake = if DNTime < 24 and DNTime > 8 then 1.0 else 0.0;
        end ModDay;
        model DAEIfReinit
          Real x(start = 0, fixed = true);
          Real v(start = 0);
          Real y;
          Real z(start = 0, fixed = true);
        equation
          der(x) = 1;
          when x > 1 then
            reinit(x, 0);
          end when;
          v + v ^ 3 = x;
          y = if v > 0.5 then 1 else 0;
          der(z) = y;
        end DAEIfReinit;
      end SmallEvents;
      """
    local fd = OMBackend.CodeGeneration.ADTypes.AutoFiniteDiff()
    #= 81 sin(t) > 80 only on (1.4135, 1.7281) and 2 pi later: der(x) = 1 took 9 s steps and the relation's checks
       within them missed the first window; the largest step keeps 10 checks per 1/500 of the span, omc's output
       intervals (`_stepKwargs`) =#
    local name::String = _translateProgram(program, "SmallEvents.ShortWindow"; scalarized = false)
    @test _arrayPathTaken(name)
    @test OMBackend.getVariableValues(OMBackend.simulateModel(name; tspan = (0.0, 10.0)), "n")[end] == 2
    #= mod(time, 24) reaches 24 just after the crossing of DNTime > 24 (2e-6 earlier, at the hysteresis): its value
       kept within the band was stale at that event, and the relations switched back and forth (OM.jl's
       PersonalityAspects); at an event it is literal, as omc's _event_floor =#
    name = _translateProgram(program, "SmallEvents.ModDay"; scalarized = false)
    local day = OMBackend.simulateModel(name; tspan = (0.0, 60.0), abstol = 1e-2, reltol = 1e-2,
                                        solver = OMBackend.OrdinaryDiffEqBDF.FBDF(autodiff = fd))
    @test day.retcode == ReturnCode.Success
    @test [day(t; idxs = :DNTime) for t in (30.0, 36.0)] ≈ [6.0, 12.0] atol = 1e-6
    #= a DAE solver on the array code: its ODE in residual form (`ode_to_dae`), initialized by Brown's method after a
       reinit (the default check failed with the residual of the reinit's jump) =#
    name = _translateProgram(program, "SmallEvents.DAEIfReinit"; scalarized = false)
    local dae = OMBackend.simulateModel(name; tspan = (0.0, 3.5), reltol = 1e-8, abstol = 1e-10,
                                        solver = OMBackend.OrdinaryDiffEqBDF.DFBDF(autodiff = fd))
    @test dae.retcode == ReturnCode.Success
    @test [dae(1.4; idxs = :z), dae(3.2; idxs = :v), dae(3.5; idxs = :z)] ≈ [0.375, 0.19282993096291295, 1.125] atol = 1e-6
  end

  @testset "An enumeration's String in a when body" begin
    local program = Modelica"""
      package ByIndex
        model EnumerationString
          type E = enumeration(one, two, three);
          parameter E e = E.three;
          Real x(start = 0, fixed = true);
          Integer n(start = 0, fixed = true);
        algorithm
          when x > 0.5 then
            n := pre(n) + 1;
            assert(n < 1, "e=" + String(e, minimumLength = 7, leftJustified = false) + "|", AssertionLevel.warning);
          end when;
        equation
          der(x) = 1;
        end EnumerationString;
      end ByIndex;
      """
    #= the literals of an enumeration are a tuple of names, indexed by its value (`_checkedIndex(::Tuple, i)`) =#
    local name::String = _translateProgram(program, "ByIndex.EnumerationString"; scalarized = false)
    @test _arrayPathTaken(name)
    @test_logs (:warn, r"e=  three\|$") match_mode = :any OMBackend.simulateModel(name; tspan = (0.0, 1.0))
  end

  @testset "A translate of unchanged code keeps the module; changed code evaluates it again" begin
    #= a model of a file simulated again is translated again: its module was compiled again (BouncingBall: 0.7 s) =#
    local decay(rate::String) = OMFrontend.OMParser.@Modelica """
      model KeptDecay
        Real x[2](each start = 1.0, each fixed = true);
      equation
        der(x) = {-x[1], -$(rate) * x[2]};
      end KeptDecay;
      """
    local name::String = _translateProgram(decay("2.0"), "KeptDecay"; scalarized = false)
    @test _arrayPathTaken(name)
    OMBackend.simulateModel(name; tspan = (0.0, 1.0))
    local evaluated::Module = OMBackend._modelModule(Symbol(name))
    _translateProgram(decay("2.0"), "KeptDecay"; scalarized = false)
    @test !OMBackend.modelWasCompiledAgainDE(name)
    local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0), reltol = 1e-8, abstol = 1e-10)
    @test OMBackend._modelModule(Symbol(name)) === evaluated
    @test _values(sol, ["x[1]", "x[2]"]) ≈ [exp(-1.0), exp(-2.0)] rtol = 1e-6
    _translateProgram(decay("3.0"), "KeptDecay"; scalarized = false)
    @test OMBackend.modelWasCompiledAgainDE(name)
    sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0), reltol = 1e-8, abstol = 1e-10)
    @test OMBackend._modelModule(Symbol(name)) !== evaluated
    @test _values(sol, ["x[1]", "x[2]"]) ≈ [exp(-1.0), exp(-3.0)] rtol = 1e-6
  end
end
