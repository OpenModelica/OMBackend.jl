#=
A cluster that reads its own pre() values iterates to a fixpoint within one
event (CodeGeneration._eventIterAffectParts, Modelica event iteration). The
cluster below is the MSL OneWayClutch freewheel: when w reaches 0, `stuck`
becomes true in the first pass, and only the second pass (pre(stuck) = true)
sets `locked`. A one-pass affect never locks the freewheel.
=#
module _EventIterationEval
  #= The generated affect calls SciMLBase.u_modified!; record it instead. =#
  module SciMLBase
    u_modified!(integrator, flag) = (integrator.flagged[] = flag)
  end
end

isdefined(@__MODULE__, :BackendTestMocks) || include(joinpath(@__DIR__, "backendTestMocks.jl"))

@testset "Event iteration" begin
  local CG = OMBackend.CodeGeneration
  local SC = OMBackend.SimCode
  local B = SC.TYPE_BOOL()
  local cref(n, ty = SC.TYPE_REAL()) = SC.EXP_CREF(SC.SimCref(Symbol(n)), ty)
  local pre(n) = SC.PRE(SC.SimCref(Symbol(n)), B)
  local builtin(name, args...) = SC.CALL(name, SC.Exp[args...], B, SC.CALL_BUILTIN)
  local rel(a, op, b) = SC.RELATION(a, op, b, -1)
  local and(a, b) = SC.LBINARY(a, SC.OP_AND, b)
  local or(a, b) = SC.LBINARY(a, SC.OP_OR, b)
  local w = cref("w")
  local sa = cref("sa")
  local wle0 = rel(w, SC.OP_LESSEQ, SC.RCONST(0.0))
  #= startForward = pre(stuck) and sa > 1; locked = pre(stuck) and not startForward;
     stuck = locked or w <= 0 =#
  local assigns = Tuple{Symbol, SC.Exp, Bool}[
    (:startForward, and(pre("stuck"), rel(sa, SC.OP_GREATER, SC.RCONST(1.0))), false),
    (:locked, and(pre("stuck"), SC.LUNARY(SC.OP_NOT, cref("startForward", B))), false),
    (:stuck, or(cref("locked", B), wle0), false),
  ]
  #= No variables needed: every name is looked up as a plain reference. =#
  local simCode = BackendTestMocks.mockSimCode()

  @test CG._clusterReadsOwnPre(assigns)
  @test !CG._clusterReadsOwnPre(assigns[3:3])

  local (fexpr, obsNT, modNT) = CG._eventIterAffectParts(assigns, simCode; relPins = Dict{SC.Exp, Bool}(wle0 => true))
  @test Set(a.args[1] for a in obsNT.args[1].args) == Set([:sa])
  @test Set(a.args[1] for a in modNT.args[1].args) == Set([:startForward, :locked, :stuck])
  local affect = Core.eval(_EventIterationEval, fexpr)
  local integrator = (t = 0.25, flagged = Ref(false))
  local sliding = (startForward = 0.0, locked = 0.0, stuck = 0.0)

  #= w falls to 0 while sliding, no breakaway torque: stuck, then locked. =#
  local r = Base.invokelatest(affect, sliding, (sa = 0.0,), nothing, integrator)
  @test r == (startForward = 0.0, locked = 1.0, stuck = 1.0)
  @test integrator.flagged[]    # the integrator is told the discretes changed

  #= Breakaway torque at the same instant: stuck but not locked. =#
  r = Base.invokelatest(affect, sliding, (sa = 2.0,), nothing, integrator)
  @test r == (startForward = 1.0, locked = 0.0, stuck = 1.0)

  #= Already at the fixpoint: nothing changes. =#
  r = Base.invokelatest(affect, (startForward = 0.0, locked = 1.0, stuck = 1.0), (sa = 0.0,), nothing, integrator)
  @test r == (startForward = 0.0, locked = 1.0, stuck = 1.0)

  #= One pass (a user when's body): stuck, but not yet locked. =#
  local (fexpr1, _, _) = CG._eventIterAffectParts(assigns, simCode; relPins = Dict{SC.Exp, Bool}(wle0 => true),
                                                  iterate = false)
  r = Base.invokelatest(Core.eval(_EventIterationEval, fexpr1), sliding, (sa = 0.0,), nothing, integrator)
  @test r == (startForward = 0.0, locked = 0.0, stuck = 1.0)

  #= initial() is what the t0 pass (initialize) says. =#
  local atStart = Tuple{Symbol, SC.Exp, Bool}[(:i, or(builtin(:initial), wle0), false)]
  for (initVal, expected) in ((true, 1.0), (false, 0.0))
    local (fx, _, _) = CG._eventIterAffectParts(atStart, simCode; initVal = initVal)
    @test Base.invokelatest(Core.eval(_EventIterationEval, fx), (i = 0.0,), (w = 1.0,), nothing, integrator) ==
          (i = expected,)
  end

  #= A body that reads a Boolean defined by a relation elsewhere (an ideal
     thyristor's `fire`) reads the relation, which joins the event sources;
     pre(fire) stays. =#
  local sLt0 = rel(cref("s"), SC.OP_LESS, SC.RCONST(0.0))
  local fireRel = rel(cref("repl"), SC.OP_LESS, cref("timer"))
  local thyristor = Tuple{Symbol, SC.Exp, Bool}[
    (:off, or(sLt0, and(pre("off"), SC.LUNARY(SC.OP_NOT, cref("fire", B)))), false)]
  local relationOf = Dict{String, SC.RELATION}("fire" => fireRel)
  local (rels2, body2) = CG._inlineReadBooleanRelations(SC.RELATION[sLt0], thyristor, relationOf, simCode)
  @test rels2 == SC.RELATION[sLt0, fireRel]
  @test SC.anyExp(==(fireRel), body2[1][2]) && !SC.anyExp(e -> e == cref("fire", B), body2[1][2])
  local preOnly = Tuple{Symbol, SC.Exp, Bool}[(:off, or(sLt0, pre("fire")), false)]
  local (rels3, body3) = CG._inlineReadBooleanRelations(SC.RELATION[sLt0], preOnly, relationOf, simCode)
  @test length(rels3) == 1 && body3 === preOnly

  #= Only synthesized conditions (an OR-chain of change(<relation>)) iterate;
     a user `when change(b)` runs its body once per event. =#
  local sagt1 = rel(sa, SC.OP_GREATER, SC.RCONST(1.0))
  @test CG._isSynthesizedChangeCondition(or(builtin(:change, wle0), builtin(:change, sagt1)))
  @test !CG._isSynthesizedChangeCondition(builtin(:change, cref("b", B)))
  @test !CG._isSynthesizedChangeCondition(wle0)

  #= Casts and edge() lower; y latches when b rises (b = x > 0.2,
     y = edge(b) or pre(y) and x > 0.1 with the Integer k cast to Real). =#
  local x = cref("x")
  local xgt = rel(x, SC.OP_GREATER, SC.CAST(SC.TYPE_REAL(), cref("k", SC.TYPE_INTEGER())))
  local latch = Tuple{Symbol, SC.Exp, Bool}[
    (:b, xgt, false),
    (:y, or(builtin(:edge, cref("b", B)), and(pre("y"), rel(x, SC.OP_GREATER, SC.RCONST(0.1)))), false),
  ]
  @test CG._eventIterLowerable(latch, simCode)
  local (fexpr2, _, _) = CG._eventIterAffectParts(latch, simCode; relPins = Dict{SC.Exp, Bool}(xgt => true))
  local latchAffect = Core.eval(_EventIterationEval, fexpr2)
  r = Base.invokelatest(latchAffect, (b = 0.0, y = 0.0), (x = 0.2, k = 0.0), nothing, integrator)
  @test r == (b = 1.0, y = 1.0)

  #= The frontend inlines b = x > 0.2 into edge(x > 0.2): the relation rises
     at this crossing, so edge() holds in the first pass only, and y latches. =#
  local xgt2 = rel(x, SC.OP_GREATER, SC.RCONST(0.2))
  local inlined = Tuple{Symbol, SC.Exp, Bool}[
    (:y, or(builtin(:edge, xgt2), and(pre("y"), rel(x, SC.OP_GREATER, SC.RCONST(0.1)))), false),
  ]
  local (fexpr3, _, _) = CG._eventIterAffectParts(inlined, simCode; relPins = Dict{SC.Exp, Bool}(xgt2 => true))
  r = Base.invokelatest(Core.eval(_EventIterationEval, fexpr3), (y = 0.0,), (x = 0.2,), nothing, integrator)
  @test r == (y = 1.0,)
  #= Falling through 0.2 is no edge; y holds while x > 0.1. =#
  local (fexpr4, _, _) = CG._eventIterAffectParts(inlined, simCode; relPins = Dict{SC.Exp, Bool}(xgt2 => false))
  local fall = Core.eval(_EventIterationEval, fexpr4)
  @test Base.invokelatest(fall, (y = 1.0,), (x = 0.2,), nothing, integrator) == (y = 1.0,)
  @test Base.invokelatest(fall, (y = 0.0,), (x = 0.2,), nothing, integrator) == (y = 0.0,)

  #= An event operator that has no affect lowering keeps the one-pass affects. =#
  local sampled = Tuple{Symbol, SC.Exp, Bool}[
    (:s, or(pre("s"), builtin(:sample, SC.RCONST(0.0), SC.RCONST(1.0))), false),
  ]
  @test !(@test_logs (:info,) CG._eventIterLowerable(sampled, simCode))
end

#= A relation whose crossing the cluster's callback located flips in the event
   iteration's update, also where the state it reads is still inside the
   hysteresis band (the algebraic solve at the event can leave it there);
   without a located crossing the hysteresis rule decides. =#
@testset "Discrete cluster: located crossings" begin
  local CG = OMBackend.CodeGeneration
  local c = CG.DiscreteCluster(["off"], Any[], Any[], 0, 0, [true], [false], nothing, 0, false)
  #= Crossing function -1e-12 (inside the band H * scale = 1e-10), scale 1. =#
  c.crossings! = (zs, u, p, t) -> (zs[1] = -1.0e-12; zs[2] = 1.0; nothing)
  local integrator = (u = Float64[], p = nothing, t = 0.0, opts = (reltol = 1.0e-6,),
                      sol = (prob = (tspan = (0.0, 1.0),),))
  @test !CG._update!(c, integrator) && c.rel == [false]
  c.crossed[1] = true
  @test CG._update!(c, integrator) && c.rel == [true] && c.crossed == [false]
end

#= The algebraic re-solve at an event (CodeGeneration.EventReinit): 0 = 1e8 (y^2 - 2)
   keeps ~4e-8 on the residual after rounding, out of BrownFullBasicInit's own 1e-10
   (as a thyristor bridge's commutation, near singular at values of 1e6); the result
   stands within the solve's abstol, and beyond it where y itself is within the
   solve's tolerance (a Newton correction of 1e-16: `_withinSolveTolerance`, MSL
   DifferenceAmplifier's ramp end). A re-solve that did not converge does not stand:
   0 = y^2 + 1 has no real solution. =#
@testset "Event re-solve: accepted within the solve's abstol" begin
  local CG = OMBackend.CodeGeneration
  local SB = CG.ModelingToolkit.SciMLBase
  local ODE = CG.OrdinaryDiffEq
  local f!(du, u, p, t) = (du[1] = -u[1]; du[2] = 1e8 * (u[2]^2 - 2); nothing)
  local prob = SB.ODEProblem(SB.ODEFunction(f!; mass_matrix = [1.0 0.0; 0.0 0.0]), [1.0, 1.0], (0.0, 1.0))
  local reinit(abstol, alg) = (local i = ODE.init(prob, ODE.Rodas5P(); initializealg = SB.NoInit(), abstol = abstol);
                               CG.DiffEqBase.initialize_dae!(i, alg); i)
  @test reinit(1e-6, CG.DiffEqBase.BrownFullBasicInit()).sol.retcode == SB.ReturnCode.InitialFailure
  local i = reinit(1e-6, CG.EventReinit())
  @test i.sol.retcode == SB.ReturnCode.Default && i.u[2] ≈ sqrt(2)
  local i12 = reinit(1e-12, CG.EventReinit())
  @test i12.sol.retcode == SB.ReturnCode.Default && i12.u[2] ≈ sqrt(2)
  local g!(du, u, p, t) = (du[1] = -u[1]; du[2] = u[2]^2 + 1; nothing)
  local none = SB.ODEProblem(SB.ODEFunction(g!; mass_matrix = [1.0 0.0; 0.0 0.0]), [1.0, 1.0], (0.0, 1.0))
  local j = ODE.init(none, ODE.Rodas5P(); initializealg = SB.NoInit())
  CG.DiffEqBase.initialize_dae!(j, CG.EventReinit())
  @test j.sol.retcode == SB.ReturnCode.InitialFailure
end

#= The re-solve at a flow reversal, MSL Fluid MeasuringTemperature reduced: a port's flow m
   from a pressure difference sits at its reversal (5e-9, within a finite-difference step of 0);
   the flow's enthalpy h is the upstream side's without an event (the cold side's before the
   affect, the hot side's after), the temperature T follows from h. BrownFullBasicInit's
   differences straddle the switch (d h-row / d m of 2e10 for the branch's 0): 1015 calls of
   the right-hand side (MeasuringTemperature: 79 s a re-solve); Newton with the problem's
   Jacobian: 3 iterations. =#
@testset "Event re-solve: Newton with the problem's Jacobian at a flow reversal" begin
  local CG = OMBackend.CodeGeneration
  local SB = CG.ModelingToolkit.SciMLBase
  local ODE = CG.OrdinaryDiffEq
  local SA = CG.Symbolics.SparseArrays
  local calls = Ref(0)
  local (hHot, hCold) = (3.35e5, 0.84e5)
  local hOfT(T) = 4180 * T + 0.5 * T^2 - 0.5 * 300^2 - 4180 * 300 + 1.0e5
  local f!(du, u, p, t) = begin
    calls[] += 1
    local (x, m, pr, h, T) = u
    du[1] = -m
    du[2] = m - 1e-9 * (pr - 1.0e5 - 1.0e4 * x) * (1000 - 0.3 * (T - 300))
    du[3] = pr - 1.0e5 - 1.0e4 * x - 1e-4 * (T - 300)
    du[4] = h - ifelse(m > 0, hHot, hCold)
    du[5] = h - hOfT(T)
    nothing
  end
  local jac!(J, u, p, t) = begin
    local (x, m, pr, h, T) = u
    local (rho, dp) = (1000 - 0.3 * (T - 300), pr - 1.0e5 - 1.0e4 * x)
    fill!(SA.nonzeros(J), 0.0)
    J[1, 2] = -1
    J[2, 1] = 1e-5 * rho; J[2, 2] = 1; J[2, 3] = -1e-9 * rho; J[2, 5] = 3e-10 * dp
    J[3, 1] = -1.0e4; J[3, 3] = 1; J[3, 5] = -1e-4
    J[4, 4] = 1
    J[5, 4] = 1; J[5, 5] = -(4180 + T)
    nothing
  end
  local proto = SA.sparse([1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5], [2, 1, 2, 3, 5, 1, 3, 5, 2, 4, 4, 5], ones(12), 5, 5)
  local u0 = [0.5, 1e-10, 1.05e5, hCold, 300.0]
  local fun = SB.ODEFunction(f!; mass_matrix = [1.0 0 0 0 0; zeros(4, 5)], jac = jac!, jac_prototype = proto)
  local i = ODE.init(SB.ODEProblem(fun, u0, (0.0, 1.0)), ODE.Rodas5P(); initializealg = SB.NoInit())
  calls[] = 0
  CG.DiffEqBase.initialize_dae!(i, CG.EventReinit())
  @test i.sol.retcode == SB.ReturnCode.Default
  @test calls[] <= 10
  @test i.u[4] == hHot && hOfT(i.u[5]) ≈ hHot && i.u[1] == 0.5 && i.u[2] > 0
end

#= A large algebraic block (3000 rows, tridiagonal, each row reading the differential state):
   the re-solve factorizes the block sparse, as the problem's Jacobian is (a dense copy would be
   72 MB at every event). =#
@testset "Event re-solve: a large algebraic block, factorized sparse" begin
  local CG = OMBackend.CodeGeneration
  local SB = CG.ModelingToolkit.SciMLBase
  local ODE = CG.OrdinaryDiffEq
  local SA = CG.Symbolics.SparseArrays
  local N = 3001
  local f!(du, u, p, t) = begin
    du[1] = -u[1]
    for i in 2:N
      du[i] = 4u[i] - (i > 2 ? u[i - 1] : 0.0) - (i < N ? u[i + 1] : 0.0) - u[1]
    end
    nothing
  end
  local jac!(J, u, p, t) = begin
    fill!(SA.nonzeros(J), 0.0)
    J[1, 1] = -1
    for i in 2:N
      J[i, i] = 4; J[i, 1] = -1
      i > 2 && (J[i, i - 1] = -1)
      i < N && (J[i, i + 1] = -1)
    end
    nothing
  end
  local (rs, cs) = (Int[1], Int[1])
  for i in 2:N
    append!(rs, (i, i)); append!(cs, (i, 1))
    i > 2 && (push!(rs, i); push!(cs, i - 1))
    i < N && (push!(rs, i); push!(cs, i + 1))
  end
  local proto = SA.sparse(rs, cs, ones(length(rs)), N, N)
  local u0 = [1.0; zeros(N - 1)]
  local fun = SB.ODEFunction(f!; mass_matrix = CG.LinearAlgebra.Diagonal([1.0; zeros(N - 1)]), jac = jac!, jac_prototype = proto)
  local i = ODE.init(SB.ODEProblem(fun, u0, (0.0, 1.0)), ODE.Rodas5P(); initializealg = SB.NoInit())
  CG.DiffEqBase.initialize_dae!(i, CG.EventReinit())
  local r = similar(u0)
  f!(r, i.u, nothing, 0.0)
  @test i.sol.retcode == SB.ReturnCode.Default
  @test maximum(abs, r[2:end]) <= 1e-10 && i.u[1] == 1.0
  i.u .= u0
  @test (@allocated CG.DiffEqBase.initialize_dae!(i, CG.EventReinit())) < 30_000_000
end
