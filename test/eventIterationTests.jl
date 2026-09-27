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

@testset "Event iteration" begin
  local CG = OMBackend.CodeGeneration
  local tyR = DAE.T_REAL_DEFAULT
  local tyB = DAE.T_BOOL_DEFAULT
  local cref(n, ty) = DAE.CREF(DAE.CREF_IDENT(n, ty, MetaModelica.nil), ty)
  local pre(n) = DAE.CALL(Absyn.IDENT("pre"), MetaModelica.list(cref(n, tyB)), DAE.callAttrBuiltinBool)
  local rel(a, op, b) = DAE.RELATION(a, op, b, -1, NONE())
  local w = cref("w", tyR)
  local sa = cref("sa", tyR)
  local wle0 = rel(w, DAE.LESSEQ(tyR), DAE.RCONST(0.0))
  #= startForward = pre(stuck) and sa > 1; locked = pre(stuck) and not startForward;
     stuck = locked or w <= 0 =#
  local assigns = Tuple{Symbol,Any,Bool}[
    (:startForward, DAE.LBINARY(pre("stuck"), DAE.AND(tyB), rel(sa, DAE.GREATER(tyR), DAE.RCONST(1.0))), false),
    (:locked, DAE.LBINARY(pre("stuck"), DAE.AND(tyB), DAE.LUNARY(DAE.NOT(tyB), cref("startForward", tyB))), false),
    (:stuck, DAE.LBINARY(cref("locked", tyB), DAE.OR(tyB), wle0), false),
  ]
  #= No SimVar table needed: every name is looked up as a plain cref. =#
  local simCode = (; stringToSimVarHT = Dict{String,Any}())

  @test CG._clusterReadsOwnPre(assigns)
  @test !CG._clusterReadsOwnPre(assigns[3:3])

  local (fexpr, obsNT, modNT) = CG._eventIterAffectParts(assigns, simCode;
                                                        relPins = Dict(string(wle0) => true))
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
  local (fexpr1, _, _) = CG._eventIterAffectParts(assigns, simCode; relPins = Dict(string(wle0) => true),
                                                  iterate = false)
  r = Base.invokelatest(Core.eval(_EventIterationEval, fexpr1), sliding, (sa = 0.0,), nothing, integrator)
  @test r == (startForward = 0.0, locked = 0.0, stuck = 1.0)

  #= initial() is what the t0 pass (initialize) says. =#
  local initialCall = DAE.CALL(Absyn.IDENT("initial"), MetaModelica.nil, DAE.callAttrBuiltinBool)
  local atStart = Tuple{Symbol,Any,Bool}[(:i, DAE.LBINARY(initialCall, DAE.OR(tyB), wle0), false)]
  for (initVal, expected) in ((true, 1.0), (false, 0.0))
    local (fx, _, _) = CG._eventIterAffectParts(atStart, simCode; initVal = initVal)
    @test Base.invokelatest(Core.eval(_EventIterationEval, fx), (i = 0.0,), (w = 1.0,), nothing, integrator) ==
          (i = expected,)
  end

  #= A body that reads a Boolean defined by a relation elsewhere (an ideal
     thyristor's `fire`) reads the relation, which joins the event sources;
     pre(fire) stays. =#
  local s = cref("s", tyR)
  local sLt0 = rel(s, DAE.LESS(tyR), DAE.RCONST(0.0))
  local fireRel = rel(cref("repl", tyR), DAE.LESS(tyR), cref("timer", tyR))
  local thyristor = Tuple{Symbol,Any,Bool}[
    (:off, DAE.LBINARY(sLt0, DAE.OR(tyB),
                       DAE.LBINARY(pre("off"), DAE.AND(tyB), DAE.LUNARY(DAE.NOT(tyB), cref("fire", tyB)))), false)]
  local relationOf = Dict{String, DAE.Exp}("fire" => fireRel)
  local (rels2, body2) = CG._inlineReadBooleanRelations(DAE.Exp[sLt0], thyristor, relationOf, simCode)
  @test string.(rels2) == string.([sLt0, fireRel])
  @test occursin(string(fireRel), string(body2[1][2])) && !occursin("fire", string(body2[1][2]))
  local preOnly = Tuple{Symbol,Any,Bool}[(:off, DAE.LBINARY(sLt0, DAE.OR(tyB), pre("fire")), false)]
  local (rels3, body3) = CG._inlineReadBooleanRelations(DAE.Exp[sLt0], preOnly, relationOf, simCode)
  @test length(rels3) == 1 && body3 === preOnly

  #= Only synthesized conditions (an OR-chain of change(<relation>)) iterate;
     a user `when change(b)` runs its body once per event. =#
  local change(e) = DAE.CALL(Absyn.IDENT("change"), MetaModelica.list(e), DAE.callAttrBuiltinBool)
  local sagt1 = rel(sa, DAE.GREATER(tyR), DAE.RCONST(1.0))
  @test CG._isSynthesizedChangeCondition(DAE.LBINARY(change(wle0), DAE.OR(tyB), change(sagt1)))
  @test !CG._isSynthesizedChangeCondition(change(cref("b", tyB)))
  @test !CG._isSynthesizedChangeCondition(wle0)

  #= Casts and edge() lower; y latches when b rises (b = x > 0.2,
     y = edge(b) or pre(y) and x > 0.1 with the Integer k cast to Real). =#
  local x = cref("x", tyR)
  local xgt = rel(x, DAE.GREATER(tyR), DAE.CAST(tyR, cref("k", DAE.T_INTEGER_DEFAULT)))
  local edgeB = DAE.CALL(Absyn.IDENT("edge"), MetaModelica.list(cref("b", tyB)), DAE.callAttrBuiltinBool)
  local latch = Tuple{Symbol,Any,Bool}[
    (:b, xgt, false),
    (:y, DAE.LBINARY(edgeB, DAE.OR(tyB),
                     DAE.LBINARY(pre("y"), DAE.AND(tyB), rel(x, DAE.GREATER(tyR), DAE.RCONST(0.1)))), false),
  ]
  @test CG._eventIterLowerable(latch, simCode)
  local (fexpr2, _, _) = CG._eventIterAffectParts(latch, simCode; relPins = Dict(string(xgt) => true))
  local latchAffect = Core.eval(_EventIterationEval, fexpr2)
  r = Base.invokelatest(latchAffect, (b = 0.0, y = 0.0), (x = 0.2, k = 0.0), nothing, integrator)
  @test r == (b = 1.0, y = 1.0)

  #= The frontend inlines b = x > 0.2 into edge(x > 0.2): the relation rises
     at this crossing, so edge() holds in the first pass only, and y latches. =#
  local xgt2 = rel(x, DAE.GREATER(tyR), DAE.RCONST(0.2))
  local edgeRel = DAE.CALL(Absyn.IDENT("edge"), MetaModelica.list(xgt2), DAE.callAttrBuiltinBool)
  local inlined = Tuple{Symbol,Any,Bool}[
    (:y, DAE.LBINARY(edgeRel, DAE.OR(tyB),
                     DAE.LBINARY(pre("y"), DAE.AND(tyB), rel(x, DAE.GREATER(tyR), DAE.RCONST(0.1)))), false),
  ]
  local (fexpr3, _, _) = CG._eventIterAffectParts(inlined, simCode; relPins = Dict(string(xgt2) => true))
  r = Base.invokelatest(Core.eval(_EventIterationEval, fexpr3), (y = 0.0,), (x = 0.2,), nothing, integrator)
  @test r == (y = 1.0,)
  #= Falling through 0.2 is no edge; y holds while x > 0.1. =#
  local (fexpr4, _, _) = CG._eventIterAffectParts(inlined, simCode; relPins = Dict(string(xgt2) => false))
  local fall = Core.eval(_EventIterationEval, fexpr4)
  @test Base.invokelatest(fall, (y = 1.0,), (x = 0.2,), nothing, integrator) == (y = 1.0,)
  @test Base.invokelatest(fall, (y = 0.0,), (x = 0.2,), nothing, integrator) == (y = 0.0,)

  #= An event operator that has no affect lowering keeps the one-pass affects. =#
  local sampled = Tuple{Symbol,Any,Bool}[
    (:s, DAE.LBINARY(pre("s"), DAE.OR(tyB),
                     DAE.CALL(Absyn.IDENT("sample"), MetaModelica.list(DAE.RCONST(0.0), DAE.RCONST(1.0)),
                              DAE.callAttrBuiltinBool)), false),
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
  local integrator = (u = Float64[], p = nothing, t = 0.0, opts = (reltol = 1.0e-6,))
  @test !CG._update!(c, integrator) && c.rel == [false]
  c.crossed[1] = true
  @test CG._update!(c, integrator) && c.rel == [true] && c.crossed == [false]
end

#= The algebraic re-solve at an event (CodeGeneration.EventReinit): 0 = 1e8 (y^2 - 2)
   keeps ~4e-8 on the residual after rounding, out of BrownFullBasicInit's own 1e-10
   (as a thyristor bridge's commutation, near singular at values of 1e6); the result
   stands within the solve's abstol, not beyond it. =#
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
  @test reinit(1e-12, CG.EventReinit()).sol.retcode == SB.ReturnCode.InitialFailure
end
