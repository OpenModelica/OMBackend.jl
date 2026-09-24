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
