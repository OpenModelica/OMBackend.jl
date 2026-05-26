#= SIM-native Exp traversal + alias-substitution unit tests.

   The DAE round-trip path (`Util.traverseExpTopDown(toDAEExp(e), substituteAliasCref, …)`)
   is the known-good oracle. SIM-native results are checked against it by comparing the
   structural `repr` of their DAE projection, so no `==` on `Exp` is required. =#

using Test

SC    = OMBackend.SimulationCode
Util  = OMBackend.FrontendUtil.Util
DAEx  = OMBackend.DAE
CG    = OMBackend.CodeGeneration

T_R = DAEx.T_REAL(nil)

# --- construction helpers ---
simCref(name)    = SC.EXP_CREF(SC.SimCref(Symbol(name)), SC.TYPE_REAL())
daeCref(name)    = DAEx.CREF_IDENT(name, T_R, nil)
daeCrefExp(name) = DAEx.CREF(daeCref(name), T_R)

# aliasMap entry: name => (repName, negated, repCref, repTy)
aliasOf(name, rep, neg) =
  Dict{String,Tuple{String,Bool,DAEx.ComponentRef,DAEx.Type}}(name => (rep, neg, daeCref(rep), T_R))

# structural equality on DAE.Exp via canonical show output
daeStructEq(a, b) = repr(a) == repr(b)

# differential check: SIM-native result projected to DAE must match the DAE oracle
function simEqualsOracle(e, m)
  (daeOut, _) = Util.traverseExpTopDown(SC.toDAEExp(e), SC.substituteAliasCref, m)
  (simOut, _) = SC.traverseExpTopDown(e, SC.substituteAliasCref, m)
  return daeStructEq(SC.toDAEExp(simOut), daeOut)
end

function cpEqualsOracle(e, m)
  (daeOut, _) = Util.traverseExpTopDown(SC.toDAEExp(e), SC.substituteConstantParameter, m)
  (simOut, _) = SC.traverseExpTopDown(e, SC.substituteConstantParameter, m)
  return daeStructEq(SC.toDAEExp(simOut), daeOut)
end

function fvEqualsOracle(e, m)
  (daeOut, _) = Util.traverseExpTopDown(SC.toDAEExp(e), SC.substituteFoldedVar, m)
  (simOut, _) = SC.traverseExpTopDown(e, SC.substituteFoldedVar, m)
  return daeStructEq(SC.toDAEExp(simOut), daeOut)
end

# _foldNumericExp is a direct recursive folder (not a traverse visitor); compare
# the SIM fold against folding the DAE projection.
function foldEqualsOracle(e)
  simOut = SC._foldNumericExp(e)
  daeOut = SC._foldNumericExp(SC.toDAEExp(e))
  return daeStructEq(SC.toDAEExp(simOut), daeOut)
end

function fsEqualsOracle(e, m)
  (daeOut, _) = Util.traverseExpTopDown(SC.toDAEExp(e), SC._substituteFrozenState, m)
  (simOut, _) = SC.traverseExpTopDown(e, SC._substituteFrozenState, m)
  return daeStructEq(SC.toDAEExp(simOut), daeOut)
end

# Minimal mock SIM_CODE so the codegen path (expToJuliaExpMTK) can be unit-tested
# without a full translate: only stringToSimVarHT is exercised by the arms under test.
function mockSimCode(ht)
  return SC.SIM_CODE("mock", ht,
    SC.RESIDUAL_EQUATION[], SC.Equation[], SC.WHEN_EQUATION[], SC.IF_EQUATION[],
    false, Int[], SC.Graphs.SimpleDiGraph(0), [], SC.StructuralTransition[], [],
    String[], String[], SC.Equation[], "mock", NONE(), NONE(), String[],
    SC.ModelicaFunction[], false, SC.RESIDUAL_EQUATION[], String[], SC.AliasEntry[],
    nothing, SC.INITIAL_ALGORITHM[])
end

# Expr equality ignoring source-location decoration and semantically-transparent
# single-expression `begin … end` wrappers (the DAE emitter wraps some arm results
# in a block; the SIM arms return the bare expression — both evaluate identically).
function _unwrapTrivialBlocks(x)
  x isa Expr || return x
  if x.head === :block && length(x.args) == 1
    return _unwrapTrivialBlocks(x.args[1])
  end
  return Expr(x.head, Any[_unwrapTrivialBlocks(a) for a in x.args]...)
end
_normExpr(x) = _unwrapTrivialBlocks(Base.remove_linenums!(deepcopy(x)))
exprEq(a, b) = _normExpr(a) == _normExpr(b)

# Differential: SIM-native expToJuliaExpMTK must match the DAE.Exp emitter (the arms mirror it).
function e2jEqualsOracle(e, sc)
  return exprEq(CG.expToJuliaExpMTK(e, sc), CG.expToJuliaExpMTK(SC.toDAEExp(e), sc))
end

# Differential: SIM-native expToJuliaExpDE (DEMode, per-variant arms) must match the DAE.Exp emitter.
function deEqualsOracle(e, layout, sc)
  return exprEq(CG.DEGen.expToJuliaExpDE(e, layout, sc), CG.DEGen.expToJuliaExpDE(SC.toDAEExp(e), layout, sc))
end

# Differential: SIM-native expToJuliaExp (generic emitter, per-variant arms) must match the DAE.Exp emitter.
function ejEqualsOracle(e, sc)
  return exprEq(CG.expToJuliaExp(e, sc), CG.expToJuliaExp(SC.toDAEExp(e), sc))
end

# SType -> DAE.Type -> SType is exact (SType carries no attrs to lose); the
# DAE -> SType direction drops attrs, so the round-trip is tested from the SType side.
tyRoundtrips(t::SC.SType) = repr(SC.toSimType(SC.toDAEType(t))) == repr(t)

# function-call construction: build via DAE then toSimExp so CallAttributes are realistic
Absynx = OMBackend.Absyn
callAttrR = DAEx.CALL_ATTR(T_R, false, true, false, false, DAEx.NO_INLINE(), DAEx.NO_TAIL())
daeCall(fn, args...) = DAEx.CALL(Absynx.IDENT(fn), list(args...), callAttrR)
simCall(fn, args...) = SC.toSimExp(daeCall(fn, args...))

@testset "Inner Component Tests" begin
  @testset "SimCode Traversal" begin

    @testset "traverseExpTopDown primitives" begin
      e = SC.BINARY(simCref("x"), SC.OP_ADD, SC.RCONST(1.0))

      # identity visitor reuses the unchanged node by ===
      (o, _) = SC.traverseExpTopDown(e, (x, a) -> (x, true, a), nothing)
      @test o === e

      # leaf-replace RCONST(1.0) -> RCONST(2.0): spine rebuilt, sibling child reused
      repl = (x, a) -> (x isa SC.RCONST && x.value == 1.0) ? (SC.RCONST(2.0), false, a) : (x, true, a)
      (o2, _) = SC.traverseExpTopDown(e, repl, nothing)
      @test o2 !== e
      @test o2.exp1 === e.exp1
      @test o2.exp2 isa SC.RCONST && o2.exp2.value == 2.0

      # cont=false short-circuits: children are never visited
      seen = Ref(0)
      SC.traverseExpTopDown(e, (x, a) -> (seen[] += 1; (x, false, a)), nothing)
      @test seen[] == 1
    end

    @testset "traverseExpBottomUp primitives" begin
      e = SC.BINARY(simCref("x"), SC.OP_ADD,
                    SC.BINARY(simCref("y"), SC.OP_MUL, SC.RCONST(3.0)))

      # one visit per node: 2 BINARY + 2 EXP_CREF + 1 RCONST
      (_, n) = SC.traverseExpBottomUp(e, (x, a) -> (x, a + 1), 0)
      @test n == 5

      # identity reuses the root
      (o, _) = SC.traverseExpBottomUp(e, (x, a) -> (x, a), 0)
      @test o === e
    end

    @testset "substituteAliasCref oracle spec (DAE path)" begin
      # bare alias x -> y
      (out, _) = Util.traverseExpTopDown(daeCrefExp("x"), SC.substituteAliasCref, aliasOf("x", "y", false))
      @test daeStructEq(out, daeCrefExp("y"))

      # negated alias x -> -y (UMINUS wrap)
      (outN, _) = Util.traverseExpTopDown(daeCrefExp("x"), SC.substituteAliasCref, aliasOf("x", "y", true))
      @test daeStructEq(outN, DAEx.UNARY(DAEx.UMINUS(T_R), daeCrefExp("y")))

      # non-aliased cref passes through unchanged
      (outU, _) = Util.traverseExpTopDown(daeCrefExp("z"), SC.substituteAliasCref, aliasOf("x", "y", false))
      @test daeStructEq(outU, daeCrefExp("z"))

      # alias inside a BINARY recurses: x + 1 -> y + 1
      bin = DAEx.BINARY(daeCrefExp("x"), DAEx.ADD(T_R), DAEx.RCONST(1.0))
      (outB, _) = Util.traverseExpTopDown(bin, SC.substituteAliasCref, aliasOf("x", "y", false))
      @test daeStructEq(outB, DAEx.BINARY(daeCrefExp("y"), DAEx.ADD(T_R), DAEx.RCONST(1.0)))
    end

    @testset "substituteAliasCref SIM-native == DAE oracle" begin
      @test simEqualsOracle(simCref("x"), aliasOf("x", "y", false))                 # bare
      @test simEqualsOracle(simCref("x"), aliasOf("x", "y", true))                  # negated
      @test simEqualsOracle(simCref("z"), aliasOf("x", "y", false))                 # passthrough
      @test simEqualsOracle(SC.BINARY(simCref("x"), SC.OP_ADD, SC.RCONST(1.0)),
                            aliasOf("x", "y", false))                               # nested in BINARY
      @test simEqualsOracle(SC.BINARY(simCref("x"), SC.OP_SUB, simCref("x")),
                            aliasOf("x", "y", true))                                # two negated subs
      @test simEqualsOracle(SC.BINARY(SC.BINARY(simCref("x"), SC.OP_MUL, SC.RCONST(2.0)),
                                      SC.OP_ADD, simCref("w")),
                            aliasOf("x", "y", false))                               # deep recursion
    end

    @testset "SIM-native — variants, builtins, ASUB" begin
      mapXY  = aliasOf("x", "y", false)
      mapXYn = aliasOf("x", "y", true)

      # variant coverage: fallback recursion through every compound shape
      @test simEqualsOracle(SC.RELATION(simCref("x"), SC.OP_LESS, SC.RCONST(1.0), 0), mapXY)
      @test simEqualsOracle(SC.IFEXP(SC.RELATION(simCref("x"), SC.OP_GREATER, SC.RCONST(0.0), 0),
                                     simCref("x"), simCref("w")), mapXY)
      @test simEqualsOracle(SC.LBINARY(simCref("x"), SC.OP_AND, simCref("z")), mapXY)
      @test simEqualsOracle(SC.LUNARY(SC.OP_NOT, simCref("x")), mapXY)
      @test simEqualsOracle(SC.ARRAY_EXP(SC.TYPE_REAL(), true, SC.Exp[simCref("x"), simCref("w")]), mapXY)
      @test simEqualsOracle(SC.CAST(T_R, simCref("x")), mapXY)
      @test simEqualsOracle(simCall("sin", daeCrefExp("x")), mapXY)

      # empty map is a no-op
      @test simEqualsOracle(SC.BINARY(simCref("x"), SC.OP_ADD, simCref("w")),
                            Dict{String,Tuple{String,Bool,DAEx.ComponentRef,DAEx.Type}}())

      # no cascade: x->y and y->z must yield y (cont=false stops re-traversal of the replacement)
      @test simEqualsOracle(simCref("x"),
              Dict{String,Tuple{String,Bool,DAEx.ComponentRef,DAEx.Type}}(
                "x" => ("y", false, daeCref("y"), T_R), "y" => ("z", false, daeCref("z"), T_R)))

      # unary-state builtins: negation lifts outside the call
      @test simEqualsOracle(simCall("der", daeCrefExp("x")), mapXY)   # der(x) -> der(y)
      @test simEqualsOracle(simCall("der", daeCrefExp("x")), mapXYn)  # der(x) -> -der(y)
      @test simEqualsOracle(simCall("pre", daeCrefExp("x")), mapXYn)  # pre(x) -> -pre(y)

      # ASUB base-name and full-name aliasing
      @test simEqualsOracle(SC.ASUB(simCref("x"), SC.Exp[SC.ICONST(1)]), mapXY)                       # x[1] -> y[1]
      @test simEqualsOracle(SC.ASUB(simCref("x"), SC.Exp[SC.ICONST(1)]), aliasOf("x[1]", "y", false)) # x[1] -> y
    end

    @testset "substituteConstantParameter SIM-native == DAE oracle" begin
      simCrefT(name, ty) = SC.EXP_CREF(SC.SimCref(Symbol(name)), ty)
      pmap = Dict{String,Float64}("p" => 3.5, "n" => 4.0, "b" => 1.0)
      @test cpEqualsOracle(simCrefT("p", SC.TYPE_REAL()), pmap)       # Real    -> RCONST
      @test cpEqualsOracle(simCrefT("n", SC.TYPE_INTEGER()), pmap)    # Integer -> ICONST(round)
      @test cpEqualsOracle(simCrefT("b", SC.TYPE_BOOL()), pmap)       # Bool    -> BCONST
      @test cpEqualsOracle(simCrefT("z", SC.TYPE_REAL()), pmap)       # non-param passthrough
      @test cpEqualsOracle(SC.BINARY(simCrefT("p", SC.TYPE_REAL()), SC.OP_ADD, SC.RCONST(1.0)), pmap)  # nested in BINARY
      @test cpEqualsOracle(SC.ASUB(simCrefT("arr", SC.TYPE_REAL()), SC.Exp[SC.ICONST(1)]),
                           Dict{String,Float64}("arr[1]" => 2.0))     # ASUB canonical name
    end

    @testset "substituteFoldedVar SIM-native == DAE oracle" begin
      fmap = Dict{String,DAEx.Exp}("f" => DAEx.RCONST(5.0))
      @test fvEqualsOracle(simCref("f"), fmap)                         # cref f -> RCONST(5.0)
      @test fvEqualsOracle(simCref("z"), fmap)                         # non-folded passthrough
      @test fvEqualsOracle(SC.BINARY(simCref("f"), SC.OP_ADD, SC.RCONST(1.0)), fmap)  # nested in BINARY
      @test fvEqualsOracle(SC.ASUB(simCref("arr"), SC.Exp[SC.ICONST(1)]),
                           Dict{String,DAEx.Exp}("arr[1]" => DAEx.RCONST(9.0)))        # ASUB fullName
    end

    @testset "_foldNumericExp SIM-native == DAE oracle" begin
      R(x) = SC.RCONST(x)
      @test foldEqualsOracle(SC.BINARY(R(2.0), SC.OP_ADD, R(3.0)))                    # 2+3 -> 5
      @test foldEqualsOracle(SC.BINARY(R(6.0), SC.OP_DIV, R(2.0)))                    # 6/2 -> 3
      @test foldEqualsOracle(SC.BINARY(R(0.0), SC.OP_MUL, simCref("x")))              # 0*x -> 0
      @test foldEqualsOracle(SC.BINARY(R(1.0), SC.OP_MUL, simCref("x")))              # 1*x -> x
      @test foldEqualsOracle(SC.BINARY(simCref("x"), SC.OP_ADD, R(0.0)))              # x+0 -> x
      @test foldEqualsOracle(SC.UNARY(SC.OP_UMINUS, R(4.0)))                          # -(4) -> -4
      @test foldEqualsOracle(SC.BINARY(SC.BINARY(R(2.0), SC.OP_MUL, R(3.0)),
                                       SC.OP_ADD, simCref("x")))                      # (2*3)+x -> 6+x
      @test foldEqualsOracle(SC.BINARY(simCref("x"), SC.OP_DIV, R(2.0)))              # x/2 unchanged
      @test foldEqualsOracle(simCall("der", daeCrefExp("x")))                         # der(x) unchanged
      @test foldEqualsOracle(simCall("der", DAEx.RCONST(5.0)))                        # der(numeric) -> 0
    end

    @testset "_substituteFrozenState SIM-native == DAE oracle" begin
      fzmap = Dict{String,DAEx.Exp}("s" => DAEx.RCONST(0.0))
      @test fsEqualsOracle(simCref("s"), fzmap)                                      # frozen cref -> value
      @test fsEqualsOracle(simCref("z"), fzmap)                                      # non-frozen passthrough
      @test fsEqualsOracle(simCall("der", daeCrefExp("s")), fzmap)                   # der(frozen) -> 0
      @test fsEqualsOracle(simCall("der", daeCrefExp("y")), fzmap)                   # der(non-frozen) unchanged
      @test fsEqualsOracle(SC.BINARY(simCref("s"), SC.OP_ADD, SC.RCONST(1.0)), fzmap) # frozen in BINARY
    end

    @testset "expToJuliaExpMTK SIM-native == DAE oracle" begin
      sc = mockSimCode(Dict{String, Tuple{Integer, SC.SimVar}}(
        "x" => (0, SC.SIMVAR("x", SOME(0), SC.ALG_VARIABLE(0), NONE()))))
      @test e2jEqualsOracle(SC.BCONST(true), sc)
      @test e2jEqualsOracle(SC.ICONST(7), sc)
      @test e2jEqualsOracle(SC.RCONST(2.5), sc)
      @test e2jEqualsOracle(SC.BINARY(SC.RCONST(2.0), SC.OP_ADD, SC.RCONST(3.0)), sc)
      @test e2jEqualsOracle(SC.UNARY(SC.OP_UMINUS, SC.RCONST(4.0)), sc)
      @test e2jEqualsOracle(SC.RELATION(SC.RCONST(1.0), SC.OP_LESS, SC.RCONST(2.0), 0), sc)
      @test e2jEqualsOracle(SC.LBINARY(SC.BCONST(true), SC.OP_AND, SC.BCONST(false)), sc)
      @test e2jEqualsOracle(simCref("x"), sc)                                          # EXP_CREF via mock HT
      @test e2jEqualsOracle(SC.BINARY(simCref("x"), SC.OP_MUL, SC.RCONST(2.0)), sc)    # cref inside expr
      @test e2jEqualsOracle(SC.IFEXP(SC.RELATION(simCref("x"), SC.OP_GREATER, SC.RCONST(0.0), 0),
                                     SC.RCONST(1.0), SC.RCONST(2.0)), sc)
      @test e2jEqualsOracle(SC.RSUB(simCref("x"), 1, "re", T_R), sc)
      @test e2jEqualsOracle(simCref("time"), sc)                                       # time -> t (ty isa T_REAL branch)
      #= CAST / ARRAY_EXP arms delegate into generateCastExpressionMTK / handleArrayExp,
         which need a real model's context — not cleanly mockable, so LRT-gated. =#
    end

    @testset "expToJuliaExpDE SIM-native == DAE oracle" begin
      # DEMode resolves STATE -> u[i], PARAMETER -> p[i], time -> t; the layout is built
      # from the mock HT's varKinds. No MSL test exercises DEMode, so this is the coverage.
      sc = mockSimCode(Dict{String, Tuple{Integer, SC.SimVar}}(
        "x" => (0, SC.SIMVAR("x", SOME(0), SC.STATE(), NONE())),
        "p" => (1, SC.SIMVAR("p", SOME(1), SC.PARAMETER(NONE()), NONE()))))
      layout = CG.DEGen.buildDELayout(sc)
      @test deEqualsOracle(SC.BCONST(true), layout, sc)
      @test deEqualsOracle(SC.ICONST(7), layout, sc)
      @test deEqualsOracle(SC.RCONST(2.5), layout, sc)
      @test deEqualsOracle(SC.SCONST("s"), layout, sc)
      @test deEqualsOracle(SC.BINARY(SC.RCONST(2.0), SC.OP_ADD, SC.RCONST(3.0)), layout, sc)
      @test deEqualsOracle(SC.UNARY(SC.OP_UMINUS, SC.RCONST(4.0)), layout, sc)
      @test deEqualsOracle(SC.LBINARY(SC.BCONST(true), SC.OP_AND, SC.BCONST(false)), layout, sc)
      @test deEqualsOracle(SC.RELATION(SC.RCONST(1.0), SC.OP_LESS, SC.RCONST(2.0), 0), layout, sc)
      @test deEqualsOracle(SC.CAST(SC.TYPE_REAL(), SC.RCONST(1.0)), layout, sc)             # CAST drops to inner
      @test deEqualsOracle(simCall("sin", SC.RCONST(1.0)), layout, sc)                      # CALL -> :call
      @test deEqualsOracle(simCref("x"), layout, sc)                                        # STATE -> u[i]
      @test deEqualsOracle(simCref("p"), layout, sc)                                        # PARAMETER -> p[i]
      @test deEqualsOracle(simCref("time"), layout, sc)                                     # time -> t
      @test deEqualsOracle(SC.BINARY(simCref("x"), SC.OP_MUL, SC.RCONST(2.0)), layout, sc)  # cref inside expr
      @test deEqualsOracle(SC.IFEXP(SC.RELATION(simCref("x"), SC.OP_GREATER, SC.RCONST(0.0), 0),
                                    simCref("x"), SC.RCONST(2.0)), layout, sc)              # IFEXP inlined
    end

    @testset "expToJuliaExp SIM-native == DAE oracle" begin
      # generic emitter resolves STATE/ALG -> <varPrefix>[i], PARAMETER -> p[i], time -> t.
      # CALL / CAST arms need real model context (LRT-gated, as for expToJuliaExpMTK).
      sc = mockSimCode(Dict{String, Tuple{Integer, SC.SimVar}}(
        "x" => (0, SC.SIMVAR("x", SOME(0), SC.ALG_VARIABLE(0), NONE())),
        "p" => (1, SC.SIMVAR("p", SOME(1), SC.PARAMETER(NONE()), NONE())),
        "s" => (2, SC.SIMVAR("s", SOME(2), SC.STATE(), NONE()))))
      @test ejEqualsOracle(SC.BCONST(true), sc)
      @test ejEqualsOracle(SC.ICONST(7), sc)
      @test ejEqualsOracle(SC.RCONST(2.5), sc)
      @test ejEqualsOracle(SC.SCONST("str"), sc)
      @test ejEqualsOracle(SC.BINARY(SC.RCONST(2.0), SC.OP_ADD, SC.RCONST(3.0)), sc)
      @test ejEqualsOracle(SC.UNARY(SC.OP_UMINUS, SC.RCONST(4.0)), sc)
      @test ejEqualsOracle(SC.LBINARY(SC.BCONST(true), SC.OP_AND, SC.BCONST(false)), sc)
      @test ejEqualsOracle(SC.RELATION(SC.RCONST(1.0), SC.OP_LESS, SC.RCONST(2.0), 0), sc)
      @test ejEqualsOracle(simCref("x"), sc)                                          # ALG_VARIABLE
      @test ejEqualsOracle(simCref("p"), sc)                                          # PARAMETER
      @test ejEqualsOracle(simCref("s"), sc)                                          # STATE
      @test ejEqualsOracle(simCref("time"), sc)                                       # time -> t
      @test ejEqualsOracle(SC.BINARY(simCref("x"), SC.OP_MUL, SC.RCONST(2.0)), sc)    # cref inside expr
      @test ejEqualsOracle(SC.IFEXP(SC.RELATION(simCref("x"), SC.OP_GREATER, SC.RCONST(0.0), 0),
                                    SC.RCONST(1.0), SC.RCONST(2.0)), sc)              # IFEXP -> ifelse
    end

    @testset "_flagCanonicalExp! SIM-native == DAE oracle" begin
      # SIM-native flagger must visit the same nodes (same violation count) as the DAE flagger
      # on the projection — the per-node checks are shared via per-cref projection.
      function flagCountEq(e)
        local o1 = SC.SimCodeCheck.CheckViolation[]; SC.SimCodeCheck._flagCanonicalExp!(o1, e, "w")
        local o2 = SC.SimCodeCheck.CheckViolation[]; SC.SimCodeCheck._flagCanonicalExp!(o2, SC.toDAEExp(e), "w")
        return length(o1) == length(o2)
      end
      @test flagCountEq(simCref("x"))
      @test flagCountEq(SC.BINARY(simCref("x"), SC.OP_ADD, SC.RCONST(1.0)))
      @test flagCountEq(simCall("sin", simCref("x")))
      @test flagCountEq(SC.IFEXP(SC.RELATION(simCref("x"), SC.OP_LESS, SC.RCONST(0.0), 0),
                                 simCref("x"), SC.RCONST(1.0)))
      @test flagCountEq(SC.UNARY(SC.OP_UMINUS, SC.BINARY(simCref("x"), SC.OP_MUL, simCref("x"))))
      @test flagCountEq(SC.RSUB(simCref("x"), 1, "re", T_R))
    end

    @testset "SimCodeCheck collectors/flaggers SIM-native == DAE oracle" begin
      local SCC = SC.SimCodeCheck
      function crefNamesEq(e, known)
        local m1 = String[]; SCC._collectCrefNames!(m1, e, known)
        local m2 = String[]; SCC._collectCrefNames!(m2, SC.toDAEExp(e), known)
        return sort(m1) == sort(m2)
      end
      function derNamesEq(e)
        local s1 = Set{String}(); SCC._collectDerFromExp!(s1, e)
        local s2 = Set{String}(); SCC._collectDerFromExp!(s2, SC.toDAEExp(e))
        return s1 == s2
      end
      function litDerPreEq(e)
        local o1 = SCC.CheckViolation[]; SCC._flagLiteralInDerPre!(o1, e, "w")
        local o2 = SCC.CheckViolation[]; SCC._flagLiteralInDerPre!(o2, SC.toDAEExp(e), "w")
        return length(o1) == length(o2)
      end
      local kn = Set{String}(["x"])
      @test crefNamesEq(simCref("x"), kn)                                                 # known -> no missing
      @test crefNamesEq(simCref("y"), kn)                                                 # unknown -> missing
      @test crefNamesEq(SC.BINARY(simCref("x"), SC.OP_ADD, simCref("z")), kn)
      @test derNamesEq(simCall("der", daeCrefExp("x")))                                   # der(x) -> "x"
      @test derNamesEq(SC.BINARY(simCall("der", daeCrefExp("x")), SC.OP_ADD, simCref("y")))
      @test derNamesEq(SC.RSUB(simCall("der", daeCrefExp("x")), 1, "re", T_R))
      @test litDerPreEq(simCall("der", DAEx.RCONST(0.0)))                                 # der(literal) -> warn
      @test litDerPreEq(simCall("der", daeCrefExp("x")))                                  # der(cref) -> no warn
      @test litDerPreEq(simCall("pre", DAEx.ICONST(3)))                                   # pre(literal) -> warn
    end

    @testset "SType conversions (toSimType / toDAEType)" begin
      # forward: DAE.Type -> SType category
      @test SC.toSimType(DAEx.T_REAL_DEFAULT)    isa SC.TYPE_REAL
      @test SC.toSimType(DAEx.T_INTEGER_DEFAULT) isa SC.TYPE_INTEGER
      @test SC.toSimType(DAEx.T_BOOL_DEFAULT)    isa SC.TYPE_BOOL
      @test SC.toSimType(DAEx.T_STRING_DEFAULT)  isa SC.TYPE_STRING
      @test SC.toSimType(DAEx.T_UNKNOWN())       isa SC.TYPE_UNKNOWN
      # array element type + dims preserved
      local at = SC.toSimType(DAEx.T_ARRAY(DAEx.T_INTEGER_DEFAULT,
                                           list(DAEx.DIM_INTEGER(2), DAEx.DIM_INTEGER(5))))
      @test at isa SC.TYPE_ARRAY && at.elementType isa SC.TYPE_INTEGER && at.dims == [2, 5]
      # nested array element type preserved
      local at2 = SC.toSimType(DAEx.T_ARRAY(DAEx.T_ARRAY(DAEx.T_REAL_DEFAULT, list(DAEx.DIM_INTEGER(2))),
                                             list(DAEx.DIM_INTEGER(3))))
      @test at2 isa SC.TYPE_ARRAY && at2.elementType isa SC.TYPE_ARRAY && at2.dims == [3]
      # unmapped DAE type collapses to UNKNOWN
      @test SC.toSimType(DAEx.T_NORETCALL()) isa SC.TYPE_UNKNOWN
      # SType -> DAE -> SType round-trips exactly
      @test tyRoundtrips(SC.TYPE_REAL())
      @test tyRoundtrips(SC.TYPE_INTEGER())
      @test tyRoundtrips(SC.TYPE_BOOL())
      @test tyRoundtrips(SC.TYPE_STRING())
      @test tyRoundtrips(SC.TYPE_UNKNOWN())
      @test tyRoundtrips(SC.TYPE_ENUM())
      @test tyRoundtrips(SC.TYPE_ARRAY(SC.TYPE_REAL(), [3]))
      @test tyRoundtrips(SC.TYPE_ARRAY(SC.TYPE_INTEGER(), [-1]))     # unknown dim
      @test tyRoundtrips(SC.TYPE_TUPLE([SC.TYPE_REAL(), SC.TYPE_BOOL()]))
      # COMPLEX keeps its field directory (names + element types) + RECORD/connector bit
      @test tyRoundtrips(SC.TYPE_COMPLEX(true, ["re", "im"], SC.SType[SC.TYPE_REAL(), SC.TYPE_REAL()]))
      @test tyRoundtrips(SC.TYPE_COMPLEX(false, ["x"], SC.SType[SC.TYPE_ARRAY(SC.TYPE_REAL(), [3])]))
      # toSimType from the reconstructed DAE T_COMPLEX recovers the field directory
      local cx = SC.toSimType(SC.toDAEType(SC.TYPE_COMPLEX(true, ["re", "im"], SC.SType[SC.TYPE_REAL(), SC.TYPE_REAL()])))
      @test cx isa SC.TYPE_COMPLEX && cx.isRecord && cx.fieldNames == ["re", "im"]
    end

    @testset "bindExp Option{Exp} migration" begin
      # SimVarType bindExp stores Option{Exp}; _toSimBindExp converts a producer's
      # Option{DAE.Exp} binding to the SimCode-native form (NONE passes through).
      local dae = DAEx.RCONST(2.5)
      local sb = SC._toSimBindExp(SOME(dae))
      @test sb isa SOME && sb.data isa SC.Exp
      @test daeStructEq(SC.toDAEExp(sb.data), dae)
      @test SC._toSimBindExp(NONE()) === NONE()
      # SimVarType variants accept a SIM binding; the no-binding default is NONE
      local p = SC.PARAMETER(SOME(SC.RCONST(1.0)))
      @test p.bindExp isa SOME && p.bindExp.data isa SC.Exp
      @test SC.ARRAY(Int[2]).bindExp === NONE()
      local rsub = SC.toSimExp(DAEx.RSUB(daeCrefExp("x"), 1, "re", T_R))
      @test rsub isa SC.RSUB
      @test daeStructEq(SC.toDAEExp(rsub), DAEx.RSUB(daeCrefExp("x"), 1, "re", T_R))
    end

    @testset "collectCrefNames! SIM-native == DAE oracle" begin
      crefNamesEq(e) = begin
        s1 = Set{String}(); SC.collectCrefNames!(s1, e)
        s2 = Set{String}(); SC.collectCrefNames!(s2, SC.toDAEExp(e))
        s1 == s2
      end
      @test crefNamesEq(simCref("a"))
      @test crefNamesEq(SC.BINARY(simCref("a"), SC.OP_ADD, simCref("b")))
      @test crefNamesEq(SC.IFEXP(simCref("c"), simCref("t"), simCref("e")))
      @test crefNamesEq(SC.RSUB(simCref("rec"), 1, "re", T_R))
      @test crefNamesEq(simCall("f", daeCrefExp("p"), daeCrefExp("q")))
      @test crefNamesEq(SC.ARRAY_EXP(T_R, true, SC.Exp[simCref("u"), simCref("v")]))
      @test crefNamesEq(SC.BINARY(simCall("g", daeCrefExp("x")), SC.OP_MUL,
                                  SC.ASUB(simCref("M"), SC.Exp[SC.ICONST(2)])))
      # ASUB-of-cref with constant subscripts reconstructs the scalarized key + base
      local sA = Set{String}()
      SC.collectCrefNames!(sA, SC.ASUB(simCref("R_T"), SC.Exp[SC.ICONST(1), SC.ICONST(1)]))
      @test "R_T[1][1]" in sA && "R_T" in sA
      @test crefNamesEq(SC.ASUB(simCref("R_T"), SC.Exp[SC.ICONST(1), SC.ICONST(1)]))
    end

    @testset "_collectIfexpConditionCrefs! SIM-native == DAE oracle" begin
      ifCondEq(e) = begin
        s1 = Set{String}(); SC._collectIfexpConditionCrefs!(s1, e)
        s2 = Set{String}(); SC._collectIfexpConditionCrefs!(s2, SC.toDAEExp(e))
        s1 == s2
      end
      @test ifCondEq(SC.IFEXP(simCref("p"), simCref("t"), simCref("e")))
      @test ifCondEq(SC.IFEXP(simCref("a"), SC.IFEXP(simCref("b"), simCref("c"), simCref("d")), simCref("e")))
      @test ifCondEq(SC.IFEXP(SC.IFEXP(simCref("b"), simCref("c"), simCref("d")), simCref("s"), simCref("t")))
      @test ifCondEq(SC.BINARY(SC.IFEXP(simCref("x"), simCref("y"), simCref("z")), SC.OP_ADD, simCref("w")))
      @test ifCondEq(SC.BINARY(simCref("m"), SC.OP_ADD, simCref("n")))
      # at a single IFEXP only the condition's crefs are collected, not the branches
      local sI = Set{String}()
      SC._collectIfexpConditionCrefs!(sI, SC.IFEXP(simCref("p"), simCref("t"), simCref("e")))
      @test sI == Set(["p"])
    end

    @testset "evalDAEConstant SIM-native literal fast-path" begin
      for (sim, val) in Any[(SC.RCONST(2.5), 2.5), (SC.ICONST(3), 3),
                            (SC.BCONST(true), true), (SC.SCONST("hi"), "hi")]
        @test CG.evalDAEConstant(sim) == CG.evalDAEConstant(SC.toDAEExp(sim))
        @test CG.evalDAEConstant(sim) == val
      end
    end

  end
end
