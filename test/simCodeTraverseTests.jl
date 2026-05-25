#= SIM-native Exp traversal + alias-substitution unit tests.

   The DAE round-trip path (`Util.traverseExpTopDown(toDAEExp(e), substituteAliasCref, …)`)
   is the known-good oracle. SIM-native results are checked against it by comparing the
   structural `repr` of their DAE projection, so no `==` on `Exp` is required. =#

using Test

SC    = OMBackend.SimulationCode
Util  = OMBackend.FrontendUtil.Util
DAEx  = OMBackend.DAE

T_R = DAEx.T_REAL(nil)

# --- construction helpers ---
simCref(name)    = SC.EXP_CREF(SC.SimCref(Symbol(name)), T_R)
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
      @test simEqualsOracle(SC.ARRAY_EXP(T_R, true, SC.Exp[simCref("x"), simCref("w")]), mapXY)
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
      @test cpEqualsOracle(simCrefT("p", DAEx.T_REAL(nil)), pmap)     # T_REAL    -> RCONST
      @test cpEqualsOracle(simCrefT("n", DAEx.T_INTEGER(nil)), pmap)  # T_INTEGER -> ICONST(round)
      @test cpEqualsOracle(simCrefT("b", DAEx.T_BOOL(nil)), pmap)     # T_BOOL    -> BCONST
      @test cpEqualsOracle(simCrefT("z", T_R), pmap)                  # non-param passthrough
      @test cpEqualsOracle(SC.BINARY(simCrefT("p", DAEx.T_REAL(nil)), SC.OP_ADD, SC.RCONST(1.0)), pmap)  # nested in BINARY
      @test cpEqualsOracle(SC.ASUB(simCrefT("arr", T_R), SC.Exp[SC.ICONST(1)]),
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

  end
end
