#= The SimCode expressions: their traversals, the substitution of eliminated variables (by an
   alias's representative, a parameter's value, a frozen state's value), the folding of
   literal arithmetic, the names an expression reads, the lowering of OMJLIR types, and the
   Julia the code generators emit for them (evaluated, not compared as text). =#

using Test

isdefined(@__MODULE__, :BackendTestMocks) || include(joinpath(@__DIR__, "backendTestMocks.jl"))

@testset "Inner Component Tests" begin
  local SC = OMBackend.SimCode
  local CG = OMBackend.CodeGeneration
  local IR = OMBackend.OMJLIR
  local M = BackendTestMocks
  local ref = M.simCref
  local R = SC.RCONST
  local x = ref("x")
  local y = ref("y")
  local neg = e -> SC.UNARY(SC.OP_UMINUS, e)
  #= the replacements of a substitution: name => (replacement, negated, time-invariant) =#
  local replacements = pairs -> SC.Replacements(n => SC.Replacement(r, negated, invariant) for (n, (r, negated, invariant)) in pairs)
  local alias = (n, r; negated = false) -> replacements([n => (ref(r), negated, false)])
  local call = (name, args...) -> SC.CALL(name, SC.Exp[args...], SC.TYPE_REAL(), SC.CALL_BUILTIN)
  #= the value of generated Julia with names bound =#
  local value = (expr; bindings...) -> Core.eval(Module(), :(let $((:($k = $v) for (k, v) in bindings)...); $expr end))

  @testset "SimCode Traversal" begin

    @testset "traverseExpTopDown primitives" begin
      local e = SC.BINARY(x, SC.OP_ADD, R(1.0))
      #= identity visitor reuses the unchanged node by === =#
      local (o, _) = SC.traverseExpTopDown(e, (x, a) -> (x, true, a), nothing)
      @test o === e
      #= leaf-replace 1.0 -> 2.0: spine rebuilt, sibling child reused =#
      local repl = (x, a) -> (x isa SC.RCONST && x.value == 1.0) ? (R(2.0), false, a) : (x, true, a)
      local (o2, _) = SC.traverseExpTopDown(e, repl, nothing)
      @test o2 !== e
      @test o2.exp1 === e.exp1
      @test o2.exp2 == R(2.0)
      #= cont = false short-circuits: children are never visited =#
      local seen = Ref(0)
      SC.traverseExpTopDown(e, (x, a) -> (seen[] += 1; (x, false, a)), nothing)
      @test seen[] == 1
    end

    @testset "traverseExpBottomUp primitives" begin
      local e = SC.BINARY(x, SC.OP_ADD, SC.BINARY(y, SC.OP_MUL, R(3.0)))
      #= one visit per node: 2 BINARY + 2 EXP_CREF + 1 RCONST =#
      local (_, n) = SC.traverseExpBottomUp(e, (x, a) -> (x, a + 1), 0)
      @test n == 5
      #= identity reuses the root =#
      local (o, _) = SC.traverseExpBottomUp(e, (x, a) -> (x, a), 0)
      @test o === e
    end

    @testset "substitute: an alias by its representative" begin
      @test SC.substitute(x, alias("x", "y")) == y                                     # bare
      @test SC.substitute(x, alias("x", "y"; negated = true)) == neg(y)                # negated
      @test SC.substitute(ref("z"), alias("x", "y")) == ref("z")                       # passthrough
      @test SC.substitute(SC.BINARY(x, SC.OP_ADD, R(1.0)), alias("x", "y")) == SC.BINARY(y, SC.OP_ADD, R(1.0))
      @test SC.substitute(SC.BINARY(x, SC.OP_SUB, x), alias("x", "y"; negated = true)) == SC.BINARY(neg(y), SC.OP_SUB, neg(y))
      @test SC.substitute(SC.BINARY(SC.BINARY(x, SC.OP_MUL, R(2.0)), SC.OP_ADD, ref("w")), alias("x", "y")) ==
            SC.BINARY(SC.BINARY(y, SC.OP_MUL, R(2.0)), SC.OP_ADD, ref("w"))          # deep recursion
    end

    @testset "substitute: every compound shape, der, pre, subscripts" begin
      local xy = alias("x", "y")
      local gt0 = e -> SC.RELATION(e, SC.OP_GREATER, R(0.0), 0)
      @test SC.substitute(SC.RELATION(x, SC.OP_LESS, R(1.0), 0), xy) == SC.RELATION(y, SC.OP_LESS, R(1.0), 0)
      @test SC.substitute(SC.IFEXP(gt0(x), x, ref("w")), xy) == SC.IFEXP(gt0(y), y, ref("w"))
      @test SC.substitute(SC.LBINARY(x, SC.OP_AND, ref("z")), xy) == SC.LBINARY(y, SC.OP_AND, ref("z"))
      @test SC.substitute(SC.LUNARY(SC.OP_NOT, x), xy) == SC.LUNARY(SC.OP_NOT, y)
      @test SC.substitute(SC.ARRAY_EXP(SC.TYPE_REAL(), true, SC.Exp[x, ref("w")]), xy) ==
            SC.ARRAY_EXP(SC.TYPE_REAL(), true, SC.Exp[y, ref("w")])
      @test SC.substitute(SC.CAST(SC.TYPE_REAL(), x), xy) == SC.CAST(SC.TYPE_REAL(), y)
      @test SC.substitute(call(:sin, x), xy) == call(:sin, y)
      #= an empty map is a no-op =#
      local e = SC.BINARY(x, SC.OP_ADD, ref("w"))
      @test SC.substitute(e, SC.Replacements()) === e
      #= no cascade: x -> y and y -> z give y (a replacement is not visited again) =#
      @test SC.substitute(x, replacements(["x" => (y, false, false), "y" => (ref("z"), false, false)])) == y
      #= der and pre: the negation outside =#
      local derX = SC.DER(SC.SimCref(:x))
      local preX = SC.PRE(SC.SimCref(:x), SC.TYPE_REAL())
      @test SC.substitute(derX, xy) == SC.DER(SC.SimCref(:y))
      @test SC.substitute(derX, alias("x", "y"; negated = true)) == neg(SC.DER(SC.SimCref(:y)))
      @test SC.substitute(preX, alias("x", "y"; negated = true)) == neg(SC.PRE(SC.SimCref(:y), SC.TYPE_REAL()))
      #= an element by its array's alias, or by its own =#
      local x1 = SC.ASUB(x, SC.Exp[SC.ICONST(1)])
      @test SC.substitute(x1, xy) == SC.ASUB(y, SC.Exp[SC.ICONST(1)])
      @test SC.substitute(x1, alias("x[1]", "y")) == y
    end

    @testset "substitute: a parameter's or a frozen state's value" begin
      local values = replacements(["p" => (R(3.5), false, true), "s" => (R(0.0), false, true)])
      @test SC.substitute(ref("p"), values) == R(3.5)
      @test SC.substitute(SC.BINARY(ref("p"), SC.OP_ADD, R(1.0)), values) == SC.BINARY(R(3.5), SC.OP_ADD, R(1.0))
      @test SC.substitute(ref("z"), values) == ref("z")
      #= time-invariant: its derivative is zero, pre() of it is the value =#
      @test SC.substitute(SC.DER(SC.SimCref(:s)), values) == R(0.0)
      @test SC.substitute(SC.DER(SC.SimCref(:y)), values) == SC.DER(SC.SimCref(:y))
      @test SC.substitute(SC.PRE(SC.SimCref(:p), SC.TYPE_REAL()), values) == R(3.5)
      #= an element by literal subscripts by its own name =#
      @test SC.substitute(SC.ASUB(ref("arr"), SC.Exp[SC.ICONST(1)]), replacements(["arr[1]" => (R(2.0), false, true)])) == R(2.0)
    end

    @testset "foldNumeric" begin
      @test SC.foldNumeric(SC.BINARY(R(2.0), SC.OP_ADD, R(3.0))) == R(5.0)
      @test SC.foldNumeric(SC.BINARY(R(6.0), SC.OP_DIV, R(2.0))) == R(3.0)
      @test SC.foldNumeric(SC.BINARY(R(0.0), SC.OP_MUL, x)) == R(0.0)
      @test SC.foldNumeric(SC.BINARY(R(1.0), SC.OP_MUL, x)) == x
      @test SC.foldNumeric(SC.BINARY(x, SC.OP_ADD, R(0.0))) == x
      @test SC.foldNumeric(neg(R(4.0))) == R(-4.0)
      @test SC.foldNumeric(SC.BINARY(SC.BINARY(R(2.0), SC.OP_MUL, R(3.0)), SC.OP_ADD, x)) == SC.BINARY(R(6.0), SC.OP_ADD, x)
      @test SC.foldNumeric(SC.BINARY(x, SC.OP_DIV, R(2.0))) == SC.BINARY(x, SC.OP_DIV, R(2.0))
      #= no division by zero =#
      @test SC.foldNumeric(SC.BINARY(R(1.0), SC.OP_DIV, R(0.0))) == SC.BINARY(R(1.0), SC.OP_DIV, R(0.0))
      @test SC.foldNumeric(SC.DER(SC.SimCref(:x))) == SC.DER(SC.SimCref(:x))
    end

    @testset "crefNames!: the names an expression reads" begin
      local names = e -> collect(SC.crefNames!(SC.OrderedSet{String}(), e))
      @test names(ref("a")) == ["a"]
      @test names(SC.BINARY(ref("a"), SC.OP_ADD, ref("b"))) == ["a", "b"]
      @test names(SC.IFEXP(ref("c"), ref("t"), ref("e"))) == ["c", "t", "e"]
      @test names(SC.RSUB(ref("rec"), 1, "re", SC.TYPE_REAL())) == ["rec"]
      @test names(SC.CALL(:f, SC.Exp[ref("p"), ref("q")], SC.TYPE_REAL(), SC.CALL_FUNCTION)) == ["p", "q"]
      @test names(SC.ARRAY_EXP(SC.TYPE_REAL(), true, SC.Exp[ref("u"), ref("v")])) == ["u", "v"]
      @test names(SC.DER(SC.SimCref(:x))) == ["x"]
      @test names(SC.PRE(SC.SimCref(:d), SC.TYPE_BOOL())) == ["d"]
      #= an element by literal subscripts: its name and its array's =#
      local rt = names(SC.ASUB(ref("R_T"), SC.Exp[SC.ICONST(1), SC.ICONST(1)]))
      @test "R_T[1][1]" in rt && "R_T" in rt
      #= by a subscript that reads a variable: the array's dynamic mark =#
      @test string("A", SC.DYNAMIC_SUBSCRIPT_MARK) in names(SC.ASUB(ref("A"), SC.Exp[ref("i", SC.TYPE_INTEGER())]))
    end

    @testset "ifConditionNames!: the names of if-expression conditions" begin
      local names = e -> SC.ifConditionNames!(SC.OrderedSet{String}(), e)
      @test names(SC.IFEXP(ref("p"), ref("t"), ref("e"))) == SC.OrderedSet(["p"])
      @test "b" in names(SC.IFEXP(ref("a"), SC.IFEXP(ref("b"), ref("c"), ref("d")), ref("e")))
      @test "x" in names(SC.BINARY(SC.IFEXP(ref("x"), ref("y"), ref("z")), SC.OP_ADD, ref("w")))
      @test isempty(names(SC.BINARY(ref("m"), SC.OP_ADD, ref("n"))))
    end

    @testset "OMJLIR types in the simulation code (simType)" begin
      @test SC.simType(IR.T_REAL()) isa SC.TYPE_REAL
      @test SC.simType(IR.T_INTEGER()) isa SC.TYPE_INTEGER
      @test SC.simType(IR.T_BOOLEAN()) isa SC.TYPE_BOOL
      @test SC.simType(IR.T_STRING()) isa SC.TYPE_STRING
      @test SC.simType(IR.T_UNKNOWN()) isa SC.TYPE_UNKNOWN
      @test SC.simType(IR.T_NORETCALL()) isa SC.TYPE_NORETCALL
      local at = SC.simType(IR.T_ARRAY(IR.T_INTEGER(), IR.Dimension[IR.DIM_SIZE(2), IR.DIM_SIZE(5)]))
      @test at isa SC.TYPE_ARRAY && at.elementType isa SC.TYPE_INTEGER && at.dims == [2, 5]
      local rt = SC.simType(IR.T_RECORD(:Complex, [IR.Field(:re, IR.T_REAL()), IR.Field(:im, IR.T_REAL())]))
      @test rt isa SC.TYPE_COMPLEX && rt.isRecord && rt.fieldNames == ["re", "im"]
      local tt = SC.simType(IR.T_TUPLE(IR.ModelicaType[IR.T_REAL(), IR.T_BOOLEAN()]))
      @test tt isa SC.TYPE_TUPLE && tt.types[2] isa SC.TYPE_BOOL
    end

    @testset "expToJuliaExpMTK: symbolic names, arithmetic Booleans" begin
      local sc = M.mockSimCode([M.simVar("x", SC.ALG_VARIABLE(0)), M.simVar("s", SC.STATE())])
      local mtk = e -> CG.expToJuliaExpMTK(e, sc)
      @test value(mtk(SC.BCONST(true))) == true
      @test value(mtk(SC.ICONST(7))) == 7
      @test value(mtk(SC.BINARY(R(2.0), SC.OP_ADD, R(3.0)))) == 5.0
      @test value(mtk(neg(R(4.0)))) == -4.0
      @test value(mtk(SC.RELATION(R(1.0), SC.OP_LESS, R(2.0), 0))) == true
      #= a Boolean `and` as a product (MTK's symbolic conditions) =#
      @test value(mtk(SC.LBINARY(SC.BCONST(true), SC.OP_AND, SC.BCONST(false)))) == 0
      @test value(mtk(SC.BINARY(x, SC.OP_MUL, R(2.0))); x = 3.0) == 6.0
      local ifx = mtk(SC.IFEXP(SC.RELATION(x, SC.OP_GREATER, R(0.0), 0), R(1.0), R(2.0)))
      @test value(ifx; x = 3.0) == 1.0 && value(ifx; x = -1.0) == 2.0
      @test value(mtk(ref("time")); t = 0.25) == 0.25
      @test value(mtk(SC.DER(SC.SimCref(:s))); D = f -> 2f, s = 1.5) == 3.0
    end

    @testset "expToJuliaExp: the solver's arrays" begin
      #= a variable by its index (`x[i]` of the solver's state, p[i] of a parameter), time t =#
      local sc = M.mockSimCode([M.simVar("x", SC.ALG_VARIABLE(0)), M.simVar("p", SC.PARAMETER(nothing)), M.simVar("s", SC.STATE())])
      local gen = e -> CG.expToJuliaExp(e, sc)
      local (ix, is) = (first(sc.stringToSimVarHT["x"]), first(sc.stringToSimVarHT["s"]))
      local u = zeros(max(ix, is)); u[ix] = 3.0; u[is] = 5.0
      @test value(gen(SC.SCONST("str"))) == "str"
      @test value(gen(SC.LBINARY(SC.BCONST(true), SC.OP_AND, SC.BCONST(false)))) == false
      @test value(gen(SC.BINARY(x, SC.OP_MUL, R(2.0))); x = u) == 6.0
      @test value(gen(ref("s")); x = u) == 5.0
      @test value(gen(ref("time")); t = 0.5) == 0.5
      local ifx = gen(SC.IFEXP(SC.RELATION(x, SC.OP_GREATER, R(0.0), 0), R(1.0), R(2.0)))
      @test value(ifx; x = u) == 1.0
      #= der() is no expression over the solver's arrays =#
      @test_throws OMBackend.UnsupportedLowering gen(SC.DER(SC.SimCref(:s)))
    end

    @testset "expToJuliaExpDE: states u[i], parameters p[i]" begin
      local sc = M.mockSimCode([M.simVar("s", SC.STATE()), M.simVar("p", SC.PARAMETER(nothing))])
      local layout = CG.DEGen.buildDELayout(sc)
      local de = e -> CG.DEGen.expToJuliaExpDE(e, layout, sc)
      local (u, p) = ([5.0], [2.0])
      @test value(de(SC.SCONST("s"))) == "s"
      @test value(de(SC.CAST(SC.TYPE_REAL(), R(1.0)))) == 1.0
      @test value(de(call(:sin, R(1.0)))) == sin(1.0)
      @test value(de(ref("s")); u = u) == 5.0
      @test value(de(ref("p")); p = p) == 2.0
      @test value(de(ref("time")); t = 0.5) == 0.5
      @test value(de(SC.BINARY(ref("s"), SC.OP_MUL, R(2.0))); u = u) == 10.0
      @test value(de(SC.IFEXP(SC.RELATION(ref("s"), SC.OP_GREATER, R(0.0), 0), ref("s"), R(2.0))); u = u) == 5.0
    end

    @testset "the checks of the simulation code" begin
      #= a residual reading a name that is no variable; a variable name that is not flat =#
      local sc = M.mockSimCode([M.simVar("x", SC.ALG_VARIABLE(0)), M.simVar("a.b", SC.ALG_VARIABLE(0))])
      push!(sc.residualEquations, SC.RESIDUAL_EQUATION(SC.BINARY(x, SC.OP_SUB, ref("undefinedName")), SC.NO_SOURCE, SC.EQ_ATTR_DEFAULT))
      local violations = SC.check(sc).violations
      @test any(v -> v.rule === :cref_resolution && occursin("undefinedName", v.detail), violations)
      @test any(v -> v.rule === :canonical_cref_names && v.where == "a.b", violations)
      @test !any(v -> v.rule === :cref_resolution && occursin("`x`", v.detail), violations)
    end

    @testset "evalConstant of literals" begin
      for (lit, val) in Any[(R(2.5), 2.5), (SC.ICONST(3), 3), (SC.BCONST(true), true), (SC.SCONST("hi"), "hi")]
        @test CG.evalConstant(lit) == val
      end
    end
  end
end
