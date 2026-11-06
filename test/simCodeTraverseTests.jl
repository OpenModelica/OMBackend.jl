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

    @testset "Traversal: one traversal, field by field, for SimCode and OMJLIR" begin
      #= every type a traversal rebuilds has its default constructor (a component reference,
         whose flat name follows its parts, is rebuilt by its own rule) =#
      local concrete = (m, S) -> filter(T -> T isa DataType && isconcretetype(T) && T <: S, Any[getfield(m, n) for n in names(m; all = true)])
      local rebuilt = vcat(concrete(SC, SC.Exp), concrete(SC, SC.Statement), concrete(SC, SC.Equation), concrete(SC, SC.Construct),
                           Any[SC.Iterator, SC.Branch, SC.WHEN_STMTS],
                           concrete(IR, IR.Expression), concrete(IR, IR.Statement), concrete(IR, IR.Subscript), concrete(IR, IR.ClockKind),
                           Any[IR.Iterator, IR.StatementBranch, IR.WhenBranch, IR.EquationBranch, IR.CrefPart])
      @test length(rebuilt) > 80
      for T in rebuilt
        @test hasmethod(T, Tuple{fieldtypes(T)...})
      end
      local gt0 = e -> SC.RELATION(e, SC.OP_GREATER, R(0.0), 0)
      #= children in field order: a reduction's iterator ranges after its body, a range's absent step skipped =#
      local r = SC.RANGE(SC.TYPE_INTEGER(), SC.ICONST(1), nothing, SC.ICONST(3))
      @test SC.children(r) == SC.Exp[SC.ICONST(1), SC.ICONST(3)]
      @test SC.children(SC.REDUCTION(:sum, x, SC.Iterator[SC.Iterator(:i, r)], SC.TYPE_REAL())) == SC.Exp[x, r]
      @test SC.children(SC.IFEXP(SC.BCONST(true), x, y)) == SC.Exp[SC.BCONST(true), x, y]
      @test isempty(SC.children(x))
      #= a one-level map: the node itself if no child changed, else rebuilt with its leaves kept =#
      local rsub = SC.RSUB(x, 2, "re", SC.TYPE_REAL())
      @test SC.mapChildren(identity, rsub) === rsub
      @test SC.mapChildren(c -> c == x ? y : c, rsub) == SC.RSUB(y, 2, "re", SC.TYPE_REAL())
      @test SC.anyExp(c -> c == y, SC.BINARY(x, SC.OP_ADD, SC.UNARY(SC.OP_UMINUS, y)))
      @test !SC.anyExp(c -> c == y, rsub)
      #= statements: the expressions of nested statements, the iterator's range; reinit's target kept =#
      local s = SC.FOR(SC.Iterator(:i, SC.RANGE(SC.TYPE_INTEGER(), SC.ICONST(1), nothing, x)),
                       SC.Statement[SC.IF(SC.Branch[SC.Branch(x, SC.Statement[SC.REINIT(SC.SimCref(:x), x, SC.NO_SOURCE)])],
                                          SC.Statement[SC.ASSIGN(y, x, SC.NO_SOURCE)], SC.NO_SOURCE)], SC.NO_SOURCE)
      local xToZ = e -> SC.substitute(e, alias("x", "z"))
      local s2 = SC.mapStatementExps(xToZ, s)
      @test s2.iterator.range.stop == ref("z")
      @test s2.body[1].branches[1].condition == ref("z")
      @test s2.body[1].branches[1].body[1] == SC.REINIT(SC.SimCref(:x), ref("z"), SC.NO_SOURCE)
      @test s2.body[1].elseBody[1] == SC.ASSIGN(y, ref("z"), SC.NO_SOURCE)
      @test SC.mapStatementExps(identity, s) === s
      #= substitution reaches a nested assignment's own rule: a negated target's negation moved to the value =#
      local ifs = SC.IF(SC.Branch[SC.Branch(SC.BCONST(true), SC.Statement[SC.ASSIGN(x, R(1.0), SC.NO_SOURCE)])], SC.Statement[], SC.NO_SOURCE)
      @test SC.substitute(ifs, alias("x", "y"; negated = true)).branches[1].body[1] == SC.ASSIGN(y, neg(R(1.0)), SC.NO_SOURCE)
      #= a when-clause's elsewhen arms, an equation =#
      local w = SC.WHEN_STMTS(gt0(x), SC.Statement[], SC.WHEN_STMTS(gt0(y), SC.Statement[SC.ASSIGN(ref("w"), x, SC.NO_SOURCE)], nothing))
      local w2 = SC.substitute(w, alias("x", "z"))
      @test w2.condition == gt0(ref("z")) && w2.elsewhenPart.condition == gt0(y)
      @test w2.elsewhenPart.body[1] == SC.ASSIGN(ref("w"), ref("z"), SC.NO_SOURCE)
      local eq = SC.EQUATION(x, y, SC.NO_SOURCE, SC.EQ_ATTR_DEFAULT)
      @test SC.substitute(eq, alias("x", "z")).lhs == ref("z")
      #= OMJLIR: a reference's subscripts are children, its flat name follows them =#
      local xi = IR.CREF(IR.ComponentRef(IR.CrefPart[IR.CrefPart(:x, IR.Subscript[IR.INDEX(IR.CREF(IR.ComponentRef(:i, IR.T_INTEGER()), IR.T_INTEGER()))], IR.T_REAL())]), IR.T_REAL())
      @test IR.children(xi) == IR.Expression[IR.CREF(IR.ComponentRef(:i, IR.T_INTEGER()), IR.T_INTEGER())]
      local x2 = IR.mapExp(e -> (e isa IR.CREF && e.cref.name === :i) ? IR.INTEGER(2) : e, xi)
      @test x2.cref.name === Symbol("x[2]")
      @test IR.mapExp(identity, xi) === xi
      #= OMJLIR's equations are entities: mapped in place, nested equations too, their identity kept =#
      local inner = IR.SCALAR_EQUATION(xi, IR.REAL(1.0), IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
      local ifEq = IR.IF_EQUATION(IR.EquationBranch[IR.EquationBranch(IR.CREF(IR.ComponentRef(:i, IR.T_INTEGER()), IR.T_BOOLEAN()), IR.Equation[inner])],
                                  IR.Equation[], IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
      @test IR.mapEquation!(e -> (e isa IR.CREF && e.cref.name === :i) ? IR.INTEGER(2) : e, ifEq) === ifEq
      @test ifEq.branches[1].equations[1] === inner
      @test inner.lhs.cref.name === Symbol("x[2]")
      @test ifEq.branches[1].condition == IR.INTEGER(2)
      local roots = IR.Expression[]
      IR.foreachRootExp(e -> push!(roots, e), ifEq)
      @test roots == IR.Expression[IR.INTEGER(2), inner.lhs, IR.REAL(1.0)]
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

    @testset "tryEvalNumeric: parameters through their bindings, sqrt" begin
      #= MSL 4 CompareLosslessLines: the Pulse's period td = 1/c0, c0 = 1/sqrt(l*c); without sqrt
         its when missed the periodic path and wrote T_start at a step's end =#
      local c0 = SC.BINARY(R(1.0), SC.OP_DIV, call(:sqrt, SC.BINARY(R(2.5e-7), SC.OP_MUL, R(1.0e-10))))
      local sc = M.mockSimCode([M.simVar("c0", SC.PARAMETER(c0)), M.simVar("td", SC.PARAMETER(SC.BINARY(R(1.0), SC.OP_DIV, ref("c0"))))])
      @test SC.tryEvalNumeric(ref("td"), sc) ≈ 5.0e-9
      @test SC.tryEvalNumeric(call(:sqrt, R(-1.0)), sc) === nothing
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
      #= by a subscript that reads a variable: the array's read, no name of an element; the read marked dynamic
         (`crefReads!`), the array recorded in `dynamic` by its reference (`arrayBase`) =#
      local aRef = OMBackend.OMJLIR.ComponentRef(:A)
      local dynamicRead = SC.ASUB(SC.EXP_CREF(SC.SimCref(:A, Int[], aRef), SC.TYPE_REAL()), SC.Exp[ref("i", SC.TYPE_INTEGER())])
      @test names(dynamicRead) == ["A", "i"]
      local dynamic = Set{Symbol}()
      SC.crefNames!(SC.OrderedSet{String}(), dynamicRead; dynamic)
      @test dynamic == Set([:A])
      @test SC.Read(SC.SimCref(:A), true) in SC.crefReads!(SC.Read[], dynamicRead)
      #= an element read: the array's key and subscripts, its reference the element's =#
      local element = first(SC.crefReads!(SC.Read[], SC.ASUB(SC.EXP_CREF(SC.SimCref(:A, Int[], aRef), SC.TYPE_REAL()),
                                                             SC.Exp[SC.ICONST(2)])))
      @test SC.crefName(element.cref) == "A[2]" && element.cref.ref.name === Symbol("A[2]") && !element.dynamic
      #= the two key spaces of an element read: the passes' IR flat name, the code view's identifier (the array's
         identifier and the subscripts, as the generated code indexes it) =#
      local abRef = OMBackend.OMJLIR.ComponentRef(OMBackend.OMJLIR.CrefPart[OMBackend.OMJLIR.CrefPart(:a, OMBackend.OMJLIR.Subscript[], OMBackend.OMJLIR.T_REAL()),
                                                                         OMBackend.OMJLIR.CrefPart(:b, OMBackend.OMJLIR.Subscript[], OMBackend.OMJLIR.T_REAL())])
      local abRead = SC.ASUB(SC.EXP_CREF(SC.SimCref(:a_b, Int[], abRef), SC.TYPE_REAL()), SC.Exp[SC.ICONST(1), SC.ICONST(2)])
      @test "a.b[1,2]" in SC.crefNames!(SC.OrderedSet{String}(), abRead)
      @test collect(SC.identifierNames!(SC.OrderedSet{String}(), abRead)) == ["a_b[1][2]", "a_b"]
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
