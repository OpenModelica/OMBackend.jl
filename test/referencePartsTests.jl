#= OMJLIR's references by their parts: a name is derived from them (`flatName`, `legacyName`), never taken apart.
   A slice of an array split along its inner dimensions (`T_SLICE`), and the algorithm lifter's questions by parts:
   which references overlap, the wholes an assignment's target is a part of, the arrays equations constrain; the
   interpreter's values by paths (Evaluation's Interpret.jl), a scalar's wholes by its reference's parts. The
   reduced models' references (state sets, derivatives, array form) are checked in indexReductionTests
   (`referenceProblems`). =#
using Test
import OMBackend

@testset "References by their parts" begin
  local IRp = OMBackend.OMJLIR
  local P = OMBackend.Passes
  local part = (n::Symbol, subs::Vector{Int} = Int[], ty::IRp.ModelicaType = IRp.T_REAL()) ->
    IRp.CrefPart(n, IRp.Subscript[IRp.INDEX(IRp.INTEGER(k)) for k in subs], ty)
  local ref = (ps::IRp.CrefPart...) -> IRp.ComponentRef(IRp.CrefPart[ps...])

  @testset "a slice's inner subscripts in its type" begin
    local T = IRp.T_ARRAY(IRp.T_REAL(), IRp.Dimension[IRp.DIM_SIZE(3), IRp.DIM_SIZE(3)])
    local G = IRp.CrefPart(Symbol("a|b"), IRp.Subscript[], IRp.T_COMPONENTS(:C, [:a, :b]))
    local slice = P.IndexReduction.sliceCref(ref(G, part(:f), part(:T, Int[], T)), [2, 1])
    @test slice.name === Symbol("a|b.f.T[2,1]") && slice.parts[end].ty == IRp.T_SLICE([2, 1], T)
    @test P.legacyName(slice) == "a|b_f_T[2][1]"
    #= an element of the slice: the instance's subscript on the group's part (named as the member), a derivative's
       on its last part =#
    local element = IRp.ComponentRef(IRp.CrefPart[IRp.CrefPart(G.name, IRp.Subscript[IRp.INDEX(IRp.INTEGER(2))], G.ty); slice.parts[2:end]])
    @test element.name === Symbol("b.f.T[2,1]") && P.unsubscriptedName(element) === slice.name
    @test IRp.withLastSubscripts(IRp.derivativeCref(slice), IRp.Subscript[IRp.INDEX(IRp.INTEGER(2))]).name === Symbol("\$DER.a|b.f.T[2,1][2]")
  end

  @testset "references overlap by their parts" begin
    @test IRp.crefsOverlap(ref(part(:buf)), ref(part(:buf, [2])))
    @test IRp.crefsOverlap(ref(part(:r)), ref(part(:r), part(:x)))
    @test IRp.crefsOverlap(ref(part(:r, [1])), ref(part(:r, [1]), part(:x)))
    @test !IRp.crefsOverlap(ref(part(:buf, [1])), ref(part(:buf, [12])))
    @test !IRp.crefsOverlap(ref(part(:x)), ref(part(:xy)))
    #= a row covers its elements; a member of rolled components by its name (`T_MEMBER`) =#
    @test IRp.crefsOverlap(ref(part(:m, [1])), ref(part(:m, [1, 2])))
    @test !IRp.crefsOverlap(ref(part(:m, [2])), ref(part(:m, [1, 2])))
    @test IRp.crefsOverlap(ref(part(:r), part(:a, [2])), ref(part(:r), part(:a, [2]), part(:b)))
    @test !IRp.crefsOverlap(ref(part(:r), part(:a, [2])), ref(part(:r), part(:a, [3]), part(:b)))
    local members = IRp.T_COMPONENTS(:C, [:c1, :c2])
    @test IRp.crefsOverlap(ref(IRp.CrefPart(:c, IRp.Subscript[IRp.INDEX(IRp.INTEGER(2))], members), part(:x)), ref(part(:c2), part(:x)))
    @test !IRp.crefsOverlap(ref(IRp.CrefPart(:c, IRp.Subscript[IRp.INDEX(IRp.INTEGER(1))], members), part(:x)), ref(part(:c2), part(:x)))
  end

  @testset "a member of rolled components: a part of its own type" begin
    local members = IRp.T_COMPONENTS(:C, [:c1, :c2, :c3])
    local G = IRp.CrefPart(Symbol("c1|c2|c3"), IRp.Subscript[], members)
    local index = (k::Int) -> IRp.Subscript[IRp.INDEX(IRp.INTEGER(k))]
    local m = IRp.CrefPart(G.name, index(2), members)
    #= the group part subscripted by a literal is the member: a name, its index in its type =#
    @test m.name === :c2 && isempty(m.subscripts) && m.ty == IRp.T_MEMBER(G.name, members, 2) && IRp.isMember(m)
    @test_throws ArgumentError IRp.CrefPart(:c2, index(1), m.ty)
    local (x::IRp.ComponentRef, aux::IRp.ComponentRef) = (ref(m, part(:x)), ref(m))
    @test x.name === Symbol("c2.x") && P.legacyName(x) == "c2_x" && IRp.upToFirstSubscripted(x) == x
    #= in the array's view the group's element =#
    @test IRp.hasSubscripts(x) && P.unsubscriptedName(x) === Symbol("c1|c2|c3.x") && P.unsubscriptedName(aux) === G.name
    @test P.allSubscripts(x) == index(2) && IRp.subscripts(aux) == index(2)
    @test IRp.withoutLastSubscripts(aux).name === G.name && IRp.withSubscripts(m, index(3)).name === :c3
    @test P.IndexReduction.isRolledReference(x) && IRp.isRolled(G.ty) && !IRp.isMember(G)
    #= the whole rolled array overlaps its member; another member and an iterator's element do not =#
    local it = IRp.CrefPart(G.name, IRp.Subscript[IRp.INDEX(IRp.CREF(IRp.ComponentRef(:i), IRp.T_INTEGER()))], members)
    @test IRp.crefsOverlap(ref(G), aux) && IRp.crefsOverlap(ref(G), ref(G, part(:x)))
    @test !IRp.crefsOverlap(ref(IRp.CrefPart(G.name, index(1), members)), aux) && !IRp.crefsOverlap(ref(it), aux)
    @test !IRp.isMember(it) && length(it.subscripts) == 1
  end

  @testset "an auxiliary for der(x): an operator part before x's parts" begin
    local x = ref(part(:a), part(:b, [1]))
    local aux = IRp.auxiliaryDerivativeCref(x)
    @test aux.name === Symbol("der.a.b[1]") && aux.parts[1].ty isa IRp.T_DERAUX && P.legacyName(aux) == "der_a_b[1]"
    @test !IRp.isDerivativeCref(aux) && IRp.operandStart(aux) == 1 && aux.parts[2:end] == x.parts
    @test IRp.auxiliaryDerivativeCref(aux).name === Symbol("der.der.a.b[1]")
  end

  @testset "equations alike as written; their forms up to the order of sides, terms and factors" begin
    local R = P.IndexReduction
    local x = (n::Symbol) -> IRp.CREF(ref(part(n)), IRp.T_REAL())
    local op = (a, o, b) -> IRp.BINARY(a, o, b, IRp.T_REAL())
    local eq = (l, r; source = IRp.NO_SOURCE) -> IRp.SCALAR_EQUATION(l, r, IRp.T_REAL(), source, IRp.DEFAULT_ATTRIBUTES)
    #= two occurrences: other sources, other relations' indices (zero crossings) alike; another operator not =#
    local rel = (k::Int, o) -> IRp.IF_EXPRESSION(IRp.RELATION(x(:a), o, x(:b), k), x(:a), x(:b), IRp.T_REAL())
    @test IRp.sameForm(eq(x(:y), rel(1, IRp.GREATER); source = IRp.Source("A.mo", 1, 1)), eq(x(:y), rel(7, IRp.GREATER)))
    @test !IRp.sameForm(eq(x(:y), rel(1, IRp.GREATER)), eq(x(:y), rel(1, IRp.LESS)))
    @test IRp.formHash(eq(x(:y), rel(1, IRp.GREATER)), UInt(0)) == IRp.formHash(eq(x(:y), rel(3, IRp.GREATER)), UInt(0))
    #= a + b = c is c - b = a; y = a * b is b * a = y; a = b is not a = c =#
    @test R.equationForm(eq(op(x(:a), IRp.ADD, x(:b)), x(:c))) == R.equationForm(eq(op(x(:c), IRp.SUB, x(:b)), x(:a)))
    @test R.equationForm(eq(x(:y), op(x(:a), IRp.MUL, x(:b)))) == R.equationForm(eq(op(x(:b), IRp.MUL, x(:a)), x(:y)))
    @test R.equationForm(eq(x(:a), x(:b))) != R.equationForm(eq(x(:a), x(:c)))
    @test hash(R.equationForm(eq(x(:a), x(:b)))) == hash(R.equationForm(eq(x(:b), x(:a))))
  end

  @testset "the interpreter's values by paths, the wholes of a reference by its parts" begin
    local Ev = P.Evaluation
    #= the records and arrays a scalar of the model is a part of, the longest first, and the paths from them =#
    @test Ev.wholesOf(ref(part(:R0), part(:T, [1, 2]))) == [(Symbol("R0.T"), Ev.Step[[1, 2]]), (:R0, Ev.Step[:T, [1, 2]])]
    @test Ev.wholesOf(ref(part(:a, [1]), part(:b, [2]))) ==
      [(Symbol("a[1].b"), Ev.Step[[2]]), (Symbol("a[1]"), Ev.Step[:b, [2]]), (:a, Ev.Step[[1], :b, [2]])]
    #= an array of records: an element, its field; a matrix as an array and back, a row of it =#
    local v = Ev.FlatValue(Ev.Step[[1], :x] => 1.0, Ev.Step[[2], :x] => 2.0)
    @test Ev.flatField(Ev.elementOf(v, [2]), :x) == Ev.scalarValue(2.0)
    local m = Ev.fromArray([1.0 2.0; 3.0 4.0])
    @test Ev.toArray(m) == [1.0 2.0; 3.0 4.0] && Ev.elementOf(m, [2]) == Ev.fromArray([3.0, 4.0])
    #= an element's path under an array literal's index: its indices after the index =#
    @test Ev.underStep([1], Ev.Step[[2], :f]) == Ev.Step[[1, 2], :f] && Ev.underStep([1], Ev.Step[:f]) == Ev.Step[[1], :f]
  end

  @testset "an algorithm's wholes; the arrays equations constrain" begin
    @test P.wholesOf(ref(part(:r), part(:a, [2]), part(:b))) == Symbol[:r, Symbol("r.a"), Symbol("r.a[2]")]
    @test P.wholesOf(ref(part(:x, [2]))) == Symbol[:x]
    local eq = IRp.SCALAR_EQUATION(IRp.CREF(ref(part(:c, [1]), part(:T)), IRp.T_REAL()), IRp.REAL(1.0), IRp.T_REAL(),
                                   IRp.NO_SOURCE, IRp.DEFAULT_ATTRIBUTES)
    local constrained = P.constrainedBy(IRp.Equation[eq])
    @test Symbol("c[1].T") in constrained.names && :c in constrained.arrays && !(Symbol("c[1].T") in constrained.arrays)
  end
end
