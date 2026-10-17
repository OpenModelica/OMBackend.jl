#= The names a Modelica function's statements read (SimCode functionBodyNames!, the parameters
   the functions read are kept): the walker once read `exp1` of an if statement and of an array
   assignment, a FieldError that a silent catch swallowed, so the names in if conditions, else
   branches, array assignments (and, not walked, tuple assignments and for ranges) were missed
   and the parameters they read could be eliminated. =#
using Test
import OMBackend

@testset "Names read by Modelica function statements" begin
  local SC = OMBackend.SimCode
  local ref(n) = SC.EXP_CREF(SC.SimCref(Symbol(n)), SC.TYPE_REAL())
  local src = SC.NO_SOURCE
  local assign(l, r) = SC.ASSIGN(ref(l), ref(r), src)
  local statements = SC.Statement[
    SC.IF([SC.Branch(ref("c"), SC.Statement[assign("x", "a")]), SC.Branch(ref("d"), SC.Statement[assign("x", "b")])],
          SC.Statement[assign("x", "e")], src),
    SC.ASSIGN(SC.TUPLE(SC.Exp[ref("y1"), ref("y2")]), ref("p"), src),
    SC.FOR(SC.Iterator(:i, SC.RANGE(SC.TYPE_ARRAY(SC.TYPE_INTEGER(), [-1]), SC.ICONST(1), nothing, ref("n"))),
           SC.Statement[assign("z", "q")], src),
    SC.ASSERT(ref("ok"), SC.SCONST("message"), SC.ICONST(1), src),
  ]
  local f = SC.MODELICA_FUNCTION("f", SC.FunctionVariable[], SC.FunctionVariable[],
                                 SC.FunctionVariable[SC.FunctionVariable("l", SC.TYPE_REAL(), SC.Exp[], ref("w"), Pair{String, SC.Exp}[])],
                                 statements, nothing)
  local out = SC.functionBodyNames!(SC.OrderedSet{String}(), SC.ModelicaFunction[f])
  for name in ("c", "a", "d", "b", "e", "y1", "y2", "p", "n", "z", "q", "ok", "w")
    @test name in out
  end
end
