#= The names a Modelica function's statements read (SimulationCode/parameterElimination.jl):
   the walker read `exp1` of an if statement and of an array assignment, a FieldError that
   a silent catch swallowed, so the names in if conditions, else branches, array assignments
   (and, not walked, tuple assignments and for ranges) were missed and the parameters they
   read could be eliminated. =#
using Test
import OMBackend
import DAE
import MetaModelica

@testset "Names read by Modelica function statements" begin
  local SC = OMBackend.SimulationCode
  local nil = MetaModelica.nil
  local ty = DAE.T_REAL(nil)
  local cref(n) = DAE.CREF(DAE.CREF_IDENT(n, ty, nil), ty)
  local src = DAE.emptyElementSource
  local assign(l, r) = DAE.STMT_ASSIGN(ty, cref(l), cref(r), src)
  local stmts = [
    DAE.STMT_IF(cref("c"), MetaModelica.list(assign("x", "a")),
                DAE.ELSEIF(cref("d"), MetaModelica.list(assign("x", "b")),
                           DAE.ELSE(MetaModelica.list(assign("x", "e")))), src),
    DAE.STMT_TUPLE_ASSIGN(ty, MetaModelica.list(cref("y1"), cref("y2")), cref("p"), src),
    DAE.STMT_FOR(ty, false, "i", -1, DAE.RANGE(DAE.T_INTEGER(nil), DAE.ICONST(1), MetaModelica.NONE(), cref("n")),
                 MetaModelica.list(assign("z", "q")), src),
    DAE.STMT_ASSERT(cref("ok"), DAE.SCONST("message"), DAE.ICONST(1), src),
  ]
  local out = SC.OrderedSet{String}()
  SC._walkStatementsForCrefs!(out, stmts)
  for name in ("c", "a", "d", "b", "e", "y1", "y2", "p", "n", "z", "q", "ok")
    @test name in out
  end
end
