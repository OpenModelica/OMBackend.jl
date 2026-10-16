#=
Relations that share a zero set get one event callback whose crossing pins all
of them (CodeGeneration._groupRelationsByZeroSet / _zeroSetPins). Two callbacks
on the same root used to fire alternately and flip a discrete cluster forever
(MSL Rotational OneWayClutchDisengaged).
=#
@testset "Relation zero sets" begin
  local CG = OMBackend.CodeGeneration
  local SC = OMBackend.SimCode
  local w = SC.EXP_CREF(SC.SimCref(:w), SC.TYPE_REAL())
  local zero = SC.RCONST(0.0)
  local rel(a, op, b) = SC.RELATION(a, op, b, -1)
  local le = rel(w, SC.OP_LESSEQ, zero)        # w <= 0
  local gt = rel(w, SC.OP_GREATER, zero)       # w > 0
  local ge = rel(zero, SC.OP_GREATEREQ, w)     # 0 >= w, same as w <= 0
  local other = rel(w, SC.OP_GREATER, SC.RCONST(1.0))

  local groups = CG._groupRelationsByZeroSet(SC.RELATION[le, other, gt, ge])
  @test length(groups) == 2
  @test groups[1] == SC.RELATION[le, gt, ge]
  @test groups[2] == SC.RELATION[other]

  local holds = CG._zeroSetPins(groups[1], le, true)     # w <= 0 became true
  @test holds[le] && !holds[gt] && holds[ge]
  local fails = CG._zeroSetPins(groups[1], le, false)    # w <= 0 became false
  @test !fails[le] && fails[gt] && !fails[ge]

  @test CG._relationZeroSet(rel(w, SC.OP_EQUAL, zero)) === nothing
  @test length(CG._groupRelationsByZeroSet(SC.RELATION[rel(w, SC.OP_EQUAL, zero), le])) == 2
end
