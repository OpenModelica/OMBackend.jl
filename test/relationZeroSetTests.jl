#=
Relations that share a zero set get one event callback whose crossing pins all
of them (CodeGeneration._groupRelationsByZeroSet / _zeroSetPins). Two callbacks
on the same root used to fire alternately and flip a discrete cluster forever
(MSL Rotational OneWayClutchDisengaged).
=#
@testset "Relation zero sets" begin
  local CG = OMBackend.CodeGeneration
  local ty = DAE.T_REAL_DEFAULT
  local w = DAE.CREF(DAE.CREF_IDENT("w", ty, MetaModelica.nil), ty)
  local zero = DAE.RCONST(0.0)
  local rel(a, op, b) = DAE.RELATION(a, op, b, -1, NONE())
  local le = rel(w, DAE.LESSEQ(ty), zero)        # w <= 0
  local gt = rel(w, DAE.GREATER(ty), zero)       # w > 0
  local ge = rel(zero, DAE.GREATEREQ(ty), w)     # 0 >= w, same as w <= 0
  local other = rel(w, DAE.GREATER(ty), DAE.RCONST(1.0))

  local groups = CG._groupRelationsByZeroSet(Any[le, other, gt, ge])
  @test length(groups) == 2
  @test groups[1] == Any[le, gt, ge]
  @test groups[2] == Any[other]

  local holds = CG._zeroSetPins(groups[1], le, true)     # w <= 0 became true
  @test holds[string(le)] && !holds[string(gt)] && holds[string(ge)]
  local fails = CG._zeroSetPins(groups[1], le, false)    # w <= 0 became false
  @test !fails[string(le)] && fails[string(gt)] && !fails[string(ge)]

  @test CG._relationZeroSet(rel(w, DAE.EQUAL(ty), zero)) === nothing
  @test length(CG._groupRelationsByZeroSet(Any[rel(w, DAE.EQUAL(ty), zero), le])) == 2
end
