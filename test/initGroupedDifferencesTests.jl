#= The initialization's finite-difference Jacobian with the unknowns each right-hand side row reads
   (_fdInitResidualJacobian, ErasedRHS's structure): the columns that share no row are perturbed together, one
   right-hand side call per group, and the Jacobian equals the column-by-column one bit for bit, the initialization
   equations' rows (`extraRes`, reading the derivatives) too. MSL BatchPlant_StandardWater: 160 calls a Jacobian. =#
using Test
import OMBackend

@testset "Initialization Jacobian by grouped differences" begin
  local CG = OMBackend.CodeGeneration
  local calls = Ref(0)
  #= a chain: row i reads u[i] and u[i + 1]; the last row a condition on u[1] =#
  local f = (du, u, p, t) -> begin
    calls[] += 1
    for i in 1:5
      du[i] = sin(u[i]) * u[i + 1] + p[1]
    end
    du[6] = ifelse(u[1] > 0.5, u[6]^2, -u[6])
    nothing
  end
  local reads = Pair{Int, Vector{Int}}[1 => [1, 2], 2 => [2, 3], 3 => [3, 4], 4 => [4, 5], 5 => [5, 6], 6 => [1, 6]]
  local structure = CG.SA.sparse(Int[r for (r, cs) in reads for _ in cs], Int[c for (_, cs) in reads for c in cs],
                                 ones(12), 6, 6)
  local u0 = [0.3, 1.1, -0.7, 2.0, 0.9, 1.5]
  local p = [0.25]
  local eqIdx = [1, 2, 3, 6]
  local targets = zeros(4)
  local extra = CG.ErasedResiduals((du, u) -> [du[4] * u[5] - 1.0, du[5] + u[2]])
  local vars = [1, 2, 3, 4, 5, 6]
  local du = similar(u0)
  f(du, u0, p, 0.0)
  local res = CG._initResidualVec(du, u0, eqIdx, targets, extra)
  calls[] = 0
  local dense = CG._fdInitResidualJacobian(CG.ErasedRHS(f), p, u0, eqIdx, targets, extra, vars, res)
  local denseCalls = calls[]
  calls[] = 0
  local grouped = CG._fdInitResidualJacobian(CG.ErasedRHS(f, structure), p, u0, eqIdx, targets, extra, vars, res)
  @test grouped == dense
  @test denseCalls == 6
  #= one call per group and the unperturbed one; a group's columns read by no common row =#
  local groups = CG._columnGroups(structure, vars)
  @test calls[] == length(groups) + 1 < denseCalls
  local rowsReading = c -> Set(CG.SA.rowvals(structure)[CG.SA.nzrange(structure, c)])
  @test all(g -> all(a == b || isempty(intersect(rowsReading(vars[a]), rowsReading(vars[b]))) for a in g for b in g), groups)
  @test sort!(reduce(vcat, groups)) == collect(eachindex(vars))
  #= a structure of other size: column by column =#
  @test CG._rhsStructure(CG.ErasedRHS(f, structure), 7) === nothing
end
