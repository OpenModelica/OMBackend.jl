#=
The initialization's one-unknown rows (CodeGeneration/DAEInitSolve.jl, _solveCausalRows!),
before the simultaneous solve: the MSL Fluid tanks of MeasuringTemperature, reduced. The mass m
is the state, the level is fixed; from the media's start values (m = 0, d = 150, the vessel
pressure 5e6) the simultaneous Newton converged or not on rounding.
=#
@testset "Initialization: one-unknown rows" begin
  local CG = OMBackend.CodeGeneration
  local SA = CG.Symbolics.SparseArrays
  #= u = [m, V, d, level, p, q, w, x]: m differential (row 1), the others algebraic, the level
     pinned. d = rho(h), V = 2 level, m = V d, p = p0 + g level d read one unknown each once the
     ones before are known; q, w are a loop (two unknowns a row); x^2 = 4 is not affine in x. =#
  local rhs!(du, u, p, t) = begin
    du[1] = 0.0
    du[2] = u[2] - 2 * u[4]
    du[3] = u[3] - (1000 - 1.0e-4 * 3.0e5)
    du[4] = u[1] - u[2] * u[3]
    du[5] = u[5] - (1.0e5 + 9.81 * u[4] * u[3])
    du[6] = u[6]^3 + u[7] - u[5] / 1.0e5
    du[7] = u[7]^3 + u[6] - 1
    du[8] = u[8]^2 - 4
    nothing
  end
  local reads = [2 => [2, 4], 3 => [3], 4 => [1, 2, 3], 5 => [3, 4, 5], 6 => [5, 6, 7], 7 => [6, 7], 8 => [8]]
  local rows = Int[r for (r, cs) in reads for _ in cs]
  local cols = Int[c for (_, cs) in reads for c in cs]
  local sparsity = SA.sparse(rows, cols, zeros(length(rows)), 8, 8)
  local entry = [0.0, 2.0, 150.0, 0.975, 5.0e6, 0.5, 0.5, 1.0]
  local eqIdx = collect(2:8)
  local u = copy(entry)
  local solved = CG._solveCausalRows!(u, rhs!, Float64[], eqIdx, zeros(7), [1, 2, 3, 5, 6, 7, 8], sparsity; tol = 1.0e-10)
  @test solved == 4
  @test u[3] ≈ 970.0
  @test u[2] ≈ 1.95
  @test u[1] ≈ 1.95 * 970.0
  @test u[5] ≈ 1.0e5 + 9.81 * 0.975 * 970.0
  @test u[[4, 6, 7, 8]] == entry[[4, 6, 7, 8]]   # the pin, the loop and the nonlinear row left
  #= The whole initialization from the entry: consistent, the level kept. =#
  local mm = zeros(8, 8)
  mm[1, 1] = 1.0
  local solvedInit = CG._solveDAEInitialization!(copy(entry), rhs!, Float64[], mm; pinned = [4], sparsity = sparsity)
  local du = zeros(8)
  rhs!(du, solvedInit, Float64[], 0.0)
  @test maximum(abs, du[2:8]) < 1.0e-8
  @test solvedInit[4] == 0.975
  @test solvedInit[1] ≈ 1.95 * 970.0
end
