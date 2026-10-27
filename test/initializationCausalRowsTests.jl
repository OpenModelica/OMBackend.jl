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

#=
The initialization's blocks in causal order (_solveCausalBlocks!): a slider-crank as in MSL EngineV6's cylinders,
reduced. The crank angle is pinned; the piston position s and the rod angle b are a block of two nonlinear rows (the
loop); the gas force f is a quartic of the piston position, enormous away from the solution (the zero guesses put it
at x = 2.9); the acceleration a follows from f. Solved together from the zero guesses, the phases stalled on EngineV6.
=#
@testset "Initialization: blocks in causal order" begin
  local CG = OMBackend.CodeGeneration
  local SA = CG.Symbolics.SparseArrays
  local (r, l, L, m) = (0.1, 0.3, 0.2, 2.0)
  local gas = (x::Real) -> 1.0e6 * (x^4 - 2.0 * x^3 + 1.5 * x^2 - 0.5 * x + 0.1)
  #= u = [theta, s, b, f, a]: theta pinned =#
  local rhs!(du, u, p, t) = begin
    du[1] = 0.0
    du[2] = r * cos(u[1]) + l * cos(u[3]) - u[2]
    du[3] = r * sin(u[1]) - l * sin(u[3]) + 0.05 * u[2]
    du[4] = u[4] - gas(1 - (u[2] - 0.25) / L)
    du[5] = m * u[5] - u[4]
    nothing
  end
  local reads::Vector{Pair{Int, Vector{Int}}} = [2 => [1, 2, 3], 3 => [1, 2, 3], 4 => [2, 4], 5 => [4, 5]]
  local rows::Vector{Int} = Int[k for (k, cs) in reads for _ in cs]
  local cols::Vector{Int} = Int[c for (_, cs) in reads for c in cs]
  local sparsity::SA.SparseMatrixCSC{Float64, Int} = SA.sparse(rows, cols, zeros(length(rows)), 5, 5)
  local entry::Vector{Float64} = [0.5, 0.0, 0.0, 0.0, 0.0]
  local u::Vector{Float64} = copy(entry)
  @test CG._solveCausalBlocks!(u, rhs!, Float64[], collect(2:5), zeros(4), nothing, [2, 3, 4, 5], sparsity;
                               tol = 1.0e-10, maxiter = 50) == 4
  local du::Vector{Float64} = zeros(5)
  rhs!(du, u, Float64[], 0.0)
  @test maximum(abs, du[2:5]) < 1.0e-6
  @test u[1] == 0.5
  @test u[5] ≈ gas(1 - (u[2] - 0.25) / L) / m
  #= the whole initialization from the zero guesses: consistent, the crank angle kept =#
  local mm::Matrix{Float64} = zeros(5, 5)
  mm[1, 1] = 1.0
  local solvedInit::Vector{Float64} = CG._solveDAEInitialization!(copy(entry), rhs!, Float64[], mm; pinned = [1], sparsity = sparsity)
  rhs!(du, solvedInit, Float64[], 0.0)
  @test maximum(abs, du[2:5]) < 1.0e-6
  @test solvedInit[1] == 0.5
  @test solvedInit[2] ≈ u[2] rtol = 1.0e-8
end

#= The blocks' incidence probes (initialization equations: by finite differences, at a point near the entry) reach
   y < 0 in sqrt(y) from its start 0: the DomainError ended the initialization, which the phases after solve. =#
@testset "Initialization: an exception in the blocks' pass" begin
  local CG = OMBackend.CodeGeneration
  local SA = CG.Symbolics.SparseArrays
  #= u = [x, y]: der(x) = -x, 0 = sqrt(y) - x, initially x = 2 =#
  local rhs!(du, u, p, t) = (du[1] = -u[1]; du[2] = sqrt(u[2]) - u[1]; nothing)
  local mm::Matrix{Float64} = [1.0 0.0; 0.0 0.0]
  local sparsity::SA.SparseMatrixCSC{Float64, Int} = SA.sparse([1, 2, 2], [1, 1, 2], zeros(3), 2, 2)
  local u::Vector{Float64} = CG._solveDAEInitialization!([0.0, 0.0], rhs!, Float64[], mm; sparsity = sparsity,
                                                         extra_residuals = (du, u) -> [u[1] - 2.0])
  @test u[1] ≈ 2.0 atol = 1.0e-8
  @test u[2] ≈ 4.0 atol = 1.0e-8
end
