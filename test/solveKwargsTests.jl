#= The keyword arguments a model's solve adds to the caller's (MTK_CodeGeneration.jl
   `defaultInitializeKwargs`, `defaultToleranceKwargs`). A caller's abstol per
   unknown reached NonlinearSolve's termination check through BrownFullBasicInit,
   which takes a scalar: a MethodError (2026-10-02). =#
using Test
import OMBackend

@testset "Solve keyword arguments with an abstol per unknown" begin
  local SB = OMBackend.ModelingToolkit.SciMLBase
  local CG = OMBackend.CodeGeneration
  local mm = [1.0 0.0; 0.0 0.0]
  local f = SB.ODEFunction((du, u, p, t) -> (du[1] = -u[1]; du[2] = u[2] - u[1]; nothing); mass_matrix = mm)
  local problem = SB.ODEProblem(f, [1.0, 1.0], (0.0, 1.0))
  @test CG.defaultInitializeKwargs(problem, (; abstol = [1.0e-6, 1.0e-8]), false).initializealg.abstol == 1.0e-8
  @test CG.defaultInitializeKwargs(problem, (; abstol = 1.0e-7), false).initializealg.abstol == 1.0e-7
  #= The caller's abstol per unknown stands; a problem without a system has no names to tell
     dummy derivatives by. =#
  @test CG.defaultToleranceKwargs(problem, (; abstol = [1.0e-6, 1.0e-8])) == (;)
  @test CG.defaultToleranceKwargs(problem, (;)) == (;)
end
