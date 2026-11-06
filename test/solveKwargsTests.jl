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
  #= the BrownFullBasicInit the start re-solve (StartReinit) runs =#
  @test CG.defaultInitializeKwargs(problem, (; abstol = [1.0e-6, 1.0e-8]), false).initializealg.alg.abstol == 1.0e-8
  @test CG.defaultInitializeKwargs(problem, (; abstol = 1.0e-7), false).initializealg.alg.abstol == 1.0e-7
  #= The caller's abstol per unknown stands; a problem without a system has no names to tell
     dummy derivatives by. =#
  @test CG.defaultToleranceKwargs(problem, (; abstol = [1.0e-6, 1.0e-8])) == (;)
  @test CG.defaultToleranceKwargs(problem, (;)) == (;)
end

#= Derivative unknowns are left out of the error control by their kind (SimCodeKind metadata): a state's derivative
   kept as an unknown (ALGEBRAIC_DER) as well as a dummy derivative (DUMMY_DER) (MSL IMC_YDarc: their noise after an
   arc went out stalled the solve). The state and the other algebraic unknowns keep the solve's abstol. =#
@testset "Derivative unknowns out of the error control" begin
  local SB::Module = OMBackend.ModelingToolkit.SciMLBase
  local CG::Module = OMBackend.CodeGeneration
  local IR::Module = OMBackend.OMJLIR
  local t = OMBackend.ModelingToolkit.t_nounits
  local kind(category::Type, irKind) = (v -> OMBackend.Symbolics.wrap(OMBackend.SymbolicUtils.setmetadata(
    OMBackend.Symbolics.unwrap(v), CG.SimCodeKind, CG.SimCodeKind(Symbol(CG.variableName(v)), category, irKind, :model))))
  local (x, dx, y) = OMBackend.ModelingToolkit.@variables x(t) dx(t) y(t)
  local unknowns::Vector{OMBackend.Symbolics.Num} =
    [kind(OMBackend.SimCode.STATE, IR.STATE(1, :dx, true))(x),
     kind(OMBackend.SimCode.ALG_VARIABLE, IR.ALGEBRAIC_DER(:x))(dx),
     kind(OMBackend.SimCode.ALG_VARIABLE, IR.ALGEBRAIC())(y)]
  local sys = OMBackend.ModelingToolkit.System(OMBackend.ModelingToolkit.Equation[], t, unknowns, []; name = :s)
  local f = SB.ODEFunction((du, u, p, t) -> nothing; mass_matrix = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], sys = sys)
  @test CG.defaultToleranceKwargs(SB.ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1.0)), (; abstol = 1.0e-7)) ==
        (; abstol = [1.0e-7, Inf, 1.0e-7])
end
