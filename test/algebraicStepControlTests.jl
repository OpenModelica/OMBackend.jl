#=
Step control for the algebraic unknowns of a mass-matrix DAE solved with a
Rosenbrock method (CodeGeneration/algebraicStepControl.jl). Rodas5P's error
estimate is zero for algebraic unknowns: y = 100 sin t is solved in 6 steps
with a dense output wrong by 200. The callback keeps the dense output within
the tolerance and projects the step ends onto the algebraic equations.
=#
@testset "Algebraic step control" begin
  local CG = OMBackend.CodeGeneration
  local ODE = OMBackend.OrdinaryDiffEq
  local rodas = ODE.Rodas5P(autodiff = CG.ADTypes.AutoFiniteDiff())
  local mm = [1.0 0.0; 0.0 0.0]

  #= x' = 0, 0 = y - 100 sin t: only the algebraic unknown moves. =#
  local driven!(du, u, p, t) = (du[1] = 0.0; du[2] = u[2] - 100 * sin(t); nothing)
  local prob = ODE.ODEProblem(ODE.ODEFunction(driven!; mass_matrix = mm), [0.0, 0.0], (0.0, 10.0))
  local denseError(s) = maximum(abs(s(t)[2] - 100 * sin(t)) for t in range(0.0, 10.0; length = 2001))
  @test CG._algebraicRows(prob.f) == [2]
  local plain = ODE.solve(prob, rodas; reltol = 1e-6, abstol = 1e-7)
  @test denseError(plain) > 1.0                        # the problem the callback addresses
  local controlled = ODE.solve(prob, rodas; reltol = 1e-6, abstol = 1e-7,
                               callback = CG.withAlgebraicStepControl(nothing, prob))
  @test controlled.retcode == ReturnCode.Success
  @test denseError(controlled) < 1e-3

  #= x' = 1, 0 = y^2 - (1 + x): the step ends satisfy the nonlinear equation. =#
  local curved!(du, u, p, t) = (du[1] = 1.0; du[2] = u[2]^2 - (1 + u[1]); nothing)
  local prob2 = ODE.ODEProblem(ODE.ODEFunction(curved!; mass_matrix = mm), [0.0, 1.0], (0.0, 10.0))
  local residual(s) = maximum(abs(u[2]^2 - (1 + u[1])) for u in s.u)
  local projected = ODE.solve(prob2, rodas; reltol = 1e-3, abstol = 1e-4,
                              callback = CG.withAlgebraicStepControl(nothing, prob2))
  @test projected.retcode == ReturnCode.Success
  @test residual(projected) < 1e-7                   # the step ends to 1e-6 of the tolerance

  #= An ODE has no algebraic rows: nothing is added. Other methods (a BDF
     controls the algebraic error itself, and its history would not match a
     changed state) and fixed steps do not use it. It is no event callback. =#
  local ode = ODE.ODEProblem((du, u, p, t) -> (du[1] = -u[1]; nothing), [1.0], (0.0, 1.0))
  @test CG.withAlgebraicStepControl(nothing, ode) === nothing
  local cbs = CG.withAlgebraicStepControl(nothing, prob)
  @test CG.isStepControl(only(cbs.discrete_callbacks))
  local control = only(cbs.discrete_callbacks).condition
  ODE.init(prob, OMBackend.OrdinaryDiffEqBDF.FBDF(autodiff = CG.ADTypes.AutoFiniteDiff()); callback = cbs)
  @test !control.active
  ODE.init(prob, rodas; callback = cbs, adaptive = false, dt = 0.5)
  @test !control.active
  ODE.init(prob, rodas; callback = cbs)
  @test control.active
end
