#=
The restart after an event (`_restartStep!` of the array code, `_restartStepSize!` of the scalarized code, both
through `_autoDtReset!`): the next step is proposed small, and the step that ended at the event keeps its length in
`integrator.dt`, so a callback after the event still interpolates that step (MSL BranchingDynamicPipes: an assert
check interpolated a step of 0.013 s with the restart's 2.2e-6 s, values 3000 steps out, a medium at 4125 K).
=#
@testset "The restart after an event keeps the step's interpolation" begin
  local CG = OMBackend.CodeGeneration
  local ODE = OMBackend.OrdinaryDiffEq
  local rodas = ODE.Rodas5P(autodiff = CG.ADTypes.AutoFiniteDiff())
  #= x' = 1 from 0 (x = t), an event at x = 0.5 inside a first step of 0.9: the affect restarts the step, then reads
     the interpolation in the middle of the step that ended at the event, and the step proposed next =#
  local ramp!(du, u, p, t) = (du[1] = 1.0; nothing)
  local rampDAE!(du, u, p, t) = (du[1] = 1.0; du[2] = u[2] - u[1]; nothing)
  local cases = ((CG.ArrayODEGen._restartStep!, ODE.ODEProblem(ramp!, [0.0], (0.0, 1.0))),
                 (integ -> CG._restartStepSize!(integ, Ref((NaN, 0))),
                  ODE.ODEProblem(ODE.ODEFunction(rampDAE!; mass_matrix = [1.0 0.0; 0.0 0.0]), [0.0, 0.0], (0.0, 1.0))))
  for (restart!, prob) in cases
    local seen = Tuple{Float64, Float64, Float64}[]
    local affect!(integ) = begin
      restart!(integ)
      local mid::Float64 = (integ.tprev + integ.t) / 2
      push!(seen, (mid, integ(mid)[1], abs(integ.dtpropose)))
      return nothing
    end
    local event = ODE.ContinuousCallback((u, t, integ) -> u[1] - 0.5, affect!)
    local sol = ODE.solve(prob, rodas; callback = event, dt = 0.9, reltol = 1e-8, abstol = 1e-10)
    @test sol.retcode == ReturnCode.Success
    @test length(seen) == 1
    @test seen[1][2] ≈ seen[1][1] atol = 1e-10        # before: the interpolation divided by the restart's step
    @test 0 < seen[1][3] <= 1e-6                      # the next step starts small (at most 1e-6 of the span)
    @test sol(0.75)[1] ≈ 0.75 atol = 1e-8
  end
end
