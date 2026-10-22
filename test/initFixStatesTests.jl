#= An underdetermined initialization completed as OpenModelica and Dymola complete it (DAEInitSolve, phase 1b,
   OMBACKEND_INIT_FIX_STATES): the unfixed states the equations leave free, chosen by a matching, at their start
   values, the remainder solved. The least-squares phase 2 moved every state (x + y = 5 from x = 1, y = 2 ended at
   x = 2, y = 3). Models/InitFixStates.mo; the values are OpenModelica 1.27.1's ("Assuming fixed start value for
   the following 1 variables: x"). =#
using Test
import OMBackend
import OMFrontend

@testset "Underdetermined initialization: states fixed at their start values" begin
  local CG = OMBackend.CodeGeneration
  #= the matching: the artificial rows take the first states; a row needing a state moves them on =#
  @test CG._initStatesToFix([[1, 2]], 2, [1, 2]) == [1]
  @test CG._initStatesToFix([[1]], 2, [1, 2]) == [2]
  @test CG._initStatesToFix([[1, 3]], 3, [1, 2]) == [1, 2]
  @test CG._initStatesToFix([[1, 2]], 3, [1, 2]) === nothing       # z read by no row: never fixed
  @test CG._initStatesToFix([[1], [2]], 2, [1, 2]) == Int[]
  @test CG._maximumMatching([[1, 2], [1], [2, 3]], 3) == [2, 1, 3]
  local file = joinpath(@__DIR__, "Models", "InitFixStates.mo")
  local scode = OMFrontend.translateToSCode(OMFrontend.parseFile(file))
  for (model, expected) in (("PartlyFixed", ("x" => 1.0, "y" => 4.0)), ("WithAlgebraic", ("x" => 1.0, "y" => 1.5, "z" => 4.0)))
    local (fm, cache) = OMFrontend.instantiateSCodeToFM("InitFixStates." * model, scode)
    OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("InitFixStates." * model); tspan = (0.0, 1.0))
    @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
    for (name, value) in expected
      @test OMBackend.getVariableValues(sol, name)[1] ≈ value atol = 1e-8
    end
  end
end
