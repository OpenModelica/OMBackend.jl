#= Unit tests for DiscreteDummyDemotion.planDemotions.

   The demotion pass removes the synthetic `der(d) ~ 0` dummy of a discrete that
   a residual already defines. The heuristic excess-fill must never demote a
   discrete that is only an *input* to an equation (no residual defines it),
   because removing its dummy strands it with no equation (under-determination)
   — the failure mode behind the FlipFlop ExtraVariablesSystemException. =#
using Test

include(joinpath(@__DIR__, "backendTestMocks.jl"))
using .BackendTestMocks

const _CG = OMBackend.CodeGeneration

@testset "DiscreteDummyDemotion.planDemotions" begin
  #= dDef is defined by a residual (`0 ~ dDef - 5`); dInput only appears as an
     input (`0 ~ algY - foo(dInput)`). With excess > 0 the heuristic fill runs;
     it may demote dDef but must leave dInput alone. =#
  local discreteVariables = ["dDef", "dInput"]
  local equations = Expr[ :(0 ~ dDef - 5), :(0 ~ algY - foo(dInput)) ]
  local ifEqComponents = _CG.IfEquationComponent[]
  local sc = mockSimCode()

  local plan = _CG.planDemotions(sc, equations, ifEqComponents,
                                 discreteVariables, 0, 0, 0)

  @testset "input-only discrete is not demoted" begin
    @test !("dInput" in plan.toDemote)
  end

  @testset "residual-defined discrete may be demoted" begin
    @test "dDef" in plan.toDemote
  end

  @testset "_residualDefinedDiscretes detects only defined discretes" begin
    local defined = _CG._residualDefinedDiscretes(equations, Set(discreteVariables))
    @test "dDef" in defined
    @test !("dInput" in defined)
  end
end
