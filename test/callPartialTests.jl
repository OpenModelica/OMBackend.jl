#= The terms of a function's element extractors and flat calls (CodeGeneration.CALL_RGFS) have partials by
   central differences in the symbolic Jacobian (CallPartial, OMBACKEND_CALL_PARTIALS): without a rule the
   whole Jacobian was finite differences of the right-hand side (MSL RoomCO2WithControls: the step of a
   pressure of 1e5 crossed its ducts' zero-flow regularization, 3000 steps to 0.0037 s). Only where the
   numeric partials are on (the Jacobian and the time derivative); the partials nest. =#
using Test
import OMBackend

@testset "Partials of call terms" begin
  local CG = OMBackend.CodeGeneration
  local S = CG.Symbolics
  CG.MODELICA_FUNCTION_IMPLS[:omjlPartialPlain] = (a, b) -> (a^2 * b, sin(a) + b)
  CG.MODELICA_FUNCTION_IMPLS[:omjlPartialFlat] = (x, v) -> x * v[1] + v[2]^3
  local field2 = CG.getOrCreateElemFunc(:omjlPartialPlain, (2,), 2)
  local flat = CG._getOrCreateFlatScalarFunc(:omjlPartialFlat, 3, ((), (2,)))
  @test field2 in CG.CALL_RGFS && flat in CG.CALL_RGFS
  local (a, b, c) = (S.variable(:a), S.variable(:b), S.variable(:c))
  local term(f, args...) = S.wrap(CG.makeSymbolicTerm(f, Any[S.unwrap(x) for x in args]))
  #= a partial term evaluated: its CallPartial at numeric arguments =#
  local at(ex, vals...) = Base.invokelatest(S.operation(S.unwrap(ex)), vals...)
  #= no rule without the numeric partials: the derivative stays a Differential term =#
  @test S.unwrap(S.expand_derivatives(S.Differential(a)(term(field2, a, b)))) isa S.SymbolicUtils.BasicSymbolic
  @test !(S.operation(S.unwrap(S.expand_derivatives(S.Differential(a)(term(field2, a, b))))) isa CG.CallPartial)
  CG.withNumericPartials() do
    local d = S.expand_derivatives(S.Differential(a)(term(field2, a, b)))
    @test S.operation(S.unwrap(d)) isa CG.CallPartial
    @test at(d, 0.3, 2.0) ≈ cos(0.3) rtol = 1e-7
    local dd = S.expand_derivatives(S.Differential(a)(d))
    @test S.operation(S.unwrap(dd)) isa CG.CallPartial{2, 2}
    @test at(dd, 0.3, 2.0) ≈ -sin(0.3) rtol = 1e-4
    #= the flat call: d/dc of x v1 + v2^3 at v2 = c =#
    local df = S.expand_derivatives(S.Differential(c)(term(flat, a, b, c)))
    @test at(df, 2.0, 5.0, 1.5) ≈ 3 * 1.5^2 rtol = 1e-7
    withenv("OMBACKEND_CALL_PARTIALS" => "false") do
      @test !(S.operation(S.unwrap(S.expand_derivatives(S.Differential(a)(term(field2, a, b))))) isa CG.CallPartial)
    end
  end
end
