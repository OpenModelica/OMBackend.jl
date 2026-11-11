#= The initial value of a state without a start value is resolved from an equation that is linear in it, when the
   equation's other variables have known values. An equation that keeps a variable gives no value, and is not
   simplified further: with SymbolicUtils 4.55 the simplification of this one at x = 1 (a difference of quotients of
   sums of exponentials) threw a BoundsError in the polynomial gcd, and the build of MSL 4.1.0
   Magnetic.FluxTubes.Examples.Hysteresis.HysteresisModelComparison failed with it (coverage of 2026-11-10). The
   error depends on the order of the variables: the names are the model's. =#
using Test
import OMBackend

@testset "An alias equation that keeps a variable gives no initial value" begin
  local CG::Module = OMBackend.CodeGeneration
  local S::Module = CG.Symbolics
  local t = only(OMBackend.CodeGeneration.Symbolics.@variables t)
  local variable = (name::String) -> only(OMBackend.CodeGeneration.Symbolics.@variables $(Symbol(name))(t))
  local dE = variable("\$DER_preisachEverett_E")
  local dAlpha = variable("\$DER_preisachEverett_alpha")
  local flag = variable("ifEq_tmp1")
  local (H1, H2, H3, H4) = (variable("preisachEverett_H1"), variable("preisachEverett_H2"),
                            variable("preisachEverett_H3"), variable("preisachEverett_H4"))
  #= the Everett function of the model's material and its derivative times s =#
  local f = h -> 0.48597552 * (1.0 + 0.6366197723675814 * atan(0.039964 * h)) +
                 0.96204896 / (1.0 + 0.5 * (exp(-0.000781 * h) + exp(-0.18807 * h)))
  local g = (h, s) -> (0.01236412725824761 * s) / (1.0 + 0.0015971212959999999 * h^2.0) +
                      (-0.48102448 * (-0.000781 * exp(-0.000781 * h) * s - 0.18807 * exp(-0.18807 * h) * s)) /
                      ((1.0 + 0.5 * (exp(-0.000781 * h) + exp(-0.18807 * h)))^2.0)
  local equation = dE + f(H3) * g(H4, flag) + f(H4) * g(H3, -dAlpha) - f(H1) * g(H2, dAlpha) - f(H2) * g(H1, -flag)
  local known::Dict = Dict(S.unwrap(flag) => 0.0)
  @test CG._resolveAliasFromExpr(S.unwrap(dAlpha), S.unwrap(equation), 7, known) === nothing

  #= one that keeps none gives the value: 2 x + 3 a - 12 = 0 at a = 2 is x = 3, with a folded branch too =#
  local a = variable("a")
  local x = variable("x")
  @test CG._resolveAliasFromExpr(S.unwrap(x), S.unwrap(2 * x + 3 * a - 12), 2, Dict(S.unwrap(a) => 2.0)) ≈ 3.0
  @test CG._resolveAliasFromExpr(S.unwrap(x), S.unwrap(ifelse(flag > 0.5, H1, 2 * x) - 4 * a), 4,
                                 Dict(S.unwrap(a) => 2.0, S.unwrap(flag) => 0.0)) ≈ 4.0
end
