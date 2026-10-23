#= A linear constraint among the states that no equation shows (IndexReduction.exposeLinearConstraints!, the switch
   OMBACKEND_LINEAR_CONSTRAINTS): three chokes feed a floating resistor bridge; its node balances add up to
   i1 + i2 + i3 = 0, while each reads a bridge current. The structure was regular, the system singular: all three
   currents stayed states (MSL IMC_InverterDrive: Unstable at its first switching; omc keeps two). With the
   constraint written in, Pantelides differentiates it and two currents are states; omc 1.27.1's values at
   tolerance 1e-8. And a constant table lookup has the derivative 0 in the symbolic Jacobian (without a rule the
   drive's PWM table sent the Jacobian and every event re-solve to finite differences). Models/HiddenConstraints.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "Hidden linear constraints" begin
  local file = joinpath(@__DIR__, "Models", "HiddenConstraints.mo")
  local name = "HiddenConstraints.FloatingBridge"
  local IR = OMBackend.OMJLIR
  local IRX = OMBackend.Passes.IndexReduction
  local flat = () -> OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  local passed = () -> begin
    local (fm, cache) = flat()
    local lowered = OMBackend.LowerFlatModel.lowerFlatModel(fm; functions = OMFrontend.cacheToFunctionList(cache))
    OMBackend.Passes.runPasses!(lowered; through = first(last(OMBackend.Passes.PIPELINE)))
  end
  local states = m -> Set{Symbol}(v.name for v in m.variables if v.kind isa IR.STATE)
  #= the sum of the balances, over the chokes' currents only (the masses' x1 = x2 the structure shows) =#
  local model = passed()
  local found = IRX.linearConstraints(model, model.continuous[1])
  @test length(found) == 1
  local reads = Symbol[]
  IR.foreachExp(x -> (x isa IR.CREF && push!(reads, x.cref.name); nothing), only(found)[2].lhs)
  @test Set(reads) == Set([:i1, :i2, :i3])
  IRX.reduceIndex!(model)
  @test length(intersect(states(model), Set([:i1, :i2, :i3]))) == 2
  withenv("OMBACKEND_LINEAR_CONSTRAINTS" => "false") do
    local m = passed()
    IRX.reduceIndex!(m)
    @test issubset(Set([:i1, :i2, :i3]), states(m))
  end
  local (fm, cache) = flat()
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName(name); tspan = (0.0, 0.02))
  @test string(sol.retcode) == "Success"
  #= omc: i1, i3, iload at 0.005, 0.01, 0.015, 0.02 =#
  local expected = [(112.3695119406272, -0.1851103830138072, 2.958310250957386),
                    (64.65869275289573, -165.8268913568252, 2.792510936634264),
                    (-107.4559523628166, -13.45954343045258, -2.739238592242196),
                    (-64.26410885623673, 164.6996901739286, -2.774711352624182)]
  for (t, (i1, i3, iload)) in zip((0.005, 0.01, 0.015, 0.02), expected)
    @test sol(t; idxs = :i1) ≈ i1 rtol = 1e-3
    @test sol(t; idxs = :i3) ≈ i3 atol = 0.05
    @test sol(t; idxs = :iload) ≈ iload rtol = 1e-3
    @test abs(sol(t; idxs = :i1) + sol(t; idxs = :i2) + sol(t; idxs = :i3)) < 1e-8
  end
end

@testset "Derivative of a constant table lookup" begin
  local CG = OMBackend.CodeGeneration
  local S = CG.Symbolics
  local (x, k) = (S.variable(:x), S.variable(:k))
  local lookup = CG.constTableLookup([1.0 2.0; 3.0 4.0], S.unwrap(k), 1)
  local J = CG._dagSparseJacobian(Any[S.unwrap(lookup * x)], Any[S.unwrap(x), S.unwrap(k)])
  @test isequal(J[1, 1], lookup)
  @test S.SymbolicUtils.unwrap_const(S.unwrap(J[1, 2])) == 0
end
