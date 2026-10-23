#= A variable a function's tuple output defines, which the simulation code finds constant: index reduction plans
   with its value (Evaluation.functionDefinedValues; `withKnownValues` took only `v = f(...)`), and the derivatives'
   solve does not divide by it (SolveDerivatives' derivativeEdges, the plan's `known`). MSL Spice3's
   FourBitBinaryAdder: its diodes without a junction capacitance, `(cc, m_dCap) = diodeNoBypassCode(...)` 0, had
   `icap = m_dCap * der(v)` solved for der(v), a division by 0 (a DivideError of the symbolic arithmetic).
   Models/TupleKnownValues.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "Known values of a tuple output" begin
  local file = joinpath(@__DIR__, "Models", "TupleKnownValues.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("TupleKnownValues.Clamp", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  local lowered = OMBackend.LowerFlatModel.lowerFlatModel(fm; functions = OMFrontend.cacheToFunctionList(cache))
  local model = OMBackend.Passes.runPasses!(lowered; through = "unrollConstantReductions")
  local partition = model.continuous[1]
  local known = OMBackend.knownValues(model)
  @test get(known, :c, nothing) == OMBackend.OMJLIR.REAL(0.0)
  local E = OMBackend.Passes.Evaluation
  local values = E.functionDefinedValues(partition, known)
  @test haskey(values, :c)
  #= the capacitor current reads no derivative with c's value: `icap = 0.0` =#
  local folded = E.withKnownValues(model, partition; known)
  local icap = only(filter(eq -> eq isa OMBackend.OMJLIR.SCALAR_EQUATION && eq.lhs isa OMBackend.OMJLIR.CREF &&
                                 eq.lhs.cref.name === :icap, folded.equations))
  @test icap.rhs == OMBackend.OMJLIR.REAL(0.0)
end
