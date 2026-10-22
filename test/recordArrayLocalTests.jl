#= A function's protected array of records bound by record-valued calls (Models/RecordArrayLocals.mo,
   as MSL ReferenceMoistAir's Ice09 Gibbs function: `final Complex[2] t = {Complex(...), Complex(...)}`):
   each field array from the calls' results. They were zero-filled, so `2 t[1] log(t[1])` was NaN, the
   ice entropy too, and ReferenceAir.Inverse_sh_TX's temperature inverse stopped at its bracket. =#
using Test
import OMBackend
import OMFrontend

@testset "Arrays of records bound by calls" begin
  local file = joinpath(@__DIR__, "Models", "RecordArrayLocals.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("RecordArrayLocals.M", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("RecordArrayLocals.M"); tspan = (0.0, 1.0))
  @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
  #= c[1].re + c[2].im = 1 + 4 =#
  @test OMBackend.getVariableValues(sol, "y")[1] ≈ 5.0
  @test OMBackend.getVariableValues(sol, "y")[end] ≈ 10.0
end
