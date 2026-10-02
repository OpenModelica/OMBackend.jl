#= A generated Modelica function, and a symbolic term calling it, hash the same in
   every process (mtkExternals.jl `_RGF_TAG`). By default a RuntimeGeneratedFunction
   hashed by the address of its body: the terms calling it, and with them MTK's
   alias choices and the generated code's summation order, changed from one process
   to the next (MSL DifferenceAmplifier with QNDF at 1e-2: Success, Unstable or
   InitialFailure by process, 2026-10-02). Within one process the same code gives
   the same RGF object, so only a second process can tell. =#
using Test
import OMBackend

@testset "Generated functions hash the same in every process" begin
  local script = """
  import OMBackend
  const CG = OMBackend.CodeGeneration
  f = CG.RuntimeGeneratedFunctions.RuntimeGeneratedFunction(CG, CG, :((a, b) -> a + 2b))
  w = CG.ModelicaFunctionWrapper{2}(:omjlHashTest, f)
  x = CG.Symbolics.variable(:x)
  term = CG.Symbolics.unwrap(CG.makeSymbolicTerm(w, Any[CG.Symbolics.unwrap(x), 1.0]))
  lookup = CG.ConstTableLookupFn([1.0, 2.0])
  print(join((hash(f), hash(w), hash(term), hash(lookup)), " "))
  """
  local runOnce() = read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`, String)
  local first, second = runOnce(), runOnce()
  @test length(split(first)) == 4
  @test first == second
end
