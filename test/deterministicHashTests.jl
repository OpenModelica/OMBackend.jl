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

#= OMJLIR's and SimCode's values hash by their type's name and their fields, an enumeration by its
   value: the same in every build. By the type itself (and Base's enumeration hash, which has its
   type's) they changed with every precompile, Julia deriving a type's hash from its module's build
   id: the when bodies' local function names (`#whenCall1_<hash>`, WhenBodyCalls) and the order of a
   Dict or Set of values (2026-10-30). A second process of the same build cannot tell; the hash's
   parts can. =#
@testset "IR and SimCode values hash by their contents" begin
  local IR = OMBackend.OMJLIR
  local SC = OMBackend.SimCode
  local h::UInt = UInt(7)
  @test hash(IR.INTEGER(1), h) == hash(1, hash(:INTEGER, h))
  @test hash(SC.ICONST(1), h) == hash(1, hash(:ICONST, h))
  local b::SC.BINARY = SC.BINARY(SC.ICONST(1), SC.OP_ADD, SC.ICONST(2))
  @test hash(b, h) == hash(SC.ICONST(2), hash(Integer(SC.OP_ADD), hash(SC.ICONST(1), hash(:BINARY, h))))
end
