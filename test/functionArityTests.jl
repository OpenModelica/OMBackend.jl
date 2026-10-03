#= The wrapper of a generated Modelica function (createModelicaFunctionWrapper) has
   the function's arity. Its RuntimeGeneratedFunction binds the arguments unchecked:
   a call with fewer read past the argument tuple and crashed the process (MSL
   Water IF97, a record argument passed whole, 2026-09-30). =#
using Test
import OMBackend

@testset "Modelica function wrappers have the function's arity" begin
  local CG = OMBackend.CodeGeneration
  local SU = CG.SymbolicUtils
  local w = CG.ModelicaFunctionWrapper{2}(:omjlWrapperArityTest, (a, b) -> a + 2b)
  @test w(1.0, 2.0) == 5.0
  @test_throws MethodError w(1.0)
  @test_throws MethodError w(1.0, 2.0, 3.0)
  @test nameof(w) === :omjlWrapperArityTest
  #= As an RGF: the operation of an opaque symbolic call keeps a Real scalar
     type when MTK rebuilds the term (`-(::SymReal, ::SymReal)` failed with Any). =#
  @test SU._promote_symtype(w, Any[1.0, 2.0]) === Real
  @test SU.promote_shape(w, SU.ShapeVecT(), SU.ShapeVecT()) == SU.ShapeVecT()
end
