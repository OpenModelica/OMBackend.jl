#= The wrapper of a generated Modelica function (createModelicaFunctionWrapper) has
   the function's arity. Its RuntimeGeneratedFunction binds the arguments unchecked:
   a call with fewer read past the argument tuple and crashed the process (MSL
   Water IF97, a record argument passed whole, 2026-09-30). It passes them as one
   tuple (_tupleArgFunction). =#
using Test
import OMBackend

@testset "Modelica function wrappers have the function's arity" begin
  local CG = OMBackend.CodeGeneration
  local SU = CG.SymbolicUtils
  local w = CG.ModelicaFunctionWrapper{2}(:omjlWrapperArityTest, args -> args[1] + 2 * args[2])
  @test w(1.0, 2.0) == 5.0
  @test_throws MethodError w(1.0)
  @test_throws MethodError w(1.0, 2.0, 3.0)
  @test nameof(w) === :omjlWrapperArityTest
  #= As an RGF: the operation of an opaque symbolic call keeps a Real scalar
     type when MTK rebuilds the term (`-(::SymReal, ::SymReal)` failed with Any). =#
  @test SU._promote_symtype(w, Any[1.0, 2.0]) === Real
  @test SU.promote_shape(w, SU.ShapeVecT(), SU.ShapeVecT()) == SU.ShapeVecT()
end

#= The element extractors of a record- or tuple-valued call (one term per field) call the
   function through `_memoizedCall`: equal plain arguments give the kept result (MSL Spice3
   Nor: each MOSFET's model ran once per field of its result); a new implementation (a new
   build) is called again; an array argument, whose contents could change, always calls. =#
@testset "Element extractors: equal arguments call a function once" begin
  local CG = OMBackend.CodeGeneration
  local calls = Ref(0)
  local impl = (a, b) -> (calls[] += 1; (a + b, a * b))
  @test CG._memoizedCall(:omjlMemoTest, impl, 1.0, 2.0) == (3.0, 2.0)
  @test CG._memoizedCall(:omjlMemoTest, impl, 1.0, 2.0) == (3.0, 2.0)
  @test calls[] == 1
  @test CG._memoizedCall(:omjlMemoTest, impl, 1.0, 3.0) == (4.0, 3.0) && calls[] == 2
  local other = (a, b) -> (calls[] += 1; (a - b, a / b))
  @test CG._memoizedCall(:omjlMemoTest, other, 1.0, 2.0) == (-1.0, 0.5) && calls[] == 3
  local v = [1.0, 2.0]
  local sumOf = x -> (calls[] += 1; sum(x))
  @test CG._memoizedCall(:omjlMemoArray, sumOf, v) == 3.0
  v[2] = 5.0
  @test CG._memoizedCall(:omjlMemoArray, sumOf, v) == 6.0 && calls[] == 5
  #= a flat extractor's key: its scalars, the array built from them =#
  @test CG._memoizedCallBy(:omjlMemoFlat, sumOf, (1.0, 2.0), [1.0, 2.0]) == 3.0
  @test CG._memoizedCallBy(:omjlMemoFlat, sumOf, (1.0, 2.0), [1.0, 2.0]) == 3.0 && calls[] == 6
end

#= An Integer input arrives as a Float64 (a discrete in the state vector); at an event the integrator
   located inside a step the state is interpolated there, a constant to 5.000000000000001 (MSL TimeTable's
   `last` at HBridge_DC_Drive's duty-cycle event, 4.1.0: "invalid index"). Within rounding it is the
   integer; another value stays as it is. =#
@testset "Integer inputs within rounding of an integer" begin
  local Alg = OMBackend.CodeGeneration.AlgorithmicCodeGeneration
  @test Alg._integerInput(5.0) === 5
  @test Alg._integerInput(5.000000000000001) === 5
  @test Alg._integerInput(2.9999999999999996) === 3
  @test Alg._integerInput(2.5) === 2.5
  @test Alg._integerInput(NaN) === NaN
  @test Alg._integerInput([1.0, 2.0000000000000004]) == [1, 2] && eltype(Alg._integerInput([1.0, 2.0000000000000004])) == Int
  @test Alg._integerInput([1.0, 2.5]) == [1.0, 2.5]
  @test Alg._integerInput(3) === 3
end
