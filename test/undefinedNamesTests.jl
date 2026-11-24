#= Author: John Tinnerholm
   Code paths that read a global their module does not define (JETLS lowering/undef-global-var, 2026-11-24). Each
   threw an UndefVarError, which the error policy counts as a bug and rethrows, where the code meant another
   result. =#
using Test
import LinearAlgebra
import OMBackend
import OMFrontend

@testset "Error paths read names their module defines" begin
  #= Runtime._isDAETransferTarget (a structural mode switch) tested `mm isa UniformScaling` without LinearAlgebra in
     scope: every new mode with a mass matrix threw. =#
  @testset "a new mode's mass matrix: singular is a DAE, I is none" begin
    local SB::Module = OMBackend.ModelingToolkit.SciMLBase
    local t = OMBackend.ModelingToolkit.t_nounits
    local (x, y) = OMBackend.ModelingToolkit.@variables x(t) y(t)
    local sys = OMBackend.ModelingToolkit.System(OMBackend.ModelingToolkit.Equation[], t, [x, y], []; name = :s)
    local problem(mm::AbstractMatrix) =
      SB.ODEProblem(SB.ODEFunction((du, u, p, t) -> nothing; mass_matrix = mm, sys = sys), [1.0, 0.0], (0.0, 1.0))
    @test !OMBackend.Runtime._isDAETransferTarget(problem(LinearAlgebra.I(2)))
    @test !OMBackend.Runtime._isDAETransferTarget(
      SB.ODEProblem(SB.ODEFunction((du, u, p, t) -> nothing; sys = sys), [1.0, 0.0], (0.0, 1.0)))
    @test OMBackend.Runtime._isDAETransferTarget(problem([1.0 0.0; 0.0 0.0]))
  end

  #= AlgorithmicCodeGeneration refused a FORTRAN 77 argument it cannot pass through `OMBackend.unsupported`, a name
     that module does not have: an UndefVarError instead of UnsupportedLowering. =#
  @testset "an external FORTRAN 77 argument it cannot pass is unsupported" begin
    local program::OMFrontend.OpenModelicaParser.Absyn.Program = OMFrontend.OpenModelicaParser.Modelica"""
      package F77
        function characterArray
          input String s[2];
          output Real y;
          external "FORTRAN 77" y = dummy(s);
        end characterArray;
        function sizeArgument
          input Real a[:];
          output Real y;
          external "FORTRAN 77" y = dummy(a, size(a, 1));
        end sizeArgument;
        model CharacterArray
          Real y = characterArray({"a", "b"});
        end CharacterArray;
        model SizeArgument
          Real y = sizeArgument({1.0, 2.0});
        end SizeArgument;
      end F77;
      """
    for (model::String, what::String) in (("F77.CharacterArray", "a FORTRAN 77 character array"),
                                          ("F77.SizeArgument", "a FORTRAN 77 argument"))
      local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, OMFrontend.translateToSCode(program); scalarize = true)
      local err::Union{Nothing, Exception} = try
        OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarize = true)
        nothing
      catch e
        e
      end
      @test err isa OMBackend.UnsupportedLowering && err.what == what
    end
  end
end
