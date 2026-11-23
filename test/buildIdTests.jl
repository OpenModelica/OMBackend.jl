#= Author: John Tinnerholm
   buildId: the identity of a model's current build. The same code keeps it, other equations under the same name change
   it, on either route (scalarized, arrays kept); a model not translated is an error. =#
using Test
import OMBackend
import OMFrontend

"""
The model `name` of `program` translated (`scalarize` as the frontend's), with `k` tunable when scalarized (a tunable
set takes the ModelingToolkit route; the arrays-kept route has every parameter tunable); its build id.
"""
function _buildIdOf(program, name::String; scalarize::Bool)::UInt64
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(name, OMFrontend.translateToSCode(program); scalarize)
  local functionList = OMFrontend.cacheToFunctionList(cache)
  if scalarize
    OMBackend.withTunableParameters(["k"]) do
      OMBackend.translate(fm; functionList, scalarize)
    end
  else
    OMBackend.translate(fm; functionList, scalarize)
  end
  return OMBackend.buildId(name)
end

@testset "buildId" begin
  local decay = OMFrontend.OpenModelicaParser.Modelica"""
    model BuildIdModel
      parameter Real k[2] = {1, 2};
      Real x[2](each start = 1, each fixed = true);
    equation
      der(x) = -k .* x;
    end BuildIdModel;"""
  local faster = OMFrontend.OpenModelicaParser.Modelica"""
    model BuildIdModel
      parameter Real k[2] = {1, 2};
      Real x[2](each start = 1, each fixed = true);
    equation
      der(x) = -2 * k .* x;
    end BuildIdModel;"""
  local xEnd(name::String)::Float64 =
    OMBackend.variableValues(OMBackend.simulate(name; reltol = 1e-10, abstol = 1e-12), "x[1]")[end]
  try
    for scalarize in (true, false)
      local id::UInt64 = _buildIdOf(decay, "BuildIdModel"; scalarize)
      @test (OMBackend.canonicalName("BuildIdModel") in OMBackend.ARRAY_PRESERVE_DE_MODELS) == !scalarize
      @test _buildIdOf(decay, "BuildIdModel"; scalarize) == id
      @test isapprox(xEnd("BuildIdModel"), exp(-1.0); rtol = 1e-7)
      local other::UInt64 = _buildIdOf(faster, "BuildIdModel"; scalarize)
      @test other != id
      @test isapprox(xEnd("BuildIdModel"), exp(-2.0); rtol = 1e-7)
      OMBackend.forgetModel("BuildIdModel")
      @test_throws ArgumentError OMBackend.buildId("BuildIdModel")
    end
  finally
    OMBackend.forgetModel("BuildIdModel")
  end
end
