#= A failed build freed its external objects (forgetBuild) while its module stayed evaluated: the module's own simulate
   read the freed table ("... was freed with the build it belonged to") instead of failing on the build's error (MSL
   FullRobot with an observed filter). The module of a failed build is evaluated again at its simulate
   (Models/FailedBuild.mo). =#
using Test
import OMBackend
import OMFrontend

@testset "A failed build: its simulate fails on the build's error" begin
  local file::String = joinpath(@__DIR__, "Models", "FailedBuild.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("FailedBuild.TableAtFailedBuild",
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local err::Union{Nothing, Exception} =
    try
      OMBackend.simulateModel(OMBackend.canonicalName("FailedBuild.TableAtFailedBuild"); tspan = (0.0, 1.0))
      nothing
    catch e
      e
    end
  @test err isa Exception
  local msg::String = err === nothing ? "" : sprint(showerror, err)
  @test occursin("DAE init", msg)
  @test !occursin("freed", msg)
end
