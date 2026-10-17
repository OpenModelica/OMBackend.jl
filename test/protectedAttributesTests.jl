#= The protection of a variable and its attributes (dropObservationOnlyVariables reads the
   protection): an enumeration's, a String's and a Clock's attributes once became an empty
   Real's when the protection was marked on them, their start, fixed, min and max lost. In
   OMJLIR and the simulation code the protection is a field of its own: a protected
   variable of each type keeps its attributes. =#
using Test
import OMBackend
import OMFrontend

@testset "Protected variables keep their attributes" begin
  local file = joinpath(@__DIR__, "Models", "ProtectedAttributes.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ProtectedAttributes", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  local model = OMBackend.omjlirModel(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local IR = OMBackend.OMJLIR
  local byName = Dict(String(v.name) => v for v in model.variables)
  local text = e -> e === nothing ? nothing : IR.toString(e)
  for (name, start) in (("e", "E.two"), ("k", "2"), ("b", "true"), ("m[2]", "4"))
    local v = byName[name]
    @test v.isProtected
    @test text(v.attributes.start) == start
    @test text(v.attributes.fixed) == "true"
  end
  @test text(byName["k"].attributes.min) == "0" && text(byName["k"].attributes.max) == "5"
  #= A public variable is not protected. =#
  @test !byName["x"].isProtected

  local SC = OMBackend.SimCode
  local ht = SC.fromOMJLIR(model).stringToSimVarHT
  for (name, ty) in (("e", SC.TYPE_ENUM), ("k", SC.TYPE_INTEGER), ("b", SC.TYPE_BOOL), ("m[2]", SC.TYPE_INTEGER))
    local v = last(ht[name])
    @test v.attributes.isProtected
    @test v.attributes.start !== nothing
    @test v.elementType isa ty
  end
  @test !last(ht["x"]).attributes.isProtected
end
