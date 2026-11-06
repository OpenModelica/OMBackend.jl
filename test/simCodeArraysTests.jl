#= The simulation code of a model with its arrays kept (SimCode.lowerArraySystem): the array route's model (index
   reduced in array form, its derivatives solved in array form, Passes.arrayCodeForm!) lowered one to one, its arrays
   whole, its for-equations kept, the values the array code takes evaluated per element. Small models of
   Models/ArrayPath.mo; no simulation. =#
using Test
import OMBackend
import OMFrontend

@testset "SimCode of a model with its arrays kept" begin
  local SC = OMBackend.SimCode
  local file::String = joinpath(@__DIR__, "Models", "ArrayPath.mo")
  local scode = OMFrontend.translateToSCode(OMFrontend.parseFile(file))
  local lowered = (model::String) -> begin
    local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, scode; scalarize = false)
    OMBackend.clearCaches!()
    local reduced = OMBackend.reducedArrayModel(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    @test reduced !== nothing
    local (solved, _) = OMBackend.Passes.IndexReduction.solvedInArrays(reduced)
    SC.lowerArraySystem(OMBackend.Passes.arrayCodeForm!(solved))
  end
  local variable = (sc::SC.SIM_CODE, key::String) -> last(sc.stringToSimVarHT[key])

  @testset "an array state and parameter whole, a for-equation kept" begin
    local sc::SC.SIM_CODE = lowered("ArrayPath.Decay")
    local x::SC.SIMVAR = variable(sc, "x")
    @test x.varKind isa SC.STATE && x.dims == [1000]
    #= `each start = 1, each fixed = true`: evaluated per element =#
    @test x.attributes.start isa SC.ARRAY_EXP && all(e -> e == SC.RCONST(1.0), x.attributes.start.elements)
    @test x.attributes.fixed isa SC.ARRAY_EXP && all(e -> e == SC.BCONST(true), x.attributes.fixed.elements)
    @test sc.referenceIndex.elementNames["x"][1] == "x[1]" && length(sc.referenceIndex.elementNames["x"]) == 1000
    local k::SC.SIMVAR = variable(sc, "k")
    @test k.varKind isa SC.PARAMETER && k.dims == [1000] && SC.isArrayParameter(k)
    @test k.varKind.value isa SC.ARRAY_EXP && k.varKind.value.elements[3] == SC.RCONST(3.0)
    #= der(x[i]) = -k[i] x[i] for i in 1:1000: one for-equation, its element references subscripted =#
    @test length(sc.arrayEquations) == 1 && isempty(sc.residualEquations)
    local eq = only(sc.arrayEquations)
    @test eq isa SC.FOR_EQUATION && only(eq.body) isa SC.EQUATION
    local body::SC.EQUATION = only(eq.body)
    @test body.lhs isa SC.ASUB && body.lhs.exp isa SC.DER && String(body.lhs.exp.cref.sym) == "x"
  end

  @testset "a when-equation's triggers as a vector, initial() among them" begin
    local sc::SC.SIM_CODE = lowered("ArrayPath.VectorWhen")
    local whens = [eq for eq in sc.arrayEquations if eq isa SC.WHEN_EQUATION]
    local triggered = [w for w in whens if w.whenEquation.condition isa SC.ARRAY_EXP]
    @test length(triggered) == 1
    local c::SC.ARRAY_EXP = only(triggered).whenEquation.condition
    @test length(c.elements) == 3 && SC.isBuiltin(c.elements[3], :initial)
    @test variable(sc, "n").varKind isa SC.DISCRETE && variable(sc, "n").attributes.start == SC.RCONST(0.0)
  end

  @testset "parameters with fixed = false: their starts, their initial equations" begin
    local sc::SC.SIM_CODE = lowered("ArrayPath.FreeParam")
    local k::SC.SIMVAR = variable(sc, "k")
    @test k.varKind isa SC.PARAMETER && k.varKind.bindExp === nothing
    @test k.attributes.fixed == SC.BCONST(false) && k.attributes.start == SC.RCONST(3.0)
    @test length(sc.arrayInitialEquations) == 2 && all(eq -> eq isa SC.EQUATION, sc.arrayInitialEquations)
  end

  @testset "a when algorithm's targets from pre(), reinit subscripts, an unlowered branch" begin
    local sc::SC.SIM_CODE = lowered("ArrayPath.WhenReadsTarget")
    local w = only(eq for eq in sc.arrayEquations if eq isa SC.WHEN_EQUATION)
    #= the body starts s from pre(s), then the tuple's elements =#
    local body = w.whenEquation.body
    @test length(body) == 3 && string(body[1]) == "s := pre(s)" && all(st -> st isa SC.ASSIGN, body)
    local re::SC.SIM_CODE = lowered("ArrayPath.ReinitElements")
    local reinits = SC.REINIT[]
    local visit = (eq) -> eq isa SC.FOR_EQUATION ? foreach(visit, eq.body) :
      eq isa SC.WHEN_EQUATION ? append!(reinits, [st for st in eq.whenEquation.body if st isa SC.REINIT]) : nothing
    foreach(visit, re.arrayEquations)
    @test length(reinits) == 2 && all(r -> String(r.stateVar.sym) == "v" && length(r.subscripts) == 1, reinits)
    #= an equation of a branch that does not lower: kept as the reason, refused only where the branch is taken =#
    local eq = OMBackend.SimCode.UNLOWERED_EQUATION("an algorithm", "x := 1", SC.Source("m.mo", 1))
    @test string(eq) == "an algorithm: x := 1"
    #= a tuple output's position in the generated function's result: a record output before it is its fields =#
    local pair = SC.TYPE_COMPLEX(true, ["a", "b"], SC.SType[SC.TYPE_REAL(), SC.TYPE_REAL()])
    local tupleTy = SC.TYPE_TUPLE(SC.SType[pair, SC.TYPE_REAL(), SC.TYPE_INTEGER()])
    @test [SC.outputPosition(tupleTy, k) for k in 1:3] == [1, 3, 4] && SC.outputPosition(SC.TYPE_REAL(), 1) == 1
    #= a table over an enumeration's dimensions: its elements' values =#
    local en::SC.SIM_CODE = lowered("ArrayPath.EnumTable")
    @test variable(en, "T").dims == [4, 4] && variable(en, "T").varKind.value isa SC.ARRAY_EXP
  end
end