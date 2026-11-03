#= A read by a literal subscript names one element: an enumeration's or a Boolean's literal (`p[E.b]`, `q[true]`) as an
   Integer's, so the parameters it reads are folded (SimCode eliminateConstantParameters; only an array read by a
   subscript that reads a variable keeps its elements, dynamicallyIndexedElements!). A small parameter array read by a
   variable subscript (`r[i]`) is lowered to a selection over `i` before: its value checked. Models/EnumIndexedParameters.mo. =#
using Test
import OMBackend
import OMFrontend
isdefined(@__MODULE__, :BackendTestMocks) || include(joinpath(@__DIR__, "backendTestMocks.jl"))

const EIP_SOLUTION = OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.AbstractODESolution

"The simulation code and the final value of `v` of `model` of Models/EnumIndexedParameters.mo, simulated to 1."
function _eipRun(model::String, v::String)::Tuple{OMBackend.SimCode.SIM_CODE, Float64}
  local file::String = joinpath(@__DIR__, "Models", "EnumIndexedParameters.mo")
  local (fm::OMFrontend.Frontend.FlatModel, cache::OMFrontend.Frontend.FunctionTreeImpl.Tree) =
    OMFrontend.instantiateSCodeToFM(model, OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local simCode::OMBackend.SimCode.SIM_CODE = OMBackend.LAST_SIM_CODE[]
  local sol::EIP_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName(model); tspan = (0.0, 1.0))
  return (simCode, OMBackend.getVariableValues(sol, v)[end])
end

"The keys of the parameters of the simulation code `sc`, sorted."
_eipParameters(sc::OMBackend.SimCode.SIM_CODE)::Vector{String} =
  sort(String[k for (k, (_, v)) in sc.stringToSimVarHT if v.varKind isa OMBackend.SimCode.PARAMETER])

@testset "Parameters read by literal and by variable subscripts" begin
  local (literal::OMBackend.SimCode.SIM_CODE, x::Float64) = _eipRun("EnumIndexedParameters.Read", "x")
  @test isempty(_eipParameters(literal))
  @test x ≈ 12.0
  local (_, y::Float64) = _eipRun("EnumIndexedParameters.Dynamic", "y")
  @test isapprox(y, 2.0; atol = 1e-6)
end

#= The rule's other half on the simulation code (no translation: a variable subscript reaches SimCode only where the
   lowering keeps it): an array read by a subscript that reads a variable keeps its elements, its other parameters are
   folded. =#
@testset "An array read by a variable subscript keeps its elements" begin
  local M = BackendTestMocks
  local SC = OMBackend.SimCode
  local element::Function = (name::String, k::Int) -> M.simVar("$(name)[$(k)]", SC.PARAMETER(SC.RCONST(Float64(k))))
  local sc::SC.SIM_CODE = M.mockSimCode(SC.SIMVAR[element("A", 1), element("A", 2), element("A", 3), element("B", 1), element("B", 2),
                                                  M.simVar("i", SC.DISCRETE(), SC.TYPE_INTEGER()), M.simVar("y", SC.ALG_VARIABLE(0))])
  local arrays::Dict{String, Symbol} = Dict{String, Symbol}("A[$(k)]" => :A for k in 1:3)
  merge!(arrays, Dict{String, Symbol}("B[$(k)]" => :B for k in 1:2))
  local aRef::OMBackend.OMJLIR.ComponentRef = OMBackend.OMJLIR.ComponentRef(:A)
  local read::SC.Exp = SC.ASUB(SC.EXP_CREF(SC.SimCref(:A, Int[], aRef), SC.TYPE_ARRAY(SC.TYPE_REAL(), Int[3])), SC.Exp[M.simCref("i", SC.TYPE_INTEGER())])
  local residual::SC.Exp = SC.BINARY(M.simCref("y"), SC.OP_SUB, SC.BINARY(read, SC.OP_ADD, M.simCref("B[2]")))
  local index::SC.ReferenceIndex = sc.referenceIndex
  sc = SC.update(sc; referenceIndex = SC.ReferenceIndex(index.references, index.keyByReference, arrays, index.codeNames, index.elementNames),
                 residualEquations = SC.RESIDUAL_EQUATION[SC.RESIDUAL_EQUATION(residual, SC.NO_SOURCE, SC.EQ_ATTR_DEFAULT)])
  local kept::Vector{String} = sort(String[k for (k, (_, v)) in SC.eliminateConstantParameters(sc).stringToSimVarHT if v.varKind isa SC.PARAMETER])
  @test kept == ["A[1]", "A[2]", "A[3]"]
end
