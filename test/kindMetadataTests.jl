#= What the generated model's variables are in the simulation code (`SimCodeKind`, `IfConditionParameter`), by the
   models of Models/KindMetadata.mo. =#
using Test
import OMBackend
import OMFrontend

"Translate `KindMetadata.<name>` and simulate it to `stop`."
function kindModel(name::String, stop::Float64)
  local file::String = joinpath(@__DIR__, "Models", "KindMetadata.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("KindMetadata." * name,
                                                      OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  return OMBackend.simulateModel(OMBackend.canonicalName("KindMetadata." * name); tspan = (0.0, stop))
end

#= MTK's derived variables keep the metadata of the variable they derive from (Symbolics' rename: `xˍt` from `x`):
   taken for x, xˍt was error-controlled and missing from MTK's derivative variables. =#
@testset "MTK's derivative variables are not the model's" begin
  local CG::Module = OMBackend.CodeGeneration
  withenv("OMBACKEND_INDEX_REDUCTION" => "false") do
    local sol = kindModel("Pendulum", 0.5)
    @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
    local sys::OMBackend.ModelingToolkit.AbstractSystem = sol.prob.f.sys
    local unknowns::Vector = OMBackend.ModelingToolkit.unknowns(sys)
    local mtkMade::Vector{String} = String[CG.variableName(u) for u in unknowns if CG.simCodeKind(u) === nothing]
    @test !isempty(mtkMade) && all(n -> endswith(n, "ˍt"), mtkMade)
    @test Set(CG.variableName(v) for v in CG.mtkDerivativeVariables(sys) if any(u -> isequal(u, v), unknowns)) == Set(mtkMade)
    @test all(u -> (local k::Union{Nothing, OMBackend.CodeGeneration.SimCodeKind} = CG.simCodeKind(u); k === nothing || String(k.name) == CG.variableName(u)), unknowns)
    local tol::NamedTuple = CG.defaultToleranceKwargs(sol.prob, (; abstol = 1.0e-7))
    @test Set(CG.variableName(unknowns[i]) for i in eachindex(unknowns) if isinf(tol.abstol[i])) == Set(mtkMade)
  end
end

#= Only the if-equations' condition parameters are marked, not the hysteresis or the homotopy parameter declared with
   them; and the generated model, printed, parses (its kinds are constructor code). =#
@testset "If-condition parameters; the printed model" begin
  local CG::Module = OMBackend.CodeGeneration
  local sol = kindModel("Limiter", 1.0)
  local marked::Dict{String, Bool} = Dict{String, Bool}(CG.variableName(p) => CG.isIfConditionParameter(p)
                                    for p in OMBackend.ModelingToolkit.parameters(sol.prob.f.sys))
  @test marked[String(CG.ZC_HYSTERESIS)] == false && marked[String(CG.HOMOTOPY_LAMBDA)] == false
  @test any(((n, m),) -> startswith(n, "ifCond") && m, marked)
  local path::String = joinpath(mktempdir(), "Limiter.jl")
  OMBackend.writeModelToFile(OMBackend.canonicalName("KindMetadata.Limiter"), path)
  local code::Expr = Meta.parseall(read(path, String))
  @test !any(e -> e isa Expr && e.head in (:error, :incomplete), code.args)
end
