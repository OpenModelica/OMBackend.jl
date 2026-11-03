#= A call a when body reads in several statements is evaluated once (createWhenStatementsMTK's
   _whenBodyCalls). A body's if-statements are an assignment per target and a tuple assignment one per
   output, each with the whole call: MSL RealFFT's `(info, abs, arg) := realFFT(buf, nf)` was the call
   of all 4000 elements of buf 803 times, and Rectifier6pulseFFT's build passed 15 GB (MSL 3.2.3,
   2026-10-24). The bindings of the names a body reads were one per occurrence (_bodyReads) as well.
   Models/WhenBodyCalls.mo: the call once in a callback's body and once in the terminal body, the values
   as stats(buf, k) gives them. =#
using Test
import OMBackend
import OMFrontend

const WBC_SOLUTION = OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.AbstractODESolution

"""
The number of nodes of `code` that `isNode` holds for (any node: an Expr's head and arguments, a symbol).
"""
function _wbcCount(isNode::Function, code::Expr)::Int
  local n::Int = 0
  local walk(e::Any)::Nothing = (isNode(e) && (n += 1); e isa Expr && foreach(walk, e.args); nothing)
  walk(code)
  return n
end

"The calls of `f` (`OMBackend.CodeGeneration.tupleArrayElementAt`, ...) in `code`."
_wbcCalls(code::Expr, f::Expr)::Int = _wbcCount(e -> e isa Expr && e.head === :call && e.args[1] == f, code)

"The definitions of a when body's local functions in `code`: without arguments, each the call of a model function (`once() = ... modelicaFunctionResult(...) ...`)."
_wbcDefinitions(code::Expr)::Int =
  _wbcCount(e -> e isa Expr && e.head === :(=) && e.args[1] isa Expr && e.args[1].head === :call && length(e.args[1].args) == 1 &&
                 e.args[2] isa Expr && _wbcCalls(e.args[2], :(OMBackend.CodeGeneration.modelicaFunctionResult)) == 1, code)

"The final value of the variable `v` of `sol`."
_wbcFinal(sol::WBC_SOLUTION, v::String)::Float64 = OMBackend.getVariableValues(sol, v)[end]

@testset "A when body's repeated call evaluated once" begin
  local file::String = joinpath(@__DIR__, "Models", "WhenBodyCalls.mo")
  local (fm::OMFrontend.Frontend.FlatModel, cache::OMFrontend.Frontend.FunctionTreeImpl.Tree) =
    OMFrontend.instantiateSCodeToFM("WhenBodyCalls.Sampled", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local name::String = OMBackend.canonicalName("WhenBodyCalls.Sampled")
  local code::Expr = OMBackend.COMPILED_MODELS_MTK[name][1]
  #= no call per output (5 in a body before: s2 and y2's 4 elements), one evaluation per local function
     (the callback set has the `time > 0.55` body twice, the terminal body once) =#
  @test _wbcCalls(code, :(OMBackend.CodeGeneration.tupleArrayElementAt)) == 0
  @test _wbcCalls(code, :(OMBackend.CodeGeneration.tupleElementCall)) == 0
  @test _wbcCalls(code, :(OMBackend.CodeGeneration.modelicaFunctionResult)) == _wbcDefinitions(code) == 3
  local sol::WBC_SOLUTION = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
  @test string(sol.retcode) == "Success"
  @test _wbcFinal(sol, "s2") ≈ 10.0 && [_wbcFinal(sol, "y2[$i]") for i in 1:4] ≈ [3.0, 6.0, 9.0, 12.0]
  @test _wbcFinal(sol, "s") ≈ 10.0 && [_wbcFinal(sol, "y[$i]") for i in 1:4] ≈ [2.0, 4.0, 6.0, 8.0]
  #= a body's bindings of the names it reads, each once =#
  local body::Vector{OMBackend.SimCode.Statement} =
    filter(OMBackend.CodeGeneration._isTerminalWhen, OMBackend.LAST_SIM_CODE[].whenEquations)[1].whenEquation.body
  local reads::Vector{String} = OMBackend.CodeGeneration._bodyReads(body)
  @test allunique(reads) && "buf[1]" in reads
end

#= Bodies in one scope (the terminal bodies in one `let`, a when's elsewhen arms in one affect) each
   their own local functions: numbered alike, the later one replaced the earlier (each terminal body
   got the second's values, the then arm the elsewhen's). =#
@testset "Repeated calls of bodies that share a scope" begin
  local file::String = joinpath(@__DIR__, "Models", "WhenBodyCalls.mo")
  local (fm::OMFrontend.Frontend.FlatModel, cache::OMFrontend.Frontend.FunctionTreeImpl.Tree) =
    OMFrontend.instantiateSCodeToFM("WhenBodyCalls.TwoBodies", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol::WBC_SOLUTION = OMBackend.simulateModel(OMBackend.canonicalName("WhenBodyCalls.TwoBodies"); tspan = (0.0, 1.0))
  @test string(sol.retcode) == "Success"
  for (out, k) in (("", 2.0), ("3", 3.0), ("4", 4.0), ("5", 5.0))
    @test _wbcFinal(sol, "s" * out) ≈ 10.0
    @test [_wbcFinal(sol, "y$(out)[$i]") for i in 1:4] ≈ k .* [1.0, 2.0, 3.0, 4.0]
  end
end
