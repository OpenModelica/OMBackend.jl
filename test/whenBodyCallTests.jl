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

@testset "A when body's repeated call evaluated once" begin
  local file = joinpath(@__DIR__, "Models", "WhenBodyCalls.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("WhenBodyCalls.Sampled", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local name = OMBackend.canonicalName("WhenBodyCalls.Sampled")
  local code = OMBackend.COMPILED_MODELS_MTK[name][1]
  local count(f) = (n = Ref(0); walk(e) = (e isa Expr && (e.head === :call && e.args[1] == f && (n[] += 1); foreach(walk, e.args))); walk(code); n[])
  #= no call per output (5 in a body before: s2 and y2's 4 elements), one evaluation per local function
     (the callback set has the `time > 0.55` body twice, the terminal body once) =#
  @test count(:(OMBackend.CodeGeneration.tupleArrayElementAt)) == 0
  @test count(:(OMBackend.CodeGeneration.tupleElementCall)) == 0
  local isDefinition(e) = e isa Expr && e.head === :(=) && e.args[1] isa Expr && e.args[1].head === :call &&
                          startswith(string(e.args[1].args[1]), "#whenCallOnce")
  local definitions = (n = Ref(0); walk(e) = (e isa Expr && (isDefinition(e) && (n[] += 1); foreach(walk, e.args))); walk(code); n[])
  @test count(:(OMBackend.CodeGeneration.modelicaFunctionResult)) == definitions == 3
  local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
  @test string(sol.retcode) == "Success"
  local last(v) = OMBackend.getVariableValues(sol, v)[end]
  @test last("s2") ≈ 10.0 && [last("y2[$i]") for i in 1:4] ≈ [3.0, 6.0, 9.0, 12.0]
  @test last("s") ≈ 10.0 && [last("y[$i]") for i in 1:4] ≈ [2.0, 4.0, 6.0, 8.0]
  #= a body's bindings of the names it reads, each once =#
  local body = filter(OMBackend.CodeGeneration._isTerminalWhen, OMBackend.LAST_SIM_CODE[].whenEquations)[1].whenEquation.body
  local reads = OMBackend.CodeGeneration._bodyReads(body)
  @test allunique(reads) && "buf[1]" in reads
end

#= Bodies in one scope (the terminal bodies in one `let`, a when's elsewhen arms in one affect) each
   their own local functions: numbered alike, the later one replaced the earlier (each terminal body
   got the second's values, the then arm the elsewhen's). =#
@testset "Repeated calls of bodies that share a scope" begin
  local file = joinpath(@__DIR__, "Models", "WhenBodyCalls.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("WhenBodyCalls.TwoBodies", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("WhenBodyCalls.TwoBodies"); tspan = (0.0, 1.0))
  @test string(sol.retcode) == "Success"
  local last(v) = OMBackend.getVariableValues(sol, v)[end]
  for (out, k) in (("", 2.0), ("3", 3.0), ("4", 4.0), ("5", 5.0))
    @test last("s" * out) ≈ 10.0
    @test [last("y$(out)[$i]") for i in 1:4] ≈ k .* [1.0, 2.0, 3.0, 4.0]
  end
end
