#= The generated right-hand side's calls of a function's element extractors (one per output field read)
   with equal arguments are one call of the function's whole result, the fields read from it
   (CodeGeneration._hoistExtractorCalls!): each extractor called the function through the call memo with all
   its arguments (MSL Spice3 Nor: 20 calls of 162 arguments; the right-hand side 21 µs with every call a
   memo hit, 1.8 µs hoisted). The plain, flat (array arguments rebuilt from scalars) and tuple extractors. =#
using Test
import OMBackend

@testset "Extractor calls hoisted" begin
  local CG = OMBackend.CodeGeneration
  local calls = Ref(0)
  CG.MODELICA_FUNCTION_IMPLS[:omjlHoistPlain] = (a, b) -> (calls[] += 1; (a + b, a * b))
  CG.MODELICA_FUNCTION_IMPLS[:omjlHoistFlat] = (x, v, m) -> (calls[] += 1; (x + v[1], x * v[2] + m[2, 1]))
  CG.MODELICA_FUNCTION_IMPLS[:omjlHoistTuple] = (a, b) -> (calls[] += 1; ([a, b], [a * b, a - b]))
  local p1 = CG.getOrCreateElemFunc(:omjlHoistPlain, (1,), 2)
  local p2 = CG.getOrCreateElemFunc(:omjlHoistPlain, (2,), 2)
  local shapes = ((), (2,), (2, 2))
  local f1 = CG._getOrCreateFlatElemFunc(:omjlHoistFlat, (1,), 7, shapes)
  local f2 = CG._getOrCreateFlatElemFunc(:omjlHoistFlat, (2,), 7, shapes)
  local t1 = CG.getOrCreateTupleElemFunc(:omjlHoistTuple, 2, (1,), 2)
  local t2 = CG.getOrCreateTupleElemFunc(:omjlHoistTuple, 1, (2,), 2)
  #= as build_function writes it with CSE: each extractor call an assignment's right-hand side =#
  local code() = :((out, u) -> begin
    c1 = $p1(u[1], u[2]); c2 = $p2(u[1], u[2]); c3 = $p1(u[2], u[1])
    c4 = $f1(u[1], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0); c5 = $f2(u[1], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    c6 = $t1(u[1], u[2]); c7 = $t2(u[1], u[2])
    out .= (c1, c2, c3, c4, c5, c6, c7)
  end)
  #= built afresh: a deepcopy would copy the extractors, which EXTRACTOR_INFO knows by identity =#
  local hoisted = CG._hoistExtractorCalls!(code())
  local ops = Any[]
  local walk(e) = e isa Expr && (e.head === :call && push!(ops, e.args[1]); foreach(walk, e.args))
  walk(hoisted)
  #= four argument sets: four whole results, no extractor left =#
  @test count(op -> op isa CG.ExtractorWhole, ops) == 4
  @test !any(op -> haskey(CG.EXTRACTOR_INFO, op), ops)
  local run(ex) = (local out = zeros(7); Base.invokelatest(CG._exprToRTGFunction(ex), out, [2.0, 3.0]); out)
  local expected = [5.0, 6.0, 5.0, 3.0, 2.0 * 2.0 + 4.0, 6.0, 3.0]
  @test run(code()) == expected
  calls[] = 0
  empty!(get(task_local_storage(), :ombackendCallMemo, Dict()))
  @test run(hoisted) == expected
  @test calls[] == 4
  #= the flat extractors' arrays rebuilt as they build them =#
  @test CG._reassembleArgs((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), Val(shapes)) == (1.0, [2.0, 3.0], [4.0 6.0; 5.0 7.0])
  #= an extractor call that is no assignment's right-hand side stays =#
  local nested() = :((out, u) -> (out[1] = 1 + $p1(u[1], u[2])))
  @test CG._hoistExtractorCalls!(nested()) == nested()
end
