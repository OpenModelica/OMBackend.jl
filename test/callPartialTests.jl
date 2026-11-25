#= The terms of a function's element extractors and flat calls (CodeGeneration.CALL_RGFS) have partials by
   central differences in the symbolic Jacobian (CallPartial, OMBACKEND_CALL_PARTIALS): without a rule the
   whole Jacobian was finite differences of the right-hand side (MSL RoomCO2WithControls: the step of a
   pressure of 1e5 crossed its ducts' zero-flow regularization, 3000 steps to 0.0037 s). Only where the
   numeric partials are on (the Jacobian and the time derivative); the partials nest. =#
using Test
import OMBackend

@testset "Partials of call terms" begin
  local CG = OMBackend.CodeGeneration
  local S = CG.Symbolics
  CG.MODELICA_FUNCTION_IMPLS[:omjlPartialPlain] = (a, b) -> (a^2 * b, sin(a) + b)
  CG.MODELICA_FUNCTION_IMPLS[:omjlPartialFlat] = (x, v) -> x * v[1] + v[2]^3
  local field2 = CG.getOrCreateElemFunc(:omjlPartialPlain, (2,), 2)
  local flat = CG._getOrCreateFlatScalarFunc(:omjlPartialFlat, 3, ((), (2,)))
  @test field2 in CG.CALL_RGFS && flat in CG.CALL_RGFS
  local (a, b, c) = (S.variable(:a), S.variable(:b), S.variable(:c))
  local term(f, args...) = S.wrap(CG.makeSymbolicTerm(f, Any[S.unwrap(x) for x in args]))
  #= a partial term evaluated: its CallPartial at numeric arguments =#
  local at(ex, vals...) = Base.invokelatest(S.operation(S.unwrap(ex)), vals...)
  #= no rule without the numeric partials: the derivative stays a Differential term =#
  @test S.unwrap(S.expand_derivatives(S.Differential(a)(term(field2, a, b)))) isa S.SymbolicUtils.BasicSymbolic
  @test !(S.operation(S.unwrap(S.expand_derivatives(S.Differential(a)(term(field2, a, b))))) isa CG.CallPartial)
  CG.withNumericPartials() do
    local d = S.expand_derivatives(S.Differential(a)(term(field2, a, b)))
    @test S.operation(S.unwrap(d)) isa CG.CallPartial
    @test at(d, 0.3, 2.0) ≈ cos(0.3) rtol = 1e-7
    local dd = S.expand_derivatives(S.Differential(a)(d))
    @test S.operation(S.unwrap(dd)) isa CG.CallPartial{2, 2}
    @test at(dd, 0.3, 2.0) ≈ -sin(0.3) rtol = 1e-4
    #= the flat call: d/dc of x v1 + v2^3 at v2 = c =#
    local df = S.expand_derivatives(S.Differential(c)(term(flat, a, b, c)))
    @test at(df, 2.0, 5.0, 1.5) ≈ 3 * 1.5^2 rtol = 1e-7
    withenv("OMBACKEND_CALL_PARTIALS" => "false") do
      @test !(S.operation(S.unwrap(S.expand_derivatives(S.Differential(a)(term(field2, a, b))))) isa CG.CallPartial)
    end
  end
end

"The calls in `ex` whose function is of type `T`."
function _callsOf(ex::Expr, T::Type)::Int
  local n::Int = 0
  local worklist::Vector{Any} = Any[ex]
  while !isempty(worklist)
    local e::Any = pop!(worklist)
    e isa Expr || continue
    (e.head === :call && !isempty(e.args) && e.args[1] isa T) && (n += 1)
    append!(worklist, e.args)
  end
  return n
end

#= The Jacobian's partials of a call's element extractors: one WholePartials call per call, each partial read from
   it (_hoistCallPartials!; a CallPartial per extractor and argument was 5146 calls of 105 arguments in MSL
   Spice3BenchmarkFourBitBinaryAdder, a function that did not compile in 12 GB). The nonzeros' code in parts of at
   most DIRECT_JAC_FUNCTION_NODE_LIMIT nodes (_nonzeroFunctions), the values as one function's. =#
@testset "A call's extractor partials grouped; the Jacobian's code in parts" begin
  local CG = OMBackend.CodeGeneration
  local S = CG.Symbolics
  CG.MODELICA_FUNCTION_IMPLS[:omjlPartialWhole] = (a, b, k) -> (a * b + k, sin(a) * k, b^3)
  local fields::Vector{Function} = Function[CG.getOrCreateElemFunc(:omjlPartialWhole, (i,), 3) for i in 1:3]
  local term(f, args...) = S.wrap(CG.makeSymbolicTerm(f, Any[S.unwrap(v) for v in args]))
  local (x, y) = (S.variable(:x), S.variable(:y))
  #= two calls: (x, y, 2) read by all three extractors, (y, x, 3) by the first =#
  local rhs::Vector{Any} = Any[S.unwrap(term(fields[1], x, y, 2.0) + term(fields[2], x, y, 2.0)),
                               S.unwrap(term(fields[3], x, y, 2.0) * term(fields[1], y, x, 3.0))]
  local states::Vector{Any} = Any[S.unwrap(x), S.unwrap(y)]
  local iv = S.unwrap(S.variable(:t))
  local u::Vector{Float64} = [0.3, 1.5]
  #= r1 = x y + 2 + 2 sin x, r2 = y^3 (y x + 3) =#
  local exact::Matrix{Float64} = [u[2] + 2cos(u[1])  u[1];
                                  u[2]^4             3u[2]^2 * (u[2] * u[1] + 3) + u[2]^3 * u[1]]
  CG.withNumericPartials() do
    local nz::Vector{Any} = collect(Any, S.SparseArrays.nonzeros(CG._dagSparseJacobian(rhs, states)))
    local ex::Expr = CG._nonzeroExpression(nz, states, Any[], iv)
    @test _callsOf(ex, CG.CallPartial) == 0
    @test _callsOf(ex, CG.WholePartials) == 2
    local (jac::Function, proto::S.SparseArrays.SparseMatrixCSC{Float64, Int}) = CG._buildSparseJacobian(rhs, states, Any[], iv, u, Float64[], 0.0)
    local J::S.SparseArrays.SparseMatrixCSC{Float64, Int} = copy(proto)
    jac(J, u, Float64[], 0.0)
    @test Matrix(J) ≈ exact rtol = 1e-7
    local limit::Int = OMBackend.DIRECT_JAC_FUNCTION_NODE_LIMIT[]
    OMBackend.DIRECT_JAC_FUNCTION_NODE_LIMIT[] = 10
    try
      @test length(CG._nonzeroFunctions(nz, states, Any[], iv)) == length(nz)
      local (jacParts::Function, protoParts::S.SparseArrays.SparseMatrixCSC{Float64, Int}) =
        CG._buildSparseJacobian(rhs, states, Any[], iv, u, Float64[], 0.0)
      local JParts::S.SparseArrays.SparseMatrixCSC{Float64, Int} = copy(protoParts)
      jacParts(JParts, u, Float64[], 0.0)
      @test JParts ≈ J rtol = 1e-12
    finally
      OMBackend.DIRECT_JAC_FUNCTION_NODE_LIMIT[] = limit
    end
  end
  #= the function and its extractors out of the registries: a later test in the process sees only its own =#
  delete!(CG.MODELICA_FUNCTION_IMPLS, :omjlPartialWhole)
  for (i, f) in enumerate(fields)
    delete!(CG.ELEM_FUNC_CACHE, (:omjlPartialWhole, (i,), 3))
    delete!(CG.EXTRACTOR_INFO, f)
    delete!(CG.CALL_RGFS, f)
  end
end
