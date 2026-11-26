#= Author: John Tinnerholm

   The array code's calls of the model's pure functions with the same arguments made once in a generated function
   (ArrayPreserveDE._shareCalls): a record-valued call is one call per field read. MSL BatchPlant_StandardWater's
   right-hand side made 988 calls of IF97's waterBaseProp_ph with 10 different arguments (solve 2.08 -> 1.40 s,
   rebuild 11.0 -> 7.8 s, results bit-identical). A write of what a call reads, or a mutating call, ends its share;
   a call in a branch only stays there; an impure function's calls are all kept. =#
using Test
import OMBackend
import OMFrontend

"""
The statements `stmts` run as a function of `a`, `u`, `p`, `f` and `g!`: the buffer `a` after them and the number of
calls of `f`.
"""
function _runStatements(stmts::Vector{Any})::Tuple{Vector{Float64}, Int}
  local calls::Base.RefValue{Int} = Base.RefValue{Int}(0)
  local f::Function = (x::Float64) -> (calls[] += 1; x^2 + 1)
  local g!::Function = (a::Vector{Float64}) -> (a[6] = 7.0; nothing)
  local fn::Function = Core.eval(Main, :((a, u, p, f, g!) -> begin
                                           $(stmts...)
                                           a
                                         end))
  local a::Vector{Float64} = Base.invokelatest(fn, zeros(6), [1.0, 2.0], (d = [3.0], values = [4.0]), f, g!)
  return (a, calls[])
end

@testset "Shared calls of pure functions (array code)" begin
  local APDE = OMBackend.CodeGeneration.ArrayPreserveDE
  local stmts::Vector{Any} = Any[
    :(a[1] = f(u[1]) + f(u[1])),                 #= shared from here =#
    :(a[2] = if u[1] > 0; f(u[2]) else 0.0 end), #= in a branch only: stays =#
    :(a[3] = f(u[2]) * f(p.d[1])),               #= both shared from here =#
    :(p.d[1] = 2.0),                             #= a write of what f(p.d[1]) reads: evaluated again after it =#
    :(a[4] = f(p.d[1]) + f(u[1])),
    :(g!(a)),                                    #= a mutating call: every share ends =#
    :(a[5] = f(u[1]))]
  local shared::Vector{Any} = APDE._shareCalls(stmts, Set{Symbol}([:f]), Base.RefValue{Int}(0))
  @test shared[3] == stmts[2]
  local (plainA::Vector{Float64}, plainCalls::Int) = _runStatements(stmts)
  local (sharedA::Vector{Float64}, sharedCalls::Int) = _runStatements(shared)
  @test sharedA == plainA
  @test (plainCalls, sharedCalls) == (8, 6)
  #= f impure (not among the pure functions): every call kept =#
  @test APDE._shareCalls(stmts, Set{Symbol}(), Base.RefValue{Int}(0)) == stmts
  #= a nested call, a `let` that binds an argument's name, a statement that leaves early, a write of `p.values` =#
  local more::Vector{Any} = Any[
    :(a[1] = f(f(u[1])) + f(f(u[1]))),       #= the inner call shared: f(u[1]) once, the outer f twice =#
    :(a[2] = let u = [5.0]; f(u[1]) end),    #= its own u: stays; a binding of a buffer's name ends every share =#
    :(if f(u[2]) > 100.0; return a; end),    #= leaves early: shares none of its own =#
    :(a[3] = f(u[2]) + f(u[2]) + f(u[1])),   #= both shared from here =#
    :(a[4] = f(p.values[1])),                #= shared from here =#
    :(p.values[1] = 5.0),                    #= a write of what f(p.values[1]) reads: evaluated again after it =#
    :(a[5] = f(p.values[1]) + f(p.values[1]))]
  local sharedMore::Vector{Any} = APDE._shareCalls(more, Set{Symbol}([:f]), Base.RefValue{Int}(0))
  @test more[2] in sharedMore && more[3] in sharedMore
  local (plainMoreA::Vector{Float64}, plainMoreCalls::Int) = _runStatements(more)
  local (sharedMoreA::Vector{Float64}, sharedMoreCalls::Int) = _runStatements(sharedMore)
  @test sharedMoreA == plainMoreA
  @test (plainMoreCalls, sharedMoreCalls) == (12, 9)

  #= a model: a record bound by a call, read by its fields, and a function called in several equations =#
  local program = OMFrontend.OpenModelicaParser.Modelica"""
    package SharedCalls
      record R
        Real a; Real b; Real c;
      end R;
      function props
        input Real x;
        output R r;
      protected
        Real s = 0;
      algorithm
        for i in 1:3 loop
          s := s + x / 3;
        end for;
        r := R(s, 2 * s, s ^ 2);
      end props;
      function sq
        input Real x;
        output Real y = 0;
      algorithm
        for i in 1:2 loop
          y := y + x ^ 2 / 2;
        end for;
      end sq;
      model M
        Real x(start = 1, fixed = true);
        R r = props(x);
        Real y;
        Real w;
      equation
        der(x) = -r.a + sq(x) - sq(x);
        y = r.b + r.c + sq(x);
        w = if time > 0.5 then sq(2 * x) else sq(x);
      end M;
    end SharedCalls;"""
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("SharedCalls.M", OMFrontend.translateToSCode(program); scalarize = false)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarize = false)
  local name::String = OMBackend.canonicalName("SharedCalls.M")
  try
    @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
    #= the calls of each generated function by callee =#
    local calls::Dict{Tuple{Symbol, Symbol}, Int} = Dict{Tuple{Symbol, Symbol}, Int}()
    local walk!(x::Any, fn::Symbol)::Nothing = begin
      x isa Expr || return nothing
      if x.head === :function && x.args[1] isa Expr && x.args[1].args[1] isa Symbol
        fn = x.args[1].args[1]
      elseif x.head === :call && x.args[1] in (:SharedCalls_props, :SharedCalls_sq)
        calls[(fn, x.args[1])] = get(calls, (fn, x.args[1]), 0) + 1
      end
      foreach(a::Any -> walk!(a, fn), x.args)
      return nothing
    end
    walk!(OMBackend.getCompiledModelArrayPreserveDE(name), :module)
    #= RHS!: sq(x) - sq(x) one call; equations!: props(x) once for r.a, r.b, r.c, sq(x) once for der(x) and y, and in
       w's branches sq(2 x) and sq(x) =#
    @test calls[(:RHS!, :SharedCalls_sq)] == 1
    @test calls[(:equations!, :SharedCalls_props)] == 1
    @test calls[(:equations!, :SharedCalls_sq)] == 3
    local sol = OMBackend.simulate(name; tspan = (0.0, 1.0), reltol = 1e-10, abstol = 1e-12)
    local x1::Float64 = exp(-1.0)
    @test isapprox(OMBackend.variableValues(sol, "x")[end], x1; rtol = 1e-9)
    @test isapprox(OMBackend.variableValues(sol, "y")[end], 2 * x1 + 2 * x1^2; rtol = 1e-9)
    @test isapprox(OMBackend.variableValues(sol, "w")[end], 4 * x1^2; rtol = 1e-9)
  finally
    OMBackend.forgetModel(name)
  end
end
