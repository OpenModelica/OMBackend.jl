#= The build function's `Base.invokelatest(() -> body)` calls hoisted into functions of the model's module
   (CodeGeneration.hoistBuildClosures!): the locals a body reads passed as arguments, the same body once, the
   parts named after the build function (the structural modes of a model share its module), and the locals the
   build function binds in each form found (_buildLocals): a local missed would be read as the module's global. =#
using Test
import OMBackend

@testset "Build closures hoisted" begin
  local CG = OMBackend.CodeGeneration
  local build::Expr = quote
    function aModel(tspan = (0.0, 1.0); scale::Float64 = 1.0)
      ModelingToolkit.@variables x(t) y(t)[1:2] z
      ModelingToolkit.@parameters begin
        k = 1.0
        m(t)
      end
      @named sys = makeSystem(x)
      (n1::Int, n2) = (1, 2)
      half(v) = v / 2
      isempty(tspan) || (flag = true)
      events = Base.invokelatest(() -> [x, k, scale, half(n1), globalName])
      again = Base.invokelatest(() -> [x, k, scale, half(n1), globalName])
      for i in 1:2
        loopOnly = i
      end
      return (events, again, sys, n2, flag, y, z, m)
    end
  end
  local fn::Expr = only(filter(a -> a isa Expr && a.head === :function, build.args))
  local locals::Set{Symbol} = CG._buildLocals(fn)
  for name in (:tspan, :scale, :x, :y, :z, :k, :m, :sys, :n1, :n2, :half, :flag, :events, :again)
    @test name in locals
  end
  @test !(:loopOnly in locals)
  @test !(:globalName in locals)

  #= Two builds in one module (two structural modes): each its own parts. =#
  local other::Expr = deepcopy(fn)
  other.args[1].args[1] = :bModel
  local model::Expr = Expr(:block, fn, other)
  CG.hoistBuildClosures!(model, :aModel)
  CG.hoistBuildClosures!(model, :bModel)
  local parts::Vector{Symbol} = Symbol[a.args[1].args[1] for a in model.args
                                       if a isa Expr && a.head === :function && !(a.args[1].args[1] in (:aModel, :bModel))]
  #= the same body twice is one part per build =#
  @test sort(parts) == [:_aModel_part1, :_bModel_part1]
  local aPart::Expr = only(filter(a -> a isa Expr && a.head === :function && a.args[1].args[1] === :_aModel_part1, model.args))
  #= the locals the body reads are its arguments; the global is read from the module =#
  @test sort(Symbol[s for s in aPart.args[1].args[2:end]]) == sort([:half, :k, :n1, :scale, :x])
end

#= The chunks of the equations, start values and parameter values (decompose*Inline) are hoisted too: the build
   function holds no closure (a 4000-element chain had 480 chunk closures in it, 10 s to lower). =#
@testset "Equation chunks hoisted" begin
  local CG = OMBackend.CodeGeneration
  local starts::Vector{Expr} = [:($(Symbol("s", i)) => k * $(i)) for i in 1:5]
  local pars::Vector{Expr} = [:($(Symbol("p", i)) => k + $(i)) for i in 1:5]
  local eqs::Vector{Expr} = [:($(Symbol("x", i)) ~ k * $(i)) for i in 1:5]
  local build::Expr = quote
    function cModel(k::Float64)
      startEquationComponents = Any[]
      $(CG.decomposeStartEquationsInline(starts, :startEquationComponents; chunkSize = 2))
      $(CG.decomposeParameterEquationsInline(pars; chunkSize = 2))
      equationComponents = Any[]
      $(CG.decomposeEquationsInline(eqs, Expr[:(scale = 2k)], :equationComponents; chunkSize = 2))
      return (collect(Iterators.flatten(startEquationComponents)), pars, scale,
              collect(Iterators.flatten(equationComponents)))
    end
  end
  local model::Expr = Expr(:block, only(filter(a -> a isa Expr && a.head === :function, build.args)))
  @test CG._closureCount(model) == 9
  CG.hoistBuildClosures!(model, :cModel)
  local fn::Expr = only(filter(a -> a isa Expr && a.head === :function && a.args[1].args[1] === :cModel, model.args))
  @test CG._closureCount(fn) == 0
  local parts::Vector{Expr} = filter(a -> a isa Expr && a.head === :function && a !== fn, model.args)
  @test length(parts) == 9
  @test all(p -> Symbol[a for a in p.args[1].args[2:end]] == [:k], parts)
  #= the values as before: the chunks in order =#
  local m::Module = Module(:HoistChunks)
  Core.eval(m, :(const Symbolics = $(CG.Symbolics)))
  Core.eval(m, :(const var"~" = $(CG.Symbolics.:~)))
  for i in 1:5
    Core.eval(m, :($(Symbol("s", i)) = $(QuoteNode(Symbol("s", i)))))
    Core.eval(m, :($(Symbol("p", i)) = $(QuoteNode(Symbol("p", i)))))
    Core.eval(m, :($(Symbol("x", i)) = Symbolics.variable($(QuoteNode(Symbol("x", i))))))
  end
  Core.eval(m, model)
  local (values, parameterValues, scale, equations) = Base.invokelatest(getglobal(m, :cModel), 2.0)
  @test values == Any[Symbol("s", i) => 2.0 * i for i in 1:5]
  @test parameterValues == Dict{Any, Any}(Symbol("p", i) => 2.0 + i for i in 1:5)
  @test scale == 4.0
  @test [string(e) for e in equations] == [string(CG.Symbolics.variable(Symbol("x", i)) ~ 2.0 * i) for i in 1:5]
end
