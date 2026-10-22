#= The model's Modelica functions compiled (OMBACKEND_COMPILED_FUNCTIONS, CodeGeneration.compiledFunction):
   functions of CodeGeneration, a body of one tuple of the arguments and an entry, not closures of the model
   module, which runs interpreted (MSL Spice3's MOSFET evaluation: 45 µs a call interpreted). A compiled
   function's call of another calls its entry, so the result is inferred; a call through the wrapper's global
   was Any. While precompiling: an RGF, which can define no closure, so a comprehension is a call of
   Base.Generator (Julia's lowering), and a form that would stay a closure keeps the function interpreted.
   Models/CompiledFunctions.mo: a comprehension and a nested call, compiled and interpreted. =#
using Test
import OMBackend
import OMFrontend

@testset "Compiled Modelica functions" begin
  local CG = OMBackend.CodeGeneration
  #= defined in this world: called as the wrappers and the call memo do, through invokelatest =#
  local call(f, args...) = Base.invokelatest(f, args...)
  #= the interpreted closure: the calls until the function has run as long as compiling it takes (CompileTier),
     and the symbolic ones (_symbolicImpl) =#
  local add = (a, b) -> a + 2b
  local f = CG.compiledFunction(:omjlCompiledAdd, :((a, b) -> a + 2b), (), @__MODULE__, add)
  @test f isa CG.CompiledModelicaFunction
  @test CG._symbolicImpl(f) === add && CG._symbolicImpl(add) === add
  local interpCalls = Ref(0)
  #= its budget of interpreted time long (the first call of this closure compiles it) =#
  local perNode = CG.COMPILE_AFTER_NS_PER_NODE[]
  CG.COMPILE_AFTER_NS_PER_NODE[] = 10^12
  local tiered = CG.compiledFunction(:omjlCompiledTiered, :((a, b) -> a * b), (), @__MODULE__, (a, b) -> (interpCalls[] += 1; a * b))
  @test !tiered.tier.compiled && tiered.tier.after == 0
  @test call(tiered, 2.0, 3.0) == 6.0 && interpCalls[] == 1 && !tiered.tier.compiled
  @test tiered.tier.after == CG._compileAfter(:omjlCompiledTiered) > 10^12
  CG.COMPILE_AFTER_NS_PER_NODE[] = perNode
  #= as if it had run that long: this call is the last interpreted one =#
  tiered.tier.interpreted = tiered.tier.after
  @test call(tiered, 2.0, 3.0) == 6.0 && interpCalls[] == 2 && tiered.tier.compiled
  @test call(tiered, 2.0, 4.0) == 8.0 && CG._callWithTuple(tiered, (1.0, 5.0)) == 5.0 && interpCalls[] == 2
  @test call(f, 1.0, 2.0) == 5.0
  @test_throws MethodError call(f, 1.0)
  #= 60 arguments: the body is inferred, no allocation =#
  local names = [Symbol("x", i) for i in 1:60]
  local sixty = Expr(:->, Expr(:tuple, names...), foldl((a, b) -> :($a + $b), names))
  local xs = ntuple(Float64, 60)
  for many in (CG.compiledFunction(:omjlCompiledMany, sixty, (), @__MODULE__, eval(sixty)), CG._compiledRGF(:omjlCompiledManyRGF, sixty, eval(sixty)))
    @test call(many, xs...) == sum(1:60)
    local alloc = () -> (many.f(xs); @allocated many.f(xs))
    @test call(alloc) == 0
  end
  #= a call of a compiled function: its entry, its record result inferred =#
  CG.createModelicaFunctionWrapper(:omjlCompiledPair, 2)
  CG.MODELICA_FUNCTION_IMPLS[:omjlCompiledPair] = CG.compiledFunction(:omjlCompiledPair, :((a, b) -> (a + b, a * b)), (), @__MODULE__, (a, b) -> (a + b, a * b))
  local outer = CG.compiledFunction(:omjlCompiledOuter, :(x -> begin (s, p) = omjlCompiledPair(x, 2.0); s * p end), (:omjlCompiledPair,), @__MODULE__,
                                    x -> begin (s, p) = CG.omjlCompiledPair(x, 2.0); s * p end)
  @test call(outer, 3.0) == 30.0
  outer.tier.compiled = true
  @test call(outer, 3.0) == 30.0
  #= its time interpreted by its code and the code of the function it calls, which compiles with it =#
  @test CG._compileAfter(:omjlCompiledOuter) ==
        CG.COMPILE_AFTER_NS_PER_NODE[] * (first(CG.COMPILED_CODE_SIZES[:omjlCompiledOuter]) + first(CG.COMPILED_CODE_SIZES[:omjlCompiledPair]))
  @test only(Base.return_types(outer.f, (Tuple{Float64},))) === Float64

  #= comprehensions (the RGF while precompiling): the frontend's form, a typed one, several iterators
     (their product), a generator =#
  for compile in ((ex, interp) -> CG.compiledFunction(gensym(:omjlCompiled), ex, (), @__MODULE__, interp),
                  (ex, interp) -> CG._compiledRGF(gensym(:omjlCompiledRGF), ex, interp))
    #= the bodies' (the interpreted closures' calls of Base.Generator are Julia's lowering) =#
    local compiled(ex) = (local c = compile(CG._compilableFunction(ex), eval(ex)); c.tier.compiled = true; c)
    local squares = compiled(Expr(:->, Expr(:tuple, :x, :n), Expr(:comprehension, :(x[k]^2), :(k = 1:n))))
    @test call(squares, [1.0, 2.0, 3.0], 3) == [1.0, 4.0, 9.0]
    local halves = compiled(Expr(:->, :n, Expr(:typed_comprehension, :Float64, Expr(:generator, :(k / 2), :(k = 1:n)))))
    @test call(halves, 3) == [0.5, 1.0, 1.5] && call(halves, 3) isa Vector{Float64}
    local grid = compiled(Expr(:->, :a, Expr(:comprehension, Expr(:generator, :(a * i + j), :(i = 1:2), :(j = 1:3)))))
    @test call(grid, 10) == [10i + j for i in 1:2, j in 1:3]
    local total = compiled(:(n -> sum(k^2 for k in 1:n)))
    @test call(total, 4) == 30
  end
  #= a name of the model's module (a constant of its code, MSL Fluid's `pipe4_Medium_nXi`) is read
     there; the body's own names (arguments, locals, iteration variables) are not =#
  local constants = Module(:OmjlCompiledModel)
  Core.eval(constants, :(omjlModelK = 3.0))
  local reads = CG.compiledFunction(:omjlCompiledModelName,
                                    :(x -> begin local y = x * omjlModelK; z = [y + k for k in 1:2]; sum(z) end), (), constants, x -> 0.0)
  reads.tier.compiled = true
  @test call(reads, 2.0) == 15.0
  #= closures it cannot turn into a call: the function stays interpreted =#
  @test CG._compilableFunction(:(x -> [k for k in x if k > 0])) === nothing
  @test CG._compilableFunction(:(x -> map(x) do k; k + 1; end)) === nothing

  local file = joinpath(@__DIR__, "Models", "CompiledFunctions.mo")
  for on in ("true", "false")
    withenv("OMBACKEND_COMPILED_FUNCTIONS" => on) do
      local (fm, cache) = OMFrontend.instantiateSCodeToFM("CompiledFunctions.M", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
      OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
      local sol = OMBackend.simulateModel(OMBackend.canonicalName("CompiledFunctions.M"); tspan = (0.0, 1.0))
      @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
      @test (CG.MODELICA_FUNCTION_IMPLS[:CompiledFunctions_norm2] isa CG.CompiledModelicaFunction) == (on == "true")
      #= the symbolic evaluations of the build called the closures: no body compiled for Symbolics' types
         (MSL R134a2: its translate 41 s, interpreted 8 s) =#
      if on == "true"
        local body = CG.MODELICA_FUNCTION_IMPLS[:CompiledFunctions_norm2].f
        local argTypes = [mi.specTypes.parameters[2] for m in methods(body) for mi in Base.specializations(m)]
        @test all(T -> all(a -> !(a <: CG.Symbolics.Num || a <: CG.Symbolics.SymbolicUtils.BasicSymbolic ||
                                   (a <: AbstractArray && eltype(a) <: Union{CG.Symbolics.Num, CG.Symbolics.SymbolicUtils.BasicSymbolic})),
                               T.parameters), argTypes)
      end
      #= y = 2 sqrt((3t)^2 + 4^2) =#
      @test OMBackend.getVariableValues(sol, "y")[1] ≈ 8.0
      @test OMBackend.getVariableValues(sol, "y")[end] ≈ 10.0
    end
  end
  #= translated twice: the second translate reuses the module and its build, and its functions
     registered in CodeGeneration are the module's again (registerModelFunctions!); a function that
     reads the model's constant was an UndefVarError (MSL RoomCO2WithControls). Both paths. Its inlined
     copy slices a constant table (constTableLookup of a range, a MethodError of its clamp). =#
  local name = OMBackend.canonicalName("CompiledFunctions.WithConstant")
  for on in ("true", "false"), k in 1:2
    withenv("OMBACKEND_COMPILED_FUNCTIONS" => on) do
      local (fm, cache) = OMFrontend.instantiateSCodeToFM("CompiledFunctions.WithConstant", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
      OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
      local sol = OMBackend.simulateModel(name; tspan = (0.0, 1.0))
      @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
      @test OMBackend.getVariableValues(sol, "h.s")[end] ≈ exp(-1) rtol = 1e-4
      @test OMBackend.getVariableValues(sol, "h.y")[end] ≈ 10.0
      @test OMBackend.getVariableValues(sol, "h.w")[end] ≈ 10.0
      @test CG.MODELICA_FUNCTION_OWNERS[:h_Med_g] === getfield(OMBackend, Symbol(name))
      @test Base.invokelatest(CG.MODELICA_FUNCTION_IMPLS[:h_Med_g], 1.0, [1.0, 2.0, 4.0]) == 7.0
    end
  end
end
