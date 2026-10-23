#= An if-expression's other branch is not evaluated: build_function's CSE assigns every subterm before the
   `if` that selects one, so a function in the branch not taken was called anyway and failed for the values
   there (MSL PsychrometricData's saturationTemperature at a partial pressure of 0); the code generation
   moves the assignments only one branch reads into it (_sinkBranchAssignments!). Models/LazyBranches.mo. =#
using Test
import OMBackend
import OMFrontend

@testset "Lazy if-expression branches" begin
  @testset "CSE assignments moved into their branch" begin
    local ex = :(function (out, u, p)
      var"##cse#1" = u[1] < 0.5
      var"##cse#2" = f(u[1])
      var"##cse#3" = g(var"##cse#2")
      var"##cse#4" = h(u[1])
      var"##cse#5" = if var"##cse#1"
        var"##cse#4"
      else
        var"##cse#3"
      end
      out[1] = var"##cse#5" + var"##cse#4"
      out
    end)
    OMBackend.CodeGeneration._sinkBranchAssignments!(ex)
    local body = ex.args[2]
    local tops = [st.args[1] for st in body.args if st isa Expr && st.head === :(=) && st.args[1] isa Symbol]
    #= ##cse#4 is read after the if as well: it stays; ##cse#2 and ##cse#3 only in the else branch =#
    @test tops == [Symbol("##cse#1"), Symbol("##cse#4"), Symbol("##cse#5")]
    local ifex = body.args[findfirst(st -> st isa Expr && st.head === :(=) && st.args[1] === Symbol("##cse#5"), body.args)].args[2]
    local elseNames = [st.args[1] for st in ifex.args[3].args if st isa Expr && st.head === :(=)]
    @test elseNames == [Symbol("##cse#2"), Symbol("##cse#3")]
  end

  @testset "a failing call in the branch not taken" begin
    local file = joinpath(@__DIR__, "Models", "LazyBranches.mo")
    local (fm, cache) = OMFrontend.instantiateSCodeToFM("LazyBranches.Guarded", OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
    OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("LazyBranches.Guarded"); tspan = (0.0, 1.0))
    @test string(sol.retcode) == "Success"
    @test sol(0.0; idxs = :y) == 0.0
    @test sol(1.0; idxs = :y) ≈ log(1.0) atol = 1e-6
    @test sol(0.1; idxs = :z) == 1.0
    @test sol(0.4; idxs = :z) == 2.0
    @test sol(1.0; idxs = :z) ≈ 3 + log(2.0) atol = 1e-6
  end
end
