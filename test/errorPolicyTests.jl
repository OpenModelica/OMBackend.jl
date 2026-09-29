#= The error policy of fallbacks (src/errorPolicy.jl). =#
using Test
import OMBackend
import DAE
import MetaModelica

module ErrorPolicyForeign
struct Callable end
(::Callable)(x::Int) = x
@noinline asserting() = (@assert false "foreign"; nothing)
end
#= An assertion in OMBackend's own code (a source module). =#
@eval OMBackend @noinline _assertForTheTest() = (@assert false "ours"; nothing)

@testset "Error policy of fallbacks" begin
  local site() = try; error("boom"); catch e; OMBackend._fallback(e, :testSite); :fallback; end
  OMBackend.resetFallbacks!()
  @testset "an expected error takes the fallback, recorded once per site" begin
    @test site() === :fallback
    @test site() === :fallback
    local rows = OMBackend.fallbackSummary()
    @test length(rows) == 1 && rows[1][1] === :testSite && rows[1][2] == 2 && rows[1][5] == false
  end
  @testset "interrupts always propagate" begin
    @test_throws InterruptException (try; throw(InterruptException()); catch e; OMBackend._fallback(e, :interrupt); end)
  end
  @testset "a programming error: logged in observe mode, propagates without it" begin
    local bug() = try; undefinedNameForTheTest + 1; catch e; OMBackend._fallback(e, :bugSite); :fallback; end
    withenv("OMBACKEND_FALLBACK_ON_BUG" => "true") do
      @test_logs (:error, r"fallback\] bugSite") match_mode = :any (@test bug() === :fallback)
    end
    withenv("OMBACKEND_FALLBACK_ON_BUG" => "false") do
      @test_throws UndefVarError bug()
    end
    #= A site that names the type expects it. =#
    local expected() = try; undefinedNameForTheTest + 1; catch e; OMBackend._fallback(e, :expectedSite; expect = UndefVarError); :fallback; end
    withenv("OMBACKEND_FALLBACK_ON_BUG" => "false") do
      @test expected() === :fallback
    end
  end
  @testset "observe mode is off by default" begin
    local bug() = try; undefinedNameForTheTest + 1; catch e; OMBackend._fallback(e, :defaultSite); :fallback; end
    withenv("OMBACKEND_FALLBACK_ON_BUG" => nothing) do
      @test_throws UndefVarError bug()
    end
  end
  @testset "what counts as a programming error of OMBackend" begin
    local classify(f) = try; f(); catch e; OMBackend.isBug(e, catch_backtrace()); end
    #= A MethodError: of our function, of a foreign one, of a callable object. =#
    @test classify(() -> OMBackend.envSwitch(1))
    @test !classify(() -> sin("x"))
    local foreign = ErrorPolicyForeign.Callable()
    @test !classify(() -> foreign("x"))
    #= An assertion: in our code is ours; ModelingToolkit's (here a foreign module's) depends on the model. =#
    @test classify(() -> OMBackend._assertForTheTest())
    @test !classify(() -> ErrorPolicyForeign.asserting())
  end
  @testset "UnsupportedLowering: what the lowering cannot lower" begin
    local e = OMBackend.UnsupportedLowering("condition expression", repeat("x", 1000))
    @test !OMBackend.isBug(e, Base.backtrace())
    local shown = sprint(showerror, e)
    @test startswith(shown, "UnsupportedLowering: condition expression: xxx") && length(shown) < 400
    @test_throws OMBackend.UnsupportedLowering OMBackend.SimulationCode.DAE_identifierToString(42)
    local ty = DAE.T_REAL(MetaModelica.nil)
    local cref = DAE.CREF(DAE.CREF_IDENT("x", ty, MetaModelica.nil), ty)
    @test_throws OMBackend.UnsupportedLowering OMBackend.CodeGeneration.evalDAEConstant(cref)
    @test OMBackend.CodeGeneration.evalDAEConstant(DAE.RCONST(2.0)) == 2.0
    #= A fallback takes it: never a programming error. =#
    withenv("OMBACKEND_FALLBACK_ON_BUG" => nothing) do
      @test OMBackend._tryOr(() -> OMBackend.unsupported("statement", 1), :fallback, :unsupportedSite) === :fallback
    end
  end
  @testset "_tryOr" begin
    @test OMBackend._tryOr(() -> error("x"), 7, :tryOrSite) == 7
    @test OMBackend._tryOr(() -> 3, 7, :tryOrSite) == 3
  end
  OMBackend.resetFallbacks!()
end
