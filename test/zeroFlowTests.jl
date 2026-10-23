#= Zero flow and zero speed (Models/ZeroFlow.mo), MSL FluidHeatFlow WaterPump's causes:
   - the semiLinear chain of components without mass (MLS 3.7.2.5): their friction loss
     `frictionLoss * m_flow * dp` of a zero frictionLoss folded, so the energy balances join the chain
     (eliminateFrozenStates), and the chain read through the double negations the second alias
     elimination leaves (simplifySemiLinear): every enthalpy explicit, none undetermined at zero flow;
   - an if-equation on a noEvent condition evaluated as written (MLS 3.7.4), its condition's variable
     explicit: the pump's `noEvent(abs(w) < small)` true again when the speed is back to zero. =#
using Test
import OMBackend
import OMFrontend

const ZERO_FLOW_FILE = joinpath(@__DIR__, "Models", "ZeroFlow.mo")

"The flat model of `ZeroFlow.<name>` and its functions."
function _zfFlat(name::String)::Tuple
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("ZeroFlow." * name, OMFrontend.translateToSCode(OMFrontend.parseFile(ZERO_FLOW_FILE)))
  return (fm, OMFrontend.cacheToFunctionList(cache))
end

"The simulation code of `ZeroFlow.<name>`."
function _zfSimCode(name::String)
  local (fm, functions) = _zfFlat(name)
  return OMBackend.mtkSimCode(OMBackend.omjlirModel(fm; functionList = functions))
end

_zfResiduals(sc) = [sprint(show, r.exp) for r in sc.residualEquations]

@testset "Zero flow and zero speed" begin
  @testset "a semiLinear chain through components without mass" begin
    #= one semiLinear left, `H_flow = semiLinear(m_flow, h_source, h_sink)`; the enthalpies explicit =#
    local res = _zfResiduals(_zfSimCode("ZeroMassChain"))
    @test count(r -> occursin("semiLinear", r), res) == 1
    @test !any(r -> occursin("Q_flow", r), res)
    local (fm, functions) = _zfFlat("ZeroMassChain")
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("ZeroFlow.ZeroMassChain"); tspan = (0.0, 1.0), reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
    local h = OMBackend.getVariableValues(sol, "c2.h")
    @test h[findfirst(>=(0.25), sol.t)] ≈ 1e6
    @test h[findfirst(>=(0.75), sol.t)] ≈ 2e6
    #= m_flow = (0.5 - t) / 30: E(1) = (0.125 * 1e6 - 0.125 * 2e6) / 30 =#
    @test OMBackend.getVariableValues(sol, "E")[end] ≈ -1.25e5 / 30 rtol = 1e-6
  end

  @testset "an if-equation on noEvent(abs(w) < small)" begin
    local sc = _zfSimCode("NoEventSpeed")
    #= w = 2 d explicit; with the switch an unknown, as the condition's variable (the solver's rounding of zero) =#
    @test !("w" in sc.irreducibleVariables)
    @test "w" in withenv(() -> _zfSimCode("NoEventSpeed"), "OMBACKEND_IF_READS_IRREDUCIBLE" => "true").irreducibleVariables
    local (fm, functions) = _zfFlat("NoEventSpeed")
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("ZeroFlow.NoEventSpeed"); tspan = (0.0, 1.0), reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
    local tau = OMBackend.getVariableValues(sol, "tau")
    local at = t -> findfirst(>=(t), sol.t)
    #= tau = dp V / w = V^2 while the speed turns; 0 after 0.5 (a relay of the condition stayed false: no crossing of
       abs(w) - small reaches below its hysteresis, and tau * 0 = 0) =#
    @test tau[at(0.25)] ≈ exp(-2 * sol.t[at(0.25)]) rtol = 1e-6
    @test tau[end] == 0
    @test OMBackend.getVariableValues(sol, "dp")[end] == 0
  end

  @testset "the initialization from a flow guessed at 0, where the losses are flat" begin
    #= the flow's Jacobian column 0 at its guess: Newton could not move it, and the pressure took the loop's
       mismatch (MomentumBalanceFittings: 3333 Pa in three rows, the phases stalled); nudged to 1 =#
    local (fm, functions) = _zfFlat("FlatZeroFlow")
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("ZeroFlow.FlatZeroFlow"); tspan = (0.0, 1.0))
    @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
    @test OMBackend.getVariableValues(sol, "m_flow")[1] ≈ sqrt(1e4 / 3) rtol = 1e-6
    @test OMBackend.getVariableValues(sol, "p")[1] ≈ 110000 - 1e4 / 3 rtol = 1e-9
  end

  @testset "smooth(0, ...) with its event, smooth(1, ...) without" begin
    #= order 0, actualStream: the flow's zero an event (as noEvent, MSL InverseParameterization's pump left IF97's
       range); E(1) = int_0^0.5 (0.5 - t)(1 + t) dt + int_0.5^1 2 (0.5 - t) dt = 7/48 - 12/48 =#
    local stream = _zfResiduals(_zfSimCode("SmoothStream"))
    @test !any(r -> occursin("noEvent", r), stream)
    #= order 1: noEvent, no stop at zero speed; E(1) = (0.75^3 - 0.25^3) / 3 = 13/96 =#
    local load = _zfResiduals(_zfSimCode("SmoothQuadraticLoad"))
    @test any(r -> occursin("noEvent", r), load) && !any(r -> occursin("smooth", r), load)
    for (name, at, crossing, E) in (("SmoothStream", 0.5, true, -5 / 48), ("SmoothQuadraticLoad", 0.25, false, 13 / 96))
      local (fm, functions) = _zfFlat(name)
      OMBackend.translate(fm; functionList = functions)
      local sol = OMBackend.simulateModel(OMBackend.canonicalName("ZeroFlow." * name); tspan = (0.0, 1.0), reltol = 1e-8, abstol = 1e-8)
      @test sol.retcode == OMBackend.CodeGeneration.ModelingToolkit.SciMLBase.ReturnCode.Success
      @test any(t -> abs(t - at) < 1e-9, sol.t) == crossing
      @test OMBackend.getVariableValues(sol, "E")[end] ≈ E rtol = 1e-5
    end
  end
end
