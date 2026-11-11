#= A discrete Real is a state with a zero derivative. Where an event moves the integrator to its time, the solver
   interpolates every state, and the dense output of a constant can be off by an ulp: a discrete Real changed at
   events that do not write it (MSL TraceSubstances.RoomCO2WithControls: the table's next event time 15.0 became
   15.000000000000002 at a controller's event, and ModelicaStandardTables compares it with ==). Every continuous
   callback moves the integrator with the held discretes put back now (HeldBack, integralDiscretes.jl). The values
   are compared exactly: repeated additions, as the when-equations compute them. Models/HeldDiscretes.mo. =#
using Test
import OMBackend
import OMFrontend

"""The values `sym` takes in `sol`, after each instant (the last entry of a repeated time), without repetitions."""
function heldDiscreteValues(sol, sym::Symbol)::Vector{Float64}
  local values::Vector{Float64} = Float64[]
  for i::Int in eachindex(sol.t)
    (i == lastindex(sol.t) || sol.t[i + 1] != sol.t[i]) || continue
    local v::Float64 = sol[sym][i]
    (isempty(values) || values[end] != v) && push!(values, v)
  end
  return values
end

function heldDiscreteRun(model::String)
  local file::String = joinpath(@__DIR__, "Models", "HeldDiscretes.mo")
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
  return OMBackend.simulateModel(OMBackend.canonicalName(model); tspan = (0.0, 2.0), reltol = 1.0e-6)
end

@testset "Held discretes keep their values where an event moves the integrator" begin
  local third::Float64 = 1 / 3
  @testset "with a re-initialization at each event" begin
    local sol = heldDiscreteRun("HeldDiscretes.NextEventBetweenCrossings")
    @test string(sol.retcode) == "Success"
    @test heldDiscreteValues(sol, :next) == accumulate(+, fill(third, 7))
  end
  @testset "an ODE: no re-initialization" begin
    local sol = heldDiscreteRun("HeldDiscretes.NextEventBetweenWhens")
    @test string(sol.retcode) == "Success"
    @test heldDiscreteValues(sol, :next) == accumulate(+, fill(third, 7))
    @test heldDiscreteValues(sol, :count) == accumulate(+, vcat(0.0, fill(0.1, 13)))
  end
end
