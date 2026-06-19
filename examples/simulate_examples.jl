# Runnable demo: translate + simulate the example DAEs through the OMBackend MTK
# pipeline. Run from an environment where OMBackend is available:
#
#     julia --project=. examples/simulate_examples.jl
#
import OMBackend
include(joinpath(@__DIR__, "ExampleDAEs.jl"))
import .ExampleDAEs

# (model name, ExampleDAEs binding, tspan)
const CASES = [
  ("HelloWorld",    :helloWorld_DAE,    (0.0, 1.0)),
  ("LotkaVolterra", :lotkaVolterra_DAE, (0.0, 10.0)),
  ("VanDerPol",     :vanDerPol_DAE,     (0.0, 10.0)),
]

for (name, sym, tspan) in CASES
  dae = getproperty(ExampleDAEs, sym)
  OMBackend.translate(dae; BackendMode = OMBackend.MTK_MODE)
  sol = OMBackend.simulateModel(name; tspan = tspan)
  inner = hasproperty(sol, :diffEqSol) ? getfield(sol, :diffEqSol) : sol
  rc = try; string(inner.retcode); catch; "?"; end
  @info "simulated $name" tspan retcode = rc
end
