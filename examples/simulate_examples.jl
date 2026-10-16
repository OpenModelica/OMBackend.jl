# Runnable demo: translate + simulate the example models through the OMBackend MTK
# pipeline. Run from an environment where OMBackend and OMFrontend are available:
#
#     julia --project=. examples/simulate_examples.jl
#
import OMBackend
include(joinpath(@__DIR__, "ExampleModels.jl"))
import .ExampleModels

# (model name, tspan)
const CASES = [
  ("HelloWorld",    (0.0, 1.0)),
  ("LotkaVolterra", (0.0, 10.0)),
  ("VanDerPol",     (0.0, 10.0)),
  ("BouncingBall",  (0.0, 3.0)),
]

for (name, tspan) in CASES
  local (fm, functions) = ExampleModels.flatModel(name)
  local (modelName, _) = OMBackend.translate(fm; functionList = functions, BackendMode = OMBackend.MTK_MODE)
  local sol = OMBackend.simulateModel(modelName; tspan = tspan)
  @info "simulated $name" tspan retcode = sol.retcode
end
