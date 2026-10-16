# OMBackend examples

This directory contains example models in Modelica (`Examples.mo`) and their flat
models as OMBackend takes them: `ExampleModels.flatModel(name)` flattens one with
OMFrontend and returns the flat model and its functions. They are the inputs of the
test suite's pipeline tests (`test/runtests.jl` includes `ExampleModels.jl`).

## Models (`Examples.mo`)

| Model                    | Notes                       |
|--------------------------|-----------------------------|
| `HelloWorld`             | trivial scalar ODE          |
| `LotkaVolterra`          | predator–prey ODE           |
| `VanDerPol`              | oscillator                  |
| `BouncingBall`           | hybrid / events             |
| `SimpleMechanicalSystem` | mechanical                  |
| `SimpleCircuit`          | electrical                  |

## Run

From an environment where `OMBackend` and `OMFrontend` are available (e.g. the OMJL dev project):

```julia
julia --project=. examples/simulate_examples.jl
```

This translates each model with `OMBackend.translate(fm; functionList, BackendMode = OMBackend.MTK_MODE)`
and simulates it with `OMBackend.simulateModel(name; tspan)`.
