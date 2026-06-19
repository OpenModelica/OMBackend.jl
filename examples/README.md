# OMBackend examples

This directory contains example models expressed as hand-built `DAE.DAE_LIST`
values — the intermediate representation OMBackend consumes from the frontend.
They double as the test suite's stable backend contract (`test/runtests.jl`
includes `ExampleDAEs.jl` from here).

## Models (`ExampleDAEs.jl`)

| Binding              | Model name              | Notes                       |
|----------------------|-------------------------|-----------------------------|
| `helloWorld_DAE`     | `HelloWorld`            | trivial scalar ODE          |
| `lotkaVolterra_DAE`  | `LotkaVolterra`         | predator–prey ODE           |
| `vanDerPol_DAE`      | `VanDerPol`             | stiff oscillator            |
| `bouncingBall_DAE`   | `BouncingBall`          | hybrid / events             |
| `simpleMech_DAE`     | `SimpleMechanicalSystem`| mechanical                  |
| `simpleCircuit_DAE`  | `SimpleCircuit`         | electrical                  |

(`HelloWorld`, `LotkaVolterra` and `VanDerPol` are verified to translate and
simulate to `retcode = Success` via the MTK backend on Julia 1.12 / Windows.)

## Run

From an environment where `OMBackend` is available (e.g. the OMJL dev project):

```julia
julia --project=. examples/simulate_examples.jl
```

This translates each model with `OMBackend.translate(...; BackendMode = OMBackend.MTK_MODE)`
and simulates it with `OMBackend.simulateModel(name; tspan, solver)`.
