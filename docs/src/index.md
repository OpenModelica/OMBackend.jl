```@meta
CurrentModule = OMBackend
```

# OMBackend.jl

OMBackend.jl is one component of OMJL, the Modelica compiler infrastructure for
Julia. It takes a model in its own intermediate representation, OMJLIR, runs its
passes there, generates the simulation code, and from it a system that is
simulated with DifferentialEquations.jl and ModelingToolkit.jl.

## Pipeline

| Step | Where in the repository |
|---|---|
| The model as OMJLIR | `src/IR` |
| Passes on OMJLIR (index reduction, matching, sorting) | `src/Backend/Passes` |
| Simulation code (SimCode) | `src/SimCode` |
| Code generation for ModelingToolkit.jl and DifferentialEquations.jl, and the runtime | `src/CodeGeneration` |
| The user-facing functions (this reference) | `src/backendAPI.jl` |

## Frontends

The backend does not depend on a frontend. A frontend lowers its model to OMJLIR
through a package extension that Julia loads together with the frontend's package:

* `OMBackendOMFrontendExt` (loaded with
  [OMFrontend.jl](https://github.com/OpenModelica/OMFrontend.jl)) lowers
  OMFrontend's flat model and adds `OMBackend.translate(fm)` and the other entry
  points on it.
* `OMBackendOMCompilerExt` (loaded with OMCompiler.jl) takes omc's flat model.

Without either, [`OMBackend.translate`](@ref) takes an `OMJLIR.Model`.

## Where to read next

* [Getting started](getting-started.md): installation and an example.
* [API reference](api.md): translate, simulate, inspect and clear models.
* [OM.jl](https://github.com/JKRT/OM.jl) is the user-level package built on
  OMFrontend.jl and OMBackend.jl: `OM.simulate` runs the whole chain.
