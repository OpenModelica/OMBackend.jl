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

The backend does not depend on a frontend. A frontend's flat model reaches OMJLIR
through a package extension, which Julia loads when the frontend's package is
loaded next to OMBackend:

| Frontend | Load | Extension | Call |
|---|---|---|---|
| [OMFrontend.jl](https://github.com/OpenModelica/OMFrontend.jl), in Julia | `import OMFrontend` | `OMBackendOMFrontendExt` | `OMBackend.translate(fm; functionList = ...)`, see [Getting started](getting-started.md) |
| omc, through [OMCompiler.jl](https://github.com/OpenModelica/OpenModelica/tree/master/OMCompiler/Compiler/OpenModelica.rs/OMCompiler.jl) (experimental) | `import OMCompiler` | `OMBackendOMCompilerExt` | `OMBackend.translate(OMBackend.lowerOMCFlatModel(model, functions))` |

On both routes [`OMBackend.simulate`](@ref) then runs the model by name.

### omc (experimental)

OMCompiler.jl loads omc's compiler library into Julia. It is not registered:
develop it from an OpenModelica checkout. It needs the install tree
(`install_cmake`) of a build with `-DOM_OMC_ENABLE_RUST=ON`, in
`OPENMODELICAHOME` or passed as `OMCompiler.init(omhome = ...)`.
`lowerOMCFlatModel` is not yet part of the [public API](api.md).

```julia
julia> import Pkg; Pkg.develop(path = "<OpenModelica>/OMCompiler/Compiler/OpenModelica.rs/OMCompiler.jl")
julia> import OMCompiler, OMBackend
julia> OMCompiler.init()  # the install tree in ENV["OPENMODELICAHOME"]
julia> OMCompiler.evaluate("loadString(\"model Decay Real x(start = 1, fixed = true); equation der(x) = -x; end Decay;\")")
julia> (model, functions) = OMCompiler.instantiate("Decay")
julia> OMBackend.translate(OMBackend.lowerOMCFlatModel(model, functions))
julia> OMBackend.simulate("Decay"; tspan = (0.0, 1.0))
```

## Where to read next

* [Getting started](getting-started.md): installation and an example.
* [API reference](api.md): translate, simulate, inspect and clear models.
* [OM.jl](https://github.com/JKRT/OM.jl) is the user-level package built on
  OMFrontend.jl and OMBackend.jl: `OM.simulate` runs the whole chain.
