```@meta
CurrentModule = OMBackend
```

# API reference

The functions below are the user-facing part of the package (`src/backendAPI.jl`).
The package has no `export` list for them: call them as `OMBackend.name`.

## Translate

```@docs
OMBackend.translate
OMBackend.generateTargetCode
OMBackend.knownValues
```

## Simulate

```@docs
OMBackend.simulateModel
OMBackend.resimulateModel
OMBackend.getMTKProblem
OMBackend.withTunableParameters
OMBackend.isTunable
OMBackend.eventCallbacks
OMBackend.pristineParameters
OMBackend.plot
```

## Inspect and write models

```@docs
OMBackend.availableModels
OMBackend.printModel
OMBackend.modelToString
OMBackend.writeModelToFile
OMBackend.dumpGeneratedCode
```

## Caches

```@docs
OMBackend.clearCaches!
OMBackend.forgetModel
```

## Index

```@index
Pages = ["api.md"]
```
