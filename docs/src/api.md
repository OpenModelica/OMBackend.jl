```@meta
CurrentModule = OMBackend
```

# API reference

The names on this page are OMBackend's public API: they are marked `public` and hold for all of 1.x.
Nothing is exported: call them as `OMBackend.name`. Every other name is internal and may change in any release.

## Translate

```@docs
OMBackend.translate
OMBackend.translateReducedArrays
OMBackend.generateTargetCode
OMBackend.knownValues
OMBackend.BackendMode
OMBackend.IMTK_MODE
OMBackend.MTK_MODE
OMBackend.EliminationOptions
OMBackend.explainIndexReduction
OMBackend.scalarizedFallbackReason
OMBackend.canonicalName
```

## Simulate

```@docs
OMBackend.simulateModel
OMBackend.resimulateModel
OMBackend.getMTKProblem
OMBackend.defaultSolver
OMBackend.daeFallbackSolver
OMBackend.getVariableValues
OMBackend.eventCallbacks
OMBackend.isEventCallback
OMBackend.isStepControl
OMBackend.ModelicaAssertionError
OMBackend.plot
```

## Tunable parameters

Parameters kept symbolic at translate time, set again at each simulate without a new translation. OMSurrogates
builds on these (parametric models, surrogates, neural blocks trained in the compiled model).

```@docs
OMBackend.withTunableParameters
OMBackend.isTunable
OMBackend.pristineParameters
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
