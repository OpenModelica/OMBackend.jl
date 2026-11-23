```@meta
CurrentModule = OMBackend
```

# API reference

The names on this page are OMBackend's public API: they are marked `public` and hold for all of 1.x.
Nothing is exported: call them as `OMBackend.name`. Every other name is internal and may change in any release.

## Translate

```@docs
OMBackend.translate
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
OMBackend.simulate
OMBackend.resimulate
OMBackend.getMTKProblem
OMBackend.defaultSolver
OMBackend.daeFallbackSolver
OMBackend.variableValues
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
OMBackend.buildId
```

## Inspect and write models

```@docs
OMBackend.availableModels
OMBackend.printModel
OMBackend.modelToString
OMBackend.writeModelToFile
```

## Caches

```@docs
OMBackend.clearCaches!
OMBackend.forgetModel
```

## Renamed in 1.0

The 0.x names still work in 1.x, with a deprecation warning (shown with `--depwarn=yes`).

| 0.x | 1.0 |
|:---|:---|
| `simulateModel(name; MODE)` | `simulate(name; mode)` |
| `resimulateModel(name; MODE)` | `resimulate(name; mode)` |
| `getVariableValues` | `variableValues` |
| `translate(fm; BackendMode, scalarized)` | `translate(fm; mode, scalarize)` |
| `translateReducedArrays(fm; BackendMode)` | `translate(fm; scalarize = false, mode)` |

`availableModels` returns a `Vector{String}` (0.x: one `String` with headings). `generateTargetCode`, `knownValues`
and `dumpGeneratedCode` are internal.

## Index

```@index
Pages = ["api.md"]
```
