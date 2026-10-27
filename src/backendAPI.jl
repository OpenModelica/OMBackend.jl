#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF AGPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GNU AGPL
* VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the OSMC (Open Source Modelica Consortium)
* Public License (OSMC-PL) are obtained from OSMC, either from the above
* address, from the URLs:
* http://www.openmodelica.org or
* https://github.com/OpenModelica/ or
* http://www.ida.liu.se/projects/OpenModelica,
* and in the OpenModelica distribution.
*
* GNU AGPL version 3 is obtained from:
* https://www.gnu.org/licenses/licenses.html#GPL
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#

using ExportAll
import MetaModelica
#= For interactive evaluation. =#
using ModelingToolkit
using SymbolicUtils
using DifferentialEquations
using OrdinaryDiffEq
import ADTypes
import OrdinaryDiffEqBDF

"""
    defaultSolver()

The solver used when none is given: Rodas5P, with central finite-difference
Jacobians. Rodas5's embedded error estimate misses a right-hand side that
depends on time explicitly: on x' = cos(39t) it took 10 steps at reltol 1e-8
and ended far from the solution, where Rodas5P is right (SciML recommends
Rodas5P over Rodas5). A mass-matrix system of a shape that Rosenbrock methods
handle badly switches to [`daeFallbackSolver`](@ref).

A Rosenbrock method (not a W-method) needs an accurate Jacobian for its order
and its error estimate. Where the model has no symbolic one, forward
differences were not accurate enough: MSL HeatingRectifier accepted a step
across a diode's conduction onset with 25 times the tolerance in its error
(Capacitor1.v 2.5 % off), where central differences, like ForwardDiff, give
the reference. Finite differences, not ForwardDiff: external C functions
take no dual numbers.
"""
defaultSolver() = Rodas5P(autodiff = ADTypes.AutoFiniteDiff(fdtype = Val(:central)))

"""
    daeFallbackSolver()

FBDF with finite-difference Jacobians: what the generated
`simulateFromBuild` uses instead of a Rosenbrock method for purely algebraic
systems and for algebraic rows of generated discrete variables.
"""
daeFallbackSolver() = FBDF(autodiff = ADTypes.AutoFiniteDiff())

import .CodeGeneration
import .Runtime
import .OMJLIR
import .LowerFlatModel
import .LowerFlatModel: FunctionList
import .Passes
import .SimCode

import Base.Meta
import JuliaFormatter
import OMFrontend
import Plots
import SCode

#= Settings =#
const WARN_MISSING_START_VALUES = Ref(false)

"""
Toggle direct RHS generation, bypassing MTK's ODEProblem constructor.
When enabled, the RHS function is built directly from symbolic equations
using Symbolics.build_function with CSE, resulting in much faster
compilation for large models.

Toggle with: `OMBackend.DIRECT_RHS_GENERATION[] = false` to disable.
"""
const DIRECT_RHS_GENERATION = Ref{Bool}(true)

"""
Toggle symbolic sparse Jacobian generation for direct-RHS problems.
When enabled, the Jacobian is built from the symbolic equations and attached
to the ODEFunction, so implicit solvers do not finite-difference one RHS
column per state every step. Falls back to finite differences automatically
for systems with non-differentiable (opaque) calls.

Toggle with: `OMBackend.DIRECT_JAC_GENERATION[] = false` to disable.
"""
const DIRECT_JAC_GENERATION = Ref{Bool}(true)

"""
The symbolic Jacobian (`DIRECT_JAC_GENERATION`) differentiates the RHS as a
DAG (`_dagSparseJacobian`). Where that has no rule, Symbolics'
sparsejacobian differentiates each equation as a tree instead, a shared
subexpression once per use, and does so only when no RHS equation has more
expression-tree nodes than this; otherwise the solver finite-differences. MSL
EngineV6_analytic's 17 equations are 1,943 DAG nodes but 14.6 million tree
nodes: its 17x17 Jacobian took Symbolics 48 s and 15 GB. Of the measured MSL
examples only EngineV6_analytic, Engine1b_analytic and RobotR3.oneAxis have
equations of more than 20,000 tree nodes (at least 45k); the others stay
below 7,500 in total.
"""
const DIRECT_JAC_TREE_NODE_LIMIT = Ref{Int}(20_000)

"""
Toggle type erasure of direct-RHS problems via `FunctionWrapperSpecialize`.
The generated RHS (and symbolic Jacobian) are runtime-generated functions
whose type is unique per model, so the resulting `ODEProblem` type differs
per model and `solve` re-specializes its whole stepping/Newton/linear-solve
machinery for every distinct model. Wrapping the inner functions in
`FunctionWrappers` makes the problem type constant across models, so that
machinery compiles once and is reused. Correctness is preserved: the mass
matrix, sparse Jacobian and initialization are attached to the `ODEFunction`
we build ourselves, not dropped by the wrapper.

Toggle with: `OMBackend.DIRECT_RHS_TYPE_ERASE[] = false` to disable.

This also governs event-callback erasure: when on, `_eraseContinuousCallbacks`
collapses the OM-generated continuous callbacks into a single, model-independent
`VectorContinuousCallback` (MTK `process_events` callbacks are left unchanged), so
event-ful models in the legacy when-equation class also share the problem type.
"""
const DIRECT_RHS_TYPE_ERASE = Ref{Bool}(true)

const OSMC_COPYRIGHT_HEADER = """
#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
* This file was generated automatically by OM.jl.
=#
"""

"""
  Toggle warnings for implicit default start values in MTK code generation.
"""
function warnMissingStartValues(enabled::Bool)
  WARN_MISSING_START_VALUES[] = enabled
end

@enum BackendMode begin
  MTK_MODE = 3
  #=
    Direct DifferentialEquations.jl emission. Builds an `ODEProblem` with an
    in-place RHS `f!(du, u, p, t)` that indexes `u`, `du`, and `p` by integer
    position. Bypasses ModelingToolkit entirely. Initial scope: pure ODE
    (no algebraic constraints), no VSS / structural transitions.
  =#
  DEMode = 4
  #=
    In-backend MTK path. Same module/codegen as MTK_MODE, but additionally
    constructs the System and runs structural_simplify in the backend at
    translate time and dumps the post-simplify System. Simulate reuses the
    existing module interface. Implementation in CodeGeneration/iMTKGen.jl.
  =#
  IMTK_MODE = 5
end

"""
Runtime-toggleable default backend mode used by `OM.translate` / `OM.simulate`
and `OMBackend.translate` / `OMBackend.simulateModel`. Defaults to `IMTK_MODE`
(cached build fast path). Flip to revert to the lazy-build MTK path:

    OMBackend.DEFAULT_BACKEND_MODE[] = OMBackend.MTK_MODE

Read at each call site, so changes take effect immediately without restarting Julia.
"""
const DEFAULT_BACKEND_MODE = Ref{BackendMode}(IMTK_MODE)

"""
  MTK Models that have been compiled one time.
TODO: Optimally we should keep the frontend structure
in memory as well s.t we only recompile if the structure of the source file changes
(Unless retranslation is forced).
"""
const COMPILED_MODELS_MTK = Dict{String, Tuple{Expr, Bool, UInt64}}()


"""
DEJL (DifferentialEquations.jl) models that have been compiled one time.
"""
const COMPILED_MODELS_DEJL = Dict{String, Tuple{Expr, Bool, UInt64}}()

"How many times each model's code was dumped in this process (`dumpGeneratedCode`)."
const CODE_DUMP_COUNT = Dict{String, Int}()
const CODE_DUMP_LOCK = ReentrantLock()

"""
    dumpGeneratedCode(modelName, code)

With `OMBACKEND_CODE_DUMP_DIR` set, write the generated module to
`<dir>/<modelName>-<n>.jl`, `n` counting the translations of the model in this
process: the reference corpus for comparing the code generation before and after
a backend change.
"""
function dumpGeneratedCode(modelName::String, code::Expr)::Nothing
  local dir = get(ENV, "OMBACKEND_CODE_DUMP_DIR", "")
  isempty(dir) && return nothing
  lock(CODE_DUMP_LOCK) do
    local n = CODE_DUMP_COUNT[modelName] = get(CODE_DUMP_COUNT, modelName, 0) + 1
    mkpath(dir)
    write(joinpath(dir, "$(modelName)-$(n).jl"), string(code))
  end
  return nothing
end

"""
Array-preserving code generation for flat models translated with `scalarized = false`
(experimental): an explicit ODE whose equations are `der(x) = e`, `der(x[i]) = e` or for-loops
of these gets a DifferentialEquations.jl module that keeps the loops (CodeGeneration.ArrayODEGen);
other models are scalarized at the backend entry as before. Toggle with
`OMBackend.ARRAY_ODE_GENERATION[] = false`.
"""
const ARRAY_ODE_GENERATION = Ref{Bool}(true)

"""
`true`: the array-preserving code generation also takes models with events (relations,
when, sample, initial()), asserts and initial equations (experimental). `false` (the default):
those go the ModelingToolkit path, whose event and initialization semantics they keep.
"""
const ARRAY_PATH_FULL = CodeGeneration.ArrayODEGen.FULL

#= The models whose current translation is an array ODE module (simulated as DE-mode modules). =#
const ARRAY_ODE_MODELS = Set{String}()

"""
The models (canonical names) whose index the last translate reduced in array form, their
arrays kept until the simulation code (`arrayReducedModel`, OMBACKEND_ARRAY_INDEX_REDUCTION).
"""
const ARRAY_REDUCED_MODELS = Set{String}()

"""
Why the arrays-kept paths did not take a model at its last translate (canonical name => path => reason): what OM's
translate reports when it falls back to the scalarizing flatten (`scalarizedFallbackReason`).
"""
const ARRAY_PATH_REASONS = Dict{String, Dict{String, String}}()

"The arrays-kept path `path` (array code generation, array index reduction) did not take the model `name`, for `reason`."
function declineArrayPath!(name::String, path::String, reason::String)::Nothing
  get!(Dict{String, String}, ARRAY_PATH_REASONS, name)[path] = reason
  return nothing
end

"""
    scalarizedFallbackReason(name) -> String

Why the model `name` (canonical) goes through the scalarizing flatten: the reasons the arrays-kept
paths gave at its last translate (array code generation, then array index reduction).
"""
function scalarizedFallbackReason(name::String)::String
  local reasons = get(ARRAY_PATH_REASONS, name, Dict{String, String}())
  local parts::Vector{String} = String["$(path): $(reasons[path])" for path in ("array code generation", "array index reduction", "array code after index reduction")
                        if haskey(reasons, path)]
  return isempty(parts) ? "no reason recorded" : join(parts, "; ")
end

# Per-model log-run directory captured at translate (`lower`) time. Looked up
# at simulate time by `simulateModel` so any dumps emitted during the
# post-MTK / `buildDirectRHSProblem` / `_solveDAEInitialization!` codegen
# paths land in the same per-model run directory as the OMJLIR/simCode logs,
# not the session root. Keyed by canonical model name.
const MODEL_RUN_DIRS = Dict{String, String}()

function logRunModelName(frontendDAE::OMFrontend.Frontend.FlatModel)::String
  return string(frontendDAE.name)
end

"""
    clearCaches!(; models=true, implementations=true, wrappers=true, extractors=true)

Clear persistent caches. By default all caches are cleared.
Use keyword arguments to selectively clear individual caches.

- `models`: compiled MTK model ASTs, the cached builds and their external objects (tables),
  which their destructors free
- `implementations`: Modelica function implementations
- `wrappers`: RTG wrapper functions for symbolic dispatch
- `extractors`: per-element array extractor functions
"""
function clearCaches!(; models::Bool=true,
                        implementations::Bool=true,
                        wrappers::Bool=true,
                        extractors::Bool=true)
  cleared = String[]
  models          && (CodeGeneration.freeAllExternalObjects!();
                      empty!(COMPILED_MODELS_MTK);
                      empty!(IMTKGen.BUILT); empty!(IMTKGen.BUILT_HASH);
                      empty!(IMTKGen.REDUCED_SYSTEMS); empty!(IMTKGen.PRISTINE_P);
                      empty!(CodeGeneration.DAE_REINIT);
                      Cache.clear!();
                      push!(cleared, "models"))
  implementations && (empty!(CodeGeneration.MODELICA_FUNCTION_IMPLS); empty!(CodeGeneration.MODELICA_FUNCTION_OWNERS);
                      empty!(CodeGeneration.MODEL_FUNCTIONS); empty!(CodeGeneration.COMPILED_CODE_SIZES);
                      push!(cleared, "implementations"))
  wrappers        && (empty!(CodeGeneration.MODELICA_FUNCTION_WRAPPERS); push!(cleared, "wrappers"))
  extractors      && (empty!(CodeGeneration.ELEM_FUNC_CACHE); empty!(CodeGeneration.TUPLE_ELEM_FUNC_CACHE);
                      empty!(CodeGeneration.EXTRACTOR_INFO); empty!(CodeGeneration.CALL_RGFS); push!(cleared, "extractors"))
  return cleared
end

"""
    forgetModel(modelName::String) -> Nothing

Drops the cached build of one model and frees its external objects (tables) by their
destructors, with its generated code and its run directory. For a long
run over many models (a coverage worker): without it every model's build, tables and code stay
for the whole session. A solution of the model that reads one of its tables afterwards throws;
simulate the model again for a new one (a translate).
"""
function forgetModel(modelName::String)::Nothing
  local cname = canonicalName(modelName)
  IMTKGen.forgetBuild(cname)
  Cache.remove!(cname)
  for name in unique((modelName, cname))
    delete!(COMPILED_MODELS_MTK, name)
    delete!(COMPILED_MODELS_DEJL, name)
    #= else a simulate by name took it for a translated array model and found no code (MSL SignConvention after
       a coverage worker forgot it: "DE-mode model ... is not compiled") =#
    delete!(ARRAY_ODE_MODELS, name)
    delete!(ARRAY_REDUCED_MODELS, name)
    delete!(MODEL_RUN_DIRS, name)
  end
  return nothing
end

"""
    translate(fm; functionList = nothing, BackendMode = DEFAULT_BACKEND_MODE[], warnMissingStartValues = nothing,
              eliminateNonDynamic = nothing, observedFilter = nothing, checkSimCode = true, scalarized = true,
              omjlir = nothing)

The target code of the flat model `fm` and its functions (the frontend's function list): `fm`
lowered to OMJLIR and through its passes (`omjlirModel`; `omjlir`: that OMJLIR model, already
lowered, `translateReducedArrays`), to the simulation code and its
passes (SimCode.simulationCode; SimCode.deSimulationCode for DEMode), checked
(`checkSimCode`), and the code generated for the backend mode. A flat model with arrays kept
(`scalarized = false`) goes to the array-preserving generation first (ARRAY_ODE_GENERATION), to
this path where it is outside its scope.
"""
function translate(fm::OMFrontend.Frontend.FlatModel;
                   functionList::Union{Nothing, FunctionList} = nothing,
                   BackendMode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[],
                   warnMissingStartValues::Union{Nothing, Bool} = nothing,
                   eliminateNonDynamic::Union{Nothing, Bool, SimCode.EliminationOptions} = nothing,
                   observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                   checkSimCode::Bool = true,
                   scalarized::Bool = true,
                   omjlir::Union{Nothing, OMJLIR.Model} = nothing)
  local previousWarnSetting = WARN_MISSING_START_VALUES[]
  local modelName = logRunModelName(fm)
  local runId = createLogRunId(modelName)
  MODEL_RUN_DIRS[canonicalName(modelName)] = runId
  resetFallbacks!()
  warnMissingStartValues === nothing || (WARN_MISSING_START_VALUES[] = warnMissingStartValues)
  try
    delete!(ARRAY_ODE_MODELS, canonicalName(fm.name))
    omjlir === nothing && delete!(ARRAY_REDUCED_MODELS, canonicalName(fm.name))
    if !scalarized && ARRAY_ODE_GENERATION[]
      local arrayResult = @BACKEND_PERFLOG "[backendAPI] generateArrayODECode" generateArrayODETargetCode(fm; functionList)
      arrayResult === nothing || return arrayResult
    end
    #= the index reduced in array form: its array code, else its scalarized form =#
    local arrayReduction::Bool = !scalarized && omjlir === nothing
    if arrayReduction
      local reduced::Union{Nothing, OMJLIR.Model} = @BACKEND_PERFLOG "[backendAPI] reducedArrayModel" reducedArrayModel(fm; functionList)
      if reduced !== nothing
        local reducedResult::Union{Nothing, Tuple{String, Expr}} = @BACKEND_PERFLOG "[backendAPI] generateReducedArrayODECode" generateReducedArrayODETargetCode(fm, reduced; functionList)
        reducedResult === nothing || return reducedResult
        omjlir = scalarizedReducedModel(fm, reduced)
      end
    end
    return withLogRunDir(runId) do
      #= The flat model with its functions as it arrives from the frontend =#
      @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "frontend_initialModel.log"),
        replace(OMFrontend.Frontend.toFlatString(fm, functionList === nothing ? MetaModelica.nil : functionList), "\\n" => "\n"))
      local model::OMJLIR.Model = omjlir !== nothing ? omjlir :
        @BACKEND_PERFLOG "[backendAPI] lower" omjlirModel(fm; functionList, scalarized, arrayReduction = !arrayReduction)
      if BackendMode == DEMode
        #= Direct DifferentialEquations.jl path: a subset of the MTK path's SimCode passes (not,
           e.g., the Complex lowering, the function evaluation, the folds, the initial values),
           no observed-equation work, the DE emitter. =#
        local simCode = SimCode.deSimulationCode(model)
        (SimCode.hasStructuralTransitions(simCode) || SimCode.hasSubModels(simCode)) &&
          error("DEMode does not support structural transitions / VSS at this time. Re-run with mode = OMBackend.MTK_MODE.")
        _checkSimCodeBeforeCodegen(simCode, checkSimCode)
        return _translationResult(generateDETargetCode(simCode))
      elseif BackendMode == MTK_MODE || BackendMode == IMTK_MODE
        local simCode = mtkSimCode(model; eliminateNonDynamic, observedFilter)
        _checkSimCodeBeforeCodegen(simCode, checkSimCode)
        local genCode = if BackendMode == IMTK_MODE
          @BACKEND_PERFLOG "[backendAPI] generateIMTKTargetCode" generateIMTKTargetCode(simCode)
        else
          @BACKEND_PERFLOG "[backendAPI] generateMTKTargetCode" generateMTKTargetCode(simCode)
        end
        return _translationResult(genCode)
      end
      error("Unsupported BackendMode: $BackendMode. Valid modes are: MTK_MODE, IMTK_MODE, DEMode")
    end
  finally
    WARN_MISSING_START_VALUES[] = previousWarnSetting
  end
end

"""
    knownValues(model) -> Dict{Symbol, OMJLIR.Expression}

The algebraic variables of `model` the simulation code's first passes fold to a literal (a
residual `v - c`, eliminated or not yet; a record function's output its interpretation finds constant), by
their OMJLIR names: what index reduction plans with (`Passes.IndexReduction.KNOWN_VALUES`).
SimCode of a copy of the model through the evaluation of constant function outputs
(`SimCode.simulationCodeThroughFunctionOutputs`); empty where that fails.
"""
function knownValues(model::OMJLIR.Model)::Dict{Symbol, OMJLIR.Expression}
  local out = Dict{Symbol, OMJLIR.Expression}()
  local sc = try
    redirect_stdout(devnull) do
      #= a copy of what the passes change (variables, partitions); the frontend model shared =#
      SimCode.simulationCodeThroughFunctionOutputs(OMJLIR.Model(model.name, deepcopy(model.variables), deepcopy(model.continuous),
                                          deepcopy(model.clocked), model.baseClocks, model.functions,
                                          model.initialMode, model.modes, model.transitions, model.frontendModel))
    end
  catch err
    _fallback(err, :knownValues)
    return out
  end
  #= OMJLIR's names by SimCode's (dots as underscores); a name two variables would share is left out =#
  local byName = Dict{String, Symbol}()
  local shared = Set{String}()
  for v in model.variables
    v.kind isa OMJLIR.ALGEBRAIC || continue
    local n = replace(String(v.name), "." => "_")
    haskey(byName, n) ? push!(shared, n) : (byName[n] = v.name)
  end
  #= `v - c`: eliminated, or a residual a later pass would eliminate =#
  for eq in Iterators.flatten((sc.eliminatedEquations, sc.residualEquations))
    local e = eq.exp
    (e isa SimCode.BINARY && e.op === SimCode.OP_SUB && e.exp1 isa SimCode.EXP_CREF && e.exp2 isa SimCode.RCONST) || continue
    local n = SimCode.crefName(e.exp1.cref)
    (haskey(byName, n) && !(n in shared)) || continue
    out[byName[n]] = OMJLIR.REAL(e.exp2.value)
  end
  return out
end

Passes.IndexReduction.KNOWN_VALUES[] = knownValues

"""
    explainIndexReduction(fm; functionList = nothing) -> Passes.IndexReduction.Explanation

What OMBackend's index reduction does to the flat model `fm` and why
(`Passes.IndexReduction.explain`): the equations it differentiates and how often, the candidates
of each choice with their StateSelect and the dummy derivatives among them, the states, the
dynamic state sets; or why the model needs no reduction or is left to MTK. Planned as a
translation would plan it, nothing translated.
"""
function explainIndexReduction(fm::OMFrontend.Frontend.FlatModel;
                               functionList::Union{Nothing, FunctionList} = nothing)::Passes.IndexReduction.Explanation
  local model = LowerFlatModel.lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  #= the passes but index reduction (it runs after the last one) =#
  model = Passes.runPasses!(model; through = first(last(Passes.PIPELINE)))
  return Passes.IndexReduction.explain(model)
end

"""
    omjlirModel(fm; functionList = nothing, scalarized = true, arrayReduction = true) -> OMJLIR.Model

The OMJLIR model of the flat model `fm` and its functions (the frontend's function list),
through the passes (Passes.PIPELINE): what the simulation code is lowered from. A flat model
with arrays kept (`scalarized = false`, OM's `scalarize = false`) whose index OMBackend reduces in
array form is lowered with its arrays (`arrayReducedModel`; not with `arrayReduction` false: a caller
that tried it); another is scalarized first: the backend takes scalars.
"""
function omjlirModel(fm::OMFrontend.Frontend.FlatModel; functionList::Union{Nothing, FunctionList} = nothing,
                     scalarized::Bool = true, arrayReduction::Bool = true)::OMJLIR.Model
  if !scalarized && arrayReduction
    local reduced = arrayReducedModel(fm; functionList)
    reduced === nothing || return reduced
  end
  scalarized || (fm = @BACKEND_PERFLOG "[backendAPI] scalarizeKeptArrays" OMFrontend.Frontend.scalarizeKeptArrays(fm))
  local model = LowerFlatModel.lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  return Passes.runPasses!(model)
end

"""
    arrayReducedModel(fm; functionList = nothing) -> Union{Nothing, OMJLIR.Model}

The OMJLIR model of a flat model with arrays kept whose index OMBackend reduces in array form (the
switch OMBACKEND_ARRAY_INDEX_REDUCTION, with OMBACKEND_INDEX_REDUCTION), for the scalar simulation
code: reduced with its arrays (`reducedArrayModel`), then scalarized (`scalarizedReducedModel`).
`nothing` where the array form does not take the model: the caller scalarizes in the frontend as
before.
"""
function arrayReducedModel(fm::OMFrontend.Frontend.FlatModel;
                           functionList::Union{Nothing, FunctionList} = nothing)::Union{Nothing, OMJLIR.Model}
  local reduced::Union{Nothing, OMJLIR.Model} = reducedArrayModel(fm; functionList)
  return reduced === nothing ? nothing : scalarizedReducedModel(fm, reduced)
end

"""
    reducedArrayModel(fm; functionList = nothing) -> Union{Nothing, OMJLIR.Model}

A flat model with arrays kept lowered with its arrays, through the passes (`runPassesKeepingArrays!`:
its array equations as for-equations), its index reduced in array form
(`Passes.IndexReduction.reduceIndexArrays!`): what the array code (`generateReducedArrayODETargetCode`)
or the scalarized form (`scalarizedReducedModel`) is made from. `nothing` where the array form does
not take the model (switched off, no for-equations or array equations over continuous arrays, index
one, what it does not handle or the lowering does not support: logged; another error: a warning),
the reason recorded (`declineArrayPath!`).
"""
function reducedArrayModel(fm::OMFrontend.Frontend.FlatModel;
                           functionList::Union{Nothing, FunctionList} = nothing)::Union{Nothing, OMJLIR.Model}
  delete!(ARRAY_REDUCED_MODELS, canonicalName(fm.name))
  local decline::Function = (reason::String) -> declineArrayPath!(canonicalName(fm.name), "array index reduction", reason)
  if !(envSwitch("OMBACKEND_ARRAY_INDEX_REDUCTION") && envSwitch("OMBACKEND_INDEX_REDUCTION"))
    decline("switched off (OMBACKEND_ARRAY_INDEX_REDUCTION, OMBACKEND_INDEX_REDUCTION)")
    return nothing
  end
  local reduction::Module = Passes.IndexReduction
  return _arrayReductionStep(fm, decline) do
    local lowered::OMJLIR.Model = LowerFlatModel.lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
    #= a model without for-equations or array equations over continuous arrays reduces as the scalarized one: the
       passes not run for it; unless its separately declared components are rolled into arrays (OMBACKEND_ROLL_COMPONENTS) =#
    local roll::Bool = envSwitch("OMBACKEND_ROLL_COMPONENTS") && !isempty(reduction.componentGroups(lowered.variables))
    if !roll && !(reduction.hasContinuousArrays(lowered) && reduction.hasArrayEquations(lowered))
      decline("no for-equations or array equations over continuous arrays")
      return nothing
    end
    local model::OMJLIR.Model = Passes.runPassesKeepingArrays!(lowered)
    if roll && reduction.rollComponents!(model) == 0 && !(reduction.hasContinuousArrays(model) && reduction.hasArrayEquations(model))
      decline("no for-equations or array equations over continuous arrays, its components not rolled")
      return nothing
    end
    if !reduction.reduceIndexArrays!(model)
      decline("index 1, nothing to reduce")
      return nothing
    end
    model
  end
end

"""
    scalarizedReducedModel(fm, reduced) -> Union{Nothing, OMJLIR.Model}

The model reduced in array form (`reducedArrayModel`; not changed) scalarized for the scalar
simulation code (`scalarizeArrays`), its derivative unknowns solved for (`solveDerivatives!`,
OMBACKEND_SOLVE_DERIVATIVES); recorded in ARRAY_REDUCED_MODELS. `nothing` where that fails (the
reason recorded as `reducedArrayModel`'s).
"""
function scalarizedReducedModel(fm::OMFrontend.Frontend.FlatModel, reduced::OMJLIR.Model)::Union{Nothing, OMJLIR.Model}
  local decline::Function = (reason::String) -> declineArrayPath!(canonicalName(fm.name), "array index reduction", reason)
  local reduction::Module = Passes.IndexReduction
  local scalar::Union{Nothing, OMJLIR.Model} = _arrayReductionStep(fm, decline) do
    local s::OMJLIR.Model = reduction.scalarizeArrays(reduced).model
    envSwitch("OMBACKEND_SOLVE_DERIVATIVES") && reduction.solveDerivatives!(s, s.continuous[1])
    s
  end
  scalar === nothing || push!(ARRAY_REDUCED_MODELS, canonicalName(fm.name))
  return scalar
end

"""
`f()`, a step of the array index reduction of `fm`: a model it does not handle (NotHandled, UnsupportedLowering)
logged and declined (`decline`), another error (the optional route fails where it should refuse) a warning and
declined; `nothing` then.
"""
function _arrayReductionStep(f::Function, fm::OMFrontend.Frontend.FlatModel, decline::Function)::Union{Nothing, OMJLIR.Model}
  local reduction::Module = Passes.IndexReduction
  try
    return f()
  catch err
    isFatal(err) && rethrow()
    if err isa Union{reduction.NotHandled, UnsupportedLowering}
      @info "[index reduction] $(canonicalName(fm.name)): arrays kept, $(sprint(showerror, err)); scalarized in the frontend"
      decline(sprint(showerror, err))
    else
      @warn "[index reduction] $(canonicalName(fm.name)): arrays kept, an error ($(sprint(showerror, err))); scalarized in the frontend"
      decline("an error: $(sprint(showerror, err))")
    end
    return nothing
  end
end

"""
    generateReducedArrayODETargetCode(fm, reduced; functionList = nothing) -> Union{Nothing, Tuple{String, Expr}}

The array ODE module of a flat model whose index was reduced in array form (`reduced`,
`reducedArrayModel`; not changed): its derivative unknowns solved in array form
(`Passes.IndexReduction.solvedInArrays`), the OMJLIR front half of ArrayODEGen; cached as a DE-mode
model (`generateArrayODETargetCode`'s), recorded in ARRAY_ODE_MODELS and ARRAY_REDUCED_MODELS.
`nothing` where it is outside the array code's scope (ARRAY_ODE_GENERATION off, tunable parameters,
an algebraic loop left, ...): the reason logged and recorded (`declineArrayPath!`, "array code after
index reduction"), and the caller scalarizes the reduced model.
"""
function generateReducedArrayODETargetCode(fm::OMFrontend.Frontend.FlatModel, reduced::OMJLIR.Model;
                                           functionList::Union{Nothing, FunctionList} = nothing)::Union{Nothing, Tuple{String, Expr}}
  local name::String = canonicalName(fm.name)
  local decline::Function = (reason::String) -> begin
    @info "[backendAPI] array code after index reduction not taken for $(name): $(reason)"
    declineArrayPath!(name, "array code after index reduction", reason)
    nothing
  end
  ARRAY_ODE_GENERATION[] || return decline("switched off (OMBackend.ARRAY_ODE_GENERATION)")
  isempty(TUNABLE_PARAMETERS[]) || return decline("withTunableParameters (the ModelingToolkit path's tunable sets)")
  local reduction::Module = Passes.IndexReduction
  local (modelName::Union{Nothing, String}, modelCode::Union{String, Expr}) = try
    #= solvedInArrays splits arrays in place (splitArrays!), a split that throws part-way leaves its model half split:
       a copy, so the scalarized fallback reads `reduced` as reduced =#
    local working::OMJLIR.Model = OMJLIR.Model(reduced.name, deepcopy(reduced.variables), deepcopy(reduced.continuous),
                                               reduced.clocked, reduced.baseClocks, reduced.functions, reduced.initialMode,
                                               reduced.modes, reduced.transitions, reduced.frontendModel)
    local (solved::OMJLIR.Model, _) = @BACKEND_PERFLOG "[backendAPI] solvedInArrays" reduction.solvedInArrays(working)
    @BACKEND_PERFLOG "[backendAPI] array code (after index reduction)" CodeGeneration.ArrayODEGen.generateArrayODECode(solved, name; functions = functionList)
  catch err
    isFatal(err) && rethrow()
    err isa reduction.NotHandled && return decline(sprint(showerror, err))
    @warn "[backendAPI] array code after index reduction for $(name): an error ($(first(sprint(showerror, err), 300)))"
    return decline("an error: $(first(sprint(showerror, err), 300))")
  end
  modelName === nothing && return decline(string(modelCode))
  _registerArrayODEModule!(name, modelCode)
  push!(ARRAY_REDUCED_MODELS, name)
  return (name, modelCode)
end

"""
    translateReducedArrays(fm; functionList, BackendMode, warnMissingStartValues, eliminateNonDynamic,
                           observedFilter, checkSimCode) -> Union{Nothing, Tuple}

`translate` of a flat model with arrays kept whose index OMBackend reduces in array form
(`arrayReducedModel`); `nothing` where it does not (OM's translate then flattens with
scalarization).
"""
function translateReducedArrays(fm::OMFrontend.Frontend.FlatModel;
                                functionList::Union{Nothing, FunctionList} = nothing,
                                BackendMode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[],
                                warnMissingStartValues::Union{Nothing, Bool} = nothing,
                                eliminateNonDynamic::Union{Nothing, Bool, SimCode.EliminationOptions} = nothing,
                                observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                                checkSimCode::Bool = true)
  local reduced::Union{Nothing, OMJLIR.Model} = @BACKEND_PERFLOG "[backendAPI] reducedArrayModel" reducedArrayModel(fm; functionList)
  reduced === nothing && return nothing
  local arrayCode::Union{Nothing, Tuple{String, Expr}} =
    @BACKEND_PERFLOG "[backendAPI] generateReducedArrayODECode" generateReducedArrayODETargetCode(fm, reduced; functionList)
  arrayCode === nothing || return arrayCode
  local model::Union{Nothing, OMJLIR.Model} = scalarizedReducedModel(fm, reduced)
  model === nothing && return nothing
  local name = canonicalName(fm.name)
  #= The array form only where it translates and builds; otherwise its build is forgotten and OM's translate
     scalarizes as before. Coverage 21 (the array form's first): a lifted if-expression left a for-equation's
     iterator free (UndefVarError), a build failed (BoundsError) and the tables it had allocated were freed
     under the problem that read them (Rotational Friction, CoupledClutches, Translational Brake). =#
  local decline = (reason::String) -> begin
    @info "[index reduction] $(name): arrays kept, the reduced model $(reason); scalarized in the frontend"
    declineArrayPath!(name, "array index reduction", "the reduced model " * reason)
    IMTKGen.forgetBuild(name)
    delete!(COMPILED_MODELS_MTK, name)
    delete!(ARRAY_REDUCED_MODELS, name)
    nothing
  end
  local result = try
    translate(fm; functionList, BackendMode, warnMissingStartValues, eliminateNonDynamic, observedFilter, checkSimCode,
              omjlir = model)
  catch err
    isFatal(err) && rethrow()
    return decline("does not translate ($(first(sprint(showerror, err), 200)))")
  end
  (BackendMode == IMTK_MODE && !haskey(IMTKGen.BUILT, name)) && return decline("does not build")
  return result
end

"""
    mtkSimCode(model; eliminateNonDynamic = nothing, observedFilter = nothing) -> SimCode.SIM_CODE

The simulation code the MTK and IMTK paths generate code from: the OMJLIR model `model` through
the SimCode passes (SimCode.simulationCode). A translate's, and that of a model recompiled
during its simulation (Runtime.translateToSimCode).
"""
mtkSimCode(model::OMJLIR.Model; eliminateNonDynamic::Union{Nothing, Bool, SimCode.EliminationOptions} = nothing,
           observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing)::SimCode.SIM_CODE =
  SimCode.simulationCode(model; eliminateNonDynamic,
                         observedFilter = observedFilter === nothing ? nothing : _observedFilterStrings(observedFilter))

function _translationResult(result::Tuple{String, Expr})::Tuple{String, Expr}
  dumpGeneratedCode(result[1], result[2])
  return result
end

"""
The simulation code checked before the code generation (SimCode.check): a reference that
resolves to no variable, eliminated variable or builtin aborts it with the located reason (the
code generation would otherwise throw a bare UndefVarError deep in MTK); the other violations
are reported.
"""
function _checkSimCodeBeforeCodegen(simCode::SimCode.SIM_CODE, checkSimCode::Bool)::Nothing
  checkSimCode || return nothing
  local result = SimCode.check(simCode)
  local blocking = filter(v -> v.severity === :error && v.rule === :cref_resolution, result.violations)
  if !isempty(blocking)
    local details = join([string("  - ", v.where, ": ", v.detail) for v in blocking], "\n")
    throw(SimCode.SimCodeCheckError(
      string("Code generation aborted for ", simCode.name, ": ", length(blocking),
             " unresolved component reference(s). These references do not resolve ",
             "against the simulation variable table, the eliminated-variable set, or ",
             "the builtin set, so code generation would raise an undefined-variable error.\n", details)))
  end
  SimCode.report(stderr, result; modelName = simCode.name)
  return nothing
end

#= Final SimCode of the most recent codegen, for interactive inspection. =#
const LAST_SIM_CODE = Ref{Any}(nothing)

"""
`generateMTKTargetCode(simCode::SimCode.SIM_CODE)`
  Generates code interfacing ModelingToolkit.jl
  The resulting code is saved in a table which contains functions that were simulated
  this session. Returns the generated modelName and corresponding generated code
"""
function generateMTKTargetCode(simCode::SimCode.SIM_CODE)::Tuple{String, Expr}
  LAST_SIM_CODE[] = simCode
  #= Target code =#
  (modelName::String, modelCode::Expr) = CodeGeneration.generateMTKCode(simCode)
  #= An IMTK build of this model from an earlier translate is not this one. =#
  IMTKGen.forgetBuild(canonicalName(modelName))
  local codeHash = hash(modelCode)
  @info "[MTK GEN] generated target code" model=modelName codeHash=codeHash
  if haskey(COMPILED_MODELS_MTK, modelName)
    #= Compare hash instead of full Expr tree to reduce compile-time overhead. =#
    local previousHash = COMPILED_MODELS_MTK[modelName][3]
    local changeDetected = previousHash != codeHash
    COMPILED_MODELS_MTK[modelName] = (modelCode, changeDetected, codeHash)
  else
    #= If the module already exists from a previous compilation (e.g. after
       clearCaches!), mark as needing re-eval so simulateModel picks up the
       new code instead of reusing the stale module. =#
    local moduleAlreadyExists = _modelDefined(Symbol(modelName))
    COMPILED_MODELS_MTK[modelName] = (modelCode, moduleAlreadyExists, codeHash)
  end
  Cache.store!(canonicalName(modelName), Cache.sizeOf(modelCode))
  return (modelName, modelCode)
end

"""
`generateIMTKTargetCode(simCode)` — like `generateMTKTargetCode`, but via
`IMTKGen.generateIMTKCode`, which also builds + structurally simplifies the
System in the backend and dumps it. Caches into `COMPILED_MODELS_MTK`.
"""
function generateIMTKTargetCode(simCode::SimCode.SIM_CODE)::Tuple{String, Expr}
  LAST_SIM_CODE[] = simCode
  #= IMTKGen.generateIMTKCode evals the (current) module while building, so the
     loaded module is already fresh: record changeDetected = false so a later
     MTK-mode simulate on this model does not needlessly re-eval the module. A
     structural model's module is evaluated at its first simulate (simulateIMTK):
     changed until then. =#
  (modelName::String, modelCode::Expr) = IMTKGen.generateIMTKCode(simCode)
  local codeHash = hash(modelCode)
  @info "[IMTK GEN] generated target code" model=modelName codeHash=codeHash
  local structural = SimCode.hasStructuralTransitions(simCode) || SimCode.hasSubModels(simCode)
  COMPILED_MODELS_MTK[modelName] = (modelCode, structural, codeHash)
  local cname = canonicalName(modelName)
  Cache.store!(cname, Cache.sizeOf((modelCode, get(IMTKGen.BUILT, cname, nothing), get(IMTKGen.PRISTINE_P, cname, nothing))))
  return (modelName, modelCode)
end

"""
The generated code of a model for writing it out: the array path's module for an array
model, else the MTK path's code.
"""
_writtenModel(modelName) = modelName in ARRAY_ODE_MODELS && haskey(COMPILED_MODELS_DEJL, modelName) ?
  COMPILED_MODELS_DEJL[modelName][1] : getCompiledModel(modelName)

function getCompiledModel(modelName)
  haskey(COMPILED_MODELS_MTK, modelName) ||
    error("Model $(modelName) is not compiled (OMBackend.translate first). $(availableModels())")
  return COMPILED_MODELS_MTK[modelName][1]
end

"""
`generateDETargetCode(simCode::SimCode.SIM_CODE)`
  Generates code interfacing DifferentialEquations.jl directly (DEMode).
  Produces an in-place ODEProblem with the RHS indexed by integer position.
  Caches the result in COMPILED_MODELS_DEJL so simulateModel can pick it up.
"""
function generateDETargetCode(simCode::SimCode.SIM_CODE)::Tuple{String, Expr}
  (modelName::String, modelCode::Expr) = CodeGeneration.DEGen.generateDECode(simCode)
  local codeHash = hash(modelCode)
  if haskey(COMPILED_MODELS_DEJL, modelName)
    local previousHash = COMPILED_MODELS_DEJL[modelName][3]
    local changeDetected = previousHash != codeHash
    COMPILED_MODELS_DEJL[modelName] = (modelCode, changeDetected, codeHash)
  else
    local moduleAlreadyExists = _modelDefined(Symbol(modelName))
    COMPILED_MODELS_DEJL[modelName] = (modelCode, moduleAlreadyExists, codeHash)
  end
  Cache.store!(canonicalName(modelName), Cache.sizeOf(modelCode))
  return (modelName, modelCode)
end

"""
    translateArrays(flatModel) -> (modelName, code) or nothing

The array-preserving translation of a flat model with arrays kept (from the frontend's
`scalarize = false`), or `nothing` when the model is outside its scope (the reason is
logged). OM falls back to the scalarizing flatten and `translate` then.
"""
function translateArrays(fm::OMFrontend.Frontend.FlatModel; functionList::Union{Nothing, FunctionList} = nothing)
  delete!(ARRAY_ODE_MODELS, canonicalName(fm.name))
  delete!(ARRAY_REDUCED_MODELS, canonicalName(fm.name))
  delete!(ARRAY_PATH_REASONS, canonicalName(fm.name))
  if !ARRAY_ODE_GENERATION[]
    declineArrayPath!(canonicalName(fm.name), "array code generation", "switched off (OMBackend.ARRAY_ODE_GENERATION)")
    return nothing
  end
  if !isempty(TUNABLE_PARAMETERS[])
    @info "[backendAPI] array ODE path not taken for $(canonicalName(fm.name)): withTunableParameters (the ModelingToolkit path's tunable sets)"
    declineArrayPath!(canonicalName(fm.name), "array code generation", "withTunableParameters (the ModelingToolkit path's tunable sets)")
    return nothing
  end
  return @BACKEND_PERFLOG "[backendAPI] generateArrayODECode" generateArrayODETargetCode(fm; functionList)
end

"""
The array ODE module of a no-scalarize flat model (ARRAY_ODE_GENERATION), cached as a
DE-mode model; nothing (and an @info with the reason) for a model outside its scope.
"""
function generateArrayODETargetCode(fm::OMFrontend.Frontend.FlatModel; functionList::Union{Nothing, FunctionList} = nothing)
  local modelName = canonicalName(fm.name)
  local (name, modelCode) = CodeGeneration.ArrayODEGen.generateArrayODECode(fm, modelName; functions = functionList)
  if name === nothing
    @info "[backendAPI] array ODE path not taken for $(modelName): $(modelCode)"
    declineArrayPath!(modelName, "array code generation", string(modelCode))
    return nothing
  end
  _registerArrayODEModule!(modelName, modelCode)
  return (modelName, modelCode)
end

"The array ODE module `modelCode` of the model `modelName` cached as a DE-mode model, recorded in ARRAY_ODE_MODELS."
function _registerArrayODEModule!(modelName::String, modelCode::Expr)::Nothing
  dumpGeneratedCode(modelName, modelCode)
  #= evaluated again at the next simulate: a module of this name may be another path's =#
  COMPILED_MODELS_DEJL[modelName] = (modelCode, true, hash(modelCode))
  Cache.store!(modelName, Cache.sizeOf(modelCode))
  #= and the ModelingToolkit path's build goes: its module is replaced, and its code (the same
     for a model without arrays) would reuse the build at its next translate =#
  IMTKGen.forgetBuild(modelName)
  push!(ARRAY_ODE_MODELS, modelName)
  return nothing
end

"""
  Returns true if the model was compiled again
"""
function modelWasCompiledAgain(modelName)
  return COMPILED_MODELS_MTK[modelName][2]
end

"""
    modelEvaluated!(modelName::String) -> Nothing

The module of `modelName` was evaluated from its latest code: `modelWasCompiledAgain` is false
until a translate changes the code.
"""
function modelEvaluated!(modelName::String)::Nothing
  local (code, _, h) = COMPILED_MODELS_MTK[modelName]
  COMPILED_MODELS_MTK[modelName] = (code, false, h)
  return nothing
end

function getCompiledModelDE(modelName)
  haskey(COMPILED_MODELS_DEJL, modelName) ||
    error("DE-mode model $(modelName) is not compiled. Available DE-mode models: $(join(keys(COMPILED_MODELS_DEJL), ", "))")
  return COMPILED_MODELS_DEJL[modelName][1]
end

modelWasCompiledAgainDE(modelName) = COMPILED_MODELS_DEJL[modelName][2]

"""
The DE-mode module of `modelName` evaluated: later calls reuse it until it changes.
"""
function _evalModelDE(modelName::String, modelCode::Expr)
  @BACKEND_PERFLOG "[backendAPI] evaluate the DE-mode module" (@eval $modelCode)
  local (code, _, h) = COMPILED_MODELS_DEJL[modelName]
  COMPILED_MODELS_DEJL[modelName] = (code, false, h)
  return nothing
end

"""

```
writeModelToFile(modelName::String, filePath::String; keepComments = true, keepBeginBlocks = true)
```
  Writes a model to file by default the file is formatted and comments are kept.
"""
function writeModelToFile(modelName::String, filePath::String; keepComments = true, keepBeginBlocks = true)
  model = _writtenModel(modelName)
  try
    mAsStr = modelToString(modelName; MTK = true,
                           keepComments = keepComments,
                           keepBeginBlocks = keepBeginBlocks)
    if model.head == :module
      #= Module expression serializes cleanly, no begin/end stripping needed =#
      mAsStr = OSMC_COPYRIGHT_HEADER * mAsStr
      writeStringToFile(filePath, mAsStr)
    else
      try
        #= Replace top level begin/end for legacy quote blocks =#
        beginIdx = last(findfirst("begin", mAsStr)) + 1
        endIdx = first(findlast("end",  mAsStr)) - 1
        mAsStr = mAsStr[beginIdx:endIdx]
        mAsStr = OSMC_COPYRIGHT_HEADER * mAsStr
        writeStringToFile(filePath, mAsStr)
      catch e
        @error "Error removing initial begin/end pairs" exception=(e, catch_backtrace())
      end
    end
  catch e
    @error "Failed writing $model to file: $filePath" exception=(e, catch_backtrace())
  end
end

"""
    Write the contents of a string to file.
"""
function writeStringToFile(fileName::String, contents::String)
  local fdesc = open(fileName, "w")
  write(fdesc, contents)
  close(fdesc)
end

"""
  Prints a model.
  If the specified model exists. Print it to stdout.
"""
function printModel(modelName::String; MTK = true, keepComments = true, keepBeginBlocks = true)
  try
    println(modelToString(modelName::String; MTK = MTK, keepComments = keepComments, keepBeginBlocks = keepBeginBlocks))
  catch e
    @error "Model: $(modelName) is not compiled. Available models are: $(availableModels())"
    error("Error printing model: $(modelName)")
  end
end

"""
 Converts a given backend model to a string
"""
function modelToString(modelName::String; MTK = true, keepComments = true, keepBeginBlocks = true)
  local model::Expr = _writtenModel(modelName)
  strippedModel = "$model"
  #= Remove all the redundant blocks from the model =#
  if keepComments == false
    strippedModel = CodeGeneration.stripComments(model)
  end
  if keepBeginBlocks == false
    strippedModel = CodeGeneration.stripBeginBlocks(model)
  end
  local modelStr::String = "$strippedModel"
  local formattedResults
  try
    formattedResults = JuliaFormatter.format_text(modelStr;
                                                  remove_extra_newlines = true,
                                                  indent = 4,
                                                  margin = 200,
                                                  always_use_return = true)
  catch e
    isFatal(e) && rethrow()
    @warn "Julia Formatter failed to format the output results due to $(e)"
    formattedResults = modelStr
  end
  return formattedResults
end


"""
    Prints available compiled models to stdout
"""
function availableModels()::String
  str = "Compiled models (MTK-MODE):\n"
  for m in keys(COMPILED_MODELS_MTK)
    str *= "  $m\n"
  end
  if !isempty(COMPILED_MODELS_DEJL)
    str *= "Compiled models (DEMode):\n"
    for m in keys(COMPILED_MODELS_DEJL)
      str *= "  $m\n"
    end
  end
  return str
end

"""
  ```
   simulateModel(modelName::String;
                       MODE = DEFAULT_BACKEND_MODE[],
                       tspan = (0.0, 1.0),
                       solver = defaultSolver(),
                       kwargs...)
  ```
  Simulates model interactively.
  The solver need to be passed with a : before the name, example:
  OMBackend.simulateModel(modelName, tspan = (0.0, 1.0), solver = :(Tsit5()));
"""
function simulateModel(modelName::String;
                       MODE = DEFAULT_BACKEND_MODE[],
                       tspan = (0.0, 1.0),
                       solver = defaultSolver(),
                       overwriteCache::Bool = false,
                       kwargs...)
  modelName = canonicalName(modelName)
  local modelCode::Expr
  #= A model translated by the array ODE path is a DE-mode module. =#
  modelName in ARRAY_ODE_MODELS && (MODE = DEMode)
  #= `parameters` (tunable parameter values for this run) is handled by the
     IMTK path only; elsewhere it would reach `solve` as an unknown keyword. =#
  haskey(kwargs, :parameters) && MODE != IMTK_MODE && !(modelName in ARRAY_ODE_MODELS) &&
    error("simulateModel: `parameters` needs IMTK mode (the default), got $(MODE)")
  if MODE == MTK_MODE
    #= This does a redundant string conversion for now due to modeling toolkit being as is...=#
    #= An uncompiled model is an error here (it went on to an UndefVarError of modelCode). =#
    modelCode = getCompiledModel(modelName)
    try
      #= Only re-eval if the module does not exist yet or the code changed =#
      local needsEval = overwriteCache || !_modelDefined(Symbol(modelName)) || modelWasCompiledAgain(modelName)
      if needsEval
        #= the replaced module's build and external objects go with it =#
        IMTKGen.forgetBuild(modelName)
        @eval $modelCode
      else
        CodeGeneration.registerModelFunctions!(_modelModule(Symbol(modelName)))
      end
      #= Run in latest world age to see the just-eval'd module. Push the
         translate-time run dir on the log stack so any dumps emitted during
         structural_simplify / buildDirectRHSProblem / _solveDAEInitialization!
         land in the per-model directory instead of the session root. =#
      local _runDir = get(MODEL_RUN_DIRS, modelName, nothing)
      local _doSim = () -> Base.invokelatest() do
        local mod = getfield(OMBackend, Symbol(modelName))
        mod.simulate(tspan, solver; kwargs...)
      end
      _runDir === nothing ? _doSim() : withLogRunDir(_doSim, _runDir)
    catch err
      err isa CodeGeneration.ModelicaAssertionError ||
        @error "Interactive evaluation failed" exception_type=typeof(err) mode=MODE model=modelName
      rethrow(err)
    end
  elseif MODE == IMTK_MODE
    #= overwriteCache forces a fresh build: bypass the codeHash reuse check so the
       cached problem is rebuilt (not just remade) before simulate. A failed
       rebuild warns and forgets the cached build; simulateIMTK then falls back
       to the module's own simulate. =#
    if overwriteCache
      IMTKGen._buildAndCache(modelName, getCompiledModel(modelName); overwriteCache = true)
    end
    #= Reuse the build cached in the backend at translate time (no
       structural_simplify re-run); simulateIMTK falls back to module simulate. =#
    local _runDir = get(MODEL_RUN_DIRS, modelName, nothing)
    local _doSim = () -> IMTKGen.simulateIMTK(modelName, tspan, solver; kwargs...)
    try
      return _runDir === nothing ? _doSim() : withLogRunDir(_doSim, _runDir)
    catch err
      #= A violated assert is the model's result, not a failure of the tool. =#
      err isa CodeGeneration.ModelicaAssertionError ||
        @error "iMTK simulate failed" exception_type=typeof(err) mode=MODE model=modelName
      rethrow(err)
    end
  elseif MODE == DEMode
    try
      modelCode = getCompiledModelDE(modelName)
    catch err
      println("Failed to simulate DE-mode model.")
      println("Available models are:")
      println(availableModels())
      rethrow(err)
    end
    try
      local needsEval = overwriteCache || !_modelDefined(Symbol(modelName)) || modelWasCompiledAgainDE(modelName)
      if needsEval
        _evalModelDE(modelName, modelCode)
      end
      #= the first simulate of a module compiles its functions =#
      @BACKEND_PERFLOG "[backendAPI] DE-mode simulate" Base.invokelatest() do
        local mod = getfield(OMBackend, Symbol(modelName))
        mod.simulate(tspan, solver; kwargs...)
      end
    catch err
      err isa CodeGeneration.ModelicaAssertionError ||
        @error "Interactive evaluation failed" exception_type=typeof(err) mode=MODE model=modelName
      rethrow(err)
    end
  else
    error("Unsupported mode: $(MODE)")
  end
end

"""
    getMTKProblem(modelName; tspan=(0.0, 1.0), overwriteCache=false) -> Tuple

Build the problem of an already-translated model without solving it. Call
`OM.translate` first. Returns the tuple of the generated model function; its
first two elements are the `ODEProblem` and the model's event callbacks,
which a solve needs too.

# Example
```julia
OM.translate("Modelica.Mechanics.MultiBody.Examples.Elementary.Pendulum")
prob, callbacks = OMBackend.getMTKProblem("Modelica.Mechanics.MultiBody.Examples.Elementary.Pendulum")
sol = solve(prob, Rodas5P(); callback = callbacks)
```
"""
function getMTKProblem(modelName::String;
                       tspan = (0.0, 1.0),
                       overwriteCache::Bool = false)
  modelName = canonicalName(modelName)
  local arrayModel = modelName in ARRAY_ODE_MODELS
  local modelCode::Expr = arrayModel ? getCompiledModelDE(modelName) : getCompiledModel(modelName)
  local needsEval = overwriteCache || !_modelDefined(Symbol(modelName)) ||
    (arrayModel ? modelWasCompiledAgainDE(modelName) : modelWasCompiledAgain(modelName))
  if needsEval
    #= the replaced module's build and external objects go with it =#
    arrayModel || IMTKGen.forgetBuild(modelName)
    arrayModel ? _evalModelDE(modelName, modelCode) : @eval($modelCode)
  elseif !arrayModel
    CodeGeneration.registerModelFunctions!(_modelModule(Symbol(modelName)))
  end
  Base.invokelatest() do
    local mod = getfield(OMBackend, Symbol(modelName))
    local modelFn = getfield(mod, Symbol(string(modelName, "Model")))
    modelFn(tspan)
  end
end

"""
    withTunableParameters(f, names)

Run `f()` (a `translate`, or a `simulate` that compiles) with the parameters
`names` kept as parameters of the generated model instead of being folded into
its equations as constants, so the compiled model can be simulated again with
other values of them. Parameters whose bindings depend on them are kept too.
Names as in Modelica (`body.m`) or flattened (`body_m`). An array parameter
(`nn.W1`, also a record field `r.T` or inside an array of components
`gs[1].g`) is kept element by element, unused elements included; its
elements are set one by one (`nn.W1[2][1]`), and uses of the whole array
(e.g. a function argument, which the frontend expands) follow them. A tunable
parameter read in an initial algorithm (evaluated at compile time) is an
error.

```julia
OMBackend.withTunableParameters(["alpha", "beta"]) do
  OM.translate("LotkaVolterra", "lv.mo")
end
```
"""
function withTunableParameters(f::Function, names)
  local canon = Set{String}(canonicalName(_elementSubscripts(String(n))) for n in names)
  return Base.ScopedValues.with(f, TUNABLE_PARAMETERS => canon)
end

"""
    isTunable(modelName, names) -> Bool

Whether the compiled `modelName` (its cached build) was compiled with all
`names` tunable (`withTunableParameters`; an array as a whole) and has them
as parameters, so that `simulate(...; parameters)` can set them. False
without a cached build, or after the model was compiled again with other
tunable parameters.
"""
function isTunable(modelName::String, names)::Bool
  local built = get(IMTKGen.BUILT, canonicalName(modelName), nothing)
  built === nothing && return false
  local prob = built[1]
  local tunable = get(IMTKGen.TUNABLE_SETS, canonicalName(modelName), Set{String}())
  local isp = s -> Runtime.ModelingToolkit.is_parameter(prob, Symbol(s))
  #= An array is a parameter through its elements (`nn_W1[1][1]`, any rank). =#
  return all(names) do n
    local s = canonicalName(_elementSubscripts(string(n)))
    isTunableParameter(s, tunable) && (isp(s) || any(d -> isp(s * "[1]"^d), 1:8))
  end
end

"""
    eventCallbacks(modelName) -> callbacks or nothing

The event callbacks of a compiled model's cached build (IMTK mode), as the
second element of `getMTKProblem`'s result, without building the problem
again. The problem carries them as well (`prob.kwargs[:callback]`, after any
events of the MTK system), and a solve uses the problem's. `nothing` without a
cached build.
"""
function eventCallbacks(modelName::String)
  local built = get(IMTKGen.BUILT, canonicalName(modelName), nothing)
  return built === nothing ? nothing : built[2]
end

"""
    pristineParameters(modelName) -> Vector or nothing

The parameter vector the simulations of a compiled model start from. A
simulation's solved problem (`sol.prob.p`) carries the vector as the event
callbacks left it (if-condition switches flipped), so solving that problem
again must start from this one. `nothing` without a cached build (IMTK mode
caches it at translate time).
"""
function pristineParameters(modelName::String)
  local p = get(IMTKGen.PRISTINE_P, canonicalName(modelName), nothing)
  return p === nothing ? nothing : deepcopy(p)
end

"""
    _modelDefined(name::Symbol) -> Bool
    _modelModule(name::Symbol) -> Module

Whether the module of a compiled model (`OMBackend.<name>`) is defined, and that module, read in the
latest world. `translate` evaluates the module; a caller that translates and simulates in one function
runs in the world it started in, where the binding does not exist yet: Julia 1.12 warns ("access to
binding ... in a world prior to its definition world") and later versions error.
"""
_modelDefined(name::Symbol)::Bool = Base.invokelatest(isdefined, OMBackend, name)
_modelModule(name::Symbol)::Module = Base.invokelatest(getglobal, OMBackend, name)

"""
    resimulateModel(modelName; solver, tspan, parameters = Dict(), kwargs...)

Simulate an already compiled model again, without recompiling. `parameters`
(name => value) sets parameters kept tunable at compile time
(`withTunableParameters`); other keyword arguments go to the solver.
"""
function resimulateModel(modelName::String;
                         solver = defaultSolver(),
                         MODE = DEFAULT_BACKEND_MODE[],
                         tspan=(0.0, 1.0),
                         parameters::AbstractDict = Dict(),
                         kwargs...)
  return simulateModel(modelName; MODE = MODE, tspan = tspan, solver = solver,
                       (isempty(parameters) ? (;) : (; parameters = parameters))..., kwargs...)
end

"
`plot(sol::Runtime.OMSolution)`
  The default plot function of OMBackend.
  All labels of the variables and the name is given by default
"
function plot(sol::Runtime.OMSolution)
  local nsolution = sol.diffEqSol
  local t = nsolution.t
  local rescols = collect(eachcol(transpose(hcat(nsolution.u...))))
  labels = permutedims(sol.idxToName.vals)
  Plots.plot(t, rescols; labels=labels)
end

"""
`function plot(sol)`
  An alternative plot function in OMBackend.
  All labels of the variables and the name is given by default
"""
function plot(sol)
  Plots.plot(sol)
end

"""
  Plot for an OMSolution that contains several sub solutions.
  Plots all part of the solution on the same graph.
"""
function plot(sol::Runtime.OMSolutions; legend = false, limX = 0.0, limY = 1.0)
  local sols = sol.diffEqSol
  local prevP = Plots.plot!(sols[1]; legend = legend, xlim=limX, ylim = limY)
  for sol in sols[2:end]
    p = Plots.plot!(prevP; legend = legend, xlim=limX, ylim = limY)
    prevP = p
  end
  return prevP
end

"""
  Plots a vector of solutions
"""
function plot(sol::Vector; legend = false, limX = 0.0, limY = 1.0, kwargs...)
  sols = sol
  local prevP = Plots.plot!(sols[1]; legend = legend, xlim=limX, ylim = limY, kwargs...)
  for sol in sols[2:end]
    p = Plots.plot!(prevP; legend = legend, xlim=limX, ylim = limY, kwargs...)
    prevP = p
  end
  return prevP
end

"""
  Returns the value of a variable given a solution and a string.
```
  getVariableValues(sol, varName::String)
```
Example use:
```julia
OM.translate("HelloWorld", "./Models/HelloWorld.mo");
sol = OM.simulate("HelloWorld");

OM.OMBackend.getVariableValues(sol, "x")
11-element Vector{Float64}:
 1.0
 0.7689684977240044
 0.555181390244627
 0.371869514552751
 0.23297802339017032
 0.13657983532457932
 0.07542877379860594
 0.039431995308782476
 0.019632381398183626
 0.009355999959425394
 0.006738051637508934
```
"""
function getVariableValues(sol::ODESolution, varName::String)
  if sol.prob.p isa CodeGeneration.ArrayODEGen.ArrayModelParameters
    local vals = CodeGeneration.ArrayODEGen.variableValues(sol, varName)
    vals === nothing && @warn "Did not locate a variable named '$(varName)' in the model"
    return vals
  end

  varAsJLSym = if varName != "time"
    canonicalSymbol(varName)
  else
    :t
  end
  try
    res = sol[varAsJLSym]
  catch e
    @warn "Did not locate a variable named '$(varName)' in the model" exception=(e, catch_backtrace())
    nothing
  end
end

"""
```
getVariableValues(sols::Vector, variables...)
```

Similar to getVariableValues, but for solutions that have went through one or more structural changes.
Here the same variable might have slightly different names depending on the context.

For instance it might be called M.A first and then M.B when the structure of the model is changed.
In the example above a pendulum goes through once such change.
Here you can specify the variables in order, they will be collected and merged into the final vector.

Example use:
```
julia> OM.OMBackend.getVariableValues(sols, "bouncingBall_x", "pendulum_x")
vcat(OM.OMBackend.getVariableValues(sols, "pendulum_y", "bouncingBall_y")...)
69-element Vector{Float64}:
  10.0
   ⋮
   8.71359663258124
   ⋮
 -11.346053750176466
     ⋮
  3.3856338791378615
```
"""
function getVariableValues(sols::Vector, variables...)
  local vals = Any[]
  for sol in sols
    for v in variables
      local vAsJLSym = if v != "time"
        canonicalSymbol(v)
      else
        :t
      end
      try
        push!(vals, sol[vAsJLSym])
      catch err
        #= A variable the solution does not keep is left out. =#
        _fallback(err, :variableValues; impact = :result)
      end
    end
  end
  return vcat(vals...)
end

"""
Wrapper function to MTK `observed`.
For models with states, the system is stored in `sol.prob.f.sys`.
For purely algebraic (0-unknown) models, MTK does not populate `sol.prob.f.sys`,
so we fall back to `LATEST_REDUCED_SYSTEM` stored as a module global in the
generated simulate function.
"""
function MTK_getObserved(sol, modelName=nothing)
  if sol.prob.f.sys !== nothing
    return ModelingToolkit.observed(sol.prob.f.sys)
  end
  if modelName !== nothing
    local modSym = canonicalSymbol(modelName)
    if _modelDefined(modSym)
      local mod = _modelModule(modSym)
      if isdefined(mod, :LATEST_REDUCED_SYSTEM)
        return ModelingToolkit.observed(mod.LATEST_REDUCED_SYSTEM)
      end
    end
  end
  return Symbolics.Equation[]
end

"""
Evaluate all observed equations at every time point in `sol.t`, resolving
inter-equation dependencies by processing equations in order and substituting
previously computed LHS values into later RHS expressions.
Returns a `Dict{String, Vector{Float64}}` mapping variable name to values, or `nothing`.
Used as fallback for purely algebraic (0-unknown) models where `sol[sym]` is unavailable.
"""
function MTK_evaluateAllObserved(sol, obs_eqs, modelName::String)
  local modSym = canonicalSymbol(modelName)
  if !_modelDefined(modSym)
    return nothing
  end
  local mod = _modelModule(modSym)
  if !isdefined(mod, :LATEST_REDUCED_SYSTEM)
    return nothing
  end
  local sys_t = ModelingToolkit.get_iv(mod.LATEST_REDUCED_SYSTEM)
  local result = Dict{String, Vector{Float64}}()
  local warnedUnresolved = OrderedSet{String}()
  for (i, ti) in enumerate(sol.t)
    local subst = Dict{Any,Any}(sys_t => ti)
    for eq in obs_eqs
      local lhsStr = string(eq.lhs)
      local substituted = Symbolics.substitute(eq.rhs, subst)
      local raw = Symbolics.value(substituted)
      if !(raw isa Number)
        if !(lhsStr in warnedUnresolved)
          push!(warnedUnresolved, lhsStr)
          @warn "MTK_evaluateAllObserved: observed equation for `$lhsStr` did not reduce to a numeric value after substitution; remaining expression: `$substituted`. This usually indicates the observed-equation list is not in topological order. Skipping this entry."
        end
        continue
      end
      local val = Float64(raw)
      subst[eq.lhs] = val
      if !haskey(result, lhsStr)
        result[lhsStr] = Vector{Float64}(undef, length(sol.t))
      end
      result[lhsStr][i] = val
    end
  end
  return result
end

"""
An observedFilter's patterns as strings.
"""
_observedFilterStrings(f::Vector{Regex})::Vector{String} = String[p.pattern for p in f]
_observedFilterStrings(f::Vector{String})::Vector{String} = f
