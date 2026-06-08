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
=#

using MetaModelica
using ExportAll
using Absyn
#= For interactive evaluation. =#
using ModelingToolkit
using SymbolicUtils
using DifferentialEquations
using OrdinaryDiffEq

import ..CodeGeneration
import ..Runtime
import .Backend.BDAE
import .Backend.BDAECreate
import .Backend.BDAEUtil
import .Backend.Causalize
import .SimulationCode

import Base.Meta
import JuliaFormatter
import OMBackend.CodeGeneration
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
  DAE_MODE = 1
  ODE_MODE = 2 #Currently not in operation
  MTK_MODE = 3
  #=
    Direct DifferentialEquations.jl emission. Builds an `ODEProblem` with an
    in-place RHS `f!(du, u, p, t)` that indexes `u`, `du`, and `p` by integer
    position. Bypasses ModelingToolkit entirely. Initial scope: pure ODE
    (no algebraic constraints), no VSS / structural transitions, no DOCC.
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

function info()
  println("OMBackend.jl")
  println("A Julia backend for the Equation Oriented Language Modelica!")
  println("Run any test module by executing runExample(<model-name>)")
  println("Available Example models include:")
  for (k,v) in EXAMPLE_MODELS
    println(k)
  end
end


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

# Per-model log-run directory captured at translate (`lower`) time. Looked up
# at simulate time by `simulateModel` so any dumps emitted during the
# post-MTK / `buildDirectRHSProblem` / `_solveDAEInitialization!` codegen
# paths land in the same per-model run directory as the BDAE/simCode logs,
# not the session root. Keyed by canonical model name.
const MODEL_RUN_DIRS = Dict{String, String}()

function logRunModelName(frontendDAE::DAE.DAE_LIST)::String
  @match DAE.COMP(ident = ident) = listHead(frontendDAE.elementLst)
  return ident
end

function logRunModelName(frontendDAE::OMFrontend.Frontend.FlatModel)::String
  return string(frontendDAE.name)
end

"""
    clearCaches!(; models=true, implementations=true, wrappers=true, extractors=true)

Clear persistent caches. By default all caches are cleared.
Use keyword arguments to selectively clear individual caches.

- `models`: compiled MTK model ASTs
- `implementations`: Modelica function implementations
- `wrappers`: RTG wrapper functions for symbolic dispatch
- `extractors`: per-element array extractor functions
"""
function clearCaches!(; models::Bool=true,
                        implementations::Bool=true,
                        wrappers::Bool=true,
                        extractors::Bool=true)
  cleared = String[]
  models          && (empty!(COMPILED_MODELS_MTK);                       push!(cleared, "models"))
  implementations && (empty!(CodeGeneration.MODELICA_FUNCTION_IMPLS);    push!(cleared, "implementations"))
  wrappers        && (empty!(CodeGeneration.MODELICA_FUNCTION_WRAPPERS); push!(cleared, "wrappers"))
  extractors      && (empty!(CodeGeneration.ELEM_FUNC_CACHE);            push!(cleared, "extractors"))
  return cleared
end

"""
 This function lowers the given Hybrid DAE to target code.
 It does so by first lowering the code to the backend representation and then to
 the simulation code representation.
 Finally, target code is generated depending on the backend mode (defaults to MTK mode).
The function list contains the sequential parts of a modelica model, that is the different functions that the model might use.
This is not part of the lowering process but it is to be generated before we generate MTK target code
"""
Base.@nospecializeinfer function translate(@nospecialize(frontendDAE::Union{DAE.DAE_LIST, OMFrontend.Frontend.FlatModel});
                   functionList = nothing,
                   BackendMode = DEFAULT_BACKEND_MODE[],
                   warnMissingStartValues = nothing,
                   eliminateNonDynamic::Union{Nothing, Bool, SimulationCode.EliminationOptions} = nothing,
                   observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                   checkSimCode::Bool = true,
                   returnNameMap::Bool = false)
  local previousWarnSetting = WARN_MISSING_START_VALUES[]
  local runId = createLogRunId(logRunModelName(frontendDAE))
  if warnMissingStartValues !== nothing
    warnMissingStartValues isa Bool || error("warnMissingStartValues must be Bool or nothing")
    WARN_MISSING_START_VALUES[] = warnMissingStartValues
  end
  try
    return withLogRunDir(runId) do
      #= Dump the flat model with functions as it arrives from the frontend =#
      @BACKEND_LOGGING if frontendDAE isa OMFrontend.Frontend.FlatModel
        local fLst = functionList !== nothing ? functionList : MetaModelica.nil
        debugWrite(logPath("backend/simCode", "frontend_initialModel.log"),
          replace(OMFrontend.Frontend.toFlatString(frontendDAE, fLst), "\\n" => "\n"))
      end
      local bDAE = @BACKEND_PERFLOG "[backendAPI] lower" lower(frontendDAE)
      local simCode
      if BackendMode == DAE_MODE
        error("DAE-mode is deprecated.")
      elseif BackendMode == DEMode
        #=
          Direct DifferentialEquations.jl path. Mirrors the MTK pipeline up
          through propagateConstants/aliasElim, but skips MTK-specific
          observed-equation work and dispatches to the DE emitter.
        =#
        simCode = generateSimulationCode(bDAE; mode = MTK_MODE)
        (simCodeFunctions, externalRuntimeNeeded) = if functionList !== nothing
          generateSimCodeFunctions(functionList)
        else
          (SimulationCode.ModelicaFunction[], false)
        end
        @assign begin
          simCode.functions = simCodeFunctions
          simCode.externalRuntime = externalRuntimeNeeded
        end
        simCodeFunctions = SimulationCode.flattenRecordParameters(simCodeFunctions)
        @assign simCode.functions = simCodeFunctions
        #= Mirror the MTK pipeline: collapse qualified ENUM_LITERAL paths up
           front. This is a pure string-shortening pass and is a no-op for
           enum-free models. =#
        simCode = SimulationCode.runSimCodePass("simplifyEnumLiteralPaths", simCode,
                                                SimulationCode.simplifyEnumLiteralPaths)
        simCode = SimulationCode.runSimCodePass("flattenRecordCallSites", simCode,
                                                SimulationCode.flattenRecordCallSites)
        local nameMap = NameRewriteMap()
        simCode = SimulationCode.runSimCodePass("canonicalizeCrefNames", simCode,
                                                sc -> SimulationCode.canonicalizeCrefNames(sc; nameMap = nameMap))
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterCanonicalNames.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("resolveIfExpInBindings", simCode,
                                                SimulationCode.resolveIfExpInBindings!)
        simCode = SimulationCode.runSimCodePass("pruneConstantConditions", simCode,
                                                SimulationCode.pruneConstantConditions)
        simCode = SimulationCode.runSimCodePass("foldParameterClosure", simCode,
                                                SimulationCode.foldParameterClosure)
        simCode = SimulationCode.runSimCodePass("inlinePreOfConstantParameters", simCode,
                                                SimulationCode.inlinePreOfConstantParameters)
        simCode = SimulationCode.runSimCodePass("propagateConstants", simCode,
                                                SimulationCode.propagateConstants)
        simCode = SimulationCode.runSimCodePass("eliminateAliasVariables", simCode,
                                                SimulationCode.eliminateAliasVariables)
        simCode = SimulationCode.runSimCodePass("eliminateRHSEquivalentEquations", simCode,
                                                SimulationCode.eliminateRHSEquivalentEquations)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterRHSEquiv.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("removeRedundantEquations", simCode,
                                                SimulationCode.removeRedundantEquations)
        simCode = SimulationCode.runSimCodePass("eliminateConstantParameters", simCode,
                                                SimulationCode.eliminateConstantParameters)
        #= eliminateFrozenStates runs AFTER eliminateConstantParameters so that
           parameter chains like `state = param` (param bound to a literal) are
           already substituted to `state = literal` before we look for them. =#
        simCode = SimulationCode.runSimCodePass("eliminateFrozenStates", simCode,
                                                SimulationCode.eliminateFrozenStates)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFrozenStates.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("pruneConstantConditions", simCode,
                                                SimulationCode.pruneConstantConditions)
        if !isempty(simCode.structuralTransitions) || !isempty(simCode.subModels)
          error("DEMode does not support structural transitions / VSS at this time. " *
                "Re-run with mode = OMBackend.MTK_MODE.")
        end
        _checkSimCodeBeforeCodegen(simCode, checkSimCode)
        return _translationResult(generateDETargetCode(simCode), nameMap, returnNameMap)
      elseif BackendMode == MTK_MODE || BackendMode == IMTK_MODE
        #@debug "Generate simulation code"
        simCode = @BACKEND_PERFLOG "[backendAPI] generateSimulationCode" generateSimulationCode(bDAE; mode = MTK_MODE)
        (simCodeFunctions, externalRuntimeNeeded) = if functionList !== nothing
          generateSimCodeFunctions(functionList)
        else
          (SimulationCode.ModelicaFunction[], false)
        end
        @assign begin
          simCode.functions = simCodeFunctions
          simCode.externalRuntime = externalRuntimeNeeded
        end
        #= Dump before record flattening =#
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_initial.log"), SimulationCode.dumpSimCode(simCode))
        #= Collapse qualified ENUM_LITERAL paths to leaf `Type.Literal` IDENT form. =#
        simCode = SimulationCode.runSimCodePass("simplifyEnumLiteralPaths", simCode,
                                                SimulationCode.simplifyEnumLiteralPaths)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterEnumSimplify.log"), SimulationCode.dumpSimCode(simCode))
        #= Flatten record parameters in functions =#
        simCodeFunctions = SimulationCode.flattenRecordParameters(simCodeFunctions)
        @assign simCode.functions = simCodeFunctions
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFlattenRecordParams.log"), SimulationCode.dumpSimCode(simCode))
        #= Flatten record arguments in equation call sites to match flattened signatures =#
        simCode = SimulationCode.runSimCodePass("flattenRecordCallSites", simCode,
                                                SimulationCode.flattenRecordCallSites)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFlattenRecordCallSites.log"), SimulationCode.dumpSimCode(simCode))
        #= Canonicalize every generated name once SimCode has its final record-call shape. =#
        local nameMap = NameRewriteMap()
        simCode = SimulationCode.runSimCodePass("canonicalizeCrefNames", simCode,
                                                sc -> SimulationCode.canonicalizeCrefNames(sc; nameMap = nameMap))
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterCanonicalNames.log"), SimulationCode.dumpSimCode(simCode))
        #= Resolve constant-condition IFEXPs in parameter bindings =#
        simCode = SimulationCode.runSimCodePass("resolveIfExpInBindings", simCode,
                                                SimulationCode.resolveIfExpInBindings!)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterResolveIfExp.log"), SimulationCode.dumpSimCode(simCode))
        #= Prune IFEXP/IF_EQUATION conditions that became compile-time constants. =#
        simCode = SimulationCode.runSimCodePass("pruneConstantConditions", simCode,
                                                SimulationCode.pruneConstantConditions)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterConstantConditionPruning.log"), SimulationCode.dumpSimCode(simCode))
        #= Constant propagation and alias elimination run AFTER record flattening
           so that all CREF references are in their final form before substitution.
           Running these earlier caused dangling references when flattenRecordCallSites
           introduced new CREFs for already-eliminated variables. =#
        #= BLT-driven parameter-closure fold runs before propagateConstants so that
           any parameter-closure chains (e.g. KinematicPTP's seven algebraic unknowns
           defined solely by parameter expressions) are promoted to parameters before
           MTK sees them. This removes the Newton-init failure mode where zero guesses
           on those unknowns produce NaN/Inf evaluations. =#
        simCode = SimulationCode.runSimCodePass("foldParameterClosure", simCode,
                                                SimulationCode.foldParameterClosure)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFoldClosure.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("inlinePreOfConstantParameters", simCode,
                                                SimulationCode.inlinePreOfConstantParameters)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterInlinePreParam.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("propagateConstants", simCode,
                                                SimulationCode.propagateConstants)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterConstantProp.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("lowerComplexOperatorRecords", simCode,
                                                SimulationCode.lowerComplexOperatorRecords)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterComplexLowering.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("eliminateAliasVariables", simCode,
                                                SimulationCode.eliminateAliasVariables)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterAliasElimination.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("eliminateRHSEquivalentEquations", simCode,
                                                SimulationCode.eliminateRHSEquivalentEquations)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterRHSEquiv.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("removeRedundantEquations", simCode,
                                                SimulationCode.removeRedundantEquations)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterRemoveRedundant.log"), SimulationCode.dumpSimCode(simCode))
        #= Constant-parameter elimination shrinks the parameter list MTK sees
           before structural_simplify. Tier-1 only: the pass internally skips
           VSS / DOCC / sub-model variants where a parameter could be re-bound
           at runtime. =#
        simCode = SimulationCode.runSimCodePass("eliminateConstantParameters", simCode,
                                                SimulationCode.eliminateConstantParameters)
        #= Drop protected sink variables and their defining equations. Runs
           before eliminateDeadParameters so any parameters whose only consumer
           was a dropped sink get caught by the dead-parameter sweep. =#
        simCode = SimulationCode.runSimCodePass("dropObservationOnlyVariables", simCode,
                                                SimulationCode.dropObservationOnlyVariables)
        simCode = SimulationCode.runSimCodePass("eliminateDeadParameters", simCode,
                                                SimulationCode.eliminateDeadParameters)
        #= eliminateFrozenStates runs AFTER eliminateConstantParameters so that
           parameter chains like `state = param` (param bound to a literal) are
           already substituted to `state = literal` before detection. =#
        simCode = SimulationCode.runSimCodePass("eliminateFrozenStates", simCode,
                                                SimulationCode.eliminateFrozenStates)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFrozenStates.log"), SimulationCode.dumpSimCode(simCode))
        #= foldExplicitSingleAssign: generalised single-defining-equation tearing.
           Substitutes `0 = v - rhs` where v is an unprotected ALG_VARIABLE
           uniquely defined by that residual. Guards: skips sub-models /
           metaModel / flatModel; skips bracketed (scalarized array) names;
           skips names already in aliasMap. Post-substitution survivor-scan
           aborts the fold if any folded name still appears anywhere. =#
        simCode = SimulationCode.runSimCodePass("foldExplicitSingleAssign", simCode,
                                                SimulationCode.foldExplicitSingleAssign)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterExplicitFold.log"), SimulationCode.dumpSimCode(simCode))
        #= Second alias-elim pass: earlier simplifiers (foldExplicitSingleAssign,
           propagateConstants, dropObservationOnlyVariables) collapse n-term
           connector flow sums into 2-term residuals like `a + b = 0` that the
           first alias-elim pass could not yet see. =#
        simCode = SimulationCode.runSimCodePass("eliminateAliasVariables", simCode,
                                                SimulationCode.eliminateAliasVariables)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterAliasElimination2.log"), SimulationCode.dumpSimCode(simCode))
        #= Second RHS-equivalence pass: the alias-elim2 above can collapse two
           equations of the form `Xi - der(s_i)` onto the same `der(s)`. The
           pass keys equations by RHS string before its own substitution
           propagates, so a single iteration can leave equivalent pairs
           ungrouped when one pair must fold first to expose another's key
           (e.g. mass1_v → brake_v before ifEq_tmp1 ↔ ifEq_tmp2 becomes
           visible via the now-shared der(brake_v)). Iterate to fixed point;
           a no-op call is cheap and returns simCode unchanged. =#
        let prev = -1
          for _ in 1:6
            simCode = SimulationCode.runSimCodePass("eliminateRHSEquivalentEquations", simCode,
                                                    SimulationCode.eliminateRHSEquivalentEquations)
            local n = length(simCode.residualEquations)
            n == prev && break
            prev = n
          end
        end
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterRHSEquiv2.log"), SimulationCode.dumpSimCode(simCode))
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterEliminateConstParams.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("classifyAdditionalDiscretes", simCode,
                                                SimulationCode._classifyAdditionalDiscreteVariables)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterClassifyDiscretes.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.runSimCodePass("pruneConstantConditions", simCode,
                                                SimulationCode.pruneConstantConditions)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterPostAliasConditionPruning.log"), SimulationCode.dumpSimCode(simCode))
        #= Output-only variable elimination runs AFTER const-prop and alias-elim,
           so that alias chains are resolved and the output-only subgraph is cleanly separated.
           A fresh matching is computed from the current equation/variable set. =#
        local elimOpts = if eliminateNonDynamic === true
          SimulationCode.EliminationOptions()
        elseif eliminateNonDynamic isa SimulationCode.EliminationOptions
          eliminateNonDynamic
        else
          nothing
        end
        if elimOpts !== nothing
          @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_beforeElimination.log"), SimulationCode.dumpSimCode(simCode))
          simCode = SimulationCode.runSimCodePass("eliminateNonDynamic", simCode,
                                                  sc -> SimulationCode.eliminateOutputOnlyVariables(sc, elimOpts))
          @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterElimination.log"), SimulationCode.dumpSimCode(simCode))
        end
        #= Observed filter: controls which alias-eliminated variables become observed equations.
           Default (nothing): skip ALL alias observed equations for fast compilation.
           With filter patterns: keep only matching aliases as observed. =#
        local observedBefore = SimulationCode.simCodeMetrics(simCode)
        local observedT0 = time()
        if observedFilter === nothing
          #= Default: drop alias-map entries whose eliminated name is an
             internal/auto-generated cref (scalarized array indices like
             `R_T[1][2]`, frame-internal `frame_a_t[3]`, etc.) but KEEP
             entries whose eliminated name is a user-visible Modelica
             identifier. Without this, state-state aliases produced by
             OMBackend's alias-elim passes (e.g. `rev_phi ~ damper_phi_rel`
             from the Revolute joint flange constraint) get silently
             dropped before codegen, so `sol(t; idxs = lookup[\"rev_phi\"])`
             raises KeyError in tests that query the eliminated name.
             A "user-visible" name is heuristic: no `[` (no scalarized array
             subscript) and not a known auto-gen prefix. The bracket guard
             catches the vast majority of MultiBody internals (Engine1a's
             876 aliases collapse to a small handful) while preserving the
             named Modelica vars that user code can reasonably query. =#
          if !isempty(simCode.aliasMap)
            local originalCount = length(simCode.aliasMap)
            local kept = filter(simCode.aliasMap) do entry
              #= Drop dangling aliases whose representative has itself been
                 eliminated from the SimVar HT (e.g. FrozenStateConstraint:
                 `a → F` survives in aliasMap after `F` is removed by
                 eliminateFrozenStates; emitting `a ~ F` then raises
                 UndefVarError at the eval that binds alias names). =#
              _isUserVisibleAliasName(entry.eliminatedName) &&
                haskey(simCode.stringToSimVarHT, entry.representativeName)
            end
            @debug "[SIMCODE: observedFilter] kept $(length(kept)) of $originalCount user-visible alias observed equations (default filter)"
            @assign simCode.aliasMap = kept
          end
        else
          local filterStrings = if observedFilter isa Vector{Regex}
            [p.pattern for p in observedFilter]
          else
            observedFilter
          end
          @assign simCode.observedFilter = filterStrings
          #= Filter alias map entries to keep only matching patterns =#
          if !isempty(simCode.aliasMap)
            local originalCount = length(simCode.aliasMap)
            local patterns = [Regex(p) for p in filterStrings]
            local filteredMap = filter(simCode.aliasMap) do entry
              any(p -> occursin(p, entry.eliminatedName), patterns)
            end
            @assign simCode.aliasMap = filteredMap
            @debug "[SIMCODE: observedFilter] kept $(length(filteredMap)) of $originalCount alias observed equations"
          end
        end
        SimulationCode.logSimCodePassMetrics("observedFilter", observedBefore, simCode, time() - observedT0)
        simCode = SimulationCode.cleanupTrivialResidualEquations(simCode; sourcePass = "observedFilter")
        #= Re-derive SCCs against the residual array MTK will actually see.
           The SCC stamp written by transformToSimCode indexes into the
           pre-pipeline residual list and is stale after alias-elim,
           foldExplicitSingleAssign, output-only elimination, etc. MTK
           codegen needs accurate cycle info to extract NonlinearSystem
           sub-blocks per cyclic SCC. =#
        simCode = SimulationCode.runSimCodePass("recomputeStronglyConnectedComponents", simCode,
                                                SimulationCode.recomputeStronglyConnectedComponents)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterRecomputeSCC.log"), SimulationCode.dumpSimCode(simCode))
        #= Standalone index-overconstraint diagnostic on the final SimCode (gated
           by OMBACKEND_INDEX_DIAG); inspects the differential-incidence
           localization without mutating the system. =#
        simCode = SimulationCode.indexOverconstraintDiagnostic(simCode)
        _checkSimCodeBeforeCodegen(simCode, checkSimCode)
        local _genCode = if BackendMode == IMTK_MODE
          @BACKEND_PERFLOG "[backendAPI] generateIMTKTargetCode" generateIMTKTargetCode(simCode)
        else
          @BACKEND_PERFLOG "[backendAPI] generateMTKTargetCode" generateMTKTargetCode(simCode)
        end
        return _translationResult(_genCode, nameMap, returnNameMap)
      else
        @error "Unsupported BackendMode: $BackendMode. Valid modes are: MTK_MODE, IMTK_MODE, DEMode"
      end
    end
  finally
    WARN_MISSING_START_VALUES[] = previousWarnSetting
  end
end

#= Heuristic predicate: is the eliminated alias name something a Modelica
   user might reasonably query via `sol(t; idxs = :name)`? Scalarized array
   indices and frame-internal connector names contain `[` and are filtered
   out by default; clean identifiers like `rev_phi` / `damper_w_rel` are
   kept so the alias observed equation reaches MTK. =#
function _isUserVisibleAliasName(name::AbstractString)::Bool
  occursin('[', name) && return false
  return true
end

function _translationResult(result::Tuple{String, Expr}, nameMap::NameRewriteMap,
                            returnNameMap::Bool)
  if returnNameMap
    return (result[1], result[2], nameMap)
  end
  return result
end

function _checkSimCodeBeforeCodegen(simCode::SimulationCode.SIM_CODE, checkSimCode::Bool)
  if !checkSimCode
    return nothing
  end
  local checkResult = SimulationCode.SimCodeCheck.check(simCode)
  # An :error cref/canonical violation means a residual references a name that
  # resolves to no SimVar/eliminated/builtin; code generation would otherwise
  # throw a bare UndefVarError deep in MTK. Abort with the located reason, and
  # skip the verbose warning report so a caller catching SimCodeCheckError can
  # silence the failure entirely.
  local blocking = filter(v -> v.severity === :error &&
                               (v.rule === :canonical_cref_names || v.rule === :cref_resolution),
                          checkResult.violations)
  if !isempty(blocking)
    local details = join([string("  - ", v.where, ": ", v.detail) for v in blocking], "\n")
    throw(SimulationCode.SimCodeCheck.SimCodeCheckError(
      string("Code generation aborted for ", simCode.name, ": ", length(blocking),
             " unresolved component reference(s). These references do not resolve ",
             "against the simulation variable table, the eliminated-variable set, or ",
             "the builtin set, so code generation would raise an undefined-variable error.\n", details)))
  end
  SimulationCode.SimCodeCheck.report(stderr, checkResult; modelName = string(simCode.name))
  return nothing
end

"""
`function dumpInitialSystem(frontendDAE::DAE.DAE_LIST)`
 Dumps a textual representation of the initial system.
"""
function dumpInitialSystem(frontendDAE::DAE.DAE_LIST)::String
  str =  "Length of frontend DAE:" * string(length(frontendDAE.elementLst)) * "\n"
  bDAE = BDAECreate.lower(frontendDAE)
  str *= BDAEUtil.stringHeading1(bDAE, "translated")
  return str
end

"""
`function printInitialSystem(frontendDAE::DAE.DAE_LIST)`
 Dumps a textual representation of the initial system.
"""
function printInitialSystem(frontendDAE::DAE.DAE_LIST)
  print(dumpInitialSystem(frontendDAE::DAE.DAE_LIST))
end

"""
 Transforms given DAE-IR/Hybrid DAE to backend DAE-IR (BDAE-IR)
"""
function lower(frontendDAE::DAE.DAE_LIST)::BDAE.BACKEND_DAE
  local modelName = logRunModelName(frontendDAE)
  local runId = createLogRunId(modelName)
  # Remember the translate-time run dir so simulate-time code can land its
  # dumps in the same per-model directory. Key by canonical name to match the
  # lookup in simulateModel.
  MODEL_RUN_DIRS[canonicalName(modelName)] = runId
  local lowerWork = function()
    local bDAE::BDAE.BACKEND_DAE
    @debug "Length of frontend DAE:" length(frontendDAE.elementLst)
    @assert typeof(listHead(frontendDAE.elementLst)) == DAE.COMP
    #= Create Backend structure from Frontend structure =#
    bDAE = BDAECreate.lower(frontendDAE)
    @debug "[BDAE] translated; full dump is available in backend/bdae logs when backend logging is enabled"
    #= Transform ASUB expressions: der(array)[i] to der(array[i]) =#
    bDAE = Causalize.transformASUBExpressions(bDAE)
    #= Mark state variables =#
    bDAE = Causalize.detectStates(bDAE)
    @debug "[BDAE] states marked"
    #= Flatten CREF_QUAL with subscripted array finals =#
    bDAE = Causalize.flattenArrayCrefs(bDAE)
    #= Transform if expressions to if equations =#
    bDAE = Causalize.detectIfExpressions(bDAE)
    @debug "[BDAE] if equations transformed"
    #= Expand record field array variables into individual scalar element variables =#
    bDAE = Causalize.expandRecordFieldArrays(bDAE)
    #= Expand COMPLEX_EQUATIONs into scalar equations =#
    bDAE = Causalize.expandComplexEquations(bDAE)
    #= We always residualize since residuals are easier to work with =#
    bDAE = Causalize.residualizeEveryEquation(bDAE)
    @debug "[BDAE] residualized"
    return bDAE
  end
  return hasActiveLogRunDir() ? lowerWork() : withLogRunDir(lowerWork, runId)
end


"""
  Transforms given FlatModelica to backend DAE-IR (BDAE-IR).
"""
function lower(fm::OMFrontend.Frontend.FLAT_MODEL)
  local modelName = logRunModelName(fm)
  local runId = createLogRunId(modelName)
  MODEL_RUN_DIRS[canonicalName(modelName)] = runId
  local lowerWork = function()
    local preprocessedFM = FrontendUtil.handleBuiltin(fm)
    local bDAE = BDAECreate.lower(preprocessedFM)
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_initial.log"), BDAEUtil.stringHeading1(bDAE, "initial BDAE"))
    #= Dump the actual DAE.Exp structure to a file for analysis =#
    @BACKEND_LOGGING begin
      open(logPath("backend/bdae", "bdae_structureDump.log"), "w") do io
        println(io, "=== BDAE Expression Structure Dump ===")
        for (i, eq) in enumerate(first(bDAE.eqs).orderedEqs)
          if i <= 20  # Dump first 20 equations
            println(io, "\n--- Equation $i: $(typeof(eq)) ---")
            dump(io, eq; maxdepth=10)
          end
        end
        println(io, "\n=== First 10 Variables ===")
        for (i, v) in enumerate(first(bDAE.eqs).orderedVars)
          if i <= 10
            println(io, "\n--- Variable $i ---")
            dump(io, v; maxdepth=8)
          end
        end
      end
    end
    #= Reclassify integer variables as parameters and remove their equations =#
    bDAE = Causalize.resolveIntegerVariables(bDAE)
    #= Transform ASUB expressions: der(array)[i] to der(array[i]) =#
    bDAE = Causalize.transformASUBExpressions(bDAE)
    #= Mark state variables =#
    bDAE = Causalize.detectStates(bDAE)
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_afterDetectStates.log"), BDAEUtil.stringHeading1(bDAE, "after detect states"))

    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_afterASUBTransform.log"), BDAEUtil.stringHeading1(bDAE, "after ASUB transformation"))
    #= Flatten CREF_QUAL with subscripted array finals to match hash table entries =#
    bDAE = Causalize.flattenArrayCrefs(bDAE)
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_afterFlattenCrefs.log"), BDAEUtil.stringHeading1(bDAE, "after CREF flattening"))
    #= Transform if expressions to if equations =#
    bDAE = Causalize.detectIfExpressions(bDAE)
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_afterIfExpressions.log"), BDAEUtil.stringHeading1(bDAE, "after if expressions"))
    #= Expand record field array variables into individual scalar element variables =#
    bDAE = Causalize.expandRecordFieldArrays(bDAE)
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_afterExpandRecordFields.log"), BDAEUtil.stringHeading1(bDAE, "after record field expansion"))
    #= Expand COMPLEX_EQUATIONs into scalar equations =#
    bDAE = Causalize.expandComplexEquations(bDAE)
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_afterExpandComplex.log"), BDAEUtil.stringHeading1(bDAE, "after complex equation expansion"))
    bDAE = Causalize.residualizeEveryEquation(bDAE)
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_afterResidualize.log"), BDAEUtil.stringHeading1(bDAE, "after residualize"))
    #=
      Remove unused parameters and or constants.
      Important optimization for some systems.
      TODO: also check bindings of all parameters before re-adding this call.
    =#
    #bDAE = Causalize.detectUnusedParametersAndConstants(bDAE)
    #= Find and reclassify discrete variables not marked as discrete. =#
    @BACKEND_LOGGING debugWrite(logPath("backend/bdae", "bdae_residualTransformation.log"), BDAEUtil.stringHeading1(bDAE, "residuals"))
    return bDAE
  end
  return hasActiveLogRunDir() ? lowerWork() : withLogRunDir(lowerWork, runId)
end

"""
  Transforms  BDAE-IR to simulation code for DAE-mode.
  If `eliminateNonDynamic` is provided, output-only variables are eliminated after SimCode creation.
"""
function generateSimulationCode(bDAE::BDAE.BACKEND_DAE;
                                mode)::SimulationCode.SimCode
  local simCode = SimulationCode.transformToSimCode(bDAE; mode = mode)
  @info "[SIMCODE: $(simCode.name)] transformed" residuals=length(simCode.residualEquations) initial=length(simCode.initialEquations) ifEquations=length(simCode.ifEquations) variables=length(simCode.stringToSimVarHT)
  #= NOTE: propagateConstants, eliminateAliasVariables, and eliminateOutputOnlyVariables
     are now called from translate() AFTER flattenRecordCallSites, so that record
     argument expansion does not introduce dangling references to already-eliminated
     variables, and output-only elimination sees the fully simplified system. =#
  return simCode
end

"""
  Translates functions to simulation code.
"""
function generateSimCodeFunctions(functions::List{OMFrontend.Frontend.M_FUNCTION})
  local simCodeFunctions = SimulationCode.generateSimCodeFunctions(functions)
  return simCodeFunctions
end

"""
  Generates code interfacing DifferentialEquations.jl
  The resulting code is saved in a dictionary which contains functions that were simulated
  this session. Returns the generated modelName and corresponding generated code
"""
function generateTargetCode(simCode::SimulationCode.SIM_CODE)
  #= Target code =#
  (modelName::String, modelCode::Expr) = CodeGeneration.generateCode(simCode)
  @debug "[CODEGEN] generated target code" model=modelName codeHash=hash(modelCode)
  COMPILED_MODELS[modelName] = modelCode
  return (modelName, modelCode)
end

"""
`generateMTKTargetCode(simCode::SimulationCode.SIM_CODE)`
  Generates code interfacing ModelingToolkit.jl
  The resulting code is saved in a table which contains functions that were simulated
  this session. Returns the generated modelName and corresponding generated code
"""
function generateMTKTargetCode(simCode::SimulationCode.SIM_CODE)
  #= Target code =#
  (modelName::String, modelCode::Expr) = CodeGeneration.generateMTKCode(simCode)
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
    local moduleAlreadyExists = isdefined(OMBackend, Symbol(modelName))
    COMPILED_MODELS_MTK[modelName] = (modelCode, moduleAlreadyExists, codeHash)
  end
  return (modelName, modelCode)
end

"""
`generateIMTKTargetCode(simCode)` — like `generateMTKTargetCode`, but via
`IMTKGen.generateIMTKCode`, which also builds + structurally simplifies the
System in the backend and dumps it. Caches into `COMPILED_MODELS_MTK`.
"""
function generateIMTKTargetCode(simCode::SimulationCode.SIM_CODE)
  #= IMTKGen.generateIMTKCode evals the (current) module while building, so the
     loaded module is already fresh: record changeDetected = false so a later
     MTK-mode simulate on this model does not needlessly re-eval the module. =#
  (modelName::String, modelCode::Expr) = IMTKGen.generateIMTKCode(simCode)
  local codeHash = hash(modelCode)
  @info "[IMTK GEN] generated target code" model=modelName codeHash=codeHash
  COMPILED_MODELS_MTK[modelName] = (modelCode, false, codeHash)
  return (modelName, modelCode)
end

function getCompiledModel(modelName)
  try
    return COMPILED_MODELS_MTK[modelName][1]
  catch e
    @error "Model: $(modelName) is not compiled. Available models are: $(availableModels())"
    throw(e)
  end
end

"""
`generateDETargetCode(simCode::SimulationCode.SIM_CODE)`
  Generates code interfacing DifferentialEquations.jl directly (DEMode).
  Produces an in-place ODEProblem with the RHS indexed by integer position.
  Caches the result in COMPILED_MODELS_DEJL so simulateModel can pick it up.
"""
function generateDETargetCode(simCode::SimulationCode.SIM_CODE)
  (modelName::String, modelCode::Expr) = CodeGeneration.generateDECode(simCode)
  local codeHash = hash(modelCode)
  if haskey(COMPILED_MODELS_DEJL, modelName)
    local previousHash = COMPILED_MODELS_DEJL[modelName][3]
    local changeDetected = previousHash != codeHash
    COMPILED_MODELS_DEJL[modelName] = (modelCode, changeDetected, codeHash)
  else
    local moduleAlreadyExists = isdefined(OMBackend, Symbol(modelName))
    COMPILED_MODELS_DEJL[modelName] = (modelCode, moduleAlreadyExists, codeHash)
  end
  return (modelName, modelCode)
end

"""
  Returns true if the model was compiled again
"""
function modelWasCompiledAgain(modelName)
  return COMPILED_MODELS_MTK[modelName][2]
end

function getCompiledModelDE(modelName)
  try
    return COMPILED_MODELS_DEJL[modelName][1]
  catch e
    local available = join(keys(COMPILED_MODELS_DEJL), ", ")
    @error "DE-mode model: $(modelName) is not compiled. Available DE-mode models: $(available)"
    throw(e)
  end
end

modelWasCompiledAgainDE(modelName) = COMPILED_MODELS_DEJL[modelName][2]

"""

```
writeModelToFile(modelName::String, filePath::String; keepComments = true, keepBeginBlocks = true)
```
  Writes a model to file by default the file is formatted and comments are kept.
"""
function writeModelToFile(modelName::String, filePath::String; keepComments = true, keepBeginBlocks = true)
  model = getCompiledModel(modelName)
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
  try
    local model::Expr
    model = getCompiledModel(modelName)
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
      @warn "Julia Formatter failed to format the output results due to $(e)"
      formattedResults = modelStr
    end
    return formattedResults
  catch e
    @error "Model: $(modelName) is not compiled.\n Available models are: $(availableModels())" exception=(e, catch_backtrace())
  end
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
                       solver = Rodas5(),
                       kwargs...)
  ```
  Simulates model interactively.
  The solver need to be passed with a : before the name, example:
  OMBackend.simulateModel(modelName, tspan = (0.0, 1.0), solver = :(Tsit5()));
"""
function simulateModel(modelName::String;
                       MODE = DEFAULT_BACKEND_MODE[],
                       tspan = (0.0, 1.0),
                       solver = Rodas5(autodiff=false),
                       overwriteCache::Bool = false,
                       kwargs...)
  modelName = canonicalName(modelName)
  local modelCode::Expr
  if MODE == MTK_MODE
    #= This does a redundant string conversion for now due to modeling toolkit being as is...=#
    try
      modelCode = getCompiledModel(modelName)
    catch err
      println("Failed to simulate model.")
      println("Available models are:")
      availableModels()
    end
    try
      #= Only re-eval if the module does not exist yet or the code changed =#
      local needsEval = overwriteCache || !isdefined(OMBackend, Symbol(modelName)) || modelWasCompiledAgain(modelName)
      if needsEval
        @eval $modelCode
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
      @error "Interactive evaluation failed" exception_type=typeof(err) mode=MODE model=modelName
      rethrow(err)
    end
  elseif MODE == IMTK_MODE
    #= Reuse the build cached in the backend at translate time (no
       structural_simplify re-run); simulateIMTK falls back to module simulate. =#
    local _runDir = get(MODEL_RUN_DIRS, modelName, nothing)
    local _doSim = () -> IMTKGen.simulateIMTK(modelName, tspan, solver; kwargs...)
    try
      return _runDir === nothing ? _doSim() : withLogRunDir(_doSim, _runDir)
    catch err
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
      local needsEval = overwriteCache || !isdefined(OMBackend, Symbol(modelName)) || modelWasCompiledAgainDE(modelName)
      if needsEval
        @eval $modelCode
      end
      Base.invokelatest() do
        local mod = getfield(OMBackend, Symbol(modelName))
        mod.simulate(tspan, solver; kwargs...)
      end
    catch err
      @error "Interactive evaluation failed" exception_type=typeof(err) mode=MODE model=modelName
      rethrow(err)
    end
  else
    error("Unsupported mode: $(MODE)")
  end
end

"""
    getMTKProblem(modelName; tspan=(0.0, 1.0), overwriteCache=false)

Return the MTK ODEProblem for an already-translated model without solving it.
Call `OM.translate` first, then use this to inspect the problem.

# Example
```julia
OM.translate("Modelica.Mechanics.MultiBody.Examples.Elementary.Pendulum")
prob = OMBackend.getMTKProblem("Modelica.Mechanics.MultiBody.Examples.Elementary.Pendulum")
```
"""
function getMTKProblem(modelName::String;
                       tspan = (0.0, 1.0),
                       overwriteCache::Bool = false)
  modelName = canonicalName(modelName)
  local modelCode::Expr
  try
    modelCode = getCompiledModel(modelName)
  catch err
    error("Model $(modelName) is not compiled. Call OMBackend.translate first. Available: $(availableModels())")
  end
  local needsEval = overwriteCache || !isdefined(OMBackend, Symbol(modelName)) || modelWasCompiledAgain(modelName)
  if needsEval
    @eval $modelCode
  end
  Base.invokelatest() do
    local mod = getfield(OMBackend, Symbol(modelName))
    local modelFn = getfield(mod, Symbol(string(modelName, "Model")))
    modelFn(tspan)
  end
end

"""
  Resimulates an already compiled model given a model that is already active in th environment
  along with a set of parameters as key value pairs.
"""
function resimulateModel(modelName::String;
                         solver = Rodas5(autodiff=false),
                         MODE = DEFAULT_BACKEND_MODE[],
                         tspan=(0.0, 1.0),
                         parameters::Dict = Dict())
  #=
  Check if a compiled instance of the model already exists in the backend.
  If that is the case we do not have to recompile it.
  =#
  try
    modelName = canonicalName(modelName)
    Base.invokelatest() do
      local mod = getfield(OMBackend, Symbol(modelName))
      mod.simulate(tspan, solver)
    end
  catch e
    availModels = availableModels()
    @error "The model $(modelName) is not compiled. Available models are: $(availModels)" exception=(e, catch_backtrace())
  end
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
      catch
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
    if isdefined(OMBackend, modSym)
      local mod = getfield(OMBackend, modSym)
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
  if !isdefined(OMBackend, modSym)
    return nothing
  end
  local mod = getfield(OMBackend, modSym)
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
