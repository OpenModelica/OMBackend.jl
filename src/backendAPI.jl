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
end

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
TODO: Optimaly we should keep the frontend structure
in memory as well s.t we only recompile if the structure of the source file changes
(Unless retranslation is forced).
"""
const COMPILED_MODELS_MTK = Dict{String, Tuple{Expr, Bool, UInt64}}()


"""
DEJL (DifferentialEquations.jl) models that have been compiled one time.
"""
const COMPILED_MODELS_DEJL = Dict{String, Tuple{Expr, Bool, UInt64}}()

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
function translate(frontendDAE::Union{DAE.DAE_LIST, OMFrontend.Frontend.FlatModel};
                   functionList = nothing,
                   BackendMode = MTK_MODE,
                   warnMissingStartValues = nothing,
                   eliminateNonDynamic::Union{Nothing, Bool, SimulationCode.EliminationOptions} = nothing,
                   observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                   checkSimCode::Bool = true)::Tuple{String, Expr}
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
      local bDAE = lower(frontendDAE)
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
        @assign simCode.functions = simCodeFunctions
        @assign simCode.externalRuntime = externalRuntimeNeeded
        simCodeFunctions = SimulationCode.flattenRecordParameters(simCodeFunctions)
        @assign simCode.functions = simCodeFunctions
        simCode = SimulationCode.flattenRecordCallSites(simCode)
        simCode = SimulationCode.resolveIfExpInBindings!(simCode)
        simCode = SimulationCode.foldParameterClosure(simCode)
        simCode = SimulationCode.propagateConstants(simCode)
        simCode = SimulationCode.eliminateAliasVariables(simCode)
        if !isempty(simCode.structuralTransitions) || !isempty(simCode.subModels)
          error("DEMode does not support structural transitions / VSS at this time. " *
                "Re-run with mode = OMBackend.MTK_MODE.")
        end
        return generateDETargetCode(simCode)
      elseif BackendMode == MTK_MODE
        #@debug "Generate simulation code"
        simCode = generateSimulationCode(bDAE; mode = MTK_MODE)
        (simCodeFunctions, externalRuntimeNeeded) = if functionList !== nothing
          generateSimCodeFunctions(functionList)
        else
          (SimulationCode.ModelicaFunction[], false)
        end
        @assign simCode.functions = simCodeFunctions
        @assign simCode.externalRuntime = externalRuntimeNeeded
        #= Dump before record flattening =#
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_initial.log"), SimulationCode.dumpSimCode(simCode))
        #= Flatten record parameters in functions =#
        simCodeFunctions = SimulationCode.flattenRecordParameters(simCodeFunctions)
        @assign simCode.functions = simCodeFunctions
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFlattenRecordParams.log"), SimulationCode.dumpSimCode(simCode))
        #= Flatten record arguments in equation call sites to match flattened signatures =#
        simCode = SimulationCode.flattenRecordCallSites(simCode)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFlattenRecordCallSites.log"), SimulationCode.dumpSimCode(simCode))
        #= Resolve constant-condition IFEXPs in parameter bindings =#
        simCode = SimulationCode.resolveIfExpInBindings!(simCode)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterResolveIfExp.log"), SimulationCode.dumpSimCode(simCode))
        #= Constant propagation and alias elimination run AFTER record flattening
           so that all CREF references are in their final form before substitution.
           Running these earlier caused dangling references when flattenRecordCallSites
           introduced new CREFs for already-eliminated variables. =#
        #= BLT-driven parameter-closure fold runs before propagateConstants so that
           any parameter-closure chains (e.g. KinematicPTP's seven algebraic unknowns
           defined solely by parameter expressions) are promoted to parameters before
           MTK sees them. This removes the Newton-init failure mode where zero guesses
           on those unknowns produce NaN/Inf evaluations. =#
        simCode = SimulationCode.foldParameterClosure(simCode)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterFoldClosure.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.inlinePreOfConstantParameters(simCode)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterInlinePreParam.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.propagateConstants(simCode)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterConstantProp.log"), SimulationCode.dumpSimCode(simCode))
        simCode = SimulationCode.eliminateAliasVariables(simCode)
        @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterAliasElimination.log"), SimulationCode.dumpSimCode(simCode))
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
          simCode = SimulationCode.eliminateOutputOnlyVariables(simCode, elimOpts)
          @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "simCode_afterElimination.log"), SimulationCode.dumpSimCode(simCode))
        end
        #= Observed filter: controls which alias-eliminated variables become observed equations.
           Default (nothing): skip ALL alias observed equations for fast compilation.
           With filter patterns: keep only matching aliases as observed. =#
        if observedFilter === nothing
          #= Fast default: no observed equations from alias elimination =#
          if !isempty(simCode.aliasMap)
            @info "observedFilter: clearing $(length(simCode.aliasMap)) alias observed equations (default: none)"
            @assign simCode.aliasMap = empty(simCode.aliasMap)
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
            @info "observedFilter: kept $(length(filteredMap)) of $originalCount alias observed equations"
          end
        end
        if checkSimCode
          local checkResult = SimulationCode.SimCodeCheck.check(simCode)
          SimulationCode.SimCodeCheck.report(stderr, checkResult)
        end
        return generateMTKTargetCode(simCode)
      else
        @error "Unsupported BackendMode: $BackendMode. Valid modes are: MTK_MODE, DEMode"
      end
    end
  finally
    WARN_MISSING_START_VALUES[] = previousWarnSetting
  end
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
  local runId = createLogRunId(logRunModelName(frontendDAE))
  local lowerWork = function()
    local bDAE::BDAE.BACKEND_DAE
    @debug "Length of frontend DAE:" length(frontendDAE.elementLst)
    @assert typeof(listHead(frontendDAE.elementLst)) == DAE.COMP
    #= Create Backend structure from Frontend structure =#
    bDAE = BDAECreate.lower(frontendDAE)
    @debug(BDAEUtil.stringHeading1(bDAE, "translated"));
    #= Transform ASUB expressions: der(array)[i] to der(array[i]) =#
    bDAE = Causalize.transformASUBExpressions(bDAE)
    #= Mark state variables =#
    bDAE = Causalize.detectStates(bDAE)
    @debug(BDAEUtil.stringHeading1(bDAE, "states marked"));
    #= Flatten CREF_QUAL with subscripted array finals =#
    bDAE = Causalize.flattenArrayCrefs(bDAE)
    #= Transform if expressions to if equations =#
    bDAE = Causalize.detectIfExpressions(bDAE)
    @debug(BDAEUtil.stringHeading1(bDAE, "if equations transformed"));
    #= Expand record field array variables into individual scalar element variables =#
    bDAE = Causalize.expandRecordFieldArrays(bDAE)
    #= Expand COMPLEX_EQUATIONs into scalar equations =#
    bDAE = Causalize.expandComplexEquations(bDAE)
    #= We always residualize since residuals are easier to work with =#
    bDAE = Causalize.residualizeEveryEquation(bDAE)
    @debug(BDAEUtil.stringHeading1(bDAE, "residuals"));
    return bDAE
  end
  return hasActiveLogRunDir() ? lowerWork() : withLogRunDir(lowerWork, runId)
end


"""
  Transforms given FlatModelica to backend DAE-IR (BDAE-IR).
"""
function lower(fm::OMFrontend.Frontend.FLAT_MODEL)
  local runId = createLogRunId(logRunModelName(fm))
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
      TODO: also check bindings of all parameters before readding this call.
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
  @debug BDAEUtil.stringHeading1(simCode, "SIM_CODE: transformed simcode")
  #= NOTE: propagateConstants, eliminateAliasVariables, and eliminateOutputOnlyVariables
     are now called from translate() AFTER flattenRecordCallSites, so that record
     argument expansion does not introduce dangling references to already-eliminated
     variables, and output-only elimination sees the fully simplified system. =#
  return simCode
end

"""
  Translates functions to simlulation code.
"""
function generateSimCodeFunctions(functions::List{OMFrontend.Frontend.M_FUNCTION})
  local simCodeFunctions = SimulationCode.generateSimCodeFunctions(functions)
  return simCodeFunctions
end

"""
  Generates code interfacing DifferentialEquations.jl
  The resulting code is saved in a dictonary which contains functions that where simulated
  this session. Returns the generated modelName and corresponding generated code
"""
function generateTargetCode(simCode::SimulationCode.SIM_CODE)
  #= Target code =#
  (modelName::String, modelCode::Expr) = CodeGeneration.generateCode(simCode)
  @debug "Functions:" modelCode
  @debug "Model:" modelName
  #= TODO: This replacement should ideally be done earlier. Or be solved in a nicer way. =#
  modelName = replace(modelName, "." => "__")
  COMPILED_MODELS[modelName] = modelCode
  return (modelName, modelCode)
end

"""
`generateMTKTargetCode(simCode::SimulationCode.SIM_CODE)`
  Generates code interfacing ModelingToolkit.jl
  The resulting code is saved in a table which contains functions that where simulated
  this session. Returns the generated modelName and corresponding generated code
"""
function generateMTKTargetCode(simCode::SimulationCode.SIM_CODE)
  #= Target code =#
  (modelName::String, modelCode::Expr) = CodeGeneration.generateMTKCode(simCode)
  @debug "Functions:" modelCode
  @debug "Model:" modelName
  modelName = replace(modelName, "." => "__")
  local codeHash = hash(modelCode)
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
  modelName = replace(modelName, "." => "__")
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
    #= Remove all the redudant blocks from the model =#
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
                       MODE = MTK_MODE,
                       tspan = (0.0, 1.0),
                       solver = Rodas5(),
                       kwargs...)
  ```
  Simulates model interactivly.
  The solver need to be passed with a : before the name, example:
  OMBackend.simulateModel(modelName, tspan = (0.0, 1.0), solver = :(Tsit5()));
"""
function simulateModel(modelName::String;
                       MODE = MTK_MODE,
                       tspan = (0.0, 1.0),
                       solver = Rodas5(autodiff=false),
                       overwriteCache::Bool = false,
                       kwargs...)
  #= Strings using "." need to be in a format suitable for Julia =#
  modelName = replace(modelName, "." => "__")
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
      #= Run in latest world age to see the just-eval'd module =#
      Base.invokelatest() do
        local mod = getfield(OMBackend, Symbol(modelName))
        mod.simulate(tspan, solver; kwargs...)
      end
    catch err
      @error "Interactive evaluation failed" exception_type=typeof(err) mode=MODE model=modelName
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
  modelName = replace(modelName, "." => "__")
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
                         MODE = MTK_MODE,
                         tspan=(0.0, 1.0),
                         parameters::Dict = Dict())
  #=
  Check if a compiled instance of the model already exists in the backend.
  If that is the case we do not have to recompile it.
  =#
  try
    modelName = replace(modelName, "." => "__")
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
    Symbol(replace(varName, "." => "__"))
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
        Symbol(replace(v, "." => "__"))
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
    local modSym = Symbol(replace(modelName, "." => "__"))
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
  local modSym = Symbol(replace(modelName, "." => "__"))
  if !isdefined(OMBackend, modSym)
    return nothing
  end
  local mod = getfield(OMBackend, modSym)
  if !isdefined(mod, :LATEST_REDUCED_SYSTEM)
    return nothing
  end
  local sys_t = ModelingToolkit.get_iv(mod.LATEST_REDUCED_SYSTEM)
  local result = Dict{String, Vector{Float64}}()
  local warnedUnresolved = Set{String}()
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
