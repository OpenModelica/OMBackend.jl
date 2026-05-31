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

#=
  Author: John Tinnerholm
  TODO: Remember the state derivative scheme. What did I mean with that?
  TODO: Make duplicate code better...
  TODO: Cleanup in general.. keep this simple. Remove hacks as the SCiML team adds new features to MTK.
=#
import ..OMBackend
import .AlgorithmicCodeGeneration

#= Size of each emitted helper chunk in `decompose*` and `generate*Block` paths.
   Tunable at runtime via `OMBackend.CodeGeneration.CHUNK_SIZE[] = N`. =#
const CHUNK_SIZE = Ref{Int}(50)

#= Julia-AST symbols used by the discrete-dummy demotion pattern matchers in
   ODE_MODE_MTK_MODEL_GENERATION. Promoted to module-level constants so a
   rename upstream (e.g. `floor` → `modelica_floor`) is a single-line change
   here instead of a scatter-hunt across closures. =#
const DERIVATIVE_HEADS        = (:der, :D)
const INTEGER_DEF_HEADS       = (:integer, :modelica_integer, :floor)
const COMPARISON_OPS          = (:<, :<=, :>, :>=, :(==), :(!=))
const IFELSE_HEAD             = :ifelse
const CONST_TABLE_LOOKUP_HEAD = :constTableLookup

"""
    evalGeneratedFunctionsAndRegister!(modelName, functions, simCode)

Phase A of `ODE_MODE_MTK_MODEL_GENERATION`: `eval` each generated Modelica
function body in OMBackend, then `eval` the `@register_symbolic` calls that
make Symbolics aware of them.

The eval must happen here (not at simulate time) because subsequent codegen
phases need the function bindings to exist when they construct symbolic
equation expressions.

On function-eval failure, the offending generated source is dumped to
`/tmp/om_bad_function.jl` and the error rethrown. Register-call failures
are tolerated when the binding "already has a value" (re-registration is
idempotent) and rethrown otherwise.
"""
function evalGeneratedFunctionsAndRegister!(modelName, functions, simCode)
  for f in functions
    try
      eval(f)
    catch e
      local dumpPath = "/tmp/om_bad_function.jl"
      try
        open(dumpPath, "w") do io
          println(io, "# Offending generated function. Eval error: ", sprint(showerror, e))
          println(io, "# Model: $modelName")
          println(io, string(Base.remove_linenums!(deepcopy(f))))
        end
        @error "Generated Modelica-function eval failed; dumped Julia source for inspection" modelName error=sprint(showerror, e) dumpPath
      catch ioErr
        @error "Generated function eval failed, also failed to dump" modelName error=sprint(showerror, e) ioErr
      end
      rethrow(e)
    end
  end
  local registrationCalls = generateRegisterCallsForCallExprs(simCode; funcArgGen = AlgorithmicCodeGeneration.generateIOL)
  for regCall in registrationCalls
    try
      eval(regCall)
    catch e
      contains(string(e), "already has a value") || rethrow(e)
    end
  end
  return nothing
end

"""
    ClassifiedVariables

Result of `classifyVariables` — every simvar in `simCode.stringToSimVarHT`
bucketed by `varKind`, plus the StateSelect priority pairs MTK needs.

The buckets are deliberately the names downstream phases use, so the
unpacking at the call site reads as a phase index.
"""
struct ClassifiedVariables
  stateVariables        :: Vector{String}
  algebraicVariables    :: Vector{String}
  discreteVariables     :: Vector{String}
  occVariables          :: Vector{String}
  parameters            :: Vector{String}
  arrayParameters       :: Vector{String}
  stateDerivatives      :: Vector{String}
  dataStructureVariables:: Vector{String}
  statePriorityPairs    :: Vector{Tuple{Symbol, Int}}
end

"""
    classifyVariables(simCode) -> ClassifiedVariables

Phase B of `ODE_MODE_MTK_MODEL_GENERATION`: walk `simCode.stringToSimVarHT`
once and bucket each variable by its `varKind`. `ALG_VARIABLE` has the
most subtle fall-through:

- match-order present → algebraic.
- involved in an event → discrete.
- system singular → algebraic (needs index reduction).
- otherwise → flag system singular and still call it algebraic.

Also extracts the per-variable `StateSelect` annotation (`NEVER`,
`AVOID`, `PREFER`, `ALWAYS`) into MTK state-priority pairs, but only for
variables that will become real MTK unknowns (states / algebraic / occ /
array). Helper parameters like `*_start` may carry stateSelect from
source attributes but never become MTK variables, so emitting a priority
for them would `UndefVarError` at the batched eval.
"""
function classifyVariables(simCode)::ClassifiedVariables
  local stateVariables         = String[]
  local algebraicVariables     = String[]
  local discreteVariables      = String[]
  local occVariables           = String[]
  local parameters             = String[]
  local arrayParameters        = String[]
  local stateDerivatives       = String[]
  local dataStructureVariables = String[]
  local statePriorityPairs     = Tuple{Symbol, Int}[]
  local ht = simCode.stringToSimVarHT
  for varName in keys(ht)
    (idx, var) = ht[varName]
    local varType = var.varKind
    @match varType begin
      SimulationCode.INPUT(__) => begin
        @error "INPUT not supported in CodeGen"
        throw()
      end
      SimulationCode.STATE(__) => push!(stateVariables, varName)
      SimulationCode.OCC_VARIABLE(__) => push!(occVariables, varName)
      SimulationCode.PARAMETER(__) => push!(parameters, varName)
      #= String parameters are non-numeric; excluded from MTK parameter system. =#
      SimulationCode.STRING(__) => nothing
      SimulationCode.ARRAY_PARAMETER(__) => push!(arrayParameters, varName)
      SimulationCode.ARRAY(__) => push!(stateVariables, varName)
      SimulationCode.DISCRETE(__) => push!(discreteVariables, varName)
      SimulationCode.ALG_VARIABLE(__) => begin
        if idx in simCode.matchOrder
          push!(algebraicVariables, varName)
        elseif involvedInEvent(idx, simCode)
          push!(discreteVariables, varName)
        elseif simCode.isSingular
          push!(algebraicVariables, varName)
        else
          @assign simCode.isSingular = true
          push!(algebraicVariables, varName)
        end
      end
      SimulationCode.DATA_STRUCTURE(__) => push!(dataStructureVariables, varName)
      SimulationCode.STATE_DERIVATIVE(__) => push!(stateDerivatives, varName)
    end
    #= StateSelect → MTK state_priority, only on actual MTK unknowns. =#
    local optAttrs::Option{DAE.VariableAttributes} = var.attributes
    local priority = @match optAttrs begin
      SOME(attrs && DAE.VAR_ATTR_REAL(__)) => begin
        @match attrs.stateSelectOption begin
          SOME(DAE.NEVER(__))  => -10
          SOME(DAE.AVOID(__))  => -2
          SOME(DAE.PREFER(__)) => 2
          SOME(DAE.ALWAYS(__)) => 10
          _                    => nothing
        end
      end
      _ => nothing
    end
    if priority !== nothing
      local supportsStatePriority =
        varType isa SimulationCode.STATE ||
        varType isa SimulationCode.ALG_VARIABLE ||
        varType isa SimulationCode.OCC_VARIABLE ||
        varType isa SimulationCode.ARRAY
      if supportsStatePriority && !startswith(string(varName), "der(")
        push!(statePriorityPairs, (Symbol(varName), priority))
      end
    end
  end
  return ClassifiedVariables(stateVariables, algebraicVariables,
                             discreteVariables, occVariables,
                             parameters, arrayParameters,
                             stateDerivatives, dataStructureVariables,
                             statePriorityPairs)
end

"""
    buildIfEquationEventDecl(events::Vector{Expr}) -> Expr

Wrap the collected `SymbolicContinuousCallback` expressions in the
`events = ...` assignment that the generated model module expects.
`Base.invokelatest` is needed because the event exprs reference variables
created via `eval` earlier in the model function body, so they must run
in the new world age.
"""
function buildIfEquationEventDecl(events::Vector{Expr})::Expr
  isempty(events) && return :(events = [])
  return :(events = Base.invokelatest(() -> [$(events...)]))
end

"""
    collectIrreducibleSymbols(simCode, conditionalEquations,
                              stateVariables, algebraicVariables, occVariables)
        -> Vector{Symbol}

Build the list of variable symbols that MTK's tearing pass must NOT
eliminate. Sources:

1. `simCode.irreducibleVariables` — names the SimCode pass already flagged.
2. The LHS of every `ifEq_tmpN ~ ifelse(...)` conditional equation —
   if MTK eliminates the LHS, the if-equation lowering breaks.
3. Variables with `fixed = true` and an explicit start value — the init
   constraint emitted by `getFixedStartConstraintsMTK` must land on a
   surviving unknown, so the symbol cannot be torn.

ifCond discrete parameters are NOT in this list: they are parameters,
not unknowns, so MTK never tries to eliminate them in the first place.
"""
function collectIrreducibleSymbols(simCode,
                                   conditionalEquations::Vector{Expr},
                                   stateVariables::Vector{String},
                                   algebraicVariables::Vector{String},
                                   occVariables::Vector{String})::Vector{Symbol}
  local syms = Symbol[Symbol(vn) for vn in simCode.irreducibleVariables]
  for ceq in conditionalEquations
    if ceq isa Expr && ceq.head == :call && length(ceq.args) >= 2
      local lhs = ceq.args[2]
      lhs isa Symbol && push!(syms, lhs)
    end
  end
  for vn in fixedStartVarNames(vcat(stateVariables, algebraicVariables, occVariables), simCode)
    local sym = Symbol(vn)
    sym in syms || push!(syms, sym)
  end
  return syms
end

#= ---- ODEProblem-construction strategies ----

   At codegen time the function picks one of three strategies for building
   the SciML ODEProblem. Each strategy lives in its own emitter so the
   WHY-comment for each lives next to the code it justifies, and the
   final call site reads as `problem = $(emitProblemConstruction(...))`. =#

"""
    emitDirectRHSProblem()

Strategy 1: build the problem via `OMBackend.CodeGeneration.buildDirectRHSProblem`.
Used when `useDirectRHS == true`. Skips MTK's standard `ODEProblem`
constructor in favor of the direct-RHS path.
"""
emitDirectRHSProblem() = :(
  problem = OMBackend.CodeGeneration.buildDirectRHSProblem(
    reducedSystem, finalInitialValues, pars, tspan, callbacks;
    allInitialValues = initialValues)
)

"""
    emitStructuralTransitionProblem()

Strategy 2: structural-transition submodel. The codegen-time decision is
already made — we know we should skip MTK's initialization problem — but
the choice between pure-ODE and DAE paths depends on the mass matrix,
which only exists at simulate time. So this emitter returns an `Expr`
that dispatches at runtime:

- Pure ODE (identity mass matrix): all unknowns are differential, no
  constraints to solve. Provide u0 for ALL unknowns (filling algebraic
  defaults via `buildDefaultGuesses` at 0.0) and skip the initialization
  solver. Preserves the fast path for models such as BouncingBall and
  FreeFall.
- DAE (singular mass matrix, e.g. Pendulum with algebraic `x = L*sin(phi)`
  constraints): `splitInitialValues` has already pinned explicit-start
  algebraic IVs as hard u0 and registered 0.0 soft guesses for uncovered
  differential states on `reducedSystem.guesses`. Pass only
  `finalInitialValues` as u0 and let MTK's initializer solve the algebraic
  residuals consistently. Injecting `_missingU0` as hard u0 here would
  override the guesses (e.g. phi=0 instead of phi=3π/4) and silently
  violate the constraint, so it must not be merged.
"""
emitStructuralTransitionProblem() = quote
  local _isPureODE = Base.invokelatest(
    OMBackend.CodeGeneration.isPureODESystem, reducedSystem)
  if _isPureODE
    local _missingU0 = Base.invokelatest(
      OMBackend.CodeGeneration.buildDefaultGuesses, reducedSystem, finalInitialValues, initialValues)
    problem = ModelingToolkit.ODEProblem(reducedSystem,
                                         merge(Dict(finalInitialValues), _missingU0, pars),
                                         tspan;
                                         callback = callbacks,
                                         warn_initialize_determined = false,
                                         build_initializeprob = false)
  else
    problem = ModelingToolkit.ODEProblem(reducedSystem,
                                         merge(Dict(finalInitialValues), pars),
                                         tspan;
                                         callback = callbacks,
                                         warn_initialize_determined = false)
  end
end

"""
    emitInitSolveDAEProblem()

Strategy 3: standard DAE-with-init-solver. Force MTK to build the
initialization problem and solve as NLS so algebraic states pinned via
`initialization_eqs` are honoured even when system algebraic residuals
would otherwise pull them to a different consistent root. Without
`fully_determined = false` MTK can sacrifice a `var ~ start` init
residual against many algebraic residuals; without
`build_initializeprob = true` MTK may skip the init solve entirely and
leave `prob.u0` inconsistent with the init eqs.
"""
emitInitSolveDAEProblem() = :(
  problem = ModelingToolkit.ODEProblem(reducedSystem,
                                       merge(Dict(finalInitialValues), pars),
                                       tspan;
                                       callback = callbacks,
                                       warn_initialize_determined = false,
                                       build_initializeprob = true,
                                       fully_determined = false)
)

"""
    emitProblemConstruction(useDirectRHS::Bool, skipInitializeProb::Bool) -> Expr

Pick the right ODEProblem-construction `Expr` for the codegen-time strategy
combination. Three mutually exclusive strategies; see
`emitDirectRHSProblem`, `emitStructuralTransitionProblem`,
`emitInitSolveDAEProblem` for the WHY of each.
"""
function emitProblemConstruction(useDirectRHS::Bool, skipInitializeProb::Bool)::Expr
  useDirectRHS         && return emitDirectRHSProblem()
  skipInitializeProb   && return emitStructuralTransitionProblem()
  return emitInitSolveDAEProblem()
end

"""
    IfEquationComponent

Codegen artifacts for one Modelica `if`-equation that has been lifted to an
MTK event + residual pair. Produced by `createIfEquation`, consumed by
`ODE_MODE_MTK_MODEL_GENERATION`.

# Fields
- `events`              : `Vector{Expr}` — one `SymbolicContinuousCallback`
                          per branch condition. Each callback flips one of
                          this if-equation's `ifCondN` discrete parameters
                          at the branch's zero crossing.
- `conditionalEquations`: `Vector{Expr}` — residual rewrites of the form
                          `lhs ~ ifelse(ifCondN == 1, thenExpr, elseExpr)`,
                          one per LHS variable the if-equation touches.
- `conditionVariables`  : `Vector{Symbol}` — the `:ifCondNI` parameter
                          symbols introduced for this if-equation. Marked
                          irreducible at codegen time so MTK does not tear
                          them.
- `conditionNameAndIV`  : `Vector{Tuple{String, Bool}}` — `(name, initialValue)`
                          pairs used to declare the discrete parameters with
                          their compile-time initial values.
"""
struct IfEquationComponent
  events               :: Vector{Expr}
  conditionalEquations :: Vector{Expr}
  conditionVariables   :: Vector{Symbol}
  conditionNameAndIV   :: Vector{Tuple{String, Bool}}
  #= Deferred pure-time-event branches: (ifCondSym, zeroCrossingLHS, mtkConditionEq,
     postCrossingValue). `createIfEquations` builds one refresh callback per entry;
     each fires at its own threshold, sets its own ifCond to the post-crossing value,
     and re-derives the OTHER pure-time ifConds from their zero-crossing sign so that
     coincident time events cannot drop one another's affect. =#
  pureTimeEvents       :: Vector{Tuple{Symbol, Any, Any, Float64}}
end

"""
  Generates simulation code targeting modeling toolkit.
  Loop code removed was on old branch.
"""
function generateMTKCode(simCode::SimulationCode.SIM_CODE)
  isCycles = isCycleInSCCs(simCode.stronglyConnectedComponents)
  ODE_MODE_MTK(simCode::SimulationCode.SIM_CODE)
end

"""
  The entry point of MTK code generation.
  Either calls ODE_MODE_MTK_PROGRAM_GENERATION
  or do code generation for a model with structural submodels.
"""
function ODE_MODE_MTK(simCode::SimulationCode.SIM_CODE)
  #=If our model name is separated by . replace it with __ =#
  local MODEL_NAME = simCode.name
  #= Generate code for algorithmic Modelica =#
  (functions, functionNames) = AlgorithmicCodeGeneration.generateFunctions(simCode.functions)
  if !SimulationCode.hasStructuralTransitions(simCode) && !SimulationCode.hasSubModels(simCode) && !SimulationCode.hasFlatModel(simCode)
    #= Generate using the standard name =#
    return ODE_MODE_MTK_PROGRAM_GENERATION(simCode, simCode.name, functions)
  end
  #= Handle structural submodels =#
  local activeModelSimCode = getActiveModel(simCode)
  local activeModelName = simCode.activeModel
  local structuralModes = Expr[]
  for mode in simCode.subModels
    push!(structuralModes, ODE_MODE_MTK_MODEL_GENERATION(mode, mode.name, functions; useDirectRHS = false))
  end
  if isempty(simCode.subModels)
    local modelName = string(MODEL_NAME, "DEFAULT")
    defaultModel = ODE_MODE_MTK_MODEL_GENERATION(simCode, modelName, functions; useDirectRHS = false)
    activeModelName = modelName
    push!(structuralModes, defaultModel)
  end
  local structuralCallbacks = createStructuralCallbacks(simCode, simCode.structuralTransitions)
  local structuralAssignments = createStructuralAssignments(simCode, simCode.structuralTransitions)
  #=
  Initialize array where the common variables are stored.
  That is variables all modes have
  =#
  local commonVariables = createCommonVariables(simCode.sharedVariables)
  #= Collect DATA_STRUCTURE (Modelica constant) assignments for module-level emission.
     Without these, parameter binding expressions that reference MSL constants
     (e.g. Modelica.Mechanics.MultiBody.Types.Defaults.*) would fail at runtime
     because the symbols are never defined in the generated module scope. =#
  local _dsVarNames = String[]
  for varName in keys(simCode.stringToSimVarHT)
    (_, var) = simCode.stringToSimVarHT[varName]
    @match var.varKind begin
      SimulationCode.DATA_STRUCTURE(__) => push!(_dsVarNames, varName)
      _ => nothing
    end
  end
  local DATA_STRUCTURE_ASSIGNMENTS = createDataStructureAssignments(_dsVarNames, simCode)
  #= Append top level variables to the common variables =#
  #= END =#
  code = quote
    import DAE
    import DataStructures.OrderedCollections
    import SCode
    import OMBackend
    import OMBackend.CodeGeneration
    import Setfield
    using ModelingToolkit
    using DifferentialEquations
    using DiffEqCallbacks
    Base.Experimental.@compiler_options optimize=0 compile=min infer=false
    $(createStringParameterAssignments(simCode)...)
    $(createArrayParameterPrelude(simCode)...)
    $(DATA_STRUCTURE_ASSIGNMENTS...)
    $(structuralModes...)
    $(structuralCallbacks...)
    #=
      This function can be used to fetch the top level callbacks that is the collected callbacks of the model.
      Each callback is coupled to each "when-equation" with a recompilation expression.
    =#
    function $(Symbol(MODEL_NAME * "Model"))(tspan = (0.0, 1.0))
      #=  Assign the initial model  =#
      (subModel, callbacks, finalInitialValues, initialValues, reducedSystem, _, pars, vars1) = $(Symbol(string(activeModelName, "Model")))(tspan)
      global LATEST_REDUCED_SYSTEM = reducedSystem
      #= Assign the structural callbacks =#
      $(structuralAssignments)
      $(commonVariables)
      #= END =#
      # Also need to have the original callbacks
      callbackConditions = $(if !isempty(structuralCallbacks)
                               :(CallbackSet(callbacks, callbackSet...))
                             else
                               :(CallbackSet(callbacks, callbackSet...))
                             end)
      #= Create the composite model. Dispatch on the mass matrix of the initial
         submodel exactly like the submodel builder does: pure ODE takes the
         fast fill-all-u0 / skip-initializer path; DAE routes finalInitialValues
         as hard u0 while letting the initializer use reducedSystem.guesses
         (already populated by splitInitialValues) to solve algebraic residuals
         such as x = L*sin(phi) for the Pendulum. Injecting _compositeGuesses
         as hard u0 for a DAE submodel would pin phi = 0 and silently violate
         the constraint. =#
      local _compositeIsPureODE = Base.invokelatest(
        OMBackend.CodeGeneration.isPureODESystem, reducedSystem)
      if _compositeIsPureODE
        local _compositeGuesses = Base.invokelatest(
          OMBackend.CodeGeneration.buildDefaultGuesses, reducedSystem, finalInitialValues, initialValues)
        compositeProblem = ModelingToolkit.ODEProblem(
          reducedSystem,
          merge(Dict(finalInitialValues), _compositeGuesses, pars),
          tspan;
          callback = callbackConditions,
          warn_initialize_determined = false,
          build_initializeprob = false,
        )
      else
        compositeProblem = ModelingToolkit.ODEProblem(
          reducedSystem,
          merge(Dict(finalInitialValues), pars),
          tspan;
          callback = callbackConditions,
          warn_initialize_determined = false,
        )
      end
      #=
      Note the difference between the two here.
      In the case of recompilation we will get fresh callbacks updated to the new structure of the final code.
      =#
      result = $(if simCode.metaModel == nothing
                   :(OMBackend.Runtime.OM_ProblemStructural($(activeModelName),
                                                            compositeProblem,
                                                            structuralCallbacks,
                                                            pars,
                                                            commonVariables,
                                                            $([Symbol(string(i,"(t)")) for i in simCode.topVariables]),
                                                            callbackSet))
                 else
                 :(OMBackend.Runtime.OM_ProblemRecompilation($(activeModelName),
                                                             compositeProblem,
                                                             structuralCallbacks,
                                                             callbackConditions))
                 end)
      return result
    end
    # function $(Symbol("$(MODEL_NAME)Simulate"))(tspan = (0.0, 1.0), solver=Rodas5(autodiff=false))
    #   $(Symbol("$(MODEL_NAME)Model_problem")) = $(Symbol("$(MODEL_NAME)Model"))(tspan)
    #   OMBackend.Runtime.solve($(Symbol("$(MODEL_NAME)Model_problem")), tspan, solver)
    # end

    function simulate(tspan = (0.0, 1.0), solver=Rodas5(); kwargs...)
      $(Symbol("$(MODEL_NAME)Model_problem")) = $(Symbol("$(MODEL_NAME)Model"))(tspan)
      OMBackend.Runtime.solve($(Symbol("$(MODEL_NAME)Model_problem")), tspan, solver; kwargs...)
    end
  end
  local moduleExpr = Expr(:module, true, Symbol(MODEL_NAME), stripBeginBlocks(code))
  return (MODEL_NAME, moduleExpr)
end

"""
  Generates a MTK program with a model
"""
function ODE_MODE_MTK_PROGRAM_GENERATION(simCode::SimulationCode.SIM_CODE, modelName, functions)
  local MODEL_NAME = modelName
  #= Functions are eval'd inside ODE_MODE_MTK_MODEL_GENERATION (called below)
     immediately before @register_symbolic, so no need to eval them here. =#
  local dataStructureVariables = String[]
  for varName in (keys(simCode.stringToSimVarHT))
    (idx, var) = simCode.stringToSimVarHT[varName]
    @match var.varKind begin
      SimulationCode.DATA_STRUCTURE(__) => begin
        push!(dataStructureVariables, varName)
      end
      _ => continue
    end
  end
  local DATA_STRUCTURE_ASSIGNMENTS = createDataStructureAssignments(dataStructureVariables, simCode)
  local model = ODE_MODE_MTK_MODEL_GENERATION(simCode, modelName, functions)
  #= Qualify bare Modelica function calls in function bodies so they resolve correctly
     when the program is eval'd in OMBackend scope (backendAPI.jl) rather than CodeGeneration scope.
     Without this, implementation bodies that call other Modelica functions (e.g., normalizeWithAssert
     calling Vectors_length) would fail with UndefVarError. =#
  local funcNames = Set{Symbol}(Symbol(f.name) for f in simCode.functions)
  if !isempty(funcNames)
    for f in functions
      qualifyModelicaFunctions!(f, funcNames)
    end
  end
  programBody = quote
    using ModelingToolkit
    using DifferentialEquations
    using DiffEqCallbacks
    using OrdinaryDiffEq
    using Symbolics
    using OMBackend
    import Setfield
    Base.Experimental.@compiler_options optimize=0 compile=min infer=false
    #= Add import to the external runtime if the generated code calls Modelica Functions =#
    $(if simCode.externalRuntime
        generateExternalRuntimeImport()
      end)
    $(functions...)
    $(createStringParameterAssignments(simCode)...)
    $(createArrayParameterPrelude(simCode)...)
    $(DATA_STRUCTURE_ASSIGNMENTS...)
    $(generateRegisterCallsForCallExprs(simCode)...)
    $(generateInitialAlgorithmEarlyFunction(simCode))
    $(generateInitialAlgorithmFunction(simCode))
    $(model)
    #= simulateFromBuild: post-build solve pipeline (init-alg, Rodas/FBDF auto-switch,
       DAE routing, InitialFailure retry, terminal events). Extracted from simulate so
       the iMTK path can drive it with a cached build; simulate behavior is unchanged. =#
    function simulateFromBuild(built, tspan = (0.0, 1.0), solver = Rodas5();  kwargs...)
      ($(Symbol("$(MODEL_NAME)Model_problem")), callbacks, ivs, _ivs_all, $(Symbol("$(MODEL_NAME)Model_ReducedSystem")), _tspan2, _pars, _vars, _irreducible) = built
      global LATEST_REDUCED_SYSTEM = $(Symbol("$(MODEL_NAME)Model_ReducedSystem"))
      global LATEST_PROBLEM = $(Symbol("$(MODEL_NAME)Model_problem"))
      #= Run in the latest world age: __runInitialAlgorithm! is compiled at
         module-eval time, before `Model()` runs `eval(_batchBlock)` to create
         the Symbolics bindings (e.g. `a`, `iNV3S_enable`) that algorithm-lifter
         bodies reference. Calling the function directly resolves those names
         in the older compile-time world and throws
         `UndefVarError: ... binding may be too new`. =#
      local _hardStarts = Base.invokelatest(__runInitialAlgorithm!)
      #= Stash the un-remake'd problem so the solve() fallback below can
         retry without enforced init-alg u0 if MTK's init system finds the
         hard-start values infeasible against the algebraic constraints. =#
      local _origProblem = $(Symbol("$(MODEL_NAME)Model_problem"))
      local _didRemake = false
      #= If the init algorithm assigned any non-parameter variables, replay
         those values through `remake(prob; u0=…)` so MTK treats them as
         hard initial conditions (Modelica §11.2). Symbolic-Num dict keys
         from LATEST_REDUCED_SYSTEM are required — bare Symbol keys are
         silently no-op'd by MTK's u0 dispatch. =#
      if _hardStarts isa AbstractDict && !isempty(_hardStarts)
        #= Filter to only the keys that are actual `unknowns` of the reduced
           system. Init-algorithm LHSs that get alias-eliminated post-simplify
           still resolve via `getproperty` (they survive as observed equations)
           but `remake`'s u0 validator rejects them with "present in the
           system but … is not an unknown". The existing `setu` mutation
           inside __runInitialAlgorithm! already propagates those via the
           alias-map's observed equation, so dropping them is safe. =#
        local _unkNames = try
          Set(string(u) for u in ModelingToolkit.unknowns(LATEST_REDUCED_SYSTEM))
        catch
          Set{String}()
        end
        local _hardFiltered = filter(p -> string(first(p)) in _unkNames, _hardStarts)
        if !isempty(_hardFiltered)
          try
            global LATEST_PROBLEM = ModelingToolkit.SciMLBase.remake(
              LATEST_PROBLEM; u0 = _hardFiltered)
            $(Symbol("$(MODEL_NAME)Model_problem")) = LATEST_PROBLEM
            _didRemake = true
          catch _ialgErr
            #= The cycle-19 remake is now redundant for variables that the
               module-load-time `__runInitialAlgorithmEarly!()` path already
               pinned via `initialization_eqs`. After MTK's init solve runs
               those constraints, alias elimination can prune the symbolic
               key out of the problem's u0 vector, and `remake(; u0 = Dict)`
               then raises `BoundsError` / "key not an unknown". That is
               harmless because the init-eq value is already in the solved
               state. Demote to debug — a real failure would still surface
               from the solve itself. =#
            @debug "[MTK GEN: simulate] init-alg hard-start remake skipped (init-eqs already covered)" exception=_ialgErr
          end
        end
      end
      #= Auto-switch from Rosenbrock (default Rodas5) to FBDF for DAE shapes
         where Rosenbrock mass-matrix stepping is known to be brittle:
         purely-algebraic systems, and mixed systems with algebraic rows for
         generated discrete variables. Brake reaches a consistent initial
         residual, but Rodas5P immediately aborts with dt_epsilon/NaN while
         FBDF advances the same mass-matrix problem. User-chosen non-Rosenbrock
         solvers are respected as-is. =#
      local _solver = solver
      local _solverName = string(nameof(typeof(solver)))
      if startswith(_solverName, "Rodas") || startswith(_solverName, "Rosenbrock")
        local _u0 = $(Symbol("$(MODEL_NAME)Model_problem")).u0
        local _n = _u0 === nothing ? 0 : length(_u0)
        local _mm = $(Symbol("$(MODEL_NAME)Model_problem")).f.mass_matrix
        local _discreteUnknownNames = Set{String}($(Expr(:vect, [string(varName, "(t)") for (varName, (_, simVar)) in simCode.stringToSimVarHT if simVar.varKind isa SimulationCode.DISCRETE]...)))
        #= UniformScaling (pure ODE) supports `_mm[i,i]` as 1; explicit Matrix
           returns the entry. Both code paths handle by indexing the diagonal.
           When u0 is nothing (purely-algebraic MTK problem) the loop runs 0
           times so _nDiff stays 0 — exactly the case where FBDF is wanted. =#
        local _nDiff = count(i -> _mm[i,i] != 0, 1:_n)
        local _nAlg = _n - _nDiff
        if _nDiff == 0
          @info "[MTK GEN: solver] zero differential states detected, switching default $(_solverName) -> FBDF for purely-algebraic DAE"
          _solver = FBDF(autodiff=false)
        elseif _nAlg > 0 && !isempty(_discreteUnknownNames)
          local _unknowns = try
            ModelingToolkit.unknowns(LATEST_REDUCED_SYSTEM)
          catch
            Any[]
          end
          local _nCheck = min(_n, length(_unknowns))
          local _hasDiscreteAlgUnknown = any(i -> _mm[i,i] == 0 && string(_unknowns[i]) in _discreteUnknownNames, 1:_nCheck)
          if _hasDiscreteAlgUnknown
            @info "[MTK GEN: solver] algebraic rows involving generated discrete variables detected in mass-matrix system; switching default $(_solverName) -> FBDF"
            _solver = FBDF(autodiff=false)
          end
        end
      end
      # Route DAE-native solvers (e.g. Sundials.IDA, DABDF2, DFBDF) through a residual-form DAEProblem rather than the ODEProblem with mass matrix.
      local _problemForSolver = if _solver isa ModelingToolkit.SciMLBase.AbstractDAEAlgorithm
        OMBackend.CodeGeneration.ode_to_dae($(Symbol("$(MODEL_NAME)Model_problem")))
      else
        $(Symbol("$(MODEL_NAME)Model_problem"))
      end
      #= Pass callbacks at solve time. MTK's ODEProblem(callback=...) kwarg
         silently drops ContinuousCallback objects (only the DiscreteCallback
         init survives), so when-clause root-find callbacks never fire when
         routed through the prob. solve() merges with prob.kwargs[:callback]
         so MTK's init still runs in addition to our callbacks. =#
      local _sol = if haskey(kwargs, :callback)
        solve(_problemForSolver, _solver; kwargs...)
      else
        solve(_problemForSolver, _solver; callback=callbacks, kwargs...)
      end
      #= If the init-alg-remake'd problem produced InitialFailure (MTK could
         not reconcile init-alg hard-start u0 with the algebraic constraints),
         fall back to the un-remake'd problem so the solver can pick any
         consistent u0. This matches pre-cycle-19 behavior for models where
         the init-alg LHS values would be silently overwritten by MTK's init
         solver anyway (e.g. KinematicPTPHandwritten — algebraic-only model
         whose init-alg assignments conflict with algebraic equations). =#
      if _didRemake && _sol.retcode == ModelingToolkit.SciMLBase.ReturnCode.InitialFailure
        @info "[MTK GEN: simulate] init-alg hard-start caused InitialFailure; retrying without hard-start"
        global LATEST_PROBLEM = _origProblem
        $(Symbol("$(MODEL_NAME)Model_problem")) = _origProblem
        local _fallbackProb = if _solver isa ModelingToolkit.SciMLBase.AbstractDAEAlgorithm
          OMBackend.CodeGeneration.ode_to_dae(_origProblem)
        else
          _origProblem
        end
        _sol = if haskey(kwargs, :callback)
          solve(_fallbackProb, _solver; kwargs...)
        else
          solve(_fallbackProb, _solver; callback=callbacks, kwargs...)
        end
      end
      #= Run `when terminal()` bodies once against the final solution (gated: emitted only if the model has a terminal event). =#
      $(createTerminalBodyRunner(simCode))
      _sol
    end
    function simulate(tspan = (0.0, 1.0), solver = Rodas5(); cached_build = nothing, kwargs...)
      local built = cached_build === nothing ? $(Symbol("$(MODEL_NAME)Model"))(tspan) : cached_build
      return simulateFromBuild(built, tspan, solver; kwargs...)
    end
  end
  #= MODEL_NAME is preprocessed with . replaced with _=#
  local moduleExpr = Expr(:module, true, Symbol(MODEL_NAME), stripBeginBlocks(programBody))
  return MODEL_NAME, moduleExpr
end

"""
  Generates a MTK model
"""
function ODE_MODE_MTK_MODEL_GENERATION(simCode::SimulationCode.SIM_CODE, modelName, functions; useDirectRHS::Bool = OMBackend.DIRECT_RHS_GENERATION[])
  RESET_CALLBACKS()

  #= Phase A — eval generated Modelica functions and their @register_symbolic
     calls into OMBackend so subsequent codegen sees the bindings. =#
  evalGeneratedFunctionsAndRegister!(modelName, functions, simCode)

  #= Phase B — bucket each simvar by varKind (state / algebraic / discrete
     / parameter / array / occ / data-structure / state-derivative) and
     extract StateSelect priority pairs. =#
  local vars = classifyVariables(simCode)
  local stateVariables         = vars.stateVariables
  local algebraicVariables     = vars.algebraicVariables
  local discreteVariables      = vars.discreteVariables
  local occVariables           = vars.occVariables
  local parameters             = vars.parameters
  local arrayParameters        = vars.arrayParameters
  local stateDerivatives       = vars.stateDerivatives
  local dataStructureVariables = vars.dataStructureVariables
  local statePriorityPairs     = vars.statePriorityPairs

  #= (b) DISCRETE_AS_PARAM: callback-driven (when-assigned) held discretes become
     PARAMETERS modified by the existing when-callbacks (ridden through the ifCond
     param declaration path below), not der~0 continuous states. Partition them out
     of `discreteVariables` here so the dummy / planDemotions / unknowns path never
     sees them — retiring the off-by-one demotion accounting for this class. =#
  local heldDiscreteParamDecls = Expr[]
  local heldDiscreteParamPairs = Expr[]
  local heldDiscreteSyms = Symbol[]
  local _heldSet = _heldDiscreteParamSet(simCode)
  if !isempty(_heldSet)
    local _held = filter(dv -> dv in _heldSet, discreteVariables)
    discreteVariables = filter(dv -> !(dv in _heldSet), discreteVariables)
    local _unwrapPair = function (b)
      local e = b
      if e isa Expr && e.head === :block
        for a in e.args
          a isa LineNumberNode && continue
          e = a
          break
        end
      end
      if e isa Expr && e.head === :call && length(e.args) == 3 && e.args[1] === :(=>)
        return (e.args[2], e.args[3])
      end
      return nothing
    end
    local _seen = Set{Symbol}()
    for _b in getStartConditionsMTK(_held, simCode)
      local _p = _unwrapPair(_b)
      (_p === nothing || !(_p[1] isa Symbol)) && continue
      push!(heldDiscreteParamDecls, Expr(:(=), _p[1], _p[2]))
      push!(heldDiscreteParamPairs, :($(_p[1]) => $(_p[2])))
      push!(heldDiscreteSyms, _p[1]); push!(_seen, _p[1])
    end
    for dv in _held
      local _s = Symbol(dv)
      if !(_s in _seen)
        push!(heldDiscreteParamDecls, Expr(:(=), _s, 0.0))
        push!(heldDiscreteParamPairs, :($(_s) => 0.0))
        push!(heldDiscreteSyms, _s)
      end
    end
  end

  local performIndexReduction = simCode.isSingular
  local skipInitializeProb = SimulationCode.hasStructuralTransitions(simCode) ||
                             SimulationCode.hasMetaModel(simCode) ||
                             SimulationCode.hasFlatModel(simCode)
  #= Solve parametric initial equations (initial equations that only involve parameters).
     This determines values for fixed=false parameters before code generation. =#
  solveParametricInitialEquations!(simCode)
  #= Create equations for variables not in a loop + parameters and stuff=#
  local EQUATIONS = createResidualEquationsMTK(stateVariables,
                                               algebraicVariables,
                                               simCode.residualEquations,
                                               simCode::SimulationCode.SIM_CODE)
  @BACKEND_LOGGING writeEqsToFile(EQUATIONS, OMBackend.logPath("backend/codeGen", "equationFirstStageCodeGen.log"))
  #=
  If missing from variable map error is thrown check the start condition.
  Readded discretes here....
  =#
  local INITIAL_GUESS_EQUATIONS = createStartConditionsEquationsMTK(vcat(stateVariables, occVariables),
                                                                      algebraicVariables,
                                                                      simCode)


  local DISCRETE_START_VALUES = vcat(generateInitialEquations(simCode.initialEquations, simCode; parameterAssignment = true),
                                     getStartConditionsMTK(discreteVariables, simCode))
  local PARAMETER_EQUATIONS = createParameterEquationsMTK(parameters, simCode)
  local PARAMETER_ASSIGNMENTS = createParameterAssignmentsMTK(parameters, simCode)
  local PARAMETER_RAW_ARRAY = createParameterArray(parameters, PARAMETER_ASSIGNMENTS, simCode)
  local ARRAY_PARAMETERS = createArrayParametersMTK(arrayParameters, simCode)
  #=
  Create callback equations.
  =#
  local CALL_BACK_EQUATIONS = createCallbackCode(modelName, simCode; generateSaveFunction = false)
  local IF_EQUATION_COMPONENTS::Vector{IfEquationComponent} =
    createIfEquations(stateVariables, algebraicVariables, simCode)
  #= Symbolic names =#
  local algebraicVariablesSym = Symbol[:($(Symbol(v))) for v in algebraicVariables]
  local dataStructureVariablesSym = Symbol[Symbol(v) for v in dataStructureVariables]
  local stateVariablesSym = Symbol[:($(Symbol(v))) for v in stateVariables]
  local occVariablesSym = Symbol[:($(Symbol(v))) for v in occVariables]
  local parVariablesSym = Symbol[Symbol(p) for p in parameters]
  #= Phase 6: discrete-dummy demotion. Each discrete variable starts with a
     placeholder `der(d) ~ 0` so SciML has a state slot for callbacks to
     write into. When a residual equation already pins `d` definitionally
     (alias, ifelse, comparison, integer cast, ifEq_tmp target, pairwise
     discrete alias, ...), MTK's structural_simplify uses that equation
     to eliminate `d`, stranding the dummy and over-determining the system.
     `planDemotions` detects those cases (plus cyclic-SCC discretes and a
     bounded heuristic for any remaining excess) and `applyDemotionPlan!`
     drops the corresponding dummies, reclassifying the names as algebraic.
     See OMBackend/src/CodeGeneration/DiscreteDummyDemotion.jl for the
     full pattern catalogue and the when-equation safety rule. =#
  local discreteVariablesSym = Symbol[:($(Symbol(v))) for v in discreteVariables]
  local DISCRETE_DUMMY_EQUATIONS = [:(der($(Symbol(dv))) ~ 0) for dv in discreteVariables]
  local _demotionPlan = planDemotions(simCode, EQUATIONS, IF_EQUATION_COMPONENTS,
                                      discreteVariables,
                                      length(stateVariables),
                                      length(algebraicVariables),
                                      length(occVariables))
  (DISCRETE_DUMMY_EQUATIONS, discreteVariablesSym) =
    applyDemotionPlan!(_demotionPlan, discreteVariables, DISCRETE_DUMMY_EQUATIONS,
                       discreteVariablesSym, algebraicVariablesSym)
  #= Phase F — flatten the per-if-equation components into one event-decl
     Expr (wrapped in invokelatest because event exprs reference Symbolics
     bindings only created later inside the model function), plus three
     flat lists used by downstream phases. =#
  local IF_EQUATION_EVENTS = collect(Iterators.flatten(c.events for c in IF_EQUATION_COMPONENTS))
  local IF_EQUATION_EVENT_DECLARATION = buildIfEquationEventDecl(IF_EQUATION_EVENTS)
  local CONDITIONAL_EQUATIONS = collect(Iterators.flatten(c.conditionalEquations for c in IF_EQUATION_COMPONENTS))
  local ifConditionNameAndIV = collect(Iterators.flatten(c.conditionNameAndIV for c in IF_EQUATION_COMPONENTS))
  local ifConditionalVariables = collect(Iterators.flatten(c.conditionVariables for c in IF_EQUATION_COMPONENTS))
  #= ifCond variables are parameters (not ODE unknowns).
     Build @parameters declarations WITHOUT time dependency to avoid MTK creating
     Shift operators. Plain parameters are still modifiable by callback affects. =#
  local ifCondParamDecls = Expr[]
  local ifCondParamPairs = Expr[]
  for (name, initVal) in ifConditionNameAndIV
    local sym = Symbol(name)
    local numVal = initVal ? 1.0 : 0.0
    push!(ifCondParamDecls, Expr(:(=), sym, numVal))
    push!(ifCondParamPairs, :($(sym) => $(numVal)))
  end
  #= (b) held discretes ride the ifCond param path: declared as @parameters,
     bound, added to `parameters`, and their start values pushed into `pars`.
     Lists stay parallel (decls/syms built together above). =#
  append!(ifCondParamDecls, heldDiscreteParamDecls)
  append!(ifCondParamPairs, heldDiscreteParamPairs)
  append!(ifConditionalVariables, heldDiscreteSyms)
  #= Phase G — collect the symbols MTK tearing must not eliminate
     (simCode-flagged irreducibles + ifEq_tmp LHS targets + fixed-start
     variables). =#
  local irreducibleSyms = collectIrreducibleSymbols(simCode, CONDITIONAL_EQUATIONS,
                                                    stateVariables, algebraicVariables,
                                                    occVariables)

  #= Heuristic for initialization:
     - If any state variable has an explicit start value, assume the system has algebraic
       constraints and only initialize states with explicit starts (avoid overdetermination).
     - If NO state has an explicit start, provide defaults for all states (pure ODE case).
     - Exception: when build_initializeprob is disabled (structural transition models),
       there is no initialization solver to infer values from constraints/guesses, so
       we MUST provide u0 defaults for all unknowns.
     This handles both constrained DAE systems (like Pendulum) and pure ODE systems
     (like MatrixVectorMult where states have no explicit start). =#
  local anyStateHasExplicitStart = hasExplicitStartValue(simCode.irreducibleVariables, simCode)
  local skipDefaultsForStates = anyStateHasExplicitStart
  #= Build default guesses for unknowns not in the heuristic-filtered u0.
     Guesses are passed to ODEProblem so the init solver has fallback values
     without overdetermining the system. =#
  local INITIAL_VALUE_EQUATIONS = unique!(createStartConditionsEquationsMTK(
    String[vn for vn in simCode.irreducibleVariables],
    String[],
    simCode; skipDefaultStateStarts = skipDefaultsForStates))
  INITIAL_VALUE_EQUATIONS = vcat(DISCRETE_START_VALUES, INITIAL_VALUE_EQUATIONS)
  INITIAL_GUESS_EQUATIONS = vcat(DISCRETE_START_VALUES, INITIAL_GUESS_EQUATIONS)
  #=
    Merge equations. ifCond variables are discrete parameters so they are NOT
    included in stateVariablesSym and do NOT get der() ~ 0 equations.
  =#
  stateVariablesSym = vcat(discreteVariablesSym,
                           stateVariablesSym,
                           occVariablesSym)
  local (_ifEqRelay_eqs, _ifEqRelay_aliases) = eliminateIfEqRelays(EQUATIONS)
  EQUATIONS = _ifEqRelay_eqs
  if !isempty(_ifEqRelay_aliases)
    @info "[RELAY] aliases" _ifEqRelay_aliases
    local _drop = Set(keys(_ifEqRelay_aliases))
    local _dropStr = Set(string.(keys(_ifEqRelay_aliases)))
    stateVariablesSym = filter(s -> s ∉ _drop, stateVariablesSym)
    algebraicVariablesSym = filter(s -> s ∉ _drop, algebraicVariablesSym)
    algebraicVariables = filter(s -> s ∉ _dropStr, algebraicVariables)
    irreducibleSyms = filter(s -> s ∉ _drop, irreducibleSyms)
    local _keepPair = eq -> begin
      local inner = _unwrapBlock(eq)
      if inner isa Expr && inner.head === :call && length(inner.args) == 3 && inner.args[1] === :(=>)
        inner.args[2] isa Symbol && inner.args[2] in _drop && return false
      end
      true
    end
    local _keepDummy = eq -> begin
      local inner = _unwrapBlock(eq)
      if inner isa Expr && inner.head === :call && length(inner.args) == 3 && inner.args[1] === :~
        local lhs = _unwrapBlock(inner.args[2])
        if lhs isa Expr && lhs.head === :call && length(lhs.args) == 2 &&
           (lhs.args[1] === :der || lhs.args[1] === :D)
          local sym = _simpleLeafSymbol(lhs.args[2])
          sym !== nothing && sym in _drop && return false
        end
      end
      true
    end
    local _nBefore = length(INITIAL_GUESS_EQUATIONS)
    INITIAL_GUESS_EQUATIONS = filter(_keepPair, INITIAL_GUESS_EQUATIONS)
    @info "[RELAY] INITIAL_GUESS_EQUATIONS filtered" before=_nBefore after=length(INITIAL_GUESS_EQUATIONS)
    INITIAL_VALUE_EQUATIONS = filter(_keepPair, INITIAL_VALUE_EQUATIONS)
    DISCRETE_START_VALUES = filter(_keepPair, DISCRETE_START_VALUES)
    #= Substitute the relay aliases inside the surviving pairs/equations so a
       value side referencing an eliminated leaf (e.g. `variance_mu => variance_u`)
       resolves to the surviving rep symbol rather than leaving an undefined name. =#
    INITIAL_GUESS_EQUATIONS = [_substSyms(eq, _ifEqRelay_aliases) for eq in INITIAL_GUESS_EQUATIONS]
    INITIAL_VALUE_EQUATIONS = [_substSyms(eq, _ifEqRelay_aliases) for eq in INITIAL_VALUE_EQUATIONS]
    DISCRETE_START_VALUES = [_substSyms(eq, _ifEqRelay_aliases) for eq in DISCRETE_START_VALUES]
    DISCRETE_DUMMY_EQUATIONS = filter(_keepDummy, DISCRETE_DUMMY_EQUATIONS)
    DISCRETE_DUMMY_EQUATIONS = [_substSyms(eq, _ifEqRelay_aliases) for eq in DISCRETE_DUMMY_EQUATIONS]
    IF_EQUATION_EVENTS = [_substSyms(ev, _ifEqRelay_aliases) for ev in IF_EQUATION_EVENTS]
    IF_EQUATION_EVENT_DECLARATION = buildIfEquationEventDecl(IF_EQUATION_EVENTS)
    CONDITIONAL_EQUATIONS = [_substSyms(eq, _ifEqRelay_aliases) for eq in CONDITIONAL_EQUATIONS]
  end
  EQUATIONS = vcat(EQUATIONS,
                   DISCRETE_DUMMY_EQUATIONS,
                   CONDITIONAL_EQUATIONS)
  EQUATIONS = rewriteEquations(EQUATIONS, simCode)
  local _seenMtkEquationExprs = Set{String}()
  local _dedupedMtkEquations = Expr[]
  local _nDedupedMtkEquations = 0
  for eq in EQUATIONS
    local key = string(stripLineNodes(eq))
    if key in _seenMtkEquationExprs
      _nDedupedMtkEquations += 1
    else
      push!(_seenMtkEquationExprs, key)
      push!(_dedupedMtkEquations, eq)
    end
  end
  if _nDedupedMtkEquations > 0
    @debug "[MTK GEN: equations] removed $(_nDedupedMtkEquations) duplicate MTK equations after rewrite"
    EQUATIONS = _dedupedMtkEquations
  end
  #= Reset the callback counter=#
  RESET_CALLBACKS()
  #=
    Formulate the problem as a DAE Problem.
    For this variant we keep it on its own line
    https://github.com/SciML/ModelingToolkit.jl/issues/998
  =#
  #=If our model name is separated by . replace it with __ =#
  local MODEL_NAME = modelName
  #= Decompose variables, equations, and start equations into (outer_defs, inner_refs).
     outer_defs go at module level (before model function) to avoid nested closure JIT.
     inner_refs go inside the model function body. =#
  local modelPrefix = "_" * MODEL_NAME * "_"
  local (varOuterDefs, varInnerRefs) = decomposeVariables(
    stateVariablesSym, algebraicVariablesSym; modelPrefix = modelPrefix)
  model = quote
    $(CALL_BACK_EQUATIONS)
    #= Variable constructor function definitions at module level (outside model function)
       to avoid JIT overhead from compiling nested closures.
       Variable constructors only return symbol tuples, so they have no scope dependencies. =#
    $(varOuterDefs)
    function $(Symbol(MODEL_NAME * "Model"))(tspan = (0.0, 1.0))
      ModelingToolkit.@independent_variables t
      D = ModelingToolkit.Differential(t)
      $(decomposeParametersDeclaration(parVariablesSym))
      #= Create array parameters with proper dimensions =#
      $(ARRAY_PARAMETERS...)
      #= Declare ifCond variables as discrete time-dependent parameters.
         These are modified by SymbolicContinuousCallback affects and are NOT
         part of the ODE state vector, so the solver never perturbs them. =#
      $(generateDiscreteIfCondDeclaration(ifCondParamDecls, ifConditionalVariables))
      #=
        Only variables that are present in the equation system later should be a part of the variables in the MTK system.
        This means that certain algebraic variables should not be listed among the variables (These are the discrete variables).
      =#
      $(varInnerRefs)
      allVariables = []
      #= Generate variables =#
      for constructor in variableConstructors
        vars = map(n -> (n, Symbolics.variable(n, T = Symbolics.FnType{Tuple, Real, Nothing})(t)), Base.invokelatest(constructor))
        push!(allVariables, vars)
      end
      vars = collect(Iterators.flatten(allVariables))
      #= Batch all variable assignments and metadata into a single eval call.
         Each individual eval triggers a world-age bump and JIT overhead.
         For models with 1000+ variables this reduces N evals to 1. =#
      local _batchBlock = Expr(:block)
      for (sym, var) in vars
        push!(_batchBlock.args, :($sym = $var))
      end
      local irreducibleSyms = $(irreducibleSyms)
      for sym in irreducibleSyms
        push!(_batchBlock.args, :($sym = SymbolicUtils.setmetadata($sym, ModelingToolkit.VariableIrreducible, true)))
      end
      local _statePriorityPairs = $(statePriorityPairs)
      for (sym, priority) in _statePriorityPairs
        push!(_batchBlock.args, :($sym = SymbolicUtils.setmetadata($sym, ModelingToolkit.VariableStatePriority, $priority)))
      end
      #= Dump the resolved variable-binding batch before `eval`. See
         CodeGeneration/mtkDump.jl. The dump runs at simulate time inside
         the model module, so it must reference MTKDump by its absolute
         module path (the model module does not import MTKDump). =#
      OMBackend.CodeGeneration.MTKDump.dumpBatchBlock(vars, irreducibleSyms, _statePriorityPairs, _batchBlock)
      eval(_batchBlock)
      # re-fetch decorated Nums from module scope (eval rebinds names but
      # local vars still holds pre-eval references)
      vars = [Base.invokelatest(getfield, @__MODULE__, sym) for (sym, _) in vars]
      #= Initial values for the continuous system. =#
      $(decomposeParameterEquationsInline(PARAMETER_EQUATIONS))
      #= Add ifCond discrete parameter values to pars dict =#
      $(generateIfCondParamAssignments(ifCondParamPairs))
      startEquationComponents = []
      $(decomposeStartEquationsInline(INITIAL_GUESS_EQUATIONS))
      for constructor in startEquationConstructors
        push!(startEquationComponents, Base.invokelatest(constructor))
      end
      initialValues = collect(Iterators.flatten(startEquationComponents))
      #= Process the final initial guesses =#
      startEquationComponents = []
      $(decomposeStartEquationsInline(INITIAL_VALUE_EQUATIONS; functionSuffix = "Final"))
      for constructor in startEquationConstructors
        push!(startEquationComponents, Base.invokelatest(constructor))
      end
      finalInitialValues = collect(Iterators.flatten(startEquationComponents))
      #= Equations =#
      equationComponents = []
      $(stripBeginBlocks(decomposeEquationsInline(EQUATIONS, PARAMETER_ASSIGNMENTS)))
      for constructor in equationConstructorCalls
        push!(equationComponents, Base.invokelatest(constructor))
      end
      eqs = collect(Iterators.flatten(equationComponents))
      eqs = Base.invokelatest(OMBackend.CodeGeneration.filterConstantEquations, eqs)
      #= System(eqs, ...) requires eqs::Vector{Equation}; an equation-free model yields an untyped empty vector. =#
      eqs = convert(Vector{Symbolics.Equation}, eqs)
      #= Events and observed equations =#
      $(IF_EQUATION_EVENT_DECLARATION)
      $(generateAliasObservedBlock(simCode, _ifEqRelay_aliases))
      $(generateEliminatedObservedBlock(simCode, _ifEqRelay_aliases))
      #= Initial-equation constraints (from Modelica `initial equation` block).
         Passed as `initialization_eqs` to MTK so they actually constrain the
         t=0 state — the `initialValues` Pair list above is only a guess.
         Wrapped in invokelatest so symbol references resolve against the
         freshly-eval'd Symbolics bindings. =#
      local _algResults = try
        Base.invokelatest(__runInitialAlgorithmEarly!)
      catch _err
        @debug "[MTK GEN: init-alg] early eval threw at constraint-build" exception=_err
        Dict{Symbol, Float64}()
      end
      function _buildInitialConstraintEqs()
        local _eqs = Symbolics.Equation[$([_substSyms(e, _ifEqRelay_aliases) for e in generateInitialEquationsAsConstraints(simCode.initialEquations, simCode)]...),
                                        $([_substSyms(e, _ifEqRelay_aliases) for e in getFixedStartConstraintsMTK(vcat(stateVariables, occVariables, algebraicVariables), simCode)]...)]
        $(emitInitAlgConstraintAppends(simCode)...)
        return _eqs
      end
      local initialConstraintEqs = Base.invokelatest(_buildInitialConstraintEqs)
      #= Also merge the early-eval init-algorithm results into `finalInitialValues`
         as hard u0 entries. Needed because the `_isPureODE` branch (state with
         der=0 and no algebraic constraints) skips `build_initializeprob` — MTK's
         init solver never runs, so the `initialization_eqs` set above would not
         be honoured on its own. With u0 set here, both the pure-ODE fast path
         and the DAE-with-init-solver path produce the same initial values. =#
      function _mergeInitAlgIntoU0!(fiv)
        $(emitInitAlgU0Appends(simCode)...)
        return fiv
      end
      Base.invokelatest(_mergeInitAlgIntoU0!, finalInitialValues)
      #= ODE System =#
      nonLinearSystem = $(odeSystemWithEvents(!isempty(ifConditionalVariables), modelName;
                                              hasObserved = !isempty(simCode.aliasMap) ||
                                                            !isempty(simCode.eliminatedVariables)))
      firstOrderSystem = nonLinearSystem
      #= Structural simplification =#
      $(performStructuralSimplify(performIndexReduction; observedFilter = simCode.observedFilter, split = !useDirectRHS))
      #= Inject observed equations post-simplification so they do not interfere
         with AffectSystem tearing during callback compilation. =#
      if @isdefined(observedEqs) && !isempty(observedEqs)
        #= Deduplicate observed equations by LHS variable name before injection.
           Both alias and eliminated observed blocks can produce the same equation. =#
        local _seenLHS = Set{String}()
        local _uniqueObs = Symbolics.Equation[]
        for _obs in observedEqs
          local _lhsKey = string(Symbolics.unwrap(_obs.lhs))
          if !(_lhsKey in _seenLHS)
            push!(_seenLHS, _lhsKey)
            push!(_uniqueObs, _obs)
          end
        end
        reducedSystem = OMBackend.CodeGeneration.injectObservedEquations(reducedSystem, _uniqueObs)
      end
      #= Callbacks setup =#
      local eventParameters = [$(PARAMETER_RAW_ARRAY...)]
      #= Wrap discrete start values in a function and call with invokelatest to avoid world-age issues =#
      function _getDiscreteVars()
        collect(values(ModelingToolkit.OrderedDict($(DISCRETE_START_VALUES...))))
      end
      local discreteVars = Base.invokelatest(_getDiscreteVars)
      eventParameters = vcat(eventParameters, discreteVars)
      local aux = Vector{Any}(undef, 3)
      aux[1] = eventParameters
      aux[2] = Float64[]
      aux[3] = reducedSystem
      #= Maps OMBackend variable indices to actual state indices =#
      callbacks = $(Symbol("$(MODEL_NAME)CallbackSet"))(aux)
      #= Split initial values =#
      local _finalInitialValuesForSplit = Pair{Any, Any}[p for p in finalInitialValues]
      local _initialValuesForSplit = Pair{Any, Any}[p for p in initialValues]
      (reducedSystem, finalInitialValues) = Base.invokelatest(
        OMBackend.CodeGeneration.splitInitialValues, reducedSystem, _finalInitialValuesForSplit, _initialValuesForSplit)
      #= Build ODEProblem. The codegen-time strategy (DirectRHS / structural
         transition / standard DAE-with-init-solver) is picked here; the
         structural-transition branch additionally dispatches at runtime on
         the mass matrix. See `emitProblemConstruction` and its three
         strategy emitters for the full rationale. =#
      $(emitProblemConstruction(useDirectRHS, skipInitializeProb))
      return (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars, irreducibleSyms)
    end
  end
  #= Qualify bare Modelica function calls with OMBackend.CodeGeneration. prefix.
     This covers all generated code: equations, parameter assignments, start conditions. =#
  local funcNames = Set{Symbol}(Symbol(f.name) for f in simCode.functions)
  if !isempty(funcNames)
    qualifyModelicaFunctions!(model, funcNames)
  end
  return model
end

"""
    generateAliasObservedBlock(simCode)

Generate a code block that creates observed equations for eliminated alias variables.
Each alias entry produces:
  - A symbolic variable declaration for the eliminated variable
  - An observed equation: `eliminated(t) ~ representative(t)` (or negated)
These are passed to `ODESystem` via the `observed` keyword so that eliminated
variables remain accessible in the solution (e.g. `sol[var"eliminated"]`).
"""
function generateAliasObservedBlock(simCode::SimulationCode.SIM_CODE,
                                    relayAliases::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}())
  if isempty(simCode.aliasMap) && isempty(relayAliases)
    return :(observedEqs = [])
  end
  #= Generate the observed equations as runtime code.
     The alias map entries are known at code-gen time, so we can embed
     the variable names as string literals. At runtime, these create
     Symbolics variables and equations. =#
  local obsEntries = Tuple{Symbol, Symbol, Bool}[]
  local elimSymbols = Symbol[]
  local emittedElims = Set{Symbol}()
  for entry in simCode.aliasMap
    local elimSym = Symbol(entry.eliminatedName)
    local repSym = Symbol(entry.representativeName)
    repSym = get(relayAliases, repSym, repSym)
    push!(elimSymbols, elimSym)
    push!(emittedElims, elimSym)
    push!(obsEntries, (elimSym, repSym, entry.negated))
  end
  local _relayRepresentative(sym::Symbol)::Symbol = begin
    local seen = Set{Symbol}()
    local cur = sym
    while haskey(relayAliases, cur) && !(cur in seen)
      push!(seen, cur)
      cur = relayAliases[cur]
    end
    cur
  end
  for elimSym in sort!(collect(keys(relayAliases)); by = string)
    elimSym in emittedElims && continue
    local repSym = _relayRepresentative(relayAliases[elimSym])
    push!(elimSymbols, elimSym)
    push!(emittedElims, elimSym)
    push!(obsEntries, (elimSym, repSym, false))
  end
  #= Collect eliminated symbol names at code-gen time. The Num objects
     are constructed at runtime (below) using the function-scope `t` so
     they share the system's independent variable. =#
  unique!(elimSymbols)
  return quote
    #= Build eliminated alias variables at function scope so they share
       the `@independent_variables t` object with the main system, then
       bind their names into the module namespace via a single eval (with
       the Num objects embedded by value). Using `ModelingToolkit.t_nounits`
       here would create variables with a different iv, which later trips
       `validate_operator` with `iv::Nothing` during Pantelides. =#
    local _elimBatch = Expr(:block)
    for _elimName in $(elimSymbols)
      local _elimVar = Symbolics.variable(_elimName,
                                          T = Symbolics.FnType{Tuple, Real, Nothing})(t)
      push!(_elimBatch.args, :($_elimName = $_elimVar))
    end
    eval(_elimBatch)
    #= Create observed equations using module lookups so symbols created by
       the preceding eval are visible without relying on generated helper
       function global resolution. =#
    observedEqs = Symbolics.Equation[]
    for (_elimName, _repName, _negated) in $(obsEntries)
      local _elimVar = Base.invokelatest(getfield, @__MODULE__, _elimName)
      local _repVar = Base.invokelatest(getfield, @__MODULE__, _repName)
      push!(observedEqs, _negated ? (_elimVar ~ -_repVar) : (_elimVar ~ _repVar))
    end
  end
end

function generateEliminatedObservedBlock(simCode::SimulationCode.SIM_CODE,
                                         relayAliases::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}())
  if isempty(simCode.eliminatedVariables)
    return :()
  end
  local elimVars = simCode.eliminatedVariables
  local elimEqs = simCode.eliminatedEquations
  @assert length(elimVars) == length(elimEqs) "eliminatedVariables and eliminatedEquations must be parallel"
  #= Always create Symbolics bindings for every eliminated variable so that
     other observed equations (and any downstream code) can resolve the
     variable name against a valid Num. Without this, an eliminated variable
     that is referenced by another eliminated variable's residual would raise
     a UndefVarError at module eval time (observed in DCEE_Start/DCPM_Start,
     where `wMechanical` is referenced by sibling eliminated equations). =#
  local allElimSymbols = Symbol[Symbol(v) for v in elimVars]
  #= Skip generating the observed equation (solve_for + push) for pairs whose
     residual contains a der() call. The solved form would be
     `elimVar ~ Differential(t)(x)`, which MTK rejects when it later builds
     the initialization system via the iv-less 3-arg
     `System(eqs, vars, ps)` constructor (validate_operator fails with
     OperatorIndepvarMismatchError). These eliminated variables are state
     derivatives whose values are already exposed by MTK's solution object. =#
  #= Names already emitted by `generateAliasObservedBlock` from `aliasMap`
     have a direct `elim ~ rep` observed equation. Re-deriving the same
     observation here via `solve_for(0 ~ residual, elim)` is redundant and
     fails when the residual has already been alias-substituted (the
     residual no longer mentions `elim` and `solve_for` returns NaN, which
     then propagates into `sol(t; idxs = elim)`). =#
  local aliasNames = Set{String}(entry.eliminatedName for entry in simCode.aliasMap)
  union!(aliasNames, string.(keys(relayAliases)))
  local solveBodyExprs = Expr[]
  for (i, varName) in enumerate(elimVars)
    if containsDerCall(SimulationCode.toDAEExp(elimEqs[i].exp))
      continue
    end
    if varName in aliasNames
      continue
    end
    local elimSym = Symbol(varName)
    local residualExpr = expToJuliaExpMTK(elimEqs[i].exp, simCode; derSymbol = false)
    if !isempty(relayAliases)
      residualExpr = _substSyms(residualExpr, relayAliases)
    end
    push!(solveBodyExprs, quote
      local _elimResidual = $(residualExpr)
      local _elimRhs = Symbolics.solve_for(0 ~ _elimResidual, $(elimSym))
      push!(_elimObsEqs, $(elimSym) ~ _elimRhs)
    end)
  end
  return quote
    #= Build eliminated non-dynamic variables at function scope so they
       share the function-scope `@independent_variables t` object with the
       main system, then bind their names into the module namespace via a
       single eval (with the Num objects embedded by value). =#
    local _elimBatch = Expr(:block)
    for _elimName in $(allElimSymbols)
      local _elimVar = Symbolics.variable(_elimName,
                                          T = Symbolics.FnType{Tuple, Real, Nothing})(t)
      push!(_elimBatch.args, :($_elimName = $_elimVar))
    end
    eval(_elimBatch)
    #= Solve residuals and create observed equations. Wrapped in a function
       + invokelatest to handle world-age from the preceding eval. Variables
       whose residual contained a der() are skipped here but still have
       bindings above, so any sibling residual referencing them resolves. =#
    function _solveEliminatedObserved()
      local _elimObsEqs = Symbolics.Equation[]
      $(solveBodyExprs...)
      return _elimObsEqs
    end
    append!(observedEqs, Base.invokelatest(_solveEliminatedObserved))
  end
end

"""
   Creates equations from the residual equations in unsorted order
"""
function createResidualEquationsMTK(stateVariables::Vector, algebraicVariables::Vector, equations::AbstractVector, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  if isempty(equations)
    return Expr[]
  end
  local eqs::Vector{Expr} = Expr[]
  for eq in equations
    #= eq.exp is `SimulationCode.Exp` post Phase 4b field migration; the
       `expToJuliaExpMTK(::SimulationCode.Exp, ...)` overload in
       MTK_CodeGenerationUtil.jl walks SIM Exp natively for the
       supported variants and delegates the rest back to the DAE
       emitter via `toDAEExp`. =#
    local eqExp = :(0 ~ $(expToJuliaExpMTK(eq.exp, simCode; derSymbol=false)))
    push!(eqs, eqExp)
  end
    return eqs
end

"""
  Generates the initial value for the equations.
  Algebraics without an explicit `start =` and without `fixed = true` are
  always skipped — MTK's init solver supplies defaults.
  States and OCC vars emit `0.0` defaults so MTK ODEProblem has a value for
  every unknown, unless `skipDefaultStateStarts` is true (used in the
  final-guess pass when an explicit user start is already pinned elsewhere).
"""
function createStartConditionsEquationsMTK(states::Vector,
                                        algebraics::Vector,
                                        simCode::SimulationCode.SIM_CODE;
                                        skipDefaultStateStarts::Bool = false)::Vector{Expr}
  local algInit = getStartConditionsMTK(algebraics, simCode; skipDefaultStarts = true)
  local stateInit = getStartConditionsMTK(states, simCode; skipDefaultStarts = skipDefaultStateStarts)
  local initialEquations = simCode.initialEquations
  local ieqInit = generateInitialEquations(initialEquations, simCode)
  #=
    Start with the start conditions above.
    Generate the equations in order afterwards
  =#
  #= Place the initial equations last =#
  return vcat(algInit, stateInit, ieqInit)
end

"""
  Generates initial equations as Symbolics `lhs ~ rhs` Equation forms suitable
  for passing to MTK's `initialization_eqs` kwarg of `System(...)`. Unlike the
  `=>` pair form (which acts as a guess only), `~` form is a real constraint
  that MTK's initialization solver must satisfy at t=0. Required for models
  with `InitialOutput` init mode (e.g. PID controllers' integrator state).
"""
function generateInitialEquationsAsConstraints(initialEqs, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local result = Expr[]
  for ieq in initialEqs
    if ieq isa BDAE.COMPLEX_EQUATION || ieq isa BDAE.ARRAY_EQUATION || ieq isa SimulationCode.ARRAY_EQUATION
      @debug "[MTK GEN: initialConstraints] skipping $(typeof(ieq)) (record/array constraints not yet lowered to scalar `~` form)"
      continue
    end
    if isParametricOnlyEquation(ieq, simCode)
      continue
    end
    local ieqLhsDAE = SimulationCode.toDAEExp(ieq.lhs)
    local ieqRhsDAE = SimulationCode.toDAEExp(ieq.rhs)
    local lhs = try
      expToJuliaExpMTK(ieqLhsDAE, simCode)
    catch err
      @warn "[CODEGEN: initialConstraints] failed to lower LHS; constraint dropped" lhs=ieqLhsDAE err
      continue
    end
    local rhs = try
      @match ieqRhsDAE begin
        DAE.CREF(DAE.CREF_IDENT("time", _, _), _) => expToJuliaExpMTK(ieqRhsDAE, simCode)
        DAE.CREF(__) => begin
          local crefAsStr = string(ieqRhsDAE)
          if haskey(simCode.stringToSimVarHT, crefAsStr)
            local simCodeVar = last(simCode.stringToSimVarHT[crefAsStr])
            if SimulationCode.isStateOrAlgebraic(simCodeVar)
              expToJuliaExpMTK(ieqRhsDAE, simCode)
            elseif SimulationCode.hasBindingExp(simCodeVar)
              evalSimCodeParameter(simCodeVar, simCode)
            else
              expToJuliaExpMTK(ieqRhsDAE, simCode)
            end
          else
            expToJuliaExpMTK(ieqRhsDAE, simCode)
          end
        end
        _ => evalDAE_Expression(ieqRhsDAE, simCode)
      end
    catch err
      @warn "[CODEGEN: initialConstraints] failed to lower RHS; constraint dropped" rhs=ieqRhsDAE err
      continue
    end
    push!(result, :($lhs ~ $rhs))
  end
  return result
end

"""
  Generates initial equations.
  Currently unsorted unless they are sorted before being passed to the simulation code phase.
"""
function generateInitialEquations(initialEqs, simCode::SimulationCode.SIM_CODE; parameterAssignment = true)::Vector{Expr}
  local initialEqsExps = Expr[]
  for ieq in initialEqs
    #= COMPLEX_EQUATION/ARRAY_EQUATION should have been expanded before this point =#
    if ieq isa BDAE.COMPLEX_EQUATION || ieq isa BDAE.ARRAY_EQUATION || ieq isa SimulationCode.ARRAY_EQUATION
      error("generateInitialEquations: unexpected unexpanded $(typeof(ieq)) in initial equations — this is a compiler bug upstream")
    end
    #= Skip parametric-only initial equations (already solved by solveParametricInitialEquations!) =#
    if isParametricOnlyEquation(ieq, simCode)
      continue
    end
    local ieqLhsDAE = SimulationCode.toDAEExp(ieq.lhs)
    local ieqRhsDAE = SimulationCode.toDAEExp(ieq.rhs)
    #= LHS will typically be a variable. Don't have to be though.. =#
    lhs = expToJuliaExpMTK(ieqLhsDAE, simCode)
    rhs = @match ieqRhsDAE begin
      #= `time` is the independent variable and never appears in
         stringToSimVarHT. Route it directly through expToJuliaExpMTK
         which emits the Julia symbol `t` for it. Without this guard
         the generic DAE.CREF arm below indexes the HT with key
         `"time"` and throws KeyError. Surfaced by models like
         Modelica.Fluid.Examples.ControlledTankSystem.ControlledTanks
         whose initial equations contain `<var> = time`. =#
      DAE.CREF(DAE.CREF_IDENT("time", _, _), _) => begin
        expToJuliaExpMTK(ieqRhsDAE, simCode)
      end
      DAE.CREF(__) => begin
        #= Evaluate the right hand side at this point =#
        local crefAsStr = string(ieqRhsDAE)
        local simCodeVar = last(simCode.stringToSimVarHT[crefAsStr])
        local res = if SimulationCode.isStateOrAlgebraic(simCodeVar)
          expToJuliaExpMTK(ieqRhsDAE, simCode)
        elseif SimulationCode.hasBindingExp(simCodeVar)
          evalSimCodeParameter(simCodeVar, simCode)
        else
          #= Parameter without binding (fixed=false): leave as symbol =#
          expToJuliaExpMTK(ieqRhsDAE, simCode)
        end
      end
      #= For more complicated expressions, we do local constant folding. =#
      _ => begin
        res = evalDAE_Expression(ieqRhsDAE, simCode)
        res
      end
    end
    if parameterAssignment
      push!(initialEqsExps,
            quote
              $lhs => $rhs
            end)
    else
      push!(initialEqsExps,
            quote
              $lhs = $rhs
            end)
    end
  end
  return initialEqsExps
end

"""
  Given a vector of variables and the simulation code
  extracts the start attributes to generate initial conditions.

If `skipDefaultStarts` is true, variables without explicit start values are skipped.
When false, variables without start values get default 0.0 initialization.
"""
function getStartConditionsMTK(vars::Vector, simCode::SimulationCode.SIM_CODE; skipDefaultStarts = false)::Vector{Expr}
  local startExprs::Vector{Expr} = Expr[]
  local residuals = simCode.residualEquations
  local ht::Dict = simCode.stringToSimVarHT
  local missingStartWarnings = Set{String}()
  if length(vars) == 0
    return Expr[]
  end
  for var in vars
    (index, simVar) = ht[var]
    varName = simVar.name
    local simVarType = simVar.varKind
    local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
    () = @match optAttributes begin
      SOME(attributes) => begin
        () = @match (attributes.start, attributes.fixed) begin
          (SOME(DAE.CREF(start)), SOME(__)) || (SOME(DAE.CREF(start)), _)  => begin
            #= Delegate to expToJuliaExpMTK so DATA_STRUCTURE / PARAMETER /
               subscripted CREFs are all handled uniformly. The previous
               two-branch split would emit `pars[name]` for non-subscripted
               CREFs, which only works when the referenced var is a
               PARAMETER (in `pars`). DATA_STRUCTURE constants and
               int/enum vars reclassified by Causalize are not in `pars`. =#
            push!(startExprs,
                  quote
                    $(Symbol("$varName")) => $(expToJuliaExpMTK(DAE.CREF(start, DAE.T_REAL(MetaModelica.Nil())), simCode))
                  end)
            continue
          end
          (SOME(start), SOME(fixed)) || (SOME(start), _)  => begin
            push!(startExprs,
                  quote
                    $(Symbol("$varName")) => $(expToJuliaExpMTK(start, simCode))
                  end)
            continue
          end
          (NONE(), SOME(fixed)) => begin
            #= `fixed = true` with no `start` pins the var at 0.0; honour even when
               default-skipping is on. `fixed = false` / non-Bool: MTK's init solver
               handles it, so skip emission in skip mode. =#
            local _fixedTrue = fixed isa DAE.BCONST && fixed.bool
            if skipDefaultStarts && !_fixedTrue
              continue
            end
            push!(startExprs, :($(Symbol(varName)) => 0.0))
            continue
          end
          (NONE(), NONE()) || (_, _) => begin
            #= No start value specified, default to 0.0 =#
            if !skipDefaultStarts
              push!(missingStartWarnings, varName)
              push!(startExprs, :($(Symbol(varName)) => 0.0))
            end
            continue
          end
        end
      end
      NONE() where {!skipDefaultStarts} => begin
        #=
        If no attribute. Let it default to zero.
        This branch should only be taken for compiler generated variables.
        =#
        push!(startExprs, :($(Symbol(varName)) => 0.0))
        continue
      end
      _ => begin
        continue
      end
    end
  end
  if OMBackend.WARN_MISSING_START_VALUES[] && !isempty(missingStartWarnings)
    local warningList = sort!(collect(missingStartWarnings))
    local maxShown = 20
    local shown = warningList[1:min(end, maxShown)]
    local omitted = length(warningList) - length(shown)
    local summary = "Assumed starting value of 0.0 for $(length(warningList)) variable(s): " * join(shown, ", ")
    if omitted > 0
      summary *= ", ... (+$(omitted) more)"
    end
    @warn summary
  end
  return startExprs
end

"""
  Emit `lhs ~ rhs` constraint Equations for state vars with `fixed=true` and an
  explicit `start`. Goes into `initialization_eqs` so MTK pins them at t=0
  rather than treating them as soft `guesses` the iteration may override.
"""
function getFixedStartConstraintsMTK(vars::Vector, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local result::Vector{Expr} = Expr[]
  if isempty(vars)
    return result
  end
  local ht::Dict = simCode.stringToSimVarHT
  local probe = Tuple{String, String, String}[]
  for var in vars
    (index, simVar) = ht[var]
    local varName = simVar.name
    local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
    local startExp = @match optAttributes begin
      SOME(attributes) => begin
        local startStr = string(attributes.start)
        local fixedStr = string(attributes.fixed)
        if occursin("Inertia_w", varName) || occursin("Inertia_phi", varName) || occursin("Cylinder_s", varName)
          push!(probe, (varName, startStr, fixedStr))
        end
        @match (attributes.start, attributes.fixed) begin
          (SOME(s), SOME(DAE.BCONST(true))) => s
          _ => nothing
        end
      end
      _ => nothing
    end
    if startExp === nothing
      continue
    end
    push!(result, :($(Symbol(varName)) ~ $(expToJuliaExpMTK(startExp, simCode))))
  end
  return result
end

"""
  Emit `Expr`s that push init-algorithm-derived (state => value) pairs into the
  `finalInitialValues` vector inside Model(). Mirrors `emitInitAlgConstraintAppends`
  but the push target is the u0-pair list (consumed by `ODEProblem(...; u0 = ...)`)
  rather than `initialization_eqs`. The merge is required because the pure-ODE
  branch of the ODEProblem build skips MTK's init solver, so the init-eq alone
  would not propagate the init-algorithm value into u0.
"""
function emitInitAlgU0Appends(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local appends::Vector{Expr} = Expr[]
  isempty(simCode.initialAlgorithms) && return appends
  local ht::Dict = simCode.stringToSimVarHT
  local lhsNames = Set{String}()
  local rhsNames = Set{String}()
  if any(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms)
    for ia in simCode.initialAlgorithms, s in ia.daeStatements
      _collectInitAlgLhsRhsCrefsDAE!(lhsNames, rhsNames, s)
    end
  else
    for ia in simCode.initialAlgorithms, op in ia.statements
      _collectInitAlgLhsRhsCrefs!(lhsNames, rhsNames, op)
    end
  end
  for name in lhsNames
    haskey(ht, name) || continue
    local (_, sv) = ht[name]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      continue
    end
    local qn = QuoteNode(Symbol(name))
    push!(appends, :(haskey(_algResults, $(qn)) &&
                     push!(fiv, $(Symbol(name)) => _algResults[$(qn)])))
  end
  return appends
end

"""
  Emit `Expr`s that conditionally push init-algorithm-derived constraints into
  the local `_eqs` vector inside `_buildInitialConstraintEqs`. Each emitted line
  looks like `haskey(_algResults, :T_start) && push!(_eqs, T_start ~ _algResults[:T_start])`.
"""
function emitInitAlgConstraintAppends(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local appends::Vector{Expr} = Expr[]
  isempty(simCode.initialAlgorithms) && return appends
  local ht::Dict = simCode.stringToSimVarHT
  local lhsNames = Set{String}()
  local rhsNames = Set{String}()
  if any(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms)
    for ia in simCode.initialAlgorithms, s in ia.daeStatements
      _collectInitAlgLhsRhsCrefsDAE!(lhsNames, rhsNames, s)
    end
  else
    for ia in simCode.initialAlgorithms, op in ia.statements
      _collectInitAlgLhsRhsCrefs!(lhsNames, rhsNames, op)
    end
  end
  for name in lhsNames
    haskey(ht, name) || continue
    local (_, sv) = ht[name]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      continue
    end
    local qn = QuoteNode(Symbol(name))
    push!(appends, :(haskey(_algResults, $(qn)) &&
                     push!(_eqs, $(Symbol(name)) ~ _algResults[$(qn)])))
  end
  return appends
end


"""
  Creates the components of the If-Equations.
Each if equation is marked by the identifier.
So the first will have 1 and so on.
"""
#= Build one SymbolicContinuousCallback per deferred pure-time event. Callback K
   fires at event K's threshold, so its affect sets ITS OWN ifCond to the known
   post-crossing value (`numVal`, exactly what the per-branch toggle sets) and
   re-derives every OTHER pure-time ifCond from its zero-crossing sign
   (`zc < 0` <=> condition TRUE). Reading the firing event's own `zc` is unusable
   because it is exactly 0 at the crossing instant; the other events are not at
   their crossing so their sign is definite. Whichever callback fires refreshes
   all, so coincident time events stay consistent even though MTK/DiffEq apply
   only one affect per coincident root. =#
function _buildTimeEventRefreshCallbacks(allPT::Vector)
  local n = length(allPT)
  local cbs = Expr[]
  for k in 1:n
    local mtkCondK = allPT[k][3]
    local numValK = allPT[k][4]
    local obsKws = Expr[]
    local retKws = Expr[]
    local modKws = Expr[]
    for j in 1:n
      local symJ = allPT[j][1]
      push!(modKws, Expr(:kw, symJ, symJ))
      if j == k
        push!(retKws, Expr(:kw, symJ, numValK))
      else
        local zcName = Symbol("_zc", j)
        push!(obsKws, Expr(:kw, zcName, allPT[j][2]))
        push!(retKws, Expr(:kw, symJ, :((observed.$(zcName) < 0) ? 1.0 : 0.0)))
      end
    end
    local modNT = Expr(:tuple, Expr(:parameters, modKws...))
    local retNT = Expr(:tuple, Expr(:parameters, retKws...))
    local fExpr = :((modified, observed, ctx, integrator) -> $(retNT))
    local affect
    if isempty(obsKws)
      affect = :(ModelingToolkit.ImperativeAffect($(fExpr), $(modNT)))
    else
      local obsNT = Expr(:tuple, Expr(:parameters, obsKws...))
      affect = :(ModelingToolkit.ImperativeAffect($(fExpr), $(modNT); observed = $(obsNT)))
    end
    push!(cbs, :(ModelingToolkit.SymbolicContinuousCallback(
      ($(mtkCondK)) => $(affect);
      reinitializealg = SciMLBase.NoInit()
    )))
  end
  return cbs
end

function createIfEquations(stateVariables, algebraicVariables, simCode)
  local ifEquations = IfEquationComponent[]
  local identifier::Int
  local sortedIfEquations = sort(collect(simCode.ifEquations);
                                 by = ifEq -> _ifEquationSortKey(ifEq, simCode))
  #= The identifier is increased by 1 in each iteration. =#
  for (identifier, ifEq) in enumerate(sortedIfEquations)
    push!(ifEquations, createIfEquation(stateVariables, algebraicVariables, ifEq, identifier, simCode))
  end
  #= Pure-time-event branches deferred their callbacks (see createIfEquation);
     build the model-level refresh callbacks now that every if-equation's
     pure-time conditions are known. =#
  local allPT = collect(Iterators.flatten(c.pureTimeEvents for c in ifEquations))
  if !isempty(allPT)
    local refreshCbs = _buildTimeEventRefreshCallbacks(allPT)
    push!(ifEquations, IfEquationComponent(refreshCbs, Expr[], Symbol[],
                                           Tuple{String, Bool}[], Tuple{Symbol, Any, Any, Float64}[]))
  end
  return ifEquations
end

function _ifEquationSortKey(ifEq::SimulationCode.IF_EQUATION, simCode)::String
  local targets = String[]
  try
    for branch in ifEq.branches
      for resEq in branch.residualEquations
        push!(targets, string(last(deCausalize(resEq, simCode))))
      end
      isempty(targets) || break
    end
  catch
    empty!(targets)
  end
  if isempty(targets)
    try
      for branch in ifEq.branches
        push!(targets, string(branch.condition))
      end
    catch
      return ""
    end
  end
  sort!(targets)
  return join(targets, "|")
end

function _ifConditionDependsOnTime(@nospecialize(condition))::Bool
  local refs::Set{String} = Set{String}()
  try
    SimulationCode.collectCrefNames!(refs, condition)
  catch
    return false
  end
  return "time" in refs
end

"""
    _ifConditionAllDiscreteOrParameter(condition, simCode) -> Bool

Return true when every variable reference in `condition` is a DISCRETE
simvar, a PARAMETER, or a constant (no STATE / ALG_VARIABLE / `time`
reference). For such conditions the value flips only through the
equations that define the discrete variables — MTK propagates the
update through the residual system on its own, so an explicit
SymbolicContinuousCallback would only add chatter without contributing
new event semantics.

Returns false on any reference to a continuous unknown so the caller
knows a continuous event is still required.
"""
function _ifConditionAllDiscreteOrParameter(@nospecialize(condition), simCode)::Bool
  local refs::Set{String} = Set{String}()
  try
    SimulationCode.collectCrefNames!(refs, condition)
  catch
    return false
  end
  isempty(refs) && return false
  local ht = simCode.stringToSimVarHT
  for name in refs
    name == "time" && return false
    local entry = get(ht, name, nothing)
    entry === nothing && return false
    local kind = entry[2].varKind
    if !(kind isa SimulationCode.DISCRETE || kind isa SimulationCode.PARAMETER)
      return false
    end
  end
  return true
end

"""
    _ifConditionIsPureTimeEvent(condition, simCode) -> Bool

Return true when `condition` is a deterministic time event: it references
`time` and every other reference is a PARAMETER (no STATE / ALG / DISCRETE).
The transition instant is then fixed a priori, so coincident time events
(two sources transitioning at the same instant) must all be applied at once.
Conservative: any non-parameter reference returns false, keeping the default
per-branch continuous callback.
"""
function _ifConditionIsPureTimeEvent(@nospecialize(condition), simCode)::Bool
  local refs::Set{String} = Set{String}()
  try
    SimulationCode.collectCrefNames!(refs, condition)
  catch
    return false
  end
  ("time" in refs) || return false
  local ht = simCode.stringToSimVarHT
  for name in refs
    name == "time" && continue
    local entry = get(ht, name, nothing)
    entry === nothing && return false
    (entry[2].varKind isa SimulationCode.PARAMETER) || return false
  end
  return true
end

"""
True when the zero-crossing Expr references a non-lifted algebraic variable
(one whose static `evalInitialCondition` value defaults to 0). Lifted helper
names (`ifEq_tmp*`, `ifCond*`) are excluded so an init affect never observes
another lifted value and forms a circular init dependency.
"""
function _zcReferencesSolvableAlgebraic(@nospecialize(zcExpr), simCode)::Bool
  local ht = simCode.stringToSimVarHT
  local stack = Any[zcExpr]
  while !isempty(stack)
    local node = pop!(stack)
    if node isa Symbol
      local key = string(node)
      if !startswith(key, "ifEq_tmp") && !startswith(key, "ifCond") && haskey(ht, key)
        local (_, sv) = ht[key]
        if SimulationCode.isAlgebraic(sv)
          return true
        end
      end
    elseif node isa Expr
      for a in node.args
        push!(stack, a)
      end
    end
  end
  return false
end

"""
This function creates symbolic if equations for use in MTK.
The function returns a tuple, where the first part of the tuple represent the conditions and the affect of the if-equation on the form:
  continuous_events = [
    <Condition> => <affect>
    <Condition> => <affect>
    ....
  ]
Each condition generates one variable with zero dynamics the variable being true or not depending on the branch.
  Example:
  if <condition> then
    <equations>
  elseif <condition> then
    <equations>
  else
    <equations>
  end if;
Would result in:
continuous_events = [
    <condition> => [ifCond1 ~ true, ifCond2 ~ false]
    <condition> => [ifCond1 ~ false, ifCond2 ~ true]
]
An if equation with a single condition would only generate one condition:
continuous_events = [
    <condition> => [ifCond1 ~ true]
]

The second value in the return tuple represent the if-equations itself:
<lhs> = IfElse.ifelse(<condition>, <value>, IfElse.ifelse(<condition>, <value>, <value>))
  lhs can be one or several variables. (TODO, fix the case for several variables in this kind of branch)

The third part of the tuple contains a set of zero dynamic equations (One for each if equation condition variable)
See the following issue: https://github.com/SciML/ModelingToolkit.jl/issues/1523

The forth part of the tuple contains a vector of symbolic variables.
One for each conditional variable created.
"""
function createIfEquation(stateVariables::Vector,
                          algebraicVariables::Vector,
                          ifEq::SimulationCode.IF_EQUATION,
                          identifier::Int,
                          simCode)::IfEquationComponent
  local i::Int = 0
  local nBranches::Int = length(ifEq.branches)
  local branchesWithConds::Int = nBranches - 1
  #= Collect all ifCond symbols for this if-equation.
     These are parameters modified by imperative affects. =#
  local allIfCondSyms = [Symbol(string("ifCond", identifier, j)) for j in 1:branchesWithConds]
  local conditions = Expr[]
  local ivConditions = Bool[]
  local pureTimeEvents = Tuple{Symbol, Any, Any, Float64}[]
  for branch in ifEq.branches
    i += 1
    @match branch begin
      SimulationCode.BRANCH(condition, residuals, -1 #= Else =#, targets, _, _, _, _, _) => begin
      end
      SimulationCode.BRANCH(condition, residuals, _, targets, _, _, _, _, _) => begin
        local mtkCond = transformToMTKContinousConditionEquation(branch.condition, simCode)
        #= Evaluate the initial value condition. =#
        local ivCond = evalInitialCondition(mtkCond, simCode)
        local numVal = ivCond ? 1.0 : 0.0
        local invVal = ivCond ? 0.0 : 1.0
        #= Build ImperativeAffect: function returns a NamedTuple of new values.
           modified NamedTuple maps aliases to the symbolic parameter variables.
           Callback fires for every branch condition; ifCondN is the load-bearing
           branch switch the residual ifelse reads. =#
        local returnKws::Vector{Expr} = Expr[Expr(:kw, sym, (j == i) ? numVal : invVal) for (j, sym) in enumerate(allIfCondSyms)]
        local returnNT::Expr = Expr(:tuple, Expr(:parameters, returnKws...))
        local fExpr::Expr = :((modified, observed, ctx, integrator) -> $returnNT)
        local modifiedKws::Vector{Expr} = Expr[Expr(:kw, sym, sym) for sym in allIfCondSyms]
        local modifiedNT::Expr = Expr(:tuple, Expr(:parameters, modifiedKws...))
        local affectTuple::Expr = :(($(fExpr), $(modifiedNT)))
        #= When the branch condition depends on a non-lifted algebraic variable
           (an operating-point value `evalInitialCondition` defaulted to 0, e.g.
           an op-amp input voltage), the static initial ifCond can be wrong with
           no zero-crossing to fire the affect. Add an `initialize` affect that
           re-evaluates the condition from the solved state (mirrors
           evalInitialCondition: zc < 0 means the condition is TRUE). Restricted
           to non-lifted algebraic zc so it does not observe other lifted ifEq_tmp
           values (which would form a circular init dependency). =#
        local zcLhs = _extractZeroCrossingLHS(mtkCond)
        local thisSym::Symbol = allIfCondSyms[i]
        if _ifConditionIsPureTimeEvent(branch.condition, simCode)
          #= Deterministic time event: defer to model-level refresh callbacks built
             in createIfEquations, so two sources whose transitions coincide cannot
             drop one another's affect. `numVal` is the post-crossing ifCond value
             (same value the per-branch toggle would set). The ifCond parameter is
             still declared and initialised below via ivConditions. =#
          push!(pureTimeEvents, (thisSym, zcLhs, mtkCond, numVal))
          push!(ivConditions, ivCond)
        else
          local cond::Expr
          if _zcReferencesSolvableAlgebraic(zcLhs, simCode)
            local initObservedNT::Expr = Expr(:tuple, Expr(:parameters, Expr(:kw, :zc, zcLhs)))
            local initModifiedNT::Expr = Expr(:tuple, Expr(:parameters, Expr(:kw, thisSym, thisSym)))
            local initRetNT::Expr = Expr(:tuple, Expr(:parameters, Expr(:kw, thisSym, :((observed.zc < 0) ? 1.0 : 0.0))))
            local initFExpr::Expr = :((modified, observed, ctx, integrator) -> $initRetNT)
            local initAffect::Expr = :(ModelingToolkit.ImperativeAffect($(initFExpr), $(initModifiedNT); observed = $(initObservedNT)))
            cond = :(ModelingToolkit.SymbolicContinuousCallback(
              ($(mtkCond)) => $(affectTuple);
              initialize = $(initAffect),
              reinitializealg = SciMLBase.NoInit()
            ))
          else
            cond = :(ModelingToolkit.SymbolicContinuousCallback(
              ($(mtkCond)) => $(affectTuple);
              reinitializealg = SciMLBase.NoInit()
            ))
          end
          push!(conditions, cond)
          push!(ivConditions, ivCond)
        end
      end
    end
  end
  #= Create the equations themselves =#
  local target = 1
  local resEqs = ifEq.branches[target].residualEquations
  local ifExpressions = Expr[]
  #= The number of residuals is the same for both branches. =#
  local nResEqsInTarget = length(resEqs)
  for resEqIdx in 1:nResEqsInTarget
    local resEq = resEqs[resEqIdx]
    push!(ifExpressions,
          :($(last(deCausalize(resEq, simCode))) ~ $(generateIfExpressions(ifEq.branches,
                                                                           target,
                                                                           resEqIdx,
                                                                           identifier,
                                                                           simCode;
                                                                           subIdentifier = 1))))
  end
  #= ifCond variables are discrete parameters (not ODE unknowns), so they do
     not need der() ~ 0 equations. Collect their names and initial values for
     parameter declaration. =#
  local conditionVariables = Symbol[]
  local conditionVariableNames = Tuple{String, Bool}[]
  for i in 1:length(ivConditions)
    push!(conditionVariables, Symbol(string("ifCond", identifier, i)))
    push!(conditionVariableNames, (string("ifCond", identifier, i), !(ivConditions[i])))
  end
  return IfEquationComponent(conditions, ifExpressions,
                             conditionVariables, conditionVariableNames, pureTimeEvents)
end

"""
  `createParameterEquationsMTK(parameters::Vector, type, simCode::SimulationCode.SIM_CODE)`
    The Type specifies what kind of parameter equation a call to this function should yield.
"""
function createParameterEquationsMTK(parameters::Vector, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local parameterEquations::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in parameters
    (index, simVar) = ht[param]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => begin
        exp
      end
      #= We have a parameter without a binding. Check if we have a start attribute...=#
      SimulationCode.PARAMETER(__) => begin
        local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
        @match optAttributes begin
          SOME(attr) where attr.start isa SOME => begin
            @assert !(attr.start.data isa DAE.CREF) "Non-numeric start attributes are not currently supported"
            @match SOME(startVal) = attr.start
            startVal
          end
          #= Either NONE() for missing attributes, or SOME(attr) whose start is
             NONE(). Both collapse to the default-float path. Without this
             catch-all the match fails on SOME{VariableAttributes} whose start
             is unset (e.g. several Blocks.Examples.Filter variants). =#
          _ => DAE.RCONST(0.0)
        end
      end
      SimulationCode.STRING(__) => begin
        @warn "String parameter $(param) found in numeric parameter list; skipping."
        continue
      end
      _ => begin
        throw(ErrorException("Unknown SimulationCode.SimVarType for parameter: " * string(param)  * " of type: " * string(simVarType)))
      end
    end
    #=
      Check if conversions are needed.
      Both sides of the Pair are wrapped with `Symbolics.wrap` to keep the
      pair element type at `Pair{Num, Num}`. Without the LHS wrap MTK fails
      `convert(Pair{Num}, Pair{BasicSymbolicImpl{SymReal}, Float64})` on
      models like SpeedControlledDCPM where the parameter symbol resolves
      to a bare `BasicSymbolic`. `Symbolics.wrap` is a no-op when the input
      is already a `Num`.
    =#
    expr = if isIntOrBool(bindExp)
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        Symbolics.wrap($(Symbol(simVar.name))) => Symbolics.wrap(float($((expToJuliaExpMTK(bindExp, simCode)))))
      end
    else
        :(Symbolics.wrap($(Symbol(simVar.name))) => Symbolics.wrap($(expToJuliaExpMTK(bindExp, simCode))))
    end
      # expr = quote
      #   $(LineNumberNode(@__LINE__, "$param eq"))
      #   $(Symbol(simVar.name)) => float($((expToJuliaExpMTK(bindExp, simCode))))
      # end
    push!(parameterEquations, expr)
  end #=For=#
  return parameterEquations
end

"""
  Creates array parameter definitions for MTK.
  Array parameters (e.g. record fields like R_T::Real[3,3]) are created as
  concrete Julia arrays assigned to their symbol names, so that the generated
  algorithmic functions can subscript into them.
"""
function createArrayParametersMTK(arrayParameters::Vector, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local exprs = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in arrayParameters
    (_, simVar) = ht[param]
    local vk = simVar.varKind
    @match vk begin
      SimulationCode.ARRAY_PARAMETER(dims, SOME(bindExp)) => begin
        local valExpr = expToJuliaExpMTK(bindExp, simCode)
        push!(exprs, :($(Symbol(simVar.name)) = $(valExpr)))
      end
      SimulationCode.ARRAY_PARAMETER(dims, NONE()) => begin
        #= No binding, create zero array with the right dimensions =#
        push!(exprs, :($(Symbol(simVar.name)) = zeros(Float64, $(dims...))))
      end
      _ => nothing
    end
  end
  return exprs
end

"""
  Creates parameters assignments *(:=) on a MTK parameters compatible format.
"""
function createParameterAssignmentsMTK(parameters::Vector,
                                       simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local parameterEquations::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in parameters
    (index, simVar) = ht[param]
    local simVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => exp
      SimulationCode.PARAMETER(__) =>  begin
        continue
      end
      _ => continue
    end
    #= Solution for https://github.com/SciML/ModelingToolkit.jl/issues/991 =#
    #TODO: Is this workaround still relevant? John 2023-02-22
    expr =  if isIntOrBool(bindExp)
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        $(Symbol(simVar.name)) = float($((expToJuliaExpMTK(bindExp, simCode))))
      end
    else
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        $(Symbol(simVar.name)) = $(expToJuliaExpMTK(bindExp, simCode))
      end
    end
    push!(parameterEquations, expr)
  end
  return parameterEquations
end


"""
  createStringParameterAssignments(simCode) -> Vector{Expr}
Emit one module-level Julia assignment per Modelica `String` parameter, e.g.
```julia
table2_combiTimeTable_fileName = "NoName"
lossTable_fileName = "NoName"
```
"""
function createStringParameterAssignments(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local exprs::Vector{Expr} = Expr[]
  for varName in keys(simCode.stringToSimVarHT)
    (idx, simVar) = simCode.stringToSimVarHT[varName]
    local bindExp = @match simVar.varKind begin
      SimulationCode.STRING(bindExp = SOME(e)) => e
      SimulationCode.PARAMETER(bindExp = SOME(e)) where _isLiteralBind(e) => e
      _ => nothing
    end
    bindExp === nothing && continue
    #= Only emit literal bindings at module level. Computed defaults / cross-
       parameter refs cannot be safely lowered before MTK builds `pars`. The
       DATA_STRUCTURE_ASSIGNMENTS at module top reference these names (e.g.
       CombiTimeTable's `startTime` / `shiftTime` / `fileName`); without an
       emission step, loading the module raises UndefVarError. =#
    local rhs = try
      expToJuliaExpMTK(bindExp, simCode)
    catch
      continue
    end
    push!(exprs, :( $(Symbol(simVar.name)) = $(rhs) ))
  end
  return exprs
end

#= Emit ARRAY_PARAMETER bindings at module top so that DATA_STRUCTURE
   constructor calls (CombiTable / CombiTimeTable / ExternalObject) can
   reference them by their bare Julia name. Without this, the in-function
   emission via createArrayParametersMTK happens too late: it lives inside
   `function <Model>Model(tspan)`, while DATA_STRUCTURE_ASSIGNMENTS run at
   module load time. =#
#= Module-level prelude for array parameters referenced by DATA_STRUCTURE
   constructors. Two cases:

     1. The HT carries a single ARRAY_PARAMETER entry with a literal-array bind.
        Emit `name = <array-expr>` directly.
     2. The HT carries scalarized entries (e.g. `tableData[1][1]`,
        `tableData[1][2]`, ..., `tableData[3][2]`) and the parent name has no
        bind of its own. Reconstruct an N-dim Julia matrix from the scalar
        element bindings and emit `tableData = <reconstructed>`. Required
        because ExternalObject constructors (CombiTable, CombiTimeTable, ...)
        appear at module top and reference the parent array name. =#
function createArrayParameterPrelude(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local exprs::Vector{Expr} = Expr[]
  local ht = simCode.stringToSimVarHT

  #= Restrict the prelude to arrays actually referenced by DATA_STRUCTURE
     constructor calls. Emitting every ARRAY_PARAMETER at module top would
     shadow MTK's per-model parameter handling for arrays not needed at
     module-load time (e.g. body_r_CM in MultiBody models), perturbing the
     resulting integration trajectory. =#
  local neededBases = Set{String}()
  for (_, (_, simVar)) in ht
    @match simVar.varKind begin
      SimulationCode.DATA_STRUCTURE(SOME(b)) => begin
        @match b begin
          SimulationCode.CALL(__) => SimulationCode.collectCrefNames!(neededBases, b)
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  #= Also include any ARRAY_PARAMETER whose subscripted form appears in
     residual equations. This rescues `world_gravityArrowHead_lengthDirection[2]`
     and the cluster of MultiBody examples where `eliminateDeadParameters` /
     `eliminateConstantParameters` did not substitute the subscripted CREF
     (parent was kept as ARRAY_PARAMETER but never emitted module-top), so
     MTK eval fails with `<name>[idx] not defined`. We collect base names of
     CREFs that appear in residuals and intersect with the set of
     ARRAY_PARAMETERs in HT so we only emit parents that actually exist. =#
  local _residualCrefs = Set{String}()
  for eq in simCode.residualEquations
    SimulationCode.collectCrefNames!(_residualCrefs, SimulationCode.toDAEExp(eq.exp))
  end
  for _n in _residualCrefs
    local _bracket = findfirst('[', _n)
    if _bracket !== nothing
      local _base = _n[1:_bracket-1]
      local _entry = get(ht, _base, nothing)
      _entry === nothing && continue
      local _isArr = @match _entry[2].varKind begin
        SimulationCode.ARRAY_PARAMETER(__) => true
        _ => false
      end
      _isArr && push!(neededBases, _base)
    end
  end
  #= Collect orphan subscripted CREFs (referenced in residuals, base NOT in HT)
     before the early-return so the defensive fallback at the end of this
     function still emits even when no DATA_STRUCTURE/ARRAY_PARAMETER paths
     fire. Recorded here so the fallback loop downstream can consume them. =#
  local _orphanRefsEarly = Set{String}()
  for _n in _residualCrefs
    local _bracket = findfirst('[', _n)
    _bracket === nothing && continue
    local _base = _n[1:_bracket-1]
    haskey(ht, _base) && continue
    haskey(ht, _n) && continue
    push!(_orphanRefsEarly, _n)
  end
  if isempty(neededBases)
    for _ref in _orphanRefsEarly
      push!(exprs, :( $(Symbol(_ref)) = 0.0 ))
    end
    return exprs
  end

  local emitted = Set{String}()
  for (varName, (_, simVar)) in ht
    local bindExp = @match simVar.varKind begin
      SimulationCode.ARRAY_PARAMETER(_, SOME(e)) => SimulationCode.toDAEExp(e)
      _ => nothing
    end
    bindExp === nothing && continue
    varName ∈ neededBases || continue
    varName ∈ emitted && continue
    push!(emitted, varName)
    local rhs = try
      expToJuliaExpMTK(bindExp, simCode)
    catch
      continue
    end
    push!(exprs, :( $(Symbol(simVar.name)) = $(rhs) ))
  end

  local scalarGroups = Dict{String, Vector{Tuple{Vector{Int}, Any, Int}}}()
  for (varName, (_, simVar)) in ht
    local bracketIdx = findfirst('[', varName)
    bracketIdx === nothing && continue
    local baseName = varName[1:bracketIdx-1]
    baseName ∈ neededBases || continue
    baseName ∈ emitted && continue
    local idxStr = varName[bracketIdx:end]
    local indices = Int[]
    for m in eachmatch(r"\[(\d+)\]", idxStr)
      push!(indices, parse(Int, m.captures[1]))
    end
    isempty(indices) && continue
    local val = @match simVar.varKind begin
      SimulationCode.PARAMETER(SOME(SimulationCode.RCONST(r))) => r
      SimulationCode.PARAMETER(SOME(SimulationCode.ICONST(i))) => i
      SimulationCode.PARAMETER(SOME(SimulationCode.BCONST(b))) => b
      _ => nothing
    end
    val === nothing && continue
    push!(get!(scalarGroups, baseName, Tuple{Vector{Int}, Any, Int}[]),
          (indices, val, length(indices)))
  end

  for (baseName, entries) in scalarGroups
    baseName ∈ emitted && continue
    local nDims = entries[1][3]
    all(e -> e[3] == nDims, entries) || continue
    local maxIdx = zeros(Int, nDims)
    for (idxs, _, _) in entries
      for d in 1:nDims
        maxIdx[d] = max(maxIdx[d], idxs[d])
      end
    end
    local expectedCount = prod(maxIdx)
    length(entries) == expectedCount || continue
    local elemType = isa(entries[1][2], Bool) ? Bool :
                     isa(entries[1][2], Integer) ? Int : Float64
    local arr = Array{elemType}(undef, maxIdx...)
    local complete = true
    for (idxs, val, _) in entries
      try
        arr[idxs...] = val
      catch
        complete = false
        break
      end
    end
    complete || continue
    push!(emitted, baseName)
    push!(exprs, :( $(Symbol(baseName)) = $(arr) ))
  end

  #= Defensive fallback: a residual references `<name>[i]` for some base
     that is NOT in HT (no ARRAY_PARAMETER, no scalarized PARAMETER) and was
     therefore not emitted by either of the two passes above. Observed for
     the `World` component's `gravityArrowHead.lengthDirection[2..3]` cluster
     on MultiBody examples (RollingWheel / Surfaces / RollingWheelSetDriving
     / ...): the frontend instantiates the parent record but never registers
     the per-index parameters as SimVars, so codegen leaks subscripted CREFs
     into the residual that point at nothing. Emit `var"<name>[i]" = 0.0`
     for every observed index — the codegen produces `Symbol("<name>[i]")`
     bindings, so the recovery variable has to be the bracketed name itself,
     not the parent. Default value 0.0 is wrong if the model actually uses
     the parameter dynamically, but for visualization-only constants
     (`gravityArrowHead`, axis arrows, ...) it is benign. =#
  local _orphanRefs = Set{String}()
  for _n in _residualCrefs
    local _bracket = findfirst('[', _n)
    _bracket === nothing && continue
    local _base = _n[1:_bracket-1]
    haskey(ht, _base) && continue
    haskey(ht, _n) && continue
    push!(_orphanRefs, _n)
  end
  for _ref in _orphanRefs
    push!(exprs, :( $(Symbol(_ref)) = 0.0 ))
  end

  return exprs
end

function createDataStructureAssignments(dataStructureVariables::Vector{String}, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local dsAssignments::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  #= Same Modelica-function name set used by rewriteEquations: needed to qualify
     bare calls (e.g. Modelica_Blocks_Types_ExternalCombiTimeTable_constructor)
     so they resolve to the OMBackend.CodeGeneration wrapper rather than failing
     with UndefVarError in the per-model module scope. Surfaces on every model
     using CombiTable / CombiTimeTable / ExternalObject constructors. =#
  local funcNames = Set{Symbol}(Symbol(f.name) for f in simCode.functions)
  for ds in dataStructureVariables
    (index, simVar) = ht[ds]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.DATA_STRUCTURE(bindExp = SOME(exp)) => exp
      _ => throw(ErrorException("createDataStructureAssignments: data structure variable $(ds) has no bound expression (got $(simVarType))."))
    end
    local rhs = expToJuliaExpMTK(bindExp, simCode)
    if rhs isa Expr
      qualifyModelicaFunctions!(rhs, funcNames)
    end
    expr = quote
      $(LineNumberNode(@__LINE__, "$ds eq"))
      $(Symbol(simVar.name)) = $(rhs)
    end
    push!(dsAssignments, expr)
  end
  return dsAssignments
end

"""
  Creates a parameter array.
  A parameter array is an array containing the values of the parameters sorted by index.
  The index here is the index assigned by the code generator earlier in the lowering
  of the hybrid DAE.
"""
function createParameterArray(parameters::Vector{T1},
                              parameterAssignments::Vector{T2},
                              simCode::SIM_T) where {T1, T2, SIM_T}
  local paramArray = []
  local hT = simCode.stringToSimVarHT
  for param in parameters
    (index, simVar) = hT[param]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => exp
      SimulationCode.PARAMETER(__) => begin
        @match simVar.attributes begin
          SOME(attr) where attr.start isa SOME => begin
            @match SOME(startVal) = attr.start
            startVal
          end
          _ => DAE.RCONST(0.0)
        end
      end
      _ => throw(ErrorException("createParameterArray: parameter $(param) has no bound expression (got $(simVarType))."))
    end
    #= Evaluate the parameters. If it is a variable, and can't be evaluated look it up in the parameter dictionary. =#
    local parValue
    try
      #= The boundvalue is known =#
      val = eval(expToJuliaExpMTK(bindExp, simCode))
      if val isa Float64
        parValue = :($(val))
      else
        parValue = :($(Symbol(param))) #:(0.0)
      end
    catch #=If the bound value is a more complex expression. =#
      parValue = :(0) #pars[Num($(param))]) (More complex parameters are yet to be used in the benchmark..)
    end
    push!(paramArray, parValue)
  end
  return paramArray
end

"""
 Decomposes the continuous variables into chunked constructor functions.
 Returns (outer_defs, inner_refs) where:
  - outer_defs: function definitions to be placed at module level (before model function)
  - inner_refs: the variableConstructors array assignment (inside model function)
 Constructor functions are defined at module level to avoid JIT overhead from
 compiling nested closures inside the model function.
 The `modelPrefix` avoids name collisions when multiple models are translated in one session.
"""
function decomposeVariables(stateVariables::Vector{Symbol}, algebraicVariables::Vector{Symbol};
                            modelPrefix::String = "")
  local stateVectors = collect(Iterators.partition(stateVariables, CHUNK_SIZE[]))
  local algVectors = collect(Iterators.partition(algebraicVariables, CHUNK_SIZE[]))
  local outerDefs = Expr[]
  local constructorNames = Symbol[]
  local i = 1::Int
  for stateVector in stateVectors
    local fName = Symbol(modelPrefix, "generateStateVariables", i)
    push!(outerDefs, quote
      function $(fName)()
        $(Tuple([stateVector...]))
      end
    end)
    push!(constructorNames, fName)
    i += 1
  end
  i = 1
  for algVector in algVectors
    local fName = Symbol(modelPrefix, "generateAlgebraicVariables", i)
    push!(outerDefs, quote
      function $(fName)()
        $(Tuple([algVector...]))
      end
    end)
    push!(constructorNames, fName)
    i += 1
  end
  local outerExpr = quote
    $(outerDefs...)
  end
  local innerExpr = quote
    variableConstructors = Function[$(constructorNames...)]
  end
  return (outerExpr, innerExpr)
end

"""
  Decomposes equations into chunked constructor functions.
  Returns (outer_defs, inner_refs) where:
  - outer_defs: function definitions at module level
  - inner_refs: parameter assignments + equationConstructorCalls array (inside model function)
  The `modelPrefix` avoids name collisions when multiple models are translated in one session.
"""
function decomposeEquations(equations, parameterAssignments; modelPrefix::String = "", chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local outerDefs = Expr[]
  local functionNames = Symbol[]
  local i = 0
  for equationVector in equationVectors
    local eqv = collect(equationVector)
    local fName = Symbol(modelPrefix, "generateEquations", i)
    push!(outerDefs, quote
      function $(fName)()
        [$(eqv...)]
      end
    end)
    push!(functionNames, fName)
    i += 1
  end
  local outerExpr = quote
    $(outerDefs...)
  end
  local innerExpr = quote
    $(parameterAssignments...)
    local equationConstructors::Vector{Function}
    local equationConstructorCalls::Vector
    equationConstructorCalls = [$(functionNames...)]
  end
  return (outerExpr, innerExpr)
end

"""
  Decomposes start equations into chunked constructor functions.
  Returns (outer_defs, inner_refs) where:
  - outer_defs: function definitions at module level
  - inner_refs: startEquationConstructors array assignment (inside model function)
  The `modelPrefix` avoids name collisions, `functionSuffix` differentiates initial vs final start eqs.
"""
function decomposeStartEquations(equations; functionSuffix = "", modelPrefix::String = "", chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local outerDefs = Expr[]
  local constructorNames = Symbol[]
  local i = 0
  for equationVector in equationVectors
    local fName = Symbol(modelPrefix, "generateStartEquations", functionSuffix, i)
    push!(outerDefs, quote
      function $(fName)()
        [$(equationVector...)]
      end
    end)
    push!(constructorNames, fName)
    i += 1
  end
  local outerExpr = quote
    $(outerDefs...)
  end
  local innerExpr = quote
    startEquationConstructors = Function[$(constructorNames...)]
  end
  return (outerExpr, innerExpr)
end

"""
  Inline variant of decomposeEquations that keeps function definitions inside
  the model function body. Equation expressions reference parameter symbols that
  are local to the model function (created by @parameters), so they cannot be
  moved to module level.
"""
function decomposeEquationsInline(equations, parameterAssignments; chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local exprs = Expr[]
  local functionNames = Symbol[]
  local constructors = quote
    $(parameterAssignments...)
    local equationConstructors::Vector{Function}
    local equationConstructorCalls::Vector
  end
  push!(exprs, constructors)
  local i = 0
  for equationVector in equationVectors
    local eqv = collect(equationVector)
    local (csEqs, csPreamble) = extractCommonHvcats(eqv)
    local fName = Symbol("generateEquations", i)
    if isempty(csPreamble)
      push!(exprs, quote
        function $(fName)()
          [$(eqv...)]
        end
      end)
    else
      push!(exprs, quote
        function $(fName)()
          $(csPreamble...)
          [$(csEqs...)]
        end
      end)
    end
    push!(functionNames, fName)
    i += 1
  end
  return quote
    $(exprs...)
    equationConstructorCalls = [$(functionNames...)]
  end
end

"""
  Inline variant of decomposeStartEquations that keeps function definitions inside
  the model function body. Start equations reference symbols that may be local to
  the model function (created by @parameters or phase 3 eval), so they cannot be
  moved to module level.
"""
function decomposeStartEquationsInline(equations; functionSuffix = "", chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local exprs = Expr[]
  local constructorNames = Symbol[]
  local i = 0
  for equationVector in equationVectors
    local eqv = collect(equationVector)
    local (csEqs, csPreamble) = extractCommonHvcats(eqv)
    local fName = Symbol("generateStartEquations", functionSuffix, i)
    if isempty(csPreamble)
      push!(exprs, quote
        function $(fName)()
          [$(eqv...)]
        end
      end)
    else
      push!(exprs, quote
        function $(fName)()
          $(csPreamble...)
          [$(csEqs...)]
        end
      end)
    end
    push!(constructorNames, fName)
    i += 1
  end
  return quote
    $(exprs...)
    startEquationConstructors = Function[$(constructorNames...)]
  end
end

"""
  Chunks the @parameters macro call into inner functions to reduce the model
  function body size. Each inner function calls @parameters with a subset of
  parameter names and returns the resulting vector. Results are concatenated.

  After chunking, parameter symbols are eval'd into module scope so that
  pars Dict closures and ARRAY_PARAMETERS code can reference them by name.
"""

"""
Generate code to declare ifCond variables as plain parameters (not time-dependent).
These parameters are modified by SymbolicContinuousCallback affects and are NOT
part of the ODE state vector, so the solver never perturbs them during Jacobian
computation. Using plain parameters (not `p(t)`) avoids MTK creating Shift operators.
Returns a no-op expression if there are no ifCond parameters.
"""
function generateDiscreteIfCondDeclaration(ifCondParamDecls::Vector{Expr}, ifCondNames::Vector{Symbol})
  if isempty(ifCondParamDecls)
    return :()
  end
  local nameQuotes = [QuoteNode(s) for s in ifCondNames]
  quote
    local _ifCondParams = ModelingToolkit.@parameters begin
      $(ifCondParamDecls...)
    end
    local _ifCondBindBlock = Expr(:block)
    for (name, p) in zip([$(nameQuotes...)], _ifCondParams)
      push!(_ifCondBindBlock.args, :($name = $p))
    end
    eval(_ifCondBindBlock)
    parameters = vcat(parameters, _ifCondParams)
  end
end

"""
Generate code to add ifCond discrete parameter values to the pars Dict.
Returns a no-op expression if there are no ifCond parameters.
"""
function generateIfCondParamAssignments(ifCondParamPairs::Vector{Expr})
  if isempty(ifCondParamPairs)
    return :()
  end
  quote
    for (k, v) in [$(ifCondParamPairs...)]
      pars[k] = v
    end
  end
end

function decomposeParametersDeclaration(parVariablesSym; chunkSize = CHUNK_SIZE[])
  if length(parVariablesSym) <= chunkSize
    return quote
      parameters = ModelingToolkit.@parameters begin
        ($(parVariablesSym...))
      end
    end
  end
  local chunks = collect(Iterators.partition(parVariablesSym, chunkSize))
  local exprs = Expr[]
  local constructorNames = Symbol[]
  for (i, chunk) in enumerate(chunks)
    local fName = Symbol("_createParams_", i - 1)
    local chunkSyms = collect(chunk)
    push!(exprs, quote
      function $(fName)()
        ModelingToolkit.@parameters begin
          ($(chunkSyms...))
        end
      end
    end)
    push!(constructorNames, fName)
  end
  local paramNameQuotes = [QuoteNode(s) for s in parVariablesSym]
  return quote
    $(exprs...)
    local _allParamChunks = []
    for _fn in [$(constructorNames...)]
      push!(_allParamChunks, Base.invokelatest(_fn))
    end
    parameters = vcat(_allParamChunks...)
    local _paramNames = [$(paramNameQuotes...)]
    local _paramBindBlock = Expr(:block)
    for (name, p) in zip(_paramNames, parameters)
      push!(_paramBindBlock.args, :($name = $p))
    end
    eval(_paramBindBlock)
  end
end

"""
  Chunks parameter equations (sym => value pairs) into small inner functions
  to reduce JIT overhead from compiling one massive Dict literal.
  Each chunk function returns a Dict of parameter pairs. Results are
  merged into the final pars Dict via invokelatest.

  Parameter equations reference @parameters symbols which are local to the
  model function, so chunk functions are defined inline as closures.
"""
function decomposeParameterEquationsInline(parameterEquations; chunkSize = CHUNK_SIZE[])
  if length(parameterEquations) <= chunkSize
    return :(pars = Dict($(parameterEquations...)))
  end
  local chunks = collect(Iterators.partition(parameterEquations, chunkSize))
  local exprs = Expr[]
  local constructorNames = Symbol[]
  for (i, chunk) in enumerate(chunks)
    local fName = Symbol("_generatePars_", i - 1)
    local chunkExprs = collect(chunk)
    push!(exprs, quote
      function $(fName)()
        Dict($(chunkExprs...))
      end
    end)
    push!(constructorNames, fName)
  end
  return quote
    $(exprs...)
    pars = Dict{Any,Any}()
    for _parFn in [$(constructorNames...)]
      merge!(pars, Base.invokelatest(_parFn))
    end
  end
end

"""
  Generate @register_array_symbolic expression for a function with array parameters.
"""
function generateArrayRegisterExpr(f::SimulationCode.ModelicaFunction, funcArgGen::Function)::Expr
  local sb = Symbol(f.name)

  #= Build typed argument list for array registration =#
  local argExprs = Expr[]
  for v in f.inputs
    local varName = Symbol(string(v.componentRef))
    if isArrayType(v)
      push!(argExprs, :($varName::AbstractArray))
    else
      push!(argExprs, :($varName::Real))
    end
  end

  #= Build the call signature =#
  local callExpr = if length(argExprs) == 0
    Expr(:call, sb)
  elseif length(argExprs) == 1
    Expr(:call, sb, argExprs[1])
  else
    Expr(:call, sb, argExprs...)
  end

  #= Determine output size and eltype =#
  #= For now, assume first output determines the result characteristics =#
  local sizeExpr = :()
  local eltypeExpr = :(Symbolics.Num)  #= Use Symbolics.Num for proper type compatibility =#

  if !isempty(f.outputs)
    local firstOutput = first(f.outputs)
    if isArrayType(firstOutput)
      sizeExpr = extractArrayDimsFromVar(firstOutput)
    end
  end

  #= Generate the @register_array_symbolic call =#
  quote
    Symbolics.@register_array_symbolic $callExpr begin
      size = $sizeExpr
      eltype = $eltypeExpr
    end
  end
end

"""
  Generates quoted Symbolics registration calls for externally defined functions.
  Scalar functions use @register_symbolic.
  Functions with array parameters that return arrays use @register_array_symbolic
  so MTK knows the output shape and can handle getindex on the result.
"""
function generateRegisterCallsForCallExprs(simCode;
                                            funcArgGen::Function = AlgorithmicCodeGeneration.generateSignatureForRegistration)
  local rFs = Expr[]
  local calledFunctions = collectCalledFunctionNames!(Set{String}(), simCode)
  for f in simCode.functions
    if !(f.name in calledFunctions)
      continue
    end
    if hasArrayParameters(f)
      #= Functions with array parameters are not registered. =#
      #= They execute eagerly with symbolic array arguments. =#
      continue
    else
      #= Use @register_symbolic for scalar functions =#
      local sb = Symbol(f.name)
      local args = funcArgGen(convert(Vector{DAE.VAR}, f.inputs))
      local nArgs = length(args)
      local cExpr = if nArgs == 1
        Expr(:call, sb, first(args))
      elseif nArgs == 0
        Expr(:call, sb)
      else
        Expr(:call, sb, tuple(args...)...)
      end
      #= Delay evaluation of the register expression until we know the call. =#
      sbRegister = :((Symbolics.@register_symbolic($(cExpr))))
      push!(rFs, sbRegister)
    end
  end
  return rFs
end

"""
  Optionally generate an import statement to OMRuntimeExternalC
"""
function generateExternalRuntimeImport()::Expr
  :(import OMRuntimeExternalC)
end

function _emitWhenTupleElementAssignMTK!(res::Vector{Expr}, lhs,
                                          rhsAccess, simCode::SimulationCode.SIM_CODE)
  @match lhs begin
    DAE.CREF(DAE.WILD(), _) => nothing
    DAE.CREF(__) => begin
      local name = SimulationCode.string(lhs)
      local entry = get(simCode.stringToSimVarHT, name, nothing)
      if entry === nothing
        push!(res, :($(Symbol(name)) = $rhsAccess))
        return res
      end
      local (_, var) = entry
      push!(res, quote
              idx = lookuptableStates[Symbol($(string(var.name)))]
              integrator.u[idx] = $rhsAccess
            end)
    end
    DAE.ARRAY(_, _, elements) => begin
      local i = 0
      for elem in elements
        i += 1
        _emitWhenTupleElementAssignMTK!(res, elem, :($rhsAccess[$i]), simCode)
      end
    end
    SimulationCode.EXP_CREF(cref, _) => begin
      local name = string(cref)
      local entry = get(simCode.stringToSimVarHT, name, nothing)
      if entry === nothing
        push!(res, :($(Symbol(name)) = $rhsAccess))
        return res
      end
      local (_, var) = entry
      push!(res, quote
              idx = lookuptableStates[Symbol($(string(var.name)))]
              integrator.u[idx] = $rhsAccess
            end)
    end
    SimulationCode.ARRAY_EXP(_, _, elements) => begin
      local i = 0
      for elem in elements
        i += 1
        _emitWhenTupleElementAssignMTK!(res, elem, :($rhsAccess[$i]), simCode)
      end
    end
    _ => throw(ErrorException("createWhenStatementsMTK: unsupported tuple-LHS element $lhs"))
  end
  return res
end

function createWhenStatementsMTK(whenStatements, simCode::SimulationCode.SIM_CODE; varPrefix = "", varSuffix = "")::Vector{Expr}
  local res::Array{Expr} = []
  local nWhenStatements = 0
  for _ in whenStatements
    nWhenStatements += 1
  end
  @debug "[MTK GEN: when] createWhenStatementsMTK" statements=nWhenStatements
  for wStmt in whenStatements
    if wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
      if wStmt.left isa DAE.TUPLE || wStmt.left isa SimulationCode.TUPLE
        local tupSym = gensym(:tupResult)
        local rhsExpr = expToJuliaExpMTK(wStmt.right, simCode;
                                         varPrefix = varPrefix, varSuffix = varSuffix)
        push!(res, :(local $tupSym = $rhsExpr))
        local i = 0
        for elem in wStmt.left.PR
          i += 1
          _emitWhenTupleElementAssignMTK!(res, elem, :($tupSym[$i]), simCode)
        end
      else
        # SimulationCode.ASSIGN.left is ::Exp post-migration; HT keys are DAE-stringified.
        local leftStr = SimulationCode.string(SimulationCode.toDAEExp(wStmt.left))
        (index, var) = simCode.stringToSimVarHT[leftStr]
        local lhsSym = Symbol(string(var.name))
        local rhsE = expToJuliaExpMTK(wStmt.right, simCode; varPrefix = varPrefix, varSuffix = varSuffix)
        if leftStr in _heldDiscreteParamSet(simCode)
          #= (b) held discrete is a parameter (not a state slot): write it through the
             SymbolicIndexingInterface by literal-Symbol name (`integrator.ps[:name]`). =#
          push!(res, quote
                  integrator.ps[$(QuoteNode(lhsSym))] = $(rhsE)
                end)
        else
          push!(res, quote
                  idx = lookuptableStates[Symbol($(string(var.name)))]
                  integrator.u[idx] = $(rhsE)
                  $(lhsSym) = integrator.u[idx]
                end)
        end
      end
    elseif wStmt isa BDAE.REINIT || wStmt isa SimulationCode.REINIT
      (index, var) = simCode.stringToSimVarHT[SimulationCode.string(wStmt.stateVar)]
      push!(res, quote
              idx = lookuptableStates[Symbol($(string(var.name)))]
              integrator.u[idx] = $(expToJuliaExpMTK(wStmt.value,
                                                     simCode; varPrefix = varPrefix, varSuffix = varSuffix))
            end)
    elseif wStmt isa BDAE.TERMINATE || wStmt isa SimulationCode.TERMINATE
      local msgExpr = expToJuliaExpMTK(wStmt.message, simCode;
                                        varPrefix = varPrefix, varSuffix = varSuffix)
      push!(res, quote
              @info "Modelica terminate() reached" message=$(msgExpr)
              OMBackend.DifferentialEquations.terminate!(integrator)
            end)
    elseif wStmt isa BDAE.NORETCALL || wStmt isa SimulationCode.NORETCALL
      local callExpr = expToJuliaExpMTK(wStmt.exp, simCode;
                                         varPrefix = varPrefix, varSuffix = varSuffix)
      push!(res, quote
              $(callExpr)
            end)
    elseif wStmt isa BDAE.ASSERT || wStmt isa SimulationCode.ASSERT
      local condExpr = expToJuliaExpMTK(wStmt.condition, simCode;
                                         varPrefix = varPrefix, varSuffix = varSuffix)
      local msgExpr = expToJuliaExpMTK(wStmt.message, simCode;
                                        varPrefix = varPrefix, varSuffix = varSuffix)
      push!(res, quote
              if !($(condExpr))
                @warn "Modelica assert()" message=$(msgExpr)
              end
            end)
    else
      throw(ErrorException("createWhenStatementsMTK: unsupported when-statement variant $(wStmt)"))
    end
  end
  return res
end

#= True when a when-equation's condition is the Modelica `terminal()` operator. =#
function _isTerminalWhen(@nospecialize(eq))::Bool
  (eq isa BDAE.WHEN_EQUATION || eq isa SimulationCode.WHEN_EQUATION) || return false
  return @match SimulationCode.toDAEExp(eq.whenEquation.condition) begin
    DAE.CALL(Absyn.IDENT("terminal"), _, _) => true
    _ => false
  end
end

#= Post-solve runner for `when terminal()` bodies, or `nothing` when the model
   has none (so models without a terminal event are unchanged). The bodies
   reuse `createWhenStatementsMTK` by mocking `integrator` from the final
   solution point — writes land in `_sol.u[end]` using the same state-index
   convention the discrete-callback affects rely on. Runs only on success. =#
function createTerminalBodyRunner(simCode::SimulationCode.SIM_CODE)
  local terminalWhens = filter(_isTerminalWhen, simCode.whenEquations)
  isempty(terminalWhens) && return nothing
  local modelFns = Set(replace(f.name, "." => "_") for f in simCode.functions)
  local calledNames = Set{String}()
  local perWhen = Expr[]
  for eq in terminalWhens
    local body = eq.whenEquation.whenStmtLst
    for s in body
      collectCalledFunctionNames!(calledNames, s)
    end
    for c in vcat(map(s -> getRHSVariables(s), body)...)
      local entry = get(simCode.stringToSimVarHT, string(c), nothing)
      #= String parameters are emitted as module-level constants; referencing them
         directly avoids shadowing that binding with a nonexistent state lookup. =#
      entry !== nothing && entry[2].varKind isa SimulationCode.STRING && continue
      push!(perWhen, Expr(:(=), Symbol(string(c)), getIdxForLookupMTK(c, simCode)))
    end
    append!(perWhen, createWhenStatementsMTK(body, simCode))
  end
  #= Bind external functions the body calls to their concrete OMBackend.CodeGeneration
     RTG wrapper: the bare model-module name is the @register_symbolic binding (symbolic
     only), which has no method for concrete runtime arguments. =#
  local fnRebinds = Expr[]
  for n in calledNames
    local nn = replace(n, "." => "_")
    nn in modelFns && push!(fnRebinds, :(local $(Symbol(nn)) = OMBackend.CodeGeneration.$(Symbol(nn))))
  end
  return quote
    if _sol.retcode == ModelingToolkit.SciMLBase.ReturnCode.Success
      #= Best-effort: a terminal body runs after a completed, valid solution, so a
         body we cannot evaluate (e.g. an unsupported external call) warns rather
         than discarding the result. =#
      try
        let integrator = (u = _sol.u[end], t = _sol.t[end], f = _sol.prob.f, dt = 0.0, ps = _sol.prob.ps),
            x = _sol.u[end],
            t = _sol.t[end],
            p = _sol.prob.p,
            lookuptableStates = Dict(sym => i for (i, sym) in enumerate(OMBackend.CodeGeneration.getStatesAsSymbols(_sol.prob.f))),
            lookuptableParams = Dict(sym => i for (i, sym) in enumerate(OMBackend.CodeGeneration.getParametersAsSymbols(_sol.prob.f)))
          local idx = 0
          $(fnRebinds...)
          $(perWhen...)
        end
      catch _terminalErr
        @warn "when terminal() body could not be evaluated; returning the completed solution unchanged" exception = _terminalErr
      end
    end
  end
end

#= Lower a single BDAE.WhenOperator from an INITIAL_ALGORITHM body to a Julia
   expression suitable for module-top eval inside `__runInitialAlgorithm!`.
   Parameter CREFs are folded to their literal bindings via _substituteBoundParameters
   before lowering with the algorithmic (non-MTK) translator, so no Symbolics
   bindings are needed at init time.

   ASSIGN / REINIT write their RHS into `LATEST_PROBLEM` via the
   SymbolicIndexingInterface `prob[:name] = value` setter. The function runs
   after the ODEProblem is constructed (see `simulate(...)` in the generated
   module), so LATEST_PROBLEM is in scope. Without this, the LHS state stayed
   at its default (0) — e.g. `T_start := startTime + count*period` in the
   trapezoid signal source was silently dropped, breaking every model that
   relies on `initial algorithm` to seed states. =#
function _initialWhenOpToJulia(wStmt, simCode::SimulationCode.SIM_CODE,
                               renamedNames::Set{String} = Set{String}())
  local sub = e -> _substituteBoundParameters(e, simCode)
  local lowerAlg = e -> _renameAlgIdentifiers(
    _resolveModelicaCallTargets(AlgorithmicCodeGeneration.expToJuliaExpAlg(sub(e))),
    renamedNames,
    "")
  local crefName = cr -> SimulationCode.DAE_identifierToString(cr)
  if wStmt isa BDAE.NORETCALL || wStmt isa SimulationCode.NORETCALL
    return :( $(lowerAlg(wStmt.exp)); nothing )
  elseif wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
    # SimulationCode.ASSIGN.left is ::Exp post-migration; convert to DAE for the @match.
    local leftDAE = wStmt isa SimulationCode.ASSIGN ? SimulationCode.toDAEExp(wStmt.left) : wStmt.left
    local name = @match leftDAE begin
      DAE.CREF(cr, _) => crefName(cr)
      _ => nothing
    end
    if name === nothing
      return :( $(lowerAlg(wStmt.right)); nothing )
    end
    local sym = Symbol(name)
    local isParam = haskey(simCode.stringToSimVarHT, name) &&
                    let (_, sv) = simCode.stringToSimVarHT[name]
                      sv.varKind isa SimulationCode.PARAMETER ||
                      sv.varKind isa SimulationCode.ARRAY_PARAMETER
                    end
    if isParam
      return :( $(sym) = $(lowerAlg(wStmt.right)); LATEST_PROBLEM.ps[$(QuoteNode(sym))] = $(sym); nothing )
    else
      return :( $(sym) = $(lowerAlg(wStmt.right));
         try
           ModelingToolkit.SciMLBase.setu(LATEST_PROBLEM, $(QuoteNode(sym)))(LATEST_PROBLEM, $(sym))
         catch
           nothing
         end;
         try
           _hard[getproperty(LATEST_REDUCED_SYSTEM, $(QuoteNode(sym)))] = $(sym)
         catch
           nothing
         end;
         nothing )
    end
  elseif wStmt isa BDAE.REINIT || wStmt isa SimulationCode.REINIT
    local name = crefName(wStmt.stateVar)
    local sym = Symbol(name)
    return :( $(sym) = $(lowerAlg(wStmt.value)); LATEST_PROBLEM[$(QuoteNode(sym))] = $(sym); nothing )
  elseif wStmt isa BDAE.ASSERT || wStmt isa SimulationCode.ASSERT
    local cond = lowerAlg(wStmt.condition)
    local msg = lowerAlg(wStmt.message)
    return :(if !($cond); @warn "Modelica assert() during init" message=$(msg); end)
  elseif wStmt isa BDAE.TERMINATE || wStmt isa SimulationCode.TERMINATE
    local msg = lowerAlg(wStmt.message)
    return :(@info "Modelica terminate() during init" message=$(msg))
  end
  throw(ErrorException("_initialWhenOpToJulia: unsupported variant $(typeof(wStmt))"))
end

#= Translate a single `BDAE.WhenOperator` from an init-algorithm body into a
   Julia statement suitable for the procedural body of
   `__runInitialAlgorithmEarly!`. ASSIGN emits `_alg_<lhs> = <rhs>` (with
   `local` on the first occurrence of that LHS); RHS identifiers are renamed
   via `_renameAlgIdentifiers` so they bind to the let-block locals rather
   than to module-level Symbolics bindings of the same name. =#
function _initialWhenOpToJuliaEarly(wStmt, simCode::SimulationCode.SIM_CODE,
                                    renamedNames::Set{String}, seenLHS::Set{String})
  local sub = e -> _substituteBoundParameters(e, simCode)
  local lowerAlg = e -> _renameAlgIdentifiers(
    _resolveModelicaCallTargets(AlgorithmicCodeGeneration.expToJuliaExpAlg(sub(e))),
    renamedNames)
  local crefName = cr -> SimulationCode.DAE_identifierToString(cr)
  if wStmt isa BDAE.NORETCALL || wStmt isa SimulationCode.NORETCALL
    return :( $(lowerAlg(wStmt.exp)); nothing )
  elseif wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
    # SimulationCode.ASSIGN.left is ::Exp post-migration; convert to DAE for the @match.
    local leftDAE = wStmt isa SimulationCode.ASSIGN ? SimulationCode.toDAEExp(wStmt.left) : wStmt.left
    local name = @match leftDAE begin
      DAE.CREF(cr, _) => crefName(cr)
      _ => nothing
    end
    if name === nothing
      return :( $(lowerAlg(wStmt.right)); nothing )
    end
    local algSym = Symbol("_alg_" * name)
    if name in seenLHS
      return :( $(algSym) = $(lowerAlg(wStmt.right)); nothing )
    end
    push!(seenLHS, name)
    return :( local $(algSym) = $(lowerAlg(wStmt.right)); nothing )
  elseif wStmt isa BDAE.ASSERT || wStmt isa SimulationCode.ASSERT
    local cond = lowerAlg(wStmt.condition)
    local msg = lowerAlg(wStmt.message)
    return :(if !($cond); @warn "Modelica assert() during init (early)" message=$(msg); end)
  elseif wStmt isa BDAE.TERMINATE || wStmt isa SimulationCode.TERMINATE
    local msg = lowerAlg(wStmt.message)
    return :(@info "Modelica terminate() during init (early)" message=$(msg))
  end
  return :( nothing )
end

"""
    generateInitialAlgorithmEarlyFunction(simCode) -> Expr

Emit `function __runInitialAlgorithmEarly!() -> Dict{Symbol, Float64}` that
executes the `initial algorithm` bodies procedurally at module-load time
(Modelica §11.4: statements run sequentially, the LHS final value becomes the
variable's initial value).

When `simCode.initialAlgorithms[i].daeStatements` is non-empty for any body,
the procedural body is lowered via `AlgorithmicCodeGeneration.generateStatements`
— the same path used for regular Modelica algorithm sections and function
bodies, with full STMT_IF / STMT_FOR / STMT_WHILE / STMT_ASSERT / STMT_REINIT
support. The resulting Julia AST is then rewritten by `_renameAlgIdentifiers`
to prefix every cref name with `_alg_`, so the locals do not collide with the
Symbolics `Num` bindings of the same name living in the surrounding model
scope. When `daeStatements` is empty (e.g. older callers that only provide a
`Vector{BDAE.WhenOperator}`), the legacy flat-WhenOperator translator
`_initialWhenOpToJuliaEarly` is used as a fallback.

The body is wrapped in `let time = 0.0 ... end`. Non-LHS crefs read on the
RHS get a pre-seeded `_alg_<name>` from the SimVar's `start` attribute or
`0.0`. After the body, each LHS final value is captured into the returned
`Dict{Symbol, Float64}` via a per-entry try/catch (so a still-undefined
`_alg_<name>` from a body that errored partway just skips that entry).

The outer try/catch returns partial results on any error; the cycle-19
runtime `remake` path remains as a fallback for state-cref-RHS reads whose
post-init value differs from the `start` attribute.
"""
function generateInitialAlgorithmEarlyFunction(simCode::SimulationCode.SIM_CODE)::Expr
  local lhsNames = Set{String}()
  local rhsNames = Set{String}()
  local useDAEPath = any(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms)
  if useDAEPath
    for ia in simCode.initialAlgorithms, s in ia.daeStatements
      _collectInitAlgLhsRhsCrefsDAE!(lhsNames, rhsNames, s)
    end
  else
    for ia in simCode.initialAlgorithms, op in ia.statements
      _collectInitAlgLhsRhsCrefs!(lhsNames, rhsNames, op)
    end
  end
  if isempty(lhsNames) && isempty(rhsNames)
    return quote
      function __runInitialAlgorithmEarly!()
        return Dict{Symbol, Float64}()
      end
    end
  end
  local ht = simCode.stringToSimVarHT
  local renamedNames = union(lhsNames, rhsNames)
  push!(renamedNames, "time")
  local prefetches = Expr[]
  for name in setdiff(rhsNames, lhsNames)
    name == "time" && continue
    haskey(ht, name) || begin
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = 0.0))
      continue
    end
    local sv = ht[name][2]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      local paramLit = @match sv.varKind begin
        SimulationCode.PARAMETER(SOME(SimulationCode.RCONST(r))) => Float64(r)
        SimulationCode.PARAMETER(SOME(SimulationCode.ICONST(i))) => Float64(i)
        SimulationCode.PARAMETER(SOME(SimulationCode.BCONST(b))) => (b ? 1.0 : 0.0)
        _ => nothing
      end
      paramLit === nothing && continue
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(paramLit)))
      continue
    end
    local lit = _readStartAttributeAsLiteral(sv)
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(lit)))
  end
  for name in lhsNames
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = 0.0))
  end
  local stmts = Expr[]
  if useDAEPath
    for ia in simCode.initialAlgorithms
      isempty(ia.daeStatements) && continue
      local body = AlgorithmicCodeGeneration.generateStatements(ia.daeStatements)
      for s in body
        push!(stmts, _renameAlgIdentifiers(s, renamedNames))
      end
    end
  else
    local seenLHS = copy(lhsNames)
    for ia in simCode.initialAlgorithms, op in ia.statements
      push!(stmts, _initialWhenOpToJuliaEarly(op, simCode, renamedNames, seenLHS))
    end
  end
  local captures = Expr[]
  for name in lhsNames
    haskey(ht, name) || continue
    local sv = ht[name][2]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      continue
    end
    local algSym = Symbol("_alg_" * name)
    local qn = QuoteNode(Symbol(name))
    push!(captures, :(try; _results[$(qn)] = Float64($(algSym)); catch; nothing; end))
  end
  return quote
    function __runInitialAlgorithmEarly!()
      local _results = Dict{Symbol, Float64}()
      try
        let time = 0.0
          $(prefetches...)
          $(stmts...)
          $(captures...)
        end
      catch _err
        @debug "[MTK GEN: init-alg early] body raised; partial results returned" exception=_err
      end
      return _results
    end
  end
end

"""
    generateInitialAlgorithmFunction(simCode) -> Expr

Emit a `function __runInitialAlgorithm!() ... end` whose body executes once
during initialization, lowered from `simCode.initialAlgorithms`. Parameter
literals are already baked into the body by `inlineParamsInInitialAlgorithms`
at SimCode construction time, so no module-scope parameter bindings are needed
here. Returns a no-op stub when the model has no `when initial()` clauses.
"""
function generateInitialAlgorithmFunction(simCode::SimulationCode.SIM_CODE)::Expr
  #= When the DAE.Statement-based early-eval path is available, it emits
     control-flow-correct `initialization_eqs` for every LHS. The runtime
     `remake` here is built from the lossy WhenOperator flattening and would
     overwrite the init-eq result with the flat-first-branch value at simulate
     time. Emit an empty stub instead — the early path covers it. =#
  local useDAEPath = any(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms)
  if useDAEPath
    return quote
      function __runInitialAlgorithm!()
        return Dict{Any, Any}()
      end
    end
  end
  local lhsNames = Set{String}()
  local rhsNames = Set{String}()
  for ia in simCode.initialAlgorithms
    for op in ia.statements
      _collectInitAlgLhsRhsCrefs!(lhsNames, rhsNames, op)
    end
  end
  local renamedNames = union(lhsNames, rhsNames)
  push!(renamedNames, "time")
  local stmts = Expr[]
  for ia in simCode.initialAlgorithms
    for op in ia.statements
      push!(stmts, _initialWhenOpToJulia(op, simCode, renamedNames))
    end
  end
  if isempty(stmts)
    return quote
      function __runInitialAlgorithm!()
        return Dict{Any, Any}()
      end
    end
  end
  #= Pre-fetch every non-parameter RHS-referenced cref. Names that ALSO
     appear as LHS still need a fetch because Julia compiles `x = if c then
     v else x end` with `x` as a function-local: the else-branch reads `x`
     before the assignment completes and throws UndefVarError. Self-referential
     IFEXP shapes come from the algorithm lifter at BDAECreate.jl:1263 when a
     non-when algorithm contains an if/elseif chain whose else-branches
     preserve a discrete LHS's previous value. =#
  local fetches = Expr[]
  local ht = simCode.stringToSimVarHT
  for name in rhsNames
    name == "time" && continue
    local sv = nothing
    if haskey(ht, name)
      sv = ht[name][2]
      if sv.varKind isa SimulationCode.PARAMETER || sv.varKind isa SimulationCode.ARRAY_PARAMETER
        continue
      end
      if sv.varKind isa SimulationCode.DATA_STRUCTURE || sv.varKind isa SimulationCode.STRING
        local sym = Symbol(name)
        local boundSym = Symbol(sv.name)
        push!(fetches, Expr(:local, Expr(:(=), sym, :(getfield(@__MODULE__, $(QuoteNode(boundSym)))))))
        continue
      end
      #= DISCRETE vars (Logic/enum/Boolean) are used as array indices. MTK
         initialisation may leave them at 0 which BoundsErrors on 1-based
         index vectors (e.g. INV3S's UX01Conv[iNV3S_enable]). Clamp to 1 as
         a band-aid until proper discrete-IC lowering lands. =#
      if sv.varKind isa SimulationCode.DISCRETE
        local sym = Symbol(name)
        push!(fetches, Expr(:local,
          Expr(:(=), sym,
            :(try
                let _g = ModelingToolkit.SciMLBase.getu(LATEST_PROBLEM, $(QuoteNode(sym)))
                  local _raw = _g(LATEST_PROBLEM)
                  local _v = if _raw isa Integer
                    Int(_raw)
                  elseif _raw isa Real
                    Int(round(Float64(_raw)))
                  else
                    1
                  end
                  _v < 1 ? 1 : _v
                end
              catch
                1
              end))))
        continue
      end
    end
    #= Non-discrete SimVars (Real states, alg vars) and local algorithm
       temporaries not in the HT: fetch as Float64, no index clamp. =#
    local sym = Symbol(name)
    push!(fetches, Expr(:local,
      Expr(:(=), sym,
        :(try
            Float64(ModelingToolkit.SciMLBase.getu(LATEST_PROBLEM, $(QuoteNode(sym)))(LATEST_PROBLEM))
          catch
            0.0
          end))))
  end
  #= Shadow `Base.time` (a UNIX-time function) with the local Modelica `time`
     value, which is 0 at simulation init. Without this, init-algorithm bodies
     that reference `time` (e.g. trapezoid sources' `count := integer((time -
     startTime) / period)`) generate `time - <Float64>` and hit MethodError
     because `Base.time` is a function, not a number. =#
  return quote
    function __runInitialAlgorithm!()
      #= `_hard` collects (symbolic_var => value) pairs for each ASSIGN to a
         non-parameter variable. simulate() passes it to `remake(prob; u0=…,
         initializealg=NoInit())` so MTK treats the init-algorithm-computed
         values as hard initial conditions (Modelica §11.2), not guesses. =#
      local _hard = Dict{Any, Any}()
      let time = 0.0
        $(fetches...)
        $(stmts...)
      end
      return _hard
    end
  end
end
