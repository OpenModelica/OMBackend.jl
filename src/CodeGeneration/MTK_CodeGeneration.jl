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

For `ODE_MODE_MTK_MODEL_GENERATION`: `eval` each generated Modelica
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
  #= Under precompile / image generation, eval'ing the model's generated Modelica functions
     into the already-closed `CodeGeneration` module is rejected by Julia ("breaks incremental
     compilation"). Skip the eval + registration here: the codegen that PRODUCED `functions`
     has already run (so its method instances are warmed for the bake), and a real runtime
     translate registers them normally — the guard is precompile-only. Mirrors the
     jl_generating_output guard in generateIMTKCode (iMTKGen.jl). =#
  if ccall(:jl_generating_output, Cint, ()) != 0
    return nothing
  end
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
  parameters            :: Vector{String}
  arrayParameters       :: Vector{String}
  stateDerivatives      :: Vector{String}
  dataStructureVariables:: Vector{String}
  statePriorityPairs    :: Vector{Tuple{Symbol, Int}}
end

"""
    classifyVariables(simCode) -> ClassifiedVariables

For `ODE_MODE_MTK_MODEL_GENERATION`: walk `simCode.stringToSimVarHT`
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
  local parameters             = String[]
  local arrayParameters        = String[]
  local stateDerivatives       = String[]
  local dataStructureVariables = String[]
  local statePriorityPairs     = Tuple{Symbol, Int}[]
  local ht = simCode.stringToSimVarHT
  #= Membership set built once: the per-variable `idx in matchOrder` below was an
     O(V) scan of the matchOrder Vector, making classification O(V^2). matchOrder
     is not mutated in this loop. =#
  local matchOrderSet = OrderedSet{Int}(simCode.matchOrder)
  for (varName, (idx, var)) in ht
    local varType = var.varKind
    @match varType begin
      SimulationCode.INPUT(__) => unsupported("INPUT variable", varName)
      SimulationCode.STATE(__) => push!(stateVariables, varName)
      SimulationCode.PARAMETER(__) => push!(parameters, varName)
      #= String parameters are non-numeric; excluded from MTK parameter system. =#
      SimulationCode.STRING(__) => nothing
      SimulationCode.ARRAY_PARAMETER(__) => push!(arrayParameters, varName)
      SimulationCode.ARRAY(__) => push!(stateVariables, varName)
      SimulationCode.DISCRETE(__) => push!(discreteVariables, varName)
      SimulationCode.ALG_VARIABLE(__) => begin
        if idx in matchOrderSet
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
        varType isa SimulationCode.ARRAY
      if supportsStatePriority && !startswith(string(varName), "der(")
        push!(statePriorityPairs, (Symbol(varName), priority))
      end
    end
  end
  return ClassifiedVariables(stateVariables, algebraicVariables,
                             discreteVariables,
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
                              stateVariables, algebraicVariables)
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
                                   algebraicVariables::Vector{String})::Vector{Symbol}
  #= Sort: `irreducibleVariables` is an unordered collection, and this list feeds
     structural_simplify's tearing — a non-deterministic order yields a
     non-deterministic (occasionally unsolvable) reduced system. =#
  local syms = Symbol[Symbol(vn) for vn in sort!(collect(simCode.irreducibleVariables))]
  for ceq in conditionalEquations
    if ceq isa Expr && ceq.head == :call && length(ceq.args) >= 2
      local lhs = ceq.args[2]
      lhs isa Symbol && push!(syms, lhs)
    end
  end
  for vn in fixedStartVarNames(vcat(stateVariables, algebraicVariables), simCode)
    local sym = Symbol(vn)
    sym in syms || push!(syms, sym)
  end
  return syms
end

"""
    whenConditionDiscreteSyms(simCode) -> OrderedSet{Symbol}

Names of DISCRETE variables referenced in any when-equation condition. The
generated DiscreteCallback condition reads each by its own name from the state
vector, so these must not be aliased away by relay elimination.
"""
function whenConditionDiscreteSyms(simCode)::OrderedSet{Symbol}
  local out = OrderedSet{Symbol}()
  local ht = simCode.stringToSimVarHT
  for weq in simCode.whenEquations
    local stmts = weq.whenEquation
    while stmts isa SimulationCode.WHEN_STMTS
      for cref in Util.getAllCrefs(SimulationCode.toDAEExp(stmts.condition))
        local nm = string(cref)
        if haskey(ht, nm) && SimulationCode.isDiscrete(last(ht[nm]))
          push!(out, Symbol(nm))
        end
      end
      stmts = stmts.elsewhenPart
    end
  end
  return out
end

"""
    substituteRelayAliasesInWhens(simCode, relayAliases) -> simCode

Re-point when-equation reads of relay-eliminated leaf names at the surviving
representative, so the generated callbacks read surviving unknowns.
"""
function substituteRelayAliasesInWhens(simCode, relayAliases::Dict{Symbol, Symbol})
  isempty(relayAliases) && return simCode
  local wanted = OrderedSet{String}(string(k) for k in keys(relayAliases))
  local tyOf = Dict{String, DAE.Type}()
  local collectTy = function (e::DAE.Exp, acc)
    if e isa DAE.CREF
      local nm = string(e.componentRef)
      if nm in wanted && !haskey(tyOf, nm)
        tyOf[nm] = e.ty
      end
    end
    return (e, true, acc)
  end
  for weq in simCode.whenEquations
    local stmts = weq.whenEquation
    while stmts isa SimulationCode.WHEN_STMTS
      Util.traverseExpTopDown(SimulationCode.toDAEExp(stmts.condition), collectTy, nothing)
      for st in stmts.whenStmtLst
        if st isa SimulationCode.ASSIGN
          Util.traverseExpTopDown(SimulationCode.toDAEExp(st.left), collectTy, nothing)
          Util.traverseExpTopDown(SimulationCode.toDAEExp(st.right), collectTy, nothing)
        elseif st isa SimulationCode.REINIT
          Util.traverseExpTopDown(SimulationCode.toDAEExp(st.value), collectTy, nothing)
        end
      end
      stmts = stmts.elsewhenPart
    end
  end
  local aliasMap = Dict{String, Tuple{String, Bool, DAE.ComponentRef, DAE.Type}}()
  for (k, r) in relayAliases
    local kStr = string(k)
    local ty = get(tyOf, kStr, nothing)
    ty === nothing && continue
    local rStr = string(r)
    aliasMap[kStr] = (rStr, false, DAE.CREF_IDENT(rStr, ty, MetaModelica.nil), ty)
  end
  isempty(aliasMap) && return simCode
  local newWhens = [begin
                      local inner = SimulationCode._substituteAliasInWhenStmts(whenEq.whenEquation, aliasMap)
                      @assign whenEq.whenEquation = inner
                      whenEq
                    end
                    for whenEq in simCode.whenEquations]
  @assign simCode.whenEquations = newWhens
  return simCode
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
    allInitialValues = initialValues,
    liftedDiscretes = (@isdefined(LIFTED_DISCRETES) ? LIFTED_DISCRETES : String[]),
    freeParameters = (@isdefined(FREE_PARAMETERS) ? FREE_PARAMETERS : String[]),
    initRelations = (@isdefined(_ifInitLiterals) ? _ifInitLiterals : Any[]),
    initClusters = (@isdefined(_initDiscreteClusters) ? _initDiscreteClusters : Any[]),
    discreteStarts = (@isdefined(LIFTED_DISCRETE_STARTS) ? LIFTED_DISCRETE_STARTS : Dict{String, Float64}()))
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
    defaultSolverFor(solver, problem, reducedSystem, discreteUnknownNames, hasWhens) -> solver

Switch the Rosenbrock default (Rodas5P) to FBDF for the DAE shapes where
Rosenbrock mass-matrix stepping is brittle: purely algebraic systems, and,
in a model without when-equations, algebraic rows of generated discrete
variables (`discreteUnknownNames`, as `name(t)`). Brake reaches a consistent
initial residual, but Rodas5P aborts at once with dt_epsilon/NaN while FBDF
advances the same mass-matrix problem. With when-equations the callbacks keep
the discretes consistent, and FBDF's post-event re-initialization collapses
dt at the first event. A solver the user chose is kept. Runs here rather than
in the generated module, where a model variable (`count`) shadows Base.
"""
function defaultSolverFor(solver, problem, reducedSystem, discreteUnknownNames::Vector{String}, hasWhens::Bool)
  local name = string(nameof(typeof(solver)))
  (startswith(name, "Rodas") || startswith(name, "Rosenbrock")) || return solver
  local n = problem.u0 === nothing ? 0 : length(problem.u0)
  local mm = problem.f.mass_matrix
  #= UniformScaling (pure ODE) reads 1 on the diagonal. =#
  local nDiff = count(i -> mm[i, i] != 0, 1:n)
  if nDiff == 0
    @info "[MTK GEN: solver] zero differential states detected, switching default $(name) -> FBDF for purely-algebraic DAE"
    return OMBackend.daeFallbackSolver()
  end
  (hasWhens || nDiff == n || isempty(discreteUnknownNames)) && return solver
  local unknowns = ModelingToolkit.unknowns(reducedSystem)
  #= MTK renders subscripted unknowns as var"name[i]"(t); strip the quotes. =#
  mtkName(u) = replace(string(u), "var\"" => "", "\"" => "")
  local names = Set(discreteUnknownNames)
  if any(i -> mm[i, i] == 0 && mtkName(unknowns[i]) in names, 1:min(n, length(unknowns)))
    @info "[MTK GEN: solver] algebraic rows involving generated discrete variables detected in mass-matrix system; switching default $(name) -> FBDF"
    return OMBackend.daeFallbackSolver()
  end
  return solver
end

"""
    defaultInitializeKwargs(problem, kwargs, tableClusters) -> NamedTuple

The DAE initialization of a solve when the caller chose none. Table-cluster
models: Newton with finite differences (`tableClusterInitAlg`). Otherwise, for
a problem without an initialization problem (OM.jl solves the initial values
itself): BrownFullBasicInit at the solve's abstol, which OrdinaryDiffEq used by
default before OrdinaryDiffEqCore 4. Since then the default only checks u0,
and fails where OM.jl's initial values leave a residual in an algebraic
equation (the PID models of PIDDecomposition.mo).
"""
function defaultInitializeKwargs(problem, kwargs, tableClusters::Bool)
  haskey(kwargs, :initializealg) && return (;)
  tableClusters && return (; initializealg = tableClusterInitAlg())
  ModelingToolkit.SciMLBase.has_initializeprob(problem.f) && return (;)
  return (; initializealg = DiffEqBase.BrownFullBasicInit(get(kwargs, :abstol, 1.0e-6)))
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
     and re-derives the OTHER pure-time ifConds just after the event time so that
     coincident time events cannot drop one another's affect. =#
  pureTimeEvents       :: Vector{Tuple{Symbol, Any, Any, Float64}}
  #= `target => value` pair Exprs: the t0-selected branch RHS evaluated at the
     start-value map, merged as soft guesses so guarded denominators do not
     start at 0/0 in the DAE init. =#
  relayGuesses         :: Vector{Expr}
  #= (ifCond, crossing function, scale) of each branch whose event has a
     hysteresis: checked after every step by the event iteration
     (withRelationRefresh). =#
  relations            :: Vector{Tuple{Symbol, Any, Any}}
  #= (ifCond, observed names, observed crossing functions, observed -> Bool)
     of the relations whose literal value the initialization settles
     (_settleInitialRelations!): the t0 value, from the solved state. =#
  initLiterals         :: Vector{Expr}
end
IfEquationComponent(events, conditionalEquations, conditionVariables, conditionNameAndIV, pureTimeEvents,
                    relayGuesses) =
  IfEquationComponent(events, conditionalEquations, conditionVariables, conditionNameAndIV, pureTimeEvents,
                      relayGuesses, Tuple{Symbol, Any, Any}[], Expr[])

#= The hysteresis parameter H of the event crossing functions (set per solve
   from its reltol, as OpenModelica's tolZC = 1e-4 * relTol). =#
const ZC_HYSTERESIS = :_zcHysteresis
const ZC_HYSTERESIS_DEFAULT = 1.0e-7   # reltol 1e-3, DifferentialEquations' default
#= After a branch switch the equations change: the algebraic unknowns are
   solved again at the event with the states kept (Modelica's event
   iteration re-solves the system; NoInit left e.g. `y = if ... then 1 else 2`
   at its old value until the end of the next step). =#
const _BRANCH_EVENT_REINIT = :(OMBackend.CodeGeneration.EventReinit())

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
  if !SimulationCode.hasStructuralTransitions(simCode) && !SimulationCode.hasSubModels(simCode)
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
    using DataStructures.OrderedCollections: OrderedSet, OrderedDict
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
    # function $(Symbol("$(MODEL_NAME)Simulate"))(tspan = (0.0, 1.0), solver = OMBackend.defaultSolver())
    #   $(Symbol("$(MODEL_NAME)Model_problem")) = $(Symbol("$(MODEL_NAME)Model"))(tspan)
    #   OMBackend.Runtime.solve($(Symbol("$(MODEL_NAME)Model_problem")), tspan, solver)
    # end

    function simulate(tspan = (0.0, 1.0), solver = OMBackend.defaultSolver(); kwargs...)
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
  local _condDiscretes = whenConditionDiscreteSyms(simCode)
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
  local model = ODE_MODE_MTK_MODEL_GENERATION(simCode, modelName, functions; earlyInitialAlgorithm = true)
  #= Qualify bare Modelica function calls in function bodies so they resolve correctly
     when the program is eval'd in OMBackend scope (backendAPI.jl) rather than CodeGeneration scope.
     Without this, implementation bodies that call other Modelica functions (e.g., normalizeWithAssert
     calling Vectors_length) would fail with UndefVarError. =#
  local funcNames = OrderedSet{Symbol}(Symbol(f.name) for f in simCode.functions)
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
    using DataStructures.OrderedCollections: OrderedSet, OrderedDict
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
    function simulateFromBuild(built, tspan = (0.0, 1.0), solver = OMBackend.defaultSolver(); kwargs...)
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
           system but … is not an unknown". Their values are dropped here;
           the early pass (`__runInitialAlgorithmEarly!`) makes the
           `initial algorithm` assignments initialization equations, which
           cover them; an assignment in a runtime `when initial()` body to
           such a variable has no effect. =#
        local _unkNames = OrderedSet(string(u) for u in ModelingToolkit.unknowns(LATEST_REDUCED_SYSTEM))
        #= Float-convert: Int-valued entries make remake_buffer promote a
           Float64 parameter buffer to Int64, failing on fractional entries. =#
        local _hardFiltered = Dict(Base.first(p) => Float64(Base.last(p))
                                   for p in _hardStarts if string(Base.first(p)) in _unkNames)
        if !isempty(_hardFiltered)
          try
            global LATEST_PROBLEM = ModelingToolkit.SciMLBase.remake(
              LATEST_PROBLEM; u0 = _hardFiltered)
            $(Symbol("$(MODEL_NAME)Model_problem")) = LATEST_PROBLEM
            _didRemake = true
          catch _ialgErr
            OMBackend._fallback(_ialgErr, :initAlgRemake)
            #= The remake is redundant for variables that the
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
      #= Event-trigger discretes (referenced in a when-condition) are latched by
         DiscreteCallbacks and are not part of the brittle Rosenbrock mass-matrix
         coupling the FBDF switch targets; excluding them keeps the default
         Rosenbrock solver, which handles them without the FBDF tstop chatter. =#
      local _solver = OMBackend.CodeGeneration.defaultSolverFor(solver, $(Symbol("$(MODEL_NAME)Model_problem")), LATEST_REDUCED_SYSTEM,
                                                                $(Expr(:ref, :String, [string(varName, "(t)") for (varName, (_, simVar)) in simCode.stringToSimVarHT if simVar.varKind isa SimulationCode.DISCRETE && !(Symbol(varName) in _condDiscretes)]...)),
                                                                $(!isempty(simCode.whenEquations)))
      # Route DAE-native solvers (e.g. Sundials.IDA, DABDF2, DFBDF) through a residual-form DAEProblem rather than the ODEProblem with mass matrix.
      OMBackend.CodeGeneration.setZCHysteresis!($(Symbol("$(MODEL_NAME)Model_problem")), $(QuoteNode(ZC_HYSTERESIS)),
                                                get(kwargs, :reltol, 1.0e-3))
      #= The initialization reads the delay() arguments themselves, not a previous solve's history. =#
      OMBackend.CodeGeneration.clearDelayHistories!()
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
      #= Stash the exact runtime solve inputs so a manual integrator loop can
         reproduce the real event wiring (debug aid for friction/event work). =#
      global LATEST_SOLVE_TRIPLE = (_problemForSolver, _solver, callbacks)
      local _initKw = OMBackend.CodeGeneration.defaultInitializeKwargs(_problemForSolver, kwargs,
                                                                       $(_modelHasTableClusters(simCode)))
      local _sol = if haskey(kwargs, :callback)
        solve(_problemForSolver, _solver; kwargs..., _initKw...)
      else
        solve(_problemForSolver, _solver; callback=callbacks, kwargs..., _initKw...)
      end
      #= If the init-alg-remake'd problem produced InitialFailure (MTK could
         not reconcile init-alg hard-start u0 with the algebraic constraints),
         fall back to the un-remake'd problem so the solver can pick any
         consistent u0. This is the behaviour without the remake for models where
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
          solve(_fallbackProb, _solver; kwargs..., _initKw...)
        else
          solve(_fallbackProb, _solver; callback=callbacks, kwargs..., _initKw...)
        end
      end
      #= Run `when terminal()` bodies once against the final solution (gated: emitted only if the model has a terminal event). =#
      $(createTerminalBodyRunner(simCode))
      OMBackend.CodeGeneration.dropPreInitializationPoint!(_sol)
    end
    function simulate(tspan = (0.0, 1.0), solver = OMBackend.defaultSolver(); cached_build = nothing, kwargs...)
      local built = cached_build === nothing ? $(Symbol("$(MODEL_NAME)Model"))(tspan) : cached_build
      return simulateFromBuild(built, tspan, solver; kwargs...)
    end
  end
  #= MODEL_NAME is preprocessed with . replaced with _=#
  local moduleExpr = Expr(:module, true, Symbol(MODEL_NAME), stripBeginBlocks(programBody))
  return MODEL_NAME, moduleExpr
end

"""
  Generates a MTK model.

  `earlyInitialAlgorithm`: the model reads the results of
  `__runInitialAlgorithmEarly!`. Only ODE_MODE_MTK_PROGRAM_GENERATION's module
  defines it (and `__runInitialAlgorithm!`); the modes of a structural model
  and the models of a runtime recompilation (Runtime.translateToMTK) have
  neither (one per mode would collide), so their initial algorithms do not
  run, a limitation of those paths.
"""
function ODE_MODE_MTK_MODEL_GENERATION(simCode::SimulationCode.SIM_CODE, modelName, functions;
                                       useDirectRHS::Bool = OMBackend.DIRECT_RHS_GENERATION[],
                                       earlyInitialAlgorithm::Bool = false)
  RESET_CALLBACKS()
  empty!(MTK_CodeGenerationUtil.DELAY_CALLS)
  MTK_CodeGenerationUtil.DELAY_MODEL[] = Symbol(modelName)

  #= Eval the generated Modelica functions and their @register_symbolic
     calls into OMBackend so subsequent codegen sees the bindings. =#
  evalGeneratedFunctionsAndRegister!(modelName, functions, simCode)

  #= Bucket each simvar by varKind (state / algebraic / discrete
     / parameter / array / occ / data-structure / state-derivative) and
     extract StateSelect priority pairs. =#
  local vars = classifyVariables(simCode)
  local stateVariables         = vars.stateVariables
  local algebraicVariables     = vars.algebraicVariables
  local discreteVariables      = vars.discreteVariables
  local parameters             = vars.parameters
  local arrayParameters        = vars.arrayParameters
  local stateDerivatives       = vars.stateDerivatives
  local dataStructureVariables = vars.dataStructureVariables
  local statePriorityPairs     = vars.statePriorityPairs


  local performIndexReduction = simCode.isSingular
  local skipInitializeProb = SimulationCode.hasStructuralTransitions(simCode) ||
                             SimulationCode.hasMetaModel(simCode)
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
  local INITIAL_GUESS_EQUATIONS = createStartConditionsEquationsMTK(stateVariables,
                                                                      algebraicVariables,
                                                                      simCode)


  local DISCRETE_START_VALUES = vcat(generateInitialEquations(simCode.initialEquations, simCode; parameterAssignment = true),
                                     getStartConditionsMTK(discreteVariables, simCode))
  local PARAMETER_EQUATIONS = createParameterEquationsMTK(parameters, simCode)
  local PARAMETER_ASSIGNMENTS = createParameterAssignmentsMTK(parameters, simCode)
  local PARAMETER_RAW_ARRAY = createParameterArray(parameters, PARAMETER_ASSIGNMENTS, simCode)
  local ARRAY_PARAMETERS = createArrayParametersMTK(arrayParameters, simCode)
  #= Legacy callback generation is deferred until after relay elimination so
     the callbacks read the surviving relay representatives (see below). =#
  local IF_EQUATION_COMPONENTS::Vector{IfEquationComponent} =
    createIfEquations(stateVariables, algebraicVariables, simCode)
  local RELAY_GUESS_PAIRS = collect(Iterators.flatten(c.relayGuesses for c in IF_EQUATION_COMPONENTS))
  #= Deterministic t0 values derived from parameters and fixed=true starts;
     they override guess-grade init values (wrong guesses near a guarded
     division start the consistent-IC solve at a blow-up point). =#
  local TRUSTED_GUESS_PAIRS = Expr[:($(QuoteNode(k)) => $(v)) for (k, v) in
                                   MTK_CodeGenerationUtil.buildT0TrustedDerivedPairs(simCode)]
  #= Symbolic names =#
  local algebraicVariablesSym = Symbol[:($(Symbol(v))) for v in algebraicVariables]
  local dataStructureVariablesSym = Symbol[Symbol(v) for v in dataStructureVariables]
  local stateVariablesSym = Symbol[:($(Symbol(v))) for v in stateVariables]
  local parVariablesSym = Symbol[Symbol(p) for p in parameters]
  #= Discrete-dummy demotion. Each discrete variable starts with a
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
                                      length(algebraicVariables))
  (DISCRETE_DUMMY_EQUATIONS, discreteVariablesSym) =
    applyDemotionPlan!(_demotionPlan, discreteVariables, DISCRETE_DUMMY_EQUATIONS,
                       discreteVariablesSym, algebraicVariablesSym)
  #= Flatten the per-if-equation components into one event-decl
     Expr (wrapped in invokelatest because event exprs reference Symbolics
     bindings only created later inside the model function), plus three
     flat lists used by downstream phases. =#
  local IF_EQUATION_EVENTS = collect(Iterators.flatten(c.events for c in IF_EQUATION_COMPONENTS))
  #= Synthesised discrete-Boolean whens become MTK SymbolicContinuousCallbacks
     (observed-variable-capable), appended to the if-equation event vector. =#
  IF_EQUATION_EVENTS = vcat(IF_EQUATION_EVENTS, createDiscreteBoolWhenEvents(simCode),
                            createSelfSchedulingTimeWhenEvents(simCode))
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
  #= Branch events with a hysteresis share the parameter H (set per solve). =#
  local IF_RELATIONS = collect(Iterators.flatten(c.relations for c in IF_EQUATION_COMPONENTS))
  #= Kill switch OMBACKEND_INIT_DISCRETES=false: no initial fixpoint of the
     discrete clusters (_settleInitialDiscretes!). =#
  local INIT_DISCRETE_CLUSTERS = OMBackend.envSwitch("OMBACKEND_INIT_DISCRETES") ?
    _discreteClusterSpecs(simCode) : Expr[]
  local IF_INIT_LITERALS = OMBackend.envSwitch("OMBACKEND_INIT_RELATIONS") ?
    collect(Iterators.flatten(c.initLiterals for c in IF_EQUATION_COMPONENTS)) : Expr[]
  local ifCondParamNames = copy(ifConditionalVariables)
  if !isempty(IF_RELATIONS)
    push!(ifCondParamDecls, Expr(:(=), ZC_HYSTERESIS, ZC_HYSTERESIS_DEFAULT))
    push!(ifCondParamPairs, :($(ZC_HYSTERESIS) => $(ZC_HYSTERESIS_DEFAULT)))
    push!(ifCondParamNames, ZC_HYSTERESIS)
  end
  #= Collect the symbols MTK tearing must not eliminate
     (simCode-flagged irreducibles + ifEq_tmp LHS targets + fixed-start
     variables). =#
  local irreducibleSyms = collectIrreducibleSymbols(simCode, CONDITIONAL_EQUATIONS,
                                                    stateVariables, algebraicVariables)

  #= Heuristic for initialization:
     - If any state variable has an explicit start value, assume the system has algebraic
       constraints and only initialize states with explicit starts (avoid overdetermination).
     - If NO state has an explicit start, provide defaults for all states (pure ODE case).
     - Exception: when build_initializeprob is disabled (structural transition models),
       there is no initialization solver to infer values from constraints/guesses, so
       we MUST provide u0 defaults for all unknowns.
     This handles both constrained DAE systems (like Pendulum) and pure ODE systems
     (like MatrixVectorMult where states have no explicit start). =#
  local startValueVariables = startValueVariableNames(simCode)
  local anyStateHasExplicitStart = hasExplicitStartValue(startValueVariables, simCode)
  local skipDefaultsForStates = anyStateHasExplicitStart
  #= Build default guesses for unknowns not in the heuristic-filtered u0.
     Guesses are passed to ODEProblem so the init solver has fallback values
     without overdetermining the system. =#
  local INITIAL_VALUE_EQUATIONS = unique!(createStartConditionsEquationsMTK(
    startValueVariables,
    String[],
    simCode; skipDefaultStateStarts = skipDefaultsForStates))
  INITIAL_VALUE_EQUATIONS = vcat(DISCRETE_START_VALUES, INITIAL_VALUE_EQUATIONS)
  INITIAL_GUESS_EQUATIONS = vcat(DISCRETE_START_VALUES, INITIAL_GUESS_EQUATIONS)
  #=
    Merge equations. ifCond variables are discrete parameters so they are NOT
    included in stateVariablesSym and do NOT get der() ~ 0 equations.
  =#
  stateVariablesSym = vcat(discreteVariablesSym, stateVariablesSym)
  #= Discretes read by a callback condition must survive relay elimination under
     their own name: the generated condition indexes the state vector by that name,
     so re-aliasing it to another leaf strands the lookup. Force them to be the
     relay component root. =#
  local _condDiscretes = whenConditionDiscreteSyms(simCode)
  #= Every discrete variable with der(v) = 0 must survive structural_simplify
     as an unknown: events change it, and the callbacks read and write it in
     the state vector by name. ModelingToolkit folds a state whose derivative
     is zero into a constant unless it is irreducible (11.45 does; 11.21 kept
     it). The condition discretes are among them. =#
  for _s in discreteVariablesSym
    _s in irreducibleSyms || push!(irreducibleSyms, _s)
  end
  #= Parameters have no module-global symbolic binding (they live in the local
     @parameters block and the pars dict), so a relay must never collapse a
     variable onto one. =#
  local _paramSyms = OrderedSet{Symbol}(Symbol(name)
    for (name, (_, sv)) in simCode.stringToSimVarHT
    if sv.varKind isa SimulationCode.PARAMETER || sv.varKind isa SimulationCode.ARRAY_PARAMETER)
  local (_ifEqRelay_eqs, _ifEqRelay_aliases) = eliminateIfEqRelays(EQUATIONS; preferKeep = _condDiscretes, paramSyms = _paramSyms)
  EQUATIONS = _ifEqRelay_eqs
  if !isempty(_ifEqRelay_aliases)
    @info "[RELAY] aliases" _ifEqRelay_aliases
    local _drop = OrderedSet(keys(_ifEqRelay_aliases))
    local _dropStr = OrderedSet(string.(keys(_ifEqRelay_aliases)))
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
  #= Generate the legacy callback set against when-equations re-pointed at the
     surviving relay representatives, so callback lookups hit live unknowns. =#
  simCode = substituteRelayAliasesInWhens(simCode, _ifEqRelay_aliases)
  local CALL_BACK_EQUATIONS = createCallbackCode(modelName, simCode; generateSaveFunction = false)
  local NAMED_STATE_LOOKUPS = namedStateLookups(CALL_BACK_EQUATIONS)
  #= The callbacks read and write these by name in the state vector, so they
     must survive structural_simplify: an explicit algebraic a when body reads
     (`z = w2` with `w2 = w + 1`) became observed, and the affect failed with a
     KeyError. =#
  local _declared = OrderedSet{Symbol}(vcat(stateVariablesSym, algebraicVariablesSym))
  for _n in NAMED_STATE_LOOKUPS
    local _s = Symbol(_n)
    _s in _declared && !(_s in irreducibleSyms) && push!(irreducibleSyms, _s)
  end
  EQUATIONS = vcat(EQUATIONS,
                   DISCRETE_DUMMY_EQUATIONS,
                   CONDITIONAL_EQUATIONS)
  EQUATIONS = rewriteEquations(EQUATIONS, simCode)
  #= A state with der(v) = 0 must survive structural_simplify too: an initial
     algorithm or initial equation sets it, and it is read by name.
     ModelingToolkit turns such a state into a parameter unless it is
     irreducible (11.45 does; 11.21 kept it). =#
  for _s in zeroDerivativeSymbols(EQUATIONS)
    _s in irreducibleSyms || push!(irreducibleSyms, _s)
  end
  local _seenMtkEquationExprs = OrderedSet{String}()
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
    #= The discretes of the discrete clusters (the direct-RHS initialization
       leaves them to the clusters' start bodies). =#
    $(liftedDiscretesDecl(simCode))
    #= The parameters the initialization computes (fixed = false, no binding). =#
    $(freeParametersDecl(simCode))
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
      $(generateDiscreteIfCondDeclaration(ifCondParamDecls, ifCondParamNames))
      #=
        Only variables that are present in the equation system later should be a part of the variables in the MTK system.
        This means that certain algebraic variables should not be listed among the variables (These are the discrete variables).
      =#
      $(varInnerRefs)
      allVariables = Any[]
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
      startEquationComponents = Any[]
      $(decomposeStartEquationsInline(INITIAL_GUESS_EQUATIONS))
      for constructor in startEquationConstructors
        push!(startEquationComponents, Base.invokelatest(constructor))
      end
      initialValues = collect(Iterators.flatten(startEquationComponents))
      #= Process the final initial guesses =#
      startEquationComponents = Any[]
      $(decomposeStartEquationsInline(INITIAL_VALUE_EQUATIONS; functionSuffix = "Final"))
      for constructor in startEquationConstructors
        push!(startEquationComponents, Base.invokelatest(constructor))
      end
      finalInitialValues = collect(Iterators.flatten(startEquationComponents))
      #= Equations =#
      equationComponents = Any[]
      $(stripBeginBlocks(decomposeEquationsInline(EQUATIONS, PARAMETER_ASSIGNMENTS)))
      for constructor in equationConstructorCalls
        push!(equationComponents, Base.invokelatest(constructor))
      end
      eqs = collect(Iterators.flatten(equationComponents))
      eqs = Base.invokelatest(OMBackend.CodeGeneration.filterConstantEquations, eqs)
      eqs = Base.invokelatest(OMBackend.CodeGeneration.expandExpressionDerivatives, eqs)
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
      local _algResults = $(earlyInitialAlgorithm ? :(Base.invokelatest(__runInitialAlgorithmEarly!)) : :(Dict{Symbol, Float64}()))
      function _buildInitialConstraintEqs()
        local _eqs = Symbolics.Equation[$([_substSyms(e, _ifEqRelay_aliases) for e in generateInitialEquationsAsConstraints(simCode.initialEquations, simCode)]...),
                                        $([_substSyms(e, _ifEqRelay_aliases) for e in getFixedStartConstraintsMTK(vcat(stateVariables, algebraicVariables, discreteVariables), simCode)]...)]
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
      nonLinearSystem = $(odeSystemWithEvents(!isempty(ifConditionalVariables) || !isempty(IF_EQUATION_EVENTS), modelName;
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
        local _seenLHS = OrderedSet{String}()
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
        collect(values(OrderedDict($(DISCRETE_START_VALUES...))))
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
        OMBackend.CodeGeneration.splitInitialValues, reducedSystem, _finalInitialValuesForSplit, _initialValuesForSplit, pars)
      reducedSystem = Base.invokelatest(OMBackend.CodeGeneration.mergeSoftGuesses,
        reducedSystem, Pair{Any, Any}[$(RELAY_GUESS_PAIRS...)])
      reducedSystem = Base.invokelatest(OMBackend.CodeGeneration.mergeSoftGuesses,
        reducedSystem, Pair{Any, Any}[$(TRUSTED_GUESS_PAIRS...)]; force = true)
      #= Build ODEProblem. The codegen-time strategy (DirectRHS / structural
         transition / standard DAE-with-init-solver) is picked here; the
         structural-transition branch additionally dispatches at runtime on
         the mass matrix. See `emitProblemConstruction` and its three
         strategy emitters for the full rationale. =#
      #= The relations the initialization settles (in the latest world, like
         the event list: the symbolic variables are globals bound while the
         model is built). =#
      _ifInitLiterals = Base.invokelatest(() -> Any[$(IF_INIT_LITERALS...)])
      #= The discrete clusters the initialization settles (instances of their own). =#
      _initDiscreteClusters = Base.invokelatest(() -> Any[$(INIT_DISCRETE_CLUSTERS...)])
      $(emitProblemConstruction(useDirectRHS, skipInitializeProb))
      OMBackend.CodeGeneration.checkNamedStateLookups(problem, $(NAMED_STATE_LOOKUPS))
      $(emitDiscreteClusters(simCode))
      #= Only with delay() calls: a callback on every model made each one a model with
         events (OMSurrogates' UDEs refuse those). =#
      $(isempty(MTK_CodeGenerationUtil.DELAY_CALLS) ? :() :
        :(callbacks = OMBackend.CodeGeneration.withDelayEvents(callbacks, $(QuoteNode(Symbol(MODEL_NAME))))))
      $(emitRelationRefresh(IF_RELATIONS))
      callbacks = OMBackend.CodeGeneration.withIntegralDiscretes(callbacks, problem, $(integralDiscreteNames(discreteVariablesSym, simCode)))
      #= Asserts after the event iteration: they check the settled state. =#
      $(emitAssertCallback(simCode))
      #= In the latest world, as the lists above: the arguments read the model's
         variables, bound by eval in this function (MSL Digital FullAdder). =#
      callbacks = OMBackend.CodeGeneration.withDelayRecords(callbacks, problem, $(QuoteNode(Symbol(MODEL_NAME))),
                    Base.invokelatest(() -> Any[$([c[2] for c in MTK_CodeGenerationUtil.DELAY_CALLS]...)]),
                    Base.invokelatest(() -> Any[$([c[3] for c in MTK_CodeGenerationUtil.DELAY_CALLS]...)]))
      #= First among the discrete callbacks: it reads the step as the solver took it. =#
      callbacks = OMBackend.CodeGeneration.withAlgebraicStepControl(callbacks, problem)
      return (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars, irreducibleSyms)
    end
  end
  #= Qualify bare Modelica function calls with OMBackend.CodeGeneration. prefix.
     This covers all generated code: equations, parameter assignments, start conditions. =#
  local funcNames = OrderedSet{Symbol}(Symbol(f.name) for f in simCode.functions)
  if !isempty(funcNames)
    qualifyModelicaFunctions!(model, funcNames)
  end
  return model
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
    #= eq.exp is a `SimulationCode.Exp`; the
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
  local emittedElims = OrderedSet{Symbol}()
  for entry in simCode.aliasMap
    local elimSym = Symbol(entry.eliminatedName)
    local repSym = Symbol(entry.representativeName)
    repSym = get(relayAliases, repSym, repSym)
    push!(elimSymbols, elimSym)
    push!(emittedElims, elimSym)
    push!(obsEntries, (elimSym, repSym, entry.negated))
  end
  local _relayRepresentative(sym::Symbol)::Symbol = begin
    local seen = OrderedSet{Symbol}()
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
  local aliasNames = OrderedSet{String}(entry.eliminatedName for entry in simCode.aliasMap)
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
