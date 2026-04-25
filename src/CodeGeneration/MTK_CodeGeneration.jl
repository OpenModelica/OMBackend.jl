#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
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
import ..AlgorithmicCodeGeneration

"""
  Generates simulation code targetting modeling toolkit.
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
  local MODEL_NAME = replace(simCode.name, "." => "__")
  #= Generate code for algorithmic Modelica =#
  (functions, functionNames) = AlgorithmicCodeGeneration.generateFunctions(simCode.functions)
  if isempty(simCode.structuralTransitions) && length(simCode.subModels) < 1 && isnothing(simCode.flatModel)
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
    using ModelingToolkit
    using DifferentialEquations
    Base.Experimental.@compiler_options optimize=0 compile=min
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
         as hard u0 while letting the initialiser use reducedSystem.guesses
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
  local MODEL_NAME = replace(modelName, "." => "__")
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
  local funcNames = Set{Symbol}(Symbol(replace(f.name, "." => "_")) for f in simCode.functions)
  if !isempty(funcNames)
    for f in functions
      qualifyModelicaFunctions!(f, funcNames)
    end
  end
  programBody = quote
    using ModelingToolkit
    using DifferentialEquations
    using OrdinaryDiffEq
    using Symbolics
    using OMBackend
    Base.Experimental.@compiler_options optimize=0 compile=min
    #= Add import to the external runtime if the generated code calls Modelica Functions =#
    $(if simCode.externalRuntime
        generateExternalRuntimeImport(simCode)
      end)
    $(functions...)
    $(DATA_STRUCTURE_ASSIGNMENTS...)
    $(generateRegisterCallsForCallExprs(simCode)...)
    $(model)
    function simulate(tspan = (0.0, 1.0), solver = Rodas5();  kwargs...)
      ($(Symbol("$(MODEL_NAME)Model_problem")), callbacks, ivs, _ivs_all, $(Symbol("$(MODEL_NAME)Model_ReducedSystem")), _tspan2, _pars, _vars, _irreductable) = $(Symbol("$(MODEL_NAME)Model"))(tspan)
      global LATEST_REDUCED_SYSTEM = $(Symbol("$(MODEL_NAME)Model_ReducedSystem"))
      solve($(Symbol("$(MODEL_NAME)Model_problem")), solver; kwargs...)
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
  #@debug "Runnning: ODE_MODE_MTK_MODEL"
  RESET_CALLBACKS()
  #= Eval functions and register them with @register_symbolic before equations are processed =#
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
      if !contains(string(e), "already has a value")
        rethrow(e)
      end
    end
  end
  local stringToSimVarHT = simCode.stringToSimVarHT
  local equations::Vector = BDAE.RESIDUAL_EQUATION[]
  local exp::DAE.Exp
  local parameters::Vector = String[]
  local arrayParameters::Vector = String[]
  local stateDerivatives::Vector = String[]
  local stateVariables::Vector = String[]
  local algebraicVariables::Vector = String[]
  local discreteVariables::Vector = String[]
  local occVariables::Vector = String[]
  local occDummyVariables::Vector = String[]
  local dataStructureVariables::Vector = String[]
  local performIndexReduction = false
  local statePriorityPairs = Tuple{Symbol, Float64}[]
  for varName in keys(stringToSimVarHT)
    (idx, var) = stringToSimVarHT[varName]
    local varType = var.varKind
    @match varType  begin
      SimulationCode.INPUT(__) => begin
        @error "INPUT not supported in CodeGen"
        throw()
      end
      SimulationCode.STATE(__) => begin
        push!(stateVariables, varName)
      end
      SimulationCode.OCC_VARIABLE(__) => begin
        push!(occVariables, varName)
      end
      SimulationCode.PARAMETER(__) => begin
        push!(parameters, varName)
      end
      SimulationCode.STRING(__) => begin
        #= String parameters are non-numeric; excluded from MTK parameter system. =#
      end
      SimulationCode.ARRAY_PARAMETER(__) => begin
        push!(arrayParameters, varName)
      end
      SimulationCode.ARRAY(__) => begin
        push!(stateVariables, varName)
      end
      SimulationCode.DISCRETE(__) => begin
        push!(discreteVariables, varName)
      end
      SimulationCode.ALG_VARIABLE(__) => begin
        #=TODO: Seems to be unable to find variables in some cases...
        should there be an and here? Keep track of variables that are also involved in if/when structures
        =#
        if idx in simCode.matchOrder
          push!(algebraicVariables, varName)
        elseif involvedInEvent(idx, simCode) #= We have a variable that is not contained in continuous system =#
          #= Treat discrete variables separate =#
          push!(discreteVariables, varName)
        elseif simCode.isSingular
          #=
          If the variable is not involved in an event and the index is not in match order and
          the system is singular.
          This means that the variable is probably algebraic however, we need to perform index reduction.
          =#
          push!(algebraicVariables, varName)
        else # The system is singular but it was not detected by the backend...
          @assign simCode.isSingular = true
          push!(algebraicVariables, varName)
        end
      end
      #=
      Handled at the top level, these variables are global constants.
      However, these are saved in the dataStructureVariables array in order to make the system available during the equation rewrite.
      =#
      SimulationCode.DATA_STRUCTURE(__) => begin
        push!(dataStructureVariables, varName)
      end
      #=TODO:johti17 Do I need to modify this?=#
      SimulationCode.STATE_DERIVATIVE(__) => push!(stateDerivatives, varName)
    end
    #= Extract StateSelect annotation and map to MTK state_priority =#
    local optAttrs::Option{DAE.VariableAttributes} = var.attributes
    local _sp = @match optAttrs begin
      SOME(attrs && DAE.VAR_ATTR_REAL(__)) => begin
        @match attrs.stateSelectOption begin
          SOME(DAE.NEVER(__)) => -10.0
          SOME(DAE.AVOID(__)) => -2.0
          SOME(DAE.PREFER(__)) => 2.0
          SOME(DAE.ALWAYS(__)) => 10.0
          _ => nothing
        end
      end
      _ => nothing
    end
    if _sp !== nothing
      #= State priority metadata only makes sense on actual MTK unknowns.
         Helper parameters such as *_start may carry stateSelect from source
         attributes, but they are not created as MTK variables and would cause
         UndefVarError during the batched eval below. =#
      local varNameStr = string(varName)
      local supportsStatePriority =
        varType isa SimulationCode.STATE ||
        varType isa SimulationCode.ALG_VARIABLE ||
        varType isa SimulationCode.OCC_VARIABLE ||
        varType isa SimulationCode.ARRAY
      if supportsStatePriority && !startswith(varNameStr, "der(")
        push!(statePriorityPairs, (Symbol(varName), _sp))
      end
    end
  end
  local performIndexReduction = simCode.isSingular
  local skipInitializeProb = !isempty(simCode.structuralTransitions) ||
                             !isnothing(simCode.metaModel) ||
                             !isnothing(simCode.flatModel)
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
  local START_CONDTIONS_EQUATIONS = createStartConditionsEquationsMTK(vcat(stateVariables, occVariables),
                                                                      algebraicVariables,
                                                                      simCode)


  local DISCRETE_START_VALUES = vcat(generateInitialEquations(simCode.initialEquations, simCode; parameterAssignment = true),
                                     getStartConditionsMTK(discreteVariables, simCode))
  local PARAMETER_EQUATIONS = createParameterEquationsMTK(parameters, simCode)
  local PARAMETER_ASSIGNMENTS = createParameterAssignmentsMTK(parameters, simCode)
  local PARAMETER_RAW_ARRAY = createParameterArray(parameters, PARAMETER_ASSIGNMENTS, simCode)
  local ARRAY_PARAMETERS = createArrayParametersMTK(arrayParameters, simCode)
  #= Create callback equations.
    For MTK we disable the saving function for now.
  =#
  local CALL_BACK_EQUATIONS = createCallbackCode(modelName, simCode; generateSaveFunction = false)
  local IF_EQUATION_COMPONENTS::Vector{Tuple{Vector{Expr}, Vector{Expr}, Vector{Expr}, Vector{Symbol}, Vector{Tuple}}} =
    createIfEquations(stateVariables, algebraicVariables, simCode)
  #= Symbolic names =#
  local algebraicVariablesSym = Symbol[:($(Symbol(v))) for v in algebraicVariables]
  local dataStructureVariablesSym = Symbol[Symbol(v) for v in dataStructureVariables]
  local stateVariablesSym = Symbol[:($(Symbol(v))) for v in stateVariables]
  local occVariablesSym = Symbol[:($(Symbol(v))) for v in occVariables]
  local parVariablesSym = Symbol[Symbol(p) for p in parameters]
  #=Preprocess the component of the if equations.
    Generate der(x) ~ 0 for all discrete variables (they need to be in the ODE
    state vector for callbacks). Then detect if the total system is over-determined
    due to discrete variables that also have alias/connector equations. Remove
    exactly the excess dummy derivatives, targeting variables that appear in
    pure alias equations (form 0 ~ a - b) in the residual system. =#
  local discreteVariablesSym = Symbol[:($(Symbol(v))) for v in discreteVariables]
  local DISCRETE_DUMMY_EQUATIONS = [:(der($(Symbol(dv))) ~ 0) for dv in discreteVariables]
  #= Check for over-determination: count total equations vs total unknowns.
     nEqs = residual equations + discrete dummy equations + conditional equations
     nVars = state vars + algebraic vars + discrete vars + occ vars
     If nEqs > nVars, we have excess from discrete alias equations. =#
  local _nConditionalEqs = sum(length(component[2]) for component in IF_EQUATION_COMPONENTS; init = 0)
  local _nTotalEqs = length(EQUATIONS) + length(DISCRETE_DUMMY_EQUATIONS) + _nConditionalEqs
  local _nTotalVars = length(stateVariables) + length(algebraicVariables) + length(discreteVariables) + length(occVariables)
  local _excess = _nTotalEqs - _nTotalVars
  if _excess > 0
    #= Find discrete variables that appear in alias-like residual equations
       (pure connector pass-throughs). These are safe to demote to algebraic
       because they are fully determined by the alias equation alone. =#
    local _discreteSet = Set(string.(Symbol.(discreteVariables)))
    local _eqStrings = [string(eq) for eq in EQUATIONS]
    local _aliasDiscrete = String[]
    for dv in discreteVariables
      local dvStr = string(Symbol(dv))
      #= Count how many residual equations mention this discrete variable =#
      local nMentions = count(s -> occursin(dvStr, s), _eqStrings)
      if nMentions > 0
        push!(_aliasDiscrete, dv)
      end
    end
    #= Remove exactly _excess dummy derivatives, preferring variables with
       the most residual equation mentions (most constrained = safest to remove). =#
    local _toRemove = Set{String}()
    for dv in _aliasDiscrete
      length(_toRemove) >= _excess && break
      push!(_toRemove, dv)
    end
    if !isempty(_toRemove)
      @info "Discrete alias fix: removing $(_excess) excess dummy der equations for $(collect(_toRemove))"
      local _newDummy = Expr[]
      local _newDiscreteSym = Symbol[]
      for (i, dv) in enumerate(discreteVariables)
        if dv in _toRemove
          push!(algebraicVariablesSym, Symbol(dv))
        else
          push!(_newDummy, DISCRETE_DUMMY_EQUATIONS[i])
          push!(_newDiscreteSym, Symbol(dv))
        end
      end
      DISCRETE_DUMMY_EQUATIONS = _newDummy
      discreteVariablesSym = _newDiscreteSym
    end
  end
  #= Create assignments for the dummies. =#
  local IF_EQUATION_EVENTS = [component[1] for component in IF_EQUATION_COMPONENTS]
  IF_EQUATION_EVENTS = collect(Iterators.flatten(IF_EQUATION_EVENTS))
  #= Use Base.invokelatest to wrap event creation, so it runs in the current world age =#
  #= This is necessary because the event expressions reference variables created via eval =#
  local IF_EQUATION_EVENT_DECLARATION = if isempty(IF_EQUATION_EVENTS)
    :(events = [])
  else
    :(events = Base.invokelatest(() -> [$(IF_EQUATION_EVENTS...)]))
  end
  local CONDITIONAL_EQUATIONS = collect(Iterators.flatten([component[2] for component in IF_EQUATION_COMPONENTS]))
  local ifConditionNameAndIV = collect(Iterators.flatten([component[5] for component in IF_EQUATION_COMPONENTS]))
  local ifConditionalVariables = collect(Iterators.flatten([component[4] for component in IF_EQUATION_COMPONENTS]))
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
  #=
  In the latest variant of MTK we can not reuse the old variables for the creation of the ODEProblem later.
  Instead, we should only use the values for the unknowns we can't remove from the system.
  =#
  local irreductableSyms = Symbol[Symbol(vn) for vn in simCode.irreductableVariables]
  #= Mark ifEq_tmp LHS variables as irreducible so MTK tearing does not
     eliminate them. ifCond variables are discrete parameters (not unknowns)
     and do not need irreducible marking. =#
  for ceq in CONDITIONAL_EQUATIONS
    if ceq isa Expr && ceq.head == :call && length(ceq.args) >= 2
      lhs = ceq.args[2]
      if lhs isa Symbol
        push!(irreductableSyms, lhs)
      end
    end
  end

  #= Heuristic for initialization:
     - If any state variable has an explicit start value, assume the system has algebraic
       constraints and only initialize states with explicit starts (avoid overdetermination).
     - If NO state has an explicit start, provide defaults for all states (pure ODE case).
     - Exception: when build_initializeprob is disabled (structural transition models),
       there is no initialization solver to infer values from constraints/guesses, so
       we MUST provide u0 defaults for all unknowns.
     This handles both constrained DAE systems (like Pendulum) and pure ODE systems
     (like MatrixVectorMult where states have no explicit start). =#
  local anyStateHasExplicitStart = hasExplicitStartValue(simCode.irreductableVariables, simCode)
  local skipDefaultsForStates = anyStateHasExplicitStart
  #= Build default guesses for unknowns not in the heuristic-filtered u0.
     Guesses are passed to ODEProblem so the init solver has fallback values
     without overdetermining the system. =#
  local FINAL_START_CONDTIONS_EQUATIONS = unique!(createStartConditionsEquationsMTK(
    String[vn for vn in simCode.irreductableVariables],
    String[],
    simCode; skipDefaultAlgebraicStarts = skipDefaultsForStates))
  FINAL_START_CONDTIONS_EQUATIONS = vcat(DISCRETE_START_VALUES, FINAL_START_CONDTIONS_EQUATIONS)
  START_CONDTIONS_EQUATIONS = vcat(DISCRETE_START_VALUES, START_CONDTIONS_EQUATIONS)
  #=
    Merge equations. ifCond variables are discrete parameters so they are NOT
    included in stateVariablesSym and do NOT get der() ~ 0 equations.
  =#
  stateVariablesSym = vcat(discreteVariablesSym,
                           stateVariablesSym,
                           occVariablesSym)
  EQUATIONS = vcat(EQUATIONS,
                   DISCRETE_DUMMY_EQUATIONS,
                   CONDITIONAL_EQUATIONS)
  EQUATIONS = rewriteEquations(EQUATIONS, simCode)
  #= Reset the callback counter=#
  RESET_CALLBACKS()
  #=
    Formulate the problem as a DAE Problem.
    For this variant we keep it on its own line
    https://github.com/SciML/ModelingToolkit.jl/issues/998
  =#
  #=If our model name is separated by . replace it with __ =#
  local MODEL_NAME = replace(modelName, "." => "__")
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
      local irreductableSyms = $(irreductableSyms)
      for sym in irreductableSyms
        push!(_batchBlock.args, :($sym = SymbolicUtils.setmetadata($sym, ModelingToolkit.VariableIrreducible, true)))
      end
      local _statePriorityPairs = $(statePriorityPairs)
      for (sym, priority) in _statePriorityPairs
        push!(_batchBlock.args, :($sym = SymbolicUtils.setmetadata($sym, ModelingToolkit.VariableStatePriority, $priority)))
      end
      eval(_batchBlock)
      #= Transform the variable vector into a vector of Nums =#
      vars = map(x -> last(x), vars)
      #= Initial values for the continuous system. =#
      $(decomposeParameterEquationsInline(PARAMETER_EQUATIONS))
      #= Add ifCond discrete parameter values to pars dict =#
      $(generateIfCondParamAssignments(ifCondParamPairs))
      startEquationComponents = []
      $(decomposeStartEquationsInline(START_CONDTIONS_EQUATIONS))
      for constructor in startEquationConstructors
        push!(startEquationComponents, Base.invokelatest(constructor))
      end
      initialValues = collect(Iterators.flatten(startEquationComponents))
      #= Process the final initial guesses =#
      startEquationComponents = []
      $(decomposeStartEquationsInline(FINAL_START_CONDTIONS_EQUATIONS; functionSuffix = "Final"))
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
      #= Events and observed equations =#
      $(IF_EQUATION_EVENT_DECLARATION)
      $(generateAliasObservedBlock(simCode))
      $(generateEliminatedObservedBlock(simCode))
      #= ODE System =#
      nonLinearSystem = $(odeSystemWithEvents(!isempty(ifConditionalVariables), modelName;
                                              hasObserved = !isempty(simCode.aliasMap) || !isempty(simCode.eliminatedVariables)))
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
      (reducedSystem, finalInitialValues) = Base.invokelatest(
        OMBackend.CodeGeneration.splitInitialValues, reducedSystem, finalInitialValues, initialValues)
      #= Build ODEProblem =#
      if $(useDirectRHS)
        problem = OMBackend.CodeGeneration.buildDirectRHSProblem(
          reducedSystem, finalInitialValues, pars, tspan, callbacks;
          allInitialValues=initialValues)
      else
        if $(skipInitializeProb)
          #= Structural-transition submodel. Dispatch on the mass matrix:
             - Pure ODE (identity mass matrix): all unknowns are differential,
               there are no constraints to solve, so provide u0 for ALL unknowns
               (filling algebraic defaults via buildDefaultGuesses at 0.0) and
               skip the initialization solver. This preserves the fast path for
               models such as BouncingBall and FreeFall.
             - DAE (singular mass matrix, e.g. Pendulum with algebraic x = L*sin(phi)
               constraints): splitInitialValues has already pinned explicit-start
               algebraic IVs as hard u0 and registered 0.0 soft guesses for
               uncovered differential states on reducedSystem.guesses. Pass only
               finalInitialValues as u0 and let MTK's initialiser solve the
               algebraic residuals consistently. Injecting _missingU0 as hard u0
               would override the guesses (phi=0 instead of phi=3π/4) and silently
               violate the constraint, so do not merge it here. =#
          local _isPureODE = Base.invokelatest(
            OMBackend.CodeGeneration.isPureODESystem, reducedSystem)
          if _isPureODE
            local _missingU0 = Base.invokelatest(
              OMBackend.CodeGeneration.buildDefaultGuesses, reducedSystem, finalInitialValues, initialValues)
            problem = ModelingToolkit.ODEProblem(reducedSystem,
                                                 merge(Dict(finalInitialValues), _missingU0, pars),
                                                 tspan;
                                                 callback=callbacks,
                                                 warn_initialize_determined=false,
                                                 build_initializeprob=false)
          else
            problem = ModelingToolkit.ODEProblem(reducedSystem,
                                                 merge(Dict(finalInitialValues), pars),
                                                 tspan;
                                                 callback=callbacks,
                                                 warn_initialize_determined=false)
          end
        else
          #= Non-structural models: let MTK default initialization handle it =#
          problem = ModelingToolkit.ODEProblem(reducedSystem,
                                               merge(Dict(finalInitialValues), pars),
                                               tspan;
                                               callback=callbacks,
                                               warn_initialize_determined=false)
        end
      end
      return (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars, irreductableSyms)
    end
  end
  #= Qualify bare Modelica function calls with OMBackend.CodeGeneration. prefix.
     This covers all generated code: equations, parameter assignments, start conditions. =#
  local funcNames = Set{Symbol}(Symbol(replace(f.name, "." => "_")) for f in simCode.functions)
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
function generateAliasObservedBlock(simCode::SimulationCode.SIM_CODE)
  if isempty(simCode.aliasMap)
    return :(observedEqs = [])
  end
  #= Generate the observed equations as runtime code.
     The alias map entries are known at code-gen time, so we can embed
     the variable names as string literals. At runtime, these create
     Symbolics variables and equations. =#
  local obsExprs = Expr[]
  for entry in simCode.aliasMap
    local elimSym = Symbol(entry.eliminatedName)
    local repSym = Symbol(entry.representativeName)
    if entry.negated
      push!(obsExprs, :($(elimSym) ~ -$(repSym)))
    else
      push!(obsExprs, :($(elimSym) ~ $(repSym)))
    end
  end
  #= Collect eliminated symbol names at code-gen time. The Num objects
     are constructed at runtime (below) using the function-scope `t` so
     they share the system's independent variable. =#
  local elimSymbols = Symbol[Symbol(entry.eliminatedName) for entry in simCode.aliasMap]
  #= Chunk observed equations into small functions (25 each) to avoid
     compiling one massive lambda. For Engine1a with 876 aliases, the single
     lambda took ~101s. Chunking into 36 functions of 25 should be much faster. =#
  local obsChunks = collect(Iterators.partition(obsExprs, 25))
  local chunkFuncDefs = Expr[]
  local chunkFuncNames = Symbol[]
  for (i, chunk) in enumerate(obsChunks)
    local fName = Symbol("_generateObservedEqs_", i - 1)
    push!(chunkFuncDefs, quote
      function $(fName)()
        [$(collect(chunk)...)]
      end
    end)
    push!(chunkFuncNames, fName)
  end
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
    #= Create observed equations in chunks =#
    $(chunkFuncDefs...)
    local _obsComponents = []
    for _obsFn in [$(chunkFuncNames...)]
      push!(_obsComponents, Base.invokelatest(_obsFn))
    end
    observedEqs = collect(Iterators.flatten(_obsComponents))
  end
end

"""
  Structurally walk a DAE expression and return true iff any subterm is a
  `der(...)` call. Used to skip emitting observed equations whose solved
  right-hand side would contain a `Differential(t)` operator; those equations
  are rejected by MTK's initialization-system builder, which constructs its
  System via the 3-argument form (eqs, unknowns, params) with iv=nothing.
"""
function containsDerCall(@nospecialize(exp::DAE.Exp))::Bool
  @match exp begin
    DAE.CALL(Absyn.IDENT("der"), _) => true
    DAE.UNARY(_, e) => containsDerCall(e)
    DAE.BINARY(e1, _, e2) => containsDerCall(e1) || containsDerCall(e2)
    DAE.LUNARY(_, e) => containsDerCall(e)
    DAE.LBINARY(e1, _, e2) => containsDerCall(e1) || containsDerCall(e2)
    DAE.RELATION(e1, _, e2) => containsDerCall(e1) || containsDerCall(e2)
    DAE.IFEXP(c, t, e) => containsDerCall(c) || containsDerCall(t) || containsDerCall(e)
    DAE.CALL(_, explst) => any(containsDerCall, explst)
    DAE.CAST(_, e) => containsDerCall(e)
    DAE.ASUB(e, _) => containsDerCall(e)
    DAE.TSUB(e, _, _) => containsDerCall(e)
    _ => false
  end
end

function generateEliminatedObservedBlock(simCode::SimulationCode.SIM_CODE)
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
  local solveBodyExprs = Expr[]
  for (i, varName) in enumerate(elimVars)
    if containsDerCall(elimEqs[i].exp)
      continue
    end
    local elimSym = Symbol(varName)
    local residualExpr = expToJuliaExpMTK(elimEqs[i].exp, simCode; derSymbol = false)
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
function createResidualEquationsMTK(stateVariables::Vector, algebraicVariables::Vector, equations::Vector{BDAE.RESIDUAL_EQUATION}, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  if isempty(equations)
    return Expr[]
  end
  local eqs::Vector{Expr} = Expr[]
  for eq in equations
    local eqDAEExp = eq.exp
    local eqExp = :(0 ~ $(expToJuliaExpMTK(eqDAEExp, simCode; derSymbol=false)))
    push!(eqs, eqExp)
  end
    return eqs
end

"""
  Generates the initial value for the equations
  TODO: Currently unable to generate start condition in order

If `skipDefaultAlgebraicStarts` is true, algebraic variables without explicit start values are skipped.
State variables always get initialization since MTK ODEProblem requires all unknowns to have initial values.
"""
function createStartConditionsEquationsMTK(states::Vector,
                                        algebraics::Vector,
                                        simCode::SimulationCode.SimCode; skipDefaultAlgebraicStarts = false)::Vector{Expr}
  #= Both states and algebraics respect skipDefaultAlgebraicStarts (reverting to original behavior) =#
  local algInit = getStartConditionsMTK(algebraics, simCode; skipDefaultStarts = skipDefaultAlgebraicStarts)
  local stateInit = getStartConditionsMTK(states, simCode; skipDefaultStarts = skipDefaultAlgebraicStarts)
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
  Generates initial equations.
  Currently unsorted unless they are sorted before being passed to the simulation code phase.
"""
function generateInitialEquations(initialEqs, simCode::SimulationCode.SimCode; parameterAssignment = true)::Vector{Expr}
  local initialEqsExps = Expr[]
  for ieq in initialEqs
    #= COMPLEX_EQUATION/ARRAY_EQUATION should have been expanded before this point =#
    if ieq isa BDAE.COMPLEX_EQUATION || ieq isa BDAE.ARRAY_EQUATION
      error("generateInitialEquations: unexpected unexpanded $(typeof(ieq)) in initial equations — this is a compiler bug upstream")
    end
    #= Skip parametric-only initial equations (already solved by solveParametricInitialEquations!) =#
    if isParametricOnlyEquation(ieq, simCode)
      continue
    end
    #= LHS will typically be a variable. Don't have to be though.. =#
    lhs = expToJuliaExpMTK(ieq.lhs, simCode)
    rhs = @match ieq.rhs begin
      #= `time` is the independent variable and never appears in
         stringToSimVarHT. Route it directly through expToJuliaExpMTK
         which emits the Julia symbol `t` for it. Without this guard
         the generic DAE.CREF arm below indexes the HT with key
         `"time"` and throws KeyError. Surfaced by models like
         Modelica.Fluid.Examples.ControlledTankSystem.ControlledTanks
         whose initial equations contain `<var> = time`. =#
      DAE.CREF(DAE.CREF_IDENT("time", _, _), _) => begin
        expToJuliaExpMTK(ieq.rhs, simCode)
      end
      DAE.CREF(__) => begin
        #= Evaluate the right hand side at this point =#
        local crefAsStr = string(ieq.rhs)
        local simCodeVar = last(simCode.stringToSimVarHT[crefAsStr])
        local res = if SimulationCode.isStateOrAlgebraic(simCodeVar)
          expToJuliaExpMTK(ieq.rhs, simCode)
        elseif SimulationCode.hasBindingExp(simCodeVar)
          evalSimCodeParameter(simCodeVar, simCode)
        else
          #= Parameter without binding (fixed=false): leave as symbol =#
          expToJuliaExpMTK(ieq.rhs, simCode)
        end
      end
      #= For more complicated expressions, we do local constant folding. =#
      _ => begin
        res = evalDAE_Expression(ieq.rhs, simCode)
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
  Checks if any variable in the list has an explicit start attribute.
  Used to determine if the system likely has algebraic constraints that
  determine some state variables, avoiding overdetermination.
"""
function hasExplicitStartValue(vars::Vector, simCode::SimulationCode.SimCode)::Bool
  local ht::Dict = simCode.stringToSimVarHT
  for var in vars
    local entry = get(ht, var, nothing)
    if entry === nothing
      continue
    end
    (_, simVar) = entry
    local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
    @match optAttributes begin
      SOME(attributes) => begin
        @match attributes.start begin
          SOME(_) => return true
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  return false
end

"""
  Given a vector of variables and the simulation code
  extracts the start attributes to generate initial conditions.

If `skipDefaultStarts` is true, variables without explicit start values are skipped.
When false, variables without start values get default 0.0 initialization.
"""
function getStartConditionsMTK(vars::Vector, simCode::SimulationCode.SimCode; skipDefaultStarts = false)::Vector{Expr}
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
            #= Check if CREF has subscripts. If so, delegate to expToJuliaExpMTK =#
            if !isempty(FrontendUtil.Util.getSubscriptsFromCref(start))
              push!(startExprs,
                    quote
                      $(Symbol("$varName")) => $(expToJuliaExpMTK(DAE.CREF(start, DAE.T_REAL(MetaModelica.Nil())), simCode))
                    end)
            else
              #= Simple CREF without subscripts. Look up in pars list =#
              push!(startExprs,
                    quote
                      $(Symbol("$varName")) => pars[$(Symbol(string(start)))]
                    end)
            end
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
            if !skipDefaultStarts
              push!(startExprs, :($(Symbol(varName)) => 0.0))
            end
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
  Creates the components of the If-Equations.
Each if equation is marked by the identifier.
So the first will have 1 and so on.
"""
function createIfEquations(stateVariables, algebraicVariables, simCode)
  local ifEquations = Tuple{Vector{Expr}, Vector{Expr}, Vector{Expr}, Vector{Symbol}, Vector{Tuple}}[]
  local identifier::Int
  #= The identifier is increased by 1 in each iteration. =#
  for (identifier, ifEq) in enumerate(simCode.ifEquations)
    push!(ifEquations, createIfEquation(stateVariables, algebraicVariables, ifEq, identifier, simCode))
  end
  return ifEquations
end

"""
This function creates symbolic if equations for use in MTK.
The function returns a tuple, where the first part of the tuple represent the conditions and the affect of the if-equation on the form:
  continuous_events = [
    <Condtion> => <affect>
    <Condtion> => <affect>
    ....
  ]
Each condition generates one variable with zero dynamics the variable being true or not depending on the branch.
  Example:
  if <condtion> then
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
An if equation with a single condition would only generate one condtion:
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
                          simCode)
  local result::Tuple{Vector{Expr}, Vector{Expr}, Vector{Expr}, Vector{Symbol}, Vector{Tuple}}
  local i::Int = 0
  local nBranches::Int = length(ifEq.branches)
  local branchesWithConds::Int = nBranches - 1
  #= Collect all ifCond symbols for this if-equation.
     These are parameters modified by imperative affects. =#
  local allIfCondSyms = [Symbol(string("ifCond", identifier, j)) for j in 1:branchesWithConds]
  local conditions = Expr[]
  local ivConditions = Bool[]
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
           modified NamedTuple maps aliases to the symbolic parameter variables. =#
        local returnKws = [Expr(:kw, sym, (j == i) ? numVal : invVal) for (j, sym) in enumerate(allIfCondSyms)]
        local returnNT = Expr(:tuple, Expr(:parameters, returnKws...))
        local fExpr = :((modified, observed, ctx, integrator) -> $returnNT)
        local modifiedKws = [Expr(:kw, sym, sym) for sym in allIfCondSyms]
        local modifiedNT = Expr(:tuple, Expr(:parameters, modifiedKws...))
        local affectTuple = :(($(fExpr), $(modifiedNT)))
        #= Wrap in SymbolicContinuousCallback with ImperativeAffect.
           Use NoInit so the solver continues from current state. =#
        local cond = :(ModelingToolkit.SymbolicContinuousCallback(
          ($(mtkCond)) => $(affectTuple);
          reinitializealg = SciMLBase.NoInit()
        ))
        push!(conditions, cond)
        push!(ivConditions, ivCond)
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
  #= ifCond variables are discrete parameters (not ODE unknowns), so they do not need
     der() ~ 0 equations. Collect their names and initial values for parameter declaration. =#
  conditionEquations = Expr[]  #= empty: no dynamics equations for discrete parameters =#
  conditionVariables = Symbol[]
  conditionVariableNames = Tuple{String, Bool}[]
  for i in 1:length(ivConditions)
    push!(conditionVariables, Symbol(string("ifCond", identifier, i)))
    push!(conditionVariableNames, (string("ifCond", identifier, i), !(ivConditions[i])))
  end
  local conditionExpr = conditions
  local result = (conditionExpr, ifExpressions, conditionEquations, conditionVariables, conditionVariableNames)
  return result
end

"""
  `createParameterEquationsMTK(parameters::Vector, type, simCode::SimulationCode.SimCode)`
    The Type specifies what kind of parameter equation a call to this function should yield.
"""
function createParameterEquationsMTK(parameters::Vector, simCode::SimulationCode.SimCode)::Vector{Expr}
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
      Check if conversions are needed
    =#
    expr = if isIntOrBool(bindExp)
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        $(Symbol(simVar.name)) => float($((expToJuliaExpMTK(bindExp, simCode))))
      end
    else
        :($(Symbol(simVar.name)) => $(expToJuliaExpMTK(bindExp, simCode)))
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
function createArrayParametersMTK(arrayParameters::Vector, simCode::SimulationCode.SimCode)::Vector{Expr}
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
                                       simCode::SimulationCode.SimCode)::Vector{Expr}
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
  Creates assignments for complex data structure that exist in the model.
"""
function createDataStructureAssignments(dataStructureVariables::Vector{String}, simCode::SimulationCode.SimCode)::Vector{Expr}
  local dsAssignments::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  for ds in dataStructureVariables
    (index, simVar) = ht[ds]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.DATA_STRUCTURE(bindExp = SOME(exp)) => exp
      _ => ErrorException("Data structure variable not assigned.")
    end
    expr = quote
      $(LineNumberNode(@__LINE__, "$ds eq"))
      $(Symbol(simVar.name)) = $(expToJuliaExpMTK(bindExp, simCode))
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
function createParameterArray(parameters::Vector{T1}, parameterAssignments::Vector{T2}, simCode::SIM_T) where {T1, T2, SIM_T}
  local paramArray = []
  local hT = simCode.stringToSimVarHT
  for param in parameters
    (index, simVar) = hT[param]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => exp
      _ => ErrorException("Unknown SimulationCode.SimVarType for parameter.")
    end
    #= Evaluate the parameters. If it is a variable, and can't be evaluated look it up in the parameter dictonary. =#
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
  local stateVectors = collect(Iterators.partition(stateVariables, 50))
  local algVectors = collect(Iterators.partition(algebraicVariables, 50))
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
function decomposeEquations(equations, parameterAssignments; modelPrefix::String = "")
  local equationVectors = collect(Iterators.partition(equations, 50))
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
function decomposeStartEquations(equations; functionSuffix = "", modelPrefix::String = "")
  local equationVectors = collect(Iterators.partition(equations, 50))
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
function decomposeEquationsInline(equations, parameterAssignments)
  local equationVectors = collect(Iterators.partition(equations, 15))
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
function decomposeStartEquationsInline(equations; functionSuffix = "")
  local equationVectors = collect(Iterators.partition(equations, 25))
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

function decomposeParametersDeclaration(parVariablesSym; chunkSize = 100)
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
function decomposeParameterEquationsInline(parameterEquations; chunkSize = 50)
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
  Check if a DAE.VAR has an array type.
  The array type info is in the componentRef's identType field, not in v.ty.
"""
function isArrayType(v::DAE.VAR)::Bool
  #= Get the type from the component reference, which contains the full type including array dimensions =#
  local crefType = @match v.componentRef begin
    DAE.CREF_IDENT(_, identType, _) => identType
    DAE.CREF_QUAL(_, identType, _, _) => identType
    _ => v.ty
  end
  @match crefType begin
    DAE.T_ARRAY(__) => true
    _ => false
  end
end

"""
  Check if any input or output of a function is an array type.
"""
function hasArrayParameters(f::SimulationCode.ModelicaFunction)::Bool
  for v in f.inputs
    if isArrayType(v)
      return true
    end
  end
  for v in f.outputs
    if isArrayType(v)
      return true
    end
  end
  return false
end

"""
  Extract dimensions from a DAE.VAR as a tuple expression.
  Returns a tuple like (3,) for a 1D array of size 3, or (3, 3) for a 3x3 matrix.
  Gets the type from componentRef.identType which contains the full array type.
"""
function extractArrayDimsFromVar(v::DAE.VAR)::Expr
  #= Get the type from the component reference =#
  local ty = @match v.componentRef begin
    DAE.CREF_IDENT(_, identType, _) => identType
    DAE.CREF_QUAL(_, identType, _, _) => identType
    _ => v.ty
  end
  @match ty begin
    DAE.T_ARRAY(_, dims) => begin
      local dimExprs = []
      for d in dims
        @match d begin
          DAE.DIM_INTEGER(n) => push!(dimExprs, n)
          DAE.DIM_UNKNOWN(__) => push!(dimExprs, :n)  #= Unknown dimension, use symbolic =#
          DAE.DIM_EXP(__) => push!(dimExprs, :n)  #= Expression dimension, use symbolic =#
          _ => push!(dimExprs, :n)
        end
      end
      if length(dimExprs) == 1
        :(($(dimExprs[1]),))
      else
        Expr(:tuple, dimExprs...)
      end
    end
    _ => :()
  end
end

"""
  Generate @register_array_symbolic expression for a function with array parameters.
"""
function generateArrayRegisterExpr(f::SimulationCode.ModelicaFunction, funcArgGen::Function)::Expr
  local normalizedName = replace(f.name, "." => "_")
  local sb = Symbol(normalizedName)

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

function collectCalledFunctionNames!(names::Set{String}, @nospecialize(exp::DAE.Exp))
  @match exp begin
    DAE.CALL(path = path, expLst = explst) => begin
      push!(names, string(path))
      for arg in explst
        collectCalledFunctionNames!(names, arg)
      end
    end
    _ => begin
      Util.traverseExpTopDown(exp,
                              (e, acc) -> begin
                                if e isa DAE.CALL
                                  push!(acc, string(e.path))
                                end
                                (e, true, acc)
                              end,
                              names)
    end
  end
  return names
end

function collectCalledFunctionNames!(names::Set{String}, eq::BDAE.RESIDUAL_EQUATION)
  collectCalledFunctionNames!(names, eq.exp)
end

function collectCalledFunctionNames!(names::Set{String}, eq::BDAE.EQUATION)
  collectCalledFunctionNames!(names, eq.lhs)
  collectCalledFunctionNames!(names, eq.rhs)
end

function collectCalledFunctionNames!(names::Set{String}, eq::BDAE.ARRAY_EQUATION)
  collectCalledFunctionNames!(names, eq.left)
  collectCalledFunctionNames!(names, eq.right)
end

function collectCalledFunctionNames!(names::Set{String}, eq::BDAE.SOLVED_EQUATION)
  collectCalledFunctionNames!(names, eq.exp)
end

function collectCalledFunctionNames!(names::Set{String}, eq::BDAE.COMPLEX_EQUATION)
  collectCalledFunctionNames!(names, eq.left)
  collectCalledFunctionNames!(names, eq.right)
end

function collectCalledFunctionNames!(names::Set{String}, stmt::BDAE.ASSIGN)
  collectCalledFunctionNames!(names, stmt.left)
  collectCalledFunctionNames!(names, stmt.right)
end

function collectCalledFunctionNames!(names::Set{String}, stmt::BDAE.REINIT)
  collectCalledFunctionNames!(names, stmt.stateVar)
  collectCalledFunctionNames!(names, stmt.value)
end

function collectCalledFunctionNames!(names::Set{String}, stmt::BDAE.NORETCALL)
  collectCalledFunctionNames!(names, stmt.exp)
end

"""
  Fallback for equation/statement kinds we do not specifically handle (ALGORITHM, BRANCH, etc.).
  Silently ignore — we only want to skip calls that the collector knows how to look inside.
"""
function collectCalledFunctionNames!(names::Set{String}, ::Any)
  return names
end

function collectCalledFunctionNames!(names::Set{String}, simCode::SimulationCode.SIM_CODE)
  for eq in simCode.residualEquations
    collectCalledFunctionNames!(names, eq)
  end
  for eq in simCode.initialEquations
    #= Dispatch over every equation kind: ARRAY, SOLVED, COMPLEX as well as
       RESIDUAL and EQUATION. Narrower filters silently dropped calls. =#
    collectCalledFunctionNames!(names, eq)
  end
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCalledFunctionNames!(names, branch.condition)
      for eq in branch.residualEquations
        collectCalledFunctionNames!(names, eq)
      end
    end
  end
  for whenEq in simCode.whenEquations
    collectCalledFunctionNames!(names, whenEq.whenEquation.condition)
    for stmt in whenEq.whenEquation.whenStmtLst
      #= Dispatch over every whenStmt kind: ASSIGN, REINIT, NORETCALL. =#
      collectCalledFunctionNames!(names, stmt)
    end
  end
  return names
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
      local normalizedName = replace(f.name, "." => "_")
      local sb = Symbol(normalizedName)
      local args = funcArgGen(f.inputs)
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
function generateExternalRuntimeImport(simCode)::Expr
  :(import OMRuntimeExternalC)
end

function createWhenStatementsMTK(whenStatements::List, simCode::SimulationCode.SIM_CODE; varPrefix = "", varSuffix = "")::Vector{Expr}
  local res::Array{Expr} = []
  @debug "Calling createWhenStatements with: $whenStatements"
  for wStmt in  whenStatements
    @match wStmt begin
      BDAE.ASSIGN(__) => begin
        local leftStr = SimulationCode.string(wStmt.left)
        (index, var) = simCode.stringToSimVarHT[leftStr]
        push!(res, quote
                idx = lookuptableStates[Symbol($(string(var.name)))]
                integrator.u[idx] = $(expToJuliaExpMTK(wStmt.right,
                                                       simCode; varPrefix = varPrefix, varSuffix = varSuffix))
              end)
      end
      #= Handles reinit =#
      BDAE.REINIT(__) => begin
        (index, var) = simCode.stringToSimVarHT[SimulationCode.string(wStmt.stateVar)]
        #=
        Note:
        The idea is that we use the variable to represent an index in the integrator to later be able to query the index via the symbol.
        =#
        push!(res, quote
                idx = lookuptableStates[Symbol($(string(var.name)))]
                integrator.u[idx] = $(expToJuliaExpMTK(wStmt.value,
                                                                                                  simCode; varPrefix = varPrefix, varSuffix = varSuffix))
              end)
      end
      #= Modelica terminate("msg") — request clean simulation stop via the
         SciML integrator, logging the user-supplied message. Surfaces on
         MSL Mechanics.MultiBody.Examples.Systems.RobotR3.{oneAxis,fullRobot}
         where PathPlanning calls terminate() at end of motion profile. =#
      BDAE.TERMINATE(__) => begin
        local msgExpr = expToJuliaExpMTK(wStmt.message, simCode;
                                          varPrefix = varPrefix, varSuffix = varSuffix)
        push!(res, quote
                @info "Modelica terminate() reached" message=$(msgExpr)
                OMBackend.DifferentialEquations.terminate!(integrator)
              end)
      end
      BDAE.NORETCALL(__) => begin
        local callExpr = expToJuliaExpMTK(wStmt.exp, simCode;
                                           varPrefix = varPrefix, varSuffix = varSuffix)
        push!(res, quote
                $(callExpr)
              end)
      end
      BDAE.ASSERT(__) => begin
        local condExpr = expToJuliaExpMTK(wStmt.condition, simCode;
                                           varPrefix = varPrefix, varSuffix = varSuffix)
        local msgExpr = expToJuliaExpMTK(wStmt.message, simCode;
                                          varPrefix = varPrefix, varSuffix = varSuffix)
        push!(res, quote
                if !($(condExpr))
                  @warn "Modelica assert()" message=$(msgExpr)
                end
              end)
      end
      _ => ErrorException("$whenStatements in @__FUNCTION__ not supported")
    end
  end
  return res
end
