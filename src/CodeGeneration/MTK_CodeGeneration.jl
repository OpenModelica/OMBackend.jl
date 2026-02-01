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
import OMBackend
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
    push!(structuralModes, ODE_MODE_MTK_MODEL_GENERATION(mode, mode.name, functions))
  end
  if isempty(simCode.subModels)
    local modelName = string(MODEL_NAME, "DEFAULT")
    defaultModel = ODE_MODE_MTK_MODEL_GENERATION(simCode, modelName, functions)
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
      #= Create the composite model =#
      compositeProblem = ModelingToolkit.ODEProblem(
        reducedSystem,
        finalInitialValues,
        tspan,
        pars,
        callback = callbackConditions,
      )
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

    function $(Symbol("$(MODEL_NAME)Simulate"))(tspan = (0.0, 1.0), solver=Rodas5(); kwargs...)
      $(Symbol("$(MODEL_NAME)Model_problem")) = $(Symbol("$(MODEL_NAME)Model"))(tspan)
      OMBackend.Runtime.solve($(Symbol("$(MODEL_NAME)Model_problem")), tspan, solver; kwargs...)
    end

  end
  return (MODEL_NAME, code)
end

"""
  Generates a MTK program with a model
"""
function ODE_MODE_MTK_PROGRAM_GENERATION(simCode::SimulationCode.SIM_CODE, modelName, functions)
  local MODEL_NAME = replace(modelName, "." => "__")
  #=
   Evaluate all functions s.t the symbols are available
   This needs to be done for the SymbolicUtils.jl inorder for it to recognise certain symbols.

   The generated function expressions create wrapper functions (via createModelicaFunctionWrapper)
   and store implementations in MODELICA_FUNCTION_IMPLS dictionary. The wrappers are defined
   at module load time, avoiding world-age issues when called from MTK's RuntimeGeneratedFunctions.
  =#
  for f in functions
    eval(f)
  end
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
  program = quote
    using ModelingToolkit
    using DifferentialEquations
    using OrdinaryDiffEq
    #= Add import to the external runtime if the generated code calls Modelica Functions =#
    $(if simCode.externalRuntime
        generateExternalRuntimeImport(simCode)
      end)
    $(functions...)
    $(DATA_STRUCTURE_ASSIGNMENTS...)
    $(generateRegisterCallsForCallExprs(simCode)...)
    $(model)
    function $(Symbol("$(MODEL_NAME)Simulate"))(tspan = (0.0, 1.0), solver = Rodas5();  kwargs...)
      ($(Symbol("$(MODEL_NAME)Model_problem")), callbacks, ivs, $(Symbol("$(MODEL_NAME)Model_ReducedSystem")), tspan, pars, vars, irreductable) = $(Symbol("$(MODEL_NAME)Model"))(tspan)
      sol = solve($(Symbol("$(MODEL_NAME)Model_problem")), solver; kwargs...)
      return sol
    end
  end
  #= MODEL_NAME is preprocessed with . replaced with _=#
  return MODEL_NAME, program
end

"""
  Generates a MTK model
"""
function ODE_MODE_MTK_MODEL_GENERATION(simCode::SimulationCode.SIM_CODE, modelName, functions)
  #@debug "Runnning: ODE_MODE_MTK_MODEL"
  RESET_CALLBACKS()
  #= Eval functions FIRST so they exist before @register_symbolic is called =#
  for f in functions
    eval(f)
  end
  #= Register functions with @register_symbolic immediately after they are defined =#
  #= This must happen before equations are processed, and at module level via eval =#
  local registrationCalls = generateRegisterCallsForCallExprs(simCode; funcArgGen = AlgorithmicCodeGeneration.generateIOL)
  for regCall in registrationCalls
    try
      eval(regCall)
    catch e
      #= Ignore "already has a value" errors - function was registered in a previous run =#
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
        push!(parameters, varName)
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
  end
  local performIndexReduction = simCode.isSingular
  #= Create equations for variables not in a loop + parameters and stuff=#
  local EQUATIONS = createResidualEquationsMTK(stateVariables,
                                               algebraicVariables,
                                               simCode.residualEquations,
                                               simCode::SimulationCode.SIM_CODE)
  writeEqsToFile(EQUATIONS, "equationFirstStageCodeGen.log")
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
  local discreteVariablesSym = Symbol[:($(Symbol(v))) for v in discreteVariables]
  local stateVariablesSym = Symbol[:($(Symbol(v))) for v in stateVariables]
  local occVariablesSym = Symbol[:($(Symbol(v))) for v in occVariables]
  local parVariablesSym = Symbol[Symbol(p) for p in parameters]
  #=Preprocess the component of the if equations =#
  local DISCRETE_DUMMY_EQUATIONS = [:(der($(Symbol(dv))) ~ 0) for dv in discreteVariables]
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
  local ZERO_DYNAMICS_COND_EQUATIONS = collect(Iterators.flatten([component[3] for component in IF_EQUATION_COMPONENTS]))
  #= Expand the start conditions with initial equations for the zero dynamic equations for the conditional equations =#
  local ifConditionalStartEquations = [:($(Symbol(first(v))) => $(last(v))) for v in ifConditionNameAndIV]
  #=
  In the latest variant of MTK we can not reuse the old variables for the creation of the ODEProblem later.
  Instead, we should only use the values for the unknowns we can't remove from the system.
  =#
  local irreductableSyms = Symbol[Symbol(vn) for vn in simCode.irreductableVariables]

  #= For pure ODE systems (no algebraic variables), always provide start values.
     For DAE systems (has algebraic variables), use onlyFixed=true to avoid
     conflicts between default values and algebraic constraints. =#
  local hasAlgebraicConstraints = !isempty(algebraicVariables)
  local FINAL_START_CONDTIONS_EQUATIONS = unique!(createStartConditionsEquationsMTK(
    String[vn for vn in simCode.irreductableVariables],
    String[],
    simCode; onlyFixed = hasAlgebraicConstraints))
  FINAL_START_CONDTIONS_EQUATIONS = vcat(ifConditionalStartEquations, DISCRETE_START_VALUES, FINAL_START_CONDTIONS_EQUATIONS)
  START_CONDTIONS_EQUATIONS = vcat(ifConditionalStartEquations, DISCRETE_START_VALUES, START_CONDTIONS_EQUATIONS)
  #=
    Merge the ifConditional components into the rest of the system and merge the state conditionals with the states
  =#
  stateVariablesSym = vcat(ifConditionalVariables,
                           discreteVariablesSym,
                           stateVariablesSym,
                           occVariablesSym)
  EQUATIONS = vcat(EQUATIONS,
                   DISCRETE_DUMMY_EQUATIONS,
                   ZERO_DYNAMICS_COND_EQUATIONS,
                   CONDITIONAL_EQUATIONS)
  EQUATIONS = rewriteEquations(EQUATIONS,
                               t,
                               vcat(stateVariablesSym, algebraicVariablesSym),
                               vcat(parVariablesSym, dataStructureVariablesSym),
                               simCode;
                               arrayParameterExprs = ARRAY_PARAMETERS)
  #= Reset the callback counter=#
  RESET_CALLBACKS()
  #=
    Formulate the problem as a DAE Problem.
    For this variant we keep it on its own line
    https://github.com/SciML/ModelingToolkit.jl/issues/998
  =#
  #=If our model name is separated by . replace it with __ =#
  local MODEL_NAME = replace(modelName, "." => "__")
  model = quote
    $(CALL_BACK_EQUATIONS)
    function $(Symbol(MODEL_NAME * "Model"))(tspan = (0.0, 1.0))
      ModelingToolkit.@independent_variables t
      D = ModelingToolkit.Differential(t)
      parameters = ModelingToolkit.@parameters begin
        ($(parVariablesSym...))
      end
      #= Create array parameters with proper dimensions =#
      $(ARRAY_PARAMETERS...)
      #=
        Only variables that are present in the equation system later should be a part of the variables in the MTK system.
        This means that certain algebraic variables should not be listed among the variables (These are the discrete variables).
      =#
      $(decomposeVariables(stateVariablesSym, algebraicVariablesSym))
      allVariables = []
      #= Generate variables =#
      for constructor in variableConstructors
        t = Symbolics.variable(:t, T = Real)
        vars = map(n -> (n, Symbolics.variable(n, T = Symbolics.FnType{Tuple{Real}, Real})(t)), Base.invokelatest(constructor))
        #= t is no longer needed here =#
        push!(allVariables, vars)
      end
      vars = collect(Iterators.flatten(allVariables))
      #= Evaluate the variables s.t the symbols are available =#
      for (sym, var) in vars
        eval(:($sym = $var))
      end
      #= Mark irreductable variables irreductable. Uncomment for reinitialization =#
      local irreductableSyms = $(irreductableSyms)
      for sym in irreductableSyms
        eval(:($sym = SymbolicUtils.setmetadata($sym, ModelingToolkit.VariableIrreducible, true)))
      end
      #= Transform the variable vector into a vector of Nums =#
      vars = map(x ->(last(x)), vars)
      #= Initial values for the continious system. =#
      pars = Dict($(PARAMETER_EQUATIONS...))
      startEquationComponents = []
      $(decomposeStartEquations(START_CONDTIONS_EQUATIONS))
      for constructor in startEquationConstructors
        push!(startEquationComponents, Base.invokelatest(constructor))
      end
      initialValues = collect(Iterators.flatten(startEquationComponents))
      #= Process the final initial guesses =#
      startEquationComponents = []
      $(decomposeStartEquations(FINAL_START_CONDTIONS_EQUATIONS; functionSuffix = "Final"))
      for constructor in startEquationConstructors
        push!(startEquationComponents, Base.invokelatest(constructor))
      end
      finalInitialValues = collect(Iterators.flatten(startEquationComponents))
      #= End construction of initial values =#
      equationComponents = []
      $(stripBeginBlocks(decomposeEquations(EQUATIONS, PARAMETER_ASSIGNMENTS)))
      #= Generate the Equations =#
      for constructor in equationConstructorCalls
        push!(equationComponents, Base.invokelatest(constructor))
      end
      eqs = collect(Iterators.flatten(equationComponents))
      $(IF_EQUATION_EVENT_DECLARATION)
      nonLinearSystem = $(odeSystemWithEvents(!(isempty(ifConditionalStartEquations)), modelName))
      firstOrderSystem = nonLinearSystem
      $(performStructuralSimplify(performIndexReduction)) #TODO. Causes issues for some systems... my own simplify?
      #=
        These arrays are introduced to handle the bolted on event handling using callbacks.
        The callback handling for MTK is subject of change should hybrid system be implemented for MTK.
      =#
      local eventParameters = [$(PARAMETER_RAW_ARRAY...)]
      #= Wrap discrete start values in a function and call with invokelatest to avoid world-age issues =#
      #= The variables referenced in DISCRETE_START_VALUES were created via eval above =#
      function _getDiscreteVars()
        collect(values(ModelingToolkit.OrderedDict($(DISCRETE_START_VALUES...))))
      end
      local discreteVars = Base.invokelatest(_getDiscreteVars)
      #= Merge the discrete and event parameters =#
      eventParameters = vcat(eventParameters, discreteVars)
      local aux = Vector{Any}(undef, 3)
      #= TODO init them with the initial values of them =#
      aux[1] = eventParameters
      aux[2] = Float64[]
      aux[3] = reducedSystem
      #=
       This array is used to help indexing state/discrete variables.
       It maps the index given by OMBackend to the actual index in the states.
       A[i] = <true_index>
      =#
      callbacks = $(Symbol("$(MODEL_NAME)CallbackSet"))(aux)
      problem = ModelingToolkit.ODEProblem(reducedSystem,
                                           merge(Dict(finalInitialValues), pars),
                                           tspan,
                                           #warn_initialize_determined = false,
                                           callback=callbacks)
      #= Check before commit. =#
      return (problem, callbacks, finalInitialValues, initialValues, reducedSystem, tspan, pars, vars, irreductableSyms)
    end
  end
  return model
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

if `onlyFixed` is true only start values for fixed variables with a specified start value are generated.
"""
function createStartConditionsEquationsMTK(states::Vector,
                                        algebraics::Vector,
                                        simCode::SimulationCode.SimCode; onlyFixed = false)::Vector{Expr}
  local algInit = getStartConditionsMTK(algebraics, simCode; onlyFixed = onlyFixed)
  local stateInit = getStartConditionsMTK(states, simCode; onlyFixed = onlyFixed)
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
    #= LHS will typically be a variable. Don't have to be though.. =#
    lhs = expToJuliaExpMTK(ieq.lhs, simCode)
    rhs = @match ieq.rhs begin
      DAE.CREF(__) => begin
        #= Evaluate the right hand side at this point =#
        local crefAsStr = string(ieq.rhs)
        local simCodeVar = last(simCode.stringToSimVarHT[crefAsStr])
        local res = if SimulationCode.isStateOrAlgebraic(simCodeVar)
          #= Otherwise get the start attribute =#
          expToJuliaExpMTK(ieq.rhs, simCode)
        else
          evalSimCodeParameter(simCodeVar, simCode)
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
  Given a vector of variables and the simulation code
  extracts the start attributes to generate initial conditions.

if `onlyFixed` is true only start values for fixed variables with a specified start value are generated.

"""
function getStartConditionsMTK(vars::Vector, simCode::SimulationCode.SimCode; onlyFixed = false)::Vector{Expr}
  local startExprs::Vector{Expr} = Expr[]
  local residuals = simCode.residualEquations
  local ht::Dict = simCode.stringToSimVarHT
  local warnings::String = ""
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
            if !onlyFixed
              push!(startExprs, :($(Symbol(varName)) => 0.0))
            end
            continue
          end
          (NONE(), NONE()) || (_, _) => begin
            #= No start value specified, default to 0.0 =#
            if !onlyFixed
              warnings *= "\n Assumed starting value of 0.0 for variable: " * varName * "\n"
              push!(startExprs, :($(Symbol(varName)) => 0.0))
            end
            continue
          end
        end
      end
      NONE() where {!onlyFixed} => begin
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
  if !isempty(warnings)
    @warn warnings
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
  #= The cond res is what we are going to evaluate it to. =#
  generateAffect(subIdentifier, nConditions, condRes) = begin
    local exprs = Expr[]
    local e
      e = :($(Symbol(("ifCond$(identifier)$(subIdentifier)"))) ~ $(condRes))
      push!(exprs, e)
    return exprs
  end
  """
  Generates the inverse affects. if one branch of the if equation is true.
  Set the other dummy variables to false.
  """
  generateInverseAffect(subIdentifier, nConditions, condRes) = begin
    local exprs = Expr[]
    for i in 1:nConditions
      if i != subIdentifier
        local e = :($(Symbol(("ifCond$(identifier)$(i)"))) ~ $(!condRes))
        push!(exprs, e)
      end
    end
    return exprs
  end
  local i::Int = 0
  local nBranches::Int = length(ifEq.branches)
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
        local ivCond = evalInitialCondition(mtkCond)
        local branchesWithConds::Int = nBranches - 1 #TODO DOCC - 1
        local affects::Vector{Expr} = generateAffect(i, branchesWithConds, ivCond)
        local inverseAffects::Vector{Expr} = generateInverseAffect(i, branchesWithConds, ivCond)
        local cond = :(($(mtkCond)) => [$(affects...), $(inverseAffects...)])
        #local inverseCond = :((!$(mtkCond)) => [$(inverseAffects...)])
        push!(conditions, cond)
        #push!(conditions, inverseCond)
        push!(ivConditions, ivCond)
      end
    end
  end
  #= Create the equations themselves =#
  local target = 1
  local resEqs = ifEq.branches[target].residualEquations
  local msg = "More than one equation in an if equation is not currently supported by OpenModelica.jl"
  @assert(length(resEqs) == 1, msg)
  local ifExpressions = Expr[]
  #= The number of residuals is the same for both branches. =#
  local nResEqsInTarget = length(ifEq.branches[target].residualEquations)
  for resEqIdx in 1:nResEqsInTarget
    local resEq = first(ifEq.branches[target].residualEquations)
    push!(ifExpressions,
          :($(last(deCausalize(resEq, simCode))) ~ $(generateIfExpressions(ifEq.branches,
                                                                           target,
                                                                           resEqIdx,
                                                                           identifier,
                                                                           simCode;
                                                                           subIdentifier = 1))))
  end
  #= Generate zero dynamic equations for the conditions =#
  conditionEquations = Expr[]
  conditionVariables = Symbol[]
  conditionVariableNames = Tuple{String, Bool}[]
  for i in 1:length(ivConditions)
    push!(conditionEquations, :(der($(Symbol(string("ifCond", identifier, i)))) ~ 0))
    push!(conditionVariables, :($(Symbol(string("ifCond", identifier, i)))))
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
          SOME(attr) where attr.start != nothing => begin
            @assert !(attr.start isa DAE.CREF) "Non-numeric start attributes are not currently supported"
            @match SOME(startVal) = attr.start
            startVal
          end
          NONE() => begin
            #= Unassigned parameters are assumed to be floats... =#
            DAE.RCONST(0.0)
          end
        end
      end
      SimulationCode.STRING(__) => begin
        DAE.SCONST("tmp") #TODO handle string parameters properly.
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
 This function decomposes the continuous variables.
 This means that if the set of variables are greater than 50 a new inner function is generated.
 This is done  to not negativly impact the JIT of the system to much.
"""
function decomposeVariables(stateVariables::Vector{Symbol}, algebraicVariables::Vector{Symbol})
  local nStateVars = length(stateVariables)
  local nAlgVars = length(algebraicVariables)
  if  1 < nStateVars < 50 &&  1 < nAlgVars < 50
    expr = quote
      function generateStateVariables()
        $(Tuple([stateVariables...]))
      end
      function generateAlgebraicVariables()
        $(Tuple([algebraicVariables...]))
      end
      variableConstructors = Function[generateStateVariables, generateAlgebraicVariables]
    end
  elseif (1 < nStateVars < 50) && nAlgVars == 0
    expr = quote
      function generateStateVariables()
        $(Tuple([stateVariables...]))
      end
      variableConstructors = Function[generateStateVariables]
    end
  else
    #= Split the array in chunks of 50 for the state and algebraic variables=#
    local stateVectors = collect(Iterators.partition(stateVariables, 50))
    local algVectors = collect(Iterators.partition(algebraicVariables, 50))
    #= For each vector in stateVectors create a constructor for the variables =#
    local i = 1::Int
    local exprs = Expr[]
    constructors = quote
      variableConstructors = Function[]
    end
    push!(exprs, constructors)
    for stateVector in stateVectors
      stateConstructorExpr = quote
        function $(Symbol("generateStateVariables" * string(i)))()
          $(Tuple([stateVector...]))
        end
        push!(variableConstructors, $(Symbol("generateStateVariables" * string(i))))
      end
      push!(exprs, stateConstructorExpr)
      i += 1
    end
    local i = 1
    #= decompose the algebraic variables if needed =#
    for algVector in algVectors
      algConstructorExpr = quote
        function $(Symbol("generateAlgebraicVariables" * string(i)))()
          $(Tuple([algVector...]))
        end
        push!(variableConstructors, $(Symbol("generateAlgebraicVariables" * string(i))))
      end
      push!(exprs, algConstructorExpr)
      i += 1
    end
    #= Generate the composite expression =#
    expr = quote
      $(exprs...)
    end
  end
  return expr
end

"""
  Similar to decompose variables, however, decomposes the set of equations instead.
  This function does so by dividing the total number of equations into separate blocks with 50 equations in each block.
  This function is suppose to be called after decompose variables.
  the parameter assignments contains the set of values assigned to parameters at the beginning of the simulation.
"""
function decomposeEquations(equations, parameterAssignments)
  local nStateVars = length(equations)
  local equationVectors = collect(Iterators.partition(equations, 50))
  local exprs = Expr[]
  local equationConstructorCalls = Function[]
  local constructors = quote
    #= Moved from line below. It is maybe not needed to repeat it for each block to repeat this to much.. =#
    $(parameterAssignments...)
    local equationConstructors::Vector{Function}
    local equationConstructorCalls::Vector
  end
  push!(exprs, constructors)
  local i = 0
  local functionNames = Symbol[]
  for equationVector in equationVectors
    eqv = [rewriteEq(i) for i in equationVector]
    local fName = string("generateEquations", i)
    equationConstructor = quote
      function $(Symbol(fName))()
        [$(eqv...)]
      end
    end
    push!(functionNames, Symbol(fName))
    push!(exprs, equationConstructor)
    i += 1
  end
  expr = quote
    $(exprs...)
    equationConstructorCalls = [$(functionNames...)]
  end
  return expr
end

"""
  Similar to decomposeEquations but for start equations.
  The optional argument `functionSuffix` can be used to specify a slightly different name to the function that is generated.
"""
function decomposeStartEquations(equations; functionSuffix = "")
  local nStateVars = length(equations)
  local equationVectors = collect(Iterators.partition(equations, 50))
  local exprs = Expr[]
  local constructors = quote
    startEquationConstructors = Function[]
  end
  push!(exprs, constructors)
  local i = 0
  for equationVector in equationVectors
    equationConstructorExpr = quote
      function $(Symbol(string("generateStartEquations",functionSuffix, string(i))))()
        [$(equationVector...)]
      end
      push!(startEquationConstructors, $(Symbol(string("generateStartEquations", functionSuffix, string(i)))))
    end
    push!(exprs, equationConstructorExpr)
    i += 1
  end
  expr = quote
    $(exprs...)
  end
  return expr
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

"""
  Generates quoted Symbolics.@register_symbolic calls s.t externaly defined functions are known during simulation.
  Functions with array parameters are not registered to avoid type issues; they rely on
  vectorDot and other helpers to handle symbolic arrays directly.
"""
function generateRegisterCallsForCallExprs(simCode;
                                            funcArgGen::Function = AlgorithmicCodeGeneration.generateSignatureForRegistration)
  local rFs = Expr[]
  for f in simCode.functions
    #= Skip registration for functions with array parameters =#
    #= These functions will be evaluated directly with symbolic arrays =#
    if hasArrayParameters(f)
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
        throw("ASSIGN not supported in this context.")
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
                @info idx
                integrator.u[idx] = $(expToJuliaExpMTK(wStmt.value,
                                                                                                  simCode; varPrefix = varPrefix, varSuffix = varSuffix))
              end)
      end
      _ => ErrorException("$whenStatements in @__FUNCTION__ not supported")
    end
  end
  return res
end
