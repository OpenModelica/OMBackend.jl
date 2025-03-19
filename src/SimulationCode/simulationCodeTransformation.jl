#= /*
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

"""
Converts the variable type in the backend DAE to the corresponding simulation code type.
Constants and parameters are mapped to the simcode parameter type.
Variables of type T_Complex are mapped to the DATA_Structure type.
Complex numbers are assumed to have been elaborated upon earlier in the translation process.
"""
function BDAE_VarKindToSimCodeVarKind(backendVar::BDAE.VAR)::SimulationCode.SimVarType
  varKind = @match (backendVar.varKind, backendVar.varType) begin
    #=  Standard cases, scalar real variables =#
    (BDAE.STATE(__), _) => begin
      SimulationCode.STATE()
    end
    (BDAE.VARIABLE(__), DAE.T_REAL(__)) => begin
      SimulationCode.ALG_VARIABLE(0)
    end
    #= Parameters are always parameters. =#
    (BDAE.PARAM(__) || BDAE.CONST(__), DAE.T_COMPLEX(__)) => begin
      SimulationCode.DATA_STRUCTURE(backendVar.bindExp)
    end
    (BDAE.PARAM(__) || BDAE.CONST(__), DAE.T_REAL(__) || DAE.T_BOOL(__) || DAE.T_INTEGER(__)) => begin
      SimulationCode.PARAMETER(backendVar.bindExp)
    end
    (_, DAE.T_INTEGER(__)) => begin
      SimulationCode.DISCRETE()
    end
    #= If none of the cases above match =#
    (BDAE.DISCRETE(__), _) => begin
      SimulationCode.DISCRETE()
    end
    _ => begin
      @error("Variable: $(backendVar.varName) \n Category: $(typeof(backendVar.varKind)).\n Type: $(typeof(backendVar.varType)) of backend variable not handled.\n")
    end
  end
end

"""
`BDAE_identifierToVarString(backendVar::BDAE.VAR)`
Converts a `BDAE.VAR` to a simcode string, for use in variable names.
The . separator is replaced with 3 `_`
Used to name variables used by the hts in the simulation code.
"""
function BDAE_identifierToVarString(backendVar::BDAE.VAR)
  local varName::DAE.ComponentRef = backendVar.varName
  @match varName begin
    #= An array component attached to a complex component. =#
    DAE.CREF_QUAL(ident, DAE.T_COMPLEX(__), subscriptLst, DAE.CREF_IDENT(innerIdent, DAE.T_ARRAY(__), iSubscriptLst)) => begin
      #      fail()
      #ident * "_" * innerIdent
      string(varName)
    end
    DAE.CREF_QUAL(_, DAE.T_ARRAY(ty, dim), _, _) => begin
      ident * string(cr)
    end
    DAE.CREF_IDENT(__) => string(varName)
    DAE.CREF_QUAL(__) => begin
      string(varName)
    end
    #= A variable of type array =#
    _ => @error("Type $(typeof(varName)) not handled.")
  end
end

"""
  Transform BDAE-Structure to SimulationCode.SIM_CODE
  The mode specifies what mode we should use for code generation.
  Currently the old DAE-mode is deprecated.

```
transformToSimCode(backendDAE::BDAE.BACKEND_DAE; mode)::SimulationCode.SIM_CODE
```
"""
function transformToSimCode(backendDAE::BDAE.BACKEND_DAE; mode)::SimulationCode.SIM_CODE
  transformToSimCode(backendDAE.eqs, backendDAE.shared; mode = mode)
end

function transformToSimCode(equationSystems::Vector{BDAE.EQSYSTEM}, shared; mode)::SimulationCode.SIM_CODE
  #=  Fech the main equation system along with all subsystems =#
  @match [equationSystem, auxEquationSystems...] = equationSystems
  #= Fetch the different components of the model.=#
  local allOrderedVars::Vector{BDAE.VAR} = BDAE.VAR[v  for v in equationSystem.orderedVars]
  local allSharedVars::Vector{BDAE.VAR} = getSharedVariablesLocalsAndGlobals(shared)
  local allBackendVars = vcat(allOrderedVars, allSharedVars)
  local simVars::Vector{SimulationCode.SIMVAR} = createAndCollectSimulationCodeVariables(allBackendVars, shared.flatModel)
  local occVars = map((v)-> v.name, filter((v) -> isOCCVar(v), simVars))
  #=
    Check if the model has state variables, if not introduce a dummy state
  =#
  local addDummyState = false
  if count(isState, simVars) < 0
    push!(simVars, SIMVAR(makeDummyVariableName(equationSystem.name), NONE(), STATE(), NONE()))
    local dummyBDAE_Var = BDAE.VAR(DAE.makeDummyCrefIdentOfTypeReal(makeDummyVariableName(equationSystem.name)),
                                   BDAE.STATE(),
                                   DAE.T_REAL_DEFAULT)
    push!(equationSystem.orderedVars, dummyBDAE_Var)
    push!(allBackendVars, dummyBDAE_Var)
    addDummyState = true
  end
  # Assign indices and put all variable into an hash table
  local stringToSimVarHT = createIndices(simVars)
  local equations = BDAE.Equation[eq for eq in equationSystem.orderedEqs]
  #= Split equations into three parts. Residuals whenEquations and If-equations =#
  (resEqs::Vector{BDAE.RESIDUAL_EQUATION},
   whenEqs::Vector{BDAE.WHEN_EQUATION},
   ifEqs::Vector{BDAE.IF_EQUATION},
   structuralTransitions::Vector{BDAE.Equation}) = allocateAndCollectSimulationEquations(equations,
                                                                                         equationSystem.name,
                                                                                         addDummyState)
  #=
    Gather all irreductable variables.
    NB: Should also include variables affected somehow with by a structural change.
  =#
  local irreductableVars::Vector{String} = vcat(occVars,
                                                getIrreductableVars(ifEqs,
                                                                    whenEqs,
                                                                    allBackendVars,
                                                                    stringToSimVarHT))
  (resEqs, irreductableVars) = handleZimmerThetaConstant(resEqs, irreductableVars, stringToSimVarHT)
  #= ...DOCC Handling... =#
  if ! isempty(shared.DOCC_equations)
    append!(structuralTransitions, shared.DOCC_equations)
  end
  #=  Convert the structural transitions to the simcode representation. =#
  local simCodeStructuralTransitions = createSimCodeStructuralTransitions(structuralTransitions)
  #= Sorting/Matching for the set of residual equations (This is used for the start condtions) =#
  local eqVariableMapping = createEquationVariableBidirectionGraph(resEqs,
                                                                   ifEqs,
                                                                   whenEqs,
                                                                   allBackendVars,
                                                                   stringToSimVarHT)
  local numberOfVariablesInMapping = length(eqVariableMapping.keys)
  (isSingular, matchOrder, digraph, stronglyConnectedComponents) = if isempty(auxEquationSystems)
    matchAndCheckStronglyConnectedComponents(eqVariableMapping, numberOfVariablesInMapping, stringToSimVarHT; mode = mode)
  else
    #=
    We have one or more subsystems.
    In this case matching etc is not done, but instead kept for each submodel.
    =#
    (false, Int[], MetaGraphs.MetaDiGraph(), Vector{Int}[])
  end
  #=
    The set of if equations needs to be handled in a separate way.
    Each branch might contain a separate section of variables etc that needs to be sorted and processed.
    !!It is assumed that the frontend have checked each branch for balance at this point!!
  =#
  simCodeIfEquations::Vector{IF_EQUATION} = constructSimCodeIFEquations(ifEqs,
                                                                        resEqs,
                                                                        whenEqs,
                                                                        allBackendVars,
                                                                        stringToSimVarHT)
  local structuralSubModels = SIM_CODE[]
  local initialState = initialModeInference(equationSystem)
  local topVars = String[]
  #= Use recursion to generate submodels =#
  local sharedVariables = if !isempty(auxEquationSystems)
    computeSharedVariables(auxEquationSystems, allBackendVars)
  else
    String[]
  end
  #= Elaborate on all structural submodels if they exists =#
  for auxSys in auxEquationSystems
    #= Add all top equations to the sub models =#
    for eq in vcat(resEqs, whenEqs, ifEqs)
      @match eq begin
        BDAE.RESIDUAL_EQUATION(__) => begin
          push!(auxSys.orderedEqs, eq)
        end
        BDAE.WHEN_EQUATION(__) => begin
          push!(auxSys.orderedEqs, eq)
        end
        BDAE.IF_EQUATION(__) => begin
          @error "If-equations are not supported as a top level construct in a model with static structural variability"
          fail()
        end
      end
    end
    for v in allBackendVars
      #= Add top level variables to the sub system. These variables are shared. =#
      push!(auxSys.orderedVars, v)
    end
    @info "All backend variables" allBackendVars
    local subSys = transformToSimCode([auxSys], shared; mode = mode)
    push!(structuralSubModels, subSys)
  end
  if !isempty(auxEquationSystems)
    for v in allBackendVars
      push!(topVars, string(v.varName))
    end
  end
  #= Construct SIM_CODE =#
  SimulationCode.SIM_CODE(equationSystem.name,
                          stringToSimVarHT,
                          resEqs,
                          equationSystem.initialEqs,
                          whenEqs,
                          simCodeIfEquations,
                          isSingular,
                          matchOrder,
                          digraph,
                          stronglyConnectedComponents,
                          simCodeStructuralTransitions,
                          structuralSubModels,
                          sharedVariables,
                          topVars,
                          if !isempty(auxEquationSystems) vcat(resEqs, whenEqs, ifEqs) else BDAE.Equation[] end,
                          initialState,
                          shared.metaModel,
                          shared.flatModel,
                          irreductableVars,
                          ModelicaFunction[],
                          #= Specify if external runtime should be used =# false,
                          )
end

"""
 Compute the state and algebraic variables that exists between one system and possible subsystems.
 We do so by looking at the final identifier for the given auxEquationSystems.
 Foo.x in one system is equal to bar.x in the other system.
 Furthermore, variables defined at the top level is also added to this set.
@author:johti17
"""
function computeSharedVariables(auxEquationSystems, allBackendVars::Vector{BDAE.VAR})
  local setOfVariables = Vector{String}[]
  local result = String[]
  local topLevelShared = String[]
  for auxSystem in auxEquationSystems
    namesAsIdentifiers = map(getInnerIdentOfVar,
                             filter(BDAEUtil.isStateOrVariable, auxSystem.orderedVars))
    push!(setOfVariables, namesAsIdentifiers)
    #= Add the top level variables to this set. These are always shared =#
    topLevelShared = map(getInnerIdentOfVar,
                         filter(BDAEUtil.isStateOrAlgebraicOrDiscrete, allBackendVars))
  end
  result = if !isempty(setOfVariables)
    local variableIntersection = intersect(setOfVariables...)
    vcat(variableIntersection, topLevelShared)
  else
    String[]
  end
  #= Returns the set of common variables =#
  return result
end

"""
  Fetch initial structural state from a BDAE equation system
"""
function initialModeInference(equationSystem::BDAE.EQSYSTEM)
  #= Possible very expensive check. Maybe this should be marked earlier.. =#
  for eq in equationSystem.orderedEqs
    @match eq begin
      BDAE.INITIAL_STRUCTURAL_STATE(initialState) => begin
        return initialState
      end
      _ => begin
        continue
      end
    end
  end
  return equationSystem.name
end

function createSimCodeStructuralTransitions(structuralTransitions::Vector{ST}) where {ST}
  local transitions = StructuralTransition[]
  for st in structuralTransitions
    sst = @match st begin
      BDAE.STRUCTURAL_TRANSISTION(__) => SimulationCode.EXPLICIT_STRUCTURAL_TRANSISTION(st)
      BDAE.STRUCTURAL_WHEN_EQUATION(__) => SimulationCode.IMPLICIT_STRUCTURAL_TRANSISTION(st)
      BDAE.STRUCTURAL_IF_EQUATION(__) => SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION(st)
    end
    push!(transitions, sst)
  end
  return transitions
end

"""
  Given a set of BDAE IF_EQUATIONS.
  Constructs the set of simulation-code if-equations.
  Each if equation can be seen as a small basic block graph.
Currently we merge the other residual equation with the equations of one branch.
TODO:
Possible rework the identifier scheme.
Unique identifier, static variable?
This should ideally be done earlier s.t we do not need to recreate the graph....
"""
function constructSimCodeIFEquations(ifEquations::Vector{BDAE.IF_EQUATION},
                                     resEqs::Vector{BDAE.RESIDUAL_EQUATION},
                                     whenEqs::Vector{BDAE.WHEN_EQUATION},
                                     allBackendVars::Vector,
                                     stringToSimVarHT)::Vector{IF_EQUATION}
  local simCodeIfEquations::Vector{IF_EQUATION} = IF_EQUATION[]
  for i in 1:length(ifEquations)
    #= Enumerate the branches of the if equation =#
    local BDAE_ifEquation = ifEquations[i]
    local otherIfEqs::Vector{BDAE.IF_EQUATION} = BDAE.IF_EQUATION[]
    for j in 1:length(ifEquations)
      if i != j
        push!(otherIfEqs, ifEquations[j])
      end
    end
    local conditions = BDAE_ifEquation.conditions
    local condition
    local equations
    local target
    local isSingular
    local matchOrder
    local equationGraph
    local sccs
    local branches::Vector{BRANCH} = BRANCH[]
    local lastConditionIdx = 0
    for conditionIdx in 1:length(conditions)
      condition = listGet(conditions, conditionIdx)
      #= Merge the given residual equations with the equations of this particular branch. =#
      local branchEquations::Vector{BDAE.RESIDUAL_EQUATION} = listArray(listGet(BDAE_ifEquation.eqnstrue, conditionIdx))
      local equations = vcat(resEqs, branchEquations)
      target = conditionIdx + 1
      identifier = conditionIdx
      local eqVariableMapping = createEquationVariableBidirectionGraph(equations, otherIfEqs, whenEqs, allBackendVars, stringToSimVarHT)
      #= Match and get the strongly connected components =#
      local numberOfVariablesInMapping = length(eqVariableMapping.keys)
      (isSingular, matchOrder, digraph, stronglyConnectedComponents) =
        matchAndCheckStronglyConnectedComponents(eqVariableMapping, numberOfVariablesInMapping, stringToSimVarHT)
      #= Add the branch to the collection. =#
      branch = BRANCH(condition,
                      branchEquations,
                      identifier,
                      target,
                      isSingular,
                      matchOrder,
                      digraph,
                      stronglyConnectedComponents,
                      stringToSimVarHT)
      push!(branches, branch)
      lastConditionIdx = conditionIdx
    end
    #=
     The procedure above added the code for the elseif branches.
      It is also possible that we have an else branch.
      The else branch is located in the false equations.
      The condition for the else branch to be inactive is active
      if all preceeding branches failed to evaluate to true.
    =#
    #= Check if we have an else if not we are done.=#
    if listEmpty(BDAE_ifEquation.eqnsfalse)
      break
    end
    condition = DAE.SCONST("ELSE_BRANCH")
    branchEquations::Vector{BDAE.RESIDUAL_EQUATION} = listArray(BDAE_ifEquation.eqnsfalse)
    #= Equations here consists of all residual equations of the system and the equations in the if-equation =#
    equations = vcat(resEqs, branchEquations)
    lastConditionIdx += 1
    target = lastConditionIdx + 1
    identifier = ELSE_BRANCH #= Indicate else =#
    local eqVariableMapping = createEquationVariableBidirectionGraph(equations, otherIfEqs, whenEqs, allBackendVars, stringToSimVarHT)
    local numberOfVariablesInMapping = length(eqVariableMapping.keys)
    (isSingular, matchOrder, digraph, stronglyConnectedComponents) =
      matchAndCheckStronglyConnectedComponents(eqVariableMapping, numberOfVariablesInMapping, stringToSimVarHT)
    branch = BRANCH(condition,
                    branchEquations,
                    identifier,
                    ELSE_BRANCH, #The target of the else branch is -1
                    isSingular,
                    matchOrder,
                    digraph,
                    stronglyConnectedComponents,
                    stringToSimVarHT)
    push!(branches, branch)
    ifEq = IF_EQUATION(branches)
    push!(simCodeIfEquations, ifEq)
  end
  return simCodeIfEquations
end

"""
Author:johti17
  This function does matching, it also checks for strongly connected components.
If the system is singular we try index reduction before proceeding.
"""
function matchAndCheckStronglyConnectedComponents(eqVariableMapping,
                                                  numberOfVariablesInMapping,
                                                  stringToSimVarHT; mode = OMBackend.MTK_MODE)::Tuple
  (isSingular::Bool, matchOrder::Vector) = GraphAlgorithms.matching(eqVariableMapping,
                                                                    numberOfVariablesInMapping)
  local digraph::MetaGraphs.MetaDiGraph
  local sccs::Vector
  #=
    Index reduction might resolve the issues with this system.
  =#
  if isSingular && mode == OMBackend.MTK_MODE
    digraph = GraphAlgorithms.merge(matchOrder, eqVariableMapping)
    sccs = GraphAlgorithms.stronglyConnectedComponents(digraph)
    #=
      Dumps the equation graph if the correct flag (OMBackend.PLOT_EQUATION_GRAPH)
      is defined earlier in the compilation process
    =#
    plotEquationGraph(digraph, matchOrder, stringToSimVarHT)
    return (isSingular, matchOrder, digraph, sccs)
  end

  if isSingular && mode == DAE_MODE
    throw("TODO: index reduction not implemented for DAE-mode")
    #= TODO do index reduction here. =#
  end
  digraph = GraphAlgorithms.merge(matchOrder, eqVariableMapping)
  sccs = GraphAlgorithms.stronglyConnectedComponents(digraph)
  plotEquationGraph(digraph, matchOrder, stringToSimVarHT)
  return (isSingular, matchOrder, digraph, sccs)
end

"""
  Function to print a representation of the equations post sorting
"""
function plotEquationGraph(g, matchOrder, stringToSimVarHT)
  if OMBackend.PLOT_EQUATION_GRAPH
    @info "Dumping the equation graph"
    local labels = makeLabels(g, matchOrder, stringToSimVarHT)
    GraphAlgorithms.plotEquationGraphPNG("./digraphOutput.png", g, labels)
  end
end

"""
  Author: johti17:
  Splits a given set of equations into different types
"""
function allocateAndCollectSimulationEquations(equations::T,
                                               equationSystemName::String,
                                               shouldAddDummyEquation::Bool)::Tuple where {T}
  local isIf(eq) = typeof(eq) === BDAE.IF_EQUATION
  local isWhen(eq) = typeof(eq) === BDAE.WHEN_EQUATION
  local isRe(eq) = typeof(eq) === BDAE.RESIDUAL_EQUATION
  local isStructuralTransition(eq) = typeof(eq) === BDAE.STRUCTURAL_TRANSISTION
  local isStructuralWhenEquation(eq) = typeof(eq) === BDAE.STRUCTURAL_WHEN_EQUATION
  local isStructuralWhenOrTransition(eq) = begin
    (isStructuralWhenEquation(eq) || isStructuralTransition(eq))
  end
  #= Split the representation =#
  regularEquations = filter(isRe, equations)
  whenEquations = filter(isWhen, equations)
  ifEquations = filter(isIf, equations)
  structuralTransitions = filter(isStructuralWhenOrTransition, equations)
  if shouldAddDummyEquation
    push!(regularEquations, makeDummyResidualEquation(equationSystemName))
  end
  return (regularEquations, whenEquations, ifEquations, structuralTransitions)
end

"""
Returns the shared global and local variable for the shared data in
an equation system. If no such data is present. Return two empty arrays
"""
function getSharedVariablesLocalsAndGlobals(shared::BDAE.SHARED)
  @match shared begin
    BDAE.SHARED(__) => vcat(shared.globalKnownVars, shared.localKnownVars)
    _ => []
  end
end

"""
  This function converts the set of backend variables (bDAEVariables)
  to a set of simulation code variables.
If the system contains the special occ construct we mark the variables involved in the OCC relation as state variables.
The reason being is that we do not want to optimize away these variables later.
"""
function createAndCollectSimulationCodeVariables(bDAEVariables::Vector{BDAE.VAR}, flatModel)
  @match flatModel begin
    NONE() => begin
      collectVariables(bDAEVariables)
    end
    SOME(fm) => begin
      local occVariables = collect(keys(first(getOCCGraph(fm))))
      collectVariables(bDAEVariables; occVariables = occVariables)
    end
  end
end

"""
  Collect variables from array of BDAE.Var:
  Save the name and it's kind of each variable.
  Index will be set to NONE.
"""
function collectVariables(allBackendVars::Vector{BDAE.VAR}; occVariables = String[])
  local numberOfVars::Int = length(allBackendVars)
  local simVars::Vector = Array{SimulationCode.SimVar}(undef, numberOfVars)
  for (i, backendVar) in enumerate(allBackendVars)
    #= In the backend we use string instead of component references. =#
    local simVarName::String = BDAE_identifierToVarString(backendVar)
    local simVarKind::SimulationCode.SimVarType = BDAE_VarKindToSimCodeVarKind(backendVar)
    simVarKind = if ! (isOverconstrainedConnectorVariable(simVarName, occVariables))
      simVarKind
    else
      SimulationCode.OCC_VARIABLE()
    end
    simVars[i] = SimulationCode.SIMVAR(simVarName, NONE(), simVarKind, backendVar.values)
  end
  return simVars
end
