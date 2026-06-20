#= /*
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
    (BDAE.PARAM(__) || BDAE.CONST(__), DAE.T_COMPLEX(__)) => begin
      SimulationCode.DATA_STRUCTURE(SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    #= Backend constants must be emitted as module-level bindings because generated
       parameter/default expressions may reference them directly by name. =#
    (BDAE.PARAM(__), DAE.T_REAL(__) || DAE.T_BOOL(__) || DAE.T_INTEGER(__)) => begin
      SimulationCode.PARAMETER(SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    (BDAE.CONST(__), DAE.T_REAL(__) || DAE.T_BOOL(__) || DAE.T_INTEGER(__)) => begin
      SimulationCode.DATA_STRUCTURE(SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    #= String parameters - used for labels, names etc. =#
    (BDAE.PARAM(__) || BDAE.CONST(__), DAE.T_STRING(__)) => begin
      SimulationCode.STRING(SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    #= Enumeration parameters =#
    (BDAE.PARAM(__), DAE.T_ENUMERATION(__)) => begin
      SimulationCode.PARAMETER(SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    (BDAE.CONST(__), DAE.T_ENUMERATION(__)) => begin
      SimulationCode.DATA_STRUCTURE(SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    (_, DAE.T_INTEGER(__)) => begin
      SimulationCode.DISCRETE()
    end
    #= If none of the cases above match =#
    (BDAE.DISCRETE(__), _) => begin
      SimulationCode.DISCRETE()
    end
    #= A variable of type bool. These can occur in structural changes... =#
    (BDAE.VARIABLE(__), DAE.T_BOOL(__)) => begin
      SimulationCode.DISCRETE()
    end
    (BDAE.PARAM(__) || BDAE.CONST(__), DAE.T_ARRAY(__)) => begin
      SimulationCode.ARRAY_PARAMETER(Int[d.integer for d in backendVar.varType.dims], SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    (_, DAE.T_ARRAY(__)) => begin
      SimulationCode.ARRAY(Int[d.integer for d in backendVar.varType.dims], SimulationCode._toSimBindExp(backendVar.bindExp))
    end

    (BDAE.VARIABLE(__), DAE.T_STRING(__)) => begin
      SimulationCode.STRING(SimulationCode._toSimBindExp(backendVar.bindExp))
    end
    #= Enumeration variables are discrete =#
    (BDAE.VARIABLE(__), DAE.T_ENUMERATION(__)) => begin
      SimulationCode.DISCRETE()
    end
    _ => begin
      @error("\n Variable: $(backendVar.varName) \n Category: $(typeof(backendVar.varKind)).\n Type: $(typeof(backendVar.varType)) of backend variable not handled.\n")
      throw("Type: $(typeof(backendVar.varType)) not handled.")
    end
  end
end

"""
`BDAE_identifierToVarString(backendVar::BDAE.VAR)`
Converts a `BDAE.VAR` to a simcode string, for use in variable names.
The `.` separator is replaced with `OMBackend.COMPONENT_SEPARATOR`.
Used to name variables used by the hts in the simulation code.
"""
function BDAE_identifierToVarString(backendVar::BDAE.VAR)
  return OMBackend.canonicalName(backendVar.varName)
end


function DAE_identifierToString(daeID::DAE.CREF)
  return DAE_identifierToString(daeID.componentRef)
end

function DAE_identifierToString(s::String)
  return OMBackend.canonicalName(s)
end

function DAE_identifierToString(u::DAE.UNARY)
  return DAE_identifierToString(u.exp)
end

#= TODO: nested/second-order derivatives (`der(der(x))`) arrive here as a
   `DAE.CALL("der", ...)` and are unsupported. Modelica permits nth-order
   derivatives; they should be reduced to auxiliary first-order states upstream
   (order lowering) before reaching SimCode rather than erroring here. =#
function DAE_identifierToString(exp)
  error("DAE_identifierToString: unsupported argument of type $(typeof(exp)) with value $exp. Expected DAE.CREF, DAE.ComponentRef, or String.")
end

"""
  Converts a DAE.ComponentRef to a string representation for use in the SimulationCode during code generation.
  This function handles different cases for arrays, complex types etc.
"""
function DAE_identifierToString(daeID::DAE.ComponentRef)
  return OMBackend.canonicalName(daeID)
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
  #=  Fetch the main equation system along with all subsystems =#
  @match [equationSystem, auxEquationSystems...] = equationSystems
  #= Fetch the different components of the model.=#
  local allSharedVars::Vector{BDAE.VAR} = getSharedVariablesLocalsAndGlobals(shared)
  local allBackendVars = vcat(equationSystem.orderedVars, allSharedVars)
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
   structuralTransitions::Vector{BDAE.Equation},
   initialWhenEqs::Vector{BDAE.INITIAL_WHEN_EQUATION}) = allocateAndCollectSimulationEquations(equations,
                                                                                               equationSystem.name,
                                                                                               addDummyState)
  #= Combine two sources of init-time algorithm bodies:
     (1) BDAE.INITIAL_WHEN_EQUATION nodes synthesized from `algorithm when initial()`
         clauses in BDAECreate.
     (2) Any BDAE.WHEN_EQUATION whose condition is bare `initial()` (rarer equation
         form). Both feed into the same INITIAL_ALGORITHM list per Modelica spec
         §8/§11. =#
  local initialAlgorithms = INITIAL_ALGORITHM[]
  for iweq in initialWhenEqs
    local daeStmts = get(Backend.BDAECreate._INIT_ALG_DAE_STMTS, iweq, DAE.Statement[])
    push!(initialAlgorithms, INITIAL_ALGORITHM(collect(iweq.whenEquation.whenStmtLst), daeStmts))
  end
  local (whenEqsKept, extractedInitAlgs) = extractInitialWhenAlgorithms(whenEqs)
  whenEqs = whenEqsKept
  append!(initialAlgorithms, extractedInitAlgs)
  #= Inline parameter literals into init bodies NOW, while the HT still has the
     scalarized array-parameter entries. Later passes (const-prop, alias-elim,
     output-only elim) drop those entries because they have no consumer in the
     equation graph, so codegen running afterwards has nothing to look up. =#
  initialAlgorithms = inlineParamsInInitialAlgorithms(initialAlgorithms, stringToSimVarHT)
  #=
    Gather all irreducible variables.
    NB: Should also include variables affected somehow with by a structural change.
  =#
  local irreducibleVars::Vector{String} = vcat(occVars,
                                                getIrreducibleVars(ifEqs,
                                                                    whenEqs,
                                                                    allBackendVars,
                                                                    stringToSimVarHT))
  (resEqs, irreducibleVars) = handleZimmerThetaConstant(resEqs, irreducibleVars, stringToSimVarHT)
  #= ...DOCC Handling... =#
  if ! isempty(shared.DOCC_equations)
    append!(structuralTransitions, shared.DOCC_equations)
  end
  #=  Convert the structural transitions to the simcode representation. =#
  local simCodeStructuralTransitions = createSimCodeStructuralTransitions(structuralTransitions)
  #= Sorting/Matching for the set of residual equations (This is used for the start conditions) =#
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
  local sharedVariables::Vector{String} = if !isempty(auxEquationSystems)
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
    local subSys = transformToSimCode([auxSys], shared; mode = mode)
    push!(structuralSubModels, subSys)
  end
  if !isempty(auxEquationSystems)
    for v in allBackendVars
      push!(topVars, string(v.varName))
    end
  end
  #= Construct SIM_CODE =#
  #= Boundary: BDAE equation types → SimCode-native types for the field
     types that have already migrated. =#
  local simResEqs = SimulationCode.RESIDUAL_EQUATION[SimulationCode.toSim(r) for r in resEqs]
  local simWhenEqs = SimulationCode.WHEN_EQUATION[SimulationCode.toSim(w) for w in whenEqs]
  local simInitialEqs = SimulationCode.Equation[SimulationCode.toSim(e) for e in equationSystem.initialEqs]
  local simSharedEqs = if !isempty(auxEquationSystems)
    SimulationCode.Equation[SimulationCode.toSim(e) for e in vcat(resEqs, whenEqs, ifEqs)]
  else
    SimulationCode.Equation[]
  end
  SimulationCode.SIM_CODE(equationSystem.name,
                          stringToSimVarHT,
                          simResEqs,
                          simInitialEqs,
                          simWhenEqs,
                          simCodeIfEquations,
                          isSingular,
                          matchOrder,
                          digraph,
                          stronglyConnectedComponents,
                          simCodeStructuralTransitions,
                          structuralSubModels,
                          sharedVariables,
                          topVars,
                          simSharedEqs,
                          initialState,
                          shared.metaModel,
                          shared.flatModel,
                          irreducibleVars,
                          ModelicaFunction[],
                          #= Specify if external runtime should be used =# false,
                          SimulationCode.RESIDUAL_EQUATION[],
                          String[],
                          AliasEntry[],
                          nothing,
                          initialAlgorithms,
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
      BDAE.STRUCTURAL_TRANSITION(__) =>
        SimulationCode.EXPLICIT_STRUCTURAL_TRANSITION(st.fromState,
                                                      st.toState,
                                                      st.transitionCondition)
      BDAE.STRUCTURAL_WHEN_EQUATION(__) =>
        SimulationCode.IMPLICIT_STRUCTURAL_TRANSITION(st.size,
                                                      SimulationCode.toWhenStmts(st.whenEquation),
                                                      st.source,
                                                      SimulationCode.toEqAttr(st.attr))
      BDAE.STRUCTURAL_IF_EQUATION(__) =>
        SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION(st.ifEquation)
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
    #= Enumerate the branches of the if equation.
       Flatten any `else if` chain that the BDAE kept in nested form;
       otherwise the typed-vector conversion at the ELSE_BRANCH step
       below would MethodError on the inner IF_EQUATION. =#
    local BDAE_ifEquation = flattenNestedThenBranchIfs(flattenNestedElseIfChain(ifEquations[i]))
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
      #= Merge the given residual equations with the equations of this particular branch.
         The listArray result is `Vector{BDAE.Equation}` (the supertype) when
         the branch contains both residuals and a sibling/nested IF_EQUATION
         (legal Modelica: `if cond then r=a+b; if x then ...; end if; end if;`).
         A direct `::Vector{BDAE.RESIDUAL_EQUATION}` cast would `MethodError:
         Cannot convert IF_EQUATION to RESIDUAL_EQUATION` on those models —
         see Modelica.Electrical.Analog.Examples.{ControlledSwitchWithArc,
         OpAmps.*, …}. Filter to residuals here and log dropped variants so
         the model at least translates; a follow-up pass should hoist the
         skipped IF_EQUATIONs with conjoined conditions for full fidelity. =#
      local rawBranchEqs = listArray(listGet(BDAE_ifEquation.eqnstrue, conditionIdx))
      local branchEquations = SimulationCode.RESIDUAL_EQUATION[]
      local branchEquationsBDAE = BDAE.RESIDUAL_EQUATION[]
      for eq in rawBranchEqs
        if eq isa BDAE.RESIDUAL_EQUATION
          push!(branchEquations, SimulationCode.toSim(eq))
          push!(branchEquationsBDAE, eq)
        else
          @warn "Skipping unsupported $(typeof(eq).name.name) inside if-branch (condition $(conditionIdx)); model may translate but lose this equation's effect"
        end
      end
      #= The variable-mapping graph builder is on the BDAE side (pre-boundary),
         so still wants the BDAE residual list. BRANCH stores the SimCode-side. =#
      local equations = vcat(resEqs, branchEquationsBDAE)
      target = conditionIdx + 1
      identifier = conditionIdx
      local eqVariableMapping = createEquationVariableBidirectionGraph(equations, otherIfEqs, whenEqs, allBackendVars, stringToSimVarHT)
      #= Match and get the strongly connected components =#
      local numberOfVariablesInMapping = length(eqVariableMapping.keys)
      (isSingular, matchOrder, digraph, stronglyConnectedComponents) =
        matchAndCheckStronglyConnectedComponents(eqVariableMapping, numberOfVariablesInMapping, stringToSimVarHT)
      #= Add the branch to the collection. =#
      branch = BRANCH(toSimExp(condition),
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
      if all preceding branches failed to evaluate to true.
    =#
    #= Check if we have an else if not we are done.=#
    if listEmpty(BDAE_ifEquation.eqnsfalse)
      break
    end
    condition = SCONST("ELSE_BRANCH")
    #= Same defensive filtering as the eqnstrue path above. =#
    local rawElseBranch = listArray(BDAE_ifEquation.eqnsfalse)
    branchEquations = SimulationCode.RESIDUAL_EQUATION[]
    local elseBranchEquationsBDAE = BDAE.RESIDUAL_EQUATION[]
    for eq in rawElseBranch
      if eq isa BDAE.RESIDUAL_EQUATION
        push!(branchEquations, SimulationCode.toSim(eq))
        push!(elseBranchEquationsBDAE, eq)
      else
        @warn "Skipping unsupported $(typeof(eq).name.name) inside if-equation else-branch; model may translate but lose this equation's effect"
      end
    end
    #= variable-mapping builder uses BDAE side; BRANCH stores SimCode side. =#
    equations = vcat(resEqs, elseBranchEquationsBDAE)
    lastConditionIdx += 1
    target = lastConditionIdx + 1
    identifier = ELSE_BRANCH #= Indicate else =#
    local eqVariableMapping = createEquationVariableBidirectionGraph(equations, otherIfEqs, whenEqs, allBackendVars, stringToSimVarHT)
    local numberOfVariablesInMapping = length(eqVariableMapping.keys)
    (isSingular, matchOrder, digraph, stronglyConnectedComponents) =
      matchAndCheckStronglyConnectedComponents(eqVariableMapping, numberOfVariablesInMapping, stringToSimVarHT)
    branch = BRANCH(toSimExp(condition),
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
  Flatten a nested `if-else-if` chain expressed as an IF_EQUATION whose
  `eqnsfalse` is itself an IF_EQUATION. Modelica treats `elseif` and
  `else if` as equivalent at the source level, but some BDAE construction
  paths produce the nested form, which `constructSimCodeIFEquations` does
  not handle (its typed-vector conversion from `List{Equation}` to
  `Vector{RESIDUAL_EQUATION}` at the ELSE_BRANCH step fails as soon as a
  nested IF_EQUATION surfaces in `eqnsfalse`).

  Equivalent to rewriting

    if c1 then T1 else (if c2 then T2 else F2 end if) end if

  as

    if c1 then T1 elseif c2 then T2 else F2 end if

  and doing so transitively for any chain length. If `eqnsfalse` contains
  anything other than a single IF_EQUATION (e.g. residuals mixed with a
  nested if, or no nested if at all), returns the original equation
  unchanged so existing behavior is preserved.
"""
function flattenNestedElseIfChain(ifEq::BDAE.IF_EQUATION)::BDAE.IF_EQUATION
  local elseList = ifEq.eqnsfalse
  if listLength(elseList) != 1
    return ifEq
  end
  local onlyElseEq = listHead(elseList)
  if !(onlyElseEq isa BDAE.IF_EQUATION)
    return ifEq
  end
  local nested = flattenNestedElseIfChain(onlyElseEq)
  local mergedConditions = listAppend(ifEq.conditions, nested.conditions)
  local mergedEqnsTrue = listAppend(ifEq.eqnstrue, nested.eqnstrue)
  return BDAE.IF_EQUATION(mergedConditions, mergedEqnsTrue, nested.eqnsfalse,
                          ifEq.source, ifEq.attr)
end

"""
  Flatten any IF_EQUATION nested inside a THEN-branch (or the ELSE-branch
  body) of an outer IF_EQUATION by conjoining the outer condition with each
  inner condition and lifting the inner branches to the outer level.

  Example (Modelica.Electrical.Analog.Ideal.IdealizedOpAmpLimited):

    if strict then
      if homotopyType then E1 else E2 end if
    else
      if homotopyType then E3 else E4 end if
    end if

  becomes

    if strict and homotopyType        then E1
    elseif strict and not homotopyType then E2
    elseif not strict and homotopyType then E3
    else                                    E4
    end if

  Without this, `constructSimCodeIFEquations` silently drops the inner
  IF_EQUATION items (its branch-typed-vector filter only keeps
  RESIDUAL_EQUATION instances), losing one residual per OpAmp instance and
  triggering ExtraVariablesSystemException downstream.
"""
function flattenNestedThenBranchIfs(ifEq::BDAE.IF_EQUATION)::BDAE.IF_EQUATION
  local conds = listArray(ifEq.conditions)
  local thens = listArray(ifEq.eqnstrue)
  local newConds = DAE.Exp[]
  local newThens = List{BDAE.Equation}[]
  for i in 1:length(conds)
    local expanded = _expandBranchEquations(collect(BDAE.Equation, listArray(thens[i])))
    if length(expanded) == 1 && expanded[1][1] === nothing
      push!(newConds, conds[i])
      push!(newThens, thens[i])
      continue
    end
    for (extra, eqsVec) in expanded
      push!(newConds, extra === nothing ? conds[i] : _conjoinConditions(conds[i], extra))
      push!(newThens, MetaModelica.list(eqsVec...))
    end
  end
  local newElse = ifEq.eqnsfalse
  local elseExpanded = _expandBranchEquations(collect(BDAE.Equation, listArray(ifEq.eqnsfalse)))
  if !(length(elseExpanded) == 1 && elseExpanded[1][1] === nothing)
    local outerElseGuard = _negateAllConditions(conds)
    #= The last expansion entry is the every-inner-else path; sequential
       branch evaluation makes its guard redundant, so it stays the ELSE. =#
    for (extra, eqsVec) in elseExpanded[1:(end - 1)]
      push!(newConds, extra === nothing ? outerElseGuard : _conjoinConditions(outerElseGuard, extra))
      push!(newThens, MetaModelica.list(eqsVec...))
    end
    newElse = MetaModelica.list(last(elseExpanded)[2]...)
  end
  return BDAE.IF_EQUATION(MetaModelica.list(newConds...),
                          MetaModelica.list(newThens...),
                          newElse,
                          ifEq.source,
                          ifEq.attr)
end

#= Expand a branch body over the IF_EQUATIONs it contains: every nested if
   multiplies the branch into one entry per inner branch, with the inner
   condition (or the negation of all inner conditions, for the inner else)
   as the entry's extra guard. Entries are (extraCond, equations) pairs;
   extraCond === nothing means the body had no nested if. The last entry is
   always the path where every nested if took its else branch. =#
function _expandBranchEquations(eqs::Vector{T})::Vector{Tuple{Union{DAE.Exp, Nothing}, Vector{T}}} where {T}
  local nestedIdx = findfirst(e -> e isa BDAE.IF_EQUATION, eqs)
  nestedIdx === nothing && return Tuple{Union{DAE.Exp, Nothing}, Vector{T}}[(nothing, eqs)]
  local inner = flattenNestedThenBranchIfs(flattenNestedElseIfChain(eqs[nestedIdx]))
  #= Splice the hoisted equations at the nested if's position: branch pairing
     downstream is positional, so the sibling branches' equation order must
     stay aligned with branches that had no nested if. =#
  local before = T[eqs[j] for j in 1:(nestedIdx - 1)]
  local after = T[eqs[j] for j in (nestedIdx + 1):length(eqs)]
  local innerConds = listArray(inner.conditions)
  local innerThens = listArray(inner.eqnstrue)
  local out = Tuple{Union{DAE.Exp, Nothing}, Vector{T}}[]
  for k in 1:length(innerConds)
    local subEqs = vcat(before, collect(T, listArray(innerThens[k])), after)
    for (extra, finalEqs) in _expandBranchEquations(subEqs)
      push!(out, (extra === nothing ? innerConds[k] : _conjoinConditions(innerConds[k], extra), finalEqs))
    end
  end
  local elseGuard = _negateAllConditions(collect(DAE.Exp, innerConds))
  local subEqsElse = vcat(before, collect(T, listArray(inner.eqnsfalse)), after)
  for (extra, finalEqs) in _expandBranchEquations(subEqsElse)
    push!(out, (extra === nothing ? elseGuard : _conjoinConditions(elseGuard, extra), finalEqs))
  end
  return out
end

function _conjoinConditions(a::DAE.Exp, b::DAE.Exp)::DAE.Exp
  DAE.LBINARY(a, DAE.AND(DAE.T_BOOL_DEFAULT), b)
end

function _negateAllConditions(conds::Vector{DAE.Exp})::DAE.Exp
  #= Build NOT(c1) AND NOT(c2) AND ... — left-fold for stability. =#
  isempty(conds) && return DAE.BCONST(true)
  local acc = DAE.LUNARY(DAE.NOT(DAE.T_BOOL_DEFAULT), conds[1])
  for k in 2:length(conds)
    acc = DAE.LBINARY(acc, DAE.AND(DAE.T_BOOL_DEFAULT),
                       DAE.LUNARY(DAE.NOT(DAE.T_BOOL_DEFAULT), conds[k]))
  end
  return acc
end

"""
Author:johti17
  This function does matching, it also checks for strongly connected components.
If the system is singular we try index reduction before proceeding.
"""
function matchAndCheckStronglyConnectedComponents(eqVariableMapping,
                                                  numberOfVariablesInMapping,
                                                  stringToSimVarHT; mode = OMBackend.MTK_MODE)::Tuple
  local isSingular::Bool
  local matchOrder::Vector
  local digraph::MetaGraphs.MetaDiGraph
  local sccs::Vector
  #=
    For MTK_MODE, catch matching failures and let MTK handle structural analysis.
    MTK has its own structural_simplify that can handle index reduction.
  =#
  try
    (isSingular, matchOrder) = GraphAlgorithms.matching(eqVariableMapping,
                                                        numberOfVariablesInMapping)
  catch e
    if mode == OMBackend.MTK_MODE
      #= Matching failed, delegating structural analysis to ModelingToolkit =#
      return (true, Int[], MetaGraphs.MetaDiGraph(), Vector{Int}[])
    else
      rethrow(e)
    end
  end
  #=
    Index reduction might resolve the issues with this system.
  =#
  if isSingular && mode == OMBackend.MTK_MODE
    digraph = GraphAlgorithms.merge(matchOrder, eqVariableMapping)
    sccs = GraphAlgorithms.stronglyConnectedComponents(digraph)
    return (isSingular, matchOrder, digraph, sccs)
  end

  if isSingular && mode == DAE_MODE
    throw("TODO: index reduction not implemented for DAE-mode")
    #= TODO do index reduction here. =#
  end
  digraph = GraphAlgorithms.merge(matchOrder, eqVariableMapping)
  sccs = GraphAlgorithms.stronglyConnectedComponents(digraph)
  return (isSingular, matchOrder, digraph, sccs)
end

"""
  Author: johti17:
  Splits a given set of equations into different types
"""
function allocateAndCollectSimulationEquations(equations::T,
                                               equationSystemName::String,
                                               shouldAddDummyEquation::Bool)::Tuple where {T}
  #= Split equations into categories in a single pass =#
  regularEquations = BDAE.RESIDUAL_EQUATION[]
  whenEquations = BDAE.WHEN_EQUATION[]
  initialWhenEquations = BDAE.INITIAL_WHEN_EQUATION[]
  ifEquations = BDAE.IF_EQUATION[]
  structuralTransitions = BDAE.Equation[]
  for eq in equations
    eqType = typeof(eq)
    if eqType === BDAE.RESIDUAL_EQUATION
      push!(regularEquations, eq)
    elseif eqType === BDAE.WHEN_EQUATION
      push!(whenEquations, eq)
    elseif eqType === BDAE.INITIAL_WHEN_EQUATION
      push!(initialWhenEquations, eq)
    elseif eqType === BDAE.IF_EQUATION
      push!(ifEquations, eq)
    elseif eqType === BDAE.STRUCTURAL_TRANSITION || eqType === BDAE.STRUCTURAL_WHEN_EQUATION
      push!(structuralTransitions, eq)
    elseif eqType === BDAE.ALGORITHM
      _algorithmToResiduals!(regularEquations, eq)
    end
  end
  if shouldAddDummyEquation
    push!(regularEquations, makeDummyResidualEquation(equationSystemName))
  end
  return (regularEquations, whenEquations, ifEquations, structuralTransitions, initialWhenEquations)
end

#= Lower the body of a non-when `BDAE.ALGORITHM` into one
   `BDAE.RESIDUAL_EQUATION` per scalar assignment. Without this step every
   such algorithm is silently dropped, so any Boolean / Integer / enum
   variable assigned in an algorithm (and not inside a when-clause) stays
   pinned at its start value. Surfaces on every
   `Modelica.Electrical.Digital.Examples` model whose Logic-enum outputs
   are driven by algorithm sections (INV3S, MUX2x1, NRXFER, ...).

   Only `STMT_ASSIGN` is lowered here; nested `STMT_IF` / `STMT_FOR` are
   not yet supported and are silently skipped. =#
function _algorithmToResiduals!(target::Vector{BDAE.RESIDUAL_EQUATION},
                                algEq::BDAE.ALGORITHM)
  local alg = algEq.alg
  alg isa DAE.ALGORITHM_STMTS || return
  for stmt in alg.statementLst
    _algorithmStmtToResidual!(target, stmt, algEq.source, algEq.attr)
  end
end

function _algorithmStmtToResidual!(target::Vector{BDAE.RESIDUAL_EQUATION},
                                   stmt, source, attr)
  if stmt isa DAE.STMT_ASSIGN
    push!(target, BDAE.RESIDUAL_EQUATION(
      DAE.BINARY(stmt.exp1, DAE.SUB(DAE.T_REAL_DEFAULT), stmt.exp),
      source, attr))
  elseif stmt isa DAE.STMT_ASSIGN_ARR
    push!(target, BDAE.RESIDUAL_EQUATION(
      DAE.BINARY(stmt.lhs, DAE.SUB(DAE.T_REAL_DEFAULT), stmt.exp),
      source, attr))
  end
  #= STMT_IF / STMT_FOR / STMT_WHILE not yet lowered. =#
end

#= True when the condition's only trigger is `initial()`: a bare `initial()` call
   or an array whose every element reduces to `initial()`. Such a when activates
   solely at initialization. =#
function _isPureInitialCondition(cond::DAE.Exp)::Bool
  @match cond begin
    DAE.CALL(Absyn.IDENT("initial"), _, _) => true
    DAE.ARRAY(_, _, lst) => (!isempty(lst) && all(_isPureInitialCondition, collect(lst)))
    _ => false
  end
end

#= True when an array-form condition carries an `initial()` trigger alongside at
   least one non-initial (runtime) trigger, i.e. `when {c1, ..., initial()}`. Per
   Modelica this is `when (c1 or ... or initial())`: the body runs at init AND on
   the runtime triggers. =#
function _hasMixedInitialCondition(cond::DAE.Exp)::Bool
  @match cond begin
    DAE.ARRAY(_, _, lst) => begin
      local elts = collect(lst)
      any(_isPureInitialCondition, elts) && !all(_isPureInitialCondition, elts)
    end
    _ => false
  end
end

#= Drop the `initial()` triggers from an array-form condition, returning the
   residual runtime trigger: the lone relation if one remains, otherwise an
   OR-chain over the survivors. =#
function _stripInitialTriggers(cond::DAE.Exp)::Union{DAE.Exp, Nothing}
  @match cond begin
    DAE.ARRAY(_, _, lst) => begin
      local rest = filter(e -> !_isPureInitialCondition(e), collect(lst))
      isempty(rest) ? nothing :
        foldl((a, b) -> DAE.LBINARY(a, DAE.OR(DAE.T_BOOL_DEFAULT), b), rest)
    end
    _ => cond
  end
end

_isTimeCrefDAE(@nospecialize(e))::Bool = @match e begin
  DAE.CREF(componentRef = cr) => string(cr) == "time"
  _ => false
end

_isPreCallDAE(@nospecialize(e))::Bool = @match e begin
  DAE.CALL(Absyn.IDENT("pre"), _, _) => true
  _ => false
end

#= True when the condition self-schedules off `time` and a `pre()` value: a
   `time <relop> pre(x)` relation, or an OR-chain containing one. The
   CombiTimeTable time event `time >= pre(nextTimeEvent)` is this shape; such
   whens reach MTK via createSelfSchedulingTimeWhenEvents. =#
function _condHasTimeAndPre(@nospecialize(cond))::Bool
  @match cond begin
    DAE.RELATION(exp1 = e1, exp2 = e2) =>
      (_isTimeCrefDAE(e1) || _isTimeCrefDAE(e2)) && (_isPreCallDAE(e1) || _isPreCallDAE(e2))
    DAE.LBINARY(exp1 = a, operator = DAE.OR(__), exp2 = b) =>
      (_condHasTimeAndPre(a) || _condHasTimeAndPre(b))
    _ => false
  end
end

"""
    extractInitialWhenAlgorithms(whenEqs) -> (runtimeWhenEqs, initialAlgorithms)

Partition a vector of `BDAE.WHEN_EQUATION`. A bare `initial()` when lowers to an
`INITIAL_ALGORITHM` only. A compound `when {c1, ..., initial()}` additionally
keeps a runtime when carrying the non-initial triggers. Others pass through.
"""
function extractInitialWhenAlgorithms(whenEqs::Vector{BDAE.WHEN_EQUATION})::Tuple{Vector{BDAE.WHEN_EQUATION}, Vector{INITIAL_ALGORITHM}}
  local kept = BDAE.WHEN_EQUATION[]
  local initialAlgs = INITIAL_ALGORITHM[]
  for weq in whenEqs
    local cond = weq.whenEquation.condition
    if _isPureInitialCondition(cond)
      local stmts = collect(weq.whenEquation.whenStmtLst)
      push!(initialAlgs, INITIAL_ALGORITHM(stmts))
    elseif _hasMixedInitialCondition(cond)
      local stmts = collect(weq.whenEquation.whenStmtLst)
      push!(initialAlgs, INITIAL_ALGORITHM(stmts))
      #= Keep the runtime arm only for a self-scheduling `time >= pre(x)` trigger,
         which has an MTK callback lowering; other compound-initial whens stay
         init-only (their prior behaviour). =#
      local runtimeCond = _stripInitialTriggers(cond)
      if runtimeCond !== nothing && _condHasTimeAndPre(runtimeCond)
        local inner = BDAE.WHEN_STMTS(runtimeCond, weq.whenEquation.whenStmtLst,
                                      weq.whenEquation.elsewhenPart)
        push!(kept, BDAE.WHEN_EQUATION(weq.size, inner, weq.source, weq.attr))
      end
    else
      push!(kept, weq)
    end
  end
  return (kept, initialAlgs)
end

#= Parse a scalarized cref key like "A[1][2]" into ("A", [1,2]).
   Returns nothing if the key has no bracketed indices. =#
function _parseScalarizedKey(key::String)::Union{Tuple{String,Vector{Int}}, Nothing}
  local openIdx = findfirst('[', key)
  openIdx === nothing && return nothing
  local base = key[1:openIdx-1]
  local rest = key[openIdx:end]
  local indices = Int[]
  while !isempty(rest)
    rest[1] == '[' || return nothing
    local closeBr = findfirst(']', rest)
    closeBr === nothing && return nothing
    local idx = tryparse(Int, rest[2:closeBr-1])
    idx === nothing && return nothing
    push!(indices, idx)
    rest = rest[closeBr+1:end]
  end
  return (base, indices)
end

#= Rebuild a DAE.ARRAY (1D or 2D) literal for a parameter name whose parent has
   been scalarized into `<name>[i]` or `<name>[i][j]` entries with literal
   bindings. Returns nothing if reconstruction is not possible. =#
function _reconstructScalarizedArrayDAE(baseName::String, ht)::Union{DAE.Exp, Nothing}
  haskey(ht, baseName) && return nothing
  local entries = Tuple{Vector{Int}, DAE.Exp}[]
  for (k, entry) in ht
    local parsed = _parseScalarizedKey(k)
    parsed === nothing && continue
    parsed[1] == baseName || continue
    local sv = last(entry)
    local be = @match sv.varKind begin
      PARAMETER(SOME(b)) => toDAEExp(b)
      _ => nothing
    end
    be === nothing && continue
    push!(entries, (parsed[2], be))
  end
  isempty(entries) && return nothing
  local nDims = length(entries[1][1])
  local realTy = DAE.T_REAL_DEFAULT
  if nDims == 1
    local nI = maximum(e -> e[1][1], entries)
    local arr = DAE.Exp[]
    for i in 1:nI
      local k = findfirst(e -> e[1] == [i], entries)
      push!(arr, k === nothing ? DAE.RCONST(0.0) : entries[k][2])
    end
    local arrTy = DAE.T_ARRAY(realTy, MetaModelica.list(DAE.DIM_INTEGER(nI)))
    return DAE.ARRAY(arrTy, false, MetaModelica.list(arr...))
  elseif nDims == 2
    local nI = maximum(e -> e[1][1], entries)
    local nJ = maximum(e -> e[1][2], entries)
    local rowTy = DAE.T_ARRAY(realTy, MetaModelica.list(DAE.DIM_INTEGER(nJ)))
    local rows = DAE.Exp[]
    for i in 1:nI
      local row = DAE.Exp[]
      for j in 1:nJ
        local k = findfirst(e -> e[1] == [i,j], entries)
        push!(row, k === nothing ? DAE.RCONST(0.0) : entries[k][2])
      end
      push!(rows, DAE.ARRAY(rowTy, false, MetaModelica.list(row...)))
    end
    local matTy = DAE.T_ARRAY(rowTy, MetaModelica.list(DAE.DIM_INTEGER(nI)))
    return DAE.ARRAY(matTy, false, MetaModelica.list(rows...))
  end
  return nothing
end

# SIM.Exp delegation: round-trip to DAE.Exp until the visitor is SIM-native.
_inlineParamsInExp(exp::Exp, ht)::Exp = toSimExp(_inlineParamsInExp(toDAEExp(exp), ht))

#= Substitute parameter CREFs with their literal bindings throughout a DAE.Exp.
   Handles PARAMETER (scalar), ARRAY_PARAMETER (direct binding), and scalarized
   array parents (reconstructed). =#
function _inlineParamsInExp(exp::DAE.Exp, ht)::DAE.Exp
  function visit(e, acc)
    if Util.isCref(e)
      local key = string(e)
      local entry = get(ht, key, nothing)
      if entry !== nothing
        local sv = last(entry)
        local be = @match sv.varKind begin
          PARAMETER(SOME(b)) => toDAEExp(b)
          ARRAY_PARAMETER(_, SOME(b)) => toDAEExp(b)
          _ => nothing
        end
        if be !== nothing
          return (be, true, acc)
        end
      else
        local recons = _reconstructScalarizedArrayDAE(key, ht)
        if recons !== nothing
          return (recons, true, acc)
        end
      end
    end
    return (e, true, acc)
  end
  #= Iterate to fixed point so that derived parameters (parameter whose bind
     expression itself references another parameter) are inlined recursively.
     A single bottom-up pass substitutes a CREF with its bind, but bottom-up
     already visited the new sub-expression's children, so inner CREFs would
     stay dangling. Loop until no further substitution occurs (or a small
     safety cap is hit, in case of an unexpected cycle in the parameter
     dependency graph). =#
  local current = exp
  for _ in 1:100
    local next = first(Util.traverseExpBottomUp(current, visit, 0))
    if next === current
      return current
    end
    current = next
  end
  return current
end

function _inlineParamsInWhenOp(wOp, ht)
  if wOp isa SimulationCode.ASSIGN
    return SimulationCode.ASSIGN(wOp.left, _inlineParamsInExp(wOp.right, ht), wOp.source)
  elseif wOp isa SimulationCode.NORETCALL
    return SimulationCode.NORETCALL(_inlineParamsInExp(wOp.exp, ht), wOp.source)
  elseif wOp isa SimulationCode.REINIT
    return SimulationCode.REINIT(wOp.stateVar, _inlineParamsInExp(wOp.value, ht), wOp.source)
  elseif wOp isa SimulationCode.ASSERT
    return SimulationCode.ASSERT(_inlineParamsInExp(wOp.condition, ht),
                                          _inlineParamsInExp(wOp.message, ht),
                                          _inlineParamsInExp(wOp.level, ht),
                                          wOp.source)
  elseif wOp isa SimulationCode.TERMINATE
    return SimulationCode.TERMINATE(_inlineParamsInExp(wOp.message, ht), wOp.source)
  end
  return @match wOp begin
    BDAE.ASSIGN(l, r, src) => BDAE.ASSIGN(l, _inlineParamsInExp(r, ht), src)
    BDAE.NORETCALL(e, src) => BDAE.NORETCALL(_inlineParamsInExp(e, ht), src)
    BDAE.REINIT(c, v, src) => BDAE.REINIT(c, _inlineParamsInExp(v, ht), src)
    BDAE.ASSERT(c, m, l, src) => BDAE.ASSERT(_inlineParamsInExp(c, ht), _inlineParamsInExp(m, ht), _inlineParamsInExp(l, ht), src)
    BDAE.TERMINATE(m, src) => BDAE.TERMINATE(_inlineParamsInExp(m, ht), src)
    _ => wOp
  end
end

"""
    inlineParamsInInitialAlgorithms(initialAlgs, ht) -> Vector{INITIAL_ALGORITHM}

For each INITIAL_ALGORITHM body, substitute parameter CREFs with their literal
bindings using the *fresh* stringToSimVarHT. Run this before any elimination
pass strips scalarized array-parameter entries; once a body carries literals,
later HT changes do not affect codegen.
"""
function inlineParamsInInitialAlgorithms(initialAlgs::Vector{INITIAL_ALGORITHM}, ht)::Vector{INITIAL_ALGORITHM}
  local result = INITIAL_ALGORITHM[]
  for ia in initialAlgs
    local newOps = [_inlineParamsInWhenOp(op, ht) for op in ia.statements]
    local newDae = [_inlineParamsInDAEStmt(s, ht) for s in ia.daeStatements]
    push!(result, INITIAL_ALGORITHM(newOps, newDae))
  end
  return result
end

#= Recursively substitute parameter CREFs with their literal bindings inside a
   `DAE.Statement`. Mirrors `_inlineParamsInWhenOp` for the parallel DAE.Statement
   representation carried by `INITIAL_ALGORITHM.daeStatements`. Compound
   statements (STMT_IF / STMT_FOR / STMT_WHILE / STMT_PARFOR) recurse into their
   bodies. Statements with no scalar expressions pass through. =#
function _inlineParamsInDAEStmt(stmt, ht)
  return @match stmt begin
    DAE.STMT_ASSIGN(ty, e1, e, src) =>
      DAE.STMT_ASSIGN(ty, e1, _inlineParamsInExp(e, ht), src)
    DAE.STMT_TUPLE_ASSIGN(ty, lhsList, e, src) =>
      DAE.STMT_TUPLE_ASSIGN(ty, lhsList, _inlineParamsInExp(e, ht), src)
    DAE.STMT_ASSIGN_ARR(ty, lhs, e, src) =>
      DAE.STMT_ASSIGN_ARR(ty, lhs, _inlineParamsInExp(e, ht), src)
    DAE.STMT_NORETCALL(e, src) =>
      DAE.STMT_NORETCALL(_inlineParamsInExp(e, ht), src)
    DAE.STMT_ASSERT(c, m, l, src) =>
      DAE.STMT_ASSERT(_inlineParamsInExp(c, ht), _inlineParamsInExp(m, ht), _inlineParamsInExp(l, ht), src)
    DAE.STMT_TERMINATE(m, src) =>
      DAE.STMT_TERMINATE(_inlineParamsInExp(m, ht), src)
    DAE.STMT_IF(cond, stmts, else_, src) =>
      DAE.STMT_IF(_inlineParamsInExp(cond, ht),
                  MetaModelica.list((_inlineParamsInDAEStmt(s, ht) for s in stmts)...),
                  _inlineParamsInDAEElse(else_, ht), src)
    DAE.STMT_FOR(ty, isArr, iter, idx, range, body, src) =>
      DAE.STMT_FOR(ty, isArr, iter, idx, _inlineParamsInExp(range, ht),
                   MetaModelica.list((_inlineParamsInDAEStmt(s, ht) for s in body)...), src)
    DAE.STMT_PARFOR(ty, isArr, iter, idx, range, body, prl, src) =>
      DAE.STMT_PARFOR(ty, isArr, iter, idx, _inlineParamsInExp(range, ht),
                      MetaModelica.list((_inlineParamsInDAEStmt(s, ht) for s in body)...), prl, src)
    DAE.STMT_WHILE(cond, body, src) =>
      DAE.STMT_WHILE(_inlineParamsInExp(cond, ht),
                     MetaModelica.list((_inlineParamsInDAEStmt(s, ht) for s in body)...), src)
    DAE.STMT_REINIT(varExp, value, src) =>
      DAE.STMT_REINIT(varExp, _inlineParamsInExp(value, ht), src)
    _ => stmt
  end
end

function _inlineParamsInDAEElse(else_, ht)
  return @match else_ begin
    DAE.ELSE(stmts) => DAE.ELSE(MetaModelica.list((_inlineParamsInDAEStmt(s, ht) for s in stmts)...))
    DAE.ELSEIF(cond, stmts, rest) =>
      DAE.ELSEIF(_inlineParamsInExp(cond, ht),
                 MetaModelica.list((_inlineParamsInDAEStmt(s, ht) for s in stmts)...),
                 _inlineParamsInDAEElse(rest, ht))
    _ => else_
  end
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
