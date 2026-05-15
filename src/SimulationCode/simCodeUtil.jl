#=
  This file contains various utility functions related to simulation code.
=#

#= SIM_CODE-level structural-variation axes. Different lowering passes guard
   on different combinations; do not collapse into one predicate. =#
hasStructuralTransitions(simCode)::Bool = !isempty(simCode.structuralTransitions)
hasSubModels(simCode)::Bool = !isempty(simCode.subModels)
hasFlatModel(simCode)::Bool = !isnothing(simCode.flatModel)
hasMetaModel(simCode)::Bool = !isnothing(simCode.metaModel)

"""
Compact structural counters for the SIM_CODE optimization pipeline.
These are intentionally cheap: they help identify which simcode pass reduced
the system before MTK sees it without walking every expression.
"""
struct SimCodeMetrics
  residualEquations::Int
  initialEquations::Int
  ifEquations::Int
  ifBranches::Int
  conditionalResidualEquations::Int
  whenEquations::Int
  variables::Int
  unknowns::Int
  parameters::Int
  aliases::Int
  eliminatedVariables::Int
end

function simCodeMetrics(simCode::SIM_CODE)::SimCodeMetrics
  local nUnknowns = 0
  local nParameters = 0
  for (_, simVar) in values(simCode.stringToSimVarHT)
    if isUnknownVarKind(simVar.varKind)
      nUnknowns += 1
    elseif isParameter(simVar)
      nParameters += 1
    end
  end
  local nIfBranches = 0
  local nConditionalResiduals = 0
  for ifEq in simCode.ifEquations
    nIfBranches += length(ifEq.branches)
    for branch in ifEq.branches
      nConditionalResiduals += length(branch.residualEquations)
    end
  end
  return SimCodeMetrics(length(simCode.residualEquations),
                        length(simCode.initialEquations),
                        length(simCode.ifEquations),
                        nIfBranches,
                        nConditionalResiduals,
                        length(simCode.whenEquations),
                        length(simCode.stringToSimVarHT),
                        nUnknowns,
                        nParameters,
                        length(simCode.aliasMap),
                        length(simCode.eliminatedVariables))
end

function _metricDelta(before::Int, after::Int)::String
  return before == after ? "$after" : "$before->$after"
end

function logSimCodePassMetrics(passName::AbstractString,
                               before::SimCodeMetrics,
                               after::SimCodeMetrics,
                               elapsed_s::Real;
                               modelName::AbstractString = "")
  if before == after
    return nothing
  end
  local label = isempty(modelName) ? passName : Base.string(modelName, ": ", passName)
  if OMBackend.BACKEND_PERFLOG[]
    @info "[SIMCODE: $label] metrics" elapsed_ms=round(1000 * elapsed_s, digits = 3) residuals=_metricDelta(before.residualEquations, after.residualEquations) initial=_metricDelta(before.initialEquations, after.initialEquations) ifEquations=_metricDelta(before.ifEquations, after.ifEquations) ifBranches=_metricDelta(before.ifBranches, after.ifBranches) conditionalResiduals=_metricDelta(before.conditionalResidualEquations, after.conditionalResidualEquations) variables=_metricDelta(before.variables, after.variables) unknowns=_metricDelta(before.unknowns, after.unknowns) parameters=_metricDelta(before.parameters, after.parameters) aliases=_metricDelta(before.aliases, after.aliases) eliminatedVariables=_metricDelta(before.eliminatedVariables, after.eliminatedVariables)
  else
    @debug "[SIMCODE: $label] metrics" elapsed_ms=round(1000 * elapsed_s, digits = 3) residuals=_metricDelta(before.residualEquations, after.residualEquations) initial=_metricDelta(before.initialEquations, after.initialEquations) ifEquations=_metricDelta(before.ifEquations, after.ifEquations) ifBranches=_metricDelta(before.ifBranches, after.ifBranches) conditionalResiduals=_metricDelta(before.conditionalResidualEquations, after.conditionalResidualEquations) variables=_metricDelta(before.variables, after.variables) unknowns=_metricDelta(before.unknowns, after.unknowns) parameters=_metricDelta(before.parameters, after.parameters) aliases=_metricDelta(before.aliases, after.aliases) eliminatedVariables=_metricDelta(before.eliminatedVariables, after.eliminatedVariables)
  end
  return nothing
end

function logSimCodePassMetrics(passName::AbstractString,
                               before::SimCodeMetrics,
                               simCode::SIM_CODE,
                               elapsed_s::Real)
  return logSimCodePassMetrics(passName, before, simCodeMetrics(simCode), elapsed_s; modelName = Base.string(simCode.name))
end

function runSimCodePass(passName::AbstractString,
                        simCode::SIM_CODE,
                        passFn::Function;
                        cleanup::Bool = true)::SIM_CODE
  local before = simCodeMetrics(simCode)
  local t0 = time()
  local afterPass = passFn(simCode)
  logSimCodePassMetrics(passName, before, afterPass, time() - t0)
  if cleanup
    afterPass = cleanupTrivialResidualEquations(afterPass; sourcePass = passName)
  end
  return afterPass
end

"""
  Returns true if simvar is either a algebraic or a state variable.
"""
function isStateOrAlgebraic(simvar::SimVar)::Bool
  return isAlgebraic(simvar) || isState(simvar)
end

"""
  Returns true if the simulation code variable is discrete.
"""
function isDiscrete(simVar::SimVar)::Bool
  res = @match simVar.varKind begin
    DISCRETE(__) => true
    _ => false
  end
end

"""
  Returns true if simvar is an algebraic variable.
"""
function isAlgebraic(simvar::SimVar)::Bool
  res = @match simvar.varKind begin
    ALG_VARIABLE(__) => true
    _ => false
  end
end

"""
  Returns true if the variable is a parameter.
"""
function isParameter(simvar::SimVar)::Bool
  res = @match simvar.varKind begin
    PARAMETER(__) => true
    _ => false
  end
end

"""
Returns true if the parameter has a binding expression.
"""
function hasBindingExp(simvar::SimVar)::Bool
  @match simvar.varKind begin
    PARAMETER(SOME(_)) => true
    _ => false
  end
end

"""
Returns true if the variable is involved in a OCC chain.
"""
function isOCCVar(simVar::SimVar)::Bool
  res = @match simVar.varKind begin
    OCC_VARIABLE(__) => true
    _ => false
  end
end

"""
  Fetches the last identifier of a variable.
That is:
getLastIdentOfVar(Foo.Bar.x) => x
"""
function getLastIdentOfVar(var)::String
  getIdentOfComponentReference(var.varName)
end


"""
 Fetches the inner identifier of a variable and converts it to a string.
That is:
getLastIdentOfVar(Foo.Bar.x) => "Bar_x"
"""
function getInnerIdentOfVar(var)::String
  res = @match var.varName begin
    DAE.CREF_IDENT(ident) => begin
      ident
    end
    DAE.CREF_QUAL(ident = ident, componentRef = componentRef) => begin
      componentRef
    end
  end
  return string(res)
end


"""
  Fetches the last ident of a component reference
"""
function getIdentOfComponentReference(cr)::String
  return begin
    @match cr begin
      DAE.CREF_QUAL(ident = ident, componentRef = componentRef) => begin
        getIdentOfComponentReference(componentRef)
      end
      DAE.CREF_IDENT(ident) => begin
        ident
      end
      DAE.CREF_ITER(ident = ident) => begin
        throw("Case not handled")
      end
    end
  end
end

"
Returns true if simvar is  an algebraic variable
"
function isState(simvar::SimVar)::Bool
  res = @match simvar.varKind begin
    STATE(__) => true
    _ => false
  end
end

"""
  Prints what equation involves which variable.
The ht maps a string to the simcode variable structure in simcode data.
"""
function dumpVariableEqMapping(mapping::OrderedDict, residualEquations, ifEquations, whenEquations, ht)::String
  local dump = IOBuffer()
  println(dump, "VARIABLES:")
  for v in keys(ht)
    println(dump, v * ":" * string(first(ht[v])))
  end
  println(dump, "EQUATION MAPPING:")
  local equations = keys(mapping)
  for e in equations
    variablesAtEq = "{"
    for v in mapping[e]
      variablesAtEq *= "$(v),"
    end
    variablesAtEq *= "}"
    println(dump, "Equation $e: involves: $(variablesAtEq): Eq $(BDAEUtil.string(e))\n")
  end
  for (i, e) in enumerate(residualEquations)
    println(dump, string("Equation " * string(i) * ":" * string(e)))
  end
  for (i, e) in enumerate(ifEquations)
    println(dump, string("IF-Equation " * string(i) * ":" * string(e)))
  end
  for (i, e) in enumerate(whenEquations)
    println(dump, string("WHEN-Equation " * string(i) * ":" * string(e)))
  end
  return String(take!(dump))
end

"""
input digraph
input variablesHT
  cref -> variable information dictionary.
output
  An array of labels for a directed graph g.
"""
function makeLabels(digraph, matchOrder, variablesHT)
  variableIndexToName::OrderedDict = makeIndexVarNameDict(matchOrder, variablesHT)
  labels = []
  for i in 1:length(matchOrder)
    try
      variableIdx = MetaGraphs.get_prop(digraph, i, :vID)
      equationIdx = matchOrder[variableIdx]
      idxToName = variableIndexToName[variableIdx]
      push!(labels, "e$(equationIdx)|$(idxToName)|index_$(i)")
    catch #= For instance the case when a vertex v does not have a prop =#
      idxToName = variableIndexToName[i]
      push!(labels, "e$(NONE)|$(idxToName)|index_$(i)")
    end
  end
  return labels
end


"""
  idx -> var-name.
  Supply matching order and a ht.
"""
function makeIndexVarNameDict(matchOrder, variablesHT)::DataStructures.OrderedDict
  local unknownVariables = filter((x) -> isVariableOrState(x[2].varKind), collect(values(variablesHT)))
  variableIndexToName::DataStructures.OrderedDict = DataStructures.OrderedDict()
  for v in unknownVariables
    variableIndexToName[v[1]] = v[2].name
  end
  return variableIndexToName
end

"""
  idx -> var-name.
  Supply matching order and a ht.
"""
function makeIndexVarNameUnorderedDict(matchOrder, variablesHT)::Dict
  local unknownVariables = filter((x) -> isVariableOrState(x[2].varKind), collect(values(variablesHT)))
  variableIndexToName::Dict = DataStructures.OrderedDict()
  for v in unknownVariables
    variableIndexToName[v[1]] = v[2].name
  end
  return variableIndexToName
end

function isVariableOrState(type::SimVarType)
  return @match type begin
    ALG_VARIABLE(__) => true
    STATE(__) => true
    _ => false
  end
end



"""
Author: John & Andreas
   This function creates and assigns indices for variables
   Thus Construct the table that maps variable name to the actual variable.
It executes the following steps:
1. Collect all variables
2. Search all states (e.g. x and y) and give them indices starting at 1 (so x=1, y=2). Then give the corresponding state derivatives (x' and y') the same indices.
3. Remaining algebraic variables will get indices starting with i+1, where i is the number of states.
4. Parameters will get own set of indices, starting at 1.
5. Discrete shares the index with the states and starts at #states + 1
6. OCC Variables also shares the indices with the states and starts at #discretes + 1
7. Data structure variables are only allowed as parameters and/or constants. They share the index with the parameters.
The index of discretes and occ is updated after the state index is calculated.
"""
function createIndices(simulationVars::Vector{SimulationCode.SIMVAR})::OrderedDict{String, Tuple{Integer, SimulationCode.SimVar}}
  local ht::OrderedDict{String, Tuple{Integer, SimulationCode.SimVar}} = OrderedDict()
  local stateCounter = 0
  local parameterCounter = 0
  local discretes = SimulationCode.SIMVAR[]
  local occVariables = SimulationCode.SIMVAR[]
  local complexVariables = SimulationCode.SIMVAR[]
  local arrayParameters = SimulationCode.SIMVAR[]
  local numberOfStates = 0
  for var in simulationVars
    @match var.varKind begin
      SimulationCode.STATE(__) => begin
        stateCounter += 1
        @assign var.index = SOME(stateCounter)
        stVar = SimulationCode.SIMVAR(var.name, var.index, SimulationCode.STATE_DERIVATIVE(var.name), var.attributes)
        push!(ht, var.name => (stateCounter, var))
        #= Adding the state derivative as well =#
        push!(ht, "der($(var.name))" => (stateCounter, stVar))
      end
      #= For Overconstrained connectors. =#
      SimulationCode.OCC_VARIABLE(__) => begin
        push!(occVariables, var)
      end
      SimulationCode.PARAMETER(__) => begin
        parameterCounter += 1
        push!(ht, var.name => (parameterCounter, var))
      end
      SimulationCode.DISCRETE(__) => begin
        push!(discretes, var)
      end
      SimulationCode.DATA_STRUCTURE(__) => begin
        parameterCounter += 1
        push!(ht, var.name => (parameterCounter, var))
      end
      SimulationCode.STRING(__) => begin
        #parameterCounter += 1
        push!(discretes, var)
      end
      SimulationCode.ARRAY_PARAMETER(__) => begin
        push!(arrayParameters, var)
      end
      _ => continue
    end
  end
  #= Assign indices to array parameters =#
  local arrayParamCounter = parameterCounter
  for var in arrayParameters
    arrayParamCounter += 1
    @assign var.index = SOME(arrayParamCounter)
    push!(ht, var.name => (arrayParamCounter, var))
  end
  local discreteCounter = stateCounter
  for var in discretes
    discreteCounter += 1
    push!(ht, var.name => (discreteCounter, var))
  end
  local occCounter = discreteCounter
  for var in occVariables
    occCounter += 1
    push!(ht, var.name => (occCounter, var))
  end
  local algIndexCounter::Int = occCounter #Change 2022-09-10
  local algSortingIdx::Int = stateCounter #This idx is used by the backend sorting algorithms
  for var in simulationVars
    @match var.varKind begin
      SimulationCode.ALG_VARIABLE(__) => begin
        algIndexCounter += 1
        algSortingIdx += 1
        @assign var.index = SOME(algIndexCounter)
        @assign var.varKind = ALG_VARIABLE(algSortingIdx)
        push!(ht, var.name => (var.index.data, var))
      end
      SimulationCode.ARRAY(__) => begin
        algIndexCounter += 1
        algSortingIdx += 1
        @assign var.index = SOME(algIndexCounter)
        push!(ht, var.name => (var.index.data, var))
      end
      _ => continue
    end
  end
  return ht
end

"""
  Given a set of residual equations, a set of if-equations and the set of all backend variables.
  This function creates a bidirectional graph between these equations and the supplied variables.
  (Note: If we need to do index reduction there might be empty equations here).
"""
function createEquationVariableBidirectionGraph(equations::Vector{BDAE.RESIDUAL_EQUATION},
                                                ifEquations::IF_EQS,
                                                whenEquations::WHEN_EQS,
                                                allBackendVars::VARS,
                                                stringToSimVarHT)::OrderedDict where{IF_EQS, WHEN_EQS, VARS}
  local eqCounter::Int = 0
  local variableEqMapping = OrderedDict()
  local unknownVariables = filter((x) -> BDAEUtil.isVariable(x.varKind), allBackendVars)
  #=TODO: The set of discrete variables are currently not in use. =#
  local discreteVariables = filter((x) -> BDAEUtil.isDiscrete(x.varKind), allBackendVars)
  local stateVariables = filter((x) -> BDAEUtil.isState(x.varKind), allBackendVars)
  local algebraicAndStateVariables = vcat(unknownVariables, stateVariables)
  local nDiscretes = length(discreteVariables)
  @debug "#stateVariables" length(stateVariables)
  @debug "#discretes" nDiscretes
  @debug "#algebraic" length(unknownVariables)
  @debug "#equations" length(equations)
  for eq in equations
    eqCounter += 1
    variablesForEq = Backend.BDAEUtil.getAllVariables(eq, algebraicAndStateVariables)
    # @debug "Variables in equation:"
    # println("Equation:", string(eq))
    # println("Variables:")
    # for v in variablesForEq
    #   println("\t", string(v))
    # end
    local indices = getIndiciesOfVariables(variablesForEq, stringToSimVarHT)
    # @debug "Indices where:"
    # for idx in indices
    #   println("\t", string(idx))
    # end
    variableEqMapping["e$(eqCounter)"] = sort(indices)
  end
  #=
   There is an additional case to consider.
   If some variables are solved by *some* branch
   (The branches are required to be balanced for ordinary if-equations)
   in an if equation it should be included in the mapping.
  =#
  for ifEq in ifEquations
    #= Select one branch. The Modelica specification requires these branches to be balanced. =#
    ifEqBranch = listArray(listGet(ifEq.eqnstrue, 1))
    for eq in ifEqBranch
      eqCounter += 1
      variablesForEq = Backend.BDAEUtil.getAllVariables(eq, algebraicAndStateVariables)
      variableEqMapping["e$(eqCounter)"] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
    end
  end
  #=
  TODO: johti17 04-13 2023:
  An additional special case occurs if an initial when equation is used.
  That is an equation on the form
  when initial()
    <equations>
  end when;
  Currently this construct breaks the compiler.
  I should investigate how to go about it.
  For now let's merge in the equations in an initial-when equation as ordinary equations. =#
  for weq in whenEquations
    @match weq begin
      BDAE.WHEN_EQUATION(_, BDAE.WHEN_STMTS(DAE.CALL(Absyn.IDENT("initial"), _, _), whenStmtLst, ewp), source, attr) => begin
        #= Go through all initial statements and add them as equations. =#
        for wstmt in weq.whenEquation.whenStmtLst
          eqCounter += 1
          variablesForEq = BDAEUtil.getAllVariables(wstmt, algebraicAndStateVariables)
          variableEqMapping["e$(eqCounter)"] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
        end
      end
      _ #=Other when equations =# => begin
        #= Assignments are added to the total #equations =#
        for wstmt in weq.whenEquation.whenStmtLst
          @match wstmt begin
            BDAE.ASSIGN(DAE.CREF(ref, DAE.T_REAL(__)), _, _) => begin
              #= Add to the total number of equation if the lhs is a real variable =#
              local refAsStr = BDAEUtil.string(ref)
              local simVar = getSimVarByName(refAsStr, stringToSimVarHT)
              #if isAlgebraic(simVar)
              eqCounter += 1
              variablesForEq = BDAEUtil.getAllVariables(wstmt, algebraicAndStateVariables)
              variableEqMapping["e$(eqCounter)"] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
              #end
            end
            _ => continue
          end
        end
      end
    end
  end
  @BACKEND_LOGGING write(OMBackend.logPath("backend/simCode", "eqMapping.log"),
                         dumpVariableEqMapping(variableEqMapping,
                                               equations,
                                               ifEquations,
                                               whenEquations,
                                               stringToSimVarHT))
  return variableEqMapping
end

"""
 Same as the other createEquationVariableBidirectionGraph however, here we assume a system that have no if-equations.
"""
function createEquationVariableBidirectionGraph(equations::RES_T,
                                                allBackendVars::VECTOR_VAR,
                                                stringToSimVarHT)::OrderedDict where{RES_T, VECTOR_VAR}
  local eqCounter::Int = 0
  local variableEqMapping = OrderedDict()
  local unknownVariables = filter((x) -> BDAEUtil.isVariable(x.varKind), allBackendVars)
  local discreteVariables = filter((x) -> BDAEUtil.isDiscrete(x.varKind), allBackendVars)
  local stateVariables = filter((x) -> BDAEUtil.isState(x.varKind), allBackendVars)
  local algebraicAndStateVariables = vcat(unknownVariables, stateVariables)
  local nDiscretes = length(discreteVariables)
  @debug "#stateVariables" length(stateVariables)
  @debug "#algebraic" length(unknownVariables)
  @debug "#equations" length(equations)
  for eq in equations
    eqCounter += 1
    variablesForEq = Backend.BDAEUtil.getAllVariables(eq, algebraicAndStateVariables)
    variableEqMapping["e$(eqCounter)"] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
  end
  return variableEqMapping
end

"""
  Given a set of variables and a dictionary that maps the component reference
  to some simulation code variable.
This function returns the indices of these variables.
*NOTE*:
  That the index of the algebraic variable is treated in a different way here.
  That is, the index of the algebraic variable is offset by the total number of discrete variables
"""
function getIndiciesOfVariables(variables,
                                stringToSimVarHT::OrderedDict{String, Tuple{Integer, SimVar}})
  local indicies = Int[]
  for v in variables
    local varName = DAE_identifierToString(v)
    local entry = get(stringToSimVarHT, varName, nothing)
    if entry === nothing
      #= TODO: Properly handle record fields and certain parameters. =#
      continue
    end
    idx, var = entry
    if isAlgebraic(var)
      #= Algebraic variables use a special idx for backend sorting purposes. =#
      push!(indicies, var.varKind.sortIdx)
    elseif isState(var)
      push!(indicies, idx)
    elseif isOCCVar(var)
      push!(indicies, idx)
    else
      continue
    end
  end
  return indicies
end

"""
  Returns the residual equation a specific variable is solved in.
  We search for this equation among the residuals in the context.
  The context should be either the top level simcode or a specific branch of some if equation.
"""
function getEquationSolvedIn(variable::V, context::C) where {V, C}
  local ht = context.stringToSimVarHT
  local variableIdx = ht[variable][1]
  local equationIdx = context.matchOrder[variableIdx]
  #= Return the equation at this specific index =#
  return context.residualEquations[equationIdx]
end

"""
  Creates a OCC graph.
  Returns the graph and the root variables.
(This function also adds info to the model)
"""
function getOCCGraph(flatModel)
  unresolvedFlatModel = OMFrontend.Frontend.FLAT_MODEL(flatModel.name,
                                                   flatModel.variables,
                                                   flatModel.unresolvedConnectEquations,
                                                   flatModel.initialEquations,
                                                   flatModel.algorithms,
                                                   flatModel.initialAlgorithms,
                                                   MetaModelica.nil,
                                                   NONE(),
                                                   flatModel.DOCC_equations,
                                                   flatModel.unresolvedConnectEquations,
                                                   flatModel.active_DOCC_Equations,
                                                   flatModel.comment)
  local name::String = unresolvedFlatModel.name
  local conns::OMFrontend.Frontend.Connections
  local conn_eql::List{OMFrontend.Frontend.Equation}
  local csets::OMFrontend.Frontend.ConnectionSets.Sets
  local csets_array::Vector{List{OMFrontend.Frontend.Connector}}
  local ctable::OMFrontend.Frontend.CardinalityTable.Table
  local broken::OMFrontend.Frontend.BrokenEdges = MetaModelica.nil
  local rootEquations::Vector{OMFrontend.Frontend.Equation} = OMFrontend.Frontend.Equation[]
  local rootReferenceVariables::Vector{Tuple} = Tuple{OMFrontend.Frontend.NFComponentRef,
                                                      OMFrontend.Frontend.NFComponentRef}[]
  (unresolvedFlatModel, conns) = OMFrontend.Frontend.collect(unresolvedFlatModel)
  (unresolvedFlatModel, conns) = OMFrontend.Frontend.elaborate(unresolvedFlatModel, conns)
  if OMFrontend.Frontend.System.getHasOverconstrainedConnectors()
    (_, broken, graph) = OMFrontend.Frontend.handleOverconstrainedConnections(unresolvedFlatModel, conns, name)
    (roots, _, broken) = OMFrontend.Frontend.findResultGraph(graph, name)
    rootEquations = OMFrontend.Frontend.findRootEquations(roots, graph,
                                                      unresolvedFlatModel.equations)
    for re in rootEquations
      push!(rootReferenceVariables,
            (re.lhs, re.rhs))
    end
  end
  #= Remove the broken edge from the set of edges =#
  @assign graph.connections = arrayList(filter((x)->(!in(x, broken)), listArray(graph.connections)))
  #= Convert the branches to regular edges =#
  local uniqueRoots = graph.uniqueRoots
  local definiteRoots = graph.definiteRoots
  local potentialRoots = graph.potentialRoots
  #= Get the roots involved in the structural change =#
  rootVariables::List{OMFrontend.Frontend.ComponentRef} = MetaModelica.list(r for r in roots)
  #= Create a graph that we can search. =#
  local connectionEdges = convertFlatEdgeToEdges(graph.connections)
  local allEdges = listAppend(connectionEdges, graph.branches)
  local searchGraph = createSearchGraph(allEdges)
  return (searchGraph, rootVariables, rootReferenceVariables)
end

"""
 Convert the component references to the backend representation and create an adjacency list representation.
"""
function createSearchGraph(allEdges)
  local edgeSet = Dict()
  local searchGraph = Dict{String, Vector{String}}()
  for edge in allEdges
    @match (e1, e2) = edge
    local s1 = OMFrontend.Frontend.toString(e1)
    local s2 = OMFrontend.Frontend.toString(e2)
    edgeSet[s1] = e1
    edgeSet[s2] = e2
  end
  for edge in keys(edgeSet)
    searchGraph[edge] = String[]
  end
  for edge in allEdges
    @match (e1, e2) = edge
    local s1 = OMFrontend.Frontend.toString(e1)
    local s2 = OMFrontend.Frontend.toString(e2)
    push!(searchGraph[s1], s2)
    push!(searchGraph[s2], s1)
  end
  return searchGraph
end

"""
  Given a list of flat edges convert them to edges.
"""
function convertFlatEdgeToEdges(connections)
  newEdges = Tuple[]
  for connection in connections
    @match connection begin
      (c1, c2, _)  => begin
        push!(newEdges, (c1, c2))
      end
    end
  end
  return arrayList(newEdges)
end

"""
 This function returns true if a backend variable is in the set of overconstrained connector variables (occVariables).

TODO: the name of the theta variable is hardcoded for now
Note that this function must be called before sorting.
"""
function isOverconstrainedConnectorVariable(simVarName::String, occVariables::Vector{String})
  #= Inefficient crap, can be done better... =#
  local isOCCVar = simVarName in occVariables
  return isOCCVar
end

"""
  Get all variables that should be marked as irreducible.
OBS:
Parameters are never added to this list.
The known irreducibles should be state variables and variables directly involved in changes that change the model structure.
"""
function getIrreductableVars(ifEquations::Vector{BDAE.IF_EQUATION},
                             whenEqs::Vector{BDAE.WHEN_EQUATION},
                             algebraicAndStateVariables::Vector{BDAE.VAR},
                             ht::OrderedDict{String, Tuple{Integer, SimulationCode.SimVar}})
  local irreductables::Vector{Any} = []
  for eq in ifEquations
    variablesForEq = Backend.BDAEUtil.getAllVariables(eq, algebraicAndStateVariables)
    push!(irreductables, variablesForEq)
  end
  #=
    Parameters should not be marked as irreducible
    Remove them from the list
  =#
  local knownIrreductables::Vector{BDAE.VAR} = filter((v) -> BDAEUtil.isState(v) , algebraicAndStateVariables)
  #@debug "Adding all states as irreducible variables" map(x->string(x.varName), knownIrreductables)
  push!(irreductables, map(x->BDAE_identifierToVarString(x), knownIrreductables))
  irreductables = collect(Iterators.flatten(irreductables))
  irreductables = filter(irv -> !(irv != "time" && isParameter(last(ht[irv]))), irreductables)
  #TODO: Fix the detection, s.t variables critical to when equations are not removed
  #for eq in whenEqs
    # variablesForEq = Backend.BDAEUtil.getAllVariables(eq, algebraicAndStateVariables)
    # push!(variablesForEq, irreductables)
  #end
  #= Add known irreducibles to the vector =#
  #push!(irreductables, map(x->x.varName, string(knownIrreductables)))
  local irreductablesAsStr = map(x -> string(x), irreductables)
  #=
  If THETA exists, treat it as an irreducible variable
  Currently, theta is a variable with "_THETA" in the variable name.
  This is subject to change
  =#
  thetaVariables = findall([endswith(x, "THETA") for x in keys(ht)])
  @assert length(thetaVariables) < 2
  if !(isempty(thetaVariables))
    #= Hardcoded for now can be fixed with annotation in the frontend =#
    push!(irreductablesAsStr, collect(keys(ht))[first(thetaVariables)])
  end
  irreductablesAsStr = filter(x -> x != "time", irreductablesAsStr)
  return irreductablesAsStr
end

"""
TODO: the name of the theta variable is hardcoded for now
Note that this function must be called before sorting.
"""
function handleZimmerThetaConstant(resEqs, irreductableVars::Vector{String}, ht)
  thetaVariables = findall([endswith(x, "THETA") for x in keys(ht)])
  if !(isempty(thetaVariables))
    #= Hardcoded for now can be fixed with annotation in the frontend =#
    thetaConstant = collect(keys(ht))[first(thetaVariables)]
    push!(irreductableVars, thetaConstant)
    tmpResEq = DAE.BINARY(
      DAE.CREF(DAE.CREF_IDENT(thetaConstant, DAE.T_REAL_DEFAULT, MetaModelica.list()), DAE.T_REAL_DEFAULT),
      DAE.SUB(DAE.T_REAL_DEFAULT),
      DAE.RCONST(1.0))
    push!(resEqs,
          BDAE.RESIDUAL_EQUATION(tmpResEq, nothing, nothing))
    (zimmerThetaIdx, simVar) = ht[thetaConstant]
    @assign simVar.varKind = ALG_VARIABLE(0)
    ht[thetaConstant] = (zimmerThetaIdx, simVar)
  end
  return(resEqs, irreductableVars)
end

function getSimVarByName(name::String, ht::AbstractDict{String, Tuple{Integer, SimVar}})
  return last(ht[name])
end

function makeDummyVariableName(equationSystemName::String; idx::Int = 1)
  return Base.string(equationSystemName, "__dummy", idx)
end

"""
  Creates a dummy residual.
  The dummy residual specifies that the derivative of a dummy variable is zero.
  0 = dx(<dummy_name><idx>)/dt - 0
"""
function makeDummyResidualEquation(equationSystemName::String, idx::Int = 1)
  local dummyName = makeDummyVariableName(equationSystemName; idx = idx)
  local crefIdent = DAE.CREF_IDENT(dummyName, DAE.T_REAL_DEFAULT, MetaModelica.list())
  local crefExpression = DAE.CREF(crefIdent, DAE.T_REAL_DEFAULT)
  return BDAE.RESIDUAL_EQUATION(
    DAE.BINARY(
      DAE.CALL(Absyn.IDENT("der"), crefExpression <| MetaModelica.list(), DAE.callAttrBuiltinReal),
      DAE.SUB(DAE.T_REAL_DEFAULT),
      DAE.RCONST(0.0)),
    DAE.T_SOURCEINFO_DEFAULT,
    BDAE.EQ_ATTR_DEFAULT_DYNAMIC,
  )
end

"""
    buildBaseNameIndex(ht::OrderedDict{String, Tuple{Integer, SimVar}})

Build a reverse index from base variable names (without subscripts) to all
subscripted full names in the hash table. For example, if the HT contains
"world_x[1]" and "world_x[2]", the result maps "world_x" => ["world_x[1]", "world_x[2]"].
This handles the ASUB case where `getAllCrefs` extracts a base CREF without subscripts.
"""
function buildBaseNameIndex(ht::OrderedDict{String, Tuple{Integer, SimVar}})::Dict{String, Vector{String}}
  local index = Dict{String, Vector{String}}()
  for (varName, _) in ht
    local bn = replace(varName, r"\[.*" => "")
    if bn != varName
      if !haskey(index, bn)
        index[bn] = String[]
      end
      push!(index[bn], varName)
    end
  end
  return index
end

"""
    collectEquationVarNames(exp::DAE.Exp,
                            ht::OrderedDict{String, Tuple{Integer, SimVar}},
                            baseNameToFullNames::Dict{String, Vector{String}})

Extract all variable names referenced by a DAE expression, using the robust
`Util.getAllCrefs` traversal (via `traverseExpTopDown`). Falls back to base-name
matching for ASUB-wrapped CREFs where subscripts are separated from the CREF.

Returns a Set{String} of variable names that exist in the HT.
"""
function collectEquationVarNames(exp::DAE.Exp,
                                 ht::OrderedDict{String, Tuple{Integer, SimVar}},
                                 baseNameToFullNames::Dict{String, Vector{String}})::Set{String}
  local crefs::List{DAE.ComponentRef} = Util.getAllCrefs(exp)
  local names = Set{String}()
  for cr in crefs
    local name = DAE_identifierToString(cr)
    if haskey(ht, name)
      push!(names, name)
    else
      #= Base name fallback: the CREF may come from inside an ASUB expression,
         missing its subscripts. Match all subscripted variants conservatively. =#
      local bn = replace(name, r"\[.*" => "")
      if bn != name && haskey(ht, bn)
        #= The CREF itself has partial subscripts; try the full name and base =#
        push!(names, bn)
      end
      local lookupKey = haskey(baseNameToFullNames, name) ? name : bn
      if haskey(baseNameToFullNames, lookupKey)
        for fullName in baseNameToFullNames[lookupKey]
          push!(names, fullName)
        end
      end
    end
  end
  return names
end

"""
    rebuildMatchOrder(simCode::SIM_CODE)

Rebuild a fresh bipartite matching from the current equations and variables.
This is needed when the original matchOrder is stale (e.g. after const-prop
and alias-elim have removed equations and variables).

Returns `(matchOrder::Vector{Int}, nameToMatchIdx::Dict{String,Int}, matchIdxToName::Dict{Int,String})`
where `matchOrder[varMatchIdx] = eqIdx` (0 = unmatched).
"""
function rebuildMatchOrder(simCode::SIM_CODE)
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  #= Collect unknown variables (those that participate in matching) =#
  local nameToMatchIdx = Dict{String, Int}()
  local matchIdxToName = Dict{Int, String}()
  local matchIdx = 0
  for (varName, (_idx, sv)) in ht
    local isUnknown = @match sv.varKind begin
      STATE(__) => true
      STATE_DERIVATIVE(__) => true
      ALG_VARIABLE(__) => true
      SimulationCode.ARRAY(__) => true
      OCC_VARIABLE(__) => true
      DISCRETE(__) => true
      _ => false
    end
    if isUnknown
      matchIdx += 1
      nameToMatchIdx[varName] = matchIdx
      matchIdxToName[matchIdx] = varName
    end
  end
  local nVars = matchIdx
  #= Build the base name index for robust CREF extraction =#
  local baseNameToFullNames = buildBaseNameIndex(ht)
  #= Build bipartite adjacency: for each equation, which variable match indices does it reference? =#
  local eqVarMapping = DataStructures.OrderedDict{String, Vector{Int}}()
  for eqI in 1:nEqs
    local refs = collectEquationVarNames(resEqs[eqI].exp, ht, baseNameToFullNames)
    local indices = Int[]
    for refName in refs
      if haskey(nameToMatchIdx, refName)
        push!(indices, nameToMatchIdx[refName])
      end
    end
    eqVarMapping["e$(eqI)"] = sort(unique(indices))
  end
  #= The matching algorithm requires a square system (n used for both eq loop
     and assign array). For over-determined systems (nVars > nEqs), pad with
     dummy empty equations so the algorithm sees a square system. The dummy
     equations will remain unmatched. For under-determined systems (nEqs > nVars),
     skip since we cannot produce a valid matching. =#
  if nEqs > nVars
    @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] under-determined system ($nEqs equations, $nVars unknowns), skipping"
    return (Int[], nameToMatchIdx, matchIdxToName)
  end
  local nMatch = nVars
  if nVars > nEqs
    for dummyI in (nEqs + 1):nVars
      eqVarMapping["e$(dummyI)"] = Int[]
    end
  end
  local matchOrder::Vector{Int}
  try
    local (_isSingular, mo) = GraphAlgorithms.matching(eqVarMapping, nMatch)
    matchOrder = mo
  catch e
    @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] matching failed, skipping DCE" exception=(e, catch_backtrace())
    return (Int[], nameToMatchIdx, matchIdxToName)
  end
  local nMatched = count(>(0), matchOrder)
  @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] $nEqs equations, $nVars unknowns, $nMatched matched"
  return (matchOrder, nameToMatchIdx, matchIdxToName)
end

"""
    identifyOutputOnlyVariables(simCode::SIM_CODE)

Identify variables and equations that do not influence the dynamic states.
Performs a backward reachability analysis from state and state-derivative equations
through the causalized equation dependency graph.

Returns `(outputOnlyVarNames::Set{String}, outputOnlyEqIndices::Set{Int})`.
Variables in the returned set are purely "output" (they can be computed from states
but do not feed back into any state derivative).
"""
#= Recursively collect all CREF variable names referenced in a DAE expression. =#
function collectCrefNames!(names::Set{String}, @nospecialize(exp))
  @match exp begin
    DAE.CREF(cr, _) => begin
      push!(names, DAE_identifierToString(cr))
    end
    DAE.BINARY(exp1 = e1, exp2 = e2) => begin
      collectCrefNames!(names, e1)
      collectCrefNames!(names, e2)
    end
    DAE.UNARY(exp = e1) => collectCrefNames!(names, e1)
    DAE.LUNARY(exp = e1) => collectCrefNames!(names, e1)
    DAE.LBINARY(exp1 = e1, exp2 = e2) => begin
      collectCrefNames!(names, e1)
      collectCrefNames!(names, e2)
    end
    DAE.CALL(expLst = args) => begin
      for arg in args
        collectCrefNames!(names, arg)
      end
    end
    DAE.IFEXP(expCond = c, expThen = t, expElse = e) => begin
      collectCrefNames!(names, c)
      collectCrefNames!(names, t)
      collectCrefNames!(names, e)
    end
    DAE.ARRAY(array = lst) => begin
      for e in lst
        collectCrefNames!(names, e)
      end
    end
    DAE.ASUB(exp = e, sub = subs) => begin
      #= When ASUB wraps a CREF with constant integer subscripts, reconstruct the
         subscripted name (e.g. "R_T[1][1]") to match the hash table key format.
         Without this, the BFS use-def chain is broken: collectCrefNames collects
         the base name "R_T" but the hash table has "R_T[1][1]". =#
      local asubHandled = false
      @match e begin
        DAE.CREF(cr, _) => begin
          local baseName = DAE_identifierToString(cr)
          local allConst = true
          local subscriptStr = ""
          for s in subs
            @match s begin
              DAE.ICONST(i) => begin subscriptStr *= string("[", i, "]") end
              _ => begin allConst = false end
            end
          end
          if allConst && !isempty(subscriptStr)
            push!(names, string(baseName, subscriptStr))
          end
          push!(names, baseName)
          asubHandled = true
        end
        _ => ()
      end
      if !asubHandled
        collectCrefNames!(names, e)
      end
      for s in subs
        collectCrefNames!(names, s)
      end
    end
    DAE.RELATION(exp1 = e1, exp2 = e2) => begin
      collectCrefNames!(names, e1)
      collectCrefNames!(names, e2)
    end
    DAE.CAST(exp = e) => collectCrefNames!(names, e)
    DAE.TSUB(exp = e) => collectCrefNames!(names, e)
    DAE.RSUB(exp = e) => collectCrefNames!(names, e)
    DAE.REDUCTION(expr = e, iterators = iters) => begin
      collectCrefNames!(names, e)
      for it in iters
        @match it begin
          DAE.REDUCTIONITER(exp = guardExp) => collectCrefNames!(names, guardExp)
          _ => ()
        end
      end
    end
    _ => ()
  end
  return nothing
end

function _hasUnknownCref(exp, ht)::Bool
  local names = Set{String}()
  collectCrefNames!(names, exp)
  for name in names
    local entry = get(ht, name, nothing)
    if entry !== nothing && isUnknownVarKind(last(entry).varKind)
      return true
    end
  end
  return false
end

function _isZeroLiteral(@nospecialize(exp))::Bool
  @match exp begin
    DAE.RCONST(v) => v == 0.0
    DAE.ICONST(v) => v == 0
    _ => false
  end
end

function _isSyntacticZeroResidual(@nospecialize(exp))::Bool
  if _isZeroLiteral(exp)
    return true
  end
  @match exp begin
    DAE.BINARY(e1, DAE.SUB(__), e2) => isequal(e1, e2)
    DAE.BINARY(e1, DAE.ADD(__), DAE.UNARY(DAE.UMINUS(__), e2)) => isequal(e1, e2)
    DAE.BINARY(DAE.UNARY(DAE.UMINUS(__), e1), DAE.ADD(__), e2) => isequal(e1, e2)
    _ => false
  end
end

function _isTrivialResidualEquation(eq::BDAE.RESIDUAL_EQUATION, simCode::SIM_CODE)::Bool
  if _hasUnknownCref(eq.exp, simCode.stringToSimVarHT)
    return false
  end
  if _isSyntacticZeroResidual(eq.exp)
    return true
  end
  local value = tryEvalNumeric(eq.exp, simCode)
  return value !== nothing && value == 0.0
end

function _isTrivialInitialEquation(@nospecialize(eq), simCode::SIM_CODE)::Bool
  if eq isa BDAE.RESIDUAL_EQUATION
    return _isTrivialResidualEquation(eq, simCode)
  elseif eq isa BDAE.EQUATION
    if _hasUnknownCref(eq.lhs, simCode.stringToSimVarHT) ||
       _hasUnknownCref(eq.rhs, simCode.stringToSimVarHT)
      return false
    end
    if isequal(eq.lhs, eq.rhs)
      return true
    end
    local lhsVal = tryEvalScalar(eq.lhs, simCode)
    local rhsVal = tryEvalScalar(eq.rhs, simCode)
    return lhsVal !== nothing && rhsVal !== nothing && lhsVal == rhsVal
  end
  return false
end

function _filterTrivialResiduals(eqs::Vector{BDAE.RESIDUAL_EQUATION},
                                 simCode::SIM_CODE)::Tuple{Vector{BDAE.RESIDUAL_EQUATION}, Int}
  local newEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newEqs, length(eqs))
  local nRemoved = 0
  for eq in eqs
    if _isTrivialResidualEquation(eq, simCode)
      nRemoved += 1
    else
      push!(newEqs, eq)
    end
  end
  return (newEqs, nRemoved)
end

function _filterTrivialInitialEquations(eqs, simCode::SIM_CODE)
  local newEqs = typeof(eqs)()
  local nRemoved = 0
  for eq in eqs
    if _isTrivialInitialEquation(eq, simCode)
      nRemoved += 1
    else
      push!(newEqs, eq)
    end
  end
  return (newEqs, nRemoved)
end

function _cleanupTrivialBranchResiduals(ifEq::IF_EQUATION,
                                        simCode::SIM_CODE)::Tuple{Union{IF_EQUATION, Nothing}, Int}
  if isempty(ifEq.branches)
    return (nothing, 0)
  end
  local nResiduals = length(first(ifEq.branches).residualEquations)
  if any(branch -> length(branch.residualEquations) != nResiduals, ifEq.branches)
    return (ifEq, 0)
  end
  local keep = trues(nResiduals)
  local nRemovedSlots = 0
  for idx in 1:nResiduals
    local allTrivial = true
    for branch in ifEq.branches
      if !_isTrivialResidualEquation(branch.residualEquations[idx], simCode)
        allTrivial = false
        break
      end
    end
    if allTrivial
      keep[idx] = false
      nRemovedSlots += 1
    end
  end
  if nRemovedSlots == 0
    return (ifEq, 0)
  end
  if nRemovedSlots == nResiduals
    return (nothing, nRemovedSlots * length(ifEq.branches))
  end
  local newBranches = BRANCH[]
  for branch in ifEq.branches
    local newResiduals = BDAE.RESIDUAL_EQUATION[branch.residualEquations[i] for i in 1:nResiduals if keep[i]]
    push!(newBranches, BRANCH(branch.condition, newResiduals,
                              branch.identifier, branch.targets, branch.isSingular,
                              branch.matchOrder, branch.equationGraph, branch.sccs,
                              branch.stringToSimVarHT))
  end
  return (IF_EQUATION(newBranches), nRemovedSlots * length(ifEq.branches))
end

"""
    cleanupTrivialResidualEquations(simCode; sourcePass = "")

Remove residuals that are provably trivial without symbolic algebra. To avoid
changing equation/unknown balance, a residual is only removed when it contains
no unknown cref and it evaluates or simplifies syntactically to zero. Branch
residuals are removed only when the same residual slot is trivial in every
branch of an IF_EQUATION, preserving the branch alignment expected by codegen.
"""
function cleanupTrivialResidualEquations(simCode::SIM_CODE;
                                         sourcePass::AbstractString = "")::SIM_CODE
  local (newResiduals, nResidualsRemoved) =
    _filterTrivialResiduals(simCode.residualEquations, simCode)
  local (newInitials, nInitialsRemoved) =
    _filterTrivialInitialEquations(simCode.initialEquations, simCode)
  local newIfEquations = IF_EQUATION[]
  local nConditionalRemoved = 0
  local nIfRemoved = 0
  for ifEq in simCode.ifEquations
    local (newIfEq, nRemoved) = _cleanupTrivialBranchResiduals(ifEq, simCode)
    nConditionalRemoved += nRemoved
    if newIfEq === nothing
      nIfRemoved += 1
    else
      push!(newIfEquations, newIfEq)
    end
  end
  if nResidualsRemoved == 0 && nInitialsRemoved == 0 &&
     nConditionalRemoved == 0 && nIfRemoved == 0
    return simCode
  end
  @assign simCode.residualEquations = newResiduals
  @assign simCode.initialEquations = newInitials
  @assign simCode.ifEquations = newIfEquations
  local afterText = isempty(sourcePass) ? "" : " after $sourcePass"
  @debug "[SIMCODE: $(simCode.name): trivialCleanup] removed trivial equations$afterText" residuals=nResidualsRemoved initial=nInitialsRemoved conditionalResiduals=nConditionalRemoved ifEquations=nIfRemoved
  return simCode
end

function _rewriteResidualIfExp(eq::BDAE.RESIDUAL_EQUATION, simCode::SIM_CODE)::BDAE.RESIDUAL_EQUATION
  local newExp = resolveConstantIfExp(eq.exp, simCode)
  return newExp === eq.exp ? eq : BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr)
end

function _rewriteInitialIfExp(@nospecialize(eq), simCode::SIM_CODE)
  if eq isa BDAE.RESIDUAL_EQUATION
    return _rewriteResidualIfExp(eq, simCode)
  elseif eq isa BDAE.EQUATION
    local newLhs = resolveConstantIfExp(eq.lhs, simCode)
    local newRhs = resolveConstantIfExp(eq.rhs, simCode)
    return (newLhs === eq.lhs && newRhs === eq.rhs) ? eq :
           BDAE.EQUATION(newLhs, newRhs, eq.source, eq.attributes)
  end
  return eq
end

function _rewriteBranchIfExp(branch::BRANCH, simCode::SIM_CODE)::BRANCH
  local newCondition = branch.identifier == ELSE_BRANCH ?
                       branch.condition :
                       resolveConstantIfExp(branch.condition, simCode)
  local newResiduals = BDAE.RESIDUAL_EQUATION[
    _rewriteResidualIfExp(eq, simCode) for eq in branch.residualEquations
  ]
  return BRANCH(newCondition, newResiduals,
                branch.identifier, branch.targets, branch.isSingular,
                branch.matchOrder, branch.equationGraph, branch.sccs,
                branch.stringToSimVarHT)
end

function _reindexIfBranches(branches::Vector{BRANCH})::Vector{BRANCH}
  local n = length(branches)
  local out = BRANCH[]
  sizehint!(out, n)
  for (idx, branch) in enumerate(branches)
    local isLast = idx == n
    local isElse = branch.identifier == ELSE_BRANCH || isLast
    local identifier = isElse ? ELSE_BRANCH : idx
    local target = isElse ? ELSE_BRANCH : idx + 1
    local condition = isElse ? DAE.SCONST("ELSE_BRANCH") : branch.condition
    push!(out, BRANCH(condition, branch.residualEquations,
                      identifier, target, branch.isSingular,
                      branch.matchOrder, branch.equationGraph, branch.sccs,
                      branch.stringToSimVarHT))
  end
  return out
end

function _pruneIfEquation(ifEq::IF_EQUATION,
                          simCode::SIM_CODE)::Tuple{Union{IF_EQUATION, Nothing}, Vector{BDAE.RESIDUAL_EQUATION}, Int, Bool}
  local rewrittenBranches = BRANCH[_rewriteBranchIfExp(branch, simCode) for branch in ifEq.branches]
  local newBranches = BRANCH[]
  local promoted = BDAE.RESIDUAL_EQUATION[]
  local nPrunedBranches = 0
  local hasUnconditionalFallback = false
  for branch in rewrittenBranches
    if branch.identifier == ELSE_BRANCH
      hasUnconditionalFallback = true
      if isempty(newBranches)
        append!(promoted, branch.residualEquations)
        return (nothing, promoted, nPrunedBranches + 1, true)
      end
      push!(newBranches, branch)
      return (IF_EQUATION(_reindexIfBranches(newBranches)), promoted, nPrunedBranches, true)
    end
    local condValue = tryEvalCondition(branch.condition, simCode)
    if condValue === false
      nPrunedBranches += 1
      continue
    elseif condValue === true
      hasUnconditionalFallback = true
      if isempty(newBranches)
        append!(promoted, branch.residualEquations)
        return (nothing, promoted, nPrunedBranches + 1, true)
      end
      push!(newBranches, BRANCH(DAE.SCONST("ELSE_BRANCH"),
                                branch.residualEquations,
                                ELSE_BRANCH, ELSE_BRANCH, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
      return (IF_EQUATION(_reindexIfBranches(newBranches)), promoted, nPrunedBranches + 1, true)
    else
      push!(newBranches, branch)
    end
  end
  if isempty(newBranches)
    return (nothing, promoted, nPrunedBranches, hasUnconditionalFallback)
  end
  if !hasUnconditionalFallback
    #= No `else` branch was found and no static-true branch fired. We saw only
       `false` and dynamic branches. The Modelica spec says an IF_EQUATION
       without `else` contributes equations only when one branch matches at
       runtime; statically-false branches are dead. We could safely drop them,
       but doing so would also need a structural recount further upstream
       (branches participate in matching/causalization). Keep the conservative
       behavior and return the IFEXP-rewritten branch list unchanged. The
       prune count is reported truthfully so the log is not misleading. =#
    return (IF_EQUATION(rewrittenBranches), promoted, nPrunedBranches, false)
  end
  return (IF_EQUATION(_reindexIfBranches(newBranches)), promoted, nPrunedBranches, true)
end

"""
    pruneConstantConditions(simCode)

Resolve constant-condition IFEXP nodes throughout the main equation vectors and
prune IF_EQUATION branches whose guards are compile-time constants. If a branch
is selected before any dynamic guard remains, its residual equations are promoted
to top-level residuals and the IF_EQUATION is removed.
"""
function pruneConstantConditions(simCode::SIM_CODE)::SIM_CODE
  local newResiduals = BDAE.RESIDUAL_EQUATION[
    _rewriteResidualIfExp(eq, simCode) for eq in simCode.residualEquations
  ]
  local newInitials = typeof(simCode.initialEquations)()
  for eq in simCode.initialEquations
    push!(newInitials, _rewriteInitialIfExp(eq, simCode))
  end
  local newIfEquations = IF_EQUATION[]
  local nPrunedBranches = 0
  local nPromotedResiduals = 0
  local nRemovedIfEquations = 0
  for ifEq in simCode.ifEquations
    local (newIfEq, promoted, pruned, _) = _pruneIfEquation(ifEq, simCode)
    nPrunedBranches += pruned
    if !isempty(promoted)
      append!(newResiduals, promoted)
      nPromotedResiduals += length(promoted)
    end
    if newIfEq === nothing
      nRemovedIfEquations += 1
    else
      push!(newIfEquations, newIfEq)
    end
  end
  @assign simCode.residualEquations = newResiduals
  @assign simCode.initialEquations = newInitials
  @assign simCode.ifEquations = newIfEquations
  if nPrunedBranches > 0 || nPromotedResiduals > 0 || nRemovedIfEquations > 0
    @debug "[SIMCODE: $(simCode.name): constantConditionPruning] pruned constant conditions" branches=nPrunedBranches promotedResiduals=nPromotedResiduals removedIfEquations=nRemovedIfEquations
  end
  return simCode
end

"""
    identifyOutputOnlyVariables(simCode::SIM_CODE,
                                matchOrder::Vector{Int},
                                matchIdxToName::Dict{Int,String})

Identify variables and equations that do not influence the dynamic states.
Uses a fresh bipartite matching and robust CREF extraction via `traverseExpTopDown`.

The BFS seeds from equations matched to essential variables (STATE, STATE_DERIVATIVE,
DISCRETE, OCC, irreducible). It propagates backward through the use-def chain: for
each essential equation, all variables it references are marked essential, and the
equations that PRODUCE those variables (via matchOrder) are enqueued.

Returns `(outputOnlyVarNames, outputOnlyEqIndices, eqRefs)`.
"""
function identifyOutputOnlyVariables(simCode::SIM_CODE,
                                     matchOrder::Vector{Int},
                                     matchIdxToName::Dict{Int,String})
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  #= Build the base name index for robust CREF extraction =#
  local baseNameToFullNames = buildBaseNameIndex(ht)
  #= Build expression-level dependency: for each equation, which variable names does it reference? =#
  local eqRefs = Vector{Set{String}}(undef, nEqs)
  for i in 1:nEqs
    eqRefs[i] = collectEquationVarNames(resEqs[i].exp, ht, baseNameToFullNames)
  end
  #= Build varName -> equation index that solves it (via fresh matchOrder).
     matchOrder[matchIdx] = eqIdx; matchIdxToName[matchIdx] = varName =#
  local varNameToEq = Dict{String, Int}()
  local baseNameToEqs = Dict{String, Vector{Int}}()
  for (matchIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0 && haskey(matchIdxToName, matchIdx)
      local vn = matchIdxToName[matchIdx]
      varNameToEq[vn] = eqIdx
      local bn = replace(vn, r"\[.*" => "")
      if bn != vn
        if !haskey(baseNameToEqs, bn)
          baseNameToEqs[bn] = Int[]
        end
        push!(baseNameToEqs[bn], eqIdx)
      end
    end
  end
  #= Find seed equations: those matched to essential variable kinds =#
  local seedEqs = Set{Int}()
  for (varName, (_idx, sv)) in ht
    local isEssentialKind = @match sv.varKind begin
      STATE(__) => true
      STATE_DERIVATIVE(__) => true
      OCC_VARIABLE(__) => true
      DISCRETE(__) => true
      _ => false
    end
    if isEssentialKind && haskey(varNameToEq, varName)
      push!(seedEqs, varNameToEq[varName])
    end
  end
  #= Add equations for irreducible variables =#
  for irName in simCode.irreductableVariables
    if haskey(varNameToEq, irName)
      push!(seedEqs, varNameToEq[irName])
    end
  end
  #= Protect alias representative variables from elimination.
     These variables appear in observed equations generated from the aliasMap.
     If they are eliminated, the observed equations will reference missing unknowns. =#
  for alias in simCode.aliasMap
    if haskey(varNameToEq, alias.representativeName)
      push!(seedEqs, varNameToEq[alias.representativeName])
    end
  end
  #= Classify unmatched equations: seed those referencing unknowns =#
  local matchedEqs = Set{Int}()
  for (matchIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      push!(matchedEqs, eqIdx)
    end
  end
  local unknownNames = Set{String}()
  for (vn, (_idx, sv)) in ht
    local isUnknown = @match sv.varKind begin
      STATE(__) => true
      STATE_DERIVATIVE(__) => true
      ALG_VARIABLE(__) => true
      SimulationCode.ARRAY(__) => true
      OCC_VARIABLE(__) => true
      DISCRETE(__) => true
      _ => false
    end
    if isUnknown
      push!(unknownNames, vn)
    end
  end
  for eqIdx in 1:nEqs
    if !(eqIdx in matchedEqs)
      #= Check if this unmatched equation references any unknowns =#
      local refsUnknown = false
      for refName in eqRefs[eqIdx]
        if refName in unknownNames
          refsUnknown = true
          break
        end
      end
      if refsUnknown
        push!(seedEqs, eqIdx)
      end
    end
  end
  #= BFS: from seed equations, follow the use-def chain backward.
     For each equation, find all variable names it references. For each referenced
     variable, find the equation that PRODUCES it (via varNameToEq). Enqueue that. =#
  local essentialEqs = Set{Int}()
  local queue = collect(seedEqs)
  while !isempty(queue)
    local eqIdx = popfirst!(queue)
    if eqIdx in essentialEqs
      continue
    end
    push!(essentialEqs, eqIdx)
    if eqIdx >= 1 && eqIdx <= nEqs
      for refVarName in eqRefs[eqIdx]
        #= Exact match =#
        if haskey(varNameToEq, refVarName)
          local prodEq = varNameToEq[refVarName]
          if !(prodEq in essentialEqs)
            push!(queue, prodEq)
          end
        end
        #= Base name match for array variables =#
        if haskey(baseNameToEqs, refVarName)
          for prodEq in baseNameToEqs[refVarName]
            if !(prodEq in essentialEqs)
              push!(queue, prodEq)
            end
          end
        end
      end
    end
  end
  #= Identify output-only equations and their matched variables =#
  local outputOnlyEqIndices = Set{Int}()
  local outputOnlyVarNames = Set{String}()
  local eqToMatchIdx = Dict{Int, Int}()
  for (matchIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      eqToMatchIdx[eqIdx] = matchIdx
    end
  end
  for eqIdx in 1:nEqs
    if !(eqIdx in essentialEqs)
      push!(outputOnlyEqIndices, eqIdx)
      if haskey(eqToMatchIdx, eqIdx)
        local mIdx = eqToMatchIdx[eqIdx]
        if haskey(matchIdxToName, mIdx)
          push!(outputOnlyVarNames, matchIdxToName[mIdx])
        end
      end
    end
  end
  return (outputOnlyVarNames, outputOnlyEqIndices, eqRefs)
end

"""
    eliminateOutputOnlyVariables(simCode::SIM_CODE, options::EliminationOptions)

Remove output-only variables and their defining equations from the SimCode.
Rebuilds a fresh bipartite matching from the current (post-optimization) equation
and variable sets, then performs backward reachability to identify output-only
equation-variable pairs. Only eliminates ALG_VARIABLE or ARRAY unknowns,
preserving the equation-unknown balance that MTK requires.

The eliminated equations and variable names are stored in `simCode.eliminatedEquations`
and `simCode.eliminatedVariables` for later reconstruction (e.g. 3D visualization).

Returns the modified SIM_CODE (uses @assign for immutable struct mutation).
"""
function eliminateOutputOnlyVariables(simCode::SIM_CODE, options::EliminationOptions)
  #= Guard: skip for VSS/multi-mode models (subModels or recompilation-based
     metaModel/flatModel), but allow DOCC models (structuralTransitions only)
     since they re-flatten at runtime =#
  if hasSubModels(simCode) || hasMetaModel(simCode) || hasFlatModel(simCode)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] skipping for VSS/multi-mode model"
    return simCode
  end
  #= Rebuild a fresh matching from the current (post-optimization) system =#
  local (matchOrder, nameToMatchIdx, matchIdxToName) = rebuildMatchOrder(simCode)
  if isempty(matchOrder)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] matching failed or system not square, skipping"
    return simCode
  end
  #= Identify output-only equations and variables using the fresh matching =#
  local (outputOnlyVarNames, outputOnlyEqIndices, eqRefs) =
    identifyOutputOnlyVariables(simCode, matchOrder, matchIdxToName)
  if isempty(outputOnlyEqIndices)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] no output-only equations found"
    return simCode
  end
  #= Build inverse matching: equation index -> match index =#
  local ht = simCode.stringToSimVarHT
  local eqToMatchIdx = Dict{Int, Int}()
  for (mIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      eqToMatchIdx[eqIdx] = mIdx
    end
  end
  #= Only eliminate output-only equation-variable PAIRS where the matched variable
     is ALG_VARIABLE or ARRAY. This preserves equation-unknown balance. =#
  local eqsToEliminate = Set{Int}()
  local varsToRemove = Set{String}()
  local eliminatedPairs = Tuple{String, Int}[]  #= (varName, eqIdx) for pairing =#
  local nSkippedNonAlg = 0
  local nSkippedUnmatched = 0
  for eqIdx in outputOnlyEqIndices
    if !haskey(eqToMatchIdx, eqIdx)
      nSkippedUnmatched += 1
      continue
    end
    local mIdx = eqToMatchIdx[eqIdx]
    if !haskey(matchIdxToName, mIdx)
      nSkippedUnmatched += 1
      continue
    end
    local vn = matchIdxToName[mIdx]
    if !haskey(ht, vn)
      nSkippedUnmatched += 1
      continue
    end
    local (_, sv) = ht[vn]
    local isEliminable = @match sv.varKind begin
      ALG_VARIABLE(__) => true
      SimulationCode.ARRAY(__) => true
      _ => false
    end
    if isEliminable
      push!(eqsToEliminate, eqIdx)
      push!(varsToRemove, vn)
      push!(eliminatedPairs, (vn, eqIdx))
    else
      nSkippedNonAlg += 1
    end
  end
  if isempty(eqsToEliminate)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] no eliminable equation-variable pairs found"
    return simCode
  end
  #= Guard: never eliminate ALL equations. A system with zero equations
     after elimination would crash downstream (filterConstantEquations, MTK). =#
  if length(eqsToEliminate) >= length(simCode.residualEquations)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] would eliminate all $(length(simCode.residualEquations)) equations, skipping"
    return simCode
  end
  #= Safety check: verify no surviving equation references an eliminated variable.
     Build reverse index: variable name -> equations that reference it. =#
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  local varNameToRefEqs = Dict{String, Set{Int}}()
  for eqIdx in 1:nEqs
    for refName in eqRefs[eqIdx]
      if !haskey(varNameToRefEqs, refName)
        varNameToRefEqs[refName] = Set{Int}()
      end
      push!(varNameToRefEqs[refName], eqIdx)
    end
  end
  #= Collect variable names referenced by when-equations so they are never eliminated.
     When-equations live outside the residual system and are not in varNameToRefEqs. =#
  local whenRefNames = Set{String}()
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(whenRefNames, whenEq.whenEquation)
  end
  local rescuedVars = Set{String}()
  for vn in varsToRemove
    local referencedBySurvivor = false
    #= Check residual equations =#
    if haskey(varNameToRefEqs, vn)
      for refEqIdx in varNameToRefEqs[vn]
        if !(refEqIdx in eqsToEliminate)
          referencedBySurvivor = true
          break
        end
      end
    end
    #= Also check base name =#
    if !referencedBySurvivor
      local bn = replace(vn, r"\[.*" => "")
      if bn != vn && haskey(varNameToRefEqs, bn)
        for refEqIdx in varNameToRefEqs[bn]
          if !(refEqIdx in eqsToEliminate)
            referencedBySurvivor = true
            break
          end
        end
      end
    end
    #= Check when-equations =#
    if !referencedBySurvivor && vn in whenRefNames
      referencedBySurvivor = true
    end
    if referencedBySurvivor
      push!(rescuedVars, vn)
    end
  end
  local nRescued = length(rescuedVars)
  if !isempty(rescuedVars)
    for vn in rescuedVars
      delete!(varsToRemove, vn)
      if haskey(nameToMatchIdx, vn)
        local rescuedMIdx = nameToMatchIdx[vn]
        local rescuedEqIdx = matchOrder[rescuedMIdx]
        if rescuedEqIdx > 0
          delete!(eqsToEliminate, rescuedEqIdx)
        end
      end
    end
  end
  #= Filter residualEquations: remove eliminated equations =#
  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(eqsToEliminate))
  for (i, eq) in enumerate(resEqs)
    if !(i in eqsToEliminate)
      push!(newResEqs, eq)
    end
  end
  #= Build parallel (varName, equation) vectors from the paired data.
     Filter out rescued variables. =#
  local survivingPairs = filter(p -> !(p[1] in rescuedVars), eliminatedPairs)
  local elimPairedVars = String[p[1] for p in survivingPairs]
  local elimPairedEqs = BDAE.RESIDUAL_EQUATION[resEqs[p[2]] for p in survivingPairs]
  #= Filter stringToSimVarHT: remove eliminated variables =#
  local newHT = copy(ht)
  for varName in varsToRemove
    delete!(newHT, varName)
  end
  @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] eliminated $(length(eqsToEliminate)) eq-var pairs, $(length(varsToRemove)) variables removed (rescued: $nRescued, skipped: $nSkippedNonAlg non-algebraic, $nSkippedUnmatched unmatched). $(length(newResEqs)) equations, $(length(newHT)) variables remain"
  @BACKEND_LOGGING begin
    local buf = IOBuffer()
    println(buf, "=== ELIMINATION DEBUG ===")
    println(buf, "Removed variables ($(length(varsToRemove))):")
    for vn in sort(collect(varsToRemove))
      println(buf, "  ", vn)
    end
    println(buf, "Rescued variables ($nRescued):")
    for vn in sort(collect(rescuedVars))
      println(buf, "  ", vn)
    end
    println(buf, "Eliminated equation indices: ", sort(collect(eqsToEliminate)))
    println(buf, "=== END DEBUG ===")
    OMBackend.debugWrite(OMBackend.logPath("backend/simCode", "elimination_debug.log"), String(take!(buf)))
  end
  @assign simCode.residualEquations = newResEqs
  @assign simCode.stringToSimVarHT = newHT
  @assign simCode.eliminatedEquations = elimPairedEqs
  @assign simCode.eliminatedVariables = elimPairedVars
  return simCode
end

"""
    buildAsubName(baseName::String, subs::Vector)::String

Reconstruct a subscripted variable name from an ASUB expression.
Turns base name "a" with subscripts [1, 2] into "a[1][2]" to match hash table keys.
"""
function buildAsubName(baseName::String, subs)::String
  buf = baseName
  for s in subs
    @match s begin
      DAE.ICONST(i) => begin buf *= string("[", i, "]") end
      _ => return ""  #= Non-constant subscript: cannot resolve statically =#
    end
  end
  return buf
end

"""
    extractCrefName(exp::DAE.Exp)

Extract the variable name from a CREF or ASUB(CREF, ...) expression.
Returns `(name::String, cref::DAE.ComponentRef, ty::DAE.Type)` or `nothing`
if the expression is not a simple variable reference.
"""
function extractCrefName(@nospecialize(exp))
  @match exp begin
    DAE.CREF(cr, ty) => begin
      return (DAE_identifierToString(cr), cr, ty)
    end
    #= ASUB-wrapped CREFs are skipped for alias detection.
       The ASUB wraps a base CREF with subscripts, but the CREF itself does not
       carry the subscripts. Eliminating an ASUB alias would replace the base CREF
       in all equations (affecting all subscripts), breaking the equation balance.
       These equations are better handled by MTK structural_simplify. =#
    _ => return nothing
  end
end

"""
    isUnknownVarKind(varKind::SimVarType)::Bool

Check if a variable kind represents an unknown (not a parameter or constant).
Only unknowns participate in the equation-unknown balance.
"""
function isUnknownVarKind(@nospecialize(varKind::SimVarType))::Bool
  @match varKind begin
    STATE(__) => true
    STATE_DERIVATIVE(__) => true
    ALG_VARIABLE(__) => true
    ARRAY(__) => true
    OCC_VARIABLE(__) => true
    DISCRETE(__) => true
    _ => false
  end
end

"""
    varKindPriority(varKind::SimVarType)::Int

Return priority of a variable kind for alias representative selection.
Higher priority variables are preferred as representatives (never eliminated).
"""
function varKindPriority(@nospecialize(varKind::SimVarType))::Int
  @match varKind begin
    STATE(__) => 100
    STATE_DERIVATIVE(__) => 90
    DISCRETE(__) => 80
    OCC_VARIABLE(__) => 70
    ALG_VARIABLE(__) => 20
    ARRAY(__) => 10
    _ => 0
  end
end

"""
    isRealValued(ty::DAE.Type)::Bool

Check if a DAE type represents a Real-valued (floating point) variable.
Only Real-valued variables are eligible for alias elimination.
"""
function isRealValued(@nospecialize(ty))::Bool
  @match ty begin
    DAE.T_REAL(__) => true
    DAE.T_ARRAY(ty = innerTy) => isRealValued(innerTy)
    _ => false
  end
end

#= Same-class check for alias eligibility: Real, Boolean, Integer, Enumeration. =#
function _aliasTypeClass(@nospecialize(ty))::Symbol
  @match ty begin
    DAE.T_REAL(__) => :real
    DAE.T_BOOL(__) => :bool
    DAE.T_INTEGER(__) => :int
    DAE.T_ENUMERATION(__) => :enum
    DAE.T_ARRAY(ty = innerTy) => _aliasTypeClass(innerTy)
    _ => :other
  end
end

"""
    detectConstantEquation(exp::DAE.Exp, ht)

Detect if a residual equation represents a constant propagation opportunity
or a trivially true equation between parameters.

Returns:
  - `(:trivial, nothing)` if both sides are parameters (equation is tautological)
  - `(:constprop, (unknownName, paramName, negated, paramCref, paramTy))` if one
    side is an unknown and the other is a parameter
  - `nothing` if the equation does not match any constant pattern
"""
function detectConstantEquation(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      local isAdd = @match op begin
        DAE.ADD(__) => true
        _ => false
      end
      if !isSub && !isAdd
        return nothing
      end
      local r1 = extractCrefName(e1)
      local r2 = extractCrefName(e2)
      if r1 === nothing || r2 === nothing
        return nothing
      end
      local (n1, cr1, t1) = r1
      local (n2, cr2, t2) = r2
      if !haskey(ht, n1) || !haskey(ht, n2)
        return nothing
      end
      local (_, sv1) = ht[n1]
      local (_, sv2) = ht[n2]
      local isUnk1 = isUnknownVarKind(sv1.varKind)
      local isUnk2 = isUnknownVarKind(sv2.varKind)
      local negated = isAdd

      if !isUnk1 && !isUnk2
        #= Both parameters: trivial equation, always satisfied =#
        return (:trivial, nothing)
      elseif isUnk1 && !isUnk2
        #= n1 is unknown, n2 is parameter: unknown = (+/-)param =#
        return (:constprop, (n1, n2, negated, cr2, t2))
      elseif !isUnk1 && isUnk2
        #= n1 is parameter, n2 is unknown: unknown = (+/-)param =#
        return (:constprop, (n2, n1, negated, cr1, t1))
      else
        #= Both unknowns: handled by alias elimination, not us =#
        return nothing
      end
    end
    _ => return nothing
  end
end

"""
    _classifyAdditionalDiscreteVariables(simCode::SIM_CODE)::SIM_CODE

Reclassify any `ALG_VARIABLE` whose only definition lives inside a
`when`-equation as `DISCRETE`. This catches Real-valued variables that are
held between events (Modelica's classic `T_start := time` pattern inside a
`when`-clause) but were not picked up by the upstream Integer/enum discrete
classification, leaving them as algebraic unknowns with no defining residual.

Without this pass, the model has fewer equations than unknowns at
`structural_simplify` time and MTK raises `ExtraVariablesSystemException`.
After this pass the variable lands in `discreteVariables` during MTK
codegen, gets a `der(x) ~ 0` dummy, and the when-clause callback affect
has a state to update.

Detection: walk every `BDAE.WHEN_EQUATION` and collect the LHS variable name
of every `BDAE.ASSIGN` operator (and the `stateVar` of every `BDAE.REINIT`).
Any name in the set whose simvar is currently `ALG_VARIABLE` is
reclassified to `DISCRETE`. Variables that already have a non-algebraic
kind (state, parameter, discrete, occ, array, data structure) are left
alone.

No-op for VSS / submodel / metaModel / flatModel variants where the
equation set is restructured at runtime.
"""
function _classifyAdditionalDiscreteVariables(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode) ||
     hasFlatModel(simCode) || hasMetaModel(simCode)
    @debug "[SIMCODE: $(simCode.name): classifyAdditionalDiscretes] skipped (VSS/multi-mode model)"
    return simCode
  end

  if isempty(simCode.whenEquations)
    return simCode
  end

  #= Step 1: collect every var name that appears as LHS of a when-ASSIGN
     or as the target of a when-REINIT. =#
  local whenLhsNames = Set{String}()
  for whenEq in simCode.whenEquations
    _collectWhenAssignTargets!(whenLhsNames, whenEq.whenEquation)
  end

  if isempty(whenLhsNames)
    return simCode
  end

  #= Step 2: reclassify ALG_VARIABLE -> DISCRETE for those names. =#
  local ht = simCode.stringToSimVarHT
  local reclassified = String[]
  for name in whenLhsNames
    haskey(ht, name) || continue
    local (idx, sv) = ht[name]
    if sv.varKind isa ALG_VARIABLE
      ht[name] = (idx, SIMVAR(sv.name, sv.index, DISCRETE(), sv.attributes))
      push!(reclassified, name)
    end
  end

  if !isempty(reclassified)
    @debug "[SIMCODE: $(simCode.name): classifyAdditionalDiscretes] reclassified $(length(reclassified)) algebraic variables to discrete (when-driven): $(reclassified)"
  end
  return simCode
end

#= Walk a BDAE.WhenEquation (WHEN_STMTS) tree, collecting every variable
   name that is assigned or reinit-ed inside. Recurses into elsewhen. =#
function _collectWhenAssignTargets!(names::Set{String}, whenEq)
  @match whenEq begin
    BDAE.WHEN_STMTS(_, stmts, elsewhen) => begin
      for stmt in stmts
        @match stmt begin
          BDAE.ASSIGN(left = lhs) => begin
            local r = extractCrefName(lhs)
            if r !== nothing
              push!(names, r[1])
            end
          end
          BDAE.REINIT(stateVar = cr) => begin
            push!(names, DAE_identifierToString(cr))
          end
          _ => nothing
        end
      end
      if isSome(elsewhen)
        @match SOME(elseEq) = elsewhen
        _collectWhenAssignTargets!(names, elseEq)
      end
    end
    _ => nothing
  end
end

"""
    foldParameterClosure(simCode::SIM_CODE)::SIM_CODE

BLT-driven parameter-closure fold.

Walks the scalar blocks of the block-lower-triangular decomposition of the
equation graph in topological order, and for each block whose matched
unknown `v` is defined by a residual of the form `v - f(...) = 0` (or
`f(...) - v = 0`) where `f` depends only on parameters, constants and
previously folded unknowns, promotes `v` to `PARAMETER(SOME(f))` and drops
the residual.

Motivating case: `Modelica.Blocks.Sources.KinematicPTP` introduces seven
algebraic unknowns (`aux1`, `sd_max`, `sdd_max`, `Ta1`, `Ta2`, `Tv`, `Te`,
`noWphase`) whose defining equations are closures over parameters. With no
fold they reach MTK as unknowns with a zero start guess, and Newton's first
evaluation produces `sqrt(1/0) = Inf` and `1/abs(0) = Inf`, aborting init.
Folding turns the chain into parameter bindings that MTK resolves at
elaboration time, so Newton never sees them.

Algebraic loops (`BLTBlock.isLoop == true`) and any non-ALG unknown
(states, derivatives, discretes, OCC, arrays) are skipped — those belong
to MTK's structural_simplify.

No-op for VSS / submodel simcodes, for empty residual sets, and when the
earlier matching flagged the system as singular (index reduction has
priority over folding there).
"""
function foldParameterClosure(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode)
    return simCode
  end
  if isempty(simCode.residualEquations)
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  #= Narrow exclusion set: names that appear inside an if-equation's
     `branch.condition` expression. `createIfEquation` ->
     `evalInitialCondition` eval's the condition at MODULE scope; a
     folded PARAMETER binding lives only in the model function's local
     scope, so evaluating a condition that references a folded name
     raises `UndefVarError`. Earlier versions excluded the full
     `simCode.irreductableVariables` set, but that was too broad:
     `getIrreductableVars` flattens every cref reachable through any
     IF_EQUATION branch body (conditions AND branch residuals), which
     accidentally blocked KinematicPTP and similar closures from folding
     even when the variable only appeared in a branch residual. Restrict
     to condition-only references here. `_THETA` markers
     (overconstrained-connector Zimmer constant) are preserved
     separately. =#
  local excludedFromFold = Set{String}()
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      #= Else branches (identifier == -1) carry a trivial/constant
         condition and are never evaluated by `evalInitialCondition`. =#
      if branch.identifier == -1
        continue
      end
      for cref in Util.getAllCrefs(branch.condition)
        push!(excludedFromFold, string(cref))
      end
    end
  end
  for name in keys(ht)
    if endswith(name, "THETA")
      push!(excludedFromFold, name)
    end
  end
  #= Snapshot of ALG-unknown names: only these are candidates for folding. =#
  local algNames = Set{String}()
  for (name, (_, sv)) in ht
    if name in excludedFromFold
      continue
    end
    @match sv.varKind begin
      ALG_VARIABLE(_) => push!(algNames, name)
      _ => nothing
    end
  end
  if isempty(algNames)
    return simCode
  end

  local nResEqs = length(simCode.residualEquations)

  #= Build per-variable counts of how many residuals place it on the LHS of a
     BINARY SUB. A variable with count == 1 has a unique defining equation
     and is a safe fold target: promoting it to PARAMETER cannot conflict
     with another residual that also "solves for" it. Variables with
     count != 1 are left to MTK (ambiguous or residual-appears-only-as-use). =#
  local defEqOfVar = Dict{String, Int}()    #= varName -> eqIdx of its defining residual =#
  local defCountOfVar = Dict{String, Int}() #= varName -> #residuals with v on LHS of BINARY SUB =#
  for (i, eq) in enumerate(simCode.residualEquations)
    local candidateName = extractBinarySubLhsCrefName(eq.exp, algNames)
    if candidateName !== nothing
      defCountOfVar[candidateName] = get(defCountOfVar, candidateName, 0) + 1
      if !haskey(defEqOfVar, candidateName)
        defEqOfVar[candidateName] = i
      end
    end
  end

  local foldMap = Dict{String, DAE.Exp}()
  local foldedNames = Set{String}()
  local elimIdxSet = Set{Int}()

  #= Iterate to fixed point: a freshly folded variable may unlock
     downstream closures (e.g. sd_max = 1/abs(aux1[1]) becomes foldable
     once aux1[1] is a parameter). =#
  local progressed = true
  while progressed
    progressed = false
    for (varName, eqIdx) in defEqOfVar
      if varName in foldedNames
        continue
      end
      if defCountOfVar[varName] != 1
        continue
      end
      local rhs = detectSolvableParameterClosure(
        simCode.residualEquations[eqIdx].exp, varName, ht, foldedNames)
      if rhs !== nothing
        foldMap[varName] = rhs
        push!(foldedNames, varName)
        push!(elimIdxSet, eqIdx)
        progressed = true
      end
    end
  end

  if isempty(foldMap)
    return simCode
  end

  #= Guard against complete elimination: static models (e.g. MatrixMultTest,
     where every output is bound to a pure-constant expression) would have
     every residual drained by the fold, leaving MTK with 0 equations and
     0 unknowns. MTK's `System(...)` constructor cannot accept an empty
     equation list and raises `MethodError` downstream. In that degenerate
     case, keep at least the original residuals so the normal alias/constant
     elimination pipeline handles the trivial simplification. =#
  if length(simCode.residualEquations) - length(elimIdxSet) == 0
    @debug "[SIMCODE: $(simCode.name): foldParameterClosure] fold would eliminate all residuals; skipping to preserve MTK build" wouldFold=length(foldMap)
    return simCode
  end

  local newHT = copy(ht)
  for (name, bindExp) in foldMap
    local (idx, oldSV) = newHT[name]
    newHT[name] = (idx, SIMVAR(oldSV.name, oldSV.index,
                               PARAMETER(SOME(bindExp)), oldSV.attributes))
  end
  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newResEqs, nResEqs - length(elimIdxSet))
  for (i, eq) in enumerate(simCode.residualEquations)
    if !(i in elimIdxSet)
      push!(newResEqs, eq)
    end
  end

  @assign simCode.residualEquations = newResEqs
  @assign simCode.stringToSimVarHT = newHT
  #= Invalidate derived name sets that were computed against the pre-fold
     HT classification. `irreductableVariables` was collected by
     `getIrreductableVars` BEFORE this pass ran, so it may still name
     variables that are now parameters. Leaving them in causes the MTK
     codegen `_batchBlock` to try `setmetadata(kinematicPTP_Ta1, Irreducible)`
     against a module-scope name that no longer exists (the parameter is
     only bound in the model function's local scope).
     Same logic applies to `sharedVariables` for completeness, though in
     practice that is populated only in multi-submodel scenarios. =#
  if !isempty(simCode.irreductableVariables)
    @assign simCode.irreductableVariables =
      filter(v -> !(v in foldedNames), simCode.irreductableVariables)
  end
  if !isempty(simCode.sharedVariables)
    @assign simCode.sharedVariables =
      filter(v -> !(v in foldedNames), simCode.sharedVariables)
  end
  #= Do not push folded entries to `eliminatedEquations` / `eliminatedVariables`.
     Those parallel arrays drive alias observed-equation reconstruction (a
     separate mechanism). Folded variables become full PARAMETERs with a bound
     expression, so MTK evaluates them directly at elaboration time.
     Their observed values appear automatically in the ODESystem's parameter
     substitution path — no observed equation needed. =#
  return simCode
end

"""
    extractBinarySubLhsCrefName(exp, candidateNames)

If `exp` has the shape `DAE.BINARY(lhs, SUB, _)` (or `DAE.BINARY(_, SUB, lhs)`)
where `lhs` is a CREF or scalarized ASUB of a name in `candidateNames`,
return that name; otherwise return `nothing`.

Used by the fold to pre-index "defining equations": residuals that name a
variable in their top-level subtraction. If a name appears in more than one
such residual, MTK owns the disambiguation.
"""
function extractBinarySubLhsCrefName(@nospecialize(exp), candidateNames::Set{String})
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      if !isSub
        return nothing
      end
      local n1 = extractCrefLikeName(e1)
      if n1 !== nothing && n1 in candidateNames
        return n1
      end
      local n2 = extractCrefLikeName(e2)
      if n2 !== nothing && n2 in candidateNames
        return n2
      end
      return nothing
    end
    _ => return nothing
  end
end

"""
    extractCrefLikeName(exp)

Reconstruct the string name for `DAE.CREF(cr, _)` or
`DAE.ASUB(DAE.CREF(cr, _), subs)` with constant `ICONST` subs. Returns
`nothing` otherwise.
"""
function extractCrefLikeName(@nospecialize(exp))
  @match exp begin
    DAE.CREF(cr, _) => DAE_identifierToString(cr)
    DAE.ASUB(exp = innerExp, sub = subs) => begin
      @match innerExp begin
        DAE.CREF(cr, _) => begin
          local full = buildAsubName(DAE_identifierToString(cr), subs)
          return isempty(full) ? nothing : full
        end
        _ => nothing
      end
    end
    _ => nothing
  end
end

"""
    detectSolvableParameterClosure(exp, matchedName, ht, foldedNames)

Return the RHS `f` if `exp` has the shape `v - f = 0` or `f - v = 0`
where `v` is a bare CREF to `matchedName` and `f` contains only parameters,
constants and names already in `foldedNames`. Otherwise return `nothing`.

Intentionally narrow: more exotic shapes (ADD with sign flip, MUL by
parameter divisor, CALL-wrapped LHS) are left to MTK.
"""
function detectSolvableParameterClosure(@nospecialize(exp), matchedName::String,
                                        ht, foldedNames::Set{String})
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      if !isSub
        return nothing
      end
      if isCrefNamed(e1, matchedName) && isParameterClosureExp(e2, ht, foldedNames, matchedName)
        return e2
      end
      if isCrefNamed(e2, matchedName) && isParameterClosureExp(e1, ht, foldedNames, matchedName)
        return e1
      end
      return nothing
    end
    _ => return nothing
  end
end

"""
    isCrefNamed(exp, name)::Bool

True iff `exp` is a variable reference whose resolved name equals `name`.

Two shapes are accepted:
  * `DAE.CREF(cr, _)`  -- the scalar case.
  * `DAE.ASUB(DAE.CREF(cr, _), subs)` where every sub is a constant `ICONST`
    -- the scalarized array-element case. The reconstructed name includes
    the literal subscript suffix, e.g. `kinematicPTP_aux1[1]`, matching the
    key format used by `stringToSimVarHT`.

Non-constant subscripts are rejected (we cannot resolve the name statically).
"""
function isCrefNamed(@nospecialize(exp), name::String)::Bool
  @match exp begin
    DAE.CREF(cr, _) => DAE_identifierToString(cr) == name
    DAE.ASUB(exp = innerExp, sub = subs) => begin
      @match innerExp begin
        DAE.CREF(cr, _) => begin
          local full = buildAsubName(DAE_identifierToString(cr), subs)
          return !isempty(full) && full == name
        end
        _ => false
      end
    end
    _ => false
  end
end

"""
    isParameterClosureExp(exp, ht, foldedNames, excludeName)::Bool

True iff every CREF name reachable in `exp`:
  * is not equal to `excludeName` (no self-reference), AND
  * either resolves to a non-unknown SimVar in `ht` (parameter/constant),
    or is already in `foldedNames`.

Names absent from `ht` are rejected conservatively: they are typically
subscripted or array-element references whose parameter status is not
robustly recoverable by string match here.
"""
function isParameterClosureExp(@nospecialize(exp), ht, foldedNames::Set{String},
                               excludeName::String)::Bool
  local names = Set{String}()
  collectCrefNames!(names, exp)
  for n in names
    if n == excludeName
      return false
    end
    if n in foldedNames
      continue
    end
    local entry = get(ht, n, nothing)
    if entry === nothing
      return false
    end
    local (_, sv) = entry
    if isUnknownVarKind(sv.varKind)
      return false
    end
  end
  return true
end

struct _CanonicalNameContext
  rename::Dict{String, String}
  known::Set{String}
  nameMap::OMBackend.NameRewriteMap
end

function _canonicalVariableKey(name::AbstractString)::String
  return OMBackend.canonicalName(name)
end

function _recordNameRewrite!(ctx::_CanonicalNameContext, original::AbstractString,
                             canonical::AbstractString)::String
  local originalName = String(original)
  local canonicalName = String(canonical)
  ctx.nameMap.originalToCanonical[originalName] = canonicalName
  if originalName != canonicalName || !haskey(ctx.nameMap.canonicalToOriginal, canonicalName)
    ctx.nameMap.canonicalToOriginal[canonicalName] = originalName
  end
  return canonicalName
end

function _canonicalVariableKey(name::AbstractString, ctx::_CanonicalNameContext)::String
  return _recordNameRewrite!(ctx, name, OMBackend.canonicalName(name))
end

function _originalPathName(path::Absyn.IDENT)::String
  return path.name
end

function _originalPathName(path::Absyn.QUALIFIED)::String
  return Base.string(path.name, ".", _originalPathName(path.path))
end

function _originalPathName(path::Absyn.FULLYQUALIFIED)::String
  return Base.string(".", _originalPathName(path.path))
end

function _originalSubscriptSuffix(subscriptLst)::String
  if listEmpty(subscriptLst)
    return ""
  end
  local buf = IOBuffer()
  for subscript in subscriptLst
    print(buf, "[")
    print(buf, Base.string(subscript))
    print(buf, "]")
  end
  return String(take!(buf))
end

function _originalCrefName(cr::DAE.CREF_IDENT)::String
  return Base.string(cr.ident, _originalSubscriptSuffix(cr.subscriptLst))
end

function _originalCrefName(cr::DAE.CREF_ITER)::String
  return Base.string(cr.ident, _originalSubscriptSuffix(cr.subscriptLst))
end

function _originalCrefName(cr::DAE.CREF_QUAL)::String
  return Base.string(cr.ident,
                     _originalSubscriptSuffix(cr.subscriptLst),
                     ".",
                     _originalCrefName(cr.componentRef))
end

function _originalCrefName(cr::DAE.WILD)::String
  return "_"
end

function _originalCrefName(cr::DAE.OPTIMICA_ATTR_INST_CREF)::String
  return _originalCrefName(cr.componentRef)
end

function _canonicalizeVarKind(kind::SimVarType, ctx::_CanonicalNameContext)::SimVarType
  return @match kind begin
    STATE_DERIVATIVE(varName) => STATE_DERIVATIVE(_canonicalVariableKey(varName, ctx))
    PARAMETER(SOME(bindExp)) => PARAMETER(SOME(_canonicalizeExp(bindExp, ctx)))
    DATA_STRUCTURE(SOME(bindExp)) => DATA_STRUCTURE(SOME(_canonicalizeExp(bindExp, ctx)))
    ARRAY(dims, SOME(bindExp)) => ARRAY(dims, SOME(_canonicalizeExp(bindExp, ctx)))
    ARRAY_PARAMETER(dims, SOME(bindExp)) => ARRAY_PARAMETER(dims, SOME(_canonicalizeExp(bindExp, ctx)))
    STRING(SOME(bindExp)) => STRING(SOME(_canonicalizeExp(bindExp, ctx)))
    _ => kind
  end
end

function _canonicalizeSimVar(sv::SIMVAR, ctx::_CanonicalNameContext)::SIMVAR
  return SIMVAR(_canonicalVariableKey(sv.name, ctx),
                sv.index,
                _canonicalizeVarKind(sv.varKind, ctx),
                sv.attributes)
end

function _canonicalizeSimVarHT(ht::AbstractDict{String, Tuple{Integer, SimVar}},
                               ctx::_CanonicalNameContext)
  local out = OrderedDict{String, Tuple{Integer, SimVar}}()
  for (name, (idx, sv)) in ht
    local canonicalName = get(ctx.rename, name, nothing)
    if canonicalName === nothing
      canonicalName = _canonicalVariableKey(name, ctx)
    else
      _recordNameRewrite!(ctx, name, canonicalName)
    end
    local newVar = _canonicalizeSimVar(sv, ctx)
    if newVar.name != canonicalName
      newVar = SIMVAR(canonicalName, newVar.index, newVar.varKind, newVar.attributes)
    end
    out[canonicalName] = (idx, newVar)
  end
  return out
end

function _canonicalizeExp(@nospecialize(exp), ctx::_CanonicalNameContext)
  local (newExp, _) = Util.traverseExpTopDown(exp, _canonicalizeCrefExp, ctx)
  return newExp
end

function _canonicalizeCrefExp(@nospecialize(exp), ctx::_CanonicalNameContext)
  @match exp begin
    DAE.CREF(cr, ty) => begin
      return (DAE.CREF(_canonicalizeComponentRef(cr, ty, ctx), ty), false, ctx)
    end
    DAE.CALL(path, expLst, attr) => begin
      local canonicalPath = OMBackend.canonicalName(path)
      _recordNameRewrite!(ctx, _originalPathName(path), canonicalPath)
      return (DAE.CALL(Absyn.IDENT(canonicalPath), expLst, attr), true, ctx)
    end
    DAE.RECORD(path, exps, comp, ty) => begin
      local canonicalPath = OMBackend.canonicalName(path)
      _recordNameRewrite!(ctx, _originalPathName(path), canonicalPath)
      return (DAE.RECORD(Absyn.IDENT(canonicalPath), exps, comp, ty), true, ctx)
    end
    DAE.PARTEVALFUNCTION(path, expList, ty, origType) => begin
      local canonicalPath = OMBackend.canonicalName(path)
      _recordNameRewrite!(ctx, _originalPathName(path), canonicalPath)
      return (DAE.PARTEVALFUNCTION(Absyn.IDENT(canonicalPath), expList, ty, origType), true, ctx)
    end
    _ => return (exp, true, ctx)
  end
end

function _stripInnermostSubscripts(cr::DAE.CREF_IDENT)
  return DAE.CREF_IDENT(cr.ident, cr.identType, MetaModelica.nil)
end

function _stripInnermostSubscripts(cr::DAE.CREF_ITER)
  return DAE.CREF_ITER(cr.ident, cr.index, cr.identType, MetaModelica.nil)
end

function _stripInnermostSubscripts(cr::DAE.CREF_QUAL)
  return DAE.CREF_QUAL(cr.ident,
                       cr.identType,
                       cr.subscriptLst,
                       _stripInnermostSubscripts(cr.componentRef))
end

function _stripInnermostSubscripts(cr::DAE.WILD)
  return cr
end

function _innermostSubscripts(cr::DAE.CREF_IDENT)
  return cr.subscriptLst
end

function _innermostSubscripts(cr::DAE.CREF_ITER)
  return cr.subscriptLst
end

function _innermostSubscripts(cr::DAE.CREF_QUAL)
  return _innermostSubscripts(cr.componentRef)
end

function _innermostSubscripts(::DAE.WILD)
  return MetaModelica.nil
end

function _innermostType(cr::DAE.CREF_IDENT)
  return cr.identType
end

function _innermostType(cr::DAE.CREF_ITER)
  return cr.identType
end

function _innermostType(cr::DAE.CREF_QUAL)
  return _innermostType(cr.componentRef)
end

function _hasDimensions(dims)::Bool
  try
    return !listEmpty(dims)
  catch
    for _ in dims
      return true
    end
  end
  return false
end

function _declaredDaeVarCrefType(v::DAE.VAR)::DAE.Type
  local crefTy = _innermostType(v.componentRef)
  if crefTy isa DAE.T_UNKNOWN
    crefTy = v.ty
  elseif !(crefTy isa DAE.T_ARRAY) && v.ty isa DAE.T_ARRAY
    crefTy = v.ty
  end
  if !(crefTy isa DAE.T_ARRAY) && _hasDimensions(v.dims)
    return DAE.T_ARRAY(crefTy, v.dims)
  end
  return crefTy
end

function _canonicalizeComponentRef(cr::DAE.ComponentRef, ty::DAE.Type,
                                   ctx::_CanonicalNameContext)::DAE.ComponentRef
  local originalFull = _originalCrefName(cr)
  local fullName = OMBackend.canonicalName(cr)
  local canonicalFull = get(ctx.rename, originalFull, nothing)
  if canonicalFull === nothing
    canonicalFull = get(ctx.rename, fullName, nothing)
  end
  if canonicalFull === nothing
    canonicalFull = _recordNameRewrite!(ctx, originalFull, fullName)
  else
    _recordNameRewrite!(ctx, originalFull, canonicalFull)
  end
  if canonicalFull in ctx.known
    return DAE.CREF_IDENT(canonicalFull, ty, MetaModelica.nil)
  end

  local baseCr = _stripInnermostSubscripts(cr)
  local originalBase = _originalCrefName(baseCr)
  local baseName = OMBackend.canonicalName(baseCr)
  local canonicalBase = get(ctx.rename, originalBase, nothing)
  if canonicalBase === nothing
    canonicalBase = get(ctx.rename, baseName, nothing)
  end
  if canonicalBase === nothing
    canonicalBase = _recordNameRewrite!(ctx, originalBase, baseName)
  else
    _recordNameRewrite!(ctx, originalBase, canonicalBase)
  end
  local finalSubs = _innermostSubscripts(cr)
  if canonicalBase in ctx.known || !listEmpty(finalSubs)
    return DAE.CREF_IDENT(canonicalBase, _innermostType(cr), finalSubs)
  end

  return DAE.CREF_IDENT(canonicalFull, ty, MetaModelica.nil)
end

function _canonicalizeEquation(eq, ctx::_CanonicalNameContext)
  if eq isa BDAE.RESIDUAL_EQUATION
    return BDAE.RESIDUAL_EQUATION(_canonicalizeExp(eq.exp, ctx), eq.source, eq.attr)
  elseif eq isa BDAE.EQUATION
    return BDAE.EQUATION(_canonicalizeExp(eq.lhs, ctx),
                         _canonicalizeExp(eq.rhs, ctx),
                         eq.source,
                         eq.attributes)
  elseif eq isa BDAE.ARRAY_EQUATION
    return BDAE.ARRAY_EQUATION(eq.dimSize,
                               _canonicalizeExp(eq.left, ctx),
                               _canonicalizeExp(eq.right, ctx),
                               eq.source,
                               eq.attr,
                               eq.recordSize)
  elseif eq isa BDAE.COMPLEX_EQUATION
    return BDAE.COMPLEX_EQUATION(eq.size,
                                 _canonicalizeExp(eq.left, ctx),
                                 _canonicalizeExp(eq.right, ctx),
                                 eq.source,
                                 eq.attr)
  elseif eq isa BDAE.SOLVED_EQUATION
    return BDAE.SOLVED_EQUATION(_canonicalizeComponentRef(eq.componentRef, _innermostType(eq.componentRef), ctx),
                                _canonicalizeExp(eq.exp, ctx),
                                eq.source,
                                eq.attr)
  elseif eq isa BDAE.WHEN_EQUATION
    return BDAE.WHEN_EQUATION(eq.size,
                              _canonicalizeWhenStmts(eq.whenEquation, ctx),
                              eq.source,
                              eq.attr)
  elseif eq isa BDAE.STRUCTURAL_WHEN_EQUATION
    return BDAE.STRUCTURAL_WHEN_EQUATION(eq.size,
                                         _canonicalizeWhenStmts(eq.whenEquation, ctx),
                                         eq.source,
                                         eq.attr)
  elseif eq isa BDAE.IF_EQUATION
    local newConditions = _mapList(e -> _canonicalizeExp(e, ctx), eq.conditions)
    local newTrue = _mapList(branch -> _mapList(e -> _canonicalizeEquation(e, ctx), branch), eq.eqnstrue)
    local newFalse = _mapList(e -> _canonicalizeEquation(e, ctx), eq.eqnsfalse)
    return BDAE.IF_EQUATION(newConditions, newTrue, newFalse, eq.source, eq.attr)
  elseif eq isa BDAE.ALGORITHM
    return BDAE.ALGORITHM(eq.size, _canonicalizeAlgorithm(eq.alg, ctx), eq.source, eq.expand, eq.attr)
  elseif eq isa BDAE.ASSERT_EQUATION
    return BDAE.ASSERT_EQUATION(_canonicalizeExp(eq.condition, ctx),
                                _canonicalizeExp(eq.message, ctx),
                                _canonicalizeExp(eq.level, ctx),
                                eq.source)
  end
  return eq
end

function _recordFunctionVarName!(known::Set{String}, v::DAE.VAR,
                                 ctx::_CanonicalNameContext)
  local original = _originalCrefName(v.componentRef)
  local canonical = OMBackend.canonicalName(v.componentRef)
  _recordNameRewrite!(ctx, original, canonical)
  push!(known, canonical)
  return nothing
end

function _mapList(f::Function, lst)
  local out = MetaModelica.nil
  for x in lst
    out = f(x) <| out
  end
  return listReverse(out)
end

function _mapVectorLike(f::Function, xs)
  local out = typeof(xs)()
  for x in xs
    push!(out, f(x))
  end
  return out
end

function _canonicalizeWhenStmts(whenStmts::BDAE.WHEN_STMTS,
                                ctx::_CanonicalNameContext)
  local newCond = _canonicalizeExp(whenStmts.condition, ctx)
  local newStmtLst = _mapList(stmt -> _canonicalizeWhenOperator(stmt, ctx),
                              whenStmts.whenStmtLst)
  local newElse = @match whenStmts.elsewhenPart begin
    SOME(elseWhenEq) => SOME(_canonicalizeElseWhenPart(elseWhenEq, ctx))
    NONE() => NONE()
    _ => whenStmts.elsewhenPart
  end
  return BDAE.WHEN_STMTS(newCond, newStmtLst, newElse)
end

function _canonicalizeElseWhenPart(elseWhen, ctx::_CanonicalNameContext)
  if elseWhen isa BDAE.WHEN_STMTS
    return _canonicalizeWhenStmts(elseWhen, ctx)
  end
  return _canonicalizeEquation(elseWhen, ctx)
end

function _canonicalizeCrefValue(exp::DAE.CREF, ctx::_CanonicalNameContext)::DAE.CREF
  local newExp = _canonicalizeExp(exp, ctx)
  return newExp isa DAE.CREF ? newExp : exp
end

function _canonicalizeWhenOperator(stmt, ctx::_CanonicalNameContext)
  if stmt isa BDAE.ASSIGN
    return BDAE.ASSIGN(_canonicalizeExp(stmt.left, ctx),
                       _canonicalizeExp(stmt.right, ctx),
                       stmt.source)
  elseif stmt isa BDAE.REINIT
    return BDAE.REINIT(_canonicalizeCrefValue(stmt.stateVar, ctx),
                       _canonicalizeExp(stmt.value, ctx),
                       stmt.source)
  elseif stmt isa BDAE.ASSERT
    return BDAE.ASSERT(_canonicalizeExp(stmt.condition, ctx),
                       _canonicalizeExp(stmt.message, ctx),
                       _canonicalizeExp(stmt.level, ctx),
                       stmt.source)
  elseif stmt isa BDAE.TERMINATE
    return BDAE.TERMINATE(_canonicalizeExp(stmt.message, ctx), stmt.source)
  elseif stmt isa BDAE.NORETCALL
    return BDAE.NORETCALL(_canonicalizeExp(stmt.exp, ctx), stmt.source)
  elseif stmt isa BDAE.RECOMPILATION
    return BDAE.RECOMPILATION(_canonicalizeCrefValue(stmt.componentToChange, ctx),
                              _canonicalizeExp(stmt.newValue, ctx))
  elseif stmt isa BDAE.AGENTIC_RECOMPILATION
    return BDAE.AGENTIC_RECOMPILATION([_canonicalizeCrefValue(c, ctx) for c in stmt.componentsToChange],
                                      stmt.prompt,
                                      stmt.initialEquations)
  end
  return stmt
end

function _canonicalizeBranch(branch::BRANCH, ctx::_CanonicalNameContext)
  return BRANCH(_canonicalizeExp(branch.condition, ctx),
                _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), branch.residualEquations),
                branch.identifier,
                branch.targets,
                branch.isSingular,
                branch.matchOrder,
                branch.equationGraph,
                branch.sccs,
                _canonicalizeSimVarHT(branch.stringToSimVarHT, ctx))
end

function _canonicalizeStructuralTransition(tr::StructuralTransition,
                                           ctx::_CanonicalNameContext)
  if tr isa EXPLICIT_STRUCTURAL_TRANSISTION
    local st = tr.structuralTransition
    return EXPLICIT_STRUCTURAL_TRANSISTION(
      BDAE.STRUCTURAL_TRANSISTION(_canonicalVariableKey(st.fromState, ctx),
                                  _canonicalVariableKey(st.toState, ctx),
                                  _canonicalizeExp(st.transistionCondition, ctx)))
  elseif tr isa IMPLICIT_STRUCTURAL_TRANSISTION
    return IMPLICIT_STRUCTURAL_TRANSISTION(_canonicalizeEquation(tr.structuralWhenEquation, ctx))
  end
  return tr
end

function _canonicalizeIfEquation(ifEq::IF_EQUATION, ctx::_CanonicalNameContext)
  return IF_EQUATION(_mapVectorLike(branch -> _canonicalizeBranch(branch, ctx),
                                    ifEq.branches))
end

function _canonicalizeDaeVar(v::DAE.VAR, ctx::_CanonicalNameContext)::DAE.VAR
  local newBinding = @match v.binding begin
    SOME(b) => SOME(_canonicalizeExp(b, ctx))
    NONE() => NONE()
  end
  return DAE.VAR(_canonicalizeComponentRef(v.componentRef, _declaredDaeVarCrefType(v), ctx),
                 v.kind,
                 v.direction,
                 v.parallelism,
                 v.protection,
                 v.ty,
                 newBinding,
                 v.dims,
                 v.connectorType,
                 v.source,
                 v.variableAttributesOption,
                 v.comment,
                 v.innerOuter)
end

function _canonicalizeStatement(stmt::DAE.Statement, ctx::_CanonicalNameContext)::DAE.Statement
  if stmt isa DAE.STMT_ASSIGN
    return DAE.STMT_ASSIGN(stmt.type_,
                           _canonicalizeExp(stmt.exp1, ctx),
                           _canonicalizeExp(stmt.exp, ctx),
                           stmt.source)
  elseif stmt isa DAE.STMT_TUPLE_ASSIGN
    return DAE.STMT_TUPLE_ASSIGN(stmt.type_,
                                 _mapList(e -> _canonicalizeExp(e, ctx), stmt.expExpLst),
                                 _canonicalizeExp(stmt.exp, ctx),
                                 stmt.source)
  elseif stmt isa DAE.STMT_ASSIGN_ARR
    return DAE.STMT_ASSIGN_ARR(stmt.type_,
                               _canonicalizeExp(stmt.lhs, ctx),
                               _canonicalizeExp(stmt.exp, ctx),
                               stmt.source)
  elseif stmt isa DAE.STMT_IF
    return DAE.STMT_IF(_canonicalizeExp(stmt.exp, ctx),
                       _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                       _canonicalizeElse(stmt.else_, ctx),
                       stmt.source)
  elseif stmt isa DAE.STMT_FOR
    return DAE.STMT_FOR(stmt.type_,
                        stmt.iterIsArray,
                        stmt.iter,
                        stmt.index,
                        _canonicalizeExp(stmt.range, ctx),
                        _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                        stmt.source)
  elseif stmt isa DAE.STMT_PARFOR
    return DAE.STMT_PARFOR(stmt.type_,
                           stmt.iterIsArray,
                           stmt.iter,
                           stmt.index,
                           _canonicalizeExp(stmt.range, ctx),
                           _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                           stmt.loopPrlVars,
                           stmt.source)
  elseif stmt isa DAE.STMT_WHILE
    return DAE.STMT_WHILE(_canonicalizeExp(stmt.exp, ctx),
                          _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                          stmt.source)
  elseif stmt isa DAE.STMT_WHEN
    local newElseWhen = @match stmt.elseWhen begin
      SOME(s) => SOME(_canonicalizeStatement(s, ctx))
      NONE() => NONE()
    end
    return DAE.STMT_WHEN(_canonicalizeExp(stmt.exp, ctx),
                         _mapList(c -> _canonicalizeComponentRef(c, _innermostType(c), ctx), stmt.conditions),
                         stmt.initialCall,
                         _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                         newElseWhen,
                         stmt.source)
  elseif stmt isa DAE.STMT_ASSERT
    return DAE.STMT_ASSERT(_canonicalizeExp(stmt.cond, ctx),
                           _canonicalizeExp(stmt.msg, ctx),
                           _canonicalizeExp(stmt.level, ctx),
                           stmt.source)
  elseif stmt isa DAE.STMT_TERMINATE
    return DAE.STMT_TERMINATE(_canonicalizeExp(stmt.msg, ctx), stmt.source)
  elseif stmt isa DAE.STMT_REINIT
    return DAE.STMT_REINIT(_canonicalizeExp(stmt.var, ctx),
                           _canonicalizeExp(stmt.value, ctx),
                           stmt.source)
  elseif stmt isa DAE.STMT_NORETCALL
    return DAE.STMT_NORETCALL(_canonicalizeExp(stmt.exp, ctx), stmt.source)
  elseif stmt isa DAE.STMT_FAILURE
    return DAE.STMT_FAILURE(_mapList(s -> _canonicalizeStatement(s, ctx), stmt.body),
                            stmt.source)
  end
  return stmt
end

function _canonicalizeElse(elseBranch::DAE.Else, ctx::_CanonicalNameContext)::DAE.Else
  if elseBranch isa DAE.ELSEIF
    return DAE.ELSEIF(_canonicalizeExp(elseBranch.exp, ctx),
                      _mapList(s -> _canonicalizeStatement(s, ctx), elseBranch.statementLst),
                      _canonicalizeElse(elseBranch.else_, ctx))
  elseif elseBranch isa DAE.ELSE
    return DAE.ELSE(_mapList(s -> _canonicalizeStatement(s, ctx), elseBranch.statementLst))
  end
  return elseBranch
end

function _canonicalizeAlgorithm(alg::DAE.Algorithm, ctx::_CanonicalNameContext)::DAE.Algorithm
  if alg isa DAE.ALGORITHM_STMTS
    return DAE.ALGORITHM_STMTS(_mapList(s -> _canonicalizeStatement(s, ctx), alg.statementLst))
  end
  return alg
end

function _functionCanonicalNameContext(f, ctx::_CanonicalNameContext)
  local known = Set{String}(["time", "pi", "e"])
  if hasproperty(f, :inputs)
    for v in f.inputs
      _recordFunctionVarName!(known, v, ctx)
    end
  end
  if hasproperty(f, :outputs)
    for v in f.outputs
      _recordFunctionVarName!(known, v, ctx)
    end
  end
  if hasproperty(f, :locals)
    for v in f.locals
      _recordFunctionVarName!(known, v, ctx)
    end
  end
  return _CanonicalNameContext(ctx.rename, known, ctx.nameMap)
end

function _canonicalizeFunction(f::MODELICA_FUNCTION, ctx::_CanonicalNameContext)
  local canonicalFunctionName = _canonicalVariableKey(f.name, ctx)
  local functionCtx = _functionCanonicalNameContext(f, ctx)
  return MODELICA_FUNCTION(canonicalFunctionName,
                           _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.inputs),
                           _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.outputs),
                           _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.locals),
                           _mapVectorLike(s -> _canonicalizeStatement(s, functionCtx), f.statements))
end

function _canonicalizeFunction(f::EXTERNAL_MODELICA_FUNCTION, ctx::_CanonicalNameContext)
  local canonicalFunctionName = _canonicalVariableKey(f.name, ctx)
  local functionCtx = _functionCanonicalNameContext(f, ctx)
  return EXTERNAL_MODELICA_FUNCTION(canonicalFunctionName,
                                    _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.inputs),
                                    _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.outputs),
                                    f.libInfo)
end

function _canonicalizeFunction(f::ModelicaFunction, ctx::_CanonicalNameContext)
  return f
end

function canonicalizeCrefNames(simCode::SIM_CODE;
                               nameMap::OMBackend.NameRewriteMap = OMBackend.NameRewriteMap())::SIM_CODE
  local rename = Dict{String, String}()
  for name in keys(simCode.stringToSimVarHT)
    rename[name] = _canonicalVariableKey(name)
  end
  for name in simCode.eliminatedVariables
    rename[name] = _canonicalVariableKey(name)
  end
  for entry in simCode.aliasMap
    rename[entry.eliminatedName] = _canonicalVariableKey(entry.eliminatedName)
    rename[entry.representativeName] = _canonicalVariableKey(entry.representativeName)
  end

  local known = Set{String}(values(rename))
  union!(known, Set(["time", "pi", "e"]))
  local ctx = _CanonicalNameContext(rename, known, nameMap)

  @assign simCode.name = _canonicalVariableKey(simCode.name, ctx)
  @assign simCode.stringToSimVarHT = _canonicalizeSimVarHT(simCode.stringToSimVarHT, ctx)
  @assign simCode.residualEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.residualEquations)
  @assign simCode.initialEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.initialEquations)
  @assign simCode.whenEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.whenEquations)
  @assign simCode.ifEquations = _mapVectorLike(ifEq -> _canonicalizeIfEquation(ifEq, ctx), simCode.ifEquations)
  @assign simCode.structuralTransitions = _mapVectorLike(tr -> _canonicalizeStructuralTransition(tr, ctx),
                                                         simCode.structuralTransitions)
  @assign simCode.subModels = _mapVectorLike(subModel -> canonicalizeCrefNames(subModel; nameMap = nameMap), simCode.subModels)
  @assign simCode.sharedVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.sharedVariables)
  @assign simCode.topVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.topVariables)
  @assign simCode.sharedEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.sharedEquations)
  @assign simCode.activeModel = _canonicalVariableKey(simCode.activeModel, ctx)
  @assign simCode.irreductableVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.irreductableVariables)
  @assign simCode.functions = _mapVectorLike(f -> _canonicalizeFunction(f, ctx), simCode.functions)
  @assign simCode.eliminatedEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.eliminatedEquations)
  @assign simCode.eliminatedVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.eliminatedVariables)
  @assign simCode.aliasMap = _mapVectorLike(entry -> AliasEntry(_canonicalVariableKey(entry.eliminatedName, ctx),
                                                               _canonicalVariableKey(entry.representativeName, ctx),
                                                               entry.negated),
                                           simCode.aliasMap)
  return simCode
end

"""
    simplifyEnumLiteralPaths(simCode::SIM_CODE)::SIM_CODE

Collapse the qualified namespace path of every `DAE.ENUM_LITERAL` to a
single `Absyn.IDENT` whose name is `Type.Literal` (the leaf two segments
joined by `.`). The integer index is preserved verbatim — that is what
arithmetic and comparison rely on. Frontend-shaped literals like

    ENUM_LITERAL(QUALIFIED("Modelica", QUALIFIED("Electrical", ...
                  QUALIFIED("Logic", IDENT("'U'")))), 1)

become

    ENUM_LITERAL(IDENT("Logic.'U'"), 1)

Reduces memory and makes downstream dumps directly readable without
custom @match arms for every nested QUALIFIED depth. Applied once at
SimCode entry — no later pass synthesises fresh ENUM_LITERAL paths,
they only substitute existing ones.
"""
function simplifyEnumLiteralPaths(simCode::SIM_CODE)::SIM_CODE
  local nRewritten = Ref(0)
  local _shortenPath = function(p)
    local segs = String[]
    local _walk = nothing
    _walk = function(x)
      if x isa Absyn.IDENT
        push!(segs, x.name)
      elseif x isa Absyn.QUALIFIED
        push!(segs, x.name)
        _walk(x.path)
      elseif x isa Absyn.FULLYQUALIFIED
        _walk(x.path)
      end
    end
    _walk(p)
    if length(segs) >= 2
      return Absyn.IDENT(segs[end-1] * "." * segs[end])
    elseif length(segs) == 1
      return Absyn.IDENT(segs[1])
    end
    return p
  end
  local _rewrite = function(exp, _)
    if exp isa DAE.ENUM_LITERAL && !(exp.name isa Absyn.IDENT && occursin('.', exp.name.name))
      nRewritten[] += 1
      return (DAE.ENUM_LITERAL(_shortenPath(exp.name), exp.index), true, nothing)
    end
    return (exp, true, nothing)
  end

  #= Helper: rewrite ENUM_LITERALs inside a single DAE.Exp. =#
  local _rewriteExp = function(e)
    local (newExp, _) = Util.traverseExpTopDown(e, _rewrite, nothing)
    return newExp
  end

  #= 1. Variable bindings (PARAMETER, DATA_STRUCTURE, ARRAY, ARRAY_PARAMETER). =#
  for (varName, (idx, sv)) in simCode.stringToSimVarHT
    local newKind = @match sv.varKind begin
      PARAMETER(SOME(b))      => PARAMETER(SOME(_rewriteExp(b)))
      DATA_STRUCTURE(SOME(b)) => DATA_STRUCTURE(SOME(_rewriteExp(b)))
      ARRAY(dims, SOME(b))    => ARRAY(dims, SOME(_rewriteExp(b)))
      ARRAY_PARAMETER(dims, SOME(b)) => ARRAY_PARAMETER(dims, SOME(_rewriteExp(b)))
      _ => sv.varKind
    end
    if newKind !== sv.varKind
      @assign sv.varKind = newKind
      simCode.stringToSimVarHT[varName] = (idx, sv)
    end
  end

  #= 2. Residual + initial equations. `initialEquations` may contain
        BDAE.EQUATION (lhs/rhs) entries alongside RESIDUAL_EQUATION; handle
        both forms. =#
  local _rewriteEq = function(eq)
    if eq isa BDAE.RESIDUAL_EQUATION
      return BDAE.RESIDUAL_EQUATION(_rewriteExp(eq.exp), eq.source, eq.attr)
    elseif eq isa BDAE.EQUATION
      return BDAE.EQUATION(_rewriteExp(eq.lhs), _rewriteExp(eq.rhs), eq.source, eq.attributes)
    end
    return eq
  end
  @assign simCode.residualEquations = [_rewriteEq(eq) for eq in simCode.residualEquations]
  @assign simCode.initialEquations = [_rewriteEq(eq) for eq in simCode.initialEquations]

  if nRewritten[] > 0
    @debug "[SIMCODE: $(simCode.name): simplifyEnumLiteralPaths] collapsed $(nRewritten[]) ENUM_LITERAL qualified paths to Type.Literal IDENT form"
  end
  return simCode
end

"""
    inlinePreOfConstantParameters(simCode::SIM_CODE)::SIM_CODE

Replace `pre(x)` with `x` inside residual equations whenever `x` is a
constant-bound PARAMETER. For a parameter the value at the previous event
is the same as the value now, so this fold is exact and lets downstream
`propagateConstants` resolve the residual naturally.
"""
function inlinePreOfConstantParameters(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode)
    return simCode
  end
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nReplaced = Ref(0)

  local _isPreCall = function(e)
    e isa DAE.CALL || return false
    length(e.expLst) == 1 || return false
    e.path isa Absyn.IDENT || return false
    e.path.name in ("pre", "previous")
  end

  local _argIsConstParam = function(e)
    local arg = listHead(e.expLst)
    arg isa DAE.CREF || return false
    local name = string(arg.componentRef)
    haskey(ht, name) || return false
    local (_, sv) = ht[name]
    sv.varKind isa PARAMETER || return false
    return true
  end

  local _rewrite = function(exp, _)
    if _isPreCall(exp) && _argIsConstParam(exp)
      nReplaced[] += 1
      return (listHead(exp.expLst), false, nothing)
    end
    return (exp, true, nothing)
  end

  local newEqs = BDAE.RESIDUAL_EQUATION[]
  for eq in resEqs
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, _rewrite, nothing)
    push!(newEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end
  if nReplaced[] > 0
    @debug "[SIMCODE: $(simCode.name): inlinePreOfConstantParameters] replaced $(nReplaced[]) `pre(constParam)` occurrences with the parameter directly"
  end
  @assign simCode.residualEquations = newEqs
  return simCode
end

"""
    propagateConstants(simCode::SIM_CODE)::SIM_CODE

Constant propagation pass. Detects equations of the form `unknown = parameter`
and substitutes the parameter CREF for the unknown CREF in all equations.
Also removes trivially true `parameter = parameter` equations.

This pass runs BEFORE alias elimination because removing unknowns may reveal
new alias opportunities.

Preserves equation-unknown balance: each constant propagation removes 1 equation
and 1 unknown. Trivial equation removal only removes equations that have no
unknowns (no balance impact).
"""
function propagateConstants(simCode::SIM_CODE)
  #= Guard: skip for VSS or multi-mode models =#
  if hasStructuralTransitions(simCode) || hasSubModels(simCode)
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  local sharedVarSet = Set{String}(simCode.sharedVariables)
  local irreducibleSet = Set{String}(simCode.irreductableVariables)

  #= Phase 1: Detect constant equations.
     First collect all base array names referenced in equations so we can skip
     eliminating scalar elements whose base array is still used (e.g. as a
     function call argument). =#
  local allBaseNames = Set{String}()
  for eq in resEqs
    local eqNames = Set{String}()
    collectCrefNames!(eqNames, eq.exp)
    for n in eqNames
      if !occursin('[', n)
        push!(allBaseNames, n)
      end
    end
  end

  if !isempty(allBaseNames)
    @debug "[SIMCODE: $(simCode.name): constantPropagation] base array names referenced" allBaseNames=collect(allBaseNames)
  end

  local constMap = Dict{String, Tuple{String, Bool, DAE.ComponentRef, DAE.Type}}()
  local constEqIndices = Set{Int}()
  local trivialEqIndices = Set{Int}()

  local changed = true
  while changed
    changed = false
    for (i, eq) in enumerate(resEqs)
      if i in constEqIndices || i in trivialEqIndices
        continue
      end
      local result = detectConstantEquation(eq.exp, ht)
      if result === nothing
        continue
      end
      local (kind, data) = result
      if kind == :trivial
        push!(trivialEqIndices, i)
        changed = true
      elseif kind == :constprop
        local (unknownName, paramName, negated, paramCref, paramTy) = data
        #= Skip shared or irreducible unknowns =#
        if unknownName in sharedVarSet || unknownName in irreducibleSet
          continue
        end
        #= Skip if the unknown is a subscripted array element whose base name
           is still referenced as a whole array (e.g. in function call arguments).
           Eliminating R_T[1][1] while R_T is passed to resolve2() would break
           code generation which looks up individual elements from the HT. =#
        local bracketIdx = findfirst('[', unknownName)
        if bracketIdx !== nothing
          local baseName = unknownName[1:bracketIdx-1]
          if baseName in allBaseNames
            continue
          end
        end
        if !haskey(constMap, unknownName)
          constMap[unknownName] = (paramName, negated, paramCref, paramTy)
          push!(constEqIndices, i)
          changed = true
        end
      end
    end
    if changed && !isempty(constMap)
      #= Apply current substitutions to all remaining equation expressions
         so that chained constant patterns are revealed in the next iteration =#
      local updatedEqs = BDAE.RESIDUAL_EQUATION[]
      for (i, eq) in enumerate(resEqs)
        local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteAliasCref, constMap)
        push!(updatedEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
      end
      resEqs = updatedEqs
    end
  end

  local nConst = length(constEqIndices)
  local nTrivial = length(trivialEqIndices)
  if nConst == 0 && nTrivial == 0
    @debug "[SIMCODE: $(simCode.name): constantPropagation] no constant equations found"
    return simCode
  end

  @debug "[SIMCODE: $(simCode.name): constantPropagation] found $nConst unknown=param equations and $nTrivial trivial param=param equations"

  #= Phase 2: Build final equation list with substitutions applied.
     We collect (varName, residual) pairs for const-bound eliminations so
     the downstream `eliminatedEquations` / `eliminatedVariables` arrays
     stay aligned. Trivial `param=param` residuals are dropped without
     recording since they have no unknown to associate. =#
  #= Reverse-map each constprop equation index to its unknown name by re-running
     `detectConstantEquation` on the equations recorded during Phase 1. =#
  local eqIdxToUnknown = Dict{Int, String}()
  for i in constEqIndices
    local eq = simCode.residualEquations[i]
    local r = detectConstantEquation(eq.exp, ht)
    r === nothing && continue
    if r[1] == :constprop
      eqIdxToUnknown[i] = r[2][1]
    end
  end
  local allRemoved = union(constEqIndices, trivialEqIndices)
  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  local elimPairs = Tuple{String, BDAE.RESIDUAL_EQUATION}[]
  sizehint!(newResEqs, nEqs - length(allRemoved))

  for (i, eq) in enumerate(simCode.residualEquations)
    if i in allRemoved
      if i in constEqIndices && haskey(eqIdxToUnknown, i)
        push!(elimPairs, (eqIdxToUnknown[i], eq))
      end
    else
      local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteAliasCref, constMap)
      push!(newResEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
    end
  end
  local elimEqs = BDAE.RESIDUAL_EQUATION[p[2] for p in elimPairs]

  #= Substitute in if-equation branches =#
  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = BDAE.RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = Util.traverseExpTopDown(brEq.exp, substituteAliasCref, constMap)
        push!(newBranchEqs, BDAE.RESIDUAL_EQUATION(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = Util.traverseExpTopDown(branch.condition, substituteAliasCref, constMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  #= Substitute in when-equation conditions =#
  local newWhenEqs = BDAE.WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local innerWhen = whenEq.whenEquation
    local (newCond, _) = Util.traverseExpTopDown(innerWhen.condition, substituteAliasCref, constMap)
    @assign innerWhen.condition = newCond
    @assign whenEq.whenEquation = innerWhen
    push!(newWhenEqs, whenEq)
  end

  #= Substitute in initial equations =#
  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION
      local (newInitExp, _) = Util.traverseExpTopDown(initEq.exp, substituteAliasCref, constMap)
      push!(newInitEqs, BDAE.RESIDUAL_EQUATION(newInitExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(initEq.lhs, substituteAliasCref, constMap)
      local (newRhs, _) = Util.traverseExpTopDown(initEq.rhs, substituteAliasCref, constMap)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    else
      push!(newInitEqs, initEq)
    end
  end

  #= Phase 3: Verify and remove eliminated unknowns =#
  local eliminatedSet = Set{String}(keys(constMap))
  local allRefNames = Set{String}()
  for eq in newResEqs
    collectCrefNames!(allRefNames, eq.exp)
  end
  for ifEq in newIfEqs
    for branch in ifEq.branches
      for brEq in branch.residualEquations
        collectCrefNames!(allRefNames, brEq.exp)
      end
    end
  end
  for initEq in newInitEqs
    if initEq isa BDAE.RESIDUAL_EQUATION
      collectCrefNames!(allRefNames, initEq.exp)
    elseif initEq isa BDAE.EQUATION
      collectCrefNames!(allRefNames, initEq.lhs)
      collectCrefNames!(allRefNames, initEq.rhs)
    end
  end

  local survivingRefs = Set{String}()
  for n in allRefNames
    if n in eliminatedSet
      push!(survivingRefs, n)
    end
  end

  if !isempty(survivingRefs)
    @warn "[SIMCODE: $(simCode.name): constantPropagation] $(length(survivingRefs)) eliminated variables still referenced, keeping them" survivingRefs=collect(survivingRefs)
  end

  local newHT = copy(ht)
  #= Build elimVarNames from elimPairs (same order as elimEqs) and
     drop any pairs whose variable is in survivingRefs. This keeps
     `eliminatedEquations` and `eliminatedVariables` aligned for
     downstream `generateEliminatedObservedBlock`. =#
  local elimVarNames = String[]
  local keptElimEqs = BDAE.RESIDUAL_EQUATION[]
  for (varName, eq) in elimPairs
    if varName in survivingRefs
      continue
    end
    if !haskey(newHT, varName)
      continue
    end
    delete!(newHT, varName)
    push!(elimVarNames, varName)
    push!(keptElimEqs, eq)
  end
  elimEqs = keptElimEqs

  @debug "[SIMCODE: $(simCode.name): constantPropagation] eliminated $(length(elimVarNames)) unknowns and $(length(allRemoved)) equations ($(length(newResEqs)) equations, $(length(newHT)) variables remain)"

  @assign simCode.residualEquations = newResEqs
  @assign simCode.initialEquations = newInitEqs
  @assign simCode.stringToSimVarHT = newHT
  @assign simCode.ifEquations = newIfEqs
  @assign simCode.whenEquations = newWhenEqs
  append!(simCode.eliminatedEquations, elimEqs)
  append!(simCode.eliminatedVariables, elimVarNames)
  return simCode
end

"""
Union-find: find with path compression.
"""
function _ufFind!(parent::Dict{String,String}, x::String)::String
  if !haskey(parent, x)
    parent[x] = x
  end
  while parent[x] != x
    parent[x] = parent[parent[x]]
    x = parent[x]
  end
  return x
end

"""
Union-find: union two elements. Returns true if they were in different sets (merged),
false if already in the same set (redundant).
"""
function _ufUnion!(parent::Dict{String,String}, a::String, b::String)::Bool
  local ra = _ufFind!(parent, a)
  local rb = _ufFind!(parent, b)
  if ra != rb
    parent[ra] = rb
    return true
  end
  return false
end

"""
    eliminateAliasVariables(simCode::SIM_CODE)::SIM_CODE

Perform alias elimination on the simulation code. Detects equations of the form
`a - b = 0` (alias) or `a + b = 0` (negated alias), builds connected components
of alias relationships, selects a representative per component, and substitutes
all eliminated variables with their representative in all equations.

This pass always runs (not opt-in) and preserves equation-unknown balance because
each eliminated equation removes exactly one variable.

Skipped for VSS/structural models where eliminated variables might be needed
in different structural modes.
"""
function eliminateAliasVariables(simCode::SIM_CODE)
  #= Guard: skip for VSS/multi-mode models (subModels or recompilation-based
     metaModel/flatModel), but allow DOCC models (structuralTransitions only)
     since they re-flatten at runtime =#
  if hasSubModels(simCode) || hasMetaModel(simCode) || hasFlatModel(simCode)
    @debug "[SIMCODE: $(simCode.name): aliasElimination] skipped (VSS/multi-mode model)"
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  local sharedVarSet = Set{String}(simCode.sharedVariables)
  local irreducibleSet = Set{String}(simCode.irreductableVariables)

  #= ===== Step 1: Detect alias equations ===== =#
  #= Each alias is (name1, name2, negated, eqIdx, cref1, ty1, cref2, ty2) =#
  local aliasPairs = Tuple{String, String, Bool, Int, DAE.ComponentRef, DAE.Type, DAE.ComponentRef, DAE.Type}[]

  for (i, eq) in enumerate(resEqs)
    local pair = detectAlias(eq.exp, ht)
    if pair !== nothing
      local (n1, n2, neg, cr1, t1, cr2, t2) = pair
      #= Skip self-loops =#
      if n1 == n2
        continue
      end
      #= Skip shared variables =#
      if n1 in sharedVarSet || n2 in sharedVarSet
        continue
      end
      push!(aliasPairs, (n1, n2, neg, i, cr1, t1, cr2, t2))
    end
  end

  if isempty(aliasPairs)
    @debug "[SIMCODE: $(simCode.name): aliasElimination] no alias equations found"
    return simCode
  end

  @debug "[SIMCODE: $(simCode.name): aliasElimination] detected $(length(aliasPairs)) alias equations"

  #= ===== Step 2: Build alias graph and find connected components via BFS ===== =#
  #= Adjacency list: varName -> [(neighborName, negated, edgeIdx)] =#
  local adjList = Dict{String, Vector{Tuple{String, Bool, Int}}}()
  for (idx, (n1, n2, neg, eqIdx, _, _, _, _)) in enumerate(aliasPairs)
    if !haskey(adjList, n1)
      adjList[n1] = Tuple{String, Bool, Int}[]
    end
    if !haskey(adjList, n2)
      adjList[n2] = Tuple{String, Bool, Int}[]
    end
    push!(adjList[n1], (n2, neg, idx))
    push!(adjList[n2], (n1, neg, idx))
  end

  #= BFS to find connected components with cumulative negation =#
  #= componentId -> [(varName, negationRelativeToRoot)] =#
  local visited = Dict{String, Bool}()  #= varName -> negation relative to component root =#
  local components = Vector{Vector{Tuple{String, Bool}}}()
  local componentEqs = Vector{Vector{Int}}()  #= equation indices per component =#
  local usedEdges = Set{Int}()

  for startNode in keys(adjList)
    if haskey(visited, startNode)
      continue
    end
    local component = Tuple{String, Bool}[]
    local compEqs = Int[]
    local queue = [(startNode, false)]  #= (name, negRelToRoot) =#
    visited[startNode] = false
    while !isempty(queue)
      local (node, negFromRoot) = popfirst!(queue)
      push!(component, (node, negFromRoot))
      if haskey(adjList, node)
        for (neighbor, edgeNeg, edgeIdx) in adjList[node]
          if !(edgeIdx in usedEdges)
            push!(usedEdges, edgeIdx)
            push!(compEqs, aliasPairs[edgeIdx][4])  #= equation index =#
          end
          if !haskey(visited, neighbor)
            local neighborNeg = xor(negFromRoot, edgeNeg)
            visited[neighbor] = neighborNeg
            push!(queue, (neighbor, neighborNeg))
          end
        end
      end
    end
    push!(components, component)
    push!(componentEqs, compEqs)
  end

  #= ===== Step 3: Select representative per component ===== =#
  #= Build alias resolution map and alias entries =#
  local aliasMap = Dict{String, Tuple{String, Bool, DAE.ComponentRef, DAE.Type}}()
  local aliasEntries = AliasEntry[]
  local aliasEqIndices = Set{Int}()

  #= Build name -> (cref, type) lookup from alias pairs =#
  local nameToCrefType = Dict{String, Tuple{DAE.ComponentRef, DAE.Type}}()
  for (n1, n2, _, _, cr1, t1, cr2, t2) in aliasPairs
    nameToCrefType[n1] = (cr1, t1)
    nameToCrefType[n2] = (cr2, t2)
  end

  #= Map equation index -> (n1, n2) for deciding which equations are trivial after substitution =#
  local eqIdxToNames = Dict{Int, Tuple{String, String}}()
  for (n1, n2, _, eqIdx, _, _, _, _) in aliasPairs
    eqIdxToNames[eqIdx] = (n1, n2)
  end

  for (compIdx, component) in enumerate(components)
    #= Select representative: highest priority varKind, with ties broken by irreducibility =#
    local bestName = ""
    local bestPriority = -1
    local bestNeg = false
    for (varName, negFromRoot) in component
      if !haskey(ht, varName)
        continue
      end
      local (_, sv) = ht[varName]
      local prio = varKindPriority(sv.varKind)
      #= Boost priority for irreducible variables =#
      if varName in irreducibleSet
        prio += 60
      end
      #= Boost priority for variables with explicit start attribute so the
         representative carries the start binding instead of defaulting to 0. =#
      local hasStart = @match sv.attributes begin
        SOME(DAE.VAR_ATTR_REAL(start = SOME(_))) => true
        SOME(DAE.VAR_ATTR_INT(start = SOME(_)))  => true
        SOME(DAE.VAR_ATTR_BOOL(start = SOME(_))) => true
        _                                        => false
      end
      if hasStart
        prio += 5
      end
      if prio > bestPriority
        bestPriority = prio
        bestName = varName
        bestNeg = negFromRoot
      end
    end

    if isempty(bestName)
      continue
    end

    #= Get representative CREF and type =#
    if !haskey(nameToCrefType, bestName)
      continue
    end
    local (repCref, repTy) = nameToCrefType[bestName]
    local (_, bestSv) = ht[bestName]
    local bestIsState = @match bestSv.varKind begin
      STATE(__) => true
      _ => false
    end

    #= Mark all other variables in this component for elimination.
       Never eliminate irreducible variables (involved in events).
       Exception: state-to-state aliases inside the same component are safe to
       collapse even when both ends are flagged irreducible — `getIrreductableVars`
       marks every STATE as irreducible by default, which prevents two states that
       are connected via algebraic-flange aliases (e.g. AIMC `aimc_inertiaRotor_phi`
       and `loadInertia_phi`) from being merged. Without merging, the residual
       `loadInertia_phi - aimc_inertiaRotor_phi = 0` survives and MTK Pantelides
       sees the system as over-determined. =#
    for (varName, negFromRoot) in component
      if varName == bestName
        continue
      end
      if !haskey(ht, varName)
        continue
      end
      local (_, sv) = ht[varName]
      local isState = @match sv.varKind begin
        STATE(__) => true
        _ => false
      end
      if varName in irreducibleSet && !(bestIsState && isState)
        continue
      end
      local negated = xor(negFromRoot, bestNeg)
      aliasMap[varName] = (bestName, negated, repCref, repTy)
      push!(aliasEntries, AliasEntry(varName, bestName, negated))
    end

    #= Mark equations for removal using union-find on surviving variables.
       Trivial equations (both sides resolve to same variable) are always removed.
       Among meaningful equations, only keep enough to span the surviving variables
       (union-find ensures a spanning tree). Redundant equations are removed. =#
    local ufParent = Dict{String,String}()
    for eqIdx in componentEqs[compIdx]
      local (n1, n2) = eqIdxToNames[eqIdx]
      local r1 = haskey(aliasMap, n1) ? aliasMap[n1][1] : n1
      local r2 = haskey(aliasMap, n2) ? aliasMap[n2][1] : n2
      if r1 == r2
        #= Trivial: both sides resolve to same variable (0 = 0). Remove. =#
        push!(aliasEqIndices, eqIdx)
      elseif _ufUnion!(ufParent, r1, r2)
        #= Non-redundant constraint between surviving variables. Keep. =#
      else
        #= Redundant: surviving variables already connected. Remove. =#
        push!(aliasEqIndices, eqIdx)
      end
    end
  end

  if isempty(aliasMap)
    @debug "[SIMCODE: $(simCode.name): aliasElimination] no variables could be eliminated"
    return simCode
  end

  #= ===== Step 4: Substitute alias CREFs in all remaining equations ===== =#
  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  local elimEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newResEqs, nEqs - length(aliasEqIndices))

  for (i, eq) in enumerate(resEqs)
    if i in aliasEqIndices
      push!(elimEqs, eq)
    else
      local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteAliasCref, aliasMap)
      push!(newResEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
    end
  end

  #= Also substitute in if-equation branches =#
  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = BDAE.RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = Util.traverseExpTopDown(brEq.exp, substituteAliasCref, aliasMap)
        push!(newBranchEqs, BDAE.RESIDUAL_EQUATION(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = Util.traverseExpTopDown(branch.condition, substituteAliasCref, aliasMap)
      #= Reconstruct BRANCH with substituted expressions but same structural info =#
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  #= When equations: substitute alias CREFs in conditions AND statements =#
  local newWhenEqs = BDAE.WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local innerWhen = _substituteAliasInWhenStmts(whenEq.whenEquation, aliasMap)
    @assign whenEq.whenEquation = innerWhen
    push!(newWhenEqs, whenEq)
  end

  #= Initial equations: substitute alias CREFs.
     initialEquations may contain EQUATION (lhs/rhs) or RESIDUAL_EQUATION (exp). =#
  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION
      local (newInitExp, _) = Util.traverseExpTopDown(initEq.exp, substituteAliasCref, aliasMap)
      push!(newInitEqs, BDAE.RESIDUAL_EQUATION(newInitExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(initEq.lhs, substituteAliasCref, aliasMap)
      local (newRhs, _) = Util.traverseExpTopDown(initEq.rhs, substituteAliasCref, aliasMap)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    else
      push!(newInitEqs, initEq)
    end
  end

  #= ===== Step 5: Verify substitution and remove eliminated variables ===== =#
  #= Collect all CREF names from remaining equations. Any eliminated variable
     still referenced means the substitution missed it (e.g. unflatten CREF form).
     Those variables must be kept in the HT to avoid KeyError during code gen. =#
  local eliminatedSet = Set{String}(keys(aliasMap))
  local survivingRefs = Set{String}()
  local allRefNames = Set{String}()
  for eq in newResEqs
    collectCrefNames!(allRefNames, eq.exp)
  end
  for ifEq in newIfEqs
    for branch in ifEq.branches
      for brEq in branch.residualEquations
        collectCrefNames!(allRefNames, brEq.exp)
      end
    end
  end
  for initEq in newInitEqs
    if initEq isa BDAE.RESIDUAL_EQUATION
      collectCrefNames!(allRefNames, initEq.exp)
    elseif initEq isa BDAE.EQUATION
      collectCrefNames!(allRefNames, initEq.lhs)
      collectCrefNames!(allRefNames, initEq.rhs)
    end
  end
  #= Also check when-equations (conditions and statements) for surviving references =#
  for whenEq in newWhenEqs
    _collectWhenCrefNames!(allRefNames, whenEq.whenEquation)
  end
  for n in allRefNames
    if n in eliminatedSet
      push!(survivingRefs, n)
    end
  end

  if !isempty(survivingRefs)
    @warn "[SIMCODE: $(simCode.name): aliasElimination] $(length(survivingRefs)) eliminated variables still referenced, keeping them" survivingRefs=collect(survivingRefs)
  end

  #= Remove only safely eliminated variables from hash table =#
  local newHT = copy(ht)
  local elimVarNames = String[]
  local keptAliasEntries = AliasEntry[]
  for (varName, _) in aliasMap
    if varName in survivingRefs
      #= Keep this variable: still referenced in equations =#
      continue
    end
    delete!(newHT, varName)
    push!(elimVarNames, varName)
  end
  #= Filter alias entries to only include actually eliminated variables =#
  for entry in aliasEntries
    if !(entry.eliminatedName in survivingRefs)
      push!(keptAliasEntries, entry)
    end
  end

  #= Build parallel eliminated-variable/equation metadata. aliasEqIndices may
     contain redundant alias equations that were removed because they add no new
     constraint after substitution; those equations do not correspond to a
     removed variable and must not be appended to eliminatedEquations. =#
  local elimVarSet = Set{String}(elimVarNames)
  local removedAliasIncidence = Tuple{Int, String, String}[]
  for (n1, n2, _, eqIdx, _, _, _, _) in aliasPairs
    if eqIdx in aliasEqIndices && (n1 in elimVarSet || n2 in elimVarSet)
      push!(removedAliasIncidence, (eqIdx, n1, n2))
    end
  end

  local eqByElimVar = Dict{String, Int}()
  local varByElimEq = Dict{Int, String}()
  function assignElimEq!(varName::String, seenEqIdxs::Set{Int})::Bool
    for (eqIdx, n1, n2) in removedAliasIncidence
      if n1 != varName && n2 != varName
        continue
      end
      if eqIdx in seenEqIdxs
        continue
      end
      push!(seenEqIdxs, eqIdx)
      if !haskey(varByElimEq, eqIdx) || assignElimEq!(varByElimEq[eqIdx], seenEqIdxs)
        varByElimEq[eqIdx] = varName
        eqByElimVar[varName] = eqIdx
        return true
      end
    end
    return false
  end

  for varName in elimVarNames
    assignElimEq!(varName, Set{Int}())
  end

  local pairedElimVarNames = String[]
  local pairedElimEqs = BDAE.RESIDUAL_EQUATION[]
  for varName in elimVarNames
    if haskey(eqByElimVar, varName)
      push!(pairedElimVarNames, varName)
      push!(pairedElimEqs, resEqs[eqByElimVar[varName]])
    end
  end
  if length(pairedElimVarNames) != length(elimVarNames)
    local unpairedVars = setdiff(elimVarNames, pairedElimVarNames)
    @info "[SIMCODE: $(simCode.name): aliasElimination] could not pair all eliminated variables with removed alias equations" unpaired=unpairedVars
    #= Fallback: synthesise an identity observation for each unpaired eliminated variable.
       This happens when the alias equation for the eliminated variable was kept as a
       non-trivial constraint between surviving variables (e.g. because the other side is
       irreducible). The variable's aliasMap entry gives us the direct assignment. =#
    for uv in unpairedVars
      if haskey(aliasMap, uv) && haskey(nameToCrefType, uv)
        local (repName, negated, repCref, repTy) = aliasMap[uv]
        local (uvCref, uvTy) = nameToCrefType[uv]
        local uvExp  = DAE.CREF(uvCref, uvTy)
        local repExp = DAE.CREF(repCref, repTy)
        #= 0 = uv - rep  (positive alias)  or  0 = uv + rep  (negated alias) =#
        local synExp = negated ?
          DAE.BINARY(uvExp, DAE.ADD(DAE.T_REAL_DEFAULT), repExp) :
          DAE.BINARY(uvExp, DAE.SUB(DAE.T_REAL_DEFAULT), repExp)
        push!(pairedElimVarNames, uv)
        push!(pairedElimEqs, BDAE.RESIDUAL_EQUATION(synExp, nothing, nothing))
      end
    end
  end

  @debug "[SIMCODE: $(simCode.name): aliasElimination] eliminated $(length(elimVarNames)) variables and removed $(length(aliasEqIndices)) equations ($(length(pairedElimVarNames)) paired for observation, $(length(newResEqs)) equations, $(length(newHT)) variables remain)"

  @assign simCode.residualEquations = newResEqs
  @assign simCode.initialEquations = newInitEqs
  @assign simCode.stringToSimVarHT = newHT
  @assign simCode.ifEquations = newIfEqs
  @assign simCode.whenEquations = newWhenEqs
  @assign simCode.aliasMap = keptAliasEntries
  #= State-state aliases collapse two STATEs marked irreducible into one.
     Drop the eliminated names from `irreductableVariables` so MTK codegen's
     start-condition lookup (`getStartConditionsMTK`) doesn't try to look up
     a name that no longer exists in `stringToSimVarHT`. =#
  local elimVarSet = Set{String}(elimVarNames)
  @assign simCode.irreductableVariables = filter(n -> !(n in elimVarSet), simCode.irreductableVariables)
  #= Append eliminated equations/variables to the existing lists =#
  append!(simCode.eliminatedEquations, pairedElimEqs)
  append!(simCode.eliminatedVariables, pairedElimVarNames)
  return simCode
end

"""
    eliminateConstantParameters(simCode::SIM_CODE) -> SIM_CODE

Find every PARAMETER whose binding evaluates to a numeric/Bool literal,
substitute the literal value at all use sites, and drop the parameter from
`stringToSimVarHT`. This shrinks the parameter list MTK sees before
`structural_simplify`, reducing per-simulate module-eval cost on large MSL
models (where `foldParameterClosure` typically inflates the parameter count
2x to 3x).

Tier-1 only: skipped on VSS / DOCC / sub-model / flat-model variants because
a parameter eliminated here can no longer be re-bound at runtime by a
structural transition or by recompilation. The gate matches the
conservative envelope used by `eliminateAliasVariables`.

Defensive checks:
- Parameters that appear as representatives in `aliasMap` are NOT eliminated
  (would orphan the alias entry).
- A survivor scan after substitution keeps any parameter still referenced
  somewhere the substitution missed (paranoia for unflatten CREF forms).
"""
#= For every DAE.CREF with T_COMPLEX type in the given equations, append
   `<base>_<fieldname>` for each field of the complex record when that scalar
   name exists in the simvar hash table. Used to protect those scalar params
   from constant-elimination — codegen later flattens the complex CREF into
   the scalar field symbols, which must resolve at module eval time. =#
function _collectComplexFieldNames!(names::Set{String}, eqs, ht)
  function visitor(exp, ctx)
    @match exp begin
      DAE.CREF(cr, ty) => begin
        local baseName = DAE_identifierToString(cr)
        if ty isa DAE.T_COMPLEX
          for field in ty.varLst
            local fname = field.name
            local fieldName = Base.string(baseName, "_", fname)
            if haskey(ht, fieldName)
              push!(names, fieldName)
            end
          end
        else
          #= Fallback: any cref X whose X_re and X_im scalars exist in HT.
             Codegen will flatten X via flattenRecordCallArg into [X_re, X_im];
             protect both even when the cref's ty was downgraded from T_COMPLEX. =#
          local reName = Base.string(baseName, "_re")
          local imName = Base.string(baseName, "_im")
          if haskey(ht, reName) && haskey(ht, imName)
            push!(names, reName)
            push!(names, imName)
          end
        end
      end
      _ => nothing
    end
    return (exp, true, ctx)
  end
  for eq in eqs
    local exps = if eq isa BDAE.RESIDUAL_EQUATION
      DAE.Exp[eq.exp]
    elseif eq isa BDAE.EQUATION
      DAE.Exp[eq.lhs, eq.rhs]
    elseif eq isa BDAE.COMPLEX_EQUATION || eq isa BDAE.ARRAY_EQUATION
      DAE.Exp[eq.left, eq.right]
    else
      DAE.Exp[]
    end
    for e in exps
      Util.traverseExpTopDown(e, visitor, nothing)
    end
  end
  return names
end

"""
    eliminateDeadParameters(simCode) -> simCode

Remove `PARAMETER(NONE)` simvars that are not referenced anywhere — no
residual, no initial equation, no if-condition, no when statement, no
DATA_STRUCTURE / parameter binding expression, no alias representative, no
attribute (`start` / `fixed` / `min` / `max` / `nominal`), no eliminated
equation. Such parameters cannot be observed and cannot be overridden
meaningfully at runtime (no consumer would see the override).

Skipped for sub-model / flatModel / metaModel variants because cross-mode
parameter references are not visible in the standard scan.
"""
function eliminateDeadParameters(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode) || hasFlatModel(simCode)
    return simCode
  end
  local ht = simCode.stringToSimVarHT

  #= Reachability scan: collect every cref name referenced from a live
     surface. Anything not in this set is dead. =#
  local referenced = Set{String}()
  for eq in simCode.residualEquations
    collectCrefNames!(referenced, eq.exp)
  end
  for eq in simCode.initialEquations
    if eq isa BDAE.RESIDUAL_EQUATION
      collectCrefNames!(referenced, eq.exp)
    elseif eq isa BDAE.EQUATION
      collectCrefNames!(referenced, eq.lhs)
      collectCrefNames!(referenced, eq.rhs)
    end
  end
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(referenced, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(referenced, brEq.exp)
      end
    end
  end
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(referenced, whenEq.whenEquation)
  end
  for eq in simCode.eliminatedEquations
    collectCrefNames!(referenced, eq.exp)
  end
  for entry in simCode.aliasMap
    push!(referenced, entry.representativeName)
    push!(referenced, entry.eliminatedName)
  end
  _collectAttributeCrefs!(referenced, ht)
  for (_n, (_, sv)) in ht
    @match sv.varKind begin
      PARAMETER(SOME(b)) => collectCrefNames!(referenced, b)
      ARRAY_PARAMETER(_, SOME(b)) => collectCrefNames!(referenced, b)
      DATA_STRUCTURE(SOME(b)) => collectCrefNames!(referenced, b)
      _ => nothing
    end
  end

  #= Sweep: drop entries where the var is PARAMETER(NONE) and unreferenced.
     The broader PARAMETER(SOME(_)) case was tested and triggered a
     UndefVarError on SimpleMechanicalSystem (a parameter referenced from
     a Modelica function body in `simCode.functions`, which the reachability
     scan above does not cover). Keep the narrow form. =#
  local toDrop = String[]
  for (name, (_, sv)) in ht
    name in referenced && continue
    local isUnboundParam = @match sv.varKind begin
      PARAMETER(NONE()) => true
      _ => false
    end
    isUnboundParam && push!(toDrop, name)
  end

  isempty(toDrop) && return simCode

  local newHT = copy(ht)
  for name in toDrop
    delete!(newHT, name)
  end
  @assign simCode.stringToSimVarHT = newHT
  @info "[SIMCODE: $(simCode.name): eliminateDeadParameters] dropped $(length(toDrop)) unbound / unused parameters"
  return simCode
end

function eliminateConstantParameters(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode) ||
     hasFlatModel(simCode) || hasMetaModel(simCode)
    @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] skipped (VSS/recompilation/sub-model variant)"
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  local paramValueMap = Dict{String, Float64}()
  local seen = Set{String}()

  #= Build the protected-from-elimination set. We keep any parameter that:
     1. Is an alias representative (eliminating orphans the alias entry).
     2. Is referenced as a CREF in another simvar's `start`/`fixed`/`min`/
        `max`/`nominal` attribute. The MTK codegen short-circuits start
        attributes via `pars[Symbol(name)]`, bypassing the equation
        substitution map; eliminating such a parameter produces a runtime
        UndefVarError when the model module evaluates.
     3. Is referenced as a condition in any IF_EQUATION branch — these are
        structural switches the user may want to flip.
     4. Is referenced as a condition in any WHEN_EQUATION.
     5. Is referenced as a condition in any IFEXP, anywhere in equations or
        in another parameter's binding.
     6. Is referenced anywhere in any initial equation. Initial equations
        carry constraints MTK uses at t=0; we keep their parameter inputs
        intact so the user can re-bind a parameter and re-initialize without
        a recompile (where supported by MTK). =#
  local protectedNames = Set{String}()
  for entry in simCode.aliasMap
    push!(protectedNames, entry.representativeName)
  end
  _collectAttributeCrefs!(protectedNames, ht)
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(protectedNames, branch.condition)
    end
  end
  for whenEq in simCode.whenEquations
    _collectWhenConditionCrefs!(protectedNames, whenEq.whenEquation)
  end
  for eq in simCode.initialEquations
    if eq isa BDAE.RESIDUAL_EQUATION
      collectCrefNames!(protectedNames, eq.exp)
    elseif eq isa BDAE.EQUATION
      collectCrefNames!(protectedNames, eq.lhs)
      collectCrefNames!(protectedNames, eq.rhs)
    end
  end
  #= IFEXP conditions inside residual equations and parameter bindings. =#
  for eq in simCode.residualEquations
    _collectIfexpConditionCrefs!(protectedNames, eq.exp)
  end
  #= Names of array bases referenced as bare CREFs in DATA_STRUCTURE constructor
     calls (ExternalObject inits like CombiTable / CombiTimeTable). Array params
     are scalarized into HT entries like `tableData[1][1]`..., but the constructor
     call bind references the whole array (`tableData`). Eliminating any
     scalarized element would leave the constructor referring to data that no
     longer survives codegen, so protect every scalar element of those arrays.

     Restricted to DS bindings whose RHS is a CALL — MSL constants
     (BDAE.CONST of scalar type) are also stored as DATA_STRUCTURE but their
     RHS is a literal and over-protecting them would block legitimate
     constant-propagation eliminations elsewhere. =#
  local dsArrayBaseNames = Set{String}()
  for (_, htEntry) in ht
    local (_, svP) = htEntry
    @match svP.varKind begin
      PARAMETER(SOME(b))            => _collectIfexpConditionCrefs!(protectedNames, b)
      ARRAY_PARAMETER(_, SOME(b))   => _collectIfexpConditionCrefs!(protectedNames, b)
      DATA_STRUCTURE(SOME(b)) => begin
        @match b begin
          DAE.CALL(__) => begin
            collectCrefNames!(protectedNames, b)
            collectCrefNames!(dsArrayBaseNames, b)
          end
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  for htKey in keys(ht)
    local bracketIdx = findfirst('[', htKey)
    bracketIdx === nothing && continue
    local baseName = htKey[1:bracketIdx-1]
    if baseName in dsArrayBaseNames
      push!(protectedNames, htKey)
    end
  end

  #= Protect scalar field params backing complex CREFs that survive in equations.
     Magnetic.QuasiStationary models reference `converter_m_N` (T_COMPLEX) in
     residual equations; codegen flattens this to `[converter_m_N_re,
     converter_m_N_im]` symbols. If those scalar fields are constant params
     they get eliminated here, but the flatten happens later and looks them up
     by symbol — UndefVarError at module eval. =#
  _collectComplexFieldNames!(protectedNames, simCode.residualEquations, ht)
  _collectComplexFieldNames!(protectedNames, simCode.initialEquations, ht)

  #= Step 1: identify eliminable parameters via _tryEvalNumeric. =#
  for (name, htEntry) in ht
    name in protectedNames && continue
    local (_, sv) = htEntry
    local bindExp = @match sv.varKind begin
      PARAMETER(SOME(e)) => e
      _ => nothing
    end
    bindExp === nothing && continue
    empty!(seen)
    local v = _tryEvalNumeric(bindExp, simCode, seen)
    v === nothing && continue
    paramValueMap[name] = v
  end

  # Enumerate ARRAY_PARAMETER element bindings; iterate to fixed point so
  # chained array references resolve in dependency order.
  local arrChanged = true
  while arrChanged
    arrChanged = false
    local mapSizeBefore = length(paramValueMap)
    for (name, htEntry) in ht
      name in protectedNames && continue
      local (_, sv) = htEntry
      local arrBind = @match sv.varKind begin
        ARRAY_PARAMETER(_, SOME(e)) => e
        _ => nothing
      end
      arrBind === nothing && continue
      _enumerateArrayParamElements!(paramValueMap, name, arrBind, simCode, seen, protectedNames)
    end
    arrChanged = length(paramValueMap) > mapSizeBefore
  end

  if isempty(paramValueMap)
    @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] no eliminable parameters found"
    return simCode
  end

  #= Step 2: substitute throughout every equation container. =#
  local newResiduals = BDAE.RESIDUAL_EQUATION[]
  for eq in simCode.residualEquations
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteConstantParameter, paramValueMap)
    push!(newResiduals, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end

  local newInitials = typeof(simCode.initialEquations)()
  for eq in simCode.initialEquations
    local newEq = if eq isa BDAE.RESIDUAL_EQUATION
      local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteConstantParameter, paramValueMap)
      BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr)
    elseif eq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(eq.lhs, substituteConstantParameter, paramValueMap)
      local (newRhs, _) = Util.traverseExpTopDown(eq.rhs, substituteConstantParameter, paramValueMap)
      BDAE.EQUATION(newLhs, newRhs, eq.source, eq.attributes)
    else
      eq
    end
    push!(newInitials, newEq)
  end

  local newIfEquations = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = BDAE.RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = Util.traverseExpTopDown(brEq.exp, substituteConstantParameter, paramValueMap)
        push!(newBranchEqs, BDAE.RESIDUAL_EQUATION(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = Util.traverseExpTopDown(branch.condition, substituteConstantParameter, paramValueMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEquations, IF_EQUATION(newBranches))
  end

  local newWhenEquations = BDAE.WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local newInner = _substituteParamInWhenStmts(whenEq.whenEquation, paramValueMap)
    @assign whenEq.whenEquation = newInner
    push!(newWhenEquations, whenEq)
  end

  # alias-eliminated residuals are emitted verbatim by codegen; substitute
  # eliminated-parameter element refs to avoid dangling identifiers
  local newElimEqs = BDAE.RESIDUAL_EQUATION[]
  for eq in simCode.eliminatedEquations
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteConstantParameter, paramValueMap)
    push!(newElimEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end

  # substitute into surviving PARAMETER and ARRAY_PARAMETER bindings
  local newHT = copy(ht)
  for (name, htEntry) in ht
    haskey(paramValueMap, name) && continue
    local (idx, sv) = htEntry
    local newKind = @match sv.varKind begin
      PARAMETER(SOME(b)) => begin
        local (nb, _) = Util.traverseExpTopDown(b, substituteConstantParameter, paramValueMap)
        nb === b ? sv.varKind : PARAMETER(SOME(nb))
      end
      ARRAY_PARAMETER(dims, SOME(b)) => begin
        local (nb, _) = Util.traverseExpTopDown(b, substituteConstantParameter, paramValueMap)
        nb === b ? sv.varKind : ARRAY_PARAMETER(dims, SOME(nb))
      end
      _ => sv.varKind
    end
    if newKind !== sv.varKind
      newHT[name] = (idx, SIMVAR(sv.name, sv.index, newKind, sv.attributes))
    end
  end

  #= Step 4: defensive survivor scan. If a CREF for a candidate parameter
     somehow survived substitution (unflatten form, etc.), keep the param. =#
  local survivorCheck = Set{String}()
  for eq in newResiduals
    collectCrefNames!(survivorCheck, eq.exp)
  end
  for eq in newInitials
    if eq isa BDAE.RESIDUAL_EQUATION
      collectCrefNames!(survivorCheck, eq.exp)
    elseif eq isa BDAE.EQUATION
      collectCrefNames!(survivorCheck, eq.lhs)
      collectCrefNames!(survivorCheck, eq.rhs)
    end
  end
  for ifEq in newIfEquations
    for branch in ifEq.branches
      for brEq in branch.residualEquations
        collectCrefNames!(survivorCheck, brEq.exp)
      end
      collectCrefNames!(survivorCheck, branch.condition)
    end
  end
  for whenEq in newWhenEquations
    _collectWhenCrefNames!(survivorCheck, whenEq.whenEquation)
  end

  #= Step 5: drop eliminated params from HT, skipping survivors. =#
  local elimNames = String[]
  local survivors = String[]
  for (name, _) in paramValueMap
    if name in survivorCheck
      push!(survivors, name)
      continue
    end
    delete!(newHT, name)
    push!(elimNames, name)
  end

  if !isempty(survivors)
    @warn "[SIMCODE: $(simCode.name): eliminateConstantParameters] $(length(survivors)) parameters still referenced after substitution; keeping them" survivors
  end

  if isempty(elimNames)
    @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] nothing eliminated (all candidates survived substitution)"
    return simCode
  end

  @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] eliminated $(length(elimNames)) parameters of $(length(paramValueMap)) candidates"

  @assign simCode.residualEquations = newResiduals
  @assign simCode.initialEquations = newInitials
  @assign simCode.ifEquations = newIfEquations
  @assign simCode.whenEquations = newWhenEquations
  @assign simCode.eliminatedEquations = newElimEqs
  @assign simCode.stringToSimVarHT = newHT
  #= Do NOT append eliminated parameter names to `simCode.eliminatedVariables`.
     That list pairs with `simCode.eliminatedEquations` 1:1 and is consumed by
     `generateEliminatedObservedBlock`, which expects each eliminated name to
     have a defining residual equation. Parameters are substituted directly
     into equations and have no residual to reconstruct, so adding them breaks
     the parallel-array invariant. =#
  return simCode
end

"""
Collect every CREF appearing in a CREF-valued attribute (`start`, `fixed`,
`min`, `max`, `nominal`) of any simvar in `ht`. These names must not be
eliminated — the MTK start-condition codegen references them via
`pars[Symbol(name)]`, which bypasses equation-level substitution.
"""
function _collectAttributeCrefs!(out::Set{String}, ht::AbstractDict)
  for (_, htEntry) in ht
    local (_, sv) = htEntry
    local optAttrs = sv.attributes
    @match optAttrs begin
      SOME(attrs) => begin
        for fname in (:start, :fixed, :min, :max, :nominal)
          if hasproperty(attrs, fname)
            local fv = getproperty(attrs, fname)
            @match fv begin
              SOME(e) => collectCrefNames!(out, e)
              _ => nothing
            end
          end
        end
      end
      _ => nothing
    end
  end
  return out
end

"""
Collect every CREF appearing in an IFEXP condition anywhere in `exp`. CREFs
appearing only in IFEXP branches (`then`/`else`) are NOT collected. Used to
protect parameters that gate runtime conditional branches from elimination.
"""
function _collectIfexpConditionCrefs!(out::Set{String}, @nospecialize(exp))
  @match exp begin
    DAE.IFEXP(expCond = c, expThen = t, expElse = e) => begin
      collectCrefNames!(out, c)
      _collectIfexpConditionCrefs!(out, t)
      _collectIfexpConditionCrefs!(out, e)
    end
    DAE.BINARY(exp1 = e1, exp2 = e2) => begin
      _collectIfexpConditionCrefs!(out, e1)
      _collectIfexpConditionCrefs!(out, e2)
    end
    DAE.UNARY(exp = e1)        => _collectIfexpConditionCrefs!(out, e1)
    DAE.LUNARY(exp = e1)       => _collectIfexpConditionCrefs!(out, e1)
    DAE.LBINARY(exp1 = e1, exp2 = e2) => begin
      _collectIfexpConditionCrefs!(out, e1)
      _collectIfexpConditionCrefs!(out, e2)
    end
    DAE.RELATION(exp1 = e1, exp2 = e2) => begin
      _collectIfexpConditionCrefs!(out, e1)
      _collectIfexpConditionCrefs!(out, e2)
    end
    DAE.CALL(expLst = args) => begin
      for arg in args
        _collectIfexpConditionCrefs!(out, arg)
      end
    end
    DAE.ARRAY(array = lst) => begin
      for e in lst
        _collectIfexpConditionCrefs!(out, e)
      end
    end
    DAE.ASUB(exp = e, sub = subs) => begin
      _collectIfexpConditionCrefs!(out, e)
      for s in subs
        _collectIfexpConditionCrefs!(out, s)
      end
    end
    DAE.CAST(exp = e1) => _collectIfexpConditionCrefs!(out, e1)
    _ => nothing
  end
  return out
end

"""
Collect CREFs in the condition of a `BDAE.WHEN_STMTS` (and any nested
`elsewhen`). Statements inside the when-clause are handled separately via
the equation walk; we only protect parameters that gate the trigger.
"""
function _collectWhenConditionCrefs!(out::Set{String}, whenStmts::BDAE.WHEN_STMTS)
  collectCrefNames!(out, whenStmts.condition)
  @match whenStmts.elsewhenPart begin
    SOME(inner) => _collectWhenConditionCrefs!(out, inner)
    _ => nothing
  end
  return out
end

function _collectWhenConditionCrefs!(out::Set{String}, whenEq::BDAE.WHEN_EQUATION)
  return _collectWhenConditionCrefs!(out, whenEq.whenEquation)
end

# Walk a DAE.ARRAY binding and add one paramValueMap entry per numeric element.
function _enumerateArrayParamElements!(paramValueMap, baseName::String,
                                       exp, simCode,
                                       seen::Set{String},
                                       protectedNames::Set{String})
  exp isa DAE.ARRAY || return nothing
  local i = 0
  for elem in exp.array
    i += 1
    local elemName = Base.string(baseName, "[", i, "]")
    elemName in protectedNames && continue
    if elem isa DAE.ARRAY
      _enumerateArrayParamElements!(paramValueMap, elemName, elem, simCode,
                                    seen, protectedNames)
    else
      empty!(seen)
      local v = _tryEvalNumeric(elem, simCode, seen)
      # fall back to map lookup when the element binding is a CREF/ASUB
      # to a previously-enumerated array element
      if v === nothing
        local refName = _asubCanonicalName(elem)
        if refName !== nothing && haskey(paramValueMap, refName)
          v = paramValueMap[refName]
        end
      end
      v !== nothing && (paramValueMap[elemName] = v)
    end
  end
  return nothing
end

# Canonical name for a (possibly nested) DAE.ASUB; nothing if non-constant.
function _asubCanonicalName(@nospecialize(exp))::Union{Nothing,String}
  @match exp begin
    DAE.CREF(cr, _) => string(cr)
    DAE.ASUB(inner, subs) => begin
      local innerName = _asubCanonicalName(inner)
      innerName === nothing && return nothing
      local idxParts = String[]
      for s in subs
        local v = @match s begin
          DAE.ICONST(i) => i
          DAE.RCONST(r) where r == round(r) => Int(round(r))
          _ => nothing
        end
        v === nothing && return nothing
        push!(idxParts, Base.string("[", v, "]"))
      end
      Base.string(innerName, idxParts...)
    end
    _ => nothing
  end
end

function substituteConstantParameter(@nospecialize(exp), paramValueMap)
  @match exp begin
    DAE.CREF(cr, ty) => begin
      local name = string(cr)
      if haskey(paramValueMap, name)
        local v = paramValueMap[name]
        local literalExp = @match ty begin
          DAE.T_REAL(__)    => DAE.RCONST(v)
          DAE.T_INTEGER(__) => DAE.ICONST(Int(round(v)))
          DAE.T_BOOL(__)    => DAE.BCONST(v != 0.0)
          _                 => DAE.RCONST(v)
        end
        return (literalExp, false, paramValueMap)
      end
      (exp, true, paramValueMap)
    end
    DAE.ASUB(__) => begin
      local name = _asubCanonicalName(exp)
      if name !== nothing && haskey(paramValueMap, name)
        local v = paramValueMap[name]
        return (DAE.RCONST(v), false, paramValueMap)
      end
      (exp, true, paramValueMap)
    end
    _ => (exp, true, paramValueMap)
  end
end

"""
Recursively substitute eliminated-parameter CREFs in a WHEN_STMTS node.
Mirrors `_substituteAliasInWhenStmts` but with `substituteConstantParameter`.
"""
function _substituteParamInWhenStmts(whenStmts::BDAE.WHEN_STMTS, paramValueMap)
  local (newCond, _) = Util.traverseExpTopDown(whenStmts.condition, substituteConstantParameter, paramValueMap)
  local newStmtLst::List{BDAE.WhenOperator} = MetaModelica.nil
  for stmt in whenStmts.whenStmtLst
    local newStmt::BDAE.WhenOperator = @match stmt begin
      BDAE.ASSIGN(__) => begin
        local (newL, _) = Util.traverseExpTopDown(stmt.left, substituteConstantParameter, paramValueMap)
        local (newR, _) = Util.traverseExpTopDown(stmt.right, substituteConstantParameter, paramValueMap)
        BDAE.ASSIGN(newL, newR, stmt.source)
      end
      BDAE.REINIT(__) => begin
        local (newSV, _) = Util.traverseExpTopDown(stmt.stateVar, substituteConstantParameter, paramValueMap)
        local (newVal, _) = Util.traverseExpTopDown(stmt.value, substituteConstantParameter, paramValueMap)
        BDAE.REINIT(newSV, newVal, stmt.source)
      end
      BDAE.NORETCALL(__) => begin
        local (newExp, _) = Util.traverseExpTopDown(stmt.exp, substituteConstantParameter, paramValueMap)
        BDAE.NORETCALL(newExp, stmt.source)
      end
      _ => stmt
    end
    newStmtLst = MetaModelica.Cons{BDAE.WhenOperator}(newStmt, newStmtLst)
  end
  newStmtLst = MetaModelica.listReverse(newStmtLst)
  local newElseWhen = @match whenStmts.elsewhenPart begin
    SOME(inner) => SOME(_substituteParamInWhenStmts(inner, paramValueMap))
    NONE() => NONE()
  end
  return BDAE.WHEN_STMTS(newCond, newStmtLst, newElseWhen)
end

"""
Recursively substitute alias CREFs in a WHEN_STMTS node (condition + statements + elsewhen).
"""
function _substituteAliasInWhenStmts(whenStmts::BDAE.WHEN_STMTS, aliasMap)
  local (newCond, _) = Util.traverseExpTopDown(whenStmts.condition, substituteAliasCref, aliasMap)
  local newStmtLst::List{BDAE.WhenOperator} = MetaModelica.nil
  for stmt in whenStmts.whenStmtLst
    local newStmt::BDAE.WhenOperator = @match stmt begin
      BDAE.ASSIGN(__) => begin
        local (newL, _) = Util.traverseExpTopDown(stmt.left, substituteAliasCref, aliasMap)
        local (newR, _) = Util.traverseExpTopDown(stmt.right, substituteAliasCref, aliasMap)
        BDAE.ASSIGN(newL, newR, stmt.source)
      end
      BDAE.REINIT(__) => begin
        local (newSV, _) = Util.traverseExpTopDown(stmt.stateVar, substituteAliasCref, aliasMap)
        local (newVal, _) = Util.traverseExpTopDown(stmt.value, substituteAliasCref, aliasMap)
        BDAE.REINIT(newSV, newVal, stmt.source)
      end
      BDAE.NORETCALL(__) => begin
        local (newExp, _) = Util.traverseExpTopDown(stmt.exp, substituteAliasCref, aliasMap)
        BDAE.NORETCALL(newExp, stmt.source)
      end
      BDAE.ASSERT(__) => begin
        local (newC, _) = Util.traverseExpTopDown(stmt.condition, substituteAliasCref, aliasMap)
        local (newM, _) = Util.traverseExpTopDown(stmt.message, substituteAliasCref, aliasMap)
        BDAE.ASSERT(newC, newM, stmt.level, stmt.source)
      end
      _ => stmt
    end
    newStmtLst = MetaModelica.Cons{BDAE.WhenOperator}(newStmt, newStmtLst)
  end
  newStmtLst = listReverse(newStmtLst)
  local newElse = @match whenStmts.elsewhenPart begin
    SOME(elseWhenEq) => SOME(_substituteAliasInElseWhen(elseWhenEq, aliasMap))
    NONE() => NONE()
  end
  return BDAE.WHEN_STMTS(newCond, newStmtLst, newElse)
end

function _substituteAliasInElseWhen(elseWhenEq, aliasMap)
  local inner = elseWhenEq.whenEquation
  local newInner = _substituteAliasInWhenStmts(inner, aliasMap)
  @assign elseWhenEq.whenEquation = newInner
  return elseWhenEq
end

"""
Collect all CREF names from a WHEN_STMTS node (condition + statements + elsewhen).
"""
function _collectWhenCrefNames!(names::Set{String}, whenStmts::BDAE.WHEN_STMTS)
  collectCrefNames!(names, whenStmts.condition)
  for stmt in whenStmts.whenStmtLst
    @match stmt begin
      BDAE.ASSIGN(__) => begin
        collectCrefNames!(names, stmt.left)
        collectCrefNames!(names, stmt.right)
      end
      BDAE.REINIT(__) => begin
        collectCrefNames!(names, stmt.stateVar)
        collectCrefNames!(names, stmt.value)
      end
      BDAE.NORETCALL(__) => collectCrefNames!(names, stmt.exp)
      BDAE.ASSERT(__) => begin
        collectCrefNames!(names, stmt.condition)
        collectCrefNames!(names, stmt.message)
      end
      _ => ()
    end
  end
  @match whenStmts.elsewhenPart begin
    SOME(elseWhenEq) => _collectWhenCrefNames!(names, elseWhenEq.whenEquation)
    NONE() => ()
  end
  return nothing
end

"""
    detectAlias(exp::DAE.Exp, ht)

Detect if an expression represents an alias equation.
Recognizes patterns:
  - `BINARY(cref_a, SUB, cref_b)` meaning `a - b = 0`, i.e. `a = b` (negated=false)
  - `BINARY(cref_a, ADD, cref_b)` meaning `a + b = 0`, i.e. `a = -b` (negated=true)

Where cref_a and cref_b can be bare CREFs or ASUB-wrapped CREFs.
Both variables must exist in the hash table and be Real-valued.

Returns `(name1, name2, negated, cref1, type1, cref2, type2)` or `nothing`.
"""
function detectAlias(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      local isAdd = @match op begin
        DAE.ADD(__) => true
        _ => false
      end
      if !isSub && !isAdd
        return nothing
      end
      local r1 = extractCrefName(e1)
      local r2 = extractCrefName(e2)
      if r1 === nothing || r2 === nothing
        return nothing
      end
      local (n1, cr1, t1) = r1
      local (n2, cr2, t2) = r2
      #= Both must exist in hash table =#
      if !haskey(ht, n1) || !haskey(ht, n2)
        return nothing
      end
      #= Both must be of an alias-eligible type, and matching class
         (Real-Real, Bool-Bool, Int-Int, Enum-Enum). Cross-class mixing
         is rejected. =#
      local (_, sv1) = ht[n1]
      local (_, sv2) = ht[n2]
      local cls1 = _aliasTypeClass(t1)
      local cls2 = _aliasTypeClass(t2)
      if cls1 === :other || cls2 === :other || cls1 !== cls2
        return nothing
      end
      #= Both must be unknowns (not parameters, strings, or data structures).
         Alias elimination removes equations and variables in pairs. If one side
         is a parameter, removing the equation leaves the unknown without a
         defining equation, breaking the equation-unknown balance. =#
      if !isUnknownVarKind(sv1.varKind) || !isUnknownVarKind(sv2.varKind)
        return nothing
      end
      local negated = isAdd  #= a + b = 0 means a = -b =#
      return (n1, n2, negated, cr1, t1, cr2, t2)
    end
    _ => return nothing
  end
end

"""
    substituteAliasCref(exp::DAE.Exp, aliasMap)

Callback for `traverseExpTopDown`. Replaces CREF expressions whose name
matches an alias map entry with the representative CREF (possibly negated).
Also handles ASUB-wrapped CREFs.
"""
function substituteAliasCref(@nospecialize(exp), aliasMap)
  @match exp begin
    DAE.ASUB(innerExp, subs) => begin
      @match innerExp begin
        DAE.CREF(cr, ty) => begin
          local baseName = DAE_identifierToString(cr)
          local fullName = buildAsubName(baseName, subs)
          if !isempty(fullName) && haskey(aliasMap, fullName)
            local (repName, negated, repCref, repTy) = aliasMap[fullName]
            #= Check if the representative also has ASUB subscripts =#
            local repBase = replace(repName, r"\[.*" => "")
            if repBase != repName
              #= Representative is also subscripted. Build ASUB with rep CREF. =#
              local repSubs = parseSubscriptsFromName(repName)
              local newInner = DAE.CREF(repCref, repTy)
              local newExp = DAE.ASUB(newInner, repSubs)
              if negated
                return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
              else
                return (newExp, false, aliasMap)
              end
            else
              #= Representative is a scalar. Use bare CREF. =#
              local newExp = DAE.CREF(repCref, repTy)
              if negated
                return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
              else
                return (newExp, false, aliasMap)
              end
            end
          end
          #= Also check the base name (for cases where ASUB+CREF base name is aliased) =#
          if haskey(aliasMap, baseName)
            local (repName, negated, repCref, repTy) = aliasMap[baseName]
            local newInner = DAE.CREF(repCref, repTy)
            local newExp = DAE.ASUB(newInner, subs)
            if negated
              return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
            else
              return (newExp, false, aliasMap)
            end
          end
          return (exp, true, aliasMap)
        end
        _ => return (exp, true, aliasMap)
      end
    end
    DAE.CREF(cr, ty) => begin
      local name = DAE_identifierToString(cr)
      if haskey(aliasMap, name)
        local (repName, negated, repCref, repTy) = aliasMap[name]
        local newExp = DAE.CREF(repCref, repTy)
        if negated
          return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
        else
          return (newExp, false, aliasMap)
        end
      end
      return (exp, true, aliasMap)
    end
    _ => return (exp, true, aliasMap)
  end
end

"""
    parseSubscriptsFromName(name::String)::Vector{DAE.Exp}

Parse subscripts from a variable name like "a[1][2]" into [DAE.ICONST(1), DAE.ICONST(2)].
Used to reconstruct ASUB subscripts for the representative variable.
"""
function parseSubscriptsFromName(name::String)::Vector{DAE.Exp}
  local subs = DAE.Exp[]
  for m in eachmatch(r"\[(\d+)\]", name)
    push!(subs, DAE.ICONST(parse(Int, m.captures[1])))
  end
  return subs
end

"""
    removeRedundantEquations(simCode::SIM_CODE) -> SIM_CODE

Post-alias-elimination over-determination reduction.

After alias elimination, some residual equations may become structurally
redundant: they mention only unknowns that are already uniquely determined
by other equations. This produces more equations than unknowns
(ExtraEquationsSystemException in MTK structural_simplify).

This pass computes a maximum bipartite matching of residual equations to
surviving unknowns. Equations that cannot be matched to any still-free
unknown are algebraically implied by the matched equations (assuming the
original Modelica model is well-posed) and are safely removed.

Typical trigger: balanced 3-phase star networks where the Kirchhoff current
law `i[1]+i[2]+i[3]=0` is a zero-sum identity implied by the three
per-phase Ohm's law equations, but survives alias elimination as an extra
residual.
"""
#= Detect residual of the form `0 = var - expr` or `0 = expr - var`
   where var is a simple unknown CREF and expr is anything more complex
   than a single CREF. Returns (name, cref, ty, exprKey) or nothing.
   Skips var-var form (handled by detectAlias). For both `var - expr`
   and `expr - var` the canonical key is `string(expr)`, so two
   equations with the same complex side group together regardless of
   which side the leaf var was on. =#
function _detectVarMinusExpr(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      isSub || return nothing
      local r1 = extractCrefName(e1)
      local r2 = extractCrefName(e2)
      #= Skip var-var (detectAlias handles this) and complex-complex. =#
      if (r1 !== nothing && r2 !== nothing) || (r1 === nothing && r2 === nothing)
        return nothing
      end
      local r, complexExp
      if r1 !== nothing
        r = r1; complexExp = e2
      else
        r = r2; complexExp = e1
      end
      local (n, cr, ty) = r
      haskey(ht, n) || return nothing
      local (_, sv) = ht[n]
      isUnknownVarKind(sv.varKind) || return nothing
      local cls = _aliasTypeClass(ty)
      cls === :other && return nothing
      return (n, cr, ty, string(complexExp))
    end
    _ => return nothing
  end
end

#= After eliminateAliasVariables, two equations may implicitly assert
   var1 = var2 via identical RHS expressions, e.g.
     0 = x - der(z)
     0 = y - der(z)
   This pass groups by string form of the non-leaf side and aliases
   matching LHS vars to a single representative. =#
function eliminateRHSEquivalentEquations(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode) || hasFlatModel(simCode)
    return simCode
  end
  local ht  = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local irreducibleSet = Set{String}(simCode.irreductableVariables)
  local sharedVarSet   = Set{String}(simCode.sharedVariables)

  local rhsGroups = Dict{String, Vector{Tuple{String, Int, DAE.ComponentRef, DAE.Type}}}()
  for (i, eq) in enumerate(resEqs)
    local pair = _detectVarMinusExpr(eq.exp, ht)
    pair === nothing && continue
    local (n, cr, ty, key) = pair
    if !haskey(rhsGroups, key)
      rhsGroups[key] = Tuple{String, Int, DAE.ComponentRef, DAE.Type}[]
    end
    push!(rhsGroups[key], (n, i, cr, ty))
  end

  local aliasMap = Dict{String, Tuple{String, Bool, DAE.ComponentRef, DAE.Type}}()
  local aliasEntries = AliasEntry[]
  local removeEqs = Set{Int}()
  local elimVarOrder = String[]
  local elimEqOrder  = BDAE.RESIDUAL_EQUATION[]

  for (_key, entries) in pairs(rhsGroups)
    length(entries) >= 2 || continue
    local bestIdx = 1
    local bestPrio = -1
    for (j, (n, _, _, _)) in enumerate(entries)
      haskey(ht, n) || continue
      local (_, sv) = ht[n]
      local prio = varKindPriority(sv.varKind)
      if n in irreducibleSet
        prio += 60
      end
      if prio > bestPrio
        bestPrio = prio
        bestIdx = j
      end
    end
    local (repName, _, repCref, repTy) = entries[bestIdx]
    local (_, repSv) = ht[repName]
    local repIsState = @match repSv.varKind begin
      STATE(__) => true
      _ => false
    end
    for (j, entry) in enumerate(entries)
      j == bestIdx && continue
      local (n, eqIdx, _, _) = entry
      n == repName && continue
      n in sharedVarSet && continue
      #= Protect `_re` / `_im` scalarized parts of Complex variables. These
         names are referenced from earlier-pass observed equations
         (`simCode.eliminatedEquations`) via the original Complex CREF
         + TSUB indexing, not via the flat `_re` name. Our
         `substituteAliasCref` matches on the flat name and would miss
         those references, leaving dangling refs at codegen time
         (observed on QuasiStationary models). Keep the scalarized
         siblings in the HT; the original RHS-equiv group still aliases
         the non-Complex members. =#
      if endswith(n, "_re") || endswith(n, "_im")
        continue
      end
      local (_, sv) = ht[n]
      local isState = @match sv.varKind begin
        STATE(__) => true
        _ => false
      end
      if n in irreducibleSet && !(repIsState && isState)
        continue
      end
      aliasMap[n] = (repName, false, repCref, repTy)
      push!(aliasEntries, AliasEntry(n, repName, false))
      push!(removeEqs, eqIdx)
      push!(elimVarOrder, n)
      push!(elimEqOrder, resEqs[eqIdx])
    end
  end

  if isempty(aliasMap)
    return simCode
  end

  @info "[SIMCODE: $(simCode.name): eliminateRHSEquivalentEquations] aliased $(length(aliasMap)) variables via RHS equivalence; removing $(length(removeEqs)) redundant equations"
  if OMBackend.BACKEND_PERFLOG[]
    @info "[SIMCODE: $(simCode.name): eliminateRHSEquivalentEquations] model size" residuals_before=length(resEqs) residuals_after=length(resEqs) - length(removeEqs) variables_before=length(ht) variables_after=length(ht) - length(aliasMap)
  end

  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(removeEqs))
  for (i, eq) in enumerate(resEqs)
    i in removeEqs && continue
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteAliasCref, aliasMap)
    push!(newResEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end

  #= Substitute in if-equation branches: conditions + branch residual equations.
     Without this, an aliased variable that appears in an if-branch becomes
     a dangling reference at codegen time. Matches eliminateAliasVariables's
     equivalent step. =#
  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = BDAE.RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = Util.traverseExpTopDown(brEq.exp, substituteAliasCref, aliasMap)
        push!(newBranchEqs, BDAE.RESIDUAL_EQUATION(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = Util.traverseExpTopDown(branch.condition, substituteAliasCref, aliasMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  #= When equations: substitute in conditions and statements. =#
  local newWhenEqs = BDAE.WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local innerWhen = _substituteAliasInWhenStmts(whenEq.whenEquation, aliasMap)
    @assign whenEq.whenEquation = innerWhen
    push!(newWhenEqs, whenEq)
  end

  #= Initial equations: substitute alias CREFs. =#
  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION
      local (newInitExp, _) = Util.traverseExpTopDown(initEq.exp, substituteAliasCref, aliasMap)
      push!(newInitEqs, BDAE.RESIDUAL_EQUATION(newInitExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(initEq.lhs, substituteAliasCref, aliasMap)
      local (newRhs, _) = Util.traverseExpTopDown(initEq.rhs, substituteAliasCref, aliasMap)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    else
      push!(newInitEqs, initEq)
    end
  end

  #= Substitute in existing eliminatedEquations too. Earlier passes (like
     eliminateAliasVariables) may have appended observed equations that
     reference variables we are now eliminating; without this substitution,
     `generateEliminatedObservedBlock` emits code referencing names that
     have been removed from the HT, causing UndefVarError at module eval. =#
  local oldElimEqs = simCode.eliminatedEquations
  local rewrittenElimEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(rewrittenElimEqs, length(oldElimEqs))
  for eq in oldElimEqs
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteAliasCref, aliasMap)
    push!(rewrittenElimEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end

  local newHT = copy(ht)
  local elimVarSet = Set{String}(keys(aliasMap))
  for varName in keys(aliasMap)
    delete!(newHT, varName)
  end

  @assign simCode.residualEquations = newResEqs
  @assign simCode.initialEquations  = newInitEqs
  @assign simCode.ifEquations       = newIfEqs
  @assign simCode.whenEquations     = newWhenEqs
  @assign simCode.stringToSimVarHT  = newHT
  @assign simCode.eliminatedEquations = rewrittenElimEqs
  @assign simCode.irreductableVariables = filter(n -> !(n in elimVarSet), simCode.irreductableVariables)
  append!(simCode.aliasMap, aliasEntries)
  append!(simCode.eliminatedVariables, elimVarOrder)
  append!(simCode.eliminatedEquations, elimEqOrder)
  return simCode
end

#= True if exp is a literal 1.0 or integer 1. =#
function _isOneLiteral(@nospecialize(exp))
  @match exp begin
    DAE.RCONST(x) => x == 1.0
    DAE.ICONST(x) => x == 1
    _ => false
  end
end

#= Return the numeric value of a literal, or nothing if not a literal. =#
function _extractNumericValue(@nospecialize(exp))
  @match exp begin
    DAE.RCONST(x) => x
    DAE.ICONST(x) => Float64(x)
    DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => begin
      local v = _extractNumericValue(inner)
      v === nothing ? nothing : -v
    end
    _ => nothing
  end
end

#= Fold numeric subexpressions in a DAE.Exp tree. Bottom-up evaluation:
   when both operands of a BINARY are numeric literals, replace with the
   evaluated result; partial-eval `0 * x`, `x * 0` to `RCONST(0)` and
   `0 + x`, `x + 0`, `x - 0` to the surviving operand.

   Used after frozen-state substitution so that residuals like
     `0 = -phasor_i_[2] - (0.0 * 0.0 + 0.5773 * 0.0)`
   collapse to `0 = -phasor_i_[2] - 0.0`, exposing a new pin in the next
   iteration of `eliminateFrozenStates`.

   Conservative: does not fold DIV by zero, sin/cos/exp of constants
   (correctness OK but produces UNARY-RCONST forms that downstream code
   may not expect). =#
function _foldNumericExp(@nospecialize(exp))
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local f1 = _foldNumericExp(e1)
      local f2 = _foldNumericExp(e2)
      local v1 = _extractNumericValue(f1)
      local v2 = _extractNumericValue(f2)
      if v1 !== nothing && v2 !== nothing
        @match op begin
          DAE.ADD(__) => return DAE.RCONST(v1 + v2)
          DAE.SUB(__) => return DAE.RCONST(v1 - v2)
          DAE.MUL(__) => return DAE.RCONST(v1 * v2)
          DAE.DIV(__) => begin
            v2 != 0 && return DAE.RCONST(v1 / v2)
          end
          _ => nothing
        end
      end
      #= Partial folds: 0 * x = 0, 1 * x = x, x * 0 = 0, x * 1 = x,
         0 + x = x, x + 0 = x, x - 0 = x, 0 - x = -x, x / 1 = x.
         Plus the structural tautology x - x = 0 (catches alias-substituted
         residuals that became `0 = c - c` after elimination). =#
      @match op begin
        DAE.MUL(__) => begin
          (v1 !== nothing && v1 == 0) && return DAE.RCONST(0.0)
          (v2 !== nothing && v2 == 0) && return DAE.RCONST(0.0)
          (v1 !== nothing && v1 == 1) && return f2
          (v2 !== nothing && v2 == 1) && return f1
        end
        DAE.ADD(__) => begin
          (v1 !== nothing && v1 == 0) && return f2
          (v2 !== nothing && v2 == 0) && return f1
        end
        DAE.SUB(__) => begin
          (v2 !== nothing && v2 == 0) && return f1
        end
        DAE.DIV(__) => begin
          (v2 !== nothing && v2 == 1) && return f1
        end
        _ => nothing
      end
      return DAE.BINARY(f1, op, f2)
    end
    DAE.UNARY(operator = op, exp = inner) => begin
      local fin = _foldNumericExp(inner)
      local vin = _extractNumericValue(fin)
      if vin !== nothing
        @match op begin
          DAE.UMINUS(__) => return DAE.RCONST(-vin)
          _ => nothing
        end
      end
      return DAE.UNARY(op, fin)
    end
    _ => exp
  end
end

#= Peel structurally-trivial wrappers around a sub-expression. Used by
   `_detectFrozenState` so equations emitted with redundant `* 1.0` or
   `--` decorations (common from inlining / parameter folding) still match
   the frozen pin pattern. Conservative: stops at the first non-peelable
   layer, so partial wrappers (e.g. `2.0 * x`) are left intact. =#
function _peelNoOpWrappers(@nospecialize(exp))
  local prev
  while true
    prev = exp
    @match exp begin
      DAE.BINARY(exp1 = e1, operator = DAE.MUL(__), exp2 = e2) => begin
        if _isOneLiteral(e2)
          exp = e1
        elseif _isOneLiteral(e1)
          exp = e2
        end
      end
      DAE.BINARY(exp1 = e1, operator = DAE.DIV(__), exp2 = e2) => begin
        if _isOneLiteral(e2)
          exp = e1
        end
      end
      DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => begin
        @match inner begin
          DAE.UNARY(operator = DAE.UMINUS(__), exp = innerInner) => begin
            exp = innerInner
          end
          _ => nothing
        end
      end
      _ => nothing
    end
    exp === prev && break
  end
  return exp
end

#= True if exp is a numeric literal (optionally wrapped in unary minus or
   no-op multiplications by 1). =#
function _isNumericLiteral(@nospecialize(exp))
  local peeled = _peelNoOpWrappers(exp)
  @match peeled begin
    DAE.RCONST(__) => true
    DAE.ICONST(__) => true
    DAE.UNARY(operator = DAE.UMINUS(__),     exp = inner) => _isNumericLiteral(inner)
    DAE.UNARY(operator = DAE.UMINUS_ARR(__), exp = inner) => _isNumericLiteral(inner)
    _ => false
  end
end

#= Detect a residual of the form `0 = var - literal` (or `0 = literal - var`)
   where `var` is structurally pinned to a constant. Eligible varKinds are
   STATE (the original kinematic-ground case, e.g. AIMC stator phi=0) and
   ALG_VARIABLE (post-parameter-elimination cases, e.g. AIMC R_actual=0.03
   after the alpha*(T-T_ref) term folds to zero). Returns
   (name, cref, ty, literalExp, isState) or nothing.

   STATE eligibility is what enables the `der(state) -> 0` substitution.
   ALG_VARIABLE is structurally identical for substitution (no derivative
   to handle). DISCRETE / ARRAY / OCC_VARIABLE are excluded because they
   carry event or connector semantics. =#
#= Extract a CREF together with its sign within a residual term.
   Returns (name, cref, ty, sign) where sign is +1 for bare CREF, -1 for
   UNARY(UMINUS, CREF). Also peels `* 1.0` / `/1.0` / `--` wrappers
   first so decorated forms like `var * 1.0` still match.
   Returns nothing if the term is anything else (multi-coefficient,
   non-leaf, etc.). =#
function _extractCrefSigned(@nospecialize(exp))
  local peeled = _peelNoOpWrappers(exp)
  @match peeled begin
    DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => begin
      local innerPeeled = _peelNoOpWrappers(inner)
      local r = extractCrefName(innerPeeled)
      r === nothing && return nothing
      local (n, cr, ty) = r
      return (n, cr, ty, -1)
    end
    _ => begin
      local r = extractCrefName(peeled)
      r === nothing && return nothing
      local (n, cr, ty) = r
      return (n, cr, ty, 1)
    end
  end
end

#= Negate a numeric literal expression, preserving its DAE structure when
   trivially possible (RCONST/ICONST get value-negated; anything else gets
   wrapped in UNARY(UMINUS)). =#
function _negateLiteralExp(@nospecialize(litExp))
  @match litExp begin
    DAE.RCONST(x) => DAE.RCONST(-x)
    DAE.ICONST(x) => DAE.ICONST(-x)
    DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => inner
    _ => DAE.UNARY(DAE.UMINUS(DAE.T_REAL_DEFAULT), litExp)
  end
end

function _detectFrozenState(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      isSub || return nothing
      local e1p = _peelNoOpWrappers(e1)
      local e2p = _peelNoOpWrappers(e2)
      local s1 = _extractCrefSigned(e1)
      local s2 = _extractCrefSigned(e2)
      local stateRef, litExp, varSign
      #= Equation form `s1*var - lit = 0` => var = lit/s1.
         Equation form `lit - s2*var = 0` => var = lit/s2. =#
      if s1 !== nothing && _isNumericLiteral(e2p)
        local (n, cr, ty, sg) = s1
        stateRef = (n, cr, ty); litExp = e2p; varSign = sg
      elseif s2 !== nothing && _isNumericLiteral(e1p)
        local (n, cr, ty, sg) = s2
        stateRef = (n, cr, ty); litExp = e1p; varSign = sg
      else
        return nothing
      end
      if varSign == -1
        litExp = _negateLiteralExp(litExp)
      end
      local (n, cr, ty) = stateRef
      haskey(ht, n) || return nothing
      local (_, sv) = ht[n]
      local isState = @match sv.varKind begin
        STATE(__) => true
        _ => false
      end
      local isAlg = @match sv.varKind begin
        ALG_VARIABLE(__) => true
        _ => false
      end
      (isState || isAlg) || return nothing
      return (n, cr, ty, litExp, isState)
    end
    _ => return nothing
  end
end

#= traverseExpTopDown visitor: substitute eliminated states. Returns
   (newExp, continueRecursion, frozenMap). Handles two patterns:
     - CREF(state)                        -> literal
     - CALL("der", [CREF(state)])         -> 0.0
   For non-frozen subtrees, returns the original exp with continueRecursion=true. =#
function _substituteFrozenState(@nospecialize(exp), frozenMap)
  @match exp begin
    DAE.CALL(Absyn.IDENT("der"), expLst, _) => begin
      local arg = listHead(expLst)
      @match arg begin
        DAE.CREF(cr, _) => begin
          local n = DAE_identifierToString(cr)
          if haskey(frozenMap, n)
            return (DAE.RCONST(0.0), false, frozenMap)
          end
          return (exp, true, frozenMap)
        end
        _ => return (exp, true, frozenMap)
      end
    end
    DAE.CREF(cr, _) => begin
      local n = DAE_identifierToString(cr)
      if haskey(frozenMap, n)
        return (frozenMap[n], false, frozenMap)
      end
      return (exp, true, frozenMap)
    end
    _ => return (exp, true, frozenMap)
  end
end

#= Eliminate variables that are algebraically pinned to a numeric literal.
   Two flavours, both covered:

   1. STATE pinned by a kinematic ground (e.g. AIMC `aimc_inertiaStator_phi = 0`
      from a Fixed-flange). The state has no time dynamics yet stays classified
      as STATE because `der(state)` appears in some inertia/connector equation.
      Pantelides then differentiates the pin and over-determines the system.
   2. ALG_VARIABLE pinned by a folded parameter expression (e.g. AIMC
      `aimc_rs_resistor[k]_R_actual = 0.03` after `R*(1 + alpha*(T-T_ref))`
      collapses with alpha=0). Treated identically — no derivative to handle,
      but the CREF substitution propagates the constant through every use.

   Strategy: full elimination. Substitute the variable with its literal value
   at every CREF site, and `der(state) -> 0.0` for the STATE case. The pin
   equation is dropped; the (var, eq) pair moves into eliminatedVariables /
   eliminatedEquations so MTK observed-equation generation can still expose
   the constant value on sol[:name].

   Excluded varKinds: DISCRETE (event semantics), ARRAY (subscript handling),
   OCC_VARIABLE (over-constrained connector special cases), STATE_DERIVATIVE
   (not a directly-pinnable form).

   Safety: never eliminate a variable that appears in any if-branch or
   when-equation (its name is needed for event registration / callback
   pre()-tracking). Skips for VSS / multi-mode SimCode variants.

   Iteration: substituting der(state) -> 0 can expose a new frozen variable
   in equations like `w - der(state) = 0` (becomes `w - 0 = 0`). The pass
   loops until no more matches surface, capped at 16 rounds defensively.

   Placement: after eliminateConstantParameters so parameter chains like
   `var = some_param` (with param folded to a literal) are already
   substituted to `var = literal` form before detection. =#
function eliminateFrozenStates(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode) || hasFlatModel(simCode)
    return simCode
  end
  #= Iterate to convergence: substituting der(state) -> 0 can turn a related
     equation like `w - der(state) = 0` into `w - 0 = 0`, exposing a new
     frozen state. Cap the loop count defensively even though the variable
     set strictly shrinks each round. =#
  #= protectedNames is invariant across rounds (if/when equations do not
     change), so compute it once and reuse. =#
  local protectedNames = _computeFrozenProtectedNames(simCode)
  local totalEliminated = 0
  local maxRounds = 16
  for round in 1:maxRounds
    local (newCode, nEliminated) = _eliminateFrozenStatesOnePass(simCode, protectedNames)
    nEliminated == 0 && break
    simCode = newCode
    totalEliminated += nEliminated
  end
  return simCode
end

function _computeFrozenProtectedNames(simCode::SIM_CODE)::Set{String}
  local protectedNames = Set{String}()
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(protectedNames, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(protectedNames, brEq.exp)
      end
    end
  end
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(protectedNames, whenEq.whenEquation)
  end
  return protectedNames
end

function _eliminateFrozenStatesOnePass(simCode::SIM_CODE, protectedNames::Set{String})
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local sharedVarSet = Set{String}(simCode.sharedVariables)

  local frozenMap   = Dict{String, DAE.Exp}()
  local frozenEqIdx = Dict{String, Int}()
  local frozenIsState = Dict{String, Bool}()
  for (i, eq) in enumerate(resEqs)
    local pair = _detectFrozenState(eq.exp, ht)
    pair === nothing && continue
    local (n, _, _, litExp, isState) = pair
    n in sharedVarSet  && continue
    n in protectedNames && continue
    haskey(frozenMap, n) && continue
    frozenMap[n]   = litExp
    frozenEqIdx[n] = i
    frozenIsState[n] = isState
  end

  isempty(frozenMap) && return (simCode, 0)

  #= Safety: never reduce the residual list to empty. MTK's `System(...)`
     constructor infers `Vector{Any}` from an empty literal `[]`, which
     does not match the typed-vector method signatures and raises
     MethodError at codegen time. If eliminating all detected frozen
     variables would empty the residual set, keep one of them so MTK
     still has a non-empty (but trivial) equation to construct from.
     Observed on MatrixMultTest where every variable is a constant pin. =#
  local _eqsLeftAfter = length(resEqs) - length(frozenMap)
  if _eqsLeftAfter <= 0
    local _keepOne = first(sort(collect(keys(frozenMap))))
    delete!(frozenMap, _keepOne)
    delete!(frozenEqIdx, _keepOne)
    delete!(frozenIsState, _keepOne)
    @info "[SIMCODE: $(simCode.name): eliminateFrozenStates] keeping $(_keepOne) to avoid emptying the residual system"
    isempty(frozenMap) && return (simCode, 0)
  end

  local nState = count(values(frozenIsState))
  local nAlg   = length(frozenMap) - nState
  @info "[SIMCODE: $(simCode.name): eliminateFrozenStates] eliminating $(length(frozenMap)) frozen variable(s) ($nState state, $nAlg algebraic): $(sort(collect(keys(frozenMap))))"
  if OMBackend.BACKEND_PERFLOG[]
    @info "[SIMCODE: $(simCode.name): eliminateFrozenStates] model size" residuals_before=length(resEqs) residuals_after=length(resEqs) - length(frozenMap) variables_before=length(ht) variables_after=length(ht) - length(frozenMap)
  end

  local removeEqs = Set{Int}(values(frozenEqIdx))
  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(removeEqs))
  for (i, eq) in enumerate(resEqs)
    i in removeEqs && continue
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, _substituteFrozenState, frozenMap)
    #= Constant-fold after substitution: `0.0 * x` and friends now reduce
       to 0 so the residual becomes a clean `0 = -y - 0` form that the
       next iteration can detect as a pin. =#
    newExp = _foldNumericExp(newExp)
    push!(newResEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end

  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION
      local (newExp, _) = Util.traverseExpTopDown(initEq.exp, _substituteFrozenState, frozenMap)
      newExp = _foldNumericExp(newExp)
      push!(newInitEqs, BDAE.RESIDUAL_EQUATION(newExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(initEq.lhs, _substituteFrozenState, frozenMap)
      local (newRhs, _) = Util.traverseExpTopDown(initEq.rhs, _substituteFrozenState, frozenMap)
      newLhs = _foldNumericExp(newLhs)
      newRhs = _foldNumericExp(newRhs)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    else
      push!(newInitEqs, initEq)
    end
  end

  local newHT = copy(ht)
  for n in keys(frozenMap)
    delete!(newHT, n)
  end

  #= Parallel arrays: variable order matches paired equation order. =#
  local elimVarOrder = sort(collect(keys(frozenMap)))
  local elimEqOrder  = BDAE.RESIDUAL_EQUATION[resEqs[frozenEqIdx[n]] for n in elimVarOrder]

  @assign simCode.residualEquations     = newResEqs
  @assign simCode.initialEquations      = newInitEqs
  @assign simCode.stringToSimVarHT      = newHT
  @assign simCode.irreductableVariables = filter(n -> !haskey(frozenMap, n), simCode.irreductableVariables)
  append!(simCode.eliminatedVariables,  elimVarOrder)
  append!(simCode.eliminatedEquations,  elimEqOrder)
  return (simCode, length(frozenMap))
end

Base.@nospecializeinfer function _isAlgebraicVarKind(@nospecialize(varKind))::Bool
  @match varKind begin
    ALG_VARIABLE(__) => true
    _ => false
  end
end

Base.@nospecializeinfer function _containsDerCallDAE(@nospecialize(exp))::Bool
  @match exp begin
    DAE.CALL(path = p) => begin
      @match p begin
        Absyn.IDENT(name) => name == "der"
        _ => false
      end
    end
    DAE.BINARY(exp1 = e1, exp2 = e2) => _containsDerCallDAE(e1) || _containsDerCallDAE(e2)
    DAE.UNARY(exp = e) => _containsDerCallDAE(e)
    DAE.LUNARY(exp = e) => _containsDerCallDAE(e)
    DAE.LBINARY(exp1 = e1, exp2 = e2) => _containsDerCallDAE(e1) || _containsDerCallDAE(e2)
    DAE.IFEXP(expCond = c, expThen = t, expElse = e) => _containsDerCallDAE(c) || _containsDerCallDAE(t) || _containsDerCallDAE(e)
    DAE.ARRAY(array = lst) => any(_containsDerCallDAE, lst)
    DAE.ASUB(exp = e, sub = subs) => _containsDerCallDAE(e) || any(_containsDerCallDAE, subs)
    DAE.RELATION(exp1 = e1, exp2 = e2) => _containsDerCallDAE(e1) || _containsDerCallDAE(e2)
    DAE.CAST(exp = e) => _containsDerCallDAE(e)
    DAE.TSUB(exp = e) => _containsDerCallDAE(e)
    DAE.RSUB(exp = e) => _containsDerCallDAE(e)
    DAE.REDUCTION(expr = e) => _containsDerCallDAE(e)
    _ => false
  end
end

Base.@nospecializeinfer function _detectVarMinusExprRaw(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      isSub || return nothing
      local r1 = extractCrefName(e1)
      local r2 = extractCrefName(e2)
      if (r1 !== nothing && r2 !== nothing) || (r1 === nothing && r2 === nothing)
        return nothing
      end
      local r, rhs
      if r1 !== nothing
        r = r1; rhs = e2
      else
        r = r2; rhs = e1
      end
      local (n, _cr, _ty) = r
      haskey(ht, n) || return nothing
      return (n, rhs)
    end
    _ => return nothing
  end
end

#= Substitution callback that, for every leaf CREF whose name is a key of
   the fold map, returns the bound RHS expression and stops traversal so
   the substituted form is not re-walked. ASUB-wrapped CREFs are handled
   by reading the constant-subscript suffix into the lookup key, matching
   `collectCrefNames!`'s asubHandled branch. =#
function substituteFoldedVar(@nospecialize(exp), foldMap::Dict{String, DAE.Exp})
  @match exp begin
    DAE.CREF(cr, _) => begin
      local name = DAE_identifierToString(cr)
      if haskey(foldMap, name)
        return (foldMap[name], false, foldMap)
      end
      return (exp, true, foldMap)
    end
    DAE.ASUB(exp = inner, sub = subs) => begin
      @match inner begin
        DAE.CREF(cr, _) => begin
          local baseName = DAE_identifierToString(cr)
          local allConst = true
          local suffix = ""
          for s in subs
            @match s begin
              DAE.ICONST(i) => begin suffix *= string("[", i, "]") end
              _ => begin allConst = false end
            end
          end
          if allConst && !isempty(suffix)
            local fullName = string(baseName, suffix)
            if haskey(foldMap, fullName)
              return (foldMap[fullName], false, foldMap)
            end
          end
          if haskey(foldMap, baseName)
            return (foldMap[baseName], false, foldMap)
          end
          return (exp, true, foldMap)
        end
        _ => return (exp, true, foldMap)
      end
    end
    _ => return (exp, true, foldMap)
  end
end

"""
    foldExplicitSingleAssign(simCode) -> simCode

Substitute every ALG_VARIABLE that is uniquely defined by a single
`0 = v - rhs` residual, where `rhs` has no derivative and no self-reference
to `v`. Variables protected by if/when references, irreducible / shared
sets are skipped. Sub-model / metaModel / flat-model variants are skipped
entirely because runtime parameter overrides interact with cross-submodel
references that the fold would break.

Iterates to a fixed point (up to 8 rounds) so transitive chains
(`v1 = v2 + 1; v2 = v3 + 1; v3 = literal`) collapse.
"""
function foldExplicitSingleAssign(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode) || hasFlatModel(simCode)
    return simCode
  end
  isempty(simCode.residualEquations) && return simCode
  local protectedNames = _computeFrozenProtectedNames(simCode)
  local irreducibleSet = Set{String}(simCode.irreductableVariables)
  local sharedVarSet   = Set{String}(simCode.sharedVariables)
  local totalFolded = 0
  local maxRounds = 8
  for round in 1:maxRounds
    local (newCode, nFolded) = _foldExplicitSingleAssignOnePass(simCode, protectedNames, irreducibleSet, sharedVarSet)
    nFolded == 0 && break
    simCode = newCode
    totalFolded += nFolded
  end
  if totalFolded > 0
    @info "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] folded $(totalFolded) explicit assignments"
    if OMBackend.BACKEND_PERFLOG[]
      @info "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] model size" residuals_after=length(simCode.residualEquations) variables_after=length(simCode.stringToSimVarHT)
    end
  end
  return simCode
end

function _foldExplicitSingleAssignOnePass(simCode::SIM_CODE,
                                          protectedNames::Set{String},
                                          irreducibleSet::Set{String},
                                          sharedVarSet::Set{String})
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations

  local defCountOfVar = Dict{String, Int}()
  local defEqOfVar    = Dict{String, Int}()
  local defRhsOfVar   = Dict{String, DAE.Exp}()

  #= Skip scalarized array elements (any name containing '[' or ']').
     The codegen rebuilds the parent array from its scalar siblings via
     ASUB indexing; dropping a single element from the HT breaks that
     reconstruction even though the algebraic substitution is sound. =#
  #= Skip variables that appear in any existing alias-map entry (either
     side). Folding a representative would orphan the alias entry; folding
     an aliased name would double-substitute via the observed-equation
     pipeline. =#
  local aliasNames = Set{String}()
  for entry in simCode.aliasMap
    push!(aliasNames, entry.eliminatedName)
    push!(aliasNames, entry.representativeName)
  end

  for (i, eq) in enumerate(resEqs)
    local pair = _detectVarMinusExprRaw(eq.exp, ht)
    pair === nothing && continue
    local (name, rhs) = pair
    occursin('[', name) && continue
    occursin(']', name) && continue
    name in protectedNames && continue
    name in irreducibleSet && continue
    name in sharedVarSet && continue
    name in aliasNames && continue
    local (_, sv) = ht[name]
    _isAlgebraicVarKind(sv.varKind) || continue
    _containsDerCallDAE(rhs) && continue
    local rhsNames = Set{String}()
    collectCrefNames!(rhsNames, rhs)
    name in rhsNames && continue
    defCountOfVar[name] = get(defCountOfVar, name, 0) + 1
    if !haskey(defEqOfVar, name)
      defEqOfVar[name]  = i
      defRhsOfVar[name] = rhs
    end
  end

  local foldMap = Dict{String, DAE.Exp}()
  local foldEqIdxSet = Set{Int}()
  for (name, cnt) in defCountOfVar
    cnt == 1 || continue
    foldMap[name] = defRhsOfVar[name]
    push!(foldEqIdxSet, defEqOfVar[name])
  end

  isempty(foldMap) && return (simCode, 0)

  #= Never empty the residual list. =#
  if length(resEqs) - length(foldEqIdxSet) <= 0
    @info "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] would empty residuals; skipping"
    return (simCode, 0)
  end

  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(foldEqIdxSet))
  for (i, eq) in enumerate(resEqs)
    i in foldEqIdxSet && continue
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteFoldedVar, foldMap)
    newExp = _foldNumericExp(newExp)
    push!(newResEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end

  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION
      local (newExp, _) = Util.traverseExpTopDown(initEq.exp, substituteFoldedVar, foldMap)
      newExp = _foldNumericExp(newExp)
      push!(newInitEqs, BDAE.RESIDUAL_EQUATION(newExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(initEq.lhs, substituteFoldedVar, foldMap)
      local (newRhs, _) = Util.traverseExpTopDown(initEq.rhs, substituteFoldedVar, foldMap)
      newLhs = _foldNumericExp(newLhs)
      newRhs = _foldNumericExp(newRhs)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    else
      push!(newInitEqs, initEq)
    end
  end

  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = BDAE.RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = Util.traverseExpTopDown(brEq.exp, substituteFoldedVar, foldMap)
        push!(newBranchEqs, BDAE.RESIDUAL_EQUATION(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = Util.traverseExpTopDown(branch.condition, substituteFoldedVar, foldMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  local newElimEqs = BDAE.RESIDUAL_EQUATION[]
  for eq in simCode.eliminatedEquations
    local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteFoldedVar, foldMap)
    push!(newElimEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
  end

  #= Sanity guard: scan all surviving surfaces for any folded name. If a
     name still appears (because substituteFoldedVar missed an exotic CREF
     wrapper, or the name is referenced from a code path we did not
     substitute), abort the fold — return the original simCode unchanged.
     Better to do zero folds than to leave a dangling reference that breaks
     codegen (observed on SimpleMechanicalSystem, where `tau_2` survived
     substitution somewhere downstream and produced UndefVarError). =#
  local foldKeys = Set{String}(keys(foldMap))
  local survivorNames = Set{String}()
  for eq in newResEqs
    collectCrefNames!(survivorNames, eq.exp)
  end
  for eq in newInitEqs
    if eq isa BDAE.RESIDUAL_EQUATION
      collectCrefNames!(survivorNames, eq.exp)
    elseif eq isa BDAE.EQUATION
      collectCrefNames!(survivorNames, eq.lhs)
      collectCrefNames!(survivorNames, eq.rhs)
    end
  end
  for ifEq in newIfEqs
    for branch in ifEq.branches
      collectCrefNames!(survivorNames, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(survivorNames, brEq.exp)
      end
    end
  end
  for eq in newElimEqs
    collectCrefNames!(survivorNames, eq.exp)
  end
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(survivorNames, whenEq.whenEquation)
  end
  for (_n, (_, sv)) in ht
    @match sv.varKind begin
      PARAMETER(SOME(b)) => collectCrefNames!(survivorNames, b)
      ARRAY_PARAMETER(_, SOME(b)) => collectCrefNames!(survivorNames, b)
      DATA_STRUCTURE(SOME(b)) => collectCrefNames!(survivorNames, b)
      _ => nothing
    end
  end
  local dangling = intersect(foldKeys, survivorNames)
  if !isempty(dangling)
    @debug "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] aborting — $(length(dangling)) folded name(s) still referenced after substitution: $(sort(collect(dangling)))"
    return (simCode, 0)
  end

  local newHT = copy(ht)
  for name in keys(foldMap)
    delete!(newHT, name)
  end

  local elimVarOrder = sort(collect(keys(foldMap)))
  local elimEqOrder  = BDAE.RESIDUAL_EQUATION[resEqs[defEqOfVar[n]] for n in elimVarOrder]

  @assign simCode.residualEquations    = newResEqs
  @assign simCode.initialEquations     = newInitEqs
  @assign simCode.ifEquations          = newIfEqs
  @assign simCode.stringToSimVarHT     = newHT
  @assign simCode.eliminatedEquations  = newElimEqs
  @assign simCode.irreductableVariables = filter(n -> !haskey(foldMap, n), simCode.irreductableVariables)
  append!(simCode.eliminatedVariables, elimVarOrder)
  append!(simCode.eliminatedEquations, elimEqOrder)
  return (simCode, length(foldMap))
end

function removeRedundantEquations(simCode::SIM_CODE)::SIM_CODE
  local ht  = simCode.stringToSimVarHT
  local res = simCode.residualEquations
  local n_eqs  = length(res)
  local n_vars = count(((_k, (_, sv)),) -> isUnknownVarKind(sv.varKind), ht)

  if n_eqs <= n_vars
    return simCode
  end

  local n_extra = n_eqs - n_vars
  @info "[SIMCODE: $(simCode.name): removeRedundantEquations] over-determined by $n_extra equation(s); running maximum matching to find redundant equations"

  #= Build incidence: eq_idx -> Set of unknown names that equation mentions =#
  local surviving_unknowns = Set{String}(k for (k, (_, sv)) in pairs(ht) if isUnknownVarKind(sv.varKind))
  local incidence = map(enumerate(res)) do (i, eq)
    local names = Set{String}()
    collectCrefNames!(names, eq.exp)
    intersect(names, surviving_unknowns)
  end

  #= Augmenting-path maximum bipartite matching (equations -> unknowns).
     var_to_eq[v] = equation currently assigned to unknown v.
     eq_to_var[i] = unknown currently assigned to equation i ("" = unmatched). =#
  local var_to_eq = Dict{String, Int}()
  local eq_to_var = fill("", n_eqs)

  function augment!(eq_idx::Int, seen::Set{Int})::Bool
    for var in incidence[eq_idx]
      eq_idx in seen && continue
      push!(seen, eq_idx)
      if !haskey(var_to_eq, var) || augment!(var_to_eq[var], seen)
        var_to_eq[var] = eq_idx
        eq_to_var[eq_idx] = var
        return true
      end
    end
    return false
  end

  map(i -> augment!(i, Set{Int}()), 1:n_eqs)

  #= Equations with no assigned unknown are unmatched = redundant =#
  local redundant = Int[i for i in 1:n_eqs if isempty(eq_to_var[i])]

  if isempty(redundant)
    @warn "[SIMCODE: $(simCode.name): removeRedundantEquations] over-determined but no unmatched equations found; leaving system unchanged"
    return simCode
  end

  map(redundant) do i
    local eqStr = try OMFrontend.Frontend.toString(res[i].exp) catch; string(res[i].exp) end
    @info "[SIMCODE: $(simCode.name): removeRedundantEquations] removing redundant equation [$i]: 0 = $eqStr"
  end

  local redundant_set = Set{Int}(redundant)
  local newRes = BDAE.RESIDUAL_EQUATION[res[i] for i in 1:n_eqs if i ∉ redundant_set]
  @assign simCode.residualEquations = newRes
  return simCode
end
