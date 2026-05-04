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
  cref -> variable information dictonary.
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
   This functions create and assigns indices for variables
   Thus Construct the table that maps variable name to the actual variable.
It executes the following steps:
1. Collect all variables
2. Search all states (e.g. x and y) and give them indices starting at 1 (so x=1, y=2). Then give the corresponding state derivatives (x' and y') the same indices.
3. Remaining algebraic variables will get indices starting with i+1, where i is the number of states.
4. Parameters will get own set of indices, starting at 1.
5. Discrete shares the index with the states and starts at #states + 1
6. OCC Variables also shares the indices with the states and starts at #discretes + 1
7. Datastructure variables, are only allowed as parameters and or constants. They share the index with the parameters.
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
  This function creates a bidrectional graph between these equations and the supplied variables.
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
  I should investigat how to go about it.
  For now lets merge in the equations in an initial-when equation as ordinary equations. =#
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
  Given a set of variables and a dictonary that maps the component reference
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
  Returns the residual equation a specfic variable is solved in.
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
 Convert the component references to the backend representation and create an adjecency list representation.
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
 This function returns true if a backend variable is in the set of of overconstrained connector variables (occVariables).

TODO: the name of the theta variable is hardcoded for now
Note that this function must be called before sorting.
"""
function isOverconstrainedConnectorVariable(simVarName::String, occVariables::Vector{String})
  #= Inefficient crap, can be done better... =#
  local isOCCVar = simVarName in occVariables
  return isOCCVar
end

"""
  Get all variables that should be marked as irreductable.
OBS:
Parameters are never added to this list.
The known irreductables should be state variables and variables directly involved in changes that change the model structure.
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
    Parameters should not be marked as irreductable
    Remove them from the list
  =#
  local knownIrreductables::Vector{BDAE.VAR} = filter((v) -> BDAEUtil.isState(v) , algebraicAndStateVariables)
  #@info "Adding all states as irreductable variables" map(x->string(x.varName), knownIrreductables)
  push!(irreductables, map(x->BDAE_identifierToVarString(x), knownIrreductables))
  irreductables = collect(Iterators.flatten(irreductables))
  irreductables = filter(irv -> !(irv != "time" && isParameter(last(ht[irv]))), irreductables)
  #TODO: Fix the dection, s.t variables critical to when equations are not removed
  #for eq in whenEqs
    # variablesForEq = Backend.BDAEUtil.getAllVariables(eq, algebraicAndStateVariables)
    # push!(variablesForEq, irreductables)
  #end
  #= Add known irreductables to the vector =#
  #push!(irreductables, map(x->x.varName, string(knownIrreductables)))
  local irreductablesAsStr = map(x -> string(x), irreductables)
  #=
  If THETA exists, treat it as a irreductable variable
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
    @info "rebuildMatchOrder: under-determined system ($nEqs equations, $nVars unknowns), skipping"
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
    @info "rebuildMatchOrder: matching failed, skipping DCE" exception=(e, catch_backtrace())
    return (Int[], nameToMatchIdx, matchIdxToName)
  end
  local nMatched = count(>(0), matchOrder)
  @info "rebuildMatchOrder: $nEqs equations, $nVars unknowns, $nMatched matched"
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
    @info "eliminateNonDynamic: skipping for VSS/multi-mode model"
    return simCode
  end
  #= Rebuild a fresh matching from the current (post-optimization) system =#
  local (matchOrder, nameToMatchIdx, matchIdxToName) = rebuildMatchOrder(simCode)
  if isempty(matchOrder)
    @info "eliminateNonDynamic: matching failed or system not square, skipping"
    return simCode
  end
  #= Identify output-only equations and variables using the fresh matching =#
  local (outputOnlyVarNames, outputOnlyEqIndices, eqRefs) =
    identifyOutputOnlyVariables(simCode, matchOrder, matchIdxToName)
  if isempty(outputOnlyEqIndices)
    @info "eliminateNonDynamic: no output-only equations found"
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
    @info "eliminateNonDynamic: no eliminable equation-variable pairs found"
    return simCode
  end
  #= Guard: never eliminate ALL equations. A system with zero equations
     after elimination would crash downstream (filterConstantEquations, MTK). =#
  if length(eqsToEliminate) >= length(simCode.residualEquations)
    @info "eliminateNonDynamic: would eliminate all $(length(simCode.residualEquations)) equations, skipping"
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
  @info "eliminateNonDynamic: eliminated $(length(eqsToEliminate)) eq-var pairs, $(length(varsToRemove)) variables removed (rescued: $nRescued, skipped: $nSkippedNonAlg non-algebraic, $nSkippedUnmatched unmatched). $(length(newResEqs)) equations, $(length(newHT)) variables remain"
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
    @info "foldParameterClosure: fold would eliminate all residuals; skipping to preserve MTK build" wouldFold=length(foldMap)
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
    @info "[inlinePreOfConstantParameters] replaced $(nReplaced[]) `pre(constParam)` occurrences with the parameter directly"
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
    @info "constantPropagation: base array names referenced" allBaseNames=collect(allBaseNames)
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
    @info "constantPropagation: no constant equations found"
    return simCode
  end

  @info "constantPropagation: found $nConst unknown=param equations and $nTrivial trivial param=param equations"

  #= Phase 2: Build final equation list with substitutions applied =#
  local allRemoved = union(constEqIndices, trivialEqIndices)
  local newResEqs = BDAE.RESIDUAL_EQUATION[]
  local elimEqs = BDAE.RESIDUAL_EQUATION[]
  sizehint!(newResEqs, nEqs - length(allRemoved))

  for (i, eq) in enumerate(simCode.residualEquations)
    if i in allRemoved
      push!(elimEqs, eq)
    else
      local (newExp, _) = Util.traverseExpTopDown(eq.exp, substituteAliasCref, constMap)
      push!(newResEqs, BDAE.RESIDUAL_EQUATION(newExp, eq.source, eq.attr))
    end
  end

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
    @warn "constantPropagation: $(length(survivingRefs)) eliminated variables still referenced, keeping them" survivingRefs=collect(survivingRefs)
  end

  local newHT = copy(ht)
  local elimVarNames = String[]
  for (varName, _) in constMap
    if varName in survivingRefs
      continue
    end
    delete!(newHT, varName)
    push!(elimVarNames, varName)
  end

  @info "constantPropagation: eliminated $(length(elimVarNames)) unknowns and $(length(allRemoved)) equations ($(length(newResEqs)) equations, $(length(newHT)) variables remain)"

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
    @info "aliasElimination: skipped (VSS/multi-mode model)"
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
    @info "aliasElimination: no alias equations found"
    return simCode
  end

  @info "aliasElimination: detected $(length(aliasPairs)) alias equations"

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

    #= Mark all other variables in this component for elimination.
       Never eliminate irreducible variables (involved in events). =#
    for (varName, negFromRoot) in component
      if varName == bestName
        continue
      end
      if !haskey(ht, varName)
        continue
      end
      if varName in irreducibleSet
        #= Irreducible variables must not be eliminated; they are referenced
           by name in start conditions and callback code. =#
        continue
      end
      #= Compute negation: eliminatedVar = (-1)^negated * representative
         negFromRoot of eliminated XOR negFromRoot of representative =#
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
    @info "aliasElimination: no variables could be eliminated"
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
    @warn "aliasElimination: $(length(survivingRefs)) eliminated variables still referenced, keeping them" survivingRefs=collect(survivingRefs)
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

  @info "aliasElimination: eliminated $(length(elimVarNames)) variables and $(length(aliasEqIndices)) equations ($(length(newResEqs)) equations, $(length(newHT)) variables remain)"

  @assign simCode.residualEquations = newResEqs
  @assign simCode.initialEquations = newInitEqs
  @assign simCode.stringToSimVarHT = newHT
  @assign simCode.ifEquations = newIfEqs
  @assign simCode.whenEquations = newWhenEqs
  @assign simCode.aliasMap = keptAliasEntries
  #= Append eliminated equations/variables to the existing lists =#
  append!(simCode.eliminatedEquations, elimEqs)
  append!(simCode.eliminatedVariables, elimVarNames)
  return simCode
end

"""
Recursively substitute alias CREFs in a WHEN_STMTS node (condition + statements + elsewhen).
"""
function _substituteAliasInWhenStmts(whenStmts::BDAE.WHEN_STMTS, aliasMap)
  local (newCond, _) = Util.traverseExpTopDown(whenStmts.condition, substituteAliasCref, aliasMap)
  local newStmtLst = MetaModelica.list()
  for stmt in whenStmts.whenStmtLst
    local newStmt = @match stmt begin
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
    newStmtLst = newStmt <| newStmtLst
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
