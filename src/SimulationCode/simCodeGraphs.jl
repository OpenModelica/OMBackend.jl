#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF AGPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GNU AGPL
* VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the OSMC (Open Source Modelica Consortium)
* Public License (OSMC-PL) are obtained from OSMC, either from the above
* address, from the URLs:
* http://www.openmodelica.org or
* https://github.com/OpenModelica/ or
* http://www.ida.liu.se/projects/OpenModelica,
* and in the OpenModelica distribution.
*
* GNU AGPL version 3 is obtained from:
* https://www.gnu.org/licenses/licenses.html#GPL
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#

#= The equation-variable graph and its strongly connected components. =#

"""
  Given a set of residual equations, a set of if-equations and the set of all backend variables.
  This function creates a bidirectional graph between these equations and the supplied variables.
  (Note: If we need to do index reduction there might be empty equations here).
"""
function createEquationVariableBidirectionGraph(equations::AbstractVector,
                                                ifEquations::IF_EQS,
                                                whenEquations::WHEN_EQS,
                                                allBackendVars::VARS,
                                                stringToSimVarHT)::OrderedDict where{IF_EQS, WHEN_EQS, VARS}
  local eqCounter::Int = 0
  local variableEqMapping = OrderedDict{Int, Vector{Int}}()
  local unknownVariables = filter((x) -> BDAEUtil.isVariable(x.varKind), allBackendVars)
  #=TODO: The set of discrete variables are currently not in use. =#
  local discreteVariables = filter((x) -> BDAEUtil.isDiscrete(x.varKind), allBackendVars)
  local stateVariables = filter((x) -> BDAEUtil.isState(x.varKind), allBackendVars)
  local algebraicAndStateVariables = vcat(unknownVariables, stateVariables)
  #= Name-keyed lookup so each equation scans its own crefs, not every model
     variable; on large models the difference is hours vs seconds. =#
  local varByName = BDAEUtil.variablesByName(algebraicAndStateVariables)
  local nDiscretes = length(discreteVariables)
  @debug "#stateVariables" length(stateVariables)
  @debug "#discretes" nDiscretes
  @debug "#algebraic" length(unknownVariables)
  @debug "#equations" length(equations)
  for eq in equations
    eqCounter += 1
    variablesForEq = Backend.BDAEUtil.getAllVariables(eq, varByName)
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
    variableEqMapping[eqCounter] = sort(indices)
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
      variablesForEq = Backend.BDAEUtil.getAllVariables(eq, varByName)
      variableEqMapping[eqCounter] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
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
    local cond = weq.whenEquation.condition
    local isInitialCond = cond isa DAE.CALL && cond.path isa Absyn.IDENT && cond.path.name == "initial"
    if isInitialCond
      for wstmt in weq.whenEquation.whenStmtLst
        eqCounter += 1
        variablesForEq = BDAEUtil.getAllVariables(wstmt, algebraicAndStateVariables)
        variableEqMapping[eqCounter] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
      end
    else
      for wstmt in weq.whenEquation.whenStmtLst
        local isAssignReal = (wstmt isa BDAE.ASSIGN || wstmt isa ASSIGN) &&
                             wstmt.left isa DAE.CREF && wstmt.left.ty isa DAE.T_REAL
        if isAssignReal
          local refAsStr = BDAEUtil.string(wstmt.left.componentRef)
          local simVar = getSimVarByName(refAsStr, stringToSimVarHT)
          eqCounter += 1
          variablesForEq = BDAEUtil.getAllVariables(wstmt, algebraicAndStateVariables)
          variableEqMapping[eqCounter] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
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
  local variableEqMapping = OrderedDict{Int, Vector{Int}}()
  local unknownVariables = filter((x) -> BDAEUtil.isVariable(x.varKind), allBackendVars)
  local discreteVariables = filter((x) -> BDAEUtil.isDiscrete(x.varKind), allBackendVars)
  local stateVariables = filter((x) -> BDAEUtil.isState(x.varKind), allBackendVars)
  local algebraicAndStateVariables = vcat(unknownVariables, stateVariables)
  #= Name-keyed lookup so each equation scans its own crefs, not every model
     variable. =#
  local varByName = BDAEUtil.variablesByName(algebraicAndStateVariables)
  local nDiscretes = length(discreteVariables)
  @debug "#stateVariables" length(stateVariables)
  @debug "#algebraic" length(unknownVariables)
  @debug "#equations" length(equations)
  for eq in equations
    eqCounter += 1
    variablesForEq = Backend.BDAEUtil.getAllVariables(eq, varByName)
    variableEqMapping[eqCounter] = sort(getIndiciesOfVariables(variablesForEq, stringToSimVarHT))
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
                                stringToSimVarHT::OrderedDict{String, Tuple{Int, SimVar}})
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
    else
      continue
    end
  end
  return indicies
end

"""
    _recomputeSCCsFromSimCode(simCode) -> (sccs::Vector{Vector{Int}}, eq_to_var::Vector{String})

Re-derive the strongly-connected components of the residual equation set
using only the post-pipeline `simCode.residualEquations` and
`simCode.stringToSimVarHT`. The original SCCs computed at
`simulationCodeTransformation.jl:217` index into the pre-pass residual list
and are stale by codegen time; this rebuilds them on the array MTK will see.

Returns the SCC partition (vector of equation-index vectors) and the
matching `eq_to_var[i]` = name of the unknown that residual `i` is causally
solved for (empty string when unmatched).
"""
function _recomputeSCCsFromSimCode(simCode::SIM_CODE)
  local ht = simCode.stringToSimVarHT
  local res = simCode.residualEquations
  local n_eqs = length(res)
  local emptySCCs = Vector{Int}[]
  local emptyMatch = String[]
  n_eqs == 0 && return (emptySCCs, emptyMatch)
  local surviving = OrderedSet{String}(k for (k, (_, sv)) in pairs(ht) if isUnknownVarKind(sv.varKind))
  local incidence = Vector{OrderedSet{String}}(undef, n_eqs)
  for (i, eq) in enumerate(res)
    local names = OrderedSet{String}()
    collectCrefNames!(names, eq.exp)
    incidence[i] = intersect(names, surviving)
  end
  local var_to_eq = Dict{String, Int}()
  local eq_to_var = fill("", n_eqs)
  function augment!(eq_idx::Int, seen::OrderedSet{String})::Bool
    for var in incidence[eq_idx]
      var in seen && continue
      push!(seen, var)
      if !haskey(var_to_eq, var) || augment!(var_to_eq[var], seen)
        var_to_eq[var] = eq_idx
        eq_to_var[eq_idx] = var
        return true
      end
    end
    return false
  end
  for i in 1:n_eqs
    augment!(i, OrderedSet{String}())
  end
  local g = MetaGraphs.MetaDiGraph(n_eqs)
  for i in 1:n_eqs
    for v in incidence[i]
      local j = get(var_to_eq, v, 0)
      if j > 0 && j != i
        Graphs.add_edge!(g, i, j)
      end
    end
  end
  local sccs = GraphAlgorithms.stronglyConnectedComponents(g)
  return (sccs, eq_to_var)
end

"""
    recomputeStronglyConnectedComponents(simCode) -> SIM_CODE

SimCode pass: refresh `simCode.stronglyConnectedComponents` against the
current residual array so MTK codegen can act on accurate cycle info.
"""
function recomputeStronglyConnectedComponents(simCode::SIM_CODE)::SIM_CODE
  hasSubModels(simCode) && return simCode
  local (sccs, _) = _recomputeSCCsFromSimCode(simCode)
  local nCyclic = count(s -> length(s) > 1, sccs)
  if nCyclic > 0
    @debug "[SIMCODE: $(simCode.name): recomputeSCCs] cyclic SCCs found" nCyclic
  end
  @assign simCode.stronglyConnectedComponents = sccs
  return simCode
end
