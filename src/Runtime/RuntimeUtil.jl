module RuntimeUtil

import Absyn
import DifferentialEquations
import DifferentialEquations.ReturnCode
import DAE
import ListUtil
import ModelingToolkit
import OMBackend
import OMBackend.@BACKEND_LOGGING
import OMBackend.SimulationCode

import OMFrontend
import OMFrontend.Frontend
import OMFrontend.Frontend.AbsynUtil
import OMFrontend.Frontend.SCodeUtil
import OMFrontend.Frontend.Util

import SCode

using MetaModelica

"""
  Wrapper to a function in SCodeUtil.
"""
function getElementFromSCodeProgram(inIdent::String, inClass::SCode.Element)
  result::SCode.Element = SCodeUtil.getElementNamed(inIdent, inClass)
  return result
end

"""
  Given a list of prefixes on the format <A>.<B>.<C>
  returns the element pointed to by C.
"""
function getElementFromSCodeProgram(prefixes::Vector{String}, inClass::SCode.Element)
  local currentElement::SCode.Element = inClass
  for p in prefixes[2:end]
    currentElement = getElementFromSCodeProgram(p, currentElement)
  end
  return currentElement
end

function _findClassPathByCanonicalName(target::String,
                                       inClass::SCode.Element,
                                       path::Vector{String})
  if OMBackend.canonicalName(join(path, ".")) == target
    return copy(path)
  end

  for element in listArray(SCodeUtil.getClassElements(inClass))
    if SCodeUtil.isClass(element)
      push!(path, SCodeUtil.elementName(element))
      local found = _findClassPathByCanonicalName(target, element, path)
      pop!(path)
      if found !== nothing
        return found
      end
    end
  end

  return nothing
end

function modelicaPathName(activeModeName::String, inClass::SCode.Element)::String
  if occursin(".", activeModeName)
    return activeModeName
  end

  local rootPath = String[SCodeUtil.elementName(inClass)]
  local found = _findClassPathByCanonicalName(OMBackend.canonicalName(activeModeName),
                                             inClass,
                                             rootPath)
  if found !== nothing
    return join(found, ".")
  end

  return activeModeName
end

"""

"""
function replaceElementInSCodeProgramByName(inClass, inElement, name::String)
  local path::Absyn.Path = AbsynUtil.stringPath(name)
  return SCodeUtil.replaceOrAddElementInProgram(list(inClass),
                                         inElement,
                                         path)
end

"""
```
setElementInSCodeProgram!(activeModeName,inIdent::String, newValue::T, inClass::SCode.Element)
```
Given a name sets that element to a new value.
It then returns the modified SCodeProgram.
Currently, it is assumed to be at the top level of the class.
A SCodeElement is either a component like a variable or a class.
See SCode.jl for info about the SCode representation.

TODO: Fix for all sub-levels as well
"""
function setElementInSCodeProgram!(activeModeName, inIdent::String, newValue::T, inClass::SCode.Element) where {T}
  #=
  Get the class name. The active class is required to be top level currently.
  =#
  local modelicaActiveModeName = modelicaPathName(activeModeName, inClass)
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "modename.log"), activeModeName)
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "modelica_modename.log"), modelicaActiveModeName)
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "original.log"), OMBackend.JuliaFormatter.format_text(string(inClass)))
  local activeModeNamePrefixes::Vector{String} = map(string, split(modelicaActiveModeName, "."))
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "prefixes.log"), OMBackend.JuliaFormatter.format_text(string(activeModeNamePrefixes)))
  local activeClass  = getElementFromSCodeProgram(activeModeNamePrefixes, inClass)
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "active.log"), OMBackend.JuliaFormatter.format_text(string(activeClass)))
  #= Get all elements from the class together with the corresponding names =#
  local elementToReplace::SCode.Element = getElementFromSCodeProgram(inIdent, activeClass)
  local elements::Vector{SCode.Element} = listArray(SCodeUtil.getClassElements(activeClass))
  local i = 1
  local indexOfElementToReplace = 0
  local modifiedClass = activeClass
  for element in elements
    if SCodeUtil.elementNameEqual(element, elementToReplace)
      indexOfElementToReplace = i
      break
    end
    i += 1
  end
  local modification = SCodeUtil.getComponentMod(elementToReplace)
  @assign modification.binding = makeCondition(newValue)
  @assign elementToReplace.modifications = modification
  elements[indexOfElementToReplace] = elementToReplace
  #write("elementToReplace.log", string(OMBackend.JuliaFormatter.format_text(string(elementToReplace))))
  @assign activeClass.classDef.elementLst = arrayList(elements)
  #=Replace the element in the specific class. =#
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "elementToReplace.log"), OMBackend.JuliaFormatter.format_text(string(elementToReplace)))
  @match modifiedProg <| MetaModelica.nil = replaceElementInSCodeProgramByName(inClass,
                                                                  activeClass,
                                                                  modelicaActiveModeName)
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "modifiedProg.log"), OMBackend.JuliaFormatter.format_text(string(modifiedProg)))
  @BACKEND_LOGGING write(OMBackend.logPath("backend/runtime", "tmpClass.log"), OMBackend.JuliaFormatter.format_text(string(inClass)))
  #=
  We need to update the top level class as well in this case.
  To do this we need to search for the element representing the class wee modified in the top level program again.
  =#
  return modifiedProg
end

makeCondition(val::Bool) = begin
  SOME(Absyn.BOOL(val))
end
makeCondition(val::Int) = begin
  SOME(Absyn.INTEGER(val))
end
makeCondition(val::Real) = begin
  SOME(Absyn.REAL(string(val)))
end
makeCondition(val) = begin
  throw("Only primitive values {Integer, Boolean, Real} are currently supported in a Recompilation call")
end

"""
  Converts a symbol (of a MTK variable) to a string.
"""
function convertSymbolToString(symbol::Symbol)
  res = replace(String(symbol), "(t)" => "")
  #= Remove prefixes in front of variables =#
  return res
end

"""
  Converts a list of symbols to a list of strings
"""
function convertSymbolsToStrings(symbols::Vector{Symbol})
  map(convertSymbolToString, symbols)
end

"""
```
createNewU0(symsOfOldProblem::Vector{Symbol},
                     symsOfNewProblem::Vector{Symbol},
                     newHT,
                     initialValues,
                     uVec,
                     specialCase)
```
  This function maps variables between two models during a structural change with recompilation.
  It returns a new vector of u₀ variables to initialize the new model.
  We do so by assigning the old values when the structural change occured for all variables
  that occured in the model before the structural change.
TODO:
Remove the special case.
The reason for it is in some example the containing model changes name wheras in others it does not.
Furthermore, the way indices are handled need to be fixed since the index after reduction in MTK is not the same as
the index in the simulation code stage of the backend.
For now we return the array for the special case with dynamic overconstrained connectors.
"""
function createNewU0(symsOfOldProblem::Vector{Symbol},
                     symsOfNewProblem::Vector{Symbol},
                     initialValues,
                     uVec,
                     specialCase)
  #=TODO: It was assumed to only be real variable not discretes, which might have other indices? =#
  # @info "Status length of both problems" begin
  #   length(symsOfOldProblem) length(symsOfNewProblem)
  #   "initialValues" initialValues
  #   "Old Problem" symsOfOldProblem
  #   "New Problem" symsOfNewProblem
  # end
  local newU0 = Float64[last(initialValues[idx]) for idx in 1:length(symsOfNewProblem)]
  local variableNamesOldProblem = RuntimeUtil.convertSymbolsToStrings(symsOfOldProblem)
  local variableNamesNewProblem = RuntimeUtil.convertSymbolsToStrings(symsOfNewProblem)
  #@info "variableNamesOldProblem" variableNamesOldProblem
  #@info "variableNamesNewProblem" variableNamesNewProblem
  #= In the special case all the indices are the same as they where before the transformation. =#
  if  specialCase
    return uVec
  end
  # strip only the leading submodel prefix (first underscore-segment);
  # greedy ".*_" would collapse distinct names sharing a final identifier
  local variableNamesWithoutPrefixesOP = String[replace(k, r"^[^_]*_" => "")
                                                for k in variableNamesOldProblem]
  local variableNamesWithoutPrefixesNP = String[replace(k, r"^[^_]*_" => "")
                                                for k in variableNamesNewProblem]
  #@info "variableNamesWithoutPrefixesOP" variableNamesWithoutPrefixesOP
  #= Build name-to-index lookup dicts for O(1) access instead of O(n) findall =#
  local oldNameToIdx = Dict{String,Int}(name => i for (i, name) in enumerate(variableNamesWithoutPrefixesOP))
  @assert(length(oldNameToIdx) == length(variableNamesWithoutPrefixesOP),
          "Duplicate variable name in old problem: $(length(variableNamesWithoutPrefixesOP)) names but $(length(oldNameToIdx)) unique")
  local newNameToIdx = Dict{String,Int}(name => i for (i, name) in enumerate(variableNamesWithoutPrefixesNP))
  @assert(length(newNameToIdx) == length(variableNamesWithoutPrefixesNP),
          "Duplicate variable name in new problem: $(length(variableNamesWithoutPrefixesNP)) names but $(length(newNameToIdx)) unique")
  local largestProblem = if length(variableNamesOldProblem) > length(variableNamesNewProblem)
    variableNamesWithoutPrefixesOP
  else
    variableNamesWithoutPrefixesNP
  end
  for v in largestProblem
    local idxOld = get(oldNameToIdx, v, 0)
    local idxNew = get(newNameToIdx, v, 0)
    if idxOld != 0 && idxNew != 0
      newU0[idxNew] = uVec[idxOld]
    end
  end
  return newU0
end

"""
  Gets the index from an entry to the symbol table
"""
function getIdxFromEntry(entry::Tuple)::Int
  first(entry)
end


"""
  Creates a new flat model.
  This model either has the equation specified in the if or these equations are to be added to the model
  when this condition is true.
  The name of this model
"""
function createNewFlatModel(flatModel,
                            unresolvedEquations,
                            newEquations)
  local newFlatModel =
    OMFrontend.Frontend.FLAT_MODEL(flatModel.name,
                               flatModel.variables,
                               flatModel.equations,
                               flatModel.initialEquations,
                               flatModel.algorithms,
                               flatModel.initialAlgorithms,
                               MetaModelica.nil,
                               NONE(),
                               MetaModelica.nil,
                               MetaModelica.nil,
                               Bool[], #= TODO: This equation might need to be changed. =#
                               flatModel.comment)
  @assign newFlatModel.equations = listAppend(unresolvedEquations, newEquations)
  # println("Unresolved System:")
  # println("********************************************************************")
  # for e in newFlatModel.equations
  #   println(OMFrontend.Frontend.toString(e))
  # end
  # println("********************************************************************")
  #=
    1. Reresolve the connect equations.
    2. Perform constant evaluation.
    3. Run the simplification pass.
  =#
  newFlatModel = OMFrontend.Frontend.resolveConnections(newFlatModel, newFlatModel.name)
  newFlatModel = OMFrontend.Frontend.evaluate(newFlatModel)
  newFlatModel = OMFrontend.Frontend.simplifyFlatModel(newFlatModel)
  # println("Final System")
  # @debug "Length of final system:" length(newFlatModel.equations)
  # println("********************************************************************")
  # for e in newFlatModel.equations
  #   println(OMFrontend.Frontend.toString(e))
  # end
  # println("********************************************************************")
  return newFlatModel
end

"""
  Creates a new flat model with a set of connection equation removed.
  Note that the flat model passed to this function does not have any active equations.
  This means that this flat model is a new flat model with the unbreakable branches removed.
"""
function createNewFlatModel(flatModel,
                            idx::Int,
                            unresolvedEquations)
  local aDoccs = flatModel.active_DOCC_Equations
  aDoccs[idx] = false
  local newFlatModel =
    OMFrontend.Frontend.FLAT_MODEL(flatModel.name,
                               flatModel.variables,
                               flatModel.unresolvedConnectEquations,
                               flatModel.initialEquations,
                               flatModel.algorithms,
                               flatModel.initialAlgorithms,
                               MetaModelica.nil,
                               NONE(),
                               flatModel.DOCC_equations,
                               flatModel.unresolvedConnectEquations,
                               aDoccs,
                               flatModel.comment)
  local variablestoReset = resolveDOOCConnections(flatModel, flatModel.name)
  #=
    1. Reresolve the connect equations.
    2. Perform constant evaluation.
    3. Run the simplification pass.
  =#
  newFlatModel = OMFrontend.Frontend.resolveConnections(newFlatModel, newFlatModel.name)
  newFlatModel = OMFrontend.Frontend.evaluate(newFlatModel)
  newFlatModel = OMFrontend.Frontend.simplifyFlatModel(newFlatModel)
  return newFlatModel
end

"""
  Resolves the system at the time of the structural change.
"""
function resolveDOOCConnections(flatModel, name)
  #= Get the relevant OCC graph =#
  local (searchGraph, rootVariables, rootEquations) = SimulationCode.getOCCGraph(flatModel)
  local pathsForRoots = Dict{String, Vector{String}}()
  for rv in rootVariables
    p = findPath(searchGraph, rv)
    pathsForRoots[OMFrontend.Frontend.toString(rv)] = p
  end
  for key in keys(pathsForRoots)
    vars = pathsForRoots[key]
  end
  local rootSources = Dict{String, String}()
  #= These are the equations for which the chain starts =#
  for (lhs, rhs) in rootEquations
    rootSources[OMFrontend.Frontend.toString(lhs)] = OMFrontend.Frontend.toString(rhs)
  end
  return (pathsForRoots, rootSources)
end

"""
Author:johti17
Iterative DFS:
  Finds the path for a root variable passed as inV (in Vertices)
"""
function findPath(g::Dict{String, Vector{String}}, inV)
  local v = OMFrontend.Frontend.toString(inV)
  local S = String[]
  local discovered = String[]
  local seen = Set{String}()
  push!(S, v)
  while !isempty(S)
    local v = pop!(S)
    if !(v in seen)
      push!(seen, v)
      push!(discovered, v)
      neighbours = g[v]
      for n in neighbours
        push!(S, n)
      end
    end
  end
  return discovered[2:end]
end

"""
Temporary function.
Evaluates a discrete events
Assuming one variable and one event that is changed.
TODO: Generalize later
"""
function evalDiscreteEvents(discreteEvents, u, t, system)
  local events = Tuple{Int, Bool, Bool}[]
  for de in discreteEvents
    push!(events, evalDiscreteEvent(de, u, t, system))
  end
  events = filter((x) -> last(x), events)
  return events
end

"""
TODO:
Refactor this function
"""
function evalDiscreteEvent(discreteEvent, u, time, system)
  @assert(length(discreteEvent.affects ) == 1, "Only length one of discrete affects supported")
  #@info "New iteration\n\n\n\n"
  local affect = first(discreteEvent.affects)
  local condition = discreteEvent.condition
  local args = condition.arguments
  local operator = condition.f
  local stateVars = ModelingToolkit.states(system)
  local lhs = first(args)
  local rhs = last(args)
  local lhsIdx::Int = 0
  local rhsIdx::Int = 0
  local isChanged = false
  local varDeps = Int[]
  #Assuming a ! for this case
  local shouldApplyNegation = false
  if length(args) == 1
    lhs = first(first(args).arguments)
    rhs = last(first(args).arguments)
    shouldApplyNegation = true
    operator = first(args).f
  end
  if typeof(lhs) != Float64 && string(lhs) != string(system.iv)
    lhsIdx = findfirst((x)->x==1, indexin(stateVars, [lhs]))
  end
  if typeof(rhs) != Float64 && string(rhs) != string(system.iv)
    rhsIdx = findfirst((x)->x==1, indexin(stateVars, [rhs]))
  end
  rhsValue = if rhsIdx != 0
    varDeps = getVariableEqDepedenceViaIdx(rhsIdx, system)
    #@info "varDeps rhs" varDeps
    rootIdx = getRootEquation(varDeps)
    getConstantValueOfEq(rootIdx, system)
  elseif string(rhs) == string(system.iv)
    time
  else
    rhs
  end
  lhsValue = if lhsIdx != 0
    varDeps = getVariableEqDepedenceViaIdx(lhsIdx, system)
    #@info "varDeps lhs" varDeps
    #@info "root eq" getRootEquation(varDeps)
    rootIdx = getRootEquation(varDeps)
    getConstantValueOfEq(rootIdx, system)
  elseif string(lhs) == string(system.iv)
    time
  else
    lhs
  end
  local affectIdx = findfirst((x)->x==1, indexin(stateVars, [affect.lhs]))
  local affectNewValue = affect.rhs
  #@info "lhs value was" lhsValue
  #@info "rhs value was" rhsValue
  if shouldApplyNegation
    if operator(lhsValue, rhsValue) == false
      #@info operator(lhsValue, rhsValue)
      #= Also assuming here that the lhs is a variable and the rhs is a value =#
      #@info "Branch 1 Value was changed" discreteEvent.condition.f
      isChanged = true
    end
  else
    if operator(lhsValue, rhsValue)
      isChanged = true
    end
  end
  return (affectIdx, affect.rhs, isChanged)
end

"""
 Given a variable index, returns the equations that the variable at this index is dependent on.
  That is, equations in which this variable is referenced.
"""
function getVariableEqDepedenceViaIdx(idx::Int, system)
  #= Get all equation dependencies for the current system =#
  local equationDependencies = ModelingToolkit.equation_dependencies(OMBackend.Runtime.REDUCED_SYSTEM)
  local vars = ModelingToolkit.states(OMBackend.Runtime.REDUCED_SYSTEM)
  local totalDependencies = Int[]
  #= Go through each equation =#
  for (equationIndex, equationDep) in enumerate(equationDependencies)
    #= Skip equations without dependencies =#
    if isempty(equationDep)
      continue
    end
    #=
      If the equation dependency is not empty
      Check if it depends on our variable
    =#
    if first(indexin([vars[idx]], equationDep)) !== nothing
      #=
      In this case we know that this equation is a depedency of the supplied variable.
      Add this equation as a possible depdency to totalDependencies
      =#
      push!(totalDependencies, equationIndex)
    end
  end
  #= We now have all indices of the variables our equation depends on =#
  return totalDependencies
end

"""
  Get the top level equation if such equation exist for a given set of equations
  Note that if the supplied equation is not solved at the top level, this function returns 0
"""
function getRootEquation(equationIndices; usedEqIndices = Set())::Int
  local G = ModelingToolkit.asgraph(OMBackend.Runtime.REDUCED_SYSTEM)
  local variableIdxToEquationIdx = G.badjlist
  local equationIdxToVariableIdx = G.fadjlist
  local idx = 0
  #= Shallow search. See if we can find the right equation directly =#
  for idx in equationIndices
    #= This equation does only depend on one variable. We are done =#
    if length(equationIdxToVariableIdx[idx]) == 1
      #= Then it is solved in this particular equation =#
      return idx
    end
    push!(usedEqIndices, idx)
  end
  for eqIdx in equationIndices
    for vIdx in equationIdxToVariableIdx[eqIdx]
      newEqIndices = filter((x) -> !(x in usedEqIndices), variableIdxToEquationIdx[vIdx])
      if isempty(newEqIndices)
        continue
      end
      idx = getRootEquation(newEqIndices; usedEqIndices = usedEqIndices)
    end
  end
  return idx
end

"""
Gets the constant value of an equation if such exist.
Throws an error otherwise
"""
function getConstantValueOfEq(eqIdx::Int, system)::Float64
  local equations = ModelingToolkit.equations(system)
  local equation = equations[eqIdx]
  @assert typeof(equation.lhs) != Number || typeof(equation.rhs) != Number "One side (lhs/rhs )needs to be a constant float"
  if equation.lhs isa Number
    return equation.lhs
  end
  return equation.rhs
end


function getCallbackSet(problem)
  last(last(problem.kwargs))
end

"""
```
isReturnCodeSuccess(integrator)
```
Returns true if the current return code of the supplied integrator argument is `Success`.
"""
function isReturnCodeSuccess(integrator)
  integrator.sol.retcode == ReturnCode.Success
end

"""
```
isReturnCodeDefault(integrator)
```
Returns true if the current return code of the supplied integrator argument is `Default`.
"""
function isReturnCodeDefault(integrator)
  integrator.sol.retcode == ReturnCode.Default
end

function getObserved(integrator)
  return [oEq.lhs for oEq in ModelingToolkit.observed(integrator.f.sys)]
end

function getUnknowns(integrator)
  return [u for u in ModelingToolkit.unknowns(integrator.f.sys)]
end

function getObservedAsStrings(integrator)
  local oStrs = String[string(o.f.name) for o in getObserved(integrator)]
end

function getUnknownsAsStrings(integrator)
  local oStrs = String[string(o.f.name) for o in getUnknowns(integrator)]
end

function getUnknownsAsStringsNoPrefix(integrator)
  local oStrs = String[join(split(string(o.f.name), "_")[2:end], "_") for o in getUnknowns(integrator)]
end

function getObservedAsStringsNoPrefix(integrator)
  local oStrs = String[join(split(string(o.f.name), "_")[2:end], "_") for o in getObserved(integrator)]
end

function _resolveObservedValue(integrator, os)
  local raw = integrator.sol[os]
  if raw isa AbstractArray
    isempty(raw) && return nothing
    return Float64(last(raw))
  end
  return Float64(raw)
end

function _resolvableObservedPairs(integrator)
  local pairs = Tuple{String, Float64}[]
  for oEq in ModelingToolkit.observed(integrator.f.sys)
    local os = oEq.lhs
    local v = try
      _resolveObservedValue(integrator, os)
    catch e
      e isa InterruptException && rethrow()
      nothing
    end
    if v !== nothing
      local name = join(split(string(os.f.name), "_")[2:end], "_")
      push!(pairs, (name, v))
    end
  end
  return pairs
end

function createLookupTableForObserved(integrator)
  return Dict{String, Float64}(_resolvableObservedPairs(integrator))
end

function createLookupTable(integrator)
  local unknownNames = getUnknownsAsStringsNoPrefix(integrator)
  local unknownVals = getValuesForUnknowns(integrator)
  local d = Dict{String, Float64}(zip(unknownNames, unknownVals))
  for (name, v) in _resolvableObservedPairs(integrator)
    if haskey(d, name)
      @warn "createLookupTable: observed name `$name` collides with an unknown of the same name after prefix stripping; keeping the unknown's value."
    else
      d[name] = v
    end
  end
  return d
end

function getPrefix(integrator)
  local u = first(ModelingToolkit.unknowns(integrator.f.sys))
  return first(split(string(u.f.name), "_"))
end

function getValuesForObserved(integrator)
  return Float64[v for (_, v) in _resolvableObservedPairs(integrator)]
end

function getValuesForUnknowns(integrator)
  local uSyms = [uEq for uEq in ModelingToolkit.unknowns(integrator.f.sys)]
  local vals::Vector{Float64} = Float64[last(integrator.sol[u]) for u in uSyms]
  return vals
end

"""
updateObservedVariables in the simCode
"""
function updateInitialConditions!(simCode, integrator)
  LT = createLookupTable(integrator)
  local vNSys = String[join(split(vs, "_")[2:end], "_") for (i, vs) in enumerate(keys(simCode.stringToSimVarHT))]
  indices = indexin(keys(LT), vNSys)
  local simCode_LT = simCode.stringToSimVarHT
  for (i, name) in enumerate(keys(LT))
    local indexInNewSys = indices[i]
    if indexInNewSys !== nothing
      (idx, vToChange) = simCode_LT.vals[indexInNewSys]
      # `start` kwarg of `DAE.makeRealAttribute` is typed `Option{Float64}`
      # (= `Union{Nothing, SOME{Float64}}`), not raw `Float64`. Without the
      # SOME wrapper, the call fails with `TypeError: in keyword argument start, ...`.
      vToChange = @assign vToChange.attributes = SOME(DAE.makeRealAttribute(;start=SOME(LT[name]), fixed=true))
      simCode_LT.vals[indexInNewSys] = (idx, vToChange)
    end
  end
end

end #= module =#
