module RuntimeUtil

import Absyn
import DifferentialEquations
import DifferentialEquations.ReturnCode
import DAE
import ModelingToolkit
import OMBackend
import OMBackend.@BACKEND_LOGGING

import OMFrontend
import OMFrontend.Frontend
import OMFrontend.Frontend.AbsynUtil
import OMFrontend.Frontend.SCodeUtil

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
                     uVec)
```
  This function maps variables between two models during a structural change with recompilation.
  It returns a new vector of u₀ variables to initialize the new model.
  We do so by assigning the old values when the structural change occurred for all variables
  that occurred in the model before the structural change.
"""
function createNewU0(symsOfOldProblem::Vector{Symbol},
                     symsOfNewProblem::Vector{Symbol},
                     initialValues,
                     uVec)
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
```
isReturnCodeSuccess(integrator)
```
Returns true if the current return code of the supplied integrator argument is `Success`.
"""
function isReturnCodeSuccess(integrator)
  integrator.sol.retcode == ReturnCode.Success
end

function getUnknowns(integrator)
  return [u for u in ModelingToolkit.unknowns(integrator.f.sys)]
end

function getUnknownsAsStringsNoPrefix(integrator)
  local oStrs = String[join(split(string(o.f.name), "_")[2:end], "_") for o in getUnknowns(integrator)]
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
