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

module RuntimeUtil

import Absyn
import DifferentialEquations
import DifferentialEquations.ReturnCode
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
    v === nothing || push!(pairs, (string(os.f.name), v))
  end
  return pairs
end

"""
    createLookupTable(integrator) -> Dict{String, Float64}

The values at the change of the integrator's unknowns, and of its observed variables that
resolve, by name: a recompilation instantiates the same model, whose variables have the same
names. The first `_` segment of every name was stripped before: different components' variables
became one (`body_frame_a_f`, `revolute_frame_a_f`), a name without `_` the empty one.
"""
function createLookupTable(integrator)::Dict{String, Float64}
  local values = Dict{String, Float64}(zip((string(u.f.name) for u in getUnknowns(integrator)),
                                           getValuesForUnknowns(integrator)))
  for (name, v) in _resolvableObservedPairs(integrator)
    get!(values, name, v)
  end
  return values
end

function getValuesForUnknowns(integrator)
  local uSyms = [uEq for uEq in ModelingToolkit.unknowns(integrator.f.sys)]
  local vals::Vector{Float64} = Float64[last(integrator.sol[u]) for u in uSyms]
  return vals
end

"""
    updateInitialConditions!(simCode, integrator) -> Nothing

The start values of the recompiled model's Real variables: their values at the change
(createLookupTable), by name, with their other attributes, not fixed (the new problem starts
from the state at the change: createNewU0, NoInit). A variable without one takes the value of
the old variable of its name without the component (`bouncingBall_y` the pendulum's
`pendulum_y`, a dynamic transition), fixed, when only one old variable has that name. Matching
every variable that way, fixed and its attributes replaced, made a MultiBody model's recompiled
system unbalanced (`body_frame_a_f` took `revolute_frame_a_f`'s value).
"""
function updateInitialConditions!(simCode::OMBackend.SimCode.SIM_CODE, integrator)::Nothing
  local SC = OMBackend.SimCode
  local values = createLookupTable(integrator)
  #= the old values by the name without the component; nothing when two old variables share it =#
  local byLocalName = Dict{String, Union{Nothing, Float64}}()
  for (name, v) in values
    local local_ = _withoutComponent(name)
    isempty(local_) && continue
    byLocalName[local_] = haskey(byLocalName, local_) ? nothing : v
  end
  local ht = simCode.stringToSimVarHT
  for (name, (idx, simVar)) in ht
    (simVar.elementType isa SC.TYPE_REAL || !SC.attributesGiven(simVar.elementType, simVar.attributes)) || continue
    local exact = get(values, name, nothing)
    local attributes = if exact !== nothing
      SC.withAttributes(simVar.attributes; start = SC.RCONST(exact),
                        fixed = SC.attributesGiven(simVar.elementType, simVar.attributes) ? simVar.attributes.fixed : SC.BCONST(false))
    else
      local other = get(byLocalName, _withoutComponent(name), nothing)
      other === nothing && continue
      SC.withAttributes(SC.NO_ATTRIBUTES; start = SC.RCONST(other), fixed = SC.BCONST(true))
    end
    ht[name] = (idx, SC.SIMVAR(simVar.name, simVar.index, simVar.varKind, attributes, simVar.elementType))
  end
  return nothing
end

"""
A flattened name without its first component (`pendulum_y` -> `y`; "" for `x`).
"""
_withoutComponent(name::String)::String = join(split(name, "_")[2:end], "_")

end #= module =#
