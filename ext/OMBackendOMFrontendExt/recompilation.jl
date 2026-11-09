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

"""
    Recompilation

The frontend's part of a structural change that recompiles the model (`OMBackend.Runtime.recompilation`): the
model's SCode class changed as the modification says, instantiated again with the classes the model was instantiated
with (a library: the MSL for a model using it), and through the backend (`recompiledSimCode`). The runtime around it
(the new problem, the transfer of the state) is OMBackend's.
"""
module Recompilation

import OMParser.Absyn
import OMFrontend
import OMFrontend.Frontend
import OMFrontend.Frontend.AbsynUtil
import OMFrontend.Frontend.SCodeUtil
import OMFrontend.SCode
import OMBackend
import OMBackend: SimCode, @BACKEND_LOGGING, @VSS_DEBUG
import OMBackend.Runtime: recompiledSimCode
import ..LowerFlatModel: FunctionList

using MetaModelica

const F = OMFrontend.Frontend

"""
    recompiledSimCode(metaModel::SCode.CLASS, activeModeName::String, elementToChange::String, newValue::Real;
                      started::Float64 = time()) -> SimCode.SIM_CODE

The simulation code of the model `metaModel` (its SCode class) recompiled with the element `elementToChange` of the
active mode `activeModeName` set to `newValue` (`setElementInSCodeProgram!`), instantiated again with the classes it
was instantiated with (OMFrontend.recompilationProgram) and through the backend (`translateToSimCode`).
"""
function recompiledSimCode(metaModel::SCode.CLASS, activeModeName::String, elementToChange::String, newValue::Real;
                           started::Float64 = time())::SimCode.SIM_CODE
  local modelicaActiveModeName::String = modelicaPathName(activeModeName, metaModel)
  local newProgram::SCode.Program = setElementInSCodeProgram!(activeModeName, elementToChange, newValue, metaModel) <|
    OMFrontend.recompilationProgram(SCodeUtil.elementName(metaModel))
  @VSS_DEBUG @info "[RECOMPILATION] step 2/7: SCode mutated" elapsed_s=round(time()-started, digits=2)
  local (flatModelica::F.FlatModel, functionCache) = OMFrontend.instantiateSCodeToFM(modelicaActiveModeName, newProgram)
  @VSS_DEBUG @info "[RECOMPILATION] step 3/7: frontend (instantiateSCodeToFM) done" elapsed_s=round(time()-started, digits=2)
  return translateToSimCode(flatModelica, OMFrontend.cacheToFunctionList(functionCache))
end

"""
    translateToSimCode(flatModelica, functionList) -> SIM_CODE

The SimCode of a recompiled model with its functions (`functionList`, the frontend's), as a
translate's (OMBackend.omjlirModel, OMBackend.mtkSimCode).
"""
translateToSimCode(flatModelica::F.FlatModel, functionList::Union{Nothing, FunctionList})::SimCode.SIM_CODE =
  OMBackend.mtkSimCode(OMBackend.omjlirModel(flatModelica; functionList))

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

end # module Recompilation
