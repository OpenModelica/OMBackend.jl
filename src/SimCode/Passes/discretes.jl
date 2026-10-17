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

#= Discrete variables the causalization left algebraic. =#

"The variables the statements of a when-clause (and its elsewhen arms) assign or reinit."
function whenTargets!(names::OrderedSet{String}, w::WHEN_STMTS)::OrderedSet{String}
  for s in w.body
    s isa ASSIGN && s.left isa EXP_CREF && push!(names, crefName(s.left.cref))
    s isa REINIT && push!(names, crefName(s.stateVar))
  end
  w.elsewhenPart === nothing || whenTargets!(names, w.elsewhenPart)
  return names
end

"The variables the when-equations assign or reinit (their elsewhen arms too)."
function whenAssignedNames(simCode::SIM_CODE)::OrderedSet{String}
  local targets = OrderedSet{String}()
  foreach(w -> whenTargets!(targets, w.whenEquation), simCode.whenEquations)
  return targets
end

"""
    classifyAdditionalDiscretes(simCode)

The algebraic variables a when-equation assigns (or reinits) as discrete: held between the
events (`T_start := time`), they have no residual, and MTK took them for extra unknowns.
Nothing for a model with structural modes or translated again at its structural changes.
"""
function classifyAdditionalDiscretes(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  isempty(simCode.whenEquations) && return simCode
  local targets = whenAssignedNames(simCode)
  local ht = simCode.stringToSimVarHT
  local algebraic = String[n for n in targets if haskey(ht, n) && last(ht[n]).varKind isa ALG_VARIABLE]
  isempty(algebraic) && return simCode
  local newHT = copy(ht)
  for n in algebraic
    local (index, v) = newHT[n]
    newHT[n] = (index, withKind(v, DISCRETE()))
  end
  return update(simCode; stringToSimVarHT = newHT)
end
