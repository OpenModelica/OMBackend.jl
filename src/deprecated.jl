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

#= The names of OMBackend 0.x that 1.0 renamed: kept for all of 1.x with a deprecation warning (shown with
   `--depwarn=yes`), not public, not documented but on the API page's list (docs/src/api.md). =#

"""
    _deprecatedKeyword(f, old, value, new, current) -> value or current

`value` of the deprecated keyword `old` of `f` where it was given (`value` not `nothing`), with a deprecation warning;
else `current`, the value of its new name `new`.
"""
function _deprecatedKeyword(f::Symbol, old::Symbol, value::Union{Nothing, T}, new::Symbol, current::T)::T where {T}
  value === nothing && return current
  Base.depwarn("the keyword `$(old)` of `$(f)` is deprecated, use `$(new)`", f)
  return value
end

function simulateModel(modelName::String; MODE::BackendMode = DEFAULT_BACKEND_MODE[], kwargs...)
  Base.depwarn("`simulateModel(name; MODE)` is deprecated, use `simulate(name; mode)`", :simulateModel)
  return simulate(modelName; mode = MODE, kwargs...)
end

function resimulateModel(modelName::String; MODE::BackendMode = DEFAULT_BACKEND_MODE[], kwargs...)
  Base.depwarn("`resimulateModel(name; MODE)` is deprecated, use `resimulate(name; mode)`", :resimulateModel)
  return resimulate(modelName; mode = MODE, kwargs...)
end

function getVariableValues(sol::ODESolution, varName::String)
  Base.depwarn("`getVariableValues` is deprecated, use `variableValues`", :getVariableValues)
  return variableValues(sol, varName)
end

function getVariableValues(sols::Vector, variables...)
  Base.depwarn("`getVariableValues` is deprecated, use `variableValues`", :getVariableValues)
  return variableValues(sols, variables...)
end

function translateReducedArrays(fm; BackendMode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[], kwargs...)
  Base.depwarn("`translateReducedArrays(fm; BackendMode)` is deprecated, use `translate(fm; scalarize = false, mode)`",
               :translateReducedArrays)
  return translate(fm; scalarize = false, mode = BackendMode, kwargs...)
end
