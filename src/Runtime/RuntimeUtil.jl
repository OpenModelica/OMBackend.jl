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

import DifferentialEquations
import DifferentialEquations.ReturnCode
import ModelingToolkit
import OMBackend
import OMBackend.@BACKEND_LOGGING

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
The symbol table of a model's simulation code (`SIM_CODE.stringToSimVarHT`): a variable's index and its
specification by its name.
"""
const SymbolTable = OMBackend.SimCode.OrderedDict{String, Tuple{Int, OMBackend.SimCode.SIMVAR}}

"""
    switchStartValues(ht::SymbolTable, values::Dict{String, Float64}) -> Dict{String, Tuple{Float64, Bool}}

The start values a structural change gives the Real variables of the recompiled model's symbol table `ht`, from the
old model's values at the change `values` (createLookupTable): the value of a variable's own name, else (`true`, by
the local name) the value of the old variable of its name without the component (`bouncingBall_y` the pendulum's
`pendulum_y`, a dynamic transition) when only one old variable has that name. `updateInitialConditions!` sets them;
a recompiled mode compiled at an earlier change compares them (`Runtime.RecompiledMode`).
"""
function switchStartValues(ht::SymbolTable, values::Dict{String, Float64})::Dict{String, Tuple{Float64, Bool}}
  local SC = OMBackend.SimCode
  #= the old values by the name without the component; nothing when two old variables share it =#
  local byLocalName::Dict{String, Union{Nothing, Float64}} = Dict{String, Union{Nothing, Float64}}()
  for (name, v) in values
    local local_::String = _withoutComponent(name)
    isempty(local_) && continue
    byLocalName[local_] = haskey(byLocalName, local_) ? nothing : v
  end
  local starts::Dict{String, Tuple{Float64, Bool}} = Dict{String, Tuple{Float64, Bool}}()
  for (name, (_, simVar)) in ht
    (simVar.elementType isa SC.TYPE_REAL || !SC.attributesGiven(simVar.elementType, simVar.attributes)) || continue
    local exact::Union{Nothing, Float64} = get(values, name, nothing)
    if exact !== nothing
      starts[name] = (exact, false)
    else
      local other::Union{Nothing, Float64} = get(byLocalName, _withoutComponent(name), nothing)
      other === nothing || (starts[name] = (other, true))
    end
  end
  return starts
end

"""
    updateInitialConditions!(simCode, starts::Dict{String, Tuple{Float64, Bool}}) -> Nothing

The start values of the recompiled model's Real variables, `starts` (`switchStartValues`): a value of the variable's
own name with its other attributes, not fixed (the new problem starts from the state at the change: createNewU0,
NoInit); a value by the local name fixed, its other attributes dropped. Matching every variable that way, fixed and
its attributes replaced, made a MultiBody model's recompiled system unbalanced (`body_frame_a_f` took
`revolute_frame_a_f`'s value).
"""
function updateInitialConditions!(simCode::OMBackend.SimCode.SIM_CODE, starts::Dict{String, Tuple{Float64, Bool}})::Nothing
  local SC = OMBackend.SimCode
  local ht::SymbolTable = simCode.stringToSimVarHT
  for (name, (value, byLocalName)) in starts
    local (idx::Int, simVar::SC.SIMVAR) = ht[name]
    local attributes::SC.SimVarAttributes = if byLocalName
      SC.withAttributes(SC.NO_ATTRIBUTES; start = SC.RCONST(value), fixed = SC.BCONST(true))
    else
      SC.withAttributes(simVar.attributes; start = SC.RCONST(value),
                        fixed = SC.attributesGiven(simVar.elementType, simVar.attributes) ? simVar.attributes.fixed : SC.BCONST(false))
    end
    ht[name] = (idx, SC.withAttributes(simVar, attributes))
  end
  return nothing
end

"""
A flattened name without its first component (`pendulum_y` -> `y`; "" for `x`).
"""
_withoutComponent(name::String)::String = join(split(name, "_")[2:end], "_")

end #= module =#
