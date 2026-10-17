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

#= The lowering of OMJLIR's variables to simulation variables, and their indices. =#

"The simulation variables by name, each with its index."
const HT = OrderedDict{String, Tuple{Int, SIMVAR}}

"""
    simVarKind(v, names)

The category of the variable `v` in the simulation code:

- a state; a Real algebraic variable;
- a record, a class or an external object parameter (and an external object variable), a
  constant: a module-level binding (DATA_STRUCTURE);
- a Real, Integer, Boolean or enumeration parameter; a String parameter or variable;
- an Integer, Boolean or enumeration variable, a discrete one: discrete;
- an array parameter, an array variable (kept as an array: its dimensions and binding).
"""
function simVarKind(v::IR.Variable, names::Names)::SimVarType
  local (k, ty) = (v.kind, v.ty)
  local binding = () -> v.binding === nothing ? nothing : simExp(names, v.binding)
  local known = k isa Union{IR.PARAMETER, IR.CONSTANT}
  k isa IR.STATE && return STATE()
  (k isa IR.ALGEBRAIC && ty isa IR.T_REAL) && return ALG_VARIABLE(0)
  (known && ty isa Union{IR.T_RECORD, IR.T_CLASS, IR.T_EXTERNAL_OBJECT}) && return DATA_STRUCTURE(binding())
  ty isa IR.T_EXTERNAL_OBJECT && return DATA_STRUCTURE(binding())
  ty isa Union{IR.T_REAL, IR.T_BOOLEAN, IR.T_INTEGER, IR.T_ENUMERATION} && known &&
    return k isa IR.PARAMETER ? PARAMETER(binding()) : DATA_STRUCTURE(binding())
  (known && ty isa IR.T_STRING) && return STRING(binding())
  ty isa IR.T_INTEGER && return DISCRETE()
  k isa Union{IR.DISCRETE, IR.CLOCKED_STATE} && return DISCRETE()
  ty isa IR.T_ARRAY && return known ? ARRAY_PARAMETER(arrayDimensions(v), binding()) : ARRAY(arrayDimensions(v), binding())
  (k isa IR.ALGEBRAIC && ty isa IR.T_STRING) && return STRING(binding())
  (k isa IR.ALGEBRAIC && ty isa Union{IR.T_BOOLEAN, IR.T_ENUMERATION}) && return DISCRETE()
  unsupported("a variable of this kind and type in the simulation code", "$(v.name): $(nameof(typeof(k))) $(IR.toString(ty))")
end

"The sizes of an array variable's dimensions."
function arrayDimensions(v::IR.Variable)::Vector{Int}
  local sizes = Int[]
  for d in (v.ty::IR.T_ARRAY).dimensions
    local n = IR.dimensionSize(d)
    n === nothing && unsupported("an array variable of a dimension of unknown size", v.name)
    push!(sizes, n)
  end
  return sizes
end

const STATE_SELECT_NAMES = Dict(IR.NEVER => :never, IR.AVOID => :avoid, IR.DEFAULT => :default,
                                IR.PREFER => :prefer, IR.ALWAYS => :always)

"The attributes of the variable `v` in the simulation code."
function simVarAttributes(v::IR.Variable, names::Names)::SimVarAttributes
  local a = v.attributes
  local e = x -> x === nothing ? nothing : simExp(names, x)
  return SimVarAttributes(e(a.start), e(a.fixed), e(a.nominal), e(a.min), e(a.max), e(a.unit), e(a.displayUnit),
                          e(a.quantity), a.stateSelect === nothing ? nothing : STATE_SELECT_NAMES[a.stateSelect],
                          v.isProtected)
end

"""
    createIndices(simVars)

The table of the simulation variables by name, each with its index (1-based, per category):
the states from 1, each followed by its derivative `der(x)` (the state's index); the
parameters and the module-level bindings from 1, in the variables' order with the states; then
the array parameters; the discrete variables (and the Strings) after the states; the algebraic
and array variables after the discretes, the algebraic ones with their sort index (after the
states) for the structural analysis.
"""
function createIndices(simVars::Vector{SIMVAR})::OrderedDict{String, Tuple{Int, SIMVAR}}
  local ht = OrderedDict{String, Tuple{Int, SIMVAR}}()
  local (stateCounter, parameterCounter) = (0, 0)
  local discretes = SIMVAR[]
  local arrayParameters = SIMVAR[]
  for v in simVars
    local k = v.varKind
    if k isa STATE
      stateCounter += 1
      local state = withIndex(v, stateCounter)
      ht[v.name] = (stateCounter, state)
      ht["der($(v.name))"] = (stateCounter, withIndex(v, stateCounter, STATE_DERIVATIVE(v.name)))
    elseif k isa Union{PARAMETER, DATA_STRUCTURE}
      parameterCounter += 1
      ht[v.name] = (parameterCounter, v)
    elseif k isa Union{DISCRETE, STRING}
      push!(discretes, v)
    elseif k isa ARRAY_PARAMETER
      push!(arrayParameters, v)
    end
  end
  local counter = parameterCounter
  for v in arrayParameters
    counter += 1
    ht[v.name] = (counter, withIndex(v, counter))
  end
  local discreteCounter = stateCounter
  for v in discretes
    discreteCounter += 1
    ht[v.name] = (discreteCounter, v)
  end
  local (index, sortIndex) = (discreteCounter, stateCounter)
  for v in simVars
    v.varKind isa Union{ALG_VARIABLE, ARRAY} || continue
    index += 1
    sortIndex += 1
    local kind = v.varKind isa ALG_VARIABLE ? ALG_VARIABLE(sortIndex) : v.varKind
    ht[v.name] = (index, withIndex(v, index, kind))
  end
  return ht
end
