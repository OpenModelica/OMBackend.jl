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
    introduceDerivatives!(model)

The states and their derivative variables: every variable `x` of a `der(x)` in the equations
of a continuous-time partition (or of a structural mode) is a state, its derivative the
variable `\$DER.x` (kind STATE_DER, after the model's variables; its reference `IR.derivativeCref` of x's), and
`der(x)` everywhere (equations and initial equations) a reference to it. With arrays kept, `der(x[i])` makes the
array `x` a state, `\$DER.x` its derivative array, and the call the element `\$DER.x[i]` (`IR.withLastSubscripts`). `der`
of another expression stays a call. The previous backend's detectStates: the states of the
equations, not of the initial equations.
"""
function introduceDerivatives!(model::IR.Model)::IR.Model
  for p in model.continuous
    introduceDerivatives!(model.variables, p)
  end
  for mode in model.modes
    introduceDerivatives!(mode.variables, mode.system)
  end
  return model
end

function introduceDerivatives!(variables::IR.VariableSet, partition::IR.Partition)::Nothing
  local states = Symbol[]
  local seen = Set{Symbol}()
  for eq in partition.equations
    IR.foreachExp(eq) do e
      (IR.isBuiltinCall(e, :der) && e.arguments[1] isa IR.CREF) || return
      #= an element of an array variable (a for-equation's `der(x[i])`, arrays kept): the array is the state =#
      local x = e.arguments[1].cref
      local array = arrayOfElement(variables, x)
      local name = array === nothing ? x.name : array.name
      name in seen || (push!(seen, name); push!(states, name))
    end
  end
  local introduced = Set{Symbol}()
  for name in states
    local v = get(variables, name, nothing)
    #= a reference that is no variable keeps its der() call =#
    v === nothing && continue
    push!(introduced, name)
    local d::IR.ComponentRef = IR.derivativeCref(v.cref)
    v.kind = IR.STATE(1, d.name, true)
    haskey(variables, d.name) && continue
    push!(variables, IR.Variable(d, v.ty, IR.STATE_DER(name), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                                 false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :introduceDerivatives))
  end
  local toDerivative = e -> begin
    (IR.isBuiltinCall(e, :der) && e.arguments[1] isa IR.CREF) || return e
    local x = e.arguments[1].cref
    x.name in introduced && return IR.CREF(variables[variables[x.name].kind.derivative].cref, e.ty)
    #= `der(x[i])` of an array state: the element `\$DER.x[i]` of its derivative array (`der(c[i].T)`, a field of
       a record array, `\$DER.c.T[i]`) =#
    local array = arrayOfElement(variables, x)
    (array !== nothing && array.name in introduced) || return e
    return IR.CREF(IR.withLastSubscripts(variables[variables[array.name].kind.derivative].cref, allSubscripts(x)), e.ty)
  end
  foreach(eq -> IR.mapEquation!(toDerivative, eq), partition.equations)
  foreach(eq -> IR.mapEquation!(toDerivative, eq), partition.initialEquations)
  return nothing
end

"""
    arrayOfElement(variables, cref) -> Union{Nothing, Variable}

The array variable whose element `cref` names (`x` of `x[i]`, `c.T` of `c[i].T`, a field of a
record array); `nothing` where `cref` is a variable itself or names no array.
"""
function arrayOfElement(variables::IR.VariableSet, cref::IR.ComponentRef)::Union{Nothing, IR.Variable}
  (haskey(variables, cref.name) || !IR.hasSubscripts(cref)) && return nothing
  local v = get(variables, unsubscriptedName(cref), nothing)
  return (v !== nothing && v.ty isa IR.T_ARRAY) ? v : nothing
end
