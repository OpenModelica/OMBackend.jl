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
variable `\$DER.x` (kind STATE_DER, after the model's variables), and `der(x)` everywhere
(equations and initial equations) a reference to it. `der` of another expression stays a
call. The previous backend's detectStates: the states of the equations, not of the initial
equations.
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
  local states = IR.ComponentRef[]
  local seen = Set{Symbol}()
  for eq in partition.equations
    IR.foreachExp(eq) do e
      (IR.isBuiltinCall(e, :der) && e.arguments[1] isa IR.CREF) || return
      local x = e.arguments[1].cref
      x.name in seen || (push!(seen, x.name); push!(states, x))
    end
  end
  local introduced = Set{Symbol}()
  for x in states
    local v = get(variables, x.name, nothing)
    #= a reference that is no variable keeps its der() call =#
    v === nothing && continue
    push!(introduced, x.name)
    local d = IR.derivativeName(x.name)
    v.kind = IR.STATE(1, d, true)
    haskey(variables, d) && continue
    push!(variables, IR.Variable(d, v.ty, IR.STATE_DER(x.name), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                                 false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :introduceDerivatives))
  end
  local toDerivative = e -> begin
    (IR.isBuiltinCall(e, :der) && e.arguments[1] isa IR.CREF && e.arguments[1].cref.name in introduced) || return e
    return IR.CREF(IR.ComponentRef(IR.derivativeName(e.arguments[1].cref.name), e.ty), e.ty)
  end
  foreach(eq -> IR.mapEquation!(toDerivative, eq), partition.equations)
  foreach(eq -> IR.mapEquation!(toDerivative, eq), partition.initialEquations)
  return nothing
end
