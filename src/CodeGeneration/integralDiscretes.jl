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

#= Integer, Boolean and enumeration variables hold integral values. They
   are unknowns with der(v) = 0, and a Rosenbrock or BDF step can leave
   round-off in them (pivoting in the linear solve mixes rows):
   3 becomes 3.0000000000000004, and change(v) or v == 3 would see it. =#

"""
    withIntegralDiscretes(callbacks, problem, names) -> callbacks

Add a DiscreteCallback ahead of the other discrete callbacks (before the event
iteration reads them), that rounds the unknowns `names` back to integers
after a step that left round-off in them.
"""
function withIntegralDiscretes(callbacks, problem, names::Vector{String})
  local states = getStatesAsSymbols(problem.f)
  local idx = Int[k for k in indexin(Symbol.(names), states) if k !== nothing]
  isempty(idx) && return callbacks
  local drifted = (u, t, integrator) -> any(k -> u[k] != round(u[k]), idx)
  local round! = function (integrator)
    for k in idx
      integrator.u[k] = round(integrator.u[k])
    end
    return nothing
  end
  local cb = DiffEqBase.DiscreteCallback(drifted, round!; save_positions = (false, false),
                                         initializealg = ModelingToolkit.SciMLBase.NoInit())
  return callbacks === nothing ? DiffEqBase.CallbackSet(cb) : DiffEqBase.CallbackSet(cb, callbacks)
end
