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

#= Running the passes of the simulation code: each followed by the removal of the trivial
   residuals, with the pass's metrics in the performance log. =#

"Structural counters of the simulation code: which pass reduced the system before MTK."
struct SimCodeMetrics
  residualEquations::Int
  initialEquations::Int
  ifEquations::Int
  ifBranches::Int
  conditionalResidualEquations::Int
  whenEquations::Int
  variables::Int
  unknowns::Int
  parameters::Int
  aliases::Int
  eliminatedVariables::Int
end

function simCodeMetrics(sc::SIM_CODE)::SimCodeMetrics
  local unknowns = count(((_, (_, v)),) -> isUnknownVarKind(v.varKind), sc.stringToSimVarHT)
  local parameters = count(((_, (_, v)),) -> isParameter(v), sc.stringToSimVarHT)
  return SimCodeMetrics(length(sc.residualEquations), length(sc.initialEquations), length(sc.ifEquations),
                        sum((length(e.branches) for e in sc.ifEquations); init = 0),
                        sum((length(b.residualEquations) for e in sc.ifEquations for b in e.branches); init = 0),
                        length(sc.whenEquations), length(sc.stringToSimVarHT), unknowns, parameters, length(sc.aliasMap),
                        length(sc.eliminatedVariables))
end

metricDelta(before::Int, after::Int)::String = before == after ? "$after" : "$before->$after"

"The changed metrics of a pass in the performance log (BACKEND_PERFLOG), else at the debug level."
function logSimCodePassMetrics(pass::AbstractString, before::SimCodeMetrics, after::SimCodeMetrics, elapsed::Real;
                               modelName::AbstractString = "")::Nothing
  before == after && return nothing
  local label = isempty(modelName) ? pass : string(modelName, ": ", pass)
  local metrics = NamedTuple(f => metricDelta(getfield(before, f), getfield(after, f)) for f in fieldnames(SimCodeMetrics))
  if BACKEND_PERFLOG[]
    @info "[SIMCODE: $label] metrics" elapsed_ms = round(1000 * elapsed, digits = 3) metrics
  else
    @debug "[SIMCODE: $label] metrics" elapsed_ms = round(1000 * elapsed, digits = 3) metrics
  end
  return nothing
end

logSimCodePassMetrics(pass::AbstractString, before::SimCodeMetrics, sc::SIM_CODE, elapsed::Real)::Nothing =
  logSimCodePassMetrics(pass, before, simCodeMetrics(sc), elapsed; modelName = sc.name)

"""
    runSimCodePass(name, simCode, pass; cleanup)

`pass(simCode)`, then (`cleanup`) without its trivial residuals; its metrics and allocations in
the performance log.
"""
function runSimCodePass(name::AbstractString, simCode::SIM_CODE, pass::F; cleanup::Bool = true)::SIM_CODE where {F}
  local perf = BACKEND_PERFLOG[]
  local before = perf ? simCodeMetrics(simCode) : nothing
  local stats = Base.@timed pass(simCode)
  local after = stats.value
  if perf
    logSimCodePassMetrics(name, before, after, stats.time)
    @info "[SIMCODE: $(simCode.name): $name] alloc" bytes = stats.bytes
  end
  return cleanup ? cleanupTrivialResidualEquations(after; sourcePass = name) : after
end
