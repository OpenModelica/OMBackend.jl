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

#= The passes of the simulation code, in the order the code generators take it. =#

"Run the named passes in order (each followed by the trivial-residual cleanup)."
function runPasses(simCode::SIM_CODE, passes::Vector{<:Pair{String}})::SIM_CODE
  for (name, pass) in passes
    simCode = runSimCodePass(name, simCode, pass)
  end
  return simCode
end

"The SimCode passes `simulationCode` runs first, in order (before the right-hand-side equivalence's fixed point)."
const FIRST_PASSES = Pair{String, Function}[
  "resolveIfExpInBindings" => resolveIfExpInBindings!,
  "pruneConstantConditions" => pruneConstantConditions,
  "foldParameterClosure" => foldParameterClosure,
  "inlinePreOfConstantParameters" => inlinePreOfConstantParameters,
  "propagateConstants" => propagateConstants,
  "demoteDeadStates" => demoteDeadStates,
  "lowerComplexOperatorRecords" => lowerComplexOperatorRecords,
  "eliminateAliasVariables" => eliminateAliasVariables,
  "eliminateRHSEquivalentEquations" => eliminateRHSEquivalentEquations,
  "simplifySemiLinear" => simplifySemiLinear,
  "removeRedundantEquations" => removeRedundantEquations,
  "eliminateConstantParameters" => eliminateConstantParameters,
  "dropObservationOnlyVariables" => dropObservationOnlyVariables,
  "eliminateDeadParameters" => eliminateDeadParameters,
  "evaluateConstantFunctionOutputs" => evaluateConstantFunctionOutputs,
  "eliminateFrozenStates" => eliminateFrozenStates,
  "demoteDeadStates" => demoteDeadStates,
  "orientReciprocalEquations" => orientReciprocalEquations,
  "foldExplicitSingleAssign" => foldExplicitSingleAssign,
  "eliminateAliasVariables" => eliminateAliasVariables,
  "simplifySemiLinear" => simplifySemiLinear]

"""
    simulationCodeThroughFunctionOutputs(model) -> SIM_CODE

The simulation code of `model` through its first passes up to the evaluation of constant function
outputs (`evaluateConstantFunctionOutputs`): the constants of the parameters' values and of
functions' outputs (MSL Spice3's capacitances), not those of the frozen states and later folds.
What index reduction plans with (`OMBackend.knownValues`).
"""
function simulationCodeThroughFunctionOutputs(model::IR.Model)::SIM_CODE
  local simCode = runSimCodePass("flattenRecordCallSites", fromOMJLIR(model), flattenRecordCallSites)
  local k = findfirst(p -> first(p) == "evaluateConstantFunctionOutputs", FIRST_PASSES)
  return runPasses(simCode, FIRST_PASSES[1:k])
end

"""
    simulationCode(model; observedFilter = nothing, eliminateNonDynamic = nothing)

The simulation code the MTK and IMTK code generators take, of the OMJLIR model `model` (after
its passes): lowered (`fromOMJLIR`), its record call sites flattened, through the SimCode
passes: the parameters' if-bindings and constant conditions resolved, parameter closures
folded, constants propagated (after `pre()` of parameters inlined), the states without der()
demoted, Complex operator records lowered, aliases and equations of equal right-hand sides
eliminated, semiLinear chains simplified, redundant residuals removed, constant and dead
parameters and observation-only variables dropped, constant function outputs evaluated, frozen
states eliminated (the states without der() then demoted again: a frozen factor drops
`0 * der(x)`), reciprocal equations oriented and explicit assignments folded, aliases eliminated again (the folds join
flow sums), the right-hand-side equivalence to a fixed point (at most 6 rounds), the discretes
of when-equations classified, the output-only variables eliminated (`eliminateNonDynamic`),
the alias observations filtered (`observedFilter`, else the user-visible names: no `[`), the
initial values propagated.
"""
function simulationCode(model::IR.Model; observedFilter::Union{Nothing, Vector{String}} = nothing,
                        eliminateNonDynamic::Union{Nothing, Bool, EliminationOptions} = nothing)::SIM_CODE
  local simCode = runSimCodePass("flattenRecordCallSites", fromOMJLIR(model), flattenRecordCallSites)
  observedFilter === nothing || (simCode = update(simCode; observedFilter))
  simCode = runPasses(simCode, FIRST_PASSES)
  #= the second alias elimination can make two residuals read one der(s): to a fixed point =#
  local previous = -1
  for _ in 1:6
    simCode = runSimCodePass("eliminateRHSEquivalentEquations", simCode, eliminateRHSEquivalentEquations)
    length(simCode.residualEquations) == previous && break
    previous = length(simCode.residualEquations)
  end
  simCode = runPasses(simCode, ["classifyAdditionalDiscretes" => classifyAdditionalDiscretes,
                                "pruneConstantConditions" => pruneConstantConditions])
  #= after OMBackend's index reduction: der() of the dummy states the aliases left in initial equations, negated left-hand sides =#
  simCode.indexReduced && (simCode = runPasses(simCode, ["derivativesOfDummyStates" => derivativesOfDummyStates,
                                                         "unnegatedInitialEquations" => unnegatedInitialEquations]))
  local options = eliminateNonDynamic === true ? EliminationOptions() :
    eliminateNonDynamic isa EliminationOptions ? eliminateNonDynamic : nothing
  options === nothing || (simCode = runSimCodePass("eliminateNonDynamic", simCode, sc -> eliminateOutputOnlyVariables(sc, options)))
  simCode = runSimCodePass("observedFilter", simCode, filterAliasObservations)
  simCode = runPasses(simCode, ["propagateInitialValues" => propagateInitialValues])
  return indexOverconstraintDiagnostic(simCode)
end

"""
    filterAliasObservations(simCode)

The alias observations the code generators emit as observed equations: those the observed
filter's patterns match; without a filter, those of a user-visible eliminated name (no `[`:
not a scalarized array element; MultiBody's hundreds of internals) whose representative is
still a variable (an eliminated one, a frozen state's, would leave the observation unbound).
"""
function filterAliasObservations(simCode::SIM_CODE)::SIM_CODE
  isempty(simCode.aliasMap) && return simCode
  local ht = simCode.stringToSimVarHT
  local keep = if simCode.observedFilter === nothing
    e::AliasEntry -> !occursin('[', e.eliminatedName) && haskey(ht, e.representativeName)
  else
    local patterns = Regex[Regex(p) for p in simCode.observedFilter]
    e::AliasEntry -> any(p -> occursin(p, e.eliminatedName), patterns)
  end
  return update(simCode; aliasMap = filter(keep, simCode.aliasMap))
end

"""
    deSimulationCode(model)

The simulation code the DifferentialEquations.jl code generator takes (DEMode): the MTK path's
first passes (no Complex lowering, observation-only or dead parameters, function evaluation,
folds), no observed equations.
"""
function deSimulationCode(model::IR.Model)::SIM_CODE
  local simCode = runSimCodePass("flattenRecordCallSites", fromOMJLIR(model), flattenRecordCallSites)
  return runPasses(simCode, [
    "resolveIfExpInBindings" => resolveIfExpInBindings!,
    "pruneConstantConditions" => pruneConstantConditions,
    "foldParameterClosure" => foldParameterClosure,
    "inlinePreOfConstantParameters" => inlinePreOfConstantParameters,
    "propagateConstants" => propagateConstants,
    "demoteDeadStates" => demoteDeadStates,
    "eliminateAliasVariables" => eliminateAliasVariables,
    "eliminateRHSEquivalentEquations" => eliminateRHSEquivalentEquations,
    "simplifySemiLinear" => simplifySemiLinear,
    "removeRedundantEquations" => removeRedundantEquations,
    "eliminateConstantParameters" => eliminateConstantParameters,
    "eliminateFrozenStates" => eliminateFrozenStates,
    "demoteDeadStates" => demoteDeadStates,
    "pruneConstantConditions" => pruneConstantConditions])
end
