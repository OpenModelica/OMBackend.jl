#= Running a SimCode pass (metrics, cleanup), and the structural-variation predicates. =#

#= SIM_CODE-level structural-variation axes. Different lowering passes guard
   on different combinations; do not collapse into one predicate. =#
hasStructuralTransitions(simCode)::Bool = !isempty(simCode.structuralTransitions)
hasSubModels(simCode)::Bool = !isempty(simCode.subModels)
hasMetaModel(simCode)::Bool = !isnothing(simCode.metaModel)

"""
Compact structural counters for the SIM_CODE optimization pipeline.
These are intentionally cheap: they help identify which simcode pass reduced
the system before MTK sees it without walking every expression.
"""
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

function simCodeMetrics(simCode::SIM_CODE)::SimCodeMetrics
  local nUnknowns = 0
  local nParameters = 0
  for (_, simVar) in values(simCode.stringToSimVarHT)
    if isUnknownVarKind(simVar.varKind)
      nUnknowns += 1
    elseif isParameter(simVar)
      nParameters += 1
    end
  end
  local nIfBranches = 0
  local nConditionalResiduals = 0
  for ifEq in simCode.ifEquations
    nIfBranches += length(ifEq.branches)
    for branch in ifEq.branches
      nConditionalResiduals += length(branch.residualEquations)
    end
  end
  return SimCodeMetrics(length(simCode.residualEquations),
                        length(simCode.initialEquations),
                        length(simCode.ifEquations),
                        nIfBranches,
                        nConditionalResiduals,
                        length(simCode.whenEquations),
                        length(simCode.stringToSimVarHT),
                        nUnknowns,
                        nParameters,
                        length(simCode.aliasMap),
                        length(simCode.eliminatedVariables))
end

function _metricDelta(before::Int, after::Int)::String
  return before == after ? "$after" : "$before->$after"
end

function logSimCodePassMetrics(passName::AbstractString,
                               before::SimCodeMetrics,
                               after::SimCodeMetrics,
                               elapsed_s::Real;
                               modelName::AbstractString = "")
  if before == after
    return nothing
  end
  local label = isempty(modelName) ? passName : Base.string(modelName, ": ", passName)
  if OMBackend.BACKEND_PERFLOG[]
    @info "[SIMCODE: $label] metrics" elapsed_ms=round(1000 * elapsed_s, digits = 3) residuals=_metricDelta(before.residualEquations, after.residualEquations) initial=_metricDelta(before.initialEquations, after.initialEquations) ifEquations=_metricDelta(before.ifEquations, after.ifEquations) ifBranches=_metricDelta(before.ifBranches, after.ifBranches) conditionalResiduals=_metricDelta(before.conditionalResidualEquations, after.conditionalResidualEquations) variables=_metricDelta(before.variables, after.variables) unknowns=_metricDelta(before.unknowns, after.unknowns) parameters=_metricDelta(before.parameters, after.parameters) aliases=_metricDelta(before.aliases, after.aliases) eliminatedVariables=_metricDelta(before.eliminatedVariables, after.eliminatedVariables)
  else
    @debug "[SIMCODE: $label] metrics" elapsed_ms=round(1000 * elapsed_s, digits = 3) residuals=_metricDelta(before.residualEquations, after.residualEquations) initial=_metricDelta(before.initialEquations, after.initialEquations) ifEquations=_metricDelta(before.ifEquations, after.ifEquations) ifBranches=_metricDelta(before.ifBranches, after.ifBranches) conditionalResiduals=_metricDelta(before.conditionalResidualEquations, after.conditionalResidualEquations) variables=_metricDelta(before.variables, after.variables) unknowns=_metricDelta(before.unknowns, after.unknowns) parameters=_metricDelta(before.parameters, after.parameters) aliases=_metricDelta(before.aliases, after.aliases) eliminatedVariables=_metricDelta(before.eliminatedVariables, after.eliminatedVariables)
  end
  return nothing
end

function logSimCodePassMetrics(passName::AbstractString,
                               before::SimCodeMetrics,
                               simCode::SIM_CODE,
                               elapsed_s::Real)
  return logSimCodePassMetrics(passName, before, simCodeMetrics(simCode), elapsed_s; modelName = Base.string(simCode.name))
end

function runSimCodePass(passName::AbstractString,
                        simCode::SIM_CODE,
                        passFn::Function;
                        cleanup::Bool = true)::SIM_CODE
  #= Pass metrics feed only the perf log, so compute them only when perf logging
     is on; otherwise skip the two full hash-table sweeps per pass. Pass execution
     and cleanup are unchanged, so this is codegen-neutral. =#
  local perf = OMBackend.BACKEND_PERFLOG[]
  local before = perf ? simCodeMetrics(simCode) : nothing
  local stats = Base.@timed passFn(simCode)
  local afterPass = stats.value
  if perf
    logSimCodePassMetrics(passName, before, afterPass, stats.time)
    @info "[SIMCODE: $(simCode.name): $passName] alloc" bytes=stats.bytes
  end
  if cleanup
    afterPass = cleanupTrivialResidualEquations(afterPass; sourcePass = passName)
  end
  return afterPass
end
