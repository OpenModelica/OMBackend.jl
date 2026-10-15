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

"""
    classifyAdditionalDiscretes(simCode)

The algebraic variables a when-equation assigns (or reinits) as discrete: held between the
events (`T_start := time`), they have no residual, and MTK took them for extra unknowns.
Nothing for a model with structural modes or translated again at its structural changes.
"""
function classifyAdditionalDiscretes(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  isempty(simCode.whenEquations) && return simCode
  local targets = OrderedSet{String}()
  foreach(w -> whenTargets!(targets, w.whenEquation), simCode.whenEquations)
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
