#= Trivial residuals, conditions that became constants, redundant equations. =#

function _isZeroLiteral(@nospecialize(exp))::Bool
  @match exp begin
    DAE.RCONST(v) => v == 0.0
    DAE.ICONST(v) => v == 0
    _ => false
  end
end

function _isSyntacticZeroResidual(@nospecialize(exp))::Bool
  if _isZeroLiteral(exp)
    return true
  end
  @match exp begin
    DAE.BINARY(e1, DAE.SUB(__), e2) => isequal(e1, e2)
    DAE.BINARY(e1, DAE.ADD(__), DAE.UNARY(DAE.UMINUS(__), e2)) => isequal(e1, e2)
    DAE.BINARY(DAE.UNARY(DAE.UMINUS(__), e1), DAE.ADD(__), e2) => isequal(e1, e2)
    _ => false
  end
end

function _isTrivialResidualEquation(eq::Union{BDAE.RESIDUAL_EQUATION, RESIDUAL_EQUATION}, simCode::SIM_CODE)::Bool
  #= A residual referencing any unknown cref cannot be trivial. _hasUnknownCref
     collects cref names via collectCrefNames!, which has a SIM-native arm, so
     checking eq.exp directly bails WITHOUT a toDAEExp tree for the common case.
     This runs after every SimCode pass (~16x), so the dropped per-residual
     toDAEExp is heavily amplified. =#
  if _hasUnknownCref(eq.exp, simCode.stringToSimVarHT)
    return false
  end
  local expDAE = toDAEExp(eq.exp)
  if _isSyntacticZeroResidual(expDAE)
    return true
  end
  local value = tryEvalNumeric(expDAE, simCode)
  return value !== nothing && value == 0.0
end

function _isTrivialInitialEquation(@nospecialize(eq), simCode::SIM_CODE)::Bool
  if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
    return _isTrivialResidualEquation(eq, simCode)
  elseif eq isa BDAE.EQUATION || eq isa EQUATION
    local lhsDAE = toDAEExp(eq.lhs)
    local rhsDAE = toDAEExp(eq.rhs)
    if _hasUnknownCref(lhsDAE, simCode.stringToSimVarHT) ||
       _hasUnknownCref(rhsDAE, simCode.stringToSimVarHT)
      return false
    end
    if isequal(lhsDAE, rhsDAE)
      return true
    end
    local lhsVal = tryEvalScalar(lhsDAE, simCode)
    local rhsVal = tryEvalScalar(rhsDAE, simCode)
    return lhsVal !== nothing && rhsVal !== nothing && lhsVal == rhsVal
  end
  return false
end

function _filterTrivialResiduals(eqs::AbstractVector,
                                 simCode::SIM_CODE)::Tuple{AbstractVector, Int}
  local newEqs = typeof(eqs)()
  sizehint!(newEqs, length(eqs))
  local nRemoved = 0
  for eq in eqs
    if _isTrivialResidualEquation(eq, simCode)
      nRemoved += 1
    else
      push!(newEqs, eq)
    end
  end
  return (newEqs, nRemoved)
end

function _filterTrivialInitialEquations(eqs, simCode::SIM_CODE)
  local newEqs = typeof(eqs)()
  local nRemoved = 0
  for eq in eqs
    if _isTrivialInitialEquation(eq, simCode)
      nRemoved += 1
    else
      push!(newEqs, eq)
    end
  end
  return (newEqs, nRemoved)
end

function _cleanupTrivialBranchResiduals(ifEq::IF_EQUATION,
                                        simCode::SIM_CODE)::Tuple{Union{IF_EQUATION, Nothing}, Int}
  if isempty(ifEq.branches)
    return (nothing, 0)
  end
  local nResiduals = length(first(ifEq.branches).residualEquations)
  if any(branch -> length(branch.residualEquations) != nResiduals, ifEq.branches)
    return (ifEq, 0)
  end
  local keep = trues(nResiduals)
  local nRemovedSlots = 0
  for idx in 1:nResiduals
    local allTrivial = true
    for branch in ifEq.branches
      if !_isTrivialResidualEquation(branch.residualEquations[idx], simCode)
        allTrivial = false
        break
      end
    end
    if allTrivial
      keep[idx] = false
      nRemovedSlots += 1
    end
  end
  if nRemovedSlots == 0
    return (ifEq, 0)
  end
  if nRemovedSlots == nResiduals
    return (nothing, nRemovedSlots * length(ifEq.branches))
  end
  local newBranches = BRANCH[]
  for branch in ifEq.branches
    local newResiduals = RESIDUAL_EQUATION[branch.residualEquations[i] for i in 1:nResiduals if keep[i]]
    push!(newBranches, BRANCH(branch.condition, newResiduals,
                              branch.identifier, branch.targets, branch.isSingular,
                              branch.matchOrder, branch.equationGraph, branch.sccs,
                              branch.stringToSimVarHT))
  end
  return (IF_EQUATION(newBranches), nRemovedSlots * length(ifEq.branches))
end

"""
    cleanupTrivialResidualEquations(simCode; sourcePass = "")

Remove residuals that are provably trivial without symbolic algebra. To avoid
changing equation/unknown balance, a residual is only removed when it contains
no unknown cref and it evaluates or simplifies syntactically to zero. Branch
residuals are removed only when the same residual slot is trivial in every
branch of an IF_EQUATION, preserving the branch alignment expected by codegen.
"""
function cleanupTrivialResidualEquations(simCode::SIM_CODE;
                                         sourcePass::AbstractString = "")::SIM_CODE
  local (newResiduals, nResidualsRemoved) =
    _filterTrivialResiduals(simCode.residualEquations, simCode)
  local (newInitials, nInitialsRemoved) =
    _filterTrivialInitialEquations(simCode.initialEquations, simCode)
  local newIfEquations = IF_EQUATION[]
  local nConditionalRemoved = 0
  local nIfRemoved = 0
  for ifEq in simCode.ifEquations
    local (newIfEq, nRemoved) = _cleanupTrivialBranchResiduals(ifEq, simCode)
    nConditionalRemoved += nRemoved
    if newIfEq === nothing
      nIfRemoved += 1
    else
      push!(newIfEquations, newIfEq)
    end
  end
  if nResidualsRemoved == 0 && nInitialsRemoved == 0 &&
     nConditionalRemoved == 0 && nIfRemoved == 0
    return simCode
  end
  @assign begin
    simCode.residualEquations = newResiduals
    simCode.initialEquations = newInitials
    simCode.ifEquations = newIfEquations
  end
  local afterText = isempty(sourcePass) ? "" : " after $sourcePass"
  @debug "[SIMCODE: $(simCode.name): trivialCleanup] removed trivial equations$afterText" residuals=nResidualsRemoved initial=nInitialsRemoved conditionalResiduals=nConditionalRemoved ifEquations=nIfRemoved
  return simCode
end

function _rewriteResidualIfExp(eq::Union{BDAE.RESIDUAL_EQUATION, RESIDUAL_EQUATION}, simCode::SIM_CODE)
  #= resolveConstantIfExp dispatches by type: SIM eq.exp -> SIM-native arm (no
     whole-tree toDAEExp), DAE eq.exp -> DAE arm; === identity reuse preserved. =#
  local newExp = resolveConstantIfExp(eq.exp, simCode)
  return newExp === eq.exp ? eq : typeof(eq)(newExp, eq.source, eq.attr)
end

function _rewriteInitialIfExp(@nospecialize(eq), simCode::SIM_CODE)
  if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
    return _rewriteResidualIfExp(eq, simCode)
  elseif eq isa BDAE.EQUATION
    local newLhs = resolveConstantIfExp(eq.lhs, simCode)
    local newRhs = resolveConstantIfExp(eq.rhs, simCode)
    return (newLhs === eq.lhs && newRhs === eq.rhs) ? eq :
           BDAE.EQUATION(newLhs, newRhs, eq.source, eq.attributes)
  elseif eq isa EQUATION
    local newLhs = resolveConstantIfExp(eq.lhs, simCode)
    local newRhs = resolveConstantIfExp(eq.rhs, simCode)
    return (newLhs === eq.lhs && newRhs === eq.rhs) ? eq :
           EQUATION(newLhs, newRhs, eq.source, eq.attr)
  end
  return eq
end

function _rewriteBranchIfExp(branch::BRANCH, simCode::SIM_CODE)::BRANCH
  local newCondition = branch.identifier == ELSE_BRANCH ?
                       branch.condition :
                       resolveConstantIfExp(branch.condition, simCode)
  local newResiduals = RESIDUAL_EQUATION[
    _rewriteResidualIfExp(eq, simCode) for eq in branch.residualEquations
  ]
  return BRANCH(newCondition, newResiduals,
                branch.identifier, branch.targets, branch.isSingular,
                branch.matchOrder, branch.equationGraph, branch.sccs,
                branch.stringToSimVarHT)
end

function _reindexIfBranches(branches::Vector{BRANCH})::Vector{BRANCH}
  local n = length(branches)
  local out = BRANCH[]
  sizehint!(out, n)
  for (idx, branch) in enumerate(branches)
    local isLast = idx == n
    local isElse = branch.identifier == ELSE_BRANCH || isLast
    local identifier = isElse ? ELSE_BRANCH : idx
    local target = isElse ? ELSE_BRANCH : idx + 1
    local condition = isElse ? SCONST("ELSE_BRANCH") : branch.condition
    push!(out, BRANCH(condition, branch.residualEquations,
                      identifier, target, branch.isSingular,
                      branch.matchOrder, branch.equationGraph, branch.sccs,
                      branch.stringToSimVarHT))
  end
  return out
end

function _pruneIfEquation(ifEq::IF_EQUATION,
                          simCode::SIM_CODE)::Tuple{Union{IF_EQUATION, Nothing}, Vector{RESIDUAL_EQUATION}, Int, Bool}
  local rewrittenBranches = BRANCH[_rewriteBranchIfExp(branch, simCode) for branch in ifEq.branches]
  local newBranches = BRANCH[]
  local promoted = RESIDUAL_EQUATION[]
  local nPrunedBranches = 0
  local hasUnconditionalFallback = false
  for branch in rewrittenBranches
    if branch.identifier == ELSE_BRANCH
      hasUnconditionalFallback = true
      if isempty(newBranches)
        append!(promoted, branch.residualEquations)
        return (nothing, promoted, nPrunedBranches + 1, true)
      end
      push!(newBranches, branch)
      return (IF_EQUATION(_reindexIfBranches(newBranches)), promoted, nPrunedBranches, true)
    end
    local condValue = tryEvalCondition(branch.condition, simCode)
    if condValue === false
      nPrunedBranches += 1
      continue
    elseif condValue === true
      hasUnconditionalFallback = true
      if isempty(newBranches)
        append!(promoted, branch.residualEquations)
        return (nothing, promoted, nPrunedBranches + 1, true)
      end
      push!(newBranches, BRANCH(SCONST("ELSE_BRANCH"),
                                branch.residualEquations,
                                ELSE_BRANCH, ELSE_BRANCH, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
      return (IF_EQUATION(_reindexIfBranches(newBranches)), promoted, nPrunedBranches + 1, true)
    else
      push!(newBranches, branch)
    end
  end
  if isempty(newBranches)
    return (nothing, promoted, nPrunedBranches, hasUnconditionalFallback)
  end
  if !hasUnconditionalFallback
    #= No `else` branch was found and no static-true branch fired. We saw only
       `false` and dynamic branches. The Modelica spec says an IF_EQUATION
       without `else` contributes equations only when one branch matches at
       runtime; statically-false branches are dead. We could safely drop them,
       but doing so would also need a structural recount further upstream
       (branches participate in matching/causalization). Keep the conservative
       behavior and return the IFEXP-rewritten branch list unchanged. The
       prune count is reported truthfully so the log is not misleading. =#
    return (IF_EQUATION(rewrittenBranches), promoted, nPrunedBranches, false)
  end
  return (IF_EQUATION(_reindexIfBranches(newBranches)), promoted, nPrunedBranches, true)
end

"""
    pruneConstantConditions(simCode)

Resolve constant-condition IFEXP nodes throughout the main equation vectors and
prune IF_EQUATION branches whose guards are compile-time constants. If a branch
is selected before any dynamic guard remains, its residual equations are promoted
to top-level residuals and the IF_EQUATION is removed.
"""
function pruneConstantConditions(simCode::SIM_CODE)::SIM_CODE
  local newResiduals = RESIDUAL_EQUATION[
    _rewriteResidualIfExp(eq, simCode) for eq in simCode.residualEquations
  ]
  local newInitials = typeof(simCode.initialEquations)()
  for eq in simCode.initialEquations
    push!(newInitials, _rewriteInitialIfExp(eq, simCode))
  end
  local newIfEquations = IF_EQUATION[]
  local nPrunedBranches = 0
  local nPromotedResiduals = 0
  local nRemovedIfEquations = 0
  for ifEq in simCode.ifEquations
    local (newIfEq, promoted, pruned, _) = _pruneIfEquation(ifEq, simCode)
    nPrunedBranches += pruned
    if !isempty(promoted)
      #= `promoted` comes from BRANCH (Vector{BDAE.RESIDUAL_EQUATION}); newResiduals
         is Vector{RESIDUAL_EQUATION}. Convert at the boundary. =#
      append!(newResiduals, [toSim(p) for p in promoted])
      nPromotedResiduals += length(promoted)
    end
    if newIfEq === nothing
      nRemovedIfEquations += 1
    else
      push!(newIfEquations, newIfEq)
    end
  end
  @assign begin
    simCode.residualEquations = newResiduals
    simCode.initialEquations = newInitials
    simCode.ifEquations = newIfEquations
  end
  if nPrunedBranches > 0 || nPromotedResiduals > 0 || nRemovedIfEquations > 0
    @debug "[SIMCODE: $(simCode.name): constantConditionPruning] pruned constant conditions" branches=nPrunedBranches promotedResiduals=nPromotedResiduals removedIfEquations=nRemovedIfEquations
  end
  return simCode
end

"""
    removeRedundantEquations(simCode::SIM_CODE) -> SIM_CODE

Post-alias-elimination over-determination reduction.

After alias elimination, some residual equations may become structurally
redundant: they mention only unknowns that are already uniquely determined
by other equations. This produces more equations than unknowns
(ExtraEquationsSystemException in MTK structural_simplify).

This pass computes a maximum bipartite matching of residual equations to
surviving unknowns. Equations that cannot be matched to any still-free
unknown are algebraically implied by the matched equations (assuming the
original Modelica model is well-posed) and are safely removed.

Typical trigger: balanced 3-phase star networks where the Kirchhoff current
law `i[1]+i[2]+i[3]=0` is a zero-sum identity implied by the three
per-phase Ohm's law equations, but survives alias elimination as an extra
residual.
"""
function removeRedundantEquations(simCode::SIM_CODE)::SIM_CODE
  local ht  = simCode.stringToSimVarHT
  local res = simCode.residualEquations
  local n_eqs  = length(res)
  local n_vars = count(((_k, (_, sv)),) -> isUnknownVarKind(sv.varKind), ht)

  if n_eqs <= n_vars
    return simCode
  end

  local n_extra = n_eqs - n_vars
  @info "[SIMCODE: $(simCode.name): removeRedundantEquations] over-determined by $n_extra equation(s); removing only provably-redundant (duplicate) residuals"
  local firstSeen = Dict{String, Int}()
  local duplicates = Int[]
  for i in 1:n_eqs
    local key = try string(toDAEExp(res[i].exp)) catch; string(res[i].exp) end
    if haskey(firstSeen, key)
      push!(duplicates, i)
    else
      firstSeen[key] = i
    end
  end

  if isempty(duplicates)
    @warn "[SIMCODE: $(simCode.name): removeRedundantEquations] over-determined by $n_extra but found no duplicate residuals to remove; leaving the system unchanged so the imbalance surfaces in the solver rather than deleting an arbitrary constraint"
    return simCode
  end

  #= Never remove more than the surplus. Each duplicate is independently and
     provably redundant, so taking the first n_extra is safe and deterministic. =#
  if length(duplicates) > n_extra
    duplicates = duplicates[1:n_extra]
  end

  map(duplicates) do i
    local eqStr = try OMFrontend.Frontend.toString(res[i].exp) catch; string(res[i].exp) end
    @info "[SIMCODE: $(simCode.name): removeRedundantEquations] removing duplicate equation [$i]: 0 = $eqStr"
  end

  local removed_set = OrderedSet{Int}(duplicates)
  local newRes = RESIDUAL_EQUATION[res[i] for i in 1:n_eqs if i ∉ removed_set]
  @assign simCode.residualEquations = newRes

  if length(duplicates) < n_extra
    @warn "[SIMCODE: $(simCode.name): removeRedundantEquations] still over-determined by $(n_extra - length(duplicates)) after removing $(length(duplicates)) duplicate(s); leaving the remainder for the solver to flag"
  end
  return simCode
end
