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

#= Trivial residuals, conditions that became constants, redundant equations, chains of
   semiLinear equations. =#

isZeroLiteral(e::Exp)::Bool = (e isa RCONST && e.value == 0.0) || (e isa ICONST && e.value == 0)

"`0`, `a - a`, `a + -a` or `-a + a`."
function isSyntacticZeroResidual(e::Exp)::Bool
  isZeroLiteral(e) && return true
  e isa BINARY || return false
  e.op === OP_SUB && return e.exp1 == e.exp2
  if e.op === OP_ADD
    (e.exp2 isa UNARY && e.exp2.op === OP_UMINUS) && return e.exp1 == e.exp2.exp
    (e.exp1 isa UNARY && e.exp1.op === OP_UMINUS) && return e.exp1.exp == e.exp2
  end
  return false
end

"A residual without unknowns that is syntactically zero or evaluates to zero."
function isTrivialResidualEquation(eq::RESIDUAL_EQUATION, simCode::SIM_CODE)::Bool
  hasUnknownCref(eq.exp, simCode.stringToSimVarHT) && return false
  isSyntacticZeroResidual(eq.exp) && return true
  local v = tryEvalNumeric(eq.exp, simCode)
  return v !== nothing && v == 0.0
end

"An initial residual so trivial, or an equation without unknowns whose sides are equal or evaluate equal."
function isTrivialInitialEquation(eq::Equation, simCode::SIM_CODE)::Bool
  eq isa RESIDUAL_EQUATION && return isTrivialResidualEquation(eq, simCode)
  eq isa EQUATION || return false
  local ht = simCode.stringToSimVarHT
  (hasUnknownCref(eq.lhs, ht) || hasUnknownCref(eq.rhs, ht)) && return false
  eq.lhs == eq.rhs && return true
  local (l, r) = (tryEvalScalar(eq.lhs, simCode), tryEvalScalar(eq.rhs, simCode))
  return l !== nothing && r !== nothing && l == r
end

"""
    cleanupTrivialBranchResiduals(ifEq, simCode)

An if-equation without the residual positions trivial in every branch (the branches' equations
pair by position); `nothing` if all are; with the number of residuals removed.
"""
function cleanupTrivialBranchResiduals(ifEq::IF_EQUATION, simCode::SIM_CODE)::Tuple{Union{IF_EQUATION, Nothing}, Int}
  isempty(ifEq.branches) && return (nothing, 0)
  local n = length(first(ifEq.branches).residualEquations)
  any(b -> length(b.residualEquations) != n, ifEq.branches) && return (ifEq, 0)
  local keep = BitVector(!all(b -> isTrivialResidualEquation(b.residualEquations[i], simCode), ifEq.branches) for i in 1:n)
  local removed = count(!, keep)
  removed == 0 && return (ifEq, 0)
  removed == n && return (nothing, removed * length(ifEq.branches))
  local branches = BRANCH[withResiduals(b, RESIDUAL_EQUATION[b.residualEquations[i] for i in 1:n if keep[i]]) for b in ifEq.branches]
  return (IF_EQUATION(branches), removed * length(ifEq.branches))
end

"A branch with other residuals."
withResiduals(b::BRANCH, residuals::Vector{RESIDUAL_EQUATION})::BRANCH =
  BRANCH(b.condition, residuals, b.identifier, b.targets)

"""
    cleanupTrivialResidualEquations(simCode; sourcePass)

The simulation code without the residuals that are provably trivial without symbolic algebra:
without an unknown, and zero syntactically or by evaluation (the balance of equations and
unknowns is kept); of the initial equations likewise; of an if-equation's branches only a
position trivial in every branch (their alignment is kept). Run after each pass.
"""
function cleanupTrivialResidualEquations(simCode::SIM_CODE; sourcePass::AbstractString = "")::SIM_CODE
  local residuals = RESIDUAL_EQUATION[eq for eq in simCode.residualEquations if !isTrivialResidualEquation(eq, simCode)]
  local initials = Equation[eq for eq in simCode.initialEquations if !isTrivialInitialEquation(eq, simCode)]
  local ifs = IF_EQUATION[]
  local changedIfs = false
  for ifEq in simCode.ifEquations
    local (new, removed) = cleanupTrivialBranchResiduals(ifEq, simCode)
    changedIfs |= removed > 0 || new === nothing
    new === nothing || push!(ifs, new)
  end
  (length(residuals) == length(simCode.residualEquations) && length(initials) == length(simCode.initialEquations) &&
   !changedIfs) && return simCode
  @debug "[SIMCODE: $(simCode.name): trivialCleanup] removed trivial equations$(isempty(sourcePass) ? "" : " after $sourcePass")"
  return update(simCode; residualEquations = residuals, initialEquations = initials, ifEquations = ifs)
end

#= ==================================================================================== =#
#= Constant conditions                                                                  =#
#= ==================================================================================== =#

resolvedResidual(eq::RESIDUAL_EQUATION, simCode::SIM_CODE)::RESIDUAL_EQUATION =
  (local e = resolveConstantIfExp(eq.exp, simCode); e === eq.exp ? eq : RESIDUAL_EQUATION(e, eq.source, eq.attr))

function resolvedInitialEquation(eq::Equation, simCode::SIM_CODE)::Equation
  eq isa RESIDUAL_EQUATION && return resolvedResidual(eq, simCode)
  if eq isa EQUATION
    local (l, r) = (resolveConstantIfExp(eq.lhs, simCode), resolveConstantIfExp(eq.rhs, simCode))
    return l === eq.lhs && r === eq.rhs ? eq : EQUATION(l, r, eq.source, eq.attr)
  end
  return eq
end

"""
    pushInitialEquation!(out, eq, simCode)

An initial equation with its constant if-expressions resolved; an initial if-equation whose
conditions evaluate as the equations of its branch (Buildings' Movers: `if curve == 1 then
preDer1 = ...` on a parameter `curve`), one whose do not as it is.
"""
function pushInitialEquation!(out::Vector{Equation}, eq::Equation, simCode::SIM_CODE)::Vector{Equation}
  eq isa INLINE_IF_EQUATION || return push!(out, resolvedInitialEquation(eq, simCode))
  local branch = nothing
  for (c, eqs) in zip(eq.conditions, eq.branchesTrue)
    local v = tryEvalCondition(c, simCode)
    v === nothing && return push!(out, eq)
    v && (branch = eqs; break)
  end
  foreach(e -> pushInitialEquation!(out, e, simCode), something(branch, eq.branchElse))
  return out
end

function resolvedBranch(b::BRANCH, simCode::SIM_CODE)::BRANCH
  local c = b.identifier == ELSE_BRANCH ? b.condition : resolveConstantIfExp(b.condition, simCode)
  return BRANCH(c, RESIDUAL_EQUATION[resolvedResidual(r, simCode) for r in b.residualEquations], b.identifier, b.targets)
end

"The branches numbered again, the last (or an else-branch) the else-branch."
function reindexIfBranches(branches::Vector{BRANCH})::Vector{BRANCH}
  local out = BRANCH[]
  for (k, b) in enumerate(branches)
    local isElse = b.identifier == ELSE_BRANCH || k == length(branches)
    push!(out, BRANCH(isElse ? SCONST("ELSE_BRANCH") : b.condition, b.residualEquations, isElse ? ELSE_BRANCH : k,
                      isElse ? ELSE_BRANCH : k + 1))
  end
  return out
end

"""
    pruneIfEquation(ifEq, simCode)

An if-equation without the branches whose conditions evaluate to false, up to the first that
evaluates to true (the else-branch from there); its residuals promoted to the top if that is
the first branch left. Without an else-branch or a true branch the branches stay (they take
part in the matching).
"""
function pruneIfEquation(ifEq::IF_EQUATION, simCode::SIM_CODE)::Tuple{Union{Nothing, IF_EQUATION}, Vector{RESIDUAL_EQUATION}}
  local rewritten = BRANCH[resolvedBranch(b, simCode) for b in ifEq.branches]
  local kept = BRANCH[]
  for b in rewritten
    if b.identifier == ELSE_BRANCH
      isempty(kept) && return (nothing, b.residualEquations)
      push!(kept, b)
      return (IF_EQUATION(reindexIfBranches(kept)), RESIDUAL_EQUATION[])
    end
    local c = tryEvalCondition(b.condition, simCode)
    if c === false
      continue
    elseif c === true
      isempty(kept) && return (nothing, b.residualEquations)
      push!(kept, BRANCH(SCONST("ELSE_BRANCH"), b.residualEquations, ELSE_BRANCH, ELSE_BRANCH))
      return (IF_EQUATION(reindexIfBranches(kept)), RESIDUAL_EQUATION[])
    end
    push!(kept, b)
  end
  isempty(kept) && return (nothing, RESIDUAL_EQUATION[])
  return (IF_EQUATION(rewritten), RESIDUAL_EQUATION[])
end

"""
    pruneConstantConditions(simCode)

The if-expressions whose conditions evaluate at compile time resolved in the residuals and
the initial equations, and the if-equations' branches so decided pruned (`pruneIfEquation`):
a branch chosen before any condition that does not evaluate is the residuals at the top.
"""
function pruneConstantConditions(simCode::SIM_CODE)::SIM_CODE
  local residuals = RESIDUAL_EQUATION[resolvedResidual(eq, simCode) for eq in simCode.residualEquations]
  local initials = Equation[]
  foreach(eq -> pushInitialEquation!(initials, eq, simCode), simCode.initialEquations)
  local ifs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local (new, promoted) = pruneIfEquation(ifEq, simCode)
    append!(residuals, promoted)
    new === nothing || push!(ifs, new)
  end
  return update(simCode; residualEquations = residuals, initialEquations = initials, ifEquations = ifs)
end

#= ==================================================================================== =#
#= Redundant equations                                                                  =#
#= ==================================================================================== =#

"""
    residualUnknownCount(ht)

The number of unknowns: a state counted once (the table has `x` and `der(x)`: the residuals
determine one of them), an algebraic and a discrete variable (a discrete a when-equation
defines too: an upper bound of the residuals' unknowns), an array variable's elements.
"""
function residualUnknownCount(ht::HT)::Int
  local n = 0
  for (_, (_, v)) in ht
    local k = v.varKind
    if k isa Union{STATE, ALG_VARIABLE, DISCRETE}
      n += 1
    elseif k isa ARRAY
      n += prod(k.dimensions; init = 1)
    end
  end
  return n
end

"""
    removeRedundantEquations(simCode)

After the alias elimination, a system with more residuals than unknowns without its
duplicate residuals (each provably redundant), at most as many as are surplus (MSL's
balanced three-phase star networks: `i[1] + i[2] + i[3] = 0` implied by the phases' equations).
Without duplicates the system stays (the imbalance surfaces in the solver).
"""
function removeRedundantEquations(simCode::SIM_CODE)::SIM_CODE
  local res = simCode.residualEquations
  local nVars = residualUnknownCount(simCode.stringToSimVarHT)
  length(res) <= nVars && return simCode
  local extra = length(res) - nVars
  @info "[SIMCODE: $(simCode.name): removeRedundantEquations] over-determined by $extra equation(s); removing only provably-redundant (duplicate) residuals"
  local seen = Set{Exp}()
  local duplicates = Int[]
  for (i, r) in enumerate(res)
    r.exp in seen ? push!(duplicates, i) : push!(seen, r.exp)
  end
  if isempty(duplicates)
    @warn "[SIMCODE: $(simCode.name): removeRedundantEquations] over-determined by $extra but found no duplicate residuals to remove; leaving the system unchanged so the imbalance surfaces in the solver rather than deleting an arbitrary constraint"
    return simCode
  end
  duplicates = duplicates[1:min(end, extra)]
  local removed = Set{Int}(duplicates)
  length(duplicates) < extra &&
    @warn "[SIMCODE: $(simCode.name): removeRedundantEquations] still over-determined by $(extra - length(duplicates)) after removing $(length(duplicates)) duplicate(s); leaving the remainder for the solver to flag"
  return update(simCode; residualEquations = RESIDUAL_EQUATION[r for (i, r) in enumerate(res) if !(i in removed)])
end

#= ==================================================================================== =#
#= semiLinear                                                                           =#
#= ==================================================================================== =#

"""
    simplifySemiLinear(simCode)

MLS 3.7.2.5, rule 1: the equations `y = semiLinear(x, sa, s1)`, `y = semiLinear(x, s1, s2)`,
..., `y = semiLinear(x, sN, sb)` of one `y` and `x` (after the alias elimination) as
`s1 = if x >= 0 then sa else sb`, `s2 = s1`, ..., and `y = semiLinear(x, sa, sb)`: at `x = 0`
the equations hold for any `s1`; the rule picks the meaningful solution (OpenModelica's
simplifysemiLinear). MSL Media's test models connect a ShortPipe to a FixedAmbient so.
"""
function simplifySemiLinear(simCode::SIM_CODE)::SIM_CODE
  local res = simCode.residualEquations
  local groups = OrderedDict{Tuple{Exp, Exp}, Vector{Tuple{Int, Exp, Exp, Exp, Exp}}}()
  for (i, r) in enumerate(res)
    local m = semiLinearEquation(r.exp)
    m === nothing && continue
    push!(get!(groups, (m[1], m[2]), Tuple{Int, Exp, Exp, Exp, Exp}[]), (i, m...))
  end
  local newRes = copy(res)
  local nChains = 0
  for (_, eqs) in groups
    length(eqs) >= 2 || continue
    local chain = semiLinearChain(eqs)
    chain === nothing && continue
    local (_, y, x, _, _) = first(eqs)
    local nodes = Exp[chain[1][4]; [e[5] for e in chain]]
    local (sa, sb) = (first(nodes), last(nodes))
    local cond = RELATION(x, OP_GREATEREQ, RCONST(0.0), 0)
    local set = (k, e) -> (newRes[chain[k][1]] = RESIDUAL_EQUATION(e, res[chain[k][1]].source, res[chain[k][1]].attr))
    local n = length(chain)
    set(1, BINARY(nodes[2], OP_SUB, IFEXP(cond, sa, sb)))
    for k in 2:n-1
      set(k, BINARY(nodes[k + 1], OP_SUB, nodes[k]))
    end
    set(n, BINARY(y, OP_SUB, CALL(:semiLinear, Exp[x, sa, sb], TYPE_REAL(), CALL_BUILTIN)))
    nChains += 1
  end
  nChains == 0 && return simCode
  @info "[SIMCODE: $(simCode.name): simplifySemiLinear] $(nChains) chain(s) of semiLinear equations rewritten (MLS 3.7.2.5)"
  return update(simCode; residualEquations = newRes)
end

"""
    semiLinearEquation(e)

`(y, x, sa, sb)` of the residual `y - semiLinear(x, sa, sb)` (or `semiLinear(...) - y`), `y` a
variable; a negated `y` turned round (`-semiLinear(x, sa, sb) = semiLinear(-x, sb, sa)`: the
alias elimination writes a port's flow as the other port's negated). `nothing` for another.
"""
function semiLinearEquation(e::Exp)::Union{Nothing, NTuple{4, Exp}}
  (e isa BINARY && e.op === OP_SUB) || return nothing
  for (y, c) in ((e.exp1, e.exp2), (e.exp2, e.exp1))
    (isBuiltin(c, :semiLinear) && length(c.args) == 3) || continue
    y isa EXP_CREF && return (y, c.args[1], c.args[2], c.args[3])
    (y isa UNARY && y.op === OP_UMINUS && y.exp isa EXP_CREF) && return (y.exp, negated(c.args[1]), c.args[3], c.args[2])
  end
  return nothing
end

negated(e::Exp)::Exp = e isa UNARY && e.op === OP_UMINUS ? e.exp : UNARY(OP_UMINUS, e)

"""
    semiLinearChain(eqs)

The equations of one `y` and `x` in chain order, `sa -> s1 -> ... -> sb` (each one's second
slope the next one's first), every intermediate slope a variable, each equation once;
`nothing` for another shape.
"""
function semiLinearChain(eqs::Vector{Tuple{Int, Exp, Exp, Exp, Exp}})::Union{Nothing, Vector{Tuple{Int, Exp, Exp, Exp, Exp}}}
  local bySa = Dict{Exp, Int}()
  for (k, e) in enumerate(eqs)
    haskey(bySa, e[4]) && return nothing
    bySa[e[4]] = k
  end
  local sbs = Set{Exp}(e[5] for e in eqs)
  local starts = [k for (k, e) in enumerate(eqs) if !(e[4] in sbs)]
  length(starts) == 1 || return nothing
  local chain = Tuple{Int, Exp, Exp, Exp, Exp}[eqs[only(starts)]]
  while length(chain) < length(eqs)
    local next = get(bySa, last(chain)[5], 0)
    (next == 0 || !(last(chain)[5] isa EXP_CREF)) && return nothing
    push!(chain, eqs[next])
  end
  haskey(bySa, last(chain)[5]) && return nothing
  return chain
end
