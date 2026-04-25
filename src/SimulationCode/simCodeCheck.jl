#=
  simCodeCheck.jl

  Lightweight invariant checks that run on a fully-optimized SIM_CODE value
  before MTK code generation. Each rule is a pure function that returns a
  Vector{CheckViolation}; no I/O, no throwing. The top-level `check(simCode)`
  aggregates results from every registered rule.

  Design notes:
  - Rules are independent and side-effect-free so new rules can be added
    in isolation without disturbing existing ones.
  - Severity is per-rule (:warn / :error) so strict mode is just a filter
    on the returned vector.
  - `where` is a short human-readable anchor (equation index, cref path,
    etc.) so messages pinpoint the offending piece of SimCode.
=#

module SimCodeCheck

import Absyn
import DAE
import ..SIM_CODE
import ..STATE
import ..STATE_DERIVATIVE
import ..DAE_identifierToString
import ..isUnknownVarKind

using MetaModelica

export CheckViolation, CheckResult, check

#= ── Public types ───────────────────────────────────────────────────── =#

"""
    CheckViolation(rule, severity, where, detail)

A single invariant violation found by a rule.

- `rule`     : stable rule identifier, e.g. `:balanced`, `:alias_consistency`.
- `severity` : `:warn` or `:error`. `:error` fires in strict mode.
- `where`    : short anchor string (equation index, variable name, ...).
- `detail`   : one-line human-readable description.
"""
struct CheckViolation
  rule::Symbol
  severity::Symbol
  where::String
  detail::String
end

"""
    CheckResult(violations, elapsed_s)

Aggregate return of `check`. `violations` is empty on a clean SimCode.
"""
struct CheckResult
  violations::Vector{CheckViolation}
  elapsed_s::Float64
end

#= ── Top-level driver ──────────────────────────────────────────────── =#

"""
    check(simCode) -> CheckResult

Run every registered rule on `simCode` and return the aggregate result.
Never throws. Callers decide whether to log, warn, or raise on the result.
"""
function check(simCode::SIM_CODE)::CheckResult
  local t0 = time()
  local violations = CheckViolation[]
  for rule in RULES
    try
      append!(violations, rule(simCode))
    catch e
      push!(violations, CheckViolation(:rule_crash, :warn,
                                       string(nameof(rule)),
                                       "Rule raised: " * sprint(showerror, e)))
    end
  end
  return CheckResult(violations, time() - t0)
end

"""
    strict(result) -> Vector{CheckViolation}

Filter a result down to the `:error`-severity violations.
"""
strict(r::CheckResult) = filter(v -> v.severity === :error, r.violations)

"""
    report(io, result; prefix="")

Pretty-print a CheckResult. Quiet when there are no violations.
"""
function report(io::IO, r::CheckResult; prefix::AbstractString = "")
  if isempty(r.violations)
    println(io, prefix, "SimCodeCheck: clean (", round(r.elapsed_s, digits = 4), "s)")
    return
  end
  println(io, prefix, "SimCodeCheck: ", length(r.violations),
          " violation(s) in ", round(r.elapsed_s, digits = 4), "s")
  for v in r.violations
    println(io, prefix, "  [", v.severity, "] ", v.rule, " @ ", v.where,
            " - ", v.detail)
  end
end

#= ── Registry ─────────────────────────────────────────────────────── =#

#= Populated below, once every rule function is declared. =#
const RULES = Function[]

#= Crefs that always resolve and need not appear in stringToSimVarHT. =#
const BUILTIN_CREFS = Set([
  "time", "pi", "e", "Modelica_Constants_pi", "Modelica_Constants_e",
])

#= DAE.Exp concrete subtypes that the MTK + algorithmic codegen handle.
   Stored as a Set so `_flagUnsupportedExp!` does an O(1) typeof check
   instead of 21 sequential `isa` probes per node. =#
const SUPPORTED_EXP_TYPES = Set{DataType}([
  DAE.ICONST, DAE.RCONST, DAE.SCONST, DAE.BCONST,
  DAE.CREF, DAE.BINARY, DAE.UNARY, DAE.LBINARY, DAE.LUNARY,
  DAE.RELATION, DAE.IFEXP, DAE.CAST, DAE.CALL,
  DAE.ARRAY, DAE.RANGE, DAE.TUPLE, DAE.ASUB, DAE.TSUB,
  DAE.RECORD, DAE.ENUM_LITERAL,
])

"""
    _walkExpChildren(visit, exp)

Visit every direct child sub-expression of `exp` once, calling `visit(child)`
on each. Does not recurse — the visitor decides whether to recurse by calling
back into `_walkExpChildren` (or a wrapper that does both). Centralizes the
DAE.Exp tree shape so individual rules don't each maintain a parallel copy.
"""
function _walkExpChildren(visit::Function, exp)
  @match exp begin
    DAE.BINARY(l, _, r)         => begin visit(l); visit(r) end
    DAE.UNARY(_, e)             => visit(e)
    DAE.LBINARY(l, _, r)        => begin visit(l); visit(r) end
    DAE.LUNARY(_, e)            => visit(e)
    DAE.RELATION(l, _, r, _, _) => begin visit(l); visit(r) end
    DAE.IFEXP(c, t, e)          => begin visit(c); visit(t); visit(e) end
    DAE.CAST(_, e)              => visit(e)
    DAE.ASUB(e, subs)           => begin visit(e); for s in subs; visit(s) end end
    DAE.TSUB(e, _, _)           => visit(e)
    DAE.ARRAY(_, _, es)         => for e in es; visit(e) end
    DAE.CALL(_, es, _)          => for e in es; visit(e) end
    _ => nothing
  end
  return nothing
end

#= ── Rule: structural balance (equations vs unknowns) ─────────────── =#

"""
`rule_balanced` compares the count of unknowns (non-parameter, non-input,
non-eliminated SimVars) against the count of residual equations for the
top-level system. A mismatch means the system is under- or overdetermined
and MTK's `structural_simplify` will fail or produce a degenerate system.

Skipped for models with structural submodels, because balance is a
per-submodel property there.
"""
function rule_balanced(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  if !isempty(simCode.subModels)
    return out   #= submodel balance is checked per-submodel elsewhere =#
  end
  local nUnknown = _countUnknowns(simCode)
  local nEq = length(simCode.residualEquations)
  if nUnknown != nEq
    push!(out, CheckViolation(:balanced, :error, simCode.name,
                              "unknowns=$(nUnknown) != equations=$(nEq)"))
  end
  return out
end
push!(RULES, rule_balanced)

function _countUnknowns(simCode::SIM_CODE)::Int
  local eliminated = Set(simCode.eliminatedVariables)
  local n = 0
  for (_, (_, v)) in simCode.stringToSimVarHT
    if v.name in eliminated
      continue
    end
    #= STATE_DERIVATIVE is bookkept but not independent, so it does not count
       toward unknowns even though `isUnknownVarKind` returns true for it. =#
    if isUnknownVarKind(v.varKind) && !(v.varKind isa STATE_DERIVATIVE)
      n += 1
    end
  end
  return n
end

#= ── Rule: alias consistency ──────────────────────────────────────── =#

"""
`rule_alias_consistency`:
- Every alias's `eliminatedName` should also appear in `eliminatedVariables`.
- Every alias's `representativeName` should resolve in `stringToSimVarHT`
  and must NOT itself be in `eliminatedVariables`.
- No alias may have eliminatedName == representativeName.
"""
function rule_alias_consistency(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  local elim = Set(simCode.eliminatedVariables)
  local vars = simCode.stringToSimVarHT
  for a in simCode.aliasMap
    if a.eliminatedName == a.representativeName
      push!(out, CheckViolation(:alias_consistency, :error,
                                a.eliminatedName, "alias points to itself"))
      continue
    end
    if !(a.eliminatedName in elim)
      push!(out, CheckViolation(:alias_consistency, :warn,
                                a.eliminatedName,
                                "alias eliminatedName missing from eliminatedVariables"))
    end
    if !haskey(vars, a.representativeName)
      push!(out, CheckViolation(:alias_consistency, :error,
                                a.representativeName,
                                "alias representative not in stringToSimVarHT"))
    elseif a.representativeName in elim
      push!(out, CheckViolation(:alias_consistency, :error,
                                a.representativeName,
                                "alias representative is itself eliminated"))
    end
  end
  return out
end
push!(RULES, rule_alias_consistency)

#= ── Rule: eliminated variables still in HT ───────────────────────── =#

"""
`rule_eliminated_vars_pruned`: every name in `eliminatedVariables` should
be absent from `stringToSimVarHT`. If both lists mention the same name,
codegen will emit a duplicate binding.
"""
function rule_eliminated_vars_pruned(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  local vars = simCode.stringToSimVarHT
  for name in simCode.eliminatedVariables
    if haskey(vars, name)
      push!(out, CheckViolation(:eliminated_vars_pruned, :warn, name,
                                "eliminated variable still present in stringToSimVarHT"))
    end
  end
  return out
end
push!(RULES, rule_eliminated_vars_pruned)

#= ── Rule: dead states ────────────────────────────────────────────── =#

"""
`rule_dead_states`: every SimVar whose kind is STATE must be referenced
inside at least one `der(...)` call in some residual equation, initial
equation, or when-equation. If not, the state is dead and MTK will
either eliminate it silently or produce an overdetermined system.
"""
function rule_dead_states(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  local derCrefs = _collectDerCrefs(simCode)
  for (_, (_, v)) in simCode.stringToSimVarHT
    if v.varKind isa STATE && !(v.name in derCrefs)
      push!(out, CheckViolation(:dead_state, :warn, v.name,
                                "STATE variable has no der() reference"))
    end
  end
  return out
end
push!(RULES, rule_dead_states)

function _collectDerCrefs(simCode::SIM_CODE)::Set{String}
  local names = Set{String}()
  for eq in simCode.residualEquations
    _collectDerFromExp!(names, eq.exp)
  end
  for eq in simCode.initialEquations
    if hasproperty(eq, :exp)
      _collectDerFromExp!(names, eq.exp)
    end
  end
  return names
end

function _collectDerFromExp!(names::Set{String}, exp)
  @match exp begin
    DAE.CALL(Absyn.IDENT("der"), expLst, _) => begin
      local nm = _derArgName(listHead(expLst))
      if nm !== nothing
        push!(names, nm)
      end
    end
    _ => nothing
  end
  _walkExpChildren(child -> _collectDerFromExp!(names, child), exp)
  return names
end

function _derArgName(arg)::Union{String, Nothing}
  @match arg begin
    DAE.CREF(cref, _) => DAE_identifierToString(cref)   #= canonical flattened name used by stringToSimVarHT =#
    _ => nothing
  end
end

#= ── Rule: cref resolution ────────────────────────────────────────── =#

"""
`rule_cref_resolution`: every DAE.CREF appearing in a residual equation
should resolve against `stringToSimVarHT`, the eliminated-variable set,
or be a well-known builtin (time, pi). Unresolved crefs indicate a
parameter-closure miss, a stale alias, or a codegen mangling bug.

This is deliberately conservative: on the first mismatch we report and
move on, rather than walking the full expression.
"""
function rule_cref_resolution(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  local known = Set(keys(simCode.stringToSimVarHT))
  union!(known, Set(simCode.eliminatedVariables))
  union!(known, BUILTIN_CREFS)
  for (i, eq) in enumerate(simCode.residualEquations)
    local missingNames = String[]
    _collectCrefNames!(missingNames, eq.exp, known)
    for nm in missingNames
      push!(out, CheckViolation(:cref_resolution, :error,
                                "residualEquations[$i]",
                                "cref `$nm` does not resolve"))
    end
  end
  return out
end
push!(RULES, rule_cref_resolution)

function _collectCrefNames!(missing::Vector{String}, exp, known::Set{String})
  @match exp begin
    DAE.CREF(cref, _) => begin
      local nm = DAE_identifierToString(cref)
      if !(nm in known)
        push!(missing, nm)
      end
    end
    _ => nothing
  end
  _walkExpChildren(child -> _collectCrefNames!(missing, child, known), exp)
  return missing
end

#= ── Rule: supported DAE.Exp variants ─────────────────────────────── =#

"""
`rule_supported_exp`: walk every DAE.Exp in every residual and flag
variants not on the known-supported list. Acts as an early-warning
system for new MSL features that need codegen support.

Current list reflects what MTK_CodeGeneration + CodeGenerationUtil
handle cleanly as of 2026-04-24. Add new variants as they become
supported downstream.
"""
function rule_supported_exp(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  for (i, eq) in enumerate(simCode.residualEquations)
    _flagUnsupportedExp!(out, eq.exp, "residualEquations[$i]")
  end
  return out
end
push!(RULES, rule_supported_exp)

function _flagUnsupportedExp!(out::Vector{CheckViolation}, exp, where::String)
  if !(typeof(exp) in SUPPORTED_EXP_TYPES)
    push!(out, CheckViolation(:supported_exp, :warn, where,
                              "unsupported DAE.Exp variant: $(typeof(exp))"))
    return out
  end
  _walkExpChildren(child -> _flagUnsupportedExp!(out, child, where), exp)
  return out
end

end # module SimCodeCheck
