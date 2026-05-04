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
  DAE.ARRAY, DAE.RANGE, DAE.TUPLE, DAE.ASUB, DAE.TSUB, DAE.RSUB,
  DAE.RECORD, DAE.ENUM_LITERAL, DAE.REDUCTION,
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
    DAE.RSUB(e, _, _, _)        => visit(e)
    DAE.ARRAY(_, _, es)         => for e in es; visit(e) end
    DAE.CALL(_, es, _)          => for e in es; visit(e) end
    DAE.REDUCTION(_, body, iters) => begin
      visit(body)
      for it in iters
        @match it begin
          DAE.REDUCTIONITER(_, rangeExp, guardExp, _) => begin
            visit(rangeExp)
            @match guardExp begin
              SOME(g) => visit(g)
              _ => nothing
            end
          end
          _ => nothing
        end
      end
    end
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

Skipped when:
- structural submodels are present (balance is a per-submodel property),
- any if-equations or when-equations exist (those contribute residuals through
  `simCode.ifEquations[i].branches[j].residualEquations` and
  `simCode.whenEquations` rather than `simCode.residualEquations`, so a naive
  `length(residualEquations)` undercounts and yields false positives).
"""
function rule_balanced(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  if !isempty(simCode.subModels) ||
     !isempty(simCode.ifEquations) ||
     !isempty(simCode.whenEquations)
    return out
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
inside at least one `der(...)` call in some residual / initial / if-branch /
when-equation. If not, the state is dead and MTK will either eliminate it
silently or produce an overdetermined system.

A state name matches a der-cref if either the literal printed cref equals
the state name, or the state's base (subscripts stripped) equals the
der-cref's base. The base-name fallback covers array-element states like
`boxBody1_v_0[1]` whose derivative appears as `der(boxBody1_v_0)[1]` and so
shows up as `boxBody1_v_0` (no subscript) in the der-cref set.
"""
function rule_dead_states(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  local derCrefs = _collectDerCrefs(simCode)
  local derBases = Set(_baseName(n) for n in derCrefs)
  for (_, (_, v)) in simCode.stringToSimVarHT
    if v.varKind isa STATE &&
       !(v.name in derCrefs) &&
       !(_baseName(v.name) in derBases)
      push!(out, CheckViolation(:dead_state, :warn, v.name,
                                "STATE variable has no der() reference"))
    end
  end
  return out
end
push!(RULES, rule_dead_states)

#= Strip array subscripts from a printed cref name, e.g. `a[1][2]` -> `a`. =#
_baseName(name::AbstractString) = String(first(split(name, '['; limit = 2)))

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
  #= Eliminated equations: a state's derivative may appear here if alias
     elimination consumed the equation that referenced it, e.g.
     `flange.w = der(flange.phi)` where `flange.w` was aliased out. =#
  for eq in simCode.eliminatedEquations
    if hasproperty(eq, :exp)
      _collectDerFromExp!(names, eq.exp)
    end
  end
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      _collectDerFromExp!(names, branch.condition)
      for eq in branch.residualEquations
        _collectDerFromExp!(names, eq.exp)
      end
    end
  end
  for whenEq in simCode.whenEquations
    _collectDerFromWhenEquation!(names, whenEq)
  end
  return names
end

function _collectDerFromWhenEquation!(names::Set{String}, whenEq)
  if hasproperty(whenEq, :condition)
    _collectDerFromExp!(names, whenEq.condition)
  end
  if hasproperty(whenEq, :whenStmtLst)
    for stmt in whenEq.whenStmtLst
      for fld in (:left, :right, :exp, :stateVar, :value, :message, :condition, :level)
        if hasproperty(stmt, fld)
          _collectDerFromExp!(names, getproperty(stmt, fld))
        end
      end
    end
  end
end

function _collectDerFromExp!(names::Set{String}, exp)
  @match exp begin
    DAE.CALL(Absyn.IDENT("der"), expLst, _) => _pushDerArgNames!(names, listHead(expLst))
    _ => nothing
  end
  _walkExpChildren(child -> _collectDerFromExp!(names, child), exp)
  return names
end

#= Extract the canonical name(s) referenced by a `der(...)` argument and
   record them. Handles a bare CREF as well as `der(arr)` where `arr` is a
   DAE.ARRAY of element CREFs (the MultiBody quaternion / linear-velocity
   pattern documented in CLAUDE.md, e.g. `der(boxBody.v_0)`). =#
function _pushDerArgNames!(names::Set{String}, arg)
  @match arg begin
    DAE.CREF(cref, _) => push!(names, DAE_identifierToString(cref))
    DAE.ARRAY(_, _, es) => for e in es; _pushDerArgNames!(names, e) end
    DAE.CAST(_, e) => _pushDerArgNames!(names, e)
    _ => nothing
  end
  return names
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

#= ── Rule: no literal inside der() / pre() ──────────────────────────── =#

"""
`rule_no_literal_in_der_pre`: walk every residual / initial / if-branch /
when equation expression and flag any `der(<literal>)` or `pre(<literal>)`
where `<literal>` is a constant (DAE.RCONST / ICONST / BCONST). The der/pre
codegen arms now fold these to `0` and the literal value respectively
(see CodeGenerationUtil.jl), so the system can still translate — but the
literal only reaches codegen when an upstream pass (typically
`solveParametricInitialEquations` or `foldParameterClosure`) substituted
a parameter cref with its default value before residual rewriting. That
upstream behaviour is the real bug and this :warn keeps it visible.
"""
function rule_no_literal_in_der_pre(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  for (i, eq) in enumerate(simCode.residualEquations)
    _flagLiteralInDerPre!(out, eq.exp, "residualEquations[$i]")
  end
  for (i, eq) in enumerate(simCode.initialEquations)
    if hasproperty(eq, :exp)
      _flagLiteralInDerPre!(out, eq.exp, "initialEquations[$i]")
    end
  end
  for (i, ifEq) in enumerate(simCode.ifEquations)
    for (j, branch) in enumerate(ifEq.branches)
      _flagLiteralInDerPre!(out, branch.condition, "ifEquations[$i].branches[$j].condition")
      for (k, eq) in enumerate(branch.residualEquations)
        _flagLiteralInDerPre!(out, eq.exp, "ifEquations[$i].branches[$j].residualEquations[$k]")
      end
    end
  end
  for (i, whenEq) in enumerate(simCode.whenEquations)
    if hasproperty(whenEq, :condition)
      _flagLiteralInDerPre!(out, whenEq.condition, "whenEquations[$i].condition")
    end
    if hasproperty(whenEq, :whenStmtLst)
      for (j, stmt) in enumerate(whenEq.whenStmtLst)
        for fld in (:left, :right, :exp, :stateVar, :value, :message, :condition, :level)
          if hasproperty(stmt, fld)
            _flagLiteralInDerPre!(out, getproperty(stmt, fld),
                                  "whenEquations[$i].stmt[$j].$fld")
          end
        end
      end
    end
  end
  return out
end
push!(RULES, rule_no_literal_in_der_pre)

_isDAEConstant(exp) = exp isa DAE.RCONST || exp isa DAE.ICONST || exp isa DAE.BCONST

function _flagLiteralInDerPre!(out::Vector{CheckViolation}, exp, where::String)
  @match exp begin
    DAE.CALL(Absyn.IDENT(name), expLst, _) where (name == "der" || name == "pre") => begin
      local arg = listHead(expLst)
      if _isDAEConstant(arg)
        push!(out, CheckViolation(:no_literal_in_der_pre, :warn, where,
                                  "$(name)(literal $(typeof(arg))) reached codegen — likely upstream parameter-eval substitution"))
      end
    end
    _ => nothing
  end
  _walkExpChildren(child -> _flagLiteralInDerPre!(out, child, where), exp)
  return out
end

end # module SimCodeCheck
