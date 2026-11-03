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

#= Invariant checks of the simulation code before the code generation: each rule a pure
   function returning its violations (no I/O, no throwing); `check` runs them all. A rule's
   severity is its own (:warn, :error): a strict caller filters. `where` anchors a message (an
   equation's index, a variable's name). =#

"""
    CheckViolation(rule, severity, where, detail)

A violation a rule found: the rule (`:balanced`, `:alias_consistency`, ...), its severity
(`:warn`, `:error`), where (an equation's index, a name) and what.
"""
struct CheckViolation
  rule::Symbol
  severity::Symbol
  where::String
  detail::String
end

"The violations `check` found (none for a clean simulation code) and its time."
struct CheckResult
  violations::Vector{CheckViolation}
  elapsed_s::Float64
end

"""
    SimCodeCheckError(msg)

Raised to abort the code generation where a rule reports an `:error` it cannot survive (an
unresolved reference): the categorized message, no stack trace of the code generation.
"""
struct SimCodeCheckError <: Exception
  msg::String
end

Base.showerror(io::IO, e::SimCodeCheckError) = print(io, e.msg)

"The names that resolve without a variable: time, the constants, the wildcard of a tuple equation."
const BUILTIN_NAMES = OrderedSet{String}(["time", "pi", "e", "Modelica_Constants_pi", "Modelica_Constants_e", "_"])

"""
    check(simCode) -> CheckResult

Every rule of `CHECK_RULES` run on `simCode`. Never throws: a rule that fails is a violation.
"""
function check(simCode::SIM_CODE)::CheckResult
  local t0 = time()
  local violations = CheckViolation[]
  for rule in CHECK_RULES
    try
      append!(violations, rule(simCode))
    catch e
      push!(violations, CheckViolation(:rule_crash, :warn, string(nameof(rule)), "Rule raised: " * sprint(showerror, e)))
    end
  end
  return CheckResult(violations, time() - t0)
end

"The `:error` violations of a result."
strict(r::CheckResult)::Vector{CheckViolation} = filter(v -> v.severity === :error, r.violations)

"""
    report(io, result; prefix = "", modelName = "")

A check's result printed (one line when clean), labelled `[SIMCODE: <modelName>: check]`.
"""
function report(io::IO, r::CheckResult; prefix::AbstractString = "", modelName::AbstractString = "")::Nothing
  local label = isempty(modelName) ? "check" : string(modelName, ": check")
  if isempty(r.violations)
    println(io, prefix, "[SIMCODE: ", label, "] clean (", round(r.elapsed_s, digits = 4), "s)")
    return nothing
  end
  println(io, prefix, "[SIMCODE: ", label, "] ", length(r.violations), " violation(s) in ", round(r.elapsed_s, digits = 4), "s")
  for v in r.violations
    println(io, prefix, "  [", v.severity, "] ", v.rule, " @ ", v.where, " - ", v.detail)
  end
  return nothing
end

"""
    checkBalanced(simCode)

The unknowns the residuals read (not eliminated; a state for its derivative, read or under
der(); a discrete with its implicit `der(d) = 0`) as many as the residuals: else the system is
under- or overdetermined and MTK's structural_simplify fails or makes a degenerate one. Not for
a model with structural modes, if- or when-equations (they give residuals of their own).
"""
function checkBalanced(simCode::SIM_CODE)::Vector{CheckViolation}
  (hasSubModels(simCode) || !isempty(simCode.ifEquations) || !isempty(simCode.whenEquations)) && return CheckViolation[]
  local read = OrderedSet{String}()
  foreach(eq -> foreachExp(e -> e isa Union{EXP_CREF, DER, PRE} && push!(read, crefName(e.cref)), eq.exp), simCode.residualEquations)
  local eliminated = OrderedSet(simCode.eliminatedVariables)
  local nUnknown = 0
  local nDiscrete = 0
  for (_, (_, v)) in simCode.stringToSimVarHT
    #= a derivative's variable is named as its state =#
    (v.name in eliminated || !isUnknownVarKind(v.varKind) || v.varKind isa STATE_DERIVATIVE || !(v.name in read)) && continue
    nUnknown += 1
    v.varKind isa DISCRETE && (nDiscrete += 1)
  end
  local nEq = length(simCode.residualEquations) + nDiscrete
  nUnknown == nEq && return CheckViolation[]
  return CheckViolation[CheckViolation(:balanced, :error, simCode.name, "unknowns=$(nUnknown) != equations=$(nEq)")]
end

"""
    checkAliasConsistency(simCode)

An alias eliminates another name than its own, one among the eliminated variables, for a
representative that is a variable (not itself eliminated).
"""
function checkAliasConsistency(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  local elim = OrderedSet(simCode.eliminatedVariables)
  for a in simCode.aliasMap
    if a.eliminatedName == a.representativeName
      push!(out, CheckViolation(:alias_consistency, :error, a.eliminatedName, "alias points to itself"))
      continue
    end
    a.eliminatedName in elim ||
      push!(out, CheckViolation(:alias_consistency, :warn, a.eliminatedName, "alias eliminatedName missing from eliminatedVariables"))
    if !haskey(simCode.stringToSimVarHT, a.representativeName)
      push!(out, CheckViolation(:alias_consistency, :error, a.representativeName, "alias representative not in stringToSimVarHT"))
    elseif a.representativeName in elim
      push!(out, CheckViolation(:alias_consistency, :error, a.representativeName, "alias representative is itself eliminated"))
    end
  end
  return out
end

"An eliminated variable is no variable any more (the code generation would bind it twice)."
checkEliminatedVariablesPruned(simCode::SIM_CODE)::Vector{CheckViolation} =
  CheckViolation[CheckViolation(:eliminated_vars_pruned, :warn, name, "eliminated variable still present in stringToSimVarHT")
                 for name in simCode.eliminatedVariables if haskey(simCode.stringToSimVarHT, name)]

"""
    checkDeadStates(simCode)

A state is under der() somewhere (a residual, an eliminated or initial equation, an if-branch,
a when-equation; an array element's by its array): else MTK eliminates it silently or the
system is overdetermined.
"""
checkDeadStates(simCode::SIM_CODE)::Vector{CheckViolation} =
  CheckViolation[CheckViolation(:dead_state, :warn, name, "STATE variable has no der() reference") for name in deadStateNames(simCode)]

"""
    checkReferenceResolution(simCode)

Every reference of a residual resolves: to a variable, an eliminated variable, a builtin name,
an iterator of a reduction of it; the array of a reference by a subscript that is not a literal
to its elements (the code generation reads them as a table). An unresolved one is a parameter
closure's miss, a stale alias, a lowering's bug: the code generation would raise an
UndefVarError.
"""
function checkReferenceResolution(simCode::SIM_CODE)::Vector{CheckViolation}
  local out = CheckViolation[]
  local ht = simCode.stringToSimVarHT
  local known = union(OrderedSet{String}(keys(ht)), OrderedSet{String}(simCode.eliminatedVariables), BUILTIN_NAMES)
  #= every variable's array (`elementArray`), or its own name for one that is no element: a subscripted read's array is one of them =#
  local bases::Set{Symbol} = Set{Symbol}(something(elementArray(simCode, k), sv.ref.name) for (k, (_, sv)) in ht if sv.ref !== nothing)
  for (i, eq) in enumerate(simCode.residualEquations)
    local bound = Set{String}()
    foreachExp(e -> e isa REDUCTION && foreach(it -> push!(bound, String(it.name)), e.iterators), eq.exp)
    local subscripted = Set{EXP_CREF}()
    foreachExp(e -> (e isa ASUB && e.exp isa EXP_CREF) && push!(subscripted, e.exp), eq.exp)
    local missing = OrderedSet{String}()
    foreachExp(eq.exp) do e
      e isa Union{EXP_CREF, DER, PRE} || return
      local nm = crefName(e.cref)
      (nm in known || nm in bound) && return
      (e isa EXP_CREF && e in subscripted && e.cref.ref !== nothing && arrayBase(e.cref.ref) in bases) && return
      push!(missing, nm)
    end
    foreach(nm -> push!(out, CheckViolation(:cref_resolution, :error, "residualEquations[$i]",
                                            "component reference `$nm` does not resolve")), missing)
  end
  return out
end

"The names of the table are the code's: flat, no dots (the lowering's simulation names)."
checkCanonicalNames(simCode::SIM_CODE)::Vector{CheckViolation} =
  CheckViolation[CheckViolation(:canonical_cref_names, :error, name, "stringToSimVarHT key is not canonical")
                 for name in keys(simCode.stringToSimVarHT) if occursin('.', name)]

"""
The references of the simulation code: every variable has its IR reference (`SIMVAR.ref`, in
`ReferenceIndex.references`), none of whose parts carries structure in its name (`IR.structureInName`), and every
reference the equations read (residual, initial, if-equations' branches, when bodies, asserts, initial
algorithms) has one (`SimCref.ref`): the structure of every name, so that no pass takes a name apart.
"""
function checkReferences(simCode::SIM_CODE)::Vector{CheckViolation}
  local out::Vector{CheckViolation} = CheckViolation[]
  for (key, (_, sv)) in simCode.stringToSimVarHT
    (sv.ref === nothing || referenceOf(simCode, key) === nothing) &&
      push!(out, CheckViolation(:references, :error, key, "variable without its IR reference"))
  end
  for (key, ref) in simCode.referenceIndex.references
    IR.structureInName(ref) && push!(out, CheckViolation(:references, :error, key, "IR reference $(ref.name) with structure in a part's name"))
  end
  local missing::Function = (place::Tuple{Symbol, Int}, c::SimCref) -> c.ref === nothing &&
    push!(out, CheckViolation(:references, :error, "$(place[1])[$(place[2])]", "reference `$(crefName(c))` without its IR reference"))
  foreachEquationExp(simCode; statement = (place::Tuple{Symbol, Int}, s::Statement) ->
                       foreach(c::SimCref -> missing(place, c), statementTargets!(SimCref[], s))) do place::Tuple{Symbol, Int}, e::Exp
    foreachExp(x::Exp -> (x isa Union{EXP_CREF, DER, PRE} && missing(place, x.cref); nothing), e)
  end
  return out
end

"The rules `check` runs, in order."
const CHECK_RULES = Function[checkBalanced, checkAliasConsistency, checkEliminatedVariablesPruned, checkDeadStates,
                             checkReferenceResolution, checkCanonicalNames, checkReferences]
