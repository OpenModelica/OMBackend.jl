#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
=#

#=
  Discrete-dummy demotion: figure out which discrete variables should NOT
  carry a `der(d) ~ 0` placeholder in the generated ODE system.

  Background
  ----------
  Modelica `discrete` variables are change-on-event values. MTK has no
  native concept of a discrete unknown, so OMBackend generates a
  "held state" pattern for each discrete variable `d`:

      d  is included in the unknowns vector
      der(d) ~ 0           # dummy continuous dynamics
      <callbacks write to d at event instants>

  The dummy `der(d) ~ 0` equation is what gives the SciML integrator a
  state slot to write into when a `SymbolicContinuousCallback` fires.

  Problem
  -------
  When a residual equation already pins `d` definitionally — e.g.
  `0 ~ d - ifelse(cond, a, b)`, `0 ~ d - integer(x)`, `0 ~ d - (a < b)`,
  or simply `0 ~ d1 - d2` (connector pass-through) — MTK's
  `structural_simplify` uses that equation to eliminate `d`. The dummy
  `der(d) ~ 0` then has no corresponding unknown and the system becomes
  over-determined by exactly one equation per such case.

  Fix
  ---
  Before MTK ever sees the system, detect those definitionally-pinned
  discrete variables and DEMOTE them: drop their dummy and reclassify
  them as algebraic. The (single) residual then defines them, just like
  any other algebraic variable. This pass implements that detection
  + planning. The caller applies the plan via `applyDemotionPlan!`.

  The detection patterns are intentionally narrow. False positives
  (demoting a variable whose dummy IS actually needed) leave the
  callback with no state slot to write into and surface as
  `UndefVarError` at integration time. False negatives leave MTK with
  excess equations and surface as `ExtraEquationsSystemException` at
  `structural_simplify` time.
=#

"""
    DemotionPlan(toDemote)

The set of discrete variable names that should be demoted from
"held state" (`der(d) ~ 0`) to "algebraic unknown". Produced by
`planDemotions`, consumed by `applyDemotionPlan!`.
"""
struct DemotionPlan
  toDemote::OrderedSet{String}
end

"""
    stripLineNodes(e)

Strip `LineNumberNode`s from a Julia `Expr` tree, returning a structurally
equivalent `Expr` (or the input unchanged if it is not an `Expr`). Used
to canonicalize generated equations before hashing — two equations that
differ only by source-location decoration must hash to the same string.
Lives at module scope because the post-Phase-6 MTK equation dedup also
needs it.
"""
function stripLineNodes(@nospecialize(e))
  if e isa Expr
    return Expr(e.head, (stripLineNodes(a) for a in e.args if !(a isa LineNumberNode))...)
  end
  return e
end

#= ---- Module-private Expr-shape match helpers ---- =#

#= Strip `begin ... end` blocks wrapping a single value (source-location
   decoration added by codegen). Equations arrive as
   `0 ~ begin <line> A end - begin <line> B end` rather than `0 ~ A - B`. =#
function _unwrap(@nospecialize(e))
  while e isa Expr && e.head === :block
    local nontrivial = filter(x -> !(x isa LineNumberNode), e.args)
    length(nontrivial) == 1 || break
    e = nontrivial[1]
  end
  return e
end

function _isConstAtom(@nospecialize(e))
  e = _unwrap(e)
  e isa Number && return true
  if e isa Expr && e.head === :call && length(e.args) == 2 &&
     e.args[1] === :- && _unwrap(e.args[2]) isa Number
    return true
  end
  return false
end

function _refName(@nospecialize(e))
  e = _unwrap(e)
  e isa Symbol && return string(e)
  if e isa Expr && e.head === :call && length(e.args) == 2 && e.args[1] isa Symbol
    return string(e.args[1])
  end
  return nothing
end

#= `0 ~ const + ref` or `0 ~ ref - const` or `0 ~ -ref`. Returns the
   pinned cref name, or nothing. =#
function _matchAlias(@nospecialize(rhs))
  rhs = _unwrap(rhs)
  rhs isa Expr || return nothing
  if rhs.head === :call && length(rhs.args) == 3
    local op = rhs.args[1]
    local a = _unwrap(rhs.args[2])
    local b = _unwrap(rhs.args[3])
    if op === :+ || op === :-
      if _isConstAtom(a)
        local nm = _refName(b)
        nm !== nothing && return nm
      end
      if _isConstAtom(b)
        local nm = _refName(a)
        nm !== nothing && return nm
      end
    end
  elseif rhs.head === :call && length(rhs.args) == 2 && rhs.args[1] === :-
    local nm = _refName(_unwrap(rhs.args[2]))
    nm !== nothing && return nm
  end
  return nothing
end

#= True iff `e` contains a call to `constTableLookup` (either bare or
   dotted, e.g. `OMBackend.CodeGeneration.constTableLookup(...)`). =#
function _isConstTableLookupCall(@nospecialize(e))
  e isa Expr || return false
  e.head === :call || return false
  isempty(e.args) && return false
  local fn = e.args[1]
  fn === CONST_TABLE_LOOKUP_HEAD && return true
  if fn isa Expr && fn.head === :. && length(fn.args) >= 2
    local last = fn.args[end]
    last isa QuoteNode && last.value === CONST_TABLE_LOOKUP_HEAD && return true
  end
  return false
end

#= True iff `e` contains an `ifelse` call OR a `constTableLookup` call
   anywhere in its subtree. Both shapes definitionally drive a discrete
   variable when they appear on one side of a `0 ~ disc - <expr>`. =#
function _containsIfelse(@nospecialize(e))
  e isa Expr || return false
  if e.head === :call && !isempty(e.args) && e.args[1] === IFELSE_HEAD
    return true
  end
  _isConstTableLookupCall(e) && return true
  for a in e.args
    _containsIfelse(a) && return true
  end
  return false
end

#= True iff `e` contains a relational comparison anywhere in its subtree.
   Comparisons return a Boolean, so `0 ~ disc - (a < b)` definitionally
   pins `disc`. =#
function _containsComparison(@nospecialize(e))
  e isa Expr || return false
  if e.head === :call && !isempty(e.args) && e.args[1] in COMPARISON_OPS
    return true
  end
  for a in e.args
    _containsComparison(a) && return true
  end
  return false
end

#= True iff `e` contains a Modelica `integer(x)` builtin (lowered as
   `integer`, `modelica_integer`, or `floor`). =#
function _containsIntegerDefCall(@nospecialize(e))
  e isa Expr || return false
  if e.head === :call && !isempty(e.args)
    local fn = e.args[1]
    fn in INTEGER_DEF_HEADS && return true
    if fn isa Expr && fn.head === :. && length(fn.args) >= 2
      local last = fn.args[end]
      last isa QuoteNode && last.value in INTEGER_DEF_HEADS && return true
    end
  end
  for a in e.args
    _containsIntegerDefCall(a) && return true
  end
  return false
end

#= True iff `e` contains a `D(...)` or `der(...)` call. A residual where
   the non-discrete side is a derivative expression (e.g. `0 ~ der(x) - k`)
   is a state equation, not a definitional binding, so must not be
   treated as one by `_matchArithmeticDefinedDiscrete`. =#
function _containsDerivative(@nospecialize(e))
  e isa Expr || return false
  if e.head === :call && !isempty(e.args) && e.args[1] in DERIVATIVE_HEADS
    return true
  end
  for a in e.args
    _containsDerivative(a) && return true
  end
  return false
end

#= `0 ~ disc ± ifelse(...)` — RHS contains an `ifelse` call. =#
function _matchIfelseDefinedDiscrete(@nospecialize(rhs), discreteSet::OrderedSet{String})
  rhs = _unwrap(rhs)
  rhs isa Expr || return nothing
  rhs.head === :call && length(rhs.args) == 3 || return nothing
  local op = rhs.args[1]
  (op === :+ || op === :-) || return nothing
  local a = _unwrap(rhs.args[2])
  local b = _unwrap(rhs.args[3])
  local aName = _refName(a)
  local bName = _refName(b)
  bName !== nothing && bName in discreteSet && _containsIfelse(a) && return bName
  aName !== nothing && aName in discreteSet && _containsIfelse(b) && return aName
  return nothing
end

#= `0 ~ disc ± (a < b)` — RHS contains a Boolean comparison. =#
function _matchComparisonDefinedDiscrete(@nospecialize(rhs), discreteSet::OrderedSet{String})
  rhs = _unwrap(rhs)
  rhs isa Expr || return nothing
  rhs.head === :call && length(rhs.args) == 3 || return nothing
  local op = rhs.args[1]
  (op === :+ || op === :-) || return nothing
  local a = _unwrap(rhs.args[2])
  local b = _unwrap(rhs.args[3])
  local aName = _refName(a)
  local bName = _refName(b)
  bName !== nothing && bName in discreteSet && _containsComparison(a) && return bName
  aName !== nothing && aName in discreteSet && _containsComparison(b) && return aName
  return nothing
end

#= `0 ~ disc ± integer(x)`. =#
function _matchIntegerDefinedDiscrete(@nospecialize(rhs), discreteSet::OrderedSet{String})
  rhs = _unwrap(rhs)
  rhs isa Expr || return nothing
  rhs.head === :call && length(rhs.args) == 3 || return nothing
  local op = rhs.args[1]
  (op === :+ || op === :-) || return nothing
  local a = _unwrap(rhs.args[2])
  local b = _unwrap(rhs.args[3])
  local aName = _refName(a)
  local bName = _refName(b)
  bName !== nothing && bName in discreteSet && _containsIntegerDefCall(a) && return bName
  aName !== nothing && aName in discreteSet && _containsIntegerDefCall(b) && return aName
  return nothing
end

#= Generic arithmetic-defined discrete: `0 ~ disc ± expr` where `expr` is
   a compound expression (`Expr`, not a bare cref) and does NOT itself
   contain a derivative call. Skips when-assigned discretes (the callback
   needs their state slot). =#
function _matchArithmeticDefinedDiscrete(@nospecialize(rhs),
                                         discreteSet::OrderedSet{String},
                                         whenAssignedSet::OrderedSet{String})
  rhs = _unwrap(rhs)
  rhs isa Expr || return nothing
  rhs.head === :call && length(rhs.args) == 3 || return nothing
  local op = rhs.args[1]
  (op === :+ || op === :-) || return nothing
  local a = _unwrap(rhs.args[2])
  local b = _unwrap(rhs.args[3])
  local aName = _refName(a)
  local bName = _refName(b)
  if bName !== nothing && bName in discreteSet && !(bName in whenAssignedSet) && a isa Expr
    _containsDerivative(a) && return nothing
    return bName
  end
  if aName !== nothing && aName in discreteSet && !(aName in whenAssignedSet) && b isa Expr
    _containsDerivative(b) && return nothing
    return aName
  end
  return nothing
end

#= `0 ~ disc ± ifEq_tmpN` where ifEq_tmpN is a lifted if-equation target.
   The target equation `ifEq_tmpN ~ ifelse(...)` already pins the value. =#
function _matchConditionalTargetDefinedDiscrete(@nospecialize(rhs),
                                                discreteSet::OrderedSet{String},
                                                conditionalTargets::OrderedSet{String})
  rhs = _unwrap(rhs)
  rhs isa Expr || return nothing
  rhs.head === :call && length(rhs.args) == 3 || return nothing
  local op = rhs.args[1]
  (op === :+ || op === :-) || return nothing
  local a = _unwrap(rhs.args[2])
  local b = _unwrap(rhs.args[3])
  local aName = _refName(a)
  local bName = _refName(b)
  aName !== nothing && aName in discreteSet &&
    bName !== nothing && bName in conditionalTargets && return aName
  bName !== nothing && bName in discreteSet &&
    aName !== nothing && aName in conditionalTargets && return bName
  return nothing
end

#= Build the set of cref names that any when-equation writes to. Discrete
   vars in this set MUST keep their `der(d) ~ 0` dummy — the callback
   needs the state slot. =#
function _collectWhenAssignedNames(simCode)::OrderedSet{String}
  local names = OrderedSet{String}()
  for whenEq in simCode.whenEquations
    SimulationCode._collectWhenAssignTargets!(names, whenEq.whenEquation)
  end
  return names
end

#= Collect the cref name from a leaf cref expression. =#
function _collectCrefsInExp!(names::OrderedSet{String}, e::SimulationCode.EXP_CREF)
  local r = SimulationCode.extractCrefName(e)
  r !== nothing && push!(names, r[1])
  return nothing
end

#= Recurse into the Exp-typed fields of any other SimCode expression. =#
function _collectCrefsInExp!(names::OrderedSet{String}, @nospecialize(e::SimulationCode.Exp))
  for fn in fieldnames(typeof(e))
    local v = getfield(e, fn)
    if v isa SimulationCode.Exp
      _collectCrefsInExp!(names, v)
    elseif v isa AbstractVector
      for x in v
        x isa SimulationCode.Exp && _collectCrefsInExp!(names, x)
      end
    end
  end
  return nothing
end

#= Discrete vars that participate in a recomputed cyclic residual SCC.
   Treating them as held states adds an equation on top of the loop
   equations — keep them algebraic and let MTK tear the loop. =#
function _collectCyclicSCCDiscretes(simCode, whenAssignedSet::OrderedSet{String})::OrderedSet{String}
  local cyclic = OrderedSet{String}()
  try
    local (sccs, eqToVar) = SimulationCode._recomputeSCCsFromSimCode(simCode)
    for scc in sccs
      length(scc) > 1 || continue
      for eqIdx in scc
        1 <= eqIdx <= length(eqToVar) || continue
        local varName = eqToVar[eqIdx]
        isempty(varName) && continue
        varName in whenAssignedSet && continue
        local entry = get(simCode.stringToSimVarHT, varName, nothing)
        entry === nothing && continue
        if entry[2].varKind isa SimulationCode.DISCRETE
          push!(cyclic, varName)
        end
      end
    end
  catch err
    @debug "[MTK GEN: discrete] cyclic SCC demotion skipped" exception=(err, catch_backtrace())
  end
  return cyclic
end

#= Walk the residual equations once and collect every cref name that is
   definitionally pinned by some match pattern. Returns a tuple of
   (aliasPinned, comparisonPinned). comparisonPinned tracks the
   comparison-derived subset separately so the duplicate-residual
   discount can reclaim them first. =#
function _scanDefinitionalPinning(equations::Vector{Expr},
                                  discreteSet::OrderedSet{String},
                                  whenAssignedSet::OrderedSet{String},
                                  conditionalTargets::OrderedSet{String})
  local aliasPinned = OrderedSet{String}()
  local comparisonPinned = String[]
  for eq in equations
    if eq isa Expr && eq.head === :call && length(eq.args) == 3 &&
       eq.args[1] === :~ && eq.args[2] == 0
      local rhsExp = eq.args[3]
      local pinned = _matchAlias(rhsExp)
      pinned !== nothing && push!(aliasPinned, pinned)

      local ifelseDef = _matchIfelseDefinedDiscrete(rhsExp, discreteSet)
      ifelseDef !== nothing && push!(aliasPinned, ifelseDef)

      local cmpDef = _matchComparisonDefinedDiscrete(rhsExp, discreteSet)
      if cmpDef !== nothing
        push!(aliasPinned, cmpDef)
        push!(comparisonPinned, cmpDef)
      end

      local intDef = _matchIntegerDefinedDiscrete(rhsExp, discreteSet)
      intDef !== nothing && push!(aliasPinned, intDef)

      local arithDef = _matchArithmeticDefinedDiscrete(rhsExp, discreteSet, whenAssignedSet)
      arithDef !== nothing && push!(aliasPinned, arithDef)

      local condDef = _matchConditionalTargetDefinedDiscrete(rhsExp, discreteSet, conditionalTargets)
      condDef !== nothing && push!(aliasPinned, condDef)

      #= Pairwise discrete alias: `0 ~ a - b` with both refs discrete.
         One alias residual + two dummies = three equations for two vars.
         Demote the LATER side (arbitrary; the other gets handled by a
         neighbouring definitional match or by the heuristic). Skip
         StateGraph-style `_suspend`/`_resume` pairs. =#
      local rhsUnwrapped = _unwrap(rhsExp)
      if rhsUnwrapped isa Expr && rhsUnwrapped.head === :call &&
         length(rhsUnwrapped.args) == 3 && rhsUnwrapped.args[1] === :-
        local nameA = _refName(_unwrap(rhsUnwrapped.args[2]))
        local nameB = _refName(_unwrap(rhsUnwrapped.args[3]))
        if nameA !== nothing && nameB !== nothing &&
           nameA in discreteSet && nameB in discreteSet
          if !(endswith(nameA, "_suspend") || endswith(nameA, "_resume") ||
               endswith(nameB, "_suspend") || endswith(nameB, "_resume"))
            push!(aliasPinned, nameB)
          end
        end
      end
    end
  end
  return (aliasPinned, comparisonPinned)
end

#= Discrete names a residual `0 ~ disc ± rhs` actually defines (the discrete is
   a top-level operand of the subtraction/sum, or `0 ~ -disc`). Only such
   discretes may be demoted by the heuristic excess fill: removing the
   `der(d) ~ 0` dummy of a discrete no residual defines strands it with no
   equation, leaving the system under-determined. =#
function _residualDefinedDiscretes(equations::Vector{Expr}, discreteSet::OrderedSet{String})::OrderedSet{String}
  local defined = OrderedSet{String}()
  for eq in equations
    if eq isa Expr && eq.head === :call && length(eq.args) == 3 &&
       eq.args[1] === :~ && eq.args[2] == 0
      local rhs = _unwrap(eq.args[3])
      rhs isa Expr || continue
      if rhs.head === :call && length(rhs.args) == 3 && (rhs.args[1] === :- || rhs.args[1] === :+)
        for operand in (rhs.args[2], rhs.args[3])
          local nm = _refName(_unwrap(operand))
          (nm !== nothing && nm in discreteSet) && push!(defined, nm)
        end
      elseif rhs.head === :call && length(rhs.args) == 2 && rhs.args[1] === :-
        local nm = _refName(_unwrap(rhs.args[2]))
        (nm !== nothing && nm in discreteSet) && push!(defined, nm)
      end
    end
  end
  return defined
end

#= Collect every if-equation target LHS (`ifEq_tmpN ~ ifelse(...)`).
   These names drive the `_matchConditionalTargetDefinedDiscrete`
   pattern. =#
function _collectConditionalTargets(ifEqComponents::Vector{IfEquationComponent})::OrderedSet{String}
  local targets = OrderedSet{String}()
  for component in ifEqComponents
    for ceq in component.conditionalEquations
      if ceq isa Expr && ceq.head === :call && length(ceq.args) == 3 &&
         ceq.args[1] === :~
        local lhsName = _refName(ceq.args[2])
        lhsName !== nothing && push!(targets, lhsName)
      end
    end
  end
  return targets
end

#= Count duplicate residual equations (modulo line-number decoration and
   pre-rewrite form). Duplicates are dropped by MTK at simplify time, so
   the codegen-time over-determination count must discount them. =#
function _countDuplicateResiduals(equations::Vector{Expr}, simCode)
  local seen = OrderedSet{String}()
  local nDup = 0
  local dupKeys = String[]
  local rewritten = rewriteEquations(deepcopy(equations), simCode)
  for eq in rewritten
    local key = string(stripLineNodes(eq))
    if key in seen
      nDup += 1
      push!(dupKeys, key)
    else
      push!(seen, key)
    end
  end
  return (nDup, dupKeys)
end

"""
    planDemotions(simCode, equations, ifEqComponents, discreteVariables,
                  nStateVars, nAlgebraicVars, nOccVars) -> DemotionPlan

Decide which discrete variables should be demoted from "held state" to
"algebraic unknown". Sources used (in priority order):

1. Definitional pinning: a residual of the form `0 ~ disc - <pattern>`
   where `<pattern>` is one of: alias-shape (`a - const`), `ifelse(...)`,
   relational comparison (`a < b`), `integer(x)`, an `ifEq_tmpN` target,
   generic arithmetic, or a pairwise-discrete alias.
2. Cyclic SCC: discrete vars inside a recomputed strongly-connected
   component get demoted so MTK can tear the loop.
3. Heuristic excess fill: if the system is still over-determined after
   the above, demote additional discrete vars sorted by how many
   residuals mention them (most-mentioned first).

When-assigned discretes (callback targets) are never demoted regardless
of pattern match — the callback needs the state slot to write into.
Pre-existing duplicate residuals are discounted from the over-determination
count, and comparison-defined demotions are reclaimed first when they
exist (since the comparison pattern coincides with StateGraph edge
pulses, which should stay held).
"""
function planDemotions(simCode,
                       equations::Vector{Expr},
                       ifEqComponents::Vector{IfEquationComponent},
                       discreteVariables::Vector{String},
                       nStateVars::Int,
                       nAlgebraicVars::Int,
                       nOccVars::Int)::DemotionPlan
  #= Codegen-time over-determination count. The +length(discreteVariables)
     accounts for the (yet-to-be-filtered) dummy `der(d) ~ 0` equations. =#
  local nConditionalEqs = sum(length(c.conditionalEquations) for c in ifEqComponents; init = 0)
  local nTotalEqs = length(equations) + length(discreteVariables) + nConditionalEqs
  local nTotalVars = nStateVars + nAlgebraicVars + length(discreteVariables) + nOccVars
  local excess = nTotalEqs - nTotalVars

  #= Discount duplicate residuals. =#
  local (nDuplicates, dupKeys) = _countDuplicateResiduals(equations, simCode)
  @debug "[MTK GEN: discrete] duplicate residual accounting" residuals=length(equations) duplicates=nDuplicates duplicateKeys=dupKeys
  if nDuplicates > 0
    excess -= nDuplicates
    @debug "[MTK GEN: discrete] discounted $(nDuplicates) duplicate residual equations from discrete dummy excess accounting"
  end

  #= Sets used by the matchers. =#
  local discreteSet = OrderedSet{String}(string(Symbol(dv)) for dv in discreteVariables)
  local whenAssignedSet = _collectWhenAssignedNames(simCode)
  local cyclicSCCDiscrete = _collectCyclicSCCDiscretes(simCode, whenAssignedSet)
  local conditionalTargets = _collectConditionalTargets(ifEqComponents)

  #= Pre-pass: definitionally pinned discretes. =#
  local (aliasPinned, comparisonPinned) =
    _scanDefinitionalPinning(equations, discreteSet, whenAssignedSet, conditionalTargets)

  #= Comparison-defined demotions are the least certain (they also match
     StateGraph edge pulses). Reclaim them up to the duplicate discount. =#
  for _ in 1:nDuplicates
    isempty(comparisonPinned) && break
    delete!(aliasPinned, pop!(comparisonPinned))
  end

  local toDemote = OrderedSet{String}()
  for dv in discreteVariables
    if string(Symbol(dv)) in aliasPinned
      push!(toDemote, dv)
    end
  end

  #= Cyclic SCC discretes. =#
  local nCyclicAdded = count(v -> !(v in toDemote), cyclicSCCDiscrete)
  if nCyclicAdded > 0
    union!(toDemote, cyclicSCCDiscrete)
    @debug "[MTK GEN: discrete] Cyclic SCC fix: demoting $(nCyclicAdded) discrete vars from held states to algebraic unknowns: $(collect(cyclicSCCDiscrete))"
  end

  if !isempty(toDemote)
    @debug "[MTK GEN: discrete] Discrete alias fix (definitional): demoting $(length(toDemote)) discrete vars pinned by definitional residuals (const / ifelse / comparison / integer): $(collect(toDemote))"
    excess -= length(toDemote)
  end

  #= Heuristic excess fill: sort remaining discretes by residual mention
     count, take the most-mentioned ones up to `excess`. =#
  if excess > 0
    local residualDefined = _residualDefinedDiscretes(equations, discreteSet)
    local eqStrings = [string(eq) for eq in equations]
    local mentions = Tuple{String, Int}[]
    for dv in discreteVariables
      dv in toDemote && continue
      dv in whenAssignedSet && continue
      #= Only demote a discrete some residual actually defines; demoting one that
         is merely an input strands it with no equation (under-determination). =#
      local dvSym = string(Symbol(dv))
      dvSym in residualDefined || continue
      local n = count(s -> equationMentionsVariableName(s, dvSym), eqStrings)
      n > 0 && push!(mentions, (dv, n))
    end
    sort!(mentions; by = t -> -t[2])
    local toRemoveHeuristic = OrderedSet{String}()
    for (dv, _) in mentions
      length(toRemoveHeuristic) >= excess && break
      push!(toRemoveHeuristic, dv)
    end
    if !isempty(toRemoveHeuristic)
      @debug "[MTK GEN: discrete] Discrete alias fix: removing $(length(toRemoveHeuristic))/$(excess) excess dummy der equations for $(collect(toRemoveHeuristic))"
      union!(toDemote, toRemoveHeuristic)
    end
  end

  return DemotionPlan(toDemote)
end

"""
    applyDemotionPlan!(plan, discreteVariables, dummyEquations,
                       discreteVariablesSym, algebraicVariablesSym)
        -> (Vector{Expr}, Vector{Symbol})

Walk the parallel `discreteVariables` / `dummyEquations` /
`discreteVariablesSym` vectors once and split each entry by whether the
discrete name is in `plan.toDemote`:

- demoted: name's symbol appended to `algebraicVariablesSym` (mutated),
  dummy equation dropped, symbol dropped from the discrete vector.
- kept: dummy + symbol survive in the returned vectors.

Returns the new `(dummyEquations, discreteVariablesSym)` pair. The
caller assigns them back; `algebraicVariablesSym` is mutated in place.

Done as a single pass so the index correspondence
`dummyEquations[i]` ↔ `discreteVariables[i]` is preserved while
filtering — two separate filter passes drift on each other.
"""
function applyDemotionPlan!(plan::DemotionPlan,
                            discreteVariables::Vector{String},
                            dummyEquations::Vector{Expr},
                            discreteVariablesSym::Vector{Symbol},
                            algebraicVariablesSym::Vector{Symbol})
  isempty(plan.toDemote) && return (dummyEquations, discreteVariablesSym)
  local newDummy = Expr[]
  local newDiscreteSym = Symbol[]
  for (i, dv) in enumerate(discreteVariables)
    if dv in plan.toDemote
      push!(algebraicVariablesSym, Symbol(dv))
    else
      push!(newDummy, dummyEquations[i])
      push!(newDiscreteSym, Symbol(dv))
    end
  end
  return (newDummy, newDiscreteSym)
end
