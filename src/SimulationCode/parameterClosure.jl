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

#= Algebraic variables that parameters alone define, folded into parameters. =#

"""
    foldParameterClosure(simCode::SIM_CODE)::SIM_CODE

BLT-driven parameter-closure fold.

Walks the scalar blocks of the block-lower-triangular decomposition of the
equation graph in topological order, and for each block whose matched
unknown `v` is defined by a residual of the form `v - f(...) = 0` (or
`f(...) - v = 0`) where `f` depends only on parameters, constants and
previously folded unknowns, promotes `v` to `PARAMETER(SOME(f))` and drops
the residual.

Motivating case: `Modelica.Blocks.Sources.KinematicPTP` introduces seven
algebraic unknowns (`aux1`, `sd_max`, `sdd_max`, `Ta1`, `Ta2`, `Tv`, `Te`,
`noWphase`) whose defining equations are closures over parameters. With no
fold they reach MTK as unknowns with a zero start guess, and Newton's first
evaluation produces `sqrt(1/0) = Inf` and `1/abs(0) = Inf`, aborting init.
Folding turns the chain into parameter bindings that MTK resolves at
elaboration time, so Newton never sees them.

Algebraic loops (`BLTBlock.isLoop == true`) and any non-ALG unknown
(states, derivatives, discretes, arrays) are skipped — those belong
to MTK's structural_simplify.

No-op for VSS / submodel simcodes, for empty residual sets, and when the
earlier matching flagged the system as singular (index reduction has
priority over folding there).
"""
function foldParameterClosure(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode)
    return simCode
  end
  if isempty(simCode.residualEquations)
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  #= Narrow exclusion set: names that appear inside an if-equation's
     `branch.condition` expression. `createIfEquation` ->
     `evalInitialCondition` eval's the condition at MODULE scope; a
     folded PARAMETER binding lives only in the model function's local
     scope, so evaluating a condition that references a folded name
     raises `UndefVarError`. Earlier versions excluded the full
     `simCode.irreducibleVariables` set, but that was too broad:
     `getIrreducibleVars` flattens every cref reachable through any
     IF_EQUATION branch body (conditions AND branch residuals), which
     accidentally blocked KinematicPTP and similar closures from folding
     even when the variable only appeared in a branch residual. Restrict
     to condition-only references here. `_THETA` markers
     (overconstrained-connector Zimmer constant) are preserved
     separately. =#
  local excludedFromFold = OrderedSet{String}()
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      #= Else branches (identifier == -1) carry a trivial/constant
         condition and are never evaluated by `evalInitialCondition`. =#
      if branch.identifier == -1
        continue
      end
      for cref in Util.getAllCrefs(branch.condition)
        push!(excludedFromFold, string(cref))
      end
    end
  end
  for name in keys(ht)
    if endswith(name, "THETA")
      push!(excludedFromFold, name)
    end
  end
  #= Snapshot of ALG-unknown names: only these are candidates for folding. =#
  local algNames = OrderedSet{String}()
  for (name, (_, sv)) in ht
    if name in excludedFromFold
      continue
    end
    @match sv.varKind begin
      ALG_VARIABLE(_) => push!(algNames, name)
      _ => nothing
    end
  end
  if isempty(algNames)
    return simCode
  end

  local nResEqs = length(simCode.residualEquations)

  #= Build per-variable counts of how many residuals place it on the LHS of a
     BINARY SUB. A variable with count == 1 has a unique defining equation
     and is a safe fold target: promoting it to PARAMETER cannot conflict
     with another residual that also "solves for" it. Variables with
     count != 1 are left to MTK (ambiguous or residual-appears-only-as-use). =#
  local defEqOfVar = Dict{String, Int}()    #= varName -> eqIdx of its defining residual =#
  local defCountOfVar = Dict{String, Int}() #= varName -> #residuals with v on LHS of BINARY SUB =#
  for (i, eq) in enumerate(simCode.residualEquations)
    local candidateName = extractBinarySubLhsCrefName(eq.exp, algNames)
    if candidateName !== nothing
      defCountOfVar[candidateName] = get(defCountOfVar, candidateName, 0) + 1
      if !haskey(defEqOfVar, candidateName)
        defEqOfVar[candidateName] = i
      end
    end
  end

  local foldMap = Dict{String, DAE.Exp}()
  local foldedNames = OrderedSet{String}()
  local elimIdxSet = OrderedSet{Int}()

  #= Iterate to fixed point: a freshly folded variable may unlock
     downstream closures (e.g. sd_max = 1/abs(aux1[1]) becomes foldable
     once aux1[1] is a parameter). =#
  local progressed = true
  while progressed
    progressed = false
    for (varName, eqIdx) in defEqOfVar
      if varName in foldedNames
        continue
      end
      if defCountOfVar[varName] != 1
        continue
      end
      local rhs = detectSolvableParameterClosure(
        toDAEExp(simCode.residualEquations[eqIdx].exp), varName, ht, foldedNames)
      if rhs !== nothing
        foldMap[varName] = rhs
        push!(foldedNames, varName)
        push!(elimIdxSet, eqIdx)
        progressed = true
      end
    end
  end

  if isempty(foldMap)
    return simCode
  end

  #= Guard against complete elimination: static models (e.g. MatrixMultTest,
     where every output is bound to a pure-constant expression) would have
     every residual drained by the fold, leaving MTK with 0 equations and
     0 unknowns. MTK's `System(...)` constructor cannot accept an empty
     equation list and raises `MethodError` downstream. In that degenerate
     case, keep at least the original residuals so the normal alias/constant
     elimination pipeline handles the trivial simplification. =#
  if length(simCode.residualEquations) - length(elimIdxSet) == 0
    @debug "[SIMCODE: $(simCode.name): foldParameterClosure] fold would eliminate all residuals; skipping to preserve MTK build" wouldFold=length(foldMap)
    return simCode
  end

  local newHT = copy(ht)
  for (name, bindExp) in foldMap
    local (idx, oldSV) = newHT[name]
    newHT[name] = (idx, SIMVAR(oldSV.name, oldSV.index,
                               PARAMETER(SOME(toSimExp(bindExp))), oldSV.attributes))
  end
  local newResEqs = RESIDUAL_EQUATION[]
  sizehint!(newResEqs, nResEqs - length(elimIdxSet))
  for (i, eq) in enumerate(simCode.residualEquations)
    if !(i in elimIdxSet)
      push!(newResEqs, eq)
    end
  end

  @assign begin
    simCode.residualEquations = newResEqs
    simCode.stringToSimVarHT = newHT
  end
  #= Invalidate derived name sets that were computed against the pre-fold
     HT classification. `irreducibleVariables` was collected by
     `getIrreducibleVars` BEFORE this pass ran, so it may still name
     variables that are now parameters. Leaving them in causes the MTK
     codegen `_batchBlock` to try `setmetadata(kinematicPTP_Ta1, Irreducible)`
     against a module-scope name that no longer exists (the parameter is
     only bound in the model function's local scope).
     Same logic applies to `sharedVariables` for completeness, though in
     practice that is populated only in multi-submodel scenarios. =#
  if !isempty(simCode.irreducibleVariables)
    @assign simCode.irreducibleVariables =
      filter(v -> !(v in foldedNames), simCode.irreducibleVariables)
  end
  if !isempty(simCode.sharedVariables)
    @assign simCode.sharedVariables =
      filter(v -> !(v in foldedNames), simCode.sharedVariables)
  end
  #= Do not push folded entries to `eliminatedEquations` / `eliminatedVariables`.
     Those parallel arrays drive alias observed-equation reconstruction (a
     separate mechanism). Folded variables become full PARAMETERs with a bound
     expression, so MTK evaluates them directly at elaboration time.
     Their observed values appear automatically in the ODESystem's parameter
     substitution path — no observed equation needed. =#
  return simCode
end

#= Cheap SIM cref-like name: only EXP_CREF / ASUB(EXP_CREF) can yield a name, so
   gate on those and convert just that small operand -- never a complex side. =#
_crefLikeNameSIM(e::Exp) =
  (e isa EXP_CREF || (e isa ASUB && e.exp isa EXP_CREF)) ? extractCrefLikeName(toDAEExp(e)) : nothing

#= SIM-native arm: inspect the top-level BINARY/SUB on the SimCode spine and
   convert only the (small) cref-like operands, so non-matching residuals bail
   with no whole-tree toDAEExp. Equivalent: a complex operand yields nothing in
   both paths; EXP_CREF/ASUB(EXP_CREF) convert to the identical DAE name. =#
"""
    extractBinarySubLhsCrefName(exp, candidateNames)

If `exp` has the shape `DAE.BINARY(lhs, SUB, _)` (or `DAE.BINARY(_, SUB, lhs)`)
where `lhs` is a CREF or scalarized ASUB of a name in `candidateNames`,
return that name; otherwise return `nothing`.

Used by the fold to pre-index "defining equations": residuals that name a
variable in their top-level subtraction. If a name appears in more than one
such residual, MTK owns the disambiguation.
"""
function extractBinarySubLhsCrefName(exp::Exp, candidateNames::OrderedSet{String})
  exp isa BINARY || return nothing
  exp.op === OP_SUB || return nothing
  local n1 = _crefLikeNameSIM(exp.exp1)
  if n1 !== nothing && n1 in candidateNames
    return n1
  end
  local n2 = _crefLikeNameSIM(exp.exp2)
  if n2 !== nothing && n2 in candidateNames
    return n2
  end
  return nothing
end

function extractBinarySubLhsCrefName(@nospecialize(exp), candidateNames::OrderedSet{String})
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      if !isSub
        return nothing
      end
      local n1 = extractCrefLikeName(e1)
      if n1 !== nothing && n1 in candidateNames
        return n1
      end
      local n2 = extractCrefLikeName(e2)
      if n2 !== nothing && n2 in candidateNames
        return n2
      end
      return nothing
    end
    _ => return nothing
  end
end

"""
    extractCrefLikeName(exp)

Reconstruct the string name for `DAE.CREF(cr, _)` or
`DAE.ASUB(DAE.CREF(cr, _), subs)` with constant `ICONST` subs. Returns
`nothing` otherwise.
"""
function extractCrefLikeName(@nospecialize(exp))
  @match exp begin
    DAE.CREF(cr, _) => DAE_identifierToString(cr)
    DAE.ASUB(exp = innerExp, sub = subs) => begin
      @match innerExp begin
        DAE.CREF(cr, _) => begin
          local full = buildAsubName(DAE_identifierToString(cr), subs)
          return isempty(full) ? nothing : full
        end
        _ => nothing
      end
    end
    _ => nothing
  end
end

"""
    detectSolvableParameterClosure(exp, matchedName, ht, foldedNames)

Return the RHS `f` if `exp` has the shape `v - f = 0` or `f - v = 0`
where `v` is a bare CREF to `matchedName` and `f` contains only parameters,
constants and names already in `foldedNames`. Otherwise return `nothing`.

Intentionally narrow: more exotic shapes (ADD with sign flip, MUL by
parameter divisor, CALL-wrapped LHS) are left to MTK.
"""
function detectSolvableParameterClosure(@nospecialize(exp), matchedName::String,
                                        ht, foldedNames::OrderedSet{String})
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      if !isSub
        return nothing
      end
      if isCrefNamed(e1, matchedName) && isParameterClosureExp(e2, ht, foldedNames, matchedName)
        return e2
      end
      if isCrefNamed(e2, matchedName) && isParameterClosureExp(e1, ht, foldedNames, matchedName)
        return e1
      end
      return nothing
    end
    _ => return nothing
  end
end

"""
    isCrefNamed(exp, name)::Bool

True iff `exp` is a variable reference whose resolved name equals `name`.

Two shapes are accepted:
  * `DAE.CREF(cr, _)`  -- the scalar case.
  * `DAE.ASUB(DAE.CREF(cr, _), subs)` where every sub is a constant `ICONST`
    -- the scalarized array-element case. The reconstructed name includes
    the literal subscript suffix, e.g. `kinematicPTP_aux1[1]`, matching the
    key format used by `stringToSimVarHT`.

Non-constant subscripts are rejected (we cannot resolve the name statically).
"""
function isCrefNamed(@nospecialize(exp), name::String)::Bool
  @match exp begin
    DAE.CREF(cr, _) => DAE_identifierToString(cr) == name
    DAE.ASUB(exp = innerExp, sub = subs) => begin
      @match innerExp begin
        DAE.CREF(cr, _) => begin
          local full = buildAsubName(DAE_identifierToString(cr), subs)
          return !isempty(full) && full == name
        end
        _ => false
      end
    end
    _ => false
  end
end

"""
    isParameterClosureExp(exp, ht, foldedNames, excludeName)::Bool

True iff every CREF name reachable in `exp`:
  * is not equal to `excludeName` (no self-reference), AND
  * either resolves to a non-unknown SimVar in `ht` (parameter/constant),
    or is already in `foldedNames`.

Names absent from `ht` are rejected conservatively: they are typically
subscripted or array-element references whose parameter status is not
robustly recoverable by string match here.
"""
function isParameterClosureExp(@nospecialize(exp), ht, foldedNames::OrderedSet{String},
                               excludeName::String)::Bool
  local names = OrderedSet{String}()
  collectCrefNames!(names, exp)
  for n in names
    if n == excludeName
      return false
    end
    if n in foldedNames
      continue
    end
    local entry = get(ht, n, nothing)
    if entry === nothing
      return false
    end
    local (_, sv) = entry
    if isUnknownVarKind(sv.varKind)
      return false
    end
  end
  return true
end
