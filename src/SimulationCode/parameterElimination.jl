#= Observation-only variables, dead parameters, constant parameters. =#

#= Returns true when the SimVar's attributes carry isProtected = SOME(true). =#
function _isProtectedSimVar(sv)::Bool
  @match sv.attributes begin
    SOME(va) where (hasproperty(va, :isProtected) &&
                    va.isProtected isa SOME &&
                    va.isProtected.data === true) => true
    _ => false
  end
end

#= Drop "observation-only" sink variables: protected variables that are leaves
   in the equation graph (referenced by at most one residual equation, and
   never by a when/if/initial condition, assertion, alias or attribute). The
   defining equation is dropped together with the variable; iterates to a
   fixed point so a protected sink whose only consumer was another sink drops
   in the next round. Skipped on VSS / sub-model / flat-model variants.

   Visibility comes from the FlatModel `protected` keyword propagated through
   `_maybeMarkAttrProtected` in BDAECreate. =#
function dropObservationOnlyVariables(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode)
    return simCode
  end
  local ht = simCode.stringToSimVarHT

  local isDroppableKind = sv -> @match sv.varKind begin
    STATE(__) || STATE_DERIVATIVE(__) || DISCRETE(__) ||
      INPUT(__) || PARAMETER(__) || ARRAY_PARAMETER(__) || STRING(__) ||
      DATA_STRUCTURE(__) => false
    _ => true
  end

  #= Cheap eligibility scan: if no protected droppable-kind candidates exist
     at all, the residual scan and eqRefs build are pure overhead. =#
  local hasCandidate = false
  for (_, (_, sv)) in ht
    if _isProtectedSimVar(sv) && isDroppableKind(sv)
      hasCandidate = true
      break
    end
  end
  hasCandidate || return simCode

  #= "Untouchable" surfaces — any var referenced from these must stay. =#
  local untouchable = OrderedSet{String}()
  for eq in simCode.initialEquations
    if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      collectCrefNames!(untouchable, eq.exp)
    elseif eq isa BDAE.EQUATION || eq isa EQUATION
      collectCrefNames!(untouchable, eq.lhs)
      collectCrefNames!(untouchable, eq.rhs)
    end
  end
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(untouchable, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(untouchable, brEq.exp)
      end
    end
  end
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(untouchable, whenEq.whenEquation)
  end
  _collectAssertCrefNames!(untouchable, simCode.asserts)
  for entry in simCode.aliasMap
    push!(untouchable, entry.representativeName)
    push!(untouchable, entry.eliminatedName)
  end
  _collectAttributeCrefs!(untouchable, ht)
  for eq in simCode.residualEquations
    _collectIfexpConditionCrefs!(untouchable, eq.exp)
  end
  for eq in simCode.eliminatedEquations
    collectCrefNames!(untouchable, eq.exp)
  end
  _collectFunctionBodyCrefs!(untouchable, simCode.functions)

  #= Per-equation ref sets for the iterative drop. =#
  local nEqs = length(simCode.residualEquations)
  local eqRefs = Vector{OrderedSet{String}}(undef, nEqs)
  local refCount = Dict{String, Int}()
  #= Inverted index name -> ascending equation indices referencing it, built in the
     same pass. Replaces the O(nEqs) linear scan for a candidate's defining equation
     with an O(degree) lookup; ascending insertion preserves the old first-match. =#
  local varToEqs = Dict{String, Vector{Int}}()
  for i in 1:nEqs
    local s = OrderedSet{String}()
    collectCrefNames!(s, simCode.residualEquations[i].exp)
    eqRefs[i] = s
    for n in s
      refCount[n] = get(refCount, n, 0) + 1
      push!(get!(() -> Int[], varToEqs, n), i)
    end
  end

  local droppedVars = OrderedSet{String}()
  local droppedEqs  = OrderedSet{Int}()
  local progressed  = true
  while progressed
    progressed = false
    for (name, (_, sv)) in ht
      name in droppedVars && continue
      name in untouchable && continue
      _isProtectedSimVar(sv) || continue
      isDroppableKind(sv) || continue
      local nref = get(refCount, name, 0)
      nref <= 1 || continue
      local definingEq = -1
      if nref == 1
        for i in get(varToEqs, name, Int[])
          i in droppedEqs && continue
          definingEq = i; break
        end
      end
      if definingEq > 0
        for n in eqRefs[definingEq]
          refCount[n] = get(refCount, n, 0) - 1
        end
        push!(droppedEqs, definingEq)
      end
      push!(droppedVars, name)
      progressed = true
    end
  end

  isempty(droppedVars) && return simCode

  local newHT = copy(ht)
  for name in droppedVars
    delete!(newHT, name)
  end
  local newResiduals = RESIDUAL_EQUATION[]
  sizehint!(newResiduals, nEqs - length(droppedEqs))
  for i in 1:nEqs
    i in droppedEqs && continue
    push!(newResiduals, simCode.residualEquations[i])
  end
  @assign simCode.stringToSimVarHT = newHT
  @assign simCode.residualEquations = newResiduals
  @info "[SIMCODE: $(simCode.name): dropObservationOnlyVariables] dropped $(length(droppedVars)) protected sink variables and $(length(droppedEqs)) defining equations"
  return simCode
end

"""
    eliminateDeadParameters(simCode) -> simCode

Remove `PARAMETER(NONE)` simvars that are not referenced anywhere — no
residual, no initial equation, no if-condition, no when statement, no
DATA_STRUCTURE / parameter binding expression, no alias representative, no
attribute (`start` / `fixed` / `min` / `max` / `nominal`), no eliminated
equation. Such parameters cannot be observed and cannot be overridden
meaningfully at runtime (no consumer would see the override).

Skipped for sub-model / metaModel variants because cross-mode
parameter references are not visible in the standard scan.
"""
function eliminateDeadParameters(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode)
    return simCode
  end
  local ht = simCode.stringToSimVarHT

  #= Reachability scan: collect every cref name referenced from a live
     surface. Anything not in this set is dead. =#
  local referenced = OrderedSet{String}()
  for eq in simCode.residualEquations
    collectCrefNames!(referenced, eq.exp)
  end
  for eq in simCode.initialEquations
    if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      collectCrefNames!(referenced, eq.exp)
    elseif eq isa BDAE.EQUATION || eq isa EQUATION
      collectCrefNames!(referenced, eq.lhs)
      collectCrefNames!(referenced, eq.rhs)
    end
  end
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(referenced, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(referenced, brEq.exp)
      end
    end
  end
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(referenced, whenEq.whenEquation)
  end
  _collectAssertCrefNames!(referenced, simCode.asserts)
  for eq in simCode.eliminatedEquations
    collectCrefNames!(referenced, eq.exp)
  end
  for entry in simCode.aliasMap
    push!(referenced, entry.representativeName)
    push!(referenced, entry.eliminatedName)
  end
  _collectAttributeCrefs!(referenced, ht)
  #= Walk every statement body inside `simCode.functions` (Modelica user
     functions) and collect referenced crefs. Without this, a parameter
     consumed only from a function body looks dead to the scan and gets
     dropped — observed on SimpleMechanicalSystem (`tau_2`) and
     ComplexBlocks.ShowTransferFunction (`transferFunction_aw_re/_im`). =#
  _collectFunctionBodyCrefs!(referenced, simCode.functions)
  #= Protect Complex `_re`/`_im` scalarized fields. Codegen flattens
     `complexCref` to `[complexCref_re, complexCref_im]`, so if the original
     Complex CREF survives anywhere those scalar siblings must too. Mirrors
     the equivalent guard inside eliminateConstantParameters. =#
  _collectComplexFieldNames!(referenced, simCode.residualEquations, ht)
  _collectComplexFieldNames!(referenced, simCode.initialEquations, ht)
  #= Track DATA_STRUCTURE constructor-bound array bases. Every scalarized
     element of those arrays (`tableData[1][1]`, etc.) must be protected
     because codegen rebuilds the parent array from scalar siblings when a
     CombiTable1D / CombiTimeTable / similar DS constructor references the
     base. Mirrors the equivalent logic in eliminateConstantParameters. =#
  local dsArrayBaseNames = OrderedSet{String}()
  for (_n, (_, sv)) in ht
    @match sv.varKind begin
      PARAMETER(SOME(b)) => collectCrefNames!(referenced, b)
      ARRAY_PARAMETER(_, SOME(b)) => collectCrefNames!(referenced, b)
      DATA_STRUCTURE(SOME(b)) => begin
        collectCrefNames!(referenced, b)
        @match b begin
          CALL(__) => collectCrefNames!(dsArrayBaseNames, b)
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  #= Only scan HT keys for scalarized DS-array elements when there are DS-array
     bases to match; otherwise this whole-HT scan does nothing. =#
  if !isempty(dsArrayBaseNames)
    for htKey in keys(ht)
      local bracketIdx = findfirst('[', htKey)
      bracketIdx === nothing && continue
      local baseName = htKey[1:bracketIdx-1]
      if baseName in dsArrayBaseNames
        push!(referenced, htKey)
      end
    end
  end

  #= Sweep: drop any PARAMETER entry (bound or unbound) that has zero
     references on any of the live surfaces scanned above. Tunable ones stay:
     an unused element of a tunable array (a network output the model does
     not use) is still part of the array the user sets and reads back. =#
  local toDrop = String[]
  for (name, (_, sv)) in ht
    (name in referenced || isTunableParameter(name)) && continue
    local isParam = @match sv.varKind begin
      PARAMETER(__) => true
      _ => false
    end
    isParam && push!(toDrop, name)
  end

  isempty(toDrop) && return simCode

  local newHT = copy(ht)
  for name in toDrop
    delete!(newHT, name)
  end
  @assign simCode.stringToSimVarHT = newHT
  @info "[SIMCODE: $(simCode.name): eliminateDeadParameters] dropped $(length(toDrop)) unbound / unused parameters"
  return simCode
end

"""
    eliminateConstantParameters(simCode::SIM_CODE) -> SIM_CODE

Find every PARAMETER whose binding evaluates to a numeric/Bool literal,
substitute the literal value at all use sites, and drop the parameter from
`stringToSimVarHT`. This shrinks the parameter list MTK sees before
`structural_simplify`, reducing per-simulate module-eval cost on large MSL
models (where `foldParameterClosure` typically inflates the parameter count
2x to 3x).

Tier-1 only: skipped on VSS / sub-model / metaModel variants because
a parameter eliminated here can no longer be re-bound at runtime by a
structural transition or by recompilation. The gate matches the
conservative envelope used by `eliminateAliasVariables`.

Defensive checks:
- Parameters that appear as representatives in `aliasMap` are NOT eliminated
  (would orphan the alias entry).
- A survivor scan after substitution keeps any parameter still referenced
  somewhere the substitution missed (paranoia for unflatten CREF forms).
"""
function eliminateConstantParameters(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode) ||
     hasMetaModel(simCode)
    @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] skipped (VSS/recompilation/sub-model variant)"
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  local paramValueMap = Dict{String, Float64}()
  local seen = OrderedSet{String}()

  #= Build the protected-from-elimination set. We keep any parameter that:
     1. Is an alias representative (eliminating orphans the alias entry).
     2. Is referenced as a CREF in another simvar's `start`/`fixed`/`min`/
        `max`/`nominal` attribute. The MTK codegen short-circuits start
        attributes via `pars[Symbol(name)]`, bypassing the equation
        substitution map; eliminating such a parameter produces a runtime
        UndefVarError when the model module evaluates.
     3. Is referenced as a condition in any IF_EQUATION branch — these are
        structural switches the user may want to flip.
     4. Is referenced as a condition in any WHEN_EQUATION.
     5. Is referenced as a condition in any IFEXP, anywhere in equations or
        in another parameter's binding.
     6. Is referenced anywhere in any initial equation. Initial equations
        carry constraints MTK uses at t=0; we keep their parameter inputs
        intact so the user can re-bind a parameter and re-initialize without
        a recompile (where supported by MTK). =#
  local protectedNames = OrderedSet{String}()
  for entry in simCode.aliasMap
    push!(protectedNames, entry.representativeName)
  end
  _collectAttributeCrefs!(protectedNames, ht)
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(protectedNames, branch.condition)
    end
  end
  for whenEq in simCode.whenEquations
    _collectWhenConditionCrefs!(protectedNames, whenEq.whenEquation)
  end
  for eq in simCode.initialEquations
    if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      collectCrefNames!(protectedNames, eq.exp)
    elseif eq isa BDAE.EQUATION || eq isa EQUATION
      collectCrefNames!(protectedNames, eq.lhs)
      collectCrefNames!(protectedNames, eq.rhs)
    end
  end
  #= IFEXP conditions inside residual equations and parameter bindings. =#
  for eq in simCode.residualEquations
    _collectIfexpConditionCrefs!(protectedNames, eq.exp)
  end
  #= Names of array bases referenced as bare CREFs in DATA_STRUCTURE constructor
     calls (ExternalObject inits like CombiTable / CombiTimeTable). Array params
     are scalarized into HT entries like `tableData[1][1]`..., but the constructor
     call bind references the whole array (`tableData`). Eliminating any
     scalarized element would leave the constructor referring to data that no
     longer survives codegen, so protect every scalar element of those arrays.

     Restricted to DS bindings whose RHS is a CALL — MSL constants
     (BDAE.CONST of scalar type) are also stored as DATA_STRUCTURE but their
     RHS is a literal and over-protecting them would block legitimate
     constant-propagation eliminations elsewhere. =#
  local dsArrayBaseNames = OrderedSet{String}()
  for (_, htEntry) in ht
    local (_, svP) = htEntry
    @match svP.varKind begin
      PARAMETER(SOME(b))            => _collectIfexpConditionCrefs!(protectedNames, b)
      ARRAY_PARAMETER(_, SOME(b))   => _collectIfexpConditionCrefs!(protectedNames, b)
      DATA_STRUCTURE(SOME(b)) => begin
        @match b begin
          CALL(__) => begin
            collectCrefNames!(protectedNames, b)
            collectCrefNames!(dsArrayBaseNames, b)
          end
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  #= Only scan HT keys for scalarized DS-array elements when there are DS-array
     bases to match; otherwise this whole-HT scan does nothing. =#
  if !isempty(dsArrayBaseNames)
    for htKey in keys(ht)
      local bracketIdx = findfirst('[', htKey)
      bracketIdx === nothing && continue
      local baseName = htKey[1:bracketIdx-1]
      if baseName in dsArrayBaseNames
        push!(protectedNames, htKey)
      end
    end
  end

  #= Protect scalar field params backing complex CREFs that survive in equations.
     Magnetic.QuasiStationary models reference `converter_m_N` (T_COMPLEX) in
     residual equations; codegen flattens this to `[converter_m_N_re,
     converter_m_N_im]` symbols. If those scalar fields are constant params
     they get eliminated here, but the flatten happens later and looks them up
     by symbol — UndefVarError at module eval. =#
  _collectComplexFieldNames!(protectedNames, simCode.residualEquations, ht)
  _collectComplexFieldNames!(protectedNames, simCode.initialEquations, ht)
  #= A parameter consumed only from a Modelica function body is otherwise
     invisible to the equation/attribute/condition scans above; without this it
     can be folded out of the HT while the function body still references its
     symbol -> UndefVarError at module eval. Mirrors the sibling passes
     dropObservationOnlyVariables (4391) and eliminateDeadParameters (4500). =#
  _collectFunctionBodyCrefs!(protectedNames, simCode.functions)
  #= Tunable parameters stay (withTunableParameters); parameters whose bindings
     depend on them do not evaluate below, so they stay too. =#
  if !isempty(TUNABLE_PARAMETERS[])
    union!(protectedNames, TUNABLE_PARAMETERS[])
    for k in keys(ht)
      isTunableParameter(k) && push!(protectedNames, k)
    end
  end

  #= Step 1: identify eliminable parameters via _tryEvalNumeric. =#
  for (name, htEntry) in ht
    name in protectedNames && continue
    local (_, sv) = htEntry
    local bindExp = @match sv.varKind begin
      PARAMETER(SOME(e)) => toDAEExp(e)
      _ => nothing
    end
    bindExp === nothing && continue
    empty!(seen)
    local v = _tryEvalNumeric(bindExp, simCode, seen)
    v === nothing && continue
    paramValueMap[name] = v
  end

  # Enumerate ARRAY_PARAMETER element bindings; iterate to fixed point so
  # chained array references resolve in dependency order.
  local arrChanged = true
  while arrChanged
    arrChanged = false
    local mapSizeBefore = length(paramValueMap)
    for (name, htEntry) in ht
      name in protectedNames && continue
      local (_, sv) = htEntry
      local arrBind = @match sv.varKind begin
        ARRAY_PARAMETER(_, SOME(e)) => toDAEExp(e)
        _ => nothing
      end
      arrBind === nothing && continue
      _enumerateArrayParamElements!(paramValueMap, name, arrBind, simCode, seen, protectedNames)
    end
    arrChanged = length(paramValueMap) > mapSizeBefore
  end

  if isempty(paramValueMap)
    @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] no eliminable parameters found"
    return simCode
  end

  #= Step 2: substitute throughout every equation container. =#
  local newResiduals = RESIDUAL_EQUATION[]
  for eq in simCode.residualEquations
    local (newExp, _) = traverseExpTopDown(eq.exp, substituteConstantParameter, paramValueMap)
    push!(newResiduals, typeof(eq)(newExp, eq.source, eq.attr))
  end

  local newInitials = typeof(simCode.initialEquations)()
  for eq in simCode.initialEquations
    local newEq = if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      local (newExp, _) = Util.traverseExpTopDown(toDAEExp(eq.exp), substituteConstantParameter, paramValueMap)
      typeof(eq)(newExp, eq.source, eq.attr)
    elseif eq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(eq.lhs), substituteConstantParameter, paramValueMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(eq.rhs), substituteConstantParameter, paramValueMap)
      BDAE.EQUATION(newLhs, newRhs, eq.source, eq.attributes)
    elseif eq isa EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(eq.lhs), substituteConstantParameter, paramValueMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(eq.rhs), substituteConstantParameter, paramValueMap)
      EQUATION(newLhs, newRhs, eq.source, eq.attr)
    else
      eq
    end
    push!(newInitials, newEq)
  end

  local newIfEquations = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = traverseExpTopDown(brEq.exp, substituteConstantParameter, paramValueMap)
        push!(newBranchEqs, typeof(brEq)(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = traverseExpTopDown(branch.condition, substituteConstantParameter, paramValueMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEquations, IF_EQUATION(newBranches))
  end

  local newWhenEquations = WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local newInner = _substituteParamInWhenStmts(whenEq.whenEquation, paramValueMap)
    @assign whenEq.whenEquation = newInner
    push!(newWhenEquations, whenEq)
  end

  # alias-eliminated residuals are emitted verbatim by codegen; substitute
  # eliminated-parameter element refs to avoid dangling identifiers
  local newElimEqs = RESIDUAL_EQUATION[]
  for eq in simCode.eliminatedEquations
    local (newExp, _) = traverseExpTopDown(eq.exp, substituteConstantParameter, paramValueMap)
    push!(newElimEqs, typeof(eq)(newExp, eq.source, eq.attr))
  end

  # substitute into surviving PARAMETER and ARRAY_PARAMETER bindings
  local newHT = copy(ht)
  for (name, htEntry) in ht
    haskey(paramValueMap, name) && continue
    local (idx, sv) = htEntry
    local newKind = @match sv.varKind begin
      PARAMETER(SOME(b)) => begin
        local (nb, _) = traverseExpTopDown(b, substituteConstantParameter, paramValueMap)
        nb === b ? sv.varKind : PARAMETER(SOME(nb))
      end
      ARRAY_PARAMETER(dims, SOME(b)) => begin
        local (nb, _) = traverseExpTopDown(b, substituteConstantParameter, paramValueMap)
        nb === b ? sv.varKind : ARRAY_PARAMETER(dims, SOME(nb))
      end
      _ => sv.varKind
    end
    if newKind !== sv.varKind
      newHT[name] = (idx, SIMVAR(sv.name, sv.index, newKind, sv.attributes))
    end
  end

  #= Step 4: defensive survivor scan. If a CREF for a candidate parameter
     somehow survived substitution (unflatten form, etc.), keep the param. =#
  local survivorCheck = OrderedSet{String}()
  for eq in newResiduals
    collectCrefNames!(survivorCheck, eq.exp)
  end
  for eq in newInitials
    if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      collectCrefNames!(survivorCheck, eq.exp)
    elseif eq isa BDAE.EQUATION || eq isa EQUATION
      collectCrefNames!(survivorCheck, eq.lhs)
      collectCrefNames!(survivorCheck, eq.rhs)
    end
  end
  for ifEq in newIfEquations
    for branch in ifEq.branches
      for brEq in branch.residualEquations
        collectCrefNames!(survivorCheck, brEq.exp)
      end
      collectCrefNames!(survivorCheck, branch.condition)
    end
  end
  for whenEq in newWhenEquations
    _collectWhenCrefNames!(survivorCheck, whenEq.whenEquation)
  end

  #= Step 5: drop eliminated params from HT, skipping survivors. =#
  local elimNames = String[]
  local survivors = String[]
  for (name, _) in paramValueMap
    if name in survivorCheck
      push!(survivors, name)
      continue
    end
    delete!(newHT, name)
    push!(elimNames, name)
  end

  if !isempty(survivors)
    @warn "[SIMCODE: $(simCode.name): eliminateConstantParameters] $(length(survivors)) parameters still referenced after substitution; keeping them" survivors
  end

  if isempty(elimNames)
    @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] nothing eliminated (all candidates survived substitution)"
    return simCode
  end

  @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] eliminated $(length(elimNames)) parameters of $(length(paramValueMap)) candidates"

  @assign begin
    simCode.residualEquations = newResiduals
    simCode.initialEquations = newInitials
    simCode.ifEquations = newIfEquations
    simCode.whenEquations = newWhenEquations
    simCode.eliminatedEquations = newElimEqs
    simCode.stringToSimVarHT = newHT
    simCode.asserts = _substituteInAsserts(simCode.asserts, paramValueMap; visitor = substituteConstantParameter)
  end
  #= Do NOT append eliminated parameter names to `simCode.eliminatedVariables`.
     That list pairs with `simCode.eliminatedEquations` 1:1 and is consumed by
     `generateEliminatedObservedBlock`, which expects each eliminated name to
     have a defining residual equation. Parameters are substituted directly
     into equations and have no residual to reconstruct, so adding them breaks
     the parallel-array invariant. =#
  return simCode
end

"""
Walk every statement body inside `simCode.functions` (user-defined Modelica
functions) and add every CREF name encountered to `out`. Used by
reachability scans that protect parameters / variables consumed only from
function bodies.
"""
function _collectFunctionBodyCrefs!(out::OrderedSet{String}, functions)
  for fn in functions
    try
      @match fn begin
        MODELICA_FUNCTION(__) => _walkStatementsForCrefs!(out, fn.statements)
        _ => nothing
      end
    catch
      #= Be tolerant: a bad statement variant or unexpected field count must
         not break the surrounding pass. Worst case is we miss a few crefs
         and over-eliminate downstream — the survivor-scan in callers (and
         SimCodeCheck `rule_cref_resolution`) flags that. =#
    end
  end
  return out
end

function _walkStatementsForCrefs!(out::OrderedSet{String}, stmts)
  for s in stmts
    try
      @match s begin
        DAE.STMT_ASSIGN(__) => begin
          collectCrefNames!(out, s.exp1)
          collectCrefNames!(out, s.exp)
        end
        DAE.STMT_ASSIGN_ARR(__) => begin
          collectCrefNames!(out, s.exp1)
          collectCrefNames!(out, s.exp)
        end
        DAE.STMT_IF(__) => begin
          collectCrefNames!(out, s.exp1)
          _walkStatementsForCrefs!(out, s.statementLst)
        end
        DAE.STMT_FOR(__) => begin
          if isdefined(s, :range)
            collectCrefNames!(out, s.range)
          end
          if isdefined(s, :statementLst)
            _walkStatementsForCrefs!(out, s.statementLst)
          end
        end
        DAE.STMT_WHILE(__) => begin
          collectCrefNames!(out, s.exp)
          _walkStatementsForCrefs!(out, s.statementLst)
        end
        DAE.STMT_WHEN(__) => begin
          collectCrefNames!(out, s.exp)
          _walkStatementsForCrefs!(out, s.statementLst)
        end
        DAE.STMT_NORETCALL(__) => collectCrefNames!(out, s.exp)
        _ => nothing
      end
    catch
      #= Skip statements with shapes we do not know about. Conservative. =#
    end
  end
  return out
end

"""
Collect every CREF appearing in a CREF-valued attribute (`start`, `fixed`,
`min`, `max`, `nominal`) of any simvar in `ht`. These names must not be
eliminated — the MTK start-condition codegen references them via
`pars[Symbol(name)]`, which bypasses equation-level substitution.
"""
function _collectAttributeCrefs!(out::OrderedSet{String}, ht::AbstractDict)
  for (_, htEntry) in ht
    local (_, sv) = htEntry
    local optAttrs = sv.attributes
    @match optAttrs begin
      SOME(attrs) => begin
        for fname in (:start, :fixed, :min, :max, :nominal)
          if hasproperty(attrs, fname)
            local fv = getproperty(attrs, fname)
            @match fv begin
              SOME(e) => collectCrefNames!(out, e)
              _ => nothing
            end
          end
        end
      end
      _ => nothing
    end
  end
  return out
end

"""
Collect every CREF appearing in an IFEXP condition anywhere in `exp`. CREFs
appearing only in IFEXP branches (`then`/`else`) are NOT collected. Used to
protect parameters that gate runtime conditional branches from elimination.
"""
# SIM-native: collect crefs appearing in any IFEXP condition (protects params used
# in conditions from constant-elimination). Walk the SIM tree; at each IFEXP collect
# its condition's crefs (collectCrefNames! grabs all of them, nested ones included).
# Pure read-only recursive walk over the SIM tree: at each IFEXP collect its
# condition's crefs (collectCrefNames! grabs all, nested ones included), and
# descend into every child. Avoids both toDAEExp and the rebuilding
# traverseExpTopDown.
function _collectIfexpConditionCrefs!(out::OrderedSet{String}, exp::Exp)
  if exp isa IFEXP
    collectCrefNames!(out, exp.cond)
    _collectIfexpConditionCrefs!(out, exp.cond)
    _collectIfexpConditionCrefs!(out, exp.thenExp)
    _collectIfexpConditionCrefs!(out, exp.elseExp)
  elseif exp isa BINARY || exp isa LBINARY || exp isa RELATION
    _collectIfexpConditionCrefs!(out, exp.exp1)
    _collectIfexpConditionCrefs!(out, exp.exp2)
  elseif exp isa UNARY || exp isa LUNARY
    _collectIfexpConditionCrefs!(out, exp.exp)
  elseif exp isa CALL
    for a in exp.args
      _collectIfexpConditionCrefs!(out, a)
    end
  elseif exp isa ARRAY_EXP
    for x in exp.elements
      _collectIfexpConditionCrefs!(out, x)
    end
  elseif exp isa ASUB
    _collectIfexpConditionCrefs!(out, exp.exp)
    for s in exp.subs
      _collectIfexpConditionCrefs!(out, s)
    end
  elseif exp isa TSUB || exp isa RSUB || exp isa CAST
    _collectIfexpConditionCrefs!(out, exp.exp)
  elseif exp isa RECORD
    for x in exp.exps
      _collectIfexpConditionCrefs!(out, x)
    end
  elseif exp isa TUPLE
    for x in exp.PR
      _collectIfexpConditionCrefs!(out, x)
    end
  elseif exp isa REDUCTION
    _collectIfexpConditionCrefs!(out, exp.body)
  end
  return out
end

function _collectIfexpConditionCrefs!(out::OrderedSet{String}, @nospecialize(exp))
  @match exp begin
    DAE.IFEXP(expCond = c, expThen = t, expElse = e) => begin
      collectCrefNames!(out, c)
      _collectIfexpConditionCrefs!(out, t)
      _collectIfexpConditionCrefs!(out, e)
    end
    DAE.BINARY(exp1 = e1, exp2 = e2) => begin
      _collectIfexpConditionCrefs!(out, e1)
      _collectIfexpConditionCrefs!(out, e2)
    end
    DAE.UNARY(exp = e1)        => _collectIfexpConditionCrefs!(out, e1)
    DAE.LUNARY(exp = e1)       => _collectIfexpConditionCrefs!(out, e1)
    DAE.LBINARY(exp1 = e1, exp2 = e2) => begin
      _collectIfexpConditionCrefs!(out, e1)
      _collectIfexpConditionCrefs!(out, e2)
    end
    DAE.RELATION(exp1 = e1, exp2 = e2) => begin
      _collectIfexpConditionCrefs!(out, e1)
      _collectIfexpConditionCrefs!(out, e2)
    end
    DAE.CALL(expLst = args) => begin
      for arg in args
        _collectIfexpConditionCrefs!(out, arg)
      end
    end
    DAE.ARRAY(array = lst) => begin
      for e in lst
        _collectIfexpConditionCrefs!(out, e)
      end
    end
    DAE.ASUB(exp = e, sub = subs) => begin
      _collectIfexpConditionCrefs!(out, e)
      #= subs are DAE.Subscript; collect from their inner expressions. =#
      for s in subs
        @match s begin
          DAE.INDEX(se) => _collectIfexpConditionCrefs!(out, se)
          DAE.SLICE(se) => _collectIfexpConditionCrefs!(out, se)
          DAE.WHOLE_NONEXP(se) => _collectIfexpConditionCrefs!(out, se)
          _ => ()
        end
      end
    end
    DAE.CAST(exp = e1) => _collectIfexpConditionCrefs!(out, e1)
    _ => nothing
  end
  return out
end

"""
Collect CREFs in the condition of a `WHEN_STMTS` node (BDAE or SIM) and
any nested `elsewhen`. Statements inside the when-clause are handled
separately via the equation walk; we only protect parameters that gate
the trigger.
"""
function _collectWhenConditionCrefs!(out::OrderedSet{String}, whenStmts::WHEN_STMTS)
  collectCrefNames!(out, whenStmts.condition)
  if whenStmts.elsewhenPart !== nothing
    _collectWhenConditionCrefs!(out, whenStmts.elsewhenPart)
  end
  return out
end

function _collectWhenConditionCrefs!(out::OrderedSet{String}, whenStmts::BDAE.WHEN_STMTS)
  collectCrefNames!(out, whenStmts.condition)
  @match whenStmts.elsewhenPart begin
    SOME(inner) => _collectWhenConditionCrefs!(out, inner)
    _ => nothing
  end
  return out
end

function _collectWhenConditionCrefs!(out::OrderedSet{String}, whenEq::Union{BDAE.WHEN_EQUATION, WHEN_EQUATION})
  return _collectWhenConditionCrefs!(out, whenEq.whenEquation)
end

# Walk a DAE.ARRAY binding and add one paramValueMap entry per numeric element.
function _enumerateArrayParamElements!(paramValueMap, baseName::String,
                                       exp, simCode,
                                       seen::OrderedSet{String},
                                       protectedNames::OrderedSet{String})
  exp isa DAE.ARRAY || return nothing
  local i = 0
  for elem in exp.array
    i += 1
    local elemName = Base.string(baseName, "[", i, "]")
    elemName in protectedNames && continue
    if elem isa DAE.ARRAY
      _enumerateArrayParamElements!(paramValueMap, elemName, elem, simCode,
                                    seen, protectedNames)
    else
      empty!(seen)
      local v = _tryEvalNumeric(elem, simCode, seen)
      # fall back to map lookup when the element binding is a CREF/ASUB
      # to a previously-enumerated array element
      if v === nothing
        local refName = _asubCanonicalName(elem)
        if refName !== nothing && haskey(paramValueMap, refName)
          v = paramValueMap[refName]
        end
      end
      v !== nothing && (paramValueMap[elemName] = v)
    end
  end
  return nothing
end

# Canonical name for a (possibly nested) DAE.ASUB; nothing if non-constant.
function _asubCanonicalName(@nospecialize(exp))::Union{Nothing,String}
  @match exp begin
    DAE.CREF(cr, _) => DAE_identifierToString(cr)
    DAE.ASUB(inner, subs) => begin
      local innerName = _asubCanonicalName(inner)
      innerName === nothing && return nothing
      local idxParts = String[]
      for s in subs
        local v = @match s begin
          DAE.INDEX(DAE.ICONST(i)) => i
          DAE.INDEX(DAE.RCONST(r)) where r == round(r) => Int(round(r))
          _ => nothing
        end
        v === nothing && return nothing
        push!(idxParts, Base.string("[", v, "]"))
      end
      Base.string(innerName, idxParts...)
    end
    _ => nothing
  end
end

function substituteConstantParameter(@nospecialize(exp), paramValueMap)
  @match exp begin
    DAE.CREF(cr, ty) => begin
      local name = DAE_identifierToString(cr)
      if haskey(paramValueMap, name)
        local v = paramValueMap[name]
        local literalExp = @match ty begin
          DAE.T_REAL(__)    => DAE.RCONST(v)
          DAE.T_INTEGER(__) => DAE.ICONST(Int(round(v)))
          DAE.T_BOOL(__)    => DAE.BCONST(v != 0.0)
          _                 => DAE.RCONST(v)
        end
        return (literalExp, false, paramValueMap)
      end
      (exp, true, paramValueMap)
    end
    DAE.ASUB(__) => begin
      local name = _asubCanonicalName(exp)
      if name !== nothing && haskey(paramValueMap, name)
        local v = paramValueMap[name]
        return (DAE.RCONST(v), false, paramValueMap)
      end
      (exp, true, paramValueMap)
    end
    _ => (exp, true, paramValueMap)
  end
end

#= SIM-native dispatch: replace a matched parameter cref with a typed literal,
   reading the literal kind from EXP_CREF.ty; only the matched leaf converts. =#
function substituteConstantParameter(exp::EXP_CREF, paramValueMap)
  local name = DAE_identifierToString(toDAECref(exp.cref).componentRef)
  if haskey(paramValueMap, name)
    local v = paramValueMap[name]
    local literalExp = exp.ty isa TYPE_REAL    ? RCONST(v) :
                       exp.ty isa TYPE_INTEGER ? ICONST(Int(round(v))) :
                       exp.ty isa TYPE_BOOL    ? BCONST(v != 0.0) : RCONST(v)
    return (literalExp, false, paramValueMap)
  end
  return (exp, true, paramValueMap)
end

function substituteConstantParameter(exp::ASUB, paramValueMap)
  local name = _asubCanonicalNameSIM(exp)
  if name !== nothing && haskey(paramValueMap, name)
    return (RCONST(paramValueMap[name]), false, paramValueMap)
  end
  return (exp, true, paramValueMap)
end

substituteConstantParameter(exp::Exp, paramValueMap) = (exp, true, paramValueMap)

# SIM-native mirror of _asubCanonicalName: nested ASUB over EXP_CREF with constant subs.
function _asubCanonicalNameSIM(@nospecialize(e))::Union{Nothing,String}
  if e isa EXP_CREF
    return DAE_identifierToString(toDAECref(e.cref).componentRef)
  elseif e isa ASUB
    local innerName = _asubCanonicalNameSIM(e.exp)
    innerName === nothing && return nothing
    local idxParts = String[]
    for s in e.subs
      local v = s isa ICONST ? s.value :
                (s isa RCONST && s.value == round(s.value)) ? Int(round(s.value)) : nothing
      v === nothing && return nothing
      push!(idxParts, Base.string("[", v, "]"))
    end
    return Base.string(innerName, idxParts...)
  end
  return nothing
end

"""
Recursively substitute eliminated-parameter CREFs in a WHEN_STMTS node.
Mirrors `_substituteAliasInWhenStmts` but with `substituteConstantParameter`.
"""
function _substituteParamInWhenStmts(whenStmts::WHEN_STMTS, paramValueMap)
  local (newCond, _) = traverseExpTopDown(whenStmts.condition, substituteConstantParameter, paramValueMap)
  local newStmtLst = WhenOperator[]
  for stmt in whenStmts.whenStmtLst
    local newStmt::WhenOperator = if stmt isa ASSIGN
      local (newL, _) = traverseExpTopDown(stmt.left, substituteConstantParameter, paramValueMap)
      local (newR, _) = traverseExpTopDown(stmt.right, substituteConstantParameter, paramValueMap)
      ASSIGN(newL, newR, stmt.source)
    elseif stmt isa REINIT
      local (newSV, _) = Util.traverseExpTopDown(stmt.stateVar, substituteConstantParameter, paramValueMap)
      local (newVal, _) = traverseExpTopDown(stmt.value, substituteConstantParameter, paramValueMap)
      REINIT(newSV, newVal, stmt.source)
    elseif stmt isa NORETCALL
      local (newExp, _) = traverseExpTopDown(stmt.exp, substituteConstantParameter, paramValueMap)
      NORETCALL(newExp, stmt.source)
    else
      stmt
    end
    push!(newStmtLst, newStmt)
  end
  local newElseWhen = whenStmts.elsewhenPart === nothing ? nothing :
                      _substituteParamInWhenStmts(whenStmts.elsewhenPart, paramValueMap)
  return WHEN_STMTS(newCond, newStmtLst, newElseWhen)
end

function _substituteParamInWhenStmts(whenStmts::BDAE.WHEN_STMTS, paramValueMap)
  local (newCond, _) = Util.traverseExpTopDown(toDAEExp(whenStmts.condition), substituteConstantParameter, paramValueMap)
  local newStmtLst::List{BDAE.WhenOperator} = MetaModelica.nil
  for stmt in whenStmts.whenStmtLst
    local newStmt::BDAE.WhenOperator = @match stmt begin
      BDAE.ASSIGN(__) => begin
        local (newL, _) = Util.traverseExpTopDown(stmt.left, substituteConstantParameter, paramValueMap)
        local (newR, _) = Util.traverseExpTopDown(stmt.right, substituteConstantParameter, paramValueMap)
        BDAE.ASSIGN(newL, newR, stmt.source)
      end
      BDAE.REINIT(__) => begin
        local (newSV, _) = Util.traverseExpTopDown(stmt.stateVar, substituteConstantParameter, paramValueMap)
        local (newVal, _) = Util.traverseExpTopDown(stmt.value, substituteConstantParameter, paramValueMap)
        BDAE.REINIT(newSV, newVal, stmt.source)
      end
      BDAE.NORETCALL(__) => begin
        local (newExp, _) = Util.traverseExpTopDown(stmt.exp, substituteConstantParameter, paramValueMap)
        BDAE.NORETCALL(newExp, stmt.source)
      end
      _ => stmt
    end
    newStmtLst = MetaModelica.Cons{BDAE.WhenOperator}(newStmt, newStmtLst)
  end
  newStmtLst = MetaModelica.listReverse(newStmtLst)
  local newElseWhen = @match whenStmts.elsewhenPart begin
    SOME(inner) => SOME(_substituteParamInWhenStmts(inner, paramValueMap))
    NONE() => NONE()
  end
  return BDAE.WHEN_STMTS(newCond, newStmtLst, newElseWhen)
end
