#= Elimination of the variables that only feed outputs (eliminateNonDynamic). =#

"""
    identifyOutputOnlyVariables(simCode::SIM_CODE,
                                matchOrder::Vector{Int},
                                matchIdxToName::Dict{Int,String})

Identify variables and equations that do not influence the dynamic states.
Uses a fresh bipartite matching and robust CREF extraction via `traverseExpTopDown`.

The BFS seeds from equations matched to essential variables (STATE, STATE_DERIVATIVE,
DISCRETE, irreducible). It propagates backward through the use-def chain: for
each essential equation, all variables it references are marked essential, and the
equations that PRODUCE those variables (via matchOrder) are enqueued.

Returns `(outputOnlyVarNames, outputOnlyEqIndices, eqRefs)`.
"""
function identifyOutputOnlyVariables(simCode::SIM_CODE,
                                     matchOrder::Vector{Int},
                                     matchIdxToName::Dict{Int,String})
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  #= Build the base name index for robust CREF extraction =#
  local baseNameToFullNames = buildBaseNameIndex(ht)
  #= Build expression-level dependency: for each equation, which variable names does it reference? =#
  local eqRefs = Vector{OrderedSet{String}}(undef, nEqs)
  for i in 1:nEqs
    eqRefs[i] = collectEquationVarNames(toDAEExp(resEqs[i].exp), ht, baseNameToFullNames)
  end
  #= Build varName -> equation index that solves it (via fresh matchOrder).
     matchOrder[matchIdx] = eqIdx; matchIdxToName[matchIdx] = varName =#
  local varNameToEq = Dict{String, Int}()
  local baseNameToEqs = Dict{String, Vector{Int}}()
  for (matchIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0 && haskey(matchIdxToName, matchIdx)
      local vn = matchIdxToName[matchIdx]
      varNameToEq[vn] = eqIdx
      local bi = findfirst('[', vn)
      local bn = bi === nothing ? vn : vn[1:(bi - 1)]
      if bn != vn
        if !haskey(baseNameToEqs, bn)
          baseNameToEqs[bn] = Int[]
        end
        push!(baseNameToEqs[bn], eqIdx)
      end
    end
  end
  #= Find seed equations: those matched to essential variable kinds =#
  local seedEqs = OrderedSet{Int}()
  for (varName, (_idx, sv)) in ht
    local isEssentialKind = @match sv.varKind begin
      STATE(__) => true
      STATE_DERIVATIVE(__) => true
      DISCRETE(__) => true
      _ => false
    end
    if isEssentialKind && haskey(varNameToEq, varName)
      push!(seedEqs, varNameToEq[varName])
    end
  end
  #= Add equations for irreducible variables =#
  for irName in simCode.irreducibleVariables
    if haskey(varNameToEq, irName)
      push!(seedEqs, varNameToEq[irName])
    end
  end
  #= Protect alias representative variables from elimination.
     These variables appear in observed equations generated from the aliasMap.
     If they are eliminated, the observed equations will reference missing unknowns. =#
  for alias in simCode.aliasMap
    if haskey(varNameToEq, alias.representativeName)
      push!(seedEqs, varNameToEq[alias.representativeName])
    end
  end
  #= Asserts are checked during the simulation: what they read is computed. =#
  local assertRefs = _collectAssertCrefNames!(OrderedSet{String}(), simCode.asserts)
  for vn in assertRefs
    haskey(varNameToEq, vn) && push!(seedEqs, varNameToEq[vn])
  end
  #= If-equations are not residual equations, so the search below does not
     pass through them: what their conditions and branches read is computed. =#
  local ifEqRefs = OrderedSet{String}()
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(ifEqRefs, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(ifEqRefs, brEq.exp)
      end
    end
  end
  for vn in ifEqRefs
    haskey(varNameToEq, vn) && push!(seedEqs, varNameToEq[vn])
  end
  #= Classify unmatched equations: seed those referencing unknowns =#
  local matchedEqs = OrderedSet{Int}()
  for (matchIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      push!(matchedEqs, eqIdx)
    end
  end
  local unknownNames = OrderedSet{String}()
  for (vn, (_idx, sv)) in ht
    local isUnknown = @match sv.varKind begin
      STATE(__) => true
      STATE_DERIVATIVE(__) => true
      ALG_VARIABLE(__) => true
      SimulationCode.ARRAY(__) => true
      DISCRETE(__) => true
      _ => false
    end
    if isUnknown
      push!(unknownNames, vn)
    end
  end
  for eqIdx in 1:nEqs
    if !(eqIdx in matchedEqs)
      #= Check if this unmatched equation references any unknowns =#
      local refsUnknown = false
      for refName in eqRefs[eqIdx]
        if refName in unknownNames
          refsUnknown = true
          break
        end
      end
      if refsUnknown
        push!(seedEqs, eqIdx)
      end
    end
  end
  #= BFS: from seed equations, follow the use-def chain backward.
     For each equation, find all variable names it references. For each referenced
     variable, find the equation that PRODUCES it (via varNameToEq). Enqueue that. =#
  local essentialEqs = OrderedSet{Int}()
  local queue = collect(seedEqs)
  while !isempty(queue)
    local eqIdx = popfirst!(queue)
    if eqIdx in essentialEqs
      continue
    end
    push!(essentialEqs, eqIdx)
    if eqIdx >= 1 && eqIdx <= nEqs
      for refVarName in eqRefs[eqIdx]
        #= Exact match =#
        if haskey(varNameToEq, refVarName)
          local prodEq = varNameToEq[refVarName]
          if !(prodEq in essentialEqs)
            push!(queue, prodEq)
          end
        end
        #= Base name match for array variables =#
        if haskey(baseNameToEqs, refVarName)
          for prodEq in baseNameToEqs[refVarName]
            if !(prodEq in essentialEqs)
              push!(queue, prodEq)
            end
          end
        end
      end
    end
  end
  #= Identify output-only equations and their matched variables =#
  local outputOnlyEqIndices = OrderedSet{Int}()
  local outputOnlyVarNames = OrderedSet{String}()
  local eqToMatchIdx = Dict{Int, Int}()
  for (matchIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      eqToMatchIdx[eqIdx] = matchIdx
    end
  end
  for eqIdx in 1:nEqs
    if !(eqIdx in essentialEqs)
      push!(outputOnlyEqIndices, eqIdx)
      if haskey(eqToMatchIdx, eqIdx)
        local mIdx = eqToMatchIdx[eqIdx]
        if haskey(matchIdxToName, mIdx)
          push!(outputOnlyVarNames, matchIdxToName[mIdx])
        end
      end
    end
  end
  return (outputOnlyVarNames, outputOnlyEqIndices, eqRefs)
end

#= True when the variable's attributes carry an explicit `fixed = true` AND
   an explicit `start = ...` value. Used to rescue variables from elimination
   passes that would otherwise drop the user-pinned initial condition. =#
function _hasExplicitFixedStart(@nospecialize(attrs))::Bool
  return @match attrs begin
    SOME(va) where (va isa DAE.VAR_ATTR_REAL) => begin
      local fixedTrue = @match va.fixed begin
        SOME(DAE.BCONST(true)) => true
        _ => false
      end
      local hasStart = @match va.start begin
        SOME(_) => true
        _ => false
      end
      fixedTrue && hasStart
    end
    _ => false
  end
end

"`true` if `exp` is a literal `1` exponent, so `base ^ exp` stays affine in base."
function _isUnitExponent(exp::Exp)::Bool
  @match exp begin
    RCONST(v) => v == 1.0
    ICONST(v) => v == 1
    _ => false
  end
end

"ASUB scalar HT name for an all-constant-subscript cref base, else `nothing`."
function _asubScalarName(exp::ASUB)::Union{String, Nothing}
  exp.exp isa EXP_CREF || return nothing
  local suffix = _simConstSubscriptSuffix(exp.subs)
  suffix === nothing && return nothing
  return Base.string(DAE_identifierToString(toDAECref(exp.exp.cref).componentRef), suffix)
end

"""
    _simCrefScalarName(exp::Exp) -> Union{String, Nothing}

Canonical scalar hash-table name for a cref-shaped `exp` (matching
`collectCrefNames!` keys), or `nothing` when `exp` is not a cref or has no
stable scalar name (e.g. an ASUB with non-constant subscripts).
"""
function _simCrefScalarName(exp::Exp)::Union{String, Nothing}
  @match exp begin
    EXP_CREF(__) => DAE_identifierToString(toDAECref(exp.cref).componentRef)
    ASUB(__) => _asubScalarName(exp)
    _ => nothing
  end
end

"`true` if `varName` is referenced anywhere in `exp` (reuses `collectCrefNames!`)."
function _occursAnywhere(exp::Exp, varName::AbstractString)::Bool
  local names = OrderedSet{String}()
  collectCrefNames!(names, exp)
  return varName in names
end

"""
    _powLinearity(exponent, oBase, lBase, oExp) -> (occurs, linear)

Linearity of `base ^ exponent` w.r.t. the target variable, from the base's
occurrence/linearity (`oBase`, `lBase`) and whether the exponent contains it
(`oExp`). Only `base ^ 1` with a linear base stays affine.
"""
function _powLinearity(exponent::Exp, oBase::Bool, lBase::Bool, oExp::Bool)::Tuple{Bool, Bool}
  oExp && return (true, false)
  oBase || return (false, true)
  return _isUnitExponent(exponent) ? (true, lBase) : (true, false)
end

"Occurrence/linearity of `varName` across a SIM `BINARY` node. Enum operators
are compared by value (`@match` treats a bare enum name as a capture binding)."
function _binaryLinearity(exp::BINARY, varName::AbstractString)::Tuple{Bool, Bool}
  local (o1, l1) = _occursLinearly(exp.exp1, varName)
  local (o2, l2) = _occursLinearly(exp.exp2, varName)
  local op = exp.op
  if op === OP_ADD || op === OP_SUB
    return (o1 || o2, l1 && l2)
  elseif op === OP_MUL || op === OP_DOT
    return (o1 || o2, l1 && l2 && !(o1 && o2))
  elseif op === OP_DIV
    return (o1 || o2, l1 && !o2)
  elseif op === OP_POW
    return _powLinearity(exp.exp2, o1, l1, o2)
  else
    return (o1 || o2, !(o1 || o2))
  end
end

"Occurrence/linearity of `varName` across a SIM `IFEXP` (var in cond ⇒ nonlinear)."
function _ifexpLinearity(exp::IFEXP, varName::AbstractString)::Tuple{Bool, Bool}
  local (oc, _) = _occursLinearly(exp.cond, varName)
  oc && return (true, false)
  local (ot, lt) = _occursLinearly(exp.thenExp, varName)
  local (oe, le) = _occursLinearly(exp.elseExp, varName)
  return (ot || oe, lt && le)
end

"""
    _occursLinearly(exp::Exp, varName::AbstractString) -> (occurs::Bool, linear::Bool)

Whether `varName` appears in `exp`, and if so only affinely (degree ≤ 1, never
inside a nonlinear operator or function argument). Conservative: any construct
whose linearity cannot be established yields `linear = false`, which keeps the
variable in the residual system (always semantically valid).
"""
function _occursLinearly(exp::Exp, varName::AbstractString)::Tuple{Bool, Bool}
  local nm = _simCrefScalarName(exp)
  nm === nothing || return (nm == varName, true)
  @match exp begin
    UNARY(__) || CAST(__) => _occursLinearly(exp.exp, varName)
    BINARY(__) => _binaryLinearity(exp, varName)
    IFEXP(__) => _ifexpLinearity(exp, varName)
    ICONST(__) || RCONST(__) || BCONST(__) || SCONST(__) || ENUM_LITERAL(__) || WILD(__) =>
      (false, true)
    _ => begin
      local occurs = _occursAnywhere(exp, varName)
      (occurs, !occurs)
    end
  end
end

"""
    _isLinearlySolvableFor(exp::Exp, varName::AbstractString) -> Bool

`true` iff `varName` appears in residual `exp` and only affinely, so
`Symbolics.solve_for(0 ~ exp, varName)` yields a valid explicit observation.
An output-only sink variable failing this test (e.g. defined by a nonlinear
closure) must stay in the residual system for MTK to solve numerically.
"""
function _isLinearlySolvableFor(exp::Exp, varName::AbstractString)::Bool
  local (occurs, linear) = _occursLinearly(exp, varName)
  return occurs && linear
end

"""
    eliminateOutputOnlyVariables(simCode::SIM_CODE, options::EliminationOptions)

Remove output-only variables and their defining equations from the SimCode.
Rebuilds a fresh bipartite matching from the current (post-optimization) equation
and variable sets, then performs backward reachability to identify output-only
equation-variable pairs. Only eliminates ALG_VARIABLE or ARRAY unknowns,
preserving the equation-unknown balance that MTK requires.

The eliminated equations and variable names are stored in `simCode.eliminatedEquations`
and `simCode.eliminatedVariables` for later reconstruction (e.g. 3D visualization).

Returns the modified SIM_CODE (uses @assign for immutable struct mutation).
"""
function eliminateOutputOnlyVariables(simCode::SIM_CODE, options::EliminationOptions)
  #= Guard: skip for VSS/multi-mode models (subModels or a recompilation-based
     metaModel); structural transitions alone are allowed. =#
  if hasSubModels(simCode) || hasMetaModel(simCode)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] skipping for VSS/multi-mode model"
    return simCode
  end
  #= Rebuild a fresh matching from the current (post-optimization) system =#
  local (matchOrder, nameToMatchIdx, matchIdxToName) = rebuildMatchOrder(simCode)
  if isempty(matchOrder)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] matching failed or system not square, skipping"
    return simCode
  end
  #= Identify output-only equations and variables using the fresh matching =#
  local (outputOnlyVarNames, outputOnlyEqIndices, eqRefs) =
    identifyOutputOnlyVariables(simCode, matchOrder, matchIdxToName)
  if isempty(outputOnlyEqIndices)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] no output-only equations found"
    return simCode
  end
  #= Build inverse matching: equation index -> match index =#
  local ht = simCode.stringToSimVarHT
  local eqToMatchIdx = Dict{Int, Int}()
  for (mIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      eqToMatchIdx[eqIdx] = mIdx
    end
  end
  #= Only eliminate output-only equation-variable PAIRS where the matched variable
     is ALG_VARIABLE or ARRAY. This preserves equation-unknown balance. =#
  local eqsToEliminate = OrderedSet{Int}()
  local varsToRemove = OrderedSet{String}()
  local eliminatedPairs = Tuple{String, Int}[]  #= (varName, eqIdx) for pairing =#
  local nSkippedNonAlg = 0
  local nSkippedUnmatched = 0
  local nSkippedNonlinear = 0
  for eqIdx in outputOnlyEqIndices
    if !haskey(eqToMatchIdx, eqIdx)
      nSkippedUnmatched += 1
      continue
    end
    local mIdx = eqToMatchIdx[eqIdx]
    if !haskey(matchIdxToName, mIdx)
      nSkippedUnmatched += 1
      continue
    end
    local vn = matchIdxToName[mIdx]
    if !haskey(ht, vn)
      nSkippedUnmatched += 1
      continue
    end
    local (_, sv) = ht[vn]
    local isEliminable = @match sv.varKind begin
      ALG_VARIABLE(__) => true
      SimulationCode.ARRAY(__) => true
      _ => false
    end
    if isEliminable
      #= The eliminated pair is later reconstructed via Symbolics.solve_for, a
         linear solver. A variable defined by an equation nonlinear in itself
         (e.g. a holonomic loop closure) must stay in the residual system for
         MTK to solve numerically; eliminating it would trip `islinear`. =#
      if !_isLinearlySolvableFor(simCode.residualEquations[eqIdx].exp, vn)
        nSkippedNonlinear += 1
        continue
      end
      push!(eqsToEliminate, eqIdx)
      push!(varsToRemove, vn)
      push!(eliminatedPairs, (vn, eqIdx))
    else
      nSkippedNonAlg += 1
    end
  end
  if isempty(eqsToEliminate)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] no eliminable equation-variable pairs found"
    return simCode
  end
  #= Guard: never eliminate ALL equations. A system with zero equations
     after elimination would crash downstream (filterConstantEquations, MTK). =#
  if length(eqsToEliminate) >= length(simCode.residualEquations)
    @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] would eliminate all $(length(simCode.residualEquations)) equations, skipping"
    return simCode
  end
  #= Safety check: verify no surviving equation references an eliminated variable.
     Build reverse index: variable name -> equations that reference it. =#
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  local varNameToRefEqs = Dict{String, OrderedSet{Int}}()
  for eqIdx in 1:nEqs
    for refName in eqRefs[eqIdx]
      if !haskey(varNameToRefEqs, refName)
        varNameToRefEqs[refName] = OrderedSet{Int}()
      end
      push!(varNameToRefEqs[refName], eqIdx)
    end
  end
  #= Collect variable names referenced by when-equations so they are never eliminated.
     When-equations live outside the residual system and are not in varNameToRefEqs. =#
  local whenRefNames = OrderedSet{String}()
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(whenRefNames, whenEq.whenEquation)
  end
  #= Collect scalar `_re`/`_im` siblings of any Complex CREF that survives in
     residuals, initial equations, if-equation branches, when-equations, or
     eliminated equations. Codegen later flattens the parent Complex CREF into
     its two scalar fields and looks them up by symbol; if either field is
     dropped here we hit `UndefVarError` at MTK module eval. =#
  local complexRefNames = OrderedSet{String}()
  _collectComplexFieldNames!(complexRefNames, simCode.residualEquations, ht)
  _collectComplexFieldNames!(complexRefNames, simCode.initialEquations, ht)
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      _collectComplexFieldNames!(complexRefNames, branch.residualEquations, ht)
    end
  end
  for eq in simCode.eliminatedEquations
    _collectComplexFieldNames!(complexRefNames, [eq], ht)
  end
  local rescuedVars = OrderedSet{String}()
  for vn in varsToRemove
    local referencedBySurvivor = false
    #= Check residual equations =#
    if haskey(varNameToRefEqs, vn)
      for refEqIdx in varNameToRefEqs[vn]
        if !(refEqIdx in eqsToEliminate)
          referencedBySurvivor = true
          break
        end
      end
    end
    #= Also check base name =#
    if !referencedBySurvivor
      local bi = findfirst('[', vn)
      local bn = bi === nothing ? vn : vn[1:(bi - 1)]
      if bn != vn && haskey(varNameToRefEqs, bn)
        for refEqIdx in varNameToRefEqs[bn]
          if !(refEqIdx in eqsToEliminate)
            referencedBySurvivor = true
            break
          end
        end
      end
    end
    #= Check when-equations =#
    if !referencedBySurvivor && vn in whenRefNames
      referencedBySurvivor = true
    end
    #= Check Complex `_re`/`_im` parent survival =#
    if !referencedBySurvivor && vn in complexRefNames
      referencedBySurvivor = true
    end
    if referencedBySurvivor
      push!(rescuedVars, vn)
    end
    #= Rescue variables carrying `fixed=true` with an explicit start value.
       These are user-pinned initial conditions (e.g. `wMechanical(fixed=true,
       start=w0)`); eliminating them strips the constraint and MTK's init
       solver lands on the algebraic default (typically 0). DCPM_Cooling,
       DCPM_QuasiStationary, DCPM_withLosses regress on this exact pattern. =#
    if !referencedBySurvivor && haskey(ht, vn)
      local (_, _sv) = ht[vn]
      if _hasExplicitFixedStart(_sv.attributes)
        push!(rescuedVars, vn)
      end
    end
  end
  local nRescued = length(rescuedVars)
  if !isempty(rescuedVars)
    for vn in rescuedVars
      delete!(varsToRemove, vn)
      if haskey(nameToMatchIdx, vn)
        local rescuedMIdx = nameToMatchIdx[vn]
        local rescuedEqIdx = matchOrder[rescuedMIdx]
        if rescuedEqIdx > 0
          delete!(eqsToEliminate, rescuedEqIdx)
        end
      end
    end
  end
  #= Filter residualEquations: remove eliminated equations =#
  local newResEqs = RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(eqsToEliminate))
  for (i, eq) in enumerate(resEqs)
    if !(i in eqsToEliminate)
      push!(newResEqs, eq)
    end
  end
  #= Build parallel (varName, equation) vectors from the paired data.
     Filter out rescued variables. =#
  local survivingPairs = filter(p -> !(p[1] in rescuedVars), eliminatedPairs)
  local elimPairedVars = String[p[1] for p in survivingPairs]
  local elimPairedEqs = RESIDUAL_EQUATION[resEqs[p[2]] for p in survivingPairs]
  #= Filter stringToSimVarHT: remove eliminated variables =#
  local newHT = copy(ht)
  for varName in varsToRemove
    delete!(newHT, varName)
  end
  @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] eliminated $(length(eqsToEliminate)) eq-var pairs, $(length(varsToRemove)) variables removed (rescued: $nRescued, skipped: $nSkippedNonAlg non-algebraic, $nSkippedNonlinear nonlinear, $nSkippedUnmatched unmatched). $(length(newResEqs)) equations, $(length(newHT)) variables remain"
  @BACKEND_LOGGING begin
    local buf = IOBuffer()
    println(buf, "=== ELIMINATION DEBUG ===")
    println(buf, "Removed variables ($(length(varsToRemove))):")
    for vn in sort(collect(varsToRemove))
      println(buf, "  ", vn)
    end
    println(buf, "Rescued variables ($nRescued):")
    for vn in sort(collect(rescuedVars))
      println(buf, "  ", vn)
    end
    println(buf, "Eliminated equation indices: ", sort(collect(eqsToEliminate)))
    println(buf, "=== END DEBUG ===")
    OMBackend.debugWrite(OMBackend.logPath("backend/simCode", "elimination_debug.log"), String(take!(buf)))
  end
  @assign begin
    simCode.residualEquations = newResEqs
    simCode.stringToSimVarHT = newHT
    simCode.eliminatedEquations = elimPairedEqs
    simCode.eliminatedVariables = elimPairedVars
  end
  return simCode
end
