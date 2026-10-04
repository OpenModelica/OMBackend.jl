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


"""
    buildBaseNameIndex(ht::OrderedDict{String, Tuple{Int, SimVar}})

Build a reverse index from base variable names (without subscripts) to all
subscripted full names in the hash table. For example, if the HT contains
"world_x[1]" and "world_x[2]", the result maps "world_x" => ["world_x[1]", "world_x[2]"].
This handles the ASUB case where `getAllCrefs` extracts a base CREF without subscripts.
"""
function buildBaseNameIndex(ht::OrderedDict{String, Tuple{Int, SimVar}})::Dict{String, Vector{String}}
  local index = Dict{String, Vector{String}}()
  for (varName, _) in ht
    local bi = findfirst('[', varName)
    local bn = bi === nothing ? varName : varName[1:(bi - 1)]
    if bn != varName
      if !haskey(index, bn)
        index[bn] = String[]
      end
      push!(index[bn], varName)
    end
  end
  return index
end

"""
    collectEquationVarNames(exp::DAE.Exp,
                            ht::OrderedDict{String, Tuple{Int, SimVar}},
                            baseNameToFullNames::Dict{String, Vector{String}})

Extract all variable names referenced by a DAE expression, using the robust
`Util.getAllCrefs` traversal (via `traverseExpTopDown`). Falls back to base-name
matching for ASUB-wrapped CREFs where subscripts are separated from the CREF.

Returns a OrderedSet{String} of variable names that exist in the HT.
"""
function collectEquationVarNames(exp::DAE.Exp,
                                 ht::OrderedDict{String, Tuple{Int, SimVar}},
                                 baseNameToFullNames::Dict{String, Vector{String}})::OrderedSet{String}
  local crefs::List{DAE.ComponentRef} = Util.getAllCrefs(exp)
  local names = OrderedSet{String}()
  for cr in crefs
    local name = DAE_identifierToString(cr)
    if haskey(ht, name)
      push!(names, name)
    else
      #= Base name fallback: the CREF may come from inside an ASUB expression,
         missing its subscripts. Match all subscripted variants conservatively. =#
      local bi = findfirst('[', name)
      local bn = bi === nothing ? name : name[1:(bi - 1)]
      if bn != name && haskey(ht, bn)
        #= The CREF itself has partial subscripts; try the full name and base =#
        push!(names, bn)
      end
      local lookupKey = haskey(baseNameToFullNames, name) ? name : bn
      if haskey(baseNameToFullNames, lookupKey)
        for fullName in baseNameToFullNames[lookupKey]
          push!(names, fullName)
        end
      end
    end
  end
  return names
end

"""
    rebuildMatchOrder(simCode::SIM_CODE)

Rebuild a fresh bipartite matching from the current equations and variables.
This is needed when the original matchOrder is stale (e.g. after const-prop
and alias-elim have removed equations and variables).

Returns `(matchOrder::Vector{Int}, nameToMatchIdx::Dict{String,Int}, matchIdxToName::Dict{Int,String})`
where `matchOrder[varMatchIdx] = eqIdx` (0 = unmatched).
"""
function rebuildMatchOrder(simCode::SIM_CODE)
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  #= Collect unknown variables (those that participate in matching) =#
  local nameToMatchIdx = Dict{String, Int}()
  local matchIdxToName = Dict{Int, String}()
  local matchIdx = 0
  for (varName, (_idx, sv)) in ht
    local isUnknown = @match sv.varKind begin
      STATE(__) => true
      STATE_DERIVATIVE(__) => true
      ALG_VARIABLE(__) => true
      SimulationCode.ARRAY(__) => true
      DISCRETE(__) => true
      _ => false
    end
    if isUnknown
      matchIdx += 1
      nameToMatchIdx[varName] = matchIdx
      matchIdxToName[matchIdx] = varName
    end
  end
  local nVars = matchIdx
  #= Build the base name index for robust CREF extraction =#
  local baseNameToFullNames = buildBaseNameIndex(ht)
  #= Build bipartite adjacency: for each equation, which variable match indices does it reference? =#
  #= Int-keyed: GraphAlgorithms.matching consumes only `.vals` positionally, so the
     interpolated "e$(i)" string keys were pure allocation/hashing overhead. =#
  local eqVarMapping = DataStructures.OrderedDict{Int, Vector{Int}}()
  for eqI in 1:nEqs
    local refs = collectEquationVarNames(toDAEExp(resEqs[eqI].exp), ht, baseNameToFullNames)
    local indices = Int[]
    for refName in refs
      if haskey(nameToMatchIdx, refName)
        push!(indices, nameToMatchIdx[refName])
      end
    end
    eqVarMapping[eqI] = sort(unique(indices))
  end
  #= The matching algorithm requires a square system (n used for both eq loop
     and assign array). For over-determined systems (nVars > nEqs), pad with
     dummy empty equations so the algorithm sees a square system. The dummy
     equations will remain unmatched. For under-determined systems (nEqs > nVars),
     skip since we cannot produce a valid matching. =#
  if nEqs > nVars
    @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] under-determined system ($nEqs equations, $nVars unknowns), skipping"
    return (Int[], nameToMatchIdx, matchIdxToName)
  end
  local nMatch = nVars
  if nVars > nEqs
    for dummyI in (nEqs + 1):nVars
      eqVarMapping[dummyI] = Int[]
    end
  end
  local matchOrder::Vector{Int}
  try
    local (_isSingular, mo) = GraphAlgorithms.matching(eqVarMapping, nMatch)
    matchOrder = mo
  catch e
    #= Matching failed (a large system can overflow the recursive matching's
       stack): no dead-code elimination. =#
    e isa StackOverflowError || OMBackend._fallback(e, :outputOnlyMatching)
    return (Int[], nameToMatchIdx, matchIdxToName)
  end
  local nMatched = count(>(0), matchOrder)
  @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] $nEqs equations, $nVars unknowns, $nMatched matched"
  return (matchOrder, nameToMatchIdx, matchIdxToName)
end

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

#= Whether an eliminated variable `vn` is still read: by a residual equation
   that stays (or one reading its array's base name), a when-equation, an
   initial equation, or a Complex parent that stays (its `_re`/`_im` fields,
   which codegen looks up by symbol). The variables carrying `fixed = true`
   (their start, or the default one) are rescued as well (by the caller):
   user-pinned initial conditions (DCPM_Cooling's `wMechanical(fixed = true,
   start = w0)`), which elimination stripped (MTK's init then landed on 0). =#
function _referencedBySurvivor(vn::String, varNameToRefEqs, eqsToEliminate, whenRefNames, initRefNames,
                               complexRefNames)::Bool
  local bi = findfirst('[', vn)
  local baseName = bi === nothing ? vn : vn[1:(bi - 1)]
  for name in (baseName == vn ? (vn,) : (vn, baseName))
    haskey(varNameToRefEqs, name) || continue
    any(i -> !(i in eqsToEliminate), varNameToRefEqs[name]) && return true
  end
  return vn in whenRefNames || vn in initRefNames || vn in complexRefNames
end

#= True when the variable's attributes carry `fixed = true`: its start, or the
   default start 0 without one (MLS 4.9.1), is an initial equation. Used to
   rescue variables from elimination passes that would otherwise drop the
   user-pinned initial condition (`v(fixed = true)` with `v = xa + 1` was
   folded away: xa(0) = 0, OpenModelica -1). =#
function _hasFixedStart(@nospecialize(attrs))::Bool
  return @match attrs begin
    SOME(va) where (va isa DAE.VAR_ATTR_REAL) => @match va.fixed begin
      SOME(DAE.BCONST(true)) => true
      _ => false
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
  #= Variables an initial equation reads are rescued like when-equation
     reads: eliminated, `initial equation z = b` (b = a + 1, a = 2x) read an
     undefined b (UndefVarError at module eval; OpenModelica z = 3). =#
  local initRefNames = OrderedSet{String}()
  for ieq in simCode.initialEquations
    if ieq isa BDAE.RESIDUAL_EQUATION || ieq isa RESIDUAL_EQUATION
      collectCrefNames!(initRefNames, ieq.exp)
    elseif ieq isa BDAE.EQUATION || ieq isa EQUATION
      collectCrefNames!(initRefNames, ieq.lhs)
      collectCrefNames!(initRefNames, ieq.rhs)
    end
  end
  #= Until nothing more is rescued: a rescued variable keeps its equation,
     and what that equation reads must stay as well (b's `b = a + 1` reads a). =#
  local rescuedVars = OrderedSet{String}()
  local rescuedMore = true
  while rescuedMore
    rescuedMore = false
    for vn in varsToRemove
      vn in rescuedVars && continue
      if _referencedBySurvivor(vn, varNameToRefEqs, eqsToEliminate, whenRefNames, initRefNames, complexRefNames) ||
         (haskey(ht, vn) && _hasFixedStart(ht[vn][2].attributes))
        push!(rescuedVars, vn)
        rescuedMore = true
        if haskey(nameToMatchIdx, vn)
          local rescuedEqIdx = matchOrder[nameToMatchIdx[vn]]
          rescuedEqIdx > 0 && delete!(eqsToEliminate, rescuedEqIdx)
        end
      end
    end
  end
  local nRescued = length(rescuedVars)
  setdiff!(varsToRemove, rescuedVars)
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
  #= Added to the pairs earlier passes eliminated (the explicit fold, frozen
     states): replaced, those variables lost their observed equations and
     could not be read from the solution (codegen orders the observed
     equations by what they read: _eliminatedDependencyOrder). =#
  @assign begin
    simCode.residualEquations = newResEqs
    simCode.stringToSimVarHT = newHT
    simCode.eliminatedEquations = vcat(simCode.eliminatedEquations, elimPairedEqs)
    simCode.eliminatedVariables = vcat(simCode.eliminatedVariables, elimPairedVars)
  end
  return simCode
end
