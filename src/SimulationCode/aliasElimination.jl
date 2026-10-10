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

#= Alias elimination (union-find over alias equations), alias substitution, RHS-equivalent equations. =#

"""
    buildAsubName(baseName::String, subs::Vector)::String

Reconstruct a subscripted variable name from an ASUB expression.
Turns base name "a" with subscripts [1, 2] into "a[1][2]" to match hash table keys.
"""
function buildAsubName(baseName::String, subs)::String
  buf = baseName
  for s in subs
    @match s begin
      DAE.INDEX(DAE.ICONST(i)) => begin buf *= Base.string("[", i, "]") end
      _ => return ""  #= Non-constant subscript: cannot resolve statically =#
    end
  end
  return buf
end

"""
    varKindPriority(varKind::SimVarType)::Int

Return priority of a variable kind for alias representative selection.
Higher priority variables are preferred as representatives (never eliminated).
"""
function varKindPriority(@nospecialize(varKind::SimVarType))::Int
  @match varKind begin
    STATE(__) => 100
    STATE_DERIVATIVE(__) => 90
    DISCRETE(__) => 80
    ALG_VARIABLE(__) => 20
    ARRAY(__) => 10
    _ => 0
  end
end

"""
    isRealValued(ty::DAE.Type)::Bool

Check if a DAE type represents a Real-valued (floating point) variable.
Only Real-valued variables are eligible for alias elimination.
"""
function isRealValued(@nospecialize(ty))::Bool
  @match ty begin
    DAE.T_REAL(__) => true
    DAE.T_ARRAY(ty = innerTy) => isRealValued(innerTy)
    _ => false
  end
end

#= Same-class check for alias eligibility: Real, Boolean, Integer, Enumeration. =#
function _aliasTypeClass(@nospecialize(ty))::Symbol
  @match ty begin
    DAE.T_REAL(__) => :real
    DAE.T_BOOL(__) => :bool
    DAE.T_INTEGER(__) => :int
    DAE.T_ENUMERATION(__) => :enum
    DAE.T_ARRAY(ty = innerTy) => _aliasTypeClass(innerTy)
    _ => :other
  end
end

"""
Union-find: find with path compression.
"""
function _ufFind!(parent::Dict{String,String}, x::String)::String
  if !haskey(parent, x)
    parent[x] = x
  end
  while parent[x] != x
    parent[x] = parent[parent[x]]
    x = parent[x]
  end
  return x
end

"""
Union-find: union two elements. Returns true if they were in different sets (merged),
false if already in the same set (redundant).
"""
function _ufUnion!(parent::Dict{String,String}, a::String, b::String)::Bool
  local ra = _ufFind!(parent, a)
  local rb = _ufFind!(parent, b)
  if ra != rb
    parent[ra] = rb
    return true
  end
  return false
end

#= Wrap a DAE expression in a numeric negation, folding the trivial constant
   cases so the representative gets a clean literal instead of an UMINUS tree.
   Used when transferring start/min/max/nominal from an alias variable that is
   the negated side of an `a + b = 0` pairing. =#
function _negateAliasExp(e::DAE.Exp)::DAE.Exp
  @match e begin
    DAE.RCONST(v) => DAE.RCONST(-v)
    DAE.ICONST(v) => DAE.ICONST(-v)
    DAE.UNARY(DAE.UMINUS(__), inner) => inner
    DAE.UNARY(DAE.UMINUS_ARR(__), inner) => inner
    _ => DAE.UNARY(DAE.UMINUS(DAE.T_REAL_DEFAULT), e)
  end
end

_negateOptExp(opt) = @match opt begin
  SOME(e) => SOME(_negateAliasExp(e))
  _       => opt
end

#= Whether an optional `fixed` attribute is literally true. =#
_fixedTrue(opt) = @match opt begin
  SOME(DAE.BCONST(true)) => true
  _ => false
end

#= Prefer rep's value when present; otherwise take the elim's. =#
_orElseOpt(repField, elimField) = @match repField begin
  SOME(_) => repField
  _       => elimField
end

#= The same for an Integer, Boolean or enumeration alias: its start and fixed
   (with min and max). They were dropped: `k(start = 3, fixed = true)` with
   `k2 = k` read 0 until k's first event (OpenModelica 3). A negated Boolean
   alias takes `not start`; an enumeration cannot be negated. =#
function _mergeDiscreteAliasAttrs(repAttr, elimVA, negated::Bool)
  elimVA isa Union{DAE.VAR_ATTR_INT, DAE.VAR_ATTR_BOOL, DAE.VAR_ATTR_ENUMERATION} || return repAttr
  negated && elimVA isa DAE.VAR_ATTR_ENUMERATION && return repAttr
  local base = @match repAttr begin
    SOME(va) where (typeof(va) == typeof(elimVA)) => va
    _ => elimVA
  end
  local hasRep = base !== elimVA
  local elimStart = !negated ? elimVA.start : elimVA isa DAE.VAR_ATTR_BOOL ? _notOptExp(elimVA.start) : _negateOptExp(elimVA.start)
  local elimFixedStart = _fixedTrue(elimVA.fixed) && !(hasRep && _fixedTrue(base.fixed))
  local start = elimFixedStart ? elimStart : hasRep ? _orElseOpt(base.start, elimStart) : elimStart
  local fixed = elimFixedStart ? elimVA.fixed : hasRep ? _orElseOpt(base.fixed, elimVA.fixed) : elimVA.fixed
  base isa DAE.VAR_ATTR_BOOL &&
    return SOME(DAE.VAR_ATTR_BOOL(base.quantity, start, fixed, base.equationBound, base.isProtected, base.finalPrefix,
                                  base.startOrigin))
  local (elimMin, elimMax) = negated ? (_negateOptExp(elimVA.max), _negateOptExp(elimVA.min)) : (elimVA.min, elimVA.max)
  local min = hasRep ? _orElseOpt(base.min, elimMin) : elimMin
  local max = hasRep ? _orElseOpt(base.max, elimMax) : elimMax
  base isa DAE.VAR_ATTR_INT &&
    return SOME(DAE.VAR_ATTR_INT(base.quantity, min, max, start, fixed, base.uncertainOption, base.distributionOption,
                                 base.equationBound, base.isProtected, base.finalPrefix, base.startOrigin))
  return SOME(DAE.VAR_ATTR_ENUMERATION(base.quantity, min, max, start, fixed, base.equationBound, base.isProtected,
                                      base.finalPrefix, base.startOrigin))
end

_notOptExp(opt) = @match opt begin
  SOME(DAE.BCONST(b)) => SOME(DAE.BCONST(!b))
  SOME(e) => SOME(DAE.LUNARY(DAE.NOT(DAE.T_BOOL_DEFAULT), e))
  _ => opt
end

"""
    _mergeAliasAttrs(repAttr, elimAttr, negated)

Lift NONE-valued fields of the representative's variable attributes from the
eliminated alias. Mirrors OMC `BackendVariable.mergeAliasVars`: the alias's
`start`, `fixed`, `nominal`, `min`, `max`, `stateSelectOption` etc. fill the
gaps left when the representative was chosen for its varKind (e.g. STATE) but
the user-supplied start/fixed lived on the alias (e.g. ALG_VARIABLE
`body2.r_0`). On a negated pairing (`a + b = 0`) `start`/`nominal` flip sign
and `min`/`max` swap-and-flip.

`start` and `fixed` go together (OMC `mergeStartFixed`): a fixed alias start
replaces a representative's free one, set or not. When both are fixed the
representative's start is kept.

Only `VAR_ATTR_REAL` is handled — `VAR_ATTR_INT` / `VAR_ATTR_BOOL` pass
through, since the Real path covers the dynamic-state IC residual cases.
"""
function _mergeAliasAttrs(repAttr, elimAttr, negated::Bool)
  local elimVA = @match elimAttr begin
    SOME(va) => va
    _        => nothing
  end
  elimVA === nothing && return repAttr
  isa(elimVA, DAE.VAR_ATTR_REAL) || return _mergeDiscreteAliasAttrs(repAttr, elimVA, negated)
  local elimStart   = elimVA.start
  local elimMin     = elimVA.min
  local elimMax     = elimVA.max
  local elimNominal = elimVA.nominal
  if negated
    elimStart   = _negateOptExp(elimStart)
    elimNominal = _negateOptExp(elimNominal)
    local newMin = _negateOptExp(elimMax)
    local newMax = _negateOptExp(elimMin)
    elimMin = newMin
    elimMax = newMax
  end
  local baseRep = @match repAttr begin
    SOME(va) where (va isa DAE.VAR_ATTR_REAL) => va
    _ => DAE.emptyVarAttrReal
  end
  #= start and fixed go together (OMC mergeStartFixed). =#
  local elimFixedStart = _fixedTrue(elimVA.fixed) && !_fixedTrue(baseRep.fixed)
  local merged = DAE.VAR_ATTR_REAL(
    _orElseOpt(baseRep.quantity,             elimVA.quantity),
    _orElseOpt(baseRep.unit,                 elimVA.unit),
    _orElseOpt(baseRep.displayUnit,          elimVA.displayUnit),
    _orElseOpt(baseRep.min,                  elimMin),
    _orElseOpt(baseRep.max,                  elimMax),
    elimFixedStart ? elimStart : _orElseOpt(baseRep.start, elimStart),
    elimFixedStart ? elimVA.fixed : _orElseOpt(baseRep.fixed, elimVA.fixed),
    _orElseOpt(baseRep.nominal,              elimNominal),
    _orElseOpt(baseRep.stateSelectOption,    elimVA.stateSelectOption),
    _orElseOpt(baseRep.uncertainOption,      elimVA.uncertainOption),
    _orElseOpt(baseRep.distributionOption,   elimVA.distributionOption),
    _orElseOpt(baseRep.equationBound,        elimVA.equationBound),
    _orElseOpt(baseRep.isProtected,          elimVA.isProtected),
    _orElseOpt(baseRep.finalPrefix,          elimVA.finalPrefix),
    _orElseOpt(baseRep.startOrigin,          elimVA.startOrigin),
  )
  return SOME(merged)
end

"""
    eliminateAliasVariables(simCode::SIM_CODE)::SIM_CODE

Perform alias elimination on the simulation code. Detects equations of the form
`a - b = 0` (alias) or `a + b = 0` (negated alias), builds connected components
of alias relationships, selects a representative per component, and substitutes
all eliminated variables with their representative in all equations.

This pass always runs (not opt-in) and preserves equation-unknown balance because
each eliminated equation removes exactly one variable.

Skipped for VSS/structural models where eliminated variables might be needed
in different structural modes.
"""
function eliminateAliasVariables(simCode::SIM_CODE)
  #= Guard: skip for VSS/multi-mode models (subModels or a recompilation-based
     metaModel); structural transitions alone are allowed. =#
  if hasSubModels(simCode) || hasMetaModel(simCode)
    @debug "[SIMCODE: $(simCode.name): aliasElimination] skipped (VSS/multi-mode model)"
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  local sharedVarSet = OrderedSet{String}(simCode.sharedVariables)
  local irreducibleSet = OrderedSet{String}(simCode.irreducibleVariables)

  #= ===== Step 1: Detect alias equations ===== =#
  #= Each alias is (name1, name2, negated, eqIdx, cref1, ty1, cref2, ty2) =#
  local aliasPairs = Tuple{String, String, Bool, Int, DAE.ComponentRef, DAE.Type, DAE.ComponentRef, DAE.Type}[]

  for (i, eq) in enumerate(resEqs)
    local pair = detectAlias(toDAEExp(eq.exp), ht)
    if pair !== nothing
      local (n1, n2, neg, cr1, t1, cr2, t2) = pair
      #= Skip self-loops =#
      if n1 == n2
        continue
      end
      #= Skip shared variables =#
      if n1 in sharedVarSet || n2 in sharedVarSet
        continue
      end
      push!(aliasPairs, (n1, n2, neg, i, cr1, t1, cr2, t2))
    end
  end

  if isempty(aliasPairs)
    @debug "[SIMCODE: $(simCode.name): aliasElimination] no alias equations found"
    return simCode
  end

  @debug "[SIMCODE: $(simCode.name): aliasElimination] detected $(length(aliasPairs)) alias equations"

  #= ===== Step 2: Build alias graph and find connected components via BFS ===== =#
  #= Adjacency list: varName -> [(neighborName, negated, edgeIdx)] =#
  local adjList = OrderedDict{String, Vector{Tuple{String, Bool, Int}}}()
  for (idx, (n1, n2, neg, eqIdx, _, _, _, _)) in enumerate(aliasPairs)
    if !haskey(adjList, n1)
      adjList[n1] = Tuple{String, Bool, Int}[]
    end
    if !haskey(adjList, n2)
      adjList[n2] = Tuple{String, Bool, Int}[]
    end
    push!(adjList[n1], (n2, neg, idx))
    push!(adjList[n2], (n1, neg, idx))
  end

  #= BFS to find connected components with cumulative negation =#
  #= componentId -> [(varName, negationRelativeToRoot)] =#
  local visited = Dict{String, Bool}()  #= varName -> negation relative to component root =#
  local components = Vector{Vector{Tuple{String, Bool}}}()
  local componentEqs = Vector{Vector{Int}}()  #= equation indices per component =#
  local usedEdges = OrderedSet{Int}()

  for startNode in keys(adjList)
    if haskey(visited, startNode)
      continue
    end
    local component = Tuple{String, Bool}[]
    local compEqs = Int[]
    local queue = [(startNode, false)]  #= (name, negRelToRoot) =#
    visited[startNode] = false
    while !isempty(queue)
      local (node, negFromRoot) = popfirst!(queue)
      push!(component, (node, negFromRoot))
      if haskey(adjList, node)
        for (neighbor, edgeNeg, edgeIdx) in adjList[node]
          if !(edgeIdx in usedEdges)
            push!(usedEdges, edgeIdx)
            push!(compEqs, aliasPairs[edgeIdx][4])  #= equation index =#
          end
          if !haskey(visited, neighbor)
            local neighborNeg = xor(negFromRoot, edgeNeg)
            visited[neighbor] = neighborNeg
            push!(queue, (neighbor, neighborNeg))
          end
        end
      end
    end
    push!(components, component)
    push!(componentEqs, compEqs)
  end

  #= ===== Step 3: Select representative per component ===== =#
  #= Build alias resolution map and alias entries =#
  local aliasMap = OrderedDict{String, Tuple{String, Bool, DAE.ComponentRef, DAE.Type}}()
  local aliasEntries = AliasEntry[]
  local aliasEqIndices = OrderedSet{Int}()
  #= Pending attribute lifts: rep name -> merged Option{VariableAttributes}.
     Built incrementally as each alias is folded into its representative so a
     user-supplied start / fixed / stateSelect on the eliminated side ends up
     on the surviving variable. Applied to newHT below. =#
  local repAttrUpdates = Dict{String, Any}()

  #= Build name -> (cref, type) lookup from alias pairs =#
  local nameToCrefType = Dict{String, Tuple{DAE.ComponentRef, DAE.Type}}()
  for (n1, n2, _, _, cr1, t1, cr2, t2) in aliasPairs
    nameToCrefType[n1] = (cr1, t1)
    nameToCrefType[n2] = (cr2, t2)
  end

  #= Map equation index -> (n1, n2) for deciding which equations are trivial after substitution =#
  local eqIdxToNames = Dict{Int, Tuple{String, String}}()
  for (n1, n2, _, eqIdx, _, _, _, _) in aliasPairs
    eqIdxToNames[eqIdx] = (n1, n2)
  end

  for (compIdx, component) in enumerate(components)
    #= Select representative: highest priority varKind, with ties broken by irreducibility =#
    local bestName = ""
    local bestPriority = -1
    local bestNeg = false
    for (varName, negFromRoot) in component
      if !haskey(ht, varName)
        continue
      end
      local (_, sv) = ht[varName]
      local prio = varKindPriority(sv.varKind)
      #= Boost priority for irreducible variables =#
      if varName in irreducibleSet
        prio += 60
      end
      #= Boost priority for variables with explicit start attribute so the
         representative carries the start binding instead of defaulting to 0. =#
      local hasStart = @match sv.attributes begin
        SOME(DAE.VAR_ATTR_REAL(start = SOME(_))) => true
        SOME(DAE.VAR_ATTR_INT(start = SOME(_)))  => true
        SOME(DAE.VAR_ATTR_BOOL(start = SOME(_))) => true
        _                                        => false
      end
      if hasStart
        prio += 5
      end
      if prio > bestPriority
        bestPriority = prio
        bestName = varName
        bestNeg = negFromRoot
      end
    end

    if isempty(bestName)
      continue
    end

    #= Get representative CREF and type =#
    if !haskey(nameToCrefType, bestName)
      continue
    end
    local (repCref, repTy) = nameToCrefType[bestName]
    local (_, bestSv) = ht[bestName]
    local bestIsState = @match bestSv.varKind begin
      STATE(__) => true
      _ => false
    end

    #= Mark all other variables in this component for elimination.
       Never eliminate irreducible variables (involved in events).
       Exception: state-to-state aliases inside the same component are safe to
       collapse even when both ends are flagged irreducible — `getIrreducibleVars`
       marks the states that when-equations use as irreducible, which would prevent two states that
       are connected via algebraic-flange aliases (e.g. AIMC `aimc_inertiaRotor_phi`
       and `loadInertia_phi`) from being merged. Without merging, the residual
       `loadInertia_phi - aimc_inertiaRotor_phi = 0` survives and MTK Pantelides
       sees the system as over-determined. =#
    for (varName, negFromRoot) in component
      if varName == bestName
        continue
      end
      if !haskey(ht, varName)
        continue
      end
      local (_, sv) = ht[varName]
      local isState = @match sv.varKind begin
        STATE(__) => true
        _ => false
      end
      if varName in irreducibleSet && !(bestIsState && isState)
        continue
      end
      local negated = xor(negFromRoot, bestNeg)
      aliasMap[varName] = (bestName, negated, repCref, repTy)
      push!(aliasEntries, AliasEntry(varName, bestName, negated))
      #= Merge eliminated alias attributes into the representative's, applying
         sign-flips for start/min/max/nominal on negated pairings. Multiple
         eliminated aliases in the same component cumulatively fill rep gaps. =#
      local repCurrentAttr = get(repAttrUpdates, bestName, bestSv.attributes)
      repAttrUpdates[bestName] = _mergeAliasAttrs(repCurrentAttr, sv.attributes, negated)
    end

    #= Mark equations for removal using union-find on surviving variables.
       Trivial equations (both sides resolve to same variable) are always removed.
       Among meaningful equations, only keep enough to span the surviving variables
       (union-find ensures a spanning tree). Redundant equations are removed. =#
    local ufParent = Dict{String,String}()
    for eqIdx in componentEqs[compIdx]
      local (n1, n2) = eqIdxToNames[eqIdx]
      local r1 = haskey(aliasMap, n1) ? aliasMap[n1][1] : n1
      local r2 = haskey(aliasMap, n2) ? aliasMap[n2][1] : n2
      if r1 == r2
        #= Trivial: both sides resolve to same variable (0 = 0). Remove. =#
        push!(aliasEqIndices, eqIdx)
      elseif _ufUnion!(ufParent, r1, r2)
        #= Non-redundant constraint between surviving variables. Keep. =#
      else
        #= Redundant: surviving variables already connected. Remove. =#
        push!(aliasEqIndices, eqIdx)
      end
    end
  end

  if isempty(aliasMap)
    @debug "[SIMCODE: $(simCode.name): aliasElimination] no variables could be eliminated"
    return simCode
  end

  #= ===== Step 4: Substitute alias CREFs in all remaining equations ===== =#
  local newResEqs = RESIDUAL_EQUATION[]
  local elimEqs = RESIDUAL_EQUATION[]
  sizehint!(newResEqs, nEqs - length(aliasEqIndices))

  for (i, eq) in enumerate(resEqs)
    if i in aliasEqIndices
      push!(elimEqs, eq)
    else
      local (newExp, _) = traverseExpTopDown(eq.exp, substituteAliasCref, aliasMap)
      push!(newResEqs, typeof(eq)(newExp, eq.source, eq.attr))
    end
  end

  #= Also substitute in if-equation branches =#
  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = traverseExpTopDown(brEq.exp, substituteAliasCref, aliasMap)
        push!(newBranchEqs, typeof(brEq)(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = traverseExpTopDown(branch.condition, substituteAliasCref, aliasMap)
      #= Reconstruct BRANCH with substituted expressions but same structural info =#
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  #= When equations: substitute alias CREFs in conditions AND statements =#
  local newWhenEqs = WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local innerWhen = _substituteAliasInWhenStmts(whenEq.whenEquation, aliasMap)
    @assign whenEq.whenEquation = innerWhen
    push!(newWhenEqs, whenEq)
  end

  #= Initial equations: substitute alias CREFs.
     initialEquations may contain EQUATION (lhs/rhs) or RESIDUAL_EQUATION (exp). =#
  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION || initEq isa RESIDUAL_EQUATION
      local (newInitExp, _) = Util.traverseExpTopDown(toDAEExp(initEq.exp), substituteAliasCref, aliasMap)
      push!(newInitEqs, typeof(initEq)(newInitExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), substituteAliasCref, aliasMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), substituteAliasCref, aliasMap)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    elseif initEq isa EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), substituteAliasCref, aliasMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), substituteAliasCref, aliasMap)
      push!(newInitEqs, EQUATION(newLhs, newRhs, initEq.source, initEq.attr))
    else
      push!(newInitEqs, initEq)
    end
  end
  local newInitialAlgs = _substituteAliasInInitialAlgorithms(simCode.initialAlgorithms, aliasMap)

  #= ===== Step 5: Verify substitution and remove eliminated variables ===== =#
  #= Collect all CREF names from remaining equations. Any eliminated variable
     still referenced means the substitution missed it (e.g. unflatten CREF form).
     Those variables must be kept in the HT to avoid KeyError during code gen. =#
  local eliminatedSet = OrderedSet{String}(keys(aliasMap))
  local survivingRefs = OrderedSet{String}()
  local allRefNames = OrderedSet{String}()
  for eq in newResEqs
    collectCrefNames!(allRefNames, eq.exp)
  end
  for ifEq in newIfEqs
    for branch in ifEq.branches
      for brEq in branch.residualEquations
        collectCrefNames!(allRefNames, brEq.exp)
      end
    end
  end
  for initEq in newInitEqs
    if initEq isa BDAE.RESIDUAL_EQUATION || initEq isa RESIDUAL_EQUATION
      collectCrefNames!(allRefNames, initEq.exp)
    elseif initEq isa BDAE.EQUATION || initEq isa EQUATION
      collectCrefNames!(allRefNames, initEq.lhs)
      collectCrefNames!(allRefNames, initEq.rhs)
    end
  end
  for ia in newInitialAlgs
    _collectInitialAlgorithmCrefNames!(allRefNames, ia)
  end
  #= Also check when-equations (conditions and statements) for surviving references =#
  for whenEq in newWhenEqs
    _collectWhenCrefNames!(allRefNames, whenEq.whenEquation)
  end
  for n in allRefNames
    if n in eliminatedSet
      push!(survivingRefs, n)
    end
  end

  if !isempty(survivingRefs)
    @warn "[SIMCODE: $(simCode.name): aliasElimination] $(length(survivingRefs)) eliminated variables still referenced, keeping them" survivingRefs=collect(survivingRefs)
  end

  #= Remove only safely eliminated variables from hash table =#
  local newHT = copy(ht)
  local elimVarNames = String[]
  local keptAliasEntries = AliasEntry[]
  for (varName, _) in aliasMap
    if varName in survivingRefs
      #= Keep this variable: still referenced in equations =#
      continue
    end
    delete!(newHT, varName)
    push!(elimVarNames, varName)
  end
  #= Apply lifted attributes onto the surviving representatives so the
     eliminated alias's start / fixed / stateSelect / min / max / nominal do
     not vanish with the deleted alias variable. =#
  for (repName, newAttr) in repAttrUpdates
    haskey(newHT, repName) || continue
    local (rIdx, rOldSv) = newHT[repName]
    if newAttr !== rOldSv.attributes
      newHT[repName] = (rIdx, SIMVAR(rOldSv.name, rOldSv.index, rOldSv.varKind, newAttr))
    end
  end
  #= Filter alias entries to only include actually eliminated variables =#
  for entry in aliasEntries
    if !(entry.eliminatedName in survivingRefs)
      push!(keptAliasEntries, entry)
    end
  end

  #= Build parallel eliminated-variable/equation metadata. aliasEqIndices may
     contain redundant alias equations that were removed because they add no new
     constraint after substitution; those equations do not correspond to a
     removed variable and must not be appended to eliminatedEquations. =#
  local elimVarSet = OrderedSet{String}(elimVarNames)
  local removedAliasIncidence = Tuple{Int, String, String}[]
  for (n1, n2, _, eqIdx, _, _, _, _) in aliasPairs
    if eqIdx in aliasEqIndices && (n1 in elimVarSet || n2 in elimVarSet)
      push!(removedAliasIncidence, (eqIdx, n1, n2))
    end
  end

  local eqByElimVar = Dict{String, Int}()
  local varByElimEq = Dict{Int, String}()
  function assignElimEq!(varName::String, seenEqIdxs::OrderedSet{Int})::Bool
    for (eqIdx, n1, n2) in removedAliasIncidence
      if n1 != varName && n2 != varName
        continue
      end
      if eqIdx in seenEqIdxs
        continue
      end
      push!(seenEqIdxs, eqIdx)
      if !haskey(varByElimEq, eqIdx) || assignElimEq!(varByElimEq[eqIdx], seenEqIdxs)
        varByElimEq[eqIdx] = varName
        eqByElimVar[varName] = eqIdx
        return true
      end
    end
    return false
  end

  for varName in elimVarNames
    assignElimEq!(varName, OrderedSet{Int}())
  end

  local pairedElimVarNames = String[]
  local pairedElimEqs = RESIDUAL_EQUATION[]
  for varName in elimVarNames
    if haskey(eqByElimVar, varName)
      push!(pairedElimVarNames, varName)
      push!(pairedElimEqs, resEqs[eqByElimVar[varName]])
    end
  end
  if length(pairedElimVarNames) != length(elimVarNames)
    local unpairedVars = setdiff(elimVarNames, pairedElimVarNames)
    @info "[SIMCODE: $(simCode.name): aliasElimination] could not pair all eliminated variables with removed alias equations" unpaired=unpairedVars
    #= Fallback: synthesise an identity observation for each unpaired eliminated variable.
       This happens when the alias equation for the eliminated variable was kept as a
       non-trivial constraint between surviving variables (e.g. because the other side is
       irreducible). The variable's aliasMap entry gives us the direct assignment. =#
    for uv in unpairedVars
      if haskey(aliasMap, uv) && haskey(nameToCrefType, uv)
        local (repName, negated, repCref, repTy) = aliasMap[uv]
        local (uvCref, uvTy) = nameToCrefType[uv]
        local uvExp  = DAE.CREF(uvCref, uvTy)
        local repExp = DAE.CREF(repCref, repTy)
        #= 0 = uv - rep  (positive alias)  or  0 = uv + rep  (negated alias) =#
        local synExp = negated ?
          DAE.BINARY(uvExp, DAE.ADD(DAE.T_REAL_DEFAULT), repExp) :
          DAE.BINARY(uvExp, DAE.SUB(DAE.T_REAL_DEFAULT), repExp)
        push!(pairedElimVarNames, uv)
        push!(pairedElimEqs, RESIDUAL_EQUATION(synExp))
      end
    end
  end

  @debug "[SIMCODE: $(simCode.name): aliasElimination] eliminated $(length(elimVarNames)) variables and removed $(length(aliasEqIndices)) equations ($(length(pairedElimVarNames)) paired for observation, $(length(newResEqs)) equations, $(length(newHT)) variables remain)"

  #= eliminateAliasVariables can run more than once; merge (do not replace) the
     alias observations so a later run does not discard a prior run's entries
     (e.g. overconstrained-connector reference-gamma), keeping them retrievable. =#
  local mergedAliasMap = copy(simCode.aliasMap)
  local seenElimNames = OrderedSet{String}(e.eliminatedName for e in mergedAliasMap)
  for e in keptAliasEntries
    if !(e.eliminatedName in seenElimNames)
      push!(mergedAliasMap, e)
      push!(seenElimNames, e.eliminatedName)
    end
  end

  @assign begin
    simCode.residualEquations = newResEqs
    simCode.initialEquations = newInitEqs
    simCode.initialAlgorithms = newInitialAlgs
    simCode.stringToSimVarHT = newHT
    simCode.ifEquations = newIfEqs
    simCode.whenEquations = newWhenEqs
    simCode.aliasMap = mergedAliasMap
    simCode.asserts = _substituteInAsserts(simCode.asserts, aliasMap)
  end
  #= State-state aliases collapse two STATEs marked irreducible into one.
     Drop the eliminated names from `irreducibleVariables` so MTK codegen's
     start-condition lookup (`getStartConditionsMTK`) doesn't try to look up
     a name that no longer exists in `stringToSimVarHT`. =#
  local elimVarSet = OrderedSet{String}(elimVarNames)
  @assign simCode.irreducibleVariables = filter(n -> !(n in elimVarSet), simCode.irreducibleVariables)
  #= The earlier eliminated equations (an output-only or a fold pair) can read
     names eliminated here: substituted, as eliminateRHSEquivalentEquations
     does (codegen otherwise resolves them through aliasMap, one scan and a
     warning per read). =#
  local elimEqs = simCode.eliminatedEquations
  for i in eachindex(elimEqs)
    local (newExp, _) = traverseExpTopDown(elimEqs[i].exp, substituteAliasCref, aliasMap)
    elimEqs[i] = typeof(elimEqs[i])(newExp, elimEqs[i].source, elimEqs[i].attr)
  end
  #= Append eliminated equations/variables to the existing lists =#
  append!(elimEqs, pairedElimEqs)
  append!(simCode.eliminatedVariables, pairedElimVarNames)
  return simCode
end

#= AUDIT (ombackend-bug-audit-2026-06-05 #9): substituteAliasCref legitimately
   wraps a NEGATED alias as UNARY(UMINUS, rep), but on an ASSIGN/REINIT target
   that is an invalid lvalue. Redistribute the sign to the value side, which is
   semantics-preserving: `-x := r` == `x := -r`, `reinit(-x, v)` == `reinit(x, -v)`.
   Non-negated targets pass through unchanged. =#
function _redistributeNegatedAliasLhs(lhsDAE::DAE.Exp, rhsDAE::DAE.Exp)
  @match lhsDAE begin
    DAE.UNARY(DAE.UMINUS(__), inner) =>
      (inner, DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), rhsDAE))
    _ => (lhsDAE, rhsDAE)
  end
end

#= SimExp call site: detect a negated-alias target via DAE normalization and
   only round-trip when it actually fires, so the common (non-negated) case is
   not perturbed. =#
function _redistributeNegatedAliasLhsSim(newL, newR)
  local lhsDAE = toDAEExp(newL)
  @match lhsDAE begin
    DAE.UNARY(DAE.UMINUS(__), _) => begin
      local (fInner, fNegR) = _redistributeNegatedAliasLhs(lhsDAE, toDAEExp(newR))
      (toSimExp(fInner), toSimExp(fNegR))
    end
    _ => (newL, newR)
  end
end

"""
Recursively substitute alias CREFs in a WHEN_STMTS node (condition + statements + elsewhen).
"""
function _substituteAliasInWhenStmts(whenStmts::WHEN_STMTS, aliasMap)
  local (newCond, _) = traverseExpTopDown(whenStmts.condition, substituteAliasCref, aliasMap)
  local newStmtLst = WhenOperator[]
  for stmt in whenStmts.whenStmtLst
    local newStmt::WhenOperator = if stmt isa ASSIGN
      local (newL, _) = traverseExpTopDown(stmt.left, substituteAliasCref, aliasMap)
      local (newR, _) = traverseExpTopDown(stmt.right, substituteAliasCref, aliasMap)
      local (fL, fR) = _redistributeNegatedAliasLhsSim(newL, newR)
      ASSIGN(fL, fR, stmt.source)
    elseif stmt isa REINIT
      local (newSV, _) = Util.traverseExpTopDown(stmt.stateVar, substituteAliasCref, aliasMap)
      local (newVal, _) = traverseExpTopDown(stmt.value, substituteAliasCref, aliasMap)
      local (fSV, fVal) = _redistributeNegatedAliasLhsSim(newSV, newVal)
      REINIT(fSV, fVal, stmt.source)
    elseif stmt isa NORETCALL
      local (newExp, _) = traverseExpTopDown(stmt.exp, substituteAliasCref, aliasMap)
      NORETCALL(newExp, stmt.source)
    elseif stmt isa ASSERT
      local (newC, _) = traverseExpTopDown(stmt.condition, substituteAliasCref, aliasMap)
      local (newM, _) = traverseExpTopDown(stmt.message, substituteAliasCref, aliasMap)
      ASSERT(newC, newM, stmt.level, stmt.source)
    else
      stmt
    end
    push!(newStmtLst, newStmt)
  end
  local newElse = whenStmts.elsewhenPart === nothing ? nothing :
                  _substituteAliasInWhenStmts(whenStmts.elsewhenPart, aliasMap)
  return WHEN_STMTS(newCond, newStmtLst, newElse)
end

function _substituteAliasInWhenStmts(whenStmts::BDAE.WHEN_STMTS, aliasMap)
  local (newCond, _) = Util.traverseExpTopDown(toDAEExp(whenStmts.condition), substituteAliasCref, aliasMap)
  local newStmtLst::List{BDAE.WhenOperator} = MetaModelica.nil
  for stmt in whenStmts.whenStmtLst
    local newStmt::BDAE.WhenOperator = @match stmt begin
      BDAE.ASSIGN(__) => begin
        local (newL, _) = Util.traverseExpTopDown(stmt.left, substituteAliasCref, aliasMap)
        local (newR, _) = Util.traverseExpTopDown(stmt.right, substituteAliasCref, aliasMap)
        local (fL, fR) = _redistributeNegatedAliasLhs(newL, newR)
        BDAE.ASSIGN(fL, fR, stmt.source)
      end
      BDAE.REINIT(__) => begin
        local (newSV, _) = Util.traverseExpTopDown(stmt.stateVar, substituteAliasCref, aliasMap)
        local (newVal, _) = Util.traverseExpTopDown(stmt.value, substituteAliasCref, aliasMap)
        local (fSV, fVal) = _redistributeNegatedAliasLhs(newSV, newVal)
        BDAE.REINIT(fSV, fVal, stmt.source)
      end
      BDAE.NORETCALL(__) => begin
        local (newExp, _) = Util.traverseExpTopDown(stmt.exp, substituteAliasCref, aliasMap)
        BDAE.NORETCALL(newExp, stmt.source)
      end
      BDAE.ASSERT(__) => begin
        local (newC, _) = Util.traverseExpTopDown(stmt.condition, substituteAliasCref, aliasMap)
        local (newM, _) = Util.traverseExpTopDown(stmt.message, substituteAliasCref, aliasMap)
        BDAE.ASSERT(newC, newM, stmt.level, stmt.source)
      end
      _ => stmt
    end
    newStmtLst = MetaModelica.Cons{BDAE.WhenOperator}(newStmt, newStmtLst)
  end
  newStmtLst = listReverse(newStmtLst)
  local newElse = @match whenStmts.elsewhenPart begin
    SOME(elseWhenEq) => SOME(_substituteAliasInElseWhen(elseWhenEq, aliasMap))
    NONE() => NONE()
  end
  return BDAE.WHEN_STMTS(newCond, newStmtLst, newElse)
end

function _substituteAliasInElseWhen(elseWhenEq, aliasMap)
  local inner = elseWhenEq.whenEquation
  local newInner = _substituteAliasInWhenStmts(inner, aliasMap)
  @assign elseWhenEq.whenEquation = newInner
  return elseWhenEq
end

#= `visitor` substitutes (substituteAliasCref; substituteConstantParameter for the
   eliminated parameters' values). =#
function _substituteAliasInInitialAlgorithms(initialAlgs::Vector{INITIAL_ALGORITHM}, aliasMap::AbstractDict{String};
                                             visitor::Function = substituteAliasCref)::Vector{INITIAL_ALGORITHM}
  local result = INITIAL_ALGORITHM[]
  sizehint!(result, length(initialAlgs))
  for ia in initialAlgs
    local newOps = [_substituteAliasInInitialWhenOp(op, aliasMap; visitor) for op in ia.statements]
    local newDae = [_substituteAliasInInitialDAEStmt(stmt, aliasMap; visitor) for stmt in ia.daeStatements]
    push!(result, INITIAL_ALGORITHM(newOps, newDae))
  end
  return result
end

function _substituteAliasInInitialWhenOp(stmt::WhenOperator, aliasMap::AbstractDict{String}; visitor::Function = substituteAliasCref)
  if stmt isa ASSIGN
    local (newL, _) = traverseExpTopDown(stmt.left, visitor, aliasMap)
    local (newR, _) = traverseExpTopDown(stmt.right, visitor, aliasMap)
    local (fL, fR) = _redistributeNegatedAliasLhsSim(newL, newR)
    return ASSIGN(fL, fR, stmt.source)
  elseif stmt isa REINIT
    local (newSV, _) = traverseExpTopDown(stmt.stateVar, visitor, aliasMap)
    local (newVal, _) = traverseExpTopDown(stmt.value, visitor, aliasMap)
    local (fSV, fVal) = _redistributeNegatedAliasLhsSim(newSV, newVal)
    return REINIT(fSV, fVal, stmt.source)
  elseif stmt isa NORETCALL
    local (newExp, _) = traverseExpTopDown(stmt.exp, visitor, aliasMap)
    return NORETCALL(newExp, stmt.source)
  elseif stmt isa ASSERT
    local (newC, _) = traverseExpTopDown(stmt.condition, visitor, aliasMap)
    local (newM, _) = traverseExpTopDown(stmt.message, visitor, aliasMap)
    local (newL, _) = traverseExpTopDown(stmt.level, visitor, aliasMap)
    return ASSERT(newC, newM, newL, stmt.source)
  elseif stmt isa TERMINATE
    local (newM, _) = traverseExpTopDown(stmt.message, visitor, aliasMap)
    return TERMINATE(newM, stmt.source)
  end
  return stmt
end

function _substituteAliasInInitialDAEStmt(stmt::DAE.Statement, aliasMap::AbstractDict{String}; visitor::Function = substituteAliasCref)
  return @match stmt begin
    DAE.STMT_ASSIGN(ty, e1, e, src) => begin
      local (newL, _) = Util.traverseExpTopDown(e1, visitor, aliasMap)
      local (newR, _) = Util.traverseExpTopDown(e, visitor, aliasMap)
      local (fL, fR) = _redistributeNegatedAliasLhs(newL, newR)
      DAE.STMT_ASSIGN(ty, fL, fR, src)
    end
    DAE.STMT_TUPLE_ASSIGN(ty, lhsList, e, src) => begin
      local newLhs = MetaModelica.list((first(Util.traverseExpTopDown(lhs, visitor, aliasMap)) for lhs in lhsList)...)
      local (newR, _) = Util.traverseExpTopDown(e, visitor, aliasMap)
      DAE.STMT_TUPLE_ASSIGN(ty, newLhs, newR, src)
    end
    DAE.STMT_ASSIGN_ARR(ty, lhs, e, src) => begin
      local (newL, _) = Util.traverseExpTopDown(lhs, visitor, aliasMap)
      local (newR, _) = Util.traverseExpTopDown(e, visitor, aliasMap)
      local (fL, fR) = _redistributeNegatedAliasLhs(newL, newR)
      DAE.STMT_ASSIGN_ARR(ty, fL, fR, src)
    end
    DAE.STMT_NORETCALL(e, src) =>
      DAE.STMT_NORETCALL(first(Util.traverseExpTopDown(e, visitor, aliasMap)), src)
    DAE.STMT_ASSERT(c, m, l, src) =>
      DAE.STMT_ASSERT(first(Util.traverseExpTopDown(c, visitor, aliasMap)),
                      first(Util.traverseExpTopDown(m, visitor, aliasMap)),
                      first(Util.traverseExpTopDown(l, visitor, aliasMap)), src)
    DAE.STMT_TERMINATE(m, src) =>
      DAE.STMT_TERMINATE(first(Util.traverseExpTopDown(m, visitor, aliasMap)), src)
    DAE.STMT_IF(cond, stmts, else_, src) =>
      DAE.STMT_IF(first(Util.traverseExpTopDown(cond, visitor, aliasMap)),
                  MetaModelica.list((_substituteAliasInInitialDAEStmt(s, aliasMap; visitor) for s in stmts)...),
                  _substituteAliasInInitialDAEElse(else_, aliasMap; visitor), src)
    DAE.STMT_FOR(ty, isArr, iter, idx, range, body, src) =>
      DAE.STMT_FOR(ty, isArr, iter, idx,
                   first(Util.traverseExpTopDown(range, visitor, aliasMap)),
                   MetaModelica.list((_substituteAliasInInitialDAEStmt(s, aliasMap; visitor) for s in body)...), src)
    DAE.STMT_PARFOR(ty, isArr, iter, idx, range, body, prl, src) =>
      DAE.STMT_PARFOR(ty, isArr, iter, idx,
                      first(Util.traverseExpTopDown(range, visitor, aliasMap)),
                      MetaModelica.list((_substituteAliasInInitialDAEStmt(s, aliasMap; visitor) for s in body)...), prl, src)
    DAE.STMT_WHILE(cond, body, src) =>
      DAE.STMT_WHILE(first(Util.traverseExpTopDown(cond, visitor, aliasMap)),
                     MetaModelica.list((_substituteAliasInInitialDAEStmt(s, aliasMap; visitor) for s in body)...), src)
    DAE.STMT_REINIT(varExp, value, src) => begin
      local (newSV, _) = Util.traverseExpTopDown(varExp, visitor, aliasMap)
      local (newVal, _) = Util.traverseExpTopDown(value, visitor, aliasMap)
      local (fSV, fVal) = _redistributeNegatedAliasLhs(newSV, newVal)
      DAE.STMT_REINIT(fSV, fVal, src)
    end
    _ => stmt
  end
end

function _substituteAliasInInitialDAEElse(else_::DAE.Else, aliasMap::AbstractDict{String}; visitor::Function = substituteAliasCref)
  return @match else_ begin
    DAE.ELSE(stmts) =>
      DAE.ELSE(MetaModelica.list((_substituteAliasInInitialDAEStmt(s, aliasMap; visitor) for s in stmts)...))
    DAE.ELSEIF(cond, stmts, rest) =>
      DAE.ELSEIF(first(Util.traverseExpTopDown(cond, visitor, aliasMap)),
                 MetaModelica.list((_substituteAliasInInitialDAEStmt(s, aliasMap; visitor) for s in stmts)...),
                 _substituteAliasInInitialDAEElse(rest, aliasMap; visitor))
    _ => else_
  end
end

function _collectInitialAlgorithmCrefNames!(names::OrderedSet{String}, ia::INITIAL_ALGORITHM)
  for stmt in ia.statements
    if stmt isa ASSIGN
      collectCrefNames!(names, stmt.left)
      collectCrefNames!(names, stmt.right)
    elseif stmt isa REINIT
      collectCrefNames!(names, stmt.stateVar)
      collectCrefNames!(names, stmt.value)
    elseif stmt isa NORETCALL
      collectCrefNames!(names, stmt.exp)
    elseif stmt isa ASSERT
      collectCrefNames!(names, stmt.condition)
      collectCrefNames!(names, stmt.message)
      collectCrefNames!(names, stmt.level)
    elseif stmt isa TERMINATE
      collectCrefNames!(names, stmt.message)
    end
  end
  _walkStatementsForCrefs!(names, ia.daeStatements)
  return names
end

function _isZeroConstExp(@nospecialize(e))::Bool
  @match e begin
    DAE.RCONST(r) => r == 0.0
    DAE.ICONST(i) => i == 0
    _ => false
  end
end

# Peel residual `lhs - 0` / `lhs + 0` wrappers so detectAlias sees the inner
# alias pattern. Modelica equations of the form `A + B = 0` lower to the
# residual `(A + B) - 0.0 = 0`, which without peeling escapes alias detection.
function _peelZeroResidualWrapper(@nospecialize(exp))
  local cur = exp
  while true
    local matched = @match cur begin
      DAE.BINARY(exp1 = inner, operator = op, exp2 = rhs) => begin
        if !_isZeroConstExp(rhs)
          nothing
        else
          local isSubOrAdd = @match op begin
            DAE.SUB(__) => true
            DAE.ADD(__) => true
            _ => false
          end
          isSubOrAdd ? inner : nothing
        end
      end
      _ => nothing
    end
    matched === nothing && return cur
    cur = matched
  end
end

# Extract `(name, cref, type, negated)` from an operand that may be wrapped in
# one or more nested unary minus expressions.
function _extractCrefWithSign(@nospecialize(e))
  local negated = false
  local cur = e
  while true
    @match cur begin
      DAE.UNARY(operator = op, exp = inner) => begin
        local isUm = @match op begin
          DAE.UMINUS(__) => true
          _ => false
        end
        isUm || break
        negated = !negated
        cur = inner
      end
      _ => break
    end
  end
  local r = extractCrefName(cur)
  r === nothing && return nothing
  local (n, cr, t) = r
  return (n, cr, t, negated)
end

"""
    detectAlias(exp::DAE.Exp, ht)

Detect if an expression represents an alias equation.
Recognizes patterns of the form `c1*a + c2*b = 0` with `c1, c2 ∈ {-1, +1}`:
  - `BINARY(a, SUB, b)` ≡ `a - b = 0`, i.e. `a = b` (negated=false)
  - `BINARY(a, ADD, b)` ≡ `a + b = 0`, i.e. `a = -b` (negated=true)
  - A trailing `- 0` / `+ 0` wrapper on the residual is peeled so connect-style
    equations `(a + b) - 0.0 = 0` are matched.
  - Either operand may be wrapped in a unary minus; the polarity is folded into
    the returned `negated` flag.

Where `a` and `b` can be bare CREFs or ASUB-wrapped CREFs. Both variables must
exist in the hash table and be of the same alias-eligible class
(Real-Real, Bool-Bool, Int-Int, Enum-Enum).

Returns `(name1, name2, negated, cref1, type1, cref2, type2)` or `nothing`.
"""
function detectAlias(@nospecialize(exp), ht)
  local peeled = _peelZeroResidualWrapper(exp)
  if peeled !== exp
    @debug "[detectAlias] peeled wrapper" original=exp peeled=peeled
  end
  @match peeled begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      local isAdd = @match op begin
        DAE.ADD(__) => true
        _ => false
      end
      if !isSub && !isAdd
        return nothing
      end
      local r1 = _extractCrefWithSign(e1)
      local r2 = _extractCrefWithSign(e2)
      if r1 === nothing || r2 === nothing
        if peeled !== exp
          @debug "[detectAlias] wrapped pattern operands not crefs" e1=e1 e2=e2
        end
        return nothing
      end
      local (n1, cr1, t1, neg1) = r1
      local (n2, cr2, t2, neg2) = r2
      #= Both must exist in hash table =#
      if !haskey(ht, n1) || !haskey(ht, n2)
        return nothing
      end
      #= Both must be of an alias-eligible type, and matching class
         (Real-Real, Bool-Bool, Int-Int, Enum-Enum). Cross-class mixing
         is rejected. =#
      local (_, sv1) = ht[n1]
      local (_, sv2) = ht[n2]
      local cls1 = _aliasTypeClass(t1)
      local cls2 = _aliasTypeClass(t2)
      if cls1 === :other || cls2 === :other || cls1 !== cls2
        return nothing
      end
      #= Both must be unknowns (not parameters, strings, or data structures).
         Alias elimination removes equations and variables in pairs. If one side
         is a parameter, removing the equation leaves the unknown without a
         defining equation, breaking the equation-unknown balance. =#
      if !isUnknownVarKind(sv1.varKind) || !isUnknownVarKind(sv2.varKind)
        return nothing
      end
      #= Polarity: equation is sa*a + opCoeff*sb*b = 0 with
         sa=±1, sb=±1, opCoeff=+1 (ADD) or -1 (SUB). After normalising
         the coefficient on `a` to +1, the coefficient on `b` is sa*sb*opCoeff.
         If positive: a + b = 0 → a = -b (negated=true).
         If negative: a - b = 0 → a = b  (negated=false). =#
      local sa = neg1 ? -1 : 1
      local sb = neg2 ? -1 : 1
      local opCoeff = isAdd ? 1 : -1
      local negated = (sa * sb * opCoeff) > 0
      return (n1, n2, negated, cr1, t1, cr2, t2)
    end
    _ => return nothing
  end
end

"""
    substituteAliasCref(exp::DAE.Exp, aliasMap)

Callback for `traverseExpTopDown`. Replaces CREF expressions whose name
matches an alias map entry with the representative CREF (possibly negated).
Also handles ASUB-wrapped CREFs.
"""
function substituteAliasCref(@nospecialize(exp), aliasMap)
  @match exp begin
    #= `der(x)` / `pre(x)` / `edge(x)` / `change(x)` builtins expect a bare CREF
       argument at codegen time. When the inner CREF is aliased with negation,
       push the UMINUS outside the call (`der(-y)` ≡ `-der(y)`) so the codegen
       still receives a CREF inside the call. =#
    DAE.CALL(path = Absyn.IDENT(fnName), expLst = expl) => begin
      if _isUnaryStateBuiltin(fnName) && _hasNegatedAliasArg(expl, aliasMap)
        local newArgs = _substituteAliasInBuiltinArgs(expl, aliasMap)
        local newCall = DAE.CALL(exp.path, newArgs, exp.attr)
        return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newCall), false, aliasMap)
      end
      return (exp, true, aliasMap)
    end
    DAE.ASUB(innerExp, subs) => begin
      @match innerExp begin
        DAE.CREF(cr, ty) => begin
          local baseName = DAE_identifierToString(cr)
          local fullName = buildAsubName(baseName, subs)
          if !isempty(fullName) && haskey(aliasMap, fullName)
            local (repName, negated, repCref, repTy) = aliasMap[fullName]
            #= Check if the representative also has ASUB subscripts =#
            local repBase = replace(repName, r"\[.*" => "")
            if repBase != repName
              #= Representative is also subscripted. Build ASUB with rep CREF. =#
              local repSubs = parseSubscriptsFromName(repName)
              local newInner = DAE.CREF(repCref, repTy)
              local newExp = DAE.ASUB(newInner, repSubs)
              if negated
                return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
              else
                return (newExp, false, aliasMap)
              end
            else
              #= Representative is a scalar. Use bare CREF. =#
              local newExp = DAE.CREF(repCref, repTy)
              if negated
                return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
              else
                return (newExp, false, aliasMap)
              end
            end
          end
          #= Also check the base name (for cases where ASUB+CREF base name is aliased) =#
          if haskey(aliasMap, baseName)
            local (repName, negated, repCref, repTy) = aliasMap[baseName]
            local newInner = DAE.CREF(repCref, repTy)
            local newExp = DAE.ASUB(newInner, subs)
            if negated
              return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
            else
              return (newExp, false, aliasMap)
            end
          end
          return (exp, true, aliasMap)
        end
        _ => return (exp, true, aliasMap)
      end
    end
    DAE.CREF(cr, ty) => begin
      local name = DAE_identifierToString(cr)
      if haskey(aliasMap, name)
        local (repName, negated, repCref, repTy) = aliasMap[name]
        local newExp = DAE.CREF(repCref, repTy)
        if negated
          return (DAE.UNARY(DAE.UMINUS(DAE.T_REAL(MetaModelica.nil)), newExp), false, aliasMap)
        else
          return (newExp, false, aliasMap)
        end
      end
      return (exp, true, aliasMap)
    end
    _ => return (exp, true, aliasMap)
  end
end

#= SIM-native dispatch so traverseExpTopDown can substitute aliases without a
   whole-tree DAE round-trip; aliasMap stays DAE-cref-valued, only the matched
   leaf cref is converted. =#
function substituteAliasCref(exp::EXP_CREF, aliasMap)
  local name = DAE_identifierToString(toDAECref(exp.cref).componentRef)
  if haskey(aliasMap, name)
    local (_, negated, repCref, repTy) = aliasMap[name]
    local newExp = EXP_CREF(SimCref(repCref), repTy)
    return (negated ? UNARY(OP_UMINUS, newExp) : newExp, false, aliasMap)
  end
  return (exp, true, aliasMap)
end

substituteAliasCref(exp::Exp, aliasMap) = (exp, true, aliasMap)

function substituteAliasCref(exp::CALL, aliasMap)
  local fnName = @match exp.path begin
    Absyn.IDENT(n) => n
    _ => ""
  end
  if _isUnaryStateBuiltin(fnName) && _hasNegatedAliasArgSIM(exp.args, aliasMap)
    local newCall = CALL(exp.path, _substituteAliasInBuiltinArgsSIM(exp.args, aliasMap), exp.attr)
    return (UNARY(OP_UMINUS, newCall), false, aliasMap)
  end
  return (exp, true, aliasMap)
end

function substituteAliasCref(exp::ASUB, aliasMap)
  exp.exp isa EXP_CREF || return (exp, true, aliasMap)
  local baseName = DAE_identifierToString(toDAECref(exp.exp.cref).componentRef)
  local fullName = buildAsubName(baseName, _subsToDAE(exp.subs))
  if !isempty(fullName) && haskey(aliasMap, fullName)
    local (repName, negated, repCref, repTy) = aliasMap[fullName]
    local newExp = replace(repName, r"\[.*" => "") != repName ?
      ASUB(EXP_CREF(SimCref(repCref), repTy), _parseSubsSIM(repName)) :
      EXP_CREF(SimCref(repCref), repTy)
    return (negated ? UNARY(OP_UMINUS, newExp) : newExp, false, aliasMap)
  end
  if haskey(aliasMap, baseName)
    local (_, negated, repCref, repTy) = aliasMap[baseName]
    local newExp = ASUB(EXP_CREF(SimCref(repCref), repTy), exp.subs)
    return (negated ? UNARY(OP_UMINUS, newExp) : newExp, false, aliasMap)
  end
  return (exp, true, aliasMap)
end

# SIM-native mirrors of _aliasLookupName / _hasNegatedAliasArg / _substituteAliasInBuiltinArgs.
_subsToDAE(subs) = DAE.Subscript[DAE.INDEX(toDAEExp(s)) for s in subs]
_parseSubsSIM(name::String) = Exp[ICONST(parse(Int, m.captures[1])) for m in eachmatch(r"\[(\d+)\]", name)]

function _aliasLookupNameSIM(@nospecialize(e))::Union{Nothing,String}
  if e isa EXP_CREF
    return DAE_identifierToString(toDAECref(e.cref).componentRef)
  elseif e isa ASUB && e.exp isa EXP_CREF
    local baseName = DAE_identifierToString(toDAECref(e.exp.cref).componentRef)
    local full = buildAsubName(baseName, _subsToDAE(e.subs))
    return isempty(full) ? baseName : full
  end
  return nothing
end

function _hasNegatedAliasArgSIM(args, aliasMap)::Bool
  for a in args
    local n = _aliasLookupNameSIM(a)
    n === nothing && continue
    haskey(aliasMap, n) || continue
    aliasMap[n][2] && return true
  end
  return false
end

function _substituteAliasInBuiltinArgsSIM(args, aliasMap)
  local rebuilt = Exp[]
  for a in args
    local n = _aliasLookupNameSIM(a)
    if n === nothing || !haskey(aliasMap, n)
      push!(rebuilt, a)
      continue
    end
    local (_, _, repCref, repTy) = aliasMap[n]
    local newCref = EXP_CREF(SimCref(repCref), repTy)
    push!(rebuilt, (a isa ASUB && !isempty(a.subs)) ? ASUB(newCref, a.subs) : newCref)
  end
  return rebuilt
end

_isUnaryStateBuiltin(fnName::String)::Bool =
  fnName == "der" || fnName == "pre" || fnName == "edge" || fnName == "change"

# Return true if any argument is a CREF/ASUB whose name maps to an alias entry
# with `negated == true`.
function _hasNegatedAliasArg(expl, aliasMap)::Bool
  for a in expl
    local n = _aliasLookupName(a)
    n === nothing && continue
    haskey(aliasMap, n) || continue
    aliasMap[n][2] && return true
  end
  return false
end

# Returns the name used for aliasMap lookup for a CREF or ASUB-wrapped CREF, else nothing.
function _aliasLookupName(@nospecialize(e))::Union{Nothing,String}
  @match e begin
    DAE.CREF(cr, _) => DAE_identifierToString(cr)
    DAE.ASUB(DAE.CREF(cr, _), subs) => begin
      local baseName = DAE_identifierToString(cr)
      local full = buildAsubName(baseName, subs)
      isempty(full) ? baseName : full
    end
    _ => nothing
  end
end

# Substitute each CREF/ASUB-wrapped CREF arg through the alias map, treating any
# negation as already lifted to the enclosing UMINUS by the caller. Returns an
# ImmutableList suitable for DAE.CALL.expLst.
function _substituteAliasInBuiltinArgs(expl, aliasMap)
  local rebuilt = DAE.Exp[]
  for a in expl
    local n = _aliasLookupName(a)
    if n === nothing || !haskey(aliasMap, n)
      push!(rebuilt, a)
      continue
    end
    local (_repName, _negated, repCref, repTy) = aliasMap[n]
    local replaced = @match a begin
      DAE.CREF(_, _) => DAE.CREF(repCref, repTy)
      DAE.ASUB(_, subs) => begin
        local newCref = DAE.CREF(repCref, repTy)
        length(subs) == 0 ? newCref : DAE.ASUB(newCref, subs)
      end
      _ => a
    end
    push!(rebuilt, replaced)
  end
  return MetaModelica.list(rebuilt...)
end

"""
    parseSubscriptsFromName(name::String)::List{DAE.Subscript}

Parse subscripts from a variable name like "a[1][2]" into INDEX(ICONST) subscripts.
Used to reconstruct ASUB subscripts for the representative variable.
"""
function parseSubscriptsFromName(name::String)::MetaModelica.List{DAE.Subscript}
  local subs = MetaModelica.nil
  for m in eachmatch(r"\[(\d+)\]", name)
    subs = MetaModelica.cons(DAE.INDEX(DAE.ICONST(parse(Int, m.captures[1]))), subs)
  end
  return MetaModelica.listReverse(subs)
end

#= Detect residual of the form `0 = var - expr` or `0 = expr - var`
   where var is a simple unknown CREF and expr is anything more complex
   than a single CREF. Returns (name, cref, ty, exprKey) or nothing.
   Skips var-var form (handled by detectAlias). For both `var - expr`
   and `expr - var` the canonical key is `string(expr)`, so two
   equations with the same complex side group together regardless of
   which side the leaf var was on. =#
function _detectVarMinusExpr(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      isSub || return nothing
      #= Peel a UMINUS wrapper on either side so `(-var) - X` and `var - (-X)`
         get canonicalized into `var = ±X` with the negation tracked. =#
      local (e1Peeled, e1Neg) = _peelUMinus(e1)
      local (e2Peeled, e2Neg) = _peelUMinus(e2)
      local r1 = extractCrefName(e1Peeled)
      local r2 = extractCrefName(e2Peeled)
      #= Skip var-var (detectAlias handles this) and complex-complex. =#
      if (r1 !== nothing && r2 !== nothing) || (r1 === nothing && r2 === nothing)
        return nothing
      end
      local r, complexExp, crefNeg, complexSign
      if r1 !== nothing
        r = r1; complexExp = e2Peeled; crefNeg = e1Neg
        #= var - complex => var = complex, with complex negated if e2 had UMINUS. =#
        complexSign = e2Neg
      else
        r = r2; complexExp = e1Peeled; crefNeg = e2Neg
        #= complex - var => var = complex, with complex negated if e1 had UMINUS. =#
        complexSign = e1Neg
      end
      local (n, cr, ty) = r
      haskey(ht, n) || return nothing
      local (_, sv) = ht[n]
      isUnknownVarKind(sv.varKind) || return nothing
      local cls = _aliasTypeClass(ty)
      cls === :other && return nothing
      local negated = xor(crefNeg, complexSign)
      return (n, cr, ty, string(complexExp), negated)
    end
    _ => return nothing
  end
end

function _peelUMinus(@nospecialize(exp))
  @match exp begin
    DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => (inner, true)
    _ => (exp, false)
  end
end

_peelUMinusSIM(e::Exp) = (e isa UNARY && e.op === OP_UMINUS) ? (e.exp, true) : (e, false)

#= Cheap SIM cref test. extractCrefName converts its arg via toDAEExp, which is
   expensive on a complex operand; only EXP_CREF/WILD map to DAE.CREF (the only
   non-nothing cases), so gate on those and convert just the leaf. =#
_simCrefName(e::Exp) = (e isa EXP_CREF || e isa WILD) ? extractCrefName(e) : nothing

#= SIM-native arm: the caller runs inside a fixpoint, so dropping the per-residual
   full-tree toDAEExp(eq.exp) is amplified. Only the complex operand is converted
   (string key must match the DAE form); non-matching residuals bail before any
   conversion. toDAEExp is homomorphic, so converting the peeled complex side
   equals peeling the converted tree -> the string key is byte-identical. =#
function _detectVarMinusExpr(exp::Exp, ht)
  exp isa BINARY || return nothing
  exp.op === OP_SUB || return nothing
  local (e1p, e1Neg) = _peelUMinusSIM(exp.exp1)
  local (e2p, e2Neg) = _peelUMinusSIM(exp.exp2)
  local r1 = _simCrefName(e1p)
  local r2 = _simCrefName(e2p)
  if (r1 !== nothing && r2 !== nothing) || (r1 === nothing && r2 === nothing)
    return nothing
  end
  local r, complexExp, crefNeg, complexSign
  if r1 !== nothing
    r = r1; complexExp = e2p; crefNeg = e1Neg; complexSign = e2Neg
  else
    r = r2; complexExp = e1p; crefNeg = e2Neg; complexSign = e1Neg
  end
  local (n, cr, ty) = r
  haskey(ht, n) || return nothing
  local (_, sv) = ht[n]
  isUnknownVarKind(sv.varKind) || return nothing
  local cls = _aliasTypeClass(ty)
  cls === :other && return nothing
  local negated = xor(crefNeg, complexSign)
  return (n, cr, ty, string(toDAEExp(complexExp)), negated)
end

#= After eliminateAliasVariables, two equations may implicitly assert
   var1 = var2 via identical RHS expressions, e.g.
     0 = x - der(z)
     0 = y - der(z)
   This pass groups by string form of the non-leaf side and aliases
   matching LHS vars to a single representative. =#
#= True when a SimVar is declared `stateSelect = StateSelect.always`: it MUST
   remain a state and carry its own start/fixed init constraint, so it must never
   be aliased away (doing so drops e.g. a fixed=true velocity IC and leaves the
   DAE init free to pick the trivial zero). =#
function _isStateSelectAlways(@nospecialize(sv))::Bool
  @match sv.attributes begin
    SOME(DAE.VAR_ATTR_REAL(stateSelectOption = SOME(DAE.ALWAYS(__)))) => true
    _ => false
  end
end

function eliminateRHSEquivalentEquations(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode)
    return simCode
  end
  local ht  = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local irreducibleSet = OrderedSet{String}(simCode.irreducibleVariables)
  local sharedVarSet   = OrderedSet{String}(simCode.sharedVariables)

  local rhsGroups = OrderedDict{String, Vector{Tuple{String, Int, DAE.ComponentRef, DAE.Type, Bool}}}()
  for (i, eq) in enumerate(resEqs)
    local pair = _detectVarMinusExpr(eq.exp, ht)
    pair === nothing && continue
    local (n, cr, ty, key, neg) = pair
    if !haskey(rhsGroups, key)
      rhsGroups[key] = Tuple{String, Int, DAE.ComponentRef, DAE.Type, Bool}[]
    end
    push!(rhsGroups[key], (n, i, cr, ty, neg))
  end

  local aliasMap = OrderedDict{String, Tuple{String, Bool, DAE.ComponentRef, DAE.Type}}()
  local aliasEntries = AliasEntry[]
  local removeEqs = OrderedSet{Int}()
  local elimVarOrder = String[]
  local elimEqOrder  = RESIDUAL_EQUATION[]
  #= A variable defined by more than one `v - expr` residual appears as a
     reducible member in several RHS-groups. It may be eliminated only once:
     the first group removes its defining equation, later groups must leave
     their equation in place (after substitution it becomes a constraint
     `rep = expr`). `claimed` tracks already-eliminated names and `repSet`
     tracks chosen representatives, so a representative is never itself
     eliminated and each removed equation maps to exactly one eliminated
     variable. =#
  local claimed = OrderedSet{String}()
  local repSet = OrderedSet{String}()
  #= Lift the eliminated member's start / fixed / stateSelect onto the surviving
     representative, mirroring eliminateAliasVariables, so a user IC on an
     RHS-equivalent alias (e.g. a connector velocity) is not lost. =#
  local repAttrUpdates = Dict{String, Any}()

  for (_key, entries) in pairs(rhsGroups)
    length(entries) >= 2 || continue
    local bestIdx = 0
    local bestPrio = -1
    for (j, (n, _, _, _, _)) in enumerate(entries)
      haskey(ht, n) || continue
      n in claimed && continue
      local (_, sv) = ht[n]
      local prio = varKindPriority(sv.varKind)
      if n in irreducibleSet
        prio += 60
      end
      #= stateSelect=always must be kept as a state: make it the representative so
         it survives and its fixed-start init constraint is emitted. =#
      if _isStateSelectAlways(sv)
        prio += 200
      end
      if prio > bestPrio
        bestPrio = prio
        bestIdx = j
      end
    end
    bestIdx == 0 && continue
    local (repName, _, repCref, repTy, repNeg) = entries[bestIdx]
    local (_, repSv) = ht[repName]
    local repIsState = @match repSv.varKind begin
      STATE(__) => true
      _ => false
    end
    push!(repSet, repName)
    for (j, entry) in enumerate(entries)
      j == bestIdx && continue
      local (n, eqIdx, _, _, entryNeg) = entry
      n == repName && continue
      n in sharedVarSet && continue
      #= Already eliminated, or serving as a representative elsewhere: keep its
         equation so the system stays balanced and no alias points at an
         eliminated representative. =#
      (n in claimed || n in repSet) && continue
      if endswith(n, "_re") || endswith(n, "_im")
        continue
      end
      local (_, sv) = ht[n]
      local isState = @match sv.varKind begin
        STATE(__) => true
        _ => false
      end
      #= Never alias away a stateSelect=always variable, even if it is not the
         chosen representative (e.g. two such variables share an RHS). Keeping it
         preserves its fixed=true start as an init constraint. =#
      if _isStateSelectAlways(sv)
        continue
      end
      if n in irreducibleSet && !(repIsState && isState)
        continue
      end
      local aliasNeg = xor(entryNeg, repNeg)
      aliasMap[n] = (repName, aliasNeg, repCref, repTy)
      local repCurrentAttr = get(repAttrUpdates, repName, repSv.attributes)
      repAttrUpdates[repName] = _mergeAliasAttrs(repCurrentAttr, sv.attributes, aliasNeg)
      push!(aliasEntries, AliasEntry(n, repName, aliasNeg))
      push!(removeEqs, eqIdx)
      push!(elimVarOrder, n)
      push!(elimEqOrder, resEqs[eqIdx])
      push!(claimed, n)
    end
  end

  if isempty(aliasMap)
    return simCode
  end

  @info "[SIMCODE: $(simCode.name): eliminateRHSEquivalentEquations] aliased $(length(aliasMap)) variables via RHS equivalence; removing $(length(removeEqs)) redundant equations"
  if OMBackend.BACKEND_PERFLOG[]
    @info "[SIMCODE: $(simCode.name): eliminateRHSEquivalentEquations] model size" residuals_before=length(resEqs) residuals_after=length(resEqs) - length(removeEqs) variables_before=length(ht) variables_after=length(ht) - length(aliasMap)
  end

  local newResEqs = RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(removeEqs))
  for (i, eq) in enumerate(resEqs)
    i in removeEqs && continue
    local (newExp, _) = traverseExpTopDown(eq.exp, substituteAliasCref, aliasMap)
    push!(newResEqs, typeof(eq)(newExp, eq.source, eq.attr))
  end

  #= Substitute in if-equation branches: conditions + branch residual equations.
     Without this, an aliased variable that appears in an if-branch becomes
     a dangling reference at codegen time. Matches eliminateAliasVariables's
     equivalent step. =#
  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = traverseExpTopDown(brEq.exp, substituteAliasCref, aliasMap)
        push!(newBranchEqs, typeof(brEq)(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = traverseExpTopDown(branch.condition, substituteAliasCref, aliasMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  #= When equations: substitute in conditions and statements. =#
  local newWhenEqs = WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local innerWhen = _substituteAliasInWhenStmts(whenEq.whenEquation, aliasMap)
    @assign whenEq.whenEquation = innerWhen
    push!(newWhenEqs, whenEq)
  end

  #= Initial equations: substitute alias CREFs. =#
  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION || initEq isa RESIDUAL_EQUATION
      local (newInitExp, _) = Util.traverseExpTopDown(initEq.exp, substituteAliasCref, aliasMap)
      push!(newInitEqs, typeof(initEq)(newInitExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), substituteAliasCref, aliasMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), substituteAliasCref, aliasMap)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    elseif initEq isa EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), substituteAliasCref, aliasMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), substituteAliasCref, aliasMap)
      push!(newInitEqs, EQUATION(newLhs, newRhs, initEq.source, initEq.attr))
    else
      push!(newInitEqs, initEq)
    end
  end

  #= Substitute in existing eliminatedEquations too. Earlier passes (like
     eliminateAliasVariables) may have appended observed equations that
     reference variables we are now eliminating; without this substitution,
     `generateEliminatedObservedBlock` emits code referencing names that
     have been removed from the HT, causing UndefVarError at module eval. =#
  local oldElimEqs = simCode.eliminatedEquations
  local rewrittenElimEqs = RESIDUAL_EQUATION[]
  sizehint!(rewrittenElimEqs, length(oldElimEqs))
  for eq in oldElimEqs
    local (newExp, _) = traverseExpTopDown(eq.exp, substituteAliasCref, aliasMap)
    push!(rewrittenElimEqs, typeof(eq)(newExp, eq.source, eq.attr))
  end

  local newHT = copy(ht)
  local elimVarSet = OrderedSet{String}(keys(aliasMap))
  for varName in keys(aliasMap)
    delete!(newHT, varName)
  end
  #= Apply lifted attributes onto the surviving representatives. =#
  for (repName, newAttr) in repAttrUpdates
    haskey(newHT, repName) || continue
    local (rIdx, rOldSv) = newHT[repName]
    if newAttr !== rOldSv.attributes
      newHT[repName] = (rIdx, SIMVAR(rOldSv.name, rOldSv.index, rOldSv.varKind, newAttr))
    end
  end

  @assign begin
    simCode.residualEquations = newResEqs
    simCode.initialEquations  = newInitEqs
    simCode.ifEquations       = newIfEqs
    simCode.whenEquations     = newWhenEqs
    simCode.stringToSimVarHT  = newHT
    simCode.eliminatedEquations = rewrittenElimEqs
    simCode.irreducibleVariables = filter(n -> !(n in elimVarSet), simCode.irreducibleVariables)
  end
  append!(simCode.aliasMap, aliasEntries)
  append!(simCode.eliminatedVariables, elimVarOrder)
  append!(simCode.eliminatedEquations, elimEqOrder)
  return simCode
end
