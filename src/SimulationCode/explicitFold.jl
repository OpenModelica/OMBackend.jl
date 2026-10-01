#= Explicitly assigned algebraic variables folded into their uses. =#

Base.@nospecializeinfer function _isAlgebraicVarKind(@nospecialize(varKind))::Bool
  @match varKind begin
    ALG_VARIABLE(__) => true
    _ => false
  end
end

Base.@nospecializeinfer function _containsDerCallDAE(@nospecialize(exp))::Bool
  @match exp begin
    DAE.CALL(path = p) => begin
      @match p begin
        Absyn.IDENT(name) => name == "der"
        _ => false
      end
    end
    DAE.BINARY(exp1 = e1, exp2 = e2) => _containsDerCallDAE(e1) || _containsDerCallDAE(e2)
    DAE.UNARY(exp = e) => _containsDerCallDAE(e)
    DAE.LUNARY(exp = e) => _containsDerCallDAE(e)
    DAE.LBINARY(exp1 = e1, exp2 = e2) => _containsDerCallDAE(e1) || _containsDerCallDAE(e2)
    DAE.IFEXP(expCond = c, expThen = t, expElse = e) => _containsDerCallDAE(c) || _containsDerCallDAE(t) || _containsDerCallDAE(e)
    DAE.ARRAY(array = lst) => any(_containsDerCallDAE, lst)
    #= ASUB subscripts are DAE.Subscript; WHOLEDIM has no expression. =#
    DAE.ASUB(exp = e, sub = subs) => _containsDerCallDAE(e) ||
      any(s -> !(s isa DAE.WHOLEDIM) && _containsDerCallDAE(s.exp), subs)
    DAE.RELATION(exp1 = e1, exp2 = e2) => _containsDerCallDAE(e1) || _containsDerCallDAE(e2)
    DAE.CAST(exp = e) => _containsDerCallDAE(e)
    DAE.TSUB(exp = e) => _containsDerCallDAE(e)
    DAE.RSUB(exp = e) => _containsDerCallDAE(e)
    DAE.REDUCTION(expr = e) => _containsDerCallDAE(e)
    _ => false
  end
end

Base.@nospecializeinfer function _detectVarMinusExprRaw(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      isSub || return nothing
      local r1 = extractCrefName(e1)
      local r2 = extractCrefName(e2)
      if (r1 !== nothing && r2 !== nothing) || (r1 === nothing && r2 === nothing)
        return nothing
      end
      local r, rhs
      if r1 !== nothing
        r = r1; rhs = e2
      else
        r = r2; rhs = e1
      end
      local (n, _cr, _ty) = r
      haskey(ht, n) || return nothing
      return (n, rhs)
    end
    _ => return nothing
  end
end

#= SIM-native arm (fixpoint caller, see _detectVarMinusExpr). rhs flows downstream
   as a DAE.Exp (substituteFoldedVar / _containsDerCallDAE), so the single complex
   operand is converted; non-matching residuals bail before any toDAEExp. =#
function _detectVarMinusExprRaw(exp::Exp, ht)
  exp isa BINARY || return nothing
  exp.op === OP_SUB || return nothing
  local r1 = _simCrefName(exp.exp1)
  local r2 = _simCrefName(exp.exp2)
  if (r1 !== nothing && r2 !== nothing) || (r1 === nothing && r2 === nothing)
    return nothing
  end
  local r, rhs
  if r1 !== nothing
    r = r1; rhs = exp.exp2
  else
    r = r2; rhs = exp.exp1
  end
  local (n, _cr, _ty) = r
  haskey(ht, n) || return nothing
  return (n, toDAEExp(rhs))
end

#= Substitution callback that, for every leaf CREF whose name is a key of
   the fold map, returns the bound RHS expression and stops traversal so
   the substituted form is not re-walked. ASUB-wrapped CREFs are handled
   by reading the constant-subscript suffix into the lookup key, matching
   `collectCrefNames!`'s asubHandled branch. =#
function substituteFoldedVar(@nospecialize(exp), foldMap::Dict{String, DAE.Exp})
  @match exp begin
    DAE.CREF(cr, _) => begin
      local name = DAE_identifierToString(cr)
      if haskey(foldMap, name)
        return (foldMap[name], false, foldMap)
      end
      return (exp, true, foldMap)
    end
    DAE.ASUB(exp = inner, sub = subs) => begin
      @match inner begin
        DAE.CREF(cr, _) => begin
          local baseName = DAE_identifierToString(cr)
          local allConst = true
          local suffix = ""
          for s in subs
            @match s begin
              DAE.INDEX(DAE.ICONST(i)) => begin suffix *= Base.string("[", i, "]") end
              _ => begin allConst = false end
            end
          end
          if allConst && !isempty(suffix)
            local fullName = Base.string(baseName, suffix)
            if haskey(foldMap, fullName)
              return (foldMap[fullName], false, foldMap)
            end
          end
          if haskey(foldMap, baseName)
            return (foldMap[baseName], false, foldMap)
          end
          return (exp, true, foldMap)
        end
        _ => return (exp, true, foldMap)
      end
    end
    _ => return (exp, true, foldMap)
  end
end

#= SIM-native dispatch: replace a matched folded cref/ASUB with the foldMap's
   replacement, converted to SIM via toSimExp (replacement applied once). =#
function substituteFoldedVar(exp::EXP_CREF, foldMap::Dict{String, DAE.Exp})
  local name = DAE_identifierToString(toDAECref(exp.cref).componentRef)
  if haskey(foldMap, name)
    return (toSimExp(foldMap[name]), false, foldMap)
  end
  return (exp, true, foldMap)
end

function substituteFoldedVar(exp::ASUB, foldMap::Dict{String, DAE.Exp})
  exp.exp isa EXP_CREF || return (exp, true, foldMap)
  local baseName = DAE_identifierToString(toDAECref(exp.exp.cref).componentRef)
  local allConst = true
  local suffix = ""
  for s in exp.subs
    if s isa ICONST
      suffix *= Base.string("[", s.value, "]")
    else
      allConst = false
    end
  end
  if allConst && !isempty(suffix)
    local fullName = Base.string(baseName, suffix)
    if haskey(foldMap, fullName)
      return (toSimExp(foldMap[fullName]), false, foldMap)
    end
  end
  if haskey(foldMap, baseName)
    return (toSimExp(foldMap[baseName]), false, foldMap)
  end
  return (exp, true, foldMap)
end

substituteFoldedVar(exp::Exp, foldMap::Dict{String, DAE.Exp}) = (exp, true, foldMap)

"""
    foldExplicitSingleAssign(simCode) -> simCode

Substitute every ALG_VARIABLE that is uniquely defined by a single
`0 = v - rhs` residual, where `rhs` has no derivative and no self-reference
to `v`. Variables protected by if/when references, irreducible / shared
sets are skipped. Sub-model / metaModel / flat-model variants are skipped
entirely because runtime parameter overrides interact with cross-submodel
references that the fold would break.

Iterates to a fixed point (up to 8 rounds) so transitive chains
(`v1 = v2 + 1; v2 = v3 + 1; v3 = literal`) collapse.
"""
function foldExplicitSingleAssign(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode)
    return simCode
  end
  isempty(simCode.residualEquations) && return simCode
  local protectedNames = _computeFrozenProtectedNames(simCode)
  local irreducibleSet = OrderedSet{String}(simCode.irreducibleVariables)
  local sharedVarSet   = OrderedSet{String}(simCode.sharedVariables)
  local totalFolded = 0
  local maxRounds = 8
  for round in 1:maxRounds
    local (newCode, nFolded) = _foldExplicitSingleAssignOnePass(simCode, protectedNames, irreducibleSet, sharedVarSet)
    nFolded == 0 && break
    simCode = newCode
    totalFolded += nFolded
  end
  if totalFolded > 0
    @info "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] folded $(totalFolded) explicit assignments"
    if OMBackend.BACKEND_PERFLOG[]
      @info "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] model size" residuals_after=length(simCode.residualEquations) variables_after=length(simCode.stringToSimVarHT)
    end
  end
  return simCode
end

function _foldExplicitSingleAssignOnePass(simCode::SIM_CODE,
                                          protectedNames::OrderedSet{String},
                                          irreducibleSet::OrderedSet{String},
                                          sharedVarSet::OrderedSet{String})
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations

  local defCountOfVar = Dict{String, Int}()
  local defEqOfVar    = Dict{String, Int}()
  local defRhsOfVar   = Dict{String, DAE.Exp}()

  #= Skip scalarized array elements (any name containing '[' or ']').
     The codegen rebuilds the parent array from its scalar siblings via
     ASUB indexing; dropping a single element from the HT breaks that
     reconstruction even though the algebraic substitution is sound. =#
  #= Skip variables that appear in any existing alias-map entry (either
     side). Folding a representative would orphan the alias entry; folding
     an aliased name would double-substitute via the observed-equation
     pipeline. =#
  local aliasNames = OrderedSet{String}()
  for entry in simCode.aliasMap
    push!(aliasNames, entry.eliminatedName)
    push!(aliasNames, entry.representativeName)
  end

  for (i, eq) in enumerate(resEqs)
    local pair = _detectVarMinusExprRaw(eq.exp, ht)
    pair === nothing && continue
    local (name, rhs) = pair
    occursin('[', name) && continue
    occursin(']', name) && continue
    name in protectedNames && continue
    name in irreducibleSet && continue
    name in sharedVarSet && continue
    name in aliasNames && continue
    local (_, sv) = ht[name]
    _isAlgebraicVarKind(sv.varKind) || continue
    #= A fixed start is an initial equation of the variable: folded away, it
       was lost (`v(start = 3, fixed = true)` with `v = x + 1` left x(0) = 0). =#
    _hasExplicitFixedStart(sv.attributes) && continue
    _containsDerCallDAE(rhs) && continue
    local rhsNames = OrderedSet{String}()
    collectCrefNames!(rhsNames, rhs)
    name in rhsNames && continue
    defCountOfVar[name] = get(defCountOfVar, name, 0) + 1
    if !haskey(defEqOfVar, name)
      defEqOfVar[name]  = i
      defRhsOfVar[name] = rhs
    end
  end

  local foldMap = Dict{String, DAE.Exp}()
  local foldEqIdxSet = OrderedSet{Int}()
  for (name, cnt) in defCountOfVar
    cnt == 1 || continue
    foldMap[name] = defRhsOfVar[name]
    push!(foldEqIdxSet, defEqOfVar[name])
  end

  isempty(foldMap) && return (simCode, 0)

  #= Never empty the residual list. =#
  if length(resEqs) - length(foldEqIdxSet) <= 0
    @info "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] would empty residuals; skipping"
    return (simCode, 0)
  end

  local newResEqs = RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(foldEqIdxSet))
  for (i, eq) in enumerate(resEqs)
    i in foldEqIdxSet && continue
    local (newExp, _) = traverseExpTopDown(eq.exp, substituteFoldedVar, foldMap)
    newExp = _foldNumericExp(newExp)
    push!(newResEqs, typeof(eq)(newExp, eq.source, eq.attr))
  end

  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION || initEq isa RESIDUAL_EQUATION
      local (newExp, _) = Util.traverseExpTopDown(initEq.exp, substituteFoldedVar, foldMap)
      newExp = _foldNumericExp(newExp)
      push!(newInitEqs, typeof(initEq)(newExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), substituteFoldedVar, foldMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), substituteFoldedVar, foldMap)
      newLhs = _foldNumericExp(newLhs)
      newRhs = _foldNumericExp(newRhs)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    elseif initEq isa EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), substituteFoldedVar, foldMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), substituteFoldedVar, foldMap)
      newLhs = _foldNumericExp(newLhs)
      newRhs = _foldNumericExp(newRhs)
      push!(newInitEqs, EQUATION(newLhs, newRhs, initEq.source, initEq.attr))
    else
      push!(newInitEqs, initEq)
    end
  end

  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = traverseExpTopDown(brEq.exp, substituteFoldedVar, foldMap)
        push!(newBranchEqs, typeof(brEq)(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = traverseExpTopDown(branch.condition, substituteFoldedVar, foldMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  local newElimEqs = RESIDUAL_EQUATION[]
  for eq in simCode.eliminatedEquations
    local (newExp, _) = traverseExpTopDown(eq.exp, substituteFoldedVar, foldMap)
    push!(newElimEqs, typeof(eq)(newExp, eq.source, eq.attr))
  end

  #= Sanity guard: scan all surviving surfaces for any folded name. If a
     name still appears (because substituteFoldedVar missed an exotic CREF
     wrapper, or the name is referenced from a code path we did not
     substitute), abort the fold — return the original simCode unchanged.
     Better to do zero folds than to leave a dangling reference that breaks
     codegen (observed on SimpleMechanicalSystem, where `tau_2` survived
     substitution somewhere downstream and produced UndefVarError). =#
  local foldKeys = OrderedSet{String}(keys(foldMap))
  local survivorNames = OrderedSet{String}()
  for eq in newResEqs
    collectCrefNames!(survivorNames, eq.exp)
  end
  for eq in newInitEqs
    if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      collectCrefNames!(survivorNames, eq.exp)
    elseif eq isa BDAE.EQUATION || eq isa EQUATION
      collectCrefNames!(survivorNames, eq.lhs)
      collectCrefNames!(survivorNames, eq.rhs)
    end
  end
  for ifEq in newIfEqs
    for branch in ifEq.branches
      collectCrefNames!(survivorNames, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(survivorNames, brEq.exp)
      end
    end
  end
  for eq in newElimEqs
    collectCrefNames!(survivorNames, eq.exp)
  end
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(survivorNames, whenEq.whenEquation)
  end
  for (_n, (_, sv)) in ht
    @match sv.varKind begin
      PARAMETER(SOME(b)) => collectCrefNames!(survivorNames, b)
      ARRAY_PARAMETER(_, SOME(b)) => collectCrefNames!(survivorNames, b)
      DATA_STRUCTURE(SOME(b)) => collectCrefNames!(survivorNames, b)
      _ => nothing
    end
  end
  local dangling = intersect(foldKeys, survivorNames)
  if !isempty(dangling)
    @debug "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] aborting — $(length(dangling)) folded name(s) still referenced after substitution: $(sort(collect(dangling)))"
    return (simCode, 0)
  end

  local newHT = copy(ht)
  for name in keys(foldMap)
    delete!(newHT, name)
  end

  local elimVarOrder = sort(collect(keys(foldMap)))
  local elimEqOrder  = RESIDUAL_EQUATION[resEqs[defEqOfVar[n]] for n in elimVarOrder]

  @assign begin
    simCode.residualEquations    = newResEqs
    simCode.initialEquations     = newInitEqs
    simCode.ifEquations          = newIfEqs
    simCode.stringToSimVarHT     = newHT
    simCode.eliminatedEquations  = newElimEqs
    simCode.irreducibleVariables = filter(n -> !haskey(foldMap, n), simCode.irreducibleVariables)
  end
  append!(simCode.eliminatedVariables, elimVarOrder)
  append!(simCode.eliminatedEquations, elimEqOrder)
  return (simCode, length(foldMap))
end
