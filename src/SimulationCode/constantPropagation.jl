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

#= Constant equations, pre() of constant parameters, constant propagation. =#

#= Classify `unknown = (+/-) param` from the two extracted (name, cref, type)
   operand results. Shared by the DAE and SIM-native entry points. =#
function _classifyConstEq(@nospecialize(r1), @nospecialize(r2), negated::Bool, ht)
  if r1 === nothing || r2 === nothing
    return nothing
  end
  local (n1, cr1, t1) = r1
  local (n2, cr2, t2) = r2
  if !haskey(ht, n1) || !haskey(ht, n2)
    return nothing
  end
  local (_, sv1) = ht[n1]
  local (_, sv2) = ht[n2]
  local isUnk1 = isUnknownVarKind(sv1.varKind)
  local isUnk2 = isUnknownVarKind(sv2.varKind)
  if !isUnk1 && !isUnk2
    #= Both parameters: trivial equation, always satisfied =#
    return (:trivial, nothing)
  elseif isUnk1 && !isUnk2
    #= n1 is unknown, n2 is parameter: unknown = (+/-)param =#
    return (:constprop, (n1, n2, negated, cr2, t2))
  elseif !isUnk1 && isUnk2
    #= n1 is parameter, n2 is unknown: unknown = (+/-)param =#
    return (:constprop, (n2, n1, negated, cr1, t1))
  else
    #= Both unknowns: handled by alias elimination, not us =#
    return nothing
  end
end

#= SIM-native fast path: avoid building a parallel DAE tree per residual every
   fixpoint round. Only a top-level `+`/`-` of two bare crefs can be a constant
   equation, so bail on cheap `isa` checks; extractCrefName then converts only
   the matched leaf. Equivalent to the DAE path: non-cref operands fail
   extractCrefName and a WILD operand (the one non-EXP_CREF that maps to a
   DAE.CREF) fails the haskey guard. =#
"""
    detectConstantEquation(exp::DAE.Exp, ht)

Detect if a residual equation represents a constant propagation opportunity
or a trivially true equation between parameters.

Returns:
  - `(:trivial, nothing)` if both sides are parameters (equation is tautological)
  - `(:constprop, (unknownName, paramName, negated, paramCref, paramTy))` if one
    side is an unknown and the other is a parameter
  - `nothing` if the equation does not match any constant pattern
"""
function detectConstantEquation(exp::Exp, ht)
  exp isa BINARY || return nothing
  (exp.op === OP_SUB || exp.op === OP_ADD) || return nothing
  (exp.exp1 isa EXP_CREF && exp.exp2 isa EXP_CREF) || return nothing
  return _classifyConstEq(extractCrefName(exp.exp1), extractCrefName(exp.exp2),
                          exp.op === OP_ADD, ht)
end

function detectConstantEquation(@nospecialize(exp), ht)
  @match exp begin
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
      return _classifyConstEq(extractCrefName(e1), extractCrefName(e2), isAdd, ht)
    end
    _ => return nothing
  end
end

"""
    inlinePreOfConstantParameters(simCode::SIM_CODE)::SIM_CODE

Replace `pre(x)` with `x` inside residual equations whenever `x` is a
constant-bound PARAMETER. For a parameter the value at the previous event
is the same as the value now, so this fold is exact and lets downstream
`propagateConstants` resolve the residual naturally.
"""
function inlinePreOfConstantParameters(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode)
    return simCode
  end
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nReplaced = Ref(0)

  #= SIM-native pre(constParam) detector: bail on the SimCode spine; only the
     small pre-arg cref is converted, keyed with the same string(cref) form the
     old DAE path used. =#
  local _isPreConstParam = function(exp)
    exp isa CALL || return false
    length(exp.args) == 1 || return false
    exp.path isa Absyn.IDENT || return false
    (exp.path.name == "pre" || exp.path.name == "previous") || return false
    local arg = exp.args[1]
    arg isa EXP_CREF || return false
    local name = string(toDAECref(arg.cref).componentRef)
    haskey(ht, name) || return false
    local (_, sv) = ht[name]
    return sv.varKind isa PARAMETER
  end

  local _rewrite = function(exp, _)
    if _isPreConstParam(exp)
      nReplaced[] += 1
      return (exp.args[1], false, nothing)
    end
    return (exp, true, nothing)
  end

  local newEqs = RESIDUAL_EQUATION[]
  for eq in resEqs
    #= SIM traverser over eq.exp directly -- no whole-tree toDAEExp; reuse the
       equation when nothing changed. =#
    local (newExp, _) = traverseExpTopDown(eq.exp, _rewrite, nothing)
    push!(newEqs, newExp === eq.exp ? eq : typeof(eq)(newExp, eq.source, eq.attr))
  end
  if nReplaced[] > 0
    @debug "[SIMCODE: $(simCode.name): inlinePreOfConstantParameters] replaced $(nReplaced[]) `pre(constParam)` occurrences with the parameter directly"
  end
  @assign simCode.residualEquations = newEqs
  return simCode
end

"""
    propagateConstants(simCode::SIM_CODE)::SIM_CODE

Constant propagation pass. Detects equations of the form `unknown = parameter`
and substitutes the parameter CREF for the unknown CREF in all equations.
Also removes trivially true `parameter = parameter` equations.

This pass runs BEFORE alias elimination because removing unknowns may reveal
new alias opportunities.

Preserves equation-unknown balance: each constant propagation removes 1 equation
and 1 unknown. Trivial equation removal only removes equations that have no
unknowns (no balance impact).
"""
#= The substitution of the unknowns bound to a parameter or constant, with der() of one 0:
   der(param) kept a state of that derivative from being one of zero derivative
   (Buildings' WaterDerivativeCheck: cpCod = Medium.cp_const, der(cpCod) = der(cpSym)), and
   ModelingToolkit made that state a parameter, without its initial equation. =#
function _substituteConstCref(exp::CALL, constMap)
  if exp.path isa Absyn.IDENT && exp.path.name == "der" && length(exp.args) == 1 && exp.args[1] isa EXP_CREF &&
     haskey(constMap, DAE_identifierToString(toDAECref(exp.args[1].cref).componentRef))
    return (RCONST(0.0), false, constMap)
  end
  return substituteAliasCref(exp, constMap)
end

function _substituteConstCref(@nospecialize(exp), constMap)
  if exp isa DAE.CALL && exp.path isa Absyn.IDENT && exp.path.name == "der" && !listEmpty(exp.expLst) &&
     listHead(exp.expLst) isa DAE.CREF && haskey(constMap, DAE_identifierToString(listHead(exp.expLst).componentRef))
    return (DAE.RCONST(0.0), false, constMap)
  end
  return substituteAliasCref(exp, constMap)
end

function propagateConstants(simCode::SIM_CODE)
  #= Guard: skip for VSS or multi-mode models =#
  if hasStructuralTransitions(simCode) || hasSubModels(simCode)
    return simCode
  end

  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  local sharedVarSet = OrderedSet{String}(simCode.sharedVariables)
  local irreducibleSet = OrderedSet{String}(simCode.irreducibleVariables)

  #= Phase 1: Detect constant equations.
     First collect all base array names referenced in equations so we can skip
     eliminating scalar elements whose base array is still used (e.g. as a
     function call argument). =#
  local allBaseNames = OrderedSet{String}()
  for eq in resEqs
    local eqNames = OrderedSet{String}()
    collectCrefNames!(eqNames, eq.exp)
    for n in eqNames
      if !occursin('[', n)
        push!(allBaseNames, n)
      end
    end
  end

  if !isempty(allBaseNames)
    @debug "[SIMCODE: $(simCode.name): constantPropagation] base array names referenced" allBaseNames=collect(allBaseNames)
  end

  local constMap = Dict{String, Tuple{String, Bool, DAE.ComponentRef, DAE.Type}}()
  local constEqIndices = OrderedSet{Int}()
  local trivialEqIndices = OrderedSet{Int}()
  #= Record eqIdx -> unknown name at detection time. Chained folds only become
     `unknown = param` after a prior substitution, so re-detecting on the
     original equation later would miss them and drop the equation without
     removing its unknown. =#
  local eqIdxToUnknown = Dict{Int, String}()

  local changed = true
  while changed
    changed = false
    for (i, eq) in enumerate(resEqs)
      if i in constEqIndices || i in trivialEqIndices
        continue
      end
      local result = detectConstantEquation(eq.exp, ht)
      if result === nothing
        continue
      end
      local (kind, data) = result
      if kind == :trivial
        push!(trivialEqIndices, i)
        changed = true
      elseif kind == :constprop
        local (unknownName, paramName, negated, paramCref, paramTy) = data
        #= Skip shared or irreducible unknowns =#
        if unknownName in sharedVarSet || unknownName in irreducibleSet
          continue
        end
        #= Skip if the unknown is a subscripted array element whose base name
           is still referenced as a whole array (e.g. in function call arguments).
           Eliminating R_T[1][1] while R_T is passed to resolve2() would break
           code generation which looks up individual elements from the HT. =#
        local bracketIdx = findfirst('[', unknownName)
        if bracketIdx !== nothing
          local baseName = unknownName[1:bracketIdx-1]
          if baseName in allBaseNames
            continue
          end
        end
        if !haskey(constMap, unknownName)
          constMap[unknownName] = (paramName, negated, paramCref, paramTy)
          push!(constEqIndices, i)
          eqIdxToUnknown[i] = unknownName
          changed = true
        end
      end
    end
    if changed && !isempty(constMap)
      #= Apply current substitutions to all remaining equation expressions
         so that chained constant patterns are revealed in the next iteration =#
      local updatedEqs = RESIDUAL_EQUATION[]
      for (i, eq) in enumerate(resEqs)
        local (newExp, _) = traverseExpTopDown(eq.exp, _substituteConstCref, constMap)
        push!(updatedEqs, typeof(eq)(newExp, eq.source, eq.attr))
      end
      resEqs = updatedEqs
    end
  end

  local nConst = length(constEqIndices)
  local nTrivial = length(trivialEqIndices)
  if nConst == 0 && nTrivial == 0
    @debug "[SIMCODE: $(simCode.name): constantPropagation] no constant equations found"
    return simCode
  end

  @debug "[SIMCODE: $(simCode.name): constantPropagation] found $nConst unknown=param equations and $nTrivial trivial param=param equations"

  #= Phase 2: Build final equation list with substitutions applied.
     We collect (varName, original-residual, substituted-residual) triples for
     const-bound eliminations so the downstream `eliminatedEquations` /
     `eliminatedVariables` arrays stay aligned, and so Phase 3 can re-add the
     substituted residual if the unknown turns out not to be eliminable.
     Trivial `param=param` residuals are dropped without recording since they
     have no unknown to associate. =#
  local allRemoved = union(constEqIndices, trivialEqIndices)
  local newResEqs = RESIDUAL_EQUATION[]
  local elimPairs = Tuple{String, RESIDUAL_EQUATION, RESIDUAL_EQUATION}[]
  sizehint!(newResEqs, nEqs - length(allRemoved))
  #= Collect surviving cref names from the substituted expressions as we build
     them, avoiding a second full traversal (and toDAEExp reconversion) in
     Phase 3. =#
  local allRefNames = OrderedSet{String}()

  for (i, eq) in enumerate(simCode.residualEquations)
    if i in allRemoved
      if i in constEqIndices && haskey(eqIdxToUnknown, i)
        local (subExp, _) = traverseExpTopDown(eq.exp, _substituteConstCref, constMap)
        push!(elimPairs, (eqIdxToUnknown[i], eq, typeof(eq)(subExp, eq.source, eq.attr)))
      end
    else
      local (newExp, _) = traverseExpTopDown(eq.exp, _substituteConstCref, constMap)
      collectCrefNames!(allRefNames, newExp)
      push!(newResEqs, typeof(eq)(newExp, eq.source, eq.attr))
    end
  end

  #= Substitute in if-equation branches =#
  local newIfEqs = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = traverseExpTopDown(brEq.exp, _substituteConstCref, constMap)
        collectCrefNames!(allRefNames, newBrExp)
        push!(newBranchEqs, typeof(brEq)(newBrExp, brEq.source, brEq.attr))
      end
      local (newCond, _) = traverseExpTopDown(branch.condition, _substituteConstCref, constMap)
      push!(newBranches, BRANCH(newCond, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEqs, IF_EQUATION(newBranches))
  end

  #= Substitute propagated constants in when-equation CONDITIONS only. The
     bodies are deliberately NOT passed through _substituteAliasInWhenStmts here
     (unlike eliminateAliasVariables): that helper also rewrites the ASSIGN/REINIT
     LHS, and substituting a CONSTANT into an assignment target is invalid. The
     survivor scan below keeps any unknown read only from a when body, so it is
     not dropped and does not dangle, even though it stays un-substituted. =#
  local newWhenEqs = WHEN_EQUATION[]
  for whenEq in simCode.whenEquations
    local innerWhen = whenEq.whenEquation
    local (newCond, _) = traverseExpTopDown(innerWhen.condition, _substituteConstCref, constMap)
    @assign innerWhen.condition = toSimExp(newCond)
    @assign whenEq.whenEquation = innerWhen
    push!(newWhenEqs, whenEq)
  end

  #= Substitute in initial equations =#
  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION || initEq isa RESIDUAL_EQUATION
      local (newInitExp, _) = Util.traverseExpTopDown(toDAEExp(initEq.exp), _substituteConstCref, constMap)
      collectCrefNames!(allRefNames, newInitExp)
      push!(newInitEqs, typeof(initEq)(newInitExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), _substituteConstCref, constMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), _substituteConstCref, constMap)
      collectCrefNames!(allRefNames, newLhs)
      collectCrefNames!(allRefNames, newRhs)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    elseif initEq isa EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), _substituteConstCref, constMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), _substituteConstCref, constMap)
      collectCrefNames!(allRefNames, newLhs)
      collectCrefNames!(allRefNames, newRhs)
      push!(newInitEqs, EQUATION(newLhs, newRhs, initEq.source, initEq.attr))
    else
      push!(newInitEqs, initEq)
    end
  end

  #= Phase 3: Verify and remove eliminated unknowns. Surviving refs from
     residual/if/init expressions were collected inline during substitution. =#
  local eliminatedSet = OrderedSet{String}(keys(constMap))

  #= Also scan when-equation conditions and statement bodies for surviving
     references (mirrors eliminateAliasVariables). Without this, a constant-bound
     unknown read only inside a when body is judged non-surviving and removed
     from the HT while still referenced. =#
  for whenEq in newWhenEqs
    _collectWhenCrefNames!(allRefNames, whenEq.whenEquation)
  end

  local survivingRefs = OrderedSet{String}()
  for n in allRefNames
    if n in eliminatedSet
      push!(survivingRefs, n)
    end
  end

  if !isempty(survivingRefs)
    @warn "[SIMCODE: $(simCode.name): constantPropagation] $(length(survivingRefs)) eliminated variables still referenced, keeping them" survivingRefs=collect(survivingRefs)
  end

  local newHT = copy(ht)
  #= Build elimVarNames from elimPairs (same order as elimEqs) and
     drop any pairs whose variable is in survivingRefs. This keeps
     `eliminatedEquations` and `eliminatedVariables` aligned for
     downstream `generateEliminatedObservedBlock`. A pair whose unknown is
     NOT eliminated (still referenced, or already gone from the HT) has its
     substituted residual re-added to `newResEqs`: the equation was removed
     in Phase 2 on the assumption the unknown would go with it, so keeping the
     unknown without the equation would unbalance the system. =#
  local elimVarNames = String[]
  local keptElimEqs = RESIDUAL_EQUATION[]
  #= A kept unknown's equation goes back with the other substitutions only:
     with its own as well, `y - u` (y and u bound to R) reads `R - R` and no
     longer defines u. =#
  local keptMap = isempty(survivingRefs) ? constMap : filter(kv -> !(first(kv) in survivingRefs), constMap)
  for (varName, origEq, subEq) in elimPairs
    if varName in survivingRefs
      local (keptExp, _) = traverseExpTopDown(origEq.exp, _substituteConstCref, keptMap)
      push!(newResEqs, typeof(origEq)(keptExp, origEq.source, origEq.attr))
      continue
    elseif !haskey(newHT, varName)
      push!(newResEqs, subEq)
      continue
    end
    delete!(newHT, varName)
    push!(elimVarNames, varName)
    push!(keptElimEqs, origEq)
  end
  local elimEqs = keptElimEqs

  @debug "[SIMCODE: $(simCode.name): constantPropagation] eliminated $(length(elimVarNames)) unknowns and $(length(elimVarNames)) equations ($(length(newResEqs)) equations, $(length(newHT)) variables remain)"

  @assign begin
    simCode.residualEquations = newResEqs
    simCode.initialEquations = newInitEqs
    simCode.stringToSimVarHT = newHT
    simCode.ifEquations = newIfEqs
    simCode.whenEquations = newWhenEqs
    simCode.asserts = _substituteInAsserts(simCode.asserts, constMap)
  end
  append!(simCode.eliminatedEquations, elimEqs)
  append!(simCode.eliminatedVariables, elimVarNames)
  return simCode
end
