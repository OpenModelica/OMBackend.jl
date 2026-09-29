#= States pinned to a constant (`0 = state - literal`), with numeric folding. =#

#= True if exp is a literal 1.0 or integer 1. =#
function _isOneLiteral(@nospecialize(exp))
  @match exp begin
    DAE.RCONST(x) => x == 1.0
    DAE.ICONST(x) => x == 1
    _ => false
  end
end

#= Return the numeric value of a literal, or nothing if not a literal. =#
function _extractNumericValue(@nospecialize(exp))
  @match exp begin
    DAE.RCONST(x) => x
    DAE.ICONST(x) => Float64(x)
    DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => begin
      local v = _extractNumericValue(inner)
      v === nothing ? nothing : -v
    end
    _ => nothing
  end
end

# SIM-native dispatch.
_extractNumericValue(e::RCONST) = e.value
_extractNumericValue(e::ICONST) = Float64(e.value)
function _extractNumericValue(e::UNARY)
  e.op === OP_UMINUS || return nothing
  local v = _extractNumericValue(e.exp)
  return v === nothing ? nothing : -v
end
_extractNumericValue(e::Exp) = nothing

#= Fold numeric subexpressions in a DAE.Exp tree. Bottom-up evaluation:
   when both operands of a BINARY are numeric literals, replace with the
   evaluated result; partial-eval `0 * x`, `x * 0` to `RCONST(0)` and
   `0 + x`, `x + 0`, `x - 0` to the surviving operand.

   Used after frozen-state substitution so that residuals like
     `0 = -phasor_i_[2] - (0.0 * 0.0 + 0.5773 * 0.0)`
   collapse to `0 = -phasor_i_[2] - 0.0`, exposing a new pin in the next
   iteration of `eliminateFrozenStates`.

   Conservative: does not fold DIV by zero, sin/cos/exp of constants
   (correctness OK but produces UNARY-RCONST forms that downstream code
   may not expect). =#
function _foldNumericExp(@nospecialize(exp))
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local f1 = _foldNumericExp(e1)
      local f2 = _foldNumericExp(e2)
      local v1 = _extractNumericValue(f1)
      local v2 = _extractNumericValue(f2)
      if v1 !== nothing && v2 !== nothing
        @match op begin
          DAE.ADD(__) => return DAE.RCONST(v1 + v2)
          DAE.SUB(__) => return DAE.RCONST(v1 - v2)
          DAE.MUL(__) => return DAE.RCONST(v1 * v2)
          DAE.DIV(__) => begin
            v2 != 0 && return DAE.RCONST(v1 / v2)
          end
          _ => nothing
        end
      end
      #= Partial folds: 0 * x = 0, 1 * x = x, x * 0 = 0, x * 1 = x,
         0 + x = x, x + 0 = x, x - 0 = x, 0 - x = -x, x / 1 = x.
         Plus the structural tautology x - x = 0 (catches alias-substituted
         residuals that became `0 = c - c` after elimination). =#
      @match op begin
        DAE.MUL(__) => begin
          (v1 !== nothing && v1 == 0) && return DAE.RCONST(0.0)
          (v2 !== nothing && v2 == 0) && return DAE.RCONST(0.0)
          (v1 !== nothing && v1 == 1) && return f2
          (v2 !== nothing && v2 == 1) && return f1
        end
        DAE.ADD(__) => begin
          (v1 !== nothing && v1 == 0) && return f2
          (v2 !== nothing && v2 == 0) && return f1
        end
        DAE.SUB(__) => begin
          (v2 !== nothing && v2 == 0) && return f1
        end
        DAE.DIV(__) => begin
          (v2 !== nothing && v2 == 1) && return f1
        end
        _ => nothing
      end
      return DAE.BINARY(f1, op, f2)
    end
    DAE.UNARY(operator = op, exp = inner) => begin
      local fin = _foldNumericExp(inner)
      local vin = _extractNumericValue(fin)
      if vin !== nothing
        @match op begin
          DAE.UMINUS(__) => return DAE.RCONST(-vin)
          _ => nothing
        end
      end
      return DAE.UNARY(op, fin)
    end
    DAE.CALL(Absyn.IDENT("der"), expLst, _) => begin
      local arg = listHead(expLst)
      local fin = _foldNumericExp(arg)
      _extractNumericValue(fin) !== nothing && return DAE.RCONST(0.0)
      return exp
    end
    _ => exp
  end
end

# SIM-native dispatch: mirrors the DAE folder over SC.Exp variants.
function _foldNumericExp(e::BINARY)
  local f1 = _foldNumericExp(e.exp1)
  local f2 = _foldNumericExp(e.exp2)
  local v1 = _extractNumericValue(f1)
  local v2 = _extractNumericValue(f2)
  if v1 !== nothing && v2 !== nothing
    e.op === OP_ADD && return RCONST(v1 + v2)
    e.op === OP_SUB && return RCONST(v1 - v2)
    e.op === OP_MUL && return RCONST(v1 * v2)
    (e.op === OP_DIV && v2 != 0) && return RCONST(v1 / v2)
  end
  if e.op === OP_MUL
    (v1 !== nothing && v1 == 0) && return RCONST(0.0)
    (v2 !== nothing && v2 == 0) && return RCONST(0.0)
    (v1 !== nothing && v1 == 1) && return f2
    (v2 !== nothing && v2 == 1) && return f1
  elseif e.op === OP_ADD
    (v1 !== nothing && v1 == 0) && return f2
    (v2 !== nothing && v2 == 0) && return f1
  elseif e.op === OP_SUB
    (v2 !== nothing && v2 == 0) && return f1
  elseif e.op === OP_DIV
    (v2 !== nothing && v2 == 1) && return f1
  end
  return BINARY(f1, e.op, f2)
end

function _foldNumericExp(e::UNARY)
  local fin = _foldNumericExp(e.exp)
  local vin = _extractNumericValue(fin)
  (vin !== nothing && e.op === OP_UMINUS) && return RCONST(-vin)
  return UNARY(e.op, fin)
end

function _foldNumericExp(e::CALL)
  local fnName = @match e.path begin
    Absyn.IDENT(n) => n
    _ => ""
  end
  if fnName == "der" && !isempty(e.args)
    local fin = _foldNumericExp(e.args[1])
    _extractNumericValue(fin) !== nothing && return RCONST(0.0)
  end
  return e
end

_foldNumericExp(e::Exp) = e

#= Peel structurally-trivial wrappers around a sub-expression. Used by
   `_detectFrozenState` so equations emitted with redundant `* 1.0` or
   `--` decorations (common from inlining / parameter folding) still match
   the frozen pin pattern. Conservative: stops at the first non-peelable
   layer, so partial wrappers (e.g. `2.0 * x`) are left intact. =#
function _peelNoOpWrappers(@nospecialize(exp))
  local prev
  while true
    prev = exp
    @match exp begin
      DAE.BINARY(exp1 = e1, operator = DAE.MUL(__), exp2 = e2) => begin
        if _isOneLiteral(e2)
          exp = e1
        elseif _isOneLiteral(e1)
          exp = e2
        end
      end
      DAE.BINARY(exp1 = e1, operator = DAE.DIV(__), exp2 = e2) => begin
        if _isOneLiteral(e2)
          exp = e1
        end
      end
      DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => begin
        @match inner begin
          DAE.UNARY(operator = DAE.UMINUS(__), exp = innerInner) => begin
            exp = innerInner
          end
          _ => nothing
        end
      end
      _ => nothing
    end
    exp === prev && break
  end
  return exp
end

#= True if exp is a numeric literal (optionally wrapped in unary minus or
   no-op multiplications by 1). =#
function _isNumericLiteral(@nospecialize(exp))
  local peeled = _peelNoOpWrappers(exp)
  @match peeled begin
    DAE.RCONST(__) => true
    DAE.ICONST(__) => true
    DAE.UNARY(operator = DAE.UMINUS(__),     exp = inner) => _isNumericLiteral(inner)
    DAE.UNARY(operator = DAE.UMINUS_ARR(__), exp = inner) => _isNumericLiteral(inner)
    _ => false
  end
end

#= Detect a residual of the form `0 = var - literal` (or `0 = literal - var`)
   where `var` is structurally pinned to a constant. Eligible varKinds are
   STATE (the original kinematic-ground case, e.g. AIMC stator phi=0) and
   ALG_VARIABLE (post-parameter-elimination cases, e.g. AIMC R_actual=0.03
   after the alpha*(T-T_ref) term folds to zero). Returns
   (name, cref, ty, literalExp, isState) or nothing.

   STATE eligibility is what enables the `der(state) -> 0` substitution.
   ALG_VARIABLE is structurally identical for substitution (no derivative
   to handle). DISCRETE / ARRAY are excluded because they carry event or
   subscript semantics. =#
#= Extract a CREF together with its sign within a residual term.
   Returns (name, cref, ty, sign) where sign is +1 for bare CREF, -1 for
   UNARY(UMINUS, CREF). Also peels `* 1.0` / `/1.0` / `--` wrappers
   first so decorated forms like `var * 1.0` still match.
   Returns nothing if the term is anything else (multi-coefficient,
   non-leaf, etc.). =#
function _extractCrefSigned(@nospecialize(exp))
  local peeled = _peelNoOpWrappers(exp)
  @match peeled begin
    DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => begin
      local innerPeeled = _peelNoOpWrappers(inner)
      local r = extractCrefName(innerPeeled)
      r === nothing && return nothing
      local (n, cr, ty) = r
      return (n, cr, ty, -1)
    end
    _ => begin
      local r = extractCrefName(peeled)
      r === nothing && return nothing
      local (n, cr, ty) = r
      return (n, cr, ty, 1)
    end
  end
end

#= Negate a numeric literal expression, preserving its DAE structure when
   trivially possible (RCONST/ICONST get value-negated; anything else gets
   wrapped in UNARY(UMINUS)). =#
function _negateLiteralExp(@nospecialize(litExp))
  @match litExp begin
    DAE.RCONST(x) => DAE.RCONST(-x)
    DAE.ICONST(x) => DAE.ICONST(-x)
    DAE.UNARY(operator = DAE.UMINUS(__), exp = inner) => inner
    _ => DAE.UNARY(DAE.UMINUS(DAE.T_REAL_DEFAULT), litExp)
  end
end

function _detectFrozenState(@nospecialize(exp), ht)
  @match exp begin
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local isSub = @match op begin
        DAE.SUB(__) => true
        _ => false
      end
      isSub || return nothing
      local e1p = _peelNoOpWrappers(e1)
      local e2p = _peelNoOpWrappers(e2)
      local s1 = _extractCrefSigned(e1)
      local s2 = _extractCrefSigned(e2)
      local stateRef, litExp, varSign
      #= Equation form `s1*var - lit = 0` => var = lit/s1.
         Equation form `lit - s2*var = 0` => var = lit/s2. =#
      if s1 !== nothing && _isNumericLiteral(e2p)
        local (n, cr, ty, sg) = s1
        stateRef = (n, cr, ty); litExp = e2p; varSign = sg
      elseif s2 !== nothing && _isNumericLiteral(e1p)
        local (n, cr, ty, sg) = s2
        stateRef = (n, cr, ty); litExp = e1p; varSign = sg
      else
        return nothing
      end
      if varSign == -1
        litExp = _negateLiteralExp(litExp)
      end
      local (n, cr, ty) = stateRef
      haskey(ht, n) || return nothing
      local (_, sv) = ht[n]
      local isState = @match sv.varKind begin
        STATE(__) => true
        _ => false
      end
      local isAlg = @match sv.varKind begin
        ALG_VARIABLE(__) => true
        _ => false
      end
      (isState || isAlg) || return nothing
      return (n, cr, ty, litExp, isState)
    end
    #= Residual exp collapsed to a single CREF or UMINUS(CREF) after fold:
       `0 = w` or `0 = -w` both mean w = 0. This shape appears in round 2+
       of eliminateFrozenStates after `w - der(phi)` substitutes der(phi)→0
       and `w - 0` folds to bare `w`. Allowed for ALG_VARIABLE and STATE
       (state pin from an aliased connector). The originator equation has
       already pinned this variable to a single literal — preserving it
       would leave a CREF that MTK structural_simplify rejects as
       "present in the system but not an unknown". =#
    _ => begin
      local s = _extractCrefSigned(exp)
      s === nothing && return nothing
      local (n, cr, ty, _sg) = s
      haskey(ht, n) || return nothing
      local (_, sv) = ht[n]
      local isAlg = @match sv.varKind begin
        ALG_VARIABLE(__) => true
        _ => false
      end
      local isState = @match sv.varKind begin
        STATE(__) => true
        _ => false
      end
      (isAlg || isState) || return nothing
      return (n, cr, ty, DAE.RCONST(0.0), isState)
    end
  end
end

#= traverseExpTopDown visitor: substitute eliminated states. Returns
   (newExp, continueRecursion, frozenMap). Handles two patterns:
     - CREF(state)                        -> literal
     - CALL("der", [CREF(state)])         -> 0.0
   For non-frozen subtrees, returns the original exp with continueRecursion=true. =#
function _substituteFrozenState(@nospecialize(exp), frozenMap)
  @match exp begin
    DAE.CALL(Absyn.IDENT("der"), expLst, _) => begin
      local arg = listHead(expLst)
      @match arg begin
        DAE.CREF(cr, _) => begin
          local n = DAE_identifierToString(cr)
          if haskey(frozenMap, n)
            return (DAE.RCONST(0.0), false, frozenMap)
          end
          return (exp, true, frozenMap)
        end
        _ => return (exp, true, frozenMap)
      end
    end
    DAE.CREF(cr, _) => begin
      local n = DAE_identifierToString(cr)
      if haskey(frozenMap, n)
        return (frozenMap[n], false, frozenMap)
      end
      return (exp, true, frozenMap)
    end
    _ => return (exp, true, frozenMap)
  end
end

#= SIM-native dispatch: der(frozen state) -> 0, frozen cref -> its (DAE) value
   converted to SIM; only the matched leaf converts. =#
function _substituteFrozenState(exp::CALL, frozenMap)
  local fnName = @match exp.path begin
    Absyn.IDENT(n) => n
    _ => ""
  end
  if fnName == "der" && !isempty(exp.args) && exp.args[1] isa EXP_CREF
    local n = DAE_identifierToString(toDAECref(exp.args[1].cref).componentRef)
    haskey(frozenMap, n) && return (RCONST(0.0), false, frozenMap)
  end
  return (exp, true, frozenMap)
end

function _substituteFrozenState(exp::EXP_CREF, frozenMap)
  local n = DAE_identifierToString(toDAECref(exp.cref).componentRef)
  if haskey(frozenMap, n)
    return (toSimExp(frozenMap[n]), false, frozenMap)
  end
  return (exp, true, frozenMap)
end

_substituteFrozenState(exp::Exp, frozenMap) = (exp, true, frozenMap)

#= Eliminate variables that are algebraically pinned to a numeric literal.
   Two flavours, both covered:

   1. STATE pinned by a kinematic ground (e.g. AIMC `aimc_inertiaStator_phi = 0`
      from a Fixed-flange). The state has no time dynamics yet stays classified
      as STATE because `der(state)` appears in some inertia/connector equation.
      Pantelides then differentiates the pin and over-determines the system.
   2. ALG_VARIABLE pinned by a folded parameter expression (e.g. AIMC
      `aimc_rs_resistor[k]_R_actual = 0.03` after `R*(1 + alpha*(T-T_ref))`
      collapses with alpha=0). Treated identically — no derivative to handle,
      but the CREF substitution propagates the constant through every use.

   Strategy: full elimination. Substitute the variable with its literal value
   at every CREF site, and `der(state) -> 0.0` for the STATE case. The pin
   equation is dropped; the (var, eq) pair moves into eliminatedVariables /
   eliminatedEquations so MTK observed-equation generation can still expose
   the constant value on sol[:name].

   Excluded varKinds: DISCRETE (event semantics), ARRAY (subscript handling),
   STATE_DERIVATIVE (not a directly-pinnable form).

   Safety: never eliminate a variable that appears in any if-branch or
   when-equation (its name is needed for event registration / callback
   pre()-tracking). Skips for VSS / multi-mode SimCode variants.

   Iteration: substituting der(state) -> 0 can expose a new frozen variable
   in equations like `w - der(state) = 0` (becomes `w - 0 = 0`). The pass
   loops until no more matches surface, capped at 16 rounds defensively.

   Placement: after eliminateConstantParameters so parameter chains like
   `var = some_param` (with param folded to a literal) are already
   substituted to `var = literal` form before detection. =#
function eliminateFrozenStates(simCode::SIM_CODE)::SIM_CODE
  if hasSubModels(simCode) || hasMetaModel(simCode)
    return simCode
  end
  #= Iterate to convergence: substituting der(state) -> 0 can turn a related
     equation like `w - der(state) = 0` into `w - 0 = 0`, exposing a new
     frozen state. Cap the loop count defensively even though the variable
     set strictly shrinks each round. =#
  #= protectedNames is invariant across rounds (if/when equations do not
     change), so compute it once and reuse. =#
  local protectedNames = _computeFrozenProtectedNames(simCode)
  local totalEliminated = 0
  local maxRounds = 16
  for round in 1:maxRounds
    local (newCode, nEliminated) = _eliminateFrozenStatesOnePass(simCode, protectedNames)
    nEliminated == 0 && break
    simCode = newCode
    totalEliminated += nEliminated
  end
  return simCode
end

function _computeFrozenProtectedNames(simCode::SIM_CODE)::OrderedSet{String}
  local protectedNames = OrderedSet{String}()
  local ht = simCode.stringToSimVarHT
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      collectCrefNames!(protectedNames, branch.condition)
      for brEq in branch.residualEquations
        collectCrefNames!(protectedNames, brEq.exp)
      end
    end
  end
  for whenEq in simCode.whenEquations
    _collectWhenCrefNames!(protectedNames, whenEq.whenEquation)
  end
  #= Protect scalar `_re`/`_im` fields of any surviving Complex CREF.
     foldExplicitSingleAssign would otherwise fold `coilQS_Psi_re` and
     `coilQS_Psi_im` (definitional residuals after Complex-record
     expansion) while the parent `coilQS_Psi` CREF still appears in
     another equation; codegen later flattens the parent into the two
     scalar siblings and fails with UndefVarError at module eval. =#
  _collectComplexFieldNames!(protectedNames, simCode.residualEquations, ht)
  _collectComplexFieldNames!(protectedNames, simCode.initialEquations, ht)
  for ifEq in simCode.ifEquations
    for branch in ifEq.branches
      _collectComplexFieldNames!(protectedNames, branch.residualEquations, ht)
    end
  end
  for eq in simCode.eliminatedEquations
    _collectComplexFieldNames!(protectedNames, [eq], ht)
  end
  return protectedNames
end

function _eliminateFrozenStatesOnePass(simCode::SIM_CODE, protectedNames::OrderedSet{String})
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local sharedVarSet = OrderedSet{String}(simCode.sharedVariables)

  local frozenMap   = Dict{String, DAE.Exp}()
  local frozenEqIdx = Dict{String, Int}()
  local frozenIsState = Dict{String, Bool}()
  for (i, eq) in enumerate(resEqs)
    local pair = _detectFrozenState(toDAEExp(eq.exp), ht)
    pair === nothing && continue
    local (n, _, _, litExp, isState) = pair
    n in sharedVarSet  && continue
    n in protectedNames && continue
    haskey(frozenMap, n) && continue
    frozenMap[n]   = litExp
    frozenEqIdx[n] = i
    frozenIsState[n] = isState
  end

  isempty(frozenMap) && return (simCode, 0)

  #= Safety: never reduce the residual list to empty. MTK's `System(...)`
     constructor infers `Vector{Any}` from an empty literal `[]`, which
     does not match the typed-vector method signatures and raises
     MethodError at codegen time. If eliminating all detected frozen
     variables would empty the residual set, keep one of them so MTK
     still has a non-empty (but trivial) equation to construct from.
     Observed on MatrixMultTest where every variable is a constant pin. =#
  local _eqsLeftAfter = length(resEqs) - length(frozenMap)
  if _eqsLeftAfter <= 0
    local _keepOne = first(sort(collect(keys(frozenMap))))
    delete!(frozenMap, _keepOne)
    delete!(frozenEqIdx, _keepOne)
    delete!(frozenIsState, _keepOne)
    @info "[SIMCODE: $(simCode.name): eliminateFrozenStates] keeping $(_keepOne) to avoid emptying the residual system"
    isempty(frozenMap) && return (simCode, 0)
  end

  local nState = count(values(frozenIsState))
  local nAlg   = length(frozenMap) - nState
  @info "[SIMCODE: $(simCode.name): eliminateFrozenStates] eliminating $(length(frozenMap)) frozen variable(s) ($nState state, $nAlg algebraic): $(sort(collect(keys(frozenMap))))"
  if OMBackend.BACKEND_PERFLOG[]
    @info "[SIMCODE: $(simCode.name): eliminateFrozenStates] model size" residuals_before=length(resEqs) residuals_after=length(resEqs) - length(frozenMap) variables_before=length(ht) variables_after=length(ht) - length(frozenMap)
  end

  local removeEqs = OrderedSet{Int}(values(frozenEqIdx))
  local newResEqs = RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(resEqs) - length(removeEqs))
  for (i, eq) in enumerate(resEqs)
    i in removeEqs && continue
    local (newExp, _) = traverseExpTopDown(eq.exp, _substituteFrozenState, frozenMap)
    #= Constant-fold after substitution: `0.0 * x` and friends now reduce
       to 0 so the residual becomes a clean `0 = -y - 0` form that the
       next iteration can detect as a pin. =#
    newExp = _foldNumericExp(newExp)
    push!(newResEqs, typeof(eq)(newExp, eq.source, eq.attr))
  end

  local newInitEqs = typeof(simCode.initialEquations)()
  for initEq in simCode.initialEquations
    if initEq isa BDAE.RESIDUAL_EQUATION || initEq isa RESIDUAL_EQUATION
      local (newExp, _) = Util.traverseExpTopDown(initEq.exp, _substituteFrozenState, frozenMap)
      newExp = _foldNumericExp(newExp)
      push!(newInitEqs, typeof(initEq)(newExp, initEq.source, initEq.attr))
    elseif initEq isa BDAE.EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), _substituteFrozenState, frozenMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), _substituteFrozenState, frozenMap)
      newLhs = _foldNumericExp(newLhs)
      newRhs = _foldNumericExp(newRhs)
      push!(newInitEqs, BDAE.EQUATION(newLhs, newRhs, initEq.source, initEq.attributes))
    elseif initEq isa EQUATION
      local (newLhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.lhs), _substituteFrozenState, frozenMap)
      local (newRhs, _) = Util.traverseExpTopDown(toDAEExp(initEq.rhs), _substituteFrozenState, frozenMap)
      newLhs = _foldNumericExp(newLhs)
      newRhs = _foldNumericExp(newRhs)
      push!(newInitEqs, EQUATION(newLhs, newRhs, initEq.source, initEq.attr))
    else
      push!(newInitEqs, initEq)
    end
  end

  local newHT = copy(ht)
  for n in keys(frozenMap)
    delete!(newHT, n)
  end

  #= Parallel arrays: variable order matches paired equation order. =#
  local elimVarOrder = sort(collect(keys(frozenMap)))
  local elimEqOrder  = RESIDUAL_EQUATION[resEqs[frozenEqIdx[n]] for n in elimVarOrder]

  @assign begin
    simCode.residualEquations     = newResEqs
    simCode.initialEquations      = newInitEqs
    simCode.stringToSimVarHT      = newHT
    simCode.irreducibleVariables = filter(n -> !haskey(frozenMap, n), simCode.irreducibleVariables)
  end
  append!(simCode.eliminatedVariables,  elimVarOrder)
  append!(simCode.eliminatedEquations,  elimEqOrder)
  return (simCode, length(frozenMap))
end
