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

#= Variables pinned to a literal (`0 = x - literal`), eliminated with numeric folding. =#

isOneLiteral(e::Exp)::Bool = (e isa RCONST && e.value == 1.0) || (e isa ICONST && e.value == 1)

"`e` without the no-op wrappers `* 1`, `1 *`, `/ 1` and `- -` (inlining and parameter folding leave them)."
function withoutNoOps(e::Exp)::Exp
  while true
    if e isa BINARY && e.op === OP_MUL && (isOneLiteral(e.exp1) || isOneLiteral(e.exp2))
      e = isOneLiteral(e.exp2) ? e.exp1 : e.exp2
    elseif e isa BINARY && e.op === OP_DIV && isOneLiteral(e.exp2)
      e = e.exp1
    elseif e isa UNARY && e.op === OP_UMINUS && e.exp isa UNARY && e.exp.op === OP_UMINUS
      e = e.exp.exp
    else
      return e
    end
  end
end

"Whether `e` is a numeric literal (under unary minuses and no-op wrappers)."
function isNumericLiteral(e::Exp)::Bool
  local p = withoutNoOps(e)
  p isa Union{RCONST, ICONST} && return true
  return p isa UNARY && p.op === OP_UMINUS && isNumericLiteral(p.exp)
end

"A variable reference (under no-op wrappers and one unary minus), and its sign; `nothing` for another expression."
function signedVariable(e::Exp)::Union{Nothing, Tuple{EXP_CREF, Int}}
  local p = withoutNoOps(e)
  if p isa UNARY && p.op === OP_UMINUS
    local inner = withoutNoOps(p.exp)
    return inner isa EXP_CREF ? (inner, -1) : nothing
  end
  return p isa EXP_CREF ? (p, 1) : nothing
end

"""
    frozenVariable(e, ht)

The state or algebraic variable a residual pins to a literal, and the literal: `±x - c` or
`c - ±x` (`x = ±c`), or a residual that is the variable itself, `x` or `-x` (`x = 0`: what
`w - der(phi)` becomes when `phi` is frozen); `nothing` for another residual.
"""
function frozenVariable(e::Exp, ht::HT)::Union{Nothing, Tuple{String, Exp}}
  local pin = if e isa BINARY
    e.op === OP_SUB || return nothing
    local (p1, p2) = (withoutNoOps(e.exp1), withoutNoOps(e.exp2))
    local (s1, s2) = (signedVariable(e.exp1), signedVariable(e.exp2))
    if s1 !== nothing && isNumericLiteral(p2)
      (s1[1], s1[2] == -1 ? negatedValue(p2) : p2)
    elseif s2 !== nothing && isNumericLiteral(p1)
      (s2[1], s2[2] == -1 ? negatedValue(p1) : p1)
    else
      return nothing
    end
  else
    local s = signedVariable(e)
    s === nothing && return nothing
    (s[1], RCONST(0.0))
  end
  local name = crefName(pin[1].cref)
  local entry = get(ht, name, nothing)
  (entry !== nothing && last(entry).varKind isa Union{STATE, ALG_VARIABLE}) || return nothing
  return (name, pin[2])
end

"""
    foldNumeric(e)

`e` with its arithmetic of literals folded (through sums, differences, products, quotients
and negations only): `+ - * /` of two literals (not by zero) to a Real, `0 * x` to `0.0`,
`1 * x`, `x + 0`, `x - 0`, `x / 1` to `x`.
"""
function foldNumeric(e::Exp)::Exp
  if e isa BINARY
    local (f1, f2) = (foldNumeric(e.exp1), foldNumeric(e.exp2))
    local (v1, v2) = (literalValue(f1), literalValue(f2))
    if v1 !== nothing && v2 !== nothing
      e.op === OP_ADD && return RCONST(v1 + v2)
      e.op === OP_SUB && return RCONST(v1 - v2)
      e.op === OP_MUL && return RCONST(v1 * v2)
      (e.op === OP_DIV && v2 != 0) && return RCONST(v1 / v2)
    end
    if e.op === OP_MUL
      (v1 == 0 || v2 == 0) && return RCONST(0.0)
      v1 == 1 && return f2
      v2 == 1 && return f1
    elseif e.op === OP_ADD
      v1 == 0 && return f2
      v2 == 0 && return f1
    elseif e.op === OP_SUB
      v2 == 0 && return f1
    elseif e.op === OP_DIV
      v2 == 1 && return f1
    end
    return BINARY(f1, e.op, f2)
  elseif e isa UNARY
    local f = foldNumeric(e.exp)
    local v = literalValue(f)
    (v !== nothing && e.op === OP_UMINUS) && return RCONST(-v)
    return UNARY(e.op, f)
  end
  return e
end

"The value of a numeric literal (under unary minuses); `nothing` for another expression."
function literalValue(e::Exp)::Union{Nothing, Float64}
  e isa RCONST && return e.value
  e isa ICONST && return Float64(e.value)
  if e isa UNARY && e.op === OP_UMINUS
    local v = literalValue(e.exp)
    return v === nothing ? nothing : -v
  end
  return nothing
end

"""
    eliminateFrozenStates(simCode)

The states and algebraic variables pinned to a literal (`frozenVariable` of the residual with
its arithmetic of literals folded: a parameter fold leaves MSL FluidHeatFlow's zero friction
loss `Q_flow - 0.0 * V_flow * dp`, whose energy balance then joins the semiLinear chain) replaced by it
everywhere (`der()` of a frozen state by 0), their residuals removed and recorded for the
observed equations, the residuals folded (`foldNumeric`); to a fixed point (at most 16
rounds: `w - der(phi)` pins `w` once `phi` is frozen). A Kinematic ground leaves a state
without dynamics (AIMC's stator phi = 0) that MTK's index reduction would differentiate; a
parameter fold leaves an algebraic one constant. Not a shared variable, one an if- or
when-equation reads (event registration), a Complex variable's part; never every residual
(MTK takes no empty system: one stays). Nothing for a model with modes or translated again
at its structural changes.
"""
function eliminateFrozenStates(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local kept = eventAndComplexNames(simCode)
  for _ in 1:16
    local next = eliminateFrozenOnce(simCode, kept)
    next === nothing && break
    simCode = next
  end
  return simCode
end

"""
    eventAndComplexNames(simCode)

The names the if- and when-equations read (the events' registration needs them) and the parts
of the Complex variables read whole (the code generators read a Complex reference as its
parts): what the passes that eliminate variables by their defining equation keep.
"""
function eventAndComplexNames(simCode::SIM_CODE)::OrderedSet{String}
  local ht = simCode.stringToSimVarHT
  local names = ifEquationNames!(OrderedSet{String}(), simCode.ifEquations)
  foreach(w -> whenCrefNames!(names, w.whenEquation), simCode.whenEquations)
  complexFieldNames!(names, simCode.residualEquations, simCode)
  complexFieldNames!(names, simCode.initialEquations, simCode)
  for ifEq in simCode.ifEquations, b in ifEq.branches
    complexFieldNames!(names, b.residualEquations, simCode)
  end
  complexFieldNames!(names, simCode.eliminatedEquations, simCode)
  return names
end

"One round of `eliminateFrozenStates`; `nothing` if no variable is frozen."
function eliminateFrozenOnce(simCode::SIM_CODE, kept::OrderedSet{String})::Union{Nothing, SIM_CODE}
  local ht = simCode.stringToSimVarHT
  local residuals = simCode.residualEquations
  local shared = Set{String}(simCode.sharedVariables)
  local frozen = OrderedDict{String, Tuple{Exp, Int}}()
  for (i, eq) in enumerate(residuals)
    local f = frozenVariable(foldNumeric(eq.exp), ht)
    f === nothing && continue
    local (name, value) = f
    (name in shared || name in kept || haskey(frozen, name)) && continue
    frozen[name] = (value, i)
  end
  isempty(frozen) && return nothing
  if length(residuals) <= length(frozen)
    delete!(frozen, first(sort(collect(keys(frozen)))))
    isempty(frozen) && return nothing
  end
  @debug "[SIMCODE: $(simCode.name): eliminateFrozenStates] frozen: $(sort(collect(keys(frozen))))"
  local rs = Replacements(n => Replacement(v, false, true) for (n, (v, _)) in frozen)
  local removed = Set{Int}(i for (_, i) in values(frozen))
  local fold = eq::RESIDUAL_EQUATION -> RESIDUAL_EQUATION(foldNumeric(substitute(eq.exp, rs)), eq.source, eq.attr)
  local newHT = copy(ht)
  #= a frozen state's der() is 0 everywhere (the replacement is time-invariant): its entry goes too =#
  local derivatives::Dict{String, String} = derivativeKeys(ht)
  foreach(n -> (haskey(derivatives, n) && delete!(newHT, derivatives[n]); delete!(newHT, n)), keys(frozen))
  local eliminated = sort(collect(keys(frozen)))
  return update(simCode;
                residualEquations = RESIDUAL_EQUATION[fold(eq) for (i, eq) in enumerate(residuals) if !(i in removed)],
                initialEquations = Equation[frozenInitial(eq, rs) for eq in simCode.initialEquations],
                asserts = ASSERT_EQUATION[substitute(a, rs) for a in simCode.asserts],
                initialAlgorithms = INITIAL_ALGORITHM[substitute(a, rs) for a in simCode.initialAlgorithms],
                stringToSimVarHT = newHT,
                irreducibleVariables = filter(n -> !haskey(frozen, n), simCode.irreducibleVariables),
                eliminatedVariables = vcat(simCode.eliminatedVariables, eliminated),
                eliminatedEquations = vcat(simCode.eliminatedEquations, RESIDUAL_EQUATION[residuals[last(frozen[n])] for n in eliminated]))
end

"An initial residual or equation with the frozen variables replaced, folded; another kept."
frozenInitial(eq::RESIDUAL_EQUATION, rs::Replacements)::Equation = RESIDUAL_EQUATION(foldNumeric(substitute(eq.exp, rs)), eq.source, eq.attr)
frozenInitial(eq::EQUATION, rs::Replacements)::Equation =
  EQUATION(foldNumeric(substitute(eq.lhs, rs)), foldNumeric(substitute(eq.rhs, rs)), eq.source, eq.attr)
frozenInitial(eq::Equation, ::Replacements)::Equation = eq

"""
    demoteDeadStates(simCode)

The states no der() reads any more (`deadStateNames`: a fold of `0 * der(x)`, MSL
FundamentalWave's eddy-current core, whose loss `Q_flow = c * V_m * der(Phi)` goes with the
frozen `V_m = 0` and `Q_flow = 0` when its conductance G is 0) as algebraic variables,
their derivatives dropped: a state without dynamics is an alias or an algebraic variable for the
later passes (as a state MTK integrated it as a free unknown: SMPM_Braking took 44k steps and
failed). Nothing for a model with modes or translated again at its structural changes. For a
model index reduction reduced, a state's link to its derivative variable is no reader where
nothing else reads that variable (`unlinkUnreadStateDerivatives`).
"""
function demoteDeadStates(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  #= after OMBackend's index reduction a state's der() is in its link to \$DER_x too =#
  simCode.indexReduced && (simCode = unlinkUnreadStateDerivatives(simCode))
  local dead = deadStateNames(simCode)
  isempty(dead) && return simCode
  local newHT = copy(simCode.stringToSimVarHT)
  local derivatives::Dict{String, String} = derivativeKeys(newHT)
  for name in dead
    local (index, v) = newHT[name]
    newHT[name] = (index, withKind(v, ALG_VARIABLE(index)))
    haskey(derivatives, name) && delete!(newHT, derivatives[name])
  end
  @debug "[SIMCODE: $(simCode.name): demoteDeadStates] $(dead)"
  return update(simCode; stringToSimVarHT = newHT)
end
