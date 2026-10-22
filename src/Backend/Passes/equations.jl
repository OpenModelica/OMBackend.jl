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

#= Passes on the equations of the continuous-time partition (the previous backend's BDAECreate):
   smooth() removed, duplicates removed, the bindings as equations, the asserts of if-equations
   hoisted. =#

"""
    removeSmooth!(model)

`smooth(order, e)` in the equations of the model's main partition (MLS 3.7.4.2: its use is
optional). Of a literal order of at least 1, `noEvent(e)` where `e` holds a relation or an
if-expression: the expression and its derivatives are continuous, so an event adds nothing to
the solve (MSL SoftStarter's quadratic load torque, `smooth(1, if w >= 0 then w^2 else -w^2)`,
validates only so). Of order 0, `e` where it is the whole side, else smooth() stays and its
relations are events: its derivative jumps, and the event keeps the solve accurate and in range
(the OpAmp limiter `homotopy(smooth(0, if ...), ...)` of MSL SignalGenerator; actualStream,
`smooth(0, m_flow * (if m_flow > 0 then inStream(h) else h))`, in MSL InverseParameterization's
pump; as noEvent, OpenModelica's choice, the one failed validation and the other left IF97's range).
"""
function removeSmooth!(model::IR.Model)::IR.Model
  local eventful = x -> x isa IR.RELATION || x isa IR.IF_EXPRESSION
  local differentiable = e -> begin
    (IR.isBuiltinCall(e, :smooth) && length(e.arguments) == 2 && e.arguments[1] isa IR.INTEGER &&
     e.arguments[1].value >= 1) || return e
    local inner = e.arguments[2]
    return IR.anyExp(eventful, inner) ? builtinCall(:noEvent, IR.Expression[inner], IR.typeOf(inner)) : inner
  end
  local strip = e -> IR.isBuiltinCall(e, :smooth) && length(e.arguments) == 2 ? e.arguments[2] : e
  for eq in model.continuous[1].equations
    IR.mapEquation!(differentiable, eq)
    eq isa IR.SCALAR_EQUATION || continue
    if IR.isBuiltinCall(eq.rhs, :smooth)
      eq.rhs = strip(eq.rhs)
    elseif IR.isBuiltinCall(eq.lhs, :smooth)
      eq.lhs = strip(eq.lhs)
    end
  end
  return model
end

"""
    equationKey(eq)

A structural hash of an equation: its kind, its expressions, its source and attributes, its
nested equations and statements. Equal for equations that are copies of each other.
"""
function equationKey(eq::IR.Equation, h::UInt = zero(UInt))::UInt
  h = hash(typeof(eq), h)
  for f in fieldnames(typeof(eq))
    local x = getfield(eq, f)
    if x isa IR.Equation
      h = equationKey(x, h)
    elseif x isa Vector{IR.Equation}
      foreach(e -> (h = equationKey(e, h)), x)
    elseif x isa Vector{IR.EquationBranch}
      for b in x
        h = hash(b.condition, h)
        foreach(e -> (h = equationKey(e, h)), b.equations)
      end
    elseif x isa IR.EquationAttributes
      h = hash((x.kind, x.differentiated), h)
    else
      h = hash(x, h)
    end
  end
  return h
end

"The seed of `deduplicateEquations!`' second key."
const DEDUP_SEED = 0x9e3779b97f4a7c15

"""
    deduplicateEquations!(model)

The equations of each continuous-time partition (and structural mode) each once, the first of
copies kept (the frontend emits some equations of inner/outer components twice).
"""
function deduplicateEquations!(model::IR.Model)::IR.Model
  for p in Iterators.flatten((model.continuous, (m.system for m in model.modes)))
    local seen = Set{Tuple{UInt, UInt}}()
    local unique = IR.Equation[]
    for eq in p.equations
      #= two keys of independent seeds: a collision of one would drop an equation =#
      local k = (equationKey(eq), equationKey(eq, DEDUP_SEED))
      k in seen && continue
      push!(seen, k)
      push!(unique, eq)
    end
    local removed = length(p.equations) - length(unique)
    removed > 0 && @debug "[OMJLIR: deduplicateEquations] $(length(p.equations)) -> $(length(unique)) equations ($(removed) duplicates removed)"
    p.equations = unique
  end
  return model
end

"""
    addBindingEquations!(model)

The declaration bindings of continuous-time and discrete variables as equations (after the
others, MLS 4.4.4): a Real's (scalar) `v = e`; a Boolean's `if a > b then true else false`
a when / elsewhen equation (`isLatchedBooleanBinding`: it updates at the condition's events);
a Boolean's, Integer's
or enumeration's `v = e` (the discrete lifting makes it a when-equation where `e` is
discrete-time). A discrete Real with a binding is not Modelica (not supported). Parameters and
constants (their bindings are their values), Strings, arrays and records (their bindings are
equations already) are left.
"""
function addBindingEquations!(model::IR.Model)::IR.Model
  for p in model.continuous
    addBindingEquations!(model.variables, p)
  end
  for mode in model.modes
    addBindingEquations!(mode.variables, mode.system)
  end
  return model
end

function addBindingEquations!(variables::IR.VariableSet, partition::IR.Partition)::Nothing
  for v in variables
    v.binding === nothing && continue
    local continuous = v.kind isa Union{IR.STATE, IR.ALGEBRAIC}
    local discrete = continuous || v.kind isa IR.DISCRETE
    local lhs = IR.CREF(v.cref, v.ty)
    if continuous && v.ty isa IR.T_REAL
      push!(partition.equations, IR.SCALAR_EQUATION(lhs, v.binding, v.ty, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    elseif discrete && v.ty isa IR.T_BOOLEAN && isLatchedBooleanBinding(v.binding)
      local b = v.binding
      local branches = IR.WhenBranch[
        IR.WhenBranch(IR.Expression[b.condition], false, false, IR.Statement[IR.ASSIGNMENT(lhs, b.thenExp, IR.NO_SOURCE)]),
        IR.WhenBranch(IR.Expression[invertCondition(b.condition)], false, false,
                      IR.Statement[IR.ASSIGNMENT(lhs, b.elseExp, IR.NO_SOURCE)])]
      push!(partition.equations, IR.WHEN_EQUATION(branches, IR.REGULAR_WHEN, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    elseif discrete && v.ty isa Union{IR.T_BOOLEAN, IR.T_INTEGER, IR.T_ENUMERATION}
      push!(partition.equations, IR.SCALAR_EQUATION(lhs, v.binding, v.ty, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    elseif v.kind isa IR.DISCRETE && v.ty isa IR.T_REAL
      unsupported("a discrete Real variable with a declaration binding", v.name)
    end
  end
  return nothing
end

"The inverted relation `a > b` -> `a < b`, `a < b` -> `a > b` (the when-equation of a Boolean binding)."
function invertCondition(c::IR.RELATION)::IR.RELATION
  c.op === IR.GREATER && return IR.RELATION(c.exp1, IR.LESS, c.exp2, c.index)
  c.op === IR.LESS && return IR.RELATION(c.exp1, IR.GREATER, c.exp2, c.index)
  unsupported("inverting this relation", IR.toString(c))
end

"""
A Boolean binding the previous backend made a when / elsewhen equation: `if a > b then true
else false` (or `<`), its branches literals. Another (a `>=`, a Boolean condition, a branch
that is a variable, which a when would latch at the crossing) is the equation `v = e`.
"""
isLatchedBooleanBinding(b::Union{Nothing, IR.Expression})::Bool =
  b isa IR.IF_EXPRESSION && b.condition isa IR.RELATION && b.condition.op in (IR.GREATER, IR.LESS) &&
  b.thenExp isa IR.BOOLEAN && b.elseExp isa IR.BOOLEAN

"""
    hoistIfEquationAsserts!(model)

The asserts in the branches of if-equations, as top-level asserts under the branch's condition
(`not guard or c`), after the other equations: the if-equation lowering takes residual
equations only (MSL Fluid's AST_BatchPlant). A call for its effects (an assert of its own) is
made under the branch's condition: `if guard then call else true` (Buildings' Movers print the
minimum-decrease warning under `not haveMinimumDecrease`). An if-equation of asserts only is
left out. Of the equations and of the initial equations.
"""
function hoistIfEquationAsserts!(model::IR.Model)::IR.Model
  for p in Iterators.flatten((model.continuous, (m.system for m in model.modes)))
    p.equations = hoistIfEquationAsserts(p.equations)
    p.initialEquations = hoistIfEquationAsserts(p.initialEquations)
  end
  return model
end

function hoistIfEquationAsserts(equations::Vector{IR.Equation})::Vector{IR.Equation}
  any(hasBranchAssert, equations) || return equations
  local out = IR.Equation[]
  local asserts = IR.Equation[]
  for eq in equations
    local h = hasBranchAssert(eq) ? hoistBranchAsserts!(asserts, eq, nothing) : eq
    isEmptyIfEquation(h) || push!(out, h)
  end
  return vcat(out, asserts)
end

isEmptyIfEquation(eq::IR.Equation)::Bool =
  eq isa IR.IF_EQUATION && all(b -> isempty(b.equations), eq.branches) && isempty(eq.elseEquations)

hasBranchAssert(eq::IR.Equation)::Bool =
  eq isa IR.IF_EQUATION &&
  any(e -> e isa IR.ASSERT_EQUATION || hasBranchAssert(e), Iterators.flatten((Iterators.flatten(b.equations for b in eq.branches),
                                                                              eq.elseEquations)))

function hoistBranchAsserts!(asserts::Vector{IR.Equation}, eq::IR.IF_EQUATION, outer::Union{Nothing, IR.Expression})::IR.IF_EQUATION
  local none = nothing
  local branches = IR.EquationBranch[]
  for b in eq.branches
    local guard = andCondition(outer, andCondition(none, b.condition))
    push!(branches, IR.EquationBranch(b.condition, takeBranchAsserts!(asserts, b.equations, guard)))
    none = andCondition(none, notCondition(b.condition))
  end
  local elseEquations = takeBranchAsserts!(asserts, eq.elseEquations, andCondition(outer, none))
  return IR.IF_EQUATION(branches, elseEquations, eq.source, eq.attributes)
end

"A branch's equations without its asserts, which go to `asserts` under `guard`."
function takeBranchAsserts!(asserts::Vector{IR.Equation}, body::Vector{IR.Equation}, guard::Union{Nothing, IR.Expression})::Vector{IR.Equation}
  local kept = IR.Equation[]
  for eq in body
    if eq isa IR.ASSERT_EQUATION
      if eq.condition isa IR.CALL && eq.condition.ty isa IR.T_NORETCALL
        push!(asserts, IR.ASSERT_EQUATION(IR.IF_EXPRESSION(guard, eq.condition, IR.BOOLEAN(true), IR.T_BOOLEAN()),
                                          eq.message, eq.level, eq.source))
      else
        push!(asserts, IR.ASSERT_EQUATION(orCondition(notCondition(guard), eq.condition), eq.message, eq.level, eq.source))
      end
    elseif eq isa IR.IF_EQUATION
      local h = hoistBranchAsserts!(asserts, eq, guard)
      isEmptyIfEquation(h) || push!(kept, h)
    else
      push!(kept, eq)
    end
  end
  return kept
end
