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
   the equations of no elements removed, smooth() removed, duplicates removed, the bindings as equations, the asserts
   of if-equations hoisted. =#

"""
    removeEmptyEquations!(model)

The equations of no elements removed from every partition, its initial equations too: an array equation with a
dimension of size 0, a for-equation over an empty range or with only such equations (MLS 10.1: an array with a
dimension of size 0 has no elements). The scalarizing frontend leaves them out; with the arrays kept they came here,
and the der() of a slice of size 0 stayed (no variable to introduce for it): MSL BatchPlant_StandardWater's media
without substances (nXi = nC = 0), 7 equations, sent the model to the scalarized path and a second flatten.
"""
function removeEmptyEquations!(model::IR.Model)::IR.Model
  for p in Iterators.flatten((model.continuous, model.clocked))
    _removeEmpty!(p.equations)
    _removeEmpty!(p.initialEquations)
  end
  return model
end

"`equations` without those of no elements (removeEmptyEquations!), the for-equations' bodies first."
function _removeEmpty!(equations::Vector{IR.Equation})::Vector{IR.Equation}
  for eq in equations
    eq isa IR.FOR_EQUATION && _removeEmpty!(eq.body)
  end
  return filter!(eq -> !_hasNoElements(eq), equations)
end

"Whether an equation has no elements (removeEmptyEquations!); its body already without such equations."
_hasNoElements(eq::IR.ARRAY_EQUATION)::Bool = _hasZeroDimension(eq.ty)
_hasNoElements(eq::IR.FOR_EQUATION)::Bool =
  isempty(eq.body) || (eq.iterator.range isa IR.RANGE && _hasZeroDimension(eq.iterator.range.ty))
_hasNoElements(::IR.Equation)::Bool = false

"Whether an array type has a dimension of size 0."
_hasZeroDimension(ty::IR.T_ARRAY)::Bool = any(d -> d isa IR.DIM_SIZE && d.size == 0, ty.dimensions)

"""
    removeSmooth!(model)

`smooth(order, e)` in the equations of the model's main partition (MLS 3.7.2.2: its use is
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
  local differentiable = (@nospecialize(e::IR.Expression)) -> begin
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
    freeDependentParameters!(model) -> Model

A scalar parameter whose binding reads a parameter with fixed = false, directly or through other bindings, gets its
value from the initialization as that one does (OpenModelica's initial system): fixed = false, its binding the initial
equation `p = binding`. MSL's noise blocks: `actualGlobalSeed = if useGlobalSeed then globalSeed.seed else 0`, the
seed an initial equation's (GlobalSeed); the array code had the value the translation computed for the binding.
For the arrays-kept route.
"""
function freeDependentParameters!(model::IR.Model)::IR.Model
  local initial::Vector{IR.Equation} = model.continuous[1].initialEquations
  for v::IR.Variable in model.variables
    (v.kind isa IR.PARAMETER && v.binding !== nothing &&
     v.ty isa Union{IR.T_REAL, IR.T_INTEGER, IR.T_BOOLEAN}) || continue
    local fixed::Union{Nothing, IR.Expression} = v.attributes.fixed
    (fixed === nothing || (fixed isa IR.BOOLEAN && fixed.value)) || continue
    IndexReduction.readsFreeParameter(v.binding, model.variables, Set{Symbol}([v.name])) || continue
    push!(initial, IR.SCALAR_EQUATION(IR.CREF(v.cref, v.ty), v.binding, v.ty, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    local a::IR.VariableAttributes = v.attributes
    v.attributes = IR.VariableAttributes(a.start, IR.BOOLEAN(false), a.nominal, a.min, a.max, a.unit, a.displayUnit,
                                         a.quantity, a.stateSelect)
    v.binding = nothing
  end
  return model
end

"""
    streamsWithoutEvents!(model) -> Model

actualStream as the frontend writes it, `smooth(0, m * (if m > 0 then a else b))` and `smooth(0, if m > 0 then a else
b)` (its factor `m` outside), with its condition in `noEvent`, in the equations of the main partition: for the
arrays-kept route. The value is continuous in `m` (what `smooth(0, ...)` says), OpenModelica makes no event in
`smooth`, and the array code computes a port's flow from the states at each point: two pressures of 1e5 Pa whose
round-off, 3e-10 Pa, times a port's conductance of 0.3 kg/s/Pa is a flow of 1e-10 kg/s, the relation's hysteresis.
After an event that starts a flow from rest the sign of such a flow is noise, each flip was an event, and MSL
Fluid's RoomCO2WithControls stopped at its table's first event (59888 steps in 0.3 s; the scalarized route, whose
flows are variables of the solver, 618 for the day).
"""
function streamsWithoutEvents!(model::IR.Model)::IR.Model
  local quiet = (@nospecialize(e::IR.Expression)) -> begin
    (IR.isBuiltinCall(e, :smooth) && length(e.arguments) == 2 && e.arguments[1] isa IR.INTEGER &&
     e.arguments[1].value == 0) || return e
    local streamed::IR.Expression = withQuietFlowDirection(e.arguments[2])
    streamed === e.arguments[2] && return e
    return IR.CALL(e.name, IR.Expression[e.arguments[1], streamed], e.ty, e.attributes)
  end
  foreach(eq -> IR.mapEquation!(quiet, eq), model.continuous[1].equations)
  return model
end

"`if m > 0 then a else b`, alone or as the factor of `m`, with its condition in noEvent; another expression itself."
function withQuietFlowDirection(@nospecialize(e::IR.Expression))::IR.Expression
  if e isa IR.IF_EXPRESSION && isFlowDirection(e.condition)
    return IR.IF_EXPRESSION(builtinCall(:noEvent, IR.Expression[e.condition], IR.T_BOOLEAN()), e.thenExp, e.elseExp, e.ty)
  elseif e isa IR.BINARY && e.op === IR.MUL && e.exp2 isa IR.IF_EXPRESSION && isFlowDirection(e.exp2.condition) &&
         e.exp1 == e.exp2.condition.exp1
    return IR.BINARY(e.exp1, e.op, withQuietFlowDirection(e.exp2), e.ty)
  end
  return e
end

"Whether `c` is `m > 0` or `m >= 0` of a variable `m` (a flow's direction)."
isFlowDirection(@nospecialize(c::IR.Expression))::Bool =
  c isa IR.RELATION && (c.op === IR.GREATER || c.op === IR.GREATEREQ) && c.exp1 isa IR.CREF &&
  ((c.exp2 isa IR.REAL && c.exp2.value == 0.0) || (c.exp2 isa IR.INTEGER && c.exp2.value == 0))

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
      foreach((@nospecialize(e::IR.Equation)) -> (h = equationKey(e, h)), x)
    elseif x isa Vector{IR.EquationBranch}
      for b in x
        h = hash(b.condition, h)
        foreach(e -> (h = equationKey(e, h)), b.equations)
      end
    elseif x isa IR.EquationAttributes
      h = hash((x.kind, x.differentiated, x.derivativeLink), h)
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
discrete-time). A Real array's `v = e` as a for-equation over its elements (`IndexReduction.bindingEquation`;
arrays kept: a field of a component array, `Real b = 2 * a` of `C c[3]`; the frontend makes the binding of an array
declared as one an equation itself and leaves none on the variable).
A Boolean, Integer or enumeration array's (arrays kept) an equation per element
(`IndexReduction.discreteBindingEquations`).
A discrete Real with a binding is not Modelica (not supported). Parameters and constants (their bindings are their
values), Strings, records and the other arrays are left.
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
    local continuous = v.kind isa Union{IR.STATE, IR.ALGEBRAIC, IR.ALGEBRAIC_DER}
    local discrete = continuous || v.kind isa IR.DISCRETE
    local lhs = IR.CREF(v.cref, v.ty)
    if continuous && v.ty isa IR.T_REAL
      push!(partition.equations, IR.SCALAR_EQUATION(lhs, v.binding, v.ty, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    elseif continuous && v.ty isa IR.T_ARRAY && v.ty.elementType isa IR.T_REAL
      local rolled::Union{Nothing, IR.FOR_EQUATION} = IndexReduction.bindingEquation(v, variables)
      rolled === nothing || push!(partition.equations, rolled)
    elseif discrete && v.ty isa IR.T_BOOLEAN && isLatchedBooleanBinding(v.binding)
      local b = v.binding
      local branches = IR.WhenBranch[
        IR.WhenBranch(IR.Expression[b.condition], false, false, IR.Statement[IR.ASSIGNMENT(lhs, b.thenExp, IR.NO_SOURCE)]),
        IR.WhenBranch(IR.Expression[invertCondition(b.condition)], false, false,
                      IR.Statement[IR.ASSIGNMENT(lhs, b.elseExp, IR.NO_SOURCE)])]
      push!(partition.equations, IR.WHEN_EQUATION(branches, IR.REGULAR_WHEN, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    elseif discrete && v.ty isa Union{IR.T_BOOLEAN, IR.T_INTEGER, IR.T_ENUMERATION}
      push!(partition.equations, IR.SCALAR_EQUATION(lhs, v.binding, v.ty, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    elseif discrete && v.ty isa IR.T_ARRAY && v.ty.elementType isa Union{IR.T_BOOLEAN, IR.T_INTEGER, IR.T_ENUMERATION}
      append!(partition.equations, IndexReduction.discreteBindingEquations(v, variables))
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
minimum-decrease warning under `not haveMinimumDecrease`). A `terminate()` (`terminateEquation`) is the
when-equations of the branch's condition: `when guard then terminate(...)` as the condition becomes true, `when
initial() then if guard then terminate(...) end if` where it holds at the start (OpenModelica terminates at the
initialization there). An if-equation of asserts only is left out. Of the equations and of the initial equations
(where a `terminate()` is not supported).
"""
function hoistIfEquationAsserts!(model::IR.Model)::IR.Model
  for p in Iterators.flatten((model.continuous, (m.system for m in model.modes)))
    p.equations = hoistIfEquationAsserts(p.equations)
    p.initialEquations = hoistIfEquationAsserts(p.initialEquations)
    for eq in p.initialEquations
      eq isa IR.WHEN_EQUATION && any(b -> any(s -> s isa IR.TERMINATE, b.body), eq.branches) &&
        unsupported("terminate() in an initial equation section", IR.toString(eq))
    end
  end
  return model
end

function hoistIfEquationAsserts(equations::Vector{IR.Equation})::Vector{IR.Equation}
  any(hasBranchAssert, equations) || return equations
  local (out::Vector{IR.Equation}, asserts::Vector{IR.Equation}) = hoistedParts(equations)
  return vcat(out, asserts)
end

"""
The equations without the asserts (and `terminate()`) of their if-equations' branches, and those hoisted
(`hoistBranchAsserts!`); of a for-equation's if-equations a for-equation of their own over the same iterator (the
arrays-kept route keeps for-equations).
"""
function hoistedParts(equations::Vector{IR.Equation})::Tuple{Vector{IR.Equation}, Vector{IR.Equation}}
  local out = IR.Equation[]
  local asserts = IR.Equation[]
  for eq in equations
    if eq isa IR.FOR_EQUATION && hasBranchAssert(eq)
      local (body::Vector{IR.Equation}, hoisted::Vector{IR.Equation}) = hoistedParts(eq.body)
      isempty(body) || push!(out, IR.FOR_EQUATION(eq.iterator, body, eq.source, eq.attributes))
      push!(asserts, IR.FOR_EQUATION(eq.iterator, hoisted, eq.source, eq.attributes))
      continue
    end
    local h = hasBranchAssert(eq) ? hoistBranchAsserts!(asserts, eq, nothing) : eq
    isEmptyIfEquation(h) || push!(out, h)
  end
  return (out, asserts)
end

isEmptyIfEquation(eq::IR.Equation)::Bool =
  eq isa IR.IF_EQUATION && all(b -> isempty(b.equations), eq.branches) && isempty(eq.elseEquations)

hasBranchAssert(eq::IR.Equation)::Bool =
  (eq isa IR.FOR_EQUATION && any(hasBranchAssert, eq.body)) ||
  (eq isa IR.IF_EQUATION &&
   any(e -> e isa IR.ASSERT_EQUATION || isTerminateEquation(e) || hasBranchAssert(e),
       Iterators.flatten((Iterators.flatten(b.equations for b in eq.branches), eq.elseEquations))))

"A `terminate()` of an equation section (`terminateEquation`): `when initial() then terminate(...)`."
isTerminateEquation(eq::IR.Equation)::Bool =
  eq isa IR.WHEN_EQUATION && length(eq.branches) == 1 && only(eq.branches).atInitial && !only(eq.branches).atTerminal &&
  isempty(only(eq.branches).triggers) && length(only(eq.branches).body) == 1 && only(only(eq.branches).body) isa IR.TERMINATE

"""
The when-equations of a `terminate()` (`terminateEquation`) under `guard`: at the condition's rising edge (without its
`noEvent()`: a when-condition generates its events, `if noEvent(x > 0.5) then terminate()` stops at 0.5), and at the
initialization where it holds.
"""
function guardedTerminate(eq::IR.WHEN_EQUATION, guard::Union{Nothing, IR.Expression})::Vector{IR.Equation}
  guard === nothing && return IR.Equation[eq]
  local t::IR.TERMINATE = only(only(eq.branches).body)
  local atStart::IR.IF_STATEMENT = IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(guard, IR.Statement[t])], IR.Statement[], t.source)
  local trigger::IR.Expression = IR.mapExp((x::IR.Expression) -> IR.isBuiltinCall(x, :noEvent) ? only(x.arguments) : x, guard)
  return IR.Equation[IR.WHEN_EQUATION(IR.WhenBranch[IR.WhenBranch(IR.Expression[trigger], false, false, IR.Statement[t])],
                                      eq.kind, eq.source, eq.attributes),
                     IR.WHEN_EQUATION(IR.WhenBranch[IR.WhenBranch(IR.Expression[], true, false, IR.Statement[atStart])],
                                      eq.kind, eq.source, eq.attributes)]
end

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
    elseif isTerminateEquation(eq)
      append!(asserts, guardedTerminate(eq, guard))
    elseif eq isa IR.IF_EQUATION
      local h = hoistBranchAsserts!(asserts, eq, guard)
      isEmptyIfEquation(h) || push!(kept, h)
    else
      push!(kept, eq)
    end
  end
  return kept
end
