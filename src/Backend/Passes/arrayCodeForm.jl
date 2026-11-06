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

#= The form the array code generation takes: a model reduced in array form, its derivatives solved in array form
   (IndexReduction.solvedInArrays), through the model-level transformations its generation made before (2026-11-01):
   the passes of the array route between index reduction and the simulation code. =#

"""
    arrayCodeForm!(model) -> Model

`model`, reduced and its derivatives solved in array form (`IndexReduction.solvedInArrays`), in the form the array
code generation takes; of its first partition (a model of several the generation declines):

  - an algebraic variable a when-equation assigns is discrete (`whenAssignedDiscrete!`);
  - the lifted discrete clusters the event iteration runs at every pass are their equations (`everyPassClusters!`);
  - the derivative variables that only their definitions read are dropped (`dropUnreadDerivatives!`);
  - the calls are inlined after index reduction (`inlineAfterIndexReduction!`);
  - the for-equations that read the elements of what is no variable, or subscript an array literal by their iterator,
    are unrolled (`unrollElementLoops!`);
  - the for-ranges that fold to Integer literals are those literal ranges (`literalRanges!`);
  - the expressions are in their element form: what the code takes per element written so (`elementForm!`), the
    parameters' bindings that read parameters too.
"""
function arrayCodeForm!(model::IR.Model)::IR.Model
  (length(model.continuous) == 1 && isempty(model.clocked) && isempty(model.modes)) || return model
  whenAssignedDiscrete!(model)
  everyPassClusters!(model)
  dropUnreadDerivatives!(model)
  inlineAfterIndexReduction!(model)
  unrollElementLoops!(model)
  literalRanges!(model)
  elementForm!(model)
  return model
end

#= ==================================================================================== =#
#= When-assigned variables                                                              =#
#= ==================================================================================== =#

"""
    whenAssignedDiscrete!(model) -> Model

An algebraic variable a when-equation of the first partition assigns (an element of it too) is discrete (MLS 3.8.3; the
frontend's implicitly discrete variable is algebraic).
"""
function whenAssignedDiscrete!(model::IR.Model)::IR.Model
  local assigned::Set{Symbol} = Set{Symbol}()
  foreach((eq::IR.Equation) -> whenTargets!(assigned, eq, model.variables), model.continuous[1].equations)
  for v in model.variables
    (v.kind isa IR.ALGEBRAIC && v.name in assigned) && (v.kind = IR.DISCRETE())
  end
  return model
end

"The variables the when-equations in `eq` (in for-equations too) assign, a tuple assignment's targets too, onto `out`."
function whenTargets!(out::Set{Symbol}, eq::IR.Equation, variables::IR.VariableSet)::Nothing
  if eq isa IR.FOR_EQUATION
    for b in eq.body
      whenTargets!(out, b, variables)
    end
  elseif eq isa IR.WHEN_EQUATION
    for br in eq.branches, st in br.body
      st isa IR.ASSIGNMENT || continue
      for t in (st.lhs isa IR.TUPLE ? st.lhs.elements : IR.Expression[st.lhs])
        t isa IR.CREF && push!(out, variableNameOf(t.cref, variables))
      end
    end
  end
  return nothing
end

"The variable a reference is to or into: its own name, else (an element) its array's."
variableNameOf(c::IR.ComponentRef, variables::IR.VariableSet)::Symbol =
  (IR.hasSubscripts(c) && !haskey(variables, c.name)) ? unsubscriptedName(c) : c.name

#= ==================================================================================== =#
#= Clusters evaluated at every pass                                                     =#
#= ==================================================================================== =#

"""
    everyPassClusters!(model) -> Model

The lifted discrete clusters the event iteration runs at every pass (`liftDiscreteEquations!`: a cluster without
relations, triggered by `change(pre(v))`, a condition the code generation never evaluates) as their discrete equations
`v = e`, which the array code evaluates at every pass of every event and at the initialization; the initial arms'
assignments of the same variables left out. A cluster on its relations (`LIFTED_CLUSTER`) stays a when-equation, held
between events (an ideal diode's `off = s < 0` out of the continuous loop); the array code runs it at every pass of
every event too (`SimCode.EQ_ATTR`'s `everyEvent`).
"""
function everyPassClusters!(model::IR.Model)::IR.Model
  local partition::IR.Partition = model.continuous[1]
  local targets::Set{Symbol} = Set{Symbol}()
  local out::Vector{IR.Equation} = IR.Equation[]
  for eq in partition.equations
    if isEveryPassCluster(eq)
      for st in (eq::IR.WHEN_EQUATION).branches[1].body
        local a::IR.ASSIGNMENT = st
        push!(targets, (a.lhs::IR.CREF).cref.name)
        push!(out, IR.SCALAR_EQUATION(a.lhs, a.rhs, IR.typeOf(a.lhs), a.source, IR.DEFAULT_ATTRIBUTES))
      end
    else
      push!(out, eq)
    end
  end
  isempty(targets) && return model
  partition.equations = IR.Equation[e for e in (withoutInitialArms(eq, targets) for eq in out) if e !== nothing]
  return model
end

"Whether `eq` is a lifted cluster run at every pass: one branch, triggered by `change(pre(v))` (or several of them), assignments."
function isEveryPassCluster(eq::IR.Equation)::Bool
  (eq isa IR.WHEN_EQUATION && eq.kind == IR.REGULAR_WHEN && length(eq.branches) == 1) || return false
  local br::IR.WhenBranch = eq.branches[1]
  return !br.atInitial && !br.atTerminal && !isempty(br.triggers) && all(isChangeOfPre, br.triggers) &&
         all(st::IR.Statement -> st isa IR.ASSIGNMENT && st.lhs isa IR.CREF, br.body)
end

"`change(pre(v))`, or an `or` of them (`changeOfPreCondition`)."
isChangeOfPre(c::IR.Expression)::Bool =
  (IR.isBuiltinCall(c, :change) && IR.isBuiltinCall(c.arguments[1], :pre)) ||
  (c isa IR.LBINARY && c.op === IR.OR && isChangeOfPre(c.exp1) && isChangeOfPre(c.exp2))

"`eq` without an initial arm's assignments to `targets` (an arm left with none: nothing)."
function withoutInitialArms(eq::IR.Equation, targets::Set{Symbol})::Union{Nothing, IR.Equation}
  (eq isa IR.WHEN_EQUATION && eq.kind == IR.INITIAL_WHEN && length(eq.branches) == 1) || return eq
  local br::IR.WhenBranch = eq.branches[1]
  local kept::Vector{IR.Statement} = IR.Statement[st for st in br.body
                                                  if !(st isa IR.ASSIGNMENT && st.lhs isa IR.CREF && st.lhs.cref.name in targets)]
  isempty(kept) && return nothing
  length(kept) == length(br.body) && return eq
  return IR.WHEN_EQUATION(IR.WhenBranch[IR.WhenBranch(br.triggers, br.atInitial, br.atTerminal, kept)], eq.kind, eq.source,
                          eq.attributes)
end

#= ==================================================================================== =#
#= Derivative variables only their definitions read                                     =#
#= ==================================================================================== =#

"""
    dropUnreadDerivatives!(model) -> Model

The derivative variables (`ALGEBRAIC_DER`, `\$DER.x`) of the first partition's equations that only their definitions
read (scalar equations of the variable, or of its elements at the iterators, on their left side; no initial equation)
dropped, with their definitions: the state's link took their value (`solvedInArrays` puts the small solved values in).
A kind that names a dropped one (a state's derivative, the dummy derivative `\$DER.\$DER.x` of `\$DER.x`) names the
state's link instead (`\$STATEDER.x`, the `STATE_DER` of the state), which holds the same value.
"""
function dropUnreadDerivatives!(model::IR.Model)::IR.Model
  local partition::IR.Partition = model.continuous[1]
  local unread::Set{Symbol} = unreadDerivatives(partition, model.variables)
  isempty(unread) && return model
  partition.equations = IR.Equation[e for e in (withoutDefinitions(eq, unread) for eq in partition.equations) if e !== nothing]
  local links::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}(v.kind.state => v.name for v in model.variables if v.kind isa IR.STATE_DER)
  local linkOf::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}(v.name => links[v.kind.state] for v in model.variables
                                                            if v.name in unread && haskey(links, v.kind.state))
  filter!((v::IR.Variable) -> !(v.name in unread), model.variables)
  filter!((n::Symbol) -> !(n in unread), partition.unknowns)
  for v in model.variables
    v.kind = IR.mapKindNames((x::Symbol) -> get(linkOf, x, x), v.kind)
  end
  return model
end

"The derivative variables of the partition's equations that only their definitions read (`dropUnreadDerivatives!`)."
function unreadDerivatives(partition::IR.Partition, variables::IR.VariableSet)::Set{Symbol}
  local candidates::Set{Symbol} = Set{Symbol}(v.name for v in variables if v.kind isa IR.ALGEBRAIC_DER)
  isempty(candidates) && return candidates
  local reads::Dict{Symbol, Int} = Dict{Symbol, Int}()
  local definitions::Dict{Symbol, Int} = Dict{Symbol, Int}()
  foreach((eq::IR.Equation) -> countDefinitions!(definitions, eq, candidates, variables), partition.equations)
  #= an initial equation reads the variable, also one with it on its left (`der(x) = 0`, a steady start) =#
  for eq in Iterators.flatten((partition.equations, partition.initialEquations))
    IR.foreachExp(eq) do x::IR.Expression
      if x isa IR.CREF
        local n::Symbol = variableNameOf(x.cref, variables)
        n in candidates && (reads[n] = get(reads, n, 0) + 1)
      end
      nothing
    end
  end
  return Set{Symbol}(n for n in candidates if get(definitions, n, 0) > 0 && get(reads, n, 0) == definitions[n])
end

"The definitions in `eq` (scalar equations, in for-equations too, with a reference on the left) of the `candidates`, counted."
function countDefinitions!(definitions::Dict{Symbol, Int}, eq::IR.Equation, candidates::Set{Symbol},
                           variables::IR.VariableSet)::Nothing
  if eq isa IR.FOR_EQUATION
    for b in eq.body
      countDefinitions!(definitions, b, candidates, variables)
    end
  elseif eq isa IR.SCALAR_EQUATION && eq.lhs isa IR.CREF
    local n::Symbol = variableNameOf(eq.lhs.cref, variables)
    n in candidates && (definitions[n] = get(definitions, n, 0) + 1)
  end
  return nothing
end

"The equation `eq` without the definitions of the variables `unread` (a for-equation's body of them: none)."
function withoutDefinitions(eq::IR.Equation, unread::Set{Symbol})::Union{Nothing, IR.Equation}
  if eq isa IR.FOR_EQUATION
    local body::Vector{IR.Equation} = IR.Equation[b for b in (withoutDefinitions(x, unread) for x in eq.body) if b !== nothing]
    isempty(body) && return nothing
    return length(body) == length(eq.body) ? eq : IR.FOR_EQUATION(eq.iterator, body, eq.source, eq.attributes)
  end
  (eq isa IR.SCALAR_EQUATION && eq.lhs isa IR.CREF) || return eq
  local c::IR.ComponentRef = eq.lhs.cref
  return (c.name in unread || (IR.hasSubscripts(c) && unsubscriptedName(c) in unread)) ? nothing : eq
end

#= ==================================================================================== =#
#= Inlining after index reduction                                                      =#
#= ==================================================================================== =#

"""
    inlineAfterIndexReduction!(model) -> Model

The first partition's equations and initial equations with their calls inlined where the functions' algorithms run
into one expression (MLS: InlineAfterIndexReduction, Inline; `inlinedEquation`); not of an external function or one
with an assert (inlined, its check is lost: normalizeWithAssert).
"""
function inlineAfterIndexReduction!(model::IR.Model)::IR.Model
  local inlinable::Dict{Symbol, IR.ModelicaFunction} =
    Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in model.functions if f.external === nothing && !hasAssert(f.body))
  local partition::IR.Partition = model.continuous[1]
  partition.equations = IR.Equation[inlinedEquation(eq, inlinable) for eq in partition.equations]
  partition.initialEquations = IR.Equation[inlinedEquation(eq, inlinable) for eq in partition.initialEquations]
  return model
end

"Whether statements hold an assert (also in branches and loops)."
function hasAssert(stmts::Vector{IR.Statement})::Bool
  for st in stmts
    st isa IR.ASSERT && return true
    st isa IR.IF_STATEMENT && (any((b::IR.StatementBranch) -> hasAssert(b.body), st.branches) || hasAssert(st.elseBody)) && return true
    (st isa IR.FOR_STATEMENT || st isa IR.WHILE_STATEMENT) && hasAssert(st.body) && return true
  end
  return false
end

"""
    inlinedEquation(eq, functions) -> Equation

`eq` with its calls of `functions` inlined where their algorithms run into one expression (`IndexReduction.inlineCalls`;
scalar, if- and for-equations, a for-equation's body): MultiBody's `planarRotation(e, phi, w)[1][2, 1]` as
`e[2] * e[1] + (-(e[2] * e[1])) * cos(phi) - e[3] * sin(phi)`, no call and no array built per element (each element of
a record result called the function again). Other kinds, and an equation without such a call, as they are (inlining
also writes a subscripted reference on its parts: `observed.offset[\$i1]` of a record variable's array field, whose
elements are variables, then read as a variable of its own).
"""
function inlinedEquation(eq::IR.Equation, functions::Dict{Symbol, IR.ModelicaFunction})::IR.Equation
  local calls::Bool = false
  IR.foreachExp(eq) do x::IR.Expression
    (x isa IR.CALL && haskey(functions, x.name)) && (calls = true)
    nothing
  end
  calls || return eq
  (eq isa IR.SCALAR_EQUATION || eq isa IR.IF_EQUATION) && return IndexReduction.inlineEquation(eq, functions)
  eq isa IR.FOR_EQUATION &&
    return IR.FOR_EQUATION(eq.iterator, IR.Equation[inlinedEquation(e, functions) for e in eq.body], eq.source, eq.attributes)
  return eq
end

#= ==================================================================================== =#
#= For-equations over elements                                                          =#
#= ==================================================================================== =#

"""
    unrollElementLoops!(model) -> Model

The for-equations of the first partition's equations and initial equations (in if-equations' branches too) whose body
reads the elements of an array that is no variable by subscripts (`readsElementArray`) or subscripts an array literal
by the iterator (`subscriptsLiteral`) unrolled: each iteration's equations, the iterator its value, in the order of the
values, then of the body (like ones the array code's rerolling makes one class again). A for-equation over a range
that is no literal one, or an empty one, as it is.
"""
function unrollElementLoops!(model::IR.Model)::IR.Model
  local partition::IR.Partition = model.continuous[1]
  local isVariable::Function = (n::Symbol) -> haskey(model.variables, n)
  partition.equations = reduce(vcat, (unrolledEquations(eq, isVariable) for eq in partition.equations); init = IR.Equation[])
  partition.initialEquations = reduce(vcat, (unrolledEquations(eq, isVariable) for eq in partition.initialEquations); init = IR.Equation[])
  return model
end

"The equations of `eq` with its for-equations unrolled where `unrollElementLoops!` unrolls them."
function unrolledEquations(eq::IR.Equation, isVariable::Function)::Vector{IR.Equation}
  if eq isa IR.FOR_EQUATION
    local range::Union{Nothing, StepRange{Int, Int}} = IndexReduction.literalRange(IndexReduction.foldTree(eq.iterator.range))
    (range === nothing || isempty(range)) && return IR.Equation[eq]
    if any((b::IR.Equation) -> readsElementArray(b, isVariable) || subscriptsLiteral(b, eq.iterator.name), eq.body)
      local out::Vector{IR.Equation} = IR.Equation[]
      for value in range, b in eq.body
        append!(out, unrolledEquations(withIterator(b, Dict{Symbol, Int}(eq.iterator.name => value)), isVariable))
      end
      return out
    end
    local body::Vector{IR.Equation} = reduce(vcat, (unrolledEquations(b, isVariable) for b in eq.body); init = IR.Equation[])
    return IR.Equation[IR.FOR_EQUATION(eq.iterator, body, eq.source, eq.attributes)]
  elseif eq isa IR.IF_EQUATION
    local unrolled::Function = (eqs::Vector{IR.Equation}) -> reduce(vcat, (unrolledEquations(b, isVariable) for b in eqs); init = IR.Equation[])
    return IR.Equation[IR.IF_EQUATION(IR.EquationBranch[IR.EquationBranch(b.condition, unrolled(b.equations)) for b in eq.branches],
                                      unrolled(eq.elseEquations), eq.source, eq.attributes)]
  end
  return IR.Equation[eq]
end

"""
Whether the equation reads an array that is no variable by subscripts (`frame_a.R.w[\$i1]`: a record's array field,
whose elements are variables of their own, `expandRecordFieldArrays!`, read by a for-equation rolled before).
"""
function readsElementArray(eq::IR.Equation, isVariable::Function)::Bool
  local found::Bool = false
  IR.foreachExp(eq) do x::IR.Expression
    #= also as a reference subscripted on its last part (`observed.offset[\$i1]`, inlined: `subscriptedReference`) =#
    local base::Union{Nothing, IR.ComponentRef} =
      (x isa IR.SUBSCRIPTED && x.exp isa IR.CREF && !IR.hasSubscripts(x.exp.cref) && IR.typeOf(x.exp) isa IR.T_ARRAY) ? x.exp.cref :
      (x isa IR.CREF && length(x.cref.parts) > 1 && !isempty(x.cref.parts[end].subscripts) &&
       !any(IR.isSubscripted, @view x.cref.parts[1:end - 1])) ? x.cref : nothing
    base === nothing && return nothing
    local last::IR.CrefPart = base.parts[end]
    local nm::Symbol = IR.ComponentRef(IR.CrefPart[base.parts[1:end - 1]; IR.CrefPart(last.name, IR.Subscript[], last.ty)]).name
    isVariable(nm) || (found = true)
    nothing
  end
  return found
end

"""
Whether the equation subscripts an array literal by the iterator `it` (an inlined call's matrix product, `inlinedEquation`:
`{{R.T[1, 1], ...}, ...}[\$i1, 2]`): unrolled, the subscripts are literal and the element is picked.
"""
function subscriptsLiteral(eq::IR.Equation, it::Symbol)::Bool
  local found::Bool = false
  local isIterator::Function = (y::IR.Expression) -> y isa IR.CREF && length(y.cref.parts) == 1 && y.cref.name === it &&
                                                      !IR.hasSubscripts(y.cref)
  IR.foreachExp(eq) do x::IR.Expression
    (x isa IR.SUBSCRIPTED && x.exp isa IR.ARRAY) || return nothing
    any((s::IR.Subscript) -> s isa IR.INDEX && IR.anyExp(isIterator, (s::IR.INDEX).exp), x.subscripts) && (found = true)
    nothing
  end
  return found
end

"A copy of the equation `eq` with the iterators of `values` replaced by their values (`IndexReduction.withIterators`)."
withIterator(eq::IR.Equation, values::Dict{Symbol, Int})::IR.Equation =
  IR.mapEquation!((x::IR.Expression) -> IndexReduction.withIterators(x, values), deepcopy(eq))

#= ==================================================================================== =#
#= The element form                                                                     =#
#= ==================================================================================== =#

"The most terms the element form writes out (a reduction's, an array argument's elements): the array code's limit."
const MAX_ELEMENT_TERMS = 1000

"What the element form looks up: the model's variables and Modelica functions by their names."
struct ElementContext
  variables::IR.VariableSet
  functions::Set{Symbol}
end

"""
    elementForm!(model) -> Model

The expressions of the first partition's equations and initial equations (their statements' too, a reinit's target)
in their element form (`elementFormExp`): what the array code takes per element written so, in OMJLIR, before the
simulation code; and the bindings of the parameters the equations read (`parametersRead`) that read parameters
(`bindingElementForm`: the array code's code of a parameter over the parameters it reads).
"""
function elementForm!(model::IR.Model)::IR.Model
  local cx::ElementContext = ElementContext(model.variables, Set{Symbol}(f.name for f in model.functions))
  local partition::IR.Partition = model.continuous[1]
  local read::Set{Symbol} = parametersRead(model)
  local root::RootMap = RootMap((e::IR.Expression) -> elementFormExp(e, cx))
  partition.equations = IR.Equation[first(IR.traverseFields(eq, root, nothing)) for eq in partition.equations]
  partition.initialEquations = IR.Equation[first(IR.traverseFields(eq, root, nothing)) for eq in partition.initialEquations]
  for v in model.variables
    (v.name in read && v.binding !== nothing && readsParameter(v.binding, model.variables)) || continue
    v.binding = bindingElementForm(v, cx)
  end
  return model
end

"""
    parametersRead(model) -> Set{Symbol}

The parameters and constants the first partition's equations and initial equations read (an element of one too), the
start attributes of the other variables (the array code computes them again from overridden parameters), and those the
bindings of these read in turn: the ones whose bindings the array code takes as code (a parameter's value over the
parameters it reads); the others' values only, evaluated.
"""
function parametersRead(model::IR.Model)::Set{Symbol}
  local out::Set{Symbol} = Set{Symbol}()
  local todo::Vector{Symbol} = Symbol[]
  local note::Function = (x::IR.Expression) -> begin
    x isa IR.CREF || return nothing
    local name::Symbol = variableNameOf(x.cref, model.variables)
    local v::Union{Nothing, IR.Variable} = get(model.variables, name, nothing)
    (v !== nothing && v.kind isa Union{IR.PARAMETER, IR.CONSTANT} && !(name in out)) || return nothing
    push!(out, name)
    push!(todo, name)
    return nothing
  end
  local partition::IR.Partition = model.continuous[1]
  foreach((eq::IR.Equation) -> IR.foreachExp(note, eq), partition.equations)
  foreach((eq::IR.Equation) -> IR.foreachExp(note, eq), partition.initialEquations)
  for v in model.variables
    (v.kind isa Union{IR.PARAMETER, IR.CONSTANT} || v.attributes.start === nothing) && continue
    IR.foreachExp(note, v.attributes.start)
  end
  while !isempty(todo)
    local b::Union{Nothing, IR.Expression} = model.variables[pop!(todo)].binding
    b === nothing || IR.foreachExp(note, b)
  end
  return out
end

"""
    RootMap(f)

`f` applied to the root expressions of a traversal's statements and equations (their sides, conditions, arguments),
and to a reinit's target as a reference (`f` giving a reference again).
"""
struct RootMap{F}
  f::F
end

"An expression of a traversal: `f` of it."
(m::RootMap)(e::IR.Expression, a::Nothing) = (m.f(e), a)

"A reinit: its target as a reference through `f` (a reference again: its element form), its value through `f`."
function (m::RootMap)(s::IR.REINIT, a::Nothing)
  local target::IR.Expression = m.f(IR.CREF(s.state, IR.T_REAL()))
  return (IR.REINIT(target isa IR.CREF ? target.cref : s.state, m.f(s.value), s.source), a)
end

"A statement or an equation of a traversal: its fields through the traversal."
(m::RootMap)(x::Union{IR.Statement, IR.Equation}, a::Nothing) = (first(IR.traverseFields(x, m, nothing)), a)

"Whether `e` reads a parameter or constant of `variables` (an element of one too)."
readsParameter(e::IR.Expression, variables::IR.VariableSet)::Bool =
  IR.anyExp(e) do x::IR.Expression
    x isa IR.CREF || return false
    local v::Union{Nothing, IR.Variable} = get(variables, variableNameOf(x.cref, variables), nothing)
    v !== nothing && v.kind isa Union{IR.PARAMETER, IR.CONSTANT}
  end

"""
    bindingElementForm(v, cx) -> Expression

The binding of the parameter `v` in its element form: of an array parameter of a known size (`MAX_ELEMENT_TERMS`
elements at most) with an array binding the array literal of its elements (`elementArray`, as the array code takes
them, a parameter's code over the parameters it reads; a larger one as it is, which the array code takes as no code);
another (a scalar's, one for every element: `each`) by `elementFormExp`.
"""
function bindingElementForm(v::IR.Variable, cx::ElementContext)::IR.Expression
  local b::IR.Expression = v.binding::IR.Expression
  if v.ty isa IR.T_ARRAY && IR.typeOf(b) isa IR.T_ARRAY
    all((d::IR.Dimension) -> d isa IR.DIM_SIZE, v.ty.dimensions) || return b
    local dims::Vector{Int} = Int[(d::IR.DIM_SIZE).size for d in v.ty.dimensions]
    prod(dims) <= MAX_ELEMENT_TERMS || return b
    return elementArray(b, dims, v.ty, cx)
  end
  return elementFormExp(b, cx)
end

"""
    literalRanges!(model) -> Model

The ranges of the first partition's equations and initial equations (for-equations', for-statements', slices') that
fold to Integer literals (`IndexReduction.literalRange`; `size()` of an array of a known size its value,
`knownSizeValues`) as those literal ranges: the one evaluation of a for-range, which the simulation code then holds as
constants.
"""
function literalRanges!(model::IR.Model)::IR.Model
  local partition::IR.Partition = model.continuous[1]
  local literal::Function = (e::IR.Expression) -> begin
    e isa IR.RANGE || return e
    local r::Union{Nothing, StepRange{Int, Int}} = IndexReduction.literalRange(IndexReduction.foldTree(knownSizeValues(e)))
    r === nothing && return e
    return IR.RANGE(IR.INTEGER(first(r)), step(r) == 1 ? nothing : IR.INTEGER(step(r)), IR.INTEGER(last(r)), e.ty)
  end
  foreach((eq::IR.Equation) -> IR.mapEquation!(literal, eq), partition.equations)
  foreach((eq::IR.Equation) -> IR.mapEquation!(literal, eq), partition.initialEquations)
  return model
end

"`e` with `size(x, k)` of an array `x` of a known size and an Integer literal `k` as its value (a for-range's bound)."
knownSizeValues(e::IR.Expression)::IR.Expression = IR.mapExp(e) do x::IR.Expression
  (x isa IR.SIZE && x.dimension isa IR.INTEGER) || return x
  local ty::IR.ModelicaType = IR.typeOf(x.exp)
  ty isa IR.T_ARRAY || return x
  local k::Int = (x.dimension::IR.INTEGER).value
  (1 <= k <= length(ty.dimensions)) || return x
  local n::Union{Nothing, Int} = IR.dimensionSize(ty.dimensions[k])
  return n === nothing ? x : IR.INTEGER(n)
end

"""
    elementFormExp(e, cx) -> Expression

`e` in its element form, top-down: an element of an array expression by its subscripts distributed
(`subscriptedForm`), a field of a record literal or variable the field (`recordElementForm`), a reference to an element
of a split array the slice's (`IndexReduction.slicedCref`), a scalar product and sum, product, min, max of an array
written out over their elements, a call of a Modelica function of the model with its array arguments as the arrays of
their elements (`valueForm`). What it cannot write so stays as it is (the array code declines it).
"""
function elementFormExp(e::IR.Expression, cx::ElementContext)::IR.Expression
  e isa IR.SUBSCRIPTED && return subscriptedForm(e, cx)
  e isa IR.RECORD_ELEMENT && return recordElementForm(e, cx)
  e isa IR.CREF && return crefForm(e, cx)
  e isa IR.BINARY && e.op == IR.SCALAR_PRODUCT && return scalarProductForm(e, cx)
  e isa IR.CALL && return callForm(e, cx)
  return IR.mapChildren((x::IR.Expression) -> elementFormExp(x, cx), e)
end

"Whether the name is of a variable of the model."
isModelVariable(cx::ElementContext, name::Symbol)::Bool = haskey(cx.variables, name)

"Whether `e` calls a Modelica function of the model."
isModelFunctionCall(cx::ElementContext, e::IR.Expression)::Bool = e isa IR.CALL && !e.attributes.builtin && e.name in cx.functions

"""
A reference: to an element of an array split along its inner dimensions the slice's (`IndexReduction.slicedCref`),
its subscripts in their element form.
"""
function crefForm(e::IR.CREF, cx::ElementContext)::IR.Expression
  local cr::IR.ComponentRef = e.cref
  local whole::Bool = !IR.hasSubscripts(cr) || isModelVariable(cx, cr.name)
  local name::Symbol = whole ? cr.name : unsubscriptedName(cr)
  if !isModelVariable(cx, name)
    local sliced::Union{Nothing, IR.ComponentRef} = IndexReduction.slicedCref(cr, (n::Symbol) -> isModelVariable(cx, n))
    sliced === nothing || return crefForm(IR.CREF(sliced, e.ty), cx)
  end
  return IR.mapChildren((x::IR.Expression) -> elementFormExp(x, cx), e)
end

"""
An element of an array expression (`{155, 155, 155}[\$i1]`, `(R.T * v)[i]`, `resolve1(R, v)[i]`): of element-wise
arithmetic, a product, a literal by literal subscripts the element distributed (`Evaluation.selectElement`); of an
array variable its element (the reference subscripted, `composedSubscripts`); of an array literal, a function's array
result, a record's array field the element of the value. Subscripts that are not all indices as they are.
"""
function subscriptedForm(e::IR.SUBSCRIPTED, cx::ElementContext)::IR.Expression
  local field::IR.Expression = fieldExpression(e)
  field === e || return elementFormExp(field, cx)
  all((s::IR.Subscript) -> s isa IR.INDEX, e.subscripts) || return e
  local picked::IR.Expression = Evaluation.selectElement(e.exp, e.subscripts, e.ty)
  picked isa IR.SUBSCRIPTED || return elementFormExp(picked, cx)
  local base::IR.Expression = fieldExpression(picked.exp)
  base === picked.exp || return elementFormExp(IR.SUBSCRIPTED(base, picked.subscripts, e.ty), cx)
  local subscripts::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(elementFormExp((s::IR.INDEX).exp, cx)) for s in picked.subscripts]
  if base isa IR.CREF
    local last::IR.CrefPart = base.cref.parts[end]
    local composed::Union{Nothing, Vector{IR.Subscript}} = composedSubscripts(last.subscripts, picked.subscripts)
    composed === nothing && return e
    local parts::Vector{IR.CrefPart} = IR.CrefPart[base.cref.parts[1:end - 1]..., IR.CrefPart(last.name, composed, last.ty)]
    return crefForm(IR.CREF(IR.ComponentRef(parts), e.ty), cx)
  elseif base isa IR.ARRAY
    return IR.SUBSCRIPTED(IR.ARRAY(IR.Expression[elementFormExp(x, cx) for x in base.elements], base.ty), subscripts, e.ty)
  elseif isModelFunctionCall(cx, base) || base isa IR.RECORD_ELEMENT
    return IR.SUBSCRIPTED(valueForm(base, cx), subscripts, e.ty)
  end
  #= another array (a built-in's: `cat(2, {{0.0}, ...}, ...)[\$i1, \$i2]`) by subscripts not all literal: the array of
     its elements (`valueForm`), indexed; by literal ones (no element it picks) as it is =#
  all((s::IR.Subscript) -> (s::IR.INDEX).exp isa IR.INTEGER, picked.subscripts) && return e
  return IR.SUBSCRIPTED(valueForm(base, cx), subscripts, e.ty)
end

"""
A record's field read by its position (`absoluteRotation(R1, R2)[1]`: the frontend reads a field of a record-valued
call so) as the record element; a field of a record literal or variable the field itself (`fieldOf`); another
expression as it is.
"""
function fieldExpression(e::IR.Expression)::IR.Expression
  if e isa IR.SUBSCRIPTED && length(e.subscripts) == 1 && e.subscripts[1] isa IR.INDEX
    local position::IR.Expression = (e.subscripts[1]::IR.INDEX).exp
    local rty::IR.ModelicaType = IR.typeOf(e.exp)
    if position isa IR.INTEGER && rty isa IR.T_RECORD
      local f::IR.Field = rty.fields[position.value]
      return fieldExpression(IR.RECORD_ELEMENT(e.exp, position.value, f.name, f.ty))
    end
  end
  if e isa IR.RECORD_ELEMENT && e.exp isa Union{IR.CREF, IR.RECORD}
    local recordType::IR.ModelicaType = IR.typeOf(e.exp)
    recordType isa IR.T_RECORD && return fieldOf(e.exp, e.index, recordType.fields[e.index])
  end
  return e
end

"""
Field `k` (`f`) of a record expression: of a record literal its element, of a reference to a record variable the
field's reference (`R.T`), of another (a call) its record element.
"""
function fieldOf(a::IR.Expression, k::Int, f::IR.Field)::IR.Expression
  a isa IR.RECORD && return a.elements[k]
  a isa IR.CREF && return IR.CREF(IR.ComponentRef(IR.CrefPart[a.cref.parts..., IR.CrefPart(f.name, IR.Subscript[], f.ty)]), f.ty)
  return IR.RECORD_ELEMENT(a, k, f.name, f.ty)
end

"""
The subscripts of a reference `x[outer]` subscripted again by `inner`: a slice (`phi1[1:2]`) or a whole dimension
indexed by the next inner subscript (a literal range's element, or `first + step * (k - 1)`), an index kept, the
inner subscripts left after them appended; `nothing` where a slice is subscripted partly or is no literal range.
"""
function composedSubscripts(outer::Vector{IR.Subscript}, inner::Vector{IR.Subscript})::Union{Nothing, Vector{IR.Subscript}}
  local out::Vector{IR.Subscript} = IR.Subscript[]
  local next::Int = 1
  for s in outer
    if s isa IR.INDEX
      push!(out, s)
      continue
    end
    next <= length(inner) || return nothing
    local k::IR.Expression = (inner[next]::IR.INDEX).exp
    next += 1
    if s isa IR.WHOLE
      push!(out, IR.INDEX(k))
      continue
    end
    local range::Union{Nothing, StepRange{Int, Int}} = IndexReduction.literalRange(IndexReduction.foldTree((s::IR.SLICE).exp))
    range === nothing && return nothing
    push!(out, IR.INDEX(k isa IR.INTEGER ? IR.INTEGER(range[k.value]) :
      IndexReduction.foldTree(IR.BINARY(IR.INTEGER(first(range) - step(range)), IR.ADD,
                                        IR.BINARY(IR.INTEGER(step(range)), IR.MUL, k, IR.T_INTEGER()), IR.T_INTEGER()))))
  end
  append!(out, inner[next:end])
  return out
end

"""
A field of a record value: of a record literal or a record variable the field's value (`valueForm`), of a call of a
Modelica function of the model the record element of the call in its element form. Another as it is.
"""
function recordElementForm(e::IR.RECORD_ELEMENT, cx::ElementContext)::IR.Expression
  local rty::IR.ModelicaType = IR.typeOf(e.exp)
  rty isa IR.T_RECORD || return e
  e.exp isa Union{IR.CREF, IR.RECORD} && return valueForm(fieldOf(e.exp, e.index, rty.fields[e.index]), cx)
  (isModelFunctionCall(cx, e.exp) && !any((f::IR.Field) -> f.ty isa IR.T_RECORD, rty.fields)) || return e
  return IR.RECORD_ELEMENT(functionCallForm(e.exp::IR.CALL, cx), e.index, e.field, e.ty)
end

"""
The value of an expression as a function's argument, in its element form: a scalar's (`elementFormExp`); of a call of
a function of the model or a record element of one, the call's; of another array of a known size (`MAX_ELEMENT_TERMS`
elements at most) the array literal of its elements (`subscriptedForm` of each).
"""
function valueForm(x::IR.Expression, cx::ElementContext)::IR.Expression
  local e::IR.Expression = fieldExpression(x)
  local ty::IR.ModelicaType = IR.typeOf(e)
  ty isa IR.T_ARRAY || return elementFormExp(e, cx)
  isModelFunctionCall(cx, e) && return functionCallForm(e::IR.CALL, cx)
  e isa IR.RECORD_ELEMENT && !(e.exp isa Union{IR.CREF, IR.RECORD}) && return recordElementForm(e, cx)
  all((d::IR.Dimension) -> d isa IR.DIM_SIZE, ty.dimensions) || return e
  local dims::Vector{Int} = Int[(d::IR.DIM_SIZE).size for d in ty.dimensions]
  prod(dims) <= MAX_ELEMENT_TERMS || return e
  return elementArray(e, dims, ty, cx)
end

"The array literal of the elements of the array `e` of dimensions `dims` (`subscriptedForm` of each), its rows nested."
function elementArray(e::IR.Expression, dims::Vector{Int}, ty::IR.T_ARRAY, cx::ElementContext,
                      prefix::Vector{Int} = Int[])::IR.Expression
  local k::Int = length(prefix) + 1
  local elements::Vector{IR.Expression} = IR.Expression[]
  for i in 1:dims[k]
    local index::Vector{Int} = Int[prefix; i]
    push!(elements, k == length(dims) ?
      elementFormExp(IR.SUBSCRIPTED(e, IR.Subscript[IR.INDEX(IR.INTEGER(j)) for j in index], ty.elementType), cx) :
      elementArray(e, dims, ty, cx, index))
  end
  return IR.ARRAY(elements, IR.T_ARRAY(ty.elementType, ty.dimensions[k:end]))
end

"""
A call of a Modelica function of the model with its arguments' values in their element form (`valueForm`); a record
argument as the record literal of its fields' values (records of records as they are).
"""
function functionCallForm(e::IR.CALL, cx::ElementContext)::IR.CALL
  local arguments::Vector{IR.Expression} = IR.Expression[]
  for a in e.arguments
    local ty::IR.ModelicaType = IR.typeOf(a)
    if ty isa IR.T_RECORD && !any((f::IR.Field) -> f.ty isa IR.T_RECORD, ty.fields)
      push!(arguments, IR.RECORD(IR.Expression[valueForm(fieldOf(a, k, f), cx) for (k, f) in enumerate(ty.fields)], ty))
    else
      push!(arguments, valueForm(a, cx))
    end
  end
  return IR.CALL(e.name, arguments, e.ty, e.attributes)
end

"""
A call: sum, product, min or max of an array written out (`reductionForm`); a call of a Modelica function of the model
with its arguments' values (`functionCallForm`); another (`pre` of a reference too) with its arguments in their element
form.
"""
function callForm(e::IR.CALL, cx::ElementContext)::IR.Expression
  local fname::Symbol = e.name
  if e.attributes.builtin && (fname in (:sum, :product) || (fname in (:min, :max) && length(e.arguments) == 1))
    local written::Union{Nothing, IR.Expression} = reductionForm(fname, e.arguments[1], cx)
    written === nothing || return written
  end
  isModelFunctionCall(cx, e) && return functionCallForm(e, cx)
  return IR.mapChildren((x::IR.Expression) -> elementFormExp(x, cx), e)
end

"""
sum, product, min or max (`fname`) of the array `arg` written out: its elements (`arg[i, ...]`, `subscriptedForm`),
row-major, `MAX_ELEMENT_TERMS` at most, combined; sum and product of an empty array 0 and 1. `nothing` for an argument
of a size not known, too many elements, or min and max of an empty array.
"""
function reductionForm(fname::Symbol, arg::IR.Expression, cx::ElementContext)::Union{Nothing, IR.Expression}
  local ty::IR.ModelicaType = IR.typeOf(arg)
  ty isa IR.T_ARRAY || return nothing
  all((d::IR.Dimension) -> d isa IR.DIM_SIZE, ty.dimensions) || return nothing
  local dims::Vector{Int} = Int[(d::IR.DIM_SIZE).size for d in ty.dimensions]
  local n::Int = prod(dims)
  n <= MAX_ELEMENT_TERMS || return nothing
  n == 0 && return fname === :product ? IR.REAL(1.0) : fname === :sum ? IR.REAL(0.0) : nothing
  local terms::Vector{IR.Expression} = IR.Expression[]
  for t in Iterators.product(reverse([1:d for d in dims])...)   #= row-major: the last index fastest =#
    local subscripts::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(IR.INTEGER(i)) for i in reverse(collect(t))]
    push!(terms, elementFormExp(IR.SUBSCRIPTED(arg, subscripts, ty.elementType), cx))
  end
  local el::IR.ModelicaType = ty.elementType
  fname === :sum && return foldl((a::IR.Expression, b::IR.Expression) -> IR.BINARY(a, IR.ADD, b, el), terms)
  fname === :product && return foldl((a::IR.Expression, b::IR.Expression) -> IR.BINARY(a, IR.MUL, b, el), terms)
  local attributes::IR.CallAttributes = IR.CallAttributes(true, false, false)
  return foldl((a::IR.Expression, b::IR.Expression) -> IR.CALL(fname, IR.Expression[a, b], el, attributes), terms)
end

"""
A scalar product of two vectors of a known size (MultiBody's `tau = -frame_b.t * e`; `MAX_ELEMENT_TERMS` elements at
most) written out: the sum of the products of their elements (`subscriptedForm`). Another as it is, its operands in
their element form.
"""
function scalarProductForm(e::IR.BINARY, cx::ElementContext)::IR.Expression
  local ty1::IR.ModelicaType = IR.typeOf(e.exp1)
  local ty2::IR.ModelicaType = IR.typeOf(e.exp2)
  (ty1 isa IR.T_ARRAY && ty2 isa IR.T_ARRAY && length(ty1.dimensions) == 1 && ty1.dimensions[1] isa IR.DIM_SIZE &&
   (ty1.dimensions[1]::IR.DIM_SIZE).size <= MAX_ELEMENT_TERMS) ||
    return IR.mapChildren((x::IR.Expression) -> elementFormExp(x, cx), e)
  local terms::Vector{IR.Expression} = IR.Expression[]
  for k in 1:(ty1.dimensions[1]::IR.DIM_SIZE).size
    local index::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(IR.INTEGER(k))]
    push!(terms, IR.BINARY(elementFormExp(IR.SUBSCRIPTED(e.exp1, index, ty1.elementType), cx), IR.MUL,
                           elementFormExp(IR.SUBSCRIPTED(e.exp2, index, ty2.elementType), cx), e.ty))
  end
  isempty(terms) && return IR.REAL(0.0)
  return foldl((a::IR.Expression, b::IR.Expression) -> IR.BINARY(a, IR.ADD, b, e.ty), terms)
end
