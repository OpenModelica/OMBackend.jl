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

#= The passes before the lowering to SimCode (the previous backend's Causalize), in its order:
   the Integer and enumeration variables bound to literals as parameters, the subscripts of
   der() moved into its argument, higher derivatives as first ones of auxiliary variables;
   after the derivative variables (derivatives.jl): the elements of vectors chosen at run
   time, the references of bindings resolved, the if-expressions as if-equations of auxiliary
   variables, the record field arrays as their elements, the record and array equations as
   scalar ones, the reductions over constant ranges unrolled.

   They reach an equation's expressions, those of a when-equation's statements included, and
   not the statements of an algorithm section: its code generation takes them as written. =#

#= ==================================================================================== =#
#= The expressions of an equation                                                       =#
#= ==================================================================================== =#

"""
    mapRoots!(g, eq)

`eq` with each of its root expressions `e` (the sides, the conditions and triggers, the
expressions of a when-equation's statements, a for-equation's range, an assert's arguments)
replaced by `g(e)`, in place, and the equations in it in turn: `g` rewrites a whole expression
and descends itself. The statements of an algorithm section are left as written.
"""
function mapRoots!(g::G, eq::Union{IR.SCALAR_EQUATION, IR.ARRAY_EQUATION, IR.RECORD_EQUATION})::IR.Equation where {G}
  eq.lhs = g(eq.lhs)
  eq.rhs = g(eq.rhs)
  return eq
end
mapRoots!(g::G, eq::IR.ALGORITHM) where {G} = eq
function mapRoots!(g::G, eq::IR.IF_EQUATION)::IR.Equation where {G}
  eq.branches = IR.EquationBranch[IR.EquationBranch(g(b.condition), b.equations) for b in eq.branches]
  for b in eq.branches
    foreach(e -> mapRoots!(g, e), b.equations)
  end
  foreach(e -> mapRoots!(g, e), eq.elseEquations)
  return eq
end
function mapRoots!(g::G, eq::IR.FOR_EQUATION)::IR.Equation where {G}
  eq.iterator = IR.Iterator(eq.iterator.name, g(eq.iterator.range))
  foreach(e -> mapRoots!(g, e), eq.body)
  return eq
end
function mapRoots!(g::G, eq::IR.WHEN_EQUATION)::IR.Equation where {G}
  eq.branches = IR.WhenBranch[IR.WhenBranch(IR.Expression[g(t) for t in b.triggers], b.atInitial, b.atTerminal,
                                            mapRoots(g, b.body)) for b in eq.branches]
  return eq
end
function mapRoots!(g::G, eq::IR.ASSERT_EQUATION)::IR.Equation where {G}
  eq.condition = g(eq.condition)
  eq.message = g(eq.message)
  eq.level = g(eq.level)
  return eq
end

"The statements with `g` applied to their root expressions (`mapRoots!`); the targets of `reinit` and of OM.jl's structural changes kept."
mapRoots(g::G, body::Vector{IR.Statement}) where {G} = IR.Statement[mapRoots(g, s) for s in body]
mapRoots(g::G, s::IR.ASSIGNMENT) where {G} = IR.ASSIGNMENT(g(s.lhs), g(s.rhs), s.source)
mapRoots(g::G, s::IR.IF_STATEMENT) where {G} =
  IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(g(b.condition), mapRoots(g, b.body)) for b in s.branches],
                  mapRoots(g, s.elseBody), s.source)
mapRoots(g::G, s::IR.FOR_STATEMENT) where {G} =
  IR.FOR_STATEMENT(IR.Iterator(s.iterator.name, g(s.iterator.range)), mapRoots(g, s.body), s.source)
mapRoots(g::G, s::IR.WHILE_STATEMENT) where {G} = IR.WHILE_STATEMENT(g(s.condition), mapRoots(g, s.body), s.source)
mapRoots(g::G, s::IR.WHEN_STATEMENT) where {G} =
  IR.WHEN_STATEMENT(IR.WhenBranch[IR.WhenBranch(IR.Expression[g(t) for t in b.triggers], b.atInitial, b.atTerminal,
                                                mapRoots(g, b.body)) for b in s.branches], s.source)
mapRoots(g::G, s::IR.ASSERT) where {G} = IR.ASSERT(g(s.condition), g(s.message), g(s.level), s.source)
mapRoots(g::G, s::IR.TERMINATE) where {G} = IR.TERMINATE(g(s.message), s.source)
mapRoots(g::G, s::IR.REINIT) where {G} = IR.REINIT(s.state, g(s.value), s.source)
mapRoots(g::G, s::IR.NORETCALL) where {G} = IR.NORETCALL(g(s.call), s.source)
mapRoots(g::G, s::IR.RECOMPILATION) where {G} = IR.RECOMPILATION(s.component, g(s.newValue))
mapRoots(g::G, s::Union{IR.RETURN, IR.BREAK, IR.AGENTIC_RECOMPILATION}) where {G} = s

"The continuous-time partitions and the structural modes' systems, each with its variables."
partitionsWithVariables(model::IR.Model) =
  Iterators.flatten((((model.variables, p) for p in model.continuous), ((m.variables, m.system) for m in model.modes)))

#= ==================================================================================== =#
#= Integer and enumeration variables bound to literals                                  =#
#= ==================================================================================== =#

"""
    resolveIntegerVariables!(model)

The Integer and enumeration variables (and elements of arrays of them) a literal binds, as
parameters of its value: a variable `i` defined by `i = literal` (or `literal = i`), or by an
alias `i = j` of one so defined, has no continuous derivative and its equation goes (as a
variable with the equation, MSL Digital's `auxiliary` tables unbalanced the system). A variable
a when-equation or an algorithm assigns is discrete state and stays, as one another equation
defines (`y = pre(auxiliary_n)`). An assignment to an element of an array keeps all of its
elements (`elementBase`).
"""
function resolveIntegerVariables!(model::IR.Model)::IR.Model
  for (variables, p) in partitionsWithVariables(model)
    resolveIntegerVariables!(variables, p)
  end
  return model
end

function resolveIntegerVariables!(variables::IR.VariableSet, partition::IR.Partition)::Nothing
  local isCandidate = v -> v.kind isa IR.ALGEBRAIC && IR.elementType(v.ty) isa Union{IR.T_INTEGER, IR.T_ENUMERATION}
  local names = Set{Symbol}(v.name for v in variables if isCandidate(v))
  isempty(names) && return nothing
  local namesAndBases = union(names, Set{Symbol}(elementBase(v.cref) for v in variables if isCandidate(v)))
  local assigned = Set{Symbol}()
  for eq in partition.equations
    if eq isa IR.WHEN_EQUATION
      foreach(b -> addAssignedNames!(assigned, b.body, namesAndBases; nested = false), eq.branches)
    elseif eq isa IR.ALGORITHM
      addAssignedNames!(assigned, eq.statements, namesAndBases; nested = true)
    end
  end
  local isLiteral = e -> e isa Union{IR.INTEGER, IR.ENUM_LITERAL}
  local nameOf = e -> e isa IR.CREF ? e.cref.name : nothing
  local isFree = n -> n !== nothing && n in names && !(n in assigned)
  local keep = trues(length(partition.equations))
  local values = Dict{Symbol, IR.Expression}()
  local defined = Set{Symbol}()
  for (i, eq) in enumerate(partition.equations)
    eq isa IR.SCALAR_EQUATION || continue
    local (l, r) = (nameOf(eq.lhs), nameOf(eq.rhs))
    if isFree(l) && isLiteral(eq.rhs)
      keep[i] = false; push!(defined, l); values[l] = eq.rhs
    elseif isFree(r) && isLiteral(eq.lhs)
      keep[i] = false; push!(defined, r); values[r] = eq.lhs
    end
  end
  #= Aliases `a = b` of variables so defined, until none is left (chains of them) =#
  local progress = true
  while progress
    progress = false
    for (i, eq) in enumerate(partition.equations)
      (keep[i] && eq isa IR.SCALAR_EQUATION) || continue
      local (l, r) = (nameOf(eq.lhs), nameOf(eq.rhs))
      (isFree(l) && isFree(r)) || continue
      for (target, source) in ((l, r), (r, l))
        if source in defined && !(target in defined)
          keep[i] = false
          push!(defined, target)
          haskey(values, source) && (values[target] = values[source])
          progress = true
          break
        end
      end
    end
  end
  for v in variables
    isCandidate(v) || continue
    local base = elementBase(v.cref)
    (v.name in assigned || base in assigned) && continue
    (v.name in defined || base in defined) || continue
    v.kind = IR.PARAMETER()
    v.ty isa Union{IR.T_INTEGER, IR.T_ENUMERATION} && (v.binding = get(values, v.name, IR.INTEGER(0)))
  end
  partition.equations = partition.equations[keep]
  return nothing
end

"""
    elementBase(cref)

The name of the array the previous backend took an element to be of: its name up to the
first subscript, of an element whose last part has subscripts (`x[2]` of `x`, `a.x[2]` of
`a.x`, `a[1].x[2]` of `a`); the reference's own name otherwise.
"""
function elementBase(cref::IR.ComponentRef)::Symbol
  isempty(IR.subscripts(cref)) && return cref.name
  local k = findfirst(p -> !isempty(p.subscripts), cref.parts)
  local parts = IR.CrefPart[cref.parts[1:k-1]..., IR.CrefPart(cref.parts[k].name, IR.Subscript[], cref.parts[k].ty)]
  return IR.ComponentRef(parts).name
end

"""
    addAssignedNames!(names, body, candidates; nested)

The names in `candidates` that the statements `body` assign, a target's own and its
`elementBase`; in the branches of if-statements, and if `nested` in loops too (a
when-equation's body, an algorithm's).
"""
function addAssignedNames!(names::Set{Symbol}, body::Vector{IR.Statement}, candidates::Set{Symbol}; nested::Bool)::Nothing
  for s in body
    if s isa IR.ASSIGNMENT && s.lhs isa IR.CREF
      s.lhs.cref.name in candidates && push!(names, s.lhs.cref.name)
      local base = elementBase(s.lhs.cref)
      base in candidates && push!(names, base)
    elseif s isa IR.IF_STATEMENT
      foreach(b -> addAssignedNames!(names, b.body, candidates; nested), s.branches)
      addAssignedNames!(names, s.elseBody, candidates; nested)
    elseif nested && s isa Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT}
      addAssignedNames!(names, s.body, candidates; nested)
    end
  end
  return nothing
end

#= ==================================================================================== =#
#= Subscripts of der()                                                                  =#
#= ==================================================================================== =#

"""
    moveSubscriptsIntoDer!(model)

`der(x)[i]` as `der(x[i])` (the derivative of an element is the element of the derivative),
and an array value's element at a literal index as that element (`{a, b}[2]` as `b`), in the
equations of the continuous-time partitions and the structural modes.
"""
function moveSubscriptsIntoDer!(model::IR.Model)::IR.Model
  local g = e -> IR.mapExp(moveSubscriptIntoDer, e)
  for (_, p) in partitionsWithVariables(model)
    foreach(eq -> mapRoots!(g, eq), p.equations)
  end
  return model
end

function moveSubscriptIntoDer(e::IR.Expression)::IR.Expression
  e isa IR.SUBSCRIPTED || return e
  if IR.isBuiltinCall(e.exp, :der) && length(e.exp.arguments) == 1
    local element = elementOfArrayValue(IR.SUBSCRIPTED(e.exp.arguments[1], e.subscripts, e.ty))
    return IR.CALL(:der, IR.Expression[element], e.ty, e.exp.attributes)
  end
  return elementOfArrayValue(e)
end

"`{e1, e2, ...}[k]` as `ek` for a literal `k`; another expression itself."
function elementOfArrayValue(e::IR.SUBSCRIPTED)::IR.Expression
  (e.exp isa IR.ARRAY && length(e.subscripts) == 1 && e.subscripts[1] isa IR.INDEX &&
   e.subscripts[1].exp isa IR.INTEGER) || return e
  local k = e.subscripts[1].exp.value
  if !(1 <= k <= length(e.exp.elements))
    @warn "The index $k is outside the array $(IR.toString(e.exp))"
    return e
  end
  return e.exp.elements[k]
end

#= ==================================================================================== =#
#= Higher derivatives                                                                   =#
#= ==================================================================================== =#

"""
    lowerHigherOrderDerivatives!(model)

A derivative of a derivative as the first derivative of an auxiliary variable:
`der(der(x))` is `der(der_x)` with the variable `der_x` and the equation `der_x = der(x)`
(after the others; once per variable), recursively for higher ones. In the equations of the
continuous-time partitions and the structural modes.
"""
function lowerHigherOrderDerivatives!(model::IR.Model)::IR.Model
  for (variables, p) in partitionsWithVariables(model)
    local auxiliaries = Dict{String, IR.ComponentRef}()
    local equations = IR.Equation[]
    local g = e -> IR.mapExp(x -> lowerNestedDer(x, variables, auxiliaries, equations), e)
    foreach(eq -> mapRoots!(g, eq), p.equations)
    append!(p.equations, equations)
  end
  return model
end

isDerOf(e::IR.Expression)::Bool = IR.isBuiltinCall(e, :der) && length(e.arguments) == 1

"Whether `e` is `der(...der(x)...)` of a reference `x`."
endsInReference(e::IR.Expression)::Bool = e isa IR.CREF || (isDerOf(e) && endsInReference(e.arguments[1]))

function lowerNestedDer(e::IR.Expression, variables::IR.VariableSet, auxiliaries::Dict{String, IR.ComponentRef},
                        equations::Vector{IR.Equation})::IR.Expression
  (isDerOf(e) && isDerOf(e.arguments[1]) && endsInReference(e.arguments[1].arguments[1])) || return e
  local aux = derivativeVariable(e.arguments[1], e.attributes, variables, auxiliaries, equations)
  return IR.CALL(:der, IR.Expression[IR.CREF(aux, IR.T_REAL())], e.ty, e.attributes)
end

"""
    derivativeVariable(d, attributes, variables, auxiliaries, equations)

The auxiliary variable equal to the derivative `d = der(a)`: `der_<a>` (of `a` lowered first
if itself a derivative), with its equation `der_<a> = der(a)`; one per variable.
"""
function derivativeVariable(d::IR.CALL, attributes::IR.CallAttributes, variables::IR.VariableSet,
                            auxiliaries::Dict{String, IR.ComponentRef}, equations::Vector{IR.Equation})::IR.ComponentRef
  local arg = d.arguments[1]
  local base = arg isa IR.CREF ? arg.cref : derivativeVariable(arg, attributes, variables, auxiliaries, equations)
  local baseName = legacyName(base)
  local existing = get(auxiliaries, baseName, nothing)
  existing === nothing || return existing
  local aux = IR.ComponentRef(Symbol("der_", baseName), IR.T_REAL())
  push!(variables, IR.Variable(aux, IR.T_REAL(), IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                               false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :lowerHigherOrderDerivatives))
  push!(equations, IR.SCALAR_EQUATION(IR.CREF(aux, IR.T_REAL()),
                                      IR.CALL(:der, IR.Expression[IR.CREF(base, IR.T_REAL())], IR.T_REAL(), attributes),
                                      IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
  auxiliaries[baseName] = aux
  return aux
end

#= ==================================================================================== =#
#= Run-time indices, binding references                                                 =#
#= ==================================================================================== =#

"""
    chooseRuntimeIndices!(model)

An element of a vector selected by an index known only at run time (MSL MultiSwitch's
`expr[firstActiveIndex]`) as the choice `if i == 1 then x[1] elseif ... else x[n]`: the
elements are separate variables. An index outside `1:n` (an error in Modelica) selects `x[n]`.
An iterator (of a reduction, an array constructor, a for-equation) as the index stays; the
unrolling replaces it. In the equations of the continuous-time partitions and the structural
modes, and the variables' bindings.
"""
function chooseRuntimeIndices!(model::IR.Model)::IR.Model
  for (variables, p) in partitionsWithVariables(model)
    for eq in p.equations
      local iterators = iteratorNames(eq)
      mapRoots!(e -> IR.mapExp(x -> runtimeIndexChoice(x, iterators), e), eq)
    end
    for v in variables
      v.binding === nothing && continue
      local iterators = iteratorNames(v.binding)
      v.binding = IR.mapExp(x -> runtimeIndexChoice(x, iterators), v.binding)
    end
  end
  return model
end

"The names of the iterators of the reductions, array constructors and for-equations in `x`."
function iteratorNames(x::Union{IR.Expression, IR.Equation})::Set{Symbol}
  local names = Set{Symbol}()
  local visit = e -> e isa Union{IR.REDUCTION, IR.ARRAY_CONSTRUCTOR} && foreach(it -> push!(names, it.name), e.iterators)
  if x isa IR.Expression
    IR.foreachExp(visit, x)
  else
    IR.foreachExp(visit, x)
    addForIterators!(names, x)
  end
  return names
end

addForIterators!(names::Set{Symbol}, eq::IR.FOR_EQUATION) =
  (push!(names, eq.iterator.name); foreach(e -> addForIterators!(names, e), eq.body))
addForIterators!(names::Set{Symbol}, eq::IR.IF_EQUATION) =
  (foreach(b -> foreach(e -> addForIterators!(names, e), b.equations), eq.branches);
   foreach(e -> addForIterators!(names, e), eq.elseEquations))
addForIterators!(names::Set{Symbol}, eq::IR.Equation) = nothing

function runtimeIndexChoice(e::IR.Expression, iterators::Set{Symbol})::IR.Expression
  e isa IR.CREF || return e
  local last = e.cref.parts[end]
  (length(last.subscripts) == 1 && last.subscripts[1] isa IR.INDEX && last.subscripts[1].exp isa IR.CREF) || return e
  local index = last.subscripts[1].exp
  (length(index.cref.parts) == 1 && index.cref.name in iterators) && return e
  local ty = last.ty
  (ty isa IR.T_ARRAY && length(ty.dimensions) == 1 && ty.dimensions[1] isa IR.DIM_SIZE) || return e
  local n = ty.dimensions[1].size
  n > 0 || return e
  local element = k -> IR.CREF(IR.ComponentRef(IR.CrefPart[e.cref.parts[1:end-1]...,
                                                           IR.CrefPart(last.name, IR.Subscript[IR.INDEX(IR.INTEGER(k))], ty)]),
                               ty.elementType)
  local choice::IR.Expression = element(n)
  for k in n-1:-1:1
    choice = IR.IF_EXPRESSION(IR.RELATION(index, IR.EQUAL, IR.INTEGER(k), 0), element(k), choice, ty.elementType)
  end
  return choice
end

"""
    resolveBindingReferences!(model)

A binding that references a variable with a binding is that binding (`p2 = p1` with
`p1 = 3` is `p2 = 3`), through chains of references, unless the referenced variable is tunable
(withTunableParameters: its value may change). The reference is looked up by its name without
its last subscripts (the previous backend's): a reference to an array's element stays.
"""
function resolveBindingReferences!(model::IR.Model)::IR.Model
  for (variables, _) in partitionsWithVariables(model)
    resolveBindingReferences!(variables)
  end
  return model
end

function resolveBindingReferences!(variables::IR.VariableSet)::Nothing
  local bindings = Dict{String, IR.Expression}()
  for v in variables
    v.binding === nothing || (bindings[legacyName(v.cref)] = v.binding)
  end
  for _ in 1:100
    local changed = false
    for v in variables
      v.binding isa IR.CREF || continue
      local target = legacyName(withoutLastSubscripts(v.binding.cref))
      isTunableParameter(target) && continue
      local b = get(bindings, target, nothing)
      (b === nothing || b isa IR.CREF) && continue
      v.binding = b
      bindings[legacyName(v.cref)] = b
      changed = true
    end
    changed || return nothing
  end
  @warn "resolveBindingReferences!: no fixed point after 100 rounds (circular parameter bindings?)"
  return nothing
end

withoutLastSubscripts(cref::IR.ComponentRef)::IR.ComponentRef =
  IR.ComponentRef(IR.CrefPart[cref.parts[1:end-1]..., IR.CrefPart(cref.parts[end].name, IR.Subscript[], cref.parts[end].ty)])

#= ==================================================================================== =#
#= If-expressions                                                                       =#
#= ==================================================================================== =#

"""
    liftIfExpressions!(model)

The if-expressions of the equations as if-equations of auxiliary variables (after the
others): `if c then a else b` is the Real `ifEq_tmp<k>` with
`if c then ifEq_tmp<k> = a else ifEq_tmp<k> = b`, whose condition makes events. An elseif
chain (an else-branch that is an if-expression with events) is one if-equation. Equal
if-expressions share one variable (two zero-crossing callbacks of one condition in MTK leave
the other's parameter pinned). Stays inline: an if-expression in `noEvent` or of a condition
whose relations all are (no events, MLS 8.5); of a record value (the variable is a Real); in
the branches of a lifted one, one whose condition does not depend on time (a state-dependent
condition crosses in both directions: its branches are evaluated as a continuous product).
The when-equations (their code generation evaluates them in place) and algorithms are left.
"""
function liftIfExpressions!(model::IR.Model)::IR.Model
  for (variables, p) in partitionsWithVariables(model)
    local lifter = IfExpressionLifter(variables, Dict{Any, IR.CREF}(), IR.Equation[], Ref(0))
    for eq in p.equations
      (eq isa IR.WHEN_EQUATION && eq.kind !== IR.STRUCTURAL_WHEN) && continue
      mapRoots!(e -> lift(lifter, e, false), eq)
    end
    append!(p.equations, lifter.equations)
  end
  return model
end

"The state of the lifting in a partition: the lifted if-expressions' variables (by shape), their equations, the next number."
struct IfExpressionLifter
  variables::IR.VariableSet
  lifted::Dict{Any, IR.CREF}
  equations::Vector{IR.Equation}
  tick::Base.RefValue{Int}
end

"""
    lift(lifter, e, timeDependentOnly)

`e` with its if-expressions lifted; `timeDependentOnly` in the branches of a lifted one.
"""
function lift(lifter::IfExpressionLifter, e::IR.Expression, timeDependentOnly::Bool)::IR.Expression
  IR.isBuiltinCall(e, :noEvent) && return e
  e isa IR.IF_EXPRESSION || return IR.mapChildren(x -> lift(lifter, x, timeDependentOnly), e)
  local inner = x -> lift(lifter, x, true)
  if (timeDependentOnly && !dependsOnTime(e.condition)) || relationsAllInNoEvent(e.condition) ||
     isRecordValued(e.thenExp) || isRecordValued(e.elseExp)
    return IR.IF_EXPRESSION(inner(e.condition), inner(e.thenExp), inner(e.elseExp), e.ty)
  end
  local conditions = IR.Expression[inner(e.condition)]
  local values = IR.Expression[inner(e.thenExp)]
  local rest = e.elseExp
  while rest isa IR.IF_EXPRESSION && !relationsAllInNoEvent(rest.condition)
    push!(conditions, inner(rest.condition))
    push!(values, inner(rest.thenExp))
    rest = rest.elseExp
  end
  local elseValue = inner(rest)
  local key = (conditions, values, elseValue)
  local existing = get(lifter.lifted, key, nothing)
  existing === nothing || return existing
  local name = Symbol("ifEq_tmp", lifter.tick[])
  lifter.tick[] += 1
  local v = IR.CREF(IR.ComponentRef(name, IR.T_REAL()), IR.T_REAL())
  push!(lifter.variables, IR.Variable(name, IR.T_REAL(), IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                                      false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :liftIfExpressions))
  local assign = x -> IR.Equation[IR.SCALAR_EQUATION(v, x, IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)]
  push!(lifter.equations, IR.IF_EQUATION(IR.EquationBranch[IR.EquationBranch(c, assign(x)) for (c, x) in zip(conditions, values)],
                                         assign(elseValue), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
  lifter.lifted[key] = v
  return v
end

dependsOnTime(e::IR.Expression)::Bool = IR.anyExp(isTime, e)

"Whether every relation of `e` is inside `noEvent`, and `e` has a `noEvent`: `e` makes no events."
function relationsAllInNoEvent(e::IR.Expression)::Bool
  local (sawNoEvent, bare) = (false, false)
  local visit
  visit = x -> begin
    if IR.isBuiltinCall(x, :noEvent)
      sawNoEvent = true
    elseif x isa IR.RELATION
      bare = true
    else
      foreach(visit, IR.children(x))
    end
  end
  visit(e)
  return sawNoEvent && !bare
end

"A record value: a reference, a call or a record of a record type, or an if-expression of one."
isRecordValued(e::IR.Expression)::Bool =
  e isa IR.IF_EXPRESSION ? isRecordValued(e.thenExp) || isRecordValued(e.elseExp) :
  e isa Union{IR.CREF, IR.CALL, IR.RECORD} && IR.typeOf(e) isa IR.T_RECORD

#= ==================================================================================== =#
#= Record field arrays                                                                  =#
#= ==================================================================================== =#

"""
    expandRecordFieldArrays!(model)

The variables that are an array field of a record (`R.T` of a `Frames.Orientation R`) as
their elements' variables, as the frontend scalarizes other arrays; for a parameter or a
constant also the array itself (`R.T`, its binding assembled from its elements' bindings, a
missing one 0.0): the code generation resolves a function call's array argument by it. The
variables of elements the frontend scalarized stay, their array added in front of the
variables.
"""
function expandRecordFieldArrays!(model::IR.Model)::IR.Model
  for (variables, _) in partitionsWithVariables(model)
    expandRecordFieldArrays!(variables)
  end
  return model
end

function expandRecordFieldArrays!(variables::IR.VariableSet)::Nothing
  any(v -> isRecordFieldArray(v.cref), variables) || return nothing
  local out = IR.Variable[]
  #= The array of each group of scalarized elements of a known array field, by the array's
     name, with its elements' bindings. A Dict: the arrays go in front in its order (the
     previous backend's). =#
  local groups = Dict{String, Tuple{IR.ComponentRef, IR.Variable, Dict{Vector{Int}, IR.Expression}}}()
  for v in variables
    if !isRecordFieldArray(v.cref)
      push!(out, v)
    elseif !isempty(v.cref.parts[end].subscripts)
      push!(out, v)
      local array = withoutLastSubscripts(v.cref)
      (IR.isKnown(v) && all(d -> d isa IR.DIM_SIZE, array.parts[end].ty.dimensions)) || continue
      local (_, _, elements) = get!(groups, legacyName(array), (array, v, Dict{Vector{Int}, IR.Expression}()))
      local index = literalIndices(v.cref.parts[end].subscripts)
      (index === nothing || v.binding === nothing) || (elements[index] = v.binding)
    else
      append!(out, recordFieldArrayElements(v))
    end
  end
  for (name, (array, template, elements)) in groups
    any(v -> v.name === array.name, out) && continue
    local ty = array.parts[end].ty
    local sizes = Int[d.size for d in ty.dimensions]
    pushfirst!(out, IR.Variable(array, ty, template.kind, assembledBinding(sizes, elements, name), template.attributes,
                                template.direction, template.isProtected, template.connector, template.source,
                                :expandRecordFieldArrays))
  end
  empty!(variables)
  foreach(v -> push!(variables, v), out)
  return nothing
end

"Whether `cref` is an array field of a record: its last part of an array type, the one before of a record type."
isRecordFieldArray(cref::IR.ComponentRef)::Bool =
  length(cref.parts) >= 2 && cref.parts[end].ty isa IR.T_ARRAY && cref.parts[end-1].ty isa IR.T_RECORD

"The literal Integer indices of `subscripts`; `nothing` if one is not."
function literalIndices(subscripts::Vector{IR.Subscript})::Union{Nothing, Vector{Int}}
  all(s -> s isa IR.INDEX && s.exp isa IR.INTEGER, subscripts) || return nothing
  return Int[s.exp.value for s in subscripts]
end

"""
    assembledBinding(sizes, elements, name)

The binding of an array of dimensions `sizes` from its elements' bindings (by index); a
missing element 0.0 (as an unbound parameter, with a warning: a binding the frontend did not
give); `nothing` without any.
"""
function assembledBinding(sizes::Vector{Int}, elements::Dict{Vector{Int}, IR.Expression}, name::String)::Union{Nothing, IR.Expression}
  isempty(elements) && return nothing
  local missing = [collect(Tuple(i)) for i in CartesianIndices(Tuple(sizes)) if !haskey(elements, collect(Tuple(i)))]
  isempty(missing) || @warn "$(length(missing)) of $(prod(sizes)) element bindings missing for the array $(name); 0.0 for them" missing
  local build
  build = (prefix::Vector{Int}) -> begin
    local k = length(prefix) + 1
    k > length(sizes) && return get(elements, prefix, IR.REAL(0.0))
    return IR.ARRAY(IR.Expression[build(vcat(prefix, i)) for i in 1:sizes[k]],
                    IR.T_ARRAY(IR.T_REAL(), IR.Dimension[IR.DIM_SIZE(s) for s in sizes[k:end]]))
  end
  return build(Int[])
end

"""
    recordFieldArrayElements(v)

The variables of the elements of the record field array `v` (in the order of Julia's
`CartesianIndices`, the first index fastest), each with its element of `v`'s binding; for a
parameter or a constant `v` itself first. `v` alone for an array of a dimension that is not a
size.
"""
function recordFieldArrayElements(v::IR.Variable)::Vector{IR.Variable}
  local last = v.cref.parts[end]
  local ty = last.ty::IR.T_ARRAY
  if !all(d -> d isa IR.DIM_SIZE, ty.dimensions)
    @warn "A record field array of a dimension that is not a size: $(v.name) stays an array"
    return IR.Variable[v]
  end
  local out = IR.isKnown(v) ? IR.Variable[v] : IR.Variable[]
  for i in CartesianIndices(Tuple(d.size for d in ty.dimensions))
    local index = collect(Tuple(i))
    local cref = IR.ComponentRef(IR.CrefPart[v.cref.parts[1:end-1]...,
                                             IR.CrefPart(last.name, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index], ty)])
    local binding = v.binding === nothing ? nothing : bindingElement(v.binding, index)
    push!(out, IR.Variable(cref, ty.elementType, v.kind, binding, v.attributes, v.direction, v.isProtected, v.connector,
                           v.source, :expandRecordFieldArrays))
  end
  return out
end

"""
The element `index` of an array binding: of an array value its element (`nothing` outside
it), of another expression the subscripted expression.
"""
function bindingElement(b::IR.Expression, index::Vector{Int})::Union{Nothing, IR.Expression}
  isempty(index) && return b
  if b isa IR.ARRAY
    1 <= index[1] <= length(b.elements) || return nothing
    return bindingElement(b.elements[index[1]], index[2:end])
  end
  return IR.SUBSCRIPTED(b, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index], IR.elementType(IR.typeOf(b)))
end

#= ==================================================================================== =#
#= Record and array equations                                                           =#
#= ==================================================================================== =#

"""
    expandEquations!(model)

The record and array equations of the continuous-time partitions and the structural modes as
scalar equations; of the initial equations the record equations (the initial-equation code
generation takes array equations). A record equation with a reference on a side is one
equation per field (per element of an array field) of the reference: `r.f = e[k]` (`f` the
`k`-th field), the reference on the left. Another, as an array equation, is one per element:
`a[i, j] = b[i, j]`, each side's element taken from an array value's elements (a cast's, an
array constructor's over a constant range), from a reference by its subscripts, from another
expression subscripted; a record value's `k`-th element is its `k`-th field.
"""
function expandEquations!(model::IR.Model)::IR.Model
  for (_, p) in partitionsWithVariables(model)
    p.equations = expandEquations(p.equations; arrays = true)
    p.initialEquations = expandEquations(p.initialEquations; arrays = false)
  end
  return model
end

function expandEquations(equations::Vector{IR.Equation}; arrays::Bool)::Vector{IR.Equation}
  local out = IR.Equation[]
  for eq in equations
    if eq isa IR.RECORD_EQUATION
      local fields = fieldEquations(eq)
      append!(out, fields !== nothing ? fields : elementEquations(eq, [[k] for k in 1:eq.size], eq.ty))
    elseif arrays && eq isa IR.ARRAY_EQUATION
      local sizes = Int[knownSize(d, eq) for d in eq.ty.dimensions]
      #= the last index fastest, as the elements of an array value =#
      local indices = [reverse(collect(Tuple(i))) for i in CartesianIndices(Tuple(reverse(sizes)))]
      append!(out, elementEquations(eq, vec(indices), eq.ty))
    else
      push!(out, eq)
    end
  end
  return out
end

"The size of an equation's dimension; an unknown one is not supported (its equations would be left out)."
function knownSize(d::IR.Dimension, eq::IR.Equation)::Int
  local n = IR.dimensionSize(d)
  n === nothing && unsupported("an array equation of a dimension of unknown size", IR.toString(eq))
  return n
end

"""
    fieldEquations(eq)

The record equation `eq` as one equation per field of the reference on one side (the left
one first): `r.f = e[k]`; `nothing` if neither side is a reference of a record type.
"""
function fieldEquations(eq::IR.RECORD_EQUATION)::Union{Nothing, Vector{IR.Equation}}
  local (r, e) = isRecordReference(eq.lhs) ? (eq.lhs, eq.rhs) : isRecordReference(eq.rhs) ? (eq.rhs, eq.lhs) : (nothing, nothing)
  r === nothing && return nothing
  local out = IR.Equation[]
  for (k, field) in enumerate(r.ty.fields)
    local value = fieldByIndex(e, k)
    local fieldRef = IR.CrefPart[r.cref.parts..., IR.CrefPart(field.name, IR.Subscript[], field.ty)]
    if field.ty isa IR.T_ARRAY
      for i in CartesianIndices(Tuple(knownSize(d, eq) for d in field.ty.dimensions))
        local subs = IR.Subscript[IR.INDEX(IR.INTEGER(j)) for j in Tuple(i)]
        local elementRef = IR.ComponentRef(IR.CrefPart[fieldRef[1:end-1]..., IR.CrefPart(field.name, subs, field.ty)])
        push!(out, IR.SCALAR_EQUATION(IR.CREF(elementRef, field.ty.elementType),
                                      IR.SUBSCRIPTED(value, subs, field.ty.elementType), field.ty.elementType,
                                      eq.source, eq.attributes))
      end
    else
      push!(out, IR.SCALAR_EQUATION(IR.CREF(IR.ComponentRef(fieldRef), field.ty), value, field.ty, eq.source, eq.attributes))
    end
  end
  return out
end

isRecordReference(e::IR.Expression)::Bool = e isa IR.CREF && e.ty isa IR.T_RECORD

"""
    elementEquations(eq, indices, ty)

The equations `lhs[i] = rhs[i]` of `eq` for each index of `indices` (in order), its sides of
type `ty` (an array's elements of its element type, a record's of its fields' types).
"""
function elementEquations(eq::Union{IR.ARRAY_EQUATION, IR.RECORD_EQUATION}, indices::Vector{Vector{Int}},
                          ty::IR.ModelicaType)::Vector{IR.Equation}
  local (lhsElements, rhsElements) = (arrayElements(eq.lhs), arrayElements(eq.rhs))
  local out = IR.Equation[]
  for (n, index) in enumerate(indices)
    local l = lhsElements !== nothing && n <= length(lhsElements) ? lhsElements[n] : elementOf(eq.lhs, index)
    local r = rhsElements !== nothing && n <= length(rhsElements) ? rhsElements[n] : elementOf(eq.rhs, index)
    local elementTy = ty isa IR.T_RECORD && index[1] <= length(ty.fields) ? ty.fields[index[1]].ty :
                      elementTypeOf(ty, length(index))
    push!(out, IR.SCALAR_EQUATION(l, r, elementTy, eq.source, eq.attributes))
  end
  return out
end

"""
    arrayElements(e)

The leaf elements of an array value (nested ones flattened, the last index fastest), of a
tuple, of a cast's expression, of an array constructor over a constant range; `nothing` for
another expression.
"""
function arrayElements(e::IR.Expression)::Union{Nothing, Vector{IR.Expression}}
  if e isa IR.ARRAY
    local out = IR.Expression[]
    for x in e.elements
      local inner = arrayElements(x)
      inner === nothing ? push!(out, x) : append!(out, inner)
    end
    return out
  end
  e isa IR.TUPLE && return copy(e.elements)
  e isa IR.CAST && return arrayElements(e.exp)
  e isa IR.ARRAY_CONSTRUCTOR && return reductionElements(e.body, e.iterators)
  return nothing
end

"""
    elementOf(e, index)

The element `index` of the array or record expression `e`: of a reference the reference with
the subscripts; of a record value its field (one index); of another expression the
subscripted expression.
"""
function elementOf(e::IR.Expression, index::Vector{Int})::IR.Expression
  local subs = IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index]
  local ty = elementTypeOf(IR.typeOf(e), length(index))
  if e isa IR.CREF
    local last = e.cref.parts[end]
    local parts = IR.CrefPart[e.cref.parts[1:end-1]..., IR.CrefPart(last.name, vcat(last.subscripts, subs), last.ty)]
    return IR.CREF(IR.ComponentRef(parts), ty)
  end
  local recordTy = IR.typeOf(e)
  (recordTy isa IR.T_RECORD && length(index) == 1 && index[1] <= length(recordTy.fields)) && return fieldByIndex(e, index[1])
  return IR.SUBSCRIPTED(e, subs, ty)
end

"""
    fieldByIndex(e, k)

The `k`-th field of the record expression `e` as `e[k]`: the previous backend's form of the
fields of an expanded record equation, which the code generation reads (a call's result is
indexed by field).
"""
fieldByIndex(e::IR.Expression, k::Int)::IR.Expression =
  IR.SUBSCRIPTED(e, IR.Subscript[IR.INDEX(IR.INTEGER(k))], (IR.typeOf(e)::IR.T_RECORD).fields[k].ty)

"The type of an element `n` subscripts into the type `ty`: its dimensions after the first `n`."
function elementTypeOf(ty::IR.ModelicaType, n::Int)::IR.ModelicaType
  ty isa IR.T_ARRAY || return ty
  n < length(ty.dimensions) || return ty.elementType
  return IR.T_ARRAY(ty.elementType, ty.dimensions[n+1:end])
end

#= ==================================================================================== =#
#= Reductions                                                                           =#
#= ==================================================================================== =#

"""
    unrollConstantReductions!(model)

The reductions of one iterator over a constant Integer range (`a:b`, `a:s:b`, `{1, 3}`)
unrolled, so that no iterator survives into SimCode (its references have literal
subscripts): `sum`/`product` as a chain of `+`/`*`, `min`/`max` as nested two-argument calls,
an array constructor as the array value. A call of `sum`, `product`, `min` or `max` of one
array value folds the same way. In the equations and the initial equations of the
continuous-time partitions and the structural modes.
"""
function unrollConstantReductions!(model::IR.Model)::IR.Model
  for (_, p) in partitionsWithVariables(model)
    foreach(eq -> mapRoots!(unroll, eq), p.equations)
    foreach(eq -> mapRoots!(unroll, eq), p.initialEquations)
  end
  return model
end

"`e` with its constant reductions unrolled, top-down: an unrolled one's elements in turn."
unroll(e::IR.Expression)::IR.Expression = IR.mapChildren(unroll, unrollAt(e))

const FOLDED_REDUCERS = (:sum, :product, :min, :max)

function unrollAt(e::IR.Expression)::IR.Expression
  if e isa IR.CALL && e.attributes.builtin && e.name in FOLDED_REDUCERS && length(e.arguments) == 1
    local arg = e.arguments[1]
    local elements = arg isa IR.ARRAY_CONSTRUCTOR ? reductionElements(arg.body, arg.iterators) :
                     arg isa IR.ARRAY ? copy(arg.elements) : nothing
    elements === nothing && return e
    return something(foldReduction(e.name, elements, e.ty, e.attributes), e)
  elseif e isa IR.REDUCTION && e.reducer in FOLDED_REDUCERS
    local elements = reductionElements(e.body, e.iterators)
    elements === nothing && return e
    return something(foldReduction(e.reducer, elements, e.ty, IR.BUILTIN_CALL), e)
  elseif e isa IR.ARRAY_CONSTRUCTOR
    local elements = reductionElements(e.body, e.iterators)
    elements === nothing && return e
    return IR.ARRAY(elements, e.ty)
  end
  return e
end

"""
    reductionElements(body, iterators)

`body` for each value of a single iterator over a constant Integer range, the iterator
replaced by the value; `nothing` for another.
"""
function reductionElements(body::IR.Expression, iterators::Vector{IR.Iterator})::Union{Nothing, Vector{IR.Expression}}
  length(iterators) == 1 || return nothing
  local range = iterators[1].range
  local values = if range isa IR.RANGE
    rangeValues(range)
  elseif range isa IR.ARRAY && all(x -> x isa IR.INTEGER, range.elements)
    Int[x.value for x in range.elements]
  else
    nothing
  end
  values === nothing && return nothing
  local name = iterators[1].name
  return IR.Expression[substituteIterators(body, Dict(name => k)) for k in values]
end

"""
The reduction `reducer` of `elements` as a chain of `+` or `*`, or nested `min`/`max` calls (of
`attributes`); `nothing` for none.
"""
function foldReduction(reducer::Symbol, elements::Vector{IR.Expression}, ty::IR.ModelicaType,
                       attributes::IR.CallAttributes)::Union{Nothing, IR.Expression}
  isempty(elements) && return nothing
  local acc = elements[1]
  for x in elements[2:end]
    acc = reducer === :sum ? IR.BINARY(acc, IR.ADD, x, ty) :
          reducer === :product ? IR.BINARY(acc, IR.MUL, x, ty) :
          IR.CALL(reducer, IR.Expression[acc, x], ty, attributes)
  end
  return acc
end
