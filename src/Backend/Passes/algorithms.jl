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

#= The algorithm sections as equations (the previous backend's algorithm synthesis,
   BDAECreate/algorithmSynthesis.jl): the code generation takes when-equations, initial
   when-equations, residual equations and asserts, not algorithm sections. =#

"""
    liftAlgorithms!(model)

The algorithm and initial algorithm sections of the continuous-time partitions (and of the
structural modes) as equations, after the partition's other equations, in this order:

1. the bodies of `when initial()` statements of the algorithm sections, as when-equations
   active at the initialization (their statements run once, as written);
2. the initial algorithm sections, the same;
3. for the regular algorithm sections: their when statements and their discrete-time
   assignments as when-equations (`liftAlgorithmWhens`), their single discrete-time
   assignments as residuals, their Real assignments run symbolically as equations
   (`residualsOfAlgorithms`), their top-level asserts as assert equations.
"""
function liftAlgorithms!(model::IR.Model)::IR.Model
  for p in model.continuous
    liftAlgorithms!(model.variables, p)
  end
  for mode in model.modes
    liftAlgorithms!(mode.variables, mode.system)
  end
  return model
end

function liftAlgorithms!(variables::IR.VariableSet, partition::IR.Partition)::Nothing
  local algorithms = IR.ALGORITHM[eq for eq in partition.equations if eq isa IR.ALGORITHM]
  local initialAlgorithms = IR.ALGORITHM[eq for eq in partition.initialEquations if eq isa IR.ALGORITHM]
  (isempty(algorithms) && isempty(initialAlgorithms)) && return nothing
  local equations = IR.Equation[eq for eq in partition.equations if !(eq isa IR.ALGORITHM)]
  partition.initialEquations = IR.Equation[eq for eq in partition.initialEquations if !(eq isa IR.ALGORITHM)]
  append!(equations, initialWhensOfAlgorithms(algorithms))
  append!(equations, initialAlgorithmWhens(initialAlgorithms))
  if !isempty(algorithms)
    local constrained = constrainedNames(equations)
    local known = knownNames(variables)
    local (whens, whenLifted) = liftAlgorithmWhens(algorithms, known)
    append!(equations, whens)
    append!(equations, residualsOfAlgorithms(algorithms, constrained, whenLifted, realStartValues(variables)))
    append!(equations, assertsOfAlgorithms(algorithms))
  end
  partition.equations = equations
  return nothing
end

"""
A when-equation active at the initialization only, with the statements `body`: of `kind`
INITIAL_SECTION for an initial algorithm section's or an algorithm's `when initial()`.
"""
initialWhen(body::Vector{IR.Statement}, source::IR.Source; kind::IR.WhenKind = IR.INITIAL_WHEN)::IR.WHEN_EQUATION =
  IR.WHEN_EQUATION([IR.WhenBranch(IR.Expression[], true, false, body)], kind, source, IR.DEFAULT_ATTRIBUTES)

"""
The bodies of the top-level `when initial()` statements of algorithm sections (a first
branch whose condition is `initial()` alone), as initial when-equations.
"""
function initialWhensOfAlgorithms(algorithms::Vector{IR.ALGORITHM})::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms, s in alg.statements
    s isa IR.WHEN_STATEMENT || continue
    local b = s.branches[1]
    (b.atInitial && isempty(b.triggers) && !b.atTerminal) || continue
    push!(out, initialWhen(b.body, s.source; kind = IR.INITIAL_SECTION))
  end
  return out
end

"The initial algorithm sections (those with statements) as initial when-equations."
initialAlgorithmWhens(algorithms::Vector{IR.ALGORITHM})::Vector{IR.Equation} =
  IR.Equation[initialWhen(alg.statements, alg.source; kind = IR.INITIAL_SECTION) for alg in algorithms if !isempty(alg.statements)]

"""
    constrainedNames(equations)

The names of the variables the equations constrain, for the residual lifter: it lifts an
algorithm's assignment only of a variable no equation defines. An alias `a = b` between two
variables (a connection) defines neither (the MSL Digital Set source `y := x`, connected to a
flip-flop's inputs, left them at 0, an invalid logic value).
"""
function constrainedNames(equations::Vector{IR.Equation})::Set{Symbol}
  local names = Set{Symbol}()
  local add = e -> foreach(c -> push!(names, c.name), IR.crefsOf(e))
  for eq in equations
    if eq isa IR.SCALAR_EQUATION
      (eq.lhs isa IR.CREF && eq.rhs isa IR.CREF) && continue
      add(eq.lhs); add(eq.rhs)
    elseif eq isa Union{IR.ARRAY_EQUATION, IR.RECORD_EQUATION}
      add(eq.lhs); add(eq.rhs)
    end
  end
  return names
end

"The start values of the Real variables that have one: where an algorithm's Real targets start."
function realStartValues(variables::IR.VariableSet)::Dict{Symbol, IR.Expression}
  local out = Dict{Symbol, IR.Expression}()
  for v in variables
    (isRealType(v.ty) && v.attributes.start !== nothing) && (out[v.name] = v.attributes.start)
  end
  return out
end

#= ==================================================================================== =#
#= The statements of a body as assignments                                              =#
#= ==================================================================================== =#

"""
    BodyLowering(initialValue, allowContinuous, where)

How a body's statements become assignments: `initialValue` the value of `initial()` (true
for the initial arm); `allowContinuous` lowers the assignments of Real targets and the
asserts and terminate() too (a when body), else only the discrete-time assignments (a regular
algorithm); `where` names the body in the messages.
"""
struct BodyLowering
  initialValue::Bool
  allowContinuous::Bool
  where::String
end

"""
    bodyStatements(stmts, lowering)

The statements as straight-line statements (the previous backend's when operators): each
assignment to the scalar variables it assigns, under the conditions of the if-statements it is
in (`v := if c then e else v`), the for loops over constant ranges unrolled, an assert under
its conditions. With the names of the variables assigned. `nothing` if the body cannot be so
lowered (a for loop over a non-constant range, an assignment to a slice or to an expression);
for a when body (`allowContinuous`) that is not supported.
"""
function bodyStatements(stmts::Vector{IR.Statement}, lowering::BodyLowering)::Tuple{Vector{IR.Statement}, Vector{Symbol}}
  local ops = IR.Statement[]
  local assigned = Symbol[]
  local ok = appendBody!(ops, assigned, stmts, nothing, Dict{Symbol, Int}(), lowering)
  ok || lowering.allowContinuous &&
    unsupported("$(lowering.where) with a for loop over a non-constant range, or an assignment to a slice or an expression", stmts)
  ok || return (IR.Statement[], Symbol[])
  return (ops, assigned)
end

"`e` with the loop iterators' values and `initial()`'s value."
prepare(e::IR.Expression, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::IR.Expression =
  replaceInitial(substituteIterators(e, iterValues), lowering.initialValue)

function appendBody!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, stmts::Vector{IR.Statement},
                     active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  for s in stmts
    appendStatement!(ops, assigned, s, active, iterValues, lowering) || return false
  end
  return true
end

function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.ASSIGNMENT,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  if s.lhs isa IR.TUPLE
    #= (a, b, ...) := f(...) (MSL TimeTable's when: (a, b, nextEventScaled, last) :=
       getInterpolationCoefficients(...)): one assignment per target; f runs once per
       target, an impure f is not supported. =#
    (s.rhs isa IR.CALL && s.rhs.attributes.impure) && unsupported("a tuple assignment from an impure function", s.rhs)
    for (target, element) in tupleAssignElements(s.lhs.elements, s.rhs)
      emitAssignment!(ops, assigned, target, element, s.source, active, iterValues, lowering) || return false
    end
    return true
  end
  return emitAssignment!(ops, assigned, s.lhs, s.rhs, s.source, active, iterValues, lowering)
end

function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.IF_STATEMENT,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  local guard = active
  for b in s.branches
    local c = prepare(b.condition, iterValues, lowering)
    appendBody!(ops, assigned, b.body, andCondition(guard, c), iterValues, lowering) || return false
    guard = andCondition(guard, notCondition(c))
  end
  return appendBody!(ops, assigned, s.elseBody, guard, iterValues, lowering)
end

function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.FOR_STATEMENT,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  local values = rangeValues(prepare(s.iterator.range, iterValues, lowering))
  values === nothing && return false
  for v in values
    local nested = copy(iterValues)
    nested[s.iterator.name] = v
    appendBody!(ops, assigned, s.body, active, nested, lowering) || return false
  end
  return true
end

#= Nested whens are not Modelica; the top-level ones are lifted on their own. =#
appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.WHEN_STATEMENT,
                 active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool = true

"""
An assert in a when body (`allowContinuous`: the when lifter) is checked where the body runs,
under its conditions: `not guard or c`. In a regular algorithm the top-level asserts are
`assertsOfAlgorithms`'.
"""
function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.ASSERT,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  lowering.allowContinuous || return true
  local c = prepare(s.condition, iterValues, lowering)
  active === nothing || (c = orCondition(notCondition(active), c))
  push!(ops, IR.ASSERT(c, prepare(s.message, iterValues, lowering), s.level, s.source))
  return true
end

"`terminate()` under a constant guard only: always or never (`if initial()` in the runtime arm)."
function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.TERMINATE,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  (lowering.allowContinuous && !isFalse(active)) || return true
  (active === nothing || active isa IR.BOOLEAN) || unsupported("terminate() under a condition in $(lowering.where)", s)
  push!(ops, IR.TERMINATE(prepare(s.message, iterValues, lowering), s.source))
  return true
end

#= A call for its side effects (a print, a file): no variable depends on it. Not run. =#
appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.NORETCALL,
                 active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool = true

function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.Statement,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  lowering.allowContinuous && unsupported("this statement in $(lowering.where)", s)
  return true
end

"""
    emitAssignment!(ops, assigned, lhs, rhs, source, active, iterValues, lowering)

`lhs := rhs` as assignments of the scalar variables `lhs` names (all of an array's elements,
one under a guard `i == k` for a subscript `i` that is not a literal), each the value `rhs`
gives it, under the active condition: `v := if active then e else v`. A Real target is
left out unless `allowContinuous`. `false` if `lhs` is not a reference or names a slice.
"""
function emitAssignment!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, lhs::IR.Expression, rhs::IR.Expression,
                         source::IR.Source, active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int},
                         lowering::BodyLowering)::Bool
  (lowering.allowContinuous || isDiscreteType(IR.typeOf(lhs))) || return true
  lhs = prepare(lhs, iterValues, lowering)
  rhs = prepare(rhs, iterValues, lowering)
  lhs isa IR.CREF || return false
  local targets = scalarTargets(lhs)
  isempty(targets) && return false
  for (target, guard, rhsIndices) in targets
    local c = andCondition(active, guard)
    isFalse(c) && continue
    local value = scalarizedRead(rhs, rhsIndices, target)
    push!(ops, IR.ASSIGNMENT(target, c === nothing ? value : IR.IF_EXPRESSION(c, value, target, target.ty), source))
    push!(assigned, target.cref.name)
  end
  return true
end

"""
The subscript of the `k`-th index of a dimension: an enumeration's literal (the scalarized
variables' names carry them), else the integer.
"""
indexOf(d::IR.Dimension, k::Int)::IR.Expression = d isa IR.DIM_ENUM ? IR.dimensionIndex(d, k) : IR.INTEGER(k)

isLiteralIndex(s::IR.Subscript)::Bool = s isa IR.INDEX && (s.exp isa IR.INTEGER || s.exp isa IR.ENUM_LITERAL)

"`cref` with the subscripts of its last part replaced by `subs`."
withLastSubscripts(cref::IR.ComponentRef, subs::Vector{IR.Subscript})::IR.ComponentRef =
  IR.ComponentRef(IR.CrefPart[cref.parts[1:end - 1]..., IR.CrefPart(cref.parts[end].name, subs, cref.parts[end].ty)])

"""
The subscripts of the last part of `cref` padded with whole dimensions to the dimensions of
its component's type, and those dimensions; `nothing` for a scalar.
"""
function paddedSubscripts(cref::IR.ComponentRef)::Union{Nothing, Tuple{Vector{IR.Subscript}, Vector{IR.Dimension}}}
  local dims = IR.dimensions(cref.parts[end].ty)
  isempty(dims) && return nothing
  local subs = copy(IR.subscripts(cref))
  length(subs) < length(dims) && append!(subs, IR.Subscript[IR.WHOLE() for _ in 1:(length(dims) - length(subs))])
  return (subs, dims)
end

"""
    scalarTargets(lhs)

The scalar variables an assignment to `lhs` assigns: `(target, guard, rhsIndices)` per
element, its guard (`nothing`, or `i == k` for a subscript `i` that is not a literal) and the
indices of the whole dimensions, which select the right-hand side's element.
"""
function scalarTargets(lhs::IR.CREF)::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}}
  local out = Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}[]
  local padded = paddedSubscripts(lhs.cref)
  padded === nothing && return Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}[(lhs, nothing, Int[])]
  local (subs, dims) = padded
  length(subs) == length(dims) || return out
  local elementTy = IR.elementType(lhs.cref.parts[end].ty)
  local visit
  visit = (pos, newSubs, guard, rhsIndices) -> begin
    if pos > length(dims)
      push!(out, (IR.CREF(withLastSubscripts(lhs.cref, newSubs), elementTy), guard, copy(rhsIndices)))
      return true
    end
    local n = IR.dimensionSize(dims[pos])
    local sub = subs[pos]
    if sub isa IR.WHOLE
      n === nothing && return false
      for k in 1:n
        visit(pos + 1, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], k))], guard, Int[rhsIndices..., k]) || return false
      end
    elseif isLiteralIndex(sub)
      visit(pos + 1, IR.Subscript[newSubs..., sub], guard, rhsIndices) || return false
    elseif sub isa IR.INDEX
      n === nothing && return false
      for k in 1:n
        visit(pos + 1, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], k))],
              andCondition(guard, indexEqualsCondition(sub.exp, k)), rhsIndices) || return false
      end
    else
      return false
    end
    return true
  end
  visit(1, IR.Subscript[], nothing, Int[]) || return Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}[]
  return out
end

"""
    scalarizedRead(rhs, rhsIndices, fallback)

The element `rhsIndices` of `rhs`: of a reference, the scalar variable (for a subscript that
is not a literal, the chain `if i == 1 then x[1] elseif ...`, `fallback` last); of another
array expression, the expression subscripted.
"""
function scalarizedRead(rhs::IR.Expression, rhsIndices::Vector{Int}, fallback::IR.Expression)::IR.Expression
  rhs isa IR.CREF && return scalarizedCref(rhs, rhsIndices, fallback)
  isempty(rhsIndices) && return rhs
  local ty = IR.typeOf(rhs)
  local dims = IR.dimensions(ty)
  local elementTy = length(dims) <= length(rhsIndices) ? IR.elementType(ty) :
                    IR.T_ARRAY(IR.elementType(ty), dims[(length(rhsIndices) + 1):end])
  return IR.SUBSCRIPTED(rhs, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in rhsIndices], elementTy)
end

function scalarizedCref(e::IR.CREF, rhsIndices::Vector{Int}, fallback::IR.Expression)::IR.Expression
  local padded = paddedSubscripts(e.cref)
  padded === nothing && return e
  local (subs, dims) = padded
  length(subs) == length(dims) || return fallback
  local elementTy = IR.elementType(e.cref.parts[end].ty)
  local candidates = Tuple{Union{Nothing, IR.Expression}, IR.Expression}[]
  local visit
  visit = (pos, rhsPos, newSubs, guard) -> begin
    if pos > length(dims)
      push!(candidates, (guard, IR.CREF(withLastSubscripts(e.cref, newSubs), elementTy)))
      return
    end
    local sub = subs[pos]
    if sub isa IR.WHOLE
      rhsPos <= length(rhsIndices) || return
      visit(pos + 1, rhsPos + 1, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], rhsIndices[rhsPos]))], guard)
    elseif isLiteralIndex(sub)
      visit(pos + 1, rhsPos, IR.Subscript[newSubs..., sub], guard)
    elseif sub isa IR.INDEX
      local n = IR.dimensionSize(dims[pos])
      n === nothing && return
      for k in 1:n
        visit(pos + 1, rhsPos, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], k))],
              andCondition(guard, indexEqualsCondition(sub.exp, k)))
      end
    end
  end
  visit(1, 1, IR.Subscript[], nothing)
  isempty(candidates) && return fallback
  local result = fallback
  for (guard, value) in Iterators.reverse(candidates)
    result = guard === nothing ? value : IR.IF_EXPRESSION(guard, value, result, elementTy)
  end
  return result
end

"""
    tupleAssignElements(targets, rhs)

`(a, b, ...) := f(...)` as one assignment per target of its output of `f`, in the order they
run. Each evaluates `f` again after the ones before it assigned their targets, so a target `f`
reads (outside pre()) goes last and `f` sees its old value in every element: `(s, y) :=
step(s, u)` gave `y` the step from the new `s`. A second such target, or one assigned element by
element (an array, a record), cannot be last: not supported. A record output is one
assignment per field (a generated `f` returns its fields in place).
"""
function tupleAssignElements(targets::Vector{IR.Expression}, rhs::IR.Expression)::Vector{Tuple{IR.Expression, IR.Expression}}
  local read = namesReadOutsidePre(rhs)
  local outputTypes = rhs isa IR.CALL && rhs.ty isa IR.T_TUPLE ? rhs.ty.types : nothing
  local elements = Tuple{IR.Expression, IR.Expression}[]
  local pos = 0
  for (k, target) in enumerate(targets)
    (target isa IR.CREF || target isa IR.WILD) || unsupported("a tuple assignment to a target that is not a variable", target)
    local outTy = outputTypes === nothing ? IR.typeOf(target) : outputTypes[k]
    if outTy isa IR.T_RECORD
      for field in outTy.fields
        pos += 1
        field.ty isa IR.T_RECORD && unsupported("a tuple assignment of a record output with record fields", rhs)
        target isa IR.WILD ||
          push!(elements, (fieldExpression(target, field.name, field.ty), IR.TUPLE_ELEMENT(rhs, pos, field.ty)))
      end
    else
      pos += 1
      target isa IR.WILD || push!(elements, (target, IR.TUPLE_ELEMENT(rhs, pos, IR.typeOf(target))))
    end
  end
  local plain = Tuple{IR.Expression, IR.Expression}[]
  local readBack = Tuple{IR.Expression, IR.Expression}[]
  for element in elements
    local name = first(element).cref.name
    if any(r -> namesOverlap(r, name), read)
      local ty = IR.typeOf(first(element))
      (ty isa IR.T_ARRAY || ty isa IR.T_RECORD) &&
        unsupported("a tuple assignment whose function reads the array or record target $(name)", rhs)
      push!(readBack, element)
    else
      push!(plain, element)
    end
  end
  length(readBack) > 1 && unsupported("a tuple assignment whose function reads two of its targets", rhs)
  return vcat(plain, readBack)
end

#= ==================================================================================== =#
#= When-equations from regular algorithm sections                                       =#
#= ==================================================================================== =#

"""
    liftAlgorithmWhens(algorithms, known)

The when-equations of regular algorithm sections, and the names of the variables they assign:

- an algorithm of when statements only: each when statement (a first branch `when initial()`
  is `initialWhensOfAlgorithms`', its elsewhen arms are here);
- else: each top-level `if c then v := e; end if` of a discrete `v` the section assigns nowhere
  else as `when c then v := e` (`liftIfToWhen!`); the whole body, unless it is one
  discrete assignment, as an initial when-equation (it runs once at the initialization, in
  order) and a when-equation on `change()` of its discrete inputs and of its relations with a
  continuous operand (MLS 17.4.4: a non-when algorithm with discrete targets runs at every
  event that changes its inputs); its when statements, as above.
"""
function liftAlgorithmWhens(algorithms::Vector{IR.ALGORITHM}, known::Set{Symbol})::Tuple{Vector{IR.Equation}, Set{Symbol}}
  local out = IR.Equation[]
  local lifted = Set{Symbol}()
  for alg in algorithms
    isempty(alg.statements) && continue
    if all(s -> s isa IR.WHEN_STATEMENT, alg.statements)
      foreach(s -> liftAlgorithmWhenStatement!(out, s, lifted), alg.statements)
      continue
    end
    local counts = assignmentCounts(alg.statements)
    #= the conditions of the ifs lifted on their own: the body's lifting has no when of their sample() =#
    local ownWhens = IR.Expression[s.branches[1].condition for s in alg.statements if liftIfToWhen!(out, s, alg.source, counts)]
    isSingleDiscreteAssignment(alg.statements) || liftBodyToWhens!(out, alg.statements, alg.source, lifted, known, ownWhens)
    for s in alg.statements
      s isa IR.WHEN_STATEMENT && liftAlgorithmWhenStatement!(out, s, lifted)
    end
  end
  return (out, lifted)
end

"Whether `stmts` is one assignment of a discrete-time variable."
isSingleDiscreteAssignment(stmts::Vector{IR.Statement})::Bool =
  length(stmts) == 1 && stmts[1] isa IR.ASSIGNMENT && !(stmts[1].lhs isa IR.TUPLE) && isDiscreteType(IR.typeOf(stmts[1].lhs))

"""
A top-level when statement of an algorithm. A first branch `when initial()` alone is
`initialWhensOfAlgorithms`' (it runs the statements): here only its elsewhen arms. Any other
when entirely, an initial arm (`when {c, initial()}`) as an initial when-equation too.
"""
function liftAlgorithmWhenStatement!(out::Vector{IR.Equation}, s::IR.WHEN_STATEMENT, lifted::Set{Symbol})::Nothing
  local first = s.branches[1]
  if first.atInitial && isempty(first.triggers) && !first.atTerminal
    length(s.branches) > 1 || return nothing
    local weq = whenEquationOf(s.branches[2:end], s.source, lifted)
    weq === nothing || push!(out, weq)
    return nothing
  end
  if first.atInitial
    #= An initial arm the lifter cannot lower stays empty (its variables start from their
       start values): a fallback that changes the result, reported. =#
    local (initOps, initAssigned) = try
      bodyStatements(first.body, BodyLowering(true, true, "a when body"))
    catch err
      _fallback(err, :algorithmInitialArm; only = UnsupportedLowering, impact = :result)
      (IR.Statement[], Symbol[])
    end
    union!(lifted, initAssigned)
    isempty(initOps) || push!(out, initialWhen(initOps, s.source))
  end
  local weq = whenEquationOf(s.branches, s.source, lifted)
  weq === nothing || push!(out, weq)
  return nothing
end

"""
The when-equation of a when statement's branches: the runtime arm (`initial()` false), every
assignment lifted (continuous and discrete: in a when all targets are event-updated), the
if-statements as guarded assignments. A branch that never runs (`when initial()` alone)
contributes nothing. `nothing` if no branch has statements.
"""
function whenEquationOf(branches::Vector{IR.WhenBranch}, source::IR.Source, lifted::Set{Symbol})::Union{Nothing, IR.WHEN_EQUATION}
  local out = IR.WhenBranch[]
  for b in branches
    local triggers = IR.Expression[replaceInitial(t, false) for t in b.triggers]
    local runs = !isempty(triggers) || b.atTerminal
    local (ops, assigned) = runs ? bodyStatements(b.body, BodyLowering(false, true, "a when body")) : (IR.Statement[], Symbol[])
    union!(lifted, assigned)
    push!(out, IR.WhenBranch(triggers, false, b.atTerminal, ops))
  end
  #= Trailing branches without statements are left out: an elsewhen without effect. =#
  while !isempty(out) && isempty(out[end].body)
    pop!(out)
  end
  isempty(out) && return nothing
  return IR.WHEN_EQUATION(out, IR.REGULAR_WHEN, source, IR.DEFAULT_ATTRIBUTES)
end

"""
A top-level `if c then v := e; end if` of a discrete `v` (one branch of one assignment, no
else) that the section assigns nowhere else (`counts`) as `when c then v := e`: its event is
`c`'s (a `sample()` the body's lifting has no trigger for). An if with an elseif or an else, or a
`v` another statement assigns (`v := v * 10` after it), is the body's lifting's alone, which runs
the section in order: a when of the first branch beside it assigned `v` twice at one event.
"""
function liftIfToWhen!(out::Vector{IR.Equation}, s::IR.Statement, source::IR.Source, counts::Dict{Symbol, Int})::Bool
  (s isa IR.IF_STATEMENT && length(s.branches) == 1 && isempty(s.elseBody) && length(s.branches[1].body) == 1) ||
    return false
  local a = s.branches[1].body[1]
  (a isa IR.ASSIGNMENT && a.lhs isa IR.CREF && isDiscreteType(IR.typeOf(a.lhs)) && !(IR.typeOf(a.lhs) isa IR.T_ARRAY)) ||
    return false
  get(counts, a.lhs.cref.name, 0) == 1 || return false
  push!(out, IR.WHEN_EQUATION([IR.WhenBranch(IR.Expression[s.branches[1].condition], false, false, IR.Statement[a])],
                              IR.REGULAR_WHEN, source, IR.DEFAULT_ATTRIBUTES))
  return true
end

"""
    assignmentCounts(stmts)

How often the statements assign each variable (the targets of assignments and tuple
assignments, in if-, for-, while- and when-statements too).
"""
function assignmentCounts(stmts::Vector{IR.Statement})::Dict{Symbol, Int}
  local counts = Dict{Symbol, Int}()
  local visit
  visit = s -> begin
    if s isa IR.ASSIGNMENT
      for t in (s.lhs isa IR.TUPLE ? s.lhs.elements : IR.Expression[s.lhs])
        t isa IR.CREF && (counts[t.cref.name] = get(counts, t.cref.name, 0) + 1)
      end
    elseif s isa Union{IR.IF_STATEMENT, IR.WHEN_STATEMENT}
      foreach(b -> foreach(visit, b.body), s.branches)
      s isa IR.IF_STATEMENT && foreach(visit, s.elseBody)
    elseif s isa Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT}
      foreach(visit, s.body)
    end
  end
  foreach(visit, stmts)
  return counts
end

"""
A regular algorithm's body as an initial when-equation (the body run once at the
initialization, `initial()` true) and a when-equation on `change()` of its discrete inputs and
of its relations with a continuous operand (`initial()` false): the source-style bodies
`y := y0; y := if time >= t[i] then x[i] else y` have their events in those relations, not in
the constant seed. A `sample()` the body reads (`if sample(t0, T) then n := 1; end if; n := n * 10`)
is an event of its own: a when-equation of the body on it, unless an if on it is lifted on its
own (`ownWhens`, the conditions of `liftIfToWhen!`'s whens: the body there twice at a tick).
"""
function liftBodyToWhens!(out::Vector{IR.Equation}, stmts::Vector{IR.Statement}, source::IR.Source,
                          lifted::Set{Symbol}, known::Set{Symbol}, ownWhens::Vector{IR.Expression})::Nothing
  local (initOps, initAssigned) = bodyStatements(stmts, BodyLowering(true, false, "an algorithm section"))
  local (runOps, runAssigned) = bodyStatements(stmts, BodyLowering(false, false, "an algorithm section"))
  union!(lifted, initAssigned)
  union!(lifted, runAssigned)
  (isempty(initOps) && isempty(runOps)) && return nothing
  isempty(initOps) || push!(out, initialWhen(initOps, source))
  local triggers = IR.Expression[changeCall(c) for c in discreteInputs(runOps, Set{Symbol}(runAssigned))]
  local relationsSeen = IR.Expression[]
  for op in runOps
    op isa IR.ASSIGNMENT || continue
    for r in relationsIn(op.rhs)
      (hasContinuousOperand(r, known) && !(r in relationsSeen)) || continue
      push!(relationsSeen, r)
    end
  end
  append!(triggers, IR.Expression[changeCall(r) for r in relationsSeen])
  if !isempty(runOps) && !isempty(triggers)
    local condition = foldl((a, b) -> IR.LBINARY(a, IR.OR, b), triggers)
    push!(out, IR.WHEN_EQUATION([IR.WhenBranch(IR.Expression[condition], false, false, runOps)],
                                IR.REGULAR_WHEN, source, IR.DEFAULT_ATTRIBUTES))
  end
  isempty(runOps) && return nothing
  local owned = Set{IR.Expression}(c for w in ownWhens for c in samplesIn(w))
  for s in unique(c for op in runOps if op isa IR.ASSIGNMENT for c in samplesIn(op.rhs))
    s in owned && continue
    push!(out, IR.WHEN_EQUATION([IR.WhenBranch(IR.Expression[s], false, false, runOps)], IR.REGULAR_WHEN, source,
                                IR.DEFAULT_ATTRIBUTES))
  end
  return nothing
end

"The `sample()` calls in `e`."
function samplesIn(e::IR.Expression)::Vector{IR.Expression}
  local out = IR.Expression[]
  IR.foreachExp(x -> IR.isBuiltinCall(x, :sample) && push!(out, x), e)
  return out
end

"""
The discrete-time variables the statements read (not `time`, not the variables they assign,
not the iterators of reductions), each once, in the order read: references whose type is
discrete.
"""
function discreteInputs(ops::Vector{IR.Statement}, assigned::Set{Symbol})::Vector{IR.CREF}
  local out = IR.CREF[]
  local seen = Set{Symbol}()
  local blocked = copy(assigned)
  local visit = e -> begin
    if e isa IR.CREF
      local name = e.cref.name
      (name === :time || name in blocked || name in seen || !isDiscreteType(e.ty)) && return
      push!(seen, name)
      push!(out, e)
    elseif e isa IR.REDUCTION
      foreach(it -> push!(blocked, it.name), e.iterators)
    end
  end
  for op in ops
    if op isa IR.ASSIGNMENT
      IR.foreachExp(visit, op.rhs)
    elseif op isa IR.NORETCALL
      IR.foreachExp(visit, op.call)
    elseif op isa IR.ASSERT
      IR.foreachExp(visit, op.condition); IR.foreachExp(visit, op.message); IR.foreachExp(visit, op.level)
    end
  end
  return out
end

#= ==================================================================================== =#
#= Residuals and asserts from regular algorithm sections                                =#
#= ==================================================================================== =#

"""
    residualsOfAlgorithms(algorithms, constrained, whenLifted, realStarts)

Equations of regular algorithm sections (not of when statements alone): their Real
assignments run symbolically (`realAssignmentEquations`), and a body of exactly one
assignment (others have order-sensitive semantics a flat list of residuals does not keep: MSL
OneWayClutch, most Digital gates) as the residual `v = e` of its discrete target, unless an
equation constrains `v` or the when lifter assigns it.
"""
function residualsOfAlgorithms(algorithms::Vector{IR.ALGORITHM}, constrained::Set{Symbol}, whenLifted::Set{Symbol},
                               realStarts::Dict{Symbol, IR.Expression})::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms
    any(s -> !(s isa IR.WHEN_STATEMENT), alg.statements) || continue
    append!(out, realAssignmentEquations(alg.statements, alg.source, realStarts, whenLifted))
    count(s -> s isa IR.ASSIGNMENT && !(s.lhs isa IR.TUPLE), alg.statements) == 1 || continue
    for s in alg.statements
      (s isa IR.ASSIGNMENT && !(s.lhs isa IR.TUPLE)) || continue
      residualsOfAssignment!(out, s, constrained, whenLifted)
    end
  end
  return out
end

"A dynamic equation: already classified, not one for the discrete synthesis."
const DYNAMIC = IR.EquationAttributes(IR.EQ_CONTINUOUS, false, nothing)

"""
The residuals of `v := e` for a discrete `v`: `v = e`; of an array `v`, one per element
(`v[k] = e[k]`), each unless an equation constrains it or the when lifter assigns it.
"""
function residualsOfAssignment!(out::Vector{IR.Equation}, s::IR.ASSIGNMENT, constrained::Set{Symbol}, whenLifted::Set{Symbol})::Nothing
  local ty = IR.typeOf(s.lhs)
  isDiscreteType(ty) || return nothing
  local lhs = s.lhs
  if !(ty isa IR.T_ARRAY)
    lhs isa IR.CREF && (lhs.cref.name in constrained || lhs.cref.name in whenLifted) && return nothing
    push!(out, IR.SCALAR_EQUATION(lhs, s.rhs, ty, s.source, DYNAMIC))
    return nothing
  end
  lhs isa IR.CREF && lhs.cref.name in whenLifted && return nothing
  local n = IR.dimensionSize(ty.dimensions[1])
  if !(lhs isa IR.CREF) || n === nothing || n <= 0
    #= Unknown shape: not if an equation constrains the variable or one of its elements. =#
    if lhs isa IR.CREF
      local name = String(lhs.cref.name)
      lhs.cref.name in constrained && return nothing
      any(c -> startswith(String(c), name * "["), constrained) && return nothing
    end
    push!(out, IR.SCALAR_EQUATION(lhs, s.rhs, ty, s.source, DYNAMIC))
    return nothing
  end
  local elementTy = IR.elementType(ty)
  for k in 1:n
    local cref = withLastSubscripts(lhs.cref, IR.Subscript[IR.subscripts(lhs.cref)..., IR.INDEX(IR.INTEGER(k))])
    (cref.name in constrained || cref.name in whenLifted) && continue
    local rhsK = IR.SUBSCRIPTED(s.rhs, IR.Subscript[IR.INDEX(IR.INTEGER(k))], elementTy)
    push!(out, IR.SCALAR_EQUATION(IR.CREF(cref, elementTy), rhsK, elementTy, s.source, DYNAMIC))
  end
  return nothing
end

"""
    AlgorithmRun(assigned, realStarts)

An algorithm section run symbolically: the values of the variables assigned so far; every
scalar name it assigns and the names of their wholes (`x` of `x[2]`, `r` of `r.a`); the
variables read in der(), edge() or change(), whose argument is the variable itself.
"""
struct AlgorithmRun
  values::Dict{Symbol, IR.Expression}
  assigned::Set{Symbol}
  wholes::Set{Symbol}
  realStarts::Dict{Symbol, IR.Expression}
  readAsVariable::Set{Symbol}
end

function AlgorithmRun(assigned::Vector{Symbol}, realStarts::Dict{Symbol, IR.Expression})
  local wholes = Set{Symbol}()
  for name in assigned
    local s = String(name)
    for i in eachindex(s)
      s[i] in ('[', '.') && push!(wholes, Symbol(s[1:prevind(s, i)]))
    end
  end
  return AlgorithmRun(Dict{Symbol, IR.Expression}(), Set{Symbol}(assigned), wholes, realStarts, Set{Symbol}())
end

"Above this many nodes a value is not built: `if v[i] > m then m := v[i]` doubles it each time."
const ALGORITHM_VALUE_NODE_LIMIT = 10_000

"""
    algorithmValue(e, run)

`e` with the section's variables read as their values so far: assigned before, the value;
assigned only later, the start value (Real) or `pre()` (discrete). Inside `pre()` a variable is
the value before the event; inside `der()`, `edge()` and `change()` the variable itself, which
its equation defines (its value at the end: not supported if it is assigned again later).
"""
function algorithmValue(e::IR.Expression, run::AlgorithmRun)::IR.Expression
  local visit = x -> begin
    if isBuiltinCallOf(x, (:pre, :der, :edge, :change))
      x.name === :pre && return (x, false)
      for c in IR.crefsOf(x)
        c.name in run.assigned || continue
        haskey(run.values, c.name) || unsupported("$(x.name)() of a variable before the algorithm section assigns it", x)
        push!(run.readAsVariable, c.name)
      end
      return (x, false)
    end
    x isa IR.CREF || return (x, true)
    local cref = foldSubscripts(x.cref)
    local name = cref.name
    haskey(run.values, name) && return (run.values[name], false)
    if name in run.assigned
      isRealType(x.ty) && return (get(run.realStarts, name, IR.REAL(0.0)), false)
      return (preCall(IR.CREF(cref, x.ty)), false)
    end
    name in run.wholes && unsupported("a read of a whole array or record that an algorithm section assigns part by part", x)
    #= x[j]: one of the elements the section assigns, chosen by the index =#
    local base = Symbol(first(split(String(name), '[')))
    if base !== name && base in run.wholes && any(s -> !isLiteralIndex(s), IR.subscripts(cref))
      local elements = scalarizedCref(IR.CREF(cref, x.ty), Int[], x)
      elements isa IR.IF_EXPRESSION || unsupported("this read of an array an algorithm section assigns", x)
      return (elementValues(elements, x, run), false)
    end
    return (x, true)
  end
  local value = IR.mapExpTopDown(visit, e)
  expressionLargerThan(value, ALGORITHM_VALUE_NODE_LIMIT) &&
    unsupported("an algorithm section whose values exceed $(ALGORITHM_VALUE_NODE_LIMIT) nodes as equations", e)
  return value
end

"The if-chain over the elements a read `x[j]` may be, each element (and guard) its value."
function elementValues(chain::IR.Expression, read::IR.Expression, run::AlgorithmRun)::IR.Expression
  chain === read && return read
  return IR.IF_EXPRESSION(algorithmValue(chain.condition, run), algorithmValue(chain.thenExp, run),
                          elementValues(chain.elseExp, read, run), chain.ty)
end

"Whether `e` has more than `cap` nodes, counted as a tree."
function expressionLargerThan(e::IR.Expression, cap::Int)::Bool
  local n = Ref(0)
  local visit
  visit = x -> (n[] += 1; n[] > cap && return; foreach(visit, IR.children(x)))
  visit(e)
  return n[] > cap
end

"""
    realAssignmentEquations(stmts, source, realStarts, whenLifted)

The Real variables an algorithm section assigns outside its when statements, as equations
(MLS 11.1.2: the section runs as a whole, in order): the statements run symbolically
(`AlgorithmRun`), each Real variable `v = <its value at the end>`, a record variable the record
equation; an assert checked on the values at its point. The discrete targets stay the when
lifter's. A section reading `initial()` is reported, not lowered.
"""
function realAssignmentEquations(stmts::Vector{IR.Statement}, source::IR.Source,
                                 realStarts::Dict{Symbol, IR.Expression}, whenLifted::Set{Symbol})::Vector{IR.Equation}
  local body = IR.Statement[s for s in stmts if !(s isa IR.WHEN_STATEMENT)]
  local targets = realTargets(body)
  isempty(targets) && return IR.Equation[]
  if any(s -> statementMentionsInitial(s), body)
    try
      unsupported("an algorithm section reading initial() with Real targets outside a when (not lowered)", join(targets, ", "))
    catch err
      _fallback(err, :algorithmRealAssign; only = UnsupportedLowering, impact = :result)
    end
    return IR.Equation[]
  end
  local (ops, assigned) = bodyStatements(body, BodyLowering(false, true, "an algorithm section with Real targets"))
  local run = AlgorithmRun(assigned, realStarts)
  local values = Tuple{Symbol, IR.CREF}[]
  local out = IR.Equation[]
  for op in ops
    if op isa IR.ASSIGNMENT
      local name = op.lhs.cref.name
      name in run.readAsVariable &&
        unsupported("an algorithm section assigning $(name) after reading it in der(), edge() or change()", op.lhs)
      run.values[name] = algorithmValue(op.rhs, run)
      any(v -> first(v) === name, values) || push!(values, (name, op.lhs))
    elseif op isa IR.ASSERT
      push!(out, IR.ASSERT_EQUATION(algorithmValue(op.condition, run), algorithmValue(op.message, run), op.level, op.source))
    else
      unsupported("this statement in an algorithm section with Real targets", op)
    end
  end
  for (name, lhs) in values
    if isRealType(lhs.ty)
      name in whenLifted && unsupported("a Real variable an algorithm section assigns both in a when and outside one", lhs)
      push!(out, IR.SCALAR_EQUATION(lhs, run.values[name], lhs.ty, source, DYNAMIC))
    elseif lhs.ty isa IR.T_RECORD
      #= as a record equation of the flat model: one of a call stays whole =#
      local value = run.values[name]
      value isa IR.CALL ? push!(out, IR.RECORD_EQUATION(lhs, value, lhs.ty, length(lhs.ty.fields), source, IR.DEFAULT_ATTRIBUTES)) :
                          append!(out, recordEquations(lhs, value, source))
    end
  end
  return out
end

"Whether a statement mentions `initial()` in one of its expressions."
function statementMentionsInitial(s::IR.Statement)::Bool
  local found = Ref(false)
  IR.foreachExp(x -> isInitialCall(x) && (found[] = true), s)
  return found[]
end

"The Real and record targets of the statements, outside when statements, as written."
function realTargets(stmts::Vector{IR.Statement})::Vector{String}
  local out = String[]
  local visit
  visit = s -> begin
    if s isa IR.ASSIGNMENT
      for t in (s.lhs isa IR.TUPLE ? s.lhs.elements : IR.Expression[s.lhs])
        local ty = IR.typeOf(t)
        (t isa IR.CREF && (isRealType(ty) || ty isa IR.T_RECORD)) && push!(out, IR.toString(t))
      end
    elseif s isa IR.IF_STATEMENT
      foreach(b -> foreach(visit, b.body), s.branches); foreach(visit, s.elseBody)
    elseif s isa Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT}
      foreach(visit, s.body)
    end
  end
  foreach(visit, stmts)
  return out
end

"""
The top-level asserts of regular algorithm sections, as assert equations: checked at run time
like those of equation sections. A section with Real targets checks its asserts where they
stand (`realAssignmentEquations`).
"""
function assertsOfAlgorithms(algorithms::Vector{IR.ALGORITHM})::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms
    isempty(alg.statements) && continue
    local body = IR.Statement[s for s in alg.statements if !(s isa IR.WHEN_STATEMENT)]
    (!isempty(realTargets(body)) && !any(statementMentionsInitial, body)) && continue
    for s in alg.statements
      s isa IR.ASSERT && push!(out, IR.ASSERT_EQUATION(s.condition, s.message, s.level, s.source))
    end
  end
  return out
end
