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
  append!(equations, initialWhensOfAlgorithms(algorithms, variables))
  append!(equations, initialAlgorithmWhens(initialAlgorithms, variables))
  if !isempty(algorithms)
    local constrained::Constrained = constrainedBy(equations)
    local known = knownNames(variables)
    #= each section's temporaries, shared by its lowerings as when-equations (its symbolic run has its own) =#
    local pools::IdDict{IR.ALGORITHM, Temporaries} = IdDict{IR.ALGORITHM, Temporaries}(alg => Temporaries(variables, false)
                                                                                         for alg in algorithms)
    local (whens::Vector{IR.Equation}, whenLifted::Set{Symbol}) = liftAlgorithmWhens(algorithms, known, pools)
    append!(equations, whens)
    append!(equations, residualsOfAlgorithms(algorithms, constrained, whenLifted, realStartValues(variables), pools))
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
branch whose condition is `initial()` alone), as initial when-equations (their self-reading array assignments split,
`splitSelfReading`).
"""
function initialWhensOfAlgorithms(algorithms::Vector{IR.ALGORITHM}, variables::IR.VariableSet)::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms, s in alg.statements
    s isa IR.WHEN_STATEMENT || continue
    local b = s.branches[1]
    (b.atInitial && isempty(b.triggers) && !b.atTerminal) || continue
    push!(out, initialWhen(splitSelfReading(b.body, Temporaries(variables, false)), s.source; kind = IR.INITIAL_SECTION))
  end
  return out
end

"The initial algorithm sections (those with statements) as initial when-equations (`splitSelfReading`)."
initialAlgorithmWhens(algorithms::Vector{IR.ALGORITHM}, variables::IR.VariableSet)::Vector{IR.Equation} =
  IR.Equation[initialWhen(splitSelfReading(alg.statements, Temporaries(variables, false)), alg.source; kind = IR.INITIAL_SECTION)
              for alg in algorithms if !isempty(alg.statements)]

"""
    Temporaries(variables, symbolic)

The temporaries of an algorithm section (`emitAssignment!`: the values of an assignment that reads elements it assigns
before, `x := {x[2], x[1]}`, `x := A * x`; an if-statement's conditions that read what its branches assign): of its
lowerings that run as when-equations discrete protected variables `\$ALG_TMP_k` added to `variables` (they hold values
at run time); of its symbolic run (`symbolic`, `realAssignmentEquations`) names only (`\$ALG_RUN_k`, no variables:
the run takes their values in place). One per key (`byKey`), shared by the section's statements, each assigning it
before reading it.
"""
struct Temporaries
  variables::IR.VariableSet
  byKey::Dict{Any, IR.CREF}
  symbolic::Bool
end
Temporaries(variables::IR.VariableSet, symbolic::Bool) = Temporaries(variables, Dict{Any, IR.CREF}(), symbolic)

"""
    splitSelfReading(stmts, temps)

The statements of a body that runs as written (an initial algorithm), each array assignment of known size as its
element assignments (the array code takes those), the values first into temporaries where they read elements it assigns
before them (`x := {x[2], x[1]}`: in order x[2] read the new x[1]; `temps`, as `emitAssignment!`); in if-, for- and
while-statements too. Others as they are.
"""
function splitSelfReading(stmts::Vector{IR.Statement}, temps::Temporaries)::Vector{IR.Statement}
  local out::Vector{IR.Statement} = IR.Statement[]
  for s in stmts
    if s isa IR.ASSIGNMENT && (s.lhs isa IR.ARRAY || (s.lhs isa IR.CREF && IR.typeOf(s.lhs) isa IR.T_ARRAY))
      local targets::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}} =
        s.lhs isa IR.CREF ? scalarTargets(s.lhs) : literalTargets(s.lhs)
      local writes::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, IR.Expression}} =
        Tuple{IR.CREF, Union{Nothing, IR.Expression}, IR.Expression}[(t, g, scalarizedRead(s.rhs, idx, t)) for (t, g, idx) in targets]
      if !isempty(writes) && all(w -> w[2] === nothing, writes)
        if readsEarlierTarget(writes)
          for (k, (t, _, v)) in enumerate(writes)
            local tmp::IR.CREF = temporary!(temps, (:value, IR.toString(t.ty), k), t.ty)
            push!(out, IR.ASSIGNMENT(tmp, v, s.source))
            writes[k] = (t, nothing, tmp)
          end
        end
        foreach(w -> push!(out, IR.ASSIGNMENT(w[1], w[3], s.source)), writes)
        continue
      end
      push!(out, s)
    elseif s isa IR.IF_STATEMENT
      push!(out, IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(b.condition, splitSelfReading(b.body, temps)) for b in s.branches],
                                 splitSelfReading(s.elseBody, temps), s.source))
    elseif s isa IR.FOR_STATEMENT
      push!(out, IR.FOR_STATEMENT(s.iterator, splitSelfReading(s.body, temps), s.source))
    elseif s isa IR.WHILE_STATEMENT
      push!(out, IR.WHILE_STATEMENT(s.condition, splitSelfReading(s.body, temps), s.source))
    else
      push!(out, s)
    end
  end
  return out
end

"""
    Constrained(names, arrays)

What the equations constrain, for the residual lifter (`constrainedBy`): the names of the variables they read, and of
the arrays an element of which they read (`x` of `x[2]`, `c` of `c[1].T`: a reference up to a part with subscripts,
without them).
"""
struct Constrained
  names::Set{Symbol}
  arrays::Set{Symbol}
end

"""
    constrainedBy(equations) -> Constrained

The variables the equations constrain, for the residual lifter: it lifts an algorithm's
assignment only of a variable no equation defines. An alias `a = b` between two
variables (a connection) defines neither (the MSL Digital Set source `y := x`, connected to a
flip-flop's inputs, left them at 0, an invalid logic value).
"""
function constrainedBy(equations::Vector{IR.Equation})::Constrained
  local names::Set{Symbol} = Set{Symbol}()
  local arrays::Set{Symbol} = Set{Symbol}()
  local add::Function = (e::IR.Expression) -> foreach(IR.crefsOf(e)) do c::IR.ComponentRef
    push!(names, c.name)
    for (k, p) in enumerate(c.parts)
      IR.isSubscripted(p) && push!(arrays, IR.ComponentRef(IR.CrefPart[c.parts[1:k - 1]; IR.unsubscriptedPart(p)]).name)
    end
  end
  for eq in equations
    if eq isa IR.SCALAR_EQUATION
      (eq.lhs isa IR.CREF && eq.rhs isa IR.CREF) && continue
      add(eq.lhs); add(eq.rhs)
    elseif eq isa Union{IR.ARRAY_EQUATION, IR.RECORD_EQUATION}
      add(eq.lhs); add(eq.rhs)
    end
  end
  return Constrained(names, arrays)
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
    temporary!(temps, key, ty) -> CREF

The temporary of type `ty` for `key` (an assignment's `k`-th value, its type: `(:value, type, k)`; an if-statement's
`k`-th condition: `(:condition, statement, k)`).
"""
function temporary!(temps::Temporaries, key::Any, ty::IR.ModelicaType)::IR.CREF
  return get!(temps.byKey, key) do
    local stem::String = temps.symbolic ? "\$ALG_RUN_" : "\$ALG_TMP_"
    local n::Int = length(temps.byKey) + 1
    while haskey(temps.variables, Symbol(stem, n))
      n += 1
    end
    local cref::IR.ComponentRef = IR.ComponentRef(Symbol(stem, n), ty)
    temps.symbolic || push!(temps.variables, IR.Variable(cref, ty, IR.DISCRETE(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                                                         true, IR.NOT_CONNECTOR, IR.NO_SOURCE, :liftAlgorithms))
    IR.CREF(cref, ty)
  end
end

"Whether `name` is one of the temporaries `temps` made."
isTemporary(temps::Temporaries, name::Symbol)::Bool = any(c::IR.CREF -> c.cref.name === name, values(temps.byKey))

"""
    BodyLowering(initialValue, allowContinuous, where, temporaries = nothing)

How a body's statements become assignments: `initialValue` the value of `initial()` (true
for the initial arm); `allowContinuous` lowers the assignments of Real targets and the
asserts and terminate() too (a when body), else only the discrete-time assignments (a regular
algorithm); `where` names the body in the messages; `temporaries` the section's (`Temporaries`).
"""
struct BodyLowering
  initialValue::Bool
  allowContinuous::Bool
  where::String
  temporaries::Union{Nothing, Temporaries}
end
BodyLowering(initialValue::Bool, allowContinuous::Bool, where::String) = BodyLowering(initialValue, allowContinuous, where, nothing)

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
  #= the lowerings that run as when-equations: the scalar code's callbacks read element variables =#
  (lowering.temporaries === nothing || lowering.temporaries.symbolic) && return (ops, assigned)
  local variables::IR.VariableSet = lowering.temporaries.variables
  return (IR.Statement[mapRoots(e -> elementChains(e, variables), op) for op in ops], assigned)
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

"""
An if-statement: each branch's statements under its condition and the earlier ones' negations. Conditions that read a
variable the branches assign are evaluated once, before them, into temporaries (each assignment is `v := if c then e
else v`: `c` read after the branch's earlier assignments, `if n == 0 then n := 1; m := m + 1` left m), each only where
the earlier conditions were false (`if d < 0 ... elseif sqrt(d) > 0` never takes the square root of a negative d).
"""
function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.IF_STATEMENT,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  local guard::Union{Nothing, IR.Expression} = active
  local conditions::Vector{IR.Expression} = IR.Expression[prepare(b.condition, iterValues, lowering) for b in s.branches]
  if lowering.temporaries !== nothing && conditionsReadAssigned(conditions, s)
    local reached::Union{Nothing, IR.Expression} = active
    for (k, c) in enumerate(conditions)
      local tmp::IR.CREF = temporary!(lowering.temporaries, (:condition, objectid(s), k), IR.T_BOOLEAN())
      #= where the if-statement runs and the earlier conditions were false (its value elsewhere is not read) =#
      push!(ops, IR.ASSIGNMENT(tmp, reached === nothing ? c : IR.IF_EXPRESSION(reached, c, IR.BOOLEAN(false), IR.T_BOOLEAN()),
                               s.source))
      push!(assigned, tmp.cref.name)
      conditions[k] = tmp
      reached = andCondition(reached, notCondition(tmp))
    end
  end
  for (b, c) in zip(s.branches, conditions)
    appendBody!(ops, assigned, b.body, andCondition(guard, c), iterValues, lowering) || return false
    guard = andCondition(guard, notCondition(c))
  end
  return appendBody!(ops, assigned, s.elseBody, guard, iterValues, lowering)
end

"Whether one of `conditions` reads (outside `pre()`) a variable the if-statement `s` assigns (by its array: `elementBase`)."
function conditionsReadAssigned(conditions::Vector{IR.Expression}, s::IR.IF_STATEMENT)::Bool
  local targets::Set{Symbol} = Set{Symbol}(keys(assignmentCounts(IR.Statement[s])))
  isempty(targets) && return false
  local hit::Bool = false
  for c in conditions
    IR.mapExpTopDown(c) do x
      isBuiltinCallOf(x, (:pre,)) && return (x, false)
      (x isa IR.CREF && elementBase(x.cref) in targets) && (hit = true)
      return (x, true)
    end
  end
  return hit
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

"""
`terminate()`; under a condition an if-statement of it (a when body runs it where the condition holds, an algorithm
section's run makes it the when-equations of its condition: `realAssignmentEquations`).
"""
function appendStatement!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, s::IR.TERMINATE,
                          active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int}, lowering::BodyLowering)::Bool
  (lowering.allowContinuous && !isFalse(active)) || return true
  local t::IR.TERMINATE = IR.TERMINATE(prepare(s.message, iterValues, lowering), s.source)
  push!(ops, (active === nothing || active isa IR.BOOLEAN) ? t :
             IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(active, IR.Statement[t])], IR.Statement[], s.source))
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
one under a guard `i == k` for a subscript `i` that is not a literal; in a lowering that runs as a when-equation a
record's fields, `recordFieldValues`), each the value `rhs` gives it, under the active condition:
`v := if active then e else v`. A Real target is left out unless `allowContinuous`. `false` if `lhs` is not a
reference or names a slice.
"""
function emitAssignment!(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, lhs::IR.Expression, rhs::IR.Expression,
                         source::IR.Source, active::Union{Nothing, IR.Expression}, iterValues::Dict{Symbol, Int},
                         lowering::BodyLowering)::Bool
  (lowering.allowContinuous || isDiscreteType(IR.typeOf(lhs))) || return true
  lhs = prepare(lhs, iterValues, lowering)
  rhs = prepare(rhs, iterValues, lowering)
  lhs isa Union{IR.CREF, IR.ARRAY} || return false
  local writes::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, IR.Expression}} =
    Tuple{IR.CREF, Union{Nothing, IR.Expression}, IR.Expression}[]
  if lhs isa IR.CREF && IR.typeOf(lhs) isa IR.T_RECORD && lowering.temporaries !== nothing && !lowering.temporaries.symbolic
    #= in a lowering that runs as a when-equation the variables are its fields (a when's `r := swapR(r)` assigned no
       variable); the symbolic run keeps the record whole (`AlgorithmRun`'s fields, a call's record equation evaluating
       it once) =#
    isFalse(active) && return true
    for (target, value) in recordFieldValues(lhs, rhs, IR.typeOf(lhs))
      push!(writes, (target, active, value))
    end
  else
    local targets::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}} = lhs isa IR.CREF ? scalarTargets(lhs) : literalTargets(lhs)
    isempty(targets) && return false
    for (target, guard, rhsIndices) in targets
      local c = andCondition(active, guard)
      isFalse(c) && continue
      push!(writes, (target, c, scalarizedRead(rhs, rhsIndices, target)))
    end
  end
  #= a value that reads an element the statement assigns before it (`x := {x[2], x[1]}`, `x := A * x`): the right-hand
     side is evaluated before the assignment (MLS 11.2.1), the values into temporaries first =#
  if lowering.temporaries !== nothing && readsEarlierTarget(writes)
    for (k, (target, c, value)) in enumerate(writes)
      local tmp::IR.CREF = temporary!(lowering.temporaries, (:value, IR.toString(target.ty), k), target.ty)
      push!(ops, IR.ASSIGNMENT(tmp, value, source))
      push!(assigned, tmp.cref.name)
      writes[k] = (target, c, tmp)
    end
  end
  for (target, c, value) in writes
    push!(ops, IR.ASSIGNMENT(target, c === nothing ? value : IR.IF_EXPRESSION(c, value, target, target.ty), source))
    push!(assigned, target.cref.name)
  end
  return true
end

"""
Whether a value of the element assignments `writes` reads an element an assignment before it writes: by its name, or
the array of one whole, through a subscript that is not a literal, or subscripted from outside (`x`, `x[i]`, `x[1]` of
the arrays kept), or the record of a field whole (`r` of `r.a`: `r := swapR(r)`); not inside `pre()`.
"""
function readsEarlierTarget(writes::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, IR.Expression}})::Bool
  local names::Set{Symbol} = Set{Symbol}()
  local bases::Set{Symbol} = Set{Symbol}()
  local earlier::Vector{IR.ComponentRef} = IR.ComponentRef[]
  for (target, _, value) in writes
    local hit::Bool = false
    IR.mapExpTopDown(value) do x
      isBuiltinCallOf(x, (:pre,)) && return (x, false)
      if x isa IR.CREF
        local subs::Vector{IR.Subscript} = IR.subscripts(x.cref)
        (x.cref.name in names ||
         (elementBase(x.cref) in bases && (isempty(subs) || !all(IR.isLiteralSubscript, subs))) ||
         any(t::IR.ComponentRef -> length(t.parts) > length(x.cref.parts) && IR.crefsOverlap(x.cref, t), earlier)) && (hit = true)
      end
      return (x, true)
    end
    hit && return true
    push!(names, target.cref.name)
    push!(bases, elementBase(target.cref))
    push!(earlier, target.cref)
  end
  return false
end

"""
    recordFieldValues(lhs, rhs, ty) -> Vector{Tuple{CREF, Expression}}

`lhs := rhs` of the record type `ty` as the assignments of its fields, a record field's in turn: the value of a
reference, a record value or an if-expression its field (`fieldExpression`); of the record's constructor its argument;
of a function's call its output's field in place (`TUPLE_ELEMENT`: a generated function returns a record's fields in
place, as for `tupleAssignElements`). An array field, or a record field of a call's output: not supported.
"""
function recordFieldValues(lhs::IR.CREF, rhs::IR.Expression, ty::IR.T_RECORD)::Vector{Tuple{IR.CREF, IR.Expression}}
  local out::Vector{Tuple{IR.CREF, IR.Expression}} = Tuple{IR.CREF, IR.Expression}[]
  local constructor::Bool = rhs isa IR.CALL && rhs.name === ty.name && length(rhs.arguments) == length(ty.fields)
  local call::Bool = rhs isa IR.CALL && !rhs.attributes.builtin && !constructor
  for (k, field) in enumerate(ty.fields)
    field.ty isa IR.T_ARRAY && unsupported("an assignment of a record with an array field whole", lhs)
    (call && field.ty isa IR.T_RECORD) && unsupported("an assignment of a function's record output with record fields", rhs)
    local target::IR.CREF = fieldExpression(lhs, field.name, field.ty)
    local value::IR.Expression = constructor ? rhs.arguments[k] : call ? IR.TUPLE_ELEMENT(rhs, k, field.ty) :
                                 fieldExpression(rhs, field.name, field.ty)
    field.ty isa IR.T_RECORD ? append!(out, recordFieldValues(target, value, field.ty)) : push!(out, (target, value))
  end
  return out
end

"""
The subscript of the `k`-th index of a dimension: an enumeration's literal (the scalarized
variables' names carry them), else the integer.
"""
indexOf(d::IR.Dimension, k::Int)::IR.Expression = d isa IR.DIM_ENUM ? IR.dimensionIndex(d, k) : IR.INTEGER(k)

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
    literalTargets(lhs)

The scalar variables an assignment to an array literal of references assigns (`{x[1], x[2]} := {x[2], x[1]}`, the
arrays-kept frontend's `x := ...`): as `scalarTargets`, each element's at its position in the literal (the right-hand
side's element); empty where an element is no reference.
"""
function literalTargets(lhs::IR.ARRAY)::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}}
  local out::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}} = Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}[]
  for (k, e) in enumerate(lhs.elements)
    local inner::Vector{Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}} =
      e isa IR.ARRAY ? literalTargets(e) : e isa IR.CREF ? scalarTargets(e) : Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}[]
    isempty(inner) && return Tuple{IR.CREF, Union{Nothing, IR.Expression}, Vector{Int}}[]
    for (t, g, idx) in inner
      push!(out, (t, g, Int[k; idx]))
    end
  end
  return out
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
      push!(out, (IR.CREF(IR.withLastSubscripts(lhs.cref, newSubs), elementTy), guard, copy(rhsIndices)))
      return true
    end
    local n = IR.dimensionSize(dims[pos])
    local sub = subs[pos]
    if sub isa IR.WHOLE
      n === nothing && return false
      for k in 1:n
        visit(pos + 1, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], k))], guard, Int[rhsIndices..., k]) || return false
      end
    elseif IR.isLiteralSubscript(sub)
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
is not a literal, the chain `if i == 1 then x[1] elseif ...`, `fallback` last; of an index vector or a range, its
element at the position: `x[{2, 1}]`'s first `x[2]`); of another array expression, the expression subscripted.
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
      push!(candidates, (guard, IR.CREF(IR.withLastSubscripts(e.cref, newSubs), elementTy)))
      return
    end
    local sub = subs[pos]
    if sub isa IR.WHOLE
      rhsPos <= length(rhsIndices) || return
      visit(pos + 1, rhsPos + 1, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], rhsIndices[rhsPos]))], guard)
    elseif IR.isLiteralSubscript(sub)
      visit(pos + 1, rhsPos, IR.Subscript[newSubs..., sub], guard)
    elseif sub isa IR.INDEX
      local n = IR.dimensionSize(dims[pos])
      n === nothing && return
      for k in 1:n
        visit(pos + 1, rhsPos, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], k))],
              andCondition(guard, indexEqualsCondition(sub.exp, k)))
      end
    elseif sub isa IR.SLICE
      #= the slice's dimension is the right-hand side's: its index there (`x := x[{2, 1}]` read x[2], then x[1]; it
         read x[1] and x[2], no swap) =#
      rhsPos <= length(rhsIndices) || return
      local picks::Union{Nothing, Vector{IR.Expression}} = sliceIndices(sub.exp)
      (picks === nothing || !(1 <= rhsIndices[rhsPos] <= length(picks))) && return
      local pick::IR.Expression = picks[rhsIndices[rhsPos]]
      if pick isa Union{IR.INTEGER, IR.ENUM_LITERAL}
        visit(pos + 1, rhsPos + 1, IR.Subscript[newSubs..., IR.INDEX(pick)], guard)
      else
        local n = IR.dimensionSize(dims[pos])
        n === nothing && return
        for k in 1:n
          visit(pos + 1, rhsPos + 1, IR.Subscript[newSubs..., IR.INDEX(indexOf(dims[pos], k))],
                andCondition(guard, indexEqualsCondition(pick, k)))
        end
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
    elementChains(e, variables)

`e` with each read of an element of a scalarized array (no variable of the whole in `variables`: the scalar code's)
through a subscript that is not a literal as the chain of the elements, `if i == 1 then x[1] elseif ... else x[n]`
(`scalarizedCref`; an index past the last reads the last): the callbacks read element variables (a when's
`x[k] := x[3 - k] + 10` read the whole `x`, not supported). Reads in `pre()` are left.
"""
elementChains(e::IR.Expression, variables::IR.VariableSet)::IR.Expression =
  IR.mapExpTopDown(e) do x::IR.Expression
    isBuiltinCallOf(x, (:pre,)) && return (x, false)
    x isa IR.CREF || return (x, true)
    local subs::Vector{IR.Subscript} = IR.subscripts(x.cref)
    (!isempty(subs) && all(s -> s isa IR.INDEX, subs) && !all(IR.isLiteralSubscript, subs) &&
     !haskey(variables, elementBase(x.cref))) || return (x, true)
    local padded = paddedSubscripts(x.cref)
    (padded === nothing || length(first(padded)) != length(subs)) && return (x, true)
    local dims::Vector{IR.Dimension} = last(padded)
    any(d -> IR.dimensionSize(d) === nothing, dims) && return (x, true)
    local lastElement::IR.CREF = IR.CREF(IR.withLastSubscripts(x.cref, IR.Subscript[IR.INDEX(indexOf(d, IR.dimensionSize(d))) for d in dims]),
                                         IR.elementType(x.cref.parts[end].ty))
    return (scalarizedCref(x, Int[], lastElement), false)
  end

"The indices a slice subscript picks (`{2, 1}`, `2:3` of literal bounds), in order; `nothing` for another."
function sliceIndices(e::IR.Expression)::Union{Nothing, Vector{IR.Expression}}
  e isa IR.ARRAY && return e.elements
  local range::Union{Nothing, StepRange{Int, Int}} = IndexReduction.literalRange(IndexReduction.foldTree(e))
  return range === nothing ? nothing : IR.Expression[IR.INTEGER(k) for k in range]
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
  local read::Vector{IR.ComponentRef} = crefsReadOutsidePre(rhs)
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
    if any(r::IR.ComponentRef -> IR.crefsOverlap(r, first(element).cref), read)
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
function liftAlgorithmWhens(algorithms::Vector{IR.ALGORITHM}, known::Set{Symbol},
                            pools::IdDict{IR.ALGORITHM, Temporaries})::Tuple{Vector{IR.Equation}, Set{Symbol}}
  local out = IR.Equation[]
  local lifted = Set{Symbol}()
  for alg in algorithms
    isempty(alg.statements) && continue
    local temps::Temporaries = pools[alg]
    if all(s -> s isa IR.WHEN_STATEMENT, alg.statements)
      foreach(s -> liftAlgorithmWhenStatement!(out, s, lifted, temps), alg.statements)
      continue
    end
    local counts = assignmentCounts(alg.statements)
    #= the conditions of the ifs lifted on their own: the body's lifting has no when of their sample() =#
    local ownWhens = IR.Expression[s.branches[1].condition for s in alg.statements if liftIfToWhen!(out, s, alg.source, counts)]
    isSingleDiscreteAssignment(alg.statements) || liftBodyToWhens!(out, alg.statements, alg.source, lifted, known, ownWhens, temps)
    for s in alg.statements
      s isa IR.WHEN_STATEMENT && liftAlgorithmWhenStatement!(out, s, lifted, temps)
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
function liftAlgorithmWhenStatement!(out::Vector{IR.Equation}, s::IR.WHEN_STATEMENT, lifted::Set{Symbol},
                                     temps::Temporaries)::Nothing
  local first = s.branches[1]
  if first.atInitial && isempty(first.triggers) && !first.atTerminal
    length(s.branches) > 1 || return nothing
    local elsewhens::Union{Nothing, IR.WHEN_EQUATION} = whenEquationOf(s.branches[2:end], s.source, lifted, temps)
    elsewhens === nothing || push!(out, elsewhens)
    return nothing
  end
  if first.atInitial
    #= An initial arm the lifter cannot lower stays empty (its variables start from their
       start values): a fallback that changes the result, reported. =#
    local (initOps, initAssigned) = try
      bodyStatements(first.body, BodyLowering(true, true, "a when body", temps))
    catch err
      _fallback(err, :algorithmInitialArm; only = UnsupportedLowering, impact = :result)
      (IR.Statement[], Symbol[])
    end
    union!(lifted, initAssigned)
    isempty(initOps) || push!(out, initialWhen(initOps, s.source))
  end
  local weq::Union{Nothing, IR.WHEN_EQUATION} = whenEquationOf(s.branches, s.source, lifted, temps)
  weq === nothing || push!(out, weq)
  return nothing
end

"""
The when-equation of a when statement's branches: the runtime arm (`initial()` false), every
assignment lifted (continuous and discrete: in a when all targets are event-updated), the
if-statements as guarded assignments. A branch that never runs (`when initial()` alone)
contributes nothing. `nothing` if no branch has statements.
"""
function whenEquationOf(branches::Vector{IR.WhenBranch}, source::IR.Source, lifted::Set{Symbol},
                        temps::Temporaries)::Union{Nothing, IR.WHEN_EQUATION}
  local out = IR.WhenBranch[]
  for b in branches
    local triggers = IR.Expression[replaceInitial(t, false) for t in b.triggers]
    local runs = !isempty(triggers) || b.atTerminal
    local (ops, assigned) = runs ? bodyStatements(b.body, BodyLowering(false, true, "a when body", temps)) : (IR.Statement[], Symbol[])
    union!(lifted, assigned)
    push!(out, IR.WhenBranch(triggers, false, b.atTerminal, preStarts(ops)))
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
  get(counts, elementBase(a.lhs.cref), 0) == 1 || return false
  push!(out, IR.WHEN_EQUATION([IR.WhenBranch(IR.Expression[s.branches[1].condition], false, false,
                                             preStarts(IR.Statement[a]))],
                              IR.REGULAR_WHEN, source, IR.DEFAULT_ATTRIBUTES))
  return true
end

"""
    preStarts(ops)

The straight-line statements of an algorithm's when-branch (or the body a when on its events runs), each target the
body reads before or in its first assignment (`eventReads!`: outside `pre()`, and the subscripts inside it; a read or a
target as far as it may reach, `readExtent`: an element target read through its whole array, a slice, a variable
subscript, `edge()` or `change()` too) first set to its `pre()` value, and every element a target through a subscript
that is not a literal may assign (`targetElements`: `s[k].a` all `s[i].a`): the variables an algorithm assigns start
from their pre() values (MLS 11.1.2), so `(s, y) := step(s)` reads the `s` before the event and `edge(flag)` before
`flag := ...` is false. The array code's event iteration reruns a fired body with the same pre() until nothing changes:
a body that read its targets' values of the previous pass never settled. A target no earlier statement assigns (through
any reference that may overlap it) and its own first assignment reads only as itself (an if-statement's guarded
`v := if c then e else v`) reads `pre(v)` there instead, no start; a body that reads no target before assigning it as
it is. UnsupportedLowering for a target through a subscript of a slice or of a dimension of no known size.
"""
function preStarts(ops::Vector{IR.Statement})::Vector{IR.Statement}
  #= the references read by the statements before, and those assigned, by their variable's first part, each as far as
     it may reach (`readExtent`) =#
  local reads::Dict{Symbol, Set{IR.ComponentRef}} = Dict{Symbol, Set{IR.ComponentRef}}()
  local assignedRefs::Dict{Symbol, Set{IR.ComponentRef}} = Dict{Symbol, Set{IR.ComponentRef}}()
  local add!::Function = (table::Dict{Symbol, Set{IR.ComponentRef}}, c::IR.ComponentRef) ->
    push!(get!(() -> Set{IR.ComponentRef}(), table, c.parts[1].name), readExtent(c))
  local note::Function = (e::IR.Expression) -> (foreach((c::IR.ComponentRef) -> add!(reads, c), eventReads!(IR.ComponentRef[], e)); e)
  local overlaps::Function = (table::Dict{Symbol, Set{IR.ComponentRef}}, target::IR.ComponentRef) -> begin
    local seen::Union{Nothing, Set{IR.ComponentRef}} = get(table, target.parts[1].name, nothing)
    seen === nothing && return false
    local extent::IR.ComponentRef = readExtent(target)
    extent in seen && return true
    return any((r::IR.ComponentRef) -> IR.crefsOverlap(r, extent), seen)
  end
  local reads!::Function = (op::IR.ASSIGNMENT) -> begin
    note(op.rhs)
    #= a target's subscripts are reads =#
    op.lhs isa IR.CREF && foreach((sub::IR.Subscript) -> (sub isa IR.INDEX && note(sub.exp)), allSubscripts(op.lhs.cref))
  end
  local assigned::Set{IR.Expression} = Set{IR.Expression}()
  local started::Set{IR.ComponentRef} = Set{IR.ComponentRef}()
  local starts::Vector{IR.Statement} = IR.Statement[]
  local start!::Function = (c::IR.ComponentRef, op::IR.ASSIGNMENT) -> begin
    c in started && return nothing
    push!(started, c)
    local ref::IR.CREF = IR.CREF(c, IR.typeOf(op.lhs))
    push!(starts, IR.ASSIGNMENT(ref, builtinCall(:pre, IR.Expression[ref], IR.typeOf(op.lhs)), op.source))
    return nothing
  end
  local out::Vector{IR.Statement} = IR.Statement[]
  for op in ops
    if !(op isa IR.ASSIGNMENT)
      mapRoots(note, op)
      push!(out, op)
      continue
    end
    if !(op.lhs isa IR.CREF) || op.lhs in assigned
      reads!(op)
      op.lhs isa IR.CREF && add!(assignedRefs, op.lhs.cref)
      push!(out, op)
      continue
    end
    push!(assigned, op.lhs)
    local target::IR.ComponentRef = op.lhs.cref
    if readExtent(target) != target
      #= through a subscript that is no literal: every element it may assign starts from pre() (its index may change
         between the passes and within the body) =#
      local elements::Union{Nothing, Vector{IR.ComponentRef}} = targetElements(target)
      elements === nothing && unsupported("an algorithm's target through a subscript of a slice or of a dimension of no known size", target)
      foreach((c::IR.ComponentRef) -> start!(c, op), elements)
      reads!(op)
      add!(assignedRefs, target)
      push!(out, op)
      continue
    end
    local start::Bool = overlaps(reads, target)
    local first::IR.ASSIGNMENT = op
    if !start
      #= its own first assignment reading it only as itself, no earlier statement assigning it (through any reference):
         that reads pre(v), no start =#
      local own::IR.ASSIGNMENT = IR.ASSIGNMENT(op.lhs, IR.mapExpTopDown(op.rhs) do x::IR.Expression
        (IR.isBuiltinCall(x, :pre) || IR.isBuiltinCall(x, :change) || IR.isBuiltinCall(x, :edge)) && return (x, false)
        x == op.lhs && return (builtinCall(:pre, IR.Expression[x], IR.typeOf(x)), false)
        return (x, true)
      end, op.source)
      local ownReads::Dict{Symbol, Set{IR.ComponentRef}} = Dict{Symbol, Set{IR.ComponentRef}}()
      foreach((c::IR.ComponentRef) -> add!(ownReads, c), eventReads!(IR.ComponentRef[], own.rhs))
      for sub in allSubscripts(target)
        sub isa IR.INDEX && foreach((c::IR.ComponentRef) -> add!(ownReads, c), eventReads!(IR.ComponentRef[], sub.exp))
      end
      local readsItself::Bool = own.rhs != op.rhs
      if overlaps(ownReads, target) || (readsItself && overlaps(assignedRefs, target))
        start = true
      else
        first = own
      end
    end
    start && start!(target, op)
    reads!(first)
    add!(assignedRefs, target)
    push!(out, first)
  end
  return IR.Statement[starts; out]
end

"""
    eventReads!(out, e) -> Vector{ComponentRef}

The references `e` reads at the event, pushed onto `out`: outside `pre()`, and inside it the subscripts' (`pre(buf[idx])`
reads `idx` now).
"""
function eventReads!(out::Vector{IR.ComponentRef}, e::IR.Expression)::Vector{IR.ComponentRef}
  if IR.isBuiltinCall(e, :pre)
    IR.foreachExp(e) do x::IR.Expression
      x isa IR.CREF || return nothing
      for sub in allSubscripts(x.cref)
        sub isa IR.INDEX && eventReads!(out, sub.exp)
      end
      return nothing
    end
    return out
  end
  e isa IR.CREF && push!(out, e.cref)
  for c::IR.Expression in IR.children(e)
    eventReads!(out, c)
  end
  return out
end

"The names of the references `ops` assign (the starts `preStarts` adds too)."
assignedNames(ops::Vector{IR.Statement})::Set{Symbol} =
  Set{Symbol}(op.lhs.cref.name for op in ops if op isa IR.ASSIGNMENT && op.lhs isa IR.CREF)

"""
    readExtent(c) -> ComponentRef

The part of the variables a reference `c` may reach: `c`, or `c` cut before its first subscript that is not a literal
(an Integer's or an enumeration literal's: an iterator's, a parameter's, a slice: `x[i]` may be any element of `x`,
`a[i].x` any of `a`).
"""
function readExtent(c::IR.ComponentRef)::IR.ComponentRef
  for (k, p) in enumerate(c.parts)
    all(IR.isLiteralSubscript, p.subscripts) && continue
    return IR.ComponentRef(IR.CrefPart[c.parts[1:(k - 1)]; IR.unsubscriptedPart(p)])
  end
  return c
end

"""
    targetElements(c) -> Union{Nothing, Vector{ComponentRef}}

The elements a target reference may assign: `c` with each subscript that is not a literal replaced by every index of its
dimension (`s[k].a`: `s[1].a`, `s[2].a`; the product of those dimensions' sizes, as a variable has elements);
`nothing` for a slice or a dimension of no known size.
"""
function targetElements(c::IR.ComponentRef)::Union{Nothing, Vector{IR.ComponentRef}}
  local prefixes::Vector{Vector{IR.CrefPart}} = Vector{IR.CrefPart}[IR.CrefPart[]]
  for p in c.parts
    local choices::Vector{Vector{IR.Subscript}} = Vector{IR.Subscript}[IR.Subscript[]]
    local dims::Vector{IR.Dimension} = IR.dimensions(p.ty)
    all(IR.isLiteralSubscript, p.subscripts) || length(dims) >= length(p.subscripts) || return nothing
    for (k, sub) in enumerate(p.subscripts)
      if IR.isLiteralSubscript(sub)
        choices = Vector{IR.Subscript}[IR.Subscript[ch; sub] for ch in choices]
        continue
      end
      sub isa IR.INDEX || return nothing
      local n::Union{Nothing, Int} = IR.dimensionSize(dims[k])
      n === nothing && return nothing
      choices = Vector{IR.Subscript}[IR.Subscript[ch; IR.INDEX(IR.dimensionIndex(dims[k], i))] for ch in choices for i in 1:n]
    end
    prefixes = Vector{IR.CrefPart}[IR.CrefPart[q; IR.CrefPart(p.name, ch, p.ty)] for q in prefixes for ch in choices]
  end
  return IR.ComponentRef[IR.ComponentRef(q) for q in prefixes]
end

"""
    assignmentCounts(stmts)

How often the statements assign each variable (the targets of assignments and tuple
assignments, in if-, for-, while- and when-statements too), an element of an array as its array
(`elementBase`: `x[k]` and `x[1]` are one variable's).
"""
function assignmentCounts(stmts::Vector{IR.Statement})::Dict{Symbol, Int}
  local counts = Dict{Symbol, Int}()
  local visit
  visit = s -> begin
    if s isa IR.ASSIGNMENT
      for t in (s.lhs isa IR.TUPLE ? s.lhs.elements : IR.Expression[s.lhs])
        t isa IR.CREF && (counts[elementBase(t.cref)] = get(counts, elementBase(t.cref), 0) + 1)
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
                          lifted::Set{Symbol}, known::Set{Symbol}, ownWhens::Vector{IR.Expression}, temps::Temporaries)::Nothing
  local (initOps, initAssigned) = bodyStatements(stmts, BodyLowering(true, false, "an algorithm section", temps))
  local (runOps, runAssigned) = bodyStatements(stmts, BodyLowering(false, false, "an algorithm section", temps))
  union!(lifted, initAssigned)
  union!(lifted, runAssigned)
  (isempty(initOps) && isempty(runOps)) && return nothing
  isempty(initOps) || push!(out, initialWhen(initOps, source))
  #= an input read only through pre() changes the body one pass after it changes: `change(pre(x))`, the
     lifter's iterated form (MSL RadioButtonSource's `pre_reset := pre(reset)` fired on change(reset), read the
     old value, and never ran again: the buttons never reset) =#
  runOps = IR.Statement[op isa IR.ASSIGNMENT ? IR.ASSIGNMENT(op.lhs, preOfElements(op.rhs), op.source) : op for op in runOps]
  #= the run arm reruns at each pass of an event's iteration: the targets it reads start from their pre() values (MLS
     11.1.2; `if u <> pre(u) then n := n + u; end if` grew by u at every pass and never settled) =#
  runOps = preStarts(runOps)
  local events::Vector{IR.CREF} = hoistEventFunctions!(out, runOps, known, temps.variables, source)
  local direct = readOutsidePre(runOps)
  local triggers = IR.Expression[c.cref.name in direct ? changeCall(c) : changeCall(builtinCall(:pre, IR.Expression[c], c.ty))
                                 for c in discreteInputs(runOps, union(Set{Symbol}(runAssigned), assignedNames(runOps)))]
  local relationsSeen = IR.Expression[]
  for op in runOps
    op isa IR.ASSIGNMENT || continue
    for r in relationsIn(op.rhs)
      (hasContinuousOperand(r, known) && !(r in relationsSeen)) || continue
      push!(relationsSeen, r)
    end
  end
  append!(triggers, IR.Expression[changeCall(r) for r in relationsSeen])
  #= a Real one (of floor()) is no discrete input by its type =#
  for k in events
    local c::IR.CALL = changeCall(k)
    c in triggers || push!(triggers, c)
  end
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

"""
    hoistEventFunctions!(out, ops, known, variables, source) -> Vector{CREF}

The run arm's assignments `ops` (in place) with each event function of a continuous-time argument
(`EVENT_FUNCTIONS`, `readsContinuousTime`, outside `noEvent()`) read from a discrete protected variable `\$ALG_EVT_k`
of `variables`, an equation of `out` its definition: `k = integer(q)`, `q` the argument with the run's earlier
assignments in place (its value where the statement reads it). The run's trigger is its change: lifted beside other
statements, `n := integer(10 x)` ran at none of the argument's crossings and stayed 0 (MLS, "Event Triggering
Mathematical Functions": these functions make events outside when-clauses). `mod(a, b)` reads `a - k b` of
`k = floor(a / b)`, `rem(a, b)` of `k = div(a, b)`.
An argument that reads what the run assigns where the substitution does not reach (a whole array of assigned
elements) keeps its call, reported. Returns the variables.
"""
function hoistEventFunctions!(out::Vector{IR.Equation}, ops::Vector{IR.Statement}, known::Set{Symbol},
                              variables::IR.VariableSet, source::IR.Source)::Vector{IR.CREF}
  local current::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}()
  local names::Set{Symbol} = assignedNames(ops)
  local bases::Set{Symbol} = Set{Symbol}(elementBase(op.lhs.cref) for op in ops if op isa IR.ASSIGNMENT && op.lhs isa IR.CREF)
  local byStep::Dict{IR.Expression, IR.CREF} = Dict{IR.Expression, IR.CREF}()
  for (i, op) in enumerate(ops)
    (op isa IR.ASSIGNMENT && op.lhs isa IR.CREF) || continue
    local rhs::IR.Expression = IR.mapExpTopDown(op.rhs) do x::IR.Expression
      IR.isBuiltinCall(x, :noEvent) && return (x, false)
      (isBuiltinCallOf(x, EVENT_FUNCTIONS) && any(a -> readsContinuousTime(a, variables, known), x.arguments)) ||
        return (x, true)
      local args::Vector{IR.Expression} = IR.Expression[inlineSiblings(a, current) for a in x.arguments]
      if any(a -> readsUnsubstituted(a, names, bases), args)
        try
          unsupported("an event function of what the algorithm assigns, read as a whole", x)
        catch err
          _fallback(err, :algorithmEventFunction; only = UnsupportedLowering, impact = :result)
        end
        return (x, false)
      end
      local step::IR.CALL = x.name === :mod ? builtinCall(:floor, IR.Expression[IR.BINARY(args[1], IR.DIV, args[2], x.ty)], x.ty) :
                            x.name === :rem ? builtinCall(:div, args, x.ty) : builtinCall(x.name, args, x.ty)
      local k::IR.CREF = get!(byStep, step) do
        local n::Int = length(byStep) + 1
        while haskey(variables, Symbol("\$ALG_EVT_", n))
          n += 1
        end
        local cref::IR.ComponentRef = IR.ComponentRef(Symbol("\$ALG_EVT_", n), x.ty)
        push!(variables, IR.Variable(cref, x.ty, IR.DISCRETE(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION, true,
                                     IR.NOT_CONNECTOR, IR.NO_SOURCE, :liftAlgorithms))
        local v::IR.CREF = IR.CREF(cref, x.ty)
        push!(out, IR.SCALAR_EQUATION(v, step, x.ty, source, IR.DEFAULT_ATTRIBUTES))
        v
      end
      x.name in (:mod, :rem) || return (k, false)
      return (IR.BINARY(x.arguments[1], IR.SUB, IR.BINARY(k, IR.MUL, x.arguments[2], x.ty), x.ty), false)
    end
    ops[i] = IR.ASSIGNMENT(op.lhs, rhs, op.source)
    current[op.lhs.cref.name] = inlineSiblings(rhs, current)
  end
  return collect(IR.CREF, values(byStep))
end

"""
Whether `e` reads, outside `pre()`, one of the assigned `names`, or an array of `bases` whole or by a subscript that is
not literal (what `inlineSiblings` does not replace by the run's values).
"""
function readsUnsubstituted(e::IR.Expression, names::Set{Symbol}, bases::Set{Symbol})::Bool
  local hit::Base.RefValue{Bool} = Ref(false)
  IR.mapExpTopDown(e) do x::IR.Expression
    isBuiltinCallOf(x, (:pre,)) && return (x, false)
    if x isa IR.CREF
      local subs::Vector{IR.Subscript} = IR.subscripts(x.cref)
      (x.cref.name in names ||
       (elementBase(x.cref) in bases && (isempty(subs) || !all(IR.isLiteralSubscript, subs)))) && (hit[] = true)
    end
    return (x, !hit[])
  end
  return hit[]
end

"The `sample()` calls in `e`."
function samplesIn(e::IR.Expression)::Vector{IR.Expression}
  local out = IR.Expression[]
  IR.foreachExp(x -> IR.isBuiltinCall(x, :sample) && push!(out, x), e)
  return out
end

"""
    preOfElements(e)

`e` with `pre({a, b, ...})[k]` (a literal index) as `pre(element k)`: the frontend's form of an array's
`pre()` read by element (RadioButtonSource's `pre(reset)`).
"""
preOfElements(e::IR.Expression)::IR.Expression = IR.mapExp(e) do x
  (x isa IR.SUBSCRIPTED && length(x.subscripts) == 1 && x.subscripts[1] isa IR.INDEX &&
   x.subscripts[1].exp isa IR.INTEGER && IR.isBuiltinCall(x.exp, :pre) && length(x.exp.arguments) == 1 &&
   x.exp.arguments[1] isa IR.ARRAY) || return x
  local elements = x.exp.arguments[1].elements
  local k = x.subscripts[1].exp.value
  (1 <= k <= length(elements)) || return x
  return builtinCall(:pre, IR.Expression[elements[k]], IR.typeOf(elements[k]))
end

"""
    readOutsidePre(ops) -> Set{Symbol}

The variables the statements' expressions read outside `pre()`.
"""
function readOutsidePre(ops::Vector{IR.Statement})::Set{Symbol}
  local out = Set{Symbol}()
  local visit
  visit = e -> begin
    IR.isBuiltinCall(e, :pre) && return nothing
    e isa IR.CREF && push!(out, e.cref.name)
    foreach(visit, IR.children(e))
    nothing
  end
  for op in ops
    if op isa IR.ASSIGNMENT
      visit(op.rhs)
    elseif op isa IR.NORETCALL
      visit(op.call)
    elseif op isa IR.ASSERT
      visit(op.condition); visit(op.message); visit(op.level)
    end
  end
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
function residualsOfAlgorithms(algorithms::Vector{IR.ALGORITHM}, constrained::Constrained, whenLifted::Set{Symbol},
                               realStarts::Dict{Symbol, IR.Expression}, pools::IdDict{IR.ALGORITHM, Temporaries})::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms
    any(s -> !(s isa IR.WHEN_STATEMENT), alg.statements) || continue
    append!(out, realAssignmentEquations(alg.statements, alg.source, realStarts, whenLifted, Temporaries(pools[alg].variables, true)))
    count(s -> s isa IR.ASSIGNMENT && !(s.lhs isa IR.TUPLE), alg.statements) == 1 || continue
    for s in alg.statements
      (s isa IR.ASSIGNMENT && !(s.lhs isa IR.TUPLE)) || continue
      residualsOfAssignment!(out, s, constrained, whenLifted)
    end
  end
  return out
end

"A dynamic equation: already classified, not one for the discrete synthesis."
const DYNAMIC = IR.EquationAttributes(IR.EQ_CONTINUOUS, false, nothing, false)

"""
The residuals of `v := e` for a discrete `v`: `v = e`; of an array `v`, one per element
(`v[k] = e[k]`), each unless an equation constrains it or the when lifter assigns it.
"""
function residualsOfAssignment!(out::Vector{IR.Equation}, s::IR.ASSIGNMENT, constrained::Constrained, whenLifted::Set{Symbol})::Nothing
  local ty = IR.typeOf(s.lhs)
  isDiscreteType(ty) || return nothing
  local lhs = s.lhs
  if !(ty isa IR.T_ARRAY)
    lhs isa IR.CREF && (lhs.cref.name in constrained.names || lhs.cref.name in whenLifted) && return nothing
    push!(out, IR.SCALAR_EQUATION(lhs, s.rhs, ty, s.source, DYNAMIC))
    return nothing
  end
  lhs isa IR.CREF && lhs.cref.name in whenLifted && return nothing
  local n = IR.dimensionSize(ty.dimensions[1])
  if !(lhs isa IR.CREF) || n === nothing || n <= 0
    #= Unknown shape: not if an equation constrains the variable or one of its elements. =#
    if lhs isa IR.CREF
      (lhs.cref.name in constrained.names || lhs.cref.name in constrained.arrays) && return nothing
    end
    push!(out, IR.SCALAR_EQUATION(lhs, s.rhs, ty, s.source, DYNAMIC))
    return nothing
  end
  local elementTy = IR.elementType(ty)
  for k in 1:n
    local cref::IR.ComponentRef = IR.appendSubscripts(lhs.cref, IR.Subscript[IR.INDEX(IR.INTEGER(k))])
    (cref.name in constrained.names || cref.name in whenLifted) && continue
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

"""
    AlgorithmRun(ops, assigned, realStarts)

The run of the lowered statements `ops` (their assignments' targets `assigned`): the wholes of each target's
reference (`wholesOf`).
"""
function AlgorithmRun(ops::Vector{IR.Statement}, assigned::Vector{Symbol}, realStarts::Dict{Symbol, IR.Expression})
  local wholes::Set{Symbol} = Set{Symbol}()
  for op in ops
    op isa IR.ASSIGNMENT && union!(wholes, wholesOf((op.lhs::IR.CREF).cref))
  end
  return AlgorithmRun(Dict{Symbol, IR.Expression}(), Set{Symbol}(assigned), wholes, realStarts, Set{Symbol}())
end

"""
    wholesOf(cref) -> Vector{Symbol}

The names of the wholes a reference is a part of, by its parts: each part with subscripts without them (`r.a` of
`r.a[2]`, `x` of `x[2]`), each part before the last as it is (`r` of `r.a`, `r.a[2]` of `r.a[2].b`).
"""
function wholesOf(cref::IR.ComponentRef)::Vector{Symbol}
  local out::Vector{Symbol} = Symbol[]
  for (k, p) in enumerate(cref.parts)
    IR.isSubscripted(p) && push!(out, IR.ComponentRef(IR.CrefPart[cref.parts[1:k - 1]; IR.unsubscriptedPart(p)]).name)
    k < length(cref.parts) && push!(out, IR.ComponentRef(cref.parts[1:k]).name)
  end
  return out
end

"Above this many nodes a value is not built: `if v[i] > m then m := v[i]` doubles it each time."
const ALGORITHM_VALUE_NODE_LIMIT = 10_000

"""
    algorithmValue(e, run; variables)

`e` with the section's variables read as their values so far: assigned before, the value;
assigned only later, the start value (Real) or `pre()` (discrete). Inside `pre()` a variable is
the value before the event; inside `der()`, `edge()` and `change()` the variable itself, which
its equation defines (its value at the end: not supported if it is assigned again later). The names `variables` are
read as the variables (their value at that point is the variable's).
"""
function algorithmValue(e::IR.Expression, run::AlgorithmRun; variables::Set{Symbol} = Set{Symbol}())::IR.Expression
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
    name in variables && return (x, false)
    haskey(run.values, name) && return (run.values[name], false)
    if name in run.assigned
      isRealType(x.ty) && return (get(run.realStarts, name, IR.REAL(0.0)), false)
      return (preCall(IR.CREF(cref, x.ty)), false)
    end
    name in run.wholes && unsupported("a read of a whole array or record that an algorithm section assigns part by part", x)
    #= x[j]: one of the elements the section assigns, chosen by the index (its array by its parts, `elementBase`) =#
    local base::Symbol = elementBase(cref)
    if base !== name && base in run.wholes && any(s -> !IR.isLiteralSubscript(s), IR.subscripts(cref))
      local elements = scalarizedCref(IR.CREF(cref, x.ty), Int[], x)
      elements isa IR.IF_EXPRESSION || unsupported("this read of an array an algorithm section assigns", x)
      return (elementValues(elements, x, run, variables), false)
    end
    return (x, true)
  end
  local value = IR.mapExpTopDown(visit, e)
  expressionLargerThan(value, ALGORITHM_VALUE_NODE_LIMIT) &&
    unsupported("an algorithm section whose values exceed $(ALGORITHM_VALUE_NODE_LIMIT) nodes as equations", e)
  return value
end

"The if-chain over the elements a read `x[j]` may be, each element (and guard) its value (`variables` read as such)."
function elementValues(chain::IR.Expression, read::IR.Expression, run::AlgorithmRun,
                       variables::Set{Symbol})::IR.Expression
  chain === read && return read
  return IR.IF_EXPRESSION(algorithmValue(chain.condition, run; variables),
                          algorithmValue(chain.thenExp, run; variables),
                          elementValues(chain.elseExp, read, run, variables), chain.ty)
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
equation (`recordValueEquations`); an assert checked on the values at its point. The discrete targets stay the when
lifter's. A section reading `initial()` is reported, not lowered.
"""
function realAssignmentEquations(stmts::Vector{IR.Statement}, source::IR.Source,
                                 realStarts::Dict{Symbol, IR.Expression}, whenLifted::Set{Symbol},
                                 temps::Temporaries)::Vector{IR.Equation}
  local body = IR.Statement[s for s in stmts if !(s isa IR.WHEN_STATEMENT)]
  local targets = realTargets(body)
  #= a section without Real targets runs for its asserts and terminate() (their values where they stand); one reading
     initial() is `assertsOfAlgorithms`' =#
  (isempty(targets) && !runsSymbolically(body)) && return IR.Equation[]
  if any(s -> statementMentionsInitial(s), body)
    try
      unsupported("an algorithm section reading initial() with Real targets outside a when (not lowered)", join(targets, ", "))
    catch err
      _fallback(err, :algorithmRealAssign; only = UnsupportedLowering, impact = :result)
    end
    return IR.Equation[]
  end
  local (ops::Vector{IR.Statement}, assigned::Vector{Symbol}) = if isempty(targets)
    try
      bodyStatements(body, BodyLowering(false, true, "an algorithm section with asserts or terminate()", temps))
    catch err
      #= a body the run does not take (a for loop over a non-constant range): its asserts as they stand =#
      _fallback(err, :algorithmEffects; only = UnsupportedLowering, impact = :result)
      local hoisted::Vector{IR.Equation} = IR.Equation[]
      effectsUnder!(hoisted, body, nothing)
      return hoisted
    end
  else
    bodyStatements(body, BodyLowering(false, true, "an algorithm section with Real targets", temps))
  end
  local run::AlgorithmRun = AlgorithmRun(ops, assigned, realStarts)
  local values = Tuple{Symbol, IR.CREF}[]
  local out = IR.Equation[]
  #= an assert or terminate() reads a discrete target the when lowering assigns as the variable where its value there is
     its last (the run's value inlined, `n >= 5` of `n := if x > 0.5 then 5 else 0` was
     `(if x > 0.5 then 5 else 0) >= 5`, whose operand lands on the threshold: the array code's hysteresis never fired it) =#
  local lastAssigned::Dict{Symbol, Int} = Dict{Symbol, Int}(op.lhs.cref.name => i for (i, op) in enumerate(ops)
                                                            if op isa IR.ASSIGNMENT && op.lhs isa IR.CREF)
  local settledAt::Function = (i::Int) -> Set{Symbol}(name for (name, j) in lastAssigned
                                            if j < i && name in whenLifted && !isTemporary(temps, name) &&
                                               isDiscreteType(ops[j].lhs.ty))
  for (i, op) in enumerate(ops)
    local settled::Set{Symbol} = op isa IR.ASSIGNMENT ? Set{Symbol}() : settledAt(i)
    if op isa IR.ASSIGNMENT
      local name = op.lhs.cref.name
      name in run.readAsVariable &&
        unsupported("an algorithm section assigning $(name) after reading it in der(), edge() or change()", op.lhs)
      run.values[name] = algorithmValue(op.rhs, run)
      op.lhs.ty isa IR.T_RECORD && assignFields!(run, op.lhs.cref, run.values[name], op.lhs.ty)
      #= a temporary's value is read in place, it is no variable =#
      (isTemporary(temps, name) || any(v -> first(v) === name, values)) || push!(values, (name, op.lhs))
    elseif op isa IR.ASSERT
      push!(out, IR.ASSERT_EQUATION(algorithmValue(op.condition, run; variables = settled),
                                    algorithmValue(op.message, run; variables = settled), op.level, op.source))
    elseif op isa IR.TERMINATE || (op isa IR.IF_STATEMENT && only(only(op.branches).body) isa IR.TERMINATE)
      #= under its condition's value there =#
      local t::IR.TERMINATE = op isa IR.TERMINATE ? op : only(only(op.branches).body)
      local guard::Union{Nothing, IR.Expression} =
        op isa IR.TERMINATE ? nothing : algorithmValue(only(op.branches).condition, run; variables = settled)
      local message::IR.Expression = algorithmValue(t.message, run; variables = settled)
      append!(out, guardedTerminate(terminateMarker(IR.TERMINATE(message, t.source)), guard))
    else
      unsupported("this statement in an algorithm section with Real targets", op)
    end
  end
  for (name, lhs) in values
    if isRealType(lhs.ty)
      name in whenLifted && unsupported("a Real variable an algorithm section assigns both in a when and outside one", lhs)
      push!(out, IR.SCALAR_EQUATION(lhs, run.values[name], lhs.ty, source, DYNAMIC))
    elseif lhs.ty isa IR.T_RECORD
      append!(out, recordValueEquations(lhs, run.values[name], IR.ComponentRef[last(v).cref for v in values], source))
    end
  end
  return out
end

"""
    assignFields!(run, cref, value, ty)

After the whole record `cref := value`: each of its fields the section also assigns by itself
its field of `value`, until that assignment (`vp := init(rp, lp)` then `vp.m_dConduct := ...`,
MSL Spice3's semiconductor resistor: a read in between is `init(rp, lp)[4]`).
"""
function assignFields!(run::AlgorithmRun, cref::IR.ComponentRef, value::IR.Expression, ty::IR.T_RECORD)::Nothing
  for (k, field) in enumerate(ty.fields)
    local fieldCref::IR.ComponentRef = IR.withField(cref, field.name, field.ty)
    local fieldName::Symbol = fieldCref.name
    local fieldValue = fieldByIndex(value, k)
    fieldName in run.assigned && (run.values[fieldName] = fieldValue)
    fieldName in run.wholes || continue
    field.ty isa IR.T_RECORD ? assignFields!(run, fieldCref, fieldValue, field.ty) :
      unsupported("an algorithm section assigning elements of an array field of a record it assigns whole", fieldName)
  end
  return nothing
end

"""
    recordValueEquations(lhs, value, own, source) -> Vector{Equation}

The equations of a record variable an algorithm section assigns whole, `lhs = value`: as a
record equation of the flat model (one of a call stays whole); without the fields the section
also assigns by themselves (`own`, the targets' references, whose equations are their own: their values at the end),
field by field.
"""
function recordValueEquations(lhs::IR.CREF, value::IR.Expression, own::Vector{IR.ComponentRef}, source::IR.Source)::Vector{IR.Equation}
  local n::Int = length(lhs.cref.parts)
  if !any(c::IR.ComponentRef -> length(c.parts) > n && IR.ComponentRef(c.parts[1:n]) == lhs.cref, own)
    return value isa IR.CALL ? IR.Equation[IR.RECORD_EQUATION(lhs, value, lhs.ty, length(lhs.ty.fields), source, IR.DEFAULT_ATTRIBUTES)] :
                               recordEquations(lhs, value, source)
  end
  local out = IR.Equation[]
  for (k, field) in enumerate(lhs.ty.fields)
    local l = fieldExpression(lhs, field.name, field.ty)::IR.CREF
    any(c::IR.ComponentRef -> c == l.cref, own) && continue
    local r = fieldByIndex(value, k)
    if field.ty isa IR.T_RECORD
      append!(out, recordValueEquations(l, r, own, source))
    elseif field.ty isa IR.T_ARRAY
      push!(out, IR.ARRAY_EQUATION(l, r, field.ty, source, IR.DEFAULT_ATTRIBUTES))
    else
      push!(out, IR.SCALAR_EQUATION(l, r, field.ty, source, IR.DEFAULT_ATTRIBUTES))
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
The asserts and `terminate()` of regular algorithm sections that do not run symbolically (`runsSymbolically`: those
reading `initial()`), their top-level ones and those of their if-statements under the branches' conditions (as
written): assert equations, checked at run time like those of equation sections, and the when-equations of a
`terminate()` (`guardedTerminate`). The others' are `realAssignmentEquations`', where they stand.
"""
function assertsOfAlgorithms(algorithms::Vector{IR.ALGORITHM})::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms
    isempty(alg.statements) && continue
    local body = IR.Statement[s for s in alg.statements if !(s isa IR.WHEN_STATEMENT)]
    runsSymbolically(body) && continue
    effectsUnder!(out, body, nothing)
  end
  return out
end

"""
Whether the body of an algorithm section (outside its when-statements) runs symbolically (`realAssignmentEquations`): it
assigns Real variables or has asserts or `terminate()` (read where they stand), and does not read `initial()`.
"""
runsSymbolically(body::Vector{IR.Statement})::Bool =
  (!isempty(realTargets(body)) || hasEffects(body)) && !any(statementMentionsInitial, body)

"Whether `stmts` have an assert or a `terminate()`, in if-, for- and while-statements too (not in when-statements)."
hasEffects(stmts::Vector{IR.Statement})::Bool = any(stmts) do s
  s isa Union{IR.ASSERT, IR.TERMINATE} ||
    (s isa IR.IF_STATEMENT && (any(b -> hasEffects(b.body), s.branches) || hasEffects(s.elseBody))) ||
    (s isa Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT} && hasEffects(s.body))
end

"The asserts and `terminate()` of `stmts` under `guard`, in if-statements under their branches' conditions too."
function effectsUnder!(out::Vector{IR.Equation}, stmts::Vector{IR.Statement}, guard::Union{Nothing, IR.Expression})::Nothing
  for s in stmts
    if s isa IR.ASSERT
      push!(out, IR.ASSERT_EQUATION(guard === nothing ? s.condition : orCondition(notCondition(guard), s.condition),
                                    s.message, s.level, s.source))
    elseif s isa IR.TERMINATE
      append!(out, guardedTerminate(terminateMarker(s), guard))
    elseif s isa IR.IF_STATEMENT
      local none::Union{Nothing, IR.Expression} = nothing
      for b in s.branches
        effectsUnder!(out, b.body, andCondition(guard, andCondition(none, b.condition)))
        none = andCondition(none, notCondition(b.condition))
      end
      effectsUnder!(out, s.elseBody, andCondition(guard, none))
    end
  end
  return nothing
end

"`when initial() then terminate(...)`: a `terminate()` of no condition (`lowerFlatModel`'s), for `guardedTerminate`."
terminateMarker(t::IR.TERMINATE)::IR.WHEN_EQUATION =
  IR.WHEN_EQUATION(IR.WhenBranch[IR.WhenBranch(IR.Expression[], true, false, IR.Statement[t])], IR.REGULAR_WHEN, t.source,
                   IR.DEFAULT_ATTRIBUTES)
