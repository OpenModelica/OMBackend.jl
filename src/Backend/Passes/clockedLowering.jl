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

#= The clocked partitions as when-clauses of the continuous system (the previous backend's
   ClockedLowering): a clocked discrete-time partition is a discrete-time system evaluated at
   its clock's ticks (MLS 16), as the body of `when sample(start, interval)` is at its events.
   One when-clause per base clock: it ticks on the base clock's grid (the gcd of its
   sub-clocks' factors and shifts), a tick counter decides which sub-partitions are active, and
   the sub-partitions' equations are its assignments, sorted by their data (a sub-sampled
   partition after the partition it samples, MLS 16.10). The partitions' variables are
   discrete; previous(x) is pre(x), hold(y) is y.

   A solver clock's partition is discretized first (ExplicitEuler, ImplicitEuler: discretizedPartition).
   Not lowered (unsupported): other solver methods, Boolean when-clauses in a partition (MLS 16.8.1),
   interval() of a sub-clock of an event clock, equations that are not assignments of one of
   the partition's variables (an algebraic loop at a tick). A periodic clock's first tick is at
   time 0, and `sample(u)` reads u at the tick, not its left limit. =#

"Whether `e` is a clock: a clock constructor, a clock-typed reference, a sub-clock operator of one."
function isClock(e::IR.Expression)::Bool
  e isa IR.CLOCK && return true
  e isa IR.CREF && return e.ty isa IR.T_CLOCK
  e isa IR.CALL && return builtinName(e) in SUB_CLOCK_OPERATORS && !isempty(e.arguments) && isClock(e.arguments[1])
  return false
end

"`noEvent(e)`: relations in clocked partitions generate no events (MLS 16.8.1)."
noEventCall(e::IR.Expression)::IR.Expression = builtinCall(:noEvent, IR.Expression[e], IR.typeOf(e))

"""
    ClockOperators(interval, firstTick)

The rewrite of a clocked partition's clock operators to the form of a when body: `previous(x)`
is `pre(x)`; `sample(u, c)`, `hold(y)` and the sub-clock operators of a partition their
argument (its latest value, MLS 16.5.2); `interval()` the partition's interval; `firstTick()`
whether this tick is its first; if-expressions `noEvent`.
"""
struct ClockOperators
  interval::Union{Nothing, IR.Expression}
  firstTick::Union{Nothing, IR.Expression}
end

function (r::ClockOperators)(e::IR.Expression)::IR.Expression
  e isa IR.IF_EXPRESSION && return noEventCall(e)
  e isa IR.CALL || return e
  local name = builtinName(e)
  local args = e.arguments
  if name === :noEvent && length(args) == 1 && args[1] isa IR.CALL && builtinName(args[1]) === :noEvent
    return args[1]
  elseif name === :previous && length(args) == 1
    return IR.CALL(:pre, args, e.ty, IR.BUILTIN_CALL)
  elseif name === :sample && length(args) == 2 && isClock(args[2])
    return args[1]
  elseif name === :hold && length(args) == 1
    return args[1]
  elseif name in SUB_CLOCK_OPERATORS && !isempty(args) && !isClock(args[1])
    return args[1]
  elseif name === :interval
    r.interval === nothing && unsupported("interval() of a sub-clock of an event clock", e)
    return r.interval
  elseif name === :firstTick
    r.firstTick === nothing && unsupported("firstTick() without a tick counter", e)
    return r.firstTick
  elseif name === :noClock
    unsupported("noClock (MLS 16.5.2)", e)
  end
  return e
end

"`e` with its clock operators rewritten."
rewrite(r::ClockOperators, e::IR.Expression)::IR.Expression = IR.mapExp(r, e)

"""
What an assignment of `target` keeps at a tick of its base clock that is not its own: `pre` of
the target, element by element for a tuple's (`(y, state) := random(pre(state))`) or an
array's.
"""
function previousValue(target::IR.Expression)::IR.Expression
  target isa IR.WILD && return IR.REAL(0.0)
  target isa IR.CREF && return preCall(target)
  target isa IR.TUPLE && return IR.TUPLE(IR.Expression[previousValue(e) for e in target.elements], target.ty)
  target isa IR.ARRAY && return IR.ARRAY(IR.Expression[previousValue(e) for e in target.elements], target.ty)
  unsupported("an assignment target of a sub-clocked partition", target)
end

"Whether the base clock's tick `counter` is a tick of a sub-clock: ticks `first`, `first + every`, ..."
function activeAt(counter::IR.Expression, first::Int, every::Int)::IR.Expression
  local reached = IR.RELATION(counter, IR.GREATEREQ, IR.REAL(Float64(first)), 0)
  every == 1 && return noEventCall(reached)
  local offset = IR.BINARY(counter, IR.SUB, IR.REAL(Float64(first)), IR.T_REAL())
  local remainder = builtinCall(:mod, IR.Expression[offset, IR.REAL(Float64(every))], IR.T_REAL())
  local onGrid = IR.RELATION(remainder, IR.EQUAL, IR.REAL(0.0), 0)
  return noEventCall(IR.LBINARY(reached, IR.AND, onGrid))
end

literalReal(e::IR.Expression)::Union{Nothing, Float64} =
  e isa IR.REAL ? e.value : e isa IR.INTEGER ? Float64(e.value) : nothing

"""
The interval of a periodic base clock as an expression (seconds): `intervalCounter/resolution`
of a rational clock, `interval` of a real one; `nothing` for an event clock.
"""
function baseInterval(k::IR.ClockKind)::Union{Nothing, IR.Expression}
  if k isa IR.INTEGER_CLOCK
    local (n, r) = (literalReal(k.intervalCounter), literalReal(k.resolution))
    (n === nothing || r === nothing) && unsupported("a rational clock whose interval counter or resolution is not a literal", k)
    return IR.REAL(n / r)
  end
  k isa IR.REAL_CLOCK && return k.interval
  k isa IR.BOOLEAN_CLOCK && return nothing
  unsupported("a base clock", k)
end

"`factor * e`, folded for a literal."
function scaled(e::IR.Expression, factor::Rational{Int})::IR.Expression
  factor == 1 && return e
  local v = literalReal(e)
  v === nothing || return IR.REAL(v * Float64(factor))
  return IR.BINARY(IR.REAL(Float64(factor)), IR.MUL, e, IR.T_REAL())
end

"""
A variable for each equation among its candidates, each variable for one equation (augmenting
paths, Kuhn); `nothing` for an equation left without one.
"""
function matchAssignments(candidates::Vector{Vector{Symbol}})::Vector{Union{Nothing, Symbol}}
  local owner = Dict{Symbol, Int}()
  local result = Vector{Union{Nothing, Symbol}}(nothing, length(candidates))
  local augment!
  augment! = (i, seen) -> begin
    for v in candidates[i]
      v in seen && continue
      push!(seen, v)
      if !haskey(owner, v) || augment!(owner[v], seen)
        owner[v] = i
        result[i] = v
        return true
      end
    end
    return false
  end
  foreach(i -> augment!(i, Set{Symbol}()), eachindex(candidates))
  return result
end

nameOf(e::IR.Expression)::Union{Nothing, Symbol} = e isa IR.CREF ? e.cref.name : nothing

"""
The variables an assignment target assigns: a variable, a tuple's or an array constructor's
elements (`(noise, {seedState[1], seedState[2]}) = random(...)`; `nothing` for an element that
is not a variable).
"""
function targetNames(e::IR.Expression)::Vector{Union{Nothing, Symbol}}
  e isa IR.CREF && return Union{Nothing, Symbol}[e.cref.name]
  e isa IR.ARRAY && return Union{Nothing, Symbol}[nameOf(x) for x in e.elements]
  if e isa IR.TUPLE
    local names = Union{Nothing, Symbol}[]
    for x in e.elements
      x isa IR.ARRAY ? append!(names, [nameOf(y) for y in x.elements]) : push!(names, nameOf(x))
    end
    return names
  end
  return Union{Nothing, Symbol}[nothing]
end

"""
    partitionStatements(partition, variables, r)

The partition's equations as assignments of its variables: the bodies of its clocked
when-clauses, its if-equations by variable, its tuple equations, its equations by a matching of
equations to the variables they can be written for; its initial when-equations apart.
"""
function partitionStatements(partition::IR.Partition, r::ClockOperators,
                             variables::IR.VariableSet)::Tuple{Vector{IR.Statement}, Vector{IR.Equation}}
  local names = Set{Symbol}(partition.unknowns)
  local statements = IR.Statement[]
  local initialWhens = IR.Equation[]
  local assigned = Set{Symbol}()
  local pending = IR.SCALAR_EQUATION[]
  for eq in partition.equations
    if eq isa IR.WHEN_EQUATION
      local b = eq.branches[1]
      if b.atInitial && isempty(b.triggers)
        push!(initialWhens, eq)
        continue
      end
      (length(b.triggers) == 1 && isClock(b.triggers[1])) ||
        unsupported("a when-clause with a Boolean condition in a clocked discrete-time partition (MLS 16.8.1)", eq)
      length(eq.branches) == 1 || unsupported("elsewhen in a clocked when-clause (MLS 16.6.1)", eq)
      for s in b.body
        if s isa IR.ASSIGNMENT
          push!(statements, IR.ASSIGNMENT(s.lhs, rewrite(r, s.rhs), s.source))
          foreach(n -> n === nothing || push!(assigned, n), targetNames(s.lhs))
        elseif s isa IR.ASSERT
          push!(statements, IR.ASSERT(rewrite(r, s.condition), rewrite(r, s.message), s.level, s.source))
        else
          unsupported("a statement of a clocked when-clause", s)
        end
      end
    elseif eq isa IR.IF_EQUATION
      append!(statements, ifAssignments(eq, r, assigned, variables))
    elseif eq isa IR.ASSERT_EQUATION
      push!(statements, IR.ASSERT(rewrite(r, eq.condition), rewrite(r, eq.message), eq.level, eq.source))
    elseif eq isa IR.SCALAR_EQUATION && (eq.lhs isa IR.TUPLE || eq.rhs isa IR.TUPLE)
      #= A tuple equation assigns all its elements: taken before the matching, which gave one
         of them to an alias equation next to it (`(y, s) = f(...); y = w`). =#
      local (target, value) = eq.lhs isa IR.TUPLE ? (eq.lhs, eq.rhs) : (eq.rhs, eq.lhs)
      local ns = targetNames(target)
      all(x -> x === nothing || x in names, ns) ||
        unsupported("a clocked tuple equation that assigns other variables than its partition's", eq)
      push!(statements, IR.ASSIGNMENT(target, rewrite(r, value), eq.source))
      foreach(n -> n === nothing || push!(assigned, n), ns)
    elseif eq isa IR.SCALAR_EQUATION
      push!(pending, eq)
    else
      unsupported("an equation of this kind in a clocked partition", eq)
    end
  end
  #= the equations: each written for one of the partition's variables =#
  local candidates = Vector{Symbol}[]
  for eq in pending
    local c = Symbol[]
    for side in (eq.lhs, eq.rhs)
      local n = nameOf(side)
      n !== nothing && n in names && !(n in assigned) && !(n in c) && push!(c, n)
    end
    push!(candidates, c)
  end
  local matched = matchAssignments(candidates)
  for (i, eq) in enumerate(pending)
    local v = matched[i]
    v === nothing && unsupported("a clocked equation that is not an assignment of one of its partition's variables", eq)
    local (target, value) = nameOf(eq.lhs) === v ? (eq.lhs, eq.rhs) : (eq.rhs, eq.lhs)
    push!(statements, IR.ASSIGNMENT(target, rewrite(r, value), eq.source))
  end
  #= each of the partition's variables assigned once at a tick =#
  local count = Dict{Symbol, Int}()
  for s in statements, t in statementTargets(s)
    count[t] = get(count, t, 0) + 1
  end
  local twice = sortedByName((v for (v, k) in count if k > 1), variables)
  isempty(twice) || unsupported("a variable of a clocked partition assigned twice at a tick", twice)
  local unassigned = sortedByName((v for v in names if !haskey(count, v)), variables)
  isempty(unassigned) || unsupported("a variable of a clocked partition no equation assigns", unassigned)
  return (statements, initialWhens)
end

"""
A clocked if-equation whose branches assign the same variables, as one assignment per variable
of an if-expression.
"""
function ifAssignments(eq::IR.IF_EQUATION, r::ClockOperators, assigned::Set{Symbol},
                       variables::IR.VariableSet)::Vector{IR.Statement}
  local branches = Dict{Symbol, Tuple{IR.Expression, IR.Expression}}[]
  for eqs in vcat([b.equations for b in eq.branches], [eq.elseEquations])
    local d = Dict{Symbol, Tuple{IR.Expression, IR.Expression}}()
    for e in eqs
      (e isa IR.SCALAR_EQUATION && nameOf(e.lhs) !== nothing) || unsupported("a clocked if-equation whose branches are not assignments", eq)
      d[nameOf(e.lhs)] = (e.lhs, e.rhs)
    end
    push!(branches, d)
  end
  local targets = keys(first(branches))
  all(b -> Set(keys(b)) == Set(targets), branches) || unsupported("a clocked if-equation whose branches assign different variables", eq)
  local out = IR.Statement[]
  for v in sortedByName(targets, variables)
    local value = rewrite(r, last(branches)[v][2])
    for k in length(eq.branches):-1:1
      local thenValue = rewrite(r, branches[k][v][2])
      value = noEventCall(IR.IF_EXPRESSION(rewrite(r, eq.branches[k].condition), thenValue, value, IR.typeOf(thenValue)))
    end
    push!(assigned, v)
    push!(out, IR.ASSIGNMENT(first(branches)[v][1], value, eq.source))
  end
  return out
end

"""
The tick number `counter` holds, rounded: the solver's interpolation at a continuous event leaves
a discrete state near its value, not at it (MSL CascadeControlledDrive.SubClocked).
"""
tickNumber(counter::IR.Expression)::IR.Expression =
  builtinCall(:floor, IR.Expression[IR.BINARY(counter, IR.ADD, IR.REAL(0.5), IR.T_REAL())], IR.T_REAL())

"A discrete Real variable of the clocked lowering with the start value `start`."
function clockVariable(name::Symbol, start::Float64)::Tuple{IR.Variable, IR.CREF}
  local attributes = IR.VariableAttributes(IR.REAL(start), IR.BOOLEAN(true), nothing, nothing, nothing, nothing, nothing,
                                           nothing, nothing)
  local v = IR.Variable(name, IR.T_REAL(), IR.DISCRETE(), nothing, attributes, IR.NO_DIRECTION, false, IR.NOT_CONNECTOR,
                        IR.NO_SOURCE, :lowerClockedPartitions)
  return (v, IR.CREF(IR.ComponentRef(name, IR.T_REAL()), IR.T_REAL()))
end

"The tick counter of base clock `b`: -1 before its first tick, the number of ticks since then."
tickCounter(b::Int)::Tuple{IR.Variable, IR.CREF} = clockVariable(Symbol("omjl_clock", b, "_tick"), -1.0)

"The time of base clock `b`'s latest tick, for the interval of an event clock."
previousTickTime(b::Int)::Tuple{IR.Variable, IR.CREF} = clockVariable(Symbol("omjl_clock", b, "_tickTime"), 0.0)

"A condition combining relations with and, or, not, without `sample()` or `terminal()`."
isCompoundExpression(c::IR.Expression)::Bool = !callsAnyOf(c, (:sample, :terminal)) && (c isa IR.LBINARY || c isa IR.LUNARY)

"""
The when-condition of base clock `b`, the event clock `Clock(c, startInterval)`: it ticks when
`edge(c)` becomes true (MLS 16.3). A compound `c` (`u < 0.5 or u > 0.8`) is a Boolean variable
defined by it, changed at its relations' crossings (as a when-condition's): on the expression
itself, the when fired at every crossing. With its new variables and equations.
"""
function eventClockCondition(b::Int, c::IR.Expression, known::Set{Symbol})::Tuple{IR.Expression, Vector{IR.Variable}, Vector{IR.Equation}}
  isCompoundExpression(c) || return (c, IR.Variable[], IR.Equation[])
  local name = Symbol("omjl_clock", b, "_condition")
  local v = IR.Variable(name, IR.T_BOOLEAN(), IR.DISCRETE(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION, false,
                        IR.NOT_CONNECTOR, IR.NO_SOURCE, :lowerClockedPartitions)
  local lhs = IR.CREF(IR.ComponentRef(name, IR.T_BOOLEAN()), IR.T_BOOLEAN())
  local definition = IR.SCALAR_EQUATION(lhs, c, IR.T_BOOLEAN(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
  local constants = (values = Dict{Symbol, IR.Expression}(), valueOrder = Symbol[], pre = Dict{Symbol, IR.Expression}(),
                     preOrder = Symbol[])
  local (lifted, names) = liftDiscreteEquations(IR.Equation[definition], known, Dict{Symbol, IR.Expression}(), constants,
                                                IR.VariableSet())
  #= without relations (`b1 or b2` of discrete Booleans) the condition stays, as a when's =#
  isempty(names) && return (c, IR.Variable[], IR.Equation[])
  return (lhs, IR.Variable[v], lifted)
end

"""
    discretizedPartition(p, method, interval, newVars) -> Partition

A clocked discretized continuous-time partition (a solver clock, MLS 16.8.2) as a discrete-time one: each
`der(x)` a discrete variable its equation assigns at the tick (pushed to `newVars`), and x integrated from its
previous tick by `method`, its start at the first tick (as omc): "ExplicitEuler" `x = previous(x) + interval() *
previous(der(x))`, "ImplicitEuler" `x = previous(x) + interval() * der(x)` (where der(x) reads x, an algebraic
loop at the tick: not supported). MSL Clocked's ClockedWithDiscretizedContinuousController (a PI),
ControlledMixingUnit (an inverse plant model).
"""
function discretizedPartition(p::IR.Partition, method::String, interval::Union{Nothing, IR.Expression},
                              newVars::Vector{IR.Variable})::IR.Partition
  method in ("ExplicitEuler", "ImplicitEuler") ||
    unsupported("a solver clock's method other than ExplicitEuler and ImplicitEuler", method)
  interval === nothing && unsupported("a solver clock of an event clock (no interval)", method)
  #= the states in the order of their first der(), each with its derivative variable =#
  local states = Symbol[]
  local derivatives = Dict{Symbol, IR.CREF}()
  local derivativeOf = e -> begin
    (e isa IR.CALL && builtinName(e) === :der && length(e.arguments) == 1) || return e
    local x = e.arguments[1]
    x isa IR.CREF || unsupported("der() of an expression in a clocked discretized partition", e)
    return get!(derivatives, x.cref.name) do
      push!(states, x.cref.name)
      IR.CREF(IR.ComponentRef(Symbol("omjl_der_", x.cref.name), IR.T_REAL()), IR.T_REAL())
    end
  end
  local equations = IR.Equation[p.equations...]
  foreach(eq -> IR.mapEquation!(derivativeOf, eq), equations)
  any(eq -> usesOperator(eq, :der), p.initialEquations) &&
    unsupported("der() in an initial equation of a clocked discretized partition", method)
  local firstTick = builtinCall(:firstTick, IR.Expression[], IR.T_BOOLEAN())
  local step = builtinCall(:interval, IR.Expression[], IR.T_REAL())
  for x in states
    local d = derivatives[x]
    local xRef = IR.CREF(IR.ComponentRef(x, IR.T_REAL()), IR.T_REAL())
    local previous = e -> builtinCall(:previous, IR.Expression[e], IR.T_REAL())
    local slope = method == "ExplicitEuler" ? previous(d) : d
    local stepped = IR.BINARY(previous(xRef), IR.ADD, IR.BINARY(step, IR.MUL, slope, IR.T_REAL()), IR.T_REAL())
    push!(equations, IR.SCALAR_EQUATION(xRef, IR.IF_EXPRESSION(firstTick, previous(xRef), stepped, IR.T_REAL()),
                                        IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    push!(newVars, IR.Variable(d.cref.name, IR.T_REAL(), IR.DISCRETE(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION, false,
                               IR.NOT_CONNECTOR, IR.NO_SOURCE, :lowerClockedPartitions))
  end
  return IR.Partition(p.index, p.association, vcat(p.unknowns, Symbol[derivatives[x].cref.name for x in states]),
                      equations, p.initialEquations)
end

"`hold(y)` as `y`: in the continuous system, the clocked variable's value at its last tick."
withoutHold(e::IR.Expression)::IR.Expression =
  IR.mapExp(x -> (x isa IR.CALL && builtinName(x) === :hold && length(x.arguments) == 1) ? x.arguments[1] : x, e)

"Whether the equation calls the clock operator `name`."
function usesOperator(eq::IR.Equation, name::Symbol)::Bool
  local found = Ref(false)
  IR.foreachExp(e -> (e isa IR.CALL && builtinName(e) === name) && (found[] = true), eq)
  return found[]
end

"""
    lowerBaseClock(b, base, parts, known)

The when-equation of base clock `b` and its sub-partitions, its new variables and its other new
equations (the partitions' initial when-equations, an event clock's condition's).
"""
function lowerBaseClock(b::Int, base::IR.ClockKind, parts::Vector{IR.Partition}, known::Set{Symbol},
                        variables::IR.VariableSet)::Tuple{IR.Equation, Vector{IR.Variable}, Vector{IR.Equation}}
  local subClocks = IR.SubClock[p.association.subClock for p in parts]
  any(c -> c.shift < 0, subClocks) && unsupported("a sub-clock that starts before its base clock (MLS 16.5.2)", IR.toString(base))
  local interval = baseInterval(base)
  local newVars = IR.Variable[]
  #= the partitions of a solver clock as discrete-time ones =#
  parts = IR.Partition[p.association.subClock.solverMethod === nothing ? p :
                       discretizedPartition(p, p.association.subClock.solverMethod, interval, newVars) for p in parts]
  #= the base clock ticks at the sub-clocks' grid; an event clock at its events, which its
     sub-clocks count in whole ticks =#
  local grid = if interval === nothing
    all(c -> isinteger(c.factor) && isinteger(c.shift), subClocks) ||
      unsupported("a sub-clock of an event clock that is not a whole number of its ticks (MLS 16.5.2)", IR.toString(base))
    1 // 1
  else
    reduce(gcd, vcat([c.factor for c in subClocks], [c.shift for c in subClocks]))
  end
  local extraEqs = IR.Equation[]
  local condition = if interval === nothing
    local (c, cVars, cEqs) = eventClockCondition(b, withoutHold(base.condition), known)
    append!(newVars, cVars)
    append!(extraEqs, cEqs)
    c
  else
    builtinCall(:sample, IR.Expression[IR.REAL(0.0), scaled(interval, grid)], IR.T_BOOLEAN())
  end
  local usesFirstTick = any(p -> any(eq -> usesOperator(eq, :firstTick), p.equations), parts)
  local usesEventInterval = interval === nothing && any(p -> any(eq -> usesOperator(eq, :interval), p.equations), parts)
  local needsCounter = usesFirstTick || usesEventInterval || any(c -> c.factor != grid || c.shift != 0, subClocks)
  local statements = IR.Statement[]
  local counter = nothing
  if needsCounter
    local (cv, ce) = tickCounter(b)
    push!(newVars, cv)
    counter = tickNumber(ce)
    push!(statements, IR.ASSIGNMENT(ce, IR.BINARY(tickNumber(preCall(ce)), IR.ADD, IR.REAL(1.0), IR.T_REAL()), IR.NO_SOURCE))
  end
  #= an event clock's interval (MLS 16.3): the time since its previous tick, its startInterval
     at the first tick =#
  local eventInterval = nothing
  if usesEventInterval
    local (tv, te) = previousTickTime(b)
    push!(newVars, tv)
    local time = IR.CREF(IR.ComponentRef(:time, IR.T_REAL()), IR.T_REAL())
    eventInterval = IR.IF_EXPRESSION(noEventCall(IR.RELATION(counter, IR.EQUAL, IR.REAL(0.0), 0)), base.startInterval,
                                     IR.BINARY(time, IR.SUB, preCall(te), IR.T_REAL()), IR.T_REAL())
    push!(statements, IR.ASSIGNMENT(te, time, IR.NO_SOURCE))
  end
  for p in parts
    local c = p.association.subClock
    local first_ = Int(c.shift / grid)
    local every = Int(c.factor / grid)
    local subInterval = interval !== nothing ? scaled(interval, c.factor) :
                        (c.factor == 1 && c.shift == 0) ? eventInterval : nothing
    local firstTick = counter === nothing ? nothing : noEventCall(IR.RELATION(counter, IR.EQUAL, IR.REAL(Float64(first_)), 0))
    local (stmts, inits) = partitionStatements(p, ClockOperators(subInterval, firstTick), variables)
    append!(extraEqs, inits)
    if first_ == 0 && every == 1
      append!(statements, stmts)
      continue
    end
    local active = activeAt(counter, first_, every)
    for s in stmts
      if s isa IR.ASSERT
        #= checked at the sub-clock's ticks =#
        push!(statements, IR.ASSERT(IR.LBINARY(IR.LUNARY(IR.NOT, active), IR.OR, s.condition), s.message, s.level, s.source))
        continue
      end
      s isa IR.ASSIGNMENT || unsupported("a statement of a sub-clocked partition that is not an assignment", s)
      (s.rhs isa IR.CALL && s.rhs.ty isa IR.T_TUPLE && any(t -> t isa IR.T_RECORD, s.rhs.ty.types)) &&
        unsupported("a record output in a sub-clocked partition", s)
      local guarded = IR.IF_EXPRESSION(active, s.rhs, previousValue(s.lhs), IR.typeOf(s.rhs))
      push!(statements, IR.ASSIGNMENT(s.lhs, s.lhs isa IR.CREF ? noEventCall(guarded) : guarded, s.source))
    end
  end
  local whenEq = IR.WHEN_EQUATION([IR.WhenBranch(IR.Expression[condition], false, false, sortStatements(statements))],
                                  IR.REGULAR_WHEN, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
  return (whenEq, newVars, extraEqs)
end

"The variables `e` reads at this tick: its references outside `pre()`."
reads(e::IR.Expression)::Set{Symbol} = Set{Symbol}(namesReadOutsidePre(e))

"The variables a statement assigns."
statementTargets(s::IR.Statement)::Vector{Symbol} =
  s isa IR.ASSIGNMENT ? Symbol[n for n in targetNames(s.lhs) if n !== nothing] : Symbol[]

"The variables a statement reads at this tick (not through `pre()`)."
statementReads(s::IR.Statement)::Set{Symbol} =
  s isa IR.ASSIGNMENT ? reads(s.rhs) : s isa IR.ASSERT ? reads(s.condition) : Set{Symbol}()

"""
The statements of a base clock's when-clause in the order of their data: a statement after the
statements assigning what it reads at this tick (`pre` reads the previous tick). Kahn's
algorithm, stable; statements in a cycle (an algebraic loop at a tick) are not supported.
"""
function sortStatements(statements::Vector{IR.Statement})::Vector{IR.Statement}
  local assignedBy = Dict{Symbol, Int}()
  for (i, s) in enumerate(statements), t in statementTargets(s)
    assignedBy[t] = i
  end
  for s in statements
    local own = intersect(Set(statementTargets(s)), statementReads(s))
    isempty(own) || unsupported("an algebraic loop in a clocked partition (a variable its own equation reads at the tick)",
                                (sort!(collect(own)), s))
  end
  local deps = [Set{Int}(assignedBy[v] for v in statementReads(s) if haskey(assignedBy, v) && assignedBy[v] != i)
                for (i, s) in enumerate(statements)]
  local done = falses(length(statements))
  local order = Int[]
  while length(order) < length(statements)
    local progressed = false
    for i in eachindex(statements)
      (done[i] || !all(d -> done[d], deps[i])) && continue
      push!(order, i)
      done[i] = true
      progressed = true
    end
    progressed || unsupported("an algebraic loop in a clocked partition (its equations at a tick)",
                              [statements[i] for i in eachindex(statements) if !done[i]])
  end
  return statements[order]
end

"""
    lowerClockedPartitions!(model)

The clocked partitions as when-equations of the continuous system (one per base clock), their
variables discrete and, with the lowering's new variables, after the others (as the previous
backend had them); `hold(y)` in the continuous system `y`. The model itself without them.
"""
function lowerClockedPartitions!(model::IR.Model)::IR.Model
  isempty(model.clocked) && return model
  local system = model.continuous[1]
  foreach(eq -> IR.mapEquation!(withoutHold, eq), system.equations)
  foreach(eq -> IR.mapEquation!(withoutHold, eq), system.initialEquations)
  local known = knownNames(model.variables)
  local partitionNames = Set{Symbol}(n for p in model.clocked for n in p.unknowns)
  local moved = IR.Variable[]
  for (b, base) in enumerate(model.baseClocks)
    local parts = IR.Partition[p for p in model.clocked if p.association.baseClock == b]
    isempty(parts) && continue
    local (whenEq, newVars, extraEqs) = lowerBaseClock(b, base, parts, known, model.variables)
    for p in parts
      for n in p.unknowns
        local v = model.variables[n]
        v.kind = IR.DISCRETE()
        push!(moved, v)
      end
      append!(system.initialEquations, p.initialEquations)
    end
    append!(moved, newVars)
    push!(system.equations, whenEq)
    append!(system.equations, extraEqs)
  end
  filter!(v -> !(v.name in partitionNames), model.variables)
  foreach(v -> push!(model.variables, v), moved)
  empty!(model.clocked)
  empty!(model.baseClocks)
  return model
end
