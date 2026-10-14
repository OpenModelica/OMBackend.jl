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


"""
    ClockedLowering

The clocked partitions of a model (ClockPartitioning) as when-clauses of the continuous
system: a clocked discrete-time partition is a discrete-time system evaluated at its clock's
ticks (MLS 16), as the body of `when sample(start, interval)` is at its events. One when-clause
per base clock: it ticks on the base clock's grid (the gcd of its sub-clocks' factors and
shifts), a tick counter decides which sub-partitions are active, and the sub-partitions'
equations are its assignments (the when-clause's sorting orders a sub-sampled partition after
the partition it samples, MLS 16.10: sub-clock partitions are synchronized at coinciding
ticks). The partitions' variables are discrete; previous(x) is pre(x) (the value at the previous
tick, the start value before the first), hold(y) is y (the value of the last tick).

Not lowered (UnsupportedLowering): clocked discretized continuous-time partitions (solver
clocks), Boolean when-clauses in a partition (MLS 16.8.1: not in a clocked discrete-time
partition), interval() of a sub-clock of an event clock, equations that are not assignments of
one of the partition's variables (an algebraic loop at a tick). Differences from MLS 16 left: a periodic
clock's first tick is at time 0 (the simulation's start time in MLS 16.3), and `sample(u)` reads
u at the tick, not its left limit: a u that changes at the tick's instant (another base clock, a
time event, a when sample() on the same grid) is read before or after its change, by the order of
the callbacks.
"""
module ClockedLowering

import Absyn
import DAE
import ..BDAE
import ..BDAEUtil
import ..BDAECreate
import ..FrontendUtil.Util
import ..ClockPartitioning
import OMBackend
using MetaModelica
using DataStructures: OrderedSet

const CP = ClockPartitioning

"""
    isClock(e::DAE.Exp) -> Bool

Whether `e` is a clock: a Clock constructor, a clock-typed variable, a sub-clock operator of one.
"""
function isClock(e::DAE.Exp)::Bool
  return @match e begin
    DAE.CLKCONST(__) => true
    DAE.CREF(ty = DAE.T_CLOCK(__)) => true
    DAE.CALL(path = p, expLst = args) => CP.builtinName(p) in CP.SUB_CLOCK_OPERATORS && !isempty(args) && isClock(listHead(args))
    _ => false
  end
end

"""
    ClockOperators

The rewrite of a clocked partition's clock operators to the form of a when-clause body:
previous(x) -> pre(x); sample(u, c) and sample(u) -> u; subSample, superSample, shiftSample,
backSample -> their argument, the latest value of the partition they sample (MLS 16.5.2);
interval() -> the partition's interval; firstTick() -> whether this tick is its first; hold(y)
-> y (the continuous side). Called by the bottom-up traversal.
"""
struct ClockOperators
  interval::Union{Nothing, DAE.Exp}
  firstTick::Union{Nothing, DAE.Exp}
end

function (r::ClockOperators)(e::DAE.Exp, arg::Nothing)::Tuple{DAE.Exp, Nothing}
  #= relations in a clocked partition make no events (MLS 16.8.1): its if-expressions are noEvent,
     lowered with ifelse (the arithmetic c*a + (1-c)*b gave NaN for a non-finite other branch) =#
  e isa DAE.IFEXP && return (noEvent(e), arg)
  e isa DAE.CALL || return (e, arg)
  local name = CP.builtinName(e.path)
  local args = DAE.Exp[a for a in e.expLst]
  if name == "noEvent" && length(args) == 1 && args[1] isa DAE.CALL && CP.builtinName(args[1].path) == "noEvent"
    return (args[1], arg)
  elseif name == "previous" && length(args) == 1
    return (DAE.CALL(Absyn.IDENT("pre"), e.expLst, e.attr), arg)
  elseif name == "sample" && length(args) == 2 && isClock(args[2])
    return (args[1], arg)
  elseif name == "hold" && length(args) == 1
    return (args[1], arg)
  elseif name in CP.SUB_CLOCK_OPERATORS && !isempty(args) && !isClock(args[1])
    return (args[1], arg)
  elseif name == "interval"
    r.interval === nothing && OMBackend.unsupported("interval() of a sub-clock of an event clock", e)
    return (r.interval, arg)
  elseif name == "firstTick"
    r.firstTick === nothing && OMBackend.unsupported("firstTick() without a tick counter", e)
    return (r.firstTick, arg)
  elseif name == "noClock"
    OMBackend.unsupported("noClock (MLS 16.5.2)", e)
  end
  return (e, arg)
end

"""
    rewrite(r::ClockOperators, e::DAE.Exp) -> DAE.Exp

`e` with its clock operators rewritten (ClockOperators).
"""
rewrite(r::ClockOperators, e::DAE.Exp)::DAE.Exp = first(Util.traverseExpBottomUp(e, r, nothing))

"""
    preCall(target::DAE.Exp) -> DAE.Exp

`pre(target)` of a variable reference, its call attributes by the variable's type.
"""
function preCall(target::DAE.Exp)::DAE.Exp
  local attr = @match target begin
    DAE.CREF(ty = DAE.T_INTEGER(__)) => DAE.callAttrBuiltinInteger
    DAE.CREF(ty = DAE.T_BOOL(__)) => DAE.callAttrBuiltinBool
    _ => DAE.callAttrBuiltinReal
  end
  return DAE.CALL(Absyn.IDENT("pre"), MetaModelica.list(target), attr)
end

realConst(x::Real)::DAE.Exp = DAE.RCONST(Float64(x))

"""
    previousValue(target::DAE.Exp) -> DAE.Exp

What an assignment of `target` keeps at a tick of its base clock that is not its own: `pre` of
the target, element by element for a tuple's (`(y, state) := random(pre(state))`) or an array's.
"""
function previousValue(target::DAE.Exp)::DAE.Exp
  return @match target begin
    DAE.CREF(componentRef = DAE.WILD(__)) => realConst(0)
    DAE.CREF(__) => preCall(target)
    DAE.TUPLE(PR = elements) => DAE.TUPLE(MetaModelica.list(DAE.Exp[previousValue(e) for e in elements]...))
    DAE.ARRAY(ty, scalar, elements) => DAE.ARRAY(ty, scalar, MetaModelica.list(DAE.Exp[previousValue(e) for e in elements]...))
    _ => OMBackend.unsupported("an assignment target of a sub-clocked partition", target)
  end
end

"""
    noEvent(e::DAE.Exp) -> DAE.Exp

`noEvent(e)`: the guards are evaluated at the ticks only, relations in clocked partitions
generate no events (MLS 16.8.1).
"""
noEvent(e::DAE.Exp)::DAE.Exp = DAE.CALL(Absyn.IDENT("noEvent"), MetaModelica.list(e), DAE.callAttrBuiltinBool)

"""
    activeAt(counter::DAE.Exp, first::Int, every::Int) -> DAE.Exp

Whether the base clock's tick `counter` is a tick of a sub-clock: ticks `first`, `first + every`, ...
"""
function activeAt(counter::DAE.Exp, first::Int, every::Int)::DAE.Exp
  local reached = DAE.RELATION(counter, DAE.GREATEREQ(DAE.T_REAL_DEFAULT), realConst(first), -1, NONE())
  every == 1 && return noEvent(reached)
  local offset = DAE.BINARY(counter, DAE.SUB(DAE.T_REAL_DEFAULT), realConst(first))
  local remainder = DAE.CALL(Absyn.IDENT("mod"), MetaModelica.list(offset, realConst(every)), DAE.callAttrBuiltinReal)
  local onGrid = DAE.RELATION(remainder, DAE.EQUAL(DAE.T_REAL_DEFAULT), realConst(0), -1, NONE())
  return noEvent(DAE.LBINARY(reached, DAE.AND(DAE.T_BOOL_DEFAULT), onGrid))
end

"""
    literalReal(e::DAE.Exp) -> Union{Nothing, Float64}
"""
literalReal(e::DAE.Exp)::Union{Nothing, Float64} = @match e begin
  DAE.RCONST(real = r) => Float64(r)
  DAE.ICONST(integer = i) => Float64(i)
  _ => nothing
end

"""
    baseInterval(k::DAE.ClockKind) -> Union{Nothing, DAE.Exp}

The interval of a periodic base clock as an expression (seconds): `intervalCounter/resolution`
of a rational clock, `interval` of a real one; `nothing` for an event clock.
"""
function baseInterval(k::DAE.ClockKind)::Union{Nothing, DAE.Exp}
  return @match k begin
    DAE.INTEGER_CLOCK(intervalCounter = n, resolution = r) => begin
      local (nv, rv) = (literalReal(n), literalReal(r))
      (nv === nothing || rv === nothing) &&
        OMBackend.unsupported("a rational clock whose interval counter or resolution is not a literal", k)
      realConst(nv / rv)
    end
    DAE.REAL_CLOCK(interval = e) => e
    DAE.BOOLEAN_CLOCK(__) => nothing
    _ => OMBackend.unsupported("a base clock", k)
  end
end

"""
    scaled(e::DAE.Exp, factor::Rational{Int}) -> DAE.Exp

`factor * e`, folded for a literal.
"""
function scaled(e::DAE.Exp, factor::Rational{Int})::DAE.Exp
  factor == 1 && return e
  local v = literalReal(e)
  v === nothing || return realConst(v * Float64(factor))
  return DAE.BINARY(realConst(Float64(factor)), DAE.MUL(DAE.T_REAL_DEFAULT), e)
end

"""
    matchAssignments(candidates::Vector{Vector{String}}) -> Vector{Union{Nothing, String}}

A variable for each equation among its candidates, each variable for one equation (augmenting
paths, Kuhn); `nothing` for an equation left without one.
"""
function matchAssignments(candidates::Vector{Vector{String}})::Vector{Union{Nothing, String}}
  local owner = Dict{String, Int}()
  local result = Vector{Union{Nothing, String}}(nothing, length(candidates))
  for i in eachindex(candidates)
    augment!(i, Set{String}(), candidates, owner, result)
  end
  return result
end

"""
    augment!(i, seen, candidates, owner, result) -> Bool

An augmenting path from equation `i`: a free variable, or one whose equation can take another.
"""
function augment!(i::Int, seen::Set{String}, candidates::Vector{Vector{String}}, owner::Dict{String, Int},
                  result::Vector{Union{Nothing, String}})::Bool
  for v in candidates[i]
    v in seen && continue
    push!(seen, v)
    if !haskey(owner, v) || augment!(owner[v], seen, candidates, owner, result)
      owner[v] = i
      result[i] = v
      return true
    end
  end
  return false
end

"""
    crefName(e::DAE.Exp) -> Union{Nothing, String}

The name of a variable reference; `nothing` for another expression.
"""
crefName(e::DAE.Exp)::Union{Nothing, String} = e isa DAE.CREF ? string(e.componentRef) : nothing

"""
    targetNames(e::DAE.Exp) -> Vector{Union{Nothing, String}}

The variables an assignment target assigns: a variable, a tuple's or an array constructor's
elements (`nothing` for an element that is not a variable).
"""
function targetNames(e::DAE.Exp)::Vector{Union{Nothing, String}}
  return @match e begin
    DAE.CREF(__) => Union{Nothing, String}[crefName(e)]
    DAE.TUPLE(__) => tupleNames(e)
    DAE.ARRAY(array = elements) => Union{Nothing, String}[crefName(x) for x in elements]
    _ => Union{Nothing, String}[nothing]
  end
end

"""
    tupleNames(e::DAE.Exp) -> Vector{Union{Nothing, String}}

The variables a tuple equation's left side assigns: its elements, an array constructor's
elements among them (`(noise, {seedState[1], seedState[2]}) = random(...)`).
"""
function tupleNames(e::DAE.Exp)::Vector{Union{Nothing, String}}
  local names = Union{Nothing, String}[]
  for x in e.PR
    if x isa DAE.ARRAY
      append!(names, [crefName(y) for y in x.array])
    else
      push!(names, crefName(x))
    end
  end
  return names
end

"""
    partitionStatements(part::BDAE.ClockedPartition, r::ClockOperators) -> (statements, initialWhens)

The partition's equations as assignments of its variables (when-clause statements): the bodies
of its clocked when-clauses, its if-equations by variable, its equations by a matching of
equations to the variables they can be written for; its `when initial()` equations apart.
"""
function partitionStatements(part::BDAE.ClockedPartition, r::ClockOperators)::Tuple{Vector{BDAE.WhenOperator}, Vector{BDAE.Equation}}
  local names = Set{String}(string(v.varName) for v in part.vars)
  local statements = BDAE.WhenOperator[]
  local initialWhens = BDAE.Equation[]
  local assigned = Set{String}()
  local pending = BDAE.EQUATION[]
  for eq in part.eqs
    if eq isa BDAE.WHEN_EQUATION
      isClock(eq.whenEquation.condition) ||
        OMBackend.unsupported("a when-clause with a Boolean condition in a clocked discrete-time partition (MLS 16.8.1)", eq)
      eq.whenEquation.elsewhenPart === nothing ||
        OMBackend.unsupported("elsewhen in a clocked when-clause (MLS 16.6.1)", eq)
      for stmt in eq.whenEquation.whenStmtLst
        if stmt isa BDAE.ASSIGN
          push!(statements, BDAE.ASSIGN(stmt.left, rewrite(r, stmt.right), stmt.source))
          foreach(n -> n === nothing || push!(assigned, n), targetNames(stmt.left))
        elseif stmt isa BDAE.ASSERT
          push!(statements, BDAE.ASSERT(rewrite(r, stmt.condition), rewrite(r, stmt.message), stmt.level, stmt.source))
        else
          OMBackend.unsupported("a statement of a clocked when-clause", stmt)
        end
      end
    elseif eq isa BDAE.INITIAL_WHEN_EQUATION
      push!(initialWhens, eq)
    elseif eq isa BDAE.IF_EQUATION
      append!(statements, ifAssignments(eq, r, assigned))
    elseif eq isa BDAE.ASSERT_EQUATION
      push!(statements, BDAE.ASSERT(rewrite(r, eq.condition), rewrite(r, eq.message), eq.level, eq.source))
    elseif eq isa BDAE.EQUATION && (eq.lhs isa DAE.TUPLE || eq.rhs isa DAE.TUPLE)
      #= a tuple equation assigns all its elements: they are taken before the matching, which
         gave one of them to an alias equation next to it (`(y, s) = f(...); y = w`) =#
      local (target, value) = eq.lhs isa DAE.TUPLE ? (eq.lhs, eq.rhs) : (eq.rhs, eq.lhs)
      local ns = tupleNames(target)
      all(x -> x === nothing || x in names, ns) ||
        OMBackend.unsupported("a clocked tuple equation that assigns other variables than its partition's", eq)
      push!(statements, BDAE.ASSIGN(target, rewrite(r, value), eq.source))
      foreach(n -> n === nothing || push!(assigned, n), ns)
    elseif eq isa BDAE.EQUATION
      push!(pending, eq)
    else
      OMBackend.unsupported("an equation of this kind in a clocked partition", eq)
    end
  end
  #= the equations: each written for one of the partition's variables =#
  local candidates = Vector{String}[]
  for eq in pending
    local c = String[]
    for side in (eq.lhs, eq.rhs)
      local n = crefName(side)
      n !== nothing && n in names && !(n in assigned) && push!(c, n)
    end
    push!(candidates, unique!(c))
  end
  local matched = matchAssignments(candidates)
  for (i, eq) in enumerate(pending)
    local v = matched[i]
    v === nothing && OMBackend.unsupported("a clocked equation that is not an assignment of one of its partition's variables", eq)
    local (target, value) = crefName(eq.lhs) == v ? (eq.lhs, eq.rhs) : (eq.rhs, eq.lhs)
    push!(statements, BDAE.ASSIGN(target, rewrite(r, value), eq.source))
  end
  #= each of the partition's variables assigned once at a tick =#
  local count = Dict{String, Int}()
  for s in statements, t in targets(s)
    count[t] = get(count, t, 0) + 1
  end
  local twice = sort!([v for (v, k) in count if k > 1])
  isempty(twice) || OMBackend.unsupported("a variable of a clocked partition assigned twice at a tick", twice)
  local unassigned = sort!([v for v in names if !haskey(count, v)])
  isempty(unassigned) || OMBackend.unsupported("a variable of a clocked partition no equation assigns", unassigned)
  return (statements, initialWhens)
end

"""
    ifAssignments(eq::BDAE.IF_EQUATION, r::ClockOperators, assigned::Set{String}) -> Vector{BDAE.WhenOperator}

A clocked if-equation whose branches assign the same variables, as one assignment per variable
of an if-expression.
"""
function ifAssignments(eq::BDAE.IF_EQUATION, r::ClockOperators, assigned::Set{String})::Vector{BDAE.WhenOperator}
  local branches = Vector{Dict{String, Tuple{DAE.Exp, DAE.Exp}}}()
  for eqs in vcat([b for b in eq.eqnstrue], [eq.eqnsfalse])
    local d = Dict{String, Tuple{DAE.Exp, DAE.Exp}}()
    for e in eqs
      (e isa BDAE.EQUATION && crefName(e.lhs) !== nothing) ||
        OMBackend.unsupported("a clocked if-equation whose branches are not assignments", eq)
      d[crefName(e.lhs)] = (e.lhs, e.rhs)
    end
    push!(branches, d)
  end
  local targets = keys(first(branches))
  all(b -> Set(keys(b)) == Set(targets), branches) ||
    OMBackend.unsupported("a clocked if-equation whose branches assign different variables", eq)
  local conditions = DAE.Exp[c for c in eq.conditions]
  local out = BDAE.WhenOperator[]
  for v in sort!(collect(targets))
    local value = rewrite(r, last(branches)[v][2])
    for k in length(conditions):-1:1
      value = noEvent(DAE.IFEXP(rewrite(r, conditions[k]), rewrite(r, branches[k][v][2]), value))
    end
    push!(assigned, v)
    push!(out, BDAE.ASSIGN(first(branches)[v][1], value, eq.source))
  end
  return out
end

"""
    tickNumber(counter::DAE.Exp) -> DAE.Exp

The tick number `counter` holds, rounded: the solver's interpolation at a continuous event leaves a
discrete state near its value, not at it (MSL CascadeControlledDrive.SubClocked's counter was
333.99999999999994 after KinematicPTP's switch at 2/3 s, and its sub-clock never ticked again).
"""
tickNumber(counter::DAE.Exp)::DAE.Exp =
  DAE.CALL(Absyn.IDENT("floor"), MetaModelica.list(DAE.BINARY(counter, DAE.ADD(DAE.T_REAL_DEFAULT), realConst(0.5))),
           DAE.callAttrBuiltinReal)

"""
    tickCounter(b::Int) -> (BDAE.VAR, DAE.Exp)

The tick counter of base clock `b`: a discrete Real, -1 before its first tick, the number of
ticks since then (read through tickNumber).
"""
function tickCounter(b::Int)::Tuple{BDAE.VAR, DAE.Exp}
  local cr = DAE.CREF_IDENT("omjl_clock$(b)_tick", DAE.T_REAL_DEFAULT, MetaModelica.nil)
  local v = BDAE.VAR(cr, BDAE.DISCRETE(), DAE.T_REAL_DEFAULT)
  v.values = SOME(DAE.makeRealAttribute(; start = SOME(-1.0), fixed = true))
  return (v, DAE.CREF(cr, DAE.T_REAL_DEFAULT))
end

"""
    eventClockCondition(b::Int, c::DAE.Exp, paramOrConst::OrderedSet{String}) -> (condition, new variables, new equations)

The when-condition of base clock `b`, the event clock `Clock(c, startInterval)`: it ticks when
edge(c) becomes true (MLS 16.3). A compound c (`u < 0.5 or u > 0.8`) is a Boolean variable defined by
it, changed at its relations' crossings (as a when-condition's, BDAECreate.booleanizeWhenRelations!):
on the expression itself, the when fired at every crossing, at u = 0.5 too, where c became false.
"""
function eventClockCondition(b::Int, c::DAE.Exp,
                             paramOrConst::OrderedSet{String})::Tuple{DAE.Exp, Vector{BDAE.VAR}, Vector{BDAE.Equation}}
  BDAECreate._isCompoundWhenCondition(c) || return (c, BDAE.VAR[], BDAE.Equation[])
  local cr = DAE.CREF_IDENT("omjl_clock$(b)_condition", DAE.T_BOOL_DEFAULT, MetaModelica.nil)
  local v = BDAE.VAR(cr, BDAE.DISCRETE(), DAE.T_BOOL_DEFAULT)
  local definition = BDAE.EQUATION(DAE.CREF(cr, DAE.T_BOOL_DEFAULT), c, DAE.emptyElementSource, BDAE.NO_ATTRIBUTES())
  local (lifted, names) = BDAECreate.synthesizeWhenEquationsFromDiscreteEquations(BDAE.Equation[definition], paramOrConst)
  #= without relations (`b1 or b2` of discrete Booleans) the condition stays, as a when's =#
  isempty(names) && return (c, BDAE.VAR[], BDAE.Equation[])
  return (DAE.CREF(cr, DAE.T_BOOL_DEFAULT), BDAE.VAR[v], lifted)
end

"""
    previousTickTime(b::Int) -> (BDAE.VAR, DAE.Exp)

The time of base clock `b`'s latest tick: a discrete Real, for the interval of an event clock.
"""
function previousTickTime(b::Int)::Tuple{BDAE.VAR, DAE.Exp}
  local cr = DAE.CREF_IDENT("omjl_clock$(b)_tickTime", DAE.T_REAL_DEFAULT, MetaModelica.nil)
  local v = BDAE.VAR(cr, BDAE.DISCRETE(), DAE.T_REAL_DEFAULT)
  v.values = SOME(DAE.makeRealAttribute(; start = SOME(0.0), fixed = true))
  return (v, DAE.CREF(cr, DAE.T_REAL_DEFAULT))
end

"""
    lowerBaseClock(b::Int, base::DAE.ClockKind, parts, paramOrConst) -> (WHEN_EQUATION, new variables, other new equations)

The when-clause of base clock `b` and its sub-partitions.
"""
function lowerBaseClock(b::Int, base::DAE.ClockKind, parts::Vector{BDAE.ClockedPartition},
                        paramOrConst::OrderedSet{String})::Tuple{BDAE.Equation, Vector{BDAE.VAR}, Vector{BDAE.Equation}}
  any(p -> p.subClock.solverMethod !== nothing, parts) &&
    OMBackend.unsupported("a clocked discretized continuous-time partition (a solver clock)", string(base))
  any(p -> p.subClock.shift < 0, parts) &&
    OMBackend.unsupported("a sub-clock that starts before its base clock (MLS 16.5.2)", string(base))
  local interval = baseInterval(base)
  #= the base clock ticks at the sub-clocks' grid; an event clock at its events, which its
     sub-clocks count in whole ticks (subSample(Clock(c), 2) ticked at every event) =#
  local grid = if interval === nothing
    all(p -> isinteger(p.subClock.factor) && isinteger(p.subClock.shift), parts) ||
      OMBackend.unsupported("a sub-clock of an event clock that is not a whole number of its ticks (MLS 16.5.2)", string(base))
    1 // 1
  else
    reduce(gcd, vcat([p.subClock.factor for p in parts], [p.subClock.shift for p in parts]))
  end
  local newVars = BDAE.VAR[]
  local extraEqs = BDAE.Equation[]
  local condition = if interval === nothing
    local (c, cVars, cEqs) = eventClockCondition(b, first(Util.traverseExpBottomUp(base.condition, HoldRewrite(), nothing)),
                                                 paramOrConst)
    append!(newVars, cVars)
    append!(extraEqs, cEqs)
    c
  else
    DAE.CALL(Absyn.IDENT("sample"), MetaModelica.list(realConst(0), scaled(interval, grid)), DAE.callAttrBuiltinBool)
  end
  local usesFirstTick = any(p -> any(eq -> usesOperator(eq, "firstTick"), p.eqs), parts)
  local usesEventInterval = interval === nothing && any(p -> any(eq -> usesOperator(eq, "interval"), p.eqs), parts)
  local needsCounter = usesFirstTick || usesEventInterval || any(p -> p.subClock.factor != grid || p.subClock.shift != 0, parts)
  local statements = BDAE.WhenOperator[]
  local counter = nothing
  if needsCounter
    local (cv, ce) = tickCounter(b)
    push!(newVars, cv)
    counter = tickNumber(ce)
    push!(statements, BDAE.ASSIGN(ce, DAE.BINARY(tickNumber(preCall(ce)), DAE.ADD(DAE.T_REAL_DEFAULT), realConst(1)),
                                  DAE.emptyElementSource))
  end
  #= an event clock's interval (MLS 16.3): the time since its previous tick, its startInterval at
     the first tick =#
  local eventInterval = nothing
  if usesEventInterval
    local (tv, te) = previousTickTime(b)
    push!(newVars, tv)
    local time = DAE.CREF(DAE.CREF_IDENT("time", DAE.T_REAL_DEFAULT, MetaModelica.nil), DAE.T_REAL_DEFAULT)
    eventInterval = DAE.IFEXP(noEvent(DAE.RELATION(counter, DAE.EQUAL(DAE.T_REAL_DEFAULT), realConst(0), -1, NONE())),
                              base.startInterval, DAE.BINARY(time, DAE.SUB(DAE.T_REAL_DEFAULT), preCall(te)))
    push!(statements, BDAE.ASSIGN(te, time, DAE.emptyElementSource))
  end
  for p in parts
    local first_ = Int(p.subClock.shift / grid)
    local every = Int(p.subClock.factor / grid)
    local subInterval = if interval !== nothing
      scaled(interval, p.subClock.factor)
    elseif p.subClock.factor == 1 && p.subClock.shift == 0
      eventInterval
    else
      nothing
    end
    local r = ClockOperators(subInterval,
                             counter === nothing ? nothing :
                               noEvent(DAE.RELATION(counter, DAE.EQUAL(DAE.T_REAL_DEFAULT), realConst(first_), -1, NONE())))
    local (stmts, inits) = partitionStatements(p, r)
    append!(extraEqs, inits)
    for v in p.vars
      v.varKind = BDAE.DISCRETE()
    end
    if first_ == 0 && every == 1
      append!(statements, stmts)
    else
      local active = activeAt(counter, first_, every)
      for s in stmts
        if s isa BDAE.ASSERT
          #= checked at the sub-clock's ticks =#
          push!(statements, BDAE.ASSERT(DAE.LBINARY(DAE.LUNARY(DAE.NOT(DAE.T_BOOL_DEFAULT), active), DAE.OR(DAE.T_BOOL_DEFAULT),
                                                    s.condition), s.message, s.level, s.source))
          continue
        end
        s isa BDAE.ASSIGN ||
          OMBackend.unsupported("a statement of a sub-clocked partition that is not an assignment", s)
        s.right isa DAE.CALL && s.right.attr.ty isa DAE.T_TUPLE && any(t -> t isa DAE.T_COMPLEX, s.right.attr.ty.types) &&
          OMBackend.unsupported("a record output in a sub-clocked partition", s)
        local guarded = DAE.IFEXP(active, s.right, previousValue(s.left))
        #= noEvent: lowered with ifelse; a tuple's if-expression is (MTK_CodeGenerationUtil) =#
        push!(statements, BDAE.ASSIGN(s.left, s.left isa DAE.CREF ? noEvent(guarded) : guarded, s.source))
      end
    end
  end
  statements = sortStatements(statements)
  local whenEq = BDAE.WHEN_EQUATION(1, BDAE.WHEN_STMTS(condition, MetaModelica.list(statements...), NONE()),
                                    DAE.emptyElementSource, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
  return (whenEq, newVars, extraEqs)
end

"""
    ReadsCollector

The variables an expression reads at this tick: its references outside `pre(...)`. Called by
the top-down traversal.
"""
struct ReadsCollector
  reads::Set{String}
end

function (c::ReadsCollector)(e::DAE.Exp, arg::Nothing)::Tuple{DAE.Exp, Bool, Nothing}
  if e isa DAE.CALL && CP.builtinName(e.path) == "pre"
    return (e, false, arg)
  elseif e isa DAE.CREF
    push!(c.reads, string(e.componentRef))
  end
  return (e, true, arg)
end

"""
    reads(e::DAE.Exp) -> Set{String}

The variables `e` reads at this tick: its references outside pre().
"""
function reads(e::DAE.Exp)::Set{String}
  local c = ReadsCollector(Set{String}())
  Util.traverseExpTopDown(e, c, nothing)
  return c.reads
end

"""
    targets(s::BDAE.WhenOperator) -> Vector{String}

The variables a statement assigns.
"""
targets(s::BDAE.WhenOperator)::Vector{String} = @match s begin
  BDAE.ASSIGN(left = t) => String[n for n in targetNames(t) if n !== nothing]
  _ => String[]
end

"""
    statementReads(s::BDAE.WhenOperator) -> Set{String}

The variables a statement reads at this tick (not through pre).
"""
statementReads(s::BDAE.WhenOperator)::Set{String} = @match s begin
  BDAE.ASSIGN(right = r) => reads(r)
  BDAE.ASSERT(condition = c) => reads(c)
  _ => Set{String}()
end

"""
    sortStatements(statements::Vector{BDAE.WhenOperator}) -> Vector{BDAE.WhenOperator}

The statements of a base clock's when-clause in the order of their data: a statement after the
statements assigning what it reads at this tick (`pre` reads the previous tick). A sub-sampled
partition reads the value of the partition it samples at a coinciding tick (MLS 16.10), a
partition's equations are solved in their causal order. The statements of the when-clauses of
the source were sorted when the BDAE was created; these are made after. Kahn's algorithm, stable;
statements in a cycle (an algebraic loop at a tick) are refused.
"""
function sortStatements(statements::Vector{BDAE.WhenOperator})::Vector{BDAE.WhenOperator}
  local assignedBy = Dict{String, Int}()
  for (i, s) in enumerate(statements), t in targets(s)
    assignedBy[t] = i
  end
  for s in statements
    local own = intersect(Set(targets(s)), statementReads(s))
    isempty(own) || OMBackend.unsupported("an algebraic loop in a clocked partition (a variable its own equation reads at the tick)",
                                          (sort!(collect(own)), s))
  end
  local deps = [Set{Int}(assignedBy[v] for v in statementReads(s) if haskey(assignedBy, v) && assignedBy[v] != i)
                for (i, s) in enumerate(statements)]
  local done = falses(length(statements))
  local order = Int[]
  while length(order) < length(statements)
    local progressed = false
    for i in eachindex(statements)
      done[i] && continue
      all(d -> done[d], deps[i]) || continue
      push!(order, i)
      done[i] = true
      progressed = true
    end
    progressed || OMBackend.unsupported("an algebraic loop in a clocked partition (its equations at a tick)",
                                        [statements[i] for i in eachindex(statements) if !done[i]])
  end
  return statements[order]
end

"""
    usesOperator(eq::BDAE.Equation, name::String) -> Bool
"""
function usesOperator(eq::BDAE.Equation, name::String)::Bool
  local finder = OperatorFinder(name, false)
  BDAEUtil.traverseEquationExpressions(eq, finder, nothing)
  return finder.found
end

mutable struct OperatorFinder
  const name::String
  found::Bool
end

function (f::OperatorFinder)(e::DAE.Exp, arg::Nothing)::Tuple{DAE.Exp, Bool, Nothing}
  e isa DAE.CALL && CP.builtinName(e.path) == f.name && (f.found = true)
  return (e, !f.found, arg)
end

"""
    HoldRewrite

hold(y) -> y in the continuous system (the clocked variable's value at its last tick, its start
value before the first). Called by the bottom-up traversal.
"""
struct HoldRewrite end
function (::HoldRewrite)(e::DAE.Exp, arg::Nothing)::Tuple{DAE.Exp, Nothing}
  if e isa DAE.CALL && CP.builtinName(e.path) == "hold" && length(e.expLst) == 1
    return (listHead(e.expLst), arg)
  end
  return (e, arg)
end

struct HoldTopDown end
function (::HoldTopDown)(e::DAE.Exp, arg::Nothing)::Tuple{DAE.Exp, Bool, Nothing}
  return (first(Util.traverseExpBottomUp(e, HoldRewrite(), nothing)), false, arg)
end

"""
    withoutHold(eq::BDAE.Equation) -> BDAE.Equation

`eq` with hold(y) -> y (HoldRewrite).
"""
withoutHold(eq::BDAE.Equation)::BDAE.Equation = first(BDAEUtil.traverseEquationExpressions(eq, HoldTopDown(), nothing))

"""
    lowerClockedPartitions(bdae::BDAE.BACKEND_DAE) -> BDAE.BACKEND_DAE

`bdae` with its clocked partitions as when-clauses of the continuous system (one per base clock)
and their variables discrete; `bdae` itself for a model without them.
"""
function lowerClockedPartitions(bdae::BDAE.BACKEND_DAE)::BDAE.BACKEND_DAE
  local cp = bdae.clocked
  isempty(cp.partitions) && return bdae
  local sys = only(bdae.eqs)
  local vars = BDAE.VAR[v for v in sys.orderedVars]
  local eqs = BDAE.Equation[withoutHold(eq) for eq in sys.orderedEqs]
  local initialEqs = BDAE.Equation[withoutHold(eq) for eq in sys.initialEqs]
  local paramOrConst = BDAECreate._collectParamOrConstNames(vcat(vars, bdae.shared.globalKnownVars, bdae.shared.localKnownVars))
  for (b, base) in enumerate(cp.baseClocks)
    local parts = BDAE.ClockedPartition[p for p in cp.partitions if p.baseClock == b]
    isempty(parts) && continue
    local (whenEq, newVars, extraEqs) = lowerBaseClock(b, base, parts, paramOrConst)
    for p in parts
      append!(vars, p.vars)
      append!(initialEqs, p.initialEqs)
    end
    append!(vars, newVars)
    push!(eqs, whenEq)
    append!(eqs, extraEqs)
  end
  local newSys = BDAE.EQSYSTEM(sys.name, vars, eqs, sys.simpleEquations, initialEqs)
  return BDAE.BACKEND_DAE(bdae.name, BDAE.EQSYSTEM[newSys], bdae.shared, BDAE.ClockedPartitioning())
end

end #= ClockedLowering =#
