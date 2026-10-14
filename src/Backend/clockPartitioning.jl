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
    ClockPartitioning

Clocked (synchronous) Modelica, MLS 16: the clocks of a model resolved and its clocked
partitions taken out of the continuous system, each with the sub-clock of its base clock
(omc's SynchronousFeatures: clock partitioning, sub-clock partitioning, clock inference).

A partition is a set of variables connected by equations. `sample(u, clock)` (continuous to
clocked) and `hold(y)` (clocked to continuous) separate partitions; `subSample`, `superSample`,
`shiftSample` and `backSample` separate the sub-partitions of one base partition and relate their
clocks. Clock variables (type Clock) and their equations are resolved to Clock constructors and
leave the model. What is left is the continuous system.
"""
module ClockPartitioning

import Absyn
import DAE
import ..BDAE
import ..BDAEUtil
import ..FrontendUtil.Util
import OMBackend
import ..@BACKEND_LOGGING
using MetaModelica

const SUB_CLOCK_OPERATORS = ("subSample", "superSample", "shiftSample", "backSample")
const CLOCKED_OPERATORS = ("previous", "interval", "firstTick")

"""
    ResolvedClock

A clock as its base clock (the Clock constructor it derives from; `nothing` for an inferred
clock) and its sub-clock of it: ticks at the base clock's `shift + k * factor` intervals.
"""
struct ResolvedClock
  base::Union{Nothing, DAE.ClockKind}
  factor::Rational{Int}
  shift::Rational{Int}
  solverMethod::Union{Nothing, String}
end

"""
    SubLink

`subSample`, `superSample`, `shiftSample` or `backSample` in an equation: its operator, its
other arguments and the variables of its first argument (the sub-partition it samples).
"""
struct SubLink
  op::String
  args::Vector{DAE.Exp}
  vars::Set{String}
end

"""
    EquationScan

What partitioning needs of an equation: its own variables (the partition it belongs to), the
variables it samples (`sample`'s first argument, continuous) and holds (`hold`'s argument, a
clocked partition), its sub-clock operators, the explicit clocks of its `sample`s, whether it
samples with an inferred clock, uses previous/interval/firstTick, or is about clocks itself
(a clock variable or Clock constructor outside `sample`). Called by the expression traversal.
"""
mutable struct EquationScan
  const unknowns::Set{String}
  const clockVars::Set{String}
  own::Set{String}
  boundary::Union{Nothing, Set{String}}
  sampled::Set{String}
  held::Set{String}
  links::Vector{SubLink}
  clocks::Vector{DAE.Exp}
  inferred::Bool
  clockedOperators::Bool
  clockExpressions::Bool
end
EquationScan(unknowns::Set{String}, clockVars::Set{String}) =
  EquationScan(unknowns, clockVars, Set{String}(), nothing, Set{String}(), Set{String}(), SubLink[],
               DAE.Exp[], false, false, false)

"""
    builtinName(path::Absyn.Path) -> String

The name of a call of a builtin (`hold`, `subSample`) or of the frontend's clocked `sample`
(`OMC_CLOCK.sample`); "" for another function.
"""
function builtinName(path::Absyn.Path)::String
  return @match path begin
    Absyn.IDENT(name = n) => n
    Absyn.QUALIFIED(name = "OMC_CLOCK", path = Absyn.IDENT(name = n)) => n
    _ => ""
  end
end

"""
    isClockExpression(e::DAE.Exp, clockVars::Set{String}) -> Bool

Whether `e` is a clock: a Clock constructor, a clock variable or a sub-clock operator of a clock.
"""
function isClockExpression(e::DAE.Exp, clockVars::Set{String})::Bool
  return @match e begin
    DAE.CLKCONST(__) => true
    DAE.CREF(componentRef = cr) => string(cr) in clockVars
    DAE.CALL(path = p, expLst = args) =>
      builtinName(p) in SUB_CLOCK_OPERATORS && !isempty(args) && isClockExpression(listHead(args), clockVars)
    _ => false
  end
end

function (s::EquationScan)(e::DAE.Exp, arg::Nothing)::Tuple{DAE.Exp, Bool, Nothing}
  return @match e begin
    DAE.CREF(componentRef = cr) => begin
      local name = string(cr)
      if name in s.clockVars
        s.boundary === nothing && (s.clockExpressions = true)
      elseif name in s.unknowns
        push!(something(s.boundary, s.own), name)
      end
      (e, true, arg)
    end
    DAE.CLKCONST(__) => begin
      s.boundary === nothing && (s.clockExpressions = true)
      (e, false, arg)
    end
    DAE.CALL(path = p, expLst = args) => scanCall!(s, e, builtinName(p), DAE.Exp[a for a in args])
    _ => (e, true, arg)
  end
end

"""
    collectInto!(s::EquationScan, target::Set{String}, e::DAE.Exp) -> Nothing

The variables of `e`, a boundary argument (`sample`'s, `hold`'s, a sub-clock operator's), into
`target` instead of the equation's own variables.
"""
function collectInto!(s::EquationScan, target::Set{String}, e::DAE.Exp)::Nothing
  local previous = s.boundary
  s.boundary = target
  Util.traverseExpTopDown(e, s, nothing)
  s.boundary = previous
  return nothing
end

function scanCall!(s::EquationScan, e::DAE.Exp, name::String, args::Vector{DAE.Exp})::Tuple{DAE.Exp, Bool, Nothing}
  s.boundary === nothing || return (e, true, nothing)
  if name == "sample" && length(args) == 2 && isClockExpression(args[2], s.clockVars)
    collectInto!(s, s.sampled, args[1])
    local clk = args[2]
    if clk isa DAE.CLKCONST && clk.clk isa DAE.INFERRED_CLOCK
      s.inferred = true
    else
      push!(s.clocks, clk)
    end
    return (e, false, nothing)
  elseif name == "hold" && length(args) == 1
    collectInto!(s, s.held, args[1])
    return (e, false, nothing)
  elseif name in SUB_CLOCK_OPERATORS && !isempty(args) && !isClockExpression(args[1], s.clockVars)
    local vars = Set{String}()
    collectInto!(s, vars, args[1])
    push!(s.links, SubLink(name, args[2:end], vars))
    return (e, false, nothing)
  elseif name in CLOCKED_OPERATORS
    s.clockedOperators = true
  elseif name == "noClock"
    OMBackend.unsupported("noClock (MLS 16.5.2)", e)
  end
  return (e, true, nothing)
end

"""
    scanEquation(eq::BDAE.Equation, unknowns::Set{String}, clockVars::Set{String}) -> EquationScan
"""
function scanEquation(eq::BDAE.Equation, unknowns::Set{String}, clockVars::Set{String})::EquationScan
  local s = EquationScan(unknowns, clockVars)
  #= A clocked when-clause (`when clock then y = u; end when`, MLS 16.6.1: MSL's AssignClock): its
     body is clocked by the clock, as `sample(·, clock)` =#
  if eq isa BDAE.WHEN_EQUATION && isClockExpression(eq.whenEquation.condition, clockVars)
    local clk = eq.whenEquation.condition
    clk isa DAE.CLKCONST && clk.clk isa DAE.INFERRED_CLOCK ? (s.inferred = true) : push!(s.clocks, clk)
    local body = eq
    @assign body.whenEquation.condition = DAE.BCONST(true)
    BDAEUtil.traverseEquationExpressions(body, s, nothing)
    return s
  end
  BDAEUtil.traverseEquationExpressions(eq, s, nothing)
  return s
end

isClocked(s::EquationScan)::Bool = s.inferred || !isempty(s.clocks) || s.clockedOperators || !isempty(s.links)

"""
    UnionFind

Disjoint sets of names (variables): `find!` gives a set's representative.
"""
struct UnionFind
  parent::Dict{String, String}
end
UnionFind() = UnionFind(Dict{String, String}())

function find!(u::UnionFind, x::String)::String
  local p = get!(u.parent, x, x)
  p == x && return x
  local r = find!(u, p)
  u.parent[x] = r
  return r
end

function unite!(u::UnionFind, a::String, b::String)::Nothing
  local (ra, rb) = (find!(u, a), find!(u, b))
  ra == rb || (u.parent[rb] = ra)
  return nothing
end

"""
    integerArgument(e::DAE.Exp, parameters::Dict{String, BDAE.VAR}) -> Int

A sub-clock operator's factor or counter: a literal, or a parameter bound to one.
"""
function integerArgument(e::DAE.Exp, parameters::Dict{String, BDAE.VAR})::Int
  return @match e begin
    DAE.ICONST(integer = i) => i
    DAE.RCONST(real = r) where {isinteger(r)} => Int(r)
    DAE.UNARY(operator = DAE.UMINUS(__), exp = x) => -integerArgument(x, parameters)
    DAE.CREF(componentRef = cr) where {haskey(parameters, string(cr)) && parameters[string(cr)].bindExp isa SOME} =>
      integerArgument(parameters[string(cr)].bindExp.data, parameters)
    _ => OMBackend.unsupported("a sub-clock factor or counter that is not an Integer literal or parameter", e)
  end
end

"""
    resolution(args::Vector{DAE.Exp}, parameters::Dict{String, BDAE.VAR}) -> (Int, Int)

`shiftSample`'s and `backSample`'s counter and resolution (1 when not given).
"""
function resolution(args::Vector{DAE.Exp}, parameters::Dict{String, BDAE.VAR})::Tuple{Int, Int}
  isempty(args) && OMBackend.unsupported("shiftSample or backSample without a counter", args)
  local counter = integerArgument(args[1], parameters)
  local res = length(args) >= 2 ? integerArgument(args[2], parameters) : 1
  res > 0 || OMBackend.unsupported("a shiftSample or backSample resolution below 1", res)
  return (counter, res)
end

"""
    forward(op::String, args::Vector{DAE.Exp}, c::ResolvedClock, parameters) -> Union{Nothing, ResolvedClock}

The clock of `op(u, args...)` for `u` on clock `c` (MLS 16.5.2); `nothing` for an inferred factor
(`subSample(u)`, `superSample(u)`: factor 0).
"""
function forward(op::String, args::Vector{DAE.Exp}, c::ResolvedClock, parameters::Dict{String, BDAE.VAR})::Union{Nothing, ResolvedClock}
  if op == "subSample" || op == "superSample"
    local n = isempty(args) ? 0 : integerArgument(args[1], parameters)
    n == 0 && return nothing
    n > 0 || OMBackend.unsupported("a negative $(op) factor", n)
    local factor = op == "subSample" ? c.factor * n : c.factor // n
    return ResolvedClock(c.base, factor, c.shift, c.solverMethod)
  end
  local (counter, res) = resolution(args, parameters)
  local delta = c.factor * (counter // res)
  return ResolvedClock(c.base, c.factor, op == "shiftSample" ? c.shift + delta : c.shift - delta, c.solverMethod)
end

"""
    backward(op::String, args::Vector{DAE.Exp}, c::ResolvedClock, parameters) -> Union{Nothing, ResolvedClock}

The clock of `u` in `y = op(u, args...)` for `y` on clock `c`: `forward`'s inverse.
"""
function backward(op::String, args::Vector{DAE.Exp}, c::ResolvedClock, parameters::Dict{String, BDAE.VAR})::Union{Nothing, ResolvedClock}
  if op == "subSample" || op == "superSample"
    local n = isempty(args) ? 0 : integerArgument(args[1], parameters)
    n == 0 && return nothing
    local factor = op == "subSample" ? c.factor // n : c.factor * n
    return ResolvedClock(c.base, factor, c.shift, c.solverMethod)
  end
  local (counter, res) = resolution(args, parameters)
  local delta = c.factor * (counter // res)
  return ResolvedClock(c.base, c.factor, op == "shiftSample" ? c.shift - delta : c.shift + delta, c.solverMethod)
end

sameClock(a::ResolvedClock, b::ResolvedClock)::Bool =
  a.base === b.base && a.factor == b.factor && a.shift == b.shift && a.solverMethod == b.solverMethod

"""
    ClockContext

The clock variables' definitions (by the representative of their alias set) and what resolving
them needs.
"""
struct ClockContext
  aliases::UnionFind
  definitions::Dict{String, DAE.Exp}
  resolved::Dict{String, ResolvedClock}
  visiting::Set{String}
  bases::Vector{DAE.ClockKind}
  parameters::Dict{String, BDAE.VAR}
  rebased::IdDict{DAE.ClockKind, Tuple{DAE.ClockKind, Rational{Int}}}
end

"""
    literalInteger(e::DAE.Exp, parameters::Dict{String, BDAE.VAR}) -> Union{Nothing, Int}

`e` as an Integer when it is a literal or a parameter bound to one; `nothing` otherwise.
"""
function literalInteger(e::DAE.Exp, parameters::Dict{String, BDAE.VAR})::Union{Nothing, Int}
  return @match e begin
    DAE.ICONST(integer = i) => i
    DAE.RCONST(real = r) where {isinteger(r)} => Int(r)
    DAE.CREF(componentRef = cr) where {haskey(parameters, string(cr)) && parameters[string(cr)].bindExp isa SOME} =>
      literalInteger(parameters[string(cr)].bindExp.data, parameters)
    _ => nothing
  end
end

"""
    rationalInterval(k::DAE.ClockKind, parameters) -> Union{Nothing, Rational{Int}}

The interval of a rational clock `Clock(intervalCounter, resolution)` in seconds; `nothing` for
another clock or counters that are not Integer literals or parameters.
"""
function rationalInterval(k::DAE.ClockKind, parameters::Dict{String, BDAE.VAR})::Union{Nothing, Rational{Int}}
  k isa DAE.INTEGER_CLOCK || return nothing
  local (n, r) = (literalInteger(k.intervalCounter, parameters), literalInteger(k.resolution, parameters))
  (n === nothing || r === nothing || r <= 0) && return nothing
  return n // r
end

"""
    canonical(ctx::ClockContext, c::ResolvedClock) -> ResolvedClock

`c` on its base clock's representative: a base clock rebased on another (rebase!) counts in the
other's intervals.
"""
function canonical(ctx::ClockContext, c::ResolvedClock)::ResolvedClock
  c.base === nothing && return c
  local (b, ratio) = (c.base, 1 // 1)
  while haskey(ctx.rebased, b)
    local (next, r) = ctx.rebased[b]
    (b, ratio) = (next, ratio * r)
  end
  b === c.base && return c
  return ResolvedClock(b, c.factor * ratio, c.shift * ratio, c.solverMethod)
end

"""
    rebase!(ctx::ClockContext, a::DAE.ClockKind, b::DAE.ClockKind) -> Bool

Two base clocks met in one base partition: two rational clocks are related, `b` then counts in
intervals of `a` (MLS 16.7.4: the clocks of a base partition must be compatible; MSL
CascadeControlledDrive.AbsoluteClocks). Whether they could be related.
"""
function rebase!(ctx::ClockContext, a::DAE.ClockKind, b::DAE.ClockKind)::Bool
  a === b && return true
  local (ia, ib) = (rationalInterval(a, ctx.parameters), rationalInterval(b, ctx.parameters))
  (ia === nothing || ib === nothing) && return false
  ctx.rebased[b] = (a, ib / ia)
  return true
end

"""
    baseClock!(ctx::ClockContext, k::DAE.ClockKind) -> DAE.ClockKind

The base clock of Clock constructor `k`: an equal one met before (two `Clock(0.1)` are one clock).
"""
function baseClock!(ctx::ClockContext, k::DAE.ClockKind)::DAE.ClockKind
  for b in ctx.bases
    (b === k || isequal(b, k)) && return b
  end
  push!(ctx.bases, k)
  return k
end

"""
    resolveClock(ctx::ClockContext, e::DAE.Exp) -> ResolvedClock

A clock expression as its base clock and sub-clock: a Clock constructor, a clock variable (its
definition), a sub-clock operator of a clock, `Clock(c, solverMethod)`.
"""
function resolveClock(ctx::ClockContext, e::DAE.Exp)::ResolvedClock
  return @match e begin
    DAE.CLKCONST(clk = DAE.INFERRED_CLOCK(__)) => ResolvedClock(nothing, 1 // 1, 0 // 1, nothing)
    DAE.CLKCONST(clk = DAE.SOLVER_CLOCK(c = c, solverMethod = m)) => begin
      local r = resolveClock(ctx, c)
      m isa DAE.SCONST || OMBackend.unsupported("a solver clock's method that is not a String literal", m)
      ResolvedClock(r.base, r.factor, r.shift, m.string)
    end
    DAE.CLKCONST(clk = k) => ResolvedClock(baseClock!(ctx, k), 1 // 1, 0 // 1, nothing)
    DAE.CREF(componentRef = cr) => resolveClockVariable(ctx, string(cr))
    DAE.CALL(path = p, expLst = args) where {builtinName(p) in SUB_CLOCK_OPERATORS && !isempty(args)} => begin
      local a = DAE.Exp[x for x in args]
      local r = resolveClock(ctx, a[1])
      local c = forward(builtinName(p), a[2:end], r, ctx.parameters)
      c === nothing && OMBackend.unsupported("a clock sub-sampled by an inferred factor", e)
      c
    end
    _ => OMBackend.unsupported("a clock expression", e)
  end
end

function resolveClockVariable(ctx::ClockContext, name::String)::ResolvedClock
  local root = find!(ctx.aliases, name)
  haskey(ctx.resolved, root) && return ctx.resolved[root]
  root in ctx.visiting && OMBackend.unsupported("a clock defined by itself", name)
  haskey(ctx.definitions, root) || OMBackend.unsupported("a clock variable without a definition", name)
  push!(ctx.visiting, root)
  local r = resolveClock(ctx, ctx.definitions[root])
  delete!(ctx.visiting, root)
  ctx.resolved[root] = r
  return r
end

"""
    clockEquations!(ctx::ClockContext, eqs, scans, clockVars, vars) -> Vector{Bool}

The clock equations and the clock variables' bindings (`Clock c = Clock(0.1)`: aliases of clock
variables, a clock variable's definition) into `ctx`; which equations are clock equations.
"""
function clockEquations!(ctx::ClockContext, eqs::Vector{BDAE.Equation}, scans::Vector{EquationScan},
                         clockVars::Set{String}, vars::Vector{BDAE.VAR})::Vector{Bool}
  local isClockEq = falses(length(eqs))
  local defs = Tuple{String, DAE.Exp}[]
  for v in vars
    (v.varType isa DAE.T_CLOCK && v.bindExp isa SOME) || continue
    local name = string(v.varName)
    local e = v.bindExp.data
    if e isa DAE.CREF && string(e.componentRef) in clockVars
      unite!(ctx.aliases, name, string(e.componentRef))
    else
      push!(defs, (name, e))
    end
  end
  for (i, (eq, s)) in enumerate(zip(eqs, scans))
    s.clockExpressions && !isClocked(s) && isempty(s.held) || continue
    isempty(s.own) || OMBackend.unsupported("an equation of clocks and variables", eq)
    isClockEq[i] = true
    local (lhs, rhs) = @match eq begin
      BDAE.EQUATION(lhs = l, rhs = r) => (l, r)
      _ => OMBackend.unsupported("a clock equation that is not an equality", eq)
    end
    local lname = lhs isa DAE.CREF && string(lhs.componentRef) in clockVars ? string(lhs.componentRef) : nothing
    local rname = rhs isa DAE.CREF && string(rhs.componentRef) in clockVars ? string(rhs.componentRef) : nothing
    if lname !== nothing && rname !== nothing
      unite!(ctx.aliases, lname, rname)
    elseif lname !== nothing
      push!(defs, (lname, rhs))
    elseif rname !== nothing
      push!(defs, (rname, lhs))
    else
      OMBackend.unsupported("a clock equation without a clock variable", eq)
    end
  end
  for (name, e) in defs
    local root = find!(ctx.aliases, name)
    local previous = get(ctx.definitions, root, nothing)
    previous === nothing || previous === e || isequal(previous, e) ||
      OMBackend.unsupported("a clock with two definitions", name)
    ctx.definitions[root] = e
  end
  return isClockEq
end

"""
    isKnown(v::BDAE.VAR) -> Bool

Whether `v` is a parameter, a constant or an external object: not a partition's variable.
"""
isKnown(v::BDAE.VAR)::Bool = v.varKind isa Union{BDAE.PARAM, BDAE.CONST, BDAE.EXTOBJ}

"""
    subPartitions(scans::Vector{EquationScan}, isClockEq::Vector{Bool}, unknowns::Set{String}) -> (sub, base, representative)

The partitions: variables connected by an equation's own variables (`sub`); a base partition also
by its sub-clock operators (`base`). `representative[i]` is a variable of equation i's
partition, `nothing` for a clock equation or one without variables.
"""
function subPartitions(scans::Vector{EquationScan}, isClockEq::Vector{Bool},
                       unknowns::Set{String})::Tuple{UnionFind, UnionFind, Vector{Union{Nothing, String}}}
  local sub = UnionFind()
  local base = UnionFind()
  foreach(v -> (find!(sub, v); find!(base, v)), unknowns)
  local representative = Vector{Union{Nothing, String}}(nothing, length(scans))
  for (i, s) in enumerate(scans)
    (isClockEq[i] || isempty(s.own)) && continue
    local own = sort!(collect(s.own))
    for v in own
      unite!(sub, own[1], v)
      unite!(base, own[1], v)
    end
    for link in s.links, v in link.vars
      unite!(base, own[1], v)
    end
    representative[i] = own[1]
  end
  return (sub, base, representative)
end

"""
    clockedSubPartitions(scans::Vector{EquationScan}, representative::Vector{Union{Nothing, String}}, sub::UnionFind) -> Set{String}

The clocked sub-partitions, by their representatives in `sub`: an equation's that samples, uses
previous, interval or firstTick or a sub-clock operator; the variables held or sub-sampled.
"""
function clockedSubPartitions(scans::Vector{EquationScan}, representative::Vector{Union{Nothing, String}},
                              sub::UnionFind)::Set{String}
  local clocked = Set{String}()
  for (i, s) in enumerate(scans)
    isClocked(s) && representative[i] !== nothing && push!(clocked, find!(sub, representative[i]))
    foreach(v -> push!(clocked, find!(sub, v)), s.held)
    for link in s.links, v in link.vars
      push!(clocked, find!(sub, v))
    end
  end
  return clocked
end

"""
    clockedVariables(variables::Vector{BDAE.VAR}, eqs::Vector{BDAE.Equation}) -> Set{String}

The variables of the clocked partitions (MLS 16.7), before the clocks are resolved; empty for a
model without clocks. For the discrete synthesis: a clocked equation is evaluated at its
clock's ticks, not at its relations' changes.
"""
function clockedVariables(variables::Vector{BDAE.VAR}, eqs::Vector{BDAE.Equation})::Set{String}
  local clockVars = Set{String}(string(v.varName) for v in variables if v.varType isa DAE.T_CLOCK)
  local unknowns = Set{String}(string(v.varName) for v in variables if !isKnown(v) && !(v.varType isa DAE.T_CLOCK))
  local scans = EquationScan[scanEquation(eq, unknowns, clockVars) for eq in eqs]
  (isempty(clockVars) && !any(s -> isClocked(s) || !isempty(s.held), scans)) && return Set{String}()
  local (sub, _, representative) = subPartitions(scans, fill(false, length(eqs)), unknowns)
  local clocked = clockedSubPartitions(scans, representative, sub)
  return Set{String}(v for v in unknowns if find!(sub, v) in clocked)
end

"""
    partitionClocks(bdae::BDAE.BACKEND_DAE) -> BDAE.BACKEND_DAE

`bdae` without its clock variables, clock equations and clocked partitions, which are in its
`clocked` field (BDAE.ClockedPartitioning); `bdae` itself for a model without clocks. Run on the
lowered model, before the passes of the continuous system.
"""
function partitionClocks(bdae::BDAE.BACKEND_DAE)::BDAE.BACKEND_DAE
  local sys = first(bdae.eqs)
  local clockVars = Set{String}(string(v.varName) for v in sys.orderedVars if v.varType isa DAE.T_CLOCK)
  local unknowns = Set{String}(string(v.varName) for v in sys.orderedVars if !isKnown(v) && !(v.varType isa DAE.T_CLOCK))
  local eqs = BDAE.Equation[eq for eq in sys.orderedEqs]
  local scans = EquationScan[scanEquation(eq, unknowns, clockVars) for eq in eqs]
  (isempty(clockVars) && !any(s -> isClocked(s) || !isempty(s.held), scans)) && return bdae
  length(bdae.eqs) == 1 || OMBackend.unsupported("clocks in a model with structural modes", bdae.name)
  local parameters = Dict{String, BDAE.VAR}(string(v.varName) => v for v in sys.orderedVars if isKnown(v))
  for v in vcat(bdae.shared.globalKnownVars, bdae.shared.localKnownVars)
    parameters[string(v.varName)] = v
  end
  local ctx = ClockContext(UnionFind(), Dict{String, DAE.Exp}(), Dict{String, ResolvedClock}(), Set{String}(),
                           DAE.ClockKind[], parameters, IdDict{DAE.ClockKind, Tuple{DAE.ClockKind, Rational{Int}}}())
  local isClockEq = clockEquations!(ctx, eqs, scans, clockVars, sys.orderedVars)
  local (sub, base, representative) = subPartitions(scans, isClockEq, unknowns)
  for (i, s) in enumerate(scans)
    isClocked(s) && representative[i] === nothing && !isClockEq[i] &&
      OMBackend.unsupported("a clocked equation without variables", eqs[i])
  end
  local clocked = clockedSubPartitions(scans, representative, sub)
  isempty(clocked) && return bdae_withoutClocks(bdae, sys, clockVars, isClockEq)
  #= clock inference: the explicit clocks of the partitions' samples, through their sub-clock
     operators to the other partitions of their base partition (MLS 16.7.4) =#
  local clockOf = Dict{String, ResolvedClock}()
  local edges = Tuple{String, String, SubLink}[]
  for (i, s) in enumerate(scans)
    representative[i] === nothing && continue
    local node = find!(sub, representative[i])
    for c in s.clocks
      local r = canonical(ctx, resolveClock(ctx, c))
      r.base === nothing && continue
      local known_ = get(clockOf, node, nothing)
      if known_ !== nothing
        known_ = canonical(ctx, known_)
        known_.base === r.base || rebase!(ctx, known_.base, r.base) ||
          OMBackend.unsupported("a partition sampled with two unrelated clocks", eqs[i])
        sameClock(canonical(ctx, known_), canonical(ctx, r)) ||
          OMBackend.unsupported("a partition sampled with two different clocks", eqs[i])
      end
      clockOf[node] = r
    end
    for link in s.links
      local sources = Set{String}(find!(sub, v) for v in link.vars)
      length(sources) == 1 || OMBackend.unsupported("a $(link.op) of variables of several partitions", eqs[i])
      push!(edges, (node, only(sources), link))
    end
  end
  local changed = true
  while changed
    changed = false
    for (p, q, link) in edges
      local (cp, cq) = (get(clockOf, p, nothing), get(clockOf, q, nothing))
      cp === nothing || (cp = canonical(ctx, cp))
      cq === nothing || (cq = canonical(ctx, cq))
      if cp !== nothing && cq !== nothing && !(cp.base === cq.base)
        rebase!(ctx, cp.base, cq.base) ||
          OMBackend.unsupported("a $(link.op) between partitions of unrelated clocks", link)
        (cp, cq) = (canonical(ctx, cp), canonical(ctx, cq))
      end
      if cq !== nothing && cp === nothing
        local c = forward(link.op, link.args, cq, parameters)
        c === nothing || (clockOf[p] = c; changed = true)
      elseif cp !== nothing && cq === nothing
        local c = backward(link.op, link.args, cp, parameters)
        c === nothing || (clockOf[q] = c; changed = true)
      elseif cp !== nothing && cq !== nothing
        local c = forward(link.op, link.args, cq, parameters)
        if c === nothing
          local ratio = link.op == "subSample" ? cp.factor / cq.factor : cq.factor / cp.factor
          (cp.base === cq.base && cp.shift == cq.shift && isinteger(ratio) && ratio >= 1) ||
            OMBackend.unsupported("an inferred $(link.op) factor that does not fit its clocks", link)
        else
          sameClock(c, cp) || OMBackend.unsupported("a $(link.op) between partitions of other clocks", link)
        end
      end
    end
  end
  for node in clocked
    haskey(clockOf, node) || OMBackend.unsupported("a clocked partition whose clock cannot be inferred",
                                                   sort!([v for v in unknowns if find!(sub, v) == node]))
  end
  #= one base clock per base partition: rational clocks met in one are related =#
  local baseOf = Dict{String, DAE.ClockKind}()
  for node in sort!(collect(clocked))
    local b = find!(base, node)
    local k = canonical(ctx, clockOf[node]).base
    if haskey(baseOf, b) && !(canonical(ctx, ResolvedClock(baseOf[b], 1 // 1, 0 // 1, nothing)).base === k)
      rebase!(ctx, canonical(ctx, ResolvedClock(baseOf[b], 1 // 1, 0 // 1, nothing)).base, k) ||
        OMBackend.unsupported("a base partition of two unrelated base clocks", (string(baseOf[b]), string(k)))
    end
    haskey(baseOf, b) || (baseOf[b] = k)
  end
  for node in clocked
    clockOf[node] = canonical(ctx, clockOf[node])
  end
  return partitioned(bdae, sys, clockVars, isClockEq, eqs, scans, representative, sub, clocked, clockOf, ctx.bases)
end

"""
    bdae_withoutClocks(bdae, sys, clockVars, isClockEq) -> BDAE.BACKEND_DAE

`bdae` without its clock variables and clock equations, for a model whose clocks no partition
samples with.
"""
function bdae_withoutClocks(bdae::BDAE.BACKEND_DAE, sys::BDAE.EQSYSTEM, clockVars::Set{String},
                            isClockEq::Vector{Bool})::BDAE.BACKEND_DAE
  local vars = BDAE.VAR[v for v in sys.orderedVars if !(string(v.varName) in clockVars)]
  local eqs = BDAE.Equation[eq for (i, eq) in enumerate(sys.orderedEqs) if !isClockEq[i]]
  local newSys = BDAE.EQSYSTEM(sys.name, vars, eqs, sys.simpleEquations, sys.initialEqs)
  return BDAE.BACKEND_DAE(bdae.name, BDAE.EQSYSTEM[newSys], bdae.shared, bdae.clocked)
end

"""
    partitioned(...) -> BDAE.BACKEND_DAE

The continuous system and the clocked partitions, in the order of their first equations.
"""
function partitioned(bdae::BDAE.BACKEND_DAE, sys::BDAE.EQSYSTEM, clockVars::Set{String}, isClockEq::Vector{Bool},
                     eqs::Vector{BDAE.Equation}, scans::Vector{EquationScan},
                     representative::Vector{Union{Nothing, String}}, sub::UnionFind, clocked::Set{String},
                     clockOf::Dict{String, ResolvedClock}, bases::Vector{DAE.ClockKind})::BDAE.BACKEND_DAE
  local order = String[]
  local partitionEqs = Dict{String, Vector{BDAE.Equation}}()
  local continuousEqs = BDAE.Equation[]
  for (i, eq) in enumerate(eqs)
    isClockEq[i] && continue
    local node = representative[i] === nothing ? nothing : find!(sub, representative[i])
    if node !== nothing && node in clocked
      haskey(partitionEqs, node) || (push!(order, node); partitionEqs[node] = BDAE.Equation[])
      push!(partitionEqs[node], eq)
    else
      push!(continuousEqs, eq)
    end
  end
  for node in sort!(collect(clocked))
    node in order || push!(order, node)
  end
  local partitionVars = Dict{String, Vector{BDAE.VAR}}(node => BDAE.VAR[] for node in order)
  local continuousVars = BDAE.VAR[]
  for v in sys.orderedVars
    local name = string(v.varName)
    name in clockVars && continue
    local node = haskey(sub.parent, name) ? find!(sub, name) : nothing
    if node !== nothing && node in clocked
      push!(partitionVars[node], v)
    else
      push!(continuousVars, v)
    end
  end
  #= initial equations of a clocked partition's variables go with it =#
  local unknownNames = Set{String}(keys(sub.parent))
  local partitionInitial = Dict{String, Vector{BDAE.Equation}}(node => BDAE.Equation[] for node in order)
  local continuousInitial = BDAE.Equation[]
  for eq in sys.initialEqs
    local s = scanEquation(eq, unknownNames, clockVars)
    local nodes = Set{String}(find!(sub, v) for v in s.own)
    local clockedNodes = intersect(nodes, clocked)
    if isempty(clockedNodes)
      push!(continuousInitial, eq)
    elseif length(clockedNodes) == 1 && length(nodes) == 1
      push!(partitionInitial[only(clockedNodes)], eq)
    else
      OMBackend.unsupported("an initial equation of several partitions", eq)
    end
  end
  local baseIndex = k -> findfirst(b -> b === k, bases)
  local partitions = BDAE.ClockedPartition[]
  for node in order
    local c = clockOf[node]
    push!(partitions, BDAE.ClockedPartition(baseIndex(c.base), BDAE.SubClock(c.factor, c.shift, c.solverMethod),
                                            partitionVars[node], get(partitionEqs, node, BDAE.Equation[]),
                                            partitionInitial[node]))
  end
  local usedBases = sort!(unique(p.baseClock for p in partitions))
  local renumber = Dict(old => new for (new, old) in enumerate(usedBases))
  partitions = BDAE.ClockedPartition[BDAE.ClockedPartition(renumber[p.baseClock], p.subClock, p.vars, p.eqs, p.initialEqs)
                                     for p in partitions]
  local newSys = BDAE.EQSYSTEM(sys.name, continuousVars, continuousEqs, sys.simpleEquations, continuousInitial)
  return BDAE.BACKEND_DAE(bdae.name, BDAE.EQSYSTEM[newSys], bdae.shared,
                          BDAE.ClockedPartitioning(DAE.ClockKind[bases[i] for i in usedBases], partitions))
end

"""
    describe(p::BDAE.ClockedPartitioning) -> String

The base clocks and their partitions: sub-clock (factor, shift), variables, equations.
"""
function describe(p::BDAE.ClockedPartitioning)::String
  local io = IOBuffer()
  for (i, b) in enumerate(p.baseClocks)
    println(io, "base clock ", i, ": ", string(b))
    for part in p.partitions
      part.baseClock == i || continue
      local sc = part.subClock
      println(io, "  sub-clock factor ", sc.factor, ", shift ", sc.shift,
              sc.solverMethod === nothing ? "" : ", solver " * sc.solverMethod,
              ": ", length(part.eqs), " equations; ", join(sort!([string(v.varName) for v in part.vars]), ", "))
    end
  end
  return String(take!(io))
end

end #= ClockPartitioning =#
