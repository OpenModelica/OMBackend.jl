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

#= Clocked (synchronous) Modelica, MLS 16: the clocks of a model resolved and its clocked
   partitions taken out of the continuous system, each with the sub-clock of its base clock
   (omc's SynchronousFeatures; the previous backend's ClockPartitioning).

   A partition is a set of variables connected by equations. `sample(u, clock)` (continuous to
   clocked) and `hold(y)` (clocked to continuous) separate partitions; `subSample`,
   `superSample`, `shiftSample` and `backSample` separate the sub-partitions of one base
   partition and relate their clocks. Clock variables and their equations are resolved to clock
   constructors and leave the model. What is left is the continuous system. =#

const SUB_CLOCK_OPERATORS = (:subSample, :superSample, :shiftSample, :backSample)
const CLOCKED_OPERATORS = (:previous, :interval, :firstTick)

"""
    ResolvedClock(base, factor, shift, solverMethod)

A clock as its base clock (the clock constructor it derives from; `nothing` for an inferred
clock) and its sub-clock of it: ticks at the base clock's `shift + k * factor` intervals.
"""
struct ResolvedClock
  base::Union{Nothing, IR.ClockKind}
  factor::Rational{Int}
  shift::Rational{Int}
  solverMethod::Union{Nothing, String}
end

"""
    SubLink(op, args, vars)

A sub-clock operator in an equation: its operator, its other arguments and the variables of
its first argument (the sub-partition it samples).
"""
struct SubLink
  op::Symbol
  args::Vector{IR.Expression}
  vars::Set{Symbol}
end

"""
    EquationScan

What partitioning needs of an equation: its own variables (the partition it belongs to), the
variables it samples (`sample`'s first argument) and holds (`hold`'s argument), its sub-clock
operators, the explicit clocks of its `sample`s, whether it samples with an inferred clock,
uses `previous`/`interval`/`firstTick`, or is about clocks itself (a clock variable or clock
constructor outside `sample`).
"""
mutable struct EquationScan
  const unknowns::Set{Symbol}
  const clockVars::Set{Symbol}
  own::Set{Symbol}
  boundary::Union{Nothing, Set{Symbol}}
  sampled::Set{Symbol}
  held::Set{Symbol}
  links::Vector{SubLink}
  clocks::Vector{IR.Expression}
  inferred::Bool
  clockedOperators::Bool
  clockExpressions::Bool
end

EquationScan(unknowns::Set{Symbol}, clockVars::Set{Symbol}) =
  EquationScan(unknowns, clockVars, Set{Symbol}(), nothing, Set{Symbol}(), Set{Symbol}(), SubLink[],
               IR.Expression[], false, false, false)

"""
The name of a call of a built-in (`hold`, `subSample`) or of the frontend's clocked `sample`
(`OMC_CLOCK.sample`); `nothing` for another function.
"""
function builtinName(e::IR.CALL)::Union{Nothing, Symbol}
  local s = String(e.name)
  startswith(s, "OMC_CLOCK.") && return Symbol(s[(length("OMC_CLOCK.") + 1):end])
  return occursin('.', s) ? nothing : e.name
end

"Whether `e` is a clock: a clock constructor, a clock variable, a sub-clock operator of a clock."
function isClockExpression(e::IR.Expression, clockVars::Set{Symbol})::Bool
  e isa IR.CLOCK && return true
  e isa IR.CREF && return e.cref.name in clockVars
  e isa IR.CALL && return builtinName(e) in SUB_CLOCK_OPERATORS && !isempty(e.arguments) &&
                          isClockExpression(e.arguments[1], clockVars)
  return false
end

"Scan the expression `e` into `s`, deciding where to descend."
function scanExp!(s::EquationScan, e::IR.Expression)::Nothing
  if e isa IR.CREF
    local name = e.cref.name
    if name in s.clockVars
      s.boundary === nothing && (s.clockExpressions = true)
    elseif name in s.unknowns
      push!(something(s.boundary, s.own), name)
    end
  elseif e isa IR.CLOCK
    s.boundary === nothing && (s.clockExpressions = true)
    return nothing
  elseif e isa IR.CALL
    scanCall!(s, e) || return nothing
  end
  foreach(c -> scanExp!(s, c), IR.children(e))
  return nothing
end

"The variables of `e`, a boundary argument (of `sample`, `hold`, a sub-clock operator), into `target`."
function collectInto!(s::EquationScan, target::Set{Symbol}, e::IR.Expression)::Nothing
  local previous = s.boundary
  s.boundary = target
  scanExp!(s, e)
  s.boundary = previous
  return nothing
end

"A call's clock operators into `s`; whether to descend into its arguments."
function scanCall!(s::EquationScan, e::IR.CALL)::Bool
  s.boundary === nothing || return true
  local name = builtinName(e)
  local args = e.arguments
  if name === :sample && length(args) == 2 && isClockExpression(args[2], s.clockVars)
    collectInto!(s, s.sampled, args[1])
    local clk = args[2]
    clk isa IR.CLOCK && clk.kind isa IR.INFERRED_CLOCK ? (s.inferred = true) : push!(s.clocks, clk)
    return false
  elseif name === :hold && length(args) == 1
    collectInto!(s, s.held, args[1])
    return false
  elseif name in SUB_CLOCK_OPERATORS && !isempty(args) && !isClockExpression(args[1], s.clockVars)
    local vars = Set{Symbol}()
    collectInto!(s, vars, args[1])
    push!(s.links, SubLink(name, args[2:end], vars))
    return false
  elseif name in CLOCKED_OPERATORS
    s.clockedOperators = true
  elseif name === :noClock
    unsupported("noClock (MLS 16.5.2)", e)
  end
  return true
end

"""
    scanEquation(eq, unknowns, clockVars)

An equation's scan. A clocked when-clause (`when clock then y = u; end when`, MLS 16.6.1: MSL's
AssignClock): its body is clocked by the clock, as `sample(·, clock)`.
"""
function scanEquation(eq::IR.Equation, unknowns::Set{Symbol}, clockVars::Set{Symbol})::EquationScan
  local s = EquationScan(unknowns, clockVars)
  if eq isa IR.WHEN_EQUATION && length(eq.branches[1].triggers) == 1 && isClockExpression(eq.branches[1].triggers[1], clockVars)
    local clk = eq.branches[1].triggers[1]
    clk isa IR.CLOCK && clk.kind isa IR.INFERRED_CLOCK ? (s.inferred = true) : push!(s.clocks, clk)
    for b in eq.branches
      foreach(st -> IR.foreachRootExp(e -> scanExp!(s, e), st), b.body)
      b === eq.branches[1] || foreach(t -> scanExp!(s, t), b.triggers)
    end
    return s
  end
  IR.foreachRootExp(e -> scanExp!(s, e), eq)
  return s
end

isClocked(s::EquationScan)::Bool = s.inferred || !isempty(s.clocks) || s.clockedOperators || !isempty(s.links)

"Disjoint sets of names: `find!` gives a set's representative."
struct UnionFind
  parent::Dict{Symbol, Symbol}
end
UnionFind() = UnionFind(Dict{Symbol, Symbol}())

function find!(u::UnionFind, x::Symbol)::Symbol
  local p = get!(u.parent, x, x)
  p === x && return x
  local r = find!(u, p)
  u.parent[x] = r
  return r
end

function unite!(u::UnionFind, a::Symbol, b::Symbol)::Nothing
  local (ra, rb) = (find!(u, a), find!(u, b))
  ra === rb || (u.parent[rb] = ra)
  return nothing
end

"The binding of a parameter, `nothing` for none."
bindingOf(parameters::Dict{Symbol, IR.Variable}, name::Symbol)::Union{Nothing, IR.Expression} =
  haskey(parameters, name) ? parameters[name].binding : nothing

"A sub-clock operator's factor or counter: a literal, or a parameter bound to one."
function integerArgument(e::IR.Expression, parameters::Dict{Symbol, IR.Variable})::Int
  e isa IR.INTEGER && return e.value
  e isa IR.REAL && isinteger(e.value) && return Int(e.value)
  e isa IR.UNARY && e.op === IR.UMINUS && return -integerArgument(e.exp, parameters)
  if e isa IR.CREF
    local b = bindingOf(parameters, e.cref.name)
    b === nothing || return integerArgument(b, parameters)
  end
  unsupported("a sub-clock factor or counter that is not an Integer literal or parameter", e)
end

"`shiftSample`'s and `backSample`'s counter and resolution (1 when not given)."
function resolution(args::Vector{IR.Expression}, parameters::Dict{Symbol, IR.Variable})::Tuple{Int, Int}
  isempty(args) && unsupported("shiftSample or backSample without a counter", args)
  local counter = integerArgument(args[1], parameters)
  local res = length(args) >= 2 ? integerArgument(args[2], parameters) : 1
  res > 0 || unsupported("a shiftSample or backSample resolution below 1", res)
  return (counter, res)
end

"""
The clock of `op(u, args...)` for `u` on clock `c` (MLS 16.5.2); `nothing` for an inferred
factor (`subSample(u)`, `superSample(u)`: factor 0).
"""
function forward(op::Symbol, args::Vector{IR.Expression}, c::ResolvedClock, parameters::Dict{Symbol, IR.Variable})::Union{Nothing, ResolvedClock}
  if op === :subSample || op === :superSample
    local n = isempty(args) ? 0 : integerArgument(args[1], parameters)
    n == 0 && return nothing
    n > 0 || unsupported("a negative $(op) factor", n)
    return ResolvedClock(c.base, op === :subSample ? c.factor * n : c.factor // n, c.shift, c.solverMethod)
  end
  local (counter, res) = resolution(args, parameters)
  local delta = c.factor * (counter // res)
  return ResolvedClock(c.base, c.factor, op === :shiftSample ? c.shift + delta : c.shift - delta, c.solverMethod)
end

"The clock of `u` in `y = op(u, args...)` for `y` on clock `c`: `forward`'s inverse."
function backward(op::Symbol, args::Vector{IR.Expression}, c::ResolvedClock, parameters::Dict{Symbol, IR.Variable})::Union{Nothing, ResolvedClock}
  if op === :subSample || op === :superSample
    local n = isempty(args) ? 0 : integerArgument(args[1], parameters)
    n == 0 && return nothing
    return ResolvedClock(c.base, op === :subSample ? c.factor // n : c.factor * n, c.shift, c.solverMethod)
  end
  local (counter, res) = resolution(args, parameters)
  local delta = c.factor * (counter // res)
  return ResolvedClock(c.base, c.factor, op === :shiftSample ? c.shift - delta : c.shift + delta, c.solverMethod)
end

sameClock(a::ResolvedClock, b::ResolvedClock)::Bool =
  a.base == b.base && a.factor == b.factor && a.shift == b.shift && a.solverMethod == b.solverMethod

"The clock variables' definitions (by the representative of their alias set) and what resolving them needs."
struct ClockContext
  aliases::UnionFind
  definitions::Dict{Symbol, IR.Expression}
  resolved::Dict{Symbol, ResolvedClock}
  visiting::Set{Symbol}
  bases::Vector{IR.ClockKind}
  parameters::Dict{Symbol, IR.Variable}
  rebased::Dict{IR.ClockKind, Tuple{IR.ClockKind, Rational{Int}}}
end

"`e` as an Integer when it is a literal or a parameter bound to one; `nothing` otherwise."
function literalInteger(e::IR.Expression, parameters::Dict{Symbol, IR.Variable})::Union{Nothing, Int}
  e isa IR.INTEGER && return e.value
  e isa IR.REAL && isinteger(e.value) && return Int(e.value)
  if e isa IR.CREF
    local b = bindingOf(parameters, e.cref.name)
    b === nothing || return literalInteger(b, parameters)
  end
  return nothing
end

"The interval of a rational clock `Clock(intervalCounter, resolution)` in seconds; `nothing` for another."
function rationalInterval(k::IR.ClockKind, parameters::Dict{Symbol, IR.Variable})::Union{Nothing, Rational{Int}}
  k isa IR.INTEGER_CLOCK || return nothing
  local (n, r) = (literalInteger(k.intervalCounter, parameters), literalInteger(k.resolution, parameters))
  (n === nothing || r === nothing || r <= 0) && return nothing
  return n // r
end

"`c` on its base clock's representative: a base clock rebased on another counts in the other's intervals."
function canonical(ctx::ClockContext, c::ResolvedClock)::ResolvedClock
  c.base === nothing && return c
  local (b, ratio) = (c.base, 1 // 1)
  while haskey(ctx.rebased, b)
    local (next, r) = ctx.rebased[b]
    (b, ratio) = (next, ratio * r)
  end
  b == c.base && return c
  return ResolvedClock(b, c.factor * ratio, c.shift * ratio, c.solverMethod)
end

"""
Two base clocks met in one base partition: two rational clocks are related, `b` then counts
in intervals of `a` (MLS 16.7.4; MSL CascadeControlledDrive.AbsoluteClocks). Whether they
could be related.
"""
function rebase!(ctx::ClockContext, a::IR.ClockKind, b::IR.ClockKind)::Bool
  a == b && return true
  local (ia, ib) = (rationalInterval(a, ctx.parameters), rationalInterval(b, ctx.parameters))
  (ia === nothing || ib === nothing) && return false
  ctx.rebased[b] = (a, ib / ia)
  return true
end

"The base clock of the constructor `k`: an equal one met before (two `Clock(0.1)` are one clock)."
function baseClock!(ctx::ClockContext, k::IR.ClockKind)::IR.ClockKind
  for b in ctx.bases
    b == k && return b
  end
  push!(ctx.bases, k)
  return k
end

"A clock expression as its base clock and sub-clock."
function resolveClock(ctx::ClockContext, e::IR.Expression)::ResolvedClock
  if e isa IR.CLOCK
    e.kind isa IR.INFERRED_CLOCK && return ResolvedClock(nothing, 1 // 1, 0 // 1, nothing)
    if e.kind isa IR.SOLVER_CLOCK
      local r = resolveClock(ctx, e.kind.clock)
      e.kind.solverMethod isa IR.STRING || unsupported("a solver clock's method that is not a String literal", e.kind.solverMethod)
      return ResolvedClock(r.base, r.factor, r.shift, e.kind.solverMethod.value)
    end
    return ResolvedClock(baseClock!(ctx, e.kind), 1 // 1, 0 // 1, nothing)
  elseif e isa IR.CREF
    return resolveClockVariable(ctx, e.cref.name)
  elseif e isa IR.CALL && builtinName(e) in SUB_CLOCK_OPERATORS && !isempty(e.arguments)
    local r = resolveClock(ctx, e.arguments[1])
    local c = forward(builtinName(e), e.arguments[2:end], r, ctx.parameters)
    c === nothing && unsupported("a clock sub-sampled by an inferred factor", e)
    return c
  end
  unsupported("a clock expression", e)
end

function resolveClockVariable(ctx::ClockContext, name::Symbol)::ResolvedClock
  local root = find!(ctx.aliases, name)
  haskey(ctx.resolved, root) && return ctx.resolved[root]
  root in ctx.visiting && unsupported("a clock defined by itself", name)
  haskey(ctx.definitions, root) || unsupported("a clock variable without a definition", name)
  push!(ctx.visiting, root)
  local r = resolveClock(ctx, ctx.definitions[root])
  delete!(ctx.visiting, root)
  ctx.resolved[root] = r
  return r
end

"""
The clock equations and the clock variables' bindings (aliases of clock variables, a clock
variable's definition) into `ctx`; which equations are clock equations.
"""
function clockEquations!(ctx::ClockContext, equations::Vector{IR.Equation}, scans::Vector{EquationScan},
                         clockVars::Set{Symbol}, variables::IR.VariableSet)::Vector{Bool}
  local isClockEq = falses(length(equations))
  local definitions = Tuple{Symbol, IR.Expression}[]
  for v in variables
    (v.ty isa IR.T_CLOCK && v.binding !== nothing) || continue
    local e = v.binding
    if e isa IR.CREF && e.cref.name in clockVars
      unite!(ctx.aliases, v.name, e.cref.name)
    else
      push!(definitions, (v.name, e))
    end
  end
  for (i, (eq, s)) in enumerate(zip(equations, scans))
    s.clockExpressions && !isClocked(s) && isempty(s.held) || continue
    isempty(s.own) || unsupported("an equation of clocks and variables", eq)
    isClockEq[i] = true
    eq isa IR.SCALAR_EQUATION || unsupported("a clock equation that is not an equality", eq)
    local lname = eq.lhs isa IR.CREF && eq.lhs.cref.name in clockVars ? eq.lhs.cref.name : nothing
    local rname = eq.rhs isa IR.CREF && eq.rhs.cref.name in clockVars ? eq.rhs.cref.name : nothing
    if lname !== nothing && rname !== nothing
      unite!(ctx.aliases, lname, rname)
    elseif lname !== nothing
      push!(definitions, (lname, eq.rhs))
    elseif rname !== nothing
      push!(definitions, (rname, eq.lhs))
    else
      unsupported("a clock equation without a clock variable", eq)
    end
  end
  for (name, e) in definitions
    local root = find!(ctx.aliases, name)
    local previous = get(ctx.definitions, root, nothing)
    previous === nothing || previous == e || unsupported("a clock with two definitions", name)
    ctx.definitions[root] = e
  end
  return isClockEq
end

"A partition's variable: not a parameter, a constant, an external object or a clock."
isPartitionVariable(v::IR.Variable)::Bool = !(IR.isKnown(v) || v.kind isa IR.EXTERNAL_OBJECT || v.ty isa IR.T_CLOCK)

"The names of variables of `variables` sorted as the previous backend sorted them."
sortedByName(names::Union{AbstractSet{Symbol}, AbstractVector{Symbol}, Base.Generator}, variables::IR.VariableSet)::Vector{Symbol} =
  sort!(collect(names); by = n -> legacyName(variables[n].cref))

"""
The partitions: variables connected by an equation's own variables (`sub`); a base partition also
by its sub-clock operators (`base`). `representative[i]` is a variable of equation `i`'s
partition, `nothing` for a clock equation or one without variables.
"""
function subPartitions(scans::Vector{EquationScan}, isClockEq::AbstractVector{Bool},
                       unknowns::Set{Symbol}, variables::IR.VariableSet)::Tuple{UnionFind, UnionFind, Vector{Union{Nothing, Symbol}}}
  local sub = UnionFind()
  local base = UnionFind()
  foreach(v -> (find!(sub, v); find!(base, v)), unknowns)
  local representative = Vector{Union{Nothing, Symbol}}(nothing, length(scans))
  for (i, s) in enumerate(scans)
    (isClockEq[i] || isempty(s.own)) && continue
    local own = sortedByName(s.own, variables)
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
The clocked sub-partitions, by their representatives in `sub`: an equation's that samples, uses
`previous`, `interval` or `firstTick` or a sub-clock operator; the variables held or sub-sampled.
"""
function clockedSubPartitions(scans::Vector{EquationScan}, representative::Vector{Union{Nothing, Symbol}}, sub::UnionFind)::Set{Symbol}
  local clocked = Set{Symbol}()
  for (i, s) in enumerate(scans)
    isClocked(s) && representative[i] !== nothing && push!(clocked, find!(sub, representative[i]))
    foreach(v -> push!(clocked, find!(sub, v)), s.held)
    for link in s.links, v in link.vars
      push!(clocked, find!(sub, v))
    end
  end
  return clocked
end

"Whether a partition has clocks: clock variables, or equations its scans find clocked or holding a clocked value."
hasClocks(clockVars::Set{Symbol}, scans::Vector{EquationScan})::Bool =
  !isempty(clockVars) || any(s::EquationScan -> isClocked(s) || !isempty(s.held), scans)

"""
    hasClocks(model) -> Bool

Whether the model's continuous partition has clocks or clocked equations (MLS 16), as `partitionClocks!` finds them
before it partitions: what the arrays-kept route asks before the passes lower the clocked partitions.
"""
function hasClocks(model::IR.Model)::Bool
  local clockVars::Set{Symbol} = Set{Symbol}(v.name for v in model.variables if v.ty isa IR.T_CLOCK)
  local unknowns::Set{Symbol} = Set{Symbol}(v.name for v in model.variables if isPartitionVariable(v))
  return hasClocks(clockVars, EquationScan[scanEquation(eq, unknowns, clockVars) for eq in model.continuous[1].equations])
end

"""
    clockedVariables(variables, equations)

The variables of the clocked partitions (MLS 16.7), before the clocks are resolved; empty for a
model without clocks. For the discrete synthesis: a clocked equation is evaluated at its clock's
ticks, not at its relations' changes.
"""
function clockedVariables(variables::IR.VariableSet, equations::Vector{IR.Equation})::Set{Symbol}
  local clockVars = Set{Symbol}(v.name for v in variables if v.ty isa IR.T_CLOCK)
  local unknowns = Set{Symbol}(v.name for v in variables if isPartitionVariable(v))
  local scans = EquationScan[scanEquation(eq, unknowns, clockVars) for eq in equations]
  hasClocks(clockVars, scans) || return Set{Symbol}()
  local (sub, _, representative) = subPartitions(scans, falses(length(equations)), unknowns, variables)
  local clocked = clockedSubPartitions(scans, representative, sub)
  return Set{Symbol}(v for v in unknowns if find!(sub, v) in clocked)
end

"""
    partitionClocks!(model)

The model without its clock variables, clock equations and clocked partitions, which are in its
`clocked` partitions with their base clocks (`baseClocks`); the model itself for a model without
clocks. The clocked partitions are in the order of their first equations.
"""
function partitionClocks!(model::IR.Model)::IR.Model
  local system = model.continuous[1]
  local variables = model.variables
  local clockVars = Set{Symbol}(v.name for v in variables if v.ty isa IR.T_CLOCK)
  local unknowns = Set{Symbol}(v.name for v in variables if isPartitionVariable(v))
  local equations = system.equations
  local scans = EquationScan[scanEquation(eq, unknowns, clockVars) for eq in equations]
  hasClocks(clockVars, scans) || return model
  isempty(model.modes) || unsupported("clocks in a model with structural modes", model.name)
  local parameters = Dict{Symbol, IR.Variable}(v.name => v for v in variables if IR.isKnown(v) || v.kind isa IR.EXTERNAL_OBJECT)
  local ctx = ClockContext(UnionFind(), Dict{Symbol, IR.Expression}(), Dict{Symbol, ResolvedClock}(), Set{Symbol}(),
                           IR.ClockKind[], parameters, Dict{IR.ClockKind, Tuple{IR.ClockKind, Rational{Int}}}())
  local isClockEq = clockEquations!(ctx, equations, scans, clockVars, variables)
  local (sub, base, representative) = subPartitions(scans, isClockEq, unknowns, variables)
  for (i, s) in enumerate(scans)
    isClocked(s) && representative[i] === nothing && !isClockEq[i] && unsupported("a clocked equation without variables", equations[i])
  end
  local clocked = clockedSubPartitions(scans, representative, sub)
  if isempty(clocked)
    withoutClocks!(model, clockVars, isClockEq)
    return model
  end
  #= Clock inference: the explicit clocks of the partitions' samples, through their sub-clock
     operators to the other partitions of their base partition (MLS 16.7.4). =#
  local clockOf = Dict{Symbol, ResolvedClock}()
  local edges = Tuple{Symbol, Symbol, SubLink}[]
  for (i, s) in enumerate(scans)
    representative[i] === nothing && continue
    local node = find!(sub, representative[i])
    for c in s.clocks
      local r = canonical(ctx, resolveClock(ctx, c))
      r.base === nothing && continue
      local known_ = get(clockOf, node, nothing)
      if known_ !== nothing
        known_ = canonical(ctx, known_)
        known_.base == r.base || rebase!(ctx, known_.base, r.base) ||
          unsupported("a partition sampled with two unrelated clocks", equations[i])
        sameClock(canonical(ctx, known_), canonical(ctx, r)) || unsupported("a partition sampled with two different clocks", equations[i])
      end
      clockOf[node] = r
    end
    for link in s.links
      local sources = Set{Symbol}(find!(sub, v) for v in link.vars)
      length(sources) == 1 || unsupported("a $(link.op) of variables of several partitions", equations[i])
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
      if cp !== nothing && cq !== nothing && !(cp.base == cq.base)
        rebase!(ctx, cp.base, cq.base) || unsupported("a $(link.op) between partitions of unrelated clocks", link)
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
          local ratio = link.op === :subSample ? cp.factor / cq.factor : cq.factor / cp.factor
          (cp.base == cq.base && cp.shift == cq.shift && isinteger(ratio) && ratio >= 1) ||
            unsupported("an inferred $(link.op) factor that does not fit its clocks", link)
        else
          sameClock(c, cp) || unsupported("a $(link.op) between partitions of other clocks", link)
        end
      end
    end
  end
  for node in clocked
    haskey(clockOf, node) ||
      unsupported("a clocked partition whose clock cannot be inferred", sortedByName((v for v in unknowns if find!(sub, v) === node), variables))
  end
  #= One base clock per base partition: rational clocks met in one are related. =#
  local baseOf = Dict{Symbol, IR.ClockKind}()
  for node in sortedByName(clocked, variables)
    local b = find!(base, node)
    local k = canonical(ctx, clockOf[node]).base
    if haskey(baseOf, b)
      local current = canonical(ctx, ResolvedClock(baseOf[b], 1 // 1, 0 // 1, nothing)).base
      current == k || rebase!(ctx, current, k) ||
        unsupported("a base partition of two unrelated base clocks", (IR.toString(baseOf[b]), IR.toString(k)))
    end
    haskey(baseOf, b) || (baseOf[b] = k)
  end
  for node in clocked
    clockOf[node] = canonical(ctx, clockOf[node])
  end
  partitioned!(model, clockVars, isClockEq, scans, representative, sub, clocked, clockOf, ctx.bases)
  return model
end

"The model without its clock variables and clock equations, for clocks no partition samples with."
function withoutClocks!(model::IR.Model, clockVars::Set{Symbol}, isClockEq::AbstractVector{Bool})::Nothing
  filter!(v -> !(v.name in clockVars), model.variables)
  local system = model.continuous[1]
  system.equations = IR.Equation[eq for (i, eq) in enumerate(system.equations) if !isClockEq[i]]
  return nothing
end

"The continuous system and the clocked partitions, in the order of their first equations."
function partitioned!(model::IR.Model, clockVars::Set{Symbol}, isClockEq::AbstractVector{Bool}, scans::Vector{EquationScan},
                      representative::Vector{Union{Nothing, Symbol}}, sub::UnionFind, clocked::Set{Symbol},
                      clockOf::Dict{Symbol, ResolvedClock}, bases::Vector{IR.ClockKind})::Nothing
  local system = model.continuous[1]
  local order = Symbol[]
  local partitionEqs = Dict{Symbol, Vector{IR.Equation}}()
  local continuousEqs = IR.Equation[]
  for (i, eq) in enumerate(system.equations)
    isClockEq[i] && continue
    local node = representative[i] === nothing ? nothing : find!(sub, representative[i])
    if node !== nothing && node in clocked
      haskey(partitionEqs, node) || (push!(order, node); partitionEqs[node] = IR.Equation[])
      push!(partitionEqs[node], eq)
    else
      push!(continuousEqs, eq)
    end
  end
  for node in sortedByName(clocked, model.variables)
    node in order || push!(order, node)
  end
  filter!(v -> !(v.name in clockVars), model.variables)
  local partitionVars = Dict{Symbol, Vector{Symbol}}(node => Symbol[] for node in order)
  for v in model.variables
    local node = haskey(sub.parent, v.name) ? find!(sub, v.name) : nothing
    node !== nothing && node in clocked && push!(partitionVars[node], v.name)
  end
  #= The initial equations of a clocked partition's variables go with it. =#
  local unknownNames = Set{Symbol}(keys(sub.parent))
  local partitionInitial = Dict{Symbol, Vector{IR.Equation}}(node => IR.Equation[] for node in order)
  local continuousInitial = IR.Equation[]
  for eq in system.initialEquations
    local s = scanEquation(eq, unknownNames, clockVars)
    local nodes = Set{Symbol}(find!(sub, v) for v in s.own)
    local clockedNodes = intersect(nodes, clocked)
    if isempty(clockedNodes)
      push!(continuousInitial, eq)
    elseif length(clockedNodes) == 1 && length(nodes) == 1
      push!(partitionInitial[only(clockedNodes)], eq)
    else
      unsupported("an initial equation of several partitions", eq)
    end
  end
  local baseIndex = k -> findfirst(b -> b == k, bases)
  local used = sort!(unique(baseIndex(clockOf[node].base) for node in order))
  local renumber = Dict(old => new for (new, old) in enumerate(used))
  empty!(model.clocked)
  for (i, node) in enumerate(order)
    local c = clockOf[node]
    local association = IR.CLOCKED_PARTITION(renumber[baseIndex(c.base)], IR.SubClock(c.factor, c.shift, c.solverMethod))
    push!(model.clocked, IR.Partition(i + 1, association, partitionVars[node], get(partitionEqs, node, IR.Equation[]),
                                      partitionInitial[node]))
  end
  empty!(model.baseClocks)
  append!(model.baseClocks, IR.ClockKind[bases[i] for i in used])
  system.equations = continuousEqs
  system.initialEquations = continuousInitial
  return nothing
end
