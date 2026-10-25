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

#= The structure of a continuous partition as index reduction sees it: scalar rows over the
   continuous-time Real variables, each variable linked to its derivative. =#

"""
    NotHandled(reason)
    NotHandled(reason, arrays, equations)

What index reduction does not handle (yet): the model is left to the MTK path. With arrays kept, the arrays whose
elements a plan treats differently and the for-equations (by index in the partition) whose iterations it
differentiates differently: the array form takes the plan once those are split (`splitArrays!`).
"""
struct NotHandled <: Exception
  reason::String
  arrays::Vector{Symbol}
  equations::Vector{Int}
end

NotHandled(reason::String) = NotHandled(reason, Symbol[], Int[])

"Whether `e` names arrays or for-equations whose splitting would let the array form take the plan (slicing)."
needsSplit(e::NotHandled)::Bool = !(isempty(e.arrays) && isempty(e.equations))

Base.showerror(io::IO, e::NotHandled) = print(io, "index reduction does not handle ", e.reason)

"""
    Row(equation, position)

A scalar equation of the partition: the `position`-th equation of `partition.equations[equation]`
(a scalar equation's only one; an if-equation's `position`-th equation of each branch; an
algorithm's `position`-th output).
"""
struct Row
  equation::Int
  position::Int
end

"""
    Structure

The continuous-time structure of a partition (Pantelides' bipartite graph with its derivative
links), grown by `pantelides!`:
- `variables`: the continuous Real variables by name, the derivatives `pantelides!` adds after
  them; `derivative[v]` the variable holding v's derivative (0: none), `antiderivative[v]` the
  variable v is the derivative of (0: none).
- `rows`: the partition's scalar equations, then the differentiated ones; `incidence[r]` the
  variables row r reads (each lower derivative too), `differentiated[r]` the row holding r's
  derivative (0: none), `origin[r]` the row r is the derivative of (0: none).
"""
mutable struct Structure
  variables::Vector{Symbol}
  index::Dict{Symbol, Int}
  derivative::Vector{Int}
  antiderivative::Vector{Int}
  rows::Vector{Row}
  incidence::Vector{Vector{Int}}
  differentiated::Vector{Int}
  origin::Vector{Int}
end

"The number of times `v` is a derivative: 0 for a variable, 1 for a derivative, ..."
function derivativeOrder(s::Structure, v::Int)::Int
  local n = 0
  while s.antiderivative[v] != 0
    v = s.antiderivative[v]
    n += 1
  end
  return n
end

"The variable `v` is a derivative of, at order 0 (itself for a variable)."
function baseVariable(s::Structure, v::Int)::Int
  while s.antiderivative[v] != 0
    v = s.antiderivative[v]
  end
  return v
end

"Whether `v` is a highest derivative: an unknown of the matching (no derivative of it occurs)."
isHighest(s::Structure, v::Int)::Bool = s.derivative[v] == 0

"""
    addVariable!(s, name) -> Int

`name` as a variable of the structure (once); its index.
"""
function addVariable!(s::Structure, name::Symbol)::Int
  local i = get(s.index, name, 0)
  i != 0 && return i
  push!(s.variables, name)
  push!(s.derivative, 0)
  push!(s.antiderivative, 0)
  s.index[name] = length(s.variables)
  return length(s.variables)
end

"""
    discreteTargets(partition) -> Set{Symbol}

The variables a when-equation of the partition assigns: discrete, known to the continuous
system (as parameters, constants and time are).
"""
function discreteTargets(partition::IR.Partition)::Set{Symbol}
  local out = Set{Symbol}()
  local visit(s::IR.ASSIGNMENT) = s.lhs isa IR.CREF && push!(out, s.lhs.cref.name)
  local visit(s::Union{IR.IF_STATEMENT, IR.WHEN_STATEMENT}) =
    (foreach(b -> foreach(visit, b.body), s.branches); s isa IR.IF_STATEMENT && foreach(visit, s.elseBody))
  local visit(s::Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT}) = foreach(visit, s.body)
  local visit(s::IR.Statement) = nothing
  for eq in partition.equations
    eq isa IR.WHEN_EQUATION || continue
    for b in eq.branches
      foreach(visit, b.body)
    end
  end
  return out
end

"""
    isContinuousVariable(v, discretes) -> Bool

Whether `v` is a continuous-time unknown or state of the continuous system: Real, a state, a
state derivative or an algebraic variable, not a when-equation's target.
"""
function isContinuousVariable(v::IR.Variable, discretes::Set{Symbol})::Bool
  v.ty isa IR.T_REAL || return false
  v.name in discretes && return false
  return v.kind isa Union{IR.STATE, IR.STATE_DER, IR.ALGEBRAIC}
end

"The built-in calls whose arguments are not read at the current time as continuous variables."
const NOT_CONTINUOUS_READS = (:pre, :previous, :edge, :change, :initial, :terminal, :sample, :delay,
                              :spatialDistribution, :cardinality)

"""
    continuousReads!(out, e, resolve)

The continuous variables `e` reads, into `out`: `resolve(name)` gives the variables a reference
names (itself; a record's fields or an array's elements for a whole record or array passed to
a function, `Frames.resolve2(frame_a.R, ...)`; none for a parameter or a discrete). Not those
under pre(), edge(), change() and delay() (known at the current time); only the actual
argument of homotopy().
"""
function continuousReads!(out::Vector{Symbol}, e::IR.Expression, resolve::F)::Vector{Symbol} where {F}
  if e isa IR.CREF
    for n in resolve(e.cref.name)
      n in out || push!(out, n)
    end
    for c in IR.children(e)
      continuousReads!(out, c, resolve)
    end
    return out
  end
  if e isa IR.CALL && e.attributes.builtin
    e.name in NOT_CONTINUOUS_READS && return out
    if e.name === :homotopy
      return continuousReads!(out, e.arguments[1], resolve)
    end
    #= der() of an expression is expanded first (`expandDerivatives!`): one left is not handled =#
    e.name === :der && throw(NotHandled("der() of an expression ($(IR.toString(e)))"))
  end
  if e isa IR.ARRAY_CONSTRUCTOR
    #= the body for each value of its iterator (`x[i]` is `x[2]`: a variable), over a literal 1:n =#
    local range = length(e.iterators) == 1 ? e.iterators[1].range : nothing
    if range isa IR.RANGE && isOne(range.start) && range.step === nothing && range.stop isa IR.INTEGER
      foreach(k -> continuousReads!(out, withIterator(e.body, e.iterators[1].name, k), resolve), 1:range.stop.value)
      return out
    end
    readsIterated(e) && throw(NotHandled("an array constructor reading variables by its iterator over a range that is not constant"))
  end
  for c in IR.children(e)
    continuousReads!(out, c, resolve)
  end
  return out
end

"""
    expandDerivatives!(equations, continuous, functions) -> Vector{Symbol}

der() of an expression in `equations` (in place) replaced by its time derivative, each variable
of `continuous` it reads by that variable's derivative (`der(a - k * b)` is `der(a) - k * der(b)`,
`der({x, y})` is `{der(x), der(y)}`, a parameter's derivative 0): MSL FluxTubes' hysteresis
`der(hystR - mu0 * Hstat)`, a MultiBody constraint's `der(frame_b.r_0)`. The variables read
this way (each one's derivative the structure adds).
"""
function expandDerivatives!(equations::Vector{IR.Equation}, continuous::Set{Symbol},
                            functions::Dict{Symbol, IR.ModelicaFunction})::Vector{Symbol}
  local derived = Symbol[]
  local variable = (name::Symbol) -> begin
    name in continuous || return nothing
    name in derived || push!(derived, name)
    return IR.CREF(IR.ComponentRef(IR.derivativeName(name), IR.T_REAL()), IR.T_REAL())
  end
  local seeds = Seeds(variable, 1.0, functions)
  local expand = (e::IR.Expression) -> (e isa IR.CALL && e.attributes.builtin && e.name === :der) ?
    differentiate(only(e.arguments), seeds) : e
  for eq in equations
    IR.mapEquation!(expand, eq)
  end
  return derived
end

"Whether the body of an array constructor reads a reference subscripted by one of its iterators (`x[i]`)."
function readsIterated(e::IR.ARRAY_CONSTRUCTOR)::Bool
  local names = Set{Symbol}(it.name for it in e.iterators)
  local found = false
  IR.foreachExp(e.body) do x
    x isa IR.CREF || return nothing
    for p in x.cref.parts, sub in p.subscripts
      sub isa IR.INDEX && any(c -> c.name in names, IR.crefsOf(sub.exp)) && (found = true)
    end
    nothing
  end
  return found
end

"""
    Resolver(names)

`resolver(name)`: the variables of `names` a reference to `name` reads: the variable itself, or
for a record or an array (not a variable after scalarization) every variable under it
(`frame_a.R` -> `frame_a.R.T[1,1]`, ..., `frame_a.R.w[3]`).
"""
struct Resolver
  variables::Set{Symbol}
  under::Dict{Symbol, Vector{Symbol}}
end

function Resolver(names::Vector{Symbol})
  local under = Dict{Symbol, Vector{Symbol}}()
  for n in names
    local str = String(n)
    for (i, c) in enumerate(str)
      (c == '.' || c == '[') || continue
      push!(get!(() -> Symbol[], under, Symbol(str[1:prevind(str, i)])), n)
    end
  end
  return Resolver(Set{Symbol}(names), under)
end

function (r::Resolver)(name::Symbol)
  name in r.variables && return (name,)
  return get(r.under, name, ())
end

"""
    equationReads(eq, resolve, functions) -> Vector{Vector{Symbol}}

The rows of an equation by position and the continuous variables each reads (`nothing`: the
position is no row, a non-Real equation): one for a scalar equation; for a tuple equation one
per continuous variable its targets name, each reading what the equation reads;
for an if-equation one per equation of a branch, row k reading the k-th equation of each branch
(the branch taken changes at events); for an algorithm one per output, each reading what the
algorithm reads and writes. An assert gives none. Array, record and for-equations, which the
scalarized pipeline has expanded, are not handled.
"""
function equationReads(eq::IR.Equation, resolve::F, functions::Dict{Symbol, IR.ModelicaFunction})::Vector{Union{Nothing, Vector{Symbol}}} where {F}
  if eq isa IR.SCALAR_EQUATION && eq.lhs isa IR.TUPLE
    #= `(cc, m_dCap) = diodeNoBypassCode(...)`: a row per continuous variable its targets name (a record
       target's fields), each reading what the equation reads (as an algorithm's) =#
    local out = Symbol[]
    continuousReads!(out, eq.lhs, resolve)
    local targets = length(out)
    continuousReads!(out, inlineCalls(eq.rhs, functions), resolve)
    return Union{Nothing, Vector{Symbol}}[copy(out) for _ in 1:targets]
  elseif eq isa IR.SCALAR_EQUATION
    eq.ty isa IR.T_REAL || return Union{Nothing, Vector{Symbol}}[nothing]
    local out = Symbol[]
    continuousReads!(out, inlineCalls(eq.lhs, functions), resolve)
    continuousReads!(out, inlineCalls(eq.rhs, functions), resolve)
    return Union{Nothing, Vector{Symbol}}[out]
  elseif eq isa IR.IF_EQUATION
    local counts = Int[length(b.equations) for b in eq.branches]
    push!(counts, length(eq.elseEquations))
    all(==(counts[1]), counts) || throw(NotHandled("an if-equation whose branches have different numbers of equations"))
    #= row k reads the k-th equation of each branch (the code generators write residual k as
       ifelse(condition, branch 1's k-th, ...)); the lowering keeps the else-branch reversed =#
    local rows = Union{Nothing, Vector{Symbol}}[nothing for _ in 1:counts[1]]
    for k in 1:counts[1], e in positionEquations(eq, k)
      #= a position of non-Real equations (a Boolean `noWphase = ...`) is no row =#
      local reads = equationReads(e, resolve, functions)
      any(r -> r !== nothing, reads) || continue
      rows[k] === nothing && (rows[k] = Symbol[])
      foreach(r -> r === nothing || append!(rows[k], r), reads)
    end
    return Union{Nothing, Vector{Symbol}}[r === nothing ? nothing : unique!(r) for r in rows]
  elseif eq isa IR.ALGORITHM
    local out = Symbol[]
    IR.foreachRootExp(e -> continuousReads!(out, e, resolve), eq)
    local outputs = sum(c -> length(resolve(c.name)), eq.outputs; init = 0)
    return Union{Nothing, Vector{Symbol}}[copy(out) for _ in 1:outputs]
  elseif eq isa Union{IR.ASSERT_EQUATION, IR.WHEN_EQUATION}
    return Union{Nothing, Vector{Symbol}}[]
  end
  throw(NotHandled("an equation of kind $(nameof(typeof(eq)))"))
end

"""
    Structure(variables, partition, functions; derived) -> Structure

The continuous-time structure of `partition`: its continuous Real variables (a state linked to
its derivative variable, so is each of `derived`: the variables expanded der() of expressions
read, `expandDerivatives!`) and its scalar rows with the variables each reads (through the
inlined form of the calls: `planarRotation(e, phi, w).T` reads phi, not w).
"""
function Structure(variables::IR.VariableSet, partition::IR.Partition, functions::Vector{IR.ModelicaFunction};
                   derived::Vector{Symbol} = Symbol[])::Structure
  local functionsByName = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in functions)
  local discretes = discreteTargets(partition)
  local s = Structure(Symbol[], Dict{Symbol, Int}(), Int[], Int[], Row[], Vector{Int}[], Int[], Int[])
  for v in variables
    isContinuousVariable(v, discretes) && addVariable!(s, v.name)
  end
  for v in variables
    v.kind isa IR.STATE && v.kind.derivative !== nothing && haskey(s.index, v.name) || continue
    local x = s.index[v.name]
    local d = get(s.index, v.kind.derivative, 0)
    d == 0 && continue
    s.derivative[x] = d
    s.antiderivative[d] = x
  end
  foreach(name -> haskey(s.index, name) && addDerivative!(s, s.index[name]), derived)
  local resolve = Resolver(copy(s.variables))
  for (i, eq) in enumerate(partition.equations)
    for (k, reads) in enumerate(equationReads(eq, resolve, functionsByName))
      reads === nothing && continue
      push!(s.rows, Row(i, k))
      push!(s.incidence, Int[s.index[n] for n in reads])
      push!(s.differentiated, 0)
      push!(s.origin, 0)
    end
  end
  #= a state whose derivative no equation reads (a fold took it out: Spice3's `capbx * der(vbx)`, capbx 0) is no
     state, as SimCode's demoteDeadStates makes it =#
  local read = falses(length(s.variables))
  foreach(row -> read[row] .= true, s.incidence)
  for v in eachindex(s.variables)
    local d = s.derivative[v]
    (d != 0 && !read[d]) || continue
    s.derivative[v] = 0
    s.antiderivative[d] = 0
  end
  return s
end

"""
    unknownCount(s) -> Int

The unknowns of the continuous system: its highest derivatives that a row reads.
"""
function unknownCount(s::Structure)::Int
  #= a derivative a fold took out of every equation (`0 * der(x)`: MSL Spice3's capbx 0, a rotor's nJ[2] 0) is no
     unknown: its state was demoted (the constructor) and no row reads it =#
  local read = falses(length(s.variables))
  foreach(row -> read[row] .= true, s.incidence)
  return count(v -> isHighest(s, v) && read[v], eachindex(s.variables))
end

"""
    inlineCalls(e, functions) -> Expression

`e` with each call of a function whose algorithm runs into one expression replaced by that
expression (its inputs substituted, `inlineBody`), and a field or element of a record or array literal
picked (`f(x)[1]` of an inlined record constructor); scalar arithmetic folded (`0 * w` is `0`:
what a literal argument, a known parameter's value, makes of a body): the dependencies as they
are, for the structure. Calls that do not inline stay, so does a recursive call within its own
inlined body (`active`: the functions being inlined), a call with a record argument at an iterator's
subscript (`iteratorIndexedRecord`), and a call of a function with a derivative
annotation whose inlined form reads a call that neither inlines nor has an annotation (`opaqueCall`):
differentiation takes its derivative from the annotation (`functionDerivative`). Inlined, MSL IF97's
`rho_props_pT(p, T, aux)` (`Inline = false, LateInline = true`, derivative `rho_pT_der` with
`noDerivative = aux`) became `waterBaseProp_pT(p, T, region).rho`, and the derivative of the property
record, which MSL does not give, was needed. Other annotated functions inline (MultiBody's `resolve1`,
`noDerivative = R`, reads the frame's variables): kept, the Jacobian entries of EngineV6 were calls of the
derivative functions nested in each other's arguments, and its choice of states ran past 8 minutes (10 s
inlined).
"""
function inlineCalls(e::IR.Expression, functions::Dict{Symbol, IR.ModelicaFunction};
                     active::Vector{Symbol} = Symbol[])::IR.Expression
  return IR.mapExp(e) do x
    if x isa IR.CALL && !x.attributes.builtin
      local f = get(functions, x.name, nothing)
      (f === nothing || x.name in active || any(iteratorIndexedRecord, x.arguments)) && return x
      local body = inlineBody(f, x)
      body === nothing && return x
      push!(active, x.name)
      local inlined = inlineCalls(body, functions; active)
      pop!(active)
      (any(d -> d.order == 1, f.derivatives) && IR.anyExp(y -> opaqueCall(y, functions), inlined)) && return x
      return inlined
    elseif x isa IR.SUBSCRIPTED
      return subscriptedReference(foldTree(selectElement(x.exp, x.subscripts, x.ty)))
    elseif x isa IR.RECORD_ELEMENT
      local record = asRecord(x.exp)
      return record === nothing ? x : foldTree(record.elements[x.index])
    end
    return foldScalar(x)
  end
end

"""
    subscriptedReference(e) -> Expression

A reference subscripted further (`(G[\$c].frame.R.T)[2, \$i1]`, an inlined body's field of a rolled component's
record) as the reference with the subscripts on its parts that have dimensions left (`G[\$c].frame.R.T[2, \$i1]`),
which differentiation and the element view read as an element; another expression as it is.
"""
function subscriptedReference(e::IR.Expression)::IR.Expression
  (e isa IR.SUBSCRIPTED && e.exp isa IR.CREF && all(s::IR.Subscript -> s isa IR.INDEX, e.subscripts)) || return e
  local parts::Vector{IR.CrefPart} = copy((e.exp::IR.CREF).cref.parts)
  local next::Int = 1
  for (k, p) in enumerate(parts)
    local free::Int = length(IR.dimensions(p.ty)) - length(p.subscripts)
    (free > 0 && next <= length(e.subscripts)) || continue
    any(s::IR.Subscript -> !(s isa IR.INDEX), p.subscripts) && return e
    local taken::Int = min(free, length(e.subscripts) - next + 1)
    parts[k] = IR.CrefPart(p.name, IR.Subscript[p.subscripts; e.subscripts[next:next + taken - 1]], p.ty)
    next += taken
  end
  next > length(e.subscripts) || return e
  return IR.CREF(IR.ComponentRef(parts), e.ty)
end

"""
Whether `e` is a record element of an array at a subscript that is no literal (`data[i]` in a constructor
over `i`): inlined, a function's body reads its fields, `data[i].Tlimit`, which the simulation code cannot
name (MSL Media's mixtures, `{h_T(data[i], T) for i in 1:nX}`: SimpleNaturalGas ran with undefined
parameters once its medium's T was a state). The call stays. Not a record of rolled components (`G[\$c].frame_a.R`,
`rollComponents!`), whose fields are arrays the code names.
"""
iteratorIndexedRecord(e::IR.Expression)::Bool =
  e isa IR.CREF && e.ty isa IR.T_RECORD &&
  any(p -> !(p.ty isa IR.T_COMPONENTS) && any(sub -> sub isa IR.INDEX && !(sub.exp isa IR.INTEGER), p.subscripts), e.cref.parts)

"""
    opaqueCall(e, functions) -> Bool

Whether `e` is a call of a user function that differentiation cannot take a derivative of: without a derivative
annotation, its body left a call by `inlineCalls` (an external function, a loop, IF97's `waterBaseProp_pT`).
"""
function opaqueCall(e::IR.Expression, functions::Dict{Symbol, IR.ModelicaFunction})::Bool
  (e isa IR.CALL && !e.attributes.builtin) || return false
  local f = get(functions, e.name, nothing)
  return f === nothing || !any(d -> d.order == 1, f.derivatives)
end

"""
    inlineEquation(eq, functions) -> Equation

A scalar or if-equation with its calls inlined (`inlineCalls`): the form the structure reads
and index reduction differentiates, so both see the same dependencies. Other kinds as they are.
"""
inlineEquation(eq::IR.SCALAR_EQUATION, functions::Dict{Symbol, IR.ModelicaFunction})::IR.Equation =
  IR.SCALAR_EQUATION(inlineCalls(eq.lhs, functions), inlineCalls(eq.rhs, functions), eq.ty, eq.source, eq.attributes)

function inlineEquation(eq::IR.IF_EQUATION, functions::Dict{Symbol, IR.ModelicaFunction})::IR.Equation
  local branches = IR.EquationBranch[IR.EquationBranch(b.condition, IR.Equation[inlineEquation(e, functions) for e in b.equations])
                                      for b in eq.branches]
  return IR.IF_EQUATION(branches, IR.Equation[inlineEquation(e, functions) for e in eq.elseEquations], eq.source, eq.attributes)
end

inlineEquation(eq::IR.Equation, ::Dict{Symbol, IR.ModelicaFunction})::IR.Equation = eq

"""
    positionEquations(eq, k) -> Vector{Equation}

The equations of an if-equation at position `k`: the k-th of each branch and the else-branch's
k-th from the end (the lowering reverses the else-branch, LowerFlatModel).
"""
function positionEquations(eq::IR.IF_EQUATION, k::Int)::Vector{IR.Equation}
  local out = IR.Equation[b.equations[k] for b in eq.branches]
  push!(out, eq.elseEquations[length(eq.elseEquations) + 1 - k])
  return out
end
