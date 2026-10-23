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

#= The derivative unknowns of a reduced partition solved for (M6, its first step): the code
   generators solve an equation for a variable whose coefficient is constant; one that reads
   unknowns (`icap = C(v) * der(v)`) they leave an algebraic equation of the derivative, an
   unknown the integrator does not control. =#

"""
    isDerivativeUnknown(v) -> Bool

Whether `v` is a derivative unknown of a reduced partition: a dummy derivative, or a state's
derivative variable `\$DER.x` that its link `\$STATEDER.x = \$DER.x` made algebraic.
"""
isDerivativeUnknown(v::IR.Variable)::Bool =
  v.kind isa IR.DUMMY_DER || (v.kind isa IR.ALGEBRAIC && startswith(String(v.name), "\$DER."))

"Whether `eq` is a state's derivative link `\$STATEDER.x = \$DER.x`: the state's differential row, which determines no derivative unknown."
isDerivativeLink(eq::IR.Equation)::Bool =
  eq isa IR.SCALAR_EQUATION && eq.lhs isa IR.CREF && startswith(String(eq.lhs.cref.name), "\$STATEDER.")

"Whether `e` reads no unknown (only parameters, constants and literals; not `time`): a coefficient the code generators solve through."
function readsNoUnknown(e::IR.Expression, variables::IR.VariableSet)::Bool
  local known = true
  IR.foreachExp(e) do x
    x isa IR.CREF || return nothing
    local v = get(variables, x.cref.name, nothing)
    (v === nothing || !(v.kind isa Union{IR.PARAMETER, IR.CONSTANT})) && (known = false)
    nothing
  end
  return known
end

"""
    isZeroConstant(e) -> Bool

Whether `e` is 0 by its form: a literal 0, an array (or a concatenation) of such, an element of
one, its negation, a product with one, a sum of them (`-cat(1, cat(2, {{0.0}}, ...), ...)[1, 1]`,
the derivative of a record's field that does not read the variable).
"""
function isZeroConstant(e::IR.Expression)::Bool
  e isa Union{IR.REAL, IR.INTEGER} && return e.value == 0
  e isa IR.ARRAY && return all(isZeroConstant, e.elements)
  e isa IR.SUBSCRIPTED && return isZeroConstant(e.exp)
  e isa IR.UNARY && e.op === IR.UMINUS && return isZeroConstant(e.exp)
  e isa IR.CALL && e.name === :cat && return length(e.arguments) > 1 && all(isZeroConstant, e.arguments[2:end])
  if e isa IR.BINARY
    e.op in (IR.MUL, IR.MUL_ARRAY_SCALAR, IR.MUL_SCALAR_ARRAY) && return isZeroConstant(e.exp1) || isZeroConstant(e.exp2)
    e.op in (IR.ADD, IR.SUB, IR.ADD_EW, IR.SUB_EW) && return isZeroConstant(e.exp1) && isZeroConstant(e.exp2)
  end
  return false
end

"""
    isPiecewise(e) -> Bool

Whether `e` reads an if-expression with a branch 0 by its form (`isZeroConstant`): a
coefficient 0 in some branch (MSL Media's smooth state transition, 1 in one branch and 0 in
the other). A coefficient piecewise with no such branch is solved through: MSL R134a's
`rho_ph_der` reads `der(p)` through one expression for the two-phase region and another
outside it; left an algebraic unknown, `der(p)` of R134a2's volumes jumped where their states
crossed the dew line, and the steps fell to 1e-16 (Unstable).
"""
function isPiecewise(e::IR.Expression)::Bool
  local piecewise = false
  IR.foreachExp(e) do x
    (x isa IR.IF_EXPRESSION && (isZeroConstant(x.thenExp) || isZeroConstant(x.elseExp))) && (piecewise = true)
    nothing
  end
  return piecewise
end

"Whether `e` reads only variables whose start the model gives (parameters, constants, a `start` attribute): its value at the start values means something."
function readsGivenStarts(e::IR.Expression, variables::IR.VariableSet)::Bool
  local given = true
  IR.foreachExp(e) do x
    x isa IR.CREF || return nothing
    local v = get(variables, x.cref.name, nothing)
    (v === nothing || !(v.kind isa Union{IR.PARAMETER, IR.CONSTANT} || v.attributes.start !== nothing)) && (given = false)
    nothing
  end
  return given
end

"Whether `eq` reads `homotopy`: solved through its actual argument only, the simplified one would drop out of the initialization."
function readsHomotopy(eq::IR.Equation)::Bool
  local found = false
  IR.foreachExp(x -> (x isa IR.CALL && x.name === :homotopy && (found = true); nothing), eq)
  return found
end

"""
    DerivativeEdge(equation, coefficient, constant, pivot, others)

An equation reading a derivative unknown linearly: the equation's index, the partial derivative
of its residual by the unknown (`nothing` for an if-equation; its value, a literal, where it
reads no unknown), whether that coefficient is such a constant, its magnitude at the start
values (`Inf` for a constant or an if-equation; 0 where it does not evaluate, or reads a
variable without a given start), the other derivative unknowns the equation reads.
"""
struct DerivativeEdge
  equation::Int
  coefficient::Union{Nothing, IR.Expression}
  constant::Bool
  pivot::Float64
  others::Vector{Symbol}
end

"A coefficient at most this large at given start values is 0: not solved through."
const ZERO_PIVOT = 1e-12

"""
    derivativeEdges(model, partition, unknowns, functions, known, start; simValues) -> Dict{Symbol, Vector{DerivativeEdge}}

For each derivative unknown the equations that can determine it: those reading it linearly (a
partial derivative that does not read it, not 0); an if-equation reading it as one with a
constant coefficient (its branches are not solved here). Not the derivative links, not another
kind of equation (a when-equation, an assert), not one that reads `homotopy`. A
coefficient that reads only parameters is evaluated with the values known before the
simulation (`known`): 0 or not finite is no edge (MSL MultiBody's axis `n = {0, 0, 1}`:
`n[1] * der(phi)`), an unknown value one through a coefficient that reads unknowns, tried by its
magnitude at the start values (`start`; 0 where it does not evaluate, or reads a variable
without a given start: an unknown's default start is not its value; 0 at given starts is no
edge, a division by 0 at the initialization). A
coefficient 0 by its form (`isZeroConstant`, in a form the folding misses: MultiBody's
`R.T[1,1] = from_Q(Q, angularVelocity2(Q, der(Q)))[1][1]` reads `der(Q)` only in the record's
other field) is no edge, nor a piecewise one 0 in a branch that reads unknowns (`isPiecewise`: MSL Media's
smooth state transition, 1 in one branch and 0 in another), nor one 0 or not finite with `simValues` (the
simulation code's values of the variables a function call defines, `functionDefinedValues`) for the
variables it reads: MSL Spice3's diodes without a junction capacitance, `icap = m_dCap * der(v)` with
`(cc, m_dCap) = diodeNoBypassCode(...)` 0, solved for der(v), divided by 0.
"""
function derivativeEdges(model::IR.Model, partition::IR.Partition, unknowns::Set{Symbol},
                         functions::Dict{Symbol, IR.ModelicaFunction}, known::StartValues,
                         start::StartValues;
                         simValues::Dict{Symbol, Tuple{IR.Expression, Int}} = Dict{Symbol, Tuple{IR.Expression, Int}}())::Dict{Symbol, Vector{DerivativeEdge}}
  local edges = Dict{Symbol, Vector{DerivativeEdge}}(d => DerivativeEdge[] for d in unknowns)
  for (i, eq) in enumerate(partition.equations)
    (isDerivativeLink(eq) || !(eq isa Union{IR.SCALAR_EQUATION, IR.IF_EQUATION}) || readsHomotopy(eq)) && continue
    local read = Set{Symbol}()
    IR.foreachExp(x -> (x isa IR.CREF && x.cref.name in unknowns && push!(read, x.cref.name); nothing), eq)
    for d in read
      local others = sort!(collect(setdiff(read, (d,))))
      if !(eq isa IR.SCALAR_EQUATION)
        push!(edges[d], DerivativeEdge(i, nothing, true, Inf, others))
        continue
      end
      local c = partialDerivative(eq, d, functions; derivativeCalls = true)
      (c === nothing || isZeroConstant(c)) && continue
      local linear = true
      IR.foreachExp(x -> (x isa IR.CREF && x.cref.name === d && (linear = false); nothing), c)
      linear || continue
      local value = readsNoUnknown(c, model.variables) ? evaluate(c, known) : nothing
      (value !== nothing && (value == 0 || !isfinite(value))) && continue
      if value !== nothing
        push!(edges[d], DerivativeEdge(i, real(value), true, Inf, others))
        continue
      end
      isPiecewise(c) && continue
      if !isempty(simValues)
        local withValues = IR.mapExp(x -> (x isa IR.CREF && haskey(simValues, x.cref.name)) ? first(simValues[x.cref.name]) : x, c)
        local byValues = readsNoUnknown(withValues, model.variables) ? evaluate(withValues, known) : nothing
        (byValues !== nothing && (byValues == 0 || !isfinite(byValues))) && continue
      end
      local atStart = evaluate(c, start)
      local given = atStart !== nothing && isfinite(atStart) && readsGivenStarts(c, model.variables)
      (given && abs(atStart) <= ZERO_PIVOT) && continue
      push!(edges[d], DerivativeEdge(i, c, false, given ? abs(atStart) : 0.0, others))
    end
  end
  return edges
end

"""
    matchDerivatives(edges) -> Dict{Symbol, DerivativeEdge}

A matching of the derivative unknowns to equations (each equation for one unknown): first over
the edges with constant coefficients, then augmented, so an equation is solved through a
coefficient that reads unknowns only for an unknown no constant one can determine, and only an
equation whose other derivative unknowns the first matching determined (not a block of
derivatives coupled through such coefficients: a quaternion's `w = angularVelocity2(Q, der(Q))`
divided one equation at a time by components that pass 0; such a block stays the solver's);
those tried by their magnitude at the start values, the largest first (pivoting).
"""
function matchDerivatives(edges::Dict{Symbol, Vector{DerivativeEdge}})::Dict{Symbol, DerivativeEdge}
  local unknownOf = Dict{Int, Symbol}()
  local matched = Dict{Symbol, DerivativeEdge}()
  local determined = Set{Symbol}()
  foreach(list -> sort!(list; by = e -> -e.pivot), values(edges))
  local order = sort(collect(keys(edges)))
  foreach(d -> augmentDerivative!(d, edges, unknownOf, matched, determined, true), order)
  union!(determined, keys(matched))
  foreach(d -> haskey(matched, d) || augmentDerivative!(d, edges, unknownOf, matched, determined, false), order)
  return matched
end

"""
    augmentDerivative!(d, edges, unknownOf, matched, determined, constantOnly) -> Bool

An augmenting path from the unknown `d` (depth first, with an explicit stack: alias chains can be
long), over constant edges only or over the edges `matchDerivatives` allows; the matching
(`unknownOf`, `matched`) flipped along it. Whether one was found.
"""
function augmentDerivative!(d::Symbol, edges::Dict{Symbol, Vector{DerivativeEdge}}, unknownOf::Dict{Int, Symbol},
                            matched::Dict{Symbol, DerivativeEdge}, determined::Set{Symbol}, constantOnly::Bool)::Bool
  local allowed = e::DerivativeEdge -> (!constantOnly || e.constant) && (e.constant || all(in(determined), e.others))
  local visited = Set{Int}()
  local stack = Tuple{Symbol, Int}[(d, 1)]
  local path = Tuple{Symbol, DerivativeEdge}[]
  while !isempty(stack)
    local (u, k) = stack[end]
    local list = edges[u]
    local advanced = false
    while k <= length(list)
      local e = list[k]
      k += 1
      (allowed(e) && !(e.equation in visited)) || continue
      push!(visited, e.equation)
      stack[end] = (u, k)
      push!(path, (u, e))
      local other = get(unknownOf, e.equation, nothing)
      if other === nothing
        for (v, edge) in path
          unknownOf[edge.equation] = v
          matched[v] = edge
        end
        return true
      end
      push!(stack, (other, 1))
      advanced = true
      break
    end
    if !advanced
      pop!(stack)
      isempty(path) || pop!(path)
    end
  end
  return false
end

"""
    solvedValue(eq, d, coefficient) -> Expression

The equation `eq`, affine in `d` (its residual `r = c * d + r0`), solved for `d`: `-r0 / c`, by
a literal `c` as the product with its reciprocal (no integer division for the code generators'
symbolic arithmetic to make rationals of).
"""
function solvedValue(eq::IR.SCALAR_EQUATION, d::Symbol, coefficient::IR.Expression)::IR.Expression
  local zero = x -> (x isa IR.CREF && x.cref.name === d) ? ZERO : x
  local r0 = minus(foldTree(IR.mapExp(zero, eq.lhs)), foldTree(IR.mapExp(zero, eq.rhs)))
  if isLiteral(coefficient)
    local k = 1.0 / literalValue(coefficient)
    return foldTree(k == -1.0 ? r0 : times(negate(r0), real(k)))
  end
  return foldTree(divide(negate(r0), coefficient))
end

"Above this many nodes a solved derivative's value is not substituted: it stays a reference (no copies of large expressions)."
const SUBSTITUTED_VALUE_NODE_LIMIT = 40

"""
    resolvedValues(values) -> Dict{Symbol, Expression}

Each solved derivative's value with the other solved derivatives it reads replaced by theirs, in
turn (in dependency order, no recursion: alias chains can be long): a value in variables that are
not solved derivatives. Derivatives that depend on each other in a cycle (`der(a) + der(b) = f`,
`der(a) - der(b) = g` solved one each), and those that read one, are not resolved: left out,
they stay unknowns. A value larger than `SUBSTITUTED_VALUE_NODE_LIMIT` nodes stays a reference
in the values that read it.
"""
function resolvedValues(values::Dict{Symbol, IR.Expression})::Dict{Symbol, IR.Expression}
  local reads = Dict{Symbol, Set{Symbol}}()
  for (d, value) in values
    local r = Set{Symbol}()
    IR.foreachExp(x -> (x isa IR.CREF && haskey(values, x.cref.name) && push!(r, x.cref.name); nothing), value)
    reads[d] = r
  end
  local readers = Dict{Symbol, Vector{Symbol}}(d => Symbol[] for d in keys(values))
  foreach(((d, r),) -> foreach(x -> push!(readers[x], d), r), reads)
  local pending = Dict{Symbol, Int}(d => length(r) for (d, r) in reads)
  local ready = sort!([d for (d, n) in pending if n == 0])
  local resolved = Dict{Symbol, IR.Expression}()
  while !isempty(ready)
    local d = popfirst!(ready)
    resolved[d] = foldTree(IR.mapExp(x -> (x isa IR.CREF && haskey(resolved, x.cref.name)) ?
                                          substitutable(x, resolved[x.cref.name]) : x, values[d]))
    for r in readers[d]
      pending[r] -= 1
      pending[r] == 0 && push!(ready, r)
    end
  end
  return resolved
end

"`value` in the place of the reference `ref` where it is small enough to copy (`SUBSTITUTED_VALUE_NODE_LIMIT`), else `ref`."
substitutable(ref::IR.CREF, value::IR.Expression)::IR.Expression =
  expressionLargerThan(value, SUBSTITUTED_VALUE_NODE_LIMIT) ? ref : value

"""
    solveThroughSolved!(solved, equationOf, edges, partition, unknowns, functions, variables, start)

Two derivative unknowns whose solved values read each other (a cycle `resolvedValues` would leave the solver's)
solved by substitution: the one whose equation, the other's value put in, reads neither the other nor another
unknown in a cycle, and is linear in it with a coefficient that does not vanish at the start values, solved from
that; the other's value then resolves. Then an unknown the matching left (an equation through a coefficient that
reads unknowns only once its other derivatives are determined by constant ones, `matchDerivatives`) the same way
from a free equation. MSL Media's lumped volumes with p and h the states (StateSelect.prefer): der(h) from the
specific internal energy reads der(p), der(p) from the density's derivative (`rho_ph_der`, linear in its derivative
inputs: `derivativeFunctionCall`) reads der(h); left the solver's, der(p) of R134a2's volumes jumped at the dew line
(rho_ph_der's two branches) and the steps fell to 1e-16. Not where a branch of the coefficient is 0
(`isPiecewise`), nor where it is 0 or does not evaluate at the start.
"""
function solveThroughSolved!(solved::Dict{Symbol, IR.Expression}, equationOf::Dict{Symbol, Int},
                             edges::Dict{Symbol, Vector{DerivativeEdge}}, partition::IR.Partition, unknowns::Set{Symbol},
                             functions::Dict{Symbol, IR.ModelicaFunction}, variables::IR.VariableSet, start::StartValues)::Nothing
  local readsOf = v -> (local r = Set{Symbol}(); IR.foreachExp(x -> (x isa IR.CREF && x.cref.name in unknowns && push!(r, x.cref.name); nothing), v); r)
  #= `d` from equation `i` with the values of `others` put in: its value, or nothing =#
  local solveFrom = (d, i, others) -> begin
    local eq = partition.equations[i]
    eq isa IR.SCALAR_EQUATION || return nothing
    local insert = x -> (x isa IR.CREF && x.cref.name in others) ? solved[x.cref.name] : x
    local substituted = IR.SCALAR_EQUATION(IR.mapExp(insert, eq.lhs), IR.mapExp(insert, eq.rhs), eq.ty, eq.source, eq.attributes)
    local reads = readsOf(substituted)
    (d in reads && !any(o -> o in reads, others)) || return nothing
    local c = partialDerivative(substituted, d, functions; derivativeCalls = true)
    (c === nothing || isZeroConstant(c) || isPiecewise(c)) && return nothing
    local linear = true
    IR.foreachExp(x -> (x isa IR.CREF && x.cref.name === d && (linear = false); nothing), c)
    linear || return nothing
    local atStart = evaluate(c, start)
    (atStart !== nothing && isfinite(atStart) && abs(atStart) > ZERO_PIVOT) || return nothing
    return solvedValue(substituted, d, c)
  end
  #= cycles of two =#
  for a in sort!(collect(keys(solved)))
    local ra = readsOf(solved[a])
    for b in sort!(collect(ra))
      (b !== a && haskey(solved, b) && a in readsOf(solved[b])) || continue
      local value = solveFrom(a, equationOf[a], Set([b]))
      value === nothing && continue
      solved[a] = value
      break
    end
  end
  #= unknowns left =#
  local used = Set{Int}(values(equationOf))
  for d in sort!([u for u in unknowns if !haskey(solved, u)])
    for e in edges[d]
      (e.coefficient === nothing || e.equation in used || !all(o -> haskey(solved, o), e.others)) && continue
      local value = solveFrom(d, e.equation, Set(e.others))
      value === nothing && continue
      solved[d] = value
      equationOf[d] = e.equation
      push!(used, e.equation)
      break
    end
  end
  return nothing
end

"""
    solveDerivatives!(model, partition; known) -> Vector{Symbol}

The derivative unknowns of a reduced partition solved for and substituted: each assigned an
equation that determines it (`matchDerivatives`), that equation solved for it (`d = -r0 / c`,
through a coefficient that reads unknowns only where no constant one determines it: MSL Spice3's
diode, `icap = m_dCap * (der(pin) - der(n.v))`, is `der(pin) = icap / m_dCap + der(n.v)`), its
value substituted into the other equations (the state's link, `\$STATEDER.x = ...`, too). A
derivative then occurs only in its own equation and the code generators compute it from the
others: the unknowns an algebraic loop keeps are the model's (the diodes' `icap`, under the
integrator's error control), not the derivatives (left out of it). An unknown an if-equation
determines stays, so do derivatives in a cycle (`resolvedValues`). `known`: the simulation code's
values the plan was made with (`Plan`'s), of which those of variables a function call defines
evaluate the coefficients (`derivativeEdges`). The unknowns so solved for, in order.
"""
function solveDerivatives!(model::IR.Model, partition::IR.Partition;
                           known::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}())::Vector{Symbol}
  local unknowns = Set{Symbol}(v.name for v in model.variables if isDerivativeUnknown(v))
  isempty(unknowns) && return Symbol[]
  local functions = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in model.functions)
  local parameters = StartValues(model.variables, model.functions; parametersOnly = true)
  local start = StartValues(model.variables, model.functions; equations = vcat(partition.equations, partition.initialEquations))
  local edges = derivativeEdges(model, partition, unknowns, functions, parameters, start;
                                simValues = functionDefinedValues(partition, known))
  local matched = matchDerivatives(edges)
  local solved = Dict{Symbol, IR.Expression}()
  local equationOf = Dict{Symbol, Int}()
  for (d, e) in matched
    e.coefficient === nothing && continue
    solved[d] = solvedValue(partition.equations[e.equation]::IR.SCALAR_EQUATION, d, e.coefficient)
    equationOf[d] = e.equation
  end
  solveThroughSolved!(solved, equationOf, edges, partition, unknowns, functions, model.variables, start)
  isempty(solved) && return Symbol[]
  local resolved = resolvedValues(solved)
  isempty(resolved) && return Symbol[]
  local definitions = Dict{Int, Symbol}(equationOf[d] => d for d in keys(resolved))
  local substitute = x -> (x isa IR.CREF && haskey(resolved, x.cref.name)) ? substitutable(x, resolved[x.cref.name]) : x
  for (i, eq) in enumerate(partition.equations)
    local d = get(definitions, i, nothing)
    partition.equations[i] = d === nothing ? IR.mapEquation!(substitute, eq) :
      IR.SCALAR_EQUATION(IR.CREF(IR.ComponentRef(d, IR.T_REAL()), IR.T_REAL()), resolved[d], IR.T_REAL(), eq.source, eq.attributes)
  end
  return sort(collect(keys(resolved)))
end
