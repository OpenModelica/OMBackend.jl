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

#= Linear constraints among the states that the structure does not show: a combination of the
   partition's linear equations of literal coefficients (the current balances of connections and
   two-pins, the aliases) in which every other unknown cancels. =#

"""
    LinearRow

A linear equation over the continuous variables (by their index), as `sum(coefficients[v] * v) +
sum(c * k for (c, k) in known) = 0`: `known` its terms that read no continuous variable (literals,
parameters, discretes, time).
"""
mutable struct LinearRow
  coefficients::Dict{Int, Rational{Int}}
  known::Vector{Tuple{Rational{Int}, IR.Expression}}
end

LinearRow() = LinearRow(Dict{Int, Rational{Int}}(), Tuple{Rational{Int}, IR.Expression}[])

"""
    LINEAR_CONSTRAINT_FILL

The elimination of `linearConstraints` gives up where its rows grow past this many times their
entries at the start (and 100000): a fill-in that large means the linear equations are no network of
balances, and the pass would cost more than it can find.
"""
const LINEAR_CONSTRAINT_FILL = 20

"""
    knownRational(e, values) -> Union{Nothing, Rational{Int}}

The value of `e`, a literal or an expression of the parameters known before the simulation (`values`, a
`StartValues` of `parametersOnly`: what the code generators substitute, MSL Machines' `turnsRatio`), as a small
exact rational (`0.5` is 1//2, `-1` is -1//1; denominator at most 1000); `nothing` for another expression or a
value that is none (`0.1234567`).
"""
function knownRational(e::IR.Expression, values::StartValues)::Union{Nothing, Rational{Int}}
  e isa IR.INTEGER && return abs(e.value) < 2^31 ? Rational{Int}(e.value) : nothing
  local x = e isa IR.REAL ? e.value : evaluate(e, values)
  x === nothing && return nothing
  (isfinite(x) && abs(x) < 2.0^31) || return nothing
  isinteger(x) && return Rational{Int}(Int(x))
  local r = rationalize(Int, x; tol = 0.0)
  return denominator(r) <= 1000 ? r : nothing
end

"Whether `e` reads a continuous variable (`continuousReads!` through `resolve`)."
readsContinuous(e::IR.Expression, resolve::Resolver)::Bool = !isempty(continuousReads!(Symbol[], e, resolve))

"""
    linearTerms!(row, e, scale, index, resolve, values) -> Bool

`scale * e` added to `row` where `e` is linear in the continuous variables with known coefficients
(sums, differences, negation, a product or quotient by a literal or a known parameter's value: `knownRational`);
a term that reads no continuous variable is a known term. `false` for another expression (the row is then
incomplete and not used).
"""
function linearTerms!(row::LinearRow, e::IR.Expression, scale::Rational{Int}, index::Dict{Symbol, Int},
                      resolve::Resolver, values::StartValues)::Bool
  if e isa IR.CREF && haskey(index, e.cref.name)
    local v = index[e.cref.name]
    row.coefficients[v] = get(row.coefficients, v, 0 // 1) + scale
    return true
  elseif !readsContinuous(e, resolve)
    push!(row.known, (scale, e))
    return true
  elseif e isa IR.UNARY && e.op === IR.UMINUS
    return linearTerms!(row, e.exp, -scale, index, resolve, values)
  elseif e isa IR.BINARY && e.op in (IR.ADD, IR.SUB)
    return linearTerms!(row, e.exp1, scale, index, resolve, values) &&
           linearTerms!(row, e.exp2, e.op === IR.ADD ? scale : -scale, index, resolve, values)
  elseif e isa IR.BINARY && e.op === IR.MUL
    local a = readsContinuous(e.exp1, resolve) ? nothing : knownRational(e.exp1, values)
    a === nothing || return linearTerms!(row, e.exp2, scale * a, index, resolve, values)
    local b = readsContinuous(e.exp2, resolve) ? nothing : knownRational(e.exp2, values)
    b === nothing || return linearTerms!(row, e.exp1, scale * b, index, resolve, values)
    return false
  elseif e isa IR.BINARY && e.op === IR.DIV
    local b = readsContinuous(e.exp2, resolve) ? nothing : knownRational(e.exp2, values)
    (b === nothing || iszero(b)) && return false
    return linearTerms!(row, e.exp1, scale / b, index, resolve, values)
  end
  return false
end

"""
    maximumMatching(rows, columns) -> Int

The size of a maximum matching of `rows` to `columns` (each row's entries among `columns`): augmenting paths
searched without recursion (deep paths overflow the stack otherwise), after a greedy start.
"""
function maximumMatching(rows::Vector{Vector{Int}}, nColumns::Int)::Int
  local rowOf = zeros(Int, nColumns)
  local colOf = zeros(Int, length(rows))
  for (r, cols) in enumerate(rows), c in cols
    rowOf[c] == 0 || continue
    rowOf[c] = r
    colOf[r] = c
    break
  end
  local visited = zeros(Int, nColumns)
  local stamp = 0
  for root in eachindex(rows)
    colOf[root] == 0 || continue
    stamp += 1
    #= the path as (row, next entry to try) frames; parent[c] the row that reached column c =#
    local stack = Tuple{Int, Int}[(root, 1)]
    local via = Dict{Int, Int}()
    local found = 0
    while !isempty(stack) && found == 0
      local (r, k) = stack[end]
      if k > length(rows[r])
        pop!(stack)
        continue
      end
      stack[end] = (r, k + 1)
      local c = rows[r][k]
      visited[c] == stamp && continue
      visited[c] = stamp
      via[c] = r
      if rowOf[c] == 0
        found = c
      else
        push!(stack, (rowOf[c], 1))
      end
    end
    found == 0 && continue
    #= the path flipped from its free column back to the root =#
    local c = found
    while true
      local r = via[c]
      local previous = colOf[r]
      rowOf[c] = r
      colOf[r] = c
      r == root && break
      c = previous
    end
  end
  return count(!=(0), colOf)
end

"""
    eliminate!(rows, isState) -> Union{Nothing, Tuple{Vector{Int}, Int}}

Gaussian elimination of the variables that are no states from `rows`, exact (rationals), column by column
(those in fewest rows first, the pivot the row of fewest entries). The rows no pivot took are the dependent
ones: those that still have entries (states only) the constraints, by their index, and the number of those
that cancelled out. `nothing` where the fill grows past `LINEAR_CONSTRAINT_FILL` or a rational overflows.
"""
function eliminate!(rows::Vector{LinearRow}, isState::BitVector)::Union{Nothing, Tuple{Vector{Int}, Int}}
  local active = trues(length(rows))
  local rowsOf = Dict{Int, Set{Int}}()
  local entries = 0
  for (r, row) in enumerate(rows), v in keys(row.coefficients)
    entries += 1
    isState[v] || push!(get!(() -> Set{Int}(), rowsOf, v), r)
  end
  local limit = LINEAR_CONSTRAINT_FILL * entries + 100_000
  local order = sort!(collect(keys(rowsOf)); by = v -> (length(rowsOf[v]), v))
  try
    for v in order
      local holding = Int[r for r in rowsOf[v] if active[r]]
      isempty(holding) && continue
      local p = holding[argmin(Int[length(rows[r].coefficients) for r in holding])]
      active[p] = false
      local pivot = rows[p]
      local pv = pivot.coefficients[v]
      for r in holding
        r == p && continue
        local row = rows[r]
        local factor = row.coefficients[v] / pv
        for (w, a) in pivot.coefficients
          local before = length(row.coefficients)
          local value = get(row.coefficients, w, 0 // 1) - factor * a
          if iszero(value)
            delete!(row.coefficients, w)
            isState[w] || delete!(rowsOf[w], r)
          else
            row.coefficients[w] = value
            isState[w] || push!(rowsOf[w], r)
          end
          entries += length(row.coefficients) - before
        end
        foreach(((c, k),) -> push!(row.known, (-factor * c, k)), pivot.known)
      end
      entries > limit && return nothing
    end
  catch err
    err isa OverflowError || rethrow()
    return nothing
  end
  local constraints = Int[r for r in eachindex(rows) if active[r] && !isempty(rows[r].coefficients)]
  return (constraints, count(r -> active[r] && isempty(rows[r].coefficients), eachindex(rows)))
end

"""
    constraintEquation(row, variables, names) -> SCALAR_EQUATION

The constraint `row` as an equation `sum(c * x) + sum(c * k) = 0` over the variables `names` (by index).
"""
function constraintEquation(row::LinearRow, variables::IR.VariableSet, names::Vector{Symbol})::IR.SCALAR_EQUATION
  local terms = IR.Expression[]
  for v in sort!(collect(keys(row.coefficients)))
    local ref = IR.CREF(variables[names[v]].cref, IR.T_REAL())
    push!(terms, times(real(Float64(row.coefficients[v])), ref))
  end
  for (c, k) in row.known
    push!(terms, times(real(Float64(c)), k))
  end
  return IR.SCALAR_EQUATION(foldl(plus, terms), ZERO, IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
end

"""
    linearConstraints(model, partition) -> Vector{Tuple{Int, SCALAR_EQUATION}}

The linear constraints among the states of `partition` that its structure does not show, each with the
equation it replaces. The partition's scalar Real equations that are linear with known coefficients
in the continuous variables (`linearTerms!`; those that read a variable that is no state) are eliminated
for those variables (`eliminate!`); a row left over the states only is a constraint, implied by the
equations it combines, the row itself one of them. Where the elimination finds no more dependent rows than
a maximum matching of the rows to their variables leaves (structural dependencies, which Pantelides' algorithm
finds), nothing: a constraint is hidden only where the coefficients cancel; of the constraints, those whose row
written in makes that structural deficiency grow (the others the structure shows already). Three chokes feeding a
diode bridge whose DC side is not grounded: the current balances of the bridge's nodes add up to
`i1 + i2 + i3 = 0` (OpenModelica differentiates it too, and keeps two of the chokes' currents as
states), while every balance alone reads a diode's current.
"""
function linearConstraints(model::IR.Model, partition::IR.Partition)::Vector{Tuple{Int, IR.SCALAR_EQUATION}}
  local none = Tuple{Int, IR.SCALAR_EQUATION}[]
  local discretes = discreteTargets(partition)
  local names = Symbol[v.name for v in model.variables if isContinuousVariable(v, discretes)]
  local index = Dict{Symbol, Int}(n => i for (i, n) in enumerate(names))
  local isState = falses(length(names))
  for v in model.variables
    (v.kind isa IR.STATE && haskey(index, v.name)) && (isState[index[v.name]] = true)
  end
  any(isState) || return none
  local resolve = Resolver(names)
  local values = StartValues(model.variables, model.functions; parametersOnly = true)
  local rows = LinearRow[]
  local origin = Int[]
  for (i, eq) in enumerate(partition.equations)
    (eq isa IR.SCALAR_EQUATION && eq.ty isa IR.T_REAL && !(eq.lhs isa IR.TUPLE)) || continue
    local row = LinearRow()
    #= an equation whose reads the structure does not handle (an array constructor over a range that is not
       constant: MSL Media's ideal gas mixtures) is no linear row; reduceIndex! decides about it =#
    local linear = try
      linearTerms!(row, eq.lhs, 1 // 1, index, resolve, values) && linearTerms!(row, eq.rhs, -1 // 1, index, resolve, values)
    catch err
      err isa NotHandled || rethrow()
      false
    end
    linear || continue
    filter!(((v, a),) -> !iszero(a), row.coefficients)
    #= a row of states only is a constraint the structure shows =#
    any(v -> !isState[v], keys(row.coefficients)) || continue
    push!(rows, row)
    push!(origin, i)
  end
  isempty(rows) && return none
  local unknownEntries = Vector{Int}[Int[v for v in keys(row.coefficients) if !isState[v]] for row in rows]
  local structural = length(rows) - maximumMatching(unknownEntries, length(names))
  local eliminated = eliminate!(rows, isState)
  eliminated === nothing && return none
  local (constraints, cancelled) = eliminated
  length(constraints) + cancelled > structural || return none
  #= only the constraints the structure does not show: each written in (its row then reads no unknown) where that
     makes the structural deficiency grow; one the structure shows already (a rigid coupling's phi1 = phi2) is
     left as written =#
  local hidden = Int[]
  for r in constraints
    local entries = unknownEntries[r]
    unknownEntries[r] = Int[]
    local deficiency = length(rows) - maximumMatching(unknownEntries, length(names))
    if deficiency > structural
      structural = deficiency
      push!(hidden, r)
    else
      unknownEntries[r] = entries
    end
  end
  return Tuple{Int, IR.SCALAR_EQUATION}[(origin[r], constraintEquation(rows[r], model.variables, names)) for r in hidden]
end

"""
    exposeLinearConstraints!(model, partition) -> Vector{Tuple{Int, Equation}}

The hidden linear constraints of `partition` (`linearConstraints`, the switch `OMBACKEND_LINEAR_CONSTRAINTS`)
written into it, each in place of an equation it combines: the equations keep their solutions, and the
structure now shows the constraint, which Pantelides' algorithm differentiates. The replaced equations by
their index, to undo (`undoLinearConstraints!`).
"""
function exposeLinearConstraints!(model::IR.Model, partition::IR.Partition)::Vector{Tuple{Int, IR.Equation}}
  local replaced = Tuple{Int, IR.Equation}[]
  envSwitch("OMBACKEND_LINEAR_CONSTRAINTS") || return replaced
  for (i, eq) in linearConstraints(model, partition)
    push!(replaced, (i, partition.equations[i]))
    partition.equations[i] = eq
  end
  return replaced
end

"The equations `exposeLinearConstraints!` replaced written back."
function undoLinearConstraints!(partition::IR.Partition, replaced::Vector{Tuple{Int, IR.Equation}})::Nothing
  foreach(((i, eq),) -> (partition.equations[i] = eq), replaced)
  return nothing
end
