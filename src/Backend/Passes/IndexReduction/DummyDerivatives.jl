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

#= Dummy derivatives (S. E. Mattsson, G. Söderlind, Index reduction in differential-algebraic
   equations using dummy derivatives, SIAM J. Sci. Comput. 14(3), 1993) with Modelica's state
   selection (MLS 4.8.8.1): which derivatives become algebraic, so the states left are the ones
   StateSelect asks for, a choice that is regular at the start. =#

"""
    declaredClasses(s, partition, variables, natural; dynamic) -> Vector{StateSelect}

Each structure variable's StateSelect as declared: its `stateSelect` attribute, DEFAULT without
one (and for a variable the model does not declare, `\$DER.x`). With `dynamic` (the dynamic
state sets' ranking), a variable without one is NEVER where only index reduction differentiates
it (no equation of the model reads its derivative, `natural`): OpenModelica's rule
(`algebraicState` in its IndexReduction.mo), by which a MultiBody body keeps its quaternion and
angular velocity (StateSelect.avoid), not the rotation matrix entries of its frames; and no
higher than the highest StateSelect of its alias set where the set has one (MSL's Body declares
`r_0` avoid, its alias `frame_a.r_0`, whose derivative the body's equations read, none: the
body's position is avoided as its quaternion is; an alias of an `always` variable stays below
it).
"""
function declaredClasses(s::Structure, partition::IR.Partition, variables::IR.VariableSet, natural::BitVector;
                         dynamic::Bool = false)::Vector{IR.StateSelect}
  local classes = fill(IR.DEFAULT, length(s.variables))
  local root = dynamic ? aliasRoots(s, partition) : Int[]
  local explicit = Dict{Int, IR.StateSelect}()
  for v in eachindex(s.variables)
    local var = get(variables, s.variables[v], nothing)
    (var === nothing || var.attributes.stateSelect === nothing) && continue
    classes[v] = var.attributes.stateSelect
    dynamic && (explicit[root[v]] = max(get(explicit, root[v], IR.NEVER), classes[v]))
  end
  dynamic || return classes
  for v in eachindex(s.variables)
    local var = get(variables, s.variables[v], nothing)
    (var === nothing || var.attributes.stateSelect !== nothing) && continue
    local own = (s.derivative[v] != 0 && natural[s.derivative[v]]) ? IR.DEFAULT : IR.NEVER
    classes[v] = min(get(explicit, root[v], own), own)
  end
  return classes
end

"""
    dummyClass(s, declared, c) -> StateSelect

How much the candidate `c` (a derivative) should stay a state's derivative: making it a dummy
derivative makes the variable it is the derivative of algebraic, so its class is that
variable's StateSelect (`declared`, `declaredClasses`); NEVER for a derivative of a
derivative: `\$DER.x` is a state only where no declared variable can be (a declared `v = der(x)`
of StateSelect.avoid before it, as OpenModelica, which has no such candidate, chooses).
"""
function dummyClass(s::Structure, declared::Vector{IR.StateSelect}, c::Int)::IR.StateSelect
  local a = s.antiderivative[c]
  a == 0 && return IR.DEFAULT
  s.antiderivative[a] != 0 && return IR.NEVER
  return declared[a]
end

"""
    aliasClasses(s, partition, variables, declared) -> Vector{StateSelect}

Each structure variable's StateSelect as OpenModelica reads it after its alias elimination: the
highest one declared in its alias set (`declared`; the variables `a = b`, `a = -b` equations
join: RollingWheelSet's `der_theta1(stateSelect = always)` and its joint's `der_theta1`);
NEVER for a set of derivative variables only (`\$DER.x` is a state only where no declared
variable can be).
"""
function aliasClasses(s::Structure, partition::IR.Partition, variables::IR.VariableSet,
                      declared::Vector{IR.StateSelect})::Vector{IR.StateSelect}
  local root = aliasRoots(s, partition)
  local best = Dict{Int, IR.StateSelect}()
  for v in eachindex(s.variables)
    (haskey(variables, s.variables[v]) && !startswith(String(s.variables[v]), '$')) || continue
    best[root[v]] = max(get(best, root[v], IR.NEVER), declared[v])
  end
  return IR.StateSelect[get(best, root[v], IR.NEVER) for v in eachindex(s.variables)]
end

"""
    aliasRoots(s, partition) -> Vector{Int}

Each structure variable's alias set, as a representative variable: the sets the alias equations
of `partition` (`a = b`, `a = -b`) join.
"""
function aliasRoots(s::Structure, partition::IR.Partition)::Vector{Int}
  local parent = collect(eachindex(s.variables))
  local find = function (v::Int)
    while parent[v] != v
      parent[v] = parent[parent[v]]
      v = parent[v]
    end
    return v
  end
  for eq in partition.equations
    local pair = aliasPair(eq)
    pair === nothing && continue
    local (a, b) = (get(s.index, first(pair), 0), get(s.index, last(pair), 0))
    (a == 0 || b == 0) && continue
    local (ra, rb) = (find(a), find(b))
    ra == rb || (parent[ra] = rb)
  end
  return Int[find(v) for v in eachindex(s.variables)]
end

"The two variables of an alias equation (`a = b`, `a = -b`, `-a = b`); `nothing` for another equation."
function aliasPair(eq::IR.Equation)::Union{Nothing, Tuple{Symbol, Symbol}}
  eq isa IR.SCALAR_EQUATION || return nothing
  local name = e::IR.Expression -> e isa IR.CREF ? e.cref.name :
    (e isa IR.UNARY && e.op === IR.UMINUS && e.exp isa IR.CREF) ? e.exp.cref.name : nothing
  local (a, b) = (name(eq.lhs), name(eq.rhs))
  return (a === nothing || b === nothing) ? nothing : (a, b)
end

"""
    candidateClass(s, aliasClass, declared, c) -> Int

The order of the candidate `c` among the dummy derivatives: by the StateSelect of the alias set
of the variable it is the derivative of (`aliasClasses`), within that by the variable's own
StateSelect (`dummyClass`: of two aliases the one declared `always` stays the state).
"""
function candidateClass(s::Structure, aliasClass::Vector{IR.StateSelect}, declared::Vector{IR.StateSelect}, c::Int)::Int
  local a = s.antiderivative[c]
  local set = a == 0 ? IR.DEFAULT : aliasClass[a]
  return 10 * Int(set) + Int(dummyClass(s, declared, c))
end

"Whether the candidate `c` is the derivative of a variable of `fixed = true`."
function fixedStart(s::Structure, variables::IR.VariableSet, c::Int)::Bool
  local a = s.antiderivative[c]
  local var = a == 0 ? nothing : get(variables, s.variables[a], nothing)
  return var !== nothing && var.attributes.fixed isa IR.BOOLEAN && var.attributes.fixed.value
end

"""
    rowEquation(s, r, partition, differentiated, functions) -> Equation

The equation of row `r`: the partition's equation (inlined, as the structure reads it), or its
derivative of the row's order; of an if-equation the row's position as one scalar equation
(`ifPosition`: its Jacobian entries are those of the branch the start values take, MSL
SMEE_Generator's ramp `if time < 0.1 then ... else ...`).
"""
function rowEquation(s::Structure, r::Int, partition::IR.Partition, differentiated::Dict{Int, Vector{IR.Equation}},
                     functions::Dict{Symbol, IR.ModelicaFunction})::IR.Equation
  local k = rowOrder(s, r)
  local eq = s.rows[r].equation
  local e = k == 0 ? inlineEquation(partition.equations[eq], functions) : differentiated[eq][k]
  return e isa IR.IF_EQUATION ? ifPosition(e, positionIndex(s, r)) : e
end

"""
    positionIndex(s, r) -> Int

The index of row `r`'s position in its equation's derivative of the row's order, which holds the
positions differentiated at least that often, in order (`positionsOf`); the position itself at
order 0.
"""
function positionIndex(s::Structure, r::Int)::Int
  local k = rowOrder(s, r)
  local (eq, p) = (s.rows[r].equation, s.rows[r].position)
  k == 0 && return p
  local n = 0
  for r0 in eachindex(s.rows)
    (s.origin[r0] == 0 && s.rows[r0].equation == eq && s.rows[r0].position <= p) || continue
    rowOrder(s, last(derivativeChain(s, r0))) >= k && (n += 1)
  end
  return n
end

"""
    ifPosition(eq, k) -> Equation

The equations of an if-equation at position `k` as one scalar equation, its residual the
if-expression of the branches' (`if c1 then lhs1 - rhs1 elseif ... else lhs - rhs end = 0`);
the if-equation itself where a position holds another kind of equation.
"""
function ifPosition(eq::IR.IF_EQUATION, k::Int)::IR.Equation
  local es = positionEquations(eq, k)
  all(e -> e isa IR.SCALAR_EQUATION, es) || return eq
  local residual = e::IR.SCALAR_EQUATION -> minus(e.lhs, e.rhs)
  local value = foldr((i, rest) -> IR.IF_EXPRESSION(eq.branches[i].condition, residual(es[i]), rest, IR.T_REAL()),
                      eachindex(eq.branches); init = residual(es[end]))
  return IR.SCALAR_EQUATION(value, ZERO, IR.T_REAL(), eq.source, eq.attributes)
end

"""
    partialDerivative(eq, name, functions) -> Union{Nothing, Expression}

The partial derivative of a scalar equation's residual (`lhs - rhs`) by the variable `name`;
`nothing` for an if-equation or where differentiation does not handle the equation.
"""
function partialDerivative(eq::IR.Equation, name::Symbol, functions::Dict{Symbol, IR.ModelicaFunction};
                           derivativeCalls::Bool = false)::Union{Nothing, IR.Expression}
  eq isa IR.SCALAR_EQUATION || return nothing
  local seeds = Seeds(n -> n === name ? ONE : nothing, 0.0, functions, Symbol[], derivativeCalls)
  return try
    minus(differentiate(eq.lhs, seeds), differentiate(eq.rhs, seeds))
  catch err
    err isa NotHandled || rethrow()
    nothing
  end
end

"""
    jacobianEntry(eq, name, functions, values) -> Union{Nothing, Float64}

`partialDerivative` at the start values; `nothing` where it does not evaluate (an if-equation,
an unknown value).
"""
function jacobianEntry(eq::IR.Equation, name::Symbol, functions::Dict{Symbol, IR.ModelicaFunction}, values::StartValues)::Union{Nothing, Float64}
  local d = partialDerivative(eq, name, functions)
  return d === nothing ? nothing : evaluate(d, values)
end

"Whether `e` reads no continuous variable of `s` (nor `time`): a constant entry of a Jacobian."
function isConstantEntry(e::IR.Expression, s::Structure)::Bool
  local constant = true
  IR.foreachExp(e) do x
    (x isa IR.CREF && (x.cref.name === :time || haskey(s.index, x.cref.name))) && (constant = false)
    nothing
  end
  return constant
end

"""
    components(rows, columns, incidence) -> Vector{Tuple{Vector{Int}, Vector{Int}}}

The independent blocks of a rows-by-columns structure (connected components of its bipartite
graph): selection can be made block by block.
"""
function components(rows::Vector{Int}, columns::Vector{Int}, reads::Dict{Int, Vector{Int}})::Vector{Tuple{Vector{Int}, Vector{Int}}}
  local parent = Dict{Int, Int}()
  local find(x::Int)::Int = (while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end; x)
  local key(c::Int)::Int = -c                        # columns as negative keys next to the rows
  foreach(r -> parent[r] = r, rows)
  foreach(c -> parent[key(c)] = key(c), columns)
  local columnSet = Set(columns)
  for r in rows, c in reads[r]
    c in columnSet || continue
    local (a, b) = (find(r), find(key(c)))
    a == b || (parent[a] = b)
  end
  local groups = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
  foreach(r -> push!(get!(() -> (Int[], Int[]), groups, find(r))[1], r), rows)
  foreach(c -> push!(get!(() -> (Int[], Int[]), groups, find(key(c)))[2], c), columns)
  return collect(values(groups))
end

"""
    choiceClasses(s, columns, aliasClass, declared, natural) -> Dict{Int, Int}

The order of the candidate `columns` by class in `chooseDummies` (the lower first a dummy): by
`candidateClass`, within one the derivatives the model's own equations read (`natural`) after
the others.
"""
choiceClasses(s::Structure, columns::Vector{Int}, aliasClass::Vector{IR.StateSelect}, declared::Vector{IR.StateSelect},
              natural::BitVector)::Dict{Int, Int} =
  Dict{Int, Int}(c => 2 * candidateClass(s, aliasClass, declared, c) + (natural[c] ? 1 : 0) for c in columns)

"""
    chooseDummies(s, rows, columns, ...) -> Vector{Int}

`length(rows)` of `columns` making the Jacobian d rows / d columns regular: the dummy
derivatives of this level, by the module's priority rule (see `IndexReduction`). The columns
are tried by StateSelect (never, avoid, default, prefer; always only if nothing else
regularizes; of the alias set, then of the variable itself: `candidateClass`), within one the
derivatives the model's own equations do not read before those they read (`natural`: of a
variable the model gives an equation for its derivative, a state as OpenModelica keeps it; MSL
MultiPhase Rectifier's inductor currents, not the source's pin currents), then first the
columns with constant entries (`x = L * sin(phi)` keeps phi the state, as
OpenModelica), then by the largest pivot of the Jacobian at the start values (Gaussian
elimination with pivoting in the class's columns), among equals the derivative of a derivative
variable, then a variable without a fixed start, then the later variable (so the first
declared and the fixed stay states, as OpenModelica chooses). Where the start values do not
give a regular choice, their pivots are kept and the rest chosen at a generic point (a special
start, all currents 0, makes entries vanish): first one that keeps the relations explicit
equations give (`propagatedValues`, computed once in `generic`), then `genericValues`, else
structurally, those pivots first.
"""
function chooseDummies(s::Structure, rows::Vector{Int}, columns::Vector{Int}, partition::IR.Partition,
                       differentiated::Dict{Int, Vector{IR.Equation}}, variables::IR.VariableSet,
                       functions::Dict{Symbol, IR.ModelicaFunction}, values::StartValues,
                       classes::Dict{Int, Int}, natural::BitVector,
                       generic::Base.RefValue{Union{Nothing, StartValues}})::Vector{Int}
  #= within a class, first the derivatives of derivative variables (`\$DER.x` has no name of its own: a
     declared variable it equals, `w = der(phi)`, makes the better state), then of variables without
     a fixed start (of the aliases `I1.phi`, `I1.flange_a.phi`, whose entries tie, the one the user
     fixed stays the state, as OpenModelica), then the later declared =#
  local firstOrder = c -> (s.antiderivative[c] != 0 && s.antiderivative[s.antiderivative[c]] == 0) ? 1 : 0
  local order = sort(columns; by = c -> (classes[c], firstOrder(c), fixedStart(s, variables, c) ? 1 : 0, -baseVariable(s, c), -c))
  local atStart = numericChoice(s, rows, order, classes, partition, differentiated, functions, values; partial = true)
  (atStart !== nothing && length(atStart) == length(rows)) && return atStart
  #= a special start (all currents 0) can make the Jacobian singular there: the pivots it gives kept, the rest at a
     generic point that keeps the relations explicit equations give (`propagatedValues`), else at one that does not =#
  local forced = something(atStart, Int[])
  generic[] === nothing && (generic[] = propagatedValues(s, partition, differentiated, functions, values, natural))
  for point in (something(generic[]), genericValues(values))
    local numeric = numericChoice(s, rows, order, classes, partition, differentiated, functions, point; forced)
    numeric === nothing || return numeric
  end
  order = vcat(forced, filter(c -> !(c in forced), order))
  return structuralChoice(s, rows, order)
end

"""
    numericChoice(s, rows, order, classes, partition, differentiated, functions, values) -> Union{Nothing, Vector{Int}}

`chooseDummies`' choice by the Jacobian of the rows by the candidate columns at `values`:
Gaussian elimination with pivoting, the `forced` columns first, then the columns in the classes'
order (each class its constant columns first), the largest pivot of a class first; `nothing`
where an entry does not evaluate, a forced column has no pivot, or the classes give fewer pivots
than rows (with `partial`, the pivots found).
"""
function numericChoice(s::Structure, rows::Vector{Int}, order::Vector{Int}, classes::Dict{Int, Int}, partition::IR.Partition,
                       differentiated::Dict{Int, Vector{IR.Equation}}, functions::Dict{Symbol, IR.ModelicaFunction},
                       values::StartValues; forced::Vector{Int} = Int[], partial::Bool = false)::Union{Nothing, Vector{Int}}
  #= the nonzero entries (row, column, value), the rows' incidence read (a block can hold thousands of rows) =#
  local position = Dict{Int, Int}(c => j for (j, c) in enumerate(order))
  local entries = Tuple{Int, Int, Float64}[]
  local constant = trues(length(order))
  local nonzero = falses(length(order))
  for (i, r) in enumerate(rows)
    local eq = rowEquation(s, r, partition, differentiated, functions)
    for c in s.incidence[r]
      local j = get(position, c, 0)
      j == 0 && continue
      local d = partialDerivative(eq, s.variables[c], functions)
      local x = d === nothing ? nothing : evaluate(d, values)
      x === nothing && return nothing
      x == 0 && continue
      push!(entries, (i, j, x))
      nonzero[j] = true
      isConstantEntry(d, s) || (constant[j] = false)
    end
  end
  #= within a class, first the columns whose entries are constant (the variable solved linearly,
     `x = L * sin(phi)`'s x): an entry that reads the states (`L * cos(phi)`) can vanish =#
  local groups = [(classes[order[j]], constant[j] && nonzero[j] ? 0 : 1) for j in eachindex(order)]
  local perm = sortperm(groups; alg = Base.Sort.MergeSort)
  local moved = invperm(perm)
  (order, groups) = (order[perm], groups[perm])
  local A = Elimination(length(rows), length(order))
  local scale = 1.0
  for (i, j, x) in entries
    setEntry!(A, i, moved[j], x)
    scale = max(scale, abs(x))
  end
  local tolerance = 1e-10 * scale
  A.negligible = 1e-15 * scale
  local chosen = Int[]
  #= the columns already chosen elsewhere (at the start values) first, each by its largest entry =#
  local at = Dict{Int, Int}(c => j for (j, c) in enumerate(order))
  for c in forced
    local bj = get(at, c, 0)
    bj == 0 && return nothing
    local (best, bi, _) = largestEntry(A, bj:bj, tolerance)
    bi == 0 && return nothing
    eliminate!(A, bi, bj)
    push!(chosen, order[bj])
  end
  local j0 = 1
  while length(chosen) < length(rows) && j0 <= length(order)
    #= the columns of the next StateSelect class (its constant columns, then the others) =#
    local j1 = j0
    while j1 < length(order) && groups[j1 + 1] == groups[j0]
      j1 += 1
    end
    while length(chosen) < length(rows)
      local (best, bi, bj) = largestEntry(A, j0:j1, tolerance)
      bi == 0 && break
      eliminate!(A, bi, bj)
      push!(chosen, order[bj])
    end
    j0 = j1 + 1
  end
  return (partial || length(chosen) == length(rows)) ? chosen : nothing
end

"""
    Elimination(m, n)

`numericChoice`'s matrix, sparse: each row's nonzero entries by column, each column's rows with
an entry, the rows still active and the columns taken as pivots; entries below `negligible`
(cancelled by elimination) are dropped.
"""
mutable struct Elimination
  rows::Vector{Dict{Int, Float64}}
  columns::Vector{Set{Int}}
  active::BitVector
  taken::BitVector
  negligible::Float64
end

Elimination(m::Int, n::Int) =
  Elimination([Dict{Int, Float64}() for _ in 1:m], [Set{Int}() for _ in 1:n], trues(m), falses(n), 0.0)

"Entry (i, j) of `A` set to `x` (dropped where it is negligible)."
function setEntry!(A::Elimination, i::Int, j::Int, x::Float64)::Nothing
  if abs(x) <= A.negligible
    haskey(A.rows[i], j) && (delete!(A.rows[i], j); delete!(A.columns[j], i))
  else
    A.rows[i][j] = x
    push!(A.columns[j], i)
  end
  return nothing
end

"The largest entry above `tolerance` of the active rows in the untaken columns `js`: (its magnitude, row, column); row 0 where none is."
function largestEntry(A::Elimination, js::UnitRange{Int}, tolerance::Float64)::Tuple{Float64, Int, Int}
  local (best, bi, bj) = (tolerance, 0, 0)
  for j in js
    A.taken[j] && continue
    for i in A.columns[j]
      A.active[i] || continue
      local x = abs(A.rows[i][j])
      x > best && ((best, bi, bj) = (x, i, j))
    end
  end
  return (best, bi, bj)
end

"""
    eliminate!(A, bi, bj)

One step of `numericChoice`'s elimination: entry (bi, bj) the pivot, its row and column taken,
the column cleared in the active rows.
"""
function eliminate!(A::Elimination, bi::Int, bj::Int)::Nothing
  A.taken[bj] = true
  A.active[bi] = false
  local pivotRow = A.rows[bi]
  local p = pivotRow[bj]
  for i in collect(A.columns[bj])
    (i == bi || !A.active[i]) && continue
    local row = A.rows[i]
    local f = row[bj] / p
    for (k, v) in pivotRow
      k == bj ? setEntry!(A, i, k, 0.0) : setEntry!(A, i, k, get(row, k, 0.0) - f * v)
    end
  end
  return nothing
end

"""
    propagatedValues(s, partition, differentiated, functions, values, natural) -> StartValues

A generic point that keeps the relations the equations give explicitly: a variable an equation
determines once the others it reads are known (linear in it: `gamma = p * phi`, a connection's
`a = b`, their derivatives) takes the value that equation gives there (`solveAt`); where none is
left to solve, the next of the others takes its start moved by `genericOffset` (`genericValues`),
the states first (of the derivatives the model's own equations read, `natural`), lower
derivatives before higher. `genericValues` alone moves apart two variables an equation ties
(MSL ActuatorWithNoise: the machine's rotor angle and its controller's), where a choice regular
there is singular on every trajectory.
"""
function propagatedValues(s::Structure, partition::IR.Partition, differentiated::Dict{Int, Vector{IR.Equation}},
                          functions::Dict{Symbol, IR.ModelicaFunction}, values::StartValues, natural::BitVector)::StartValues
  local point = genericValues(values)
  local assigned = falses(length(s.variables))
  local open = Int[length(s.incidence[r]) for r in eachindex(s.rows)]
  local readers = [Int[] for _ in s.variables]
  for r in eachindex(s.rows), c in s.incidence[r]
    push!(readers[c], r)
  end
  local queue = Int[r for r in eachindex(s.rows) if open[r] == 1]
  local state = v::Int -> s.derivative[v] != 0 && natural[s.derivative[v]]
  local seeds = sort(collect(eachindex(s.variables)); by = v -> (state(v) ? 0 : 1, derivativeOrder(s, v), v))
  local next = 1
  while true
    while !isempty(queue)
      local r = pop!(queue)
      open[r] == 1 || continue
      local c = s.incidence[r][findfirst(v -> !assigned[v], s.incidence[r])]
      local value = solveAt(rowEquation(s, r, partition, differentiated, functions), s.variables[c], functions, point)
      value === nothing && continue
      point.cache[s.variables[c]] = value
      markAssigned!(queue, assigned, open, readers, c)
    end
    while next <= length(seeds) && assigned[seeds[next]]
      next += 1
    end
    next > length(seeds) && break
    #= its start moved by its offset (a derivative, no variable of its own: its offset from 0) =#
    local name = s.variables[seeds[next]]
    point.cache[name] = something(point(name), genericOffset(name))
    markAssigned!(queue, assigned, open, readers, seeds[next])
  end
  return point
end

"`propagatedValues`' step: variable `c` known, the rows that then read one unknown queued."
function markAssigned!(queue::Vector{Int}, assigned::BitVector, open::Vector{Int}, readers::Vector{Vector{Int}}, c::Int)::Nothing
  assigned[c] = true
  for r in readers[c]
    open[r] -= 1
    open[r] == 1 && push!(queue, r)
  end
  return nothing
end

"""
    solveAt(eq, name, functions, point) -> Union{Nothing, Float64}

The value of `name` that solves the scalar equation `eq`, the other variables at `point`: one
Newton step from its value there, exact as `eq` is linear in it; `nothing` where it is not, the
coefficient vanishes or a value is not known.
"""
function solveAt(eq::IR.Equation, name::Symbol, functions::Dict{Symbol, IR.ModelicaFunction}, point::StartValues)::Union{Nothing, Float64}
  eq isa IR.SCALAR_EQUATION || return nothing
  local d = partialDerivative(eq, name, functions)
  (d === nothing || readsVariable(d, name)) && return nothing
  local coefficient = evaluate(d, point)
  (coefficient === nothing || abs(coefficient) < ZERO_PIVOT) && return nothing
  #= the guess in the cache while the residual is evaluated (a derivative has no variable of its own: 0) =#
  local guess = something(point(name), 0.0)
  point.cache[name] = guess
  local residual = evaluate(minus(eq.lhs, eq.rhs), point)
  residual === nothing && (delete!(point.cache, name); return nothing)
  return guess - residual / coefficient
end

"Whether `e` reads the variable `name`."
function readsVariable(e::IR.Expression, name::Symbol)::Bool
  local found = false
  IR.foreachExp(e) do x
    (x isa IR.CREF && x.cref.name === name) && (found = true)
    nothing
  end
  return found
end

"""
    structuralChoice(s, rows, order) -> Vector{Int}

The first columns in `order` that can be matched to the rows (a greedy choice in the transversal
matroid: a column is taken where a matching of the rows to the taken columns grows).
"""
function structuralChoice(s::Structure, rows::Vector{Int}, order::Vector{Int})::Vector{Int}
  local rowIndex = Dict{Int, Int}(r => i for (i, r) in enumerate(rows))
  local columnRows = Dict{Int, Vector{Int}}(c => Int[] for c in order)
  for (i, r) in enumerate(rows), c in s.incidence[r]
    haskey(columnRows, c) && push!(columnRows[c], i)
  end
  local rowMatch = zeros(Int, length(rows))     # row -> column taken
  local chosen = Int[]
  function augment(c::Int, seen::Set{Int})::Bool
    for i in columnRows[c]
      i in seen && continue
      push!(seen, i)
      if rowMatch[i] == 0 || augment(rowMatch[i], seen)
        rowMatch[i] = c
        return true
      end
    end
    return false
  end
  for c in order
    length(chosen) == length(rows) && break
    augment(c, Set{Int}()) && push!(chosen, c)
  end
  length(chosen) == length(rows) || throw(NotHandled("dummy derivatives: no regular choice of $(length(rows)) of $(length(order)) derivatives (a structural error)"))
  return chosen
end

"""
    DynamicSet(level, candidates, chosen, jacobian, linked)

A dynamic state set (OpenModelica's state sets): of the derivatives `candidates` of a block of the
dummy derivatives' choice, as many as `jacobian` has rows are dummy derivatives, which ones
switched while the simulation runs to keep the largest pivots. The set's states are a selection of
the candidates' variables `V`, `x = A * V` and `der(x) = A * der(V)` with a discrete selection
matrix `A` (`dynamicSet!`). `chosen` are the dummies the start values choose, `jacobian` the
block's Jacobian by the candidates once its dummies of constant pivots are eliminated
(`staticElimination`: a quaternion's norm `2 Q`; a Cartesian pendulum's `2 x, 2 y` at both levels,
positions and velocities), whose pivots decide the choice. Each candidate is the derivative of a
variable that is no derivative itself, so the choice at one level does not change the candidates
of the next. `linked`: the candidates of the sets of the higher derivative levels that select
with it (`linkSets`), each in the order of `candidates`: a pendulum's velocities `vx, vy` with
its positions `x, y`, where `der(x) = vx`.
"""
struct DynamicSet
  level::Int
  candidates::Vector{Int}
  chosen::Vector{Int}
  jacobian::Matrix{IR.Expression}
  linked::Vector{Vector{Int}}
end

"""
    dynamicSets(s, level, rows, columns, chosen, classes, partition, differentiated, functions, values,
                natural, generic) -> Vector{DynamicSet}

The dynamic state sets of a block (`rows` by the candidate `columns`) whose dummies are `chosen`
(`chooseDummies` by `classes`), where its choice depends on the state: the dummies that remain
once those of constant pivots are eliminated (`staticElimination`, in the classes' order: a
pendulum's `der(\$DER.x)` by `der(\$DER.x) = der(vx)`, a body's position by its frame's) are
the choice to make, all of one class; its candidates are the columns of that class not
eliminated, its Jacobian the rows that remain (an entry that vanishes at the generic point that
keeps the explicit relations, `propagatedValues` in `generic`, dropped: zero on the
trajectories), split into its independent parts (two bodies' quaternions). A part of one row
and at least two candidates, each the derivative of a variable that is no derivative, is a set.
None where the choice is static (no class with both dummies and states, every dummy eliminated,
or its pivots constants), the remaining dummies are of more than one class, or an entry does
not differentiate. One row only for now (the switch, `dynamicSet!`, compares single pivots): a
quaternion's norm (MSL MultiBody's bodies), a pendulum.
"""
function dynamicSets(s::Structure, level::Int, rows::Vector{Int}, columns::Vector{Int}, chosen::Vector{Int},
                     classes::Dict{Int, Int}, partition::IR.Partition, differentiated::Dict{Int, Vector{IR.Equation}},
                     functions::Dict{Symbol, IR.ModelicaFunction}, values::StartValues, natural::BitVector,
                     generic::Base.RefValue{Union{Nothing, StartValues}})::Vector{DynamicSet}
  #= a class with dummies and states has a choice to make =#
  local dummyCount = Dict{Int, Int}()
  foreach(c -> dummyCount[classes[c]] = get(dummyCount, classes[c], 0) + 1, chosen)
  any(((k, n),) -> count(c -> classes[c] == k, columns) > n, dummyCount) || return DynamicSet[]
  local equations = IR.Equation[rowEquation(s, r, partition, differentiated, functions) for r in rows]
  local order = sort(chosen; by = c -> classes[c], alg = Base.Sort.MergeSort)
  local reduced = staticElimination(s, rows, equations, order, columns, functions, values)
  reduced === nothing && return DynamicSet[]
  local (dummies, remaining, M) = reduced
  (isempty(dummies) || length(unique(Int[classes[c] for c in dummies])) != 1) && return DynamicSet[]
  local k = classes[first(dummies)]
  local eliminated = Set{Int}(c for c in chosen if !(c in dummies))
  local group = sort(Int[c for c in columns if classes[c] == k && !(c in eliminated)])
  local sets = DynamicSet[]
  #= an entry that vanishes at the generic point that keeps the explicit relations is zero on the trajectories
     (MSL ActuatorWithNoise: a rotor angle's, the machine's and its controller's angles tied, gamma = gammaC) =#
  generic[] === nothing && (generic[] = propagatedValues(s, partition, differentiated, functions, values, natural))
  for i in remaining
    local magnitudes = Dict{Int, Float64}(c => abs(something(evaluate(e, something(generic[])), Inf)) for (c, e) in M[i] if c in group)
    local scale = maximum((x for (_, x) in magnitudes if isfinite(x)); init = 0.0)
    for (c, x) in magnitudes
      (isfinite(x) && x <= 1e-10 * scale) && delete!(M[i], c)
    end
  end
  local reads = Dict{Int, Vector{Int}}(i => Int[c for c in group if haskey(M[i], c)] for i in remaining)
  for (partRows, partColumns) in components(remaining, group, reads)
    local partDummies = filter(c -> c in dummies, partColumns)
    (length(partRows) == 1 && length(partDummies) == 1 && length(partColumns) >= 2) || continue
    all(c -> s.antiderivative[c] != 0 && s.antiderivative[s.antiderivative[c]] == 0, partColumns) || continue
    local candidates = sort(partColumns)
    local J = IR.Expression[get(M[only(partRows)], c, ZERO) for _ in 1:1, c in candidates]
    isConstantEntry(M[only(partRows)][only(partDummies)], s) && continue
    push!(sets, DynamicSet(level, candidates, partDummies, J, Vector{Int}[]))
  end
  return sets
end

"""
    staticElimination(s, rows, equations, order, columns, functions, values)
        -> Union{Nothing, Tuple{Vector{Int}, Vector{Int}, Vector{Dict{Int, Expression}}}}

The dummies of a block (`order`, the classes' order) that a constant pivot eliminates, the same
at every point, by symbolic Gaussian elimination of the block's Jacobian (`equations` by
`columns`): each dummy in turn by its largest constant entry in a row not yet a pivot's, again
while one more is eliminated. The dummies left (in `order`), the rows left (indices into `rows`)
and every row's entries by column (`nothing` where an entry does not differentiate or a constant
does not evaluate).
"""
function staticElimination(s::Structure, rows::Vector{Int}, equations::Vector{IR.Equation}, order::Vector{Int},
                           columns::Vector{Int}, functions::Dict{Symbol, IR.ModelicaFunction},
                           values::StartValues)::Union{Nothing, Tuple{Vector{Int}, Vector{Int}, Vector{Dict{Int, IR.Expression}}}}
  local wanted = Set{Int}(columns)
  local M = [Dict{Int, IR.Expression}() for _ in rows]
  for (i, r) in enumerate(rows), c in s.incidence[r]
    c in wanted || continue
    local d = partialDerivative(equations[i], s.variables[c], functions)
    d === nothing && return nothing
    isZeroExp(d) || (M[i][c] = d)
  end
  local active = trues(length(rows))
  local pending = copy(order)
  local progress = true
  while progress
    progress = false
    for c in copy(pending)
      local (pivot, value) = (0, 0.0)
      for i in eachindex(rows)
        (active[i] && haskey(M[i], c) && isConstantEntry(M[i][c], s)) || continue
        local x = evaluate(M[i][c], values)
        x === nothing && return nothing
        abs(x) > max(abs(value), ZERO_PIVOT) && ((pivot, value) = (i, x))
      end
      pivot == 0 && continue
      for i in eachindex(rows)
        (i == pivot || !active[i] || !haskey(M[i], c)) && continue
        local factor = times(real(1.0 / value), M[i][c])
        for (j, e) in M[pivot]
          local updated = j == c ? ZERO : minus(get(M[i], j, ZERO), times(factor, e))
          isZero(updated) ? delete!(M[i], j) : (M[i][j] = updated)
        end
      end
      active[pivot] = false
      filter!(!=(c), pending)
      progress = true
    end
  end
  return (pending, findall(active), M)
end

"""
    linkSets(s, sets, root) -> Vector{DynamicSet}

The dynamic sets with the set of each one's derivative level linked to it: a set whose variables
are aliases (`root`, `aliasRoots`) of the candidates of a set one level lower (`der(x) = vx`: a
pendulum's velocities `vx, vy` and the derivatives of its positions `x, y`) selects as that one
does, by its matrix, and goes into its `linked` (the switches would come at the same instant:
one switch keeps them together, the velocities the positions' derivatives).
"""
function linkSets(s::Structure, sets::Vector{DynamicSet}, root::Vector{Int})::Vector{DynamicSet}
  local linked = falses(length(sets))
  local bases = Pair{Int, DynamicSet}[]
  #= the lowest derivative levels first (the positions), each followed up its derivative chain =#
  for b in sortperm([set.level for set in sets]; rev = true)
    linked[b] && continue
    local levels = Vector{Int}[]
    local (level, candidates) = (sets[b].level, sets[b].candidates)
    while true
      local next = findfirst(eachindex(sets)) do a
        local set = sets[a]
        (!linked[a] && a != b && set.level == level - 1 && length(set.candidates) == length(candidates)) || return false
        local roots = Set(root[s.antiderivative[c]] for c in set.candidates)
        return length(roots) == length(set.candidates) && all(c -> root[c] in roots, candidates)
      end
      next === nothing && break
      local set = sets[next]
      candidates = Int[set.candidates[findfirst(d -> root[s.antiderivative[d]] == root[c], set.candidates)] for c in candidates]
      push!(levels, candidates)
      linked[next] = true
      level = set.level
    end
    push!(bases, b => DynamicSet(sets[b].level, sets[b].candidates, sets[b].chosen, sets[b].jacobian, levels))
  end
  return DynamicSet[set for (b, set) in sort(bases; by = first) if !linked[b]]
end

"A block of the dummy derivatives' choice: its level, rows, candidates and the dummies chosen among them."
struct BlockTrace
  level::Int
  rows::Vector{Int}
  candidates::Vector{Int}
  dummies::Vector{Int}
end

"""
    dummyDerivatives(s, partition, differentiated, variables, functions; sets, trace) -> Set{Int}

The dummy derivatives (Mattsson-Söderlind): level by level from the highest, the differentiated
equations (in their highest form, then one differentiation less, while still differentiated)
and as candidates the highest derivatives they read (then the chosen ones one differentiation
less), a regular choice of as many candidates as equations per independent block
(`chooseDummies`). Throws NotHandled where a StateSelect.always variable would have to become
a dummy. With OMBACKEND_DYNAMIC_STATE_SETS, a variable without a StateSelect is ranked as
OpenModelica ranks it (`declaredClasses`), and a block whose choice depends on the state adds
its sets (`dynamicSets`) to `sets`, those of derivative levels linked (`linkSets`). `trace`
gets each block: its level (1 the highest derivatives), rows, candidates and dummies
(`explain`).
"""
function dummyDerivatives(s::Structure, partition::IR.Partition, differentiated::Dict{Int, Vector{IR.Equation}},
                          variables::IR.VariableSet, functions::Vector{IR.ModelicaFunction};
                          sets::Vector{DynamicSet} = DynamicSet[],
                          trace::Union{Nothing, Vector{BlockTrace}} = nothing)::Set{Int}
  local functionsByName = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in functions)
  local values = StartValues(variables, functions; equations = vcat(partition.equations, partition.initialEquations))
  local generic = Ref{Union{Nothing, StartValues}}(nothing)
  local dynamic = envSwitch("OMBACKEND_DYNAMIC_STATE_SETS")
  local natural = naturalDerivatives(s)
  local declared = declaredClasses(s, partition, variables, natural; dynamic)
  local aliasClass = aliasClasses(s, partition, variables, declared)
  local dummies = Set{Int}()
  local rows = Int[r for r in eachindex(s.rows) if s.differentiated[r] == 0 && s.origin[r] != 0]
  local candidates = unique(Int[v for r in rows for v in s.incidence[r] if isHighest(s, v) && s.antiderivative[v] != 0])
  local reads = Dict{Int, Vector{Int}}(r => s.incidence[r] for r in eachindex(s.rows))
  local level = 0
  while !isempty(rows)
    level += 1
    local chosenLevel = Int[]
    for (blockRows, blockColumns) in components(rows, candidates, reads)
      isempty(blockRows) && continue
      length(blockColumns) < length(blockRows) &&
        throw(NotHandled("dummy derivatives: $(length(blockRows)) differentiated equations and $(length(blockColumns)) derivatives in a block"))
      local classes = choiceClasses(s, blockColumns, aliasClass, declared, natural)
      local chosen = chooseDummies(s, blockRows, blockColumns, partition, differentiated, variables, functionsByName, values,
                                   classes, natural, generic)
      append!(chosenLevel, chosen)
      trace === nothing || push!(trace, BlockTrace(level, blockRows, blockColumns, chosen))
      dynamic && append!(sets, dynamicSets(s, level, blockRows, blockColumns, chosen, classes, partition, differentiated,
                                           functionsByName, values, natural, generic))
    end
    local forced = [s.variables[s.antiderivative[c]] for c in chosenLevel if dummyClass(s, declared, c) == IR.ALWAYS]
    isempty(forced) || throw(NotHandled("StateSelect.always: $(join(first(forced, 5), ", ")) cannot stay states"))
    union!(dummies, chosenLevel)
    rows = Int[s.origin[r] for r in rows if s.origin[s.origin[r]] != 0]
    candidates = unique(Int[s.antiderivative[c] for c in chosenLevel if s.antiderivative[s.antiderivative[c]] != 0])
  end
  if !isempty(sets)
    local linked = linkSets(s, sets, aliasRoots(s, partition))
    empty!(sets)
    append!(sets, linked)
  end
  return dummies
end

"""
    naturalDerivatives(s) -> BitVector

The variables the model's own (undifferentiated) equations read: of a derivative, its variable a
natural state (a filter's output, a body's position).
"""
function naturalDerivatives(s::Structure)::BitVector
  local natural = falses(length(s.variables))
  for r in eachindex(s.rows)
    s.origin[r] == 0 && (natural[s.incidence[r]] .= true)
  end
  return natural
end

"""
    selectedStates(s, dummies) -> Vector{Int}

The states the dummy derivatives leave: the variables whose derivative is in the system and not
a dummy derivative.
"""
selectedStates(s::Structure, dummies::Set{Int})::Vector{Int} =
  Int[v for v in eachindex(s.variables) if s.derivative[v] != 0 && !(s.derivative[v] in dummies)]
