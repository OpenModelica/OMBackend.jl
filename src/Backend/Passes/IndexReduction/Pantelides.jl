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

#= Pantelides' algorithm (C. C. Pantelides, The consistent initialization of differential-
   algebraic systems, SIAM J. Sci. Stat. Comput. 9(2), 1988) on a Structure. =#

"""
    PantelidesMatching(variableRow, rowVariable)

A matching of rows and highest derivatives: `variableRow[v]` the row variable v is matched to,
`rowVariable[r]` the variable row r is matched to (0: none).
"""
mutable struct PantelidesMatching
  variableRow::Vector{Int}
  rowVariable::Vector{Int}
end

"""
    Search

The work arrays of the augmenting-path searches: a visit is marked with the search's number
(`epoch`), so no array is cleared between searches; the rows and variables the last search
visited, for differentiating them where it failed.
"""
mutable struct Search
  epoch::Int
  rowMark::Vector{Int}
  variableMark::Vector{Int}
  visitedRows::Vector{Int}
  visitedVariables::Vector{Int}
end

Search() = Search(0, Int[], Int[], Int[], Int[])

"The search's marks grown to `nRows` rows and `nVariables` variables (differentiation adds both)."
function growMarks!(search::Search, nRows::Int, nVariables::Int)::Search
  length(search.rowMark) < nRows && append!(search.rowMark, zeros(Int, nRows - length(search.rowMark)))
  length(search.variableMark) < nVariables && append!(search.variableMark, zeros(Int, nVariables - length(search.variableMark)))
  return search
end

"The matching grown to `nRows` rows and `nVariables` variables, the new ones unmatched."
function growMatching!(m::PantelidesMatching, nRows::Int, nVariables::Int)::PantelidesMatching
  length(m.rowVariable) < nRows && append!(m.rowVariable, zeros(Int, nRows - length(m.rowVariable)))
  length(m.variableRow) < nVariables && append!(m.variableRow, zeros(Int, nVariables - length(m.variableRow)))
  return m
end

"""
    augmentingPath!(s, m, search, root) -> Bool

An augmenting path from the unmatched row `root` over the highest derivatives (depth first,
with an explicit stack: a model's paths can be thousands long); where one exists the matching
is changed along it. The rows and variables visited are left in `search`.
"""
function augmentingPath!(s::Structure, m::PantelidesMatching, search::Search, root::Int)::Bool
  search.epoch += 1
  local epoch = search.epoch
  empty!(search.visitedRows)
  empty!(search.visitedVariables)
  local rows = Int[root]          # the path's rows
  local position = Int[0]         # the next neighbor to try of each row on the path; 0: not looked at yet
  local through = Int[]           # through[k]: the variable leading from rows[k] to rows[k + 1]
  search.rowMark[root] = epoch
  push!(search.visitedRows, root)
  while !isempty(rows)
    local r = rows[end]
    local neighbors = s.incidence[r]
    if position[end] == 0
      #= a free highest derivative ends the path =#
      for v in neighbors
        if isHighest(s, v) && m.variableRow[v] == 0
          m.variableRow[v] = r
          m.rowVariable[r] = v
          for k in length(through):-1:1
            m.variableRow[through[k]] = rows[k]
            m.rowVariable[rows[k]] = through[k]
          end
          return true
        end
      end
      position[end] = 1
    end
    local next = 0
    while position[end] <= length(neighbors)
      local v = neighbors[position[end]]
      position[end] += 1
      if isHighest(s, v) && search.variableMark[v] != epoch
        next = v
        break
      end
    end
    if next == 0
      pop!(rows); pop!(position)
      isempty(through) || pop!(through)
      continue
    end
    search.variableMark[next] = epoch
    push!(search.visitedVariables, next)
    local owner = m.variableRow[next]
    if search.rowMark[owner] != epoch
      search.rowMark[owner] = epoch
      push!(search.visitedRows, owner)
      push!(through, next)
      push!(rows, owner)
      push!(position, 0)
    end
  end
  return false
end

"""
    addDerivative!(s, v) -> Int

The derivative of variable `v` as a new variable of the structure (`\$DER.` before its name);
its index.
"""
function addDerivative!(s::Structure, v::Int)::Int
  s.derivative[v] == 0 || return s.derivative[v]
  local d = addVariable!(s, IR.derivativeName(s.variables[v]))
  s.derivative[v] = d
  s.antiderivative[d] = v
  return d
end

"""
    artificialStates!(s, partition, variables; prefer = true) -> Int

A derivative for each continuous variable of StateSelect.always, and with `prefer` of
StateSelect.prefer, whose alias set the model does not differentiate (MLS 4.8.8.1: always "use
it as a state", prefer "use it as a state if possible"; OpenModelica's artificial states,
BackendDAECreate.lowerVarkind): Pantelides then differentiates the equations that make it one,
and the choice of states ranks it (`chooseDummies`). RollingWheelSet's `der_theta1` (always);
MSL Fluid's medium states (`p`, `T` or `h` of `preferredMediumStates`, a tank's `level`), which
the mass and energy balances otherwise leave to `m` and `U`. Not where an alias is
differentiated (RollingWheelSet's `x`, its prismatic joint's `s`): that alias is the state, a
derivative of `x` would be a dummy (`der(x) = der(s)` read at different orders). The number of
prefer variables given a derivative.
"""
function artificialStates!(s::Structure, partition::IR.Partition, variables::IR.VariableSet; prefer::Bool = true)::Int
  local root = aliasRoots(s, partition)
  local differentiated = Set{Int}(root[v] for v in eachindex(s.variables) if s.derivative[v] != 0)
  local preferred = 0
  for v in 1:length(s.variables)
    (s.derivative[v] == 0 && s.antiderivative[v] == 0 && !(root[v] in differentiated)) || continue
    local var = get(variables, s.variables[v], nothing)
    var === nothing && continue
    local select = var.attributes.stateSelect
    (select === IR.ALWAYS || (prefer && select === IR.PREFER)) || continue
    addDerivative!(s, v)
    push!(differentiated, root[v])
    select === IR.PREFER && (preferred += 1)
  end
  return preferred
end

"""
    addDerivativeRow!(s, r) -> Int

The derivative of row `r` as a new row: it reads what r reads and each of their derivatives.
"""
function addDerivativeRow!(s::Structure, r::Int)::Int
  s.differentiated[r] == 0 || return s.differentiated[r]
  local reads = copy(s.incidence[r])
  for v in s.incidence[r]
    local d = s.derivative[v]
    d == 0 && throw(NotHandled("a derivative missing for $(s.variables[v]) (a structural error)"))
    d in reads || push!(reads, d)
  end
  push!(s.rows, s.rows[r])
  push!(s.incidence, reads)
  push!(s.differentiated, 0)
  push!(s.origin, r)
  s.differentiated[r] = length(s.rows)
  return length(s.rows)
end

"The number of times row `r` is a derivative of a row of the partition."
function rowOrder(s::Structure, r::Int)::Int
  local n = 0
  while s.origin[r] != 0
    r = s.origin[r]
    n += 1
  end
  return n
end

"""
    pantelides!(s; maxOrder = 10) -> PantelidesMatching

Pantelides' algorithm on `s`: each row of the partition is matched to a highest derivative by an
augmenting path; where none exists, the rows and variables the search visited are a
structurally singular subset: they are differentiated (new rows and derivative variables in
`s`), the visited variables' derivatives matched to the visited rows' derivatives, and the
search repeats from the row's derivative. Rows are matched greedily first (most need no
search). Throws NotHandled for a system that is structurally singular itself (a row reading
no continuous variable, or a row differentiated more than `maxOrder` times).
"""
function pantelides!(s::Structure; maxOrder::Int = 10)::PantelidesMatching
  local nRows = length(s.rows)
  local m = PantelidesMatching(zeros(Int, length(s.variables)), zeros(Int, nRows))
  for r in 1:nRows
    for v in s.incidence[r]
      if isHighest(s, v) && m.variableRow[v] == 0
        m.variableRow[v] = r
        m.rowVariable[r] = v
        break
      end
    end
  end
  local search = Search()
  for k in 1:nRows
    #= a row differentiated by an earlier search is no longer an equation of the system: its derivative is =#
    (m.rowVariable[k] == 0 && s.differentiated[k] == 0) || continue
    local r = k
    while true
      growMarks!(search, length(s.rows), length(s.variables))
      augmentingPath!(s, m, search, r) && break
      #= a row reading no continuous variable is not helped by differentiating (0 = 0); one reading only states is: its derivative reads theirs =#
      isempty(s.incidence[r]) && throw(NotHandled(singularMessage(s, search.visitedRows)))
      rowOrder(s, r) >= maxOrder && throw(NotHandled(singularMessage(s, search.visitedRows)))
      local rows = copy(search.visitedRows)
      local variables = copy(search.visitedVariables)
      foreach(v -> addDerivative!(s, v), variables)
      foreach(row -> addDerivativeRow!(s, row), rows)
      growMatching!(m, length(s.rows), length(s.variables))
      for v in variables
        local d = s.derivative[v]
        m.variableRow[d] = s.differentiated[m.variableRow[v]]
        m.rowVariable[m.variableRow[d]] = d
        m.variableRow[v] = 0
      end
      for row in rows
        m.rowVariable[row] = 0
      end
      #= the rows' derivatives matched above; the search's own row continues as its derivative =#
      r = s.differentiated[r]
      m.rowVariable[r] == 0 || break
    end
  end
  return m
end

"The message for a structurally singular subset: its equations and the variables they read."
function singularMessage(s::Structure, rows::Vector{Int})::String
  local names = unique(Symbol[s.variables[v] for r in rows for v in s.incidence[r]])
  return "a structurally singular system: $(length(rows)) equations (rows $(join(first(rows, 8), ", "))) " *
         "over the variables $(join(first(names, 12), ", "))$(length(names) > 12 ? ", ..." : "")"
end

"""
    derivativeSeeds(s, functions) -> Seeds

The seeds of d/dt in the structure: a variable's derivative is its derivative variable (a
reference of type Real); a name outside the structure (a parameter, a constant, a discrete
variable) has none (0).
"""
function derivativeSeeds(s::Structure, functions::Vector{IR.ModelicaFunction})::Seeds
  local variable = name -> begin
    local v = get(s.index, name, 0)
    v == 0 && return nothing
    local d = s.derivative[v]
    d == 0 && throw(NotHandled("the derivative of $(name), which Pantelides' algorithm gave none"))
    return IR.CREF(IR.ComponentRef(s.variables[d], IR.T_REAL()), IR.T_REAL())
  end
  return Seeds(variable, 1.0, Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in functions))
end

"""
    differentiatedEquations(s, partition, functions) -> Dict{Int, Vector{Equation}}

The equations Pantelides' algorithm differentiated, symbolically: for each equation of the
partition (by index) its first, second, ... derivative. An if-equation's rows (its positions)
can be differentiated unequally often: its derivative of order j holds the positions
differentiated at least j times (an if-equation of those positions, its conditions kept).
"""
function differentiatedEquations(s::Structure, partition::IR.Partition, functions::Vector{IR.ModelicaFunction})::Dict{Int, Vector{IR.Equation}}
  local orders = Dict{Int, Vector{Int}}()                 # equation => order of each position
  for r in eachindex(s.rows)
    s.origin[r] == 0 || continue
    local row = s.rows[r]
    local byPosition = get!(() -> Int[], orders, row.equation)
    length(byPosition) < row.position && append!(byPosition, zeros(Int, row.position - length(byPosition)))
    byPosition[row.position] = rowOrder(s, last(derivativeChain(s, r)))
  end
  local seeds = derivativeSeeds(s, functions)
  local out = Dict{Int, Vector{IR.Equation}}()
  for (eq, byPosition) in orders
    #= every position of an if-equation, the ones that are no rows (a Boolean `on = true`) of order 0:
       its reversed else-branch is indexed by the count of them all =#
    local e = partition.equations[eq]
    e isa IR.IF_EQUATION && length(byPosition) < length(e.elseEquations) &&
      append!(byPosition, zeros(Int, length(e.elseEquations) - length(byPosition)))
    local n = maximum(byPosition; init = 0)
    n == 0 && continue
    local chain = IR.Equation[]
    #= the inlined form, as the structure read it =#
    local current = inlineEquation(partition.equations[eq], seeds.functions)
    for j in 1:n
      local positions = Int[k for k in eachindex(byPosition) if byPosition[k] >= j]
      current = differentiateEquation(positionsOf(current, positions, byPosition, j), seeds)
      push!(chain, current)
    end
    out[eq] = chain
  end
  return out
end

"""
    positionsOf(eq, positions, byPosition, j) -> Equation

The equation of the positions differentiated at least `j` times: a scalar equation itself; of
an if-equation (`eq` holding the positions differentiated at least `j - 1` times), the if-equation
of those positions.
"""
positionsOf(eq::IR.Equation, ::Vector{Int}, ::Vector{Int}, ::Int)::IR.Equation = eq

function positionsOf(eq::IR.IF_EQUATION, positions::Vector{Int}, byPosition::Vector{Int}, j::Int)::IR.Equation
  #= `eq` holds the positions with order >= j - 1, in order: the indices of `positions` among them =#
  local held = Int[k for k in eachindex(byPosition) if byPosition[k] >= j - 1]
  local keep = Int[findfirst(==(k), held) for k in positions]
  length(keep) == length(held) && return eq
  local pick = es -> IR.Equation[es[i] for i in keep]
  #= the else-branch reversed (as the lowering keeps it): its equation of position i is its (n + 1 - i)-th =#
  local n = length(held)
  local elseEquations = IR.Equation[eq.elseEquations[n + 1 - i] for i in reverse(keep)]
  return IR.IF_EQUATION(IR.EquationBranch[IR.EquationBranch(b.condition, pick(b.equations)) for b in eq.branches],
                        elseEquations, eq.source, eq.attributes)
end

"The row `r` and its derivatives, in order."
function derivativeChain(s::Structure, r::Int)::Vector{Int}
  local out = Int[r]
  while s.differentiated[out[end]] != 0
    push!(out, s.differentiated[out[end]])
  end
  return out
end
