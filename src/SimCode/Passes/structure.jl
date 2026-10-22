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

#= The structure of the final residual system: its strongly connected components, the index
   over-constraint diagnostic. =#

"""
    augmentMatching!(i, incidence, columnOf, rowOf, seen)

Whether an augmenting path from row `i` matches it to one of its `incidence[i]` columns (not
in `seen`), re-matching the rows on the path.
"""
function augmentMatching!(i::Int, incidence::Vector{OrderedSet{String}}, columnOf::Vector{String},
                          rowOf::Dict{String, Int}, seen::OrderedSet{String})::Bool
  for c in incidence[i]
    c in seen && continue
    push!(seen, c)
    if !haskey(rowOf, c) || augmentMatching!(rowOf[c], incidence, columnOf, rowOf, seen)
      rowOf[c] = i
      columnOf[i] = c
      return true
    end
  end
  return false
end

"A maximum matching of the rows of `incidence` to their columns: each row's column (`\"\"` for none), each column's row."
function maximumMatching(incidence::Vector{OrderedSet{String}})::Tuple{Vector{String}, Dict{String, Int}}
  local columnOf = fill("", length(incidence))
  local rowOf = Dict{String, Int}()
  for i in eachindex(incidence)
    augmentMatching!(i, incidence, columnOf, rowOf, OrderedSet{String}())
  end
  return (columnOf, rowOf)
end

"""
    cyclicNames(deps)

The names of `deps` on a cycle of what each reads (`deps[n]`, its names of `deps` counted): as observed
equations, each solved for its name, they would have no order (foldExplicitSingleAssign,
eliminateOutputOnlyVariables).
"""
function cyclicNames(deps::OrderedDict{String, OrderedSet{String}})::OrderedSet{String}
  local names = collect(keys(deps))
  local index = Dict{String, Int}(n => k for (k, n) in enumerate(names))
  local g = Graphs.SimpleDiGraph(length(names))
  for (k, n) in enumerate(names), m in deps[n]
    local j = get(index, m, 0)
    (j > 0 && j != k) && Graphs.add_edge!(g, k, j)
  end
  local cyclic = OrderedSet{String}()
  for c in GraphAlgorithms.stronglyConnectedComponents(g)
    length(c) > 1 && foreach(k -> push!(cyclic, names[k]), c)
  end
  return cyclic
end

"""
    residualComponents(simCode)

The strongly connected components of the residuals and the unknown each residual is matched
to (`""` for none): the residuals matched to the unknowns they read, an edge from a residual to
the one that solves an unknown it reads.
"""
function residualComponents(simCode::SIM_CODE)::Tuple{Vector{Vector{Int}}, Vector{String}}
  local residuals = simCode.residualEquations
  isempty(residuals) && return (Vector{Int}[], String[])
  local unknowns = OrderedSet{String}(k for (k, (_, v)) in simCode.stringToSimVarHT if isUnknownVarKind(v.varKind))
  local incidence = OrderedSet{String}[intersect(crefNames!(OrderedSet{String}(), eq.exp), unknowns) for eq in residuals]
  local (columnOf, rowOf) = maximumMatching(incidence)
  local g = Graphs.SimpleDiGraph(length(residuals))
  for (i, names) in enumerate(incidence), v in names
    local j = get(rowOf, v, 0)
    (j > 0 && j != i) && Graphs.add_edge!(g, i, j)
  end
  return (Vector{Int}[Vector{Int}(c) for c in GraphAlgorithms.stronglyConnectedComponents(g)], columnOf)
end

"The names `e` reads differentiated (`der(x)`) and plainly, apart."
function differentialNames!(differentiated::OrderedSet{String}, plain::OrderedSet{String}, e::Exp)::Nothing
  if e isa DER
    push!(differentiated, crefName(e.cref))
  elseif e isa CALL && e.name === :der
    foreach(a -> crefNames!(differentiated, a), e.args)
  elseif e isa EXP_CREF
    push!(plain, crefName(e.cref))
  else
    foreach(c -> differentialNames!(differentiated, plain, c), children(e))
  end
  return nothing
end

"""
    indexOverconstraintDiagnostic(simCode)

With `OMBACKEND_INDEX_DIAG` set: the residuals a matching to their highest derivatives (`der(x)`
of a state, `y` of an algebraic variable) leaves unmatched, the over-constraining ones,
logged. `simCode` unchanged.
"""
function indexOverconstraintDiagnostic(simCode::SIM_CODE)::SIM_CODE
  envSwitch("OMBACKEND_INDEX_DIAG") || return simCode
  local ht = simCode.stringToSimVarHT
  local states = Set{String}(k for (k, (_, v)) in ht if isUnknownVarKind(v.varKind) && isState(v))
  local algebraic = Set{String}(k for (k, (_, v)) in ht if isUnknownVarKind(v.varKind) && !isState(v))
  local incidence = OrderedSet{String}[]
  for eq in simCode.residualEquations
    local (d, p) = (OrderedSet{String}(), OrderedSet{String}())
    differentialNames!(d, p, eq.exp)
    local columns = OrderedSet{String}()
    for n in d
      n in states && push!(columns, "der_" * n)
      n in algebraic && push!(columns, n)
    end
    foreach(n -> n in algebraic && push!(columns, n), p)
    push!(incidence, columns)
  end
  local (columnOf, _) = maximumMatching(incidence)
  local unmatched = findall(isempty, columnOf)
  @info "[SIMCODE: $(simCode.name): indexDiag] differential-incidence localization" n_eqs = length(incidence) nHighestVars = length(states) + length(algebraic) nUnmatched = length(unmatched)
  for i in unmatched
    @info "[SIMCODE: $(simCode.name): indexDiag] unmatched [$i]" equation = simCode.residualEquations[i]
  end
  return simCode
end
