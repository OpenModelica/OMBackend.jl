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

"
  This files contains the various graph algorithms.

  More specifically routines for matching, merging and strongly connected components

  Author: John Tinnerholm

"
module GraphAlgorithms

import Graphs
import MetaGraphs
using DataStructures

"""
  Regular matching. (Does not solve singularities).
  Author: John Tinnerholm
  input:
        dict, adjacency list representation of the equation-variable graph
        n, the number of unknown and equations.
  output:
         assign::Array. assign[j] = i,
         where j is a variable and i the equation in which it is assigned.
         isSingular::Boolean,
         Boolean indicating if the system is singular or not.
"""
function matching(dict::DataStructures.OrderedDict, n::Int)
  #= Global arrays for bookkeeping =#
  local assign = [0 for i in 1:n]
  local vMark = fill(false, n)
  local eMark = fill(false, n)
  """
    Calculates the path for equation i.
    returns true if a path is found.
  """
  function pathFound(i)
    eMark[i] = true
    local success = false
    local equationsI = dict.vals[i]
    for j in equationsI
      if assign[j] == 0
        assign[j] = i
        success = true
        return success
      end
    end
    #= Otherwise =#
    success = false
    for j in equationsI
      if vMark[j] != false
        continue
      end
      vMark[j] = true
      success = pathFound(assign[j])
      if success
        assign[j] = i
        return success
      end
    end
    return success
  end
  #=Entry of algorithm=#
  local isSingular = false
  local success::Bool
  for i in 1:n
    fill!(vMark, false)
    fill!(eMark, false)
    try
      success = pathFound(i)
    catch e
      local msg = "Failed to match equations to variables.
                   A possible reason is that the system is over/underdetermined.
                   Matching will be done by later backend processing."
      @info msg
      throw(e)
    end
    if !success
      isSingular = !success
    end
  end
  return isSingular, assign
end

"""
Author: John Tinnerholm
  Given an order, and a graph represented as an adjacency list creates a new digraph
  representing a causalized system.
  input matchOrder, assign array(j) = i The variable j is solved in equation i
  input graph equation -> {Equation -> variables belonging to it}
  output Graphs.SimpleDiGraph
"""
function merge(matchOrder::Vector, graph::OrderedDict)::MetaGraphs.MetaDiGraph
  "Remove function for arrays.."
  function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
  end
  #=
    Convert the given map into an array representation.
    Similar format to the assign matrix but represent the dependencies
    depends(i) = {Set of variables used in equation i}.
  =#
  local depends = graph.vals
  local g = MetaGraphs.MetaDiGraph()
  #=
    Create vertices. Each equation in matchorder is one vertex in the graph.
  =#
  local nMatchOrder = length(matchOrder)
  for eq in 1:nMatchOrder
    Graphs.add_vertex!(g)
    MetaGraphs.set_prop!(g, eq, :eID, eq)
  end
  #= Build inverse mapping: equation -> variable indices assigned to it =#
  local invMatch = Dict{Int, Vector{Int}}()
  for (varIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      if !haskey(invMatch, eqIdx)
        invMatch[eqIdx] = Int[]
      end
      push!(invMatch[eqIdx], varIdx)
    end
  end
  for eq in 1:nMatchOrder
    varIdx = get(invMatch, eq, Int[])
    if length(varIdx) == 0
      continue
    end
    local depVariables = remove!(depends[eq], first(varIdx))
    #= Solve for the remaining variables =#
    if ! isempty(depVariables)
      for v in depVariables
        MetaGraphs.set_prop!(g, matchOrder[v], :vID, v)
        MetaGraphs.set_prop!(g, eq, :vID, varIdx[1])
        Graphs.add_edge!(g, matchOrder[v], eq)
      end
    else
      MetaGraphs.set_prop!(g, eq, :vID, varIdx[1])
    end
  end
  local nWithEquationId = count(i -> haskey(MetaGraphs.props(g, i), :eID), Graphs.vertices(g))
  local nWithVariableId = count(i -> haskey(MetaGraphs.props(g, i), :vID), Graphs.vertices(g))
  @debug "[GRAPH] matching graph built" vertices=Graphs.nv(g) edges=Graphs.ne(g) equationVertices=nWithEquationId variableVertices=nWithVariableId
  return g
end

"""
  Dumps the properties of a given MetaDiGraph.
"""
function dumpGraphProperties(g::MetaGraphs.MetaDiGraph)
  local nVertices = Graphs.vertices(g).stop
  local str = "Meta properties of the graph:\n"
  for i in 1:nVertices
    str *= "Properties: $(MetaGraphs.props(g, i))\n"
  end
  return str
end

"""
  Topological sort
"""
function topological_sort(g::Graphs.AbstractGraph)::Vector
  Graphs.topological_sort_by_dfs(g)
end

function stronglyConnectedComponents(g::Graphs.AbstractGraph)::Vector
  Graphs.strongly_connected_components_kosaraju(g)
end


function connected_components(g::Graphs.AbstractGraph)
  Graphs.connected_components(g)
end

"""
  Author: John Tinnerholm
  Tarjans algorithm
"""
function tarjan(g::OrderedDict)::Vector{Vector{Int}}
  tarjan(g, length(g.keys))
end

"""
 Helper function.
 It is assumed that the dict g is ordered 1->N where 1->N is the indices of the nodes.
 input g::OrderedDict
 input n::Int, the number of vertices
 output sccs: The set of strongly connected components
"""
function tarjan(g::OrderedDict, n)::Vector{Vector{Int}}
  function strongConnect(v::Int)
    vIndicies[v] = indexRef[]
    vLowLinks[v] = indexRef[]
    indexRef[] += 1
    push!(stack, v)
    vOnStack[v] = true
    v2S = g[v]
    for v2 in v2S
      if vIndicies[v2] == 0
        strongConnect(v2)
        vLowLinks[v] = min(vLowLinks[v], vLowLinks[v2])
      elseif vOnStack[v2]
        vLowLinks[v] = min(vLowLinks[v], vLowLinks[v2])
      end
    end
    if vLowLinks[v] == vIndicies[v]
      scc = Int[]
      while true
        w = pop!(stack)
        vOnStack[w] = false
        push!(scc, w)
        if w == v
          break
        end
      end
      push!(sccs, scc)
    end
  end
  local indexRef = Ref{Int}(1)
  local sccs = Vector{Int}[]
  local stack = Int[]
  #= Indices for the vertices 1->n. 0 = undefined. =#
  local vIndicies = zeros(Int, n)
  local vLowLinks = zeros(Int, n)
  local vOnStack = fill(false, n)
  for v in g.keys
    #= If v is undefined =#
    if vIndicies[v] == 0
      strongConnect(v)
    end
  end
  return sccs
end

"""
  A single block in a Block Lower Triangular (BLT) decomposition.

  Fields:
  - `eqs`    equation indices belonging to the block
  - `vars`   variable indices matched to those equations
  - `isLoop` true iff `length(eqs) > 1` (genuine algebraic loop requiring a
             nonlinear solver or tearing; false iff the block is a scalar
             assignment `vars[1] := solve(eqs[1] for vars[1])`)

  A `Vector{BLTBlock}` in topological order is the canonical intermediate
  form for backend codegen: scalar blocks lower to direct assignments;
  loop blocks lower to per-block solver calls.
"""
struct BLTBlock
  eqs::Vector{Int}
  vars::Vector{Int}
  isLoop::Bool
end

"""
  Block Lower Triangular (BLT) decomposition.

  Given the bipartite equation-variable adjacency and a perfect matching,
  returns the BLT form: equations grouped into strongly connected components
  ordered so that each block depends only on variables solved in earlier
  blocks.

  Algorithm:
    1. `merge(matchOrder, adjacency)` builds the oriented equation graph where
       an edge `eq_i -> eq_j` means `eq_j uses the variable that eq_i solves`.
    2. SCC decomposition of that graph yields the blocks.
    3. Topologically sort the condensation so block order is explicit and
       independent of the underlying SCC routine's ordering convention.
    4. Package each SCC with its matched variables.

  input  adjacency   equation index -> list of variable indices
  input  matchOrder  variable index -> equation index (output of `matching`)
  output Vector{BLTBlock}  blocks in execution (topological) order
"""
function blt(adjacency::DataStructures.OrderedDict, matchOrder::Vector{Int})::Vector{BLTBlock}
  return blt(merge(matchOrder, adjacency), matchOrder)
end

"""
  Overload that accepts a pre-built oriented digraph, for callers that
  already ran `merge` (e.g. `matchAndCheckStronglyConnectedComponents`).
"""
function blt(digraph::MetaGraphs.MetaDiGraph, matchOrder::Vector{Int})::Vector{BLTBlock}
  local sccs = Graphs.strongly_connected_components_kosaraju(digraph)
  local nSCC = length(sccs)
  #= Build an explicit condensation DAG so the topological order of blocks
     does not depend on the internal iteration order of Kosaraju's routine. =#
  local vertexToSCC = Dict{Int, Int}()
  for (i, scc) in enumerate(sccs)
    for v in scc
      vertexToSCC[v] = i
    end
  end
  local condensation = Graphs.DiGraph(nSCC)
  for e in Graphs.edges(digraph)
    local srcSCC = vertexToSCC[Graphs.src(e)]
    local dstSCC = vertexToSCC[Graphs.dst(e)]
    if srcSCC != dstSCC
      Graphs.add_edge!(condensation, srcSCC, dstSCC)
    end
  end
  local order = Graphs.topological_sort_by_dfs(condensation)
  #= Invert the matching once so per-block variable lookup is O(1). =#
  local eqToVars = Dict{Int, Vector{Int}}()
  for (varIdx, eqIdx) in enumerate(matchOrder)
    if eqIdx > 0
      push!(get!(eqToVars, eqIdx, Int[]), varIdx)
    end
  end
  local blocks = Vector{BLTBlock}(undef, nSCC)
  for (outIdx, sccIdx) in enumerate(order)
    local scc = sccs[sccIdx]
    local vars = Int[]
    for eq in scc
      append!(vars, get(eqToVars, eq, Int[]))
    end
    blocks[outIdx] = BLTBlock(sort(scc), sort(vars), length(scc) > 1)
  end
  return blocks
end

"""
  Human-readable dump of a BLT decomposition for debugging and log output.
"""
function dumpBLT(blocks::Vector{BLTBlock})::String
  local io = IOBuffer()
  println(io, "BLT decomposition: $(length(blocks)) block(s)")
  for (i, b) in enumerate(blocks)
    local kind = b.isLoop ? "LOOP(size=$(length(b.eqs)))" : "SCALAR"
    println(io, "  [$i] $kind  eqs=$(b.eqs)  vars=$(b.vars)")
  end
  return String(take!(io))
end

end #= GraphAlgorithms =#
