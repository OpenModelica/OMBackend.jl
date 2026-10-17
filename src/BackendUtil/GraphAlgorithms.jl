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
    GraphAlgorithms

The graph algorithms of the simulation code's passes: a maximum matching of equations to
variables (`matching`), the strongly connected components of a directed graph.
"""
module GraphAlgorithms

import Graphs
using DataStructures: OrderedDict

"""
    matching(rows, n) -> (isSingular, assign)

A maximum matching (augmenting paths) of the equations `1:n` to the variables `1:n`:
`rows[i]` the variables of equation `i` (the rows of a square system: an equation without
variables has an empty row). `assign[j]` is the equation variable `j` is matched to (0 for
none); `isSingular` when an equation is left unmatched. Recursive: a large system can overflow
the stack (the callers take that as no matching).
"""
function matching(rows::OrderedDict{Int, Vector{Int}}, n::Int)::Tuple{Bool, Vector{Int}}
  local assign = zeros(Int, n)
  local vMark = falses(n)
  local equations = collect(values(rows))
  local isSingular = false
  for i in 1:n
    fill!(vMark, false)
    augment!(i, equations, assign, vMark) || (isSingular = true)
  end
  return (isSingular, assign)
end

"Whether an augmenting path from equation `i` matches it, re-matching the equations on the path."
function augment!(i::Int, equations::Vector{Vector{Int}}, assign::Vector{Int}, vMark::BitVector)::Bool
  for j in equations[i]
    assign[j] == 0 && (assign[j] = i; return true)
  end
  for j in equations[i]
    vMark[j] && continue
    vMark[j] = true
    if augment!(assign[j], equations, assign, vMark)
      assign[j] = i
      return true
    end
  end
  return false
end

"The strongly connected components of the directed graph `g` (Kosaraju)."
stronglyConnectedComponents(g::Graphs.AbstractGraph)::Vector{Vector{Int}} = Graphs.strongly_connected_components_kosaraju(g)

end #= GraphAlgorithms =#
