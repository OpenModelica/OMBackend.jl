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

#=
  Array utility functions for code generation.
  Handles conversion between Modelica's nested vector representation
  and Julia's multi-dimensional array types.

  Author: johti17
=#

"""
  Convert nested vectors to proper Julia multi-dimensional arrays.
  Handles any depth of nesting:
  - Vector{Vector{T}} -> Matrix{T} (2D)
  - Vector{Vector{Vector{T}}} -> Array{T,3} (3D)
  - etc.
  If already a proper Array or not a nested vector, return as-is.

  This is needed because Modelica arrays may be stored as nested vectors
  but Julia requires proper Array types for multi-dimensional indexing A[i,j,k].
"""
function ensureArray(x)
  if x isa Vector && !isempty(x) && first(x) isa Vector
    #= Recursively convert inner vectors first =#
    converted = [ensureArray(v) for v in x]
    #= Stack along first dimension - builds the array row by row =#
    reduce(vcat, [reshape(c, 1, size(c)...) for c in converted])
  else
    x
  end
end

#= Alias for backwards compatibility with existing code =#
const ensureMatrix = ensureArray

"""
  Compute dot product of two vectors.
  This is a wrapper for sum(a .* b) that works with any numeric vectors.
  Used by generated code for Modelica's vector scalar product.
"""
function vectorDot(a, b)
  return sum(a .* b)
end
