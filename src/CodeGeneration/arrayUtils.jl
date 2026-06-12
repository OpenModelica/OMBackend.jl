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
