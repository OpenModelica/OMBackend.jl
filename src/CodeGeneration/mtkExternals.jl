#=
This file contains slightly modified code from MTK.
This is used here according to the MIT license.
Details below.
=#

#= The ModelingToolkit.jl package is licensed under the MIT "Expat" License:
# Copyright (c) 2018-25: Christopher Rackauckas, Julia Computing.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  SOFTWARE.
=#


#=
This file contains "hacks".
This is done in order to get the equations on a MTK compatible format before calling functions such as structurally simplify.
TODO:
!Adjust the uncessary string conversions!
=#


#= Global dictionary to store dynamically generated function implementations =#
#= The key is the function name, the value is the implementation function =#
const MODELICA_FUNCTION_IMPLS = Dict{Symbol, Function}()

#= Global dictionary to store RTG wrappers for each function =#
const MODELICA_FUNCTION_WRAPPERS = Dict{Symbol, Any}()

#= Cache for per-element extractor functions.
   Key: (funcName::Symbol, indices::Tuple{Vararg{Int}}, nArgs::Int)
   Value: the created function object
   nArgs is part of the key because the extractor RGF is built with
   fixed arity — calling an extractor with the wrong number of args
   triggers @inbounds __args[i] out-of-bounds → LLVM unreachable → SIGILL. =#
const ELEM_FUNC_CACHE = Dict{Tuple{Symbol, Tuple, Int}, Any}()
const TUPLE_ELEM_FUNC_CACHE = Dict{Tuple, Any}()

#= Stores the count of array-shaped subtrees found in the last structural_simplify call.
   Used by tests to assert the shape invariant (0 = clean for Pantelides).
   Contract: only updated when ENABLE_BACKEND_LOGGING is true at module load.
   When logging is off, this Ref stays at its sentinel -1 because the
   diagnostic walk that writes it is gated behind @BACKEND_LOGGING. Tests
   that read this value must guard with `!OMBackend.ENABLE_BACKEND_LOGGING`
   to skip the assertion in production runs. =#
const _LAST_ARRAY_SHAPE_COUNT = Ref{Int}(-1)

"""
Unwrap a value for use in symbolic Terms.
For arrays, unwraps each element. For scalars, unwraps directly.
Non-symbolic values (plain Float64, Int, etc.) pass through unchanged.
"""
function unwrapForSymbolic(x)
  if x isa AbstractArray
    return map(Symbolics.unwrap, x)
  else
    return Symbolics.unwrap(x)
  end
end

"""
Get or create a per-element extractor function for an array-returning Modelica function.
The extractor calls the implementation directly via MODELICA_FUNCTION_IMPLS and extracts
a single element by index. This avoids creating getindex(Term{Real}(...), i) symbolic
terms which crash Symbolics._linear_expansion (OffsetArrays.Origin error).
"""
function getOrCreateElemFunc(funcName::Symbol, indices::Tuple{Vararg{Int}}, nArgs::Int)
  local key = (funcName, indices, nArgs)
  if haskey(ELEM_FUNC_CACHE, key)
    return ELEM_FUNC_CACHE[key]
  end
  local fnQuote = QuoteNode(funcName)
  local argNames = [Symbol("a", k) for k in 1:nArgs]
  local implCall = Expr(:call, :(Base.invokelatest), :impl, argNames...)
  local body
  if length(indices) == 1
    local idx = indices[1]
    body = Expr(:->, Expr(:tuple, argNames...), Expr(:block,
      :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]),
      :(result = $implCall),
      :(return result[$idx])
    ))
  elseif length(indices) == 2
    local i = indices[1]
    local j = indices[2]
    body = Expr(:->, Expr(:tuple, argNames...), Expr(:block,
      :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]),
      :(result = $implCall),
      :(row = result[$i]),
      Expr(:if, :(row isa AbstractVector),
        :(return row[$j]),
        :(return result[$i, $j])
      )
    ))
  end
  local f = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  ELEM_FUNC_CACHE[key] = f
  return f
end

"""
Flatten array arguments into individual scalar elements.
Returns (flatArgs, shapes) where shapes is a tuple of () for scalars,
(n,) for vectors, or (n,m) for matrices. The flat element extractors
use shapes to reassemble arrays before calling the real implementation.
"""
function _flattenSymArgs(uwArgs::Vector{Any})
  flatArgs = Any[]
  shapes = Tuple[]
  for a in uwArgs
    if a isa AbstractMatrix
      push!(shapes, size(a))
      for el in vec(a)
        push!(flatArgs, el)
      end
    elseif a isa AbstractVector && !isempty(a) && first(a) isa AbstractVector
      #= Nested vector: Modelica matrix stored as Vector{Vector{T}}.
         Flatten to individual scalar elements, record as 2D shape. =#
      local nrows = length(a)
      local ncols = length(first(a))
      push!(shapes, (nrows, ncols))
      for row in a
        for el in row
          push!(flatArgs, el)
        end
      end
    elseif a isa AbstractVector
      push!(shapes, (length(a),))
      for el in a
        push!(flatArgs, el)
      end
    else
      #= Scalar: pass through as-is. =#
      push!(shapes, ())
      push!(flatArgs, a)
    end
  end
  return flatArgs, Tuple(shapes)
end

"""
Get or create a flat element extractor that accepts individual scalar arguments,
reassembles them into the original array shapes, calls the implementation, and
extracts a single element. This avoids array_literal in Term arguments, which
Pantelides index reduction cannot differentiate.
"""
function _getOrCreateFlatElemFunc(funcName::Symbol, indices::Tuple{Vararg{Int}}, nFlat::Int, shapes::Tuple)
  local key = (funcName, :flat, indices, nFlat, shapes)
  if haskey(TUPLE_ELEM_FUNC_CACHE, key)
    return TUPLE_ELEM_FUNC_CACHE[key]
  end
  local fnQuote = QuoteNode(funcName)
  local flatArgNames = [Symbol("f", k) for k in 1:nFlat]

  #= Build reassembly statements: reconstruct each original arg from flat scalars =#
  local stmts = Expr[]
  local origArgNames = Symbol[]
  local offset = 1
  for (k, shape) in enumerate(shapes)
    local orig = Symbol("a", k)
    push!(origArgNames, orig)
    if shape == ()
      push!(stmts, :($orig = $(flatArgNames[offset])))
      offset += 1
    elseif length(shape) == 1
      local n = shape[1]
      push!(stmts, :($orig = [$(flatArgNames[offset:offset+n-1]...)]))
      offset += n
    elseif length(shape) == 2
      local total = shape[1] * shape[2]
      push!(stmts, :($orig = reshape([$(flatArgNames[offset:offset+total-1]...)], $(shape[1]), $(shape[2]))))
      offset += total
    end
  end

  local implCall = Expr(:call, :(Base.invokelatest), :impl, origArgNames...)
  push!(stmts, :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]))
  push!(stmts, :(result = $implCall))

  if length(indices) == 1
    push!(stmts, :(return result[$(indices[1])]))
  elseif length(indices) == 2
    local i = indices[1]
    local j = indices[2]
    push!(stmts, :(row = result[$i]))
    push!(stmts, Expr(:if, :(row isa AbstractVector),
      :(return row[$j]),
      :(return result[$i, $j])
    ))
  end

  local body = Expr(:->, Expr(:tuple, flatArgNames...), Expr(:block, stmts...))
  local f = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  TUPLE_ELEM_FUNC_CACHE[key] = f
  return f
end

"""
    makeSymbolicTerm(f, args)

Create a `Symbolics.Num`-wrapped `SymbolicUtils.Term` for a Modelica function call.
Centralizes the SymbolicUtils type parameter so there is a single point of change
when the SymbolicUtils API evolves.

Variant type: `SymReal` (the default variant in SymbolicUtils, and the variant
that `Symbolics.Num` wraps via `infer_vartype(::Type{Num}) = SymReal`).
MTK unknowns are `BasicSymbolic{SymReal}`, so terms wrapping them must also
be `SymReal` to avoid cross-variant errors.

symtype: `Real` (passed as `type = Real` keyword). This is required because the
`Num` constructor asserts `symtype(ex) <: Number`. Without it, the safe_ctors.jl
default `_promote_symtype(f, args)` cannot infer the return type of our RTG
element extractor functions.
"""
function makeSymbolicTerm(f, args::Vector{Any})
  #= Guard: reject ALL array arguments. SymbolicUtils wraps Term arguments into
     BasicSymbolic nodes. Arrays (whether symbolic or numeric) get array shape
     metadata that triggers "Differentiation with array expressions is not yet
     supported" during Pantelides index reduction.
     Callers must flatten array args via _flattenSymArgs before reaching here. =#
  for (i, a) in enumerate(args)
    if a isa AbstractArray
      error("makeSymbolicTerm: argument $i is an AbstractArray ($(typeof(a)), length=$(length(a))). " *
            "All array arguments must be flattened to scalars before creating Terms. " *
            "Use _flattenSymArgs to decompose arrays into individual scalar elements.")
    end
  end
  return Symbolics.Num(SymbolicUtils.Term{SymbolicUtils.SymReal}(f, args; type = Real))
end

"""
Call a tuple-returning Modelica function and extract a specific element.
Used by the TSUB handler in equation code generation to avoid the problem
where Num(scalar_term)[ix] is a no-op in the Symbolics framework.

When called with symbolic arguments, creates a Term{Real} wrapping an
element-specific RTG function that extracts element `ix` at numeric evaluation time.
When called with numeric arguments, calls the function impl directly and extracts element `ix`.

All symbolic terms are created via `makeSymbolicTerm`.
"""
function tupleElementCall(funcName::Symbol, ix::Int, args...)
  if hasSymbolicArgs(args...)
    #= Try eager evaluation: call impl with original (Num-wrapped) args.
       Produces elementary symbolic expressions Symbolics can differentiate.
       Falls back to opaque RTG extractors if eval fails (if-statement on symbolic). =#
    try
      local impl = MODELICA_FUNCTION_IMPLS[funcName]
      local preparedArgs = _prepareArgsForEagerEval(args)
      local result = Base.invokelatest(impl, preparedArgs...)
      return result[ix]
    catch
    end
    #= Fallback: opaque RTG Term extractors =#
    local uwArgs = Any[unwrapForSymbolic(a) for a in args]
    local hasArrayArgs = any(a -> a isa AbstractArray, uwArgs)
    if hasArrayArgs
      local flatArgs, shapes = _flattenSymArgs(uwArgs)
      local nFlat = length(flatArgs)
      local elemFunc = _getOrCreateFlatElemFunc(funcName, (ix,), nFlat, shapes)
      return makeSymbolicTerm(elemFunc, flatArgs)
    else
      local nArgs = length(uwArgs)
      local elemFunc = getOrCreateElemFunc(funcName, (ix,), nArgs)
      return makeSymbolicTerm(elemFunc, uwArgs)
    end
  else
    local impl = MODELICA_FUNCTION_IMPLS[funcName]
    local result = Base.invokelatest(impl, args...)
    return result[ix]
  end
end

"""
Get or create a per-element extractor function for an array element within a
tuple-returning Modelica function. The extractor calls the implementation,
extracts the tuple element at tupleIdx, then extracts the array element at arrayIndices.
"""
function getOrCreateTupleElemFunc(funcName::Symbol, tupleIdx::Int, arrayIndices::Tuple{Vararg{Int}}, nArgs::Int)
  local key = (funcName, :tsub, tupleIdx, arrayIndices, nArgs)
  if haskey(TUPLE_ELEM_FUNC_CACHE, key)
    return TUPLE_ELEM_FUNC_CACHE[key]
  end
  local fnQuote = QuoteNode(funcName)
  local argNames = [Symbol("a", k) for k in 1:nArgs]
  local implCall = Expr(:call, :(Base.invokelatest), :impl, argNames...)
  local body
  if length(arrayIndices) == 1
    local idx = arrayIndices[1]
    body = Expr(:->, Expr(:tuple, argNames...), Expr(:block,
      :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]),
      :(result = $implCall),
      :(tupleElem = result[$tupleIdx]),
      :(return tupleElem[$idx])
    ))
  elseif length(arrayIndices) == 2
    local i = arrayIndices[1]
    local j = arrayIndices[2]
    body = Expr(:->, Expr(:tuple, argNames...), Expr(:block,
      :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]),
      :(result = $implCall),
      :(tupleElem = result[$tupleIdx]),
      :(row = tupleElem[$i]),
      Expr(:if, :(row isa AbstractVector),
        :(return row[$j]),
        :(return tupleElem[$i, $j])
      )
    ))
  else
    error("3D+ tuple array indices not supported")
  end
  local f = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  TUPLE_ELEM_FUNC_CACHE[key] = f
  return f
end

"""
Flat variant of `getOrCreateTupleElemFunc`. Accepts individual scalar arguments,
reassembles them into the original array shapes, calls the implementation, extracts
tuple element at `tupleIdx`, then extracts array element at `arrayIndices`.
This avoids array_literal in Term arguments, which Pantelides cannot differentiate.
"""
function _getOrCreateFlatTupleElemFunc(funcName::Symbol, tupleIdx::Int, arrayIndices::Tuple{Vararg{Int}}, nFlat::Int, shapes::Tuple)
  local key = (funcName, :flat_tsub, tupleIdx, arrayIndices, nFlat, shapes)
  if haskey(TUPLE_ELEM_FUNC_CACHE, key)
    return TUPLE_ELEM_FUNC_CACHE[key]
  end
  local fnQuote = QuoteNode(funcName)
  local flatArgNames = [Symbol("f", k) for k in 1:nFlat]

  #= Build reassembly statements: reconstruct each original arg from flat scalars =#
  local stmts = Expr[]
  local origArgNames = Symbol[]
  local offset = 1
  for (k, shape) in enumerate(shapes)
    local orig = Symbol("a", k)
    push!(origArgNames, orig)
    if shape == ()
      push!(stmts, :($orig = $(flatArgNames[offset])))
      offset += 1
    elseif length(shape) == 1
      local n = shape[1]
      push!(stmts, :($orig = [$(flatArgNames[offset:offset+n-1]...)]))
      offset += n
    elseif length(shape) == 2
      local total = shape[1] * shape[2]
      push!(stmts, :($orig = reshape([$(flatArgNames[offset:offset+total-1]...)], $(shape[1]), $(shape[2]))))
      offset += total
    end
  end

  local implCall = Expr(:call, :(Base.invokelatest), :impl, origArgNames...)
  push!(stmts, :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]))
  push!(stmts, :(result = $implCall))
  push!(stmts, :(tupleElem = result[$tupleIdx]))

  if length(arrayIndices) == 1
    local idx = arrayIndices[1]
    push!(stmts, :(return tupleElem[$idx]))
  elseif length(arrayIndices) == 2
    local i = arrayIndices[1]
    local j = arrayIndices[2]
    push!(stmts, :(row = tupleElem[$i]))
    push!(stmts, Expr(:if, :(row isa AbstractVector),
      :(return row[$j]),
      :(return tupleElem[$i, $j])
    ))
  else
    error("3D+ tuple array indices not supported")
  end

  local body = Expr(:->, Expr(:tuple, flatArgNames...), Expr(:block, stmts...))
  local f = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  TUPLE_ELEM_FUNC_CACHE[key] = f
  return f
end

"""
Call a tuple-returning Modelica function and extract a specific tuple element
that is an array. Returns a symbolic array (Vector{Num} or Matrix{Num}) where
each element is a per-element extractor Term.

Used by the TSUB handler when the tuple element type is T_ARRAY.
"""
function tupleArrayElementCall(funcName::Symbol, tupleIdx::Int, dims::Tuple{Vararg{Int}}, args...)
  if hasSymbolicArgs(args...)
    #= Try eager evaluation: call impl with Num-wrapped args, extract tuple element.
       Produces elementary symbolic expressions for Pantelides differentiation. =#
    try
      local impl = MODELICA_FUNCTION_IMPLS[funcName]
      local preparedArgs = _prepareArgsForEagerEval(args)
      local result = Base.invokelatest(impl, preparedArgs...)
      local tupleElem = result[tupleIdx]
      return _ensureArrayShape(tupleElem, dims)
    catch
    end
    #= Fallback: opaque RTG Term extractors =#
    local uwArgs = Any[unwrapForSymbolic(a) for a in args]
    local hasArrayArgs = any(a -> a isa AbstractArray, uwArgs)
    if hasArrayArgs
      local flatArgs, shapes = _flattenSymArgs(uwArgs)
      local nFlat = length(flatArgs)
      if length(dims) == 1
        return [makeSymbolicTerm(_getOrCreateFlatTupleElemFunc(funcName, tupleIdx, (i,), nFlat, shapes), flatArgs) for i in 1:dims[1]]
      elseif length(dims) == 2
        return [makeSymbolicTerm(_getOrCreateFlatTupleElemFunc(funcName, tupleIdx, (i, j), nFlat, shapes), flatArgs) for i in 1:dims[1], j in 1:dims[2]]
      else
        error("3D+ tuple array elements not supported")
      end
    else
      local nArgs = length(uwArgs)
      if length(dims) == 1
        return [makeSymbolicTerm(getOrCreateTupleElemFunc(funcName, tupleIdx, (i,), nArgs), uwArgs) for i in 1:dims[1]]
      elseif length(dims) == 2
        return [makeSymbolicTerm(getOrCreateTupleElemFunc(funcName, tupleIdx, (i, j), nArgs), uwArgs) for i in 1:dims[1], j in 1:dims[2]]
      else
        error("3D+ tuple array elements not supported")
      end
    end
  else
    local impl = MODELICA_FUNCTION_IMPLS[funcName]
    local result = Base.invokelatest(impl, args...)
    return result[tupleIdx]
  end
end

"""
Call a tuple-returning Modelica function and extract a single element at
(tupleIdx, arrayIndices...). Returns a symbolic Num for symbolic args, or the
actual element value for numeric args. Used when codegen knows the specific
indices at compile time (e.g., ASUB(ASUB(CALL, [tupleIx]), [i, j])).
"""
function tupleArrayElementAt(funcName::Symbol, tupleIdx::Int, arrayIndices::Tuple{Vararg{Int}}, args...)
  if hasSymbolicArgs(args...)
    #= Try eager evaluation: call impl with Num-wrapped args, extract scalar. =#
    try
      local impl = MODELICA_FUNCTION_IMPLS[funcName]
      local preparedArgs = _prepareArgsForEagerEval(args)
      local result = Base.invokelatest(impl, preparedArgs...)
      local tupleElem = result[tupleIdx]
      if length(arrayIndices) == 1
        return tupleElem[arrayIndices[1]]
      elseif length(arrayIndices) == 2
        local row = tupleElem[arrayIndices[1]]
        if row isa AbstractVector
          return row[arrayIndices[2]]
        else
          return tupleElem[arrayIndices[1], arrayIndices[2]]
        end
      end
    catch
    end
    #= Fallback: opaque RTG Term extractors =#
    local uwArgs = Any[unwrapForSymbolic(a) for a in args]
    local hasArrayArgs = any(a -> a isa AbstractArray, uwArgs)
    if hasArrayArgs
      local flatArgs, shapes = _flattenSymArgs(uwArgs)
      local nFlat = length(flatArgs)
      local f = _getOrCreateFlatTupleElemFunc(funcName, tupleIdx, arrayIndices, nFlat, shapes)
      return makeSymbolicTerm(f, flatArgs)
    else
      local nArgs = length(uwArgs)
      local f = getOrCreateTupleElemFunc(funcName, tupleIdx, arrayIndices, nArgs)
      return makeSymbolicTerm(f, uwArgs)
    end
  else
    local impl = MODELICA_FUNCTION_IMPLS[funcName]
    local result = Base.invokelatest(impl, args...)
    local tupleElem = result[tupleIdx]
    if length(arrayIndices) == 1
      return tupleElem[arrayIndices[1]]
    elseif length(arrayIndices) == 2
      local row = tupleElem[arrayIndices[1]]
      if row isa AbstractVector
        return row[arrayIndices[2]]
      else
        return tupleElem[arrayIndices[1], arrayIndices[2]]
      end
    else
      error("3D+ tuple array indices not supported")
    end
  end
end

"""
Get or create a flat scalar function. Like `_getOrCreateFlatElemFunc` but returns
the function result directly (no element indexing). Used for scalar-returning
functions that have array input arguments.
"""
function _getOrCreateFlatScalarFunc(funcName::Symbol, nFlat::Int, shapes::Tuple)
  local key = (funcName, :flat_scalar, nFlat, shapes)
  if haskey(TUPLE_ELEM_FUNC_CACHE, key)
    return TUPLE_ELEM_FUNC_CACHE[key]
  end
  local fnQuote = QuoteNode(funcName)
  local flatArgNames = [Symbol("f", k) for k in 1:nFlat]

  local stmts = Expr[]
  local origArgNames = Symbol[]
  local offset = 1
  for (k, shape) in enumerate(shapes)
    local orig = Symbol("a", k)
    push!(origArgNames, orig)
    if shape == ()
      push!(stmts, :($orig = $(flatArgNames[offset])))
      offset += 1
    elseif length(shape) == 1
      local n = shape[1]
      push!(stmts, :($orig = [$(flatArgNames[offset:offset+n-1]...)]))
      offset += n
    elseif length(shape) == 2
      local total = shape[1] * shape[2]
      push!(stmts, :($orig = reshape([$(flatArgNames[offset:offset+total-1]...)], $(shape[1]), $(shape[2]))))
      offset += total
    end
  end

  local implCall = Expr(:call, :(Base.invokelatest), :impl, origArgNames...)
  push!(stmts, :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]))
  push!(stmts, :(return $implCall))

  local body = Expr(:->, Expr(:tuple, flatArgNames...), Expr(:block, stmts...))
  local f = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  TUPLE_ELEM_FUNC_CACHE[key] = f
  return f
end

"""
Create a scalar symbolic Term for a function call, handling array arguments.
If any argument is a symbolic array, flattens to scalar elements and creates a
flat scalar extractor. Otherwise calls makeSymbolicTerm directly.
Used by wrapper functions for scalar-returning Modelica functions with array inputs.
"""
function _makeScalarSymbolicTerm(funcName::Symbol, uwArgs::Vector{Any})
  local hasArrayArgs = any(a -> a isa AbstractArray, uwArgs)
  if hasArrayArgs
    local flatArgs, shapes = _flattenSymArgs(uwArgs)
    local nFlat = length(flatArgs)
    local f = _getOrCreateFlatScalarFunc(funcName, nFlat, shapes)
    return makeSymbolicTerm(f, flatArgs)
  else
    return makeSymbolicTerm(MODELICA_FUNCTION_WRAPPERS[funcName], uwArgs)
  end
end

"""
Precreate extractor functions for array-returning Modelica wrappers.
Creating them eagerly during wrapper setup avoids first-use world-age issues
when symbolic array calls are constructed during MTK lowering.
"""
function precreateElementExtractors(funcName::Symbol, nArgs::Int, dims::Tuple{Vararg{Int}})
  if isempty(dims)
    return nothing
  elseif length(dims) == 1
    for i in 1:dims[1]
      getOrCreateElemFunc(funcName, (i,), nArgs)
    end
  elseif length(dims) == 2
    for i in 1:dims[1], j in 1:dims[2]
      getOrCreateElemFunc(funcName, (i, j), nArgs)
    end
  else
    for i in 1:prod(dims)
      getOrCreateElemFunc(funcName, (i,), nArgs)
    end
  end
  return nothing
end

"""
Create a symbolic array representation for an array-returning function call.
Returns a Vector{Num} for 1D outputs or Matrix{Num} for 2D outputs.
Each element is a scalar Term{Real} calling a per-element extractor function,
avoiding the getindex operation that triggers OffsetArrays.Origin errors in
Symbolics._linear_expansion.

Each element is a scalar symbolic term created via `makeSymbolicTerm`.
"""
function createSymbolicArrayCall(funcRef, uwArgs::Vector{Any}, dims::Tuple{Vararg{Int}}; funcName::Symbol = Symbol())
  #= If funcName not provided, try to extract it from funcRef =#
  if funcName === Symbol()
    funcName = funcRef isa Symbol ? funcRef : nameof(funcRef)
  end
  #= Check if any argument is an array. ALL arrays must be flattened to scalars
     because SymbolicUtils wraps Term arguments with array shape metadata that
     triggers "Differentiation with array expressions" in Pantelides. =#
  local hasArrayArgs = any(a -> a isa AbstractArray, uwArgs)
  if hasArrayArgs
    local flatArgs, shapes = _flattenSymArgs(uwArgs)
    local nFlat = length(flatArgs)
    if length(dims) == 1
      return [makeSymbolicTerm(_getOrCreateFlatElemFunc(funcName, (i,), nFlat, shapes), flatArgs) for i in 1:dims[1]]
    elseif length(dims) == 2
      return [makeSymbolicTerm(_getOrCreateFlatElemFunc(funcName, (i, j), nFlat, shapes), flatArgs) for i in 1:dims[1], j in 1:dims[2]]
    else
      local totalLen = prod(dims)
      return [makeSymbolicTerm(_getOrCreateFlatElemFunc(funcName, (i,), nFlat, shapes), flatArgs) for i in 1:totalLen]
    end
  else
    local nArgs = length(uwArgs)
    if length(dims) == 1
      return [makeSymbolicTerm(getOrCreateElemFunc(funcName, (i,), nArgs), uwArgs) for i in 1:dims[1]]
    elseif length(dims) == 2
      return [makeSymbolicTerm(getOrCreateElemFunc(funcName, (i, j), nArgs), uwArgs) for i in 1:dims[1], j in 1:dims[2]]
    else
      local totalLen = prod(dims)
      return [makeSymbolicTerm(getOrCreateElemFunc(funcName, (i,), nArgs), uwArgs) for i in 1:totalLen]
    end
  end
end

"""
Helper to check if a value is symbolic (Symbolics.Num or contains symbolic expressions).
Also recursively checks arrays, since record field arrays assembled from scalarized
parameters may be Vector{Symbolics.Num} or Vector{Vector{Symbolics.Num}}.
"""
isSymbolicArg(::Float64) = false
isSymbolicArg(::Int64) = false
isSymbolicArg(::Bool) = false
isSymbolicArg(::AbstractArray{Float64}) = false
isSymbolicArg(::AbstractArray{Int64}) = false
function isSymbolicArg(x)
  x isa Symbolics.Num && return true
  x isa Symbolics.Arr && return true
  x isa SymbolicUtils.BasicSymbolic && return true
  if x isa AbstractArray
    for el in x
      isSymbolicArg(el) && return true
    end
  end
  return false
end

"""
Helper to check if any argument in a tuple/collection is symbolic.
"""
function hasSymbolicArgs(args...)
  return any(isSymbolicArg, args)
end

"""
Look up an element of a constant Modelica array using runtime-resolvable
subscripts. Used for `Table[in1, in2]` patterns where the table is constant
but indices are runtime CREFs (Modelica.Electrical.Digital gates, lookup
tables, etc.).

When called with numeric args, returns `table[Int(round(i)), Int(round(j))]`
directly. When called with at least one symbolic arg, wraps the lookup in an
opaque Symbolics term so MTK structural-simplify treats it as a black box
(rather than refusing to use a Num as an array index, or constructing an
unwieldy ifelse chain).

Vararg subscripts to support both vectors and N-dimensional arrays.
"""
function constTableLookup(table::AbstractArray, idxs...)
  if hasSymbolicArgs(idxs...)
    local _table = table
    local f = (rt_idxs...) -> begin
      local resolved = ntuple(length(rt_idxs)) do k
        local v = rt_idxs[k]
        if v isa Integer
          Int(v)
        else
          Int(round(Float64(v)))
        end
      end
      _table[resolved...]
    end
    return makeSymbolicTerm(f, Any[idxs...])
  else
    return table[ntuple(k -> _toIndex(idxs[k]), length(idxs))...]
  end
end

function _toIndex(v)
  if v isa Integer
    return Int(v)
  else
    return Int(round(Float64(v)))
  end
end

"""
Prepare arguments for eager evaluation of Modelica functions with symbolic args.
Converts Vector{Vector{T}} (Modelica nested-vector matrices) to Matrix{T} so that
generated function bodies using A[i,j] indexing work correctly.
Other argument types pass through unchanged.
"""
function _prepareArgsForEagerEval(args)
  return map(args) do a
    if a isa AbstractVector && !isempty(a) && first(a) isa AbstractVector
      local nrows = length(a)
      local ncols = length(first(a))
      return [a[i][j] for i in 1:nrows, j in 1:ncols]
    else
      return a
    end
  end
end

"""
Ensure an array result from eager evaluation has the expected shape.
Converts Vector{Vector{T}} (nested vectors) to Matrix{T} for 2D output.
"""
function _ensureArrayShape(result, dims::Tuple{Vararg{Int}})
  if length(dims) == 2 && result isa AbstractVector && !isempty(result) && first(result) isa AbstractVector
    local nrows = length(result)
    local ncols = length(first(result))
    return [result[i][j] for i in 1:nrows, j in 1:ncols]
  end
  return result
end

"""
Try eager symbolic evaluation of a Modelica function, falling back to RTG Terms.
When the impl can be called directly with symbolic (Num) args, it produces
elementary expressions (sin, cos, +, *) that Symbolics CAN differentiate via
chain rule during Pantelides index reduction. If eager eval fails (e.g.,
if-statement branches on a symbolic value), falls back to opaque RTG Term
extractors that cannot be differentiated.
"""
function _symbolicFuncDispatch(funcName::Symbol, origArgs::Vector{Any}, isArray::Bool, dims::Tuple)
  try
    local impl = MODELICA_FUNCTION_IMPLS[funcName]
    local preparedArgs = _prepareArgsForEagerEval(origArgs)
    local result = Base.invokelatest(impl, preparedArgs...)
    if isArray && !isempty(dims)
      return _ensureArrayShape(result, dims)
    end
    return result
  catch
    local uwArgs = Any[unwrapForSymbolic(a) for a in origArgs]
    if isArray && !isempty(dims)
      return createSymbolicArrayCall(MODELICA_FUNCTION_WRAPPERS[funcName], uwArgs, dims; funcName=funcName)
    else
      return _makeScalarSymbolicTerm(funcName, uwArgs)
    end
  end
end

"""
Build the body Expr for a wrapper function with the given arity and dispatch mode.
Returns a quoted `function(args...) ... end` expression suitable for RuntimeGeneratedFunctions.
"""
function _buildWrapperBody(funcName::Symbol, nArgs::Int, arrayFunction::Bool, outputDims::Tuple{Vararg{Int}})
  local fnQuote = QuoteNode(funcName)
  local hasArrayDims = !isempty(outputDims)

  #= Build argument names. RTG does not support varargs, so always use fixed arity. =#
  local argNames = [Symbol("arg", i) for i in 1:nArgs]

  #= Build the symbolic check =#
  local symbolicCheckExpr
  if nArgs == 0
    symbolicCheckExpr = nothing
  elseif nArgs == 1
    symbolicCheckExpr = :(isSymbolicArg(arg1))
  else
    symbolicCheckExpr = Expr(:call, :hasSymbolicArgs, argNames...)
  end

  #= Build the symbolic return expression.
     Uses _symbolicFuncDispatch which tries eager evaluation first (producing
     elementary Num expressions Symbolics can differentiate), falling back to
     opaque RTG Terms for functions with data-dependent if-statements. =#
  local symbolicReturnExpr
  if nArgs == 0
    symbolicReturnExpr = nothing
  else
    local origArgsExpr = Expr(:ref, :Any, argNames...)
    symbolicReturnExpr = :(return _symbolicFuncDispatch($fnQuote, $origArgsExpr, $arrayFunction, $outputDims))
  end

  #= Build the impl call =#
  local implCallExpr
  if nArgs == 0
    implCallExpr = :(Base.invokelatest(impl))
  else
    implCallExpr = Expr(:call, :(Base.invokelatest), :impl, argNames...)
  end

  #= Assemble the body block =#
  local stmts = Expr[]

  #= Add symbolic dispatch if applicable =#
  if symbolicCheckExpr !== nothing && symbolicReturnExpr !== nothing
    push!(stmts, Expr(:if, symbolicCheckExpr, symbolicReturnExpr))
  end

  #= Add impl lookup and call =#
  push!(stmts, :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]))
  push!(stmts, implCallExpr)

  local bodyBlock = Expr(:block, stmts...)

  #= Build the arrow function expression: (args...) -> begin ... end
     RTG requires arrow form with fixed arity (no varargs). =#
  if nArgs == 0
    return Expr(:->, Expr(:tuple), bodyBlock)
  else
    return Expr(:->, Expr(:tuple, argNames...), bodyBlock)
  end
end

function createModelicaFunctionWrapper(funcName::Symbol, nArgs::Int, arrayFunction::Bool = false, outputDims::Tuple{Vararg{Int}} = ())
  #= Always (re-)create the wrapper so the correct arrayFunction flag is applied. =#

  #= Build the function body expression and create an RTG function.
     RTG functions are world-age safe: they can be called from any world age,
     unlike @eval-created functions which are "too new" when called from
     RuntimeGeneratedFunction context (e.g., MTK equation evaluation). =#
  local body = _buildWrapperBody(funcName, nArgs, arrayFunction, outputDims)
  local rtg = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  MODELICA_FUNCTION_WRAPPERS[funcName] = rtg
  if arrayFunction && !isempty(outputDims)
    precreateElementExtractors(funcName, nArgs, outputDims)
  end

  #= Create/update a module-level binding so equation expressions that reference
     the function by name (e.g., in rewritten equations) can find it.
     Always update the binding to point to the latest RTG wrapper. =#
  local fnQuote2 = QuoteNode(funcName)
  @eval $funcName = MODELICA_FUNCTION_WRAPPERS[$fnQuote2]
end

#=
So we know about t an der in the global scope.
This is needed for the rules below to match correctly.
=#

@independent_variables t
const D = Differential(t)
using DataStructures

"""
Rewrite equations for MTK: move derivatives to the LHS, rename der to D,
qualify Modelica function calls, and wrap dynamic calls with invokelatest.
"""
function rewriteEquations(edeqs, simCode)
  local funcNames = Set{Symbol}(Symbol(f.name) for f in simCode.functions)
  return rewriteEquationsExprLevel(edeqs isa Vector{Expr} ? edeqs : Expr[e for e in edeqs];
                                   modelicaFuncNames = funcNames)
end

"""
  Check if a symbol is a registered dynamic Modelica function.
  Uses the MODELICA_FUNCTION_WRAPPERS dictionary populated by createModelicaFunctionWrapper.
"""
function isDynamicModelicaFunction(sym::Symbol)
  return haskey(MODELICA_FUNCTION_WRAPPERS, sym)
end

"""
Check if an Expr represents a qualified call to OMBackend.CodeGeneration.X
by inspecting the Expr structure directly instead of stringifying.
"""
function _isOMBackendQualifiedCall(e::Expr)
  e.head == :. || return false
  length(e.args) >= 1 || return false
  local lhs = e.args[1]
  #= Check for nested dot: OMBackend.CodeGeneration =#
  if lhs isa Expr && lhs.head == :.
    return lhs == :(OMBackend.CodeGeneration)
  end
  return false
end
_isOMBackendQualifiedCall(_) = false

"""
  Wrap function calls to dynamically generated Modelica functions with Base.invokelatest
  to avoid world-age issues.
"""
function wrapWithInvokelatest(expr::Expr)
  if expr.head == :call
    func = expr.args[1]
    #= Check if the function is a qualified call to OMBackend.CodeGeneration =#
    if func isa Expr && _isOMBackendQualifiedCall(func)
      #= Wrap with Base.invokelatest =#
      local newArgs = Any[:(Base.invokelatest), func]
      for a in expr.args[2:end]
        push!(newArgs, wrapWithInvokelatest(a))
      end
      return Expr(:call, newArgs...)
    #= Check if the function is a bare symbol that is a registered dynamic function =#
    elseif func isa Symbol && isDynamicModelicaFunction(func)
      #= Wrap with Base.invokelatest =#
      local newArgs = Any[:(Base.invokelatest), func]
      for a in expr.args[2:end]
        push!(newArgs, wrapWithInvokelatest(a))
      end
      return Expr(:call, newArgs...)
    end
    #= Recursively process arguments; return original if nothing changed =#
    local changed = false
    local newArgs = copy(expr.args)
    for (i, a) in enumerate(expr.args)
      r = wrapWithInvokelatest(a)
      newArgs[i] = r
      changed |= r !== a
    end
    return changed ? Expr(:call, newArgs...) : expr
  end
  #= For all other Expr types; return original if nothing changed =#
  local changed = false
  local newArgs = copy(expr.args)
  for (i, a) in enumerate(expr.args)
    r = wrapWithInvokelatest(a)
    newArgs[i] = r
    changed |= r !== a
  end
  return changed ? Expr(expr.head, newArgs...) : expr
end

wrapWithInvokelatest(x) = x  #= For non-Expr types, return as-is =#


"""
  $(SIGNATURES)

  Structurally simplify algebraic equations in a system and compute the
  topological sort of the observed equations. When `simplify=true`, the `simplify`
  function will be applied during the tearing process. It also takes kwargs
  `allow_symbolic=false` and `allow_parameter=true` which limits the coefficient
  types during tearing.

  The optional argument `io` may take a tuple `(inputs, outputs)`.
  This will convert all `inputs` to parameters and allow them to be unconnected, i.e.,
  simplification will allow models where `n_states = n_equations - n_inputs`.
  """

"""
  Filter out equations that contain no symbolic variables (e.g. `0 ~ 0.0`, `0 ~ 255.0`).
  These arise when a variable is reclassified as a parameter but its defining equation is kept.
  Such equations are either tautologies or contradictions and confuse the initialization system.
"""
function filterConstantEquations(eqs::AbstractVector)
  filtered = filter(eqs) do eq
    lhs_vars = Symbolics.get_variables(eq.lhs)
    rhs_vars = Symbolics.get_variables(eq.rhs)
    !isempty(lhs_vars) || !isempty(rhs_vars)
  end
  local n_removed = length(eqs) - length(filtered)
  if n_removed > 0
    @info "Removed $n_removed constant-only equations (no unknowns)"
  end
  return filtered
end

"""
    resolveAliasInitialValue(diffState, fullEqs, ivMap)

  Resolve the initial value of a differential state that has no direct Modelica
  start value (typically an MTK-generated derivative variable like `Xˍt`).

  Scans `fullEqs` for a 2-variable linear equation where one variable is
  `diffState` and the other has a known value in `ivMap`. Uses symbolic
  variable analysis (`Symbolics.get_variables`, `isequal`, `substitute`).

  Returns the resolved value, or `nothing` if no alias equation was found.
"""
function resolveAliasInitialValue(diffState, fullEqs, ivMap::Dict)
  #= Substitute all known values into each equation, then check if the equation
     becomes a simple linear expression in diffState alone. This avoids type
     mismatches between get_variables output and ivMap keys. =#
  for eq in fullEqs
    local eqStr = string(eq)
    local diffStr = string(diffState)
    if !contains(eqStr, diffStr)
      continue
    end
    #= Skip differential equations (Differential(t)(X) ~ ...) as these are
       dynamic equations, not algebraic alias equations. =#
    if contains(string(eq.lhs), "Differential")
      continue
    end
    local expr = eq.lhs - eq.rhs
    local exprSub = Symbolics.substitute(expr, ivMap)
    #= Symbolics.substitute returns a wrapped BasicSymbolic constant after a
       full substitution; unwrap with `Symbolics.value` so `isa Number` holds. =#
    local intercept = Symbolics.value(Symbolics.substitute(exprSub, Dict(diffState => 0)))
    local sumOnePoint = Symbolics.value(Symbolics.substitute(exprSub, Dict(diffState => 1)))
    local fullyResolvedToNumber = intercept isa Number && sumOnePoint isa Number
    if !fullyResolvedToNumber
      continue
    end
    local slope = sumOnePoint - intercept
    if !iszero(slope)
      return Float64(-intercept / slope)
    end
  end
  return nothing
end

"""
    splitInitialValues(reducedSystem, finalInitialValues)

  Split initial values into hard constraints and soft guesses based on the mass matrix.
  Differential states (mass matrix diagonal != 0) get hard u0 values.
  Algebraic states (mass matrix diagonal == 0) become guesses to avoid
  overdetermining the initialization system.
  For pure ODE systems (identity mass matrix), all values stay hard.

  Returns `(system, hardInitialValues)` where system may have updated guesses.
"""
function splitInitialValues(reducedSystem, finalInitialValues, allInitialValues = Pair[])
  local massMatrix = ModelingToolkit.calculate_massmatrix(reducedSystem)
  local reducedUnks = unknowns(reducedSystem)
  #= Identity mass matrix means pure ODE: all states are differential =#
  if massMatrix isa LinearAlgebra.UniformScaling
    @info "ODEProblem: pure ODE (identity mass matrix), $(length(finalInitialValues)) hard u0, $(length(reducedUnks)) unknowns"
    return (reducedSystem, finalInitialValues)
  end
  #= DAE system: classify states by mass matrix diagonal =#
  @BACKEND_LOGGING @info "[splitIV] finalInitialValues keys:" [string(p.first) for p in finalInitialValues]
  @BACKEND_LOGGING @info "[splitIV] reducedUnks:" [string(u) for u in reducedUnks]
  local diffStateSet = Set{Any}()
  local diffStateStrSet = Set{String}()
  for i in 1:min(size(massMatrix, 1), length(reducedUnks))
    if massMatrix[i, i] != 0
      push!(diffStateSet, reducedUnks[i])
      push!(diffStateStrSet, string(reducedUnks[i]))
    end
  end
  #= Use string comparison for hard/soft split because finalInitialValues keys
     are Num-wrapped while diffStateSet contains unwrapped BasicSymbolic values.
     isequal(Num(x), x) can fail depending on Symbolics version. =#
  local hardInitialValues = filter(finalInitialValues) do pair
    string(pair.first) in diffStateStrSet
  end
  local softInitialValues = filter(finalInitialValues) do pair
    !(string(pair.first) in diffStateStrSet)
  end
  #= When a DAE has only algebraic vars with explicit starts and no differential
     state has one, pin the algebraic starts as hard u0. MTK's DAE initialiser
     then has a well-determined system (algebraic-pin + residuals) and converges
     to the correct root for differential states (e.g. Pendulum: x,y pinned at 10
     forces phi = 3π/4 via x=L*sin(phi), y=-L*cos(phi)). Soft guesses alone are
     insufficient because the NLS minimises motion from the guess and moves x,y
     instead of phi, collapsing to phi=0 and x=0,y=-L.

     Skip when the System carries non-empty `initialization_eqs`: those
     constraints define the IC, and pinning every default-0 algebraic IV as
     hard u0 over-constrains the init problem so MTK silently picks a
     trivial-zero root and violates the user's `start = N`. =#
  local initEqs = ModelingToolkit.initialization_equations(reducedSystem)
  local hasInitConstraints = !isempty(initEqs)
  if isempty(hardInitialValues) && !isempty(softInitialValues) && !hasInitConstraints
    @info "No differential states have explicit IVs; pinning $(length(softInitialValues)) algebraic IVs as hard u0 so DAE init can solve for diff states"
    hardInitialValues = softInitialValues
    softInitialValues = Pair[]
  end
  #= Promote `lhs ~ literal` init_eqs to hard u0 entries when `lhs` is a
     reduced unknown. MTK's init solver treats `initialization_eqs` as
     least-squares residuals, so a single `Inertia_w ~ 10` can be sacrificed
     against many algebraic residuals minimised toward zero. =#
  if hasInitConstraints
    local hardKeyStrSet = Set(string(p.first) for p in hardInitialValues)
    local reducedUnkStrSet = Set(string(u) for u in reducedUnks)
    local promoted = 0
    for eq in initEqs
      local rhsVal = Symbolics.value(eq.rhs)
      rhsVal isa Number || continue
      local lhsStr = string(eq.lhs)
      lhsStr in reducedUnkStrSet || continue
      lhsStr in hardKeyStrSet && continue
      push!(hardInitialValues, eq.lhs => Float64(rhsVal))
      push!(hardKeyStrSet, lhsStr)
      softInitialValues = filter(p -> string(p.first) != lhsStr, softInitialValues)
      promoted += 1
    end
    if promoted > 0
      @info "Promoted $(promoted) initialization_eqs literal constraints to hard u0"
    end
  end
  if !isempty(softInitialValues)
    local currentGuesses = ModelingToolkit.guesses(reducedSystem)
    local newGuesses = merge(currentGuesses, Dict(softInitialValues))
    @set! reducedSystem.guesses = newGuesses
  end
  #= Ensure all differential states have hard initial values.
     MTK's order-lowering creates derivative variables (e.g. Inertia_phiˍt for
     der(Inertia_phi)) that have no Modelica start value.
     Use resolveAliasInitialValue to find alias equations in the reduced system.
     The allIVMap includes both reduced-system unknowns AND pre-simplification
     variables (allInitialValues) so we can resolve aliases to variables that
     were eliminated by structural_simplify. =#
  #= Use string-based set for checking existing hard IVs, because isequal between
     Num-wrapped keys (from finalInitialValues) and BasicSymbolic values (from
     reducedUnks/diffStateSet) can fail. =#
  local hardSymStrSet = Set(string(iv.first) for iv in hardInitialValues)
  #= Only feed user-explicit start values (hard + soft, both filtered through
     `skipDefaultsForStates`) to the alias resolver. `allInitialValues` carries
     default 0.0 entries for non-explicit vars; substituting those lets a
     trivial-zero alias win over a non-zero user-explicit one. =#
  local explicitIVMap = Dict{Any, Any}(iv.first => iv.second
                                       for iv in vcat(hardInitialValues, softInitialValues))
  local fullEqs = ModelingToolkit.full_equations(reducedSystem)
  #= Propagate hard u0 entries through algebraic alias chains in the reduced
     system. Iterates to a fixed point so multi-step chains resolve, and
     covers MTK-generated derivative-suffix vars regardless of mass-matrix
     classification (Engine1a's `Inertia_phiˍt` is the differential state but
     `Inertia_w` is its algebraic alias; promoting `Inertia_w ~ 10` to hard u0
     lets us propagate to `Inertia_phiˍt` via the alias). =#
  local progressed = true
  while progressed
    progressed = false
    explicitIVMap = Dict{Any, Any}(iv.first => iv.second
                                   for iv in vcat(hardInitialValues, softInitialValues))
    for unk in reducedUnks
      local unkStr = string(unk)
      unkStr in hardSymStrSet && continue
      contains(unkStr, "ˍ") || continue
      local ivMapForResolve = filter(p -> string(p.first) != unkStr, explicitIVMap)
      local resolved = resolveAliasInitialValue(unk, fullEqs, ivMapForResolve)
      resolved === nothing && continue
      push!(hardInitialValues, unk => Float64(resolved))
      push!(hardSymStrSet, unkStr)
      progressed = true
      @info "Resolved $(unk) to $(resolved) via equation alias"
    end
  end
  for diffState in diffStateSet
    if !(string(diffState) in hardSymStrSet)
      local diffStateStr = string(diffState)
      #= Only attempt alias resolution for MTK-generated derivative variables
         (e.g. Inertia_phiˍt created by order-lowering). These have the Unicode
         dot character ˍ in their name and no Modelica start value.
         For regular Modelica variables without explicit start values, do NOT add
         a hard u0 entry. Let MTK's initialization solver infer their values from
         algebraic constraints and guesses (e.g. phi inferred from x,y via
         x = L*sin(phi)). Adding hard 0.0 would override this inference. =#
      if contains(diffStateStr, "\u02cd")
        local ivMapForResolve = filter(p -> string(p.first) != diffStateStr, explicitIVMap)
        local resolved = resolveAliasInitialValue(diffState, fullEqs, ivMapForResolve)
        local finalVal = something(resolved, 0.0)
        push!(hardInitialValues, diffState => finalVal)
        if resolved !== nothing
          @info "Resolved differential state $(diffState) to $(finalVal) via equation alias"
        else
          @info "Defaulted MTK derivative $(diffState) to 0.0 (no alias equation found)"
        end
      else
        #= Provide 0.0 as a guess (not hard) for Modelica variables without explicit start.
           This gives MTK an initial iterate. If algebraic constraints determine the
           value, MTK can override the guess during initialization. =#
        push!(softInitialValues, diffState => 0.0)
        local currentGuesses2 = ModelingToolkit.guesses(reducedSystem)
        local newGuesses2 = merge(currentGuesses2, Dict(diffState => 0.0))
        @set! reducedSystem.guesses = newGuesses2
        @info "Providing default 0.0 guess for differential state $(diffState) (no explicit start value)"
      end
    end
  end
  #= Final sweep: ensure every reduced unknown has at least a guess.
     The loop above only covers differential states (non-zero mass matrix diagonal).
     Algebraic unknowns that were omitted from finalInitialValues (because
     skipDefaultStarts was true) and are not in diffStateSet fall through with
     nothing. MTK requires every unknown to have either a hard u0 or a guess. =#
  #= Final sweep: ensure every reduced unknown has at least a guess.
     Even when the initialization problem is built, some unknowns may still lack
     coverage. The sweep provides 0.0 guesses as a safety net. =#
  local hardSymStrSetFinal = Set(string(iv.first) for iv in hardInitialValues)
  local currentGuessKeysFinal = Set(string(k) for k in keys(ModelingToolkit.guesses(reducedSystem)))
  local missingUnks = filter(reducedUnks) do unk
    local s = string(unk)
    !(s in hardSymStrSetFinal) && !(s in currentGuessKeysFinal)
  end
  if !isempty(missingUnks)
    local fallbackGuesses = Dict{Any, Any}(unk => 0.0 for unk in missingUnks)
    local currentGuesses3 = ModelingToolkit.guesses(reducedSystem)
    @set! reducedSystem.guesses = merge(currentGuesses3, fallbackGuesses)
    @info "Providing default 0.0 guesses for $(length(missingUnks)) uncovered unknowns: $(join([string(u) for u in missingUnks], ", "))"
  end
  local totalGuesses = length(softInitialValues) + length(missingUnks)
  @info "ODEProblem: DAE, $(length(hardInitialValues)) hard u0 (differential), $(totalGuesses) as guesses (algebraic), $(length(reducedUnks)) unknowns ($(length(diffStateSet)) differential)"
  return (reducedSystem, hardInitialValues)
end

"""
    isPureODESystem(reducedSystem) -> Bool

Return `true` when `reducedSystem`'s mass matrix is the identity (pure ODE),
`false` when it is a singular DAE mass matrix. Callable from generated code
that does not import LinearAlgebra directly.
"""
function isPureODESystem(reducedSystem)
  local mm = ModelingToolkit.calculate_massmatrix(reducedSystem)
  return mm isa LinearAlgebra.UniformScaling
end

"""
    buildDefaultGuesses(reducedSystem, finalInitialValues, allInitialValues)

Build a Dict of default guesses for all reduced unknowns that are NOT already
covered by `finalInitialValues` (hard u0). The init solver uses guesses as
fallback iterates without adding equations, so this does not overdetermine
the system. Values are taken from `allInitialValues` when available, otherwise
defaulted to 0.0.
"""
function buildDefaultGuesses(reducedSystem, finalInitialValues, allInitialValues)
  local reducedUnks = ModelingToolkit.unknowns(reducedSystem)
  local hardKeys = Set(string(iv.first) for iv in finalInitialValues)
  local allIVMap = Dict{String, Any}(string(iv.first) => iv.second for iv in allInitialValues)
  local guessDict = Dict{Any, Any}()
  for unk in reducedUnks
    local unkStr = string(unk)
    if !(unkStr in hardKeys)
      guessDict[unk] = get(allIVMap, unkStr, 0.0)
    end
  end
  if !isempty(guessDict)
    @info "buildDefaultGuesses: $(length(guessDict)) guesses for unknowns not in u0"
  end
  return guessDict
end

"""
    injectObservedEquations(sys, observedEqs)

Append `observedEqs` to the observed equations of a completed system.
Used to inject observed equations AFTER structural_simplify so they do not
interfere with AffectSystem tearing during callback compilation.

Updates the full parent chain so that MTK's getproperty delegation
(which walks parents until it finds a root) can resolve the new variables.
Both `observed` and `var_to_name` are updated at every level.
"""
function injectObservedEquations(sys, observedEqs::Vector)
  local existingObs = ModelingToolkit.get_observed(sys)
  local seenLHS = Set{String}()
  for eq in existingObs
    push!(seenLHS, string(Symbolics.unwrap(eq.lhs)))
  end
  local newEqs = Symbolics.Equation[]
  for eq in observedEqs
    local lhsKey = string(Symbolics.unwrap(eq.lhs))
    if !(lhsKey in seenLHS)
      push!(seenLHS, lhsKey)
      push!(newEqs, eq)
    end
  end
  if isempty(newEqs)
    return sys
  end
  @info "injectObservedEquations: existing=$(length(existingObs)), new=$(length(newEqs))"
  local allObs = vcat(existingObs, newEqs)
  #= Collect the full parent chain: [sys, parent, grandparent, ...] =#
  local chain = [sys]
  local cur = sys
  while true
    local p = ModelingToolkit.get_parent(cur)
    if p === nothing
      break
    end
    push!(chain, p)
    cur = p
  end
  #= Update from the deepest (root) back to sys.
     Each level gets updated observed, var_to_name, and parent pointer. =#
  local updated = nothing
  for i in length(chain):-1:1
    local node = chain[i]
    node = Setfield.set(node, Setfield.PropertyLens{:observed}(), allObs)
    #= Add new observed LHS variables to var_to_name for getproperty lookup =#
    local vtn = copy(ModelingToolkit.get_var_to_name(node))
    for eq in newEqs
      local lhsUW = Symbolics.unwrap(eq.lhs)
      local varName = SymbolicUtils.hasmetadata(lhsUW, Symbolics.VariableSource) ?
        SymbolicUtils.getmetadata(lhsUW, Symbolics.VariableSource)[2] : nothing
      if varName !== nothing
        vtn[varName] = lhsUW
      end
    end
    node = Setfield.set(node, Setfield.PropertyLens{:var_to_name}(), vtn)
    if updated !== nothing
      node = Setfield.set(node, Setfield.PropertyLens{:parent}(), updated)
    end
    updated = node
  end
  return updated
end

"""
  TODO:
  Document why some parts here are outcommented
  The irreductable variables scheme does not work using plain simplify.

  It should be noted that for some models both running tearing and structurally simplify is needed.
  Report and issue for the MTK reporters giving an example of this behavior.

  One example is running tearing twice broke the system
"""
function structural_simplify(sys::ModelingToolkit.AbstractSystem,
                             io = nothing;
                             simplify = false,
                             allow_parameter = true,
                             kwargs...)
  local pre_eqs = length(equations(sys))
  local pre_unknowns = length(unknowns(sys))
  @info "Before structural_simplify: $pre_eqs equations, $pre_unknowns unknowns"
  @BACKEND_LOGGING begin
    open(OMBackend.logPath("backend/mtk", "mtk_preSimplify.log"), "w") do io
      println(io, "############################################")
      println(io, "MTK system before structural_simplify")
      println(io, "############################################")
      println(io)
      println(io, "Unknowns ($pre_unknowns):")
      println(io, "---------------------------------------------")
      for (i, u) in enumerate(unknowns(sys))
        println(io, "  [$i] $u")
      end
      println(io)
      println(io, "Equations ($pre_eqs):")
      println(io, "---------------------------------------------")
      for (i, eq) in enumerate(equations(sys))
        println(io, "  [$i] $eq")
      end
      println(io)
      println(io, "Parameters ($(length(parameters(sys)))):")
      println(io, "---------------------------------------------")
      for (i, p) in enumerate(parameters(sys))
        println(io, "  [$i] $p")
      end
      try
        local defs = ModelingToolkit.defaults(sys)
        println(io)
        println(io, "Defaults ($(length(defs))):")
        println(io, "---------------------------------------------")
        for (k, v) in defs
          println(io, "  $k => $v")
        end
      catch
        println(io, "\n(defaults not available in this MTK version)")
      end
    end
  end
  #= DirectRHS uses a plain Float64[] parameter vector, so the reduced system
     must use split=false (flat vector) rather than split=true (MTKParameters).
     This also makes generate_continuous_callbacks produce callbacks compatible
     with the flat format. The non-DirectRHS path (VSS, standard ODEProblem)
     needs split=true for SCCNonlinearProblem initialization to work.
     The split value is embedded at codegen time by performStructuralSimplify. =#
  #= Diagnostic: scan for array-shaped subtrees before structural_simplify.
     Gated behind @BACKEND_LOGGING (compile-time) so production runs pay
     nothing. _LAST_ARRAY_SHAPE_COUNT[] stays at its sentinel -1 in that
     case; tests that need the invariant must run with ENABLE_BACKEND_LOGGING
     set, and the assertion in mslTests.jl handles the off case. =#
  @BACKEND_LOGGING begin
    let
      function _find_array_shapes(expr, path, results; depth=0)
        depth > 50 && return  # guard against infinite recursion
        if expr isa SymbolicUtils.BasicSymbolic
          local sh = try; Symbolics.shape(expr); catch; nothing; end
          if sh !== nothing && sh !== () && SymbolicUtils.is_array_shape(sh)
            push!(results, (path, sh, expr))
            return  # do not recurse further into this subtree
          end
          if SymbolicUtils.iscall(expr)
            local args = SymbolicUtils.arguments(expr)
            for (j, a) in enumerate(args)
              _find_array_shapes(a, "$path.args[$j]", results; depth=depth+1)
            end
          end
        elseif expr isa AbstractArray
          push!(results, (path, :raw_array, expr))
        end
      end
      local allResults = []
      for (i, eq) in enumerate(equations(sys))
        local lhsR = Pair{String,Any}[]
        local rhsR = Pair{String,Any}[]
        _find_array_shapes(Symbolics.unwrap(eq.lhs), "eq[$i].lhs", lhsR)
        _find_array_shapes(Symbolics.unwrap(eq.rhs), "eq[$i].rhs", rhsR)
        for (p, sh, node) in lhsR
          push!(allResults, (i, p, sh, node))
        end
        for (p, sh, node) in rhsR
          push!(allResults, (i, p, sh, node))
        end
      end
      _LAST_ARRAY_SHAPE_COUNT[] = length(allResults)
      if !isempty(allResults)
        @warn "Found $(length(allResults)) array-shaped subtree(s) in equations before structural_simplify"
        for (idx, p, sh, node) in allResults[1:min(10, length(allResults))]
          nodeStr = try; string(node)[1:min(200, length(string(node)))]; catch e; "$(typeof(node))"; end
          @warn "  eq $idx at $p: shape=$sh node=$nodeStr"
        end
      else
        @info "No array-shaped subtrees found in equations (clean for Pantelides)"
      end
    end
  end

  local useSplit = get(kwargs, :split, true)
  if OMBackend.ENABLE_BACKEND_LOGGING
    local _ss_timed = @timed ModelingToolkit.structural_simplify(sys; simplify = simplify, split = useSplit)
    sys = _ss_timed.value
    @info "structural_simplify took $(_ss_timed.time)s, $(round(_ss_timed.bytes / 1e9, digits=2)) GiB"
  else
    sys = ModelingToolkit.structural_simplify(sys; simplify = simplify, split = useSplit)
  end
  local post_eqs = length(equations(sys))
  local post_full_eqs = length(ModelingToolkit.full_equations(sys))
  local post_unknowns = length(unknowns(sys))
  @info "After structural_simplify: equations=$(post_eqs), full_equations=$(post_full_eqs), unknowns=$(post_unknowns)"
  if post_eqs != post_unknowns
    @warn "equations(sys) != unknowns(sys): $post_eqs vs $post_unknowns"
    for (i, eq) in enumerate(equations(sys))
      @info "  eq[$i]: $eq"
    end
    for (i, u) in enumerate(unknowns(sys))
      @info "  unk[$i]: $u"
    end
  end
  if post_full_eqs != post_unknowns
    @warn "full_equations(sys) != unknowns(sys): $post_full_eqs vs $post_unknowns"
  end
  @BACKEND_LOGGING begin
    open(OMBackend.logPath("backend/mtk", "mtk_postSimplify.log"), "w") do io
      println(io, "############################################")
      println(io, "MTK system after structural_simplify")
      println(io, "############################################")
      println(io)
      println(io, "Unknowns ($post_unknowns):")
      for (i, u) in enumerate(unknowns(sys))
        println(io, "  [$i] $u")
      end
      println(io)
      println(io, "Equations ($post_eqs):")
      for (i, eq) in enumerate(equations(sys))
        println(io, "  [$i] $eq")
      end
      println(io)
      println(io, "Guesses ($(length(ModelingToolkit.guesses(sys)))):")
      for (k, v) in ModelingToolkit.guesses(sys)
        println(io, "  $k => $v")
      end
      println(io)
      println(io, "Full equations ($post_full_eqs):")
      for (i, eq) in enumerate(ModelingToolkit.full_equations(sys))
        println(io, "  [$i] $eq")
      end
    end
  end
  #= Workaround: after structural_simplify, the tearing state's graph may have
     stale dimensions (more equations/variables than equations(sys)/unknowns(sys)).
     This causes a DimensionMismatch in W_sparsity when the graph-based Jacobian
     sparsity (nsrcs x nsrcs) is broadcast against the mass matrix (neqs x neqs).
     Clearing the tearing state forces jacobian_sparsity to fall back to symbolic
     computation, which uses the correct equation/unknown counts. =#
  try
    local ts = ModelingToolkit.get_tearing_state(sys)
    if ts !== nothing
      local g = ts.structure.graph
      local graph_eqs = ModelingToolkit.BipartiteGraphs.nsrcs(g)
      local sys_eqs = length(equations(sys))
      if graph_eqs != sys_eqs
        @warn "Tearing state graph has $graph_eqs equations but system has $sys_eqs; clearing tearing state"
        @set! sys.tearing_state = nothing
      end
    end
  catch ex
    @warn "Could not inspect tearing state" exception=(ex, catch_backtrace())
  end
  #= Filter guesses to only include variables that are unknowns of the reduced system.
     After structural_simplify, many original unknowns become observed (algebraically
     determined). Keeping guesses for those creates an overdetermined initialization. =#
  local reducedUnknowns = Set(unknowns(sys))
  local currentGuesses = ModelingToolkit.guesses(sys)
  local filteredGuesses = Dict(k => v for (k, v) in currentGuesses if k in reducedUnknowns)
  if length(filteredGuesses) != length(currentGuesses)
    @info "Filtered guesses: $(length(currentGuesses)) -> $(length(filteredGuesses)) (removed $(length(currentGuesses) - length(filteredGuesses)) non-unknown guesses)"
    @set! sys.guesses = filteredGuesses
  end
  return sys
end

"""
  $(TYPEDSIGNATURES)

  Takes a Nth order System and returns a new System written in first order
  form by defining new variables which represent the N-1 derivatives.
  """
function ode_order_lowering(sys::System)
  iv = ModelingToolkit.get_iv(sys)
  eqs_lowered, new_vars = ode_order_lowering(equations(sys), iv, unknowns(sys))
  @set! sys.eqs = eqs_lowered
  @set! sys.unknowns = new_vars
  return sys
end

function dae_order_lowering(sys::System)
  iv = get_iv(sys)
  eqs_lowered, new_vars = dae_order_lowering(equations(sys), iv, unknowns(sys))
  @set! sys.eqs = eqs_lowered
  @set! sys.unknowns = new_vars
  return sys
end

function ode_order_lowering(eqs, iv, unknown_vars)
  var_order = OrderedDict{Any, Int}()
  D = Differential(iv)
  diff_eqs = Equation[]
  diff_vars = []
  alge_eqs = Equation[]
  for (i, eq) in enumerate(eqs)
    if !isdiffeq(eq)
      push!(alge_eqs, eq)
    else
      var, maxorder = ModelingToolkit.var_from_nested_derivative(eq.lhs)
      maxorder > get(var_order, var, 1) && (var_order[var] = maxorder)
      var′ = ModelingToolkit.lower_varname(var, iv, maxorder - 1)
      if ! isreal(eq.rhs) #= Modification by me. =#
        rhs′ = ModelingToolkit.diff2term_with_unit(eq.rhs, iv)
      else
        rhs′ = eq.rhs
      end
      push!(diff_vars, var′)
      push!(diff_eqs, D(var′) ~ rhs′)
    end
  end
  for (var, order) in var_order
    for o in (order - 1):-1:1
      lvar = lower_varname(var, iv, o - 1)
      rvar = lower_varname(var, iv, o)
      push!(diff_vars, lvar)

      rhs = rvar
      eq = Differential(iv)(lvar) ~ rhs
      push!(diff_eqs, eq)
    end
  end
  # we want to order the equations and variables to be `(diff, alge)`
  return (vcat(diff_eqs, alge_eqs), vcat(diff_vars, setdiff(unknown_vars, diff_vars)))
end

function dae_order_lowering(eqs, iv, unknown_vars)
  var_order = OrderedDict{Any, Int}()
  D = Differential(iv)
  diff_eqs = Equation[]
  diff_vars = OrderedSet()
  alge_eqs = Equation[]
  vars = Set()
  subs = Dict()

  for (i, eq) in enumerate(eqs)
    vars!(vars, eq)
    n_diffvars = 0
    for vv in vars
      isdifferential(vv) || continue
      var, maxorder = var_from_nested_derivative(vv)
      isparameter(var) && continue
      n_diffvars += 1
      order = get(var_order, var, nothing)
      seen = order !== nothing
      if !seen
        order = 1
      end
      maxorder > order && (var_order[var] = maxorder)
      var′ = lower_varname(var, iv, maxorder - 1)
      subs[vv] = D(var′)
      if !seen
        push!(diff_vars, var′)
      end
    end
    n_diffvars == 0 && push!(alge_eqs, eq)
    empty!(vars)
  end

  for (var, order) in var_order
    for o in (order - 1):-1:1
      lvar = lower_varname(var, iv, o - 1)
      rvar = lower_varname(var, iv, o)
      push!(diff_vars, lvar)

      rhs = rvar
      eq = Differential(iv)(lvar) ~ rhs
      push!(diff_eqs, eq)
    end
  end

  return ([diff_eqs; substitute.(eqs, (subs,))],
          vcat(collect(diff_vars), setdiff(unknown_vars, diff_vars)))
end

function getStatesAsSymbolicVariables(odeFunc::ODEFunction)
  return ModelingToolkit.get_unknowns(odeFunc.sys)
end

function getStatesAsSymbols(odeFunc::ODEFunction)
  local states = ModelingToolkit.get_unknowns(odeFunc.sys)
  map(x->x.f.name, states)
end

function getParametersAsSymbols(odeFunc::ODEFunction)
  local params = ModelingToolkit.parameters(odeFunc.sys)
  map(params) do x
    local uw = SymbolicUtils.unwrap(x)
    #= Regular parameters are Sym (have .name), time-dependent discrete
       parameters like ifCond(t) are Term (have .f.name). =#
    hasproperty(uw, :name) ? uw.name : uw.f.name
  end
end

function getSymsAsStrings(odeFunc::ODEFunction)
  local unknowns = ModelingToolkit.parameters(odeFunc.sys)
  return map(string, unknowns)
end

"""
Convert an ODEProblem (possibly with mass matrix) to a DAEProblem in residual
form F(du, u, p, t) = M*du - f(u, p, t) = 0. This allows solvers like
Sundials.IDA to be used, which share the same BDF/DASPK lineage as
OpenModelica's DASSL and produce closely matching results.

Handles both pure ODEs (UniformScaling mass matrix) and semi-explicit DAEs
(singular mass matrix from structural_simplify).
"""
function ode_to_dae(prob::ODEProblem)
  local M = prob.f.mass_matrix
  local ode_f! = prob.f.f
  local n = length(prob.u0)
  local M_mat = M isa UniformScaling ? Matrix{Float64}(M, n, n) : M
  function residual!(resid, du, u, p, t)
    ode_f!(resid, u, p, t)
    resid .= M_mat * du .- resid
  end
  local du0 = zeros(n)
  local tmp = zeros(n)
  ode_f!(tmp, prob.u0, prob.p, prob.tspan[1])
  for i in 1:n
    if M_mat[i, i] != 0.0
      du0[i] = tmp[i] / M_mat[i, i]
    end
  end
  local diff_vars = [M_mat[i, i] != 0.0 for i in 1:n]
  SciMLBase.DAEProblem(residual!, du0, prob.u0, prob.tspan, prob.p;
                       differential_vars = diff_vars)
end
