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
!Adjust the unnecessary string conversions!
=#


# MTK-stage dump helpers (see CodeGeneration/mtkDump.jl).
import .MTKDump: dumpMTKPreSimplify, dumpMTKPostSimplify

#= Global dictionary to store dynamically generated function implementations =#
#= The key is the function name, the value is the implementation function =#
const MODELICA_FUNCTION_IMPLS = Dict{Symbol, Function}()

#= Global dictionary to store RTG wrappers for each function =#
const MODELICA_FUNCTION_WRAPPERS = Dict{Symbol, Any}()

#= A Modelica function's wrapper (createModelicaFunctionWrapper): its
   RuntimeGeneratedFunction behind a method of the function's arity. An RGF
   binds its arguments unchecked, so a call with fewer read past the argument
   tuple and crashed the process (MSL Water IF97, 2026-09-30); here a wrong
   argument count is a MethodError. =#
struct ModelicaFunctionWrapper{N, F} <: Function
  name::Symbol
  rgf::F
end
ModelicaFunctionWrapper{N}(name::Symbol, rgf::F) where {N, F} = ModelicaFunctionWrapper{N, F}(name, rgf)
(w::ModelicaFunctionWrapper{N})(args::Vararg{Any, N}) where {N} = w.rgf(args...)
Base.nameof(w::ModelicaFunctionWrapper) = w.name
Base.show(io::IO, w::ModelicaFunctionWrapper) = print(io, w.name)

#= The same hash in every process for the functions of this module's terms
   (the wrappers above and the element extractors below). By default a
   RuntimeGeneratedFunction hashes by the address of its body Expr, and so
   does a wrapper holding one: the hash of every term calling a Modelica
   function changed from one process to the next, and with it the order of
   MTK's dictionaries, its alias choices and the summation order of the
   generated code. A stiff model at a loose tolerance went from Success to
   Unstable or InitialFailure by process (MSL DifferenceAmplifier with QNDF
   at 1e-2, 4 of 52 verify runs). An RGF's type holds a hash of its body's
   content (`id`), so the type identifies the function. =#
const _RGF_TAG = getfield(@__MODULE__, Symbol("#_RGF_ModTag"))
Base.hash(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction{<:Any, _RGF_TAG, _RGF_TAG}, h::UInt) = hash(typeof(f), h)
Base.hash(w::ModelicaFunctionWrapper, h::UInt) = hash(typeof(w), hash(w.name, h))

#= der() of a call of a function with a derivative annotation that stays a term (an
   if-statement on its input: Buildings' equalPercentage, DerivativeCheck): the partial
   derivative by an input is the derivative function with that input's derivative 1 and the
   others' 0 (the derivative function is linear in them); 0 by an input without one
   (zeroDerivative, noDerivative, not Real). Name => (derivative function, the inputs whose
   derivatives it takes), set by the generated functions (generateFunctions). =#
const FUNCTION_DERIVATIVE_RULES = Dict{Symbol, Tuple{Symbol, Vector{Int}}}()

function Symbolics.derivative_rule(w::ModelicaFunctionWrapper{N}, ::Val{N},
                                   args::SymbolicUtils.ROArgsT{Symbolics.VartypeT}, ::Val{I}) where {N, I}
  local rule = get(FUNCTION_DERIVATIVE_RULES, w.name, nothing)
  local dw = rule === nothing ? nothing : get(MODELICA_FUNCTION_WRAPPERS, rule[1], nothing)
  dw === nothing && return NUMERIC_PARTIALS[] ? _partialTerm(ModelicaFunctionPartial{N, 1}(w.name, (I,)), args) : nothing
  local withDer = rule[2]
  local k = findfirst(==(I), withDer)
  k === nothing && return Symbolics.SConst(0)
  local r = Symbolics.unwrap(dw(args..., ntuple(j -> j == k ? 1.0 : 0.0, length(withDer))...))
  return r isa SymbolicUtils.BasicSymbolic ? r : Symbolics.SConst(r)
end

#= The partials below in the index reduction only for a build that failed without them
   (iMTKGen._buildAndCache): a rule for every call let ModelingToolkit differentiate calls it
   otherwise leaves alone, central differences in the index reduction (MSL AIMC_Conveyor:
   Unstable at 10 s). The Jacobian and the time derivative always have them
   (DirectRHSGeneration). =#
const NUMERIC_PARTIALS = Ref(false)

"""
    withNumericPartials(f)

Run `f()` with the numeric partials of calls without a derivative annotation, restoring
the previous setting afterwards.
"""
function withNumericPartials(f::Function)
  local previous = NUMERIC_PARTIALS[]
  NUMERIC_PARTIALS[] = true
  try
    return f()
  finally
    NUMERIC_PARTIALS[] = previous
  end
end

#= The partial derivative of a function call that stays a term and has no derivative
   annotation (an if-statement on an input: Buildings' smoothExponential, Media property
   functions; the derivative function of an annotation, for second derivatives): the
   function's implementation by central differences at run time (external C functions too),
   by the inputs `idx` in turn. Its own partials nest. =#
struct ModelicaFunctionPartial{N, K} <: Function
  name::Symbol
  idx::NTuple{K, Int}
end
Base.nameof(p::ModelicaFunctionPartial) = Symbol(p.name, "_d", Base.join(p.idx, "_"))
Base.show(io::IO, p::ModelicaFunctionPartial) = print(io, nameof(p))
Base.hash(p::ModelicaFunctionPartial, h::UInt) = hash(p.idx, hash(p.name, hash(:ModelicaFunctionPartial, h)))

function (p::ModelicaFunctionPartial{N, K})(args::Vararg{Any, N}) where {N, K}
  any(a -> a isa Symbolics.Num || a isa SymbolicUtils.BasicSymbolic, args) &&
    return _partialTerm(p, Any[Symbolics.unwrap(a) for a in args])
  return _centralPartial(MODELICA_FUNCTION_IMPLS[p.name], args, p.idx, eps(Float64)^(1 / (K + 2)))
end

function _centralPartial(impl, args::Tuple, idx::Tuple, rel::Float64)::Float64
  if isempty(idx)
    local r = Base.invokelatest(impl, args...)
    return Float64(r isa Tuple ? first(r) : r)
  end
  local i = last(idx)
  local x = Float64(args[i])
  local h = rel * max(1.0, abs(x))
  local up = _centralPartial(impl, Base.setindex(args, x + h, i), Base.front(idx), rel)
  local dn = _centralPartial(impl, Base.setindex(args, x - h, i), Base.front(idx), rel)
  return (up - dn) / (2h)
end

_partialTerm(p::ModelicaFunctionPartial, args::AbstractVector) = SymbolicUtils.Term{SymbolicUtils.SymReal}(p, collect(Any, args); type = Real)

function Symbolics.derivative_rule(p::ModelicaFunctionPartial{N, K}, ::Val{N},
                                   args::SymbolicUtils.ROArgsT{Symbolics.VartypeT}, ::Val{I}) where {N, K, I}
  return _partialTerm(ModelicaFunctionPartial{N, K + 1}(p.name, (p.idx..., I)), args)
end

#= Cache for per-element extractor functions.
   Key: (funcName::Symbol, indices::Tuple{Vararg{Int}}, nArgs::Int)
   Value: the created function object
   nArgs is part of the key because the extractor RGF is built with
   fixed arity — calling an extractor with the wrong number of args
   triggers @inbounds __args[i] out-of-bounds → LLVM unreachable → SIGILL. =#
const ELEM_FUNC_CACHE = Dict{Tuple{Symbol, Tuple, Int}, Any}()
const TUPLE_ELEM_FUNC_CACHE = Dict{Tuple, Any}()

function _isKnownBooleanAnyTrueFunction(normalizedFuncName::String)::Bool
  return normalizedFuncName == "Modelica_StateGraph_Temporary_anyTrue" ||
         normalizedFuncName == "Modelica_Math_BooleanVectors_anyTrue"
end

function _isKnownBooleanAllTrueFunction(normalizedFuncName::String)::Bool
  return normalizedFuncName == "Modelica_StateGraph_Temporary_allTrue" ||
         normalizedFuncName == "Modelica_Math_BooleanVectors_allTrue"
end

function _booleanVectorTerms(expLst,
                             simCode,
                             hashTable;
                             varPrefix = "",
                             varSuffix = "",
                             derSymbol = false)
  local nArgs = 0
  local onlyArg = nothing
  for arg in expLst
    nArgs += 1
    onlyArg = arg
  end
  nArgs == 1 || return nothing

  local terms = Any[]
  @match onlyArg begin
    DAE.ARRAY(_, _, array) => begin
      for e in array
        push!(terms, expToJuliaExpMTK(e, simCode;
                                      varPrefix = varPrefix,
                                      varSuffix = varSuffix,
                                      derSymbol = derSymbol))
      end
    end
    _ => return nothing
  end

  return terms
end

function _booleanAnyTrueCallExpr(expLst,
                                 simCode,
                                 hashTable;
                                 varPrefix = "",
                                 varSuffix = "",
                                 derSymbol = false)
  local terms = _booleanVectorTerms(expLst, simCode, hashTable;
                                    varPrefix = varPrefix,
                                    varSuffix = varSuffix,
                                    derSymbol = derSymbol)
  terms === nothing && return nothing
  isempty(terms) && return :(0)
  local result = :(1 - $(terms[1]))
  for term in terms[2:end]
    result = :($result * (1 - $term))
  end
  return :(1 - $result)
end

function _booleanAllTrueCallExpr(expLst,
                                 simCode,
                                 hashTable;
                                 varPrefix = "",
                                 varSuffix = "",
                                 derSymbol = false)
  local terms = _booleanVectorTerms(expLst, simCode, hashTable;
                                    varPrefix = varPrefix,
                                    varSuffix = varSuffix,
                                    derSymbol = derSymbol)
  terms === nothing && return nothing
  isempty(terms) && return :(1)
  local result = terms[1]
  for term in terms[2:end]
    result = :($result * $term)
  end
  return result
end

"""
Lower known Modelica function calls that have a stable symbolic equivalent.

This keeps simple Boolean reductions out of the opaque RuntimeGeneratedFunction
path. MTK can rebuild opaque terms with `symtype=Any` during alias elimination,
which later trips SymbolicUtils arithmetic in linear-coefficient extraction.
"""
function lowerKnownSymbolicFunctionCall(normalizedFuncName::String,
                                        expLst,
                                        simCode,
                                        hashTable;
                                        varPrefix = "",
                                        varSuffix = "",
                                        derSymbol = false)
  if _isKnownBooleanAnyTrueFunction(normalizedFuncName)
    return _booleanAnyTrueCallExpr(expLst, simCode, hashTable;
                                  varPrefix = varPrefix,
                                  varSuffix = varSuffix,
                                  derSymbol = derSymbol)
  elseif _isKnownBooleanAllTrueFunction(normalizedFuncName)
    return _booleanAllTrueCallExpr(expLst, simCode, hashTable;
                                  varPrefix = varPrefix,
                                  varSuffix = varSuffix,
                                  derSymbol = derSymbol)
  end
  return nothing
end

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
  #= Results pass through exactly. Converting to Float64 corrupts Integer
     results (RNG state words exceed 2^53) and float-ifies integer exponents,
     so non-float Reals must keep their type. =#
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

  #= Results pass through exactly. Converting to Float64 corrupts Integer
     results (RNG state words exceed 2^53) and float-ifies integer exponents,
     so non-float Reals must keep their type. =#
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

# OMBackend always wraps Modelica-function impls in RTG callables (a function's
# own behind a ModelicaFunctionWrapper) that return Real (or tuples of Real, but
# extractor RGFs project a single Real). Pin the
# symtype here so any SymbolicUtils path that hits _promote_symtype on an RTG
# (e.g. hashcons-cached Terms, internal rewrites, default safe_ctors.jl Term
# construction) yields Real instead of Any. Without this, an Any-typed Term
# can be cached and later returned even when makeSymbolicTerm passes
# `type = Real` explicitly, poisoning subsequent sums and breaking
# `-(::SymReal, ::SymReal)` in MTK alias_elimination.
SymbolicUtils._promote_symtype(::Union{RuntimeGeneratedFunctions.RuntimeGeneratedFunction, ModelicaFunctionWrapper, ModelicaFunctionPartial}, args) = Real
SymbolicUtils._promote_symtype(::typeof(floor), args) = Real

# Same reason for shape. Scalar-returning extractor RGFs must be reported as
# scalar (`ShapeVecT()`) rather than the default `Unknown(-1)` so they can be
# summed against other scalar terms without triggering `_added_shape`'s rank
# mismatch in MTK's substitution rebuild path (terminterface.jl `maketerm` for
# `+`/`-`). Without this, a wrapper whose body returns a Modelica vector can
# poison a sum with shape `Unknown(2)`.
SymbolicUtils.promote_shape(::Union{RuntimeGeneratedFunctions.RuntimeGeneratedFunction, ModelicaFunctionWrapper, ModelicaFunctionPartial}, args::SymbolicUtils.ShapeT...) = SymbolicUtils.ShapeVecT()
SymbolicUtils.promote_shape(::typeof(floor), args::SymbolicUtils.ShapeT...) = SymbolicUtils.ShapeVecT()
SymbolicUtils.promote_shape(::typeof(floor), arg::SymbolicUtils.Unknown) = SymbolicUtils.ShapeVecT()

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

#= Closed-form symbolic expansions for library functions whose general
   implementation branches on symbolic values (so eager evaluation fails).
   An expander receives the original call arguments and returns the full
   result (tuple for tuple-returning functions), or nothing when the call
   shape is outside its closed form. Expanded calls stay differentiable and
   never reach the opaque invokelatest extractors. =#
const EAGER_SYMBOLIC_EXPANSIONS = Dict{Symbol, Function}()

#= A 2-point abscissa has a single interval: linear inter-/extrapolation,
   loop-free. The second tuple element is the returned interval index. =#
EAGER_SYMBOLIC_EXPANSIONS[:Modelica_Math_Vectors_interpolate] = function (args...)
  length(args) < 3 && return nothing
  local x, y, xi = args[1], args[2], args[3]
  (x isa AbstractVector && y isa AbstractVector) || return nothing
  (length(x) == 2 && length(y) == 2) || return nothing
  local yi = y[1] + (xi - x[1]) * ((y[2] - y[1]) / (x[2] - x[1]))
  return (yi, 1)
end

#= What an eager evaluation of a Modelica function throws for symbolic
   arguments: a MethodError of the implementation (an operation without a
   symbolic method), a FieldError (a record field read from a symbolic
   value, as in the MSL records' functions), an UndefVarError (the MSL
   QuasiStatic machines' Complex functions: `v` undefined in the model
   module, open); a TypeError (a condition on a symbolic value) is not a
   programming error anyway. The opaque extractors then take over. =#
const _EAGER_SYMBOLIC_FAILURE = Union{MethodError, FieldError, UndefVarError}

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
    local expander = get(EAGER_SYMBOLIC_EXPANSIONS, funcName, nothing)
    if expander !== nothing
      local expanded = expander(args...)
      expanded === nothing || return expanded[ix]
    end
    #= Try eager evaluation: call impl with original (Num-wrapped) args.
       Produces elementary symbolic expressions Symbolics can differentiate.
       Falls back to opaque RTG extractors if eval fails (if-statement on symbolic). =#
    try
      local impl = MODELICA_FUNCTION_IMPLS[funcName]
      local preparedArgs = _prepareArgsForEagerEval(args)
      local result = Base.invokelatest(impl, preparedArgs...)
      return result[ix]
    catch err
      #= Symbolic arguments: the opaque extractors below (_EAGER_SYMBOLIC_FAILURE). =#
      OMBackend._fallback(err, :eagerFunctionElement; expect = _EAGER_SYMBOLIC_FAILURE)
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
  #= Results pass through exactly. Converting to Float64 corrupts Integer
     results (RNG state words exceed 2^53) and float-ifies integer exponents,
     so non-float Reals must keep their type. =#
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

  #= Results pass through exactly. Converting to Float64 corrupts Integer
     results (RNG state words exceed 2^53) and float-ifies integer exponents,
     so non-float Reals must keep their type. =#
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
    catch err
      #= Symbolic arguments: the opaque extractors below (_EAGER_SYMBOLIC_FAILURE). =#
      OMBackend._fallback(err, :eagerFunctionTupleElement; expect = _EAGER_SYMBOLIC_FAILURE)
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
    catch err
      #= Symbolic arguments: the opaque extractors below (_EAGER_SYMBOLIC_FAILURE). =#
      OMBackend._fallback(err, :eagerFunctionMatrixElement; expect = _EAGER_SYMBOLIC_FAILURE)
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
  #= Result passes through exactly; Float64() here corrupts Integer results. =#
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
#= Named functor for the symbolic-argument table lookup. Must be a NAMED type, not
   an anonymous closure: MTK's `substitute` / `alias_elimination` rebuild the term
   via `maketerm`, which RE-INFERS the symtype from `promote_symtype(op, ...)`,
   ignoring the explicit `type = Real` set in `makeSymbolicTerm`. An anonymous
   closure has no `promote_symtype` method, so the rebuilt term defaults to symtype
   `Any`; arithmetic on it then trips the SymReal/`Any` subtraction guard in
   `find_eq_solvables!` (`-(::SymReal, ::SymReal)` throws when a symtype is
   non-numeric). Declaring the result `Real` keeps the rebuilt term numeric. =#
struct ConstTableLookupFn{A <: AbstractArray}
  table::A
end
#= By its table's content: the default hashed the table's address, which
   differs by process (see `_RGF_TAG`). =#
Base.hash(f::ConstTableLookupFn, h::UInt) = hash(f.table, hash(:ConstTableLookupFn, h))

#= Primal value of a table index. A constant table lookup is piecewise-constant
   in its (discrete/enum) index, so an autodiff `Dual` index must collapse to its
   primal before rounding to an integer: this both avoids `Float64(::Dual)` in the
   nonlinear-solve Jacobian and gives the lookup an identically-zero derivative
   w.r.t. the index (the returned table entry carries no partials). Dependency-free
   so OMBackend need not depend on ForwardDiff; `Dual` exposes a `.value` field. =#
function _primalValue(@nospecialize(v))
  v isa Integer && return v
  v isa AbstractFloat && return v
  hasproperty(v, :value) && return _primalValue(getproperty(v, :value))
  return Float64(v)
end

function (c::ConstTableLookupFn)(rt_idxs...)
  local resolved = ntuple(length(rt_idxs)) do k
    local v = rt_idxs[k]
    local raw = v isa Integer ? Int(v) : Int(round(_primalValue(v)))
    clamp(raw, 1, size(c.table, k))
  end
  return c.table[resolved...]
end
SymbolicUtils.promote_symtype(::ConstTableLookupFn, args...) = Real
#= maketerm RE-INFERS shape too, not just symtype; the generic fallback yields
   Unknown, which trips `promote_shape(==, Unknown, scalar)` once the rebuilt term
   lands in a relational during substitute/full_equations. The lookup is scalar. =#
SymbolicUtils.promote_shape(::ConstTableLookupFn, szs::SymbolicUtils.ShapeT...) = SymbolicUtils.ShapeVecT()

function constTableLookup(table::AbstractArray, idxs...)
  if hasSymbolicArgs(idxs...)
    return makeSymbolicTerm(ConstTableLookupFn(table), Any[idxs...])
  else
    return table[ntuple(k -> clamp(_toIndex(idxs[k]), 1, size(table, k)),
                       length(idxs))...]
  end
end

#= Clamp the resolved table index to the valid range [1, size(table, k)].
   During DAE initialization the integer / enum simvars that index into
   constant tables are still at their default 0.0 (Float64 unknown
   storage) until the `start` attributes are applied. Without clamping,
   a `constTableLookup` invoked during the init Newton solve hits a
   BoundsError at index 0. Treating an out-of-range index as the first
   row/column (typically the "Unknown" / "U" entry in MSL Digital
   tables) is consistent with Modelica §17.4.4 (uninitialised discretes
   are treated as their start values, and the init solver is allowed to
   probe the residual with provisional values that have not yet been
   pinned). Once init completes the indices are valid and clamping is
   a no-op. =#
function _toIndex(v)
  if v isa Integer
    return Int(v)
  else
    return Int(round(_primalValue(v)))
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
  local expander = get(EAGER_SYMBOLIC_EXPANSIONS, funcName, nothing)
  if expander !== nothing
    local expanded = expander(origArgs...)
    expanded === nothing || return expanded
  end
  try
    local impl = MODELICA_FUNCTION_IMPLS[funcName]
    local preparedArgs = _prepareArgsForEagerEval(origArgs)
    local result = Base.invokelatest(impl, preparedArgs...)
    if isArray && !isempty(dims)
      return _ensureArrayShape(result, dims)
    end
    return result
  catch err
    OMBackend._fallback(err, :eagerFunctionCall; expect = _EAGER_SYMBOLIC_FAILURE)
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
  MODELICA_FUNCTION_WRAPPERS[funcName] = ModelicaFunctionWrapper{nArgs}(funcName, rtg)
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
  local funcNames = OrderedSet{Symbol}(Symbol(f.name) for f in simCode.functions)
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
    #= Recursively process arguments; allocate a new args vector only when a
       child actually changes, else return the original Expr unchanged. =#
    local newArgs = nothing
    for (i, a) in enumerate(expr.args)
      local r = wrapWithInvokelatest(a)
      if r !== a
        newArgs === nothing && (newArgs = copy(expr.args))
        newArgs[i] = r
      end
    end
    return newArgs === nothing ? expr : Expr(:call, newArgs...)
  end
  #= For all other Expr types; allocate only when a child changes. =#
  local newArgs = nothing
  for (i, a) in enumerate(expr.args)
    local r = wrapWithInvokelatest(a)
    if r !== a
      newArgs === nothing && (newArgs = copy(expr.args))
      newArgs[i] = r
    end
  end
  return newArgs === nothing ? expr : Expr(expr.head, newArgs...)
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
    @debug "[MTK GEN: cleanup] Removed $n_removed constant-only equations (no unknowns)"
  end
  return filtered
end

#= der(<expression>) in the continuous equations by the chain rule, down to
   derivatives of variables (MSL FluxTubes' Tellinen hysteresis:
   `dHyst = der(hystR - mu0*Hstat)`): MTK takes a Differential only of an
   unknown. A variable an equation `x ~ f` defines explicitly (hystR) has
   der(f) for der(x), as OpenModelica differentiates it: left as D(x), index
   reduction made x a state and recovered the chain behind f backwards
   (P3 = (... - hystR)/P4, H3 from P3's atan), singular once the branch
   saturated. Only in these equations; the model's own der(x) stay. An
   equation with a function Symbolics has no derivative rule for (a Modelica
   function, a table lookup) stays as it is. The initial equations keep
   D(expr), which the init solve differentiates itself
   (_observedDerivativeTargets). =#
function expandExpressionDerivatives(eqs::AbstractVector)
  local touched = BitVector(_hasExpressionDerivative(eq.lhs) || _hasExpressionDerivative(eq.rhs) for eq in eqs)
  any(touched) || return eqs
  local definitions = _explicitDefinitions(eqs)
  local memo = Dict{Any, Any}()
  local side = ex -> _derivativesByDefinitions(_expandDerivatives(ex), definitions, memo, Set{Any}())
  return map(eachindex(eqs)) do i
    touched[i] || return eqs[i]
    try
      side(eqs[i].lhs) ~ side(eqs[i].rhs)
    catch e
      e isa InterruptException && rethrow()
      eqs[i]
    end
  end
end

#= Symbolics' chain rule, throwing at a function without a derivative rule
   (by default it writes D(f(u))*D(u) for it). =#
_expandDerivatives(ex) = Symbolics.expand_derivatives(ex, false; throw_no_derivative = true)

#= The Differential terms of `ex` (or of a vector or tuple of them), outermost
   ones: get_variables keeps a Differential term whole (an Operator is atomic)
   and, unlike a walk over `arguments`, builds no argument lists (every
   model's equations pass through _hasExpressionDerivative). =#
function _differentials(ex)
  local out = OrderedSet{Any}()
  for e in (ex isa Union{AbstractArray, Tuple} ? ex : (ex,)), v in Symbolics.get_variables(e)
    SymbolicUtils.iscall(v) && SymbolicUtils.operation(v) isa ModelingToolkit.Differential && push!(out, v)
  end
  return out
end

_hasExpressionDerivative(ex) = any(v -> !_isDifferentiableUnknown(v), _differentials(ex))

#= x => f for the equations `x ~ f` with x a variable: f neither reads x nor
   differentiates, x has no other such equation and the model does not
   differentiate x itself (a state: its D(x) is the state derivative, and
   D(f) would make index reduction solve f for f's operands). =#
function _explicitDefinitions(eqs)
  local out = Dict{Any, Any}()
  local seen = Set{Any}()
  local differentiated = Set{Any}()
  for eq in eqs, d in _differentials((eq.lhs, eq.rhs))
    push!(differentiated, Symbolics.unwrap(SymbolicUtils.arguments(d)[1]))
  end
  for eq in eqs
    local x = Symbolics.unwrap(eq.lhs)
    (SymbolicUtils.iscall(x) && SymbolicUtils.issym(SymbolicUtils.operation(x))) || continue
    if x in seen
      delete!(out, x)
      continue
    end
    push!(seen, x)
    x in differentiated && continue
    local f = Symbolics.unwrap(eq.rhs)
    (!isempty(_differentials(f)) || any(v -> isequal(v, x), Symbolics.get_variables(f))) && continue
    out[x] = f
  end
  return out
end

#= `ex` with each D(x) of an explicitly defined x replaced by the derivative
   of its definition, recursively; a D(x) inside its own chain (`open`),
   deeper than _MAX_DEFINITION_CHAIN, or whose definition Symbolics cannot
   differentiate, stays. The memo ignores `open`: a result a cycle left less
   expanded is still the derivative. =#
const _MAX_DEFINITION_CHAIN = 32

function _derivativesByDefinitions(ex, definitions, memo, open::Set{Any})
  local subs = Dict{Any, Any}()
  for d in _differentials(ex)
    local x = Symbolics.unwrap(SymbolicUtils.arguments(d)[1])
    (haskey(definitions, x) && !(x in open) && length(open) < _MAX_DEFINITION_CHAIN) || continue
    local dx = get(memo, x, nothing)
    if dx === nothing
      dx = try
        local inner = _expandDerivatives(SymbolicUtils.operation(d)(definitions[x]))
        local e = _derivativesByDefinitions(inner, definitions, memo, union(open, Set{Any}([x])))
        _hasExpressionDerivative(e) ? d : e
      catch err
        err isa InterruptException && rethrow()
        d
      end
      memo[x] = dx
    end
    subs[d] = dx
  end
  return isempty(subs) ? ex : Symbolics.substitute(ex, subs)
end

#= A variable x(t), or a derivative of one (D(D(x))): what MTK differentiates. =#
function _isDifferentiableUnknown(v)
  SymbolicUtils.iscall(v) || return SymbolicUtils.issym(v)
  local op = SymbolicUtils.operation(v)
  op isa ModelingToolkit.Differential && return _isDifferentiableUnknown(Symbolics.unwrap(SymbolicUtils.arguments(v)[1]))
  return SymbolicUtils.issym(op)
end

#= build_explicit_observed_function for reads that may hold derivatives (an
   event relation `asc = der(Hstat) > 0`, MSL FluxTubes' Tellinen
   hysteresis), which it takes only as variables of the system: D(x) of a
   differential unknown becomes the right-hand side of its equation, any
   other D(x) the derivative variable index reduction made for x (xˍt, an
   unknown or observed). A D(x) the system has neither for is left (the
   build then fails as before). =#
function _buildObservedFunction(sys, exprs; kwargs...)
  return ModelingToolkit.build_explicit_observed_function(sys, _derivativesAsSystemTerms(sys, exprs); kwargs...)
end

function _derivativesAsSystemTerms(sys, exprs)
  local ds = _differentials(exprs)
  isempty(ds) && return exprs
  local rhsOf = Dict{Any, Any}()
  for eq in ModelingToolkit.equations(sys)
    local l = Symbolics.unwrap(eq.lhs)
    SymbolicUtils.iscall(l) && SymbolicUtils.operation(l) isa ModelingToolkit.Differential && (rhsOf[l] = eq.rhs)
  end
  local byName = Dict{Symbol, Any}()
  for x in ModelingToolkit.unknowns(sys)
    byName[ModelingToolkit.getname(x)] = x
  end
  for eq in ModelingToolkit.observed(sys)
    byName[ModelingToolkit.getname(eq.lhs)] = eq.lhs
  end
  local subs = Dict{Any, Any}()
  for d in ds
    if haskey(rhsOf, d)
      subs[d] = rhsOf[d]
    else
      local term = get(byName, ModelingToolkit.getname(Symbolics.diff2term(d)), nothing)
      term === nothing || (subs[d] = term)
    end
  end
  isempty(subs) && return exprs
  return exprs isa AbstractArray ? Any[Symbolics.substitute(e, subs) for e in exprs] : Symbolics.substitute(exprs, subs)
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
#= Per-call-site index of the candidate (non-differential) equations, keyed by
   the exact variable names each contains. Avoids re-stringifying and
   re-scanning every equation for every resolve: on large reduced systems the
   unindexed scan made initialization quadratic in system size and dominated
   the whole simulate call. =#
struct AliasEqIndex
  exprs::Vector{Any}
  varCounts::Vector{Int}
  byVar::OrderedDict{String, Vector{Int}}
end

function buildAliasEqIndex(eqs)::AliasEqIndex
  local exprs = Any[]
  local varCounts = Int[]
  local byVar = OrderedDict{String, Vector{Int}}()
  for eq in eqs
    local lhsV = Symbolics.value(eq.lhs)
    #= Differential equations are dynamics, not algebraic alias candidates. =#
    if SymbolicUtils.iscall(lhsV) && (SymbolicUtils.operation(lhsV) isa ModelingToolkit.Differential)
      continue
    end
    local expr = eq.lhs - eq.rhs
    local vs = Symbolics.get_variables(expr)
    push!(exprs, expr)
    push!(varCounts, length(vs))
    local i = length(exprs)
    for v in vs
      push!(get!(() -> Int[], byVar, string(v)), i)
    end
  end
  return AliasEqIndex(exprs, varCounts, byVar)
end

function resolveAliasInitialValue(diffState, idx::AliasEqIndex, ivMap::Dict)
  local diffStr = string(diffState)
  if !OMBackend.envSwitch("OMBACKEND_ALIAS_INDEX")
    #= Legacy-faithful scan: every candidate equation whose text mentions the
       state, simplify uncapped. =#
    for i in 1:length(idx.exprs)
      contains(string(idx.exprs[i]), diffStr) || continue
      local resolved = try
        _resolveAliasFromExpr(diffState, idx.exprs[i], 0, ivMap)
      catch e
        e isa Union{DivideError, OverflowError, InexactError} || rethrow()
        nothing
      end
      resolved === nothing || return resolved
    end
    return nothing
  end
  for i in get(idx.byVar, diffStr, Int[])
    #= Rational{Int} arithmetic inside substitute/simplify can overflow the
       denominator to 0 (DivideError) on pathological candidates; skip them. =#
    local resolved = try
      _resolveAliasFromExpr(diffState, idx.exprs[i], idx.varCounts[i], ivMap)
    catch e
      e isa Union{DivideError, OverflowError, InexactError} || rethrow()
      nothing
    end
    resolved === nothing || return resolved
  end
  return nothing
end

#= Whether a symbolic expression has more than `cap` nodes, counted as a tree
   (shared subexpressions count each time); stops at the cap. =#
function _exprLargerThan(@nospecialize(ex), cap::Int)::Bool
  local n = 0
  local stack = Any[Symbolics.unwrap(ex)]
  while !isempty(stack)
    local e = pop!(stack)
    n += 1
    n > cap && return true
    SymbolicUtils.iscall(e) && append!(stack, SymbolicUtils.arguments(e))
  end
  return false
end

function _resolveAliasFromExpr(diffState, @nospecialize(expr), varCount::Int, ivMap::Dict)
  local exprSub = Symbolics.substitute(expr, ivMap)
  #= Fast path: plain substitution. =#
  local intercept = Symbolics.value(Symbolics.substitute(exprSub, Dict(diffState => 0)))
  local sumOnePoint = Symbolics.value(Symbolics.substitute(exprSub, Dict(diffState => 1)))
  if !(intercept isa Number && sumOnePoint isa Number)
    #= Slow path only when free vars remain: simplify folds a constant-condition
       `ifelse` (e.g. a pre-evaluated `ifCond` flag) to its taken branch, so the
       expression collapses to numeric-affine form instead of keeping spurious
       free branch variables that would block resolution. Capped to small
       equations: simplify on a large dynamics expression takes minutes and an
       alias equation never has many variables. Its size is capped too: in a
       system with few unknowns every equation has few variables, but with the
       observed variables substituted (full_equations) an equation can be huge,
       and simplify then runs out of memory (the V6 cylinder rig: 8 unknowns). =#
    varCount <= 8 || return nothing
    _exprLargerThan(exprSub, 200) && return nothing
    local exprS = Symbolics.simplify(exprSub)
    intercept = Symbolics.value(Symbolics.simplify(Symbolics.substitute(exprS, Dict(diffState => 0))))
    sumOnePoint = Symbolics.value(Symbolics.simplify(Symbolics.substitute(exprS, Dict(diffState => 1))))
  end
  if !(intercept isa Number && sumOnePoint isa Number)
    return nothing
  end
  #= Convert to Float64 before arithmetic: pure Rational{Int} subtraction here
     can overflow the denominator product to 0 and throw DivideError. =#
  local interceptF = Float64(intercept)
  local sumOnePointF = Float64(sumOnePoint)
  local slope = sumOnePointF - interceptF
  if !iszero(slope)
    return -interceptF / slope
  end
  return nothing
end

function resolveAliasInitialValue(diffState, fullEqs, ivMap::Dict)
  return resolveAliasInitialValue(diffState, buildAliasEqIndex(fullEqs), ivMap)
end

#= Sidecar from splitInitialValues to the DirectRHS init solve: which initial
   values are user/Modelica constraints (pinnable) as opposed to propagated
   seeds. Keyed by system NAME: the system object is rebound between the
   record and lookup sites (guess merging via @set!), so identity keys miss. =#
const _EXPLICIT_PINNED_INITIAL_VALUE_KEYS = Dict{Symbol, OrderedSet{String}}()

function _pinnedSidecarKey(reducedSystem)::Symbol
  return nameof(reducedSystem)
end

function explicitPinnedInitialValueKeys(reducedSystem, hardInitialValues)::OrderedSet{String}
  return get(() -> OrderedSet(string(p.first) for p in hardInitialValues),
             _EXPLICIT_PINNED_INITIAL_VALUE_KEYS, _pinnedSidecarKey(reducedSystem))
end

#= Names (as strings) of variables that appear in an algebraic (non-differential)
   equation of `sys`, including observed equations: structural_simplify moves
   eliminated algebraic constraints to observed, so a state coupled only through
   them is still algebraically coupled. A differential equation `D(x) ~ rhs` is
   skipped: the state `x`'s initial condition is free unless it is also
   constrained by an algebraic equation. Used by splitInitialValues to decide
   which non-fixed starts are safe to keep as hard u0 (uncoupled states) vs must
   be relaxed to guesses (algebraically coupled states). =#
function _algebraicCoupledVarStrs(sys)::OrderedSet{String}
  local out = OrderedSet{String}()
  local eqs = equations(sys)
  for eq in eqs
    local lhs = Symbolics.value(eq.lhs)
    local isDiff = SymbolicUtils.iscall(lhs) && (SymbolicUtils.operation(lhs) isa ModelingToolkit.Differential)
    isDiff && continue
    for v in Symbolics.get_variables(eq.lhs)
      push!(out, string(v))
    end
    for v in Symbolics.get_variables(eq.rhs)
      push!(out, string(v))
    end
  end
  local obsEqs = observed(sys)
  for eq in obsEqs
    #= An observed lhs is the eliminated variable itself; only the rhs couples. =#
    for v in Symbolics.get_variables(eq.rhs)
      push!(out, string(v))
    end
  end
  return out
end

#= Names (as strings) of the variables initialization equations determine:
   the left-hand side of each, and x for every derivative D(x) in them. The
   arguments of other right-hand sides (e.g. positions passed to a branch
   selection function) are only read. =#
#= The left sides of the initialization equations that fix a value (a literal
   or parameter right side). A signal-valued one (its rhs references a
   time-dependent variable, unknown or observed) determines its lhs through
   the init solve's residual rows; marking the lhs fixed would hold it at a
   stale numeric guess that fights the very equation it encodes. Parameters
   print without the (t) suffix and stay pinnable; occursin also catches array
   elements and derivative forms whose printed form does not END with it. =#
function _fixedTrueLhsStrs(initEqs)::OrderedSet{String}
  local out = OrderedSet{String}()
  for eq in initEqs
    local rhsVars = Symbolics.get_variables(eq.rhs)
    any(occursin("(t)", string(v)) for v in rhsVars) && continue
    push!(out, string(eq.lhs))
  end
  return out
end

#= With `reads`, the variables a right side reads too: an initialization
   equation determines them as much as its left side (`der(v) = x - 1` fixes
   x; pinned at its start, the init solve moved the fixed v instead). =#
function _initializationVarStrs(initEqs; reads::Bool = false)::OrderedSet{String}
  local out = OrderedSet{String}()
  local visit
  visit = function (ex, isLhs0::Bool)
    local isLhs = isLhs0 || reads
    local v = Symbolics.unwrap(ex)
    SymbolicUtils.iscall(v) || (isLhs && push!(out, string(v)); return nothing)
    local op = SymbolicUtils.operation(v)
    if op isa ModelingToolkit.Differential
      #= D(x), or D(expr): an observed variable the backend replaced by its
         definition; every variable in it. =#
      foreach(a -> foreach(x -> push!(out, string(x)), Symbolics.get_variables(a)), SymbolicUtils.arguments(v))
      return nothing
    end
    #= x(t) is a call of x on t: a variable, not an expression. =#
    if SymbolicUtils.issym(op)
      isLhs && push!(out, string(v))
      return nothing
    end
    #= An expression on the left (`x + y = 1`) determines every variable in it. =#
    foreach(a -> visit(a, isLhs), SymbolicUtils.arguments(v))
    return nothing
  end
  for eq in initEqs
    visit(eq.lhs, true)
    visit(eq.rhs, false)
  end
  return out
end

"Seed guesses for reduced unknowns by name; overrides only absent or default-0.0 entries."
function mergeSoftGuesses(reducedSystem, pairs::AbstractVector; force::Bool = false)
  isempty(pairs) && return reducedSystem
  local unkByStr = Dict{String, Any}()
  for u in unknowns(reducedSystem)
    unkByStr[_plainVariableName(u)] = u
  end
  local gs = Dict{Any, Any}(ModelingToolkit.guesses(reducedSystem))
  local changed = false
  for p in pairs
    local nm = _plainVariableName(String(first(p)))
    haskey(unkByStr, nm) || continue
    local cur = nothing
    for (k, v) in gs
      if _plainVariableName(k) == nm
        cur = v
        break
      end
    end
    #= guesses values may be Num-wrapped; unwrap before the default-0.0 test =#
    local curv = cur === nothing ? nothing : Symbolics.value(cur)
    if force || curv === nothing || (curv isa Number && iszero(curv))
      for k in collect(keys(gs))
        replace(string(k), "(t)" => "") == nm && delete!(gs, k)
      end
      gs[unkByStr[nm]] = last(p)
      changed = true
    end
  end
  changed || return reducedSystem
  @set! reducedSystem.guesses = gs
  return reducedSystem
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
function splitInitialValues(reducedSystem,
                            finalInitialValues::Vector{<:Pair{Symbolics.Num}},
                            allInitialValues::Vector{<:Pair},
                            pars::AbstractDict = Dict{Any, Any}())
  return splitInitialValues(reducedSystem,
                            Pair{Any, Any}[p for p in finalInitialValues],
                            Pair{Any, Any}[p for p in allInitialValues],
                            pars)
end

function splitInitialValues(reducedSystem, finalInitialValues::AbstractVector,
                            allInitialValues::AbstractVector = Pair[],
                            pars::AbstractDict = Dict{Any, Any}())
  local massMatrix = ModelingToolkit.calculate_massmatrix(reducedSystem)
  local reducedUnks = unknowns(reducedSystem)
  #= Identity mass matrix means pure ODE: all states are differential =#
  if massMatrix isa LinearAlgebra.UniformScaling
    @debug "[MTK GEN: init] ODEProblem: pure ODE (identity mass matrix), $(length(finalInitialValues)) hard u0, $(length(reducedUnks)) unknowns"
    #= The starts stay u0; as pins (the init solve's fixed values, where it
       runs: _hasSolvedInitializationRows) only the fixed ones, as in a DAE.
       A non-fixed start an initialization equation determines was pinned:
       the init solve then freed every variable and moved the fixed ones
       (`x = 2y + 1` with y fixed at 2: y = 0, OpenModelica 2). =#
    local pinnedKeys = OrderedSet(string(p.first) for p in finalInitialValues)
    local odeInitEqs = ModelingToolkit.initialization_equations(reducedSystem)
    if !isempty(odeInitEqs)
      local fixedLhs = _fixedTrueLhsStrs(odeInitEqs)
      local initCoupled = _initializationVarStrs(odeInitEqs; reads = true)
      filter!(k -> k in fixedLhs || !(k in initCoupled), pinnedKeys)
    end
    _EXPLICIT_PINNED_INITIAL_VALUE_KEYS[_pinnedSidecarKey(reducedSystem)] = pinnedKeys
    return (reducedSystem, finalInitialValues)
  end
  #= DAE system: classify states by mass matrix diagonal =#
  @BACKEND_LOGGING @debug "[splitIV] finalInitialValues keys:" [string(p.first) for p in finalInitialValues]
  @BACKEND_LOGGING @debug "[splitIV] reducedUnks:" [string(u) for u in reducedUnks]
  local diffStateSet = OrderedSet{Any}()
  local diffStateStrSet = OrderedSet{String}()
  for i in 1:min(size(massMatrix, 1), length(reducedUnks))
    if massMatrix[i, i] != 0
      push!(diffStateSet, reducedUnks[i])
      push!(diffStateStrSet, string(reducedUnks[i]))
    end
  end
  #= Use string comparison for hard/soft split because finalInitialValues keys
     are Num-wrapped while diffStateSet contains unwrapped BasicSymbolic values.
     isequal(Num(x), x) can fail depending on Symbolics version. =#
  local hardInitialValues = Pair{Any, Any}[pair for pair in finalInitialValues
                                           if string(pair.first) in diffStateStrSet]
  local softInitialValues = Pair{Any, Any}[pair for pair in finalInitialValues
                                           if !(string(pair.first) in diffStateStrSet)]
  #= When a DAE has only algebraic vars with explicit starts and no differential
     state has one, pin the algebraic starts as hard u0. MTK's DAE initializer
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
    @debug "[MTK GEN: init] No differential states have explicit IVs; pinning $(length(softInitialValues)) algebraic IVs as hard u0 so DAE init can solve for diff states"
    hardInitialValues = softInitialValues
    softInitialValues = Pair{Any, Any}[]
  end
  #= Promote `lhs ~ literal` init_eqs to hard u0 entries when `lhs` is a
     reduced unknown. MTK's init solver treats `initialization_eqs` as
     least-squares residuals, so a single `Inertia_w ~ 10` can be sacrificed
     against many algebraic residuals minimised toward zero. =#
  #= Build the set of variables that have an explicit fixed=true init constraint.
     Modelica spec: only `fixed=true` variables get hard initial conditions; vars
     with explicit `start` but `fixed=false` are guesses for the init solver.
     OMBackend's getFixedStartConstraintsMTK emits one init eq per fixed=true var,
     so reading initialization_equations(reducedSystem) gives us the authoritative
     fixed=true set. Vars whose start landed in finalInitialValues but whose lhs
     is not in this set are non-fixed defaults — relax them to guesses so the
     algebraic constraints can solve consistently with the user's hard pins. =#
  local fixedTrueLhsSet = _fixedTrueLhsStrs(initEqs)
  if hasInitConstraints
    local hardKeyStrSet = OrderedSet(string(p.first) for p in hardInitialValues)
    local reducedUnkStrSet = OrderedSet(string(u) for u in reducedUnks)
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
      @debug "[MTK GEN: init] Promoted $(promoted) initialization_eqs literal constraints to hard u0"
    end
    #= Demote hard u0 entries that came from `fixed=false` defaults.
       Modelica spec: only `fixed=true` vars get hard initial conditions; vars
       with explicit `start` but `fixed=false` are guesses for the init solver.
       Without this demote, e.g. m1.s(start=0) without fixed=true would over-pin
       the DAE system and force algebraic vars (sd1.s_rel) to satisfy the
       residual at the wrong value.

       BUT only demote a non-fixed differential state that is actually COUPLED to an
       algebraic constraint (appears in some non-differential equation). A differential
       state whose IC is genuinely free — it occurs only as `D(x)` in its own equation,
       never in an algebraic constraint — cannot over-pin anything, so its start must
       remain a hard u0; demoting it to a soft guess that the DAE initializer then
       fails to honour drops the user's start (e.g. an event-held discrete's consumer
       state landing at 0 instead of its start). =#
    local _algCoupled = _algebraicCoupledVarStrs(reducedSystem)
    #= Likewise a state an initialization equation determines (its left-hand
       side, or x in D(x)): `der(x) = 0` (steady state, MSL filters) fixes x
       through its differential equation, and pinning x at its start as well
       over-determines the initialization (the init solve then frees every
       variable and moves the fixed=true ones; the MSL EngineV6's crank). =#
    local _initCoupled = _initializationVarStrs(initEqs; reads = true)
    local demoted = filter(p -> !(string(p.first) in fixedTrueLhsSet) &&
                                (string(p.first) in _algCoupled || string(p.first) in _initCoupled),
                           hardInitialValues)
    if !isempty(demoted)
      local _demotedKeys = OrderedSet(string(p.first) for p in demoted)
      hardInitialValues = filter(p -> !(string(p.first) in _demotedKeys), hardInitialValues)
      append!(softInitialValues, demoted)
      @debug "[MTK GEN: init] Demoted $(length(demoted)) non-fixed algebraic-coupled hard u0 entries to guesses (start without fixed=true)"
    end
  end
  #= Pin only user/Modelica constraints (fixed=true starts, promoted literal
     initialization equations, and the deliberate algebraic-start special case).
     Alias-propagated values and MTK derivative defaults added below are seeds,
     not constraints; DirectRHS uses this sidecar to keep them adjustable. =#
  local explicitPinnedKeyStrSet = OrderedSet(string(p.first) for p in hardInitialValues)
  if !isempty(softInitialValues)
    #= Drop pairs whose key unwrapped to a numeric constant. This happens when
       the keyed variable was eliminated by `structural_simplify` so its module
       symbol resolved to a constant Num at start-equation eval time. Feeding
       such a `Num(0.0) => ...` to `guesses` later crashes `AtomicArrayDict`
       conversion because the key cannot be a non-symbolic value. =#
    local sanitisedSoft = filter(softInitialValues) do p
      local k = Symbolics.value(p.first)
      !(k isa Number)
    end
    if length(sanitisedSoft) != length(softInitialValues)
      @debug "[MTK GEN: init] dropped $(length(softInitialValues) - length(sanitisedSoft)) soft IVs whose keys folded to numeric constants (eliminated by structural_simplify)"
    end
    if !isempty(sanitisedSoft)
      local currentGuesses = ModelingToolkit.guesses(reducedSystem)
      local newGuesses = merge(currentGuesses, Dict(sanitisedSoft))
      @set! reducedSystem.guesses = newGuesses
    end
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
  local hardSymStrSet = OrderedSet(string(iv.first) for iv in hardInitialValues)
  #= Only feed user-explicit start values (hard + soft, both filtered through
     `skipDefaultsForStates`) to the alias resolver. `allInitialValues` carries
     default 0.0 entries for non-explicit vars; substituting those lets a
     trivial-zero alias win over a non-zero user-explicit one. =#
  local explicitIVMap = Dict{Any, Any}(iv.first => iv.second
                                       for iv in vcat(hardInitialValues, softInitialValues))
  local fullEqs = ModelingToolkit.full_equations(reducedSystem)
  local aliasIdx = buildAliasEqIndex(fullEqs)
  #= Propagate hard u0 entries through algebraic alias chains in the reduced
     system. Iterates to a fixed point so multi-step chains resolve, and
     covers MTK-generated derivative-suffix vars regardless of mass-matrix
     classification (Engine1a's `Inertia_phiˍt` is the differential state but
     `Inertia_w` is its algebraic alias; promoting `Inertia_w ~ 10` to hard u0
     lets us propagate to `Inertia_phiˍt` via the alias). =#
  # resolver consults hard u0 only; soft default-zero entries would let
  # the wrong alias chain match first
  local progressed = true
  while progressed
    progressed = false
    explicitIVMap = Dict{Any, Any}(iv.first => iv.second for iv in hardInitialValues)
    for unk in reducedUnks
      local unkStr = string(unk)
      unkStr in hardSymStrSet && continue
      contains(unkStr, "ˍ") || continue
      local ivMapForResolve = filter(p -> string(p.first) != unkStr, explicitIVMap)
      local resolved = resolveAliasInitialValue(unk, aliasIdx, ivMapForResolve)
      resolved === nothing && continue
      push!(hardInitialValues, unk => Float64(resolved))
      push!(hardSymStrSet, unkStr)
      progressed = true
      @debug "[MTK GEN: init] Resolved $(unk) to $(resolved) via equation alias"
    end
  end
  local _defaulted0Starts = String[]
  #= A start demoted to a guess above is an explicit start: the 0.0 default
     below must not replace it (a capacitor's vc(start = 10) went to 0). =#
  local softSymStrSet = OrderedSet(string(p.first) for p in softInitialValues)
  for diffState in diffStateSet
    if !(string(diffState) in hardSymStrSet) && !(string(diffState) in softSymStrSet)
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
        local resolved = resolveAliasInitialValue(diffState, aliasIdx, ivMapForResolve)
        local finalVal = something(resolved, 0.0)
        push!(hardInitialValues, diffState => finalVal)
        if resolved !== nothing
          @debug "[MTK GEN: init] Resolved differential state $(diffState) to $(finalVal) via equation alias"
        else
          @debug "[MTK GEN: init] Defaulted MTK derivative $(diffState) to 0.0 (no alias equation found)"
        end
      else
        #= Provide 0.0 as a guess (not hard) for Modelica variables without explicit start.
           This gives MTK an initial iterate. If algebraic constraints determine the
           value, MTK can override the guess during initialization. =#
        push!(softInitialValues, diffState => 0.0)
        push!(_defaulted0Starts, diffStateStr)
        local currentGuesses2 = ModelingToolkit.guesses(reducedSystem)
        local newGuesses2 = merge(currentGuesses2, Dict(diffState => 0.0))
        @set! reducedSystem.guesses = newGuesses2
        @debug "[MTK GEN: init] Providing default 0.0 guess for differential state $(diffState) (no explicit start value)"
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
  local hardSymStrSetFinal = OrderedSet(string(iv.first) for iv in hardInitialValues)
  local currentGuessKeysFinal = OrderedSet(string(k) for k in keys(ModelingToolkit.guesses(reducedSystem)))
  local missingUnks = filter(reducedUnks) do unk
    local s = string(unk)
    !(s in hardSymStrSetFinal) && !(s in currentGuessKeysFinal)
  end
  if !isempty(missingUnks)
    local fallbackGuesses = Dict{Any, Any}(unk => 0.0 for unk in missingUnks)
    local currentGuesses3 = ModelingToolkit.guesses(reducedSystem)
    @set! reducedSystem.guesses = merge(currentGuesses3, fallbackGuesses)
    append!(_defaulted0Starts, [string(u) for u in missingUnks])
    @debug "[MTK GEN: init] Providing default 0.0 guesses for $(length(missingUnks)) uncovered unknowns: $(join([string(u) for u in missingUnks], ", "))"
  end
  local totalGuesses = length(softInitialValues) + length(missingUnks)
  @debug "[MTK GEN: init] ODEProblem: DAE, $(length(hardInitialValues)) hard u0 (differential), $(totalGuesses) as guesses (algebraic), $(length(reducedUnks)) unknowns ($(length(diffStateSet)) differential)"
  if OMBackend.WARN_MISSING_START_VALUES[] && !isempty(_defaulted0Starts)
    local _u = sort!(unique(_defaulted0Starts))
    local _shown = _u[1:min(end, 30)]
    @warn "[MTK GEN: init] Defaulted $(length(_u)) unknown(s) to a 0.0 start/guess (no explicit start). A wrong 0.0 here can make the DAE-init residual NaN: " *
          join(_shown, ", ") * (length(_u) > 30 ? ", ... (+$(length(_u) - 30) more)" : "")
  end
  #= Propagate non-zero initial guesses. A unknown that defaulted to a 0.0 guess
     (no explicit start) whose true init value is set by the equations (e.g. a
     pump speed driven by a source's offset) makes the DAE-init residual NaN via
     downstream divisions. Seed with parameters + t0 + hard u0, then resolve the
     0.0-guessed unknowns from their defining equations in fixed-point order; a
     target resolves only when every other variable in some equation is already
     known, so the chain (ramp output -> speed -> flow) resolves correctly and
     coupled/nonlinear equations are skipped until their operands are known.
     mergeSoftGuesses overrides only 0.0 guesses, so non-zero results replace the
     bad defaults while genuine 0.0 starts are untouched. =#
  if !isempty(pars)
    local _eqsR = vcat(ModelingToolkit.equations(reducedSystem), ModelingToolkit.observed(reducedSystem))
    local _idxR = buildAliasEqIndex(_eqsR)
    local _seed = Dict{Any, Any}(ModelingToolkit.get_iv(reducedSystem) => 0.0)
    for (k, v) in pars; _seed[k] = v; end
    for p in hardInitialValues; _seed[p.first] = p.second; end
    local _targets = Any[k for (k, v) in ModelingToolkit.guesses(reducedSystem)
                         if (Symbolics.value(v) isa Number && iszero(Symbolics.value(v)))]
    local _resolved = Dict{Any, Float64}()
    for _round in 1:6
      local _progress = false
      for X in _targets
        haskey(_resolved, X) && continue
        local _ivm = Dict{Any, Any}(_seed)
        for (k, v) in _resolved; _ivm[k] = v; end
        delete!(_ivm, X)
        local _rv = resolveAliasInitialValue(X, _idxR, _ivm)
        if _rv !== nothing && isfinite(_rv)
          _resolved[X] = _rv; _progress = true
        end
      end
      _progress || break
    end
    if !isempty(_resolved)
      reducedSystem = mergeSoftGuesses(reducedSystem, Pair{Any, Any}[k => v for (k, v) in _resolved])
      @info "[MTK GEN: init] PROPAGATED $(length(_resolved)) init guesses: " *
            join([string(replace(string(k), "(t)" => ""), "=", round(v; digits=4)) for (k, v) in _resolved], ", ")
    else
      @info "[MTK GEN: init] propagation resolved nothing (targets=$(length(_targets)), pars=$(length(pars)))"
    end
  end
  _EXPLICIT_PINNED_INITIAL_VALUE_KEYS[_pinnedSidecarKey(reducedSystem)] = explicitPinnedKeyStrSet
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
  local hardKeys = OrderedSet(string(iv.first) for iv in finalInitialValues)
  local allIVMap = Dict{String, Any}(string(iv.first) => iv.second for iv in allInitialValues)
  local guessDict = Dict{Any, Any}()
  for unk in reducedUnks
    local unkStr = string(unk)
    if !(unkStr in hardKeys)
      guessDict[unk] = get(allIVMap, unkStr, 0.0)
    end
  end
  if !isempty(guessDict)
    @debug "[MTK GEN: init] buildDefaultGuesses: $(length(guessDict)) guesses for unknowns not in u0"
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
  local seenLHS = OrderedSet{String}()
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
  @debug "[MTK GEN: observed] injectObservedEquations: existing=$(length(existingObs)), new=$(length(newEqs))"
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
  if updated !== nothing && hasfield(typeof(updated), :index_cache) &&
     ModelingToolkit.get_index_cache(updated) !== nothing
    try
      local _icFresh = ModelingToolkit.IndexCache(updated)
      updated = Setfield.set(updated, Setfield.PropertyLens{:index_cache}(), _icFresh)
    catch _err
      #= The index_cache rebuild is skipped. =#
      OMBackend._fallback(_err, :observedIndexCache)
    end
  end
  return updated
end

#= StateSelect.prefer / avoid lower to soft (|priority| <= 2) VariableStatePriority
   hints. They are advisory; StateSelect.always / never (|priority| = 10) are hard.
   When a soft hint forces MTK index reduction into a structurally singular reduced
   system, neutralizing it to the default (0) lets MTK pick a feasible state set.
   Returns the count of neutralized unknowns and a freshly-constructed System that
   carries the neutralized unknowns. The state priority that MTK honours is read
   from the unknowns vector passed to the constructor, so the system must be rebuilt
   through `System(...)` (mutating the field via `@set`/`substitute` is a no-op:
   `setmetadata` leaves variables `isequal`, so MTK keeps the original priority). =#
function neutralizeSoftStatePriority(sys::ModelingToolkit.AbstractSystem)
  local n = 0
  local newUnknowns = map(ModelingToolkit.get_unknowns(sys)) do u
    local uv = Symbolics.unwrap(u)
    local p = SymbolicUtils.getmetadata(uv, ModelingToolkit.VariableStatePriority, nothing)
    if p !== nothing && abs(p) <= 2
      n += 1
      SymbolicUtils.setmetadata(uv, ModelingToolkit.VariableStatePriority, 0)
    else
      uv
    end
  end
  n == 0 && return (0, sys)
  local relaxed = ModelingToolkit.System(
    equations(sys),
    ModelingToolkit.get_iv(sys),
    newUnknowns,
    ModelingToolkit.get_ps(sys);
    name = nameof(sys),
    continuous_events = ModelingToolkit.get_continuous_events(sys),
    discrete_events = ModelingToolkit.get_discrete_events(sys),
    guesses = ModelingToolkit.get_guesses(sys),
    initialization_eqs = ModelingToolkit.get_initialization_eqs(sys),
  )
  return (n, relaxed)
end

#= simplify=true polynomial normalization rationalizes Float64 coefficients;
   exact folding can overflow into Rational{BigInt}, and a single such literal
   promotes every numeric consumer (RHS, Jacobian, observed, event functions)
   to allocating BigFloat arithmetic. Demote back to Float64 once the symbolic
   algebra is done. Only equations that actually carry a wide literal are
   rebuilt, so shape/metadata reconstruction risk stays confined to offenders. =#
#= Exact BigInt integer literals stay untouched: RNG state constants are
   bit-exact and exceed Float64's 2^53 integer range. =#
const _WideNumeric = Union{Rational{BigInt}, BigFloat}

function _containsWideNumeric(ex)::Bool
  ex isa _WideNumeric && return true
  if ex isa SymbolicUtils.BasicSymbolic && SymbolicUtils.iscall(ex)
    return any(_containsWideNumeric, SymbolicUtils.arguments(ex))
  end
  return false
end

function _demoteWideNumerics(ex)
  ex isa _WideNumeric && return Float64(ex)
  if ex isa SymbolicUtils.BasicSymbolic && SymbolicUtils.iscall(ex)
    local args = SymbolicUtils.arguments(ex)
    local newArgs = Any[_demoteWideNumerics(a) for a in args]
    if newArgs != args
      return SymbolicUtils.maketerm(typeof(ex), SymbolicUtils.operation(ex),
                                    newArgs, SymbolicUtils.metadata(ex))
    end
  end
  return ex
end

function _demoteWideNumericsInEquations(eqs)
  local changed = 0
  local newEqs = map(eqs) do eq
    local l = Symbolics.unwrap(eq.lhs)
    local r = Symbolics.unwrap(eq.rhs)
    if _containsWideNumeric(l) || _containsWideNumeric(r)
      changed += 1
      Symbolics.Equation(_demoteWideNumerics(l), _demoteWideNumerics(r))
    else
      eq
    end
  end
  return (newEqs, changed)
end

#= Mark time as possibly zero (MTK's maybe_zeros): tearing then solves no
   variable through a coefficient with time or sin(time) as a factor.
   DOCCMinimal's M9 (v = (cos(time), sin(time)), v*conj(i) = 1): sin(t)*i_re =
   cos(t)*i_im was solved as i_re = cos(t)*i_im/sin(t), 0/0 at t = 0. =#
function _timeMayBeZero(sys::ModelingToolkit.AbstractSystem)
  local iv = ModelingToolkit.get_iv(sys)
  iv === nothing && return sys
  local mayBeZero = copy(ModelingToolkit.get_maybe_zeros(sys))
  push!(mayBeZero, Symbolics.unwrap(iv))
  return @set sys.maybe_zeros = mayBeZero
end

"""
  The irreducible variables scheme does not work using plain simplify.

  It should be noted that for some models both running tearing and structural simplification are needed.
  Report an issue for the MTK reporters giving an example of this behavior.

  One example is running tearing twice broke the system
"""
function structural_simplify(sys::ModelingToolkit.AbstractSystem,
                             io = nothing;
                             simplify = false,
                             allow_parameter = true,
                             kwargs...)
  local pre_eqs = length(equations(sys))
  local pre_unknowns = length(unknowns(sys))
  @info "[MTK GEN: simplify] Before structural_simplify: $pre_eqs equations, $pre_unknowns unknowns"
  dumpMTKPreSimplify(sys, pre_eqs, pre_unknowns)
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
      local allResults = Tuple{Int,String,Symbol,Any}[]
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
        @info "[MTK GEN: simplify] No array-shaped subtrees found in equations (clean for Pantelides)"
      end
    end
  end

  local useSplit = get(kwargs, :split, true)
  sys = _timeMayBeZero(sys)
  local _preSimplifySys = sys
  if OMBackend.BACKEND_LOGGING[]
    local _ss_timed = @timed ModelingToolkit.structural_simplify(sys; simplify = simplify, split = useSplit)
    sys = _ss_timed.value
    @debug "[MTK GEN: simplify] structural_simplify took $(_ss_timed.time)s, $(round(_ss_timed.bytes / 1e9, digits=2)) GiB"
  else
    sys = ModelingToolkit.structural_simplify(sys; simplify = simplify, split = useSplit)
  end
  #= A structurally singular reduction (equations != unknowns) can be caused by a
     soft StateSelect hint conflicting with a constraint MTK index-reduces. Retry
     once with soft state-priority hints neutralized before accepting the result. =#
  if length(equations(sys)) != length(unknowns(sys))
    local (nNeutralized, relaxedSys) = neutralizeSoftStatePriority(_preSimplifySys)
    if nNeutralized > 0
      local retried = try
        ModelingToolkit.structural_simplify(relaxedSys; simplify = simplify, split = useSplit)
      catch e
        @warn "[MTK GEN: simplify] retry without soft state-priority hints threw" exception = (e, catch_backtrace())
        nothing
      end
      if retried !== nothing && length(equations(retried)) == length(unknowns(retried))
        @info "[MTK GEN: simplify] recovered structural balance by neutralizing $(nNeutralized) soft state-priority hint(s)"
        sys = retried
      end
    end
  end
  try
    local (newEqs, nEq) = _demoteWideNumericsInEquations(equations(sys))
    local (newObs, nObs) = _demoteWideNumericsInEquations(ModelingToolkit.observed(sys))
    if nEq > 0
      @set! sys.eqs = newEqs
    end
    if nObs > 0
      @set! sys.observed = newObs
    end
    (nEq + nObs) > 0 &&
      @info "[MTK GEN: simplify] demoted wide numeric literals (Rational{BigInt}/BigFloat) in $(nEq) equation(s), $(nObs) observed equation(s)"
  catch ex
    @warn "[MTK GEN: simplify] wide-numeric demotion failed; continuing with original system" exception = (ex, catch_backtrace())
  end
  local post_eqs = length(equations(sys))
  #= Diagnostic only: full_equations expands observed eqs and can hit
     SymbolicUtils Rational{Int64} overflow on some systems. A log count must
     never abort the build, so fall back to -1 (rendered as "n/a") on failure. =#
  local post_full_eqs = OMBackend._tryOr(() -> length(ModelingToolkit.full_equations(sys)), -1, :fullEquationsCount)
  local post_unknowns = length(unknowns(sys))
  @info "[MTK GEN: simplify] After structural_simplify: equations=$(post_eqs), full_equations=$(post_full_eqs), unknowns=$(post_unknowns)"
  if post_eqs != post_unknowns
    @warn "equations(sys) != unknowns(sys): $post_eqs vs $post_unknowns"
    for (i, eq) in enumerate(equations(sys))
      @debug "[MTK GEN: simplify] eq[$i]: $eq"
    end
    for (i, u) in enumerate(unknowns(sys))
      @debug "[MTK GEN: simplify] unk[$i]: $u"
    end
  end
  if post_full_eqs >= 0 && post_full_eqs != post_unknowns
    @warn "full_equations(sys) != unknowns(sys): $post_full_eqs vs $post_unknowns"
  end
  dumpMTKPostSimplify(sys)
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
  local reducedUnknowns = OrderedSet(unknowns(sys))
  local currentGuesses = ModelingToolkit.guesses(sys)
  local filteredGuesses = Dict(k => v for (k, v) in currentGuesses if k in reducedUnknowns)
  if length(filteredGuesses) != length(currentGuesses)
    @debug "[MTK GEN: init] Filtered guesses: $(length(currentGuesses)) -> $(length(filteredGuesses)) (removed $(length(currentGuesses) - length(filteredGuesses)) non-unknown guesses)"
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
  diff_vars = Any[]
  alge_eqs = Equation[]
  for (i, eq) in enumerate(eqs)
    if !isdiffeq(eq)
      push!(alge_eqs, eq)
    else
      var, maxorder = Symbolics.var_from_nested_derivative(eq.lhs)
      maxorder > get(var_order, var, 1) && (var_order[var] = maxorder)
      var′ = Symbolics.lower_varname(var, iv, maxorder - 1)
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
      lvar = Symbolics.lower_varname(var, iv, o - 1)
      rvar = Symbolics.lower_varname(var, iv, o)
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
  vars = OrderedSet()
  subs = Dict()

  for (i, eq) in enumerate(eqs)
    vars!(vars, eq)
    n_diffvars = 0
    for vv in vars
      isdifferential(vv) || continue
      var, maxorder = Symbolics.var_from_nested_derivative(vv)
      isparameter(var) && continue
      n_diffvars += 1
      order = get(var_order, var, nothing)
      seen = order !== nothing
      if !seen
        order = 1
      end
      maxorder > order && (var_order[var] = maxorder)
      var′ = Symbolics.lower_varname(var, iv, maxorder - 1)
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
      lvar = Symbolics.lower_varname(var, iv, o - 1)
      rvar = Symbolics.lower_varname(var, iv, o)
      push!(diff_vars, lvar)

      rhs = rvar
      eq = Differential(iv)(lvar) ~ rhs
      push!(diff_eqs, eq)
    end
  end

  return ([diff_eqs; substitute.(eqs, (subs,))],
          vcat(collect(diff_vars), setdiff(unknown_vars, diff_vars)))
end

function getStatesAsSymbols(odeFunc::ODEFunction)
  odeFunc.sys === nothing && return Symbol[]
  local states = ModelingToolkit.get_unknowns(odeFunc.sys)
  map(x->x.f.name, states)
end

function getStatesAsSymbols(daeFunc::ModelingToolkit.SciMLBase.DAEFunction)
  daeFunc.sys === nothing && return Symbol[]
  local states = ModelingToolkit.get_unknowns(daeFunc.sys)
  map(x->x.f.name, states)
end

#= The names the legacy callbacks index the state vector by
   (`lookuptableStates[Symbol("name")]`, or `[:name]` for a pre() read) in
   generated code `ex`. =#
function namedStateLookups(ex)::Vector{String}
  local names = OrderedSet{String}()
  local walk
  walk = function (e)
    e isa Expr || return nothing
    local key = if e.head == :ref && length(e.args) == 2 && e.args[1] === :lookuptableStates
      e.args[2]
    elseif e.head == :call && length(e.args) == 3 && e.args[1] in (:getindex, getindex) && e.args[2] === :lookuptableStates
      e.args[3]
    else
      nothing
    end
    if key isa Expr && key.head == :call && length(key.args) == 2 && key.args[1] === :Symbol && key.args[2] isa String
      push!(names, key.args[2])
    elseif key isa QuoteNode && key.value isa Symbol
      push!(names, string(key.value))
    end
    foreach(walk, e.args)
    return nothing
  end
  walk(ex)
  return collect(names)
end

"""
    checkNamedStateLookups(problem, names)

Warn when a variable the callbacks read from or write to the state vector by
name is not an unknown of the simplified system: its lookup would fail when
the callback runs.
"""
function checkNamedStateLookups(problem, names::Vector{String})
  isempty(names) && return nothing
  local f = problem.f
  (hasproperty(f, :sys) && f.sys !== nothing) || return nothing
  local have = Set{String}(string(s) for s in getStatesAsSymbols(f))
  local missingNames = filter(n -> !(n in have), names)
  isempty(missingNames) ||
    @warn "[events] callbacks index the state vector by these names, but they are not unknowns of the simplified system" missingNames
  return nothing
end

function getParametersAsSymbols(odeFunc::ODEFunction)
  odeFunc.sys === nothing && return Symbol[]
  local params = ModelingToolkit.parameters(odeFunc.sys)
  map(params) do x
    local uw = SymbolicUtils.unwrap(x)
    #= Regular parameters are Sym (have .name), time-dependent discrete
       parameters like ifCond(t) are Term (have .f.name). =#
    hasproperty(uw, :name) ? uw.name : uw.f.name
  end
end

function getParametersAsSymbols(daeFunc::ModelingToolkit.SciMLBase.DAEFunction)
  daeFunc.sys === nothing && return Symbol[]
  local params = ModelingToolkit.parameters(daeFunc.sys)
  map(params) do x
    local uw = SymbolicUtils.unwrap(x)
    hasproperty(uw, :name) ? uw.name : uw.f.name
  end
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
  # Explicitly clear initialization_data — MTK's ODE init metadata triggers
  # CheckInit on the DAE residual form and rejects perfectly fine guesses.
  local daefunc = ModelingToolkit.SciMLBase.DAEFunction(residual!; sys = prob.f.sys, initialization_data = nothing)
  # Only forward the callback from the source ODEProblem; MTK's initialization_data
  # describes ODE-mass-matrix init and conflicts with DAE residual form (gives
  # spurious CheckInit failures with residual norm > tol).
  local _cb = get(prob.kwargs, :callback, nothing)
  if _cb === nothing
    ModelingToolkit.SciMLBase.DAEProblem(daefunc, du0, prob.u0, prob.tspan, prob.p;
                                         differential_vars = diff_vars)
  else
    ModelingToolkit.SciMLBase.DAEProblem(daefunc, du0, prob.u0, prob.tspan, prob.p;
                                         differential_vars = diff_vars,
                                         callback = _cb)
  end
end
