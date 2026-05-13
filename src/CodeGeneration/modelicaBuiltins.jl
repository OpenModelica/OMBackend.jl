#=
  Modelica built-in function implementations for Julia code generation.

  This file provides Julia implementations for all Modelica built-in functions
  defined in NFModelicaBuiltin.mo. Functions are organized by category.

  Functions that map 1:1 to Julia builtins are re-exported for completeness
  so the mapping table is the single source of truth.

  Reference: Modelica Specification 3.5, Section 3.7 (Built-in Functions)
  Source: OMFrontend.jl/lib/NFModelicaBuiltin.mo
=#

import LinearAlgebra
import Symbolics

#= ============================================================================
   Mapping table: Modelica function name -> Julia function
   Used by the code generator to resolve Modelica built-in function calls.
   ============================================================================ =#

"""
    MODELICA_BUILTIN_FUNCTIONS

Maps Modelica built-in function names (as they appear in DAE.CALL paths)
to qualified Julia function references in OMBackend.CodeGeneration scope.

The code generator looks up function names here before falling through
to the default (emit function name as-is) behavior.
"""
const MODELICA_BUILTIN_FUNCTIONS = Dict{String, Symbol}(
  #= --- Mathematical (scalar) --- =#
  "abs"       => :modelica_abs,
  "sign"      => :modelica_sign,
  "sqrt"      => :modelica_sqrt,
  "ceil"      => :modelica_ceil,
  "floor"     => :modelica_floor,
  "integer"   => :modelica_integer,
  "Integer"   => :modelica_Integer,
  "div"       => :modelica_div,
  "mod"       => :modelica_mod,
  "rem"       => :modelica_rem,

  #= --- String formatting --- =#
  "String"    => :modelica_String,

  #= --- Trigonometric --- =#
  "sin"       => :modelica_sin,
  "cos"       => :modelica_cos,
  "tan"       => :modelica_tan,
  "asin"      => :modelica_asin,
  "acos"      => :modelica_acos,
  "atan"      => :modelica_atan,
  "atan2"     => :modelica_atan2,
  "sinh"      => :modelica_sinh,
  "cosh"      => :modelica_cosh,
  "tanh"      => :modelica_tanh,

  #= --- Exponential and logarithmic --- =#
  "exp"       => :modelica_exp,
  "log"       => :modelica_log,
  "log10"     => :modelica_log10,

  #= --- Array/matrix constructors --- =#
  "identity"  => :modelica_identity,
  "diagonal"  => :modelica_diagonal,
  "zeros"     => :modelica_zeros,
  "ones"      => :modelica_ones,
  "fill"      => :modelica_fill,
  "linspace"  => :modelica_linspace,
  "array"     => :modelica_array,
  "cat"       => :modelica_cat,

  #= --- Array/matrix operations --- =#
  "transpose"    => :modelica_transpose,
  "outerProduct" => :modelica_outerProduct,
  "cross"        => :modelica_cross,
  "skew"         => :modelica_skew,
  "symmetric"    => :modelica_symmetric,
  "promote"      => :modelica_promote,

  #= --- Array reductions --- =#
  "sum"       => :modelica_sum,
  "product"   => :modelica_product,
  "min"       => :modelica_min,
  "max"       => :modelica_max,

  #= --- Array shape/query --- =#
  "size"      => :modelica_size,
  "ndims"     => :modelica_ndims,
  "scalar"    => :modelica_scalar,
  "vector"    => :modelica_vector,
  "matrix"    => :modelica_matrix,

  #= --- Event and state --- =#
  "smooth"       => :modelica_smooth,
  "noEvent"      => :modelica_noEvent,
  "sample"       => :modelica_sample,
  "edge"         => :modelica_edge,
  "change"       => :modelica_change,
  "homotopy"     => :modelica_homotopy,
  "semiLinear"   => :modelica_semiLinear,
)

"""
    MODELICA_UTILITIES_TO_RUNTIME_C

Maps the dot-flattened (`_`-joined) name of qualified Modelica.Utilities.*
function paths to the corresponding flat C-level function names exported by
OMRuntimeExternalC. Keys use the underscore form because `string(path)` on
an `Absyn.QUALIFIED` already collapses the segments with `_`. Used by
`_modelicaFunctionCallExpr` to route qualified Modelica.Utilities.Strings /
Files / Streams calls to the OMRuntimeExternalC API. Without this mapping
the per-model module raises UndefVarError for the dot-flattened Modelica
name (e.g. `Modelica_Utilities_Strings_substring`) because no such Symbol
is bound in the generated module.

Only entries whose OMRuntimeExternalC counterpart actually exists are listed.
Add new entries here as additional Modelica.Utilities calls show up in
real models.
"""
const MODELICA_UTILITIES_TO_RUNTIME_C = Dict{String, Symbol}(
  "Modelica_Utilities_Strings_length"          => :ModelicaStrings_length,
  "Modelica_Utilities_Strings_substring"       => :ModelicaStrings_substring,
  "Modelica_Utilities_Strings_compare"         => :ModelicaStrings_compare,
  "Modelica_Utilities_Strings_skipWhiteSpace"  => :ModelicaStrings_skipWhiteSpace,
  "Modelica_Utilities_Strings_scanIdentifier"  => :ModelicaStrings_scanIdentifier,
  "Modelica_Utilities_Strings_scanInteger"     => :ModelicaStrings_scanInteger,
  "Modelica_Utilities_Strings_scanReal"        => :ModelicaStrings_scanReal,
  "Modelica_Utilities_Strings_scanString"      => :ModelicaStrings_scanString,
  "Modelica_Utilities_Strings_hashString"      => :ModelicaStrings_hashString,
)


#= ============================================================================
   Mathematical functions (scalar)
   ============================================================================ =#

modelica_abs(x) = abs(x)
modelica_sign(x) = sign(x)
modelica_sqrt(x) = sqrt(x)
modelica_ceil(x) = ceil(x)
modelica_floor(x) = floor(x)
modelica_integer(x::Symbolics.Num) = floor(x)
modelica_integer(x::Number) = floor(Int, x)
modelica_integer(x) = floor(x)

#= Modelica `Integer(enum_value)` returns the 1-based index of an enum literal.
   Our codegen lowers enum CREFs to integer indices and ENUM_LITERAL to its
   `index` field, so this is the identity at the Julia level. =#
modelica_Integer(x) = x
modelica_div(x, y) = div(x, y)
modelica_div(x::Symbolics.Num, y) = floor(x / y)
modelica_div(x, y::Symbolics.Num) = floor(x / y)
modelica_div(x::Symbolics.Num, y::Symbolics.Num) = floor(x / y)
modelica_mod(x, y) = mod(x, y)
modelica_rem(x, y) = rem(x, y)


#= ============================================================================
   Trigonometric functions
   ============================================================================ =#

modelica_sin(x)  = sin(x)
modelica_cos(x)  = cos(x)
modelica_tan(x)  = tan(x)
modelica_asin(x) = asin(x)
modelica_acos(x) = acos(x)
modelica_atan(x) = atan(x)
modelica_atan2(y, x) = atan(y, x)
modelica_sinh(x) = sinh(x)
modelica_cosh(x) = cosh(x)
modelica_tanh(x) = tanh(x)


#= ============================================================================
   Exponential and logarithmic functions
   ============================================================================ =#

modelica_exp(x)   = exp(x)
modelica_log(x)   = log(x)
modelica_log10(x) = log10(x)


#= ============================================================================
   Array/matrix constructors
   Modelica semantics differ from Julia for several of these.
   ============================================================================ =#

"""
    modelica_identity(n)

Modelica: identity(n) returns an n x n identity matrix (Integer).
Julia's identity(x) returns x, so we need a custom implementation.
"""
modelica_identity(n) = Float64[i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

"""
    modelica_diagonal(v)

Modelica: diagonal(v) returns a square matrix with v on the diagonal.
Julia: diagm(v) or diagm(0 => v).
"""
modelica_diagonal(v) = LinearAlgebra.diagm(0 => collect(v))

"""
    modelica_zeros(dims...)

Modelica: zeros(n1, n2, ...) returns an array of zeros.
Julia: zeros(dims...) is identical.
"""
modelica_zeros(dims...) = zeros(Float64, dims...)

"""
    modelica_ones(dims...)

Modelica: ones(n1, n2, ...) returns an array of ones.
Julia: ones(dims...) is identical.
"""
modelica_ones(dims...) = ones(Float64, dims...)

"""
    modelica_fill(s, dims...)

Modelica: fill(s, n1, n2, ...) returns an array filled with value s.
Julia: fill(s, dims...) is identical.
"""
modelica_fill(s, dims...) = fill(s, dims...)

"""
    modelica_linspace(x1, x2, n)

Modelica: linspace(x1, x2, n) returns n equally spaced points from x1 to x2.
Julia: range or LinRange.
"""
modelica_linspace(x1, x2, n) = collect(range(x1, x2, length=n))

"""
    modelica_array(args...)

Modelica: array(a, b, c) constructs an array {a, b, c}.
Julia: collect or vcat.
"""
modelica_array(args...) = collect(args)

"""
    modelica_cat(dim, arrays...)

Modelica: cat(k, A, B, ...) concatenates arrays along dimension k.
Julia: cat(dims=k, A, B, ...).
"""
modelica_cat(dim, arrays...) = cat(arrays...; dims=dim)


#= ============================================================================
   Array/matrix operations
   ============================================================================ =#

"""
    modelica_transpose(A)

Modelica: transpose(A) transposes a matrix.
Julia: transpose(A) or permutedims(A).
Note: Julia's transpose is recursive (conjugate for complex). For Modelica
semantics (pure structural transpose) we use permutedims for matrices.
"""
function modelica_transpose(A)
  if A isa AbstractMatrix
    return permutedims(A)
  elseif A isa AbstractVector
    return permutedims(A)
  else
    return Base.transpose(A)
  end
end

"""
    modelica_outerProduct(v1, v2)

Modelica: outerProduct(v1, v2) = v1 * transpose(v2), producing a matrix.
"""
modelica_outerProduct(v1, v2) = collect(v1) * permutedims(collect(v2))

"""
    modelica_cross(x, y)

Modelica: cross(x, y) computes the cross product of two 3-vectors.
Julia: LinearAlgebra.cross(x, y).
"""
modelica_cross(x, y) = LinearAlgebra.cross(collect(x), collect(y))

"""
    modelica_skew(x)

Modelica: skew(x) returns the skew-symmetric matrix associated with vector x.
  skew(x) = [0, -x[3], x[2]; x[3], 0, -x[1]; -x[2], x[1], 0]
"""
function modelica_skew(x)
  return Float64[
     0.0   -x[3]  x[2]
     x[3]   0.0  -x[1]
    -x[2]   x[1]  0.0
  ]
end

"""
    modelica_symmetric(A)

Modelica: symmetric(A) returns (A + A') / 2, making a symmetric matrix.
"""
modelica_symmetric(A) = (A .+ permutedims(A)) ./ 2

"""
    modelica_promote(A, n)

Modelica: promote(A, n) adds dimensions of size 1 from the right
until ndims(A) == n. For example, promote(vector, 2) turns a
1D vector of length N into an Nx1 matrix (column matrix).
See Modelica Spec 3.4, Section 10.3.3.
NOT the same as Julia's promote() which does type promotion.
"""
function modelica_promote(A, n::Integer)
  currentDims = ndims(A)
  if currentDims >= n
    return A
  end
  newShape = (size(A)..., ntuple(_ -> 1, n - currentDims)...)
  return reshape(A, newShape)
end


#= ============================================================================
   Array reductions
   ============================================================================ =#

modelica_sum(a) = sum(a)
modelica_product(a) = prod(a)

#= min/max: Modelica overloads these for both scalars and arrays =#
modelica_min(a) = minimum(a)
modelica_min(a, b) = min(a, b)
modelica_max(a) = maximum(a)
modelica_max(a, b) = max(a, b)


#= ============================================================================
   Array shape/query functions
   ============================================================================ =#

modelica_size(a) = size(a)
modelica_size(a, dim) = size(a, dim)
modelica_ndims(a) = ndims(a)

"""
    modelica_scalar(a)

Modelica: scalar(a) converts a one-element array to a scalar.
"""
function modelica_scalar(a)
  if a isa AbstractArray
    return only(a)
  else
    return a
  end
end

"""
    modelica_vector(a)

Modelica: vector(a) converts an array to a 1D vector.
"""
modelica_vector(a) = vec(collect(a))

"""
    modelica_matrix(a)

Modelica: matrix(a) returns the first two dimensions of an array as a matrix.
For a vector, wraps it as a column matrix.
"""
function modelica_matrix(a)
  if a isa AbstractVector
    return reshape(a, length(a), 1)
  elseif a isa AbstractMatrix
    return a
  else
    return collect(a)
  end
end


#= ============================================================================
   Event and state functions
   ============================================================================ =#

"""
    modelica_smooth(p, expr)

Modelica: smooth(p, expr) guarantees expr is p-times continuously differentiable.
At code generation level this is a no-op; the hint is for symbolic analysis only.
"""
modelica_smooth(p, expr) = expr

"""
    modelica_noEvent(expr)

Modelica: noEvent(expr) disables event triggering for the expression.
At code generation level this is a no-op (the solver does not see events).
"""
modelica_noEvent(expr) = expr

"""
    modelica_sample(start, period)

Modelica: sample(start, period) is true at sample times start+i*period.
In continuous expression context the integrator is between sample events,
so this returns false. The actual periodic firing is handled by a
PeriodicCallback emitted from the when-equation pass (see codeGen.jl
isPeriodic branch). Without this stub the per-model module raises
UndefVarError("sample") whenever sample() leaks into a non-when expression
(e.g. inside SignalPWM blocks used by DC-DC choppers).
"""
modelica_sample(start, period) = false

"""
    modelica_edge(b)

Modelica: edge(b) is `b and not pre(b)`, true at the instant Boolean b
becomes true. In continuous expression context the integrator sits between
events, so b == pre(b) and edge(b) is false. The actual rising-edge firing
is handled by the when-clause callback infrastructure. Without this stub
the per-model module raises UndefVarError("edge") whenever edge() leaks
into a residual or other non-when expression (e.g. switches with arc).
"""
modelica_edge(b) = false

"""
    modelica_change(v)

Modelica: change(v) is `v <> pre(v)`, true at the instant v changes value.
In continuous expression context v == pre(v), so this returns false.
Discrete change-event firing is handled by when-clause callbacks.
"""
modelica_change(v) = false

"""
    modelica_String(x[, sigDigits, padding, leftAdjust])
    modelica_String(x[, minLen, leftAdjust])

Modelica: `String(x, ...)` converts a value to its formatted string form.
The signature has several variants for Real / Integer / Boolean / Enumeration
inputs with optional formatting controls. We map all variants to a single
helper so the per-model module sees a defined function rather than the
raw Julia `String` constructor (which only accepts byte vectors / pointers
and rejects `String(::Float64, ::Int64, ::Int64, ::Bool)` etc.).

Returns Julia's `string(x)` representation. Padding / leftAdjust / sigDigits
are accepted but ignored; the result is sufficient for any downstream use
that just consumes the digits (e.g. table-name suffixes in Media examples).
"""
modelica_String(x) = string(x)
modelica_String(x, sigDigits::Integer) = string(x)
modelica_String(x, sigDigits::Integer, minLen::Integer) = string(x)
modelica_String(x, sigDigits::Integer, minLen::Integer, leftAdjust::Bool) = string(x)

"""
    modelica_homotopy(actual, simplified)

Modelica: homotopy(actual, simplified) for initialization.
We return the actual expression (lambda=1 case).
"""
modelica_homotopy(actual, simplified) = actual

"""
    modelica_semiLinear(x, k_pos, k_neg)

Modelica: semiLinear(x, k_pos, k_neg) = if x >= 0 then k_pos*x else k_neg*x.
"""
modelica_semiLinear(x, k_pos, k_neg) = ifelse(x >= 0, k_pos * x, k_neg * x)
