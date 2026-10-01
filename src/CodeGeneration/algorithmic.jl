#=
Code generation for algorithmic Modelica.
author:johti17
=#

#= Return expression for the current function being generated.
   Set before generateStatements, read by STMT_RETURN handler. =#
const _CURRENT_RETURN_EXPR = Ref{Any}(nothing)

#= Unwrap a single-valued :block expression to its inner value.
   `expToJuliaExpAlg` wraps literal DAE.ICONST/RCONST in `quote $int end`,
   which flattens to `Expr(:block, int)`. Inside `Expr(:ref, ...)` that
   prints as `a[(1;)]` — invalid Julia at eval time. =#
function _unwrapSubscriptExpr(raw)
  if raw isa Expr && raw.head === :block
    local stripped = filter(a -> !(a isa LineNumberNode), raw.args)
    length(stripped) == 1 ? stripped[1] : raw
  else
    raw
  end
end

function ensureAlgArrayLength!(arr::Vector, idx)
  local n = _algAssignedMaxIndex(idx)
  if n > length(arr)
    resize!(arr, n)
  end
  return arr
end
ensureAlgArrayLength!(arr, idx) = arr

_algAssignedMaxIndex(idx::Int) = Int(idx)
_algAssignedMaxIndex(idx::AbstractUnitRange) = isempty(idx) ? 0 : Int(last(idx))
_algAssignedMaxIndex(idx::Colon) = 0
function _algAssignedMaxIndex(idx)
  try
    return maximum(Int, idx; init=0)
  catch err
    #= Not integer indices (an InexactError): no growth. =#
    CodeGeneration.OMBackend._fallback(err, :algAssignedMaxIndex)
    return 0
  end
end

function _algIndexExpr(sub)
  local raw = @match sub begin
    DAE.INDEX(e) => expToJuliaExpAlg(e)
    DAE.SLICE(e) => expToJuliaExpAlg(e)
    DAE.WHOLEDIM() => :(:)
    _ => expToJuliaExpAlg(sub)
  end
  _unwrapSubscriptExpr(raw)
end

function _algAssignmentPreallocation(@nospecialize(lhs))
  @match lhs begin
    DAE.CREF(DAE.CREF_IDENT(ident, _, subscriptLst), _) where {length(subscriptLst) == 1} => begin
      local idx = _algIndexExpr(first(subscriptLst))
      :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.ensureAlgArrayLength!($(Symbol(ident)), $idx))
    end
    DAE.ASUB(DAE.CREF(DAE.CREF_IDENT(ident, _, _), _), subLst) where {length(subLst) == 1} => begin
      local idx = _algIndexExpr(first(subLst))
      :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.ensureAlgArrayLength!($(Symbol(ident)), $idx))
    end
    DAE.CREF(cr, _) => begin
      local allSubscripts = collect(CodeGeneration.FrontendUtil.Util.getSubscriptsFromCref(cr))
      if length(allSubscripts) == 1
        local baseName = first(split(SimulationCode.string(cr), "["))
        local idx = _algIndexExpr(first(allSubscripts))
        :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.ensureAlgArrayLength!($(Symbol(baseName)), $idx))
      else
        nothing
      end
    end
    _ => nothing
  end
end

#= The lowered left-hand side indexes the array `name`: a scalarized element
   (signalPWM[3].sawtooth.count lowered to one name) has no array to grow. =#
_indexesArray(lhs, name) = lhs isa Expr && lhs.head === :ref && _unwrapSubscriptExpr(lhs.args[1]) === name

function _algAssignment(@nospecialize(lhsExp), rhs::Expr)
  local lhs = _unwrapSubscriptExpr(expToJuliaExpAlg(lhsExp))
  local prealloc = _algAssignmentPreallocation(lhsExp)
  if prealloc === nothing || !_indexesArray(lhs, prealloc.args[2])
    return :($lhs = $rhs)
  else
    return quote
      $prealloc
      $lhs = $rhs
    end
  end
end

#= Check if a DAE.VAR is a multi-dimensional array (2+ dimensions). =#
#= Dimensions may be in v.ty (T_ARRAY) or in v.dims field. =#
function isMultiDimArray(v::DAE.VAR)::Bool
  #= First check if ty is T_ARRAY with 2+ dims =#
  tyHasMultiDims = @match v.ty begin
    DAE.T_ARRAY(dims = dims) => length(dims) >= 2
    _ => false
  end
  if tyHasMultiDims
    return true
  end
  #= Also check v.dims field (used for function parameters) =#
  hasproperty(v, :dims) || return false
  local dimCount = 0
  for _ in v.dims
    dimCount += 1
  end
  return dimCount >= 2
end

#= Check if a ModelicaFunction has any array-typed output.
   Used to decide whether the function wrapper should skip the symbolic short-circuit:
   array-returning functions must always execute their body so the result is indexable. =#
function hasArrayOutput(f::SimulationCode.ModelicaFunction)::Bool
  for v in f.outputs
    local crefType = @match v.componentRef begin
      DAE.CREF_IDENT(_, identType, _) => identType
      DAE.CREF_QUAL(_, identType, _, _) => identType
      _ => v.ty
    end
    @match crefType begin
      DAE.T_ARRAY(__) => return true
      _ => nothing
    end
  end
  return false
end

#= Compute the output dimensions for a single-array-output function.
   Returns a tuple of ints, e.g. (4,) for a vector or (3,3) for a matrix.
   Returns () if not applicable (multiple outputs, unknown dimensions, etc.). =#
function computeArrayOutputDims(f::SimulationCode.ModelicaFunction)::Tuple{Vararg{Int}}
  if length(f.outputs) != 1
    return ()
  end
  local v = f.outputs[1]
  local crefType = @match v.componentRef begin
    DAE.CREF_IDENT(_, identType, _) => identType
    DAE.CREF_QUAL(_, identType, _, _) => identType
    _ => v.ty
  end
  @match crefType begin
    DAE.T_ARRAY(_, dims) => begin
      local dimVals = Int[]
      for d in dims
        @match d begin
          DAE.DIM_INTEGER(n) => push!(dimVals, n)
          _ => return ()
        end
      end
      return Tuple(dimVals)
    end
    _ => return ()
  end
end

#= An Integer input is an Int in the function: read from the integrator (a
   discrete's value) it arrives as a Float64 (MSL TimeTable's `last`, a table
   index in getInterpolationCoefficients: "invalid index: 2.0"). A symbolic
   argument is left as it is. =#
function generateIntegerInputConversions(inputs::Vector)::Vector{Expr}
  local conversions = Expr[]
  for v in inputs
    (_funcParamIsArray(v) || !(v.ty isa DAE.T_INTEGER)) && continue
    local s = DAE_VAR_ToJulia(v)
    push!(conversions, :($s = OMBackend.CodeGeneration.AlgorithmicCodeGeneration._integerInput($s)))
  end
  return conversions
end
_integerInput(x::AbstractFloat) = isinteger(x) ? Int(x) : x
_integerInput(x) = x

#= Generate ensureArray conversion statements for multi-dimensional array parameters. =#
function generateArrayConversions(inputs::Vector)::Vector{Expr}
  conversions = Expr[]
  for v in inputs
    if isMultiDimArray(v)
      varName = Symbol(string(v.componentRef))
      push!(conversions, :($varName = OMBackend.CodeGeneration.ensureArray($varName)))
    end
  end
  return conversions
end

"""
  Generates algorithmic Modelica Code.
  Returns the generated Julia code + the names of the functions that has been generated.

  To avoid world-age issues when these functions are called from MTK's RuntimeGeneratedFunctions,
  we use a two-step approach:
  1. Generate the implementation as an anonymous function stored in MODELICA_FUNCTION_IMPLS dictionary
  2. Create a wrapper function at module load time that looks up the implementation

  The wrapper is created via createModelicaFunctionWrapper which must be called before
  the implementation is stored. This is handled in ODE_MODE_MTK_PROGRAM_GENERATION.
"""
function generateFunctions(functions::Vector{SimulationCode.ModelicaFunction})::Tuple{Vector{Expr}, Vector{String}}
  local jFuncs = Expr[]
  local names = String[]
  for func in functions
    local inputs = generateIOL(func.inputs)
    local outputs = generateIOL(func.outputs)
    local f
    #= Normalize function name: replace dots with underscores for valid Julia identifiers =#
    local normalizedName = func.name
    local nArgs = length(inputs)
    local isArrayFunc = hasArrayOutput(func)
    #= Enable scalar element-extraction wrappers for ALL array-returning functions.
       When called with symbolic args, array functions produce SymbolicUtils.array_literal
       nodes that Pantelides index reduction cannot differentiate. Scalar wrappers produce
       Term{Real} nodes instead, which Symbolics can differentiate via chain rule. =#
    local outputDims = isArrayFunc ? computeArrayOutputDims(func) : ()
    inputsJL = if nArgs > 1
      tuple(inputs...)
    elseif nArgs == 1
      inputs[1]
    else
      ()  #= Empty tuple for no inputs =#
    end
    @match func begin
      SimulationCode.MODELICA_FUNCTION(__) => begin
        local locals = generateLocals(func.locals)
        local outputDefaults = generateOutputDefaults(func.outputs)
        local inputConversions = vcat(generateArrayConversions(func.inputs), generateIntegerInputConversions(func.inputs))
        local returnExpr = if length(outputs) > 1
          Expr(:tuple, outputs...)
        elseif length(outputs) == 1
          outputs[1]
        else
          nothing
        end
        _CURRENT_RETURN_EXPR[] = returnExpr
        local statements = generateStatements(func.statements)
        #= Build the anonymous function expression manually to avoid parsing issues =#
        local funcBody = Expr(:block, inputConversions..., outputDefaults..., locals..., statements..., :(return $(returnExpr)))
        local anonFunc = if nArgs == 0
          Expr(:->, Expr(:tuple), funcBody)
        elseif inputsJL isa Tuple
          Expr(:->, Expr(:tuple, inputsJL...), funcBody)
        else
          Expr(:->, inputsJL, funcBody)
        end

        f = quote
          #= Create the wrapper function (always re-created to apply correct flags) =#
          OMBackend.CodeGeneration.createModelicaFunctionWrapper($(QuoteNode(Symbol(normalizedName))), $(nArgs), $(isArrayFunc), $(outputDims))
          #= Store the implementation in the dictionary =#
          OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[$(QuoteNode(Symbol(normalizedName)))] = $(anonFunc)
        end
      end
      SimulationCode.EXTERNAL_MODELICA_FUNCTION(__) => begin
        local funcBody = if func.language == "FORTRAN 77"
          _fortranExternalBody(func)
        else
          local extCall = namespaceifyExternalFunction(Meta.parse(func.libInfo))
          #= Allocate ccall-mutable buffers for every output, convert array inputs
             to the right C element type, then dereference Refs in the return. =#
          local extInputConversions = generateExternalInputConversions(func.inputs)
          local extOutputAllocs = generateExternalOutputAllocations(func.outputs)
          local returnExpr = generateExternalReturnExpr(func.outputs)
          #= The protected locals the call passes (MSL realFFT_raw's work array). =#
          Expr(:block, extInputConversions..., extOutputAllocs..., generateLocals(func.locals)..., extCall, returnExpr)
        end
        local anonFunc = if nArgs == 0
          Expr(:->, Expr(:tuple), funcBody)
        elseif inputsJL isa Tuple
          Expr(:->, Expr(:tuple, inputsJL...), funcBody)
        else
          Expr(:->, inputsJL, funcBody)
        end

        f = quote
          #= Create the wrapper function (always re-created to apply correct flags) =#
          OMBackend.CodeGeneration.createModelicaFunctionWrapper($(QuoteNode(Symbol(normalizedName))), $(nArgs), $(isArrayFunc), $(outputDims))
          #= Store the implementation in the dictionary =#
          OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[$(QuoteNode(Symbol(normalizedName)))] = $(anonFunc)
        end
      end
    end
    push!(jFuncs, f)
    push!(names, normalizedName)
  end
  return jFuncs, names
end

function generateIOL(inputs::Vector)::Vector{Symbol}
  local jInputs::Vector{Symbol} = Symbol[]
  for i in inputs
    #= Check if this is a record type. If so, flatten into individual field parameters =#
    local flattenedInputs::Vector{Symbol} = flattenRecordInput(i)
    if !isempty(flattenedInputs)
      append!(jInputs, flattenedInputs)
    else
      local s = DAE_VAR_ToJulia(i)
      #= Complex type, prefixed with void* =#
      push!(jInputs, s)
    end
  end
  return jInputs
end

"""
  Check if a DAE.VAR is a record type and flatten it into individual field parameters.
  Returns a vector of Symbols for the flattened fields, or empty vector if not a record.
"""
function flattenRecordInput(v::DAE.VAR)::Vector{Symbol}
  local baseName::String = string(v.componentRef)
  @match v.ty begin
    DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _) => begin
      local flattenedSymbols::Vector{Symbol} = Symbol[]
      for field in varLst
        @match field begin
          DAE.TYPES_VAR(fieldName, _, _, _, _) => begin
            local flatName::String = baseName * COMPONENT_SEPARATOR * fieldName
            push!(flattenedSymbols, Symbol(flatName))
          end
          _ => nothing
        end
      end
      return flattenedSymbols
    end
    _ => return Symbol[]
  end
end

function generateLocals(inputs::Vector)
  local jInputs = Expr[]
  for i in inputs
    #= Record-typed local: mirror generateIOL — flatten into per-field locals
       using `<baseName>_<fieldName>`, matching the convention used elsewhere
       (e.g. expToJuliaExpAlg's CREF_QUAL site). =#
    local flat = flattenRecordInput(i)
    if !isempty(flat)
      for fs in flat
        push!(jInputs, Expr(:local, fs))
      end
      continue
    end
    local s = DAE_VAR_ToJulia(i)
    #= If the variable has a binding expression (e.g., protected constants),
       generate local s = <bindingExpr> instead of just local s =#
    local hasBinding = @match i.binding begin
      SOME(bindingExp) => begin
        #= An array as a copy: `Real Awork[n, n] = A` is written into, `A` not. =#
        local bindExpr = expToJuliaExpAlg(bindingExp)
        push!(jInputs, :(local $s = $(_funcParamIsArray(i) ? :(Base.copy($bindExpr)) : bindExpr)))
        true
      end
      _ => false
    end
    if !hasBinding && _funcParamIsArray(i)
      #= `Real invMMX[size(X, 1)]` (MSL Media massToMoleFractions): an array, not the
         scalar default of its element type. =#
      push!(jInputs, :(local $s = $(_arrayDefault(i))))
    elseif !hasBinding
      local defaultVal = @match i.ty begin
        DAE.T_REAL(__) => 0.0
        DAE.T_INTEGER(__) => 0
        DAE.T_BOOL(__) => false
        DAE.T_STRING(__) => ""
        DAE.T_ENUMERATION(__) => 1
        _ => nothing
      end
      if defaultVal === nothing
        push!(jInputs, Expr(:local, s))
      else
        push!(jInputs, :(local $s = $defaultVal))
      end
    end
  end
  return jInputs
end

#= A zero-filled array for an array-typed function variable, its dimensions possibly
   depending on the inputs; an empty one where they are not known. =#
function _arrayDefault(v::DAE.VAR)
  local jlElemDefault = @match _funcParamElemType(v) begin
    DAE.T_REAL(__) => :Float64
    DAE.T_INTEGER(__) => :Int
    DAE.T_BOOL(__) => :Bool
    _ => :Float64
  end
  local dimExprs = map(_daeDimToJulia, collect(_funcParamDims(v)))
  local unresolved = any(d -> d isa Number && d <= 0, dimExprs)
  return !isempty(dimExprs) && !unresolved ? :(zeros($(jlElemDefault), $(dimExprs...))) : :($(jlElemDefault)[])
end

"""
  Generate default-initialized local declarations for Modelica function output variables.
  In Modelica, output variables are implicitly initialized (Real=0.0, Integer=0, Bool=false).
  Without this, variables assigned only in one if-branch are undefined in other branches.
"""
function generateOutputDefaults(outputs::Vector)::Vector{Expr}
  local decls = Expr[]
  for v in outputs
    local s = DAE_VAR_ToJulia(v)
    #= An output with a binding starts at it (`output Real x[n] = b`), an array
       as a copy: the body or a FORTRAN 77 routine writes into it, not into `b`. =#
    if v.binding isa SOME
      local bindExpr = expToJuliaExpAlg(v.binding.data)
      push!(decls, :(local $s = $(_funcParamIsArray(v) ? :(Base.copy($bindExpr)) : bindExpr)))
      continue
    end
    local defaultVal = if _funcParamIsArray(v)
      _arrayDefault(v)
    else
      @match v.ty begin
        DAE.T_REAL(__) => 0.0
        DAE.T_INTEGER(__) => 0
        DAE.T_BOOL(__) => false
        DAE.T_STRING(__) => ""
        _ => 0
      end
    end
    push!(decls, :(local $s = $defaultVal))
  end
  return decls
end

#=
EXTERNAL_MODELICA_FUNCTION codegen helpers.

The MODELICA_FUNCTION arm uses Julia-level rebinding via assignment, so a
default like `local result = 0.0` works because the body's `result := expr`
overwrites it. EXTERNAL functions cannot rebind from C — the C function
mutates a buffer reachable through a pointer. The Julia ccall ABI requires:

- A Modelica scalar `Real` output  → ccall arg type `Ref{Cdouble}` → allocate
  `Ref{Cdouble}(0.0)` on the Julia side, dereference with `[]` for the return.
- A Modelica scalar `Integer`/`Boolean` output → `Ref{Cint}` (Modelica Int
  maps to C `int` not Int64) → dereference with `[]` for the return.
- A Modelica array `Real[N]` → `Vector{Cdouble}` of length N → return as-is.
- A Modelica array `Integer[N]` → `Vector{Cint}` of length N → return as-is.

Inputs that are arrays of `Integer` need conversion to `Vector{Cint}` before
the ccall, since the caller may legitimately pass `Vector{Int64}` and Julia's
ccall will not silently widen the pointer cast.
=#

#= The size of a dimension of a function variable; `[:]` starts empty and
   grows as it is assigned (ensureAlgArrayLength!). A Boolean dimension
   (indexed by false and true) is not supported: an ArgumentError at the
   first index before. =#
_daeDimToJulia(d) = @match d begin
  DAE.DIM_INTEGER(int) => int
  DAE.DIM_EXP(exp) => expToJuliaExpAlg(exp)
  DAE.DIM_ENUM(size = n) => n
  DAE.DIM_UNKNOWN(__) => 0
  _ => CodeGeneration.unsupported("this array dimension of a function variable", d)
end

function _ccallElemType(elemTy)
  @match elemTy begin
    DAE.T_REAL(__) => :Cdouble
    DAE.T_INTEGER(__) => :Cint
    DAE.T_BOOL(__) => :Cint
    _ => :Float64
  end
end

#= DAE.VAR encodes function-parameter array shape as (ty=elementType, dims=List).
   Top-level array vars use DAE.T_ARRAY(ty=elementType, dims=List). Treat both
   as arrays. =#
function _funcParamIsArray(v::DAE.VAR)::Bool
  @match v.ty begin
    DAE.T_ARRAY(__) => true
    _ => begin
      local n = 0
      for _ in v.dims
        n += 1
      end
      n > 0
    end
  end
end

function _funcParamElemType(v::DAE.VAR)
  @match v.ty begin
    DAE.T_ARRAY(ty = e) => e
    _ => v.ty
  end
end

function _funcParamDims(v::DAE.VAR)
  @match v.ty begin
    DAE.T_ARRAY(dims = d) => d
    _ => v.dims
  end
end

function generateExternalOutputAllocations(outputs::Vector)::Vector{Expr}
  local decls = Expr[]
  for v in outputs
    local s = DAE_VAR_ToJulia(v)
    local alloc = if _funcParamIsArray(v)
      local jlElemType = _ccallElemType(_funcParamElemType(v))
      local dimExprs = map(_daeDimToJulia, collect(_funcParamDims(v)))
      local unresolved = any(d -> d isa Number && d <= 0, dimExprs)
      if !isempty(dimExprs) && !unresolved
        :(zeros($(jlElemType), $(dimExprs...)))
      else
        :($(jlElemType)[])
      end
    else
      @match v.ty begin
        DAE.T_REAL(__) => :(Ref{Cdouble}(0.0))
        DAE.T_INTEGER(__) => :(Ref{Cint}(Cint(0)))
        DAE.T_BOOL(__) => :(Ref{Cint}(Cint(0)))
        DAE.T_STRING(__) => ""
        _ => 0
      end
    end
    push!(decls, :(local $s = $alloc))
  end
  return decls
end

#= A Modelica Integer (or Boolean) array for a C `int*`: rounded, since an
   Integer held in the Float64 state vector can be an ulp off after a solver
   step (the MSL noise generators' xorshift states, 1.3705433889999998e9 for
   1370543389: `convert` threw InexactError in ActuatorWithNoise). =#
_roundToCint(x::AbstractArray{Cint}) = x
_roundToCint(x::AbstractArray) = map(v -> v isa Integer ? Cint(v) : round(Cint, v), x)
_roundToCint(x) = x

function generateExternalInputConversions(inputs::Vector)::Vector{Expr}
  local conversions = Expr[]
  for v in inputs
    local s = DAE_VAR_ToJulia(v)
    if _funcParamIsArray(v)
      local jlElemType = _ccallElemType(_funcParamElemType(v))
      local nDims = length(collect(_funcParamDims(v)))
      local containerTy = nDims >= 2 ? :(Matrix{$jlElemType}) : :(Vector{$jlElemType})
      local value = jlElemType === :Cint ? :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration._roundToCint($s)) : s
      push!(conversions, :($s = convert($containerTy, $value)))
    end
  end
  return conversions
end

#= An external "FORTRAN 77" function (Modelica.Math.Matrices.LAPACK): its
   locals and outputs initialized from their bindings (Awork = A, x = b), the
   routine of Julia's LAPACK called with every argument by reference (arrays as
   fresh Float64/Int64 copies it may overwrite, scalars in Refs read back after
   the call, a character with its hidden length last, as gfortran passes it),
   and the outputs returned. =#
function _fortranExternalBody(func)::Expr
  local call = Meta.parse(func.libInfo)
  if call isa Expr && call.head === :toplevel && length(call.args) == 1
    call = call.args[1]
  end
  local (resultVar, callExpr) = call isa Expr && call.head === :(=) ? (call.args[1], call.args[2]) : (nothing, call)
  (callExpr isa Expr && callExpr.head === :call && callExpr.args[1] isa Symbol) ||
    OMBackend.unsupported("an external FORTRAN 77 call", func.libInfo)
  local routine = callExpr.args[1]
  local vars = Dict{Symbol, DAE.VAR}()
  for v in Iterators.flatten((func.inputs, func.outputs, func.locals))
    vars[Symbol(DAE_VAR_ToJulia(v))] = v
  end
  local prep = Expr[]; local types = Any[]; local args = Any[]; local readBack = Expr[]
  local hiddenLengths = Any[]
  local refs = Dict{Symbol, Symbol}()
  for a in callExpr.args[2:end]
    if a isa String
      push!(types, :(Ref{UInt8})); push!(args, UInt8(first(a)))
      push!(hiddenLengths, 1)
    elseif a isa Integer
      push!(types, :(Ref{Int64})); push!(args, Int64(a))
    elseif a isa AbstractFloat
      push!(types, :(Ref{Float64})); push!(args, Float64(a))
    elseif a isa Symbol && haskey(vars, a)
      local v = vars[a]
      local elem = _funcParamIsArray(v) ? _funcParamElemType(v) : v.ty
      if elem isa DAE.T_STRING
        _funcParamIsArray(v) && OMBackend.unsupported("a FORTRAN 77 character array", a)
        push!(types, :(Ref{UInt8})); push!(args, :(UInt8(first($a))))
        push!(hiddenLengths, 1)
        continue
      end
      local jlType = elem isa DAE.T_REAL ? :Float64 : :Int64
      if _funcParamIsArray(v)
        haskey(refs, a) || (refs[a] = a; push!(prep, :($a = Array{$jlType}($a))))
        push!(types, :(Ptr{$jlType})); push!(args, a)
      else
        local r = get!(refs, a) do
          local r = Symbol("_ref_", a)
          push!(prep, :(local $r = Ref{$jlType}($a)))
          push!(readBack, elem isa DAE.T_BOOL ? :($a = $r[] != 0) :
                          elem isa DAE.T_REAL ? :($a = $r[]) : :($a = Int($r[])))
          r
        end
        push!(types, :(Ref{$jlType})); push!(args, r)
      end
    else
      OMBackend.unsupported("a FORTRAN 77 argument", a)
    end
  end
  append!(types, fill(:Clong, length(hiddenLengths))); append!(args, hiddenLengths)
  local resultType = resultVar === nothing ? :Cvoid :
    (vars[resultVar].ty isa DAE.T_REAL ? :Float64 : :Int64)
  local ccallExpr = Expr(:call, :ccall,
                         :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.lapackFunction($(QuoteNode(routine)))),
                         resultType, Expr(:tuple, types...), args...)
  local callStmt = resultVar === nothing ? ccallExpr : :($resultVar = $ccallExpr)
  local outputs = [Symbol(DAE_VAR_ToJulia(v)) for v in func.outputs]
  local returnExpr = length(outputs) == 1 ? outputs[1] : Expr(:tuple, outputs...)
  return Expr(:block, generateArrayConversions(func.inputs)..., generateOutputDefaults(func.outputs)...,
              generateLocals(func.locals)..., prep..., callStmt, readBack..., returnExpr)
end

#= The routine `name` of Julia's LAPACK: libblastrampoline's ILP64 interface
   (`dgesv_64_`, 64-bit integers). =#
const LAPACK_POINTERS = Dict{Symbol, Ptr{Cvoid}}()
function lapackFunction(name::Symbol)::Ptr{Cvoid}
  return get!(LAPACK_POINTERS, name) do
    local lib = Base.Libc.Libdl.dlopen(LinearAlgebra.BLAS.libblastrampoline)
    Base.Libc.Libdl.dlsym(lib, Symbol(name, "_64_"))
  end
end

function generateExternalReturnExpr(outputs::Vector)
  if isempty(outputs)
    return nothing
  end
  local accessors = Any[]
  for v in outputs
    local s = DAE_VAR_ToJulia(v)
    local accessor = if _funcParamIsArray(v)
      :($s)
    else
      @match v.ty begin
        DAE.T_REAL(__) => :($s[])
        DAE.T_INTEGER(__) => :(Int($s[]))
        DAE.T_BOOL(__) => :($s[] != 0)
        _ => :($s)
      end
    end
    push!(accessors, accessor)
  end
  return length(accessors) == 1 ? accessors[1] : Expr(:tuple, accessors...)
end

function generateStatements(statements::Union{List{DAE.Statement}, Vector{DAE.Statement}})
  local jStmts = Expr[]
  for s in statements
    stmt = generateStatement(s)
    push!(jStmts, stmt)
  end
  return jStmts
end

Base.@nospecializeinfer function generateStatement(@nospecialize(s::DAE.Statement))
  CodeGeneration.unsupported("statement", s)
end

function generateStatement(stmt::DAE.STMT_NORETCALL)
  return expToJuliaExpAlg(stmt.exp)
end

function generateStatement(stmt::DAE.STMT_ASSIGN)
  local scalarised = _recordAssignment(stmt.exp1, stmt.exp)
  scalarised === nothing || return scalarised
  local rhs = expToJuliaExpAlg(stmt.exp)
  return _algAssignment(stmt.exp1, rhs)
end

"""
    _recordAssignment(lhsExp::DAE.Exp, rhsExp::DAE.Exp) -> Union{Nothing,Expr}

Scalarise a record-typed assignment onto its flattened `<base>_<field>` symbols
(the naming `flattenRecordInput` uses), or return `nothing` when `lhsExp` is not a
plain record cref. A record copy `lhs := rhs` becomes per-field assignments; a
record-valued call `lhs := f(args...)`, a record literal and an if-expression choosing
between records scatter their field tuple (a one-field record is its field). Any
other value of a record is refused.
"""
function _recordAssignment(lhsExp::DAE.Exp, rhsExp::DAE.Exp)::Union{Nothing, Expr}
  local lhs = _recordCrefFields(lhsExp)
  lhs === nothing && return nothing
  local lhsBase = lhs[1]
  local fieldNames = lhs[2]
  local rhsBase = _plainCrefName(rhsExp)
  if rhsBase !== nothing
    return Expr(:block,
      Expr[:($(_flatFieldSymbol(lhsBase, f)) = $(_flatFieldSymbol(rhsBase, f))) for f in fieldNames]...)
  elseif _isFunctionCall(rhsExp) ||
         ((rhsExp isa DAE.IFEXP || rhsExp isa DAE.RECORD) && _hasNoRecordFields(lhsExp.ty))
    #= A record literal, or an if-expression choosing between records, evaluates to the
       fields as a tuple too (MSL MixtureGasNasa setState_pTX: `state := if ... then
       ThermodynamicState(...) else ...` went to an unused local, and the fields stayed 0). =#
    local targets = [_flatFieldSymbol(lhsBase, f) for f in fieldNames]
    local rhs = _recordTupleExpr(rhsExp)
    return length(targets) == 1 ? Expr(:(=), targets[1], rhs) : Expr(:(=), Expr(:tuple, targets...), rhs)
  end
  #= Assigned whole, the record's name would be bound and its fields, which
     are what is read, kept: an element of a tuple, a field or an element of
     another record, a literal with record fields. =#
  CodeGeneration.unsupported("a record assignment from this expression", rhsExp)
end

#= A record value as its field tuple: a record variable's flattened fields (it has no
   tuple of its own), an if-expression's branches each so, anything else as it lowers. =#
function _recordTupleExpr(@nospecialize(exp::DAE.Exp))
  local rec = _recordCrefFields(exp)
  rec === nothing || return Expr(:tuple, [_flatFieldSymbol(rec[1], f) for f in rec[2]]...)
  if exp isa DAE.IFEXP
    return :(if $(expToJuliaExpAlg(exp.expCond)) != 0
               $(_recordTupleExpr(exp.expThen))
             else
               $(_recordTupleExpr(exp.expElse))
             end)
  end
  return expToJuliaExpAlg(exp)
end

#= The types of a record type's fields, in declaration order; empty for another type. =#
function _recordFieldTypes(@nospecialize(ty::DAE.Type))::Vector{DAE.Type}
  local types = DAE.Type[]
  @match ty begin
    DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _) => begin
      for field in varLst
        push!(types, field.ty)
      end
    end
    _ => nothing
  end
  return types
end

#= Whether no field of record type `ty` is a record: its value's tuple is then field for
   field (a nested record's fields are flattened in its place). =#
function _hasNoRecordFields(@nospecialize(ty::DAE.Type))::Bool
  local types = _recordFieldTypes(ty)
  return !isempty(types) && all(t -> isempty(_recordFieldNames(t)), types)
end

"""
    _flatFieldSymbol(base::AbstractString, field::AbstractString) -> Symbol

Symbol for a flattened record field, `<base><COMPONENT_SEPARATOR><field>`, matching
the names emitted by `flattenRecordInput` for record parameters and locals.
"""
_flatFieldSymbol(base::AbstractString, field::AbstractString)::Symbol = Symbol(base, COMPONENT_SEPARATOR, field)

"""
    _isFunctionCall(exp::DAE.Exp) -> Bool

True for a (record-valued) function-call expression.
"""
function _isFunctionCall(exp::DAE.Exp)::Bool
  return @match exp begin
    DAE.CALL(__) => true
    _ => false
  end
end

"""
    _algCallArgs(argExps::List; builtin = false) -> Vector{Any}

Expand function-call arguments, replacing each record-typed cref with its flattened
`<base>_<field>` field symbols and splatting any other record value, so a record argument
is passed as its scalar fields, matching the callee's flattened parameter list
(`flattenRecordInput`). A builtin's arguments stay whole.
"""
function _algCallArgs(argExps::List; builtin::Bool = false)::Vector{Any}
  local out = Any[]
  for arg in argExps
    #= A record without fields (MSL Media's f_nonlinear_Data()) is no argument:
       the callee's flattened inputs have none for it. Passed as an empty value,
       every later argument was shifted (a MethodError of the wrapper's arity). =#
    (!builtin && _isEmptyRecordValue(arg)) && continue
    local rec = _recordCrefFields(arg)
    local valueTy = rec === nothing && !builtin ? _recordValueType(arg) : nothing
    #= A builtin takes a record whole; a one-field record is returned bare. =#
    if valueTy !== nothing && length(_recordFieldNames(valueTy)) > 1
      #= A record that is not a named variable (a call's result, an element of a record
         array) evaluates to its fields as a tuple: splatted, as the callee takes a record
         input field by field (the MSL ReferenceAir's `rho_props_pT(p, T,
         airBaseProp_pT(p, T))`, the ideal gases' `h_T(data[i], T, ...)`). =#
      push!(out, Expr(:..., expToJuliaExpAlg(arg)))
    elseif rec === nothing
      push!(out, expToJuliaExpAlg(arg))
    else
      for fieldName in rec[2]
        push!(out, _flatFieldSymbol(rec[1], fieldName))
      end
    end
  end
  return out
end

function _isEmptyRecordValue(@nospecialize(exp::DAE.Exp))::Bool
  local ty = @match exp begin
    DAE.CALL(attr = attr) => attr.ty
    DAE.RECORD(ty = ty) => ty
    DAE.CREF(_, ty) => ty
    _ => nothing
  end
  return ty isa DAE.T_COMPLEX && ty.complexClassType isa DAE.ClassInf.RECORD && isempty(ty.varLst)
end

#= The record type of `exp` where it is not a named record variable: a call returning a
   record, a record literal (a constant the frontend folded: the ideal gas `data`), or one
   element of an array of records (a scalar subscript per dimension); nothing otherwise. Such
   an expression evaluates to the record's fields as a tuple. =#
function _recordValueType(@nospecialize(exp::DAE.Exp))
  local ty = @match exp begin
    DAE.CALL(attr = attr) => attr.ty
    DAE.RECORD(ty = ty) => ty
    DAE.CREF(_, ty) => ty
    DAE.ASUB(exp = DAE.ARRAY(ty = ty), sub = subs) => _elementTypeAt(ty, subs)
    DAE.ASUB(exp = DAE.CREF(_, ty), sub = subs) => _elementTypeAt(ty, subs)
    _ => nothing
  end
  return ty !== nothing && !isempty(_recordFieldNames(ty)) ? ty : nothing
end

#= The element type of array type `ty` subscripted by `subs`, when there is one scalar
   subscript per dimension; nothing for a slice or a partial subscript. =#
function _elementTypeAt(@nospecialize(ty::DAE.Type), subs::List)
  local nDims = 0
  local t = ty
  while t isa DAE.T_ARRAY
    nDims += length(collect(t.dims))
    t = t.ty
  end
  local scalar = all(subs) do sub
    sub isa DAE.INDEX && !(sub.exp isa DAE.RANGE || sub.exp isa DAE.ARRAY)
  end
  return scalar && length(collect(subs)) == nDims ? t : nothing
end

#= The position of field `ix` in the tuple a record-valued `exp` evaluates to, where every
   field up to it is a scalar: a function returns a nested record's fields flattened in its
   place, so a later field's position is not its index. -1 (read it by name) otherwise. =#
function _positionalFieldIndex(@nospecialize(exp::DAE.Exp), ix::Integer)::Int
  local ty = _recordValueType(exp)
  ty === nothing && return -1
  local fieldTypes = _recordFieldTypes(ty)
  1 <= ix <= length(fieldTypes) || return -1
  any(k -> !isempty(_recordFieldNames(fieldTypes[k])), 1:ix) && return -1
  return ix
end

"""
    _recordCrefFields(exp::DAE.Exp) -> Union{Nothing,Tuple{String,Vector{String}}}

`(baseIdent, fieldNames)` for a subscript-free simple CREF of a record or an array of records
(whose fields are then arrays), else `nothing`.
"""
function _recordCrefFields(exp::DAE.Exp)::Union{Nothing, Tuple{String, Vector{String}}}
  local base = _plainCrefName(exp)
  base === nothing && return nothing
  return @match exp begin
    DAE.CREF(_, ty) => begin
      #= A whole array of records too: its fields are arrays (y_re, y_im). =#
      local fieldNames = _recordFieldNames(ty isa DAE.T_ARRAY ? _arrayElementType(ty) : ty)
      isempty(fieldNames) ? nothing : (base, fieldNames)
    end
    _ => nothing
  end
end

"""
    _recordFieldSymbol(base::AbstractString, recordType::DAE.Type, subscripts) -> Union{Nothing,Symbol}

Flat field symbol `<base>_<kth field>` when `recordType` is a record and
`subscripts` is a single constant integer index in range; `nothing` otherwise.
This is field access on a record, not array indexing.
"""
function _recordFieldSymbol(base::AbstractString, recordType::DAE.Type, subscripts)::Union{Nothing, Symbol}
  local fieldNames = _recordFieldNames(recordType)
  isempty(fieldNames) && return nothing
  length(subscripts) == 1 || return nothing
  local index = _constantIntegerIndex(first(subscripts))
  (index === nothing || index < 1 || index > length(fieldNames)) && return nothing
  return _flatFieldSymbol(base, fieldNames[index])
end

"""
    _recordCrefIndexFieldSymbol(cref::DAE.ComponentRef) -> Union{Nothing,Symbol}

Flat field symbol for a record cref carrying one constant-integer subscript, else `nothing`.
"""
function _recordCrefIndexFieldSymbol(cref::DAE.ComponentRef)::Union{Nothing, Symbol}
  return @match cref begin
    DAE.CREF_IDENT(ident, identType, subscripts) => _recordFieldSymbol(ident, identType, subscripts)
    _ => nothing
  end
end

"""
    _recordIndexFieldSymbol(arrExp::DAE.Exp, subscripts) -> Union{Nothing,Symbol}

Flat field symbol for an ASUB indexing a record cref by one constant integer, else `nothing`.
"""
function _recordIndexFieldSymbol(arrExp::DAE.Exp, subscripts)::Union{Nothing, Symbol}
  local base = _plainCrefName(arrExp)
  base === nothing && return nothing
  return @match arrExp begin
    DAE.CREF(_, ty) => _recordFieldSymbol(base, ty, subscripts)
    _ => nothing
  end
end

"""
    _constantIntegerIndex(indexExp::DAE.Exp) -> Union{Nothing,Int}

Constant integer value of an ASUB index expression (`a[k]` carries `DAE.ICONST`),
or `nothing` if the index is not a constant integer.
"""
function _constantIntegerIndex(indexExp::DAE.Exp)::Union{Nothing, Int}
  return @match indexExp begin
    DAE.ICONST(value) => value
    _ => nothing
  end
end

"""
    _constantIntegerIndex(subscript::DAE.Subscript) -> Union{Nothing,Int}

Constant integer value of a subscripted-CREF subscript (`DAE.INDEX(exp)`), or
`nothing` if it is not a constant integer index.
"""
function _constantIntegerIndex(subscript::DAE.Subscript)::Union{Nothing, Int}
  return @match subscript begin
    DAE.INDEX(indexExp) => _constantIntegerIndex(indexExp)
    _ => nothing
  end
end

#= The element type of a (nested) array type. =#
_arrayElementType(@nospecialize(ty::DAE.Type)) = ty isa DAE.T_ARRAY ? _arrayElementType(ty.ty) : ty

"""
    _recordFieldNames(ty::DAE.Type) -> Vector{String}

Field names of a record type's `varLst` in declaration order; empty if `ty` is
not a record.
"""
function _recordFieldNames(ty::DAE.Type)::Vector{String}
  local names = String[]
  @match ty begin
    DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _) => begin
      for field in varLst
        @match field begin
          DAE.TYPES_VAR(fieldName, _, _, _, _) => push!(names, fieldName)
          _ => nothing
        end
      end
    end
    _ => nothing
  end
  return names
end

"""
    _plainCrefName(exp::DAE.Exp) -> Union{String,Nothing}

Identifier of a subscript-free simple CREF (the base for its flattened
`<base>_<field>` symbols), or `nothing` for qualified/subscripted/non-CREF exps.
"""
function _plainCrefName(exp::DAE.Exp)::Union{String, Nothing}
  return @match exp begin
    DAE.CREF(DAE.CREF_IDENT(ident, _, subscripts), _) => isempty(subscripts) ? ident : nothing
    _ => nothing
  end
end

#= `(a, , b) := f(...)` as a Julia destructuring `(a, _, b) = f(...)`. A
   generated function returns a record output as its fields in place
   (generateIOL), so a record target takes them into its flattened field
   symbols and an omitted record output skips one `_` per field. A target that
   is no plain name (`v[2]`, `eigenvalues[:, 1]`, a record's field) takes its
   element through a temporary and an ordinary assignment. =#
function generateStatement(stmt::DAE.STMT_TUPLE_ASSIGN)
  local outputTypes = stmt.type_ isa DAE.T_TUPLE ? collect(stmt.type_.types) : nothing
  outputTypes === nothing || length(outputTypes) >= listLength(stmt.expExpLst) ||
    CodeGeneration.unsupported("a tuple assignment with more targets than outputs", stmt)
  local targets = Any[]
  local stores = Expr[]
  for (k, target) in enumerate(stmt.expExpLst)
    local fields = outputTypes === nothing ? String[] : _recordFieldNames(outputTypes[k])
    if !isempty(fields)
      _hasNoRecordFields(outputTypes[k]) ||
        CodeGeneration.unsupported("a tuple assignment of a record output with record fields", stmt)
      append!(targets, _recordTupleTargets(target, fields, stmt))
    elseif target isa DAE.CREF && target.componentRef isa DAE.WILD
      push!(targets, :_)
    elseif !(target isa DAE.CREF) || _recordCrefFields(target) !== nothing
      CodeGeneration.unsupported("this target of a tuple assignment", target)
    elseif _plainCrefName(target) !== nothing
      push!(targets, Symbol(_plainCrefName(target)))
    else
      local tmp = Symbol("#tupleTarget", k)  #= no Modelica name has a '#' =#
      push!(targets, tmp)
      push!(stores, _algAssignment(target, Expr(:block, tmp)))
    end
  end
  local assignment = Expr(:(=), Expr(:tuple, targets...), expToJuliaExpAlg(stmt.exp))
  return isempty(stores) ? assignment : Expr(:block, assignment, stores...)
end

#= The targets of a record output's fields: an omitted output's `_`s, a plain
   record name's flattened field symbols. =#
function _recordTupleTargets(@nospecialize(target::DAE.Exp), fields::Vector{String}, stmt)::Vector{Symbol}
  target isa DAE.CREF && target.componentRef isa DAE.WILD && return fill(:_, length(fields))
  local rec = _recordCrefFields(target)
  (rec === nothing || rec[2] != fields) &&
    CodeGeneration.unsupported("this record target of a tuple assignment", stmt)
  return Symbol[_flatFieldSymbol(rec[1], f) for f in fields]
end

function generateStatement(stmt::DAE.STMT_ASSIGN_ARR)
  local scalarised = _recordAssignment(stmt.lhs, stmt.exp)
  scalarised === nothing || return scalarised
  local rhs = expToJuliaExpAlg(stmt.exp)
  return _algAssignment(stmt.lhs, rhs)
end

function generateStatement(stmt::DAE.STMT_WHILE)
  local cond = expToJuliaExpAlg(stmt.exp)
  local stmts = generateStatements(stmt.statementLst)
  quote
    while ($(cond))
      $(stmts...)
    end
  end
end

function generateStatement(stmt::DAE.STMT_RETURN)::Expr
  local retExpr = _CURRENT_RETURN_EXPR[]
  if retExpr === nothing
    return :(return)
  else
    return :(return $(retExpr))
  end
end

function generateStatement(stmt::DAE.STMT_BREAK)::Expr
  :(break)
end

function generateStatement(stmt::DAE.STMT_CONTINUE)::Expr
  :(continue)
end

"""
  Generates for statements.
"""
function generateStatement(stmt::DAE.STMT_FOR)::Expr
  local iterVar = Symbol(stmt.iter)
  local rangeExpr = expToJuliaExpAlg(stmt.range)
  local bodyStmts = generateStatements(stmt.statementLst)
  local blck = Expr(:block)
  for s in bodyStmts
    push!(blck.args, s)
  end
  return Expr(:for, Expr(:(=), iterVar, rangeExpr), blck)
end

"""
  Generates If statements
"""
function generateStatement(stmt::DAE.STMT_IF)::Expr
  #= Coerce to Bool: a Boolean discrete lives numerically (0.0/1.0) on the state
     vector, so a bare `if <float>` throws TypeError. `!= 0` is idempotent on a
     genuine Bool. =#
  local cond = :($(expToJuliaExpAlg(stmt.exp)) != 0)
  local stmts = generateStatements(stmt.statementLst)
  local res = @match stmt.else_ begin
    DAE.NOELSE(__) => begin
      local expr = Expr(:if, cond)
      local blck = Expr(:block)
      for stmt in stmts
        push!(blck.args, stmt)
      end
      push!(expr.args, blck)
      expr
    end
    DAE.ELSE(__) => begin
      local expr = Expr(:if, cond)
      local blck = Expr(:block)
      for stmt in stmts
        push!(blck.args, stmt)
      end
      push!(expr.args, blck)
      local elseStmts = generateStatement(stmt.else_)
      push!(expr.args, elseStmts)
      expr
    end
    DAE.ELSEIF(__) => begin
      local expr = Expr(:if, cond)
      local blck = Expr(:block)
      for stmt in stmts
        push!(blck.args, stmt)
      end
      push!(expr.args, blck)
      local elseIfs = generateStatement(stmt.else_)
      push!(expr.args, elseIfs)
      expr
    end
  end
  return res
end

"""
For the else branch we generate a block and add the statements of the ELSE to this block.
Should never be called at the top level.
"""
function generateStatement(stmt::DAE.ELSE)::Expr
  local block = Expr(:block)
  stmts = generateStatements(stmt.statementLst)
  for stmt in stmts
    push!(block.args, stmt)
  end
  return block
end

"""
For the elseif branch we create an elseif expression.
Similar to the else this should never be called from the top level.
"""
function generateStatement(stmt::DAE.ELSEIF)::Expr
  #= See STMT_IF: coerce a numerically-encoded Boolean condition to Bool. =#
  local cond = :($(expToJuliaExpAlg(stmt.exp)) != 0)
  local stmts = generateStatements(stmt.statementLst)
  local blck = Expr(:block)
  for s in stmts
    push!(blck.args, s)
  end
  local elseExpr = @match stmt.else_ begin
    DAE.NOELSE(__) => nothing
    _ => generateStatement(stmt.else_)
  end
  if elseExpr === nothing
    Expr(:elseif, cond, blck)
  else
    Expr(:elseif, cond, blck, elseExpr)
  end
end

#= AssertionLevel = enumeration(warning, error): warning is literal 1, so the
   level is matched by name. Any other level expression means error. =#
function isWarningAssertionLevel(@nospecialize(level))::Bool
  level isa DAE.ENUM_LITERAL || return false
  local p = level.name
  while !(p isa Absyn.IDENT)
    p = p.path                    # QUALIFIED(name, path), FULLYQUALIFIED(path)
  end
  return p.name == "warning"
end

"""
  Generates Julia code for Modelica assert statements.
  AssertionLevel.error (the default) throws an error when the condition is false,
  AssertionLevel.warning prints a warning.
"""
function generateStatement(stmt::DAE.STMT_ASSERT)::Expr
  local condExpr = expToJuliaExpAlg(stmt.cond)
  local msgExpr = expToJuliaExpAlg(stmt.msg)
  if !isWarningAssertionLevel(stmt.level)
    #= AssertionLevel.error - throw an error. Base.error: a Modelica local may be
       called `error` (MSL Water IF97's `Integer error` flag of the inverse
       iterations: "objects of type Int64 are not callable"). =#
    quote
      if !($condExpr)
        Base.error($msgExpr)
      end
    end
  else
    #= AssertionLevel.warning - print a warning =#
    quote
      if !($condExpr)
        @warn $msgExpr
      end
    end
  end
end

#= SimCode-Exp entry: codegen consumes `SimulationCode.Exp`. =#
"""
  Maps a DAE expression to a Julia expression for algorithmic code in Modelica Functions(!).
  Since functions do not use the model HT the original name is preserved for algorithmic generation.
For algorithmic code outside Modelica functions do not call this function.
"""
Base.@nospecializeinfer function expToJuliaExpAlg(@nospecialize(exp::SimulationCode.Exp))::Expr
  return expToJuliaExpAlg(SimulationCode.toDAEExp(exp))
end

Base.@nospecializeinfer function expToJuliaExpAlg(@nospecialize(exp::DAE.Exp))::Expr
  local expr::Expr = begin
    local int::Int64
    local real::Float64
    local bool::Bool
    local tmpStr::String
    local cr::DAE.ComponentRef
    local e1::DAE.Exp
    local e2::DAE.Exp
    local e3::DAE.Exp
    local expl::List{DAE.Exp}
    local lstexpl::List{List{DAE.Exp}}
    @match exp begin
      DAE.BCONST(bool) => quote $bool end
      DAE.ICONST(int) => quote $int end
      DAE.RCONST(real) => quote $real end
      DAE.SCONST(tmpStr) => quote $tmpStr end
      DAE.CREF(Absyn.IDENT("time"), _) => begin
        quote t end
      end
      #= Array accesses for simple CREF_IDENT =#
      DAE.CREF(DAE.CREF_IDENT(ident, identType, subscriptLst), _) where !isempty(subscriptLst)  => begin
        local idxExprs = map(_algIndexExpr, subscriptLst)
        #= Construct proper Julia multi-dimensional indexing: arr[i, j, ...] =#
        expr = Expr(:ref, Symbol(ident), idxExprs...)
      end
      #= Qualified CREF (record field access like R.T) =#
      DAE.CREF(DAE.CREF_QUAL(ident, identType, qualSubscriptLst, innerCref), _) => begin
        local varName::String = SimulationCode.string(exp.componentRef)
        #= Replace dots with underscores to match flattened parameter names =#
        local flatName::String = varName
        #= Always emit the whole flat name as a single Symbol. Earlier code
           routed names containing subscripts through `Meta.parse` so that
           a final `[k]` would parse as Julia array indexing, but the
           flattened scalar simvars carry their subscripts INLINE in the
           identifier (e.g. `comp[2]_field`). `Meta.parse("comp[2]_field")`
           sees `[2]` then a leading-underscore identifier and emits
           `comp[2] * _field` (implicit multiplication after `]`), which
           splits the qualified name and causes UndefVarError on the bare
           array reference at simulate time. MSL Digital.Examples.RAM was
           the canonical trip via DLATRAM's inertialDelaySensitive[i]
           algorithm body. =#
        quote
          $(Symbol(flatName))
        end
      end
      DAE.CREF(cr, _)  => begin
        # A record cref with one constant-integer subscript is field access, not
        # array indexing; emit the flat field symbol so it matches scalarised fields.
        local recordField = _recordCrefIndexFieldSymbol(cr)
        if recordField !== nothing
          quote
            $(recordField)
          end
        else
          local varName::String = SimulationCode.string(cr)
          local allSubscripts = collect(CodeGeneration.FrontendUtil.Util.getSubscriptsFromCref(cr))
          if !isempty(allSubscripts)
            local baseName = first(split(varName, "["))
            local idxExprs = map(_algIndexExpr, allSubscripts)
            Expr(:ref, Symbol(baseName), idxExprs...)
          else
            quote
              $(Symbol(varName))
            end
          end
        end
      end
      DAE.UNARY(operator = op, exp = e1) => begin
        o = CodeGeneration.DAE_OP_toJuliaOperator(op)
        :($(o)($(expToJuliaExpAlg(e1))))
      end
      DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpAlg(e1)
        local rhs = expToJuliaExpAlg(e2)
        #= Special handling for vector dot product and matrix product =#
        @match op begin
          DAE.MUL_SCALAR_PRODUCT(__) => begin
            :(OMBackend.CodeGeneration.vectorDot($(lhs), $(rhs)))
          end
          DAE.MUL_MATRIX_PRODUCT(__) => begin
            #= Operands are proper Matrix: function impl params are pre-converted
               by generateArrayConversions, and array literals use ensureArray. =#
            :($(lhs) * $(rhs))
          end
          _ => begin
            local opSym = CodeGeneration.DAE_OP_toJuliaOperator(op)
            :($opSym($(lhs), $(rhs)))
          end
        end
      end
      DAE.LUNARY(operator = op, exp = e1)  => begin
        local operand = expToJuliaExpAlg(e1)
        local opSym = CodeGeneration.DAE_OP_toJuliaOperator(op)
        # Boolean discretes read back from state are 0/1 numbers; coerce before logical not.
        :($opSym($(operand) != 0))
      end
      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpAlg(e1)
        local rhs = expToJuliaExpAlg(e2)
        #= || and && are special forms in Julia, not regular functions.
           Must use Expr(:||, ...) / Expr(:&&, ...) instead of Expr(:call, :||, ...) =#
        #= Coerce operands: Boolean discretes read back from state are 0/1
           numbers, so a bare `&&`/`||` throws TypeError. `!= 0` is idempotent
           on a genuine Bool (mirrors the LUNARY `not` coercion above). =#
        @match op begin
          DAE.OR(__) => Expr(:||, :($(lhs) != 0), :($(rhs) != 0))
          DAE.AND(__) => Expr(:&&, :($(lhs) != 0), :($(rhs) != 0))
          _ => begin
            local opSym = CodeGeneration.DAE_OP_toJuliaOperator(op)
            :($opSym($(lhs), $(rhs)))
          end
        end
      end
      DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpAlg(e1)
        local rhs = expToJuliaExpAlg(e2)
        local op = CodeGeneration.DAE_OP_toJuliaOperator(op)
        :($op($(lhs), $(rhs)))
      end
      DAE.IFEXP(expCond = e1, expThen = e2, expElse = e3) => begin
        #= Coerce to Bool: a numerically-encoded Boolean discrete condition would
           make this `if <float>` throw TypeError; `!= 0` is idempotent on Bool. =#
        local cond = :($(expToJuliaExpAlg(e1)) != 0)
        local thenExp = expToJuliaExpAlg(e2)
        local elseExp = expToJuliaExpAlg(e3)
        quote
          if $(cond)
            $(thenExp)
          else
            $(elseExp)
          end
        end
      end
      DAE.CALL(path = Absyn.IDENT("pre"), expLst = explst) => begin
        #= pre(x) in an (initial) algorithm body equals the sequentially-seeded
           held value; unwrap to the argument, as the MTK expression path does. =#
        expToJuliaExpAlg(first(explst))
      end
      #= An enumeration value is an Integer here: String takes the literal names
         from the argument's type. =#
      DAE.CALL(path = Absyn.IDENT("String"), expLst = explst, attr = attr) where attr.builtin =>
        modelicaStringCall(collect(explst), expToJuliaExpAlg)
      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = explst, attr = attr)  => begin
        local funcSym = Symbol(tmpStr)
        #= Use Base.invokelatest for non-builtin functions to avoid world-age issues =#
        local callTarget = if !(attr.builtin)
          :(Base.invokelatest)
        elseif haskey(MODELICA_BUILTIN_FUNCTIONS, tmpStr)
          Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(MODELICA_BUILTIN_FUNCTIONS[tmpStr]))
        else
          funcSym
        end
        local expr = Expr(:call, callTarget)
        if !(attr.builtin)
          push!(expr.args, funcSym)
        end
        append!(expr.args, _algCallArgs(explst; builtin = attr.builtin))
        quote
          $(expr)
        end
      end
      DAE.CALL(path, expLst, attr) => begin
        local funcName = string(path)
        local funcSym = Symbol(funcName)
        local utilRuntimeName = get(MODELICA_UTILITIES_TO_RUNTIME_C, funcName, nothing)
        #= Use Base.invokelatest for non-builtin functions to avoid world-age issues.
           Route Modelica.Utilities.* qualified calls (e.g. Strings.substring)
           to OMRuntimeExternalC stubs whether or not the call is flagged builtin
           since the per-model module never binds the dot-flattened qualified name. =#
        local callTarget = if utilRuntimeName !== nothing
          Expr(:., :OMRuntimeExternalC, QuoteNode(utilRuntimeName))
        elseif !(attr.builtin)
          :(Base.invokelatest)
        elseif haskey(MODELICA_BUILTIN_FUNCTIONS, funcName)
          Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(MODELICA_BUILTIN_FUNCTIONS[funcName]))
        else
          funcSym
        end
        local expr = Expr(:call, callTarget)
        if utilRuntimeName === nothing && !(attr.builtin)
          push!(expr.args, funcSym)
        end
        append!(expr.args, _algCallArgs(expLst; builtin = attr.builtin))
        expr
      end
      DAE.CAST(ty, exp)  => begin
        #= Type cast expression =#
        local innerExpr = expToJuliaExpAlg(exp)
        @match ty begin
          DAE.T_REAL(__) => :(float($innerExpr))
          DAE.T_INTEGER(__) => :(Int(round($innerExpr)))
          DAE.T_BOOL(__) => :(Bool($innerExpr))
          #= The frontend casts to Real only (typeCast): an Integer array to a
             Real one. Left uncast, `y := m; y[1] := 1.5` failed (InexactError). =#
          DAE.T_ARRAY(__) where _arrayElementType(ty) isa DAE.T_REAL => :(float.($innerExpr))
          _ => CodeGeneration.unsupported("a cast to $(ty)", exp)
        end
      end
      DAE.ARRAY(ty, scalar, expl) => begin
        local elements = map(expl) do e
          expToJuliaExpAlg(e)
        end
        local arrExpr = Expr(:vect, elements...)
        #= If elements are themselves arrays (matrix literal), wrap with ensureArray
           to convert Vector{Vector{T}} to a proper Matrix{T} =#
        isNested = @match ty begin
          DAE.T_ARRAY(__) => true
          _ => false
        end
        if isNested
          :(OMBackend.CodeGeneration.ensureArray($arrExpr))
        else
          arrExpr
        end
      end
      DAE.RANGE(_, startExp, NONE(), stopExp) => begin
        local startExpr = expToJuliaExpAlg(startExp)
        local stopExpr = expToJuliaExpAlg(stopExp)
        :($startExpr:$stopExpr)
      end
      DAE.RANGE(_, startExp, SOME(stepExp), stopExp) => begin
        local startExpr = expToJuliaExpAlg(startExp)
        local stepExpr = expToJuliaExpAlg(stepExp)
        local stopExpr = expToJuliaExpAlg(stopExp)
        :($startExpr:$stepExpr:$stopExpr)
      end
      DAE.SIZE(arrExp, SOME(dimExp)) => begin
        local arrExpr = expToJuliaExpAlg(arrExp)
        local dimExpr = expToJuliaExpAlg(dimExp)
        :(size($arrExpr, $dimExpr))
      end
      DAE.SIZE(arrExp, NONE()) => begin
        local arrExpr = expToJuliaExpAlg(arrExp)
        :(size($arrExpr))
      end
      DAE.ENUM_LITERAL(name, index) => begin
        #= Enum literals are represented by their integer index =#
        quote $index end
      end
      DAE.ASUB(arrExp, subLst) => begin
        # A record cref indexed by a constant integer is field access, not array
        # indexing; emit the flat field symbol so it matches the scalarised fields.
        local recordField = _recordIndexFieldSymbol(arrExp, subLst)
        if recordField !== nothing
          quote
            $(recordField)
          end
        else
          local arrExpr = expToJuliaExpAlg(arrExp)
          local subs = map(_algIndexExpr, subLst)
          Expr(:ref, arrExpr, subs...)
        end
      end
      DAE.RECORD(path, exps, fieldNames, ty) => begin
        #= Record constructor: generate as a simple tuple =#
        local fieldExprs = map(expToJuliaExpAlg, exps)
        if length(fieldExprs) == 1
          #= Single element - return as-is or wrap =#
          first(fieldExprs)
        else
          Expr(:tuple, fieldExprs...)
        end
      end
      DAE.REDUCTION(reductionInfo, bodyExp, iterators) => begin
        #= Array comprehensions/reductions inside algorithmic code
           (Modelica function bodies). Mirrors the handler in expToJuliaExpMTK
           but routes bodies through expToJuliaExpAlg so iterator CREFs are
           emitted as plain Symbols rather than hash-table lookups. =#
        local bodyExpr = expToJuliaExpAlg(bodyExp)
        local iterExprs = Expr[]
        for iter in iterators
          @match iter begin
            DAE.REDUCTIONITER(id, rangeExp, guardExp, _) => begin
              guardExp === nothing || CodeGeneration.unsupported("a reduction iterator with a guard", exp)
              local rangeExpr = expToJuliaExpAlg(rangeExp)
              push!(iterExprs, Expr(:(=), Symbol(id), rangeExpr))
            end
          end
        end
        @match reductionInfo.path begin
          #= {e for i in u, j in v}: the last iterator is the first dimension
             (MLS 10.4.1.2, size [size(v), size(u)]); a Julia comprehension's
             first iterator is. It came out transposed. =#
          Absyn.IDENT("array") => Expr(:comprehension, bodyExpr, reverse(iterExprs)...)
          Absyn.IDENT("sum") => :(sum($(Expr(:generator, bodyExpr, iterExprs...))))
          Absyn.IDENT("product") => :(prod($(Expr(:generator, bodyExpr, iterExprs...))))
          Absyn.IDENT("min") => :(minimum($(Expr(:generator, bodyExpr, iterExprs...))))
          Absyn.IDENT("max") => :(maximum($(Expr(:generator, bodyExpr, iterExprs...))))
          _ => CodeGeneration.unsupported("the reduction $(reductionInfo.path)", exp)
        end
      end
      DAE.TSUB(tupleExp, ix, _) => begin
        #= Tuple-return element access inside algorithmic code. No Symbolics
           quirks to worry about here, so a direct Julia index works. =#
        local tupExpr = expToJuliaExpAlg(tupleExp)
        :($tupExpr[$ix])
      end
      #= Record-field subscript inside algorithmic code (Modelica function
         body). Mirrors the MTK arm in CodeGenerationUtil.jl. Algorithmic
         code never wraps values in Symbolics.Num, so we can use the same
         `_recordFieldRe` / `_recordFieldIm` helpers without a separate
         symbolic path. =#
      DAE.RSUB(exp = innerExp, ix = ix, fieldName = fname) => begin
        local innerJL = expToJuliaExpAlg(innerExp)
        if fname == "re"
          :(OMBackend.CodeGeneration._recordFieldRe($innerJL))
        elseif fname == "im"
          :(OMBackend.CodeGeneration._recordFieldIm($innerJL))
        else
          :(OMBackend.CodeGeneration._recordField($innerJL, $(QuoteNode(Symbol(fname))), $(_positionalFieldIndex(innerExp, ix))))
        end
      end
      DAE.BOX(exp = innerExp) => expToJuliaExpAlg(innerExp)
      DAE.UNBOX(exp = innerExp) => expToJuliaExpAlg(innerExp)
      _ => CodeGeneration.unsupported("expression", exp)
    end
  end
  return expr
end

"""
  Converts a DAE var to an equivalent Julia repr.
  Simple for now.
"""
function DAE_VAR_ToJulia(v::DAE.VAR)
  local vName = string(v.componentRef)
  Symbol(vName)
end

"""
  Resolves external "C" calls to concrete OMRuntimeExternalC function objects.
  This avoids scoping issues: the function object is captured directly in the
  generated closure rather than relying on OMRuntimeExternalC being in scope
  at runtime.
"""
Base.@nospecializeinfer function namespaceifyExternalFunction(@nospecialize(expr::Expr))
  #= Meta.parse may wrap in :toplevel -- unwrap it =#
  if expr.head == :toplevel && length(expr.args) == 1 && expr.args[1] isa Expr
    expr = expr.args[1]
  end
  res = if expr.head == :(=)
    local callExpr = last(expr.args)
    @match Expr(:call, [funcName, y...,z]) = callExpr
    local resolvedFunc = getfield(OMRuntimeExternalC, funcName)
    exp = Expr(:call, resolvedFunc, y..., z)
    expr.args[2] = exp
    expr
  else #Otherwise a side effect call or a call that returns directly.
    @assert expr.head === :call "Invalid call passed to namespaceifyExternalFunction"
    @match Expr(:call, [funcName, y...,z]) = expr
    local resolvedFunc = getfield(OMRuntimeExternalC, funcName)
    Expr(:call, resolvedFunc, y..., z)
  end
  return res
end
