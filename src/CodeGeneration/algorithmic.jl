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

_algAssignedMaxIndex(idx::Integer) = Int(idx)
_algAssignedMaxIndex(idx::AbstractUnitRange) = isempty(idx) ? 0 : Int(last(idx))
_algAssignedMaxIndex(idx::Colon) = 0
function _algAssignedMaxIndex(idx)
  try
    return maximum(Int, idx; init=0)
  catch
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

function _algAssignment(@nospecialize(lhsExp), rhs::Expr)
  local lhs = _unwrapSubscriptExpr(expToJuliaExpAlg(lhsExp))
  local prealloc = _algAssignmentPreallocation(lhsExp)
  if prealloc === nothing
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
  try
    dimCount = 0
    for _ in v.dims
      dimCount += 1
    end
    return dimCount >= 2
  catch
    return false
  end
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

#= Check if a function body contains conditionals that would fail with symbolic args.
   This includes STMT_IF statements and IFEXP (ternary if-expressions) in assignment RHS.
   Only these functions need the symbolic Term dispatch; pure-arithmetic functions
   work fine when called directly with symbolic values.
   Checks recursively inside for-loops and other compound statements. =#
function hasIfStatements(func::SimulationCode.ModelicaFunction)::Bool
  if !(func isa SimulationCode.MODELICA_FUNCTION)
    return false
  end
  local stmts = func.statements
  local locals = func.locals
  _statementsContainIf(stmts) && return true
  for v in locals
    @match v.binding begin
      SOME(bindingExp) => begin
        _expContainsIfExp(bindingExp) && return true
      end
      _ => nothing
    end
  end
  return false
end

function _statementsContainIf(stmts)::Bool
  for s in stmts
    if s isa DAE.STMT_IF
      return true
    elseif s isa DAE.STMT_FOR
      _statementsContainIf(s.statementLst) && return true
    elseif s isa DAE.STMT_WHILE
      _statementsContainIf(s.statementLst) && return true
    elseif s isa DAE.STMT_ASSIGN
      _expContainsIfExp(s.exp) && return true
    elseif s isa DAE.STMT_ASSIGN_ARR
      _expContainsIfExp(s.exp) && return true
    end
  end
  return false
end

Base.@nospecializeinfer function _expContainsIfExp(@nospecialize(exp::DAE.Exp))::Bool
  @match exp begin
    DAE.IFEXP(__) => return true
    DAE.BINARY(exp1 = e1, exp2 = e2) => begin
      return _expContainsIfExp(e1) || _expContainsIfExp(e2)
    end
    DAE.UNARY(exp = e1) => begin
      return _expContainsIfExp(e1)
    end
    DAE.CALL(expLst = args) => begin
      for a in args
        _expContainsIfExp(a) && return true
      end
      return false
    end
    DAE.ARRAY(array = elems) => begin
      for e in elems
        _expContainsIfExp(e) && return true
      end
      return false
    end
    DAE.ASUB(exp = inner) => begin
      return _expContainsIfExp(inner)
    end
    _ => return false
  end
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
        local arrayConversions = generateArrayConversions(func.inputs)
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
        local funcBody = Expr(:block, arrayConversions..., outputDefaults..., locals..., statements..., :(return $(returnExpr)))
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
        local extCall = Meta.parse(func.libInfo)
        extCall = namespaceifyExternalFunction(extCall)
        #= Allocate ccall-mutable buffers for every output, convert array inputs
           to the right C element type, then dereference Refs in the return. =#
        local extInputConversions = generateExternalInputConversions(func.inputs)
        local extOutputAllocs = generateExternalOutputAllocations(func.outputs)
        local returnExpr = generateExternalReturnExpr(func.outputs)

        #= Build the anonymous function expression manually =#
        local funcBody = Expr(:block, extInputConversions..., extOutputAllocs..., extCall, returnExpr)
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

"""
`generateSignatureForRegistration(inputs::Vector{DAE.VAR})`
This function generates the input signature for calls to Symbolics.register.
Record inputs are flattened into individual field parameters to match the
generated function signature from generateIOL/flattenRecordInput.
"""
function generateSignatureForRegistration(inputs::Vector{DAE.VAR})
  local jInputs = Expr[]
  for i in inputs
    @match i.ty begin
      DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), _, _) => begin
        #= Flatten record into individual field parameters =#
        local flattenedSymbols = flattenRecordInput(i)
        for s in flattenedSymbols
          push!(jInputs, Expr(:(::), s, :(Any)))
        end
      end
      _ => begin
        local s = DAE_VAR_ToJulia(i)
        push!(jInputs, Expr(:(::), s, :(Any)))
      end
    end
  end
  return jInputs
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
        local bindExpr = expToJuliaExpAlg(bindingExp)
        push!(jInputs, :(local $s = $bindExpr))
        true
      end
      _ => false
    end
    if !hasBinding
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

"""
  Generate default-initialized local declarations for Modelica function output variables.
  In Modelica, output variables are implicitly initialized (Real=0.0, Integer=0, Bool=false).
  Without this, variables assigned only in one if-branch are undefined in other branches.
"""
function generateOutputDefaults(outputs::Vector)::Vector{Expr}
  local decls = Expr[]
  for v in outputs
    local s = DAE_VAR_ToJulia(v)
    local defaultVal = if _funcParamIsArray(v)
      local jlElemDefault = @match _funcParamElemType(v) begin
        DAE.T_REAL(__) => :Float64
        DAE.T_INTEGER(__) => :Int
        DAE.T_BOOL(__) => :Bool
        _ => :Float64
      end
      local dimExprs = map(_daeDimToJulia, collect(_funcParamDims(v)))
      local unresolved = any(d -> d isa Number && d <= 0, dimExprs)
      if !isempty(dimExprs) && !unresolved
        :(zeros($(jlElemDefault), $(dimExprs...)))
      else
        :($(jlElemDefault)[])
      end
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

_daeDimToJulia(d) = @match d begin
  DAE.DIM_INTEGER(int) => int
  DAE.DIM_EXP(exp) => expToJuliaExpAlg(exp)
  _ => 0
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

function generateExternalInputConversions(inputs::Vector)::Vector{Expr}
  local conversions = Expr[]
  for v in inputs
    local s = DAE_VAR_ToJulia(v)
    if _funcParamIsArray(v)
      local jlElemType = _ccallElemType(_funcParamElemType(v))
      local nDims = length(collect(_funcParamDims(v)))
      local containerTy = nDims >= 2 ? :(Matrix{$jlElemType}) : :(Vector{$jlElemType})
      push!(conversions, :($s = convert($containerTy, $s)))
    end
  end
  return conversions
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
  throw("Unsupported stmt:" * string(s))
end

function generateStatement(stmt::DAE.STMT_NORETCALL)
  return expToJuliaExpAlg(stmt.exp)
end

function generateStatement(stmt::DAE.STMT_ASSIGN)
  local rhs = expToJuliaExpAlg(stmt.exp)
  return _algAssignment(stmt.exp1, rhs)
end

function generateStatement(stmt::DAE.STMT_TUPLE_ASSIGN)
  #= Emit a proper Julia tuple destructuring `(a, _, b) = rhs`.
     The old code did `Symbol(string(expExpLst))`, producing a single
     identifier like `var"(a, _, b)"` that is never actually bound —
     silently wrong on multi-return function calls. Mapping each CREF
     to its identifier gives a real tuple-assign. =#
  local lhsSyms = map(_crefToTupleTarget, collect(stmt.expExpLst))
  local rhs = expToJuliaExpAlg(stmt.exp)
  return Expr(:(=), Expr(:tuple, lhsSyms...), rhs)
end

#= Translate a single target expression inside a tuple-assign LHS to a
   Julia symbol suitable for `Expr(:tuple, …)`. =#
Base.@nospecializeinfer function _crefToTupleTarget(@nospecialize(exp::DAE.Exp))
  @match exp begin
    DAE.CREF(DAE.WILD(), _) => :_
    DAE.CREF(DAE.CREF_IDENT(ident, _, _), _) => Symbol(ident)
    DAE.CREF(cr, _) => Symbol(SimulationCode.string(cr))
    _ => Symbol(string(exp))
  end
end

function generateStatement(stmt::DAE.STMT_ASSIGN_ARR)
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
  local cond = expToJuliaExpAlg(stmt.exp)
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
  local cond = expToJuliaExpAlg(stmt.exp)
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

"""
  Generates Julia code for Modelica assert statements.
  If level is error (index 1), throws an error when condition is false.
  If level is warning (index 2), prints a warning when condition is false.
"""
function generateStatement(stmt::DAE.STMT_ASSERT)::Expr
  local condExpr = expToJuliaExpAlg(stmt.cond)
  local msgExpr = expToJuliaExpAlg(stmt.msg)
  #= Check assertion level: error (1) or warning (2) =#
  local level = @match stmt.level begin
    DAE.ENUM_LITERAL(_, idx) => idx
    _ => 1  #= Default to error =#
  end
  if level == 1
    #= AssertionLevel.error - throw an error =#
    quote
      if !($condExpr)
        error($msgExpr)
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

"""
  Maps a DAE expression to a Julia expression for algorithmic code in Modelica Functions(!).
  Since functions do not use the model HT the original name is preserved for algorithmic generation.
For algorithmic code outside Modelica functions do not call this function.
"""
#= SimCode-Exp entry: codegen consumes `SimulationCode.Exp` (Phase 4b API). =#
Base.@nospecializeinfer function expToJuliaExpAlg(@nospecialize(exp::SimulationCode.Exp))::Expr
  return expToJuliaExpAlg(SimulationCode.toDAEExp(exp))
end

function expToJuliaExpAlg(@nospecialize(exp::DAE.Exp))::Expr
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
        local op = CodeGeneration.DAE_OP_toJuliaOperator(op)
        :($op($(operand)))
      end
      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpAlg(e1)
        local rhs = expToJuliaExpAlg(e2)
        #= || and && are special forms in Julia, not regular functions.
           Must use Expr(:||, ...) / Expr(:&&, ...) instead of Expr(:call, :||, ...) =#
        @match op begin
          DAE.OR(__) => Expr(:||, lhs, rhs)
          DAE.AND(__) => Expr(:&&, lhs, rhs)
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
        local cond = expToJuliaExpAlg(e1)
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
        local args = map(explst) do arg
          expToJuliaExpAlg(arg)
        end
        append!(expr.args, args)
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
        local args = map(expLst) do arg
          expToJuliaExpAlg(arg)
        end
        append!(expr.args, args)
        expr
      end
      DAE.CAST(ty, exp)  => begin
        #= Type cast expression =#
        local innerExpr = expToJuliaExpAlg(exp)
        @match ty begin
          DAE.T_REAL(__) => :(float($innerExpr))
          DAE.T_INTEGER(__) => :(Int(round($innerExpr)))
          DAE.T_BOOL(__) => :(Bool($innerExpr))
          _ => innerExpr  #= For other types, just return the inner expression =#
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
        local arrExpr = expToJuliaExpAlg(arrExp)
        local subs = map(_algIndexExpr, subLst)
        Expr(:ref, arrExpr, subs...)
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
              local rangeExpr = expToJuliaExpAlg(rangeExp)
              push!(iterExprs, Expr(:(=), Symbol(id), rangeExpr))
            end
          end
        end
        @match reductionInfo.path begin
          Absyn.IDENT("array") => Expr(:comprehension, bodyExpr, iterExprs...)
          Absyn.IDENT("sum") => :(sum($(Expr(:generator, bodyExpr, iterExprs...))))
          Absyn.IDENT("product") => :(prod($(Expr(:generator, bodyExpr, iterExprs...))))
          Absyn.IDENT("min") => :(minimum($(Expr(:generator, bodyExpr, iterExprs...))))
          Absyn.IDENT("max") => :(maximum($(Expr(:generator, bodyExpr, iterExprs...))))
          _ => Expr(:comprehension, bodyExpr, iterExprs...)
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
      DAE.RSUB(exp = innerExp, fieldName = fname) => begin
        local innerJL = expToJuliaExpAlg(innerExp)
        if fname == "re"
          :(OMBackend.CodeGeneration._recordFieldRe($innerJL))
        elseif fname == "im"
          :(OMBackend.CodeGeneration._recordFieldIm($innerJL))
        else
          :(getproperty($innerJL, $(QuoteNode(Symbol(fname)))))
        end
      end
      DAE.BOX(exp = innerExp) => expToJuliaExpAlg(innerExp)
      DAE.UNBOX(exp = innerExp) => expToJuliaExpAlg(innerExp)
      _ =>  throw(ErrorException("$exp not yet supported"))
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
function namespaceifyExternalFunction(expr::Expr)
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
