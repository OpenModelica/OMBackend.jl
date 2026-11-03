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
Code generation for algorithmic Modelica: the model's functions (SimCode.ModelicaFunction) as
Julia functions, their statements and expressions.
author:johti17
=#

const SC = SimCode

#= The return expression of the function being generated (a `return` statement's). =#
const _CURRENT_RETURN_EXPR = Ref{Any}(nothing)

"""
A single-valued block as its value: `quote \$int end` is `Expr(:block, int)`, which inside
`Expr(:ref, ...)` prints as `a[(1;)]`, invalid Julia.
"""
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

"A subscript as a Julia index (`:` as itself)."
_algIndexExpr(s::SC.Exp) = s isa SC.WHOLE_DIMENSION ? :(:) : _unwrapSubscriptExpr(expToJuliaExpAlg(s))

"""
    _algAssignmentPreallocation(lhs)

The growth of the array an assignment to an element by one subscript writes (`x[i] := v`,
`x` a vector the body fills: `[:]` starts empty); `nothing` for another target.
"""
function _algAssignmentPreallocation(lhs::SC.Exp)
  local (name, index) = if lhs isa SC.EXP_CREF && length(lhs.cref.subs) == 1
    (lhs.cref.sym, only(lhs.cref.subs))
  elseif lhs isa SC.ASUB && lhs.exp isa SC.EXP_CREF && isempty(lhs.exp.cref.subs) && length(lhs.subs) == 1
    (lhs.exp.cref.sym, _algIndexExpr(only(lhs.subs)))
  else
    return nothing
  end
  return :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.ensureAlgArrayLength!($(name), $index))
end

"""
The lowered left-hand side indexes the array `name`: a scalarized element
(signalPWM[3].sawtooth.count lowered to one name) has no array to grow.
"""
_indexesArray(lhs, name) = lhs isa Expr && lhs.head === :ref && _unwrapSubscriptExpr(lhs.args[1]) === name

"""
A range (`a:b`, `a:s:b`).
"""
_isRangeExpr(@nospecialize(x)) = x isa Expr && x.head === :call && !isempty(x.args) && x.args[1] === :(:)

function _algAssignment(lhsExp::SC.Exp, rhs::Expr)
  local lhs = _unwrapSubscriptExpr(expToJuliaExpAlg(lhsExp))
  #= A range assigned: a Vector, its elements assigned after (MSL Vectors.sort's
     `indices := 1:size(v, 1)`: setindex! on a UnitRange; Buildings' SignalRanker). =#
  _isRangeExpr(rhs) && (rhs = :(collect($rhs)))
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

#= ==================================================================================== =#
#= Function variables                                                                   =#
#= ==================================================================================== =#

"A function variable as a Julia name."
_varSymbol(v::SC.FunctionVariable)::Symbol = Symbol(v.name)

"Whether a function variable is an array."
_funcParamIsArray(v::SC.FunctionVariable)::Bool = SC.isArrayVariable(v)

"A function variable's element type (of an array, its elements')."
_funcParamElemType(v::SC.FunctionVariable)::SC.SType = SC.scalarType(v.ty)

"The number of a function variable's dimensions."
_funcParamRank(v::SC.FunctionVariable)::Int = max(length(v.dimensions), v.ty isa SC.TYPE_ARRAY ? length(v.ty.dims) : 0)

"""
A multi-dimensional array (2 dimensions or more).
"""
isMultiDimArray(v::SC.FunctionVariable)::Bool = _funcParamRank(v) >= 2

"""
An array output: the wrapper must run the body (its result is indexed).
"""
hasArrayOutput(f::SC.ModelicaFunction)::Bool = any(_funcParamIsArray, f.outputs)

"""
The dimensions of the one array output of known sizes, `(4,)`, `(3, 3)`; `()` otherwise.
"""
function computeArrayOutputDims(f::SC.ModelicaFunction)::Tuple{Vararg{Int}}
  length(f.outputs) == 1 || return ()
  local t = only(f.outputs).ty
  (t isa SC.TYPE_ARRAY && all(d -> d >= 0, t.dims)) || return ()
  return Tuple(t.dims)
end

"""
The size of a dimension of a function variable: a literal, or an expression of the inputs;
`[:]` (0) starts empty and grows as it is assigned (ensureAlgArrayLength!).
"""
_daeDimToJulia(d::SC.Exp) = d isa SC.ICONST ? d.value : _unwrapSubscriptExpr(expToJuliaExpAlg(d))

"""
An Integer or enumeration input is an Int in the function, an array of them an Int array:
read from the integrator (a discrete's value) it arrives as a Float64 (MSL TimeTable's
`last`, a table index in getInterpolationCoefficients: "invalid index: 2.0"; Digital's
Logic values indexing its tables in RAM: "invalid index: 1.0"). A symbolic argument is
left as it is.
"""
function generateIntegerInputConversions(inputs::Vector{SC.FunctionVariable})::Vector{Expr}
  local conversions = Expr[]
  for v in inputs
    _funcParamElemType(v) isa Union{SC.TYPE_INTEGER, SC.TYPE_ENUM} || continue
    local s = _varSymbol(v)
    push!(conversions, :($s = OMBackend.CodeGeneration.AlgorithmicCodeGeneration._integerInput($s)))
  end
  return conversions
end
"""
An Integer input's value as an Int: exact, or within rounding of an integer (a discrete read
at an event the integrator located inside a step is interpolated there, a constant to
5.000000000000001: MSL TimeTable's `last` at HBridge_DC_Drive's duty-cycle event, 4.1.0).
Another value as it is.
"""
function _integerInput(x::AbstractFloat)
  local r = round(x)
  return isfinite(x) && abs(x - r) <= 1.0e3 * eps(max(1.0, abs(x))) ? Int(r) : x
end
_integerInput(x::AbstractArray{<:AbstractFloat}) = all(v -> _integerInput(v) isa Int, x) ? Int.(round.(x)) : x
_integerInput(x) = x

"""
The multi-dimensional array inputs as proper arrays (ensureArray).
"""
function generateArrayConversions(inputs::Vector{SC.FunctionVariable})::Vector{Expr}
  local conversions = Expr[]
  for v in inputs
    isMultiDimArray(v) || continue
    local s = _varSymbol(v)
    push!(conversions, :($s = OMBackend.CodeGeneration.ensureArray($s)))
  end
  return conversions
end

"""
A record-typed variable's fields (an array of records': its elements') as their flattened
symbols (`_fieldSymbol` of its reference: `<name>_<field>`); empty for another variable.
"""
function flattenRecordInput(v::SC.FunctionVariable)::Vector{Symbol}
  local fields = _recordFieldNames(_funcParamElemType(v))
  return Symbol[_fieldSymbol(v.cref, f) for f in fields]
end

"The Julia names of the inputs (or outputs): a record's as its fields."
function generateIOL(variables::Vector{SC.FunctionVariable})::Vector{Symbol}
  local names = Symbol[]
  for v in variables
    local flat = flattenRecordInput(v)
    isempty(flat) ? push!(names, _varSymbol(v)) : append!(names, flat)
  end
  return names
end

"The default of a scalar of type `t` (`nothing` for none)."
function _scalarDefault(t::SC.SType; enumeration = 1, other = nothing)
  t isa SC.TYPE_REAL && return 0.0
  t isa SC.TYPE_INTEGER && return 0
  t isa SC.TYPE_BOOL && return false
  t isa SC.TYPE_STRING && return ""
  t isa SC.TYPE_ENUM && return enumeration
  return other
end

"""
The locals: from their defaults (an array as a copy: `Real Awork[n, n] = A` is written into,
`A` not), an array zero-filled, a scalar its type's default; a record as its fields'.
"""
function generateLocals(locals::Vector{SC.FunctionVariable})::Vector{Expr}
  local decls = Expr[]
  for v in locals
    local flat = flattenRecordInput(v)
    if !isempty(flat)
      append!(decls, _recordLocal(v, flat))
      continue
    end
    local s = _varSymbol(v)
    if v.default !== nothing
      local bindExpr = expToJuliaExpAlg(v.default)
      push!(decls, :(local $s = $(_funcParamIsArray(v) ? :(collect($bindExpr)) : bindExpr)))
    elseif _funcParamIsArray(v)
      #= `Real invMMX[size(X, 1)]` (MSL Media massToMoleFractions): an array, not the
         scalar default of its element type. =#
      push!(decls, :(local $s = $(_arrayDefault(v))))
    else
      local defaultVal = _scalarDefault(v.ty)
      push!(decls, defaultVal === nothing ? Expr(:local, s) : :(local $s = $defaultVal))
    end
  end
  return decls
end

"""
A record-typed local as its per-field locals (`flat`), set as omc's C code sets a record
variable: from its default (`Sat s = Sat(p, 0)`), else each field from its own default
(the instance's modifiers, the record type's: `SaturationProperties sat(psat = p, Tsat =
0)`, MSL R134a's getPhase_ph read sat_psat unset). A field without one, or of a record type,
stays unset, and so does an array of records' (`_recordArrayLocal`).
"""
function _recordLocal(v::SC.FunctionVariable, flat::Vector{Symbol})::Vector{Expr}
  _funcParamIsArray(v) && return _recordArrayLocal(v, flat)
  local decls = Expr[Expr(:local, fs) for fs in flat]
  if v.default !== nothing
    local assign = _recordAssignment(SC.EXP_CREF(SC.SimCref(Symbol(v.name), Int[], v.cref), v.ty), v.default)
    assign === nothing || push!(decls, assign)
    return decls
  end
  local defaults = Dict{String, SC.Exp}(v.fieldDefaults)
  local t = v.ty::SC.TYPE_COMPLEX
  for (fs, name, ft) in zip(flat, t.fieldNames, t.fieldTypes)
    (haskey(defaults, name) && !(ft isa SC.TYPE_COMPLEX)) || continue
    push!(decls, :($fs = $(expToJuliaExpAlg(defaults[name]))))
  end
  return decls
end

"""
An array of records as a local (MSL ReferenceAir's Ice09 Gibbs function: `MyComplex[3]
r_2 = {MyComplex(...), ...}`, `MyComplex o[12]`): each field an array of the local's
dimensions (r_2_re, r_2_im), from a default by the records' fields (record literals, or
record-valued calls each evaluated once: `Complex[2] t = {Complex(...), Complex(...)}`,
Complex's constructor; zero-filled, `2 t[1] log(t[1])` was NaN and so was the ice's entropy),
else zero-filled for the elements the body assigns. A field of a record or array type, or
dimensions not known here, leave it declared only.
"""
function _recordArrayLocal(v::SC.FunctionVariable, flat::Vector{Symbol})::Vector{Expr}
  local decls = Expr[Expr(:local, fs) for fs in flat]
  local elemTy = _funcParamElemType(v)
  elemTy isa SC.TYPE_COMPLEX || return decls
  local fieldTys = elemTy.fieldTypes
  length(fieldTys) == length(flat) || return decls
  all(t -> t isa Union{SC.TYPE_REAL, SC.TYPE_INTEGER, SC.TYPE_BOOL}, fieldTys) || return decls
  local dimExprs = map(_daeDimToJulia, v.dimensions)
  (isempty(dimExprs) || any(d -> d isa Number && d <= 0, dimExprs)) && return decls
  local jlTy = t -> t isa SC.TYPE_INTEGER ? :Int : t isa SC.TYPE_BOOL ? :Bool : :Float64
  local elements = v.default isa SC.ARRAY_EXP ? _recordArrayElementTuples(v.default.elements, length(flat)) : nothing
  if elements === nothing
    for (k, fs) in enumerate(flat)
      push!(decls, :($fs = zeros($(jlTy(fieldTys[k])), $(dimExprs...))))
    end
    return decls
  end
  local tuples = Symbol("__omjl_", v.name, "_elements")
  push!(decls, :(local $tuples = ($(elements...),)))
  for (k, fs) in enumerate(flat)
    push!(decls, :($fs = $(jlTy(fieldTys[k]))[e[$k] for e in $tuples]))
  end
  return decls
end

"""
The elements of a one-dimensional array of records, each as its fields' tuple: a record
literal's fields lowered, a record-valued call's result (`_recordTupleExpr`); nothing for
another element.
"""
function _recordArrayElementTuples(elements::Vector{SC.Exp}, nFields::Int)::Union{Nothing, Vector{Any}}
  local out = Any[]
  for e in elements
    if e isa SC.RECORD && length(e.exps) == nFields
      push!(out, Expr(:tuple, Any[expToJuliaExpAlg(x) for x in e.exps]...))
    elseif e isa SC.CALL && e.ty isa SC.TYPE_COMPLEX
      push!(out, _recordTupleExpr(e))
    else
      return nothing
    end
  end
  return out
end

"""
A zero-filled array for an array-typed function variable, its dimensions possibly
depending on the inputs; an empty one where they are not known.
"""
function _arrayDefault(v::SC.FunctionVariable)
  local t = _funcParamElemType(v)
  local jlElemDefault = t isa SC.TYPE_INTEGER ? :Int : t isa SC.TYPE_BOOL ? :Bool : :Float64
  local dimExprs = map(_daeDimToJulia, v.dimensions)
  local unresolved = any(d -> d isa Number && d <= 0, dimExprs)
  return !isempty(dimExprs) && !unresolved ? :(zeros($(jlElemDefault), $(dimExprs...))) : :($(jlElemDefault)[])
end

"""
  The outputs' declarations: Modelica initializes an output (Real 0.0, Integer 0, Boolean
  false; a variable assigned in one if-branch only is defined in the others), an output with
  a default starts at it (an array as a copy: the body or a FORTRAN 77 routine writes into it,
  not into the default; a Vector: a range's copy was a range, MSL Vectors.sort's `indices =
  1:size(v, 1)`).
"""
function generateOutputDefaults(outputs::Vector{SC.FunctionVariable})::Vector{Expr}
  local decls = Expr[]
  for v in outputs
    local s = _varSymbol(v)
    if v.default !== nothing
      local bindExpr = expToJuliaExpAlg(v.default)
      push!(decls, :(local $s = $(_funcParamIsArray(v) ? :(collect($bindExpr)) : bindExpr)))
      continue
    end
    local defaultVal = _funcParamIsArray(v) ? _arrayDefault(v) : _scalarDefault(v.ty; enumeration = 0, other = 0)
    push!(decls, :(local $s = $defaultVal))
  end
  return decls
end

#= ==================================================================================== =#
#= Functions                                                                            =#
#= ==================================================================================== =#

"""
  Generates algorithmic Modelica Code.
  Returns the generated Julia code + the names of the functions that has been generated.

  To avoid world-age issues when these functions are called from MTK's RuntimeGeneratedFunctions,
  each is an implementation stored in MODELICA_FUNCTION_IMPLS and a wrapper created at module
  load time that looks it up (createModelicaFunctionWrapper).
"""
function generateFunctions(functions::Vector{SC.ModelicaFunction};
                           compiled::Bool = CodeGeneration.OMBackend.envSwitch("OMBACKEND_COMPILED_FUNCTIONS"))::Tuple{Vector{Expr}, Vector{String}}
  local jFuncs = Expr[]
  local names = String[]
  local codes = Tuple{SC.ModelicaFunction, Int, Bool, Tuple{Vararg{Int}}, Expr, Union{Expr, Nothing}}[]
  for func in functions
    local inputs = generateIOL(func.inputs)
    local outputs = generateIOL(func.outputs)
    local nArgs = length(inputs)
    local isArrayFunc = hasArrayOutput(func)
    #= Scalar element-extraction wrappers for all array-returning functions: called with
       symbolic arguments, an array function gives an array literal node that the index
       reduction cannot differentiate; a scalar wrapper a term it can. =#
    local outputDims = isArrayFunc ? computeArrayOutputDims(func) : ()
    local funcBody = if func isa SC.MODELICA_FUNCTION
      local inputConversions = vcat(generateArrayConversions(func.inputs), generateIntegerInputConversions(func.inputs))
      local returnExpr = length(outputs) > 1 ? Expr(:tuple, outputs...) : length(outputs) == 1 ? outputs[1] : nothing
      _CURRENT_RETURN_EXPR[] = returnExpr
      _directCalls!(Expr(:block, inputConversions..., generateOutputDefaults(func.outputs)..., generateLocals(func.locals)...,
                         generateStatements(func.statements)..., :(return $(returnExpr))))
    elseif func.external.language == "FORTRAN 77"
      _fortranExternalBody(func)
    else
      ensureExternalC!(func)
      #= ccall-mutable buffers for every output, the array inputs converted to their C element
         type, the protected locals the call passes (MSL realFFT_raw's work array); the Refs
         dereferenced in the return. =#
      local cArguments = _externalArgumentNames(func.external)
      Expr(:block, generateExternalInputConversions(func.inputs)..., generateExternalOutputAllocations(func.outputs, cArguments)...,
           generateLocals(func.locals)..., namespaceifyExternalFunction(_externalCallExpr(func.external)),
           generateExternalReturnExpr(func.outputs, cArguments))
    end
    local anonFunc = nArgs == 0 ? Expr(:->, Expr(:tuple), funcBody) :
      nArgs == 1 ? Expr(:->, inputs[1], funcBody) : Expr(:->, Expr(:tuple, inputs...), funcBody)
    #= compiled: CodeGeneration's functions (compiledFunction), not a closure of the model module,
       which runs interpreted =#
    local compilable = compiled && func isa SC.MODELICA_FUNCTION ? CodeGeneration._compilableFunction(anonFunc) : nothing
    push!(codes, (func, nArgs, isArrayFunc, outputDims, anonFunc, compilable))
  end
  local compiledNames = Set{Symbol}(Symbol(c[1].name) for c in codes if c[6] !== nothing)
  for (func, nArgs, isArrayFunc, outputDims, anonFunc, compilable) in codes
    #= a compiled function's calls of the compiled ones: their entries (compiledFunction) =#
    local impl = compilable === nothing ? anonFunc :
      :(OMBackend.CodeGeneration.compiledFunction($(QuoteNode(Symbol(func.name))), $(QuoteNode(compilable)),
                                                  $(Expr(:tuple, QuoteNode.(sort!(collect(intersect(_calledNames(compilable), compiledNames))))...)),
                                                  @__MODULE__, $(anonFunc)))
    local f = quote
      #= Create the wrapper function (always re-created to apply correct flags) =#
      OMBackend.CodeGeneration.createModelicaFunctionWrapper($(QuoteNode(Symbol(func.name))), $(nArgs), $(isArrayFunc), $(outputDims))
      #= Store the implementation in the dictionary =#
      OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[$(QuoteNode(Symbol(func.name)))] = $(impl)
      OMBackend.CodeGeneration.MODELICA_FUNCTION_OWNERS[$(QuoteNode(Symbol(func.name)))] = @__MODULE__
      $(_functionDerivativeRuleExpr(func, functions))
    end
    push!(jFuncs, f)
    push!(names, func.name)
  end
  return jFuncs, names
end

"""
The names `ex` calls (`name(...)`, the first argument of a call when it is a Symbol).
"""
function _calledNames(ex)::Set{Symbol}
  local found = Set{Symbol}()
  local worklist = Any[ex]
  local visited = Base.IdSet{Any}()
  while !isempty(worklist)
    local e = pop!(worklist)
    e isa Expr || continue
    e in visited && continue
    push!(visited, e)
    e.head === :call && !isempty(e.args) && e.args[1] isa Symbol && push!(found, e.args[1])
    append!(worklist, e.args)
  end
  return found
end

"""
A function body's calls of the model's functions made directly, in place:
`Base.invokelatest(name, args...)` (expToJuliaExpAlg's call, for code compiled before the
callee's wrapper exists) becomes `name(args...)`. A body runs in the latest world (every caller
enters it through `Base.invokelatest`: its wrapper, the call memo, the numeric partials), where
the callee's wrapper is defined. The hop splatted every argument into a new tuple: an MSL
Spice3 MOSFET evaluation (mosCalcNoBypassCode, 159 arguments) spent most of its time in the
hops to its 10 nested calls (drainCur's 139 arguments: 15 µs through the hop, 7 µs direct).
A worklist with identity dedup, as qualifyModelicaFunctions!: bodies nest deeply.
"""
function _directCalls!(body::Expr)::Expr
  local worklist = Any[body]
  local visited = Base.IdSet{Any}()
  while !isempty(worklist)
    local e = pop!(worklist)
    e isa Expr || continue
    e in visited && continue
    push!(visited, e)
    if e.head === :call && length(e.args) >= 2 && e.args[1] == :(Base.invokelatest) && e.args[2] isa Symbol
      popfirst!(e.args)
    end
    for a in e.args
      a isa Expr && push!(worklist, a)
    end
  end
  return body
end

"""
The derivative rule of `func`'s derivative annotation (CodeGeneration.FUNCTION_DERIVATIVE_RULES):
the derivative function and, in order, the inputs whose derivatives it takes (the Real ones
without zeroDerivative/noDerivative). Scalar inputs only: flattened records move the
positions.
"""
function _functionDerivativeRuleExpr(func::SC.ModelicaFunction, functions::Vector{SC.ModelicaFunction})
  local d = func.derivative
  d === nothing && return nothing
  length(func.inputs) == d.nInputs || return nothing
  local withDer = Int[]
  for (i, v) in enumerate(func.inputs)
    (_funcParamIsArray(v) || v.ty isa SC.TYPE_COMPLEX) && return nothing
    v.ty isa SC.TYPE_REAL && !(i in d.excludedInputs) && push!(withDer, i)
  end
  local k = findfirst(g -> g.name == d.name, functions)
  (k === nothing || length(functions[k].inputs) != d.nInputs + length(withDer)) && return nothing
  return :(OMBackend.CodeGeneration.FUNCTION_DERIVATIVE_RULES[$(QuoteNode(Symbol(func.name)))] =
           ($(QuoteNode(Symbol(d.name))), $(withDer)))
end

#=
EXTERNAL_MODELICA_FUNCTION code generation helpers.

A MODELICA_FUNCTION rebinds its variables by assignment, so `local result = 0.0` works because
the body's `result := expr` overwrites it. An external function cannot rebind from C: the C
function writes into a buffer reachable through a pointer. The ccall ABI wants:

- a scalar `Real` output: `Ref{Cdouble}`, allocated `Ref{Cdouble}(0.0)`, read back with `[]`;
- a scalar `Integer`/`Boolean` output: `Ref{Cint}` (a Modelica Integer is a C `int`);
- an array `Real[N]`: a `Vector{Cdouble}` of length N, returned as it is;
- an array `Integer[N]`: a `Vector{Cint}`.

An Integer array input is converted to a `Vector{Cint}` before the call: the caller may pass a
`Vector{Int64}`, and ccall does not widen the pointer cast.
=#

function _ccallElemType(t::SC.SType)
  t isa SC.TYPE_REAL && return :Cdouble
  t isa Union{SC.TYPE_INTEGER, SC.TYPE_BOOL} && return :Cint
  return :Float64
end

function generateExternalOutputAllocations(outputs::Vector{SC.FunctionVariable}, cArguments::Set{Symbol})::Vector{Expr}
  local decls = Expr[]
  for v in outputs
    local s = _varSymbol(v)
    local alloc = if _funcParamIsArray(v)
      local jlElemType = _ccallElemType(_funcParamElemType(v))
      local dimExprs = map(_daeDimToJulia, v.dimensions)
      local unresolved = any(d -> d isa Number && d <= 0, dimExprs)
      !isempty(dimExprs) && !unresolved ? :(zeros($(jlElemType), $(dimExprs...))) : :($(jlElemType)[])
    else
      #= a String output the call passes, a `const char**` the C function sets (ModelicaStrings_scanString):
         a "" could not be set, and no method took it (MSL readRealParameter, a MethodError); the call's
         result (ModelicaInternal_readLine's) is assigned =#
      v.ty isa SC.TYPE_REAL ? :(Ref{Cdouble}(0.0)) : v.ty isa Union{SC.TYPE_INTEGER, SC.TYPE_BOOL} ? :(Ref{Cint}(Cint(0))) :
        v.ty isa SC.TYPE_STRING ? (s in cArguments ? :(Ref{Cstring}(C_NULL)) : "") : 0
    end
    push!(decls, :(local $s = $alloc))
  end
  return decls
end

"""
A Modelica Integer (or Boolean) array for a C `int*`: rounded, since an
Integer held in the Float64 state vector can be an ulp off after a solver
step (the MSL noise generators' xorshift states, 1.3705433889999998e9 for
1370543389: `convert` threw InexactError in ActuatorWithNoise).
"""
_roundToCint(x::AbstractArray{Cint}) = x
_roundToCint(x::AbstractArray) = map(v -> v isa Integer ? Cint(v) : round(Cint, v), x)
_roundToCint(x) = x

function generateExternalInputConversions(inputs::Vector{SC.FunctionVariable})::Vector{Expr}
  local conversions = Expr[]
  for v in inputs
    _funcParamIsArray(v) || continue
    local s = _varSymbol(v)
    local jlElemType = _ccallElemType(_funcParamElemType(v))
    local containerTy = _funcParamRank(v) >= 2 ? :(Matrix{$jlElemType}) : :(Vector{$jlElemType})
    local value = jlElemType === :Cint ? :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration._roundToCint($s)) : s
    push!(conversions, :($s = convert($containerTy, $value)))
  end
  return conversions
end

"The variables an external call passes as its arguments (generateExternalOutputAllocations)."
_externalArgumentNames(x::SC.ExternalCall)::Set{Symbol} =
  Set{Symbol}(_externalArgument(a) for a in x.arguments if a isa SC.EXP_CREF)

"""
    _externalCallExpr(x)

The external clause's call as a Julia expression, `f(a, b)` or `y = f(a, b)`: its arguments
in the function's variables (`size(A, 1)`, literals) as the external declaration writes them.
"""
function _externalCallExpr(x::SC.ExternalCall)::Expr
  local call = Expr(:call, Symbol(x.name), Any[_unwrapSubscriptExpr(_externalArgument(a)) for a in x.arguments]...)
  return x.output === nothing ? call : Expr(:(=), _unwrapSubscriptExpr(_externalArgument(x.output)), call)
end

"""
An argument of an external call: a variable by its name, `size(a, i)`, a literal.
"""
_externalArgument(e::SC.EXP_CREF) = Symbol(SC.crefName(e.cref))
_externalArgument(e::SC.Exp) = expToJuliaExpAlg(e)

"""
The variables of an external function by their Julia names.
"""
_externalVariables(func::SC.EXTERNAL_MODELICA_FUNCTION)::Dict{Symbol, SC.FunctionVariable} =
  Dict{Symbol, SC.FunctionVariable}(_varSymbol(v) => v for v in Iterators.flatten((func.inputs, func.outputs, func.locals)))

"""
An external "FORTRAN 77" function (Modelica.Math.Matrices.LAPACK): its locals and outputs
initialized from their defaults (Awork = A, x = b), the routine of Julia's LAPACK called
with every argument by reference (arrays as fresh Float64/Int64 copies it may overwrite,
scalars in Refs read back after the call, a character with its hidden length last, as
gfortran passes it), and the outputs returned.
"""
function _fortranExternalBody(func::SC.EXTERNAL_MODELICA_FUNCTION)::Expr
  local call = _externalCallExpr(func.external)
  local (resultVar, callExpr) = call.head === :(=) ? (call.args[1], call.args[2]) : (nothing, call)
  local routine = callExpr.args[1]
  local vars = _externalVariables(func)
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
      local elem = _funcParamElemType(v)
      if elem isa SC.TYPE_STRING
        _funcParamIsArray(v) && OMBackend.unsupported("a FORTRAN 77 character array", a)
        push!(types, :(Ref{UInt8})); push!(args, :(UInt8(first($a))))
        push!(hiddenLengths, 1)
        continue
      end
      local jlType = elem isa SC.TYPE_REAL ? :Float64 : :Int64
      if _funcParamIsArray(v)
        haskey(refs, a) || (refs[a] = a; push!(prep, :($a = Array{$jlType}($a))))
        push!(types, :(Ptr{$jlType})); push!(args, a)
      else
        local r = get!(refs, a) do
          local r = Symbol("_ref_", a)
          push!(prep, :(local $r = Ref{$jlType}($a)))
          push!(readBack, elem isa SC.TYPE_BOOL ? :($a = $r[] != 0) :
                          elem isa SC.TYPE_REAL ? :($a = $r[]) : :($a = Int($r[])))
          r
        end
        push!(types, :(Ref{$jlType})); push!(args, r)
      end
    else
      OMBackend.unsupported("a FORTRAN 77 argument", a)
    end
  end
  append!(types, fill(:Clong, length(hiddenLengths))); append!(args, hiddenLengths)
  local resultType = resultVar === nothing ? :Cvoid : (_funcParamElemType(vars[resultVar]) isa SC.TYPE_REAL ? :Float64 : :Int64)
  local ccallExpr = Expr(:call, :ccall,
                         :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.lapackFunction($(QuoteNode(routine)))),
                         resultType, Expr(:tuple, types...), args...)
  local callStmt = resultVar === nothing ? ccallExpr : :($resultVar = $ccallExpr)
  local outputs = [_varSymbol(v) for v in func.outputs]
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

function generateExternalReturnExpr(outputs::Vector{SC.FunctionVariable}, cArguments::Set{Symbol})
  isempty(outputs) && return nothing
  local accessors = Any[]
  for v in outputs
    local s = _varSymbol(v)
    push!(accessors, _funcParamIsArray(v) ? s : v.ty isa SC.TYPE_REAL ? :($s[]) : v.ty isa SC.TYPE_INTEGER ? :(Int($s[])) :
                     v.ty isa SC.TYPE_BOOL ? :($s[] != 0) :
                     v.ty isa SC.TYPE_STRING && s in cArguments ? :($s[] == C_NULL ? "" : unsafe_string($s[])) : s)
  end
  return length(accessors) == 1 ? accessors[1] : Expr(:tuple, accessors...)
end

"""
An external function: OMRuntimeExternalC's, or one ensureExternalC! defined (externalC.jl),
read by invokelatest: defined in this world (an older world's read of a newer binding is
deprecated, an error in later Julia versions).
"""
_externalFunction(name::Symbol) =
  isdefined(externalCModule(), name) ? Base.invokelatest(getglobal, externalCModule(), name) :
                                       getfield(OMRuntimeExternalC, name)

"""
A C function's arguments: an external object as its pointer (CodeGeneration._externalPtr).
"""
_externalPtrArgs(args::AbstractVector)::Vector{Any} =
  Any[:(OMBackend.CodeGeneration._externalPtr($a)) for a in args]

"""
The external call with its function resolved to the function object (captured in the
generated closure: no name of OMRuntimeExternalC need be in scope where it runs) and its
arguments as pointers where they are external objects. Any number of arguments: MSL's
getPid has none (pid = ModelicaInternal_getpid()).
"""
function namespaceifyExternalFunction(expr::Expr)::Expr
  if expr.head === :(=)
    local callExpr = last(expr.args)
    return Expr(:(=), expr.args[1], Expr(:call, _externalFunction(callExpr.args[1]), _externalPtrArgs(callExpr.args[2:end])...))
  end
  expr.head === :call || error("namespaceifyExternalFunction: not a call: $(expr)")
  return Expr(:call, _externalFunction(expr.args[1]), _externalPtrArgs(expr.args[2:end])...)
end

#= ==================================================================================== =#
#= Statements                                                                           =#
#= ==================================================================================== =#

function generateStatements(statements::Vector{SC.Statement})::Vector{Expr}
  return Expr[generateStatement(s) for s in statements]
end

generateStatement(s::SC.Statement) = CodeGeneration.unsupported("statement", s)

generateStatement(stmt::SC.NORETCALL) = expToJuliaExpAlg(stmt.exp)

function generateStatement(stmt::SC.ASSIGN)
  stmt.left isa SC.TUPLE && return _tupleAssignment(stmt)
  local scalarised = _recordAssignment(stmt.left, stmt.right)
  scalarised === nothing || return scalarised
  return _algAssignment(stmt.left, _algValue(stmt.right))
end

"""
The value an assignment stores: an array variable read whole as a copy, as the defaults are
(Modelica assigns values). `PRea := PRea_new` bound both names to one array, and the writes
into PRea_new[m, k] changed PRea: Buildings' multipoleFluidTemperature saw no change after
its first iteration and stopped.
"""
function _algValue(exp::SC.Exp)::Expr
  local rhs = expToJuliaExpAlg(exp)
  return _isWholeArrayVariable(exp) ? :(collect($rhs)) : rhs
end

_isWholeArrayVariable(exp::SC.Exp)::Bool = exp isa SC.EXP_CREF && exp.ty isa SC.TYPE_ARRAY && isempty(exp.cref.subs)

"""
    _recordAssignment(lhsExp, rhsExp) -> Union{Nothing,Expr}

A record-typed assignment onto the flattened `<base>_<field>` symbols, or `nothing` when
`lhsExp` is no plain record reference (nor an element of an array of records). A record copy
becomes per-field assignments; a record-valued call, a record literal and an if-expression
choosing between records scatter their field tuple (a one-field record is its field). Another
value of a record is refused.
"""
function _recordAssignment(lhsExp::SC.Exp, rhsExp::SC.Exp)::Union{Nothing, Expr}
  local lhs = _recordCrefFields(lhsExp)
  #= An element of an array of records on either side (`r2 := r_2[1]`, `o[1] := f(...)`):
     field by field, each field array indexed alike. =#
  local lhsElement = lhs === nothing ? _recordElementFieldExps(lhsExp) : nothing
  if lhsElement !== nothing
    local targets = Any[expToJuliaExpAlg(f) for f in lhsElement]
    local rhsFields = _recordElementFieldExps(rhsExp)
    local rhsTuple = rhsFields !== nothing ? Expr(:tuple, Any[expToJuliaExpAlg(f) for f in rhsFields]...) : _recordTupleExpr(rhsExp)
    return length(targets) == 1 ? Expr(:(=), targets[1], rhsTuple) : Expr(:(=), Expr(:tuple, targets...), rhsTuple)
  end
  lhs === nothing && return nothing
  local (lhsRecord, fieldNames) = lhs
  local rhsElement = _recordElementFieldExps(rhsExp)
  if rhsElement !== nothing && length(rhsElement) == length(fieldNames)
    return Expr(:block, Expr[:($(_flatFieldSymbol(lhsRecord, f)) = $(expToJuliaExpAlg(e))) for (f, e) in zip(fieldNames, rhsElement)]...)
  end
  if _plainCrefName(rhsExp) !== nothing
    local rhsRecord::SC.SimCref = (rhsExp::SC.EXP_CREF).cref
    return Expr(:block, Expr[:($(_flatFieldSymbol(lhsRecord, f)) = $(_flatFieldSymbol(rhsRecord, f))) for f in fieldNames]...)
  elseif rhsExp isa SC.CALL || (rhsExp isa Union{SC.IFEXP, SC.RECORD} && _hasNoRecordFields(SC.typeOf(lhsExp)))
    #= A record literal, or an if-expression choosing between records, evaluates to the fields
       as a tuple too (MSL MixtureGasNasa setState_pTX: `state := if ... then
       ThermodynamicState(...) else ...`). =#
    local targets = [_flatFieldSymbol(lhsRecord, f) for f in fieldNames]
    local rhs = _recordTupleExpr(rhsExp)
    return length(targets) == 1 ? Expr(:(=), targets[1], rhs) : Expr(:(=), Expr(:tuple, targets...), rhs)
  end
  #= Assigned whole, the record's name would be bound and its fields, which are what is read,
     kept: an element of a tuple, a field or an element of another record. =#
  CodeGeneration.unsupported("a record assignment from this expression", rhsExp)
end

"""
    _recordElementFieldExps(exp)

The fields of an element of an array of records, `u[k]`, as `u_re[k]`, `u_im[k]` (the flattened
field arrays, indexed alike); `nothing` for another expression. Scalar fields only.
"""
function _recordElementFieldExps(exp::SC.Exp)::Union{Nothing, Vector{SC.Exp}}
  local ty::SC.SType = if exp isa SC.EXP_CREF && !isempty(exp.cref.subs)
    exp.ty
  elseif exp isa SC.ASUB && exp.exp isa SC.EXP_CREF && isempty(exp.exp.cref.subs)
    SC.typeOf(exp)
  else
    return nothing
  end
  local t::Union{Nothing, SC.TYPE_COMPLEX} = ty isa SC.TYPE_COMPLEX ? ty : nothing
  (t === nothing || !t.isRecord || isempty(t.fieldNames)) && return nothing
  all(ft -> !(ft isa SC.TYPE_ARRAY) && isempty(_recordFieldNames(ft)), t.fieldTypes) || return nothing
  #= the array of records: an element's reference without the element's subscripts (on its last part) =#
  local (array::SC.IR.ComponentRef, subs::Vector{Any}) = exp isa SC.EXP_CREF ?
    (SC.IR.withoutLastSubscripts(SC.irRef(exp.cref)), Any[s for s in exp.cref.subs]) :
    (SC.irRef(((exp::SC.ASUB).exp::SC.EXP_CREF).cref), Any[s for s in (exp::SC.ASUB).subs])
  return SC.Exp[_fieldElement(SC.fieldCref(array, n), subs, ft) for (n, ft) in zip(t.fieldNames, t.fieldTypes)]
end

"""
The element `subs` of the field array `field` (a record array's flattened field: `u.re`), by its identifier (`_identifier`) and its reference.
"""
function _fieldElement(field::SC.IR.ComponentRef, subs::Vector{Any}, ty::SC.SType)::SC.Exp
  local name::Symbol = Symbol(_identifier(field))
  if all(s -> s isa Int, subs)
    local index::Vector{Int} = Int[s for s in subs]
    return SC.EXP_CREF(SC.SimCref(name, index, SC.IR.elementCref(field, index)), ty)
  end
  return SC.ASUB(SC.EXP_CREF(SC.SimCref(name, Int[], field), SC.TYPE_ARRAY(ty, fill(-1, length(subs)))),
                 SC.Exp[s isa Int ? SC.ICONST(s) : s for s in subs])
end

"""
A record value as its field tuple: a record variable's flattened fields (it has no tuple of
its own), an element of an array of records its fields' elements, an if-expression's
branches each so, anything else as it lowers.
"""
function _recordTupleExpr(exp::SC.Exp)
  local rec = _recordCrefFields(exp)
  rec === nothing || return Expr(:tuple, [_flatFieldSymbol(rec[1], f) for f in rec[2]]...)
  local element = _recordElementFieldExps(exp)
  element === nothing || return Expr(:tuple, Any[expToJuliaExpAlg(f) for f in element]...)
  if exp isa SC.IFEXP
    return :(if $(expToJuliaExpAlg(exp.cond)) != 0
               $(_recordTupleExpr(exp.thenExp))
             else
               $(_recordTupleExpr(exp.elseExp))
             end)
  end
  return expToJuliaExpAlg(exp)
end

"""
The field names of a record type, in declaration order; empty for another type.
"""
_recordFieldNames(t::SC.SType)::Vector{String} = t isa SC.TYPE_COMPLEX && t.isRecord ? t.fieldNames : String[]

"""
The types of a record type's fields; empty for another type.
"""
_recordFieldTypes(t::SC.SType)::Vector{SC.SType} = t isa SC.TYPE_COMPLEX && t.isRecord ? t.fieldTypes : SC.SType[]

"""
Whether no field of the record type `t` is a record: its value's tuple is then field for
field (a nested record's fields are flattened in its place).
"""
function _hasNoRecordFields(t::SC.SType)::Bool
  local types = _recordFieldTypes(t)
  return !isempty(types) && all(ft -> isempty(_recordFieldNames(ft)), types)
end

"""
The flattened symbol of the field `field` of the record the reference `c` reads (a function's flattened field, a model
record's field variable): the field's reference's identifier (`_fieldSymbol`).
"""
_flatFieldSymbol(c::SC.SimCref, field::AbstractString)::Symbol = _fieldSymbol(SC.irRef(c), field)

"The identifier of the field `field` of the record reference `record` (`_identifier` of the field's reference: `r_f`, `u_re`)."
_fieldSymbol(record::SC.IR.ComponentRef, field::AbstractString)::Symbol = Symbol(_identifier(SC.fieldCref(record, field)))

"""
The code view (`SC.codeView`) of the model whose algorithm code is being generated (its initial algorithms and
initial when-statements, `inCodeView`); `nothing` for a function's body.
"""
const CODE_VIEW = Base.ScopedValues.ScopedValue{Union{Nothing, SC.SIM_CODE}}(nothing)

"""
The identifier of the reference `ref` in the code being generated: in a model's code view its variable's
(`SC.identifierOf`: a renumbered clash's too), in a function's body derived from it (`SC.derivedKey`, the function's
locals' names).
"""
function _identifier(ref::SC.IR.ComponentRef)::String
  local view::Union{Nothing, SC.SIM_CODE} = CODE_VIEW[]
  return view === nothing ? SC.derivedKey(ref) : SC.identifierOf(view, ref)
end

"`f()`, generating algorithm code of the model whose code view is `view` (`CODE_VIEW`)."
inCodeView(f::F, view::SC.SIM_CODE) where {F} = Base.ScopedValues.with(f, CODE_VIEW => view)

"""
    _algCallArgs(args; builtin = false, shared) -> Vector{Any}

A call's arguments: a record variable as its flattened field symbols, any other record value
splatted (its field tuple), so a record argument is passed as the callee's flattened inputs
take it; a builtin's arguments whole. The fields of a call read several times (TSUBs of one
call, equal ones: a call returning an array of records as one array per field) are the call
once, bound in `shared`.
"""
function _algCallArgs(argExps::Vector{SC.Exp}; builtin::Bool = false, shared::Vector{Expr} = Expr[])::Vector{Any}
  local out = Any[]
  local fieldReads = Dict{SC.Exp, Int}()
  for arg in argExps
    arg isa SC.TSUB && arg.exp isa SC.CALL && (fieldReads[arg.exp] = get(fieldReads, arg.exp, 0) + 1)
  end
  local boundCalls = Dict{SC.Exp, Symbol}()
  for arg in argExps
    if arg isa SC.TSUB && arg.exp isa SC.CALL && fieldReads[arg.exp] > 1
      local fields = get!(boundCalls, arg.exp) do
        local sym = gensym(:fields)
        push!(shared, :($sym = $(expToJuliaExpAlg(arg.exp))))
        sym
      end
      push!(out, :($fields[$(arg.index)]))
      continue
    end
    #= A record without fields (MSL Media's f_nonlinear_Data()) is no argument: the callee's
       flattened inputs have none for it. =#
    (!builtin && _isEmptyRecordValue(arg)) && continue
    #= An element of a local array of records, `r[k]`: its fields' elements, r_re[k], r_im[k]. =#
    local element = builtin ? nothing : _recordElementFieldExps(arg)
    if element !== nothing
      append!(out, Any[expToJuliaExpAlg(f) for f in element])
      continue
    end
    local rec = _recordCrefFields(arg)
    local valueTy = rec === nothing && !builtin ? _recordValueType(arg) : nothing
    if valueTy !== nothing && length(_recordFieldNames(valueTy)) > 1
      #= A record that is no named variable (a call's result, an element of a record array):
         its fields as a tuple, splatted (the MSL ReferenceAir's `rho_props_pT(p, T,
         airBaseProp_pT(p, T))`, the ideal gases' `h_T(data[i], T, ...)`). =#
      push!(out, Expr(:..., expToJuliaExpAlg(arg)))
    elseif rec === nothing
      push!(out, expToJuliaExpAlg(arg))
    else
      append!(out, [_flatFieldSymbol(rec[1], f) for f in rec[2]])
    end
  end
  return out
end

function _isEmptyRecordValue(exp::SC.Exp)::Bool
  exp isa Union{SC.CALL, SC.RECORD, SC.EXP_CREF} || return false
  local t = exp.ty
  return t isa SC.TYPE_COMPLEX && t.isRecord && isempty(t.fieldNames)
end

"""
The record type of `exp` where it is no named record variable: a call returning a record,
a record literal (a constant the frontend folded: the ideal gas `data`), an element of an
array of records (a scalar subscript per dimension); `nothing` otherwise. Such an
expression evaluates to the record's fields as a tuple.
"""
function _recordValueType(exp::SC.Exp)
  local t = if exp isa Union{SC.CALL, SC.RECORD, SC.EXP_CREF}
    exp.ty
  elseif exp isa SC.ASUB && exp.exp isa Union{SC.ARRAY_EXP, SC.EXP_CREF}
    local at = SC.typeOf(exp.exp)
    (at isa SC.TYPE_ARRAY && length(exp.subs) == length(at.dims) &&
     all(s -> !(s isa Union{SC.RANGE, SC.ARRAY_EXP, SC.WHOLE_DIMENSION}), exp.subs)) ? at.elementType : nothing
  else
    nothing
  end
  return t !== nothing && !isempty(_recordFieldNames(t)) ? t : nothing
end

"""
The position of field `ix` in the tuple a record-valued `exp` evaluates to, where every
field up to it is a scalar: a function returns a nested record's fields flattened in its
place, so a later field's position is not its index. -1 (read it by name) otherwise.
"""
function _positionalFieldIndex(exp::SC.Exp, ix::Integer)::Int
  local t = _recordValueType(exp)
  t === nothing && return -1
  local fieldTypes = _recordFieldTypes(t)
  1 <= ix <= length(fieldTypes) || return -1
  any(k -> !isempty(_recordFieldNames(fieldTypes[k])), 1:ix) && return -1
  return ix
end

"""
    _recordCrefFields(exp) -> Union{Nothing,Tuple{SimCref,Vector{String}}}

`(record, fieldNames)` of a reference without subscripts to a record or an array of records
(whose fields are then arrays): the reference and its type's field names; `nothing` otherwise.
"""
function _recordCrefFields(exp::SC.Exp)::Union{Nothing, Tuple{SC.SimCref, Vector{String}}}
  _plainCrefName(exp) === nothing && return nothing
  local fieldNames = _recordFieldNames(SC.scalarType(exp.ty))
  return isempty(fieldNames) ? nothing : ((exp::SC.EXP_CREF).cref, fieldNames)
end

"""
The flat field symbol `<base>_<kth field>` of a record read by one literal index (`r[k]`,
field access, not array indexing); `nothing` otherwise.
"""
function _recordIndexFieldSymbol(arrExp::SC.Exp, subs::Vector{SC.Exp})::Union{Nothing, Symbol}
  _plainCrefName(arrExp) === nothing && return nothing
  local fieldNames = _recordFieldNames(arrExp.ty)
  (isempty(fieldNames) || length(subs) != 1 || !(only(subs) isa SC.ICONST)) && return nothing
  local k = only(subs).value
  1 <= k <= length(fieldNames) || return nothing
  return _flatFieldSymbol((arrExp::SC.EXP_CREF).cref, fieldNames[k])
end

"The name of a reference without subscripts; `nothing` for another expression."
_plainCrefName(exp::SC.Exp)::Union{Nothing, String} =
  exp isa SC.EXP_CREF && isempty(exp.cref.subs) ? String(exp.cref.sym) : nothing

"""
`(a, , b) := f(...)` as a Julia destructuring `(a, _, b) = f(...)`. A generated function
returns a record output as its fields in place (generateIOL), so a record target takes them
into its flattened field symbols and an omitted record output skips one `_` per field. A
target that is no plain name (`v[2]`, `eigenvalues[:, 1]`, a record's field) takes its
element through a temporary and an ordinary assignment.
"""
function _tupleAssignment(stmt::SC.ASSIGN)::Expr
  local targetsIn = (stmt.left::SC.TUPLE).elements
  local rt = SC.typeOf(stmt.right)
  local outputTypes = rt isa SC.TYPE_TUPLE ? rt.types : nothing
  outputTypes === nothing || length(outputTypes) >= length(targetsIn) ||
    CodeGeneration.unsupported("a tuple assignment with more targets than outputs", stmt)
  local targets = Any[]
  local stores = Expr[]
  for (k, target) in enumerate(targetsIn)
    local fields = outputTypes === nothing ? String[] : _recordFieldNames(outputTypes[k])
    if !isempty(fields)
      _hasNoRecordFields(outputTypes[k]) ||
        CodeGeneration.unsupported("a tuple assignment of a record output with record fields", stmt)
      append!(targets, _recordTupleTargets(target, fields, stmt))
    elseif target isa SC.WILD
      push!(targets, :_)
    elseif !(target isa Union{SC.EXP_CREF, SC.ASUB}) || _recordCrefFields(target) !== nothing
      CodeGeneration.unsupported("this target of a tuple assignment", target)
    elseif _plainCrefName(target) !== nothing
      push!(targets, Symbol(_plainCrefName(target)))
    else
      local tmp = Symbol("#tupleTarget", k)  #= no Modelica name has a '#' =#
      push!(targets, tmp)
      push!(stores, _algAssignment(target, Expr(:block, tmp)))
    end
  end
  local assignment = Expr(:(=), Expr(:tuple, targets...), expToJuliaExpAlg(stmt.right))
  return isempty(stores) ? assignment : Expr(:block, assignment, stores...)
end

"""
The targets of a record output's fields: an omitted output's `_`s, a plain record name's
flattened field symbols.
"""
function _recordTupleTargets(target::SC.Exp, fields::Vector{String}, stmt::SC.ASSIGN)::Vector{Symbol}
  target isa SC.WILD && return fill(:_, length(fields))
  local rec = _recordCrefFields(target)
  (rec === nothing || rec[2] != fields) && CodeGeneration.unsupported("this record target of a tuple assignment", stmt)
  return Symbol[_flatFieldSymbol(rec[1], f) for f in fields]
end

function generateStatement(stmt::SC.WHILE)
  local cond = expToJuliaExpAlg(stmt.condition)
  local stmts = generateStatements(stmt.body)
  quote
    while ($(cond))
      $(stmts...)
    end
  end
end

function generateStatement(::SC.RETURN)::Expr
  local retExpr = _CURRENT_RETURN_EXPR[]
  return retExpr === nothing ? :(return) : :(return $(retExpr))
end

generateStatement(::SC.BREAK)::Expr = :(break)

"`for i in range loop body end for`."
function generateStatement(stmt::SC.FOR)::Expr
  return Expr(:for, Expr(:(=), Symbol(stmt.iterator.name), expToJuliaExpAlg(stmt.iterator.range)),
              Expr(:block, generateStatements(stmt.body)...))
end

"""
  `if ... elseif ... else ... end if`: each condition compared with 0 (a Boolean discrete lives
  numerically, 0.0/1.0, on the state vector, so a bare `if <float>` throws a TypeError; `!= 0`
  is idempotent on a Bool).
"""
function generateStatement(stmt::SC.IF)::Expr
  local tail = isempty(stmt.elseBody) ? nothing : Expr(:block, generateStatements(stmt.elseBody)...)
  for (k, b) in Iterators.reverse(enumerate(stmt.branches))
    local cond = :($(expToJuliaExpAlg(b.condition)) != 0)
    local blck = Expr(:block, generateStatements(b.body)...)
    local head = k == 1 ? :if : :elseif
    tail = tail === nothing ? Expr(head, cond, blck) : Expr(head, cond, blck, tail)
  end
  return tail
end

"""
AssertionLevel = enumeration(warning, error): warning is literal 1, so the level is matched
by name. Any other level expression means error.
"""
isWarningAssertionLevel(level::SC.Exp)::Bool = level isa SC.ENUM_LITERAL && endswith(String(level.name), "warning")

"""
  Modelica's assert: AssertionLevel.error (the default) throws an error when the condition is
  false, AssertionLevel.warning prints a warning.
"""
function generateStatement(stmt::SC.ASSERT)::Expr
  local condExpr = expToJuliaExpAlg(stmt.condition)
  local msgExpr = expToJuliaExpAlg(stmt.message)
  if !isWarningAssertionLevel(stmt.level)
    #= Base.error: a Modelica local may be called `error` (MSL Water IF97's `Integer error`
       flag of the inverse iterations: "objects of type Int64 are not callable"). =#
    quote
      if !($condExpr)
        Base.error($msgExpr)
      end
    end
  else
    quote
      if !($condExpr)
        @warn $msgExpr
      end
    end
  end
end

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

"""
    expToJuliaExpAlg(exp)

A SimCode expression in algorithmic code (a Modelica function's body) as Julia: functions do
not use the model's table, their variables keep their names.
"""
expToJuliaExpAlg(e::SC.BCONST)::Expr = quote $(e.value) end
expToJuliaExpAlg(e::SC.ICONST)::Expr = quote $(e.value) end
expToJuliaExpAlg(e::SC.RCONST)::Expr = quote $(e.value) end
expToJuliaExpAlg(e::SC.SCONST)::Expr = quote $(e.value) end
"""
An enumeration literal is its index.
"""
expToJuliaExpAlg(e::SC.ENUM_LITERAL)::Expr = quote $(e.index) end
expToJuliaExpAlg(::SC.WILD)::Expr = quote _ end
expToJuliaExpAlg(::SC.WHOLE_DIMENSION)::Expr = quote $(:(:)) end

function expToJuliaExpAlg(e::SC.EXP_CREF)::Expr
  isempty(e.cref.subs) && return quote $(e.cref.sym) end
  return Expr(:ref, e.cref.sym, e.cref.subs...)
end

"""
pre(x) in an (initial) algorithm body is the sequentially seeded held value: the argument,
as the MTK expression path does.
"""
expToJuliaExpAlg(e::SC.PRE)::Expr = expToJuliaExpAlg(SC.EXP_CREF(e.cref, e.ty))

expToJuliaExpAlg(e::SC.UNARY)::Expr = :($(CodeGeneration.opKindToJuliaOperator(e.op))($(expToJuliaExpAlg(e.exp))))

function expToJuliaExpAlg(e::SC.BINARY)::Expr
  local (lhs, rhs) = (expToJuliaExpAlg(e.exp1), expToJuliaExpAlg(e.exp2))
  e.op === SC.OP_SCALAR_PRODUCT && return :(OMBackend.CodeGeneration.vectorDot($(lhs), $(rhs)))
  #= Operands are proper Matrix: function inputs are converted by generateArrayConversions,
     array literals use ensureArray. =#
  e.op in (SC.OP_MATRIX_PRODUCT, SC.OP_MUL_VECTOR_MATRIX, SC.OP_MUL_MATRIX_VECTOR) && return :($(lhs) * $(rhs))
  return :($(CodeGeneration.opKindToJuliaOperator(e.op))($(lhs), $(rhs)))
end

"""
Boolean discretes read back from the state are 0/1 numbers: compared with 0 before `not`.
"""
expToJuliaExpAlg(e::SC.LUNARY)::Expr = :($(CodeGeneration.opKindToJuliaOperator(e.op))($(expToJuliaExpAlg(e.exp)) != 0))

"""
`||` and `&&` are special forms, `Expr(:||, ...)`; their operands compared with 0 (Boolean
discretes read back from the state are 0/1 numbers: a bare `&&` threw a TypeError).
"""
function expToJuliaExpAlg(e::SC.LBINARY)::Expr
  local (lhs, rhs) = (expToJuliaExpAlg(e.exp1), expToJuliaExpAlg(e.exp2))
  e.op === SC.OP_OR && return Expr(:||, :($(lhs) != 0), :($(rhs) != 0))
  e.op === SC.OP_AND && return Expr(:&&, :($(lhs) != 0), :($(rhs) != 0))
  return :($(CodeGeneration.opKindToJuliaOperator(e.op))($(lhs), $(rhs)))
end

expToJuliaExpAlg(e::SC.RELATION)::Expr =
  :($(CodeGeneration.opKindToJuliaOperator(e.op))($(expToJuliaExpAlg(e.exp1)), $(expToJuliaExpAlg(e.exp2))))

"""
The condition compared with 0: a numerically encoded Boolean discrete made `if <float>`
throw a TypeError.
"""
function expToJuliaExpAlg(e::SC.IFEXP)::Expr
  local cond = :($(expToJuliaExpAlg(e.cond)) != 0)
  local thenExp = expToJuliaExpAlg(e.thenExp)
  local elseExp = expToJuliaExpAlg(e.elseExp)
  quote
    if $(cond)
      $(thenExp)
    else
      $(elseExp)
    end
  end
end

"""
    expToJuliaExpAlg(e::CALL)

A call: `pre(x)` as `x`; `String(x, ...)` from the argument's type (an enumeration value is an
Integer here); a builtin by its implementation (MODELICA_BUILTIN_FUNCTIONS) or its Julia name;
a function of the model through `Base.invokelatest` (world age: its wrapper is defined after
the code that calls it is compiled), its record arguments as their fields.
"""
function expToJuliaExpAlg(e::SC.CALL)::Expr
  local builtin = e.kind === SC.CALL_BUILTIN
  builtin && e.name === :pre && return expToJuliaExpAlg(first(e.args))
  builtin && e.name === :String && return modelicaStringCall(e.args, expToJuliaExpAlg)
  local name = String(e.name)
  local callTarget = !builtin ? :(Base.invokelatest) :
    haskey(MODELICA_BUILTIN_FUNCTIONS, name) ?
      Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)),
           QuoteNode(MODELICA_BUILTIN_FUNCTIONS[name])) : e.name
  local expr = Expr(:call, callTarget)
  builtin || push!(expr.args, e.name)
  local shared = Expr[]
  append!(expr.args, _algCallArgs(e.args; builtin, shared))
  return isempty(shared) ? quote $(expr) end : Expr(:let, Expr(:block, shared...), expr)
end

"""
A function argument with bound arguments: a closure calling the function as a call in a
function body does; a bound record argument by its fields, as the callee takes it (MSL 4
Inverse_sh_T's `function f_nonlinear(data = ..., ...)`).
"""
function expToJuliaExpAlg(e::SC.PARTEVALFUNCTION)::Expr
  local bound = Any[(_recordCrefFields(b) !== nothing || _recordValueType(b) !== nothing) ? Expr(:..., _recordTupleExpr(b)) :
                    expToJuliaExpAlg(b) for b in e.args]
  local closure = _partialApplicationExpr(:(Base.invokelatest), bound, e; boundRecordsSplatted = true)
  insert!(closure.args[2].args, 2, e.name)
  return closure
end

"""
A cast: the frontend casts to Real only (typeCast), an Integer array to a Real one too
(left uncast, `y := m; y[1] := 1.5` failed with an InexactError).
"""
function expToJuliaExpAlg(e::SC.CAST)::Expr
  local innerExpr = expToJuliaExpAlg(e.exp)
  e.ty isa SC.TYPE_REAL && return :(float($innerExpr))
  e.ty isa SC.TYPE_INTEGER && return :(Int(round($innerExpr)))
  e.ty isa SC.TYPE_BOOL && return :(Bool($innerExpr))
  (e.ty isa SC.TYPE_ARRAY && SC.scalarType(e.ty) isa SC.TYPE_REAL) && return :(float.($innerExpr))
  CodeGeneration.unsupported("a cast to $(e.ty)", e)
end

"""
An array literal; of rows (a matrix literal) a proper Matrix (ensureArray).
"""
function expToJuliaExpAlg(e::SC.ARRAY_EXP)::Expr
  local arrExpr = Expr(:vect, Any[expToJuliaExpAlg(x) for x in e.elements]...)
  return e.ty isa SC.TYPE_ARRAY && length(e.ty.dims) >= 2 ? :(OMBackend.CodeGeneration.ensureArray($arrExpr)) : arrExpr
end

function expToJuliaExpAlg(e::SC.RANGE)::Expr
  local (startExpr, stopExpr) = (expToJuliaExpAlg(e.start), expToJuliaExpAlg(e.stop))
  e.step === nothing && return :($startExpr:$stopExpr)
  return :($startExpr:$(expToJuliaExpAlg(e.step)):$stopExpr)
end

"""
An array element; a record read by a literal index its field (field access, not array
indexing: the flat field symbol, as the record's fields are flattened).
"""
function expToJuliaExpAlg(e::SC.ASUB)::Expr
  local recordField = _recordIndexFieldSymbol(e.exp, e.subs)
  recordField === nothing || return quote $(recordField) end
  local arrExpr = e.exp isa SC.EXP_CREF && isempty(e.exp.cref.subs) ? e.exp.cref.sym : expToJuliaExpAlg(e.exp)
  return Expr(:ref, arrExpr, Any[_algIndexExpr(s) for s in e.subs]...)
end

"""
A record constructor: a tuple (one field: the field).
"""
function expToJuliaExpAlg(e::SC.RECORD)::Expr
  local fieldExprs = Any[expToJuliaExpAlg(x) for x in e.exps]
  return length(fieldExprs) == 1 ? first(fieldExprs) : Expr(:tuple, fieldExprs...)
end

"""
An array comprehension or reduction ({e for i in u, j in v}: the last iterator is the first
dimension, MLS 10.4.1.2; a Julia comprehension's first iterator is).
"""
function expToJuliaExpAlg(e::SC.REDUCTION)::Expr
  local bodyExpr = expToJuliaExpAlg(e.body)
  local iterExprs = Expr[Expr(:(=), Symbol(it.name), expToJuliaExpAlg(it.range)) for it in e.iterators]
  e.reducer === :array && return Expr(:comprehension, bodyExpr, reverse(iterExprs)...)
  local f = get(REDUCERS, e.reducer, nothing)
  f === nothing && CodeGeneration.unsupported("the reduction $(e.reducer)", e)
  return :($(f)($(Expr(:generator, bodyExpr, iterExprs...))))
end

"The Julia functions of Modelica's reductions."
const REDUCERS = Dict{Symbol, Symbol}(:sum => :sum, :product => :prod, :min => :minimum, :max => :maximum)

"""
A tuple output's element: a direct index.
"""
expToJuliaExpAlg(e::SC.TSUB)::Expr = :($(expToJuliaExpAlg(e.exp))[$(e.index)])

"""
A record field of a value (the `re`/`im` of a Complex value, another by its position or
name): the MTK path's helpers (algorithmic code never has Symbolics values).
"""
function expToJuliaExpAlg(e::SC.RSUB)::Expr
  local innerJL = expToJuliaExpAlg(e.exp)
  e.fieldName == "re" && return :(OMBackend.CodeGeneration._recordFieldRe($innerJL))
  e.fieldName == "im" && return :(OMBackend.CodeGeneration._recordFieldIm($innerJL))
  return :(OMBackend.CodeGeneration._recordField($innerJL, $(QuoteNode(Symbol(e.fieldName))), $(_positionalFieldIndex(e.exp, e.index))))
end

expToJuliaExpAlg(e::SC.TUPLE)::Expr = Expr(:tuple, Any[expToJuliaExpAlg(x) for x in e.elements]...)

expToJuliaExpAlg(e::SC.Exp)::Expr = CodeGeneration.unsupported("expression", e)

"""
    _partialApplicationExpr(callee, boundArgs, e; boundRecordsSplatted = false)

`function f(b = e, ...)`, an argument of a function that takes a function (Buildings'
Borefields: quadratureLobatto of an integrand): a closure over the bound arguments, its own
arguments the inputs of `f` the partial application leaves free, calling `callee` with every
input in place (the bound arguments in their order). A record input (flattened in the callee)
only bound, as its fields splatted (`boundRecordsSplatted`).
"""
function _partialApplicationExpr(callee::Union{Symbol, Expr}, boundArgs::Vector{Any}, e::SC.PARTEVALFUNCTION;
                                 boundRecordsSplatted::Bool = false)::Expr
  local bound = Set{Symbol}(e.argNames)
  local params = Symbol[]; local callArgs = Any[]; local k = 0
  for (name, t) in e.inputs
    t isa SC.TYPE_COMPLEX && (!(name in bound) || !boundRecordsSplatted) &&
      CodeGeneration.unsupported("a partial application of a function with a record input", name)
    if !(name in bound)
      local s = Symbol("#pa#", name)
      push!(params, s); push!(callArgs, s)
    else
      k += 1
      k <= length(boundArgs) || CodeGeneration.unsupported("a partial application's bound arguments", e)
      push!(callArgs, boundArgs[k])
    end
  end
  k == length(boundArgs) || CodeGeneration.unsupported("a partial application's bound arguments", e)
  return Expr(:->, Expr(:tuple, params...), Expr(:call, callee, callArgs...))
end
