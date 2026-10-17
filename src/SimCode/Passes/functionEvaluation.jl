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

#= Partial evaluation of Modelica function calls (OpenModelica's evalFunc).

   A call whose inputs are partly literals (parameters folded in) and partly variables can
   have outputs that are the same constant for every value of the variables: the MSL Spice3
   MOSFET's capacitances with its default parameters (CBD = CBS = 0, no oxide thickness) are
   0 whatever the node voltages, and `icBS = cBS*(der(B.v) - der(Sinternal))` is then 0 = 0,
   without the derivatives. Code generation solves such an equation for a derivative,
   dividing by cBS: 0/0 at the start of the Spice3 examples.

   The function bodies are interpreted abstractly: each scalar variable is a constant or
   UNKNOWN, keyed by its flattened name (`<record>_<field>`, array elements `name[i]`, `name[i,j]`
   for constant subscripts; a whole array or record is read as UNKNOWN). Arithmetic of
   constants folds (a non-finite result is UNKNOWN); a scalar product with an exact 0 is 0;
   `false and x` is false, `true or x` true. An if whose condition is UNKNOWN runs both branches
   and keeps a variable constant only where both give the identical value; a branch under a
   constant false condition is never evaluated (the Spice3 side-wall junction's 0/0 lies
   behind one). Calls recurse (memoized by the typed abstract arguments); loops with a
   constant range iterate; anything else (a while, a return, an impure or external function,
   an array or record valued call) is UNKNOWN. A call site whose scalar output comes out
   constant is replaced by the literal of the output's declared type.

   Abstract values are `Bool`, `Int`, `Float64`, `UNKNOWN`, and vectors of them (an array, a
   function's outputs): the interpretation's methods take them untyped. =#

"The abstract value of what the interpretation does not know."
struct UnknownValue end
const UNKNOWN = UnknownValue()

"Thrown to stop interpreting a body the interpretation does not follow (control flow)."
struct InterpretationBail <: Exception end

"""
    Interpretation(functions)

The state of interpreting one model's calls: its Modelica functions by name, the memo of
evaluated calls, the call depth and the step budget of the current call site.
"""
mutable struct Interpretation
  functions::Dict{String, MODELICA_FUNCTION}
  memo::Dict{Any, Any}
  depth::Int
  steps::Int
end

Interpretation(simCode::SIM_CODE)::Interpretation =
  Interpretation(Dict{String, MODELICA_FUNCTION}(f.name => f for f in simCode.functions if f isa MODELICA_FUNCTION),
                 Dict{Any, Any}(), 0, 0)

const MAX_CALL_DEPTH = 32
const SITE_STEP_BUDGET = 2_000_000

"An interpretation step; bail out past the budget of the call site."
function step!(ctx::Interpretation, n::Int = 1)::Nothing
  ctx.steps += n
  ctx.steps > SITE_STEP_BUDGET && throw(InterpretationBail())
  return nothing
end

isConstantValue(v)::Bool = v isa Real
"A memo key: 1, 1.0 and true are different arguments."
memoKey(v) = v isa AbstractVector ? Tuple(memoKey(x) for x in v) : (typeof(v), v)
"The same value of the same type (true and 1, 1 and 1.0 are not); the signs of a zero are not told apart."
sameValue(a, b)::Bool = (a isa Real && b isa Real) ? (typeof(a) == typeof(b) && (a === b || (iszero(a) && iszero(b)))) : isequal(a, b)
isFiniteValue(v)::Bool = !(v isa AbstractFloat) || isfinite(v)
finiteOrUnknown(v) = isFiniteValue(v) ? v : UNKNOWN
isNumericZero(v)::Bool = v isa Real && !(v isa Bool) && iszero(v)
"A condition's truth as the generated code tests it (`(c) != 0`: a Boolean record field can be an Integer literal)."
truthOf(v) = v isa Bool ? v : (v isa Real ? v != 0 : UNKNOWN)

"Whether a function variable is an array."
isArrayVariable(v::FunctionVariable)::Bool = v.ty isa TYPE_ARRAY || !isempty(v.dimensions)

"The field names of a record type, in declaration order; `nothing` for another type."
recordFields(t::SType)::Union{Nothing, Vector{String}} = t isa TYPE_COMPLEX && t.isRecord ? t.fieldNames : nothing

"The number of elements of an array type of known dimensions; `nothing` for another type."
arrayLength(t::SType)::Union{Nothing, Int} = (t isa TYPE_ARRAY && all(d -> d >= 0, t.dims)) ? prod(t.dims; init = 1) : nothing

subscriptSuffix(ix::Vector{Int})::String = isempty(ix) ? "" : "[" * join(ix, ",") * "]"

"The values of subscripts as integers; `nothing` if one is not a constant."
function subscriptValues(subs::Vector{Exp}, env::Dict{String, Any}, ctx::Interpretation)::Union{Nothing, Vector{Int}}
  local out = Int[]
  for s in subs
    local v = interpret(s, env, ctx)
    (v isa Integer && !(v isa Bool)) || return nothing
    push!(out, Int(v))
  end
  return out
end

"A reference's key: its name, its literal subscripts as `[i,j]`."
referenceKey(c::SimCref)::String = String(c.sym) * subscriptSuffix(c.subs)

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

const UNARY_BUILTINS = Dict{Symbol, Function}(
  :exp => exp, :log => log, :log10 => log10, :sqrt => sqrt, :sin => sin, :cos => cos, :tan => tan, :asin => asin,
  :acos => acos, :atan => atan, :sinh => sinh, :cosh => cosh, :tanh => tanh, :abs => abs, :sign => sign,
  :floor => floor, :ceil => ceil)

"The value of a built-in function of abstract arguments."
function builtinValue(name::Symbol, args::Vector{Any})
  name in (:noEvent, :homotopy) && return isempty(args) ? UNKNOWN : args[1]
  name === :smooth && return length(args) == 2 ? args[2] : UNKNOWN
  all(isConstantValue, args) || return UNKNOWN
  local f = get(UNARY_BUILTINS, name, nothing)
  try
    (f !== nothing && length(args) == 1) && return finiteOrUnknown(f(args[1]))
    length(args) == 2 || return name === :integer && length(args) == 1 ? Int(floor(args[1])) : UNKNOWN
    name === :max && return max(args[1], args[2])
    name === :min && return min(args[1], args[2])
    name === :atan2 && return finiteOrUnknown(atan(args[1], args[2]))
    name === :div && return finiteOrUnknown(div(args[1], args[2]))
    name === :mod && return finiteOrUnknown(mod(args[1], args[2]))
    name === :rem && return finiteOrUnknown(rem(args[1], args[2]))
  catch err
    #= outside the function's domain (a DomainError, a DivideError, an InexactError) =#
    _fallback(err, :interpretBuiltin)
    return UNKNOWN
  end
  return UNKNOWN
end

"`a op b` of abstract values; a product with an exact 0 is 0 (of the zero's type) for a finite or unknown other side."
function binaryValue(op::OpKind, a, b)
  if op === OP_MUL
    (isNumericZero(a) && (b === UNKNOWN || (b isa Real && isFiniteValue(b)))) && return zero(a)
    (isNumericZero(b) && (a === UNKNOWN || (a isa Real && isFiniteValue(a)))) && return zero(b)
  end
  (isConstantValue(a) && isConstantValue(b)) || return UNKNOWN
  try
    op === OP_ADD && return finiteOrUnknown(a + b)
    op === OP_SUB && return finiteOrUnknown(a - b)
    op === OP_MUL && return finiteOrUnknown(a * b)
    op === OP_DIV && return finiteOrUnknown(a / b)
    op === OP_POW && return finiteOrUnknown(Float64(a)^b)
  catch err
    _fallback(err, :interpretBinary)
  end
  return UNKNOWN
end

function relationValue(op::OpKind, a, b)
  (isConstantValue(a) && isConstantValue(b)) || return UNKNOWN
  local r = compareValues(Float64(a), op, Float64(b))
  return r === nothing ? UNKNOWN : r
end

"""
    interpret(e, env, ctx)

The abstract value of the scalar expression `e` in `env` (a whole array or record, and
anything not followed, is UNKNOWN).
"""
function interpret(e::Exp, env::Dict{String, Any}, ctx::Interpretation)
  step!(ctx)
  e isa RCONST && return e.value
  e isa ICONST && return e.value
  e isa BCONST && return e.value
  if e isa EXP_CREF
    (e.ty isa TYPE_ARRAY || recordFields(e.ty) !== nothing) && return UNKNOWN
    return get(env, referenceKey(e.cref), UNKNOWN)
  elseif e isa BINARY
    return binaryValue(e.op, interpret(e.exp1, env, ctx), interpret(e.exp2, env, ctx))
  elseif e isa UNARY
    local v = interpret(e.exp, env, ctx)
    return (e.op === OP_UMINUS && v isa Real && !(v isa Bool)) ? -v : UNKNOWN
  elseif e isa LBINARY
    local a = truthOf(interpret(e.exp1, env, ctx))
    if e.op === OP_AND
      a === false && return false
      local b = truthOf(interpret(e.exp2, env, ctx))
      b === false && return false
      return (a === true && b === true) ? true : UNKNOWN
    elseif e.op === OP_OR
      a === true && return true
      local b = truthOf(interpret(e.exp2, env, ctx))
      b === true && return true
      return (a === false && b === false) ? false : UNKNOWN
    end
    return UNKNOWN
  elseif e isa LUNARY
    local v = truthOf(interpret(e.exp, env, ctx))
    return v isa Bool ? !v : UNKNOWN
  elseif e isa RELATION
    return relationValue(e.op, interpret(e.exp1, env, ctx), interpret(e.exp2, env, ctx))
  elseif e isa IFEXP
    local c = truthOf(interpret(e.cond, env, ctx))
    c === true && return interpret(e.thenExp, env, ctx)
    c === false && return interpret(e.elseExp, env, ctx)
    local (a, b) = (interpret(e.thenExp, env, ctx), interpret(e.elseExp, env, ctx))
    return sameValue(a, b) ? a : UNKNOWN
  elseif e isa CAST
    local v = interpret(e.exp, env, ctx)
    return (e.ty isa TYPE_REAL && v isa Real && !(v isa Bool)) ? Float64(v) : UNKNOWN
  elseif e isa CALL
    (e.ty isa TYPE_ARRAY || recordFields(e.ty) !== nothing) && return UNKNOWN
    local r = callValue(e, env, ctx)
    return r isa AbstractVector ? (isempty(r) ? UNKNOWN : r[1]) : r
  elseif e isa ASUB
    local ix = subscriptValues(e.subs, env, ctx)
    ix === nothing && return UNKNOWN
    if e.exp isa EXP_CREF
      local k = referenceKey(e.exp.cref)
      local fields = recordFields(e.exp.ty)
      fields === nothing && return get(env, k * subscriptSuffix(ix), UNKNOWN)
      #= a record's field read by position =#
      return (length(ix) == 1 && 1 <= ix[1] <= length(fields)) ? get(env, k * "_" * fields[ix[1]], UNKNOWN) : UNKNOWN
    elseif e.exp isa CALL && !(e.exp.ty isa TYPE_ARRAY)
      local r = callValue(e.exp, env, ctx)
      return (r isa AbstractVector && length(ix) == 1 && 1 <= ix[1] <= length(r)) ? r[ix[1]] : UNKNOWN
    end
    return UNKNOWN
  elseif e isa TSUB
    e.exp isa CALL || return UNKNOWN
    local r = callValue(e.exp, env, ctx)
    return (r isa AbstractVector && 1 <= e.index <= length(r)) ? r[e.index] : UNKNOWN
  end
  return UNKNOWN
end

"The abstract outputs of a call of a model's function (a vector over its flattened outputs), or a built-in's value; an impure function's are UNKNOWN."
function callValue(e::CALL, env::Dict{String, Any}, ctx::Interpretation)
  e.kind === CALL_BUILTIN && return builtinValue(e.name, Any[interpret(a, env, ctx) for a in e.args])
  local f = get(ctx.functions, String(e.name), nothing)
  f === nothing && return UNKNOWN
  e.kind === CALL_IMPURE_FUNCTION && return Any[UNKNOWN for _ in f.outputs]
  local args = Any[]
  foreach(a -> argumentValues!(args, a, env, ctx), e.args)
  return functionValues(f, args, ctx)
end

"""
    argumentValues!(out, e, env, ctx)

The values of a call's argument, as the function's inputs are flattened: a record variable's
fields (nested records recursively, arrays as vectors), a record literal's or constructor's
values, a record-valued function's outputs, an array as a vector, else the scalar value.
"""
function argumentValues!(out::Vector{Any}, e::Exp, env::Dict{String, Any}, ctx::Interpretation)::Vector{Any}
  if e isa EXP_CREF && recordFields(e.ty) !== nothing
    recordFieldValues!(out, referenceKey(e.cref), e.ty::TYPE_COMPLEX, env)
  elseif e isa EXP_CREF && e.ty isa TYPE_ARRAY
    local n = arrayLength(e.ty)
    n === nothing && throw(InterpretationBail())
    local k = referenceKey(e.cref)
    push!(out, Any[get(env, k * "[" * string(i) * "]", UNKNOWN) for i in 1:n])
  elseif e isa RECORD
    foreach(x -> argumentValues!(out, x, env, ctx), e.exps)
  elseif e isa CALL && recordFields(e.ty) !== nothing
    if haskey(ctx.functions, String(e.name))
      local r = callValue(e, env, ctx)
      r isa AbstractVector || throw(InterpretationBail())
      append!(out, r)
    else
      #= a record constructor =#
      foreach(x -> argumentValues!(out, x, env, ctx), e.args)
    end
  elseif e isa ARRAY_EXP
    push!(out, Any[interpret(x, env, ctx) for x in e.elements])
  else
    push!(out, interpret(e, env, ctx))
  end
  return out
end

function recordFieldValues!(out::Vector{Any}, k::String, t::TYPE_COMPLEX, env::Dict{String, Any})::Vector{Any}
  for (name, ft) in zip(t.fieldNames, t.fieldTypes)
    local fk = k * "_" * name
    if recordFields(ft) !== nothing
      recordFieldValues!(out, fk, ft::TYPE_COMPLEX, env)
    else
      local n = arrayLength(ft)
      push!(out, n === nothing ? get(env, fk, UNKNOWN) : Any[get(env, fk * "[" * string(i) * "]", UNKNOWN) for i in 1:n])
    end
  end
  return out
end

#= ==================================================================================== =#
#= Functions and statements                                                             =#
#= ==================================================================================== =#

"""
    functionValues(f, args, ctx)

The outputs of `f` for abstract arguments: a vector over its (flattened) outputs, all
UNKNOWN when the body is not followed; an array output is UNKNOWN.
"""
function functionValues(f::MODELICA_FUNCTION, args::Vector{Any}, ctx::Interpretation)::Vector{Any}
  local unknown = Any[UNKNOWN for _ in f.outputs]
  local key = (f.name, memoKey(args))
  local hit = get(ctx.memo, key, missing)
  hit === missing || return hit === nothing ? unknown : hit
  ctx.depth >= MAX_CALL_DEPTH && return unknown
  ctx.memo[key] = nothing
  local env = Dict{String, Any}()
  local result = unknown
  ctx.depth += 1
  try
    length(args) == length(f.inputs) || throw(InterpretationBail())
    for (v, a) in zip(f.inputs, args)
      if isArrayVariable(v)
        a isa AbstractVector || throw(InterpretationBail())
        foreach(((i, x),) -> env[v.name * "[" * string(i) * "]"] = x, enumerate(a))
      else
        a isa AbstractVector && throw(InterpretationBail())
        env[v.name] = a
      end
    end
    #= as the generated code: an output starts at its default, else at 0.0 / 0 / false =#
    for v in f.outputs
      isArrayVariable(v) && continue
      if v.default !== nothing
        env[v.name] = interpret(v.default, env, ctx)
      elseif v.ty isa TYPE_REAL
        env[v.name] = 0.0
      elseif v.ty isa TYPE_INTEGER
        env[v.name] = 0
      elseif v.ty isa TYPE_BOOL
        env[v.name] = false
      end
    end
    for v in f.locals
      (isArrayVariable(v) || v.default === nothing) && continue
      env[v.name] = interpret(v.default, env, ctx)
    end
    interpretStatements!(f.statements, env, ctx)
    result = Any[isArrayVariable(v) ? UNKNOWN : get(env, v.name, UNKNOWN) for v in f.outputs]
  catch err
    #= a bail, or a construct the interpretation does not know (a MethodError of it) =#
    err isa InterpretationBail || _fallback(err, :interpretFunction; expect = MethodError)
    result = unknown
  finally
    ctx.depth -= 1
  end
  ctx.memo[key] = result
  return result
end

"`env` without the key `k` and its elements and record fields."
function forget!(env::Dict{String, Any}, k::String)::Dict{String, Any}
  for kk in collect(keys(env))
    (kk == k || startswith(kk, k * "[") || startswith(kk, k * "_")) && delete!(env, kk)
  end
  return env
end

"Assign the abstract `value` to the target `lhs` (a variable, a record's fields, an array's elements, an element)."
function assignValue!(lhs::Exp, value, env::Dict{String, Any}, ctx::Interpretation)::Nothing
  lhs isa WILD && return nothing
  if lhs isa EXP_CREF
    local k = referenceKey(lhs.cref)
    step!(ctx, length(env))
    forget!(env, k)
    local fields = recordFields(lhs.ty)
    if fields !== nothing
      #= a record: its fields from the value's (flattened, one level) =#
      if value isa AbstractVector && length(value) == length(fields) && !any(x -> x isa AbstractVector, value)
        foreach(((f, x),) -> env[k * "_" * f] = x, zip(fields, value))
      end
    elseif lhs.ty isa TYPE_ARRAY
      value isa AbstractVector && foreach(((i, x),) -> env[k * "[" * string(i) * "]"] = x, enumerate(value))
    elseif !(value isa AbstractVector)
      env[k] = value
    end
    return nothing
  elseif lhs isa ASUB && lhs.exp isa EXP_CREF
    local ix = subscriptValues(lhs.subs, env, ctx)
    ix === nothing && throw(InterpretationBail())
    local ek = referenceKey(lhs.exp.cref) * subscriptSuffix(ix)
    step!(ctx, length(env))
    forget!(env, ek)
    value isa AbstractVector || (env[ek] = value)
    return nothing
  end
  throw(InterpretationBail())
end

"The value a record-typed right-hand side assigns: its fields'."
function recordValue(rhs::Exp, fields::Vector{String}, env::Dict{String, Any}, ctx::Interpretation)
  if rhs isa EXP_CREF
    local k = referenceKey(rhs.cref)
    return Any[get(env, k * "_" * f, UNKNOWN) for f in fields]
  elseif rhs isa CALL
    return haskey(ctx.functions, String(rhs.name)) ? callValue(rhs, env, ctx) : Any[interpret(x, env, ctx) for x in rhs.args]
  elseif rhs isa RECORD
    return Any[interpret(x, env, ctx) for x in rhs.exps]
  elseif rhs isa IFEXP
    local c = truthOf(interpret(rhs.cond, env, ctx))
    c === true && return recordValue(rhs.thenExp, fields, env, ctx)
    c === false && return recordValue(rhs.elseExp, fields, env, ctx)
    local (a, b) = (recordValue(rhs.thenExp, fields, env, ctx), recordValue(rhs.elseExp, fields, env, ctx))
    (a isa AbstractVector && b isa AbstractVector && length(a) == length(b)) || return UNKNOWN
    return Any[sameValue(x, y) ? x : UNKNOWN for (x, y) in zip(a, b)]
  end
  return UNKNOWN
end

"`env` as the join of two branches' environments: a variable keeps a value both give identically."
function joinBranches!(env::Dict{String, Any}, a::Dict{String, Any}, b::Dict{String, Any})::Dict{String, Any}
  empty!(env)
  for (k, va) in a
    sameValue(va, get(b, k, UNKNOWN)) && (env[k] = va)
  end
  return env
end

"Interpret the branches of an if from the `k`th on, then its else-statements."
function interpretBranches!(s::IF, k::Int, env::Dict{String, Any}, ctx::Interpretation)::Nothing
  k > length(s.branches) && return interpretStatements!(s.elseBody, env, ctx)
  local b = s.branches[k]
  local c = truthOf(interpret(b.condition, env, ctx))
  c === true && return interpretStatements!(b.body, env, ctx)
  c === false && return interpretBranches!(s, k + 1, env, ctx)
  step!(ctx, 3 * length(env))
  local (thenEnv, elseEnv) = (copy(env), copy(env))
  interpretStatements!(b.body, thenEnv, ctx)
  interpretBranches!(s, k + 1, elseEnv, ctx)
  joinBranches!(env, thenEnv, elseEnv)
  return nothing
end

"Whether an assertion level is AssertionLevel.warning."
isWarningLevel(level::Exp)::Bool = level isa ENUM_LITERAL && endswith(String(level.name), "warning")

"""
    interpretStatements!(body, env, ctx)

Interpret statements: assignments (of records, arrays, tuples of a call's outputs), ifs, for
loops of a constant range, asserts that hold on the constants (one of level warning that
fails is reported, once); bail out of anything else (a while, a return, a call for its side
effects, terminate, an assert not known to hold).
"""
function interpretStatements!(body::Vector{Statement}, env::Dict{String, Any}, ctx::Interpretation)::Nothing
  for s in body
    step!(ctx)
    if s isa ASSIGN && s.left isa TUPLE
      s.right isa CALL || throw(InterpretationBail())
      local outs = callValue(s.right, env, ctx)
      (outs isa AbstractVector && length(outs) >= length(s.left.elements)) || throw(InterpretationBail())
      for (l, v) in zip(s.left.elements, outs)
        (l isa EXP_CREF && (recordFields(l.ty) !== nothing || l.ty isa TYPE_ARRAY)) && throw(InterpretationBail())
        assignValue!(l, v, env, ctx)
      end
    elseif s isa ASSIGN
      local fields = s.left isa EXP_CREF ? recordFields(s.left.ty) : nothing
      local value = if fields !== nothing
        recordValue(s.right, fields, env, ctx)
      elseif s.left isa EXP_CREF && s.left.ty isa TYPE_ARRAY
        s.right isa ARRAY_EXP ? Any[interpret(x, env, ctx) for x in s.right.elements] : UNKNOWN
      else
        interpret(s.right, env, ctx)
      end
      assignValue!(s.left, value, env, ctx)
    elseif s isa IF
      interpretBranches!(s, 1, env, ctx)
    elseif s isa FOR
      local r = s.iterator.range
      r isa RANGE || throw(InterpretationBail())
      local (a, st, b) = (interpret(r.start, env, ctx), r.step === nothing ? 1 : interpret(r.step, env, ctx), interpret(r.stop, env, ctx))
      (a isa Integer && st isa Integer && b isa Integer && st != 0) || throw(InterpretationBail())
      for i in a:st:b
        step!(ctx)
        env[s.iterator.name] = i
        interpretStatements!(s.body, env, ctx)
      end
    elseif s isa ASSERT
      local holds = try
        truthOf(interpret(s.condition, env, ctx))
      catch err
        err isa InterpretationBail || rethrow()
        nothing
      end
      if holds === false && isWarningLevel(s.level)
        @warn "an assert (level warning) fails in a call folded at the build" message = sprint(show, s.message)
      elseif holds !== true
        throw(InterpretationBail())
      end
    else
      throw(InterpretationBail())
    end
  end
  return nothing
end

#= ==================================================================================== =#
#= The pass                                                                             =#
#= ==================================================================================== =#

"The literal of a constant output of type `t`; `nothing` for another value."
function outputLiteral(v, t::SType)::Union{Nothing, Exp}
  t isa TYPE_REAL && return (v isa Real && !(v isa Bool) && isfinite(Float64(v))) ? RCONST(Float64(v)) : nothing
  t isa TYPE_INTEGER && return (v isa Integer && !(v isa Bool)) ? ICONST(Int(v)) : nothing
  t isa TYPE_BOOL && return v isa Bool ? BCONST(v) : nothing
  return nothing
end

"The abstract outputs of a call from its arguments (literals known, anything else UNKNOWN), with a step budget of its own; `nothing` for an impure call."
function callOutputs(call::CALL, f::MODELICA_FUNCTION, ctx::Interpretation)::Union{Nothing, Vector{Any}}
  call.kind === CALL_IMPURE_FUNCTION && return nothing
  local env = Dict{String, Any}()
  local args = Any[]
  (ctx.steps, ctx.depth) = (0, 0)
  try
    #= record arguments as their fields, as the function's inputs are flattened =#
    local flat = Exp[]
    local flatten! = nothing
    flatten! = a -> (a isa RECORD ? foreach(flatten!, a.exps) : push!(flat, a))
    foreach(flatten!, call.args)
    foreach(a -> argumentValues!(args, a, env, ctx), flat)
  catch err
    err isa InterpretationBail || _fallback(err, :interpretArguments; expect = MethodError)
    return nothing
  end
  return functionValues(f, args, ctx)
end

"""
    constantOutput(e, ctx)

The literal of a function call's scalar output (`f(x)` of a function of one output, `f(x)[i]`,
the `i`th output of a tuple) that is the same constant for every value of the call's
variables; `nothing` for another expression.
"""
function constantOutput(e::Exp, ctx::Interpretation)::Union{Nothing, Exp}
  local (call, ix) = if e isa ASUB && e.exp isa CALL && length(e.subs) == 1 && e.subs[1] isa ICONST && !(e.exp.ty isa TYPE_ARRAY)
    (e.exp, e.subs[1].value)
  elseif e isa TSUB && e.exp isa CALL
    (e.exp, e.index)
  elseif e isa CALL && !(e.ty isa TYPE_ARRAY)
    (e, 1)
  else
    (nothing, 0)
  end
  (call === nothing || call.kind === CALL_BUILTIN) && return nothing
  local f = get(ctx.functions, String(call.name), nothing)
  #= a plain call only of a function of one output (another's is read by subscript) =#
  (f === nothing || (e isa CALL && length(f.outputs) != 1)) && return nothing
  (1 <= ix <= length(f.outputs) && !isArrayVariable(f.outputs[ix])) || return nothing
  local outs = callOutputs(call, f, ctx)
  return outs === nothing ? nothing : outputLiteral(outs[ix], f.outputs[ix].ty)
end

"""
    evaluateConstantFunctionOutputs(simCode)

The calls of Modelica functions whose scalar output is the same constant for every value of
the call's variables replaced by that literal (see the top of the file), in the residuals and
the if-equations' branches; the constant equations are then propagated (an output that
multiplied a derivative drops it: MSL Spice3's MOSFET capacitances). Nothing for a model with
structural modes or translated again at its structural changes.
"""
function evaluateConstantFunctionOutputs(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local ctx = Interpretation(simCode)
  isempty(ctx.functions) && return simCode
  local replaced = Ref(0)
  local visit = (e::Exp, acc::Nothing) -> begin
    local lit = constantOutput(e, ctx)
    lit === nothing && return (e, true, acc)
    replaced[] += 1
    return (lit, false, acc)
  end
  local fold = eq::RESIDUAL_EQUATION -> RESIDUAL_EQUATION(first(traverseExpTopDown(eq.exp, visit, nothing)), eq.source, eq.attr)
  local residuals = RESIDUAL_EQUATION[fold(eq) for eq in simCode.residualEquations]
  local ifEquations = IF_EQUATION[IF_EQUATION(BRANCH[withResiduals(b, RESIDUAL_EQUATION[fold(eq) for eq in b.residualEquations])
                                                     for b in ifEq.branches]) for ifEq in simCode.ifEquations]
  replaced[] == 0 && return simCode
  @debug "[SIMCODE: $(simCode.name): evaluateConstantFunctionOutputs] $(replaced[]) function outputs are constants"
  return propagateConstants(update(simCode; residualEquations = residuals, ifEquations))
end

#= ==================================================================================== =#
#= Values at the start of the build                                                     =#
#= ==================================================================================== =#

"The memo of `initialAlgorithmEnvironment`: the simulation code it was computed for (kept, compared by identity), its environment and interpretation."
const INITIAL_ALGORITHM_ENVIRONMENT = Ref{Union{Nothing, Tuple{SIM_CODE, Dict{String, Any}, Interpretation}}}(nothing)

"""
    initialAlgorithmEnvironment(simCode)

The initial algorithms interpreted at the start time 0 (the build's, as for the sample()
ticks), with the values of the parameters that have one: what they assign (CDL's samplers set
their sample start so, from a free parameter). An algorithm the interpretation does not follow
assigns nothing. Memoized for the simulation code being generated (by identity: the memo keeps
it, so its identity is not a later one's, as an `objectid` would be after the GC reused its
fields' memory).
"""
function initialAlgorithmEnvironment(simCode::SIM_CODE)::Tuple{Dict{String, Any}, Interpretation}
  local cached = INITIAL_ALGORITHM_ENVIRONMENT[]
  cached !== nothing && cached[1] === simCode && return (cached[2], cached[3])
  local ctx = Interpretation(simCode)
  local env = Dict{String, Any}("time" => 0.0)
  for (k, (_, v)) in simCode.stringToSimVarHT
    (v.varKind isa PARAMETER && v.varKind.bindExp !== nothing) || continue
    local x = tryEvalNumeric(v.varKind.bindExp, simCode)
    x === nothing || (env[k] = x)
  end
  for a in simCode.initialAlgorithms
    local trial = copy(env)
    (ctx.steps, ctx.depth) = (0, 0)
    try
      interpretStatements!(a.statements, trial, ctx)
      env = trial
    catch err
      err isa InterpretationBail || _fallback(err, :initialAlgorithmInterpretation; expect = MethodError)
    end
  end
  INITIAL_ALGORITHM_ENVIRONMENT[] = (simCode, env, ctx)
  return (env, ctx)
end

"""
    valueAtBuildStart(e, simCode)

The value of `e` at the start time 0 from the parameters' values and what the initial
algorithms assign (`initialAlgorithmEnvironment`), the model's function calls interpreted;
`nothing` if it is not known.
"""
function valueAtBuildStart(e::Exp, simCode::SIM_CODE)::Union{Nothing, Float64}
  local (env, ctx) = initialAlgorithmEnvironment(simCode)
  ctx.steps = 0
  local v = try
    interpret(e, env, ctx)
  catch err
    err isa InterpretationBail || _fallback(err, :initialAlgorithmValue; expect = MethodError)
    nothing
  end
  return (v isa Real && !(v isa Bool) && isfinite(v)) ? Float64(v) : nothing
end
