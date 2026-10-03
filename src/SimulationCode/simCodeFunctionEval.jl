#= Partial evaluation of Modelica function calls (OpenModelica's evalFunc).

   A call whose inputs are partly literals (parameters folded in) and partly
   variables can have outputs that are the same constant for every value of
   the variables: the MSL Spice3 MOSFET's capacitances with its default
   parameters (CBD = CBS = 0, no oxide thickness) are 0 whatever the node
   voltages, and `icBS = cBS*(der(B.v) - der(Sinternal))` is then 0 = 0,
   without the derivatives. Code generation solves such an equation for a
   derivative, dividing by cBS: 0/0 at the start of the Spice3 examples.

   The function bodies are interpreted abstractly: each scalar variable is a
   constant or Unknown, keyed by its flattened name (`<record>_<field>`,
   array elements `name[i]` for constant subscripts; a whole array or record
   is read as Unknown). Arithmetic of constants folds (a non-finite result is
   Unknown); a scalar product with an exact 0 is 0; `false and x` is false,
   `true or x` true. An if whose condition is Unknown runs both branches and
   keeps a variable constant only where both give the identical value; a
   branch under a constant false condition is never evaluated (the Spice3
   side-wall junction's 0/0 lies behind one). Calls recurse (memoized by the
   typed abstract arguments); loops with a constant range iterate; anything
   else (a while, a return, an impure or external function, an array or
   record valued call) is Unknown. A call site whose scalar output comes out
   constant is replaced by the literal of the output's declared type. =#

struct _AUnknown end
const _AUNK = _AUnknown()

#= The interpretation's state for one model: the functions by canonical name,
   the memo of evaluated calls, the depth and the step budget of the current
   call site. =#
mutable struct _AEvalContext
  functions::Dict{String, MODELICA_FUNCTION}
  memo::Dict{Any, Any}
  depth::Int
  steps::Int
end

const _AEVAL_MAX_DEPTH = 32
const _AEVAL_SITE_STEPS = 2_000_000

struct _ABail <: Exception end

_aIsConst(v) = v isa Real
#= Memo keys typed: 1, 1.0 and true are different arguments. =#
_aKey(v) = v isa AbstractVector ? Tuple(_aKey(x) for x in v) : (typeof(v), v)
#= The same value of the same type (true and 1, or 1 and 1.0, are not); the
   signs of a zero are not told apart (c*x and c*(-x) with c = 0). =#
_aSame(a, b) = (a isa Real && b isa Real) ? (typeof(a) == typeof(b) && (a === b || (iszero(a) && iszero(b)))) :
  isequal(a, b)
_aFinite(v) = !(v isa AbstractFloat) || isfinite(v)
_aResult(v) = _aFinite(v) ? v : _AUNK

function _aStep!(ctx::_AEvalContext, n::Int = 1)
  ctx.steps += n
  ctx.steps > _AEVAL_SITE_STEPS && throw(_ABail())
  return nothing
end

#= Whether a function variable is an array: for function parameters the
   element type is in `ty` and the shape in `dims`. =#
_aIsArrayVar(v) = v.ty isa DAE.T_ARRAY || !isempty(v.dims)

#= The subscripts of a cref or an ASUB as integers, or nothing when one is not constant. =#
function _aSubscripts(subs, env, ctx)::Union{Vector{Int}, Nothing}
  local out = Int[]
  for s in subs
    local e = s isa DAE.INDEX ? s.exp : s
    e isa DAE.Exp || return nothing
    local v = _aEval(e, env, ctx)
    (v isa Integer && !(v isa Bool)) || return nothing
    push!(out, Int(v))
  end
  return out
end

_aSubsSuffix(ix::Vector{Int}) = isempty(ix) ? "" : "[" * join(ix, ",") * "]"

#= A cref's key: its identifiers joined with `_` (the flattened record
   fields), constant subscripts as `[i]`; nothing when a subscript is not constant. =#
function _aCrefKey(cr, env, ctx)::Union{String, Nothing}
  if cr isa DAE.CREF_IDENT
    local ix = _aSubscripts(collect(cr.subscriptLst), env, ctx)
    return ix === nothing ? nothing : cr.ident * _aSubsSuffix(ix)
  elseif cr isa DAE.CREF_QUAL
    local ix = _aSubscripts(collect(cr.subscriptLst), env, ctx)
    ix === nothing && return nothing
    local rest = _aCrefKey(cr.componentRef, env, ctx)
    return rest === nothing ? nothing : cr.ident * _aSubsSuffix(ix) * "_" * rest
  end
  return nothing
end

#= The field names of a record type, in declaration order; nothing for another type. =#
function _aRecordFields(@nospecialize(ty))::Union{Vector{String}, Nothing}
  ty isa DAE.T_COMPLEX || return nothing
  ty.complexClassType isa DAE.ClassInf.RECORD || return nothing
  return String[v.name for v in ty.varLst]
end

#= Whether a value is exactly a numeric zero (not false). =#
_aIsZero(v) = v isa Real && !(v isa Bool) && iszero(v)

const _A_UNARY_BUILTINS = Dict{String, Function}(
  "exp" => exp, "log" => log, "log10" => log10, "sqrt" => sqrt, "sin" => sin, "cos" => cos,
  "tan" => tan, "asin" => asin, "acos" => acos, "atan" => atan, "sinh" => sinh, "cosh" => cosh,
  "tanh" => tanh, "abs" => abs, "sign" => sign, "floor" => floor, "ceil" => ceil)

function _aBuiltin(name::String, args::Vector{Any})
  if name in ("noEvent", "homotopy", "smooth")
    #= smooth(order, expr); homotopy(actual, simplified) =#
    return name == "smooth" ? (length(args) == 2 ? args[2] : _AUNK) : (isempty(args) ? _AUNK : args[1])
  end
  all(_aIsConst, args) || return _AUNK
  local f = get(_A_UNARY_BUILTINS, name, nothing)
  try
    f !== nothing && length(args) == 1 && return _aResult(f(args[1]))
    name == "max" && length(args) == 2 && return max(args[1], args[2])
    name == "min" && length(args) == 2 && return min(args[1], args[2])
    name == "atan2" && length(args) == 2 && return _aResult(atan(args[1], args[2]))
    name == "integer" && length(args) == 1 && return Int(floor(args[1]))
    name == "div" && length(args) == 2 && return _aResult(div(args[1], args[2]))
    name == "mod" && length(args) == 2 && return _aResult(mod(args[1], args[2]))
    name == "rem" && length(args) == 2 && return _aResult(rem(args[1], args[2]))
  catch err
    #= Outside the function's domain (a DomainError, a DivideError, an InexactError): Unknown. =#
    OMBackend._fallback(err, :interpretBuiltin)
    return _AUNK
  end
  return _AUNK
end

function _aBinary(op, a, b)
  if op isa DAE.MUL
    #= A scalar product with an exact 0 (the other operand finite or Unknown). =#
    local z = op.ty isa DAE.T_INTEGER ? 0 : 0.0
    (_aIsZero(a) && (b === _AUNK || (b isa Real && _aFinite(b)))) && return z
    (_aIsZero(b) && (a === _AUNK || (a isa Real && _aFinite(a)))) && return z
  end
  (_aIsConst(a) && _aIsConst(b)) || return _AUNK
  try
    op isa DAE.ADD && return _aResult(a + b)
    op isa DAE.SUB && return _aResult(a - b)
    op isa DAE.MUL && return _aResult(a * b)
    op isa DAE.DIV && return _aResult(a / b)
    op isa DAE.POW && return _aResult(Float64(a)^b)
  catch err
    OMBackend._fallback(err, :interpretBinary)
    return _AUNK
  end
  return _AUNK
end

function _aRelation(op, a, b)
  (_aIsConst(a) && _aIsConst(b)) || return _AUNK
  op isa DAE.LESS && return a < b
  op isa DAE.LESSEQ && return a <= b
  op isa DAE.GREATER && return a > b
  op isa DAE.GREATEREQ && return a >= b
  op isa DAE.EQUAL && return a == b
  op isa DAE.NEQUAL && return a != b
  return _AUNK
end

#= A condition's truth as the generated code tests it (`(c) != 0`: Boolean
   record fields can arrive as integer literals). =#
_aBool(v) = v isa Bool ? v : (v isa Real ? v != 0 : _AUNK)

#= AssertionLevel.warning (AlgorithmicCodeGeneration.isWarningAssertionLevel). =#
_isWarningLevel(@nospecialize(level))::Bool = level isa DAE.ENUM_LITERAL && endswith(string(level.name), "warning")

#= The abstract value of a scalar DAE exp in env (a whole array or record,
   and anything not followed, is Unknown). =#
function _aEval(@nospecialize(e), env::Dict{String, Any}, ctx::_AEvalContext)
  _aStep!(ctx)
  if e isa DAE.RCONST
    return Float64(e.real)
  elseif e isa DAE.ICONST
    return e.integer
  elseif e isa DAE.BCONST
    return e.bool
  elseif e isa DAE.CREF
    (e.ty isa DAE.T_ARRAY || _aRecordFields(e.ty) !== nothing) && return _AUNK
    local k = _aCrefKey(e.componentRef, env, ctx)
    return k === nothing ? _AUNK : get(env, k, _AUNK)
  elseif e isa DAE.BINARY
    return _aBinary(e.operator, _aEval(e.exp1, env, ctx), _aEval(e.exp2, env, ctx))
  elseif e isa DAE.UNARY
    local v = _aEval(e.exp, env, ctx)
    return (e.operator isa DAE.UMINUS && v isa Real && !(v isa Bool)) ? -v : _AUNK
  elseif e isa DAE.LBINARY
    local a = _aBool(_aEval(e.exp1, env, ctx))
    if e.operator isa DAE.AND
      a === false && return false
      local b = _aBool(_aEval(e.exp2, env, ctx))
      b === false && return false
      return (a === true && b === true) ? true : _AUNK
    elseif e.operator isa DAE.OR
      a === true && return true
      local b = _aBool(_aEval(e.exp2, env, ctx))
      b === true && return true
      return (a === false && b === false) ? false : _AUNK
    end
    return _AUNK
  elseif e isa DAE.LUNARY
    local v = _aBool(_aEval(e.exp, env, ctx))
    return v isa Bool ? !v : _AUNK
  elseif e isa DAE.RELATION
    return _aRelation(e.operator, _aEval(e.exp1, env, ctx), _aEval(e.exp2, env, ctx))
  elseif e isa DAE.IFEXP
    local c = _aBool(_aEval(e.expCond, env, ctx))
    c === true && return _aEval(e.expThen, env, ctx)
    c === false && return _aEval(e.expElse, env, ctx)
    local a = _aEval(e.expThen, env, ctx)
    local b = _aEval(e.expElse, env, ctx)
    return _aSame(a, b) ? a : _AUNK
  elseif e isa DAE.CAST
    local v = _aEval(e.exp, env, ctx)
    e.ty isa DAE.T_REAL && return (v isa Real && !(v isa Bool)) ? Float64(v) : _AUNK
    return _AUNK
  elseif e isa DAE.CALL
    #= A scalar call (a Modelica function's first output, or a builtin). =#
    (e.attr.ty isa DAE.T_ARRAY || _aRecordFields(e.attr.ty) !== nothing) && return _AUNK
    local r = _aCall(e, env, ctx)
    return r isa AbstractVector ? (isempty(r) ? _AUNK : r[1]) : r
  elseif e isa DAE.ASUB
    local ix = _aSubscripts(collect(e.sub), env, ctx)
    ix === nothing && return _AUNK
    if e.exp isa DAE.CREF
      local k = _aCrefKey(e.exp.componentRef, env, ctx)
      k === nothing && return _AUNK
      local fields = _aRecordFields(e.exp.ty)
      if fields !== nothing
        #= A record's field read by position. =#
        return (length(ix) == 1 && 1 <= ix[1] <= length(fields)) ? get(env, k * "_" * fields[ix[1]], _AUNK) : _AUNK
      end
      return get(env, k * _aSubsSuffix(ix), _AUNK)
    elseif e.exp isa DAE.CALL && !(e.exp.attr.ty isa DAE.T_ARRAY)
      local r = _aCall(e.exp, env, ctx)
      return (r isa AbstractVector && length(ix) == 1 && 1 <= ix[1] <= length(r)) ? r[ix[1]] : _AUNK
    end
    return _AUNK
  elseif e isa DAE.TSUB
    e.exp isa DAE.CALL || return _AUNK
    local r = _aCall(e.exp, env, ctx)
    return (r isa AbstractVector && 1 <= e.ix <= length(r)) ? r[e.ix] : _AUNK
  end
  return _AUNK
end

#= The abstract outputs of a call (a vector over the callee's flattened
   outputs), or a builtin's value. Impure functions are not followed. =#
function _aCall(e::DAE.CALL, env::Dict{String, Any}, ctx::_AEvalContext)
  local name = string(e.path)
  local f = get(ctx.functions, OMBackend.canonicalName(name), nothing)
  f === nothing && return _aBuiltin(name, Any[_aEval(a, env, ctx) for a in e.expLst])
  e.attr.isImpure && return Any[_AUNK for _ in f.outputs]
  local args = Any[]
  foreach(a -> _aArgValues!(args, a, env, ctx), e.expLst)
  return _aFunction(f, args, ctx)
end

#= The number of elements of an array type with constant dimensions, or nothing. =#
function _aArrayLength(@nospecialize(ty))::Union{Int, Nothing}
  ty isa DAE.T_ARRAY || return nothing
  local n = 1
  for d in ty.dims
    d isa DAE.DIM_INTEGER || return nothing
    n *= d.integer
  end
  return n
end

#= An argument's values: a record variable's fields (nested records
   recursively, arrays as vectors), a record literal's or constructor's
   values, an array as a vector, else the scalar value. =#
function _aArgValues!(out::Vector{Any}, @nospecialize(e), env::Dict{String, Any}, ctx::_AEvalContext)
  if e isa DAE.CREF && _aRecordFields(e.ty) !== nothing
    local k = _aCrefKey(e.componentRef, env, ctx)
    k === nothing && throw(_ABail())
    _aRecordFieldValues!(out, k, e.ty, env)
  elseif e isa DAE.CREF && e.ty isa DAE.T_ARRAY
    local k = _aCrefKey(e.componentRef, env, ctx)
    local n = _aArrayLength(e.ty)
    (k === nothing || n === nothing) && throw(_ABail())
    push!(out, Any[get(env, k * "[" * string(i) * "]", _AUNK) for i in 1:n])
  elseif e isa DAE.RECORD
    foreach(x -> _aArgValues!(out, x, env, ctx), e.exps)
  elseif e isa DAE.CALL && _aRecordFields(e.attr.ty) !== nothing
    if haskey(ctx.functions, OMBackend.canonicalName(string(e.path)))
      #= A record-valued function: its flattened outputs. =#
      local r = _aCall(e, env, ctx)
      r isa AbstractVector || throw(_ABail())
      append!(out, r)
    else
      #= A record constructor. =#
      foreach(x -> _aArgValues!(out, x, env, ctx), e.expLst)
    end
  elseif e isa DAE.ARRAY
    push!(out, Any[_aEval(x, env, ctx) for x in e.array])
  else
    push!(out, _aEval(e, env, ctx))
  end
  return out
end

function _aRecordFieldValues!(out::Vector{Any}, k::String, @nospecialize(ty), env::Dict{String, Any})
  for v in ty.varLst
    local fk = k * "_" * v.name
    if _aRecordFields(v.ty) !== nothing
      _aRecordFieldValues!(out, fk, v.ty, env)
    else
      local n = _aArrayLength(v.ty)
      push!(out, n === nothing ? get(env, fk, _AUNK) : Any[get(env, fk * "[" * string(i) * "]", _AUNK) for i in 1:n])
    end
  end
  return out
end

#= A function variable's key, as its crefs in the body are keyed. =#
function _aVarName(v, env, ctx)::String
  local k = _aCrefKey(v.componentRef, env, ctx)
  k === nothing && throw(_ABail())
  return k
end

#= The outputs of a function for abstract arguments: a vector over its
   (flattened) outputs, all Unknown when the body cannot be followed. Array
   outputs are Unknown. =#
function _aFunction(f::MODELICA_FUNCTION, args::Vector{Any}, ctx::_AEvalContext)::Vector{Any}
  local unknown = Any[_AUNK for _ in f.outputs]
  local key = (f.name, _aKey(args))
  local hit = get(ctx.memo, key, missing)
  hit === missing || return hit === nothing ? unknown : hit
  ctx.depth >= _AEVAL_MAX_DEPTH && return unknown
  ctx.memo[key] = nothing
  local env = Dict{String, Any}()
  local result = unknown
  ctx.depth += 1
  try
    length(args) == length(f.inputs) || throw(_ABail())
    for (v, a) in zip(f.inputs, args)
      local name = _aVarName(v, env, ctx)
      if _aIsArrayVar(v)
        a isa AbstractVector || throw(_ABail())
        for (i, x) in enumerate(a)
          env[name * "[" * string(i) * "]"] = x
        end
      else
        a isa AbstractVector && throw(_ABail())
        env[name] = a
      end
    end
    for v in f.outputs
      #= As the generated code (generateOutputDefaults): an output starts at its
         binding, else at 0.0 / 0 / false. The binding was not used: an output
         the body does not assign folded to 0, also on unknown inputs
         (`f(time)`, MSL Media). =#
      _aIsArrayVar(v) && continue
      local name = _aVarName(v, env, ctx)
      if v.binding isa SOME
        env[name] = _aEval(v.binding.data, env, ctx)
      else
        v.ty isa DAE.T_REAL && (env[name] = 0.0)
        v.ty isa DAE.T_INTEGER && (env[name] = 0)
        v.ty isa DAE.T_BOOL && (env[name] = false)
      end
    end
    for v in f.locals
      (_aIsArrayVar(v) || !(v.binding isa SOME)) && continue
      env[_aVarName(v, env, ctx)] = _aEval(v.binding.data, env, ctx)
    end
    _aExec!(f.statements, env, ctx)
    result = Any[_aIsArrayVar(v) ? _AUNK : get(env, _aVarName(v, env, ctx), _AUNK) for v in f.outputs]
  catch err
    #= _ABail (control flow), or a construct the interpretation does not know
       (a MethodError of its evaluator): Unknown. =#
    err isa _ABail || OMBackend._fallback(err, :interpretFunction; expect = MethodError)
    result = unknown
  finally
    ctx.depth -= 1
  end
  ctx.memo[key] = result
  return result
end

#= Forget key k and everything under it (its elements and record fields). =#
function _aForget!(env::Dict{String, Any}, k::String)
  for kk in collect(keys(env))
    (kk == k || startswith(kk, k * "[") || startswith(kk, k * "_")) && delete!(env, kk)
  end
  return env
end

function _aAssign!(@nospecialize(lhs), value, env::Dict{String, Any}, ctx::_AEvalContext)
  if lhs isa DAE.CREF
    local k = _aCrefKey(lhs.componentRef, env, ctx)
    k === nothing && throw(_ABail())
    _aStep!(ctx, length(env))
    _aForget!(env, k)
    local fields = _aRecordFields(lhs.ty)
    if fields !== nothing
      #= A record: its fields from the value's (flattened, one level) fields. =#
      if value isa AbstractVector && length(value) == length(fields) && !any(x -> x isa AbstractVector, value)
        for (fname, x) in zip(fields, value)
          env[k * "_" * fname] = x
        end
      end
    elseif lhs.ty isa DAE.T_ARRAY
      if value isa AbstractVector
        for (i, x) in enumerate(value)
          env[k * "[" * string(i) * "]"] = x
        end
      end
    elseif !(value isa AbstractVector)
      env[k] = value
    end
    return nothing
  elseif lhs isa DAE.ASUB && lhs.exp isa DAE.CREF
    local ix = _aSubscripts(collect(lhs.sub), env, ctx)
    local k = _aCrefKey(lhs.exp.componentRef, env, ctx)
    (ix === nothing || k === nothing) && throw(_ABail())
    local ek = k * _aSubsSuffix(ix)
    _aStep!(ctx, length(env))
    _aForget!(env, ek)
    value isa AbstractVector || (env[ek] = value)
    return nothing
  elseif lhs isa DAE.WILD
    return nothing
  end
  throw(_ABail())
end

#= The value a record-typed right-hand side assigns (its fields). =#
function _aRecordValue(@nospecialize(rhs), fields::Vector{String}, env::Dict{String, Any}, ctx::_AEvalContext)
  if rhs isa DAE.CREF
    local k = _aCrefKey(rhs.componentRef, env, ctx)
    k === nothing && return _AUNK
    return Any[get(env, k * "_" * f, _AUNK) for f in fields]
  elseif rhs isa DAE.CALL
    return haskey(ctx.functions, OMBackend.canonicalName(string(rhs.path))) ? _aCall(rhs, env, ctx) :
      Any[_aEval(x, env, ctx) for x in rhs.expLst]
  elseif rhs isa DAE.RECORD
    return Any[_aEval(x, env, ctx) for x in rhs.exps]
  elseif rhs isa DAE.IFEXP
    local c = _aBool(_aEval(rhs.expCond, env, ctx))
    c === true && return _aRecordValue(rhs.expThen, fields, env, ctx)
    c === false && return _aRecordValue(rhs.expElse, fields, env, ctx)
    local a = _aRecordValue(rhs.expThen, fields, env, ctx)
    local b = _aRecordValue(rhs.expElse, fields, env, ctx)
    (a isa AbstractVector && b isa AbstractVector && length(a) == length(b)) || return _AUNK
    return Any[_aSame(x, y) ? x : _AUNK for (x, y) in zip(a, b)]
  end
  return _AUNK
end

#= Join the environments of two branches into env: a variable keeps a value
   both give identically, else it is Unknown (absent). =#
function _aJoin!(env::Dict{String, Any}, a::Dict{String, Any}, b::Dict{String, Any})
  empty!(env)
  for (k, va) in a
    local vb = get(b, k, _AUNK)
    _aSame(va, vb) && (env[k] = va)
  end
  return env
end

function _aExecElse!(els, env::Dict{String, Any}, ctx::_AEvalContext)
  if els isa DAE.ELSE
    _aExec!(els.statementLst, env, ctx)
  elseif els isa DAE.ELSEIF
    _aExecIf!(els.exp, els.statementLst, els.else_, env, ctx)
  end
  return nothing
end

function _aExecIf!(cond, stmts, els, env::Dict{String, Any}, ctx::_AEvalContext)
  local c = _aBool(_aEval(cond, env, ctx))
  if c === true
    _aExec!(stmts, env, ctx)
  elseif c === false
    _aExecElse!(els, env, ctx)
  else
    _aStep!(ctx, 3 * length(env))
    local a = copy(env)
    local b = copy(env)
    _aExec!(stmts, a, ctx)
    _aExecElse!(els, b, ctx)
    _aJoin!(env, a, b)
  end
  return nothing
end

function _aExec!(stmts, env::Dict{String, Any}, ctx::_AEvalContext)
  for s in stmts
    _aStep!(ctx)
    if s isa DAE.STMT_ASSIGN
      local fields = s.exp1 isa DAE.CREF ? _aRecordFields(s.exp1.ty) : nothing
      local value = if fields !== nothing
        _aRecordValue(s.exp, fields, env, ctx)
      elseif s.exp1 isa DAE.CREF && s.exp1.ty isa DAE.T_ARRAY
        s.exp isa DAE.ARRAY ? Any[_aEval(x, env, ctx) for x in s.exp.array] : _AUNK
      else
        _aEval(s.exp, env, ctx)
      end
      _aAssign!(s.exp1, value, env, ctx)
    elseif s isa DAE.STMT_TUPLE_ASSIGN
      s.exp isa DAE.CALL || throw(_ABail())
      local outs = _aCall(s.exp, env, ctx)
      local lhss = collect(s.expExpLst)
      (outs isa AbstractVector && length(outs) >= length(lhss)) || throw(_ABail())
      for (l, v) in zip(lhss, outs)
        (l isa DAE.CREF && (_aRecordFields(l.ty) !== nothing || l.ty isa DAE.T_ARRAY)) && throw(_ABail())
        l isa DAE.WILD || _aAssign!(l, v, env, ctx)
      end
    elseif s isa DAE.STMT_IF
      _aExecIf!(s.exp, s.statementLst, s.else_, env, ctx)
    elseif s isa DAE.STMT_FOR
      local r = s.range
      r isa DAE.RANGE || throw(_ABail())
      local a = _aEval(r.start, env, ctx)
      local st = r.step isa SOME ? _aEval(r.step.data, env, ctx) : 1
      local b = _aEval(r.stop, env, ctx)
      (a isa Integer && st isa Integer && b isa Integer && st != 0) || throw(_ABail())
      for i in a:st:b
        _aStep!(ctx)
        env[s.iter] = i
        _aExec!(s.statementLst, env, ctx)
      end
    elseif s isa DAE.STMT_ASSERT
      #= Dropped with the folded call where it holds on the constants. Where it
         fails, an error-level assert keeps the call unfolded, so that it runs
         (folded, it was a value), and a warning is reported here, once; where
         the condition is not known, the call is kept (it was folded). =#
      local holds = try
        _aBool(_aEval(s.cond, env, ctx))
      catch e
        e isa _ABail || rethrow()
        nothing
      end
      if holds === false && _isWarningLevel(s.level)
        @warn "an assert (level warning) fails in a call folded at the build" message = string(s.msg)
      elseif holds !== true
        throw(_ABail())
      end
    else
      #= while, return, break, a call for its side effects, terminate,
         reinit, ...: not followed. =#
      throw(_ABail())
    end
  end
  return nothing
end

#= The literal of a constant output of declared type ty, or nothing. =#
function _aLiteral(v, @nospecialize(ty))::Union{Exp, Nothing}
  if ty isa DAE.T_REAL
    (v isa Real && !(v isa Bool) && isfinite(Float64(v))) || return nothing
    return RCONST(Float64(v))
  elseif ty isa DAE.T_INTEGER
    (v isa Integer && !(v isa Bool)) || return nothing
    return ICONST(Int(v))
  elseif ty isa DAE.T_BOOL
    v isa Bool || return nothing
    return BCONST(v)
  end
  return nothing
end

"""
    evaluateConstantFunctionOutputs(simCode) -> simCode

Replace the calls of Modelica functions whose scalar output is the same
constant for every value of the call's variable arguments by that literal
(see the top of the file), in the residual equations and the
if-equations' branches; constant equations are propagated afterwards, so
an output that multiplied a derivative drops it (MSL Spice3: the MOSFET
capacitances).
"""
function evaluateConstantFunctionOutputs(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) ||
   hasMetaModel(simCode)) && return simCode
  local fs = Dict{String, MODELICA_FUNCTION}()
  for f in simCode.functions
    f isa MODELICA_FUNCTION && (fs[OMBackend.canonicalName(f.name)] = f)
  end
  isempty(fs) && return simCode
  local ctx = _AEvalContext(fs, Dict{Any, Any}(), 0, 0)
  local replaced = Ref(0)
  #= A call's outputs from its arguments (literals known, anything else
     Unknown), with a step budget of its own. =#
  local outputsOf = function (call::CALL, f::MODELICA_FUNCTION)
    call.attr.isImpure && return nothing
    local env = Dict{String, Any}()
    local args = Any[]
    ctx.steps = 0
    ctx.depth = 0
    try
      #= Record arguments as their fields, as the function's inputs are flattened. =#
      local flat = Exp[]
      local flatten! = a -> (a isa RECORD ? foreach(flatten!, a.exps) : push!(flat, a))
      foreach(flatten!, call.args)
      foreach(a -> _aArgValues!(args, toDAEExp(a), env, ctx), flat)
    catch err
      err isa _ABail || OMBackend._fallback(err, :interpretArguments; expect = MethodError)
      return nothing
    end
    return _aFunction(f, args, ctx)
  end
  local visit = function (e::Exp, acc)
    local call = nothing
    local ix = 0
    if e isa ASUB && e.exp isa CALL && length(e.subs) == 1 && e.subs[1] isa ICONST &&
       !(e.exp.attr.ty isa DAE.T_ARRAY)
      call = e.exp; ix = e.subs[1].value
    elseif e isa TSUB && e.exp isa CALL
      call = e.exp; ix = e.index
    elseif e isa CALL && !(e.attr.ty isa DAE.T_ARRAY)
      call = e; ix = 1
    end
    call === nothing && return (e, true, acc)
    local f = get(fs, OMBackend.canonicalName(string(call.path)), nothing)
    #= A plain call only of a function with one output (a multi-output one is
       read through ASUB/TSUB, whose call must not become output 1). =#
    (f === nothing || (e isa CALL && length(f.outputs) != 1)) && return (e, true, acc)
    (1 <= ix <= length(f.outputs) && !_aIsArrayVar(f.outputs[ix])) || return (e, true, acc)
    local outs = outputsOf(call, f)
    outs === nothing && return (e, true, acc)
    local lit = _aLiteral(outs[ix], f.outputs[ix].ty)
    lit === nothing && return (e, true, acc)
    replaced[] += 1
    return (lit, false, acc)
  end
  local newResiduals = RESIDUAL_EQUATION[]
  for eq in simCode.residualEquations
    local (newExp, _) = traverseExpTopDown(eq.exp, visit, nothing)
    push!(newResiduals, newExp === eq.exp ? eq : typeof(eq)(newExp, eq.source, eq.attr))
  end
  local newIfEquations = IF_EQUATION[]
  for ifEq in simCode.ifEquations
    local newBranches = BRANCH[]
    for branch in ifEq.branches
      local newBranchEqs = RESIDUAL_EQUATION[]
      for brEq in branch.residualEquations
        local (newBrExp, _) = traverseExpTopDown(brEq.exp, visit, nothing)
        push!(newBranchEqs, newBrExp === brEq.exp ? brEq : typeof(brEq)(newBrExp, brEq.source, brEq.attr))
      end
      push!(newBranches, BRANCH(branch.condition, newBranchEqs,
                                branch.identifier, branch.targets, branch.isSingular,
                                branch.matchOrder, branch.equationGraph, branch.sccs,
                                branch.stringToSimVarHT))
    end
    push!(newIfEquations, IF_EQUATION(newBranches))
  end
  replaced[] == 0 && return simCode
  @assign simCode.residualEquations = newResiduals
  @assign simCode.ifEquations = newIfEquations
  @info "[SIMCODE: $(simCode.name): evaluateConstantFunctionOutputs] $(replaced[]) function outputs are constants"
  return propagateConstants(simCode)
end
