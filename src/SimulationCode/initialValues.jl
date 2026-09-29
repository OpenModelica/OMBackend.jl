#= Initial values propagated forward through the causalized equations. =#

#= Numeric evaluation of a DAE expression at initialization (time = 0) given an
   environment of known variable/parameter values. Returns the Float64 value, or
   `nothing` when the expression is not (yet) fully determined (a free variable, a
   `der`/`pre`, a divide-by-zero, or an unsupported construct). Used by
   propagateInitialValues to forward-evaluate the causalized initial equations. =#
Base.@nospecializeinfer function _evalDAEInit(@nospecialize(e), env::AbstractDict{String, Float64})::Union{Float64, Nothing}
  rec(@nospecialize x) = _evalDAEInit(x, env)
  @match e begin
    DAE.RCONST(r) => Float64(r)
    DAE.ICONST(i) => Float64(i)
    DAE.BCONST(b) => b ? 1.0 : 0.0
    DAE.ENUM_LITERAL(index = idx) => Float64(idx)
    DAE.CREF(componentRef = cr) => get(env, string(cr), nothing)
    DAE.UNARY(DAE.UMINUS(__), e1) => begin local v = rec(e1); v === nothing ? nothing : -v end
    DAE.UNARY(DAE.UMINUS_ARR(__), e1) => rec(e1)
    DAE.BINARY(e1, op, e2) => begin
      local a = rec(e1); local b = rec(e2)
      (a === nothing || b === nothing) && return nothing
      @match op begin
        DAE.ADD(__) => a + b
        DAE.SUB(__) => a - b
        DAE.MUL(__) => a * b
        DAE.DIV(__) => b == 0.0 ? nothing : a / b
        DAE.POW(__) => (a < 0.0 && b != round(b)) ? nothing : Float64(a)^Float64(b)
        _ => nothing
      end
    end
    DAE.IFEXP(c, t, f) => begin
      local cv = _evalDAEInitBool(c, env)
      cv === nothing ? nothing : (cv ? rec(t) : rec(f))
    end
    DAE.CAST(_, e1) => rec(e1)
    DAE.CALL(path = Absyn.IDENT(fn), expLst = args) => _evalDAECallInit(fn, listArray(args), env)
    _ => nothing
  end
end

Base.@nospecializeinfer function _evalDAECallInit(fn::String, a::Vector, env::AbstractDict{String, Float64})::Union{Float64, Nothing}
  local v1 = isempty(a) ? nothing : _evalDAEInit(a[1], env)
  #= Event-control wrappers are init no-ops; `der`/`pre` are free at t0. =#
  if fn == "noEvent"
    return v1
  elseif fn == "smooth"
    return length(a) >= 2 ? _evalDAEInit(a[2], env) : nothing
  elseif fn in ("der", "pre", "previous", "edge", "change", "initial", "sample", "terminal")
    return nothing
  elseif fn in ("max", "min")
    length(a) >= 2 || return nothing
    local x = _evalDAEInit(a[1], env); local y = _evalDAEInit(a[2], env)
    (x === nothing || y === nothing) && return nothing
    return fn == "max" ? max(x, y) : min(x, y)
  end
  v1 === nothing && return nothing
  if fn == "exp"; return exp(v1)
  elseif fn == "log"; return v1 <= 0.0 ? nothing : log(v1)
  elseif fn == "log10"; return v1 <= 0.0 ? nothing : log10(v1)
  elseif fn == "sqrt"; return v1 < 0.0 ? nothing : sqrt(v1)
  elseif fn == "abs"; return abs(v1)
  elseif fn == "sign"; return Float64(sign(v1))
  elseif fn == "floor"; return floor(v1)
  elseif fn == "ceil"; return ceil(v1)
  elseif fn == "integer"; return Float64(round(v1))
  elseif fn == "sin"; return sin(v1)
  elseif fn == "cos"; return cos(v1)
  elseif fn == "tan"; return tan(v1)
  elseif fn == "asin"; return abs(v1) > 1.0 ? nothing : asin(v1)
  elseif fn == "acos"; return abs(v1) > 1.0 ? nothing : acos(v1)
  elseif fn == "atan"; return atan(v1)
  elseif fn == "sinh"; return sinh(v1)
  elseif fn == "cosh"; return cosh(v1)
  elseif fn == "tanh"; return tanh(v1)
  end
  return nothing
end

Base.@nospecializeinfer function _evalDAEInitBool(@nospecialize(e), env::AbstractDict{String, Float64})::Union{Bool, Nothing}
  @match e begin
    DAE.BCONST(b) => b
    DAE.LUNARY(DAE.NOT(__), e1) => begin local v = _evalDAEInitBool(e1, env); v === nothing ? nothing : !v end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      local a = _evalDAEInitBool(e1, env); local b = _evalDAEInitBool(e2, env)
      (a === nothing || b === nothing) ? nothing : (a && b)
    end
    DAE.LBINARY(e1, DAE.OR(__), e2) => begin
      local a = _evalDAEInitBool(e1, env); local b = _evalDAEInitBool(e2, env)
      (a === nothing || b === nothing) ? nothing : (a || b)
    end
    DAE.RELATION(e1, op, e2) => begin
      local a = _evalDAEInit(e1, env); local b = _evalDAEInit(e2, env)
      (a === nothing || b === nothing) && return nothing
      @match op begin
        DAE.LESS(__)      => a < b
        DAE.LESSEQ(__)    => a <= b
        DAE.GREATER(__)   => a > b
        DAE.GREATEREQ(__) => a >= b
        DAE.EQUAL(__)     => a == b
        DAE.NEQUAL(__)    => a != b
        _ => nothing
      end
    end
    DAE.CALL(path = Absyn.IDENT("initial")) => true
    DAE.CALL(path = Absyn.IDENT("noEvent"), expLst = args) => _evalDAEInitBool(listHead(args), env)
    _ => nothing
  end
end

#= Set/replace the `start` attribute of a Real variable-attribute option with the
   resolved init value, preserving the other fields. =#
Base.@nospecializeinfer function _withStartValue(@nospecialize(attrOpt), val::Float64)
  return @match attrOpt begin
    SOME(a && DAE.VAR_ATTR_REAL(__)) => SOME(@set a.start = SOME{DAE.Exp}(DAE.RCONST(val)))
    _ => SOME(DAE.makeRealAttribute(; start = SOME(val), fixed = false))
  end
end

#= Pick the branch of an if-equation active at initialization (time = 0) given the
   current value environment. Conditional branches are tried in order; the first
   whose condition is TRUE wins. Returns the else branch when every condition is
   FALSE, or `nothing` when a needed condition is still undetermined (so the
   if-equation is revisited in a later fixpoint round once more values are known). =#
function _selectActiveInitBranch(ifEq::IF_EQUATION, env::AbstractDict{String, Float64})
  local elseB = nothing
  for branch in ifEq.branches
    if branch.identifier == -1
      elseB = branch
      continue
    end
    local c = _evalDAEInitBool(toDAEExp(branch.condition), env)
    c === nothing && return nothing
    c === true && return branch
  end
  return elseB
end

"""
    propagateInitialValues(simCode) -> SIM_CODE

Forward-propagate initialization values through the causalized equations. Seed an
environment with `time = 0`, constant parameter bindings and explicit start
attributes, then repeatedly solve any equation that has a single still-unknown
variable appearing affinely (the rest evaluating numerically, including `exp` /
`max` / `min`). Each resolved value is attached as the variable's `start`
attribute so the init solver starts from a consistent, finite iterate instead of
defaulting to 0.0 (which makes source-driven flow/pressure networks divide by
zero). Runs at the SimCode layer where every variable is still present.
"""
function propagateInitialValues(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) ||
   hasMetaModel(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  local env = OrderedDict{String, Float64}("time" => 0.0)
  local eqExprs = DAE.Exp[]
  #= Parameter bindings as residuals `name - bind`, plus explicit constant starts. =#
  for (name, idxSv) in ht
    local sv = idxSv[2]
    @match sv.varKind begin
      SimulationCode.PARAMETER(SOME(b)) =>
        push!(eqExprs, DAE.BINARY(DAE.CREF(DAE.CREF_IDENT(name, DAE.T_REAL_DEFAULT, MetaModelica.nil), DAE.T_REAL_DEFAULT),
                                  DAE.SUB(DAE.T_REAL_DEFAULT), toDAEExp(b)))
      _ => nothing
    end
    @match sv.attributes begin
      SOME(DAE.VAR_ATTR_REAL(start = SOME(s))) => begin
        local v = _evalDAEInit(s, env)
        v !== nothing && (env[name] = v)
      end
      _ => nothing
    end
  end
  for eq in simCode.residualEquations
    push!(eqExprs, toDAEExp(eq.exp))
  end
  #= Fixpoint: solve single-free-variable equations affinely via two evaluations.
     If-equation branches join the working set once their condition is decided. =#
  local resolved = OrderedDict{String, Float64}()
  local changed = true
  local rounds = 0
  local trySolve! = function (ex)
    local names = OrderedSet{String}()
    collectCrefNames!(names, ex)
    local free = String[n for n in names if !haskey(env, n)]
    length(free) == 1 || return
    local v = free[1]
    env[v] = 0.0; local b = _evalDAEInit(ex, env)
    env[v] = 1.0; local apb = _evalDAEInit(ex, env)
    delete!(env, v)
    (b === nothing || apb === nothing) && return
    local a = apb - b
    a == 0.0 && return
    local val = -b / a
    isfinite(val) || return
    #= Reject when the equation is not affine in `v`: the two-point slope only
       extrapolates a linear residual, so verify the solution actually zeroes it. =#
    env[v] = val
    local check = _evalDAEInit(ex, env)
    if check === nothing || abs(check) > 1.0e-6 * (1.0 + abs(val))
      delete!(env, v); return
    end
    resolved[v] = val; changed = true
    return
  end
  while changed && rounds < 100
    changed = false; rounds += 1
    for ex in eqExprs
      trySolve!(ex)
    end
    for ifEq in simCode.ifEquations
      local br = _selectActiveInitBranch(ifEq, env)
      br === nothing && continue
      for req in br.residualEquations
        trySolve!(toDAEExp(req.exp))
      end
    end
  end
  isempty(resolved) && return simCode
  #= Attach resolved values as start attributes (skip vars with an explicit start). =#
  local newHT = copy(ht)
  local nAttached = 0
  for (name, val) in resolved
    haskey(ht, name) || continue
    local (idx, sv) = ht[name]
    sv.varKind isa SimulationCode.PARAMETER && continue
    local hasStart = @match sv.attributes begin
      SOME(DAE.VAR_ATTR_REAL(start = SOME(_))) => true
      _ => false
    end
    hasStart && continue
    newHT[name] = (idx, SIMVAR(sv.name, sv.index, sv.varKind, _withStartValue(sv.attributes, val)))
    nAttached += 1
  end
  @assign simCode.stringToSimVarHT = newHT
  @info "[SIMCODE: $(simCode.name): propagateInitialValues] resolved $(length(resolved)), attached $(nAttached) start value(s) (rounds=$(rounds))"
  return simCode
end
