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

#= Initial values propagated forward through the causalized equations. =#

"""
    initialValue(e, env)

The value of `e` at the initialization (time 0) from the values `env` knows: literals (a
Boolean as 0 or 1, an enumeration literal as its index), arithmetic, if-expressions, casts, the
elementary functions in their domains, `noEvent`, `smooth`, `max`, `min`; `nothing` for a free
variable, `der()`, `pre()`, a division by zero, an array element, another construct.
"""
function initialValue(e::Exp, env::AbstractDict{String, Float64})::Union{Nothing, Float64}
  e isa RCONST && return e.value
  e isa ICONST && return Float64(e.value)
  e isa BCONST && return e.value ? 1.0 : 0.0
  e isa ENUM_LITERAL && return Float64(e.index)
  e isa EXP_CREF && return get(env, crefName(e.cref), nothing)
  if e isa UNARY && e.op === OP_UMINUS
    local v = initialValue(e.exp, env)
    return v === nothing ? nothing : -v
  elseif e isa BINARY
    local (a, b) = (initialValue(e.exp1, env), initialValue(e.exp2, env))
    (a === nothing || b === nothing) && return nothing
    e.op === OP_ADD && return a + b
    e.op === OP_SUB && return a - b
    e.op === OP_MUL && return a * b
    e.op === OP_DIV && return b == 0.0 ? nothing : a / b
    e.op === OP_POW && return (a < 0.0 && b != round(b)) ? nothing : a^b
    return nothing
  elseif e isa IFEXP
    local c = initialTruth(e.cond, env)
    return c === nothing ? nothing : initialValue(c ? e.thenExp : e.elseExp, env)
  elseif e isa CAST
    return initialValue(e.exp, env)
  elseif e isa CALL && e.kind === CALL_BUILTIN
    return builtinInitialValue(e.name, e.args, env)
  end
  return nothing
end

const INITIAL_FUNCTIONS = Dict{Symbol, Tuple{Function, Function}}(
  :exp => (exp, _ -> true), :log => (log, x -> x > 0.0), :log10 => (log10, x -> x > 0.0), :sqrt => (sqrt, x -> x >= 0.0),
  :abs => (abs, _ -> true), :sign => (x -> Float64(sign(x)), _ -> true), :floor => (floor, _ -> true),
  :ceil => (ceil, _ -> true), :integer => (floor, _ -> true), :sin => (sin, _ -> true), :cos => (cos, _ -> true),
  :tan => (tan, _ -> true), :asin => (asin, x -> abs(x) <= 1.0), :acos => (acos, x -> abs(x) <= 1.0),
  :atan => (atan, _ -> true), :sinh => (sinh, _ -> true), :cosh => (cosh, _ -> true), :tanh => (tanh, _ -> true))

"A built-in call's value at the initialization; the event operators (`pre`, `edge`, `sample`, ...) have none."
function builtinInitialValue(name::Symbol, args::Vector{Exp}, env::AbstractDict{String, Float64})::Union{Nothing, Float64}
  name === :noEvent && return isempty(args) ? nothing : initialValue(args[1], env)
  name === :smooth && return length(args) >= 2 ? initialValue(args[2], env) : nothing
  name in (:der, :pre, :previous, :edge, :change, :initial, :sample, :terminal) && return nothing
  if name in (:max, :min)
    length(args) >= 2 || return nothing
    local (x, y) = (initialValue(args[1], env), initialValue(args[2], env))
    (x === nothing || y === nothing) && return nothing
    return name === :max ? max(x, y) : min(x, y)
  end
  local f = get(INITIAL_FUNCTIONS, name, nothing)
  (f === nothing || isempty(args)) && return nothing
  local v = initialValue(args[1], env)
  (v === nothing || !f[2](v)) && return nothing
  return f[1](v)
end

"The truth of a condition at the initialization (`initial()` holds); `nothing` if it is not known."
function initialTruth(e::Exp, env::AbstractDict{String, Float64})::Union{Nothing, Bool}
  e isa BCONST && return e.value
  if e isa LUNARY && e.op === OP_NOT
    local v = initialTruth(e.exp, env)
    return v === nothing ? nothing : !v
  elseif e isa LBINARY && e.op in (OP_AND, OP_OR)
    local (a, b) = (initialTruth(e.exp1, env), initialTruth(e.exp2, env))
    (a === nothing || b === nothing) && return nothing
    return e.op === OP_AND ? (a && b) : (a || b)
  elseif e isa RELATION
    local (a, b) = (initialValue(e.exp1, env), initialValue(e.exp2, env))
    (a === nothing || b === nothing) && return nothing
    return compareValues(a, e.op, b)
  elseif e isa CALL && e.kind === CALL_BUILTIN && e.name === :initial
    return true
  elseif e isa CALL && e.kind === CALL_BUILTIN && e.name === :noEvent && !isempty(e.args)
    return initialTruth(e.args[1], env)
  end
  return nothing
end

"The branch of an if-equation active at the initialization; `nothing` while a condition before it is not known."
function activeInitialBranch(ifEq::IF_EQUATION, env::AbstractDict{String, Float64})::Union{Nothing, BRANCH}
  local elseBranch = nothing
  for b in ifEq.branches
    b.identifier == ELSE_BRANCH && (elseBranch = b; continue)
    local c = initialTruth(b.condition, env)
    c === nothing && return nothing
    c && return b
  end
  return elseBranch
end

"""
    solveAffine!(env, resolved, e)

The value of the one variable of the residual `e` that `env` does not know, if `e` is affine
in it (two evaluations give the slope; the solution must zero `e`): into `env` and
`resolved`. Whether one was found.
"""
function solveAffine!(env::OrderedDict{String, Float64}, resolved::OrderedDict{String, Float64}, e::Exp)::Bool
  local free = String[n for n in crefNames!(OrderedSet{String}(), e) if !haskey(env, n)]
  length(free) == 1 || return false
  local v = only(free)
  env[v] = 0.0
  local b = initialValue(e, env)
  env[v] = 1.0
  local apb = initialValue(e, env)
  delete!(env, v)
  (b === nothing || apb === nothing || apb - b == 0.0) && return false
  local value = -b / (apb - b)
  isfinite(value) || return false
  env[v] = value
  local check = initialValue(e, env)
  if check === nothing || abs(check) > 1.0e-6 * (1.0 + abs(value))
    delete!(env, v)
    return false
  end
  resolved[v] = value
  return true
end

"""
    propagateInitialValues(simCode)

The initialization's values propagated forward through the causalized equations, as start
values of the variables without one: from `time = 0`, the parameters' bindings and the
constant Real starts, every residual (and the active branch of an if-equation, once its
condition is known) with one unknown variable appearing affinely is solved, to a fixed point
(at most 100 rounds). The initialization starts from a consistent, finite iterate instead of
0.0 (source-driven flow networks divided by zero). A tunable parameter is seeded with its
compiled value too. Nothing for a model with structural modes or translated again at its
structural changes.
"""
function propagateInitialValues(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  local env = OrderedDict{String, Float64}("time" => 0.0)
  local residuals = Exp[]
  for (name, (_, v)) in ht
    if v.varKind isa PARAMETER && v.varKind.bindExp !== nothing
      push!(residuals, BINARY(EXP_CREF(SimCref(Symbol(name)), TYPE_REAL()), OP_SUB, v.varKind.bindExp))
    end
    if v.elementType isa TYPE_REAL && v.attributes.start !== nothing
      local s = initialValue(v.attributes.start, env)
      s === nothing || (env[name] = s)
    end
  end
  foreach(eq -> push!(residuals, eq.exp), simCode.residualEquations)
  local resolved = OrderedDict{String, Float64}()
  local (changed, rounds) = (true, 0)
  while changed && rounds < 100
    (changed, rounds) = (false, rounds + 1)
    for e in residuals
      solveAffine!(env, resolved, e) && (changed = true)
    end
    for ifEq in simCode.ifEquations
      local b = activeInitialBranch(ifEq, env)
      b === nothing && continue
      for eq in b.residualEquations
        solveAffine!(env, resolved, eq.exp) && (changed = true)
      end
    end
  end
  isempty(resolved) && return simCode
  local newHT = copy(ht)
  for (name, value) in resolved
    haskey(ht, name) || continue
    local (index, v) = ht[name]
    (v.varKind isa PARAMETER || v.attributes.start !== nothing) && continue
    local a = withAttributes(v.attributes; start = RCONST(value),
                             fixed = v.elementType isa TYPE_REAL ? v.attributes.fixed : BCONST(false))
    newHT[name] = (index, SIMVAR(v.name, v.index, v.varKind, a, v.elementType))
  end
  @debug "[SIMCODE: $(simCode.name): propagateInitialValues] resolved $(length(resolved)) (rounds = $(rounds))"
  return update(simCode; stringToSimVarHT = newHT)
end
