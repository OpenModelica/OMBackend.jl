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
# Author: John Tinnerholm (johti17)
=#
module CodeGenerationUtil

import DataStructures
using DataStructures: OrderedSet
import MacroTools
using ModelingToolkit
using LinearAlgebra

import ...SimCode
import ...OMBackend
import ..AlgorithmicCodeGeneration
import ...@BACKEND_LOGGING
import ...COMPONENT_SEPARATOR

import Symbolics
import Symbolics.RuntimeGeneratedFunctions
import SymbolicUtils
import OMRuntimeExternalC

const SC = SimCode

"""
Return string containing the OSMC copyright stuff.
"""
function copyrightString()

  strOut = string("#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
=#")
  return strOut
end


#= ==================================================================================== =#
#= Zero-crossing functions                                                              =#
#= ==================================================================================== =#

realBinary(a::SC.Exp, op::SC.OpKind, b::SC.Exp)::SC.Exp = SC.BINARY(a, op, b)
realCall(name::Symbol, args::SC.Exp...)::SC.Exp = SC.CALL(name, SC.Exp[args...], SC.TYPE_REAL(), SC.CALL_BUILTIN)

"""
integer(X) is a step (often cast to Real); a continuous zero-crossing on it never triggers,
so `when integer(X) <> Y` froze (Modelica.Blocks.Sources.Pulse): the argument X of an
integer() call (under a cast), else nothing.
"""
function _zcIntArg(e::SC.Exp)::Union{Nothing, SC.Exp}
  local inner = e isa SC.CAST ? e.exp : e
  return SC.isBuiltin(inner, :integer) && length(inner.args) == 1 ? inner.args[1] : nothing
end

"""
A relation with an integer() operand as a smooth zero-crossing on X: floor(X) >= Y is
X >= Y for an integer-valued Y, floor(X) > Y is X >= Y + 1; `nothing` for another
condition.
"""
function _zcRewriteIntegerRel(cond::SC.Exp)::Union{Nothing, SC.Exp}
  cond isa SC.RELATION || return nothing
  local (e1, op, e2) = (cond.exp1, cond.op, cond.exp2)
  local (x1, x2) = (_zcIntArg(e1), _zcIntArg(e2))
  (x1 === nothing && x2 === nothing) && return nothing
  local one = SC.RCONST(1.0)
  local S = (a, b) -> realBinary(a, SC.OP_SUB, b)
  local A = (a, b) -> realBinary(a, SC.OP_ADD, b)
  if x1 !== nothing
    op === SC.OP_GREATER && return S(A(e2, one), x1)
    op === SC.OP_GREATEREQ && return S(e2, x1)
    op === SC.OP_LESS && return S(x1, e2)
    op === SC.OP_LESSEQ && return S(x1, A(e2, one))
  else
    op === SC.OP_LESS && return S(A(e1, one), x2)
    op === SC.OP_LESSEQ && return S(e1, x2)
    op === SC.OP_GREATER && return S(x2, e1)
    op === SC.OP_GREATEREQ && return S(x2, A(e1, one))
  end
  return nothing
end

"""
An operand of a logical operator in a zero-crossing condition: a relation (or a logical
expression of them) its zero-crossing function, a Boolean value (a parameter, a discrete)
1 - 2b, -1 where true.
"""
function _zeroCrossingOperand(e::SC.Exp)::SC.Exp
  (e isa Union{SC.RELATION, SC.LBINARY, SC.LUNARY} || SC.isBuiltin(e, :change)) && return transformToZeroCrossingCondition(e)
  return realBinary(SC.RCONST(1.0), SC.OP_SUB, realBinary(SC.RCONST(2.0), SC.OP_MUL, e))
end

"""
    transformToZeroCrossingCondition(cond)

A zero-crossing function `f` of a condition, true where `f` goes from positive to negative
(the direction of SciML's `affect!`): `e1 - e2` of `<` and `<=` (true when e1 falls below e2),
`e2 - e1` of `>` and `>=` (an upcrossing of e1 above e2), so that the caller passes
`affect_neg! = nothing` and a value oscillating around the other does not fire twice (the
bouncing ball's Zeno pattern). A relation with integer() operands on its argument; `change(r)`
the relation's; `not` negated, `and` the maximum (true where both are), `or` the minimum (true
where either is): `a - b` of the operands themselves did not change sign (Buildings' weather
data readers' `canRepeatWeatherFile and modTimAux > pre(tNext)`). The larger or smaller of two
by max() or min(): an if-expression held each operand twice, and a nesting grew exponentially
(MSL Digital's registers).
"""
function transformToZeroCrossingCondition(cond::SC.Exp)::SC.Exp
  local rewritten = _zcRewriteIntegerRel(cond)
  rewritten === nothing || return rewritten
  if cond isa SC.RELATION
    cond.op in (SC.OP_GREATER, SC.OP_GREATEREQ) && return realBinary(cond.exp2, SC.OP_SUB, cond.exp1)
    return realBinary(cond.exp1, SC.OP_SUB, cond.exp2)
  elseif SC.isBuiltin(cond, :change)
    return transformToZeroCrossingCondition(first(cond.args))
  elseif cond isa SC.BINARY
    return realBinary(cond.exp1, SC.OP_SUB, cond.exp2)
  elseif cond isa SC.LUNARY && cond.op === SC.OP_NOT
    return SC.UNARY(SC.OP_UMINUS, _zeroCrossingOperand(cond.exp))
  elseif cond isa SC.LBINARY && cond.op === SC.OP_AND
    return realCall(:max, _zeroCrossingOperand(cond.exp1), _zeroCrossingOperand(cond.exp2))
  elseif cond isa SC.LBINARY && cond.op === SC.OP_OR
    return realCall(:min, _zeroCrossingOperand(cond.exp1), _zeroCrossingOperand(cond.exp2))
  end
  return cond
end

#= ==================================================================================== =#
#= Operators                                                                            =#
#= ==================================================================================== =#

"""
    opKindToJuliaOperator(k)

The Julia operator of a SimCode operator: element-wise ones broadcast (`.*`, `./`, `.^`; the
same on scalars), a scalar with an array `.+`, `.-`, `./`, `.^` and `*`, `/`; the scalar product
`vectorDot` (Julia's `*` has no vector-vector method), the matrix products and the matrix power
`*`, `^`; `and` `&`, `or` `||`.
"""
function opKindToJuliaOperator(k::SC.OpKind)::Union{Symbol, Expr}
  k in (SC.OP_ADD, SC.OP_ADD_EW) && return :+
  k in (SC.OP_SUB, SC.OP_SUB_EW, SC.OP_UMINUS) && return :-
  k in (SC.OP_MUL, SC.OP_MUL_SCALAR_ARRAY, SC.OP_MUL_ARRAY_SCALAR, SC.OP_MATRIX_PRODUCT,
        SC.OP_MUL_VECTOR_MATRIX, SC.OP_MUL_MATRIX_VECTOR) && return :*
  k in (SC.OP_DIV, SC.OP_DIV_ARRAY_SCALAR) && return :/
  k in (SC.OP_POW, SC.OP_POW_MATRIX) && return :^
  k === SC.OP_MUL_EW && return Symbol(".*")
  k in (SC.OP_DIV_EW, SC.OP_DIV_SCALAR_ARRAY) && return Symbol("./")
  k in (SC.OP_POW_EW, SC.OP_POW_SCALAR_ARRAY, SC.OP_POW_ARRAY_SCALAR) && return Symbol(".^")
  k in (SC.OP_ADD_SCALAR_ARRAY, SC.OP_ADD_ARRAY_SCALAR) && return Symbol(".+")
  k in (SC.OP_SUB_SCALAR_ARRAY, SC.OP_SUB_ARRAY_SCALAR) && return Symbol(".-")
  k === SC.OP_SCALAR_PRODUCT && return :(OMBackend.CodeGeneration.vectorDot)
  k === SC.OP_AND && return :(&)
  k === SC.OP_OR && return :(||)
  k === SC.OP_NOT && return :(!)
  k === SC.OP_LESS && return :(<)
  k === SC.OP_LESSEQ && return :(<=)
  k === SC.OP_GREATER && return :(>)
  k === SC.OP_GREATEREQ && return :(>=)
  k === SC.OP_EQUAL && return :(==)
  k === SC.OP_NEQUAL && return :(!=)
  OMBackend.unsupported("operator", k)
end

"""
    builtinCallToJulia(e, lower)

A call outside the MTK equations (callbacks, the DifferentialEquations code generator) with
its arguments lowered by `lower`: `Integer(e)` the identity (an enumeration is its index
already), `String` from the arguments' types, a built-in by its implementation
(MODELICA_BUILTIN_FUNCTIONS: `sample()` in a when body has no other binding in the model's
module), another function by its name.
"""
function builtinCallToJulia(e::SC.CALL, lower::F)::Expr where {F}
  local name = String(e.name)
  (e.kind === SC.CALL_BUILTIN && e.name === :Integer) && return lower(first(e.args))
  (e.kind === SC.CALL_BUILTIN && e.name === :String) && return AlgorithmicCodeGeneration.modelicaStringCall(e.args, lower)
  local args = Any[lower(x) for x in e.args]
  local builtinSym = get(AlgorithmicCodeGeneration.MODELICA_BUILTIN_FUNCTIONS, name, nothing)
  if builtinSym !== nothing
    local qualified = Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(builtinSym))
    return quote $(qualified)($(args...)) end
  end
  return quote $(e.name)($(args...)) end
end

"""
  Get the name of the active model.
  The default is the original model.
"""
function getActiveModel(simCode)
  local activeModelName = simCode.activeModel
  for sm in simCode.subModels
    if sm.name == activeModelName
      return sm
    end
  end
  return activeModelName
end

"Whether `name` names a function of the model."
_isSimCodeFunctionName(name::AbstractString, simCode::SC.SIM_CODE)::Bool =
  any(f -> f.name == OMBackend.canonicalName(name), simCode.functions)

"""
  Removes all comments from a given exp
"""
function stripComments(ex::Expr)::Expr
  return Base.remove_linenums!(ex)
end

"""
Iterative equivalent of `MacroTools.postwalk`. Visits each subexpression
bottom-up, replacing each node with the result of `f(rebuilt_node)`.
Used in place of the recursive walker for huge generated function-body
ASTs (e.g. wrappers for Modelica.Utilities.Strings.scanToken family),
where the recursive form blows past Julia's runtime stack guard.
"""
function _iterativePostwalk(f, root)
  isa(root, Expr) || return f(root)
  local todo = Any[(root, false)]
  local newMap = Base.IdDict{Any,Any}()
  #= Dedup by object identity: generated ASTs are DAGs with shared sub-Expr
     nodes, so expanding every reference re-walks shared subtrees and blows up
     exponentially. Each unique node is expanded and rebuilt once; `f` is a pure
     function of the node, so all references resolve to the same cached result. =#
  local expanded = Base.IdSet{Any}()
  while !isempty(todo)
    local (node, processedChildren) = pop!(todo)
    if !(node isa Expr)
      haskey(newMap, node) || (newMap[node] = f(node))
      continue
    end
    if !processedChildren
      node in expanded && continue
      push!(expanded, node)
      push!(todo, (node, true))
      for a in node.args
        push!(todo, (a, false))
      end
    else
      #= Lazy rebuild: only allocate a new args vector + Expr when a child
         actually changed; otherwise pass the original node to `f`. `f` is a
         pure function of structure, so a structurally identical node yields
         the same result -> emitted code is unchanged. =#
      local changed = false
      for a in node.args
        if get(newMap, a, a) !== a
          changed = true
          break
        end
      end
      local rebuilt = changed ? Expr(node.head, Any[get(newMap, a, a) for a in node.args]...) : node
      newMap[node] = f(rebuilt)
    end
  end
  return get(newMap, root, root)
end

"""
 Removes all redundant blocks from a generated expression
"""
function stripBeginBlocks(e)::Expr
  _iterativePostwalk(MacroTools.unblock, e)
end

"""
    resolveAliasedCref(fullName, simCode)

A reference to a variable alias elimination removed that survived in an expression (missed by
the substitution): `(true, expr)` with its representative (negated for a negated alias),
`(false, nothing)` for a name the alias map does not have.
"""
function resolveAliasedCref(fullName::String, simCode::SC.SIM_CODE)::Tuple{Bool, Union{Nothing, Expr}}
  for entry in simCode.aliasMap
    entry.eliminatedName == fullName || continue
    local repEntry = get(simCode.stringToSimVarHT, entry.representativeName, nothing)
    repEntry === nothing && continue
    local sym = Symbol(last(repEntry).name)
    return (true, entry.negated ? :(-$(sym)) : quote $(sym) end)
  end
  return (false, nothing)
end

"""
  Helper used by the lowered RSUB(`re`) form. Falls back through:
    Base.Complex — `.re`
    NamedTuple / structs with `:re` — `getproperty`
    Symbolics.Num wrapping a Complex term — `real(num)`
  Annotated as `Base.Complex` because the module rebinds the bare name
  `Complex` to a Modelica function wrapper via
  `createModelicaFunctionWrapper(:Complex, …)`, which would otherwise
  shadow the type and turn this method definition into "invalid type
  for argument".
"""
@inline _recordFieldRe(x::Base.Complex) = x.re
@inline _recordFieldRe(x::Tuple) = x[1]
@inline _recordFieldRe(x) = hasproperty(x, :re) ? getproperty(x, :re) : real(x)

"""
A record's field `name`, the `ix`th: by position in a tuple (a function returning a
record returns its fields in order: the MSL Media `setState_psX(...).T`), by name
otherwise (a record constructor's NamedTuple).
"""
@inline _recordField(x::Tuple, name::Symbol, ix::Integer) = ix >= 1 ? x[ix] : getproperty(x, name)
@inline _recordField(x, name::Symbol, ::Integer) = getproperty(x, name)

"""
  Companion to `_recordFieldRe` for the imaginary part (RSUB(`im`)).
  Same `Base.Complex` reason: avoid the module-scope shadow of `Complex`.
"""
@inline _recordFieldIm(x::Base.Complex) = x.im
@inline _recordFieldIm(x::Tuple) = x[2]
@inline _recordFieldIm(x) = hasproperty(x, :im) ? getproperty(x, :im) : imag(x)

"""
  The value of a constant: a literal's, or an expression of constants and parameters evaluated
  (operators and calls of constants: MSL Fluid's tanks' max(1.0, 1e-15); a negation, a
  relation, an if-expression, a parameter: MSL Blocks' RSFlipFlop pre_u_start = not Qini).
"""
function evalConstant(c::SC.Exp, simCode::SC.SIM_CODE)
  c isa Union{SC.BCONST, SC.ICONST, SC.RCONST, SC.SCONST} && return c.value
  c isa Union{SC.BINARY, SC.LBINARY, SC.UNARY, SC.CALL, SC.LUNARY, SC.RELATION, SC.IFEXP, SC.EXP_CREF} &&
    return OMBackend.CodeGeneration.evalExpression(c, simCode)
  OMBackend.unsupported("not a constant", c)
end

"The value of a literal."
function evalConstant(c::SC.Exp)
  c isa Union{SC.BCONST, SC.ICONST, SC.RCONST, SC.SCONST} && return c.value
  OMBackend.unsupported("not a constant", c)
end

"""
  The value of a bound parameter's binding (evalConstant); UnsupportedLowering for another
  variable or a binding that is no constant.
"""
function evalSimCodeParameter(v::SC.SIMVAR, simCode::SC.SIM_CODE)
  (v.varKind isa SC.PARAMETER && v.varKind.bindExp !== nothing) ||
    OMBackend.unsupported("a constant value of a variable other than a bound parameter", v.name)
  return evalConstant(v.varKind.bindExp, simCode)
end

"""
    _substituteBoundParameters(exp, simCode; skipNames, shouldEval, seen, keepTunable)

`exp` with its parameters replaced by their bindings, themselves so substituted; `shouldEval`
cleared where it reads a variable, a parameter without a binding (a free one an initial
equation determines), a name the table does not have, a cyclic binding. With `keepTunable`, a
tunable parameter (withTunableParameters) stays a reference, read at run time.
"""
function _substituteBoundParameters(exp::SC.Exp, simCode::SC.SIM_CODE;
                                    skipNames::OrderedSet{String} = OrderedSet{String}(),
                                    shouldEval::Base.RefValue{Bool} = Ref(true),
                                    seen::OrderedSet{String} = OrderedSet{String}(),
                                    keepTunable::Bool = false)::SC.Exp
  local replace = e::SC.Exp -> begin
    e isa SC.EXP_CREF || return e
    local key = SC.crefName(e.cref)
    key in skipNames && return e
    if keepTunable && OMBackend.isTunableParameter(key)
      shouldEval[] = false
      return e
    end
    local entry = get(simCode.stringToSimVarHT, key, nothing)
    if entry === nothing || SC.isStateOrAlgebraic(last(entry))
      shouldEval[] = false
      return e
    end
    local k = last(entry).varKind
    local bindExp = k isa Union{SC.PARAMETER, SC.ARRAY_PARAMETER} ? k.bindExp : nothing
    if bindExp === nothing || key in seen
      shouldEval[] = false
      return e
    end
    push!(seen, key)
    local resolved = _substituteBoundParameters(bindExp, simCode; skipNames, shouldEval, seen, keepTunable)
    delete!(seen, key)
    return resolved
  end
  return SC.mapExp(replace, exp)
end

"""
    equationSides(eq) -> Tuple{Exp, Exp}

The sides of an equation: an equation's and an array equation's, a residual's `(exp, 0.0)`.
"""
equationSides(eq::SC.EQUATION)::Tuple{SC.Exp, SC.Exp} = (eq.lhs, eq.rhs)
equationSides(eq::SC.ARRAY_EQUATION)::Tuple{SC.Exp, SC.Exp} = (eq.left, eq.right)
equationSides(eq::SC.RESIDUAL_EQUATION)::Tuple{SC.Exp, SC.Exp} = (eq.exp, SC.RCONST(0.0))
equationSides(eq::SC.Equation) = error("equationSides: no (lhs, rhs) for $(typeof(eq))")

"""
The equations `equationSides` takes.
"""
hasEquationSides(eq::SC.Equation)::Bool = eq isa Union{SC.EQUATION, SC.ARRAY_EQUATION, SC.RESIDUAL_EQUATION}

"""
    isParametricOnlyEquation(eq, simCode) -> Bool

Whether an initial equation involves parameters only (no state, algebraic or discrete
variable): it determines parameter values, it is no initial condition. An if-equation (CDL
SunRiseSet, the hydronic networks) has no sides: false.
"""
function isParametricOnlyEquation(eq::SC.Equation, simCode::SC.SIM_CODE)::Bool
  hasEquationSides(eq) || return false
  local ht = simCode.stringToSimVarHT
  local names = OrderedSet{String}()
  foreach(e -> SC.crefNames!(names, e), equationSides(eq))
  return !any(names) do n
    local entry = get(ht, n, nothing)
    entry !== nothing && (SC.isStateOrAlgebraic(last(entry)) || SC.isDiscrete(last(entry)))
  end
end

function _substituteExprValues(expr, valMap::Dict{Symbol, Float64})
  if expr isa Number
    return Float64(expr)
  end
  if expr isa Symbol
    if expr == :t
      return 0.0
    end
    if haskey(valMap, expr)
      return valMap[expr]
    end
    #= Defensive fallback: a missing symbol would hit `eval` at module
       scope and raise `UndefVarError`. Returning 0.0 (which is `false`
       for boolean thresholds) keeps initial-condition evaluation
       robust. The outer try/catch in `evalInitialCondition` already
       defaults to `true` on any remaining failure, so this is defense
       in depth for fold-promoted parameters whose valMap entry was
       skipped. =#
    return 0.0
  end
  if !(expr isa Expr)
    return expr
  end
  if expr.head == :call
    local fname = expr.args[1]
    #= Pattern: varName(t) -> substitute with value from valMap =#
    if fname isa Symbol && length(expr.args) == 2 && expr.args[2] == :t
      if haskey(valMap, fname)
        return valMap[fname]
      end
      #= `varName(t)` where `varName` is not in valMap: recursion below
         would rebuild `Expr(:call, :varName, 0.0)` and hit
         `UndefVarError` on final eval. Defensive fallback to 0.0, same
         policy as the bare-symbol branch above. =#
      return 0.0
    end
    #= Recurse into call arguments =#
    local newArgs = Any[fname]
    for i in 2:length(expr.args)
      push!(newArgs, _substituteExprValues(expr.args[i], valMap))
    end
    return Expr(:call, newArgs...)
  end
  #= Broadcast call `f.(args)`: the callee position is code, not a variable. =#
  if expr.head == :. && length(expr.args) == 2 && expr.args[2] isa Expr && expr.args[2].head == :tuple
    return Expr(:., expr.args[1], _substituteExprValues(expr.args[2], valMap))
  end
  #= Qualified name `A.b`: not a variable reference. =#
  if expr.head == :. && length(expr.args) == 2 && expr.args[2] isa QuoteNode
    return expr
  end
  #= Generic recursion for other Expr types =#
  local newExpr = Expr(expr.head)
  for arg in expr.args
    push!(newExpr.args, _substituteExprValues(arg, valMap))
  end
  return newExpr
end

isIntOrBool(exp::SC.Exp)::Bool =
  exp isa Union{SC.ICONST, SC.BCONST} || (exp isa SC.EXP_CREF && exp.ty isa Union{SC.TYPE_INTEGER, SC.TYPE_BOOL})

"""
`writeEqsToFile(elems::Vector{Expr}, filename)`
  Used for logging.
"""
function writeEqsToFile(elems::Vector{Expr}, filename)
  local buffer = IOBuffer()
  try
    for e in elems
      e = stripComments(e)
      e = stripBeginBlocks(e)
      local eStr = string(e)
      eqStr = replace(eStr, "~" => "=")
      eqStr = replace(eqStr, "(t)" => "")
      eqStr = replace(eqStr, "Differential" => "der")
      println(buffer, eqStr)
    end
  catch e
    @error string("Failed writing the model to the file:",  filename)
    rethrow()
  end
  println(buffer, "------------------------------------")
  println(buffer, "Statistics:")
  println(buffer, "Number of items:" * string(length(elems)))
  println(buffer, "------------------------------------")
  write(filename, String(take!(buffer)))
end

"""
  Whether a condition reads no discrete variable: `time` alone, or with a continuous variable
  (a relation against `time` such as `time >= t_next` is a time event, a zero-crossing of
  `time - rhs`, even with a discrete operand: an edge-triggered ContinuousCallback rather than a
  level-triggered discrete callback that re-fires every step); a name the table does not have.
  Parameters are constants and discretes change at events only: a condition of those (`(tLH >
  0 or tHL > 0) and change(x)`) is an event-driven discrete callback.
"""
function isContinuousCondition(cond::SC.Exp, simCode::SC.SIM_CODE)::Bool
  local names = SC.allCrefNames(cond)
  (length(names) == 1 && only(names) == "time") && return true
  local ht = simCode.stringToSimVarHT
  local continuous = false
  for n in names
    if n == "time" || !haskey(ht, n)
      continuous = true
      continue
    end
    local v = last(ht[n])
    continuous = continuous || (!SC.isDiscrete(v) && !SC.isParameter(v))
  end
  return continuous
end

"""
Replace a variable with an optional naming convention specified by the prefix and suffix arguments
"""
function replaceVars(sym::Symbol; kwargs...)
  if Base.isoperator(sym)
    sym
  elseif sym === :t
    sym
  else
    local vStr = string(sym)
    #= If the symbol is not a simulation variable (e.g. a function or module name),
       pass it through unchanged rather than crashing with a KeyError. =#
    if !haskey(kwargs[:ht], vStr)
      return sym
    end
    #= Lookup the variable. =#
    local htEntry = kwargs[:ht][vStr]
    local sVar = last(htEntry)
    local isStateOrAlg = SimCode.isStateOrAlgebraic(sVar)
    if isStateOrAlg
      #= useIndexedU: for ContinuousCallback conditions the trial state is passed as u,
         so generate u[idx] instead of integrator[:sym] to correctly detect sign changes. =#
      if get(kwargs, :useIndexedU, false)
        local idx = first(htEntry)
        #= useMTKIdx: use runtime MTK variable ordering instead of the SimCode index.
           MTK may reorder unknowns (e.g., algebraic vars before differential),
           so _mtk_idx (built from unknowns(reducedSystem)) gives the correct u position. =#
        if get(kwargs, :useMTKIdx, false)
          :(u[get(_mtk_idx, $vStr, $idx)])
        else
          :(u[$idx])
        end
      else
        :($(Meta.parse(string(kwargs[:integratorCref],
                              kwargs[:prefix],
                              ":$sym",
                              kwargs[:suffix]))))
      end
    else #Parameter or Discrete.
      :($(Meta.parse(string(kwargs[:integratorCref],
                            ".ps",
                            kwargs[:prefix],
                            ":$sym",
                            kwargs[:suffix]))))
    end
  end
end
function replaceVars(expr::Expr; kwargs...)
  Expr(expr.head,
       map((x) -> replaceVars(x; kwargs...), expr.args)...)
end
replaceVars(x; kwargs...) = x


"""
  Utility function to check if a given expression contains a specific datatype.
"""
function exprContainsDatatype(ex, datatype)
  if ex isa Symbol
    return false
  elseif ex isa datatype
    return true
  elseif ex isa Expr
    return any(arg -> exprContainsDatatype(arg, datatype), ex.args)
  else
    return false
  end
end

"""
    flattenRecordCallArg(arg, simCode)

The fields of a record argument of a call, as their flattened variables (`<name>_<field>`);
empty for another argument. A connector-typed Complex reference
(`Modelica.ComplexBlocks.Interfaces.ComplexInput u`) the same, if one of its fields is a
variable.
"""
function flattenRecordCallArg(arg::SC.Exp, simCode::SC.SIM_CODE)::Vector{Symbol}
  (arg isa SC.EXP_CREF && arg.ty isa SC.TYPE_COMPLEX) || return Symbol[]
  local baseName = SC.crefName(arg.cref)
  local flat = Symbol[Symbol(baseName * COMPONENT_SEPARATOR * f) for f in arg.ty.fieldNames]
  arg.ty.isRecord && return flat
  return any(s -> haskey(simCode.stringToSimVarHT, String(s)), flat) ? flat : Symbol[]
end

"""
  Extract common hvcat subexpressions from a list of equation Exprs.
  Returns (modifiedEquations, preambleAssignments) where preamble contains
  local variable assignments for deduplicated hvcat calls.
"""
function extractCommonHvcats(equations::Vector)
  local allHvcats = Dict{String, Int}()
  local hvcatExprs = Dict{String, Expr}()
  for eq in equations
    local found = Dict{String, Bool}()
    _collectHvcats!(eq, allHvcats, hvcatExprs, found)
  end
  local duplicates = filter(p -> p.second >= 2, allHvcats)
  if isempty(duplicates)
    return equations, Expr[]
  end
  local replacements = Dict{String, Symbol}()
  local preamble = Expr[]
  local idx = 0
  for (key, _) in duplicates
    local varName = Symbol("_hv_", idx)
    replacements[key] = varName
    push!(preamble, :(local $varName = $(hvcatExprs[key])))
    idx += 1
  end
  local modifiedEqs = [_replaceHvcats(deepcopy(eq), replacements) for eq in equations]
  return modifiedEqs, preamble
end

function _collectHvcats!(expr, counts::Dict{String,Int}, exprs::Dict{String,Expr}, found::Dict{String,Bool})
  if !(expr isa Expr)
    return
  end
  if expr.head == :call && length(expr.args) >= 1 && expr.args[1] == :hvcat
    local key = string(expr)
    if !haskey(found, key)
      found[key] = true
      counts[key] = get(counts, key, 0) + 1
      if !haskey(exprs, key)
        exprs[key] = expr
      end
    end
  end
  for arg in expr.args
    _collectHvcats!(arg, counts, exprs, found)
  end
end

function _replaceHvcats(expr, replacements::Dict{String,Symbol})
  if !(expr isa Expr)
    return expr
  end
  if expr.head == :call && length(expr.args) >= 1 && expr.args[1] == :hvcat
    local key = string(expr)
    if haskey(replacements, key)
      return replacements[key]
    end
  end
  for i in eachindex(expr.args)
    expr.args[i] = _replaceHvcats(expr.args[i], replacements)
  end
  return expr
end

#= ===================================================================
   Utilities migrated out of MTK_CodeGeneration.jl. These are pure
   predicates / collectors / Expr transforms — no MTK-specific codegen.
   =================================================================== =#

_isCodegenNameChar(c::Char)::Bool =
  isletter(c) || isdigit(c) || c == '_' || c == 'ˍ' || c == '[' || c == ']' || c == '"'

function equationMentionsVariableName(eqStr::AbstractString, varName::AbstractString)::Bool
  isempty(varName) && return false
  local startIdx = firstindex(eqStr)
  while true
    local match = findnext(varName, eqStr, startIdx)
    match === nothing && return false
    local firstIdx = first(match)
    local lastIdx = last(match)
    local beforeOk = firstIdx == firstindex(eqStr) || !_isCodegenNameChar(eqStr[prevind(eqStr, firstIdx)])
    local afterOk = lastIdx == lastindex(eqStr) || !_isCodegenNameChar(eqStr[nextind(eqStr, lastIdx)])
    beforeOk && afterOk && return true
    startIdx = nextind(eqStr, firstIdx)
  end
end

"Whether `e` reads a derivative."
containsDerCall(e::SC.Exp)::Bool = SC.anyExp(x -> x isa SC.DER || SC.isBuiltin(x, :der), e)

"""
    startValueVariableNames(simCode) -> Vector{String}

The variables whose start values go into the initial-value map: the flagged irreducibles,
then every state not among them (a state keeps its start value whether or not it is
irreducible: one MTK may reduce still needs one, and a model with structural transitions has
no initialization problem to find it).
"""
function startValueVariableNames(simCode::SC.SIM_CODE)::Vector{String}
  local names = String[vn for vn in simCode.irreducibleVariables]
  local seen = Set{String}(names)
  for (name, (_, simVar)) in simCode.stringToSimVarHT
    simVar.varKind isa SC.STATE || continue
    name in seen && continue
    push!(names, name)
    push!(seen, name)
  end
  return names
end

"Whether one of `vars` has a start value."
function hasExplicitStartValue(vars::Vector, simCode::SC.SIM_CODE)::Bool
  local ht = simCode.stringToSimVarHT
  return any(v -> haskey(ht, v) && last(ht[v]).attributes.start !== nothing, vars)
end

"The names of `vars` of `fixed = true` (without a start too: the default start 0 is fixed, MLS 4.9.1)."
function fixedStartVarNames(vars::Vector, simCode::SC.SIM_CODE)::Vector{String}
  local ht = simCode.stringToSimVarHT
  return String[last(ht[v]).name for v in vars if haskey(ht, v) && SC.isTrueLiteral(last(ht[v]).attributes.fixed)]
end

_isLiteralBind(exp::SC.Exp)::Bool = exp isa Union{SC.RCONST, SC.ICONST, SC.BCONST, SC.SCONST, SC.ENUM_LITERAL}

#= ==================================================================================== =#
#= The functions the model calls                                                        =#
#= ==================================================================================== =#

"The names of the functions an expression (an equation's, a statement's) calls, built-ins too."
function collectCalledFunctionNames!(names::OrderedSet{String}, e::SC.Exp)::OrderedSet{String}
  SC.foreachExp(x -> x isa SC.CALL && push!(names, String(x.name)), e)
  return names
end
collectCalledFunctionNames!(names::OrderedSet{String}, eq::SC.RESIDUAL_EQUATION) = collectCalledFunctionNames!(names, eq.exp)
function collectCalledFunctionNames!(names::OrderedSet{String}, eq::Union{SC.EQUATION, SC.ARRAY_EQUATION})
  foreach(e -> collectCalledFunctionNames!(names, e), equationSides(eq))
  return names
end
function collectCalledFunctionNames!(names::OrderedSet{String}, s::SC.ASSIGN)
  collectCalledFunctionNames!(names, s.left)
  return collectCalledFunctionNames!(names, s.right)
end
collectCalledFunctionNames!(names::OrderedSet{String}, s::SC.REINIT) = collectCalledFunctionNames!(names, s.value)
collectCalledFunctionNames!(names::OrderedSet{String}, s::SC.NORETCALL) = collectCalledFunctionNames!(names, s.exp)
collectCalledFunctionNames!(names::OrderedSet{String}, ::Union{SC.Equation, SC.Statement}) = names

function collectCalledFunctionNames!(names::OrderedSet{String}, simCode::SC.SIM_CODE)::OrderedSet{String}
  foreach(eq -> collectCalledFunctionNames!(names, eq), simCode.residualEquations)
  foreach(eq -> collectCalledFunctionNames!(names, eq), simCode.initialEquations)
  for ifEq in simCode.ifEquations, b in ifEq.branches
    collectCalledFunctionNames!(names, b.condition)
    foreach(eq -> collectCalledFunctionNames!(names, eq), b.residualEquations)
  end
  for w in simCode.whenEquations
    collectCalledFunctionNames!(names, w.whenEquation.condition)
    foreach(s -> collectCalledFunctionNames!(names, s), w.whenEquation.body)
  end
  return names
end

"""
    getRHSVariables(s) -> Vector{String}

The names a when-clause's statement reads (the previous backend's order: the reverse of
their first occurrence): an assignment's value, a reinit's value, a call's, an assert's
condition, message and level, a terminate's message.
"""
getRHSVariables(s::SC.ASSIGN)::Vector{String} = SC.allCrefNames(s.right)
getRHSVariables(s::SC.REINIT)::Vector{String} = SC.allCrefNames(s.value)
getRHSVariables(s::SC.NORETCALL)::Vector{String} = SC.allCrefNames(s.exp)
getRHSVariables(s::SC.ASSERT)::Vector{String} = vcat(SC.allCrefNames(s.condition), SC.allCrefNames(s.message), SC.allCrefNames(s.level))
getRHSVariables(s::SC.TERMINATE)::Vector{String} = SC.allCrefNames(s.message)
getRHSVariables(::SC.Statement)::Vector{String} = String[]

function _resolveModelicaCallTargets(expr)
  if !(expr isa Expr)
    return expr
  end
  if expr.head === :call && length(expr.args) >= 2 &&
     expr.args[1] == :(Base.invokelatest) && expr.args[2] isa Symbol
    local funcSym = expr.args[2]
    local registry = :(OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[$(QuoteNode(funcSym))])
    local rest = Any[_resolveModelicaCallTargets(a) for a in expr.args[3:end]]
    return Expr(:call, expr.args[1], registry, rest...)
  end
  return Expr(expr.head, Any[_resolveModelicaCallTargets(a) for a in expr.args]...)
end

function _literalRefName(expr::Expr)
  expr.head === :ref || return nothing
  isempty(expr.args) && return nothing
  expr.args[1] isa Symbol || return nothing
  local parts = String[string(expr.args[1])]
  for idx in expr.args[2:end]
    local v = if idx isa Integer
      idx
    elseif idx isa QuoteNode && idx.value isa Integer
      idx.value
    else
      return nothing
    end
    push!(parts, string("[", v, "]"))
  end
  return join(parts)
end

function _renameAlgIdentifiers(expr, names::OrderedSet{String}, prefix::String)
  if expr isa Symbol
    local s = string(expr)
    s == "time" && return expr
    return s in names ? Symbol(prefix * s) : expr
  elseif expr isa Expr
    #= The growth guard of an array whose elements are locals of their own
       here (x[1] := 1.0 is var"_alg_x[1]" = 1.0): no array to grow. =#
    if expr.head === :call && length(expr.args) >= 2 && expr.args[2] isa Symbol &&
       endswith(string(expr.args[1]), "ensureAlgArrayLength!") && !(string(expr.args[2]) in names)
      return nothing
    end
    if expr.head === :ref
      local refName = _literalRefName(expr)
      if refName !== nothing && refName in names
        return Symbol(prefix * refName)
      end
    end
    if expr.head === :call && length(expr.args) >= 1
      local newArgs = Any[expr.args[1]]
      for i in 2:length(expr.args)
        push!(newArgs, _renameAlgIdentifiers(expr.args[i], names, prefix))
      end
      return Expr(:call, newArgs...)
    else
      return Expr(expr.head, map(a -> _renameAlgIdentifiers(a, names, prefix), expr.args)...)
    end
  end
  return expr
end

function _renameAlgIdentifiers(expr, names::OrderedSet{String})
  return _renameAlgIdentifiers(expr, names, "_alg_")
end

"The literal start value of a variable as a Float64 (0.0 for none or one not a literal)."
function _readStartAttributeAsLiteral(sv::SC.SIMVAR)::Float64
  local s = sv.attributes.start
  s isa Union{SC.RCONST, SC.ICONST} && return Float64(s.value)
  s isa SC.BCONST && return s.value ? 1.0 : 0.0
  return 0.0
end

using ExportAll
@exportAll()

end #= module CodeGenerationUtil =#
