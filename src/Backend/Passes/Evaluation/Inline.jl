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

#= A call inlined: the expression a function's algorithm computes, run symbolically
   (`inlineBody`). =#

"Above this many nodes an inlined body is not built: the values of its if-statements nest."
const INLINED_BODY_NODE_LIMIT = 2000

"""
    inlineBody(f, call) -> Union{Nothing, Expression}

The expression `call` computes, `f`'s algorithm run symbolically (`runBody!`): its assignments to
its only output and to its scalar locals (a local's binding its first value), its if-statements
as if-expressions of the values their branches assign (Buildings' smoothExponential,
`if x < 0 then y := -x*x; else y := x*x; end if`), its asserts skipped; the output's binding
or a record output's field bindings its value before the statements (MSL Complex's
`'constructor'.fromReal`: `output Complex result(re = re, im = im)`, no statement); the inputs
replaced by the call's arguments, an omitted one by its default (a reference to a field or an
element of an input, `R_rel.T`, `e[3]`, by that field or element of the argument). `nothing` for
another body: a loop, an assignment to an
element or a field, a local read before it has a value, an output some path leaves unassigned, a
value over `INLINED_BODY_NODE_LIMIT` nodes.
"""
function inlineBody(f::IR.ModelicaFunction, call::IR.CALL)::Union{Nothing, IR.Expression}
  (f.external === nothing && length(f.outputs) == 1) || return nothing
  length(call.arguments) <= length(f.inputs) || return nothing
  local output = f.outputs[1].name
  local assignable = Set{Symbol}(v.name for v in f.locals if !(v.ty isa Union{IR.T_ARRAY, IR.T_RECORD}))
  push!(assignable, output)
  local own = Set{Symbol}(v.name for v in f.locals)
  push!(own, output)
  local env = Dict{Symbol, IR.Expression}()
  local unknown = Ref(false)
  local inputs = Dict{Symbol, IR.Expression}()
  local substitute = e::IR.Expression -> substituteBody(e, inputs, env, own, unknown)
  #= an omitted argument: the input's default, in terms of the inputs before it (`fromReal(re)`, `im = 0`) =#
  for (i, v) in enumerate(f.inputs)
    if i <= length(call.arguments)
      inputs[v.name] = call.arguments[i]
    else
      v.binding === nothing && return nothing
      inputs[v.name] = substitute(v.binding)
    end
  end
  #= the output's binding, or a record output's field bindings (`output Complex result(re = re, im = im)`, the
     constructors of Complex and its operators, no statement), its value before the statements =#
  local out = f.outputs[1]
  if out.binding !== nothing
    env[output] = substitute(out.binding)
  elseif out.ty isa IR.T_RECORD && !isempty(out.fieldBindings)
    local fields = Dict{Symbol, IR.Expression}(out.fieldBindings)
    all(fd -> haskey(fields, fd.name), out.ty.fields) &&
      (env[output] = IR.RECORD(IR.Expression[substitute(fields[fd.name]) for fd in out.ty.fields], out.ty))
  end
  for v in f.locals
    (v.binding === nothing || !(v.name in assignable)) && continue
    env[v.name] = substitute(v.binding)
  end
  runBody!(env, f.body, assignable, substitute) || return nothing
  local value = get(env, output, nothing)
  (value === nothing || unknown[] || expressionLargerThan(value, INLINED_BODY_NODE_LIMIT)) && return nothing
  return value
end

"""
    substituteBody(e, inputs, env, own, unknown) -> Expression

`e` of a function's body with its inputs replaced by the call's arguments (`inputs`; a field or
an element of an input by that of the argument) and its scalar locals and output by their values
so far (`env`); `unknown[]` set where it reads one of the function's own variables (`own`: its
locals and output) without a value (an array or record local, `c[1]`: no value is kept), or an
input's field the argument does not have.
"""
function substituteBody(e::IR.Expression, inputs::Dict{Symbol, IR.Expression}, env::Dict{Symbol, IR.Expression},
                        own::Set{Symbol}, unknown::Ref{Bool})::IR.Expression
  return IR.mapExpTopDown(e) do x
    x isa IR.CREF || return (x, true)
    local parts = x.cref.parts
    if parts[1].name in own
      (length(parts) == 1 && isempty(parts[1].subscripts) && haskey(env, parts[1].name)) && return (env[parts[1].name], false)
      unknown[] = true
      return (x, false)
    end
    local v = get(inputs, parts[1].name, nothing)
    v === nothing && return (x, true)
    local out = v
    isempty(parts[1].subscripts) || (out = subscripted(out, parts[1].subscripts, length(parts) == 1 ? x.ty : parts[1].ty))
    for k in 2:length(parts)
      out = fieldOf(out, parts[k].name, parts[k].ty)
      out === nothing && (unknown[] = true; return (x, false))
      isempty(parts[k].subscripts) || (out = subscripted(out, parts[k].subscripts, k == length(parts) ? x.ty : parts[k].ty))
    end
    return (expandWhole(out), false)
  end
end

"""
    runBody!(env, statements, assignable, substitute) -> Bool

The statements run symbolically on `env` (the values of the locals and the output so far): an
assignment to an assignable variable as a whole its value; an if-statement each branch on a copy,
then each variable a branch assigns the if-expression of the branches' values (a branch that
does not assign it: the value before; none before and a branch without one: no value); an assert
skipped. False for another statement.
"""
function runBody!(env::Dict{Symbol, IR.Expression}, statements::Vector{IR.Statement}, assignable::Set{Symbol},
                  substitute::F)::Bool where {F}
  for s in statements
    if s isa IR.ASSIGNMENT
      (s.lhs isa IR.CREF && length(s.lhs.cref.parts) == 1 && isempty(s.lhs.cref.parts[1].subscripts) &&
       s.lhs.cref.name in assignable) || return false
      env[s.lhs.cref.name] = substitute(s.rhs)
    elseif s isa IR.IF_STATEMENT
      local conditions = IR.Expression[substitute(b.condition) for b in s.branches]
      local branches = Dict{Symbol, IR.Expression}[]
      for b in s.branches
        local branch = copy(env)
        runBody!(branch, b.body, assignable, substitute) || return false
        push!(branches, branch)
      end
      local otherwise = copy(env)
      runBody!(otherwise, s.elseBody, assignable, substitute) || return false
      local assigned = union(Set{Symbol}(), (keys(b) for b in branches)..., keys(otherwise))
      for name in assigned
        local values = Union{Nothing, IR.Expression}[get(b, name, nothing) for b in branches]
        local last = get(otherwise, name, nothing)
        if last === nothing || any(isnothing, values)
          delete!(env, name)
          continue
        end
        env[name] = foldr((k, rest) -> values[k] === rest ? rest : IR.IF_EXPRESSION(conditions[k], values[k], rest, IR.typeOf(rest)),
                          eachindex(values); init = last)
      end
    elseif !(s isa IR.ASSERT)
      return false
    end
  end
  return true
end
