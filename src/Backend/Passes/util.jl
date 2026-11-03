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

#= Helpers the passes share: built-in calls, conditions, type queries, names. =#

#= Built-in calls =#

"The call of the built-in `name` with `args`, of type `ty`."
builtinCall(name::Symbol, args::Vector{IR.Expression}, ty::IR.ModelicaType)::IR.CALL =
  IR.CALL(name, args, ty, IR.BUILTIN_CALL)

"`initial()`."
initialCall()::IR.CALL = builtinCall(:initial, IR.Expression[], IR.T_BOOLEAN())

"`change(e)`."
changeCall(e::IR.Expression)::IR.CALL = builtinCall(:change, IR.Expression[e], IR.T_BOOLEAN())

"`edge(e)`."
edgeCall(e::IR.Expression)::IR.CALL = builtinCall(:edge, IR.Expression[e], IR.T_BOOLEAN())

"`pre(e)`, of `e`'s type."
preCall(e::IR.Expression)::IR.CALL = builtinCall(:pre, IR.Expression[e], IR.typeOf(e))

"Whether `e` is a call of one of the built-ins `names`."
isBuiltinCallOf(e::IR.Expression, names::Tuple{Vararg{Symbol}})::Bool =
  e isa IR.CALL && e.attributes.builtin && e.name in names

isInitialCall(e::IR.Expression)::Bool = IR.isBuiltinCall(e, :initial)

#= Conditions, simplified where an operand is a literal =#

"`a and b`; `nothing` is no condition (true)."
function andCondition(a::Union{Nothing, IR.Expression}, b::Union{Nothing, IR.Expression})::Union{Nothing, IR.Expression}
  a === nothing && return b
  b === nothing && return a
  a isa IR.BOOLEAN && return a.value ? b : a
  b isa IR.BOOLEAN && return b.value ? a : b
  return IR.LBINARY(a, IR.AND, b)
end

"`a or b`; `nothing` is no condition."
function orCondition(a::Union{Nothing, IR.Expression}, b::Union{Nothing, IR.Expression})::Union{Nothing, IR.Expression}
  a === nothing && return b
  b === nothing && return a
  a isa IR.BOOLEAN && return a.value ? a : b
  b isa IR.BOOLEAN && return b.value ? b : a
  return IR.LBINARY(a, IR.OR, b)
end

"`not c`."
notCondition(c::IR.Expression)::IR.Expression = c isa IR.BOOLEAN ? IR.BOOLEAN(!c.value) : IR.LUNARY(IR.NOT, c)

"`e == value`, a relation without events."
indexEqualsCondition(e::IR.Expression, value::Int)::IR.Expression = IR.RELATION(e, IR.EQUAL, IR.INTEGER(value), 0)

isFalse(e::Union{Nothing, IR.Expression})::Bool = e isa IR.BOOLEAN && !e.value

#= Types =#

"A discrete-time type: Integer, Boolean or an enumeration, or an array of them."
isDiscreteType(ty::IR.ModelicaType)::Bool =
  IR.elementType(ty) isa Union{IR.T_INTEGER, IR.T_BOOLEAN, IR.T_ENUMERATION}

"A continuous type: Real, or an array of Real."
isRealType(ty::IR.ModelicaType)::Bool = IR.elementType(ty) isa IR.T_REAL

isRecordType(ty::IR.ModelicaType)::Bool = ty isa IR.T_RECORD

#= Expressions =#

"Whether `e` mentions `initial()`."
mentionsInitial(e::IR.Expression)::Bool = IR.anyExp(isInitialCall, e)

"Whether `e` is the built-in variable `time`."
isTime(e::IR.Expression)::Bool = e isa IR.CREF && e.cref.name === :time

"""
    rowMajor(sizes) -> Vector{Vector{Int}}

The subscripts of an array of dimensions `sizes`, the last subscript fastest (the order of an
array value's elements, and of the frontend's scalarized elements: `x[1,1]`, `x[1,2]`, ...).
"""
function rowMajor(sizes::Vector{Int})::Vector{Vector{Int}}
  local out = Vector{Int}[]
  any(==(0), sizes) && return out
  sizehint!(out, prod(sizes))
  local index = ones(Int, length(sizes))
  while true
    push!(out, copy(index))
    local d = length(sizes)
    while d > 0 && index[d] == sizes[d]
      index[d] = 1
      d -= 1
    end
    d == 0 && return out
    index[d] += 1
  end
end

"The flat name of `cref` without its subscripts (`c[2].p.T` and `c.p.T[2]` are `c.p.T`; a slice's part keeps its inner ones, `T_SLICE`): the name of the variable it reads."
unsubscriptedName(cref::IR.ComponentRef)::Symbol =
  !IR.hasSubscripts(cref) ? cref.name :
    (length(cref.parts) == 1 && !(cref.parts[1].ty isa IR.T_SLICE)) ? IR.arrayName(cref.parts[1]) : IR.withoutSubscripts(cref).name

"The subscripts of all parts of `cref`, in order (`c[2].v[3]`: 2, 3; `IR.partSubscripts`): those of the element of an array variable it names."
allSubscripts(cref::IR.ComponentRef)::Vector{IR.Subscript} = IR.Subscript[s for p in cref.parts for s in IR.partSubscripts(p)]

"""
    foldInteger(e)

The value of an Integer expression of literals, `+`, `-` and `*` (`2 - 1` after a loop's
iterator was replaced); `nothing` for another.
"""
function foldInteger(e::IR.Expression)::Union{Nothing, Int}
  e isa IR.INTEGER && return e.value
  if e isa IR.UNARY && e.op === IR.UMINUS
    local v = foldInteger(e.exp)
    return v === nothing ? nothing : -v
  end
  if e isa IR.BINARY && e.op in (IR.ADD, IR.SUB, IR.MUL)
    local (a, b) = (foldInteger(e.exp1), foldInteger(e.exp2))
    (a === nothing || b === nothing) && return nothing
    return e.op === IR.ADD ? a + b : e.op === IR.SUB ? a - b : a * b
  end
  return nothing
end

"""
    substituteIterators(e, values)

`e` with the iterators of `values` (name => value) replaced by their values (`x[i]` with
`i = 2` is the reference to the variable `x[2]`; `x[i - 1]` stays an expression subscript,
`foldSubscripts` folds it).
"""
function substituteIterators(e::IR.Expression, values::Dict{Symbol, Int})::IR.Expression
  isempty(values) && return e
  local replace = x -> begin
    (x isa IR.CREF && length(x.cref.parts) == 1 && !IR.hasSubscripts(x.cref) && haskey(values, x.cref.name)) || return x
    return IR.INTEGER(values[x.cref.name])
  end
  return IR.mapExp(replace, e)
end

"`cref` with its index subscripts folded to literals where they are integer constants."
function foldSubscripts(cref::IR.ComponentRef)::IR.ComponentRef
  local fold = s -> begin
    s isa IR.INDEX || return s
    s.exp isa IR.INTEGER && return s
    local v = foldInteger(s.exp)
    return v === nothing ? s : IR.INDEX(IR.INTEGER(v))
  end
  return IR.ComponentRef(IR.CrefPart[IR.CrefPart(p.name, IR.Subscript[fold(s) for s in p.subscripts], p.ty)
                                     for p in cref.parts])
end

"`e` with `initial()` replaced by `value`."
replaceInitial(e::IR.Expression, value::Bool)::IR.Expression =
  IR.mapExp(x -> isInitialCall(x) ? IR.BOOLEAN(value) : x, e)

"""
    rangeValues(range)

The values of a range of Integer literals, `a:b` or `a:s:b`; `nothing` for another.
"""
function rangeValues(range::IR.Expression)::Union{Nothing, Vector{Int}}
  range isa IR.RANGE || return nothing
  local (a, b) = (foldInteger(range.start), foldInteger(range.stop))
  local s = range.step === nothing ? 1 : foldInteger(range.step)
  (a === nothing || b === nothing || s === nothing || s == 0) && return nothing
  return collect(a:s:b)
end

"The references `e` reads, those inside `pre()` (values before the event) left out."
crefsReadOutsidePre(e::IR.Expression)::Vector{IR.ComponentRef} = crefsReadOutsidePre!(IR.ComponentRef[], e)

"The references `e` reads outside `pre()`, pushed onto `out`."
function crefsReadOutsidePre!(out::Vector{IR.ComponentRef}, e::IR.Expression)::Vector{IR.ComponentRef}
  IR.isBuiltinCall(e, :pre) && return out
  e isa IR.CREF && push!(out, e.cref)
  for c::IR.Expression in IR.children(e)
    crefsReadOutsidePre!(out, c)
  end
  return out
end

"The names of the variables `e` reads, those inside `pre()` (values before the event) left out."
namesReadOutsidePre(e::IR.Expression)::Vector{Symbol} = Symbol[c.name for c in crefsReadOutsidePre(e)]

"The relations in `e`, each once (structurally), in a bottom-up order."
function relationsIn(e::IR.Expression)::Vector{IR.RELATION}
  local out = IR.RELATION[]
  local visit
  visit = x -> begin
    foreach(visit, IR.children(x))
    x isa IR.RELATION && !(x in out) && push!(out, x)
  end
  visit(e)
  return out
end

"""
    hasContinuousOperand(rel, known)

Whether the relation `rel` reads a continuous Real variable that is not in `known` (the
parameters and constants): a relation with a zero crossing.
"""
hasContinuousOperand(rel::IR.Expression, known::Set{Symbol})::Bool =
  IR.anyExp(x -> x isa IR.CREF && !(x.cref.name in known) && isRealType(x.ty), rel)

"The names of the parameters and constants of `variables`."
knownNames(variables::IR.VariableSet)::Set{Symbol} = Set{Symbol}(v.name for v in variables if IR.isKnown(v))

"""
The built-ins whose value jumps where their argument crosses an integer: outside when-clauses and `noEvent()` they
make events (MLS, "Event Triggering Mathematical Functions").
"""
const EVENT_FUNCTIONS = (:integer, :floor, :ceil, :div, :mod, :rem)

"""
    readsContinuousTime(e, variables, known)

Whether `e` reads, outside `pre()`, `time` or a Real variable that is no parameter or constant (`known`) and no
discrete one of `variables`: an event function of it makes events.
"""
function readsContinuousTime(e::IR.Expression, variables::IR.VariableSet, known::Set{Symbol})::Bool
  local hit::Base.RefValue{Bool} = Ref(false)
  IR.mapExpTopDown(e) do x::IR.Expression
    IR.isBuiltinCall(x, :pre) && return (x, false)
    if x isa IR.CREF && isRealType(x.ty) && !(x.cref.name in known)
      (haskey(variables, x.cref.name) && variables[x.cref.name].kind isa IR.DISCRETE) || (hit[] = true)
    end
    return (x, !hit[])
  end
  return hit[]
end

"""
    legacyName(cref)

The previous backend's name of a reference (BDAE's `string` of a DAE reference): the parts
joined by `_`, each subscript in brackets of its own, an enumeration literal as its type's
last name, the literal and its index (`Logic.'1'(4)`): `a.b[1,2]` was `a_b[1][2]`. Where an
order sorted by name must be the previous one, and the names of the simulation code.
"""
function legacyName(cref::IR.ComponentRef)::String
  local io = IOBuffer()
  for (i, p) in enumerate(cref.parts)
    i > 1 && print(io, '_')
    print(io, p.name)
    #= a slice's inner subscripts (`T[1,2]`, `T_SLICE`, splitInner!) each in brackets of its own, as subscripts are =#
    p.ty isa IR.T_SLICE && foreach(k::Int -> print(io, '[', k, ']'), (p.ty::IR.T_SLICE).inner)
    for sub in p.subscripts
      print(io, '[')
      printLegacySubscript(io, sub)
      print(io, ']')
    end
  end
  return String(take!(io))
end

function printLegacySubscript(io::IO, s::IR.Subscript)
  if s isa IR.INDEX && s.exp isa IR.ENUM_LITERAL
    local segments = split(String(s.exp.ty.name), '.')
    print(io, segments[end], '.', s.exp.name, '(', s.exp.index, ')')
  elseif s isa IR.INDEX && (s.exp isa IR.INTEGER || s.exp isa IR.BOOLEAN)
    print(io, s.exp.value)
  else
    print(io, IR.toString(s))
  end
end
