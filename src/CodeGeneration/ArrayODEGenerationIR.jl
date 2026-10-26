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
The OMJLIR front half of the array code generation (2026-10-25): a model lowered with its arrays
kept and reduced in array form (Passes.IndexReduction.reduceIndexArrays!, its derivative unknowns
solved in array form: solvedInArrays) to the variables and classes of this module, whose back half
(_emitModule) generates the module as for a flat model.

  - Its for-equations are the classes, their iterators the slots; a scalar equation reading
    elements by literal subscripts is a class of one instance (re-rolled with its like ones).
  - A state's derivative link `$STATEDER.x[i] = $DER.x[i]` is der(x[i]) = $DER.x[i]; the dummy
    states, dummy derivatives and derivative variables are algebraic variables.
  - Start values, fixed attributes and parameter values are evaluated on the model's element view
    (Evaluation.StartValues of scalarizeArrays), each element's own.

  - Records reach the code through functions: a record argument by its fields (as the generated
    functions take it), an array argument as the array of its elements, a record result as the
    tuple of its fields; an element of an array expression by distributing its subscripts
    (`selectElement`) or indexing its value. If-equations as the flat model's; parameters with
    fixed = false unknowns of the initialization.

  - When-equations as the flat model's (`_addWhenIR!`): their triggers the condition, sample() in it.

Not (yet), NotSupported (the caller takes the scalarized path): array equations, algorithms, a
when-equation of the passes' initial or structural kinds, sample() outside a when-condition, a
fixed start on a dummy state (the reduced initialization's), whole arrays in scalar equations,
records of records through functions.
=#

const OIR = OMJLIR
const _IRD = Passes.IndexReduction
const _EV = Passes.Evaluation

"""
    generateArrayODECode(model::OMJLIR.Model, modelName; functions) -> (modelName, moduleExpr) or (nothing, reason)

The array-preserving module of an OMJLIR model with its arrays kept (reduced in array form), or `nothing` and the
reason the model is outside the supported scope. `functions`: the frontend's function list of the model's Modelica
functions.
"""
function generateArrayODECode(model::OIR.Model, modelName::String; functions::Union{Nothing, FunctionList} = nothing)
  try
    return (modelName, _generateIR(model, modelName; functions = functions))
  catch err
    err isa Union{NotSupported, _IRD.NotHandled} || rethrow(err)
    return (nothing, err isa NotSupported ? err.msg : sprint(showerror, err))
  end
end

"An OMJLIR expression or equation in a reason message (printing must not fail)."
_strIR(@nospecialize(x)) = try
  replace(Base.first(string(x), 200), r"\s*\n\s*" => " ")
catch
  string("<", nameof(typeof(x)), ">")
end

"The value type of an OMJLIR type's elements: :real, :int (Integer, enumerations), :bool, or :other."
function _vtypeIR(ty::OIR.ModelicaType)::Symbol
  local el::OIR.ModelicaType = ty isa OIR.T_ARRAY ? ty.elementType : ty
  el isa OIR.T_REAL && return :real
  el isa Union{OIR.T_INTEGER, OIR.T_ENUMERATION} && return :int
  el isa OIR.T_BOOLEAN && return :bool
  return :other
end

"The dimensions of an OMJLIR variable (none for a scalar); NotSupported for one of a size not known."
function _dimsIR(v::OIR.Variable)::Vector{Int}
  v.ty isa OIR.T_ARRAY || return Int[]
  return Int[d isa OIR.DIM_SIZE ? d.size : ns("$(v.name) of a dimension not known") for d in v.ty.dimensions]
end

"An element's value as its variable's type has it (Evaluation gives Float64)."
_typedValue(x::Float64, vtype::Symbol) = vtype == :int ? round(Int, x) : vtype == :bool ? x != 0 : x

"""
The parameters an OMJLIR variable's binding reads.
"""
function _bindingDepsIR(g::Gen, v::OIR.Variable)::Vector{String}
  local deps::Vector{String} = String[]
  v.binding === nothing && return deps
  OIR.foreachExp(v.binding) do x::OIR.Expression
    x isa OIR.CREF || return nothing
    local nm::String = (!OIR.hasSubscripts(x.cref) || haskey(g.vars, String(x.cref.name))) ? String(x.cref.name) :
      String(Passes.unsubscriptedName(x.cref))
    local info::Union{Nothing, VarInfo} = get(g.vars, nm, nothing)
    info !== nothing && info.kind == :param && !(nm in deps) && push!(deps, nm)
    nothing
  end
  return deps
end

"""
A scalar parameter's binding as code over the parameters it reads (`_bindingCode`); an array parameter's is not
expressed: its value then does not follow a changed dependency.
"""
function _bindingCodeIR(g::Gen, info::VarInfo)
  isempty(info.dims) || ns("an array parameter's binding")
  local v::OIR.Variable = (info.var::IRVariable).variable
  local deps::Vector{String} = _bindingDepsIR(g, v)
  local locals::Vector{Expr} = Expr[:(local $(g.paramSyms[d])::$(typeof(_paramValue!(g, d))) = vals[$(QuoteNode(g.paramSyms[d]))])
                                    for d in deps]
  local value::Any = jl(g, toIR(g, v.binding, Ctx(Dict{String, Int}(), Int[])))
  return :((vals) -> begin
    $(locals...)
    $value
  end)
end

function _binaryOpIR(op::OIR.Op)::Symbol
  op in (OIR.ADD, OIR.ADD_EW) && return :+
  op in (OIR.SUB, OIR.SUB_EW) && return :-
  op in (OIR.MUL, OIR.MUL_EW) && return :*
  op in (OIR.DIV, OIR.DIV_EW) && return :/
  op in (OIR.POW, OIR.POW_EW) && return :^
  ns("operator $(op) on arrays")
end

function _relationOpIR(op::OIR.Op)::Symbol
  op == OIR.LESS && return :<
  op == OIR.LESSEQ && return :<=
  op == OIR.GREATER && return :>
  op == OIR.GREATEREQ && return :>=
  op == OIR.EQUAL && return :(==)
  op == OIR.NEQUAL && return :!=
  ns("relation operator $(op)")
end

"An assert's message as an expression; its text where the array path cannot evaluate it."
function _messageIRx(g::Gen, m::OIR.Expression, ctx::Ctx)::IR
  try
    return toIR(g, m, _with(ctx; noEvent = true))
  catch err
    err isa NotSupported || rethrow(err)
    return Lit(_strIR(m))
  end
end

function toIR(g::Gen, e::OIR.Expression, ctx::Ctx)::IR
  if e isa OIR.INTEGER || e isa OIR.REAL || e isa OIR.BOOLEAN || e isa OIR.STRING
    return Lit(e.value)
  elseif e isa OIR.ENUM_LITERAL
    return Lit(e.index)
  elseif e isa OIR.CAST
    return toIR(g, e.exp, ctx)
  elseif e isa OIR.CREF
    return _crefIRx(g, e, ctx)
  elseif e isa OIR.UNARY
    e.op == OIR.UMINUS || ns("unary operator $(e.op)")
    return Op(:neg, IR[toIR(g, e.exp, ctx)])
  elseif e isa OIR.BINARY
    e.ty isa OIR.T_STRING && return Op(:string, IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
    e.op == OIR.SCALAR_PRODUCT && return _scalarProductIRx(g, e, ctx)
    return Op(_binaryOpIR(e.op), IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
  elseif e isa OIR.LUNARY
    e.op == OIR.NOT || ns("logical operator $(e.op)")
    return Op(:!, IR[toIR(g, e.exp, ctx)])
  elseif e isa OIR.LBINARY
    local lop::Symbol = e.op == OIR.AND ? :&& : e.op == OIR.OR ? :|| : ns("logical operator $(e.op)")
    return Op(lop, IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
  elseif e isa OIR.RELATION
    local op::Symbol = _relationOpIR(e.op)
    local l::IR = toIR(g, e.exp1, ctx)
    local r::IR = toIR(g, e.exp2, ctx)
    (ctx.noEvent || !(_continuous(g, l) || _continuous(g, r))) && return Op(op, IR[l, r])
    op in (:(==), :!=) && ns("equality relation on continuous variables")
    ctx.nrel[] += 1
    return Rel(op, l, r, ctx.nrel[])
  elseif e isa OIR.IF_EXPRESSION
    return Op(:if, IR[toIR(g, e.condition, ctx), toIR(g, e.thenExp, ctx), toIR(g, e.elseExp, ctx)])
  elseif e isa OIR.TUPLE_ELEMENT
    e.ty isa Union{OIR.T_ARRAY, OIR.T_RECORD} && ns("an array or record output: $(_strIR(e))")
    return Op(:index, IR[toIR(g, e.exp, ctx), Lit(_outputPosition(OIR.typeOf(e.exp), e.index))])
  elseif e isa OIR.CALL
    return _callIRx(g, e, ctx)
  elseif e isa OIR.SUBSCRIPTED
    return _subscriptedIRx(g, e, ctx)
  elseif e isa OIR.RECORD_ELEMENT
    return _recordElementIRx(g, e, ctx)
  end
  ns("expression $(nameof(typeof(e))): $(_strIR(e))")
end

"""
A reference of an OMJLIR model: a for-equation's iterator its slot, `time`, a state's derivative link
(`\$STATEDER.x[i]`) der(x[i]), a variable's element by its subscripts (all of them, each an index).
"""
function _crefIRx(g::Gen, e::OIR.CREF, ctx::Ctx)::IR
  local cr::OIR.ComponentRef = e.cref
  if length(cr.parts) == 1 && !OIR.hasSubscripts(cr)
    local own::String = String(cr.name)
    haskey(ctx.iters, own) && return Slot(ctx.iters[own])
    (own == "time" && !haskey(g.vars, own)) && return TimeIR()
  end
  #= an element variable of its own (an array split for index reduction, `x[2]`), else the array it subscripts =#
  local whole::Bool = !OIR.hasSubscripts(cr) || haskey(g.vars, String(cr.name)) || haskey(g.derOf, String(cr.name))
  local name::String = whole ? String(cr.name) : String(Passes.unsubscriptedName(cr))
  if !(haskey(g.vars, name) || haskey(g.derOf, name))
    #= an element of an array split along its inner dimensions: the slice's (`slicedCref`) =#
    local sliced::Union{Nothing, OIR.ComponentRef} = _IRD.slicedCref(cr, (n::Symbol) -> haskey(g.vars, String(n)) || haskey(g.derOf, String(n)))
    sliced === nothing || return _crefIRx(g, OIR.CREF(sliced, e.ty), ctx)
  end
  local subscripts::Vector{OIR.Subscript} = whole ? OIR.Subscript[] : Passes.allSubscripts(cr)
  local subs::Vector{IR} = IR[s isa OIR.INDEX ? toIR(g, s.exp, ctx) : ns("a slice of $(name)") for s in subscripts]
  return _elementRefIR(g, name, subs)
end

"""
The element `subs` (all of its subscripts, or none for a scalar) of the variable `name`; of a state derivative's link
(`\$STATEDER.x`) der of the state's element.
"""
function _elementRefIR(g::Gen, name::String, subs::Vector{IR})::IR
  local der::Bool = haskey(g.derOf, name)
  local target::String = der ? g.derOf[name] : name
  local info::Union{Nothing, VarInfo} = get(g.vars, target, nothing)
  info === nothing && ns("unknown variable $(target)")
  isempty(subs) && !isempty(info.dims) && ns("the array $(target) as a whole in an equation of scalars")
  length(subs) == length(info.dims) || ns("partial subscripts on $(target)")
  der && info.kind != :state && ns("der of non-state $(target)")
  return Ref(target, subs, der)
end

"""
A scalar product of two vectors of a known size (MultiBody's `tau = -frame_b.t * e`): the sum of the products of
their elements (`_subscriptedIRx`).
"""
function _scalarProductIRx(g::Gen, e::OIR.BINARY, ctx::Ctx)::IR
  local ty1::OIR.ModelicaType = OIR.typeOf(e.exp1)
  local ty2::OIR.ModelicaType = OIR.typeOf(e.exp2)
  (ty1 isa OIR.T_ARRAY && ty2 isa OIR.T_ARRAY && length(ty1.dimensions) == 1 && ty1.dimensions[1] isa OIR.DIM_SIZE) ||
    ns("a scalar product of $(_strIR(e))")
  local terms::Vector{IR} = IR[]
  for k in 1:ty1.dimensions[1].size
    local index::Vector{OIR.Subscript} = OIR.Subscript[OIR.INDEX(OIR.INTEGER(k))]
    push!(terms, Op(:*, IR[toIR(g, OIR.SUBSCRIPTED(e.exp1, index, ty1.elementType), ctx),
                           toIR(g, OIR.SUBSCRIPTED(e.exp2, index, ty2.elementType), ctx)]))
  end
  isempty(terms) && return Lit(0.0)
  return foldl((a::IR, b::IR) -> Op(:+, IR[a, b]), terms)
end

"""
An element of an array expression (`{155, 155, 155}[\$i1]`, `(R.T * v)[i]`, `resolve1(R, v)[i]`): of element-wise
arithmetic, a product, a literal by literal subscripts the element distributed (`Evaluation.selectElement`); of an
array variable its element; of an array literal the element at the subscripts (the literal array indexed, else the
vector of its elements); of a function's array result or a record's array field the value indexed.
"""
function _subscriptedIRx(g::Gen, e::OIR.SUBSCRIPTED, ctx::Ctx)::IR
  local field::OIR.Expression = _fieldExpression(e)
  field === e || return toIR(g, field, ctx)
  all(s::OIR.Subscript -> s isa OIR.INDEX, e.subscripts) || ns("a slice of $(_strIR(e))")
  local picked::OIR.Expression = _EV.selectElement(e.exp, e.subscripts, e.ty)
  picked isa OIR.SUBSCRIPTED || return toIR(g, picked, ctx)
  local base::OIR.Expression = _fieldExpression(picked.exp)
  base === picked.exp || return toIR(g, OIR.SUBSCRIPTED(base, picked.subscripts, e.ty), ctx)
  if base isa OIR.CREF
    #= the reference subscripted (an element variable of its own where the array was split: `R.T[1,2]`) =#
    local last::OIR.CrefPart = base.cref.parts[end]
    local parts::Vector{OIR.CrefPart} = OIR.CrefPart[base.cref.parts[1:end - 1]...,
                                                     OIR.CrefPart(last.name, _composedSubscripts(last.subscripts, picked.subscripts), last.ty)]
    return _crefIRx(g, OIR.CREF(OIR.ComponentRef(parts), e.ty), ctx)
  end
  local subs::Vector{IR} = IR[toIR(g, (s::OIR.INDEX).exp, ctx) for s in picked.subscripts]
  if base isa OIR.ARRAY
    local literal::Union{Nothing, Array} = _literalArrayIR(base)
    literal === nothing || return Op(:index, IR[Lit(literal); subs])
    length(subs) == 1 || ns("an array of arrays subscripted: $(_strIR(e))")
    return Op(:index, IR[Op(:vect, IR[toIR(g, x, ctx) for x in base.elements]), subs[1]])
  end
  #= another array (a built-in's: `cat(2, {{0.0}, ...}, ...)[\$i1, \$i2]`) by subscripts not all literal: its value
     (its elements by literal subscripts, `_valueIRx`) indexed =#
  local literalSubscripts::Bool = all(s::OIR.Subscript -> (s::OIR.INDEX).exp isa OIR.INTEGER, picked.subscripts)
  (_isFunctionCall(g, base) || base isa OIR.RECORD_ELEMENT || !literalSubscripts) || ns("an element of $(_strIR(base))")
  return Op(:index, IR[_valueIRx(g, base, ctx); subs])
end

"""
A record's field read by its position (`absoluteRotation(R1, R2)[1]`: the frontend reads a field of a record-valued
call so) as the record element; a field of a record literal or variable the field itself (`_fieldOf`); another
expression as it is.
"""
function _fieldExpression(e::OIR.Expression)::OIR.Expression
  if e isa OIR.SUBSCRIPTED && length(e.subscripts) == 1 && e.subscripts[1] isa OIR.INDEX
    local position::OIR.Expression = (e.subscripts[1]::OIR.INDEX).exp
    local rty::OIR.ModelicaType = OIR.typeOf(e.exp)
    if position isa OIR.INTEGER && rty isa OIR.T_RECORD
      local f::OIR.Field = rty.fields[position.value]
      return _fieldExpression(OIR.RECORD_ELEMENT(e.exp, position.value, f.name, f.ty))
    end
  end
  if e isa OIR.RECORD_ELEMENT && e.exp isa Union{OIR.CREF, OIR.RECORD}
    local recordType::OIR.ModelicaType = OIR.typeOf(e.exp)
    recordType isa OIR.T_RECORD && return _fieldOf(e.exp, e.index, recordType.fields[e.index])
  end
  return e
end

"""
The subscripts of a reference `x[outer]` subscripted again by `inner`: a slice (`phi1[1:2]`) or a whole dimension
indexed by the next inner subscript (a literal range's element, or `first + step * (k - 1)`), an index kept, the
inner subscripts left after them appended.
"""
function _composedSubscripts(outer::Vector{OIR.Subscript}, inner::Vector{OIR.Subscript})::Vector{OIR.Subscript}
  local out::Vector{OIR.Subscript} = OIR.Subscript[]
  local next::Int = 1
  for s in outer
    if s isa OIR.INDEX
      push!(out, s)
      continue
    end
    next <= length(inner) || ns("a slice subscripted partly")
    local k::OIR.Expression = (inner[next]::OIR.INDEX).exp
    next += 1
    if s isa OIR.WHOLE
      push!(out, OIR.INDEX(k))
      continue
    end
    local range::Union{Nothing, StepRange{Int, Int}} = _IRD.literalRange(_IRD.foldTree((s::OIR.SLICE).exp))
    range === nothing && ns("a slice by $(_strIR(s.exp))")
    push!(out, OIR.INDEX(k isa OIR.INTEGER ? OIR.INTEGER(range[k.value]) :
      _IRD.foldTree(OIR.BINARY(OIR.INTEGER(first(range) - step(range)), OIR.ADD,
                               OIR.BINARY(OIR.INTEGER(step(range)), OIR.MUL, k, OIR.T_INTEGER()), OIR.T_INTEGER()))))
  end
  append!(out, inner[next:end])
  return out
end

"The value of an array literal of literals (a vector, a matrix of rows of equal length); `nothing` for another."
function _literalArrayIR(e::OIR.ARRAY)::Union{Nothing, Array}
  isempty(e.elements) && return nothing
  all(x::OIR.Expression -> x isa Union{OIR.INTEGER, OIR.REAL, OIR.BOOLEAN}, e.elements) &&
    return map(x::OIR.Expression -> x.value, e.elements)
  all(x::OIR.Expression -> x isa OIR.ARRAY, e.elements) || return nothing
  local rows::Vector{Union{Nothing, Array}} = Union{Nothing, Array}[_literalArrayIR(x) for x in e.elements]
  all(r::Union{Nothing, Array} -> r !== nothing && ndims(r) == 1 && length(r) == length(rows[1]), rows) || return nothing
  return permutedims(reduce(hcat, rows))
end

"Whether `e` calls a Modelica function of the model (`_functionDefinitions`)."
_isFunctionCall(g::Gen, e::OIR.Expression)::Bool =
  e isa OIR.CALL && !e.attributes.builtin && haskey(g.functions, replace(String(e.name), "." => "_"))

"""
The position of output `k` of a call of type `ty` in the tuple a generated function returns: a record output is its
fields (`generateIOL`).
"""
function _outputPosition(ty::OIR.ModelicaType, k::Int)::Int
  ty isa OIR.T_TUPLE || return k
  return 1 + sum((t isa OIR.T_RECORD ? length(t.fields) : 1 for t in ty.types[1:k - 1]); init = 0)
end

"""
The value of an expression as a function's argument: a scalar's (`toIR`); an array's the array of its elements
(column-major, `Op(:arrayval)`), of a function's array result the call's, of a record's array field the field's.
"""
function _valueIRx(g::Gen, x::OIR.Expression, ctx::Ctx)::IR
  local e::OIR.Expression = _fieldExpression(x)
  local ty::OIR.ModelicaType = OIR.typeOf(e)
  ty isa OIR.T_ARRAY || return toIR(g, e, ctx)
  _isFunctionCall(g, e) && return _functionCallIRx(g, e::OIR.CALL, ctx)
  e isa OIR.RECORD_ELEMENT && !(e.exp isa Union{OIR.CREF, OIR.RECORD}) && return _recordElementIRx(g, e, ctx)
  local dims::Vector{Int} = Int[d isa OIR.DIM_SIZE ? d.size : ns("an array argument of a size not known: $(_strIR(e))")
                                for d in ty.dimensions]
  prod(dims) <= MAX_REDUCTION_TERMS || ns("an array argument of $(prod(dims)) elements")
  local elements::Vector{IR} = IR[]
  for t in Iterators.product((1:d for d in dims)...)   #= column-major: the first index fastest =#
    local subscripts::Vector{OIR.Subscript} = OIR.Subscript[OIR.INDEX(OIR.INTEGER(i)) for i in t]
    push!(elements, toIR(g, OIR.SUBSCRIPTED(e, subscripts, ty.elementType), ctx))
  end
  #= a constant array: its value in the code =#
  if all(x::IR -> x isa Lit, elements)
    local constant::Vector = map(x::IR -> (x::Lit).v, elements)
    return Lit(length(dims) == 1 ? constant : reshape(constant, dims...))
  end
  return Op(:arrayval, IR[Lit(Tuple(dims)); elements])
end

"""
Field `k` (`f`) of a record expression: of a record literal its element, of a reference to a record variable the
field's reference (`R.T`), of another (a call) its record element.
"""
function _fieldOf(a::OIR.Expression, k::Int, f::OIR.Field)::OIR.Expression
  a isa OIR.RECORD && return a.elements[k]
  a isa OIR.CREF && return OIR.CREF(OIR.ComponentRef(OIR.CrefPart[a.cref.parts..., OIR.CrefPart(f.name, OIR.Subscript[], f.ty)]), f.ty)
  return OIR.RECORD_ELEMENT(a, k, f.name, f.ty)
end

"""
A field of a record value: of a record literal or a record variable the field (`_fieldOf`), of a function's record
result the call's tuple at the field (the function returns a record by its fields, `generateIOL`).
"""
function _recordElementIRx(g::Gen, e::OIR.RECORD_ELEMENT, ctx::Ctx)::IR
  local rty::OIR.ModelicaType = OIR.typeOf(e.exp)
  rty isa OIR.T_RECORD || ns("a field of $(_strIR(e.exp)), not a record")
  e.exp isa Union{OIR.CREF, OIR.RECORD} && return _valueIRx(g, _fieldOf(e.exp, e.index, rty.fields[e.index]), ctx)
  _isFunctionCall(g, e.exp) || ns("a field of $(_strIR(e.exp))")
  any(f::OIR.Field -> f.ty isa OIR.T_RECORD, rty.fields) && ns("a record result with record fields: $(_strIR(e.exp))")
  local call::IR = _functionCallIRx(g, e.exp::OIR.CALL, ctx)
  return g.functions[replace(String((e.exp::OIR.CALL).name), "." => "_")] == 1 ? call : Op(:index, IR[call, Lit(e.index)])
end

"""
An argument's values (onto `out`): a record's fields each (a generated function takes a record by its fields, `R_T`,
`R_w`: `generateIOL`), an array's or a scalar's value (`_valueIRx`).
"""
function _argumentsIRx!(out::Vector{IR}, g::Gen, a::OIR.Expression, ctx::Ctx)::Nothing
  local ty::OIR.ModelicaType = OIR.typeOf(a)
  if ty isa OIR.T_RECORD
    for (k, f) in enumerate(ty.fields)
      f.ty isa OIR.T_RECORD && ns("a record argument with record fields: $(_strIR(a))")
      push!(out, _valueIRx(g, _fieldOf(a, k, f), ctx))
    end
  else
    push!(out, _valueIRx(g, a, ctx))
  end
  return nothing
end

"""
A call of a Modelica function of the model (`_isFunctionCall`) as its value: its arguments' values
(`_argumentsIRx!`); of a function of more outputs typed as its first, that output; of a record result the tuple of
its fields (and the outputs after it).
"""
function _functionCallIRx(g::Gen, e::OIR.CALL, ctx::Ctx)::IR
  local jn::String = replace(String(e.name), "." => "_")
  local callArgs::Vector{IR} = IR[]
  for a in e.arguments
    _argumentsIRx!(callArgs, g, a, ctx)
  end
  local call::IR = Op(:fcall, IR[Lit(Symbol(jn)), callArgs...])
  #= the outputs counted as returned, a record by its fields (generateIOL): a record result is the tuple's first =#
  g.functions[jn] > 1 && !(e.ty isa Union{OIR.T_TUPLE, OIR.T_RECORD}) && (call = Op(:index, IR[call, Lit(1)]))
  return call
end

function _callIRx(g::Gen, e::OIR.CALL, ctx::Ctx)::IR
  local fname::String = String(e.name)
  local args::Vector{OIR.Expression} = e.arguments
  fname in ("delay", "OpenModelica.Internal.delay2", "OpenModelica.Internal.delay3") && ns("delay()")
  if fname == "der"
    ns("der() in a reduced model: $(_strIR(e))")
  elseif fname == "sample" || endswith(fname, ".sample")
    #= in a when-condition (an equation refuses it: ctx.nsam) =#
    length(args) == 2 || ns("sample with $(length(args)) arguments")
    ctx.nsam[] += 1
    return Op(:sample, IR[toIR(g, args[1], ctx), toIR(g, args[2], ctx), Lit(ctx.nsam[])])
  elseif fname == "pre"
    args[1] isa OIR.CREF || ns("pre of an expression: $(_strIR(e))")
    local preRef::IR = _crefIRx(g, args[1], ctx)
    preRef isa Ref || ns("pre of $(_strIR(e))")
    return Pre(preRef)
  elseif fname == "homotopy"
    return toIR(g, args[1], ctx)
  elseif fname == "noEvent"
    return toIR(g, args[1], _with(ctx; noEvent = true))
  elseif fname == "smooth"
    return toIR(g, args[2], ctx)
  elseif fname == "initial"
    g.usesInitial = true
    return Op(:initial, IR[])
  elseif fname == "edge" || fname == "change"
    local changed::IR = toIR(g, args[1], ctx)
    changed isa Ref || ns("$(fname) of $(_strIR(args[1]))")
    return fname == "edge" ? Op(:&&, IR[changed, Op(:!, IR[Pre(changed)])]) : Op(:!=, IR[changed, Pre(changed)])
  elseif fname == "semiLinear"
    local x::IR = toIR(g, args[1], ctx)
    return Op(:if, IR[Op(:>=, IR[x, Lit(0.0)]), Op(:*, IR[x, toIR(g, args[2], ctx)]), Op(:*, IR[x, toIR(g, args[3], ctx)])])
  elseif fname in ("integer", "floor", "ceil", "div", "mod", "rem")
    local xs::Vector{IR} = IR[toIR(g, a, ctx) for a in args]
    local q::IR = length(xs) == 2 ? Op(:/, IR[xs[1], xs[2]]) : xs[1]
    local kind::Symbol = fname in ("integer", "floor", "mod") ? :floor : fname == "ceil" ? :ceil : :trunc
    local v::IR = if ctx.noEvent || !_continuous(g, q)
      Op(kind, IR[q])
    else
      ctx.nrel[] += 2
      EvFn(kind, q, ctx.nrel[] - 1)
    end
    fname in ("mod", "rem") || return v
    return Op(:-, IR[xs[1], Op(:*, IR[v, xs[2]])])
  end
  #= a Modelica function of the model (_functionDefinitions); its array or record result only under a subscript or
     a field (_subscriptedIRx, _recordElementIRx) =#
  if _isFunctionCall(g, e)
    e.ty isa Union{OIR.T_ARRAY, OIR.T_RECORD} && ns("function $(fname) of an array or record result outside an element")
    return _functionCallIRx(g, e, ctx)
  end
  local jf::Union{Nothing, Symbol} = startswith(fname, "Modelica.Math.") && fname[15:end] in MATH_WRAPPED ?
    MATH_BUILTINS[fname[15:end]] : get(MATH_BUILTINS, fname, nothing)
  jf === nothing && ns("function $(fname)")
  return Op(jf, IR[toIR(g, a, ctx) for a in args])
end

"""
The classes of an OMJLIR equation (`_addEquation!` of the flat model's): a for-equation's body over its iterator's
range (the domain crossed), a scalar equation one class over the domain, an assert one checked over it.
"""
function _addEquationIR!(g::Gen, eq::OIR.Equation, iterNames::Vector{String}, idom::Matrix{Int}; initial::Bool = false)
  size(idom, 2) == 0 && return
  local iters::Dict{String, Int} = Dict{String, Int}(n => k for (k, n) in enumerate(iterNames))
  local nIter::Int = length(iterNames)
  if eq isa OIR.FOR_EQUATION
    local range::Union{Nothing, StepRange{Int, Int}} = _IRD.literalRange(_IRD.foldTree(eq.iterator.range))
    range === nothing && ns("for range $(_strIR(eq.iterator.range))")
    local values::Vector{Int} = collect(Int, range)
    isempty(values) && return
    if any(b::OIR.Equation -> _readsElementArray(g, b) || _subscriptsLiteral(b, eq.iterator.name), eq.body)
      #= unrolled: each iteration's equations read the elements by literal subscripts (like ones one class again) =#
      for value in values, b in eq.body
        _addEquationIR!(g, _withIteratorIR(b, Dict{Symbol, Int}(eq.iterator.name => value)), iterNames, idom; initial = initial)
      end
      return
    end
    local dom::Matrix{Int} = _crossDomain(idom, values)
    for b in eq.body
      _addEquationIR!(g, b, [iterNames; String(eq.iterator.name)], dom; initial = initial)
    end
    return
  elseif eq isa OIR.SCALAR_EQUATION
    #= relations of initial equations are evaluated at the start (no events) =#
    local ctx::Ctx = Ctx(iters, Int[], Base.RefValue(0), initial)
    local lhs::IR = toIR(g, eq.lhs, ctx)
    local rhs::IR = toIR(g, eq.rhs, ctx)
    ctx.nsam[] == 0 || ns("sample() in an equation")
    push!(initial ? g.initClasses : g.classes, EqClass(lhs, rhs, nIter, idom, ctx.nrel[], _strIR(eq)))
    return
  elseif eq isa OIR.IF_EQUATION
    _addIfEquationIR!(g, eq, iterNames, idom; initial = initial)
    return
  elseif eq isa OIR.ASSERT_EQUATION
    local actx::Ctx = Ctx(iters, Int[], Base.RefValue(0), true)
    push!(initial ? g.initAsserts : g.asserts,
          AssertClass(toIR(g, eq.condition, actx), occursin("warning", _strIR(eq.level)), nIter, idom,
                      _strIR(eq.condition), _strIR(eq.message), _messageIRx(g, eq.message, actx)))
    return
  elseif eq isa OIR.WHEN_EQUATION
    initial && ns("a when-equation among the initial equations: $(_strIR(eq))")
    _addWhenIR!(g, eq, iterNames, idom)
    return
  end
  ns("$(nameof(typeof(eq))) in an OMJLIR model: $(_strIR(eq))")
end

"""
A when-equation (`_addWhen!` of the flat model's), over the domain: each branch's triggers its condition (several
the vector of them, each an edge of its own; a branch active at the initialization with initial() among them: `when
initial()` alone), its statements the body: assignments to discrete variables, reinit of states, asserts (checked
when it runs) and calls. Not a when of the passes' initial or structural kinds, terminal(), or other statements.
"""
function _addWhenIR!(g::Gen, eq::OIR.WHEN_EQUATION, iterNames::Vector{String}, idom::Matrix{Int})::Nothing
  eq.kind == OIR.REGULAR_WHEN || ns("a when-equation of kind $(eq.kind): $(_strIR(eq))")
  local ctx::Ctx = Ctx(Dict{String, Int}(n => k for (k, n) in enumerate(iterNames)), Int[])
  local conds::Vector{IR} = IR[]
  local body::Vector{Tuple{Symbol, Ref, IR}} = Tuple{Symbol, Ref, IR}[]
  local starts::Vector{Int} = Int[]
  local noTarget::Ref = Ref("", IR[], false)
  for br in eq.branches
    br.atTerminal && ns("terminal() in a when-condition: $(_strIR(eq))")
    local triggers::Vector{IR} = IR[_topLevelInitial(toIR(g, c, ctx)) for c in br.triggers]
    br.atInitial && push!(triggers, Op(:initial, IR[]))
    isempty(triggers) && ns("a when-branch without a condition: $(_strIR(eq))")
    push!(conds, length(triggers) == 1 ? triggers[1] : Op(:vect, triggers))
    push!(starts, length(body) + 1)
    for st in br.body
      if st isa OIR.ASSIGNMENT
        st.lhs isa OIR.CREF || ns("a when-statement assigning $(_strIR(st.lhs))")
        local target::IR = toIR(g, st.lhs, ctx)
        (target isa Ref && g.vars[target.name].kind == :discrete) || ns("when assigns a non-discrete variable: $(_strIR(st.lhs))")
        push!(body, (:assign, target, toIR(g, st.rhs, ctx)))
      elseif st isa OIR.REINIT
        local state::IR = toIR(g, OIR.CREF(st.state, OIR.T_REAL()), ctx)
        (state isa Ref && g.vars[state.name].kind == :state) || ns("reinit of a non-state: $(_strIR(eq))")
        push!(body, (:reinit, state, toIR(g, st.value, ctx)))
      elseif st isa OIR.ASSERT
        push!(g.assertTexts, (_strIR(st.condition), _strIR(st.message), occursin("warning", _strIR(st.level))))
        local actx::Ctx = _with(ctx; noEvent = true)
        push!(body, (:call, noTarget, Op(:assertcall, IR[toIR(g, st.condition, actx), _messageIRx(g, st.message, actx),
                                                         Lit(length(g.assertTexts))])))
      elseif st isa OIR.NORETCALL
        push!(body, (:call, noTarget, toIR(g, st.call, _with(ctx; noEvent = true))))
      else
        ns("a when-statement $(nameof(typeof(st))): $(_strIR(eq))")
      end
    end
  end
  local cond::IR = length(conds) == 1 ? conds[1] : Op(:elsewhen, conds)
  push!(g.whens, WhenClass(cond, body, starts, length(iterNames), idom, ctx.nrel[], ctx.nsam[], _strIR(eq)))
  return nothing
end

"""
An if-equation (`_addDynamicIf!` of the flat model's): on conditions of parameters and iterators each instance its
branch's equations; on a variable condition each variable its branches assign (`v = e` in every branch and the else)
one class `v = if c1 then e1 elseif ... else en`.
"""
function _addIfEquationIR!(g::Gen, eq::OIR.IF_EQUATION, iterNames::Vector{String}, idom::Matrix{Int}; initial::Bool = false)
  local iters::Dict{String, Int} = Dict{String, Int}(n => k for (k, n) in enumerate(iterNames))
  local nIter::Int = length(iterNames)
  local bodies::Vector{Vector{OIR.Equation}} = Vector{OIR.Equation}[[br.equations for br in eq.branches]; [eq.elseEquations]]
  local staticCtx::Ctx = Ctx(iters, Int[], Base.RefValue(0), true)
  local conditions::Vector{IR} = IR[toIR(g, br.condition, staticCtx) for br in eq.branches]
  if all(c::IR -> _isStatic(g, c), conditions)
    local groups::Vector{Vector{Int}} = Vector{Int}[Int[] for _ in bodies]
    for j in 1:size(idom, 2)
      local k::Union{Nothing, Int} = findfirst(c::IR -> _evalStatic(g, c, idom, j) == true, conditions)
      push!(groups[k === nothing ? length(bodies) : k], j)
    end
    for (body, cols) in zip(bodies, groups)
      isempty(cols) && continue
      foreach(b::OIR.Equation -> _addEquationIR!(g, b, iterNames, idom[:, cols]; initial = initial), body)
    end
    return
  end
  local keyed::Vector{Dict{String, OIR.SCALAR_EQUATION}} = Dict{String, OIR.SCALAR_EQUATION}[]
  for body in bodies
    local assigned::Dict{String, OIR.SCALAR_EQUATION} = Dict{String, OIR.SCALAR_EQUATION}()
    for b in body
      (b isa OIR.SCALAR_EQUATION && b.lhs isa OIR.CREF) || ns("if-equation on a variable condition: branch equation $(_strIR(b))")
      local key::String = String(b.lhs.cref.name)
      haskey(assigned, key) && ns("if-equation on a variable condition: two equations for $(key)")
      assigned[key] = b
    end
    push!(keyed, assigned)
  end
  all(d::Dict{String, OIR.SCALAR_EQUATION} -> Set(keys(d)) == Set(keys(keyed[1])), keyed) ||
    ns("if-equation on a variable condition: branches give different variables")
  for key in sort!(collect(keys(keyed[1])))
    local ctx::Ctx = Ctx(iters, Int[], Base.RefValue(0), initial)
    local lhs::IR = toIR(g, keyed[1][key].lhs, ctx)
    local rhs::IR = toIR(g, keyed[end][key].rhs, ctx)
    for i in length(eq.branches):-1:1
      rhs = Op(:if, IR[toIR(g, eq.branches[i].condition, ctx), toIR(g, keyed[i][key].rhs, ctx), rhs])
    end
    ctx.nsam[] == 0 || ns("sample() in an equation")
    push!(initial ? g.initClasses : g.classes,
          EqClass(lhs, rhs, nIter, idom, initial ? 0 : ctx.nrel[], "$(key) = if ... ($(_strIR(eq.branches[1].condition)))"))
  end
end

"""
Whether the equation reads an array that is no variable by subscripts (`frame_a.R.w[\$i1]`: a record's array field,
whose elements are variables of their own, `expandRecordFieldArrays!`, read by a for-equation rolled before).
"""
function _readsElementArray(g::Gen, eq::OIR.Equation)::Bool
  local found::Bool = false
  OIR.foreachExp(eq) do x::OIR.Expression
    #= also as a reference subscripted on its last part (`observed.offset[\$i1]`, inlined: `subscriptedReference`) =#
    local base::Union{Nothing, OIR.ComponentRef} =
      (x isa OIR.SUBSCRIPTED && x.exp isa OIR.CREF && !OIR.hasSubscripts(x.exp.cref) && OIR.typeOf(x.exp) isa OIR.T_ARRAY) ? x.exp.cref :
      (x isa OIR.CREF && length(x.cref.parts) > 1 && !isempty(x.cref.parts[end].subscripts) &&
       all(p::OIR.CrefPart -> isempty(p.subscripts), x.cref.parts[1:end - 1])) ? x.cref : nothing
    base === nothing && return nothing
    local last::OIR.CrefPart = base.parts[end]
    local nm::String = String(OIR.ComponentRef(OIR.CrefPart[base.parts[1:end - 1]; OIR.CrefPart(last.name, OIR.Subscript[], last.ty)]).name)
    (haskey(g.vars, nm) || haskey(g.derOf, nm)) || (found = true)
    nothing
  end
  return found
end

"""
Whether the equation subscripts an array literal by the iterator `it` (an inlined call's matrix product, `_inlinedEquation`:
`{{R.T[1, 1], ...}, ...}[\$i1, 2]`): unrolled, the subscripts are literal and the element is picked.
"""
function _subscriptsLiteral(eq::OIR.Equation, it::Symbol)::Bool
  local found::Bool = false
  local isIterator::Function = (y::OIR.Expression) -> y isa OIR.CREF && length(y.cref.parts) == 1 && y.cref.name === it &&
                                                      !OIR.hasSubscripts(y.cref)
  OIR.foreachExp(eq) do x::OIR.Expression
    (x isa OIR.SUBSCRIPTED && x.exp isa OIR.ARRAY) || return nothing
    any(s::OIR.Subscript -> s isa OIR.INDEX && OIR.anyExp(isIterator, (s::OIR.INDEX).exp), x.subscripts) && (found = true)
    nothing
  end
  return found
end

"A copy of the equation `eq` with the iterators of `values` replaced by their values (`withIterators`)."
_withIteratorIR(eq::OIR.Equation, values::Dict{Symbol, Int})::OIR.Equation =
  OIR.mapEquation!(x::OIR.Expression -> _IRD.withIterators(x, values), deepcopy(eq))

"Whether statements hold an assert (also in branches and loops)."
function _hasAssert(stmts::Vector{OIR.Statement})::Bool
  for st in stmts
    st isa OIR.ASSERT && return true
    st isa OIR.IF_STATEMENT && (any(b::OIR.StatementBranch -> _hasAssert(b.body), st.branches) || _hasAssert(st.elseBody)) && return true
    (st isa OIR.FOR_STATEMENT || st isa OIR.WHILE_STATEMENT) && _hasAssert(st.body) && return true
  end
  return false
end

"""
    _inlinedEquation(eq, functions) -> Equation

`eq` with its calls of `functions` inlined where their algorithms run into one expression (`IndexReduction.inlineCalls`;
scalar, if- and for-equations, a for-equation's body): MultiBody's `planarRotation(e, phi, w)[1][2, 1]` as
`e[2] * e[1] + (-(e[2] * e[1])) * cos(phi) - e[3] * sin(phi)`, no call and no array built per element (each element of
a record result called the function again). Other kinds, and an equation without such a call, as they are (inlining
also writes a subscripted reference on its parts: `observed.offset[\$i1]` of a record variable's array field, whose
elements are variables, then read as a variable of its own).
"""
function _inlinedEquation(eq::OIR.Equation, functions::Dict{Symbol, OIR.ModelicaFunction})::OIR.Equation
  local calls::Bool = false
  OIR.foreachExp(eq) do x::OIR.Expression
    (x isa OIR.CALL && haskey(functions, x.name)) && (calls = true)
    nothing
  end
  calls || return eq
  (eq isa OIR.SCALAR_EQUATION || eq isa OIR.IF_EQUATION) && return _IRD.inlineEquation(eq, functions)
  eq isa OIR.FOR_EQUATION &&
    return OIR.FOR_EQUATION(eq.iterator, OIR.Equation[_inlinedEquation(e, functions) for e in eq.body], eq.source, eq.attributes)
  return eq
end

function _generateIR(model::OIR.Model, modelName::String; functions::Union{Nothing, FunctionList} = nothing)::Expr
  (length(model.continuous) == 1 && isempty(model.clocked) && isempty(model.modes)) ||
    ns("an OMJLIR model of several partitions, clocks or structural modes")
  local (functionDefs::Vector{Any}, functionOutputs::Dict{String, Int}) = _functionDefinitions(functions)
  local partition::OIR.Partition = model.continuous[1]
  #= the values of the elements: on the element view =#
  local view::_IRD.Scalarized = _IRD.scalarizeArrays(model)
  local E::_IRD.Elements = view.elements
  local values::_EV.StartValues = _EV.StartValues(view.model.variables, view.model.functions)
  local elementVariable::Function = (name::Symbol) -> view.model.variables[name]
  local startOf::Function = function (name::Symbol)
    local s::Union{Nothing, OIR.Expression} = elementVariable(name).attributes.start
    s === nothing && return 0.0
    local v::Union{Nothing, Float64} = _EV.evaluate(s, values)
    v === nothing && ns("the start of $(name) does not evaluate")
    return v
  end
  local fixedOf::Function = function (name::Symbol, default::Bool)
    local f::Union{Nothing, OIR.Expression} = elementVariable(name).attributes.fixed
    f === nothing && return default
    local c::Union{Nothing, Bool} = _EV.evaluateCondition(f, values)
    c === nothing && ns("the fixed attribute of $(name) does not evaluate")
    return c
  end
  local g::Gen = Gen(Dict{String, VarInfo}(), 0, 0, 0, Dict{String, Any}(), Dict{String, Symbol}(), Set{String}(),
                EqClass[], EqClass[], AlgClass[], Tuple{String, String, Bool}[], WhenClass[], AssertClass[], 0, 0, 0, 0,
                -1000, false, Dict{Int, Int}(), AlgClass[], String[], AssertClass[], functionOutputs, Dict{String, String}())
  local stateNamesOrdered::Vector{String} = String[]; local algNamesOrdered::Vector{String} = String[]
  local discreteNamesOrdered::Vector{String} = String[]
  local u0::Vector{Float64} = Float64[]; local d0::Vector{Float64} = Float64[]; local a0::Vector{Float64} = Float64[]
  local fixedMask::Vector{Bool} = Bool[]
  local fixedNonStates::Vector{String} = String[]
  for v in model.variables
    local nm::String = String(v.name)
    local kind::OIR.VariableKind = v.kind
    if kind isa OIR.STATE_DER
      g.derOf[nm] = String(kind.state)
      continue
    end
    local dims::Vector{Int} = _dimsIR(v)
    local n::Int = isempty(dims) ? 1 : prod(dims)
    #= the element view's elements; of a record field array of a parameter (its elements parameters of their own, the
       array kept for a call's whole argument: `expandRecordFieldArrays`) its elements' references =#
    local names::Vector{Symbol} = haskey(E.elements, v.name) ? E.elements[v.name] :
      (!isempty(dims) && kind isa Union{OIR.PARAMETER, OIR.CONSTANT}) ?
        Symbol[_IRD.modelElementCref(v, index).name for index in Passes.rowMajor(dims)] : Symbol[v.name]
    length(names) == n || ns("the elements of $(nm)")
    local elementNames::Vector{String} = String.(names)
    local vtype::Symbol = _vtypeIR(v.ty)
    local perPart::Vector{Int} = Int[length(dims)]
    if kind isa Union{OIR.PARAMETER, OIR.CONSTANT}
      #= fixed = false: an unknown of the initialization (its initial equations), from its start =#
      local free::Vector{Bool} = Bool[!fixedOf(x, true) for x in names]
      (any(free) && !all(free)) && ns("parameter $(nm) with fixed = false on some elements")
      local value::Any = nothing
      if all(free)
        v.binding === nothing || ns("parameter $(nm) with fixed = false and a binding")
        push!(g.freeParams, nm)
        local starts::Vector{Float64} = startOf.(names)
        value = isempty(dims) ? starts[1] :
          length(dims) == 1 ? starts : permutedims(reshape(starts, reverse(dims)...), length(dims):-1:1)
      elseif vtype != :other
        local vals::Vector{Any} = Any[]
        for x in names
          local elementValue::Union{Nothing, Float64} = values(x)
          elementValue === nothing && ns("the value of parameter $(x)")
          push!(vals, _typedValue(elementValue, vtype))
        end
        value = isempty(dims) ? vals[1] :
          length(dims) == 1 ? [x for x in vals] : permutedims(reshape([x for x in vals], reverse(dims)...), length(dims):-1:1)
      end
      g.vars[nm] = VarInfo(:param, dims, perPart, 0, vtype, IRVariable(v, elementNames, value))
      continue
    end
    #= a variable's binding is an equation of the partition already (Passes.addBindingEquations!) =#
    if kind isa OIR.STATE
      vtype == :real || ns("the state $(nm), not Real")
      g.vars[nm] = VarInfo(:state, dims, perPart, g.nStates, :real, IRVariable(v, elementNames, nothing))
      g.nStates += n
      append!(u0, startOf.(names))
      append!(fixedMask, Bool[fixedOf(x, false) for x in names])
      append!(stateNamesOrdered, elementNames)
    elseif kind isa OIR.DISCRETE || (kind isa OIR.ALGEBRAIC && vtype in (:int, :bool))
      vtype == :other && ns("the discrete variable $(nm) of type $(v.ty)")
      #= an Integer or Boolean an equation computes (MultiBody's colors): discrete, in the discrete buffer =#
      g.vars[nm] = VarInfo(:discrete, dims, perPart, g.nDiscrete, vtype, IRVariable(v, elementNames, nothing))
      g.nDiscrete += n
      append!(d0, startOf.(names))
      append!(discreteNamesOrdered, elementNames)
    elseif kind isa Union{OIR.ALGEBRAIC, OIR.DUMMY_STATE, OIR.DUMMY_DER}
      vtype == :real || ns("the algebraic variable $(nm), not Real")
      local fixed::Vector{Bool} = Bool[fixedOf(x, false) for x in names]
      kind isa OIR.DUMMY_STATE && any(fixed) && ns("a fixed start on the dummy state $(nm)")
      any(fixed) && push!(fixedNonStates, nm)
      g.vars[nm] = VarInfo(:alg, dims, perPart, g.nAlg, :real, IRVariable(v, elementNames, nothing))
      g.nAlg += n
      append!(a0, startOf.(names))
      append!(algNamesOrdered, elementNames)
    else
      ns("the variable $(nm) of kind $(nameof(typeof(kind)))")
    end
  end
  g.nStates == 0 && ns("no states")
  for (link, state) in g.derOf
    haskey(g.vars, state) && g.vars[state].kind == :state || ns("the state derivative $(link) of $(state), not a state")
  end
  #= after index reduction, the calls inlined (MLS: InlineAfterIndexReduction, Inline) where a body runs into one
     expression; not a function with an assert (inlined, its check is lost: normalizeWithAssert) =#
  local inlinable::Dict{Symbol, OIR.ModelicaFunction} =
    Dict{Symbol, OIR.ModelicaFunction}(f.name => f for f in model.functions if f.external === nothing && !_hasAssert(f.body))
  for eq in partition.equations
    _addEquationIR!(g, _inlinedEquation(eq, inlinable), String[], NO_SLOTS)
  end
  for nm in fixedNonStates
    #= fixed = true on a variable that is no state: the initial equation v = start =#
    local info::VarInfo = g.vars[nm]
    local names::Vector{String} = (info.var::IRVariable).names
    local subs::Vector = isempty(info.dims) ? [()] : vec([reverse(t) for t in Iterators.product(reverse([1:d for d in info.dims])...)])
    for (k, t) in enumerate(subs)
      local x::Symbol = Symbol(names[k])
      fixedOf(x, false) || continue
      push!(g.initClasses, EqClass(Ref(nm, IR[Lit(i) for i in t], false), Lit(startOf(x)), 0, NO_SLOTS, 0, "$(names[k]) = start"))
    end
  end
  for eq in partition.initialEquations
    _addEquationIR!(g, _inlinedEquation(eq, inlinable), String[], NO_SLOTS; initial = true)
  end
  return _emitModule(g, modelName, functionDefs, u0, d0, a0, fixedMask, stateNamesOrdered, algNamesOrdered,
                     discreteNamesOrdered)
end
