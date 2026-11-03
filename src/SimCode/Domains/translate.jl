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
The domain level's translation (2026-11-01): the simulation code of a model with its arrays kept
(SimCode.lowerArraySystem: reduced in array form, Passes.IndexReduction.reduceIndexArrays!, its derivative unknowns
solved in array form, solvedInArrays, in the array code's form, Passes.arrayCodeForm!) to the variables and classes of
the domain level (`domainSystem`), from which CodeGeneration's ArrayODEGen (_emitModule) generates the module.

  - Its for-equations are the classes, their iterators the slots; a scalar equation reading
    elements by literal subscripts is a class of one instance (re-rolled with its like ones).
  - der(x[i]) is the state's derivative (the link `$STATEDER.x`); the dummy states, dummy derivatives and derivative
    variables are algebraic variables.
  - Start values, fixed attributes and parameter values as the lowering evaluated them (constants of the simulation
    code, `SimCode.evaluatedVariable`), element by element.

  - Records reach the code through functions: a record argument by its fields (as the generated
    functions take it), an array argument as the array of its elements, a record result as the
    tuple of its fields; an element of an array literal or a function's array result by indexing its value.
    If-equations as the flat model's; parameters with fixed = false unknowns of the initialization.

  - When-equations as the flat model's (`_addWhen!`): their triggers the condition, sample() in it.

Not (yet), NotSupported (the caller takes the scalarized path): array equations, algorithms, a
when-equation of the passes' structural kinds, sample() outside a when-condition, a
fixed start on a dummy state (the reduced initialization's), whole arrays in scalar equations,
records of records through functions.
=#

"A simulation code expression, statement or equation in a reason message (printing must not fail)."
_str(@nospecialize(x)) = try
  replace(Base.first(string(x), 200), r"\s*\n\s*" => " ")
catch
  string("<", nameof(typeof(x)), ">")
end

"Whether an assert's level is `AssertionLevel.warning`: the enumeration literal by its name."
_isWarning(level::SC.Exp)::Bool = level isa SC.ENUM_LITERAL && endswith(String(level.name), ".warning")


"The values of a constant of the simulation code (`SimCode.constantExp`), row-major."
function _constantValues(e::SC.Exp)::Vector{Any}
  e isa SC.ARRAY_EXP && return Any[x for el in e.elements for x in _constantValues(el)]
  e isa Union{SC.RCONST, SC.ICONST, SC.BCONST} && return Any[e.value]
  ns("a value that is no constant: $(_str(e))")
end

"""
The parameters a variable's binding reads.
"""
function _bindingDeps(g::Gen, v::SimVariable)::Vector{String}
  #= a parameter of no numeric value (a String, a record or external object field) no dependency: the code holds no value of it =#
  local deps::Vector{String} = String[]
  local b::Union{Nothing, SC.Exp} = SC.bindingOf(v.simVar.varKind)
  b === nothing && return deps
  SC.foreachExp(b) do x::SC.Exp
    x isa SC.EXP_CREF || return nothing
    local info::Union{Nothing, VarInfo} = get(g.vars, String(x.cref.sym), nothing)
    info !== nothing && info.kind == :param && info.vtype != :other && !(String(x.cref.sym) in deps) && push!(deps, String(x.cref.sym))
    nothing
  end
  return deps
end

"""
An assert's condition with its calls without a result made, then true: a call equation run where the asserts are
checked (`callEquation`), alone or under its if-equation branch's condition (`hoistIfEquationAsserts!`: `if c then f()
else true`, `not c or f()`). `noEvent(e)` in its and/or/not/if structure kept as a mark (`Op(:noEvent)`, the array code
emits `e`): its relations are no crossings (`_assertRelations!`), as on the scalarized path.
"""
function _checkedCondition(g::Gen, e::SC.Exp, ctx::Ctx)::IR
  (e isa SC.CALL && e.ty isa SC.TYPE_NORETCALL) && return Op(:seq, IR[toIR(g, e, ctx), Lit(true)])
  (e isa SC.CALL && e.kind == SC.CALL_BUILTIN && e.name === :noEvent && length(e.args) == 1) &&
    return Op(:noEvent, IR[_checkedCondition(g, e.args[1], ctx)])
  (e isa SC.LUNARY && e.op == SC.OP_NOT) && return Op(:!, IR[_checkedCondition(g, e.exp, ctx)])
  e isa SC.IFEXP &&
    return Op(:if, IR[toIR(g, e.cond, ctx), _checkedCondition(g, e.thenExp, ctx), _checkedCondition(g, e.elseExp, ctx)])
  (e isa SC.LBINARY && e.op in (SC.OP_AND, SC.OP_OR)) &&
    return Op(e.op == SC.OP_AND ? :&& : :||, IR[_checkedCondition(g, e.exp1, ctx), _checkedCondition(g, e.exp2, ctx)])
  return toIR(g, e, ctx)
end

"""
    ParameterBinding(name, deps, kind, values)

A parameter the array code holds (`parameterBindings`): its name, the parameters its binding reads, and its binding as
the domain level's expressions, where the array code computes it again when one of those changes or, a parameter the
translation has no value of (`Gen.startComputed`), at each start (`kind`): `:scalar` one value, `:each` one value for
every element, `:elements` an array's elements row-major (the lowering's array literal of them), `:value` an array's
value whole (a function's array result); `:none` where it has none (no parameter read, or a binding the array code
does not take).
"""
struct ParameterBinding
  name::String
  deps::Vector{String}
  kind::Symbol
  values::Vector{IR}
end

"""
    parameterBindings(g, names) -> Vector{ParameterBinding}

The parameters `names` (those the code reads) and the ones their bindings read, ordered so that a binding's parameters
come before it (NotSupported for a cycle), each with its binding (`ParameterBinding`); their values in the domain
system's parameter table (`_paramValue!`).
"""
function parameterBindings(g::Gen, names::Vector{String})::Vector{ParameterBinding}
  local order::Vector{String} = String[]
  local state::Dict{String, Int} = Dict{String, Int}()
  foreach((nm::String) -> _orderParameters!(g, nm, state, order), sort(names))
  return ParameterBinding[_parameterBinding(g, nm) for nm in order]
end

"The parameter `nm` after the ones its binding reads, into `order` (`state`: 1 being visited, 2 done)."
function _orderParameters!(g::Gen, nm::String, state::Dict{String, Int}, order::Vector{String})::Nothing
  get(state, nm, 0) == 2 && return nothing
  get(state, nm, 0) == 1 && ns("cyclic parameter bindings at $(nm)")
  state[nm] = 1
  foreach((d::String) -> _orderParameters!(g, d, state, order), _bindingDeps(g, g.vars[nm].var))
  _paramValue!(g, nm)
  state[nm] = 2
  push!(order, nm)
  return nothing
end

"The binding of the parameter `nm` (`ParameterBinding`): `:none` where it reads no parameter or does not translate."
function _parameterBinding(g::Gen, nm::String)::ParameterBinding
  local info::VarInfo = g.vars[nm]
  local deps::Vector{String} = _bindingDeps(g, info.var)
  (isempty(deps) && !(nm in g.startComputed)) && return ParameterBinding(nm, deps, :none, IR[])
  try
    local b::SC.Exp = SC.bindingOf(info.var.simVar.varKind)::SC.Exp
    local ctx::Ctx = Ctx(Dict{String, Int}(), Int[])
    isempty(info.dims) && return ParameterBinding(nm, deps, :scalar, IR[toIR(g, b, ctx)])
    (_isFunctionCall(g, b) && _expType(b) isa SC.TYPE_ARRAY) && return ParameterBinding(nm, deps, :value, IR[_valueIRx(g, b, ctx)])
    if !(b isa SC.ARRAY_EXP)
      #= one value for every element (`each`); an array binding not written per element has no code =#
      _expType(b) isa SC.TYPE_ARRAY && ns("the array binding of $(nm), not per element")
      return ParameterBinding(nm, deps, :each, IR[toIR(g, b, ctx)])
    end
    return ParameterBinding(nm, deps, :elements, IR[toIR(g, x, ctx) for x in _rowMajorElements(b)])
  catch err
    err isa NotSupported || rethrow(err)
    return ParameterBinding(nm, deps, :none, IR[])
  end
end

"The elements of an array literal, row-major (its rows nested)."
_rowMajorElements(e::SC.Exp)::Vector{SC.Exp} = e isa SC.ARRAY_EXP ? SC.Exp[x for el in e.elements for x in _rowMajorElements(el)] : SC.Exp[e]

"The Julia operator of an arithmetic operator of the simulation code, scalar or element-wise; NotSupported for another."
function _binaryOp(op::SC.OpKind)::Symbol
  op in (SC.OP_ADD, SC.OP_ADD_EW) && return :+
  op in (SC.OP_SUB, SC.OP_SUB_EW) && return :-
  op in (SC.OP_MUL, SC.OP_MUL_EW) && return :*
  op in (SC.OP_DIV, SC.OP_DIV_EW) && return :/
  op in (SC.OP_POW, SC.OP_POW_EW) && return :^
  ns("operator $(replace(string(op), "OP_" => "")) on arrays")
end

"The Julia operator of a relational operator of the simulation code; NotSupported for another."
function _relationOp(op::SC.OpKind)::Symbol
  op == SC.OP_LESS && return :<
  op == SC.OP_LESSEQ && return :<=
  op == SC.OP_GREATER && return :>
  op == SC.OP_GREATEREQ && return :>=
  op == SC.OP_EQUAL && return :(==)
  op == SC.OP_NEQUAL && return :!=
  ns("relation operator $(replace(string(op), "OP_" => ""))")
end

"Whether a simulation code expression is a String's (the type a binary `+` concatenates)."
_isString(e::SC.Exp)::Bool = _expType(e) isa SC.TYPE_STRING

"An assert's message as an expression; its text where the array path cannot evaluate it."
function _messageIRx(g::Gen, m::SC.Exp, ctx::Ctx)::IR
  try
    return toIR(g, m, _with(ctx; noEvent = true))
  catch err
    err isa NotSupported || rethrow(err)
    return Lit(_str(m))
  end
end

"The elements of an array literal (of nested ones too), in order; an expression that is none itself."
function _literalElements!(out::Vector{SC.Exp}, e::SC.Exp)::Vector{SC.Exp}
  e isa SC.ARRAY_EXP ? foreach(x::SC.Exp -> _literalElements!(out, x), e.elements) : push!(out, e)
  return out
end

"""
    toIR(g, e, ctx) -> IR

A simulation code expression in the domain level's expression form (`IR`): constants as literals, an iterator as its
slot (`ctx`), references as element references, operations, calls (`_callIRx`), relations numbered (events, unless
`ctx` says none), subscripts and fields; NotSupported for what the array code does not take.
"""
function toIR(g::Gen, e::SC.Exp, ctx::Ctx)::IR
  if e isa Union{SC.ICONST, SC.RCONST, SC.BCONST, SC.SCONST}
    return Lit(e.value)
  elseif e isa SC.ENUM_LITERAL
    return Lit(e.index)
  elseif e isa SC.CAST
    return toIR(g, e.exp, ctx)
  elseif e isa SC.EXP_CREF
    return _crefIRx(g, e, ctx)
  elseif e isa SC.DER
    return _elementRefIR(g, String(e.cref.sym), IR[Lit(k) for k in e.cref.subs]; der = true)
  elseif e isa SC.PRE
    local preRef::IR = _crefIRx(g, SC.EXP_CREF(e.cref, e.ty), ctx)
    preRef isa Ref || ns("pre of $(_str(e))")
    return Pre(preRef)
  elseif e isa SC.UNARY
    e.op == SC.OP_UMINUS || ns("unary operator $(e.op)")
    return Op(:neg, IR[toIR(g, e.exp, ctx)])
  elseif e isa SC.BINARY
    _isString(e) && return Op(:string, IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
    e.op == SC.OP_SCALAR_PRODUCT && ns("a scalar product of $(_str(e))")
    return Op(_binaryOp(e.op), IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
  elseif e isa SC.LUNARY
    e.op == SC.OP_NOT || ns("logical operator $(e.op)")
    return Op(:!, IR[toIR(g, e.exp, ctx)])
  elseif e isa SC.LBINARY
    local lop::Symbol = e.op == SC.OP_AND ? :&& : e.op == SC.OP_OR ? :|| : ns("logical operator $(e.op)")
    return Op(lop, IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
  elseif e isa SC.RELATION
    local op::Symbol = _relationOp(e.op)
    local l::IR = toIR(g, e.exp1, ctx)
    local r::IR = toIR(g, e.exp2, ctx)
    (ctx.noEvent || !(_continuous(g, l) || _continuous(g, r))) && return Op(op, IR[l, r])
    op in (:(==), :!=) && ns("equality relation on continuous variables")
    ctx.nrel[] += 1
    return Rel(op, l, r, ctx.nrel[])
  elseif e isa SC.IFEXP
    return Op(:if, IR[toIR(g, e.cond, ctx), toIR(g, e.thenExp, ctx), toIR(g, e.elseExp, ctx)])
  elseif e isa SC.TSUB
    #= its index the position in the generated function's result (a record output's fields in place, SC.TSUB) =#
    e.ty isa Union{SC.TYPE_ARRAY, SC.TYPE_COMPLEX} && ns("an array or record output: $(_str(e))")
    return Op(:index, IR[toIR(g, e.exp, ctx), Lit(e.index)])
  elseif e isa SC.CALL
    return _callIRx(g, e, ctx)
  elseif e isa SC.ASUB
    return _subscriptedIRx(g, e, ctx)
  elseif e isa SC.RSUB
    return _recordElementIRx(g, e, ctx)
  end
  ns("expression $(nameof(typeof(e))): $(_str(e))")
end

"""
The String parameter of a literal binding that `name` is, through its references to others (`s2 = s1`, kept by the
arrays-kept passes), and its literal; nothing where it has none.
"""
function _stringRoot(g::Gen, name::String)::Union{Nothing, Tuple{String, SC.SCONST}}
  local seen::Set{String} = Set{String}()
  local nm::String = name
  while !(nm in seen)
    push!(seen, nm)
    local info::Union{Nothing, VarInfo} = get(g.vars, nm, nothing)
    (info !== nothing && info.kind == :param && info.vtype == :other) || return nothing
    local b::Union{Nothing, SC.Exp} = SC.bindingOf(info.var.simVar.varKind)
    b isa SC.SCONST && return (nm, b)
    (b isa SC.EXP_CREF && isempty(b.cref.subs)) || return nothing
    nm = String(b.cref.sym)
  end
  return nothing
end

"""
A read of the String parameter `name` of the translated value `value`: its slot in the run state (`Gen.strings`; an
override reaches it), its value and name (`Op(:stringparam)`; a parameter binding or a static condition reads the value,
which fixes it).
"""
function _stringRead(g::Gen, name::String, value::String)::IR
  local k::Union{Nothing, Int} = findfirst(s::Tuple{String, String} -> s[1] == name, g.strings)
  k === nothing && (push!(g.strings, (name, value)); k = length(g.strings))
  return Op(:stringparam, IR[Lit(k), Lit(value), Lit(name)])
end

"""
A reference of the simulation code: a for-equation's iterator its slot, `time`, a variable (its element by its literal
subscripts); a String parameter of a literal binding (an assert's message: CDL Utilities.Assert), through references
to others, its read (`_stringRead`).
"""
function _crefIRx(g::Gen, e::SC.EXP_CREF, ctx::Ctx)::IR
  local c::SC.SimCref = e.cref
  local own::String = String(c.sym)
  if isempty(c.subs)
    haskey(ctx.iters, own) && return Slot(ctx.iters[own])
    (own == "time" && !haskey(g.vars, own)) && return TimeIR()
    local root::Union{Nothing, Tuple{String, SC.SCONST}} = _stringRoot(g, own)
    root === nothing || return _stringRead(g, root[1], root[2].value)
  end
  (haskey(g.vars, own) || haskey(g.derOf, own)) || ns("unknown variable $(own)")
  return _elementRefIR(g, own, IR[Lit(k) for k in c.subs])
end

"""
The element `subs` (all of its subscripts, or none for a scalar) of the variable `name`; of a state derivative's link
(`\$STATEDER.x`), or `der`, der of the state's element.
"""
function _elementRefIR(g::Gen, name::String, subs::Vector{IR}; der::Bool = false)::IR
  local link::Bool = haskey(g.derOf, name)
  local target::String = link ? g.derOf[name] : name
  local info::Union{Nothing, VarInfo} = get(g.vars, target, nothing)
  info === nothing && ns("unknown variable $(target)")
  isempty(subs) && !isempty(info.dims) && ns("the array $(target) as a whole in an equation of scalars")
  length(subs) == length(info.dims) || ns("partial subscripts on $(target)")
  (der || link) && info.kind != :state && ns("der of non-state $(target)")
  return Ref(target, subs, der || link)
end

"""
An element of an array value: of an array variable (or `der` of one) its element; of an array literal the element at
the subscripts (the literal array indexed, else the vector of its elements); of a function's array result or a
record's array field the value indexed.
"""
function _subscriptedIRx(g::Gen, e::SC.ASUB, ctx::Ctx)::IR
  any((s::SC.Exp) -> s isa SC.WHOLE_DIMENSION || s isa SC.RANGE, e.subs) && ns("a slice of $(_str(e))")
  local subs::Vector{IR} = IR[toIR(g, s, ctx) for s in e.subs]
  local base::SC.Exp = e.exp
  base isa SC.EXP_CREF && isempty(base.cref.subs) && return _elementRefIR(g, String(base.cref.sym), subs)
  base isa SC.DER && return _elementRefIR(g, String(base.cref.sym), subs; der = true)
  #= an element of pre() of an array (a function's array argument, `f(pre(state))`): pre() of the element =#
  if base isa SC.PRE && isempty(base.cref.subs)
    local element::IR = _elementRefIR(g, String(base.cref.sym), subs)
    element isa Ref || ns("an element of $(_str(base))")
    return Pre(element)
  end
  if base isa SC.ARRAY_EXP
    local literal::Union{Nothing, Array} = _literalArray(base)
    literal === nothing || return Op(:index, IR[Lit(literal); subs])
    length(subs) == 1 && return Op(:index, IR[Op(:vect, IR[toIR(g, x, ctx) for x in base.elements]), subs[1]])
    #= an array of rows by several subscripts (a built-in's elements, `cat(2, ...)[\$i1, \$i2]`): its value indexed =#
    return Op(:index, IR[_valueIRx(g, base, ctx); subs])
  end
  (_isFunctionCall(g, base) || base isa SC.RSUB) || ns("an element of $(_str(base))")
  return Op(:index, IR[_valueIRx(g, base, ctx); subs])
end

"The value of an array literal of literals (a vector, a matrix of rows of equal length); `nothing` for another."
function _literalArray(e::SC.ARRAY_EXP)::Union{Nothing, Array}
  isempty(e.elements) && return nothing
  all((x::SC.Exp) -> x isa Union{SC.ICONST, SC.RCONST, SC.BCONST}, e.elements) && return map((x::SC.Exp) -> x.value, e.elements)
  all((x::SC.Exp) -> x isa SC.ARRAY_EXP, e.elements) || return nothing
  local rows::Vector{Union{Nothing, Array}} = Union{Nothing, Array}[_literalArray(x) for x in e.elements]
  all((r::Union{Nothing, Array}) -> r !== nothing && ndims(r) == 1 && length(r) == length(rows[1]), rows) || return nothing
  return permutedims(reduce(hcat, rows))
end

"Whether `e` calls a Modelica function of the model (`SIM_CODE.functions`: `Gen.functions`, their outputs' number)."
_isFunctionCall(g::Gen, e::SC.Exp)::Bool = e isa SC.CALL && e.kind != SC.CALL_BUILTIN && haskey(g.functions, String(e.name))

"The dimensions of an array type of known sizes; NotSupported for another."
function _knownDims(ty::SC.SType, what::String)::Vector{Int}
  ty isa SC.TYPE_ARRAY || ns(what)
  any((d::Int) -> d < 0, ty.dims) && ns(what)
  return ty.dims
end

"""
The value of an expression as a function's argument: a scalar's (`toIR`); an array literal's the array of its elements
(column-major, `Op(:arrayval)`; a constant one its value); of a function's array result the call's, of a record's array
field the field's.
"""
function _valueIRx(g::Gen, e::SC.Exp, ctx::Ctx)::IR
  _isFunctionCall(g, e) && e.ty isa SC.TYPE_ARRAY && return _functionCallIRx(g, e::SC.CALL, ctx)
  e isa SC.RSUB && e.ty isa SC.TYPE_ARRAY && return _recordElementIRx(g, e, ctx)
  e isa SC.ARRAY_EXP || return toIR(g, e, ctx)
  local dims::Vector{Int} = _knownDims(e.ty, "an array argument of a size not known: $(_str(e))")
  prod(dims) <= Passes.MAX_ELEMENT_TERMS || ns("an array argument of $(prod(dims)) elements")
  local rowMajor::Vector{SC.Exp} = _rowMajorElements(e)
  local elements::Vector{IR} = IR[]
  for t in Iterators.product((1:d for d in dims)...)   #= column-major: the first index fastest =#
    push!(elements, toIR(g, rowMajor[Passes.IndexReduction.rowMajorIndex(collect(t), dims)], ctx))
  end
  #= a constant array: its value in the code =#
  if all((x::IR) -> x isa Lit, elements)
    local constant::Vector = map((x::IR) -> (x::Lit).v, elements)
    return Lit(length(dims) == 1 ? constant : reshape(constant, dims...))
  end
  return Op(:arrayval, IR[Lit(Tuple(dims)); elements])
end

"""
A field of a record value: of a function's record result the call's tuple at the field (the function returns a record
by its fields, `generateIOL`).
"""
function _recordElementIRx(g::Gen, e::SC.RSUB, ctx::Ctx)::IR
  _isFunctionCall(g, e.exp) || ns("a field of $(_str(e.exp))")
  local rty::SC.SType = (e.exp::SC.CALL).ty
  (rty isa SC.TYPE_COMPLEX && rty.isRecord) || ns("a field of $(_str(e.exp)), not a record")
  any((t::SC.SType) -> t isa SC.TYPE_COMPLEX && t.isRecord, rty.fieldTypes) && ns("a record result with record fields: $(_str(e.exp))")
  local call::IR = _functionCallIRx(g, e.exp::SC.CALL, ctx)
  return g.functions[String((e.exp::SC.CALL).name)] == 1 ? call : Op(:index, IR[call, Lit(e.index)])
end

"""
An argument's values (onto `out`): a record literal's fields each (a generated function takes a record by its fields,
`R_T`, `R_w`: `generateIOL`), an array's or a scalar's value (`_valueIRx`).
"""
function _argumentsIRx!(out::Vector{IR}, g::Gen, a::SC.Exp, ctx::Ctx)::Nothing
  if a isa SC.RECORD
    for x in a.exps
      (x isa SC.RECORD || (x isa SC.EXP_CREF && x.ty isa SC.TYPE_COMPLEX)) && ns("a record argument with record fields: $(_str(a))")
      push!(out, _valueIRx(g, x, ctx))
    end
  elseif a isa SC.EXP_CREF && a.ty isa SC.TYPE_COMPLEX
    ns("a record argument: $(_str(a))")
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
function _functionCallIRx(g::Gen, e::SC.CALL, ctx::Ctx)::IR
  local jn::String = String(e.name)
  local callArgs::Vector{IR} = IR[]
  for a in e.args
    _argumentsIRx!(callArgs, g, a, ctx)
  end
  local call::IR = Op(:fcall, IR[Lit(Symbol(jn)), callArgs...])
  #= the outputs counted as returned, a record by its fields (generateIOL): a record result is the tuple's first =#
  g.functions[jn] > 1 && !(e.ty isa SC.TYPE_TUPLE || (e.ty isa SC.TYPE_COMPLEX && e.ty.isRecord)) && (call = Op(:index, IR[call, Lit(1)]))
  return call
end

"""
    _callIRx(g, e, ctx) -> IR

A call of the simulation code in the domain level's form (`toIR`): `sample` (in a when-condition) and the event
operators, the built-in functions (event-generating ones as event functions), a Modelica function of the model (a
scalar result; an array or record one only under a subscript or a field); NotSupported for `delay` and another.
"""
function _callIRx(g::Gen, e::SC.CALL, ctx::Ctx)::IR
  local fname::String = String(e.name)
  local args::Vector{SC.Exp} = e.args
  local builtin::Bool = e.kind == SC.CALL_BUILTIN
  builtin && fname in ("delay", "OpenModelica.Internal.delay2", "OpenModelica.Internal.delay3") && ns("delay()")
  if fname == "der"
    ns("der() in a reduced model: $(_str(e))")
  elseif builtin && (fname == "sample" || endswith(fname, ".sample"))
    #= in a when-condition (an equation refuses it: ctx.nsam) =#
    length(args) == 2 || ns("sample with $(length(args)) arguments")
    ctx.nsam[] += 1
    return Op(:sample, IR[toIR(g, args[1], ctx), toIR(g, args[2], ctx), Lit(ctx.nsam[])])
  elseif fname == "pre"
    local preRef::IR = toIR(g, args[1], ctx)
    preRef isa Ref || ns("pre of $(_str(e))")
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
  elseif fname == "terminal"
    ns("terminal() in a when-condition: $(_str(e))")
  elseif fname == "edge" || fname == "change"
    local changed::IR = toIR(g, args[1], ctx)
    changed isa Ref || ns("$(fname) of $(_str(args[1]))")
    return fname == "edge" ? Op(:&&, IR[changed, Op(:!, IR[Pre(changed)])]) : Op(:!=, IR[changed, Pre(changed)])
  elseif fname == "semiLinear"
    local x::IR = toIR(g, args[1], ctx)
    return Op(:if, IR[Op(:>=, IR[x, Lit(0.0)]), Op(:*, IR[x, toIR(g, args[2], ctx)]), Op(:*, IR[x, toIR(g, args[3], ctx)])])
  elseif fname in ("sum", "product") || (fname in ("min", "max") && length(args) == 1)
    #= of an array literal (a start's `sum({q[1], q[2], q[3]})`): its elements folded; else the lowering's element form
       wrote out what it could =#
    local elements::Vector{SC.Exp} = _literalElements!(SC.Exp[], args[1])
    (args[1] isa SC.ARRAY_EXP && !isempty(elements)) || ns("$(fname) of $(_str(args[1]))")
    local fold::Symbol = fname == "sum" ? :+ : fname == "product" ? :* : Symbol(fname)
    return foldl((a::IR, b::IR) -> Op(fold, IR[a, b]), IR[toIR(g, x, ctx) for x in elements])
  elseif fname == "String"
    #= String(r, significantDigits, minimumLength, leftJustified), String(r, format), String(i|b|e, minimumLength,
       leftJustified): OpenModelica's formatting (modelica_String) =#
    local strs::Vector{IR} = IR[toIR(g, a, ctx) for a in args]
    length(strs) == 2 && return Op(:String, strs)
    length(strs) == 4 && return Op(:String, IR[strs[1], Op(:Int, IR[strs[2]]), Op(:Int, IR[strs[3]]), Op(:Bool, IR[strs[4]])])
    length(strs) == 3 || ns("String with $(length(strs)) arguments")
    local strTy::SC.SType = _expType(args[1])
    local strValue::IR = strTy isa SC.TYPE_ENUM ?
      Op(:index, IR[Lit(Tuple(String[String(l) for l in strTy.names])), Op(:Int, IR[strs[1]])]) :
      strTy isa SC.TYPE_BOOL ? Op(:Bool, IR[strs[1]]) :
      strTy isa SC.TYPE_INTEGER ? Op(:Int, IR[strs[1]]) : ns("String of $(_str(args[1]))")
    return Op(:String, IR[strValue, Op(:Int, IR[strs[2]]), Op(:Bool, IR[strs[3]])])
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
  #= a Modelica function of the model (SIM_CODE.functions); its array or record result only under a subscript or
     a field (_subscriptedIRx, _recordElementIRx) =#
  if _isFunctionCall(g, e)
    (e.ty isa SC.TYPE_ARRAY || (e.ty isa SC.TYPE_COMPLEX && e.ty.isRecord)) &&
      ns("function $(fname) of an array or record result outside an element")
    return _functionCallIRx(g, e, ctx)
  end
  local mathName::String = startswith(fname, "Modelica_Math_") ? fname[15:end] : startswith(fname, "Modelica.Math.") ? fname[15:end] : ""
  local jf::Union{Nothing, Symbol} = (!isempty(mathName) && mathName in MATH_WRAPPED) ? MATH_BUILTINS[mathName] : get(MATH_BUILTINS, fname, nothing)
  jf === nothing && ns("function $(fname)")
  return Op(jf, IR[toIR(g, a, ctx) for a in args])
end

"""
The type of a simulation code expression: a reference's, a call's, a field's, an output's, a cast's, `pre`'s as given;
an element's its array's element type; a binary operation's by its operator (`_binaryType`), a unary one's its
operand's; a relation's or logical operation's Boolean; an if-expression's its branches'; a constant's. Unknown for
another (an enumeration literal, whose type the simulation code does not hold).
"""
function _expType(e::SC.Exp)::SC.SType
  e isa Union{SC.EXP_CREF, SC.CALL, SC.CAST, SC.PRE, SC.RSUB, SC.TSUB, SC.ARRAY_EXP} && return e.ty
  e isa SC.ICONST && return SC.TYPE_INTEGER()
  e isa SC.BCONST && return SC.TYPE_BOOL()
  e isa SC.RCONST && return SC.TYPE_REAL()
  e isa SC.SCONST && return SC.TYPE_STRING()
  e isa SC.DER && return SC.TYPE_REAL()
  e isa Union{SC.RELATION, SC.LBINARY, SC.LUNARY} && return SC.TYPE_BOOL()
  e isa SC.UNARY && return _expType(e.exp)
  if e isa SC.ASUB
    local base::SC.SType = _expType(e.exp)
    return base isa SC.TYPE_ARRAY ? base.elementType : SC.TYPE_UNKNOWN()
  elseif e isa SC.BINARY
    return _binaryType(e)
  elseif e isa SC.IFEXP
    local branch::SC.SType = _expType(e.thenExp)
    return branch isa SC.TYPE_UNKNOWN ? _expType(e.elseExp) : branch
  end
  return SC.TYPE_UNKNOWN()
end

"""
    _binaryType(e) -> SType

The type of a binary operation by its operator: its element type Real for `/` and `^` (MLS 10.6), else of its operands'
element types String if both are, Integer if both are, Real if both are numbers (unknown if one is unknown); its shape
an array operand's for the element-wise and the scalar-array operations, a matrix's and a vector's products' by their
rules, a scalar product's a scalar's.
"""
function _binaryType(e::SC.BINARY)::SC.SType
  local t1::SC.SType = _expType(e.exp1)
  local t2::SC.SType = _expType(e.exp2)
  local el1::SC.SType = t1 isa SC.TYPE_ARRAY ? t1.elementType : t1
  local el2::SC.SType = t2 isa SC.TYPE_ARRAY ? t2.elementType : t2
  local op::SC.OpKind = e.op
  local real::Bool = op in (SC.OP_DIV, SC.OP_DIV_EW, SC.OP_DIV_ARRAY_SCALAR, SC.OP_DIV_SCALAR_ARRAY, SC.OP_POW, SC.OP_POW_EW,
                            SC.OP_POW_SCALAR_ARRAY, SC.OP_POW_ARRAY_SCALAR, SC.OP_POW_MATRIX)
  local el::SC.SType = real ? SC.TYPE_REAL() :
    (el1 isa SC.TYPE_UNKNOWN || el2 isa SC.TYPE_UNKNOWN) ? SC.TYPE_UNKNOWN() :
    (el1 isa SC.TYPE_STRING && el2 isa SC.TYPE_STRING) ? SC.TYPE_STRING() :
    (el1 isa SC.TYPE_INTEGER && el2 isa SC.TYPE_INTEGER) ? SC.TYPE_INTEGER() :
    (el1 isa Union{SC.TYPE_REAL, SC.TYPE_INTEGER} && el2 isa Union{SC.TYPE_REAL, SC.TYPE_INTEGER}) ? SC.TYPE_REAL() :
    SC.TYPE_UNKNOWN()
  local dims1::Vector{Int} = t1 isa SC.TYPE_ARRAY ? t1.dims : Int[]
  local dims2::Vector{Int} = t2 isa SC.TYPE_ARRAY ? t2.dims : Int[]
  local dims::Vector{Int} =
    op in (SC.OP_ADD_SCALAR_ARRAY, SC.OP_SUB_SCALAR_ARRAY, SC.OP_MUL_SCALAR_ARRAY, SC.OP_DIV_SCALAR_ARRAY, SC.OP_POW_SCALAR_ARRAY) ? dims2 :
    op == SC.OP_SCALAR_PRODUCT ? Int[] :
    op == SC.OP_MUL_MATRIX_VECTOR ? Int[_dimension(dims1, 1)] :
    op == SC.OP_MUL_VECTOR_MATRIX ? Int[_dimension(dims2, 2)] :
    op == SC.OP_MATRIX_PRODUCT ? Int[_dimension(dims1, 1), _dimension(dims2, 2)] :
    !isempty(dims1) ? dims1 : dims2
  return isempty(dims) ? el : SC.TYPE_ARRAY(el, dims)
end

"The size of dimension `k` of `dims`, `-1` (unknown) where it has none."
_dimension(dims::Vector{Int}, k::Int)::Int = k <= length(dims) ? dims[k] : -1

"""
The values of a for-range of Integer literals (`Passes.literalRanges!` folds the ones that fold); `nothing` for another.
"""
function _literalRange(e::SC.Exp)::Union{Nothing, StepRange{Int, Int}}
  (e isa SC.RANGE && e.start isa SC.ICONST && e.stop isa SC.ICONST && (e.step === nothing || e.step isa SC.ICONST)) || return nothing
  local s::Int = e.step === nothing ? 1 : (e.step::SC.ICONST).value
  s == 0 && return nothing
  return (e.start::SC.ICONST).value:s:(e.stop::SC.ICONST).value
end

"An equation of the simulation code in a text (a class's): `lhs = rhs`, another as it prints."
_eqText(eq::SC.EQUATION)::String = string(_str(eq.lhs), " = ", _str(eq.rhs))

_eqText(@nospecialize(eq))::String = _str(eq)

"""
Whether the initial equation `lhs = rhs` is an assignment, made once as an initial algorithm's (no unknown of the
initialization's solve): a scalar discrete variable's initial value (`d = expr`, CDL Discrete's `y = y_start`, assigned
after the solve), or a scalar parameter with fixed = false given the value of an expression of parameters (`p =
expr`): MSL GlobalSeed's `id_impure = initializeImpureRandom(seed)`, an impure call the solve's Newton made at every
evaluation, a new value each time (ImpureGenerator stalled). Not of free parameters the solve determines (a double
exponential's `eta = exp(-T/tau1) - exp(-T/tau2)`); those an earlier such assignment gives are known.
"""
function _initialAssignment(g::Gen, lhs::IR, rhs::IR)::Bool
  (lhs isa Ref && !lhs.der && isempty(g.vars[lhs.name].dims)) || return false
  local known::Function = (name::String) -> !(name in g.freeParams) ||
    any(a::AlgClass -> any(st::Stmt -> st isa SAssign && st.target.name == name, a.body), g.initAlgs)
  local reads::Vector{Ref} = accesses!(Ref[], rhs)
  g.vars[lhs.name].kind == :discrete &&
    return all(r::Ref -> r.name != lhs.name && (g.vars[r.name].kind != :param || known(r.name)), reads)
  lhs.name in g.freeParams || return false
  return all(r::Ref -> r.name != lhs.name && g.vars[r.name].kind == :param && known(r.name), reads)
end

"""
The initial equations: the assignments among them (a discrete variable's initial value, a parameter with fixed = false
given its value from parameters: `_initialAssignment`) as initial algorithms, in the order their parameters become
known (GlobalSeed's `seed`, then `id_impure = initializeImpureRandom(seed)`, whatever the equations' order); the others
as initial equations (`_addEquation!`).
"""
function _addInitialEquations!(g::Gen, equations::Vector{SC.Equation})::Nothing
  local pending::Vector{Tuple{SC.EQUATION, IR, IR, Int}} = Tuple{SC.EQUATION, IR, IR, Int}[]
  for eq in equations
    if eq isa SC.EQUATION && !(eq.lhs isa SC.TUPLE)
      local ctx::Ctx = Ctx(Dict{String, Int}(), Int[], Base.RefValue(0), true)
      local lhs::IR = toIR(g, eq.lhs, ctx)
      local rhs::IR = toIR(g, eq.rhs, ctx)
      ctx.nsam[] == 0 || ns("sample() in an equation")
      push!(pending, (eq, lhs, rhs, ctx.nrel[]))
    else
      _addEquation!(g, eq, String[], NO_SLOTS; initial = true)
    end
  end
  local assigned::Bool = true
  while assigned
    assigned = false
    for k in eachindex(pending)
      local (eq::SC.EQUATION, lhs::IR, rhs::IR, _) = pending[k]
      _initialAssignment(g, lhs, rhs) || continue
      push!(g.initAlgs, AlgClass(Stmt[SAssign(lhs::Ref, rhs)], 0, NO_SLOTS, 0, 0, _eqText(eq)))
      deleteat!(pending, k)
      assigned = true
      break
    end
  end
  for (eq, lhs, rhs, nrel) in pending
    push!(g.initClasses, EqClass(lhs, rhs, 0, NO_SLOTS, nrel, _eqText(eq)))
  end
  return nothing
end

"""
The classes of an equation of the simulation code: a for-equation's body over its iterator's range (the domain
crossed), a scalar equation one class over the domain, an assert one checked over it.
"""
function _addEquation!(g::Gen, eq::SC.Equation, iterNames::Vector{String}, idom::Matrix{Int}; initial::Bool = false)
  size(idom, 2) == 0 && return
  local iters::Dict{String, Int} = Dict{String, Int}(n => k for (k, n) in enumerate(iterNames))
  local nIter::Int = length(iterNames)
  if eq isa SC.FOR_EQUATION
    local range::Union{Nothing, StepRange{Int, Int}} = _literalRange(eq.iterator.range)
    range === nothing && ns("for range $(_str(eq.iterator.range))")
    local values::Vector{Int} = collect(Int, range)
    isempty(values) && return
    local dom::Matrix{Int} = _crossDomain(idom, values)
    for b in eq.body
      _addEquation!(g, b, [iterNames; String(eq.iterator.name)], dom; initial = initial)
    end
    return
  elseif eq isa SC.EQUATION && eq.lhs isa SC.TUPLE
    #= (a, b) = f(x): an algorithm node assigning each target its output =#
    local tupleCtx::Ctx = Ctx(iters, Int[])
    local tupleCall::IR = toIR(g, eq.rhs, tupleCtx)
    tupleCtx.nsam[] == 0 || ns("sample() in an equation")
    push!(g.algs, AlgClass(_tupleAssignments(g, eq.lhs, tupleCall, _expType(eq.rhs), tupleCtx), nIter, idom, tupleCtx.nrel[], 0,
                           _eqText(eq)))
    return
  elseif eq isa SC.EQUATION
    #= relations of initial equations are evaluated at the start (no events) =#
    local ctx::Ctx = Ctx(iters, Int[], Base.RefValue(0), initial)
    local lhs::IR = toIR(g, eq.lhs, ctx)
    local rhs::IR = toIR(g, eq.rhs, ctx)
    #= sample() in an initial equation: no ticks there =#
    (initial && ctx.nsam[] > 0) && ns("sample() in an initial equation")
    push!(initial ? g.initClasses : g.classes, EqClass(lhs, rhs, nIter, idom, ctx.nrel[], _eqText(eq), ctx.nsam[]))
    return
  elseif eq isa SC.INLINE_IF_EQUATION
    _addIfEquation!(g, eq, iterNames, idom; initial = initial)
    return
  elseif eq isa SC.ASSERT_EQUATION
    local actx::Ctx = Ctx(iters, Int[], Base.RefValue(0), true)
    local checked::IR = _checkedCondition(g, eq.condition, actx)
    push!(initial ? g.initAsserts : g.asserts,
          AssertClass(checked, _isWarning(eq.level), nIter, idom, _str(eq.condition), _str(eq.message),
                      _messageIRx(g, eq.message, actx)))
    return
  elseif eq isa SC.UNLOWERED_EQUATION
    ns("$(eq.what): $(eq.construct)")
  elseif eq isa SC.WHEN_EQUATION
    initial && ns("a when-equation among the initial equations: $(_str(eq))")
    _addWhen!(g, eq, iterNames, idom)
    return
  elseif eq isa SC.INITIAL_WHEN_EQUATION
    initial && ns("a when-equation among the initial equations: $(_str(eq))")
    _addInitialWhen!(g, eq, iterNames, idom)
    return
  end
  ns("$(nameof(typeof(eq))) in the simulation code: $(_str(eq))")
end

"The triggers of a when-branch's condition (`SimCode.lowerArraySystem`: the vector of several, none the empty one; one itself)."
_triggers(c::SC.Exp)::Vector{SC.Exp} = c isa SC.ARRAY_EXP ? c.elements : SC.Exp[c]

"Whether `e` is a reference: a variable, `der` of a state (an element of one too), as it is; no expression of it."
_isReference(e::SC.Exp)::Bool = e isa Union{SC.EXP_CREF, SC.DER} || (e isa SC.ASUB && e.exp isa Union{SC.EXP_CREF, SC.DER})

"""
A when-equation, over the domain: each branch's triggers its condition (several the vector of them, each an edge of its
own; a branch active at the initialization with initial() among them: `when initial()` alone; change(e) of a Boolean
expression the vector {e, not e}), its statements the body: assignments to discrete variables, reinit of states,
asserts (checked when it runs), calls, `terminate()`, and if-statements of the last three (`_whenEffect`).
"""
function _addWhen!(g::Gen, eq::SC.WHEN_EQUATION, iterNames::Vector{String}, idom::Matrix{Int})::Nothing
  local ctx::Ctx = Ctx(Dict{String, Int}(n => k for (k, n) in enumerate(iterNames)), Int[])
  local conds::Vector{IR} = IR[]
  local body::Vector{Tuple{Symbol, Ref, IR}} = Tuple{Symbol, Ref, IR}[]
  local starts::Vector{Int} = Int[]
  local noTarget::Ref = Ref("", IR[], false)
  local branch::Union{Nothing, SC.WHEN_STMTS} = eq.whenEquation
  while branch !== nothing
    local triggers::Vector{IR} = IR[]
    for c in _triggers(branch.condition)
      if c isa SC.CALL && c.kind == SC.CALL_BUILTIN && c.name === :change && !_isReference(c.args[1])
        #= change(e) of a Boolean expression (a lifted discrete cluster's trigger): e or not e becomes true =#
        local x::IR = toIR(g, c.args[1], ctx)
        _isBooleanIR(g, x) || ns("change of $(_str(c.args[1]))")
        append!(triggers, IR[x, Op(:!, IR[x])])
      else
        push!(triggers, _topLevelInitial(toIR(g, c, ctx)))
      end
    end
    isempty(triggers) && ns("a when-branch without a condition: $(_str(eq))")
    push!(conds, length(triggers) == 1 ? triggers[1] : Op(:vect, triggers))
    push!(starts, length(body) + 1)
    for st in branch.body
      if st isa SC.ASSIGN && st.left isa SC.TUPLE && _isFunctionCall(g, st.right)
        #= (r, state) = f(pre(state)): the call once, each target from its output (Buildings.Occupants' random numbers) =#
        append!(body, _whenTupleAssignments(g, st.left, toIR(g, st.right, ctx), _expType(st.right), ctx))
      elseif st isa SC.ASSIGN
        _isReference(st.left) || ns("a when-statement assigning $(_str(st.left))")
        local target::IR = toIR(g, st.left, ctx)
        (target isa Ref && g.vars[target.name].kind == :discrete) || ns("when assigns a non-discrete variable: $(_str(st.left))")
        push!(body, (:assign, target, toIR(g, st.right, ctx)))
      elseif st isa SC.REINIT
        #= of an element of an array state: the state at the element's subscripts =#
        local stateRef::SC.Exp = SC.EXP_CREF(st.stateVar, SC.TYPE_REAL())
        local state::IR = toIR(g, isempty(st.subscripts) ? stateRef : SC.ASUB(stateRef, st.subscripts), ctx)
        (state isa Ref && g.vars[state.name].kind == :state) || ns("reinit of a non-state: $(_str(eq))")
        push!(body, (:reinit, state, toIR(g, st.value, ctx)))
      elseif st isa Union{SC.ASSERT, SC.NORETCALL, SC.TERMINATE, SC.IF}
        push!(body, (:call, noTarget, _whenEffect(g, st, ctx)))
      else
        ns("a when-statement $(nameof(typeof(st))): $(_str(eq))")
      end
    end
    branch = branch.elsewhenPart
  end
  local cond::IR = length(conds) == 1 ? conds[1] : Op(:elsewhen, conds)
  push!(g.whens, WhenClass(cond, body, starts, length(iterNames), idom, ctx.nrel[], ctx.nsam[], _str(eq), eq.attr.everyEvent))
  return nothing
end

"""
A when body's statement for its effects, as the expression the body runs: an assert (checked when it runs), a call, a
`terminate()` (the run ends after the event: `_terminate!`), an if-statement of them (`SimCode.whenOperators`: under
their branch's condition), its conditions evaluated at the instant.
"""
function _whenEffect(g::Gen, st::SC.Statement, ctx::Ctx)::IR
  local actx::Ctx = _with(ctx; noEvent = true)
  if st isa SC.ASSERT
    push!(g.assertTexts, (_str(st.condition), _str(st.message), _isWarning(st.level)))
    return Op(:assertcall, IR[toIR(g, st.condition, actx), _messageIRx(g, st.message, actx), Lit(length(g.assertTexts))])
  elseif st isa SC.NORETCALL
    return toIR(g, st.exp, actx)
  elseif st isa SC.TERMINATE
    return Op(:terminate, IR[_messageIRx(g, st.message, actx)])
  elseif st isa SC.IF
    local chain::IR = Op(:seq, IR[_whenEffect(g, x, ctx) for x in st.elseBody])
    for b in Iterators.reverse(st.branches)
      chain = Op(:if, IR[toIR(g, b.condition, actx), Op(:seq, IR[_whenEffect(g, x, ctx) for x in b.body]), chain])
    end
    return chain
  end
  ns("a when-statement $(nameof(typeof(st))): $(_str(st))")
end

"""
The statements of `(a, _, c) = call` (a tuple equation): the call once, into a temporary, and an assignment of each
target that is not a wildcard from its output (at its position in the result, `SimCode.outputPosition` of the call's type
`callType`: a record output before it is its fields); a whole array target element by element from that output
(`(r, state) = random(pre(state))`: Buildings.Occupants).
"""
function _tupleAssignments(g::Gen, lhs::SC.TUPLE, call::IR, callType::SC.SType, ctx::Ctx)::Vector{Stmt}
  local out::Vector{Stmt} = Stmt[]
  local res::Slot = Slot(g.tmpSlot -= 1)
  for (i, e) in enumerate(lhs.elements)
    e isa SC.WILD && continue
    _isReference(e) || ns("tuple target $(_str(e))")
    if e isa SC.EXP_CREF && isempty(e.cref.subs) && haskey(g.vars, String(e.cref.sym)) && !isempty(g.vars[String(e.cref.sym)].dims)
      local name::String = String(e.cref.sym)
      local info::VarInfo = g.vars[name]
      info.kind in (:alg, :discrete) || ns("tuple target $(_str(e))")
      for t in Iterators.product((1:d for d in info.dims)...)
        local subs::Vector{IR} = IR[Lit(k) for k in t]
        push!(out, SAssign(Ref(name, subs, false), Op(:index, IR[Op(:index, IR[res, Lit(SC.outputPosition(callType, i))]), subs...])))
      end
      continue
    end
    local target::IR = toIR(g, e, ctx)
    (target isa Ref && g.vars[target.name].kind in (:alg, :discrete)) || ns("tuple target $(_str(e))")
    push!(out, SAssign(target, Op(:index, IR[res, Lit(SC.outputPosition(callType, i))])))
  end
  isempty(out) || pushfirst!(out, SCall(Op(:setlocal, IR[res, call])))
  return out
end

"""
The body entries of a tuple assignment in a when-branch (`_addWhen!`): the call once into a temporary (a `:call` entry),
then an assignment of each target that is not a wildcard from its output (at its position in the result,
`SimCode.outputPosition`), a whole array target element by element; the targets discrete variables.
"""
function _whenTupleAssignments(g::Gen, lhs::SC.TUPLE, call::IR, callType::SC.SType, ctx::Ctx)::Vector{Tuple{Symbol, Ref, IR}}
  local out::Vector{Tuple{Symbol, Ref, IR}} = Tuple{Symbol, Ref, IR}[]
  local res::Slot = Slot(g.tmpSlot -= 1)
  for (i, e) in enumerate(lhs.elements)
    e isa SC.WILD && continue
    _isReference(e) || ns("tuple target $(_str(e))")
    local output::IR = Op(:index, IR[res, Lit(SC.outputPosition(callType, i))])
    if e isa SC.EXP_CREF && isempty(e.cref.subs) && haskey(g.vars, String(e.cref.sym)) && !isempty(g.vars[String(e.cref.sym)].dims)
      local name::String = String(e.cref.sym)
      local info::VarInfo = g.vars[name]
      info.kind == :discrete || ns("when assigns a non-discrete variable: $(_str(e))")
      for t in Iterators.product((1:d for d in info.dims)...)
        local subs::Vector{IR} = IR[Lit(k) for k in t]
        push!(out, (:assign, Ref(name, subs, false), Op(:index, IR[output; subs])))
      end
      continue
    end
    local target::IR = toIR(g, e, ctx)
    (target isa Ref && g.vars[target.name].kind == :discrete) || ns("when assigns a non-discrete variable: $(_str(e))")
    push!(out, (:assign, target, output))
  end
  isempty(out) || pushfirst!(out, (:call, Ref("", IR[], false), Op(:setlocal, IR[res, call])))
  return out
end

"Whether `x` is Boolean-valued in the code: a relation, a logical operation, a Boolean variable or its pre value."
function _isBooleanIR(g::Gen, x::IR)::Bool
  x isa Rel && return true
  x isa Lit && return x.v isa Bool
  x isa Op && return x.op in (:!, :&&, :||, :initial)
  x isa Ref && return g.vars[x.name].vtype == :bool
  x isa Pre && return _isBooleanIR(g, x.ref)
  return false
end

"""
An initial when-equation (`INITIAL_WHEN_EQUATION`: an initial algorithm section, an algorithm's `when initial()`, the
initial arm of a lifted algorithm or discrete cluster), over the domain: an initial algorithm, run once at the
initialization after the free states are solved, before the first event iteration (its relations literal); it assigns
discrete variables and parameters with fixed = false (`_initAlgTargets`).
"""
function _addInitialWhen!(g::Gen, eq::SC.INITIAL_WHEN_EQUATION, iterNames::Vector{String}, idom::Matrix{Int})::Nothing
  local w::SC.WHEN_STMTS = eq.whenEquation
  (w.elsewhenPart === nothing && isempty(_triggers(w.condition))) || ns("an initial when-equation with triggers: $(_str(eq))")
  local ctx::Ctx = Ctx(Dict{String, Int}(n => k for (k, n) in enumerate(iterNames)), Int[], Base.RefValue(0), true)
  local body::Vector{Stmt} = Stmt[_stmtIR(g, st, ctx) for st in w.body]
  ctx.nsam[] == 0 || ns("sample() in an initial algorithm: $(_str(eq))")
  _initAlgTargets(g, body)
  push!(g.initAlgs, AlgClass(body, length(iterNames), idom, 0, 0, _str(eq)))
  return nothing
end

"""
A statement of an algorithm the array code runs (an initial algorithm's): assignments to a scalar variable (an
algebraic or discrete one, a parameter with fixed = false; to an element chosen by an if-expression, an if-statement),
if, for (over a literal range: its iterator a temporary slot), while, assert, a call without a result, break,
`terminate()` (the solve ends at its start). Not when-statements (an initial algorithm has none), reinit, return.
"""
function _stmtIR(g::Gen, st::SC.Statement, ctx::Ctx)::Stmt
  if st isa SC.ASSIGN && st.left isa SC.IFEXP
    #= an element of a split array by a variable index (`if i == 1 then y[1] else ...`): each branch assigns its element =#
    return SIf(Tuple{IR, Vector{Stmt}}[(toIR(g, st.left.cond, ctx), Stmt[_stmtIR(g, SC.ASSIGN(st.left.thenExp, st.right, st.source), ctx)]),
                                       (Lit(true), Stmt[_stmtIR(g, SC.ASSIGN(st.left.elseExp, st.right, st.source), ctx)])])
  elseif st isa SC.ASSIGN
    _isReference(st.left) || ns("an assignment to $(_str(st.left))")
    local target::IR = toIR(g, st.left, ctx)
    (target isa Ref && (g.vars[target.name].kind in (:alg, :discrete) || target.name in g.freeParams)) ||
      ns("an assignment to $(_str(st.left))")
    return SAssign(target, toIR(g, st.right, ctx))
  elseif st isa SC.IF
    local branches::Vector{Tuple{IR, Vector{Stmt}}} =
      Tuple{IR, Vector{Stmt}}[(toIR(g, b.condition, ctx), Stmt[_stmtIR(g, x, ctx) for x in b.body]) for b in st.branches]
    isempty(st.elseBody) || push!(branches, (Lit(true), Stmt[_stmtIR(g, x, ctx) for x in st.elseBody]))
    return SIf(branches)
  elseif st isa SC.FOR
    local range::Union{Nothing, StepRange{Int, Int}} = _literalRange(st.iterator.range)
    range === nothing && ns("for range $(_str(st.iterator.range))")
    local k::Int = (g.tmpSlot -= 1)
    local inner::Ctx = Ctx(merge(ctx.iters, Dict{String, Int}(String(st.iterator.name) => k)), ctx.elem, ctx.nrel, ctx.noEvent, ctx.nsam)
    return SFor(k, collect(range), Stmt[_stmtIR(g, x, inner) for x in st.body])
  elseif st isa SC.WHILE
    return SWhile(toIR(g, st.condition, _with(ctx; noEvent = true)), Stmt[_stmtIR(g, x, ctx) for x in st.body])
  elseif st isa SC.ASSERT
    push!(g.assertTexts, (_str(st.condition), _str(st.message), _isWarning(st.level)))
    local actx::Ctx = _with(ctx; noEvent = true)
    return SAssert(toIR(g, st.condition, actx), length(g.assertTexts), _messageIRx(g, st.message, actx))
  elseif st isa SC.NORETCALL
    return SCall(toIR(g, st.exp, _with(ctx; noEvent = true)))
  elseif st isa SC.TERMINATE
    return SCall(Op(:terminate, IR[_messageIRx(g, st.message, _with(ctx; noEvent = true))]))
  elseif st isa SC.BREAK
    return SBreak()
  end
  ns("a statement $(nameof(typeof(st))): $(_str(st))")
end

"""
An if-equation: on conditions of parameters and iterators each instance its branch's equations; on a variable
condition each variable its branches assign (`v = e` in every branch and the else) one class
`v = if c1 then e1 elseif ... else en`.
"""
function _addIfEquation!(g::Gen, eq::SC.INLINE_IF_EQUATION, iterNames::Vector{String}, idom::Matrix{Int}; initial::Bool = false)
  local iters::Dict{String, Int} = Dict{String, Int}(n => k for (k, n) in enumerate(iterNames))
  local nIter::Int = length(iterNames)
  local bodies::Vector{Vector{SC.Equation}} = Vector{SC.Equation}[eq.branchesTrue; [eq.branchElse]]
  local staticCtx::Ctx = Ctx(iters, Int[], Base.RefValue(0), true)
  local conditions::Vector{IR} = IR[toIR(g, c, staticCtx) for c in eq.conditions]
  if all(c::IR -> _isStatic(g, c), conditions)
    local groups::Vector{Vector{Int}} = Vector{Int}[Int[] for _ in bodies]
    for j in 1:size(idom, 2)
      local k::Union{Nothing, Int} = findfirst(c::IR -> _evalStatic(g, c, idom, j) == true, conditions)
      push!(groups[k === nothing ? length(bodies) : k], j)
    end
    for (body, cols) in zip(bodies, groups)
      isempty(cols) && continue
      foreach(b::SC.Equation -> _addEquation!(g, b, iterNames, idom[:, cols]; initial = initial), body)
    end
    return
  end
  local keyed::Vector{Dict{String, SC.EQUATION}} = Dict{String, SC.EQUATION}[]
  for body in bodies
    local assigned::Dict{String, SC.EQUATION} = Dict{String, SC.EQUATION}()
    for b in body
      (b isa SC.EQUATION && _isReference(b.lhs)) || ns("if-equation on a variable condition: branch equation $(_eqText(b))")
      local key::String = string(b.lhs)
      haskey(assigned, key) && ns("if-equation on a variable condition: two equations for $(key)")
      assigned[key] = b
    end
    push!(keyed, assigned)
  end
  all(d::Dict{String, SC.EQUATION} -> Set(keys(d)) == Set(keys(keyed[1])), keyed) ||
    ns("if-equation on a variable condition: branches give different variables")
  for key in sort!(collect(keys(keyed[1])))
    local ctx::Ctx = Ctx(iters, Int[], Base.RefValue(0), initial)
    local lhs::IR = toIR(g, keyed[1][key].lhs, ctx)
    local rhs::IR = toIR(g, keyed[end][key].rhs, ctx)
    for i in length(eq.conditions):-1:1
      rhs = Op(:if, IR[toIR(g, eq.conditions[i], ctx), toIR(g, keyed[i][key].rhs, ctx), rhs])
    end
    ctx.nsam[] == 0 || ns("sample() in an equation")
    push!(initial ? g.initClasses : g.classes,
          EqClass(lhs, rhs, nIter, idom, initial ? 0 : ctx.nrel[], "$(key) = if ... ($(_str(eq.conditions[1])))"))
  end
end

"""
The values per element (row-major) of a variable's start or fixed attribute as the lowering evaluated them
(`SimCode.evaluatedVariable`): `default` for each where it has none, the constants where it is one; NotSupported naming
the variable's first element where the attribute did not evaluate.
"""
function _attributeValues(a::Union{Nothing, SC.Exp}, n::Int, default::Any, what::String, firstName::String)::Vector{Any}
  a === nothing && return Any[default for _ in 1:n]
  local vals::Vector{Any} = try
    _constantValues(a)
  catch err
    err isa NotSupported || rethrow(err)
    ns("the $(what) of $(firstName) does not evaluate")
  end
  length(vals) == n || ns("the $(what) of $(firstName) does not evaluate")
  return vals
end

"""
    DomainSystem(g, u0, d0, a0, fixedMask, stateNames, algebraicNames, discreteNames, keepsArrays, startRules,
                 startFixed, placeholderStarts)

The domain level of the simulation code of a model with its arrays kept (`domainSystem`): its variables and classes
(`g`), the initial values of the states, discrete and algebraic variables (row-major), the states' fixed attributes,
the element names of each buffer, whether it keeps arrays an algebraic loop may go through (its own, or rolled
components of a reduced model). `startRules`: the starts of states (`:u`) and discrete variables (`:d`) that read a
parameter, by their index in the buffer, as expressions the array code computes again from overridden parameters;
`startFixed`: the parameters a start reads that it cannot compute (an override of them refused); `placeholderStarts`:
a start did not evaluate at the translation (it reads a parameter computed at the start, a reduction the evaluation
does not take): a placeholder in `u0`, `d0`, its rule computes it at each start.
"""
struct DomainSystem
  g::Gen
  u0::Vector{Float64}
  d0::Vector{Float64}
  a0::Vector{Float64}
  fixedMask::Vector{Bool}
  stateNames::Vector{String}
  algebraicNames::Vector{String}
  discreteNames::Vector{String}
  keepsArrays::Bool
  startRules::Vector{Tuple{Symbol, Int, IR}}
  startFixed::Vector{String}
  placeholderStarts::Bool
end

"""
The element starts of `sv` that read a parameter (`SC.SimVarAttributes`' `startExps`) as the domain level's
expressions; nothing where it has none. One that reads anything but parameters or does not translate: nothing, the
names it reads into `fixed` (an override of them refused: the array code holds the translated start).
"""
function _startIRs(g::Gen, sv::SC.SIMVAR, fixed::Set{String})::Union{Nothing, Vector{IR}}
  local exps::Union{Nothing, Vector{SC.Exp}} = sv.attributes.startExps
  exps === nothing && return nothing
  local read::Set{String} = Set{String}()
  for x in exps
    SC.foreachExp(x) do y::SC.Exp
      y isa SC.EXP_CREF && push!(read, String(y.cref.sym))
      nothing
    end
  end
  local parametersOnly::Bool = all((nm::String) -> haskey(g.vars, nm) && g.vars[nm].kind == :param, read)
  if parametersOnly
    try
      return IR[toIR(g, x, Ctx(Dict{String, Int}(), Int[])) for x in exps]
    catch err
      err isa NotSupported || rethrow(err)
    end
  end
  union!(fixed, read)
  return nothing
end

"Whether a start of `sv` (its `startExps`) reads one of `names`."
function _startReadsAny(sv::SC.SIMVAR, names::Vector{String})::Bool
  local exps::Union{Nothing, Vector{SC.Exp}} = sv.attributes.startExps
  exps === nothing && return false
  local hit::Bool = false
  for x in exps
    SC.foreachExp(x) do y::SC.Exp
      (y isa SC.EXP_CREF && String(y.cref.sym) in names) && (hit = true)
      nothing
    end
  end
  return hit
end

"The subscripts of the elements of an array of dimensions `dims`, row-major (one empty tuple for a scalar)."
_elementSubscripts(dims::Vector{Int})::Vector{Tuple{Vararg{Int}}} = isempty(dims) ? Tuple{Vararg{Int}}[()] :
  vec(Tuple{Vararg{Int}}[reverse(t) for t in Iterators.product(reverse([1:d for d in dims])...)])

"""
    domainSystem(sc) -> DomainSystem

The domain level of the simulation code `sc` of a model with its arrays kept (`SimCode.lowerArraySystem`): its
variables laid out in their order (states, algebraic and discrete variables each in their buffer, parameters by their
values), its equations, when-equations, asserts and initial equations as classes over domains. NotSupported for what
the array code does not take.
"""
function domainSystem(sc::SC.SIM_CODE)::DomainSystem
  local functionOutputs::Dict{String, Int} = Dict{String, Int}(f.name => length(f.outputs) for f in sc.functions)
  local g::Gen = Gen(Dict{String, VarInfo}(), 0, 0, 0, Dict{String, Any}(), Dict{String, Symbol}(), Set{String}(),
                EqClass[], EqClass[], AlgClass[], Tuple{String, String, Bool}[], WhenClass[], AssertClass[], 0, 0, 0, 0,
                -1000, false, Dict{Int, Int}(), AlgClass[], String[], AssertClass[], functionOutputs, Dict{String, String}(),
                Set{String}(), Tuple{String, String}[])
  local stateNamesOrdered::Vector{String} = String[]; local algNamesOrdered::Vector{String} = String[]
  local discreteNamesOrdered::Vector{String} = String[]
  local u0::Vector{Float64} = Float64[]; local d0::Vector{Float64} = Float64[]; local a0::Vector{Float64} = Float64[]
  local fixedMask::Vector{Bool} = Bool[]
  local fixedNonStates::Vector{Tuple{String, Vector{Any}, Vector{Any}}} = Tuple{String, Vector{Any}, Vector{Any}}[]
  #= the variables whose starts read a parameter (their buffer, offset), translated once all variables are known =#
  local startOwners::Vector{Tuple{Symbol, Int, SC.SIMVAR}} = Tuple{Symbol, Int, SC.SIMVAR}[]
  #= the states and discrete variables whose starts do not evaluate at the translation (they read a parameter computed
     at the start, `Gen.startComputed`, or a reduction the evaluation does not take): a placeholder, their start rules
     compute them at each start =#
  local placeholders::Set{String} = Set{String}()
  for (nm::String, (_, sv::SC.SIMVAR)) in sc.stringToSimVarHT
    if sv.varKind isa SC.STATE_DERIVATIVE
      g.derOf[nm] = sv.varKind.varName
      continue
    end
    local category::Symbol = SC.irCategory(sv)
    local dims::Vector{Int} = sv.dims
    local n::Int = isempty(dims) ? 1 : prod(dims)
    local elementNames::Vector{String} = isempty(dims) ? String[nm] : get(sc.referenceIndex.elementNames, nm, String[])
    length(elementNames) == n || ns("the elements of $(nm)")
    local vtype::Symbol = SC.valueType(sv.elementType)
    local perPart::Vector{Int} = Int[length(dims)]
    local startOf::Function = () -> Float64[Float64(x) for x in _attributeValues(sv.attributes.start, n, 0.0, "start", elementNames[1])]
    local fixedOf::Function = (default::Bool) -> Bool[x for x in _attributeValues(sv.attributes.fixed, n, default, "fixed attribute", elementNames[1])]
    local startOrRule::Function = () -> try
      startOf()
    catch err
      (err isa NotSupported && sv.attributes.startExps !== nothing) || rethrow(err)
      push!(placeholders, nm)
      zeros(n)
    end
    if category == :known
      #= one the frontend evaluated for the structure (a folded if-equation's condition): the code holds its value =#
      (sv.irKind isa SC.IR.PARAMETER && sv.irKind.structural) && push!(g.structural, nm)
      #= fixed = false: an unknown of the initialization (its initial equations), from its start =#
      local free::Vector{Bool} = Bool[!x for x in fixedOf(true)]
      (any(free) && !all(free)) && ns("parameter $(nm) with fixed = false on some elements")
      local value::Any = nothing
      if all(free)
        SC.bindingOf(sv.varKind) === nothing || ns("parameter $(nm) with fixed = false and a binding")
        push!(g.freeParams, nm)
        local starts::Vector{Float64} = startOf()
        value = isempty(dims) ? starts[1] :
          length(dims) == 1 ? starts : permutedims(reshape(starts, reverse(dims)...), length(dims):-1:1)
      elseif vtype != :other
        local constant::Union{Nothing, SC.Exp} = sv.varKind isa SC.PARAMETER ? sv.varKind.value : nothing
        if constant === nothing
          #= no value the translation evaluates, but a binding (an impure call: Borefields' TemperatureResponseMatrix):
             computed from it at each start (`parameterBindings`), a placeholder of its shape until then =#
          SC.bindingOf(sv.varKind) === nothing && ns("the value of parameter $(elementNames[1])")
          local T::Type = vtype == :int ? Int : vtype == :bool ? Bool : Float64
          value = isempty(dims) ? zero(T) : zeros(T, dims...)
          push!(g.startComputed, nm)
        else
          local vals::Vector{Any} = _constantValues(constant)
          value = isempty(dims) ? vals[1] :
            length(dims) == 1 ? [x for x in vals] : permutedims(reshape([x for x in vals], reverse(dims)...), length(dims):-1:1)
        end
      end
      g.vars[nm] = VarInfo(:param, dims, perPart, 0, vtype, SimVariable(sv, elementNames, value))
      continue
    end
    #= a variable's binding is an equation of the partition already (Passes.addBindingEquations!) =#
    if category == :state
      vtype == :real || ns("the state $(nm), not Real")
      g.vars[nm] = VarInfo(:state, dims, perPart, g.nStates, :real, SimVariable(sv, elementNames, nothing))
      sv.attributes.startExps === nothing || push!(startOwners, (:u, g.nStates, sv))
      g.nStates += n
      append!(u0, startOrRule())
      append!(fixedMask, fixedOf(false))
      append!(stateNamesOrdered, elementNames)
    elseif category == :discrete || (category == :algebraic && vtype in (:int, :bool))
      vtype == :other && ns("the discrete variable $(nm) of type $(sv.elementType)")
      #= an Integer or Boolean an equation computes (MultiBody's colors): discrete, in the discrete buffer =#
      g.vars[nm] = VarInfo(:discrete, dims, perPart, g.nDiscrete, vtype, SimVariable(sv, elementNames, nothing))
      sv.attributes.startExps === nothing || push!(startOwners, (:d, g.nDiscrete, sv))
      g.nDiscrete += n
      append!(d0, startOrRule())
      append!(discreteNamesOrdered, elementNames)
    elseif category in (:algebraic, :derivative, :dummyState)
      vtype == :real || ns("the algebraic variable $(nm), not Real")
      local fixed::Vector{Bool} = fixedOf(false)
      category == :dummyState && any(fixed) && ns("a fixed start on the dummy state $(nm)")
      local algStarts::Vector{Float64} = startOf()
      any(fixed) && push!(fixedNonStates, (nm, Any[x for x in fixed], Any[x for x in algStarts]))
      g.vars[nm] = VarInfo(:alg, dims, perPart, g.nAlg, :real, SimVariable(sv, elementNames, nothing))
      g.nAlg += n
      append!(a0, algStarts)
      append!(algNamesOrdered, elementNames)
    else
      ns("the variable $(nm) of kind $(SC.irKindName(sv))")
    end
  end
  if g.nStates == 0
    #= a model without states gets one with der = 0 (as omc), so that it can be integrated =#
    local dummy::SC.SIMVAR = SC.SIMVAR("\$dummy", nothing, SC.STATE(), SC.NO_ATTRIBUTES, SC.TYPE_REAL())
    g.vars["\$dummy"] = VarInfo(:state, Int[], [0], 0, :real, SimVariable(dummy, String["\$dummy"], nothing))
    g.nStates = 1
    push!(u0, 0.0); push!(fixedMask, true); push!(stateNamesOrdered, "\$dummy")
    push!(g.classes, EqClass(Ref("\$dummy", IR[], true), Lit(0.0), 0, Matrix{Int}(undef, 0, 1), 0, "der(\$dummy) = 0"))
  end
  for eq in sc.arrayEquations
    _addEquation!(g, eq, String[], NO_SLOTS)
  end
  #= the starts that read a parameter, the array code computes them again from overridden ones =#
  local startRules::Vector{Tuple{Symbol, Int, IR}} = Tuple{Symbol, Int, IR}[]
  local startFixed::Set{String} = Set{String}()
  for (buffer, offset, sv) in startOwners
    local irs::Union{Nothing, Vector{IR}} = _startIRs(g, sv, startFixed)
    (irs === nothing && sv.name in placeholders) && ns("the start of $(sv.name) does not evaluate")
    irs === nothing && continue
    #= a fixed start of a state that reads a parameter the initialization computes (fixed = false): the initial
       equation x = start, the state free there (its translated start took the parameter's start value) =#
    local readsFree::Bool = buffer === :u && _startReadsAny(sv, g.freeParams)
    local subs::Vector{Tuple{Vararg{Int}}} = _elementSubscripts(sv.dims)
    for (k, x) in enumerate(irs)
      if readsFree && fixedMask[offset + k]
        fixedMask[offset + k] = false
        push!(g.initClasses, EqClass(Ref(sv.name, IR[Lit(i) for i in subs[k]], false), x, 0, NO_SLOTS, 0,
                                     "$(sv.name) = start"))
      elseif !(x isa Lit) || sv.name in placeholders
        push!(startRules, (buffer, offset + k, x))
      end
    end
  end
  for (nm, fixed, starts) in fixedNonStates
    #= fixed = true on a variable that is no state: the initial equation v = start (a start that reads a parameter as
       its expression: an override reaches it) =#
    local info::VarInfo = g.vars[nm]
    local startIRs::Union{Nothing, Vector{IR}} = _startIRs(g, info.var.simVar, startFixed)
    local names::Vector{String} = info.var.names
    local subs::Vector{Tuple{Vararg{Int}}} = isempty(info.dims) ? Tuple{Vararg{Int}}[()] :
      vec(Tuple{Vararg{Int}}[reverse(t) for t in Iterators.product(reverse([1:d for d in info.dims])...)])
    for (k, t) in enumerate(subs)
      fixed[k] || continue
      push!(g.initClasses, EqClass(Ref(nm, IR[Lit(i) for i in t], false), startIRs === nothing ? Lit(starts[k]) : startIRs[k], 0,
                                   NO_SLOTS, 0, "$(names[k]) = start"))
    end
  end
  _addInitialEquations!(g, sc.arrayInitialEquations)
  #= arrays it keeps: of its own, or (rolled components too) of a model whose index was reduced in array form; the
     components of a model of index 1 are rolled for nothing it keeps (its loops the scalar path's) =#
  local keepsArrays::Bool = sc.indexReduced || any(((_, (_, v)),) -> SC.isOwnArray(v), sc.stringToSimVarHT)
  domainPasses!(g)
  return DomainSystem(g, u0, d0, a0, fixedMask, stateNamesOrdered, algNamesOrdered, discreteNamesOrdered, keepsArrays,
                      startRules, sort!(collect(startFixed)), !isempty(placeholders))
end
