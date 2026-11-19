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
the domain level (`domainSystem`), from which CodeGeneration's ArrayPreserveDE (_emitModule) generates the module.

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
when-equation of the passes' structural kinds, sample() outside a when-condition, whole arrays in scalar equations,
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
    #= an external object: the parameters its constructor reads (the object is made from them, `_objectRead`) =#
    if info !== nothing && info.kind == :param && info.vtype == :other
      local root::Union{Nothing, Tuple{String, SC.CALL}} = _objectRoot(g, String(x.cref.sym))
      root === nothing || foreach((d::String) -> d in deps || push!(deps, d), _bindingDeps(g, g.vars[root[1]].var))
    end
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
  #= an array whose elements read other elements of it is ordered by the others (`_selfReadsResolved`) =#
  foreach((d::String) -> d == nm || _orderParameters!(g, d, state, order), _bindingDeps(g, g.vars[nm].var))
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
    local elements::Vector{IR} = IR[toIR(g, x, ctx) for x in _rowMajorElements(b)]
    nm in deps || return ParameterBinding(nm, deps, :elements, elements)
    #= elements that read other elements of the parameter (a rolled component's parameter bound to another
       instance's): each as its binding over the other parameters =#
    return ParameterBinding(nm, filter(!=(nm), deps), :elements, _selfReadsResolved(nm, info.dims, elements))
  catch err
    err isa NotSupported || rethrow(err)
    return ParameterBinding(nm, deps, :none, IR[])
  end
end

"""
    _selfReadsResolved(nm, dims, elements) -> Vector{IR}

The elements (row-major, of the dimensions `dims`) of the array binding of the parameter `nm` with each read of an
element of `nm` itself as that element's binding: MSL MultiBody's `bodyOfConstraint(m = bodyOfJoint.m)`, the two
bodies rolled into one array, is `m[2] = m[1]`. NotSupported for elements that read each other in a cycle, or one
read by a subscript that is no literal.
"""
function _selfReadsResolved(nm::String, dims::Vector{Int}, elements::Vector{IR})::Vector{IR}
  local out::Vector{IR} = copy(elements)
  local state::Vector{Int} = zeros(Int, length(out))
  for k::Int in eachindex(out)
    _resolveSelfReads!(out, state, nm, dims, k)
  end
  return out
end

"The element `k` of `out` with its reads of `nm` resolved (`_selfReadsResolved`; `state`: 1 being resolved, 2 done)."
function _resolveSelfReads!(out::Vector{IR}, state::Vector{Int}, nm::String, dims::Vector{Int}, k::Int)::IR
  state[k] == 2 && return out[k]
  state[k] == 1 && ns("cyclic parameter bindings at $(nm)")
  state[k] = 1
  out[k] = _withSelfReads(out[k], out, state, nm, dims)
  state[k] = 2
  return out[k]
end

"`x` with each read of an element of the parameter `nm` as that element's resolved binding (`_resolveSelfReads!`)."
function _withSelfReads(x::IR, out::Vector{IR}, state::Vector{Int}, nm::String, dims::Vector{Int})::IR
  if x isa Ref
    x.name == nm || return Ref(x.name, IR[_withSelfReads(s, out, state, nm, dims) for s in x.subs], x.der)
    (length(x.subs) == length(dims) && all(s::IR -> s isa Lit && s.v isa Integer, x.subs)) ||
      ns("the parameter $(nm) read in its own binding by a subscript that is no literal")
    local index::Int = 0
    for (d::Int, s::IR) in enumerate(x.subs)
      index = index * dims[d] + (Int((s::Lit).v) - 1)
    end
    return _resolveSelfReads!(out, state, nm, dims, index + 1)
  elseif x isa Op
    return Op(x.op, IR[_withSelfReads(y, out, state, nm, dims) for y in x.args])
  elseif x isa Union{Pre, Rel, EvFn}
    ns("the parameter $(nm) read in its own binding under pre() or a relation")
  end
  return x
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
    #= an equal relation in the same scope: the same relation, its slot (OpenModelica's relations and the MTK path's
       are unique by expression). A discrete cluster's inlined siblings repeat them: MSL Clutch's `locked` and `mode`
       read `startBackward`'s `sa < -tau0_max`; a copy that a short-circuit skipped kept an old value, the locked
       combination was never consistent (MSL Rotational's Friction stopped at the clutch's sticking, 0.109). =#
    local key::RelationKey = (SC.RELATION(e.exp1, e.op, e.exp2, 0), ctx.iters, ctx.elem)
    local known::Union{Nothing, IR} = ctx.noEvent ? nothing : get(ctx.relations, key, nothing)
    known === nothing || return known
    local op::Symbol = _relationOp(e.op)
    local l::IR = toIR(g, e.exp1, ctx)
    local r::IR = toIR(g, e.exp2, ctx)
    local continuous::Bool = _continuous(g, l) || _continuous(g, r)
    #= an initial algorithm's (a lifted cluster's initial arm): MSL RobotR3's bearing friction at w = -5e-32 from the
       solve, `initial() and w < 0` made it start Backward (omc and the MTK path, `_tieLiteral`: Stuck) =#
    (ctx.atStart && continuous && op in (:<, :<=, :>, :>=)) && return Op(:startRelation, IR[Lit(op), l, r])
    (ctx.noEvent || !continuous) && return Op(op, IR[l, r])
    op in (:(==), :!=) && ns("equality relation on continuous variables")
    ctx.nrel[] += 1
    return ctx.relations[key] = _withAbsProbe(g, Rel(op, l, r, ctx.nrel[]), op in (:<, :<=) ? l : r, ctx)
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
  elseif e isa SC.REDUCTION
    return _reductionIRx(g, e, ctx)
  end
  ns("expression $(nameof(typeof(e))): $(_str(e))")
end

"""
A reduction `sum(body for i in range)` (`product`, `min`, `max`) over iterators of known ranges: the body written out
for each iterator value (the iterator replaced by it), the terms folded (MSL BooleanNetwork1's
`sum(if {u[1], u[2]}[i] then 1 else 0 for i in 1:2)`). NotSupported for an array constructor (`:array`), an empty
range, and more than `Passes.MAX_ELEMENT_TERMS` terms.
"""
function _reductionIRx(g::Gen, e::SC.REDUCTION, ctx::Ctx)::IR
  local fold::Symbol = e.reducer === :sum ? :+ : e.reducer === :product ? :* :
    e.reducer in (:min, :max) ? e.reducer : ns("the reduction $(e.reducer): $(_str(e))")
  local terms::Vector{SC.Exp} = _reductionTerms!(SC.Exp[], g, e.body, e.iterators, ctx)
  isempty(terms) && ns("a reduction over an empty range: $(_str(e))")
  return foldl((a::IR, b::IR) -> Op(fold, IR[a, b]), IR[toIR(g, t, ctx) for t in terms])
end

"""
The terms of a reduction (onto `out`): `body` for each value of the first of `iterators`, that iterator replaced by
the value in the body and in the later iterators' ranges, for the later iterators in turn.
"""
function _reductionTerms!(out::Vector{SC.Exp}, g::Gen, body::SC.Exp, iterators::Vector{SC.Iterator},
                          ctx::Ctx)::Vector{SC.Exp}
  isempty(iterators) && return push!(out, body)
  local it::SC.Iterator = iterators[1]
  for v::SC.Exp in _iteratorValues(g, it.range, ctx)
    local rest::Vector{SC.Iterator} =
      SC.Iterator[SC.Iterator(r.name, _withIterator(r.range, it.name, v)) for r in @view iterators[2:end]]
    _reductionTerms!(out, g, _withIterator(body, it.name, v), rest, ctx)
    length(out) <= Passes.MAX_ELEMENT_TERMS || ns("a reduction of more than $(Passes.MAX_ELEMENT_TERMS) terms")
  end
  return out
end

"""
The values of a reduction's iterator range: an Integer range of structural bounds (`1:n`, `a:s:b`) its Integers, an
array literal its elements.
"""
function _iteratorValues(g::Gen, range::SC.Exp, ctx::Ctx)::Vector{SC.Exp}
  range isa SC.ARRAY_EXP && return range.elements
  range isa SC.RANGE || ns("an iterator range $(_str(range))")
  local start::Int = _structuralInt(g, range.start, ctx)
  local step::Int = range.step === nothing ? 1 : _structuralInt(g, range.step, ctx)
  local stop::Int = _structuralInt(g, range.stop, ctx)
  return SC.Exp[SC.ICONST(k) for k in start:step:stop]
end

"`e` with the iterator `name` replaced by `value` (not inside a nested reduction of an iterator of that name)."
_withIterator(e::SC.Exp, name::Symbol, value::SC.Exp)::SC.Exp =
  first(SC.traverseExpTopDown(e, _iteratorVisitor, (name, value)))

function _iteratorVisitor(e::SC.Exp, arg::Tuple{Symbol, SC.Exp})::Tuple{SC.Exp, Bool, Tuple{Symbol, SC.Exp}}
  local (name::Symbol, value::SC.Exp) = arg
  (e isa SC.EXP_CREF && isempty(e.cref.subs) && e.cref.sym === name) && return (value, false, arg)
  (e isa SC.REDUCTION && any(it::SC.Iterator -> it.name === name, e.iterators)) && return (e, false, arg)
  return (e, true, arg)
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
    if b isa SC.CALL
      local value::Union{Nothing, String} = _stringCallValue(g, b, seen)
      return value === nothing ? nothing : (nm, SC.SCONST(value))
    end
    (b isa SC.EXP_CREF && isempty(b.cref.subs)) || return nothing
    nm = String(b.cref.sym)
  end
  return nothing
end

"""
The String a call of a Modelica.Utilities function of the runtime library gives for arguments that are literals or
String parameters with a value (`_stringRoot`): `Modelica.Utilities.Strings.substring("Dy01", 1, 1)`, MSL Machines'
transformer data `C1`; `substring(group, 1, 1)` of a parameter `group`. Called at the translation; nothing for
another call, other arguments or another result. `seen`: the parameters whose value is being looked up.
"""
function _stringCallValue(g::Gen, call::SC.CALL, seen::Set{String})::Union{Nothing, String}
  local generation::Module = _OMBACKEND.CodeGeneration
  local runtime::Union{Nothing, Symbol} =
    get(generation.AlgorithmicCodeGeneration.MODELICA_UTILITIES_TO_RUNTIME_C, String(call.name), nothing)
  runtime === nothing && return nothing
  local arguments::Vector{Any} = Any[]
  for a::SC.Exp in call.args
    if a isa Union{SC.SCONST, SC.ICONST, SC.BCONST}
      push!(arguments, a.value)
    elseif a isa SC.EXP_CREF && isempty(a.cref.subs) && !(String(a.cref.sym) in seen)
      local root::Union{Nothing, Tuple{String, SC.SCONST}} = _stringRoot(g, String(a.cref.sym))
      root === nothing && return nothing
      push!(arguments, root[2].value)
    else
      return nothing
    end
  end
  local value::Any = getfield(generation.OMRuntimeExternalC, runtime)(arguments...)
  return value isa AbstractString ? String(value) : nothing
end

"""
The external object that `name` is, through its references to others (`tableID = table.tableID`), and its constructor's
call; nothing where it is none.
"""
function _objectRoot(g::Gen, name::String)::Union{Nothing, Tuple{String, SC.CALL}}
  local seen::Set{String} = Set{String}()
  local nm::String = name
  while !(nm in seen)
    push!(seen, nm)
    local info::Union{Nothing, VarInfo} = get(g.vars, nm, nothing)
    (info !== nothing && info.kind == :param && info.vtype == :other) || return nothing
    local ty::SC.SType = info.var.simVar.elementType
    (ty isa SC.TYPE_COMPLEX && !ty.isRecord && isempty(info.dims)) || return nothing
    local b::Union{Nothing, SC.Exp} = SC.bindingOf(info.var.simVar.varKind)
    b isa SC.CALL && return (nm, b)
    (b isa SC.EXP_CREF && isempty(b.cref.subs)) || return nothing
    nm = String(b.cref.sym)
  end
  return nothing
end

"""
A read of the external object `name` (a table: an argument of the functions that read it), `Op(:object)` of its slot
in `Gen.objects` and its name; nothing where `name` is none. The object is made once per build by its constructor's
call (as on the scalarized path: an MSL table keeps the state of the simulation reading it), of the translated values
of the parameters the call reads: the code depends on them (`Gen.structural`, an override refused).
"""
function _objectRead(g::Gen, name::String)::Union{Nothing, IR}
  local root::Union{Nothing, Tuple{String, SC.CALL}} = _objectRoot(g, name)
  root === nothing && return nothing
  local (nm::String, call::SC.CALL) = root
  local k::Union{Nothing, Int} = findfirst(o::Tuple{String, IR, String} -> o[1] == nm, g.objects)
  if k === nothing
    _isFunctionCall(g, call) || ns("the constructor of the external object $(nm): $(_str(call))")
    local value::IR = _functionCallIRx(g, call, Ctx(Dict{String, Int}(), Int[]))
    _parametersOnly(g, value) || ns("the constructor of the external object $(nm) reads a variable: $(_str(call))")
    for d in _bindingDeps(g, g.vars[nm].var)
      push!(g.structural, d)
      _paramValue!(g, d)
    end
    push!(g.objects, (nm, value, replace(String(call.name), r"_constructor$" => "_destructor")))
    k = length(g.objects)
  end
  return Op(:object, IR[Lit(k), Lit(nm)])
end

"""
A read of an element of the String parameter array `name` whose elements have one literal (a String parameter of
rolled components, each with the same value: the `substanceName` of MSL Fluid's trace substance sensors, read as
`sensors[\$c].substanceName` by an initial algorithm): that literal's read (`_stringRead`), whatever the element. The
binding one literal for all elements (through references to others, `_stringRoot`) or an array literal of equal ones;
nothing for another parameter or binding.
"""
function _stringElementRead(g::Gen, name::String)::Union{Nothing, IR}
  local info::Union{Nothing, VarInfo} = get(g.vars, name, nothing)
  (info !== nothing && info.kind == :param && info.vtype == :other && !isempty(info.dims)) || return nothing
  local b::Union{Nothing, SC.Exp} = SC.bindingOf(info.var.simVar.varKind)
  if b isa SC.ARRAY_EXP && !isempty(b.elements) && b.elements[1] isa SC.SCONST
    local value::String = (b.elements[1]::SC.SCONST).value
    all((x::SC.Exp) -> x isa SC.SCONST && x.value == value, b.elements) || return nothing
    return _stringRead(g, name, value)
  end
  local root::Union{Nothing, Tuple{String, SC.SCONST}} = _stringRoot(g, name)
  return root === nothing ? nothing : _stringRead(g, root[1], root[2].value)
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
  else
    local text::Union{Nothing, IR} = _stringElementRead(g, own)
    text === nothing || return text
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
  local base::SC.Exp = e.exp
  #= an element of an element (`{{1.0, 2.0}, {3.0, 4.0}}[\$i1][\$i2]`, a matrix modifier over a component array): the
     element at both subscripts =#
  if base isa SC.ASUB && !any((s::SC.Exp) -> s isa SC.WHOLE_DIMENSION || s isa SC.RANGE, base.subs)
    return _subscriptedIRx(g, SC.ASUB(base.exp, SC.Exp[base.subs; e.subs]), ctx)
  end
  #= an array literal at a literal first subscript: that element, at the subscripts left (`{X[1], X[2]}[1][2]`, the
     rows of a matrix variable written out: MSL Fluid's `mediums.state.X`, "partial subscripts on ...") =#
  if base isa SC.ARRAY_EXP && e.subs[1] isa SC.ICONST && 1 <= e.subs[1].value <= length(base.elements)
    local picked::SC.Exp = base.elements[e.subs[1].value]
    return length(e.subs) == 1 ? toIR(g, picked, ctx) : _subscriptedIRx(g, SC.ASUB(picked, e.subs[2:end]), ctx)
  end
  if base isa SC.EXP_CREF && isempty(base.cref.subs)
    local text::Union{Nothing, IR} = _stringElementRead(g, String(base.cref.sym))
    text === nothing || return text
  end
  local subs::Vector{IR} = IR[toIR(g, s, ctx) for s in e.subs]
  base isa SC.EXP_CREF && isempty(base.cref.subs) && return _elementRefIR(g, String(base.cref.sym), subs)
  #= a reference by its leading subscripts (`X[1]`, a row) at the others: the element at all of them =#
  base isa SC.EXP_CREF && return _elementRefIR(g, String(base.cref.sym), IR[IR[Lit(k) for k in base.cref.subs]; subs])
  base isa SC.DER && return _elementRefIR(g, String(base.cref.sym), subs; der = true)
  #= an element of pre() of an array (a function's array argument, `f(pre(state))`): pre() of the element =#
  if base isa SC.PRE && isempty(base.cref.subs)
    local element::IR = _elementRefIR(g, String(base.cref.sym), subs)
    element isa Ref || ns("an element of $(_str(base))")
    return Pre(element)
  end
  #= an element of the negation of a Boolean array (`not fire_p`, element by element: MSL PowerConverters' space
     vector PWM): the negation of the element =#
  (base isa SC.LUNARY && base.op == SC.OP_NOT) && return Op(:!, IR[_subscriptedIRx(g, SC.ASUB(base.exp, e.subs), ctx)])
  if base isa SC.ARRAY_EXP
    local literal::Union{Nothing, Array} = _literalArray(base)
    literal === nothing || return Op(:index, IR[Lit(literal); subs])
    length(subs) == 1 && return Op(:index, IR[Op(:vect, IR[toIR(g, x, ctx) for x in base.elements]), subs[1]])
    #= an array of rows by several subscripts (a built-in's elements, `cat(2, ...)[\$i1, \$i2]`): its value indexed =#
    return Op(:index, IR[_valueIRx(g, base, ctx); subs])
  end
  #= an element of an array output of a function of several outputs (`ToSpacePhasor(i)[1][k]`, a tuple's element): the
     output at its position in the function's result, indexed =#
  if base isa SC.TSUB && base.ty isa SC.TYPE_ARRAY && _isFunctionCall(g, base.exp)
    return Op(:index, IR[Op(:index, IR[_functionCallIRx(g, base.exp::SC.CALL, ctx), Lit(base.index)]); subs])
  end
  #= of a function's array result, a record's array field, a concatenation of vectors (`cat(1, ...)[\$i1]`): the value
     indexed =#
  (_isFunctionCall(g, base) || base isa SC.RSUB ||
   (base isa SC.CALL && base.kind == SC.CALL_BUILTIN && base.name === :cat)) || ns("an element of $(_str(base))")
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
  if e isa SC.EXP_CREF && isempty(e.cref.subs)
    local whole::Union{Nothing, IR} = _wholeArrayIRx(g, String(e.cref.sym))
    whole === nothing || return whole
    #= a parameter array as a whole (a table's `table` given to its constructor, MSL ComparisonPullInStroke's 530 x 4):
       its value =#
    local param::Union{Nothing, VarInfo} = get(g.vars, String(e.cref.sym), nothing)
    (param !== nothing && param.kind == :param && param.vtype != :other && !isempty(param.dims)) &&
      return Ref(String(e.cref.sym), IR[], false)
  end
  #= pre() of an array variable as a whole (`random(pre(state))`, MSL's random number generators): the array of
     pre() of its elements =#
  if e isa SC.PRE && isempty(e.cref.subs)
    local wholePre::Union{Nothing, IR} = _wholeArrayIRx(g, String(e.cref.sym))
    if wholePre !== nothing
      local elementsOf::Vector{IR} = (wholePre::Op).args
      return Op(:arrayval, IR[elementsOf[1]; IR[Pre(x::Ref) for x in @view elementsOf[2:end]]])
    end
  end
  #= pre() of an array expression of variables (a slice, `pre(u_buffer[1:2])`: MSL Clocked's FIR buffers; the slice's
     type is its element type): the array of pre() of the elements of its value =#
  if e isa SC.CALL && e.kind == SC.CALL_BUILTIN && e.name === :pre && length(e.args) == 1 &&
     (_expType(e.args[1]) isa SC.TYPE_ARRAY ||
      (e.args[1] isa SC.ASUB && any((s::SC.Exp) -> s isa Union{SC.RANGE, SC.WHOLE_DIMENSION}, e.args[1].subs)))
    local inner::IR = _valueIRx(g, e.args[1], ctx)
    (inner isa Op && inner.op === :arrayval && all((x::IR) -> x isa Ref, @view inner.args[2:end])) ||
      ns("pre of $(_str(e.args[1]))")
    return Op(:arrayval, IR[inner.args[1]; IR[Pre(x::Ref) for x in @view inner.args[2:end]]])
  end
  #= cat(1, v1, v2, ...) of vectors (MSL Media's mass fractions, `cat(1, Xi, {1 - sum(Xi)})`): the vector of their
     elements =#
  if e isa SC.CALL && e.kind == SC.CALL_BUILTIN && e.name === :cat && length(e.args) >= 2 && e.args[1] isa SC.ICONST &&
     e.args[1].value == 1
    local parts::Vector{IR} = IR[]
    for a::SC.Exp in @view e.args[2:end]
      #= a parameter vector as a whole (`cat(1, {roughnesses[1]}, roughnesses[:], ...)`, MSL Fluid's pipes): its
         elements =#
      local whole::SC.Exp = (a isa SC.ASUB && all((x::SC.Exp) -> x isa SC.WHOLE_DIMENSION, a.subs)) ? a.exp : a
      local vector::Union{Nothing, VarInfo} =
        (whole isa SC.EXP_CREF && isempty(whole.cref.subs)) ? get(g.vars, String(whole.cref.sym), nothing) : nothing
      if vector !== nothing && vector.kind == :param && length(vector.dims) == 1
        append!(parts, IR[Ref(String(whole.cref.sym), IR[Lit(i)], false) for i in 1:vector.dims[1]])
        continue
      end
      local part::IR = _valueIRx(g, a, ctx)
      if part isa Op && part.op === :arrayval && length((part.args[1]::Lit).v) == 1
        append!(parts, @view part.args[2:end])
      elseif part isa Lit && part.v isa AbstractVector
        append!(parts, IR[Lit(x) for x in part.v])
      else
        ns("cat of $(_str(a))")
      end
    end
    return Op(:arrayval, IR[Lit((length(parts),)); parts])
  end
  #= an array by all of each dimension (`v[:]`): the array =#
  (e isa SC.ASUB && all((x::SC.Exp) -> x isa SC.WHOLE_DIMENSION, e.subs)) && return _valueIRx(g, e.exp, ctx)
  #= a vector's slice by a range of structural bounds (`{Xi_outflow[2, 1]}[1:Medium.nXi]`, the mass fractions a medium
     function of MSL Fluid's pipes takes): the vector of the elements it picks =#
  if e isa SC.ASUB && length(e.subs) == 1 && e.subs[1] isa SC.RANGE
    local range::SC.RANGE = e.subs[1]
    local step::Int = range.step === nothing ? 1 : _structuralInt(g, range.step, ctx)
    step == 0 && ns("a slice of step 0: $(_str(e))")
    local picks::StepRange{Int, Int} = _structuralInt(g, range.start, ctx):step:_structuralInt(g, range.stop, ctx)
    local source::IR = _valueIRx(g, e.exp, ctx)
    local sourceElements::Vector{IR} =
      (source isa Op && source.op === :arrayval && length((source.args[1]::Lit).v) == 1) ? source.args[2:end] :
      (source isa Lit && source.v isa AbstractVector) ? IR[Lit(x) for x in source.v] : ns("a slice of $(_str(e.exp))")
    all(k::Int -> 1 <= k <= length(sourceElements), picks) || ns("a slice outside its vector: $(_str(e))")
    return Op(:arrayval, IR[Lit((length(picks),)); sourceElements[picks]])
  end
  #= an array's sum, difference, negation or scaling (`f(v - w)`): the array of its elements =#
  if e isa Union{SC.BINARY, SC.UNARY} && _expType(e) isa SC.TYPE_ARRAY
    local opDims::Vector{Int} = _knownDims(_expType(e), "an array argument of a size not known: $(_str(e))")
    prod(opDims) <= Passes.MAX_ELEMENT_TERMS || ns("an array argument of $(prod(opDims)) elements")
    local opElements::Vector{IR} = IR[]
    for t in Iterators.product((1:d for d in opDims)...)   #= column-major: the first index fastest =#
      push!(opElements, _elementIRx(g, e, collect(Int, t), ctx))
    end
    return Op(:arrayval, IR[Lit(Tuple(opDims)); opElements])
  end
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

"The operators of arrays taken element by element (`_elementIRx`): of two arrays."
const ELEMENTWISE_OPERATORS = Dict{SC.OpKind, Symbol}(
  SC.OP_ADD => :+, SC.OP_SUB => :-, SC.OP_ADD_EW => :+, SC.OP_SUB_EW => :-, SC.OP_MUL_EW => :*, SC.OP_DIV_EW => :/)
"Of a scalar and an array (`_elementIRx`)."
const SCALAR_ARRAY_OPERATORS = Dict{SC.OpKind, Symbol}(
  SC.OP_ADD_SCALAR_ARRAY => :+, SC.OP_SUB_SCALAR_ARRAY => :-, SC.OP_MUL_SCALAR_ARRAY => :*,
  SC.OP_DIV_SCALAR_ARRAY => :/)
"Of an array and a scalar (`_elementIRx`)."
const ARRAY_SCALAR_OPERATORS = Dict{SC.OpKind, Symbol}(
  SC.OP_ADD_ARRAY_SCALAR => :+, SC.OP_SUB_ARRAY_SCALAR => :-, SC.OP_MUL_ARRAY_SCALAR => :*,
  SC.OP_DIV_ARRAY_SCALAR => :/)

"""
The Integer value of `e`, an expression of literals and parameters (a slice's bound, `Medium.nXi`): the code depends
on the parameters it reads (structural, `_evalStatic`). NotSupported for another expression.
"""
function _structuralInt(g::Gen, e::SC.Exp, ctx::Ctx)::Int
  local x::IR = _folded(toIR(g, e, ctx))
  (_parametersOnly(g, x) && !_readsSlot(x)) || ns("no structural value: $(_str(e))")
  local value = _evalStatic(g, x, Matrix{Int}(undef, 0, 0), 0)
  value isa Integer || ns("no Integer value: $(_str(e))")
  return Int(value)
end

"""
The element `idx` of an array-valued expression (of a function's argument, `f(v - w)`): of a sum, difference or
negation of arrays the operation on the elements, of an array and a scalar the element with the scalar; of another (a
variable, a call, a literal) the expression subscripted (`_subscriptedIRx`).
"""
function _elementIRx(g::Gen, e::SC.Exp, idx::Vector{Int}, ctx::Ctx)::IR
  (e isa SC.UNARY && e.op == SC.OP_UMINUS) && return Op(:neg, IR[_elementIRx(g, e.exp, idx, ctx)])
  if e isa SC.BINARY
    local array1::Bool = _expType(e.exp1) isa SC.TYPE_ARRAY
    local array2::Bool = _expType(e.exp2) isa SC.TYPE_ARRAY
    if array1 && array2 && haskey(ELEMENTWISE_OPERATORS, e.op)
      return Op(ELEMENTWISE_OPERATORS[e.op], IR[_elementIRx(g, e.exp1, idx, ctx), _elementIRx(g, e.exp2, idx, ctx)])
    elseif !array1 && array2 && (haskey(SCALAR_ARRAY_OPERATORS, e.op) || e.op == SC.OP_MUL)
      return Op(get(SCALAR_ARRAY_OPERATORS, e.op, :*), IR[toIR(g, e.exp1, ctx), _elementIRx(g, e.exp2, idx, ctx)])
    elseif array1 && !array2 && (haskey(ARRAY_SCALAR_OPERATORS, e.op) || e.op in (SC.OP_MUL, SC.OP_DIV))
      return Op(get(ARRAY_SCALAR_OPERATORS, e.op, e.op == SC.OP_DIV ? :/ : :*),
                IR[_elementIRx(g, e.exp1, idx, ctx), toIR(g, e.exp2, ctx)])
    end
  end
  return toIR(g, SC.ASUB(e, SC.Exp[SC.ICONST(i) for i in idx]), ctx)
end

"""
An array variable of the equations as a whole (a function's argument, `f(x)`): the array of its elements (column-major,
`Op(:arrayval)`); nothing for a scalar, a parameter (its value is an array already) or another name.
"""
function _wholeArrayIRx(g::Gen, name::String)::Union{Nothing, IR}
  local info::Union{Nothing, VarInfo} = get(g.vars, get(g.derOf, name, name), nothing)
  (info === nothing || isempty(info.dims) || info.kind == :param) && return nothing
  prod(info.dims) <= Passes.MAX_ELEMENT_TERMS || ns("an array argument of $(prod(info.dims)) elements")
  local elements::Vector{IR} = IR[]
  for t in Iterators.product((1:d for d in info.dims)...)   #= column-major: the first index fastest =#
    push!(elements, _elementRefIR(g, name, IR[Lit(i) for i in t]))
  end
  return Op(:arrayval, IR[Lit(Tuple(info.dims)); elements])
end

"""
A field of a record value: of a function's record result the call's tuple at the field (the function returns a record
by its fields, `generateIOL`).
"""
function _recordElementIRx(g::Gen, e::SC.RSUB, ctx::Ctx)::IR
  if e.exp isa SC.IFEXP
    #= of an if-expression of records its branches' fields (an inlined function's `if k == 0 then Complex(1, 0)
       else Complex(0, 1)`: MSL ComplexBlocks' powerOfJ) =#
    local branches::SC.IFEXP = e.exp
    #= a condition of literals (an inlined `if size(X, 1) == nX then ... else ...`, `1 == 2`: MSL Media's
       setState_pTX): the branch taken alone =#
    local condition::IR = _folded(toIR(g, branches.cond, ctx))
    if condition isa Lit
      local taken::SC.Exp = condition.v == true ? branches.thenExp : branches.elseExp
      return _recordElementIRx(g, SC.RSUB(taken, e.index, e.fieldName, e.ty), ctx)
    end
    return Op(:if, IR[condition,
                      _recordElementIRx(g, SC.RSUB(branches.thenExp, e.index, e.fieldName, e.ty), ctx),
                      _recordElementIRx(g, SC.RSUB(branches.elseExp, e.index, e.fieldName, e.ty), ctx)])
  end
  _isRecordValue(e.exp) && return _recordFieldIRx(g, e.exp, e.index, e.fieldName, ctx)
  #= a field of an element of a vector of record literals at a literal subscript (MSL Media's `cp_T(data[1], T)` inlined
     after index reduction, `data` evaluated: `{DataRecord(...), DataRecord(...)}[1].Tlimit`): the record's field =#
  if e.exp isa SC.ASUB && length(e.exp.subs) == 1 && e.exp.subs[1] isa SC.ICONST && e.exp.exp isa SC.ARRAY_EXP
    local records::Vector{SC.Exp} = e.exp.exp.elements
    local at::Int = (e.exp.subs[1]::SC.ICONST).value
    (1 <= at <= length(records) && records[at] isa SC.RECORD) &&
      return _recordElementIRx(g, SC.RSUB(records[at], e.index, e.fieldName, e.ty), ctx)
  end
  _isFunctionCall(g, e.exp) || ns("a field of $(_str(e.exp))")
  local rty::SC.SType = (e.exp::SC.CALL).ty
  (rty isa SC.TYPE_COMPLEX && rty.isRecord) || ns("a field of $(_str(e.exp)), not a record")
  any((t::SC.SType) -> t isa SC.TYPE_COMPLEX && t.isRecord, rty.fieldTypes) && ns("a record result with record fields: $(_str(e.exp))")
  local call::IR = _functionCallIRx(g, e.exp::SC.CALL, ctx)
  return g.functions[String((e.exp::SC.CALL).name)] == 1 ? call : Op(:index, IR[call, Lit(e.index)])
end

"Whether `x` is a record value without record fields: a reference to a record variable or element, a record literal."
function _isRecordValue(x::SC.Exp)::Bool
  x isa Union{SC.EXP_CREF, SC.RECORD} || return false
  local ty::SC.SType = x.ty
  (ty isa SC.TYPE_COMPLEX && ty.isRecord) || return false
  return !any((t::SC.SType) -> t isa SC.TYPE_COMPLEX, ty.fieldTypes)
end

"""
The field number `k`, `field`, of the record value `x` (`_isRecordValue`): a literal's expression, of a reference
the field's variable at the reference's subscripts (the variables of a record are its fields': `u.re` of `u[1]` is
`u.re[1]`).
"""
function _recordFieldIRx(g::Gen, x::SC.Exp, k::Int, field::String, ctx::Ctx)::IR
  #= a field that is an array its value (`ThermodynamicState(p, T, {Xi[1]}).X`, subscripted by the caller) =#
  x isa SC.RECORD && return _valueIRx(g, x.exps[k], ctx)
  local c::SC.SimCref = (x::SC.EXP_CREF).cref
  return _elementRefIR(g, string(c.sym, ".", field), IR[Lit(i) for i in c.subs])
end

"""
The fields' vectors of the reference `a` to a vector of records without record fields (`Complex k[3]`: `k.re`, `k.im`,
each the array of its elements), for a function's argument; `nothing` where `a` is none.
"""
function _recordVectorFields(g::Gen, a::SC.EXP_CREF)::Union{Nothing, Vector{IR}}
  local ty::SC.SType = a.ty
  (isempty(a.cref.subs) && ty isa SC.TYPE_ARRAY && ty.elementType isa SC.TYPE_COMPLEX && ty.elementType.isRecord) ||
    return nothing
  any((t::SC.SType) -> t isa SC.TYPE_COMPLEX, ty.elementType.fieldTypes) && return nothing
  local out::Vector{IR} = IR[]
  for field::String in ty.elementType.fieldNames
    local name::String = string(a.cref.sym, ".", field)
    local info::Union{Nothing, VarInfo} = get(g.vars, name, nothing)
    (info === nothing || isempty(info.dims)) && return nothing
    prod(info.dims) <= Passes.MAX_ELEMENT_TERMS || ns("an array argument of $(prod(info.dims)) elements")
    local elements::Vector{IR} = IR[]
    for t in Iterators.product((1:d for d in info.dims)...)   #= column-major: the first index fastest =#
      push!(elements, _elementRefIR(g, name, IR[Lit(i) for i in t]))
    end
    push!(out, Op(:arrayval, IR[Lit(Tuple(info.dims)); elements]))
  end
  return out
end

"""
An argument's values (onto `out`): a record literal's fields each (a generated function takes a record by its fields,
`R_T`, `R_w`: `generateIOL`), a call returning an array of records its outputs, the fields' arrays, spread
(`_recordArrayCallIRx`), an array's or a scalar's value (`_valueIRx`).
"""
function _argumentsIRx!(out::Vector{IR}, g::Gen, a::SC.Exp, ctx::Ctx)::Nothing
  if _isFunctionCall(g, a) && _recordArrayFieldCount(a.ty) > 0
    push!(out, _recordArrayCallIRx(g, a::SC.CALL, ctx))
  elseif a isa SC.RECORD
    for x in a.exps
      (x isa SC.RECORD || (x isa SC.EXP_CREF && x.ty isa SC.TYPE_COMPLEX)) && ns("a record argument with record fields: $(_str(a))")
      push!(out, _valueIRx(g, x, ctx))
    end
  elseif a isa SC.ARRAY_EXP && !isempty(a.elements) && all(_isRecordValue, a.elements)
    #= a vector of records (`Complex u[:]`, MSL QuasiStatic's quasiRMS(u)): a generated function takes it by its
       fields' vectors, one argument each (`flattenRecordInput`) =#
    local fields::Vector{String} = (a.elements[1].ty::SC.TYPE_COMPLEX).fieldNames
    for (k::Int, field::String) in enumerate(fields)
      local elements::Vector{IR} = IR[_recordFieldIRx(g, x, k, field, ctx) for x in a.elements]
      push!(out, Op(:arrayval, IR[Lit((length(elements),)); elements]))
    end
  elseif a isa SC.EXP_CREF && _recordVectorFields(g, a) !== nothing
    #= a reference to a vector of records (the variables are its fields' vectors): by those, one argument each =#
    append!(out, _recordVectorFields(g, a)::Vector{IR})
  elseif a isa SC.EXP_CREF && a.ty isa SC.TYPE_COMPLEX
    #= an external object (a table) as it is; a record variable has no value here =#
    local object::Union{Nothing, IR} = a.ty.isRecord ? nothing : _objectRead(g, String(a.cref.sym))
    object === nothing && ns("a record argument: $(_str(a))")
    push!(out, object)
  elseif a isa SC.PARTEVALFUNCTION
    push!(out, _partialApplicationIRx(g, a, ctx))
  else
    push!(out, _valueIRx(g, a, ctx))
  end
  return nothing
end

"""
    _partialApplicationIRx(g, e, ctx) -> IR

A function of the model with inputs bound, an argument of a function that takes a function (`function
f_nonlinear(p = p, h = h, X = X)`: MSL Media moist air's `T_phX` through `solveOneNonlinearEquation`, in the pipes
and volumes of MSL Fluid): `Op(:partial, [Lit(name), Lit(free), values...])`, `free` telling of each input of the
function, in its order, whether it is left free, `values` the bound ones' values in that order. NotSupported for a
record input (flattened in the function) and a function of more outputs.
"""
function _partialApplicationIRx(g::Gen, e::SC.PARTEVALFUNCTION, ctx::Ctx)::IR
  local name::String = String(e.name)
  get(g.functions, name, 0) == 1 || ns("a partial application of $(name)")
  local free::Vector{Bool} = Bool[]
  local values::Vector{IR} = IR[]
  for (input::Symbol, ty::SC.SType) in e.inputs
    ty isa SC.TYPE_COMPLEX && ns("a partial application of $(name), whose input $(input) is a record")
    local at::Union{Nothing, Int} = findfirst(==(input), e.argNames)
    push!(free, at === nothing)
    at === nothing || push!(values, _valueIRx(g, e.args[at], ctx))
  end
  length(values) == length(e.args) || ns("a partial application of $(name): bound arguments that are no inputs")
  return Op(:partial, IR[Lit(Symbol(name)), Lit(free), values...])
end

"""
A call of a Modelica function of the model (`_isFunctionCall`) as its value (`_allOutputsCallIRx`); of a function of
more outputs typed as its first, that output; of a record result the tuple of its fields (and the outputs after it).
"""
function _functionCallIRx(g::Gen, e::SC.CALL, ctx::Ctx)::IR
  local call::IR = _allOutputsCallIRx(g, e, ctx)
  #= the outputs counted as returned, a record by its fields (generateIOL): a record result is the tuple's first =#
  g.functions[String(e.name)] > 1 && !(e.ty isa SC.TYPE_TUPLE || (e.ty isa SC.TYPE_COMPLEX && e.ty.isRecord)) &&
    (call = Op(:index, IR[call, Lit(1)]))
  return call
end

"A call of a Modelica function of the model (`_isFunctionCall`) with its arguments' values (`_argumentsIRx!`): all its outputs, as returned."
function _allOutputsCallIRx(g::Gen, e::SC.CALL, ctx::Ctx)::IR
  local callArgs::Vector{IR} = IR[]
  for a in e.args
    _argumentsIRx!(callArgs, g, a, ctx)
  end
  return Op(:fcall, IR[Lit(Symbol(String(e.name))), callArgs...])
end

"""
The number of fields of the records of an array of records without record fields (`Complex[n]`); 0 for another type.
"""
function _recordArrayFieldCount(ty::SC.SType)::Int
  (ty isa SC.TYPE_ARRAY && ty.elementType isa SC.TYPE_COMPLEX && ty.elementType.isRecord) || return 0
  any((t::SC.SType) -> t isa SC.TYPE_COMPLEX, ty.elementType.fieldTypes) && return 0
  return length(ty.elementType.fieldNames)
end

"""
    _recordArrayCallIRx(g, e, ctx) -> IR

A call returning an array of records as an argument of a function: the generated function returns the records' fields'
arrays in place (generateIOL), one argument each of the callee (`flattenRecordInput`), spread (`Op(:splat)`): the call
runs once. NotSupported where the function has outputs after the records.
"""
function _recordArrayCallIRx(g::Gen, e::SC.CALL, ctx::Ctx)::IR
  local nFields::Int = _recordArrayFieldCount(e.ty)
  g.functions[String(e.name)] == nFields ||
    ns("a call of $(e.name) returning an array of records and further outputs as an argument")
  return Op(:splat, IR[_allOutputsCallIRx(g, e, ctx)])
end

"""
    _callIRx(g, e, ctx) -> IR

A call of the simulation code in the domain level's form (`toIR`): `sample` (in a when-condition) and the event
operators, the built-in functions (event-generating ones as event functions), a Modelica function of the model (a
scalar result; an array or record one only under a subscript or a field), `delay()` as `Op(:delay)`; NotSupported
for another.
"""
function _callIRx(g::Gen, e::SC.CALL, ctx::Ctx)::IR
  local fname::String = String(e.name)
  local args::Vector{SC.Exp} = e.args
  local builtin::Bool = e.kind == SC.CALL_BUILTIN
  if builtin && fname in ("delay", "OpenModelica.Internal.delay2", "OpenModelica.Internal.delay3")
    #= delay(value, time[, max]): the value `time` ago, from a history of the run (`_numberDelays!` gives the call
       its histories; -1 until then) =#
    length(args) >= 2 || ns("delay() of $(length(args)) arguments")
    local delayed::IR = Op(:delay, IR[Lit(-1), toIR(g, args[1], ctx), toIR(g, args[2], ctx)])
    #= the history holds numbers: an Integer or enumeration value again as one (a table's subscript, MSL Digital's
       `LogicMap[delay(x, t)]`); a Boolean is not supported =#
    local valueType::SC.SType = _expType(args[1])
    valueType isa SC.TYPE_BOOL && ns("delay() of a Boolean")
    return valueType isa Union{SC.TYPE_INTEGER, SC.TYPE_ENUM} ? Op(:Int, IR[delayed]) : delayed
  end
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
    #= homotopy(actual, simplified) (MLS 3.7.4.4): the blend in the initialization's λ (`homotopyStart!`) =#
    length(args) == 2 || ns("homotopy with $(length(args)) arguments")
    g.usesHomotopy = true
    return Op(:homotopy, IR[toIR(g, args[1], ctx), toIR(g, args[2], ctx)])
  elseif builtin && fname == "Integer" && length(args) == 1
    #= Integer(e) of an enumeration value: its position, the number the code holds the value as, an Integer where
       it is a table's subscript (MSL Digital's `Integer(pre(x))`, `Integer(delay(x, t))` of the Logic values) =#
    return Op(:Int, IR[toIR(g, args[1], ctx)])
  elseif fname == "noEvent"
    return toIR(g, args[1], _with(ctx; noEvent = true, atStart = false))
  elseif fname == "smooth"
    return toIR(g, args[2], ctx)
  elseif fname == "initial"
    g.usesInitial = true
    return Op(:initial, IR[])
  elseif fname == "terminal"
    #= true at the terminal event only, after the solve (`Op(:terminal)`, the array code's `_terminalEvent!`) =#
    return Op(:terminal, IR[])
  elseif fname == "edge" || fname == "change"
    local changed::IR = toIR(g, args[1], ctx)
    #= of an expression of parameters (a lifted cluster's `change(tLH[1] > 0.0)`, MSL Digital's delays): it holds its
       value through the simulation, so it neither changes nor rises =#
    (!(changed isa Ref) && _ofParameters(g, changed)) && return Lit(false)
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
initialization's solve): the initial value of a discrete variable, scalar or an array's element (`d = expr`, CDL
Discrete's `y = y_start`, assigned after the solve; `pre(d) = expr` too, MSL Timer's `pre(entryTime) = 0`, the
caller passes `d`), or such a parameter with fixed = false given the value of an
expression of parameters (`p = expr`): MSL GlobalSeed's `id_impure = initializeImpureRandom(seed)`, an impure call the solve's Newton made at every
evaluation, a new value each time (ImpureGenerator stalled). Not of free parameters the solve determines (a double
exponential's `eta = exp(-T/tau1) - exp(-T/tau2)`); those an earlier such assignment gives are known.
"""
function _initialAssignment(g::Gen, lhs::IR, rhs::IR)::Bool
  (lhs isa Ref && !lhs.der) || return false
  #= a scalar, or an element of an array by literal subscripts (an initial array equation, an element at a time) =#
  local dims::Int = length(g.vars[lhs.name].dims)
  (length(lhs.subs) == dims && all(x::IR -> x isa Lit, lhs.subs)) || return false
  local known::Function = (name::String) -> !(name in g.freeParams) ||
    any(a::AlgClass -> any(st::Stmt -> st isa SAssign && st.target.name == name, a.body), g.initAlgs)
  local reads::Vector{Ref} = accesses!(Ref[], rhs)
  g.vars[lhs.name].kind == :discrete &&
    return all(r::Ref -> r.name != lhs.name && (g.vars[r.name].kind != :param || known(r.name)), reads)
  lhs.name in g.freeParams || return false
  return all(r::Ref -> r.name != lhs.name && g.vars[r.name].kind == :param && known(r.name), reads)
end

"""
Whether the initial equation `lhs = rhs` of a for-equation's body gives pre() of a discrete variable's element its
value (`pre(d[i]) = expr`): each subscript a literal or an iterator, `expr` reading neither `d` nor a parameter with
fixed = false (one an initial algorithm gives is known only after it: the scalar equations' order,
`_addInitialEquations!`).
"""
function _initialPreStart(g::Gen, lhs::IR, rhs::IR)::Bool
  lhs isa Pre || return false
  local target::Ref = lhs.ref
  local info::VarInfo = g.vars[target.name]
  (info.kind == :discrete && !target.der && length(target.subs) == length(info.dims)) || return false
  all(x::IR -> x isa Union{Lit, Slot}, target.subs) || return false
  return all(r::Ref -> r.name != target.name && !(r.name in g.freeParams), accesses!(Ref[], rhs))
end

"Whether `x` is a discrete variable or element, or pre() of one: what an initial assignment gives its value."
_isDiscreteTarget(g::Gen, x::IR)::Bool =
  (x isa Ref && !x.der && g.vars[x.name].kind == :discrete) || (x isa Pre && g.vars[x.ref.name].kind == :discrete)

"Whether the expression reads parameters, literals and iterators only (no variable, no time, no pre() or event)."
function _ofParameters(g::Gen, x::IR)::Bool
  x isa Union{Lit, Slot} && return true
  x isa Ref && return !x.der && g.vars[x.name].kind == :param && all(s::IR -> _ofParameters(g, s), x.subs)
  x isa Op && return all(a::IR -> _ofParameters(g, a), x.args)
  x isa Rel && return _ofParameters(g, x.lhs) && _ofParameters(g, x.rhs)
  return false
end

"Whether a when-equation assigns the variable `name` (an element of it)."
_whenAssigned(g::Gen, name::String)::Bool =
  any(w::WhenClass -> any(b::Tuple -> b[1] == :assign && b[2] isa Ref && b[2].name == name, w.body), g.whens)

"""
The left side `d` of an initial equation `d = value` read through the equations that compute `d`: a scalar discrete
variable no when-equation assigns that is, by equations between such variables (`active = localActive`, a
connection's `active = timer.u`), equal to one an equation gives as pre() of a variable
(`localActive = pre(newActive)`) is that pre(); equal to a variable the when-equations assign, that variable (its
initial value), or the pre() they all assign it (the passes lift `localActive = pre(newActive)` into a when-equation
of every event: `_whenPre`). MSL StateGraph's steps:
`initial equation active = true` gives pre(newActive). As an assignment of `active` it took `active` out of the
unknowns, and `active = localActive` was an equation too many ("52 equations for 50 unknowns"). Another left side,
and one equal to several such, as it is.
"""
function _throughDefinitions(g::Gen, lhs::IR)::IR
  local discrete::Function = (x::IR) -> x isa Ref && !x.der && isempty(x.subs) && g.vars[x.name].kind == :discrete
  (discrete(lhs) && !_whenAssigned(g, (lhs::Ref).name)) || return lhs
  local equal::Set{String} = Set{String}([lhs.name])
  local pres::Vector{Pre} = Pre[]
  local assigned::Vector{Ref} = Ref[]
  local grew::Bool = true
  while grew
    grew = false
    for c::EqClass in g.classes
      c.nslots == 0 || continue
      for (a::IR, b::IR) in ((c.lhs, c.rhs), (c.rhs, c.lhs))
        (discrete(a) && (a::Ref).name in equal) || continue
        if discrete(b) && _whenAssigned(g, (b::Ref).name)
          local lifted::Union{Nothing, Pre} = _whenPre(g, b.name)
          if lifted === nothing
            any(q::Ref -> q.name == b.name, assigned) || push!(assigned, b)
          else
            any(q::Pre -> q.ref.name == lifted.ref.name, pres) || push!(pres, lifted)
          end
        elseif discrete(b) && !((b::Ref).name in equal)
          push!(equal, b.name)
          grew = true
        elseif b isa Pre && isempty(b.ref.subs) && !any(q::Pre -> q.ref.name == b.ref.name, pres)
          push!(pres, b)
        end
      end
    end
  end
  (length(pres) == 1 && isempty(assigned)) && return pres[1]
  (isempty(pres) && length(assigned) == 1) && return assigned[1]
  return lhs
end

"""
pre() of the scalar variable every when-equation that assigns the scalar `name` assigns it (`name := pre(y)`, a
lifted `name = pre(y)`); nothing where an assignment differs.
"""
function _whenPre(g::Gen, name::String)::Union{Nothing, Pre}
  local found::Union{Nothing, Pre} = nothing
  for w::WhenClass in g.whens, (kind::Symbol, target::Ref, value::IR) in w.body
    (kind === :assign && target.name == name) || continue
    (w.nslots == 0 && isempty(target.subs) && value isa Pre && isempty(value.ref.subs)) || return nothing
    (found === nothing || found.ref.name == value.ref.name) || return nothing
    found = value
  end
  return found
end

"The variable an initial equation's left side gives a value: `d` of `d = ...` and of `pre(d) = ...`; nothing otherwise."
_initialTarget(lhs::IR)::Union{Nothing, String} =
  lhs isa Pre ? lhs.ref.name : (lhs isa Ref && !lhs.der) ? lhs.name : nothing

"The variables `x` reads pre() of, into `names`."
function _preNames!(names::Set{String}, x::IR)::Set{String}
  if x isa Pre
    push!(names, x.ref.name)
  elseif x isa Op
    foreach(a::IR -> _preNames!(names, a), x.args)
  elseif x isa Rel
    _preNames!(names, x.lhs); _preNames!(names, x.rhs)
  elseif x isa EvFn
    _preNames!(names, x.arg)
  elseif x isa Ref
    foreach(a::IR -> _preNames!(names, a), x.subs)
  end
  return names
end

"""
The initial equations: the assignments among them (a discrete variable's initial value, a parameter with fixed = false
given its value from parameters: `_initialAssignment`) as initial algorithms, in the order their parameters become
known (GlobalSeed's `seed`, then `id_impure = initializeImpureRandom(seed)`, whatever the equations' order); the others
as initial equations (`_addEquation!`). An equation on a variable an equation computes gives pre() of what that reads
(`_throughDefinitions`); of two that give pre() of one variable, `pre(a) = pre(b)` gives pre(b) where nothing else
does; one that reads pre() of a variable another gives comes after it.
"""
function _addInitialEquations!(g::Gen, equations::Vector{SC.Equation})::Nothing
  local pending::Vector{Tuple{SC.EQUATION, IR, IR, Int}} = Tuple{SC.EQUATION, IR, IR, Int}[]
  for eq in equations
    if eq isa SC.EQUATION && !(eq.lhs isa SC.TUPLE)
      local ctx::Ctx = Ctx(Dict{String, Int}(), Int[], Base.RefValue(0), true)
      local lhs::IR = toIR(g, eq.lhs, ctx)
      local rhs::IR = toIR(g, eq.rhs, ctx)
      ctx.nsam[] == 0 || ns("sample() in an equation")
      #= `value = d`, the discrete variable on the right (MSL Mean's `y0 = y_last`): its initial value, `d = value` =#
      (_isDiscreteTarget(g, rhs) && !_isDiscreteTarget(g, lhs)) && ((lhs, rhs) = (rhs, lhs))
      push!(pending, (eq, _throughDefinitions(g, lhs), rhs, ctx.nrel[]))
    else
      _addEquation!(g, eq, String[], NO_SLOTS; initial = true)
    end
  end
  #= two equations that give pre() of one variable: `pre(a) = pre(b)` gives pre(b) where no other does =#
  local given::Dict{String, Int} = Dict{String, Int}()
  for (_, lhs, _, _) in pending
    local name::Union{Nothing, String} = _initialTarget(lhs)
    name === nothing || (given[name] = get(given, name, 0) + 1)
  end
  for k in eachindex(pending)
    local (eq::SC.EQUATION, lhs::IR, rhs::IR, nrel::Int) = pending[k]
    (lhs isa Pre && rhs isa Pre && get(given, lhs.ref.name, 0) > 1 && get(given, rhs.ref.name, 0) == 0) || continue
    pending[k] = (eq, rhs, lhs, nrel)
    given[lhs.ref.name] -= 1
    given[rhs.ref.name] = 1
  end
  local assigned::Bool = true
  #= `ordered`: an equation that reads pre() of a variable another pending one gives waits for it; given up where
     that leaves none to take (a cycle) =#
  local ordered::Bool = true
  while assigned || ordered
    (assigned || !ordered) || (ordered = false)
    assigned = false
    for k in eachindex(pending)
      local (eq::SC.EQUATION, lhs::IR, rhs::IR, _) = pending[k]
      if ordered
        local read::Set{String} = _preNames!(Set{String}(), rhs)
        any(j::Int -> j != k && _initialTarget(pending[j][2]) in read, eachindex(pending)) && continue
      end
      #= `pre(d) = value`: pre() of the discrete variable at the initialization (`SPreStart`); of a variable a
         when-equation assigns also its initial value, held until a when fires (MSL Timer's `pre(entryTime) = 0`).
         One an equation computes is not assigned: the equation gives it, from that pre() (MSL MultiSwitch's
         `pre(y) = y_default`, a hysteresis' `pre(y) = pre_y_start`) =#
      local target::IR = lhs isa Pre ? lhs.ref : lhs
      _initialAssignment(g, target, rhs) || continue
      local body::Vector{Stmt} = Stmt[]
      (lhs isa Pre && !_whenAssigned(g, target.name)) || push!(body, SAssign(target::Ref, rhs))
      lhs isa Pre && push!(body, SPreStart(target::Ref, rhs))
      push!(g.initAlgs, AlgClass(body, 0, NO_SLOTS, 0, 0, _eqText(eq)))
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

"Whether `x` reads an iterator of its for-equation."
_readsSlot(x::IR)::Bool =
  x isa Slot || (x isa Op && any(_readsSlot, x.args)) || (x isa Ref && any(_readsSlot, x.subs))

"What `_iteratorIf` finds: the if-expression and its condition."
const IteratorIf = Tuple{SC.IFEXP, IR}

"""
    _iteratorIf(g, e, iters) -> Union{Nothing, IteratorIf}

The first if-expression in `e` (top-down) whose condition is static (`_isStatic`: literals, parameters, iterators)
and reads an iterator, with that condition; nothing where there is none. A condition `toIR` does not take is not one.
"""
function _iteratorIf(g::Gen, e::SC.Exp, iters::Dict{String, Int})::Union{Nothing, IteratorIf}
  local found::Base.RefValue{Union{Nothing, IteratorIf}} = Base.RefValue{Union{Nothing, IteratorIf}}(nothing)
  local visit = (x::SC.Exp, acc::Base.RefValue{Union{Nothing, IteratorIf}}) -> begin
    acc[] === nothing || return (x, false, acc)
    x isa SC.IFEXP || return (x, true, acc)
    local condition::Union{Nothing, IR} = try
      toIR(g, x.cond, Ctx(iters, Int[], Base.RefValue(0), true))
    catch err
      err isa NotSupported || rethrow()
      nothing
    end
    (condition !== nothing && _isStatic(g, condition) && _readsSlot(condition)) || return (x, true, acc)
    acc[] = (x, condition)
    return (x, false, acc)
  end
  SC.traverseExpTopDown(e, visit, found)
  return found[]
end

"`e` with the if-expression `target` (that node) replaced by `branch`."
_withBranch(e::SC.Exp, target::SC.IFEXP, branch::SC.Exp)::SC.Exp =
  first(SC.traverseExpTopDown(e, (x::SC.Exp, acc::Nothing) -> x === target ? (branch, false, acc) : (x, true, acc),
                              nothing))

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
    #= of an initial equation whose targets are parameters with fixed = false (MSL Filter's coefficients,
       `(cr, c0, c1) = coefficients.lowPass(...)`): an initial algorithm, its relations no events =#
    local initialTuple::Bool = initial &&
      all((e::SC.Exp) -> e isa SC.WILD || _isFreeParameter(g, e) || _isEmptyArray(g, e), eq.lhs.elements)
    local tupleCtx::Ctx = initialTuple ? Ctx(iters, Int[], Base.RefValue(0), true) : Ctx(iters, Int[])
    local tupleCall::IR = toIR(g, eq.rhs, tupleCtx)
    tupleCtx.nsam[] == 0 || ns("sample() in an equation")
    local assignments::Vector{Stmt} =
      _tupleAssignments(g, eq.lhs, tupleCall, _expType(eq.rhs), tupleCtx; free = initialTuple)
    if initialTuple
      push!(g.initAlgs, AlgClass(assignments, nIter, idom, 0, 0, _eqText(eq)))
    else
      push!(g.algs, AlgClass(assignments, nIter, idom, tupleCtx.nrel[], 0, _eqText(eq)))
    end
    return
  elseif eq isa SC.EQUATION
    #= an if-expression on an iterator (MSL StateGraph's
       `outPort[i].available = if i == 1 then active else outPort[i - 1].available and not outPort[i - 1].reset`):
       each instance its branch, the class split by the condition (as an if-equation's, `_addIfEquation!`). As one
       class the branch not taken was read at every instance: "subscript 0 out of 1:1" at i = 1 =#
    local split::Union{Nothing, IteratorIf} = nIter == 0 ? nothing : _iteratorIf(g, eq.rhs, iters)
    (split === nothing && nIter > 0) && (split = _iteratorIf(g, eq.lhs, iters))
    if split !== nothing
      local taken::Vector{Int} = Int[j for j in 1:size(idom, 2) if _evalStatic(g, split[2], idom, j) == true]
      local others::Vector{Int} = setdiff(1:size(idom, 2), taken)
      for (branch::SC.Exp, columns::Vector{Int}) in ((split[1].thenExp, taken), (split[1].elseExp, others))
        _addEquation!(g, SC.EQUATION(_withBranch(eq.lhs, split[1], branch), _withBranch(eq.rhs, split[1], branch),
                                     eq.source, eq.attr), iterNames, idom[:, columns]; initial = initial)
      end
      return
    end
    #= relations of initial equations are evaluated at the start (no events) =#
    local ctx::Ctx = Ctx(iters, Int[], Base.RefValue(0), initial)
    local lhs::IR = toIR(g, eq.lhs, ctx)
    local rhs::IR = toIR(g, eq.rhs, ctx)
    #= sample() in an initial equation: no ticks there =#
    (initial && ctx.nsam[] > 0) && ns("sample() in an initial equation")
    if initial && nIter > 0 && _initialPreStart(g, lhs, rhs)
      #= `pre(d[i]) = value` in an initial for-equation: pre() of each element at the initialization, as the scalar
         one's (`_addInitialEquations!`; MSL Fluid's tank: `pre(aboveLevel[i]) = level_start >= height[i]`) =#
      local preTarget::Ref = (lhs::Pre).ref
      local preBody::Vector{Stmt} = Stmt[]
      _whenAssigned(g, preTarget.name) && push!(preBody, SAssign(preTarget, rhs))
      push!(preBody, SPreStart(preTarget, rhs))
      push!(g.initAlgs, AlgClass(preBody, nIter, idom, 0, 0, _eqText(eq)))
      return
    end
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

"""
The triggers of a when-branch's condition (`SimCode.lowerArraySystem`: the vector of several, none the empty one; one
itself): an or of `change()` calls its terms (the condition of a lifted discrete cluster or algorithm with several
relations, `Passes.changeCondition`: each change holds at its own event only, so the or rises where one of them does).
"""
function _triggers(c::SC.Exp)::Vector{SC.Exp}
  c isa SC.ARRAY_EXP && return c.elements
  local terms::Vector{SC.Exp} = SC.Exp[]
  return c isa SC.LBINARY && _changeTerms!(terms, c) ? terms : SC.Exp[c]
end

"The terms of an or of `change()` calls, in order, into `terms`; false where `c` is no such or (`terms` then unused)."
function _changeTerms!(terms::Vector{SC.Exp}, c::SC.Exp)::Bool
  if c isa SC.CALL
    (c.kind == SC.CALL_BUILTIN && c.name === :change) || return false
    push!(terms, c)
    return true
  end
  return c isa SC.LBINARY && c.op == SC.OP_OR && _changeTerms!(terms, c.exp1) && _changeTerms!(terms, c.exp2)
end

"Whether `e` is a reference: a variable, `der` of a state (an element of one too), as it is; no expression of it."
_isReference(e::SC.Exp)::Bool = e isa Union{SC.EXP_CREF, SC.DER} || (e isa SC.ASUB && e.exp isa Union{SC.EXP_CREF, SC.DER})

"""
The elements (row-major) of an array variable referred to as a whole, `v` or `pre(v)` (a call or `SC.PRE`); nothing
for another expression.
"""
function _wholeArrayElements(g::Gen, e::SC.Exp)::Union{Nothing, Vector{SC.Exp}}
  if e isa SC.CALL && e.kind == SC.CALL_BUILTIN && e.name === :pre && length(e.args) == 1
    local inner::Union{Nothing, Vector{SC.Exp}} = _wholeArrayElements(g, e.args[1])
    inner === nothing && return nothing
    local preType::SC.SType = e.ty isa SC.TYPE_ARRAY ? e.ty.elementType : e.ty
    return SC.Exp[SC.CALL(e.name, SC.Exp[x], preType, e.kind) for x in inner]
  end
  #= pre() of an array variable as a whole (a clocked partition's `previous(u)`, `sample(u)` at the tick) =#
  if e isa SC.PRE && isempty(e.cref.subs) && e.ty isa SC.TYPE_ARRAY
    local preInfo::Union{Nothing, VarInfo} = get(g.vars, String(e.cref.sym), nothing)
    (preInfo === nothing || isempty(preInfo.dims)) && return nothing
    local preElementType::SC.SType = (e.ty::SC.TYPE_ARRAY).elementType
    return SC.Exp[SC.PRE(SC.SimCref(e.cref.sym, Int[t...], e.cref.ref), preElementType)
                  for t in _elementSubscripts(preInfo.dims)]
  end
  (e isa SC.EXP_CREF && isempty(e.cref.subs) && e.ty isa SC.TYPE_ARRAY) || return nothing
  local info::Union{Nothing, VarInfo} = get(g.vars, String(e.cref.sym), nothing)
  (info === nothing || isempty(info.dims)) && return nothing
  local elementType::SC.SType = (e.ty::SC.TYPE_ARRAY).elementType
  return SC.Exp[SC.EXP_CREF(SC.SimCref(e.cref.sym, Int[t...], e.cref.ref), elementType)
                for t in _elementSubscripts(info.dims)]
end

"""
The triggers of the trigger `c` of a when-branch: of a Boolean array as a whole (`when shut.pre_reset then`: a vector
of conditions, MLS 8.3.5, the when fires where an element becomes true) its elements; of change() or edge() of an
array that of each element (`change(pre(shut.reset))`, a lifted cluster's); another trigger itself. MSL's
ControlledTanks (StateGraph, Fluid): "the array shut.pre_reset as a whole in an equation of scalars".
"""
function _elementTriggers(g::Gen, c::SC.Exp)::Vector{SC.Exp}
  local elements::Union{Nothing, Vector{SC.Exp}} = _wholeArrayElements(g, c)
  elements === nothing || return elements
  (c isa SC.CALL && c.kind == SC.CALL_BUILTIN && c.name in (:change, :edge) && length(c.args) == 1) || return SC.Exp[c]
  elements = _wholeArrayElements(g, c.args[1])
  elements === nothing && return SC.Exp[c]
  return SC.Exp[SC.CALL(c.name, SC.Exp[x], c.ty, c.kind) for x in elements]
end

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
  local branch::Union{Nothing, SC.WHEN_STMTS} = eq.whenEquation
  while branch !== nothing
    local triggers::Vector{IR} = IR[]
    local constant::Bool = false
    for whole in _triggers(branch.condition), c in _elementTriggers(g, whole)
      if c isa SC.CALL && c.kind == SC.CALL_BUILTIN && c.name === :change && !_isReference(c.args[1])
        #= change(e) of a Boolean expression (a lifted discrete cluster's trigger): e or not e becomes true =#
        local x::IR = toIR(g, c.args[1], ctx)
        #= of parameters (`change(tLH[1] > 0.0)`, MSL Digital's delays): no trigger, it never changes =#
        if _ofParameters(g, x)
          constant = true
          continue
        end
        _isBooleanIR(g, x) || ns("change of $(_str(c.args[1]))")
        append!(triggers, IR[x, Op(:!, IR[x])])
      else
        push!(triggers, _topLevelInitial(toIR(g, c, ctx)))
      end
    end
    #= every trigger the change of an expression of parameters: the branch never runs (its variables keep what the
       initialization gives them) =#
    (isempty(triggers) && constant) && push!(triggers, Lit(false))
    isempty(triggers) && ns("a when-branch without a condition: $(_str(eq))")
    push!(conds, length(triggers) == 1 ? triggers[1] : Op(:vect, triggers))
    push!(starts, length(body) + 1)
    _whenStatements!(body, g, branch.body, nothing, ctx, eq)
    branch = branch.elsewhenPart
  end
  local cond::IR = length(conds) == 1 ? conds[1] : Op(:elsewhen, conds)
  push!(g.whens, WhenClass(cond, body, starts, length(iterNames), idom, ctx.nrel[], ctx.nsam[], _str(eq), eq.attr.everyEvent))
  return nothing
end

"""
    _whenStatements!(body, g, statements, guard, ctx, eq)

A when-branch's statements onto its body (`_addWhen!`), in order. `guard`: `nothing` at the branch's top level; in a
branch of an if-statement that assigns (`_whenIf!`) the temporary that holds whether that branch is taken: each
assignment then `x := if taken then value else x`, each effect under it (`_guarded`).
"""
function _whenStatements!(body::Vector{Tuple{Symbol, Ref, IR}}, g::Gen, statements::Vector{SC.Statement},
                          guard::Union{Nothing, Slot}, ctx::Ctx, eq::SC.WHEN_EQUATION)::Nothing
  local noTarget::Ref = Ref("", IR[], false)
  local entries::Vector{Tuple{Symbol, Ref, IR}} = Tuple{Symbol, Ref, IR}[]
  for st in statements
    empty!(entries)
    if st isa SC.ASSIGN && st.left isa SC.TUPLE && st.right isa SC.IFEXP
      #= a tuple of an if-expression (a subsampled clock's `(noise, seedState) := if tick then random(pre(seedState))
         else (pre(noise), pre(seedState))`: MSL Clocked's Noise in ClockedWith_AD_DA_Effects): the if-statement of
         the tuple assignments =#
      local branch::SC.Branch = SC.Branch(st.right.cond, SC.Statement[SC.ASSIGN(st.left, st.right.thenExp, st.source)])
      _whenIf!(body, g, SC.IF([branch], SC.Statement[SC.ASSIGN(st.left, st.right.elseExp, st.source)], st.source),
               guard, ctx, eq)
    elseif st isa SC.ASSIGN && st.left isa SC.TUPLE && st.right isa SC.TUPLE
      append!(entries, _tupleOfPreAssignments(g, st, ctx))
    elseif st isa SC.ASSIGN && st.left isa SC.TUPLE && _isFunctionCall(g, st.right)
      #= (r, state) = f(pre(state)): the call once, each target from its output (Buildings.Occupants' random numbers) =#
      append!(entries, _whenTupleAssignments(g, st.left, toIR(g, st.right, ctx), _expType(st.right), ctx))
    elseif st isa SC.ASSIGN && _wholeArrayElements(g, st.left) !== nothing
      append!(entries, _wholeArrayAssignments(g, st, ctx))
    elseif st isa SC.ASSIGN
      _isReference(st.left) || ns("a when-statement assigning $(_str(st.left))")
      local target::IR = toIR(g, st.left, ctx)
      (target isa Ref && g.vars[target.name].kind == :discrete) ||
        ns("when assigns a non-discrete variable: $(_str(st.left))")
      push!(entries, (:assign, target, toIR(g, st.right, ctx)))
    elseif st isa SC.REINIT
      guard === nothing || ns("reinit in an if-statement of a when-statement: $(_str(eq))")
      #= of an element of an array state: the state at the element's subscripts =#
      local stateRef::SC.Exp = SC.EXP_CREF(st.stateVar, SC.TYPE_REAL())
      local state::IR = toIR(g, isempty(st.subscripts) ? stateRef : SC.ASUB(stateRef, st.subscripts), ctx)
      (state isa Ref && g.vars[state.name].kind == :state) || ns("reinit of a non-state: $(_str(eq))")
      #= of a dynamic state set's state: the set's switch, in effect with the selection the body assigns =#
      local ref::Union{Nothing, SC.IR.ComponentRef} = st.stateVar.ref
      local switch::Bool = ref !== nothing && Passes.IndexReduction.isStateSetRef(ref)
      push!(entries, (switch ? :switch : :reinit, state, toIR(g, st.value, ctx)))
    elseif st isa SC.IF && _assignsIn(st)
      _whenIf!(body, g, st, guard, ctx, eq)
    elseif st isa Union{SC.ASSERT, SC.NORETCALL, SC.TERMINATE, SC.IF}
      push!(entries, (:call, noTarget, _whenEffect(g, st, ctx)))
    else
      ns("a when-statement $(nameof(typeof(st))): $(_str(eq))")
    end
    foreach(e::Tuple{Symbol, Ref, IR} -> push!(body, _guarded(e, guard)), entries)
  end
  return nothing
end

"""
A tuple's assignment from a tuple of pre() values, `(noise, seedState) := (pre(noise), pre(seedState))` (a subsampled
clock's hold, `_whenStatements!`): each target from its value, in order (the values read no target's value of this
instant); a whole array target element by element.
"""
function _tupleOfPreAssignments(g::Gen, st::SC.ASSIGN, ctx::Ctx)::Vector{Tuple{Symbol, Ref, IR}}
  local targets::Vector{SC.Exp} = (st.left::SC.TUPLE).elements
  local values::Vector{SC.Exp} = (st.right::SC.TUPLE).elements
  local isPre::Function = (v::SC.Exp) -> v isa SC.PRE || (v isa SC.CALL && v.kind == SC.CALL_BUILTIN && v.name === :pre)
  (length(targets) == length(values) && all(isPre, values)) || ns("a when-statement assigning $(_str(st.left))")
  local out::Vector{Tuple{Symbol, Ref, IR}} = Tuple{Symbol, Ref, IR}[]
  for (t::SC.Exp, v::SC.Exp) in zip(targets, values)
    t isa SC.WILD && continue
    local single::SC.ASSIGN = SC.ASSIGN(t, v, st.source)
    if _wholeArrayElements(g, t) !== nothing
      append!(out, _wholeArrayAssignments(g, single, ctx))
    else
      local target::IR = toIR(g, t, ctx)
      (target isa Ref && g.vars[target.name].kind == :discrete) ||
        ns("when assigns a non-discrete variable: $(_str(t))")
      push!(out, (:assign, target, toIR(g, v, ctx)))
    end
  end
  return out
end

"Whether the if-statement `st` assigns a variable or reinitializes a state in one of its branches (nested too)."
function _assignsIn(st::SC.IF)::Bool
  local assigns::Function = (x::SC.Statement) -> x isa Union{SC.ASSIGN, SC.REINIT} || (x isa SC.IF && _assignsIn(x))
  return any((b::SC.Branch) -> any(assigns, b.body), st.branches) || any(assigns, st.elseBody)
end

"""
    _whenIf!(body, g, st, guard, ctx, eq)

An if-statement that assigns, in a when-branch (MSL SignalExtrema's `if y_min < pre(y_min) then t_min = time; t_max =
pre(t_max); elseif ... end if` in its when-equation): its conditions evaluated first, at the instant (noEvent, as
`_whenEffect`'s), into temporaries, each whether its branch is taken (its condition holds, no earlier one's does, and
the enclosing `guard`); then each branch's statements under its temporary (`_whenStatements!`), the else-branch's
under none of the conditions holding.
"""
function _whenIf!(body::Vector{Tuple{Symbol, Ref, IR}}, g::Gen, st::SC.IF, guard::Union{Nothing, Slot}, ctx::Ctx,
                  eq::SC.WHEN_EQUATION)::Nothing
  local actx::Ctx = _with(ctx; noEvent = true)
  local noTarget::Ref = Ref("", IR[], false)
  local open::Union{Nothing, Slot} = guard   #= no earlier branch taken (and the guard) =#
  local taken::Vector{Slot} = Slot[]
  for b in st.branches
    local condition::IR = toIR(g, b.condition, actx)
    local slot::Slot = Slot(g.tmpSlot -= 1)
    push!(body, (:call, noTarget, Op(:setlocal, IR[slot, open === nothing ? condition : Op(:&&, IR[open, condition])])))
    push!(taken, slot)
    local rest::Slot = Slot(g.tmpSlot -= 1)
    push!(body, (:call, noTarget, Op(:setlocal, IR[rest, open === nothing ? Op(:!, IR[slot]) :
                                                         Op(:&&, IR[open, Op(:!, IR[slot])])])))
    open = rest
  end
  for (b, slot) in zip(st.branches, taken)
    _whenStatements!(body, g, b.body, slot, ctx, eq)
  end
  isempty(st.elseBody) || _whenStatements!(body, g, st.elseBody, open, ctx, eq)
  return nothing
end

"""
A when body's entry under `guard` (`_whenStatements!`): itself without one; an assignment `x := if guard then value
else x`, a statement for its effects (a call, a temporary's value) `if guard then ... end`.
"""
function _guarded(entry::Tuple{Symbol, Ref, IR}, guard::Union{Nothing, Slot})::Tuple{Symbol, Ref, IR}
  guard === nothing && return entry
  local (kind::Symbol, target::Ref, value::IR) = entry
  kind === :assign && return (kind, target, Op(:if, IR[guard, value, target]))
  kind === :call && return (kind, target, Op(:if, IR[guard, value, Op(:seq, IR[])]))
  ns("a when-statement $(kind) in an if-statement")
end

"""
A when body's assignment to an array variable as a whole, element by element: `y := u` (MSL Clocked's
AssignClockVectorized, its clocked when-clause lowered to a when-equation) `y[1] := u[1]; y[2] := u[2]`. The value an
array variable as a whole or pre() of one (`_wholeArrayElements`), or an array literal (UpSample2's `FIR1.a :=
{1.0, 1.0, 1.0}`), of as many elements; both row-major.
"""
function _wholeArrayAssignments(g::Gen, st::SC.ASSIGN, ctx::Ctx)::Vector{Tuple{Symbol, Ref, IR}}
  local targets::Vector{SC.Exp} = something(_wholeArrayElements(g, st.left))
  local values::Union{Nothing, Vector{SC.Exp}} =
    st.right isa SC.ARRAY_EXP ? _rowMajorElements(st.right) : _wholeArrayElements(g, st.right)
  values === nothing && return _arrayValueAssignments(g, targets, st, ctx)
  length(values) != length(targets) &&
    ns("a when-statement assigning the array $(_str(st.left)) the value $(_str(st.right))")
  local out::Vector{Tuple{Symbol, Ref, IR}} = Tuple{Symbol, Ref, IR}[]
  for (t::SC.Exp, v::SC.Exp) in zip(targets, values)
    local target::IR = toIR(g, t, ctx)
    (target isa Ref && g.vars[target.name].kind == :discrete) || ns("when assigns a non-discrete variable: $(_str(t))")
    push!(out, (:assign, target, toIR(g, v, ctx)))
  end
  return out
end

"""
A when body's assignment to an array variable as a whole (its elements `targets`, row-major) from an array expression
evaluated as one value (`_arrayValueIRx`: MSL Clocked's FIR buffer `u_buffer := if pre(first) then cat(1, {u}, u *
cBufStart) else cat(1, {u}, pre(u_buffer[1:2]))`): the value into a temporary (a `:call` entry), as an assignment
evaluates its right-hand side before it assigns, then each element from it.
"""
function _arrayValueAssignments(g::Gen, targets::Vector{SC.Exp}, st::SC.ASSIGN,
                                ctx::Ctx)::Vector{Tuple{Symbol, Ref, IR}}
  local what::String = "a when-statement assigning the array $(_str(st.left)) the value $(_str(st.right))"
  local dims::Vector{Int} = _knownDims(_expType(st.left), what)
  _knownDims(_expType(st.right), what) == dims || ns(what)
  local value::Slot = Slot(g.tmpSlot -= 1)
  local out::Vector{Tuple{Symbol, Ref, IR}} =
    Tuple{Symbol, Ref, IR}[(:call, Ref("", IR[], false), Op(:setlocal, IR[value, _arrayValueIRx(g, st.right, ctx)]))]
  for (t::SC.Exp, subs) in zip(targets, _elementSubscripts(dims))
    local target::IR = toIR(g, t, ctx)
    (target isa Ref && g.vars[target.name].kind == :discrete) || ns("when assigns a non-discrete variable: $(_str(t))")
    push!(out, (:assign, target, Op(:index, IR[value; IR[Lit(k) for k in subs]])))
  end
  return out
end

"""
The value of an array expression (`_arrayValueAssignments`): of an if-expression the if of its branches' values (its
condition a scalar's), of `noEvent(e)` that of `e` without events, of another `_valueIRx`.
"""
function _arrayValueIRx(g::Gen, e::SC.Exp, ctx::Ctx)::IR
  e isa SC.IFEXP &&
    return Op(:if, IR[toIR(g, e.cond, ctx), _arrayValueIRx(g, e.thenExp, ctx), _arrayValueIRx(g, e.elseExp, ctx)])
  (e isa SC.CALL && e.kind == SC.CALL_BUILTIN && e.name === :noEvent && length(e.args) == 1) &&
    return _arrayValueIRx(g, e.args[1], _with(ctx; noEvent = true))
  return _valueIRx(g, e, ctx)
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

"Whether `e` references a parameter with fixed = false (the variable, an element of it, or a row: `r[1]`)."
_isFreeParameter(g::Gen, e::SC.Exp)::Bool =
  (e isa SC.EXP_CREF && String(e.cref.sym) in g.freeParams) ||
  (e isa SC.ASUB && e.exp isa SC.EXP_CREF && String(e.exp.cref.sym) in g.freeParams)

"""
Whether the tuple target `e` is an array without elements (a dimension of size 0: a filter without conjugate
complex poles, `(r[1], a[1], b[1], ku[1]) = roots.lowPass(...)` with `a` and `b` of size [2, 0]): nothing to assign.
"""
function _isEmptyArray(g::Gen, e::SC.Exp)::Bool
  local ty::SC.SType = _expType(e)
  (ty isa SC.TYPE_ARRAY && any(==(0), ty.dims)) && return true
  local name::String = e isa SC.EXP_CREF ? String(e.cref.sym) :
    (e isa SC.ASUB && e.exp isa SC.EXP_CREF) ? String(e.exp.cref.sym) : ""
  local info::Union{Nothing, VarInfo} = get(g.vars, name, nothing)
  return info !== nothing && any(==(0), info.dims)
end

"A tuple target that is an array (`_arrayTarget`): the variable, its leading subscripts, the dimensions left."
const ArrayTarget = Tuple{String, Vector{IR}, Vector{Int}}

"""
A tuple's target that is an array: a variable as a whole (`r`), or an array by its leading subscripts only (`r[1]`,
`state[\$c]`: the array of one of the components rolled into the variable; MSL's filters' roots and random number
generators, "partial subscripts on uniformNoise1|uniformNoise2.state"). Nothing for a scalar or an element.
"""
function _arrayTarget(g::Gen, e::SC.Exp, ctx::Ctx)::Union{Nothing, ArrayTarget}
  local name::String = ""
  local leading::Vector{IR} = IR[]
  if e isa SC.EXP_CREF
    name = String(e.cref.sym)
    leading = IR[Lit(k) for k in e.cref.subs]
  elseif e isa SC.ASUB && e.exp isa SC.EXP_CREF && isempty(e.exp.cref.subs)
    name = String(e.exp.cref.sym)
    leading = IR[toIR(g, s, ctx) for s in e.subs]
  else
    return nothing
  end
  local info::Union{Nothing, VarInfo} = get(g.vars, name, nothing)
  (info === nothing || length(leading) >= length(info.dims)) && return nothing
  return (name, leading, info.dims[(length(leading) + 1):end])
end

"""
The statements of `(a, _, c) = call` (a tuple equation): the call once, into a temporary, and an assignment of each
target that is not a wildcard from its output (at its position in the result, `SimCode.outputPosition` of the call's type
`callType`: a record output before it is its fields); a whole array target element by element from that output
(`(r, state) = random(pre(state))`: Buildings.Occupants). The targets algebraic or discrete variables; `free`:
parameters with fixed = false (an initial equation's).
"""
function _tupleAssignments(g::Gen, lhs::SC.TUPLE, call::IR, callType::SC.SType, ctx::Ctx;
                           free::Bool = false)::Vector{Stmt}
  local out::Vector{Stmt} = Stmt[]
  local kinds::Tuple{Vararg{Symbol}} = free ? (:param,) : (:alg, :discrete)
  local res::Slot = Slot(g.tmpSlot -= 1)
  for (i, e) in enumerate(lhs.elements)
    (e isa SC.WILD || _isEmptyArray(g, e)) && continue
    _isReference(e) || ns("tuple target $(_str(e))")
    local rows::Union{Nothing, ArrayTarget} = _arrayTarget(g, e, ctx)
    if rows !== nothing
      local (name::String, leading::Vector{IR}, rest::Vector{Int}) = rows
      g.vars[name].kind in kinds || ns("tuple target $(_str(e))")
      local result::IR = Op(:index, IR[res, Lit(SC.outputPosition(callType, i))])
      for t in Iterators.product((1:d for d in rest)...)
        local subs::Vector{IR} = IR[Lit(k) for k in t]
        push!(out, SAssign(Ref(name, IR[leading; subs], false), Op(:index, IR[result; subs])))
      end
      continue
    end
    local target::IR = toIR(g, e, ctx)
    (target isa Ref && g.vars[target.name].kind in kinds) || ns("tuple target $(_str(e))")
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
    local rows::Union{Nothing, ArrayTarget} = _arrayTarget(g, e, ctx)
    if rows !== nothing
      local (name::String, leading::Vector{IR}, rest::Vector{Int}) = rows
      g.vars[name].kind == :discrete || ns("when assigns a non-discrete variable: $(_str(e))")
      for t in Iterators.product((1:d for d in rest)...)
        local subs::Vector{IR} = IR[Lit(k) for k in t]
        push!(out, (:assign, Ref(name, IR[leading; subs], false), Op(:index, IR[output; subs])))
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

"""
The relation `rel` with a probe of its bounded side's `abs()` argument when that side (`bounded`, the side that is to
be the smaller) is `abs(a)` of a continuous `a`: `Op(:probed, [rel, a >= 0])`, the probe's crossing an event where `a`
passes zero, where `rel` is evaluated inside the narrow window in which it holds (`abs(x) <= 1e-7` of a 50 Hz signal:
nanoseconds, which a step crossed with the crossing function positive at both ends). As the MTK path's clusters
(`_absProbes`).
"""
function _withAbsProbe(g::Gen, rel::Rel, bounded::IR, ctx::Ctx)::IR
  (bounded isa Op && bounded.op === :abs && length(bounded.args) == 1 && _continuous(g, only(bounded.args))) || return rel
  ctx.nrel[] += 1
  return Op(:probed, IR[rel, Rel(:>=, only(bounded.args), Lit(0.0), ctx.nrel[])])
end

"Whether `x` is Boolean-valued in the code: a relation, a logical operation, a Boolean variable or its pre value."
function _isBooleanIR(g::Gen, x::IR)::Bool
  x isa Rel && return true
  x isa Lit && return x.v isa Bool
  x isa Op && return x.op in (:!, :&&, :||, :initial, :terminal, :probed)
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
  local ctx::Ctx = Ctx(Dict{String, Int}(n => k for (k, n) in enumerate(iterNames)), Int[], Base.RefValue(0), true,
                       Base.RefValue(0), Dict{RelationKey, IR}(), true)
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
    local inner::Ctx = Ctx(merge(ctx.iters, Dict{String, Int}(String(st.iterator.name) => k)), ctx.elem, ctx.nrel,
                           ctx.noEvent, ctx.nsam, ctx.relations, ctx.atStart)
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
The if-equation `eq` on a variable condition with its branches in the form `_ifAssigned` and `_ifEquationCount` take
(`_branchEquations`), its nested if-equations' too.
"""
_branchForm(eq::SC.INLINE_IF_EQUATION)::SC.INLINE_IF_EQUATION =
  SC.INLINE_IF_EQUATION(eq.conditions, Vector{SC.Equation}[_branchEquations(b) for b in eq.branchesTrue],
                        _branchEquations(eq.branchElse), eq.source, eq.attr)

"""
The equations of a branch of an if-equation on a variable condition, element by element: a for-equation over a literal
range unrolled, the iterator its value in each (MSL LossyGear's `if ideal then ... else for i in 1:4 loop
interpolation_result[1, i] = ... end for`); an element of an array equality the lowering wrote `value = reference`
(`{{1, 1, 0, 0}}[1, 1] = interpolation_result[1, 1]`) as `reference = value`. Another equation as it is.
"""
function _branchEquations(body::Vector{SC.Equation})::Vector{SC.Equation}
  local out::Vector{SC.Equation} = SC.Equation[]
  for b in body
    local range::Union{Nothing, StepRange{Int, Int}} = b isa SC.FOR_EQUATION ? _literalRange(b.iterator.range) : nothing
    if range !== nothing
      local loop::SC.FOR_EQUATION = b::SC.FOR_EQUATION
      for value in range
        append!(out, _branchEquations(SC.Equation[_withIteratorValue(x, loop.iterator.name, value) for x in loop.body]))
      end
    elseif b isa SC.INLINE_IF_EQUATION
      push!(out, _branchForm(b))
    elseif b isa SC.EQUATION && !_isReference(b.lhs) && _isReference(b.rhs)
      push!(out, SC.EQUATION(b.rhs, b.lhs, b.source, b.attr))
    else
      push!(out, b)
    end
  end
  return out
end

"""
The equation `eq` of a for-equation's body with the iterator `name` its value `value` (`_branchEquations`); a nested
for-equation of the same iterator keeps its own in its body.
"""
function _withIteratorValue(eq::SC.Equation, name::Symbol, value::Int)::SC.Equation
  local isIterator::Function = (x::SC.Exp) -> x isa SC.EXP_CREF && x.cref.sym === name && isempty(x.cref.subs)
  local valued::Function = (e::SC.Exp) -> first(SC.traverseExpTopDown(e, (x::SC.Exp, acc::Nothing) ->
    isIterator(x) ? (SC.ICONST(value), false, acc) : (x, true, acc), nothing))
  local valuedAll::Function = (eqs::Vector{SC.Equation}) ->
    SC.Equation[_withIteratorValue(x, name, value) for x in eqs]
  if eq isa SC.EQUATION
    return SC.EQUATION(valued(eq.lhs), valued(eq.rhs), eq.source, eq.attr)
  elseif eq isa SC.FOR_EQUATION
    local body::Vector{SC.Equation} = eq.iterator.name === name ? eq.body : valuedAll(eq.body)
    return SC.FOR_EQUATION(SC.Iterator(eq.iterator.name, valued(eq.iterator.range)), body, eq.source, eq.attr)
  elseif eq isa SC.INLINE_IF_EQUATION
    return SC.INLINE_IF_EQUATION(SC.Exp[valued(c) for c in eq.conditions],
                                 Vector{SC.Equation}[valuedAll(b) for b in eq.branchesTrue], valuedAll(eq.branchElse),
                                 eq.source, eq.attr)
  end
  ns("if-equation on a variable condition: branch equation $(_eqText(eq)) in a for-equation")
end

"""
The variables the branches of an if-equation on a variable condition assign, each with its left side: `v = e`, or an
if-equation in a branch that assigns them likewise (MSL KinematicPTP's phases). The same variables in every branch
and the else, each once; NotSupported otherwise.
"""
function _ifAssigned(eq::SC.INLINE_IF_EQUATION)::Dict{String, SC.Exp}
  local bodies::Vector{Vector{SC.Equation}} = Vector{SC.Equation}[eq.branchesTrue; [eq.branchElse]]
  local perBranch::Vector{Dict{String, SC.Exp}} = Dict{String, SC.Exp}[_bodyAssigned(body) for body in bodies]
  for d in perBranch
    Set(keys(d)) == Set(keys(perBranch[1])) ||
      ns("if-equation on a variable condition: branches give different variables")
  end
  return perBranch[1]
end

"The variables the equations of one branch of an if-equation assign (`_ifAssigned`), each with its left side."
function _bodyAssigned(body::Vector{SC.Equation})::Dict{String, SC.Exp}
  local assigned::Dict{String, SC.Exp} = Dict{String, SC.Exp}()
  local add::Function = (key::String, lhs::SC.Exp) -> begin
    haskey(assigned, key) && ns("if-equation on a variable condition: two equations for $(key)")
    assigned[key] = lhs
  end
  for b in body
    if b isa SC.INLINE_IF_EQUATION
      foreach(kv::Pair{String, SC.Exp} -> add(kv.first, kv.second), collect(_ifAssigned(b)))
    elseif b isa SC.EQUATION && _isReference(b.lhs)
      add(string(b.lhs), b.lhs)
    else
      ns("if-equation on a variable condition: branch equation $(_eqText(b))")
    end
  end
  return assigned
end

"What the if-equation `eq` gives the variable `key` (`_ifAssigned`): `if c1 then e1 elseif ... else en`."
function _ifRhs(g::Gen, eq::SC.INLINE_IF_EQUATION, key::String, ctx::Ctx)::IR
  local rhs::IR = _bodyRhs(g, eq.branchElse, key, ctx)
  for i in length(eq.conditions):-1:1
    rhs = Op(:if, IR[toIR(g, eq.conditions[i], ctx), _bodyRhs(g, eq.branchesTrue[i], key, ctx), rhs])
  end
  return rhs
end

"What the equations of one branch give the variable `key`: its equation's right side, or a nested if-equation's."
function _bodyRhs(g::Gen, body::Vector{SC.Equation}, key::String, ctx::Ctx)::IR
  for b in body
    if b isa SC.INLINE_IF_EQUATION
      haskey(_ifAssigned(b), key) && return _ifRhs(g, b, key, ctx)
    elseif b isa SC.EQUATION && string(b.lhs) == key
      return toIR(g, b.rhs, ctx)
    end
  end
  ns("if-equation on a variable condition: no equation for $(key)")
end

"""
The number of scalar equations of each branch of the if-equation `eq` (a nested if-equation's counted); NotSupported
where the branches differ in it or hold another kind of equation.
"""
function _ifEquationCount(eq::SC.INLINE_IF_EQUATION)::Int
  local bodies::Vector{Vector{SC.Equation}} = Vector{SC.Equation}[eq.branchesTrue; [eq.branchElse]]
  local counts::Vector{Int} = Int[_bodyEquationCount(body) for body in bodies]
  all(==(counts[1]), counts) || ns("if-equation on a variable condition: branches of different numbers of equations")
  return counts[1]
end

"The scalar equations of one branch of an if-equation (`_ifEquationCount`)."
function _bodyEquationCount(body::Vector{SC.Equation})::Int
  local n::Int = 0
  for b in body
    if b isa SC.INLINE_IF_EQUATION
      n += _ifEquationCount(b)
    elseif b isa SC.EQUATION && !(b.lhs isa SC.TUPLE)
      n += 1
    else
      ns("if-equation on a variable condition: branch equation $(_eqText(b))")
    end
  end
  return n
end

"""
The residual of the `k`-th equation of the if-equation `eq`, the k-th of every branch paired:
`if c1 then l1 - r1 elseif ... else ln - rn`. One branch holds at a time, so the pairing does not matter; the class
`0 = residual` reads what every branch reads (a switch with an arc: `i = Goff * v` in one branch, `v = ...` in the
other; a power supply's `-i = I0 - Gcc * v`).
"""
function _ifResidual(g::Gen, eq::SC.INLINE_IF_EQUATION, k::Int, ctx::Ctx)::IR
  local residual::IR = _bodyResidual(g, eq.branchElse, k, ctx)
  for i in length(eq.conditions):-1:1
    residual = Op(:if, IR[toIR(g, eq.conditions[i], ctx), _bodyResidual(g, eq.branchesTrue[i], k, ctx), residual])
  end
  return residual
end

"The residual `lhs - rhs` of the `k`-th scalar equation of one branch of an if-equation (`_ifResidual`)."
function _bodyResidual(g::Gen, body::Vector{SC.Equation}, k::Int, ctx::Ctx)::IR
  local before::Int = 0
  for b in body
    if b isa SC.INLINE_IF_EQUATION
      local n::Int = _ifEquationCount(b)
      k <= before + n && return _ifResidual(g, b, k - before, ctx)
      before += n
    else
      before += 1
      before == k && return Op(:-, IR[toIR(g, (b::SC.EQUATION).lhs, ctx), toIR(g, b.rhs, ctx)])
    end
  end
  ns("if-equation on a variable condition: no equation $(k) in a branch")
end

"""
An if-equation: on conditions of parameters and iterators each instance its branch's equations; on a variable
condition each variable its branches assign (`v = e` in every branch and the else) one class
`v = if c1 then e1 elseif ... else en`; branches that assign different variables, or by equations that are no
`v = e`, a residual class per equation (`_ifResidual`).
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
  eq = _branchForm(eq)
  local assigned::Union{Nothing, Dict{String, SC.Exp}} = try
    _ifAssigned(eq)
  catch err
    err isa NotSupported || rethrow(err)
    nothing
  end
  if assigned === nothing
    #= branches that do not assign the same variables by `v = e`: a residual per equation, the k-th of every branch
       paired (`_ifResidual`) =#
    local count::Int = _ifEquationCount(eq)
    for k in 1:count
      local rctx::Ctx = Ctx(iters, Int[], Base.RefValue(0), initial)
      local residual::IR = _ifResidual(g, eq, k, rctx)
      rctx.nsam[] == 0 || ns("sample() in an equation")
      push!(initial ? g.initClasses : g.classes,
            EqClass(Lit(0.0), residual, nIter, idom, initial ? 0 : rctx.nrel[],
                    "0 = if ... ($(_str(eq.conditions[1]))), equation $(k)"))
    end
    return
  end
  for key in sort!(collect(keys(assigned)))
    local ctx::Ctx = Ctx(iters, Int[], Base.RefValue(0), initial)
    local lhs::IR = toIR(g, assigned[key], ctx)
    local rhs::IR = _ifRhs(g, eq, key, ctx)
    ctx.nsam[] == 0 || ns("sample() in an equation")
    if initial && nIter == 0 && lhs isa Ref && _initialAssignment(g, lhs, rhs) &&
       !any(r::Ref -> r.name in g.freeParams, accesses!(Ref[], rhs))
      #= an initial if-equation that gives a discrete variable its initial value in every branch (MSL SoftStarter's
         `if start then modeOfOperation = Up; else ...`): an initial assignment of the if-expression =#
      local text::String = "$(key) = if ... ($(_str(eq.conditions[1])))"
      push!(g.initAlgs, AlgClass(Stmt[SAssign(lhs, rhs)], 0, NO_SLOTS, 0, 0, text))
      continue
    end
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
                Set{String}(), Tuple{String, String}[], Tuple{String, IR, String}[], false, DelaySite[], false)
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
    #= without a start attribute: 0, an enumeration its first literal (MLS 4.9.5) =#
    local defaultStart::Float64 = sv.elementType isa SC.TYPE_ENUM ? 1.0 : 0.0
    local startOf::Function = () -> Float64[Float64(x) for x in
      _attributeValues(sv.attributes.start, n, defaultStart, "start", elementNames[1])]
    local fixedOf::Function = (default::Bool) -> Bool[x for x in _attributeValues(sv.attributes.fixed, n, default, "fixed attribute", elementNames[1])]
    local startOrRule::Function = () -> try
      startOf()
    catch err
      (err isa NotSupported && sv.attributes.startExps !== nothing) || rethrow(err)
      push!(placeholders, nm)
      zeros(n)
    end
    if category == :known
      #= fixed = false: an unknown of the initialization (its initial equations), from its start =#
      local free::Vector{Bool} = Bool[!x for x in fixedOf(true)]
      (any(free) && !all(free)) && ns("parameter $(nm) with fixed = false on some elements")
      #= one the frontend evaluated for the structure (a folded if-equation's condition): the code holds its value. Not
         one with fixed = false: the equations hold the value the frontend took, none of the array code reads it (a
         subscript an initial algorithm gives, `CVec[ind]` of MSL Fluid's trace substance sensor: `CVec[1]`), and what
         the array code itself evaluates makes it structural as any other (`_evalStatic`: refused then) =#
      (sv.irKind isa SC.IR.PARAMETER && sv.irKind.structural && !any(free)) && push!(g.structural, nm)
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
      #= a guess that does not evaluate at the translation (it reads a parameter computed at the start: MSL Fluid's
         `medium.h(start = h_start)`, h_start the enthalpy at the start pressure and temperature, a function's): a
         placeholder; its rule gives the loops their first guesses at each start (the array code's `:a` start rules) =#
      local algStarts::Vector{Float64} = try
        startOf()
      catch err
        (err isa NotSupported && sv.attributes.startExps !== nothing && !any(fixed)) || rethrow(err)
        push!(placeholders, nm)
        push!(startOwners, (:a, g.nAlg, sv))
        zeros(n)
      end
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
