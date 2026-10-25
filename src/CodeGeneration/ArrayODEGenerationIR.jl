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

Not (yet), NotSupported (the caller takes the scalarized path): when-, if- and array equations,
algorithms, sample(), a fixed start on a dummy state (the reduced initialization's), parameters
with fixed = false, records, array operands outside subscripts.
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
    local nm::String = String(OIR.hasSubscripts(x.cref) ? Passes.unsubscriptedName(x.cref) : x.cref.name)
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
    return Op(:index, IR[toIR(g, e.exp, ctx), Lit(e.index)])
  elseif e isa OIR.CALL
    return _callIRx(g, e, ctx)
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
  local name::String = String(OIR.hasSubscripts(cr) ? Passes.unsubscriptedName(cr) : cr.name)
  local der::Bool = haskey(g.derOf, name)
  der && (name = g.derOf[name])
  local info::Union{Nothing, VarInfo} = get(g.vars, name, nothing)
  info === nothing && ns("unknown variable $(name)")
  local subscripts::Vector{OIR.Subscript} = Passes.allSubscripts(cr)
  isempty(subscripts) && !isempty(info.dims) && ns("the array $(name) as a whole in an equation of scalars")
  length(subscripts) == length(info.dims) || ns("partial subscripts on $(name)")
  local subs::Vector{IR} = IR[s isa OIR.INDEX ? toIR(g, s.exp, ctx) : ns("a slice of $(name)") for s in subscripts]
  der && info.kind != :state && ns("der of non-state $(name)")
  return Ref(name, subs, der)
end

function _callIRx(g::Gen, e::OIR.CALL, ctx::Ctx)::IR
  local fname::String = String(e.name)
  local args::Vector{OIR.Expression} = e.arguments
  fname in ("delay", "OpenModelica.Internal.delay2", "OpenModelica.Internal.delay3") && ns("delay()")
  if fname == "der"
    ns("der() in a reduced model: $(_strIR(e))")
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
  #= a Modelica function of the model (_functionDefinitions), of scalar arguments =#
  local jn::String = replace(fname, "." => "_")
  if haskey(g.functions, jn)
    local callArgs::Vector{IR} = IR[]
    for a in args
      OIR.typeOf(a) isa Union{OIR.T_ARRAY, OIR.T_RECORD} && ns("function $(fname) with an array or record argument")
      push!(callArgs, toIR(g, a, ctx))
    end
    e.ty isa Union{OIR.T_ARRAY, OIR.T_RECORD} && ns("function $(fname) of an array or record result")
    local call::IR = Op(:fcall, IR[Lit(Symbol(jn)), callArgs...])
    g.functions[jn] > 1 && !(e.ty isa OIR.T_TUPLE) && (call = Op(:index, IR[call, Lit(1)]))
    return call
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
  elseif eq isa OIR.ASSERT_EQUATION
    local actx::Ctx = Ctx(iters, Int[], Base.RefValue(0), true)
    push!(initial ? g.initAsserts : g.asserts,
          AssertClass(toIR(g, eq.condition, actx), occursin("warning", _strIR(eq.level)), nIter, idom,
                      _strIR(eq.condition), _strIR(eq.message), _messageIRx(g, eq.message, actx)))
    return
  end
  ns("$(nameof(typeof(eq))) in an OMJLIR model: $(_strIR(eq))")
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
    local names::Vector{Symbol} = get(E.elements, v.name, Symbol[v.name])
    length(names) == n || ns("the elements of $(nm)")
    local elementNames::Vector{String} = String.(names)
    local vtype::Symbol = _vtypeIR(v.ty)
    local perPart::Vector{Int} = Int[length(dims)]
    if kind isa Union{OIR.PARAMETER, OIR.CONSTANT}
      any(x::Symbol -> !fixedOf(x, true), names) && ns("parameter $(nm) with fixed = false")
      local value::Any = nothing
      if vtype != :other
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
    v.binding === nothing || ns("the binding of $(nm), not a parameter")
    if kind isa OIR.STATE
      vtype == :real || ns("the state $(nm), not Real")
      g.vars[nm] = VarInfo(:state, dims, perPart, g.nStates, :real, IRVariable(v, elementNames, nothing))
      g.nStates += n
      append!(u0, startOf.(names))
      append!(fixedMask, Bool[fixedOf(x, false) for x in names])
      append!(stateNamesOrdered, elementNames)
    elseif kind isa Union{OIR.ALGEBRAIC, OIR.DUMMY_STATE, OIR.DUMMY_DER}
      vtype == :real || ns("the algebraic variable $(nm), not Real")
      local fixed::Vector{Bool} = Bool[fixedOf(x, false) for x in names]
      kind isa OIR.DUMMY_STATE && any(fixed) && ns("a fixed start on the dummy state $(nm)")
      any(fixed) && push!(fixedNonStates, nm)
      g.vars[nm] = VarInfo(:alg, dims, perPart, g.nAlg, :real, IRVariable(v, elementNames, nothing))
      g.nAlg += n
      append!(a0, startOf.(names))
      append!(algNamesOrdered, elementNames)
    elseif kind isa OIR.DISCRETE
      vtype == :other && ns("the discrete variable $(nm) of type $(v.ty)")
      g.vars[nm] = VarInfo(:discrete, dims, perPart, g.nDiscrete, vtype, IRVariable(v, elementNames, nothing))
      g.nDiscrete += n
      append!(d0, startOf.(names))
      append!(discreteNamesOrdered, elementNames)
    else
      ns("the variable $(nm) of kind $(nameof(typeof(kind)))")
    end
  end
  g.nStates == 0 && ns("no states")
  for (link, state) in g.derOf
    haskey(g.vars, state) && g.vars[state].kind == :state || ns("the state derivative $(link) of $(state), not a state")
  end
  for eq in partition.equations
    _addEquationIR!(g, eq, String[], NO_SLOTS)
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
    _addEquationIR!(g, eq, String[], NO_SLOTS; initial = true)
  end
  return _emitModule(g, modelName, functionDefs, u0, d0, a0, fixedMask, stateNamesOrdered, algNamesOrdered,
                     discreteNamesOrdered)
end
