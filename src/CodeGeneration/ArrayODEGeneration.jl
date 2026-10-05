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
Array-preserving code generation (experimental, 2026-10-05).

For a flat model from the frontend's no-scalarize mode (`scalarize = false`), builds a
DifferentialEquations.jl module whose right-hand side keeps the model's array structure: the
generated code grows with the number of equation classes, not with the array sizes.

  1. Equation classes. Every equation becomes a template over iterator slots and an index
     domain: a for-loop gives its iterators, an array equation one slot per dimension
     (element-wise), and scalar equations that differ only in literal subscripts (the
     connection equations of a component array) are re-rolled into one class whose domain
     lists their subscripts.
  2. Element level. Every instance of every class is matched to one unknown (a state
     derivative or an algebraic element) and the instances are ordered by their
     dependencies (Kahn), taking all ready instances of one class and matched slot at a
     time, so the order comes out as a few loops.
  3. Each (class, matched slot) is solved once, symbolically, for its unknown (an unknown
     that occurs once, under +, -, *, /).
  4. Code: one loop per batch over its domain columns (an affine range, else a table passed
     as data), parameters and start values evaluated by the frontend once and passed as
     data (all parameters tunable; bindings that read a changed parameter are computed
     again), the Jacobian's sparsity pattern from the element dependencies.
  5. Events: a relation on continuous variables is a zero crossing whose value stays fixed
     between events; when-equations (also in loops) assign discrete variables or reinit
     states; at an event, relations and when-conditions are evaluated again until nothing
     changes (event iteration), pre() being the values before the event.

Not (yet) handled, reported with the reason so the caller scalarizes as before: algebraic
loops, systems needing index reduction, if-equations, elsewhen, sample/initial()/edge/change,
event-generating functions (div, mod, floor, ...), initial equations, algorithms, asserts.
=#
module ArrayODEGen

import OMFrontend
import ModelingToolkit
import ..CodeGeneration: stripBeginBlocks, ModelicaAssertionError

const SII = ModelingToolkit.SymbolicIndexingInterface

const F = OMFrontend.Frontend

#= The parameter argument of an array model's problem: the parameter values (one flat
   vector, laid out by the model's PARAMETER_LAYOUT; one type for every array model, so the
   solver is compiled once), the discrete state (relation values, discrete variables and their
   pre values, when-condition values, state values before the event) and the generated module. =#
struct ArrayModelParameters
  values::Vector{Float64}
  rel::Vector{Bool}
  d::Vector{Float64}
  dpre::Vector{Float64}
  wcond::Vector{Bool}
  upre::Vector{Float64}
  dlog::Vector{Tuple{Float64, Vector{Float64}}}   #= discrete values from each event on =#
  model::Module
end

const _OMBACKEND = parentmodule(parentmodule(@__MODULE__))

#= Name lookups accept the Modelica name (c[2].T) and the canonical one
   (OMBackend.canonicalName). =#
function _nameIndex(names::Vector{String})::Dict{String, Int}
  local d = Dict{String, Int}()
  for (i, n) in enumerate(names)
    d[n] = i
    d[_OMBACKEND.canonicalName(n)] = i
  end
  return d
end

"""
    variableValues(sol, name)

The values of a state, algebraic or discrete variable element (`"c[2].T"`) at the time
points of a solution of an array model; `nothing` for an unknown name.
"""
variableValues(sol, name::String) = Base.invokelatest(_variableValues, sol, name)

#= In the latest world: the generated module is newer than the caller. =#
function _variableValues(sol, name::String)
  local p = sol.prob.p
  local m = p.model
  local key = haskey(m.STATE_INDEX, name) ? name : _OMBACKEND.canonicalName(name)
  local i = get(m.STATE_INDEX, key, 0)
  i > 0 && return [u[i] for u in sol.u]
  local j = get(m.ALGEBRAIC_INDEX, key, 0)
  j > 0 && return [m.algebraics(u, p, t)[j] for (u, t) in zip(sol.u, sol.t)]
  local k = get(m.DISCRETE_INDEX, key, 0)
  if k > 0 && !isempty(p.dlog)
    local times = first.(p.dlog)
    return [p.dlog[max(1, searchsortedlast(times, t))][2][k] for t in sol.t]
  end
  return nothing
end

#= The symbolic system of an array model's ODEFunction: names for solution indexing
   (sol[:x], sol(t; idxs = Symbol("c[2].T"))), states as variables, algebraic and discrete
   variables as observed. Names as Modelica writes them or canonical (c[2]_T). =#
struct ArraySystem
  states::Dict{Symbol, Int}
  stateSyms::Vector{Symbol}
  observed::Dict{Symbol, Tuple{Symbol, Int}}   #= (:alg or :discrete, index) =#
  model::Module
end

function ArraySystem(model::Module)
  local states = Dict{Symbol, Int}(Symbol(k) => v for (k, v) in model.STATE_INDEX)
  local observed = Dict{Symbol, Tuple{Symbol, Int}}()
  for (k, v) in model.ALGEBRAIC_INDEX
    observed[Symbol(k)] = (:alg, v)
  end
  for (k, v) in model.DISCRETE_INDEX
    observed[Symbol(k)] = (:discrete, v)
  end
  return ArraySystem(states, Symbol.(model.STATE_NAMES), observed, model)
end

#= A model variable of an ArraySystem: what sys.<name> gives (SciMLBase looks Symbols up that
   way, as on an MTK system). =#
struct ArrayVar
  name::Symbol
end

function Base.getproperty(sys::ArraySystem, s::Symbol)
  hasfield(ArraySystem, s) && return getfield(sys, s)
  return ArrayVar(s)
end

_sym(x) = x isa Symbol ? x : x isa ArrayVar ? x.name : x isa AbstractString ? Symbol(x) : nothing
_states(sys::ArraySystem) = getfield(sys, :states)
_observedIdx(sys::ArraySystem) = getfield(sys, :observed)
SII.symbolic_type(::Type{ArrayVar}) = SII.ScalarSymbolic()
SII.is_time_dependent(::ArraySystem) = true
SII.constant_structure(::ArraySystem) = true
SII.is_variable(sys::ArraySystem, x) = (s = _sym(x); s !== nothing && haskey(_states(sys), s))
SII.variable_index(sys::ArraySystem, x) = (s = _sym(x); s === nothing ? nothing : get(_states(sys), s, nothing))
SII.variable_symbols(sys::ArraySystem) = getfield(sys, :stateSyms)
SII.is_parameter(::ArraySystem, x) = false
SII.parameter_index(::ArraySystem, x) = nothing
SII.parameter_symbols(::ArraySystem) = Symbol[]
SII.is_independent_variable(::ArraySystem, x) = x === :t || x === :time
SII.independent_variable_symbols(::ArraySystem) = [:t]
SII.is_observed(sys::ArraySystem, x) = (s = _sym(x); s !== nothing && haskey(_observedIdx(sys), s))
function SII.observed(sys::ArraySystem, x)
  local (kind, i) = _observedIdx(sys)[_sym(x)]
  local model = getfield(sys, :model)
  if kind == :alg
    return (u, p, t) -> Base.invokelatest(model.algebraics, u, p, t)[i]
  end
  return (u, p, t) -> begin
    local times = first.(p.dlog)
    isempty(times) ? NaN : p.dlog[max(1, searchsortedlast(times, t))][2][i]
  end
end
SII.observed(sys::ArraySystem, x::ArrayVar) = invoke(SII.observed, Tuple{ArraySystem, Any}, sys, x)
SII.all_variable_symbols(sys::ArraySystem) = vcat(getfield(sys, :stateSyms), collect(keys(_observedIdx(sys))))
SII.all_symbols(sys::ArraySystem) = vcat(SII.all_variable_symbols(sys), :t)
SII.default_values(::ArraySystem) = Dict()

#= A relation of the model: its crossing function goes to z (when given), its value is
   updated in mode, and between events it keeps its value. =#
@inline function _rel(z, rel::Vector{Bool}, mode::Bool, k::Int, x, y, op)
  z === nothing || (z[k] = x - y)
  mode && (rel[k] = op(x, y))
  return rel[k]
end

#= Sizes of the last generated model (classes, instances, batches = loops, ...), for diagnostics. =#
const LAST_STATS = Ref{Any}(nothing)

struct NotSupported <: Exception
  msg::String
end
ns(msg::AbstractString) = throw(NotSupported(String(msg)))

#= ---------------------------------------------------------------- IR =#

abstract type IR end
struct Lit <: IR
  v::Any
end
struct Slot <: IR
  k::Int
end
struct TimeIR <: IR end
#= A variable element: state, state derivative (der), algebraic, discrete or parameter. =#
struct Ref <: IR
  name::String
  subs::Vector{IR}
  der::Bool
end
#= pre(x): the value before the current event. =#
struct Pre <: IR
  ref::Ref
end
#= A relation that generates events; id numbers it within its class template. =#
struct Rel <: IR
  op::Symbol
  lhs::IR
  rhs::IR
  id::Int
end
struct Op <: IR
  op::Symbol
  args::Vector{IR}
end

irKey(x::Lit) = x.v isa AbstractArray ? string("arr", objectid(x.v)) : repr(x.v)
irKey(x::Slot) = string("\$", x.k)
irKey(::TimeIR) = "time"
irKey(x::Ref) = string(x.der ? "der(" : "", x.name, "[", join(irKey.(x.subs), ","), "]", x.der ? ")" : "")
irKey(x::Pre) = string("pre(", irKey(x.ref), ")")
irKey(x::Rel) = string("rel", x.id, x.op, "(", irKey(x.lhs), ",", irKey(x.rhs), ")")
irKey(x::Op) = string(x.op, "(", join(irKey.(x.args), ","), ")")

#= The variable accesses of an expression, in a fixed order (not those inside subscripts,
   not pre values). =#
function accesses!(acc::Vector{Ref}, x::IR)
  if x isa Ref
    push!(acc, x)
  elseif x isa Op
    for a in x.args
      accesses!(acc, a)
    end
  elseif x isa Rel
    accesses!(acc, x.lhs)
    accesses!(acc, x.rhs)
  end
  return acc
end

_nRel(x::IR) = x isa Rel ? max(x.id, _nRel(x.lhs), _nRel(x.rhs)) :
               x isa Op ? maximum(_nRel, x.args; init = 0) : 0

#= ---------------------------------------------------------------- variables, classes =#

mutable struct VarInfo
  kind::Symbol            #= :state, :alg, :discrete or :param =#
  dims::Vector{Int}
  perPart::Vector{Int}    #= subscripts per cref part, root first =#
  offset::Int             #= first element in u, the algebraic buffer or the discrete buffer =#
  vtype::Symbol           #= :real, :int or :bool =#
  var::F.Variable
end

struct EqClass
  lhs::IR
  rhs::IR
  nslots::Int
  domain::Matrix{Int}     #= nslots x instances =#
  nrel::Int
  text::String
end

#= A when-equation over a domain: its condition and body (discrete assignments, reinits). =#
struct WhenClass
  cond::IR
  body::Vector{Tuple{Symbol, Ref, IR}}   #= (:assign or :reinit, target, value) =#
  nslots::Int
  domain::Matrix{Int}
  nrel::Int
  text::String
end

#= An assert over a domain: its condition (no events), level and texts. =#
struct AssertClass
  cond::IR
  warning::Bool
  nslots::Int
  domain::Matrix{Int}
  condition::String
  message::String
end

mutable struct Gen
  vars::Dict{String, VarInfo}
  nStates::Int
  nAlg::Int
  nDiscrete::Int
  params::Dict{String, Any}
  paramSyms::Dict{String, Symbol}
  structural::Set{String}   #= parameters the code depends on structurally (subscripts) =#
  classes::Vector{EqClass}
  whens::Vector{WhenClass}
  asserts::Vector{AssertClass}
  relBase::Int              #= relation numbering of the class being generated =#
  relCount::Int
end

struct Ctx
  iters::Dict{String, Int}  #= frontend iterator name -> slot =#
  elem::Vector{Int}         #= slots of the element-wise dimensions =#
  nrel::Base.RefValue{Int}  #= relations numbered so far in this template =#
  noEvent::Bool
end
Ctx(iters, elem) = Ctx(iters, elem, Base.RefValue(0), false)

function _partsRootFirst(cr::F.ComponentRef)::Vector{F.ComponentRef}
  local parts = F.ComponentRef[]
  while F.isvariant(cr, F.COMPONENT_REF_CREF)
    pushfirst!(parts, cr)
    cr = cr.restCref
  end
  return parts
end

_crefName(cr::F.ComponentRef)::String = join([F.name(p.node) for p in _partsRootFirst(cr)], ".")

#= A literal frontend expression as a Julia value (scalars, nested arrays row-major). =#
function _literalValue(e::F.Expression)
  e = F.getBindingExp(e)
  if e isa F.REAL_EXPRESSION
    return Float64(e.value)
  elseif e isa F.INTEGER_EXPRESSION
    return Int(e.value)
  elseif e isa F.BOOLEAN_EXPRESSION
    return Bool(e.value)
  elseif e isa F.CAST_EXPRESSION
    return _literalValue(e.exp)
  elseif e isa F.ARRAY_EXPRESSION
    local els = [_literalValue(x) for x in e.elements]
    isempty(els) && return Float64[]
    if els[1] isa AbstractArray
      local nd = ndims(els[1])
      return permutedims(cat(els...; dims = nd + 1), (nd + 1, 1:nd...))
    end
    return [x for x in els]
  end
  ns("value not literal after evaluation: " * F.toString(e))
end

function _evalBinding(b::F.Binding, v::F.Variable, what::String)
  F.isBound(b) || ns("$(what) of $(F.toString(v.name)) has no binding")
  local val = _literalValue(F.evalExp(F.getTypedExp(b)))
  local dims = F.isArray(v.ty) ? [F.size(d) for d in F.arrayDims(v.ty)] : Int[]
  if !isempty(dims) && !(val isa AbstractArray)
    return fill(val, dims...)   #= each =#
  end
  return val
end

function _paramValue!(g::Gen, name::String)
  haskey(g.params, name) && return g.params[name]
  local info = g.vars[name]
  local val = _evalBinding(info.var.binding, info.var, "parameter")
  g.params[name] = val
  g.paramSyms[name] = Symbol("P_", replace(name, r"[^A-Za-z0-9_]" => "_"))
  return val
end

#= Start values of a variable, row-major. =#
function _startValues(v::F.Variable, n::Int)::Vector{Float64}
  local start = nothing
  for (an, ab) in v.typeAttributes
    an == "start" && (start = _evalBinding(ab, v, "start"))
  end
  start === nothing && return zeros(n)
  local vals = start isa AbstractArray ? Float64.(vec(permutedims(start, ndims(start):-1:1))) : fill(Float64(start), n)
  length(vals) == n || ns("start of $(F.toString(v.name))")
  return vals
end

#= ---------------------------------------------------------------- frontend expressions to IR =#

const MATH_BUILTINS = Dict("sin" => :sin, "cos" => :cos, "tan" => :tan, "asin" => :asin,
                           "acos" => :acos, "atan" => :atan, "atan2" => :atan, "exp" => :exp,
                           "log" => :log, "log10" => :log10, "sqrt" => :sqrt, "abs" => :abs,
                           "sinh" => :sinh, "cosh" => :cosh, "tanh" => :tanh,
                           "min" => :min, "max" => :max, "sign" => :sign)

function _binaryOp(op)::Symbol
  O = F.Op
  op in (O.ADD, O.ADD_EW, O.ADD_SCALAR_ARRAY, O.ADD_ARRAY_SCALAR) && return :+
  op in (O.SUB, O.SUB_EW, O.SUB_SCALAR_ARRAY, O.SUB_ARRAY_SCALAR) && return :-
  op in (O.MUL, O.MUL_EW, O.MUL_SCALAR_ARRAY, O.MUL_ARRAY_SCALAR) && return :*
  op in (O.DIV, O.DIV_EW, O.DIV_SCALAR_ARRAY, O.DIV_ARRAY_SCALAR) && return :/
  op in (O.POW, O.POW_EW, O.POW_SCALAR_ARRAY, O.POW_ARRAY_SCALAR) && return :^
  ns("operator $(op) (matrix products and the like)")
end

function _relationOp(op)::Symbol
  O = F.Op
  op == O.LESS && return :<
  op == O.LESSEQ && return :<=
  op == O.GREATER && return :>
  op == O.GREATEREQ && return :>=
  op == O.EQUAL && return :(==)
  op == O.NEQUAL && return :!=
  ns("relation operator $(op)")
end

#= Whether an expression varies continuously (states, algebraic variables, time). =#
function _continuous(g::Gen, x::IR)::Bool
  x isa TimeIR && return true
  x isa Ref && return g.vars[x.name].kind in (:state, :alg) || any(s -> _continuous(g, s), x.subs)
  x isa Op && return any(a -> _continuous(g, a), x.args)
  x isa Rel && return true
  return false
end

function toIR(g::Gen, @nospecialize(e::F.Expression), ctx::Ctx)::IR
  if e isa F.REAL_EXPRESSION || e isa F.INTEGER_EXPRESSION || e isa F.BOOLEAN_EXPRESSION
    return Lit(_literalValue(e))
  elseif e isa F.CAST_EXPRESSION
    return toIR(g, e.exp, ctx)
  elseif e isa F.CREF_EXPRESSION
    return _crefIR(g, e, ctx, false)
  elseif e isa F.UNARY_EXPRESSION
    e.operator.op == F.Op.UMINUS || ns("unary operator")
    return Op(:neg, IR[toIR(g, e.exp, ctx)])
  elseif e isa F.BINARY_EXPRESSION
    return Op(_binaryOp(e.operator.op), IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
  elseif e isa F.LUNARY_EXPRESSION
    e.operator.op == F.Op.NOT || ns("logical operator")
    return Op(:!, IR[toIR(g, e.exp, ctx)])
  elseif e isa F.LBINARY_EXPRESSION
    local lop = e.operator.op == F.Op.AND ? :&& : e.operator.op == F.Op.OR ? :|| : ns("logical operator")
    return Op(lop, IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
  elseif e isa F.RELATION_EXPRESSION
    local op = _relationOp(e.operator.op)
    local l = toIR(g, e.exp1, ctx)
    local r = toIR(g, e.exp2, ctx)
    if ctx.noEvent || !(_continuous(g, l) || _continuous(g, r))
      return Op(op, IR[l, r])
    end
    op in (:(==), :!=) && ns("equality relation on continuous variables")
    ctx.nrel[] += 1
    return Rel(op, l, r, ctx.nrel[])
  elseif e isa F.IF_EXPRESSION
    return Op(:if, IR[toIR(g, e.condition, ctx), toIR(g, e.trueBranch, ctx), toIR(g, e.falseBranch, ctx)])
  elseif e isa F.CALL_EXPRESSION
    return _callIR(g, e, ctx)
  elseif e isa F.ARRAY_EXPRESSION && !isempty(ctx.elem)
    #= An element of an array in an element-wise equation. =#
    local val = try
      _literalValue(e)
    catch err
      err isa NotSupported || rethrow(err)
      nothing
    end
    if val === nothing
      #= {a, 2*a, b}: a vector of scalar expressions =#
      length(ctx.elem) == 1 || ns("non-literal array expression " * F.toString(e))
      local inner = Ctx(ctx.iters, Int[], ctx.nrel, ctx.noEvent)
      return Op(:index, IR[Op(:vect, IR[toIR(g, x, inner) for x in e.elements]), Slot(ctx.elem[1])])
    end
    ndims(val) == length(ctx.elem) || ns("array expression rank")
    return Op(:index, IR[Lit(val); IR[Slot(k) for k in ctx.elem]])
  end
  ns("expression " * string(nameof(typeof(e))) * ": " * F.toString(e))
end

function _callIR(g::Gen, e::F.CALL_EXPRESSION, ctx::Ctx)::IR
  local call = e.call
  F.isvariant(call, F.TYPED_CALL) || ns("call " * F.toString(e))
  local fname = F.AbsynUtil.pathString(F.name(call.fn))
  local args = collect(call.arguments)
  if fname == "der"
    args[1] isa F.CREF_EXPRESSION || ns("der of an expression: " * F.toString(e))
    return _crefIR(g, args[1], ctx, true)
  elseif fname == "pre"
    args[1] isa F.CREF_EXPRESSION || ns("pre of an expression: " * F.toString(e))
    local r = _crefIR(g, args[1], ctx, false)
    r isa Ref || ns("pre of " * F.toString(e))
    return Pre(r)
  elseif fname == "homotopy"
    #= homotopy(actual, simplified): the actual expression outside initialization =#
    return toIR(g, args[1], ctx)
  elseif fname == "noEvent"
    return toIR(g, args[1], Ctx(ctx.iters, ctx.elem, ctx.nrel, true))
  elseif fname == "smooth"
    return toIR(g, args[2], ctx)
  end
  local jf = get(MATH_BUILTINS, fname, nothing)
  jf === nothing && ns("function $(fname)")
  return Op(jf, IR[toIR(g, a, ctx) for a in args])
end

function _crefIR(g::Gen, e::F.CREF_EXPRESSION, ctx::Ctx, der::Bool)::IR
  local cr = e.cref
  F.isvariant(cr, F.COMPONENT_REF_CREF) || ns("cref " * F.toString(e))
  if F.isIterator(cr)
    local nm = F.name(cr.node)
    haskey(ctx.iters, nm) || ns("iterator $(nm) outside its loop")
    return Slot(ctx.iters[nm])
  end
  local name = _crefName(cr)
  if name == "time" && !haskey(g.vars, name)
    return TimeIR()
  end
  local info = get(g.vars, name, nothing)
  info === nothing && ns("unknown variable $(name)")
  local parts = _partsRootFirst(cr)
  length(parts) == length(info.perPart) || ns("cref parts of $(name)")
  local subs = IR[]
  local free = 0
  local nextFree = function ()
    free += 1
    free <= length(ctx.elem) || ns("array operand $(F.toString(e)) outside an element-wise context")
    return Slot(ctx.elem[free])
  end
  for (p, n) in zip(parts, info.perPart)
    local ps = collect(p.subscripts)
    if isempty(ps)
      for _ in 1:n
        push!(subs, nextFree())
      end
    else
      length(ps) == n || ns("partial subscripts on $(name)")
      for s in ps
        if s isa F.SUBSCRIPT_INDEX
          push!(subs, toIR(g, s.index, ctx))
        elseif s isa F.SUBSCRIPT_WHOLE
          push!(subs, nextFree())
        else
          ns("subscript " * F.toString(s))
        end
      end
    end
  end
  (free == 0 || free == length(ctx.elem)) || ns("operand rank of $(F.toString(e))")
  der && info.kind != :state && ns("der of non-state $(name)")
  return Ref(name, subs, der)
end

#= ---------------------------------------------------------------- equation classes =#

function _loopValues(@nospecialize(r::F.Expression))::Vector{Int}
  if r isa F.RANGE_EXPRESSION
    local start = _literalValue(F.evalExp(r.start))
    local stop = _literalValue(F.evalExp(r.stop))
    local step = r.step === nothing ? 1 : _literalValue(F.evalExp(F.Util.getOption(r.step)))
    (start isa Int && stop isa Int && step isa Int) && return collect(start:step:stop)
  end
  local vals = _literalValue(F.evalExp(r))
  (vals isa AbstractVector && all(v -> v isa Int, vals)) || ns("for range " * F.toString(r))
  return Vector{Int}(vals)
end

function _productDomain(ranges::Vector{Vector{Int}})::Matrix{Int}
  isempty(ranges) && return Matrix{Int}(undef, 0, 1)
  local n = prod(length.(ranges))
  local d = Matrix{Int}(undef, length(ranges), n)
  local c = 0
  for t in Iterators.product(reverse(ranges)...)   #= first slot varies slowest =#
    c += 1
    for (k, v) in enumerate(reverse(t))
      d[k, c] = v
    end
  end
  return d
end

function _addEquation!(g::Gen, @nospecialize(eq::F.Equation), iterNames::Vector{String}, ranges::Vector{Vector{Int}})
  if F.isvariant(eq, F.EQUATION_FOR)
    local values = _loopValues(F.Util.getOption(eq.range))
    isempty(values) && return
    for b in eq.body
      _addEquation!(g, b, [iterNames; F.name(eq.iterator)], [ranges; [values]])
    end
    return
  end
  local nIter = length(iterNames)
  if F.isvariant(eq, F.EQUATION_WHEN)
    _addWhen!(g, eq, iterNames, ranges)
    return
  end
  if F.isvariant(eq, F.EQUATION_ASSERT)
    any(isempty, ranges) && return
    local ctx = Ctx(Dict(n => k for (k, n) in enumerate(iterNames)), Int[], Base.RefValue(0), true)
    push!(g.asserts, AssertClass(toIR(g, eq.condition, ctx), occursin("warning", F.toString(eq.level)),
                                 length(ranges), _productDomain(ranges), F.toString(eq.condition), F.toString(eq.message)))
    return
  end
  (F.isvariant(eq, F.EQUATION_EQUALITY) || F.isvariant(eq, F.EQUATION_ARRAY_EQUALITY)) ||
    ns("equation " * first(F.toString(eq), 200))
  local edims = F.isArray(eq.ty) ? [F.size(d) for d in F.arrayDims(eq.ty)] : Int[]
  local ctx = Ctx(Dict(n => k for (k, n) in enumerate(iterNames)), collect((nIter + 1):(nIter + length(edims))))
  local lhs = toIR(g, eq.lhs, ctx)
  local rhs = toIR(g, eq.rhs, ctx)
  local allRanges = [ranges; [collect(1:d) for d in edims]]
  any(isempty, allRanges) && return
  push!(g.classes, EqClass(lhs, rhs, length(allRanges), _productDomain(allRanges), ctx.nrel[], first(F.toString(eq), 200)))
end

function _addWhen!(g::Gen, @nospecialize(eq::F.Equation), iterNames::Vector{String}, ranges::Vector{Vector{Int}})
  length(eq.branches) == 1 || ns("elsewhen")
  local br = eq.branches[1]
  F.isvariant(br, F.EQUATION_BRANCH) || ns("when branch")
  any(isempty, ranges) && return
  local ctx = Ctx(Dict(n => k for (k, n) in enumerate(iterNames)), Int[])
  local cond = toIR(g, br.condition, ctx)
  local body = Tuple{Symbol, Ref, IR}[]
  for b in br.body
    if F.isvariant(b, F.EQUATION_REINIT)
      local target = toIR(g, b.cref, ctx)
      (target isa Ref && g.vars[target.name].kind == :state) || ns("reinit of a non-state")
      push!(body, (:reinit, target, toIR(g, b.reinitExp, ctx)))
    elseif F.isvariant(b, F.EQUATION_EQUALITY) && b.lhs isa F.CREF_EXPRESSION && !F.isArray(b.ty)
      local target = toIR(g, b.lhs, ctx)
      (target isa Ref && g.vars[target.name].kind == :discrete) || ns("when assigns a non-discrete variable")
      push!(body, (:assign, target, toIR(g, b.rhs, ctx)))
    else
      ns("when body equation " * first(F.toString(b), 120))
    end
  end
  push!(g.whens, WhenClass(cond, body, length(ranges), _productDomain(ranges), ctx.nrel[], first(F.toString(eq), 200)))
end

#= Sums as sorted term lists, so connection sums that list their terms in another order share
   a shape. =#
function _canonical(x::IR)::IR
  x isa Op || return x
  local args = IR[_canonical(a) for a in x.args]
  if x.op == :+
    local terms = IR[]
    local flat = function (t)
      if t isa Op && t.op == :+
        foreach(flat, t.args)
      else
        push!(terms, t)
      end
    end
    foreach(flat, args)
    sort!(terms; by = t -> irKey(_abstracted(t, Int[])))
    local acc = terms[1]
    for t in terms[2:end]
      acc = Op(:+, IR[acc, t])
    end
    return acc
  end
  return Op(x.op, args)
end

#= Literal integer subscripts replaced by new slots (their values pushed to `vals`). =#
function _abstracted(x::IR, vals::Vector{Int})::IR
  if x isa Ref
    local subs = IR[]
    for s in x.subs
      if s isa Lit && s.v isa Int
        push!(vals, s.v)
        push!(subs, Slot(length(vals)))
      else
        push!(subs, s)
      end
    end
    return Ref(x.name, subs, x.der)
  elseif x isa Pre
    return Pre(_abstracted(x.ref, vals))
  elseif x isa Rel
    local l = _abstracted(x.lhs, vals)
    return Rel(x.op, l, _abstracted(x.rhs, vals), x.id)
  elseif x isa Op
    return Op(x.op, IR[_abstracted(a, vals) for a in x.args])
  end
  return x
end

#= Scalar equations (no slots) that differ only in literal subscripts become one class. =#
function _reroll(classes::Vector{EqClass})::Vector{EqClass}
  local out = EqClass[]
  local groups = Dict{String, Int}()
  local members = Vector{Tuple{IR, IR, Vector{Vector{Int}}, Int, String}}()
  for c in classes
    if c.nslots > 0 || c.nrel > 0
      push!(out, c)
      continue
    end
    local vals = Int[]
    local (l, r) = (_canonical(c.lhs), _canonical(c.rhs))
    #= a = b and b = a are one shape =#
    irKey(_abstracted(l, Int[])) > irKey(_abstracted(r, Int[])) && ((l, r) = (r, l))
    local lhs = _abstracted(l, vals)
    local rhs = _abstracted(r, vals)
    local key = irKey(lhs) * "=" * irKey(rhs)
    local gi = get(groups, key, 0)
    if gi == 0
      push!(members, (lhs, rhs, [vals], c.nrel, c.text))
      groups[key] = length(members)
    else
      push!(members[gi][3], vals)
    end
  end
  for (lhs, rhs, vals, nrel, text) in members
    local nslots = length(vals[1])
    if nslots == 0
      for _ in vals
        push!(out, EqClass(lhs, rhs, 0, Matrix{Int}(undef, 0, 1), nrel, text))
      end
    else
      push!(out, EqClass(lhs, rhs, nslots, reduce(hcat, vals), nrel, text))
    end
  end
  return out
end

#= ---------------------------------------------------------------- element level =#

function _evalInt(g::Gen, x::IR, d::Matrix{Int}, col::Int)::Int
  if x isa Lit
    x.v isa Integer || ns("non-integer subscript")
    return Int(x.v)
  elseif x isa Slot
    return d[x.k, col]
  elseif x isa Op
    local a = [_evalInt(g, y, d, col) for y in x.args]
    x.op == :+ && return a[1] + a[2]
    x.op == :- && return a[1] - a[2]
    x.op == :* && return a[1] * a[2]
    x.op == :neg && return -a[1]
    ns("subscript operator $(x.op)")
  elseif x isa Ref
    local info = g.vars[x.name]
    (info.kind == :param && isempty(info.dims)) || ns("subscript depends on variable $(x.name)")
    push!(g.structural, x.name)
    local v = _paramValue!(g, x.name)
    v isa Integer || ns("non-integer subscript $(x.name)")
    return Int(v)
  end
  ns("subscript")
end

#= Zero-based position of an element in a variable (row-major); subscripts 1-based. =#
function _elementOffset(info::VarInfo, idx::Vector{Int}, name::String)::Int
  length(idx) == length(info.dims) || ns("subscript count of $(name)")
  local off = 0
  for (i, d) in zip(idx, info.dims)
    (1 <= i <= d) || ns("subscript $(i) out of 1:$(d) on $(name)")
    off = off * d + (i - 1)
  end
  return off
end

#= Element structure: per instance its unknowns (with their access positions) and the states
   it reads. Unknown ids: state derivatives 1:nStates, algebraic elements after them. =#
struct Elements
  class::Vector{Int}
  col::Vector{Int}
  uptr::Vector{Int}
  uid::Vector{Int}
  upos::Vector{Int}
  sptr::Vector{Int}
  sid::Vector{Int}
end

function _elements(g::Gen)::Elements
  local class = Int[]; local col = Int[]
  local uptr = Int[1]; local uid = Int[]; local upos = Int[]
  local sptr = Int[1]; local sid = Int[]
  local idx = Int[]
  for (ci, c) in enumerate(g.classes)
    local acc = accesses!(accesses!(Ref[], c.lhs), c.rhs)
    for j in 1:size(c.domain, 2)
      push!(class, ci); push!(col, j)
      for (pos, r) in enumerate(acc)
        local info = g.vars[r.name]
        info.kind in (:param, :discrete) && continue
        empty!(idx)
        for s in r.subs
          push!(idx, _evalInt(g, s, c.domain, j))
        end
        local off = info.offset + _elementOffset(info, idx, r.name) + 1
        if info.kind == :alg
          push!(uid, g.nStates + off); push!(upos, pos)
        elseif r.der
          push!(uid, off); push!(upos, pos)
        else
          push!(sid, off)
        end
      end
      push!(uptr, length(uid) + 1)
      push!(sptr, length(sid) + 1)
    end
  end
  return Elements(class, col, uptr, uid, upos, sptr, sid)
end

#= Maximum matching instances <-> unknowns (greedy, then augmenting paths). =#
function _match(el::Elements, nU::Int)::Vector{Int}
  local N = length(el.class)
  N == nU || ns("$(N) equations for $(nU) unknowns")
  local matchRow = zeros(Int, N)
  local matchCol = zeros(Int, nU)
  for r in 1:N
    for i in el.uptr[r]:(el.uptr[r + 1] - 1)
      local c = el.uid[i]
      if matchCol[c] == 0
        matchRow[r] = c; matchCol[c] = r
        break
      end
    end
  end
  local visited = zeros(Int, nU)
  local stamp = 0
  for r0 in 1:N
    matchRow[r0] == 0 || continue
    stamp += 1
    local rows = [r0]; local iters = [el.uptr[r0]]; local cols = Int[]
    local found = false
    while !isempty(rows)
      local r = rows[end]; local i = iters[end]
      if i >= el.uptr[r + 1]
        pop!(rows); pop!(iters)
        isempty(cols) || pop!(cols)
        continue
      end
      iters[end] = i + 1
      local c = el.uid[i]
      visited[c] == stamp && continue
      visited[c] = stamp
      push!(cols, c)
      if matchCol[c] == 0
        for k in eachindex(rows)
          matchRow[rows[k]] = cols[k]; matchCol[cols[k]] = rows[k]
        end
        found = true
        break
      end
      push!(rows, matchCol[c]); push!(iters, el.uptr[matchCol[c]])
    end
    found || ns("structurally singular (index reduction needed, or under/over-determined)")
  end
  return matchRow
end

#= Kahn's algorithm in batches: all ready instances of one (class, matched access) at a time,
   a key whose remaining instances are all ready first. Returns the batches (consecutive equal
   keys merged), the matched access per instance and the producer per unknown. =#
function _schedule(el::Elements, matchRow::Vector{Int}, nU::Int)
  local N = length(el.class)
  local producer = zeros(Int, nU)
  for r in 1:N
    producer[matchRow[r]] = r
  end
  local pos = zeros(Int, N)
  for r in 1:N
    for i in el.uptr[r]:(el.uptr[r + 1] - 1)
      if el.uid[i] == matchRow[r]
        pos[r] == 0 || ns("the matched unknown occurs twice in an equation instance")
        pos[r] = el.upos[i]
      end
    end
  end
  local indeg = zeros(Int, N)
  local dependents = [Int[] for _ in 1:N]
  local seen = Set{Int}()
  for r in 1:N
    empty!(seen)
    for i in el.uptr[r]:(el.uptr[r + 1] - 1)
      local u = el.uid[i]
      u == matchRow[r] && continue
      local q = producer[u]
      q in seen && continue
      push!(seen, q)
      push!(dependents[q], r)
      indeg[r] += 1
    end
  end
  local queues = Dict{Tuple{Int, Int}, Vector{Int}}()
  local remaining = Dict{Tuple{Int, Int}, Int}()
  local key = r -> (el.class[r], pos[r])
  for r in 1:N
    remaining[key(r)] = get(remaining, key(r), 0) + 1
    indeg[r] == 0 && push!(get!(queues, key(r), Int[]), r)
  end
  local batches = Tuple{Tuple{Int, Int}, Vector{Int}}[]
  local done = 0
  while true
    local best = nothing
    local bestFull = false
    for (k, q) in queues
      isempty(q) && continue
      local full = length(q) == remaining[k]
      if best === nothing || (full && !bestFull) ||
         (full == bestFull && (length(q) > length(queues[best]) || (length(q) == length(queues[best]) && k < best)))
        best = k
        bestFull = full
      end
    end
    best === nothing && break
    local batch = queues[best]
    queues[best] = Int[]
    remaining[best] -= length(batch)
    done += length(batch)
    if !isempty(batches) && batches[end][1] == best
      append!(batches[end][2], batch)
    else
      push!(batches, (best, batch))
    end
    for r in batch
      for d in dependents[r]
        indeg[d] -= 1
        indeg[d] == 0 && push!(get!(queues, key(d), Int[]), d)
      end
    end
  end
  done == N || ns("algebraic loop ($(N - done) equation instances in cycles)")
  return (batches, pos, producer)
end

#= State dependencies of every state derivative (the Jacobian's rows), through the
   algebraic unknowns; nothing when the pattern would be large. =#
function _jacobianPattern(el::Elements, matchRow, batches, producer, nStates::Int)
  local N = length(el.class)
  local deps = Vector{Vector{Int}}(undef, N)
  local total = 0
  for (_, batch) in batches
    for r in batch
      local s = Set{Int}(el.sid[el.sptr[r]:(el.sptr[r + 1] - 1)])
      for i in el.uptr[r]:(el.uptr[r + 1] - 1)
        local u = el.uid[i]
        u == matchRow[r] && continue
        union!(s, deps[producer[u]])
      end
      deps[r] = sort!(collect(s))
      total += length(deps[r])
      total > 20_000_000 && return nothing
    end
  end
  local I = Int[]; local J = Int[]
  for r in 1:N
    local u = matchRow[r]
    u <= nStates || continue
    push!(I, u); push!(J, u)
    for c in deps[r]
      push!(I, u); push!(J, c)
    end
  end
  return (I, J)
end

#= ---------------------------------------------------------------- solving =#

_nAccesses(x::IR) = length(accesses!(Ref[], x))

#= The equation lhs = rhs solved for its access number `pos` (accesses numbered over lhs,
   then rhs): the access and the expression it equals. =#
function _solveFor(lhs::IR, rhs::IR, pos::Int, text::String)
  local nl = _nAccesses(lhs)
  local (s, o, k) = pos <= nl ? (lhs, rhs, pos) : (rhs, lhs, pos - nl)
  while true
    s isa Ref && k == 1 && return (s, o)
    s isa Op || ns("cannot solve for an unknown in: $(text)")
    #= the argument holding access k =#
    local j = 0; local before = 0
    for (ai, a) in enumerate(s.args)
      local na = _nAccesses(a)
      if k <= before + na
        j = ai
        break
      end
      before += na
    end
    local a = s.args[j]
    local others = IR[s.args[i] for i in eachindex(s.args) if i != j]
    if s.op == :+
      o = Op(:-, IR[o, length(others) == 1 ? others[1] : Op(:+, others)])
    elseif s.op == :- && j == 1
      o = Op(:+, IR[o, s.args[2]])
    elseif s.op == :- && j == 2
      o = Op(:-, IR[s.args[1], o])
    elseif s.op == :neg
      o = Op(:neg, IR[o])
    elseif s.op == :* && length(s.args) == 2
      o = Op(:/, IR[o, others[1]])
    elseif s.op == :/ && j == 1
      o = Op(:*, IR[o, s.args[2]])
    elseif s.op == :/ && j == 2
      o = Op(:/, IR[s.args[1], o])
    else
      ns("cannot solve for an unknown under $(s.op) in: $(text)")
    end
    s = a
    k -= before
  end
end

#= ---------------------------------------------------------------- code =#

function _linear(info::VarInfo, idx::Vector)
  isempty(idx) && return info.offset + 1
  local terms = Any[]
  local stride = 1
  for k in length(idx):-1:1
    push!(terms, stride == 1 ? :($(idx[k]) - 1) : :(($(idx[k]) - 1) * $stride))
    stride *= info.dims[k]
  end
  return Expr(:call, :+, info.offset + 1, terms...)
end

function _discreteRead(info::VarInfo, loc)
  info.vtype == :bool && return :($loc != 0.0)
  info.vtype == :int && return :(round(Int, $loc))
  return loc
end

function jl(g::Gen, x::IR)
  if x isa Lit
    return x.v
  elseif x isa Slot
    return Symbol("s", x.k)
  elseif x isa TimeIR
    return :t
  elseif x isa Ref
    local info = g.vars[x.name]
    local idx = Any[jl(g, s) for s in x.subs]
    if info.kind == :param
      info.vtype == :other && ns("parameter $(x.name) of a type the array path does not pass")
      local v = _paramValue!(g, x.name)
      v isa AbstractArray && return _discreteRead(info, Expr(:ref, g.paramSyms[x.name], idx...))
      return g.paramSyms[x.name]
    elseif info.kind == :state
      return x.der ? :(du[$(_linear(info, idx))]) : :(u[$(_linear(info, idx))])
    elseif info.kind == :discrete
      return _discreteRead(info, :(p.d[$(_linear(info, idx))]))
    else
      return :(a[$(_linear(info, idx))])
    end
  elseif x isa Pre
    local info = g.vars[x.ref.name]
    local idx = Any[jl(g, s) for s in x.ref.subs]
    info.kind == :state && return :(p.upre[$(_linear(info, idx))])
    info.kind == :discrete && return _discreteRead(info, :(p.dpre[$(_linear(info, idx))]))
    return jl(g, x.ref)   #= pre of a parameter =#
  elseif x isa Rel
    local k = :($(g.relBase) + (_col - 1) * $(g.relCount) + $(x.id))
    return :($(_rel)(z, p.rel, mode, $k, $(jl(g, x.lhs)), $(jl(g, x.rhs)), $(x.op)))
  elseif x isa Op
    local args = Any[jl(g, a) for a in x.args]
    x.op == :neg && return :(-$(args[1]))
    x.op == :index && return Expr(:ref, args...)
    x.op == :vect && return Expr(:vect, args...)
    x.op == :if && return Expr(:if, args...)
    x.op == :&& && return Expr(:&&, args...)
    x.op == :|| && return Expr(:||, args...)
    return Expr(:call, x.op, args...)
  end
  error("IR")
end

#= A loop over the domain columns `cols` of a class running `body` with the slots (and _col,
   the column) bound: an affine range when the columns allow it, else a table. =#
function _loopCode(nslots::Int, domain::Matrix{Int}, cols::Vector{Int}, body::Expr)
  local M = vcat(domain[:, cols], reshape(cols, 1, :))
  local m = size(M, 2)
  local syms = [[Symbol("s", k) for k in 1:nslots]; :_col]
  local rows = nslots + 1
  if m == 1
    return Expr(:let, Expr(:block, [:($(syms[k]) = $(M[k, 1])) for k in 1:rows]...), body)
  end
  local a = M[:, 1]
  local b = M[:, 2] .- M[:, 1]
  if all(j -> M[:, j] == a .+ b .* (j - 1), 1:m)
    local sets = [b[k] == 0 ? :(local $(syms[k]) = $(a[k])) : :(local $(syms[k]) = $(a[k]) + $(b[k]) * _k) for k in 1:rows]
    return :(for _k in 0:$(m - 1); $(sets...); $body; end)
  end
  local sets = [:(local $(syms[k]) = $M[$k, _c]) for k in 1:rows]
  return :(for _c in 1:$m; $(sets...); $body; end)
end

function _elementNames(v::F.Variable)::Vector{String}
  F.isArray(v.ty) || return [F.toString(v.name)]
  return [F.toString(cr) for cr in F._expandArrayParts(v.name)]
end

#= ---------------------------------------------------------------- parameters =#

#= The parameters a binding reads. =#
function _bindingDeps(g::Gen, b::F.Binding)::Vector{String}
  F.isBound(b) || return String[]
  local deps = String[]
  F.fold(F.getTypedExp(b), (e, acc) -> begin
    if e isa F.CREF_EXPRESSION && F.isvariant(e.cref, F.COMPONENT_REF_CREF) && !F.isIterator(e.cref)
      local nm = _crefName(e.cref)
      local info = get(g.vars, nm, nothing)
      info !== nothing && info.kind == :param && !(nm in deps) && push!(deps, nm)
    end
    acc
  end, 0)
  return deps
end

#= The parameters the equations read and, transitively, those their bindings read, in
   dependency order, each with its binding as code over the others (nothing when the
   binding is not expressible here: the value then cannot follow a changed dependency). =#
function _parameterRules(g::Gen)
  local order = String[]
  local state = Dict{String, Int}()   #= 1 visiting, 2 done =#
  local visit
  visit = function (nm)
    get(state, nm, 0) == 2 && return
    get(state, nm, 0) == 1 && ns("cyclic parameter bindings at $(nm)")
    state[nm] = 1
    for d in _bindingDeps(g, g.vars[nm].var.binding)
      visit(d)
    end
    _paramValue!(g, nm)
    state[nm] = 2
    push!(order, nm)
  end
  foreach(visit, sort!(collect(keys(g.params))))
  local rules = Any[]
  for nm in order
    local info = g.vars[nm]
    local deps = _bindingDeps(g, info.var.binding)
    local code = isempty(deps) ? nothing : try
      _bindingCode(g, info)
    catch err
      err isa NotSupported || rethrow(err)
      nothing
    end
    push!(rules, (g.paramSyms[nm], Symbol[g.paramSyms[d] for d in deps], code))
  end
  return (order, rules)
end

function _bindingCode(g::Gen, info::VarInfo)
  local b = info.var.binding
  local e = F.getTypedExp(b)
  local nd = length(info.dims)
  local deps = _bindingDeps(g, b)
  local locals = [:(local $(g.paramSyms[d]) = vals[$(QuoteNode(g.paramSyms[d]))]) for d in deps]
  local value
  if nd == 0 || F.isEach(b) || F.dimensionCount(F.typeOf(e)) == 0
    local ir = toIR(g, e, Ctx(Dict{String, Int}(), Int[]))
    value = nd == 0 ? jl(g, ir) : :(fill($(jl(g, ir)), $(info.dims...)))
  else
    F.dimensionCount(F.typeOf(e)) == nd || ns("binding rank")
    local ir = toIR(g, e, Ctx(Dict{String, Int}(), collect(1:nd)))
    value = Expr(:comprehension, Expr(:generator, jl(g, ir), [:($(Symbol("s", k)) = 1:$(info.dims[k])) for k in 1:nd]...))
  end
  return :((vals) -> begin
    $(locals...)
    $value
  end)
end

#= ---------------------------------------------------------------- entry =#

"""
    generateArrayODECode(flatModel, modelName) -> (modelName, moduleExpr) or (nothing, reason)

The array-preserving module of a no-scalarize flat model, or `nothing` and the reason the
model is outside the supported scope.
"""
function generateArrayODECode(fm::F.FlatModel, modelName::String)
  try
    return (modelName, _generate(fm, modelName))
  catch err
    err isa NotSupported || rethrow(err)
    return (nothing, err.msg)
  end
end

_vtypeOrOther(v::F.Variable)::Symbol = try
  _vtype(v)
catch err
  err isa NotSupported || rethrow(err)
  :other
end

function _vtype(v::F.Variable)::Symbol
  local ty = F.arrayElementType(v.ty)
  F.isBoolean(ty) && return :bool
  (F.isInteger(ty) || F.isEnumeration(ty)) && return :int
  F.isReal(ty) && return :real
  ns("variable type of $(F.toString(v.name))")
end

function _generate(fm::F.FlatModel, modelName::String)::Expr
  isempty(fm.initialEquations) || ns("initial equations")
  (isempty(fm.algorithms) && isempty(fm.initialAlgorithms)) || ns("algorithms")
  #= States: the variables under der(). =#
  local stateNames = Set{String}()
  local collectDer = function (e, acc)
    if e isa F.CALL_EXPRESSION && F.isvariant(e.call, F.TYPED_CALL) &&
       F.AbsynUtil.pathString(F.name(e.call.fn)) == "der"
      local a = e.call.arguments[1]
      a isa F.CREF_EXPRESSION && push!(stateNames, _crefName(a.cref))
    end
    return acc
  end
  for eq in fm.equations
    F.foldExp(eq, (e, acc) -> F.fold(e, collectDer, acc), 0)
  end
  local g = Gen(Dict{String, VarInfo}(), 0, 0, 0, Dict{String, Any}(), Dict{String, Symbol}(), Set{String}(),
                EqClass[], WhenClass[], AssertClass[], 0, 0)
  local stateNamesOrdered = String[]; local algNamesOrdered = String[]
  local u0 = Float64[]; local d0 = Float64[]; local discreteNamesOrdered = String[]
  for v in fm.variables
    local nm = _crefName(v.name)
    F.hasKnownSize(v.ty) || ns("$(nm) of unknown size")
    local dims = F.isArray(v.ty) ? [F.size(d) for d in F.arrayDims(v.ty)] : Int[]
    local nparts = length(_partsRootFirst(v.name))
    local perPart = isempty(dims) ? zeros(Int, nparts) :
      [length(collect(p.subscripts)) for p in _partsRootFirst(F._expandArrayParts(v.name)[1])]
    local n = isempty(dims) ? 1 : prod(dims)
    local var = F.variability(v)
    if nm in stateNames
      g.vars[nm] = VarInfo(:state, dims, perPart, g.nStates, :real, v)
      g.nStates += n
      append!(u0, _startValues(v, n))
      append!(stateNamesOrdered, _elementNames(v))
    elseif var <= F.Variability.NON_STRUCTURAL_PARAMETER
      g.vars[nm] = VarInfo(:param, dims, perPart, 0, _vtypeOrOther(v), v)
    elseif var == F.Variability.CONTINUOUS
      g.vars[nm] = VarInfo(:alg, dims, perPart, g.nAlg, :real, v)
      g.nAlg += n
      append!(algNamesOrdered, _elementNames(v))
    else
      g.vars[nm] = VarInfo(:discrete, dims, perPart, g.nDiscrete, _vtype(v), v)
      g.nDiscrete += n
      append!(d0, _startValues(v, n))
      append!(discreteNamesOrdered, _elementNames(v))
    end
  end
  g.nStates > 0 || ns("no states")
  for eq in fm.equations
    _addEquation!(g, eq, String[], Vector{Int}[])
  end
  g.classes = _reroll(g.classes)
  #= Every discrete variable is assigned in a when-equation. =#
  local assigned = Set{String}(t.name for w in g.whens for (kind, t, _) in w.body if kind == :assign)
  for (nm, info) in g.vars
    info.kind == :discrete && !(nm in assigned) && ns("discrete variable $(nm) not assigned in a when-equation")
  end
  local el = _elements(g)
  local nU = g.nStates + g.nAlg
  local matchRow = _match(el, nU)
  local (batches, pos, producer) = _schedule(el, matchRow, nU)
  local pattern = _jacobianPattern(el, matchRow, batches, producer, g.nStates)
  #= Relation numbering: per class (then per when) a block of nrel x instances. =#
  local relBases = Int[]; local nRelTotal = 0
  for c in g.classes
    push!(relBases, nRelTotal)
    nRelTotal += c.nrel * size(c.domain, 2)
  end
  local whenRelBases = Int[]; local whenCondBases = Int[]; local nWhenInst = 0
  for w in g.whens
    push!(whenRelBases, nRelTotal)
    nRelTotal += w.nrel * size(w.domain, 2)
    push!(whenCondBases, nWhenInst)
    nWhenInst += size(w.domain, 2)
  end
  #= Code: one loop per batch. =#
  local body = Any[]
  local solved = Dict{Tuple{Int, Int}, Tuple{Ref, IR}}()
  for ((ci, p), batch) in batches
    local c = g.classes[ci]
    local (target, value) = get!(() -> _solveFor(c.lhs, c.rhs, p, c.text), solved, (ci, p))
    g.relBase = relBases[ci]; g.relCount = c.nrel
    local assign = :($(jl(g, target)) = $(jl(g, value)))
    push!(body, c.nslots == 0 ? Expr(:let, :(_col = 1), assign) : _loopCode(c.nslots, c.domain, [el.col[r] for r in batch], assign))
  end
  #= When-equations: each instance fires when its condition becomes true. =#
  local whenBody = Any[]
  for (wi, w) in enumerate(g.whens)
    g.relBase = whenRelBases[wi]; g.relCount = w.nrel
    local stmts = Any[]
    for (kind, target, value) in w.body
      local info = g.vars[target.name]
      local idx = Any[jl(g, s) for s in target.subs]
      if kind == :reinit
        push!(stmts, :(u[$(_linear(info, idx))] = $(jl(g, value))))
      else
        push!(stmts, :(p.d[$(_linear(info, idx))] = Float64($(jl(g, value)))))
      end
    end
    local k = :($(whenCondBases[wi]) + _col)
    local inst = quote
      local _c = $(jl(g, w.cond))
      if fire && _c && !p.wcond[$k]
        $(stmts...)
        _fired = true
      end
      mode && (p.wcond[$k] = _c)
    end
    push!(whenBody, w.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), inst))
  end
  #= Asserts: each instance checked at the start and after every step. =#
  local assertBody = Any[]
  local assertTexts = Tuple{String, String, Bool}[]
  local nAssertInst = 0
  for a in g.asserts
    push!(assertTexts, (a.condition, a.message, a.warning))
    local base = nAssertInst
    local k = length(assertTexts)
    local inst = :(($(jl(g, a.cond))) || _assertFailed(p, $k, $base + _col, t))
    push!(assertBody, a.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(a.nslots, a.domain, collect(1:size(a.domain, 2)), inst))
    nAssertInst += size(a.domain, 2)
  end
  LAST_STATS[] = (classes = length(g.classes), whens = length(g.whens), asserts = length(g.asserts), instances = length(el.class), batches = length(batches),
                  states = g.nStates, algebraics = g.nAlg, discretes = g.nDiscrete, relations = nRelTotal,
                  jacobianEntries = pattern === nothing ? -1 : length(pattern[1]))
  local nStates = g.nStates
  local nAlg = g.nAlg
  local usedParams = collect(keys(g.paramSyms))
  local (paramOrder, rules) = _parameterRules(g)
  #= Layout of the flat parameter vector: (symbol, offset, length, dims), Julia (column-major)
     order inside an array. =#
  local layout = Tuple{Symbol, Int, Int, Vector{Int}}[]
  local layoutOf = Dict{String, Tuple{Int, Int}}()
  local poff = 0
  for nm in paramOrder
    local v = g.params[nm]
    local len = v isa AbstractArray ? length(v) : 1
    push!(layout, (g.paramSyms[nm], poff, len, v isa AbstractArray ? collect(size(v)) : Int[]))
    layoutOf[nm] = (poff, len)
    poff += len
  end
  local paramLocals = Any[]
  for nm in usedParams
    local info = g.vars[nm]
    local (off, len) = layoutOf[nm]
    local sym = g.paramSyms[nm]
    if g.params[nm] isa AbstractArray
      local dims = size(g.params[nm])
      push!(paramLocals, length(dims) == 1 ? :(local $sym = view(p.values, $(off + 1):$(off + len))) :
                                             :(local $sym = reshape(view(p.values, $(off + 1):$(off + len)), $(dims...))))
    else
      push!(paramLocals, :(local $sym = $(_discreteRead(info, :(p.values[$(off + 1)])))))
    end
  end
  local paramTuple = (; (g.paramSyms[k] => g.params[k] for k in paramOrder)...)
  #= Element names of the parameters, for overrides of single elements. =#
  local paramElements = Dict{String, Tuple{Symbol, Tuple}}()
  local paramNames = Dict{String, Symbol}()
  for nm in paramOrder
    local sym = g.paramSyms[nm]
    paramNames[nm] = sym
    paramNames[_OMBACKEND.canonicalName(nm)] = sym
    local info = g.vars[nm]
    isempty(info.dims) && continue
    local subs = vec([reverse(t) for t in Iterators.product(reverse([1:d for d in info.dims])...)])
    for (en, t) in zip(_elementNames(info.var), subs)
      paramElements[en] = (sym, t)
      paramElements[_OMBACKEND.canonicalName(en)] = (sym, t)
    end
  end
  local fixedParams = Symbol[g.paramSyms[nm] for nm in paramOrder
                             if nm in g.structural || F.variability(g.vars[nm].var) <= F.Variability.STRUCTURAL_PARAMETER]
  local ruleExprs = [:(($(QuoteNode(sym)), $(deps), $(code === nothing ? :(nothing) : code))) for (sym, deps, code) in rules]
  local jacProto = pattern === nothing ? :(nothing) :
    :(Symbolics.SparseArrays.sparse($(pattern[1]), $(pattern[2]), ones($(length(pattern[1]))), $nStates, $nStates, (x, y) -> x))
  local paramsType = ArrayModelParameters
  local code = quote
    using DifferentialEquations
    using OrdinaryDiffEq
    import Symbolics

    const STATE_NAMES = $(stateNamesOrdered)
    const ALGEBRAIC_NAMES = $(algNamesOrdered)
    const STATE_INDEX = $(_nameIndex(stateNamesOrdered))
    const ALGEBRAIC_INDEX = $(_nameIndex(algNamesOrdered))
    const DISCRETE_INDEX = $(_nameIndex(discreteNamesOrdered))
    const PARAMS = $(paramTuple)
    const PARAMETER_LAYOUT = $(layout)
    const PARAMETER_NAMES = $(paramNames)
    const PARAMETER_ELEMENTS = $(paramElements)
    const FIXED_PARAMETERS = $(fixedParams)
    #= (parameter, the parameters its binding reads, its binding as a function of them) in
       dependency order =#
    const PARAMETER_RULES = Any[$(ruleExprs...)]
    const U0 = $(u0)
    const D0 = $(d0)
    const N_RELATIONS = $(nRelTotal)
    const N_WHENS = $(nWhenInst)

    #= The parameter values with `overrides` (Modelica names of parameters or of their
       elements => values) applied; parameters whose bindings read a changed one are
       computed again. =#
    function parameterValues(overrides::AbstractDict = Dict{String, Any}())
      local vals = Dict{Symbol, Any}(k => copy(v) for (k, v) in pairs(PARAMS))
      local changed = Set{Symbol}()
      for (name, v) in overrides
        local name_s = string(name)
        if haskey(PARAMETER_NAMES, name_s)
          local sym = PARAMETER_NAMES[name_s]
          vals[sym] = convert(typeof(vals[sym]), v)
          push!(changed, sym)
        elseif haskey(PARAMETER_ELEMENTS, name_s)
          local (sym, t) = PARAMETER_ELEMENTS[name_s]
          vals[sym][t...] = v
          push!(changed, sym)
        else
          error(string("array model: no parameter named ", name_s, " (or it is not used by the equations)"))
        end
      end
      for sym in changed
        sym in FIXED_PARAMETERS && error(string("array model: parameter ", sym, " is structural (array sizes, subscripts); translate again"))
      end
      for (sym, deps, rule) in PARAMETER_RULES
        any(d -> d in changed, deps) || continue
        sym in changed && continue
        rule === nothing && error(string("array model: parameter ", sym, " depends on a changed parameter, but its binding cannot be computed here"))
        vals[sym] = rule(vals)
        push!(changed, sym)
      end
      return (; (k => vals[k] for k in keys(PARAMS))...)
    end

    #= Parameter values (as PARAMS) as the flat vector of PARAMETER_LAYOUT. =#
    function flatParameters(vals)
      local pv = Vector{Float64}(undef, $(poff))
      for (sym, off, len, _) in PARAMETER_LAYOUT
        local v = getfield(vals, sym)
        if v isa AbstractArray
          pv[(off + 1):(off + len)] .= Float64.(vec(v))
        else
          pv[off + 1] = Float64(v)
        end
      end
      return pv
    end

    #= The continuous equations: state derivatives into du, algebraic variables into a;
       relation crossing functions into z (when given), relation values updated in mode. =#
    function equations!(du, a, u, p, t, z, mode::Bool)
      $(paramLocals...)
      @inbounds begin
        $(body...)
      end
      return nothing
    end

    #= The when-equations: conditions (crossing functions into z), bodies of those whose
       condition became true when fire; returns whether one fired. =#
    function whens!(du, a, u, p, t, z, mode::Bool, fire::Bool)
      $(paramLocals...)
      local _fired = false
      @inbounds begin
        $(whenBody...)
      end
      return _fired
    end

    const ASSERTS = $(assertTexts)
    const WARNED = Set{Int}()

    function _assertFailed(p, k::Int, instance::Int, t)
      local (condition, message, warning) = ASSERTS[k]
      warning || throw($(ModelicaAssertionError)(Float64(t), message, condition))
      instance in WARNED || (push!(WARNED, instance); @warn string("Assertion violated at time ", t, ": ", message) condition)
      return true
    end

    function checkAsserts(u, p, t)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, nothing, false)
      $(paramLocals...)
      @inbounds begin
        $(assertBody...)
      end
      return false
    end

    function RHS!(du, u, p, t)
      equations!(du, similar(u, $nAlg), u, p, t, nothing, false)
      return nothing
    end

    #= The algebraic variables at a state (u, t). =#
    function algebraics(u, p, t)
      local a = similar(u, $nAlg)
      equations!(similar(u), a, u, p, t, nothing, false)
      return a
    end

    #= Event iteration at (u, t): relations and when-conditions evaluated again, fired whens
       applied, until nothing changes. pre() reads the values from before the event. =#
    function eventIteration!(u, p, t, fire::Bool)
      local du = similar(u); local a = similar(u, $nAlg)
      p.upre .= u
      p.dpre .= p.d
      for _ in 1:50
        local relBefore = copy(p.rel)
        equations!(du, a, u, p, t, nothing, true)
        local fired = whens!(du, a, u, p, t, nothing, true, fire)
        (fired || relBefore != p.rel) || return nothing
        p.dpre .= p.d
      end
      error(string("array model: event iteration did not converge at t = ", t))
    end

    function conditions!(out, u, t, integrator)
      local p = integrator.p
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, out, false)
      whens!(du, a, u, p, t, out, false, false)
      return nothing
    end

    function _logDiscretes!(p, t)
      (isempty(p.dlog) || p.dlog[end][2] != p.d) && push!(p.dlog, (t, copy(p.d)))
      return nothing
    end

    #= A crossing: relations and when-conditions first take their values at the left limit
       (the state just before the event), so a when fires exactly when its condition rises
       across the event; then the event iteration at the event. =#
    function affectCrossing!(integrator, _)
      local t = integrator.t
      local tl = max(integrator.tprev, t - 1e-10 * (1 + abs(t)))
      local ul = tl < t ? integrator(tl) : copy(integrator.uprev)
      local p = integrator.p
      local du = similar(ul); local a = similar(ul, $nAlg)
      equations!(du, a, ul, p, tl, nothing, true)
      whens!(du, a, ul, p, tl, nothing, true, false)
      eventIteration!(integrator.u, p, t, true)
      _logDiscretes!(p, t)
      DifferentialEquations.u_modified!(integrator, true)
      return nothing
    end

    function affect!(integrator)
      eventIteration!(integrator.u, integrator.p, integrator.t, true)
      _logDiscretes!(integrator.p, integrator.t)
      DifferentialEquations.u_modified!(integrator, true)
      return nothing
    end

    #= After a step: whether a relation has another value than the stored one. A crossing
       right after an event (the ball leaving the floor it was reinitialized just below) has
       no sign change the root finder sees; it is caught here, at the end of the step. =#
    function relationsChanged(u, t, integrator)
      local p = integrator.p
      local rel0 = copy(p.rel); local w0 = copy(p.wcond)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, nothing, true)
      whens!(du, a, u, p, t, nothing, true, false)
      local changed = rel0 != p.rel
      p.rel .= rel0; p.wcond .= w0
      return changed
    end

    #= The problem and its event callbacks, as getMTKProblem returns them. =#
    function $(Symbol(modelName, "Model"))(tspan = (0.0, 1.0); parameters::AbstractDict = Dict{String, Any}())
      local p = $(paramsType)(flatParameters(isempty(parameters) ? PARAMS : parameterValues(parameters)),
                              zeros(Bool, N_RELATIONS), copy(D0), copy(D0), zeros(Bool, N_WHENS), copy(U0),
                              Tuple{Float64, Vector{Float64}}[], @__MODULE__)
      local u0 = copy(U0)
      #= relations and when-conditions at the start (no when fires there) =#
      (N_RELATIONS > 0 || N_WHENS > 0) && eventIteration!(u0, p, tspan[1], false)
      _logDiscretes!(p, tspan[1])
      local f = ODEFunction(RHS!; jac_prototype = $jacProto, sys = $(ArraySystem)(@__MODULE__))
      #= RightRootFind: the event lands just past the root, where the relation has its new value. =#
      local cbs = Any[]
      N_RELATIONS > 0 && push!(cbs, VectorContinuousCallback(conditions!, affectCrossing!, N_RELATIONS;
                                                             rootfind = DifferentialEquations.SciMLBase.RightRootFind),
                                    DiscreteCallback(relationsChanged, affect!; save_positions = (false, false)))
      if !isempty(ASSERTS)
        empty!(WARNED)
        push!(cbs, DiscreteCallback((u, t, integrator) -> checkAsserts(u, integrator.p, t), integrator -> nothing;
                                    initialize = (c, u, t, integrator) -> checkAsserts(u, integrator.p, t),
                                    save_positions = (false, false)))
      end
      local cb = isempty(cbs) ? nothing : CallbackSet(cbs...)
      return (ODEProblem(f, u0, tspan, p), cb)
    end

    function simulate(tspan = (0.0, 1.0), solver = Tsit5(); parameters::AbstractDict = Dict{String, Any}(), kwargs...)
      local (prob, callbacks) = $(Symbol(modelName, "Model"))(tspan; parameters = parameters)
      return DifferentialEquations.solve(prob, solver; callback = callbacks, kwargs...)
    end
  end
  return Expr(:module, true, Symbol(modelName), stripBeginBlocks(code))
end

end #= module ArrayODEGen =#
