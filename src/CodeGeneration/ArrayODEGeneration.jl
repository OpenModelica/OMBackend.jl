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
  5. Events, as OpenModelica (and OMBackend's MTK path, relationRefresh.jl): a relation on
     continuous variables keeps its value between events, its crossing function shifted by a
     hysteresis; the event-generating functions (integer, floor, ceil, div, mod, rem) the
     same; when-equations and when-statements (also in loops, vector conditions) assign
     discrete variables or reinit states; at an event, sweeps with pre() fixed until nothing
     changes, reinit() applied after each sweep; sample() ticks as preset times; asserts
     reported where their condition fails.
  6. Initialization: the initial equations and fixed starts over the free states and the
     fixed = false parameters (NonlinearSolve, relations fixed per solve until they settle),
     then the initial algorithms.

Not (yet) handled, reported with the reason so the caller scalarizes as before: algebraic
loops, systems needing index reduction, calls of Modelica functions, record variables,
elsewhen in when-equations, array slices.
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
  sampleActive::Vector{Bool}   #= sample() instances true at the current event =#
  initPhase::Vector{Bool}      #= [true] while initializing: initial() =#
  checking::Vector{Bool}       #= [true] while asserts in algorithms are checked =#
  upre::Vector{Float64}
  ureinit::Vector{Float64}     #= the state reinit() sets in a sweep, applied at its end =#
  wnew::Vector{Bool}           #= when-conditions of the current pass, before the bodies run =#
  dlog::Vector{Tuple{Float64, Vector{Float64}}}   #= from each event on: discrete values, then relations and event-function values =#
  evv::Vector{Float64}         #= values of the event-generating functions (_evfn), by relation number =#
  hyst::Vector{Float64}        #= [H]: the relations' hysteresis (_rel); 0 (literal relations) while initializing =#
  tend::Vector{Float64}        #= [stop time]: a solution read there gives the final values (_dlogIndex) =#
  model::Module
end

#= The entry of the discrete log for time t: at an event instant the values before it (the left
   limit, as the states' interpolation and OpenModelica's result files), at the stop time the
   final ones; `right` takes the values after an event instant (the second of a saved pair). =#
_dlogIndex(times, t, tend, right::Bool = false) =
  (right || t >= tend) ? max(1, searchsortedlast(times, t)) : max(1, searchsortedfirst(times, t) - 1)

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
  j > 0 && return [m.algebraics(sol.u[i], p, sol.t[i], i > 1 && sol.t[i - 1] == sol.t[i])[j] for i in eachindex(sol.t)]
  local k = get(m.DISCRETE_INDEX, key, 0)
  if k > 0 && !isempty(p.dlog)
    local times = first.(p.dlog)
    #= an event saves its instant twice: before (left limit), then after =#
    local after(i) = i > 1 && sol.t[i - 1] == sol.t[i]
    return [p.dlog[_dlogIndex(times, sol.t[i], p.tend[1], after(i))][2][k] for i in eachindex(sol.t)]
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
    isempty(times) ? NaN : p.dlog[_dlogIndex(times, t, p.tend[1])][2][i]
  end
end
SII.observed(sys::ArraySystem, x::ArrayVar) = invoke(SII.observed, Tuple{ArraySystem, Any}, sys, x)
SII.all_variable_symbols(sys::ArraySystem) = vcat(getfield(sys, :stateSyms), collect(keys(_observedIdx(sys))))
SII.all_symbols(sys::ArraySystem) = vcat(SII.all_variable_symbols(sys), :t)
SII.default_values(::ArraySystem) = Dict()

#= A relation of the model (MLS 8.5, OpenModelica's relationhysteresis, as the MTK path's
   relationRefresh.jl): it keeps its value p.rel[k] between events and changes only in mode
   (an event). zc <= 0 where it is true; its crossing function, into z when given, is shifted
   by eps = H * (1 + max(|x|, |y|)) away from its value, so the root lands where the new value
   is clear: a true relation stays true while zc <= eps, a false one becomes true at zc <= -eps.
   With H = 0 (initialization) it is evaluated literally. =#
_finiteAbs(x) = (local v = abs(x); v < 1.0e300 ? v : 0.0)
@inline function _rel(z, p, mode::Bool, k::Int, x, y, op)
  local H = p.hyst[1]
  local zc = (op === (>) || op === (>=)) ? y - x : x - y
  local old = p.rel[k]
  local eps = H * (1 + max(_finiteAbs(x), _finiteAbs(y)))
  z === nothing || (z[k] = old ? zc - eps : zc + eps)
  mode && (p.rel[k] = H == 0 ? op(x, y) : old ? zc <= eps : zc <= -eps)
  return p.rel[k]
end

#= An event-generating function (integer, floor, ceil, div, mod, rem; MLS 3.7.2): its value
   v = floor/ceil/trunc(q) changes only at events (in mode), and then (as a relation, _rel)
   only where q has left the interval that keeps v by more than the hysteresis eps: the state
   just before an event (an interpolation a hair past the boundary) keeps the old value. Two
   crossing functions bound that interval: z[k] crosses where q passes its upper end, z[k + 1]
   its lower end. At the initialization (H = 0) the value is literal. =#
_evInterval(kind::Symbol, v) = kind === :floor ? (v, v + 1) : kind === :ceil ? (v - 1, v) :
                               v > 0 ? (v, v + 1) : v < 0 ? (v - 1, v) : (-1.0, 1.0)
@inline function _evfn(z, p, mode::Bool, k::Int, kind::Symbol, q)
  local v = p.evv[k]
  local eps = p.hyst[1] * (1 + _finiteAbs(q))
  local (lo, hi) = _evInterval(kind, v)
  if mode && (p.hyst[1] == 0 || q - hi - eps >= 0 || q - lo + eps < 0)
    v = p.evv[k] = Float64(kind === :floor ? floor(q) : kind === :ceil ? ceil(q) : trunc(q))
    (lo, hi) = _evInterval(kind, v)
  end
  if z !== nothing
    z[k] = q - hi - eps
    z[k + 1] = q - lo + eps
  end
  return v
end

#= The hysteresis H for a solve, as the MTK path (relationRefresh.jl _hysteresis): OpenModelica's
   tolZC = 1e-4 * min(reltol, span / 500). =#
_hysteresis(integrator) = 1.0e-4 * max(min(Float64(first(integrator.opts.reltol)),
                                           abs(Float64(integrator.sol.prob.tspan[2] - integrator.sol.prob.tspan[1])) / 500), 1.0e-12)

#= true (the default): the array path takes events (relations with OpenModelica's hysteresis,
   when, sample, initial(), event iteration in sweeps), asserts and initialization too. false:
   continuous models only (no events, asserts, initial equations or fixed starts on non-states);
   the others go to OMBackend's ModelingToolkit path. =#
const FULL = Ref(true)

#= Sizes of the last generated model (classes, instances, batches = loops, ...), for diagnostics. =#
const LAST_STATS = Ref{Any}(nothing)

struct NotSupported <: Exception
  msg::String
end
ns(msg::AbstractString) = throw(NotSupported(String(msg)))

#= The frontend's evaluation of a binding, range or start value; a failure there (an
   expression its evaluator does not handle) puts the model outside the array path. =#
function _evalFrontend(@nospecialize(e))
  try
    return F.evalExp(e)
  catch err
    err isa NotSupported && rethrow(err)
    ns("the frontend cannot evaluate " * _str(e) * ": " * Base.first(sprint(showerror, err), 120))
  end
end

#= An expression in a reason message (printing must not fail). =#
_str(@nospecialize(x)) = try
  Base.first(F.toString(x), 200)
catch
  string("<", nameof(typeof(x)), ">")
end

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
#= An event-generating function floor/ceil/trunc of q (_evfn); id and id + 1 are its crossing
   slots in the relation numbering of its template. =#
struct EvFn <: IR
  kind::Symbol
  arg::IR
  id::Int
end

irKey(x::Lit) = x.v isa AbstractArray ? string("arr", objectid(x.v)) : repr(x.v)
irKey(x::Slot) = string("\$", x.k)
irKey(::TimeIR) = "time"
irKey(x::Ref) = string(x.der ? "der(" : "", x.name, "[", join(irKey.(x.subs), ","), "]", x.der ? ")" : "")
irKey(x::Pre) = string("pre(", irKey(x.ref), ")")
irKey(x::Rel) = string("rel", x.id, x.op, "(", irKey(x.lhs), ",", irKey(x.rhs), ")")
irKey(x::Op) = string(x.op, "(", join(irKey.(x.args), ","), ")")
irKey(x::EvFn) = string("ev", x.id, x.kind, "(", irKey(x.arg), ")")

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
  elseif x isa EvFn
    accesses!(acc, x.arg)
  end
  return acc
end

_nRel(x::IR) = x isa Rel ? max(x.id, _nRel(x.lhs), _nRel(x.rhs)) :
               x isa EvFn ? max(x.id + 1, _nRel(x.arg)) :
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
  cond::IR                               #= Op(:elsewhen, conditions) for when ... elsewhen =#
  body::Vector{Tuple{Symbol, Ref, IR}}   #= (:assign, :reinit or :call (a statement), target, value) =#
  starts::Vector{Int}                    #= where each branch's body begins =#
  nslots::Int
  domain::Matrix{Int}
  nrel::Int
  nsample::Int
  text::String
end

#= Statements of an algorithm section. =#
abstract type Stmt end
struct SAssign <: Stmt
  target::Ref
  value::IR
end
struct SIf <: Stmt
  branches::Vector{Tuple{IR, Vector{Stmt}}}   #= else: condition Lit(true) =#
end
struct SFor <: Stmt
  slot::Int
  range::Vector{Int}
  body::Vector{Stmt}
end
struct SWhile <: Stmt
  cond::IR
  body::Vector{Stmt}
end
#= A when-statement: each branch has its condition value in wcond (ids within the template). =#
struct SWhen <: Stmt   #= branches (condition, body); ids: a condition slot per element of each branch =#
  branches::Vector{Tuple{IR, Vector{Stmt}}}
  ids::Vector{Vector{Int}}
end
struct SCall <: Stmt   #= a call without a result (a check, a table's validation) =#
  call::IR
end
struct SAssert <: Stmt
  cond::IR
  k::Int   #= in ASSERTS =#
  msg::IR  #= the message, evaluated when the assert fails =#
end
struct SBreak <: Stmt end

#= An algorithm section over a domain (a vectorized algorithm keeps its loop). =#
struct AlgClass
  body::Vector{Stmt}
  nslots::Int
  domain::Matrix{Int}
  nrel::Int
  nwhen::Int
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
  msg::IR
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
  initClasses::Vector{EqClass}
  algs::Vector{AlgClass}
  assertTexts::Vector{Tuple{String, String, Bool}}
  whens::Vector{WhenClass}
  asserts::Vector{AssertClass}
  relBase::Int              #= relation numbering of the class being generated =#
  relCount::Int
  sampleBase::Int           #= sample numbering of the when being generated =#
  sampleCount::Int
  tmpSlot::Int              #= temporary slots (expanded reductions, products): -1, -2, ... =#
  usesInitial::Bool
  discUnknown::Dict{Int, Int}  #= discrete element (offset) computed by an equation -> unknown id =#
  initAlgs::Vector{AlgClass}   #= initial algorithms (run once at the initialization) =#
  freeParams::Vector{String}   #= parameters with fixed = false: unknowns of the initialization =#
  initAsserts::Vector{AssertClass}   #= asserts of initial equations: checked once after the initialization =#
  functions::Dict{String, Int}  #= the model's Modelica functions (Julia name: dots as _) -> number of outputs =#
end

struct Ctx
  iters::Dict{String, Int}  #= frontend iterator name -> slot =#
  elem::Vector{Int}         #= slots of the element-wise dimensions =#
  nrel::Base.RefValue{Int}  #= relations numbered so far in this template =#
  noEvent::Bool
  nsam::Base.RefValue{Int}  #= sample() calls numbered so far in this template =#
end
Ctx(iters, elem) = Ctx(iters, elem, Base.RefValue(0), false, Base.RefValue(0))
Ctx(iters, elem, nrel, noEvent) = Ctx(iters, elem, nrel, noEvent, Base.RefValue(0))
_with(ctx::Ctx; elem = ctx.elem, noEvent = ctx.noEvent) = Ctx(ctx.iters, elem, ctx.nrel, noEvent, ctx.nsam)

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
  elseif e isa F.ENUM_LITERAL_EXPRESSION
    return e.index
  elseif e isa F.STRING_EXPRESSION
    return String(e.value)
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
  ns("value not literal after evaluation: " * _str(e))
end

function _evalBinding(b::F.Binding, v::F.Variable, what::String)
  F.isBound(b) || ns("$(what) of $(_str(v.name)) has no binding")
  local val = _literalValue(_evalFrontend(F.getTypedExp(b)))
  local dims = F.isArray(v.ty) ? [F.size(d) for d in F.arrayDims(v.ty)] : Int[]
  if !isempty(dims) && !(val isa AbstractArray)
    return fill(val, dims...)   #= each =#
  end
  return val
end

function _paramValue!(g::Gen, name::String)
  haskey(g.params, name) && return g.params[name]
  local info = g.vars[name]
  local val = name in g.freeParams ? _freeParamStart(info.var) : _evalBinding(info.var.binding, info.var, "parameter")
  g.params[name] = val
  g.paramSyms[name] = Symbol("P_", replace(name, r"[^A-Za-z0-9_]" => "_"))
  return val
end

#= A parameter with fixed = false starts (the initialization's guess) from its start value. =#
function _freeParamStart(v::F.Variable)
  local dims = F.isArray(v.ty) ? [F.size(d) for d in F.arrayDims(v.ty)] : Int[]
  local st = _startValues(v, isempty(dims) ? 1 : prod(dims))
  isempty(dims) && return st[1]
  return permutedims(reshape(st, reverse(dims)...), length(dims):-1:1)   #= row-major start values =#
end

#= Start values of a variable, row-major. =#
function _startValues(v::F.Variable, n::Int)::Vector{Float64}
  local start = nothing
  for (an, ab) in v.typeAttributes
    an == "start" && (start = _evalBinding(ab, v, "start"))
  end
  start === nothing && return zeros(n)
  local vals = start isa AbstractArray ? Float64.(vec(permutedims(start, ndims(start):-1:1))) : fill(Float64(start), n)
  length(vals) == n || ns("start of $(_str(v.name))")
  return vals
end

#= The fixed attribute of a variable's elements, row-major (default false). =#
function _fixedValues(v::F.Variable, n::Int; default::Bool = false)::Vector{Bool}
  for (an, ab) in v.typeAttributes
    if an == "fixed"
      local f = _evalBinding(ab, v, "fixed")
      local vals = f isa AbstractArray ? Bool.(vec(permutedims(f, ndims(f):-1:1))) : fill(Bool(f), n)
      length(vals) == n || ns("fixed of $(_str(v.name))")
      return vals
    end
  end
  return fill(default, n)
end

#= ---------------------------------------------------------------- frontend expressions to IR =#

const MATH_BUILTINS = Dict("sin" => :sin, "cos" => :cos, "tan" => :tan, "asin" => :asin,
                           "acos" => :acos, "atan" => :atan, "atan2" => :atan, "exp" => :exp,
                           "log" => :log, "log10" => :log10, "sqrt" => :sqrt, "abs" => :abs,
                           "sinh" => :sinh, "cosh" => :cosh, "tanh" => :tanh,
                           "min" => :min, "max" => :max, "sign" => :sign)
#= String(...) of Modelica, as the MTK path formats it (resolved when called: that module
   may load after this one) =#
_modelicaString(args...) = _OMBACKEND.CodeGeneration.AlgorithmicCodeGeneration.modelica_String(args...)

const MATH_WRAPPED = ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log", "log10")

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
  return false   #= an EvFn is constant between events =#
end

#= A Boolean-valued expression (whatever type the call's argument was given: String(n > 3)). =#
_booleanExp(@nospecialize(e)) = e isa F.CAST_EXPRESSION ? _booleanExp(e.exp) :
  (e isa F.RELATION_EXPRESSION || e isa F.LBINARY_EXPRESSION || e isa F.LUNARY_EXPRESSION || e isa F.BOOLEAN_EXPRESSION)

#= An assert's message as an expression (String(x, ...) formatted as OpenModelica does);
   its text where the array path cannot evaluate it. =#
function _messageIR(g::Gen, @nospecialize(m::F.Expression), ctx::Ctx)::IR
  try
    return toIR(g, m, _with(ctx; noEvent = true))
  catch err
    err isa NotSupported || rethrow(err)
    return Lit(_str(m))
  end
end

function toIR(g::Gen, @nospecialize(e::F.Expression), ctx::Ctx)::IR
  if e isa F.REAL_EXPRESSION || e isa F.INTEGER_EXPRESSION || e isa F.BOOLEAN_EXPRESSION || e isa F.ENUM_LITERAL_EXPRESSION
    return Lit(_literalValue(e))
  elseif e isa F.BINDING_EXP
    #= a binding the frontend propagated (into an argument): its expression =#
    return toIR(g, F.getBindingExp(e), ctx)
  elseif e isa F.STRING_EXPRESSION
    return Lit(String(e.value))
  elseif e isa F.TUPLE_ELEMENT_EXPRESSION
    #= the output, and in an element-wise context its element (Buildings' SignalRanker,
       y = Vectors.sort(u): the first output, a vector, was assigned to y[k]) =#
    local out = Op(:index, IR[toIR(g, e.tupleExp, _with(ctx; elem = Int[])), Lit(e.index)])
    if F.isArray(F.typeOf(e)) && !isempty(ctx.elem)
      length(ctx.elem) == F.dimensionCount(F.typeOf(e)) || ns("an array output outside an element-wise context")
      return Op(:index, IR[out, (Slot(k) for k in ctx.elem)...])
    end
    return out
  elseif e isa F.CAST_EXPRESSION
    return toIR(g, e.exp, ctx)
  elseif e isa F.CREF_EXPRESSION
    return _crefIR(g, e, ctx, false)
  elseif e isa F.UNARY_EXPRESSION
    e.operator.op == F.Op.UMINUS || ns("unary operator")
    return Op(:neg, IR[toIR(g, e.exp, ctx)])
  elseif e isa F.BINARY_EXPRESSION
    local O = F.Op
    e.operator.op in (O.SCALAR_PRODUCT, O.MUL_MATRIX_VECTOR, O.MUL_VECTOR_MATRIX, O.MATRIX_PRODUCT) &&
      return _productIR(g, e, ctx)
    #= a + b of Strings: concatenation =#
    F.isString(F.typeOf(e)) && return Op(:string, IR[toIR(g, e.exp1, ctx), toIR(g, e.exp2, ctx)])
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
      length(ctx.elem) == 1 || ns("non-literal array expression " * _str(e))
      local inner = _with(ctx; elem = Int[])
      return Op(:index, IR[Op(:vect, IR[toIR(g, x, inner) for x in e.elements]), Slot(ctx.elem[1])])
    end
    ndims(val) == length(ctx.elem) || ns("array expression rank")
    return Op(:index, IR[Lit(val); IR[Slot(k) for k in ctx.elem]])
  end
  ns("expression " * string(nameof(typeof(e))) * ": " * _str(e))
end

function _callIR(g::Gen, e::F.CALL_EXPRESSION, ctx::Ctx)::IR
  local call = e.call
  F.isvariant(call, F.TYPED_REDUCTION) && return _iterReductionIR(g, call, ctx, _str(e))
  F.isvariant(call, F.TYPED_CALL) || ns("call " * _str(e))
  local fname = F.AbsynUtil.pathString(F.name(call.fn))
  local args = collect(call.arguments)
  #= delay() reads its argument's history: the ModelingToolkit path's (delays.jl). It was
     the function bodies' builtin, delay(x, d) = x (OM.jl's DelayChain: no delay). =#
  fname in ("delay", "OpenModelica.Internal.delay2", "OpenModelica.Internal.delay3") && ns("delay()")
  if fname == "der"
    args[1] isa F.CREF_EXPRESSION || ns("der of an expression: " * _str(e))
    return _crefIR(g, args[1], ctx, true)
  elseif fname == "pre"
    args[1] isa F.CREF_EXPRESSION || ns("pre of an expression: " * _str(e))
    local r = _crefIR(g, args[1], ctx, false)
    r isa Ref || ns("pre of " * _str(e))
    return Pre(r)
  elseif fname == "homotopy"
    #= homotopy(actual, simplified): the actual expression outside initialization =#
    return toIR(g, args[1], ctx)
  elseif fname == "noEvent"
    return toIR(g, args[1], _with(ctx; noEvent = true))
  elseif fname == "smooth"
    return toIR(g, args[2], ctx)
  elseif fname == "initial"
    g.usesInitial = true
    return Op(:initial, IR[])
  elseif fname == "edge" || fname == "change"
    local r = toIR(g, args[1], ctx)
    r isa Ref || ns("$(fname) of " * _str(args[1]))
    return fname == "edge" ? Op(:&&, IR[r, Op(:!, IR[Pre(r)])]) : Op(:!=, IR[r, Pre(r)])
  elseif fname == "sample" || endswith(fname, ".sample")
    length(args) == 2 || ns("sample with $(length(args)) arguments")
    ctx.nsam[] += 1
    return Op(:sample, IR[toIR(g, args[1], ctx), toIR(g, args[2], ctx), Lit(ctx.nsam[])])
  elseif fname == "semiLinear"
    #= semiLinear(x, k1, k2) = smooth(0, if x >= 0 then k1*x else k2*x) =#
    local x = toIR(g, args[1], ctx)
    return Op(:if, IR[Op(:>=, IR[x, Lit(0.0)]), Op(:*, IR[x, toIR(g, args[2], ctx)]), Op(:*, IR[x, toIR(g, args[3], ctx)])])
  elseif fname in ("sum", "product") || (fname in ("min", "max") && length(args) == 1)
    return _reductionIR(g, fname, args[1], ctx)
  elseif fname == "String"
    #= String(r, significantDigits, minimumLength, leftJustified), String(r, format),
       String(i|b|e, minimumLength, leftJustified): OpenModelica's formatting (modelica_String) =#
    local xs = IR[toIR(g, a, ctx) for a in args]
    local ty = F.typeOf(args[1])
    length(xs) == 2 && return Op(:String, xs)
    length(xs) == 4 && return Op(:String, IR[xs[1], Op(:Int, IR[xs[2]]), Op(:Int, IR[xs[3]]), Op(:Bool, IR[xs[4]])])
    length(xs) == 3 || ns("String with $(length(xs)) arguments")
    local v = F.isEnumeration(ty) ? Op(:index, IR[Lit(Tuple(String[String(l) for l in ty.literals])), Op(:Int, IR[xs[1]])]) :
              (F.isBoolean(ty) || _booleanExp(args[1])) ? Op(:Bool, IR[xs[1]]) :
              F.isInteger(ty) ? Op(:Int, IR[xs[1]]) : ns("String of " * _str(args[1]))
    return Op(:String, IR[v, Op(:Int, IR[xs[2]]), Op(:Bool, IR[xs[3]])])
  elseif fname in ("integer", "floor", "ceil", "div", "mod", "rem")
    #= event-generating (MLS 3.7.2) unless under noEvent or of discrete arguments:
       mod(x, y) = x - floor(x / y) * y, rem(x, y) = x - div(x, y) * y =#
    local xs = IR[toIR(g, a, ctx) for a in args]
    local q = length(xs) == 2 ? Op(:/, IR[xs[1], xs[2]]) : xs[1]
    local kind = fname in ("integer", "floor", "mod") ? :floor : fname == "ceil" ? :ceil : :trunc
    local v = if ctx.noEvent || !_continuous(g, q)
      Op(kind, IR[q])
    else
      ctx.nrel[] += 2
      EvFn(kind, q, ctx.nrel[] - 1)
    end
    fname in ("mod", "rem") || return v
    return Op(:-, IR[xs[1], Op(:*, IR[v, xs[2]])])
  end
  #= a Modelica function of the model (_functionDefinitions): its outputs as a tuple if more
     than one (the frontend picks one with a TUPLE_ELEMENT_EXPRESSION) =#
  local jn = replace(fname, "." => "_")
  if haskey(g.functions, jn)
    local xs = IR[]
    for a in args
      F.isRecord(F.typeOf(a)) && ns("function $(fname) with a record argument")
      push!(xs, F.isArray(F.typeOf(a)) ? _arrayValueIR(g, a, ctx) : toIR(g, a, _with(ctx; elem = Int[])))
    end
    local call = Op(:fcall, IR[Lit(Symbol(jn)), xs...])
    #= a function of more outputs typed as its first (no tuple element around it: Buildings'
       SignalRanker, `y = Modelica.Math.Vectors.sort(u)`): that output, not the tuple =#
    g.functions[jn] > 1 && !F.isvariant(F.typeOf(e), F.TYPE_TUPLE) && (call = Op(:index, IR[call, Lit(1)]))
    #= an array result in an element-wise context: the element =#
    if F.isArray(F.typeOf(e))
      local nd = F.dimensionCount(F.typeOf(e))
      length(ctx.elem) == nd || ns("array result of $(fname) outside an element-wise context")
      return Op(:index, IR[call, (Slot(k) for k in ctx.elem)...])
    end
    return call
  end
  #= array built-ins in an element-wise context: the element =#
  if F.isArray(F.typeOf(e)) && fname in ("fill", "zeros", "ones", "identity", "transpose")
    local nd = F.dimensionCount(F.typeOf(e))
    length(ctx.elem) == nd || ns("$(fname) outside an element-wise context")
    fname == "fill" && (F.isArray(F.typeOf(args[1])) ? ns("fill of an array") : return toIR(g, args[1], _with(ctx; elem = Int[])))
    fname == "zeros" && return Lit(0.0)
    fname == "ones" && return Lit(1.0)
    fname == "identity" && return Op(:if, IR[Op(:(==), IR[Slot(ctx.elem[1]), Slot(ctx.elem[2])]), Lit(1.0), Lit(0.0)])
    return toIR(g, args[1], _with(ctx; elem = reverse(ctx.elem)))
  end
  if fname == "size" && length(args) == 2
    #= size(a, k) of a known size =#
    local ty = F.typeOf(args[1])
    (F.isArray(ty) && F.hasKnownSize(ty)) || ns("size of " * _str(args[1]))
    local k = _literalValue(_evalFrontend(args[2]))
    return Lit([F.size(d) for d in F.arrayDims(ty)][k])
  end
  #= Modelica.Math's elementary functions are the built-in ones (external "builtin") =#
  local jf = startswith(fname, "Modelica.Math.") && fname[15:end] in MATH_WRAPPED ?
    MATH_BUILTINS[fname[15:end]] : get(MATH_BUILTINS, fname, nothing)
  jf === nothing && ns("function $(fname)")
  return Op(jf, IR[toIR(g, a, ctx) for a in args])
end

#= An array value (a function's array argument): its elements written out (as a reduction's
   terms), column-major, reshaped to its dimensions. =#
function _arrayValueIR(g::Gen, @nospecialize(e::F.Expression), ctx::Ctx)::IR
  local ty = F.typeOf(e)
  F.hasKnownSize(ty) || ns("array argument of unknown size " * _str(e))
  local dims = [F.size(d) for d in F.arrayDims(ty)]
  prod(dims) <= MAX_REDUCTION_TERMS || ns("array argument of $(prod(dims)) elements")
  local tmp = Int[(g.tmpSlot -= 1) for _ in dims]
  local body = toIR(g, e, _with(ctx; elem = tmp))
  local elems = IR[]
  for t in Iterators.product((1:d for d in dims)...)   #= column-major: the first index fastest =#
    push!(elems, _substSlots(body, Dict(tmp[k] => t[k] for k in eachindex(tmp))))
  end
  return Op(:arrayval, IR[Lit(Tuple(dims)), elems...])
end

#= sum/product/min/max of an array of known (small) size: the terms written out. =#
const MAX_REDUCTION_TERMS = 1000

function _reductionIR(g::Gen, fname::String, @nospecialize(arg::F.Expression), ctx::Ctx)::IR
  local ty = F.typeOf(arg)
  (F.isArray(ty) && F.hasKnownSize(ty)) || ns("$(fname) of " * _str(arg))
  local dims = [F.size(d) for d in F.arrayDims(ty)]
  local n = prod(dims)
  n <= MAX_REDUCTION_TERMS || ns("$(fname) over $(n) elements")
  n == 0 && return Lit(fname == "product" ? 1.0 : fname == "sum" ? 0.0 : ns("$(fname) of an empty array"))
  #= the element at temporary slots -1, -2, ... then each element's subscripts substituted =#
  local tmp = collect(-1:-1:-length(dims))
  local body = toIR(g, arg, _with(ctx; elem = tmp))
  local terms = IR[]
  for t in Iterators.product(reverse([1:d for d in dims])...)
    local vals = reverse(collect(t))
    push!(terms, _substSlots(body, Dict(tmp[k] => vals[k] for k in eachindex(tmp))))
  end
  local op = fname == "sum" ? :+ : fname == "product" ? :* : Symbol(fname)
  local acc = terms[1]
  for t in terms[2:end]
    acc = Op(op, IR[acc, t])
  end
  return acc
end

#= sum(e for i in r, j in s), product, min, max: the terms for every iterator value written out. =#
function _iterReductionIR(g::Gen, call, ctx::Ctx, text::String)::IR
  local fname = F.AbsynUtil.pathString(F.name(call.fn))
  fname in ("sum", "product", "min", "max") || ns("reduction " * text)
  local iters = copy(ctx.iters)
  local slots = Int[]; local ranges = Vector{Int}[]
  for (node, range) in call.iters
    local k = (g.tmpSlot -= 1)
    iters[F.name(node)] = k
    push!(slots, k)
    push!(ranges, _loopValues(range))
  end
  local n = prod(length.(ranges))
  n <= MAX_REDUCTION_TERMS || ns("reduction over $(n) values: " * text)
  n == 0 && return Lit(fname == "product" ? 1.0 : fname == "sum" ? 0.0 : ns("empty reduction " * text))
  local body = toIR(g, call.exp, Ctx(iters, ctx.elem, ctx.nrel, ctx.noEvent, ctx.nsam))
  local terms = IR[]
  for t in Iterators.product(reverse(ranges)...)
    local vals = reverse(collect(t))
    push!(terms, _substSlots(body, Dict(slots[k] => vals[k] for k in eachindex(slots))))
  end
  local op = fname == "sum" ? :+ : fname == "product" ? :* : Symbol(fname)
  return foldl((a, b) -> Op(op, IR[a, b]), terms)
end

#= Matrix and vector products in an element-wise context: the inner dimension written out. =#
function _productIR(g::Gen, e::F.BINARY_EXPRESSION, ctx::Ctx)::IR
  local O = F.Op
  local op = e.operator.op
  local t1 = F.typeOf(e.exp1); local t2 = F.typeOf(e.exp2)
  (F.hasKnownSize(t1) && F.hasKnownSize(t2)) || ns("product of unknown size")
  local d1 = [F.size(d) for d in F.arrayDims(t1)]; local d2 = [F.size(d) for d in F.arrayDims(t2)]
  local k = (g.tmpSlot -= 1)
  local (e1, e2, inner) = if op == O.SCALAR_PRODUCT
    (Int[k], Int[k], d1[1])
  elseif op == O.MUL_MATRIX_VECTOR
    length(ctx.elem) == 1 || ns("matrix-vector product rank")
    ([ctx.elem[1], k], Int[k], d1[2])
  elseif op == O.MUL_VECTOR_MATRIX
    length(ctx.elem) == 1 || ns("vector-matrix product rank")
    (Int[k], [k, ctx.elem[1]], d1[1])
  elseif op == O.MATRIX_PRODUCT
    length(ctx.elem) == 2 || ns("matrix product rank")
    ([ctx.elem[1], k], [k, ctx.elem[2]], d1[2])
  else
    ns("operator $(op)")
  end
  inner <= MAX_REDUCTION_TERMS || ns("product with inner dimension $(inner)")
  inner == 0 && return Lit(0.0)
  local a = toIR(g, e.exp1, _with(ctx; elem = e1))
  local b = toIR(g, e.exp2, _with(ctx; elem = e2))
  local terms = IR[_substSlots(Op(:*, IR[a, b]), Dict(k => i)) for i in 1:inner]
  return foldl((x, y) -> Op(:+, IR[x, y]), terms)
end

function _substSlots(x::IR, vals::Dict{Int, Int})::IR
  if x isa Slot
    return haskey(vals, x.k) ? Lit(vals[x.k]) : x
  elseif x isa Ref
    return Ref(x.name, IR[_substSlots(s, vals) for s in x.subs], x.der)
  elseif x isa Pre
    return Pre(_substSlots(x.ref, vals))
  elseif x isa Rel
    return Rel(x.op, _substSlots(x.lhs, vals), _substSlots(x.rhs, vals), x.id)
  elseif x isa EvFn
    return EvFn(x.kind, _substSlots(x.arg, vals), x.id)
  elseif x isa Op
    local args = IR[_substSlots(a, vals) for a in x.args]
    #= {a, b, c}[2] is b =#
    if x.op == :index && length(args) == 2 && args[1] isa Op && args[1].op == :vect && args[2] isa Lit
      return args[1].args[args[2].v]
    end
    return Op(x.op, args)
  end
  return x
end

function _crefIR(g::Gen, e::F.CREF_EXPRESSION, ctx::Ctx, der::Bool)::IR
  local cr = e.cref
  F.isvariant(cr, F.COMPONENT_REF_CREF) || ns("cref " * _str(e))
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
    free <= length(ctx.elem) || ns("array operand $(_str(e)) outside an element-wise context")
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
          ns("subscript " * _str(s))
        end
      end
    end
  end
  (free == 0 || free == length(ctx.elem)) || ns("operand rank of $(_str(e))")
  der && info.kind != :state && ns("der of non-state $(name)")
  return Ref(name, subs, der)
end

#= ---------------------------------------------------------------- equation classes =#

function _loopValues(@nospecialize(r::F.Expression))::Vector{Int}
  if r isa F.RANGE_EXPRESSION
    local start = _literalValue(_evalFrontend(r.start))
    local stop = _literalValue(_evalFrontend(r.stop))
    local step = r.step === nothing ? 1 : _literalValue(_evalFrontend(F.Util.getOption(r.step)))
    (start isa Int && stop isa Int && step isa Int) && return collect(start:step:stop)
  end
  local vals = _literalValue(_evalFrontend(r))
  (vals isa AbstractVector && all(v -> v isa Int, vals)) || ns("for range " * _str(r))
  return Vector{Int}(vals)
end

#= Domains: one column per instance, one row per slot (the first slot varies slowest). =#
const NO_SLOTS = Matrix{Int}(undef, 0, 1)

function _crossDomain(d::Matrix{Int}, vals::Vector{Int})::Matrix{Int}
  local out = Matrix{Int}(undef, size(d, 1) + 1, size(d, 2) * length(vals))
  local c = 0
  for j in 1:size(d, 2), v in vals
    c += 1
    out[1:(end - 1), c] = d[:, j]
    out[end, c] = v
  end
  return out
end

_crossDomain(d::Matrix{Int}, ranges::Vector{Vector{Int}}) = foldl(_crossDomain, ranges; init = d)

#= sample() of an equation as the sample slot of the when wi that schedules its instants. =#
function _sampleSlots(x::IR, wi::Int)::IR
  x isa Op && x.op == :sample && return Op(:sampleof, IR[Lit(wi), x.args[3]])
  x isa Op && return Op(x.op, IR[_sampleSlots(a, wi) for a in x.args])
  x isa Rel && return Rel(x.op, _sampleSlots(x.lhs, wi), _sampleSlots(x.rhs, wi), x.id)
  x isa EvFn && return EvFn(x.kind, _sampleSlots(x.arg, wi), x.id)
  return x
end

#= The first sample slot of when wi: the slots are numbered when by when, instance by instance. =#
_whenSampleBase(g::Gen, wi::Int) = sum((g.whens[k].nsample * size(g.whens[k].domain, 2) for k in 1:(wi - 1)); init = 0)

#= Whether x reads only literals, iterators and parameters (no variable, no time). =#
function _parametersOnly(g::Gen, x::IR)::Bool
  (x isa Lit || x isa Slot) && return true
  x isa Ref && return g.vars[x.name].kind == :param && !x.der && all(s -> _parametersOnly(g, s), x.subs)
  x isa Op && return !(x.op in (:sample, :initial)) && all(a -> _parametersOnly(g, a), x.args)
  return false
end

#= The value of an expression that depends only on slots and parameters, at a domain column. =#
function _evalStatic(g::Gen, x::IR, d::Matrix{Int}, col::Int)
  if x isa Lit
    return x.v
  elseif x isa Slot
    return d[x.k, col]
  elseif x isa Ref
    local info = g.vars[x.name]
    (info.kind == :param && !x.der) || ns("condition on variable $(x.name)")
    push!(g.structural, x.name)
    local v = _paramValue!(g, x.name)
    isempty(x.subs) && return v
    return v[(_evalStatic(g, s, d, col) for s in x.subs)...]
  elseif x isa Op
    local a = [_evalStatic(g, y, d, col) for y in x.args]
    x.op == :neg && return -a[1]
    x.op == :! && return !a[1]
    x.op == :&& && return a[1] && a[2]
    x.op == :|| && return a[1] || a[2]
    x.op == :if && return a[1] ? a[2] : a[3]
    x.op == :index && return a[1][a[2:end]...]
    x.op == :vect && return a
    (x.op in (:initial, :sample) || !isdefined(Base, x.op)) && ns("condition $(x.op)() in an if-equation")
    return getfield(Base, x.op)(a...)
  end
  ns("condition not static")
end

function _addEquation!(g::Gen, @nospecialize(eq::F.Equation), iterNames::Vector{String}, idom::Matrix{Int};
                       initial::Bool = false)
  size(idom, 2) == 0 && return
  local iters = Dict(n => k for (k, n) in enumerate(iterNames))
  local nIter = length(iterNames)
  if F.isvariant(eq, F.EQUATION_FOR)
    local values = _loopValues(F.Util.getOption(eq.range))
    isempty(values) && return
    local dom = _crossDomain(idom, values)
    for b in eq.body
      _addEquation!(g, b, [iterNames; F.name(eq.iterator)], dom; initial = initial)
    end
    return
  end
  if F.isvariant(eq, F.EQUATION_IF)
    #= Conditions on loop indices and parameters only: each instance takes its branch. =#
    local ctx = Ctx(iters, Int[], Base.RefValue(0), true)
    local conds = IR[]
    for br in eq.branches
      F.isvariant(br, F.EQUATION_BRANCH) || ns("if-equation branch")
      push!(conds, toIR(g, br.condition, ctx))
    end
    all(c -> _isStatic(g, c), conds) || return _addDynamicIf!(g, eq, iterNames, idom; initial = initial)
    local groups = [Int[] for _ in eq.branches]
    for j in 1:size(idom, 2)
      local k = findfirst(c -> _evalStatic(g, c, idom, j) == true, conds)
      k === nothing || push!(groups[k], j)
    end
    for (br, cols) in zip(eq.branches, groups)
      isempty(cols) && continue
      for b in br.body
        _addEquation!(g, b, iterNames, idom[:, cols]; initial = initial)
      end
    end
    return
  end
  if F.isvariant(eq, F.EQUATION_NORETCALL)
    #= a call without a result (Modelica.Fluid.Utilities.checkBoundary): run with the asserts =#
    local cctx = Ctx(iters, Int[], Base.RefValue(0), true)
    push!(initial ? g.initAsserts : g.asserts,
          AssertClass(Op(:seq, IR[toIR(g, eq.exp, cctx), Lit(true)]), false, nIter, idom, _str(eq), "", Lit("")))
    return
  end
  if initial && F.isvariant(eq, F.EQUATION_ASSERT)
    #= checked once, after the initialization =#
    local actx = Ctx(iters, Int[], Base.RefValue(0), true)
    push!(g.initAsserts, AssertClass(toIR(g, eq.condition, actx), occursin("warning", _str(eq.level)),
                                     nIter, idom, _str(eq.condition), _str(eq.message), _messageIR(g, eq.message, actx)))
    return
  end
  if initial
    (F.isvariant(eq, F.EQUATION_EQUALITY) || F.isvariant(eq, F.EQUATION_ARRAY_EQUALITY)) ||
      ns("initial equation " * first(_str(eq), 200))
    local edims = F.isArray(eq.ty) ? [F.size(d) for d in F.arrayDims(eq.ty)] : Int[]
    #= relations of initial equations are evaluated at the start (no events) =#
    local ctx = Ctx(iters, collect((nIter + 1):(nIter + length(edims))), Base.RefValue(0), true)
    local dom = _crossDomain(idom, [collect(1:d) for d in edims])
    size(dom, 2) == 0 && return
    local ilhs = toIR(g, eq.lhs, ctx); local irhs = toIR(g, eq.rhs, ctx)
    #= d = expr for a discrete variable (CDL Discrete: y = y_start): the initial value, assigned
       after the initialization's solve like an initial algorithm's (it is no unknown of it) =#
    if isempty(edims) && ilhs isa Ref && !ilhs.der && g.vars[ilhs.name].kind == :discrete &&
       !any(r -> r.name == ilhs.name, accesses!(Ref[], irhs))
      push!(g.initAlgs, AlgClass(Stmt[SAssign(ilhs, irhs)], nIter, idom, 0, 0, _str(eq)))
      return
    end
    push!(g.initClasses, EqClass(ilhs, irhs, size(dom, 1), dom, 0, _str(eq)))
    return
  end
  if F.isvariant(eq, F.EQUATION_WHEN)
    _addWhen!(g, eq, iterNames, idom)
    return
  end
  if F.isvariant(eq, F.EQUATION_ASSERT)
    local ctx = Ctx(iters, Int[], Base.RefValue(0), true)
    push!(g.asserts, AssertClass(toIR(g, eq.condition, ctx), occursin("warning", _str(eq.level)),
                                 nIter, idom, _str(eq.condition), _str(eq.message), _messageIR(g, eq.message, ctx)))
    return
  end
  if F.isvariant(eq, F.EQUATION_EQUALITY) && eq.lhs isa F.TUPLE_EXPRESSION
    #= (a, b) = f(x): an algorithm node assigning each target its output =#
    local ctx = Ctx(iters, Int[])
    local call = toIR(g, eq.rhs, ctx)
    ctx.nsam[] == 0 || ns("sample outside a when-condition")
    push!(g.algs, AlgClass(_tupleAssignments(g, eq.lhs, call, ctx), nIter, idom, ctx.nrel[], 0, _str(eq)))
    return
  end
  (F.isvariant(eq, F.EQUATION_EQUALITY) || F.isvariant(eq, F.EQUATION_ARRAY_EQUALITY)) ||
    ns("equation " * first(_str(eq), 200))
  local edims = F.isArray(eq.ty) ? [F.size(d) for d in F.arrayDims(eq.ty)] : Int[]
  local ctx = Ctx(iters, collect((nIter + 1):(nIter + length(edims))))
  local lhs = toIR(g, eq.lhs, ctx)
  local rhs = toIR(g, eq.rhs, ctx)
  if ctx.nsam[] > 0
    #= sampleTrigger = sample(t0, samplePeriod) (CDL Discrete): true at the instants only. A
       when without a body schedules the instants; the equation reads its sample slots. =#
    isempty(edims) || ns("sample outside a when-condition, in an array equation")
    local samples = IR[]
    local collect! = function (x::IR)
      x isa Op && x.op == :sample && push!(samples, x)
      x isa Op && foreach(collect!, x.args)
      x isa Rel && (collect!(x.lhs); collect!(x.rhs))
      x isa EvFn && collect!(x.arg)
    end
    collect!(lhs); collect!(rhs)
    push!(g.whens, WhenClass(length(samples) == 1 ? samples[1] : Op(:||, samples), Tuple{Symbol, Ref, IR}[], [1],
                             nIter, idom, 0, ctx.nsam[], "sample() of " * _str(eq)))
    local wi = length(g.whens)
    lhs = _sampleSlots(lhs, wi); rhs = _sampleSlots(rhs, wi)
  end
  local dom = _crossDomain(idom, [collect(1:d) for d in edims])
  size(dom, 2) == 0 && return
  push!(g.classes, EqClass(lhs, rhs, size(dom, 1), dom, ctx.nrel[], _str(eq)))
end

#= Whether a condition depends on loop indices and parameters only. =#
_isStatic(g::Gen, x::IR)::Bool =
  x isa Lit || x isa Slot ||
  (x isa Ref && !x.der && g.vars[x.name].kind == :param && all(s -> _isStatic(g, s), x.subs)) ||
  (x isa Op && !(x.op in (:initial, :sample)) && all(a -> _isStatic(g, a), x.args))

#= An if-equation on variable conditions whose branches (with an else) each give the same
   variables as v = e: one equation per variable, v = if c1 then e1 elseif ... else en (as
   omc's if-equation to if-expression); its relations generate events. =#
function _addDynamicIf!(g::Gen, @nospecialize(eq::F.Equation), iterNames::Vector{String}, idom::Matrix{Int};
                        initial::Bool = false)
  local iters = Dict(n => k for (k, n) in enumerate(iterNames))
  local nIter = length(iterNames)
  local branches = collect(eq.branches)
  local last = branches[end].condition
  (last isa F.BOOLEAN_EXPRESSION && last.value) || ns("if-equation on a variable condition without else")
  local keyed = Vector{Dict{String, Any}}()
  for br in branches
    local d = Dict{String, Any}()
    for b in br.body
      (F.isvariant(b, F.EQUATION_EQUALITY) || F.isvariant(b, F.EQUATION_ARRAY_EQUALITY)) && b.lhs isa F.CREF_EXPRESSION ||
        ns("if-equation on a variable condition: branch equation " * first(_str(b), 120))
      local k = _str(b.lhs)
      haskey(d, k) && ns("if-equation on a variable condition: two equations for " * k)
      d[k] = b
    end
    push!(keyed, d)
  end
  all(d -> Set(keys(d)) == Set(keys(keyed[1])), keyed) ||
    ns("if-equation on a variable condition: branches give different variables")
  for k in sort!(collect(keys(keyed[1])))
    local first = keyed[1][k]
    local edims = F.isArray(first.ty) ? [F.size(d) for d in F.arrayDims(first.ty)] : Int[]
    local ctx = Ctx(iters, collect((nIter + 1):(nIter + length(edims))), Base.RefValue(0), initial)
    local lhs = toIR(g, first.lhs, ctx)
    local rhs = toIR(g, keyed[end][k].rhs, ctx)
    for i in (length(branches) - 1):-1:1
      rhs = Op(:if, IR[toIR(g, branches[i].condition, _with(ctx; elem = Int[])), toIR(g, keyed[i][k].rhs, ctx), rhs])
    end
    ctx.nsam[] == 0 || ns("sample outside a when-condition")
    local dom = _crossDomain(idom, [collect(1:d) for d in edims])
    size(dom, 2) == 0 && continue
    push!(initial ? g.initClasses : g.classes,
          EqClass(lhs, rhs, size(dom, 1), dom, initial ? 0 : ctx.nrel[], string(k, " = if ... (", _str(eq.branches[1].condition), ")")))
  end
end

#= (a, _, c) = call: the call once, into a temporary, and an assignment of each
   (non-wildcard) target from its output (the call was made per target, and per element of
   an array target). =#
function _tupleAssignments(g::Gen, @nospecialize(lhs), call::IR, ctx::Ctx)::Vector{Stmt}
  local out = Stmt[]
  local res = Slot(g.tmpSlot -= 1)
  for (i, e) in enumerate(lhs.elements)
    (e isa F.CREF_EXPRESSION && F.isvariant(e.cref, F.COMPONENT_REF_WILD)) && continue
    e isa F.CREF_EXPRESSION || ns("tuple target " * _str(e))
    if F.isArray(F.typeOf(e))
      #= a whole array ((r, state) = random(pre(state)): Buildings.Occupants): element by
         element from that output =#
      local name = _crefName(e.cref)
      local info = get(g.vars, name, nothing)
      (info !== nothing && info.kind in (:alg, :discrete) && !isempty(info.dims) &&
       all(p -> isempty(collect(p.subscripts)), _partsRootFirst(e.cref))) || ns("tuple target " * _str(e))
      for t in Iterators.product((1:d for d in info.dims)...)
        local subs = IR[Lit(k) for k in t]
        push!(out, SAssign(Ref(name, subs, false), Op(:index, IR[Op(:index, IR[res, Lit(i)]), subs...])))
      end
      continue
    end
    local target = toIR(g, e, ctx)
    (target isa Ref && g.vars[target.name].kind in (:alg, :discrete)) || ns("tuple target " * _str(e))
    push!(out, SAssign(target, Op(:index, IR[res, Lit(i)])))
  end
  isempty(out) || pushfirst!(out, SCall(Op(:setlocal, IR[res, call])))
  return out
end

function _addWhen!(g::Gen, @nospecialize(eq::F.Equation), iterNames::Vector{String}, idom::Matrix{Int})
  local ctx = Ctx(Dict(n => k for (k, n) in enumerate(iterNames)), Int[])
  local conds = IR[]
  local body = Tuple{Symbol, Ref, IR}[]
  local starts = Int[]
  local noTarget = Ref("", IR[], false)
  for br in eq.branches
    F.isvariant(br, F.EQUATION_BRANCH) || ns("when branch")
    push!(conds, _whenCondition(g, br.condition, ctx))
    push!(starts, length(body) + 1)
    for b in br.body
      if F.isvariant(b, F.EQUATION_REINIT)
        local target = toIR(g, b.cref, ctx)
        (target isa Ref && g.vars[target.name].kind == :state) || ns("reinit of a non-state")
        push!(body, (:reinit, target, toIR(g, b.reinitExp, ctx)))
      elseif F.isvariant(b, F.EQUATION_EQUALITY) && b.lhs isa F.TUPLE_EXPRESSION
        for st in _tupleAssignments(g, b.lhs, toIR(g, b.rhs, ctx), ctx)
          st isa SCall && (push!(body, (:call, noTarget, st.call)); continue)
          g.vars[st.target.name].kind == :discrete || ns("when assigns a non-discrete variable")
          push!(body, (:assign, st.target, st.value))
        end
      elseif F.isvariant(b, F.EQUATION_EQUALITY) && b.lhs isa F.CREF_EXPRESSION && !F.isArray(b.ty)
        local target = toIR(g, b.lhs, ctx)
        (target isa Ref && g.vars[target.name].kind == :discrete) || ns("when assigns a non-discrete variable")
        push!(body, (:assign, target, toIR(g, b.rhs, ctx)))
      elseif F.isvariant(b, F.EQUATION_ASSERT)
        #= checked when the body runs =#
        push!(g.assertTexts, (_str(b.condition), _str(b.message), occursin("warning", _str(b.level))))
        local actx = _with(ctx; noEvent = true)
        push!(body, (:call, noTarget, Op(:assertcall, IR[toIR(g, b.condition, actx), _messageIR(g, b.message, actx),
                                                            Lit(length(g.assertTexts))])))
      elseif F.isvariant(b, F.EQUATION_NORETCALL)
        push!(body, (:call, noTarget, toIR(g, b.exp, _with(ctx; noEvent = true))))
      else
        ns("when body equation " * first(_str(b), 120))
      end
  end
  end
  local cond = length(conds) == 1 ? conds[1] : Op(:elsewhen, conds)
  push!(g.whens, WhenClass(cond, body, starts, length(iterNames), idom, ctx.nrel[], ctx.nsam[], _str(eq)))
end

#= initial() activates a when at the initialization only as its condition (MLS 8.6, as
   OpenModelica): inside an expression (initial() or x > 0.6) it is false. =#
_topLevelInitial(x::IR) = (x isa Op && x.op == :initial) ? x : _noInitial(x)

#= A when-condition: a Boolean expression, or a vector of them (when {c1, c2}: Op(:vect),
   each element with its own condition slot and edge; initial() counts as an element). =#
function _whenCondition(g::Gen, @nospecialize(c::F.Expression), ctx::Ctx)::IR
  c isa F.ARRAY_EXPRESSION || return _topLevelInitial(toIR(g, c, ctx))
  return Op(:vect, IR[_topLevelInitial(toIR(g, el, ctx)) for el in c.elements])
end

#= The condition elements of a when (one for a scalar condition). =#
_condElements(c::IR)::Vector{IR} = (c isa Op && c.op == :elsewhen) ? IR[e for b in c.args for e in _condElements(b)] :
                                   (c isa Op && c.op == :vect) ? c.args : IR[c]
_noInitial(x::IR) = !(x isa Op) ? x : x.op == :initial ? Lit(false) : Op(x.op, IR[_noInitial(a) for a in x.args])

#= An algorithm section: statements to IR. A vectorized algorithm (one for-loop over a $
   iterator, from a component array) keeps the loop as the class domain. =#
function _addAlgorithm!(g::Gen, alg)
  local stmts = collect(alg.statements)
  local iterNames = String[]
  local idom = NO_SLOTS
  while length(stmts) == 1 && F.isvariant(stmts[1], F.ALG_FOR) && startswith(F.name(stmts[1].iterator), "\$")
    local values = _loopValues(F.Util.getOption(stmts[1].range))
    isempty(values) && return
    push!(iterNames, F.name(stmts[1].iterator))
    idom = _crossDomain(idom, values)
    stmts = collect(stmts[1].body)
  end
  local ctx = Ctx(Dict(n => k for (k, n) in enumerate(iterNames)), Int[])
  local nwhen = Base.RefValue(0)
  local body = Stmt[_stmtIR(g, st, ctx, nwhen) for st in stmts]
  ctx.nsam[] == 0 || ns("sample in an algorithm")
  push!(g.algs, AlgClass(body, length(iterNames), idom, ctx.nrel[], nwhen[], _str(alg.statements[1])))
end

#= An initial algorithm: run once at the initialization, after the free states are solved and
   before the first event iteration, without events (its relations and event-generating
   functions literal). It may assign discrete variables (and assert); initial equations may not
   read what it assigns. =#
function _addInitialAlgorithm!(g::Gen, alg)
  local stmts = collect(alg.statements)
  local iterNames = String[]
  local idom = NO_SLOTS
  while length(stmts) == 1 && F.isvariant(stmts[1], F.ALG_FOR) && startswith(F.name(stmts[1].iterator), "\$")
    local values = _loopValues(F.Util.getOption(stmts[1].range))
    isempty(values) && return
    push!(iterNames, F.name(stmts[1].iterator))
    idom = _crossDomain(idom, values)
    stmts = collect(stmts[1].body)
  end
  local ctx = Ctx(Dict(n => k for (k, n) in enumerate(iterNames)), Int[], Base.RefValue(0), true)
  local nwhen = Base.RefValue(0)
  local body = Stmt[_stmtIR(g, st, ctx, nwhen) for st in stmts]
  (nwhen[] == 0 && ctx.nsam[] == 0) || ns("when or sample in an initial algorithm")
  _initAlgTargets(g, body)
  push!(g.initAlgs, AlgClass(body, length(iterNames), idom, 0, 0, _str(alg.statements[1])))
end

function _initAlgTargets(g::Gen, stmts::Vector{Stmt})
  for st in stmts
    if st isa SAssign
      (g.vars[st.target.name].kind == :discrete || st.target.name in g.freeParams) ||
        ns("initial algorithm assigns $(st.target.name), not a discrete variable or a parameter with fixed = false")
    elseif st isa SIf
      foreach(b -> _initAlgTargets(g, b[2]), st.branches)
    elseif st isa SFor || st isa SWhile
      _initAlgTargets(g, st.body)
    end
  end
end

function _stmtIR(g::Gen, @nospecialize(st), ctx::Ctx, nwhen::Base.RefValue{Int})::Stmt
  if F.isvariant(st, F.ALG_NORETCALL)
    return SCall(toIR(g, st.exp, ctx))
  end
  if F.isvariant(st, F.ALG_ASSIGNMENT) && st.lhs isa F.TUPLE_EXPRESSION
    return SIf([(Lit(true), _tupleAssignments(g, st.lhs, toIR(g, st.rhs, ctx), ctx))])
  end
  if F.isvariant(st, F.ALG_ASSIGNMENT)
    (st.lhs isa F.CREF_EXPRESSION && !F.isArray(st.ty)) || ns("assignment " * _str(st))
    local target = toIR(g, st.lhs, ctx)
    #= a parameter with fixed = false: an initial algorithm computes it (_initAlgTargets) =#
    (target isa Ref && (g.vars[target.name].kind in (:alg, :discrete) || target.name in g.freeParams)) ||
      ns("assignment to " * _str(st.lhs))
    return SAssign(target, toIR(g, st.rhs, ctx))
  elseif F.isvariant(st, F.ALG_IF)
    return SIf([(toIR(g, c, ctx), Stmt[_stmtIR(g, b, ctx, nwhen) for b in body]) for (c, body) in st.branches])
  elseif F.isvariant(st, F.ALG_FOR)
    local values = _loopValues(F.Util.getOption(st.range))
    local k = (g.tmpSlot -= 1)
    local inner = Ctx(merge(ctx.iters, Dict(F.name(st.iterator) => k)), ctx.elem, ctx.nrel, ctx.noEvent, ctx.nsam)
    return SFor(k, values, Stmt[_stmtIR(g, b, inner, nwhen) for b in st.body])
  elseif F.isvariant(st, F.ALG_WHILE)
    return SWhile(toIR(g, st.condition, _with(ctx; noEvent = true)), Stmt[_stmtIR(g, b, ctx, nwhen) for b in st.body])
  elseif F.isvariant(st, F.ALG_WHEN)
    local branches = Tuple{IR, Vector{Stmt}}[]
    local ids = Vector{Int}[]
    for (c, body) in st.branches
      local cond = _whenCondition(g, c, ctx)
      push!(ids, [(nwhen[] += 1) for _ in _condElements(cond)])
      push!(branches, (cond, Stmt[_stmtIR(g, b, ctx, nwhen) for b in body]))
    end
    return SWhen(branches, ids)
  elseif F.isvariant(st, F.ALG_ASSERT)
    push!(g.assertTexts, (_str(st.condition), _str(st.message), occursin("warning", _str(st.level))))
    return SAssert(toIR(g, st.condition, _with(ctx; noEvent = true)), length(g.assertTexts), _messageIR(g, st.message, ctx))
  elseif F.isvariant(st, F.ALG_BREAK)
    return SBreak()
  end
  ns("statement " * _str(st))
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
  elseif x isa EvFn
    return EvFn(x.kind, _abstracted(x.arg, vals), x.id)
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
  class::Vector{Int}     #= > 0 an equation class, < 0 an algorithm (-index) =#
  col::Vector{Int}
  uptr::Vector{Int}
  uid::Vector{Int}
  upos::Vector{Int}
  sptr::Vector{Int}
  sid::Vector{Int}
  outs::Vector{Vector{Int}}   #= unknowns an algorithm instance assigns (empty for equations) =#
  nEq::Int
end

#= The unknown id of a variable element, or 0 (parameters, states read, when-assigned
   discretes). =#
function _unknownOf(g::Gen, info::VarInfo, off::Int, der::Bool)::Int
  info.kind == :alg && return g.nStates + off
  info.kind == :discrete && return get(g.discUnknown, off, 0)
  info.kind == :state && der && return off
  return 0
end

#= The elements an algorithm instance reads and assigns, walking its statements (for-loops
   with their static ranges). =#
function _algAccesses!(g::Gen, stmts::Vector{Stmt}, env::Dict{Int, Int}, d::Matrix{Int}, col::Int,
                       outs::Set{Int}, ins::Set{Int}, sids::Set{Int})
  local readIR! = function (x::IR)
    if x isa Ref
      local info = g.vars[x.name]
      if info.kind != :param
        local idx = Int[_evalIntEnv(g, s, env, d, col) for s in x.subs]
        local off = info.offset + _elementOffset(info, idx, x.name) + 1
        local u = _unknownOf(g, info, off, x.der)
        u > 0 && push!(ins, u)
        info.kind == :state && !x.der && push!(sids, off)
      end
    elseif x isa Op
      foreach(readIR!, x.args)
    elseif x isa Rel
      readIR!(x.lhs); readIR!(x.rhs)
    elseif x isa EvFn
      readIR!(x.arg)
    end
  end
  for st in stmts
    if st isa SAssign
      local info = g.vars[st.target.name]
      local idx = Int[_evalIntEnv(g, s, env, d, col) for s in st.target.subs]
      local off = info.offset + _elementOffset(info, idx, st.target.name) + 1
      local u = _unknownOf(g, info, off, false)
      u > 0 || ns("algorithm assigns $(st.target.name), which an equation or when-equation defines")
      push!(outs, u)
      readIR!(st.value)
    elseif st isa SIf || st isa SWhen
      for (c, body) in st.branches
        readIR!(c)
        _algAccesses!(g, body, env, d, col, outs, ins, sids)
      end
    elseif st isa SFor
      for v in st.range
        env[st.slot] = v
        _algAccesses!(g, st.body, env, d, col, outs, ins, sids)
      end
      delete!(env, st.slot)
    elseif st isa SWhile
      readIR!(st.cond)
      _algAccesses!(g, st.body, env, d, col, outs, ins, sids)
    elseif st isa SAssert
      readIR!(st.cond)
    elseif st isa SCall
      readIR!(st.call)
    end
  end
end

function _evalIntEnv(g::Gen, x::IR, env::Dict{Int, Int}, d::Matrix{Int}, col::Int)::Int
  x isa Slot && x.k < 0 && return get(env, x.k) do
    ns("subscript with a loop variable outside its loop")
  end
  x isa Op && return _evalInt(g, Op(x.op, IR[Lit(_evalIntEnv(g, a, env, d, col)) for a in x.args]), d, col)
  return _evalInt(g, x, d, col)
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
        info.kind == :param && continue
        empty!(idx)
        for s in r.subs
          push!(idx, _evalInt(g, s, c.domain, j))
        end
        local off = info.offset + _elementOffset(info, idx, r.name) + 1
        if info.kind == :discrete
          local du = get(g.discUnknown, off, 0)
          du > 0 && (push!(uid, du); push!(upos, pos))
        elseif info.kind == :alg
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
  local nEq = length(class)
  local outs = [Int[] for _ in 1:nEq]
  for (ai, a) in enumerate(g.algs)
    for j in 1:size(a.domain, 2)
      local o = Set{Int}(); local ins = Set{Int}(); local sids = Set{Int}()
      _algAccesses!(g, a.body, Dict{Int, Int}(), a.domain, j, o, ins, sids)
      push!(class, -ai); push!(col, j)
      for u in sort!(collect(setdiff(ins, o)))
        push!(uid, u); push!(upos, 0)
      end
      append!(sid, sort!(collect(sids)))
      push!(uptr, length(uid) + 1); push!(sptr, length(sid) + 1)
      push!(outs, sort!(collect(o)))
    end
  end
  return Elements(class, col, uptr, uid, upos, sptr, sid, outs, nEq)
end

#= Maximum matching instances <-> unknowns (greedy, then augmenting paths). =#
function _match(el::Elements, nU::Int)::Vector{Int}
  local N = el.nEq
  local matchRow = zeros(Int, length(el.class))
  local matchCol = zeros(Int, nU)
  #= the unknowns algorithms assign are theirs =#
  local nOut = 0
  for r in (N + 1):length(el.class), u in el.outs[r]
    matchCol[u] == 0 || ns("two algorithms assign one variable")
    matchCol[u] = r
    nOut += 1
  end
  N == nU - nOut || ns("$(N) equations for $(nU - nOut) unknowns (and $(nOut) assigned by algorithms)")
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
  for r in (N + 1):length(el.class), u in el.outs[r]
    visited[u] = -1   #= never part of an augmenting path =#
  end
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
      (visited[c] == stamp || visited[c] == -1) && continue
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
  for r in 1:el.nEq
    producer[matchRow[r]] = r
  end
  for r in (el.nEq + 1):N, u in el.outs[r]
    producer[u] = r
  end
  local pos = zeros(Int, N)
  for r in 1:el.nEq
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
      q == 0 && ns("an unknown no equation or algorithm defines")
      (q in seen || q == r) && continue
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

#= The states every equation instance depends on, through the algebraic unknowns it reads
   (in schedule order); nothing when that would be large. =#
function _stateDeps(el::Elements, matchRow, batches, producer)
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
  return deps
end

#= State dependencies of every state derivative (the Jacobian's rows). =#
function _jacobianPattern(el::Elements, matchRow, deps, nStates::Int)
  deps === nothing && return nothing
  local N = length(el.class)
  local I = Int[]; local J = Int[]
  for r in 1:N
    local u = matchRow[r]
    (1 <= u <= nStates) || continue
    push!(I, u); push!(J, u)
    for c in deps[r]
      push!(I, u); push!(J, c)
    end
  end
  return (I, J)
end

#= ---------------------------------------------------------------- initialization =#

#= The initial equations as a square system: each instance matched to a state element that
   is not fixed and that it depends on (directly or through the algebraic unknowns); free
   states no initial equation reaches keep their start values (as omc). Returns the matched
   states, the residual code and the number of residuals. =#
function _initialization(g::Gen, deps, producer, fixedMask::Vector{Bool})
  isempty(g.initClasses) && return (Int[], Any[], 0, Tuple{String, Int}[])
  #= a free parameter element (name, column-major index) is unknown g.nStates + k =#
  local freeCols = Tuple{String, Int}[]
  local freeCol = Dict{Tuple{String, Int}, Int}()
  deps === nothing && ns("initialization of a model with a large dependency pattern")
  #= candidate free states per initial equation instance =#
  local rows = Vector{Vector{Int}}()
  local body = Any[]
  local base = 0
  for c in g.initClasses
    local acc = accesses!(accesses!(Ref[], c.lhs), c.rhs)
    for j in 1:size(c.domain, 2)
      local cand = Set{Int}()
      for r in acc
        local info = g.vars[r.name]
        if info.kind == :param
          r.name in g.freeParams || continue
          local pidx = Int[_evalInt(g, s, c.domain, j) for s in r.subs]
          local lin = isempty(info.dims) ? 1 : LinearIndices(Tuple(info.dims))[pidx...]
          local key = (r.name, lin)
          haskey(freeCol, key) || (push!(freeCols, key); freeCol[key] = length(freeCols))
          push!(cand, g.nStates + freeCol[key])
          continue
        end
        local idx = Int[_evalInt(g, s, c.domain, j) for s in r.subs]
        local off = info.offset + _elementOffset(info, idx, r.name) + 1
        if info.kind == :state && !r.der
          push!(cand, off)
        elseif info.kind == :discrete
          local du = get(g.discUnknown, off, 0)
          du > 0 && union!(cand, deps[producer[du]])
        else
          local u = info.kind == :alg ? g.nStates + off : off
          union!(cand, deps[producer[u]])
          r.der && push!(cand, off)
        end
      end
      push!(rows, sort!([k for k in cand if k > g.nStates || !fixedMask[k]]))
    end
    g.relBase = 0; g.relCount = 0
    local assign = :(res[$base + _col] = $(jl(g, c.lhs)) - $(jl(g, c.rhs)))
    push!(body, c.nslots == 0 ? Expr(:let, :(_col = 1), assign) : _loopCode(c.nslots, c.domain, collect(1:size(c.domain, 2)), assign))
    base += size(c.domain, 2)
  end
  #= matching rows -> free states (augmenting paths) =#
  local matchCol = Dict{Int, Int}()
  local matchRow = zeros(Int, length(rows))
  for r0 in eachindex(rows)
    local visited = Set{Int}()
    local augment
    augment = function (r)
      for c in rows[r]
        c in visited && continue
        push!(visited, c)
        if !haskey(matchCol, c) || augment(matchCol[c])
          matchCol[c] = r; matchRow[r] = c
          return true
        end
      end
      return false
    end
    augment(r0) || ns("initialization over-determined (an initial equation without a free state)")
  end
  return (matchRow, body, length(rows), freeCols)
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
    return x.k > 0 ? Symbol("s", x.k) : Symbol("t", -x.k)
  elseif x isa TimeIR
    return :t
  elseif x isa Ref
    local info = g.vars[x.name]
    local idx = Any[jl(g, s) for s in x.subs]
    if info.kind == :param
      if info.vtype == :other
        #= A String parameter (CDL Utilities.Assert's message): its value, as a literal. =#
        F.isString(F.arrayElementType(info.var.ty)) || ns("parameter $(x.name) of a type the array path does not pass")
        local sv = _evalBinding(info.var.binding, info.var, "parameter")
        (sv isa AbstractString || sv isa AbstractArray{<:AbstractString}) || ns("parameter $(x.name): no String value")
        return isempty(idx) ? sv : Expr(:ref, sv, idx...)
      end
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
    return :($(_rel)(z, p, mode, $k, $(jl(g, x.lhs)), $(jl(g, x.rhs)), $(x.op)))
  elseif x isa EvFn
    local k = :($(g.relBase) + (_col - 1) * $(g.relCount) + $(x.id))
    return :($(_evfn)(z, p, mode, $k, $(QuoteNode(x.kind)), $(jl(g, x.arg))))
  elseif x isa Op
    local args = Any[jl(g, a) for a in x.args]
    x.op == :neg && return :(-$(args[1]))
    x.op == :index && return Expr(:ref, args...)
    x.op == :vect && return Expr(:vect, args...)
    x.op == :if && return Expr(:if, args...)
    x.op == :&& && return Expr(:&&, args...)
    x.op == :|| && return Expr(:||, args...)
    x.op == :sample && return :(p.sampleActive[$(g.sampleBase) + (_col - 1) * $(g.sampleCount) + $(x.args[3].v)])
    if x.op == :sampleof
      local wi = x.args[1].v
      return :(p.sampleActive[$(_whenSampleBase(g, wi)) + (_col - 1) * $(g.whens[wi].nsample) + $(x.args[2].v)])
    end
    x.op == :initial && return :(p.initPhase[1])
    x.op == :String && return Expr(:call, _modelicaString, args...)
    x.op == :fcall && return Expr(:call, args...)
    #= a temporary's value (a tuple equation's call: _tupleAssignments) =#
    x.op == :setlocal && return Expr(:(=), args...)
    x.op == :assertcall && return :(($(args[1])) || _assertFailed(p, $(args[3]), ($(args[3]), _col), t, $(args[2])))
    x.op == :seq && return Expr(:block, args...)
    x.op == :arrayval && return length(args[1]) == 1 ? Expr(:vect, args[2:end]...) :
                                :(reshape($(Expr(:vect, args[2:end]...)), $(args[1]...)))
    x.op == :Int && return :(round(Int, $(args[1])))
    x.op == :Bool && return :(($(args[1])) != 0)
    return Expr(:call, x.op, args...)
  end
  error("IR")
end

#= Statement code of an algorithm; when-statements use wcond from `condBase` (nwhen per
   instance) and run their bodies only when fire (event iteration). =#
function _stmtCode(g::Gen, stmts::Vector{Stmt}, condBase::Int, nwhen::Int; inWhen::Bool = false)::Vector{Any}
  local out = Any[]
  for st in stmts
    if st isa SAssign
      local info = g.vars[st.target.name]
      local idx = Any[jl(g, s) for s in st.target.subs]
      if info.kind == :param
        #= a parameter with fixed = false, in an initial algorithm: its local (a view of
           p.values for an array; a scalar is written back after the algorithms) =#
        local sym = (_paramValue!(g, st.target.name); g.paramSyms[st.target.name])
        push!(out, isempty(idx) ? :($sym = $(jl(g, st.value))) : :($sym[$(idx...)] = $(jl(g, st.value))))
        continue
      end
      push!(out, info.kind == :discrete ? :(p.d[$(_linear(info, idx))] = Float64($(jl(g, st.value)))) :
                                          :(a[$(_linear(info, idx))] = $(jl(g, st.value))))
    elseif st isa SIf
      local ex = nothing
      for (c, body) in reverse(st.branches)
        local blk = Expr(:block, _stmtCode(g, body, condBase, nwhen; inWhen)...)
        if ex === nothing && c isa Lit && c.v == true
          ex = blk
        else
          ex = ex === nothing ? Expr(:if, jl(g, c), blk) : Expr(:if, jl(g, c), blk, ex)
        end
      end
      ex === nothing || push!(out, ex)
    elseif st isa SFor
      local sym = jl(g, Slot(st.slot))
      local r = st.range
      local rng = length(r) > 1 && all(diff(r) .== 1) ? :($(r[1]):$(r[end])) : r
      push!(out, :(for $sym in $rng; $(_stmtCode(g, st.body, condBase, nwhen; inWhen)...); end))
    elseif st isa SWhile
      push!(out, :(while $(jl(g, st.cond)); $(_stmtCode(g, st.body, condBase, nwhen; inWhen)...); end))
    elseif st isa SWhen
      #= per branch, a value and a slot per condition element; a branch fires when one of its
         elements became true (against the value at the start of the sweep) =#
      local cs = [[gensym("wc") for _ in _condElements(c)] for (c, _) in st.branches]
      local ks = [[:($condBase + (_col - 1) * $nwhen + $id) for id in ids] for ids in st.ids]
      for (i, (c, _)) in enumerate(st.branches), (e, v) in zip(_condElements(c), cs[i])
        push!(out, :(local $v = $(jl(g, e))))
      end
      local ex = nothing
      for i in length(st.branches):-1:1
        local blk = Expr(:block, _stmtCode(g, st.branches[i][2], condBase, nwhen; inWhen = true)..., :(_fired = true))
        local cond = foldl((x, y) -> :($x || $y), [:($v && !p.wcond[$k]) for (v, k) in zip(cs[i], ks[i])])
        ex = ex === nothing ? Expr(:if, cond, blk) : Expr(:elseif, cond, blk, ex)
      end
      ex = Expr(:if, ex.args...)
      push!(out, :(fire && $ex))
      push!(out, :(mode && $(Expr(:block, [:(p.wnew[$k] = $v) for (kk, vv) in zip(ks, cs) for (k, v) in zip(kk, vv)]...))))
    elseif st isa SAssert
      #= checked with the other asserts after each step; in a when body, when the body runs =#
      local failed = :(!($(jl(g, st.cond))) && _assertFailed(p, $(st.k), ($(st.k), _col), t, $(jl(g, st.msg))))
      push!(out, inWhen ? failed : :(p.checking[1] && $failed))
    elseif st isa SCall
      push!(out, jl(g, st.call))
    elseif st isa SBreak
      push!(out, :(break))
    end
  end
  return out
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
function generateArrayODECode(fm::F.FlatModel, modelName::String; functions = nothing)
  try
    return (modelName, _generate(fm, modelName; functions = functions))
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
  ns("variable type of $(_str(v.name))")
end

#= Whether generated code raises an error (a function's assert: Base.error). =#
function _containsErrorCall(@nospecialize(x))::Bool
  x isa AbstractVector && return any(_containsErrorCall, x)
  x isa Expr || return false
  x.head == :call && (x.args[1] == :error || x.args[1] == :(Base.error)) && return true
  return any(_containsErrorCall, x.args)
end

#= The model's Modelica functions as Julia functions of the module (the MTK path's code
   generation: SimulationCode.generateSimCodeFunctions, AlgorithmicCodeGeneration.generateFunctions),
   each a module constant named as the frontend's function list names it (dots as _), so the
   equations and the functions' calls of each other find it. Returns (definitions, outputs). =#
function _functionDefinitions(functions)
  local defs = Any[]; local outs = Dict{String, Int}()
  (functions === nothing || isempty(functions)) && return (defs, outs)
  local SC = _OMBACKEND.SimulationCode
  local (mfs, _) = try
    SC.generateSimCodeFunctions(functions)
  catch err
    ns("the model's functions: " * first(sprint(showerror, err), 160))
  end
  mfs = SC.flattenRecordParameters(mfs)
  local (exprs, names) = try
    _OMBACKEND.CodeGeneration.AlgorithmicCodeGeneration.generateFunctions(mfs)
  catch err
    ns("the model's functions: " * first(sprint(showerror, err), 160))
  end
  for (mf, ex, nm) in zip(mfs, exprs, names)
    local impl = nothing
    local find = function (x)
      x isa Expr || return
      #= MODELICA_FUNCTION_IMPLS[:name] = impl (FUNCTION_DERIVATIVE_RULES[:name] = (derivative,
         inputs) is one too: the rule became the function, "Tuple not callable") =#
      if x.head == :(=) && x.args[1] isa Expr && x.args[1].head == :ref && x.args[1].args[end] == QuoteNode(Symbol(nm)) &&
         endswith(string(x.args[1].args[1]), "MODELICA_FUNCTION_IMPLS")
        impl = x.args[2]
      else
        foreach(find, x.args)
      end
    end
    find(ex)
    impl === nothing && ns("the code of function $(nm)")
    push!(defs, :(const $(Symbol(nm)) = $impl))
    outs[nm] = length(mf.outputs)
  end
  return (defs, outs)
end

function _generate(fm::F.FlatModel, modelName::String; functions = nothing)::Expr
  local (functionDefs, functionOutputs) = _functionDefinitions(functions)
  (FULL[] || isempty(fm.initialAlgorithms)) || ns("initial algorithms (OMBackend.ARRAY_PATH_FULL)")
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
                EqClass[], EqClass[], AlgClass[], Tuple{String, String, Bool}[], WhenClass[], AssertClass[], 0, 0, 0, 0,
                -1000, false, Dict{Int, Int}(), AlgClass[], String[], AssertClass[], functionOutputs)
  local stateNamesOrdered = String[]; local algNamesOrdered = String[]
  local u0 = Float64[]; local d0 = Float64[]; local discreteNamesOrdered = String[]; local a0 = Float64[]
  local fixedMask = Bool[]
  local fixedNonStates = String[]
  for v in fm.variables
    local nm = _crefName(v.name)
    F.hasKnownSize(v.ty) || ns("$(nm) of unknown size")
    #= its value a C pointer, no Float64 (Buildings' ExtendableArray in the borehole
       examples); the MTK path holds it as a data structure =#
    F.isExternalObject(F.arrayElementType(v.ty)) && ns("the external object $(nm)")
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
      append!(fixedMask, _fixedValues(v, n))
      append!(stateNamesOrdered, _elementNames(v))
    elseif var <= F.Variability.NON_STRUCTURAL_PARAMETER
      local fx = _fixedValues(v, n; default = true)
      if any(!, fx)
        FULL[] || ns("parameter $(nm) with fixed = false (OMBackend.ARRAY_PATH_FULL)")
        all(!, fx) || ns("parameter $(nm) with fixed = false on some elements")
        F.isBound(v.binding) && ns("parameter $(nm) with fixed = false and a binding")
        push!(g.freeParams, nm)
      end
      g.vars[nm] = VarInfo(:param, dims, perPart, 0, _vtypeOrOther(v), v)
    elseif var == F.Variability.CONTINUOUS
      g.vars[nm] = VarInfo(:alg, dims, perPart, g.nAlg, :real, v)
      g.nAlg += n
      append!(a0, _startValues(v, n))
      any(_fixedValues(v, n)) && push!(fixedNonStates, nm)
      append!(algNamesOrdered, _elementNames(v))
    else
      g.vars[nm] = VarInfo(:discrete, dims, perPart, g.nDiscrete, _vtype(v), v)
      g.nDiscrete += n
      append!(d0, _startValues(v, n))
      append!(discreteNamesOrdered, _elementNames(v))
    end
  end
  if g.nStates == 0
    #= A model without states gets one with der = 0 (as omc), so it can be integrated. =#
    isempty(fm.variables) && ns("no variables")
    g.vars["\$dummy"] = VarInfo(:state, Int[], [0], 0, :real, fm.variables[1])
    g.nStates = 1
    push!(u0, 0.0); push!(fixedMask, true); push!(stateNamesOrdered, "\$dummy")
    push!(g.classes, EqClass(Ref("\$dummy", IR[], true), Lit(0.0), 0, Matrix{Int}(undef, 0, 1), 0, "der(\$dummy) = 0"))
  end
  for eq in fm.equations
    _addEquation!(g, eq, String[], NO_SLOTS)
  end
  #= Declaration bindings of variables (Real y = e;) are equations too. =#
  for v in fm.variables
    local nm = _crefName(v.name)
    local info = g.vars[nm]
    (info.kind in (:alg, :discrete, :state) && F.isBound(v.binding)) || continue
    local nd = length(info.dims)
    local ctx = Ctx(Dict{String, Int}(), collect(1:nd))
    local lhs = Ref(nm, IR[Slot(k) for k in 1:nd], false)
    local rhs = toIR(g, F.getTypedExp(v.binding), ctx)
    local dom = _crossDomain(NO_SLOTS, [collect(1:d) for d in info.dims])
    size(dom, 2) == 0 && continue
    push!(g.classes, EqClass(lhs, rhs, nd, dom, ctx.nrel[], string(nm, " = ", _str(F.getTypedExp(v.binding)))))
  end
  (FULL[] || isempty(fm.initialEquations)) || ns("initial equations (OMBackend.ARRAY_PATH_FULL)")
  (FULL[] || isempty(fixedNonStates)) || ns("fixed start on non-state $(fixedNonStates[1]) (OMBackend.ARRAY_PATH_FULL)")
  for nm in fixedNonStates
    #= fixed = true on a variable that is no state: the initial equation v = start =#
    local info = g.vars[nm]
    local v = info.var
    local n = isempty(info.dims) ? 1 : prod(info.dims)
    local fx = _fixedValues(v, n); local st = _startValues(v, n)
    local k = 0
    for t in (isempty(info.dims) ? [()] : vec([reverse(t) for t in Iterators.product(reverse([1:d for d in info.dims])...)]))
      k += 1
      fx[k] || continue
      push!(g.initClasses, EqClass(Ref(nm, IR[Lit(i) for i in t], false), Lit(st[k]), 0, NO_SLOTS, 0, "$(nm) = start"))
    end
  end
  for eq in fm.initialEquations
    _addEquation!(g, eq, String[], NO_SLOTS; initial = true)
  end
  for alg in fm.algorithms
    _addAlgorithm!(g, alg)
  end
  for alg in fm.initialAlgorithms
    _addInitialAlgorithm!(g, alg)
  end
  g.classes = _reroll(g.classes)
  #= Every discrete variable is assigned in a when-equation. =#
  #= Discrete variables not assigned in a when-equation are computed by equations: unknowns
     of the system, kept in the discrete buffer (they change only at events: their
     relations are fixed between events). =#
  local assigned = Set{String}(t.name for w in g.whens for (kind, t, _) in w.body if kind == :assign)
  local nU = g.nStates + g.nAlg
  for (nm, info) in sort!(collect(g.vars); by = first)
    (info.kind == :discrete && !(nm in assigned)) || continue
    for k in 1:(isempty(info.dims) ? 1 : prod(info.dims))
      g.discUnknown[info.offset + k] = (nU += 1)
    end
  end
  if !FULL[]
    isempty(g.whens) || ns("when-equations (OMBackend.ARRAY_PATH_FULL)")
    isempty(g.asserts) && isempty(g.assertTexts) || ns("asserts (OMBackend.ARRAY_PATH_FULL)")
    any(c -> c.nrel > 0, g.classes) && ns("relations on continuous variables: events (OMBackend.ARRAY_PATH_FULL)")
    any(a -> a.nrel > 0 || a.nwhen > 0, g.algs) && ns("events in an algorithm (OMBackend.ARRAY_PATH_FULL)")
    g.usesInitial && ns("initial() (OMBackend.ARRAY_PATH_FULL)")
  end
  local el = _elements(g)
  local matchRow = _match(el, nU)
  local (batches, pos, producer) = _schedule(el, matchRow, nU)
  local deps = _stateDeps(el, matchRow, batches, producer)
  local pattern = _jacobianPattern(el, matchRow, deps, g.nStates)
  local (initUnknowns, initBody, nInit, freeCols) = _initialization(g, deps, producer, fixedMask)
  #= initial algorithms: code, and what they assign (initial equations may not read it) =#
  local initAlgOuts = Set{String}()
  local collectOuts! = function (stmts)
    for st in stmts
      st isa SAssign && push!(initAlgOuts, st.target.name)
      st isa SIf && foreach(b -> collectOuts!(b[2]), st.branches)
      (st isa SFor || st isa SWhile) && collectOuts!(st.body)
    end
  end
  foreach(ia -> collectOuts!(ia.body), g.initAlgs)
  for c in g.initClasses, r in accesses!(accesses!(Ref[], c.lhs), c.rhs)
    r.name in initAlgOuts && ns("initial equation reads $(r.name), which an initial algorithm assigns")
  end
  g.relBase = 0; g.relCount = 0
  local initAlgBody = Any[]
  for ia in g.initAlgs
    local blk = Expr(:block, _stmtCode(g, ia.body, 0, 0)...)
    push!(initAlgBody, ia.nslots == 0 ? Expr(:let, :(_col = 1), blk) : _loopCode(ia.nslots, ia.domain, collect(1:size(ia.domain, 2)), blk))
  end
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
    nWhenInst += length(_condElements(w.cond)) * size(w.domain, 2)
  end
  local algRelBases = Int[]; local algCondBases = Int[]
  for a in g.algs
    push!(algRelBases, nRelTotal)
    nRelTotal += a.nrel * size(a.domain, 2)
    push!(algCondBases, nWhenInst)
    nWhenInst += a.nwhen * size(a.domain, 2)
  end
  #= the outputs of each algorithm instance: algebraic and discrete buffer positions =#
  local discOf = Dict(u => off for (off, u) in g.discUnknown)
  local algOuts = [[Int[] for _ in 1:size(a.domain, 2)] for a in g.algs]
  local algDOuts = [[Int[] for _ in 1:size(a.domain, 2)] for a in g.algs]
  for r in (el.nEq + 1):length(el.class)
    local ai = -el.class[r]
    for u in el.outs[r]
      u <= g.nStates + g.nAlg ? push!(algOuts[ai][el.col[r]], u - g.nStates) : push!(algDOuts[ai][el.col[r]], discOf[u])
    end
  end
  #= The instances the state derivatives need (backwards from the rows that solve for a
     derivative, through the algebraic unknowns they read; discrete values are read from the
     discrete buffer, set at events): the right-hand side computes only these (the MTK path
     likewise leaves outputs to the observed functions: acos(u) of an output was evaluated
     a hair outside its domain by the solver's finite-difference time gradient). =#
  local needed = falses(length(el.class))
  local stack = Int[r for r in eachindex(el.class) if 0 < matchRow[r] <= g.nStates]
  foreach(r -> (needed[r] = true), stack)
  while !isempty(stack)
    local r = pop!(stack)
    for i in el.uptr[r]:(el.uptr[r + 1] - 1)
      local u = el.uid[i]
      (u > g.nStates + g.nAlg || u == matchRow[r] || u in el.outs[r]) && continue
      local q = producer[u]
      (q > 0 && !needed[q]) || continue
      needed[q] = true
      push!(stack, q)
    end
  end
  #= Code: one loop per batch (rhsBody: the needed columns only). =#
  local body = Any[]; local rhsBody = Any[]
  local emit! = function (nslots, domain, batch, code)
    push!(body, nslots == 0 ? Expr(:let, :(_col = 1), code) : _loopCode(nslots, domain, [el.col[r] for r in batch], code))
    local need = [r for r in batch if needed[r]]
    isempty(need) && return
    push!(rhsBody, nslots == 0 ? Expr(:let, :(_col = 1), copy(code)) : _loopCode(nslots, domain, [el.col[r] for r in need], copy(code)))
  end
  local solved = Dict{Tuple{Int, Int}, Tuple{Ref, IR}}()
  for ((ci, p), batch) in batches
    if ci < 0
      #= an algorithm: assigned variables start from their start (continuous) or pre
         (discrete) values, then the statements =#
      local ai = -ci
      local alg = g.algs[ai]
      g.relBase = algRelBases[ai]; g.relCount = alg.nrel
      local stmts = Any[
        :(for _o in $(algOuts[ai])[_col]; a[_o] = $(a0)[_o]; end),
        :(for _o in $(algDOuts[ai])[_col]; p.d[_o] = p.dpre[_o]; end)]
      append!(stmts, _stmtCode(g, alg.body, algCondBases[ai], alg.nwhen))
      emit!(alg.nslots, alg.domain, batch, Expr(:block, stmts...))
      continue
    end
    local c = g.classes[ci]
    local (target, value) = get!(() -> _solveFor(c.lhs, c.rhs, p, c.text), solved, (ci, p))
    g.relBase = relBases[ci]; g.relCount = c.nrel
    local tinfo = g.vars[target.name]
    local assign = tinfo.kind == :discrete ?
      :(p.d[$(_linear(tinfo, Any[jl(g, s) for s in target.subs]))] = Float64($(jl(g, value)))) :
      :($(jl(g, target)) = $(jl(g, value)))
    emit!(c.nslots, c.domain, batch, assign)
  end
  #= When-equations: each instance fires when its condition becomes true. =#
  local whenBody = Any[]; local whenConds = Any[]
  #= sample(start, interval): per instance, code for its start and interval from the parameters,
     run after the initialization (a start may be a parameter with fixed = false that an initial
     algorithm computes: CDL Logical.Sources.Pulse) =#
  local sampleSchedule = Any[]
  for (wi, w) in enumerate(g.whens)
    g.relBase = whenRelBases[wi]; g.relCount = w.nrel
    g.sampleBase = length(sampleSchedule); g.sampleCount = w.nsample
    if w.nsample > 0
      local samples = Dict{Int, Op}()
      local collect! = function (x)
        x isa Op && x.op == :sample && (samples[x.args[3].v] = x)
        x isa Op && foreach(collect!, x.args)
        x isa Rel && (collect!(x.lhs); collect!(x.rhs))
        x isa EvFn && collect!(x.arg)
      end
      collect!(w.cond)
      for j in 1:size(w.domain, 2), id in 1:w.nsample
        local sx = samples[id]
        (_parametersOnly(g, sx.args[1]) && _parametersOnly(g, sx.args[2])) ||
          ns("sample() with a start or interval that is not a parameter expression")
        local k = length(sampleSchedule) + 1
        local slots = [:($(jl(g, Slot(r))) = $(w.domain[r, j])) for r in 1:size(w.domain, 1)]
        push!(sampleSchedule, Expr(:let, Expr(:block, slots...),
                                   :(SAMPLE_START[$k] = Float64($(jl(g, sx.args[1])));
                                     SAMPLE_INTERVAL[$k] = Float64($(jl(g, sx.args[2]))))))
      end
    end
    local branchConds = (w.cond isa Op && w.cond.op == :elsewhen) ? w.cond.args : IR[w.cond]
    local stmtsOf = [Any[] for _ in branchConds]
    for (i, (kind, target, value)) in enumerate(w.body)
      local b = findlast(st -> st <= i, w.starts)
      if kind == :call
        push!(stmtsOf[b], jl(g, value))
        continue
      end
      local info = g.vars[target.name]
      local idx = Any[jl(g, s) for s in target.subs]
      if kind == :reinit
        push!(stmtsOf[b], :(p.ureinit[$(_linear(info, idx))] = $(jl(g, value))))
      else
        push!(stmtsOf[b], :(p.d[$(_linear(info, idx))] = Float64($(jl(g, value)))))
      end
    end
    local elems = _condElements(w.cond)
    local ks = [:($(whenCondBases[wi]) + (_col - 1) * $(length(elems)) + $j) for j in eachindex(elems)]
    local cond = Expr(:block, [:(p.wnew[$k] = $(jl(g, e))) for (k, e) in zip(ks, elems)]...)
    #= when c1 then ... elsewhen c2 then ...: the first branch whose condition rose =#
    local inst = nothing
    local pos = 1
    local rises = Any[]
    for bc in branchConds
      local n = length(_condElements(bc))
      push!(rises, foldl((x, y) -> :($x || $y), [:(p.wnew[$k] && !p.wcond[$k]) for k in ks[pos:(pos + n - 1)]]))
      pos += n
    end
    for b in length(branchConds):-1:1
      local blk = Expr(:block, stmtsOf[b]..., :(_fired = true))
      inst = inst === nothing ? Expr(:if, :(fire && ($(rises[b]))), blk) :
                                Expr(b == 1 ? :if : :elseif, :(fire && ($(rises[b]))), blk, inst)
    end
    push!(whenConds, w.nslots == 0 ? Expr(:let, :(_col = 1), cond) : _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), cond))
    push!(whenBody, w.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), inst))
  end
  #= Asserts: each instance checked at the start and after every step. =#
  local assertBody = Any[]
  local assertTexts = copy(g.assertTexts)   #= the algorithms' asserts first (their numbers) =#
  local nAssertInst = 0
  for a in g.asserts
    push!(assertTexts, (a.condition, a.message, a.warning))
    local base = nAssertInst
    local k = length(assertTexts)
    local inst = :(($(jl(g, a.cond))) || _assertFailed(p, $k, $base + _col, t, $(jl(g, a.msg))))
    push!(assertBody, a.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(a.nslots, a.domain, collect(1:size(a.domain, 2)), inst))
    nAssertInst += size(a.domain, 2)
  end
  local initAssertBody = Any[]
  for a in g.initAsserts
    push!(assertTexts, (a.condition, a.message, a.warning))
    local base = nAssertInst
    local k = length(assertTexts)
    local inst = :(($(jl(g, a.cond))) || _assertFailed(p, $k, $base + _col, t, $(jl(g, a.msg))))
    push!(initAssertBody, a.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(a.nslots, a.domain, collect(1:size(a.domain, 2)), inst))
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
  for nm in g.freeParams
    nm in g.structural && ns("parameter $(nm) with fixed = false used structurally")
    haskey(layoutOf, nm) || continue
  end
  for (sym, deps, _) in rules
    any(d -> d in Symbol[g.paramSyms[f] for f in g.freeParams if haskey(g.paramSyms, f)], deps) &&
      ns("a parameter binding depends on a parameter with fixed = false")
  end
  local initStates = Int[k for k in initUnknowns if k <= g.nStates]
  local initParams = Int[layoutOf[freeCols[k - g.nStates][1]][1] + freeCols[k - g.nStates][2] for k in initUnknowns if k > g.nStates]
  #= the scalar parameters the initial algorithms computed, from their locals to p.values =#
  local initAlgWriteBack = Any[:(p.values[$(layoutOf[nm][1] + 1)] = $(g.paramSyms[nm]))
                               for nm in sort!(collect(initAlgOuts))
                               if nm in g.freeParams && haskey(layoutOf, nm) && !(g.params[nm] isa AbstractArray)]
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
    import NonlinearSolve
    import ADTypes
    import DiffEqCallbacks
    #= the MTK path's function code names it (algorithmic.jl: an output array's growth) =#
    const OMBackend = $(_OMBACKEND)

    $(functionDefs...)
    const STATE_NAMES = $(stateNamesOrdered)
    const ALGEBRAIC_NAMES = $(algNamesOrdered)
    const DISCRETE_NAMES = $(discreteNamesOrdered)
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
    #= initialization: the states the initial equations determine (one per residual) =#
    const INIT_UNKNOWNS = $(initStates)
    const INIT_PARAMS = $(initParams)   #= fixed = false parameters: positions in p.values =#
    const N_INIT = $(nInit)
    const D0 = $(d0)
    const N_RELATIONS = $(nRelTotal)
    const USES_INITIAL = $(g.usesInitial)
    #= set by sampleSchedule! after the initialization =#
    const SAMPLE_START = zeros($(length(sampleSchedule)))
    const SAMPLE_INTERVAL = zeros($(length(sampleSchedule)))
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
    function equations!(du, a, u, p, t, z, mode::Bool, fire::Bool = false)
      $(paramLocals...)
      local _fired = false
      @inbounds begin
        $(body...)
      end
      return _fired
    end

    #= The when-equations: all conditions first (into p.wnew, crossing functions into z), then,
       when fire, the bodies of those whose condition is true and was false at the start of the
       sweep (p.wcond, pre() of the condition), in order (OpenModelica's order: a when on a
       variable another body sets fires in the next pass, after the equations); reinit() goes
       to p.ureinit (eventIteration! applies it after the sweep). Returns whether one fired. =#
    function whens!(du, a, u, p, t, z, mode::Bool, fire::Bool)
      $(paramLocals...)
      local _fired = false
      @inbounds begin
        $(whenConds...)
        $(whenBody...)
      end
      return _fired
    end

    const ASSERTS = $(assertTexts)
    #= A model function with an assert: the equations are evaluated after each step as for
       the model's asserts (a call only an algebraic variable reads was never made: OM.jl's
       AssertTests.ConstantCall, positiveTwice(-1.0)). =#
    const FUNCTION_ASSERTS = $(_containsErrorCall(functionDefs))
    const WARNED = Set{Any}()

    #= PROBE: _violated! looks for a violation without reporting it =#
    const PROBE = Ref(false)
    const VIOLATED = Ref(false)

    function _assertFailed(p, k::Int, instance, t, message)
      local (condition, _, warning) = ASSERTS[k]
      if PROBE[]
        (warning && instance in WARNED) || (VIOLATED[] = true)
        return true
      end
      warning || throw($(ModelicaAssertionError)(Float64(t), message, condition))
      instance in WARNED || (push!(WARNED, instance); @warn string("Assertion violated at time ", t, ": ", message) condition)
      return true
    end

    #= Whether an assert is violated at (u, t) (a warning reported already does not count). =#
    function _violated!(u, p, t)
      PROBE[] = true; VIOLATED[] = false
      try
        checkAsserts(u, p, t)
      finally
        PROBE[] = false
      end
      return VIOLATED[]
    end

    #= After a step: an assert violated at its end is reported at the first violating time in
       the step (bisection on the step's interpolation), as the MTK path reports the crossing. =#
    function _assertStep(u, t, integrator)
      local p = integrator.p
      _violated!(u, p, t) || return false
      local lo = integrator.tprev; local hi = t
      if hi > lo
        for _ in 1:100
          local mid = (lo + hi) / 2
          _violated!(integrator(mid), p, mid) ? (hi = mid) : (lo = mid)
          hi - lo <= 1.0e-12 * max(1.0, abs(hi)) && break
        end
      end
      checkAsserts(hi == t ? u : integrator(hi), p, hi)
      return false
    end

    function checkAsserts(u, p, t)
      local du = similar(u); local a = similar(u, $nAlg)
      p.checking[1] = true
      try
        equations!(du, a, u, p, t, nothing, false)
      finally
        p.checking[1] = false
      end
      $(paramLocals...)
      @inbounds begin
        $(assertBody...)
      end
      return false
    end

    #= Residuals of the initial equations at (u, t), du and a from equations!. =#
    function initialResiduals!(res, du, a, u, p, t)
      $(paramLocals...)
      @inbounds begin
        $(initBody...)
      end
      return nothing
    end

    #= The initial state: the free states (and fixed = false parameters, into p.values) the
       initial equations determine, solved for them. =#
    function initialize(u0, p, t)
      local nS = length(INIT_UNKNOWNS)
      local u = copy(u0)
      local du = similar(u); local a = similar(u, $nAlg)
      local setUnknowns! = function (uu, z)
        uu[INIT_UNKNOWNS] .= view(z, 1:nS)
        p.values[INIT_PARAMS] .= view(z, (nS + 1):length(z))
        return uu
      end
      #= the relations keep their values during a solve =#
      local residual = function (res, z, _)
        local uz = similar(z, length(u0))
        uz .= u0
        setUnknowns!(uz, z)
        local duz = similar(uz); local az = similar(uz, $nAlg)
        equations!(duz, az, uz, p, t, nothing, false)
        initialResiduals!(res, duz, az, uz, p, t)
        return nothing
      end
      #= MLS 8.6, as the MTK path: the relations at the guess, a solve with them, then at the
         solution again, until they settle (a branch the solved state selects); relations that
         cycle (no consistent branch) keep the first solution =#
      local z = vcat(u0[INIT_UNKNOWNS], p.values[INIT_PARAMS])
      local first = nothing
      local seen = Tuple{Vector{Bool}, Vector{Float64}}[]
      for _ in 1:20
        equations!(du, a, setUnknowns!(u, z), p, t, nothing, true)
        local rel0 = copy(p.rel); local e0 = copy(p.evv)
        if (rel0, e0) in seen
          equations!(du, a, setUnknowns!(u, first), p, t, nothing, true)
          return u
        end
        push!(seen, (rel0, e0))
        local prob = NonlinearSolve.NonlinearProblem(NonlinearSolve.NonlinearFunction(residual), z)
        local sol = NonlinearSolve.solve(prob, NonlinearSolve.NewtonRaphson(autodiff = ADTypes.AutoFiniteDiff());
                                         abstol = 1e-10, reltol = 1e-10)
        #= no solution: refused as the MTK path refuses it (OpenModelica fails there too) =#
        DifferentialEquations.SciMLBase.successful_retcode(sol) ||
          $(_OMBACKEND.unsupported)("fixed start values or initial equations that the initialization cannot hold",
                                    string("the initial system (", sol.retcode, ")"))
        z = copy(sol.u)
        first === nothing && (first = copy(z))
        equations!(du, a, setUnknowns!(u, z), p, t, nothing, true)
        (p.rel == rel0 && p.evv == e0) && return u
      end
      equations!(du, a, setUnknowns!(u, first), p, t, nothing, true)
      return u
    end

    #= The asserts of the initial equations, once after the initialization. =#
    function checkInitialAsserts(u, p, t)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, nothing, false)
      $(paramLocals...)
      @inbounds begin
        $(initAssertBody...)
      end
      return nothing
    end

    #= The initial algorithms, once at the start (asserts checked); the algebraic variables
       they read from the equations at the initial state. =#
    function initialAlgorithms!(u, p, t)
      $(paramLocals...)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, nothing, true)
      local z = nothing; local mode = true; local fire = false; local _fired = false
      p.checking[1] = true
      try
        @inbounds begin
          $(initAlgBody...)
        end
        $(initAlgWriteBack...)
      finally
        p.checking[1] = false
      end
      return nothing
    end

    #= The sample() starts and intervals from the parameter values (after the initialization). =#
    function sampleSchedule!(p)
      $(paramLocals...)
      $(sampleSchedule...)
      return nothing
    end

    #= The right-hand side: only what the state derivatives need (relations keep their values). =#
    function RHS!(du, u, p, t)
      $(paramLocals...)
      local a = similar(u, $nAlg); local z = nothing; local mode = false; local fire = false; local _fired = false
      @inbounds begin
        $(rhsBody...)
      end
      return nothing
    end

    #= The algebraic variables at a point (u, t) of a solution: the discrete, relation and
       event-function values of that time from the event log (after a solve p holds the final
       ones), on a copy of the discrete state. =#
    function algebraics(u, p, t, right::Bool = false)
      local a = similar(u, $nAlg)
      local times = first.(p.dlog)
      local entry = isempty(times) ? vcat(p.d, Float64.(p.rel), p.evv) : p.dlog[$(_dlogIndex)(times, t, p.tend[1], right)][2]
      local nD = length(p.d); local nR = length(p.rel)
      local d = entry[1:nD]
      local rel = entry[(nD + 1):(nD + nR)] .!= 0
      local evv = entry[(nD + nR + 1):end]
      local q = $(paramsType)(p.values, rel, d, copy(d), copy(p.wcond), copy(p.sampleActive),
                              [false], [false], copy(p.upre), copy(p.ureinit), copy(p.wnew), p.dlog, evv, copy(p.hyst), copy(p.tend), p.model)
      equations!(similar(u), a, u, q, t, nothing, false)
      return a
    end

    #= Event iteration at (u, t), in sweeps as OpenModelica's updateDiscreteSystem (and the MTK
       path, relationRefresh.jl): within a sweep pre() keeps the values from before it, and the
       relations, equations and fired when bodies are evaluated again until they settle (an
       equation reading a variable a when body just set: ch = change(k)); a sweep that changed
       a discrete value starts the next one from the new values, until one changes nothing. =#
    function eventIteration!(u, p, t, fire::Bool)
      local du = similar(u); local a = similar(u, $nAlg)
      for _ in 1:50
        p.dpre .= p.d; p.upre .= u; p.ureinit .= u
        #= passes with the same pre() until nothing changes (a fired body runs in each: with
           pre() fixed it gives the same values) =#
        local settled = false
        for _ in 1:50
          local relBefore = copy(p.rel); local dBefore = copy(p.d)
          local rBefore = copy(p.ureinit); local wBefore = copy(p.wnew); local eBefore = copy(p.evv)
          equations!(du, a, u, p, t, nothing, true, fire)
          whens!(du, a, u, p, t, nothing, true, fire)
          settled = relBefore == p.rel && dBefore == p.d && rBefore == p.ureinit && wBefore == p.wnew &&
                    eBefore == p.evv
          settled && break
        end
        settled || return false
        #= reinit() takes effect after the sweep; the next one reads it (and pre() of it), and
           the conditions' values become pre() of the conditions =#
        #= (at the initialization, as OpenModelica, reinit() has no effect) =#
        local moved = !p.initPhase[1] && p.ureinit != u
        moved && (u .= p.ureinit)
        local switched = p.wnew != p.wcond
        p.wcond .= p.wnew
        (moved || switched || p.d != p.dpre) || return true
      end
      return false
    end

    #= The event iteration at an event during the solve: a model that does not settle
       (relations switching back and forth) stops the simulation, as OpenModelica. =#
    function _eventAt!(integrator)
      eventIteration!(integrator.u, integrator.p, integrator.t, true) && return true
      @error string("[events] the event iteration did not settle at t = ", integrator.t,
                    " (relations switching back and forth: a chattering model); the simulation stops")
      DifferentialEquations.SciMLBase.terminate!(integrator, DifferentialEquations.SciMLBase.ReturnCode.Failure)
      return false
    end

    function conditions!(out, u, t, integrator)
      local p = integrator.p
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, out, false)
      whens!(du, a, u, p, t, out, false, false)
      return nothing
    end

    #= at the start of a solve: the relations' hysteresis for its tolerance and span =#
    function _startHysteresis!(c, u, t, integrator)
      integrator.p.hyst[1] = $(_hysteresis)(integrator)
      return nothing
    end

    function _logDiscretes!(p, t)
      local entry = vcat(p.d, Float64.(p.rel), p.evv)
      (isempty(p.dlog) || p.dlog[end][2] != entry) && push!(p.dlog, (t, entry))
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
      p.wcond .= p.wnew
      _eventAt!(integrator)
      _logDiscretes!(p, t)
      DifferentialEquations.u_modified!(integrator, true)
      return nothing
    end

    #= A sample instant: the samples due now are true for the event iteration, false after it
       (the when-conditions take that value too, so the next instant is a rising edge). =#
    function affectSample!(integrator)
      local t = integrator.t; local p = integrator.p; local u = integrator.u
      for id in eachindex(SAMPLE_START)
        local k = round((t - SAMPLE_START[id]) / SAMPLE_INTERVAL[id])
        p.sampleActive[id] = k >= 0 && abs(t - (SAMPLE_START[id] + k * SAMPLE_INTERVAL[id])) <= 1e-9 * max(1.0, abs(t))
      end
      #= the relations literal at the instant (H = 0): one whose crossing function is 0 there (a
         threshold the tick reaches, `time >= pulseStart + 0.2` at 1) changes with it, as
         OpenModelica takes that state event together with the time event =#
      local _hyst = p.hyst[1]
      p.hyst[1] = 0.0
      try
        _eventAt!(integrator)
      finally
        p.hyst[1] = _hyst
      end
      fill!(p.sampleActive, false)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, nothing, true)
      whens!(du, a, u, p, t, nothing, true, false)
      p.wcond .= p.wnew
      _logDiscretes!(p, t)
      DifferentialEquations.u_modified!(integrator, true)
      return nothing
    end

    function sampleTimes(tspan)
      local times = Float64[]
      for (st, dt) in zip(SAMPLE_START, SAMPLE_INTERVAL)
        dt > 0 || error("array model: sample interval must be positive")
        #= a tick within round-off of the start or stop time is at it (sample(-0.9, 0.3): 0) =#
        local tol(x) = 1.0e-9 * max(1.0, abs(x))
        local k = max(0, ceil((tspan[1] - st - tol(tspan[1])) / dt))
        while st + k * dt <= tspan[2] + tol(tspan[2])
          local tk = st + k * dt
          push!(times, abs(tk - tspan[1]) <= tol(tspan[1]) ? Float64(tspan[1]) :
                       abs(tk - tspan[2]) <= tol(tspan[2]) ? Float64(tspan[2]) : tk)
          k += 1
        end
      end
      return sort!(unique!(times))
    end

    function affect!(integrator)
      _eventAt!(integrator)
      _logDiscretes!(integrator.p, integrator.t)
      DifferentialEquations.u_modified!(integrator, true)
      return nothing
    end

    #= After a step: whether a relation has another value than the stored one. A crossing
       right after an event (the ball leaving the floor it was reinitialized just below) has
       no sign change the root finder sees; it is caught here, at the end of the step. =#
    function relationsChanged(u, t, integrator)
      local p = integrator.p
      local rel0 = copy(p.rel); local w0 = copy(p.wcond); local e0 = copy(p.evv)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, nothing, true)
      whens!(du, a, u, p, t, nothing, true, false)
      local changed = rel0 != p.rel || e0 != p.evv
      p.rel .= rel0; p.wcond .= w0; p.evv .= e0
      return changed
    end

    #= The problem and its event callbacks, as getMTKProblem returns them. =#
    function $(Symbol(modelName, "Model"))(tspan = (0.0, 1.0); parameters::AbstractDict = Dict{String, Any}())
      local p = $(paramsType)(flatParameters(isempty(parameters) ? PARAMS : parameterValues(parameters)),
                              zeros(Bool, N_RELATIONS), copy(D0), copy(D0), zeros(Bool, N_WHENS),
                              zeros(Bool, length(SAMPLE_START)), [false], [false], copy(U0), copy(U0),
                              zeros(Bool, N_WHENS), Tuple{Float64, Vector{Float64}}[], zeros(N_RELATIONS), [0.0], [Float64(tspan[2])], @__MODULE__)
      local u0 = copy(U0)
      N_INIT > 0 && (u0 = initialize(u0, p, tspan[1]))
      $(isempty(initAlgBody) ? nothing : :(initialAlgorithms!(u0, p, tspan[1])))
      isempty(SAMPLE_START) || sampleSchedule!(p)
      #= relations and when-conditions at the start (no when fires there) =#
      (N_RELATIONS > 0 || N_WHENS > 0) && (eventIteration!(u0, p, tspan[1], false) ||
                                           error("array model: the initial relations do not settle"))
      if USES_INITIAL
        #= when initial(): fire during initialization, then initial() is false again =#
        p.initPhase[1] = true
        eventIteration!(u0, p, tspan[1], true) || error("array model: the initial events do not settle")
        p.initPhase[1] = false
        local du = similar(u0); local a = similar(u0, $nAlg)
        equations!(du, a, u0, p, tspan[1], nothing, true)
        whens!(du, a, u0, p, tspan[1], nothing, true, false)
        p.wcond .= p.wnew
      end
      _logDiscretes!(p, tspan[1])
      $(isempty(initAssertBody) ? nothing : :(checkInitialAsserts(u0, p, tspan[1])))
      local f = ODEFunction(RHS!; jac_prototype = $jacProto, sys = $(ArraySystem)(@__MODULE__))
      #= RightRootFind: the event lands just past the root, where the relation has its new value. =#
      local cbs = Any[]
      #= the relations were set literally above; from the solve on they have the hysteresis =#
      N_RELATIONS > 0 && push!(cbs, VectorContinuousCallback(conditions!, affectCrossing!, N_RELATIONS;
                                                             rootfind = DifferentialEquations.SciMLBase.RightRootFind,
                                                             initialize = _startHysteresis!),
                                    DiscreteCallback(relationsChanged, affect!; save_positions = (false, false)))
      isempty(SAMPLE_START) || push!(cbs, DiffEqCallbacks.PresetTimeCallback(sampleTimes(tspan), affectSample!))
      if !isempty(ASSERTS) || FUNCTION_ASSERTS
        empty!(WARNED)
        push!(cbs, DiscreteCallback(_assertStep, integrator -> nothing;
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
