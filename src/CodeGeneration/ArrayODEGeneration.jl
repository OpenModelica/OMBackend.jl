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
     time, so the order comes out as a few loops. Before: operations on literals folded,
     alias variables read through one representative (`_eliminateAliases!`).
  3. Each (class, matched slot) is solved once, symbolically, for its unknown (an unknown
     that occurs once, under +, -, *, /). Algebraic loops: small ones by a Newton solve per
     element, large ones torn (`_tearBlock`) and solved where the equations reach them
     (`blockSolve!`); values from parameters only computed once per solve (`invariants!`),
     the relations' crossing functions from what they need only (`crossings!`). From an
     OMJLIR model, calls inlined after index reduction where a function's body runs into
     one expression (`_inlinedEquation`).
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

Not (yet) handled, reported with the reason so the caller scalarizes as before: systems
needing index reduction (the flat model's; an OMJLIR model comes reduced), elsewhen in
when-equations, array slices.
=#
module ArrayODEGen

import OMFrontend
import ModelingToolkit
import ..CodeGeneration: stripBeginBlocks, ModelicaAssertionError
import ...LowerFlatModel: FunctionList
import ...OMJLIR
import ...Passes

const SII = ModelingToolkit.SymbolicIndexingInterface

const F = OMFrontend.Frontend

include("ArrayBlockSolve.jl")

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
  abuf::Vector{Float64}        #= the algebraic variables of a right-hand side evaluated on Float64 (RHS!): no buffer per call =#
  blocks::Vector{BlockSolver}  #= the large algebraic loops' solvers (blockSolve!), in the order of the schedule =#
  model::Module
end

"""
The entry of the discrete log for time t: at an event instant the values before it (the left
limit, as the states' interpolation and OpenModelica's result files), at the stop time the
final ones; `right` takes the values after an event instant (the second of a saved pair).
"""
_dlogIndex(times, t, tend, right::Bool = false) =
  (right || t >= tend) ? max(1, searchsortedlast(times, t)) : max(1, searchsortedfirst(times, t) - 1)

const _OMBACKEND = parentmodule(parentmodule(@__MODULE__))

"""
    StepRejectingRHS(f)

An array model's right-hand side `f` whose error (a domain error, a Modelica assert or error: the scalar path's
`_RHS_STEP_ERRORS`) rejects the step: the derivatives NaN, the solver retries smaller; the error is kept until the
run ends (`_rethrowRHSError` throws it if the run fails), not forgotten when a later call succeeds (the scalar path's
`_StepRejectingRHS` clears it: an error-level assert in a function ended an array model's run without its message).
"""
struct StepRejectingRHS{F} <: Function
  f::F
end

function (r::StepRejectingRHS)(du, u, p, t)
  try
    r.f(du, u, p, t)
  catch err
    err isa _OMBACKEND.CodeGeneration._RHS_STEP_ERRORS || rethrow()
    _OMBACKEND.CodeGeneration.LAST_RHS_ERROR[] = err
    fill!(du, NaN)
  end
  return nothing
end

"""
Name lookups accept the Modelica name (c[2].T) and the canonical one
(OMBackend.canonicalName).
"""
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

"""
In the latest world: the generated module is newer than the caller.
"""
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
    #= the binding read in the latest world too: the module is newer than the caller =#
    return (u, p, t) -> (Base.@invokelatest model.algebraics(u, p, t))[i]
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

"""
A relation of the model (MLS 8.5, OpenModelica's relationhysteresis, as the MTK path's
relationRefresh.jl): it keeps its value p.rel[k] between events and changes only in mode
(an event). zc <= 0 where it is true; its crossing function, into z when given, is shifted
by eps = H * (1 + max(|x|, |y|)) away from its value, so the root lands where the new value
is clear: a true relation stays true while zc <= eps, a false one becomes true at zc <= -eps.
With H = 0 (initialization) it is evaluated literally.
"""
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

"""
An event-generating function (integer, floor, ceil, div, mod, rem; MLS 3.7.2): its value
v = floor/ceil/trunc(q) changes only at events (in mode), and then (as a relation, _rel)
only where q has left the interval that keeps v by more than the hysteresis eps: the state
just before an event (an interpolation a hair past the boundary) keeps the old value. Two
crossing functions bound that interval: z[k] crosses where q passes its upper end, z[k + 1]
its lower end. At the initialization (H = 0) the value is literal.
"""
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

"""
The hysteresis H for a solve, as the MTK path (relationRefresh.jl _hysteresis): OpenModelica's
tolZC = 1e-4 * min(reltol, span / 500).
"""
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

"""
The frontend's evaluation of a binding, range or start value; a failure there (an
expression its evaluator does not handle) puts the model outside the array path.
"""
function _evalFrontend(@nospecialize(e))
  try
    return F.evalExp(e)
  catch err
    err isa NotSupported && rethrow(err)
    ns("the frontend cannot evaluate " * _str(e) * ": " * Base.first(sprint(showerror, err), 120))
  end
end

"""
An expression in a reason message (printing must not fail).
"""
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

"""
The variable accesses of an expression, in a fixed order (not those inside subscripts,
not pre values).
"""
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

"""
    IRVariable(variable, names, value)

A variable of an OMJLIR model (the OMJLIR front half, ArrayODEGenerationIR.jl): the variable, its elements' names
(row-major, as the frontend's `c[2].T`), and a parameter's value (a scalar, or an array row-major), evaluated on the
model's element view; `nothing` for another variable.
"""
struct IRVariable
  variable::OMJLIR.Variable
  names::Vector{String}
  value::Any
end

mutable struct VarInfo
  kind::Symbol            #= :state, :alg, :discrete or :param =#
  dims::Vector{Int}
  perPart::Vector{Int}    #= subscripts per cref part, root first =#
  offset::Int             #= first element in u, the algebraic buffer or the discrete buffer =#
  vtype::Symbol           #= :real, :int or :bool =#
  var::Union{F.Variable, IRVariable}
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
  derOf::Dict{String, String}   #= an OMJLIR model's state derivatives: \$STATEDER.x -> x =#
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

"""
A literal frontend expression as a Julia value (scalars, nested arrays row-major).
"""
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
  local val::Any = _parameterValue(g, name, info.var)
  g.params[name] = val
  g.paramSyms[name] = Symbol("P_", replace(name, r"[^A-Za-z0-9_]" => "_"))
  return val
end

"""
The value of the parameter `name` (its variable `v`): a parameter with fixed = false its start, another its binding,
evaluated by the frontend; an OMJLIR model's as evaluated on its element view.
"""
_parameterValue(g::Gen, name::String, v::F.Variable) =
  name in g.freeParams ? _freeParamStart(v) : _evalBinding(v.binding, v, "parameter")
_parameterValue(::Gen, name::String, v::IRVariable) = v.value === nothing ? ns("the value of parameter $(name)") : v.value

"""
A parameter with fixed = false starts (the initialization's guess) from its start value.
"""
function _freeParamStart(v::F.Variable)
  local dims = F.isArray(v.ty) ? [F.size(d) for d in F.arrayDims(v.ty)] : Int[]
  local st = _startValues(v, isempty(dims) ? 1 : prod(dims))
  isempty(dims) && return st[1]
  return permutedims(reshape(st, reverse(dims)...), length(dims):-1:1)   #= row-major start values =#
end

"""
Start values of a variable, row-major.
"""
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

"""
The fixed attribute of a variable's elements, row-major (default false).
"""
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
"""
String(...) of Modelica, as the MTK path formats it (resolved when called: that module
may load after this one)
"""
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

"""
Whether an expression varies continuously (states, algebraic variables, time).
"""
function _continuous(g::Gen, x::IR)::Bool
  x isa TimeIR && return true
  x isa Ref && return g.vars[x.name].kind in (:state, :alg) || any(s -> _continuous(g, s), x.subs)
  x isa Op && return any(a -> _continuous(g, a), x.args)
  x isa Rel && return true
  return false   #= an EvFn is constant between events =#
end

"""
A Boolean-valued expression (whatever type the call's argument was given: String(n > 3)).
"""
_booleanExp(@nospecialize(e)) = e isa F.CAST_EXPRESSION ? _booleanExp(e.exp) :
  (e isa F.RELATION_EXPRESSION || e isa F.LBINARY_EXPRESSION || e isa F.LUNARY_EXPRESSION || e isa F.BOOLEAN_EXPRESSION)

"""
An assert's message as an expression (String(x, ...) formatted as OpenModelica does);
its text where the array path cannot evaluate it.
"""
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

"""
An array value (a function's array argument): its elements written out (as a reduction's
terms), column-major, reshaped to its dimensions.
"""
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

"""
sum(e for i in r, j in s), product, min, max: the terms for every iterator value written out.
"""
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

"""
Matrix and vector products in an element-wise context: the inner dimension written out.
"""
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

"""
sample() of an equation as the sample slot of the when wi that schedules its instants.
"""
function _sampleSlots(x::IR, wi::Int)::IR
  x isa Op && x.op == :sample && return Op(:sampleof, IR[Lit(wi), x.args[3]])
  x isa Op && return Op(x.op, IR[_sampleSlots(a, wi) for a in x.args])
  x isa Rel && return Rel(x.op, _sampleSlots(x.lhs, wi), _sampleSlots(x.rhs, wi), x.id)
  x isa EvFn && return EvFn(x.kind, _sampleSlots(x.arg, wi), x.id)
  return x
end

"""
The first sample slot of when wi: the slots are numbered when by when, instance by instance.
"""
_whenSampleBase(g::Gen, wi::Int) = sum((g.whens[k].nsample * size(g.whens[k].domain, 2) for k in 1:(wi - 1)); init = 0)

"""
Whether `x` calls a function (`:fcall`) and reads parameters only (`_parametersOnly`).
"""
function _callOfParameters(g::Gen, x::IR)::Bool
  _parametersOnly(g, x) || return false
  local found = false
  local visit = function (y)
    y isa Op || return
    y.op == :fcall && (found = true)
    foreach(visit, y.args)
  end
  visit(x)
  return found
end

"""
Whether x reads only literals, iterators and parameters (no variable, no time).
"""
function _parametersOnly(g::Gen, x::IR)::Bool
  (x isa Lit || x isa Slot) && return true
  x isa Ref && return g.vars[x.name].kind == :param && !x.der && all(s -> _parametersOnly(g, s), x.subs)
  x isa Op && return !(x.op in (:sample, :initial)) && all(a -> _parametersOnly(g, a), x.args)
  return false
end

"""
The value of an expression that depends only on slots and parameters, at a domain column; `structural`: the code
depends on the parameters it reads (they are not tunable).
"""
function _evalStatic(g::Gen, x::IR, d::Matrix{Int}, col::Int; structural::Bool = true)
  if x isa Lit
    return x.v
  elseif x isa Slot
    return d[x.k, col]
  elseif x isa Ref
    local info = g.vars[x.name]
    (info.kind == :param && !x.der) || ns("condition on variable $(x.name)")
    structural && push!(g.structural, x.name)
    local v = _paramValue!(g, x.name)
    isempty(x.subs) && return v
    return v[(_evalStatic(g, s, d, col; structural = structural) for s in x.subs)...]
  elseif x isa Op
    local a = [_evalStatic(g, y, d, col; structural = structural) for y in x.args]
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
    #= p = expr of parameters for a parameter with fixed = false: assigned once, as an initial
       algorithm's (no unknown of the solve): MSL GlobalSeed's `id_impure =
       initializeImpureRandom(seed)`, an impure call the solve's Newton made at every evaluation,
       a new value each time (ImpureGenerator stalled). Not of free parameters the solve determines
       (a double exponential's `eta = exp(-T/tau1) - exp(-T/tau2)`); those an earlier such assignment
       gives are known (GlobalSeed's `id_impure` of its `seed`) =#
    local assignedOnce::Function = (name::String) -> any(a::AlgClass -> any(st::Stmt -> st isa SAssign && st.target.name == name, a.body), g.initAlgs)
    if isempty(edims) && ilhs isa Ref && !ilhs.der && ilhs.name in g.freeParams && !(get(g.params, ilhs.name, nothing) isa AbstractArray) &&
       all(r -> r.name != ilhs.name && g.vars[r.name].kind == :param && (!(r.name in g.freeParams) || assignedOnce(r.name)), accesses!(Ref[], irhs))
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

"""
Whether a condition depends on loop indices and parameters only.
"""
_isStatic(g::Gen, x::IR)::Bool =
  x isa Lit || x isa Slot ||
  (x isa Ref && !x.der && g.vars[x.name].kind == :param && all(s -> _isStatic(g, s), x.subs)) ||
  (x isa Op && !(x.op in (:initial, :sample)) && all(a -> _isStatic(g, a), x.args))

"""
An if-equation on variable conditions whose branches (with an else) each give the same
variables as v = e: one equation per variable, v = if c1 then e1 elseif ... else en (as
omc's if-equation to if-expression); its relations generate events.
"""
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

"""
(a, _, c) = call: the call once, into a temporary, and an assignment of each
(non-wildcard) target from its output (the call was made per target, and per element of
an array target).
"""
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

"""
initial() activates a when at the initialization only as its condition (MLS 8.6, as
OpenModelica): inside an expression (initial() or x > 0.6) it is false.
"""
_topLevelInitial(x::IR) = (x isa Op && x.op == :initial) ? x : _noInitial(x)

"""
A when-condition: a Boolean expression, or a vector of them (when {c1, c2}: Op(:vect),
each element with its own condition slot and edge; initial() counts as an element).
"""
function _whenCondition(g::Gen, @nospecialize(c::F.Expression), ctx::Ctx)::IR
  c isa F.ARRAY_EXPRESSION || return _topLevelInitial(toIR(g, c, ctx))
  return Op(:vect, IR[_topLevelInitial(toIR(g, el, ctx)) for el in c.elements])
end

"""
The condition elements of a when (one for a scalar condition).
"""
_condElements(c::IR)::Vector{IR} = (c isa Op && c.op == :elsewhen) ? IR[e for b in c.args for e in _condElements(b)] :
                                   (c isa Op && c.op == :vect) ? c.args : IR[c]
_noInitial(x::IR) = !(x isa Op) ? x : x.op == :initial ? Lit(false) : Op(x.op, IR[_noInitial(a) for a in x.args])

"""
An algorithm section: statements to IR. A vectorized algorithm (one for-loop over a \$
iterator, from a component array) keeps the loop as the class domain.
"""
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

"""
An initial algorithm: run once at the initialization, after the free states are solved and
before the first event iteration, without events (its relations and event-generating
functions literal). It may assign discrete variables (and assert); initial equations may not
read what it assigns.
"""
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

"""
Sums as sorted term lists, so connection sums that list their terms in another order share
a shape.
"""
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

"""
Literal integer subscripts replaced by new slots (their values pushed to `vals`).
"""
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

"""
Scalar equations (no slots) that differ only in literal subscripts become one class.
"""
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

"""
Zero-based position of an element in a variable (row-major); subscripts 1-based.
"""
function _elementOffset(info::VarInfo, idx::Vector{Int}, name::String)::Int
  length(idx) == length(info.dims) || ns("subscript count of $(name)")
  local off = 0
  for (i, d) in zip(idx, info.dims)
    (1 <= i <= d) || ns("subscript $(i) out of 1:$(d) on $(name)")
    off = off * d + (i - 1)
  end
  return off
end

"The ids of the event-generating functions (`EvFn`) in an expression, into `ids`."
function _eventFunctionIds!(ids::Set{Int}, x::IR)::Set{Int}
  if x isa EvFn
    push!(ids, x.id)
    _eventFunctionIds!(ids, x.arg)
  elseif x isa Op
    foreach(a::IR -> _eventFunctionIds!(ids, a), x.args)
  elseif x isa Rel
    _eventFunctionIds!(ids, x.lhs)
    _eventFunctionIds!(ids, x.rhs)
  elseif x isa Ref
    foreach(s::IR -> _eventFunctionIds!(ids, s), x.subs)
  end
  return ids
end

"The ids of the event-generating functions in statements, into `ids`."
function _eventFunctionIds!(ids::Set{Int}, stmts::Vector{Stmt})::Set{Int}
  for st in stmts
    st isa SAssign && _eventFunctionIds!(ids, st.value)
    if st isa SIf || st isa SWhen
      for (c, b) in st.branches
        _eventFunctionIds!(ids, c)
        _eventFunctionIds!(ids, b)
      end
    end
    st isa SFor && _eventFunctionIds!(ids, st.body)
    st isa SWhile && (_eventFunctionIds!(ids, st.cond); _eventFunctionIds!(ids, st.body))
    st isa SCall && _eventFunctionIds!(ids, st.call)
    st isa SAssert && _eventFunctionIds!(ids, st.cond)
  end
  return ids
end

"""
The slots of the relation numbering that are relations (`_rel`, `p.rel`), not an event-generating function's two
crossing functions (`_evfn`): each template's ids at each column of its domain from its base.
"""
function _relationSlots(g::Gen, nRelTotal::Int, relBases::Vector{Int}, whenRelBases::Vector{Int}, algRelBases::Vector{Int})::BitVector
  local slots::BitVector = trues(nRelTotal)
  local mark!::Function = function (ids::Set{Int}, base::Int, nrel::Int, ncols::Int)
    for col in 1:ncols, id in ids, k in (id, id + 1)
      slots[base + (col - 1) * nrel + k] = false
    end
  end
  for (ci, c) in enumerate(g.classes)
    c.nrel == 0 || mark!(_eventFunctionIds!(_eventFunctionIds!(Set{Int}(), c.lhs), c.rhs), relBases[ci], c.nrel, size(c.domain, 2))
  end
  for (wi, w) in enumerate(g.whens)
    w.nrel == 0 && continue
    local ids::Set{Int} = _eventFunctionIds!(Set{Int}(), w.cond)
    foreach(b::Tuple{Symbol, Ref, IR} -> _eventFunctionIds!(ids, b[3]), w.body)
    mark!(ids, whenRelBases[wi], w.nrel, size(w.domain, 2))
  end
  for (ai, a) in enumerate(g.algs)
    a.nrel == 0 || mark!(_eventFunctionIds!(Set{Int}(), a.body), algRelBases[ai], a.nrel, size(a.domain, 2))
  end
  return slots
end

"""
Whether an expression varies in time other than through the unknowns it reads: time, a state, a discrete variable
(changed at events), a relation or event-generating function, pre(), sample(), initial().
"""
function _readsDynamic(g::Gen, x::IR)::Bool
  (x isa TimeIR || x isa Pre || x isa Rel || x isa EvFn) && return true
  x isa Ref && return g.vars[x.name].kind in (:state, :discrete) || x.der || any(s::IR -> _readsDynamic(g, s), x.subs)
  x isa Op && return x.op in (:sample, :sampleof, :initial) || any(a::IR -> _readsDynamic(g, a), x.args)
  return false
end

"The operators `_folded` evaluates on literal arguments."
const _FOLDABLE = Set{Symbol}([:+, :-, :*, :/, :^, :(==), :!=, :<, :<=, :>, :>=, :!, :min, :max, :abs])

"""
`x` with the operations on literals evaluated and an if-expression of a literal condition its branch: an inlined call's
structural choices (MSL MultiBody's gravity, `if 2 == 2 then g * normalizeWithAssert({0, -1, 0}) else ...` of the
positions, read the positions only in the branch never taken: time-dependent, it was evaluated in every right-hand
side).
"""
function _folded(x::IR)::IR
  if x isa Op
    local args::Vector{IR} = IR[_folded(a) for a in x.args]
    local literal::Function = (a::IR) -> a isa Lit && a.v isa Number
    if x.op == :if && args[1] isa Lit && args[1].v isa Bool
      return args[1].v ? args[2] : args[3]
    elseif x.op in (:&&, :||) && all(a::IR -> a isa Lit && a.v isa Bool, args)
      return Lit(x.op == :&& ? all(a::IR -> a.v, args) : any(a::IR -> a.v, args))
    elseif x.op == :neg && literal(args[1])
      return Lit(-args[1].v)
    elseif x.op in _FOLDABLE && all(literal, args)
      return Lit(getfield(Base, x.op)(Any[a.v for a in args]...))
    end
    return Op(x.op, args)
  elseif x isa Ref
    return Ref(x.name, IR[_folded(s) for s in x.subs], x.der)
  elseif x isa Rel
    return Rel(x.op, _folded(x.lhs), _folded(x.rhs), x.id)
  elseif x isa EvFn
    return EvFn(x.kind, _folded(x.arg), x.id)
  end
  return x
end

#= ---------------------------------------------------------------- aliases =#

"""
`e`'s terms of a sum (through +, binary - and negation) into `acc` as (negated, term); false past four terms.
"""
function _signedTerms!(acc::Vector{Tuple{Bool, IR}}, e::IR, negated::Bool)::Bool
  length(acc) > 3 && return false
  if e isa Op && e.op == :+
    return all(a::IR -> _signedTerms!(acc, a, negated), e.args)
  elseif e isa Op && e.op == :- && length(e.args) == 2
    return _signedTerms!(acc, e.args[1], negated) && _signedTerms!(acc, e.args[2], !negated)
  elseif e isa Op && e.op == :neg
    return _signedTerms!(acc, e.args[1], !negated)
  end
  push!(acc, (negated, e))
  return true
end

"""
    _aliasTerms(c) -> Union{Nothing, Tuple{Ref, Ref, Bool}}

The two variables of an alias equation `x = y` or `x = -y` (also written `x - y = 0`, `0 = x + y`): `lhs - rhs` the
signed sum of two plain references (not under der); whether `x = -y`.
"""
function _aliasTerms(c::EqClass)::Union{Nothing, Tuple{Ref, Ref, Bool}}
  local terms::Vector{Tuple{Bool, IR}} = Tuple{Bool, IR}[]
  (_signedTerms!(terms, c.lhs, false) && _signedTerms!(terms, c.rhs, true)) || return nothing
  filter!(t::Tuple{Bool, IR} -> !_isLit(t[2], 0.0), terms)
  length(terms) == 2 || return nothing
  local (n1::Bool, x::IR) = terms[1]; local (n2::Bool, y::IR) = terms[2]
  (x isa Ref && y isa Ref && !x.der && !y.der) || return nothing
  return (x, y, n1 == n2)
end

"The variables algorithms (also initial ones) and when-equations assign."
function _assignedNames(g::Gen)::Set{String}
  local names::Set{String} = Set{String}()
  local visit!::Function = function (stmts::Vector{Stmt})
    for st in stmts
      st isa SAssign && push!(names, st.target.name)
      st isa SIf && foreach(b::Tuple{IR, Vector{Stmt}} -> visit!(b[2]), st.branches)
      st isa SWhen && foreach(b::Tuple{IR, Vector{Stmt}} -> visit!(b[2]), st.branches)
      (st isa SFor || st isa SWhile) && visit!(st.body)
    end
  end
  foreach(a::AlgClass -> visit!(a.body), g.algs)
  foreach(a::AlgClass -> visit!(a.body), g.initAlgs)
  for w in g.whens, (_, target, _) in w.body
    push!(names, target.name)
  end
  return names
end

"""
`x` with every read of an alias member as ±its root (`alias`: member => (root, negated)); the same subscripts.
"""
function _substituted(x::IR, alias::Dict{String, Tuple{String, Bool}})::IR
  if x isa Ref
    local subs::Vector{IR} = IR[_substituted(s, alias) for s in x.subs]
    local a::Union{Nothing, Tuple{String, Bool}} = x.der ? nothing : get(alias, x.name, nothing)
    a === nothing && return Ref(x.name, subs, x.der)
    local root::Ref = Ref(a[1], subs, false)
    return a[2] ? Op(:neg, IR[root]) : root
  elseif x isa Pre
    local b::Union{Nothing, Tuple{String, Bool}} = get(alias, x.ref.name, nothing)
    b === nothing && return x
    local before::Pre = Pre(Ref(b[1], x.ref.subs, false))
    return b[2] ? Op(:neg, IR[before]) : before
  elseif x isa Op
    return Op(x.op, IR[_substituted(y, alias) for y in x.args])
  elseif x isa Rel
    return Rel(x.op, _substituted(x.lhs, alias), _substituted(x.rhs, alias), x.id)
  elseif x isa EvFn
    return EvFn(x.kind, _substituted(x.arg, alias), x.id)
  end
  return x
end

"An assignment's target with the reads in its subscripts substituted (`_substituted`)."
_substitutedTarget(t::Ref, alias::Dict{String, Tuple{String, Bool}})::Ref =
  Ref(t.name, IR[_substituted(s, alias) for s in t.subs], t.der)

"A statement with its reads substituted (`_substituted`)."
function _substituted(st::Stmt, alias::Dict{String, Tuple{String, Bool}})::Stmt
  local body::Function = (b::Vector{Stmt}) -> Stmt[_substituted(x, alias) for x in b]
  st isa SAssign && return SAssign(_substitutedTarget(st.target, alias), _substituted(st.value, alias))
  st isa SIf && return SIf(Tuple{IR, Vector{Stmt}}[(_substituted(c, alias), body(b)) for (c, b) in st.branches])
  st isa SFor && return SFor(st.slot, st.range, body(st.body))
  st isa SWhile && return SWhile(_substituted(st.cond, alias), body(st.body))
  st isa SWhen && return SWhen(Tuple{IR, Vector{Stmt}}[(_substituted(c, alias), body(b)) for (c, b) in st.branches], st.ids)
  st isa SCall && return SCall(_substituted(st.call, alias))
  st isa SAssert && return SAssert(_substituted(st.cond, alias), st.k, _substituted(st.msg, alias))
  return st
end

"""
    _eliminateAliases!(g) -> Int

Alias variables read through one representative, so the equations compute each alias set once (MSL connections:
`frame_b.r_0 = frame_a.r_0`, flange chains). A pair: two real variables of the same dimensions whose alias equations
(`_aliasTerms`, the same subscripts on both, no relations) cover every element once; the pairs join sets (signed
union-find), each rooted at its state or parameter, else at an algebraic variable (a set with two of them, a discrete
variable or one an algorithm or a when-equation assigns: not joined). Every read of another member becomes ±its
root's, in the equations, when-equations, algorithms and asserts; one alias equation per member, `member = ±root`,
still gives its value (the results, the initial equations). The initial equations read the members as written: read
through the root, a fixed start on an alias of a free parameter (`y = k`, `y(start = 3, fixed = true)`) became an
initial equation of the parameter, which OpenModelica leaves at its start. Returns the number of members read through
their roots.
"""
function _eliminateAliases!(g::Gen)::Int
  local assigned::Set{String} = _assignedNames(g)
  local eligible::Function = function (name::String)
    local info::VarInfo = g.vars[name]
    return info.kind in (:state, :param, :alg) && info.vtype == :real && !(name in assigned)
  end
  local known::Function = (name::String) -> g.vars[name].kind != :alg
  #= the pairs: (x, y) by name => (negated, classes, the elements of x they cover) =#
  local pairs::Dict{Tuple{String, String}, Tuple{Bool, Vector{Int}, Vector{Int}}} = Dict{Tuple{String, String}, Tuple{Bool, Vector{Int}, Vector{Int}}}()
  local conflicting::Set{Tuple{String, String}} = Set{Tuple{String, String}}()
  local idx::Vector{Int} = Int[]
  for (ci, c) in enumerate(g.classes)
    c.nrel == 0 || continue
    local terms::Union{Nothing, Tuple{Ref, Ref, Bool}} = _aliasTerms(c)
    terms === nothing && continue
    local (x::Ref, y::Ref, negated::Bool) = terms
    (x.name != y.name && eligible(x.name) && eligible(y.name) && !(known(x.name) && known(y.name))) || continue
    local ix::VarInfo = g.vars[x.name]
    (ix.dims == g.vars[y.name].dims && length(x.subs) == length(y.subs) &&
     all(k::Int -> irKey(x.subs[k]) == irKey(y.subs[k]), eachindex(x.subs))) || continue
    local covered::Vector{Int} = Int[]
    try
      for j in 1:size(c.domain, 2)
        empty!(idx)
        foreach(s::IR -> push!(idx, _evalInt(g, s, c.domain, j)), x.subs)
        push!(covered, _elementOffset(ix, idx, x.name) + 1)
      end
    catch err
      err isa NotSupported || rethrow(err)
      continue
    end
    local key::Tuple{String, String} = x.name < y.name ? (x.name, y.name) : (y.name, x.name)
    local entry::Tuple{Bool, Vector{Int}, Vector{Int}} = get!(() -> (negated, Int[], Int[]), pairs, key)
    entry[1] == negated || (push!(conflicting, key); continue)
    push!(entry[2], ci)
    append!(entry[3], covered)
  end
  #= the sets: a signed union-find over the pairs that cover their variables, at most one state or parameter each =#
  local parent::Dict{String, String} = Dict{String, String}(); local flip::Dict{String, Bool} = Dict{String, Bool}()
  local hasKnown::Dict{String, Bool} = Dict{String, Bool}()
  local find!::Function = function (v::String)
    haskey(parent, v) || (parent[v] = v; flip[v] = false; hasKnown[v] = known(v))
    local path::Vector{String} = String[]
    local r::String = v
    while parent[r] != r
      push!(path, r)
      r = parent[r]
    end
    #= compressed: each on the path straight to the root, its sign relative to it =#
    for w in reverse(path)
      parent[w] == r && continue
      flip[w] = xor(flip[w], flip[parent[w]])
      parent[w] = r
    end
    return (r, v == r ? false : flip[v])
  end
  local tree::Vector{Tuple{String, String, Bool, Vector{Int}}} = Tuple{String, String, Bool, Vector{Int}}[]
  for key in sort!(collect(keys(pairs)))
    key in conflicting && continue
    local (negated::Bool, classes::Vector{Int}, covered::Vector{Int}) = pairs[key]
    local n::Int = isempty(g.vars[key[1]].dims) ? 1 : prod(g.vars[key[1]].dims)
    (length(covered) == n && sort(covered) == collect(1:n)) || continue
    local (ra::String, fa::Bool) = find!(key[1]); local (rb::String, fb::Bool) = find!(key[2])
    (ra == rb || (hasKnown[ra] && hasKnown[rb])) && continue
    #= a = ±b: the root of b's set under a's, its sign so that b's members keep theirs =#
    parent[rb] = ra
    flip[rb] = xor(fa, fb, negated)
    hasKnown[ra] = hasKnown[ra] || hasKnown[rb]
    push!(tree, (key[1], key[2], negated, classes))
  end
  isempty(tree) && return 0
  #= each set rooted at its state or parameter, else its first algebraic variable by name; the members by a walk from
     there, each computed by the alias equations of the pair it was reached by =#
  local adjacent::Dict{String, Vector{Tuple{String, Bool, Vector{Int}}}} = Dict{String, Vector{Tuple{String, Bool, Vector{Int}}}}()
  for (a, b, negated, classes) in tree
    push!(get!(() -> Tuple{String, Bool, Vector{Int}}[], adjacent, a), (b, negated, classes))
    push!(get!(() -> Tuple{String, Bool, Vector{Int}}[], adjacent, b), (a, negated, classes))
  end
  local members::Dict{String, Vector{String}} = Dict{String, Vector{String}}()
  foreach(v::String -> push!(get!(() -> String[], members, first(find!(v))), v), sort!(collect(keys(adjacent))))
  local alias::Dict{String, Tuple{String, Bool}} = Dict{String, Tuple{String, Bool}}()
  local computing::Vector{Tuple{String, Vector{Int}}} = Tuple{String, Vector{Int}}[]
  for (_, vs) in sort!(collect(members); by = first)
    local k::Union{Nothing, Int} = findfirst(known, vs)
    local root::String = k === nothing ? vs[1] : vs[k]
    local sign::Dict{String, Bool} = Dict{String, Bool}(root => false)
    local queue::Vector{String} = String[root]
    while !isempty(queue)
      local u::String = popfirst!(queue)
      for (v, negated, classes) in adjacent[u]
        haskey(sign, v) && continue
        sign[v] = xor(sign[u], negated)
        alias[v] = (root, sign[v])
        push!(computing, (v, classes))
        push!(queue, v)
      end
    end
  end
  #= each member's alias equations as `member = ±root` (the subscripts both sides had), then every read substituted =#
  local oriented::Dict{Int, EqClass} = Dict{Int, EqClass}()
  for (v, classes) in computing, ci in classes
    local c::EqClass = g.classes[ci]
    local subs::Vector{IR} = IR[_substituted(x, alias) for x in something(_aliasTerms(c))[1].subs]
    local (root::String, negated::Bool) = alias[v]
    local value::Ref = Ref(root, subs, false)
    oriented[ci] = EqClass(Ref(v, subs, false), negated ? Op(:neg, IR[value]) : value, c.nslots, c.domain, c.nrel, c.text)
  end
  local sub::Function = (c::EqClass) -> EqClass(_substituted(c.lhs, alias), _substituted(c.rhs, alias), c.nslots, c.domain, c.nrel, c.text)
  g.classes = EqClass[get(() -> sub(c), oriented, ci) for (ci, c) in enumerate(g.classes)]
  g.whens = WhenClass[WhenClass(_substituted(w.cond, alias),
                                Tuple{Symbol, Ref, IR}[(kind, _substitutedTarget(t, alias), _substituted(e, alias)) for (kind, t, e) in w.body],
                                w.starts, w.nslots, w.domain, w.nrel, w.nsample, w.text) for w in g.whens]
  local subAlg::Function = (a::AlgClass) -> AlgClass(Stmt[_substituted(st, alias) for st in a.body], a.nslots, a.domain, a.nrel, a.nwhen, a.text)
  g.algs = AlgClass[subAlg(a) for a in g.algs]
  g.initAlgs = AlgClass[subAlg(a) for a in g.initAlgs]
  local subAssert::Function = (a::AssertClass) -> AssertClass(_substituted(a.cond, alias), a.warning, a.nslots, a.domain, a.condition, a.message,
                                                              _substituted(a.msg, alias))
  g.asserts = AssertClass[subAssert(a) for a in g.asserts]
  g.initAsserts = AssertClass[subAssert(a) for a in g.initAsserts]
  return length(alias)
end

#= ---------------------------------------------------------------- semiLinear =#

"""
`(x, sa, sb)` of the form `semiLinear(x, sa, sb)` lowers to (`if x >= 0 then x*sa else x*sb`, `toIR`); `nothing` for
another expression.
"""
function _semiLinearForm(e::IR)::Union{Nothing, NTuple{3, IR}}
  (e isa Op && e.op == :if && length(e.args) == 3) || return nothing
  local (cond::IR, kpos::IR, kneg::IR) = (e.args[1], e.args[2], e.args[3])
  (cond isa Op && cond.op == :>= && length(cond.args) == 2 && _isLit(cond.args[2], 0.0)) || return nothing
  (kpos isa Op && kpos.op == :* && length(kpos.args) == 2 && kneg isa Op && kneg.op == :* && length(kneg.args) == 2) ||
    return nothing
  local key::String = irKey(cond.args[1])
  (irKey(kpos.args[1]) == key && irKey(kneg.args[1]) == key) || return nothing
  return (cond.args[1], kpos.args[2], kneg.args[2])
end

"""
An instance of `y = semiLinear(x, sa, sb)` (`_simplifySemiLinear!`): its class and instance, `y` and `x` as written
(slots substituted), whether `y` is its root negated, the slopes oriented to the root (`-y = semiLinear(x, a, b)` is
`y = semiLinear(-x, b, a)`) with their keys (a variable's root element, else the expression).
"""
struct SemiLinearInstance
  class::Int
  instance::Int
  y::IR
  x::IR
  yNegated::Bool
  saKey::String
  sa::IR
  sbKey::String
  sb::IR
end

"""
The instances of one `y` and `x` in chain order, `sa -> s1 -> ... -> sb` (each one's second slope the next one's
first), every intermediate slope a variable, each slope once; `nothing` for another shape (the SimCode pass's
`semiLinearChain`).
"""
function _semiLinearChain(eqs::Vector{SemiLinearInstance})::Union{Nothing, Vector{SemiLinearInstance}}
  length(eqs) >= 2 || return nothing
  local bySa::Dict{String, Int} = Dict{String, Int}()
  for (k, e) in enumerate(eqs)
    haskey(bySa, e.saKey) && return nothing
    bySa[e.saKey] = k
  end
  local sbs::Set{String} = Set{String}(e.sbKey for e in eqs)
  local starts::Vector{Int} = Int[k for (k, e) in enumerate(eqs) if !(e.saKey in sbs)]
  length(starts) == 1 || return nothing
  local chain::Vector{SemiLinearInstance} = SemiLinearInstance[eqs[only(starts)]]
  while length(chain) < length(eqs)
    local next::Int = get(bySa, last(chain).sbKey, 0)
    (next == 0 || !(last(chain).sb isa Ref)) && return nothing
    push!(chain, eqs[next])
  end
  haskey(bySa, last(chain).sbKey) && return nothing
  return chain
end

"""
    _simplifySemiLinear!(g) -> Int

MLS 3.7.2.5, rule 1, per element (on the scalar path the SimCode pass `simplifySemiLinear`): the instances
`y = semiLinear(x, sa, s1)`, `y = semiLinear(x, s1, s2)`, ..., `y = semiLinear(x, sN, sb)` of one `y` and `x` up to
the alias equations as `s1 = if x >= 0 then sa else sb`, `s2 = s1`, ..., `y = semiLinear(x, sa, sb)`, each instance
taken out of its class into one of its own. At `x = 0` the chain holds for any `s1`; an instance matched to `s1` is
solved through `x` (NaN at no flow). The elements are joined through the alias equations `_eliminateAliases!` leaves:
MSL DryAir2's rolled ShortPipes leave `ambient.port.H_flow = shortPipe1.port_a.H_flow`, a scalar and an array
element. Returns the number of chains.
"""
function _simplifySemiLinear!(g::Gen)::Int
  local forms::Vector{Int} = Int[ci for (ci, c) in enumerate(g.classes)
                                 if c.nrel == 0 && (_semiLinearForm(c.lhs) !== nothing || _semiLinearForm(c.rhs) !== nothing)]
  isempty(forms) && return 0
  #= the elements (variable, offset), joined by a signed union-find over the alias equations' instances =#
  local parent::Dict{Tuple{String, Int}, Tuple{String, Int}} = Dict{Tuple{String, Int}, Tuple{String, Int}}()
  local flip::Dict{Tuple{String, Int}, Bool} = Dict{Tuple{String, Int}, Bool}()
  local idx::Vector{Int} = Int[]
  local element::Function = function (r::Ref, c::EqClass, j::Int)
    local info::Union{Nothing, VarInfo} = get(g.vars, r.name, nothing)
    (info === nothing || r.der || length(r.subs) != length(info.dims)) && return nothing
    empty!(idx)
    try
      foreach(s::IR -> push!(idx, _evalInt(g, s, c.domain, j)), r.subs)
      return (r.name, _elementOffset(info, idx, r.name))
    catch err
      err isa NotSupported || rethrow(err)
      return nothing
    end
  end
  local find::Function = function (v::Tuple{String, Int})
    local negated::Bool = false
    while haskey(parent, v)
      negated = xor(negated, flip[v])
      v = parent[v]
    end
    return (v, negated)
  end
  for c in g.classes
    c.nrel == 0 || continue
    local terms::Union{Nothing, Tuple{Ref, Ref, Bool}} = _aliasTerms(c)
    terms === nothing && continue
    for j in 1:size(c.domain, 2)
      local a::Union{Nothing, Tuple{String, Int}} = element(terms[1], c, j)
      local b::Union{Nothing, Tuple{String, Int}} = element(terms[2], c, j)
      (a === nothing || b === nothing) && continue
      local (ra::Tuple{String, Int}, fa::Bool) = find(a); local (rb::Tuple{String, Int}, fb::Bool) = find(b)
      ra == rb && continue
      #= a = ±b: b's root under a's, its sign so that b keeps its own =#
      parent[rb] = ra
      flip[rb] = xor(fa, fb, terms[3])
    end
  end
  #= an expression's root element and sign, if it is a (negated) variable =#
  local signed::Function = function (e::IR, c::EqClass, j::Int)
    local negated::Bool = false
    while e isa Op && e.op == :neg
      negated = !negated
      e = e.args[1]
    end
    e isa Ref || return nothing
    local el::Union{Nothing, Tuple{String, Int}} = element(e, c, j)
    el === nothing && return nothing
    local (root::Tuple{String, Int}, f::Bool) = find(el)
    return (root, xor(f, negated))
  end
  local slopeKey::Function = function (e::IR, c::EqClass, j::Int, written::IR)
    local s::Union{Nothing, Tuple{Tuple{String, Int}, Bool}} = signed(e, c, j)
    return s === nothing ? string("=", irKey(written)) : string(s[2] ? "-" : "+", s[1][1], "#", s[1][2])
  end
  #= the instances by (y's root, x's root, x negated relative to y) =#
  local groups::Dict{Tuple{Tuple{String, Int}, Tuple{String, Int}, Bool}, Vector{SemiLinearInstance}} =
    Dict{Tuple{Tuple{String, Int}, Tuple{String, Int}, Bool}, Vector{SemiLinearInstance}}()
  for ci in forms
    local c::EqClass = g.classes[ci]
    for j in 1:size(c.domain, 2)
      local vals::Dict{Int, Int} = Dict{Int, Int}(k => c.domain[k, j] for k in 1:c.nslots)
      for (ySide, formSide) in ((c.lhs, c.rhs), (c.rhs, c.lhs))
        local form::Union{Nothing, NTuple{3, IR}} = _semiLinearForm(formSide)
        form === nothing && continue
        local y::Union{Nothing, Tuple{Tuple{String, Int}, Bool}} = signed(ySide, c, j)
        local x::Union{Nothing, Tuple{Tuple{String, Int}, Bool}} = signed(form[1], c, j)
        (y === nothing || x === nothing || y[1] == x[1]) && continue
        local (a::IR, b::IR) = y[2] ? (form[3], form[2]) : (form[2], form[3])
        local (aw::IR, bw::IR) = (_substSlots(a, vals), _substSlots(b, vals))
        push!(get!(() -> SemiLinearInstance[], groups, (y[1], x[1], xor(x[2], y[2]))),
              SemiLinearInstance(ci, j, _substSlots(ySide, vals), _substSlots(form[1], vals), y[2],
                                 slopeKey(a, c, j, aw), aw, slopeKey(b, c, j, bw), bw))
        break
      end
    end
  end
  #= each chain's instances out of their classes, its equations as scalar classes of their own =#
  local taken::Dict{Int, Set{Int}} = Dict{Int, Set{Int}}()
  local added::Vector{EqClass} = EqClass[]
  local scalar::Matrix{Int} = Matrix{Int}(undef, 0, 1)
  local text::Function = (e::SemiLinearInstance) -> string("semiLinear chain (MLS 3.7.2.5): ", g.classes[e.class].text)
  #= an intermediate slope the rewrite gives an equation of: an algebraic unknown (not a parameter or a state) =#
  local algebraic::Function = (s::IR) -> s isa Ref && haskey(g.vars, s.name) && g.vars[s.name].kind == :alg
  local nChains::Int = 0
  for key in sort!(collect(keys(groups)))
    local chain::Union{Nothing, Vector{SemiLinearInstance}} = _semiLinearChain(groups[key])
    chain === nothing && continue
    local n::Int = length(chain)
    all(k::Int -> algebraic(chain[k].sb), 1:(n - 1)) || continue
    local (head::SemiLinearInstance, tail::SemiLinearInstance) = (chain[1], chain[n])
    local (sa::IR, sb::IR) = (head.sa, tail.sb)
    #= s1 = if x >= 0 then sa else sb, x the root's: the first instance's, negated with its y =#
    local x1::IR = head.yNegated ? Op(:neg, IR[head.x]) : head.x
    push!(added, EqClass(head.sb, Op(:if, IR[Op(:>=, IR[x1, Lit(0.0)]), sa, sb]), 0, scalar, 0, text(head)))
    for k in 2:(n - 1)
      push!(added, EqClass(chain[k].sb, chain[k - 1].sb, 0, scalar, 0, text(chain[k])))
    end
    #= y = semiLinear(x, sa, sb) as the last instance writes y and x: for a negated y the slopes swapped back =#
    local (p::IR, q::IR) = tail.yNegated ? (sb, sa) : (sa, sb)
    push!(added, EqClass(tail.y, Op(:if, IR[Op(:>=, IR[tail.x, Lit(0.0)]), Op(:*, IR[tail.x, p]), Op(:*, IR[tail.x, q])]),
                         0, scalar, 0, text(tail)))
    foreach(e::SemiLinearInstance -> push!(get!(() -> Set{Int}(), taken, e.class), e.instance), chain)
    nChains += 1
  end
  nChains == 0 && return 0
  local out::Vector{EqClass} = EqClass[]
  for (ci, c) in enumerate(g.classes)
    local js::Union{Nothing, Set{Int}} = get(taken, ci, nothing)
    if js === nothing
      push!(out, c)
      continue
    end
    local keep::Vector{Int} = Int[j for j in 1:size(c.domain, 2) if !(j in js)]
    isempty(keep) || push!(out, EqClass(c.lhs, c.rhs, c.nslots, c.domain[:, keep], c.nrel, c.text))
  end
  g.classes = append!(out, added)
  return nChains
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
  aliased::Set{Tuple{Int, Int}}   #= (instance, unknown) read through two different references (x[i] and x[2] at i = 2) =#
end

"""
The unknown id of a variable element, or 0 (parameters, states read, when-assigned
discretes).
"""
function _unknownOf(g::Gen, info::VarInfo, off::Int, der::Bool)::Int
  info.kind == :alg && return g.nStates + off
  info.kind == :discrete && return get(g.discUnknown, off, 0)
  info.kind == :state && der && return off
  return 0
end

"""
The elements an algorithm instance reads and assigns, walking its statements (for-loops
with their static ranges).
"""
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
  local aliased::Set{Tuple{Int, Int}} = Set{Tuple{Int, Int}}()
  local keyOf::Dict{Int, String} = Dict{Int, String}()
  for (ci, c) in enumerate(g.classes)
    local acc = accesses!(accesses!(Ref[], c.lhs), c.rhs)
    local keys::Vector{String} = String[irKey(r) for r in acc]
    for j in 1:size(c.domain, 2)
      push!(class, ci); push!(col, j)
      empty!(keyOf)
      local firstAccess::Int = length(uid) + 1
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
      for i in firstAccess:length(uid)
        local k::String = get!(keyOf, uid[i], keys[upos[i]])
        k == keys[upos[i]] || push!(aliased, (length(class), uid[i]))
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
  return Elements(class, col, uptr, uid, upos, sptr, sid, outs, nEq, aliased)
end

"""
Maximum matching instances <-> unknowns (greedy, then augmenting paths).
"""
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

"""
Kahn's algorithm in batches: all ready instances of one (class, matched access) at a time,
a key whose remaining instances are all ready first. Returns the batches (consecutive equal
keys merged), the matched access per instance and the producer per unknown.
"""
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
        #= read again: through the same reference (solved for linearly, `_solveFor`), or through another that is the
           same element only at this instance (Elements' aliased: solved by its residual, `_loopFamilyCode`) =#
        pos[r] == 0 && (pos[r] = el.upos[i])
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
  done == N && return (batches, pos, producer, LoopBlocks(Vector{Int}[], zeros(Int, N), Vector{Tuple{Int, Int}}[]))
  return _scheduleWithLoops(el, matchRow, nU, producer, pos)
end

"""
    LoopBlocks(blocks, blockOf, families)

The algebraic loops of an array model: each block's equation instances (a strongly connected set of the instance
graph, ordered by class, matched access and column: its template's order), the block of each instance (0 for none),
and the families' templates (the (class, matched access) of each position; the blocks of one template are solved by
one generated Newton solve, `_loopFamilyCode`).
"""
struct LoopBlocks
  blocks::Vector{Vector{Int}}
  blockOf::Vector{Int}
  families::Vector{Vector{Tuple{Int, Int}}}
end

"""
Blocks of more equation instances than this are solved where the equations reach them by a sparse Newton
(`blockSolve!`, the Jacobian's factorization kept from call to call), smaller ones by a generated Newton solve per
block (`_loopFamilyCode`).
"""
const NEWTON_BLOCK_SIZE = Base.RefValue(8)

"""
Expression nodes per function that the generated module's large functions are split into (`_splitStatements`).
"""
const CHUNK_NODES = Base.RefValue(20_000)

"The number of expression nodes of `x`."
_nodeCount(x)::Int = x isa Expr ? 1 + sum(_nodeCount, x.args; init = 0) : 1

"Every symbol in `x`, into `out`."
function _symbolsIn!(out::Set{Symbol}, x)::Set{Symbol}
  if x isa Symbol
    push!(out, x)
  elseif x isa Expr
    foreach(a -> _symbolsIn!(out, a), x.args)
  end
  return out
end

"The name a generated `local` declares (`local x = e`, `local x::T = e`)."
function _localName(l::Expr)::Symbol
  local x::Any = l.args[1]
  x isa Expr && x.head == :(=) && (x = x.args[1])
  x isa Expr && x.head == :(::) && (x = x.args[1])
  return x
end

"""
    _splitStatements(name, args, passed, paramLocals, stmts; fired) -> (defs, stmts)

A generated function's statements (run under @inbounds) in functions of about `CHUNK_NODES` expression nodes each
(`name_chunk1`, ...), called in order, where they are at least twice that: Julia's compilation of one function grows
faster than its size. MSL EngineV6's module (917k nodes, equations! 445k) took 56 s and 11.7 GB max RSS to compile;
split at 20k nodes 24 s and 3.8 GB, the right-hand side as fast (at 5k nodes 17 % slower). The statements write only
to the arrays. Each part takes the function's arguments `args` and its locals `passed`, declares the parameter views
it reads (of `paramLocals`) and returns its `_fired` (an algorithm's when-statement fired), which the calls OR into
the caller's where it has one (`fired`). No definitions and the statements as they are below the size.
"""
function _splitStatements(name::Symbol, args::Vector{Symbol}, passed::Vector{Symbol}, paramLocals::Vector{Any},
                          stmts::Vector{Any}; fired::Bool)::Tuple{Vector{Expr}, Vector{Any}}
  local limit::Int = CHUNK_NODES[]
  sum(_nodeCount, stmts; init = 0) < 2 * limit && return (Expr[], stmts)
  local parts::Vector{Vector{Any}} = Vector{Any}[Any[]]
  local size::Int = 0
  for st in stmts
    push!(last(parts), st)
    size += _nodeCount(st)
    size >= limit || continue
    push!(parts, Any[])
    size = 0
  end
  isempty(last(parts)) && pop!(parts)
  local defs::Vector{Expr} = Expr[]
  local calls::Vector{Any} = Any[]
  for (k, part) in enumerate(parts)
    local partName::Symbol = Symbol(name, "_chunk", k)
    local used::Set{Symbol} = _symbolsIn!(Set{Symbol}(), Expr(:block, part...))
    local views::Vector{Any} = Any[l for l in paramLocals if _localName(l) in used]
    push!(defs, :(function $partName($(args...), $(passed...))
                    $(views...)
                    local _fired::Bool = false
                    @inbounds begin
                      $(part...)
                    end
                    return _fired
                  end))
    local call::Expr = :($partName($(args...), $(passed...)))
    push!(calls, fired ? :(_fired = $call || _fired) : call)
  end
  return (defs, calls)
end

"""
The large blocks of `loops` (more than `NEWTON_BLOCK_SIZE` instances: loops that couple the elements, solved by
`blockSolve!`); NotSupported for a discrete unknown in one.
"""
function _largeBlocks(g::Gen, loops::LoopBlocks, matchRow::Vector{Int})::BitVector
  local large::BitVector = BitVector(length(b) > NEWTON_BLOCK_SIZE[] for b in loops.blocks)
  for (b, rs) in enumerate(loops.blocks)
    (large[b] && any(r::Int -> matchRow[r] > g.nStates + g.nAlg, rs)) && ns("a discrete unknown in an algebraic loop")
  end
  return large
end

"""
    _explicitDivisors(g, lhs, rhs, pos) -> Union{Nothing, Vector{IR}}

The divisors `_solveFor` divides by when it solves `lhs = rhs` for its access `pos`, where all read parameters and
literals only; nothing where it would divide by a variable or cannot solve (omc's strict tearing: an equation of a
loop is not solved by dividing by a variable, which may be zero; `_tearBlock` also checks the divisors' values).
"""
function _explicitDivisors(g::Gen, lhs::IR, rhs::IR, pos::Int)::Union{Nothing, Vector{IR}}
  local acc::Vector{Ref} = accesses!(accesses!(Ref[], lhs), rhs)
  local key::String = irKey(acc[pos])
  if count(r::Ref -> irKey(r) == key, acc) > 1
    local l::Union{Nothing, Tuple{IR, IR}} = _affine(lhs, key)
    local r::Union{Nothing, Tuple{IR, IR}} = _affine(rhs, key)
    (l === nothing || r === nothing) && return nothing
    local coefficient::IR = _sub(l[1], r[1])
    (_isLit(coefficient, 0.0) || !_parametersOnly(g, coefficient)) && return nothing
    return _isLit(coefficient, 1.0) ? IR[] : IR[coefficient]
  end
  local divisors::Vector{IR} = IR[]
  local nl::Int = _nAccesses(lhs)
  local (e::IR, k::Int) = pos <= nl ? (lhs, pos) : (rhs, pos - nl)
  while true
    e isa Ref && k == 1 && return divisors
    e isa Op || return nothing
    local j::Int = 0; local before::Int = 0
    for (ai, arg) in enumerate(e.args)
      local na::Int = _nAccesses(arg)
      if k <= before + na
        j = ai
        break
      end
      before += na
    end
    if e.op == :* && length(e.args) == 2
      _parametersOnly(g, e.args[3 - j]) || return nothing
      push!(divisors, e.args[3 - j])
    elseif !(e.op in (:+, :neg) || (e.op == :- && length(e.args) == 2) || (e.op == :/ && j == 1))
      return nothing
    end
    e = e.args[j]
    k -= before
  end
end

"""
Whether the divisors of an explicit solve (`_explicitDivisors`) are all nonzero at the instance column `col` of the
class `c` (with the parameters' values; one that cannot be evaluated counts as zero).
"""
function _nonzeroDivisors(g::Gen, divisors::Vector{IR}, c::EqClass, col::Int)::Bool
  for d in divisors
    local value::Any = try
      _evalStatic(g, d, c.domain, col; structural = false)
    catch err
      err isa NotSupported || rethrow(err)
      0.0
    end
    (value isa Real && !iszero(value)) || return false
  end
  return true
end

"""
    _largestCoefficient(g, c, col, acc, access, others) -> Bool

Whether solving the equation class `c` at the instance column `col` for its access `access` divides by a coefficient at
least as large as those of the other accesses `others` to elements of the same variable (literal or parameter
coefficients, the affine forms of `lhs - rhs`; false where one has no such coefficient, a nonlinear term): tearing a
tridiagonal loop to one iteration variable solved its equations `-v[i - 1] + 3 * v[i] - v[i + 1] = x[i]` for
`v[i + 1]`, amplifying round-off threefold per element, and `... + 0.1 * v[i]^3` ran into NaN (the same variable, so
the magnitudes compare; another variable's coefficient has other units).
"""
function _largestCoefficient(g::Gen, c::EqClass, col::Int, acc::Vector{Ref}, access::Int, others::Vector{Int})::Bool
  local name::String = acc[access].name
  local same::Vector{Int} = Int[k for k in others if acc[k].name == name && irKey(acc[k]) != irKey(acc[access])]
  isempty(same) && return true
  local residual::IR = Op(:-, IR[c.lhs, c.rhs])
  local magnitude::Function = function (k::Int)
    local form::Union{Nothing, Tuple{IR, IR}} = _affine(residual, irKey(acc[k]))
    (form === nothing || !_parametersOnly(g, form[1])) && return nothing
    local value::Any = try
      _evalStatic(g, form[1], c.domain, col; structural = false)
    catch err
      err isa NotSupported || rethrow(err)
      nothing
    end
    return value isa Real ? abs(Float64(value)) : nothing
  end
  local own::Union{Nothing, Float64} = magnitude(access)
  own === nothing && return true
  for k in same
    local m::Union{Nothing, Float64} = magnitude(k)
    (m === nothing || m > own) && return false
  end
  return true
end

"""
    TornBlock(tears, residuals, batches, pattern)

A large block torn (`_tearBlock`): its iteration variables (unknown ids), its residual instances (positions in the
block, as many), the batches of the others in causal order ((class, the access each is solved for), positions in the
block), each solved explicitly for that access's unknown given the iteration variables, and the residuals' sparsity in
the iteration variables (rows, columns).
"""
struct TornBlock
  tears::Vector{Int}
  residuals::Vector{Int}
  batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}}
  pattern::Tuple{Vector{Int}, Vector{Int}}
end

"""
    _tearBlock(g, el, block, matchRow) -> TornBlock

A large block torn as omc tears (greedily, each instance solved for the unknown it determines last, not its matched
one): while an instance has one unknown of the block left that it solves explicitly (`_explicitDivisors`, a divisor
nonzero at the instance, the largest coefficient among the elements of its variable it reads, `_largestCoefficient`;
not one it reads through two references), it is solved for it; when none has, the unknown whose being known lets the
most instances be solved so is torn (an iteration variable). The instances left are
the residuals. Batches by level (an instance after those it reads), then (class, access).
"""
function _tearBlock(g::Gen, el::Elements, block::Vector{Int}, matchRow::Vector{Int})::TornBlock
  local n::Int = length(block)
  local index::Dict{Int, Int} = Dict{Int, Int}(matchRow[r] => k for (k, r) in enumerate(block))
  #= each instance's unknowns of the block (by index) and the access of each =#
  local reads::Vector{Vector{Tuple{Int, Int}}} = Vector{Tuple{Int, Int}}[Tuple{Int, Int}[] for _ in 1:n]
  local readers::Vector{Vector{Int}} = Vector{Int}[Int[] for _ in 1:n]
  for (e, r) in enumerate(block), i in el.uptr[r]:(el.uptr[r + 1] - 1)
    local v::Int = get(index, el.uid[i], 0)
    (v == 0 || any(x::Tuple{Int, Int} -> x[1] == v, reads[e])) && continue
    push!(reads[e], (v, el.upos[i]))
    push!(readers[v], e)
  end
  local divisors::Dict{Tuple{Int, Int}, Union{Nothing, Vector{IR}}} = Dict{Tuple{Int, Int}, Union{Nothing, Vector{IR}}}()
  local accessesOf::Dict{Int, Vector{Ref}} = Dict{Int, Vector{Ref}}()
  local solves::Function = function (e::Int, v::Int, access::Int)
    local r::Int = block[e]
    (r, matchRow[block[v]]) in el.aliased && return false
    local c::EqClass = g.classes[el.class[r]]
    local ds::Union{Nothing, Vector{IR}} = get!(() -> _explicitDivisors(g, c.lhs, c.rhs, access), divisors, (el.class[r], access))
    (ds !== nothing && _nonzeroDivisors(g, ds, c, el.col[r])) || return false
    local acc::Vector{Ref} = get!(() -> accesses!(accesses!(Ref[], c.lhs), c.rhs), accessesOf, el.class[r])
    return _largestCoefficient(g, c, el.col[r], acc, access, Int[x[2] for x in reads[e] if x[1] != v])
  end
  local known::BitVector = falses(n); local used::BitVector = falses(n)
  local open::Vector{Int} = Int[length(reads[e]) for e in 1:n]
  local order::Vector{Tuple{Int, Int, Int}} = Tuple{Int, Int, Int}[]   #= (instance, unknown, access) in causal order =#
  local tears::Vector{Int} = Int[]
  local queue::Vector{Int} = Int[e for e in 1:n if open[e] == 1]
  local learn!::Function = function (v::Int)
    known[v] = true
    for e in readers[v]
      open[e] -= 1
      (open[e] == 1 && !used[e]) && push!(queue, e)
    end
  end
  while true
    while !isempty(queue)
      local e::Int = popfirst!(queue)
      (used[e] || open[e] != 1) && continue
      local (v::Int, access::Int) = reads[e][findfirst(x::Tuple{Int, Int} -> !known[x[1]], reads[e])]
      solves(e, v, access) || continue
      used[e] = true
      push!(order, (e, v, access))
      learn!(v)
    end
    all(known) && break
    #= the unknown that leaves the most instances with one unknown, then the most read =#
    local best::Int = 0; local bestScore::Tuple{Int, Int} = (-1, -1)
    for v in 1:n
      known[v] && continue
      local unlocks::Function = function (e::Int)
        (used[e] || open[e] != 2) && return false
        local other::Union{Nothing, Int} = findfirst(x::Tuple{Int, Int} -> !known[x[1]] && x[1] != v, reads[e])
        return other !== nothing && solves(e, reads[e][other]...)
      end
      local score::Tuple{Int, Int} = (count(unlocks, readers[v]), count(e::Int -> !used[e], readers[v]))
      score > bestScore && ((best, bestScore) = (v, score))
    end
    push!(tears, best)
    learn!(best)
  end
  local residuals::Vector{Int} = Int[e for e in 1:n if !used[e]]
  #= as many as iteration variables: each instance solved determines one unknown, each other unknown is torn =#
  #= levels: an instance after the instances that solve what it reads =#
  local level::Vector{Int} = zeros(Int, n)   #= by unknown: 0 for an iteration variable =#
  local byKey::Dict{Tuple{Int, Int, Int}, Vector{Int}} = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
  for (e, v, access) in order
    level[v] = 1 + maximum((level[x] for (x, _) in reads[e] if x != v); init = 0)
    push!(get!(() -> Int[], byKey, (level[v], el.class[block[e]], access)), e)
  end
  local batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}} =
    Tuple{Tuple{Int, Int}, Vector{Int}}[((k[2], k[3]), byKey[k]) for k in sort!(collect(keys(byKey)))]
  #= the residuals' sparsity: the iteration variables each reads, directly or through the instances solved before =#
  local dep::Vector{BitSet} = BitSet[BitSet() for _ in 1:n]
  for (t, v) in enumerate(tears)
    push!(dep[v], t)
  end
  for (e, v, _) in order, (x, _) in reads[e]
    x == v || union!(dep[v], dep[x])
  end
  local I::Vector{Int} = Int[]; local J::Vector{Int} = Int[]
  for (t, e) in enumerate(residuals)
    local cols::BitSet = BitSet()
    foreach(x::Tuple{Int, Int} -> union!(cols, dep[x[1]]), reads[e])
    for j in cols
      push!(I, t); push!(J, j)
    end
  end
  return TornBlock(Int[matchRow[block[v]] for v in tears], residuals, batches, (I, J))
end

"The family of each block of `loops` (by its template)."
function _blockFamilies(el::Elements, pos::Vector{Int}, loops::LoopBlocks)::Vector{Int}
  local byTemplate::Dict{Vector{Tuple{Int, Int}}, Int} = Dict{Vector{Tuple{Int, Int}}, Int}(t => f for (f, t) in enumerate(loops.families))
  return Int[byTemplate[Tuple{Int, Int}[(el.class[r], pos[r]) for r in b]] for b in loops.blocks]
end

"""
The strongly connected components of a graph (successor lists; Tarjan's algorithm with an explicit stack): the
component of each node.
"""
function _components(succ::Vector{Vector{Int}})::Vector{Int}
  local n::Int = length(succ)
  local index::Vector{Int} = zeros(Int, n); local low::Vector{Int} = zeros(Int, n); local onStack::BitVector = falses(n)
  local comp::Vector{Int} = zeros(Int, n)
  local stack::Vector{Int} = Int[]; local counter::Int = 0; local ncomp::Int = 0
  local work::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[]
  for v0 in 1:n
    index[v0] == 0 || continue
    push!(work, (v0, 1))
    while !isempty(work)
      local (v::Int, i::Int) = work[end]
      if i == 1
        counter += 1; index[v] = counter; low[v] = counter
        push!(stack, v); onStack[v] = true
      end
      local descended::Bool = false
      local k::Int = i
      while k <= length(succ[v])
        local w::Int = succ[v][k]
        k += 1
        if index[w] == 0
          work[end] = (v, k)
          push!(work, (w, 1))
          descended = true
          break
        elseif onStack[w]
          low[v] = min(low[v], index[w])
        end
      end
      descended && continue
      pop!(work)
      if low[v] == index[v]
        ncomp += 1
        while true
          local w::Int = pop!(stack)
          onStack[w] = false
          comp[w] = ncomp
          w == v && break
        end
      end
      isempty(work) || (low[work[end][1]] = min(low[work[end][1]], low[v]))
    end
  end
  return comp
end

"""
The schedule of a model with algebraic loops (`_schedule`'s, where Kahn's algorithm leaves instances in cycles): the
strongly connected sets of instances blocks (`LoopBlocks`), each solved as one; the blocks of a template batched by
family (key (0, family), the batch listing block numbers), the other instances as `_schedule` batches them. Not an
algorithm or a discrete unknown in a loop.
"""
function _scheduleWithLoops(el::Elements, matchRow::Vector{Int}, nU::Int, producer::Vector{Int}, pos::Vector{Int})
  local N::Int = length(el.class)
  local succ::Vector{Vector{Int}} = [Int[] for _ in 1:N]
  for r in 1:N, i in el.uptr[r]:(el.uptr[r + 1] - 1)
    local u::Int = el.uid[i]
    u == matchRow[r] && continue
    local q::Int = producer[u]
    (q == 0 || q == r) && continue
    push!(succ[q], r)
  end
  local comp::Vector{Int} = _components(succ)
  local members::Dict{Int, Vector{Int}} = Dict{Int, Vector{Int}}()
  foreach(r::Int -> push!(get!(() -> Int[], members, comp[r]), r), 1:N)
  local blocks::Vector{Vector{Int}} = Vector{Int}[]
  local blockOf::Vector{Int} = zeros(Int, N)
  for c in sort!(collect(keys(members)))
    local rs::Vector{Int} = members[c]
    length(rs) > 1 || continue
    any(r::Int -> el.class[r] < 0, rs) && ns("an algorithm in an algebraic loop")
    sort!(rs; by = r::Int -> (el.class[r], pos[r], el.col[r]))
    push!(blocks, rs)
    foreach(r::Int -> (blockOf[r] = length(blocks)), rs)
  end
  local families::Vector{Vector{Tuple{Int, Int}}} = Vector{Tuple{Int, Int}}[]
  local familyOf::Dict{Vector{Tuple{Int, Int}}, Int} = Dict{Vector{Tuple{Int, Int}}, Int}()
  local blockFamily::Vector{Int} = Int[]
  for b in blocks
    local t::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[(el.class[r], pos[r]) for r in b]
    push!(blockFamily, get!(() -> (push!(families, t); length(families)), familyOf, t))
  end
  #= Kahn's algorithm on the condensed graph: an instance, or a block (node N + block) =#
  local node::Function = (r::Int) -> blockOf[r] > 0 ? N + blockOf[r] : r
  local nNodes::Int = N + length(blocks)
  local indeg::Vector{Int} = zeros(Int, nNodes)
  local dependents::Vector{Vector{Int}} = [Int[] for _ in 1:nNodes]
  local seen::Set{Tuple{Int, Int}} = Set{Tuple{Int, Int}}()
  for q in 1:N, r in succ[q]
    local (a::Int, b::Int) = (node(q), node(r))
    (a == b || (a, b) in seen) && continue
    push!(seen, (a, b))
    push!(dependents[a], b)
    indeg[b] += 1
  end
  local live::Vector{Int} = Int[n for n in 1:nNodes if n > N || blockOf[n] == 0]
  local key::Function = (n::Int) -> n <= N ? (el.class[n], pos[n]) : (0, blockFamily[n - N])
  local queues::Dict{Tuple{Int, Int}, Vector{Int}} = Dict{Tuple{Int, Int}, Vector{Int}}()
  local remaining::Dict{Tuple{Int, Int}, Int} = Dict{Tuple{Int, Int}, Int}()
  for n in live
    remaining[key(n)] = get(remaining, key(n), 0) + 1
    indeg[n] == 0 && push!(get!(queues, key(n), Int[]), n)
  end
  local batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}} = Tuple{Tuple{Int, Int}, Vector{Int}}[]
  local done::Int = 0
  while true
    local best::Union{Nothing, Tuple{Int, Int}} = nothing
    local bestFull::Bool = false
    for (k, q) in queues
      isempty(q) && continue
      local full::Bool = length(q) == remaining[k]
      if best === nothing || (full && !bestFull) ||
         (full == bestFull && (length(q) > length(queues[best]) || (length(q) == length(queues[best]) && k < best)))
        best = k
        bestFull = full
      end
    end
    best === nothing && break
    local batch::Vector{Int} = queues[best]
    queues[best] = Int[]
    remaining[best] -= length(batch)
    done += length(batch)
    local items::Vector{Int} = best[1] == 0 ? Int[n - N for n in batch] : batch
    if !isempty(batches) && batches[end][1] == best
      append!(batches[end][2], items)
    else
      push!(batches, (best, items))
    end
    for n in batch, d in dependents[n]
      indeg[d] -= 1
      indeg[d] == 0 && push!(get!(queues, key(d), Int[]), d)
    end
  end
  done == length(live) || ns("algebraic loop through the blocks ($(length(live) - done) left)")
  return (batches, pos, producer, LoopBlocks(blocks, blockOf, families))
end

"""
The states every equation instance depends on, through the algebraic unknowns it reads
(in schedule order); nothing when that would be large.
"""
function _stateDeps(el::Elements, matchRow, batches, producer, loops::LoopBlocks, nStates::Int)
  local N = length(el.class)
  local deps = Vector{Vector{Int}}(undef, N)
  local total = 0
  for ((ci, _), batch) in batches
    if ci == 0
      #= a block: the states its instances read, through the unknowns of others (solved together) =#
      for b in batch
        local rs::Vector{Int} = loops.blocks[b]
        local s::Set{Int} = Set{Int}()
        for r in rs
          union!(s, el.sid[el.sptr[r]:(el.sptr[r + 1] - 1)])
          for i in el.uptr[r]:(el.uptr[r + 1] - 1)
            local q::Int = producer[el.uid[i]]
            (q == 0 || loops.blockOf[q] == b) && continue
            union!(s, deps[q])
          end
        end
        local d::Vector{Int} = sort!(collect(s))
        foreach(r::Int -> (deps[r] = d), rs)
        total += length(d) * length(rs)
        total > 20_000_000 && return nothing
      end
      continue
    end
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

"""
State dependencies of every state derivative (the Jacobian's rows).
"""
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

"""
The initial equations as a square system: each instance matched to a state element that
is not fixed and that it depends on (directly or through the algebraic unknowns); free
states no initial equation reaches keep their start values (as omc). Returns the matched
states, the residual code and the number of residuals.
"""
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

"""
The equation lhs = rhs solved for its access number `pos` (accesses numbered over lhs,
then rhs): the access and the expression it equals.
"""
function _solveFor(lhs::IR, rhs::IR, pos::Int, text::String)
  local acc::Vector{Ref} = accesses!(accesses!(Ref[], lhs), rhs)
  local key::String = irKey(acc[pos])
  count(r::Ref -> irKey(r) == key, acc) > 1 && return _solveLinear(lhs, rhs, acc[pos], text)
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

"""
`x` as `a * u + b` for the access `u` (by its key), `a` and `b` free of it: (a, b); `nothing` where `x` is not affine
in it (through +, -, negation, a product of one factor that reads it, a quotient of a numerator that does).
"""
function _affine(x::IR, key::String)::Union{Nothing, Tuple{IR, IR}}
  any(r::Ref -> irKey(r) == key, accesses!(Ref[], x)) || return (Lit(0.0), x)
  x isa Ref && return (Lit(1.0), Lit(0.0))
  x isa Op || return nothing
  local parts::Vector{Tuple{IR, IR}} = Tuple{IR, IR}[]
  if x.op in (:+, :-, :neg)
    for a in x.args
      local ab::Union{Nothing, Tuple{IR, IR}} = _affine(a, key)
      ab === nothing && return nothing
      push!(parts, ab)
    end
    x.op == :neg && return (_neg(parts[1][1]), _neg(parts[1][2]))
    x.op == :- && length(parts) == 2 && return (_sub(parts[1][1], parts[2][1]), _sub(parts[1][2], parts[2][2]))
    x.op == :- && return nothing
    return (foldl(_add, first.(parts)), foldl(_add, last.(parts)))
  elseif x.op == :*
    local reading::Vector{Int} = findall(a::IR -> any(r::Ref -> irKey(r) == key, accesses!(Ref[], a)), x.args)
    length(reading) == 1 || return nothing
    local abFactor::Union{Nothing, Tuple{IR, IR}} = _affine(x.args[reading[1]], key)
    abFactor === nothing && return nothing
    local factor::IR = foldl(_mul, IR[a for (i, a) in enumerate(x.args) if i != reading[1]])
    return (_mul(factor, abFactor[1]), _mul(factor, abFactor[2]))
  elseif x.op == :/ && length(x.args) == 2 && !any(r::Ref -> irKey(r) == key, accesses!(Ref[], x.args[2]))
    local abNumerator::Union{Nothing, Tuple{IR, IR}} = _affine(x.args[1], key)
    abNumerator === nothing && return nothing
    return (Op(:/, IR[abNumerator[1], x.args[2]]), Op(:/, IR[abNumerator[2], x.args[2]]))
  end
  return nothing
end

_isLit(x::IR, v::Float64)::Bool = x isa Lit && x.v isa Real && x.v == v
_add(a::IR, b::IR)::IR = _isLit(a, 0.0) ? b : _isLit(b, 0.0) ? a : Op(:+, IR[a, b])
_sub(a::IR, b::IR)::IR = _isLit(b, 0.0) ? a : _isLit(a, 0.0) ? _neg(b) : Op(:-, IR[a, b])
_neg(a::IR)::IR = _isLit(a, 0.0) ? a : (a isa Lit && a.v isa Real) ? Lit(-a.v) : Op(:neg, IR[a])
_mul(a::IR, b::IR)::IR = (_isLit(a, 0.0) || _isLit(b, 0.0)) ? Lit(0.0) : _isLit(a, 1.0) ? b : _isLit(b, 1.0) ? a :
  (a isa Lit && b isa Lit && a.v isa Real && b.v isa Real) ? Lit(a.v * b.v) : Op(:*, IR[a, b])

"""
lhs = rhs, which reads the access `target` more than once, solved for it where it is affine in it (`_affine`): the
access and `-b / a`.
"""
function _solveLinear(lhs::IR, rhs::IR, target::Ref, text::String)
  local key::String = irKey(target)
  local l::Union{Nothing, Tuple{IR, IR}} = _affine(lhs, key)
  local r::Union{Nothing, Tuple{IR, IR}} = _affine(rhs, key)
  (l === nothing || r === nothing) && ns("cannot solve for an unknown it reads nonlinearly in: $(text)")
  local a::IR = _sub(l[1], r[1])
  _isLit(a, 0.0) && ns("an unknown whose coefficient vanishes in: $(text)")
  local b::IR = _sub(r[2], l[2])
  return (target, _isLit(a, 1.0) ? b : Op(:/, IR[b, a]))
end

"""
    _loopFamilyCode(g, f, template, blocks, el, matchRow, a0, relBases) -> Tuple{Vector{Any}, Expr}

The solve of the loop family `f` (`LOOPFAMILY_f!`): its tables (each member's instance columns and unknowns, a guess
per unknown: the last solution, at first the start values) and the body that solves the members it is called for,
each by Newton's method on the residuals of its instances (lhs - rhs, by the template's classes at the member's
columns), the unknowns set in place (a state derivative in du, an algebraic variable in a), the Jacobian by forward
differences; the last solution kept as the next guess (in Float64 solves); residuals or a Jacobian not finite, or a
singular Jacobian, make the unknowns NaN (the solver rejects the step). Not relations in a loop (events), nor a discrete unknown.
"""
function _loopFamilyCode(g::Gen, f::Int, template::Vector{Tuple{Int, Int}}, blocks::Vector{Vector{Int}}, el::Elements,
                         matchRow::Vector{Int}, a0::Vector{Float64}, relBases::Vector{Int})::Tuple{Vector{Any}, Expr}
  local k::Int = length(template); local M::Int = length(blocks)
  local nS::Int = g.nStates
  local cols::Matrix{Int} = Matrix{Int}(undef, k, M); local ids::Matrix{Int} = Matrix{Int}(undef, k, M)
  local guess::Matrix{Float64} = Matrix{Float64}(undef, k, M)
  for (m, b) in enumerate(blocks), j in 1:k
    local r::Int = b[j]
    local u::Int = matchRow[r]
    u <= nS + g.nAlg || ns("a discrete unknown in an algebraic loop")
    cols[j, m] = el.col[r]; ids[j, m] = u
    guess[j, m] = u <= nS ? 0.0 : a0[u - nS]
  end
  local (C::Symbol, I::Symbol, G::Symbol) = (Symbol("LOOP_COLS_", f), Symbol("LOOP_IDS_", f), Symbol("LOOP_GUESS_", f))
  local residuals::Function = function (rv::Symbol)
    local parts::Vector{Any} = Any[]
    for (j, (ci, _)) in enumerate(template)
      local c::EqClass = g.classes[ci]
      c.nrel == 0 || ns("relations in an algebraic loop: $(c.text)")
      g.relBase = relBases[ci]; g.relCount = c.nrel
      local sets::Vector{Any} = Any[:(_col = $C[$j, _m]); Any[:($(Symbol("s", q)) = $(c.domain)[$q, _col]) for q in 1:c.nslots]]
      push!(parts, Expr(:let, Expr(:block, sets...), :($rv[$j] = $(jl(g, c.lhs)) - $(jl(g, c.rhs)))))
    end
    return Expr(:block, parts...)
  end
  local set::Expr = :(for _j in 1:$k
    local _id::Int = $I[_j, _m]
    if _id <= $nS
      du[_id] = _x[_j]
    else
      a[_id - $nS] = _x[_j]
    end
  end)
  local code::Expr = quote
    local _x::AbstractVector = similar(a, $k); local _r::AbstractVector = similar(a, $k)
    local _r2::AbstractVector = similar(a, $k); local _J::AbstractMatrix = similar(a, $k, $k)
    for _m in members
      for _j in 1:$k
        _x[_j] = $G[_j, _m]
      end
      for _it in 1:50
        $set
        $(residuals(:_r))
        for _j in 1:$k
          local _xj::Real = _x[_j]
          local _h::Real = 1.4901161193847656e-8 * max(1.0, abs(_xj))
          _x[_j] = _xj + _h
          $set
          $(residuals(:_r2))
          for _i in 1:$k
            _J[_i, _j] = (_r2[_i] - _r[_i]) / _h
          end
          _x[_j] = _xj
        end
        #= residuals or a Jacobian not finite, or a singular one (a trial point of the solver where the equations
           overflow, or a junction's derivative underflows: MSL DifferenceAmplifier's transistors): the unknowns NaN, so
           the step is rejected =#
        if !(all(isfinite, _r) && all(isfinite, _J))
          fill!(_x, NaN)
          break
        end
        local _F = LinearAlgebra.lu(_J; check = false)
        if !LinearAlgebra.issuccess(_F)
          fill!(_x, NaN)
          break
        end
        local _dx::AbstractVector = _F \ _r
        local _step::Real = 0.0
        for _j in 1:$k
          _x[_j] -= _dx[_j]
          _step = max(_step, abs(_dx[_j]) / (1.0 + abs(_x[_j])))
        end
        _step <= 1.0e-12 && break
      end
      $set
      #= the solution as the next guess, not a failed solve's NaN =#
      if eltype(a) === Float64 && all(isfinite, _x)
        for _j in 1:$k
          $G[_j, _m] = _x[_j]
        end
      end
    end
  end
  return (Any[:(const $C = $cols), :(const $I = $ids), :(const $G = $guess)], code)
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
        info.var isa IRVariable && ns("parameter $(x.name) of a type the array path does not pass")
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
    #= An algebraic variable's value before the instant is not kept: its current value was
       read (`peak = pre(u)` at u's step took the value after it, OpenModelica the one
       before); the scalarized path keeps it. =#
    info.kind == :alg && ns("pre() of the algebraic variable $(x.ref.name)")
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

"""
Statement code of an algorithm; when-statements use wcond from `condBase` (nwhen per
instance) and run their bodies only when fire (event iteration).
"""
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

"""
A loop over the domain columns `cols` of a class running `body` with the slots (and _col,
the column) bound: an affine range when the columns allow it, else a table.
"""
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
_elementNames(v::IRVariable)::Vector{String} = v.names

"""
Whether the code may not change the parameter of variable `v` at a simulation (a structural parameter); an OMJLIR
model's parameters are structural only where the code reads them so (Gen's `structural`).
"""
_structuralParameter(v::F.Variable)::Bool = F.variability(v) <= F.Variability.STRUCTURAL_PARAMETER
_structuralParameter(::IRVariable)::Bool = false

#= ---------------------------------------------------------------- parameters =#

"""
The parameters a binding reads (of a variable's: an OMJLIR variable's binding, `_bindingDepsIR`).
"""
_bindingDeps(g::Gen, v::F.Variable)::Vector{String} = _bindingDeps(g, v.binding)
_bindingDeps(g::Gen, v::IRVariable)::Vector{String} = _bindingDepsIR(g, v.variable)
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

"""
The parameters the equations read and, transitively, those their bindings read, in
dependency order, each with its binding as code over the others (nothing when the
binding is not expressible here: the value then cannot follow a changed dependency).
"""
function _parameterRules(g::Gen)
  local order = String[]
  local state = Dict{String, Int}()   #= 1 visiting, 2 done =#
  local visit
  visit = function (nm)
    get(state, nm, 0) == 2 && return
    get(state, nm, 0) == 1 && ns("cyclic parameter bindings at $(nm)")
    state[nm] = 1
    for d in _bindingDeps(g, g.vars[nm].var)
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
    local deps::Vector{String} = _bindingDeps(g, info.var)
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
  info.var isa IRVariable && return _bindingCodeIR(g, info)
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
function generateArrayODECode(fm::F.FlatModel, modelName::String; functions::Union{Nothing, FunctionList} = nothing)
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

"""
Whether generated code raises an error (a function's assert: Base.error).
"""
function _containsErrorCall(@nospecialize(x))::Bool
  x isa AbstractVector && return any(_containsErrorCall, x)
  x isa Expr || return false
  x.head == :call && (x.args[1] == :error || x.args[1] == :(Base.error)) && return true
  return any(_containsErrorCall, x.args)
end

"""
The model's Modelica functions as Julia functions of the module (the MTK path's code
generation: AlgorithmicCodeGeneration.generateFunctions),
each a module constant named as the frontend's function list names it (dots as _), so the
equations and the functions' calls of each other find it. Returns (definitions, outputs).
"""
function _functionDefinitions(functions::Union{Nothing, FunctionList})
  local defs = Any[]; local outs = Dict{String, Int}()
  (functions === nothing || isempty(functions)) && return (defs, outs)
  local mfs = try
    _OMBACKEND.SimCode.simFunctions(_OMBACKEND.LowerFlatModel.lowerFunctions(functions))
  catch err
    ns("the model's functions: " * first(sprint(showerror, err), 160))
  end
  local (exprs, names) = try
    #= closures of the model module: its functions are its constants, not the wrappers a
       compiled body (compiledFunction) would call =#
    _OMBACKEND.CodeGeneration.AlgorithmicCodeGeneration.generateFunctions(mfs; compiled = false)
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

function _generate(fm::F.FlatModel, modelName::String; functions::Union{Nothing, FunctionList} = nothing)::Expr
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
                -1000, false, Dict{Int, Int}(), AlgClass[], String[], AssertClass[], functionOutputs, Dict{String, String}())
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
  return _emitModule(g, modelName, functionDefs, u0, d0, a0, fixedMask, stateNamesOrdered, algNamesOrdered,
                     discreteNamesOrdered)
end

"""
The module of an array model from its variables and classes (`g`, from a front half: the flat model's `_generate`,
the OMJLIR model's `_generateIR`), its Modelica functions' definitions, the initial values of the states, discrete
variables and algebraic variables (row-major), the states' fixed attributes and the element names: the matching,
the schedule, the initialization and the code.
"""
function _emitModule(g::Gen, modelName::String, functionDefs::Vector{Any}, u0::Vector{Float64}, d0::Vector{Float64},
                     a0::Vector{Float64}, fixedMask::Vector{Bool}, stateNamesOrdered::Vector{String},
                     algNamesOrdered::Vector{String}, discreteNamesOrdered::Vector{String})::Expr
  g.classes = _reroll(g.classes)
  g.classes = EqClass[EqClass(_folded(c.lhs), _folded(c.rhs), c.nslots, c.domain, c.nrel, c.text) for c in g.classes]
  g.initClasses = EqClass[EqClass(_folded(c.lhs), _folded(c.rhs), c.nslots, c.domain, c.nrel, c.text) for c in g.initClasses]
  local nAliases::Int = _eliminateAliases!(g)
  _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE") && println("[arraycode] ", nAliases, " alias variables read through their roots")
  local nChains::Int = _simplifySemiLinear!(g)
  (nChains > 0 && _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE")) && println("[arraycode] ", nChains, " semiLinear chain(s) rewritten (MLS 3.7.2.5)")
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
  local (batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}}, pos::Vector{Int}, producer::Vector{Int}, loops::LoopBlocks) =
    _schedule(el, matchRow, nU)
  if _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE")
    local blockSizes::Dict{Int, Int} = Dict{Int, Int}()
    foreach(b::Vector{Int} -> (blockSizes[length(b)] = get(blockSizes, length(b), 0) + 1), loops.blocks)
    println("[arraycode] ", length(g.classes), " classes, ", length(el.class), " instances; loop blocks (instances => count): ",
            sort!(collect(blockSizes)), "; ", length(loops.families), " families")
  end
  #= algebraic loops in the array code only where it keeps arrays (a class of more than one instance): a model without
     arrays gains nothing from it, and the scalar path solves its loops (tearing) with its initialization and events =#
  local hasArrays::Bool = any(c::EqClass -> size(c.domain, 2) > 1, g.classes)
  (isempty(loops.blocks) || hasArrays) || ns("algebraic loop in a model without arrays (the scalar path solves it)")
  local large::BitVector = _largeBlocks(g, loops, matchRow)
  local deps::Union{Nothing, Vector{Vector{Int}}} = _stateDeps(el, matchRow, batches, producer, loops, g.nStates)
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
  local neededFrom::Function = function (roots::Vector{Int})
    local needed = falses(length(el.class))
    local stack::Vector{Int} = copy(roots)
    foreach(r -> (needed[r] = true), stack)
    while !isempty(stack)
      local r = pop!(stack)
      #= a block is solved whole =#
      if loops.blockOf[r] > 0
        for q in loops.blocks[loops.blockOf[r]]
          needed[q] && continue
          needed[q] = true
          push!(stack, q)
        end
      end
      for i in el.uptr[r]:(el.uptr[r + 1] - 1)
        local u = el.uid[i]
        (u > g.nStates + g.nAlg || u == matchRow[r] || u in el.outs[r]) && continue
        local q = producer[u]
        (q > 0 && !needed[q]) || continue
        needed[q] = true
        push!(stack, q)
      end
    end
    return needed
  end
  local needed = neededFrom(Int[r for r in eachindex(el.class) if 0 < matchRow[r] <= g.nStates])
  #= likewise what the relations need: the crossing functions (crossings!, without when-equations) =#
  local relNeeded::BitVector = neededFrom(Int[r for r in eachindex(el.class)
                                             if el.class[r] > 0 ? g.classes[el.class[r]].nrel > 0 :
                                                (g.algs[-el.class[r]].nrel > 0 || g.algs[-el.class[r]].nwhen > 0)])
  #= Code: one loop per batch (rhsBody: the needed columns only). =#
  local body = Any[]; local rhsBody = Any[]
  local solved::Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}} = Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}}()
  local solution!::Function = (c::EqClass, ci::Int, p::Int) -> get!(solved, (ci, p)) do
    try
      _solveFor(c.lhs, c.rhs, p, c.text)
    catch err
      (err isa NotSupported && startswith(err.msg, "cannot solve")) || rethrow(err)
      nothing
    end
  end
  #= time-invariant instances: an equation solved explicitly for an algebraic unknown from parameters and invariant
     unknowns only (`_readsDynamic` neither side): computed once per solve after the initialization (`invariants!`),
     not in the right-hand side (MSL MultiBody's normalized axes, world frames, fixed rotations) =#
  local invariant::BitVector = falses(length(el.class))
  local staticClass::Dict{Int, Bool} = Dict{Int, Bool}()
  for ((ci, p), batch) in batches
    ci > 0 || continue
    local c::EqClass = g.classes[ci]
    get!(() -> c.nrel == 0 && !_readsDynamic(g, c.lhs) && !_readsDynamic(g, c.rhs), staticClass, ci) || continue
    any(r::Int -> (r, matchRow[r]) in el.aliased, batch) && continue
    solution!(c, ci, p) === nothing && continue
    for r in batch
      (loops.blockOf[r] == 0 && g.nStates < matchRow[r] <= g.nStates + g.nAlg && el.sptr[r] == el.sptr[r + 1]) || continue
      invariant[r] = all(i::Int -> (local u::Int = el.uid[i];
                                    u == matchRow[r] || (u <= g.nStates + g.nAlg && producer[u] > 0 && invariant[producer[u]])),
                         el.uptr[r]:(el.uptr[r + 1] - 1))
    end
  end
  local invariantBody = Any[]; local condBody = Any[]
  local emit! = function (nslots, domain, batch, code)
    push!(body, nslots == 0 ? Expr(:let, :(_col = 1), code) : _loopCode(nslots, domain, [el.col[r] for r in batch], code))
    local fixed = [r for r in batch if invariant[r]]
    isempty(fixed) || push!(invariantBody, nslots == 0 ? Expr(:let, :(_col = 1), copy(code)) : _loopCode(nslots, domain, [el.col[r] for r in fixed], copy(code)))
    local crossing = [r for r in batch if relNeeded[r] && !invariant[r]]
    isempty(crossing) || push!(condBody, nslots == 0 ? Expr(:let, :(_col = 1), copy(code)) : _loopCode(nslots, domain, [el.col[r] for r in crossing], copy(code)))
    local need = [r for r in batch if needed[r] && !invariant[r]]
    isempty(need) && return
    push!(rhsBody, nslots == 0 ? Expr(:let, :(_col = 1), copy(code)) : _loopCode(nslots, domain, [el.col[r] for r in need], copy(code)))
  end
  #= the loop families: the scheduled blocks' (LoopBlocks), then single instances an equation determines only
     implicitly (nonlinear in its unknown); their solves' code (_loopFamilyCode), the members of each call =#
  local familyBlocks::Vector{Vector{Vector{Int}}} = [Vector{Int}[] for _ in loops.families]
  local blockMember::Vector{Int} = zeros(Int, length(loops.blocks))
  for (b, f) in enumerate(_blockFamilies(el, pos, loops))
    push!(familyBlocks[f], loops.blocks[b])
    blockMember[b] = length(familyBlocks[f])
  end
  local familyTemplates::Vector{Vector{Tuple{Int, Int}}} = copy(loops.families)
  local loopDefs::Vector{Any} = Any[]; local loopBodies::Vector{Any} = Any[]
  local loopFamily!::Function = function (f::Int, template::Vector{Tuple{Int, Int}}, blocks::Vector{Vector{Int}})
    local (defs::Vector{Any}, code::Expr) = _loopFamilyCode(g, f, template, blocks, el, matchRow, a0, relBases)
    append!(loopDefs, defs)
    push!(loopBodies, code)
  end
  #= a family's blocks are alike: all large (blockSolve!) or all Newton's =#
  local largeFamily::BitVector = falses(length(familyTemplates))
  for (b, f) in enumerate(_blockFamilies(el, pos, loops))
    largeFamily[f] = large[b]
  end
  foreach(f::Int -> largeFamily[f] ? push!(loopBodies, :(nothing)) : loopFamily!(f, familyTemplates[f], familyBlocks[f]),
          eachindex(familyTemplates))
  local callFamily!::Function = function (f::Int, members::Vector{Int}, neededMembers::Vector{Int}, crossingMembers::Vector{Int})
    push!(body, :($(Symbol("LOOPFAMILY_", f, "!"))(du, a, u, p, t, z, mode, fire, $(members))))
    isempty(neededMembers) || push!(rhsBody, :($(Symbol("LOOPFAMILY_", f, "!"))(du, a, u, p, t, z, mode, fire, $(neededMembers))))
    isempty(crossingMembers) || push!(condBody, :($(Symbol("LOOPFAMILY_", f, "!"))(du, a, u, p, t, z, mode, fire, $(crossingMembers))))
  end
  #= the large blocks, in schedule order: each a residual function (its unknowns placed where the equations read them,
     then its instances' residuals, a loop per class over their columns) and its solver's call (blockSolve!) =#
  local blockBodies::Vector{Expr} = Expr[]
  local blockPatterns::Vector{Tuple{Vector{Int}, Vector{Int}}} = Tuple{Vector{Int}, Vector{Int}}[]
  local blockGuesses::Vector{Vector{Float64}} = Vector{Float64}[]
  local largeBlock!::Function = function (b::Int)
    local rs::Vector{Int} = loops.blocks[b]
    local torn::TornBlock = _tearBlock(g, el, rs, matchRow)
    local ids::Vector{Int} = torn.tears
    push!(blockPatterns, torn.pattern)
    push!(blockGuesses, Float64[u <= g.nStates ? 0.0 : a0[u - g.nStates] for u in ids])
    _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE") &&
      println("[arraycode] block of ", length(rs), " instances torn: ", length(ids), " iteration variables, ",
              length(torn.batches), " explicit batches")
    local parts::Vector{Any} = Any[:(for _j in 1:$(length(ids))
      local _id::Int = $(ids)[_j]
      if _id <= $(g.nStates)
        du[_id] = y[_j]
      else
        a[_id - $(g.nStates)] = y[_j]
      end
    end)]
    #= the explicitly solved instances given the iteration variables, then the residuals =#
    for ((ci, p), batch) in torn.batches
      local c::EqClass = g.classes[ci]
      local (target::Ref, value::IR) = get!(() -> _solveFor(c.lhs, c.rhs, p, c.text), solved, (ci, p))
      g.relBase = relBases[ci]; g.relCount = c.nrel
      local assign::Expr = :($(jl(g, target)) = $(jl(g, value)))
      push!(parts, c.nslots == 0 ? Expr(:let, :(_col = 1), assign) : _loopCode(c.nslots, c.domain, Int[el.col[rs[i]] for i in batch], assign))
    end
    local byClass::Dict{Int, Vector{Int}} = Dict{Int, Vector{Int}}()
    for (t, i) in enumerate(torn.residuals)
      push!(get!(() -> Int[], byClass, el.class[rs[i]]), t)
    end
    for ci in sort!(collect(keys(byClass)))
      local c::EqClass = g.classes[ci]
      local rows::Vector{Int} = zeros(Int, size(c.domain, 2))
      foreach(t::Int -> (rows[el.col[rs[torn.residuals[t]]]] = t), byClass[ci])
      g.relBase = relBases[ci]; g.relCount = c.nrel
      local code::Expr = :(res[$(rows)[_col]] = $(jl(g, c.lhs)) - $(jl(g, c.rhs)))
      push!(parts, c.nslots == 0 ? Expr(:let, :(_col = 1), code) :
                   _loopCode(c.nslots, c.domain, Int[el.col[rs[torn.residuals[t]]] for t in byClass[ci]], code))
    end
    push!(blockBodies, Expr(:block, parts...))
    local call::Expr = :($(blockSolve!)(p.blocks[$(length(blockBodies))], $(Symbol("BLOCK_", length(blockBodies), "!")),
                                        du, a, u, p, t, z, mode, fire))
    push!(body, call)
    needed[rs[1]] && push!(rhsBody, copy(call))
    relNeeded[rs[1]] && push!(condBody, copy(call))
  end
  for ((ci, p), batch) in batches
    if ci == 0 && large[batch[1]]
      foreach(largeBlock!, batch)
      continue
    end
    if ci == 0
      local members::Vector{Int} = Int[blockMember[b] for b in batch]
      callFamily!(p, members, Int[blockMember[b] for b in batch if needed[loops.blocks[b][1]]],
                  Int[blockMember[b] for b in batch if relNeeded[loops.blocks[b][1]]])
      continue
    end
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
    local solution::Union{Nothing, Tuple{Ref, IR}} = any(r::Int -> (r, matchRow[r]) in el.aliased, batch) ? nothing : solution!(c, ci, p)
    if solution === nothing
      #= nonlinear in its unknown, or reading it through a reference that is it only at some instance: each instance
         solved by Newton on its residual, as a loop of one (in a model with arrays, as the loops) =#
      hasArrays || ns("cannot solve for an unknown in a model without arrays: $(c.text)")
      push!(familyTemplates, [(ci, p)])
      local f::Int = length(familyTemplates)
      loopFamily!(f, familyTemplates[f], [[r] for r in batch])
      callFamily!(f, collect(1:length(batch)), Int[k for (k, r) in enumerate(batch) if needed[r]],
                  Int[k for (k, r) in enumerate(batch) if relNeeded[r]])
      continue
    end
    local (target::Ref, value::IR) = solution
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
      elseif _callOfParameters(g, value)
        #= a call of parameters only: once per pass of the event iteration (WHEN_MEMO), as the body
           fires once in an instant; the passes rerun a fired body, and an impure function (MSL
           impureRandom(id)) gave a new value in each: the initial events of ImpureGenerator never
           settled. A pure one gives the same value. =#
        push!(stmtsOf[b], :(p.d[$(_linear(info, idx))] = Float64(get!(() -> $(jl(g, value)), WHEN_MEMO, ($wi, $i, _col)))))
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
                             if nm in g.structural || _structuralParameter(g.vars[nm].var)]
  local ruleExprs = [:(($(QuoteNode(sym)), $(deps), $(code === nothing ? :(nothing) : code))) for (sym, deps, code) in rules]
  local jacProto = pattern === nothing ? :(nothing) :
    :(Symbolics.SparseArrays.sparse($(pattern[1]), $(pattern[2]), ones($(length(pattern[1]))), $nStates, $nStates, (x, y) -> x))
  local paramsType = ArrayModelParameters
  #= the large functions' statements in functions of bounded size (_splitStatements) =#
  local (eqDefs::Vector{Expr}, eqStmts::Vector{Any}) =
    _splitStatements(:equations!, Symbol[:du, :a, :u, :p, :t, :z, :mode, :fire], Symbol[], paramLocals, body; fired = true)
  local (rhsDefs::Vector{Expr}, rhsStmts::Vector{Any}) =
    _splitStatements(:RHS!, Symbol[:du, :u, :p, :t], Symbol[:a, :z, :mode, :fire], paramLocals, rhsBody; fired = true)
  local (condDefs::Vector{Expr}, condStmts::Vector{Any}) =
    _splitStatements(:crossings!, Symbol[:du, :a, :u, :p, :t, :z, :mode], Symbol[:fire], paramLocals, condBody; fired = true)
  local (invDefs::Vector{Expr}, invStmts::Vector{Any}) =
    _splitStatements(:invariants!, Symbol[:a, :p], Symbol[], paramLocals, invariantBody; fired = false)
  local blockSplits::Vector{Tuple{Vector{Expr}, Vector{Any}}} =
    Tuple{Vector{Expr}, Vector{Any}}[_splitStatements(Symbol("BLOCK_", k, "!"), Symbol[:res, :y, :du, :a, :u, :p, :t, :z, :mode, :fire],
                                                      Symbol[], paramLocals, code.args; fired = false)
                                     for (k, code) in enumerate(blockBodies)]
  local splitDefs::Vector{Expr} = vcat(eqDefs, rhsDefs, condDefs, invDefs, Expr[d for (defs, _) in blockSplits for d in defs])
  local code = quote
    using DifferentialEquations
    using OrdinaryDiffEq
    import Symbolics
    import NonlinearSolve
    import ADTypes
    import DiffEqCallbacks
    import LinearAlgebra
    #= the MTK path's function code names it (algorithmic.jl: an output array's growth) =#
    const OMBackend = $(_OMBACKEND)

    $(functionDefs...)
    $(loopDefs...)
    $(splitDefs...)
    $([:(function $(Symbol("LOOPFAMILY_", f, "!"))(du, a, u, p, t, z, mode, fire, members)
           $(paramLocals...)
           @inbounds begin
             $(code)
           end
           return nothing
         end) for (f, code) in enumerate(loopBodies)]...)
    $([:(function $(Symbol("BLOCK_", k, "!"))(res, y, du, a, u, p, t, z, mode, fire)
           $(paramLocals...)
           @inbounds begin
             $(stmts...)
           end
           return nothing
         end) for (k, (_, stmts)) in enumerate(blockSplits)]...)
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
    #= the large algebraic loops (blockSolve!), in schedule order: their residuals' sparsity in their unknowns, its
       column colors, the unknowns' start values =#
    const BLOCK_PATTERNS = Symbolics.SparseArrays.SparseMatrixCSC{Float64, Int}[$([:(Symbolics.SparseArrays.sparse($I, $J, ones($(length(I))), $(length(gs)), $(length(gs)), (x, y) -> x))
                                                                                  for ((I, J), gs) in zip(blockPatterns, blockGuesses)]...)]
    const BLOCK_COLORS = Vector{Int}[$(columnColors)(S) for S in BLOCK_PATTERNS]
    const BLOCK_GUESSES = $(blockGuesses)
    #= initialization: the states the initial equations determine (one per residual) =#
    const INIT_UNKNOWNS = $(initStates)
    const INIT_PARAMS = $(initParams)   #= fixed = false parameters: positions in p.values =#
    const N_INIT = $(nInit)
    const D0 = $(d0)
    const N_RELATIONS = $(nRelTotal)
    #= the relation numbering's relations (not event-generating functions' crossing functions) =#
    const RELATION_SLOT = $(_relationSlots(g, nRelTotal, relBases, whenRelBases, algRelBases))
    const USES_INITIAL = $(g.usesInitial)
    #= set by sampleSchedule! after the initialization =#
    const SAMPLE_START = zeros($(length(sampleSchedule)))
    const SAMPLE_INTERVAL = zeros($(length(sampleSchedule)))
    const N_WHENS = $(nWhenInst)
    #= the values of when-body calls of parameters in a pass of the event iteration =#
    const WHEN_MEMO = Dict{Tuple{Int, Int, Int}, Any}()

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
        $(eqStmts...)
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
      #= scaled as OpenModelica scales its nonlinear systems: each unknown by its start's magnitude (1 for a zero
         start), so the finite differences step relative to it (MSL LightningImpulse: T10 ~ 4e-7 and tau ~ 1e-5 beside
         eta ~ 1), each residual by its largest Jacobian entry at the guess; solved in a trust region (OpenModelica's
         hybrid dogleg): undamped Newton ran to LightningImpulse's degenerate root (tau1 = tau2, eta = 0). Newton
         where the trust region fails. =#
      local scale = map(x -> x == 0 ? 1.0 : abs(x), z)
      local scaledSolve = function (z0)
        local r0 = similar(z0); local rp = similar(z0); local rows = zeros(length(z0))
        residual(r0, z0, nothing)
        for k in eachindex(z0)
          local zp = copy(z0)
          local h = 1.0e-7 * max(abs(z0[k]) / scale[k], 1.0)
          zp[k] += h * scale[k]
          residual(rp, zp, nothing)
          rows .= max.(rows, abs.(rp .- r0) ./ h)
        end
        rows .= map(r -> (r > 0 && isfinite(r)) ? 1 / r : 1.0, rows)
        local scaledResidual = function (res, w, _)
          residual(res, w .* scale, nothing)
          res .*= rows
          return nothing
        end
        local prob = NonlinearSolve.NonlinearProblem(NonlinearSolve.NonlinearFunction(scaledResidual), z0 ./ scale)
        local sol = NonlinearSolve.solve(prob, NonlinearSolve.TrustRegion(autodiff = ADTypes.AutoFiniteDiff());
                                         abstol = 1e-10, reltol = 1e-10)
        return (DifferentialEquations.SciMLBase.successful_retcode(sol), sol.u .* scale, sol.retcode)
      end
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
        local (ok, zs, _) = scaledSolve(z)
        if ok
          z = zs
          first === nothing && (first = copy(z))
          equations!(du, a, setUnknowns!(u, z), p, t, nothing, true)
          (p.rel == rel0 && p.evv == e0) && return u
          continue
        end
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

    #= The time-invariant algebraic values (from parameters only) into a: once per solve, after the initialization. =#
    function invariants!(a, p)
      $(paramLocals...)
      @inbounds begin
        $(invStmts...)
      end
      return nothing
    end

    #= The relations' crossing functions into z (and their values, in mode): only what the relations need, on p.abuf
       (the invariant values from invariants!). A model with when-equations evaluates all (their conditions read what
       the equations give). =#
    const CROSSINGS_ONLY = $(isempty(g.whens))
    function crossings!(du, a, u, p, t, z, mode::Bool)
      $(paramLocals...)
      local fire::Bool = false; local _fired::Bool = false
      @inbounds begin
        $(condStmts...)
      end
      return nothing
    end

    #= The right-hand side: only what the state derivatives need (relations keep their values; the invariant values
       in p.abuf from invariants!). =#
    function RHS!(du, u, p, t)
      $(paramLocals...)
      local a::AbstractVector = eltype(u) === Float64 ? p.abuf : similar(u, $nAlg)
      a === p.abuf || invariants!(a, p)
      local z::Nothing = nothing; local mode::Bool = false; local fire::Bool = false; local _fired::Bool = false
      @inbounds begin
        $(rhsStmts...)
      end
      return nothing
    end

    #= The right-hand side's derivative in time (Rosenbrock methods): central differences of the step the solver's own
       take away from t = 0 (relative to |t|), near t = 0 a billionth of the run: the solver's step 6e-6 there spans all
       of MSL DifferenceAmplifier's sources (a damped 0.2 GHz sine, ramps of 1 ns), and its steps shrank to 1e-25. An
       error in the right-hand side is recorded as the step-rejecting RHS's (`StepRejectingRHS`). =#
    function tgrad!(dT, u, p, t)
      local h::Float64 = cbrt(eps(Float64)) * max(abs(Float64(t)), 1.0e-9 * abs(p.tend[1]), floatmin(Float64))
      local after = similar(dT); local before = similar(dT)
      try
        RHS!(after, u, p, t + h)
        RHS!(before, u, p, t - h)
      catch err
        err isa OMBackend.CodeGeneration._RHS_STEP_ERRORS || rethrow()
        OMBackend.CodeGeneration.LAST_RHS_ERROR[] = err
        fill!(dT, NaN)
        return nothing
      end
      dT .= (after .- before) ./ (2h)
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
                              [false], [false], copy(p.upre), copy(p.ureinit), copy(p.wnew), p.dlog, evv, copy(p.hyst), copy(p.tend),
                              zeros($nAlg), p.blocks, p.model)
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
        empty!(WHEN_MEMO)
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
      local du = similar(u)
      CROSSINGS_ONLY && return crossings!(du, p.abuf, u, p, t, out, false)
      local a = similar(u, $nAlg)
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
       across the event; a relation whose crossing the root finder found (`events`: +1 its crossing
       function rose past zero, -1 it fell) changes where that is its change (a true one's rises, a
       false one's falls) and its value at the left limit did not change already; then the event
       iteration at the event. Re-evaluated there, a hair off its bound, the relation kept its value
       where an algebraic loop solved inside the equations gave its last digits from another guess,
       and the same crossing was found again in every step (MSL EngineV6's piston speed at dead
       center). =#
    function affectCrossing!(integrator, events::AbstractVector{Int8})
      local t = integrator.t
      local tl = max(integrator.tprev, t - 1e-10 * (1 + abs(t)))
      local ul = tl < t ? integrator(tl) : copy(integrator.uprev)
      local p = integrator.p
      local du = similar(ul); local a = similar(ul, $nAlg)
      local before::Vector{Bool} = copy(p.rel)
      equations!(du, a, ul, p, tl, nothing, true)
      whens!(du, a, ul, p, tl, nothing, true, false)
      p.wcond .= p.wnew
      for k in eachindex(events)
        (events[k] != 0 && RELATION_SLOT[k] && p.rel[k] == before[k] && p.rel[k] == (events[k] > 0)) && (p.rel[k] = !p.rel[k])
      end
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
      local du = similar(u)
      if CROSSINGS_ONLY
        crossings!(du, p.abuf, u, p, t, nothing, true)
      else
        local a = similar(u, $nAlg)
        equations!(du, a, u, p, t, nothing, true)
        whens!(du, a, u, p, t, nothing, true, false)
      end
      local changed = rel0 != p.rel || e0 != p.evv
      p.rel .= rel0; p.wcond .= w0; p.evv .= e0
      return changed
    end

    #= The parameters and discrete state of a simulation over `tspan` (before the initialization). =#
    function modelParameters(tspan = (0.0, 1.0); parameters::AbstractDict = Dict{String, Any}())
      return $(paramsType)(flatParameters(isempty(parameters) ? PARAMS : parameterValues(parameters)),
                           zeros(Bool, N_RELATIONS), copy(D0), copy(D0), zeros(Bool, N_WHENS),
                           zeros(Bool, length(SAMPLE_START)), [false], [false], copy(U0), copy(U0),
                           zeros(Bool, N_WHENS), Tuple{Float64, Vector{Float64}}[], zeros(N_RELATIONS), [0.0], [Float64(tspan[2])],
                           zeros($nAlg),
                           $(BlockSolver)[$(BlockSolver)(BLOCK_PATTERNS[k], BLOCK_COLORS[k], BLOCK_GUESSES[k]) for k in eachindex(BLOCK_PATTERNS)],
                           @__MODULE__)
    end

    #= The problem and its event callbacks, as getMTKProblem returns them. =#
    function $(Symbol(modelName, "Model"))(tspan = (0.0, 1.0); parameters::AbstractDict = Dict{String, Any}())
      local p = modelParameters(tspan; parameters = parameters)
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
      invariants!(p.abuf, p)
      #= a domain error or an assert in the right-hand side rejects the step (the solver retries smaller), as in the
         scalar path (MSL DifferenceAmplifier's x^y of a negative x at a trial point) =#
      #= tgrad! wrapped to one type for every array model (the right-hand side is, AutoSpecialize): the model's own
         function type in the problem compiled the solver again for each model =#
      local f::ODEFunction = ODEFunction($(StepRejectingRHS)(RHS!); jac_prototype = $jacProto,
                                         tgrad = OMBackend.CodeGeneration.DiffEqBase.wrapfun_jac_iip(tgrad!, (u0, u0, p, Float64(tspan[1]))),
                                         sys = $(ArraySystem)(@__MODULE__))
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
      OMBackend.CodeGeneration._resetRHSError!()
      local sol = DifferentialEquations.solve(prob, solver; callback = callbacks, kwargs...)
      #= an error that every smaller step met ends the run =#
      OMBackend.CodeGeneration._rethrowRHSError(sol)
      return sol
    end
  end
  return Expr(:module, true, Symbol(modelName), stripBeginBlocks(code))
end

include("ArrayODEGenerationIR.jl")

end #= module ArrayODEGen =#
