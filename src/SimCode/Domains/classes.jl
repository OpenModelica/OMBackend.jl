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

#= The domain level's classes and their expression form, its passes (rerolling, folding, alias elimination, semiLinear
   chains) and the causal level (elements, matching, scheduling with loops, tearing, solving for an unknown): moved
   from CodeGeneration's ArrayODEGeneration.jl (2026-11-01), unchanged. =#

struct NotSupported <: Exception
  msg::String
end

ns(msg::AbstractString) = throw(NotSupported(String(msg)))

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

"""
    SimVariable(simVar, names, value)

A variable of the simulation code (the front half, ArrayODEGenerationSC.jl): the variable, its elements' names
(row-major, as the frontend's `c[2].T`), and a parameter's value (a scalar, or an array row-major) as the lowering
evaluated it; `nothing` for another variable.
"""
struct SimVariable
  simVar::SimCode.SIMVAR
  names::Vector{String}
  value::Any
end

mutable struct VarInfo
  kind::Symbol            #= :state, :alg, :discrete or :param =#
  dims::Vector{Int}
  perPart::Vector{Int}    #= subscripts per cref part, root first =#
  offset::Int             #= first element in u, the algebraic buffer or the discrete buffer =#
  vtype::Symbol           #= :real, :int or :bool =#
  var::SimVariable
end

"""
    EqClass(lhs, rhs, nslots, domain, nrel, text, nsample)

An equation over a domain: its sides over iterator slots, the slots' values per instance (`domain`, nslots x
instances), its relations' and `sample()` calls' numbers per instance (a sample-defined Boolean, CDL Discrete's
`sampleTrigger = sample(t0, samplePeriod)`: true at its ticks), its text.
"""
struct EqClass
  lhs::IR
  rhs::IR
  nslots::Int
  domain::Matrix{Int}
  nrel::Int
  text::String
  nsample::Int
end

"An equation over a domain without `sample()` calls."
EqClass(lhs::IR, rhs::IR, nslots::Int, domain::Matrix{Int}, nrel::Int, text::String) = EqClass(lhs, rhs, nslots, domain, nrel, text, 0)

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
  everyEvent::Bool                       #= run at every pass of every event (SC.EQ_ATTR's everyEvent) =#
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
  derOf::Dict{String, String}   #= the states' derivative links (SimCode's STATE_DERIVATIVE entries): \$STATEDER.x -> x =#
  startComputed::Set{String}    #= parameters the translation has no value of (an impure call): computed at each start =#
  strings::Vector{Tuple{String, String}}  #= String parameters the code reads (name, translated value): the run state's slots =#
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

function _paramValue!(g::Gen, name::String)
  haskey(g.params, name) && return g.params[name]
  local v::SimVariable = g.vars[name].var
  local val::Any = v.value === nothing ? ns("the value of parameter $(name)") : v.value
  g.params[name] = val
  #= the name's symbol, numbered where another parameter's name gives the same (`c.k` and `c_k`: P_c_k) =#
  local base::String = string("P_", replace(name, r"[^A-Za-z0-9_]" => "_"))
  local sym::Symbol = Symbol(base)
  local taken::Set{Symbol} = Set{Symbol}(values(g.paramSyms))
  local k::Int = 1
  while sym in taken
    k += 1
    sym = Symbol(base, "_", k)
  end
  g.paramSyms[name] = sym
  return val
end

const MATH_BUILTINS = Dict("sin" => :sin, "cos" => :cos, "tan" => :tan, "asin" => :asin,
                           "acos" => :acos, "atan" => :atan, "atan2" => :atan, "exp" => :exp,
                           "log" => :log, "log10" => :log10, "sqrt" => :sqrt, "abs" => :abs,
                           "sinh" => :sinh, "cosh" => :cosh, "tanh" => :tanh,
                           "min" => :min, "max" => :max, "sign" => :sign)

const MATH_WRAPPED = ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log", "log10")

"""
Whether an expression varies continuously (states, algebraic variables, time). A relation and an event function are
constant between events (discrete-time, MLS 3.8: `(if x > 0.5 then 5 else 0) >= 5.0` is no continuous comparison, its
operand jumps at the relation's own event: a crossing relation of it never fired).
"""
function _continuous(g::Gen, x::IR)::Bool
  x isa TimeIR && return true
  x isa Ref && return g.vars[x.name].kind in (:state, :alg) || any(s -> _continuous(g, s), x.subs)
  x isa Op && return any(a -> _continuous(g, a), x.args)
  return false
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
    #= a String parameter's translated value: the code depends on it =#
    x.op == :stringparam && (structural && push!(g.structural, a[3]); return a[2])
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

"""
Whether a condition depends on loop indices and parameters only.
"""
_isStatic(g::Gen, x::IR)::Bool =
  x isa Lit || x isa Slot ||
  (x isa Ref && !x.der && g.vars[x.name].kind == :param && all(s -> _isStatic(g, s), x.subs)) ||
  (x isa Op && !(x.op in (:initial, :sample)) && all(a -> _isStatic(g, a), x.args))

"""
initial() activates a when at the initialization only as its condition (MLS 8.6, as
OpenModelica): inside an expression (initial() or x > 0.6) it is false.
"""
_topLevelInitial(x::IR) = (x isa Op && x.op == :initial) ? x : _noInitial(x)

"""
The condition elements of a when (one for a scalar condition).
"""
_condElements(c::IR)::Vector{IR} = (c isa Op && c.op == :elsewhen) ? IR[e for b in c.args for e in _condElements(b)] :
                                   (c isa Op && c.op == :vect) ? c.args : IR[c]

_noInitial(x::IR) = !(x isa Op) ? x : x.op == :initial ? Lit(false) : Op(x.op, IR[_noInitial(a) for a in x.args])

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
    if c.nslots > 0 || c.nrel > 0 || c.nsample > 0
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

"The ids of the event sources (a relation `Rel`, an event-generating function `EvFn`) in an expression that `want` takes, into `ids`."
function _sourceIds!(ids::Set{Int}, x::IR, want::F)::Set{Int} where {F}
  if x isa EvFn
    want(x) && push!(ids, x.id)
    _sourceIds!(ids, x.arg, want)
  elseif x isa Op
    foreach(a::IR -> _sourceIds!(ids, a, want), x.args)
  elseif x isa Rel
    want(x) && push!(ids, x.id)
    _sourceIds!(ids, x.lhs, want)
    _sourceIds!(ids, x.rhs, want)
  elseif x isa Ref
    foreach(s::IR -> _sourceIds!(ids, s, want), x.subs)
  end
  return ids
end

"The ids of the event sources in statements that `want` takes, into `ids`."
function _sourceIds!(ids::Set{Int}, stmts::Vector{Stmt}, want::F)::Set{Int} where {F}
  for st in stmts
    st isa SAssign && _sourceIds!(ids, st.value, want)
    if st isa SIf || st isa SWhen
      for (c, b) in st.branches
        _sourceIds!(ids, c, want)
        _sourceIds!(ids, b, want)
      end
    end
    st isa SFor && _sourceIds!(ids, st.body, want)
    st isa SWhile && (_sourceIds!(ids, st.cond, want); _sourceIds!(ids, st.body, want))
    st isa SCall && _sourceIds!(ids, st.call, want)
    st isa SAssert && _sourceIds!(ids, st.cond, want)
  end
  return ids
end

"""
The slots of the relation numbering of the event sources `want` takes, `width` from each id (an event-generating
function has two crossing functions): each template's ids at each column of its domain from its base.
"""
function _sourceSlots(g::Gen, nRelTotal::Int, relBases::Vector{Int}, whenRelBases::Vector{Int}, algRelBases::Vector{Int},
                      want::F, width::Int)::BitVector where {F}
  local slots::BitVector = falses(nRelTotal)
  local mark!::Function = function (ids::Set{Int}, base::Int, nrel::Int, ncols::Int)
    for col in 1:ncols, id in ids, k in id:(id + width - 1)
      slots[base + (col - 1) * nrel + k] = true
    end
  end
  for (ci, c) in enumerate(g.classes)
    c.nrel == 0 || mark!(_sourceIds!(_sourceIds!(Set{Int}(), c.lhs, want), c.rhs, want), relBases[ci], c.nrel, size(c.domain, 2))
  end
  for (wi, w) in enumerate(g.whens)
    w.nrel == 0 && continue
    local ids::Set{Int} = _sourceIds!(Set{Int}(), w.cond, want)
    foreach(b::Tuple{Symbol, Ref, IR} -> _sourceIds!(ids, b[3], want), w.body)
    mark!(ids, whenRelBases[wi], w.nrel, size(w.domain, 2))
  end
  for (ai, a) in enumerate(g.algs)
    a.nrel == 0 || mark!(_sourceIds!(Set{Int}(), a.body, want), algRelBases[ai], a.nrel, size(a.domain, 2))
  end
  return slots
end

"""
The slots of the relation numbering that are relations (`_rel`, `p.rel`), not an event-generating function's two
crossing functions (`_evfn`).
"""
_relationSlots(g::Gen, nRelTotal::Int, relBases::Vector{Int}, whenRelBases::Vector{Int}, algRelBases::Vector{Int})::BitVector =
  .!_sourceSlots(g, nRelTotal, relBases, whenRelBases, algRelBases, (x::IR) -> x isa EvFn, 2)

"The slots of the relation numbering that are relations of time and a parameter expression (`_isTimeRelation`)."
_timeRelationSlots(g::Gen, nRelTotal::Int, relBases::Vector{Int}, whenRelBases::Vector{Int}, algRelBases::Vector{Int})::BitVector =
  _sourceSlots(g, nRelTotal, relBases, whenRelBases, algRelBases, (x::IR) -> x isa Rel && _isTimeRelation(g, x), 1)

"Whether the relation `x` is of `time` and a parameter expression (`time > 0.6`): a time event at its exact time."
_isTimeRelation(g::Gen, x::Rel)::Bool =
  (x.lhs isa TimeIR && _parametersOnly(g, x.rhs)) || (x.rhs isa TimeIR && _parametersOnly(g, x.lhs))

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

"The variables the initial algorithms assign."
function _initAlgAssigned(g::Gen)::Set{String}
  local names::Set{String} = Set{String}()
  local visit!::Function = function (stmts::Vector{Stmt})
    for st in stmts
      st isa SAssign && push!(names, st.target.name)
      st isa SIf && foreach(b::Tuple{IR, Vector{Stmt}} -> visit!(b[2]), st.branches)
      (st isa SFor || st isa SWhile) && visit!(st.body)
    end
  end
  foreach(a::AlgClass -> visit!(a.body), g.initAlgs)
  return names
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
    oriented[ci] = EqClass(Ref(v, subs, false), negated ? Op(:neg, IR[value]) : value, c.nslots, c.domain, c.nrel, c.text,
                           c.nsample)
  end
  local sub::Function = (c::EqClass) -> EqClass(_substituted(c.lhs, alias), _substituted(c.rhs, alias), c.nslots, c.domain, c.nrel,
                                                c.text, c.nsample)
  g.classes = EqClass[get(() -> sub(c), oriented, ci) for (ci, c) in enumerate(g.classes)]
  g.whens = WhenClass[WhenClass(_substituted(w.cond, alias),
                                Tuple{Symbol, Ref, IR}[(kind, _substitutedTarget(t, alias), _substituted(e, alias)) for (kind, t, e) in w.body],
                                w.starts, w.nslots, w.domain, w.nrel, w.nsample, w.text, w.everyEvent) for w in g.whens]
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
    isempty(keep) || push!(out, EqClass(c.lhs, c.rhs, c.nslots, c.domain[:, keep], c.nrel, c.text, c.nsample))
  end
  g.classes = append!(out, added)
  return nChains
end

"""
    InstanceTable(class, col, uptr, uid, upos, sptr, sid, outs, nEq, aliased)

The instances of a domain system's classes (`_elements`), each a row: its class (> 0 an equation class, < 0 an
algorithm by its negated index) and column, its unknowns with their access positions (CSR: `uptr`, `uid`, `upos`), the
states it reads (`sptr`, `sid`), the unknowns an algorithm instance assigns (empty for equations), the number of
equation instances, and the (instance, unknown) pairs read through two different references (`x[i]` and `x[2]` at
`i = 2`). Unknown ids: state derivatives `1:nStates`, algebraic elements after them, then the discrete unknowns.
"""
struct InstanceTable
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

function _elements(g::Gen)::InstanceTable
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
  return InstanceTable(class, col, uptr, uid, upos, sptr, sid, outs, nEq, aliased)
end

"""
Maximum matching instances <-> unknowns (greedy, then augmenting paths).
"""
function _match(el::InstanceTable, nU::Int)::Vector{Int}
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
function _schedule(el::InstanceTable, matchRow::Vector{Int}, nU::Int)
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
           same element only at this instance (InstanceTable' aliased: solved by its residual, `_loopFamilyCode`) =#
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
function _tearBlock(g::Gen, el::InstanceTable, block::Vector{Int}, matchRow::Vector{Int})::TornBlock
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
function _blockFamilies(el::InstanceTable, pos::Vector{Int}, loops::LoopBlocks)::Vector{Int}
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
function _scheduleWithLoops(el::InstanceTable, matchRow::Vector{Int}, nU::Int, producer::Vector{Int}, pos::Vector{Int})
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
function _stateDeps(el::InstanceTable, matchRow, batches, producer, loops::LoopBlocks, nStates::Int)
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
function _jacobianPattern(el::InstanceTable, matchRow, deps, nStates::Int)
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
