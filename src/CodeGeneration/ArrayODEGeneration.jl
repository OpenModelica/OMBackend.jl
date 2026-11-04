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

For the simulation code of a model with its arrays kept (the frontend's `scalarize = false`, OMJLIR, then
SimCode.lowerArraySystem; its variables and equations the classes of its domain level, SimCode.Domains, whose causal
level matches and schedules them), builds a
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
     the relations' crossing functions from what they need only (`crossings!`). Calls
     inlined after index reduction where a function's body runs into one expression
     (`Passes.inlineAfterIndexReduction!`).
  4. Code: one loop per batch over its domain columns (an affine range, else a table passed
     as data), parameters and start values evaluated once (on the model's element view) and
     passed as data (all parameters tunable; bindings that read a changed parameter are computed
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

What the front half does not (yet) handle it reports with the reason, and the caller scalarizes
(SimCode/Domains/translate.jl's header lists it).
=#
module ArrayODEGen

import ModelingToolkit
import ..CodeGeneration: stripBeginBlocks, ModelicaAssertionError, solveStoppedAtStart
import ...Passes
import ...SimCode
import ...SimCode.Domains: DomainSystem, InstanceTable, EqClass, EvFn, Gen, IR, Lit, LoopBlocks, NotSupported, Op, Pre,
  Ref, Rel, SAssert, SAssign, SBreak, SCall, SFor, SIf, SWhen, SWhile, Slot, Stmt, TimeIR, TornBlock, VarInfo, WhenClass,
  AlgClass,
  _blockFamilies, _callOfParameters, _condElements, _initAlgAssigned, _paramValue!, _parametersOnly, _relationSlots,
  _isTimeRelation, _timeRelationSlots, _continuous,
  domainSystem, ns, Schedule, causalSchedule!, ParameterBinding, parameterBindings

const SII = ModelingToolkit.SymbolicIndexingInterface

include("ArrayBlockSolve.jl")

#= The parameter argument of an array model's problem: the parameter values (one flat
   vector, laid out by the model's PARAMETER_LAYOUT; one type for every array model, so the
   solver is compiled once), the discrete state (relation values, discrete variables and their
   pre values, when-condition values, state values before the event) and the generated module. =#
"""
    ArrayStartState(discrete, wcond, wnew, ndlog, upre, ureinit)

A problem's state after its initialization (`ArrayRuntimeState.atStart`): the discrete state (`_discreteState`), the
when-conditions, the length of the event log, the pre() and reinit buffers of the states.
"""
struct ArrayStartState
  discrete::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}}
  wcond::Vector{Bool}
  wnew::Vector{Bool}
  ndlog::Int
  upre::Vector{Float64}
  ureinit::Vector{Float64}
end

"""
    ArrayRuntimeState(nSamples)

What one problem of an array model holds besides its values (`ArrayModelParameters.state`), so that problems of one model
(an ensemble's) run apart: the sample() starts and intervals (`sampleSchedule!`) and the stop times of the last
`sampleTimes`, the when bodies' calls of parameters in a pass of the event iteration (`whenMemo`), the warnings reported,
the asserts' probe (`_violated!`), an assert crossing to decide (`pending`: its time and state), the message of a
terminate() reached, the discrete state at the start of the current step (`stepStart`, `_discreteState`: within a step
that ends at an event the asserts read it), the String parameters' values (`Gen.strings`, with their overrides), the
state just before the last event (`leftLimit`: its time and state, a reinit moved it after), the problem's state after
its initialization (`atStart`, `ArrayStartState`: a solve of the problem starts from it again).
"""
mutable struct ArrayRuntimeState
  sampleStart::Vector{Float64}
  sampleInterval::Vector{Float64}
  sampleStops::Vector{Float64}
  whenMemo::Dict{Tuple{Int, Int, Int}, Any}
  warned::Set{Any}
  probe::Bool
  violated::Bool
  pending::Tuple{Float64, Vector{Float64}}
  terminated::Union{Nothing, String}
  stepStart::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}}
  strings::Vector{String}
  leftLimit::Tuple{Float64, Vector{Float64}}
  atStart::Union{Nothing, ArrayStartState}
end
ArrayRuntimeState(nSamples::Int, strings::Vector{String}) =
  ArrayRuntimeState(zeros(nSamples), zeros(nSamples), Float64[], Dict{Tuple{Int, Int, Int}, Any}(), Set{Any}(), false, false,
                    (NaN, Float64[]), nothing, (Float64[], Float64[], Bool[], Float64[]), strings, (NaN, Float64[]), nothing)

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
  state::ArrayRuntimeState     #= the run's state (samples, memo, warnings, asserts, terminate) =#
  model::Module
end

"""
A copy of a problem's parameters (`remake(prob; p = deepcopy(prob.p))`, an ensemble's trajectories): its buffers,
solvers and run state copied, the model's module shared (a module is no value: `deepcopy` refused it).
"""
function Base.deepcopy_internal(p::ArrayModelParameters, dict::IdDict)::ArrayModelParameters
  haskey(dict, p) && return dict[p]
  local fields::Vector{Any} = Any[f === :model ? p.model : Base.deepcopy_internal(getfield(p, f), dict)
                                  for f in fieldnames(ArrayModelParameters)]
  local q::ArrayModelParameters = ArrayModelParameters(fields...)
  dict[p] = q
  return q
end

"The discrete state of `p`: the discrete values, their pre() values, the relations, the event-generating functions' values."
_discreteState(p::ArrayModelParameters)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}} =
  (copy(p.d), copy(p.dpre), copy(p.rel), copy(p.evv))

"The state of `p` after its initialization (`ArrayStartState`)."
_startState(p::ArrayModelParameters)::ArrayStartState =
  ArrayStartState(_discreteState(p), copy(p.wcond), copy(p.wnew), length(p.dlog), copy(p.upre), copy(p.ureinit))

"""
    _restoreStart!(p)

`p` as after its initialization again (`ArrayRuntimeState.atStart`), at the start of a solve: a second solve of the
problem started from the first one's end (its discrete values, the event log).
"""
function _restoreStart!(p::ArrayModelParameters)::Nothing
  local s::Union{Nothing, ArrayStartState} = p.state.atStart
  s === nothing && return nothing
  p.d .= s.discrete[1]; p.dpre .= s.discrete[2]; p.rel .= s.discrete[3]; p.evv .= s.discrete[4]
  p.wcond .= s.wcond; p.wnew .= s.wnew; p.upre .= s.upre; p.ureinit .= s.ureinit
  resize!(p.dlog, s.ndlog)
  fill!(p.sampleActive, false)
  p.state.pending = (NaN, Float64[]); p.state.leftLimit = (NaN, Float64[]); p.state.terminated = nothing
  empty!(p.state.whenMemo)
  return nothing
end

"""
    _withDiscreteState(f, p, s)

`f()` with the discrete state of `p` set to `s` (`_discreteState`; the step start's, where an event ended the step), restored
after; with the state as it is where `s` holds none (before the first step).
"""
function _withDiscreteState(f::Function, p::ArrayModelParameters, s::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}})
  (length(s[1]) == length(p.d) && length(s[3]) == length(p.rel)) || return f()
  local now::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}} = _discreteState(p)
  p.d .= s[1]; p.dpre .= s[2]; p.rel .= s[3]; p.evv .= s[4]
  try
    return f()
  finally
    p.d .= now[1]; p.dpre .= now[2]; p.rel .= now[3]; p.evv .= now[4]
  end
end

"""
The entry of the discrete log for time t: at an event instant the values before it (the left
limit, as the states' interpolation and OpenModelica's result files), at the start time the
values after its events (nothing is before it: a clock's first tick, `sample(0, T)`), at the stop
time the final ones; `right` takes the values after an event instant (the second of a saved pair).
"""
_dlogIndex(times, t, tend, right::Bool = false) =
  (right || t >= tend || t <= times[1]) ? max(1, searchsortedlast(times, t)) : max(1, searchsortedfirst(times, t) - 1)

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

"A copy of the system (`deepcopy` of a problem: an ensemble's `safetycopy`): its tables copied, the model's module shared."
function Base.deepcopy_internal(s::ArraySystem, dict::IdDict)::ArraySystem
  haskey(dict, s) && return dict[s]
  local c::ArraySystem = ArraySystem(copy(s.states), copy(s.stateSyms), copy(s.observed), s.model)
  dict[s] = c
  return c
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
is clear: a true relation stays true while zc <= eps, a false one becomes true at zc <= -eps, or at an exact zc = 0
where the relation holds (an operand that jumped onto its threshold at an event, `(if x > 0.5 then 5 else 0) >= 5.0`
or an opaque function's value: it stays there and never leaves the band; the MTK path's cluster relations take the
literal value at an exact zero the same way, `_update!`; a true relation is not turned off at an exact zero, which
would chatter). With H = 0 (initialization) it is evaluated literally.
"""
_finiteAbs(x) = (local v = abs(float(x)); v < 1.0e300 ? v : 0.0)   #= as Float64: abs(typemin(Int)) stays negative =#
@inline _rel(z, p, mode::Bool, k::Int, x, y, op) = _relationAt(z, p, mode, k, x, y, op, p.hyst[1])

"""
A relation of `time` and a parameter expression (`time > 0.6`): a time event at its exact time, as OpenModelica takes
it, its hysteresis only the floating-point resolution (none at the initialization).
"""
@inline _timeRel(z, p, mode::Bool, k::Int, x, y, op) = _relationAt(z, p, mode, k, x, y, op, p.hyst[1] == 0 ? 0.0 : 8 * eps(Float64))

"A relation of hysteresis `H` (`_rel`, `_timeRel`)."
@inline function _relationAt(z, p, mode::Bool, k::Int, x, y, op, H::Float64)
  #= as Float64: the difference of two native Integers (an opaque function's result) wraps around beyond 2^63 =#
  local zc = (op === (>) || op === (>=)) ? float(y) - float(x) : float(x) - float(y)
  local old = p.rel[k]
  local eps = H * (1 + max(_finiteAbs(x), _finiteAbs(y)))
  z === nothing || (z[k] = old ? zc - eps : zc + eps)
  mode && (p.rel[k] = H == 0 ? op(x, y) : old ? zc <= eps : (zc <= -eps || (zc == 0 && op(x, y))))
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
const FULL = Base.Ref(true)

#= Sizes of the last generated model (classes, instances, batches = loops, ...), for diagnostics. =#
const LAST_STATS = Base.Ref{Any}(nothing)

"""
String(...) of Modelica, as the MTK path formats it (resolved when called: that module
may load after this one)
"""
_modelicaString(args...) = _OMBACKEND.CodeGeneration.AlgorithmicCodeGeneration.modelica_String(args...)

"""
The first sample slot of when wi: the slots are numbered when by when, instance by instance.
"""
_whenSampleBase(g::Gen, wi::Int) = sum((g.whens[k].nsample * size(g.whens[k].domain, 2) for k in 1:(wi - 1)); init = 0)

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

#= ---------------------------------------------------------------- initialization =#

"""
    _initialResiduals(g) -> Tuple{Vector{Any}, Int}

The residual code of the initial equations' instances (`res[k] = lhs - rhs`, by class in their order, then by column:
the rows of the causal level's `InitialMatching`) and their number.
"""
function _initialResiduals(g::Gen)::Tuple{Vector{Any}, Int}
  local body::Vector{Any} = Any[]
  local base::Int = 0
  for c in g.initClasses
    g.relBase = 0; g.relCount = 0
    local assign::Expr = :(res[$base + _col] = $(jl(g, c.lhs)) - $(jl(g, c.rhs)))
    push!(body, c.nslots == 0 ? Expr(:let, :(_col = 1), assign) : _loopCode(c.nslots, c.domain, collect(1:size(c.domain, 2)), assign))
    base += size(c.domain, 2)
  end
  return (body, base)
end

#= ---------------------------------------------------------------- loop families =#

"""
    _loopFamilyCode(g, f, template, blocks, el, matchRow, a0, relBases) -> Tuple{Vector{Any}, Expr}

The solve of the loop family `f` (`LOOPFAMILY_f!`): its tables (each member's instance columns and unknowns, a guess
per unknown: the last solution, at first the start values) and the body that solves the members it is called for,
each by Newton's method on the residuals of its instances (lhs - rhs, by the template's classes at the member's
columns), the unknowns set in place (a state derivative in du, an algebraic variable in a), the Jacobian by forward
differences; the last solution kept as the next guess (in Float64 solves); residuals or a Jacobian not finite, or a
singular Jacobian, make the unknowns NaN (the solver rejects the step). Not relations in a loop (events), nor a discrete unknown.
"""
function _loopFamilyCode(g::Gen, f::Int, template::Vector{Tuple{Int, Int}}, blocks::Vector{Vector{Int}}, el::InstanceTable,
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
    #= An algebraic variable's value before the instant is not kept: its current value was
       read (`peak = pre(u)` at u's step took the value after it, OpenModelica the one
       before); the scalarized path keeps it. =#
    info.kind == :alg && ns("pre() of the algebraic variable $(x.ref.name)")
    return jl(g, x.ref)   #= pre of a parameter =#
  elseif x isa Rel
    local k = :($(g.relBase) + (_col - 1) * $(g.relCount) + $(x.id))
    #= time and a parameter expression: a time event at its exact time =#
    return :($(_isTimeRelation(g, x) ? _timeRel : _rel)(z, p, mode, $k, $(jl(g, x.lhs)), $(jl(g, x.rhs)), $(x.op)))
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
    #= a relation with the probe of its abs() argument (`_withAbsProbe`): the probe's crossing kept, its value not =#
    x.op == :probed && return Expr(:block, args[2], args[1])
    #= a temporary's value (a tuple equation's call) =#
    x.op == :setlocal && return Expr(:(=), args...)
    x.op == :assertcall && return :(($(args[1])) || _assertFailed(p, $(args[3]), ($(args[3]), _col), t, $(args[2])))
    x.op == :terminate && return :(_terminate!(p, $(args[1])))
    x.op == :stringparam && return :(p.state.strings[$(x.args[1].v)])
    x.op == :seq && return Expr(:block, args...)
    x.op == :noEvent && return args[1]
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

"""
The parameters the equations read and, transitively, those their bindings read, in
dependency order, each with its binding as code over the others (nothing when the
binding is not expressible here: the value then cannot follow a changed dependency).
"""
function _parameterRules(g::Gen)
  local bindings::Vector{ParameterBinding} = parameterBindings(g, collect(keys(g.params)))
  local rules::Vector{Any} = Any[(g.paramSyms[b.name], Symbol[g.paramSyms[d] for d in b.deps], _bindingCode(g, b))
                                 for b in bindings]
  return (String[b.name for b in bindings], rules, Dict{String, Symbol}(b.name => b.kind for b in bindings))
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
function _functionDefinitions(mfs::Vector{SimCode.ModelicaFunction})
  local defs = Any[]; local outs = Dict{String, Int}()
  isempty(mfs) && return (defs, outs)
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

"""
The module of an array model from its variables and classes (`g`, the domain level of its simulation code:
`SimCode.Domains.domainSystem`), its Modelica functions' definitions, the initial values of the states, discrete
variables and algebraic variables (row-major), the states' fixed attributes and the element names: its causal level
(`SimCode.Domains.causalSchedule!`), the initialization and the code. `keepsArrays`: whether the model keeps arrays an algebraic loop may
go through (its own, or rolled components of a reduced model).
"""
function _emitModule(g::Gen, modelName::String, functionDefs::Vector{Any}, u0::Vector{Float64}, d0::Vector{Float64},
                     a0::Vector{Float64}, fixedMask::Vector{Bool}, stateNamesOrdered::Vector{String},
                     algNamesOrdered::Vector{String}, discreteNamesOrdered::Vector{String}; keepsArrays::Bool = true,
                     startRules::Vector{Tuple{Symbol, Int, IR}} = Tuple{Symbol, Int, IR}[],
                     startFixed::Vector{String} = String[], placeholderStarts::Bool = false)::Expr
  #= the domain level's passes and the causal level (SimCode.Domains) =#
  local schedule::Schedule = causalSchedule!(g; keepsArrays, full = FULL[], fixedMask)
  #= a terminate() in a when body or an initial algorithm: the run's stop (`_terminate!`, emitted for such a model only) =#
  local hasTerminate::Bool = any(w::WhenClass -> any(e -> _hasOp(e[3], :terminate), w.body), g.whens) ||
                             any(ia::AlgClass -> any(st::Stmt -> _stmtHasOp(st, :terminate), ia.body), g.initAlgs)
  #= the starts computed at each start: a placeholder of the translation's (`placeholderStarts`), or a rule that reads a
     parameter computed at the start =#
  local startsAtEachStart::Bool = placeholderStarts ||
    any(r::Tuple{Symbol, Int, IR} -> _readsParameterOf(r[3], g.startComputed), startRules)
  local (el::InstanceTable, nU::Int, matchRow::Vector{Int}) = (schedule.elements, schedule.nUnknowns, schedule.matchRow)
  local (batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}}, pos::Vector{Int}, producer::Vector{Int}, loops::LoopBlocks) =
    (schedule.batches, schedule.pos, schedule.producer, schedule.loops)
  local (hasArrays::Bool, large::BitVector) = (schedule.hasArrays, schedule.large)
  local (deps::Union{Nothing, Vector{Vector{Int}}}, pattern::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}}) =
    (schedule.deps, schedule.pattern)
  local solutions::Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}} = schedule.solutions
  local (needed::BitVector, relNeeded::BitVector, invariant::BitVector) = (schedule.needed, schedule.relNeeded, schedule.invariant)
  local (initUnknowns::Vector{Int}, nInit::Int, freeCols::Vector{Tuple{String, Int}}) =
    (schedule.initial.matchRow, schedule.initial.nRows, schedule.initial.freeColumns)
  local (initBody::Vector{Any}, _) = _initialResiduals(g)
  #= initial algorithms: code, and what they assign =#
  local initAlgOuts::Set{String} = _initAlgAssigned(g)
  g.relBase = 0; g.relCount = 0
  local initAlgBody = Any[]
  for ia in _initAlgOrder(g.initAlgs)
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
  #= the sample() calls of the classes (sample-defined Booleans): their ticks' flags after the when-equations', per
     instance =#
  local classSampleBases::Vector{Int} = zeros(Int, length(g.classes))
  local nClassSamples::Int = sum((w.nsample * size(w.domain, 2) for w in g.whens); init = 0)
  for (ci, c) in enumerate(g.classes)
    classSampleBases[ci] = nClassSamples
    nClassSamples += c.nsample * size(c.domain, 2)
  end
  #= Code: one loop per batch (rhsBody: the rows the state derivatives need, condBody: those the relations need,
     invariantBody: the time-invariant ones; Schedule). =#
  local body = Any[]; local rhsBody = Any[]
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
    local torn::TornBlock = schedule.torn[b]::TornBlock
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
      local (target::Ref, value::IR) = solutions[(ci, p)]::Tuple{Ref, IR}
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
    g.sampleBase = classSampleBases[ci]; g.sampleCount = c.nsample
    local solution::Union{Nothing, Tuple{Ref, IR}} = any(r::Int -> (r, matchRow[r]) in el.aliased, batch) ? nothing : solutions[(ci, p)]
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
                                   :(p.state.sampleStart[$k] = Float64($(jl(g, sx.args[1])));
                                     p.state.sampleInterval[$k] = Float64($(jl(g, sx.args[2]))))))
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
        #= a call of parameters only: once per pass of the event iteration (p.state.whenMemo), as the body
           fires once in an instant; the passes rerun a fired body, and an impure function (MSL
           impureRandom(id)) gave a new value in each: the initial events of ImpureGenerator never
           settled. A pure one gives the same value. =#
        push!(stmtsOf[b], :(p.d[$(_linear(info, idx))] = Float64(get!(() -> $(jl(g, value)), p.state.whenMemo, ($wi, $i, _col)))))
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
      #= a lifted cluster of discrete equations (`everyEvent`): wherever the discrete equations run in event mode, at
         every pass of every event and of the start's iteration (pre() the initial values), its conditions the events'
         sources =#
      local guard = w.everyEvent ? :mode : :(fire && ($(rises[b])))
      inst = inst === nothing ? Expr(:if, guard, blk) : Expr(b == 1 ? :if : :elseif, guard, blk, inst)
    end
    push!(whenConds, w.nslots == 0 ? Expr(:let, :(_col = 1), cond) : _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), cond))
    push!(whenBody, w.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), inst))
  end
  #= the classes' sample() calls (classSampleBases), after the when-equations' =#
  for (ci, c) in enumerate(g.classes)
    c.nsample > 0 || continue
    local samples::Dict{Int, Op} = Dict{Int, Op}()
    local collect!::Function = function (x)
      x isa Op && x.op == :sample && (samples[x.args[3].v] = x)
      x isa Op && foreach(collect!, x.args)
      x isa Rel && (collect!(x.lhs); collect!(x.rhs))
      x isa EvFn && collect!(x.arg)
      return nothing
    end
    collect!(c.lhs); collect!(c.rhs)
    for j in 1:size(c.domain, 2), id in 1:c.nsample
      local sx::Op = samples[id]
      (_parametersOnly(g, sx.args[1]) && _parametersOnly(g, sx.args[2])) ||
        ns("sample() with a start or interval that is not a parameter expression")
      local k::Int = length(sampleSchedule) + 1
      k == classSampleBases[ci] + (j - 1) * c.nsample + id || error("array code: the samples' numbering")
      local slots::Vector{Expr} = Expr[:($(jl(g, Slot(r))) = $(c.domain[r, j])) for r in 1:size(c.domain, 1)]
      push!(sampleSchedule, Expr(:let, Expr(:block, slots...),
                                 :(p.state.sampleStart[$k] = Float64($(jl(g, sx.args[1])));
                                   p.state.sampleInterval[$k] = Float64($(jl(g, sx.args[2]))))))
    end
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
  #= the relations of the asserts' conditions on continuous values as crossings (assertCrossings!), each instance's:
     a violation that starts and ends within a step is found where it starts (the scalarized path's too) =#
  local crossBody = Any[]
  local nCross = 0
  for a in g.asserts
    local rels::Vector{Tuple{Union{Nothing, IR}, IR, IR}} = _assertRelations!(Tuple{Union{Nothing, IR}, IR, IR}[], g, a.cond, nothing)
    isempty(rels) && continue
    local (base, nc) = (nCross, length(rels))
    local crossing::Function = (reached::Union{Nothing, IR}, l::IR, r::IR) -> begin
      local zc = :(Float64($(jl(g, l))) - Float64($(jl(g, r))))
      reached === nothing ? zc : :($(jl(g, reached)) ? $zc : 1.0)
    end
    local inst = Expr(:block, Any[:(out[$base + (_col - 1) * $nc + $j] = $(crossing(rel...))) for (j, rel) in enumerate(rels)]...)
    push!(crossBody, a.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(a.nslots, a.domain, collect(1:size(a.domain, 2)), inst))
    nCross += nc * size(a.domain, 2)
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
  #= the starts that read a parameter, computed again from the parameter values (before the layout: their parameters
     are in it) =#
  local startStmts::Vector{Any} = Any[]
  #= a discrete variable's start that reads a parameter the initialization or the initial algorithms compute
     (fixed = false): again after them, but where an initial algorithm assigns the variable =#
  local freeStartStmts::Vector{Any} = Any[]
  local free::Set{String} = Set{String}(g.freeParams)
  local initAssigned::Set{String} = _initAlgAssigned(g)
  local discreteOwner::Dict{Int, String} = Dict{Int, String}()
  for (nm, info) in g.vars
    info.kind == :discrete || continue
    for j in 1:prod(info.dims; init = 1)
      discreteOwner[info.offset + j] = nm
    end
  end
  for (buffer, k, x) in startRules
    local v = jl(g, x)
    push!(startStmts, buffer === :u ? :(u0[$k] = Float64($v)) : :(p.d[$k] = Float64($v); p.dpre[$k] = p.d[$k]))
    (buffer === :d && _readsParameterOf(x, free) && !(get(discreteOwner, k, "") in initAssigned)) &&
      push!(freeStartStmts, :(p.d[$k] = Float64($v); p.dpre[$k] = p.d[$k]))
  end
  local usedParams = collect(keys(g.paramSyms))
  local (paramOrder, rules, bindingKinds) = _parameterRules(g)
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
  #= a parameter the translation has no value of is computed at each start (parameterValues) from its binding: it needs
     the binding as code (a call with a function argument, a comprehension over such an array have none) and sizes or
     subscripts nothing; one with a value that reads it keeps that value (the translation computed it without it) =#
  local computable::Set{Symbol} = Set{Symbol}(sym for (sym, _, code) in rules if code !== nothing)
  for nm in g.startComputed
    nm in g.structural && ns("parameter $(nm), computed at the start, used structurally")
    (haskey(g.paramSyms, nm) && !(g.paramSyms[nm] in computable)) &&
      ns("the value of parameter $(nm): computed at the start, and its binding has no array code")
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
  #= the canonical names (dots as underscores) too, where they name one parameter or element only (`c.k` and `c_k`
     do not) =#
  local canonicalNames = Dict{String, Vector{Symbol}}()
  local canonicalElements = Dict{String, Vector{Tuple{Symbol, Any}}}()
  for nm in paramOrder
    local sym = g.paramSyms[nm]
    paramNames[nm] = sym
    push!(get!(Vector{Symbol}, canonicalNames, _OMBACKEND.canonicalName(nm)), sym)
    local info = g.vars[nm]
    isempty(info.dims) && continue
    local subs = vec([reverse(t) for t in Iterators.product(reverse([1:d for d in info.dims])...)])
    for (en, t) in zip(info.var.names, subs)
      paramElements[en] = (sym, t)
      push!(get!(Vector{Tuple{Symbol, Any}}, canonicalElements, _OMBACKEND.canonicalName(en)), (sym, t))
    end
  end
  for (cn, syms) in canonicalNames
    (length(syms) == 1 && !haskey(paramNames, cn) && !haskey(paramElements, cn)) && (paramNames[cn] = syms[1])
  end
  for (cn, els) in canonicalElements
    (length(els) == 1 && !haskey(paramElements, cn) && !haskey(paramNames, cn)) && (paramElements[cn] = els[1])
  end
  #= structural parameters, and the ones a start reads that the array code does not compute =#
  local fixedParams = Symbol[g.paramSyms[nm] for nm in paramOrder
                             if nm in g.structural || nm in startFixed]
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
    #= the String parameters the code reads: their translated values (the run state's), names (Modelica's and with
       underscores) to their slots, those a binding or the structure reads (an override refused) =#
    const STRING_VALUES = $(String[s[2] for s in g.strings])
    const STRING_PARAMETERS = $(Dict{String, Int}(Iterators.flatten(((s[1] => k), (replace(s[1], "." => "_") => k)) for (k, s) in enumerate(g.strings))))
    const FIXED_STRINGS = $(Set{Int}(k for (k, s) in enumerate(g.strings) if s[1] in g.structural))
    #= (parameter, the parameters its binding reads, its binding as a function of them) in
       dependency order =#
    const PARAMETER_RULES = Any[$(ruleExprs...)]
    #= parameters the translation has no value of: their rules run at each start =#
    const START_COMPUTED = Set{Symbol}($(Symbol[g.paramSyms[nm] for nm in g.startComputed if haskey(g.paramSyms, nm)]))
    #= the ones the translation knew nothing of (a scalar, or a whole call's result): a parameter with a value computed
       without them (an if-branch not taken) =#
    const START_OPAQUE = Set{Symbol}($(Symbol[g.paramSyms[nm] for nm in g.startComputed
                                              if haskey(g.paramSyms, nm) && get(bindingKinds, nm, :none) in (:scalar, :each, :value)]))
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
    #= the slots of relations of time and parameters (`_timeRel`): their crossings are time events =#
    const TIME_SLOT = $(_timeRelationSlots(g, nRelTotal, relBases, whenRelBases, algRelBases))
    const USES_INITIAL = $(g.usesInitial)
    #= the sample() instances (their starts and intervals each problem's, set by sampleSchedule! after the initialization) =#
    const N_SAMPLES = $(length(sampleSchedule))
    #= the round-off of a tick st + k dt, relative to the largest magnitude in it (omc's SAMPLE_EPS 1e-14): ticks within
       it of each other or of the start or stop time are one instant =#
    const SAMPLE_EPS = 1.0e-14
    const N_WHENS = $(nWhenInst)
    #= discrete variables equations compute (unknowns kept in the discrete buffer, set by the event iteration) =#
    const N_DISCRETE_UNKNOWNS = $(length(g.discUnknown))

    #= The parameter values with `overrides` (Modelica names of parameters or of their
       elements => values) applied; parameters whose bindings read a changed one are
       computed again. =#
    #= The parameter values (PARAMS) with `overrides` (Modelica names of parameters or of their elements => values),
       in the rules' dependency order:
       - one overridden whole takes the given value; an element override is set after its parameter's rule, when that
         runs, else into its value;
       - one computed at the start (START_COMPUTED) from its rule, changed for its readers when a dependency changed
         or an element of it was overridden;
       - one with a value computed again when a dependency changed, changed when its value differs from the
         translated one; one whose changed dependencies are all START_OPAQUE keeps its value without a rule (the
         translation computed it without them);
       - a structural one (FIXED_PARAMETERS) the overrides change is an error: the code holds its translated value. =#
    function parameterValues(overrides::AbstractDict = Dict{String, Any}())
      local vals = Dict{Symbol, Any}(k => copy(v) for (k, v) in pairs(PARAMS))
      local whole = Set{Symbol}()
      local later = Dict{Symbol, Vector{Any}}()
      for (name, v) in overrides
        local name_s = string(name)
        if haskey(PARAMETER_NAMES, name_s)
          local sym = PARAMETER_NAMES[name_s]
          vals[sym] = $(_overrideValue)(name_s, vals[sym], v)
          push!(whole, sym)
        elseif haskey(PARAMETER_ELEMENTS, name_s)
          local (sym, t) = PARAMETER_ELEMENTS[name_s]
          push!(get!(Vector{Any}, later, sym), (t, $(_overrideValue)(name_s, vals[sym][t...], v)))
        else
          error(string("array model: no parameter named ", name_s, " (or it is not used by the equations)"))
        end
      end
      for sym in union(whole, keys(later))
        sym in FIXED_PARAMETERS && error(string("array model: parameter ", sym, " is structural (array sizes, subscripts, if-equation conditions) or read by a start the code does not compute; translate again"))
      end
      local changed = copy(whole)
      for (sym, deps, rule) in PARAMETER_RULES
        if !(sym in whole)
          local atStart = sym in START_COMPUTED
          local depChanged = any(d -> d in changed, deps)
          local opaqueOnly = all(d -> !(d in changed) || d in START_OPAQUE, deps)
          if atStart || (depChanged && !(rule === nothing && opaqueOnly))
            rule === nothing && error(string("array model: parameter ", sym, " depends on a changed parameter, but its binding cannot be computed here"))
            #= of the translated value's type (a Real array bound to Integers: {m, n}) =#
            vals[sym] = convert(typeof(getfield(PARAMS, sym)), rule(vals))
          end
          (atStart ? depChanged : vals[sym] != getfield(PARAMS, sym)) && push!(changed, sym)
        end
        local elems = get(later, sym, nothing)
        if elems !== nothing
          for (t, v) in elems
            vals[sym][t...] = v
          end
          push!(changed, sym)
        end
      end
      for sym in FIXED_PARAMETERS
        sym in changed && error(string("array model: parameter ", sym, " is structural (array sizes, subscripts, if-equation conditions) and changes with the overrides; translate again"))
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

    $(hasTerminate ? quote
        #= a terminate() reached (its message, p.state.terminated): the run ends after the event, at the start if the
           initialization reached it =#
        _terminate!(p, message) = (p.state.terminated === nothing && (p.state.terminated = string(message)); nothing)
        function _terminateReached!(integrator)
          local message = integrator.p.state.terminated
          message === nothing && return true
          integrator.p.state.terminated = nothing
          @info "Modelica terminate() reached" message time = integrator.t
          DifferentialEquations.SciMLBase.terminate!(integrator)
          return true
        end
      end : nothing)

    #= p.state.probe: _violated! looks for a violation without reporting it =#
    function _assertFailed(p, k::Int, instance, t, message)
      local (condition, _, warning) = ASSERTS[k]
      if p.state.probe
        (warning && instance in p.state.warned) || (p.state.violated = true)
        return true
      end
      warning || throw($(ModelicaAssertionError)(Float64(t), message, condition))
      instance in p.state.warned ||
        (push!(p.state.warned, instance); @warn string("Assertion violated at time ", t, ": ", message) condition)
      return true
    end

    #= Whether an assert is violated at (u, t) (a warning reported already does not count). =#
    function _violated!(u, p, t)
      p.state.probe = true; p.state.violated = false
      try
        checkAsserts(u, p, t)
      finally
        p.state.probe = false
      end
      return p.state.violated
    end

    #= After a step: an assert violated at its end is reported at the first violating time in
       the step (bisection on the step's interpolation), as the MTK path reports the crossing. =#
    function _assertStep(u, t, integrator)
      local p = integrator.p
      try
        $(nCross == 0 ? nothing : quote
            #= a violation at a crossing is decided halfway to the next check, not at the step that ends at it =#
            _assertPending!(integrator)
            p.state.pending[1] == t && return false
          end)
        _violated!(u, p, t) || return false
        #= violated within the step, with its discrete state (the step start's: an event that ended it changed them)
           and the state just before an event that ended it (a reinit moved it): at the first violating time, by
           bisection; else the event's, at it =#
        local stepStart = p.state.stepStart
        local uLeft = p.state.leftLimit[1] == t ? p.state.leftLimit[2] : u
        local lo = integrator.tprev; local hi = t
        $(_withDiscreteState)(() -> _violated!(uLeft, p, t), p, stepStart) || return (checkAsserts(u, p, t); false)
        $(_withDiscreteState)(p, stepStart) do
          if hi > lo
            for _ in 1:100
              local mid = (lo + hi) / 2
              _violated!(integrator(mid), p, mid) ? (hi = mid) : (lo = mid)
              hi - lo <= 1.0e-12 * max(1.0, abs(hi)) && break
            end
          end
          checkAsserts(hi == t ? uLeft : integrator(hi), p, hi)
        end
        return false
      finally
        p.state.stepStart = $(_discreteState)(p)
      end
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

    $(nCross == 0 ? nothing : quote
        #= The asserts' relations as crossings (the right-hand side's equations at (u, t) first), checked at the step
           ends: a violation that starts and ends within a step is found where it starts. =#
        function assertCrossings!(out, u, t, integrator)
          local p = integrator.p
          local du = similar(u); local a = similar(u, $nAlg)
          equations!(du, a, u, p, t, nothing, false)
          $(paramLocals...)
          @inbounds begin
            $(crossBody...)
          end
          return nothing
        end

        #= A crossing (its time and state): decided at the next check, halfway to it, as the scalarized path: a relation
           that touches its boundary (a piston at its top dead center) is past it by rounding there and holds again; one
           that crosses fails halfway too, reported at the crossing, or after it where the relation is at its boundary
           there (`x <= lim` at x = lim holds) at the first violation, by bisection on the step's interpolation. =#
        function _assertPending!(integrator)
          local p = integrator.p
          local (τp, up) = p.state.pending
          (isnan(τp) || integrator.t <= τp) && return nothing
          p.state.pending = (NaN, Float64[])
          local τm = (τp + integrator.t) / 2
          #= within the step (its discrete state the step start's: an event that ended it changed them) =#
          local within = integrator.tprev <= τm
          local decide = () -> begin
            local (um, tm) = within ? (integrator(τm), τm) : (copy(integrator.u), Float64(integrator.t))
            _violated!(um, p, tm) || return nothing
            _violated!(up, p, τp) && return (checkAsserts(up, p, τp); nothing)
            (within && integrator.tprev <= τp) || return (checkAsserts(um, p, tm); nothing)
            local lo = τp; local hi = tm
            for _ in 1:100
              local mid = (lo + hi) / 2
              _violated!(integrator(mid), p, mid) ? (hi = mid) : (lo = mid)
              hi - lo <= 1.0e-12 * max(1.0, abs(hi)) && break
            end
            checkAsserts(integrator(hi), p, hi)
            return nothing
          end
          within ? $(_withDiscreteState)(decide, p, p.state.stepStart) : decide()
          return nothing
        end

        function affectAssertCrossing!(integrator, _)
          _assertPending!(integrator)
          isnan(integrator.p.state.pending[1]) && (integrator.p.state.pending = (Float64(integrator.t), copy(integrator.u)))
          DifferentialEquations.u_modified!(integrator, false)
          return nothing
        end

        #= at the end, a violation still pending is decided on the final state =#
        function _assertPendingAtEnd!(integrator)
          local p = integrator.p
          local (τp, up) = p.state.pending
          isnan(τp) && return nothing
          p.state.pending = (NaN, Float64[])
          _violated!(integrator.u, p, integrator.t) || return nothing
          _violated!(up, p, τp) ? checkAsserts(up, p, τp) : checkAsserts(integrator.u, p, integrator.t)
          return nothing
        end
      end)

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

    #= The starts of states and discrete variables that read a parameter, from the parameter values (an override
       reaches them). =#
    function parameterStarts!(u0, p)
      $(paramLocals...)
      @inbounds begin
        $(startStmts...)
      end
      return nothing
    end

    $(isempty(freeStartStmts) ? nothing : quote
        #= The starts of discrete variables that read a parameter with fixed = false, from its value the
           initialization and the initial algorithms computed (after them). =#
        function freeParameterStarts!(p)
          $(paramLocals...)
          @inbounds begin
            $(freeStartStmts...)
          end
          return nothing
        end
      end)

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
                              zeros($nAlg), p.blocks, p.state, p.model)
      equations!(similar(u), a, u, q, t, nothing, false)
      return a
    end

    #= Event iteration at (u, t), in sweeps as OpenModelica's updateDiscreteSystem (and the MTK
       path, relationRefresh.jl): within a sweep pre() keeps the values from before it, and the
       relations, equations and fired when bodies are evaluated again until they settle (an
       equation reading a variable a when body just set: ch = change(k)); a sweep that changed
       a discrete value starts the next one from the new values, until one changes nothing. =#
    function eventIteration!(u, p, t, fire::Bool, sampleInstant::Bool = false)
      local du = similar(u); local a = similar(u, $nAlg)
      for _ in 1:50
        p.dpre .= p.d; p.upre .= u; p.ureinit .= u
        empty!(p.state.whenMemo)
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
        #= at a sample instant the samples are true for the first pass only (OpenModelica's
           updateDiscreteSystem: functionDAE once, the samples off, then iterated) =#
        sampleInstant && fill!(p.sampleActive, false)
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
    function _eventAt!(integrator, sampleInstant::Bool = false)
      #= the state just before the event (a reinit moves it): the asserts' left limit =#
      integrator.p.state.leftLimit = (Float64(integrator.t), copy(integrator.u))
      eventIteration!(integrator.u, integrator.p, integrator.t, true, sampleInstant) &&
        return $(hasTerminate ? :(_terminateReached!(integrator)) : true)
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
       center). A time event (a relation of time and parameters crossed): the event iteration with
       the relations literal at the instant (H = 0), as at a sample instant: one on its boundary
       there changes with it (a brake's normal force ramped to 0 at the time event: f <= 0 holds). =#
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
      local timed::Bool = false
      for k in eachindex(events)
        (events[k] != 0 && RELATION_SLOT[k] && p.rel[k] == before[k] && p.rel[k] == (events[k] > 0)) && (p.rel[k] = !p.rel[k])
        timed |= events[k] != 0 && TIME_SLOT[k]
      end
      local _hyst = p.hyst[1]
      timed && (p.hyst[1] = 0.0)
      try
        _eventAt!(integrator)
      finally
        p.hyst[1] = _hyst
      end
      _logDiscretes!(p, t)
      DifferentialEquations.u_modified!(integrator, true)
      return nothing
    end

    #= A sample instant: the samples due now are true for the first pass of the event iteration,
       false in the passes after it (OpenModelica's): what reads sample() other than as a when's
       rising edge takes its fall at the instant (`y = not u and pre(u)` of `u = sample(...)`, CDL
       FallingEdge: y rises and falls there; `k = if u then pre(k) + 1 else pre(k)` counts once),
       and the when-conditions take that value too, so the next instant is a rising edge. =#
    function affectSample!(integrator)
      local t = integrator.t; local p = integrator.p; local u = integrator.u
      local tspan = integrator.sol.prob.tspan
      local (starts, intervals, stops) = (p.state.sampleStart, p.state.sampleInterval, p.state.sampleStops)
      local i = searchsortedfirst(stops, t)
      local next = i < length(stops) ? stops[i + 1] : Inf
      for id in eachindex(starts)
        local k = round((t - starts[id]) / intervals[id])
        local (ts, w) = _sampleTick(starts[id], intervals[id], k, tspan)
        #= active at the stop its tick joined (sampleTimes): at or after it within its round-off, before the next =#
        p.sampleActive[id] = k >= 0 && 0.0 <= ts - t <= w && ts < next
      end
      #= the relations literal at the instant (H = 0): one whose crossing function is 0 there (a
         threshold the tick reaches, `time >= pulseStart + 0.2` at 1) changes with it, as
         OpenModelica takes that state event together with the time event =#
      local _hyst = p.hyst[1]
      p.hyst[1] = 0.0
      try
        _eventAt!(integrator, true)
      finally
        p.hyst[1] = _hyst
      end
      fill!(p.sampleActive, false)
      _logDiscretes!(p, t)
      DifferentialEquations.u_modified!(integrator, true)
      return nothing
    end

    #= Tick k of sample(st, dt) and its round-off w (SAMPLE_EPS of the largest magnitude in st + k dt): at the start
       or stop time when within w of it (sample(-0.9, 0.3): 0). =#
    function _sampleTick(st, dt, k, tspan)
      local tk = st + k * dt
      local w = SAMPLE_EPS * max(1.0, abs(st), abs(k * dt), abs(tk))
      local ts = abs(tk - tspan[1]) <= w ? Float64(tspan[1]) : abs(tk - tspan[2]) <= w ? Float64(tspan[2]) : tk
      return (ts, w)
    end

    #= the sample instants of p's samples over tspan, kept as p's stops (affectSample! activates a sample at the stop its
       tick joined) =#
    function sampleTimes(p, tspan)
      local ticks = Tuple{Float64, Float64}[]
      local stops = p.state.sampleStops
      for (st, dt) in zip(p.state.sampleStart, p.state.sampleInterval)
        (isfinite(st) && isfinite(dt)) || error("array model: sample start and interval must be finite")
        dt > 0 || error("array model: sample interval must be positive")
        local k = max(0.0, ceil((tspan[1] - st) / dt) - 1)
        while true
          local (ts, w) = _sampleTick(st, dt, k, tspan)
          ts > tspan[2] && break
          ts >= tspan[1] && push!(ticks, (ts, w))
          k += 1
        end
      end
      #= ticks within round-off of each other are one instant, a tick joining the stop before it (3 * 0.2 and 2 * 0.3;
         sample(-100, 0.1) and sample(0, 0.1) near 0); affectSample! activates each sample at the stop its tick joined =#
      sort!(ticks)
      empty!(stops)
      for (ts, w) in ticks
        (isempty(stops) || ts - stops[end] > w) && push!(stops, ts)
      end
      return copy(stops)
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
      #= the overrides of String parameters into the run state's values, the others parameterValues' =#
      local strings = copy(STRING_VALUES)
      local numeric = Dict{String, Any}()
      for (name, v) in parameters
        local k = get(STRING_PARAMETERS, string(name), 0)
        if k == 0
          numeric[string(name)] = v
        else
          v isa AbstractString ||
            throw(ArgumentError(string("array model: the override of parameter ", name, ", ", repr(v), ", is not a value of its type String")))
          k in FIXED_STRINGS &&
            error(string("array model: parameter ", name, " is read by a parameter's binding or the structure; translate again"))
          strings[k] = String(v)
        end
      end
      return $(paramsType)(flatParameters(isempty(numeric) && isempty(START_COMPUTED) ? PARAMS : parameterValues(numeric)),
                           zeros(Bool, N_RELATIONS), copy(D0), copy(D0), zeros(Bool, N_WHENS),
                           zeros(Bool, N_SAMPLES), [false], [false], copy(U0), copy(U0),
                           zeros(Bool, N_WHENS), Tuple{Float64, Vector{Float64}}[], zeros(N_RELATIONS), [0.0], [Float64(tspan[2])],
                           zeros($nAlg),
                           $(BlockSolver)[$(BlockSolver)(BLOCK_PATTERNS[k], BLOCK_COLORS[k], BLOCK_GUESSES[k]) for k in eachindex(BLOCK_PATTERNS)],
                           $(ArrayRuntimeState)(N_SAMPLES, strings), @__MODULE__)
    end

    #= The problem and its event callbacks, as getMTKProblem returns them. =#
    function $(Symbol(modelName, "Model"))(tspan = (0.0, 1.0); parameters::AbstractDict = Dict{String, Any}())
      local p = modelParameters(tspan; parameters = parameters)
      local u0 = copy(U0)
      #= the starts from the parameters: with overrides, and at each start where one reads a parameter computed there =#
      if $(startsAtEachStart ? true : :(!isempty(parameters)))
        parameterStarts!(u0, p)
        p.upre .= u0; p.ureinit .= u0
      end
      N_INIT > 0 && (u0 = initialize(u0, p, tspan[1]))
      $(isempty(initAlgBody) ? nothing : :(initialAlgorithms!(u0, p, tspan[1])))
      $(isempty(freeStartStmts) ? nothing : :(freeParameterStarts!(p)))
      N_SAMPLES == 0 || sampleSchedule!(p)
      #= relations, when-conditions and the discrete variables equations compute at the start (no when fires there;
         without events the only evaluation of those discrete variables) =#
      (N_RELATIONS > 0 || N_WHENS > 0 || N_DISCRETE_UNKNOWNS > 0) &&
        (eventIteration!(u0, p, tspan[1], false) || error("array model: the relations and discrete values at the start do not settle"))
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
      N_SAMPLES == 0 || push!(cbs, DiffEqCallbacks.PresetTimeCallback(sampleTimes(p, tspan), affectSample!))
      if !isempty(ASSERTS) || FUNCTION_ASSERTS
        $(nCross == 0 ?
          :(push!(cbs, DiscreteCallback(_assertStep, integrator -> nothing;
                                        initialize = (c, u, t, integrator) -> (integrator.p.state.stepStart = $(_discreteState)(integrator.p);
                                                                               checkAsserts(u, integrator.p, t)),
                                        save_positions = (false, false)))) :
          quote
            push!(cbs, DiscreteCallback(_assertStep, integrator -> nothing;
                                        initialize = (c, u, t, integrator) -> (integrator.p.state.pending = (NaN, Float64[]);
                                                                               integrator.p.state.stepStart = $(_discreteState)(integrator.p);
                                                                               checkAsserts(u, integrator.p, t)),
                                        finalize = (c, u, t, integrator) -> _assertPendingAtEnd!(integrator),
                                        save_positions = (false, false)))
            #= the sign at the step ends only (no points within a step: an equations! each) =#
            push!(cbs, VectorContinuousCallback(assertCrossings!, affectAssertCrossing!, $nCross;
                                                rootfind = DifferentialEquations.SciMLBase.RightRootFind,
                                                interp_points = 0, save_positions = (false, false)))
          end)
      end
      #= a solve of the problem starts from its state after the initialization (again, where it is solved again) =#
      p.state.atStart = $(_startState)(p)
      pushfirst!(cbs, DiscreteCallback((u, t, integrator) -> false, integrator -> nothing;
                                       initialize = (c, u, t, integrator) -> $(_restoreStart!)(integrator.p),
                                       save_positions = (false, false)))
      local cb = isempty(cbs) ? nothing : OMBackend.CodeGeneration._joinCallbacks(cbs...)
      $(hasTerminate ? quote
          #= a terminate() at the initialization: the problem ends at its start =#
          p.state.terminated === nothing || return (ODEProblem(f, u0, (tspan[1], tspan[1]), p), cb)
        end : nothing)
      return (ODEProblem(f, u0, tspan, p), cb)
    end

    function simulate(tspan = (0.0, 1.0), solver = Tsit5(); parameters::AbstractDict = Dict{String, Any}(), kwargs...)
      local (prob, callbacks) = $(Symbol(modelName, "Model"))(tspan; parameters = parameters)
      OMBackend.CodeGeneration._resetRHSError!()
      local sol = $(hasTerminate ?
                    :(prob.p.state.terminated === nothing ? DifferentialEquations.solve(prob, solver; callback = callbacks, kwargs...) :
                      $(solveStoppedAtStart)(prob, solver, something(prob.p.state.terminated); callback = callbacks, kwargs...)) :
                    :(DifferentialEquations.solve(prob, solver; callback = callbacks, kwargs...)))
      #= an error that every smaller step met ends the run =#
      OMBackend.CodeGeneration._rethrowRHSError(sol)
      return sol
    end
  end
  return Expr(:module, true, Symbol(modelName), stripBeginBlocks(code))
end

"""
The relations of an assert's condition `x` (evaluated without events: an `Op` of the comparison) that compare
continuous values (`_continuous`), each `(reached, lhs, rhs)`: where the condition's short-circuit reaches it, `nothing`
for always (`x <= 0 or sqrt(x) < 2`: sqrt(x) of a negative threw), as the scalarized path's crossings. Through and, or,
not and if-expressions; not inside another expression (a call's argument).
"""
function _assertRelations!(rels::Vector{Tuple{Union{Nothing, IR}, IR, IR}}, g::Gen, x::IR,
                           reached::Union{Nothing, IR})::Vector{Tuple{Union{Nothing, IR}, IR, IR}}
  x isa Op || return rels
  local both::Function = (c::IR) -> reached === nothing ? c : Op(:&&, IR[reached, c])
  if x.op in (:&&, :||)
    _assertRelations!(rels, g, x.args[1], reached)
    _assertRelations!(rels, g, x.args[2], both(x.op == :&& ? x.args[1] : Op(:!, IR[x.args[1]])))
  elseif x.op == :!
    _assertRelations!(rels, g, x.args[1], reached)
  elseif x.op == :if && length(x.args) == 3
    _assertRelations!(rels, g, x.args[1], reached)
    _assertRelations!(rels, g, x.args[2], both(x.args[1]))
    _assertRelations!(rels, g, x.args[3], both(Op(:!, IR[x.args[1]])))
  elseif x.op in (:<, :<=, :>, :>=) && length(x.args) == 2 && any(a::IR -> _continuous(g, a), x.args)
    push!(rels, (reached, x.args[1], x.args[2]))
  end
  return rels
end

"""
`x` with the String parameters it reads (`Op(:stringparam)`) as their translated values, those parameters fixed
(`Gen.structural`: an override of one is refused).
"""
function _stringValues!(g::Gen, x::IR)::IR
  x isa Op || return x
  x.op == :stringparam && (push!(g.structural, x.args[3].v); return Lit(x.args[2].v))
  return Op(x.op, IR[_stringValues!(g, a) for a in x.args])
end

"""
    _overrideValue(name, translated, v)

The value `v` of an override of the parameter (or element) `name` as the type of its translated value `translated`, of
its size; an ArgumentError naming the parameter for a value of another kind (a String for a Real, a fraction for an
Integer, an array of another size).
"""
function _overrideValue(name::String, translated, v)
  local bad::Function = () -> throw(ArgumentError(string("array model: the override of parameter ", name, ", ", repr(v),
                                                        ", is not a value of its type ", typeof(translated),
                                                        translated isa AbstractArray ? string(" of size ", size(translated)) : "")))
  local x::typeof(translated) = try
    convert(typeof(translated), v)
  catch err
    err isa Union{MethodError, InexactError, ArgumentError} || rethrow()
    bad()
  end
  (translated isa AbstractArray && size(x) != size(translated)) && bad()
  return x
end

"Whether `x` reads one of the parameters `names`."
function _readsParameterOf(x::IR, names::Set{String})::Bool
  x isa Ref && return x.name in names || any(s::IR -> _readsParameterOf(s, names), x.subs)
  x isa Op && return any(a::IR -> _readsParameterOf(a, names), x.args)
  x isa Rel && return _readsParameterOf(x.lhs, names) || _readsParameterOf(x.rhs, names)
  x isa EvFn && return _readsParameterOf(x.arg, names)
  x isa Pre && return _readsParameterOf(x.ref, names)
  return false
end

"Whether `x` has an operation `op` (`Op`, in the operations' arguments)."
_hasOp(x::IR, op::Symbol)::Bool = x isa Op && (x.op == op || any(a::IR -> _hasOp(a, op), x.args))

"""
    _initAlgOrder(algs) -> Vector{AlgClass}

The initial algorithms in an order that runs each after those that assign what it reads (as written otherwise, and for
a cycle), as the initialization solves them: an initial section reading another arm's target ran first and read its
start (MSL OnDelay's `delaySignal := u` before `u := time >= 0`: false; the MTK code runs it late, true).
"""
function _initAlgOrder(algs::Vector{AlgClass})::Vector{AlgClass}
  length(algs) <= 1 && return algs
  local writes::Vector{Set{String}} = Set{String}[_stmtNames!(Set{String}(), a.body, false) for a in algs]
  local reads::Vector{Set{String}} = Set{String}[_stmtNames!(Set{String}(), a.body, true) for a in algs]
  local done::BitVector = falses(length(algs))
  local out::Vector{AlgClass} = AlgClass[]
  while length(out) < length(algs)
    local ready::Union{Nothing, Int} = findfirst(i::Int -> !done[i] &&
      all(j::Int -> done[j] || j == i || isdisjoint(writes[j], reads[i]), eachindex(algs)), eachindex(algs))
    local k::Int = ready === nothing ? findfirst(!, done) : ready
    done[k] = true
    push!(out, algs[k])
  end
  return out
end

"The names the statements read outside pre() (`reads`) or assign, into `out`."
function _stmtNames!(out::Set{String}, stmts::Vector{Stmt}, reads::Bool)::Set{String}
  local exp!::Function = (x::IR) -> reads && _readNames!(out, x)
  for st in stmts
    if st isa SAssign
      reads ? (exp!(st.value); foreach(exp!, st.target.subs)) : push!(out, st.target.name)
    elseif st isa Union{SIf, SWhen}
      foreach(b::Tuple{IR, Vector{Stmt}} -> (exp!(b[1]); _stmtNames!(out, b[2], reads)), st.branches)
    elseif st isa SFor
      _stmtNames!(out, st.body, reads)
    elseif st isa SWhile
      exp!(st.cond)
      _stmtNames!(out, st.body, reads)
    elseif st isa SCall
      exp!(st.call)
    elseif st isa SAssert
      exp!(st.cond); exp!(st.msg)
    end
  end
  return out
end

"The names `x` reads outside pre() (a reference's subscripts too), into `out`."
function _readNames!(out::Set{String}, x::IR)::Set{String}
  if x isa Ref
    push!(out, x.name)
    foreach(s::IR -> _readNames!(out, s), x.subs)
  elseif x isa Rel
    _readNames!(out, x.lhs); _readNames!(out, x.rhs)
  elseif x isa Op
    foreach(a::IR -> _readNames!(out, a), x.args)
  elseif x isa EvFn
    _readNames!(out, x.arg)
  end
  return out
end

"Whether the statement `st` has an operation `op` (`_hasOp`)."
function _stmtHasOp(st::Stmt, op::Symbol)::Bool
  st isa SCall && return _hasOp(st.call, op)
  st isa SIf && return any(b::Tuple{IR, Vector{Stmt}} -> _hasOp(b[1], op) || any(s::Stmt -> _stmtHasOp(s, op), b[2]), st.branches)
  st isa Union{SFor, SWhile} && return any(s::Stmt -> _stmtHasOp(s, op), st.body)
  return false
end

"""
    generateArrayODECode(sc::SimCode.SIM_CODE, modelName) -> (modelName, moduleExpr) or (nothing, reason)

The array-preserving module of the simulation code of a model with its arrays kept (`SimCode.lowerArraySystem`),
through its domain level (`SimCode.Domains.domainSystem`) and causal level (`SimCode.Domains.causalSchedule!`); or
`nothing` and the reason the model is outside the supported scope.
"""
function generateArrayODECode(sc::SimCode.SIM_CODE, modelName::String)
  try
    local (functionDefs::Vector{Any}, _) = _functionDefinitions(sc.functions)
    local ds::DomainSystem = domainSystem(sc)
    return (modelName, _emitModule(ds.g, modelName, functionDefs, ds.u0, ds.d0, ds.a0, ds.fixedMask, ds.stateNames,
                                   ds.algebraicNames, ds.discreteNames; keepsArrays = ds.keepsArrays,
                                   startRules = ds.startRules, startFixed = ds.startFixed, placeholderStarts = ds.placeholderStarts))
  catch err
    err isa Union{NotSupported, Passes.IndexReduction.NotHandled} || rethrow(err)
    return (nothing, err isa NotSupported ? err.msg : sprint(showerror, err))
  end
end

"""
A parameter's binding as code over the parameters it reads (`_bindingCode`, of the domain level's `ParameterBinding`):
a scalar's value; an array's its elements, shaped as the parameter's value, or one value for every element (`each`);
`nothing` where it has none.
"""
function _bindingCode(g::Gen, b::ParameterBinding)
  b.kind == :none && return nothing
  local dims::Vector{Int} = g.vars[b.name].dims
  #= a String parameter it reads is its translated value: an override of it does not reach the binding (refused) =#
  b = ParameterBinding(b.name, b.deps, b.kind, IR[_stringValues!(g, x) for x in b.values])
  try
    local locals::Vector{Expr} = Expr[:(local $(g.paramSyms[d])::$(typeof(_paramValue!(g, d))) = vals[$(QuoteNode(g.paramSyms[d]))])
                                      for d in b.deps]
    local value::Any = if b.kind in (:scalar, :value)
      jl(g, b.values[1])
    elseif b.kind == :each
      :(fill($(jl(g, b.values[1])), $(dims...)))
    else
      local vect::Expr = Expr(:vect, Any[jl(g, x) for x in b.values]...)
      length(dims) == 1 ? vect : :(permutedims(reshape($vect, $(reverse(dims)...)), $(length(dims)):-1:1))
    end
    return :((vals) -> begin
      $(locals...)
      $value
    end)
  catch err
    err isa NotSupported || rethrow(err)
    return nothing
  end
end

end #= module ArrayODEGen =#
