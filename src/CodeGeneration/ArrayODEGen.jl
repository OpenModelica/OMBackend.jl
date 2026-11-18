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
import ..CodeGeneration: stripBeginBlocks, ModelicaAssertionError, solveStoppedAtStart, _autoDtReset!, _finiteFirstStep!,
  ode_to_dae
import ..CodeGeneration: DelayState, delayedValue, emptyDelayHistories!, startDelayHistories!, recordDelayStepEnd!,
  recordDelayJumps!, delayJumpsDue!
import ...Passes
import ...SimCode
import ...SimCode.Domains: DomainSystem, InstanceTable, EqClass, EvFn, Gen, IR, Lit, LoopBlocks, NotSupported, Op, Pre,
  Ref, Rel, SAssert, SAssign, SBreak, SCall, SFor, SIf, SPreStart, SWhen, SWhile, Slot, Stmt, TimeIR, TornBlock, VarInfo,
  WhenClass,
  AlgClass, DelaySite,
  _affineIn, _unknownVariables, _blockFamilies, _callOfParameters, _condElements, _initAlgAssigned, _parameterAlgorithms, _paramValue!,
  _parametersOnly, _relationSlots, _relationInstances,
  _isTimeRelation, _timeRelationSlots, _continuous,
  domainSystem, ns, Schedule, causalSchedule!, ParameterBinding, parameterBindings, _bindingDeps, InitialSystem

const SII = ModelingToolkit.SymbolicIndexingInterface
import ..CodeGeneration: NonlinearSolve, ADTypes, DiffEqBase

include("arrayBlockSolve.jl")
include("arrayInitialBlocks.jl")

"""
The array model's ODE `prob` (mass matrix I: its algebraic loops are solved in the right-hand side) as the problem
`solver` takes: for a DAE solver (DFBDF, Sundials' IDA) the residual form `du - f(u) = 0` (`ode_to_dae`, as the
scalarized path's), for an ODE solver the ODE. A DAE solver got the ODE: "Incompatible problem+solver pairing".
"""
_problemFor(prob::ModelingToolkit.SciMLBase.ODEProblem,
            ::ModelingToolkit.SciMLBase.AbstractDAEAlgorithm)::ModelingToolkit.SciMLBase.DAEProblem = ode_to_dae(prob)
_problemFor(prob::ModelingToolkit.SciMLBase.ODEProblem, ::Any)::ModelingToolkit.SciMLBase.ODEProblem = prob

"""
The initialization a solve of `prob` (`_problemFor`) takes unless the caller gives one (`kwargs`): for the residual
form Brown's, the derivatives solved again from the residual at the values the model's initialization or an event left
(as the scalarized path's `defaultInitializeKwargs`); the DAE solvers' default only checks them, and failed after a
reinit (residual 0.71). The ODE needs none.
"""
function _initializeKwargs(::ModelingToolkit.SciMLBase.DAEProblem, kwargs::Base.Pairs,
                           abstol::Union{Real, AbstractVector{<:Real}})::NamedTuple
  haskey(kwargs, :initializealg) && return (;)
  return (; initializealg = DiffEqBase.BrownFullBasicInit(Float64(abstol isa Real ? abstol : minimum(abstol))))
end
_initializeKwargs(::ModelingToolkit.SciMLBase.ODEProblem, ::Base.Pairs, ::Union{Real, AbstractVector{<:Real}})::NamedTuple =
  (;)

"The points of a step at which the relations' crossing functions are checked (their callback's `interp_points`), its ends included."
const RELATION_CHECKS = 10

"OpenModelica's output intervals of a simulation (numberOfIntervals): it checks the crossing functions at each."
const OUTPUT_INTERVALS = 500

"""
The largest step of a solve over `tspan` of a model with `nRelations` relations, unless the caller gives one (`kwargs`):
the crossing functions checked (RELATION_CHECKS per step) at least as often as OpenModelica checks them, at each of its
OUTPUT_INTERVALS output points (at least 56 steps a solve). The states of a clutch held stuck by a sine torque stay
constant, and the steps grew to 9 s: the stuck torque's excesses over the breakaway limit (0.31 s each,
FrictionEvents.Narrow) fell between the checks.
"""
function _stepKwargs(kwargs::Base.Pairs, tspan::Tuple{Real, Real}, nRelations::Int)::NamedTuple
  local span::Float64 = abs(Float64(tspan[2]) - Float64(tspan[1]))
  (nRelations == 0 || span == 0.0 || haskey(kwargs, :dtmax)) && return (;)
  return (; dtmax = (RELATION_CHECKS - 1) * span / OUTPUT_INTERVALS)
end

#= The parameter argument of an array model's problem: the parameter values (one flat
   vector, laid out by the model's PARAMETER_LAYOUT; one type for every array model, so the
   solver is compiled once), the discrete state (relation values, discrete variables and their
   pre values, when-condition values, state values before the event) and the generated module. =#
"""
    RootLogEntry

The algebraic loops' roots at a time of a run (`ArrayRuntimeState.rootLog`): the time, each loop family's guesses
(`ArrayModelParameters.loops`) and each large block's (`BlockSolver.y`) as the last solve left them.
"""
const RootLogEntry = Tuple{Float64, Vector{Matrix{Float64}}, Vector{Vector{Float64}}}

"""
    ArrayStartState(discrete, wcond, wnew, ndlog, upre, ureinit, loops, blocks, nRootLog, nObjectLog)

A problem's state after its initialization (`ArrayRuntimeState.atStart`): the discrete state (`_discreteState`), the
when-conditions, the length of the event log, the pre() and reinit buffers of the states, the loops' roots (the
families' and the large blocks' guesses) and the length of their log, the length of the external objects' call log.
"""
struct ArrayStartState
  discrete::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}}
  wcond::Vector{Bool}
  wnew::Vector{Bool}
  ndlog::Int
  upre::Vector{Float64}
  ureinit::Vector{Float64}
  loops::Vector{Matrix{Float64}}
  blocks::Vector{Vector{Float64}}
  nRootLog::Int
  nObjectLog::Int
end

"""
    ArrayRuntimeState(nSamples)

What one problem of an array model holds besides its values (`ArrayModelParameters.state`), so that problems of one model
(an ensemble's) run apart: the sample() starts and intervals (`sampleSchedule!`), their last stop and the next one
(`_nextSampleStop`), the when bodies' calls of parameters in a pass of the event iteration (`whenMemo`), the
warnings reported,
the asserts' probe (`_violated!`), an assert crossing to decide (`pending`: its time and state), the message of a
terminate() reached, the discrete state at the start of the current step (`stepStart`, `_discreteState`: within a step
that ends at an event the asserts read it), the String parameters' values (`Gen.strings`, with their overrides), the
state just before the last event (`leftLimit`: its time and state, a reinit moved it after), the problem's state after
its initialization (`atStart`, `ArrayStartState`: a solve of the problem starts from it again), the loops' roots at
the start and at each accepted step (`rootLog`, `_logRoots!`: a loop with several roots, a quaternion's
`Q * Q = 1`, is read after the run on the root the run was on at that time, `_rootsAt`), the calls of the when-bodies
that read an external object (`objectLog`: time, call, instance, as the run made them) and how many of them the objects
are at (`objectAt`): an MSL table keeps the event interval of its last `getNextTimeEvent` call, and a read of the
solution at a time makes the calls again up to that time (`replayObjects!`).
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
  rootLog::Vector{RootLogEntry}
  forced::Bool   #= the buffered relations hold a combination being tried (`_searchRelations!`) =#
  delays::DelayState   #= the histories of the delay() calls' arguments, one per instance (`delayedValue`, delays.jl) =#
  terminal::Bool   #= the terminal event, after the solve: terminal() is true (`_terminalEvent!`) =#
  objectLog::Vector{Tuple{Float64, Int, Int}}
  objectAt::Int
end
ArrayRuntimeState(nSamples::Int, strings::Vector{String}, nDelays::Int = 0) =
  ArrayRuntimeState(zeros(nSamples), zeros(nSamples), Float64[], Dict{Tuple{Int, Int, Int}, Any}(), Set{Any}(), false, false,
                    (NaN, Float64[]), nothing, (Float64[], Float64[], Bool[], Float64[]), strings, (NaN, Float64[]),
                    nothing, RootLogEntry[], false, DelayState(nDelays), false, Tuple{Float64, Int, Int}[], 0)

"""
    InitialPass(on, x, r)

The algebraic loops in the initial solve (`InitialSystem`): while `on`, a loop the initial system holds (a family's
member, a large block) takes its unknowns from `x` and writes its residuals to `r` (at its offset), not solved in the
equations.
"""
mutable struct InitialPass
  on::Bool
  x::Vector{Float64}
  r::Vector{Float64}
end

"""
    _passBlock!(s, residual!, pass, base, du, a, u, p, t, fire)

A large algebraic loop in the initial solve (`InitialPass`): its iteration variables from the solve's unknowns at
`base`, its residuals to the solve's rows there; the relations hold their values.
"""
function _passBlock!(s::BlockSolver, residual!::R, pass::InitialPass, base::Int, du::AbstractVector, a::AbstractVector,
                     u::AbstractVector, p, t::Real, fire::Bool)::Nothing where {R <: Function}
  for j::Int in eachindex(s.x)
    s.x[j] = pass.x[base + j]
  end
  residual!(s.r, s.x, du, a, u, p, t, nothing, false, fire)
  for j::Int in eachindex(s.r)
    pass.r[base + j] = s.r[j]
  end
  return nothing
end

"""
    LoopWork{T}(k)

The work arrays of a loop family's solve (`LOOPFAMILY_f!`) of `k` unknowns: the iterate, the residuals there and at a
perturbed point, the step, an attempt's start, the Jacobian, its LU factors (`_luFactor!`) and their pivots. One per
family in the parameters for solves on Float64 (`ArrayModelParameters.loopWork`), so that a right-hand side allocates
none; a solve on another number type (a solver's dual numbers) makes its own.
"""
struct LoopWork{T}
  x::Vector{T}
  r::Vector{T}
  r2::Vector{T}
  dx::Vector{T}
  xs::Vector{T}
  J::Matrix{T}
  F::Matrix{T}
  pivots::Vector{Int}
end

LoopWork{T}(k::Int) where {T} = LoopWork{T}(zeros(T, k), zeros(T, k), zeros(T, k), zeros(T, k), zeros(T, k), zeros(T, k, k),
                                             zeros(T, k, k), zeros(Int, k))

"""
    _luFactor!(F, pivots) -> Bool

`F` replaced by its LU factors with partial pivoting (LAPACK getrf's layout: L below the diagonal with a unit
diagonal, U on and above it; row `j` exchanged with `pivots[j]`), for the small loops of a loop family on any number
type; false for an exactly zero pivot (a singular matrix, as `issuccess` of `lu`).
"""
function _luFactor!(F::Matrix{T}, pivots::Vector{Int})::Bool where {T}
  local n::Int = size(F, 1)
  for j::Int in 1:n
    local q::Int = j
    local best::T = abs(F[j, j])
    for i::Int in (j + 1):n
      local v::T = abs(F[i, j])
      if v > best
        best = v
        q = i
      end
    end
    pivots[j] = q
    iszero(best) && return false
    if q != j
      for c::Int in 1:n
        local swap::T = F[j, c]
        F[j, c] = F[q, c]
        F[q, c] = swap
      end
    end
    local pivot::T = F[j, j]
    for i::Int in (j + 1):n
      local l::T = F[i, j] / pivot
      F[i, j] = l
      for c::Int in (j + 1):n
        F[i, c] -= l * F[j, c]
      end
    end
  end
  return true
end

"""
    _luSolve!(x, F, pivots, b) -> x

`x` the solution of A x = b for the LU factors `F` of A with their `pivots` (`_luFactor!`).
"""
function _luSolve!(x::Vector{T}, F::Matrix{T}, pivots::Vector{Int}, b::Vector{T})::Vector{T} where {T}
  local n::Int = length(x)
  copyto!(x, b)
  for j::Int in 1:n
    local q::Int = pivots[j]
    if q != j
      local swap::T = x[j]
      x[j] = x[q]
      x[q] = swap
    end
  end
  for i::Int in 2:n, c::Int in 1:(i - 1)
    x[i] -= F[i, c] * x[c]
  end
  for i::Int in n:-1:1
    for c::Int in (i + 1):n
      x[i] -= F[i, c] * x[c]
    end
    x[i] /= F[i, i]
  end
  return x
end

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
  homotopy::Vector{Float64}    #= [λ] of homotopy(): 1 (actual) but in the continuation (homotopyStart!) =#
  abuf::Vector{Float64}        #= the algebraic variables of a right-hand side evaluated on Float64 (RHS!): no buffer per call =#
  aleft::Vector{Float64}       #= the algebraic variables before the event's sweep: pre() of one (its left limit) =#
  tgradAfter::Vector{Float64}  #= tgrad!'s right-hand side after t (Float64 states: no buffer per call) =#
  tgradBefore::Vector{Float64} #= tgrad!'s right-hand side before t =#
  blocks::Vector{BlockSolver}  #= the large algebraic loops' solvers (blockSolve!), in the order of the schedule =#
  #= per loop family its members' guesses (unknown, member): the run's last solutions =#
  loops::Vector{Matrix{Float64}}
  loopWork::Vector{LoopWork{Float64}}   #= per loop family its solve's work arrays (Float64 solves) =#
  pass::InitialPass           #= the loops the initial solve holds (`initialize`) =#
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

"""
    _holdDiscretes!(s, p)

What the residuals of the loop of `s` read besides its unknowns, the states, time and the parameters, kept in
`s.held` at a Jacobian (`_blockJacobian!`): the discrete state of `p` (`_discreteState`), the sample() instances
active, whether it initializes and homotopy()'s λ. They change only at events and in the initialization, and with
them the matrix of a linear loop (its relations, an `if` on a discrete value): its kept factors are for it while they
are the same (`_sameDiscretes`).
"""
function _holdDiscretes!(s::BlockSolver, p::ArrayModelParameters)::Nothing
  local h::Vector{Float64} = s.held
  empty!(h)
  append!(h, p.d); append!(h, p.dpre); append!(h, p.rel); append!(h, p.evv)
  append!(h, p.sampleActive); append!(h, p.initPhase); append!(h, p.homotopy)
  return nothing
end

"Whether the values `_holdDiscretes!` kept in `s` are those of `p` now (`isequal`): its kept factors are for them."
function _sameDiscretes(s::BlockSolver, p::ArrayModelParameters)::Bool
  local h::Vector{Float64} = s.held
  length(h) == length(p.d) + length(p.dpre) + length(p.rel) + length(p.evv) + length(p.sampleActive) +
               length(p.initPhase) + length(p.homotopy) || return false
  local k::Int = 0
  (k = _heldMatch(h, k, p.d)) < 0 && return false
  (k = _heldMatch(h, k, p.dpre)) < 0 && return false
  (k = _heldMatch(h, k, p.rel)) < 0 && return false
  (k = _heldMatch(h, k, p.evv)) < 0 && return false
  (k = _heldMatch(h, k, p.sampleActive)) < 0 && return false
  (k = _heldMatch(h, k, p.initPhase)) < 0 && return false
  return _heldMatch(h, k, p.homotopy) >= 0
end

"The values `v` at `h[k + 1]` on (`_sameDiscretes`): the index of the last, or -1 where one differs."
function _heldMatch(h::Vector{Float64}, k::Int, v::Vector{T})::Int where {T <: Union{Float64, Bool}}
  for x::T in v
    k += 1
    isequal(h[k], Float64(x)) || return -1
  end
  return k
end

"The state of `p` after its initialization (`ArrayStartState`)."
_startState(p::ArrayModelParameters)::ArrayStartState =
  ArrayStartState(_discreteState(p), copy(p.wcond), copy(p.wnew), length(p.dlog), copy(p.upre), copy(p.ureinit),
                  map(copy, p.loops), Vector{Float64}[copy(b.y) for b in p.blocks], length(p.state.rootLog),
                  length(p.state.objectLog))

"""
    _checkBlocksSolved(p)

An error for a large algebraic loop that the initialization tried and never solved, as the scalar path's "the
initial system". Its unknowns are NaN (`blockSolve!`); where no derivative reads them the solver neither rejects a
step nor stops, and the run ended in Success with them NaN throughout (MSL QuasiStatic SinglePhase Rectifier).
"""
function _checkBlocksSolved(p::ArrayModelParameters)::Nothing
  for (k::Int, b::BlockSolver) in enumerate(p.blocks)
    (b.failures > 0 && !b.solved) &&
      error("array model: the initial system: the algebraic loop $(k) of $(length(b.y)) unknowns has no solution " *
            "from its start values")
  end
  return nothing
end

"""
    _restartStep!(integrator)

The next step after an event that changed the equations starts small, as on the MTK path (relationRefresh.jl
`_restartStepSize!`): the derivatives can jump by orders of magnitude, and a step sized before the event crosses the
fast transient at once, the values interpolated inside it far off (MSL SlewRateLimiter: one step of 0.4 s over a
transient of 1e-4 s, `a` 12 for 0). The step is OrdinaryDiffEq's estimate of a first step (`_autoDtReset!`), at most
1e-6 of the time span and the step proposed before. The estimate alone is not enough: the algebraic variables are
solved inside the right-hand side, so a state that follows its input through a high gain has a derivative of
round-off size at the event, and the estimate from it steps far over the transient (MSL OpAmps.Differentiator,
V0 = 15000: 5e-3 s over a transient of 2e-6 s, the capacitor's voltage 0.02 V behind over the ramp). No step
follows an event at the last stop. The step is proposed (`dtpropose`): `integrator.dt` stays the length of the step
that ended at the event, which the assert checks after the event interpolate (`_assertPending!`, `_autoDtReset!`).
"""
function _restartStep!(integrator)::Nothing
  isempty(integrator.opts.tstops) && return nothing
  local before::Float64 = abs(integrator.dtpropose)
  _autoDtReset!(integrator)
  local span::Float64 = abs(integrator.sol.prob.tspan[2] - integrator.sol.prob.tspan[1])
  local cap::Float64 = 1.0e-6 * (isfinite(span) ? span : 1.0)
  integrator.dtpropose = integrator.tdir * min(abs(integrator.dtpropose), cap, before)
  return nothing
end

"""
    _logRoots!(p, t)

The loops' roots of the run at time `t` logged (`ArrayRuntimeState.rootLog`): after the initialization and at each
accepted step. Nothing for a model without algebraic loops.
"""
function _logRoots!(p::ArrayModelParameters, t::Real)::Nothing
  (isempty(p.loops) && isempty(p.blocks)) && return nothing
  push!(p.state.rootLog, (Float64(t), map(copy, p.loops), Vector{Float64}[copy(b.y) for b in p.blocks]))
  return nothing
end

"""
    _rootsAt(p, t, right) -> Vector{Matrix{Float64}}

The loop families' guesses for a read of the solution at time `t`: the roots logged at the last step before `t`
(`_logRoots!`; the left limit, as the states' interpolation: an entry at `t` itself may be the one after an event
there), with `right` those at the last step at or before `t` (the second of a saved pair); the large blocks'
guesses set to that entry's too; the current ones where nothing is logged. A read after the run started every loop
from the run's last roots: at t = 0, after a run that ended on the other side of a quaternion's `Q * Q = 1`, on the
other root.
"""
function _rootsAt(p::ArrayModelParameters, t::Real, right::Bool)::Vector{Matrix{Float64}}
  local log::Vector{RootLogEntry} = p.state.rootLog
  isempty(log) && return map(copy, p.loops)
  local at::Int = right ? searchsortedlast(log, (Float64(t),); by = first) :
                          searchsortedfirst(log, (Float64(t),); by = first) - 1
  local entry::RootLogEntry = log[clamp(at, 1, length(log))]
  for (b, y) in zip(p.blocks, entry[3])
    b.y .= y
  end
  return map(copy, entry[2])
end

"""
    _restoreStart!(p)

`p` as after its initialization again (`ArrayRuntimeState.atStart`), at the start of a solve: a second solve of the
problem started from the first one's end (its discrete values, the event log, its loops' roots).
"""
function _restoreStart!(p::ArrayModelParameters)::Nothing
  #= the delay() histories are a solve's: its initialization reads the arguments themselves =#
  emptyDelayHistories!(p.state.delays)
  local s::Union{Nothing, ArrayStartState} = p.state.atStart
  s === nothing && return nothing
  p.d .= s.discrete[1]; p.dpre .= s.discrete[2]; p.rel .= s.discrete[3]; p.evv .= s.discrete[4]
  p.wcond .= s.wcond; p.wnew .= s.wnew; p.upre .= s.upre; p.ureinit .= s.ureinit
  resize!(p.dlog, s.ndlog)
  #= the external objects as the initialization left them: its calls made again (before the loops' roots are
     restored: the calls' arguments are evaluated) =#
  resize!(p.state.objectLog, s.nObjectLog)
  p.state.objectAt = typemax(Int)
  Base.invokelatest(p.model.replayObjects!, s.upre, p, Inf)
  for (G, start) in zip(p.loops, s.loops)
    G .= start
  end
  for (b, y) in zip(p.blocks, s.blocks)
    b.y .= y
  end
  resize!(p.state.rootLog, s.nRootLog)
  fill!(p.sampleActive, false)
  p.state.terminal = false
  p.state.pending = (NaN, Float64[]); p.state.leftLimit = (NaN, Float64[]); p.state.terminated = nothing
  empty!(p.state.whenMemo)
  return nothing
end

"""
The combination of duplicate entries of a Jacobian's sparsity pattern (`sparse`): the first. One function for every model
(a closure in the generated code compiled `sparse!` again for each model).
"""
_keepFirst(x::Float64, ::Float64)::Float64 = x

"""
    _withDiscreteState(f, p, s)

`f()` with the discrete state of `p` set to `s` (`_discreteState`; the step start's, where an event ended the step), restored
after; with the state as it is where `s` holds none (before the first step).
"""
function _withDiscreteState(@nospecialize(f::Function), p::ArrayModelParameters, s::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}})
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
The relative step of a state in the Jacobian's central differences, its least 1 times it: the square root of the
machine precision (1.5e-8; OpenModelica's numerical Jacobians step by about as much). The cube root, FiniteDiff's
for central differences (6e-6), is 0.6 Pa on a pressure of 1e5 Pa: a duct's flow into a room follows pressure
differences of 1e-3 Pa with a time constant of 2e-6 s, the Jacobian over 0.6 Pa missed that stiffness, and MSL
Fluid's RoomCO2WithControls took 40000 steps for 2 s after its fan started (38 for 100 s with 1e-8). The central
form keeps the truncation error of the smaller step's forward difference away.
"""
const JACOBIAN_STEP = sqrt(eps(Float64))

"""
The absolute tolerance of an array model's solve when the caller gives none. The array code controls the error of its
states only; a state far below the solvers' default of 1e-6 was free to be off by its own size. MSL Fluid's
RoomCO2WithControls: the ducts' trace substance 1 to 5% off omc at 1e-6 (Rodas5P, 400 steps for the day), equal
with 1e-10 (724 steps); IMC_DOL's currents before its switch closes. omc's DASSL at 1e-6 keeps them.
"""
const DEFAULT_ABSTOL = 1.0e-10

"""
    ColoredJacobian(rhs, pattern)

The Jacobian of an array model's right-hand side `rhs` by central differences in the sparsity `pattern`: the columns
that share no row perturbed together (`columnGroups`), each state by its own step, `JACOBIAN_STEP * max(|u[j]|, 1)`.
Left to the solver, a sparse prototype takes one step for every state, from
the norm of the whole state vector (DifferentiationInterface's pushforward): beside a pressure of 1e5 Pa a mass
fraction of 0.01 was moved by 0.61, and the medium's `log` of a negative fraction rejected every first step (MSL
Media ReferenceAir MoistAir1; Rodas5P, whose stages need the Jacobian). Where the right-hand side fails on one side
of a state (a value at the edge of the equations' domain) the difference is the one to the point itself.
"""
struct ColoredJacobian{F} <: Function
  rhs::F
  groups::Vector{Vector{Int}}
  steps::Vector{Float64}
  point::Vector{Float64}
  above::Vector{Float64}
  below::Vector{Float64}
  base::Vector{Float64}
end

function ColoredJacobian(rhs::F, pattern::_SPARSE.SparseMatrixCSC{Float64, Int}) where {F}
  local n::Int = size(pattern, 2)
  return ColoredJacobian{F}(rhs, columnGroups(pattern), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))
end

function (c::ColoredJacobian)(J::_SPARSE.SparseMatrixCSC{Float64, Int}, u::AbstractVector{Float64}, p, t)::Nothing
  local rows::Vector{Int} = _SPARSE.rowvals(J)
  local values::Vector{Float64} = _SPARSE.nonzeros(J)
  local (steps::Vector{Float64}, point::Vector{Float64}) = (c.steps, c.point)
  local (above::Vector{Float64}, below::Vector{Float64}) = (c.above, c.below)
  local based::Bool = false
  copyto!(point, u)
  for group::Vector{Int} in c.groups
    for j::Int in group
      steps[j] = JACOBIAN_STEP * max(abs(u[j]), 1.0)
      point[j] = u[j] + steps[j]
    end
    c.rhs(above, point, p, t)
    for j::Int in group
      point[j] = u[j] - steps[j]
    end
    c.rhs(below, point, p, t)
    for j::Int in group
      point[j] = u[j]
    end
    local (up::Bool, down::Bool) = (all(isfinite, above), all(isfinite, below))
    if up != down
      based || c.rhs(c.base, point, p, t)
      based = true
      copyto!(up ? below : above, c.base)
    end
    local span::Float64 = up == down ? 2.0 : 1.0
    for j::Int in group
      for k::Int in _SPARSE.nzrange(J, j)
        values[k] = (above[rows[k]] - below[rows[k]]) / (span * steps[j])
      end
    end
  end
  return nothing
end

"""
    columnGroups(pattern) -> Vector{Vector{Int}}

The columns of the sparsity `pattern` in groups whose columns share no row (greedy, in the columns' order: a column
joins the first group none of whose rows it has): one perturbation of a group's states gives each of their columns.
"""
function columnGroups(pattern::_SPARSE.SparseMatrixCSC{Float64, Int})::Vector{Vector{Int}}
  local rows::Vector{Int} = _SPARSE.rowvals(pattern)
  local groups::Vector{Vector{Int}} = Vector{Int}[]
  local taken::Vector{BitVector} = BitVector[]
  for j::Int in 1:size(pattern, 2)
    local at::Int = 0
    for k::Int in eachindex(taken)
      local free::Bool = true
      for i::Int in _SPARSE.nzrange(pattern, j)
        taken[k][rows[i]] && (free = false; break)
      end
      free && (at = k; break)
    end
    if at == 0
      push!(groups, Int[])
      push!(taken, falses(size(pattern, 1)))
      at = length(groups)
    end
    push!(groups[at], j)
    for i::Int in _SPARSE.nzrange(pattern, j)
      taken[at][rows[i]] = true
    end
  end
  return groups
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
    _parameterIndex(model) -> Dict{String, Int}

The positions in `p.values` of the generated model's scalar parameters and parameter elements, by their names
(`PARAMETER_NAMES`, `PARAMETER_ELEMENTS`: Modelica's and canonical), in `PARAMETER_LAYOUT`'s column-major order: the
parameters a solution is indexed by (a reference signal of MSL Utilities.Examples.readRealParameterModel: `J`).
"""
function _parameterIndex(model::Module)::Dict{String, Int}
  local layout::Dict{Symbol, Tuple{Int, Int, Vector{Int}}} =
    Dict{Symbol, Tuple{Int, Int, Vector{Int}}}(sym => (off, len, dims)
                                               for (sym, off, len, dims) in model.PARAMETER_LAYOUT)
  local index::Dict{String, Int} = Dict{String, Int}()
  for (name::String, sym::Symbol) in model.PARAMETER_NAMES
    local (off::Int, len::Int, _) = get(layout, sym, (0, 0, Int[]))
    len == 1 && (index[name] = off + 1)
  end
  for (name::String, (sym::Symbol, t::Tuple)) in model.PARAMETER_ELEMENTS
    haskey(layout, sym) || continue
    local (off::Int, _, dims::Vector{Int}) = layout[sym]
    index[name] = off + LinearIndices(Tuple(dims))[t...]
  end
  return index
end

"""
    variableValues(sol, name)

The values of a state, algebraic or discrete variable element (`"c[2].T"`), or a parameter, at the time
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
  local params::Dict{String, Int} = _parameterIndex(m)
  local q::Int = get(params, name, get(params, key, 0))
  q > 0 && return fill(p.values[q], length(sol.t))
  return nothing
end

#= The symbolic system of an array model's ODEFunction: names for solution indexing
   (sol[:x], sol(t; idxs = Symbol("c[2].T")), sol.ps[:k]), states as variables, algebraic and discrete
   variables as observed, parameters as parameters (their values in p.values). Names as Modelica writes them or
   canonical (c[2]_T). =#
struct ArraySystem
  states::Dict{Symbol, Int}
  stateSyms::Vector{Symbol}
  observed::Dict{Symbol, Tuple{Symbol, Int}}   #= (:alg, :discrete or :param, index; :param in p.values) =#
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
  for (k::String, v::Int) in _parameterIndex(model)
    local s::Symbol = Symbol(k)
    (haskey(states, s) || haskey(observed, s)) || (observed[s] = (:param, v))
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
#= the parameters by their position in p.values (`sol.ps[:k]`, as of the ModelingToolkit path's solution) =#
SII.is_parameter(sys::ArraySystem, x) = _parameterPosition(sys, x) !== nothing
SII.parameter_index(sys::ArraySystem, x) = _parameterPosition(sys, x)
SII.parameter_symbols(sys::ArraySystem) = Symbol[s for (s, (kind, _)) in _observedIdx(sys) if kind === :param]
SII.parameter_values(p::ArrayModelParameters, i::Int) = p.values[i]
SII.is_independent_variable(::ArraySystem, x) = x === :t || x === :time
SII.independent_variable_symbols(::ArraySystem) = [:t]
SII.is_observed(sys::ArraySystem, x) =
  (s = _sym(x); s !== nothing && haskey(_observedIdx(sys), s) && first(_observedIdx(sys)[s]) !== :param)

"The position in p.values of the parameter `x` of an array model; `nothing` for another name."
function _parameterPosition(sys::ArraySystem, @nospecialize(x))::Union{Nothing, Int}
  local s::Union{Nothing, Symbol} = _sym(x)
  s === nothing && return nothing
  local entry::Union{Nothing, Tuple{Symbol, Int}} = get(_observedIdx(sys), s, nothing)
  return (entry === nothing || first(entry) !== :param) ? nothing : last(entry)
end
function SII.observed(sys::ArraySystem, x)
  local (kind, i) = _observedIdx(sys)[_sym(x)]
  local model = getfield(sys, :model)
  if kind == :alg
    #= the binding read in the latest world too: the module is newer than the caller =#
    return (u, p, t) -> (Base.@invokelatest model.algebraics(u, p, t))[i]
  end
  kind == :param && return (u, p, t) -> p.values[i]
  return (u, p, t) -> begin
    local times = first.(p.dlog)
    isempty(times) ? NaN : p.dlog[_dlogIndex(times, t, p.tend[1])][2][i]
  end
end
SII.observed(sys::ArraySystem, x::ArrayVar) = invoke(SII.observed, Tuple{ArraySystem, Any}, sys, x)
SII.all_variable_symbols(sys::ArraySystem) =
  vcat(getfield(sys, :stateSyms), Symbol[s for (s, (kind, _)) in _observedIdx(sys) if kind !== :param])
SII.all_symbols(sys::ArraySystem) = vcat(SII.all_variable_symbols(sys), SII.parameter_symbols(sys), :t)
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
@inline _rel(z, p, mode::Bool, k::Int, x, y, op) =
  _relationAt(z, p, mode, k, x, y, op, p.hyst[1] == 0, p.hyst[1] * (1 + max(_finiteAbs(x), _finiteAbs(y))))

"""
A relation of `time` and an expression that changes only at events (`time > 0.6`, `time < T0 + Trising` with T0
discrete: `_isTimeRelation`): a time event at its exact time, as OpenModelica takes it, its hysteresis only the
floating-point resolution of its operands (none at the initialization): of their size, not of a second, which is
0.4 % of a pulse of 0.5 ps.
"""
@inline _timeRel(z, p, mode::Bool, k::Int, x, y, op) =
  _relationAt(z, p, mode, k, x, y, op, p.hyst[1] == 0,
              8 * eps(Float64) * max(_finiteAbs(x), _finiteAbs(y), floatmin(Float64)))

"""
A relation of continuous values in an initial algorithm (a lifted cluster's initial arm, `Op(:startRelation)`):
literal (MLS 8.5); on its boundary to rounding the exact tie's value, a non-strict relation true, a strict one false,
whichever side the initialization's solve left it on (the MTK path's `_tieLiteral`, the same bound). MSL RobotR3's
bearing friction: `initial() and w_relfric < 0` at w = -5e-32 started it Backward; omc: Stuck.
"""
@inline function _startRelation(x, y, op)::Bool
  local zc::Float64 = float(x) - float(y)
  abs(zc) <= 1.0e-12 * max(_finiteAbs(x), _finiteAbs(y), 1.0) && return op === (<=) || op === (>=)
  return op(x, y)
end

"""
    CrossingOut(z)

Where `equations!`, `whens!` and `crossings!` of an array model write their crossing functions: `z`, the event
callback's output (a view, or a vector), or nowhere (`nothing`, `NO_CROSSINGS`). One type for all three: a model's
functions compile once; with `nothing` or a view as their argument they compiled twice (MSL PID_Controller's
`equations!` 5,400 and 5,700 statements, 2026-11-18).
"""
struct CrossingOut
  z::Union{Nothing, SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}, Vector{Float64}}
end

"No crossing functions written (`CrossingOut`)."
const NO_CROSSINGS = CrossingOut(nothing)

"A relation of the hysteresis `band`, evaluated literally where `literal` (`_rel`, `_timeRel`)."
@inline _relationAt(z::CrossingOut, p, mode::Bool, k::Int, x, y, op, literal::Bool, band::Float64) =
  _relationAt(z.z, p, mode, k, x, y, op, literal, band)
@inline function _relationAt(z, p, mode::Bool, k::Int, x, y, op, literal::Bool, band::Float64)
  #= as Float64: the difference of two native Integers (an opaque function's result) wraps around beyond 2^63 =#
  local zc = (op === (>) || op === (>=)) ? float(y) - float(x) : float(x) - float(y)
  local old = p.rel[k]
  local eps = literal ? 0.0 : band
  z === nothing || (z[k] = old ? zc - eps : zc + eps)
  mode && (p.rel[k] = literal ? op(x, y) : old ? zc <= eps : (zc <= -eps || (zc == 0 && op(x, y))))
  return p.rel[k]
end

"""
An event-generating function (integer, floor, ceil, div, mod, rem; MLS 3.7.2): its value
v = floor/ceil/trunc(q) changes only at events (in mode), and there it is literal, as
OpenModelica's `_event_floor` at a discrete call. Two crossing functions bound the interval
that keeps v, widened by the hysteresis eps: z[k] crosses where q passes its upper end,
z[k + 1] its lower end. The value kept within eps past the interval's end at an event
(another relation's) was stale where that relation read it: `DNTime > 24` of `DNTime =
mod(time, 24)`, `if DNTime > 24 then DNTime = 0` switched back and forth at t = 24 (its
crossing comes 2e-6 before the mod's: PersonalityAspects).
"""
_evInterval(kind::Symbol, v) = kind === :floor ? (v, v + 1) : kind === :ceil ? (v - 1, v) :
                               v > 0 ? (v, v + 1) : v < 0 ? (v - 1, v) : (-1.0, 1.0)
@inline _evfn(z::CrossingOut, p, mode::Bool, k::Int, kind::Symbol, q) = _evfn(z.z, p, mode, k, kind, q)
@inline function _evfn(z, p, mode::Bool, k::Int, kind::Symbol, q)
  local v = p.evv[k]
  local eps = p.hyst[1] * (1 + _finiteAbs(q))
  local (lo, hi) = _evInterval(kind, v)
  if mode
    local literal::Float64 = Float64(kind === :floor ? floor(q) : kind === :ceil ? ceil(q) : trunc(q))
    if literal != v
      v = p.evv[k] = literal
      (lo, hi) = _evInterval(kind, v)
    end
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
    ModelicaSubscriptError(name, index, size)

A subscript computed from variable values that is outside its dimension. The generated functions run under
`@inbounds`: such subscripts are checked (`_checkedSubscript`, `_checkedIndex`, `_checkedAssign!`). An error of the
model, not of a step (it is not one of `_RHS_STEP_ERRORS`).
"""
struct ModelicaSubscriptError <: Exception
  name::String
  index::Int
  size::Int
end

Base.showerror(io::IO, e::ModelicaSubscriptError) =
  print(io, "ModelicaSubscriptError: subscript ", e.index, " of ", e.name, " is outside 1:", e.size)

@noinline _subscriptError(i::Int, n::Int, name::String) = throw(ModelicaSubscriptError(name, i, n))

@noinline function _indexError(a::AbstractArray, i::Tuple)
  local k::Int = something(findfirst(d::Int -> !(1 <= i[d] <= size(a, d)), 1:length(i)), 1)
  throw(ModelicaSubscriptError(string("an array of size ", size(a)), Int(i[k]), size(a, k)))
end

"The subscript `i` of a dimension of size `n` of the variable `name`, checked."
@inline function _checkedSubscript(i::Integer, n::Int, name::String)::Int
  (1 <= i <= n) || _subscriptError(Int(i), n, name)
  return Int(i)
end

"`a[i...]` with subscripts computed from variable values, checked."
@inline function _checkedIndex(a::AbstractArray, i::Vararg{Integer, N}) where {N}
  checkbounds(Bool, a, i...) || _indexError(a, i)
  return @inbounds a[i...]
end

"`t[i]` of a literal tuple (an enumeration's literal names, `String(e)`) by a subscript computed from variable values, checked."
@inline function _checkedIndex(t::Tuple, i::Integer)
  (1 <= i <= length(t)) || _subscriptError(Int(i), length(t), "the literals of an enumeration")
  return @inbounds t[i]
end

"`a[i...] = v` with subscripts computed from variable values, checked."
@inline function _checkedAssign!(a::AbstractArray, v, i::Vararg{Integer, N}) where {N}
  checkbounds(Bool, a, i...) || _indexError(a, i)
  @inbounds a[i...] = v
  return v
end

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

"The declarations of `paramLocals` (the parameter views, `local x = p.<field>`) whose names `code` reads."
function _usedLocals(paramLocals::Vector{Any}, code::Union{Expr, Symbol})::Vector{Any}
  local used::Set{Symbol} = _symbolsIn!(Set{Symbol}(), code)
  return Any[l for l in paramLocals if _localName(l) in used]
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
    local views::Vector{Any} = _usedLocals(paramLocals, Expr(:block, part...))
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

#= ---------------------------------------------------------------- delay() =#

"""
    _delayRecords(g) -> Tuple{Vector{Any}, Int}

The code that takes the present values of the delay() calls' arguments and delay times (`Gen.delays`), each call
over its class's domain, into the run's delay state (`_now`, `_times` at the call's base plus the column), and the
number of histories. Run after the equations (`delayNow!`): the arguments are read where the equations left them.
"""
function _delayRecords(g::Gen)::Tuple{Vector{Any}, Int}
  local body::Vector{Any} = Any[]
  local total::Int = 0
  for site::DelaySite in g.delays
    local n::Int = size(site.domain, 2)
    local record::Expr = quote
      _now[$(site.base) + _col] = Float64($(jl(g, site.value)))
      _times[$(site.base) + _col] = Float64($(jl(g, site.time)))
    end
    push!(body, site.nslots == 0 ? Expr(:let, :(_col = 1), record) :
                                   _loopCode(site.nslots, site.domain, collect(1:n), record))
    total = site.base + n
  end
  return (body, total)
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
    _unknownLabel(g, id) -> String

The unknown `id` (a state's derivative up to `g.nStates`, an algebraic element after) by its variable and its element's
linear index (`der(x)[2]`, `y[5]`), for traces.
"""
function _unknownLabel(g::Gen, id::Int)::String
  local state::Bool = id <= g.nStates
  local k::Int = state ? id : id - g.nStates
  for (name::String, info::VarInfo) in g.vars
    (info.kind == (state ? :state : :alg) && info.offset < k <= info.offset + prod(info.dims; init = 1)) || continue
    local element::String = isempty(info.dims) ? "" : string("[", k - info.offset, "]")
    return state ? string("der(", name, ")", element) : string(name, element)
  end
  return string(state ? "state#" : "alg#", k)
end

"""
    _loopFamilyCode(g, f, template, blocks, el, matchRow, a0, relBases, bases) -> Tuple{Vector{Any}, Expr, Expr}

The solve of the loop family `f` (`LOOPFAMILY_f!`): its tables (each member's instance columns and unknowns, the first
guess per unknown: the start values; a run keeps its last solutions as the next guesses in its parameters, `p.loops`)
and the body that solves the members it is called for,
each by Newton's method on the residuals of its instances (lhs - rhs, by the template's classes at the member's
columns), the unknowns set in place (a state derivative in du, an algebraic variable in a), the Jacobian by forward
differences; the last solution kept as the next guess (in Float64 solves); residuals or a Jacobian not finite, or a
singular Jacobian, make the unknowns NaN (the solver rejects the step); a solve that does not converge is made again
with damped steps, from the last solution and then from the start values. Not a discrete unknown.
Relations in the loop hold their values in Newton's iterations (MLS 8.5: a relation changes at events only; `z`,
`mode` shadowed: no crossing written, no relation changed); the residuals once more at the solution write the crossings
and, at an event, the relations, whose change the event iteration sweeps again (`_settle!`; a combination of relations
`_searchRelations!` tries is solved with, not replaced by the relations at the point a solve starts from). MSL's
FluxTubes hysteresis, `\$DER.ifEq_tmp1 = if core.Hstat < -650.0 then ...`; a comparator in an op-amp's loop.
In the initial solve (`p.pass.on`) a member the initial system holds (`bases`: its unknowns' offset there, -1 for
none) takes its unknowns from it and gives it its residuals (`InitialPass`).
A family whose residuals are affine in its unknowns (`_affineIn`) is solved by one Jacobian, by differences over the
unknowns' size (exact for an affine residual), and one LU: one step from the last solution and one more at the
residuals there; Newton's method as above where that fails. The work arrays are the parameters' (`LoopWork`) in a
Float64 solve. The residuals are a function of their own (`_loopResidualFunction`, of the third value returned: its
body).
"""
function _loopFamilyCode(g::Gen, f::Int, template::Vector{Tuple{Int, Int}}, blocks::Vector{Vector{Int}}, el::InstanceTable,
                         matchRow::Vector{Int}, a0::Vector{Float64}, relBases::Vector{Int},
                         bases::Vector{Int})::Tuple{Vector{Any}, Expr, Expr}
  local k::Int = length(template); local M::Int = length(blocks)
  local nS::Int = g.nStates
  local cols::Matrix{Int} = Matrix{Int}(undef, k, M); local ids::Matrix{Int} = Matrix{Int}(undef, k, M)
  local guess::Matrix{Float64} = Matrix{Float64}(undef, k, M)
  for (m, b) in enumerate(blocks), j in 1:k
    local r::Int = b[j]
    local u::Int = matchRow[r]
    u <= nS + g.nAlg || ns("a discrete unknown in an algebraic loop: " * first(g.classes[template[j][1]].text, 120))
    cols[j, m] = el.col[r]; ids[j, m] = u
    guess[j, m] = u <= nS ? 0.0 : a0[u - nS]
  end
  local (C::Symbol, I::Symbol, G::Symbol) = (Symbol("LOOP_COLS_", f), Symbol("LOOP_IDS_", f), Symbol("LOOP_GUESS_", f))
  local B::Symbol = Symbol("LOOP_INITIAL_", f)
  local hasRelations::Bool = any(((ci, _),) -> g.classes[ci].nrel > 0, template)
  #= the residuals once, in their own function (`LOOPRES_<f>!`, `_loopResidualFunction`), called where the solve
     evaluates them: inlined at each of those places they were compiled five times =#
  local parts::Vector{Any} = Any[]
  for (j, (ci, _)) in enumerate(template)
    local c::EqClass = g.classes[ci]
    g.relBase = relBases[ci]; g.relCount = c.nrel
    local sets::Vector{Any} = Any[:(_col = $C[$j, _m]); Any[:($(Symbol("s", q)) = $(c.domain)[$q, _col]) for q in 1:c.nslots]]
    push!(parts, Expr(:let, Expr(:block, sets...), :(_rv[$j] = $(jl(g, c.lhs)) - $(jl(g, c.rhs)))))
  end
  local residualBody::Expr = Expr(:block, parts...)
  local R::Symbol = Symbol("LOOPRES_", f, "!")
  #= in Newton's iterations the relations hold their values: no crossing written, none changed =#
  local residuals::Function = (rv::Symbol, held::Bool = true) ->
    (held && hasRelations) ? :($R($rv, du, a, u, p, t, nothing, false, fire, _m)) : :($R($rv, du, a, u, p, t, z, mode, fire, _m))
  local set::Expr = :(for _j in 1:$k
    local _id::Int = $I[_j, _m]
    if _id <= $nS
      du[_id] = _x[_j]
    else
      a[_id - $nS] = _x[_j]
    end
  end)
  #= affine in its unknowns (`_affineIn`, by the unknowns' variables): solved by one Jacobian =#
  local reads::Set{Tuple{String, Bool}} = _unknownVariables(g, ids)
  local readsUnknown::Function = (r::Ref) -> (r.name, r.der) in reads
  local linear::Bool = all(((ci, _),) -> _affineIn(g.classes[ci].lhs, readsUnknown)[1] &&
                                         _affineIn(g.classes[ci].rhs, readsUnknown)[1], template)
  _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE") &&
    println("[arraycode] loop family ", f, ": ", k, " unknowns, ", M, " members, ", linear ? "linear" : "nonlinear",
            " (", join(String[_unknownLabel(g, u) for u in view(ids, :, 1)], ", "), ")",
            join(String["\n[arraycode]   " * first(g.classes[ci].text, 1200) for (ci, _) in template]))
  #= the step of a linear loop's solve and of each Newton step: its unknowns moved by the LU factors' solve =#
  local solveStep::Expr = :($(_luSolve!)(_dx, _F, _w.pivots, _r))
  local code::Expr = quote
    #= the work arrays: the parameters' on Float64 (no allocation in a right-hand side), else of the number type =#
    local _w::$(LoopWork){eltype(a)} = eltype(a) === Float64 ? p.loopWork[$f] : $(LoopWork){eltype(a)}($k)
    local _x::Vector{eltype(a)} = _w.x; local _r::Vector{eltype(a)} = _w.r; local _r2::Vector{eltype(a)} = _w.r2
    local _dx::Vector{eltype(a)} = _w.dx; local _xs::Vector{eltype(a)} = _w.xs
    local _J::Matrix{eltype(a)} = _w.J; local _F::Matrix{eltype(a)} = _w.F
    local _G::Matrix{Float64} = p.loops[$f]
    for _m in members
      $(any(>=(0), bases) ? quote
          if p.pass.on && $B[_m] >= 0
            local _b::Int = $B[_m]
            for _j in 1:$k
              _x[_j] = p.pass.x[_b + _j]
            end
            $set
            $(residuals(:_r))
            for _j in 1:$k
              p.pass.r[_b + _j] = _r[_j]
            end
            continue
          end
        end : nothing)
      local _converged::Bool = false
      $(linear ? quote
          #= affine in its unknowns: the Jacobian by differences over the unknowns' size (exact for an affine
             residual), one step from the last solution, and the step at the residuals there once more (the round-off
             of a badly conditioned loop); the Newton attempts below where it is singular or the second step not
             within 1e-8 =#
          for _j in 1:$k
            _x[_j] = _G[_j, _m]
          end
          $set
          $(residuals(:_r))
          for _j in 1:$k
            local _xj::eltype(a) = _x[_j]
            local _h::eltype(a) = max(one(_xj), abs(_xj))
            _x[_j] = _xj + _h
            $set
            $(residuals(:_r2))
            for _i in 1:$k
              _J[_i, _j] = (_r2[_i] - _r[_i]) / _h
            end
            _x[_j] = _xj
          end
          if all(isfinite, _r) && all(isfinite, _J) && $(_luFactor!)(copyto!(_F, _J), _w.pivots)
            $solveStep
            for _j in 1:$k
              _x[_j] -= _dx[_j]
            end
            $set
            $(residuals(:_r))
            $solveStep
            local _refined::eltype(a) = zero(eltype(a))
            for _j in 1:$k
              _x[_j] -= _dx[_j]
              _refined = max(_refined, abs(_dx[_j]) / (1.0 + abs(_x[_j])))
            end
            _converged = all(isfinite, _x) && _refined <= 1.0e-8
          end
        end : nothing)
      #= from the last solution; where that does not converge, again with the step damped (halved until the residuals
         decrease), then so from the start values. A piecewise equation's Newton step from a solution on another piece
         can cycle between two points that are no solutions (MSL IdealizedOpAmpLimited: from the saturated output,
         reached in an event's sweep, back to the linear range; the last iterate went into the equations as the
         solution), an exponential's can overflow (a Zener diode's) =#
      for _attempt in 1:3
        _converged && break
        for _j in 1:$k
          _attempt == 2 && (_xs[_j] = _x[_j])
          _x[_j] = _attempt == 3 ? $G[_j, _m] : _G[_j, _m]
        end
        for _it in 1:50
          $set
          $(residuals(:_r))
          for _j in 1:$k
            local _xj::eltype(a) = _x[_j]
            local _h::eltype(a) = 1.4901161193847656e-8 * max(one(_xj), abs(_xj))
            _x[_j] = _xj + _h
            $set
            $(residuals(:_r2))
            for _i in 1:$k
              _J[_i, _j] = (_r2[_i] - _r[_i]) / _h
            end
            _x[_j] = _xj
          end
          #= residuals or a Jacobian not finite, or a singular Jacobian (a trial point of the solver where the equations
             overflow, or a junction's derivative underflows: MSL DifferenceAmplifier's transistors): the unknowns NaN,
             so the step is rejected =#
          if !(all(isfinite, _r) && all(isfinite, _J))
            fill!(_x, NaN)
            break
          end
          if $(_luFactor!)(copyto!(_F, _J), _w.pivots)
            $solveStep
          elseif _attempt > 1 && eltype(a) === Float64
            #= singular where two unknowns whose product the equations take are both zero (`v = R_actual * i` at
               the starts, MSL's resistor with a heat port): the least-squares step of the smallest norm =#
            copyto!(_dx, LinearAlgebra.pinv(_J; rtol = $(RANK_TOLERANCE)) * _r)
          else
            fill!(_x, NaN)
            break
          end
          local _scale::Float64 = 1.0
          if _attempt > 1
            local _norm::eltype(a) = sum(abs2, _r)
            for _j in 1:$k
              _r2[_j] = _x[_j]
            end
            while true
              for _j in 1:$k
                _x[_j] = _r2[_j] - _scale * _dx[_j]
              end
              $set
              $(residuals(:_r))
              (sum(abs2, _r) <= _norm || _scale < 1.0e-3) && break
              _scale *= 0.5
            end
            for _j in 1:$k
              _x[_j] = _r2[_j]
            end
          end
          #= converged by the whole step of Newton's method, damped or not =#
          local _step::eltype(a) = zero(eltype(a))
          for _j in 1:$k
            _x[_j] -= _scale * _dx[_j]
            _step = max(_step, abs(_dx[_j]) / (1.0 + abs(_x[_j])))
          end
          _converged = _step <= 1.0e-8
          _step <= 1.0e-12 && break
        end
        _converged && break
        #= no attempt converged: the first one's last iterate (NaN where it was not finite: the step is rejected) =#
        if _attempt == 3
          for _j in 1:$k
            _x[_j] = _xs[_j]
          end
        end
      end
      $set
      #= the relations at the solution: their crossings, at an event their values =#
      $(hasRelations ? residuals(:_r, false) : nothing)
      #= the solution as the next guess: not a failed solve's NaN, nor the last iterate of a solve that did not
         converge (unknowns that hold no solution, as a state set's selection read with the other side's states: the
         next solve went to another root from there) =#
      if _converged && eltype(a) === Float64 && all(isfinite, _x)
        for _j in 1:$k
          _G[_j, _m] = _x[_j]
        end
      end
    end
  end
  return (Any[:(const $C = $cols), :(const $I = $ids), :(const $G = $guess), :(const $B = $bases)], code, residualBody)
end

"""
    _loopResidualFunction(f, body, paramLocals) -> Expr

The residuals of the loop family `f` (`body`, from `_loopFamilyCode`: `_rv[j]` the j-th residual of the member `_m`)
as the function `LOOPRES_<f>!(_rv, du, a, u, p, t, z, mode, fire, _m)`, with the parameter views it reads (of
`paramLocals`): not inlined, so compiled once for the places of the family's solve that call it (z and mode `nothing`
and `false` where the relations hold their values: a second specialization).
"""
function _loopResidualFunction(f::Int, body::Expr, paramLocals::Vector{Any})::Expr
  return :(@noinline function $(Symbol("LOOPRES_", f, "!"))(_rv, du, a, u, p, t, z, mode, fire, _m)
             $(_usedLocals(paramLocals, body)...)
             @inbounds begin
               $(body)
             end
             return nothing
           end)
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

"Whether the subscript `x` is known without the variables' values: literals, loop slots, operations of those."
_staticSubscript(x::IR)::Bool = x isa Lit || x isa Slot || (x isa Op && all(_staticSubscript, x.args))

"""
The code of the subscripts `subs` of the variable `name`: one computed from variable values (a table row chosen by a
discrete variable) is checked against its dimension (`_checkedSubscript`), as the generated functions run under
`@inbounds`; a start outside the range read memory outside the buffer (MSL Digital BUF3S).
"""
function _subscriptCode(g::Gen, info::VarInfo, name::String, subs::Vector{IR})::Vector{Any}
  local idx::Vector{Any} = Any[]
  for (k::Int, s::IR) in enumerate(subs)
    local code::Any = jl(g, s)
    if _staticSubscript(s)
      push!(idx, code)
    else
      k <= length(info.dims) || ns("a subscript of $(name) beyond its dimensions")
      push!(idx, Expr(:call, _checkedSubscript, code, info.dims[k], name))
    end
  end
  return idx
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
    if info.kind == :param
      info.vtype == :other && ns("parameter $(x.name) of a type the array path does not pass")
      local v = _paramValue!(g, x.name)
      v isa AbstractArray || return g.paramSyms[x.name]
      local (psym, pidx) = (g.paramSyms[x.name], Any[jl(g, s) for s in x.subs])
      return _discreteRead(info, all(_staticSubscript, x.subs) ? Expr(:ref, psym, pidx...) :
                                                                 Expr(:call, _checkedIndex, psym, pidx...))
    end
    local idx = _subscriptCode(g, info, x.name, x.subs)
    if info.kind == :state
      return x.der ? :(du[$(_linear(info, idx))]) : :(u[$(_linear(info, idx))])
    elseif info.kind == :discrete
      return _discreteRead(info, :(p.d[$(_linear(info, idx))]))
    else
      return :(a[$(_linear(info, idx))])
    end
  elseif x isa Pre
    local info = g.vars[x.ref.name]
    info.kind == :param && return jl(g, x.ref)   #= pre of a parameter =#
    local idx = _subscriptCode(g, info, x.ref.name, x.ref.subs)
    info.kind == :state && return :(p.upre[$(_linear(info, idx))])
    info.kind == :discrete && return _discreteRead(info, :(p.dpre[$(_linear(info, idx))]))
    #= an algebraic variable's value before the event's sweep (`p.aleft`, kept by `eventIteration!`): at the first
       sweep its left limit, the value `sample(u, clock)` takes at a tick where u jumps (MLS 16.5.1) =#
    if info.kind == :alg
      g.usesLeft = true
      return :(p.aleft[$(_linear(info, idx))])
    end
    error("IR: pre() of $(x.ref.name)")
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
    if x.op == :delay
      #= delay(value, time): the value `time` ago from the instance's history (`_numberDelays!`: its base and the
         class's column); the present value where the history has no point yet, or within a step longer than `time` =#
      x.args[1].v >= 0 ||
        ns("delay() where the array code keeps no history (an algorithm, an assert, an initial equation, a binding)")
      local history::Expr = :($(x.args[1].v) + _col)
      return :($(delayedValue)(p.state.delays, $history, Float64(t), Float64($(args[3])), Float64($(args[2]))))
    end
    #= a literal table: a subscript computed from variable values is checked (the code runs under @inbounds); an
       event function's value (`table[integer(delay(x, t))]`, MSL Digital's delays) is a number held as a Real =#
    if x.op == :index
      all(_staticSubscript, @view x.args[2:end]) && return Expr(:ref, args...)
      local subscripts::Vector{Any} = Any[]
      for k::Int in 2:length(args)
        push!(subscripts, x.args[k] isa EvFn ? :(round(Int, $(args[k]))) : args[k])
      end
      return Expr(:call, _checkedIndex, args[1], subscripts...)
    end
    x.op == :startRelation && return :($(_startRelation)($(args[2]), $(args[3]), $(getfield(Base, x.args[1].v))))
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
    x.op == :terminal && return :(p.state.terminal)
    x.op == :String && return Expr(:call, _modelicaString, args...)
    x.op == :fcall && return Expr(:call, args...)
    #= a call's outputs as arguments of an enclosing call (`_recordArrayCallIRx`) =#
    x.op == :splat && return Expr(:..., args[1])
    if x.op == :partial
      #= a function with inputs bound (`_partialApplicationIRx`): a closure of its free inputs =#
      local freeInputs::Vector{Symbol} = Symbol[]
      local callArgs::Vector{Any} = Any[]
      local boundAt::Int = 2
      for free::Bool in (x.args[2]::Lit).v
        free ? push!(freeInputs, Symbol("#pa#", length(freeInputs) + 1)) : (boundAt += 1)
        push!(callArgs, free ? freeInputs[end] : args[boundAt])
      end
      return Expr(:->, Expr(:tuple, freeInputs...), Expr(:call, args[1], callArgs...))
    end
    #= a relation with the probe of its abs() argument (`_withAbsProbe`): the probe's crossing kept, its value not =#
    x.op == :probed && return Expr(:block, args[2], args[1])
    #= a temporary's value (a tuple equation's call) =#
    x.op == :setlocal && return Expr(:(=), args...)
    x.op == :assertcall && return :(($(args[1])) || _assertFailed(p, $(args[3]), ($(args[3]), _col), t, $(args[2])))
    x.op == :terminate && return :(_terminate!(p, $(args[1])))
    x.op == :stringparam && return :(p.state.strings[$(x.args[1].v)])
    #= an external object: the module's constant (`_objectDefinitions`) =#
    x.op == :object && return Symbol("OBJECT_", x.args[1].v)
    x.op == :seq && return Expr(:block, args...)
    x.op == :noEvent && return args[1]
    #= homotopy(actual, simplified): actual at λ = 1, the blend in the continuation (`homotopyStart!`) =#
    x.op == :homotopy &&
      return :(let _λ = p.homotopy[1]; _λ >= 1 ? $(args[1]) : _λ * $(args[1]) + (1 - _λ) * $(args[2]) end)
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
      if info.kind == :param
        #= a parameter with fixed = false, in an initial algorithm: its local (a view of
           p.values for an array; a scalar is written back after the algorithms) =#
        local sym = (_paramValue!(g, st.target.name); g.paramSyms[st.target.name])
        local pidx = Any[jl(g, s) for s in st.target.subs]
        push!(out, isempty(pidx) ? :($sym = $(jl(g, st.value))) :
                   all(_staticSubscript, st.target.subs) ? :($sym[$(pidx...)] = $(jl(g, st.value))) :
                   Expr(:call, _checkedAssign!, sym, jl(g, st.value), pidx...))
        continue
      end
      local idx = _subscriptCode(g, info, st.target.name, st.target.subs)
      push!(out, info.kind == :discrete ? :(p.d[$(_linear(info, idx))] = Float64($(jl(g, st.value)))) :
                                          :(a[$(_linear(info, idx))] = $(jl(g, st.value))))
    elseif st isa SPreStart
      local preInfo::VarInfo = g.vars[st.target.name]
      local preIdx = _subscriptCode(g, preInfo, st.target.name, st.target.subs)
      push!(out, :(p.dpre[$(_linear(preInfo, preIdx))] = Float64($(jl(g, st.value)))))
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
  #= a when-branch with terminal() among its triggers: the terminal event after the solve (`_terminalEvent!`) =#
  local usesTerminal::Bool = any(w::WhenClass -> _hasOp(w.cond, :terminal), g.whens)
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
  local initialSystem::InitialSystem = schedule.initialSystem
  local (initBody::Vector{Any}, _) = _initialResiduals(g)
  #= initial algorithms: code, and what they assign =#
  local initAlgOuts::Set{String} = _initAlgAssigned(g)
  #= the variables of lifted clusters of discrete equations that an initial algorithm gives their initialized values
     (the cluster's initial when-equation, `initial()` true): where the cluster runs at the start's iteration, pre()
     of them is that value, not the start (`initialAlgorithms!`) =#
  local clusterNames::Set{String} = Set{String}()
  for w::WhenClass in g.whens, (kind::Symbol, target::Ref, _) in w.body
    (w.everyEvent && kind == :assign) && push!(clusterNames, target.name)
  end
  local clusterPre::Vector{Int} = Int[]
  for nm::String in sort!(collect(intersect(clusterNames, initAlgOuts)))
    g.vars[nm].kind == :discrete || continue
    append!(clusterPre, (g.vars[nm].offset + 1):(g.vars[nm].offset + prod(g.vars[nm].dims; init = 1)))
  end
  g.relBase = 0; g.relCount = 0
  #= the initial algorithms in their order: those that compute parameters from parameters before the initial system
     (`_parameterAlgorithms`, `parameterAlgorithms!`), the others after it; a refresh of the equations before an
     early one is the next late one's =#
  local early::BitVector = _parameterAlgorithms(g)
  local earlyAlgs::Vector{AlgClass} = g.initAlgs[early]
  local paramAlgOuts::Set{String} = _initAlgAssigned(g, early)
  local initAlgBody = Any[]
  local paramAlgBody = Any[]
  local (initAlgOrdered::Vector{AlgClass}, planRefresh::BitVector) = _initAlgPlan(g)
  local initAlgRefresh::BitVector = falses(0)
  local carried::Bool = false
  for (k::Int, ia::AlgClass) in enumerate(initAlgOrdered)
    local blk = Expr(:block, _stmtCode(g, ia.body, 0, 0)...)
    local code = ia.nslots == 0 ? Expr(:let, :(_col = 1), blk) :
      _loopCode(ia.nslots, ia.domain, collect(1:size(ia.domain, 2)), blk)
    if any(e::AlgClass -> e === ia, earlyAlgs)
      push!(paramAlgBody, code)
      carried |= planRefresh[k]
    else
      push!(initAlgBody, code)
      push!(initAlgRefresh, carried || planRefresh[k])
      carried = false
    end
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
  #= Code: one loop per batch (body: all rows but the time-invariant ones, which equations! takes from invariants!;
     rhsBody: the rows the state derivatives need, condBody: those the relations need, invariantBody: the
     time-invariant ones; Schedule). MSL EngineV6: the invariant rows were 28 % of equations! (120k of 426k nodes),
     compiled there and in invariants!. =#
  local body = Any[]; local rhsBody = Any[]
  local invariantBody = Any[]; local condBody = Any[]
  local emit! = function (nslots, domain, batch, code)
    local dynamic = [r for r in batch if !invariant[r]]
    isempty(dynamic) ||
      push!(body, nslots == 0 ? Expr(:let, :(_col = 1), code) : _loopCode(nslots, domain, [el.col[r] for r in dynamic], code))
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
  local familyBases::Vector{Vector{Int}} = [Int[] for _ in loops.families]
  local blockMember::Vector{Int} = zeros(Int, length(loops.blocks))
  for (b, f) in enumerate(_blockFamilies(el, pos, loops))
    push!(familyBlocks[f], loops.blocks[b])
    push!(familyBases[f], initialSystem.blockBase[b])
    blockMember[b] = length(familyBlocks[f])
  end
  #= per unknown of the loops in the initial solve (`InitialSystem`) its guess: (family, position, member) of
     p.loops, (-k, position, 0) of the k-th large block's =#
  local initPlaces::Vector{Tuple{Int, Int, Int}} = fill((0, 0, 0), initialSystem.nLoop)
  local familyTemplates::Vector{Vector{Tuple{Int, Int}}} = copy(loops.families)
  local loopDefs::Vector{Any} = Any[]; local loopBodies::Vector{Any} = Any[]
  local loopResiduals::Dict{Int, Expr} = Dict{Int, Expr}()
  #= where each algebraic element is a loop's first guess (by its index among the algebraic variables): a start
     rule of an algebraic variable sets them (`:a`, a guess computed at the start) =#
  local guessTargets::Dict{Int, Vector{Any}} = Dict{Int, Vector{Any}}()
  local loopFamily!::Function = function (f::Int, template::Vector{Tuple{Int, Int}}, blocks::Vector{Vector{Int}},
                                          bases::Vector{Int})
    for (m::Int, b::Vector{Int}) in enumerate(blocks), j::Int in eachindex(b)
      local unknown::Int = matchRow[b[j]]
      unknown > g.nStates && push!(get!(() -> Any[], guessTargets, unknown - g.nStates), :(p.loops[$f][$j, $m]))
      bases[m] >= 0 && (initPlaces[bases[m] + j] = (f, j, m))
    end
    local (defs::Vector{Any}, code::Expr, residualBody::Expr) = _loopFamilyCode(g, f, template, blocks, el, matchRow, a0, relBases, bases)
    append!(loopDefs, defs)
    push!(loopBodies, code)
    loopResiduals[f] = residualBody
  end
  #= a family's blocks are alike: all large (blockSolve!) or all Newton's =#
  local largeFamily::BitVector = falses(length(familyTemplates))
  for (b, f) in enumerate(_blockFamilies(el, pos, loops))
    largeFamily[f] = large[b]
  end
  foreach(f::Int -> largeFamily[f] ? push!(loopBodies, :(nothing)) :
                                      loopFamily!(f, familyTemplates[f], familyBlocks[f], familyBases[f]),
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
  local blockLinear::Vector{Bool} = Bool[]
  local largeBlock!::Function = function (b::Int)
    local rs::Vector{Int} = loops.blocks[b]
    local torn::TornBlock = schedule.torn[b]::TornBlock
    local ids::Vector{Int} = torn.tears
    push!(blockPatterns, torn.pattern)
    push!(blockGuesses, Float64[u <= g.nStates ? 0.0 : a0[u - g.nStates] for u in ids])
    for (j::Int, u::Int) in enumerate(ids)
      u > g.nStates && push!(get!(() -> Any[], guessTargets, u - g.nStates), :(p.blocks[$(length(blockGuesses))].y[$j]))
    end
    #= affine in its unknowns (all of them: the iteration variables and the explicitly solved ones), by the unknowns'
       variables: the explicit solutions and the residuals each affine (`_affineIn`), so the residuals are affine in the
       iteration variables =#
    local reads::Set{Tuple{String, Bool}} = _unknownVariables(g, Int[matchRow[r] for r in rs])
    local readsUnknown::Function = (r::Ref) -> (r.name, r.der) in reads
    local linear::Bool = all(((key, _),) -> _affineIn(last(solutions[key]::Tuple{Ref, IR}), readsUnknown)[1], torn.batches) &&
                         all(i::Int -> _affineIn(g.classes[el.class[rs[i]]].lhs, readsUnknown)[1] &&
                                       _affineIn(g.classes[el.class[rs[i]]].rhs, readsUnknown)[1], torn.residuals)
    push!(blockLinear, linear)
    _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE") &&
      println("[arraycode] block of ", length(rs), " instances torn: ", length(ids), " iteration variables, ",
              length(torn.batches), " explicit batches, ", linear ? "linear" : "nonlinear", "\n[arraycode]   ",
              join(String[_unknownLabel(g, u) for u in ids], ", "), "\n[arraycode]   residuals: ",
              join(String[first(g.classes[el.class[rs[i]]].text, 100) for i in torn.residuals], "; "))
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
    local kb::Int = length(blockBodies)
    local call::Expr = :($(blockSolve!)(p.blocks[$kb], $(Symbol("BLOCK_", kb, "!")), du, a, u, p, t, z, mode, fire))
    #= held by the initial solve: its unknowns from it, its residuals to it (`InitialPass`) =#
    local base::Int = initialSystem.blockBase[b]
    if base >= 0
      foreach(j::Int -> (initPlaces[base + j] = (-kb, j, 0)), eachindex(ids))
      call = :(p.pass.on ? $(_passBlock!)(p.blocks[$kb], $(Symbol("BLOCK_", kb, "!")), p.pass, $base, du, a, u, p, t,
                                          fire) : $call)
    end
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
         solved by Newton on its residual, as a loop of one =#
      push!(familyTemplates, [(ci, p)])
      local f::Int = length(familyTemplates)
      loopFamily!(f, familyTemplates[f], [[r] for r in batch], Int[get(initialSystem.rowBase, r, -1) for r in batch])
      callFamily!(f, collect(1:length(batch)), Int[k for (k, r) in enumerate(batch) if needed[r]],
                  Int[k for (k, r) in enumerate(batch) if relNeeded[r]])
      continue
    end
    local (target::Ref, value::IR) = solution
    g.relBase = relBases[ci]; g.relCount = c.nrel
    local tinfo = g.vars[target.name]
    local assign = tinfo.kind == :discrete ?
      :(p.d[$(_linear(tinfo, _subscriptCode(g, tinfo, target.name, target.subs)))] = Float64($(jl(g, value)))) :
      :($(jl(g, target)) = $(jl(g, value)))
    emit!(c.nslots, c.domain, batch, assign)
  end
  #= When-equations: each instance fires when its condition becomes true. =#
  local whenBody = Any[]; local whenConds = Any[]
  #= the when-bodies' calls that read an external object, one entry per branch that has any: its when-equation and the
     calls (`objectCall!`, `replayObjects!`) =#
  local objectCalls = Tuple{Int, Expr}[]
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
    #= per branch: the number of its object calls (0: none) and the calls =#
    local callIds::Vector{Int} = zeros(Int, length(branchConds))
    for b in eachindex(branchConds)
      local reads::Vector{Any} = Any[jl(g, value) for (i, (kind, _, value)) in enumerate(w.body)
                                     if findlast(st -> st <= i, w.starts) == b && kind != :reinit && kind != :switch &&
                                        _hasOp(value, :object)]
      isempty(reads) && continue
      push!(objectCalls, (wi, Expr(:block, reads...)))
      callIds[b] = length(objectCalls)
    end
    for (i, (kind, target, value)) in enumerate(w.body)
      local b = findlast(st -> st <= i, w.starts)
      if kind == :call
        push!(stmtsOf[b], jl(g, value))
        continue
      end
      local info = g.vars[target.name]
      local idx = _subscriptCode(g, info, target.name, target.subs)
      if kind == :reinit
        push!(stmtsOf[b], :(p.ureinit[$(_linear(info, idx))] = $(jl(g, value))))
      elseif kind == :switch
        #= a dynamic state set's switch: the state takes the newly selected candidate's value with the selection
           matrix the body assigns, not after the sweep. A pass with the new selection and the old state read
           the candidates wrong (the state another candidate's value): the switch fired back, and the event
           iteration did not settle (PendulumArray at t = 0.356) =#
        push!(stmtsOf[b], :(p.ureinit[$(_linear(info, idx))] = $(jl(g, value))))
        push!(stmtsOf[b], :(u[$(_linear(info, idx))] = p.ureinit[$(_linear(info, idx))]))
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
      local logged::Vector{Any} = callIds[b] == 0 ? Any[] :
        Any[:(push!(p.state.objectLog, (Float64(t), $(callIds[b]), _col))),
            :(p.state.objectAt = length(p.state.objectLog))]
      local blk = Expr(:block, logged..., stmtsOf[b]..., :(_fired = true))
      #= a lifted cluster of discrete equations (`everyEvent`): wherever the discrete equations run in event mode, at
         every pass of every event and of the start's iteration (pre() the initial values), its conditions the events'
         sources =#
      local guard = w.everyEvent ? :(mode || p.state.forced) : :(fire && ($(rises[b])))
      inst = inst === nothing ? Expr(:if, guard, blk) : Expr(b == 1 ? :if : :elseif, guard, blk, inst)
    end
    push!(whenConds, w.nslots == 0 ? Expr(:let, :(_col = 1), cond) : _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), cond))
    push!(whenBody, w.nslots == 0 ? Expr(:let, :(_col = 1), inst) : _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), inst))
  end
  #= the object calls of one logged firing: its entry `_id`, its instance `_which` =#
  local objectCases::Vector{Any} = Any[]
  for (id::Int, (wi::Int, calls::Expr)) in enumerate(objectCalls)
    local w::WhenClass = g.whens[wi]
    local one::Expr = :(if _col == _which; $calls; end)
    local instances::Expr = w.nslots == 0 ? Expr(:let, :(_col = 1), one) :
      _loopCode(w.nslots, w.domain, collect(1:size(w.domain, 2)), one)
    push!(objectCases, :(if _id == $id; $instances; end))
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
    if buffer === :a
      #= an algebraic variable's start: the first guess of the loops that iterate on it =#
      foreach(target -> push!(startStmts, :($target = Float64($v))), get(guessTargets, k, Any[]))
      continue
    end
    push!(startStmts, buffer === :u ? :(u0[$k] = Float64($v)) : :(p.d[$k] = Float64($v); p.dpre[$k] = p.d[$k]))
    (buffer === :d && _readsParameterOf(x, free) && !(get(discreteOwner, k, "") in initAssigned)) &&
      push!(freeStartStmts, :(p.d[$k] = Float64($v); p.dpre[$k] = p.d[$k]))
  end
  local (delayBody::Vector{Any}, nDelays::Int) = _delayRecords(g)
  local usedParams = collect(keys(g.paramSyms))
  #= a model without continuous unknowns (its one state the `\$dummy` it is given to be integrated): its parameters
     are its result, as the scalar route keeps them (`eliminateDeadParameters`): every parameter with a value in the
     layout, those no equation reads too (MSL readRealParameterModel: J, phi_rel0, w_rel0 read from a file, which its
     reference compares); no locals of them =#
  if stateNamesOrdered == String["\$dummy"] && isempty(algNamesOrdered)
    for (nm::String, info::VarInfo) in g.vars
      (info.kind == :param && info.var.value !== nothing) && _paramValue!(g, nm)
    end
  end
  local (paramOrder, rules, bindingKinds) = _parameterRules(g)
  local objectDefs::Vector{Any} = _objectDefinitions(g, modelName)
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
  local initStates = Int[k for k in initUnknowns if 0 < k <= g.nStates]
  local initParams = Int[layoutOf[freeCols[k - g.nStates][1]][1] + freeCols[k - g.nStates][2] for k in initUnknowns if k > g.nStates]
  #= the initial equations' instances of the square system, and those beyond the free states (matched to none:
     checked at the solution, `checkRedundantInitial`), with their equations' texts =#
  local initRows::Vector{Int} = findall(!=(0), initUnknowns)
  local initRedundant::Vector{Int} = findall(==(0), initUnknowns)
  local initTexts::Vector{String} = String[c.text for c in g.initClasses for _ in 1:size(c.domain, 2)]
  local redundantTexts::Vector{String} = String[first(initTexts[r], 120) for r in initRedundant]
  #= the scalar parameters the initial algorithms computed, from their locals to p.values =#
  local initAlgWriteBack = Any[:(p.values[$(layoutOf[nm][1] + 1)] = $(g.paramSyms[nm]))
                               for nm in sort!(collect(initAlgOuts))
                               if nm in g.freeParams && haskey(layoutOf, nm) && !(g.params[nm] isa AbstractArray)]
  local paramAlgWriteBack = Any[:(p.values[$(layoutOf[nm][1] + 1)] = $(g.paramSyms[nm]))
                                for nm in sort!(collect(paramAlgOuts))
                                if nm in g.freeParams && haskey(layoutOf, nm) && !(g.params[nm] isa AbstractArray)]
  #= the initial algorithms in their order; before one that reads what the equations compute from an earlier one's
     assignments (`_initAlgPlan`), the equations again, with the parameters assigned so far =#
  local initAlgCode = Any[]
  for (k::Int, body::Any) in enumerate(initAlgBody)
    initAlgRefresh[k] &&
      push!(initAlgCode, Expr(:block, initAlgWriteBack..., :(equations!(du, a, u, p, t, $(NO_CROSSINGS), true))))
    push!(initAlgCode, body)
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
    :(Symbolics.SparseArrays.sparse($(pattern[1]), $(pattern[2]), ones($(length(pattern[1]))), $nStates, $nStates, $(_keepFirst)))
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
    import LinearAlgebra
    #= the MTK path's function code names it (algorithmic.jl: an output array's growth) =#
    const OMBackend = $(_OMBACKEND)

    $(functionDefs...)
    $(loopDefs...)
    $(splitDefs...)
    $([:(function $(Symbol("LOOPFAMILY_", f, "!"))(du, a, u, p, t, z, mode, fire, members)
           $(_usedLocals(paramLocals, code)...)
           @inbounds begin
             $(code)
           end
           return nothing
         end) for (f, code) in enumerate(loopBodies)]...)
    $([_loopResidualFunction(f, body, paramLocals) for (f, body) in sort!(collect(loopResiduals); by = first)]...)
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
    $(objectDefs...)
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
    #= whether each is affine in its unknowns (`_affineIn`: a direct solve) =#
    const BLOCK_LINEAR = $(blockLinear)
    #= the loop families' first guesses; none of a family of large blocks (BLOCK_GUESSES) =#
    const LOOP_GUESSES = Matrix{Float64}[$([get(largeFamily, f, false) ? :(zeros(0, 0)) : Symbol("LOOP_GUESS_", f)
                                            for f in eachindex(loopBodies)]...)]
    #= initialization: the states the initial equations determine (one per residual) =#
    const INIT_UNKNOWNS = $(initStates)
    const INIT_PARAMS = $(initParams)   #= fixed = false parameters: positions in p.values =#
    const N_INIT = $(nInit)
    #= the initial equations' instances the unknowns are solved from, and the ones beyond them =#
    const INIT_ROWS = $(initRows)
    const INIT_REDUNDANT = $(initRedundant)
    const INIT_REDUNDANT_TEXTS = $(redundantTexts)
    #= the loops the initial solve holds (`InitialSystem`): their unknowns' number and guesses' places, and per row of
       the solve the unknowns it reads (`nothing`: from finite differences) =#
    const N_INIT_LOOPS = $(initialSystem.nLoop)
    const INIT_LOOP_PLACES = $(initPlaces)
    const INIT_READS = $(initialSystem.reads)
    const D0 = $(d0)
    const N_RELATIONS = $(nRelTotal)
    #= the relation numbering's relations (not event-generating functions' crossing functions) =#
    const RELATION_SLOT = $(_relationSlots(g, nRelTotal, relBases, whenRelBases, algRelBases))
    #= per slot the first slot of its equation instance: the relations of one if-expression (`_searchRelations!`) =#
    const RELATION_INSTANCE = $(_relationInstances(g, nRelTotal, relBases, whenRelBases, algRelBases))
    #= the slots of relations of time and what changes only at events (`_timeRel`): their crossings are time events =#
    const TIME_SLOT = $(_timeRelationSlots(g, nRelTotal, relBases, whenRelBases, algRelBases))
    const USES_INITIAL = $(g.usesInitial)
    const USES_LEFT = $(g.usesLeft)
    const USES_HOMOTOPY = $(g.usesHomotopy)
    #= the histories of the delay() calls' instances (`delayNow!`) =#
    const N_DELAYS = $(nDelays)
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
       relation crossing functions into z (when given), relation values updated in mode. The time-invariant values
       first (invariants!: they read parameters and each other only). =#
    function equations!(du, a, u, p, t, z, mode::Bool, fire::Bool = false)
      $(paramLocals...)
      local _fired = false
      invariants!(a, p)
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

    #= The when-bodies' calls that read an external object, made again for a read of the solution (`algebraics`) and
       for a second solve (`_restoreStart!`): an MSL table keeps the event interval of its last `getNextTimeEvent`
       call, after a solve the last one (a CombiTimeTable's outputs read the last segment's value at every time). Its
       C code moves the interval forward by one at a call (an event), and starts again at a call before the last =#
    const POSITIONS_OBJECTS = $(!isempty(objectCalls))
    function objectCall!(du, a, u, p, t, z, mode::Bool, fire::Bool, _id::Int, _which::Int)
      $(paramLocals...)
      @inbounds begin
        $(objectCases...)
      end
      return nothing
    end

    #= The objects as the run left them at time `tau`: the logged calls up to it made again in their order, from
       where the objects are (`objectAt`) or, for an earlier time, from the first; each with the variables of
       its time (the state `u` is the read's: the calls' arguments are functions of time) =#
    function replayObjects!(u, p, tau)
      POSITIONS_OBJECTS || return nothing
      local log = p.state.objectLog
      local m = searchsortedlast(log, (Float64(tau), typemax(Int), typemax(Int)))
      local from = m >= p.state.objectAt ? p.state.objectAt + 1 : 1
      local a = zeros($nAlg); local du = similar(u)
      for i in from:m
        local (ti, id, which) = log[i]
        equations!(du, a, u, p, ti, $(NO_CROSSINGS), false)
        objectCall!(du, a, u, p, ti, nothing, false, false, id, which)
      end
      p.state.objectAt = m
      return nothing
    end

    const ASSERTS = $(assertTexts)
    #= A model function with an assert: the equations are evaluated after each step as for
       the model's asserts (a call only an algebraic variable reads was never made: OMJL's
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
        equations!(du, a, u, p, t, $(NO_CROSSINGS), false)
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
          equations!(du, a, u, p, t, $(NO_CROSSINGS), false)
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
          DifferentialEquations.derivative_discontinuity!(integrator, false)
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

    #= The present values of the delay() calls' arguments and delay times at (u, t), into the run's delay state. =#
    function delayNow!(u, p, t)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, $(NO_CROSSINGS), false)
      $(paramLocals...)
      local _now = p.state.delays.now; local _times = p.state.delays.times
      @inbounds begin
        $(delayBody...)
      end
      return nothing
    end

    #= The delay() arguments at the integrator's point as left limits: a step's end, the state before an event. =#
    function _delayLeft!(integrator)
      delayNow!(integrator.u, integrator.p, integrator.t)
      $(recordDelayStepEnd!)(integrator.p.state.delays, Float64(integrator.t))
      return false
    end

    #= After the events of a step: an argument that changed at its end jumped (due again its delay time later). =#
    function _delayJumps!(integrator)
      delayNow!(integrator.u, integrator.p, integrator.t)
      $(recordDelayJumps!)(() -> nothing, integrator.p.state.delays, Float64(integrator.t), integrator)
      return false
    end

    #= At a solve's start, after the other callbacks' initializations: each history's first point. =#
    function _delayStart!(integrator)
      delayNow!(integrator.u, integrator.p, integrator.t)
      $(startDelayHistories!)(integrator.p.state.delays, Float64(integrator.t))
      return nothing
    end

    #= A delayed jump is due (delays.jl): the event iteration with the delayed values as right limits, until the
       recorder after the events (`_delayJumps!`). =#
    function affectDelay!(integrator)
      integrator.p.state.delays.rightLimit = true
      _eventAt!(integrator)
      _logDiscretes!(integrator.p, integrator.t)
      DifferentialEquations.derivative_discontinuity!(integrator, true)
      return nothing
    end

    #= Residuals of the initial equations at (u, t), du and a from equations!. =#
    function initialResiduals!(res, du, a, u, p, t)
      $(paramLocals...)
      @inbounds begin
        $(initBody...)
      end
      return nothing
    end

    #= The loops' unknowns of the initial solve (INIT_LOOP_PLACES): their guesses, and the guesses set to them. =#
    function loopGuesses(p)
      return Float64[f > 0 ? p.loops[f][j, m] : p.blocks[-f].y[j] for (f, j, m) in INIT_LOOP_PLACES]
    end
    function loopGuesses!(p, x)
      for (k, (f, j, m)) in enumerate(INIT_LOOP_PLACES)
        f > 0 ? (p.loops[f][j, m] = x[k]) : (p.blocks[-f].y[j] = x[k])
      end
      return p
    end

    #= The initial state: the free states (and fixed = false parameters, into p.values) the
       initial equations determine, solved for them with the unknowns of the loops the initial equations reach
       (`InitialSystem`, from `guesses`; the loops' residuals rows of the solve), as OpenModelica's initial system;
       the loops solved again in the equations at the solution, from it. =#
    function initialize(u0, p, t, guesses = loopGuesses(p))
      local nS = length(INIT_UNKNOWNS)
      local nR = nS + length(INIT_PARAMS)
      local u = copy(u0)
      local du = similar(u); local a = similar(u, $nAlg)
      local setUnknowns! = function (uu, z)
        uu[INIT_UNKNOWNS] .= view(z, 1:nS)
        p.values[INIT_PARAMS] .= view(z, (nS + 1):nR)
        loopGuesses!(p, view(z, (nR + 1):length(z)))
        return uu
      end
      #= the relations keep their values during a solve =#
      local residual = function (res, z, _)
        local uz = similar(z, length(u0))
        uz .= u0
        setUnknowns!(uz, z)
        local duz = similar(uz); local az = similar(uz, $nAlg)
        p.pass.x .= view(z, (nR + 1):length(z))
        p.pass.on = true
        try
          equations!(duz, az, uz, p, t, $(NO_CROSSINGS), false)
        finally
          p.pass.on = false
        end
        if isempty(INIT_REDUNDANT)
          initialResiduals!(view(res, 1:nR), duz, az, uz, p, t)
        else
          local every = similar(res, N_INIT)
          initialResiduals!(every, duz, az, uz, p, t)
          view(res, 1:nR) .= view(every, INIT_ROWS)
        end
        view(res, (nR + 1):length(res)) .= p.pass.r
        return nothing
      end
      #= MLS 8.6, as the MTK path: the relations at the guess, a solve with them, then at the
         solution again, until they settle (a branch the solved state selects); relations that
         cycle (no consistent branch) keep the first solution =#
      local z = vcat(u0[INIT_UNKNOWNS], p.values[INIT_PARAMS], guesses)
      #= sorted into blocks as OpenModelica's initial system (arrayInitialBlocks.jl), each solved scaled in a trust
         region; Newton where that fails =#
      local scale = map(x -> x == 0 ? 1.0 : abs(x), z)
      local whole = (res, w) -> residual(res, w, nothing)
      local scaledSolve = z0 -> $(sortedInitialSolve)(whole, Vector{Float64}(z0), scale, INIT_READS)
      local first = nothing
      local seen = Tuple{Vector{Bool}, Vector{Float64}}[]
      for _ in 1:20
        equations!(du, a, setUnknowns!(u, z), p, t, $(NO_CROSSINGS), true)
        local rel0 = copy(p.rel); local e0 = copy(p.evv)
        if (rel0, e0) in seen
          equations!(du, a, setUnknowns!(u, first), p, t, $(NO_CROSSINGS), true)
          return u
        end
        push!(seen, (rel0, e0))
        local (ok, zs, _) = scaledSolve(z)
        if ok
          z = zs
          first === nothing && (first = copy(z))
          equations!(du, a, setUnknowns!(u, z), p, t, $(NO_CROSSINGS), true)
          (p.rel == rel0 && p.evv == e0) && return u
          continue
        end
        local sol = $(newtonInitialSolve)(residual, Vector{Float64}(z))
        #= no solution: refused as the MTK path refuses it (OpenModelica fails there too) =#
        DifferentialEquations.SciMLBase.successful_retcode(sol) ||
          $(_OMBACKEND.unsupported)("fixed start values or initial equations that the initialization cannot hold",
                                    string("the initial system (", sol.retcode, ")"))
        z = copy(sol.u)
        first === nothing && (first = copy(z))
        equations!(du, a, setUnknowns!(u, z), p, t, $(NO_CROSSINGS), true)
        (p.rel == rel0 && p.evv == e0) && return u
      end
      equations!(du, a, setUnknowns!(u, first), p, t, $(NO_CROSSINGS), true)
      return u
    end

    #= The initial equations beyond the free states (INIT_REDUNDANT) at the initial state: each satisfied by what the
       others gave, as OpenModelica keeps a model whose redundant initial equations are consistent; an error for
       one that is not. =#
    function checkRedundantInitial(u, p, t)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, $(NO_CROSSINGS), false)
      local res = zeros(N_INIT)
      initialResiduals!(res, du, a, u, p, t)
      for (k, r) in enumerate(INIT_REDUNDANT)
        abs(res[r]) <= 1.0e-6 ||
          $(_OMBACKEND.unsupported)("fixed start values or initial equations that the initialization cannot hold",
                                    string("the initialization is over-determined: ", INIT_REDUNDANT_TEXTS[k],
                                           " (residual ", res[r], ")"))
      end
      return nothing
    end

    #= The asserts of the initial equations, once after the initialization. =#
    function checkInitialAsserts(u, p, t)
      local du = similar(u); local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, $(NO_CROSSINGS), false)
      $(paramLocals...)
      @inbounds begin
        $(initAssertBody...)
      end
      return nothing
    end

    #= The initial algorithms that compute parameters from parameters (`_parameterAlgorithms`), once at the start,
       before the initial system that reads them. =#
    function parameterAlgorithms!(u, p, t)
      $(paramLocals...)
      local z = nothing; local mode = true; local fire = false; local _fired = false
      p.checking[1] = true
      try
        @inbounds begin
          $(paramAlgBody...)
        end
        $(paramAlgWriteBack...)
      finally
        p.checking[1] = false
      end
      return nothing
    end

    #= The initial algorithms, once at the start (asserts checked); the variables they read that the equations
       compute, from the equations at the initial state, evaluated before the first algorithm that reads one and
       again where an earlier algorithm's assignments reach them (`_initAlgPlan`). =#
    function initialAlgorithms!(u, p, t)
      $(paramLocals...)
      local du = similar(u); local a = similar(u, $nAlg)
      local z = nothing; local mode = true; local fire = false; local _fired = false
      p.checking[1] = true
      try
        @inbounds begin
          $(initAlgCode...)
        end
        $(initAlgWriteBack...)
        #= a lifted cluster's variables: their initialized values are pre() of them from here on (a friction
           element's mode, Backward by `initial() and w_relfric < 0`; with pre() the start, Unknown, the start's
           iteration made a clutch at a relative speed of -10 Stuck: MSL Rotational.Examples.Friction) =#
        for _k in $clusterPre
          p.dpre[_k] = p.d[_k]
        end
      finally
        p.checking[1] = false
      end
      return nothing
    end

    #= The start's mixed system (MLS 8.6: pre() the start values), as OpenModelica and the MTK path
       (`_settleInitialDiscretes!`): the initial system was solved with the lifted clusters' variables at `used`
       (their start values); their initial arms (`initialAlgorithms!`) gave their values at that solution. Where
       those differ, the initial system again from the start values `uStart` with them, pre() of them the start
       values `preStart`, until they stay (a cycle or 10 passes: the last solution stands, the start's event
       iteration settles them). MSL RobotR3's bearing friction: solved unlocked (`a = sa + tau0_max`), the steady
       start held the motor at the friction's limit (La.i = -0.43, omc 0); locked, sa keeps its start in the
       initial system (`InitialSystem`). =#
    $(isempty(clusterPre) ? nothing : quote
        function initialClusters!(uStart, u, p, t, preStart::Vector{Float64}, used::Vector{Float64},
                                  guessStart::Vector{Float64})
          length(INIT_ROWS) > 0 || return u
          local seen = Vector{Float64}[used]
          for _ in 1:10
            local now::Vector{Float64} = p.d[$clusterPre]
            now in seen && return u
            push!(seen, now)
            p.dpre[$clusterPre] .= preStart
            u = initialize(copy(uStart), p, t, copy(guessStart))
            isempty(INIT_REDUNDANT) || checkRedundantInitial(u, p, t)
            initialAlgorithms!(u, p, t)
          end
          return u
        end
      end)

    #= The starts of states and discrete variables that read a parameter, from the parameter values (an override
       reaches them); the loops' first guesses that are starts of algebraic variables computed at the start. =#
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
       of MSL DifferenceAmplifier's sources (a damped 0.2 GHz sine, ramps of 1 ns), and its steps shrank to 1e-25. At
       most a thousandth of the run times that: a step relative to |t| grew with the run, and its truncation error
       with its square (MSL Machines SMEE_Generator at t = 22 s: a step of 1.3e-4 s, 3e-4 off the 50 Hz sources'
       derivative; Rosenbrock's steps shrank from 5e-3 to 1.5e-4 s over the run, 619000 for 30 s). Divided by the
       difference of the two times as represented. An error in the right-hand side is recorded as the step-rejecting
       RHS's (`StepRejectingRHS`). =#
    function tgrad!(dT, u, p, t)
      local tend::Float64 = abs(p.tend[1])
      local h::Float64 = cbrt(eps(Float64)) * max(min(abs(Float64(t)), 1.0e-3 * tend), 1.0e-9 * tend, floatmin(Float64))
      local (tAfter::Float64, tBefore::Float64) = (Float64(t) + h, Float64(t) - h)
      local after::Vector{eltype(dT)} = eltype(dT) === Float64 ? p.tgradAfter : similar(dT)
      local before::Vector{eltype(dT)} = eltype(dT) === Float64 ? p.tgradBefore : similar(dT)
      try
        RHS!(after, u, p, tAfter)
        RHS!(before, u, p, tBefore)
      catch err
        err isa OMBackend.CodeGeneration._RHS_STEP_ERRORS || rethrow()
        OMBackend.CodeGeneration.LAST_RHS_ERROR[] = err
        fill!(dT, NaN)
        return nothing
      end
      dT .= (after .- before) ./ (tAfter - tBefore)
      return nothing
    end

    #= The algebraic variables at a point (u, t) of a solution: the discrete, relation and
       event-function values of that time from the event log (after a solve p holds the final
       ones), on a copy of the discrete state; the loops started from the run's roots of that time (_rootsAt).
       The external objects the when-bodies call as the run left them at the log entry's time (`replayObjects!`). =#
    function algebraics(u, p, t, right::Bool = false)
      local a = similar(u, $nAlg)
      local times = first.(p.dlog)
      local k = isempty(times) ? 0 : $(_dlogIndex)(times, t, p.tend[1], right)
      local entry = k == 0 ? vcat(p.d, Float64.(p.rel), p.evv) : p.dlog[k][2]
      local nD = length(p.d); local nR = length(p.rel)
      local d = entry[1:nD]
      local rel = entry[(nD + 1):(nD + nR)] .!= 0
      local evv = entry[(nD + nR + 1):end]
      local q = $(paramsType)(p.values, rel, d, copy(d), copy(p.wcond), copy(p.sampleActive),
                              [false], [false], copy(p.upre), copy(p.ureinit), copy(p.wnew), p.dlog, evv, copy(p.hyst), copy(p.tend),
                              copy(p.homotopy),
                              zeros($nAlg), copy(p.aleft), p.tgradAfter, p.tgradBefore, p.blocks, $(_rootsAt)(p, t, right), p.loopWork, p.pass, p.state,
                              p.model)
      POSITIONS_OBJECTS && k > 0 && replayObjects!(u, q, times[k])
      equations!(similar(u), a, u, q, t, $(NO_CROSSINGS), false)
      return a
    end

    #= One sweep at (u, t): passes with the same pre() until nothing changes (a fired body runs in each: with pre()
       fixed it gives the same values). Returns whether they settled. =#
    function _settle!(du, a, u, p, t, fire::Bool, searched::Bool = false)
      p.upre .= u; p.ureinit .= u
      empty!(p.state.whenMemo)
      local rel0 = copy(p.rel); local d0 = copy(p.d)
      local flips = falses(length(rel0))
      for pass in 1:50
        local relBefore = copy(p.rel); local dBefore = copy(p.d)
        local rBefore = copy(p.ureinit); local wBefore = copy(p.wnew); local eBefore = copy(p.evv)
        equations!(du, a, u, p, t, $(NO_CROSSINGS), true, fire)
        whens!(du, a, u, p, t, $(NO_CROSSINGS), true, fire)
        (relBefore == p.rel && dBefore == p.d && rBefore == p.ureinit && wBefore == p.wnew && eBefore == p.evv) &&
          return true
        pass > 25 && (flips .|= relBefore .!= p.rel)
        pass > 46 && $(_traceSettlePass)(t, pass, (relBefore, dBefore, eBefore, wBefore, rBefore),
                                         (p.rel, p.d, p.evv, p.wnew, p.ureinit))
      end
      #= relations that keep changing (a mixed system that cycles: a bridge's commutation): the combination of their
         values that gives itself back, then the sweeps from it =#
      searched && return false
      $(_searchRelations!)(equations!, whens!, du, a, u, p, t, fire, rel0, d0, flips, RELATION_INSTANCE,
                           RELATION_SLOT) || return false
      return _settle!(du, a, u, p, t, fire, true)
    end

    #= The initialization's solution at (u, t): the relations, the when-conditions and the discrete variables the
       equations compute, with pre() the start values (OpenModelica: pre(v) = v.start for the discrete variables).
       No when fires on its condition there, the conditions' values become pre() of the conditions; with fire
       (initial() true) the bodies of the whens initial() turns true run. reinit() has no effect. =#
    function startSolution!(u, p, t, fire::Bool)
      local du = similar(u); local a = similar(u, $nAlg)
      _settle!(du, a, u, p, t, fire) || return false
      p.ureinit .= u
      p.wcond .= p.wnew
      return true
    end

    #= homotopy() (MLS 3.7.4.4), as OpenModelica's default and the scalarized route (`_homotopyContinuation`): the
       start solved at λ = 0 (the simplified expressions), then from each solution at λ = 1/3 and 2/3, the relations
       and the loops' guesses at each step's solution; the initialization at λ = 1 follows from the last. MSL's
       IdealizedOpAmpLimited (SignalGenerator): from the start values the limiter's loop went to the unstable linear
       solution, all values 0 (OpenModelica: saturated). A step that fails: the initialization from the start values. =#
    function homotopyStart!(u0, p, t)
      local u = copy(u0)
      local before = $(_discreteState)(p); local wcond = copy(p.wcond); local guesses = deepcopy(p.loops)
      try
        for λ in (0.0, 1 / 3, 2 / 3)
          p.homotopy[1] = λ
          length(INIT_ROWS) > 0 && (u = initialize(u, p, t))
          startSolution!(u, p, t, false)
        end
      catch e
        $(_OMBACKEND._fallback)(e, :arrayHomotopyContinuation, impact = :result)
        u = copy(u0)
        p.d .= before[1]; p.dpre .= before[2]; p.rel .= before[3]; p.evv .= before[4]
        p.wcond .= wcond
        for (g, g0) in zip(p.loops, guesses)
          copyto!(g, g0)
        end
      finally
        p.homotopy[1] = 1.0
      end
      return u
    end

    #= Event iteration at (u, t), in sweeps as OpenModelica's updateDiscreteSystem (and the MTK
       path, relationRefresh.jl): within a sweep pre() keeps the values from before it, and the
       relations, equations and fired when bodies are evaluated again until they settle (an
       equation reading a variable a when body just set: ch = change(k)); a sweep that changed
       a discrete value starts the next one from the new values, until one changes nothing. =#
    #= The terminal event (MLS 8.6: terminal() is true at the end of a successful analysis): an event iteration at
       the final time with terminal() true, in which the when-branches with it among their triggers run (MSL's
       ContinuousSignalExtrema, `when {u <= x, u >= x, terminal()}`; an FFT's `when terminal()`); the discrete
       values it gives logged at that time. =#
    function _terminalEvent!(integrator)
      local p = integrator.p
      p.state.terminal = true
      try
        eventIteration!(integrator.u, p, integrator.t, true) ||
          @error string("[events] the terminal event did not settle at t = ", integrator.t)
        _logDiscretes!(p, integrator.t)
      finally
        p.state.terminal = false
      end
      return nothing
    end

    function eventIteration!(u, p, t, fire::Bool, sampleInstant::Bool = false; left::Bool = true)
      local du = similar(u); local a = similar(u, $nAlg)
      #= the algebraic variables' left limit: the equations with the relations and discrete values of before the
         event (`left`; a crossing computed it already, before its relations changed: `affectCrossing!`) =#
      if USES_LEFT && left
        equations!(du, a, u, p, t, $(NO_CROSSINGS), false)
        p.aleft .= a
      end
      for _ in 1:50
        p.dpre .= p.d
        _settle!(du, a, u, p, t, fire) || return false
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
        #= pre() in the next sweep: this one's values =#
        USES_LEFT && (p.aleft .= a)
      end
      return false
    end

    #= The event iteration at an event during the solve: a model that does not settle
       (relations switching back and forth) stops the simulation, as OpenModelica. =#
    function _eventAt!(integrator, sampleInstant::Bool = false; left::Bool = true)
      local p = integrator.p
      #= the delay() arguments just before the event: their left limits (a jump is recorded after it, `_delayJumps!`) =#
      N_DELAYS == 0 || _delayLeft!(integrator)
      #= the state just before the event (a reinit moves it): the asserts' left limit =#
      p.state.leftLimit = (Float64(integrator.t), copy(integrator.u))
      local relBefore::Vector{Bool} = copy(p.rel); local dBefore::Vector{Float64} = copy(p.d)
      if eventIteration!(integrator.u, p, integrator.t, true, sampleInstant; left)
        #= After an event that changed the equations the next step starts small (`_restartStep!`). A relation's
           crossing changed them; a sample instant did where a relation, a discrete variable or the state changed. =#
        (!sampleInstant || relBefore != p.rel || dBefore != p.d || p.state.leftLimit[2] != integrator.u) &&
          $(_restartStep!)(integrator)
        return $(hasTerminate ? :(_terminateReached!(integrator)) : true)
      end
      @error string("[events] the event iteration did not settle at t = ", integrator.t,
                    " (relations switching back and forth: a chattering model); the simulation stops")
      DifferentialEquations.SciMLBase.terminate!(integrator, DifferentialEquations.SciMLBase.ReturnCode.Failure)
      return false
    end

    function conditions!(out, u, t, integrator)
      local p = integrator.p
      local du = similar(u)
      CROSSINGS_ONLY && return crossings!(du, p.abuf, u, p, t, $(CrossingOut)(out), false)
      local a = similar(u, $nAlg)
      equations!(du, a, u, p, t, $(CrossingOut)(out), false)
      whens!(du, a, u, p, t, $(CrossingOut)(out), false, false)
      return nothing
    end

    #= at the start of a solve: the relations' hysteresis for its tolerance and span =#
    function _startHysteresis!(c, u, t, integrator)
      integrator.p.hyst[1] = $(_hysteresis)(integrator)
      return nothing
    end

    #= The loops' roots at the end of an accepted step, logged (_logRoots!). After an event at that time p holds
       the roots of the event iteration's passes (the states before the reinit with the new discrete values): the
       right-hand side at the state the next step starts from gives the roots the run goes on with. =#
    function logStepRoots!(integrator)
      local p = integrator.p
      local t = integrator.t
      if p.state.leftLimit[1] == t
        local u = integrator.u
        equations!(similar(u), similar(u, $nAlg), u, p, t, $(NO_CROSSINGS), false)
      end
      $(_logRoots!)(p, t)
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
       there changes with it (a brake's normal force ramped to 0 at the time event: f <= 0 holds).
       The left limit lies beyond the hysteresis band (2H): a relation of time is located at its band's edge,
       zc = -eps with eps = H*(1 + |t|), and at H = 1e-10 (reltol 1e-6) a left limit 1e-10*(1 + |t|) back put it
       exactly on its threshold, where the exact-zero rule made it true already: the when on it took its rising
       edge before the event and never fired (MSL 4.1 Trapezoid's `time >= (pre(count) + 1)*period +
       startTime`: OpAmps.Differentiator's source stayed in its first period). =#
    #= Whether t is the instant of the sample tick still to come (`sampleStops[2]`). A relation's change there
       belongs to the tick's event (`affectSample!` takes the relations literal at the instant), which follows at this
       time: handled first as an event of its own, the tick read the values of after the change (a clocked `sample(u)`
       at the instant of u's step takes u's left limit, MLS 16.5.1: MSL Clocked BooleanSignals.Sample1). At: within
       1e-12 of the run's time scale (its span, or the time where larger), not of a second: a pulse edge 1e-13 s after
       a tick is no part of the tick (MSL Spice3 Inverter, a run of 1e-11 s: its sources' edges were dropped). Not a
       tick that ran (`sampleStops[1]`): it took the relations at its time, and a threshold an ulp after it is a
       crossing of its own (MSL SawTooth's `time >= (pre(count) + 1)*period + startTime` at 1251 * 25e-6, an ulp after
       the tick of a sample of the same period: dropped, the reset waited for the end of the step after it, 1e-7 s, and
       ChopperBuckBoost_DutyCycle's switching every other period with it). =#
    function _atSampleInstant(p, t, tspan)
      N_SAMPLES > 0 || return false
      local stops = p.state.sampleStops
      local span = abs(tspan[2] - tspan[1])
      local w = 1.0e-12 * (isfinite(span) ? max(span, abs(t)) : 1 + abs(t))
      return length(stops) >= 2 && abs(stops[2] - t) <= w
    end

    function affectCrossing!(integrator, events::AbstractVector{Int8})
      local t = integrator.t
      local p = integrator.p
      if _atSampleInstant(p, t, integrator.sol.prob.tspan)
        DifferentialEquations.derivative_discontinuity!(integrator, false)
        return nothing
      end
      #= the delay() arguments' left limits, before the relations take their new values =#
      N_DELAYS == 0 || _delayLeft!(integrator)
      local tl = max(integrator.tprev, t - max(1e-10, 2 * p.hyst[1]) * (1 + abs(t)))
      local ul = tl < t ? integrator(tl) : copy(integrator.uprev)
      local du = similar(ul); local a = similar(ul, $nAlg)
      local before::Vector{Bool} = copy(p.rel)
      equations!(du, a, ul, p, tl, $(NO_CROSSINGS), true)
      whens!(du, a, ul, p, tl, $(NO_CROSSINGS), true, false)
      p.wcond .= p.wnew
      #= the algebraic variables' left limit at t, with the relations of before the crossing: pre(u) of
         u = if time < T then ... is u before the switch (MSL DemonstrateContinuousSignalExtrema's pulse edges) =#
      if USES_LEFT
        equations!(du, a, integrator.u, p, t, $(NO_CROSSINGS), false)
        p.aleft .= a
      end
      local timed::Bool = false
      for k in eachindex(events)
        (events[k] != 0 && RELATION_SLOT[k] && p.rel[k] == before[k] && p.rel[k] == (events[k] > 0)) && (p.rel[k] = !p.rel[k])
        timed |= events[k] != 0 && TIME_SLOT[k]
      end
      local _hyst = p.hyst[1]
      timed && (p.hyst[1] = 0.0)
      try
        _eventAt!(integrator; left = false)
      finally
        p.hyst[1] = _hyst
      end
      _logDiscretes!(p, t)
      DifferentialEquations.derivative_discontinuity!(integrator, true)
      return nothing
    end

    #= A sample instant: the samples due now are true for the first pass of the event iteration,
       false in the passes after it (OpenModelica's): what reads sample() other than as a when's
       rising edge takes its fall at the instant (`y = not u and pre(u)` of `u = sample(...)`, CDL
       FallingEdge: y rises and falls there; `k = if u then pre(k) + 1 else pre(k)` counts once),
       and the when-conditions take that value too, so the next instant is a rising edge. =#
    function affectSample!(integrator)
      local t = integrator.t; local p = integrator.p; local u = integrator.u
      N_DELAYS == 0 || _delayLeft!(integrator)
      local tspan = integrator.sol.prob.tspan
      local (starts, intervals, stops) = (p.state.sampleStart, p.state.sampleInterval, p.state.sampleStops)
      local next = _nextSampleStop(p, t, tspan)
      for id in eachindex(starts)
        local k = round((t - starts[id]) / intervals[id])
        local (ts, w) = _sampleTick(starts[id], intervals[id], k, tspan)
        #= active at the stop its tick joined (_nextSampleStop): at or after it within its round-off, before the next =#
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
      DifferentialEquations.derivative_discontinuity!(integrator, true)
      #= the stop after this one =#
      stops[1] = t
      stops[2] = next
      next < tspan[2] && DifferentialEquations.add_tstop!(integrator, next)
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

    #= The stop of p's samples after the stop `after` (NaN: the first one, at or after the start time); Inf where no
       tick is left in tspan. Ticks within round-off of each other are one instant, a tick joining the stop before it
       (3 * 0.2 and 2 * 0.3; sample(-100, 0.1) and sample(0, 0.1) near 0): affectSample! activates each sample at the
       stop its tick joined. One stop at a time, each from the samples' ticks next to the last: a table of the ticks
       over tspan took all memory where the period is far below the span (MSL Spice3 Inverter: V_pulse's
       sample(TD, PER) of period 2e-12 s, its problem built over 1 s for the start check, 5e11 ticks). =#
    function _nextSampleStop(p, after, tspan)
      local next = Inf
      local initial = isnan(after)
      local from = initial ? Float64(tspan[1]) : after
      for (st, dt) in zip(p.state.sampleStart, p.state.sampleInterval)
        (isfinite(st) && isfinite(dt)) || error("array model: sample start and interval must be finite")
        dt > 0 || error("array model: sample interval must be positive")
        local k = max(0.0, ceil((from - st) / dt) - 1)
        while true
          local (ts, w) = _sampleTick(st, dt, k, tspan)
          ts > tspan[2] && break
          if initial ? ts >= tspan[1] : ts - after > w
            next = min(next, ts)
            break
          end
          k + 1 > k || error(string("array model: sample interval ", dt, " below the round-off of the time ", from))
          k += 1
        end
      end
      return next
    end

    #= The samples' stops as a callback: the first stop at the start of a solve (a tick at the start time handled
       there), then each from the one before (affectSample!), a solver stop where within tspan. =#
    function _sampleStart!(c, u, t, integrator)
      local stops = integrator.p.state.sampleStops
      local tspan = integrator.sol.prob.tspan
      resize!(stops, 2)
      stops[1] = -Inf
      stops[2] = _nextSampleStop(integrator.p, NaN, tspan)
      if stops[2] == t
        affectSample!(integrator)
      elseif stops[2] < tspan[2]
        DifferentialEquations.add_tstop!(integrator, stops[2])
      end
      return nothing
    end

    _sampleDue(u, t, integrator) = t == integrator.p.state.sampleStops[2]

    function affect!(integrator)
      if _atSampleInstant(integrator.p, integrator.t, integrator.sol.prob.tspan)
        DifferentialEquations.derivative_discontinuity!(integrator, false)
        return nothing
      end
      _eventAt!(integrator)
      _logDiscretes!(integrator.p, integrator.t)
      DifferentialEquations.derivative_discontinuity!(integrator, true)
      return nothing
    end

    #= After a step: whether a relation has another value than the stored one. A crossing
       right after an event (the ball leaving the floor it was reinitialized just below) has
       no sign change the root finder sees; it is caught here, at the end of the step. =#
    function relationsChanged(u, t, integrator)
      local p = integrator.p
      local rel0 = copy(p.rel); local w0 = copy(p.wcond); local e0 = copy(p.evv)
      #= the discrete values too: this evaluation runs the lifted clusters of discrete equations (and computes the
         discrete unknowns) with the relations as they are now; left in place, pre() at the event that follows was
         the value after it (`when sample() then bs := pre(b)` of `b = time >= 0.2` at the tick at 0.2) =#
      local d0 = copy(p.d)
      local du = similar(u)
      if CROSSINGS_ONLY
        crossings!(du, p.abuf, u, p, t, $(NO_CROSSINGS), true)
      else
        local a = similar(u, $nAlg)
        equations!(du, a, u, p, t, $(NO_CROSSINGS), true)
        whens!(du, a, u, p, t, $(NO_CROSSINGS), true, false)
      end
      local changed = rel0 != p.rel || e0 != p.evv
      p.rel .= rel0; p.wcond .= w0; p.evv .= e0; p.d .= d0
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
                           [1.0], zeros($nAlg), zeros($nAlg), zeros(length(U0)), zeros(length(U0)),
                           $(BlockSolver)[$(BlockSolver)(BLOCK_PATTERNS[k], BLOCK_COLORS[k], BLOCK_GUESSES[k], BLOCK_LINEAR[k]) for k in eachindex(BLOCK_PATTERNS)],
                           Matrix{Float64}[copy(G) for G in LOOP_GUESSES],
                           $(LoopWork){Float64}[$(LoopWork){Float64}(size(G, 1)) for G in LOOP_GUESSES],
                           $(InitialPass)(false, zeros(N_INIT_LOOPS), zeros(N_INIT_LOOPS)),
                           $(ArrayRuntimeState)(N_SAMPLES, strings, N_DELAYS), @__MODULE__)
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
      $(isempty(paramAlgBody) ? nothing : :(parameterAlgorithms!(u0, p, tspan[1])))
      USES_HOMOTOPY && (u0 = homotopyStart!(u0, p, tspan[1]))
      $(isempty(clusterPre) ? nothing :
        :(local (uStart, preStart, used, guessStart) =
            (copy(u0), p.dpre[$clusterPre], p.d[$clusterPre], loopGuesses(p))))
      length(INIT_ROWS) > 0 && (u0 = initialize(u0, p, tspan[1]))
      isempty(INIT_REDUNDANT) || checkRedundantInitial(u0, p, tspan[1])
      $(isempty(initAlgBody) ? nothing : :(initialAlgorithms!(u0, p, tspan[1])))
      $(isempty(clusterPre) ? nothing : :(u0 = initialClusters!(uStart, u0, p, tspan[1], preStart, used, guessStart)))
      $(isempty(freeStartStmts) ? nothing : :(freeParameterStarts!(p)))
      N_SAMPLES == 0 || sampleSchedule!(p)
      if N_RELATIONS > 0 || N_WHENS > 0 || N_DISCRETE_UNKNOWNS > 0
        #= the initialization's solution: pre() is the start values, no when fires on its condition =#
        startSolution!(u0, p, tspan[1], false) ||
          error("array model: the relations and discrete values at the start do not settle")
        if USES_INITIAL
          #= when initial(): its body runs in the initialization, then initial() is false again =#
          p.initPhase[1] = true
          startSolution!(u0, p, tspan[1], true) || error("array model: the initial events do not settle")
          p.initPhase[1] = false
          startSolution!(u0, p, tspan[1], false) || error("array model: the initial events do not settle")
        end
        #= the event iteration that follows the initialization: pre() takes the initialized values, and a when whose
           condition that turns true fires (MSL Digital's gates: y = pre(auxiliary_n) leaves its start, the delay's
           change(x) fires and schedules the output, as OpenModelica) =#
        eventIteration!(u0, p, tspan[1], true) || error("array model: the events at the start do not settle")
      end
      $(_checkBlocksSolved)(p)
      _logDiscretes!(p, tspan[1])
      $(_logRoots!)(p, tspan[1])
      $(isempty(initAssertBody) ? nothing : :(checkInitialAsserts(u0, p, tspan[1])))
      invariants!(p.abuf, p)
      #= a domain error or an assert in the right-hand side rejects the step (the solver retries smaller), as in the
         scalar path (MSL DifferenceAmplifier's x^y of a negative x at a trial point) =#
      #= tgrad! wrapped to one type for every array model (the right-hand side is, AutoSpecialize): the model's own
         function type in the problem compiled the solver again for each model =#
      #= the Jacobian in a sparsity pattern by the array code's own differences, a step per state (ColoredJacobian) =#
      local jacPattern = $jacProto
      local rhs = $(StepRejectingRHS)(RHS!)
      local jac = jacPattern === nothing ? nothing :
        OMBackend.CodeGeneration.DiffEqBase.wrapfun_jac_iip($(ColoredJacobian)(rhs, jacPattern),
                                                            (jacPattern, u0, p, Float64(tspan[1])))
      #= with a Jacobian given and no ForwardDiff, DiffEqBase leaves the right-hand side as it is (`promote_f`: its
         wrap path needs `f.jac === nothing`): the model's own function type in the problem compiled the solver again
         for each model (3 s each; the package tests 27 min for 15). Wrapped here then, as DiffEqBase wraps it =#
      local wrapped = jac === nothing ? rhs :
        OMBackend.CodeGeneration.DiffEqBase.wrapfun_jac_iip(rhs, (u0, u0, p, Float64(tspan[1])))
      local f::ODEFunction = ODEFunction(wrapped; jac_prototype = jacPattern, jac = jac,
                                         tgrad = OMBackend.CodeGeneration.DiffEqBase.wrapfun_jac_iip(tgrad!, (u0, u0, p, Float64(tspan[1]))),
                                         sys = $(ArraySystem)(@__MODULE__))
      #= RightRootFind: the event lands just past the root, where the relation has its new value. =#
      local cbs = Any[]
      #= the relations were set literally above; from the solve on they have the hysteresis =#
      N_RELATIONS > 0 && push!(cbs, VectorContinuousCallback(conditions!, affectCrossing!, N_RELATIONS;
                                                             rootfind = DifferentialEquations.SciMLBase.RightRootFind,
                                                             interp_points = $(RELATION_CHECKS),
                                                             initialize = _startHysteresis!),
                                    DiscreteCallback(relationsChanged, affect!; save_positions = (false, false)))
      N_SAMPLES == 0 || push!(cbs, DiscreteCallback(_sampleDue, affectSample!; initialize = _sampleStart!))
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
      #= after the others, its condition at each accepted step: the loops' roots of that time, for the reads after
         the run =#
      (isempty(p.loops) && isempty(p.blocks)) ||
        push!(cbs, DiscreteCallback((u, t, integrator) -> (logStepRoots!(integrator); false), integrator -> nothing;
                                    save_positions = (false, false)))
      if N_DELAYS > 0
        #= delay() (delays.jl, as the scalarized path's callbacks): first the step's end as the solver took it (the
           left limits), then a delayed jump's event; last, after the other callbacks, the values after the events (a
           jump is due again its delay time later) and, at the start, each history's first point =#
        pushfirst!(cbs, DiscreteCallback((u, t, integrator) -> $(delayJumpsDue!)(integrator.p.state.delays, Float64(t)),
                                         affectDelay!; save_positions = (false, false)))
        pushfirst!(cbs, DiscreteCallback((u, t, integrator) -> _delayLeft!(integrator), integrator -> nothing;
                                         save_positions = (false, false)))
        push!(cbs, DiscreteCallback((u, t, integrator) -> _delayJumps!(integrator), integrator -> nothing;
                                    initialize = (c, u, t, integrator) -> _delayStart!(integrator),
                                    save_positions = (false, false)))
      end
      pushfirst!(cbs, DiscreteCallback((u, t, integrator) -> false, integrator -> nothing;
                                       initialize = (c, u, t, integrator) -> $(_restoreStart!)(integrator.p),
                                       save_positions = (false, false)))
      #= the first step, after the other callbacks' initializations: finite (`_finiteFirstStep!`) =#
      push!(cbs, DiscreteCallback((u, t, integrator) -> false, integrator -> nothing;
                                  initialize = (c, u, t, integrator) -> $(_finiteFirstStep!)(integrator),
                                  save_positions = (false, false)))
      #= the terminal event, after the last step (a solve's end: its stop time, a terminate()) =#
      $(usesTerminal ? :(push!(cbs, DiscreteCallback((u, t, integrator) -> false, integrator -> nothing;
                                                     finalize = (c, u, t, integrator) -> _terminalEvent!(integrator),
                                                     save_positions = (false, false)))) : nothing)
      #= one callback type for every model (`_erasedCallbacks`): the solver's callback functions compiled once =#
      local cb = isempty(cbs) ? nothing : OMBackend.CodeGeneration._erasedCallbacks(cbs)
      $(hasTerminate ? quote
          #= a terminate() at the initialization: the problem ends at its start =#
          p.state.terminated === nothing || return (ODEProblem(f, u0, (tspan[1], tspan[1]), p), cb)
        end : nothing)
      return (ODEProblem(f, u0, tspan, p), cb)
    end

    function simulate(tspan = (0.0, 1.0), solver = Tsit5(); parameters::AbstractDict = Dict{String, Any}(),
                      abstol = $(DEFAULT_ABSTOL), kwargs...)
      local (ode, callbacks) = $(Symbol(modelName, "Model"))(tspan; parameters = parameters)
      local prob = $(_problemFor)(ode, solver)
      local initKw = (; $(_initializeKwargs)(prob, kwargs, abstol)..., $(_stepKwargs)(kwargs, tspan, N_RELATIONS)...)
      OMBackend.CodeGeneration._resetRHSError!()
      local sol = $(hasTerminate ?
                    :(prob.p.state.terminated === nothing ?
                        DifferentialEquations.solve(prob, solver; callback = callbacks, abstol = abstol, initKw..., kwargs...) :
                      $(solveStoppedAtStart)(prob, solver, something(prob.p.state.terminated); callback = callbacks,
                                             abstol = abstol, initKw..., kwargs...)) :
                    :(DifferentialEquations.solve(prob, solver; callback = callbacks, abstol = abstol, initKw..., kwargs...)))
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

"""
    _traceSettlePass(t, pass, before, after)

The changes of one of the last passes of an event iteration's sweep that does not settle (`_settle!`), with
OMBACKEND_ARRAY_TRACE: the slots of the relations, discrete values, event-generating functions' values,
when-conditions and reinit values that changed. `before` and `after` hold those five, in that order.
"""
function _traceSettlePass(t::Real, pass::Int, before::NTuple{5, AbstractVector},
                          after::NTuple{5, AbstractVector})::Nothing
  _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE") || return nothing
  local labels::NTuple{5, String} = ("relations", "discretes", "event functions", "when-conditions", "reinit")
  local parts::Vector{String} = String[]
  for (label::String, b::AbstractVector, x::AbstractVector) in zip(labels, before, after)
    local changed::Vector{Int} = findall(b .!= x)
    isempty(changed) || push!(parts, string(label, " ", changed, ": ", b[changed], " -> ", x[changed]))
  end
  println("[arraycode] sweep at t = ", t, ", pass ", pass, ": ", join(parts, "; "))
  return nothing
end

"""
    _searchRelations!(equations!, whens!, du, a, u, p, t, fire, rel0, d0, flips) -> Bool

The solution of a mixed system whose relations cycle in the event iteration's sweeps (`flips`: the relations that
kept changing): the combinations of their values, fewest changes from the values before the sweeps (`rel0`, the
discrete values `d0`) first; each is put in place of the buffered relations, the equations and the lifted clusters
of discrete equations evaluated with them (`forced`: the relations as buffered, the discrete variables from them,
until those stay) and then once with the relations evaluated on that solution. The first combination that gives
itself back is left in place: true. OpenModelica solves a mixed system so (a friction element's `startForward` and
`locked` with the torque that decides them); the scalarized path's `_search!` (discreteClusters.jl). False, the values
before the sweeps restored, where none does or more than 12 relations changed.
Also searched: the relations within their hysteresis band (their literal value differs, `_literalChanges!`): two
relations of one threshold in different units are not both past their bands at an event (MSL OpenTank's port at the
level: `fluidLevel >= height` within its band, `s > 0` with `s = (p - ps) / p_default * (height - fluidLevel)` past
its own; the port's `inFlow` switched every pass, TanksWithOverflow stopped at 2025 s).
Where no combination of those does, again with the other relations of their equation instances (`instance`: per slot
the first slot of its instance; `relation`: the slots that are relations): an if-expression's other conditions, which
did not change in the sweeps. MSL IdealizedOpAmpLimited's `if V0*v_in > vps then vps elseif V0*v_in < vns then vns else
V0*v_in` in SignalGenerator's comparator loop: at the fold the lower limit and the linear branch each gave the other,
the upper limit (the jump to +15 V) is the solution.
"""
function _searchRelations!(@nospecialize(equations!::Function), @nospecialize(whens!::Function), du::AbstractVector, a::AbstractVector, u::AbstractVector,
                           p::ArrayModelParameters, t::Real, fire::Bool, rel0::Vector{Bool}, d0::Vector{Float64},
                           flips::BitVector, instance::Vector{Int},
                           relation::BitVector)::Bool
  local slots::Vector{Int} =
    sort!(union(findall(flips), _literalChanges!(equations!, whens!, du, a, u, p, t, fire, rel0, d0)))
  _searchCombinations!(equations!, whens!, du, a, u, p, t, fire, rel0, d0, slots) && return true
  local instances::Set{Int} = Set{Int}(instance[s] for s in slots)
  local wider::Vector{Int} = filter(s::Int -> relation[s] && instance[s] in instances, eachindex(instance))
  length(wider) > length(slots) || return false
  return _searchCombinations!(equations!, whens!, du, a, u, p, t, fire, rel0, d0, wider)
end

"""
    _searchCombinations!(equations!, whens!, du, a, u, p, t, fire, rel0, d0, slots) -> Bool

The combinations of the relations `slots` for `_searchRelations!`, fewest changes from `rel0` first; the first that
gives itself back left in place (true), else the relations and discrete values `rel0` and `d0` restored (false); none
tried for none or more than 12 slots.
"""
function _searchCombinations!(@nospecialize(equations!::Function), @nospecialize(whens!::Function), du::AbstractVector, a::AbstractVector, u::AbstractVector,
                              p::ArrayModelParameters, t::Real, fire::Bool, rel0::Vector{Bool}, d0::Vector{Float64},
                              slots::Vector{Int})::Bool
  local trace::Bool = _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE")
  trace && println("[arraycode] relation search at t = ", t, ": relations ", slots)
  (isempty(slots) || length(slots) > 12) && return false
  local masks::Vector{Int} = sort!(collect(0:((1 << length(slots)) - 1)); by = count_ones)
  for mask::Int in masks
    p.rel .= rel0
    p.d .= d0
    for (k::Int, slot::Int) in enumerate(slots)
      (mask >> (k - 1)) & 1 == 1 && (p.rel[slot] = !rel0[slot])
    end
    local candidate::Vector{Bool} = copy(p.rel)
    #= the discrete variables of these relations: a loop cut at one reads its last value, so until they stay =#
    local stable::Bool = false
    p.state.forced = true
    try
      for _ in 1:6
        local dBefore::Vector{Float64} = copy(p.d)
        equations!(du, a, u, p, t, NO_CROSSINGS, false, fire)
        whens!(du, a, u, p, t, NO_CROSSINGS, false, false)
        if dBefore == p.d
          stable = true
          break
        end
      end
    finally
      p.state.forced = false
    end
    (stable && all(isfinite, a) && all(isfinite, du)) || continue
    equations!(du, a, u, p, t, NO_CROSSINGS, true, fire)
    whens!(du, a, u, p, t, NO_CROSSINGS, true, fire)
    trace && println("[arraycode]   combination ", candidate[slots], " gives ", p.rel[slots], "; others changed: ",
                     setdiff(findall(p.rel .!= candidate), slots))
    p.rel == candidate && return true
  end
  p.rel .= rel0
  p.d .= d0
  return false
end

"""
    _literalChanges!(equations!, whens!, du, a, u, p, t, fire, rel0, d0) -> Vector{Int}

The relations whose literal value (no hysteresis) at (u, t), with the relations `rel0` and the discrete values `d0`,
differs from theirs: those within their hysteresis band (`_relationAt`). The equations and the lifted clusters of
discrete equations (`whens!`: a Boolean `regular = x >= 6`) evaluated once; the relations, discrete values,
event-generating functions' values, when-conditions and reinit values are restored.
"""
function _literalChanges!(@nospecialize(equations!::Function), @nospecialize(whens!::Function), du::AbstractVector, a::AbstractVector, u::AbstractVector,
                          p::ArrayModelParameters, t::Real, fire::Bool, rel0::Vector{Bool},
                          d0::Vector{Float64})::Vector{Int}
  local hyst::Float64 = p.hyst[1]
  local (evv::Vector{Float64}, wnew::Vector{Bool}, ureinit::Vector{Float64}) =
    (copy(p.evv), copy(p.wnew), copy(p.ureinit))
  p.rel .= rel0
  p.d .= d0
  p.hyst[1] = 0.0
  local changed::Vector{Int} = Int[]
  try
    equations!(du, a, u, p, t, NO_CROSSINGS, true, fire)
    whens!(du, a, u, p, t, NO_CROSSINGS, true, fire)
    changed = findall(p.rel .!= rel0)
  finally
    p.hyst[1] = hyst
    p.rel .= rel0
    p.d .= d0
    p.evv .= evv
    p.wnew .= wnew
    p.ureinit .= ureinit
  end
  return changed
end

"Whether `x` has an operation `op` (`Op`, in the operations' arguments)."
_hasOp(x::IR, op::Symbol)::Bool = x isa Op && (x.op == op || any(a::IR -> _hasOp(a, op), x.args))

"""
    _initAlgPlan(g) -> (order, refresh)

The initial algorithms in an order that runs each after those whose assignments reach what it reads (as written
otherwise, and for a cycle), as the initialization solves them: an initial section reading another arm's target ran
first and read its start (MSL OnDelay's `delaySignal := u` before `u := time >= 0`: false; the MTK code runs it late,
true). An assignment reaches a variable directly, or through the equations that compute it (a connection between two
blocks: MSL Digital's gate reads `x`, equal to the source's `y` by a connect equation; before the source it read the
start). `refresh[k]`: the equations are evaluated before the k-th of the order, as it reads what they compute and they
were not evaluated yet, or an algorithm run since their last evaluation assigned what a variable it reads is computed
from. Not before an algorithm that reads none of it: the equations may read what it assigns (a lifted cluster's
initial arm `b := if time < 0.5 then 3 else 4` before `y = table[b, a]`: index 0 with b's start).
"""
function _initAlgPlan(g::Gen)::Tuple{Vector{AlgClass}, BitVector}
  local algs::Vector{AlgClass} = g.initAlgs
  local n::Int = length(algs)
  n == 0 && return (algs, falses(0))
  local writes::Vector{Set{String}} = Set{String}[_stmtNames!(Set{String}(), a.body, false) for a in algs]
  local reads::Vector{Set{String}} = Set{String}[_stmtNames!(Set{String}(), a.body, true) for a in algs]
  local writeElements::Vector{ElementSets} = ElementSets[_stmtElements(a.body, false) for a in algs]
  local readElements::Vector{ElementSets} = ElementSets[_stmtElements(a.body, true) for a in algs]
  #= what the equations compute: the algebraic variables, the discrete ones that are unknowns of the equations =#
  local computed::Set{String} = Set{String}()
  for (nm::String, info::VarInfo) in g.vars
    if info.kind == :alg || (info.kind == :discrete &&
        any(k::Int -> haskey(g.discUnknown, info.offset + k), 1:(isempty(info.dims) ? 1 : prod(info.dims))))
      push!(computed, nm)
    end
  end
  #= the names of each equation and algorithm, and the equations of each name =#
  local members::Vector{Set{String}} =
    Set{String}[_readNames!(_readNames!(Set{String}(), c.lhs), c.rhs) for c in g.classes]
  for a::AlgClass in g.algs
    push!(members, _stmtNames!(_stmtNames!(Set{String}(), a.body, true), a.body, false))
  end
  local classesOf::Dict{String, Vector{Int}} = Dict{String, Vector{Int}}()
  for (ci::Int, names::Set{String}) in enumerate(members), nm::String in names
    push!(get!(() -> Int[], classesOf, nm), ci)
  end
  #= the computed variables each algorithm's assignments reach through the equations =#
  local downstream::Vector{Set{String}} = Set{String}[]
  for w::Set{String} in writes
    local seen::Set{String} = Set{String}()
    local visited::Set{Int} = Set{Int}()
    local queue::Vector{String} = collect(w)
    while !isempty(queue)
      local nm::String = pop!(queue)
      for ci::Int in get(classesOf, nm, Int[])
        ci in visited && continue
        push!(visited, ci)
        for other::String in members[ci]
          (other in computed && !(other in w) && !(other in seen)) || continue
          push!(seen, other); push!(queue, other)
        end
      end
    end
    push!(downstream, seen)
  end
  #= the algorithms each one waits for: those that assign what it reads, or reach it through the equations. An
     assignment by element where the subscripts are literals: the sections of the instances of a component array
     assign and read the same names, each its own elements, and do not wait for one another (MSL Digital's delay in
     two tristate buffers: as a cycle its `y := y_auxiliary` ran before the sections that assign y_auxiliary) =#
  local assigning::Dict{String, Vector{Int}} = Dict{String, Vector{Int}}()
  local reaching::Dict{String, Vector{Int}} = Dict{String, Vector{Int}}()
  for j::Int in 1:n
    foreach(nm::String -> push!(get!(() -> Int[], assigning, nm), j), writes[j])
    foreach(nm::String -> push!(get!(() -> Int[], reaching, nm), j), downstream[j])
  end
  local waitsFor::Vector{Set{Int}} = Set{Int}[Set{Int}() for _ in 1:n]
  for i::Int in 1:n, nm::String in reads[i]
    for j::Int in get(assigning, nm, Int[])
      (j != i && _elementsMeet(readElements[i][nm], writeElements[j][nm])) && push!(waitsFor[i], j)
    end
    for j::Int in get(reaching, nm, Int[])
      j == i || push!(waitsFor[i], j)
    end
  end
  local done::BitVector = falses(n)
  local order::Vector{AlgClass} = AlgClass[]
  local refresh::BitVector = falses(n)
  #= before the first evaluation of the equations each computed variable holds its start =#
  local stale::Set{String} = copy(computed)
  while length(order) < n
    local ready::Union{Nothing, Int} = findfirst(i::Int -> !done[i] && all(j::Int -> done[j], waitsFor[i]), 1:n)
    local k::Int = ready === nothing ? findfirst(!, done) : ready
    done[k] = true
    push!(order, algs[k])
    if !isdisjoint(reads[k], stale)
      refresh[length(order)] = true
      empty!(stale)
    end
    union!(stale, downstream[k])
  end
  return (order, refresh)
end

"`f(ref)` for each reference the statements read outside pre() (`reads`) or assign."
function _eachStmtRef(f::F, stmts::Vector{Stmt}, reads::Bool)::Nothing where {F <: Function}
  local exp!::Function = (x::IR) -> (reads && _eachRead(f, x); nothing)
  for st in stmts
    if st isa SAssign
      reads ? (exp!(st.value); foreach(exp!, st.target.subs)) : f(st.target)
    elseif st isa SPreStart
      #= pre() of the target: what the equations compute from it follows, as from an assignment =#
      reads ? (exp!(st.value); foreach(exp!, st.target.subs)) : f(st.target)
    elseif st isa Union{SIf, SWhen}
      foreach(b::Tuple{IR, Vector{Stmt}} -> (exp!(b[1]); _eachStmtRef(f, b[2], reads)), st.branches)
    elseif st isa SFor
      _eachStmtRef(f, st.body, reads)
    elseif st isa SWhile
      exp!(st.cond)
      _eachStmtRef(f, st.body, reads)
    elseif st isa SCall
      exp!(st.call)
    elseif st isa SAssert
      exp!(st.cond); exp!(st.msg)
    end
  end
  return nothing
end

"`f(ref)` for each reference `x` reads outside pre() (a reference's subscripts too)."
function _eachRead(f::F, x::IR)::Nothing where {F <: Function}
  if x isa Ref
    f(x)
    foreach(s::IR -> _eachRead(f, s), x.subs)
  elseif x isa Rel
    _eachRead(f, x.lhs); _eachRead(f, x.rhs)
  elseif x isa Op
    foreach(a::IR -> _eachRead(f, a), x.args)
  elseif x isa EvFn
    _eachRead(f, x.arg)
  end
  return nothing
end

"The names the statements read outside pre() (`reads`) or assign, into `out`."
function _stmtNames!(out::Set{String}, stmts::Vector{Stmt}, reads::Bool)::Set{String}
  _eachStmtRef((r::Ref) -> push!(out, r.name), stmts, reads)
  return out
end

"The names `x` reads outside pre() (a reference's subscripts too), into `out`."
function _readNames!(out::Set{String}, x::IR)::Set{String}
  _eachRead((r::Ref) -> push!(out, r.name), x)
  return out
end

"The elements of one name: those of literal subscripts, or `nothing`, the whole array (another subscript, none)."
const Elements = Union{Nothing, Set{Vector{Int}}}
"The elements by name."
const ElementSets = Dict{String, Elements}

"The elements the statements read outside pre() (`reads`) or assign, by name."
function _stmtElements(stmts::Vector{Stmt}, reads::Bool)::ElementSets
  local out::ElementSets = ElementSets()
  _eachStmtRef(stmts, reads) do r::Ref
    if isempty(r.subs) || !all(s::IR -> s isa Lit && s.v isa Integer && !(s.v isa Bool), r.subs)
      out[r.name] = nothing
    else
      local known::Elements = get!(() -> Set{Vector{Int}}(), out, r.name)
      known === nothing || push!(known, Int[Int((s::Lit).v) for s in r.subs])
    end
  end
  return out
end

"Whether the elements `a` and `b` of one name have one in common."
_elementsMeet(a::Elements, b::Elements)::Bool = a === nothing || b === nothing || !isdisjoint(a, b)

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
The external objects the code reads (`Gen.objects`) as the module's constants `OBJECT_k`: each made once, when the
module is evaluated, by its constructor's call on the translated parameter values (`PARAMS`), and registered under the
model with its destructor (`CodeGeneration._externalObjectRef`); the objects of the model's build before this one are
freed first. NotSupported for a constructor call without code.
"""
function _objectDefinitions(g::Gen, modelName::String)::Vector{Any}
  isempty(g.objects) && return Any[]
  local model::String = _OMBACKEND.canonicalName(modelName)
  local defs::Vector{Any} = Any[:($(_OMBACKEND.CodeGeneration.freeExternalObjects!)($model))]
  for (k::Int, (nm::String, value::IR, destructor::String)) in enumerate(g.objects)
    local rule::Any = _bindingCode(g, ParameterBinding(nm, _bindingDeps(g, g.vars[nm].var), :scalar, IR[value]))
    rule === nothing && ns("the constructor of the external object $(nm)")
    push!(defs, :(const $(Symbol("OBJECT_", k)) =
                    $(_OMBACKEND.CodeGeneration._externalObjectRef)($(QuoteNode(Symbol(nm))), ($rule)(PARAMS),
                                                                    $(QuoteNode(Symbol(destructor))), $model)))
  end
  return defs
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
