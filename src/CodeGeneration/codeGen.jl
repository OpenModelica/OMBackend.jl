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
  The when-callback emitter the MTK path uses, and expToJuliaExp (SimCode expressions over the
  solver's arrays) for the structural callbacks and the sample() intervals. The direct
  DifferentialEquations.jl backend is DECodeGeneration.jl.

  Author: John Tinnerholm
=#

"""
  Contains the headerstring defining the OpenModelica copyright notice.
"""
const HEADER_STRING ="
  $(copyrightString())"

"""
An elsewhen arm as a when-equation of its own, `nothing` for none.
"""
_elsewhenInner(::Nothing) = nothing
_elsewhenInner(p::SC.WHEN_STMTS)::SC.WHEN_EQUATION = SC.WHEN_EQUATION(p, SC.NO_SOURCE, SC.EQ_ATTR_DEFAULT)

_elsewhenCondition(arm::SC.WHEN_EQUATION)::SC.Exp = arm.whenEquation.condition
_elsewhenStmtLst(arm::SC.WHEN_EQUATION)::Vector{SC.Statement} = arm.whenEquation.body

"""
The names a when body reads (getRHSVariables of each statement).
"""
_bodyReads(body::Vector{SC.Statement})::Vector{String} = reduce(vcat, (getRHSVariables(s) for s in body); init = String[])

"""
The names a when body assigns.
"""
function _bodyWrites(body::Vector{SC.Statement})::OrderedSet{String}
  local written = OrderedSet{String}()
  for s in body
    s isa SC.ASSIGN && union!(written, SC.allCrefNames(s.left))
  end
  return written
end

"""
The names a condition reads OUTSIDE any `change(...)` or `pre(...)`: the level-valued
Booleans of the condition (`when boolVar then`), for which the discrete callback keeps an
edge latch.
"""
function _collectBareCrefStrings(cond::SC.Exp)::OrderedSet{String}
  local out = OrderedSet{String}()
  local walk = function (e::SC.Exp, insideObs::Bool)
    if e isa Union{SC.EXP_CREF, SC.DER}
      insideObs || push!(out, SC.crefName(e.cref))
    elseif e isa SC.CALL
      local nested = insideObs || e.name in (:change, :pre)
      foreach(a -> walk(a, nested), e.args)
    elseif e isa Union{SC.BINARY, SC.LBINARY, SC.RELATION, SC.UNARY, SC.LUNARY, SC.IFEXP}
      foreach(c -> walk(c, insideObs), SC.children(e))
    end
    return
  end
  walk(cond, false)
  return out
end

#= To keep track of generated callbacks. =#
let CALLBACKS = 0
  global function ADD_CALLBACK()
    CALLBACKS += 1
    return CALLBACKS
  end
  global function RESET_CALLBACKS()
    CALLBACKS = 0
    return CALLBACKS
  end
  global function COUNT_CALLBACKS()
    return CALLBACKS
  end
end

"""
    createCallbackCode(modelName, simCode; generateSaveFunction = true)

The model's callback set of its when-equations (those that are no discrete clusters or
self-scheduling time events, emitted elsewhere), with a saving callback unless
`generateSaveFunction` is false.
"""
function createCallbackCode(modelName::String, simCode::SC.SIM_CODE; generateSaveFunction::Bool = true)
  #= Synthesised discrete-Boolean whens (`change(rel)` conditions) are discrete clusters of the
     event iteration (emitDiscreteClusters) or, with OMBACKEND_DISCRETE_PRE_MEMORY=false, MTK
     SymbolicContinuousCallbacks (createDiscreteBoolWhenEvents); not also in this CallbackSet
     (which cannot read MTK observed variables). =#
  local legacyWhens = filter(w -> _extractChangeRelations(w.whenEquation.condition, simCode) === nothing &&
                                  isempty(_selfSchedulingTimeRels(w)),
                             simCode.whenEquations)
  local WHEN_EQUATIONS = createEquations(legacyWhens, simCode)
  local SAVE_FUNCTION = generateSaveFunction ? createSaveFunction(modelName) : nothing
  #= saved_values_<model> = SavedValues(...) only with the save function: SavedValues lives in
     DiffEqCallbacks, a transitive dependency; its unconditional emission was an UndefVarError
     at simulate time in MTK mode. =#
  local SAVED_VALUES_DECL = generateSaveFunction ?
    :($(Symbol("saved_values_$(modelName)")) = SavedValues(Float64, Tuple{Float64,Array})) : nothing
  quote
    $(SAVED_VALUES_DECL)
    function $(Symbol("$(modelName)CallbackSet"))(aux)
      #= These are the locations of the parameters and auxiliary real variables respectively =#
      local p = aux[1]
      local reals = aux[2]
      local reducedSystem = aux[3]
      $(LineNumberNode((@__LINE__), "WHEN EQUATIONS"))
      $(WHEN_EQUATIONS...)
      $(LineNumberNode((@__LINE__), "IF EQUATIONS"))
      $(SAVE_FUNCTION)
      return $(Expr(:call, :CallbackSet, returnCallbackSet()...))
    end
  end
end

"""
  Creates the save-callback.
  saved_values_\$(modelName) is provided
  as a shared global for the specific model under compilation.
"""
function createSaveFunction(modelName::String)::Expr
  ADD_CALLBACK()
  local callbacks = COUNT_CALLBACKS()
  local cbSym = Symbol("cb$(callbacks)")
  return quote
    savingFunction(u, t, integrator) = let
      (t, deepcopy(integrator.p))
    end
    $cbSym = SavingCallback(savingFunction, $(Symbol("saved_values_$(modelName)")))
  end
end

"""
  Returns the argument array for the callback set.
"""
returnCallbackSet()::Vector{Symbol} = Symbol[Symbol("cb", t) for t in 1:COUNT_CALLBACKS()]

"The callbacks of the when-equations."
createEquations(equations::Vector{SC.WHEN_EQUATION}, simCode::SC.SIM_CODE)::Vector{Expr} =
  Expr[eqToJulia(eq, simCode, i) for (i, eq) in enumerate(equations)]

"""
The lookups of the variables name[1], name[2], ... (none when name[1] is not a variable).
"""
function _scalarizedElementLookups(name::String, simCode::SC.SIM_CODE)::Vector{Any}
  local elems = Any[]
  local key = string(name, "[1]")
  while haskey(simCode.stringToSimVarHT, key)
    push!(elems, getIdxForLookupMTK(key, simCode))
    key = string(name, "[", length(elems) + 1, "]")
  end
  return elems
end

#= The names the when callbacks bind themselves: a variable binding of the same name would
   replace them. =#
const _WHEN_CALLBACK_LOCALS = ("x", "p", "t", "integrator", "lookuptableStates", "lookuptableParams")

"""
`name = <state/param lookup>` bindings for the names a when callback reads, one per name. A
name the table does not have is an inlined constant (a logic ResetMap[i] element) the
lowering emits as a literal, and gets none (a state/param index for it would KeyError),
unless it is a whole array of variables: then the vector of their values (MSL
GenerateRandomNumbers' when reads pre(state64), a discrete Integer[2]). A String or a data
structure (an external object: Buildings' borehole ExtendableArray) is a module-level
binding (createDataStructureAssignments), never a state or parameter slot.
"""
function _whenLookupBindings(names::Vector{String}, simCode::SC.SIM_CODE)::Vector{Expr}
  local out = Expr[]
  for name in unique(names)
    local entry = get(simCode.stringToSimVarHT, name, nothing)
    if entry === nothing
      local elems = _scalarizedElementLookups(name, simCode)
      isempty(elems) && continue
      name in _WHEN_CALLBACK_LOCALS &&
        OMBackend.unsupported("a when reading the whole array $(name) (a name the event callback uses itself)", name)
      push!(out, Expr(:(=), Symbol(name), Expr(:vect, elems...)))
      continue
    end
    OMBackend.envSwitch("OMBACKEND_WHEN_STRING_SKIP") && last(entry).varKind isa SC.STRING && continue
    last(entry).varKind isa SC.DATA_STRUCTURE && continue
    push!(out, Expr(:(=), Symbol(name), getIdxForLookupMTK(name, simCode)))
  end
  return out
end

_isTimeCref(e::SC.Exp)::Bool = e isa SC.EXP_CREF && e.cref.sym === :time && isempty(e.cref.subs)

"""
The constant threshold of a `time <relop> c` relation, or nothing.
"""
function _timeThreshold(rel::SC.Exp, simCode::SC.SIM_CODE)::Union{Nothing, Vector{Float64}}
  (rel isa SC.RELATION && rel.op in (SC.OP_LESS, SC.OP_LESSEQ, SC.OP_GREATER, SC.OP_GREATEREQ)) || return nothing
  local thr = _isTimeCref(rel.exp1) ? rel.exp2 : _isTimeCref(rel.exp2) ? rel.exp1 : nothing
  thr === nothing && return nothing
  local v = OMBackend._tryOr(() -> SC.tryEvalNumeric(thr, simCode), nothing, :timeThreshold)
  return v === nothing ? nothing : Float64[v]
end

"""
The threshold of a `time >= thr` (or mirrored `thr <= time`) relation that reads a runtime
discrete variable, or nothing: a runtime-scheduled time event (the threshold is known once
the assigning when fires): no PresetTimeCallback, and a ContinuousCallback is unsafe
(simultaneous crossings of several such callbacks are tie-broken to a single applied
affect).
"""
function _discreteTimeEventThreshold(cond::SC.Exp, simCode::SC.SIM_CODE)::Union{Nothing, SC.Exp}
  cond isa SC.RELATION || return nothing
  local thr = (_isTimeCref(cond.exp1) && cond.op in (SC.OP_GREATEREQ, SC.OP_GREATER)) ? cond.exp2 :
              (_isTimeCref(cond.exp2) && cond.op in (SC.OP_LESSEQ, SC.OP_LESS)) ? cond.exp1 : nothing
  thr === nothing && return nothing
  local hasDiscrete = false
  for k in SC.allCrefNames(thr)
    k == "time" && return nothing
    local entry = get(simCode.stringToSimVarHT, k, nothing)
    entry === nothing && return nothing
    if SC.isDiscrete(last(entry))
      hasDiscrete = true
    elseif !SC.isParameter(last(entry))
      return nothing
    end
  end
  return hasDiscrete ? thr : nothing
end

"""
`change(p)` of a parameter or a literal never fires: no event, and it does not disqualify a
pure time-threshold chain.
"""
function _isConstChangeArg(a::SC.Exp, simCode::SC.SIM_CODE)::Bool
  a isa Union{SC.RCONST, SC.ICONST, SC.BCONST} && return true
  a isa SC.EXP_CREF || return false
  local entry = get(simCode.stringToSimVarHT, SC.crefName(a.cref), nothing)
  return entry !== nothing && SC.isParameter(last(entry))
end

"""
A synthesized table/time when-condition: an or-chain whose every leaf is `change(time
<relop> const)` (a time threshold) or `change(param)` (never fires). The threshold times (may
be none), or nothing for a condition with another trigger. Such whens fire AT the thresholds
(a PresetTimeCallback): a ContinuousCallback rootfinding on the spiky change() value never
detects the crossings (Digital.Sources.Table, time-driven sources).
"""
function _collectTimeThresholds(cond::SC.Exp, simCode::SC.SIM_CODE)::Union{Nothing, Vector{Float64}}
  if cond isa SC.LBINARY && cond.op === SC.OP_OR
    local l = _collectTimeThresholds(cond.exp1, simCode)
    local r = _collectTimeThresholds(cond.exp2, simCode)
    return (l === nothing || r === nothing) ? nothing : vcat(l, r)
  elseif SC.isBuiltin(cond, :change) && length(cond.args) == 1
    local thr = _timeThreshold(only(cond.args), simCode)
    thr === nothing || return thr
    return _isConstChangeArg(only(cond.args), simCode) ? Float64[] : nothing
  end
  return nothing
end

"""
The lookup tables of a callback, built at its first call (`cache` a Ref of them).
"""
function _lookupTables(cache::Symbol)::Expr
  quote
    local lookuptableStates
    local lookuptableParams
    if $(cache)[] === nothing
      local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
      local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
      lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
      lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
      $(cache)[] = (lookuptableStates, lookuptableParams)
    else
      local cached = $(cache)[]
      lookuptableStates = cached[1]
      lookuptableParams = cached[2]
    end
  end
end

"""
An `elsewhen time >= thr` arm (thr reading a runtime discrete) as an edge-guarded
DiscreteCallback: ContinuousCallbacks are unsafe here, when several such arms cross zero at
the same instant the integrator applies only one and the rest never re-fire. `ewRefSym`
names a Ref shared with the parent when-branch holding the last consumed threshold: the
parent consumes the threshold when it fires at or past it (elsewhen exclusivity), and this
callback consumes it on firing so a level-true condition stays edge-only.
"""
function _emitElsewhenThresholdTimeWhen(elseArm::SC.WHEN_EQUATION, simCode::SC.SIM_CODE, ewRefSym::Symbol, thr::SC.Exp)::Expr
  ADD_CALLBACK()
  local callbacks = COUNT_CALLBACKS()
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(_elsewhenStmtLst(elseArm), simCode)
  end
  local thrCrefs = SC.allCrefNames(thr)
  local affBindCrefs = vcat(_bodyReads(_elsewhenStmtLst(elseArm)), thrCrefs)
  quote
    let _condCache = Ref{Any}(nothing), _affCache = Ref{Any}(nothing)
      global $(Symbol("condition$(callbacks)"))
      #= `_follow` as a discrete when's (no change()/edge() memory here). =#
      $(Symbol("condition$(callbacks)")) = (x, t, integrator, _follow::Bool = true) -> begin
        $(_lookupTables(:_condCache))
        $(_whenLookupBindings(thrCrefs, simCode)...)
        local _thr = Float64($(expToJuliaExpMTK(thr, simCode)))
        t >= _thr && _thr != $(ewRefSym)[]
      end
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator, $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)) -> begin
        local t = integrator.t
        local x = integrator.u
        @debug "[CB-EW$($(callbacks)) affect] firing" t=integrator.t
        $(_lookupTables(:_affCache))
        $(_whenLookupBindings(affBindCrefs, simCode)...)
        $(ewRefSym)[] = Float64($(expToJuliaExpMTK(thr, simCode)))
        $(whenStmts...)
        OMBackend.CodeGeneration._autoDtReset!(integrator)
        add_tstop!(integrator, integrator.t + 1E-12)
      end
    end
    #= A discrete when, run after its parent's (in the event iteration where the model has
       buffered relations). =#
    $(Symbol("cb$(callbacks)")) = OMBackend.CodeGeneration.discreteWhenCallback($(Symbol("condition$(callbacks)")),
                                                                              $(Symbol("affect$(callbacks)!")))
  end
end

"""
A PresetTimeCallback for a table/time when: the (time-dependent) body at each threshold, so
a stepped output (a Digital Table) lands on its sample times.
"""
function _emitPresetTimeWhen(eq::SC.WHEN_EQUATION, simCode::SC.SIM_CODE, callbacks::Int, thresholds::Vector{Float64})::Expr
  local body = eq.whenEquation.body
  #= pre(v) from the state before the instant (instantPre): an earlier statement's new value is not pre(v). =#
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(body, simCode)
  end
  local times = sort(unique(filter(>(0.0), thresholds)))
  quote
    let _affCache = Ref{Any}(nothing)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
        local t = integrator.t
        local x = integrator.u
        $(_lookupTables(:_affCache))
        $(_whenLookupBindings(_bodyReads(body), simCode)...)
        $(whenStmts...)
      end
    end
    $(Symbol("cb$(callbacks)")) = PresetTimeCallback($(times), $(Symbol("affect$(callbacks)!")))
  end
end

_isPreCref(e::SC.Exp)::Bool = e isa SC.PRE || SC.isBuiltin(e, :pre)

"""
`(time - S)/P` (or `time/P`): (S, P) as numbers, or nothing.
"""
function _timeOffsetOverPeriod(e::SC.Exp, simCode::SC.SIM_CODE)::Union{Nothing, Tuple{Float64, Float64}}
  (e isa SC.BINARY && e.op === SC.OP_DIV) || return nothing
  local p = OMBackend._tryOr(() -> SC.tryEvalNumeric(e.exp2, simCode), nothing, :timePeriod)
  p === nothing && return nothing
  local num = e.exp1
  local s = if num isa SC.BINARY && num.op === SC.OP_SUB
    _isTimeCref(num.exp1) ? OMBackend._tryOr(() -> SC.tryEvalNumeric(num.exp2, simCode), nothing, :timeShift) : nothing
  else
    _isTimeCref(num) ? 0.0 : nothing
  end
  s === nothing && return nothing
  return (Float64(s), Float64(p))
end

"""
MSL 4.1's Pulse: `time >= (pre(count) + 1)*period + startTime`, a time event (4.0 had
`integer((time - startTime)/period) > pre(count)`). The threshold, affine in pre(count), at
counts 0, 1 and 2 gives (startTime, period), or nothing; the when must count it up by one
(`count = pre(count) + 1`: `t = pre(t) + dt` under `time >= pre(t)` is no such clock). As a
generic when the pulse lost the end of most periods: the relation `time < T_start +
T_width` jumped with T_start (the 4.0 path refreshes it, _collectIfCondRefresh).
"""
function _affinePreThreshold(thr::SC.Exp, simCode::SC.SIM_CODE, stmts::Vector{SC.Statement})::Union{Nothing, Tuple{Float64, Float64}}
  local name = Ref("")
  local f = Float64[]
  for k in 0:2
    local s = _substitutePre(thr, Float64(k), name)
    s === nothing && return nothing
    local v = OMBackend._tryOr(() -> SC.tryEvalNumeric(s, simCode), nothing, :pulseThreshold)
    v === nothing && return nothing
    push!(f, v)
  end
  (isempty(name[]) || !_countsUpByOne(stmts, name[], simCode)) && return nothing
  local period = f[2] - f[1]
  (period > 0 && isapprox(f[3] - f[2], period; rtol = 1e-12)) || return nothing
  return (f[1] - period, period)
end

"""
Whether the statements assign `name = pre(name) + 1`.
"""
function _countsUpByOne(stmts::Vector{SC.Statement}, name::String, simCode::SC.SIM_CODE)::Bool
  for st in stmts
    (st isa SC.ASSIGN && SC.crefNameOf(st.left) == name) || continue
    local values = Float64[]
    for k in (0.0, 5.0)
      local own = Ref("")
      local s = _substitutePre(st.right, k, own)
      (s === nothing || own[] != name) && return false
      local v = OMBackend._tryOr(() -> SC.tryEvalNumeric(s, simCode), nothing, :pulseCount)
      v === nothing && return false
      push!(values, v)
    end
    return values == [1.0, 6.0]
  end
  return false
end

"""
`e` with its pre(v), all of one v (its name into `name`), replaced by the literal k; nothing
for an expression other than arithmetic of references and literals.
"""
function _substitutePre(e::SC.Exp, k::Float64, name::Base.RefValue{String})::Union{Nothing, SC.Exp}
  if e isa SC.PRE
    local nm = SC.crefName(e.cref)
    (isempty(name[]) || name[] == nm) || return nothing
    name[] = nm
    return SC.RCONST(k)
  elseif e isa SC.BINARY
    local a = _substitutePre(e.exp1, k, name)
    local b = a === nothing ? nothing : _substitutePre(e.exp2, k, name)
    return b === nothing ? nothing : SC.BINARY(a, e.op, b)
  elseif e isa SC.UNARY
    local a = _substitutePre(e.exp, k, name)
    return a === nothing ? nothing : SC.UNARY(e.op, a)
  elseif e isa SC.CAST
    local a = _substitutePre(e.exp, k, name)
    return a === nothing ? nothing : SC.CAST(e.ty, a)
  end
  return e isa Union{SC.EXP_CREF, SC.RCONST, SC.ICONST} ? e : nothing
end

"""
The Modelica Source.Pulse / SignalSource periodic when-condition
`integer((time - startTime)/period) <relop> pre(counter)` (or MSL 4.1's
`_affinePreThreshold`): (startTime, period), or nothing. A periodic clock: it fires AT
t = startTime + n*period (a PeriodicCallback). A ContinuousCallback rootfinding on the
staircase `integer(...) > pre(count)`, a piecewise-constant 0/1, cannot reliably catch it:
the pulse counter and T_start froze (Blocks.Sources.Pulse, the machines driven by it).
"""
function _pulsePeriodicSpec(cond::SC.Exp, simCode::SC.SIM_CODE, stmts::Vector{SC.Statement})::Union{Nothing, Tuple{Float64, Float64}}
  (cond isa SC.RELATION && cond.op in (SC.OP_GREATER, SC.OP_GREATEREQ)) || return nothing
  _isTimeCref(cond.exp1) && return _affinePreThreshold(cond.exp2, simCode, stmts)
  _isPreCref(cond.exp2) || return nothing
  (SC.isBuiltin(cond.exp1, :integer) && length(cond.exp1.args) == 1) || return nothing
  return _timeOffsetOverPeriod(only(cond.exp1.args), simCode)
end

"""
An if-condition that reads a discrete a periodic when updates (the Pulse `time < T_start +
T_width`, T_start set at each period boundary) cannot be refreshed by its own MTK continuous
callback: when T_start jumps the zero-crossing expression jumps across 0 with no smooth
crossing for the rootfinder to catch. For each if-condition that reads a variable the when
body writes, the assignment that re-derives its `ifCondNI` discrete parameter from the
(now-current) condition value (the naming of createIfEquations: `ifCond<sortIndex><branch>`).
"""
function _collectIfCondRefresh(writtenLHS::OrderedSet{String}, simCode::SC.SIM_CODE)::Tuple{Vector{String}, Vector{Expr}}
  local refreshCrefs = String[]
  local assigns = Expr[]
  isempty(simCode.ifEquations) && return (refreshCrefs, assigns)
  for (identifier, ifEq) in enumerate(_sortedIfEquations(simCode))
    for (i, branch) in enumerate(ifEq.branches)
      branch.identifier == -1 && continue
      local cCrefs = SC.allCrefNames(branch.condition)
      any(c -> c in writtenLHS, cCrefs) || continue
      append!(refreshCrefs, cCrefs)
      local nameSym = Symbol("ifCond$(identifier)$(i)")
      push!(assigns, quote
        let _pidx = get(lookuptableParams, $(QuoteNode(nameSym)), nothing)
          if _pidx !== nothing
            integrator.p[_pidx] = ($(expToJuliaBoolMTK(branch.condition, simCode)) ? 1.0 : 0.0)
          end
        end
      end)
    end
  end
  return (refreshCrefs, assigns)
end

"""
Companion to _collectIfCondRefresh for discrete-Boolean whens: one whose condition reads a
discrete the periodic body writes (BooleanPulse `y = time >= pulseStart and time <
pulseStart + Twidth`, pulseStart re-sampled each period) is lowered to MTK continuous
callbacks that cannot catch the discontinuous threshold jump. Its body runs again in the
periodic callback, from the now-current threshold. A discrete cluster needs none: the event
iteration after the step sees its relation on the written discrete flip (exact, no
hysteresis) and solves the cluster again with the algebraic unknowns. The refresh names
(bound again from the updated state) and the body statements.
"""
function _collectDiscreteBoolWhenRefresh(writtenLHS::OrderedSet{String}, simCode::SC.SIM_CODE)::Tuple{Vector{String}, Vector{Expr}}
  local refreshCrefs = String[]
  local refreshStmts = Expr[]
  local clusters = _usesDiscreteClusters(simCode)
  for weq in simCode.whenEquations
    _extractChangeRelations(weq.whenEquation.condition, simCode) === nothing && continue
    clusters && _gatherClusterAssigns(weq, simCode) !== nothing && continue
    local condCrefs = SC.allCrefNames(weq.whenEquation.condition)
    any(c -> c in writtenLHS, condCrefs) || continue
    #= pre(v) from the periodic affect's snapshot (the values before the tick). =#
    append!(refreshStmts, Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
      createWhenStatementsMTK(weq.whenEquation.body, simCode)
    end)
    append!(refreshCrefs, condCrefs)
    for st in weq.whenEquation.body
      st isa SC.ASSIGN && append!(refreshCrefs, SC.allCrefNames(st.right))
    end
  end
  return (refreshCrefs, refreshStmts)
end

"""
A PeriodicCallback for a Pulse-style periodic when: the body at the
`integer((time-startTime)/period)` increments, t = startTime + n*period for the n in
(tspan[1], stopTime]. `_firstEdge` is the first such instant assuming tspan[1] = 0 (Modelica's
`integer` is floor), in (0, period]: a valid non-negative phase. A bare `phase =
mod(startTime, period)` with `initial_affect = false` dropped the first edge whenever
startTime < 0 (-0.035), shifting the whole pulse train by one period.
"""
function _emitPulsePeriodicWhen(eq::SC.WHEN_EQUATION, simCode::SC.SIM_CODE, callbacks::Int, startTime::Float64, period::Float64)::Expr
  local body = eq.whenEquation.body
  local _firstEdge = startTime + (floor(-startTime / period) + 1.0) * period
  #= pre(v) from the state before the instant (instantPre), as the periodic sample() path. =#
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(body, simCode)
  end
  local (refreshCrefs, refreshAssigns) = _collectIfCondRefresh(_bodyWrites(body), simCode)
  quote
    let _affCache = Ref{Any}(nothing)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        OMBackend.CodeGeneration._isPeriodicTick(integrator, $(_firstEdge), $(period)) || return nothing
        local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
        local t = integrator.t
        local x = integrator.u
        $(_lookupTables(:_affCache))
        $(_whenLookupBindings(_bodyReads(body), simCode)...)
        $(whenStmts...)
        #= Re-derive if-conditions that read the discretes just written. =#
        $(_whenLookupBindings(refreshCrefs, simCode)...)
        $(refreshAssigns...)
      end
    end
    #= Fire AT the first edge (phase in (0, period], non-negative) then every period. t0 is never
       an edge: the when fires only when the integer ratio increases, and an initial_affect body
       run would overwrite the init-algorithm phase (T_start := 0) of negative-startTime sources. =#
    $(Symbol("cb$(callbacks)")) = PeriodicCallback($(Symbol("affect$(callbacks)!")), $(period);
                                                   phase = $(_firstEdge), initial_affect = false,
                                                   final_affect = true)
  end
end

const _RELATION_WHEN_EXCLUDED_CALLS = (:edge, :change, :sample, :initial, :terminal, :der, :pre,
                                       :delay, :noEvent, :smooth, :reinit)

"""
A when condition that is one relation `a op b` (op one of < <= > >=) whose operands call
none of the operators above (nor a delay, pre(), der()): (a, b, isLess, strict), or
nothing.
"""
function _singleRelationWhen(cond::SC.Exp)::Union{Nothing, Tuple{SC.Exp, SC.Exp, Bool, Bool}}
  (cond isa SC.RELATION && cond.op in (SC.OP_LESS, SC.OP_LESSEQ, SC.OP_GREATER, SC.OP_GREATEREQ)) || return nothing
  local excluded = e::SC.Exp -> e isa Union{SC.PRE, SC.DER} || _isDelayCall(e) ||
    (e isa SC.CALL && e.kind === SC.CALL_BUILTIN && e.name in _RELATION_WHEN_EXCLUDED_CALLS)
  (SC.anyExp(excluded, cond.exp1) || SC.anyExp(excluded, cond.exp2)) && return nothing
  return (cond.exp1, cond.exp2, cond.op in (SC.OP_LESS, SC.OP_LESSEQ), cond.op in (SC.OP_LESS, SC.OP_GREATER))
end

"""
A when-equation on one relation (relationRefresh.jl relationWhenCallback): the relation is
buffered, set literally at the start, with a hysteresis, and the body runs only when it
becomes true. `zc` is `a - b` for < and <=, `b - a` for > and >=, so the relation is true
when zc < 0 (or <= 0).
"""
function _emitRelationWhen(eq::SC.WHEN_EQUATION, simCode::SC.SIM_CODE, callbacks::Int, rel::Tuple{SC.Exp, SC.Exp, Bool, Bool})::Expr
  local body = eq.whenEquation.body
  local (e1, e2, isLess, strict) = rel
  local zc = isLess ? SC.BINARY(e1, SC.OP_SUB, e2) : SC.BINARY(e2, SC.OP_SUB, e1)
  local names = String[]
  for n in SC.allCrefNames(SC.BINARY(e1, SC.OP_SUB, e2))
    (n == "time" || n in names) && continue
    local entry = get(simCode.stringToSimVarHT, n, nothing)
    (entry === nothing || last(entry).varKind isa SC.STRING) && continue
    push!(names, n)
  end
  local args = Symbol[Symbol(n) for n in names]
  #= The body runs in the event iteration: pre(v) from the sweep's snapshot. =#
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(body, simCode)
  end
  quote
    $(Symbol("cb$(callbacks)")) = let _affCache = Ref{Any}(nothing)
      local _eval = (t, $(args...)) -> (Float64($(expToJuliaExpMTK(zc, simCode))),
                                        1.0 + max(abs(Float64($(expToJuliaExpMTK(e1, simCode)))),
                                                  abs(Float64($(expToJuliaExpMTK(e2, simCode))))))
      local _body! = (integrator, $(MTK_CodeGenerationUtil.PRE_SNAPSHOT)) -> begin
        local t = integrator.t
        local x = integrator.u
        $(_lookupTables(:_affCache))
        $(_whenLookupBindings(_bodyReads(body), simCode)...)
        $(whenStmts...)
        nothing
      end
      OMBackend.CodeGeneration.relationWhenCallback($(names), _eval, $(strict), _body!)
    end
  end
end

"""
Whether the time is a tick of a PeriodicCallback, t0 + phase + k*period. With final_affect
the callback also runs when the integration ends: a tick there runs (omc samples at the stop
time too: n(1) = 6 for ticks at 0.5, ..., 1.0; the callback left out the final time), any
other end not.
"""
function _isPeriodicTick(integrator, phase::Real, period::Real)::Bool
  local n = (integrator.t - first(integrator.sol.prob.tspan) - phase) / period
  return n >= -1e-9 && abs(n - round(n)) <= 1e-9 * max(1.0, abs(n))
end

"""
Whether `t` is a tick of sample(start, interval): start + k*interval, k = 0, 1, ...
"""
function _isSampleInstant(t::Real, start::Real, interval::Real)::Bool
  local n = (t - start) / interval
  return n >= -1e-9 && abs(n - round(n)) <= 1e-9 * max(1.0, abs(n))
end

"""
The conjuncts of `e` (`a and b and c`: a, b, c).
"""
function _conjuncts!(out::Vector{SC.Exp}, e::SC.Exp)::Vector{SC.Exp}
  (e isa SC.LBINARY && e.op === SC.OP_AND) || return push!(out, e)
  _conjuncts!(out, e.exp1)
  return _conjuncts!(out, e.exp2)
end

"""
`(sample call, guard)` of a when condition that is `sample(...)` or a conjunction with
exactly one sample() conjunct (the guard: the other conjuncts, or nothing); `(nothing,
nothing)` otherwise.
"""
function _splitSampleCondition(cond::SC.Exp)::Tuple{Union{Nothing, SC.CALL}, Union{Nothing, SC.Exp}}
  local conjuncts = _conjuncts!(SC.Exp[], cond)
  local isSample = e::SC.Exp -> SC.isBuiltin(e, :sample)
  local samples = filter(isSample, conjuncts)
  length(samples) == 1 || return (nothing, nothing)
  local rest = filter(!isSample, conjuncts)
  any(_containsSampleCall, rest) && return (nothing, nothing)
  isempty(rest) && return (only(samples), nothing)
  return (only(samples), foldl((a, b) -> SC.LBINARY(a, SC.OP_AND, b), rest))
end

"""
The sample() calls of a condition that is only sample()s or-ed (more than one), or nothing.
"""
function _sampleDisjuncts(cond::SC.Exp)::Union{Vector{SC.CALL}, Nothing}
  local out = SC.CALL[]
  local collect! = nothing
  collect! = e::SC.Exp -> begin
    (e isa SC.LBINARY && e.op === SC.OP_OR) && return collect!(e.exp1) && collect!(e.exp2)
    SC.isBuiltin(e, :sample) || return false
    push!(out, e)
    return true
  end
  return (collect!(cond) && length(out) > 1) ? out : nothing
end

"""
`(sample calls, the other disjuncts or-ed)` of a condition that or-s sample()s with other
triggers, none of them under a sample() (Buildings' occupant lighting: `when {occ,
sampleTrigger}`, edge(occ) or sample(t0, samplePeriod)), or nothing.
"""
function _sampleOrOthers(cond::SC.Exp)::Union{Tuple{Vector{SC.CALL}, SC.Exp}, Nothing}
  local samples = SC.CALL[]
  local others = SC.Exp[]
  local collect! = nothing
  collect! = e::SC.Exp -> begin
    if e isa SC.LBINARY && e.op === SC.OP_OR
      collect!(e.exp1); collect!(e.exp2)
    elseif SC.isBuiltin(e, :sample)
      push!(samples, e)
    else
      push!(others, e)
    end
    return nothing
  end
  collect!(cond)
  (isempty(samples) || isempty(others) || any(_containsSampleCall, others)) && return nothing
  return (samples, foldl((a, b) -> SC.LBINARY(a, SC.OP_OR, b), others))
end

"""
The ticks of the sample()s of a when that or-s them with other triggers (`clocks`:
(interval, phase, a tick at the initial time) each). A tick marks its instant in `tickAt`:
the event iteration, which runs after the instant's other ticks, then runs the when once
(`affect!` takes the tick). At the start of a solve no iteration follows: there the tick runs
`affect!` itself. The end of a solve is a tick only where a clock ticks.
"""
function _sampleTickCallback(tickAt::Base.RefValue{Float64}, affect!::Function, clocks::Vector)
  local ticks = map(clocks) do (dt, phase, initialTick)
    local tick! = integrator -> begin
      _isPeriodicTick(integrator, phase, dt) || return nothing
      tickAt[] = integrator.t
      integrator.t == first(integrator.sol.prob.tspan) && affect!(integrator)
      return nothing
    end
    DiffEqCallbacks.PeriodicCallback(tick!, dt; phase = phase, initial_affect = initialTick, final_affect = true,
                                     save_positions = (false, false))
  end
  return DiffEqBase.CallbackSet(ticks...)
end

"""
(interval, phase, a tick at the initial time) of `sample(start, interval)`. The ticks are
start + i*interval, i = 0, 1, ...: the phase is the first one from the initial time (taken as
0). A start not known here is refused (generated code cannot read the parameters at module
level); a negative one ticks from its first tick after 0 (sample(-0.15, 0.25) at 0.1, not
0.25). A tick at the initial time is taken too, after the initialization (omc, Dymola: MSL
RealFFT1's `when sample(0, Ts)` samples y(0) into its FFT buffer).
"""
function _sampleClock(sampleCall::SC.CALL, simCode::SC.SIM_CODE)::Tuple{Any, Float64, Bool}
  local (start, interval) = (sampleCall.args[1], sampleCall.args[2])
  local intervalValue = SC.tryEvalNumeric(interval, simCode)
  local dtExpr = intervalValue === nothing ? expToJuliaExp(interval, simCode) : intervalValue
  local startValue = SC.tryEvalNumeric(start, simCode)
  #= one an initial algorithm assigns (CDL's samplers and pulses: t0) =#
  startValue === nothing && (startValue = SC.valueAtBuildStart(start, simCode))
  startValue === nothing && unsupported("a sample() start not known at the build", start)
  local firstTick = startValue
  if firstTick < 0
    intervalValue === nothing && unsupported("a negative sample() start with an interval not known at the build", sampleCall)
    firstTick += ceil(-firstTick / intervalValue) * intervalValue
    #= Rounding: sample(-0.9, 0.3) gave -1.1e-16, a negative phase (an ArgumentError) and no tick at 0. =#
    abs(firstTick) <= 8 * eps(max(1.0, abs(intervalValue))) && (firstTick = 0.0)
    firstTick = max(0.0, firstTick)
  end
  return (dtExpr, firstTick, iszero(firstTick))
end

_containsSampleCall(e::SC.Exp)::Bool = _containsCallTo(e, :sample)
_containsCallTo(e::SC.Exp, name::Symbol)::Bool = SC.anyExp(x -> SC.isBuiltin(x, name), e)

"""
    eqToJulia(eq, simCode, arrayIdx)

The callback of a when-equation: by its condition, a PresetTimeCallback (a table/time when), a
PeriodicCallback (a sample() clock, alone or and-ed with a guard, or-ed with others; a Pulse's
`integer((time - startTime)/period) > pre(count)`), a buffered relation (one relation, MLS
8.5), a ContinuousCallback (a continuous condition), else a discrete callback of the event
iteration; an elsewhen arm a callback of its own.
"""
function eqToJulia(eq::SC.WHEN_EQUATION, simCode::SC.SIM_CODE, arrayIdx::Int)::Expr
  local wEq = eq.whenEquation
  local condition = wEq.condition
  local cond = transformToZeroCrossingCondition(condition)
  ADD_CALLBACK()
  local callbacks = COUNT_CALLBACKS()
  #= sample(start, interval), alone or and-ed with a guard (MSL PartialNoise: `when
     generateNoise and sample(startTime, samplePeriod)`): periodic, the guard read at each tick. =#
  local (sampleCall, sampleGuard) = _splitSampleCondition(condition)
  #= or-ed sample()s alone (CDL's Boolean and Integer TimeTable: `when {sample(t0 + timeStamps[1],
     period), sample(t0 + timeStamps[2], period), ...}`): a clock each, one body. =#
  local sampleCalls = sampleCall !== nothing ? SC.CALL[sampleCall] : _sampleDisjuncts(condition)
  #= sample()s or-ed with other triggers (Buildings' occupant lighting: `when {occ,
     sampleTrigger}`): a when on the others or a tick of theirs (_sampleTickCallback), on a
     discrete condition. =#
  local tickCalls = nothing
  if sampleCalls === nothing
    local mixed = _sampleOrOthers(condition)
    if mixed !== nothing
      (tickCalls, condition) = mixed
      cond = transformToZeroCrossingCondition(condition)
    end
  end
  #= Lowered elsewhere: a sample() alone or and-ed with a guard, terminal() alone (after the
     solve). Not in MSL 3.2.3; sample() read false, terminal() undefined: never fired. =#
  sampleCalls === nothing && _containsSampleCall(condition) &&
    unsupported("sample() under not, and-ed with another sample(), or and-ed within an or, in a when-condition", condition)
  _containsCallTo(condition, :terminal) && !SC.isBuiltin(condition, :terminal) &&
    unsupported("terminal() with another trigger in a when-condition", condition)
  local isPeriodic = sampleCalls !== nothing
  local isContinuousCond = isContinuousCondition(condition, simCode)
  tickCalls !== nothing && isContinuousCond &&
    unsupported("sample() or-ed with a continuous-time condition in a when-condition", condition)
  #= Table / time-driven sources: a when whose condition is purely change(time>=c) thresholds
     fires AT those times (PresetTimeCallback). =#
  if !isPeriodic && tickCalls === nothing
    local thresholds = _collectTimeThresholds(condition, simCode)
    (thresholds !== nothing && !isempty(thresholds)) && return _emitPresetTimeWhen(eq, simCode, callbacks, thresholds)
    #= Source.Pulse periodic clock `integer((time-startTime)/period) > pre(count)`: fire AT the
       period boundaries (PeriodicCallback). =#
    local pulse = _pulsePeriodicSpec(condition, simCode, wEq.body)
    (pulse !== nothing && pulse[2] > 0.0) && return _emitPulsePeriodicWhen(eq, simCode, callbacks, pulse[1], pulse[2])
  end
  #= A when on one relation: buffered relation with a hysteresis (MLS 8.5). =#
  if isContinuousCond && !isPeriodic && wEq.elsewhenPart === nothing
    local rel = _singleRelationWhen(condition)
    rel === nothing || return _emitRelationWhen(eq, simCode, callbacks, rel)
  end
  #= A `sample(start, period)` is a periodic clock even when its interval is a parameter,
     which isContinuousCondition flags as continuous: all samples on the periodic branch. =#
  if isContinuousCond && !isPeriodic
    local elsePart = _elsewhenInner(wEq.elsewhenPart)
    local isElseIf = elsePart !== nothing && cond == transformToZeroCrossingCondition(_elsewhenCondition(elsePart))
    if isElseIf
      return _continuousWhenElseWhen(eq, elsePart, cond, simCode, callbacks)
    end
    return _continuousWhen(eq, cond, simCode, callbacks)
  elseif isPeriodic
    return _periodicWhen(eq, sampleCalls, sampleGuard, simCode, callbacks)
  end
  return _discreteWhen(eq, condition, cond, tickCalls, simCode, callbacks)
end

"""
The condition function of a continuous when's ContinuousCallback, its zero crossing `cond`
lowered with the variables looked up (MTK may reorder the unknowns: an index of the
variables' order is wrong after structural_simplify). Where none of the names is a state or
a parameter `noTrigger` (the condition cannot be evaluated).
"""
function _continuousWhenCondition(cond::SC.Exp, simCode::SC.SIM_CODE, callbacks::Int; withParameters::Bool)::Expr
  local names = filter(n -> n != "time", SC.allCrefNames(cond))
  local xs = Symbol[Symbol(n) for n in names]
  local unresolved = withParameters ?
    :(!isempty(xs) && all(isnothing, indices) && all(isnothing, indexin(xs, params))) :
    :(!isempty(xs) && all(isnothing, indices))
  quote
    let _condCache = Ref{Any}(nothing)
      global $(Symbol("condition$(callbacks)"))
      $(Symbol("condition$(callbacks)")) = (x, t, integrator) -> begin
        local lookuptableStates
        local lookuptableParams
        if _condCache[] === nothing
          local xs = $(xs)
          local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          if $(unresolved)
            @debug "[CB-CC$($(callbacks)) cond] no trigger (no state/param mapping)" t xs
            return 1.0
          end
          lookuptableStates = isempty(xs) ? Dict{Symbol, Union{Nothing, Int}}() : Dict(xs .=> indices)
          lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          _condCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _condCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        $(_whenLookupBindings(SC.allCrefNames(cond), simCode)...)
        local _result = $(expToJuliaExpMTK(cond, simCode))
        @debug "[CB-CC$($(callbacks)) cond] eval" t value=_result
        _result
      end
    end
  end
end

"""
A continuous when and its elsewhen of the same zero crossing: one ContinuousCallback whose
affect runs the arm the condition selects. pre(v) from the state before the instant
(instantPre): an earlier statement's new value is not pre(v).
"""
function _continuousWhenElseWhen(eq::SC.WHEN_EQUATION, elsePart::SC.WHEN_EQUATION, cond::SC.Exp, simCode::SC.SIM_CODE, callbacks::Int)::Expr
  local wEq = eq.whenEquation
  local (thenStatements, elseStatements) = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    (createWhenStatementsMTK(wEq.body, simCode), createWhenStatementsMTK(_elsewhenStmtLst(elsePart), simCode))
  end
  local reads = vcat(SC.allCrefNames(wEq.condition), _bodyReads(wEq.body), _bodyReads(_elsewhenStmtLst(elsePart)))
  quote
    $(_continuousWhenCondition(cond, simCode, callbacks; withParameters = false))
    let _affCache = Ref{Any}(nothing)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
        local t = integrator.t + integrator.dt
        local x = integrator.u
        $(_lookupTables(:_affCache))
        $(_whenLookupBindings(reads, simCode)...)
        if integrator.dt == 0.0
          @error "integrator.dt was zero. Aborting."
          fail()
        end
        if $(expToJuliaBoolMTK(wEq.condition, simCode))
          $(thenStatements...)
          add_tstop!(integrator, integrator.t + 1E-12) #=TODO: Some small number for now=#
        else
          $(elseStatements...)
          add_tstop!(integrator, integrator.t + 1E-12) #=TODO: Some small number for now=#
        end
      end
    end
    $(Symbol("cb$(callbacks)")) = ContinuousCallback($(Symbol("condition$(callbacks)")),
                                                     $(Symbol("affect$(callbacks)!")),
                                                     rootfind = ModelingToolkit.SciMLBase.RightRootFind,
                                                     save_positions=(true, true),
                                                     affect_neg! = $(Symbol("affect$(callbacks)!")))
  end
end

"""
A continuous when: a ContinuousCallback on its zero crossing. pre(v) from the state before
the instant (instantPre): `b = not pre(c); k = if pre(c) ...` with c = b read b's new value
(k = 20, OpenModelica 10). RightRootFind: the event lands just past the root, where the
condition has changed sign; from the left of it the next step finds the same crossing
again (DiffEqBase 7 no longer suppresses that repeat: ElseWhenBasic looped at x = 0.3 until
MaxIters). No `affect_neg!`: transformToZeroCrossingCondition encodes the direction
(positive to negative triggers); `affect_neg! = affect!` double-fired on each oscillation
(the bouncing ball's reinit at the down- and the up-crossing: Zeno, maxiters).
"""
function _continuousWhen(eq::SC.WHEN_EQUATION, cond::SC.Exp, simCode::SC.SIM_CODE, callbacks::Int)::Expr
  local wEq = eq.whenEquation
  local whenStatements = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(wEq.body, simCode)
  end
  if wEq.elsewhenPart !== nothing
    #= AUDIT (ombackend-bug-audit-2026-06-05 #11): a continuous when/elsewhen whose arm
       conditions differ is lowered to independent ContinuousCallbacks (the recursion below), no
       shared per-instant exclusivity guard: elsewhen ordering holds for non-simultaneous events
       but NOT when both arm conditions cross zero at the same instant (both bodies execute,
       last-writer-wins on a shared discrete). A shared fired-at-instant guard threaded
       through the recursion is the proper fix. =#
    @warn "[CodeGen: continuous when/elsewhen] $(simCode.name): continuous `when`/`elsewhen` with distinct arm conditions is lowered to independent ContinuousCallbacks; elsewhen mutual-exclusion is NOT enforced when both arm conditions cross zero at the same instant (simultaneous-event edge case). See ombackend-bug-audit-2026-06-05 #11."
  end
  quote
    $(_continuousWhenCondition(cond, simCode, callbacks; withParameters = true))
    let _affCache = Ref{Any}(nothing)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
        local t = integrator.t + integrator.dt
        local x = integrator.u
        @debug "[CB-CC$($(callbacks)) affect] firing" t=integrator.t
        $(_lookupTables(:_affCache))
        $(_whenLookupBindings(_bodyReads(wEq.body), simCode)...)
        if integrator.dt == 0.0
          @error "integrator.dt was zero. Aborting."
          fail()
        end
        $(whenStatements...)
        @debug "[CB-CC$($(callbacks)) affect] done" t=integrator.t u=copy(integrator.u)
      end
    end
    $(Symbol("cb$(callbacks)")) = ContinuousCallback($(Symbol("condition$(callbacks)")),
                                                     $(Symbol("affect$(callbacks)!")),
                                                     rootfind = ModelingToolkit.SciMLBase.RightRootFind, save_positions=(true, true))
    $(wEq.elsewhenPart === nothing ? nothing : eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 0))
  end
end

"""
A when on sample() clocks: a PeriodicCallback per clock (the body once at an instant two
clocks share), the guard read at each tick. The variables are looked up (MTK reorders the
unknowns). pre(v) from the state before the instant (instantPre): a body reading pre(c)
after setting its alias b (`b = not pre(c); k = if pre(c) ...`) read the new value. The
discrete-Boolean whens whose condition reads a discrete the body writes (BooleanPulse `y`
reads the re-sampled `pulseStart`) are refreshed in the affect: their own continuous
callbacks cannot catch the threshold jump.
"""
function _periodicWhen(eq::SC.WHEN_EQUATION, sampleCalls::Vector{SC.CALL}, sampleGuard::Union{Nothing, SC.Exp},
                       simCode::SC.SIM_CODE, callbacks::Int)::Expr
  local wEq = eq.whenEquation
  local whenStatements = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(wEq.body, simCode)
  end
  local (refreshCrefs, refreshStmts) = _collectDiscreteBoolWhenRefresh(_bodyWrites(wEq.body), simCode)
  #= (interval, phase, a tick at the initial time) of each clock =#
  local clocks = [_sampleClock(c, simCode) for c in sampleCalls]
  local guardCrefs = sampleGuard === nothing ? String[] : SC.allCrefNames(sampleGuard)
  local guardExpr = sampleGuard === nothing ? true : expToJuliaExpMTK(sampleGuard, simCode)
  quote
    let _affCache = Ref{Any}(nothing), _lastTick = Ref(NaN)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        $(length(clocks) == 1 ?
          :(OMBackend.CodeGeneration._isPeriodicTick(integrator, $(Symbol("samplePhase$(callbacks)")),
                                                     $(Symbol("sampleDt$(callbacks)"))) || return nothing) :
          quote
            any(c -> OMBackend.CodeGeneration._isPeriodicTick(integrator, c[2], c[1]), $(Symbol("sampleClocks$(callbacks)"))) ||
              return nothing
            #= two clocks on one instant: the body once =#
            _lastTick[] == integrator.t && return nothing
            _lastTick[] = integrator.t
          end)
        local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
        local t = integrator.t
        local x = integrator.u
        local p = integrator.p
        $(_lookupTables(:_affCache))
        $(_whenLookupBindings(guardCrefs, simCode)...)
        $(guardExpr) == true || return nothing
        $(_whenLookupBindings(_bodyReads(wEq.body), simCode)...)
        $(whenStatements...)
        #= Re-derive dependent discrete-bool whens from the just-written threshold. =#
        $(_whenLookupBindings(refreshCrefs, simCode)...)
        $(refreshStmts...)
      end
    end
    $(Symbol("sampleDt$(callbacks)")) = $(clocks[1][1])
    $(Symbol("samplePhase$(callbacks)")) = $(clocks[1][2])
    $(Symbol("sampleClocks$(callbacks)")) = Any[$([:(($(c[1]), $(c[2]))) for c in clocks]...)]
    #= Ticks at start + i*interval, the initial and the final time included (_isPeriodicTick). =#
    $(Symbol("cb$(callbacks)")) = $(length(clocks) == 1 ?
      :(PeriodicCallback($(Symbol("affect$(callbacks)!")), $(Symbol("sampleDt$(callbacks)"));
                         phase = $(Symbol("samplePhase$(callbacks)")),
                         initial_affect = $(clocks[1][3]), final_affect = true,
                         save_positions = (true, true))) :
      Expr(:call, :CallbackSet, [:(PeriodicCallback($(Symbol("affect$(callbacks)!")), $(c[1]);
                                                      phase = $(c[2]), initial_affect = $(c[3]), final_affect = true,
                                                      save_positions = (true, true))) for c in clocks]...))
    $(wEq.elsewhenPart === nothing ? nothing : eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 4))
  end
end

"""
`_changePreValues[x] = <state vector>[i]` for the non-parameter names of a condition: the
change()/edge() memory.
"""
function _changeMemoryUpdates(names::Vector{String}, simCode::SC.SIM_CODE, stateVector::Union{Symbol, Expr})::Vector{Expr}
  local out = Expr[]
  for n in names
    local entry = get(simCode.stringToSimVarHT, n, nothing)
    (entry === nothing || SC.isParameter(last(entry))) && continue
    local sym = QuoteNode(Symbol(last(entry).name))
    push!(out, quote
      let _idx = get(lookuptableStates, $(sym), nothing)
        if _idx !== nothing
          _changePreValues[$(sym)] = $(stateVector)[_idx]
        end
      end
    end)
  end
  return out
end

"""
A when on a discrete condition: a callback of the event iteration (discreteWhenCallback).
The variables are looked up (MTK reorders the unknowns). pre(v) from a snapshot: the event
iteration passes the state before its sweep (the algebraic unknowns are solved again before
the discrete whens run, and `iAtOpen = pre(i)` read the current i). An elsewhen arm `time >=
thr` with a runtime-discrete threshold is a scheduled time event: the parent affect (which
assigns the threshold) adds a tstop at it, the arm an edge-guarded DiscreteCallback ordered
after the parent (_emitElsewhenThresholdTimeWhen).
"""
function _discreteWhen(eq::SC.WHEN_EQUATION, condition::SC.Exp, cond::SC.Exp, tickCalls::Union{Nothing, Vector{SC.CALL}},
                       simCode::SC.SIM_CODE, callbacks::Int)::Expr
  local wEq = eq.whenEquation
  local whenStatements = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(wEq.body, simCode)
  end
  local ewArm = _elsewhenInner(wEq.elsewhenPart)
  local ewThr = ewArm === nothing ? nothing : _discreteTimeEventThreshold(_elsewhenCondition(ewArm), simCode)
  local ewRefSym = Symbol("ewLastThr$(callbacks)")
  local ewDecl = ewThr === nothing ? :() : :(local $(ewRefSym) = Ref(NaN))
  local ewSchedule = ewThr === nothing ? :() : quote
    $(_whenLookupBindings(SC.allCrefNames(ewThr), simCode)...)
    local _ewThr = Float64($(expToJuliaExpMTK(ewThr, simCode)))
    if _ewThr > integrator.t
      add_tstop!(integrator, _ewThr)
    else
      $(ewRefSym)[] = _ewThr
    end
  end
  local condCrefs = filter(c -> c != "time", SC.allCrefNames(cond))
  #= The ticks of or-ed sample()s (_sampleTickCallback): the when fires at a tick it has not
     taken, once at an instant. =#
  local tickCb = tickCalls === nothing ? 0 : ADD_CALLBACK()
  local tickNow = :(_tickAt[] == integrator.t && _tickTaken[] != integrator.t)
  local tickClocks = tickCalls === nothing ? Expr[] :
    [:(($(c[1]), $(c[2]), $(c[3]))) for c in (_sampleClock(s, simCode) for s in tickCalls)]
  #= From the condition as written: the zero-crossing form unwraps a top-level `change(k)` to `k`. =#
  local bareCrefs = _collectBareCrefStrings(condition)
  #= A condition on a Boolean (`when u`) is true for as long as u is, but the when fires once,
     when it becomes true: an edge latch. The condition fires only while the latch is clear and
     clears it when it reads false; the affect sets it to the condition's value after the
     event (change() terms are false again then), and the start of a solve to its value on the
     initialized state (a Boolean true from the start is no edge). The condition only clears
     it: the event iteration evaluates a condition several times before running the affect.
     Setting u itself to false instead corrupted it where it is read elsewhere (the MSL Timer's
     input, a threshold block's output, read by `y = if u then time - entryTime ...`). =#
  local useLatch = any(condCrefs) do c
    local entry = get(simCode.stringToSimVarHT, c, nothing)
    c in bareCrefs && entry !== nothing && !SC.isParameter(last(entry))
  end
  #= A DiscreteCallback condition returns a Bool (SciMLBase): a Boolean discrete is a Float64 in
     `integrator.u` (0.0/1.0), a bare reference a Float64 ("non-boolean (Float64) used in
     boolean context": the PowerConverters Thyristor models). =#
  local condValue = quote
    $(_whenLookupBindings(SC.allCrefNames(cond), simCode)...)
    local _r = $(expToJuliaBoolMTK(condition, simCode; cachedChange = true))
    _r isa Bool ? _r : (_r != 0)
  end
  local changeSeedPairs = Expr[]
  for c in condCrefs
    local entry = get(simCode.stringToSimVarHT, c, nothing)
    (entry === nothing || SC.isParameter(last(entry))) && continue
    push!(changeSeedPairs, :($(QuoteNode(Symbol(last(entry).name))) => $(_readStartAttributeAsLiteral(last(entry)))))
  end
  quote
    $(ewDecl)
    #= Typed caches: the condition runs after every step. =#
    let _condCache = Ref{Union{Nothing, Tuple{Dict{Symbol, Union{Nothing, Int}}, Dict{Symbol, Int}}}}(nothing),
        _affCache = Ref{Union{Nothing, Tuple{Dict{Symbol, Int}, Dict{Symbol, Int}}}}(nothing),
        _changeCache = Ref{Union{Nothing, Dict{Symbol, Any}}}(nothing),
        _latch = Ref{Bool}(false),     # the edge latch
        $((tickCalls === nothing ? () : (:(_tickAt = Ref(NaN)), :(_tickTaken = Ref(NaN))))...)
        _changeSeedValues = Dict{Symbol, Any}($(changeSeedPairs...))
      global $(Symbol("condition$(callbacks)"))
      #= Whether the when fires. `_follow`: whether change()/edge() may take the values now,
         where it does not fire (false within an event iteration's sweep; see below). =#
      $(Symbol("condition$(callbacks)")) = (x, t, integrator, _follow::Bool = true) -> begin
        local lookuptableStates
        local lookuptableParams
        if _condCache[] === nothing
          local xs = $(Symbol[Symbol(c) for c in condCrefs])
          local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
          if !isempty(xs) && all(isnothing, indices)
            @debug "[CB-DC$($(callbacks)) cond] false (no state mapping)" t xs
            return false
          end
          lookuptableStates = Dict{Symbol, Union{Nothing, Int}}((xs) .=> indices)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          lookuptableParams = Dict{Symbol, Int}(sym => i for (i, sym) in enumerate(params))
          _condCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _condCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        local _changePreValues
        if _changeCache[] === nothing
          _changePreValues = copy(_changeSeedValues)
          _changeCache[] = _changePreValues
        else
          _changePreValues = _changeCache[]
        end
        local _result = $(condValue)
        @debug "[CB-DC$($(callbacks)) cond] eval" t value=_result
        $(useLatch ? :(_result || (_latch[] = false)) : :())
        local _fires = $(let f = useLatch ? :(_result && !_latch[]) : :_result
                           tickCalls === nothing ? f : :($f || $tickNow)
                         end)
        #= change()/edge() compare with the values at the previous event (pre()): where the when
           does not fire, the memory follows the values now (off turned false at 1/12 without
           firing a `when edge(off)`, which then missed off's edge at 5/12); where it fires, its
           affect does. Not within a sweep of the event iteration: an algebraic value the next
           sweep solves again is stale there, and `edge(b) and c` would lose b's edge. =#
        $(isempty(changeSeedPairs) ? :() : :(_follow && !_fires && begin $(_changeMemoryUpdates(condCrefs, simCode, :x)...) end))
        _fires
      end
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator, $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)) -> begin
        local t = integrator.t
        local x = integrator.u
        @debug "[CB-DC$($(callbacks)) affect] firing" t=integrator.t
        $(tickCalls === nothing ? :() : :(_tickAt[] == integrator.t && (_tickTaken[] = integrator.t)))
        local lookuptableStates
        local lookuptableParams
        if _affCache[] === nothing
          local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          lookuptableStates = Dict{Symbol, Int}(sym => i for (i, sym) in enumerate(states))
          lookuptableParams = Dict{Symbol, Int}(sym => i for (i, sym) in enumerate(params))
          _affCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _affCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        $(_whenLookupBindings(_bodyReads(wEq.body), simCode)...)
        $(whenStatements...)
        $(ewSchedule)
        OMBackend.CodeGeneration._autoDtReset!(integrator)
        add_tstop!(integrator, integrator.t + 1E-12)
        local _changePreValues = _changeCache[] === nothing ? Dict{Symbol, Any}() : _changeCache[]
        _changeCache[] = _changePreValues
        $(_changeMemoryUpdates(condCrefs, simCode, :(integrator.u))...)
        #= On the state vector: the body bound the model's variables by name (a variable `x`
           replaced the state vector `x`: a BoundsError). =#
        $(useLatch ? :(_latch[] = let x = integrator.u; $(condValue) end) : :())
        @debug "[CB-DC$($(callbacks)) affect] done" t=integrator.t u=copy(integrator.u)
      end
      #= At the start of a solve: change()/edge() compare with the initialized values (pre() at
         the first event), not the start attributes: `when edge(off)` with off(start = true)
         initialized false never fired (the MSL switch with arc: its arc voltage never started
         ramping). And the edge latch from the initialized state. The latch implies a seed: it
         is used for a non-parameter reference of the condition. =#
      global $(Symbol("initialize$(callbacks)!"))
      $(Symbol("initialize$(callbacks)!")) = $(isempty(changeSeedPairs) ?
        :(OMBackend.CodeGeneration._NO_WHEN_INITIALIZE) :
        :((x, t, integrator) -> begin
            local _states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
            local _pre = copy(_changeSeedValues)
            for _sym in keys(_changeSeedValues)
              local _i = findfirst(==(_sym), _states)
              _i === nothing || (_pre[_sym] = x[_i])
            end
            _changeCache[] = _pre
            $(useLatch ? :(_latch[] = false; _latch[] = $(Symbol("condition$(callbacks)"))(x, t, integrator)) : :())
            nothing
          end))
      $(tickCalls === nothing ? :() : quote
          global $(Symbol("cb$(tickCb)"))
          $(Symbol("cb$(tickCb)")) = OMBackend.CodeGeneration._sampleTickCallback(_tickAt, $(Symbol("affect$(callbacks)!")),
                                                                                 Any[$(tickClocks...)])
        end)
    end
    #= Part of the event iteration where the model has buffered relations. =#
    $(Symbol("cb$(callbacks)")) = OMBackend.CodeGeneration.discreteWhenCallback($(Symbol("condition$(callbacks)")),
                                                                              $(Symbol("affect$(callbacks)!")),
                                                                              $(Symbol("initialize$(callbacks)!")))
    $(if ewThr !== nothing
        _emitElsewhenThresholdTimeWhen(ewArm, simCode, ewRefSym, ewThr)
      elseif wEq.elsewhenPart !== nothing
        eqToJulia(ewArm, simCode, 4)
      end)
  end
end

#= ==================================================================================== =#
#= Expressions over the solver's arrays                                                 =#
#= ==================================================================================== =#

"""
    expToJuliaExp(e, simCode; varPrefix = "x", varSuffix = "")

The Julia expression of `e` over the solver's arrays (the structural callbacks of models with
structural modes, a sample() interval read from the parameters): a state, an algebraic or a
discrete variable as `x[i]` (`varPrefix`: `integrator.u`), a parameter `p[i]`, a state
derivative `dx[i]`, a data structure or a String by its module-level name, `time` as `t`.
"""
function expToJuliaExp end

expToJuliaExp(e::Union{SC.BCONST, SC.ICONST, SC.RCONST, SC.SCONST}, ::SC.SIM_CODE;
              varPrefix::String = "x", varSuffix::String = "")::Expr = quote $(e.value) end

function expToJuliaExp(e::SC.EXP_CREF, simCode::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr
  local name = SC.crefName(e.cref)
  name == "time" && return quote t end
  local (index, v) = simCode.stringToSimVarHT[name]
  local k = v.varKind
  k isa SC.PARAMETER && return quote p[$index] end
  k isa Union{SC.STATE, SC.ALG_VARIABLE, SC.DISCRETE} && return quote $(Meta.parse(varPrefix))[$index] end
  k isa SC.STATE_DERIVATIVE && return quote $(Symbol("dx", varSuffix))[$index] end
  k isa Union{SC.DATA_STRUCTURE, SC.STRING} && return quote $(Symbol(v.name)) end
  unsupported("a reference to a $(nameof(typeof(k))) over the solver's arrays", name)
end

function expToJuliaExp(e::Union{SC.UNARY, SC.LUNARY}, simCode::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr
  local o = opKindToJuliaOperator(e.op)
  return quote $(o)($(expToJuliaExp(e.exp, simCode; varPrefix, varSuffix))) end
end

function expToJuliaExp(e::Union{SC.BINARY, SC.RELATION}, simCode::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr
  local o = opKindToJuliaOperator(e.op)
  return quote $(o)($(expToJuliaExp(e.exp1, simCode; varPrefix, varSuffix)), $(expToJuliaExp(e.exp2, simCode; varPrefix, varSuffix))) end
end

function expToJuliaExp(e::SC.LBINARY, simCode::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr
  local a = expToJuliaExp(e.exp1, simCode; varPrefix, varSuffix)
  local b = expToJuliaExp(e.exp2, simCode; varPrefix, varSuffix)
  return quote $(Expr(e.op === SC.OP_AND ? :&& : :||, a, b)) end
end

function expToJuliaExp(e::SC.IFEXP, simCode::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr
  local lower = x::SC.Exp -> expToJuliaExp(x, simCode; varPrefix, varSuffix)
  return :(ifelse($(lower(e.cond)), $(lower(e.thenExp)), $(lower(e.elseExp))))
end

function expToJuliaExp(e::SC.CALL, simCode::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr
  local lower = x::SC.Exp -> expToJuliaExp(x, simCode; varPrefix, varSuffix)
  e.kind === SC.CALL_BUILTIN && return builtinCallToJulia(e, lower)
  return Expr(:call, e.name, map(lower, e.args)...)
end

function expToJuliaExp(e::SC.CAST, simCode::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr
  e.ty isa SC.TYPE_REAL || unsupported("cast to $(e.ty) over the solver's arrays", e)
  return quote float($(expToJuliaExp(e.exp, simCode; varPrefix, varSuffix))) end
end

expToJuliaExp(e::SC.Exp, ::SC.SIM_CODE; varPrefix::String = "x", varSuffix::String = "")::Expr =
  unsupported("expression over the solver's arrays", e)
