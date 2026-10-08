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
  The when-callback emitter the MTK path uses, and expToJuliaExp (DAE.Exp and
  SimCode Exp to Julia) for the structural callbacks and CodeGenerationUtil.
  The direct DifferentialEquations.jl backend is DECodeGeneration.jl.

  Author: John Tinnerholm
=#

"""
  Contains the headerstring defining the OpenModelica copyright notice.
"""
const HEADER_STRING ="
  $(copyrightString())"

#= Unwrap a `WHEN_STMTS.elsewhenPart` to the inner WHEN_EQUATION-or-WHEN_STMTS,
   accounting for the two storage shapes carried during the BDAE → SimCode
   migration. BDAE wraps with `SOME(WHEN_EQUATION(WHEN_STMTS(...)))`; SimCode
   stores the bare `SIM_WHEN_STMTS`. Returns `nothing` if there is no
   elsewhen. =#
_elsewhenInner(::Nothing) = nothing
_elsewhenInner(p::MetaModelica.SOME) = p.data
_elsewhenInner(p::SimulationCode.WHEN_STMTS) =
  SimulationCode.WHEN_EQUATION(0, p, DAE.emptyElementSource, SimulationCode.EQ_ATTR_DEFAULT)
_elsewhenInner(p) = p

#= Condition expression of an elsewhen arm regardless of wrapper shape. =#
_elsewhenCondition(arm::SimulationCode.WHEN_STMTS) = arm.condition
_elsewhenCondition(arm::SimulationCode.WHEN_EQUATION) = arm.whenEquation.condition
_elsewhenCondition(arm) = arm.whenEquation.condition

#= Statement list of an elsewhen arm regardless of wrapper shape. =#
_elsewhenStmtLst(arm::SimulationCode.WHEN_STMTS) = arm.whenStmtLst
_elsewhenStmtLst(arm::SimulationCode.WHEN_EQUATION) = arm.whenEquation.whenStmtLst
_elsewhenStmtLst(arm) = arm.whenEquation.whenStmtLst

#= Walk a DAE.Exp condition and collect the set of cref-name strings that
   appear OUTSIDE any `change(...)` or `pre(...)` wrapper: the level-valued
   Booleans of the condition (`when boolVar then`), for which the discrete
   callback keeps an edge latch. =#
Base.@nospecializeinfer function _collectBareCrefStrings(@nospecialize(cond))::OrderedSet{String}
  local out = OrderedSet{String}()
  local walk = function(e, insideObs)
    if e isa DAE.CREF
      insideObs || push!(out, string(e))
      return
    elseif e isa DAE.CALL
      local nameStr = string(e.path)
      local nestedObs = insideObs || nameStr == "change" || nameStr == "pre"
      for arg in e.expLst
        walk(arg, nestedObs)
      end
    elseif e isa DAE.BINARY
      walk(e.exp1, insideObs); walk(e.exp2, insideObs)
    elseif e isa DAE.LBINARY
      walk(e.exp1, insideObs); walk(e.exp2, insideObs)
    elseif e isa DAE.RELATION
      walk(e.exp1, insideObs); walk(e.exp2, insideObs)
    elseif e isa DAE.UNARY
      walk(e.exp, insideObs)
    elseif e isa DAE.LUNARY
      walk(e.exp, insideObs)
    elseif e isa DAE.IFEXP
      walk(e.expCond, insideObs); walk(e.expThen, insideObs); walk(e.expElse, insideObs)
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
  Creates runnable code for the different callbacks.
  By default a saving function is generated.
  This function can be disabled by setting the named argument
  generateSaveFunction to false.
"""
function createCallbackCode(modelName, simCode; generateSaveFunction = true)
  #= Synthesised discrete-Boolean whens (`change(rel)` conditions) are discrete
     clusters of the event iteration (emitDiscreteClusters) or, with
     OMBACKEND_DISCRETE_PRE_MEMORY=false, MTK SymbolicContinuousCallbacks
     (createDiscreteBoolWhenEvents); exclude them here so they are not also
     built into the legacy CallbackSet (which cannot read MTK observed
     variables). =#
  local _legacyWhens = filter(w -> _extractChangeRelations(w.whenEquation.condition, simCode) === nothing &&
                                   isempty(_selfSchedulingTimeRels(w)),
                              simCode.whenEquations)
  local WHEN_EQUATIONS = createEquations(_legacyWhens,  simCode)
  #=
    For if equations we create zero crossing functions (Based on the conditions).
    The body of these equations are evaluated in the main body of the solver itself.
  =#
  #local IF_EQUATIONS = createIfEquationCallbacks(simCode.ifEquations, simCode) Deprecated
  local SAVE_FUNCTION = if generateSaveFunction
    createSaveFunction(modelName)
  else
  end
  local MODEL_NAME = modelName
  #= Only emit the saved_values_<model> = SavedValues(...) declaration when the
     save function is actually generated. SavedValues lives in DiffEqCallbacks,
     which is a transitive (not direct) dep of OMBackend; in MTK mode the save
     function is disabled and the saved_values_ binding was never read, but its
     unconditional emission caused UndefVarError at simulate time when the
     per-model module tried to evaluate it without DiffEqCallbacks imported. =#
  local SAVED_VALUES_DECL = if generateSaveFunction
    :( $(Symbol("saved_values_$(modelName)")) = SavedValues(Float64, Tuple{Float64,Array}) )
  else
    nothing
  end
  quote
    $(SAVED_VALUES_DECL)
    function $(Symbol("$(MODEL_NAME)CallbackSet"))(aux)
      #= These are the locations of the parameters and auxiliary real variables respectively =#
      local p = aux[1]
      local reals = aux[2]
      local reducedSystem = aux[3]
      $(LineNumberNode((@__LINE__), "WHEN EQUATIONS"))
      $(WHEN_EQUATIONS...)
      $(LineNumberNode((@__LINE__), "IF EQUATIONS"))
      #      $(IF_EQUATIONS...)
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
function createSaveFunction(modelName)::Expr
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
function returnCallbackSet()::Array
  local cbs::Vector{Symbol} = Symbol[]
  for t in 1:COUNT_CALLBACKS()
    cb = Symbol("cb", t)
    push!(cbs, cb)
  end
  return cbs
end

"""
 Create a set for all equations.
"""
function createEquations(equations::Vector{T}, simCode::SimulationCode.SIM_CODE)::Vector{Expr} where T
  local eqs = Expr[]
  for (equationCounter, eq) in enumerate(equations)
    local eqJL::Expr = eqToJulia(eq, simCode, equationCounter)
    push!(eqs, eqJL)
  end
  return eqs
end


#= The lookups of the variables name[1], name[2], ... (none when name[1] is
   not a variable). =#
function _scalarizedElementLookups(name::String, simCode)::Vector{Any}
  local elems = Any[]
  local key = string(name, "[1]")
  while haskey(simCode.stringToSimVarHT, key)
    push!(elems, getIdxForLookupMTK(key, simCode))
    key = string(name, "[", length(elems) + 1, "]")
  end
  return elems
end

#= The names the when callbacks bind themselves: a variable binding of the
   same name would replace them. =#
const _WHEN_CALLBACK_LOCALS = ("x", "p", "t", "integrator", "lookuptableStates", "lookuptableParams")

#= Build `name = <state/param lookup index>` pre-bindings for the crefs a when
   callback reads, one per name. A cref absent from the simvar table is an
   inlined constant (e.g. a logic ResetMap[i] element) that expToJulia emits as
   a literal, and gets none (a state/param index for it would KeyError in
   getIdxForLookupMTK), unless it is a whole array of variables: then the
   vector of their values (MSL GenerateRandomNumbers' when reads pre(state64),
   a discrete Integer[2]). =#
function _whenLookupBindings(crefs, simCode)::Vector{Expr}
  local out = Expr[]
  local seen = Set{String}()
  for x in collect(map(identity, crefs))
    local name = string(x)
    name in seen && continue
    push!(seen, name)
    local entry = get(simCode.stringToSimVarHT, name, nothing)
    if entry === nothing
      local elems = _scalarizedElementLookups(name, simCode)
      isempty(elems) && continue
      name in _WHEN_CALLBACK_LOCALS &&
        OMBackend.unsupported("a when reading the whole array $(name) (a name the event callback uses itself)", x)
      push!(out, Expr(:(=), Symbol(name), Expr(:vect, elems...)))
      continue
    end
    #= String simvars live as module-level bindings, never as state or MTK
       parameter slots; an index binding here would KeyError at runtime. =#
    if OMBackend.envSwitch("OMBACKEND_WHEN_STRING_SKIP")
      entry[2].varKind isa SimulationCode.STRING && continue
    end
    #= A data structure (an external object: Buildings' borehole ExtendableArray) is a
       module-level binding too (createDataStructureAssignments). =#
    entry[2].varKind isa SimulationCode.DATA_STRUCTURE && continue
    push!(out, Expr(:(=), Symbol(name), getIdxForLookupMTK(x, simCode)))
  end
  return out
end

_isTimeCref(@nospecialize(e)) = @match e begin
  DAE.CREF(componentRef = cr) => string(cr) == "time"
  _ => false
end

#= Extract the constant threshold of a `time <relop> c` relation, or nothing. =#
function _timeThreshold(@nospecialize(rel), simCode)
  @match rel begin
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      local isCmp = @match op begin
        DAE.LESS(__) => true
        DAE.LESSEQ(__) => true
        DAE.GREATER(__) => true
        DAE.GREATEREQ(__) => true
        _ => false
      end
      isCmp || return nothing
      local thr = _isTimeCref(e1) ? e2 : (_isTimeCref(e2) ? e1 : nothing)
      thr === nothing && return nothing
      local v = OMBackend._tryOr(() -> SimulationCode.tryEvalNumeric(thr, simCode), nothing, :timeThreshold)
      v === nothing ? nothing : Float64[Float64(v)]
    end
    _ => nothing
  end
end

#= Threshold expression of a `time >= thr` (or mirrored `thr <= time`) relation
   whose threshold contains at least one runtime discrete variable, or nothing.
   Such a condition is a runtime-scheduled time event: the threshold is only
   known once the assigning when fires, so it cannot use PresetTimeCallback,
   and a ContinuousCallback is unsafe (simultaneous crossings of several such
   callbacks are tie-broken to a single applied affect). =#
function _discreteTimeEventThreshold(@nospecialize(cond), simCode)
  @match cond begin
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      local thr = if _isTimeCref(e1)
        @match op begin
          DAE.GREATEREQ(__) => e2
          DAE.GREATER(__) => e2
          _ => nothing
        end
      elseif _isTimeCref(e2)
        @match op begin
          DAE.LESSEQ(__) => e1
          DAE.LESS(__) => e1
          _ => nothing
        end
      else
        nothing
      end
      thr === nothing && return nothing
      local hasDiscrete = false
      for c in Util.getAllCrefs(thr)
        local k = string(c)
        k == "time" && return nothing
        haskey(simCode.stringToSimVarHT, k) || return nothing
        local v = simCode.stringToSimVarHT[k][2]
        if SimulationCode.isDiscrete(v)
          hasDiscrete = true
        elseif !SimulationCode.isParameter(v)
          return nothing
        end
      end
      hasDiscrete ? thr : nothing
    end
    _ => nothing
  end
end

#= True for `change(p)` where p is a parameter/constant — it never fires, so it
   contributes no event and does not disqualify a pure time-threshold chain. =#
function _isConstChangeArg(@nospecialize(a), simCode)
  @match a begin
    DAE.CREF(componentRef = cr) => begin
      local k = string(cr)
      haskey(simCode.stringToSimVarHT, k) && SimulationCode.isParameter(simCode.stringToSimVarHT[k][2])
    end
    DAE.RCONST(_) => true
    DAE.ICONST(_) => true
    DAE.BCONST(_) => true
    _ => false
  end
end

#= Detect a synthesized table/time when-condition: an OR-chain whose every leaf is
   `change(time <relop> const)` (a time threshold) or `change(param)` (never fires).
   Returns the threshold times (Float64, may be empty) or nothing when the condition
   has any other trigger. Such whens must fire AT the thresholds via a
   PresetTimeCallback — a ContinuousCallback rootfinding on the spiky change() value
   never detects the crossings (Digital.Sources.Table / time-driven sources). =#
function _collectTimeThresholds(@nospecialize(cond), simCode)
  @match cond begin
    DAE.LBINARY(exp1 = e1, operator = DAE.OR(__), exp2 = e2) => begin
      local l = _collectTimeThresholds(e1, simCode)
      local r = _collectTimeThresholds(e2, simCode)
      (l === nothing || r === nothing) ? nothing : vcat(l, r)
    end
    DAE.CALL(Absyn.IDENT("change"), lst, _) => begin
      local args = listArray(lst)
      length(args) == 1 || return nothing
      local thr = _timeThreshold(args[1], simCode)
      thr !== nothing && return thr
      _isConstChangeArg(args[1], simCode) ? Float64[] : nothing
    end
    _ => nothing
  end
end

#= Emit an `elsewhen time >= thr` arm (thr containing a runtime discrete) as an
   edge-guarded DiscreteCallback. ContinuousCallbacks are unsafe here: when
   several such arms cross zero at the same instant the integrator applies only
   one and the rest never re-fire. `ewRefSym` names a Ref shared with the parent
   when-branch holding the last consumed threshold: the parent consumes the
   threshold when it fires at or past it (elsewhen exclusivity), and this
   callback consumes it on firing so a level-true condition stays edge-only. =#
function _emitElsewhenThresholdTimeWhen(elseArm, simCode, ewRefSym::Symbol, thrDAE)
  ADD_CALLBACK()
  local callbacks = COUNT_CALLBACKS()
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(_elsewhenStmtLst(elseArm), simCode)
  end
  local thrCrefs = listArray(Util.getAllCrefs(thrDAE))
  local affBindCrefs = vcat(map(x -> getRHSVariables(x), _elsewhenStmtLst(elseArm))..., thrCrefs)
  quote
    let _condCache = Ref{Any}(nothing), _affCache = Ref{Any}(nothing)
      global $(Symbol("condition$(callbacks)"))
      #= `_follow` as a discrete when's (no change()/edge() memory here). =#
      $(Symbol("condition$(callbacks)")) = (x, t, integrator, _follow::Bool = true) -> begin
        local lookuptableStates
        local lookuptableParams
        if _condCache[] === nothing
          local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
          lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          _condCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _condCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        $(_whenLookupBindings(thrCrefs, simCode)...)
        local _thr = Float64($(expToJuliaExpMTK(thrDAE, simCode)))
        t >= _thr && _thr != $(ewRefSym)[]
      end
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator, $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)) -> begin
        local t = integrator.t
        local x = integrator.u
        @debug "[CB-EW$($(callbacks)) affect] firing" t=integrator.t
        local lookuptableStates
        local lookuptableParams
        if _affCache[] === nothing
          local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
          lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          _affCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _affCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        $(_whenLookupBindings(affBindCrefs, simCode)...)
        $(ewRefSym)[] = Float64($(expToJuliaExpMTK(thrDAE, simCode)))
        $(whenStmts...)
        auto_dt_reset!(integrator)
        add_tstop!(integrator, integrator.t + 1E-12)
      end
    end
    #= A discrete when, run after its parent's (in the event iteration where
       the model has buffered relations). =#
    $(Symbol("cb$(callbacks)")) = OMBackend.CodeGeneration.discreteWhenCallback($(Symbol("condition$(callbacks)")),
                                                                              $(Symbol("affect$(callbacks)!")))
  end
end

#= Emit a PresetTimeCallback for a table/time when: fire the (time-dependent) body at
   each threshold so a stepped output (e.g. a Digital Table) lands on its sample times. =#
function _emitPresetTimeWhen(eq, simCode, callbacks::Int, thresholds::Vector{Float64})
  local wEq = eq.whenEquation
  #= pre(v) from the state before the instant (instantPre): an earlier statement's new value is not pre(v). =#
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(wEq.whenStmtLst, simCode)
  end
  local bodyCrefs = vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...)
  local times = sort(unique(filter(>(0.0), thresholds)))
  quote
    let _affCache = Ref{Any}(nothing)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
        local t = integrator.t
        local x = integrator.u
        local lookuptableStates
        local lookuptableParams
        if _affCache[] === nothing
          local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
          lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          _affCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _affCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        $(_whenLookupBindings(bodyCrefs, simCode)...)
        $(whenStmts...)
      end
    end
    $(Symbol("cb$(callbacks)")) = PresetTimeCallback($(times), $(Symbol("affect$(callbacks)!")))
  end
end

_isPreCref(@nospecialize(e)) = @match e begin
  DAE.CALL(Absyn.IDENT("pre"), _, _) => true
  _ => false
end

#= Match `(time - S)/P` (or `time/P`) and return (S, P) as numerics, or nothing. =#
function _timeOffsetOverPeriod(@nospecialize(e), simCode)
  @match e begin
    DAE.BINARY(exp1 = num, operator = DAE.DIV(__), exp2 = per) => begin
      local p = OMBackend._tryOr(() -> SimulationCode.tryEvalNumeric(per, simCode), nothing, :timePeriod)
      p === nothing && return nothing
      local s = @match num begin
        DAE.BINARY(exp1 = t, operator = DAE.SUB(__), exp2 = sExp) =>
          (_isTimeCref(t) ? OMBackend._tryOr(() -> SimulationCode.tryEvalNumeric(sExp, simCode), nothing, :timeShift) : nothing)
        _ => (_isTimeCref(num) ? 0.0 : nothing)
      end
      s === nothing && return nothing
      (Float64(s), Float64(p))
    end
    _ => nothing
  end
end

#= MSL 4.1's Pulse: `time >= (pre(count) + 1)*period + startTime`, a time event (4.0 had
   `integer((time - startTime)/period) > pre(count)`). The threshold, affine in pre(count),
   at counts 0, 1 and 2 gives (startTime, period), or nothing. As a generic when the pulse
   lost the end of most periods: the relation `time < T_start + T_width` jumped with T_start
   (the 4.0 path refreshes it, _collectIfCondRefresh). =#
function _affinePreThreshold(@nospecialize(thr::DAE.Exp), simCode::SimulationCode.SIM_CODE)
  local name = Ref("")
  local f = Float64[]
  for k in 0:2
    local s = _substitutePre(thr, Float64(k), name)
    s === nothing && return nothing
    local v = OMBackend._tryOr(() -> SimulationCode.tryEvalNumeric(s, simCode), nothing, :pulseThreshold)
    v === nothing && return nothing
    push!(f, Float64(v))
  end
  isempty(name[]) && return nothing
  local period = f[2] - f[1]
  (period > 0 && isapprox(f[3] - f[2], period; rtol = 1e-12)) || return nothing
  return (f[1] - period, period)
end

#= `e` with its pre(v) calls, all of one v (its name into `name`), replaced by the literal k;
   nothing for an expression other than arithmetic of crefs and literals. =#
function _substitutePre(@nospecialize(e::DAE.Exp), k::Float64, name::Base.RefValue{String})
  return @match e begin
    DAE.CALL(Absyn.IDENT("pre"), args, _) => begin
      local a = listArray(args)
      (length(a) == 1 && a[1] isa DAE.CREF) || return nothing
      local nm = string(a[1])
      (isempty(name[]) || name[] == nm) || return nothing
      name[] = nm
      DAE.RCONST(k)
    end
    DAE.BINARY(exp1 = a, operator = op, exp2 = b) => begin
      local sa = _substitutePre(a, k, name)
      local sb = sa === nothing ? nothing : _substitutePre(b, k, name)
      sb === nothing ? nothing : DAE.BINARY(sa, op, sb)
    end
    DAE.UNARY(operator = op, exp = a) => begin
      local sa = _substitutePre(a, k, name)
      sa === nothing ? nothing : DAE.UNARY(op, sa)
    end
    DAE.CAST(ty = ty, exp = a) => begin
      local sa = _substitutePre(a, k, name)
      sa === nothing ? nothing : DAE.CAST(ty, sa)
    end
    DAE.CREF(__) || DAE.RCONST(__) || DAE.ICONST(__) => e
    _ => nothing
  end
end

#= Detect the Modelica Source.Pulse / SignalSource periodic when-condition
   `integer((time - startTime)/period) <relop> pre(counter)`. Returns
   (startTime, period) or nothing. Such a condition is a periodic clock — it
   must fire AT t = startTime + n*period via a PeriodicCallback. The legacy
   ContinuousCallback rootfinds on the staircase `integer(...) > pre(count)`,
   a piecewise-constant 0/1 the rootfinder cannot reliably catch, so the
   pulse counter / T_start freeze (Blocks.Sources.Pulse and machines driven
   by it). =#
function _pulsePeriodicSpec(@nospecialize(cond), simCode)
  @match cond begin
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      local isGt = @match op begin
        DAE.GREATER(__) => true
        DAE.GREATEREQ(__) => true
        _ => false
      end
      isGt || return nothing
      _isTimeCref(e1) && return _affinePreThreshold(e2, simCode)
      _isPreCref(e2) || return nothing
      @match e1 begin
        DAE.CALL(Absyn.IDENT("integer"), arglst, _) => begin
          local args = listArray(arglst)
          length(args) == 1 ? _timeOffsetOverPeriod(args[1], simCode) : nothing
        end
        _ => nothing
      end
    end
    _ => nothing
  end
end

#= An if-condition that reads a discrete a periodic when updates (e.g. the Pulse
   `time < T_start + T_width`, T_start set at each period boundary) cannot be
   refreshed by its own MTK continuous callback: when T_start jumps the
   zero-crossing expression jumps across 0 with no smooth crossing for the
   rootfinder to catch. Collect, for each if-condition referencing a variable the
   when body writes, the assignment that re-derives its `ifCondNI` discrete
   parameter directly from the (now-current) condition value. The ifCond naming
   mirrors createIfEquations (`ifCond<sortIndex><branchIndex>`). =#
function _collectIfCondRefresh(writtenLHS::OrderedSet{String}, simCode)
  local refreshCrefs = Any[]
  local assigns = Expr[]
  isempty(simCode.ifEquations) && return (refreshCrefs, assigns)
  local sortedIfEqs = _sortedIfEquations(simCode)
  for (identifier, ifEq) in enumerate(sortedIfEqs)
    local i = 0
    for branch in ifEq.branches
      i += 1
      branch.identifier == -1 && continue
      local condDAE = SimulationCode.toDAEExp(branch.condition)
      local cCrefs = listArray(Util.getAllCrefs(condDAE))
      any(c -> string(c) in writtenLHS, cCrefs) || continue
      append!(refreshCrefs, cCrefs)
      local nameSym = Symbol("ifCond$(identifier)$(i)")
      push!(assigns, quote
        let _pidx = get(lookuptableParams, $(QuoteNode(nameSym)), nothing)
          if _pidx !== nothing
            integrator.p[_pidx] = ($(expToJuliaBoolMTK(condDAE, simCode)) ? 1.0 : 0.0)
          end
        end
      end)
    end
  end
  return (refreshCrefs, assigns)
end

#= Companion to _collectIfCondRefresh for DISCRETE-BOOL whens. A discrete-bool
   when whose condition relation reads a discrete the periodic body writes (e.g.
   BooleanPulse `y = time >= pulseStart and time < pulseStart + Twidth`, with
   pulseStart re-sampled each period) is lowered to MTK continuous callbacks that
   cannot catch the discontinuous threshold jump: when pulseStart jumps, the
   zero-crossing `time - pulseStart` jumps across 0 with no smooth crossing. So
   re-run such a when's body (its full rhs, the now-current threshold) inside the
   periodic callback to re-derive the dependent discrete at the jump. Returns the
   refresh crefs (rebound from the updated state) and the body statements. =#
function _collectDiscreteBoolWhenRefresh(writtenLHS::OrderedSet{String}, simCode)
  local refreshCrefs = Any[]
  local refreshStmts = Expr[]
  #= A discrete cluster needs none: the event iteration after the step sees
     its relation on the written discrete flip (exact, no hysteresis) and
     re-solves the cluster with the algebraic unknowns. =#
  local clusters = _usesDiscreteClusters(simCode)
  for weq in simCode.whenEquations
    _extractChangeRelations(weq.whenEquation.condition, simCode) === nothing && continue
    clusters && _gatherClusterAssigns(weq, simCode) !== nothing && continue
    local condDAE = SimulationCode.toDAEExp(weq.whenEquation.condition)
    local condCrefs = listArray(Util.getAllCrefs(condDAE))
    any(c -> string(c) in writtenLHS, condCrefs) || continue
    #= pre(v) from the periodic affect's snapshot (the values before the tick). =#
    append!(refreshStmts, Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
      createWhenStatementsMTK(weq.whenEquation.whenStmtLst, simCode)
    end)
    append!(refreshCrefs, condCrefs)
    for st in collect(weq.whenEquation.whenStmtLst)
      (st isa SimulationCode.ASSIGN || st isa BDAE.ASSIGN) || continue
      append!(refreshCrefs, listArray(Util.getAllCrefs(SimulationCode.toDAEExp(st.right))))
    end
  end
  return (refreshCrefs, refreshStmts)
end

#= Emit a PeriodicCallback for a Pulse-style periodic when: fire the body at the
   `integer((time-startTime)/period)` increments, i.e. at t = startTime + n*period
   for the n that fall in (tspan[1], stopTime]. `_firstEdge` is the first such
   instant assuming tspan[1] = 0 (the Modelica `integer` = floor convention), and
   it lies in (0, period], so it is a valid non-negative PeriodicCallback phase.
   `initial_affect = true` fires AT that first edge. A bare `phase = mod(startTime,
   period)` with `initial_affect = false` dropped the first edge whenever
   startTime < 0 (e.g. -0.035), shifting the whole pulse train by one period. =#
function _emitPulsePeriodicWhen(eq, simCode, callbacks::Int, startTime::Float64, period::Float64)
  local wEq = eq.whenEquation
  local _firstEdge = startTime + (floor(-startTime / period) + 1.0) * period
  #= pre(v) from the state before the instant (instantPre), as the periodic sample() path. =#
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(wEq.whenStmtLst, simCode)
  end
  local bodyCrefs = vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...)
  local writtenLHS = OrderedSet{String}()
  for wStmt in wEq.whenStmtLst
    (wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN) || continue
    for c in listArray(Util.getAllCrefs(SimulationCode.toDAEExp(wStmt.left)))
      push!(writtenLHS, string(c))
    end
  end
  local (refreshCrefs, refreshAssigns) = _collectIfCondRefresh(writtenLHS, simCode)
  quote
    let _affCache = Ref{Any}(nothing)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        OMBackend.CodeGeneration._isPeriodicTick(integrator, $(_firstEdge), $(period)) || return nothing
        local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
        local t = integrator.t
        local x = integrator.u
        local lookuptableStates
        local lookuptableParams
        if _affCache[] === nothing
          local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
          lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          _affCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _affCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        $(_whenLookupBindings(bodyCrefs, simCode)...)
        $(whenStmts...)
        #= Re-derive if-conditions that read the discretes just written. =#
        $(_whenLookupBindings(refreshCrefs, simCode)...)
        $(refreshAssigns...)
      end
    end
    #= Fire AT the first edge (phase in (0, period], non-negative) then every
       period. t0 is never an edge: the when fires only when the integer ratio
       increases, and an initial_affect body run would overwrite the
       init-algorithm phase (T_start := 0) of negative-startTime sources. =#
    $(Symbol("cb$(callbacks)")) = PeriodicCallback($(Symbol("affect$(callbacks)!")), $(period);
                                                   phase = $(_firstEdge), initial_affect = false,
                                                   final_affect = true)
  end
end

const _RELATION_WHEN_EXCLUDED_CALLS = ("pre", "edge", "change", "sample", "initial", "terminal", "der",
                                       "delay", "noEvent", "smooth", "reinit")

#= A when condition that is one relation `a op b` (op one of < <= > >=) whose
   operands call none of the operators above: (a, b, isLess, strict), or
   nothing. =#
function _singleRelationWhen(@nospecialize(cond))
  @match cond begin
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      local kind = @match op begin
        DAE.LESS(__) => (true, true)
        DAE.LESSEQ(__) => (true, false)
        DAE.GREATER(__) => (false, true)
        DAE.GREATEREQ(__) => (false, false)
        _ => nothing
      end
      kind === nothing && return nothing
      local plain = Ref(true)
      local visit = function (e, arg)
        @match e begin
          DAE.CALL(Absyn.IDENT(name), _, _) where (name in _RELATION_WHEN_EXCLUDED_CALLS) => begin
            plain[] = false
            return (e, false, arg)
          end
          DAE.CALL(path, _, _) where _isDelayCall(path) => begin
            plain[] = false
            return (e, false, arg)
          end
          _ => return (e, true, arg)
        end
      end
      Util.traverseExpTopDown(e1, visit, nothing)
      Util.traverseExpTopDown(e2, visit, nothing)
      plain[] ? (e1, e2, kind[1], kind[2]) : nothing
    end
    _ => nothing
  end
end

#= A when-equation on one relation (relationRefresh.jl relationWhenCallback):
   the relation is buffered, set literally at the start, with a hysteresis,
   and the body runs only when it becomes true. `zc` is `a - b` for < and <=,
   `b - a` for > and >=, so the relation is true when zc < 0 (or <= 0). =#
function _emitRelationWhen(eq, simCode, callbacks::Int, rel)
  local wEq = eq.whenEquation
  local (e1, e2, isLess, strict) = rel
  local sub = DAE.SUB(DAE.T_REAL_DEFAULT)
  local zcDAE = isLess ? DAE.BINARY(e1, sub, e2) : DAE.BINARY(e2, sub, e1)
  local names = String[]
  for c in listArray(Util.getAllCrefs(DAE.BINARY(e1, sub, e2)))
    local n = string(c)
    (n == "time" || n in names) && continue
    local entry = get(simCode.stringToSimVarHT, n, nothing)
    (entry === nothing || entry[2].varKind isa SimulationCode.STRING) && continue
    push!(names, n)
  end
  local args = Symbol[Symbol(n) for n in names]
  #= The body runs in the event iteration: pre(v) from the sweep's snapshot. =#
  local whenStmts = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
    createWhenStatementsMTK(wEq.whenStmtLst, simCode)
  end
  local bodyCrefs = vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...)
  quote
    $(Symbol("cb$(callbacks)")) = let _affCache = Ref{Any}(nothing)
      local _eval = (t, $(args...)) -> (Float64($(expToJuliaExpMTK(zcDAE, simCode))),
                                        1.0 + max(abs(Float64($(expToJuliaExpMTK(e1, simCode)))),
                                                  abs(Float64($(expToJuliaExpMTK(e2, simCode))))))
      local _body! = (integrator, $(MTK_CodeGenerationUtil.PRE_SNAPSHOT)) -> begin
        local t = integrator.t
        local x = integrator.u
        local lookuptableStates
        local lookuptableParams
        if _affCache[] === nothing
          local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
          lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          _affCache[] = (lookuptableStates, lookuptableParams)
        else
          local cached = _affCache[]
          lookuptableStates = cached[1]
          lookuptableParams = cached[2]
        end
        $(_whenLookupBindings(bodyCrefs, simCode)...)
        $(whenStmts...)
        nothing
      end
      OMBackend.CodeGeneration.relationWhenCallback($(names), _eval, $(strict), _body!)
    end
  end
end

#= Whether the time is a tick of a PeriodicCallback, t0 + phase + k*period.
   With final_affect the callback also runs when the integration ends: a tick
   there runs (omc samples at the stop time too: n(1) = 6 for ticks at 0.5,
   ..., 1.0; the callback left out the final time), any other end not. =#
function _isPeriodicTick(integrator, phase, period)::Bool
  local n = (integrator.t - first(integrator.sol.prob.tspan) - phase) / period
  return n >= -1e-9 && abs(n - round(n)) <= 1e-9 * max(1.0, abs(n))
end

#= `(sample call, guard)` of a when condition that is `sample(...)` or a
   conjunction with exactly one sample() conjunct (the guard: the other
   conjuncts, or nothing); `(nothing, nothing)` otherwise. =#
function _splitSampleCondition(cond)
  local isSample = e -> e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name == "sample"
  local conjuncts = Any[]
  local collect! = nothing
  collect! = e -> begin
    @match e begin
      DAE.LBINARY(exp1 = e1, operator = DAE.AND(__), exp2 = e2) => (collect!(e1); collect!(e2))
      _ => push!(conjuncts, e)
    end
    nothing
  end
  collect!(cond)
  local samples = filter(isSample, conjuncts)
  length(samples) == 1 || return (nothing, nothing)
  local rest = filter(!isSample, conjuncts)
  any(e -> _containsSampleCall(e), rest) && return (nothing, nothing)
  isempty(rest) && return (samples[1], nothing)
  local guard = rest[1]
  for e in rest[2:end]
    guard = DAE.LBINARY(guard, DAE.AND(DAE.T_BOOL(MetaModelica.Nil())), e)
  end
  return (samples[1], guard)
end

_containsSampleCall(e) = _containsCallTo(e, "sample")
_containsCallTo(e, name::String) = any(c -> _isCallNamed(c, name), _allCalls(e))
_isCallNamed(@nospecialize(e), name::String) = e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name == name
function _allCalls(e)
  local out = Any[]
  Util.traverseExpBottomUp(e, (x, acc) -> (x isa DAE.CALL && push!(out, x); (x, true, acc)), 0)
  return out
end

"""
  This function creates a representation of a when equation in Julia.
"""
function eqToJulia(eq::Union{BDAE.WHEN_EQUATION, SimulationCode.WHEN_EQUATION}, simCode::SimulationCode.SIM_CODE, arrayIdx::Int)::Expr
  local wEq = eq.whenEquation
  local wEqCondDAE = SimulationCode.toDAEExp(wEq.condition)
  local cond = transformToZeroCrossingCondition(wEqCondDAE)
  ADD_CALLBACK()
  local callbacks = COUNT_CALLBACKS()
  #=
    Find the type of the condition.
    For continuous variables we should create continuous callbacks.
    However, for discrete conditions we should create discrete callbacks.
  =#
  #=
    Get all component references.
    If this set is empty it means that we have a condition involving continuous time
  =#
  #= sample(start, interval), alone or and-ed with a guard (MSL PartialNoise:
     `when generateNoise and sample(startTime, samplePeriod)`): periodic, the
     guard read at each tick. =#
  local (sampleCall, sampleGuard) = _splitSampleCondition(wEqCondDAE)
  #= Lowered elsewhere: a sample() alone or and-ed with a guard, terminal()
     alone (after the solve). Not in MSL 3.2.3; sample() read false, terminal()
     undefined: never fired. =#
  sampleCall === nothing && _containsSampleCall(wEqCondDAE) &&
    unsupported("sample() under or/not, or two sample() calls, in a when-condition", wEqCondDAE)
  _containsCallTo(wEqCondDAE, "terminal") && !_isCallNamed(wEqCondDAE, "terminal") &&
    unsupported("terminal() with another trigger in a when-condition", wEqCondDAE)
  local isPeriodic = sampleCall !== nothing
  local isContinuousCond::Bool = isContinuousCondition(wEqCondDAE, simCode)
  #= Table / time-driven sources: a when whose condition is purely change(time>=c)
     thresholds must fire AT those times via PresetTimeCallback — a ContinuousCallback
     rootfinding on the spiky change() value never detects the crossings. =#
  if !isPeriodic
    local _thr = _collectTimeThresholds(wEqCondDAE, simCode)
    if _thr !== nothing && !isempty(_thr)
      return _emitPresetTimeWhen(eq, simCode, callbacks, _thr)
    end
    #= Source.Pulse periodic clock `integer((time-startTime)/period) > pre(count)`:
       fire AT the period boundaries via PeriodicCallback. =#
    local _pulse = _pulsePeriodicSpec(wEqCondDAE, simCode)
    if _pulse !== nothing && _pulse[2] > 0.0
      return _emitPulsePeriodicWhen(eq, simCode, callbacks, _pulse[1], _pulse[2])
    end
  end
  #= A when on one relation: buffered relation with a hysteresis (MLS 8.5). =#
  if isContinuousCond && !isPeriodic && wEq.elsewhenPart === nothing
    local rel = _singleRelationWhen(wEqCondDAE)
    rel === nothing || return _emitRelationWhen(eq, simCode, callbacks, rel)
  end
  #= A `sample(start, period)` is a periodic clock even when its interval is a
     parameter, which isContinuousCondition mis-flags as continuous; keep all
     samples on the periodic branch. =#
  if isContinuousCond && !isPeriodic
    local isElseIf = if wEq.elsewhenPart !== nothing
      local elsePart = _elsewhenInner(wEq.elsewhenPart)
      local elseCond = SimulationCode.toDAEExp(_elsewhenCondition(elsePart))
      cond2 = transformToZeroCrossingCondition(elseCond)
      cond == cond2
    else
      false
    end
    if isElseIf
      #= Use MTK-aware runtime symbol lookup for the elseif continuous path.
         The hardcoded x[N] indices from expToJuliaExp become invalid after
         MTK structural_simplify reorders unknowns. =#
      #= pre(v) from the state before the instant (instantPre) (an earlier statement's new value is not pre(v)). =#
      local (whenStatementsMTKIf, whenStatementsMTKElse) = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
        (createWhenStatementsMTK(wEq.whenStmtLst, simCode), createWhenStatementsMTK(_elsewhenStmtLst(elsePart), simCode))
      end
      local condCrefsElseIf = filter(c -> string(c) != "time", listArray(Util.getAllCrefs(cond)))
      quote
        let _condCache = Ref{Any}(nothing)
          global $(Symbol("condition$(callbacks)"))
          $(Symbol("condition$(callbacks)")) = (x, t, integrator) -> begin
            local lookuptableStates
            local lookuptableParams
            if _condCache[] === nothing
              local xs = $(map(x -> Symbol(string(x)), condCrefsElseIf))
              local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
              if !isempty(xs) && all(isnothing, indices)
                return 1.0
              end
              lookuptableStates = isempty(xs) ? Dict{Symbol,Union{Nothing,Int}}() : Dict(xs .=> indices)
              local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
              lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
              _condCache[] = (lookuptableStates, lookuptableParams)
            else
              local cached = _condCache[]
              lookuptableStates = cached[1]
              lookuptableParams = cached[2]
            end
            $(_whenLookupBindings(Util.getAllCrefs(cond), simCode)...)
            $(expToJuliaExpMTK(cond, simCode))
          end
        end
        let _affCache = Ref{Any}(nothing)
          global $(Symbol("affect$(callbacks)!"))
          $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
            local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
            local t = integrator.t + integrator.dt
            local x = integrator.u
            local lookuptableStates
            local lookuptableParams
            if _affCache[] === nothing
              local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
              local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
              lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
              lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
              _affCache[] = (lookuptableStates, lookuptableParams)
            else
              local cached = _affCache[]
              lookuptableStates = cached[1]
              lookuptableParams = cached[2]
            end
            $(_whenLookupBindings(vcat(
                    listArray(Util.getAllCrefs(wEqCondDAE)),
                    vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...),
                    vcat(map(x -> getRHSVariables(x), _elsewhenStmtLst(elsePart))...)
                  ), simCode)...)
            if integrator.dt == 0.0
              @error "integrator.dt was zero. Aborting."
              fail()
            end
            if $(expToJuliaBoolMTK(wEqCondDAE, simCode))
              $(whenStatementsMTKIf...)
              add_tstop!(integrator, integrator.t + 1E-12) #=TODO: Some small number for now=#
            else
              $(whenStatementsMTKElse...)
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
    else #= No elseif =#
      #= pre(v) from the state before the instant (instantPre): `b = not pre(c); k = if pre(c) ...`
         with c = b read b's new value (k = 20, OpenModelica 10). =#
      whenStatementsMTK = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
        createWhenStatementsMTK(wEq.whenStmtLst, simCode)
      end
      local cond = quote
        let _condCache = Ref{Any}(nothing)
          global $(Symbol("condition$(callbacks)"))
          $(Symbol("condition$(callbacks)")) = (x, t, integrator) -> begin
            local NO_TRIGGER = 1.0
            local lookuptableStates
            local lookuptableParams
            if _condCache[] === nothing
              local xs = $(map(x -> Symbol(string(x)), filter(c -> string(c) != "time", listArray(Util.getAllCrefs(cond)))))
              local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
              local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
              local paramIdxs = indexin(xs, params)
              #= Short-circuit only when NONE of the non-time crefs resolve to
                 either a state or a parameter — i.e. the cref is genuinely
                 unknown and the condition cannot be evaluated. A cref that
                 lives in `params` (e.g. `x_table_t[1]` in
                 `when time >= x_table_t[1]`) is perfectly fine to read via
                 the param lookup table, so the callback should still run. =#
              if !isempty(xs) && all(isnothing, indices) && all(isnothing, paramIdxs)
                @debug "[CB-CC$($(callbacks)) cond] NO_TRIGGER (no state/param mapping)" t xs
                return NO_TRIGGER
              end
              lookuptableStates = Dict((xs) .=> indices)
              lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
              _condCache[] = (lookuptableStates, lookuptableParams)
            else
              local cached = _condCache[]
              lookuptableStates = cached[1]
              lookuptableParams = cached[2]
            end
            $(_whenLookupBindings(Util.getAllCrefs(cond), simCode)...)
            local _result = $(expToJuliaExpMTK(cond, simCode))
            @debug "[CB-CC$($(callbacks)) cond] eval" t value=_result
            _result
          end
        end
      end
      local affect = quote
        let _affCache = Ref{Any}(nothing)
          global $(Symbol("affect$(callbacks)!"))
          $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
            local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
            local t = integrator.t + integrator.dt
            local x = integrator.u
            @debug "[CB-CC$($(callbacks)) affect] firing" t=integrator.t
            local lookuptableStates
            local lookuptableParams
            if _affCache[] === nothing
              local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
              local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
              lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
              lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
              _affCache[] = (lookuptableStates, lookuptableParams)
            else
              local cached = _affCache[]
              lookuptableStates = cached[1]
              lookuptableParams = cached[2]
            end
            $(_whenLookupBindings(vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...), simCode)...)
            if integrator.dt == 0.0
              @error "integrator.dt was zero. Aborting."
              fail()
            end
            $(whenStatementsMTK...)
            @debug "[CB-CC$($(callbacks)) affect] done" t=integrator.t u=copy(integrator.u)
          end
        end
      end
      if wEq.elsewhenPart !== nothing
        #= AUDIT (ombackend-bug-audit-2026-06-05 #11): a continuous when/elsewhen
           whose arm conditions differ (the normal elsewhen case) is lowered to
           independent ContinuousCallbacks via the recursion below. There is no
           shared per-instant exclusivity guard, so Modelica elsewhen ordering is
           honoured for non-simultaneous events (correct, since exclusivity is
           per-instant) but NOT when both arm conditions cross zero at the same
           instant: both bodies execute, last-writer-wins on a shared discrete.
           Measure-zero in practice and confined to this legacy DE callback path
           (the modern MTK path bakes events into the problem). Warned so the
           limitation is attributable; a shared fired-at-instant guard threaded
           through the recursion is the proper fix. =#
        @warn "[CodeGen: continuous when/elsewhen] $(simCode.name): continuous `when`/`elsewhen` with distinct arm conditions is lowered to independent ContinuousCallbacks; elsewhen mutual-exclusion is NOT enforced when both arm conditions cross zero at the same instant (simultaneous-event edge case). See ombackend-bug-audit-2026-06-05 #11."
      end
      quote
        $cond
        $affect
        #= RightRootFind: the event lands just past the root, where the
           condition has changed sign. From the left of it the next step
           finds the same crossing again (DiffEqBase 7 no longer suppresses
           that repeat: ElseWhenBasic looped at x = 0.3 until MaxIters).
           No `affect_neg!` set: transformToZeroCrossingCondition has already
           encoded direction (positive→negative = trigger) so the same
           Modelica `when cond then` semantics fall on `affect!` only. Setting
           `affect_neg! = affect!` would double-fire on each oscillation
           (classic bouncing-ball: downcrossing reinit then upcrossing reinit
           again at the same event), driving Zeno / maxiters. =#
        $(Symbol("cb$(callbacks)")) = ContinuousCallback($(Symbol("condition$(callbacks)")),
                                                         $(Symbol("affect$(callbacks)!")),
                                                         rootfind = ModelingToolkit.SciMLBase.RightRootFind, save_positions=(true, true))
        $(if wEq.elsewhenPart !== nothing
            eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 0)
          end)
      end
    end
  elseif isPeriodic
    @match DAE.CALL(Absyn.IDENT("sample"), args, attrs) = sampleCall
    @match start <| interval <| tail = args
    #= MTK-aware periodic affect: the hardcoded x[N]/p[N] indices from
       expToJuliaExp are invalid after MTK structural_simplify reorders unknowns,
       so resolve the interval to a literal Δt and write state via
       getStatesAsSymbols + lookuptable, mirroring the discrete branch. =#
    #= pre(v) from the state before the instant (instantPre): a body reading pre(c) after setting its
       alias b (`b = not pre(c); k = if pre(c) ...`) read the new value. =#
    local whenStatementsMTKPeriodic = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
      createWhenStatementsMTK(wEq.whenStmtLst, simCode)
    end
    #= Refresh discrete-bool whens whose condition reads a discrete this periodic
       body writes (e.g. BooleanPulse `y` reads the re-sampled `pulseStart`): their
       own continuous callbacks cannot catch the threshold jump, so re-derive them
       here from the now-current threshold. =#
    local _periodicWrittenLHS = OrderedSet{String}()
    for wStmt in wEq.whenStmtLst
      (wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN) || continue
      for c in listArray(Util.getAllCrefs(SimulationCode.toDAEExp(wStmt.left)))
        push!(_periodicWrittenLHS, string(c))
      end
    end
    local (_dbRefreshCrefs, _dbRefreshStmts) = _collectDiscreteBoolWhenRefresh(_periodicWrittenLHS, simCode)
    local _intervalVal = SimulationCode.tryEvalNumeric(interval, simCode)
    local _dtExpr = _intervalVal === nothing ? expToJuliaExp(interval, simCode) : _intervalVal
    #= The ticks are start + i*interval, i = 0, 1, ...: the phase is the first
       one from the initial time (taken as 0). A start not known here was taken
       as 0 (generated code cannot read the parameters at module level), a
       negative one as 0 too (sample(-0.15, 0.25) ticked at 0.25, not 0.1). =#
    local _startVal = SimulationCode.tryEvalNumeric(start, simCode)
    _startVal === nothing && unsupported("a sample() start not known at the build", start)
    local _firstTick = Float64(_startVal)
    if _firstTick < 0
      _intervalVal === nothing && unsupported("a negative sample() start with an interval not known at the build", sampleCall)
      _firstTick += ceil(-_firstTick / Float64(_intervalVal)) * Float64(_intervalVal)
      #= Rounding: sample(-0.9, 0.3) gave -1.1e-16, a negative phase (an
         ArgumentError) and no tick at 0. =#
      abs(_firstTick) <= 8 * eps(max(1.0, abs(Float64(_intervalVal)))) && (_firstTick = 0.0)
      _firstTick = max(0.0, _firstTick)
    end
    local _phaseExpr = _firstTick
    #= A tick at the initial time is taken too, after initialization (omc,
       Dymola): MSL RealFFT1's `when sample(0, Ts)` samples y(0) into its FFT
       buffer. =#
    local _initialTick = iszero(_firstTick)
    local _guardCrefs = sampleGuard === nothing ? DAE.ComponentRef[] : listArray(Util.getAllCrefs(sampleGuard))
    local _guardExpr = sampleGuard === nothing ? true : expToJuliaExpMTK(sampleGuard, simCode)
    quote
      let _affCache = Ref{Any}(nothing)
        global $(Symbol("affect$(callbacks)!"))
        $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
          OMBackend.CodeGeneration._isPeriodicTick(integrator, $(Symbol("samplePhase$(callbacks)")),
                                                   $(Symbol("sampleDt$(callbacks)"))) || return nothing
          local $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)
          local t = integrator.t
          local x = integrator.u
          local p = integrator.p
          local lookuptableStates
          local lookuptableParams
          if _affCache[] === nothing
            local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
            local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
            lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
            lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
            _affCache[] = (lookuptableStates, lookuptableParams)
          else
            local cached = _affCache[]
            lookuptableStates = cached[1]
            lookuptableParams = cached[2]
          end
          $(_whenLookupBindings(_guardCrefs, simCode)...)
          $(_guardExpr) == true || return nothing
          $(_whenLookupBindings(vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...), simCode)...)
          $(whenStatementsMTKPeriodic...)
          #= Re-derive dependent discrete-bool whens from the just-written threshold. =#
          $(_whenLookupBindings(_dbRefreshCrefs, simCode)...)
          $(_dbRefreshStmts...)
        end
      end
      $(Symbol("sampleDt$(callbacks)")) = $(_dtExpr)
      $(Symbol("samplePhase$(callbacks)")) = $(_phaseExpr)
      #= Ticks at start + i*interval, the initial and the final time included
         (_isPeriodicTick). =#
      $(Symbol("cb$(callbacks)")) = PeriodicCallback($(Symbol("affect$(callbacks)!")), $(Symbol("sampleDt$(callbacks)"));
                                                      phase = $(Symbol("samplePhase$(callbacks)")),
                                                      initial_affect = $(_initialTick), final_affect = true,
                                                      save_positions = (true, true))
      $(if wEq.elsewhenPart !== nothing
          eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 4)
        end)
    end
  else #= If none of the variables in the condition was continuous.. =#
    #= Use MTK-aware runtime symbol lookup for discrete callbacks.
       The hardcoded x[N] indices from expToJuliaExp become invalid after
       MTK structural_simplify reorders unknowns. Mirror the continuous
       callback path (above) which uses getStatesAsSymbols + lookuptable. =#
    #= pre(v) from a snapshot: the event iteration passes the state before its
       sweep (the algebraic unknowns are solved again before the discrete whens
       run, and `iAtOpen = pre(i)` read the current i). =#
    whenStatementsMTKDiscrete = Base.ScopedValues.with(MTK_CodeGenerationUtil.PRE_FROM_SNAPSHOT => true) do
      createWhenStatementsMTK(wEq.whenStmtLst, simCode)
    end
    #= An elsewhen arm `time >= thr` with a runtime-discrete threshold is a
       scheduled time event: the parent affect (which assigns the threshold)
       adds a tstop at it, and the arm itself is emitted as an edge-guarded
       DiscreteCallback ordered after the parent (see
       _emitElsewhenThresholdTimeWhen). =#
    local _ewArm = _elsewhenInner(wEq.elsewhenPart)
    local _ewThrDAE = _ewArm === nothing ? nothing :
      _discreteTimeEventThreshold(SimulationCode.toDAEExp(_elsewhenCondition(_ewArm)), simCode)
    local _ewRefSym = Symbol("ewLastThr$(callbacks)")
    local _ewDecl = _ewThrDAE === nothing ? :() : :(local $(_ewRefSym) = Ref(NaN))
    local _ewSchedule = if _ewThrDAE === nothing
      :()
    else
      quote
        $(_whenLookupBindings(listArray(Util.getAllCrefs(_ewThrDAE)), simCode)...)
        local _ewThr = Float64($(expToJuliaExpMTK(_ewThrDAE, simCode)))
        if _ewThr > integrator.t
          add_tstop!(integrator, _ewThr)
        else
          $(_ewRefSym)[] = _ewThr
        end
      end
    end
    local condCrefs = filter(c -> string(c) != "time", listArray(Util.getAllCrefs(cond)))
    #= From the condition as written: the zero-crossing form unwraps a
       top-level `change(k)` to `k`. =#
    local _bareCrefStrs = _collectBareCrefStrings(wEqCondDAE)
    #= A condition on a Boolean (`when u`) is true for as long as u is, but the when
       fires once, when it becomes true: an edge latch. The condition fires only
       while the latch is clear and clears it when it reads false; the affect sets
       it to the condition's value after the event (change() terms are false again
       then), and the start of a solve to its value on the initialized state (a
       Boolean true from the start is no edge). The condition only clears it: the
       event iteration evaluates a condition several times before running the affect.
       Setting u itself to false instead corrupted it where it is read elsewhere
       (the MSL Timer's input, a threshold block's output, read by
       `y = if u then time - entryTime ...`). =#
    local useLatch = any(condCrefs) do c
      local entry = get(simCode.stringToSimVarHT, string(c), nothing)
      string(c) in _bareCrefStrs && entry !== nothing && !SimulationCode.isParameter(entry[2])
    end
    local condValue = quote
      $(_whenLookupBindings(Util.getAllCrefs(cond), simCode)...)
      local _r = $(expToJuliaBoolMTK(wEqCondDAE, simCode; cachedChange = true))
      #= DiscreteCallback condition must return Bool per SciMLBase. Modelica
         Boolean discrete states are stored as Float64 in `integrator.u`
         (0.0/1.0), so a bare cref read returns Float64 and triggers
         "TypeError: non-boolean (Float64) used in boolean context" in
         SciML's callback dispatch (affects PowerConverters Thyristor
         models). Cast via `!= 0` so any numeric cref-as-condition
         evaluates correctly. Bool results pass through unchanged. =#
      _r isa Bool ? _r : (_r != 0)
    end
    local changeInitExprs = map(condCrefs) do c
      local cStr = string(c)
      local entry = get(simCode.stringToSimVarHT, cStr, nothing)
      if entry !== nothing && !SimulationCode.isParameter(entry[2])
        local sym = Symbol(entry[2].name)
        quote
          let _idx = get(lookuptableStates, $(QuoteNode(sym)), nothing)
            if _idx !== nothing
              _changePreValues[$(QuoteNode(sym))] = x[_idx]
            end
          end
        end
      else
        :()
      end
    end
    local changeUpdateExprs = map(condCrefs) do c
      local cStr = string(c)
      local entry = get(simCode.stringToSimVarHT, cStr, nothing)
      if entry !== nothing && !SimulationCode.isParameter(entry[2])
        local sym = Symbol(entry[2].name)
        quote
          let _idx = get(lookuptableStates, $(QuoteNode(sym)), nothing)
            if _idx !== nothing
              _changePreValues[$(QuoteNode(sym))] = integrator.u[_idx]
            end
          end
        end
      else
        :()
      end
    end
    local changeSeedPairs = Expr[]
    for c in condCrefs
      local cStr = string(c)
      local entry = get(simCode.stringToSimVarHT, cStr, nothing)
      if entry !== nothing && !SimulationCode.isParameter(entry[2])
        local sym = Symbol(entry[2].name)
        local lit = _readStartAttributeAsLiteral(entry[2])
        push!(changeSeedPairs, :($(QuoteNode(sym)) => $(lit)))
      end
    end
    quote
      $(_ewDecl)
      #= Typed caches: the condition runs after every step. =#
      let _condCache = Ref{Union{Nothing, Tuple{Dict{Symbol, Union{Nothing, Int}}, Dict{Symbol, Int}}}}(nothing),
          _affCache = Ref{Union{Nothing, Tuple{Dict{Symbol, Int}, Dict{Symbol, Int}}}}(nothing),
          _changeCache = Ref{Union{Nothing, Dict{Symbol, Any}}}(nothing),
          _latch = Ref{Bool}(false),     # the edge latch
          _changeSeedValues = Dict{Symbol, Any}($(changeSeedPairs...))
        global $(Symbol("condition$(callbacks)"))
        #= Whether the when fires. `_follow`: whether change()/edge() may take the values now,
           where it does not fire (false within an event iteration's sweep; see below). =#
        $(Symbol("condition$(callbacks)")) = (x, t, integrator, _follow::Bool = true) -> begin
          local lookuptableStates
          local lookuptableParams
          if _condCache[] === nothing
            local xs = $(map(c -> Symbol(string(c)), condCrefs))
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
          local _fires = $(useLatch ? :(_result && !_latch[]) : :_result)
          #= change()/edge() compare with the values at the previous event (pre()): where the when
             does not fire, the memory follows the values now (off turned false at 1/12 without
             firing a `when edge(off)`, which then missed off's edge at 5/12); where it fires, its
             affect does. Not within a sweep of the event iteration: an algebraic value the next
             sweep solves again is stale there, and `edge(b) and c` would lose b's edge. =#
          $(isempty(changeSeedPairs) ? :() : :(_follow && !_fires && begin $(changeInitExprs...) end))
          _fires
        end
        global $(Symbol("affect$(callbacks)!"))
        $(Symbol("affect$(callbacks)!")) = (integrator, $(MTK_CodeGenerationUtil.PRE_SNAPSHOT) = OMBackend.CodeGeneration.instantPre(integrator)) -> begin
          local t = integrator.t
          local x = integrator.u
          @debug "[CB-DC$($(callbacks)) affect] firing" t=integrator.t
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
          $(_whenLookupBindings(vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...), simCode)...)
          $(whenStatementsMTKDiscrete...)
          $(_ewSchedule)
          auto_dt_reset!(integrator)
          add_tstop!(integrator, integrator.t + 1E-12)
          local _changePreValues = _changeCache[] === nothing ? Dict{Symbol, Any}() : _changeCache[]
          _changeCache[] = _changePreValues
          $(changeUpdateExprs...)
          #= On the state vector: the body bound the model's variables by name
             (a variable `x` replaced the state vector `x`: a BoundsError). =#
          $(useLatch ? :(_latch[] = let x = integrator.u; $(condValue) end) : :())
          @debug "[CB-DC$($(callbacks)) affect] done" t=integrator.t u=copy(integrator.u)
        end
        #= At the start of a solve: change()/edge() compare with the initialized values (pre() at the
           first event), not the start attributes: `when edge(off)` with off(start = true) initialized
           false never fired (the MSL switch with arc: its arc voltage never started ramping). And the
           edge latch from the initialized state. =#
        global $(Symbol("initialize$(callbacks)!"))
        #= The latch implies a seed: it is used for a non-parameter cref of the condition. =#
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
      end
      #= Part of the event iteration where the model has buffered relations. =#
      $(Symbol("cb$(callbacks)")) = OMBackend.CodeGeneration.discreteWhenCallback($(Symbol("condition$(callbacks)")),
                                                                                $(Symbol("affect$(callbacks)!")),
                                                                                $(Symbol("initialize$(callbacks)!")))
      $(if _ewThrDAE !== nothing
          _emitElsewhenThresholdTimeWhen(_ewArm, simCode, _ewRefSym, _ewThrDAE)
        elseif wEq.elsewhenPart !== nothing
          eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 4)
        end)
    end
  end
end


#= SimCode-Exp entry: per-variant dispatch mirrors the DAE.Exp emitter;
   only the EXP_CREF leaf, CALL args, and CAST touch a per-node DAE projection. =#
"""
  Converts a DAE expression into a Julia expression
  $(SIGNATURES)
The context can be any type that contains a set of residual equations.
"""
function expToJuliaExp(e::SimulationCode.BCONST, context, varSuffix=""; varPrefix="x")::Expr
  quote $(e.value) end
end
function expToJuliaExp(e::SimulationCode.ICONST, context, varSuffix=""; varPrefix="x")::Expr
  quote $(e.value) end
end
function expToJuliaExp(e::SimulationCode.RCONST, context, varSuffix=""; varPrefix="x")::Expr
  quote $(e.value) end
end
function expToJuliaExp(e::SimulationCode.SCONST, context, varSuffix=""; varPrefix="x")::Expr
  quote $(e.value) end
end

function expToJuliaExp(e::SimulationCode.EXP_CREF, context, varSuffix=""; varPrefix="x")::Expr
  local hashTable = context.stringToSimVarHT
  local varName = SimulationCode.string(SimulationCode.toDAECref(e.cref).componentRef)
  if varName == "time"
    return quote t end
  end
  local indexAndVar = hashTable[varName]
  local varKind::SimulationCode.SimVarType = indexAndVar[2].varKind
  @match varKind begin
    SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
    SimulationCode.STATE(__) => quote
      $(LineNumberNode(@__LINE__, "$varName state"))
      $(Symbol(varPrefix))[$(indexAndVar[1])]
    end
    SimulationCode.PARAMETER(__) => quote
      $(LineNumberNode(@__LINE__, "$varName parameter"))
      p[$(indexAndVar[1])]
    end
    SimulationCode.ALG_VARIABLE(__) => quote
      $(LineNumberNode(@__LINE__, "$varName, algebraic"))
      $(Symbol(varPrefix))[$(indexAndVar[1])]
    end
    SimulationCode.DISCRETE(__) => quote
      $(LineNumberNode(@__LINE__, "$varName, Discrete"))
      $(Symbol(varPrefix))[$(indexAndVar[1])]
    end
    SimulationCode.STATE_DERIVATIVE(__) => :(dx$(varSuffix)[$(indexAndVar[1])] #= der($varName) =#)
    SimulationCode.DATA_STRUCTURE(__) => quote
      $(LineNumberNode(@__LINE__, "$varName, datastructure"))
      $(Symbol(indexAndVar[2].name))
    end
    SimulationCode.STRING(__) => quote
      $(LineNumberNode(@__LINE__, "$varName, string"))
      $(Symbol(indexAndVar[2].name))
    end
  end
end

function expToJuliaExp(e::SimulationCode.UNARY, context, varSuffix=""; varPrefix="x")::Expr
  local o = opKindToJuliaOperator(e.op)
  quote
    $(o)($(expToJuliaExp(e.exp, context, varPrefix=varPrefix)))
  end
end
function expToJuliaExp(e::SimulationCode.BINARY, context, varSuffix=""; varPrefix="x")::Expr
  local a = expToJuliaExp(e.exp1, context, varPrefix=varPrefix)
  local b = expToJuliaExp(e.exp2, context, varPrefix=varPrefix)
  local o = opKindToJuliaOperator(e.op)
  quote
    $o($(a), $(b))
  end
end
function expToJuliaExp(e::SimulationCode.LUNARY, context, varSuffix=""; varPrefix="x")::Expr
  local lhs = expToJuliaExp(e.exp, context, varPrefix=varPrefix)
  local o = opKindToJuliaOperator(e.op)
  quote
    $o($(lhs))
  end
end
function expToJuliaExp(e::SimulationCode.LBINARY, context, varSuffix=""; varPrefix="x")::Expr
  local l = expToJuliaExp(e.exp1, context, varPrefix=varPrefix)
  local o = opKindToJuliaOperator(e.op)
  local r = expToJuliaExp(e.exp2, context, varPrefix=varPrefix)
  quote
    $o($(l), $(r))
  end
end
function expToJuliaExp(e::SimulationCode.RELATION, context, varSuffix=""; varPrefix="x")::Expr
  local lhs = expToJuliaExp(e.exp1, context, varPrefix=varPrefix)
  local o = opKindToJuliaOperator(e.op)
  local rhs = expToJuliaExp(e.exp2, context, varPrefix=varPrefix)
  quote
    $o($(lhs), $(rhs))
  end
end
function expToJuliaExp(e::SimulationCode.IFEXP, context, varSuffix=""; varPrefix="x")::Expr
  local condJL = expToJuliaExp(e.cond, context, varPrefix=varPrefix)
  local thenJL = expToJuliaExp(e.thenExp, context, varPrefix=varPrefix)
  local elseJL = expToJuliaExp(e.elseExp, context, varPrefix=varPrefix)
  :(ifelse($(condJL), $(thenJL), $(elseJL)))
end
function expToJuliaExp(e::SimulationCode.CALL, context, varSuffix=""; varPrefix="x")::Expr
  local hashTable = context.stringToSimVarHT
  @match e.path begin
    Absyn.IDENT(nm) => begin
      local daeArgs = MetaModelica.list((SimulationCode.toDAEExp(a) for a in e.args)...)
      DAECallExpressionToJuliaCallExpression(nm, daeArgs, context, hashTable, varPrefix=varPrefix)
    end
    _ => begin
      local expr = Expr(:call, Symbol(string(e.path)))
      local args::Vector{Any} = Any[]
      for arg in e.args
        push!(args, expToJuliaExp(arg, context, varSuffix, varPrefix=varPrefix))
      end
      append!(expr.args, args)
      expr
    end
  end
end
function expToJuliaExp(e::SimulationCode.CAST, context, varSuffix=""; varPrefix="x")::Expr
  quote
    $(generateCastExpression(SimulationCode.toDAEType(e.ty), SimulationCode.toDAEExp(e.exp), context, varPrefix))
  end
end
Base.@nospecializeinfer function expToJuliaExp(@nospecialize(exp::SimulationCode.Exp),
                                               @nospecialize(context),
                                               varSuffix = ""; varPrefix = "x")::Expr
  unsupported("expression", exp)
end

function expToJuliaExp(exp::DAE.Exp, context, varSuffix=""; varPrefix="x")::Expr
  hashTable = context.stringToSimVarHT
  local expr::Expr = begin
    local int::Int64
    local real::Float64
    local bool::Bool
    local tmpStr::String
    local cr::DAE.ComponentRef
    local e1::DAE.Exp
    local e2::DAE.Exp
    local e3::DAE.Exp
    local expl::List{DAE.Exp}
    local lstexpl::List{List{DAE.Exp}}
    @match exp begin
      DAE.BCONST(bool) => quote $bool end
      DAE.ICONST(int) => quote $int end
      DAE.RCONST(real) => quote $real end
      DAE.SCONST(tmpStr) => quote $tmpStr end
      DAE.CREF(cr, _)  => begin
        varName = SimulationCode.string(cr)
        builtin = if varName == "time"
          true
        else
          false
        end
        if ! builtin
          #= If we refer to time, we  return t instead of a concrete variable =#
          indexAndVar = hashTable[varName]
          varKind::SimulationCode.SimVarType = indexAndVar[2].varKind
          @match varKind begin
            SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
            SimulationCode.STATE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName state"))
              $(Symbol(varPrefix))[$(indexAndVar[1])]
            end
            SimulationCode.PARAMETER(__) => quote
              $(LineNumberNode(@__LINE__, "$varName parameter"))
              p[$(indexAndVar[1])]
            end
            SimulationCode.ALG_VARIABLE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, algebraic"))
              $(Symbol(varPrefix))[$(indexAndVar[1])]
            end
            SimulationCode.DISCRETE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, Discrete"))
              $(Symbol(varPrefix))[$(indexAndVar[1])]
            end
            SimulationCode.STATE_DERIVATIVE(__) => :(dx$(varSuffix)[$(indexAndVar[1])] #= der($varName) =#)
            #=
            DATA_STRUCTURE / STRING: opaque / discrete-only
            variables that do not live in the integrator's continuous state
            vector. Emit by the SimVar's registered `name`, matching how
            expToJuliaExpMTK lowers the same cases. Without these arms the
            legacy `expToJuliaExp` path (still used by when-clause codegen,
            parameter-assignment codegen, etc.) hits MetaModelica
            MatchFailure on any reference to an external-object handle like
            `combiTimeTable.tableID`. Surfaced by
            Modelica.Thermal.FluidHeatFlow.Examples.TestOpenTank and every
            model that passes a CombiTimeTable handle to getNextTimeEvent
            inside a when-clause.
            =#
            SimulationCode.DATA_STRUCTURE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, datastructure"))
              $(Symbol(indexAndVar[2].name))
            end
            SimulationCode.STRING(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, string"))
              $(Symbol(indexAndVar[2].name))
            end
          end
        else #= Currently only time is a builtin variable. Time is represented as t in the generated code =#
          quote
            t
          end
        end
      end
      DAE.UNARY(operator = op, exp = e1) => begin
        o = DAE_OP_toJuliaOperator(op)
        quote
          $(o)($(expToJuliaExp(e1, context, varPrefix=varPrefix)))
        end
      end
      DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        a = expToJuliaExp(e1, context, varPrefix=varPrefix)
        b = expToJuliaExp(e2, context, varPrefix=varPrefix)
        o = DAE_OP_toJuliaOperator(op)
        quote
          $o($(a), $(b))
        end
      end
      DAE.LUNARY(operator = op, exp = e1)  => begin
        lhs = expToJuliaExp(e1, context, varPrefix=varPrefix)
        o = DAE_OP_toJuliaOperator(op)
        quote
          $o($(lhs))
        end
      end
      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        l = expToJuliaExp(e1, context, varPrefix=varPrefix)
        o = DAE_OP_toJuliaOperator(op)
        r = expToJuliaExp(e2, context, varPrefix=varPrefix)
        quote
          $o($(l), $(r))
        end
      end
      DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
        lhs = expToJuliaExp(e1, context, varPrefix=varPrefix)
        o = DAE_OP_toJuliaOperator(op)
        rhs = expToJuliaExp(e2, context, varPrefix=varPrefix)
        quote
          $o($(lhs), $(rhs))
        end
      end
      DAE.IFEXP(expCond = e1, expThen = e2, expElse = e3) => begin
        local condJL = expToJuliaExp(e1, context, varPrefix=varPrefix)
        local thenJL = expToJuliaExp(e2, context, varPrefix=varPrefix)
        local elseJL = expToJuliaExp(e3, context, varPrefix=varPrefix)
        :(ifelse($(condJL), $(thenJL), $(elseJL)))
      end
      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = explst)  => begin
        DAECallExpressionToJuliaCallExpression(tmpStr, explst, context, hashTable, varPrefix=varPrefix)
      end
      #=
      Qualified-path DAE.CALL in the legacy (non-MTK) code path. Mirrors the
      handler already present in expToJuliaExpMTK: emit a direct Julia call
      with a dot-to-underscore-normalized name and recursively lower the
      arguments. Covers Modelica-function calls appearing inside
      when-statements (e.g. `getNextTimeEvent(...)` from CombiTimeTable),
      which previously fell through to the `_ => throw(...)` fallback and
      blocked translate on every model that uses a table function in a
      when-clause.

      NOTE: this makes translate succeed. Whether the generated call
      actually resolves at runtime depends on the target function being
      registered (external-object runtime / @register_symbolic). Models
      whose runtime depends on these externals may still fail at simulate.
      =#
      DAE.CALL(path, expLst) => begin
        local expr = Expr(:call, Symbol(string(path)))
        local args::Vector{Any} = Any[]
        for arg in expLst
          push!(args, expToJuliaExp(arg, context, varSuffix, varPrefix=varPrefix))
        end
        append!(expr.args, args)
        expr
      end
      DAE.CAST(ty, exp)  => begin
        quote
          $(generateCastExpression(ty, exp, context, varPrefix))
        end
      end
      _ => unsupported("expression", exp)
    end
  end
  return expr
end

