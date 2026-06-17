#=
  This file contains the code generation for the DifferentialEquations.jl backend.

TODO:
  Add support for if equations
  Current approach. One separate function for each branch.

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
   appear OUTSIDE any `change(...)` or `pre(...)` wrapper. Used by the
   discrete-callback codegen so the auto-reset block (which zeroes
   condition-triggers after firing — appropriate for `when boolVar then`)
   does NOT zero observed inputs of `change(x)` / `pre(x)` (which are state
   variables we're watching, not latched flags). =#
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
function createCallbackCode(modelName::N, simCode::S; generateSaveFunction = true) where {N, S}
  #= Synthesised discrete-Boolean whens (`change(rel)` conditions) are emitted as
     MTK SymbolicContinuousCallbacks (createDiscreteBoolWhenEvents); exclude them
     here so they are not also built into the legacy CallbackSet (which cannot
     read MTK observed variables). =#
  local _legacyWhens = filter(w -> _extractChangeRelations(w.whenEquation.condition, simCode) === nothing &&
                                   isempty(_selfSchedulingTimeRels(w.whenEquation.condition)),
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

function createParameterCode(modelName, parameters, stateVariables, algVariables, simCode)::Expr
  local PARAMETER_EQUATIONS = createParameterEquations(parameters, simCode)
  quote
    function $(Symbol("$(modelName)ParameterVars"))()
      local aux = Array{Array{Float64}}(undef, 2)
      local p = Array{Float64}(undef, $(arrayLength(parameters)))
      local reals = Array{Float64}(undef, $(arrayLength(stateVariables) + arrayLength(algVariables)))
      aux[1] = p
      aux[2] = reals
      $(PARAMETER_EQUATIONS...)
      return aux
    end
  end
end

"""
  Creates equation code from the set of residual equations and a supplied set of variables.
  This is the method used for the solver.
  $(SIGNATURES)
"""
function createSolverCode(functionName::Symbol,
                          auxFuncSymbol::Symbol,
                          variables::Vector{V},
                          residuals::Vector{R},
                          ifEquations::Vector{IF_EQ},
                          simCode::SimulationCode.SIM_CODE;
                          eqLhsName, eqRhsName)::Expr where {V, R, IF_EQ}
  local UPDATE_VECTOR = createRealToStateVariableMapping(variables, simCode)
  #= Creates the equations =#
  local EQUATIONS = createEquations(variables, residuals, simCode; eqLhsName = eqLhsName, eqRhsName)
  local IF_EQUATIONS = createEquations(variables, ifEquations, simCode; eqLhsName = eqLhsName, eqRhsName = eqRhsName)
  local modelName = simCode.name
  quote
    function $(functionName)(res, dx, x, aux, t)
      $(auxFuncSymbol)(res, dx, x, aux, t)
      local p = aux[1]
      local reals = aux[2]
      $(EQUATIONS...)
      $(IF_EQUATIONS...)
      $(UPDATE_VECTOR...)
    end
  end
end

"""
  This method creates a runnable for a linear/non-linear system of equations.
  That is a system that does not contain differential equations
"""
function createLinearRunnable(modelName::String, simCode::SimulationCode.SIM_CODE)
  quote
    import NonlinearSolve
    function $(Symbol("$(modelName)Simulate"))(tspan = (0.0, 1.0))
      $(LineNumberNode((@__LINE__), "Auxilary variables"))
      local aux = $(Symbol("$(modelName)ParameterVars"))()
      (x0, dx0) =$(Symbol("$(modelName)StartConditions"))(aux, tspan[1])
      local differential_vars = $(Symbol("$(modelName)DifferentialVars"))()
      #= Pass the residual equations =#
      local problem = NonlinearProblem($(Symbol("$(modelName)DAE_equations")), dx0, x0,
                                       tspan, aux, differential_vars=differential_vars,
                                       callback=$(Symbol("$(modelName)CallbackSet"))(aux))
      #= Solve with IDA =#
      local solution = Runtime.solve(problem::NonlinearProblem, IDA())
      #= Convert into OM compatible format =#
      local savedSol = map(collect, $(Symbol("saved_values_$(modelName)")).saveval)
      local t = [savedSol[i][1] for i in 1:length(savedSol)]
      local vars = [savedSol[i][2] for i in 1:length(savedSol)]
      local T = eltype(eltype(vars))
      local N = length(aux[2])
      local nsolution = DAESolution{Float64,N,typeof(vars),Nothing, Nothing, Nothing, typeof(t),
                                    typeof(problem),typeof(solution.alg),
                                    typeof(solution.interp),typeof(solution.destats)}(
                                      vars, nothing, nothing, nothing, t, problem, solution.alg,
                                      solution.interp, solution.dense, 0, solution.destats, solution.retcode)
      ht = $(SimulationCode.makeIndexVarNameUnorderedDict(simCode.matchOrder, simCode.stringToSimVarHT))
      omSolution = OMBackend.Runtime.OMSolution(nsolution, ht)
      return omSolution
    end
  end
end



"""
  This function creates the update equations for the auxiliary variables.
  The set of auxiliary variables is the set of variables of other types than state variables.
  That is booleans integers and algebraic variables.
TODO:
  Currently only being done for the algebraic variables.
"""
function createAuxEquationCode(algVariables::Array{V},
                               simCode::SimulationCode.SIM_CODE
                               ;arrayName)::Array{Expr} where {V}
  #= Sorted equations for the algebraic variables. =#
  local auxEquations::Array{Expr} = []
  auxEquations = vcat(createSortedEquations([algVariables...], simCode; arrayName = "reals"))
  return auxEquations
end

function createStateMarkings(algVariables::Array, stateVariables::Array, simCode::SimulationCode.SIM_CODE)::Array{Bool}
  local stateMarkings::Array = [false for i in 1:length(stateVariables) + length(algVariables)]
  for sName in stateVariables
    stateMarkings[simCode.stringToSimVarHT[sName][1]] = true
  end
  return stateMarkings
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

function createRealToStateVariableMapping(stateVariables::Array, simCode::SimulationCode.SIM_CODE; toFrom::Tuple=("reals", "x"))::Array{Expr}
  local daeStateUpdateVector::Vector{Expr} = Expr[]
  for svName in stateVariables
    local varIdx = simCode.stringToSimVarHT[svName][1]
    push!(daeStateUpdateVector, :($(Symbol(toFrom[1]))[$varIdx] = $(Symbol(toFrom[2]))[$varIdx]))
  end
  return daeStateUpdateVector
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

"""
  Create equations for the parameters.
"""
function createParameterEquations(parameters::Array, simCode::SimulationCode.SimCode)
  local parameterEquations::Vector{Expr} = Expr[]
  local hT = simCode.stringToSimVarHT
  for param in parameters
    (index, simVar) = hT[param]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => SimulationCode.toDAEExp(exp)
      _ => throw(ErrorException("Unknown SimulationCode.SimVarType for parameter."))
    end
    push!(parameterEquations,
          quote
          $(LineNumberNode(@__LINE__, "$param"))
          p[$index] = $(expToJuliaExp(bindExp, simCode))
          end
          )
  end
  return parameterEquations
end


#= Build `name = <state/param lookup index>` pre-bindings for the crefs a when
   callback reads. Skips crefs absent from the simvar table: those are inlined
   constants (e.g. a logic ResetMap[i] element) that expToJulia emits as literals,
   so requesting a state/param index for them would KeyError in getIdxForLookupMTK. =#
function _whenLookupBindings(crefs, simCode)::Vector{Expr}
  local out = Expr[]
  for x in collect(map(identity, crefs))
    local entry = get(simCode.stringToSimVarHT, string(x), nothing)
    entry === nothing && continue
    #= String simvars live as module-level bindings, never as state or MTK
       parameter slots; an index binding here would KeyError at runtime. =#
    if get(ENV, "OMBACKEND_WHEN_STRING_SKIP", "true") == "true"
      entry[2].varKind isa SimulationCode.STRING && continue
    end
    push!(out, Expr(:(=), Symbol(string(x)), getIdxForLookupMTK(x, simCode)))
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
      local v = try
        SimulationCode.tryEvalNumeric(thr, simCode)
      catch
        nothing
      end
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
  local whenStmts = createWhenStatementsMTK(_elsewhenStmtLst(elseArm), simCode)
  local thrCrefs = listArray(Util.getAllCrefs(thrDAE))
  local affBindCrefs = vcat(map(x -> getRHSVariables(x), _elsewhenStmtLst(elseArm))..., thrCrefs)
  quote
    let _condCache = Ref{Any}(nothing), _affCache = Ref{Any}(nothing)
      global $(Symbol("condition$(callbacks)"))
      $(Symbol("condition$(callbacks)")) = (x, t, integrator) -> begin
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
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
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
    $(Symbol("cb$(callbacks)")) = DiscreteCallback($(Symbol("condition$(callbacks)")),
                                                   $(Symbol("affect$(callbacks)!"));
                                                   save_positions=(true, true))
  end
end

#= Emit a PresetTimeCallback for a table/time when: fire the (time-dependent) body at
   each threshold so a stepped output (e.g. a Digital Table) lands on its sample times. =#
function _emitPresetTimeWhen(eq, simCode, callbacks::Int, thresholds::Vector{Float64})
  local wEq = eq.whenEquation
  local whenStmts = createWhenStatementsMTK(wEq.whenStmtLst, simCode)
  local bodyCrefs = vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...)
  local times = sort(unique(filter(>(0.0), thresholds)))
  quote
    let _affCache = Ref{Any}(nothing)
      global $(Symbol("affect$(callbacks)!"))
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
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
      local p = try SimulationCode.tryEvalNumeric(per, simCode) catch; nothing end
      p === nothing && return nothing
      local s = @match num begin
        DAE.BINARY(exp1 = t, operator = DAE.SUB(__), exp2 = sExp) =>
          (_isTimeCref(t) ? (try SimulationCode.tryEvalNumeric(sExp, simCode) catch; nothing end) : nothing)
        _ => (_isTimeCref(num) ? 0.0 : nothing)
      end
      s === nothing && return nothing
      (Float64(s), Float64(p))
    end
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
  local sortedIfEqs = sort(collect(simCode.ifEquations); by = ifEq -> _ifEquationSortKey(ifEq, simCode))
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
  for weq in simCode.whenEquations
    _extractChangeRelations(weq.whenEquation.condition, simCode) === nothing && continue
    local condDAE = SimulationCode.toDAEExp(weq.whenEquation.condition)
    local condCrefs = listArray(Util.getAllCrefs(condDAE))
    any(c -> string(c) in writtenLHS, condCrefs) || continue
    append!(refreshStmts, createWhenStatementsMTK(weq.whenEquation.whenStmtLst, simCode))
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
  local whenStmts = createWhenStatementsMTK(wEq.whenStmtLst, simCode)
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
                                                   phase = $(_firstEdge), initial_affect = false)
  end
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
  local isPeriodic = @match wEqCondDAE begin
    DAE.CALL(Absyn.IDENT("sample"), args, attrs) => true
    _ => false
  end
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
      local whenStatementsMTKIf = createWhenStatementsMTK(wEq.whenStmtLst, simCode)
      local whenStatementsMTKElse = createWhenStatementsMTK(_elsewhenStmtLst(elsePart), simCode)
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
                                                         rootfind=true,
                                                         save_positions=(true, true),
                                                         affect_neg! = $(Symbol("affect$(callbacks)!")))
      end
    else #= No elseif =#
      whenStatementsMTK  = createWhenStatementsMTK(wEq.whenStmtLst, simCode)
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
        #= No `affect_neg!` set: transformToZeroCrossingCondition has already
           encoded direction (positive→negative = trigger) so the same
           Modelica `when cond then` semantics fall on `affect!` only. Setting
           `affect_neg! = affect!` would double-fire on each oscillation
           (classic bouncing-ball: downcrossing reinit then upcrossing reinit
           again at the same event), driving Zeno / maxiters. =#
        $(Symbol("cb$(callbacks)")) = ContinuousCallback($(Symbol("condition$(callbacks)")),
                                                         $(Symbol("affect$(callbacks)!")),
                                                         rootfind=true, save_positions=(true, true))
        $(if wEq.elsewhenPart !== nothing
            eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 0)
          end)
      end
    end
  elseif isPeriodic
    @match DAE.CALL(Absyn.IDENT("sample"), args, attrs) = wEqCondDAE
    @match start <| interval <| tail = args
    #= MTK-aware periodic affect: the hardcoded x[N]/p[N] indices from
       expToJuliaExp are invalid after MTK structural_simplify reorders unknowns,
       so resolve the interval to a literal Δt and write state via
       getStatesAsSymbols + lookuptable, mirroring the discrete branch. =#
    local whenStatementsMTKPeriodic = createWhenStatementsMTK(wEq.whenStmtLst, simCode)
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
    quote
      let _affCache = Ref{Any}(nothing)
        global $(Symbol("affect$(callbacks)!"))
        $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
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
          $(_whenLookupBindings(vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...), simCode)...)
          $(whenStatementsMTKPeriodic...)
          #= Re-derive dependent discrete-bool whens from the just-written threshold. =#
          $(_whenLookupBindings(_dbRefreshCrefs, simCode)...)
          $(_dbRefreshStmts...)
        end
      end
      Δt = $(_dtExpr)
      $(Symbol("cb$(callbacks)")) = PeriodicCallback($(Symbol("affect$(callbacks)!")), Δt; save_positions = (true, true))
      $(if wEq.elsewhenPart !== nothing
          eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 4)
        end)
    end
  else #= If none of the variables in the condition was continuous.. =#
    #= Use MTK-aware runtime symbol lookup for discrete callbacks.
       The hardcoded x[N] indices from expToJuliaExp become invalid after
       MTK structural_simplify reorders unknowns. Mirror the continuous
       callback path (above) which uses getStatesAsSymbols + lookuptable. =#
    whenStatementsMTKDiscrete = createWhenStatementsMTK(wEq.whenStmtLst, simCode)
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
    #= Crefs that appear inside `change(...)` or `pre(...)` are OBSERVED, not
       latched boolean triggers — they must not be reset after the callback
       fires (e.g. INV3S-class algorithms whose synthesised condition is
       `change(iNV3S_enable)` would otherwise zero the Logic-enum input on
       every event). Build the bare-cref set so condReset only touches
       standalone Boolean triggers, the original use case. =#
    local _bareCrefStrs = _collectBareCrefStrings(cond)
    #= Build condition-reset expressions: set each state cref in the condition to false =#
    local condResetExprs = map(condCrefs) do c
      local cStr = string(c)
      cStr in _bareCrefStrs || return :()
      local entry = get(simCode.stringToSimVarHT, cStr, nothing)
      if entry !== nothing && !SimulationCode.isParameter(entry[2])
        local sym = Symbol(cStr)
        #= AUDIT (ombackend-bug-audit-2026-06-05 #5): guard the index exactly as
           changeInitExprs/changeUpdateExprs do. An unguarded
           `lookuptableStates[sym]` KeyErrors when the trigger is not a state
           unknown, and a Boolean DEFINED by a relation is observed/algebraic
           (not in the state vector): force-clearing it would corrupt a value the
           integrator re-derives. Only a genuine discrete STATE Boolean is
           de-bounced here. NOTE: for a discrete-state Boolean that is also read
           elsewhere and meant to persist true, edge de-bounce via state mutation
           is still a shortcut; a private per-callback latch is the proper fix. =#
        quote
          let _idx = get(lookuptableStates, $(QuoteNode(sym)), nothing)
            if _idx !== nothing
              x[_idx] = false
            end
          end
        end
      else
        :()
      end
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
      let _condCache = Ref{Any}(nothing),
          _affCache = Ref{Any}(nothing),
          _changeCache = Ref{Any}(nothing),
          _changeSeedValues = Dict{Symbol, Any}($(changeSeedPairs...))
        global $(Symbol("condition$(callbacks)"))
        $(Symbol("condition$(callbacks)")) = (x, t, integrator) -> begin
          local lookuptableStates
          local lookuptableParams
          if _condCache[] === nothing
            local xs = $(map(c -> Symbol(string(c)), condCrefs))
            local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
            if !isempty(xs) && all(isnothing, indices)
              @debug "[CB-DC$($(callbacks)) cond] false (no state mapping)" t xs
              return false
            end
            lookuptableStates = Dict((xs) .=> indices)
            local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
            lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
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
            if isempty(_changePreValues)
              $(changeInitExprs...)
            end
          else
            _changePreValues = _changeCache[]
          end
          $(_whenLookupBindings(Util.getAllCrefs(cond), simCode)...)
          local _result = $(expToJuliaBoolMTK(wEqCondDAE, simCode; cachedChange = true))
          @debug "[CB-DC$($(callbacks)) cond] eval" t value=_result
          #= DiscreteCallback condition must return Bool per SciMLBase. Modelica
             Boolean discrete states are stored as Float64 in `integrator.u`
             (0.0/1.0), so a bare cref read returns Float64 and triggers
             "TypeError: non-boolean (Float64) used in boolean context" in
             SciML's callback dispatch (affects PowerConverters Thyristor
             models). Cast via `!= 0` so any numeric cref-as-condition
             evaluates correctly. Bool results pass through unchanged. =#
          _result isa Bool ? _result : (_result != 0)
        end
        global $(Symbol("affect$(callbacks)!"))
        $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
          local t = integrator.t
          local x = integrator.u
          @debug "[CB-DC$($(callbacks)) affect] firing" t=integrator.t
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
          $(whenStatementsMTKDiscrete...)
          $(_ewSchedule)
          auto_dt_reset!(integrator)
          add_tstop!(integrator, integrator.t + 1E-12)
          local _changePreValues = _changeCache[] === nothing ? Dict{Symbol, Any}() : _changeCache[]
          _changeCache[] = _changePreValues
          $(changeUpdateExprs...)
          $(condResetExprs...)
          @debug "[CB-DC$($(callbacks)) affect] done" t=integrator.t u=copy(integrator.u)
        end
      end
      $(Symbol("cb$(callbacks)")) = DiscreteCallback($(Symbol("condition$(callbacks)")),
                                                     $(Symbol("affect$(callbacks)!"));
                                                     save_positions=(true, true))
      $(if _ewThrDAE !== nothing
          _emitElsewhenThresholdTimeWhen(_ewArm, simCode, _ewRefSym, _ewThrDAE)
        elseif wEq.elsewhenPart !== nothing
          eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 4)
        end)
    end
  end
end


"""
  Converts a DAE expression into a Julia expression
  $(SIGNATURES)
The context can be any type that contains a set of residual equations.
"""
#= SimCode-Exp entry: codegen consumes `SimulationCode.Exp` (Phase 4b API).
   See comment on the MTK variant. =#
#= SimCode-Exp entry (Phase 4b API): per-variant dispatch mirrors the DAE.Exp emitter;
   only the EXP_CREF leaf, CALL args, and CAST touch a per-node DAE projection. =#
function expToJuliaExp(e::SimulationCode.BCONST, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  quote $(e.value) end
end
function expToJuliaExp(e::SimulationCode.ICONST, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  quote $(e.value) end
end
function expToJuliaExp(e::SimulationCode.RCONST, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  quote $(e.value) end
end
function expToJuliaExp(e::SimulationCode.SCONST, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  quote $(e.value) end
end

function expToJuliaExp(e::SimulationCode.EXP_CREF, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
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
    SimulationCode.OCC_VARIABLE(__) => quote
      $(LineNumberNode(@__LINE__, "$varName, occ variable"))
      $(Symbol(indexAndVar[2].name))
    end
    SimulationCode.STRING(__) => quote
      $(LineNumberNode(@__LINE__, "$varName, string"))
      $(Symbol(indexAndVar[2].name))
    end
  end
end

function expToJuliaExp(e::SimulationCode.UNARY, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  local o = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(e.op))
  quote
    $(o)($(expToJuliaExp(e.exp, context, varPrefix=varPrefix)))
  end
end
function expToJuliaExp(e::SimulationCode.BINARY, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  local a = expToJuliaExp(e.exp1, context, varPrefix=varPrefix)
  local b = expToJuliaExp(e.exp2, context, varPrefix=varPrefix)
  local o = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(e.op))
  quote
    $o($(a), $(b))
  end
end
function expToJuliaExp(e::SimulationCode.LUNARY, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  local lhs = expToJuliaExp(e.exp, context, varPrefix=varPrefix)
  local o = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(e.op))
  quote
    $o($(lhs))
  end
end
function expToJuliaExp(e::SimulationCode.LBINARY, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  local l = expToJuliaExp(e.exp1, context, varPrefix=varPrefix)
  local o = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(e.op))
  local r = expToJuliaExp(e.exp2, context, varPrefix=varPrefix)
  quote
    $o($(l), $(r))
  end
end
function expToJuliaExp(e::SimulationCode.RELATION, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  local lhs = expToJuliaExp(e.exp1, context, varPrefix=varPrefix)
  local o = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(e.op))
  local rhs = expToJuliaExp(e.exp2, context, varPrefix=varPrefix)
  quote
    $o($(lhs), $(rhs))
  end
end
function expToJuliaExp(e::SimulationCode.IFEXP, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  local condJL = expToJuliaExp(e.cond, context, varPrefix=varPrefix)
  local thenJL = expToJuliaExp(e.thenExp, context, varPrefix=varPrefix)
  local elseJL = expToJuliaExp(e.elseExp, context, varPrefix=varPrefix)
  :(ifelse($(condJL), $(thenJL), $(elseJL)))
end
function expToJuliaExp(e::SimulationCode.CALL, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
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
function expToJuliaExp(e::SimulationCode.CAST, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
  quote
    $(generateCastExpression(SimulationCode.toDAEType(e.ty), SimulationCode.toDAEExp(e.exp), context, varPrefix))
  end
end
Base.@nospecializeinfer function expToJuliaExp(@nospecialize(exp::SimulationCode.Exp),
                                               @nospecialize(context::C),
                                               varSuffix = ""; varPrefix = "x")::Expr where {C}
  throw(ErrorException("$exp not yet supported"))
end

function expToJuliaExp(exp::DAE.Exp, context::C, varSuffix=""; varPrefix="x")::Expr where {C}
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
            DATA_STRUCTURE / OCC_VARIABLE / STRING: opaque / discrete-only
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
            SimulationCode.OCC_VARIABLE(__) => quote
              $(LineNumberNode(@__LINE__, "$varName, occ variable"))
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
      _ =>  throw(ErrorException("$exp not yet supported"))
    end
  end
  return expr
end


"""
  Generates the start conditions.
  All variables default to zero if they are not specified by the user.
"""
function getStartConditions(vars::Array, condName::String, simCode::SimulationCode.SimCode)::Expr
  local startExprs::Array{Expr} = []
  local residuals = simCode.residualEquations
  local ht::Dict = simCode.stringToSimVarHT
  if length(vars) == 0
    return quote
    end
  end
  for var in vars
    (index, simVar) = ht[var]
    local simVarType = simVar.varKind
    local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
    if simVar.attributes == nothing
      continue
    end
    () = @match optAttributes begin
      SOME(attributes) => begin
        () = @match (attributes.start, attributes.fixed) begin
          (SOME(start), SOME(fixed)) || (SOME(start), _)  => begin
            @debug "Start value is:" start
            push!(startExprs,
                  quote
                  $(LineNumberNode(@__LINE__, "$var"))
                  $(Symbol("$condName"))[$index] = $(expToJuliaExp(start, simCode))
                  end)
            ()
          end
          (NONE(), SOME(fixed)) => begin
            push!(startExprs, :($(condName)[$(index)] = 0.0))
            ()
          end
          (_, _) => ()
        end
      end
      NONE() => ()
    end
  end
  return quote
    $(startExprs...)
  end
end
