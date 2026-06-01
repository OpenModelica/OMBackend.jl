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
Base.@nospecializeinfer function _collectBareCrefStrings(@nospecialize(cond))::Set{String}
  local out = Set{String}()
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
  local _legacyWhens = filter(w -> _extractChangeRelations(w.whenEquation.condition) === nothing,
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
  local isContinuousCond::Bool = isContinousCondition(wEqCondDAE, simCode)
  if isContinuousCond
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
         The hardcoded x[N] indices from expToJuliaExp/createWhenStatements
         become invalid after MTK structural_simplify reorders unknowns. =#
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
            $(map(x -> Expr(Symbol("="),
                            Symbol(string(x)),
                            getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)),
                  Util.getAllCrefs(cond))...)
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
            $(map(x -> Expr(Symbol("="),
                            Symbol(string(x)),
                            getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)),
                  vcat(
                    listArray(Util.getAllCrefs(wEqCondDAE)),
                    vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...),
                    vcat(map(x -> getRHSVariables(x), _elsewhenStmtLst(elsePart))...)
                  ))...)
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
            $(map(x -> Expr(Symbol("="),
                            Symbol(string(x)),
                            getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)) ,
                  Util.getAllCrefs(cond))...)
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
            $(map(x->Expr(Symbol("="),
                          Symbol(string(x)),
                          getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)) ,
                  vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...))...)
            if integrator.dt == 0.0
              @error "integrator.dt was zero. Aborting."
              fail()
            end
            $(whenStatementsMTK...)
            @debug "[CB-CC$($(callbacks)) affect] done" t=integrator.t u=copy(integrator.u)
          end
        end
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
    local whenStmts = createWhenStatements(wEq.whenStmtLst, simCode)
    quote
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        local t = integrator.t + integrator.dt
        local x = integrator.u
        if integrator.dt == 0.0
          @error "integrator.dt was zero. Aborting."
          fail()
        end
        $(whenStmts...)
      end
      Δt = $(expToJuliaExp(interval, simCode))
      $(Symbol("cb$(callbacks)")) = PeriodicCallback($(Symbol("affect$(callbacks)!")), Δt)
    end
  else #= If none of the variables in the condition was continuous.. =#
    #= Use MTK-aware runtime symbol lookup for discrete callbacks.
       The hardcoded x[N] indices from expToJuliaExp become invalid after
       MTK structural_simplify reorders unknowns. Mirror the continuous
       callback path (above) which uses getStatesAsSymbols + lookuptable. =#
    whenStatementsMTKDiscrete = createWhenStatementsMTK(wEq.whenStmtLst, simCode)
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
        :(x[lookuptableStates[$(QuoteNode(sym))]] = false)
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
          $(map(x -> Expr(Symbol("="),
                          Symbol(string(x)),
                          getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)),
                Util.getAllCrefs(cond))...)
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
          $(map(x -> Expr(Symbol("="),
                          Symbol(string(x)),
                          getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)),
                vcat(map(x -> getRHSVariables(x), wEq.whenStmtLst)...))...)
          $(whenStatementsMTKDiscrete...)
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
      $(if wEq.elsewhenPart !== nothing
          eqToJulia(_elsewhenInner(wEq.elsewhenPart), simCode, 4)
        end)
    end
  end
end


"""
   Creates Julia code for the set of whenStatements in the when equation.
   There are some constructs that may only occur in a when equations.
"""
function createWhenStatements(whenStatements, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local res::Array{Expr} = []
  local nWhenStatements = 0
  for _ in whenStatements
    nWhenStatements += 1
  end
  @debug "[CODEGEN: when] createWhenStatements" statements=nWhenStatements
  for wStmt in whenStatements
    local isAssign = wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
    local isReinit = wStmt isa BDAE.REINIT || wStmt isa SimulationCode.REINIT
    if isAssign && wStmt.left isa DAE.TUPLE
      local tupSym = gensym(:tupResult)
      local rhsExpr = expToJuliaExp(wStmt.right, simCode)
      push!(res, :(local $tupSym = $rhsExpr))
      local i = 0
      for elem in wStmt.left.PR
        i += 1
        _emitWhenTupleElementAssign!(res, elem, :($tupSym[$i]), simCode)
      end
    elseif isAssign
      # SimulationCode.ASSIGN.left is ::Exp post-migration; HT keys are DAE-stringified.
      (index, var) = simCode.stringToSimVarHT[SimulationCode.string(SimulationCode.toDAEExp(wStmt.left))]
      if typeof(var.varKind) === SimulationCode.STATE
        exp1 = expToJuliaExp(wStmt.left, simCode, varPrefix="integrator.u")
        exp2 = expToJuliaExp(wStmt.right, simCode)
        push!(res, :($(exp1) = $(exp2)))
      elseif typeof(var.varKind) === SimulationCode.ALG_VARIABLE
        exp1 = expToJuliaExp(wStmt.left, simCode, varPrefix="reals")
        exp2 = expToJuliaExp(wStmt.right, simCode)
        push!(res, :($(exp1) = $(exp2)))
      elseif var.varKind isa SimulationCode.DISCRETE || var.varKind isa SimulationCode.PARAMETER
        exp1 = expToJuliaExp(wStmt.left, simCode)
        exp2 = expToJuliaExp(wStmt.right, simCode)
        push!(res, :($(exp1) = $(exp2)))
      end
    elseif isReinit
      (index, var) = simCode.stringToSimVarHT[SimulationCode.string(wStmt.stateVar)]
      if typeof(var.varKind) === SimulationCode.STATE
        push!(res, quote
                integrator.u[$(index)] = $(expToJuliaExp(wStmt.value, simCode))
              end)
      elseif var.varKind isa SimulationCode.ALG_VARIABLE
        push!(res, quote
                integrator.u[$(index)] = $(expToJuliaExp(wStmt.value, simCode))
              end)
      else
        throw("Unimplemented branch for: $(var.varKind)")
      end
    else
      throw(ErrorException("$whenStatements in @__FUNCTION__ not supported"))
    end
  end
  return res
end

#=
  Recursively unpack a tuple-LHS element into per-cref Julia assignments.
  `lhs` is one element of a DAE.TUPLE LHS (CREF, ARRAY, or WILD).
  `rhsAccess` is the Julia Expr that pulls this element out of the tuple temp,
  e.g. `tup[1]` or `tup[2][3]`. Output Julia LHS form is selected by the SimVar's
  varKind, matching the single-cref BDAE.ASSIGN arm above.
=#
function _emitWhenTupleElementAssign!(res::Vector{Expr}, lhs::DAE.Exp,
                                       rhsAccess, simCode::SimulationCode.SIM_CODE)
  @match lhs begin
    DAE.CREF(DAE.WILD(), _) => nothing
    DAE.CREF(__) => begin
      local name = SimulationCode.string(lhs)
      local entry = get(simCode.stringToSimVarHT, name, nothing)
      if entry === nothing
        push!(res, :($(Symbol(name)) = $rhsAccess))
        return res
      end
      local (index, var) = entry
      local lhsJulia = if typeof(var.varKind) === SimulationCode.STATE
        :(integrator.u[$index])
      elseif typeof(var.varKind) === SimulationCode.ALG_VARIABLE
        :(reals[$index])
      else
        Symbol(name)
      end
      push!(res, :($lhsJulia = $rhsAccess))
    end
    DAE.ARRAY(_, _, elements) => begin
      local i = 0
      for elem in elements
        i += 1
        _emitWhenTupleElementAssign!(res, elem, :($rhsAccess[$i]), simCode)
      end
    end
    _ => throw(ErrorException("createWhenStatements: unsupported tuple-LHS element $lhs"))
  end
  return res
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
