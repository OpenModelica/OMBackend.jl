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
  local WHEN_EQUATIONS = createEquations(simCode.whenEquations,  simCode)
  #=
    For if equations we create zero crossing functions (Based on the conditions).
    The body of these equations are evaluated in the main body of the solver itself.
  =#
  #local IF_EQUATIONS = createIfEquationCallbacks(simCode.ifEquations, simCode) Deprecated
  local SAVE_FUNCTION = if generateSaveFunction
    createSaveFunction(modelName)
  else
  end
  local MODEL_NAME = replace(modelName, "." => "__")
  quote
    $(Symbol("saved_values_$(modelName)")) = SavedValues(Float64, Tuple{Float64,Array})
    function $(Symbol("$(MODEL_NAME)CallbackSet"))(aux)
      #= These are the location of the parameters and auxilary real variables respectivly =#
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
  This functions creates the update equations for the auxilary variables.
  The set of auxilary variables is the set of variables of other types than state variables.
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
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => exp
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
function eqToJulia(eq::BDAE.WHEN_EQUATION, simCode::SimulationCode.SIM_CODE, arrayIdx::Int)::Expr
  local wEq = eq.whenEquation
  local whenStmts = createWhenStatements(wEq.whenStmtLst, simCode)
  local cond = transformToZeroCrossingCondition(wEq.condition)
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
  local isPeriodic = @match wEq.condition begin
    DAE.CALL(Absyn.IDENT("sample"), args, attrs) => true
    _ => false
  end
  local isContinuousCond::Bool = isContinousCondition(wEq.condition, simCode)
  if isContinuousCond
    local isElseIf = if wEq.elsewhenPart !== nothing
      local elsePart = wEq.elsewhenPart.data
      local elseCond = elsePart.whenEquation.condition
      cond2 = transformToZeroCrossingCondition(elseCond)
      cond == cond2
    else
      false
    end
    if isElseIf
      quote
        $(Symbol("condition$(callbacks)")) = (x,t,integrator) -> begin
          $(expToJuliaExp(cond, simCode))
        end
        $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
          local t = integrator.t + integrator.dt
          local x = integrator.u
          if integrator.dt == 0.0
            @error "integrator.dt was zero. Aborting."
            fail()
          end
          if (Bool($(expToJuliaExp(wEq.condition, simCode))))
            $(whenStmts...)
            add_tstop!(integrator, integrator.t + 1E-12) #=TODO: Some small number for now=#
          else
            $(createWhenStatements(elsePart.whenEquation.whenStmtLst, simCode)...)
            add_tstop!(integrator, integrator.t + 1E-12) #=TODO: Some small number for now=#
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
      local cond = quote  #= NOTE. Very expensive to get it as strings. This can be done better but requires slight redesign =#
        $(Symbol("condition$(callbacks)")) = (x, t, integrator) -> begin
          #local xStrs = $(map(x -> string(x), Util.getAllCrefs(cond)))
          local xs = $(map(x -> Symbol(string(x)), filter(c -> string(c) != "time", listArray(Util.getAllCrefs(cond)))))
          local indices = indexin(xs, OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f))
          local NO_TRIGGER = 1.0
          if !isempty(xs) && all(isnothing, indices)
            return NO_TRIGGER
          end
          local lookuptableStates = Dict((xs) .=> indices)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          local lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          $(map(x -> Expr(Symbol("="),
                          Symbol(string(x)),
                          getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)) ,
                Util.getAllCrefs(cond))...)
          $(expToJuliaExpMTK(cond, simCode))
        end
      end
      local affect = quote
        $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
          local t = integrator.t + integrator.dt
          local x = integrator.u
          local states = OMBackend.CodeGeneration.getStatesAsSymbols(integrator.f)
          local params = OMBackend.CodeGeneration.getParametersAsSymbols(integrator.f)
          local lookuptableStates = Dict(sym => i for (i, sym) in enumerate(states))
          local lookuptableParams = Dict(sym => i for (i, sym) in enumerate(params))
          $(map(x->Expr(Symbol("="),
                        Symbol(string(x)),
                        getIdxForLookupMTK(x::Union{DAE.CREF, DAE.ComponentRef}, simCode)) ,
                vcat(map(x -> BDAEUtil.getRHSVariables(x), listArray(wEq.whenStmtLst))...))...)
          if integrator.dt == 0.0
            @error "integrator.dt was zero. Aborting."
            fail()
          end
          $(whenStatementsMTK...)
        end
      end
      quote
        $cond
        $affect
        $(Symbol("cb$(callbacks)")) = ContinuousCallback($(Symbol("condition$(callbacks)")),
                                                         $(Symbol("affect$(callbacks)!")),
                                                         rootfind=true, save_positions=(true, true),
                                                         affect_neg! = $(Symbol("affect$(callbacks)!")))
        $(if wEq.elsewhenPart !== nothing
            eqToJulia(wEq.elsewhenPart.data, simCode, 0)
          end)
      end
    end
  elseif isPeriodic
    @match DAE.CALL(Absyn.IDENT("sample"), args, attrs) = wEq.condition
    @match start <| interval <| tail = args
    quote
      $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
        #@info "Calling affect! at $(integrator.t)"
        local t = integrator.t + integrator.dt
        local x = integrator.u
        if integrator.dt == 0.0
          @error "integrator.dt was zero. Aborting."
          fail()
        end
        #@info "t + dt = " t
        #if (Bool($(expToJuliaExp(wEq.condition, simCode))))
        $(whenStmts...)
        #end
      end
      Δt = $(expToJuliaExp(interval, simCode))
      $(Symbol("cb$(callbacks)")) = PeriodicCallback($(Symbol("affect$(callbacks)!")), Δt)
    end
  else #= If none of the variables in the condition was continuous.. =#
    quote
      $(Symbol("condition$(callbacks)")) = (x,t,integrator) -> begin
        Bool($(expToJuliaExp(cond, simCode)))
      end
    $(Symbol("affect$(callbacks)!")) = (integrator) -> begin
      # @info "Calling affect for discrete at $(integrator.t). Condition was:"
      # @info "Δt was:" integrator.dt
      # @info "Value of x is:" integrator.u
      local t = integrator.t
      local x = integrator.u
      $(whenStmts...)
      auto_dt_reset!(integrator)
      add_tstop!(integrator, integrator.t + 1E-12) #=TODO: Some small number for now=#
      #= TODO:
      This will not work if the condition consists of several boolean operators.
      The purpose is to reset the condition so that it is not triggered again at the end of the next integration step.
      =#
      $(expToJuliaExp(cond, simCode)) = false
    end
      $(Symbol("cb$(callbacks)")) = DiscreteCallback($(Symbol("condition$(callbacks)")),
                                                     $(Symbol("affect$(callbacks)!"));
                                                     save_positions=(true, true))
      $(if wEq.elsewhenPart !== nothing
          eqToJulia(wEq.elsewhenPart.data, simCode, 4)
        end
        )
    end
  end
end


"""
   Creates Julia code for the set of whenStatements in the when equation.
   There are some constructs that may only occur in a when equations.
"""
function createWhenStatements(whenStatements::List, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local res::Array{Expr} = []
  @debug "Calling createWhenStatements with: $whenStatements"
  for wStmt in  whenStatements
    @match wStmt begin
      BDAE.ASSIGN(__) => begin
        (index, var) = simCode.stringToSimVarHT[SimulationCode.string(wStmt.left)]
        if typeof(var.varKind) === SimulationCode.STATE
          exp1 = expToJuliaExp(wStmt.left, simCode, varPrefix="integrator.u")
          exp2 = expToJuliaExp(wStmt.right, simCode)
          push!(res, :($(exp1) = $(exp2)))
        elseif typeof(var.varKind) === SimulationCode.ALG_VARIABLE #=TODO also check type=#
          exp1 = expToJuliaExp(wStmt.left, simCode, varPrefix="reals")
          exp2 = expToJuliaExp(wStmt.right, simCode)
          push!(res, :($(exp1) = $(exp2)))
        elseif var.varKind isa SimulationCode.DISCRETE || var.varKind isa SimulationCode.PARAMETER
          #=
            TODO: In reality a branch of a when equation should be sorted.
          ==#
          exp1 = expToJuliaExp(wStmt.left, simCode)
          exp2 = expToJuliaExp(wStmt.right, simCode)
          push!(res, :($(exp1) = $(exp2)))
        end
      end
      #= Handles reinit =#
      BDAE.REINIT(__) => begin
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
      end
      _ => throw(ErrorException("$whenStatements in @__FUNCTION__ not supported"))
    end
  end
  return res
end

"""
  Converts a DAE expression into a Julia expression
  $(SIGNATURES)
The context can be any type that contains a set of residual equations.
"""
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
        throw(ErrorException("If expressions not allowed in backend code.
                              They should have been eliminated by a previous pass."))
      end
      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = explst)  => begin
        DAECallExpressionToJuliaCallExpression(tmpStr, explst, context, hashTable, varPrefix=varPrefix)
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
        NONE() => ()
      end
    end
  end
  return quote
    $(startExprs...)
  end
end
