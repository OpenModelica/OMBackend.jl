#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* This program is distributed WITHOUT ANY WARRANTY. See the OSMC Public License
* for details.
=#

#=
Direct DifferentialEquations.jl code generation (DEMode).

Builds a self-contained Julia module that defines:
  - <Model>RHS!(du, u, p, t)        in-place ODE RHS, integer-indexed
  - <Model>StartConditions(p)       state initial vector u0
  - <Model>ParameterVars()          parameter vector p
  - <Model>Model(tspan)             constructs ODEProblem
  - simulate(tspan, solver; ...)    user-facing entry point

The emitter intentionally does NOT depend on ModelingToolkit, on
`expToJuliaExpMTK`, or on any of the MTK helpers. It mirrors the
SIM_CODE -> Julia mapping in the donor `codeGen.jl`
=#

module DEGen

using MetaModelica
import ..SimulationCode
import ..CodeGeneration
import ..CodeGeneration: DAE_OP_toJuliaOperator, stripBeginBlocks
import ..Backend.BDAE
import DAE
import Absyn

"""
Layout descriptor: ordered names of states / parameters / discretes,
plus the final integer position used by the generated f!(du, u, p, t).
"""
struct DELayout
  modelName::String
  stateNames::Vector{String}
  algNames::Vector{String}
  paramNames::Vector{String}
  discreteNames::Vector{String}
  stateIndex::Dict{String, Int}
  paramIndex::Dict{String, Int}
  discreteIndex::Dict{String, Int}
end

"""
Build a DELayout from a SIM_CODE. States and algebraic variables are kept
separate; only states occupy slots in `u`. Algebraic variables are not
supported in the initial DEMode scope and trigger an explicit error.
"""
function buildDELayout(simCode::SimulationCode.SIM_CODE)::DELayout
  local stateNames = String[]
  local algNames = String[]
  local paramNames = String[]
  local discreteNames = String[]
  local ht = simCode.stringToSimVarHT
  for (name, (_, simVar)) in ht
    @match simVar.varKind begin
      SimulationCode.STATE(__) => push!(stateNames, name)
      SimulationCode.PARAMETER(__) => push!(paramNames, name)
      SimulationCode.ALG_VARIABLE(__) => push!(algNames, name)
      SimulationCode.DISCRETE(__) => push!(discreteNames, name)
      SimulationCode.STATE_DERIVATIVE(__) => nothing
      _ => nothing
    end
  end
  sort!(stateNames)
  sort!(algNames)
  sort!(paramNames)
  sort!(discreteNames)
  local stateIndex = Dict(n => i for (i, n) in enumerate(stateNames))
  local paramIndex = Dict(n => i for (i, n) in enumerate(paramNames))
  local discreteIndex = Dict(n => i for (i, n) in enumerate(discreteNames))
  return DELayout(simCode.name, stateNames, algNames, paramNames, discreteNames,
                  stateIndex, paramIndex, discreteIndex)
end

"""
Lower a DAE.Exp to Julia for the DE backend. Emits expressions that read
states from `u[i]`, parameters from `p[i]`, and time from `t`. State
derivatives are emitted as `du[i]`. Discrete variables (currently
unsupported in user-facing code) follow the parameter slot.

Differs from `expToJuliaExp` in `codeGen.jl` by:
  - using the DELayout integer indices instead of `simCode.stringToSimVarHT[name][1]`
  - rejecting CALL fallthrough that would silently emit unresolved Julia symbols
  - inlining IFEXP (the donor in codeGen.jl rejects it; the audit T2.1 flagged
    that as inconsistent. We allow it because IFEXP can survive when the if-
    expression-to-if-equation pass leaves expressions that depend on time).
"""
function expToJuliaExpDE(exp::DAE.Exp, layout::DELayout, simCode::SimulationCode.SIM_CODE)::Expr
  local hashTable = simCode.stringToSimVarHT
  @match exp begin
    DAE.BCONST(b) => Expr(:block, b)
    DAE.ICONST(i) => Expr(:block, i)
    DAE.RCONST(r) => Expr(:block, r)
    DAE.SCONST(s) => Expr(:block, s)
    DAE.CREF(cr, _) => begin
      local varName = SimulationCode.string(cr)
      if varName == "time"
        Expr(:block, :t)
      else
        local entry = get(hashTable, varName, nothing)
        entry === nothing && error("DEMode: unknown CREF '$(varName)' in expression lowering")
        local kind = entry[2].varKind
        @match kind begin
          SimulationCode.STATE(__) => begin
            local idx = layout.stateIndex[varName]
            Expr(:ref, :u, idx)
          end
          SimulationCode.STATE_DERIVATIVE(__) => begin
            local stateName = kind.varName
            local idx = layout.stateIndex[stateName]
            Expr(:ref, :du, idx)
          end
          SimulationCode.PARAMETER(__) => begin
            local idx = layout.paramIndex[varName]
            Expr(:ref, :p, idx)
          end
          SimulationCode.ALG_VARIABLE(__) => begin
            error("DEMode: algebraic variable '$(varName)' encountered. " *
                  "Pure-ODE scope only at this milestone.")
          end
          SimulationCode.DISCRETE(__) => begin
            error("DEMode: discrete variable '$(varName)' encountered. " *
                  "Discrete/event support is not in the initial milestone.")
          end
          _ => error("DEMode: unsupported variable kind $(kind) for '$(varName)'")
        end
      end
    end
    DAE.UNARY(operator = op, exp = e1) => begin
      local o = DAE_OP_toJuliaOperator(op)
      :($o($(expToJuliaExpDE(e1, layout, simCode))))
    end
    DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local o = DAE_OP_toJuliaOperator(op)
      :($o($(expToJuliaExpDE(e1, layout, simCode)), $(expToJuliaExpDE(e2, layout, simCode))))
    end
    DAE.LUNARY(operator = op, exp = e1) => begin
      local o = DAE_OP_toJuliaOperator(op)
      :($o($(expToJuliaExpDE(e1, layout, simCode))))
    end
    DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
      local o = DAE_OP_toJuliaOperator(op)
      :($o($(expToJuliaExpDE(e1, layout, simCode)), $(expToJuliaExpDE(e2, layout, simCode))))
    end
    DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
      local o = DAE_OP_toJuliaOperator(op)
      :($o($(expToJuliaExpDE(e1, layout, simCode)), $(expToJuliaExpDE(e2, layout, simCode))))
    end
    DAE.IFEXP(expCond = c, expThen = t1, expElse = e1) => begin
      :(($(expToJuliaExpDE(c, layout, simCode))) ?
        $(expToJuliaExpDE(t1, layout, simCode)) :
        $(expToJuliaExpDE(e1, layout, simCode)))
    end
    DAE.CAST(exp = e1) => expToJuliaExpDE(e1, layout, simCode)
    DAE.CALL(path = Absyn.IDENT(name), expLst = args) => begin
      local fn = Symbol(name)
      local argExprs = [expToJuliaExpDE(a, layout, simCode) for a in args]
      Expr(:call, fn, argExprs...)
    end
    _ => error("DEMode: unsupported DAE.Exp shape $(typeof(exp)): $exp")
  end
end

"""
Lower a single residual equation (lhs - rhs == 0) to a Julia statement
of the form `du[i] = <rhs>`. The emitter assumes the residual has been
pre-rewritten so `der(state)` appears solitary on one side.

Returns nothing if the residual cannot be solved for a single derivative;
the caller is expected to detect this condition and refuse the model.
"""
function residualToDerivativeAssignment(eq::Union{BDAE.RESIDUAL_EQUATION, SimulationCode.RESIDUAL_EQUATION},
                                        layout::DELayout,
                                        simCode::SimulationCode.SIM_CODE)
  local rhsExp = eq.exp
  #=
    The residualization pass produces equations of the form lhs - rhs = 0.
    Inspect the top-level shape: BINARY(SUB, lhs, rhs) is the canonical case.
    Decide which side is `der(<state>)` and emit `du[idx] = otherSide`.
  =#
  @match rhsExp begin
    DAE.BINARY(exp1 = e1, operator = DAE.SUB(__), exp2 = e2) => begin
      local lhsCall = _matchDerCall(e1)
      local rhsCall = _matchDerCall(e2)
      if lhsCall !== nothing
        local stateName = lhsCall
        local idx = get(layout.stateIndex, stateName, nothing)
        idx === nothing && error("DEMode: der() of unknown state '$(stateName)'")
        return :(du[$idx] = $(expToJuliaExpDE(e2, layout, simCode)))
      elseif rhsCall !== nothing
        local stateName = rhsCall
        local idx = get(layout.stateIndex, stateName, nothing)
        idx === nothing && error("DEMode: der() of unknown state '$(stateName)'")
        return :(du[$idx] = $(expToJuliaExpDE(e1, layout, simCode)))
      else
        error("DEMode: residual is not a pure ODE for any state. Equation: $(rhsExp)")
      end
    end
    _ => error("DEMode: residual has unexpected top-level shape: $(typeof(rhsExp))")
  end
end

"""
Detect a `der(<state>)` call and return the state name. Returns `nothing`
if the expression is not a call to `der`.
"""
function _matchDerCall(e::DAE.Exp)
  @match e begin
    DAE.CALL(path = Absyn.IDENT("der"), expLst = args) => begin
      length(args) == 1 || return nothing
      local arg = listHead(args)
      @match arg begin
        DAE.CREF(cr, _) => SimulationCode.string(cr)
        _ => nothing
      end
    end
    _ => nothing
  end
end

"""
Build the parameter array initializer.
Returns an `Expr(:block, ...)` that writes each parameter binding into `p`
in dependency order (we emit textual order; chained parameter dependencies
are already resolved by `propagateConstants` upstream).
"""
function generateDEParameterAssignments(layout::DELayout,
                                        simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local stmts = Expr[]
  local ht = simCode.stringToSimVarHT
  for name in layout.paramNames
    (_, simVar) = ht[name]
    local bindExp = @match simVar.varKind begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => exp
      SimulationCode.PARAMETER(bindExp = NONE()) => nothing
      _ => error("DEMode: parameter '$(name)' has non-parameter SimVarType")
    end
    local idx = layout.paramIndex[name]
    if bindExp === nothing
      push!(stmts, :(p[$idx] = 0.0))
    else
      push!(stmts, :(p[$idx] = $(expToJuliaExpDE(bindExp, layout, simCode))))
    end
  end
  return stmts
end

"""
Build the state initial-value vector. For each state, look up `start`
attribute. Missing-start defaults to 0.0 (matches Modelica spec).
"""
function generateDEStartConditions(layout::DELayout,
                                   simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local stmts = Expr[]
  local ht = simCode.stringToSimVarHT
  for name in layout.stateNames
    (_, simVar) = ht[name]
    local idx = layout.stateIndex[name]
    local startExp = _readStartAttribute(simVar)
    if startExp === nothing
      push!(stmts, :(u0[$idx] = 0.0))
    else
      push!(stmts, :(u0[$idx] = $(expToJuliaExpDE(startExp, layout, simCode))))
    end
  end
  return stmts
end

function _readStartAttribute(simVar::SimulationCode.SIMVAR)
  simVar.attributes === nothing && return nothing
  @match simVar.attributes begin
    SOME(attrs) => @match attrs.start begin
      SOME(s) => s
      NONE() => nothing
    end
    NONE() => nothing
  end
end

"""
Generate the residual-equation body of the RHS function.
"""
function generateDERHSBody(layout::DELayout,
                           simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local stmts = Expr[]
  for eq in simCode.residualEquations
    push!(stmts, residualToDerivativeAssignment(eq, layout, simCode))
  end
  return stmts
end

"""
Top-level DEMode emitter. Mirrors the shape of `ODE_MODE_MTK` (returns
`(modelName, moduleExpr)`) but the body is a pure DifferentialEquations.jl
problem, no MTK call.
"""
function generateDECode(simCode::SimulationCode.SIM_CODE)
  local layout = buildDELayout(simCode)
  if !isempty(layout.algNames)
    error("DEMode: model '$(simCode.name)' has $(length(layout.algNames)) algebraic " *
          "variable(s) after alias elimination. Pure-ODE scope only.")
  end
  if length(layout.stateNames) != length(simCode.residualEquations)
    error("DEMode: model '$(simCode.name)' is not balanced as pure ODE: " *
          "$(length(layout.stateNames)) state(s) vs $(length(simCode.residualEquations)) residual(s).")
  end

  local MODEL_NAME = simCode.name
  local nStates = length(layout.stateNames)
  local nParams = length(layout.paramNames)

  local rhsBody = generateDERHSBody(layout, simCode)
  local paramAssignments = generateDEParameterAssignments(layout, simCode)
  local startConditions = generateDEStartConditions(layout, simCode)

  local code = quote
    using DifferentialEquations
    using OrdinaryDiffEq

    const STATE_NAMES = $(layout.stateNames)
    const PARAM_NAMES = $(layout.paramNames)
    const STATE_INDEX = $(layout.stateIndex)
    const PARAM_INDEX = $(layout.paramIndex)

    function $(Symbol("$(MODEL_NAME)RHS!"))(du, u, p, t)
      $(rhsBody...)
      return nothing
    end

    function $(Symbol("$(MODEL_NAME)ParameterVars"))()
      local p = Vector{Float64}(undef, $nParams)
      $(paramAssignments...)
      return p
    end

    function $(Symbol("$(MODEL_NAME)StartConditions"))(p)
      local u0 = Vector{Float64}(undef, $nStates)
      $(startConditions...)
      return u0
    end

    function $(Symbol("$(MODEL_NAME)Model"))(tspan = (0.0, 1.0))
      local p = $(Symbol("$(MODEL_NAME)ParameterVars"))()
      local u0 = $(Symbol("$(MODEL_NAME)StartConditions"))(p)
      return ODEProblem($(Symbol("$(MODEL_NAME)RHS!")), u0, tspan, p)
    end

    function simulate(tspan = (0.0, 1.0), solver = Tsit5(); kwargs...)
      local prob = $(Symbol("$(MODEL_NAME)Model"))(tspan)
      return DifferentialEquations.solve(prob, solver; kwargs...)
    end
  end
  local moduleExpr = Expr(:module, true, Symbol(MODEL_NAME), stripBeginBlocks(code))
  return (MODEL_NAME, moduleExpr)
end


end #= module DEGen =#
