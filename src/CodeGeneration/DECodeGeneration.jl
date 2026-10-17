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
Direct DifferentialEquations.jl code generation (DEMode).

Builds a self-contained Julia module that defines:
  - <Model>RHS!(du, u, p, t)        in-place ODE RHS, integer-indexed
  - <Model>StartConditions(p)       state initial vector u0
  - <Model>ParameterVars()          parameter vector p
  - <Model>Model(tspan)             constructs ODEProblem
  - simulate(tspan, solver; ...)    user-facing entry point

The emitter intentionally does NOT depend on ModelingToolkit, on
`expToJuliaExpMTK`, or on any of the MTK helpers. It mirrors the
SIM_CODE -> Julia mapping of `expToJuliaExp` in `codeGen.jl`
=#

module DEGen

import ..SimCode
import ..CodeGeneration
import ..CodeGeneration: opKindToJuliaOperator, stripBeginBlocks

const SC = SimCode

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
function buildDELayout(simCode::SC.SIM_CODE)::DELayout
  local stateNames = String[]
  local algNames = String[]
  local paramNames = String[]
  local discreteNames = String[]
  for (name, (_, simVar)) in simCode.stringToSimVarHT
    local k = simVar.varKind
    k isa SC.STATE && push!(stateNames, name)
    k isa SC.PARAMETER && push!(paramNames, name)
    k isa SC.ALG_VARIABLE && push!(algNames, name)
    k isa SC.DISCRETE && push!(discreteNames, name)
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
    expToJuliaExpDE(e, layout, simCode)

An expression for the DE backend: states from `u[i]`, parameters from `p[i]`, time from `t`,
state derivatives `du[i]` (the DELayout's integer indices). An algebraic or a discrete
variable is refused (the pure-ODE scope), and so is any other expression.
"""
function expToJuliaExpDE end

expToJuliaExpDE(e::Union{SC.BCONST, SC.ICONST, SC.RCONST, SC.SCONST}, ::DELayout, ::SC.SIM_CODE)::Expr = Expr(:block, e.value)

function expToJuliaExpDE(e::SC.EXP_CREF, layout::DELayout, simCode::SC.SIM_CODE)::Expr
  local varName = SC.crefName(e.cref)
  varName == "time" && return Expr(:block, :t)
  local entry = get(simCode.stringToSimVarHT, varName, nothing)
  entry === nothing && error("DEMode: unknown CREF '$(varName)' in expression lowering")
  local kind = last(entry).varKind
  kind isa SC.STATE && return Expr(:ref, :u, layout.stateIndex[varName])
  kind isa SC.STATE_DERIVATIVE && return Expr(:ref, :du, layout.stateIndex[kind.varName])
  kind isa SC.PARAMETER && return Expr(:ref, :p, layout.paramIndex[varName])
  kind isa SC.ALG_VARIABLE && error("DEMode: algebraic variable '$(varName)' encountered. Pure-ODE scope only at this milestone.")
  kind isa SC.DISCRETE && error("DEMode: discrete variable '$(varName)' encountered. Discrete/event support is not in the initial milestone.")
  error("DEMode: unsupported variable kind $(kind) for '$(varName)'")
end

expToJuliaExpDE(e::SC.DER, layout::DELayout, ::SC.SIM_CODE)::Expr = Expr(:ref, :du, layout.stateIndex[SC.crefName(e.cref)])

expToJuliaExpDE(e::Union{SC.UNARY, SC.LUNARY}, layout::DELayout, simCode::SC.SIM_CODE)::Expr =
  :($(opKindToJuliaOperator(e.op))($(expToJuliaExpDE(e.exp, layout, simCode))))

expToJuliaExpDE(e::Union{SC.BINARY, SC.RELATION}, layout::DELayout, simCode::SC.SIM_CODE)::Expr =
  :($(opKindToJuliaOperator(e.op))($(expToJuliaExpDE(e.exp1, layout, simCode)), $(expToJuliaExpDE(e.exp2, layout, simCode))))

expToJuliaExpDE(e::SC.LBINARY, layout::DELayout, simCode::SC.SIM_CODE)::Expr =
  Expr(e.op === SC.OP_AND ? :&& : :||, expToJuliaExpDE(e.exp1, layout, simCode), expToJuliaExpDE(e.exp2, layout, simCode))

"An if-expression inlined: it survives where the if-expression-to-if-equation pass leaves expressions that depend on time."
expToJuliaExpDE(e::SC.IFEXP, layout::DELayout, simCode::SC.SIM_CODE)::Expr =
  :(($(expToJuliaExpDE(e.cond, layout, simCode))) ? $(expToJuliaExpDE(e.thenExp, layout, simCode)) : $(expToJuliaExpDE(e.elseExp, layout, simCode)))

expToJuliaExpDE(e::SC.CAST, layout::DELayout, simCode::SC.SIM_CODE)::Expr = expToJuliaExpDE(e.exp, layout, simCode)

expToJuliaExpDE(e::SC.CALL, layout::DELayout, simCode::SC.SIM_CODE)::Expr =
  Expr(:call, e.name, (expToJuliaExpDE(a, layout, simCode) for a in e.args)...)

expToJuliaExpDE(e::SC.Exp, ::DELayout, ::SC.SIM_CODE)::Expr = error("DEMode: unsupported expression $(typeof(e)): $e")

"""
Lower a single residual equation (lhs - rhs == 0) to a Julia statement
of the form `du[i] = <rhs>`. The emitter assumes the residual has been
pre-rewritten so `der(state)` appears solitary on one side; another
residual is refused.
"""
function residualToDerivativeAssignment(eq::SC.RESIDUAL_EQUATION, layout::DELayout, simCode::SC.SIM_CODE)::Expr
  local e = eq.exp
  (e isa SC.BINARY && e.op === SC.OP_SUB) || error("DEMode: residual has unexpected top-level shape: $(e)")
  local (derivative, value) = e.exp1 isa SC.DER ? (e.exp1, e.exp2) : e.exp2 isa SC.DER ? (e.exp2, e.exp1) :
    error("DEMode: residual is not a pure ODE for any state. Equation: $(e)")
  local stateName = SC.crefName(derivative.cref)
  local idx = get(layout.stateIndex, stateName, nothing)
  idx === nothing && error("DEMode: der() of unknown state '$(stateName)'")
  return :(du[$idx] = $(expToJuliaExpDE(value, layout, simCode)))
end

"""
Build the parameter array initializer.
Returns an `Expr(:block, ...)` that writes each parameter binding into `p`
in dependency order (we emit textual order; chained parameter dependencies
are already resolved by `propagateConstants` upstream).
"""
function generateDEParameterAssignments(layout::DELayout, simCode::SC.SIM_CODE)::Vector{Expr}
  local stmts = Expr[]
  local ht = simCode.stringToSimVarHT
  for name in layout.paramNames
    local simVar = last(ht[name])
    simVar.varKind isa SC.PARAMETER || error("DEMode: parameter '$(name)' has non-parameter SimVarType")
    local bindExp = simVar.varKind.bindExp
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
function generateDEStartConditions(layout::DELayout, simCode::SC.SIM_CODE)::Vector{Expr}
  local stmts = Expr[]
  local ht = simCode.stringToSimVarHT
  for name in layout.stateNames
    local simVar = last(ht[name])
    local idx = layout.stateIndex[name]
    local startExp = simVar.attributes.start
    if startExp === nothing
      push!(stmts, :(u0[$idx] = 0.0))
    else
      push!(stmts, :(u0[$idx] = $(expToJuliaExpDE(startExp, layout, simCode))))
    end
  end
  return stmts
end

"""
Generate the residual-equation body of the RHS function.
"""
generateDERHSBody(layout::DELayout, simCode::SC.SIM_CODE)::Vector{Expr} =
  Expr[residualToDerivativeAssignment(eq, layout, simCode) for eq in simCode.residualEquations]

"""
Top-level DEMode emitter. Mirrors the shape of `ODE_MODE_MTK` (returns
`(modelName, moduleExpr)`) but the body is a pure DifferentialEquations.jl
problem, no MTK call.
"""
function generateDECode(simCode::SC.SIM_CODE)
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
