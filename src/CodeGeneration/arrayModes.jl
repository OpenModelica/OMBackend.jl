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
  The array models of variable structure systems (OMJL): a module of array modules, one per structural mode (the
  modes of `structuralmode`, a model recompiled at a structural change), and the run that switches between them at
  their structural changes (`simulateModes`).

  Author: John Tinnerholm
=#

"""
    structuralModesModule(name, modes, initialMode, frontendModel) -> Expr

The module `name` of the array model of a variable structure system: the array modules of its modes (`modes`: each
mode's name and module, `generateArrayPreserveDECode` with the model as their owner), started in `initialMode`, and
the frontend's model `frontendModel` (the SCode class a recompilation changes, `nothing` without one). It has the
functions of an array model's module that a simulation calls: `<name>Model(tspan; parameters)`, the initial mode's
problem and callbacks, and `simulate(tspan, solver; parameters, kwargs...)`, the solutions of the modes in their
order (`simulateModes`). The external objects of the model's build before this one are freed when it is evaluated,
those of all its modes are the model's.
"""
function structuralModesModule(name::String, modes::Vector{Tuple{String, Expr}}, initialMode::String,
                               @nospecialize(frontendModel))::Expr
  local modeModules::Vector{Any} = Any[m[2] for m in modes]
  local pairs::Vector{Any} = Any[:($(m[1]) => $(m[2].args[2]::Symbol)) for m in modes]
  local code::Expr = quote
    using DifferentialEquations
    const OMBackend = $(_OMBACKEND)
    $(_OMBACKEND.CodeGeneration.freeExternalObjects!)($(_OMBACKEND.canonicalName(name)))
    $(modeModules...)
    #= the modes by name, the one the simulation starts in =#
    const MODES = Dict{String, Module}($(pairs...))
    const INITIAL_MODE = $(initialMode)
    const MODEL_NAME = $(name)
    #= of each mode the modes it switches to by name (its family: the model's modes, or those of a recompilation of
       them) and the frontend's model a recompilation changes; the families recompiled during the simulations, by the
       family, element and value of their change (`_recompiledMode`) =#
    const FAMILIES = IdDict{Module, Dict{String, Module}}(m => MODES for m in values(MODES))
    const FRONTEND_MODELS = IdDict{Module, Any}(m => $(QuoteNode(frontendModel)) for m in values(MODES))
    const RECOMPILED = Dict{Tuple{UInt, String, Float64}, Dict{String, Module}}()

    function $(Symbol(name, "Model"))(tspan = (0.0, 1.0); parameters::AbstractDict = Dict{String, Any}())
      local m::Module = MODES[INITIAL_MODE]
      return Base.invokelatest(getfield(m, Symbol(nameof(m), "Model")), tspan;
                               parameters = $(_modeParameters)(m, parameters))
    end

    simulate(tspan = (0.0, 1.0), solver = OMBackend.defaultSolver(); kwargs...) =
      $(simulateModes)(@__MODULE__, tspan, solver; kwargs...)
  end
  return Expr(:module, true, Symbol(name), stripBeginBlocks(code))
end

"""
    simulateModes(model, tspan, solver; parameters, kwargs...) -> Vector{Any}

The simulation of the array model of a variable structure system (`model`, `structuralModesModule`) over `tspan`: its
initial mode's from the start, then at each structural change (`_structuralChange!`, which ends the mode's run) the
next mode's from the change to the end, started from the values at the change (`_valuesAtEnd`) by name (`_carried`):
an explicit transition's target mode of the mode's family (`FAMILIES`), or the mode of that name of the family
recompiled with the element's new value (`_recompiledMode`). The
solutions of the modes, in their order. `parameters` reach each mode that has them; the solver options
(`kwargs`) each mode's run. Modes that change again at the instant they start, `MODE_CHANGES_AT_AN_INSTANT` times, stop
the simulation (an error: the changes do not settle).
"""
function simulateModes(model::Module, tspan::Tuple{Real, Real}, solver; parameters::AbstractDict = Dict{String, Any}(),
                       kwargs...)::Vector{Any}
  _checkModeParameters(model, parameters)
  local sols::Vector{Any} = Any[]
  local modeName::String = model.INITIAL_MODE
  local mode::Module = model.MODES[modeName]
  local starts::Union{Nothing, Dict{String, Float64}} = nothing
  local t0::Float64 = Float64(tspan[1])
  local tEnd::Float64 = Float64(tspan[2])
  local atInstant::Int = 0
  while true
    atInstant < MODE_CHANGES_AT_AN_INSTANT ||
      error(string("array model ", model.MODEL_NAME, ": ", atInstant, " structural changes at t = ", t0,
                   " (the modes change again where they start)"))
    #= a mode recompiled during the run is newer than this call: its bindings read in the latest world =#
    local run::Function = Base.invokelatest(getglobal, mode, :simulate)
    local own::Dict{String, Any} = Base.invokelatest(_modeParameters, mode, parameters)
    local sol::DiffEqBase.SciMLBase.AbstractTimeseriesSolution =
      Base.invokelatest(run, (t0, tEnd), solver; parameters = own, starts, kwargs...)
    push!(sols, sol)
    local p::ArrayModelParameters = sol.prob.p
    local (k::Int, value::Float64) = p.state.structural
    (k == 0 || sol.t[end] >= tEnd) && return sols
    local values::Dict{String, Float64} = Base.invokelatest(_valuesAtEnd, sol)
    local changes::Vector{Tuple{Symbol, String, Symbol}} = Base.invokelatest(getglobal, mode, :STRUCTURAL_CHANGES)
    local (kind::Symbol, target::String, valueType::Symbol) = changes[k]
    if kind === :transition
      local next::Module = model.FAMILIES[mode][target]
      starts = Base.invokelatest(_carried, values, next, modeName, target)
      (mode, modeName) = (next, target)
    else
      mode = _recompiledMode(model, mode, modeName, target, _typedValue(value, valueType))
      starts = Base.invokelatest(_carried, values, mode, modeName, modeName)
    end
    atInstant = sol.t[end] == t0 ? atInstant + 1 : 0
    t0 = Float64(sol.t[end])
  end
end

"How many structural changes one instant takes before the simulation stops (`simulateModes`)."
const MODE_CHANGES_AT_AN_INSTANT = 100

"""
    _valuesAtEnd(sol) -> Dict{String, Float64}

The values at the end of a mode's solution `sol` (a structural change), by the mode's names: its states, algebraic
variables (after the change's event) and discrete variables.
"""
function _valuesAtEnd(sol::DiffEqBase.SciMLBase.AbstractTimeseriesSolution)::Dict{String, Float64}
  local p::ArrayModelParameters = sol.prob.p
  local m::Module = p.model
  local u::Vector{Float64} = sol.u[end]
  local t::Float64 = sol.t[end]
  local values::Dict{String, Float64} = Dict{String, Float64}()
  for (i::Int, n::String) in enumerate(m.STATE_NAMES)
    values[n] = u[i]
  end
  local right::Bool = length(sol.t) > 1 && sol.t[end - 1] == t
  local a::Vector{Float64} = m.algebraics(u, p, t, right)
  for (j::Int, n::String) in enumerate(m.ALGEBRAIC_NAMES)
    values[n] = a[j]
  end
  for (k::Int, n::String) in enumerate(m.DISCRETE_NAMES)
    values[n] = p.d[k]
  end
  return values
end

"""
    _carried(values, target, from, to) -> Dict{String, Float64}

The starts of the mode `target` (named `to`) entered from the mode `from` with the values `values` at the change: its
states and discrete variables of a name the mode before has, `to.x` that of `from.x` (the modes of `structuralmode`),
another name its own (the model's variables outside the modes, a recompiled mode's).
"""
function _carried(values::Dict{String, Float64}, target::Module, from::String, to::String)::Dict{String, Float64}
  local starts::Dict{String, Float64} = Dict{String, Float64}()
  local prefix::String = string(to, ".")
  for n::String in Iterators.flatten((target.STATE_NAMES, target.DISCRETE_NAMES))
    local source::String = (from != to && startswith(n, prefix)) ? string(from, ".", SubString(n, ncodeunits(prefix) + 1)) : n
    local v::Union{Nothing, Float64} = get(values, source, nothing)
    v === nothing || (starts[n] = v)
  end
  return starts
end

"""
    _typedValue(value, valueType) -> Union{Bool, Int, Float64}

A recompilation's new value as its element's type (`:Integer`, `:Boolean` or `:Real`, `STRUCTURAL_CHANGES`): the
binding the frontend gives the element (an Integer's `3`, not `3.0`).
"""
function _typedValue(value::Float64, valueType::Symbol)::Union{Bool, Int, Float64}
  valueType === :Integer && return round(Int, value)
  valueType === :Boolean && return value != 0
  return value
end

"""
    _recompiledMode(model, origin, modeName, element, value) -> Module

The mode `modeName` of the family (`model.FAMILIES`) the family of the mode `origin` of `model` gives with its element
`element` set to `value` (a recompilation: all its modes, a later transition goes to one of them): compiled before
for the same change (`model.RECOMPILED`), or now from the frontend's model of `origin`
(`OMBackend.recompiledArrayModules`), evaluated into `model`, with their own frontend model for the changes after it.
"""
function _recompiledMode(model::Module, origin::Module, modeName::String, element::String,
                         value::Union{Bool, Int, Float64})::Module
  local family::Dict{String, Module} = model.FAMILIES[origin]
  local key::Tuple{UInt, String, Float64} = (objectid(family), element, Float64(value))
  local cached::Union{Nothing, Dict{String, Module}} = get(model.RECOMPILED, key, nothing)
  cached === nothing || return cached[modeName]
  local frontendModel::Any = get(model.FRONTEND_MODELS, origin, nothing)
  frontendModel === nothing &&
    error(string("array model ", model.MODEL_NAME, ": a recompilation of ", element, " in a mode without its frontend model"))
  local moduleName::String = string("Recompiled_", length(model.RECOMPILED) + 1)
  local (codes::Vector{Tuple{String, Expr}}, changed::Any) =
    _OMBACKEND.recompiledArrayModules(model.MODEL_NAME, frontendModel, element, value, moduleName)
  local recompiled::Dict{String, Module} = Dict{String, Module}()
  for (name::String, code::Expr) in codes
    Core.eval(model, code)
    recompiled[name] = Base.invokelatest(getglobal, model, code.args[2]::Symbol)
  end
  for m::Module in values(recompiled)
    model.FAMILIES[m] = recompiled
    model.FRONTEND_MODELS[m] = changed
  end
  model.RECOMPILED[key] = recompiled
  return recompiled[modeName]
end

"Whether the array module `mode` has the parameter or parameter element `name` (Modelica's or canonical)."
_hasParameter(mode::Module, name::String)::Bool =
  haskey(mode.PARAMETER_NAMES, name) || haskey(mode.PARAMETER_ELEMENTS, name) || haskey(mode.STRING_PARAMETERS, name)

"""
    _modeParameters(mode, parameters) -> Dict{String, Any}

The overrides of `parameters` the mode `mode` has (`_hasParameter`): a mode's parameters are its own, another mode's
are not passed to it.
"""
function _modeParameters(mode::Module, parameters::AbstractDict)::Dict{String, Any}
  local own::Dict{String, Any} = Dict{String, Any}()
  for (k, v) in parameters
    local n::String = string(k)
    _hasParameter(mode, n) && (own[n] = v)
  end
  return own
end

"An override of `parameters` that none of the modes of `model` has: an ArgumentError."
function _checkModeParameters(model::Module, parameters::AbstractDict)::Nothing
  for k in keys(parameters)
    local n::String = string(k)
    any(m::Module -> _hasParameter(m, n), values(model.MODES)) ||
      throw(ArgumentError(string("array model ", model.MODEL_NAME, ": no mode has the parameter ", n)))
  end
  return nothing
end

"""
    modesAreTunable(model, names) -> Bool

`areTunable` of the array model of a variable structure system (`structuralModesModule`): each of `names` a
parameter of one of its modes at least, and each mode takes the overrides of those it has.
"""
function modesAreTunable(model::Module, names::Vector{String})::Bool
  for n::String in names
    any(m::Module -> _hasParameter(m, n), values(model.MODES)) || return false
  end
  for m::Module in values(model.MODES)
    local own::Vector{String} = filter(n::String -> _hasParameter(m, n), names)
    (isempty(own) || areTunable(m, own)) || return false
  end
  return true
end
