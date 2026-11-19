#= The names of OMBackend 0.x that 1.0 renamed: kept for all of 1.x with a deprecation warning (shown with
   `--depwarn=yes`), not public, not documented but on the API page's list (docs/src/api.md). =#

"""
    _deprecatedKeyword(f, old, value, new, current) -> value or current

`value` of the deprecated keyword `old` of `f` where it was given (`value` not `nothing`), with a deprecation warning;
else `current`, the value of its new name `new`.
"""
function _deprecatedKeyword(f::Symbol, old::Symbol, value::Union{Nothing, T}, new::Symbol, current::T)::T where {T}
  value === nothing && return current
  Base.depwarn("the keyword `$(old)` of `$(f)` is deprecated, use `$(new)`", f)
  return value
end

function simulateModel(modelName::String; MODE::BackendMode = DEFAULT_BACKEND_MODE[], kwargs...)
  Base.depwarn("`simulateModel(name; MODE)` is deprecated, use `simulate(name; mode)`", :simulateModel)
  return simulate(modelName; mode = MODE, kwargs...)
end

function resimulateModel(modelName::String; MODE::BackendMode = DEFAULT_BACKEND_MODE[], kwargs...)
  Base.depwarn("`resimulateModel(name; MODE)` is deprecated, use `resimulate(name; mode)`", :resimulateModel)
  return resimulate(modelName; mode = MODE, kwargs...)
end

function getVariableValues(sol::ODESolution, varName::String)
  Base.depwarn("`getVariableValues` is deprecated, use `variableValues`", :getVariableValues)
  return variableValues(sol, varName)
end

function getVariableValues(sols::Vector, variables...)
  Base.depwarn("`getVariableValues` is deprecated, use `variableValues`", :getVariableValues)
  return variableValues(sols, variables...)
end

function translateReducedArrays(fm; BackendMode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[], kwargs...)
  Base.depwarn("`translateReducedArrays(fm; BackendMode)` is deprecated, use `translate(fm; scalarize = false, mode)`",
               :translateReducedArrays)
  return translate(fm; scalarize = false, mode = BackendMode, kwargs...)
end
