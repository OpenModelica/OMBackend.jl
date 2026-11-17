#= Helpers of the array path tests (arrayPathTests.jl, arrayPathRoutesTests.jl): the models of Models/ArrayPath.mo and
   of Modelica texts translated on either route. =#
const ARRAY_PATH_FILE = joinpath(@__DIR__, "Models", "ArrayPath.mo")
const _ARRAY_PATH_SCODE = Ref{Any}(nothing)

function _arrayPathTranslate(model::String; scalarized::Bool)
  _ARRAY_PATH_SCODE[] === nothing &&
    (_ARRAY_PATH_SCODE[] = OMFrontend.translateToSCode(OMFrontend.parseFile(ARRAY_PATH_FILE)))
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, _ARRAY_PATH_SCODE[]; scalarize = scalarized)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarized = scalarized)
  return OMBackend.canonicalName(model)
end

#= A test's own models written beside it, `Modelica"""..."""` (OMParser 1.0: parsed when this file is loaded), for
   `_bothWaysOf`; the models of Models/ArrayPath.mo are those of the tests before. =#
const var"@Modelica_str" = OMFrontend.OMParser.var"@Modelica_str"

"The model `model` of the Modelica text `program` translated (as `_arrayPathTranslate`); its name."
function _translateProgram(program, model::String; scalarized::Bool)::String
  local scode = OMFrontend.translateToSCode(program)
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, scode; scalarize = scalarized)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarized = scalarized)
  return OMBackend.canonicalName(model)
end

_arrayPathTaken(name) = name in OMBackend.ARRAY_ODE_MODELS
_values(sol, names) = [OMBackend.getVariableValues(sol, n)[end] for n in names]
