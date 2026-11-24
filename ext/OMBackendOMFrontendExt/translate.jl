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

#= The backend's entry points on OMFrontend's flat model: lowered (`LowerFlatModel`), through the passes, to the
   target code; the array-preserving route first where the arrays are kept. Methods of the functions OMBackend
   declares (frontendInterface.jl); the steps on OMJLIR alone are OMBackend's. =#

"The keyword arguments of `translate` its routes pass on."
const TranslateOptions = NamedTuple{(:functionList, :mode, :warnMissingStartValues, :eliminateNonDynamic,
                                     :observedFilter, :checkSimCode)}

"""
    translate(fm; functionList = nothing, mode = DEFAULT_BACKEND_MODE[], scalarize = true,
              warnMissingStartValues = nothing, eliminateNonDynamic = nothing, observedFilter = nothing,
              checkSimCode = true) -> Union{Nothing, Tuple{String, Expr}}

The target code of the flat model `fm` and its functions (the frontend's function list, `functionList`), for the
backend mode `mode` (a [`BackendMode`](@ref)): the model name and the generated module.

`scalarize = true` (`fm` flattened with its arrays scalarized): `fm` lowered to OMJLIR and through its passes
(`omjlirModel`), to the simulation code and its passes (SimCode.simulationCode), checked (`checkSimCode`), and the
code generated for `mode`.

`scalarize = false` (`fm` flattened with its arrays kept): its index reduced in array form (`reducedArrayModel`), to
the array code (`generateReducedArrayPreserveDETargetCode`), else the reduced model scalarized
(`scalarizedReducedModel`); where the array form does not take it, or what it made does not translate or build, the
model scalarized by the backend before the scalar passes (`backendScalarizedModel`, recorded in
BACKEND_SCALARIZED_MODELS). `nothing` where none of these takes the model; [`scalarizedFallbackReason`](@ref) gives
the reasons (OM's translate then throws `ARRAY_PRESERVING_CODEGEN_NOT_SUPPORTED`).
"""
function translate(fm::F.FLAT_MODEL;
                   functionList::Union{Nothing, FunctionList} = nothing,
                   mode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[],
                   scalarize::Bool = true,
                   warnMissingStartValues::Union{Nothing, Bool} = nothing,
                   eliminateNonDynamic::SimCode.EliminationSetting = nothing,
                   observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                   checkSimCode::Bool = true,
                   BackendMode::Union{Nothing, OMBackend.BackendMode} = nothing,
                   scalarized::Union{Nothing, Bool} = nothing)::Union{Nothing, Tuple{String, Expr}}
  mode = OMBackend._deprecatedKeyword(:translate, :BackendMode, BackendMode, :mode, mode)
  scalarize = OMBackend._deprecatedKeyword(:translate, :scalarized, scalarized, :scalarize, scalarize)
  local options::TranslateOptions = (; functionList, mode, warnMissingStartValues, eliminateNonDynamic, observedFilter,
                                     checkSimCode)
  return scalarize ? _translateScalar(fm, nothing; options...) : _translateArraysKept(fm; options...)
end

"""
`translate` of the scalar flat model `fm`: lowered to OMJLIR and through its passes (`omjlirModel`), or `omjlir`, the
OMJLIR model a route with the arrays kept made of it, to the code for `mode`.
"""
function _translateScalar(fm::F.FLAT_MODEL, omjlir::Union{Nothing, OMJLIR.Model};
                          functionList::Union{Nothing, FunctionList},
                          mode::OMBackend.BackendMode,
                          warnMissingStartValues::Union{Nothing, Bool},
                          eliminateNonDynamic::SimCode.EliminationSetting,
                          observedFilter::Union{Nothing, Vector{String}, Vector{Regex}},
                          checkSimCode::Bool)::Tuple{String, Expr}
  local previousWarnSetting::Bool = WARN_MISSING_START_VALUES[]
  local modelName::String = string(fm.name)
  local runId = createLogRunId(modelName)
  MODEL_RUN_DIRS[canonicalName(modelName)] = runId
  resetFallbacks!()
  warnMissingStartValues === nothing || (WARN_MISSING_START_VALUES[] = warnMissingStartValues)
  try
    delete!(ARRAY_PRESERVE_DE_MODELS, canonicalName(fm.name))
    omjlir === nothing && delete!(ARRAY_REDUCED_MODELS, canonicalName(fm.name))
    omjlir === nothing && delete!(BACKEND_SCALARIZED_MODELS, canonicalName(fm.name))
    return withLogRunDir(runId) do
      #= The flat model with its functions as it arrives from the frontend =#
      @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "frontend_initialModel.log"),
        replace(F.toFlatString(fm, functionList === nothing ? MetaModelica.nil : functionList), "\\n" => "\n"))
      local model::OMJLIR.Model = omjlir !== nothing ? omjlir :
        @BACKEND_PERFLOG "[backendAPI] lower" omjlirModel(fm; functionList)
      return generateTargetCode(model; mode, eliminateNonDynamic, observedFilter, checkSimCode)
    end
  finally
    WARN_MISSING_START_VALUES[] = previousWarnSetting
  end
end

function explainIndexReduction(fm::F.FLAT_MODEL;
                               functionList::Union{Nothing, FunctionList} = nothing)::Passes.IndexReduction.Explanation
  local model = lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  #= the passes but index reduction (it runs after the last one) =#
  model = Passes.runPasses!(model; through = first(last(Passes.PIPELINE)))
  return Passes.IndexReduction.explain(model)
end

function omjlirModel(fm::F.FLAT_MODEL; functionList::Union{Nothing, FunctionList} = nothing)::OMJLIR.Model
  local model = lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  return Passes.runPasses!(model)
end

function reducedArrayModel(fm::F.FLAT_MODEL;
                           functionList::Union{Nothing, FunctionList} = nothing)::Union{Nothing, OMJLIR.Model}
  return reducedArrayModel(canonicalName(fm.name)) do
    lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  end
end

function backendScalarizedModel(fm::F.FLAT_MODEL;
                                functionList::Union{Nothing, FunctionList} = nothing)::Union{Nothing, OMJLIR.Model}
  return backendScalarizedModel(canonicalName(fm.name)) do
    lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  end
end

"`translate(fm; scalarize = false)`: the routes with the arrays kept, in order; `nothing` where none takes `fm`."
function _translateArraysKept(fm::F.FLAT_MODEL;
                              functionList::Union{Nothing, FunctionList},
                              mode::OMBackend.BackendMode,
                              warnMissingStartValues::Union{Nothing, Bool},
                              eliminateNonDynamic::SimCode.EliminationSetting,
                              observedFilter::Union{Nothing, Vector{String}, Vector{Regex}},
                              checkSimCode::Bool)::Union{Nothing, Tuple{String, Expr}}
  local name::String = canonicalName(fm.name)
  delete!(ARRAY_PRESERVE_DE_MODELS, name)
  delete!(ARRAY_PATH_REASONS, name)
  delete!(BACKEND_SCALARIZED_MODELS, name)
  local options::TranslateOptions = (; functionList, mode, warnMissingStartValues, eliminateNonDynamic, observedFilter,
                                     checkSimCode)
  local result::Union{Nothing, Tuple{String, Expr}}
  local reduced::Union{Nothing, OMJLIR.Model} =
    @BACKEND_PERFLOG "[backendAPI] reducedArrayModel" reducedArrayModel(fm; functionList)
  if reduced !== nothing
    result = @BACKEND_PERFLOG "[backendAPI] generateReducedArrayPreserveDECode" generateReducedArrayPreserveDETargetCode(name,
                                                                                                           reduced)
    result === nothing || return result
    local model::Union{Nothing, OMJLIR.Model} = scalarizedReducedModel(name, reduced)
    model === nothing || (result = _translateKept(fm, model, "array index reduction", "the reduced model"; options...))
    result === nothing || return result
  end
  #= the array form does not take the model, or what it made does not translate: the kept arrays scalarized by the
     backend before the scalar passes, so the frontend does not flatten again =#
  local scalar::Union{Nothing, OMJLIR.Model} =
    @BACKEND_PERFLOG "[backendAPI] backendScalarizedModel" backendScalarizedModel(fm; functionList)
  scalar === nothing && return nothing
  result = _translateKept(fm, scalar, "backend scalarization", "the model scalarized by the backend"; options...)
  result === nothing || push!(BACKEND_SCALARIZED_MODELS, name)
  return result
end

"""
`translate` of the flat model `fm` with arrays kept from `model`, the OMJLIR model a route with the arrays kept made
of it (`path`, one of ARRAY_PATHS; `subject`: what `model` is, for the reason). Taken only where it translates and
builds: otherwise `nothing`, the reason recorded under `path`, its build forgotten, and the caller goes on with the
next route. Coverage 21 (the array form's first): a lifted if-expression left a for-equation's iterator free
(UndefVarError), a build failed (BoundsError) and the tables it had allocated were freed under the problem that read
them (Rotational Friction, CoupledClutches, Translational Brake).
"""
function _translateKept(fm::F.FLAT_MODEL, model::OMJLIR.Model, path::String, subject::String;
                        functionList::Union{Nothing, FunctionList},
                        mode::OMBackend.BackendMode,
                        warnMissingStartValues::Union{Nothing, Bool},
                        eliminateNonDynamic::SimCode.EliminationSetting,
                        observedFilter::Union{Nothing, Vector{String}, Vector{Regex}},
                        checkSimCode::Bool)::Union{Nothing, Tuple{String, Expr}}
  local name::String = canonicalName(fm.name)
  local reason::String
  try
    local result::Tuple{String, Expr} =
      _translateScalar(fm, model; functionList, mode, warnMissingStartValues, eliminateNonDynamic, observedFilter,
                       checkSimCode)
    (mode == IMTK_MODE && !haskey(IMTKGen.BUILT, name) && !(name in IMTKGen.BUILT_AT_SIMULATE)) || return result
    reason = "does not build"
  catch err
    isFatal(err) && rethrow()
    reason = "does not translate ($(first(sprint(showerror, err), 200)))"
  end
  @info "[index reduction] $(name): arrays kept, $(subject) $(reason)"
  declineArrayPath!(name, path, "$(subject) $(reason)")
  IMTKGen.forgetBuild(name)
  delete!(COMPILED_MODELS_MTK, name)
  delete!(ARRAY_REDUCED_MODELS, name)
  return nothing
end
