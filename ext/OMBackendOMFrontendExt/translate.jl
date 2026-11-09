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

"""
    translate(fm; functionList = nothing, BackendMode = DEFAULT_BACKEND_MODE[], warnMissingStartValues = nothing,
              eliminateNonDynamic = nothing, observedFilter = nothing, checkSimCode = true, scalarized = true,
              omjlir = nothing)

The target code of the flat model `fm` and its functions (the frontend's function list): `fm`
lowered to OMJLIR and through its passes (`omjlirModel`; `omjlir`: that OMJLIR model, already
lowered, `translateReducedArrays`), to the simulation code and its
passes (SimCode.simulationCode; SimCode.deSimulationCode for DEMode), checked
(`checkSimCode`), and the code generated for the backend mode. A flat model with arrays kept
(`scalarized = false`) is lowered with its arrays and goes to the array-preserving generation first
(`reducedArrayModel`, ARRAY_ODE_GENERATION), to this path where it is outside its scope.
"""
function translate(fm::F.FlatModel;
                   functionList::Union{Nothing, FunctionList} = nothing,
                   BackendMode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[],
                   warnMissingStartValues::Union{Nothing, Bool} = nothing,
                   eliminateNonDynamic::Union{Nothing, Bool, SimCode.EliminationOptions} = nothing,
                   observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                   checkSimCode::Bool = true,
                   scalarized::Bool = true,
                   omjlir::Union{Nothing, OMJLIR.Model} = nothing)
  local previousWarnSetting = WARN_MISSING_START_VALUES[]
  local modelName = string(fm.name)
  local runId = createLogRunId(modelName)
  MODEL_RUN_DIRS[canonicalName(modelName)] = runId
  resetFallbacks!()
  warnMissingStartValues === nothing || (WARN_MISSING_START_VALUES[] = warnMissingStartValues)
  try
    delete!(ARRAY_ODE_MODELS, canonicalName(fm.name))
    omjlir === nothing && delete!(ARRAY_REDUCED_MODELS, canonicalName(fm.name))
    #= an attempt with the arrays kept records its own reasons (as translateReducedArrays); a scalarized translate after
       it, or the reduced model's (`omjlir`), keeps them for OM's report =#
    (scalarized || omjlir !== nothing) || delete!(ARRAY_PATH_REASONS, canonicalName(fm.name))
    #= the index reduced in array form: its array code, else its scalarized form =#
    local arrayReduction::Bool = !scalarized && omjlir === nothing
    if arrayReduction
      local reduced::Union{Nothing, OMJLIR.Model} = @BACKEND_PERFLOG "[backendAPI] reducedArrayModel" reducedArrayModel(fm; functionList)
      if reduced !== nothing
        local reducedResult::Union{Nothing, Tuple{String, Expr}} = @BACKEND_PERFLOG "[backendAPI] generateReducedArrayODECode" generateReducedArrayODETargetCode(canonicalName(fm.name), reduced)
        reducedResult === nothing || return reducedResult
        omjlir = scalarizedReducedModel(canonicalName(fm.name), reduced)
      end
    end
    return withLogRunDir(runId) do
      #= The flat model with its functions as it arrives from the frontend =#
      @BACKEND_LOGGING debugWrite(logPath("backend/simCode", "frontend_initialModel.log"),
        replace(F.toFlatString(fm, functionList === nothing ? MetaModelica.nil : functionList), "\\n" => "\n"))
      local model::OMJLIR.Model = omjlir !== nothing ? omjlir :
        @BACKEND_PERFLOG "[backendAPI] lower" omjlirModel(fm; functionList, scalarized, arrayReduction = !arrayReduction)
      return generateTargetCode(model; BackendMode, eliminateNonDynamic, observedFilter, checkSimCode)
    end
  finally
    WARN_MISSING_START_VALUES[] = previousWarnSetting
  end
end

function explainIndexReduction(fm::F.FlatModel;
                               functionList::Union{Nothing, FunctionList} = nothing)::Passes.IndexReduction.Explanation
  local model = lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  #= the passes but index reduction (it runs after the last one) =#
  model = Passes.runPasses!(model; through = first(last(Passes.PIPELINE)))
  return Passes.IndexReduction.explain(model)
end

function omjlirModel(fm::F.FlatModel; functionList::Union{Nothing, FunctionList} = nothing,
                     scalarized::Bool = true, arrayReduction::Bool = true)::OMJLIR.Model
  if !scalarized && arrayReduction
    local reduced = arrayReducedModel(fm; functionList)
    reduced === nothing || return reduced
  end
  scalarized || (fm = @BACKEND_PERFLOG "[backendAPI] scalarizeKeptArrays" F.scalarizeKeptArrays(fm))
  local model = lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  return Passes.runPasses!(model)
end

function arrayReducedModel(fm::F.FlatModel;
                           functionList::Union{Nothing, FunctionList} = nothing)::Union{Nothing, OMJLIR.Model}
  local reduced::Union{Nothing, OMJLIR.Model} = reducedArrayModel(fm; functionList)
  return reduced === nothing ? nothing : scalarizedReducedModel(canonicalName(fm.name), reduced)
end

function reducedArrayModel(fm::F.FlatModel;
                           functionList::Union{Nothing, FunctionList} = nothing)::Union{Nothing, OMJLIR.Model}
  return reducedArrayModel(canonicalName(fm.name)) do
    lowerFlatModel(fm; functions = something(functionList, MetaModelica.nil))
  end
end

function translateReducedArrays(fm::F.FlatModel;
                                functionList::Union{Nothing, FunctionList} = nothing,
                                BackendMode::OMBackend.BackendMode = DEFAULT_BACKEND_MODE[],
                                warnMissingStartValues::Union{Nothing, Bool} = nothing,
                                eliminateNonDynamic::Union{Nothing, Bool, SimCode.EliminationOptions} = nothing,
                                observedFilter::Union{Nothing, Vector{String}, Vector{Regex}} = nothing,
                                checkSimCode::Bool = true)
  delete!(ARRAY_ODE_MODELS, canonicalName(fm.name))
  delete!(ARRAY_PATH_REASONS, canonicalName(fm.name))
  local reduced::Union{Nothing, OMJLIR.Model} = @BACKEND_PERFLOG "[backendAPI] reducedArrayModel" reducedArrayModel(fm; functionList)
  reduced === nothing && return nothing
  local arrayCode::Union{Nothing, Tuple{String, Expr}} =
    @BACKEND_PERFLOG "[backendAPI] generateReducedArrayODECode" generateReducedArrayODETargetCode(canonicalName(fm.name), reduced)
  arrayCode === nothing || return arrayCode
  local model::Union{Nothing, OMJLIR.Model} = scalarizedReducedModel(canonicalName(fm.name), reduced)
  model === nothing && return nothing
  local name = canonicalName(fm.name)
  #= The array form only where it translates and builds; otherwise its build is forgotten and OM's translate
     scalarizes as before. Coverage 21 (the array form's first): a lifted if-expression left a for-equation's
     iterator free (UndefVarError), a build failed (BoundsError) and the tables it had allocated were freed
     under the problem that read them (Rotational Friction, CoupledClutches, Translational Brake). =#
  local decline = (reason::String) -> begin
    @info "[index reduction] $(name): arrays kept, the reduced model $(reason); scalarized in the frontend"
    declineArrayPath!(name, "array index reduction", "the reduced model " * reason)
    IMTKGen.forgetBuild(name)
    delete!(COMPILED_MODELS_MTK, name)
    delete!(ARRAY_REDUCED_MODELS, name)
    nothing
  end
  local result = try
    translate(fm; functionList, BackendMode, warnMissingStartValues, eliminateNonDynamic, observedFilter, checkSimCode,
              omjlir = model)
  catch err
    isFatal(err) && rethrow()
    return decline("does not translate ($(first(sprint(showerror, err), 200)))")
  end
  (BackendMode == IMTK_MODE && !haskey(IMTKGen.BUILT, name)) && return decline("does not build")
  return result
end
