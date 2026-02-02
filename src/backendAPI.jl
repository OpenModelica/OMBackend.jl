#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
=#

using MetaModelica
using ExportAll
using Absyn
#= For interactive evaluation. =#
using ModelingToolkit
using SymbolicUtils
using DifferentialEquations
using OrdinaryDiffEq

import ..CodeGeneration
import ..Runtime
import .Backend.BDAE
import .Backend.BDAECreate
import .Backend.BDAEUtil
import .Backend.Causalize
import .SimulationCode

import Base.Meta
import JuliaFormatter
import OMBackend.CodeGeneration
import OMFrontend
import Plots
import REPL
import SCode

const LATEX_SYMBOLS = REPL.REPLCompletions.latex_symbols
#= Settings =#
@enum BackendMode begin
  DAE_MODE = 1
  ODE_MODE = 2 #Currently not in operation
  MTK_MODE = 3
end

function info()
  println("OMBackend.jl")
  println("A Julia backend for the Equation Oriented Language Modelica!")
  println("Run any test module by executing runExample(<model-name>)")
  println("Available Example models include:")
  for (k,v) in EXAMPLE_MODELS
    println(k)
  end
end


"""
  If this function is called a DAG representing the equations is plotted to your working directory.
"""
function plotGraph(shouldPlot::Bool)
  global PLOT_EQUATION_GRAPH = shouldPlot
end

"""
  MTK Models that have been compiled one time.
TODO: Optimaly we should keep the fronten structure
in memory as well s.t we only recompile if the structure of the source file changes
(Unless retranslation is forced).
"""
const COMPILED_MODELS_MTK = Dict()

"""
 This function lowers the given Hybrid DAE to target code.
 It does so by first lowering the code to the backend representation and then to
 the simulation code representation.
 Finally, target code is generated depending on the backend mode (defaults to MTK mode).
The function list contains the sequential parts of a modelica model, that is the different functions that the model might use.
This is not part of the lowering process but it is to be generated before we generate MTK target code
"""
function translate(frontendDAE::Union{DAE.DAE_LIST, OMFrontend.Frontend.FlatModel};
                   functionList = nothing, BackendMode = MTK_MODE)::Tuple{String, Expr}
  local bDAE = lower(frontendDAE)
  local simCode
  if BackendMode == DAE_MODE
    throw("DAE-mode is deprecated.")
  elseif BackendMode == MTK_MODE
    #@debug "Generate simulation code"
    simCode = generateSimulationCode(bDAE; mode = MTK_MODE)
    (simCodeFunctions, externalRuntimeNeeded) = if functionList !== nothing
      generateSimCodeFunctions(functionList)
    else
      (SimulationCode.ModelicaFunction[], false)
    end
    @assign simCode.functions = simCodeFunctions
    @assign simCode.externalRuntime = externalRuntimeNeeded
    #= Dump before record flattening =#
    debugWrite("simulationCodeStatistics.log", SimulationCode.dumpSimCode(simCode))
    #= Flatten record parameters in functions =#
    simCodeFunctions = SimulationCode.flattenRecordParameters(simCodeFunctions)
    @assign simCode.functions = simCodeFunctions
    #= Dump after record flattening =#
    debugWrite("simulationCodePostFlattenRecords.log", SimulationCode.dumpSimCode(simCode))
    return generateMTKTargetCode(simCode)
  else
    @error "No mode specified: valid modes are: MTK_MODE"
  end
end

"""
`function dumpInitialSystem(frontendDAE::DAE.DAE_LIST)`
 Dumps a textual representation of the initial system.
"""
function dumpInitialSystem(frontendDAE::DAE.DAE_LIST)::String
  str =  "Length of frontend DAE:" * string(length(frontendDAE.elementLst)) * "\n"
  bDAE = BDAECreate.lower(frontendDAE)
  str *= BDAEUtil.stringHeading1(bDAE, "translated")
  return str
end

"""
`function printInitialSystem(frontendDAE::DAE.DAE_LIST)`
 Dumps a textual representation of the initial system.
"""
function printInitialSystem(frontendDAE::DAE.DAE_LIST)
  print(dumpInitialSystem(frontendDAE::DAE.DAE_LIST))
end

"""
 Transforms given DAE-IR/Hybrid DAE to backend DAE-IR (BDAE-IR)
"""
function lower(frontendDAE::DAE.DAE_LIST)::BDAE.BACKEND_DAE
  local bDAE::BDAE.BACKEND_DAE
  local simCode::SIM_CODE
  @debug "Length of frontend DAE:" length(frontendDAE.elementLst)
  @assert typeof(listHead(frontendDAE.elementLst)) == DAE.COMP
  #= Create Backend structure from Frontend structure =#
  bDAE = BDAECreate.lower(frontendDAE)
  @debug(BDAEUtil.stringHeading1(bDAE, "translated"));
  #= Expand arrays =#
  (bDAE, expandedVars) = Causalize.expandArrayVariables(bDAE)
  @debug(BDAEUtil.stringHeading1(bDAE, "Array variables expanded"));
  #= Expand Array variables in equation system=#
  bDAE = Causalize.detectAndReplaceArrayVariables(bDAE, expandedVars)
  @debug(BDAEUtil.stringHeading1(bDAE, "Equation system variables expanded"));
  #= Transform if expressions to if equations =#
  @debug(BDAEUtil.stringHeading1(bDAE, "if equations transformed"));
  bDAE = Causalize.detectIfExpressions(bDAE)
  #= Mark state variables =#
  bDAE = Causalize.detectStates(bDAE)
  @debug(BDAEUtil.stringHeading1(bDAE, "states marked"));
  #= We always residualize since residuals are easier to work with =#
  bDAE = Causalize.residualizeEveryEquation(bDAE)
  @debug(BDAEUtil.stringHeading1(bDAE, "residuals"));
  return bDAE
end


"""
  Transforms given FlatModelica to backend DAE-IR (BDAE-IR).
"""
function lower(fm::OMFrontend.Frontend.FLAT_MODEL)
  local preprocessedFM = FrontendUtil.handleBuiltin(fm)
  local bDAE = BDAECreate.lower(preprocessedFM)
  @BACKEND_LOGGING debugWrite("initialBDAE.log", BDAEUtil.stringHeading1(bDAE, "initial BDAE"))
  #= Dump the actual DAE.Exp structure to a file for analysis =#
  @BACKEND_LOGGING begin
    open("bdae_structure_dump.log", "w") do io
      println(io, "=== BDAE Expression Structure Dump ===")
      for (i, eq) in enumerate(first(bDAE.eqs).orderedEqs)
        if i <= 20  # Dump first 20 equations
          println(io, "\n--- Equation $i: $(typeof(eq)) ---")
          dump(io, eq; maxdepth=10)
        end
      end
      println(io, "\n=== First 10 Variables ===")
      for (i, v) in enumerate(first(bDAE.eqs).orderedVars)
        if i <= 10
          println(io, "\n--- Variable $i ---")
          dump(io, v; maxdepth=8)
        end
      end
    end
  end
  #= Transform ASUB expressions: der(array)[i] to der(array[i]) =#
  bDAE = Causalize.transformASUBExpressions(bDAE)
  #= Mark state variables =#
  bDAE = Causalize.detectStates(bDAE)
  @BACKEND_LOGGING debugWrite("afterDetectStates.log", BDAEUtil.stringHeading1(bDAE, "after detect states"))
    
  @BACKEND_LOGGING debugWrite("afterASUBTransform.log", BDAEUtil.stringHeading1(bDAE, "after ASUB transformation"))
  #= Flatten CREF_QUAL with subscripted array finals to match hash table entries =#
  bDAE = Causalize.flattenArrayCrefs(bDAE)
  @BACKEND_LOGGING debugWrite("afterFlattenCrefs.log", BDAEUtil.stringHeading1(bDAE, "after CREF flattening"))
  #= Transform if expressions to if equations =#
  bDAE = Causalize.detectIfExpressions(bDAE)
  @BACKEND_LOGGING debugWrite("afterIfExpressions.log", BDAEUtil.stringHeading1(bDAE, "after if expressions"))
  #= Expand COMPLEX_EQUATIONs into scalar equations =#
  bDAE = Causalize.expandComplexEquations(bDAE)
  @BACKEND_LOGGING debugWrite("afterExpandComplex.log", BDAEUtil.stringHeading1(bDAE, "after complex equation expansion"))
  bDAE = Causalize.residualizeEveryEquation(bDAE)
  @BACKEND_LOGGING debugWrite("afterResidualize.log", BDAEUtil.stringHeading1(bDAE, "after residualize"))
  #=
    Remove unused parameters and or constants.
    Important optimization for some systems.
    TODO: also check bindings of all parameters before readding this call.
  =#
  #bDAE = Causalize.detectUnusedParametersAndConstants(bDAE)
  #= Find and reclassify discrete variables not marked as discrete. =#
  @BACKEND_LOGGING debugWrite("residualTransformation.log", BDAEUtil.stringHeading1(bDAE, "residuals"))
  return bDAE
end

"""
  Transforms  BDAE-IR to simulation code for DAE-mode
"""
function generateSimulationCode(bDAE::BDAE.BACKEND_DAE; mode)::SimulationCode.SimCode
  local simCode = SimulationCode.transformToSimCode(bDAE; mode = mode)
  @debug BDAEUtil.stringHeading1(simCode, "SIM_CODE: transformed simcode")
  return simCode
end

"""
  Translates functions to simlulation code.
"""
function generateSimCodeFunctions(functions::List{OMFrontend.Frontend.M_FUNCTION})
  local simCodeFunctions = SimulationCode.generateSimCodeFunctions(functions)
  return simCodeFunctions
end

"""
  Generates code interfacing DifferentialEquations.jl
  The resulting code is saved in a dictonary which contains functions that where simulated
  this session. Returns the generated modelName and corresponding generated code
"""
function generateTargetCode(simCode::SimulationCode.SIM_CODE)
  #= Target code =#
  (modelName::String, modelCode::Expr) = CodeGeneration.generateCode(simCode)
  @debug "Functions:" modelCode
  @debug "Model:" modelName
  #= TODO: This replacement should ideally be done earlier. Or be solved in a nicer way. =#
  modelName = replace(modelName, "." => "__")
  COMPILED_MODELS[modelName] = modelCode
  return (modelName, modelCode)
end

"""
`generateMTKTargetCode(simCode::SimulationCode.SIM_CODE)`
  Generates code interfacing ModelingToolkit.jl
  The resulting code is saved in a table which contains functions that where simulated
  this session. Returns the generated modelName and corresponding generated code
"""
function generateMTKTargetCode(simCode::SimulationCode.SIM_CODE)
  #= Target code =#
  (modelName::String, modelCode::Expr) = CodeGeneration.generateMTKCode(simCode)
  @debug "Functions:" modelCode
  @debug "Model:" modelName
  modelName = replace(modelName, "." => "__")
  if haskey(COMPILED_MODELS_MTK, modelName)
    #= Check if the newly generated model is different from the previous model =#
    local changeDetected = COMPILED_MODELS_MTK[modelName] == modelCode
    COMPILED_MODELS_MTK[modelName] = (modelCode, changeDetected)
  else
    COMPILED_MODELS_MTK[modelName] = (modelCode, false)
  end
  return (modelName, modelCode)
end

function getCompiledModel(modelName)
  try
    return first(COMPILED_MODELS_MTK[modelName])
  catch e
    @error "Model: $(modelName) is not compiled. Available models are: $(availableModels())"
    throw(e)
  end
end

"""
  Returns true if the model was compiled again
"""
function modelWasCompiledAgain(modelName)
  return last(COMPILED_MODELS_MTK[modelName])
end

"""

```
writeModelToFile(modelName::String, filePath::String; keepComments = true, keepBeginBlocks = true)
```
  Writes a model to file by default the file is formatted and comments are kept.
"""
function writeModelToFile(modelName::String, filePath::String; keepComments = true, keepBeginBlocks = true)
  model = getCompiledModel(modelName)
  try
    mAsStr = modelToString(modelName; MTK = true,
                           keepComments = keepComments,
                           keepBeginBlocks = keepBeginBlocks)
    try
      #= Replace top level begin/end =#
      beginIdx = last(findfirst("begin", mAsStr)) + 1
      endIdx = first(findlast("end",  mAsStr)) - 1
      mAsStr = mAsStr[beginIdx:endIdx]
      writeStringToFile(filePath, mAsStr)
    catch e
      @error "Error removing initial begin/end pairs" exception=(e, catch_backtrace())
    end
  catch e
    @error "Failed writing $model to file: $filePath" exception=(e, catch_backtrace())
  end
end

"""
    Write the contents of a string to file.
"""
function writeStringToFile(fileName::String, contents::String)
  local fdesc = open(fileName, "w")
  write(fdesc, contents)
  close(fdesc)
end

"""
  Prints a model.
  If the specified model exists. Print it to stdout.
"""
function printModel(modelName::String; MTK = true, keepComments = true, keepBeginBlocks = true)
  try
    println(modelToString(modelName::String; MTK = MTK, keepComments = keepComments, keepBeginBlocks = keepBeginBlocks))
  catch e
    @error "Model: $(modelName) is not compiled. Available models are: $(availableModels())"
    throw("Error printing model")
  end
end

"""
 Converts a given backend model to a string
"""
function modelToString(modelName::String; MTK = true, keepComments = true, keepBeginBlocks = true)
  try
    local model::Expr
    model = getCompiledModel(modelName)
    strippedModel = "$model"
    #= Remove all the redudant blocks from the model =#
    if keepComments == false
      strippedModel = CodeGeneration.stripComments(model)
    end
    if keepBeginBlocks == false
      strippedModel = CodeGeneration.stripBeginBlocks(model)
    end
    local modelStr::String = "$strippedModel"
    local formattedResults
    try
      formattedResults = JuliaFormatter.format_text(modelStr;
                                                    remove_extra_newlines = true,
                                                    indent = 4,
                                                    margin = 200,
                                                    always_use_return = true)
    catch e
      @warn "Julia Formatter failed to format the output results due to $(e)"
      formattedResults = modelStr
    end
    return formattedResults
  catch e
    @error "Model: $(modelName) is not compiled.\n Available models are: $(availableModels())" exception=(e, catch_backtrace())
  end
end


"""
    Prints available compiled models to stdout
"""
function availableModels()::String
  str = "Compiled models (MTK-MODE):\n"
    for m in keys(COMPILED_MODELS_MTK)
      str *= "  $m\n"
    end
  return str
end

"""
  ```
   simulateModel(modelName::String;
                       MODE = MTK_MODE,
                       tspan = (0.0, 1.0),
                       solver = Rodas5(),
                       kwargs...)
  ```
  Simulates model interactivly.
  The solver need to be passed with a : before the name, example:
  OMBackend.simulateModel(modelName, tspan = (0.0, 1.0), solver = :(Tsit5()));
"""
function simulateModel(modelName::String;
                       MODE = MTK_MODE,
                       tspan = (0.0, 1.0),
                       solver = Rodas5(),
                       kwargs...)
  #= Strings using "." need to be in a format suitable for Julia =#
  modelName = replace(modelName, "." => "__")
  local modelCode::Expr
  local strippedModel::Expr
  if MODE == MTK_MODE
    #= This does a redundant string conversion for now due to modeling toolkit being as is...=#
    try
      modelCode = getCompiledModel(modelName)
    catch err
      println("Failed to simulate model.")
      println("Available models are:")
      availableModels()
    end
    try
      @eval $(:(import OMBackend))
      #= Added to adjust the recent DEFAULT_PRECS issue =#
      @eval $(:(import OrdinaryDiffEq.DEFAULT_PRECS))
      @eval $(:(using DifferentialEquations))
      #= Below is needed to pass the custom solver=#
      strippedModel = CodeGeneration.stripBeginBlocks(modelCode)
      @eval $strippedModel
      #=
      Evaluate the model runnable.
      Is used to specify the keyword arguments that are being passed along.
      =#
      local modelRunnable =  Expr(:call,
                                  Symbol(modelName, "Simulate"),
                                  Expr(:parameters,
                                       Expr(:(...), kwargs)),
                                  tspan,
                                  solver)

      #= Run the model =#
      @eval $modelRunnable
      #=
      The model is now compiled and a part of the OMBackend module.
      In the following path OMBackend.<modelName>Simulate
      =#
    catch err
      @error "Interactive evaluation failed" exception_type=typeof(err) mode=MODE model=modelName
      rethrow(err)
    end
  else
    throw("Unsupported mode")
  end
end

"""
  Resimulates an already compiled model given a model that is already active in th environment
  along with a set of parameters as key value pairs.
"""
function resimulateModel(modelName::String;
                         solver = Rodas5(),
                         MODE = MTK_MODE,
                         tspan=(0.0, 1.0),
                         parameters::Dict = Dict())
  #=
  Check if a compiled instance of the model already exists in the backend.
  If that is the case we do not have to recompile it.
  =#
  try
    local modelRunnable = Meta.parse("OMBackend.$(modelName)Simulate($(tspan); solver = $(solver))")
    res = @eval $modelRunnable
    return res
  catch #= The model is not compiled yet. Proceeding.. =#
    availModels = availableModels()
    @error "The model $(modelName) is not compiled.\n Available models are: $(availModels)"
  end
end

"
`plot(sol::Runtime.OMSolution)`
  The default plot function of OMBackend.
  All labels of the variables and the name is given by default
"
function plot(sol::Runtime.OMSolution)
  local nsolution = sol.diffEqSol
  local t = nsolution.t
  local rescols = collect(eachcol(transpose(hcat(nsolution.u...))))
  labels = permutedims(sol.idxToName.vals)
  Plots.plot(t, rescols; labels=labels)
end

"""
`function plot(sol)`
  An alternative plot function in OMBackend.
  All labels of the variables and the name is given by default
"""
function plot(sol)
  Plots.plot(sol)
end

"""
  Plot for an OMSolution that contains several sub solutions.
  Plots all part of the solution on the same graph.
"""
function plot(sol::Runtime.OMSolutions; legend = false, limX = 0.0, limY = 1.0)
  local sols = sol.diffEqSol
  local prevP = Plots.plot!(sols[1]; legend = legend, xlim=limX, ylim = limY)
  for sol in sols[2:end]
    p = Plots.plot!(prevP; legend = legend, xlim=limX, ylim = limY)
    prevP = p
  end
  return prevP
end

"""
  Plots a vector of solutions
"""
function plot(sol::Vector; legend = false, limX = 0.0, limY = 1.0, kwargs...)
  sols = sol
  local prevP = Plots.plot!(sols[1]; legend = legend, xlim=limX, ylim = limY, kwargs...)
  for sol in sols[2:end]
    p = Plots.plot!(prevP; legend = legend, xlim=limX, ylim = limY, kwargs...)
    prevP = p
  end
  return prevP
end

"""
  Returns the value of a variable given a solution and a string.
```
  getVariableValues(sol, varName::String)
```
Example use:
```julia
OM.translate("HelloWorld", "./Models/HelloWorld.mo");
sol = OM.simulate("HelloWorld");

OM.OMBackend.getVariableValues(sol, "x")
11-element Vector{Float64}:
 1.0
 0.7689684977240044
 0.555181390244627
 0.371869514552751
 0.23297802339017032
 0.13657983532457932
 0.07542877379860594
 0.039431995308782476
 0.019632381398183626
 0.009355999959425394
 0.006738051637508934
```
"""
function getVariableValues(sol::ODESolution, varName::String)

  varAsJLSym = if varName != "time"
    Symbol(replace(varName, "." => "__"))
  else
    :t
  end
  try
    res = sol[varAsJLSym]
  catch
    println("Did not locate a variable named " * varName * " in the model")
  end
end

"""
```
getVariableValues(sols::Vector, variables...)
```

Similar to getVariableValues, but for solutions that have went through one or more structural changes.
Here the same variable might have slightly different names depending on the context.

For instance it might be called M.A first and then M.B when the structure of the model is changed.
In the example above a pendulum goes through once such change.
Here you can specify the variables in order, they will be collected and merged into the final vector.

Example use:
```
julia> OM.OMBackend.getVariableValues(sols, "bouncingBall_x", "pendulum_x")
vcat(OM.OMBackend.getVariableValues(sols, "pendulum_y", "bouncingBall_y")...)
69-element Vector{Float64}:
  10.0
   ⋮
   8.71359663258124
   ⋮
 -11.346053750176466
     ⋮
  3.3856338791378615
```
"""
function getVariableValues(sols::Vector, variables...)
  local vals = Any[]
  for sol in sols
    for v in variables
      local vAsJLSym = if v != "time"
        Symbol(replace(v, "." => "__"))
      else
        :t
      end
      try
        push!(vals, sol[vAsJLSym])
      catch
      end
    end
  end
  return vcat(vals...)
end

"""
Wrapper function to MTK `observed`
"""
function MTK_getObserved(sol)
  ModelingToolkit.observed(sol.prob.f.sys)
end
