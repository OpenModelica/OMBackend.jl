#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
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

using DataStructures
"""
  File: simCodeDump.jl
  Dumping functions for simulation code structures.
"""
function dumpSimCode(simCode::SimulationCode.SIM_CODE, heading::String = simCode.name)
  local buffer = IOBuffer()
  println(buffer, BDAEUtil.DOUBLE_LINE)
  println(buffer, "SIMULATION CODE: ", heading)
  println(buffer, BDAEUtil.DOUBLE_LINE)
  println(buffer)

  #= Classify variables =#
  local algVariables = String[]
  local arrayVariables = Tuple{String, Int}[]
  local arrayParameters = Tuple{String, Int}[]
  local discreteVariables = String[]
  local dsVariables = String[]
  local occVariables = String[]
  local parameters = String[]
  local stateVariables = String[]
  for varName in keys(simCode.stringToSimVarHT)
    (_, var) = simCode.stringToSimVarHT[varName]
    local varType = var.varKind
    @match varType begin
      SimulationCode.INPUT(__) => @error "INPUT not supported in CodeGen"
      SimulationCode.STATE(__) => push!(stateVariables, varName)
      SimulationCode.STATE_DERIVATIVE(__) => nothing
      SimulationCode.PARAMETER(__) => push!(parameters, varName)
      SimulationCode.STRING(__) => push!(parameters, varName)
      SimulationCode.ALG_VARIABLE(__) => push!(algVariables, varName)
      SimulationCode.ARRAY_PARAMETER(__) => push!(arrayParameters, (varName, sum(varType.dimensions)))
      SimulationCode.ARRAY(__) => push!(arrayVariables, (varName, sum(varType.dimensions)))
      SimulationCode.DISCRETE(__) => push!(discreteVariables, varName)
      SimulationCode.OCC_VARIABLE(__) => push!(occVariables, varName)
      SimulationCode.DATA_STRUCTURE(__) => push!(dsVariables, varName)
    end
  end
  local algAndState = vcat(algVariables, stateVariables)

  #= Functions =#
  if !isempty(simCode.functions)
    println(buffer, "Modelica Functions:")
    println(buffer, BDAEUtil.LINE)
    for f in simCode.functions
      println(buffer, string(f))
    end
    println(buffer, BDAEUtil.LINE)
  end

  #= Variables =#
  println(buffer, "Variables:")
  println(buffer, BDAEUtil.LINE)

  dumpVarSection(buffer, "State Variables", stateVariables, simCode)
  dumpVarSection(buffer, "Parameters & Constants", parameters, simCode)
  if !isempty(arrayParameters)
    dumpVarSection(buffer, "Array Parameters", map(first, arrayParameters), simCode)
  end
  dumpVarSection(buffer, "Algebraic Variables", algVariables, simCode)
  if !isempty(arrayVariables)
    dumpVarSection(buffer, "Array Variables", map(first, arrayVariables), simCode)
  end
  dumpVarSection(buffer, "Discrete Variables", discreteVariables, simCode)
  dumpVarSection(buffer, "OCC Variables", occVariables, simCode)
  dumpVarSection(buffer, "Data Structure Variables", dsVariables, simCode)
  println(buffer, BDAEUtil.LINE)

  #= Initial equations =#
  if !isempty(simCode.initialEquations)
    println(buffer, "Initial Equations:")
    println(buffer, BDAEUtil.LINE)
    for ieq in simCode.initialEquations
      print(buffer, string(ieq))
    end
    println(buffer, BDAEUtil.LINE)
  end

  #= Residual equations =#
  println(buffer, "Residual Equations:")
  println(buffer, BDAEUtil.LINE)
  for (i, eq) in enumerate(simCode.residualEquations)
    println(buffer, "  [", i, "] ", BDAE.string(eq))
  end
  println(buffer, BDAEUtil.LINE)

  #= If-equations =#
  if !isempty(simCode.ifEquations)
    println(buffer, "If-Equations:")
    println(buffer, BDAEUtil.LINE)
    for ifEq in simCode.ifEquations
      print(buffer, string(ifEq))
    end
    println(buffer, BDAEUtil.LINE)
  end

  #= When-equations =#
  if !isempty(simCode.whenEquations)
    println(buffer, "When-Equations:")
    println(buffer, BDAEUtil.LINE)
    for wEq in simCode.whenEquations
      print(buffer, string(wEq))
    end
    println(buffer, BDAEUtil.LINE)
  end

  #= Structural transitions =#
  if !isempty(simCode.structuralTransitions)
    println(buffer, "Structural Transitions:")
    println(buffer, BDAEUtil.LINE)
    for st in simCode.structuralTransitions
      print(buffer, string(st))
    end
    println(buffer, BDAEUtil.LINE)
  end

  #= Shared variables and equations =#
  if !isempty(simCode.sharedEquations)
    println(buffer, "Shared Variables:")
    println(buffer, BDAEUtil.LINE)
    for sv in simCode.sharedVariables
      println(buffer, "  ", string(sv))
    end
    println(buffer, BDAEUtil.LINE)
    println(buffer, "Shared Equations:")
    println(buffer, BDAEUtil.LINE)
    for se in simCode.sharedEquations
      print(buffer, string(se))
    end
    println(buffer, BDAEUtil.LINE)
  end

  #= Statistics =#
  println(buffer)
  println(buffer, "Statistics:")
  println(buffer, BDAEUtil.LINE)
  nArrayElems = isempty(arrayVariables) ? 0 : sum(map(last, arrayVariables))
  nIfEqs = sum(length(first(ifEq.branches).residualEquations) for ifEq in simCode.ifEquations; init=0)
  nWhenEqs = sum(length(wEq.whenEquation.whenStmtLst) for wEq in simCode.whenEquations; init=0)
  nTotalVars = length(algAndState) + length(discreteVariables) + length(occVariables) + nArrayElems
  nTotalEqs = length(simCode.residualEquations) + nIfEqs + nWhenEqs
  println(buffer, "  Variables:  ", nTotalVars, " total")
  println(buffer, "    State:      ", length(stateVariables))
  println(buffer, "    Algebraic:  ", length(algVariables))
  println(buffer, "    Array:      ", nArrayElems)
  println(buffer, "    Discrete:   ", length(discreteVariables))
  println(buffer, "    OCC:        ", length(occVariables))
  println(buffer, "  Equations:  ", nTotalEqs, " total")
  println(buffer, "    Residual:   ", length(simCode.residualEquations))
  println(buffer, "    If:         ", nIfEqs)
  println(buffer, "    When:       ", nWhenEqs)
  println(buffer, "  Functions:  ", length(simCode.functions))
  println(buffer, BDAEUtil.LINE)

  println(buffer, BDAEUtil.DOUBLE_LINE)
  println(buffer, "END SIM_CODE")
  println(buffer, BDAEUtil.DOUBLE_LINE)

  if !isempty(simCode.subModels)
    for (i, sm) in enumerate(simCode.subModels)
      println(buffer)
      println(buffer, dumpSimCode(sm, "Structural-Sub-model #" * string(i)))
    end
  end

  return String(take!(buffer))
end

"""
  Helper to dump a section of variables. Skips the section entirely if empty.
"""
function dumpVarSection(buffer::IOBuffer, sectionName::String, varNames::Vector{String}, simCode)
  isempty(varNames) && return
  println(buffer, "  ", sectionName, ":")
  for name in varNames
    (idx, var) = simCode.stringToSimVarHT[name]
    println(buffer, "    [", idx, "] ", name, dumpSimVarKind(var.varKind))
  end
end

"""
  Pretty-print a SimVarType, showing dimensions and binding expressions readably.
"""
function dumpSimVarKind(vk::SimVarType)::String
  @match vk begin
    SimulationCode.STATE(__) => ""
    SimulationCode.ALG_VARIABLE(__) => ""
    SimulationCode.DISCRETE(__) => ""
    SimulationCode.OCC_VARIABLE(__) => ""
    SimulationCode.STRING(__) => " :: String"
    SimulationCode.PARAMETER(bindExp) => begin
      binding = dumpBindExp(bindExp)
      isempty(binding) ? " (parameter)" : " = " * binding
    end
    SimulationCode.ARRAY(dims, bindExp) => begin
      dimStr = "[" * join(string.(dims), ", ") * "]"
      binding = dumpBindExp(bindExp)
      isempty(binding) ? " :: Real" * dimStr : " :: Real" * dimStr * " = " * binding
    end
    SimulationCode.ARRAY_PARAMETER(dims, bindExp) => begin
      dimStr = "[" * join(string.(dims), ", ") * "]"
      binding = dumpBindExp(bindExp)
      isempty(binding) ? " (parameter) :: Real" * dimStr : " (parameter) :: Real" * dimStr * " = " * binding
    end
    SimulationCode.DATA_STRUCTURE(bindExp) => begin
      binding = dumpBindExp(bindExp)
      isempty(binding) ? " (data structure)" : " (data structure) = " * binding
    end
    _ => ""
  end
end

"""
  Pretty-print an optional binding expression using BDAE.string for the DAE.Exp.
"""
function dumpBindExp(bindExp::Option{Exp})::String
  @match bindExp begin
    SOME(exp) => BDAE.string(toDAEExp(exp))
    NONE() => ""
  end
end

function string(ht::OrderedDict{T1, Tuple{T2, SimVar}}) where {T1, T2}
  ks = keys(ht)
  res = ""
  for k in ks
    res *= "Name:" * k * "|Index:" * string(first(ht[k])) * "|Attributes:{" * string(last(ht[k])) * "}|\n"
  end
  return res
end


"""
 Converts a ```backendVar::BDAE.Var``` to the simcode format.
"""
function string(backendVar::BDAE.Var)
  BDAE.string(backendVar.varName; separator = OMBackend.COMPONENT_SEPARATOR)
end


"""
  This function just forwards the call to BDAE
"""
function string(element)
  BDAE.string(element)
end

function string(v::SIMVAR)
  return v.name  * "," * string(v.index) * ","  * string(v.varKind)
end

function string(ifEq::IF_EQUATION)
  res = ""
  for branch in ifEq.branches
    res *= "IF (" * string(branch.condition) * ")\n"
    for eq in branch.residualEquations
      res *= string(eq)
    end
    res *= "END\n"
  end
  return res
end

function Base.string(ieq::SimulationCode.DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION)
  "STRUCTURAL_DOCC_IF_EQUATION: " * string(ieq.ifEquation) * "\n"
end

function string(st::IMPLICIT_STRUCTURAL_TRANSISTION)
  "STRUCTURAL_WHEN_EQUATION:\n" * string(st.whenEquation)
end

function Base.string(simStructChange::SimulationCode.EXPLICIT_STRUCTURAL_TRANSISTION)
  return "STRUCTURAL_TRANSITION: FROM: <" * simStructChange.fromState *
         "> TO: <" * simStructChange.toState *
         "> WHEN: " * string(simStructChange.transistionCondition) * "\n"
end

function string(f::EXTERNAL_MODELICA_FUNCTION)
  local buffer = IOBuffer()
  println(buffer, "function EXTERNAL " * f.name)
  for arg in f.inputs
    println(buffer, " " * string(arg))
  end
  for arg in f.outputs
    println(buffer, " " * string(arg))
  end
  println(buffer, "calling externally defined function: " * f.libInfo)
  println(buffer, "end " * f.name)
  return String(take!(buffer))
end

function string(f::MODELICA_FUNCTION)
  local buffer = IOBuffer()
  println(buffer, "function " * f.name)
  for arg in f.inputs
    println(buffer, "  input " * safeDumpString(arg) * " :: " * safeDumpDAEVarType(arg))
  end
  for arg in f.outputs
    println(buffer, "  output " * safeDumpString(arg) * " :: " * safeDumpDAEVarType(arg))
  end
  for l in f.locals
    println(buffer, "  local " * safeDumpString(l) * " :: " * safeDumpDAEVarType(l))
  end
  println(buffer, "algorithm")
  #= Per-statement TYPE only, no recursive body dump. DAE.jl's string(::Statement)
     recurses through nested STMT_IF/STMT_WHILE bodies; for big functions
     (Modelica.Utilities.Strings.scanToken family) that blows past Julia's
     runtime stack guard. The type list keeps the structure visible for
     debugging without firing the recursion. =#
  for s in f.statements
    println(buffer, "  ", string(typeof(s).name.name))
  end
  println(buffer, "end " * f.name)
  return String(take!(buffer))
end

function safeDumpString(x)::String
  try
    local s = string(x)
    return s === nothing ? "<nothing>" : string(s)
  catch err
    return "<dump error: " * string(typeof(err)) * ">"
  end
end

function safeDumpDAEVarType(v::DAE.VAR)::String
  try
    local s = dumpDAEVarType(v)
    return s === nothing ? "<unknown type>" : s
  catch err
    return "<type dump error: " * string(typeof(err)) * ">"
  end
end

"""
  Dumps a DAE.VAR's type, taking into account both ty and dims fields.
  For arrays, dimensions may be in either the ty field (T_ARRAY) or the dims field.
"""
function dumpDAEVarType(v::DAE.VAR)::String
  local baseType = dumpDAEType(v.ty)
  #= Only append dims if the type itself is not already an array type,
     otherwise we double-print the dimensions =#
  @match v.ty begin
    DAE.T_ARRAY(__) => return baseType
    _ => nothing
  end
  #= Check if dims field contains array dimensions (MetaModelica list) =#
  local hasDims = false
  local dimStrs = String[]
  try
    for dim in v.dims
      hasDims = true
      @match dim begin
        DAE.DIM_INTEGER(n) => push!(dimStrs, string(n))
        DAE.DIM_UNKNOWN(__) => push!(dimStrs, ":")
        _ => push!(dimStrs, "?")
      end
    end
  catch
    #= Empty list or iteration error =#
  end
  if hasDims
    return baseType * "[" * join(dimStrs, ", ") * "]"
  end
  return baseType
end

"""
  Dumps a DAE.Type to a human-readable string.
  Includes array dimensions and record field information.
"""
function dumpDAEType(ty::DAE.Type)::String
  @match ty begin
    DAE.T_REAL(__) => "Real"
    DAE.T_INTEGER(__) => "Integer"
    DAE.T_BOOL(__) => "Bool"
    DAE.T_STRING(__) => "String"
    DAE.T_ARRAY(elemTy, dims) => begin
      local dimStrs = String[]
      for dim in dims
        @match dim begin
          DAE.DIM_INTEGER(n) => push!(dimStrs, string(n))
          DAE.DIM_UNKNOWN(__) => push!(dimStrs, ":")
          _ => push!(dimStrs, "?")
        end
      end
      dumpDAEType(elemTy) * "[" * join(dimStrs, ", ") * "]"
    end
    DAE.T_COMPLEX(state, varLst, _) => begin
      local stateName = @match state begin
        DAE.ClassInf.RECORD(path) => "record " * string(path)
        _ => "complex"
      end
      local fieldStrs = String[]
      for field in varLst
        @match field begin
          DAE.TYPES_VAR(name, _, fieldTy, _, _) => begin
            push!(fieldStrs, name * "::" * dumpDAEType(fieldTy))
          end
          _ => nothing
        end
      end
      stateName * "{" * join(fieldStrs, ", ") * "}"
    end
    DAE.T_TUPLE(types, _) => begin
      local typeStrs = map(dumpDAEType, types)
      "Tuple{" * join(typeStrs, ", ") * "}"
    end
    DAE.T_UNKNOWN(__) => "Unknown"
    DAE.T_ANYTYPE(__) => "Any"
    _ => "<?" * string(typeof(ty)) * ">"
  end
end
