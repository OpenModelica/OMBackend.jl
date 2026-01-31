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

#=
  The code in this file is used to convert frontend functions to a definition that can be used by the backend, and the code generators there.
=#

const FRONTEND_FUNCTION = OMFrontend.Frontend.M_FUNCTION

"""
  Generates algorithmic simcode
TODO:
Handle concrete non variable arguments.
"""
function generateSimCodeFunctions(functionList::List{FRONTEND_FUNCTION})::Tuple{Vector{ModelicaFunction}, Bool}
  local functions = ModelicaFunction[]
  local externalFunctionsUsed = false
  for f in functionList
    local n = string(f.path)
    local inputs = map(f.inputs) do input
      OMFrontend.Frontend.convertFunctionParam(input)
    end
    local outputs = map(f.outputs) do output
      OMFrontend.Frontend.convertFunctionParam(output)
    end
    local locals = map(f.locals) do l
      OMFrontend.Frontend.convertFunctionParam(l)
    end
    if ! OMFrontend.Frontend.isExternal(f)
      local body::Vector{OMFrontend.Frontend.Statement} = OMFrontend.Frontend.getBody(f)
      local stmts = OMFrontend.Frontend.convertStatements(body)
      #= Remove smooth calls from statements =#
      stmts = FrontendUtil.removeSmoothFromStatements(collect(stmts))
      local mf = MODELICA_FUNCTION(n, inputs, outputs, locals, listArray(MetaModelica.list(stmts...)))
      push!(functions, mf)
    else #= The function is a wrapper for some internal builtin Modelica Function =#
      externalFunctionsUsed = true
      s = OMFrontend.Frontend.IOStream_M.create(getInstanceName(), OMFrontend.Frontend.IOStream_M.LIST())
      s = OMFrontend.Frontend.toFlatStream(OMFrontend.Frontend.getSections(f.node), f.path, s)#"dummy"
      str = OMFrontend.Frontend.IOStream_M.string(s)
      #=This should really really not be done by string splitting magic... =#
      local libInfo = first(split(str, "annotation"))
      libInfo = replace(libInfo, "external \"C\"" => "")
      libInfo = replace(libInfo, "'" => "")
      push!(functions, EXTERNAL_MODELICA_FUNCTION(n, inputs, outputs, libInfo))
    end
  end
  return (functions, externalFunctionsUsed)
end

"""
  Transforms SimCode functions by flattening record inputs/outputs.
  This makes record fields explicit as separate parameters.
"""
function flattenRecordParameters(functions::Vector{ModelicaFunction})::Vector{ModelicaFunction}
  return map(flattenRecordParametersInFunction, functions)
end

"""
  Flatten record parameters in a single function.
"""
function flattenRecordParametersInFunction(func::MODELICA_FUNCTION)::MODELICA_FUNCTION
  local flattenedInputs = DAE.VAR[]
  local flattenedOutputs = DAE.VAR[]
  local recordFieldMap = Dict{String, Vector{Tuple{String, DAE.Type}}}()  # Maps record name to (fieldName, fieldType) pairs

  #= Flatten inputs =#
  for input in func.inputs
    flattenedVars = flattenRecordVar(input, recordFieldMap)
    append!(flattenedInputs, flattenedVars)
  end

  #= Flatten outputs =#
  for output in func.outputs
    flattenedVars = flattenRecordVar(output, recordFieldMap)
    append!(flattenedOutputs, flattenedVars)
  end

  #= Transform statements to use flattened names =#
  local transformedStatements = transformStatementsForFlattenedRecords(func.statements, recordFieldMap)

  return MODELICA_FUNCTION(func.name, flattenedInputs, flattenedOutputs, func.locals, transformedStatements)
end

function flattenRecordParametersInFunction(func::EXTERNAL_MODELICA_FUNCTION)::EXTERNAL_MODELICA_FUNCTION
  #= External functions are not transformed for now =#
  return func
end

"""
  Flatten a single variable. If it's a record type, returns multiple variables for each field.
  Otherwise returns the original variable in a vector.
"""
function flattenRecordVar(v::DAE.VAR, recordFieldMap::Dict{String, Vector{Tuple{String, DAE.Type}}})::Vector{DAE.VAR}
  local baseName = string(v.componentRef)
  @match v.ty begin
    DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _) => begin
      local flattenedVars = DAE.VAR[]
      local fieldInfo = Tuple{String, DAE.Type}[]
      for field in varLst
        @match field begin
          DAE.TYPES_VAR(fieldName, _, fieldTy, _, _) => begin
            local flatName = baseName * "_" * fieldName
            #= Extract dims from fieldTy if it is an array type =#
            local fieldDims = @match fieldTy begin
              DAE.T_ARRAY(_, dims) => dims
              _ => MetaModelica.nil
            end
            #= Create a new DAE.VAR with flattened name and field type =#
            local flatCref = DAE.CREF_IDENT(flatName, fieldTy, MetaModelica.nil)
            local flatVar = DAE.VAR(
              flatCref,
              v.kind,
              v.direction,
              v.parallelism,
              v.protection,
              fieldTy,
              NONE(),  #= No binding for flattened fields =#
              fieldDims,
              v.connectorType,
              v.source,
              NONE(),
              NONE(),
              v.innerOuter
            )
            push!(flattenedVars, flatVar)
            push!(fieldInfo, (fieldName, fieldTy))
          end
          _ => nothing
        end
      end
      recordFieldMap[baseName] = fieldInfo
      return flattenedVars
    end
    _ => return [v]
  end
end

"""
  Transform statements to replace record field accesses with flattened names.
  E.g., R.T[1,2] becomes R_T[1,2], and R.w becomes R_w
"""
function transformStatementsForFlattenedRecords(statements::Vector{DAE.Statement}, recordFieldMap::Dict)::Vector{DAE.Statement}
  return map(statements) do stmt
    transformStatementForFlattenedRecords(stmt, recordFieldMap)
  end
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_ASSIGN, recordFieldMap::Dict)::DAE.Statement
  local newExp1 = transformExpForFlattenedRecords(stmt.exp1, recordFieldMap)
  local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
  return DAE.STMT_ASSIGN(stmt.type_, newExp1, newExp, stmt.source)
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_ASSIGN_ARR, recordFieldMap::Dict)::DAE.Statement
  local newLhs = transformExpForFlattenedRecords(stmt.lhs, recordFieldMap)
  local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
  return DAE.STMT_ASSIGN_ARR(stmt.type_, newLhs, newExp, stmt.source)
end

function transformStatementForFlattenedRecords(stmt::DAE.Statement, recordFieldMap::Dict)::DAE.Statement
  #= For other statement types, return unchanged for now =#
  return stmt
end

"""
  Transform expressions to replace record field accesses with flattened names.
"""
function transformExpForFlattenedRecords(exp::DAE.Exp, recordFieldMap::Dict)::DAE.Exp
  @match exp begin
    #= Handle qualified CREF like R.T or R.w =#
    DAE.CREF(DAE.CREF_QUAL(ident, identType, subscriptLst, componentRef), ty) => begin
      #= Check if the base ident is a flattened record =#
      if haskey(recordFieldMap, ident)
        #= Get the inner field name =#
        local innerName = string(componentRef)
        local flatName = ident * "_" * innerName
        #= Preserve subscripts from the inner cref =#
        local innerSubscripts = @match componentRef begin
          DAE.CREF_IDENT(_, _, subs) => subs
          _ => MetaModelica.nil
        end
        local flatCref = DAE.CREF_IDENT(flatName, ty, innerSubscripts)
        return DAE.CREF(flatCref, ty)
      end
      return exp
    end
    #= Recursively transform binary expressions =#
    DAE.BINARY(e1, op, e2) => begin
      DAE.BINARY(transformExpForFlattenedRecords(e1, recordFieldMap), op,
                 transformExpForFlattenedRecords(e2, recordFieldMap))
    end
    #= Recursively transform arrays =#
    DAE.ARRAY(ty, scalar, arr) => begin
      local newArr = map(arr) do e
        transformExpForFlattenedRecords(e, recordFieldMap)
      end
      DAE.ARRAY(ty, scalar, MetaModelica.list(newArr...))
    end
    #= Recursively transform unary expressions =#
    DAE.UNARY(op, e1) => begin
      DAE.UNARY(op, transformExpForFlattenedRecords(e1, recordFieldMap))
    end
    #= Recursively transform function calls =#
    DAE.CALL(path, expLst, attr) => begin
      local newExpLst = map(expLst) do e
        transformExpForFlattenedRecords(e, recordFieldMap)
      end
      DAE.CALL(path, MetaModelica.list(newExpLst...), attr)
    end
    _ => exp
  end
end
