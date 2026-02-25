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
  local result = DAE.Statement[]
  for stmt in statements
    append!(result, transformStatementForFlattenedRecords(stmt, recordFieldMap))
  end
  return result
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_ASSIGN, recordFieldMap::Dict)::Vector{DAE.Statement}
  #= Check if LHS is a flattened record variable =#
  local lhsName = @match stmt.exp1 begin
    DAE.CREF(DAE.CREF_IDENT(ident, _, _), _) => ident
    _ => nothing
  end
  #= If LHS is a record variable and RHS is a RECORD expression, expand into field assignments =#
  if lhsName !== nothing && haskey(recordFieldMap, lhsName)
    @match stmt.exp begin
      DAE.RECORD(path, exps, fieldNames, ty) => begin
        local fieldInfo = recordFieldMap[lhsName]
        local expVec = collect(exps)
        local stmts = DAE.Statement[]
        for (i, (fieldName, fieldTy)) in enumerate(fieldInfo)
          local flatName = lhsName * "_" * fieldName
          local flatCref = DAE.CREF_IDENT(flatName, fieldTy, MetaModelica.nil)
          local lhsExp = DAE.CREF(flatCref, fieldTy)
          local rhsExp = transformExpForFlattenedRecords(expVec[i], recordFieldMap)
          push!(stmts, DAE.STMT_ASSIGN(fieldTy, lhsExp, rhsExp, stmt.source))
        end
        return stmts
      end
      _ => begin
        local newExp1 = transformExpForFlattenedRecords(stmt.exp1, recordFieldMap)
        local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
        return [DAE.STMT_ASSIGN(stmt.type_, newExp1, newExp, stmt.source)]
      end
    end
  else
    local newExp1 = transformExpForFlattenedRecords(stmt.exp1, recordFieldMap)
    local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
    return [DAE.STMT_ASSIGN(stmt.type_, newExp1, newExp, stmt.source)]
  end
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_ASSIGN_ARR, recordFieldMap::Dict)::Vector{DAE.Statement}
  local newLhs = transformExpForFlattenedRecords(stmt.lhs, recordFieldMap)
  local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
  return [DAE.STMT_ASSIGN_ARR(stmt.type_, newLhs, newExp, stmt.source)]
end

function transformStatementForFlattenedRecords(stmt::DAE.Statement, recordFieldMap::Dict)::Vector{DAE.Statement}
  #= For other statement types, return unchanged for now =#
  return [stmt]
end

"""
  If exp is a plain CREF to a record variable in recordFieldMap, expand it into
  a vector of field CREFs (e.g., R_rel becomes [R_rel_T, R_rel_w]).
  Returns nothing if exp is not an expandable record reference.
"""
function expandRecordArgForCall(exp::DAE.Exp, recordFieldMap::Dict)
  @match exp begin
    DAE.CREF(DAE.CREF_IDENT(ident, _, _), _) => begin
      if haskey(recordFieldMap, ident)
        local fieldInfo = recordFieldMap[ident]
        local fieldExps = DAE.Exp[]
        for (fieldName, fieldTy) in fieldInfo
          local flatName = ident * "_" * fieldName
          local flatCref = DAE.CREF_IDENT(flatName, fieldTy, MetaModelica.nil)
          push!(fieldExps, DAE.CREF(flatCref, fieldTy))
        end
        return fieldExps
      end
      return nothing
    end
    _ => return nothing
  end
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
      local new_e1 = transformExpForFlattenedRecords(e1, recordFieldMap)
      local new_e2 = transformExpForFlattenedRecords(e2, recordFieldMap)
      (new_e1 === e1 && new_e2 === e2) ? exp : DAE.BINARY(new_e1, op, new_e2)
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
      local new_e1 = transformExpForFlattenedRecords(e1, recordFieldMap)
      new_e1 === e1 ? exp : DAE.UNARY(op, new_e1)
    end
    #= Recursively transform function calls, expanding record args into flattened fields =#
    DAE.CALL(path, expLst, attr) => begin
      local newArgs = DAE.Exp[]
      for e in expLst
        local expanded = expandRecordArgForCall(e, recordFieldMap)
        if expanded !== nothing
          append!(newArgs, expanded)
        else
          push!(newArgs, transformExpForFlattenedRecords(e, recordFieldMap))
        end
      end
      DAE.CALL(path, MetaModelica.list(newArgs...), attr)
    end
    _ => exp
  end
end

#= ============================================================================
   Flatten record arguments in equation call sites.
   After flattenRecordParameters has modified function signatures to accept
   individual fields instead of records, this pass rewrites the CALL expressions
   in equations to match. A record CREF argument like R (T_COMPLEX) is replaced
   with individual field arguments: R_T (DAE.ARRAY of element CREFs) and R_w
   (DAE.ARRAY of element CREFs).
   ============================================================================ =#

"""
  Rewrite CALL expressions in all residual equations so that record arguments
  are expanded into individual field arguments matching the flattened function
  signatures.
"""
function flattenRecordCallSites(simCode)
  #= Expand record arguments in residual equations =#
  local newResEqs = map(simCode.residualEquations) do eq
    @match eq begin
      BDAE.RESIDUAL_EQUATION(exp, source, attr) => begin
        local newExp = expandRecordArgsInExp(exp)
        BDAE.RESIDUAL_EQUATION(newExp, source, attr)
      end
      _ => eq
    end
  end
  @assign simCode.residualEquations = newResEqs
  #= Expand record arguments in parameter and array-parameter binding expressions =#
  local ht = simCode.stringToSimVarHT
  for (name, (idx, simVar)) in ht
    local newVarKind = @match simVar.varKind begin
      SimulationCode.PARAMETER(SOME(bindExp)) => begin
        local newBind = expandRecordArgsInExp(bindExp)
        newBind === bindExp ? nothing : SimulationCode.PARAMETER(SOME(newBind))
      end
      SimulationCode.ARRAY_PARAMETER(dims, SOME(bindExp)) => begin
        local newBind = expandRecordArgsInExp(bindExp)
        newBind === bindExp ? nothing : SimulationCode.ARRAY_PARAMETER(dims, SOME(newBind))
      end
      _ => nothing
    end
    if newVarKind !== nothing
      local newSimVar = SimulationCode.SIMVAR(simVar.name, simVar.index, newVarKind, simVar.attributes)
      ht[name] = (idx, newSimVar)
    end
  end
  return simCode
end

"""
  Recursively traverse an expression and expand record arguments inside CALL nodes.
"""
function expandRecordArgsInExp(exp::DAE.Exp)::DAE.Exp
  @match exp begin
    DAE.CALL(path, expLst, attr) => begin
      local newArgs = DAE.Exp[]
      for arg in expLst
        @match arg begin
          DAE.CREF(cr, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _)) => begin
            local baseName = replace(string(cr), "." => "_")
            for field in varLst
              @match field begin
                DAE.TYPES_VAR(fieldName, _, fieldTy, _, _) => begin
                  local flatName = baseName * "_" * fieldName
                  push!(newArgs, buildFieldArgExp(flatName, fieldTy))
                end
                _ => nothing
              end
            end
          end
          _ => push!(newArgs, expandRecordArgsInExp(arg))
        end
      end
      DAE.CALL(path, MetaModelica.list(newArgs...), attr)
    end
    DAE.BINARY(e1, op, e2) => begin
      local new_e1 = expandRecordArgsInExp(e1)
      local new_e2 = expandRecordArgsInExp(e2)
      (new_e1 === e1 && new_e2 === e2) ? exp : DAE.BINARY(new_e1, op, new_e2)
    end
    DAE.UNARY(op, e1) => begin
      local new_e1 = expandRecordArgsInExp(e1)
      new_e1 === e1 ? exp : DAE.UNARY(op, new_e1)
    end
    DAE.ASUB(innerExp, subscripts) => begin
      local newInner = expandRecordArgsInExp(innerExp)
      newInner === innerExp ? exp : DAE.ASUB(newInner, subscripts)
    end
    DAE.IFEXP(cond, e1, e2) => begin
      local newCond = expandRecordArgsInExp(cond)
      local new_e1 = expandRecordArgsInExp(e1)
      local new_e2 = expandRecordArgsInExp(e2)
      (newCond === cond && new_e1 === e1 && new_e2 === e2) ? exp : DAE.IFEXP(newCond, new_e1, new_e2)
    end
    DAE.ARRAY(ty, scalar, arr) => begin
      local newArr = map(expandRecordArgsInExp, arr)
      DAE.ARRAY(ty, scalar, MetaModelica.list(newArr...))
    end
    _ => exp
  end
end

"""
  Build a DAE expression for a single record field argument.
  For scalar fields: a simple CREF.
  For 1D array fields: a DAE.ARRAY of element CREFs with subscripts.
  For 2D array fields: a nested DAE.ARRAY of element CREFs.
"""
function buildFieldArgExp(flatName::String, fieldTy::DAE.Type)::DAE.Exp
  @match fieldTy begin
    DAE.T_ARRAY(elemTy, dims) => begin
      local dimSizes = Int[]
      for d in dims
        @match d begin
          DAE.DIM_INTEGER(size) => push!(dimSizes, size)
          _ => return DAE.CREF(DAE.CREF_IDENT(flatName, fieldTy, MetaModelica.nil), fieldTy)
        end
      end
      if length(dimSizes) == 1
        #= 1D array: ARRAY([CREF(name, subs=[1]), CREF(name, subs=[2]), ...]) =#
        local elems = DAE.Exp[]
        for i in 1:dimSizes[1]
          local subs = MetaModelica.list(DAE.INDEX(DAE.ICONST(i)))
          local cr = DAE.CREF_IDENT(flatName, fieldTy, subs)
          push!(elems, DAE.CREF(cr, elemTy))
        end
        return DAE.ARRAY(fieldTy, true, MetaModelica.list(elems...))
      elseif length(dimSizes) == 2
        #= 2D array: nested ARRAY of row ARRAYs =#
        local rowTy = DAE.T_ARRAY(elemTy, MetaModelica.list(DAE.DIM_INTEGER(dimSizes[2])))
        local rows = DAE.Exp[]
        for i in 1:dimSizes[1]
          local rowElems = DAE.Exp[]
          for j in 1:dimSizes[2]
            local subs = MetaModelica.list(DAE.INDEX(DAE.ICONST(i)), DAE.INDEX(DAE.ICONST(j)))
            local cr = DAE.CREF_IDENT(flatName, fieldTy, subs)
            push!(rowElems, DAE.CREF(cr, elemTy))
          end
          push!(rows, DAE.ARRAY(rowTy, true, MetaModelica.list(rowElems...)))
        end
        return DAE.ARRAY(fieldTy, false, MetaModelica.list(rows...))
      else
        #= Higher dimensions: pass as bare CREF =#
        return DAE.CREF(DAE.CREF_IDENT(flatName, fieldTy, MetaModelica.nil), fieldTy)
      end
    end
    _ => begin
      #= Scalar field: simple CREF =#
      return DAE.CREF(DAE.CREF_IDENT(flatName, fieldTy, MetaModelica.nil), fieldTy)
    end
  end
end

# ============================================================================
#  IFEXP resolution in parameter/variable bindings
#
#  Resolves constant-condition IFEXPs at the simcode level, before code gen.
#  For non-constant conditions, the expression is left unchanged and the
#  code-gen fallback generates ModelingToolkit.ifelse.
# ============================================================================

"""
  Traverse all parameter and array-parameter bindings in the simcode and
  resolve IFEXP nodes whose conditions can be evaluated at compile time.
"""
function resolveIfExpInBindings!(simCode)
  local ht = simCode.stringToSimVarHT
  for (name, (idx, simVar)) in ht
    local newVarKind = @match simVar.varKind begin
      SimulationCode.PARAMETER(SOME(bindExp)) => begin
        local newBind = resolveConstantIfExp(bindExp)
        newBind === bindExp ? nothing : SimulationCode.PARAMETER(SOME(newBind))
      end
      SimulationCode.ARRAY_PARAMETER(dims, SOME(bindExp)) => begin
        local newBind = resolveConstantIfExp(bindExp)
        newBind === bindExp ? nothing : SimulationCode.ARRAY_PARAMETER(dims, SOME(newBind))
      end
      _ => nothing
    end
    if newVarKind !== nothing
      local newSimVar = SimulationCode.SIMVAR(simVar.name, simVar.index, newVarKind, simVar.attributes)
      ht[name] = (idx, newSimVar)
    end
  end
  return simCode
end

"""
  Recursively resolve IFEXP nodes in a DAE expression.
  - BCONST(true/false): select the correct branch
  - Comparison of two constants (RCONST/ICONST): evaluate and select
  - noEvent wrapper: strip and recurse into the inner expression
  - Otherwise: leave unchanged (code-gen handles with ModelingToolkit.ifelse)
"""
function resolveConstantIfExp(exp::DAE.Exp)::DAE.Exp
  @match exp begin
    DAE.IFEXP(DAE.BCONST(true), thenExp, _) => resolveConstantIfExp(thenExp)
    DAE.IFEXP(DAE.BCONST(false), _, elseExp) => resolveConstantIfExp(elseExp)
    DAE.IFEXP(cond, thenExp, elseExp) => begin
      #= Try to evaluate the condition to a boolean =#
      local resolved = tryEvalCondition(cond)
      if resolved === true
        resolveConstantIfExp(thenExp)
      elseif resolved === false
        resolveConstantIfExp(elseExp)
      else
        #= Cannot resolve: recurse into sub-expressions but keep IFEXP =#
        DAE.IFEXP(resolveConstantIfExp(cond),
                  resolveConstantIfExp(thenExp),
                  resolveConstantIfExp(elseExp))
      end
    end
    #= Recurse into common expression wrappers =#
    DAE.BINARY(e1, op, e2) => begin
      local ne1 = resolveConstantIfExp(e1)
      local ne2 = resolveConstantIfExp(e2)
      (ne1 === e1 && ne2 === e2) ? exp : DAE.BINARY(ne1, op, ne2)
    end
    DAE.UNARY(op, e1) => begin
      local ne1 = resolveConstantIfExp(e1)
      ne1 === e1 ? exp : DAE.UNARY(op, ne1)
    end
    DAE.CALL(path, expLst, attr) => begin
      local changed = false
      local newArgs = DAE.Exp[]
      for arg in expLst
        local newArg = resolveConstantIfExp(arg)
        if newArg !== arg
          changed = true
        end
        push!(newArgs, newArg)
      end
      changed ? DAE.CALL(path, MetaModelica.list(newArgs...), attr) : exp
    end
    DAE.ARRAY(ty, scalar, arr) => begin
      local changed = false
      local newArr = DAE.Exp[]
      for elem in arr
        local newElem = resolveConstantIfExp(elem)
        if newElem !== elem
          changed = true
        end
        push!(newArr, newElem)
      end
      changed ? DAE.ARRAY(ty, scalar, MetaModelica.list(newArr...)) : exp
    end
    _ => exp
  end
end

"""
  Try to evaluate a DAE condition expression to a Bool.
  Returns `true`, `false`, or `nothing` if evaluation is not possible.
"""
function tryEvalCondition(cond::DAE.Exp)::Union{Bool, Nothing}
  @match cond begin
    DAE.BCONST(val) => val
    #= Strip noEvent wrapper =#
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      length(innerArgs) == 1 ? tryEvalCondition(innerArgs[1]) : nothing
    end
    #= Relational comparisons between constants =#
    DAE.RELATION(e1, op, e2, _, _) => begin
      local v1 = tryEvalNumeric(e1)
      local v2 = tryEvalNumeric(e2)
      if v1 !== nothing && v2 !== nothing
        @match op begin
          DAE.LESS(__) => v1 < v2
          DAE.LESSEQ(__) => v1 <= v2
          DAE.GREATER(__) => v1 > v2
          DAE.GREATEREQ(__) => v1 >= v2
          DAE.EQUAL(__) => v1 == v2
          DAE.NEQUAL(__) => v1 != v2
          _ => nothing
        end
      else
        nothing
      end
    end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      local r1 = tryEvalCondition(e1)
      local r2 = tryEvalCondition(e2)
      (r1 !== nothing && r2 !== nothing) ? (r1 && r2) : nothing
    end
    DAE.LBINARY(e1, DAE.OR(__), e2) => begin
      local r1 = tryEvalCondition(e1)
      local r2 = tryEvalCondition(e2)
      (r1 !== nothing && r2 !== nothing) ? (r1 || r2) : nothing
    end
    DAE.LUNARY(DAE.NOT(__), e1) => begin
      local r1 = tryEvalCondition(e1)
      r1 !== nothing ? !r1 : nothing
    end
    _ => nothing
  end
end

"""
  Try to evaluate a DAE expression to a numeric value.
  Returns Float64, or nothing if evaluation is not possible.
"""
function tryEvalNumeric(exp::DAE.Exp)::Union{Float64, Nothing}
  @match exp begin
    DAE.RCONST(val) => Float64(val)
    DAE.ICONST(val) => Float64(val)
    DAE.UNARY(DAE.UMINUS(__), inner) => begin
      local v = tryEvalNumeric(inner)
      v !== nothing ? -v : nothing
    end
    DAE.UNARY(DAE.UMINUS_ARR(__), inner) => begin
      local v = tryEvalNumeric(inner)
      v !== nothing ? -v : nothing
    end
    DAE.CALL(Absyn.IDENT("abs"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        local v = tryEvalNumeric(innerArgs[1])
        v !== nothing ? abs(v) : nothing
      else
        nothing
      end
    end
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      length(innerArgs) == 1 ? tryEvalNumeric(innerArgs[1]) : nothing
    end
    DAE.BINARY(e1, DAE.ADD(__), e2) => begin
      local v1 = tryEvalNumeric(e1)
      local v2 = tryEvalNumeric(e2)
      (v1 !== nothing && v2 !== nothing) ? v1 + v2 : nothing
    end
    DAE.BINARY(e1, DAE.SUB(__), e2) => begin
      local v1 = tryEvalNumeric(e1)
      local v2 = tryEvalNumeric(e2)
      (v1 !== nothing && v2 !== nothing) ? v1 - v2 : nothing
    end
    DAE.BINARY(e1, DAE.MUL(__), e2) => begin
      local v1 = tryEvalNumeric(e1)
      local v2 = tryEvalNumeric(e2)
      (v1 !== nothing && v2 !== nothing) ? v1 * v2 : nothing
    end
    DAE.BINARY(e1, DAE.DIV(__), e2) => begin
      local v1 = tryEvalNumeric(e1)
      local v2 = tryEvalNumeric(e2)
      (v1 !== nothing && v2 !== nothing && v2 != 0.0) ? v1 / v2 : nothing
    end
    _ => nothing
  end
end
