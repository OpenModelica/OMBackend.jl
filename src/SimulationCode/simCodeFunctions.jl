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
      local language = occursin("external \"FORTRAN 77\"", libInfo) ? "FORTRAN 77" : "C"
      libInfo = replace(libInfo, "external \"C\"" => "", "external \"FORTRAN 77\"" => "")
      libInfo = replace(libInfo, "'" => "")
      push!(functions, EXTERNAL_MODELICA_FUNCTION(n, inputs, outputs, locals, language, libInfo))
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
  local inputNames = Set{String}(string(i.componentRef) for i in func.inputs)
  for output in func.outputs
    flattenedVars = flattenRecordVar(output, recordFieldMap; inputNames = inputNames)
    append!(flattenedOutputs, flattenedVars)
  end

  #= Transform statements to use flattened names =#
  local transformedStatements = transformStatementsForFlattenedRecords(func.statements, recordFieldMap)

  #= For record constructors with empty algorithm sections, generate synthetic
     assignments binding each output field to the matching input by name.
     E.g., Complex(re, im) with output Complex result -> result_re = re; result_im = im.
     This handles implicit record constructors and constructor functions where
     output fields are bound via modifiers (output Complex result(re=re, im=im)). =#
  if isempty(transformedStatements) && !isempty(recordFieldMap)
    local inputNameSet = OrderedSet{String}(string(inp.componentRef) for inp in flattenedInputs)
    for output in func.outputs
      local outName = string(output.componentRef)
      if haskey(recordFieldMap, outName)
        for (fieldName, fieldTy) in recordFieldMap[outName]
          if fieldName in inputNameSet
            local flatOutName = outName * OMBackend.COMPONENT_SEPARATOR * fieldName
            local outCref = DAE.CREF_IDENT(flatOutName, fieldTy, MetaModelica.nil)
            local inCref = DAE.CREF_IDENT(fieldName, fieldTy, MetaModelica.nil)
            push!(transformedStatements, DAE.STMT_ASSIGN(
              fieldTy,
              DAE.CREF(outCref, fieldTy),
              DAE.CREF(inCref, fieldTy),
              DAE.emptyElementSource
            ))
          end
        end
      end
    end
  end

  local locals = isempty(recordFieldMap) ? func.locals :
                 [_transformLocalForFlattenedRecords(l, recordFieldMap) for l in func.locals]
  return MODELICA_FUNCTION(func.name, flattenedInputs, flattenedOutputs, locals, transformedStatements)
end

function flattenRecordParametersInFunction(func::EXTERNAL_MODELICA_FUNCTION)::EXTERNAL_MODELICA_FUNCTION
  #= External functions are not transformed for now =#
  return func
end

"""
  Flatten a single variable. If it's a record type, returns multiple variables for each field.
  Otherwise returns the original variable in a vector.
"""
function flattenRecordVar(v::DAE.VAR, recordFieldMap::Dict{String, Vector{Tuple{String, DAE.Type}}};
                          inputNames::Set{String} = Set{String}())::Vector{DAE.VAR}
  local baseName = string(v.componentRef)
  #= An array of records (`input Complex u[:]`: ty is the record, dims the array) keeps
     its dimensions on every field: u_re[:], u_im[:]. A dimension may refer to an
     earlier record array (`c2[size(c1, 1)]`), whose fields are already in the map. =#
  local recordDims = _transformDimsForFlattenedRecords(v.dims, recordFieldMap)
  @match v.ty begin
    DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _) => begin
      local flattenedVars = DAE.VAR[]
      local fieldInfo = Tuple{String, DAE.Type}[]
      local fieldNames = String[f.name for f in varLst]
      for (fieldIndex, field) in enumerate(varLst)
        @match field begin
          DAE.TYPES_VAR(fieldName, _, fieldTy, _, _) => begin
            local flatName = baseName * OMBackend.COMPONENT_SEPARATOR * fieldName
            #= Extract dims from fieldTy if it is an array type =#
            local fieldDims = @match fieldTy begin
              DAE.T_ARRAY(_, dims) => dims
              _ => MetaModelica.nil
            end
            #= Create a new DAE.VAR with flattened name and field type =#
            local flatIdentTy = isempty(recordDims) ? fieldTy : DAE.T_ARRAY(fieldTy, recordDims)
            #= In a record array the VAR's dims carry every dimension; ty is the element type. =#
            local varTy = isempty(recordDims) ? fieldTy : _elementType(fieldTy)
            local flatCref = DAE.CREF_IDENT(flatName, flatIdentTy, MetaModelica.nil)
            local flatVar = DAE.VAR(
              flatCref,
              v.kind,
              v.direction,
              v.parallelism,
              v.protection,
              varTy,
              v.direction isa DAE.OUTPUT ? _outputFieldBinding(v, fieldIndex, field, fieldNames, inputNames, recordFieldMap) : NONE(),
              listAppend(recordDims, fieldDims),
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
    #= Its dimensions or binding may read an earlier record (`indices[size(v, 1)]`). =#
    _ => return [_transformLocalForFlattenedRecords(v, recordFieldMap)]
  end
end

_elementType(@nospecialize(ty::DAE.Type)) = ty isa DAE.T_ARRAY ? _elementType(ty.ty) : ty

#= Where a field of a record output starts (MLS 12.4.4: an output starts at its
   binding): the record's binding (a constructor's argument for the field, else
   the field of it), else the field's binding: a modifier of the output, read
   in the function (`output Complex result(re = re)`: the input re), or a
   default of the record, its references to other fields renamed to theirs.
   The fields had none: an output field the body does not assign was 0.0
   (`record R Real b = 2;` gave r.b = 0). =#
function _outputFieldBinding(v::DAE.VAR, fieldIndex::Int, field::DAE.TYPES_VAR, fieldNames::Vector{String},
                             inputNames::Set{String}, recordFieldMap::Dict)
  if v.binding isa SOME
    local e = transformExpForFlattenedRecords(v.binding.data, recordFieldMap)
    local parts = if e isa DAE.RECORD
      collect(e.exps)
    elseif e isa DAE.CALL && _isConstructorCall(e)
      collect(e.expLst)
    else
      nothing
    end
    return parts !== nothing && length(parts) == length(fieldNames) ? SOME(parts[fieldIndex]) :
      SOME(DAE.RSUB(e, fieldIndex, field.name, field.ty))
  end
  field.binding isa DAE.EQBOUND || return NONE()
  local base = string(v.componentRef)
  local (renamed, _) = Util.traverseExpBottomUp(field.binding.exp, (x, acc) -> begin
      if x isa DAE.CREF && x.componentRef isa DAE.CREF_IDENT && x.componentRef.ident in fieldNames &&
         !(x.componentRef.ident in inputNames)
        x = DAE.CREF(DAE.CREF_IDENT(base * OMBackend.COMPONENT_SEPARATOR * x.componentRef.ident, x.ty, MetaModelica.nil), x.ty)
      end
      (x, acc)
    end, nothing)
  return SOME(transformExpForFlattenedRecords(renamed, recordFieldMap))
end

#= A call of a record's constructor: its arguments are the fields. =#
_isConstructorCall(e::DAE.CALL)::Bool =
  e.attr.ty isa DAE.T_COMPLEX && e.attr.ty.complexClassType isa DAE.ClassInf.RECORD &&
  string(e.attr.ty.complexClassType.path) == string(e.path)

function _transformDimsForFlattenedRecords(dims::List, recordFieldMap::Dict)::List
  return MetaModelica.list((_transformDimForFlattenedRecords(d, recordFieldMap) for d in dims)...)
end

function _transformDimForFlattenedRecords(@nospecialize(d::DAE.Dimension), recordFieldMap::Dict)::DAE.Dimension
  @match d begin
    DAE.DIM_EXP(e) => DAE.DIM_EXP(transformExpForFlattenedRecords(e, recordFieldMap))
    _ => d
  end
end

#= A variable whose binding or dimensions read a flattened record (`m = size(u, 1)`). =#
function _transformLocalForFlattenedRecords(v::DAE.VAR, recordFieldMap::Dict)::DAE.VAR
  local binding = @match v.binding begin
    SOME(e) => SOME(transformExpForFlattenedRecords(e, recordFieldMap))
    _ => v.binding
  end
  return DAE.VAR(v.componentRef, v.kind, v.direction, v.parallelism, v.protection, v.ty, binding,
                 _transformDimsForFlattenedRecords(v.dims, recordFieldMap), v.connectorType, v.source,
                 v.variableAttributesOption, v.comment, v.innerOuter)
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
  #= Check if LHS is a flattened record variable, or an element of a flattened record array =#
  local (lhsName, lhsSubs) = @match stmt.exp1 begin
    DAE.CREF(DAE.CREF_IDENT(ident, _, subs), _) => (ident, subs)
    _ => (nothing, MetaModelica.nil)
  end
  #= If LHS is a record variable and RHS is a RECORD expression, expand into field assignments =#
  if lhsName !== nothing && haskey(recordFieldMap, lhsName)
    local fieldLhs = (fieldName, fieldTy) -> DAE.CREF(DAE.CREF_IDENT(lhsName * OMBackend.COMPONENT_SEPARATOR * fieldName,
                                                                     fieldTy, lhsSubs), fieldTy)
    isempty(lhsSubs) || return _recordElementAssignment(stmt, lhsName, fieldLhs, recordFieldMap)
    #= A record reference on the right (a record, or an element of a record array:
       result := v[index]) is copied field by field. =#
    local rhsFields = expandRecordArgForCall(stmt.exp, recordFieldMap)
    if rhsFields !== nothing
      return DAE.Statement[DAE.STMT_ASSIGN(fieldTy, fieldLhs(fieldName, fieldTy), rhsFields[i], stmt.source)
                           for (i, (fieldName, fieldTy)) in enumerate(recordFieldMap[lhsName])]
    end
    @match stmt.exp begin
      DAE.RECORD(path, exps, fieldNames, ty) => begin
        local fieldInfo = recordFieldMap[lhsName]
        local expVec = collect(exps)
        local stmts = DAE.Statement[]
        for (i, (fieldName, fieldTy)) in enumerate(fieldInfo)
          local rhsExp = transformExpForFlattenedRecords(expVec[i], recordFieldMap)
          push!(stmts, DAE.STMT_ASSIGN(fieldTy, fieldLhs(fieldName, fieldTy), rhsExp, stmt.source))
        end
        return stmts
      end
      _ => begin
        local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
        #= RHS is not a RECORD literal but LHS is a record variable.
           Keep the original assignment to evaluate the RHS once (into a tuple),
           then extract each field via tuple indexing. A whole record array is
           scattered by codegen from the one assignment. =#
        local fieldInfo = recordFieldMap[lhsName]
        local stmts = DAE.Statement[]
        push!(stmts, DAE.STMT_ASSIGN(stmt.type_, stmt.exp1, newExp, stmt.source))
        stmt.type_ isa DAE.T_ARRAY && return stmts
        for (i, (fieldName, fieldTy)) in enumerate(fieldInfo)
          local flatName = lhsName * OMBackend.COMPONENT_SEPARATOR * fieldName
          local flatCref = DAE.CREF_IDENT(flatName, fieldTy, MetaModelica.nil)
          local lhsExp = DAE.CREF(flatCref, fieldTy)
          local rhsCref = DAE.CREF_IDENT(lhsName, stmt.type_, MetaModelica.nil)
          local rhsExp = DAE.ASUB(DAE.CREF(rhsCref, stmt.type_), MetaModelica.list(DAE.INDEX(DAE.ICONST(i))))
          push!(stmts, DAE.STMT_ASSIGN(fieldTy, lhsExp, rhsExp, stmt.source))
        end
        return stmts
      end
    end
  else
    local newExp1 = transformExpForFlattenedRecords(stmt.exp1, recordFieldMap)
    local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
    return [DAE.STMT_ASSIGN(stmt.type_, newExp1, newExp, stmt.source)]
  end
end

#= An element of a record array, v[i] := value: the value goes into a temporary record
   first (a call is evaluated once and scattered by codegen), then into the field
   elements, so a value that reads v[i] itself (a swap of fields) sees the old element. =#
function _recordElementAssignment(stmt::DAE.STMT_ASSIGN, lhsName::String, fieldLhs, recordFieldMap::Dict)::Vector{DAE.Statement}
  local fields = recordFieldMap[lhsName]
  local tmpBase = "__omjl_" * lhsName
  local tmpField = (fieldName, fieldTy) -> DAE.CREF(DAE.CREF_IDENT(tmpBase * OMBackend.COMPONENT_SEPARATOR * fieldName,
                                                                   fieldTy, MetaModelica.nil), fieldTy)
  local stmts = DAE.Statement[]
  local rhsFields = expandRecordArgForCall(stmt.exp, recordFieldMap)
  if rhsFields !== nothing
    for (k, (fieldName, fieldTy)) in enumerate(fields)
      push!(stmts, DAE.STMT_ASSIGN(fieldTy, tmpField(fieldName, fieldTy), rhsFields[k], stmt.source))
    end
  elseif stmt.exp isa DAE.RECORD
    for (k, (fieldName, fieldTy)) in enumerate(fields)
      push!(stmts, DAE.STMT_ASSIGN(fieldTy, tmpField(fieldName, fieldTy),
                                   transformExpForFlattenedRecords(listGet(stmt.exp.exps, k), recordFieldMap), stmt.source))
    end
  else
    local tmpCref = DAE.CREF(DAE.CREF_IDENT(tmpBase, stmt.type_, MetaModelica.nil), stmt.type_)
    push!(stmts, DAE.STMT_ASSIGN(stmt.type_, tmpCref, transformExpForFlattenedRecords(stmt.exp, recordFieldMap), stmt.source))
  end
  for (fieldName, fieldTy) in fields
    push!(stmts, DAE.STMT_ASSIGN(fieldTy, fieldLhs(fieldName, fieldTy), tmpField(fieldName, fieldTy), stmt.source))
  end
  return stmts
end

#= A whole record array (y := rotateAll(u, phi), result := v) keeps one statement: codegen
   scatters the call's field arrays in one evaluation, or copies field by field
   (_recordAssignment in CodeGeneration/algorithmic.jl). Any other value (an if-expression,
   an array constructor) is not supported: an error, not fields left at their defaults. =#
function transformStatementForFlattenedRecords(stmt::DAE.STMT_ASSIGN_ARR, recordFieldMap::Dict)::Vector{DAE.Statement}
  local newLhs = transformExpForFlattenedRecords(stmt.lhs, recordFieldMap)
  local newExp = transformExpForFlattenedRecords(stmt.exp, recordFieldMap)
  if expandRecordArgForCall(stmt.lhs, recordFieldMap) !== nothing &&
     !(newExp isa DAE.CALL) && expandRecordArgForCall(stmt.exp, recordFieldMap) === nothing
    error("assigning $(string(stmt.exp)) to the record array $(string(stmt.lhs)) in a function is not supported")
  end
  return [DAE.STMT_ASSIGN_ARR(stmt.type_, newLhs, newExp, stmt.source)]
end

#= A body's statement list as a Vector{DAE.Statement}: collect of an empty list is a
   Vector{Any} (an empty loop or branch body; MSL Fluid's AST_BatchPlant). =#
_statementVector(statementLst)::Vector{DAE.Statement} = DAE.Statement[s for s in statementLst]

function transformStatementForFlattenedRecords(stmt::DAE.STMT_FOR, recordFieldMap::Dict)::Vector{DAE.Statement}
  local body = transformStatementsForFlattenedRecords(_statementVector(stmt.statementLst), recordFieldMap)
  return [DAE.STMT_FOR(stmt.type_, stmt.iterIsArray, stmt.iter, stmt.index,
                       transformExpForFlattenedRecords(stmt.range, recordFieldMap),
                       MetaModelica.list(body...), stmt.source)]
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_WHILE, recordFieldMap::Dict)::Vector{DAE.Statement}
  local body = transformStatementsForFlattenedRecords(_statementVector(stmt.statementLst), recordFieldMap)
  return [DAE.STMT_WHILE(transformExpForFlattenedRecords(stmt.exp, recordFieldMap),
                         MetaModelica.list(body...), stmt.source)]
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_IF, recordFieldMap::Dict)::Vector{DAE.Statement}
  local body = transformStatementsForFlattenedRecords(_statementVector(stmt.statementLst), recordFieldMap)
  return [DAE.STMT_IF(transformExpForFlattenedRecords(stmt.exp, recordFieldMap), MetaModelica.list(body...),
                      _transformElseForFlattenedRecords(stmt.else_, recordFieldMap), stmt.source)]
end

function _transformElseForFlattenedRecords(@nospecialize(e::DAE.Else), recordFieldMap::Dict)::DAE.Else
  @match e begin
    DAE.ELSEIF(cond, stmts, rest) => DAE.ELSEIF(transformExpForFlattenedRecords(cond, recordFieldMap),
      MetaModelica.list(transformStatementsForFlattenedRecords(_statementVector(stmts), recordFieldMap)...),
      _transformElseForFlattenedRecords(rest, recordFieldMap))
    DAE.ELSE(stmts) => DAE.ELSE(MetaModelica.list(transformStatementsForFlattenedRecords(_statementVector(stmts), recordFieldMap)...))
    _ => e
  end
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_NORETCALL, recordFieldMap::Dict)::Vector{DAE.Statement}
  return [DAE.STMT_NORETCALL(transformExpForFlattenedRecords(stmt.exp, recordFieldMap), stmt.source)]
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_ASSERT, recordFieldMap::Dict)::Vector{DAE.Statement}
  return [DAE.STMT_ASSERT(transformExpForFlattenedRecords(stmt.cond, recordFieldMap),
                          transformExpForFlattenedRecords(stmt.msg, recordFieldMap),
                          transformExpForFlattenedRecords(stmt.level, recordFieldMap), stmt.source)]
end

function transformStatementForFlattenedRecords(stmt::DAE.STMT_TUPLE_ASSIGN, recordFieldMap::Dict)::Vector{DAE.Statement}
  #= A record target takes the callee's flattened field outputs: (result, index) := 'max'(v).
     The tuple type lists the fields' types in the record's place: one type per
     target, as codegen reads them. =#
  local outputTypes = stmt.type_ isa DAE.T_TUPLE ? collect(stmt.type_.types) : nothing
  local targets = DAE.Exp[]
  local types = DAE.Type[]
  for (k, e) in enumerate(stmt.expExpLst)
    local fields = expandRecordArgForCall(e, recordFieldMap)
    if fields === nothing
      push!(targets, transformExpForFlattenedRecords(e, recordFieldMap))
      outputTypes === nothing || push!(types, outputTypes[k])
    else
      append!(targets, fields)
      append!(types, (f.ty for f in fields))
    end
  end
  local type_ = outputTypes === nothing ? stmt.type_ : DAE.T_TUPLE(MetaModelica.list(types...), NONE())
  return [DAE.STMT_TUPLE_ASSIGN(type_, MetaModelica.list(targets...),
                                transformExpForFlattenedRecords(stmt.exp, recordFieldMap), stmt.source)]
end

Base.@nospecializeinfer function transformStatementForFlattenedRecords(@nospecialize(stmt::DAE.Statement), recordFieldMap::Dict)::Vector{DAE.Statement}
  #= For other statement types, return unchanged for now =#
  return [stmt]
end

"""
  If exp is a CREF to a record variable in recordFieldMap, expand it into a vector
  of field CREFs (e.g., R_rel becomes [R_rel_T, R_rel_w]); an element of a record
  array keeps its subscripts (u[k] becomes [u_re[k], u_im[k]]).
  Returns nothing if exp is not an expandable record reference.
"""
Base.@nospecializeinfer function expandRecordArgForCall(@nospecialize(exp::DAE.Exp), recordFieldMap::Dict)
  @match exp begin
    DAE.CREF(DAE.CREF_IDENT(ident, _, subs), _) => begin
      if haskey(recordFieldMap, ident)
        local fieldInfo = recordFieldMap[ident]
        local fieldExps = DAE.Exp[]
        for (fieldName, fieldTy) in fieldInfo
          local flatName = ident * OMBackend.COMPONENT_SEPARATOR * fieldName
          local flatCref = DAE.CREF_IDENT(flatName, fieldTy, subs)
          push!(fieldExps, DAE.CREF(flatCref, fieldTy))
        end
        return fieldExps
      end
      return nothing
    end
    _ => return nothing
  end
end

function _recordFieldRefParts(cr::DAE.CREF_IDENT)
  return (cr.ident, cr.identType, cr.subscriptLst)
end

function _recordFieldRefParts(cr::DAE.CREF_QUAL)
  local (innerName, innerTy, innerSubs) = _recordFieldRefParts(cr.componentRef)
  return (cr.ident * OMBackend.COMPONENT_SEPARATOR * innerName, innerTy, innerSubs)
end

"""
  Transform expressions to replace record field accesses with flattened names,
  anywhere in the expression (reductions, relations, ranges, ...):
  R.T[1,2] becomes R_T[1,2], an element's field u[k].re becomes u_re[k], the size of
  a record array size(u, 1) becomes size(u_re, 1), and record arguments of calls
  become their fields (expandRecordArgForCall). In a function without record inputs or
  outputs only an expression with a call that has a record-valued argument is
  traversed: the callee takes the record's fields (MSL Water IF97's
  T_props_ph(p, h, waterBaseProp_ph(p, h, phase, region)) inside T_ph; passed whole,
  its wrapper was called with 3 of its 18 arguments and the process crashed).
"""
function transformExpForFlattenedRecords(exp::DAE.Exp, recordFieldMap::Dict)::DAE.Exp
  isempty(recordFieldMap) && !_hasRecordValuedCallArgument(exp) && return exp
  return first(Util.traverseExpTopDown(exp, _flattenRecordRefs, recordFieldMap))
end

function _hasRecordValuedCallArgument(exp::DAE.Exp)::Bool
  local found = Ref(false)
  Util.traverseExpTopDown(exp, (e, acc) -> begin
      if e isa DAE.CALL && !e.attr.builtin && any(a -> _recordValueFieldTypes(a) !== nothing, e.expLst)
        found[] = true
      end
      (e, !found[], acc)
    end, nothing)
  return found[]
end

Base.@nospecializeinfer function _flattenRecordRefs(@nospecialize(exp::DAE.Exp), recordFieldMap::Dict)
  @match exp begin
    DAE.CREF(DAE.CREF_QUAL(ident, _, outerSubs, componentRef), ty) where {haskey(recordFieldMap, ident)} => begin
      #= Keep the field base name; the element's subscripts come first, then the field's. =#
      local (innerName, fieldTy, innerSubs) = _recordFieldRefParts(componentRef)
      local flatName = ident * OMBackend.COMPONENT_SEPARATOR * innerName
      #= Continue into the subscripts: the flat name is not a record of the map. =#
      (DAE.CREF(DAE.CREF_IDENT(flatName, fieldTy, listAppend(outerSubs, innerSubs)), ty), true, recordFieldMap)
    end
    DAE.SIZE(DAE.CREF(DAE.CREF_IDENT(ident, _, subs), _), sz) where {haskey(recordFieldMap, ident) &&
                                                                     !isempty(recordFieldMap[ident]) &&
                                                                     (isSome(sz) || !(recordFieldMap[ident][1][2] isa DAE.T_ARRAY))} => begin
      #= The record array's dimensions come first in every field: size(u, d) is size(u_f, d);
         size(u) needs a scalar field. =#
      local (fieldName, fieldTy) = first(recordFieldMap[ident])
      local fieldCref = DAE.CREF_IDENT(ident * OMBackend.COMPONENT_SEPARATOR * fieldName, fieldTy, subs)
      (DAE.SIZE(DAE.CREF(fieldCref, fieldTy), sz), true, recordFieldMap)
    end
    DAE.CALL(path, expLst, attr) => begin
      local newArgs = DAE.Exp[]
      for e in expLst
        local expanded = expandRecordArgForCall(e, recordFieldMap)
        if expanded !== nothing
          append!(newArgs, expanded)
          continue
        end
        local newE = transformExpForFlattenedRecords(e, recordFieldMap)
        #= A record-valued call as an argument passes its fields, as at equation call
           sites (expandRecordArgsInExp): '+'(c3, multiply(c1[i], c2[i])). The call is
           evaluated once per field. Builtins take the record whole. =#
        local fieldTys = attr.builtin ? nothing : _recordValueFieldTypes(newE)
        if fieldTys === nothing
          push!(newArgs, newE)
        else
          for (k, ty) in enumerate(fieldTys)
            push!(newArgs, DAE.TSUB(newE, k, ty))
          end
        end
      end
      (DAE.CALL(path, MetaModelica.list(newArgs...), attr), false, recordFieldMap)
    end
    _ => (exp, true, recordFieldMap)
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
  #= Typed-eltype Vector to satisfy `Vector{RESIDUAL_EQUATION}` field under `infer=false`. =#
  local newResEqs = RESIDUAL_EQUATION[]
  sizehint!(newResEqs, length(simCode.residualEquations))
  for eq in simCode.residualEquations
    if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      local expDAE = toDAEExp(eq.exp)
      local newExp = expandRecordArgsInExp(expDAE)
      push!(newResEqs, newExp === expDAE ? eq : typeof(eq)(newExp, eq.source, eq.attr))
    else
      push!(newResEqs, eq)
    end
  end
  @assign simCode.residualEquations = newResEqs
  #= And in the when-equations: a later pass must see the record fields they
     read (MSL Water: medium.phase's discrete equation, lifted into a when, calls
     bubbleEnthalpy(medium.sat); foldExplicitSingleAssign saw only medium_sat and
     folded medium_sat_Tsat away, which the when still read). =#
  @assign simCode.whenEquations = WHEN_EQUATION[WHEN_EQUATION(w.size, _expandRecordArgsInWhen(w.whenEquation), w.source, w.attr)
                                                for w in simCode.whenEquations]
  #= And the initial algorithms (the same when's initial part: the early pass
     binds the names its statements read). =#
  @assign simCode.initialAlgorithms = INITIAL_ALGORITHM[
    INITIAL_ALGORITHM(WhenOperator[_expandRecordArgsInWhenOp(op) for op in ia.statements],
                      DAE.Statement[Util.mapDAEStatementExps(expandRecordArgsInExp, s) for s in ia.daeStatements])
    for ia in simCode.initialAlgorithms]
  #= And the asserts: passed whole, a record read a name the simulation does
     not keep, and the assert was not checked (MSL Machines'
     brushVoltageDrop(brushParameters, ...), Media's bubbleEnthalpy(medium.sat)). =#
  @assign simCode.asserts = BDAE.ASSERT_EQUATION[
    BDAE.ASSERT_EQUATION(expandRecordArgsInExp(a.condition), expandRecordArgsInExp(a.message), a.level, a.source)
    for a in simCode.asserts]
  #= And the initial equations: a record argument was passed whole, its
     fields' names undefined in the initialization (MSL JointRRP's
     `e_im = resolve2(frame_im.R, ...)`, solved there since a parametric
     equation reading an unbound parameter goes to the initialization). =#
  local newInitEqs = typeof(simCode.initialEquations)()
  for eq in simCode.initialEquations
    push!(newInitEqs, if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      typeof(eq)(expandRecordArgsInExp(toDAEExp(eq.exp)), eq.source, eq.attr)
    elseif eq isa BDAE.EQUATION
      BDAE.EQUATION(expandRecordArgsInExp(toDAEExp(eq.lhs)), expandRecordArgsInExp(toDAEExp(eq.rhs)), eq.source, eq.attributes)
    elseif eq isa EQUATION
      EQUATION(expandRecordArgsInExp(toDAEExp(eq.lhs)), expandRecordArgsInExp(toDAEExp(eq.rhs)), eq.source, eq.attr)
    else
      eq
    end)
  end
  @assign simCode.initialEquations = newInitEqs
  #= Expand record arguments in parameter and array-parameter binding expressions =#
  local ht = simCode.stringToSimVarHT
  for (name, (idx, simVar)) in ht
    local newVarKind = @match simVar.varKind begin
      SimulationCode.PARAMETER(SOME(bindExp)) => begin
        local db = SimulationCode.toDAEExp(bindExp)
        local newBind = expandRecordArgsInExp(db)
        newBind === db ? nothing : SimulationCode.PARAMETER(SOME(SimulationCode.toSimExp(newBind)))
      end
      SimulationCode.ARRAY_PARAMETER(dims, SOME(bindExp)) => begin
        local db = SimulationCode.toDAEExp(bindExp)
        local newBind = expandRecordArgsInExp(db)
        newBind === db ? nothing : SimulationCode.ARRAY_PARAMETER(dims, SOME(SimulationCode.toSimExp(newBind)))
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

function _expandRecordArgsInSimExp(e::Exp)::Exp
  local d = toDAEExp(e)
  local n = expandRecordArgsInExp(d)
  return n === d ? e : toSimExp(n)
end

function _expandRecordArgsInWhen(w::WHEN_STMTS)::WHEN_STMTS
  return WHEN_STMTS(_expandRecordArgsInSimExp(w.condition),
                    WhenOperator[_expandRecordArgsInWhenOp(s) for s in w.whenStmtLst],
                    w.elsewhenPart === nothing ? nothing : _expandRecordArgsInWhen(w.elsewhenPart))
end

_expandRecordArgsInWhenOp(s::ASSIGN) = ASSIGN(s.left, _expandRecordArgsInSimExp(s.right), s.source)
_expandRecordArgsInWhenOp(s::REINIT) = REINIT(s.stateVar, _expandRecordArgsInSimExp(s.value), s.source)
_expandRecordArgsInWhenOp(s::NORETCALL) = NORETCALL(_expandRecordArgsInSimExp(s.exp), s.source)
_expandRecordArgsInWhenOp(s::ASSERT) = ASSERT(_expandRecordArgsInSimExp(s.condition), s.message, s.level, s.source)
_expandRecordArgsInWhenOp(@nospecialize(s::WhenOperator)) = s

"""
  Recursively traverse an expression and expand record arguments inside CALL nodes.
"""
function expandRecordArgsInExp(exp::DAE.Exp)::DAE.Exp
  @match exp begin
    DAE.CALL(path, expLst, attr) => begin
      local newArgs = DAE.Exp[]
      for arg in expLst
        local fieldArrays = attr.builtin ? nothing : _splitRecordArrayArg(arg)
        if fieldArrays !== nothing
          append!(newArgs, fieldArrays)
          continue
        end
        local ifFields = arg isa DAE.IFEXP && _isRecordShaped(arg) ? _recordIfExpFields(arg) : nothing
        if ifFields !== nothing
          append!(newArgs, ifFields)
          continue
        end
        @match arg begin
          DAE.CREF(cr, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _)) => begin
            local baseName = OMBackend.canonicalName(cr)
            for field in varLst
              @match field begin
                DAE.TYPES_VAR(fieldName, _, fieldTy, _, _) => begin
                  local flatName = baseName * OMBackend.COMPONENT_SEPARATOR * fieldName
                  push!(newArgs, buildFieldArgExp(flatName, fieldTy))
                end
                _ => nothing
              end
            end
          end
          _ => begin
            local expandedArg = expandRecordArgsInExp(arg)
            #= Check if the expanded argument has Complex record return type.
               If so, split it into TSUB expressions for each field, because the
               outer function wrapper expects flattened scalar arguments. =#
            local fieldTys = _recordValueFieldTypes(expandedArg)
            if fieldTys !== nothing
              for (fieldIdx, fieldTy) in enumerate(fieldTys)
                push!(newArgs, DAE.TSUB(expandedArg, fieldIdx, fieldTy))
              end
            else
              push!(newArgs, expandedArg)
            end
          end
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
    #= Relations and logical operators too: a call inside a condition
       (MSL Water: medium.h < bubbleEnthalpy(medium.sat) or ...). =#
    DAE.RELATION(e1, op, e2, idx, opt) => begin
      local new_e1 = expandRecordArgsInExp(e1)
      local new_e2 = expandRecordArgsInExp(e2)
      (new_e1 === e1 && new_e2 === e2) ? exp : DAE.RELATION(new_e1, op, new_e2, idx, opt)
    end
    DAE.LBINARY(e1, op, e2) => begin
      local new_e1 = expandRecordArgsInExp(e1)
      local new_e2 = expandRecordArgsInExp(e2)
      (new_e1 === e1 && new_e2 === e2) ? exp : DAE.LBINARY(new_e1, op, new_e2)
    end
    DAE.LUNARY(op, e1) => begin
      local new_e1 = expandRecordArgsInExp(e1)
      new_e1 === e1 ? exp : DAE.LUNARY(op, new_e1)
    end
    DAE.CAST(ty, e1) => begin
      local new_e1 = expandRecordArgsInExp(e1)
      new_e1 === e1 ? exp : DAE.CAST(ty, new_e1)
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

#= A record-valued if-expression argument, field by field: if c then a else b
   passes if c then a_f else b_f for each field f (the ComplexBlocks'
   if useConjugateInput1 then conj(u1) else u1). nothing unless both branches
   are records of the same fields: a record reference or a record-valued call. =#
function _recordIfExpFields(exp::DAE.IFEXP)::Union{Nothing, Vector{DAE.Exp}}
  local a = _recordFields(exp.expThen); a === nothing && return nothing
  local b = _recordFields(exp.expElse)
  (b === nothing || length(a) != length(b)) && return nothing
  local c = expandRecordArgsInExp(exp.expCond)
  return DAE.Exp[DAE.IFEXP(c, a[k], b[k]) for k in eachindex(a)]
end

#= An if-expression between record references or record-valued calls (checked
   before anything is expanded: most if-expression arguments are Real). =#
_isRecordShaped(@nospecialize(e::DAE.Exp))::Bool = @match e begin
  DAE.IFEXP(_, t, f) => _isRecordShaped(t) && _isRecordShaped(f)
  DAE.CREF(_, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), _, _)) => true
  DAE.CALL(attr = DAE.CALL_ATTR(ty = DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), _, _))) => true
  _ => false
end

Base.@nospecializeinfer function _recordFields(@nospecialize(exp::DAE.Exp))::Union{Nothing, Vector{DAE.Exp}}
  @match exp begin
    DAE.CREF(cr, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _)) => begin
      local baseName = OMBackend.canonicalName(cr)
      return DAE.Exp[buildFieldArgExp(baseName * OMBackend.COMPONENT_SEPARATOR * f.name, f.ty) for f in varLst]
    end
    DAE.IFEXP(__) => return _recordIfExpFields(exp)
    _ => begin
      local expanded = expandRecordArgsInExp(exp)
      local fieldTys = _recordValueFieldTypes(expanded)
      return fieldTys === nothing ? nothing : DAE.Exp[DAE.TSUB(expanded, k, ty) for (k, ty) in enumerate(fieldTys)]
    end
  end
end

"""
  An array of records passed to a function whose flattened signature takes one array
  per field (flattenRecordVar): {u[1], u[2]} becomes {u[1]_re, u[2]_re}, {u[1]_im, u[2]_im}
  (the scalarized names), a record-valued call element gives its field by TSUB. A reference
  to a whole record array with fixed dimensions is split the same way. Returns nothing
  unless the argument is such an array.
"""
function _splitRecordArrayArg(@nospecialize(arg::DAE.Exp))::Union{Nothing, Vector{DAE.Exp}}
  local elems = @match arg begin
    DAE.ARRAY(_, _, arr) => collect(arr)
    DAE.CREF(cr, DAE.T_ARRAY(elemTy && DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), _, _), dims)) => begin
      local n = @match collect(dims) begin
        [DAE.DIM_INTEGER(k)] => k
        _ => return nothing
      end
      local name = OMBackend.canonicalName(cr)
      occursin('[', name) && return nothing
      [DAE.CREF(DAE.CREF_IDENT(name, elemTy, MetaModelica.list(DAE.INDEX(DAE.ICONST(i)))), elemTy) for i in 1:n]
    end
    _ => return nothing
  end
  isempty(elems) && return nothing
  local fields = _recordElementFields(first(elems))
  (fields === nothing || !all(e -> _recordElementFields(e) !== nothing, elems)) && return nothing
  #= Array-valued fields are scalarized per element in the simvars: not split here. =#
  any(f -> f.ty isa DAE.T_ARRAY, fields) && return nothing
  local fieldArrays = DAE.Exp[]
  for (k, field) in enumerate(fields)
    local (fieldName, fieldTy) = @match field begin
      DAE.TYPES_VAR(fName, _, fTy, _, _) => (fName, fTy)
    end
    local parts = DAE.Exp[_recordElementField(e, k, fieldName, fieldTy) for e in elems]
    local arrTy = DAE.T_ARRAY(fieldTy, MetaModelica.list(DAE.DIM_INTEGER(length(parts))))
    push!(fieldArrays, DAE.ARRAY(arrTy, true, MetaModelica.list(parts...)))
  end
  return fieldArrays
end

#= Field k (`fieldName`) of a record element of an array argument: the scalarized name of a
   record reference (u[1] -> u[1]_re), a TSUB of a record-valued call. =#
Base.@nospecializeinfer function _recordElementField(@nospecialize(e::DAE.Exp), k::Int, fieldName::String,
                                                     @nospecialize(fieldTy::DAE.Type))::DAE.Exp
  @match e begin
    DAE.CREF(cr, _) => DAE.CREF(DAE.CREF_IDENT(OMBackend.canonicalName(cr) * OMBackend.COMPONENT_SEPARATOR * fieldName,
                                               fieldTy, MetaModelica.nil), fieldTy)
    DAE.RECORD(_, exps, _, _) => expandRecordArgsInExp(listGet(exps, k))
    _ => DAE.TSUB(expandRecordArgsInExp(e), k, fieldTy)
  end
end

#= The fields of a record reference, literal or record-valued call; nothing for other expressions. =#
Base.@nospecializeinfer function _recordElementFields(@nospecialize(e::DAE.Exp))
  @match e begin
    DAE.CREF(_, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _)) => collect(varLst)
    DAE.RECORD(_, _, _, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _)) => collect(varLst)
    _ => _getComplexReturnFields(e)
  end
end

#= The field types of a call returning a record, or an array of records (a flattened
   function returns one array per field, `output Complex v[:]` -> (v_re, v_im), each typed
   as an array so that codegen takes an array tuple element); nothing otherwise. =#
Base.@nospecializeinfer function _recordValueFieldTypes(@nospecialize(exp::DAE.Exp))::Union{Nothing, Vector{DAE.Type}}
  @match exp begin
    DAE.CALL(_, _, DAE.CALL_ATTR(ty = DAE.T_ARRAY(DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _), dims))) =>
      DAE.Type[DAE.T_ARRAY(f.ty, dims) for f in varLst]
    _ => begin
      local fields = _getComplexReturnFields(exp)
      fields === nothing ? nothing : DAE.Type[f.ty for f in fields]
    end
  end
end

"""
  Extract the field list from a DAE expression with Complex record return type.
  Returns the varLst if the expression has T_COMPLEX(RECORD) type, nothing otherwise.
  Used to split non-CREF Complex-typed arguments into per-field TSUB expressions.
"""
Base.@nospecializeinfer function _getComplexReturnFields(@nospecialize(exp::DAE.Exp))
  @match exp begin
    DAE.CALL(_, _, DAE.CALL_ATTR(ty = DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _))) => begin
      return collect(varLst)
    end
    _ => return nothing
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
        local db = SimulationCode.toDAEExp(bindExp)
        local newBind = resolveConstantIfExp(db, simCode)
        newBind === db ? nothing : SimulationCode.PARAMETER(SOME(SimulationCode.toSimExp(newBind)))
      end
      SimulationCode.ARRAY_PARAMETER(dims, SOME(bindExp)) => begin
        local db = SimulationCode.toDAEExp(bindExp)
        local newBind = resolveConstantIfExp(db, simCode)
        newBind === db ? nothing : SimulationCode.ARRAY_PARAMETER(dims, SOME(SimulationCode.toSimExp(newBind)))
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
# SIM.Exp delegation: BRANCH.condition / EQUATION.lhs|rhs are SIM.Exp post-migration.
resolveConstantIfExp(exp::Exp)::Exp = toSimExp(resolveConstantIfExp(toDAEExp(exp)))

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

function resolveConstantIfExp(exp::DAE.Exp, simCode::SIM_CODE)::DAE.Exp
  @match exp begin
    DAE.IFEXP(cond, thenExp, elseExp) => begin
      local resolved = tryEvalCondition(cond, simCode)
      if resolved === true
        resolveConstantIfExp(thenExp, simCode)
      elseif resolved === false
        resolveConstantIfExp(elseExp, simCode)
      else
        DAE.IFEXP(resolveConstantIfExp(cond, simCode),
                  resolveConstantIfExp(thenExp, simCode),
                  resolveConstantIfExp(elseExp, simCode))
      end
    end
    DAE.BINARY(e1, op, e2) => begin
      local ne1 = resolveConstantIfExp(e1, simCode)
      local ne2 = resolveConstantIfExp(e2, simCode)
      (ne1 === e1 && ne2 === e2) ? exp : DAE.BINARY(ne1, op, ne2)
    end
    DAE.UNARY(op, e1) => begin
      local ne1 = resolveConstantIfExp(e1, simCode)
      ne1 === e1 ? exp : DAE.UNARY(op, ne1)
    end
    DAE.LBINARY(e1, op, e2) => begin
      local ne1 = resolveConstantIfExp(e1, simCode)
      local ne2 = resolveConstantIfExp(e2, simCode)
      (ne1 === e1 && ne2 === e2) ? exp : DAE.LBINARY(ne1, op, ne2)
    end
    DAE.LUNARY(op, e1) => begin
      local ne1 = resolveConstantIfExp(e1, simCode)
      ne1 === e1 ? exp : DAE.LUNARY(op, ne1)
    end
    DAE.RELATION(e1, op, e2, idx, opt) => begin
      local ne1 = resolveConstantIfExp(e1, simCode)
      local ne2 = resolveConstantIfExp(e2, simCode)
      (ne1 === e1 && ne2 === e2) ? exp : DAE.RELATION(ne1, op, ne2, idx, opt)
    end
    DAE.CAST(ty, e1) => begin
      local ne1 = resolveConstantIfExp(e1, simCode)
      ne1 === e1 ? exp : DAE.CAST(ty, ne1)
    end
    DAE.CALL(path, expLst, attr) => begin
      local changed = false
      local newArgs = DAE.Exp[]
      for arg in expLst
        local newArg = resolveConstantIfExp(arg, simCode)
        changed |= newArg !== arg
        push!(newArgs, newArg)
      end
      changed ? DAE.CALL(path, MetaModelica.list(newArgs...), attr) : exp
    end
    DAE.ARRAY(ty, scalar, arr) => begin
      local changed = false
      local newArr = DAE.Exp[]
      for elem in arr
        local newElem = resolveConstantIfExp(elem, simCode)
        changed |= newElem !== elem
        push!(newArr, newElem)
      end
      changed ? DAE.ARRAY(ty, scalar, MetaModelica.list(newArr...)) : exp
    end
    DAE.ASUB(e1, subs) => begin
      #= subs are DAE.Subscript: recurse into each subscript's inner exp. =#
      local ne1 = resolveConstantIfExp(e1, simCode)
      local changed = ne1 !== e1
      local newSubs = DAE.Subscript[]
      for sub in subs
        local newSub = @match sub begin
          DAE.INDEX(se) => begin
            local ns = resolveConstantIfExp(se, simCode)
            ns === se ? sub : DAE.INDEX(ns)
          end
          DAE.SLICE(se) => begin
            local ns = resolveConstantIfExp(se, simCode)
            ns === se ? sub : DAE.SLICE(ns)
          end
          DAE.WHOLE_NONEXP(se) => begin
            local ns = resolveConstantIfExp(se, simCode)
            ns === se ? sub : DAE.WHOLE_NONEXP(ns)
          end
          _ => sub
        end
        changed |= newSub !== sub
        push!(newSubs, newSub)
      end
      changed ? DAE.ASUB(ne1, MetaModelica.list(newSubs...)) : exp
    end
    _ => exp
  end
end

#= SIM-native mirror of resolveConstantIfExp(::DAE.Exp, simCode): recurses on the
   SimCode Exp spine so the per-residual caller (pruneConstantConditions via
   _rewriteResidualIfExp) need not build a whole-tree DAE copy. === identity is
   preserved so unchanged subtrees are reused (no per-node toSimExp round-trip);
   only the small IFEXP condition round-trips through tryEvalCondition's DAE arm.
   Arms mirror the DAE method 1:1 on SIM struct fields. =#
function resolveConstantIfExp(exp::Exp, simCode::SIM_CODE)::Exp
  if exp isa IFEXP
    local resolved = tryEvalCondition(exp.cond, simCode)
    if resolved === true
      return resolveConstantIfExp(exp.thenExp, simCode)
    elseif resolved === false
      return resolveConstantIfExp(exp.elseExp, simCode)
    end
    local nc = resolveConstantIfExp(exp.cond, simCode)
    local nt = resolveConstantIfExp(exp.thenExp, simCode)
    local ne = resolveConstantIfExp(exp.elseExp, simCode)
    return (nc === exp.cond && nt === exp.thenExp && ne === exp.elseExp) ? exp : IFEXP(nc, nt, ne)
  elseif exp isa BINARY
    local n1 = resolveConstantIfExp(exp.exp1, simCode)
    local n2 = resolveConstantIfExp(exp.exp2, simCode)
    return (n1 === exp.exp1 && n2 === exp.exp2) ? exp : BINARY(n1, exp.op, n2)
  elseif exp isa UNARY
    local n1 = resolveConstantIfExp(exp.exp, simCode)
    return n1 === exp.exp ? exp : UNARY(exp.op, n1)
  elseif exp isa LBINARY
    local n1 = resolveConstantIfExp(exp.exp1, simCode)
    local n2 = resolveConstantIfExp(exp.exp2, simCode)
    return (n1 === exp.exp1 && n2 === exp.exp2) ? exp : LBINARY(n1, exp.op, n2)
  elseif exp isa LUNARY
    local n1 = resolveConstantIfExp(exp.exp, simCode)
    return n1 === exp.exp ? exp : LUNARY(exp.op, n1)
  elseif exp isa RELATION
    local n1 = resolveConstantIfExp(exp.exp1, simCode)
    local n2 = resolveConstantIfExp(exp.exp2, simCode)
    return (n1 === exp.exp1 && n2 === exp.exp2) ? exp : RELATION(n1, exp.op, n2, exp.index)
  elseif exp isa CAST
    local n1 = resolveConstantIfExp(exp.exp, simCode)
    return n1 === exp.exp ? exp : CAST(exp.ty, n1)
  elseif exp isa CALL
    #= Allocate lazily; an unchanged arg list returns the original node. =#
    local newArgs::Union{Nothing, Vector{Exp}} = nothing
    local i = 0
    for arg in exp.args
      i += 1
      local na = resolveConstantIfExp(arg, simCode)
      if newArgs === nothing
        if na !== arg
          newArgs = Exp[]
          for j in 1:(i - 1); push!(newArgs, exp.args[j]); end
          push!(newArgs, na)
        end
      else
        push!(newArgs, na)
      end
    end
    return newArgs === nothing ? exp : CALL(exp.path, newArgs, exp.attr)
  elseif exp isa ARRAY_EXP
    local newEls::Union{Nothing, Vector{Exp}} = nothing
    local i = 0
    for el in exp.elements
      i += 1
      local nel = resolveConstantIfExp(el, simCode)
      if newEls === nothing
        if nel !== el
          newEls = Exp[]
          for j in 1:(i - 1); push!(newEls, exp.elements[j]); end
          push!(newEls, nel)
        end
      else
        push!(newEls, nel)
      end
    end
    return newEls === nothing ? exp : ARRAY_EXP(exp.ty, exp.scalar, newEls)
  elseif exp isa ASUB
    local n1 = resolveConstantIfExp(exp.exp, simCode)
    local changed = n1 !== exp.exp
    local newSubs = Exp[]
    for sub in exp.subs
      local ns = resolveConstantIfExp(sub, simCode)
      changed |= ns !== sub
      push!(newSubs, ns)
    end
    return changed ? ASUB(n1, newSubs) : exp
  end
  return exp
end

"""
  Try to evaluate a DAE condition expression to a Bool.
  Returns `true`, `false`, or `nothing` if evaluation is not possible.
"""
# SIM.Exp delegation: callers post-migration pass SIM-native Exp.
tryEvalCondition(cond::Exp)::Union{Bool, Nothing} = tryEvalCondition(toDAEExp(cond))
tryEvalCondition(cond::Exp, simCode::SIM_CODE)::Union{Bool, Nothing} =
  tryEvalCondition(toDAEExp(cond), simCode)

Base.@nospecializeinfer function tryEvalCondition(@nospecialize(cond::DAE.Exp))::Union{Bool, Nothing}
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

Base.@nospecializeinfer function tryEvalCondition(@nospecialize(cond::DAE.Exp), simCode::SIM_CODE)::Union{Bool, Nothing}
  return _tryEvalCondition(cond, simCode, OrderedSet{String}())
end

Base.@nospecializeinfer function _tryEvalCondition(@nospecialize(cond::DAE.Exp), simCode::SIM_CODE, seen::OrderedSet{String})::Union{Bool, Nothing}
  @match cond begin
    DAE.BCONST(val) => val
    DAE.CREF(__) => begin
      local value = tryEvalScalar(cond, simCode, seen)
      value isa Bool ? value : nothing
    end
    DAE.IFEXP(c, t, e) => begin
      local cVal = _tryEvalCondition(c, simCode, seen)
      cVal === true ? _tryEvalCondition(t, simCode, seen) :
      cVal === false ? _tryEvalCondition(e, simCode, seen) : nothing
    end
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      length(innerArgs) == 1 ? _tryEvalCondition(innerArgs[1], simCode, seen) : nothing
    end
    DAE.RELATION(e1, op, e2, _, _) => begin
      local v1 = tryEvalScalar(e1, simCode, seen)
      local v2 = tryEvalScalar(e2, simCode, seen)
      _compareScalarValues(v1, op, v2)
    end
    DAE.LBINARY(e1, DAE.AND(__), e2) => begin
      local r1 = _tryEvalCondition(e1, simCode, seen)
      r1 === false && return false
      local r2 = _tryEvalCondition(e2, simCode, seen)
      r2 === false && return false
      (r1 === true && r2 === true) ? true : nothing
    end
    DAE.LBINARY(e1, DAE.OR(__), e2) => begin
      local r1 = _tryEvalCondition(e1, simCode, seen)
      r1 === true && return true
      local r2 = _tryEvalCondition(e2, simCode, seen)
      r2 === true && return true
      (r1 === false && r2 === false) ? false : nothing
    end
    DAE.LUNARY(DAE.NOT(__), e1) => begin
      local r1 = _tryEvalCondition(e1, simCode, seen)
      r1 !== nothing ? !r1 : nothing
    end
    DAE.CAST(_, e1) => _tryEvalCondition(e1, simCode, seen)
    _ => nothing
  end
end

"""
  Try to evaluate a DAE expression to a numeric value.
  Returns Float64, or nothing if evaluation is not possible.
"""
Base.@nospecializeinfer function tryEvalNumeric(@nospecialize(exp::DAE.Exp))::Union{Float64, Nothing}
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

Base.@nospecializeinfer function tryEvalNumeric(@nospecialize(exp::DAE.Exp), simCode::SIM_CODE)::Union{Float64, Nothing}
  return _tryEvalNumeric(exp, simCode, OrderedSet{String}())
end

Base.@nospecializeinfer function tryEvalScalar(@nospecialize(exp::DAE.Exp), simCode::SIM_CODE)
  return tryEvalScalar(exp, simCode, OrderedSet{String}())
end

Base.@nospecializeinfer function tryEvalScalar(@nospecialize(exp::DAE.Exp), simCode::SIM_CODE, seen::OrderedSet{String})
  @match exp begin
    DAE.BCONST(v) => v
    DAE.SCONST(v) => v
    DAE.ICONST(v) => v
    DAE.RCONST(v) => v
    DAE.ENUM_LITERAL(_, index) => index
    DAE.CREF(__) => begin
      local bound = _boundParameterExpression(exp, simCode, seen)
      if bound === nothing
        nothing
      else
        local (name, bindExp) = bound
        #= Backtracking cycle guard on a shared set avoids copying `seen` per CREF. =#
        push!(seen, name)
        try
          tryEvalScalar(bindExp, simCode, seen)
        finally
          delete!(seen, name)
        end
      end
    end
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      length(innerArgs) == 1 ? tryEvalScalar(innerArgs[1], simCode, seen) : nothing
    end
    DAE.CALL(Absyn.IDENT("Integer"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        local v = tryEvalScalar(innerArgs[1], simCode, seen)
        v isa Number ? Int(v) : nothing
      else
        nothing
      end
    end
    DAE.CAST(_, e1) => tryEvalScalar(e1, simCode, seen)
    _ => begin
      local numeric = _tryEvalNumeric(exp, simCode, seen)
      numeric === nothing ? nothing : numeric
    end
  end
end

function _tryEvalNumeric(exp::DAE.Exp, simCode::SIM_CODE, seen::OrderedSet{String})::Union{Float64, Nothing}
  @match exp begin
    DAE.RCONST(val) => Float64(val)
    DAE.ICONST(val) => Float64(val)
    DAE.ENUM_LITERAL(_, index) => Float64(index)
    DAE.CREF(__) => begin
      local bound = _boundParameterExpression(exp, simCode, seen)
      if bound === nothing
        nothing
      else
        local (name, bindExp) = bound
        #= Backtracking cycle guard on a shared set avoids copying `seen` per CREF. =#
        push!(seen, name)
        try
          _tryEvalNumeric(bindExp, simCode, seen)
        finally
          delete!(seen, name)
        end
      end
    end
    DAE.UNARY(DAE.UMINUS(__), inner) => begin
      local v = _tryEvalNumeric(inner, simCode, seen)
      v !== nothing ? -v : nothing
    end
    DAE.UNARY(DAE.UMINUS_ARR(__), inner) => begin
      local v = _tryEvalNumeric(inner, simCode, seen)
      v !== nothing ? -v : nothing
    end
    DAE.CALL(Absyn.IDENT("abs"), lst, _) => begin
      local innerArgs = collect(lst)
      if length(innerArgs) == 1
        local v = _tryEvalNumeric(innerArgs[1], simCode, seen)
        v !== nothing ? abs(v) : nothing
      else
        nothing
      end
    end
    DAE.CALL(Absyn.IDENT("noEvent"), lst, _) => begin
      local innerArgs = collect(lst)
      length(innerArgs) == 1 ? _tryEvalNumeric(innerArgs[1], simCode, seen) : nothing
    end
    DAE.CALL(Absyn.IDENT("Integer"), lst, _) => begin
      local innerArgs = collect(lst)
      length(innerArgs) == 1 ? _tryEvalNumeric(innerArgs[1], simCode, seen) : nothing
    end
    DAE.BINARY(e1, DAE.ADD(__), e2) => begin
      local v1 = _tryEvalNumeric(e1, simCode, seen)
      local v2 = _tryEvalNumeric(e2, simCode, seen)
      (v1 !== nothing && v2 !== nothing) ? v1 + v2 : nothing
    end
    DAE.BINARY(e1, DAE.SUB(__), e2) => begin
      local v1 = _tryEvalNumeric(e1, simCode, seen)
      local v2 = _tryEvalNumeric(e2, simCode, seen)
      (v1 !== nothing && v2 !== nothing) ? v1 - v2 : nothing
    end
    DAE.BINARY(e1, DAE.MUL(__), e2) => begin
      local v1 = _tryEvalNumeric(e1, simCode, seen)
      local v2 = _tryEvalNumeric(e2, simCode, seen)
      (v1 !== nothing && v2 !== nothing) ? v1 * v2 : nothing
    end
    DAE.BINARY(e1, DAE.DIV(__), e2) => begin
      local v1 = _tryEvalNumeric(e1, simCode, seen)
      local v2 = _tryEvalNumeric(e2, simCode, seen)
      (v1 !== nothing && v2 !== nothing && v2 != 0.0) ? v1 / v2 : nothing
    end
    DAE.CAST(_, e1) => _tryEvalNumeric(e1, simCode, seen)
    _ => nothing
  end
end

function _boundParameterExpression(exp::DAE.Exp, simCode::SIM_CODE, seen::OrderedSet{String})
  local extracted = extractCrefName(exp)
  extracted === nothing && return nothing
  local name = extracted[1]
  name in seen && return nothing
  #= A tunable parameter has no compile-time value. =#
  isTunableParameter(name) && return nothing
  local entry = get(simCode.stringToSimVarHT, name, nothing)
  entry === nothing && return nothing
  local (_, simVar) = entry
  local bindExp = @match simVar.varKind begin
    PARAMETER(SOME(e)) => SimulationCode.toDAEExp(e)
    ARRAY_PARAMETER(_, SOME(e)) => SimulationCode.toDAEExp(e)
    STRING(SOME(e)) => SimulationCode.toDAEExp(e)
    _ => nothing
  end
  bindExp === nothing && return nothing
  return (name, bindExp)
end

function _compareScalarValues(v1, @nospecialize(op), v2)::Union{Bool, Nothing}
  if v1 === nothing || v2 === nothing
    return nothing
  end
  if v1 isa Number && v2 isa Number
    local n1 = Float64(v1)
    local n2 = Float64(v2)
    return @match op begin
      DAE.LESS(__) => n1 < n2
      DAE.LESSEQ(__) => n1 <= n2
      DAE.GREATER(__) => n1 > n2
      DAE.GREATEREQ(__) => n1 >= n2
      DAE.EQUAL(__) => n1 == n2
      DAE.NEQUAL(__) => n1 != n2
      _ => nothing
    end
  end
  return @match op begin
    DAE.EQUAL(__) => v1 == v2
    DAE.NEQUAL(__) => v1 != v2
    _ => nothing
  end
end
