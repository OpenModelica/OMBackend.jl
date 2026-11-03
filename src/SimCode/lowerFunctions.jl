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

#= The functions of the simulation code: OMJLIR's functions with their record inputs and
   outputs as the records' fields, `smooth()` removed. =#

"A function with its records flattened, and its number of inputs as declared."
struct FlattenedFunction
  fn::IR.ModelicaFunction
  nInputs::Int
end

"The fields of the flattened records by the record variables' names: (field, type)."
const RecordFields = Dict{Symbol, Vector{Tuple{Symbol, IR.ModelicaType}}}

"""
    simFunctions(functions)

The functions in the simulation code, in their order (an inlined function's derivative is
OMJLIR's, `withInlineDerivatives`).
"""
simFunctions(functions::Vector{IR.ModelicaFunction})::Vector{ModelicaFunction} =
  ModelicaFunction[simFunction(flattenRecordParameters(f)) for f in functions]

"""
    simFunction(f)

A function (its records flattened) in the simulation code: its variables, its statements
(without `smooth()`) or its external call, its derivative annotation of order 1.
"""
function simFunction(f::FlattenedFunction)::ModelicaFunction
  local fn = f.fn
  local names = functionNames(fn)
  local variables = vs -> FunctionVariable[simFunctionVariable(names, v) for v in vs]
  local d = isempty(fn.derivatives) ? nothing : first(fn.derivatives)
  local derivative = d === nothing ? nothing : FunctionDerivative(String(codeName(d.derivative)), d.excludedInputs, f.nInputs)
  local name = String(codeName(fn.name))
  if fn.external === nothing
    local body = Statement[removeSmooth(s) for s in simStatements(names, fn.body)]
    return MODELICA_FUNCTION(name, variables(fn.inputs), variables(fn.outputs), variables(fn.locals), body, derivative)
  end
  local x = fn.external
  local call = ExternalCall(x.name, x.language, simExps(names, x.arguments),
                            x.output === nothing ? nothing : simCrefExp(names, x.output, IR.T_UNKNOWN()), x.annotation)
  return EXTERNAL_MODELICA_FUNCTION(name, variables(fn.inputs), variables(fn.outputs), variables(fn.locals), call, derivative)
end

"The names a function's references are known by: its variables' (and `time`, `pi`, `e`)."
function functionNames(fn::IR.ModelicaFunction)::Names
  local known = Set{String}(["time", "pi", "e"])
  foreach(v -> push!(known, derivedKey(v.cref)), Iterators.flatten((fn.inputs, fn.outputs, fn.locals)))
  return Names(false, known, Dict{Symbol, IR.ComponentRef}(), Set{String}())
end

function simFunctionVariable(names::Names, v::IR.FunctionVariable)::FunctionVariable
  local dimensions = Exp[]
  for d in IR.dimensions(v.ty)
    push!(dimensions, d isa IR.DIM_EXP ? simExp(names, d.exp) :
                      d isa IR.DIM_BOOLEAN ? unsupported("a Boolean array dimension of a function variable", v.name) :
                      ICONST(something(IR.dimensionSize(d), 0)))
  end
  return FunctionVariable(derivedKey(v.cref), simType(v.ty), dimensions, v.binding === nothing ? nothing : simExp(names, v.binding),
                          Pair{String, Exp}[String(n) => simExp(names, e) for (n, e) in v.fieldBindings], v.cref)
end

"`s` without `smooth(p, e)` (as `e`, its own smooth() kept), in an assignment, a call, a condition, a range, a loop's body (the previous backend's)."
function removeSmooth(s::Statement)::Statement
  local x = e -> withoutSmooth(e)
  s isa ASSIGN && return ASSIGN(x(s.left), x(s.right), s.source)
  s isa IF && return IF(Branch[Branch(i == 1 ? x(b.condition) : b.condition, i == 1 ? Statement[removeSmooth(t) for t in b.body] : b.body)
                               for (i, b) in enumerate(s.branches)], s.elseBody, s.source)
  s isa FOR && return FOR(Iterator(s.iterator.name, x(s.iterator.range)), Statement[removeSmooth(t) for t in s.body], s.source)
  s isa WHILE && return WHILE(x(s.condition), Statement[removeSmooth(t) for t in s.body], s.source)
  s isa NORETCALL && return NORETCALL(x(s.exp), s.source)
  return s
end

withoutSmooth(e::Exp)::Exp =
  e isa CALL && e.kind === CALL_BUILTIN && e.name === :smooth && length(e.args) == 2 ? e.args[2] : mapChildren(withoutSmooth, e)

#= ==================================================================================== =#
#= Record inputs and outputs as their fields                                            =#
#= ==================================================================================== =#

"""
    flattenRecordParameters(f)

`f` with its record inputs and outputs as their fields (`r` as `r.a`, `r.b`; an array of
records `u` as one array per field, `u.re`, `u.im`, of its dimensions and the field's), and
its local arrays of records with scalar fields read and written by field (`o[1].re` as
`o.re[1]`: the code generation declares one array per field). A field's variable and
references keep the record's and the field's parts; its identifier (`r_a`, `o_re`) is derived
(`derivedKey`). Its statements follow: a field
read by its flat name, a record argument as its fields, a record assignment field by field
(of a call's result by indexing it once), a tuple's record target as its fields. A record
constructor without statements gets an assignment of each output field from the input of its
name. An external function is left as it is.
"""
function flattenRecordParameters(fn::IR.ModelicaFunction)::FlattenedFunction
  local nInputs = length(fn.inputs)
  fn.external === nothing || return FlattenedFunction(fn, nInputs)
  local fields = RecordFields()
  local inputs = reduce(vcat, (flattenRecordVariable(v, fields) for v in fn.inputs); init = IR.FunctionVariable[])
  local outputs = reduce(vcat, (flattenRecordVariable(v, fields; output = true) for v in fn.outputs);
                         init = IR.FunctionVariable[])
  for l in fn.locals
    local lf = localRecordArrayFields(l)
    lf === nothing || (fields[l.name] = lf)
  end
  local body = flattenedStatements(fn.body, fields)
  if isempty(body) && !isempty(fields)
    local flatInputs = Set{Symbol}(v.name for v in inputs)
    for o in fn.outputs, (field, ty) in get(fields, o.name, Tuple{Symbol, IR.ModelicaType}[])
      field in flatInputs || continue
      push!(body, IR.ASSIGNMENT(IR.CREF(IR.withField(o.cref, field, ty), ty),
                                IR.CREF(IR.ComponentRef(field, ty), ty), IR.NO_SOURCE))
    end
  end
  local locals = isempty(fields) ? fn.locals : IR.FunctionVariable[flattenedVariable(l, fields) for l in fn.locals]
  return FlattenedFunction(IR.ModelicaFunction(fn.name, inputs, outputs, locals, body, fn.external, fn.derivatives,
                                               fn.impure, fn.inline, fn.source), nInputs)
end

recordTypeOf(ty::IR.ModelicaType)::Union{Nothing, IR.T_RECORD} =
  ty isa IR.T_RECORD ? ty : ty isa IR.T_ARRAY && ty.elementType isa IR.T_RECORD ? ty.elementType : nothing

"""
    flattenRecordVariable(v, fields; output)

A record variable as its fields' variables (`r.a`, ...; of a record array of its dimensions
and the field's), its fields recorded in `fields`; another variable with its dimensions and
binding flattened. An output field starts at its value of the record's binding (a record
value's or a constructor call's element, else the field of the value), else at its field
binding (MLS 12.4.4).
"""
function flattenRecordVariable(v::IR.FunctionVariable, fields::RecordFields; output::Bool = false)::Vector{IR.FunctionVariable}
  local rec = recordTypeOf(v.ty)
  rec === nothing && return IR.FunctionVariable[flattenedVariable(v, fields)]
  local recordDims = IR.Dimension[flattenedDimension(d, fields) for d in IR.dimensions(v.ty)]
  local out = IR.FunctionVariable[]
  for (k, field) in enumerate(rec.fields)
    local fieldDims = IR.dimensions(field.ty)
    local ty = isempty(recordDims) ? field.ty : IR.T_ARRAY(IR.elementType(field.ty), vcat(recordDims, fieldDims))
    local binding = output ? outputFieldBinding(v, k, field, fields) : nothing
    push!(out, IR.FunctionVariable(IR.withField(v.cref, field.name, ty), ty, binding, Pair{Symbol, IR.Expression}[]))
  end
  fields[v.name] = Tuple{Symbol, IR.ModelicaType}[(f.name, f.ty) for f in rec.fields]
  return out
end

function outputFieldBinding(v::IR.FunctionVariable, k::Int, field::IR.Field, fields::RecordFields)::Union{Nothing, IR.Expression}
  if v.binding !== nothing
    local e = flattenedExpression(v.binding, fields)
    local parts = e isa IR.RECORD ? e.elements : e isa IR.CALL && isConstructorCall(e) ? e.arguments : nothing
    (parts !== nothing && length(parts) == length(recordTypeOf(v.ty).fields)) && return parts[k]
    return IR.RECORD_ELEMENT(e, k, field.name, field.ty)
  end
  local b = findfirst(p -> p.first === field.name, v.fieldBindings)
  return b === nothing ? nothing : flattenedExpression(v.fieldBindings[b].second, fields)
end

"A call of a record's constructor (its arguments are the fields)."
isConstructorCall(e::IR.CALL)::Bool = e.ty isa IR.T_RECORD && e.ty.name === e.name

"A variable with the records in its dimensions and binding flattened (`m = size(u, 1)`)."
flattenedVariable(v::IR.FunctionVariable, fields::RecordFields)::IR.FunctionVariable =
  IR.FunctionVariable(v.cref, flattenedType(v.ty, fields), v.binding === nothing ? nothing : flattenedExpression(v.binding, fields),
                      v.fieldBindings)

flattenedType(ty::IR.T_ARRAY, fields::RecordFields)::IR.ModelicaType =
  IR.T_ARRAY(ty.elementType, IR.Dimension[flattenedDimension(d, fields) for d in ty.dimensions])
flattenedType(ty::IR.ModelicaType, ::RecordFields)::IR.ModelicaType = ty

flattenedDimension(d::IR.DIM_EXP, fields::RecordFields)::IR.Dimension = IR.DIM_EXP(flattenedExpression(d.exp, fields))
flattenedDimension(d::IR.Dimension, ::RecordFields)::IR.Dimension = d

"The fields (scalar ones) of a local array of records; `nothing` for another local."
function localRecordArrayFields(v::IR.FunctionVariable)::Union{Nothing, Vector{Tuple{Symbol, IR.ModelicaType}}}
  (v.ty isa IR.T_ARRAY && v.ty.elementType isa IR.T_RECORD) || return nothing
  local fs = Tuple{Symbol, IR.ModelicaType}[(f.name, f.ty) for f in v.ty.elementType.fields]
  (isempty(fs) || any(f -> f[2] isa Union{IR.T_ARRAY, IR.T_RECORD}, fs)) && return nothing
  return fs
end

"""
    flattenedExpression(e, fields)

`e` with the flattened records' fields read as their variables (`u[k].re` as `u.re[k]`: the
element's subscripts on the field's part, first, then the field's), the size of a record array as its
first field's (`size(u, 1)` as `size(u.re, 1)`), and the record arguments of calls as their fields
(`expandedArguments`). In a function without flattened records only a call with a
record-valued argument is changed (the callee takes the record's fields).
"""
function flattenedExpression(e::IR.Expression, fields::RecordFields)::IR.Expression
  isempty(fields) && !hasRecordValuedArgument(e) && return e
  return flattenRecordReferences(e, fields)
end

function flattenRecordReferences(e::IR.Expression, fields::RecordFields)::IR.Expression
  local x = a -> flattenRecordReferences(a, fields)
  if e isa IR.CREF && length(e.cref.parts) >= 2 && haskey(fields, e.cref.parts[1].name)
    return IR.mapChildren(x, IR.CREF(fieldElementCref(e.cref), e.ty))
  elseif e isa IR.SIZE && e.exp isa IR.CREF && length(e.exp.cref.parts) == 1 && haskey(fields, e.exp.cref.name)
    local fs = fields[e.exp.cref.name]
    if !isempty(fs) && (e.dimension !== nothing || !(fs[1][2] isa IR.T_ARRAY))
      local (field, ty) = fs[1]
      local cref = fieldOfElement(e.exp.cref.parts[1], field, ty)
      return IR.mapChildren(x, IR.SIZE(IR.CREF(cref, ty), e.dimension))
    end
  elseif e isa IR.CALL
    return IR.CALL(e.name, expandedArguments(e, fields), e.ty, e.attributes)
  end
  return IR.mapChildren(x, e)
end

"""
    expandedArguments(call, fields)

A call's arguments with a flattened record as its fields, and a call returning an array of
records as its fields by TUPLE_ELEMENT (one array per field; a built-in takes the records
whole). A call returning a record of scalar or array fields stays one argument: its value is
the record's fields, which the code generation spreads, evaluated once (by TUPLE_ELEMENT, a
call nested in a call's argument ran twice at each level: Buildings' multipoleFmk, ten levels
of Complex operators).
"""
function expandedArguments(call::IR.CALL, fields::RecordFields)::Vector{IR.Expression}
  local out = IR.Expression[]
  for a in call.arguments
    local expanded = recordFieldReferences(a, fields)
    if expanded !== nothing
      append!(out, expanded)
      continue
    end
    local e = flattenedExpression(a, fields)
    local types = call.attributes.builtin || spreadsAsArgument(e) ? nothing : recordValueFieldTypes(e)
    types === nothing ? push!(out, e) : append!(out, IR.Expression[IR.TUPLE_ELEMENT(e, k, t) for (k, t) in enumerate(types)])
  end
  return out
end

"A call returning a record without record fields: an argument its fields as a tuple, spread."
spreadsAsArgument(e::IR.Expression)::Bool =
  e isa IR.CALL && e.ty isa IR.T_RECORD && !any(f -> recordTypeOf(f.ty) !== nothing, e.ty.fields)

"A flattened record's reference (an element of a record array keeps its subscripts) as its fields' references; `nothing` for another expression."
function recordFieldReferences(e::IR.Expression, fields::RecordFields)::Union{Nothing, Vector{IR.Expression}}
  (e isa IR.CREF && length(e.cref.parts) == 1 && haskey(fields, e.cref.parts[1].name)) || return nothing
  local part = e.cref.parts[1]
  return IR.Expression[IR.CREF(fieldOfElement(part, f, t), t) for (f, t) in fields[part.name]]
end

"""
    fieldOfElement(record, field, ty)

The field `field` (of type `ty`) of the record reference of the one part `record`, as its flattened variable reads it: an
element of a record array's subscripts on the field's part (`u[k]`'s `re` as `u.re[k]`).
"""
fieldOfElement(record::IR.CrefPart, field::Symbol, ty::IR.ModelicaType)::IR.ComponentRef =
  IR.ComponentRef(IR.CrefPart[IR.CrefPart(record.name, IR.Subscript[], record.ty), IR.CrefPart(field, record.subscripts, ty)])

"""
    fieldElementCref(cref)

A reference of a flattened record's field (`r.a`, `u[k].re`, `u[k].x[2]`) as its variable reads it: the record part's
subscripts after the field's (`u.re[k]`, `u.x[k,2]`), the parts between kept.
"""
function fieldElementCref(cref::IR.ComponentRef)::IR.ComponentRef
  local (first::IR.CrefPart, last::IR.CrefPart) = (cref.parts[1], cref.parts[end])
  return IR.ComponentRef(IR.CrefPart[IR.CrefPart(first.name, IR.Subscript[], first.ty); cref.parts[2:end - 1];
                                     IR.CrefPart(last.name, vcat(first.subscripts, last.subscripts), last.ty)])
end

"The field types of a record-valued call (of an array of records, one array per field); `nothing` for another expression."
function recordValueFieldTypes(e::IR.Expression)::Union{Nothing, Vector{IR.ModelicaType}}
  e isa IR.CALL || return nothing
  e.ty isa IR.T_RECORD && return IR.ModelicaType[f.ty for f in e.ty.fields]
  (e.ty isa IR.T_ARRAY && e.ty.elementType isa IR.T_RECORD) &&
    return IR.ModelicaType[IR.T_ARRAY(f.ty, e.ty.dimensions) for f in e.ty.elementType.fields]
  return nothing
end

hasRecordValuedArgument(e::IR.Expression)::Bool =
  IR.anyExp(x -> x isa IR.CALL && !x.attributes.builtin && any(a -> recordValueFieldTypes(a) !== nothing, x.arguments), e)

"The statements with the flattened records (`flattenedStatement`)."
function flattenedStatements(body::Vector{IR.Statement}, fields::RecordFields)::Vector{IR.Statement}
  local out = IR.Statement[]
  for s in body
    append!(out, flattenedStatement(s, fields))
  end
  return out
end

"""
    flattenedStatement(s, fields)

A statement with the flattened records: an assignment to a flattened record field by field
(from a record reference's or a record value's fields; of another value once, then each field
by indexing the record: `r := f(x); r_a := r[1]; ...`), to an element of a record array
through a temporary record (`recordElementAssignment`); a tuple assignment's record targets as
their fields.
"""
function flattenedStatement(s::IR.ASSIGNMENT, fields::RecordFields)::Vector{IR.Statement}
  local x = e -> flattenedExpression(e, fields)
  s.lhs isa IR.TUPLE && return IR.Statement[flattenedTupleAssignment(s, fields)]
  if IR.typeOf(s.lhs) isa IR.T_ARRAY
    #= an array (a whole record array: one statement, scattered by the code generation) =#
    local rhs = x(s.rhs)
    (recordFieldReferences(s.lhs, fields) !== nothing && !(rhs isa IR.CALL) && recordFieldReferences(s.rhs, fields) === nothing) &&
      unsupported("an assignment of this value to a record array in a function", IR.toString(s))
    return IR.Statement[IR.ASSIGNMENT(x(s.lhs), rhs, s.source)]
  end
  local lhsPart = s.lhs isa IR.CREF && length(s.lhs.cref.parts) == 1 ? s.lhs.cref.parts[1] : nothing
  (lhsPart === nothing || !haskey(fields, lhsPart.name)) && return IR.Statement[IR.ASSIGNMENT(x(s.lhs), x(s.rhs), s.source)]
  local fs = fields[lhsPart.name]
  isempty(lhsPart.subscripts) || return recordElementAssignment(s, lhsPart, fs, fields)
  local fieldLhs = (f, t) -> IR.CREF(IR.withField(s.lhs.cref, f, t), t)
  local rhsFields = recordFieldReferences(s.rhs, fields)
  rhsFields === nothing ||
    return IR.Statement[IR.ASSIGNMENT(fieldLhs(f, t), rhsFields[k], s.source) for (k, (f, t)) in enumerate(fs)]
  s.rhs isa IR.RECORD &&
    return IR.Statement[IR.ASSIGNMENT(fieldLhs(f, t), x(s.rhs.elements[k]), s.source) for (k, (f, t)) in enumerate(fs)]
  local out = IR.Statement[IR.ASSIGNMENT(s.lhs, x(s.rhs), s.source)]
  for (k, (f, t)) in enumerate(fs)
    push!(out, IR.ASSIGNMENT(fieldLhs(f, t), IR.SUBSCRIPTED(s.lhs, IR.Subscript[IR.INDEX(IR.INTEGER(k))], t), s.source))
  end
  return out
end

"A tuple assignment with its record targets as their fields (the callee returns them so), the tuple's type following."
function flattenedTupleAssignment(s::IR.ASSIGNMENT, fields::RecordFields)::IR.ASSIGNMENT
  local targets = IR.Expression[]
  local types = IR.ModelicaType[]
  local outputTypes = s.rhs isa IR.CALL && s.rhs.ty isa IR.T_TUPLE ? s.rhs.ty.types : nothing
  for (k, t) in enumerate(s.lhs.elements)
    local fs = recordFieldReferences(t, fields)
    if fs === nothing
      push!(targets, flattenedExpression(t, fields))
      outputTypes === nothing || push!(types, outputTypes[k])
    else
      append!(targets, fs)
      append!(types, (IR.typeOf(f) for f in fs))
    end
  end
  return IR.ASSIGNMENT(IR.TUPLE(targets, outputTypes === nothing ? s.lhs.ty : IR.T_TUPLE(types)),
                       flattenedExpression(s.rhs, fields), s.source)
end

flattenedStatement(s::IR.IF_STATEMENT, fields::RecordFields)::Vector{IR.Statement} =
  IR.Statement[IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(flattenedExpression(b.condition, fields),
                                                                     flattenedStatements(b.body, fields)) for b in s.branches],
                               flattenedStatements(s.elseBody, fields), s.source)]
flattenedStatement(s::IR.FOR_STATEMENT, fields::RecordFields)::Vector{IR.Statement} =
  IR.Statement[IR.FOR_STATEMENT(IR.Iterator(s.iterator.name, flattenedExpression(s.iterator.range, fields)),
                                flattenedStatements(s.body, fields), s.source)]
flattenedStatement(s::IR.WHILE_STATEMENT, fields::RecordFields)::Vector{IR.Statement} =
  IR.Statement[IR.WHILE_STATEMENT(flattenedExpression(s.condition, fields), flattenedStatements(s.body, fields), s.source)]
flattenedStatement(s::IR.NORETCALL, fields::RecordFields)::Vector{IR.Statement} =
  IR.Statement[IR.NORETCALL(flattenedExpression(s.call, fields), s.source)]
flattenedStatement(s::IR.ASSERT, fields::RecordFields)::Vector{IR.Statement} =
  IR.Statement[IR.ASSERT(flattenedExpression(s.condition, fields), flattenedExpression(s.message, fields),
                         flattenedExpression(s.level, fields), s.source)]
flattenedStatement(s::IR.Statement, ::RecordFields)::Vector{IR.Statement} = IR.Statement[s]

"""
    recordElementAssignment(s, lhs, fs, fields)

`v[i] := value` of a record array `v`: the value into a temporary record first (a call
evaluated once, scattered by the code generation), then into the fields' elements, so a value
reading `v[i]` (a swap of fields) sees the old element.
"""
function recordElementAssignment(s::IR.ASSIGNMENT, lhs::IR.CrefPart, fs::Vector{Tuple{Symbol, IR.ModelicaType}},
                                 fields::RecordFields)::Vector{IR.Statement}
  local tmp = Symbol("__omjl_", lhs.name)
  local tmpRecord::IR.ComponentRef = IR.ComponentRef(tmp, IR.typeOf(s.lhs))
  local tmpField = (f, t) -> IR.CREF(IR.withField(tmpRecord, f, t), t)
  local out = IR.Statement[]
  local rhsFields = recordFieldReferences(s.rhs, fields)
  if rhsFields !== nothing
    for (k, (f, t)) in enumerate(fs)
      push!(out, IR.ASSIGNMENT(tmpField(f, t), rhsFields[k], s.source))
    end
  elseif s.rhs isa IR.RECORD
    for (k, (f, t)) in enumerate(fs)
      push!(out, IR.ASSIGNMENT(tmpField(f, t), flattenedExpression(s.rhs.elements[k], fields), s.source))
    end
  else
    local ty = IR.typeOf(s.lhs)
    push!(out, IR.ASSIGNMENT(IR.CREF(IR.ComponentRef(tmp, ty), ty), flattenedExpression(s.rhs, fields), s.source))
  end
  for (f, t) in fs
    local target = IR.CREF(fieldOfElement(lhs, f, t), t)
    push!(out, IR.ASSIGNMENT(target, tmpField(f, t), s.source))
  end
  return out
end
