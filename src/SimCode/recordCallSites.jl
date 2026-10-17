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

#= The record arguments of calls as their fields: the model's functions take a record's
   fields, not the record (their record inputs and outputs are flattened). =#

"""
    flattenRecordCallSites(simCode)

The calls of the simulation code with their record arguments as the records' fields
(`expandRecordArguments`), in the residuals, the when-equations, the if-equations' branches,
the initial algorithms, the asserts, the initial equations and the parameters' bindings; a
tuple residual `(a, b) - f(...)` as one residual per target, `a - f(...)[1]`, ... (the code
generation has no tuple residuals). Of the structural modes too.
"""
function flattenRecordCallSites(simCode::SIM_CODE)::SIM_CODE
  local ht = simCode.stringToSimVarHT
  local x = e -> expandRecordArguments(e, ht)
  local residuals = RESIDUAL_EQUATION[]
  for eq in simCode.residualEquations
    local e = x(eq.exp)
    local split = splitTupleResidual(e)
    split === nothing ? push!(residuals, RESIDUAL_EQUATION(e, eq.source, eq.attr)) :
                        append!(residuals, RESIDUAL_EQUATION[RESIDUAL_EQUATION(s, eq.source, eq.attr) for s in split])
  end
  local whens = WHEN_EQUATION[WHEN_EQUATION(whenRecordArguments(w.whenEquation, x), w.source, w.attr) for w in simCode.whenEquations]
  local ifs = IF_EQUATION[IF_EQUATION(BRANCH[branchRecordArguments(b, x) for b in ifEq.branches]) for ifEq in simCode.ifEquations]
  local initialAlgorithms = INITIAL_ALGORITHM[INITIAL_ALGORITHM(Statement[mapValues(x, s) for s in a.statements], a.section)
                                              for a in simCode.initialAlgorithms]
  local asserts = ASSERT_EQUATION[ASSERT_EQUATION(x(a.condition), x(a.message), a.level, a.source) for a in simCode.asserts]
  local initialEquations = Equation[initialRecordArguments(eq, x) for eq in simCode.initialEquations]
  for (name, (index, sv)) in ht
    local k = sv.varKind
    if k isa PARAMETER && k.bindExp !== nothing
      ht[name] = (index, withKind(sv, PARAMETER(x(k.bindExp))))
    elseif k isa ARRAY_PARAMETER && k.bindExp !== nothing
      ht[name] = (index, withKind(sv, ARRAY_PARAMETER(k.dimensions, x(k.bindExp))))
    end
  end
  return update(simCode; residualEquations = residuals, whenEquations = whens, ifEquations = ifs, initialAlgorithms, asserts,
         initialEquations, subModels = SIM_CODE[flattenRecordCallSites(m) for m in simCode.subModels])
end

"A when-clause with its condition's and its statements' record arguments flattened (an assignment's value, a reinit's, a call, an assert's condition)."
function whenRecordArguments(w::WHEN_STMTS, x::F)::WHEN_STMTS where {F}
  local body = Statement[s isa ASSIGN ? ASSIGN(s.left, x(s.right), s.source) :
                         s isa REINIT ? REINIT(s.stateVar, x(s.value), s.source) :
                         s isa NORETCALL ? NORETCALL(x(s.exp), s.source) :
                         s isa ASSERT ? ASSERT(x(s.condition), s.message, s.level, s.source) : s for s in w.body]
  return WHEN_STMTS(x(w.condition), body, w.elsewhenPart === nothing ? nothing : whenRecordArguments(w.elsewhenPart, x))
end

function branchRecordArguments(b::BRANCH, x::F)::BRANCH where {F}
  return BRANCH(b.identifier == ELSE_BRANCH ? b.condition : x(b.condition),
                RESIDUAL_EQUATION[RESIDUAL_EQUATION(x(r.exp), r.source, r.attr) for r in b.residualEquations],
                b.identifier, b.targets)
end

function initialRecordArguments(eq::Equation, x::F)::Equation where {F}
  eq isa RESIDUAL_EQUATION && return RESIDUAL_EQUATION(x(eq.exp), eq.source, eq.attr)
  eq isa EQUATION && return EQUATION(x(eq.lhs), x(eq.rhs), eq.source, eq.attr)
  return eq
end

"""
    mapValues(g, s)

The statement `s` with `g` applied to the expressions it evaluates (values, conditions,
ranges, an assert's arguments), not to its targets; `g` rewrites a whole expression.
"""
mapValues(g::G, s::ASSIGN) where {G} = ASSIGN(s.left, g(s.right), s.source)
mapValues(g::G, s::IF) where {G} =
  IF(Branch[Branch(g(b.condition), Statement[mapValues(g, t) for t in b.body]) for b in s.branches],
     Statement[mapValues(g, t) for t in s.elseBody], s.source)
mapValues(g::G, s::FOR) where {G} = FOR(Iterator(s.iterator.name, g(s.iterator.range)), Statement[mapValues(g, t) for t in s.body], s.source)
mapValues(g::G, s::WHILE) where {G} = WHILE(g(s.condition), Statement[mapValues(g, t) for t in s.body], s.source)
mapValues(g::G, s::WHEN) where {G} = WHEN(Branch[Branch(g(b.condition), Statement[mapValues(g, t) for t in b.body]) for b in s.branches], s.source)
mapValues(g::G, s::ASSERT) where {G} = ASSERT(g(s.condition), g(s.message), g(s.level), s.source)
mapValues(g::G, s::TERMINATE) where {G} = TERMINATE(g(s.message), s.source)
mapValues(g::G, s::REINIT) where {G} = REINIT(s.stateVar, g(s.value), s.source)
mapValues(g::G, s::NORETCALL) where {G} = NORETCALL(g(s.exp), s.source)
mapValues(g::G, s::RECOMPILATION) where {G} = RECOMPILATION(s.componentToChange, g(s.newValue))
mapValues(g::G, s::Union{RETURN, BREAK, AGENTIC_RECOMPILATION}) where {G} = s

"""
    splitTupleResidual(e)

`(a, b, ...) - f(...)` as `a - f(...)[1]`, `b - f(...)[2]`, ... (a discarded output left
out); `nothing` for another shape, or a target that is an array or a record.
"""
function splitTupleResidual(e::Exp)::Union{Nothing, Vector{Exp}}
  (e isa BINARY && e.op === OP_SUB && e.exp1 isa TUPLE && e.exp2 isa CALL) || return nothing
  local call = e.exp2
  (call.kind !== CALL_IMPURE_FUNCTION && call.ty isa TYPE_TUPLE) || return nothing
  local out = Exp[]
  for (k, target) in enumerate(e.exp1.elements)
    target isa WILD && continue
    target isa EXP_CREF || return nothing
    target.ty isa Union{TYPE_ARRAY, TYPE_COMPLEX} && return nothing
    push!(out, BINARY(target, OP_SUB, TSUB(call, k, target.ty)))
  end
  return out
end

"""
    expandRecordArguments(e, ht)

`e` with the record arguments of its calls as the records' fields, as the model's functions
take them: a record reference `r` as `r_f1, r_f2, ...` (an array field as the array of its
elements), a record value as its elements, a record-valued call as its fields by TSUB (the
call evaluated once per field), a record-valued if-expression field by field, an array of
records (a record array reference of a known size) as one array per field (a function's
record array input is one array per field). The records of a tuple's elements too (a tuple
equation's record target is its fields, as the function returns them). Builtins take an array
of records whole.
"""
function expandRecordArguments(e::Exp, ht::HT)::Exp
  local x = a -> expandRecordArguments(a, ht)
  if e isa CALL
    local args = Exp[]
    for arg in e.args
      local fieldArrays = e.kind === CALL_BUILTIN ? nothing : splitRecordArrayArgument(arg, ht)
      if fieldArrays !== nothing
        append!(args, fieldArrays)
      elseif arg isa IFEXP && isRecordShaped(arg) && (local f = recordIfFields(arg, ht); f !== nothing)
        append!(args, f)
      elseif isRecordReference(arg)
        append!(args, fieldReferences(arg, ht))
      elseif arg isa RECORD
        append!(args, Exp[x(a) for a in arg.exps])
      else
        local expanded = x(arg)
        local fieldTypes = recordValueFieldTypes(expanded)
        fieldTypes === nothing ? push!(args, expanded) : append!(args, Exp[TSUB(expanded, k, ty) for (k, ty) in enumerate(fieldTypes)])
      end
    end
    return CALL(e.name, args, e.ty, e.kind)
  elseif e isa TUPLE
    any(isRecordReference, e.elements) || return e
    return TUPLE(reduce(vcat, (isRecordReference(el) ? fieldReferences(el, ht) : Exp[el] for el in e.elements)))
  end
  #= every other expression's subexpressions (a call under a TSUB, an RSUB, a subscript, a
     reduction's body) =#
  return mapChildren(x, e)
end

isRecordType(ty::SType)::Bool = ty isa TYPE_COMPLEX && ty.isRecord

"A reference of a record type."
isRecordReference(e::Exp)::Bool = e isa EXP_CREF && isRecordType(e.ty)

"The references of a record reference's fields, `r_f` (an array field the array of its elements)."
function fieldReferences(e::EXP_CREF, ht::HT)::Vector{Exp}
  local base = crefName(e.cref)
  return Exp[fieldArgument(string(base, "_", n), t, ht) for (n, t) in zip(e.ty.fieldNames, e.ty.fieldTypes)]
end

"""
    fieldArgument(name, ty, ht)

A record field `name` as an argument: a scalar field's reference; an array field of one or
two known dimensions as the array (of rows) of its elements' references; another the whole
reference.
"""
function fieldArgument(name::String, ty::SType, ht::HT)::Exp
  if ty isa TYPE_ARRAY && all(d -> d >= 0, ty.dims) && length(ty.dims) in (1, 2)
    if length(ty.dims) == 1
      return ARRAY_EXP(ty, true, Exp[EXP_CREF(elementCref(name, [i], ht), ty.elementType) for i in 1:ty.dims[1]])
    end
    local rowTy = TYPE_ARRAY(ty.elementType, Int[ty.dims[2]])
    return ARRAY_EXP(ty, false, Exp[ARRAY_EXP(rowTy, true, Exp[EXP_CREF(elementCref(name, [i, j], ht), ty.elementType)
                                                                for j in 1:ty.dims[2]]) for i in 1:ty.dims[1]])
  end
  return EXP_CREF(SimCref(Symbol(name)), ty)
end

"The reference of the element `index` of the array `name`: a variable's name, else the array's with the subscripts."
function elementCref(name::String, index::Vector{Int}, ht::HT)::SimCref
  local full = string(name, join(("[$(i)]" for i in index)))
  return haskey(ht, full) ? SimCref(Symbol(full)) : SimCref(Symbol(name), index)
end

"An if-expression between record references or record-valued calls."
isRecordShaped(e::Exp)::Bool =
  e isa IFEXP ? isRecordShaped(e.thenExp) && isRecordShaped(e.elseExp) :
  isRecordReference(e) || (e isa CALL && isRecordType(e.ty))

"A record-valued if-expression field by field (`if c then a_f else b_f`); `nothing` unless both branches have the same fields."
function recordIfFields(e::IFEXP, ht::HT)::Union{Nothing, Vector{Exp}}
  local a = recordFields(e.thenExp, ht)
  a === nothing && return nothing
  local b = recordFields(e.elseExp, ht)
  (b === nothing || length(a) != length(b)) && return nothing
  local c = expandRecordArguments(e.cond, ht)
  return Exp[IFEXP(c, a[k], b[k]) for k in eachindex(a)]
end

function recordFields(e::Exp, ht::HT)::Union{Nothing, Vector{Exp}}
  isRecordReference(e) && return fieldReferences(e, ht)
  e isa IFEXP && return recordIfFields(e, ht)
  local expanded = expandRecordArguments(e, ht)
  local fieldTypes = recordValueFieldTypes(expanded)
  return fieldTypes === nothing ? nothing : Exp[TSUB(expanded, k, ty) for (k, ty) in enumerate(fieldTypes)]
end

"""
    recordValueFieldTypes(e)

The types of the fields of a record-valued call, or of an array of records (one array per
field); `nothing` for another expression.
"""
function recordValueFieldTypes(e::Exp)::Union{Nothing, Vector{SType}}
  e isa CALL || return nothing
  isRecordType(e.ty) && return e.ty.fieldTypes
  (e.ty isa TYPE_ARRAY && isRecordType(e.ty.elementType)) && return SType[TYPE_ARRAY(t, e.ty.dims) for t in e.ty.elementType.fieldTypes]
  return nothing
end

"""
    splitRecordArrayArgument(arg, ht)

An array of records as one array per field (`{u[1], u[2]}` as `{u[1]_re, u[2]_re}`,
`{u[1]_im, u[2]_im}`), a record-valued call element's field by TSUB; a reference to a whole
record array of one known dimension the same way. `nothing` for another argument, or one with
an array-valued field (scalarized per element).
"""
function splitRecordArrayArgument(arg::Exp, ht::HT)::Union{Nothing, Vector{Exp}}
  local elements = if arg isa ARRAY_EXP
    arg.elements
  elseif arg isa EXP_CREF && arg.ty isa TYPE_ARRAY && isRecordType(arg.ty.elementType) && length(arg.ty.dims) == 1 &&
         arg.ty.dims[1] >= 0 && !occursin('[', crefName(arg.cref))
    Exp[EXP_CREF(SimCref(arg.cref.sym, [i]), arg.ty.elementType) for i in 1:arg.ty.dims[1]]
  else
    return nothing
  end
  isempty(elements) && return nothing
  local fieldsOf = el -> isRecordReference(el) || (el isa RECORD && isRecordType(el.ty)) ? (el.ty.fieldNames, el.ty.fieldTypes) :
                         (el isa CALL && isRecordType(el.ty)) ? (el.ty.fieldNames, el.ty.fieldTypes) : nothing
  local fields = fieldsOf(first(elements))
  (fields === nothing || any(el -> fieldsOf(el) === nothing, elements)) && return nothing
  local (fieldNames, fieldTypes) = fields
  any(t -> t isa TYPE_ARRAY, fieldTypes) && return nothing
  return Exp[ARRAY_EXP(TYPE_ARRAY(t, Int[length(elements)]), true, Exp[recordElementField(el, k, n, t, ht) for el in elements])
             for (k, (n, t)) in enumerate(zip(fieldNames, fieldTypes))]
end

"Field `k` (`name`) of a record element of an array argument: a reference's field's name, a record value's element, a call's field by TSUB."
function recordElementField(e::Exp, k::Int, name::String, ty::SType, ht::HT)::Exp
  e isa EXP_CREF && return EXP_CREF(SimCref(Symbol(crefName(e.cref), "_", name)), ty)
  e isa RECORD && return expandRecordArguments(e.exps[k], ht)
  return TSUB(expandRecordArguments(e, ht), k, ty)
end
