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

#= Queries on the simulation code: its variables' categories, the references of its
   expressions and equations. =#

isState(v::SIMVAR)::Bool = v.varKind isa STATE
isAlgebraic(v::SIMVAR)::Bool = v.varKind isa ALG_VARIABLE
isStateOrAlgebraic(v::SIMVAR)::Bool = isAlgebraic(v) || isState(v)
isDiscrete(v::SIMVAR)::Bool = v.varKind isa DISCRETE
"The value type of a type's elements (an array's): `:real`, `:int` (Integer, enumerations), `:bool`, or `:other`."
function valueType(ty::SType)::Symbol
  local el::SType = ty isa TYPE_ARRAY ? ty.elementType : ty
  el isa TYPE_REAL && return :real
  el isa Union{TYPE_INTEGER, TYPE_ENUM} && return :int
  el isa TYPE_BOOL && return :bool
  return :other
end

"An array variable: of dimensions (`SIMVAR.dims`)."
isArray(v::SIMVAR)::Bool = !isempty(v.dims)
"A parameter, a scalar or an array."
isParameter(v::SIMVAR)::Bool = v.varKind isa PARAMETER
"A scalar parameter."
isScalarParameter(v::SIMVAR)::Bool = v.varKind isa PARAMETER && !isArray(v)
"An array parameter."
isArrayParameter(v::SIMVAR)::Bool = v.varKind isa PARAMETER && isArray(v)
"An algebraic array variable."
isArrayAlgebraic(v::SIMVAR)::Bool = v.varKind isa ALG_VARIABLE && isArray(v)
"A scalar parameter with a binding."
hasBindingExp(v::SIMVAR)::Bool = isScalarParameter(v) && v.varKind.bindExp !== nothing

"""
    outputPosition(ty, k) -> Int

The position of output `k` of a call of the tuple type `ty` in the function's result as generated (`TSUB`'s index): a
record output before it is its fields (a generated function returns them in place); `k` for another type. For a tuple of
one target per output (the array route's tuple equations, `Domains._tupleAssignments`); the scalar route's tuples have
a record target's fields in place already (`expandRecordArguments`) and number their targets by position.
"""
outputPosition(ty::SType, k::Int)::Int =
  ty isa TYPE_TUPLE ? 1 + sum((t isa TYPE_COMPLEX && t.isRecord ? length(t.fieldNames) : 1 for t in ty.types[1:k - 1]); init = 0) : k

"""
    irCategory(v) -> Symbol

What `v` is by the kind it was lowered from (`SIMVAR.irKind`), as a model with its arrays kept lays its variables out
(`Domains.domainSystem`): `:known` (a parameter or a constant), `:state`, `:discrete`, `:algebraic`, `:derivative` (an
algebraic variable's derivative, index reduction's dummy derivative), `:dummyState` (index reduction's), `:none` for
another kind or none.
"""
function irCategory(v::SIMVAR)::Symbol
  local k::Union{Nothing, IR.VariableKind} = v.irKind
  k isa Union{IR.PARAMETER, IR.CONSTANT} && return :known
  k isa IR.STATE && return :state
  k isa IR.DISCRETE && return :discrete
  k isa IR.ALGEBRAIC && return :algebraic
  k isa Union{IR.ALGEBRAIC_DER, IR.DUMMY_DER} && return :derivative
  k isa IR.DUMMY_STATE && return :dummyState
  return :none
end

"The name of the kind `v` was lowered from (`SIMVAR.irKind`), `none` for none: a message's."
irKindName(v::SIMVAR)::String = v.irKind === nothing ? "none" : String(nameof(typeof(v.irKind)))

"""
    isOwnArray(v) -> Bool

Whether `v` is an array state or algebraic variable of the model's own: not a member of a component array the index
reduction rolled (`IR.isRolled` on a part of its reference).
"""
isOwnArray(v::SIMVAR)::Bool =
  isArray(v) && v.irKind isa Union{IR.STATE, IR.ALGEBRAIC, IR.ALGEBRAIC_DER} && v.ref !== nothing &&
  !any((p::IR.CrefPart) -> IR.isRolled(p.ty), v.ref.parts)

"An unknown: a state, a state derivative, an algebraic or a discrete variable, a scalar or an array (not a parameter or a constant)."
isUnknownVarKind(k::SimVarType)::Bool = k isa Union{STATE, STATE_DERIVATIVE, ALG_VARIABLE, DISCRETE}

"The binding of a variable category that has one (a parameter's, a data structure's, a String's)."
bindingOf(k::SimVarType)::Union{Nothing, Exp} = hasproperty(k, :bindExp) ? k.bindExp : nothing

"A variable category with its binding replaced by `b` (of one that has a binding)."
withBinding(k::PARAMETER, b::Union{Nothing, Exp})::SimVarType = PARAMETER(b)
withBinding(k::DATA_STRUCTURE, b::Union{Nothing, Exp})::SimVarType = DATA_STRUCTURE(b)
withBinding(k::STRING, b::Union{Nothing, Exp})::SimVarType = STRING(b)
withBinding(k::SimVarType, ::Union{Nothing, Exp})::SimVarType = k

"The keys of the states' derivative entries by the states' keys (`STATE_DERIVATIVE`, `createIndices`)."
derivativeKeys(ht::HT)::Dict{String, String} =
  Dict{String, String}(v.varKind.varName => k for (k, (_, v)) in ht if v.varKind isa STATE_DERIVATIVE)

"A variable with another category."
withKind(v::SIMVAR, k::SimVarType)::SIMVAR = SIMVAR(v.name, v.index, k, v.attributes, v.elementType, v.dims, v.irKind, v.irOrigin, v.ref)
"A variable with the index `index` (and the category `k`)."
withIndex(v::SIMVAR, index::Int, k::SimVarType = v.varKind)::SIMVAR =
  SIMVAR(v.name, index, k, v.attributes, v.elementType, v.dims, v.irKind, v.irOrigin, v.ref)
"A variable with the IR reference `cref`."
withCref(v::SIMVAR, cref::Union{Nothing, IR.ComponentRef})::SIMVAR =
  SIMVAR(v.name, v.index, v.varKind, v.attributes, v.elementType, v.dims, v.irKind, v.irOrigin, cref)
"A variable with the attributes `a`."
withAttributes(v::SIMVAR, a::SimVarAttributes)::SIMVAR =
  SIMVAR(v.name, v.index, v.varKind, a, v.elementType, v.dims, v.irKind, v.irOrigin, v.ref)

hasStructuralTransitions(sc::SIM_CODE)::Bool = !isempty(sc.structuralTransitions)
hasSubModels(sc::SIM_CODE)::Bool = !isempty(sc.subModels)
hasFrontendModel(sc::SIM_CODE)::Bool = sc.frontendModel !== nothing

"Whether `e` reads an unknown (`isUnknownVarKind`)."
function hasUnknownCref(e::Exp, ht::HT)::Bool
  for n in crefNames!(OrderedSet{String}(), e)
    local entry = get(ht, n, nothing)
    (entry !== nothing && isUnknownVarKind(last(entry).varKind)) && return true
  end
  return false
end

"The name of a variable reference; `nothing` for another expression (an array element read by ASUB too)."
crefNameOf(e::Exp)::Union{Nothing, String} = e isa EXP_CREF ? crefName(e.cref) : nothing

"""
    complexFieldNames!(names, equations, simCode)

For each reference of a record type in the equations, the names of its fields' variables
(`r_f`: the variables of its reference's fields, `keyOf`; of a reference whose `r_re` and `r_im`
are variables, those): the code generation flattens a record reference into its fields, which must stay.
"""
function complexFieldNames!(names::OrderedSet{String}, equations::AbstractVector{<:Equation}, simCode::SIM_CODE)::OrderedSet{String}
  local ht::HT = simCode.stringToSimVarHT
  local fieldKey::Function = (x::EXP_CREF, f::String) -> x.cref.ref === nothing ? nothing : keyOf(simCode, fieldCref(x.cref.ref, f))
  local visit::Function = (x::Exp) -> begin
    x isa EXP_CREF || return
    if x.ty isa TYPE_COMPLEX
      for f in x.ty.fieldNames
        local n::Union{Nothing, String} = fieldKey(x, f)
        (n !== nothing && haskey(ht, n)) && push!(names, n)
      end
    else
      local (re::Union{Nothing, String}, im::Union{Nothing, String}) = (fieldKey(x, "re"), fieldKey(x, "im"))
      (re !== nothing && im !== nothing && haskey(ht, re) && haskey(ht, im)) && push!(names, re, im)
    end
  end
  for eq in equations
    for e in equationExps(eq)
      foreachExp(visit, e)
    end
  end
  return names
end

"""
The reference of the field `f` of the record reference `ref` (`r.f`), its part of the field's type
in the record type (of a subscripted part the element's, as `recordFieldExp`); `T_UNKNOWN` if the
record type is not known.
"""
function fieldCref(ref::IR.ComponentRef, f::AbstractString)::IR.ComponentRef
  local name::Symbol = Symbol(f)
  local last::IR.CrefPart = ref.parts[end]
  local record::IR.ModelicaType = isempty(last.subscripts) ? last.ty : IR.elementType(last.ty)
  local ty::IR.ModelicaType = IR.T_UNKNOWN()
  if record isa IR.T_RECORD
    local i::Union{Nothing, Int} = findfirst(fl::IR.Field -> fl.name === name, record.fields)
    i === nothing || (ty = record.fields[i].ty)
  end
  return IR.withField(ref, name, ty)
end

"The expressions of an equation (its sides; a residual's expression)."
equationExps(eq::RESIDUAL_EQUATION)::Vector{Exp} = Exp[eq.exp]
equationExps(eq::EQUATION)::Vector{Exp} = Exp[eq.lhs, eq.rhs]
equationExps(eq::ARRAY_EQUATION)::Vector{Exp} = Exp[eq.left, eq.right]
equationExps(eq::Equation)::Vector{Exp} = Exp[]

"""
    whenCrefNames!(names, w)

The references of a when-clause: its condition's, its statements' (`statementCrefNames!`: an
assignment's sides, a reinit's state and value, a call's, an assert's, a terminate's message,
nested statements'), its elsewhen arms'.
"""
function whenCrefNames!(names::OrderedSet{String}, w::WHEN_STMTS; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  crefNames!(names, w.condition; dynamic)
  statementCrefNames!(names, w.body; dynamic)
  w.elsewhenPart === nothing || whenCrefNames!(names, w.elsewhenPart; dynamic)
  return names
end

"""
    arrayBase(ref) -> Symbol

The name of the array the reference `ref` is an element of, as the scalarized model names its elements: `ref` up to
its first part with subscripts, without them (`x` of `x[1,2]`, `c` of `c[2].T`); a rolled component's member by its
literal index (`cylinder3`) is a name there, not a subscript; `ref`'s own name where it has none.
"""
arrayBase(ref::IR.ComponentRef)::Symbol = IR.upToFirstSubscripted(ref).name

"Whether the reference names an element of an array (`arrayBase` is not its own name)."
isElementCref(ref::IR.ComponentRef)::Bool = arrayBase(ref) !== ref.name

"Whether the reference names the field `re` or `im` of a record (a Complex's part: its part before of a record type)."
isComplexPartCref(ref::IR.ComponentRef)::Bool =
  length(ref.parts) >= 2 && ref.parts[end].name in (:re, :im) && IR.elementType(ref.parts[end - 1].ty) isa IR.T_RECORD

"""
    elementIndex(ref, array) -> Union{Nothing, Vector{Int}}

The Integer subscripts by which the reference `ref` is an element of the array reference `array`: `array`'s parts, its
last part subscripted (`A[1,2]` of `A`, `a.b[3]` of `a.b`); `nothing` for another reference.
"""
function elementIndex(ref::IR.ComponentRef, array::IR.ComponentRef)::Union{Nothing, Vector{Int}}
  local n::Int = length(array.parts)
  (length(ref.parts) == n && isempty(array.parts[n].subscripts) && ref.parts[n].name === array.parts[n].name &&
   !isempty(ref.parts[n].subscripts)) || return nothing
  (n == 1 || IR.ComponentRef(ref.parts[1:n - 1]) == IR.ComponentRef(array.parts[1:n - 1])) || return nothing
  return IR.literalIndices(ref.parts[n].subscripts)
end

"""
    derivedStateCrefs(simCode)

The references under der() (a residual, an eliminated or initial equation, an if-branch, a
when-equation, an initial algorithm, an assert) and those a reinit() sets: the states with
dynamics.
"""
function derivedStateCrefs(simCode::SIM_CODE)::OrderedSet{SimCref}
  local derived::OrderedSet{SimCref} = OrderedSet{SimCref}()
  local visit::Function = (e::Exp) -> foreachExp(e) do x::Exp
    x isa DER && push!(derived, x.cref)
    (isBuiltin(x, :der) && first(x.args) isa EXP_CREF) && push!(derived, first(x.args).cref)
  end
  foreachEquationExp((_::Tuple{Symbol, Int}, e::Exp) -> visit(e), simCode; eliminated = true,
                     statement = (_::Tuple{Symbol, Int}, s::Statement) -> (s isa REINIT && push!(derived, s.stateVar); nothing))
  return derived
end

"""
    deadStateNames(simCode)

The states without dynamics: none of `derivedStateCrefs` (an array element's by its array,
`arrayBase` of the references).
"""
function deadStateNames(simCode::SIM_CODE)::Vector{String}
  local derived::OrderedSet{SimCref} = derivedStateCrefs(simCode)
  local names::Set{String} = Set{String}(crefName(c) for c in derived)
  local bases::Set{Symbol} = Set{Symbol}(arrayBase(c.ref) for c in derived if c.ref !== nothing)
  return String[v.name for (_, (_, v)) in simCode.stringToSimVarHT
                if v.varKind isa STATE && !(v.name in names) && !(v.ref !== nothing && something(elementArray(simCode, v.name), v.ref.name) in bases)]
end

"""
    statementCrefNames!(names, body)

The references of statements: their expressions' (an assert's level too), a reinit's state,
of the nested statements.
"""
function statementCrefNames!(names::OrderedSet{String}, body::Vector{Statement}; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  foreachStatementExp((e::Exp) -> crefNames!(names, e; dynamic), body,
                      (s::Statement) -> (s isa REINIT && push!(names, crefName(s.stateVar)); nothing))
  return names
end

statementCrefNames!(names::OrderedSet{String}, s::Statement; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String} =
  statementCrefNames!(names, Statement[s]; dynamic)

#= ==================================================================================== =#
#= The types of expressions                                                             =#
#= ==================================================================================== =#

"""
    typeOf(e)

The type of the expression `e`: a literal's, a reference's, a call's and the other typed
nodes' own; arithmetic Integer of Integer operands, else Real (a division and a power Real),
an array of the array operand's dimensions; a relation and a logical expression Boolean; an
if-expression its branch's; an array element its array's element type.
"""
typeOf(::ICONST)::SType = TYPE_INTEGER()
typeOf(::RCONST)::SType = TYPE_REAL()
typeOf(::BCONST)::SType = TYPE_BOOL()
typeOf(::SCONST)::SType = TYPE_STRING()
typeOf(::ENUM_LITERAL)::SType = TYPE_ENUM(())
typeOf(::DER)::SType = TYPE_REAL()
typeOf(::Union{WILD, WHOLE_DIMENSION})::SType = TYPE_UNKNOWN()
typeOf(e::Union{EXP_CREF, PRE, CALL, CAST, RSUB, TSUB, RECORD, ARRAY_EXP, RANGE, REDUCTION, PARTEVALFUNCTION})::SType = e.ty
typeOf(::Union{RELATION, LBINARY, LUNARY})::SType = TYPE_BOOL()
typeOf(e::UNARY)::SType = typeOf(e.exp)
typeOf(e::IFEXP)::SType = typeOf(e.thenExp)
typeOf(e::TUPLE)::SType = TYPE_TUPLE(SType[typeOf(x) for x in e.elements])

function typeOf(e::BINARY)::SType
  local (t1, t2) = (typeOf(e.exp1), typeOf(e.exp2))
  e.op === OP_SCALAR_PRODUCT && return scalarType(t1)
  local product = productType(e.op, t1, t2)
  product === nothing || return product
  local array = t1 isa TYPE_ARRAY ? t1 : t2 isa TYPE_ARRAY ? t2 : nothing
  local element = e.op in (OP_DIV, OP_DIV_EW, OP_DIV_SCALAR_ARRAY, OP_DIV_ARRAY_SCALAR, OP_POW, OP_POW_EW,
                           OP_POW_SCALAR_ARRAY, OP_POW_ARRAY_SCALAR, OP_POW_MATRIX) ? TYPE_REAL() :
    (scalarType(t1) isa TYPE_INTEGER && scalarType(t2) isa TYPE_INTEGER) ? TYPE_INTEGER() : TYPE_REAL()
  array === nothing && return element
  return TYPE_ARRAY(element, array.dims)
end

function typeOf(e::ASUB)::SType
  local t = typeOf(e.exp)
  t isa TYPE_ARRAY || return t
  local kept = Int[d for (k, d) in enumerate(t.dims) if k > length(e.subs) ||
                   e.subs[k] isa WHOLE_DIMENSION || typeOf(e.subs[k]) isa TYPE_ARRAY]
  return isempty(kept) ? t.elementType : TYPE_ARRAY(t.elementType, kept)
end

"""
The type of a vector-matrix, matrix-vector or matrix product of operands of known dimensions
(`A[n, m] * v[m]` is `[n]`); `nothing` for another operator or operands.
"""
function productType(op::OpKind, t1::SType, t2::SType)::Union{Nothing, SType}
  (op in (OP_MUL_MATRIX_VECTOR, OP_MUL_VECTOR_MATRIX, OP_MATRIX_PRODUCT) && t1 isa TYPE_ARRAY && t2 isa TYPE_ARRAY) ||
    return nothing
  local element = (t1.elementType isa TYPE_INTEGER && t2.elementType isa TYPE_INTEGER) ? TYPE_INTEGER() : TYPE_REAL()
  op === OP_MUL_MATRIX_VECTOR && length(t1.dims) == 2 && return TYPE_ARRAY(element, Int[t1.dims[1]])
  op === OP_MUL_VECTOR_MATRIX && length(t2.dims) == 2 && return TYPE_ARRAY(element, Int[t2.dims[2]])
  op === OP_MATRIX_PRODUCT && length(t1.dims) == 2 && length(t2.dims) == 2 && return TYPE_ARRAY(element, Int[t1.dims[1], t2.dims[2]])
  return nothing
end

"The scalar type of `t` (of an array, its element type)."
scalarType(t::TYPE_ARRAY)::SType = t.elementType
scalarType(t::SType)::SType = t
