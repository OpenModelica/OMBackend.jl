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
isParameter(v::SIMVAR)::Bool = v.varKind isa PARAMETER
"A parameter with a binding."
hasBindingExp(v::SIMVAR)::Bool = v.varKind isa PARAMETER && v.varKind.bindExp !== nothing

"An unknown: a state, a state derivative, an algebraic, an array or a discrete variable (not a parameter or a constant)."
isUnknownVarKind(k::SimVarType)::Bool = k isa Union{STATE, STATE_DERIVATIVE, ALG_VARIABLE, ARRAY, DISCRETE}

"The binding of a variable category that has one (a parameter's, a data structure's, an array's, a String's)."
bindingOf(k::SimVarType)::Union{Nothing, Exp} = hasproperty(k, :bindExp) ? k.bindExp : nothing

"A variable category with its binding replaced by `b` (of one that has a binding)."
withBinding(k::PARAMETER, b::Union{Nothing, Exp})::SimVarType = PARAMETER(b)
withBinding(k::DATA_STRUCTURE, b::Union{Nothing, Exp})::SimVarType = DATA_STRUCTURE(b)
withBinding(k::STRING, b::Union{Nothing, Exp})::SimVarType = STRING(b)
withBinding(k::ARRAY, b::Union{Nothing, Exp})::SimVarType = ARRAY(k.dimensions, b)
withBinding(k::ARRAY_PARAMETER, b::Union{Nothing, Exp})::SimVarType = ARRAY_PARAMETER(k.dimensions, b)
withBinding(k::SimVarType, ::Union{Nothing, Exp})::SimVarType = k

"A variable with another category."
withKind(v::SIMVAR, k::SimVarType)::SIMVAR = SIMVAR(v.name, v.index, k, v.attributes, v.elementType)
"A variable with the index `index` (and the category `k`)."
withIndex(v::SIMVAR, index::Int, k::SimVarType = v.varKind)::SIMVAR = SIMVAR(v.name, index, k, v.attributes, v.elementType)

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
    complexFieldNames!(names, equations, ht)

For each reference of a record type in the equations, the names of its fields' variables
(`r_f`; of a reference whose `r_re` and `r_im` are variables, those): the code generation
flattens a record reference into its fields, which must stay.
"""
function complexFieldNames!(names::OrderedSet{String}, equations::AbstractVector{<:Equation}, ht::HT)::OrderedSet{String}
  local visit = x -> begin
    x isa EXP_CREF || return
    local base = crefName(x.cref)
    if x.ty isa TYPE_COMPLEX
      for f in x.ty.fieldNames
        local n = string(base, "_", f)
        haskey(ht, n) && push!(names, n)
      end
    else
      local (re, im) = (string(base, "_re"), string(base, "_im"))
      (haskey(ht, re) && haskey(ht, im)) && push!(names, re, im)
    end
  end
  for eq in equations
    for e in equationExps(eq)
      foreachExp(visit, e)
    end
  end
  return names
end

"The expressions of an equation (its sides; a residual's expression)."
equationExps(eq::RESIDUAL_EQUATION)::Vector{Exp} = Exp[eq.exp]
equationExps(eq::EQUATION)::Vector{Exp} = Exp[eq.lhs, eq.rhs]
equationExps(eq::ARRAY_EQUATION)::Vector{Exp} = Exp[eq.left, eq.right]
equationExps(eq::Equation)::Vector{Exp} = Exp[]

"""
    whenCrefNames!(names, w)

The references of a when-clause: its condition's, its statements' (an assignment's sides, a
reinit's state and value, a call's, an assert's condition and message), its elsewhen arms'.
"""
function whenCrefNames!(names::OrderedSet{String}, w::WHEN_STMTS)::OrderedSet{String}
  crefNames!(names, w.condition)
  for s in w.body
    if s isa ASSIGN
      crefNames!(names, s.left); crefNames!(names, s.right)
    elseif s isa REINIT
      push!(names, crefName(s.stateVar)); crefNames!(names, s.value)
    elseif s isa NORETCALL
      crefNames!(names, s.exp)
    elseif s isa ASSERT
      crefNames!(names, s.condition); crefNames!(names, s.message)
    end
  end
  w.elsewhenPart === nothing || whenCrefNames!(names, w.elsewhenPart)
  return names
end

"A name without its subscripts (`a[1][2]`: `a`)."
baseName(name::AbstractString)::String = String(first(split(name, '['; limit = 2)))

"""
    derivedStateNames(simCode)

The variables under der() (a residual, an eliminated or initial equation, an if-branch, a
when-equation, an initial algorithm, an assert) and those a reinit() sets: the states with
dynamics.
"""
function derivedStateNames(simCode::SIM_CODE)::OrderedSet{String}
  local derived = OrderedSet{String}()
  local visit = e::Exp -> foreachExp(e) do x
    x isa DER && push!(derived, crefName(x.cref))
    (isBuiltin(x, :der) && first(x.args) isa EXP_CREF) && push!(derived, crefName(first(x.args).cref))
  end
  local visitStatement = s::Statement -> begin
    s isa REINIT && push!(derived, crefName(s.stateVar))
    mapStatementExps(e -> (visit(e); e), s)
  end
  foreach(eq -> visit(eq.exp), simCode.residualEquations)
  foreach(eq -> visit(eq.exp), simCode.eliminatedEquations)
  foreach(eq -> foreach(visit, equationExps(eq)), simCode.initialEquations)
  for ifEq in simCode.ifEquations, b in ifEq.branches
    visit(b.condition)
    foreach(eq -> visit(eq.exp), b.residualEquations)
  end
  for w in simCode.whenEquations
    local stmts = w.whenEquation
    while stmts !== nothing
      visit(stmts.condition)
      foreach(visitStatement, stmts.body)
      stmts = stmts.elsewhenPart
    end
  end
  foreach(a -> foreach(visitStatement, a.statements), simCode.initialAlgorithms)
  foreach(a -> (visit(a.condition); visit(a.message)), simCode.asserts)
  return derived
end

"""
    deadStateNames(simCode)

The states without dynamics: none of `derivedStateNames` (an array element's by its array's
name too).
"""
function deadStateNames(simCode::SIM_CODE)::Vector{String}
  local derived = derivedStateNames(simCode)
  local derivedBases = Set{String}(baseName(n) for n in derived)
  return String[v.name for (_, (_, v)) in simCode.stringToSimVarHT
                if v.varKind isa STATE && !(v.name in derived) && !(baseName(v.name) in derivedBases)]
end

"""
    statementCrefNames!(names, body)

The references of statements: their expressions' (an assert's level too), a reinit's state,
of the nested statements.
"""
function statementCrefNames!(names::OrderedSet{String}, body::Vector{Statement})::OrderedSet{String}
  for s in body
    statementCrefNames!(names, s)
  end
  return names
end

statementCrefNames!(names::OrderedSet{String}, s::ASSIGN) = (crefNames!(names, s.left); crefNames!(names, s.right))
statementCrefNames!(names::OrderedSet{String}, s::REINIT) = (push!(names, crefName(s.stateVar)); crefNames!(names, s.value))
statementCrefNames!(names::OrderedSet{String}, s::NORETCALL) = crefNames!(names, s.exp)
statementCrefNames!(names::OrderedSet{String}, s::ASSERT) =
  (crefNames!(names, s.condition); crefNames!(names, s.message); crefNames!(names, s.level))
statementCrefNames!(names::OrderedSet{String}, s::TERMINATE) = crefNames!(names, s.message)
statementCrefNames!(names::OrderedSet{String}, s::Union{IF, WHEN}) =
  (foreach(b -> (crefNames!(names, b.condition); statementCrefNames!(names, b.body)), s.branches);
   s isa IF && statementCrefNames!(names, s.elseBody); names)
statementCrefNames!(names::OrderedSet{String}, s::FOR) = (crefNames!(names, s.iterator.range); statementCrefNames!(names, s.body))
statementCrefNames!(names::OrderedSet{String}, s::WHILE) = (crefNames!(names, s.condition); statementCrefNames!(names, s.body))
statementCrefNames!(names::OrderedSet{String}, s::RECOMPILATION) = crefNames!(names, s.newValue)
statementCrefNames!(names::OrderedSet{String}, ::Statement) = names

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
