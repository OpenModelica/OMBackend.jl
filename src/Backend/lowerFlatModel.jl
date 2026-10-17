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

"""
    LowerFlatModel

The frontend's flat model (NF) lowered into OMJLIR: the one place in OMBackend that reads the
frontend's representation. `lowerFlatModel(fm)` gives the `OMJLIR.Model`.

The lowering translates; it does not transform. One equation of the flat model is one
equation of OMJLIR (a connection equality `a = b` is the residual `a + (-b) = 0`, as the
frontend's DAE conversion made it), a when-equation's body is statements, the algorithm
sections are ALGORITHM equations and the initial algorithm sections ALGORITHM initial
equations. Derivatives stay `der(x)` calls; the passes after the lowering introduce the
derivative variables, decompose the record equations and normalize the when-equations.

The order of the equations is the order the previous backend (BDAE, through the frontend's
DAE conversion) had, so that the code generated from OMJLIR is the code generated before:
the equations and the initial equations of the flat model in reverse; the equations of an
if-equation's branches and of a when-equation's body in their order, the else-branch's in
reverse. The variables are in their order. An array equation `lhs = rhs` is `rhs = lhs`, the
previous backend's orientation (the residuals of its elements are `rhs[i] - lhs[i]`).

Component references keep the frontend's parts, each name with its subscripts: `a[2].b` (a
variable of a scalarized array of records), `x[i]` (in a loop, before the passes substitute
the iterator), `r.x[2]` (a record's field in a function).
"""
module LowerFlatModel

import Absyn
import MetaModelica
import MetaModelica: isvariant
import OMFrontend
import ..OMJLIR
import ..unsupported

const F = OMFrontend.Frontend
const IR = OMJLIR

export lowerFlatModel, FunctionList

"The frontend's list of the functions a model calls (OMFrontend.cacheToFunctionList)."
const FunctionList = MetaModelica.List{F.M_FUNCTION}

#= ==================================================================================== =#
#= Names and sources                                                                    =#
#= ==================================================================================== =#

"The dotted name of a class or function path."
pathName(path::Absyn.Path)::Symbol = Symbol(F.AbsynUtil.pathString(path))

"The location in the Modelica source, from a frontend `SourceInfo`."
lowerSource(info::MetaModelica.SourceInfo)::IR.Source = IR.Source(info.fileName, info.lineNumberStart, info.columnNumberStart)

"The frontend's element source of an equation or a statement (its location)."
const ElementSource = fieldtype(F.NFEquationImpl, :source)

"The location of an equation or a statement, from the frontend's element source."
lowerElementSource(source::ElementSource)::IR.Source = lowerSource(source.info)

#= ==================================================================================== =#
#= Types, dimensions and subscripts                                                     =#
#= ==================================================================================== =#

"The OMJLIR type of the frontend type `ty`."
function lowerType(ty::F.NFType)::IR.ModelicaType
  isvariant(ty, F.TYPE_REAL) && return IR.T_REAL()
  isvariant(ty, F.TYPE_INTEGER) && return IR.T_INTEGER()
  isvariant(ty, F.TYPE_BOOLEAN) && return IR.T_BOOLEAN()
  isvariant(ty, F.TYPE_STRING) && return IR.T_STRING()
  isvariant(ty, F.TYPE_CLOCK) && return IR.T_CLOCK()
  isvariant(ty, F.TYPE_ENUMERATION) && return enumerationType(ty)
  isvariant(ty, F.TYPE_ARRAY) &&
    return IR.T_ARRAY(lowerType(ty.elementType), IR.Dimension[lowerDimension(d) for d in ty.dimensions])
  isvariant(ty, F.TYPE_TUPLE) && return IR.T_TUPLE(IR.ModelicaType[lowerType(t) for t in ty.types])
  isvariant(ty, F.TYPE_COMPLEX) && return complexType(ty)
  isvariant(ty, F.TYPE_FUNCTION) && return functionType(ty.fn)
  isvariant(ty, F.TYPE_NORETCALL) && return IR.T_NORETCALL()
  isvariant(ty, F.TYPE_SUBSCRIPTED) && return lowerType(ty.subscriptedTy)
  isvariant(ty, F.TYPE_METABOXED) && return lowerType(ty.ty)
  #= TYPE_UNKNOWN, TYPE_ANY, TYPE_ENUMERATION_ANY, TYPE_POLYMORPHIC: the types of the
     generic built-in functions' inputs. =#
  return IR.T_UNKNOWN()
end

enumerationType(ty::F.NFType)::IR.T_ENUMERATION =
  IR.T_ENUMERATION(pathName(ty.typePath), Symbol[Symbol(l) for l in ty.literals])

"""
A record type with its fields, an external object class (MLS 12.9.7), or the class of a
model, block or connector component (a part of a component reference). A record that a
field's dimension reads again (`a[size(r.b, 1)]` in a function on `r`: Buildings' cooling
towers, Movers, HydronicConfigurations) is there without its fields: lowering them again
recursed without end.
"""
function complexType(ty::F.NFType)::IR.ModelicaType
  local cls = ty.cls
  local name = pathName(F.scopePath(cls; includeRoot = true))
  isvariant(ty.complexTy, F.COMPLEX_EXTERNAL_OBJECT) && return IR.T_EXTERNAL_OBJECT(name)
  isvariant(ty.complexTy, F.COMPLEX_RECORD) || return IR.T_CLASS(name)
  local inProgress = get!(() -> Set{Symbol}(), task_local_storage(), :omjlir_records_lowering)::Set{Symbol}
  name in inProgress && return IR.T_RECORD(name, IR.Field[])
  push!(inProgress, name)
  try
    local fields = IR.Field[]
    for c in F.getComponents(F.getClass(cls).elements)
      F.isOnlyOuter(c) && continue
      push!(fields, IR.Field(Symbol(F.name(c)), lowerType(F.getType(F.component(c)))))
    end
    return IR.T_RECORD(name, fields)
  finally
    delete!(inProgress, name)
  end
end

"The type of a function value: its name, its inputs' and its outputs' types."
function functionType(fn::F.M_Function)::IR.T_FUNCTION
  local inputs = IR.ModelicaType[lowerType(F.getType(F.component(n))) for n in fn.inputs]
  local outputs = IR.ModelicaType[lowerType(F.getType(F.component(n))) for n in fn.outputs]
  return IR.T_FUNCTION(pathName(F.nameConsiderBuiltin(fn)), inputs, outputs)
end

"An OMJLIR dimension."
function lowerDimension(d::F.Dimension)::IR.Dimension
  isvariant(d, F.DIMENSION_INTEGER) && return IR.DIM_SIZE(d.size)
  isvariant(d, F.DIMENSION_BOOLEAN) && return IR.DIM_BOOLEAN()
  isvariant(d, F.DIMENSION_ENUM) && return IR.DIM_ENUM(enumerationType(d.enumType))
  isvariant(d, F.DIMENSION_EXP) && return IR.DIM_EXP(lowerExp(d.exp))
  return IR.DIM_UNKNOWN()
end

"The OMJLIR subscript of the frontend subscript `s`."
function lowerSubscript(s::F.Subscript)::IR.Subscript
  s isa F.SUBSCRIPT_INDEX && return IR.INDEX(lowerExp(s.index))
  s isa F.SUBSCRIPT_SLICE && return IR.SLICE(lowerExp(s.slice))
  s isa F.SUBSCRIPT_WHOLE && return IR.WHOLE()
  error("OMJLIR lowering: subscript $(F.toString(s)) of an untyped or split form")
end

#= ==================================================================================== =#
#= Component references                                                                 =#
#= ==================================================================================== =#

"The parts of the frontend component reference `cref`, the outermost first."
function crefParts(cref::F.ComponentRef)::Vector{IR.CrefPart}
  local parts = IR.CrefPart[]
  while isvariant(cref, F.COMPONENT_REF_CREF)
    push!(parts, IR.CrefPart(Symbol(F.name(cref.node)), IR.Subscript[lowerSubscript(s) for s in cref.subscripts],
                             lowerType(cref.ty)))
    cref = cref.restCref
  end
  return reverse!(parts)
end

"The OMJLIR reference of the frontend component reference `cref`."
lowerCref(cref::F.ComponentRef)::IR.ComponentRef = IR.ComponentRef(crefParts(cref))

"A reference to a variable as an expression: `_` for the frontend's wildcard."
lowerCrefExp(cref::F.ComponentRef, ty::IR.ModelicaType)::IR.Expression =
  isvariant(cref, F.COMPONENT_REF_WILD) ? IR.WILD() : IR.CREF(lowerCref(cref), ty)

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

"NF's operator kinds by number, as OMJLIR's."
const OPS = let ops = Dict{Int, IR.Op}()
  for name in fieldnames(typeof(F.Op))
    name === :USERDEFINED && continue
    ops[Int(getfield(F.Op, name))] = getfield(IR, name)
  end
  ops
end

"The OMJLIR operator of the frontend operator `operator`."
function lowerOp(operator::F.Operator)::IR.Op
  local op = get(OPS, Int(operator.op), nothing)
  op === nothing && error("OMJLIR lowering: a user-defined operator the frontend did not resolve")
  return op
end

"The OMJLIR type of the frontend expression `e`."
typeOfExp(e::F.NFExpression)::IR.ModelicaType = lowerType(F.typeOf(e))

"The frontend's expressions, of a vector or a MetaModelica list."
const Expressions = Union{AbstractVector{<:F.NFExpression}, MetaModelica.List{<:F.NFExpression}}

"The OMJLIR expressions of the frontend expressions `es`."
lowerExps(es::Expressions)::Vector{IR.Expression} = IR.Expression[lowerExp(e) for e in es]

"An optional expression (a MetaModelica `Option`): lowered, or `nothing`."
lowerOptExp(o::MetaModelica.Option{<:F.NFExpression})::Union{Nothing, IR.Expression} =
  o isa MetaModelica.SOME ? lowerExp(o.data) : nothing

"The OMJLIR expression of the frontend expression `e`."
lowerExp(e::F.INTEGER_EXPRESSION)::IR.Expression = IR.INTEGER(e.value)
lowerExp(e::F.REAL_EXPRESSION)::IR.Expression = IR.REAL(e.value)
lowerExp(e::F.STRING_EXPRESSION)::IR.Expression = IR.STRING(e.value)
lowerExp(e::F.BOOLEAN_EXPRESSION)::IR.Expression = IR.BOOLEAN(e.value)
lowerExp(e::F.ENUM_LITERAL_EXPRESSION)::IR.Expression =
  IR.ENUM_LITERAL(enumerationType(e.ty), Symbol(e.name), e.index)
lowerExp(e::F.CLKCONST_EXPRESSION)::IR.Expression = IR.CLOCK(lowerClock(e.clk))
"""
    lowerExp(e::F.CREF_EXPRESSION)

A reference of its expression's type; of a scalar element, the element's type: NF's
scalarization of kept arrays leaves an element with the array's type (Buildings' fluid ports,
`vol.ports[1].Xi_outflow[1]` of type `Real[1]`, which the code generation expanded as an
array: `[vol_ports[1]_Xi_outflow[1][1]]`).
"""
function lowerExp(e::F.CREF_EXPRESSION)::IR.Expression
  local ty = lowerType(e.ty)
  if ty isa IR.T_ARRAY && isvariant(e.cref, F.COMPONENT_REF_CREF)
    local element = lowerType(F.getSubscriptedType(e.cref))
    isScalarType(element) && (ty = element)
  end
  return lowerCrefExp(e.cref, ty)
end
lowerExp(e::F.TYPENAME_EXPRESSION)::IR.Expression = lowerExp(F.expandTypename(e.ty))
lowerExp(e::F.ARRAY_EXPRESSION)::IR.Expression = IR.ARRAY(lowerExps(e.elements), lowerType(e.ty))
function lowerExp(e::F.MATRIX_EXPRESSION)::IR.Expression
  local ty = typeOfExp(e)::IR.T_ARRAY
  local rowTy = IR.T_ARRAY(ty.elementType, ty.dimensions[2:end])
  return IR.ARRAY(IR.Expression[IR.ARRAY(lowerExps(row), rowTy) for row in e.elements], ty)
end
lowerExp(e::F.RECORD_EXPRESSION)::IR.Expression = IR.RECORD(lowerExps(e.elements), lowerType(e.ty))
lowerExp(e::F.RANGE_EXPRESSION)::IR.Expression =
  IR.RANGE(lowerExp(e.start), lowerOptExp(e.step), lowerExp(e.stop), lowerType(e.ty))
lowerExp(e::F.TUPLE_EXPRESSION)::IR.Expression = IR.TUPLE(lowerExps(e.elements), lowerType(e.ty))
lowerExp(e::F.CALL_EXPRESSION)::IR.Expression = lowerCall(e.call)
lowerExp(e::F.SIZE_EXPRESSION)::IR.Expression = IR.SIZE(lowerExp(e.exp), lowerOptExp(e.dimIndex))
"""
A binary operation. The frontend's scalarization keeps an array operator on the elements it
subscripts (`crossAreas[1] * lengths[1] * nParallel` an array times a scalar, of type
`Real[2]`), and the array's type on a scalar operator (a stream connection's
`vol.steBal.port_a.Xi_outflow[1] + -vol.ports[1].Xi_outflow[1]` of type `Real[1]`): of two
scalar operands the scalar operator, of the element type. A negation of a scalar likewise.
"""
function lowerExp(e::F.BINARY_EXPRESSION)::IR.Expression
  local (a, b) = (lowerExp(e.exp1), lowerExp(e.exp2))
  local (op, ty) = (lowerOp(e.operator), lowerType(e.operator.ty))
  if isScalarType(IR.typeOf(a)) && isScalarType(IR.typeOf(b))
    (op, ty) = (get(SCALAR_OPS, op, op), IR.elementType(ty))
  end
  return IR.BINARY(a, op, b, ty)
end

"The scalar operators of the element-wise and scalar-array ones."
const SCALAR_OPS = Dict{IR.Op, IR.Op}(
  IR.ADD_EW => IR.ADD, IR.ADD_SCALAR_ARRAY => IR.ADD, IR.ADD_ARRAY_SCALAR => IR.ADD,
  IR.SUB_EW => IR.SUB, IR.SUB_SCALAR_ARRAY => IR.SUB, IR.SUB_ARRAY_SCALAR => IR.SUB,
  IR.MUL_EW => IR.MUL, IR.MUL_SCALAR_ARRAY => IR.MUL, IR.MUL_ARRAY_SCALAR => IR.MUL,
  IR.DIV_EW => IR.DIV, IR.DIV_SCALAR_ARRAY => IR.DIV, IR.DIV_ARRAY_SCALAR => IR.DIV,
  IR.POW_EW => IR.POW, IR.POW_SCALAR_ARRAY => IR.POW, IR.POW_ARRAY_SCALAR => IR.POW)

"A type of a scalar value (not an array, a record or an unknown type)."
isScalarType(ty::IR.ModelicaType)::Bool = ty isa Union{IR.T_REAL, IR.T_INTEGER, IR.T_BOOLEAN, IR.T_ENUMERATION}
function lowerExp(e::F.UNARY_EXPRESSION)::IR.Expression
  local a = lowerExp(e.exp)
  local ty = lowerType(e.operator.ty)
  return IR.UNARY(lowerOp(e.operator), a, isScalarType(IR.typeOf(a)) ? IR.elementType(ty) : ty)
end
lowerExp(e::F.LBINARY_EXPRESSION)::IR.Expression =
  IR.LBINARY(lowerExp(e.exp1), lowerOp(e.operator), lowerExp(e.exp2))
lowerExp(e::F.LUNARY_EXPRESSION)::IR.Expression = IR.LUNARY(lowerOp(e.operator), lowerExp(e.exp))
lowerExp(e::F.RELATION_EXPRESSION)::IR.Expression =
  IR.RELATION(lowerExp(e.exp1), lowerOp(e.operator), lowerExp(e.exp2), 0)
lowerExp(e::F.IF_EXPRESSION)::IR.Expression =
  IR.IF_EXPRESSION(lowerExp(e.condition), lowerExp(e.trueBranch), lowerExp(e.falseBranch), typeOfExp(e))
lowerExp(e::F.CAST_EXPRESSION)::IR.Expression = IR.CAST(lowerExp(e.exp), lowerType(e.ty))
lowerExp(e::F.SUBSCRIPTED_EXP_EXPRESSION)::IR.Expression =
  IR.SUBSCRIPTED(lowerExp(e.exp), IR.Subscript[lowerSubscript(s) for s in e.subscripts], lowerType(e.ty))
lowerExp(e::F.TUPLE_ELEMENT_EXPRESSION)::IR.Expression =
  IR.TUPLE_ELEMENT(lowerExp(e.tupleExp), e.index, lowerType(e.ty))
lowerExp(e::F.RECORD_ELEMENT_EXPRESSION)::IR.Expression =
  IR.RECORD_ELEMENT(lowerExp(e.recordExp), e.index, Symbol(e.fieldName), lowerType(e.ty))
function lowerExp(e::F.PARTIAL_FUNCTION_APPLICATION_EXPRESSION)::IR.Expression
  local fn = F.typeRefCache(e.fn)[1]
  local inputs = Pair{Symbol, IR.ModelicaType}[Symbol(F.name(n)) => lowerType(F.getType(n)) for n in fn.inputs]
  return IR.PARTIAL_APPLICATION(pathName(F.nameConsiderBuiltin(fn)), lowerExps(e.args),
                                Symbol[Symbol(n) for n in e.argNames], inputs, lowerType(e.ty))
end
lowerExp(e::F.BINDING_EXP)::IR.Expression = lowerExp(e.exp)
"""
Boxing (of a function's polymorphic or functional arguments) has no meaning in Julia, where
every value is a value: the code generation always took the boxed expression itself.
"""
lowerExp(e::F.BOX_EXPRESSION)::IR.Expression = lowerExp(e.exp)
lowerExp(e::F.UNBOX_EXPRESSION)::IR.Expression = lowerExp(e.exp)
lowerExp(e::F.MUTABLE_EXPRESSION)::IR.Expression = lowerExp(F.P_Pointer.access(e.exp))
lowerExp(e::F.NFExpression)::IR.Expression =
  error("OMJLIR lowering: the expression $(F.toString(e)) ($(nameof(typeof(e)))) has no OMJLIR form")

"`inline`: the frontend's inline types that inline the function."
const INLINE_TYPES = (:NORM_INLINE, :BUILTIN_EARLY_INLINE, :EARLY_INLINE)

"A typed call, array constructor or reduction."
function lowerCall(call::F.Call)::IR.Expression
  if isvariant(call, F.TYPED_CALL)
    local a = call.attributes
    local attributes = IR.CallAttributes(a.builtin, a.isImpure, nameof(typeof(a.inlineType)) in INLINE_TYPES)
    return IR.CALL(pathName(F.nameConsiderBuiltin(call.fn)), lowerExps(call.arguments),
                   lowerType(call.ty), attributes)
  elseif isvariant(call, F.TYPED_ARRAY_CONSTRUCTOR)
    return IR.ARRAY_CONSTRUCTOR(lowerExp(call.exp), lowerIterators(call.iters), lowerType(call.ty))
  elseif isvariant(call, F.TYPED_REDUCTION)
    return IR.REDUCTION(pathName(F.nameConsiderBuiltin(call.fn)), lowerExp(call.exp), lowerIterators(call.iters),
                        lowerType(call.ty))
  end
  error("OMJLIR lowering: an untyped call")
end

"The iterators of an array constructor or a reduction: (iterator node, range) pairs."
lowerIterators(iters::Union{AbstractVector{<:Tuple}, MetaModelica.List{<:Tuple}})::Vector{IR.Iterator} =
  IR.Iterator[IR.Iterator(Symbol(F.name(node)), lowerExp(range)) for (node, range) in iters]

lowerClock(k::F.INFERRED_CLOCK)::IR.ClockKind = IR.INFERRED_CLOCK()
lowerClock(k::F.INTEGER_CLOCK)::IR.ClockKind =
  IR.INTEGER_CLOCK(lowerExp(k.intervalCounter), lowerExp(k.resolution))
lowerClock(k::F.REAL_CLOCK)::IR.ClockKind = IR.REAL_CLOCK(lowerExp(k.interval))
lowerClock(k::F.BOOLEAN_CLOCK)::IR.ClockKind =
  IR.BOOLEAN_CLOCK(lowerExp(k.condition), lowerExp(k.startInterval))
lowerClock(k::F.SOLVER_CLOCK)::IR.ClockKind =
  IR.SOLVER_CLOCK(lowerExp(k.c), lowerExp(k.solverMethod))

#= ==================================================================================== =#
#= Statements                                                                           =#
#= ==================================================================================== =#

"The OMJLIR statements of the frontend statements `stmts`, of a vector or a MetaModelica list."
lowerStatements(stmts::Union{AbstractVector{<:F.Statement}, MetaModelica.List{<:F.Statement}})::Vector{IR.Statement} =
  IR.Statement[lowerStatement(s) for s in stmts]

"""
    lowerAssignment(lhs, rhs, source)

An assignment, a tuple's as the frontend's DAE conversion made it: `() := f(x)` the call as a
statement, `(a) := f(x)` the assignment of the call's first output.
"""
function lowerAssignment(lhs::IR.Expression, rhs::IR.Expression, source::IR.Source)::IR.Statement
  lhs isa IR.TUPLE || return IR.ASSIGNMENT(lhs, rhs, source)
  isempty(lhs.elements) && return IR.NORETCALL(rhs, source)
  length(lhs.elements) == 1 &&
    return IR.ASSIGNMENT(lhs.elements[1], IR.TUPLE_ELEMENT(rhs, 1, IR.typeOf(lhs.elements[1])), source)
  return IR.ASSIGNMENT(lhs, rhs, source)
end

"The OMJLIR statement of the frontend statement `s`."
function lowerStatement(s::F.Statement)::IR.Statement
  local source = lowerElementSource(s.source)
  if isvariant(s, F.ALG_ASSIGNMENT)
    return lowerAssignment(lowerExp(s.lhs), lowerExp(s.rhs), source)
  elseif isvariant(s, F.ALG_IF)
    local branches = IR.StatementBranch[]
    local elseBody = IR.Statement[]
    for (k, (cond, body)) in enumerate(s.branches)
      if k == length(s.branches) && F.isTrue(cond)
        elseBody = lowerStatements(body)
      else
        push!(branches, IR.StatementBranch(lowerExp(cond), lowerStatements(body)))
      end
    end
    return IR.IF_STATEMENT(branches, elseBody, source)
  elseif isvariant(s, F.ALG_FOR)
    local name = Symbol(F.name(s.iterator))
    MetaModelica.isSome(s.range) || error("OMJLIR lowering: a for-statement without a range")
    local iterator = IR.Iterator(name, lowerExp(s.range.data))
    return IR.FOR_STATEMENT(iterator, lowerStatements(s.body), source)
  elseif isvariant(s, F.ALG_WHILE)
    return IR.WHILE_STATEMENT(lowerExp(s.condition), lowerStatements(s.body), source)
  elseif isvariant(s, F.ALG_WHEN)
    return IR.WHEN_STATEMENT(IR.WhenBranch[whenBranch(lowerExp(c), lowerStatements(b); dropInitialDisjuncts = false)
                                           for (c, b) in s.branches], source)
  elseif isvariant(s, F.ALG_ASSERT)
    return IR.ASSERT(lowerExp(s.condition), lowerExp(s.message), lowerExp(s.level), source)
  elseif isvariant(s, F.ALG_TERMINATE)
    return IR.TERMINATE(lowerExp(s.message), source)
  elseif isvariant(s, F.ALG_REINIT)
    return IR.REINIT(lowerCref(F.toCref(s.cref)), lowerExp(s.reinitExp), source)
  elseif isvariant(s, F.ALG_NORETCALL)
    return IR.NORETCALL(lowerExp(s.exp), source)
  elseif isvariant(s, F.ALG_RETURN)
    return IR.RETURN(source)
  elseif isvariant(s, F.ALG_BREAK)
    return IR.BREAK(source)
  end
  error("OMJLIR lowering: the statement $(F.toString(s)) has no OMJLIR form")
end

#= ==================================================================================== =#
#= Equations                                                                            =#
#= ==================================================================================== =#

"""
    lowerEquations(eqs; reversed)

The equations `eqs`, in reverse for the top-level, initial and else-branch equations (the
previous backend's order, see the module's documentation).
"""
function lowerEquations(eqs::Vector{F.Equation}; reversed::Bool)::Vector{IR.Equation}
  local out = IR.Equation[]
  for eq in (reversed ? Iterators.reverse(eqs) : eqs)
    local lowered = lowerEquation(eq)
    lowered === nothing || push!(out, lowered)
  end
  return out
end

const ATTRIBUTES = IR.DEFAULT_ATTRIBUTES

"The OMJLIR equation of the frontend equation `eq`; `nothing` for a connect equation."
function lowerEquation(eq::F.Equation)::Union{Nothing, IR.Equation}
  local source = lowerElementSource(eq.source)
  if isvariant(eq, F.EQUATION_EQUALITY) || isvariant(eq, F.EQUATION_ARRAY_EQUALITY)
    local ty = lowerType(eq.ty)
    local lhs = lowerExp(eq.lhs)
    local rhs = lowerExp(eq.rhs)
    F.isComplex(eq.ty) && return IR.RECORD_EQUATION(lhs, rhs, ty, recordSize(ty), source, ATTRIBUTES)
    #= `rhs = lhs`: see the module's documentation =#
    ty isa IR.T_ARRAY && return IR.ARRAY_EQUATION(rhs, lhs, ty, source, ATTRIBUTES)
    return IR.SCALAR_EQUATION(lhs, rhs, ty, source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_CREF_EQUALITY)
    #= A connection's equality, as the frontend's DAE conversion made it: a + (-b) = 0, of the
       type of `a` with its subscripts (the reference's own is its last part's declared one:
       `vol.ports[1].Xi_outflow[1]` of a scalarized kept array would be `Real[1]`) =#
    local ty = lowerType(F.getSubscriptedType(eq.lhs))
    local a = IR.CREF(lowerCref(eq.lhs), ty)
    local b = IR.CREF(lowerCref(eq.rhs), ty)
    return IR.SCALAR_EQUATION(IR.BINARY(a, IR.ADD, IR.UNARY(IR.UMINUS, b, ty), ty), IR.REAL(0.0), ty, source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_FOR)
    local name = Symbol(F.name(eq.iterator))
    MetaModelica.isSome(eq.range) || error("OMJLIR lowering: a for-equation without a range")
    local iterator = IR.Iterator(name, lowerExp(eq.range.data))
    return IR.FOR_EQUATION(iterator, lowerEquations(eq.body; reversed = true), source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_IF)
    return lowerIfEquation(eq, source)
  elseif isvariant(eq, F.EQUATION_WHEN)
    return lowerWhenEquation(eq, source)
  elseif isvariant(eq, F.EQUATION_ASSERT)
    return IR.ASSERT_EQUATION(lowerExp(eq.condition), lowerExp(eq.message), lowerExp(eq.level), source)
  elseif isvariant(eq, F.EQUATION_TERMINATE)
    unsupported("terminate() in an equation section outside a when-equation", F.toString(eq))
  elseif isvariant(eq, F.EQUATION_REINIT)
    unsupported("reinit() outside a when-equation (MLS 8.3.6)", F.toString(eq))
  elseif isvariant(eq, F.EQUATION_NORETCALL)
    return callEquation(lowerExp(eq.exp), source)
  elseif isvariant(eq, F.EQUATION_RECONFIGURE)
    return lowerReconfigureEquation(eq, source)
  end
  #= EQUATION_CONNECT: the frontend's connection handling removed them =#
  return nothing
end

"`AssertionLevel.error`."
const ASSERTION_ERROR = IR.ENUM_LITERAL(IR.T_ENUMERATION(:AssertionLevel, [:error, :warning]), :error, 1)

"""
    callEquation(call, source)

A call of a Modelica function for its effects in an equation section (MSL Fluid's
checkBoundary, a print): an assert whose condition is the call, run where the asserts are
checked (after the initialization and after each step), its message the function's name (as
the previous backend had it). A built-in's call is not supported.
"""
function callEquation(call::IR.Expression, source::IR.Source)::IR.Equation
  (call isa IR.CALL && !call.attributes.builtin) || unsupported("this builtin call as an equation", IR.toString(call))
  return IR.ASSERT_EQUATION(call, IR.STRING(replace(String(call.name), "." => "_")), ASSERTION_ERROR, source)
end

"The number of equations of a record equation: one per field (`expandEquations` indexes them so), per element of an array of records."
recordSize(ty::IR.T_RECORD)::Int = length(ty.fields)
recordSize(ty::IR.T_ARRAY)::Int =
  recordSize(ty.elementType) * prod(something(IR.dimensionSize(d), 1) for d in ty.dimensions; init = 1)
recordSize(ty::IR.ModelicaType)::Int = 1

"""
An if-equation: the branches in order, a last branch of condition `true` as the else-branch
(as the frontend's DAE conversion did).
"""
function lowerIfEquation(eq::F.Equation, source::IR.Source)::IR.IF_EQUATION
  local branches = IR.EquationBranch[]
  local elseEquations = IR.Equation[]
  for (k, b) in enumerate(eq.branches)
    isvariant(b, F.EQUATION_BRANCH) || error("OMJLIR lowering: an if-equation's branch the frontend did not type")
    if k == length(eq.branches) && F.isTrue(b.condition)
      elseEquations = lowerEquations(b.body; reversed = true)
    else
      push!(branches, IR.EquationBranch(lowerExp(b.condition), lowerEquations(b.body; reversed = false)))
    end
  end
  return IR.IF_EQUATION(branches, elseEquations, source, ATTRIBUTES)
end

"""
    whenBranch(condition, body; dropInitialDisjuncts = true)

A when-branch of the condition `condition` (MLS 8.6): `initial()` and `terminal()` as members
of a vector condition are the branch's activity at the initialization and at the end, the
other members its triggers. In a when-equation an `initial()` disjunct of `or` is dropped: it
is true only at the initialization, where the when is not active, and false after (`when
initial() or x > 0.6` fires at x's crossing only, as in OpenModelica). A when-statement's
condition keeps it (the algorithm lifting reads it as false, as the previous backend did).
"""
function whenBranch(condition::IR.Expression, body::Vector{IR.Statement}; dropInitialDisjuncts::Bool = true)::IR.WhenBranch
  local triggers = IR.Expression[]
  local (atInitial, atTerminal) = (false, false)
  for member in (condition isa IR.ARRAY ? condition.elements : IR.Expression[condition])
    local m = dropInitialDisjuncts ? withoutInitialDisjunct(member) : member
    if IR.isBuiltinCall(m, :initial)
      atInitial = true
    elseif IR.isBuiltinCall(m, :terminal)
      atTerminal = true
    else
      push!(triggers, m)
    end
  end
  return IR.WhenBranch(triggers, atInitial, atTerminal, body)
end

"`e` without its `initial()` disjuncts (of `or`)."
function withoutInitialDisjunct(e::IR.Expression)::IR.Expression
  (e isa IR.LBINARY && e.op === IR.OR) || return e
  local l = withoutInitialDisjunct(e.exp1)
  local r = withoutInitialDisjunct(e.exp2)
  IR.isBuiltinCall(l, :initial) && return r
  IR.isBuiltinCall(r, :initial) && return l
  return IR.LBINARY(l, e.op, r)
end

"""
A when-equation: each branch's triggers and its equations as statements (an equality an
assignment, an if-equation an if-statement). One with a structural change (`recompilation`)
is a structural when-equation.
"""
function lowerWhenEquation(eq::F.Equation, source::IR.Source)::IR.WHEN_EQUATION
  local branches = IR.WhenBranch[]
  for b in eq.branches
    push!(branches, whenBranch(lowerExp(b.condition), whenStatements(b.body)))
  end
  local structural = any(br -> any(s -> s isa IR.RECOMPILATION || s isa IR.AGENTIC_RECOMPILATION, br.body), branches)
  return IR.WHEN_EQUATION(branches, structural ? IR.STRUCTURAL_WHEN : IR.REGULAR_WHEN, source, ATTRIBUTES)
end

"The equations of a when-equation's branch, or of an if-equation in it, as statements."
function whenStatements(eqs::Vector{F.Equation})::Vector{IR.Statement}
  local out = IR.Statement[]
  for eq in eqs
    local source = lowerElementSource(eq.source)
    if isvariant(eq, F.EQUATION_EQUALITY) || isvariant(eq, F.EQUATION_ARRAY_EQUALITY)
      push!(out, IR.ASSIGNMENT(lowerExp(eq.lhs), lowerExp(eq.rhs), source))
    elseif isvariant(eq, F.EQUATION_CREF_EQUALITY)
      #= of the type of `lhs` with its subscripts, as in an equation section (`lowerEquation`) =#
      local ty = lowerType(F.getSubscriptedType(eq.lhs))
      push!(out, IR.ASSIGNMENT(IR.CREF(lowerCref(eq.lhs), ty), IR.CREF(lowerCref(eq.rhs), ty), source))
    elseif isvariant(eq, F.EQUATION_REINIT)
      push!(out, IR.REINIT(lowerCref(F.toCref(eq.cref)), lowerExp(eq.reinitExp), source))
    elseif isvariant(eq, F.EQUATION_ASSERT)
      push!(out, IR.ASSERT(lowerExp(eq.condition), lowerExp(eq.message), lowerExp(eq.level), source))
    elseif isvariant(eq, F.EQUATION_TERMINATE)
      push!(out, IR.TERMINATE(lowerExp(eq.message), source))
    elseif isvariant(eq, F.EQUATION_NORETCALL)
      push!(out, noretcallStatement(eq.exp, source))
    elseif isvariant(eq, F.EQUATION_IF)
      local branches = IR.StatementBranch[]
      local elseBody = IR.Statement[]
      for (k, b) in enumerate(eq.branches)
        if k == length(eq.branches) && F.isTrue(b.condition)
          elseBody = whenStatements(b.body)
        else
          push!(branches, IR.StatementBranch(lowerExp(b.condition), whenStatements(b.body)))
        end
      end
      push!(out, IR.IF_STATEMENT(branches, elseBody, source))
    else
      #= An array, record or for-equation in a when-equation =#
      error("OMJLIR lowering: $(F.toString(eq)) in a when-equation is not supported")
    end
  end
  return out
end

"""
A call without outputs in a when-equation: OM.jl's structural changes `recompilation(c, v)`
and `agentic_recompilation(c...)`, or a call for its effects.
"""
function noretcallStatement(call::F.Expression, source::IR.Source)::IR.Statement
  local e = lowerExp(call)
  if e isa IR.CALL && e.name === :recompilation
    local (c, v) = e.arguments
    c isa IR.CREF || error("OMJLIR lowering: recompilation of $(IR.toString(c)), not a component")
    return IR.RECOMPILATION(c.cref, v)
  elseif e isa IR.CALL && e.name === :agentic_recompilation
    return IR.AGENTIC_RECOMPILATION(IR.ComponentRef[a.cref for a in e.arguments if a isa IR.CREF], nothing, nothing)
  end
  return IR.NORETCALL(e, source)
end

"""
OM.jl's reconfigure equation: a structural when-equation whose change an agent decides,
with the variables it may change, its prompt and the initial equations for the new model.
"""
function lowerReconfigureEquation(eq::F.Equation, source::IR.Source)::IR.WHEN_EQUATION
  local condition = lowerExp(eq.whenConditions[1])
  local components = IR.ComponentRef[IR.ComponentRef(Symbol(n)) for n in reconfigureVariableNames(eq.variables)]
  local prompt = MetaModelica.isSome(eq.prompt) ? promptString(eq.prompt.data) : nothing
  local initialEquations = MetaModelica.isSome(eq.initialEquations) ? serializeInitialEquations(eq.initialEquations.data) : nothing
  local body = IR.Statement[IR.AGENTIC_RECOMPILATION(components, prompt, initialEquations)]
  return IR.WHEN_EQUATION([whenBranch(condition, body)], IR.STRUCTURAL_WHEN, source, ATTRIBUTES)
end

"The prompt of a reconfigure equation: a string literal."
function promptString(e::F.Expression)::String
  e isa F.STRING_EXPRESSION || error("OMJLIR lowering: the prompt of a reconfigure equation is not a string literal")
  return e.value
end

"The names of the variables a reconfigure equation declares (Absyn element items)."
function reconfigureVariableNames(items::MetaModelica.List{Absyn.ElementItem})::Vector{String}
  local names = String[]
  for item in items
    for c in item.element.specification.components
      push!(names, c.component.name)
    end
  end
  return names
end

"""
The initial equations of a reconfigure equation (Absyn) as Modelica text, for the agent:
`assert(cond, msg)` calls with their arguments, the others as the parser prints them.
"""
function serializeInitialEquations(items::MetaModelica.List{Absyn.EquationItem})::String
  local parts = String[]
  for item in items
    local eq = item.equation_
    if nameof(typeof(eq)) === :EQ_NORETCALL
      push!(parts, string(Absyn.dumpCref(eq.functionName), "(", Absyn.dumpFunctionArgs(eq.functionArgs), ")"))
    else
      push!(parts, string(eq))
    end
  end
  return join(parts, "; ")
end

#= ==================================================================================== =#
#= Variables                                                                            =#
#= ==================================================================================== =#

"The structural kind of a variable of variability `variability` and type `ty`."
function variableKind(variability::Integer, ty::IR.ModelicaType)::IR.VariableKind
  local V = F.Variability
  variability == V.CONSTANT && return IR.CONSTANT()
  variability in (V.STRUCTURAL_PARAMETER, V.PARAMETER, V.NON_STRUCTURAL_PARAMETER) && return IR.PARAMETER()
  IR.elementType(ty) isa IR.T_CLOCK && return IR.CLOCK_VARIABLE()
  variability == V.DISCRETE && return IR.DISCRETE()
  return IR.ALGEBRAIC()
end

"The causality of a top-level variable."
function lowerDirection(direction::Integer)::IR.Direction
  direction == F.Direction.INPUT && return IR.INPUT
  direction == F.Direction.OUTPUT && return IR.OUTPUT
  return IR.NO_DIRECTION
end

"The connector prefix of a variable, from the frontend's connector type bits."
function lowerConnector(cty::Integer)::IR.ConnectorKind
  local C = F.ConnectorType
  (cty & C.POTENTIAL) > 0 && return IR.POTENTIAL
  (cty & C.FLOW) > 0 && return IR.FLOW
  (cty & C.STREAM) > 0 && return IR.STREAM
  return IR.NOT_CONNECTOR
end

"A binding's expression: of a typed or flattened binding; `nothing` for one evaluated or none."
function lowerBinding(b::F.Binding)::Union{Nothing, IR.Expression}
  (isvariant(b, F.TYPED_BINDING) || isvariant(b, F.FLAT_BINDING)) || return nothing
  return lowerExp(b.bindingExp)
end

const STATE_SELECTS = Dict("never" => IR.NEVER, "avoid" => IR.AVOID, "default" => IR.DEFAULT,
                           "prefer" => IR.PREFER, "always" => IR.ALWAYS)

"StateSelect: a literal of the enumeration; `nothing` for another expression (a parameter)."
function lowerStateSelect(b::F.Binding)::Union{Nothing, IR.StateSelect}
  local e = F.getBindingExp(F.getTypedExp(b))
  local member = e isa F.ENUM_LITERAL_EXPRESSION ? e.name :
                 e isa F.CREF_EXPRESSION && isvariant(e.cref, F.COMPONENT_REF_CREF) ? F.name(e.cref.node) : ""
  local s = get(STATE_SELECTS, member, nothing)
  s === nothing && @warn "OMJLIR lowering: StateSelect $(F.toString(e)) is not a literal of StateSelect, ignored"
  return s
end

"The built-in attributes of a variable (MLS 4.8), from the frontend's type attributes."
function lowerAttributes(typeAttributes::Vector{Tuple{String, F.Binding}})::IR.VariableAttributes
  isempty(typeAttributes) && return IR.NO_ATTRIBUTES
  local values = Dict{String, Any}()
  for (name, b) in typeAttributes
    values[name] = name == "stateSelect" ? lowerStateSelect(b) :
                   name == "unbounded" ? nothing : lowerExp(F.getTypedExp(b))
  end
  local get_(n) = get(values, n, nothing)
  return IR.VariableAttributes(get_("start"), get_("fixed"), get_("nominal"), get_("min"), get_("max"),
                               get_("unit"), get_("displayUnit"), get_("quantity"), get_("stateSelect"))
end

"The OMJLIR variable of the frontend variable `v`."
function lowerVariable(v::F.Variable)::IR.Variable
  local cref = lowerCref(v.name)
  local ty = lowerType(v.ty)
  local a = v.attributes
  return IR.Variable(cref, ty, variableKind(a.variability, ty), lowerBinding(v.binding),
                     lowerAttributes(v.typeAttributes), lowerDirection(a.direction),
                     v.visibility == F.Visibility.PROTECTED, lowerConnector(a.connectorType), lowerSource(v.info),
                     :model)
end

#= ==================================================================================== =#
#= The model                                                                            =#
#= ==================================================================================== =#

"The algorithm sections as ALGORITHM equations, their assigned variables their outputs."
function lowerAlgorithms(algorithms::Vector{F.Algorithm})::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms
    local statements = lowerStatements(alg.statements)
    push!(out, IR.ALGORITHM(statements, assignedCrefs(statements), lowerElementSource(alg.source), ATTRIBUTES))
  end
  return out
end

"The variables `statements` assign, in the order of their first assignment."
function assignedCrefs(statements::Vector{IR.Statement})::Vector{IR.ComponentRef}
  local out = IR.ComponentRef[]
  local seen = Set{Symbol}()
  local visit(s::IR.ASSIGNMENT) = foreach(t -> (t isa IR.CREF && !(t.cref.name in seen)) && (push!(seen, t.cref.name); push!(out, IR.ComponentRef(t.cref.name))),
                                         s.lhs isa IR.TUPLE ? s.lhs.elements : IR.Expression[s.lhs])
  local visit(s::Union{IR.IF_STATEMENT, IR.WHEN_STATEMENT}) =
    (foreach(b -> foreach(visit, b.body), s.branches); s isa IR.IF_STATEMENT && foreach(visit, s.elseBody))
  local visit(s::Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT}) = foreach(visit, s.body)
  local visit(s::IR.Statement) = nothing
  foreach(visit, statements)
  return out
end

"""
    lowerSystem(equations, initialEquations, algorithms, initialAlgorithms)

A continuous-time partition: the equations, then the algorithm sections; the initial
equations, then the initial algorithm sections.
"""
function lowerSystem(eqs::Vector{F.Equation}, initialEqs::Vector{F.Equation}, algorithms::Vector{F.Algorithm},
                     initialAlgorithms::Vector{F.Algorithm})::IR.Partition
  local equations = lowerEquations(eqs; reversed = true)
  append!(equations, lowerAlgorithms(algorithms))
  local initialEquations = lowerEquations(initialEqs; reversed = true)
  append!(initialEquations, lowerAlgorithms(initialAlgorithms))
  local unknowns = Symbol[]
  return IR.Partition(1, IR.CONTINUOUS_PARTITION(), unknowns, equations, initialEquations)
end

"The variables of a flat model."
function lowerVariables(fm::F.FlatModel)::IR.VariableSet
  local variables = IR.VariableSet()
  for v in fm.variables
    local lowered = lowerVariable(v)
    #= The frontend emits an inner/outer component's variable twice (the previous backend
       kept the first, or the one with a start or fixed attribute) =#
    if haskey(variables, lowered.name)
      local kept = variables[lowered.name]
      local hasStartOrFixed(x) = x.attributes.start !== nothing || x.attributes.fixed !== nothing
      !hasStartOrFixed(kept) && hasStartOrFixed(lowered) &&
        (variables.variables[variables.index[lowered.name]] = lowered)
      continue
    end
    push!(variables, lowered)
  end
  return variables
end

"""
    structuralCall(eq)

OM.jl's structural states and transitions, `initialStructuralState(s)` and
`structuralTransition(from, to, condition)` in the equation section: their arguments, or
`nothing` for another equation.
"""
function structuralCall(eq::F.Equation)::Union{Nothing, Tuple{Symbol, Vector{Any}}}
  isvariant(eq, F.EQUATION_NORETCALL) && eq.exp isa F.CALL_EXPRESSION && isvariant(eq.exp.call, F.TYPED_CALL) || return nothing
  local name = Symbol(F.AbsynUtil.pathString(F.nameConsiderBuiltin(eq.exp.call.fn)))
  name in (:initialStructuralState, :structuralTransition) || return nothing
  return (name, collect(eq.exp.call.arguments))
end

#= ==================================================================================== =#
#= Functions                                                                            =#
#= ==================================================================================== =#

"""
    lowerFunctions(functions)

The functions the model calls (the frontend's list, in its order) in OMJLIR.
"""
lowerFunctions(functions::FunctionList)::Vector{IR.ModelicaFunction} =
  IR.ModelicaFunction[lowerFunction(f) for f in functions]

"""
    lowerFunction(fn)

A function in OMJLIR: its variables, its algorithm or its external clause, its derivative
annotations of order 1 (the inputs zeroDerivative or noDerivative exclude by position), its
prefixes.
"""
function lowerFunction(fn::F.M_FUNCTION)::IR.ModelicaFunction
  local external = F.isExternal(fn) ? lowerExternal(fn) : nothing
  local body = external === nothing ? lowerStatements(F.getBody(fn)) : IR.Statement[]
  local derivatives = IR.FunctionDerivative[]
  for d in fn.derivatives
    (isvariant(d.order, F.INTEGER_EXPRESSION) && d.order.value == 1) || continue
    local fns = F.getCachedFuncs(d.derivativeFn)
    isempty(fns) && continue
    push!(derivatives, IR.FunctionDerivative(pathName(F.name(first(fns))), 1, Int[c[1] for c in d.conditions]))
  end
  local inline = nameof(typeof(fn.attributes.inline)) in (:EARLY_INLINE, :NORM_INLINE)
  return IR.ModelicaFunction(pathName(fn.path), functionVariables(fn.inputs), functionVariables(fn.outputs),
                             functionVariables(fn.locals), body, external, derivatives, F.isImpure(fn), inline,
                             IR.NO_SOURCE)
end

"The variables of a function's inputs, outputs or protected section."
functionVariables(nodes::Union{AbstractVector{<:F.InstNode}, MetaModelica.List{<:F.InstNode}})::Vector{IR.FunctionVariable} =
  IR.FunctionVariable[functionVariable(n) for n in nodes]

"""
    functionVariable(node)

A function's variable: its name, type, binding, and for a record instance its fields'
bindings (the instance's modifiers and the record's defaults).
"""
function functionVariable(node::F.InstNode)::IR.FunctionVariable
  local comp = F.component(node)
  local ty = F.getType(comp)
  local fieldBindings = Pair{Symbol, IR.Expression}[]
  if isvariant(ty, F.TYPE_COMPLEX)
    local cls = F.getClass(ty.cls)
    if cls isa F.INSTANCED_CLASS && cls.restriction isa F.RESTRICTION_RECORD
      for c in F.getComponents(cls.elements)
        local b = lowerBinding(F.getBinding(F.component(c)))
        b === nothing || push!(fieldBindings, Symbol(F.name(c)) => b)
      end
    end
  end
  return IR.FunctionVariable(Symbol(F.name(node)), lowerType(ty), lowerBinding(comp.binding), fieldBindings)
end

"""
    lowerExternal(fn)

An external function's clause: the external function's name, language, arguments and
output, and its annotation as the flat model writes it (with the default
`LibraryDirectory` and `IncludeDirectory` of the library).
"""
function lowerExternal(fn::F.M_FUNCTION)::IR.ExternalDeclaration
  local sections = F.getSections(F.getClass(fn.node))
  local output = F.isEmpty(sections.outputRef) ? nothing : lowerCref(sections.outputRef)
  local s = F.IOStream_M.create("external", F.IOStream_M.LIST())
  local flat = F.IOStream_M.string(F.toFlatStream(sections, fn.path, s; inFunction = true))
  local at = findfirst("annotation", flat)
  return IR.ExternalDeclaration(sections.name, sections.language, lowerExps(collect(F.NFExpression, sections.args)), output,
                                at === nothing ? "" : flat[first(at):end])
end

"""
    lowerFlatModel(fm)

The OMJLIR model of the frontend's flat model `fm`: its variables, its continuous-time
partition (equations, algorithms, initial equations and algorithms), its structural modes
(the structural submodels), states and transitions; and the functions of `functions` (the
frontend's function list). The frontend model to translate again at a structural change
is the SCode class the frontend kept for a model with recompilations (`nothing` for another).
"""
function lowerFlatModel(fm::F.FlatModel; functions::FunctionList = MetaModelica.nil)::IR.Model
  local variables = lowerVariables(fm)
  local initialMode = nothing
  local transitions = IR.StructuralTransition[]
  local equations = F.NFEquationImpl[]
  for eq in fm.equations
    local sc = structuralCall(eq)
    if sc === nothing
      push!(equations, eq)
    elseif sc[1] === :initialStructuralState
      initialMode = F.toString(sc[2][1])
    else
      local (from, to, condition) = sc[2]
      push!(transitions, IR.StructuralTransition(F.toString(from), F.toString(to), lowerExp(condition)))
    end
  end
  local system = lowerSystem(equations, fm.initialEquations, fm.algorithms, fm.initialAlgorithms)
  local modes = IR.StructuralMode[]
  for sub in fm.structuralSubmodels
    push!(modes, IR.StructuralMode(sub.name, lowerVariables(sub),
                                   lowerSystem(sub.equations, sub.initialEquations, sub.algorithms,
                                               sub.initialAlgorithms)))
  end
  return IR.Model(fm.name, variables, IR.Partition[system], IR.Partition[], IR.ClockKind[],
                  lowerFunctions(functions), initialMode, modes, transitions,
                  fm.scodeProgram === nothing ? nothing : fm.scodeProgram.data)
end

end # module LowerFlatModel
