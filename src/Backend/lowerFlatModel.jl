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
reverse. The variables are in their order.

Component references: a reference to a variable of the model is its flat name with the
subscripts that are not part of the name (an element `x[2]` of a scalarized array is a
variable of its own, `x[i]` of an array variable `x` is `x` subscripted). In a function, a
reference to a field of a record is a RECORD_ELEMENT of the record.
"""
module LowerFlatModel

import Absyn
import MetaModelica
import MetaModelica: isvariant
import OMFrontend
import ..OMJLIR

const F = OMFrontend.Frontend
const IR = OMJLIR

export lowerFlatModel

"""
    Scope(variables, iterators)

What a component reference can name where an expression is lowered: the variables (of the
model, or the inputs, outputs and locals of a function) and the iterators of the enclosing
for-loops, array constructors and reductions. `inFunction`: references to records are not
split into field variables.
"""
struct Scope
  variables::Set{Symbol}
  iterators::Vector{Symbol}
  inFunction::Bool
end

"`scope` with `name` as an iterator."
withIterator(scope::Scope, name::Symbol)::Scope =
  Scope(scope.variables, vcat(scope.iterators, name), scope.inFunction)

#= ==================================================================================== =#
#= Names and sources                                                                    =#
#= ==================================================================================== =#

"The dotted name of a class or function path."
pathName(path)::Symbol = Symbol(F.AbsynUtil.pathString(path))

"The location in the Modelica source, from a frontend `SourceInfo`."
lowerSource(info)::IR.Source = IR.Source(info.fileName, info.lineNumberStart, info.columnNumberStart)

"The location of an equation or a statement, from the frontend's element source."
lowerElementSource(source)::IR.Source = lowerSource(source.info)

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

"A record type with its fields, or an external object class (MLS 12.9.7)."
function complexType(ty::F.NFType)::IR.ModelicaType
  local cls = ty.cls
  local name = pathName(F.scopePath(cls; includeRoot = true))
  isvariant(ty.complexTy, F.COMPLEX_EXTERNAL_OBJECT) && return IR.T_EXTERNAL_OBJECT(name)
  local fields = IR.Field[]
  for c in F.getComponents(F.getClass(cls).elements)
    F.isOnlyOuter(c) && continue
    push!(fields, IR.Field(Symbol(F.name(c)), lowerType(F.getType(F.component(c)))))
  end
  return IR.T_RECORD(name, fields)
end

"The type of a function value: its name, its inputs' and its outputs' types."
function functionType(fn::F.M_Function)::IR.T_FUNCTION
  local inputs = IR.ModelicaType[lowerType(F.getType(F.component(n))) for n in fn.inputs]
  local outputs = IR.ModelicaType[lowerType(F.getType(F.component(n))) for n in fn.outputs]
  return IR.T_FUNCTION(pathName(F.nameConsiderBuiltin(fn)), inputs, outputs)
end

"An OMJLIR dimension; Boolean and enumeration dimensions have the number of their values."
function lowerDimension(d)::IR.Dimension
  isvariant(d, F.DIMENSION_INTEGER) && return IR.DIM_SIZE(d.size)
  isvariant(d, F.DIMENSION_BOOLEAN) && return IR.DIM_SIZE(2)
  isvariant(d, F.DIMENSION_ENUM) && return IR.DIM_SIZE(length(d.enumType.literals))
  isvariant(d, F.DIMENSION_EXP) && return IR.DIM_EXP(lowerExp(d.exp, Scope(Set{Symbol}(), Symbol[], false)))
  return IR.DIM_UNKNOWN()
end

function lowerSubscript(s, scope::Scope)::IR.Subscript
  s isa F.SUBSCRIPT_INDEX && return IR.INDEX(lowerExp(s.index, scope))
  s isa F.SUBSCRIPT_SLICE && return IR.SLICE(lowerExp(s.slice, scope))
  s isa F.SUBSCRIPT_WHOLE && return IR.WHOLE()
  error("OMJLIR lowering: subscript $(F.toString(s)) of an untyped or split form")
end

#= ==================================================================================== =#
#= Component references                                                                 =#
#= ==================================================================================== =#

"The parts of the frontend component reference `cref`, the outermost first."
function crefParts(cref)::Vector{Any}
  local parts = Any[]
  while isvariant(cref, F.COMPONENT_REF_CREF)
    push!(parts, cref)
    cref = cref.restCref
  end
  return reverse!(parts)
end

"The name of a subscript in a variable's name: a literal index (`2`, `true`, `E.one`)."
function subscriptName(s)::Union{Nothing, String}
  s isa F.SUBSCRIPT_INDEX || return nothing
  local e = s.index
  e isa F.INTEGER_EXPRESSION && return string(e.value)
  e isa F.BOOLEAN_EXPRESSION && return string(e.value)
  e isa F.ENUM_LITERAL_EXPRESSION && return string(F.AbsynUtil.pathString(e.ty.typePath), ".", e.name)
  return nothing
end

"""
    crefName(parts; withSubscripts)

The flat name of the parts `a.b[2].c`; without its subscripts `a.b.c`. `nothing` if a
subscript to be named is not a literal.
"""
function crefName(parts::Vector{Any}; withSubscripts::Bool)::Union{Nothing, Symbol}
  local io = IOBuffer()
  for (i, p) in enumerate(parts)
    i > 1 && print(io, '.')
    print(io, F.name(p.node))
    if withSubscripts && !isempty(p.subscripts)
      print(io, '[')
      for (j, s) in enumerate(p.subscripts)
        local n = subscriptName(s)
        n === nothing && return nothing
        j > 1 && print(io, ',')
        print(io, n)
      end
      print(io, ']')
    end
  end
  return Symbol(String(take!(io)))
end

"""
    lowerCrefExp(cref, ty, scope)

A reference to a variable: of the model, by its flat name, with the subscripts that are not
part of it; of a function, the variable subscripted and then its record fields.
"""
function lowerCrefExp(cref, ty::IR.ModelicaType, scope::Scope)::IR.Expression
  isvariant(cref, F.COMPONENT_REF_WILD) && return IR.WILD()
  local parts = crefParts(cref)
  scope.inFunction && return functionCref(parts, ty, scope)
  #= An element of a scalarized array, a scalar, an iterator, `time`: a name =#
  local full = crefName(parts; withSubscripts = true)
  if full !== nothing && (full in scope.variables || full in scope.iterators || isempty(scope.variables))
    return IR.CREF(IR.ComponentRef(full), ty)
  end
  #= An array variable subscripted (a field of an array of records `r[i].v`: the field's
     array variable `r.v`, subscripted) =#
  local base = crefName(parts; withSubscripts = false)
  local subs = IR.Subscript[lowerSubscript(s, scope) for p in parts for s in p.subscripts]
  base in scope.variables && return IR.CREF(IR.ComponentRef(base, subs), ty)
  #= Not a variable of the model: a built-in variable (time) or a name the frontend left
     (an enumeration type, a function), by its name as written =#
  return full !== nothing ? IR.CREF(IR.ComponentRef(full), ty) : IR.CREF(IR.ComponentRef(base, subs), ty)
end

"""
A reference in a function: its variable, subscripted, then the record fields after it, each
subscripted (`r[i].x[2]`). A part's type is its component's, before its subscripts.
"""
function functionCref(parts::Vector{Any}, ty::IR.ModelicaType, scope::Scope)::IR.Expression
  local exp::IR.Expression = IR.WILD()
  for (i, p) in enumerate(parts)
    local subs = IR.Subscript[lowerSubscript(s, scope) for s in p.subscripts]
    local partTy = lowerType(p.ty)
    local resultTy = i == length(parts) ? ty : afterSubscripts(partTy, subs)
    if i == 1
      exp = IR.CREF(IR.ComponentRef(Symbol(F.name(p.node)), subs), resultTy)
      continue
    end
    local field = Symbol(F.name(p.node))
    local recTy = IR.typeOf(exp)
    local index = recTy isa IR.T_RECORD ? something(findfirst(f -> f.name === field, recTy.fields), 0) : 0
    exp = IR.RECORD_ELEMENT(exp, index, field, isempty(subs) ? resultTy : partTy)
    isempty(subs) || (exp = IR.SUBSCRIPTED(exp, subs, resultTy))
  end
  return exp
end

"The type of a value of type `ty` subscripted by `subs`: an index removes its dimension."
function afterSubscripts(ty::IR.ModelicaType, subs::Vector{IR.Subscript})::IR.ModelicaType
  (ty isa IR.T_ARRAY && !isempty(subs)) || return ty
  local dims = IR.Dimension[d for (k, d) in enumerate(ty.dimensions) if k > length(subs) || !(subs[k] isa IR.INDEX)]
  return isempty(dims) ? ty.elementType : IR.T_ARRAY(ty.elementType, dims)
end

"The component reference of a `reinit` target or an assignment's left-hand side."
function lowerCref(cref, scope::Scope)::IR.ComponentRef
  local e = lowerCrefExp(cref, lowerType(cref.ty), scope)
  e isa IR.CREF || error("OMJLIR lowering: $(F.toString(cref)) is not a reference to a variable")
  return e.cref
end

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

function lowerOp(operator)::IR.Op
  local op = get(OPS, Int(operator.op), nothing)
  op === nothing && error("OMJLIR lowering: a user-defined operator the frontend did not resolve")
  return op
end

typeOfExp(e)::IR.ModelicaType = lowerType(F.typeOf(e))

lowerExps(es, scope::Scope)::Vector{IR.Expression} = IR.Expression[lowerExp(e, scope) for e in es]

"An optional expression (a MetaModelica `Option`): lowered, or `nothing`."
lowerOptExp(o, scope::Scope)::Union{Nothing, IR.Expression} =
  o isa MetaModelica.SOME ? lowerExp(o.data, scope) : nothing

"The OMJLIR expression of the frontend expression `e`."
lowerExp(e::F.INTEGER_EXPRESSION, scope::Scope)::IR.Expression = IR.INTEGER(e.value)
lowerExp(e::F.REAL_EXPRESSION, scope::Scope)::IR.Expression = IR.REAL(e.value)
lowerExp(e::F.STRING_EXPRESSION, scope::Scope)::IR.Expression = IR.STRING(e.value)
lowerExp(e::F.BOOLEAN_EXPRESSION, scope::Scope)::IR.Expression = IR.BOOLEAN(e.value)
lowerExp(e::F.ENUM_LITERAL_EXPRESSION, scope::Scope)::IR.Expression =
  IR.ENUM_LITERAL(enumerationType(e.ty), Symbol(e.name), e.index)
lowerExp(e::F.CLKCONST_EXPRESSION, scope::Scope)::IR.Expression = IR.CLOCK(lowerClock(e.clk, scope))
lowerExp(e::F.CREF_EXPRESSION, scope::Scope)::IR.Expression = lowerCrefExp(e.cref, lowerType(e.ty), scope)
lowerExp(e::F.TYPENAME_EXPRESSION, scope::Scope)::IR.Expression = lowerExp(F.expandTypename(e.ty), scope)
lowerExp(e::F.ARRAY_EXPRESSION, scope::Scope)::IR.Expression = IR.ARRAY(lowerExps(e.elements, scope), lowerType(e.ty))
function lowerExp(e::F.MATRIX_EXPRESSION, scope::Scope)::IR.Expression
  local ty = typeOfExp(e)::IR.T_ARRAY
  local rowTy = IR.T_ARRAY(ty.elementType, ty.dimensions[2:end])
  return IR.ARRAY(IR.Expression[IR.ARRAY(lowerExps(row, scope), rowTy) for row in e.elements], ty)
end
lowerExp(e::F.RECORD_EXPRESSION, scope::Scope)::IR.Expression = IR.RECORD(lowerExps(e.elements, scope), lowerType(e.ty))
lowerExp(e::F.RANGE_EXPRESSION, scope::Scope)::IR.Expression =
  IR.RANGE(lowerExp(e.start, scope), lowerOptExp(e.step, scope), lowerExp(e.stop, scope), lowerType(e.ty))
lowerExp(e::F.TUPLE_EXPRESSION, scope::Scope)::IR.Expression = IR.TUPLE(lowerExps(e.elements, scope), lowerType(e.ty))
lowerExp(e::F.CALL_EXPRESSION, scope::Scope)::IR.Expression = lowerCall(e.call, scope)
lowerExp(e::F.SIZE_EXPRESSION, scope::Scope)::IR.Expression = IR.SIZE(lowerExp(e.exp, scope), lowerOptExp(e.dimIndex, scope))
lowerExp(e::F.BINARY_EXPRESSION, scope::Scope)::IR.Expression =
  IR.BINARY(lowerExp(e.exp1, scope), lowerOp(e.operator), lowerExp(e.exp2, scope), lowerType(e.operator.ty))
lowerExp(e::F.UNARY_EXPRESSION, scope::Scope)::IR.Expression =
  IR.UNARY(lowerOp(e.operator), lowerExp(e.exp, scope), lowerType(e.operator.ty))
lowerExp(e::F.LBINARY_EXPRESSION, scope::Scope)::IR.Expression =
  IR.LBINARY(lowerExp(e.exp1, scope), lowerOp(e.operator), lowerExp(e.exp2, scope))
lowerExp(e::F.LUNARY_EXPRESSION, scope::Scope)::IR.Expression = IR.LUNARY(lowerOp(e.operator), lowerExp(e.exp, scope))
lowerExp(e::F.RELATION_EXPRESSION, scope::Scope)::IR.Expression =
  IR.RELATION(lowerExp(e.exp1, scope), lowerOp(e.operator), lowerExp(e.exp2, scope), 0)
lowerExp(e::F.IF_EXPRESSION, scope::Scope)::IR.Expression =
  IR.IF_EXPRESSION(lowerExp(e.condition, scope), lowerExp(e.trueBranch, scope), lowerExp(e.falseBranch, scope), typeOfExp(e))
lowerExp(e::F.CAST_EXPRESSION, scope::Scope)::IR.Expression = IR.CAST(lowerExp(e.exp, scope), lowerType(e.ty))
lowerExp(e::F.SUBSCRIPTED_EXP_EXPRESSION, scope::Scope)::IR.Expression =
  IR.SUBSCRIPTED(lowerExp(e.exp, scope), IR.Subscript[lowerSubscript(s, scope) for s in e.subscripts], lowerType(e.ty))
lowerExp(e::F.TUPLE_ELEMENT_EXPRESSION, scope::Scope)::IR.Expression =
  IR.TUPLE_ELEMENT(lowerExp(e.tupleExp, scope), e.index, lowerType(e.ty))
lowerExp(e::F.RECORD_ELEMENT_EXPRESSION, scope::Scope)::IR.Expression =
  IR.RECORD_ELEMENT(lowerExp(e.recordExp, scope), e.index, Symbol(e.fieldName), lowerType(e.ty))
function lowerExp(e::F.PARTIAL_FUNCTION_APPLICATION_EXPRESSION, scope::Scope)::IR.Expression
  local fn = F.typeRefCache(e.fn)[1]
  return IR.PARTIAL_APPLICATION(pathName(F.nameConsiderBuiltin(fn)), lowerExps(e.args, scope),
                                Symbol[Symbol(n) for n in e.argNames], lowerType(e.ty))
end
lowerExp(e::F.BINDING_EXP, scope::Scope)::IR.Expression = lowerExp(e.exp, scope)
#= Boxing (of a function's polymorphic or functional arguments) has no meaning in Julia, where
   every value is a value: the code generation always took the boxed expression itself. =#
lowerExp(e::F.BOX_EXPRESSION, scope::Scope)::IR.Expression = lowerExp(e.exp, scope)
lowerExp(e::F.UNBOX_EXPRESSION, scope::Scope)::IR.Expression = lowerExp(e.exp, scope)
lowerExp(e::F.MUTABLE_EXPRESSION, scope::Scope)::IR.Expression = lowerExp(F.P_Pointer.access(e.exp), scope)
lowerExp(e::F.NFExpression, scope::Scope)::IR.Expression =
  error("OMJLIR lowering: the expression $(F.toString(e)) ($(nameof(typeof(e)))) has no OMJLIR form")

"`inline`: the frontend's inline types that inline the function."
const INLINE_TYPES = (:NORM_INLINE, :BUILTIN_EARLY_INLINE, :EARLY_INLINE)

function lowerCall(call, scope::Scope)::IR.Expression
  if isvariant(call, F.TYPED_CALL)
    local a = call.attributes
    local attributes = IR.CallAttributes(a.builtin, a.isImpure, nameof(typeof(a.inlineType)) in INLINE_TYPES)
    return IR.CALL(pathName(F.nameConsiderBuiltin(call.fn)), lowerExps(call.arguments, scope),
                   lowerType(call.ty), attributes)
  elseif isvariant(call, F.TYPED_ARRAY_CONSTRUCTOR)
    local (iterators, inner) = lowerIterators(call.iters, scope)
    return IR.ARRAY_CONSTRUCTOR(lowerExp(call.exp, inner), iterators, lowerType(call.ty))
  elseif isvariant(call, F.TYPED_REDUCTION)
    local (iterators, inner) = lowerIterators(call.iters, scope)
    return IR.REDUCTION(pathName(F.nameConsiderBuiltin(call.fn)), lowerExp(call.exp, inner), iterators, lowerType(call.ty))
  end
  error("OMJLIR lowering: an untyped call")
end

"The iterators of an array constructor or a reduction, and the scope of its body."
function lowerIterators(iters, scope::Scope)::Tuple{Vector{IR.Iterator}, Scope}
  local iterators = IR.Iterator[]
  local inner = scope
  for (node, range) in iters
    local name = Symbol(F.name(node))
    push!(iterators, IR.Iterator(name, lowerExp(range, inner)))
    inner = withIterator(inner, name)
  end
  return (iterators, inner)
end

lowerClock(k::F.INFERRED_CLOCK, scope::Scope)::IR.ClockKind = IR.INFERRED_CLOCK()
lowerClock(k::F.INTEGER_CLOCK, scope::Scope)::IR.ClockKind =
  IR.INTEGER_CLOCK(lowerExp(k.intervalCounter, scope), lowerExp(k.resolution, scope))
lowerClock(k::F.REAL_CLOCK, scope::Scope)::IR.ClockKind = IR.REAL_CLOCK(lowerExp(k.interval, scope))
lowerClock(k::F.BOOLEAN_CLOCK, scope::Scope)::IR.ClockKind =
  IR.BOOLEAN_CLOCK(lowerExp(k.condition, scope), lowerExp(k.startInterval, scope))
lowerClock(k::F.SOLVER_CLOCK, scope::Scope)::IR.ClockKind =
  IR.SOLVER_CLOCK(lowerExp(k.c, scope), lowerExp(k.solverMethod, scope))

#= ==================================================================================== =#
#= Statements                                                                           =#
#= ==================================================================================== =#

lowerStatements(stmts, scope::Scope)::Vector{IR.Statement} = IR.Statement[lowerStatement(s, scope) for s in stmts]

"The OMJLIR statement of the frontend statement `s`."
function lowerStatement(s, scope::Scope)::IR.Statement
  local source = lowerElementSource(s.source)
  if isvariant(s, F.ALG_ASSIGNMENT)
    return IR.ASSIGNMENT(lowerExp(s.lhs, scope), lowerExp(s.rhs, scope), source)
  elseif isvariant(s, F.ALG_IF)
    local branches = IR.StatementBranch[]
    local elseBody = IR.Statement[]
    for (k, (cond, body)) in enumerate(s.branches)
      if k == length(s.branches) && F.isTrue(cond)
        elseBody = lowerStatements(body, scope)
      else
        push!(branches, IR.StatementBranch(lowerExp(cond, scope), lowerStatements(body, scope)))
      end
    end
    return IR.IF_STATEMENT(branches, elseBody, source)
  elseif isvariant(s, F.ALG_FOR)
    local name = Symbol(F.name(s.iterator))
    MetaModelica.isSome(s.range) || error("OMJLIR lowering: a for-statement without a range")
    local iterator = IR.Iterator(name, lowerExp(s.range.data, scope))
    return IR.FOR_STATEMENT(iterator, lowerStatements(s.body, withIterator(scope, name)), source)
  elseif isvariant(s, F.ALG_WHILE)
    return IR.WHILE_STATEMENT(lowerExp(s.condition, scope), lowerStatements(s.body, scope), source)
  elseif isvariant(s, F.ALG_WHEN)
    return IR.WHEN_STATEMENT(IR.StatementBranch[IR.StatementBranch(lowerExp(c, scope), lowerStatements(b, scope))
                                                for (c, b) in s.branches], source)
  elseif isvariant(s, F.ALG_ASSERT)
    return IR.ASSERT(lowerExp(s.condition, scope), lowerExp(s.message, scope), lowerExp(s.level, scope), source)
  elseif isvariant(s, F.ALG_TERMINATE)
    return IR.TERMINATE(lowerExp(s.message, scope), source)
  elseif isvariant(s, F.ALG_REINIT)
    return IR.REINIT(lowerCref(F.toCref(s.cref), scope), lowerExp(s.reinitExp, scope), source)
  elseif isvariant(s, F.ALG_NORETCALL)
    return IR.NORETCALL(lowerExp(s.exp, scope), source)
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
    lowerEquations(eqs, scope; reversed)

The equations `eqs`, in reverse for the top-level, initial and else-branch equations (the
previous backend's order, see the module's documentation).
"""
function lowerEquations(eqs, scope::Scope; reversed::Bool)::Vector{IR.Equation}
  local out = IR.Equation[]
  for eq in (reversed ? Iterators.reverse(eqs) : eqs)
    local lowered = lowerEquation(eq, scope)
    lowered === nothing || push!(out, lowered)
  end
  return out
end

const ATTRIBUTES = IR.DEFAULT_ATTRIBUTES

"The OMJLIR equation of the frontend equation `eq`; `nothing` for a connect equation."
function lowerEquation(eq, scope::Scope)::Union{Nothing, IR.Equation}
  local source = lowerElementSource(eq.source)
  if isvariant(eq, F.EQUATION_EQUALITY) || isvariant(eq, F.EQUATION_ARRAY_EQUALITY)
    local ty = lowerType(eq.ty)
    local lhs = lowerExp(eq.lhs, scope)
    local rhs = lowerExp(eq.rhs, scope)
    F.isComplex(eq.ty) && return IR.RECORD_EQUATION(lhs, rhs, ty, recordSize(ty), source, ATTRIBUTES)
    ty isa IR.T_ARRAY && return IR.ARRAY_EQUATION(lhs, rhs, ty, source, ATTRIBUTES)
    return IR.SCALAR_EQUATION(lhs, rhs, ty, source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_CREF_EQUALITY)
    #= A connection's equality, as the frontend's DAE conversion made it: a + (-b) = 0 =#
    local ty = lowerType(eq.lhs.ty)
    local a = IR.CREF(lowerCref(eq.lhs, scope), ty)
    local b = IR.CREF(lowerCref(eq.rhs, scope), ty)
    return IR.SCALAR_EQUATION(IR.BINARY(a, IR.ADD, IR.UNARY(IR.UMINUS, b, ty), ty), IR.REAL(0.0), ty, source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_FOR)
    local name = Symbol(F.name(eq.iterator))
    MetaModelica.isSome(eq.range) || error("OMJLIR lowering: a for-equation without a range")
    local iterator = IR.Iterator(name, lowerExp(eq.range.data, scope))
    return IR.FOR_EQUATION(iterator, lowerEquations(eq.body, withIterator(scope, name); reversed = true), source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_IF)
    return lowerIfEquation(eq, source, scope)
  elseif isvariant(eq, F.EQUATION_WHEN)
    return lowerWhenEquation(eq, source, scope)
  elseif isvariant(eq, F.EQUATION_ASSERT)
    return IR.ASSERT_EQUATION(lowerExp(eq.condition, scope), lowerExp(eq.message, scope), lowerExp(eq.level, scope), source)
  elseif isvariant(eq, F.EQUATION_TERMINATE)
    return IR.ALGORITHM(IR.Statement[IR.TERMINATE(lowerExp(eq.message, scope), source)], IR.ComponentRef[], source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_REINIT)
    error("OMJLIR lowering: reinit outside a when-equation (MLS 8.3.6)")
  elseif isvariant(eq, F.EQUATION_NORETCALL)
    return IR.ALGORITHM(IR.Statement[IR.NORETCALL(lowerExp(eq.exp, scope), source)], IR.ComponentRef[], source, ATTRIBUTES)
  elseif isvariant(eq, F.EQUATION_RECONFIGURE)
    return lowerReconfigureEquation(eq, source, scope)
  end
  #= EQUATION_CONNECT: the frontend's connection handling removed them =#
  return nothing
end

"The number of scalar equations of a record equation: of a record type, its fields."
recordSize(ty::IR.T_RECORD)::Int = length(ty.fields)
recordSize(ty::IR.T_ARRAY)::Int =
  recordSize(ty.elementType) * prod(d isa IR.DIM_SIZE ? d.size : 1 for d in ty.dimensions; init = 1)
recordSize(ty::IR.ModelicaType)::Int = 1

"""
An if-equation: the branches in order, a last branch of condition `true` as the else-branch
(as the frontend's DAE conversion did).
"""
function lowerIfEquation(eq, source::IR.Source, scope::Scope)::IR.IF_EQUATION
  local branches = IR.EquationBranch[]
  local elseEquations = IR.Equation[]
  for (k, b) in enumerate(eq.branches)
    isvariant(b, F.EQUATION_BRANCH) || error("OMJLIR lowering: an if-equation's branch the frontend did not type")
    if k == length(eq.branches) && F.isTrue(b.condition)
      elseEquations = lowerEquations(b.body, scope; reversed = true)
    else
      push!(branches, IR.EquationBranch(lowerExp(b.condition, scope), lowerEquations(b.body, scope; reversed = false)))
    end
  end
  return IR.IF_EQUATION(branches, elseEquations, source, ATTRIBUTES)
end

"""
A when-equation: each branch's condition and its equations as statements (an equality an
assignment, an if-equation an if-statement). One with a structural change (`recompilation`)
is a structural when-equation.
"""
function lowerWhenEquation(eq, source::IR.Source, scope::Scope)::IR.WHEN_EQUATION
  local branches = IR.StatementBranch[]
  for b in eq.branches
    push!(branches, IR.StatementBranch(lowerExp(b.condition, scope), whenStatements(b.body, scope)))
  end
  local structural = any(br -> any(s -> s isa IR.RECOMPILATION || s isa IR.AGENTIC_RECOMPILATION, br.body), branches)
  return IR.WHEN_EQUATION(branches, structural ? IR.STRUCTURAL_WHEN : IR.REGULAR_WHEN, source, ATTRIBUTES)
end

"The equations of a when-equation's branch, or of an if-equation in it, as statements."
function whenStatements(eqs, scope::Scope)::Vector{IR.Statement}
  local out = IR.Statement[]
  for eq in eqs
    local source = lowerElementSource(eq.source)
    if isvariant(eq, F.EQUATION_EQUALITY) || isvariant(eq, F.EQUATION_ARRAY_EQUALITY)
      push!(out, IR.ASSIGNMENT(lowerExp(eq.lhs, scope), lowerExp(eq.rhs, scope), source))
    elseif isvariant(eq, F.EQUATION_CREF_EQUALITY)
      local ty = lowerType(eq.lhs.ty)
      push!(out, IR.ASSIGNMENT(IR.CREF(lowerCref(eq.lhs, scope), ty), IR.CREF(lowerCref(eq.rhs, scope), ty), source))
    elseif isvariant(eq, F.EQUATION_REINIT)
      push!(out, IR.REINIT(lowerCref(F.toCref(eq.cref), scope), lowerExp(eq.reinitExp, scope), source))
    elseif isvariant(eq, F.EQUATION_ASSERT)
      push!(out, IR.ASSERT(lowerExp(eq.condition, scope), lowerExp(eq.message, scope), lowerExp(eq.level, scope), source))
    elseif isvariant(eq, F.EQUATION_TERMINATE)
      push!(out, IR.TERMINATE(lowerExp(eq.message, scope), source))
    elseif isvariant(eq, F.EQUATION_NORETCALL)
      push!(out, noretcallStatement(eq.exp, source, scope))
    elseif isvariant(eq, F.EQUATION_IF)
      local branches = IR.StatementBranch[]
      local elseBody = IR.Statement[]
      for (k, b) in enumerate(eq.branches)
        if k == length(eq.branches) && F.isTrue(b.condition)
          elseBody = whenStatements(b.body, scope)
        else
          push!(branches, IR.StatementBranch(lowerExp(b.condition, scope), whenStatements(b.body, scope)))
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
function noretcallStatement(call, source::IR.Source, scope::Scope)::IR.Statement
  local e = lowerExp(call, scope)
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
function lowerReconfigureEquation(eq, source::IR.Source, scope::Scope)::IR.WHEN_EQUATION
  local condition = lowerExp(eq.whenConditions[1], scope)
  local components = IR.ComponentRef[IR.ComponentRef(Symbol(n)) for n in reconfigureVariableNames(eq.variables)]
  local prompt = MetaModelica.isSome(eq.prompt) ? promptString(eq.prompt.data) : nothing
  local initialEquations = MetaModelica.isSome(eq.initialEquations) ? serializeInitialEquations(eq.initialEquations.data) : nothing
  local body = IR.Statement[IR.AGENTIC_RECOMPILATION(components, prompt, initialEquations)]
  return IR.WHEN_EQUATION([IR.StatementBranch(condition, body)], IR.STRUCTURAL_WHEN, source, ATTRIBUTES)
end

"The prompt of a reconfigure equation: a string literal."
function promptString(e)::String
  e isa F.STRING_EXPRESSION || error("OMJLIR lowering: the prompt of a reconfigure equation is not a string literal")
  return e.value
end

"The names of the variables a reconfigure equation declares (Absyn element items)."
function reconfigureVariableNames(items)::Vector{String}
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
function serializeInitialEquations(items)::String
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

function lowerDirection(direction::Integer)::IR.Direction
  direction == F.Direction.INPUT && return IR.INPUT
  direction == F.Direction.OUTPUT && return IR.OUTPUT
  return IR.NO_DIRECTION
end

function lowerConnector(cty::Integer)::IR.ConnectorKind
  local C = F.ConnectorType
  (cty & C.POTENTIAL) > 0 && return IR.POTENTIAL
  (cty & C.FLOW) > 0 && return IR.FLOW
  (cty & C.STREAM) > 0 && return IR.STREAM
  return IR.NOT_CONNECTOR
end

"A binding's expression: of a typed or flattened binding; `nothing` for one evaluated or none."
function lowerBinding(b, scope::Scope)::Union{Nothing, IR.Expression}
  (isvariant(b, F.TYPED_BINDING) || isvariant(b, F.FLAT_BINDING)) || return nothing
  return lowerExp(b.bindingExp, scope)
end

const STATE_SELECTS = Dict("never" => IR.NEVER, "avoid" => IR.AVOID, "default" => IR.DEFAULT,
                           "prefer" => IR.PREFER, "always" => IR.ALWAYS)

"StateSelect: a literal of the enumeration; `nothing` for another expression (a parameter)."
function lowerStateSelect(b)::Union{Nothing, IR.StateSelect}
  local e = F.getBindingExp(F.getTypedExp(b))
  local member = e isa F.ENUM_LITERAL_EXPRESSION ? e.name :
                 e isa F.CREF_EXPRESSION && isvariant(e.cref, F.COMPONENT_REF_CREF) ? F.name(e.cref.node) : ""
  local s = get(STATE_SELECTS, member, nothing)
  s === nothing && @warn "OMJLIR lowering: StateSelect $(F.toString(e)) is not a literal of StateSelect, ignored"
  return s
end

"The built-in attributes of a variable (MLS 4.8), from the frontend's type attributes."
function lowerAttributes(typeAttributes, scope::Scope)::IR.VariableAttributes
  isempty(typeAttributes) && return IR.NO_ATTRIBUTES
  local values = Dict{String, Any}()
  for (name, b) in typeAttributes
    values[name] = name == "stateSelect" ? lowerStateSelect(b) :
                   name == "unbounded" ? nothing : lowerExp(F.getTypedExp(b), scope)
  end
  local get_(n) = get(values, n, nothing)
  return IR.VariableAttributes(get_("start"), get_("fixed"), get_("nominal"), get_("min"), get_("max"),
                               get_("unit"), get_("displayUnit"), get_("quantity"), get_("stateSelect"))
end

"The OMJLIR variable of the frontend variable `v`."
function lowerVariable(v, scope::Scope)::IR.Variable
  local name = crefName(crefParts(v.name); withSubscripts = true)
  name === nothing && error("OMJLIR lowering: the variable $(F.toString(v.name)) has a subscript that is not a literal")
  local ty = lowerType(v.ty)
  local a = v.attributes
  return IR.Variable(name, ty, variableKind(a.variability, ty), lowerBinding(v.binding, scope),
                     lowerAttributes(v.typeAttributes, scope), lowerDirection(a.direction),
                     v.visibility == F.Visibility.PROTECTED, lowerConnector(a.connectorType), lowerSource(v.info))
end

#= ==================================================================================== =#
#= The model                                                                            =#
#= ==================================================================================== =#

"The algorithm sections as ALGORITHM equations, their assigned variables their outputs."
function lowerAlgorithms(algorithms, scope::Scope)::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms
    local statements = lowerStatements(alg.statements, scope)
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
    lowerSystem(equations, initialEquations, algorithms, initialAlgorithms, scope)

A continuous-time partition: the equations, then the algorithm sections; the initial
equations, then the initial algorithm sections.
"""
function lowerSystem(eqs, initialEqs, algorithms, initialAlgorithms, scope::Scope)::IR.Partition
  local equations = lowerEquations(eqs, scope; reversed = true)
  append!(equations, lowerAlgorithms(algorithms, scope))
  local initialEquations = lowerEquations(initialEqs, scope; reversed = true)
  append!(initialEquations, lowerAlgorithms(initialAlgorithms, scope))
  local unknowns = Symbol[]
  return IR.Partition(1, IR.CONTINUOUS_PARTITION(), unknowns, equations, initialEquations)
end

"The variables of a flat model, and the scope of its expressions."
function lowerVariables(fm)::Tuple{IR.VariableSet, Scope}
  local names = Set{Symbol}()
  for v in fm.variables
    local n = crefName(crefParts(v.name); withSubscripts = true)
    n === nothing || push!(names, n)
  end
  push!(names, :time)
  local scope = Scope(names, Symbol[], false)
  local variables = IR.VariableSet()
  for v in fm.variables
    local lowered = lowerVariable(v, scope)
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
  return (variables, scope)
end

"""
    structuralCall(eq)

OM.jl's structural states and transitions, `initialStructuralState(s)` and
`structuralTransition(from, to, condition)` in the equation section: their arguments, or
`nothing` for another equation.
"""
function structuralCall(eq)::Union{Nothing, Tuple{Symbol, Vector{Any}}}
  isvariant(eq, F.EQUATION_NORETCALL) && eq.exp isa F.CALL_EXPRESSION && isvariant(eq.exp.call, F.TYPED_CALL) || return nothing
  local name = Symbol(F.AbsynUtil.pathString(F.nameConsiderBuiltin(eq.exp.call.fn)))
  name in (:initialStructuralState, :structuralTransition) || return nothing
  return (name, collect(eq.exp.call.arguments))
end

"""
    lowerFlatModel(fm)

The OMJLIR model of the frontend's flat model `fm`: its variables, its continuous-time
partition (equations, algorithms, initial equations and algorithms), its structural modes
(the structural submodels), states and transitions. The functions are added by
`lowerFunctions`.
"""
function lowerFlatModel(fm)::IR.Model
  local (variables, scope) = lowerVariables(fm)
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
      push!(transitions, IR.StructuralTransition(F.toString(from), F.toString(to), lowerExp(condition, scope)))
    end
  end
  local system = lowerSystem(equations, fm.initialEquations, fm.algorithms, fm.initialAlgorithms, scope)
  local modes = IR.StructuralMode[]
  for sub in fm.structuralSubmodels
    local (subVariables, subScope) = lowerVariables(sub)
    push!(modes, IR.StructuralMode(sub.name, subVariables,
                                   lowerSystem(sub.equations, sub.initialEquations, sub.algorithms,
                                               sub.initialAlgorithms, subScope)))
  end
  return IR.Model(fm.name, variables, IR.Partition[system], IR.Partition[], IR.ClockKind[],
                  Dict{Symbol, IR.ModelicaFunction}(), initialMode, modes, transitions, fm.scodeProgram)
end

end # module LowerFlatModel
