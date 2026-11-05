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
    OMBackendOMCompilerExt

The flat model of omc's frontend (an `NFFlatModel` read through OMCompiler.jl) lowered into
OMJLIR, as `LowerFlatModel` lowers OMFrontend.jl's. Its order and forms are that module's.
"""
module OMBackendOMCompilerExt

import OMBackend
import OMBackend: OMJLIR as IR, LowerFlatModel as L, unsupported
import OMCompiler
import OMCompiler: Node, EnumLiteral, ctor, apply

#= ==================================================================================== =#
#= Names and sources                                                                    =#
#= ==================================================================================== =#

pathName(path::Node)::Symbol = Symbol(apply("Path.toString", path))

lowerSource(info::Node)::IR.Source = IR.Source(info.fileName, info.lineNumberStart, info.columnNumberStart)

lowerElementSource(source::Node)::IR.Source = lowerSource(source.info)

expString(e::Node)::String = apply("Expression.toString", e)

"The name of an instance node."
function nodeName(node::Node)::String
  local c = ctor(node)
  c === :VALUE && return nodeName(node.node)
  c === :INNER_OUTER_NODE && return nodeName(node.innerNode)
  c in (:CELL, :CLASS_NODE, :COMPONENT_NODE, :NAME_NODE, :VAR_NODE) && return node.name
  error("OMJLIR lowering: the name of a $c")
end

"The component of a component node."
function componentOf(node::Node)::Node
  local c = ctor(node)
  c === :VALUE && return componentOf(node.node)
  c === :INNER_OUTER_NODE && return componentOf(node.innerNode)
  c === :CELL && return componentOf(node.cell)
  return node.component
end

"The class of a class node."
function classOf(node::Node)::Node
  local c = ctor(node)
  c === :VALUE && return classOf(node.node)
  c === :CELL && return classOf(node.cell)
  return node.cls
end

"The components of an instantiated class."
classComponents(cls::Node)::Vector = ctor(cls) === :INSTANCED_CLASS ? cls.elements.components : Node[]

"The constructor of a uniontype value or the literal of an enumeration value."
variant(x::Node)::Symbol = ctor(x)
variant(x::EnumLiteral)::Symbol = x.literal

#= ==================================================================================== =#
#= Types, dimensions and subscripts                                                     =#
#= ==================================================================================== =#

function lowerType(ty::Node)::IR.ModelicaType
  local c = ctor(ty)
  c === :REAL && return IR.T_REAL()
  c === :INTEGER && return IR.T_INTEGER()
  c === :BOOLEAN && return IR.T_BOOLEAN()
  c === :STRING && return IR.T_STRING()
  c === :CLOCK && return IR.T_CLOCK()
  c === :ENUMERATION && return enumerationType(ty)
  c === :ARRAY && return IR.T_ARRAY(lowerType(ty.elementType), IR.Dimension[lowerDimension(d) for d in ty.dimensions])
  c === :TUPLE && return IR.T_TUPLE(IR.ModelicaType[lowerType(t) for t in ty.types])
  c === :COMPLEX && return complexType(ty)
  c === :FUNCTION && return functionType(ty.fn)
  c === :NORETCALL && return IR.T_NORETCALL()
  c === :METABOXED && return lowerType(ty.ty)
  return IR.T_UNKNOWN()
end

enumerationType(ty::Node)::IR.T_ENUMERATION = IR.T_ENUMERATION(pathName(ty.typePath), Symbol.(ty.literals))

"The complex types lowered so far, by their class node (`nodeid`), while a model is lowered."
const COMPLEX_TYPES = Dict{UInt, IR.ModelicaType}()

"""
A record type with its fields, an external object class or another class, named as the
frontend prints the type. A record a field's type reaches again is there without its fields
(see LowerFlatModel.complexType).
"""
function complexType(ty::Node)::IR.ModelicaType
  local key = OMCompiler.nodeid(ty.cls)
  local cached = get(COMPLEX_TYPES, key, nothing)
  cached === nothing || return cached
  local lowered = lowerComplexType(ty)
  (lowered isa IR.T_RECORD && isempty(lowered.fields)) || (COMPLEX_TYPES[key] = lowered)
  return lowered
end

function lowerComplexType(ty::Node)::IR.ModelicaType
  local name = Symbol(apply("Type.toString", ty))
  local c = ctor(ty.complexTy)
  c === :EXTERNAL_OBJECT && return IR.T_EXTERNAL_OBJECT(name)
  c === :RECORD || return IR.T_CLASS(name)
  local inProgress = get!(() -> Set{Symbol}(), task_local_storage(), :omc_records_lowering)::Set{Symbol}
  name in inProgress && return IR.T_RECORD(name, IR.Field[])
  push!(inProgress, name)
  try
    return IR.T_RECORD(name, IR.Field[IR.Field(Symbol(nodeName(f)), lowerType(componentOf(f).ty))
                                     for f in classComponents(classOf(ty.cls))])
  finally
    delete!(inProgress, name)
  end
end

function functionType(fn::Node)::IR.T_FUNCTION
  local types(nodes) = IR.ModelicaType[lowerType(componentOf(n).ty) for n in nodes]
  return IR.T_FUNCTION(Symbol(apply("Function.name", fn)), types(fn.inputs), types(fn.outputs))
end

function lowerDimension(d::Node)::IR.Dimension
  local c = ctor(d)
  c === :INTEGER && return IR.DIM_SIZE(d.size)
  c === :BOOLEAN && return IR.DIM_BOOLEAN()
  c === :ENUM && return IR.DIM_ENUM(enumerationType(d.enumType))
  c === :EXP && return IR.DIM_EXP(lowerExp(d.exp))
  return IR.DIM_UNKNOWN()
end

function lowerSubscript(s::Node)::IR.Subscript
  local c = ctor(s)
  c === :INDEX && return IR.INDEX(lowerExp(s.index))
  c === :SLICE && return IR.SLICE(lowerExp(s.slice))
  c === :WHOLE && return IR.WHOLE()
  error("OMJLIR lowering: a subscript of the form $c")
end

#= ==================================================================================== =#
#= Component references                                                                 =#
#= ==================================================================================== =#

function crefParts(cref::Node)::Vector{IR.CrefPart}
  local parts = IR.CrefPart[]
  while ctor(cref) === :CREF
    push!(parts, IR.CrefPart(Symbol(nodeName(cref.node)), IR.Subscript[lowerSubscript(s) for s in cref.subscripts],
                             lowerType(cref.ty)))
    cref = cref.restCref
  end
  return reverse!(parts)
end

lowerCref(cref::Node)::IR.ComponentRef = IR.ComponentRef(crefParts(cref))

"The names of the model's variables while the model is lowered."
const VARIABLE_NAMES = Ref{Union{Nothing, Set{Symbol}}}(nothing)

"""
A reference to a whole array whose elements are the model's variables, as the array of its
elements: omc's frontend scalarizes the variables but keeps such a reference in a call's
arguments (MultiBody's `Frames.resolve1(R, body.r_CM)`). `nothing` for another reference.
"""
function expandedArrayCref(cref::IR.ComponentRef, ty::IR.T_ARRAY)::Union{Nothing, IR.Expression}
  local names = VARIABLE_NAMES[]
  (names === nothing || cref.name in names) && return nothing
  local last = cref.parts[end]
  isempty(last.subscripts) || return nothing
  any(d -> IR.dimensionSize(d) === nothing, ty.dimensions) && return nothing
  local elty = ty.elementType
  local element(subs) = IR.CREF(IR.ComponentRef([cref.parts[1:end-1]..., IR.CrefPart(last.name, subs, last.ty)]), elty)
  local build(level, subs) = level > length(ty.dimensions) ? element(subs) :
    IR.ARRAY(IR.Expression[build(level + 1, IR.Subscript[subs..., IR.INDEX(IR.dimensionIndex(ty.dimensions[level], k))])
                           for k in 1:IR.dimensionSize(ty.dimensions[level])],
             IR.T_ARRAY(elty, ty.dimensions[level:end]))
  local first = element(IR.Subscript[IR.INDEX(IR.dimensionIndex(d, 1)) for d in ty.dimensions])
  first.cref.name in names || return nothing
  return build(1, IR.Subscript[])
end

lowerCrefExp(cref::Node, ty::IR.ModelicaType)::IR.Expression =
  ctor(cref) === :WILD ? IR.WILD() : IR.CREF(lowerCref(cref), ty)

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

function lowerOp(operator::Node)::IR.Op
  local op = operator.op::EnumLiteral
  op.literal === :USERDEFINED && error("OMJLIR lowering: a user-defined operator the frontend did not resolve")
  return getfield(IR, op.literal)
end

lowerExps(es)::Vector{IR.Expression} = IR.Expression[lowerExp(e) for e in es]

lowerOptExp(o)::Union{Nothing, IR.Expression} = o === nothing ? nothing : lowerExp(o)

isTrue(e::Node)::Bool = ctor(e) === :BOOLEAN && e.value

const INLINE_TYPES = (:NORM_INLINE, :BUILTIN_EARLY_INLINE, :EARLY_INLINE)

function lowerExp(e::Node)::IR.Expression
  local c = ctor(e)
  if c === :INTEGER
    return IR.INTEGER(e.value)
  elseif c === :REAL
    return IR.REAL(e.value)
  elseif c === :STRING
    return IR.STRING(e.value)
  elseif c === :BOOLEAN
    return IR.BOOLEAN(e.value)
  elseif c === :ENUM_LITERAL
    return IR.ENUM_LITERAL(enumerationType(e.ty), Symbol(e.name), e.index)
  elseif c === :CLKCONST
    return IR.CLOCK(lowerClock(e.clk))
  elseif c === :CREF
    #= A scalarized element keeps the array's type: see LowerFlatModel.lowerExp =#
    local ty = lowerType(e.ty)
    if ty isa IR.T_ARRAY && ctor(e.cref) === :CREF
      local element = lowerType(apply("ComponentRef.getSubscriptedType", e.cref))
      L.isScalarType(element) && (ty = element)
    end
    if ty isa IR.T_ARRAY && ctor(e.cref) === :CREF
      local expanded = expandedArrayCref(lowerCref(e.cref), ty)
      expanded === nothing || return expanded
    end
    return lowerCrefExp(e.cref, ty)
  elseif c === :ARRAY
    return IR.ARRAY(lowerExps(e.elements), lowerType(e.ty))
  elseif c === :MATRIX
    local ty = lowerType(apply("Expression.typeOf", e))::IR.T_ARRAY
    local rowTy = IR.T_ARRAY(ty.elementType, ty.dimensions[2:end])
    return IR.ARRAY(IR.Expression[IR.ARRAY(lowerExps(row), rowTy) for row in e.elements], ty)
  elseif c === :RECORD
    return IR.RECORD(lowerExps(e.elements), lowerType(e.ty))
  elseif c === :RANGE
    return IR.RANGE(lowerExp(e.start), lowerOptExp(e.step), lowerExp(e.stop), lowerType(e.ty))
  elseif c === :TUPLE
    return IR.TUPLE(lowerExps(e.elements), lowerType(e.ty))
  elseif c === :CALL
    return lowerCall(e.call)
  elseif c === :SIZE
    return IR.SIZE(lowerExp(e.exp), lowerOptExp(e.dimIndex))
  elseif c === :BINARY
    local (a, b) = (lowerExp(e.exp1), lowerExp(e.exp2))
    local (op, ty) = (lowerOp(e.operator), lowerType(e.operator.ty))
    if L.isScalarType(IR.typeOf(a)) && L.isScalarType(IR.typeOf(b))
      (op, ty) = (get(L.SCALAR_OPS, op, op), IR.elementType(ty))
    end
    return IR.BINARY(a, op, b, ty)
  elseif c === :MULTARY
    return lowerMultary(e)
  elseif c === :UNARY
    local a = lowerExp(e.exp)
    local ty = lowerType(e.operator.ty)
    return IR.UNARY(lowerOp(e.operator), a, L.isScalarType(IR.typeOf(a)) ? IR.elementType(ty) : ty)
  elseif c === :LBINARY
    return IR.LBINARY(lowerExp(e.exp1), lowerOp(e.operator), lowerExp(e.exp2))
  elseif c === :LUNARY
    return IR.LUNARY(lowerOp(e.operator), lowerExp(e.exp))
  elseif c === :RELATION
    return IR.RELATION(lowerExp(e.exp1), lowerOp(e.operator), lowerExp(e.exp2), 0)
  elseif c === :IF
    return IR.IF_EXPRESSION(lowerExp(e.condition), lowerExp(e.trueBranch), lowerExp(e.falseBranch), lowerType(e.ty))
  elseif c === :CAST
    return IR.CAST(lowerExp(e.exp), lowerType(e.ty))
  elseif c === :SUBSCRIPTED_EXP
    return IR.SUBSCRIPTED(lowerExp(e.exp), IR.Subscript[lowerSubscript(s) for s in e.subscripts], lowerType(e.ty))
  elseif c === :TUPLE_ELEMENT
    return IR.TUPLE_ELEMENT(lowerExp(e.tupleExp), e.index, lowerType(e.ty))
  elseif c === :RECORD_ELEMENT
    return IR.RECORD_ELEMENT(lowerExp(e.recordExp), e.index, Symbol(e.fieldName), lowerType(e.ty))
  elseif c in (:BOX, :UNBOX, :MUTABLE)
    return lowerExp(e.exp)
  end
  unsupported("the expression form $c", expString(e))
end

"""
NF's n-ary sum or product, `a1 + ... + an - (b1 + ... + bm)` or `a1 * ... * an / (b1 * ... * bm)`
(`arguments`, `inv_arguments`): the binary operations from left to right.
"""
function lowerMultary(e::Node)::IR.Expression
  local args = lowerExps(e.arguments)
  local invs = lowerExps(e.inv_arguments)
  local (op, ty) = (lowerOp(e.operator), lowerType(e.operator.ty))
  if all(x -> L.isScalarType(IR.typeOf(x)), args) && all(x -> L.isScalarType(IR.typeOf(x)), invs)
    (op, ty) = (get(L.SCALAR_OPS, op, op), IR.elementType(ty))
  end
  local inv = op === IR.ADD ? IR.SUB : op === IR.MUL ? IR.DIV : unsupported("an n-ary $op", expString(e))
  local unit = ty isa IR.T_INTEGER ? IR.INTEGER(op === IR.ADD ? 0 : 1) : IR.REAL(op === IR.ADD ? 0.0 : 1.0)
  local acc = isempty(args) ? unit : foldl((a, b) -> IR.BINARY(a, op, b, ty), args)
  for b in invs
    acc = IR.BINARY(acc, inv, b, ty)
  end
  return acc
end

const ELEMENTWISE_BUILTINS = (:pre, :der, :edge, :change, :previous)

"""
`pre(x)` of an array `x` expanded into its elements (`expandedArrayCref`), as the array of
`pre` of each element, as OMFrontend.jl's scalarization writes it.
"""
function elementwiseCall(name::Symbol, arg::IR.ARRAY, ty::IR.ModelicaType, attributes::IR.CallAttributes)::IR.Expression
  local elementTy = ty isa IR.T_ARRAY && length(ty.dimensions) > 1 ? IR.T_ARRAY(ty.elementType, ty.dimensions[2:end]) :
                    IR.elementType(ty)
  return IR.ARRAY(IR.Expression[e isa IR.ARRAY ? elementwiseCall(name, e, elementTy, attributes) :
                                IR.CALL(name, IR.Expression[e], elementTy, attributes) for e in arg.elements], ty)
end

function lowerCall(call::Node)::IR.Expression
  local c = ctor(call)
  if c === :TYPED_CALL
    local a = call.attributes
    local attributes = IR.CallAttributes(a.builtin, a.isImpure, variant(a.inlineType) in INLINE_TYPES)
    local name = Symbol(apply("Function.name", call.fn))
    local args = lowerExps(call.arguments)
    local ty = lowerType(call.ty)
    if a.builtin && name in ELEMENTWISE_BUILTINS && length(args) == 1 && args[1] isa IR.ARRAY
      return elementwiseCall(name, args[1], ty, attributes)
    end
    return IR.CALL(name, args, ty, attributes)
  elseif c === :TYPED_ARRAY_CONSTRUCTOR
    return IR.ARRAY_CONSTRUCTOR(lowerExp(call.exp), lowerIterators(call.iters), lowerType(call.ty))
  elseif c === :TYPED_REDUCTION
    return IR.REDUCTION(Symbol(apply("Function.name", call.fn)), lowerExp(call.exp), lowerIterators(call.iters),
                        lowerType(call.ty))
  end
  error("OMJLIR lowering: an untyped call")
end

lowerIterators(iters)::Vector{IR.Iterator} =
  IR.Iterator[IR.Iterator(Symbol(nodeName(node)), lowerExp(range)) for (node, range) in iters]

function lowerClock(k::Node)::IR.ClockKind
  local c = ctor(k)
  c === :INFERRED_CLOCK && return IR.INFERRED_CLOCK()
  c === :RATIONAL_CLOCK && return IR.INTEGER_CLOCK(lowerExp(k.intervalCounter), lowerExp(k.resolution))
  c === :REAL_CLOCK && return IR.REAL_CLOCK(lowerExp(k.interval))
  c === :EVENT_CLOCK && return IR.BOOLEAN_CLOCK(lowerExp(k.condition), lowerExp(k.startInterval))
  c === :SOLVER_CLOCK && return IR.SOLVER_CLOCK(lowerExp(k.c), lowerExp(k.solverMethod))
  error("OMJLIR lowering: a clock of the form $c")
end

#= ==================================================================================== =#
#= Statements                                                                           =#
#= ==================================================================================== =#

lowerStatements(stmts)::Vector{IR.Statement} = IR.Statement[lowerStatement(s) for s in stmts]

function lowerStatement(s::Node)::IR.Statement
  local c = ctor(s)
  local source = lowerElementSource(s.source)
  if c === :ASSIGNMENT
    return L.lowerAssignment(lowerExp(s.lhs), lowerExp(s.rhs), source)
  elseif c === :IF
    local branches = IR.StatementBranch[]
    local elseBody = IR.Statement[]
    local all = s.branches
    for (k, (cond, body)) in enumerate(all)
      if k == length(all) && isTrue(cond)
        elseBody = lowerStatements(body)
      else
        push!(branches, IR.StatementBranch(lowerExp(cond), lowerStatements(body)))
      end
    end
    return IR.IF_STATEMENT(branches, elseBody, source)
  elseif c === :FOR
    local range = s.range
    range === nothing && error("OMJLIR lowering: a for-statement without a range")
    return IR.FOR_STATEMENT(IR.Iterator(Symbol(nodeName(s.iterator)), lowerExp(range)), lowerStatements(s.body), source)
  elseif c === :WHILE
    return IR.WHILE_STATEMENT(lowerExp(s.condition), lowerStatements(s.body), source)
  elseif c === :WHEN
    return IR.WHEN_STATEMENT(IR.WhenBranch[L.whenBranch(lowerExp(cond), lowerStatements(body); dropInitialDisjuncts = false)
                                           for (cond, body) in s.branches], source)
  elseif c === :ASSERT
    return IR.ASSERT(lowerExp(s.condition), lowerExp(s.message), lowerExp(s.level), source)
  elseif c === :TERMINATE
    return IR.TERMINATE(lowerExp(s.message), source)
  elseif c === :REINIT
    return IR.REINIT(lowerCref(s.cref.cref), lowerExp(s.reinitExp), source)
  elseif c === :NORETCALL
    return IR.NORETCALL(lowerExp(s.exp), source)
  elseif c === :RETURN
    return IR.RETURN(source)
  elseif c === :BREAK
    return IR.BREAK(source)
  end
  error("OMJLIR lowering: the statement form $c has no OMJLIR form")
end

#= ==================================================================================== =#
#= Equations                                                                            =#
#= ==================================================================================== =#

function lowerEquations(eqs::Vector; reversed::Bool)::Vector{IR.Equation}
  local out = IR.Equation[]
  for eq in (reversed ? Iterators.reverse(eqs) : eqs)
    local lowered = lowerEquation(eq)
    lowered isa Vector ? append!(out, lowered) : lowered === nothing || push!(out, lowered)
  end
  return out
end

function lowerEquation(eq::Node)::Union{Nothing, IR.Equation, Vector{IR.Equation}}
  local c = ctor(eq)
  c === :CONNECT && return nothing
  local source = lowerElementSource(eq.source)
  if c === :EQUALITY
    local ty = lowerType(eq.ty)
    local lhs = lowerExp(eq.lhs)
    local rhs = lowerExp(eq.rhs)
    ctor(eq.ty) === :COMPLEX && return IR.RECORD_EQUATION(lhs, rhs, ty, L.recordSize(ty), source, L.ATTRIBUTES)
    if ty isa IR.T_ARRAY
      if lhs isa IR.ARRAY || rhs isa IR.ARRAY
        local elements = elementEquations(lhs, rhs, ty, source)
        elements === nothing || return elements
      end
      return IR.ARRAY_EQUATION(rhs, lhs, ty, source, L.ATTRIBUTES)
    end
    return IR.SCALAR_EQUATION(lhs, rhs, ty, source, L.ATTRIBUTES)
  elseif c === :FOR
    local range = eq.range
    range === nothing && error("OMJLIR lowering: a for-equation without a range")
    local iterator = IR.Iterator(Symbol(nodeName(eq.iterator)), lowerExp(range))
    return IR.FOR_EQUATION(iterator, lowerEquations(eq.body; reversed = true), source, L.ATTRIBUTES)
  elseif c === :IF
    local branches = IR.EquationBranch[]
    local elseEquations = IR.Equation[]
    local all = eq.branches
    for (k, b) in enumerate(all)
      ctor(b) === :BRANCH || error("OMJLIR lowering: an if-equation's branch the frontend did not type")
      if k == length(all) && isTrue(b.condition)
        elseEquations = lowerEquations(b.body; reversed = true)
      else
        push!(branches, IR.EquationBranch(lowerExp(b.condition), lowerEquations(b.body; reversed = false)))
      end
    end
    return IR.IF_EQUATION(branches, elseEquations, source, L.ATTRIBUTES)
  elseif c === :WHEN
    local branches = IR.WhenBranch[L.whenBranch(lowerExp(b.condition), whenStatements(b.body)) for b in eq.branches]
    return IR.WHEN_EQUATION(branches, IR.REGULAR_WHEN, source, L.ATTRIBUTES)
  elseif c === :ASSERT
    return IR.ASSERT_EQUATION(lowerExp(eq.condition), lowerExp(eq.message), lowerExp(eq.level), source)
  elseif c === :TERMINATE
    unsupported("terminate() in an equation section outside a when-equation", apply("Equation.toString", eq))
  elseif c === :REINIT
    unsupported("reinit() outside a when-equation (MLS 8.3.6)", apply("Equation.toString", eq))
  elseif c === :NORETCALL
    return L.callEquation(lowerExp(eq.exp), source)
  end
  error("OMJLIR lowering: the equation form $c")
end

"""
An array equation of the scalarized variables as one equation per element, as OMFrontend.jl's
scalarization makes it: `cr = f(n)` is `cr[1] = f(n)[1]`, ... An element of a side that is not an
array literal is that side subscripted. `nothing` for an array of an unknown size.
"""
function elementEquations(lhs::IR.Expression, rhs::IR.Expression, ty::IR.T_ARRAY,
                          source::IR.Source)::Union{Nothing, Vector{IR.Equation}}
  any(d -> IR.dimensionSize(d) === nothing, ty.dimensions) && return nothing
  local out = IR.Equation[]
  #= A side is an expression and the subscripts still to apply to it: an array literal is
     stepped into, another expression subscripted once with all its indices. =#
  local step((e, subs), d, k) = e isa IR.ARRAY && isempty(subs) ? (e.elements[k], IR.Subscript[]) :
                                                                  (e, IR.Subscript[subs..., IR.INDEX(IR.dimensionIndex(d, k))])
  local finish((e, subs)) = isempty(subs) ? e : IR.SUBSCRIPTED(e, subs, ty.elementType)
  local function expand(l, r, level)
    if level > length(ty.dimensions)
      push!(out, IR.SCALAR_EQUATION(finish(l), finish(r), ty.elementType, source, L.ATTRIBUTES))
      return
    end
    local d = ty.dimensions[level]
    for k in 1:IR.dimensionSize(d)
      expand(step(l, d, k), step(r, d, k), level + 1)
    end
  end
  expand((lhs, IR.Subscript[]), (rhs, IR.Subscript[]), 1)
  return out
end

function whenStatements(eqs::Vector)::Vector{IR.Statement}
  local out = IR.Statement[]
  for eq in eqs
    local c = ctor(eq)
    local source = lowerElementSource(eq.source)
    if c === :EQUALITY
      push!(out, IR.ASSIGNMENT(lowerExp(eq.lhs), lowerExp(eq.rhs), source))
    elseif c === :REINIT
      push!(out, IR.REINIT(lowerCref(eq.cref.cref), lowerExp(eq.reinitExp), source))
    elseif c === :ASSERT
      push!(out, IR.ASSERT(lowerExp(eq.condition), lowerExp(eq.message), lowerExp(eq.level), source))
    elseif c === :TERMINATE
      push!(out, IR.TERMINATE(lowerExp(eq.message), source))
    elseif c === :NORETCALL
      push!(out, IR.NORETCALL(lowerExp(eq.exp), source))
    elseif c === :IF
      local branches = IR.StatementBranch[]
      local elseBody = IR.Statement[]
      local all = eq.branches
      for (k, b) in enumerate(all)
        if k == length(all) && isTrue(b.condition)
          elseBody = whenStatements(b.body)
        else
          push!(branches, IR.StatementBranch(lowerExp(b.condition), whenStatements(b.body)))
        end
      end
      push!(out, IR.IF_STATEMENT(branches, elseBody, source))
    else
      error("OMJLIR lowering: $(apply("Equation.toString", eq)) in a when-equation is not supported")
    end
  end
  return out
end

#= ==================================================================================== =#
#= Variables                                                                            =#
#= ==================================================================================== =#

function variableKind(variability::EnumLiteral, ty::IR.ModelicaType)::IR.VariableKind
  local v = variability.literal
  v === :CONSTANT && return IR.CONSTANT()
  v in (:STRUCTURAL_PARAMETER, :PARAMETER, :NON_STRUCTURAL_PARAMETER) && return IR.PARAMETER()
  IR.elementType(ty) isa IR.T_CLOCK && return IR.CLOCK_VARIABLE()
  v === :DISCRETE && return IR.DISCRETE()
  return IR.ALGEBRAIC()
end

function lowerDirection(direction::EnumLiteral)::IR.Direction
  direction.literal === :INPUT && return IR.INPUT
  direction.literal === :OUTPUT && return IR.OUTPUT
  return IR.NO_DIRECTION
end

"The connector prefix, from NFPrefixes.ConnectorType's bits."
function lowerConnector(cty::Integer)::IR.ConnectorKind
  (cty & 1) > 0 && return IR.POTENTIAL
  (cty & 2) > 0 && return IR.FLOW
  (cty & 4) > 0 && return IR.STREAM
  return IR.NOT_CONNECTOR
end

bindingExp(b::Node) = ctor(b) in (:TYPED_BINDING, :FLAT_BINDING, :CEVAL_BINDING) ? b.bindingExp : nothing

function lowerBinding(b::Node)::Union{Nothing, IR.Expression}
  ctor(b) in (:TYPED_BINDING, :FLAT_BINDING) || return nothing
  return lowerExp(b.bindingExp)
end

function lowerStateSelect(b::Node)::Union{Nothing, IR.StateSelect}
  local e = bindingExp(b)
  local member = e === nothing ? "" :
                 ctor(e) === :ENUM_LITERAL ? e.name :
                 ctor(e) === :CREF && ctor(e.cref) === :CREF ? nodeName(e.cref.node) : ""
  local s = get(L.STATE_SELECTS, member, nothing)
  s === nothing && @warn "OMJLIR lowering: StateSelect $(e === nothing ? "" : expString(e)) is not a literal of StateSelect, ignored"
  return s
end

function lowerAttributes(typeAttributes::Vector)::IR.VariableAttributes
  isempty(typeAttributes) && return IR.NO_ATTRIBUTES
  local values = Dict{String, Any}()
  for (name, b) in typeAttributes
    local e = bindingExp(b)
    values[name] = name == "stateSelect" ? lowerStateSelect(b) :
                   name == "unbounded" || e === nothing ? nothing : lowerExp(e)
  end
  local get_(n) = get(values, n, nothing)
  return IR.VariableAttributes(get_("start"), get_("fixed"), get_("nominal"), get_("min"), get_("max"),
                               get_("unit"), get_("displayUnit"), get_("quantity"), get_("stateSelect"))
end

function lowerVariable(v::Node)::IR.Variable
  local ty = lowerType(v.ty)
  local a = v.attributes
  return IR.Variable(lowerCref(v.name), ty, variableKind(a.variability, ty), lowerBinding(v.binding),
                     lowerAttributes(v.typeAttributes), lowerDirection(a.direction),
                     v.visibility.literal === :PROTECTED, lowerConnector(a.connectorType), lowerSource(v.info), :model)
end

function lowerVariables(fm::Node)::IR.VariableSet
  local variables = IR.VariableSet()
  for v in fm.variables
    local lowered = lowerVariable(v)
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

#= ==================================================================================== =#
#= The model                                                                            =#
#= ==================================================================================== =#

function lowerAlgorithms(algorithms::Vector)::Vector{IR.Equation}
  local out = IR.Equation[]
  for alg in algorithms
    local statements = lowerStatements(alg.statements)
    push!(out, IR.ALGORITHM(statements, L.assignedCrefs(statements), lowerElementSource(alg.source), L.ATTRIBUTES))
  end
  return out
end

function lowerSystem(fm::Node)::IR.Partition
  local equations = lowerEquations(fm.equations; reversed = true)
  append!(equations, lowerAlgorithms(fm.algorithms))
  local initialEquations = lowerEquations(fm.initialEquations; reversed = true)
  append!(initialEquations, lowerAlgorithms(fm.initialAlgorithms))
  return IR.Partition(1, IR.CONTINUOUS_PARTITION(), Symbol[], equations, initialEquations)
end

#= ==================================================================================== =#
#= Functions                                                                            =#
#= ==================================================================================== =#

"A function variable's name; a record constructor's output `\$outR` as OMFrontend.jl names it, `outR`."
functionVariableName(node::Node)::Symbol = Symbol(lstrip(nodeName(node), '$'))

function functionVariable(node::Node)::IR.FunctionVariable
  local comp = componentOf(node)
  return IR.FunctionVariable(functionVariableName(node), lowerType(comp.ty), lowerBinding(comp.binding),
                             Pair{Symbol, IR.Expression}[])
end

"The node a scope reference or a node handle refers to."
function resolveNode(n::Node)::Node
  local c = ctor(n)
  c === :VALUE && return resolveNode(n.node)
  c === :CELL && return resolveNode(n.cell)
  return n
end

"The name of the function a function class node holds in its cache; `nothing` if it holds none."
function cachedFunctionName(node::Node)::Union{Nothing, Symbol}
  local n = resolveNode(node)
  ctor(n) === :CLASS_NODE || return nothing
  local cache = first(n.caches)
  (cache isa Node && ctor(cache) === :FUNCTION && !isempty(cache.funcs)) || return nothing
  return Symbol(apply("Function.name", first(cache.funcs)))
end

"A function's derivative annotations of order 1, as LowerFlatModel.lowerFunction takes them."
function lowerDerivatives(fn::Node)::Vector{IR.FunctionDerivative}
  local out = IR.FunctionDerivative[]
  for d in fn.derivatives
    (ctor(d.order) === :INTEGER && d.order.value == 1) || continue
    local name = cachedFunctionName(d.derivativeFn)
    name === nothing && continue
    push!(out, IR.FunctionDerivative(name, 1, Int[c[1] for c in d.conditions]))
  end
  return out
end

"An Absyn string or array of strings as Modelica text; `nothing` for another expression."
function absynStringText(e::Node)::Union{Nothing, String}
  local c = ctor(e)
  c === :STRING && return string('"', e.value, '"')
  if c === :ARRAY
    local parts = [absynStringText(x) for x in e.arrayExp]
    any(isnothing, parts) && return nothing
    return string("{", join(parts, ", "), "}")
  end
  return nothing
end

"""
An external function's annotation as the flat model writes it, as far as the code generation
reads it: the default IncludeDirectory of the function's library, then the string-valued
entries (Library, Include, LibraryDirectory, IncludeDirectory).
"""
function externalAnnotation(name::Symbol, ann)::String
  local entries = [string("IncludeDirectory = \"modelica://", first(split(String(name), ".")), "/Resources/Include\"")]
  if ann isa Node && ctor(ann.modification) === :MOD
    for sub in ann.modification.subModLst
      local binding = ctor(sub.mod) === :MOD ? sub.mod.binding : nothing
      local text = binding isa Node ? absynStringText(binding) : nothing
      text === nothing || push!(entries, string(sub.ident, " = ", text))
    end
  end
  return string("annotation(", join(entries, ", "), ")")
end

function lowerExternal(name::Symbol, sections::Node)::IR.ExternalDeclaration
  local output = ctor(sections.outputRef) === :CREF ? lowerCref(sections.outputRef) : nothing
  return IR.ExternalDeclaration(sections.name, sections.language, lowerExps(sections.args), output,
                                externalAnnotation(name, sections.ann))
end

function lowerFunction(fn::Node)::IR.ModelicaFunction
  local name = Symbol(apply("Function.name", fn))
  local sections = classOf(fn.node).sections
  local external = ctor(sections) === :EXTERNAL ? lowerExternal(name, sections) : nothing
  local body = ctor(sections) === :SECTIONS ? reduce(vcat, (lowerStatements(a.statements) for a in sections.algorithms);
                                                     init = IR.Statement[]) : IR.Statement[]
  local attributes = fn.attributes
  local impure = variant(attributes.purity) === :IMPURE
  local inline = variant(attributes.inline) in (:EARLY_INLINE, :NORM_INLINE)
  local variables(nodes) = IR.FunctionVariable[functionVariable(n) for n in nodes]
  return IR.ModelicaFunction(name, variables(fn.inputs), variables(fn.outputs), variables(fn.locals), body, external,
                             lowerDerivatives(fn), impure, inline, IR.NO_SOURCE)
end

function lowerFlatModel(fm::Node, functions::Vector)::IR.Model
  empty!(COMPLEX_TYPES)
  try
    VARIABLE_NAMES[] = Set{Symbol}(lowerCref(v.name).name for v in fm.variables)
    local (variables, system) = try
      (lowerVariables(fm), lowerSystem(fm))
    finally
      VARIABLE_NAMES[] = nothing
    end
    return IR.Model(apply("Path.toString", fm.name), variables, IR.Partition[system], IR.Partition[], IR.ClockKind[],
                    L.withInlineDerivatives(IR.ModelicaFunction[lowerFunction(f) for f in functions]),
                    nothing, IR.StructuralMode[], IR.StructuralTransition[], nothing)
  finally
    empty!(COMPLEX_TYPES)
  end
end

OMBackend.lowerOMCFlatModel(fm::Node, functions::Vector) = OMBackend.Passes.runPasses!(lowerFlatModel(fm, functions))

end # module OMBackendOMCompilerExt
