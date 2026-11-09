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
    LowerUtil

What the lowerings of a frontend's flat model into OMJLIR share, on OMJLIR alone: OMFrontend.jl's
(`OMBackendOMFrontendExt`, its `LowerFlatModel`) and omc's (`OMBackendOMCompilerExt`). The scalar operators of
a scalarized element, assignments of tuples, calls in equation sections, when-branches and the order of a
when-equation's statements, the variables an algorithm assigns, structural parameters, the derivatives of inlined
functions. The order and forms of the lowered model are those `LowerFlatModel` documents.
"""
module LowerUtil

import ..OMJLIR
import ..unsupported

const IR = OMJLIR

"The scalar operators of the element-wise and scalar-array ones."
const SCALAR_OPS = Dict{IR.Op, IR.Op}(
  IR.ADD_EW => IR.ADD, IR.ADD_SCALAR_ARRAY => IR.ADD, IR.ADD_ARRAY_SCALAR => IR.ADD,
  IR.SUB_EW => IR.SUB, IR.SUB_SCALAR_ARRAY => IR.SUB, IR.SUB_ARRAY_SCALAR => IR.SUB,
  IR.MUL_EW => IR.MUL, IR.MUL_SCALAR_ARRAY => IR.MUL, IR.MUL_ARRAY_SCALAR => IR.MUL,
  IR.DIV_EW => IR.DIV, IR.DIV_SCALAR_ARRAY => IR.DIV, IR.DIV_ARRAY_SCALAR => IR.DIV,
  IR.POW_EW => IR.POW, IR.POW_SCALAR_ARRAY => IR.POW, IR.POW_ARRAY_SCALAR => IR.POW)

"A type of a scalar value (not an array, a record or an unknown type)."
isScalarType(ty::IR.ModelicaType)::Bool = ty isa Union{IR.T_REAL, IR.T_INTEGER, IR.T_BOOLEAN, IR.T_ENUMERATION}

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

const ATTRIBUTES = IR.DEFAULT_ATTRIBUTES

"""
    terminateEquation(terminate, source)

`terminate()` in an equation section outside a when-equation (MLS 8.3.8): the analysis ends where it is reached; one
of no condition at the initialization, `when initial() then terminate(...)`. In an if-equation's branch
`hoistIfEquationAsserts!` makes it when-equations of the branch's condition.
"""
terminateEquation(t::IR.TERMINATE, source::IR.Source)::IR.WHEN_EQUATION =
  IR.WHEN_EQUATION(IR.WhenBranch[IR.WhenBranch(IR.Expression[], true, false, IR.Statement[t])], IR.REGULAR_WHEN, source, ATTRIBUTES)

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
What a reference reads or writes (`whenEquationOrder`): its variable (`crefBase`) and, where it is one element (by
literal subscripts) or a scalar, its name; `nothing` for a whole array or an element by a variable subscript (any of them).
"""
const Touch = Tuple{Symbol, Union{Nothing, Symbol}}

touchOf(c::IR.CREF)::Touch =
  (crefBase(c.cref), (c.ty isa IR.T_ARRAY || !all(IR.isLiteralSubscript, IR.subscripts(c.cref))) ? nothing : c.cref.name)

"Whether two references may be the same element: of one variable, the same element or one of them all of it."
overlaps(a::Touch, b::Touch)::Bool = a[1] === b[1] && (a[2] === nothing || b[2] === nothing || a[2] === b[2])

"""
    whenEquationOrder(stmts)

The statements of a when-equation's branch (its equations, `whenStatements`) in an order that solves them, as the
equations of a when-equation have none (MLS 8.3.5): the assignments each after those of what it reads (outside
`pre()`; in their order where they read each other in a cycle), then the reinit, assert, terminate() and calls in their
order (they read the solved values: `if n >= 3 then terminate() end if; n = pre(n) + 1` stops at the third tick). An
if-statement of both is split in two (its assignments, its others).
"""
function whenEquationOrder(stmts::Vector{IR.Statement})::Vector{IR.Statement}
  local assigning::Vector{IR.Statement} = IR.Statement[]
  local effects::Vector{IR.Statement} = IR.Statement[]
  for s in stmts
    local (a::Union{Nothing, IR.Statement}, e::Union{Nothing, IR.Statement}) = splitWhenStatement(s)
    a === nothing || push!(assigning, a)
    e === nothing || push!(effects, e)
  end
  local writes::Vector{Vector{Touch}} = Vector{Touch}[assignedTouches!(Touch[], s) for s in assigning]
  local reads::Vector{Vector{Touch}} = Vector{Touch}[readTouches!(Touch[], s) for s in assigning]
  local readsFrom::Function = (i::Int, j::Int) -> any(r::Touch -> any(w::Touch -> overlaps(r, w), writes[j]), reads[i])
  local ordered::Vector{IR.Statement} = IR.Statement[]
  local done::BitVector = falses(length(assigning))
  while !all(done)
    #= the first one in order whose reads no other one left writes; in a cycle the first left =#
    local k::Union{Nothing, Int} = findfirst(i -> !done[i] &&
      !any(j -> j != i && !done[j] && readsFrom(i, j), eachindex(assigning)), eachindex(assigning))
    k === nothing && (k = findfirst(!, done))
    push!(ordered, assigning[k])
    done[k] = true
  end
  return vcat(ordered, effects)
end

"""
A when-equation's statement as its assignments and its other statements (reinit, assert, terminate(), calls), each
`nothing` where it has none: an if-statement of both as two with the same conditions.
"""
function splitWhenStatement(s::IR.Statement)::Tuple{Union{Nothing, IR.Statement}, Union{Nothing, IR.Statement}}
  s isa IR.ASSIGNMENT && return (s, nothing)
  s isa IR.IF_STATEMENT || return (nothing, s)
  local parts::Function = (body::Vector{IR.Statement}) -> begin
    local (a::Vector{IR.Statement}, e::Vector{IR.Statement}) = (IR.Statement[], IR.Statement[])
    for x in body
      local (xa::Union{Nothing, IR.Statement}, xe::Union{Nothing, IR.Statement}) = splitWhenStatement(x)
      xa === nothing || push!(a, xa)
      xe === nothing || push!(e, xe)
    end
    (a, e)
  end
  local branchParts::Vector{Tuple{Vector{IR.Statement}, Vector{IR.Statement}}} = [parts(b.body) for b in s.branches]
  local (elseA::Vector{IR.Statement}, elseE::Vector{IR.Statement}) = parts(s.elseBody)
  local half::Function = (k::Int, elseBody::Vector{IR.Statement}) -> begin
    (all(bp -> isempty(bp[k]), branchParts) && isempty(elseBody)) && return nothing
    IR.IF_STATEMENT(IR.StatementBranch[IR.StatementBranch(b.condition, bp[k]) for (b, bp) in zip(s.branches, branchParts)],
                    elseBody, s.source)
  end
  return (half(1, elseA), half(2, elseE))
end

"What a statement assigns (`Touch`)."
function assignedTouches!(out::Vector{Touch}, s::IR.Statement)::Vector{Touch}
  if s isa IR.ASSIGNMENT
    for t in (s.lhs isa IR.TUPLE ? s.lhs.elements : s.lhs isa IR.ARRAY ? s.lhs.elements : IR.Expression[s.lhs])
      t isa IR.CREF && push!(out, touchOf(t))
    end
  elseif s isa IR.IF_STATEMENT
    foreach(x -> assignedTouches!(out, x), Iterators.flatten((Iterators.flatten(b.body for b in s.branches), s.elseBody)))
  end
  return out
end

"What a statement reads outside `pre()` (its values and conditions; `Touch`)."
function readTouches!(out::Vector{Touch}, s::IR.Statement)::Vector{Touch}
  local visit::Function = (e::IR.Expression) -> IR.mapExpTopDown(e) do x
    IR.isBuiltinCall(x, :pre) && return (x, false)
    x isa IR.CREF && push!(out, touchOf(x))
    return (x, true)
  end
  if s isa IR.ASSIGNMENT
    visit(s.rhs)
  elseif s isa IR.IF_STATEMENT
    for b in s.branches
      visit(b.condition)
      foreach(x -> readTouches!(out, x), b.body)
    end
    foreach(x -> readTouches!(out, x), s.elseBody)
  end
  return out
end

"A reference's variable up to its first part with subscripts (`x` of `x[2]`, `c` of `c[1].v`), else its name."
crefBase(c::IR.ComponentRef)::Symbol = isempty(IR.subscripts(c)) ? c.name : IR.upToFirstSubscripted(c).name

const STATE_SELECTS = Dict("never" => IR.NEVER, "avoid" => IR.AVOID, "default" => IR.DEFAULT,
                           "prefer" => IR.PREFER, "always" => IR.ALWAYS)

"The variables `statements` assign, in the order of their first assignment."
function assignedCrefs(statements::Vector{IR.Statement})::Vector{IR.ComponentRef}
  local out = IR.ComponentRef[]
  local seen = Set{Symbol}()
  local visit(s::IR.ASSIGNMENT) = foreach(t -> (t isa IR.CREF && !(t.cref.name in seen)) && (push!(seen, t.cref.name); push!(out, t.cref)),
                                         s.lhs isa IR.TUPLE ? s.lhs.elements : IR.Expression[s.lhs])
  local visit(s::Union{IR.IF_STATEMENT, IR.WHEN_STATEMENT}) =
    (foreach(b -> foreach(visit, b.body), s.branches); s isa IR.IF_STATEMENT && foreach(visit, s.elseBody))
  local visit(s::Union{IR.FOR_STATEMENT, IR.WHILE_STATEMENT}) = foreach(visit, s.body)
  local visit(s::IR.Statement) = nothing
  foreach(visit, statements)
  return out
end

"""
    structuralBindingReads!(variables)

The parameters a structural parameter's binding reads structural too, transitively (the frontend marks the one it
evaluated for the structure, `R` of `if R > 0` with `R = 2 * R0`, not `R0`; omc's frontend marks both): an override of
`R0` would change the structure the code was generated for.
"""
function structuralBindingReads!(variables::IR.VariableSet)::Nothing
  local todo::Vector{IR.Variable} = IR.Variable[v for v in variables if v.kind isa IR.PARAMETER && v.kind.structural]
  while !isempty(todo)
    local v::IR.Variable = pop!(todo)
    v.binding === nothing && continue
    IR.foreachExp(v.binding) do x::IR.Expression
      x isa IR.CREF || return nothing
      local w::Union{Nothing, IR.Variable} = get(variables, x.cref.name, nothing)
      (w !== nothing && w.kind isa IR.PARAMETER && !w.kind.structural) || return nothing
      w.kind = IR.PARAMETER(true)
      push!(todo, w)
      return nothing
    end
  end
  return nothing
end

"""
    withInlineDerivatives(functions)

`functions` with a derivative for each `Inline = true` function without one whose body is
`y := g(a1, ..., an)`, `g` with a derivative annotation (`inlineDerivative`): the derivative
added after the functions and the function annotated with it. OpenModelica inlines such a
function before the index reduction (MSL IF97's `rho_ph(p, h, phase, region) :=
rho_props_ph(p, h, waterBaseProp_ph(p, h, phase, region))`) and evaluates the auxiliary record
once. Differentiated inlined, the record was a call argument of `g`'s derivative in an
equation, and the code generation evaluates such a call once per field of the record (16 for
IF97's, MSL Fluid's MeasuringTemperature 5 times slower); in a function body it is evaluated
once.
"""
function withInlineDerivatives(functions::Vector{IR.ModelicaFunction})::Vector{IR.ModelicaFunction}
  local byName = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in functions)
  local out = copy(functions)
  for (k, f) in enumerate(functions)
    local derivative = inlineDerivative(f, byName)
    derivative === nothing && continue
    out[k] = IR.ModelicaFunction(f.name, f.inputs, f.outputs, f.locals, f.body, f.external,
                                 IR.FunctionDerivative[IR.FunctionDerivative(derivative.name, 1, Int[])], f.impure, f.inline,
                                 f.source)
    push!(out, derivative)
  end
  return out
end

"""
    inlineDerivative(f, byName)

The derivative of an `Inline = true` function `f` without a derivative annotation whose body
is `y := g(a1, ..., an)`, `g` with a derivative annotation of order 1: `g`'s derivative function
at the same arguments, named `f__inlineDer`. It takes `f`'s inputs and a derivative for each
Real one. Of `g`'s arguments, a Real one that takes a derivative must be one of `f`'s inputs or
a literal; another kind of one `g`'s annotation excludes (IF97's auxiliary record). `nothing`
for another function.
"""
function inlineDerivative(f::IR.ModelicaFunction, byName::Dict{Symbol, IR.ModelicaFunction})::Union{Nothing, IR.ModelicaFunction}
  (f.inline && isempty(f.derivatives) && f.external === nothing && length(f.outputs) == 1 && length(f.body) == 1) ||
    return nothing
  local out = only(f.outputs)
  out.ty isa IR.T_REAL || return nothing
  all(v -> v.ty isa Union{IR.T_REAL, IR.T_INTEGER, IR.T_BOOLEAN}, f.inputs) || return nothing
  local stmt = only(f.body)
  (stmt isa IR.ASSIGNMENT && stmt.lhs isa IR.CREF && stmt.lhs.cref.name === out.name && stmt.rhs isa IR.CALL &&
   !stmt.rhs.attributes.builtin) || return nothing
  local call = stmt.rhs
  local g = get(byName, call.name, nothing)
  g === nothing && return nothing
  local gk = findfirst(d -> d.order == 1, g.derivatives)
  (gk === nothing || length(call.arguments) != length(g.inputs)) && return nothing
  local gd = g.derivatives[gk]
  local derOf = Dict{Symbol, IR.Expression}()
  local derInputs = IR.FunctionVariable[]
  for v in f.inputs
    v.ty isa IR.T_REAL || continue
    local name = Symbol(v.name, "__der")
    push!(derInputs, IR.FunctionVariable(name, v.ty, nothing, Pair{Symbol, IR.Expression}[]))
    derOf[v.name] = IR.CREF(IR.ComponentRef(name, v.ty), v.ty)
  end
  local derArgs = IR.Expression[]
  for (i, a) in enumerate(call.arguments)
    local ty = g.inputs[i].ty
    (i in gd.excludedInputs || ty isa Union{IR.T_INTEGER, IR.T_BOOLEAN, IR.T_ENUMERATION, IR.T_STRING}) && continue
    ty isa IR.T_REAL || return nothing
    local d = a isa IR.CREF ? get(derOf, a.cref.name, nothing) : a isa Union{IR.REAL, IR.INTEGER} ? IR.REAL(0.0) : nothing
    d === nothing && return nothing
    push!(derArgs, d)
  end
  local derCall = IR.CALL(gd.derivative, IR.Expression[call.arguments..., derArgs...], out.ty, call.attributes)
  return IR.ModelicaFunction(Symbol(f.name, "__inlineDer"), vcat(f.inputs, derInputs), IR.FunctionVariable[out],
                             IR.FunctionVariable[], IR.Statement[IR.ASSIGNMENT(stmt.lhs, derCall, stmt.source)], nothing,
                             IR.FunctionDerivative[], f.impure, false, f.source)
end

end # module LowerUtil
