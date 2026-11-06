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

#= Traversal of SimCode expressions, statements and equations, by `Traversal` (IR/Traversal.jl):
   what a node's field holds is declared here, per type; every traversal follows from it. =#

"Expressions and statements: a traversal's children, handed to its `recur`."
const TraversalChild = Union{Exp, Statement}

"Iterators, branches, when-clauses and equations: they hold children, traversed field by field."
const TraversalContainer = Union{Iterator, Branch, WHEN_STMTS, Equation, Construct}

#= Anything else (types, references, sources, attributes) is a leaf, kept. =#
Traversal.traverseField(x::TraversalChild, recur::R, arg::A) where {R, A} = recur(x, arg)
Traversal.traverseField(x::TraversalContainer, recur::R, arg::A) where {R, A} = traverseFields(x, recur, arg)
Traversal.traverseField(xs::Vector{<:Union{TraversalChild, TraversalContainer}}, recur::R, arg::A) where {R, A} =
  traverseElements(xs, recur, arg)
Traversal.traverseField(xss::Vector{<:Vector{<:Union{TraversalChild, TraversalContainer}}}, recur::R, arg::A) where {R, A} =
  traverseElements(xss, recur, arg)

"""
    children(e)

The direct subexpressions of `e`, in field order (a reduction's iterator ranges after its body).
"""
children(e::Exp)::Vector{Exp} = last(traverseFields(e, _pushChild, Exp[]))

_pushChild(x::Exp, out::Vector{Exp})::Tuple{Exp, Vector{Exp}} = (push!(out, x); (x, out))

"""
    mapChildren(g, e)

`e` with `g` applied to each of its direct subexpressions, one level; `e` itself if none changed.
"""
mapChildren(g::G, e::Exp) where {G} = first(traverseFields(e, _Apply(g), nothing))

"`g` as a traversal's `recur`: a child replaced by `g(child)`."
struct _Apply{G}
  g::G
end
(m::_Apply)(x::Exp, a::Nothing) = (m.g(x), a)

"`mapExp` of `f` as a callable, for `mapChildren`."
struct _BottomUp{F}
  f::F
end
(m::_BottomUp)(e::Exp) = mapExp(m.f, e)

"""
    mapExp(f, e)

`e` with every subexpression `s` replaced by `f(s)`, bottom-up: `f` sees a node whose children
are already mapped.
"""
mapExp(f::F, e::Exp) where {F} = f(mapChildren(_BottomUp(f), e))

"""
    foreachExp(f, e)

Call `f` on `e` and on every subexpression, top-down.
"""
function foreachExp(f::F, e::Exp)::Nothing where {F}
  f(e)
  traverseFields(e, _Visit(f), nothing)
  return nothing
end

"`foreachExp` of `f` as a traversal's `recur`."
struct _Visit{F}
  f::F
end
(v::_Visit)(x::Exp, a::Nothing) = (foreachExp(v.f, x); (x, a))

"Whether `p` holds for `e` or one of its subexpressions."
function anyExp(p::F, e::Exp)::Bool where {F}
  p(e) && return true
  return last(traverseFields(e, _Any(p), false))
end

"`anyExp` of `p` as a traversal's `recur`: the children after the first that holds are not visited."
struct _Any{F}
  p::F
end
(s::_Any)(x::Exp, found::Bool) = (x, found || anyExp(s.p, x))

"""
    mapStatementExps(g, s)

`s` with each of its expressions `e` (of nested statements too) replaced by `g(e)`, whole;
the targets of `reinit` and of OM.jl's structural changes (references, not expressions) kept.
"""
mapStatementExps(g::G, s::Statement) where {G} = first(traverseFields(s, _StatementExps(g), nothing))

mapStatementExps(g::G, body::Vector{Statement}) where {G} = Statement[mapStatementExps(g, s) for s in body]

"`g` as a traversal's `recur` over a statement: an expression replaced by `g` of it, a nested statement mapped in turn."
struct _StatementExps{G}
  g::G
end
(m::_StatementExps)(e::Exp, a::Nothing) = (m.g(e), a)
(m::_StatementExps)(s::Statement, a::Nothing) = (mapStatementExps(m.g, s), a)

"`s` with `f` mapped (as by `mapExp`) over its expressions; the targets of `reinit` and of OM.jl's structural changes kept."
mapStatement(f::F, s::Statement) where {F} = mapStatementExps(e -> mapExp(f, e), s)

mapStatements(f::F, body::Vector{Statement}) where {F} = Statement[mapStatement(f, s) for s in body]

"""
    crefNames!(names, e; dynamic, element = elementKey)

The names of the variables `e` reads, in order: a reference's name (with its subscripts,
`x[1]` of an array element), a derivative's state's, a `pre(x)`'s variable;
an array read by literal subscripts also as its element, `element(c, index)` (`elementKey`: its reference's flat
name). An array read by a subscript that reads a variable is the array's read only; with
`dynamic`, that array (its `arrayBase`) recorded there.
"""
function crefNames!(names::OrderedSet{String}, e::Exp; dynamic::Union{Nothing, Set{Symbol}} = nothing,
                    element::K = elementKey)::OrderedSet{String} where {K}
  foreachRead(e) do c::SimCref, index::Union{Nothing, Vector{Int}}, isDynamic::Bool
    if isDynamic
      (dynamic !== nothing && c.ref !== nothing) && push!(dynamic, arrayBase(c.ref))
    else
      push!(names, index === nothing ? crefName(c) : element(c, index))
    end
  end
  return names
end

"""
    identifierNames!(names, e) -> OrderedSet{String}

`crefNames!` in the code view (`codeView`): the identifiers of the variables `e` reads, an array element read by
literal subscripts by its array's identifier and the subscripts (`crefName(c, index)`: `a_b[1][2]`), as the
generated code indexes it.
"""
identifierNames!(names::OrderedSet{String}, e::Exp)::OrderedSet{String} = crefNames!(names, e; element = crefName)

"""
    foreachRead(f, e)

`f(c, index, dynamic)` for each reference `e` reads, in order, the one traversal of `crefNames!`, `crefReads!` and
`dynamicArrays!`: a reference, a derivative's state, a `pre(x)`'s variable (`index` nothing); an array read by literal
subscripts, `c` the array and `index` the subscripts (its element's name `crefName(c, index)`, its reference
`elementRead(c, index)`, made only where asked); by a subscript that reads a variable, the array, `dynamic`.
"""
function foreachRead(f::F, e::Exp)::Nothing where {F}
  if e isa Union{EXP_CREF, DER, PRE}
    f(e.cref, nothing, false)
  elseif e isa ASUB && e.exp isa EXP_CREF
    if all(s::Exp -> s isa ICONST, e.subs)
      f(e.exp.cref, Int[(s::ICONST).value for s in e.subs], false)
    elseif any(s::Exp -> anyExp(x::Exp -> x isa Union{EXP_CREF, DER, PRE}, s), e.subs)
      f(e.exp.cref, nothing, true)
    end
  end
  traverseFields(e, _VisitReads(f), nothing)
  return nothing
end

"`foreachRead` of `f` as a traversal's `recur`."
struct _VisitReads{F}
  f::F
end
(v::_VisitReads)(x::Exp, a::Nothing) = (foreachRead(v.f, x); (x, a))

"""
    Read(cref, dynamic)

A reference an expression reads (`crefReads!`): a variable's, an array element's by literal subscripts (`elementRead`),
or, `dynamic`, an array read by a subscript that reads a variable, whose element no subscript names.
"""
struct Read <: Node
  cref::SimCref
  dynamic::Bool
end

"""
    crefReads!(reads, e) -> Vector{Read}

The references `e` reads, in order (`foreachRead`), an element's by literal subscripts as its reference (`elementRead`).
"""
function crefReads!(reads::Vector{Read}, e::Exp)::Vector{Read}
  foreachRead(e) do c::SimCref, index::Union{Nothing, Vector{Int}}, isDynamic::Bool
    push!(reads, Read(index === nothing ? c : elementRead(c, index), isDynamic))
  end
  return reads
end

"The arrays `e` reads by a subscript that reads a variable (`foreachRead`'s dynamic reads, by `arrayBase`), pushed onto `dynamic`."
function dynamicArrays!(dynamic::Set{Symbol}, e::Exp)::Set{Symbol}
  foreachRead(e) do c::SimCref, _::Union{Nothing, Vector{Int}}, isDynamic::Bool
    (isDynamic && c.ref !== nothing) && push!(dynamic, arrayBase(c.ref))
  end
  return dynamic
end

"""
    elementKey(c, index) -> String

The key of the element `index` (literal subscripts) of the array reference `c` in the simulation code: the flat name
of its reference (`elementRead`: `A[1,2]` of `A`, `c[2].T` of `c` and `[2]`... as the IR names it); without a
reference (a test's) its name `crefName(c, index)`.
"""
elementKey(c::SimCref, index::Vector{Int})::String = c.ref === nothing ? crefName(c, index) : refKey(irRef(elementRead(c, index)))

"""
    readKey(c) -> String

The key of the variable the reference `c` reads: its symbol, an element of an array kept as an array (`c.subs`) its
reference's flat name (`x[1,2]`); without a reference (a test's) its name `crefName(c)`.
"""
readKey(c::SimCref)::String = (isempty(c.subs) || c.ref === nothing) ? crefName(c) : refKey(c.ref)

"""
    elementRead(c, index) -> SimCref

The element `index` (literal subscripts) of the array reference `c`: its key and subscripts
followed by `index`, its IR reference the array's subscripted on its last part (`x[1,2]` of `x`).
"""
function elementRead(c::SimCref, index::Vector{Int})::SimCref
  local ref::Union{Nothing, IR.ComponentRef} = c.ref === nothing ? nothing : IR.elementCref(c.ref, index)
  return SimCref(c.sym, Int[c.subs; index], ref)
end

"""
    foreachStatementExp(f, body, statement = noStatement)

`f(e)` for each expression of the statements `body`, nested statements' too, in field order (a branch's condition
before its body, a loop's range before its body); `statement(s)` on each statement before its expressions. The
targets of `reinit` and of OM.jl's structural changes are references, not expressions (`statementTargets!`).
"""
function foreachStatementExp(f::F, body::Vector{Statement}, statement::G = noStatement)::Nothing where {F, G}
  local visit::_StatementVisit{F, G} = _StatementVisit(f, statement)
  foreach(s::Statement -> visit(s, nothing), body)
  return nothing
end

"No visit of a statement (`foreachStatementExp`'s default)."
noStatement(::Statement)::Nothing = nothing

"`foreachStatementExp` as a traversal's `recur`: an expression visited, a statement visited and traversed in turn."
struct _StatementVisit{F, G}
  f::F
  statement::G
end
(v::_StatementVisit)(e::Exp, a::Nothing) = (v.f(e); (e, a))
(v::_StatementVisit)(s::Statement, a::Nothing) = (v.statement(s); traverseFields(s, v, nothing); (s, a))

"The references a statement holds outside its expressions, pushed onto `out`: a `reinit`'s state, OM.jl's structural changes' components."
statementTargets!(out::Vector{SimCref}, s::REINIT)::Vector{SimCref} = push!(out, s.stateVar)
statementTargets!(out::Vector{SimCref}, s::RECOMPILATION)::Vector{SimCref} = push!(out, s.componentToChange)
statementTargets!(out::Vector{SimCref}, s::AGENTIC_RECOMPILATION)::Vector{SimCref} = append!(out, s.componentsToChange)
statementTargets!(out::Vector{SimCref}, ::Statement)::Vector{SimCref} = out

"""
    foreachEquationExp(f, simCode; eliminated = false, statement = (place, s) -> nothing)

`f(place, e)` for each expression of the simulation code's equations, `place` the equation's (`(:residualEquations, 3)`):
the residual equations (with `eliminated` the eliminated ones too), the initial equations' sides, the if-equations'
conditions and branches, the when-equations' conditions and statements (`foreachStatementExp`), the initial
algorithms' statements, the asserts' conditions and messages; `statement(place, s)` on each statement before its
expressions.
"""
function foreachEquationExp(f::F, simCode::SIM_CODE; eliminated::Bool = false, statement::G = noPlacedStatement)::Nothing where {F, G}
  foreach(((i, eq),) -> f((:residualEquations, i), eq.exp), enumerate(simCode.residualEquations))
  eliminated && foreach(((i, eq),) -> f((:eliminatedEquations, i), eq.exp), enumerate(simCode.eliminatedEquations))
  foreach(((i, eq),) -> foreach(e::Exp -> f((:initialEquations, i), e), equationExps(eq)), enumerate(simCode.initialEquations))
  for (i, ifEq) in enumerate(simCode.ifEquations), b in ifEq.branches
    f((:ifEquations, i), b.condition)
    foreach(eq -> f((:ifEquations, i), eq.exp), b.residualEquations)
  end
  for (i, w) in enumerate(simCode.whenEquations)
    local stmts::Union{Nothing, WHEN_STMTS} = w.whenEquation
    while stmts !== nothing
      f((:whenEquations, i), stmts.condition)
      foreachStatementExp(e::Exp -> f((:whenEquations, i), e), stmts.body, s::Statement -> statement((:whenEquations, i), s))
      stmts = stmts.elsewhenPart
    end
  end
  for (i, a) in enumerate(simCode.initialAlgorithms)
    foreachStatementExp(e::Exp -> f((:initialAlgorithms, i), e), a.statements, s::Statement -> statement((:initialAlgorithms, i), s))
  end
  foreach(((i, a),) -> (f((:asserts, i), a.condition); f((:asserts, i), a.message)), enumerate(simCode.asserts))
  return nothing
end

"No visit of a placed statement (`foreachEquationExp`'s default)."
noPlacedStatement(::Tuple{Symbol, Int}, ::Statement)::Nothing = nothing

"A reference's name: its symbol, with its Integer subscripts (`A[1][2]`)."
crefName(c::SimCref)::String = isempty(c.subs) ? String(c.sym) : string(c.sym, join(("[$(i)]" for i in c.subs)))

"The name of the element `index` of the array reference `c` (`crefName` of `elementRead`), without making its reference."
crefName(c::SimCref, index::Vector{Int})::String = string(crefName(c), join(("[$(i)]" for i in index)))

#= ==================================================================================== =#
#= Traversals with an accumulator (the passes')                                         =#
#= ==================================================================================== =#

"""
    traverseExpTopDown(e, visitor, arg)

`e` traversed top-down by `visitor(x, arg) -> (newX, descend, newArg)`: `newX` replaces `x`,
its children are traversed in turn if `descend`; `arg` is threaded through. A node whose
children did not change is returned itself (`===`).
"""
function traverseExpTopDown(e::Exp, visitor::F, arg::T)::Tuple{Exp, T} where {F, T}
  local (out, descend, a) = visitor(e, arg)
  descend || return (out, a)
  return traverseFields(out, (x, acc) -> traverseExpTopDown(x, visitor, acc), a)
end

"""
    traverseExpBottomUp(e, visitor, arg)

`e` traversed bottom-up by `visitor(x, arg) -> (newX, newArg)`: `x`'s children are traversed
first.
"""
function traverseExpBottomUp(e::Exp, visitor::F, arg::T)::Tuple{Exp, T} where {F, T}
  local (out, a) = traverseFields(e, (x, acc) -> traverseExpBottomUp(x, visitor, acc), arg)
  return visitor(out, a)
end
