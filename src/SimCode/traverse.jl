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
    crefNames!(names, e)

The names of the variables `e` reads, in order: a reference's name (with its subscripts,
`x[1]` of an array element), a derivative's state's, a `pre(x)`'s variable;
an array read by literal subscripts also as its element (`R_T[1][2]`), by a subscript that
reads a variable as `R_T[?]` (`DYNAMIC_SUBSCRIPT_MARK`).
"""
function crefNames!(names::OrderedSet{String}, e::Exp)::OrderedSet{String}
  if e isa EXP_CREF
    push!(names, crefName(e.cref))
  elseif e isa DER
    push!(names, crefName(e.cref))
  elseif e isa PRE
    push!(names, crefName(e.cref))
  elseif e isa ASUB && e.exp isa EXP_CREF
    local base = crefName(e.exp.cref)
    if all(s -> s isa ICONST, e.subs)
      push!(names, string(base, join(("[$(s.value)]" for s in e.subs))))
    elseif any(s -> anyExp(x -> x isa Union{EXP_CREF, DER, PRE}, s), e.subs)
      push!(names, string(base, DYNAMIC_SUBSCRIPT_MARK))
    end
  end
  for c in children(e)
    crefNames!(names, c)
  end
  return names
end

"The name an array read with a subscript that reads a variable is known by (no element is named)."
const DYNAMIC_SUBSCRIPT_MARK = "[?]"

"A reference's name: its symbol, with its Integer subscripts (`A[1][2]`)."
crefName(c::SimCref)::String = isempty(c.subs) ? String(c.sym) : string(c.sym, join(("[$(i)]" for i in c.subs)))

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
