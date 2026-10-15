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

#= Traversal of SimCode expressions and statements. =#

"""
    children(e)

The direct subexpressions of `e` (a reduction's iterator ranges included).
"""
children(::Union{ICONST, RCONST, BCONST, SCONST, ENUM_LITERAL, EXP_CREF, DER, PRE, WILD})::Vector{Exp} = Exp[]
children(e::Union{BINARY, LBINARY, RELATION})::Vector{Exp} = Exp[e.exp1, e.exp2]
children(e::Union{UNARY, LUNARY, CAST, TSUB, RSUB})::Vector{Exp} = Exp[e.exp]
children(e::IFEXP)::Vector{Exp} = Exp[e.cond, e.thenExp, e.elseExp]
children(e::ARRAY_EXP)::Vector{Exp} = copy(e.elements)
children(e::RANGE)::Vector{Exp} = e.step === nothing ? Exp[e.start, e.stop] : Exp[e.start, e.step, e.stop]
children(e::ASUB)::Vector{Exp} = Exp[e.exp, e.subs...]
children(e::Union{CALL, PARTEVALFUNCTION})::Vector{Exp} = copy(e.args)
children(e::RECORD)::Vector{Exp} = copy(e.exps)
children(e::TUPLE)::Vector{Exp} = copy(e.elements)
children(e::REDUCTION)::Vector{Exp} = Exp[e.body, (it.range for it in e.iterators)...]

"""
    mapChildren(g, e)

`e` with `g` applied to each of its direct subexpressions, one level.
"""
mapChildren(g::G, e::Union{ICONST, RCONST, BCONST, SCONST, ENUM_LITERAL, EXP_CREF, DER, PRE, WILD}) where {G} = e
mapChildren(g::G, e::BINARY) where {G} = BINARY(g(e.exp1), e.op, g(e.exp2))
mapChildren(g::G, e::LBINARY) where {G} = LBINARY(g(e.exp1), e.op, g(e.exp2))
mapChildren(g::G, e::RELATION) where {G} = RELATION(g(e.exp1), e.op, g(e.exp2), e.index)
mapChildren(g::G, e::UNARY) where {G} = UNARY(e.op, g(e.exp))
mapChildren(g::G, e::LUNARY) where {G} = LUNARY(e.op, g(e.exp))
mapChildren(g::G, e::CAST) where {G} = CAST(e.ty, g(e.exp))
mapChildren(g::G, e::TSUB) where {G} = TSUB(g(e.exp), e.index, e.ty)
mapChildren(g::G, e::RSUB) where {G} = RSUB(g(e.exp), e.index, e.fieldName, e.ty)
mapChildren(g::G, e::IFEXP) where {G} = IFEXP(g(e.cond), g(e.thenExp), g(e.elseExp))
mapChildren(g::G, e::ARRAY_EXP) where {G} = ARRAY_EXP(e.ty, e.scalar, Exp[g(x) for x in e.elements])
mapChildren(g::G, e::RANGE) where {G} = RANGE(e.ty, g(e.start), e.step === nothing ? nothing : g(e.step), g(e.stop))
mapChildren(g::G, e::ASUB) where {G} = ASUB(g(e.exp), Exp[g(x) for x in e.subs])
mapChildren(g::G, e::CALL) where {G} = CALL(e.name, Exp[g(x) for x in e.args], e.ty, e.kind)
mapChildren(g::G, e::PARTEVALFUNCTION) where {G} = PARTEVALFUNCTION(e.name, Exp[g(x) for x in e.args], e.argNames, e.ty)
mapChildren(g::G, e::RECORD) where {G} = RECORD(e.name, Exp[g(x) for x in e.exps], e.fieldNames, e.ty)
mapChildren(g::G, e::TUPLE) where {G} = TUPLE(Exp[g(x) for x in e.elements])
mapChildren(g::G, e::REDUCTION) where {G} =
  REDUCTION(e.reducer, g(e.body), Iterator[Iterator(it.name, g(it.range)) for it in e.iterators], e.ty)

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
  for c in children(e)
    foreachExp(f, c)
  end
  return nothing
end

"Whether `p` holds for `e` or one of its subexpressions."
function anyExp(p::F, e::Exp)::Bool where {F}
  p(e) && return true
  return any(c -> anyExp(p, c), children(e))
end

"`s` with `f` mapped (as by `mapExp`) over its expressions; the targets of `reinit` and of OM.jl's structural changes kept."
mapStatement(f::F, s::ASSIGN) where {F} = ASSIGN(mapExp(f, s.left), mapExp(f, s.right), s.source)
mapStatement(f::F, s::IF) where {F} =
  IF(Branch[Branch(mapExp(f, b.condition), mapStatements(f, b.body)) for b in s.branches], mapStatements(f, s.elseBody), s.source)
mapStatement(f::F, s::FOR) where {F} = FOR(Iterator(s.iterator.name, mapExp(f, s.iterator.range)), mapStatements(f, s.body), s.source)
mapStatement(f::F, s::WHILE) where {F} = WHILE(mapExp(f, s.condition), mapStatements(f, s.body), s.source)
mapStatement(f::F, s::WHEN) where {F} =
  WHEN(Branch[Branch(mapExp(f, b.condition), mapStatements(f, b.body)) for b in s.branches], s.source)
mapStatement(f::F, s::ASSERT) where {F} = ASSERT(mapExp(f, s.condition), mapExp(f, s.message), mapExp(f, s.level), s.source)
mapStatement(f::F, s::TERMINATE) where {F} = TERMINATE(mapExp(f, s.message), s.source)
mapStatement(f::F, s::REINIT) where {F} = REINIT(s.stateVar, mapExp(f, s.value), s.source)
mapStatement(f::F, s::NORETCALL) where {F} = NORETCALL(mapExp(f, s.exp), s.source)
mapStatement(f::F, s::RECOMPILATION) where {F} = RECOMPILATION(s.componentToChange, mapExp(f, s.newValue))
mapStatement(f::F, s::Union{RETURN, BREAK, AGENTIC_RECOMPILATION}) where {F} = s

mapStatements(f::F, body::Vector{Statement}) where {F} = Statement[mapStatement(f, s) for s in body]

"""
    crefNames!(names, e)

The names of the variables `e` reads, in order: a reference's name (with its subscripts,
`x[1]` of an array element), a derivative's `der(x)` and its state's, a `pre(x)`'s variable;
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
