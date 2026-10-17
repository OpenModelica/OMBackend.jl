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

#= The substitution of the variables a pass eliminates: by a parameter (constant propagation),
   by an alias's representative (alias elimination). =#

"""
    Replacement(ref, negated, timeInvariant)

What a pass replaces an eliminated variable by: the reference `ref` (a variable, an array
element), or its negation (`negated`: `a + b = 0` makes `a` read `-b`). `timeInvariant` for a
parameter or a constant: the derivative of the variable it replaces is zero.
"""
struct Replacement
  ref::Exp
  negated::Bool
  timeInvariant::Bool
end

"The replacements of the eliminated variables, by name, in the order they were eliminated."
const Replacements = OrderedDict{String, Replacement}

"The reference `r` replaces a variable by, negated if `r` is."
replacementExp(r::Replacement)::Exp = r.negated ? UNARY(OP_UMINUS, r.ref) : r.ref

"`e` negated if `negate`."
negatedIf(e::Exp, negate::Bool)::Exp = negate ? UNARY(OP_UMINUS, e) : e

"""
    substitute(e, replacements)

`e` with the variables of `replacements` replaced, top-down (a replacement is not visited
again): a reference, or an element read by literal subscripts (`x[1]`), by its replacement;
an array of `replacements` read by subscripts as its replacement subscripted; `der(x)` by
`der` of its replacement (of an expression: the call `der(e)`), negated outside (`der(-y)` as
`-der(y)`), by `0.0` if the replacement is time-invariant; `pre(x)` (`previous(x)`) by that of
its replacement, negated outside (of a constant, by the constant).
"""
function substitute(e::Exp, replacements::Replacements)::Exp
  isempty(replacements) && return e
  return first(traverseExpTopDown(e, substitutionVisitor, replacements))
end

function substitutionVisitor(e::Exp, rs::Replacements)::Tuple{Exp, Bool, Replacements}
  local out = substituted(e, rs)
  return out === nothing ? (e, true, rs) : (out, false, rs)
end

"""
    substituted(e, replacements)

`e` replaced (see `substitute`) if it is a reference to a variable of `replacements` or reads
one as a whole (`der(x)`, `pre(x)`); `nothing` to descend into its subexpressions.
"""
substituted(::Exp, ::Replacements)::Union{Nothing, Exp} = nothing

function substituted(e::EXP_CREF, rs::Replacements)::Union{Nothing, Exp}
  local r = get(rs, crefName(e.cref), nothing)
  return r === nothing ? nothing : replacementExp(r)
end

function substituted(e::ASUB, rs::Replacements)::Union{Nothing, Exp}
  e.exp isa EXP_CREF || return nothing
  local base = crefName(e.exp.cref)
  if all(s -> s isa ICONST, e.subs)
    local r = get(rs, string(base, join(("[$(s.value)]" for s in e.subs))), nothing)
    r === nothing || return replacementExp(r)
  end
  local r = get(rs, base, nothing)
  r === nothing && return nothing
  return negatedIf(ASUB(r.ref, Exp[substitute(s, rs) for s in e.subs]), r.negated)
end

function substituted(e::DER, rs::Replacements)::Union{Nothing, Exp}
  local r = get(rs, crefName(e.cref), nothing)
  r === nothing && return nothing
  r.timeInvariant && return RCONST(0.0)
  r.ref isa EXP_CREF || return negatedIf(CALL(:der, Exp[r.ref], TYPE_REAL(), CALL_BUILTIN), r.negated)
  return negatedIf(DER(r.ref.cref), r.negated)
end

function substituted(e::PRE, rs::Replacements)::Union{Nothing, Exp}
  local r = get(rs, crefName(e.cref), nothing)
  r === nothing && return nothing
  if !(r.ref isa EXP_CREF)
    #= pre() of a constant is the constant =#
    r.timeInvariant && return replacementExp(r)
    return negatedIf(CALL(:pre, Exp[r.ref], e.ty, CALL_BUILTIN), r.negated)
  end
  return negatedIf(PRE(r.ref.cref, e.ty), r.negated)
end

"The variable reference of a replacement (the argument of a `der` or `pre` that read the replaced one)."
function replacementCref(r::Replacement)::SimCref
  r.ref isa EXP_CREF || unsupported("der() or pre() of a variable replaced by an expression", r.ref)
  return r.ref.cref
end

#= ==================================================================================== =#
#= Statements and equations                                                             =#
#= ==================================================================================== =#

"""
    substitute(x, replacements)

A statement, an equation, a when-clause or an initial algorithm with the variables of
`replacements` substituted in its expressions and in its statements, field by field (see
`Traversal`). An assignment to (a reinit of) a variable replaced by a negated one assigns the
representative the negated value (`-x := r` as `x := -r`).
"""
substitute(x::Union{Statement, Equation, Construct, WHEN_STMTS, INITIAL_ALGORITHM}, rs::Replacements) =
  first(traverseFields(x, _substituteChild, rs))

"`substitute` as a traversal's `recur`: an expression or a nested statement substituted."
_substituteChild(x::Union{Exp, Statement}, rs::Replacements) = (substitute(x, rs), rs)

function substitute(s::ASSIGN, rs::Replacements)::Statement
  local (left, right) = withNegationMoved(substitute(s.left, rs), substitute(s.right, rs))
  return ASSIGN(left, right, s.source)
end

function substitute(s::REINIT, rs::Replacements)::Statement
  local r = get(rs, crefName(s.stateVar), nothing)
  local value = substitute(s.value, rs)
  r === nothing && return REINIT(s.stateVar, value, s.source)
  return REINIT(replacementCref(r), negatedIf(value, r.negated), s.source)
end

substitute(body::Vector{Statement}, rs::Replacements)::Vector{Statement} = Statement[substitute(s, rs) for s in body]

"An assignment's sides with a negated target's negation moved to the value."
function withNegationMoved(left::Exp, right::Exp)::Tuple{Exp, Exp}
  (left isa UNARY && left.op === OP_UMINUS) || return (left, right)
  return (left.exp, UNARY(OP_UMINUS, right))
end

"An initial equation substituted: a residual, an equation, an array equation, an if-equation of them; another kept."
substituteInitial(eq::Union{RESIDUAL_EQUATION, EQUATION, ARRAY_EQUATION}, rs::Replacements)::Equation = substitute(eq, rs)
substituteInitial(eq::INLINE_IF_EQUATION, rs::Replacements)::Equation =
  INLINE_IF_EQUATION(Exp[substitute(c, rs) for c in eq.conditions],
                     Vector{Equation}[Equation[substituteInitial(e, rs) for e in b] for b in eq.branchesTrue],
                     Equation[substituteInitial(e, rs) for e in eq.branchElse], eq.source, eq.attr)
substituteInitial(eq::Equation, ::Replacements)::Equation = eq
