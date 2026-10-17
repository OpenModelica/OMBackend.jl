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

#= MTK code generation: the relations of whens and discrete clusters (change conditions,
   relations read through, zero sets and their pins). Relations are keyed by themselves
   (SimCode expressions compare and hash structurally). =#

"""
A synthesized cluster's body can read a Boolean defined by a relation elsewhere (an ideal
thyristor's `fire`, the output of a comparison block). The body reads the relation instead
(outside pre()), and the relation joins the cluster's event sources: the cluster is
evaluated again when it changes, as every discrete equation is at every event (MLS 8.6),
with the relation pinned to its value after the crossing (at the crossing itself `fire` can
still read false). Returns the relations and the body.
"""
function _inlineReadBooleanRelations(rels::Vector{SC.RELATION}, assigns::Vector{Tuple{Symbol, SC.Exp, Bool}},
                                     relationOf::Dict{String, SC.RELATION}, simCode::SC.SIM_CODE)
  local members = Set{String}(string(d) for (d, _, _) in assigns)
  local subst = Dict{String, SC.Exp}()
  local rels2 = copy(rels)
  for (_, rhs, _) in assigns, name in _crefNamesOutsidePre(rhs)
    (name in members || haskey(subst, name)) && continue
    local rel = get(relationOf, name, nothing)
    #= A relation of discretes only changes at events the cluster already sees (the lifter
       does not make such relations event sources either). =#
    (rel === nothing || _withoutContinuousOperand(rel, simCode)) && continue
    subst[name] = rel
    rel in rels2 || push!(rels2, rel)
  end
  isempty(subst) && return (rels, assigns)
  return (rels2, Tuple{Symbol, SC.Exp, Bool}[(d, _inlineOutsidePre(rhs, subst), isInt) for (d, rhs, isInt) in assigns])
end

"""
`e` with the references `subst` names replaced, outside pre(), edge() and change().
"""
function _inlineOutsidePre(e::SC.Exp, subst::Dict{String, SC.Exp})::SC.Exp
  e isa SC.EXP_CREF && return get(subst, SC.crefName(e.cref), e)
  (e isa SC.CALL && e.kind === SC.CALL_BUILTIN && e.name in (:pre, :edge, :change)) && return e
  return SC.mapChildren(x -> _inlineOutsidePre(x, subst), e)
end

"""
The names of the references an expression reads outside pre().
"""
function _crefNamesOutsidePre(e::SC.Exp)::Set{String}
  local names = Set{String}()
  local visit = function (x::SC.Exp)
    SC.isBuiltin(x, :pre) && return
    x isa SC.EXP_CREF && return push!(names, SC.crefName(x.cref))
    foreach(visit, SC.children(x))
  end
  visit(e)
  return names
end

"""
A Boolean condition variable resolved to its defining relation: a residual `0 ~ v - REL`
(`above = x > 0.1`); nothing for another. Lets `change(b)`/`edge(b)` over an observed Boolean
route to an MTK SymbolicContinuousCallback (which reads observed variables and root-finds)
instead of the CallbackSet (which cannot read the observed `b`).
"""
function _condVarRelation(crefName::String, simCode::SC.SIM_CODE)::Union{Nothing, SC.RELATION}
  for req in simCode.residualEquations
    local r = _definingRelation(req)
    (r !== nothing && r[1] == crefName) && return r[2]
  end
  return nothing
end

"""
(name, relation) of a residual `0 ~ v - REL`, or nothing.
"""
function _definingRelation(req::SC.RESIDUAL_EQUATION)::Union{Nothing, Tuple{String, SC.RELATION}}
  local b = req.exp
  (b isa SC.BINARY && b.op === SC.OP_SUB && b.exp1 isa SC.EXP_CREF && b.exp2 isa SC.RELATION) || return nothing
  return (SC.crefName(b.exp1.cref), b.exp2)
end

"""
Every Boolean defined by a relation (_condVarRelation), by name.
"""
function _condVarRelations(simCode::SC.SIM_CODE)::Dict{String, SC.RELATION}
  local out = Dict{String, SC.RELATION}()
  for req in simCode.residualEquations
    local r = _definingRelation(req)
    r === nothing || get!(out, r[1], r[2])
  end
  return out
end

"""
The zero-crossing relations of a `change(...)`/`edge(...)` condition (or an or-chain of them)
into `rels`. The argument may be a relation or a Boolean variable defined by one
(`_condVarRelation`).
"""
function _collectChangeRelations!(rels::Vector{SC.RELATION}, e::SC.Exp, simCode::SC.SIM_CODE)::Bool
  if e isa SC.LBINARY && e.op === SC.OP_OR
    return _collectChangeRelations!(rels, e.exp1, simCode) && _collectChangeRelations!(rels, e.exp2, simCode)
  end
  (SC.isBuiltin(e, :change) || SC.isBuiltin(e, :edge)) || return false
  local inner = first(e.args)
  if inner isa SC.RELATION
    push!(rels, inner)
    return true
  elseif inner isa SC.EXP_CREF
    local rel = _condVarRelation(SC.crefName(inner.cref), simCode)
    rel === nothing && return false
    push!(rels, rel)
    return true
  end
  #= A lifted cluster without relations: no crossing to add. =#
  return _isChangeOfPre(e)
end

"""
`change(pre(v))`: a term of the condition of a cluster the lifter made without relations;
invalid in Modelica, so no user when has it.
"""
_isChangeOfPre(e::SC.Exp)::Bool = SC.isBuiltin(e, :change) && _isPreRead(first(e.args))

"""
pre(v), also behind the sign a negated alias substitution puts in front.
"""
function _isPreRead(a::SC.Exp)::Bool
  (a isa SC.PRE || SC.isBuiltin(a, :pre)) && return true
  a isa Union{SC.UNARY, SC.LUNARY} && return _isPreRead(a.exp)
  return false
end

"""
The relations of a lifted when's condition (empty for a cluster without relations, which
only the event iteration evaluates), or nothing for another condition.
"""
function _extractChangeRelations(cond::SC.Exp, simCode::SC.SIM_CODE)::Union{Nothing, Vector{SC.RELATION}}
  local rels = SC.RELATION[]
  return _collectChangeRelations!(rels, cond, simCode) ? rels : nothing
end

"""
Whether a when-condition marks a synthesized (lifter) when: a literal `initial()` term or a
`change()`/`edge()` call. Such whens carry the implied §17.4.4 initial() term and run their
body at t0; user whens with bare relation conditions must not.
"""
function _condHasInitial(e::SC.Exp)::Bool
  (SC.isBuiltin(e, :initial) || SC.isBuiltin(e, :change) || SC.isBuiltin(e, :edge)) && return true
  e isa SC.LBINARY && return _condHasInitial(e.exp1) || _condHasInitial(e.exp2)
  e isa SC.LUNARY && return _condHasInitial(e.exp)
  return false
end

"""
Whether a when-condition is `edge(...)` (or an or-chain of edges), which fires on the rising
transition only (false to true), unlike `change` (both).
"""
function _isEdgeWhenCondition(e::SC.Exp)::Bool
  SC.isBuiltin(e, :edge) && return true
  (e isa SC.LBINARY && e.op === SC.OP_OR) && return _isEdgeWhenCondition(e.exp1) && _isEdgeWhenCondition(e.exp2)
  return false
end

"""
`e` with each relation `pins` has replaced by its pinned value.
"""
_substRelations(e::SC.Exp, pins::Dict{SC.Exp, Bool})::SC.Exp =
  SC.mapExp(x -> x isa SC.RELATION && haskey(pins, x) ? SC.BCONST(pins[x]) : x, e)

"""
The zero set of an ordering relation: its two operands in a canonical order, and whether
the relation holds when the first canonical operand is the larger one. `a <= b`, `a > b`,
`b >= a`, `b < a` all switch where a - b crosses zero. `nothing` for relations that are not
orderings (==, <>).
"""
function _relationZeroSet(rel::SC.Exp)::Union{Nothing, Tuple{String, String, Bool}}
  (rel isa SC.RELATION && rel.op in (SC.OP_GREATER, SC.OP_GREATEREQ, SC.OP_LESS, SC.OP_LESSEQ)) || return nothing
  local greater = rel.op in (SC.OP_GREATER, SC.OP_GREATEREQ)
  local s1 = repr(rel.exp1)
  local s2 = repr(rel.exp2)
  return s1 <= s2 ? (s1, s2, greater) : (s2, s1, !greater)
end

"""
Group relations that share a zero set, in first-occurrence order.
"""
function _groupRelationsByZeroSet(rels::Vector{SC.RELATION})::Vector{Vector{SC.RELATION}}
  local groups = Vector{SC.RELATION}[]
  local groupOf = Dict{Tuple{String, String}, Int}()
  for r in rels
    local zs = _relationZeroSet(r)
    if zs === nothing
      push!(groups, SC.RELATION[r])
      continue
    end
    local key = (zs[1], zs[2])
    if haskey(groupOf, key)
      push!(groups[groupOf[key]], r)
    else
      push!(groups, SC.RELATION[r])
      groupOf[key] = length(groups)
    end
  end
  return groups
end

"""
Values of every relation in a zero-set group once `rel` has value `val`.
"""
function _zeroSetPins(group::Vector{SC.RELATION}, rel::SC.RELATION, val::Bool)::Dict{SC.Exp, Bool}
  local pins = Dict{SC.Exp, Bool}(rel => val)
  local zs = _relationZeroSet(rel)
  zs === nothing && return pins
  local firstIsLarger = (val == zs[3])
  for r in group
    local rzs = _relationZeroSet(r)
    rzs === nothing || (pins[r] = (firstIsLarger == rzs[3]))
  end
  return pins
end
