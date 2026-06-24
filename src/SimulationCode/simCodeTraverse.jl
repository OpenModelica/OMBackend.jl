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

#= SIM-Exp-native tree traversal primitives.

   Mirrors `FrontendUtil.Util.traverseExpTopDown` / `traverseExpBottomUp` but
   walks `SimulationCode.Exp` natively. The visitor contract matches Util:
       visitor(exp::Exp, arg) -> (newExp::Exp, cont::Bool, newArg)
   - `cont` controls whether to recurse into children (top-down) or short-circuit
   - `newExp` replaces the node if !== inExp
   - `newArg` threads accumulator state

   Eliminates the three-walk-per-call cost of the pre-existing
   `Util.traverseExpTopDown(toDAEExp(simExp), …)` pattern (one walk to convert
   to DAE.Exp, one to traverse, one to convert back via boundary constructor).
=#

# -------------------- traverseExpTopDown --------------------

Base.@nospecializeinfer function traverseExpTopDown(@nospecialize(inExp::Exp),
                                                    @nospecialize(visitor),
                                                    @nospecialize(arg))
  (outExp, cont, outArg) = visitor(inExp, arg)
  if !cont
    return (outExp, outArg)
  end
  return _traverseChildrenTopDown(outExp, visitor, outArg)
end

# Leaves (no children): return unchanged
_traverseChildrenTopDown(e::ICONST,        visitor, arg) = (e, arg)
_traverseChildrenTopDown(e::RCONST,        visitor, arg) = (e, arg)
_traverseChildrenTopDown(e::BCONST,        visitor, arg) = (e, arg)
_traverseChildrenTopDown(e::SCONST,        visitor, arg) = (e, arg)
_traverseChildrenTopDown(e::ENUM_LITERAL,  visitor, arg) = (e, arg)
_traverseChildrenTopDown(e::EXP_CREF,      visitor, arg) = (e, arg)
_traverseChildrenTopDown(e::WILD,          visitor, arg) = (e, arg)

# Unary / single-child shapes
function _traverseChildrenTopDown(e::UNARY, visitor, arg)
  (ne, narg) = traverseExpTopDown(e.exp, visitor, arg)
  return (ne === e.exp ? e : UNARY(e.op, ne), narg)
end
function _traverseChildrenTopDown(e::LUNARY, visitor, arg)
  (ne, narg) = traverseExpTopDown(e.exp, visitor, arg)
  return (ne === e.exp ? e : LUNARY(e.op, ne), narg)
end
function _traverseChildrenTopDown(e::CAST, visitor, arg)
  (ne, narg) = traverseExpTopDown(e.exp, visitor, arg)
  return (ne === e.exp ? e : CAST(e.ty, ne), narg)
end
function _traverseChildrenTopDown(e::TSUB, visitor, arg)
  (ne, narg) = traverseExpTopDown(e.exp, visitor, arg)
  return (ne === e.exp ? e : TSUB(ne, e.index, e.ty), narg)
end
function _traverseChildrenTopDown(e::RSUB, visitor, arg)
  (ne, narg) = traverseExpTopDown(e.exp, visitor, arg)
  return (ne === e.exp ? e : RSUB(ne, e.index, e.fieldName, e.ty), narg)
end

# Binary shapes
function _traverseChildrenTopDown(e::BINARY, visitor, arg)
  (ne1, a1) = traverseExpTopDown(e.exp1, visitor, arg)
  (ne2, a2) = traverseExpTopDown(e.exp2, visitor, a1)
  return ((ne1 === e.exp1 && ne2 === e.exp2) ? e : BINARY(ne1, e.op, ne2), a2)
end
function _traverseChildrenTopDown(e::LBINARY, visitor, arg)
  (ne1, a1) = traverseExpTopDown(e.exp1, visitor, arg)
  (ne2, a2) = traverseExpTopDown(e.exp2, visitor, a1)
  return ((ne1 === e.exp1 && ne2 === e.exp2) ? e : LBINARY(ne1, e.op, ne2), a2)
end
function _traverseChildrenTopDown(e::RELATION, visitor, arg)
  (ne1, a1) = traverseExpTopDown(e.exp1, visitor, arg)
  (ne2, a2) = traverseExpTopDown(e.exp2, visitor, a1)
  return ((ne1 === e.exp1 && ne2 === e.exp2) ? e : RELATION(ne1, e.op, ne2, e.index), a2)
end

# Ternary
function _traverseChildrenTopDown(e::IFEXP, visitor, arg)
  (nc, a1) = traverseExpTopDown(e.cond, visitor, arg)
  (nt, a2) = traverseExpTopDown(e.thenExp, visitor, a1)
  (nl, a3) = traverseExpTopDown(e.elseExp, visitor, a2)
  return ((nc === e.cond && nt === e.thenExp && nl === e.elseExp) ? e :
          IFEXP(nc, nt, nl), a3)
end

# List-bearing shapes (always rebuild — referenceEq on Vector is rare)
function _traverseChildrenTopDown(e::CALL, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.args
    (nx, a) = traverseExpTopDown(x, visitor, a)
    push!(out, nx)
  end
  return (CALL(e.path, out, e.attr), a)
end
function _traverseChildrenTopDown(e::ARRAY_EXP, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.elements
    (nx, a) = traverseExpTopDown(x, visitor, a)
    push!(out, nx)
  end
  return (ARRAY_EXP(e.ty, e.scalar, out), a)
end
function _traverseChildrenTopDown(e::RECORD, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.exps
    (nx, a) = traverseExpTopDown(x, visitor, a)
    push!(out, nx)
  end
  return (RECORD(e.path, out, e.fieldNames, e.ty), a)
end
function _traverseChildrenTopDown(e::TUPLE, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.PR
    (nx, a) = traverseExpTopDown(x, visitor, a)
    push!(out, nx)
  end
  return (TUPLE(out), a)
end
function _traverseChildrenTopDown(e::ASUB, visitor, arg)
  (nex, a1) = traverseExpTopDown(e.exp, visitor, arg)
  local out = Exp[]
  local a = a1
  for s in e.subs
    (ns, a) = traverseExpTopDown(s, visitor, a)
    push!(out, ns)
  end
  return (ASUB(nex, out), a)
end
# Reduction: only the body is SimCode-recursive; info/iterators are opaque.
function _traverseChildrenTopDown(e::REDUCTION, visitor, arg)
  (nb, a) = traverseExpTopDown(e.body, visitor, arg)
  return (nb === e.body ? e : REDUCTION(e.info, nb, e.iterators), a)
end

# -------------------- traverseExpBottomUp --------------------

Base.@nospecializeinfer function traverseExpBottomUp(@nospecialize(inExp::Exp),
                                                     @nospecialize(visitor),
                                                     @nospecialize(arg))
  (newExp, newArg) = _traverseChildrenBottomUp(inExp, visitor, arg)
  return visitor(newExp, newArg)
end

_traverseChildrenBottomUp(e::ICONST,        visitor, arg) = (e, arg)
_traverseChildrenBottomUp(e::RCONST,        visitor, arg) = (e, arg)
_traverseChildrenBottomUp(e::BCONST,        visitor, arg) = (e, arg)
_traverseChildrenBottomUp(e::SCONST,        visitor, arg) = (e, arg)
_traverseChildrenBottomUp(e::ENUM_LITERAL,  visitor, arg) = (e, arg)
_traverseChildrenBottomUp(e::EXP_CREF,      visitor, arg) = (e, arg)
_traverseChildrenBottomUp(e::WILD,          visitor, arg) = (e, arg)

function _traverseChildrenBottomUp(e::UNARY, visitor, arg)
  (ne, narg) = traverseExpBottomUp(e.exp, visitor, arg)
  return (ne === e.exp ? e : UNARY(e.op, ne), narg)
end
function _traverseChildrenBottomUp(e::LUNARY, visitor, arg)
  (ne, narg) = traverseExpBottomUp(e.exp, visitor, arg)
  return (ne === e.exp ? e : LUNARY(e.op, ne), narg)
end
function _traverseChildrenBottomUp(e::CAST, visitor, arg)
  (ne, narg) = traverseExpBottomUp(e.exp, visitor, arg)
  return (ne === e.exp ? e : CAST(e.ty, ne), narg)
end
function _traverseChildrenBottomUp(e::TSUB, visitor, arg)
  (ne, narg) = traverseExpBottomUp(e.exp, visitor, arg)
  return (ne === e.exp ? e : TSUB(ne, e.index, e.ty), narg)
end
function _traverseChildrenBottomUp(e::RSUB, visitor, arg)
  (ne, narg) = traverseExpBottomUp(e.exp, visitor, arg)
  return (ne === e.exp ? e : RSUB(ne, e.index, e.fieldName, e.ty), narg)
end
function _traverseChildrenBottomUp(e::BINARY, visitor, arg)
  (ne1, a1) = traverseExpBottomUp(e.exp1, visitor, arg)
  (ne2, a2) = traverseExpBottomUp(e.exp2, visitor, a1)
  return ((ne1 === e.exp1 && ne2 === e.exp2) ? e : BINARY(ne1, e.op, ne2), a2)
end
function _traverseChildrenBottomUp(e::LBINARY, visitor, arg)
  (ne1, a1) = traverseExpBottomUp(e.exp1, visitor, arg)
  (ne2, a2) = traverseExpBottomUp(e.exp2, visitor, a1)
  return ((ne1 === e.exp1 && ne2 === e.exp2) ? e : LBINARY(ne1, e.op, ne2), a2)
end
function _traverseChildrenBottomUp(e::RELATION, visitor, arg)
  (ne1, a1) = traverseExpBottomUp(e.exp1, visitor, arg)
  (ne2, a2) = traverseExpBottomUp(e.exp2, visitor, a1)
  return ((ne1 === e.exp1 && ne2 === e.exp2) ? e : RELATION(ne1, e.op, ne2, e.index), a2)
end
function _traverseChildrenBottomUp(e::IFEXP, visitor, arg)
  (nc, a1) = traverseExpBottomUp(e.cond, visitor, arg)
  (nt, a2) = traverseExpBottomUp(e.thenExp, visitor, a1)
  (nl, a3) = traverseExpBottomUp(e.elseExp, visitor, a2)
  return ((nc === e.cond && nt === e.thenExp && nl === e.elseExp) ? e :
          IFEXP(nc, nt, nl), a3)
end
function _traverseChildrenBottomUp(e::CALL, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.args
    (nx, a) = traverseExpBottomUp(x, visitor, a)
    push!(out, nx)
  end
  return (CALL(e.path, out, e.attr), a)
end
function _traverseChildrenBottomUp(e::ARRAY_EXP, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.elements
    (nx, a) = traverseExpBottomUp(x, visitor, a)
    push!(out, nx)
  end
  return (ARRAY_EXP(e.ty, e.scalar, out), a)
end
function _traverseChildrenBottomUp(e::RECORD, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.exps
    (nx, a) = traverseExpBottomUp(x, visitor, a)
    push!(out, nx)
  end
  return (RECORD(e.path, out, e.fieldNames, e.ty), a)
end
function _traverseChildrenBottomUp(e::TUPLE, visitor, arg)
  local out = Exp[]
  local a = arg
  for x in e.PR
    (nx, a) = traverseExpBottomUp(x, visitor, a)
    push!(out, nx)
  end
  return (TUPLE(out), a)
end
function _traverseChildrenBottomUp(e::ASUB, visitor, arg)
  (nex, a1) = traverseExpBottomUp(e.exp, visitor, arg)
  local out = Exp[]
  local a = a1
  for s in e.subs
    (ns, a) = traverseExpBottomUp(s, visitor, a)
    push!(out, ns)
  end
  return (ASUB(nex, out), a)
end
function _traverseChildrenBottomUp(e::REDUCTION, visitor, arg)
  (nb, a) = traverseExpBottomUp(e.body, visitor, arg)
  return (nb === e.body ? e : REDUCTION(e.info, nb, e.iterators), a)
end
