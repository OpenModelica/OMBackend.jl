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

#= Canonical names for every cref; enumeration literal paths. =#

struct _CanonicalNameContext
  rename::Dict{String, String}
  known::OrderedSet{String}
  nameMap::OMBackend.NameRewriteMap
end

function _canonicalVariableKey(name::AbstractString)::String
  return OMBackend.canonicalName(name)
end

function _recordNameRewrite!(ctx::_CanonicalNameContext, original::AbstractString,
                             canonical::AbstractString)::String
  local originalName = String(original)
  local canonicalName = String(canonical)
  ctx.nameMap.originalToCanonical[originalName] = canonicalName
  if originalName != canonicalName || !haskey(ctx.nameMap.canonicalToOriginal, canonicalName)
    ctx.nameMap.canonicalToOriginal[canonicalName] = originalName
  end
  return canonicalName
end

function _canonicalVariableKey(name::AbstractString, ctx::_CanonicalNameContext)::String
  #= Honour an explicit override in ctx.rename (e.g. the reserved-name rename of a
     model variable literally named `t`). For every ordinary name the override and
     the plain canonical form coincide, so this is behaviour-preserving except for
     the reserved override. =#
  local override = get(ctx.rename, name, nothing)
  if override !== nothing
    return _recordNameRewrite!(ctx, name, override)
  end
  return _recordNameRewrite!(ctx, name, OMBackend.canonicalName(name))
end

function _originalPathName(path::Absyn.IDENT)::String
  return path.name
end

function _originalPathName(path::Absyn.QUALIFIED)::String
  return Base.string(path.name, ".", _originalPathName(path.path))
end

function _originalPathName(path::Absyn.FULLYQUALIFIED)::String
  return Base.string(".", _originalPathName(path.path))
end

function _originalSubscriptSuffix(subscriptLst)::String
  if listEmpty(subscriptLst)
    return ""
  end
  local buf = IOBuffer()
  for subscript in subscriptLst
    print(buf, "[")
    print(buf, Base.string(subscript))
    print(buf, "]")
  end
  return String(take!(buf))
end

function _originalCrefName(cr::DAE.CREF_IDENT)::String
  return Base.string(cr.ident, _originalSubscriptSuffix(cr.subscriptLst))
end

function _originalCrefName(cr::DAE.CREF_ITER)::String
  return Base.string(cr.ident, _originalSubscriptSuffix(cr.subscriptLst))
end

function _originalCrefName(cr::DAE.CREF_QUAL)::String
  return Base.string(cr.ident,
                     _originalSubscriptSuffix(cr.subscriptLst),
                     ".",
                     _originalCrefName(cr.componentRef))
end

function _originalCrefName(cr::DAE.WILD)::String
  return "_"
end

function _originalCrefName(cr::DAE.OPTIMICA_ATTR_INST_CREF)::String
  return _originalCrefName(cr.componentRef)
end

function _canonicalizeVarKind(kind::SimVarType, ctx::_CanonicalNameContext)::SimVarType
  return @match kind begin
    STATE_DERIVATIVE(varName) => STATE_DERIVATIVE(_canonicalVariableKey(varName, ctx))
    PARAMETER(SOME(bindExp)) => PARAMETER(SOME(_canonicalizeExp(bindExp, ctx)))
    DATA_STRUCTURE(SOME(bindExp)) => DATA_STRUCTURE(SOME(_canonicalizeExp(bindExp, ctx)))
    ARRAY(dims, SOME(bindExp)) => ARRAY(dims, SOME(_canonicalizeExp(bindExp, ctx)))
    ARRAY_PARAMETER(dims, SOME(bindExp)) => ARRAY_PARAMETER(dims, SOME(_canonicalizeExp(bindExp, ctx)))
    STRING(SOME(bindExp)) => STRING(SOME(_canonicalizeExp(bindExp, ctx)))
    _ => kind
  end
end

function _canonicalizeSimVar(sv::SIMVAR, ctx::_CanonicalNameContext)::SIMVAR
  return SIMVAR(_canonicalVariableKey(sv.name, ctx),
                sv.index,
                _canonicalizeVarKind(sv.varKind, ctx),
                sv.attributes)
end

function _canonicalizeSimVarHT(ht::AbstractDict{String, Tuple{Int, SimVar}},
                               ctx::_CanonicalNameContext)
  local out = OrderedDict{String, Tuple{Int, SimVar}}()
  for (name, (idx, sv)) in ht
    local canonicalName = get(ctx.rename, name, nothing)
    if canonicalName === nothing
      canonicalName = _canonicalVariableKey(name, ctx)
    else
      _recordNameRewrite!(ctx, name, canonicalName)
    end
    local newVar = _canonicalizeSimVar(sv, ctx)
    if newVar.name != canonicalName
      newVar = SIMVAR(canonicalName, newVar.index, newVar.varKind, newVar.attributes)
    end
    out[canonicalName] = (idx, newVar)
  end
  return out
end

# SIM-native: walk the SimCode tree, canonicalizing CALL/RECORD paths natively and
# crefs via the DAE ComponentRef canonicalizer (rebuilt as a SimCref). Removes the
# whole-tree DAE round-trip; only the per-cref name canonicalization touches DAE,
# because `_canonicalizeComponentRef` is ComponentRef-shaped (subscripts / rename map).
function _canonicalizeCrefExpSIM(@nospecialize(exp), ctx::_CanonicalNameContext)
  if exp isa EXP_CREF
    local dty = toDAEType(exp.ty)
    local canon = _canonicalizeComponentRef(toDAECref(exp.cref).componentRef, dty, ctx)
    return (toSimExp(DAE.CREF(canon, dty)), false, ctx)
  elseif exp isa CALL
    local cp = OMBackend.canonicalName(exp.path)
    _recordNameRewrite!(ctx, _originalPathName(exp.path), cp)
    return (CALL(Absyn.IDENT(cp), exp.args, exp.attr), true, ctx)
  elseif exp isa PARTEVALFUNCTION
    local cp = OMBackend.canonicalName(exp.path)
    _recordNameRewrite!(ctx, _originalPathName(exp.path), cp)
    return (PARTEVALFUNCTION(Absyn.IDENT(cp), exp.args, exp.ty, exp.origType), true, ctx)
  elseif exp isa RECORD
    local cp = OMBackend.canonicalName(exp.path)
    _recordNameRewrite!(ctx, _originalPathName(exp.path), cp)
    return (RECORD(Absyn.IDENT(cp), exp.exps, exp.fieldNames, exp.ty), true, ctx)
  end
  return (exp, true, ctx)
end
_canonicalizeExp(exp::Exp, ctx::_CanonicalNameContext) =
  traverseExpTopDown(exp, _canonicalizeCrefExpSIM, ctx)[1]

function _canonicalizeExp(@nospecialize(exp), ctx::_CanonicalNameContext)
  local (newExp, _) = Util.traverseExpTopDown(exp, _canonicalizeCrefExp, ctx)
  return newExp
end

function _canonicalizeCrefExp(@nospecialize(exp), ctx::_CanonicalNameContext)
  @match exp begin
    #= An omitted output of a tuple assignment stays one (not a variable `_`). =#
    DAE.CREF(DAE.WILD(), _) => (exp, false, ctx)
    DAE.CREF(cr, ty) => begin
      return (DAE.CREF(_canonicalizeComponentRef(cr, ty, ctx), ty), false, ctx)
    end
    DAE.CALL(path, expLst, attr) => begin
      local canonicalPath = OMBackend.canonicalName(path)
      _recordNameRewrite!(ctx, _originalPathName(path), canonicalPath)
      return (DAE.CALL(Absyn.IDENT(canonicalPath), expLst, attr), true, ctx)
    end
    DAE.RECORD(path, exps, comp, ty) => begin
      local canonicalPath = OMBackend.canonicalName(path)
      _recordNameRewrite!(ctx, _originalPathName(path), canonicalPath)
      return (DAE.RECORD(Absyn.IDENT(canonicalPath), exps, comp, ty), true, ctx)
    end
    DAE.PARTEVALFUNCTION(path, expList, ty, origType) => begin
      local canonicalPath = OMBackend.canonicalName(path)
      _recordNameRewrite!(ctx, _originalPathName(path), canonicalPath)
      return (DAE.PARTEVALFUNCTION(Absyn.IDENT(canonicalPath), expList, ty, origType), true, ctx)
    end
    _ => return (exp, true, ctx)
  end
end

function _stripInnermostSubscripts(cr::DAE.CREF_IDENT)
  return DAE.CREF_IDENT(cr.ident, cr.identType, MetaModelica.nil)
end

function _stripInnermostSubscripts(cr::DAE.CREF_ITER)
  return DAE.CREF_ITER(cr.ident, cr.index, cr.identType, MetaModelica.nil)
end

function _stripInnermostSubscripts(cr::DAE.CREF_QUAL)
  return DAE.CREF_QUAL(cr.ident,
                       cr.identType,
                       cr.subscriptLst,
                       _stripInnermostSubscripts(cr.componentRef))
end

function _stripInnermostSubscripts(cr::DAE.WILD)
  return cr
end

function _innermostSubscripts(cr::DAE.CREF_IDENT)
  return cr.subscriptLst
end

function _innermostSubscripts(cr::DAE.CREF_ITER)
  return cr.subscriptLst
end

function _innermostSubscripts(cr::DAE.CREF_QUAL)
  return _innermostSubscripts(cr.componentRef)
end

function _innermostSubscripts(::DAE.WILD)
  return MetaModelica.nil
end

function _innermostType(cr::DAE.CREF_IDENT)
  return cr.identType
end

function _innermostType(cr::DAE.CREF_ITER)
  return cr.identType
end

function _innermostType(cr::DAE.CREF_QUAL)
  return _innermostType(cr.componentRef)
end

_hasDimensions(dims)::Bool = !isempty(dims)

function _declaredDaeVarCrefType(v::DAE.VAR)::DAE.Type
  local crefTy = _innermostType(v.componentRef)
  if crefTy isa DAE.T_UNKNOWN
    crefTy = v.ty
  elseif !(crefTy isa DAE.T_ARRAY) && v.ty isa DAE.T_ARRAY
    crefTy = v.ty
  end
  if !(crefTy isa DAE.T_ARRAY) && _hasDimensions(v.dims)
    return DAE.T_ARRAY(crefTy, v.dims)
  end
  return crefTy
end

function _canonicalizeComponentRef(cr::DAE.ComponentRef, ty::DAE.Type,
                                   ctx::_CanonicalNameContext)::DAE.ComponentRef
  local originalFull = _originalCrefName(cr)
  local fullName = OMBackend.canonicalName(cr)
  local canonicalFull = get(ctx.rename, originalFull, nothing)
  if canonicalFull === nothing
    canonicalFull = get(ctx.rename, fullName, nothing)
  end
  if canonicalFull === nothing
    canonicalFull = _recordNameRewrite!(ctx, originalFull, fullName)
  else
    _recordNameRewrite!(ctx, originalFull, canonicalFull)
  end
  if canonicalFull in ctx.known
    return DAE.CREF_IDENT(canonicalFull, ty, MetaModelica.nil)
  end

  local baseCr = _stripInnermostSubscripts(cr)
  local originalBase = _originalCrefName(baseCr)
  local baseName = OMBackend.canonicalName(baseCr)
  local canonicalBase = get(ctx.rename, originalBase, nothing)
  if canonicalBase === nothing
    canonicalBase = get(ctx.rename, baseName, nothing)
  end
  if canonicalBase === nothing
    canonicalBase = _recordNameRewrite!(ctx, originalBase, baseName)
  else
    _recordNameRewrite!(ctx, originalBase, canonicalBase)
  end
  local finalSubs = _innermostSubscripts(cr)
  if canonicalBase in ctx.known || !listEmpty(finalSubs)
    return DAE.CREF_IDENT(canonicalBase, _innermostType(cr), finalSubs)
  end

  return DAE.CREF_IDENT(canonicalFull, ty, MetaModelica.nil)
end

function _canonicalizeEquation(eq, ctx::_CanonicalNameContext)
  if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
    return typeof(eq)(_canonicalizeExp(toDAEExp(eq.exp), ctx), eq.source, eq.attr)
  elseif eq isa BDAE.EQUATION
    return BDAE.EQUATION(_canonicalizeExp(eq.lhs, ctx),
                         _canonicalizeExp(eq.rhs, ctx),
                         eq.source,
                         eq.attributes)
  elseif eq isa EQUATION
    return EQUATION(_canonicalizeExp(toDAEExp(eq.lhs), ctx),
                    _canonicalizeExp(toDAEExp(eq.rhs), ctx),
                    eq.source,
                    eq.attr)
  elseif eq isa BDAE.ARRAY_EQUATION
    return BDAE.ARRAY_EQUATION(eq.dimSize,
                               _canonicalizeExp(eq.left, ctx),
                               _canonicalizeExp(eq.right, ctx),
                               eq.source,
                               eq.attr,
                               eq.recordSize)
  elseif eq isa ARRAY_EQUATION
    return ARRAY_EQUATION(eq.dimSize,
                          _canonicalizeExp(toDAEExp(eq.left), ctx),
                          _canonicalizeExp(toDAEExp(eq.right), ctx),
                          eq.source,
                          eq.attr)
  elseif eq isa BDAE.COMPLEX_EQUATION
    return BDAE.COMPLEX_EQUATION(eq.size,
                                 _canonicalizeExp(eq.left, ctx),
                                 _canonicalizeExp(eq.right, ctx),
                                 eq.source,
                                 eq.attr)
  elseif eq isa BDAE.SOLVED_EQUATION
    return BDAE.SOLVED_EQUATION(_canonicalizeComponentRef(eq.componentRef, _innermostType(eq.componentRef), ctx),
                                _canonicalizeExp(eq.exp, ctx),
                                eq.source,
                                eq.attr)
  elseif eq isa BDAE.WHEN_EQUATION || eq isa WHEN_EQUATION || eq isa INITIAL_WHEN_EQUATION
    return typeof(eq)(eq.size,
                      _canonicalizeWhenStmts(eq.whenEquation, ctx),
                      eq.source,
                      eq.attr)
  elseif eq isa BDAE.STRUCTURAL_WHEN_EQUATION
    return BDAE.STRUCTURAL_WHEN_EQUATION(eq.size,
                                         _canonicalizeWhenStmts(eq.whenEquation, ctx),
                                         eq.source,
                                         eq.attr)
  elseif eq isa BDAE.IF_EQUATION
    local newConditions = _mapList(e -> _canonicalizeExp(e, ctx), eq.conditions)
    local newTrue = _mapList(branch -> _mapList(e -> _canonicalizeEquation(e, ctx), branch), eq.eqnstrue)
    local newFalse = _mapList(e -> _canonicalizeEquation(e, ctx), eq.eqnsfalse)
    return BDAE.IF_EQUATION(newConditions, newTrue, newFalse, eq.source, eq.attr)
  elseif eq isa BDAE.ALGORITHM
    return BDAE.ALGORITHM(eq.size, _canonicalizeAlgorithm(eq.alg, ctx), eq.source, eq.expand, eq.attr)
  elseif eq isa ALGORITHM
    return ALGORITHM(eq.size, _canonicalizeAlgorithm(eq.alg, ctx), eq.source, eq.expand, eq.attr)
  elseif eq isa INLINE_IF_EQUATION
    local newConds = DAE.Exp[_canonicalizeExp(c, ctx) for c in eq.conditions]
    local newTrue = Vector{Equation}[Equation[_canonicalizeEquation(e, ctx) for e in br] for br in eq.branchesTrue]
    local newElse = Equation[_canonicalizeEquation(e, ctx) for e in eq.branchElse]
    return INLINE_IF_EQUATION(newConds, newTrue, newElse, eq.source, eq.attr)
  elseif eq isa BDAE.ASSERT_EQUATION
    return BDAE.ASSERT_EQUATION(_canonicalizeExp(eq.condition, ctx),
                                _canonicalizeExp(eq.message, ctx),
                                _canonicalizeExp(eq.level, ctx),
                                eq.source)
  end
  return eq
end

function _recordFunctionVarName!(known::OrderedSet{String}, v::DAE.VAR,
                                 ctx::_CanonicalNameContext)
  local original = _originalCrefName(v.componentRef)
  local canonical = OMBackend.canonicalName(v.componentRef)
  _recordNameRewrite!(ctx, original, canonical)
  push!(known, canonical)
  return nothing
end

function _mapList(f::Function, lst)
  local out = MetaModelica.nil
  for x in lst
    out = f(x) <| out
  end
  return listReverse(out)
end

function _collectAssertCrefNames!(out, asserts)
  for a in asserts
    collectCrefNames!(out, a.condition)
    collectCrefNames!(out, a.message)
  end
  return out
end

#= Asserts read the substituted variables (constants, alias representatives)
   like the equations do, so they never refer to an eliminated unknown. =#
function _substituteInAsserts(asserts::Vector{BDAE.ASSERT_EQUATION}, map;
                              visitor = substituteAliasCref)::Vector{BDAE.ASSERT_EQUATION}
  return BDAE.ASSERT_EQUATION[BDAE.ASSERT_EQUATION(first(Util.traverseExpTopDown(a.condition, visitor, map)),
                                                   first(Util.traverseExpTopDown(a.message, visitor, map)),
                                                   a.level, a.source) for a in asserts]
end

function _mapVectorLike(f::Function, xs)
  local out = typeof(xs)()
  for x in xs
    push!(out, f(x))
  end
  return out
end

function _canonicalizeWhenStmts(whenStmts, ctx::_CanonicalNameContext)
  if whenStmts isa WHEN_STMTS
    local newCondS = _canonicalizeExp(toDAEExp(whenStmts.condition), ctx)
    local newStmtLstS = WhenOperator[_canonicalizeWhenOperator(s, ctx) for s in whenStmts.whenStmtLst]
    local newElseS = whenStmts.elsewhenPart === nothing ? nothing : _canonicalizeWhenStmts(whenStmts.elsewhenPart, ctx)
    return WHEN_STMTS(newCondS, newStmtLstS, newElseS)
  end
  local newCond = _canonicalizeExp(toDAEExp(whenStmts.condition), ctx)
  local newStmtLst = _mapList(stmt -> _canonicalizeWhenOperator(stmt, ctx),
                              whenStmts.whenStmtLst)
  local newElse = @match whenStmts.elsewhenPart begin
    SOME(elseWhenEq) => SOME(_canonicalizeElseWhenPart(elseWhenEq, ctx))
    NONE() => NONE()
    _ => whenStmts.elsewhenPart
  end
  return BDAE.WHEN_STMTS(newCond, newStmtLst, newElse)
end

function _canonicalizeElseWhenPart(elseWhen, ctx::_CanonicalNameContext)
  if elseWhen isa BDAE.WHEN_STMTS || elseWhen isa WHEN_STMTS
    return _canonicalizeWhenStmts(elseWhen, ctx)
  end
  return _canonicalizeEquation(elseWhen, ctx)
end

function _canonicalizeCrefValue(exp::DAE.CREF, ctx::_CanonicalNameContext)::DAE.CREF
  local newExp = _canonicalizeExp(exp, ctx)
  return newExp isa DAE.CREF ? newExp : exp
end

function _canonicalizeWhenOperator(stmt, ctx::_CanonicalNameContext)
  if stmt isa BDAE.ASSIGN || stmt isa ASSIGN
    return typeof(stmt)(_canonicalizeExp(stmt.left, ctx),
                        _canonicalizeExp(stmt.right, ctx),
                        stmt.source)
  elseif stmt isa BDAE.REINIT || stmt isa REINIT
    return typeof(stmt)(_canonicalizeCrefValue(stmt.stateVar, ctx),
                        _canonicalizeExp(stmt.value, ctx),
                        stmt.source)
  elseif stmt isa BDAE.ASSERT || stmt isa ASSERT
    return typeof(stmt)(_canonicalizeExp(stmt.condition, ctx),
                        _canonicalizeExp(stmt.message, ctx),
                        _canonicalizeExp(stmt.level, ctx),
                        stmt.source)
  elseif stmt isa BDAE.TERMINATE || stmt isa TERMINATE
    return typeof(stmt)(_canonicalizeExp(stmt.message, ctx), stmt.source)
  elseif stmt isa BDAE.NORETCALL || stmt isa NORETCALL
    return typeof(stmt)(_canonicalizeExp(stmt.exp, ctx), stmt.source)
  elseif stmt isa BDAE.RECOMPILATION || stmt isa RECOMPILATION
    return typeof(stmt)(_canonicalizeCrefValue(stmt.componentToChange, ctx),
                        _canonicalizeExp(stmt.newValue, ctx))
  elseif stmt isa BDAE.AGENTIC_RECOMPILATION || stmt isa AGENTIC_RECOMPILATION
    return typeof(stmt)([_canonicalizeCrefValue(c, ctx) for c in stmt.componentsToChange],
                        stmt.prompt,
                        stmt.initialEquations)
  end
  return stmt
end

function _canonicalizeBranch(branch::BRANCH, ctx::_CanonicalNameContext)
  return BRANCH(_canonicalizeExp(branch.condition, ctx),
                _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), branch.residualEquations),
                branch.identifier,
                branch.targets,
                branch.isSingular,
                branch.matchOrder,
                branch.equationGraph,
                branch.sccs,
                _canonicalizeSimVarHT(branch.stringToSimVarHT, ctx))
end

function _canonicalizeStructuralTransition(tr::StructuralTransition,
                                           ctx::_CanonicalNameContext)
  if tr isa EXPLICIT_STRUCTURAL_TRANSITION
    return EXPLICIT_STRUCTURAL_TRANSITION(_canonicalVariableKey(tr.fromState, ctx),
                                           _canonicalVariableKey(tr.toState, ctx),
                                           _canonicalizeExp(tr.transitionCondition, ctx))
  elseif tr isa IMPLICIT_STRUCTURAL_TRANSITION
    return IMPLICIT_STRUCTURAL_TRANSITION(tr.size,
                                           _canonicalizeWhenStmts(tr.whenEquation, ctx),
                                           tr.source,
                                           tr.attr)
  end
  return tr
end

function _canonicalizeIfEquation(ifEq::IF_EQUATION, ctx::_CanonicalNameContext)
  return IF_EQUATION(_mapVectorLike(branch -> _canonicalizeBranch(branch, ctx),
                                    ifEq.branches))
end

function _canonicalizeDaeVar(v::DAE.VAR, ctx::_CanonicalNameContext)::DAE.VAR
  local newBinding = @match v.binding begin
    SOME(b) => SOME(_canonicalizeExp(b, ctx))
    NONE() => NONE()
  end
  return DAE.VAR(_canonicalizeComponentRef(v.componentRef, _declaredDaeVarCrefType(v), ctx),
                 v.kind,
                 v.direction,
                 v.parallelism,
                 v.protection,
                 v.ty,
                 newBinding,
                 v.dims,
                 v.connectorType,
                 v.source,
                 v.variableAttributesOption,
                 v.comment,
                 v.innerOuter)
end

function _canonicalizeStatement(stmt::DAE.Statement, ctx::_CanonicalNameContext)::DAE.Statement
  if stmt isa DAE.STMT_ASSIGN
    return DAE.STMT_ASSIGN(stmt.type_,
                           _canonicalizeExp(stmt.exp1, ctx),
                           _canonicalizeExp(stmt.exp, ctx),
                           stmt.source)
  elseif stmt isa DAE.STMT_TUPLE_ASSIGN
    return DAE.STMT_TUPLE_ASSIGN(stmt.type_,
                                 _mapList(e -> _canonicalizeExp(e, ctx), stmt.expExpLst),
                                 _canonicalizeExp(stmt.exp, ctx),
                                 stmt.source)
  elseif stmt isa DAE.STMT_ASSIGN_ARR
    return DAE.STMT_ASSIGN_ARR(stmt.type_,
                               _canonicalizeExp(stmt.lhs, ctx),
                               _canonicalizeExp(stmt.exp, ctx),
                               stmt.source)
  elseif stmt isa DAE.STMT_IF
    return DAE.STMT_IF(_canonicalizeExp(stmt.exp, ctx),
                       _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                       _canonicalizeElse(stmt.else_, ctx),
                       stmt.source)
  elseif stmt isa DAE.STMT_FOR
    return DAE.STMT_FOR(stmt.type_,
                        stmt.iterIsArray,
                        stmt.iter,
                        stmt.index,
                        _canonicalizeExp(stmt.range, ctx),
                        _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                        stmt.source)
  elseif stmt isa DAE.STMT_PARFOR
    return DAE.STMT_PARFOR(stmt.type_,
                           stmt.iterIsArray,
                           stmt.iter,
                           stmt.index,
                           _canonicalizeExp(stmt.range, ctx),
                           _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                           stmt.loopPrlVars,
                           stmt.source)
  elseif stmt isa DAE.STMT_WHILE
    return DAE.STMT_WHILE(_canonicalizeExp(stmt.exp, ctx),
                          _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                          stmt.source)
  elseif stmt isa DAE.STMT_WHEN
    local newElseWhen = @match stmt.elseWhen begin
      SOME(s) => SOME(_canonicalizeStatement(s, ctx))
      NONE() => NONE()
    end
    return DAE.STMT_WHEN(_canonicalizeExp(stmt.exp, ctx),
                         _mapList(c -> _canonicalizeComponentRef(c, _innermostType(c), ctx), stmt.conditions),
                         stmt.initialCall,
                         _mapList(s -> _canonicalizeStatement(s, ctx), stmt.statementLst),
                         newElseWhen,
                         stmt.source)
  elseif stmt isa DAE.STMT_ASSERT
    return DAE.STMT_ASSERT(_canonicalizeExp(stmt.cond, ctx),
                           _canonicalizeExp(stmt.msg, ctx),
                           _canonicalizeExp(stmt.level, ctx),
                           stmt.source)
  elseif stmt isa DAE.STMT_TERMINATE
    return DAE.STMT_TERMINATE(_canonicalizeExp(stmt.msg, ctx), stmt.source)
  elseif stmt isa DAE.STMT_REINIT
    return DAE.STMT_REINIT(_canonicalizeExp(stmt.var, ctx),
                           _canonicalizeExp(stmt.value, ctx),
                           stmt.source)
  elseif stmt isa DAE.STMT_NORETCALL
    return DAE.STMT_NORETCALL(_canonicalizeExp(stmt.exp, ctx), stmt.source)
  elseif stmt isa DAE.STMT_FAILURE
    return DAE.STMT_FAILURE(_mapList(s -> _canonicalizeStatement(s, ctx), stmt.body),
                            stmt.source)
  end
  return stmt
end

function _canonicalizeElse(elseBranch::DAE.Else, ctx::_CanonicalNameContext)::DAE.Else
  if elseBranch isa DAE.ELSEIF
    return DAE.ELSEIF(_canonicalizeExp(elseBranch.exp, ctx),
                      _mapList(s -> _canonicalizeStatement(s, ctx), elseBranch.statementLst),
                      _canonicalizeElse(elseBranch.else_, ctx))
  elseif elseBranch isa DAE.ELSE
    return DAE.ELSE(_mapList(s -> _canonicalizeStatement(s, ctx), elseBranch.statementLst))
  end
  return elseBranch
end

function _canonicalizeAlgorithm(alg::DAE.Algorithm, ctx::_CanonicalNameContext)::DAE.Algorithm
  if alg isa DAE.ALGORITHM_STMTS
    return DAE.ALGORITHM_STMTS(_mapList(s -> _canonicalizeStatement(s, ctx), alg.statementLst))
  end
  return alg
end

function _functionCanonicalNameContext(f, ctx::_CanonicalNameContext)
  local known = OrderedSet{String}(["time", "pi", "e"])
  if hasproperty(f, :inputs)
    for v in f.inputs
      _recordFunctionVarName!(known, v, ctx)
    end
  end
  if hasproperty(f, :outputs)
    for v in f.outputs
      _recordFunctionVarName!(known, v, ctx)
    end
  end
  if hasproperty(f, :locals)
    for v in f.locals
      _recordFunctionVarName!(known, v, ctx)
    end
  end
  return _CanonicalNameContext(ctx.rename, known, ctx.nameMap)
end

function _canonicalizeFunction(f::MODELICA_FUNCTION, ctx::_CanonicalNameContext)
  local canonicalFunctionName = _canonicalVariableKey(f.name, ctx)
  local functionCtx = _functionCanonicalNameContext(f, ctx)
  return MODELICA_FUNCTION(canonicalFunctionName,
                           _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.inputs),
                           _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.outputs),
                           _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.locals),
                           _mapVectorLike(s -> _canonicalizeStatement(s, functionCtx), f.statements))
end

function _canonicalizeFunction(f::EXTERNAL_MODELICA_FUNCTION, ctx::_CanonicalNameContext)
  local canonicalFunctionName = _canonicalVariableKey(f.name, ctx)
  local functionCtx = _functionCanonicalNameContext(f, ctx)
  return EXTERNAL_MODELICA_FUNCTION(canonicalFunctionName,
                                    _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.inputs),
                                    _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.outputs),
                                    _mapVectorLike(v -> _canonicalizeDaeVar(v, functionCtx), f.locals),
                                    f.language, f.libInfo)
end

function _canonicalizeFunction(f::ModelicaFunction, ctx::_CanonicalNameContext)
  return f
end

function canonicalizeCrefNames(simCode::SIM_CODE;
                               nameMap::OMBackend.NameRewriteMap = OMBackend.NameRewriteMap())::SIM_CODE
  local rename = Dict{String, String}()
  for name in keys(simCode.stringToSimVarHT)
    rename[name] = _canonicalVariableKey(name)
  end
  for name in simCode.eliminatedVariables
    rename[name] = _canonicalVariableKey(name)
  end
  for entry in simCode.aliasMap
    rename[entry.eliminatedName] = _canonicalVariableKey(entry.eliminatedName)
    rename[entry.representativeName] = _canonicalVariableKey(entry.representativeName)
  end

  #= Reserved-name rename: a model variable literally named `t` collides with the
     MTK independent variable `t` (yields a dangling `der(t) ~ 1` and a SymReal
     clash in alias elimination). Rewrite it to `<modelName>V_t` everywhere via the
     override; the cref/HT canonicalization both consult `rename`. =#
  if haskey(simCode.stringToSimVarHT, "t")
    local _renamedT = _canonicalVariableKey(simCode.name) * "V_t"
    @warn "Variable name t clash with builtin symbol in MTK. Variable renamed $(_renamedT)"
    rename["t"] = _renamedT
  end

  local known = OrderedSet{String}(values(rename))
  union!(known, OrderedSet(["time", "pi", "e"]))
  local ctx = _CanonicalNameContext(rename, known, nameMap)

  @assign begin
    simCode.name = _canonicalVariableKey(simCode.name, ctx)
    simCode.stringToSimVarHT = _canonicalizeSimVarHT(simCode.stringToSimVarHT, ctx)
    simCode.residualEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.residualEquations)
    simCode.initialEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.initialEquations)
    simCode.whenEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.whenEquations)
    simCode.asserts = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.asserts)
    simCode.ifEquations = _mapVectorLike(ifEq -> _canonicalizeIfEquation(ifEq, ctx), simCode.ifEquations)
    simCode.structuralTransitions = _mapVectorLike(tr -> _canonicalizeStructuralTransition(tr, ctx),
                                                   simCode.structuralTransitions)
    simCode.subModels = _mapVectorLike(subModel -> canonicalizeCrefNames(subModel; nameMap = nameMap), simCode.subModels)
    simCode.sharedVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.sharedVariables)
    simCode.topVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.topVariables)
    simCode.sharedEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.sharedEquations)
    simCode.activeModel = _canonicalVariableKey(simCode.activeModel, ctx)
    simCode.irreducibleVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.irreducibleVariables)
    simCode.functions = _mapVectorLike(f -> _canonicalizeFunction(f, ctx), simCode.functions)
    simCode.eliminatedEquations = _mapVectorLike(eq -> _canonicalizeEquation(eq, ctx), simCode.eliminatedEquations)
    simCode.eliminatedVariables = _mapVectorLike(name -> _canonicalVariableKey(name, ctx), simCode.eliminatedVariables)
    simCode.aliasMap = _mapVectorLike(entry -> AliasEntry(_canonicalVariableKey(entry.eliminatedName, ctx),
                                                         _canonicalVariableKey(entry.representativeName, ctx),
                                                         entry.negated),
                                     simCode.aliasMap)
  end
  return simCode
end

"""
    simplifyEnumLiteralPaths(simCode::SIM_CODE)::SIM_CODE

Collapse the qualified namespace path of every `DAE.ENUM_LITERAL` to a
single `Absyn.IDENT` whose name is `Type.Literal` (the leaf two segments
joined by `.`). The integer index is preserved verbatim — that is what
arithmetic and comparison rely on. Frontend-shaped literals like

    ENUM_LITERAL(QUALIFIED("Modelica", QUALIFIED("Electrical", ...
                  QUALIFIED("Logic", IDENT("'U'")))), 1)

become

    ENUM_LITERAL(IDENT("Logic.'U'"), 1)

Reduces memory and makes downstream dumps directly readable without
custom @match arms for every nested QUALIFIED depth. Applied once at
SimCode entry — no later pass synthesises fresh ENUM_LITERAL paths,
they only substitute existing ones.
"""
function simplifyEnumLiteralPaths(simCode::SIM_CODE)::SIM_CODE
  local nRewritten = Ref(0)
  local _shortenPath = function(p)
    local segs = String[]
    local _walk = nothing
    _walk = function(x)
      if x isa Absyn.IDENT
        push!(segs, x.name)
      elseif x isa Absyn.QUALIFIED
        push!(segs, x.name)
        _walk(x.path)
      elseif x isa Absyn.FULLYQUALIFIED
        _walk(x.path)
      end
    end
    _walk(p)
    if length(segs) >= 2
      return Absyn.IDENT(segs[end-1] * "." * segs[end])
    elseif length(segs) == 1
      return Absyn.IDENT(segs[1])
    end
    return p
  end
  local _rewrite = function(exp, _)
    if exp isa DAE.ENUM_LITERAL && !(exp.name isa Absyn.IDENT && occursin('.', exp.name.name))
      nRewritten[] += 1
      return (DAE.ENUM_LITERAL(_shortenPath(exp.name), exp.index), true, nothing)
    end
    return (exp, true, nothing)
  end
  #= SIM-native rewriter (SIM ENUM_LITERAL's path field is `path`; DAE's is `name`). =#
  local _rewriteSIM = function(exp, _)
    if exp isa ENUM_LITERAL && !(exp.path isa Absyn.IDENT && occursin('.', exp.path.name))
      nRewritten[] += 1
      return (ENUM_LITERAL(_shortenPath(exp.path), exp.index), true, nothing)
    end
    return (exp, true, nothing)
  end

  #= Rewrite ENUM_LITERALs in a single Exp: SIM-native for SimCode Exps, DAE path
     for the BDAE.EQUATION entries that still carry a DAE.Exp. =#
  local _rewriteExp = function(e)
    if e isa Exp
      local (newExp, _) = traverseExpTopDown(e, _rewriteSIM, nothing)
      return newExp
    end
    local (newExp, _) = Util.traverseExpTopDown(e, _rewrite, nothing)
    return newExp
  end

  #= 1. Variable bindings (PARAMETER, DATA_STRUCTURE, ARRAY, ARRAY_PARAMETER). =#
  for (varName, (idx, sv)) in simCode.stringToSimVarHT
    local newKind = @match sv.varKind begin
      PARAMETER(SOME(b))      => PARAMETER(SOME(_rewriteExp(b)))
      DATA_STRUCTURE(SOME(b)) => DATA_STRUCTURE(SOME(_rewriteExp(b)))
      ARRAY(dims, SOME(b))    => ARRAY(dims, SOME(_rewriteExp(b)))
      ARRAY_PARAMETER(dims, SOME(b)) => ARRAY_PARAMETER(dims, SOME(_rewriteExp(b)))
      _ => sv.varKind
    end
    if newKind !== sv.varKind
      @assign sv.varKind = newKind
      simCode.stringToSimVarHT[varName] = (idx, sv)
    end
  end

  #= 2. Residual + initial equations. `initialEquations` may contain
        BDAE.EQUATION (lhs/rhs) entries alongside RESIDUAL_EQUATION; handle
        both forms. =#
  local _rewriteEq = function(eq)
    if eq isa BDAE.RESIDUAL_EQUATION || eq isa RESIDUAL_EQUATION
      #= SIM eq.exp -> _rewriteExp's SIM arm (already used on bindings above);
         BDAE eq.exp -> its DAE arm. Drops the per-residual whole-tree toDAEExp. =#
      return typeof(eq)(_rewriteExp(eq.exp), eq.source, eq.attr)
    elseif eq isa BDAE.EQUATION
      return BDAE.EQUATION(_rewriteExp(eq.lhs), _rewriteExp(eq.rhs), eq.source, eq.attributes)
    elseif eq isa EQUATION
      return EQUATION(_rewriteExp(eq.lhs), _rewriteExp(eq.rhs), eq.source, eq.attr)
    end
    return eq
  end
  @assign begin
    simCode.residualEquations = RESIDUAL_EQUATION[_rewriteEq(eq) for eq in simCode.residualEquations]
    simCode.initialEquations = Equation[_rewriteEq(eq) for eq in simCode.initialEquations]
  end

  if nRewritten[] > 0
    @debug "[SIMCODE: $(simCode.name): simplifyEnumLiteralPaths] collapsed $(nRewritten[]) ENUM_LITERAL qualified paths to Type.Literal IDENT form"
  end
  return simCode
end
