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

#= BDAECreate: residuals, when-equations, asserts and initial whens synthesized from
   algorithm sections, with the scalarization of their assignments, and the lifting of
   statement-level whens. =#

"""
    synthesizeFromInitialAlgorithms(iAlgorithms) -> Vector{BDAE.Equation}

Lift each `initial algorithm` body into a `BDAE.INITIAL_WHEN_EQUATION` with a
synthetic `initial()` condition. The downstream pipeline already routes any
INITIAL_WHEN_EQUATION whose condition is `initial()` through `INITIAL_ALGORITHM`
and the `__runInitialAlgorithm!()` codegen path, so this just funnels the
otherwise-orphaned `initial algorithm` blocks into that same path.
"""
function synthesizeFromInitialAlgorithms(iAlgorithms)::Vector{BDAE.Equation}
  local out = BDAE.Equation[]
  for alg in iAlgorithms
    local daeStmts = OMFrontend.Frontend.convertStatements(alg.statements)
    isempty(daeStmts) && continue
    local whenOps = _daeStmtsToWhenOps(daeStmts)
    local initialCall = DAE.CALL(Absyn.IDENT("initial"),
                                 MetaModelica.list(),
                                 DAE.callAttrBuiltinBool)
    local node = BDAE.INITIAL_WHEN_EQUATION(
      length(alg.statements),
      BDAE.WHEN_STMTS(initialCall, whenOps, NONE()),
      alg.source,
      BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
    )
    saveInitialAlgorithmStatements!(node, collect(daeStmts))
    push!(out, node)
  end
  return out
end

Base.@nospecializeinfer function _pushExpCrefStrings!(names::OrderedSet{String}, @nospecialize(exp))
  exp === nothing && return
  local crefs = Util.getAllCrefs(exp)
  for c in crefs
    push!(names, string(c))
  end
  return nothing
end

#= Collect every CREF name appearing anywhere (LHS or RHS) inside a list of
   BDAE equations. Used as the "already constrained" set for the
   algorithm-residual lifter, so we do not introduce a competing residual for
   a variable that a connect-style or normal equation already binds. =#
#= The names the equations constrain, for the algorithm residual lifter's guard. An alias `a = b`
   between two variables (a connect) defines neither and is left out: an algorithm's lhs is defined by
   the algorithm alone, and a connected lhs was never lifted (the MSL Digital Set source `y := x`,
   connected to a flip-flop's inputs, left them at 0, an invalid logic value). =#
function _collectAllCrefsInEquations(equations)::OrderedSet{String}
  local names = OrderedSet{String}()
  for eq in equations
    if eq isa BDAE.EQUATION
      (eq.lhs isa DAE.CREF && eq.rhs isa DAE.CREF) && continue
      _pushExpCrefStrings!(names, eq.lhs); _pushExpCrefStrings!(names, eq.rhs)
    elseif eq isa BDAE.RESIDUAL_EQUATION
      _pushExpCrefStrings!(names, eq.exp)
    elseif eq isa BDAE.ARRAY_EQUATION
      _pushExpCrefStrings!(names, eq.left); _pushExpCrefStrings!(names, eq.right)
    elseif eq isa BDAE.COMPLEX_EQUATION
      _pushExpCrefStrings!(names, eq.left); _pushExpCrefStrings!(names, eq.right)
    elseif eq isa BDAE.SOLVED_EQUATION
      push!(names, string(eq.componentRef))
      _pushExpCrefStrings!(names, eq.exp)
    end
  end
  return names
end

#= Walk a list of `DAE.Statement` (already converted from the frontend) and
   collect a `BDAE.RESIDUAL_EQUATION` for every scalar assignment whose LHS
   is not already constrained by an equation. ALG_WHEN bodies (already
   handled by `synthesizeInitialWhenFromAlgorithms`) and unsupported
   compound forms (FOR / IF / WHILE) are skipped — they can be lowered later
   if needed. =#
#= True if a DAE.Type is a Modelica discrete-time type (Boolean / Integer /
   enumeration, or arrays thereof). Algorithm sections whose LHS is a
   continuous Real variable have order-sensitive semantics that the simple
   residual lifter cannot represent; restricting the lift to discrete LHSes
   keeps the fix narrow to the cluster-A class of bugs (INV3S / Digital
   gates / Thyristor `fire` etc.) without risking continuous-state models. =#
Base.@nospecializeinfer function _isDiscreteDAEType(@nospecialize(ty))::Bool
  ty isa DAE.T_INTEGER || ty isa DAE.T_BOOL || ty isa DAE.T_ENUMERATION ||
    (ty isa DAE.T_ARRAY && _isDiscreteDAEType(ty.ty))
end

#= True only for continuous Real types — used as a NEGATIVE filter in the
   `change(...)` trigger synthesis. Unknown / complex types fall through to
   "not continuous" so we err on the side of generating a `change()` call
   (over-triggering on a connector read is harmless if the underlying
   value is discrete, which is the cluster-A case). =#
Base.@nospecializeinfer function _isContinuousRealType(@nospecialize(ty))::Bool
  ty isa DAE.T_REAL || (ty isa DAE.T_ARRAY && _isContinuousRealType(ty.ty))
end

#= Collect the cref-string names of every `flatModel.variables` entry whose
   variability classification puts it in the parameter / constant family
   (CONSTANT, STRUCTURAL_PARAMETER, PARAMETER, NON_STRUCTURAL_PARAMETER).
   These names are forwarded to the WHEN lifter so that `change(<param>)`
   triggers are dropped from the synthesized condition; without this the
   lifted condition stays as `initial() OR change(<param>)` and the
   downstream pipeline cannot recognise the equivalent `INITIAL_WHEN`
   shape, which makes the lifted equation a runtime DiscreteCallback that
   races with sibling data-flow callbacks at t=0. =#
#= Walk the already-converted `Vector{BDAE.VAR}` and collect cref-string
   names of every entry whose `varKind` is `PARAM` or `CONST`. Iterating
   the materialized BDAE vector avoids touching the lazier frontend list
   that triggered a multi-minute stall on first call. =#
function _collectParamOrConstNames(variables::Vector{BDAE.VAR},
                                   varNames::Vector{String} = String[string(v.varName) for v in variables])::OrderedSet{String}
  local names = OrderedSet{String}()
  sizehint!(names, 2 * length(variables))
  for (i, var) in enumerate(variables)
    local k = var.varKind
    if k isa BDAE.PARAM || k isa BDAE.CONST
      local s = varNames[i]
      push!(names, s)
      local u = replace(s, "." => "_")
      u === s || push!(names, u)
    end
  end
  return names
end

#= Walk to the innermost CREF_IDENT and append `sub` to its subscript list.
   Used when scalarising an array LHS assignment — turns `comp.yy` into
   `comp.yy[k]` for each k while preserving the qualifier chain. =#
Base.@nospecializeinfer function _appendSubscriptToInnermost(@nospecialize(cref), @nospecialize(sub))
  @match cref begin
    DAE.CREF_IDENT(ident, ty, subs) => begin
      DAE.CREF_IDENT(ident, ty, listAppend(subs, MetaModelica.list(sub)))
    end
    DAE.CREF_QUAL(ident, ty, subs, inner) => begin
      DAE.CREF_QUAL(ident, ty, subs, _appendSubscriptToInnermost(inner, sub))
    end
    _ => cref
  end
end

Base.@nospecializeinfer function _collectAssignResidualsFromDAEStmts!(out::Vector{BDAE.Equation},
                                              @nospecialize(daeStmts),
                                              @nospecialize(source),
                                              eqLhsBoundCrefs::OrderedSet{String},
                                              whenLifterSkipLhs::OrderedSet{String} = OrderedSet{String}())
  for s in daeStmts
    @match s begin
      DAE.STMT_ASSIGN(ty, lhs, rhs, src) => begin
        _isDiscreteDAEType(ty) || continue
        if lhs isa DAE.CREF
          local crStr = string(lhs.componentRef)
          (crStr in eqLhsBoundCrefs) && continue
          (crStr in whenLifterSkipLhs) && continue
        end
        push!(out, BDAE.RESIDUAL_EQUATION(
          DAE.BINARY(lhs, DAE.SUB(DAE.T_REAL_DEFAULT), rhs),
          src,
          BDAE.EQ_ATTR_DEFAULT_DYNAMIC,
        ))
      end
      DAE.STMT_ASSIGN_ARR(ty, lhs, rhs, src) => begin
        _isDiscreteDAEType(ty) || continue
        if lhs isa DAE.CREF
          local crStr = string(lhs.componentRef)
          (crStr in whenLifterSkipLhs) && continue
        end
        #= Scalarise the array assignment to per-element residuals so
           the codegen emits constraints on the scalarised simvars
           (`<comp>.yy[1]`, `<comp>.yy[2]`, …) rather than a bare-array
           residual on the undeclared base symbol. The per-element
           residual `<lhs>[k] - <rhs>[k] = 0` is built by attaching a
           DAE.INDEX subscript to the innermost CREF_IDENT of the LHS
           and using DAE.ASUB to index the RHS expression. =#
        local _arrLen::Int = 0
        @match ty begin
          DAE.T_ARRAY(_, _dims) => begin
            local _dVec = BDAEUtil.DAE_DimensionToIntVector(_dims)
            _arrLen = isempty(_dVec) ? 0 : _dVec[1]
          end
          _ => begin _arrLen = 0 end
        end
        if !(lhs isa DAE.CREF) || _arrLen <= 0
          #= Unknown shape — collision-guard against element-bound LHSes,
             else fall back to the bare-array residual. =#
          if lhs isa DAE.CREF
            local crStr2 = string(lhs.componentRef)
            (crStr2 in eqLhsBoundCrefs) && continue
            local _be = false
            for _bn in eqLhsBoundCrefs
              if startswith(_bn, crStr2 * "[")
                _be = true; break
              end
            end
            _be && continue
          end
          push!(out, BDAE.RESIDUAL_EQUATION(
            DAE.BINARY(lhs, DAE.SUB(DAE.T_REAL_DEFAULT), rhs),
            src,
            BDAE.EQ_ATTR_DEFAULT_DYNAMIC,
          ))
        else
          local _baseCref = lhs.componentRef
          local _elemTy = @match ty begin
            DAE.T_ARRAY(et, _) => et
            _ => ty
          end
          for _k in 1:_arrLen
            local _idxSub = DAE.INDEX(DAE.ICONST(_k))
            local _newCref = _appendSubscriptToInnermost(_baseCref, _idxSub)
            local _lhsK::DAE.Exp = DAE.CREF(_newCref, _elemTy)
            local _rhsK::DAE.Exp = DAE.ASUB(rhs, MetaModelica.list(DAE.INDEX(DAE.ICONST(_k))))
            #= Per-element collision: skip if this scalar element is
               already bound by another equation. =#
            local _lhsKStr = string(_newCref)
            if _lhsKStr in eqLhsBoundCrefs || _lhsKStr in whenLifterSkipLhs
              continue
            end
            push!(out, BDAE.RESIDUAL_EQUATION(
              DAE.BINARY(_lhsK, DAE.SUB(DAE.T_REAL_DEFAULT), _rhsK),
              src,
              BDAE.EQ_ATTR_DEFAULT_DYNAMIC,
            ))
          end
        end
      end
      _ => nothing
    end
  end
end

"""
    synthesizeResidualsFromRegularAlgorithms(algorithms, eqLhsBoundCrefs) -> Vector{BDAE.Equation}

Lift each non-when, non-initial `algorithm` section into a list of
`BDAE.RESIDUAL_EQUATION`s, one per `STMT_ASSIGN`/`STMT_ASSIGN_ARR`. Statements
wrapped inside `ALG_WHEN` (including the `algorithm when initial()` shape)
are skipped because `synthesizeInitialWhenFromAlgorithms` already handles
those bodies via the WhenEquation path. An assignment is also skipped when
the LHS cref already appears in another equation (e.g. driven by a
`connect(...)`), to avoid introducing a competing residual that would
over-determine the system.
"""
Base.@nospecializeinfer function synthesizeResidualsFromRegularAlgorithms(@nospecialize(algorithms),
                                                  eqLhsBoundCrefs::OrderedSet{String} = OrderedSet{String}(),
                                                  whenLifterSkipLhs::OrderedSet{String} = OrderedSet{String}();
                                                  realStarts::Dict{String, DAE.Exp} = Dict{String, DAE.Exp}())::Vector{BDAE.Equation}
  local out = BDAE.Equation[]
  for alg in algorithms
    #= Skip whole algorithm if every top-level statement is ALG_WHEN — those are
       already lifted to (INITIAL_)WHEN_EQUATION by the companion synth pass. =#
    local hasNonWhen = false
    for stmt in alg.statements
      if !isvariant(stmt, OMFrontend.Frontend.ALG_WHEN)
        hasNonWhen = true
        break
      end
    end
    hasNonWhen || continue
    local daeStmts = try
      OMFrontend.Frontend.convertStatements(alg.statements)
    catch err
      #= An algorithm the frontend cannot convert is left out. =#
      OMBackend._fallback(err, :convertAlgorithmResiduals; impact = :result)
      continue
    end
    append!(out, _realAssignmentEquations(daeStmts, alg.source, realStarts, whenLifterSkipLhs))
    #= Conservative narrowing: only lift single-statement algorithm bodies.
       Multi-statement algorithms (Modelica.Mechanics.Rotational.Examples.OneWayClutch
       and most Modelica.Electrical.Digital gates) have order-sensitive
       semantics that a flat residual list does not preserve, and lifting
       them as independent residuals over-determines or imbalances MTK's
       reduced system. Bodies with one assignment (the reproducer
       `Models/AlgorithmDiscreteAssign.mo` shape) are safe. =#
    local stmtCount = 0
    for s in daeStmts
      (s isa DAE.STMT_ASSIGN || s isa DAE.STMT_ASSIGN_ARR) || continue
      stmtCount += 1
    end
    stmtCount == 1 || continue
    _collectAssignResidualsFromDAEStmts!(out, daeStmts, alg.source, eqLhsBoundCrefs, whenLifterSkipLhs)
  end
  return out
end

#= The Real variables an algorithm section assigns outside its when
   statements, as equations (MLS 11.1.2: the section runs as a whole, in
   order). The statements run symbolically (_AlgorithmRun): each assignment's
   value, its reads of variables assigned before replaced by their values at
   that point, becomes the variable's value; a variable the section assigns
   only later reads its start value (a discrete one pre()). Each Real variable
   gets `v = <its value at the end>`, a record variable the record equation;
   an assert is checked on the values at its point. The Integer, Boolean and
   enumeration targets stay the when lifter's. Every path above took discrete
   targets only: the Real ones were left without an equation. =#
function _realAssignmentEquations(@nospecialize(daeStmts), @nospecialize(source),
                                  realStarts::Dict{String, DAE.Exp},
                                  whenLifted::OrderedSet{String})::Vector{BDAE.Equation}
  local stmts = DAE.Statement[s for s in daeStmts if !(s isa DAE.STMT_WHEN)]
  _lowersRealTargets(stmts) || (_reportRealAssignments(stmts); return BDAE.Equation[])
  local (ops, assigned) = _buildAlgorithmBodyOps(stmts, false, true; where = "an algorithm section with Real targets")
  local run = _AlgorithmRun(assigned, realStarts)
  local targets = OrderedDict{String, DAE.Exp}()
  local out = BDAE.Equation[]
  for op in ops
    if op isa BDAE.ASSIGN
      local name = string(op.left.componentRef)
      name in run.readAsVariable &&
        OMBackend.unsupported("an algorithm section assigning $(name) after reading it in der(), edge() or change()", op.left)
      run.values[name] = _algorithmValue(op.right, run)
      targets[name] = op.left
    elseif op isa BDAE.ASSERT
      push!(out, BDAE.ASSERT_EQUATION(_algorithmValue(op.condition, run), _algorithmValue(op.message, run),
                                      op.level, op.source))
    else
      OMBackend.unsupported("this statement in an algorithm section with Real targets", op)
    end
  end
  for (name, lhs) in targets
    if _isContinuousRealType(lhs.ty)
      name in whenLifted &&
        OMBackend.unsupported("a Real variable an algorithm section assigns both in a when and outside one", lhs)
      push!(out, BDAE.RESIDUAL_EQUATION(DAE.BINARY(lhs, DAE.SUB(DAE.T_REAL_DEFAULT), run.values[name]),
                                        source, BDAE.EQ_ATTR_DEFAULT_DYNAMIC))
    elseif _isRecordType(lhs.ty)
      local eqs = equationToBackendEquation(DAE.COMPLEX_EQUATION(lhs, run.values[name], source))
      eqs isa Vector ? append!(out, eqs) : push!(out, eqs)
    end
  end
  return out
end

#= Whether _realAssignmentEquations lowers a section's statements (its whens
   left out): it has Real or record targets, and no initial() (the ops replace
   it by a constant: such a section is reported, not lowered). =#
_lowersRealTargets(@nospecialize(stmts))::Bool =
  !isempty(_collectRealTargets!(String[], stmts)) && !any(_statementMentionsInitial, stmts)

#= An algorithm section run symbolically: the values of the variables assigned
   so far; every scalar name it assigns and the names of their wholes (`x` of
   `x[2]`, `r` of `r.a`); the variables read in der(), edge() or change(),
   whose argument is the variable itself. =#
struct _AlgorithmRun
  values::Dict{String, DAE.Exp}
  assigned::OrderedSet{String}
  wholes::Set{String}
  realStarts::Dict{String, DAE.Exp}
  readAsVariable::Set{String}
end

function _AlgorithmRun(assigned::OrderedSet{String}, realStarts::Dict{String, DAE.Exp})
  local wholes = Set{String}()
  for name in assigned, i in eachindex(name)
    name[i] in ('[', '.') && push!(wholes, name[1:prevind(name, i)])
  end
  return _AlgorithmRun(Dict{String, DAE.Exp}(), assigned, wholes, realStarts, Set{String}())
end

#= Above this many nodes a value is not built: a conditional assignment that
   reads its own variable (`if v[i] > m then m := v[i]`) doubles it each time. =#
const _ALGORITHM_VALUE_NODE_LIMIT = 10_000

#= `e` with the section's variables read as their values so far: assigned
   before, the value; assigned only later, the start value (Real) or pre()
   (discrete). Inside pre() a variable is the value before the event; inside
   der(), edge() and change() the variable itself, which its equation defines
   (its value at the end: refused if it is assigned again later). =#
function _algorithmValue(@nospecialize(e), run::_AlgorithmRun)
  local visit = (x, arg) -> begin
    if x isa DAE.CALL && x.path isa Absyn.IDENT && x.path.name in ("pre", "der", "edge", "change")
      x.path.name == "pre" && return (x, false, arg)
      for c in Util.getAllCrefs(x)
        local n = string(c)
        n in run.assigned || continue
        haskey(run.values, n) ||
          OMBackend.unsupported("$(x.path.name)() of a variable before the algorithm section assigns it", x)
        push!(run.readAsVariable, n)
      end
      return (x, false, arg)
    end
    x isa DAE.CREF || return (x, true, arg)
    local cr = _foldConstantSubscripts(x.componentRef)
    local name = string(cr)
    haskey(run.values, name) && return (run.values[name], false, arg)
    if name in run.assigned
      _isContinuousRealType(x.ty) && return (get(run.realStarts, name, DAE.RCONST(0.0)), false, arg)
      local attr = DAE.CALL_ATTR(x.ty, false, true, false, false, DAE.NO_INLINE(), DAE.NO_TAIL())
      return (DAE.CALL(Absyn.IDENT("pre"), MetaModelica.list(DAE.CREF(cr, x.ty)), attr), false, arg)
    end
    name in run.wholes &&
      OMBackend.unsupported("a read of a whole array or record that an algorithm section assigns part by part", x)
    #= x[j]: one of the elements the section assigns, chosen by the index. =#
    local base = first(split(name, '['))
    if base != name && base in run.wholes && any(s -> !(s isa DAE.INDEX && s.exp isa DAE.ICONST), _innermostSubscripts(cr))
      local elements = _scalarizeCrefRead(cr, x.ty, Int[], x)
      elements isa DAE.IFEXP || OMBackend.unsupported("this read of an array an algorithm section assigns", x)
      return (_elementValues(elements, x, run), false, arg)
    end
    return (x, true, arg)
  end
  local value = first(Util.traverseExpTopDown(e, visit, nothing))
  _expLargerThan(value, _ALGORITHM_VALUE_NODE_LIMIT) &&
    OMBackend.unsupported("an algorithm section whose values exceed $(_ALGORITHM_VALUE_NODE_LIMIT) nodes as equations", e)
  return value
end

#= The if-chain over the elements a read `x[j]` may be, each element (and
   guard) read as its value; the chain's last else, `x[j]` itself, as it is. =#
function _elementValues(@nospecialize(chain), @nospecialize(read), run::_AlgorithmRun)
  chain === read && return read
  return DAE.IFEXP(_algorithmValue(chain.expCond, run), _algorithmValue(chain.expThen, run),
                   _elementValues(chain.expElse, read, run))
end

#= The innermost subscripts of `cr` folded where they are integer constants
   (`x[2 - 1]` after a loop's iterator was replaced is `x[1]`). =#
function _foldConstantSubscripts(@nospecialize(cr))
  local subs = _innermostSubscripts(cr)
  any(s -> s isa DAE.INDEX && !(s.exp isa DAE.ICONST) && _foldInteger(s.exp) !== nothing, subs) || return cr
  local folded = DAE.Subscript[(s isa DAE.INDEX && _foldInteger(s.exp) !== nothing) ? DAE.INDEX(DAE.ICONST(_foldInteger(s.exp))) : s
                               for s in subs]
  return _replaceInnermostSubscripts(cr, folded)
end

Base.@nospecializeinfer function _foldInteger(@nospecialize(e))::Union{Int, Nothing}
  @match e begin
    DAE.ICONST(i) => i
    DAE.UNARY(DAE.UMINUS(__), a) => (local v = _foldInteger(a); v === nothing ? nothing : -v)
    DAE.BINARY(a, op, b) => begin
      local (va, vb) = (_foldInteger(a), _foldInteger(b))
      (va === nothing || vb === nothing) && return nothing
      op isa DAE.ADD ? va + vb : op isa DAE.SUB ? va - vb : op isa DAE.MUL ? va * vb : nothing
    end
    _ => nothing
  end
end

#= Whether a DAE expression has more than `cap` nodes, counted as a tree. =#
function _expLargerThan(@nospecialize(e), cap::Int)::Bool
  local n = 0
  Util.traverseExpTopDown(e, (x, arg) -> (n += 1; (x, n <= cap, arg)), nothing)
  return n > cap
end

_isRecordType(@nospecialize(ty))::Bool = ty isa DAE.T_COMPLEX && ty.complexClassType isa DAE.ClassInf.RECORD
_isRealOrRecordType(@nospecialize(ty))::Bool = _isContinuousRealType(ty) || _isRecordType(ty)

function _statementMentionsInitial(@nospecialize(stmt))::Bool
  local found = false
  Util.mapDAEStatementExps(e -> (found |= _expMentionsInitial(e); e), stmt)
  return found
end

#= An algorithm section whose Real targets are not lowered: reported as a
   fallback that changes the result. =#
function _reportRealAssignments(@nospecialize(daeStmts))
  local targets = String[]
  _collectRealTargets!(targets, daeStmts)
  isempty(targets) && return nothing
  try
    OMBackend.unsupported("an algorithm section reading initial() with Real targets outside a when (not lowered)", join(targets, ", "))
  catch err
    OMBackend._fallback(err, :algorithmRealAssign; only = OMBackend.UnsupportedLowering, impact = :result)
  end
  return nothing
end

Base.@nospecializeinfer function _collectRealTargets!(targets::Vector{String}, @nospecialize(daeStmts))
  for s in daeStmts
    @match s begin
      DAE.STMT_ASSIGN(ty, lhs, _, _) => _isRealOrRecordType(ty) && push!(targets, string(lhs))
      DAE.STMT_ASSIGN_ARR(ty, lhs, _, _) => _isRealOrRecordType(ty) && push!(targets, string(lhs))
      DAE.STMT_TUPLE_ASSIGN(_, lhs, _, _) => for t in lhs
        (t isa DAE.CREF && _isRealOrRecordType(t.ty)) && push!(targets, string(t))
      end
      DAE.STMT_IF(_, body, else_, _) => begin
        _collectRealTargets!(targets, body)
        while !(else_ isa DAE.NOELSE)
          _collectRealTargets!(targets, else_.statementLst)
          else_ isa DAE.ELSEIF || break
          else_ = else_.else_
        end
      end
      DAE.STMT_FOR(statementLst = body) => _collectRealTargets!(targets, body)
      DAE.STMT_WHILE(statementLst = body) => _collectRealTargets!(targets, body)
      #= A when statement's body is lowered with its when. =#
      _ => nothing
    end
  end
  return targets
end

#= Best-effort: extract the type carried by a `DAE.ComponentRef`. Each
   CREF_IDENT / CREF_QUAL stores its identType; CREF_ITER and WILD are not
   useful triggers. =#
Base.@nospecializeinfer function _crefType(@nospecialize(cref))
  @match cref begin
    DAE.CREF_IDENT(_, ty, subs) => _typeAfterSubscripts(ty, subs)
    DAE.CREF_QUAL(_, _, _, cr) => _crefType(cr)
    _ => nothing
  end
end

Base.@nospecializeinfer function _typeAfterSubscripts(@nospecialize(ty), @nospecialize(subs))
  local out = ty
  for sub in subs
    if sub isa DAE.INDEX
      out = _dropLeadingArrayDim(out)
    end
  end
  return out
end

Base.@nospecializeinfer function _dropLeadingArrayDim(@nospecialize(ty))
  if ty isa DAE.T_ARRAY
    local dims = collect(ty.dims)
    if length(dims) <= 1
      return ty.ty
    end
    return DAE.T_ARRAY(ty.ty, MetaModelica.list(dims[2:end]...))
  end
  return ty
end

Base.@nospecializeinfer function _isSingleStraightDiscreteAssign(@nospecialize(daeStmts))::Bool
  local n = 0
  for s in daeStmts
    @match s begin
      DAE.STMT_ASSIGN(ty, _, _, _) => begin
        _isDiscreteDAEType(ty) || return false
        n += 1
      end
      DAE.STMT_ASSIGN_ARR(ty, _, _, _) => begin
        _isDiscreteDAEType(ty) || return false
        n += 1
      end
      _ => return false
    end
  end
  return n == 1
end

"""
    synthesizeWhenEquationsFromRegularAlgorithms(algorithms) -> Vector{BDAE.Equation}

For each regular (non-when, non-initial) `algorithm` section whose top-level
assignments all target discrete-time LHSes (Integer / Boolean / enumeration),
synthesize a `BDAE.WHEN_EQUATION` with condition
`initial() or change(rhs1) or change(rhs2) ...` whose body is the algorithm
statements lowered via `_daeStmtsToWhenOps`. The RHS CREF list is
deduplicated and filtered to discrete-typed crefs (continuous Real RHS
references would over-trigger). The `time` cref is also filtered out — its
"change" is the integrator stepping forward, not a discrete event.

This is the Modelica-spec-correct lowering of Logic-enum algorithms like
INV3S's `nextstate := Buf3sTable[...]; yy := nextstate;` and resolves the
INV3S/MUX2x1/NRXFER/NXFER/BUF3S cluster-A validate failures.

An algorithm of `when` statements only is lifted too, each statement to its
WHEN_EQUATION(s) (a `when sample(...)`; after a first branch `when initial()`,
which `synthesizeInitialWhenFromAlgorithms` lifts, its `elsewhen` arms).
"""
function synthesizeWhenEquationsFromRegularAlgorithms(algorithms,
                                                      paramOrConstNames::OrderedSet{String} = OrderedSet{String}())
  local out = BDAE.Equation[]
  local liftedLhsNames = OrderedSet{String}()
  for alg in algorithms
    local statements = alg.statements
    isempty(statements) && continue
    local hasNonWhen = false
    for stmt in statements
      if !isvariant(stmt, OMFrontend.Frontend.ALG_WHEN)
        hasNonWhen = true
        break
      end
    end
    local daeStmts = try
      OMFrontend.Frontend.convertStatements(statements)
    catch err
      #= An algorithm the frontend cannot convert is left out. =#
      OMBackend._fallback(err, :convertAlgorithmWhens; impact = :result)
      continue
    end
    #= An algorithm of whens only: the companion pass
       (synthesizeInitialWhenFromAlgorithms) lifts a first branch that is
       `when initial()`; the rest is lifted here, or it was lost: a when on
       another condition (a sample, a time relation) and the elsewhen arms
       after `when initial()` (MSL GenerateRandomNumbers' samples). =#
    if !hasNonWhen
      for s in daeStmts
        s isa DAE.STMT_WHEN && _liftAlgorithmWhenStatement!(out, s, liftedLhsNames)
      end
      continue
    end
    #= Sources.Table / Step / Pulse / Clock have an unrolled body of the shape
         y := y0;                              (single ALG_ASSIGNMENT)
         if time >= t[1] then y := x[1]; end if;  (ALG_IF { ALG_ASSIGNMENT })
         if time >= t[2] then y := x[2]; end if;
         ...
       Each ALG_IF is semantically a `when cond then body end when` — a
       discrete callback that updates `y` when its condition crosses to
       true. Lift each top-level ALG_IF{single ALG_ASSIGNMENT} into its own
       BDAE.WHEN_EQUATION so MSL Digital / Analog sources emit step-hold
       outputs at runtime. =#
    local ifLifted = false
    for s in daeStmts
      _liftAlgIfToWhen!(out, s, alg.source) && (ifLifted = true)
    end
    #= Build ONE unified INITIAL_WHEN_EQUATION whose body is the entire
       algorithm sequence rewritten so that each STMT_IF becomes
       `lhs := IFEXP(cond, then-expr, lhs)`. At init time the algorithm
       runs in source order: bare assignments fire unconditionally, and
       conditional assignments fire iff their guard already holds at t=0.
       This is what Modelica spec requires for `algorithm y := y0; if
       time >= t[1] then y := x[1]; end if;` when t[1] is ≤ startTime
       — the time-trigger boundary case that a runtime ContinuousCallback
       cannot catch via root-finding alone. The per-STMT_IF WHEN_EQUATIONs
       emitted above continue to handle real time-event crossings later
       in the simulation. =#
    if !_isSingleStraightDiscreteAssign(daeStmts)
      _liftAlgorithmBodyToInitialWhen!(out, daeStmts, alg.source, liftedLhsNames, paramOrConstNames)
    end
    #= A mixed algorithm body may also contain an explicit `when/elsewhen`
       statement (e.g. MSL InertialDelaySensitive's scheduling block). The
       body lifter above skips STMT_WHEN; lift each into a real WHEN_EQUATION
       so its LHS (t_next, y_auxiliary, ...) are actually assigned. =#
    for s in daeStmts
      s isa DAE.STMT_WHEN && _liftAlgorithmWhenStatement!(out, s, liftedLhsNames)
    end
    #= Per-statement lifting via `_liftAlgAssignToInitialWhen!` and
       `_liftAlgIfToWhen!` covers every shape the cluster-A Digital examples
       need. The legacy single-block lifter (which combined all
       STMT_ASSIGNs into one when whose condition was the union of all RHS
       changes) is intentionally removed because it produced a duplicate of
       what the per-statement passes already emit. =#
  end
  return (out, liftedLhsNames)
end

#= A top-level when statement of an algorithm. A first branch `when initial()`
   is synthesizeInitialWhenFromAlgorithms' (it runs the DAE statements): here
   only its elsewhen arms (lifted twice before, its init arm a second node of
   the same source). Any other when entirely. =#
function _liftAlgorithmWhenStatement!(out::Vector{BDAE.Equation}, @nospecialize(s), liftedLhsNames::OrderedSet{String})
  if _isInitialCall(s.exp)
    @match s.elseWhen begin
      SOME(esw) => begin
        local weq = _stmtWhenToBdaeWhenEquation(esw, liftedLhsNames)
        weq !== nothing && push!(out, weq)
      end
      _ => nothing
    end
  else
    _liftStmtWhenToWhenEquations!(out, s, liftedLhsNames)
  end
  return nothing
end

#= The top-level asserts of regular algorithm sections, as assert equations:
   checked at run time like those of equation sections. The lifters above
   turn the assignments into equations and leave the asserts out. =#
function synthesizeAssertsFromRegularAlgorithms(algorithms)::Vector{BDAE.Equation}
  local out = BDAE.Equation[]
  for alg in algorithms
    isempty(alg.statements) && continue
    local daeStmts = try
      OMFrontend.Frontend.convertStatements(alg.statements)
    catch err
      #= An algorithm the frontend cannot convert is left out. =#
      OMBackend._fallback(err, :convertAlgorithmAsserts; impact = :result)
      continue
    end
    #= A section with Real targets checks its asserts where they stand
       (_realAssignmentEquations). =#
    _lowersRealTargets(DAE.Statement[s for s in daeStmts if !(s isa DAE.STMT_WHEN)]) && continue
    for s in daeStmts
      s isa DAE.STMT_ASSERT && push!(out, BDAE.ASSERT_EQUATION(s.cond, s.msg, s.level, s.source))
    end
  end
  return out
end

Base.@nospecializeinfer function _isTimeCref(@nospecialize(cref))::Bool
  @match cref begin
    DAE.CREF_IDENT("time", _, _) => true
    _ => false
  end
end

Base.@nospecializeinfer function _arrayDimsFromType(@nospecialize(ty))::Vector{Int}
  if ty isa DAE.T_ARRAY
    return BDAEUtil.DAE_DimensionToIntVector(ty.dims)
  end
  return Int[]
end

Base.@nospecializeinfer function _rawArrayDims(@nospecialize(ty))::Vector{Any}
  if ty isa DAE.T_ARRAY
    return Any[d for d in ty.dims]
  end
  return Any[]
end

Base.@nospecializeinfer function _suffixEnumPath(@nospecialize(p), name::String)
  @match p begin
    Absyn.IDENT(n) => Absyn.QUALIFIED(n, Absyn.IDENT(name))
    Absyn.QUALIFIED(n, rest) => Absyn.QUALIFIED(n, _suffixEnumPath(rest, name))
    Absyn.FULLYQUALIFIED(rest) => Absyn.FULLYQUALIFIED(_suffixEnumPath(rest, name))
  end
end

#= Subscript expression for the k-th element along a dimension. Enumeration
   dimensions must use the enum literal, not the plain integer: the scalarized
   declaration names carry enum-literal subscripts, and a numerically spelled
   element name strands the lookup at codegen time. =#
Base.@nospecializeinfer function _dimIndexExp(@nospecialize(dim), k::Int)
  @match dim begin
    DAE.DIM_ENUM(__) => begin
      local lits = collect(dim.literals)
      (1 <= k <= length(lits)) ?
        DAE.ENUM_LITERAL(_suffixEnumPath(dim.enumTypeName, lits[k]), k) :
        DAE.ICONST(k)
    end
    _ => DAE.ICONST(k)
  end
end

Base.@nospecializeinfer function _arrayElementType(@nospecialize(ty))
  local out = ty
  while out isa DAE.T_ARRAY
    out = out.ty
  end
  return out
end

Base.@nospecializeinfer function _innermostType(@nospecialize(cref))
  @match cref begin
    DAE.CREF_IDENT(_, ty, _) => ty
    DAE.CREF_QUAL(_, _, _, cr) => _innermostType(cr)
    _ => DAE.T_UNKNOWN_DEFAULT
  end
end

Base.@nospecializeinfer function _innermostSubscripts(@nospecialize(cref))::Vector{DAE.Subscript}
  @match cref begin
    DAE.CREF_IDENT(_, _, subs) => collect(subs)
    DAE.CREF_QUAL(_, _, _, cr) => _innermostSubscripts(cr)
    _ => DAE.Subscript[]
  end
end

Base.@nospecializeinfer function _replaceInnermostSubscripts(@nospecialize(cref),
                                                             subs::Vector)
  @match cref begin
    DAE.CREF_IDENT(ident, ty, _) => DAE.CREF_IDENT(ident, ty, MetaModelica.list(subs...))
    DAE.CREF_QUAL(ident, ty, qsubs, cr) =>
      DAE.CREF_QUAL(ident, ty, qsubs, _replaceInnermostSubscripts(cr, subs))
    _ => cref
  end
end

#= Integer index values as DAE.INDEX subscripts (DAE.ASUB.sub is List{Subscript}). =#
Base.@nospecializeinfer function _iconstIndexSubList(vals::Vector{Int})
  local subs = DAE.Subscript[DAE.INDEX(DAE.ICONST(v)) for v in vals]
  return MetaModelica.list(subs...)
end

Base.@nospecializeinfer function _andCondition(@nospecialize(a), @nospecialize(b))
  a === nothing && return b
  b === nothing && return a
  if a isa DAE.BCONST
    return a.bool ? b : a
  elseif b isa DAE.BCONST
    return b.bool ? a : b
  end
  return DAE.LBINARY(a, DAE.AND(DAE.T_BOOL_DEFAULT), b)
end

Base.@nospecializeinfer function _notCondition(@nospecialize(cond))
  if cond isa DAE.BCONST
    return DAE.BCONST(!cond.bool)
  end
  return DAE.LUNARY(DAE.NOT(DAE.T_BOOL_DEFAULT), cond)
end

Base.@nospecializeinfer function _indexEqualsCondition(@nospecialize(exp), value::Int)
  return DAE.RELATION(exp,
                      DAE.EQUAL(DAE.T_INTEGER_DEFAULT),
                      DAE.ICONST(value),
                      -1,
                      NONE())
end

Base.@nospecializeinfer function _replaceInitialCall(@nospecialize(exp), initialValue::Bool)
  function repl(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("initial"), _, _) => (DAE.BCONST(arg), arg)
      _ => (e, arg)
    end
  end
  return first(Util.traverseExpBottomUp(exp, repl, initialValue))
end

Base.@nospecializeinfer function _substituteLoopIters(@nospecialize(exp),
                                                      iterVals::Dict{String, Int})
  isempty(iterVals) && return exp
  function repl(@nospecialize(e), arg)
    @match e begin
      DAE.CREF(DAE.CREF_IDENT(id, _, _), _) where haskey(arg, id) =>
        (DAE.ICONST(arg[id]), arg)
      _ => (e, arg)
    end
  end
  return first(Util.traverseExpBottomUp(exp, repl, iterVals))
end

Base.@nospecializeinfer function _prepareAlgorithmExp(@nospecialize(exp),
                                                      iterVals::Dict{String, Int},
                                                      initialValue::Bool)
  return _replaceInitialCall(_substituteLoopIters(exp, iterVals), initialValue)
end

Base.@nospecializeinfer function _rangeIntValues(@nospecialize(range))
  @match range begin
    DAE.RANGE(_, DAE.ICONST(firstVal), stepOpt, DAE.ICONST(lastVal)) => begin
      local stepVal = 1
      @match stepOpt begin
        SOME(DAE.ICONST(s)) => (stepVal = s)
        NONE() => nothing
        _ => return nothing
      end
      stepVal == 0 && return nothing
      return collect(firstVal:stepVal:lastVal)
    end
    _ => return nothing
  end
end

Base.@nospecializeinfer function _scalarLhsTargets(@nospecialize(lhs::DAE.CREF),
                                                   @nospecialize(assignTy))
  local cr = lhs.componentRef
  local baseTy = _innermostType(cr)
  local dims = _arrayDimsFromType(baseTy)
  local _emptyTargets = Tuple{DAE.Exp, Union{Nothing, DAE.Exp}, Vector{Int}}[]
  isempty(dims) && return [(lhs, nothing, Int[])]
  local rawDims = _rawArrayDims(baseTy)

  local subs = _innermostSubscripts(cr)
  if isempty(subs)
    subs = DAE.Subscript[DAE.WHOLEDIM() for _ in dims]
  elseif length(subs) < length(dims)
    append!(subs, DAE.Subscript[DAE.WHOLEDIM() for _ in 1:(length(dims) - length(subs))])
  end
  length(subs) == length(dims) || return _emptyTargets

  local elemTy = _arrayElementType(baseTy)
  local out = Tuple{DAE.Exp, Union{Nothing, DAE.Exp}, Vector{Int}}[]
  function rec(pos::Int, newSubs::Vector{DAE.Subscript}, guard, rhsIdxs::Vector{Int})
    if pos > length(dims)
      local newCr = _replaceInnermostSubscripts(cr, newSubs)
      push!(out, (DAE.CREF(newCr, elemTy), guard, copy(rhsIdxs)))
      return
    end
    local sub = subs[pos]
    if sub isa DAE.WHOLEDIM
      for k in 1:dims[pos]
        rec(pos + 1, DAE.Subscript[newSubs..., DAE.INDEX(_dimIndexExp(rawDims[pos], k))],
            guard, Int[rhsIdxs..., k])
      end
    elseif sub isa DAE.INDEX
      local idx = sub.exp
      if idx isa DAE.ICONST || idx isa DAE.ENUM_LITERAL
        rec(pos + 1, DAE.Subscript[newSubs..., DAE.INDEX(idx)], guard, rhsIdxs)
      else
        for k in 1:dims[pos]
          local g = _andCondition(guard, _indexEqualsCondition(idx, k))
          rec(pos + 1, DAE.Subscript[newSubs..., DAE.INDEX(_dimIndexExp(rawDims[pos], k))], g, rhsIdxs)
        end
      end
    else
      return
    end
  end
  rec(1, DAE.Subscript[], nothing, Int[])
  return out
end

Base.@nospecializeinfer function _scalarizeCrefRead(@nospecialize(cr),
                                                    @nospecialize(expTy),
                                                    rhsIdxs::Vector{Int},
                                                    @nospecialize(fallback))
  local baseTy = _innermostType(cr)
  local dims = _arrayDimsFromType(baseTy)
  if isempty(dims)
    return DAE.CREF(cr, expTy)
  end
  local rawDims = _rawArrayDims(baseTy)
  local subs = _innermostSubscripts(cr)
  if isempty(subs)
    subs = DAE.Subscript[DAE.WHOLEDIM() for _ in dims]
  elseif length(subs) < length(dims)
    append!(subs, DAE.Subscript[DAE.WHOLEDIM() for _ in 1:(length(dims) - length(subs))])
  end
  length(subs) == length(dims) || return fallback

  local elemTy = _arrayElementType(baseTy)
  local candidates = Tuple{Union{Nothing, DAE.Exp}, DAE.Exp}[]
  function rec(pos::Int, rhsPos::Int, newSubs::Vector{DAE.Subscript}, guard)
    if pos > length(dims)
      local newCr = _replaceInnermostSubscripts(cr, newSubs)
      push!(candidates, (guard, DAE.CREF(newCr, elemTy)))
      return
    end
    local sub = subs[pos]
    if sub isa DAE.WHOLEDIM
      rhsPos <= length(rhsIdxs) || return
      local k = rhsIdxs[rhsPos]
      rec(pos + 1, rhsPos + 1, DAE.Subscript[newSubs..., DAE.INDEX(_dimIndexExp(rawDims[pos], k))], guard)
    elseif sub isa DAE.INDEX
      local idx = sub.exp
      if idx isa DAE.ICONST || idx isa DAE.ENUM_LITERAL
        rec(pos + 1, rhsPos, DAE.Subscript[newSubs..., DAE.INDEX(idx)], guard)
      else
        for k in 1:dims[pos]
          local g = _andCondition(guard, _indexEqualsCondition(idx, k))
          rec(pos + 1, rhsPos, DAE.Subscript[newSubs..., DAE.INDEX(_dimIndexExp(rawDims[pos], k))], g)
        end
      end
    else
      return
    end
  end
  rec(1, 1, DAE.Subscript[], nothing)
  isempty(candidates) && return fallback

  local result = fallback
  for (guard, value) in reverse(candidates)
    result = guard === nothing ? value : DAE.IFEXP(guard, value, result)
  end
  return result
end

Base.@nospecializeinfer function _scalarizeRhs(@nospecialize(rhs),
                                               rhsIdxs::Vector{Int},
                                               @nospecialize(fallback))
  if rhs isa DAE.CREF
    return _scalarizeCrefRead(rhs.componentRef, rhs.ty, rhsIdxs, fallback)
  elseif isempty(rhsIdxs)
    return rhs
  else
    return DAE.ASUB(rhs, _iconstIndexSubList(rhsIdxs))
  end
end

Base.@nospecializeinfer function _emitAlgorithmAssignOps!(ops::Vector{BDAE.WhenOperator},
                                                          liftedLhsNames::OrderedSet{String},
                                                          @nospecialize(lhs),
                                                          @nospecialize(rhs),
                                                          @nospecialize(ty),
                                                          @nospecialize(source),
                                                          @nospecialize(activeCond),
                                                          iterVals::Dict{String, Int},
                                                          initialValue::Bool,
                                                          allowContinuous::Bool = false)::Bool
  (allowContinuous || _isDiscreteDAEType(ty)) || return true
  lhs = _prepareAlgorithmExp(lhs, iterVals, initialValue)
  rhs = _prepareAlgorithmExp(rhs, iterVals, initialValue)
  lhs isa DAE.CREF || return false
  local targets = _scalarLhsTargets(lhs, ty)
  isempty(targets) && return false
  for (lhsK, lhsGuard, rhsIdxs) in targets
    lhsK isa DAE.CREF || continue
    local cond = _andCondition(activeCond, lhsGuard)
    if cond isa DAE.BCONST && !cond.bool
      continue
    end
    local rhsK = _scalarizeRhs(rhs, rhsIdxs, lhsK)
    local finalRhs = cond === nothing ? rhsK : DAE.IFEXP(cond, rhsK, lhsK)
    push!(ops, BDAE.ASSIGN(lhsK, finalRhs, source))
    push!(liftedLhsNames, string(lhsK.componentRef))
  end
  return true
end

Base.@nospecializeinfer function _appendElseAlgorithmOps!(ops::Vector{BDAE.WhenOperator},
                                                          liftedLhsNames::OrderedSet{String},
                                                          @nospecialize(elsePart),
                                                          @nospecialize(activeCond),
                                                          iterVals::Dict{String, Int},
                                                          initialValue::Bool,
                                                          allowContinuous::Bool = false;
                                                          where::String = "a when body")::Bool
  if elsePart isa DAE.NOELSE
    return true
  elseif elsePart isa DAE.ELSE
    return _appendAlgorithmStmtOps!(ops, liftedLhsNames, elsePart.statementLst,
                                    activeCond, iterVals, initialValue, allowContinuous; where = where)
  elseif elsePart isa DAE.ELSEIF
    local cond = _prepareAlgorithmExp(elsePart.exp, iterVals, initialValue)
    local branchCond = _andCondition(activeCond, cond)
    _appendAlgorithmStmtOps!(ops, liftedLhsNames, elsePart.statementLst,
                             branchCond, iterVals, initialValue, allowContinuous; where = where) || return false
    local restCond = _andCondition(activeCond, _notCondition(cond))
    return _appendElseAlgorithmOps!(ops, liftedLhsNames, elsePart.else_,
                                    restCond, iterVals, initialValue, allowContinuous; where = where)
  end
  return true
end

Base.@nospecializeinfer function _appendAlgorithmStmtOps!(ops::Vector{BDAE.WhenOperator},
                                                          liftedLhsNames::OrderedSet{String},
                                                          @nospecialize(stmts),
                                                          @nospecialize(activeCond),
                                                          iterVals::Dict{String, Int},
                                                          initialValue::Bool,
                                                          allowContinuous::Bool = false;
                                                          where::String = "a when body")::Bool
  for s in stmts
    @match s begin
      DAE.STMT_ASSIGN(ty, lhs, rhs, src) => begin
        _emitAlgorithmAssignOps!(ops, liftedLhsNames, lhs, rhs, ty, src,
                                 activeCond, iterVals, initialValue, allowContinuous) || return false
      end
      DAE.STMT_ASSIGN_ARR(ty, lhs, rhs, src) => begin
        _emitAlgorithmAssignOps!(ops, liftedLhsNames, lhs, rhs, ty, src,
                                 activeCond, iterVals, initialValue, allowContinuous) || return false
      end
      DAE.STMT_IF(cond, body, elsePart, _) => begin
        local c = _prepareAlgorithmExp(cond, iterVals, initialValue)
        _appendAlgorithmStmtOps!(ops, liftedLhsNames, body, _andCondition(activeCond, c),
                                 iterVals, initialValue, allowContinuous; where = where) || return false
        _appendElseAlgorithmOps!(ops, liftedLhsNames, elsePart,
                                 _andCondition(activeCond, _notCondition(c)),
                                 iterVals, initialValue, allowContinuous; where = where) || return false
      end
      DAE.STMT_FOR(_, _, iter, _, range, body, _) => begin
        local r = _prepareAlgorithmExp(range, iterVals, initialValue)
        local vals = _rangeIntValues(r)
        vals === nothing && return false
        for v in vals
          local nested = copy(iterVals)
          nested[iter] = v
          _appendAlgorithmStmtOps!(ops, liftedLhsNames, body, activeCond,
                                   nested, initialValue, allowContinuous; where = where) || return false
        end
      end
      #= (a, b, ...) := f(...) (MSL TimeTable's when: (a, b, nextEventScaled,
         last) := getInterpolationCoefficients(...); dropped, the table output
         had no definition). f runs once per target: an impure f is not
         supported. =#
      DAE.STMT_TUPLE_ASSIGN(_, targets, rhs, src) => begin
        (rhs isa DAE.CALL && rhs.attr.isImpure) &&
          OMBackend.unsupported("a tuple assignment from an impure function", rhs)
        for (target, element) in _tupleAssignElements(targets, rhs)
          _emitAlgorithmAssignOps!(ops, liftedLhsNames, target, element, target.ty, src,
                                   activeCond, iterVals, initialValue, allowContinuous) || return false
        end
      end
      #= Nested whens are not Modelica; the top-level ones are lifted on their own. =#
      DAE.STMT_WHEN(__) => nothing
      #= An assert in a when body (`allowContinuous`: the when lifter) is checked
         where the body runs, under its conditions: `not guard or c`. In a regular
         algorithm the top-level asserts are synthesizeAssertsFromRegularAlgorithms'. =#
      DAE.STMT_ASSERT(c, msg, level, src) => begin
        if allowContinuous
          local cond = _prepareAlgorithmExp(c, iterVals, initialValue)
          activeCond === nothing || (cond = _orCondition(_notCondition(activeCond), cond))
          push!(ops, BDAE.ASSERT(cond, _prepareAlgorithmExp(msg, iterVals, initialValue), level, src))
        end
      end
      DAE.STMT_TERMINATE(msg, src) => begin
        #= Under a constant guard: always or never (`if initial()` in the
           runtime arm). =#
        if allowContinuous && !(activeCond isa DAE.BCONST && !activeCond.bool)
          (activeCond === nothing || activeCond isa DAE.BCONST) ||
            OMBackend.unsupported("terminate() under a condition in $(where)", s)
          push!(ops, BDAE.TERMINATE(_prepareAlgorithmExp(msg, iterVals, initialValue), src))
        end
      end
      #= A call for its side effects (a print, a file): no variable depends on it.
         Not run. =#
      DAE.STMT_NORETCALL(__) => nothing
      _ => (allowContinuous && OMBackend.unsupported("this statement in $(where)", s))
    end
  end
  return true
end

#= `(a, b, ...) := f(...)` as one assignment per target of its element of f's
   result, in the order they run. Each evaluates f again after the ones
   before it have assigned their targets, so a target f reads (outside pre())
   goes last and f sees its old value in every element: `(s, y) := step(s, u)`
   gave y the step from the new s. A second such target, or one assigned
   element by element (an array, a record), cannot be last: not supported.
   A generated f returns a record output as its fields in place: a record
   target is one assignment per field, and f's elements are counted so
   (each target after a record took a field of it). =#
function _tupleAssignElements(@nospecialize(targets), @nospecialize(rhs))::Vector{Tuple{DAE.Exp, DAE.Exp}}
  local read = _crefNamesOutsidePre(rhs)
  local plain = Tuple{DAE.Exp, DAE.Exp}[]
  local readBack = Tuple{DAE.Exp, DAE.Exp}[]
  local outputTypes = rhs isa DAE.CALL && rhs.attr.ty isa DAE.T_TUPLE ? collect(rhs.attr.ty.types) : nothing
  local pos = 0
  local elements = Tuple{DAE.Exp, DAE.Exp}[]
  for (k, target) in enumerate(targets)
    target isa DAE.CREF ||
      OMBackend.unsupported("a tuple assignment to a target that is not a variable", target)
    local outTy = outputTypes === nothing ? target.ty : outputTypes[k]
    if _isRecordType(outTy)
      for field in outTy.varLst
        pos += 1
        _isRecordType(field.ty) && OMBackend.unsupported("a tuple assignment of a record output with record fields", rhs)
        target.componentRef isa DAE.WILD ||
          push!(elements, (BDAEUtil.appendFieldToCref(target, field.name, field.ty), DAE.TSUB(rhs, pos, field.ty)))
      end
    else
      pos += 1
      target.componentRef isa DAE.WILD || push!(elements, (target, DAE.TSUB(rhs, pos, target.ty)))
    end
  end
  for element in elements
    local target = first(element)
    local name = string(target.componentRef)
    if any(r -> _crefNamesOverlap(r, name), read)
      (target.ty isa DAE.T_ARRAY || target.ty isa DAE.T_COMPLEX) &&
        OMBackend.unsupported("a tuple assignment whose function reads the array or record target $(name)", rhs)
      push!(readBack, element)
    else
      push!(plain, element)
    end
  end
  length(readBack) > 1 && OMBackend.unsupported("a tuple assignment whose function reads two of its targets", rhs)
  return vcat(plain, readBack)
end

#= The names of the crefs e reads, those inside pre() (the value before the
   event) left out. =#
function _crefNamesOutsidePre(@nospecialize(e))::Vector{String}
  local visit = (x, acc) -> begin
    (x isa DAE.CALL && x.path isa Absyn.IDENT && x.path.name == "pre") && return (x, false, acc)
    x isa DAE.CREF && push!(acc, string(x.componentRef))
    return (x, true, acc)
  end
  return last(Util.traverseExpTopDown(e, visit, String[]))
end

#= Whether a and b name the same variable or one a part of the other
   (`buf` and `buf[2]`, `r` and `r.x`). =#
function _crefNamesOverlap(a::String, b::String)::Bool
  local (short, long) = length(a) <= length(b) ? (a, b) : (b, a)
  startswith(long, short) || return false
  return length(long) == length(short) || long[nextind(long, lastindex(short))] in ('[', '.')
end

Base.@nospecializeinfer function _buildAlgorithmBodyOps(@nospecialize(daeStmts),
                                                        initialValue::Bool,
                                                        allowContinuous::Bool = false;
                                                        where::String = "a when body")
  local ops = BDAE.WhenOperator[]
  local lhsNames = OrderedSet{String}()
  local ok = _appendAlgorithmStmtOps!(ops, lhsNames, daeStmts, nothing,
                                      Dict{String, Int}(), initialValue, allowContinuous; where = where)
  #= A when body (allowContinuous) that cannot be lowered would drop its when:
     a for loop over a non-constant range, an assignment to a slice or to an
     expression. A regular body is lifted only where it can be. =#
  ok || allowContinuous &&
    OMBackend.unsupported("$(where) with a for loop over a non-constant range, or an assignment to a slice or an expression", daeStmts)
  ok || return (BDAE.WhenOperator[], OrderedSet{String}())
  return (ops, lhsNames)
end

Base.@nospecializeinfer function _pushDiscreteCref!(out::Vector{DAE.ComponentRef},
                                                    seen::OrderedSet{String},
                                                    blocked::OrderedSet{String},
                                                    @nospecialize(cref))
  cref isa DAE.ComponentRef || return nothing
  _isTimeCref(cref) && return nothing
  local key = string(cref)
  key in blocked && return nothing
  local ty = _crefType(cref)
  ty === nothing && return nothing
  _isDiscreteDAEType(ty) || return nothing
  key in seen && return nothing
  push!(seen, key)
  push!(out, cref)
  return nothing
end

# Collect discrete RHS crefs into `out` (deduped via `seen`, skipping `blocked`
# reduction/for iterators). Typed functor replacing the threaded ctx tuple.
struct DiscreteRhsCrefVisitor
  out::Vector{DAE.ComponentRef}
  seen::OrderedSet{String}
  blocked::OrderedSet{String}
end
Base.@nospecializeinfer function (v::DiscreteRhsCrefVisitor)(@nospecialize(exp), arg::Nothing)
  @match exp begin
    DAE.CREF(cr, _) => _pushDiscreteCref!(v.out, v.seen, v.blocked, cr)
    DAE.REDUCTION(_, _, iters) => _collectReductionIterNames!(v.blocked, iters)
    _ => nothing
  end
  return (exp, true, arg)
end

Base.@nospecializeinfer function _collectReductionIterNames!(blocked::OrderedSet{String}, @nospecialize(iters))
  for it in iters
    @match it begin
      DAE.REDUCTIONITER(id, _, _, _) => push!(blocked, id)
      _ => nothing
    end
  end
  return nothing
end

Base.@nospecializeinfer function _makeChangeCall(@nospecialize(cref))
  local ty = _crefType(cref)
  local callArg = DAE.CREF(cref, ty === nothing ? DAE.T_REAL_DEFAULT : ty)
  return DAE.CALL(Absyn.IDENT("change"),
                  MetaModelica.list(callArg),
                  DAE.callAttrBuiltinBool)
end


Base.@nospecializeinfer function _collectDiscreteRhsCrefsFromWhenOps(ops::Vector{BDAE.WhenOperator},
                                                                     assignedLhs::OrderedSet{String})
  local out = DAE.ComponentRef[]
  local seen = OrderedSet{String}()
  local blocked = copy(assignedLhs)
  local ctx = DiscreteRhsCrefVisitor(out, seen, blocked)
  for op in ops
    @match op begin
      BDAE.ASSIGN(_, rhs, _) => Util.traverseExpTopDown(rhs, ctx, nothing)
      BDAE.NORETCALL(exp, _) => Util.traverseExpTopDown(exp, ctx, nothing)
      BDAE.ASSERT(c, m, l, _) => begin
        Util.traverseExpTopDown(c, ctx, nothing)
        Util.traverseExpTopDown(m, ctx, nothing)
        Util.traverseExpTopDown(l, ctx, nothing)
      end
      _ => nothing
    end
  end
  return out
end

#= Lift an entire (non-when, non-initial) algorithm body into a single
   INITIAL_WHEN_EQUATION whose body executes the algorithm sequentially at
   init time. Each top-level STMT_ASSIGN with a discrete LHS becomes a
   BDAE.ASSIGN(lhs, rhs); each top-level STMT_IF with `{ STMT_ASSIGN(disc, e) }`
   body becomes a BDAE.ASSIGN(lhs, IFEXP(cond, e, lhs)) so the if-check is
   re-evaluated at init and the assignment is conditional. Compound
   shapes (multi-stmt if-bodies, FOR, WHILE) and continuous-LHS assigns
   are skipped. Records every LHS that contributed an ASSIGN op into
   `liftedLhsNames` so the residual lifter does not also emit a competing
   residual for the same variable. =#
Base.@nospecializeinfer function _liftAlgorithmBodyToInitialWhen!(out::Vector{BDAE.Equation},
                                                                  daeStmts,
                                                                  @nospecialize(source),
                                                                  liftedLhsNames::OrderedSet{String},
                                                                  paramOrConstNames::OrderedSet{String} = OrderedSet{String}())
  local initOps, initLhs = _buildAlgorithmBodyOps(daeStmts, true)
  local runOps, runLhs = _buildAlgorithmBodyOps(daeStmts, false)
  union!(liftedLhsNames, initLhs)
  union!(liftedLhsNames, runLhs)
  isempty(initOps) && isempty(runOps) && return
  local initialCall = DAE.CALL(Absyn.IDENT("initial"),
                               MetaModelica.list(),
                               DAE.callAttrBuiltinBool)
  if !isempty(initOps)
    push!(out, BDAE.INITIAL_WHEN_EQUATION(
      length(initOps),
      BDAE.WHEN_STMTS(initialCall, MetaModelica.list(initOps...), NONE()),
      source,
      BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
    ))
  end
  #= Per Modelica spec §17.4.4: a non-when algorithm with discrete LHS fires
     at any event that changes its RHS inputs. The INITIAL_WHEN above sets the
     LHS at t=0; we also need a regular WHEN_EQUATION whose condition is
     `change(d1) OR change(d2) ... ` over every discrete cref referenced
     in the body, so the LHS keeps tracking those inputs as they flip during
     simulation. Without this, AlgorithmDiscreteAssign's `out := trigger + 10`
     would stay pinned at its t=0 value (out = 13) even after `trigger`
     becomes 7 at t=0.5. =#
  local discRhsCrefs = _collectDiscreteRhsCrefsFromWhenOps(runOps, runLhs)
  #= Source-style bodies (Sources.Table/Step/Pulse) have the shape
     `y := y0; y := IFEXP(time>=t[i], x[i], y)`. Their event sources are the
     `time>=t[i]` RELATIONS in the IFEXP conditions, not the seed cref — so also
     trigger on `change(rel)` for every body relation with a continuous operand.
     Without this a body whose only discrete RHS cref is the constant seed (y0)
     gets a dead `change(<param>)` trigger and never fires. =#
  local relTriggers = DAE.Exp[]
  local relSeen = OrderedSet{String}()
  for op in runOps
    @match op begin
      BDAE.ASSIGN(_, rhs, _) => begin
        for r in _collectRelationsInExp(rhs)
          _relationHasContinuousOperand(r, paramOrConstNames) || continue
          local k = string(r)
          k in relSeen && continue
          push!(relSeen, k)
          push!(relTriggers, r)
        end
      end
      _ => nothing
    end
  end
  local changeCalls = DAE.Exp[]
  for cr in discRhsCrefs
    push!(changeCalls, _makeChangeCall(cr))
  end
  for r in relTriggers
    push!(changeCalls, _makeChangeCallExp(r))
  end
  if !isempty(runOps) && !isempty(changeCalls)
    local cond = changeCalls[1]
    for i in 2:length(changeCalls)
      cond = DAE.LBINARY(cond, DAE.OR(DAE.T_BOOL_DEFAULT), changeCalls[i])
    end
    push!(out, BDAE.WHEN_EQUATION(
      length(runOps),
      BDAE.WHEN_STMTS(cond, MetaModelica.list(runOps...), NONE()),
      source,
      BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
    ))
  end
  return
end

#= Logical OR of two condition expressions with BCONST simplification, the
   disjunctive analogue of `_andCondition`. =#
Base.@nospecializeinfer function _orCondition(@nospecialize(a), @nospecialize(b))
  a === nothing && return b
  b === nothing && return a
  if a isa DAE.BCONST
    return a.bool ? a : b
  elseif b isa DAE.BCONST
    return b.bool ? b : a
  end
  return DAE.LBINARY(a, DAE.OR(DAE.T_BOOL_DEFAULT), b)
end

#= A `when {c1, c2, ...}` array condition means "fire when any member becomes
   true". Return the member expressions so they can be OR-folded; a scalar
   condition is returned as a singleton. =#
Base.@nospecializeinfer function _whenConditionMembers(@nospecialize(exp))
  @match exp begin
    DAE.ARRAY(_, _, arr) => collect(arr)
    _ => Any[exp]
  end
end

Base.@nospecializeinfer function _expMentionsInitial(@nospecialize(exp))::Bool
  local found = false
  function visit(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("initial"), _, _) => (found = true)
      _ => nothing
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(exp, visit, nothing)
  return found
end

_isInitialCall(@nospecialize(e)) = e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name == "initial"

#= MLS 8.6: a when is active at the initialization only in the forms `when
   initial()` and `when {..., initial(), ...}`. =#
_isInitialWhenCondition(@nospecialize(cond)) =
  _isInitialCall(cond) || (cond isa DAE.ARRAY && any(_isInitialCall, cond.array))

#= Elsewhere initial() is true only at the initialization, where the when is not
   active, and false after it: an initial() disjunct of an `or` is dropped
   (`when initial() or x > 0.6` fires at x's crossing only, as OpenModelica; its
   crossing function was `0 - (x > 0.6)`, 0 then -1, and never fired). =#
function dropInitialDisjuncts(@nospecialize(cond))
  _isInitialCall(cond) && return cond
  cond isa DAE.ARRAY &&
    return DAE.ARRAY(cond.ty, cond.scalar, MetaModelica.list((_isInitialCall(e) ? e : _withoutInitialDisjunct(e) for e in cond.array)...))
  return _withoutInitialDisjunct(cond)
end

function _withoutInitialDisjunct(@nospecialize(e))
  e isa DAE.LBINARY && e.operator isa DAE.OR || return e
  local l = _withoutInitialDisjunct(e.exp1)
  local r = _withoutInitialDisjunct(e.exp2)
  _isInitialCall(l) && return r
  _isInitialCall(r) && return l
  return DAE.LBINARY(l, e.operator, r)
end

#= Build a runtime `BDAE.WHEN_EQUATION` (with chained elsewhen) from a
   `DAE.STMT_WHEN` that appears inside a regular (non-when) algorithm body.
   Every assignment in each branch body is lifted (continuous and discrete
   alike — inside a `when` all LHS are event-updated), `if` guards become
   IFEXP-conditional assigns, and an array condition `{c1, c2}` stays one.
   `initial()` is substituted to `false` for the runtime arm.
   Assigned LHS names accumulate into `allLhs`. Returns the WHEN_EQUATION or
   `nothing` if the branch contributes no operators. =#
Base.@nospecializeinfer function _stmtWhenToBdaeWhenEquation(@nospecialize(stmtWhen),
                                                             allLhs::OrderedSet{String})
  #= A vector `{c1, c2}` stays one (each element's edge fires it, the shared
     fold in simulationCodeTransformation); OR-folded, it fired on the OR's
     edge only. initial() is false here: no trigger. =#
  local members = Any[_prepareAlgorithmExp(e, Dict{String, Int}(), false) for e in _whenConditionMembers(stmtWhen.exp)]
  filter!(m -> !(m isa DAE.BCONST && !m.bool), members)
  local cond = if isempty(members)
    DAE.BCONST(false)
  elseif length(members) == 1 || !(stmtWhen.exp isa DAE.ARRAY)
    foldl(_orCondition, members)
  else
    DAE.ARRAY(DAE.T_ARRAY(DAE.T_BOOL_DEFAULT, MetaModelica.list(DAE.DIM_INTEGER(length(members)))), false,
              MetaModelica.list(members...))
  end
  #= A branch that never runs here (`when initial()`: false after the start)
     contributes nothing; its body is the initial lowering's. =#
  local (ops, lhs) = cond isa DAE.BCONST && !cond.bool ? (BDAE.WhenOperator[], OrderedSet{String}()) :
                     _buildAlgorithmBodyOps(stmtWhen.statementLst, false, true)
  union!(allLhs, lhs)
  local elseOpt = NONE()
  @match stmtWhen.elseWhen begin
    SOME(esw) => begin
      if esw isa DAE.STMT_WHEN
        local eswEq = _stmtWhenToBdaeWhenEquation(esw, allLhs)
        eswEq !== nothing && (elseOpt = SOME(eswEq))
      end
    end
    _ => nothing
  end
  (isempty(ops) && elseOpt === NONE()) && return nothing
  local whenStmts = BDAE.WHEN_STMTS(cond, MetaModelica.list(ops...), elseOpt)
  return BDAE.WHEN_EQUATION(length(ops), whenStmts, stmtWhen.source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
end

#= Lift a top-level `DAE.STMT_WHEN` from a regular algorithm body into BDAE
   equations: an INITIAL_WHEN_EQUATION (when the first branch carries
   `initial()`, so the scheduling state is set at t=0) plus a runtime
   WHEN_EQUATION with the elsewhen arm. Without this a mixed algorithm body
   (a `when/elsewhen` followed by plain assignments) loses the `when` block
   entirely, leaving its LHS frozen at the start value. =#
Base.@nospecializeinfer function _liftStmtWhenToWhenEquations!(out::Vector{BDAE.Equation},
                                                               @nospecialize(stmtWhen),
                                                               liftedLhsNames::OrderedSet{String})::Bool
  local allLhs = OrderedSet{String}()
  if stmtWhen.initialCall || _isInitialWhenCondition(stmtWhen.exp)
    #= An initial arm the lifter cannot lower stays empty, as before: a first
       branch `when initial()` is synthesizeInitialWhenFromAlgorithms' (it runs
       the DAE statements). =#
    local (initOps, initLhs) = try
      _buildAlgorithmBodyOps(stmtWhen.statementLst, true, true)
    catch err
      err isa OMBackend.UnsupportedLowering || rethrow()
      (BDAE.WhenOperator[], OrderedSet{String}())
    end
    union!(allLhs, initLhs)
    if !isempty(initOps)
      local initialCall = DAE.CALL(Absyn.IDENT("initial"),
                                   MetaModelica.list(),
                                   DAE.callAttrBuiltinBool)
      push!(out, BDAE.INITIAL_WHEN_EQUATION(
        length(initOps),
        BDAE.WHEN_STMTS(initialCall, MetaModelica.list(initOps...), NONE()),
        stmtWhen.source,
        BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
      ))
    end
  end
  local weq = _stmtWhenToBdaeWhenEquation(stmtWhen, allLhs)
  weq !== nothing && push!(out, weq)
  union!(liftedLhsNames, allLhs)
  return weq !== nothing
end


#= If `stmt` is a top-level `STMT_IF { cond, body = [STMT_ASSIGN(disc, expr)] }`
   (no else branch needed for sources; the assignment is idempotent and
   monotone-time conditions sustain), synthesise a
   `BDAE.WHEN_EQUATION` triggered by `cond` whose body assigns the discrete
   LHS to `expr`. Returns `true` when a lift fired so the caller can record
   that this algorithm has been (partly) handled. =#
Base.@nospecializeinfer function _liftAlgIfToWhen!(out::Vector{BDAE.Equation},
                                                   @nospecialize(stmt), @nospecialize(source))::Bool
  @match stmt begin
    DAE.STMT_IF(cond, body, _, src) => begin
      local bodyVec = listArray(body)
      length(bodyVec) == 1 || return false
      local b1 = bodyVec[1]
      @match b1 begin
        DAE.STMT_ASSIGN(ty, lhs, rhs, asrc) => begin
          _isDiscreteDAEType(ty) || return false
          lhs isa DAE.CREF || return false
          local whenOps = MetaModelica.list(BDAE.ASSIGN(lhs, rhs, asrc))
          push!(out, BDAE.WHEN_EQUATION(
            1,
            BDAE.WHEN_STMTS(cond, whenOps, NONE()),
            source,
            BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
          ))
          return true
        end
        _ => return false
      end
    end
    _ => return false
  end
end

#= Lift a bare `STMT_ASSIGN(disc_lhs, expr)` to a `BDAE.WHEN_EQUATION` whose
   condition is `initial() or change(rhs_crefs)`. Returns `(lifted, lhsName)`
   where `lhsName` is the LHS cref string when lifted. The condition mirrors
   the multi-statement WHEN lifter so callers can rely on the same semantics
   (Modelica §17.4.4: a non-when algorithm with discrete LHS fires at events
   when any of its inputs change). =#
Base.@nospecializeinfer function _liftAlgAssignToInitialWhen!(out::Vector{BDAE.Equation},
                                                              @nospecialize(stmt),
                                                              @nospecialize(source),
                                                              paramOrConstNames::OrderedSet{String} = OrderedSet{String}())
  @match stmt begin
    DAE.STMT_ASSIGN(ty, lhs, rhs, asrc) => begin
      _isDiscreteDAEType(ty) || return (false, nothing)
      lhs isa DAE.CREF || return (false, nothing)
      local bareInitial::DAE.Exp = DAE.CALL(Absyn.IDENT("initial"),
                                            MetaModelica.list(),
                                            DAE.callAttrBuiltinBool)
      local changeCond::Union{DAE.Exp, Nothing} = nothing
      local rhsCrefs = OrderedSet{Tuple{DAE.ComponentRef, DAE.Type}}()
      for c in Util.getAllCrefs(rhs)
        local cty = _crefType(c)
        cty === nothing && continue
        push!(rhsCrefs, (c, cty))
      end
      for (cr, cty) in rhsCrefs
        _isContinuousRealType(cty) && continue
        cty isa DAE.T_ARRAY && continue
        _isTimeCref(cr) && continue
        (string(cr) in paramOrConstNames) && continue
        local changeCall = DAE.CALL(Absyn.IDENT("change"),
                                    MetaModelica.list(DAE.CREF(cr, cty)),
                                    DAE.callAttrBuiltinBool)
        changeCond = if changeCond === nothing
          changeCall
        else
          DAE.LBINARY(changeCond, DAE.OR(DAE.T_BOOL_DEFAULT), changeCall)
        end
      end
      local whenOps = MetaModelica.list(BDAE.ASSIGN(lhs, rhs, asrc))
      #= Always emit an INITIAL_WHEN_EQUATION so the assign fires through the
         `__runInitialAlgorithm!` path at t=0. The synthesised `WHEN_EQUATION`
         with `cond = initial()` would not work because `expToJuliaBoolMTK`
         lowers `initial()` to `false` (the runtime DiscreteCallback never
         runs during MTK's InitializationProblem). =#
      push!(out, BDAE.INITIAL_WHEN_EQUATION(
        1,
        BDAE.WHEN_STMTS(bareInitial,
                        MetaModelica.list(BDAE.ASSIGN(lhs, rhs, asrc)),
                        NONE()),
        source,
        BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
      ))
      #= Plus a runtime WHEN_EQUATION for any change(rhs) trigger so the
         assign re-fires whenever a non-parameter input changes. =#
      if changeCond !== nothing
        push!(out, BDAE.WHEN_EQUATION(
          1,
          BDAE.WHEN_STMTS(changeCond, whenOps, NONE()),
          source,
          BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
        ))
      end
      local lhsName::Union{String, Nothing} = @match lhs begin
        DAE.CREF(cr, _) => string(cr)
        _ => nothing
      end
      return (true, lhsName)
    end
    _ => return (false, nothing)
  end
end

"""
    synthesizeInitialWhenFromAlgorithms(algorithms) -> Vector{BDAE.Equation}

Scan flat-model algorithm sections for `algorithm when initial() then ... end when`
statements and lift each into a `BDAE.INITIAL_WHEN_EQUATION`. Bodies are translated
via OMFrontend's existing Statement → DAE.Statement conversion, then mapped to
BDAE.WhenOperator entries. Compound conditions (e.g. `when (initial() or c)`) are
intentionally skipped per Modelica spec §8/§11.
"""
function synthesizeInitialWhenFromAlgorithms(algorithms)::Vector{BDAE.Equation}
  local out = BDAE.Equation[]
  for alg in algorithms
    for stmt in alg.statements
      isvariant(stmt, OMFrontend.Frontend.ALG_WHEN) || continue
      isempty(stmt.branches) && continue
      local (frontendCond, frontendBody) = stmt.branches[1]
      local daeCond = OMFrontend.Frontend.toDAE(frontendCond)
      @match daeCond begin
        DAE.CALL(Absyn.IDENT("initial"), _, _) => begin
          local daeStmts = OMFrontend.Frontend.convertStatements(frontendBody)
          local whenOps = _daeStmtsToWhenOps(daeStmts)
          local node = BDAE.INITIAL_WHEN_EQUATION(
            length(frontendBody),
            BDAE.WHEN_STMTS(daeCond, whenOps, NONE()),
            stmt.source,
            BDAE.EQ_ATTR_DEFAULT_UNKNOWN,
          )
          saveInitialAlgorithmStatements!(node, collect(daeStmts))
          push!(out, node)
        end
        _ => nothing
      end
    end
  end
  return out
end
