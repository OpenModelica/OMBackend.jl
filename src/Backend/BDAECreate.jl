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
This module contain the various functions that are related to the lowering
of the DAE IR into Backend DAE IR (BDAE IR). BDAE IR is the representation we use
before code generation.
"""
module BDAECreate

using MetaModelica
using ExportAll
using DataStructures: OrderedDict, OrderedSet

import ..BDAE
import ..BDAEUtil
import ..FrontendUtil.Util
import ..@BACKEND_PERFLOG
import ...OMBackend
import Absyn
import DAE
import OMFrontend

#= The DAE statements of initial algorithms: BDAEUtil.INIT_ALG_DAE_STMTS. =#
import ..BDAEUtil: saveInitialAlgorithmStatements!, initialAlgorithmStatements, rekeyInitialAlgorithmStatements!

"""
  This function translates a DAE, which is the result from instantiating a
  class, into a more precise form, called BDAE.BDAE defined in this module.
  The BDAE.BDAE representation splits the DAE into equations and variables
  and further divides variables into known and unknown variables and the
  equations into simple and nonsimple equations.
  inputs:  lst: DAE.DAE_LIST
  outputs: BDAE.BACKEND_DAE
"""
function lower(lst::DAE.DAE_LIST)::BDAE.BACKEND_DAE
  local outBDAE::BDAE.BACKEND_DAE
  local eqSystems::Vector{BDAE.EQSYSTEM}
  local varArray::Vector{BDAE.VAR}
  local eqArray::Vector{BDAE.Equation}
  local name = listHead(lst.elementLst).ident
  (varArray, eqArray, initialEquations) = begin
    local elementLst::List{DAE.Element}
    local variableLst::List{BDAE.VAR}
    local equationLst::List{BDAE.Equation}
    @match lst begin
      DAE.DAE_LIST(elementLst) => begin
        (variableLst, equationLst, initialEquations) = splitEquationsAndVars(elementLst)
        (listArray(listReverse(variableLst)), listArray(listReverse(equationLst)), initialEquations)
      end
    end
  end
  local variables = BDAEUtil.convertVarArrayToBDAE_Variables(varArray)
  # @debug "varArray:" length(variableLst)
  #@debug "eqLst:" length(equationLst)
  #= We start with an array of one system =#
  eqSystems = BDAE.EQSYSTEM[BDAE.EQSYSTEM(name, variables, eqArray, BDAE.Equation[], BDAE.Equation[])]
  outBDAE = BDAE.BACKEND_DAE(name, eqSystems, BDAE.SHARED(BDAE.VAR[], BDAE.VAR[], NONE()))
end

"""
  Lowers a FlatModelica defined in the new frontend into BDAE.
  1. We translate all different components of the flat model into the DAE representation.
  2. We convert this representation into the BackendDAE representation.
  3. We return backend DAE to be used in the remainder of the compilation before code generation.
"""
function lower(flatModelica::OMFrontend.Frontend.FlatModel)
  #= Creates a list of flat equation systems =#
  local eqSystems = createEqSystems(flatModelica)
  local shared = BDAE.SHARED(BDAE.VAR[], BDAE.VAR[], flatModelica.scodeProgram)
  #= The resulting backend DAE. =#
  return createBackendDAE(flatModelica.name, eqSystems, shared)
end

function createBackendDAE(name, eqSystems, shared)
  local outBDAE = BDAE.BACKEND_DAE(name, eqSystems, shared)
  return outBDAE
end

"""
  Creates one or more equation systems
"""
function createEqSystems(frontendDAE::OMFrontend.Frontend.FlatModel)::Vector{BDAE.EQSYSTEM}
  #= Create the first main equation system. =#
  local eqSystems = Any[createEqSystem(frontendDAE)]
  if ! listEmpty(frontendDAE.structuralSubmodels)
    local res = createEqSystemsWork(frontendDAE.structuralSubmodels)
    push!(eqSystems, res)
  end
  #= But what if a submodel in turn has more equation systems in it..  Currently this only handles one level. =#
  local res2 = vcat(eqSystems...)
  return res2
end

"""
  Creates a flat list of equation systems.
"""
function createEqSystemsWork(structuralSubmodels::List{OMFrontend.Frontend.FlatModel})
  local eqSystems = BDAE.EQSYSTEM[]
  for subModel in structuralSubmodels
    push!(eqSystems, createEqSystem(subModel))
  end
  return eqSystems
end

"""
  Creates a single equation system
"""
function createEqSystem(flatModel::OMFrontend.Frontend.FlatModel)
  local name = flatModel.name
  @info "[BDAE: createEqSystem] start" name
  empty!(BDAEUtil.INIT_ALG_DAE_STMTS)
  local equations = BDAE.Equation[]
  for eq in OMFrontend.Frontend.convertEquations(flatModel.equations)
    local result = equationToBackendEquation(eq)
    if result isa Vector
      append!(equations, result)
    else
      push!(equations, result)
    end
  end
  @info "[BDAE: createEqSystem] equations converted" name n=length(equations)
  local variables = [variableToBackendVariable(var)
                     for var in OMFrontend.Frontend.convertVariables(flatModel.variables, list())]
  @info "[BDAE: createEqSystem] variables converted" name n=length(variables)
  local algorithms = [alg for alg in flatModel.algorithms]
  local iAlgorithms = [iAlg for iAlg in flatModel.initialAlgorithms]
  @info "[BDAE: createEqSystem] algorithms collected" name n_alg=length(algorithms) n_initAlg=length(iAlgorithms)
  #= Synthesize BDAE.INITIAL_WHEN_EQUATION entries from `algorithm when initial()`
     statements; without this they vanish at the flat-model → BDAE boundary. =#
  for ieq in synthesizeInitialWhenFromAlgorithms(algorithms)
    push!(equations, ieq)
  end
  #= Lift `initial algorithm` sections into the same INITIAL_WHEN_EQUATION shape
     so the simCode pipeline funnels both `algorithm when initial()` and
     `initial algorithm` into the same `__runInitialAlgorithm!` codegen path.
     Without this every `initial algorithm` block was silently dropped — every
     state seeded only by an init-alg (e.g. trapezoid sources' T_start, count)
     stayed at its default 0. =#
  for ieq in synthesizeFromInitialAlgorithms(iAlgorithms)
    push!(equations, ieq)
  end
  #= Lower the body of each regular `algorithm` section (not `when` and not
     `initial`) into one `BDAE.RESIDUAL_EQUATION` per scalar assignment,
     but ONLY for LHSes that no other equation constrains (a competing
     residual would over-determine MTK's structural-simplify). A connect's
     alias `a = b` does not constrain: the MSL Digital Set source `y := x`,
     connected to a flip-flop's inputs, takes the lift. Multi-statement
     bodies (INV3S's `nextstate := ...; yy := nextstate;`) go through the
     when lifter instead. =#
  #= Residual-lift for the simple `Integer out := trigger + 10` reproducer
     shape (single-statement, LHS not connect-bound). Skipped when the LHS
     would collide with another equation, when the body has multiple
     statements (order-sensitive), or when the LHS is Real. =#
  #= Both the WHEN_EQUATION lifter and the residual lifter are no-ops for
     models without regular algorithm sections. The lifters themselves
     auto-detect per-statement whether their pattern applies (discrete LHS
     plus the right body shape); models that do not match get no emitted
     equations. So no global flag is needed — just gate the eager
     pre-collection on `isempty(algorithms)` to keep LotkaVolterra-style
     models paying zero per-model overhead. =#
  local _whenLifterSkipLhs = OrderedSet{String}()
  if !isempty(algorithms)
    local _eqLhsBoundCrefs = @BACKEND_PERFLOG "[BDAE: lifter] collectAllCrefsInEquations" _collectAllCrefsInEquations(equations)
    local _paramOrConstNames = @BACKEND_PERFLOG "[BDAE: lifter] collectParamOrConstNames" _collectParamOrConstNames(variables)
    local _whenLiftedEqs, _whenLiftedLhs = @BACKEND_PERFLOG "[BDAE: lifter] synthesizeWhenEquationsFromRegularAlgorithms" synthesizeWhenEquationsFromRegularAlgorithms(algorithms, _paramOrConstNames)
    for ieq in _whenLiftedEqs
      push!(equations, ieq)
    end
    _whenLifterSkipLhs = _whenLiftedLhs
    @BACKEND_PERFLOG "[BDAE: lifter] synthesizeResidualsFromRegularAlgorithms" begin
      for ieq in synthesizeResidualsFromRegularAlgorithms(algorithms, _eqLhsBoundCrefs, _whenLifterSkipLhs;
                                                          realStarts = _realStartValues(variables))
        push!(equations, ieq)
      end
    end
    append!(equations, synthesizeAssertsFromRegularAlgorithms(algorithms))
  end
  #= Stringify each varName once; the four name-keyed sweeps below
     (collision-resolve, dedup, param/const, discrete-start) all use the
     default-separator string and share this vector instead of recomputing it. =#
  local varNames = String[string(v.varName) for v in variables]
  local initialEquations = BDAE.Equation[]
  for ieq in OMFrontend.Frontend.convertEquations(flatModel.initialEquations)
    local iresult = equationToBackendEquation(ieq)
    if iresult isa Vector
      append!(initialEquations, iresult)
    else
      push!(initialEquations, iresult)
    end
  end
  #= Distinct crefs can mangle to the same flat name (a.b vs a_b); resolve
     before the name-keyed deduplication silently swallows a variable. =#
  resolveMangledNameCollisions!(variables, equations, initialEquations, varNames)
  #= Deduplicate variables by name (handles inner/outer duplicate emission) =#
  variables = deduplicateVariables(variables, varNames)
  length(variables) == length(varNames) || (varNames = String[string(v.varName) for v in variables])
  #= Deduplicate explicit equations =#
  equations = deduplicateEquations(equations)
  #= The set of equations might also contain a  set of "binding equations" =#
  local bindingEquations = createBindingEquations(variables)
  if !isempty(bindingEquations)
    equations = vcat(equations, bindingEquations)
  end
  #= §17.4.4: lift discrete (Bool/Int/enum) definitions whose RHS is a
     discrete-time relation into event-driven held discretes, so the continuous
     integrator never interpolates step-valued logic. After the bindings: a
     `Boolean open = time > 0.5` defines `open` as an equation does (lifted
     before them, its relation made no event and the switch it drives never
     opened). =#
  if !isempty(equations) && !isempty(variables)
    local _discParamConst = _collectParamOrConstNames(variables, varNames)
    booleanizeWhenRelations!(equations, variables, varNames, _discParamConst)
    local _discStarts = _discreteStartExpLookup(variables, varNames)
    local (_discEqs, _discLifted) = synthesizeWhenEquationsFromDiscreteEquations(equations, _discParamConst, _discStarts;
                                                                               initialConstants = _initialConstants(initialEquations, _discParamConst))
    if !isempty(_discLifted)
      @info "[BDAE: lifter] synthesizeWhenEquationsFromDiscreteEquations lifted $(length(_discLifted)) discrete equation(s)" lifted=collect(_discLifted)
    end
    equations = _discEqs
  end
  equations = _hoistIfEquationAsserts(equations)
  #= TODO Extract the simple equations =#
  local simpleEquations = BDAE.Equation[]
  return BDAE.EQSYSTEM(name, variables, equations, simpleEquations, initialEquations)
end

#= The asserts in the branches of if-equations, as top-level asserts under the
   branch's condition (`not guard or c`): the if-equation lowering takes
   residual equations only and left them out with a warning (MSL Fluid's
   AST_BatchPlant). =#
function _hoistIfEquationAsserts(equations::Vector)::Vector
  any(_hasBranchAssert, equations) || return equations
  local out = BDAE.Equation[]
  local asserts = BDAE.Equation[]
  for eq in equations
    push!(out, _hasBranchAssert(eq) ? _hoistBranchAsserts!(asserts, eq, nothing) : eq)
  end
  return vcat(out, asserts)
end

_hasBranchAssert(@nospecialize(eq))::Bool =
  eq isa BDAE.IF_EQUATION &&
  any(b -> any(e -> e isa BDAE.ASSERT_EQUATION || _hasBranchAssert(e), b), Iterators.flatten((eq.eqnstrue, (eq.eqnsfalse,))))

function _hoistBranchAsserts!(asserts::Vector, ifEq::BDAE.IF_EQUATION, @nospecialize(outerGuard))::BDAE.IF_EQUATION
  local none = nothing  #= no branch before this one was taken =#
  local trueEquations::List{List{BDAE.Equation}} = nil
  for (cond, body) in zip(ifEq.conditions, ifEq.eqnstrue)
    local guard = _andCondition(outerGuard, _andCondition(none, cond))
    trueEquations = _equationList(_takeBranchAsserts!(asserts, body, guard)) <| trueEquations
    none = _andCondition(none, _notCondition(cond))
  end
  local falseEquations = _equationList(_takeBranchAsserts!(asserts, ifEq.eqnsfalse, _andCondition(outerGuard, none)))
  return BDAE.IF_EQUATION(ifEq.conditions, listReverse(trueEquations), falseEquations, ifEq.source, ifEq.attr)
end

function _equationList(eqs::Vector{BDAE.Equation})::List{BDAE.Equation}
  local out::List{BDAE.Equation} = nil
  for eq in Iterators.reverse(eqs)
    out = eq <| out
  end
  return out
end

#= A branch's equations without its asserts, which go to `asserts` under `guard`. =#
function _takeBranchAsserts!(asserts::Vector, body, @nospecialize(guard))::Vector{BDAE.Equation}
  local kept = BDAE.Equation[]
  for eq in body
    if eq isa BDAE.ASSERT_EQUATION
      #= A call for its effects (an assert of its own: _assertConditionExpr) does
         not reach here: the conversion refuses one in a branch. =#
      eq.condition isa DAE.CALL && eq.condition.attr.ty isa DAE.T_NORETCALL &&
        OMBackend.unsupported("a call for its effects in a branch of an if-equation", eq.condition)
      push!(asserts, BDAE.ASSERT_EQUATION(_orCondition(_notCondition(guard), eq.condition), eq.message, eq.level, eq.source))
    elseif eq isa BDAE.IF_EQUATION
      push!(kept, _hoistBranchAsserts!(asserts, eq, guard))
    else
      push!(kept, eq)
    end
  end
  return kept
end

function _crefDepth(cref::DAE.ComponentRef)::Int
  @match cref begin
    DAE.CREF_QUAL(__) => 1 + _crefDepth(cref.componentRef)
    _ => 1
  end
end

function _crefIsSubscriptFree(cref::DAE.ComponentRef)::Bool
  @match cref begin
    DAE.CREF_IDENT(__) => listEmpty(cref.subscriptLst)
    DAE.CREF_QUAL(__) => listEmpty(cref.subscriptLst) && _crefIsSubscriptFree(cref.componentRef)
    _ => false
  end
end

"""
  Distinct component references can mangle to the same flat name, e.g. `a.b`
  and `a_b`. Keep the least-qualified claimant and rename the others to fresh
  unique names, rewriting every occurrence (equations, initial equations and
  bindings) so the name-keyed passes downstream stay sound.
"""
function resolveMangledNameCollisions!(variables::Vector, equations::Vector, initialEquations::Vector,
                                       varNames::Vector{String} = String[string(v.varName) for v in variables])
  local idxsByMangled = OrderedDict{String, Vector{Int}}()
  for (i, v) in enumerate(variables)
    push!(get!(() -> Int[], idxsByMangled, varNames[i]), i)
  end
  local taken = OrderedSet{String}(keys(idxsByMangled))
  local renames = OrderedDict{String, DAE.ComponentRef}()
  for (mangled, idxs) in idxsByMangled
    length(idxs) < 2 && continue
    local groups = OrderedDict{String, Vector{Int}}()
    for i in idxs
      push!(get!(() -> Int[], groups, string(variables[i].varName; separator = ".")), i)
    end
    #= A single group is the inner/outer duplicate-emission case, which
       deduplicateVariables handles. =#
    length(groups) < 2 && continue
    local keepKey = argmin(k -> _crefDepth(variables[first(groups[k])].varName), collect(keys(groups)))
    for (dotted, gidxs) in groups
      dotted == keepKey && continue
      local cref = variables[first(gidxs)].varName
      if !_crefIsSubscriptFree(cref)
        @warn "Mangled-name collision on subscripted variable left unresolved" mangled
        continue
      end
      local k = 1
      local newName = string(mangled, "_", k)
      while newName in taken
        k += 1
        newName = string(mangled, "_", k)
      end
      push!(taken, newName)
      local newCref = DAE.CREF_IDENT(newName, BDAEUtil.crefLeafType(cref), nil)
      renames[dotted] = newCref
      for i in gidxs
        variables[i].varName = newCref
        #= Keep the shared name vector in sync so downstream dedup keys correctly. =#
        varNames[i] = string(newCref)
      end
    end
  end
  isempty(renames) && return nothing
  local rewrite = function (exp::DAE.Exp, arg)
    local res = exp
    @match exp begin
      DAE.CREF(__) => begin
        if _crefIsSubscriptFree(exp.componentRef)
          local hit = get(renames, string(exp.componentRef; separator = "."), nothing)
          if hit !== nothing
            res = DAE.CREF(hit, exp.ty)
          end
        end
        ()
      end
      _ => begin
        ()
      end
    end
    return (res, true, arg)
  end
  #= An initial algorithm's DAE statements are renamed too (the traversal
     re-keyed them to the rebuilt node; a condition or a range is not among
     its flattened ops, so the node can be unchanged while they are not). =#
  local renameExp = e -> first(Util.traverseExpTopDown(e, rewrite, 0))
  for eqs in (equations, initialEquations), i in 1:length(eqs)
    local old = eqs[i]
    (eqs[i], _) = BDAEUtil.traverseEquationExpressions(old, rewrite, 0)
    local stmts = initialAlgorithmStatements(eqs[i])
    isempty(stmts) ||
      rekeyInitialAlgorithmStatements!(eqs[i], eqs[i], DAE.Statement[Util.mapDAEStatementExps(renameExp, s; targets = true) for s in stmts])
  end
  for v in variables
    local b = v.bindExp
    if b isa SOME
      newBind, _ = Util.traverseExpTopDown(b.data, rewrite, 0)
      v.bindExp = SOME(newBind)
    end
  end
  @info "[BDAE] resolved $(length(renames)) mangled-name collision(s) by renaming"
  return nothing
end

"""
  Deduplicate variables by their component reference name.
  Keeps the first occurrence of each uniquely-named variable.
"""
function deduplicateVariables(variables::Vector,
                              varNames::Vector{String} = String[string(v.varName) for v in variables])::Vector
  local idxByName = Dict{String, Int}()
  local unique_vars = similar(variables, 0)
  local duplicateCount = 0
  for (i, v) in enumerate(variables)
    local varStr = varNames[i]
    local existing = get(idxByName, varStr, 0)
    if existing != 0
      duplicateCount += 1
      #= Connect/alias expansion can emit an attribute-less same-named copy; prefer
         the copy carrying the declared start/fixed so the initial condition survives. =#
      if !_varHasStartOrFixed(unique_vars[existing]) && _varHasStartOrFixed(v)
        unique_vars[existing] = v
      end
      continue
    end
    push!(unique_vars, v)
    idxByName[varStr] = length(unique_vars)
  end
  if duplicateCount > 0
    println("[dedup] Variables: $(length(variables)) -> $(length(unique_vars)) (removed $duplicateCount duplicates)")
  end
  return unique_vars
end

function _varHasStartOrFixed(v)::Bool
  local o = v.values
  o isa SOME || return false
  local a = o.data
  return (hasproperty(a, :start) && getproperty(a, :start) isa SOME) ||
         (hasproperty(a, :fixed) && getproperty(a, :fixed) isa SOME)
end

"""
  Generic structural hash for @Record structs and DAE IR nodes.
  Recursively hashes all fields without allocating intermediate strings.
"""
structuralHash(x::Number, h::UInt) = hash(x, h)
structuralHash(x::Symbol, h::UInt) = hash(x, h)
structuralHash(x::String, h::UInt) = hash(x, h)
structuralHash(x::Bool, h::UInt) = hash(x, h)
structuralHash(::Nothing, h::UInt) = hash(nothing, h)
structuralHash(x::Cons, h::UInt) = begin
  for el in x
    h = structuralHash(el, h)
  end
  h
end
structuralHash(::Nil, h::UInt) = hash(:nil, h)
structuralHash(x::SOME, h::UInt) = structuralHash(x.data, hash(:SOME, h))
structuralHash(x::Vector, h::UInt) = begin
  h = hash(length(x), h)
  for el in x
    h = structuralHash(el, h)
  end
  h
end
#= Generic struct fold: unrolled per concrete type so each `getfield(x, i)` has a
   concrete field type and the recursive call dispatches statically (no boxing).
   Semantics identical to the previous runtime-loop fallback: hash the type, then
   each field in order. =#
@generated function structuralHash(x, h::UInt)
  local body = Expr(:block)
  push!(body.args, :(h = hash($x, h)))
  for i in 1:fieldcount(x)
    push!(body.args, :(h = structuralHash(getfield(x, $i), h)))
  end
  push!(body.args, :(return h))
  return body
end
structuralHash(@nospecialize(x)) = structuralHash(x, zero(UInt))

"""
  Deduplicate equations using structural hashing.
  Keeps the first occurrence of each unique equation.
"""
function deduplicateEquations(equations::Vector)::Vector
  local seen = OrderedSet{UInt}()
  local unique_eqs = similar(equations, 0)
  local duplicateCount = 0
  for eq in equations
    local h = structuralHash(eq)
    if h in seen
      duplicateCount += 1
      continue
    end
    push!(seen, h)
    push!(unique_eqs, eq)
  end
  if duplicateCount > 0
    println("[dedup] Equations: $(length(equations)) -> $(length(unique_eqs)) (removed $duplicateCount duplicates)")
  end
  return unique_eqs
end

#= Carry the DAE.VAR `protection` flag onto the variable attribute Option so
   the SimCode-layer `dropObservationOnlyVariables` pass can pick it up. A
   variable without attributes gets the empty ones of its type `ty`. =#
function _maybeMarkAttrProtected(vattr, protection, @nospecialize(ty))
  protection isa DAE.PROTECTED || return vattr
  local va = vattr isa SOME ? vattr.data : _emptyVarAttr(ty)
  @assign va.isProtected = SOME(true)
  return SOME(va)
end

#= The start values of the Real variables that have one: where an algorithm
   section's Real targets start (MLS 11.1.2). =#
function _realStartValues(variables::Vector{BDAE.VAR})::Dict{String, DAE.Exp}
  local out = Dict{String, DAE.Exp}()
  for v in variables
    local attrs = v.values
    (attrs isa SOME && attrs.data isa DAE.VAR_ATTR_REAL && attrs.data.start isa SOME) || continue
    out[string(v.varName)] = attrs.data.start.data
  end
  return out
end

#= The empty attributes of a variable of type `ty` (of an array: of its elements). =#
function _emptyVarAttr(@nospecialize(ty))::DAE.VariableAttributes
  @match ty begin
    DAE.T_ARRAY(ty = elementTy) => _emptyVarAttr(elementTy)
    DAE.T_INTEGER(__) => DAE.emptyVarAttrInt
    DAE.T_BOOL(__) => DAE.emptyVarAttrBool
    DAE.T_STRING(__) => DAE.emptyVarAttrString
    DAE.T_ENUMERATION(__) => DAE.emptyVarAttrEnum
    DAE.T_CLOCK(__) => DAE.emptyVarAttrClock
    _ => DAE.emptyVarAttrReal
  end
end



"""
  Splits a given DAE.DAEList and converts it into a set of BDAE equations and BDAE variables.
  In addition provides the initial equations for the system.
  TODO: Optimize by using List instead of array.
"""
function splitEquationsAndVars(elementLst::List{DAE.Element})::Tuple{List, List, List}
  local variableLst::List{BDAE.VAR} = nil
  local equationLst::List{BDAE.Equation} = nil
  local initialEquationLst::List{BDAE.Equation} = nil
  for elem in elementLst
    _ = begin
      local backendDAE_Var
      local backendDAE_Equation
      @match elem begin
        DAE.VAR(__) => begin
          variableLst = BDAE.VAR(elem.componentRef,
          BDAEUtil.DAE_VarKind_to_BDAE_VarKind(elem.kind),
          elem.direction,
          elem.ty,
          elem.binding,
          elem.dims,
          elem.source,
          _maybeMarkAttrProtected(elem.variableAttributesOption, elem.protection, elem.ty),
          NONE(), #=Tearing=#
          elem.connectorType,
          false #=We do not know if we can replace or not yet=#
          ) <| variableLst
        end
        DAE.EQUATION(__) => begin
          equationLst = BDAE.EQUATION(elem.exp,
                                      elem.scalar,
                                      elem.source,
                                      BDAE.EQ_ATTR_DEFAULT_UNKNOWN) <| equationLst
        end
        DAE.WHEN_EQUATION(__) => begin
          equationLst = lowerWhenEquation(elem) <| equationLst
        end
        DAE.IF_EQUATION(__) => begin
          equationLst = lowerIfEquation(elem) <| equationLst
        end
        DAE.INITIALEQUATION(__) => begin
          initialEquationLst = BDAE.EQUATION(elem.exp1,
          elem.exp2,
          elem.source,
          BDAE.EQ_ATTR_DEFAULT_UNKNOWN) <| initialEquationLst
        end
        DAE.COMP(__) => begin
          (subVars, subEqs, subInitEqs) = splitEquationsAndVars(elem.dAElist)
          variableLst = listAppend(subVars, variableLst)
          equationLst = listAppend(subEqs, equationLst)
          initialEquationLst = listAppend(subInitEqs, initialEquationLst)
        end
        DAE.NORETCALL(DAE.CALL(Absyn.IDENT("branch"), args)) => begin
          @match arg1 <| arg2 <| nil = args
          equationLst = BDAE.BRANCH(arg1, arg2) <| equationLst
        end
        DAE.RECONFIGURE_EQUATION(__) => begin
          equationLst = lowerReconfigureEquation(elem) <| equationLst
        end
        DAE.ASSERT(c, msg, level, source) => begin
          #= Mirror `equationToBackendEquation`: treat asserts as
             ASSERT_EQUATIONs in the main equation list. Without this
             branch, asserts nested directly under a DAE.COMP (rather than
             inside an inner equation list) hit the catch-all below and
             fail translate with "Unsupported equation: DAE.ASSERT(...)".
             Surfaced by e.g. Modelica.Fluid.Examples.AST_BatchPlant.BatchPlant_StandardWater
             whose "Attempt to fill tank while evaporating" assert lives
             at component scope. =#
          equationLst = BDAE.ASSERT_EQUATION(c, msg, level, source) <| equationLst
        end
        _ => begin
          @error "Skipped:" elem
          throw("Unsupported equation: $elem")
        end
      end
    end
  end
  return (variableLst, equationLst, initialEquationLst)
end

Base.@nospecializeinfer function equationToBackendEquation(@nospecialize(elem::DAE.Element))
  @match elem begin
    DAE.EQUATION(__) => begin
      BDAE.EQUATION(elem.exp,
                    elem.scalar,
                    elem.source,
                    BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
    end
    DAE.WHEN_EQUATION(__) => begin
      lowerWhenEquation(elem)
    end
    DAE.IF_EQUATION(__) => begin
      lowerIfEquation(elem)
    end
    DAE.INITIALEQUATION(__) => begin
      BDAE.EQUATION(elem.exp1,
                    elem.exp2,
                    elem.source,
                    BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
    end
    DAE.COMP(__) => begin
      throw("Components not directly allowed in equation sections")
    end
    DAE.NORETCALL(call, source) where call isa DAE.CALL => begin
      #= A structural state, a transition between structural states, or a
         call for its effects. =#
      local path = call.path
      local expLst = call.expLst
      res = @match path begin
        Absyn.IDENT("initialStructuralState") => begin
          BDAE.INITIAL_STRUCTURAL_STATE(string(listHead(expLst)))
        end
        Absyn.IDENT("structuralTransition") => begin
          @match fromStateExp <| toStateExp <| conditionExp <| nil = expLst
          local fromStateIdent = string(fromStateExp)
          local toStateIdent = string(toStateExp)
          BDAE.STRUCTURAL_TRANSITION(fromStateIdent, toStateIdent, conditionExp)
        end
        #= A call of a Modelica function for its effects (MSL Fluid's
           checkBoundary: asserts on the medium; a print) runs where the
           asserts are checked, after the initialization and after each step
           (emitAssertCallback). As an assert its condition is the call, which
           has no value (T_NORETCALL). It was dropped (a DUMMY_EQUATION). =#
        _ where !call.attr.builtin =>
          BDAE.ASSERT_EQUATION(call, DAE.SCONST(string(path)), DAE.ASSERTIONLEVEL_ERROR, source)
        _ => OMBackend.unsupported("this builtin call as an equation", call)
      end
      res
    end
    DAE.ASSERT(c, msg, level, source) => begin
      BDAE.ASSERT_EQUATION(c, msg, level, source)
    end
    DAE.ARRAY_EQUATION(dim, exp, arr, source) => begin
      dVec = BDAEUtil.DAE_DimensionToIntVector(dim)
      BDAE.ARRAY_EQUATION(dVec, arr, exp, source, BDAE.NO_ATTRIBUTES(), NONE())
    end

    DAE.COMPLEX_EQUATION(lhs, rhs, source) where rhs isa DAE.CALL => begin
      @match DAE.CALL(path, expLst, DAE.CALL_ATTR(ty)) = rhs
      #= We assume the same size and  that the frontend made sure to check it. =#
      dVec = BDAEUtil.getDimensionFromComplexType(ty)
      size = isempty(dVec) ? 1 : prod(dVec)
      BDAE.COMPLEX_EQUATION(size, lhs, rhs, source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
    end
    #= Record-to-record equality: lhs.R = rhs.R where both are CREFs with T_COMPLEX type.
       Decompose into per-field equations using the record type's varLst. =#
    DAE.COMPLEX_EQUATION(lhs, rhs, source) => begin
      decomposeComplexEquation(lhs, rhs, source)
    end
    DAE.RECONFIGURE_EQUATION(__) => begin
      lowerReconfigureEquation(elem)
    end
    _ => begin
      @error "Skipped processing" elem OMFrontend.Frontend.toString(elem)
      throw("Unsupported equation: $elem")
    end
  end
end

"""
  Decompose `lhs = rhs` where lhs is a record-typed expression into per-field
  equations, using the record's fieldList from its T_COMPLEX type.
  For a record with fields T[3,3] and w[3], this emits one ARRAY_EQUATION per
  array field and one EQUATION per scalar field.

  Resolution of recTy:
    1. getComplexType(lhs)   — matches CREF with T_COMPLEX identType, or a
                                CALL whose CALL_ATTR returns T_COMPLEX
    2. getComplexType(rhs)   — symmetric fallback
    3. nothing               — we cannot model this shape; emit a single
                                opaque COMPLEX_EQUATION as a backstop

  Resolution of splittability:
    - If LHS is a CREF or RECORD literal: per-field split via appendFieldToCref
    - Otherwise (CALL / BINARY / IFEXP / ...): emit COMPLEX_EQUATION with
      correct nFields from recTy, do not split
"""
function decomposeComplexEquation(lhs::DAE.Exp, rhs::DAE.Exp, source::DAE.ElementSource)::Vector{BDAE.Equation}
  local eqs = BDAE.Equation[]
  local recTy = BDAEUtil.getComplexType(lhs)
  if recTy === nothing
    recTy = BDAEUtil.getComplexType(rhs)
  end
  if recTy === nothing
    @info "DBG: decomposeComplexEquation: neither LHS nor RHS yields T_COMPLEX; emitting opaque COMPLEX_EQUATION(size=1). This path may be legitimate for BINARY/IFEXP/ASUB record expressions — leaving as info until the envelope of shapes is understood." lhsType=typeof(lhs) rhsType=typeof(rhs) lhsSummary=first(string(lhs), 160) rhsSummary=first(string(rhs), 160) maxlog=5
    push!(eqs, BDAE.COMPLEX_EQUATION(1, lhs, rhs, source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN))
    return eqs
  end
  if !(lhs isa DAE.CREF || lhs isa DAE.RECORD)
    @match DAE.T_COMPLEX(varLst = varLst) = recTy
    local nFields = length(collect(varLst))
    push!(eqs, BDAE.COMPLEX_EQUATION(nFields, lhs, rhs, source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN))
    return eqs
  end
  @match DAE.T_COMPLEX(varLst = varLst) = recTy
  for field in varLst
    @match DAE.TYPES_VAR(name = fieldName, ty = fieldTy) = field
    local lhsField = BDAEUtil.appendFieldToCref(lhs, fieldName, fieldTy)
    local rhsField = BDAEUtil.appendFieldToCref(rhs, fieldName, fieldTy)
    @match fieldTy begin
      DAE.T_ARRAY(dims = dims) => begin
        local dVec = BDAEUtil.DAE_DimensionToIntVector(dims)
        push!(eqs, BDAE.ARRAY_EQUATION(dVec, lhsField, rhsField, source, BDAE.NO_ATTRIBUTES(), NONE()))
      end
      DAE.T_COMPLEX(__) => begin
        #= Nested record: recurse =#
        local nested = decomposeComplexEquation(lhsField, rhsField, source)
        append!(eqs, nested)
      end
      _ => begin
        push!(eqs, BDAE.EQUATION(lhsField, rhsField, source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN))
      end
    end
  end
  return eqs
end

function variableToBackendVariable(elem::DAE.Element)
  @match elem begin
    DAE.VAR(__) => begin
      variableLst = BDAE.VAR(elem.componentRef,
      BDAEUtil.DAE_VarKind_to_BDAE_VarKind(elem.kind),
      elem.direction,
      elem.ty,
      elem.binding,
      elem.dims,
      elem.source,
      _maybeMarkAttrProtected(elem.variableAttributesOption, elem.protection, elem.ty),
      NONE(), #=Tearing=#
      elem.connectorType,
      false #=We do not know if we can replace or not yet=#)
    end
  end
end


function lowerWhenEquation(eq::DAE.WHEN_EQUATION)::BDAE.Equation
  local whenOperatorLst::List{BDAE.WhenOperator} = nil
  local whenEquation::BDAE.WhenEquation
  local elseOption
  local elseEq::DAE.Element
  whenOperatorLst = createWhenOperators(eq.equations, whenOperatorLst)
  #= Check if the list of whenOperators contains a BDAE.RECOMPILATION or BDAE.AGENTIC_RECOMPILATION call. =#
  local containsRecompilation = length(findall(elem->typeof(elem)==BDAE.RECOMPILATION || typeof(elem)==BDAE.AGENTIC_RECOMPILATION, listArray(whenOperatorLst))) >= 1
  elseOption = if isSome(eq.elsewhen_)
    @match SOME(elseEq) = eq.elsewhen_
    bdaeElse = lowerWhenEquation(elseEq)
    SOME(bdaeElse)
  else
    NONE()
  end
  #= initial() outside the initial forms (dropInitialDisjuncts). =#
  local condition = dropInitialDisjuncts(eq.condition)
  whenEquation = if isSome(elseOption)
    BDAE.WHEN_STMTS(condition, whenOperatorLst, elseOption)
  else
    BDAE.WHEN_STMTS(condition, whenOperatorLst, NONE())
  end
  result = if !containsRecompilation
    BDAE.WHEN_EQUATION(1, whenEquation, eq.source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
  else
    BDAE.STRUCTURAL_WHEN_EQUATION(1, whenEquation, eq.source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
  end
  return result
end

"""
  Serialize a List{Absyn.EquationItem} to a human-readable Modelica string.
  Converts assert(cond, msg) calls to readable form for the LLM agent context.
"""
function serializeInitialEquations(eqs::MetaModelica.List)::String
  parts = String[]
  for item in eqs
    s = @match item begin
      Absyn.EQUATIONITEM(__) => begin
        @match item.equation_ begin
          Absyn.EQ_NORETCALL(__) => begin
            fn_name = Absyn.dumpCref(item.equation_.functionName)
            args_str = Absyn.dumpFunctionArgs(item.equation_.functionArgs)
            "$(fn_name)($(args_str))"
          end
          _ => string(item.equation_)
        end
      end
      _ => string(item)
    end
    push!(parts, s)
  end
  return join(parts, "; ")
end

"""
  Extract variable names from a List{Absyn.ElementItem} as used in a reconfigure block.
"""
function extractVariableNames(variables::List{Absyn.ElementItem})::Vector{String}
  names = String[]
  for item in variables
    @match Absyn.ELEMENTITEM(element = Absyn.ELEMENT(
      specification = Absyn.COMPONENTS(components = comps))) = item
    for c in comps
      @match Absyn.COMPONENTITEM(component = Absyn.COMPONENT(name = varName)) = c
      push!(names, varName)
    end
  end
  return names
end

"""
  Lower a DAE.RECONFIGURE_EQUATION into a BDAE.STRUCTURAL_WHEN_EQUATION
  with an AGENTIC_RECOMPILATION when-operator.
"""
function lowerReconfigureEquation(eq::DAE.RECONFIGURE_EQUATION)::BDAE.Equation
  varNames = extractVariableNames(eq.variables)
  #= Type is a placeholder: downstream consumers (structuralCallbacks.jl) only
     use the name via `string(c)`. Using T_UNKNOWN_DEFAULT avoids falsely
     claiming Real for variables that may be Integer/Boolean/String. =#
  crefs = DAE.CREF[
    DAE.CREF(DAE.CREF_IDENT(name, DAE.T_UNKNOWN_DEFAULT, nil), DAE.T_UNKNOWN_DEFAULT)
    for name in varNames
  ]
  promptStr = if isSome(eq.prompt)
    @match SOME(DAE.SCONST(s)) = eq.prompt
    SOME(s)
  else
    NONE()
  end
  initEqStr = if isSome(eq.initialEquations)
    @match SOME(eqs) = eq.initialEquations
    SOME(serializeInitialEquations(eqs))
  else
    NONE()
  end
  agenticOp = BDAE.AGENTIC_RECOMPILATION(crefs, promptStr, initEqStr)
  whenStmts = BDAE.WHEN_STMTS(eq.whenCondition, list(agenticOp), NONE())
  return BDAE.STRUCTURAL_WHEN_EQUATION(1, whenStmts, eq.source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
end

function createWhenOperators(elementLst::List{DAE.Element},lst::List{BDAE.WhenOperator})::List{BDAE.WhenOperator}
  lst = begin
    local rest::List{DAE.Element}
    local acc::List{BDAE.WhenOperator}
    local cref::DAE.ComponentRef
    local e1::DAE.Exp
    local e2::DAE.Exp
    local e3::DAE.Exp
    local source::DAE.ElementSource
    @match elementLst begin
      DAE.EQUATION(exp = e1, scalar = e2, source = source) <| rest => begin
        acc = BDAE.ASSIGN(e1, e2, source) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.ASSERT(condition = e1, message = e2, level = e3, source = source) <| rest => begin
        acc = BDAE.ASSERT(e1, e2, e3, source) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.TERMINATE(message = e1, source = source) <| rest => begin
        acc = BDAE.TERMINATE(e1, source) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.REINIT(componentRef = cref, exp = e1, source = source) <| rest => begin
        #= BDAE uses an exp here instead of a cref =#
        expTy = if typeof(cref.identType) == DAE.T_ARRAY
          #= If we are referring to an array it is the content of the array that is the type of the exp. =#
          cref.identType.ty #= Note this would be wrong if we would consider other compound types. =#
        else
          cref.identType #=OK it is the type of the component reference directly=#
        end
        local crefExp = DAE.CREF(cref, expTy)
        acc = BDAE.REINIT(crefExp, e1, source) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.NORETCALL(exp = DAE.CALL(Absyn.IDENT("recompilation"), expLst, attr), source = source) <| rest => begin
        @match componentToChange <| newValue <| nil = expLst
        acc = BDAE.RECOMPILATION(componentToChange, newValue) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.NORETCALL(exp = DAE.CALL(Absyn.IDENT("agentic_recompilation"), expLst, attr), source = source) <| rest => begin
        componentsToChange = DAE.CREF[cref for cref in expLst]
        acc = BDAE.AGENTIC_RECOMPILATION(componentsToChange, NONE(), NONE()) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.NORETCALL(exp = e1, source = source) <| rest => begin
        acc = BDAE.NORETCALL(e1, source) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.EQUEQUATION(cr1 = c1, cr2 = c2, source = source) <| rest => begin
        acc = BDAE.ASSIGN(DAE.CREF(c1, BDAEUtil.crefLeafType(c1)), DAE.CREF(c2, BDAEUtil.crefLeafType(c2)), source) <| lst
        createWhenOperators(rest, acc)
      end
      DAE.IF_EQUATION(condition1 = conds, equations2 = branches, equations3 = elseBranch, source = source) <| rest => begin
        createWhenOperators(rest, _ifEquationWhenOperators(conds, branches, elseBranch, source, lst))
      end
      nil => begin
        (lst)
      end
      #= An array, record or for equation in a when (ARRAY_EQUATION,
         COMPLEX_EQUATION, FOR_EQUATION): dropping it left its variables
         unassigned at the event. =#
      e <| _ => OMBackend.unsupported("this equation in a when-equation", e)
    end
  end
end

#= An if-equation in a when-equation (MLS 8.3.5: every branch assigns the same
   variables): one assignment per variable of `if c1 then e1 elseif ... else en`,
   a branch without one keeping the value; a reinit likewise, a branch without
   one reinitializing the state to itself; an assert under the branch's
   condition. A call for its side effects is not run (as in a when algorithm). =#
function _ifEquationWhenOperators(conds, branches, elseBranch, source, lst::List{BDAE.WhenOperator})::List{BDAE.WhenOperator}
  local branchOps = [listArray(createWhenOperators(b, nil)) for b in vcat(collect(branches), [elseBranch])]
  local guards = Any[]
  local before = nothing
  for c in conds
    push!(guards, before === nothing ? c : DAE.LBINARY(before, DAE.AND(DAE.T_BOOL_DEFAULT), c))
    local notC = DAE.LUNARY(DAE.NOT(DAE.T_BOOL_DEFAULT), c)
    before = before === nothing ? notC : DAE.LBINARY(before, DAE.AND(DAE.T_BOOL_DEFAULT), notC)
  end
  push!(guards, before === nothing ? DAE.BCONST(true) : before)
  #= The value of each assigned variable (or reinitialized state) per branch. =#
  local targets = OrderedDict{String, Tuple{Symbol, DAE.Exp}}()
  local values = [Dict{String, DAE.Exp}() for _ in branchOps]
  local asserts = BDAE.WhenOperator[]
  for (k, ops) in enumerate(branchOps)
    for op in ops
      @match op begin
        BDAE.ASSIGN(left, right, _) => begin
          targets[string(left)] = (:assign, left)
          values[k][string(left)] = right
        end
        BDAE.REINIT(stateVar, value, _) => begin
          targets["reinit " * string(stateVar)] = (:reinit, stateVar)
          values[k]["reinit " * string(stateVar)] = value
        end
        BDAE.ASSERT(c, m, l, s) => push!(asserts, BDAE.ASSERT(DAE.LBINARY(DAE.LUNARY(DAE.NOT(DAE.T_BOOL_DEFAULT), guards[k]),
                                                                           DAE.OR(DAE.T_BOOL_DEFAULT), c), m, l, s))
        BDAE.NORETCALL(__) => nothing
        _ => OMBackend.unsupported("this equation in an if-equation in a when-equation", op)
      end
    end
  end
  local ifOps = BDAE.WhenOperator[]
  for (key, (kind, target)) in targets
    local value = get(values[end], key, target)
    for k in (length(branchOps) - 1):-1:1
      value = DAE.IFEXP(collect(conds)[k], get(values[k], key, target), value)
    end
    push!(ifOps, kind === :assign ? BDAE.ASSIGN(target, value, source) : BDAE.REINIT(target, value, source))
  end
  #= In the order written, the asserts after the assignments (they check the
     new values); `lst` is built back to front. =#
  for op in Iterators.reverse(vcat(ifOps, asserts))
    lst = op <| lst
  end
  return lst
end

"""
  Transform a DAE if-equation into a BDAE if-equation
"""
function lowerIfEquation(eq::IF_EQ) where {IF_EQ}
  local trueEquations::List{List{BDAE.Equation}} = nil
  local tmpTrue::List{BDAE.Equation}
  local falseEquations::List
  for lst in eq.equations2
    (_, tmpTrue, _) = splitEquationsAndVars(lst)
    trueEquations = tmpTrue <| trueEquations
  end
  (_, falseEquations, _) = splitEquationsAndVars(eq.equations3)
  res = BDAE.IF_EQUATION(eq.condition1,
                         listReverse(trueEquations),
                         listReverse(falseEquations), #Should not really matter but I reverse just in case.
                         eq.source,
                         BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
  return res
end

"""
```
createBindingEquations(variables::Vector)
```
Create the equation from the binding equations.
See: https://specification.modelica.org/master/equations.html
TODO:
 - Add discrete binding equations in some other pile.
"""
function createBindingEquations(variables::Vector)
  bindingEqs = BDAE.Equation[]
  for v in variables
    @match v begin
      #= Real continuous declaration binding -> defining equation. =#
      BDAE.VAR(vName, BDAE.STATE() || BDAE.VARIABLE(), _, DAE.T_REAL(__),
               SOME(bindExp), _, _, _, _, _) => begin
                 local lhs = DAE.CREF(vName, v.varType)
                 local rhs = bindExp
                 local eq =  BDAE.EQUATION(lhs, rhs, v.source, BDAE.NO_ATTRIBUTES())
                 push!(bindingEqs, eq)
               end
      #= Boolean declaration binding given as an if-expression -> when/elsewhen
         equation (the value updates at the branch condition's events). =#
      BDAE.VAR(vName, BDAE.STATE() || BDAE.VARIABLE() || BDAE.DISCRETE(), _, DAE.T_BOOL(__),
               SOME(bindExp), _, _, _, _, _) where (bindExp isa DAE.IFEXP) => begin
                 local lhs = DAE.CREF(vName, v.varType)
                 @match DAE.IFEXP(cond, thenExp, elseExp) = bindExp
                 local elsePart = BDAE.WHEN_STMTS(BDAEUtil.invertCondition(cond) #= The else when here has the inverted condition of the first part. =#
                                                  ,list(BDAE.ASSIGN(lhs, elseExp, v.source))
                                                  ,nothing)
                 local elseWeqPart = BDAE.WHEN_EQUATION(1, elsePart, v.source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
                 local stmts = BDAE.WHEN_STMTS(cond
                                               ,list(BDAE.ASSIGN(lhs, thenExp, v.source))
                                               ,SOME(elseWeqPart))
                 local weq = BDAE.WHEN_EQUATION(1, stmts, v.source, BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
                 push!(bindingEqs, weq)
               end
      #= Boolean (non-ifexp), Integer or enumeration declaration binding ->
         plain defining equation. The downstream discrete classification and
         when-synthesis passes lift `disc = expr` into a when-equation when the
         RHS is discrete-time. Without this, such a binding was silently dropped
         (hitting the catch-all below), leaving the variable under-determined. =#
      BDAE.VAR(vName, BDAE.STATE() || BDAE.VARIABLE() || BDAE.DISCRETE(), _,
               DAE.T_BOOL(__) || DAE.T_INTEGER(__) || DAE.T_ENUMERATION(__),
               SOME(bindExp), _, _, _, _, _) => begin
                 local lhs = DAE.CREF(vName, v.varType)
                 local eq =  BDAE.EQUATION(lhs, bindExp, v.source, BDAE.NO_ATTRIBUTES())
                 push!(bindingEqs, eq)
               end
      #= A discrete Real bound to an expression: not Modelica (omc: "variable is
         discrete, but does not appear on the LHS of a when-statement"); dropped
         before. =#
      BDAE.VAR(vName, BDAE.DISCRETE(), _, DAE.T_REAL(__), SOME(_), _, _, _, _, _) =>
        OMBackend.unsupported("a discrete Real variable with a declaration binding", vName)
      #= Parameters and constants (their bindings are their values), variables
         without a binding, Strings (createStringParameterAssignments); array and
         record bindings are equations already (the frontend's flattening). =#
      _ => continue
    end
  end
  return bindingEqs
end

#= Convert a list of DAE.Statement (from converting an ALG_WHEN or
   `initial algorithm` body) into BDAE.WhenOperator entries. Compound
   statements (FOR, IF, WHILE) are flattened by recursing into their bodies;
   control-flow structure is preserved at the codegen layer by re-grouping in
   __runInitialAlgorithm!'s lowering. Unsupported variants are skipped. =#
function _daeStmtsToWhenOps(daeStmts)::List
  local ops = BDAE.WhenOperator[]
  _appendStmtsToOps!(ops, daeStmts)
  return list(ops...)
end

function _appendStmtsToOps!(ops::Vector, daeStmts)
  for s in daeStmts
    @match s begin
      DAE.STMT_ASSIGN(_, e1, e, src) => push!(ops, BDAE.ASSIGN(e1, e, src))
      #= (a, b, ...) := f(...): each target its element of f's result. These
         ops name what the algorithm assigns; the generated code runs the
         algorithm's DAE statements (BDAEUtil.INIT_ALG_DAE_STMTS), which call f once. =#
      DAE.STMT_TUPLE_ASSIGN(_, targets, e, src) => begin
        for (k, target) in enumerate(targets)
          (target isa DAE.CREF && !(target.componentRef isa DAE.WILD)) || continue
          push!(ops, BDAE.ASSIGN(target, DAE.TSUB(e, k, target.ty), src))
        end
      end
      DAE.STMT_NORETCALL(exp, src) => push!(ops, BDAE.NORETCALL(exp, src))
      DAE.STMT_ASSERT(c, m, l, src) => push!(ops, BDAE.ASSERT(c, m, l, src))
      DAE.STMT_TERMINATE(m, src) => push!(ops, BDAE.TERMINATE(m, src))
      DAE.STMT_ASSIGN_ARR(_, lhs, e, src) => push!(ops, BDAE.ASSIGN(lhs, e, src))
      DAE.STMT_REINIT(varExp, value, src) => varExp isa DAE.CREF && push!(ops, BDAE.REINIT(varExp, value, src))
      #= The bodies of a FOR / PARFOR / IF (every branch) / WHILE, flattened:
         the ops name what the algorithm assigns; the generated code runs its
         DAE statements (BDAEUtil.INIT_ALG_DAE_STMTS), which keep the structure. =#
      DAE.STMT_FOR(_, _, _, _, _, body, _) => _appendStmtsToOps!(ops, body)
      DAE.STMT_PARFOR(_, _, _, _, _, body, _, _) => _appendStmtsToOps!(ops, body)
      DAE.STMT_IF(_, tb, else_, _) => (_appendStmtsToOps!(ops, tb); _appendElseStmtsToOps!(ops, else_))
      DAE.STMT_WHILE(_, body, _) => _appendStmtsToOps!(ops, body)
      #= return, break, continue: nothing assigned. =#
      _ => nothing
    end
  end
end

function _appendElseStmtsToOps!(ops::Vector, else_)
  @match else_ begin
    DAE.ELSEIF(_, stmts, rest) => (_appendStmtsToOps!(ops, stmts); _appendElseStmtsToOps!(ops, rest))
    DAE.ELSE(stmts) => _appendStmtsToOps!(ops, stmts)
    _ => nothing
  end
end

#= The synthesis passes: residuals, whens, asserts and initial whens from
   algorithm sections, and the lifting of discrete equations into when-clusters. =#
include("algorithmSynthesis.jl")
include("discreteSynthesis.jl")

@exportAll()
end
