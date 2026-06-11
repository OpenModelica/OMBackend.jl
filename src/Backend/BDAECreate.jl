#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
=#
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
import Absyn
import DAE
import OMFrontend

#= Side-channel from `synthesizeFromInitialAlgorithms` /
   `synthesizeInitialWhenFromAlgorithms` to `SimulationCode`'s
   `INITIAL_ALGORITHM` construction. Keyed by the produced
   `BDAE.INITIAL_WHEN_EQUATION` wrapper (object identity), value is the
   original DAE.Statement list before it was flattened to a WhenOperator list
   by `_daeStmtsToWhenOps`. Cleared at the start of each `createEqSystem`
   call so the dict tracks only the current model's init-algorithm bodies.
   `IdDict` because the keys are mutable Julia structs whose equality is
   identity-based at this layer. =#
const _INIT_ALG_DAE_STMTS = IdDict{Any, Vector{DAE.Statement}}()

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
  outBDAE = BDAE.BACKEND_DAE(name, eqSystems, BDAE.SHARED(BDAE.VAR[], BDAE.VAR[], NONE(), NONE(), BDAE.Equation[]))
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
  local shared
  if ! listEmpty(flatModelica.DOCC_equations)
    shared = BDAE.SHARED(BDAE.VAR[],
                         BDAE.VAR[],
                         flatModelica.scodeProgram,
                         SOME(flatModelica),
                         createStructuralIfEquations(flatModelica.DOCC_equations))
  else
    shared = BDAE.SHARED(BDAE.VAR[],
                         BDAE.VAR[],
                         flatModelica.scodeProgram,
                         NONE(),
                         BDAE.Equation[])
  end
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
  empty!(_INIT_ALG_DAE_STMTS)
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
     but ONLY for LHSes that are not already constrained by an equation.
     The LHS-collision guard skips connect-driven LHSes (e.g. INV3S's
     `yy := nextstate;` where `yy` is also bound by
     `connect(yy, inertialDelaySensitive.x)`), which would otherwise
     over-determine MTK's structural-simplify. Models where the algorithm
     LHS has no competing equation (the reproducer
     `Models/AlgorithmDiscreteAssign.mo`) take the lift and gain a defining
     residual. =#
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
      for ieq in synthesizeResidualsFromRegularAlgorithms(algorithms, _eqLhsBoundCrefs, _whenLifterSkipLhs)
        push!(equations, ieq)
      end
    end
  end
  #= §17.4.4: lift discrete (Bool/Int/enum) equation-section definitions whose RHS
     is a discrete-time relation into event-driven held discretes, so the
     continuous integrator never interpolates step-valued logic. =#
  if !isempty(equations) && !isempty(variables)
    local _discParamConst = _collectParamOrConstNames(variables)
    local _discStarts = _discreteStartExpLookup(variables)
    local (_discEqs, _discLifted) = synthesizeWhenEquationsFromDiscreteEquations(equations, _discParamConst, _discStarts)
    if !isempty(_discLifted)
      @info "[BDAE: lifter] synthesizeWhenEquationsFromDiscreteEquations lifted $(length(_discLifted)) discrete equation(s)" lifted=collect(_discLifted)
    end
    equations = _discEqs
  end
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
  resolveMangledNameCollisions!(variables, equations, initialEquations)
  #= Deduplicate variables by name (handles inner/outer duplicate emission) =#
  variables = deduplicateVariables(variables)
  #= Deduplicate explicit equations =#
  equations = deduplicateEquations(equations)
  #= The set of equations might also contain a  set of "binding equations" =#
  local bindingEquations = createBindingEquations(variables)
  equations = vcat(equations, bindingEquations)
  #= TODO Extract the simple equations =#
  local simpleEquations = BDAE.Equation[]
  return BDAE.EQSYSTEM(name, variables, equations, simpleEquations, initialEquations)
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
function resolveMangledNameCollisions!(variables::Vector, equations::Vector, initialEquations::Vector)
  local idxsByMangled = OrderedDict{String, Vector{Int}}()
  for (i, v) in enumerate(variables)
    push!(get!(() -> Int[], idxsByMangled, string(v.varName)), i)
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
  for i in 1:length(equations)
    (equations[i], _) = BDAEUtil.traverseEquationExpressions(equations[i], rewrite, 0)
  end
  for i in 1:length(initialEquations)
    (initialEquations[i], _) = BDAEUtil.traverseEquationExpressions(initialEquations[i], rewrite, 0)
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
function deduplicateVariables(variables::Vector)::Vector
  local idxByName = Dict{String, Int}()
  local unique_vars = similar(variables, 0)
  local duplicateCount = 0
  for v in variables
    local varStr = string(v.varName)
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
function structuralHash(@nospecialize(x), h::UInt)
  T = typeof(x)
  h = hash(T, h)
  for i in 1:fieldcount(T)
    h = structuralHash(getfield(x, i), h)
  end
  h
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

function convertVariableIntoBDAEVariable(var::OMFrontend.Frontend.Variable)
  elem = OMFrontend.Frontend.convertVariable(var, OMFrontend.Frontend.VARIABLE_CONVERSION_SETTINGS(true, false, true))
  BDAE.VAR(elem.componentRef,
           BDAEUtil.DAE_VarKind_to_BDAE_VarKind(elem.kind),
           elem.direction,
           elem.ty,
           elem.binding,
           elem.dims,
           elem.source,
           _maybeMarkAttrProtected(elem.variableAttributesOption, elem.protection),
           NONE(), #=Tearing=#
           elem.connectorType,
           false #=We do not know if we can replace or not yet=#
           )
end

#= Carry the DAE.VAR `protection` flag onto the variable attribute Option so
   the SimCode-layer `dropObservationOnlyVariables` pass can pick it up. =#
function _maybeMarkAttrProtected(vattr, protection)
  @match protection begin
    DAE.PROTECTED(__) => _markAttrProtected(vattr)
    _ => vattr
  end
end

function _markAttrProtected(vattr)
  local protOpt = SOME(true)
  @match vattr begin
    SOME(va) where va isa DAE.VAR_ATTR_REAL => SOME(DAE.VAR_ATTR_REAL(
        va.quantity, va.unit, va.displayUnit, va.min, va.max, va.start,
        va.fixed, va.nominal, va.stateSelectOption, va.uncertainOption,
        va.distributionOption, va.equationBound, protOpt, va.finalPrefix,
        va.startOrigin))
    SOME(va) where va isa DAE.VAR_ATTR_INT => SOME(DAE.VAR_ATTR_INT(
        va.quantity, va.min, va.max, va.start, va.fixed, va.uncertainOption,
        va.distributionOption, va.equationBound, protOpt, va.finalPrefix,
        va.startOrigin))
    SOME(va) where va isa DAE.VAR_ATTR_BOOL => SOME(DAE.VAR_ATTR_BOOL(
        va.quantity, va.start, va.fixed, va.equationBound, protOpt,
        va.finalPrefix, va.startOrigin))
    _ => SOME(DAE.VAR_ATTR_REAL(NONE(), NONE(), NONE(), NONE(), NONE(),
        NONE(), NONE(), NONE(), NONE(), NONE(), NONE(), NONE(), protOpt,
        NONE(), NONE()))
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
          _maybeMarkAttrProtected(elem.variableAttributesOption, elem.protection),
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
    DAE.NORETCALL(DAE.CALL(path, expLst)) => begin
      #=
      Currently there are two options here.
      Either we have an initialStructuralState
      or we have some transition between structural states.
      =#
      res = @match path begin
        Absyn.IDENT("initialStructuralState") => begin
          BDAE.INITIAL_STRUCTURAL_STATE(string(listHead(expLst)))
        end
        Absyn.IDENT("structuralTransition") => begin
          @match fromStateExp <| toStateExp <| conditionExp <| nil = expLst
          local fromStateIdent = string(fromStateExp)
          local toStateIdent = string(toStateExp)
          BDAE.STRUCTURAL_TRANSISTION(fromStateIdent, toStateIdent, conditionExp)
        end
        _ => begin
          #= Skip unknown NORETCALL statements (e.g. checkBoundary, assert-like calls).
             These are validation calls that do not contribute to the equation system.
             Drop `maxlog` so every unique skipped call is visible — silently discarding
             unknown semantics is how real bugs hide. =#
          @warn "Skipping unknown NORETCALL (frontend emitted a call whose semantics the backend does not handle; treating as dummy). If this call has side effects or constraints, it will not be preserved." path
          BDAE.DUMMY_EQUATION()
        end
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
      _maybeMarkAttrProtected(elem.variableAttributesOption, elem.protection),
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
  whenEquation = if isSome(elseOption)
    BDAE.WHEN_STMTS(eq.condition, whenOperatorLst, elseOption)
  else
    BDAE.WHEN_STMTS(eq.condition, whenOperatorLst, NONE())
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
      #= MAYBE MORE CASES NEEDED =#
      nil => begin
        (lst)
      end
      _ <| rest => begin
        createWhenOperators(rest, lst)
      end
    end
  end
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
  #= Check if this equation contains an Connections.branch call. DOCC case=#
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
                 local elseWeqPart = BDAE.WHEN_EQUATION(1, elsePart, v.source, nothing)
                 local stmts = BDAE.WHEN_STMTS(cond
                                               ,list(BDAE.ASSIGN(lhs, thenExp, v.source))
                                               ,SOME(elseWeqPart))
                 local weq = BDAE.WHEN_EQUATION(1, stmts, v.source, nothing)
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
      _ => begin
        continue
      end
    end
  end
  return bindingEqs
end

"""
  Wraps the special if equation in a BDAE construct.
"""
function createStructuralIfEquations(ifEquations::List)
  eqs = BDAE.Equation[]
  for ifEq in ifEquations
    push!(eqs, BDAE.STRUCTURAL_IF_EQUATION(ifEq))
  end
  return eqs
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
      DAE.STMT_NORETCALL(exp, src) => push!(ops, BDAE.NORETCALL(exp, src))
      DAE.STMT_ASSERT(c, m, l, src) => push!(ops, BDAE.ASSERT(c, m, l, src))
      DAE.STMT_TERMINATE(m, src) => push!(ops, BDAE.TERMINATE(m, src))
      DAE.STMT_REINIT(varExp, value, src) => begin
        @match varExp begin
          DAE.CREF(cr, _) => push!(ops, BDAE.REINIT(cr, value, src))
          _ => nothing
        end
      end
      #= Flatten the body of a FOR / IF / WHILE into the same op list. This is
         a simplification — sequential semantics of the body are preserved
         (operators run in order) but the loop/branch structure is lost. For
         `initial algorithm` bodies that only contain straight-line code over
         scalar variables this is adequate; bodies that depend on iteration
         variables (DAE.STMT_FOR) need full lowering, which can be added later. =#
      DAE.STMT_FOR(_, _, _, _, _, body, _) => _appendStmtsToOps!(ops, body)
      DAE.STMT_IF(_, tb, _, _) => _appendStmtsToOps!(ops, tb)
      DAE.STMT_WHILE(_, body, _) => _appendStmtsToOps!(ops, body)
      _ => nothing
    end
  end
end

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
    local whenOps = _daeStmtsToWhenOps(daeStmts)
    isempty(whenOps) && continue
    local initialCall = DAE.CALL(Absyn.IDENT("initial"),
                                 MetaModelica.list(),
                                 DAE.callAttrBuiltinBool)
    local node = BDAE.INITIAL_WHEN_EQUATION(
      length(alg.statements),
      BDAE.WHEN_STMTS(initialCall, whenOps, NONE()),
      alg.source,
      nothing,
    )
    _INIT_ALG_DAE_STMTS[node] = collect(daeStmts)
    push!(out, node)
  end
  return out
end

"""
    synthesizeInitialWhenFromAlgorithms(algorithms) -> Vector{BDAE.Equation}

Scan flat-model algorithm sections for `algorithm when initial() then ... end when`
statements and lift each into a `BDAE.INITIAL_WHEN_EQUATION`. Bodies are translated
via OMFrontend's existing Statement → DAE.Statement conversion, then mapped to
BDAE.WhenOperator entries. Compound conditions (e.g. `when (initial() or c)`) are
intentionally skipped per Modelica spec §8/§11.
"""
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
function _collectAllCrefsInEquations(equations)::OrderedSet{String}
  local names = OrderedSet{String}()
  for eq in equations
    if eq isa BDAE.EQUATION
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
function _collectParamOrConstNames(variables::Vector{BDAE.VAR})::OrderedSet{String}
  local names = OrderedSet{String}()
  sizehint!(names, 2 * length(variables))
  for var in variables
    local k = var.varKind
    if k isa BDAE.PARAM || k isa BDAE.CONST
      local s = string(var.varName)
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
            local _rhsK::DAE.Exp = DAE.ASUB(rhs, MetaModelica.list(DAE.ICONST(_k)))
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
                                                  whenLifterSkipLhs::OrderedSet{String} = OrderedSet{String}())::Vector{BDAE.Equation}
  local out = BDAE.Equation[]
  for alg in algorithms
    #= Skip whole algorithm if every top-level statement is ALG_WHEN — those are
       already lifted to (INITIAL_)WHEN_EQUATION by the companion synth pass. =#
    local hasNonWhen = false
    for stmt in alg.statements
      if !(stmt isa OMFrontend.Frontend.ALG_WHEN)
        hasNonWhen = true
        break
      end
    end
    hasNonWhen || continue
    local daeStmts = try
      OMFrontend.Frontend.convertStatements(alg.statements)
    catch
      continue
    end
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

#= Collect every CREF occurring inside a list of DAE.Statement bodies (across
   RHS of STMT_ASSIGN / STMT_ASSIGN_ARR and inside nested STMT_IF / STMT_FOR
   / STMT_WHILE). Used by the when-equation lifter to build the `change(...)`
   trigger condition. =#
Base.@nospecializeinfer function _pushExpRhsCrefs!(out::OrderedSet{Tuple{DAE.ComponentRef, DAE.Type}}, @nospecialize(exp))
  exp === nothing && return nothing
  for c in Util.getAllCrefs(exp)
    local ty = _crefType(c)
    ty === nothing && continue
    push!(out, (c, ty))
  end
  return nothing
end

Base.@nospecializeinfer function _walkRhsCrefsInDAEStmts!(out::OrderedSet{Tuple{DAE.ComponentRef, DAE.Type}}, @nospecialize(stmts))
  for s in stmts
    @match s begin
      DAE.STMT_ASSIGN(_, _, rhs, _) => _pushExpRhsCrefs!(out, rhs)
      DAE.STMT_ASSIGN_ARR(_, _, rhs, _) => _pushExpRhsCrefs!(out, rhs)
      DAE.STMT_IF(cond, body, els, _) => begin
        _pushExpRhsCrefs!(out, cond)
        _walkRhsCrefsInDAEStmts!(out, body)
      end
      DAE.STMT_FOR(_, _, _, _, _, body, _) => _walkRhsCrefsInDAEStmts!(out, body)
      DAE.STMT_WHILE(cond, body, _) => begin
        _pushExpRhsCrefs!(out, cond); _walkRhsCrefsInDAEStmts!(out, body)
      end
      _ => nothing
    end
  end
  return nothing
end

function _collectRhsCrefsInDAEStmts(daeStmts)::OrderedSet{Tuple{DAE.ComponentRef, DAE.Type}}
  local out = OrderedSet{Tuple{DAE.ComponentRef, DAE.Type}}()
  _walkRhsCrefsInDAEStmts!(out, daeStmts)
  return out
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

#= True if all top-level STMT_ASSIGN / STMT_ASSIGN_ARR LHSes in `daeStmts`
   have a discrete Modelica type (Integer / Boolean / enumeration, or arrays
   thereof). Statements that are not assignments contribute nothing. Per
   Modelica spec §17.4.4 this is the gating condition for lifting the
   algorithm body into a when-equation; algorithms with any Real LHS keep
   continuous semantics and are routed to the residual lifter (if at all). =#
function _allAssignsDiscreteLhs(daeStmts)::Bool
  local sawAssign = false
  for s in daeStmts
    @match s begin
      DAE.STMT_ASSIGN(ty, _, _, _) => begin
        sawAssign = true
        _isDiscreteDAEType(ty) || return false
      end
      DAE.STMT_ASSIGN_ARR(ty, _, _, _) => begin
        sawAssign = true
        _isDiscreteDAEType(ty) || return false
      end
      _ => nothing
    end
  end
  return sawAssign
end

#= Build the `BDAE.WhenOperator` list from a list of `DAE.Statement` items
   that come from a non-when `algorithm` body. Reuses `_daeStmtsToWhenOps`
   for the underlying assignment / noretcall / reinit lowering. =#
function _algStmtsToWhenOpsDiscrete(daeStmts)
  return _daeStmtsToWhenOps(daeStmts)
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
"""
function synthesizeWhenEquationsFromRegularAlgorithms(algorithms,
                                                      paramOrConstNames::OrderedSet{String} = OrderedSet{String}())
  local out = BDAE.Equation[]
  local liftedLhsNames = OrderedSet{String}()
  for alg in algorithms
    local statements = alg.statements
    isempty(statements) && continue
    #= Skip whole algorithm if every top-level statement is ALG_WHEN — those are
       already lifted to (INITIAL_)WHEN_EQUATION by the companion synth pass. =#
    local hasNonWhen = false
    for stmt in statements
      if !(stmt isa OMFrontend.Frontend.ALG_WHEN)
        hasNonWhen = true
        break
      end
    end
    hasNonWhen || continue
    local daeStmts = try
      OMFrontend.Frontend.convertStatements(statements)
    catch
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
      if s isa DAE.STMT_WHEN
        _liftStmtWhenToWhenEquations!(out, s, liftedLhsNames)
      end
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

Base.@nospecializeinfer function _innermostSubscripts(@nospecialize(cref))::Vector
  @match cref begin
    DAE.CREF_IDENT(_, _, subs) => collect(subs)
    DAE.CREF_QUAL(_, _, _, cr) => _innermostSubscripts(cr)
    _ => Any[]
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

Base.@nospecializeinfer function _iconstExpList(vals::Vector{Int})
  local exps = DAE.Exp[DAE.ICONST(v) for v in vals]
  return MetaModelica.list(exps...)
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
  isempty(dims) && return Any[(lhs, nothing, Int[])]
  local rawDims = _rawArrayDims(baseTy)

  local subs = _innermostSubscripts(cr)
  if isempty(subs)
    subs = Any[DAE.WHOLEDIM() for _ in dims]
  elseif length(subs) < length(dims)
    append!(subs, Any[DAE.WHOLEDIM() for _ in 1:(length(dims) - length(subs))])
  end
  length(subs) == length(dims) || return Any[]

  local elemTy = _arrayElementType(baseTy)
  local out = Any[]
  function rec(pos::Int, newSubs::Vector, guard, rhsIdxs::Vector{Int})
    if pos > length(dims)
      local newCr = _replaceInnermostSubscripts(cr, newSubs)
      push!(out, (DAE.CREF(newCr, elemTy), guard, copy(rhsIdxs)))
      return
    end
    local sub = subs[pos]
    if sub isa DAE.WHOLEDIM
      for k in 1:dims[pos]
        rec(pos + 1, Any[newSubs...; DAE.INDEX(_dimIndexExp(rawDims[pos], k))],
            guard, Int[rhsIdxs...; k])
      end
    elseif sub isa DAE.INDEX
      local idx = sub.exp
      if idx isa DAE.ICONST || idx isa DAE.ENUM_LITERAL
        rec(pos + 1, Any[newSubs...; DAE.INDEX(idx)], guard, rhsIdxs)
      else
        for k in 1:dims[pos]
          local g = _andCondition(guard, _indexEqualsCondition(idx, k))
          rec(pos + 1, Any[newSubs...; DAE.INDEX(_dimIndexExp(rawDims[pos], k))], g, rhsIdxs)
        end
      end
    else
      return
    end
  end
  rec(1, Any[], nothing, Int[])
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
    subs = Any[DAE.WHOLEDIM() for _ in dims]
  elseif length(subs) < length(dims)
    append!(subs, Any[DAE.WHOLEDIM() for _ in 1:(length(dims) - length(subs))])
  end
  length(subs) == length(dims) || return fallback

  local elemTy = _arrayElementType(baseTy)
  local candidates = Any[]
  function rec(pos::Int, rhsPos::Int, newSubs::Vector, guard)
    if pos > length(dims)
      local newCr = _replaceInnermostSubscripts(cr, newSubs)
      push!(candidates, (guard, DAE.CREF(newCr, elemTy)))
      return
    end
    local sub = subs[pos]
    if sub isa DAE.WHOLEDIM
      rhsPos <= length(rhsIdxs) || return
      local k = rhsIdxs[rhsPos]
      rec(pos + 1, rhsPos + 1, Any[newSubs...; DAE.INDEX(_dimIndexExp(rawDims[pos], k))], guard)
    elseif sub isa DAE.INDEX
      local idx = sub.exp
      if idx isa DAE.ICONST || idx isa DAE.ENUM_LITERAL
        rec(pos + 1, rhsPos, Any[newSubs...; DAE.INDEX(idx)], guard)
      else
        for k in 1:dims[pos]
          local g = _andCondition(guard, _indexEqualsCondition(idx, k))
          rec(pos + 1, rhsPos, Any[newSubs...; DAE.INDEX(_dimIndexExp(rawDims[pos], k))], g)
        end
      end
    else
      return
    end
  end
  rec(1, 1, Any[], nothing)
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
    return DAE.ASUB(rhs, _iconstExpList(rhsIdxs))
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
                                                          allowContinuous::Bool = false)::Bool
  if elsePart isa DAE.NOELSE
    return true
  elseif elsePart isa DAE.ELSE
    return _appendAlgorithmStmtOps!(ops, liftedLhsNames, elsePart.statementLst,
                                    activeCond, iterVals, initialValue, allowContinuous)
  elseif elsePart isa DAE.ELSEIF
    local cond = _prepareAlgorithmExp(elsePart.exp, iterVals, initialValue)
    local branchCond = _andCondition(activeCond, cond)
    _appendAlgorithmStmtOps!(ops, liftedLhsNames, elsePart.statementLst,
                             branchCond, iterVals, initialValue, allowContinuous) || return false
    local restCond = _andCondition(activeCond, _notCondition(cond))
    return _appendElseAlgorithmOps!(ops, liftedLhsNames, elsePart.else_,
                                    restCond, iterVals, initialValue, allowContinuous)
  end
  return true
end

Base.@nospecializeinfer function _appendAlgorithmStmtOps!(ops::Vector{BDAE.WhenOperator},
                                                          liftedLhsNames::OrderedSet{String},
                                                          @nospecialize(stmts),
                                                          @nospecialize(activeCond),
                                                          iterVals::Dict{String, Int},
                                                          initialValue::Bool,
                                                          allowContinuous::Bool = false)::Bool
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
                                 iterVals, initialValue, allowContinuous) || return false
        _appendElseAlgorithmOps!(ops, liftedLhsNames, elsePart,
                                 _andCondition(activeCond, _notCondition(c)),
                                 iterVals, initialValue, allowContinuous) || return false
      end
      DAE.STMT_FOR(_, _, iter, _, range, body, _) => begin
        local r = _prepareAlgorithmExp(range, iterVals, initialValue)
        local vals = _rangeIntValues(r)
        vals === nothing && return false
        for v in vals
          local nested = copy(iterVals)
          nested[iter] = v
          _appendAlgorithmStmtOps!(ops, liftedLhsNames, body, activeCond,
                                   nested, initialValue, allowContinuous) || return false
        end
      end
      DAE.STMT_WHEN(__) => nothing
      DAE.STMT_ASSERT(__) => nothing
      DAE.STMT_NORETCALL(__) => nothing
      _ => nothing
    end
  end
  return true
end

Base.@nospecializeinfer function _buildAlgorithmBodyOps(@nospecialize(daeStmts),
                                                        initialValue::Bool,
                                                        allowContinuous::Bool = false)
  local ops = BDAE.WhenOperator[]
  local lhsNames = OrderedSet{String}()
  local ok = _appendAlgorithmStmtOps!(ops, lhsNames, daeStmts, nothing,
                                      Dict{String, Int}(), initialValue, allowContinuous)
  ok || return (BDAE.WhenOperator[], OrderedSet{String}())
  return (ops, lhsNames)
end

Base.@nospecializeinfer function _collectDiscreteRhsCrefsFromWhenOps(ops::Vector{BDAE.WhenOperator},
                                                                     assignedLhs::OrderedSet{String})
  local out = DAE.ComponentRef[]
  local seen = OrderedSet{String}()
  local blocked = copy(assignedLhs)
  local ctx = (out, seen, blocked)
  for op in ops
    @match op begin
      BDAE.ASSIGN(_, rhs, _) => Util.traverseExpTopDown(rhs, _visitDiscreteRhsCref, ctx)
      BDAE.NORETCALL(exp, _) => Util.traverseExpTopDown(exp, _visitDiscreteRhsCref, ctx)
      BDAE.ASSERT(c, m, l, _) => begin
        Util.traverseExpTopDown(c, _visitDiscreteRhsCref, ctx)
        Util.traverseExpTopDown(m, _visitDiscreteRhsCref, ctx)
        Util.traverseExpTopDown(l, _visitDiscreteRhsCref, ctx)
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
      nothing,
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
      nothing,
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

#= Build a runtime `BDAE.WHEN_EQUATION` (with chained elsewhen) from a
   `DAE.STMT_WHEN` that appears inside a regular (non-when) algorithm body.
   Every assignment in each branch body is lifted (continuous and discrete
   alike — inside a `when` all LHS are event-updated), `if` guards become
   IFEXP-conditional assigns, and the array condition `{c1, c2}` is OR-folded
   to a scalar. `initial()` is substituted to `false` for the runtime arm.
   Assigned LHS names accumulate into `allLhs`. Returns the WHEN_EQUATION or
   `nothing` if the branch contributes no operators. =#
Base.@nospecializeinfer function _stmtWhenToBdaeWhenEquation(@nospecialize(stmtWhen),
                                                             allLhs::OrderedSet{String})
  local ops, lhs = _buildAlgorithmBodyOps(stmtWhen.statementLst, false, true)
  union!(allLhs, lhs)
  local cond = nothing
  for e in _whenConditionMembers(stmtWhen.exp)
    cond = _orCondition(cond, _prepareAlgorithmExp(e, Dict{String, Int}(), false))
  end
  cond === nothing && (cond = DAE.BCONST(true))
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
  if stmtWhen.initialCall || _expMentionsInitial(stmtWhen.exp)
    local initOps, initLhs = _buildAlgorithmBodyOps(stmtWhen.statementLst, true, true)
    union!(allLhs, initLhs)
    if !isempty(initOps)
      local initialCall = DAE.CALL(Absyn.IDENT("initial"),
                                   MetaModelica.list(),
                                   DAE.callAttrBuiltinBool)
      push!(out, BDAE.INITIAL_WHEN_EQUATION(
        length(initOps),
        BDAE.WHEN_STMTS(initialCall, MetaModelica.list(initOps...), NONE()),
        stmtWhen.source,
        nothing,
      ))
    end
  end
  local weq = _stmtWhenToBdaeWhenEquation(stmtWhen, allLhs)
  weq !== nothing && push!(out, weq)
  union!(liftedLhsNames, allLhs)
  return weq !== nothing
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

Base.@nospecializeinfer function _visitDiscreteRhsCref(@nospecialize(exp),
                                                       ctx::Tuple{Vector{DAE.ComponentRef}, OrderedSet{String}, OrderedSet{String}})
  @match exp begin
    DAE.CREF(cr, _) => _pushDiscreteCref!(ctx[1], ctx[2], ctx[3], cr)
    DAE.REDUCTION(_, _, iters) => _collectReductionIterNames!(ctx[3], iters)
    _ => nothing
  end
  return (exp, true, ctx)
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

Base.@nospecializeinfer function _walkDiscreteStmtsForRhsCrefs!(out::Vector{DAE.ComponentRef},
                                                                seen::OrderedSet{String},
                                                                blocked::OrderedSet{String},
                                                                @nospecialize(stmts))
  local ctx = (out, seen, blocked)
  for s in stmts
    @match s begin
      DAE.STMT_ASSIGN(_, _, rhs, _) => Util.traverseExpTopDown(rhs, _visitDiscreteRhsCref, ctx)
      DAE.STMT_ASSIGN_ARR(_, _, rhs, _) => Util.traverseExpTopDown(rhs, _visitDiscreteRhsCref, ctx)
      DAE.STMT_IF(cond, body, _, _) => begin
        Util.traverseExpTopDown(cond, _visitDiscreteRhsCref, ctx)
        _walkDiscreteStmtsForRhsCrefs!(out, seen, blocked, body)
      end
      DAE.STMT_FOR(_, _, iter, _, _, body, _) => begin
        local pushed = !(iter in blocked)
        pushed && push!(blocked, iter)
        _walkDiscreteStmtsForRhsCrefs!(out, seen, blocked, body)
        pushed && delete!(blocked, iter)
      end
      DAE.STMT_WHILE(cond, body, _) => begin
        Util.traverseExpTopDown(cond, _visitDiscreteRhsCref, ctx)
        _walkDiscreteStmtsForRhsCrefs!(out, seen, blocked, body)
      end
      _ => nothing
    end
  end
  return nothing
end

Base.@nospecializeinfer function _collectDiscreteRhsCrefs(@nospecialize(daeStmts))::Vector{DAE.ComponentRef}
  local out = DAE.ComponentRef[]
  local seen = OrderedSet{String}()
  local blocked = OrderedSet{String}()
  _walkDiscreteStmtsForRhsCrefs!(out, seen, blocked, daeStmts)
  return out
end

Base.@nospecializeinfer function _makeChangeCall(@nospecialize(cref))
  local ty = _crefType(cref)
  local callArg = DAE.CREF(cref, ty === nothing ? DAE.T_REAL_DEFAULT : ty)
  return DAE.CALL(Absyn.IDENT("change"),
                  MetaModelica.list(callArg),
                  DAE.callAttrBuiltinBool)
end

Base.@nospecializeinfer function _buildChangeOrCondition(crefs::Vector{DAE.ComponentRef})
  isempty(crefs) && return DAE.BCONST(false)
  local acc = _makeChangeCall(crefs[1])
  for i in 2:length(crefs)
    acc = DAE.LBINARY(acc, DAE.OR(DAE.T_BOOL_DEFAULT), _makeChangeCall(crefs[i]))
  end
  return acc
end

#= §17.4.4 equation-section lift. A discrete (Bool/Int/enum) variable defined by
   `lhs = relexpr`, where `relexpr` is a discrete-time expression (relations,
   pre/initial/change, logical ops over discrete/param/const operands), is held
   constant between events and recomputed only at zero-crossings of the relations
   in its RHS. Such an equation is replaced by a paired INITIAL_WHEN_EQUATION
   (t=0 value) + runtime WHEN_EQUATION triggered by `change(rel1) or … or change(relK)`.
   This is the equation-section analogue of `synthesizeWhenEquationsFromRegularAlgorithms`,
   except the trigger is built over the RELATIONS (the event sources) rather than over
   discrete RHS crefs — `change(w_rel <= 0)` fires only at the sign flip, whereas
   `change(w_rel)` would over-trigger every step. =#

#= Discrete-time predicate: an expression with no continuous-Real dependence
   OUTSIDE a relation. Relations are event sources, so their (possibly continuous)
   operands are allowed. =#
Base.@nospecializeinfer function _isDiscreteTimeExp(@nospecialize(exp), paramOrConstNames::OrderedSet{String})::Bool
  @match exp begin
    DAE.RELATION(__) => true
    DAE.LBINARY(e1, _, e2) => _isDiscreteTimeExp(e1, paramOrConstNames) && _isDiscreteTimeExp(e2, paramOrConstNames)
    DAE.LUNARY(_, e1) => _isDiscreteTimeExp(e1, paramOrConstNames)
    DAE.BCONST(__) => true
    DAE.ICONST(__) => true
    DAE.RCONST(__) => true
    DAE.SCONST(__) => true
    DAE.ENUM_LITERAL(__) => true
    DAE.CALL(Absyn.IDENT(n), _, _) => (n in ("pre", "initial", "change", "edge", "sample", "noEvent"))
    DAE.IFEXP(c, t, f) => _isDiscreteTimeExp(c, paramOrConstNames) &&
                          _isDiscreteTimeExp(t, paramOrConstNames) &&
                          _isDiscreteTimeExp(f, paramOrConstNames)
    DAE.CREF(cr, ty) => ((string(cr) in paramOrConstNames) ? true : !_isContinuousRealType(ty))
    _ => false
  end
end

#= Collect the (structurally deduplicated) RELATION subtrees of `exp`. =#
Base.@nospecializeinfer function _collectRelationsInExp(@nospecialize(exp))::Vector{DAE.Exp}
  local rels = DAE.Exp[]
  local seen = OrderedSet{String}()
  function visit(@nospecialize(e), arg)
    if e isa DAE.RELATION
      local key = string(e)
      if !(key in seen)
        push!(seen, key)
        push!(rels, e)
      end
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(exp, visit, nothing)
  return rels
end

Base.@nospecializeinfer function _makeChangeCallExp(@nospecialize(relExp))
  return DAE.CALL(Absyn.IDENT("change"), MetaModelica.list(relExp), DAE.callAttrBuiltinBool)
end

#= True if a relation has at least one continuous-Real operand, i.e. a genuine
   zero-crossing event source. A relation over only discrete/param/const operands
   (e.g. `pre(mode) == Stuck`) has no smooth crossing and must NOT become an
   event — it is re-evaluated inside the affect from pre-event state instead. =#
Base.@nospecializeinfer function _relationHasContinuousOperand(@nospecialize(rel::DAE.Exp), paramOrConstNames::OrderedSet{String})::Bool
  local found = false
  function visit(@nospecialize(e), arg)
    if e isa DAE.CREF && !found
      if !(string(e.componentRef) in paramOrConstNames) && _isContinuousRealType(e.ty)
        found = true
      end
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(rel, visit, nothing)
  return found
end

Base.@nospecializeinfer function _buildChangeOrConditionFromExps(rels::Vector{DAE.Exp})
  isempty(rels) && return DAE.BCONST(false)
  local acc = _makeChangeCallExp(rels[1])
  for i in 2:length(rels)
    acc = DAE.LBINARY(acc, DAE.OR(DAE.T_BOOL_DEFAULT), _makeChangeCallExp(rels[i]))
  end
  return acc
end

#= A discrete (Bool/Int/enum) equation `lhs = rhs` whose RHS is a discrete-time
   expression. Returns `(name, lhs, rhs, src, eq)` or `nothing`. Unlike the
   when-emit step this does NOT require the RHS to contain a relation: a
   no-relation member like `locked = pre(stuck) and not startForward` qualifies
   as a candidate so it can join a coupled cluster (it is event-driven through
   its siblings' relations). =#
Base.@nospecializeinfer function _discreteBoolCandidate(@nospecialize(eq::BDAE.Equation),
                                                        paramOrConstNames::OrderedSet{String})
  eq isa BDAE.EQUATION || return nothing
  local lhs = eq.lhs
  lhs isa DAE.CREF || return nothing
  local lhsDiscrete = _isDiscreteDAEType(lhs.ty)
  if !lhsDiscrete
    local ct = _crefType(lhs.componentRef)
    lhsDiscrete = ct !== nothing && _isDiscreteDAEType(ct)
  end
  lhsDiscrete || return nothing
  local lhsName = string(lhs.componentRef)
  (lhsName == "time" || lhsName in paramOrConstNames) && return nothing
  _isDiscreteTimeExp(eq.rhs, paramOrConstNames) || return nothing
  return (name = lhsName, lhs = lhs, rhs = eq.rhs, src = eq.source, eq = eq)
end

#= Candidate cref names referenced anywhere in `exp` (including inside pre()).
   Used to connect a cluster: two candidates are coupled if either references
   the other. =#
Base.@nospecializeinfer function _candRefsAnywhere(@nospecialize(exp::DAE.Exp), restrict::OrderedSet{String})::OrderedSet{String}
  local found = OrderedSet{String}()
  function visit(@nospecialize(e), arg)
    if e isa DAE.CREF
      local nm = string(e.componentRef)
      (nm in restrict) && push!(found, nm)
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(exp, visit, nothing)
  return found
end

#= Candidate cref names referenced OUTSIDE any pre(): the topological-order
   edges. References inside pre() are loop-breakers (read the pre-event value)
   and impose no order. =#
Base.@nospecializeinfer function _candRefsOutsidePre(@nospecialize(exp::DAE.Exp), restrict::OrderedSet{String})::OrderedSet{String}
  local found = OrderedSet{String}()
  function f(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("pre"), _, _) => (e, false, arg)
      DAE.CREF(cr, _) => begin
        (string(cr) in restrict) && push!(found, string(cr))
        (e, true, arg)
      end
      _ => (e, true, arg)
    end
  end
  Util.traverseExpTopDown(exp, f, nothing)
  return found
end

#= Replace every bare (outside-pre) occurrence of a sibling cref with its
   already-inlined RHS. pre(sibling) is left untouched so it keeps reading the
   pre-event held value. =#
Base.@nospecializeinfer function _inlineSiblingsOutsidePre(@nospecialize(exp::DAE.Exp), subst::Dict{String, DAE.Exp})
  function f(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("pre"), _, _) => (e, false, arg)
      DAE.CREF(cr, _) => begin
        local nm = string(cr)
        haskey(subst, nm) ? (subst[nm], false, arg) : (e, true, arg)
      end
      _ => (e, true, arg)
    end
  end
  return first(Util.traverseExpTopDown(exp, f, nothing))
end

#= Start-attribute expression for each variable that has one (Bool/Int/Real/enum).
   Used to fold `pre(member)` at initialization: Modelica §8.6.2 — before the
   first event `pre(v)` is `v.start`. =#
Base.@nospecializeinfer function _discreteStartExpLookup(variables::Vector{BDAE.VAR})::Dict{String, DAE.Exp}
  local d = Dict{String, DAE.Exp}()
  for var in variables
    local s = @match var.values begin
      SOME(va) => @match va begin
        DAE.VAR_ATTR_BOOL(start = SOME(e)) => e
        DAE.VAR_ATTR_INT(start = SOME(e)) => e
        DAE.VAR_ATTR_REAL(start = SOME(e)) => e
        DAE.VAR_ATTR_ENUMERATION(start = SOME(e)) => e
        _ => nothing
      end
      _ => nothing
    end
    s === nothing && continue
    d[string(var.varName)] = s
  end
  return d
end

#= Fold `pre(member)` (member a lifted cluster discrete) to the member's start
   value for the INITIAL_WHEN body. The default false matches the Boolean start
   default; pre() of non-members is left untouched. =#
Base.@nospecializeinfer function _foldPreOfMembers(@nospecialize(exp::DAE.Exp), members::OrderedSet{String},
                                                   startLookup::Dict{String, DAE.Exp})
  function f(@nospecialize(e), arg)
    @match e begin
      DAE.CALL(Absyn.IDENT("pre"), args, _) => begin
        local inner = listHead(args)
        if inner isa DAE.CREF && (string(inner.componentRef) in members)
          (get(startLookup, string(inner.componentRef), DAE.BCONST(false)), false, arg)
        else
          (e, false, arg)
        end
      end
      _ => (e, true, arg)
    end
  end
  return first(Util.traverseExpTopDown(exp, f, nothing))
end

#= Connected components over the undirected coupling graph. =#
function _connectedComponents(names::Vector{String}, adj::Dict{String, OrderedSet{String}})::Vector{Vector{String}}
  local seen = OrderedSet{String}()
  local comps = Vector{String}[]
  for start in names
    start in seen && continue
    local comp = String[]
    local stack = String[start]
    push!(seen, start)
    while !isempty(stack)
      local n = pop!(stack)
      push!(comp, n)
      for m in adj[n]
        if !(m in seen)
          push!(seen, m); push!(stack, m)
        end
      end
    end
    push!(comps, comp)
  end
  return comps
end

#= Topological order of a cluster by non-pre dependency edges (dep before the
   member that references it outside pre). Returns the ordered names, or
   `nothing` if a non-pre cycle survives (the cluster is then left unlifted). =#
function _topoOrderCluster(cluster::Vector{String}, candByName::Dict{String, Any})::Union{Vector{String}, Nothing}
  local clusterSet = OrderedSet(cluster)
  local indeg = Dict{String, Int}(n => 0 for n in cluster)
  local succ = Dict{String, Vector{String}}(n => String[] for n in cluster)
  for n in cluster
    for d in _candRefsOutsidePre(candByName[n].rhs, clusterSet)
      if d != n && d in clusterSet
        push!(succ[d], n); indeg[n] += 1
      end
    end
  end
  local q = sort!([n for n in cluster if indeg[n] == 0])
  local order = String[]
  while !isempty(q)
    local n = popfirst!(q)
    push!(order, n)
    for m in succ[n]
      indeg[m] -= 1
      indeg[m] == 0 && push!(q, m)
    end
    sort!(q)
  end
  return length(order) == length(cluster) ? order : nothing
end

#= Emit one cluster of coupled discrete-Boolean equations as a single ordered
   when-cluster: topologically sort, inline non-pre sibling references so each
   ASSIGN RHS is pure in pre-event state + relations, and emit one
   INITIAL_WHEN + one runtime WHEN (triggered by `change()` over the UNION of the
   cluster's relations) whose bodies are the ordered ASSIGN list. =#
function _emitDiscreteCluster!(out::Vector{BDAE.Equation}, cluster::Vector{String},
                               candByName::Dict{String, Any}, liftedLhs::OrderedSet{String},
                               startLookup::Dict{String, DAE.Exp},
                               paramOrConstNames::OrderedSet{String})
  local order = _topoOrderCluster(cluster, candByName)
  if order === nothing
    @warn "[BDAE: lifter] cyclic discrete cluster left unlifted" cluster
    for n in cluster; push!(out, candByName[n].eq); end
    return
  end
  local members = OrderedSet(cluster)
  local subst = Dict{String, DAE.Exp}()
  local body = Tuple{DAE.Exp, DAE.Exp, Any}[]
  for n in order
    local c = candByName[n]
    local inlined = isempty(subst) ? c.rhs : _inlineSiblingsOutsidePre(c.rhs, subst)
    subst[n] = inlined
    push!(body, (c.lhs, inlined, c.src))
  end
  #= Trigger only on relations with a continuous operand (real zero-crossings).
     All-discrete relations such as `pre(mode) == Stuck` are kept in the affect
     RHS but excluded as event sources. =#
  local rels = DAE.Exp[]
  local seen = OrderedSet{String}()
  for (_, r, _) in body
    for rel in _collectRelationsInExp(r)
      local k = string(rel)
      if !(k in seen) && _relationHasContinuousOperand(rel, paramOrConstNames)
        push!(seen, k); push!(rels, rel)
      end
    end
  end
  if isempty(rels)
    #= No continuous event source: leave as residuals. =#
    for n in cluster; push!(out, candByName[n].eq); end
    return
  end
  local bareInitial = DAE.CALL(Absyn.IDENT("initial"), MetaModelica.list(), DAE.callAttrBuiltinBool)
  local changeCond = _buildChangeOrConditionFromExps(rels)
  local src0 = body[1][3]
  #= INITIAL body: pre(member) ≡ member.start (no held value exists yet). The
     runtime body keeps pre() — it resolves to the affect-entry held value. =#
  local initAssigns = [BDAE.ASSIGN(lhs, _replaceInitialCall(_foldPreOfMembers(r, members, startLookup), true), s)
                       for (lhs, r, s) in body]
  local runAssigns  = [BDAE.ASSIGN(lhs, _replaceInitialCall(r, false), s) for (lhs, r, s) in body]
  push!(out, BDAE.INITIAL_WHEN_EQUATION(
    1,
    BDAE.WHEN_STMTS(bareInitial, MetaModelica.list(initAssigns...), NONE()),
    src0,
    nothing,
  ))
  push!(out, BDAE.WHEN_EQUATION(
    1,
    BDAE.WHEN_STMTS(changeCond, MetaModelica.list(runAssigns...), NONE()),
    src0,
    nothing,
  ))
  for (lhs, _, _) in body; push!(liftedLhs, string(lhs.componentRef)); end
  return
end

"""
    synthesizeWhenEquationsFromDiscreteEquations(equations, paramOrConstNames) -> (Vector{BDAE.Equation}, OrderedSet{String})

Replace qualifying discrete-Boolean/Integer equation-section definitions with
paired INITIAL_WHEN + runtime WHEN equations. Mutually-referencing definitions
(e.g. the Coulomb-friction `{startForward, locked, stuck}` FSM) are grouped into
one ordered cluster so a no-relation member is still event-driven through its
siblings' relations, and the cluster recomputes in topological order on any
member relation crossing. Self-gating: equations that do not qualify are returned
unchanged, so models with no discrete-time defining equations pay only a single
linear scan.
"""
function synthesizeWhenEquationsFromDiscreteEquations(equations::Vector{BDAE.Equation},
                                                      paramOrConstNames::OrderedSet{String} = OrderedSet{String}(),
                                                      startLookup::Dict{String, DAE.Exp} = Dict{String, DAE.Exp}())
  local out = BDAE.Equation[]
  local cands = Any[]
  for eq in equations
    local c = _discreteBoolCandidate(eq, paramOrConstNames)
    if c === nothing
      push!(out, eq)
    else
      push!(cands, c)
    end
  end
  isempty(cands) && return (equations, OrderedSet{String}())
  #= An LHS with more than one defining equation is a connect/alias chain, not
     a discrete definition; lifting it through a name-keyed map would silently
     drop all but one of its equations. Keep such equations unchanged. =#
  local lhsCount = Dict{String, Int}()
  for c in cands
    lhsCount[c.name] = get(lhsCount, c.name, 0) + 1
  end
  local candByName = Dict{String, Any}()
  local candNames = String[]
  for c in cands
    if lhsCount[c.name] > 1
      push!(out, c.eq)
    else
      candByName[c.name] = c
      push!(candNames, c.name)
    end
  end
  isempty(candNames) && return (equations, OrderedSet{String}())
  local candSet = OrderedSet(candNames)
  local adj = Dict{String, OrderedSet{String}}(n => OrderedSet{String}() for n in candNames)
  for n in candNames
    for r in _candRefsAnywhere(candByName[n].rhs, candSet)
      if r != n
        push!(adj[n], r); push!(adj[r], n)
      end
    end
  end
  local liftedLhs = OrderedSet{String}()
  for cluster in _connectedComponents(candNames, adj)
    _emitDiscreteCluster!(out, cluster, candByName, liftedLhs, startLookup, paramOrConstNames)
  end
  return (out, liftedLhs)
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
            nothing,
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
        nothing,
      ))
      #= Plus a runtime WHEN_EQUATION for any change(rhs) trigger so the
         assign re-fires whenever a non-parameter input changes. =#
      if changeCond !== nothing
        push!(out, BDAE.WHEN_EQUATION(
          1,
          BDAE.WHEN_STMTS(changeCond, whenOps, NONE()),
          source,
          nothing,
        ))
      end
      local lhsName::Union{String, Nothing} = try
        @match lhs begin
          DAE.CREF(cr, _) => string(cr)
          _ => nothing
        end
      catch
        nothing
      end
      return (true, lhsName)
    end
    _ => return (false, nothing)
  end
end

function synthesizeInitialWhenFromAlgorithms(algorithms)::Vector{BDAE.Equation}
  local out = BDAE.Equation[]
  for alg in algorithms
    for stmt in alg.statements
      stmt isa OMFrontend.Frontend.ALG_WHEN || continue
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
            nothing,
          )
          _INIT_ALG_DAE_STMTS[node] = collect(daeStmts)
          push!(out, node)
        end
        _ => nothing
      end
    end
  end
  return out
end

@exportAll()
end
