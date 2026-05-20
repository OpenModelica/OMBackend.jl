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
  local _whenLifterSkipLhs = Set{String}()
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
  local initialEquations = BDAE.Equation[]
  for ieq in OMFrontend.Frontend.convertEquations(flatModel.initialEquations)
    local iresult = equationToBackendEquation(ieq)
    if iresult isa Vector
      append!(initialEquations, iresult)
    else
      push!(initialEquations, iresult)
    end
  end
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

"""
  Deduplicate variables by their component reference name.
  Keeps the first occurrence of each uniquely-named variable.
"""
function deduplicateVariables(variables::Vector)::Vector
  local seen = Set{String}()
  local unique_vars = similar(variables, 0)
  local duplicateCount = 0
  for v in variables
    local varStr = string(v.varName)
    if varStr in seen
      duplicateCount += 1
      continue
    end
    push!(seen, varStr)
    push!(unique_vars, v)
  end
  if duplicateCount > 0
    println("[dedup] Variables: $(length(variables)) -> $(length(unique_vars)) (removed $duplicateCount duplicates)")
  end
  return unique_vars
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
  local seen = Set{UInt}()
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
      BDAE.VAR(vName, BDAE.STATE() || BDAE.VARIABLE(), _, DAE.T_REAL(__),
               SOME(bindExp), _, _, _, _, _) => begin
                 local lhs = DAE.CREF(vName, v.varType)
                 local rhs = bindExp
                 local eq =  BDAE.EQUATION(lhs, rhs, v.source, BDAE.NO_ATTRIBUTES())
                 push!(bindingEqs, eq)
               end
      #= Binding equations with discrete type =#
      BDAE.VAR(vName, BDAE.STATE() || BDAE.VARIABLE(), _, DAE.T_BOOL(__),
               SOME(bindExp), _, _, _, _, _) => begin
                 #= Treat discrete binding equations as when equations =#
                 if bindExp isa DAE.IFEXP
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
Base.@nospecializeinfer function _pushExpCrefStrings!(names::Set{String}, @nospecialize(exp))
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
function _collectAllCrefsInEquations(equations)::Set{String}
  local names = Set{String}()
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
function _collectParamOrConstNames(variables::Vector{BDAE.VAR})::Set{String}
  local names = Set{String}()
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

Base.@nospecializeinfer function _collectAssignResidualsFromDAEStmts!(out::Vector{BDAE.Equation},
                                              @nospecialize(daeStmts),
                                              @nospecialize(source),
                                              eqLhsBoundCrefs::Set{String},
                                              whenLifterSkipLhs::Set{String} = Set{String}())
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
          (crStr in eqLhsBoundCrefs) && continue
          (crStr in whenLifterSkipLhs) && continue
        end
        push!(out, BDAE.RESIDUAL_EQUATION(
          DAE.BINARY(lhs, DAE.SUB(DAE.T_REAL_DEFAULT), rhs),
          src,
          BDAE.EQ_ATTR_DEFAULT_DYNAMIC,
        ))
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
                                                  eqLhsBoundCrefs::Set{String} = Set{String}(),
                                                  whenLifterSkipLhs::Set{String} = Set{String}())::Vector{BDAE.Equation}
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
Base.@nospecializeinfer function _pushExpRhsCrefs!(out::Set{Tuple{DAE.ComponentRef, DAE.Type}}, @nospecialize(exp))
  exp === nothing && return nothing
  for c in Util.getAllCrefs(exp)
    local ty = _crefType(c)
    ty === nothing && continue
    push!(out, (c, ty))
  end
  return nothing
end

Base.@nospecializeinfer function _walkRhsCrefsInDAEStmts!(out::Set{Tuple{DAE.ComponentRef, DAE.Type}}, @nospecialize(stmts))
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

function _collectRhsCrefsInDAEStmts(daeStmts)::Set{Tuple{DAE.ComponentRef, DAE.Type}}
  local out = Set{Tuple{DAE.ComponentRef, DAE.Type}}()
  _walkRhsCrefsInDAEStmts!(out, daeStmts)
  return out
end

#= Best-effort: extract the type carried by a `DAE.ComponentRef`. Each
   CREF_IDENT / CREF_QUAL stores its identType; CREF_ITER and WILD are not
   useful triggers. =#
Base.@nospecializeinfer function _crefType(@nospecialize(cref))
  @match cref begin
    DAE.CREF_IDENT(_, ty, _) => ty
    DAE.CREF_QUAL(_, ty, _, _) => ty
    _ => nothing
  end
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
                                                      paramOrConstNames::Set{String} = Set{String}())
  local out = BDAE.Equation[]
  local liftedLhsNames = Set{String}()
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
    _liftAlgorithmBodyToInitialWhen!(out, daeStmts, alg.source, liftedLhsNames)
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
                                                                  liftedLhsNames::Set{String})
  local ops = BDAE.WhenOperator[]
  for s in daeStmts
    @match s begin
      DAE.STMT_ASSIGN(ty, lhs, rhs, asrc) => begin
        _isDiscreteDAEType(ty) || continue
        lhs isa DAE.CREF || continue
        push!(ops, BDAE.ASSIGN(lhs, rhs, asrc))
        push!(liftedLhsNames, string(lhs.componentRef))
      end
      DAE.STMT_IF(cond, body, _, _) => begin
        local bodyVec = listArray(body)
        length(bodyVec) == 1 || continue
        local b1 = bodyVec[1]
        @match b1 begin
          DAE.STMT_ASSIGN(ity, ilhs, irhs, asrc) => begin
            _isDiscreteDAEType(ity) || continue
            ilhs isa DAE.CREF || continue
            #= `lhs := if cond then irhs else lhs` — re-reading the current
               value preserves whatever previous statements set. =#
            local ifExp = DAE.IFEXP(cond, irhs, ilhs)
            push!(ops, BDAE.ASSIGN(ilhs, ifExp, asrc))
            push!(liftedLhsNames, string(ilhs.componentRef))
          end
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  isempty(ops) && return
  local initialCall = DAE.CALL(Absyn.IDENT("initial"),
                               MetaModelica.list(),
                               DAE.callAttrBuiltinBool)
  push!(out, BDAE.INITIAL_WHEN_EQUATION(
    length(ops),
    BDAE.WHEN_STMTS(initialCall, MetaModelica.list(ops...), NONE()),
    source,
    nothing,
  ))
  #= Per Modelica spec §17.4.4: a non-when algorithm with discrete LHS fires
     at any event that changes its RHS inputs. The INITIAL_WHEN above sets the
     LHS at t=0; we also need a regular WHEN_EQUATION whose condition is
     `change(d1) OR change(d2) ... ` over every discrete cref referenced
     in the body, so the LHS keeps tracking those inputs as they flip during
     simulation. Without this, AlgorithmDiscreteAssign's `out := trigger + 10`
     would stay pinned at its t=0 value (out = 13) even after `trigger`
     becomes 7 at t=0.5. =#
  local discRhsCrefs = _collectDiscreteRhsCrefs(daeStmts)
  if !isempty(discRhsCrefs)
    local cond = _buildChangeOrCondition(discRhsCrefs)
    push!(out, BDAE.WHEN_EQUATION(
      length(ops),
      BDAE.WHEN_STMTS(cond, MetaModelica.list(ops...), NONE()),
      source,
      nothing,
    ))
  end
  return
end

Base.@nospecializeinfer function _pushDiscreteCref!(out::Vector{DAE.ComponentRef},
                                                    seen::Set{String},
                                                    blocked::Set{String},
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
                                                       ctx::Tuple{Vector{DAE.ComponentRef}, Set{String}, Set{String}})
  @match exp begin
    DAE.CREF(cr, _) => _pushDiscreteCref!(ctx[1], ctx[2], ctx[3], cr)
    DAE.REDUCTION(_, _, iters) => _collectReductionIterNames!(ctx[3], iters)
    _ => nothing
  end
  return (exp, true, ctx)
end

Base.@nospecializeinfer function _collectReductionIterNames!(blocked::Set{String}, @nospecialize(iters))
  for it in iters
    @match it begin
      DAE.REDUCTIONITER(id, _, _, _) => push!(blocked, id)
      _ => nothing
    end
  end
  return nothing
end

Base.@nospecializeinfer function _walkDiscreteStmtsForRhsCrefs!(out::Vector{DAE.ComponentRef},
                                                                seen::Set{String},
                                                                blocked::Set{String},
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
  local seen = Set{String}()
  local blocked = Set{String}()
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
                                                              paramOrConstNames::Set{String} = Set{String}())
  @match stmt begin
    DAE.STMT_ASSIGN(ty, lhs, rhs, asrc) => begin
      _isDiscreteDAEType(ty) || return (false, nothing)
      lhs isa DAE.CREF || return (false, nothing)
      local bareInitial::DAE.Exp = DAE.CALL(Absyn.IDENT("initial"),
                                            MetaModelica.list(),
                                            DAE.callAttrBuiltinBool)
      local changeCond::Union{DAE.Exp, Nothing} = nothing
      local rhsCrefs = Set{Tuple{DAE.ComponentRef, DAE.Type}}()
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
