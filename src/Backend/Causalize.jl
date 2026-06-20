#= /*
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

module Causalize

using Setfield
using ExportAll
using MetaModelica
using DataStructures: OrderedSet, OrderedDict

import Absyn
import ..BDAE
import ..BDAEUtil
import ..BackendEquation
import ..@BACKEND_LOGGING
import ..FrontendUtil.Util
import DAE
import OMBackend


"""
    Variable can be: Variable, Discrete, Constant and Parameters
    From this create Algebraic and State Variables (Known variable)
    Traverse all equations and locate the variables that are derived.
    These we mark as states
"""
function detectStates(dae::BDAE.BACKEND_DAE)
  BDAEUtil.mapEqSystems(dae, detectStatesEqSystem)
end

"""
This function detects and removes unused parameters a modeler might have introduced by mistake.
This reduces the explosion of parameters common for instance when using large matrices.
author:johti17
"""
function detectUnusedParametersAndConstants(bdae::BDAE.BACKEND_DAE)
  #= Those that are to be kept have been temporary marked DUMMY_STATE =#
  bdae = BDAEUtil.mapEqSystems(bdae, detectParamsEqSystem)
  @assert length(bdae.eqs) == 1 "Eq systems larger than 1 not supported"
  local sys = first(bdae.eqs)
  #= Remove parameters of all types, but leave complex types alone. =#
  newOrderedVars = filter((x) -> (x.varKind !== BDAE.PARAM() || x.varType isa DAE.T_COMPLEX), sys.orderedVars)
  for (i, v) in enumerate(newOrderedVars)
    if v.varKind === BDAE.DUMMY_STATE()
      tv = newOrderedVars[i]
      @assign tv.varKind = BDAE.PARAM()
      newOrderedVars[i] = tv
    end
  end
  @assign first(bdae.eqs).orderedVars = newOrderedVars
  return bdae
end


"""
  Replaces all if expressions with a temporary variable.
  Adds an equation assigning this variable to the set of equations.
"""
function detectIfExpressions(dae::BDAE.BACKEND_DAE)
  BDAEUtil.mapEqSystems(dae, detectIfEquationsEqSystem)
end


"""
    kabdelhak:
    Detects all states in the system by looking for component references in
    der() calls.
    Updates all variables with those component references to
    varKind BDAE.STATE()
"""
function detectStatesEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  syst = begin
    local vars::Vector{BDAE.VAR}
    local eqs::Vector{BDAE.Equation}
    local stateCrefs = Dict{DAE.ComponentRef, Bool}()
    @match syst begin
      BDAE.EQSYSTEM(name, vars, eqs, simpleEqs, initialEqs) => begin
        for eq in eqs
          (_, stateCrefs) = BDAEUtil.traverseEquationExpressions(eq, detectStateExpression, stateCrefs)
        end
        #= Do replacements for stateCrefs =#
        @assign syst.orderedVars = updateStates(vars, stateCrefs)
        syst
      end
    end
  end
  return syst
end

"""
 Detect parameters used in the equations.
Save those variables in a HT.
"""
function detectParamsEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  local pars = filter((x) -> x.varKind === BDAE.PARAM(), syst.orderedVars)
  local parStrs = OrderedSet(map((x) -> string(x.varName), pars))
  local buffer = IOBuffer()
  @BACKEND_LOGGING write(OMBackend.logPath("backend/bdae", "allpars.log"), String(take!(buffer)))

  function detectParamExpression(exp::DAE.Exp, paramCrefs::Dict{DAE.ComponentRef, Bool})
    local cont::Bool
    local outCrefs = paramCrefs
    (outCrefs, cont) = begin
      local param::DAE.ComponentRef
      @match exp begin
        #= Ignore complex components =#
        DAE.CREF(c, DAE.T_COMPLEX(__)) => begin
          outCrefs[exp.componentRef] = true
          (outCrefs, true)
        end
        DAE.CREF(__) => begin
          local cand = string(exp.componentRef)
          if (cand in parStrs)
            #println("Located param in the variables:" * cand)
            #println(exp)
            outCrefs[exp.componentRef] = true
          end
          (outCrefs, true)
        end
        DAE.IFEXP(cond, expThen, expElse) => begin
          #fail()
          (outCrefs, true)
        end
        _ => begin
          (outCrefs, true)
        end
      end
    end
    return (exp, cont, outCrefs)
  end

  function updateParams(vars::Vector, paramCrefs::Dict{DAE.ComponentRef, Bool})
    local varArr::Vector{BDAE.VAR} = vars
    for i in 1:arrayLength(varArr)
      varArr[i] = begin
        local cref::DAE.ComponentRef
        local var::BDAE.Var
        @match varArr[i] begin
          var && BDAE.VAR(varName = cref) where (haskey(paramCrefs, cref)) => begin
            @assign var.varKind = BDAE.DUMMY_STATE()#= In the meantime. Mark it for keeping=#
            var
          end
          _ => begin
            varArr[i]
          end
        end
      end
      vars = varArr
    end
    return vars
  end

  syst = begin
    local vars::BDAE.Variables
    local eqs::Array
    local paramCrefs = Dict{DAE.ComponentRef, Bool}()
    @match syst begin
      BDAE.EQSYSTEM(name, vars, eqs, simpleEqs, initialEqs) => begin
        for eq in eqs
          (_, paramCrefs) = BDAEUtil.traverseEquationExpressions(eq, detectParamExpression, paramCrefs)
        end
        #= Go through the initial equations =#
        for ieq in initialEqs
          (_, paramCrefs) = BDAEUtil.traverseEquationExpressions(ieq, detectParamExpression, paramCrefs)
        end
        #= Do replacements for paramCrefs =#
        @assign syst.orderedVars = updateParams(vars, paramCrefs)
        syst
      end
    end
  end
  return syst
end

"""
johti17:
  Detects if-equations.
  Returns new temporary variables and an array of equations
"""
function detectIfEquationsEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  syst = begin
    local vars::BDAE.Variables
    local eqs::Array
    #= Tick is used to keep track of generated if-equations =#
    local tick::Ref{Int} = 0
    #= AUDIT (ombackend-bug-audit-2026-06-05 #7): BDAE.VAR is a mutable struct
       hashed by objectid, so a plain Dict iterates the lifted ifEq_tmp vars/eqs
       in a run-dependent order; that order perturbs variable/equation numbering
       -> matching -> tearing-variable selection, a real determinism defect.
       OrderedDict makes the append order deterministic (equation-traversal
       order). =#
    local tmpVarToElement = OrderedDict{BDAE.VAR, BDAE.IF_EQUATION}()
    #= Canonical (cond|then|else) string -> existing CREF for structural
       dedup. Identical lifted (cond, then, else) shapes share a single
       ifEq_tmp var so MTK does not get two SymbolicContinuousCallbacks
       with byte-identical zero-crossing conditions (it drops all but one,
       leaving the other branch's ifCond parameter pinned at its initial
       value). =#
    local dedup = Dict{String, DAE.Exp}()
    local tmpVarToElementAndTick = (tmpVarToElement, tick, dedup)
    @match syst begin
      BDAE.EQSYSTEM(__) => begin
        for i in 1:length(syst.orderedEqs)
          local eq = syst.orderedEqs[i]
          #= Discrete callback bodies (WHEN / INITIAL_WHEN) carry IFEXPs that
             must be lowered in-place by the callback codegen, not extracted
             into global algebraic ifEq_tmp* equations whose conditions get
             pinned at MTK simplify time. =#
          if eq isa BDAE.WHEN_EQUATION || eq isa BDAE.INITIAL_WHEN_EQUATION
            continue
          end
          (eq2, _) = BDAEUtil.traverseEquationExpressions(eq, replaceIfExpressionWithTmpVar,
                                                                  tmpVarToElementAndTick)
          if ! (eq === eq2)
            @assign syst.orderedEqs[i] = eq2
          end
        end

        #= Append the new variables to the list of variables =#
        local newVariables = collect(keys(tmpVarToElement))
        local newEquations = collect(values(tmpVarToElement))
        append!(syst.orderedEqs, newEquations)
        append!(syst.orderedVars, newVariables)
        syst
      end
    end
  end
  return syst
end

"""
  Walk a DAE expression and report whether any CREF named `time` appears in it.
  Used to distinguish monotonic time-dependent if-conditions (safe to lift
  to one-shot zero-crossing events) from state-dependent if-conditions
  (which need continuous bool-product evaluation because the underlying
  comparison can cross its threshold in either direction).
"""
function _expDependsOnTime(@nospecialize(e::DAE.Exp))::Bool
  local found = Ref{Bool}(false)
  function _scan(@nospecialize(x::DAE.Exp), seen::Ref{Bool})
    seen[] && return (x, false, seen)
    @match x begin
      DAE.CREF(componentRef = c) => begin
        local s = string(c)
        if s == "time"
          seen[] = true
        end
        (x, true, seen)
      end
      _ => (x, true, seen)
    end
  end
  Util.traverseExpTopDown(e, _scan, found)
  return found[]
end

"""
  Detects if expression.
  We replace the if expression with our temporary variable.
  These variables are assigned in newly created if equations that we add to the tmpVarToElement::Dict.
  We create the mapping:
  tmpVar -> equation it is assigned in
"""
Base.@nospecializeinfer function replaceIfExpressionWithTmpVar(@nospecialize(exp::DAE.Exp), tmpVarToElementAndTick::Tuple{AbstractDict{BDAE.VAR, BDAE.IF_EQUATION}, Ref{Int}, Dict{String, DAE.Exp}})
  (newExp, cont, tmpVarToElementAndTick) = begin
    local tmpVarToElement::AbstractDict{BDAE.VAR, BDAE.IF_EQUATION} = tmpVarToElementAndTick[1]
    local tick::Ref{Int} = tmpVarToElementAndTick[2]
    local dedup::Dict{String, DAE.Exp} = tmpVarToElementAndTick[3]
    @match exp begin
      #= Per Modelica spec, `noEvent(expr)` takes relations literally and
         triggers no events. An IFEXP inside it must stay inline (codegen emits
         a continuous ifelse/bool-product) — lifting it to an event-driven
         IF_EQUATION would latch the branch at the first crossing and break a
         feedback saturation whose condition never re-crosses zero. Stop
         descent so the whole noEvent subtree is preserved. =#
      DAE.CALL(Absyn.IDENT("noEvent"), _, _) => (exp, false, tmpVarToElementAndTick)
      DAE.IFEXP(cond, expThen, expElse) => begin
        #= Recursively process the branches. Use the time-dependent-only
           variant so that nested state-dependent IFEXPs (e.g. LimPID's
           saturation) keep their bool-product lowering — one-shot events
           would freeze the saturation at its first crossing. Nested
           time-dependent IFEXPs DO get lifted so each transition produces
           a proper SymbolicContinuousCallback. =#
        local (liftedCond, _)  = Util.traverseExpTopDown(cond,    _replaceTimeDepIfExpressionWithTmpVar, tmpVarToElementAndTick)
        local (liftedThen, _)  = Util.traverseExpTopDown(expThen, _replaceTimeDepIfExpressionWithTmpVar, tmpVarToElementAndTick)
        local (liftedElse, _)  = Util.traverseExpTopDown(expElse, _replaceTimeDepIfExpressionWithTmpVar, tmpVarToElementAndTick)
        #= Structural dedup: if a lifted IF_EQUATION with the same
           (cond | then | else) already exists, reuse its CREF instead of
           generating a duplicate. MTK drops SymbolicContinuousCallbacks
           with byte-identical zero-crossing conditions; this dedup
           prevents two separate ifCond parameters from depending on the
           same event and ending up with one of them pinned. =#
        local key = string(liftedCond, "|", liftedThen, "|", liftedElse)
        local existing = get(dedup, key, nothing)
        if existing !== nothing
          (existing, true, tmpVarToElementAndTick)
        else
          #= Bump per IFEXP so siblings within the same residual get distinct
             names; a single per-equation bump would collide every nested
             IFEXP onto one symbolic var. =#
          local varType = DAE.T_REAL_DEFAULT
          local varName = string("ifEq_tmp", tick.x)
          tick.x += 1
          local var::DAE.ComponentRef = DAE.CREF_IDENT(varName, varType, nil)
          local varAsCREF::DAE.CREF = DAE.CREF(var, varType)
          local emptySource = DAE.emptyElementSource
          local attr = BDAE.EQ_ATTR_DEFAULT_UNKNOWN
          local backendVar = BDAE.VAR(DAE.CREF_IDENT(varName, DAE.T_UNKNOWN_DEFAULT, nil),
                                      BDAE.VARIABLE(), varType)
          tmpVarToElement[backendVar] = BDAE.IF_EQUATION(list(liftedCond),
                                                         list(list(BDAE.EQUATION(varAsCREF, liftedThen, emptySource, attr))),
                                                         list(BDAE.EQUATION(varAsCREF, liftedElse, emptySource, attr)),
                                                         emptySource,
                                                         BDAE.EQ_ATTR_DEFAULT_UNKNOWN)
          dedup[key] = varAsCREF
          (varAsCREF, true, tmpVarToElementAndTick)
        end
      end
      _ => begin
        (exp, true, tmpVarToElementAndTick)
      end
    end
  end
  return (newExp, cont, tmpVarToElementAndTick)
end

"""
  Recursive variant used inside `replaceIfExpressionWithTmpVar` for the
  branches of an already-lifted IFEXP. Lifts only IFEXPs whose condition
  references `time`; leaves state-dependent IFEXPs in place so codegen
  emits a continuous bool-product (correct for non-monotonic conditions).
  Recursion descends into the branches either way so deeper nested
  time-dependent IFEXPs still reach the lifter.
"""
Base.@nospecializeinfer function _replaceTimeDepIfExpressionWithTmpVar(@nospecialize(exp::DAE.Exp), tmpVarToElementAndTick::Tuple{AbstractDict{BDAE.VAR, BDAE.IF_EQUATION}, Ref{Int}, Dict{String, DAE.Exp}})
  (newExp, cont, tmpVarToElementAndTick) = begin
    @match exp begin
      #= See replaceIfExpressionWithTmpVar: never lift inside noEvent. =#
      DAE.CALL(Absyn.IDENT("noEvent"), _, _) => (exp, false, tmpVarToElementAndTick)
      DAE.IFEXP(cond, expThen, expElse) => begin
        if _expDependsOnTime(cond)
          #= Time-dependent → delegate to the always-lift path. =#
          replaceIfExpressionWithTmpVar(exp, tmpVarToElementAndTick)
        else
          #= State-dependent or parameter-only → do not lift. Still recurse
             so nested time-dependent IFEXPs below get lifted. =#
          local (lc, _)  = Util.traverseExpTopDown(cond,    _replaceTimeDepIfExpressionWithTmpVar, tmpVarToElementAndTick)
          local (lt, _)  = Util.traverseExpTopDown(expThen, _replaceTimeDepIfExpressionWithTmpVar, tmpVarToElementAndTick)
          local (le, _)  = Util.traverseExpTopDown(expElse, _replaceTimeDepIfExpressionWithTmpVar, tmpVarToElementAndTick)
          (DAE.IFEXP(lc, lt, le), false, tmpVarToElementAndTick)
        end
      end
      _ => begin
        (exp, true, tmpVarToElementAndTick)
      end
    end
  end
  return (newExp, cont, tmpVarToElementAndTick)
end

"""
    kabdelhak:
    Detects if a given expression is a der() call and adds the corresponding
    cref to a hashmap
"""
function detectStateExpression(exp::DAE.Exp, stateCrefs::Dict{DAE.ComponentRef, Bool})
  local cont::Bool
  local outCrefs = stateCrefs
  (outCrefs, cont) = begin
    local state::DAE.ComponentRef
    @match exp begin
      DAE.CALL(Absyn.IDENT("der"), DAE.CREF(state) <| _ ) => begin
        #= Add state with boolean value that does not matter,
        it is later only BDAE.BACKEND_DAE(eqs = eqs) checked if it exists at all  =#
        outCrefs[state] = true
        (outCrefs, true)
      end
      _ => begin
        (outCrefs, true)
      end
    end
  end
  return (exp, cont, outCrefs)
end

"""
  kabdelhak:
    Traverses all variables and uses a hashmap to determine if a variable needs
    to be updated to be a BDAE.STATE()
"""
function updateStates(vars::Vector, stateCrefs::Dict{DAE.ComponentRef, Bool})
  local varArr::Vector{BDAE.VAR} = vars
  for i in 1:arrayLength(varArr)
    varArr[i] = begin
      local cref::DAE.ComponentRef
      local var::BDAE.Var
      @match varArr[i] begin
        var && BDAE.VAR(varName = cref) where (haskey(stateCrefs, cref)) => begin
          @assign var.varKind = BDAE.STATE(0, NONE(), true)
          var
        end
        _ => begin
          varArr[i]
        end
      end
    end
    vars = varArr
  end
  return vars
end


"""
  Author: johti17

"""
function updateArrayCrefs(vars::BDAE.Variables, arrayCrefs::Dict{DAE.ComponentRef, Bool})
  vars = begin
    @match vars begin
      BDAE.VARIABLES(varArr) => begin
        for i in 1:arrayLength(varArr)
          varArr[i] = begin
            local cref::DAE.ComponentRef
            local var::BDAE.Var
            @match varArr[i] begin
              var && BDAE.VAR(varName = cref) where (haskey(arrayCrefs, cref)) => begin
                var
              end
              _ => begin
                varArr[i]
              end
            end
          end
        end
        @assign vars.varArr = varArr
        (vars)
      end
    end
  end
end

"""
    kabdelhak:
    Residualize every equation in each system of the dae by subtracting the rhs
    from the lhs.
    (daeMode)
"""
function residualizeEveryEquation(dae::BDAE.BACKEND_DAE)
  return BDAEUtil.mapEqSystems(dae, makeResidualEquations)
end

"""
    Expand record field array variables into individual scalar element variables.
    A variable like R.T with CREF_QUAL("R", T_COMPLEX, [], CREF_IDENT("T", T_ARRAY([3,3]), []))
    is expanded into R_T[1][1], R_T[1][2], ..., R_T[3][3] as individual BDAE.VARs.
    This mirrors how the frontend scalarizes standalone array variables (e.g., w_out[3]).
"""
function expandRecordFieldArrays(dae::BDAE.BACKEND_DAE)
  for system in dae.eqs
    newVars = BDAE.VAR[]
    #= Track scalarized record field array elements to synthesize parents.
       Key: baseName, Value: (fieldType, dims, elementBindings, templateVar) =#
    scalarizedGroups = Dict{String, Tuple{DAE.Type, Vector{Int}, Dict{Tuple, Any}, BDAE.VAR}}()
    for v in system.orderedVars
      local found = findRecordFieldArray(v.varName)
      if found !== nothing && isAlreadyScalarizedCref(v.varName)
        #= Already scalarized by the frontend: pass through unchanged,
           but record the element for parent reconstruction. =#
        push!(newVars, v)
        local (baseName, fieldType, _, dims) = found
        local isParam = v.varKind isa BDAE.PARAM || v.varKind isa BDAE.CONST
        if isParam
          local dimSizes = Int[]
          for d in dims
            @match d begin
              DAE.DIM_INTEGER(size) => push!(dimSizes, size)
              _ => nothing
            end
          end
          if !isempty(dimSizes)
            if !haskey(scalarizedGroups, baseName)
              scalarizedGroups[baseName] = (fieldType, dimSizes, Dict{Tuple, Any}(), v)
            end
            local indices = extractInnermostSubscripts(v.varName)
            if indices !== nothing
              @match v.bindExp begin
                SOME(bindExp) => begin
                  scalarizedGroups[baseName][3][indices] = bindExp
                end
                _ => nothing
              end
            end
          end
        end
      else
        expanded = expandSingleRecordFieldVar(v)
        append!(newVars, expanded)
      end
    end
    #= Synthesize parent ARRAY_PARAMETERs from scalarized groups =#
    for (baseName, (fieldType, dimSizes, elemBindings, templateVar)) in scalarizedGroups
      local parentExists = any(v -> begin
        @match v.varName begin
          DAE.CREF_IDENT(n, _, MetaModelica.Nil(__)) where n == baseName => true
          _ => false
        end
      end, newVars)
      if parentExists
        continue
      end
      local arrayBind = reconstructArrayBinding(dimSizes, elemBindings; baseName = baseName)
      local parentCr = DAE.CREF_IDENT(baseName, fieldType, MetaModelica.nil)
      pushfirst!(newVars, BDAE.VAR(parentCr, templateVar.varKind, templateVar.varDirection,
                                   fieldType, arrayBind, templateVar.arryDim, templateVar.source,
                                   templateVar.values, templateVar.tearingSelectOption,
                                   templateVar.connectorType, templateVar.unreplaceable))
    end
    system.orderedVars = newVars
  end
  return dae
end

"""Check if the innermost CREF_IDENT in a CREF chain has subscripts (already scalarized)."""
Base.@nospecializeinfer function isAlreadyScalarizedCref(@nospecialize(cr::DAE.ComponentRef))::Bool
  @match cr begin
    DAE.CREF_QUAL(_, _, _, inner) => isAlreadyScalarizedCref(inner)
    DAE.CREF_IDENT(_, _, subs) => begin
      try
        return !isempty(collect(subs))
      catch
        return false
      end
    end
    _ => false
  end
end

"""Extract integer subscript indices from the innermost CREF_IDENT."""
Base.@nospecializeinfer function extractInnermostSubscripts(@nospecialize(cr::DAE.ComponentRef))
  @match cr begin
    DAE.CREF_QUAL(_, _, _, inner) => extractInnermostSubscripts(inner)
    DAE.CREF_IDENT(_, _, subs) => begin
      local indices = Int[]
      for s in subs
        @match s begin
          DAE.INDEX(DAE.ICONST(i)) => push!(indices, i)
          _ => return nothing
        end
      end
      return isempty(indices) ? nothing : tuple(indices...)
    end
    _ => nothing
  end
end

"""Reconstruct a DAE.ARRAY binding from scalarized element bindings.
   Emits a warning when only a subset of elements have bindings, because
   the missing elements are silently filled with 0.0. That default follows
   Modelica semantics for unbound parameters, but mixed-binding scenarios
   can otherwise mask a frontend extraction bug.
"""
function reconstructArrayBinding(dimSizes::Vector{Int}, elemBindings::Dict;
                                 baseName::String = "")
  if isempty(elemBindings)
    return NONE()
  end
  local expectedCount = prod(dimSizes)
  if length(elemBindings) < expectedCount
    local missingIndices = Tuple[]
    _collectMissingIndices!(missingIndices, dimSizes, 1, elemBindings, ())
    @warn "Causalize.reconstructArrayBinding: $(length(missingIndices)) of $(expectedCount) element bindings missing for array '$(baseName)'; defaulting to 0.0" missing_indices=missingIndices
  end
  local arr = buildArrayFromBindings(dimSizes, 1, elemBindings, ())
  return arr === nothing ? NONE() : SOME(arr)
end

function _collectMissingIndices!(out::Vector{Tuple}, dimSizes::Vector{Int}, dimIdx::Int,
                                 elemBindings::Dict, prefix::Tuple)
  if dimIdx > length(dimSizes)
    if !haskey(elemBindings, prefix)
      push!(out, prefix)
    end
    return
  end
  for i in 1:dimSizes[dimIdx]
    _collectMissingIndices!(out, dimSizes, dimIdx + 1, elemBindings, (prefix..., i))
  end
end

function buildArrayFromBindings(dimSizes::Vector{Int}, dimIdx::Int,
                                elemBindings::Dict, prefix::Tuple)
  if dimIdx > length(dimSizes)
    return get(elemBindings, prefix, DAE.RCONST(0.0))
  end
  local n = dimSizes[dimIdx]
  local elems = DAE.Exp[]
  for i in 1:n
    local newPrefix = (prefix..., i)
    local elem = buildArrayFromBindings(dimSizes, dimIdx + 1, elemBindings, newPrefix)
    if elem === nothing
      push!(elems, DAE.RCONST(0.0))
    else
      push!(elems, elem)
    end
  end
  local innerTy = if dimIdx < length(dimSizes)
    DAE.T_ARRAY(DAE.T_REAL(MetaModelica.nil),
                list((DAE.DIM_INTEGER(dimSizes[j]) for j in (dimIdx+1):length(dimSizes))...))
  else
    DAE.T_REAL(MetaModelica.nil)
  end
  local arrayTy = DAE.T_ARRAY(innerTy, list(DAE.DIM_INTEGER(n)))
  return DAE.ARRAY(arrayTy, false, MetaModelica.list(elems...))
end

"""
    Walk a CREF chain to find the T_COMPLEX -> T_ARRAY boundary at any depth.
    Returns (baseName, fieldType, elemTy, dims) or nothing if no such boundary exists.
    For comp[1].R.T where R is T_COMPLEX and T is T_ARRAY, the baseName is "comp[1]_R_T".
"""
Base.@nospecializeinfer function findRecordFieldArray(@nospecialize(cr::DAE.ComponentRef))
  @match cr begin
    #= Direct match: record.field where record is T_COMPLEX and field is T_ARRAY =#
    DAE.CREF_QUAL(ident, DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), _, _), outerSubs,
                  DAE.CREF_IDENT(fieldName, fieldType && DAE.T_ARRAY(elemTy, dims), _)) => begin
      local baseName = string(ident) * subscriptListToString(outerSubs) * OMBackend.COMPONENT_SEPARATOR * string(fieldName)
      return (baseName, fieldType, elemTy, dims)
    end
    #= Recursive case: prefix.rest where rest contains the record boundary deeper down =#
    DAE.CREF_QUAL(ident, _, outerSubs, innerCref) => begin
      local inner = findRecordFieldArray(innerCref)
      if inner !== nothing
        local (innerBase, fieldType, elemTy, dims) = inner
        local baseName = string(ident) * subscriptListToString(outerSubs) * OMBackend.COMPONENT_SEPARATOR * innerBase
        return (baseName, fieldType, elemTy, dims)
      end
      return nothing
    end
    _ => return nothing
  end
end

"""
    Extract a single element from a DAE.ARRAY binding expression at the given indices.
    For 1D: indices = (2,) extracts the 2nd element.
    For 2D: indices = (2,3) extracts row 2, column 3.
    Returns the element expression, or nothing if the binding is not a literal array.
"""
Base.@nospecializeinfer function extractBindingElement(@nospecialize(bindExp::DAE.Exp), indices::NTuple{N, Int}) where N
  if N == 0
    return bindExp
  end
  @match bindExp begin
    DAE.ARRAY(array = elements) => begin
      local elemArray = collect(elements)
      local idx = indices[1]
      if idx < 1 || idx > length(elemArray)
        return nothing
      end
      if N == 1
        return elemArray[idx]
      else
        return extractBindingElement(elemArray[idx], ntuple(i -> indices[i + 1], N - 1))
      end
    end
    _ => begin
      #= Non-literal binding: wrap with ASUB for runtime indexing =#
      local asubSubs = list((DAE.ICONST(indices[i]) for i in 1:N)...)
      return DAE.ASUB(bindExp, asubSubs)
    end
  end
end

"""
    If a variable is a record field with array type (CREF_QUAL with T_COMPLEX base
    and T_ARRAY inner at any depth), expand it into individual scalar element variables.
    Otherwise return the variable unchanged in a vector.
"""
function expandSingleRecordFieldVar(v::BDAE.VAR)::Vector{BDAE.VAR}
  local found = findRecordFieldArray(v.varName)
  if found === nothing
    return [v]
  end
  local (baseName, fieldType, elemTy, dims) = found
  #= Extract integer dimensions =#
  local dimSizes = Int[]
  for d in dims
    @match d begin
      DAE.DIM_INTEGER(size) => push!(dimSizes, size)
      _ => begin
        @warn "Non-integer dimension in record field array, skipping expansion"
        return [v]
      end
    end
  end
  local result = BDAE.VAR[]
  #= Keep a synthetic parent array variable so that the base array name appears
     in the simcode hash table as ARRAY_PARAMETER. This enables:
     1. createArrayParametersMTK to generate a concrete local assignment
     2. tryHandleSubscriptedArrayCref to resolve subscripts to concrete values
     Without this, function call arguments assembled from scalarized @parameter
     elements would be symbolic, causing failures in functions with control flow.
     Only create the parent for parameters/constants: regular variables are fully
     represented by their scalarized children. Keeping the parent for variables
     inflates the variable count and breaks equation-variable matching. =#
  local isParam = v.varKind isa BDAE.PARAM || v.varKind isa BDAE.CONST
  if isParam
    local parentCr = DAE.CREF_IDENT(baseName, fieldType, MetaModelica.nil)
    push!(result, BDAE.VAR(parentCr, v.varKind, v.varDirection, fieldType,
                           v.bindExp, v.arryDim, v.source, v.values,
                           v.tearingSelectOption, v.connectorType, v.unreplaceable))
  end
  #= Generate all index combinations for N dimensions =#
  for ci in CartesianIndices(ntuple(i -> dimSizes[i], length(dimSizes)))
    local subs = list((DAE.INDEX(DAE.ICONST(ci[i])) for i in 1:length(dimSizes))...)
    local cr = DAE.CREF_IDENT(baseName, fieldType, subs)
    #= Extract element binding from the array binding expression =#
    local elemBind = @match v.bindExp begin
      SOME(bindExp) => begin
        local elem = extractBindingElement(bindExp, ntuple(i -> ci[i], length(dimSizes)))
        elem === nothing ? NONE() : SOME(elem)
      end
      NONE() => NONE()
      _ => NONE()
    end
    push!(result, BDAE.VAR(cr, v.varKind, v.varDirection, elemTy,
                           elemBind, v.arryDim, v.source, v.values,
                           v.tearingSelectOption, v.connectorType, v.unreplaceable))
  end
  return result
end

"""
    Expand COMPLEX_EQUATIONs and ARRAY_EQUATIONs into multiple scalar EQUATION objects.
    An array equation like:
      {result[1], result[2], result[3]} = transformVector(...)
    is expanded to:
      result[1] = transformVector(...)[1]
      result[2] = transformVector(...)[2]
      result[3] = transformVector(...)[3]
"""
function expandComplexEquations(dae::BDAE.BACKEND_DAE)
  for system in dae.eqs
    newEqs = BDAE.Equation[]
    for eq in system.orderedEqs
      @match eq begin
        BDAE.COMPLEX_EQUATION(size, left, right, source, attr) => begin
          expanded = tryExpandRecordEquation(left, right, source, attr)
          if expanded !== nothing
            append!(newEqs, expanded)
          else
            expanded = expandSingleArrayEquation(size, left, right, source, attr)
            append!(newEqs, expanded)
          end
        end
        BDAE.ARRAY_EQUATION(dimSize, left, right, source, attr, _) => begin
          expanded = expandArrayEquationWithDims(dimSize, left, right, source, attr)
          append!(newEqs, expanded)
        end
        _ => begin
          push!(newEqs, eq)
        end
      end
    end
    system.orderedEqs = newEqs
  end
  return dae
end

"""
    Expand a record equation (R = func(...)) into fully scalar equations.
    When the LHS is a CREF with T_COMPLEX type, extracts the field names from
    the record type and generates per-element equations using the flattened naming
    convention with CREF subscripts (e.g. CREF_IDENT("R_rot_T", T_REAL, [1][1]))
    so that the code generator resolves them to scalarized variable names.
    Returns nothing if the equation is not a record equation.
"""
function tryExpandRecordEquation(left::DAE.Exp, right::DAE.Exp,
                                  source::DAE.ElementSource, attr::BDAE.EquationAttributes)
  #= Extract (baseName, T_COMPLEX type, subscript-string) from a CREF
     (CREF_IDENT or CREF_QUAL). The subscript-string is the flat-name
     representation of any array indices on the LHS, e.g. "[3]" for
     `transferFunction_aw[3]`. We need this so a Complex-array element
     equation like `transferFunction_aw[3] = expr` produces field equations
     with subscripted LHS names (`transferFunction_aw[3]_re`,
     `transferFunction_aw[3]_im`) matching the scalarized HT keys, not
     the bare names that would collapse three array entries into one. =#
  function _extractCrefBase(e)
    @match e begin
      DAE.CREF(DAE.CREF_IDENT(name, _, subs), t) where {t isa DAE.T_COMPLEX} => begin
        (name, t, subscriptListToString(subs))
      end
      DAE.CREF(cr::DAE.CREF_QUAL, t) where {t isa DAE.T_COMPLEX} => begin
        (flatName, _, leafSubs) = crefToFlatName(cr)
        (flatName, t, subscriptListToString(leafSubs))
      end
      _ => nothing
    end
  end

  #= Try LHS first; fall back to RHS if LHS isn't a record CREF. The latter
     case occurs for `func(...) = recordCref` shapes (e.g. Magnetic.FW
     `subtract(port_p.V_m, port_n.V_m) = converter.V_m`). Scalarising the
     CREF side here produces `recordCref_field = exprSide[i]` instead of
     leaving `recordCref[i]` references that the simvar HT cannot resolve. =#
  local extracted = _extractCrefBase(left)
  local crefOnLeft = true
  if extracted === nothing
    extracted = _extractCrefBase(right)
    crefOnLeft = false
  end
  extracted === nothing && return nothing
  (baseName, ty, subsStr) = extracted
  local exprSide = crefOnLeft ? right : left

  equations = BDAE.Equation[]
  fieldIdx = 0
  for field in ty.varLst
    fieldIdx += 1
    #= Embed any LHS array subscript directly in the field name string so the
       resulting flat name matches the scalarized HT key:
         transferFunction_aw[3] (Complex) ⇒ transferFunction_aw[3]_re, _im
       (not the bare `transferFunction_aw_re` that collapses all three
       array entries.) =#
    fieldName = string(baseName, subsStr, OMBackend.COMPONENT_SEPARATOR, field.name)
    fieldTy = field.ty
    exprField = DAE.ASUB(exprSide, list(DAE.ICONST(fieldIdx)))
    @match fieldTy begin
      DAE.T_ARRAY(elemTy, dims) => begin
        dimVec = Int[d.integer for d in dims]
        for idx in CartesianIndices(Tuple(dimVec))
          idxTuple = Tuple(idx)
          subs = list((DAE.INDEX(DAE.ICONST(i)) for i in idxTuple)...)
          lhsCref = DAE.CREF_IDENT(fieldName, elemTy, subs)
          lhsExp = DAE.CREF(lhsCref, elemTy)
          rhsExp = DAE.ASUB(exprField, list((DAE.ICONST(i) for i in idxTuple)...))
          push!(equations, BDAE.EQUATION(lhsExp, rhsExp, source, attr))
        end
      end
      _ => begin
        fieldCref = DAE.CREF_IDENT(fieldName, fieldTy, MetaModelica.nil)
        fieldExp = DAE.CREF(fieldCref, fieldTy)
        push!(equations, BDAE.EQUATION(fieldExp, exprField, source, attr))
      end
    end
  end
  return equations
end

"""
    Expand a single array equation into multiple scalar EQUATION objects.
"""
function expandSingleArrayEquation(size::Int, left::DAE.Exp, right::DAE.Exp,
                                    source::DAE.ElementSource, attr::BDAE.EquationAttributes)
  equations = BDAE.Equation[]
  leftFlat = flattenDAEArray(left)
  rightFlat = flattenDAEArray(right)
  for i in 1:size
    local subs = list(DAE.ICONST(i))
    leftElem = if leftFlat !== nothing && i <= length(leftFlat)
      leftFlat[i]
    else
      makeScalarElement(left, subs)
    end
    rightElem = if rightFlat !== nothing && i <= length(rightFlat)
      rightFlat[i]
    else
      makeScalarElement(right, subs)
    end
    push!(equations, BDAE.EQUATION(leftElem, rightElem, source, attr))
  end
  return equations
end

"""
    Given a DAE.Exp and integer subscripts, produce a scalar element expression.
    If the expression is a CREF (possibly CREF_QUAL), flatten the entire CREF chain
    into a CREF_IDENT with the subscripts baked in. This ensures expanded array
    equations use scalarized variable names (var"name[i]") instead of bare symbol
    indexing (name[i]) which would fail at runtime with MTK's scalar Num variables.
    Falls back to ASUB wrapping for non-CREF expressions.
"""
Base.@nospecializeinfer function makeScalarElement(@nospecialize(exp::DAE.Exp), subscripts::Union{Cons{DAE.ICONST}, Cons{DAE.Exp}})
  @match exp begin
    DAE.CREF(cr, _) => begin
      local (flatName, crefTy, existingSubs) = crefToFlatName(cr)
      #= Collect existing subs into array, append new ones, convert to list once =#
      local subsArr = DAE.Subscript[s for s in existingSubs]
      for s in subscripts
        @match s begin
          DAE.ICONST(i) => push!(subsArr, DAE.INDEX(DAE.ICONST(i)))
          _ => push!(subsArr, DAE.INDEX(s))
        end
      end
      local allSubs = list(subsArr...)
      #= After subscripting, unwrap T_ARRAY to get the element type.
         For each new subscript dimension consumed, peel one T_ARRAY layer. =#
      local scalarTy = crefTy
      for _ in subscripts
        scalarTy = @match scalarTy begin
          DAE.T_ARRAY(ty = inner) => inner
          _ => scalarTy
        end
      end
      local newCref = DAE.CREF_IDENT(flatName, scalarTy, allSubs)
      DAE.CREF(newCref, scalarTy)
    end
    _ => DAE.ASUB(exp, subscripts)
  end
end

"""
    Expand an array equation with known dimensions into scalar equations.
    For multi-dimensional arrays, generates proper [i,j,...] indexing.
"""
function expandArrayEquationWithDims(dimSize::Vector, left::DAE.Exp, right::DAE.Exp,
                                      source::DAE.ElementSource, attr::BDAE.EquationAttributes)
  equations = BDAE.Equation[]
  leftFlat = flattenDAEArray(left)
  rightFlat = flattenDAEArray(right)

  if length(dimSize) == 1
    #= 1D array: use single index =#
    for i in 1:dimSize[1]
      local subs1 = list(DAE.ICONST(i))
      leftElem = if leftFlat !== nothing && i <= length(leftFlat)
        leftFlat[i]
      else
        makeScalarElement(left, subs1)
      end
      rightElem = if rightFlat !== nothing && i <= length(rightFlat)
        rightFlat[i]
      else
        makeScalarElement(right, subs1)
      end
      push!(equations, BDAE.EQUATION(leftElem, rightElem, source, attr))
    end
  elseif length(dimSize) == 2
    #= 2D array: use [row, col] indexing =#
    flatIdx = 1
    for i in 1:dimSize[1]
      for j in 1:dimSize[2]
        local subs2 = list(DAE.ICONST(i), DAE.ICONST(j))
        leftElem = if leftFlat !== nothing && flatIdx <= length(leftFlat)
          leftFlat[flatIdx]
        else
          makeScalarElement(left, subs2)
        end
        rightElem = if rightFlat !== nothing && flatIdx <= length(rightFlat)
          rightFlat[flatIdx]
        else
          makeScalarElement(right, subs2)
        end
        push!(equations, BDAE.EQUATION(leftElem, rightElem, source, attr))
        flatIdx += 1
      end
    end
  else
    #= Higher dimensions: fall back to flat indexing =#
    size = prod(dimSize)
    return expandSingleArrayEquation(size, left, right, source, attr)
  end
  return equations
end

"""
    Recursively flatten a DAE.ARRAY (possibly nested) into a flat Vector of leaf expressions.
    For non-array expressions (CAST, TUPLE, etc.), unwraps where possible.
    Returns nothing if the expression cannot be flattened into individual elements.
"""
Base.@nospecializeinfer function flattenDAEArray(@nospecialize(exp::DAE.Exp))::Union{Vector{DAE.Exp}, Nothing}
  @match exp begin
    DAE.ARRAY(_, _, elements) => begin
      result = DAE.Exp[]
      for elem in elements
        inner = flattenDAEArray(elem)
        if inner !== nothing
          append!(result, inner)
        else
          push!(result, elem)
        end
      end
      return result
    end
    DAE.TUPLE(elements) => begin
      return collect(elements)
    end
    DAE.CAST(_, innerExp) => begin
      return flattenDAEArray(innerExp)
    end
    DAE.REDUCTION(reductionInfo, bodyExp, iterators) => begin
      expanded = tryExpandReduction(reductionInfo, bodyExp, iterators)
      if expanded !== nothing
        return expanded
      end
      return nothing
    end
    _ => nothing
  end
end

"""
    Try to expand a DAE.REDUCTION with \"array\" path and a single iterator
    over a constant integer range into a flat vector of DAE.Exp elements.
    Returns nothing if expansion is not possible.
"""
function tryExpandReduction(reductionInfo, bodyExp::DAE.Exp, iterators)
  @match reductionInfo.path begin
    Absyn.IDENT("array") => nothing  # continue below
    _ => return nothing
  end
  return _expandReductionElems(bodyExp, iterators)
end

"""
    Replace occurrences of an iterator variable in a DAE expression with
    a constant integer value. Traverses the expression tree and replaces
    CREF(CREF_IDENT(iterId, ...)) with ICONST(value).
"""
function substituteIteratorInExp(exp::DAE.Exp, iterId::String, value::Int)::DAE.Exp
  function replacer(e::DAE.Exp, arg)
    @match e begin
      DAE.CREF(DAE.CREF_IDENT(id, _, _), _) where (id == iterId) => begin
        (DAE.ICONST(value), arg)
      end
      _ => (e, arg)
    end
  end
  return first(Util.traverseExpBottomUp(exp, replacer, nothing))
end

#= Expand a single-iterator reduction body over a constant integer range.
   Returns the per-index substituted expressions, or nothing when the range
   is not statically known. =#
function _expandReductionElems(bodyExp::DAE.Exp, iterators)::Union{Vector{DAE.Exp}, Nothing}
  length(iterators) == 1 || return nothing
  local iterId::String = ""
  local rangeExp = nothing
  @match first(iterators) begin
    DAE.REDUCTIONITER(id, rExp, _, _) => begin
      iterId = id
      rangeExp = rExp
    end
    _ => return nothing
  end
  local startVal::Int = 0
  local stepVal::Int = 1
  local stopVal::Int = 0
  @match rangeExp begin
    DAE.RANGE(_, DAE.ICONST(s), NONE(), DAE.ICONST(e)) => begin
      startVal = s
      stopVal = e
    end
    DAE.RANGE(_, DAE.ICONST(s), SOME(DAE.ICONST(st)), DAE.ICONST(e)) => begin
      startVal = s
      stepVal = st
      stopVal = e
    end
    _ => return nothing
  end
  stepVal == 0 && return nothing
  return DAE.Exp[substituteIteratorInExp(bodyExp, iterId, i) for i in startVal:stepVal:stopVal]
end

function _callAttrType(@nospecialize(attr))
  @match attr begin
    DAE.CALL_ATTR(ty = ty) => ty
    _ => DAE.T_REAL_DEFAULT
  end
end

#= Fold expanded reduction elements into a scalar expression: sum / product
   as operator chains, min / max as nested two-argument calls. =#
function _foldReductionScalar(fname::String, elems::Vector{DAE.Exp}, attr,
                              foldTy::DAE.Type)::Union{DAE.Exp, Nothing}
  isempty(elems) && return nothing
  local acc::DAE.Exp = elems[1]
  for k in 2:length(elems)
    acc = if fname == "sum"
      DAE.BINARY(acc, DAE.ADD(foldTy), elems[k])
    elseif fname == "product"
      DAE.BINARY(acc, DAE.MUL(foldTy), elems[k])
    elseif fname == "min" || fname == "max"
      DAE.CALL(Absyn.IDENT(fname), list(acc, elems[k]), attr)
    else
      return nothing
    end
  end
  return acc
end

#= Elements of a reducing call's single argument: an array-path reduction
   over a constant range, or an array constructor. =#
function _reducingCallArgElems(@nospecialize(arg))::Union{Vector{DAE.Exp}, Nothing}
  @match arg begin
    DAE.REDUCTION(reductionInfo, bodyExp, iterators) => begin
      @match reductionInfo.path begin
        Absyn.IDENT("array") => _expandReductionElems(bodyExp, iterators)
        _ => nothing
      end
    end
    DAE.ARRAY(_, _, elements) => collect(elements)
    _ => nothing
  end
end

function unrollReductionTraverser(exp::DAE.Exp, acc)
  local newExp::DAE.Exp = exp
  @match exp begin
    DAE.CALL(Absyn.IDENT(fname), args, attr) where (fname == "max" || fname == "min" ||
                                                    fname == "sum" || fname == "product") => begin
      local argv = collect(args)
      if length(argv) == 1
        local elems = _reducingCallArgElems(argv[1])
        if elems !== nothing
          local folded = _foldReductionScalar(fname, elems, attr, _callAttrType(attr))
          folded === nothing || (newExp = folded)
        end
      end
      ()
    end
    DAE.REDUCTION(reductionInfo, bodyExp, iterators) => begin
      local pathName = @match reductionInfo.path begin
        Absyn.IDENT(n) => n
        _ => ""
      end
      local elems = _expandReductionElems(bodyExp, iterators)
      if elems !== nothing
        if pathName == "array"
          newExp = DAE.ARRAY(reductionInfo.exprType, true, list(elems...))
        elseif pathName == "sum" || pathName == "product" || pathName == "min" || pathName == "max"
          local attr = DAE.CALL_ATTR(reductionInfo.exprType, false, true, false, false,
                                     DAE.NO_INLINE(), DAE.NO_TAIL())
          local folded = _foldReductionScalar(pathName, elems, attr, reductionInfo.exprType)
          folded === nothing || (newExp = folded)
        end
      end
      ()
    end
    _ => ()
  end
  return (newExp, true, acc)
end

"""
  Unroll reductions whose single iterator spans a constant integer range, so
  no symbolic iterator subscript survives into SimCode (SimCref carries
  integer subscripts only). Reducing calls over array constructors fold to
  the same scalar form.
"""
function unrollConstantReductions(dae::BDAE.BACKEND_DAE)
  BDAEUtil.mapEqSystems(dae, unrollConstantReductionsSystem)
end

function unrollConstantReductionsSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  for eqs in (syst.orderedEqs, syst.initialEqs)
    for i in 1:length(eqs)
      (eq2, _) = BDAEUtil.traverseEquationExpressions(eqs[i], unrollReductionTraverser, nothing)
      eqs[i] === eq2 || (eqs[i] = eq2)
    end
  end
  return syst
end

"""
    kabdelhak:
    Traverser for daeMode() to map all equations of an equation system
"""
function makeResidualEquations(syst::BDAE.EQSYSTEM)
  syst = BDAEUtil.mapEqSystemEquations(syst, BackendEquation.makeResidualEquation)
end


"""
  Transform ASUB expressions where the inner expression is a der() call.
  This transforms: der(array)[i] → der(array[i])
  This is mathematically valid since differentiation distributes over array elements.
"""
function transformASUBExpressions(dae::BDAE.BACKEND_DAE)
  BDAEUtil.mapEqSystems(dae, transformASUBEqSystem)
end

"""
  Apply ASUB transformation to an equation system.
"""
function transformASUBEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  syst = begin
    @match syst begin
      BDAE.EQSYSTEM(__) => begin
        for i in 1:length(syst.orderedEqs)
          local eq = syst.orderedEqs[i]
          (eq2, _) = BDAEUtil.traverseEquationExpressions(eq, transformASUBInDer, nothing)
          if !(eq === eq2)
            @assign syst.orderedEqs[i] = eq2
          end
        end
        syst
      end
    end
  end
  return syst
end

"""
  Simplify ASUB(ARRAY([e1, e2, ...]), [ICONST(i)]) → e_i
  When subscripting into an array constructor with a constant index, return that element directly.
"""
function simplifyASUBofARRAY(asub::DAE.Exp)::DAE.Exp
  @match asub begin
    #= ASUB with a single integer constant subscript into an ARRAY constructor =#
    DAE.ASUB(DAE.ARRAY(array = elements), Cons(DAE.ICONST(idx), Nil())) => begin
      #= Convert the list to an array to access by index =#
      elemArray = collect(elements)
      if idx >= 1 && idx <= length(elemArray)
        return elemArray[idx]
      else
        @warn "ASUB index $idx out of bounds for array of length $(length(elemArray))"
        return asub
      end
    end
    #= Also handle INDEX wrapped subscripts =#
    DAE.ASUB(DAE.ARRAY(array = elements), Cons(DAE.INDEX(DAE.ICONST(idx)), Nil())) => begin
      elemArray = collect(elements)
      if idx >= 1 && idx <= length(elemArray)
        return elemArray[idx]
      else
        @warn "ASUB index $idx out of bounds for array of length $(length(elemArray))"
        return asub
      end
    end
    #= Not an ASUB(ARRAY, const) pattern - return unchanged =#
    _ => asub
  end
end

"""
  Transform ASUB(CALL(der, [array]), subscripts) → CALL(der, [ASUB(array, subscripts)])
  This pushes the subscript inside the der() call.
  Also simplifies ASUB(ARRAY([e1, e2, ...]), i) → e_i
"""
function transformASUBInDer(exp::DAE.Exp, acc)
  (newExp, cont, acc) = begin
    @match exp begin
      #= Transform der(array)[subscripts] → der(array[subscripts]) =#
      DAE.ASUB(DAE.CALL(Absyn.IDENT("der"), Cons(arrayExp, Nil()), attr), subscripts) => begin
        #= Create the subscripted array expression =#
        innerASUB = DAE.ASUB(arrayExp, subscripts)
        #= Try to simplify if the inner expression is an ARRAY constructor =#
        simplifiedInner = simplifyASUBofARRAY(innerASUB)
        #= Wrap with der() call =#
        newDer = DAE.CALL(Absyn.IDENT("der"), list(simplifiedInner), attr)
        (newDer, true, acc)
      end
      #= Also simplify standalone ASUB(ARRAY([...]), i) expressions =#
      DAE.ASUB(DAE.ARRAY(__), _) => begin
        simplified = simplifyASUBofARRAY(exp)
        (simplified, true, acc)
      end
      _ => (exp, true, acc)
    end
  end
  return (newExp, cont, acc)
end

"""
  Resolve CREF bindings to their actual values.
  When a variable's binding is a CREF pointing to another variable,
  replace it with that variable's binding. Handles chains by iterating until stable.
"""
function resolveCrefBindings!(orderedVars::Vector{BDAE.VAR})
  local bindingMap = Dict{String, DAE.Exp}()
  for v in orderedVars
    local varName = string(v.varName)
    @match v.bindExp begin
      SOME(bindExp) => begin
        bindingMap[varName] = bindExp
      end
      NONE() => ()
    end
  end
  local changed = true
  local maxIterations = 100
  local iteration = 0
  while changed && iteration < maxIterations
    changed = false
    iteration += 1
    for i in 1:length(orderedVars)
      local bindExp = orderedVars[i].bindExp
      @match bindExp begin
        SOME(DAE.CREF(cr, _)) => begin
          (targetName, _, _) = crefToFlatName(cr)
          local targetBinding = get(bindingMap, targetName, nothing)
          if targetBinding !== nothing
            if !(targetBinding isa DAE.CREF)
              orderedVars[i].bindExp = SOME(targetBinding)
              local varName = string(orderedVars[i].varName)
              bindingMap[varName] = targetBinding
              changed = true
            end
          end
        end
        _ => ()
      end
    end
  end
  if iteration >= maxIterations
    @warn "resolveCrefBindings! reached maximum iterations ($maxIterations). This may indicate circular references in parameter bindings."
  end
end

"""
  Transform CREF_QUAL structures with subscripted array finals into flat CREF_IDENT.
  This ensures that CREFs in equations match the scalarized variable names in the hash table.

  Example: CREF_QUAL("a", ..., CREF_IDENT("b", T_ARRAY, [1,2])) → CREF_IDENT("a_b[1][2]", T_REAL, [])
"""
function flattenArrayCrefs(dae::BDAE.BACKEND_DAE)
  BDAEUtil.mapEqSystems(dae, flattenArrayCrefsEqSystem)
end

"""
  Apply CREF flattening transformation to an equation system.
  Transforms both equations and variable bindings.
"""
function flattenArrayCrefsEqSystem(syst::BDAE.EQSYSTEM)::BDAE.EQSYSTEM
  #= CREFs already contain the full component path, no prefix needed =#
  local prefix = ""
  #= Create a closure that captures the prefix =#
  flattenWithPrefix = (exp, acc) -> flattenArrayCrefInExp(exp, acc, prefix)
  syst = begin
    @match syst begin
      BDAE.EQSYSTEM(__) => begin
        #= Transform equations =#
        for i in 1:length(syst.orderedEqs)
          local eq = syst.orderedEqs[i]
          (eq2, _) = BDAEUtil.traverseEquationExpressions(eq, flattenWithPrefix, nothing)
          if !(eq === eq2)
            @assign syst.orderedEqs[i] = eq2
          end
        end
        #= Transform variable bindings =#
        for i in 1:length(syst.orderedVars)
          local v = syst.orderedVars[i]
          @match v.bindExp begin
            SOME(bindingExp) => begin
              #= traverseExpTopDown applies the func to the root first, then
                 descends, so a separate root-only call is redundant. =#
              (newBindExp, _) = Util.traverseExpTopDown(bindingExp, flattenWithPrefix, nothing)
              if !(bindingExp === newBindExp)
                @assign syst.orderedVars[i].bindExp = SOME(newBindExp)
              end
            end
            NONE() => ()
          end
        end
        #= Resolve CREF bindings to actual values =#
        resolveCrefBindings!(syst.orderedVars)
        syst
      end
    end
  end
  return syst
end

"""
  Helper to build subscript string from a subscript list.
  Converts [INDEX(ICONST(1)), INDEX(ICONST(2))] → "[1][2]"
"""
function subscriptListToString(subscriptLst::List{DAE.Subscript})::String
  #= Common case is an empty subscript list -> "" without touching IOBuffer. =#
  listEmpty(subscriptLst) && return ""
  local buf = IOBuffer()
  for s in subscriptLst
    @match s begin
      DAE.INDEX(DAE.ICONST(i)) => begin
        print(buf, "[", i, "]")
      end
      DAE.INDEX(exp) => begin
        #= Non-constant subscript - keep as is =#
        print(buf, "[", exp, "]")
      end
      DAE.WHOLEDIM() => begin
        print(buf, "[:]")
      end
      _ => begin
        @warn "Unhandled subscript type in flattenArrayCrefInExp: $(typeof(s))"
      end
    end
  end
  return String(take!(buf))
end

"""
  Build the full flattened name from a CREF_QUAL chain.
  Recursively processes the chain and builds "prefix_ident_ident_...[subscripts]"
  The prefix is prepended only at the top level (when building the final name).
"""
function crefToFlatName(cref::DAE.ComponentRef, prefix::String="")::Tuple{String, DAE.Type, List{DAE.Subscript}}
  @match cref begin
    DAE.CREF_IDENT(ident, identType, subscriptLst) => begin
      #= Only extract element type if there are subscripts; otherwise keep array type =#
      resultType = if !isempty(subscriptLst)
        @match identType begin
          DAE.T_ARRAY(ty = ty) => ty
          _ => identType
        end
      else
        identType
      end
      baseName = isempty(prefix) ? ident : string(prefix, OMBackend.COMPONENT_SEPARATOR, ident)
      (baseName, resultType, subscriptLst)
    end
    DAE.CREF_QUAL(ident, identType, subscriptLst, componentRef) => begin
      (restName, elementType, finalSubscripts) = crefToFlatName(componentRef, "")
      subsStr = subscriptListToString(subscriptLst)
      baseName = isempty(prefix) ? string(ident, subsStr, OMBackend.COMPONENT_SEPARATOR, restName) : string(prefix, OMBackend.COMPONENT_SEPARATOR, ident, subsStr, OMBackend.COMPONENT_SEPARATOR, restName)
      (baseName, elementType, finalSubscripts)
    end
    _ => begin
      @warn "Unhandled CREF type in crefToFlatName: $(typeof(cref))"
      (string(cref), DAE.T_REAL_DEFAULT, MetaModelica.nil)
    end
  end
end

"""
  Check if a CREF has a final array component with subscripts.
  This identifies CREFs like a.b.c[1][2] where c has T_ARRAY type and subscripts.
"""
function hasFinalArrayWithSubscripts(cref::DAE.ComponentRef)::Bool
  result = @match cref begin
    #= Final CREF_IDENT with array type and subscripts =#
    DAE.CREF_IDENT(ident, identType, subscriptLst) => begin
      isArray = identType isa DAE.T_ARRAY
      hasSubs = !isempty(subscriptLst)
      isArray && hasSubs
    end
    #= CREF_QUAL - check the rest =#
    DAE.CREF_QUAL(ident, _, _, componentRef) => begin
      res = hasFinalArrayWithSubscripts(componentRef)
      res
    end
    _ => false
  end
  return result
end

"""
  Transform CREF expressions: flatten CREF_QUAL references with array subscripts to CREF_IDENT.
  Only flattens CREFs that have array subscripts on the final component (e.g., R.w[1]).
  CREFs without subscripts (e.g., myRecord.z) are left unchanged to preserve hash table lookup.
"""
function flattenArrayCrefInExp(exp::DAE.Exp, acc, prefix::String="")
  (newExp, cont, acc) = begin
    @match exp begin
      DAE.CREF(cr, ty) => begin
        hasArraySubs = hasFinalArrayWithSubscripts(cr)
        if hasArraySubs
          (flatName, elementType, finalSubscripts) = crefToFlatName(cr, prefix)
          newCref = DAE.CREF_IDENT(flatName, elementType, finalSubscripts)
          newExp = DAE.CREF(newCref, elementType)
          (newExp, true, acc)
        else
          (exp, true, acc)
        end
      end
      _ => (exp, true, acc)
    end
  end
  return (newExp, cont, acc)
end

"""
  Reclassify integer VARIABLE types as PARAM and remove their equations.
  Integer variables (e.g., color values for visualization) should not be part of
  the continuous ODE system. Handles both scalar T_INTEGER and T_ARRAY(T_INTEGER).
  This pass runs early, before expansion passes.
"""
function resolveIntegerVariables(dae::BDAE.BACKEND_DAE)
  for system in dae.eqs
    _resolveIntVarsInSystem!(system)
  end
  return dae
end

function _isIntegerVarType(varType)
  varType isa DAE.T_INTEGER ||
    (varType isa DAE.T_ARRAY && varType.ty isa DAE.T_INTEGER)
end

#= Discrete-parameter-like variable types: Integer, Enumeration (e.g.
   Modelica.Electrical.Digital.Interfaces.Logic 9-value-logic), and their
   array variants. These can all be reclassified from BDAE.VARIABLE to
   BDAE.PARAM when fully constrained by a constant-RHS equation, since
   none of them have a continuous derivative. Without this `auxiliary*`
   variables in Digital gates remain as VARIABLE and create unbalanced
   systems via their dummy `der(x) ~ 0` plus their definitional equation. =#
function _isIntOrEnumVarType(varType)
  varType isa DAE.T_INTEGER || varType isa DAE.T_ENUMERATION ||
    (varType isa DAE.T_ARRAY && (varType.ty isa DAE.T_INTEGER || varType.ty isa DAE.T_ENUMERATION))
end

"""
Identify (target, source) pairs in an equation whose target is fully
determined by the source via direct CREF aliasing. Used by the
alias-chain fixpoint to propagate removal: when source is already a
removed definer (constant parameter), target can also be removed and
inherits source's value.

Only the `intA = intB` shape is handled here. Propagation through
value-preserving wrappers like `pre(intB)` is done at the SimCode level
because those forms can survive Causalize and only become folding
candidates after BDAE→SimVar reclassifies intB as PARAMETER.
"""
function _candidateAliasPairs(eq::BDAE.EQUATION, intNames::OrderedSet{String})
  pairs = Tuple{String, String}[]
  if eq.lhs isa DAE.CREF && eq.rhs isa DAE.CREF
    lhsName = string(eq.lhs.componentRef)
    rhsName = string(eq.rhs.componentRef)
    if lhsName in intNames && rhsName in intNames
      push!(pairs, (lhsName, rhsName))
      push!(pairs, (rhsName, lhsName))
    end
  end
  return pairs
end

function _resolveIntVarsInSystem!(syst::BDAE.EQSYSTEM)
  #= Collect integer / enumeration VARIABLE names (scalarized, e.g.
     "world_axisColor_x[1]", "Adder_AND_G1_auxiliary[2]") and base names
     (un-indexed, e.g. "world_axisColor_x") for matching array/complex
     equations that have not been expanded yet. Enumerations are treated
     the same as integers here because Modelica.Electrical.Digital uses
     enum-typed `auxiliary` lookup variables that are constant-bound
     (DAE.ENUM_LITERAL on RHS) and would otherwise produce both a dummy
     `der(x) ~ 0` and a defining equation, unbalancing MTK. =#
  intVarNames = OrderedSet{String}()
  intVarBaseNames = OrderedSet{String}()
  for v in syst.orderedVars
    if v.varKind isa BDAE.VARIABLE && _isIntOrEnumVarType(v.varType)
      name = string(v.varName)
      push!(intVarNames, name)
      push!(intVarBaseNames, replace(name, r"\[.*\]$" => ""))
    end
  end
  isempty(intVarNames) && return
  #= Integers written inside a when-clause are DISCRETE state, not constant
     parameters. Collect the set of such names so we skip them during
     reclassification and keep their defining WHEN_EQUATIONs intact. =#
  intVarsWrittenInWhen = union(
    _collectIntegerLhsInWhen(syst.orderedEqs, intVarNames, intVarBaseNames),
    _collectIntegerLhsInAlgorithm(syst.orderedEqs, intVarNames, intVarBaseNames),
  )
  #= Scan equations: extract constant values and mark for removal.
     An equation is removed if either side references an integer variable.
     For ARRAY_EQUATION/COMPLEX_EQUATION, match against base names too. =#
  allIntNames = union(intVarNames, intVarBaseNames)
  valueMap = Dict{String, DAE.Exp}()
  keepEq = trues(length(syst.orderedEqs))
  #= A literal value used as a binding for an int/enum variable. Anything
     else (algorithm-table lookup, function call, expression) means the
     equation is doing real computation and must NOT be dropped — that
     was the Digital-gate `auxiliary[2] := and_table[...]` issue. =#
  local _isLiteral = e -> (e isa DAE.ICONST || e isa DAE.ENUM_LITERAL)
  #= Track exactly which int/enum vars had their *only* defining equation
     removed. Only those are safe to reclassify as PARAM. Variables with
     surviving non-literal defining equations (like `Adder_AND_G2_y =
     pre(auxiliary_n)`) must remain VARIABLE so MTK can compute them. =#
  local removedDefiners = OrderedSet{String}()
  for (i, eq) in enumerate(syst.orderedEqs)
    if eq isa BDAE.EQUATION
      lhsName = eq.lhs isa DAE.CREF ? string(eq.lhs.componentRef) : nothing
      rhsName = eq.rhs isa DAE.CREF ? string(eq.rhs.componentRef) : nothing
      lhsIsInt = lhsName !== nothing && lhsName in intVarNames
      rhsIsInt = rhsName !== nothing && rhsName in intVarNames
      #= If this integer is actually when-driven discrete state, leave the
         equation alone (it might be a continuous default / initial value
         assignment). =#
      isWhenDriven = (lhsIsInt && lhsName in intVarsWrittenInWhen) ||
                     (rhsIsInt && rhsName in intVarsWrittenInWhen)
      if (lhsIsInt || rhsIsInt) && !isWhenDriven
        #= Drop a `intvar = literal` binding outright; capture the value. =#
        if lhsIsInt && _isLiteral(eq.rhs)
          keepEq[i] = false
          push!(removedDefiners, lhsName)
          valueMap[lhsName] = eq.rhs
        elseif rhsIsInt && _isLiteral(eq.lhs)
          keepEq[i] = false
          push!(removedDefiners, rhsName)
          valueMap[rhsName] = eq.lhs
        end
        #= int-to-int aliases (e.g. `auxiliary_n = auxiliary[2]`) are
           handled in the alias-chain follow-up pass below — we cannot
           decide here because the RHS may not yet be in removedDefiners. =#
      end
    elseif eq isa BDAE.ARRAY_EQUATION
      leftName = eq.left isa DAE.CREF ? string(eq.left.componentRef) : nothing
      rightName = eq.right isa DAE.CREF ? string(eq.right.componentRef) : nothing
      leftIsInt = leftName !== nothing && leftName in allIntNames
      rightIsInt = rightName !== nothing && rightName in allIntNames
      isWhenDriven = (leftIsInt && leftName in intVarsWrittenInWhen) ||
                     (rightIsInt && rightName in intVarsWrittenInWhen)
      if (leftIsInt || rightIsInt) && !isWhenDriven
        if leftIsInt && _isLiteral(eq.right)
          keepEq[i] = false
          push!(removedDefiners, leftName)
        elseif rightIsInt && _isLiteral(eq.left)
          keepEq[i] = false
          push!(removedDefiners, rightName)
        end
      end
    elseif eq isa BDAE.COMPLEX_EQUATION
      leftName = eq.left isa DAE.CREF ? string(eq.left.componentRef) : nothing
      rightName = eq.right isa DAE.CREF ? string(eq.right.componentRef) : nothing
      leftIsInt = leftName !== nothing && leftName in allIntNames
      rightIsInt = rightName !== nothing && rightName in allIntNames
      isWhenDriven = (leftIsInt && leftName in intVarsWrittenInWhen) ||
                     (rightIsInt && rightName in intVarsWrittenInWhen)
      if (leftIsInt || rightIsInt) && !isWhenDriven
        if leftIsInt && _isLiteral(eq.right)
          keepEq[i] = false
          push!(removedDefiners, leftName)
        elseif rightIsInt && _isLiteral(eq.left)
          keepEq[i] = false
          push!(removedDefiners, rightName)
        end
      end
    end
  end
  #= Alias-chain fixpoint: an equation `intA = intB` where intB is already
     in `removedDefiners` (i.e. now a PARAM with known value) means intA
     can also be removed and reclassified. Iterate until no more changes
     so chains like `auxiliary_n = auxiliary[2]; auxiliary[2] = 'U'`
     fully collapse. Order-of-equations does not matter with iteration. =#
  while true
    local progress = false
    for (i, eq) in enumerate(syst.orderedEqs)
      keepEq[i] || continue
      eq isa BDAE.EQUATION || continue
      for (target, source) in _candidateAliasPairs(eq, intVarNames)
        target in intVarsWrittenInWhen && continue
        source in intVarsWrittenInWhen && continue
        if source in removedDefiners && !(target in removedDefiners)
          keepEq[i] = false
          push!(removedDefiners, target)
          if haskey(valueMap, source)
            valueMap[target] = valueMap[source]
          end
          progress = true
          break
        end
      end
    end
    progress || break
  end
  #= Reclassify integer VARIABLEs as PARAMs with extracted or default values.
     Skip any integer written inside a when-clause — that variable is
     discrete state and must stay a VARIABLE so the when-update has
     somewhere to land. =#
  nReclassified = 0
  for v in syst.orderedVars
    if v.varKind isa BDAE.VARIABLE && _isIntOrEnumVarType(v.varType)
      name = string(v.varName)
      baseName = replace(name, r"\[.*\]$" => "")
      if name in intVarsWrittenInWhen || baseName in intVarsWrittenInWhen
        continue
      end
      #= Only reclassify variables whose defining equation we actually
         removed. Variables with surviving non-literal definers (e.g.
         `Adder_AND_G2_y = pre(auxiliary_n)`) must remain VARIABLE so MTK
         continues to compute them — otherwise the gate output gets pinned
         to a constant parameter and the model produces meaningless output. =#
      if !(name in removedDefiners) && !(baseName in removedDefiners)
        continue
      end
      v.varKind = BDAE.PARAM()
      if v.varType isa DAE.T_INTEGER
        v.bindExp = SOME(get(valueMap, name, DAE.ICONST(0)))
      elseif v.varType isa DAE.T_ENUMERATION
        #= Use captured ENUM_LITERAL when the equation pinned the variable
           to a specific literal (e.g. 'U' for Logic). Fall back to ICONST(0)
           for partially-bound enum arrays — the integer index 0 is a safe
           default that downstream codegen will treat as "first literal" /
           uninitialized, matching prior behavior for unconstrained enums. =#
        v.bindExp = SOME(get(valueMap, name, DAE.ICONST(0)))
      end
      nReclassified += 1
    end
  end
  #= Remove equations that defined integer variables =#
  nRemoved = count(!, keepEq)
  if nRemoved > 0
    syst.orderedEqs = syst.orderedEqs[keepEq]
  end
  local nWhenDiscrete = length(intVarsWrittenInWhen)
  @info "[resolveIntegers] Reclassified $nReclassified integer variables as parameters, kept $nWhenDiscrete as when-driven discrete, removed $nRemoved equations"
end

"""
  Collect the set of Integer variable names (or base names) that appear on
  the LHS of any ASSIGN statement inside a BDAE.WHEN_EQUATION. These are
  discrete state variables — resolveIntegers must not reclassify them as
  parameters or drop their defining when-clauses.
"""
function _collectIntegerLhsInWhen(orderedEqs, intVarNames::OrderedSet{String},
                                   intVarBaseNames::OrderedSet{String})::OrderedSet{String}
  local allIntNames = union(intVarNames, intVarBaseNames)
  local result = OrderedSet{String}()
  for eq in orderedEqs
    if !(eq isa BDAE.WHEN_EQUATION)
      continue
    end
    _collectWhenAssignsLhs!(result, eq.whenEquation, allIntNames)
  end
  return result
end

function _collectWhenAssignsLhs!(result::OrderedSet{String}, we,
                                  allIntNames::OrderedSet{String})
  @match we begin
    BDAE.WHEN_STMTS(_, stmts, elsewhenPart) => begin
      for stmt in stmts
        if stmt isa BDAE.ASSIGN && stmt.left isa DAE.CREF
          local name = string(stmt.left.componentRef)
          if name in allIntNames
            push!(result, name)
          end
          local baseName = replace(name, r"\[.*\]$" => "")
          if baseName in allIntNames
            push!(result, baseName)
          end
        end
      end
      @match elsewhenPart begin
        SOME(inner) => _collectWhenAssignsLhs!(result, inner, allIntNames)
        _ => nothing
      end
    end
    _ => nothing
  end
end

"""
  Collect the set of Integer/enum variable names that appear on the LHS of
  any assignment statement inside a BDAE.ALGORITHM equation. Gate logic in
  Modelica.Electrical.Digital uses algorithm sections (not when-clauses) to
  update auxiliary lookup arrays — variables assigned here must not be
  reclassified as constant parameters.
"""
function _collectIntegerLhsInAlgorithm(orderedEqs, intVarNames::OrderedSet{String},
                                        intVarBaseNames::OrderedSet{String})::OrderedSet{String}
  local allIntNames = union(intVarNames, intVarBaseNames)
  local result = OrderedSet{String}()
  for eq in orderedEqs
    if !(eq isa BDAE.ALGORITHM)
      continue
    end
    alg = eq.alg
    if alg isa DAE.ALGORITHM_STMTS
      _collectAlgorithmStmtLhs!(result, alg.statementLst, allIntNames)
    end
  end
  return result
end

function _collectAlgorithmStmtLhs!(result::OrderedSet{String}, stmts, allIntNames::OrderedSet{String})
  for stmt in stmts
    if stmt isa DAE.STMT_ASSIGN || stmt isa DAE.STMT_ASSIGN_ARR
      local lhsExp = stmt isa DAE.STMT_ASSIGN ? stmt.exp1 : stmt.lhs
      if lhsExp isa DAE.CREF
        local name = string(lhsExp.componentRef)
        if name in allIntNames
          push!(result, name)
        end
        local baseName = replace(name, r"\[.*\]$" => "")
        if baseName in allIntNames
          push!(result, baseName)
        end
      end
    elseif stmt isa DAE.STMT_FOR || stmt isa DAE.STMT_PARFOR || stmt isa DAE.STMT_WHILE
      _collectAlgorithmStmtLhs!(result, stmt.statementLst, allIntNames)
    elseif stmt isa DAE.STMT_IF
      _collectAlgorithmStmtLhs!(result, stmt.statementLst, allIntNames)
      _collectAlgorithmElseLhs!(result, stmt.else_, allIntNames)
    end
  end
end

function _collectAlgorithmElseLhs!(result::OrderedSet{String}, elseBranch, allIntNames::OrderedSet{String})
  if elseBranch isa DAE.ELSEIF
    _collectAlgorithmStmtLhs!(result, elseBranch.statementLst, allIntNames)
    _collectAlgorithmElseLhs!(result, elseBranch.else_, allIntNames)
  elseif elseBranch isa DAE.ELSE
    _collectAlgorithmStmtLhs!(result, elseBranch.statementLst, allIntNames)
  end
end

  @exportAll()
end
