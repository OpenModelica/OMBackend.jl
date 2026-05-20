#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortiurm (OSMC),
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

module BDAEUtil

using ExportAll
using MetaModelica
using Setfield

import ..BDAE
import ..BackendEquation
import OMBackend
import ..FrontendUtil.Util
import Absyn
import DAE
import OMFrontend

"""
This function converts an array of variables to the BDAE variable structure
"""
function convertVarArrayToBDAE_Variables(vars::Vector{BDAE.VAR})::Vector
  local variables = [i for i in vars]
  return variables
end

"""
  Creates a flat list of equation systems.
"""
function createEqSystems(frontendDAE::OMFrontend.Frontend.FlatModel)::BDAE.EQSYSTEM
  #= Create the first main equation system. =#
  local eqSystems = BDAE.EQSYSTEM[createEqSystem(frontendDAE)]
  for subModel in frontendDAE.structuralSubmodels
    push!(eqSystem, createEqSystems(subModel))
  end
  #=
  We will have a list of lists.
  For code generation this does not matter.
  Return a flat list.
  =#
  return [eqSystems...]
end

"""
  Creates a single equation system
"""
function createEqSystem(flatModel::OMFrontend.Frontend.FlatModel)
  #= TODO Extract the simple equations =#
  local equations = [equationToBackendEquation(eq) for eq in OMFrontend.Frontend.convertEquations(flatModel.equations)]
  local variables = [variableToBackendVariable(var) for var in OMFrontend.Frontend.convertVariables(flatModel.variables, list())]
  local algorithms = [alg for alg in flatModel.algorithms]
  local iAlgorithms = [iAlg for iAlg in flatModel.initialAlgorithms]
  local initialEquations = [equationToBackendEquation(ieq) for ieq in OMFrontend.Frontend.convertEquations(flatModel.initialEquations)]
  eqSystems = [BDAEUtil.createEqSystem(flatModel.name, variables, equations)]
  #= Treat structural submodels =#
  subModels = []
  BDAE.EQSYSTEM(vars, eqs, [])
end

"""
  Traverse and update a given structure BDAE.BDAEStructure given a traversalOperation and optional arguments
"""
function mapEqSystems(dae::BDAE.BACKEND_DAE, traversalOperation::Function, args...)
  dae = begin
    local eqs::Array{BDAE.EqSystem, 1}
    @match dae begin
      BDAE.BACKEND_DAE(eqs = eqs) => begin
        for i in 1:arrayLength(eqs)
          eqs[i] = traversalOperation(eqs[i], args...)
        end
        @assign dae.eqs = eqs
        dae
      end
      _ => begin
        dae
      end
    end
  end
end

function mapEqSystems(dae::BDAE.BACKEND_DAE, traversalOperation::Function)
  dae = begin
    local eqs::Vector{BDAE.EQSYSTEM}
    @match dae begin
      BDAE.BACKEND_DAE(eqs = eqs) => begin
        for i in 1:arrayLength(eqs)
          eqs[i] = traversalOperation(eqs[i])
        end
        @assign dae.eqs = eqs
        dae
      end
      _ => begin
        dae
      end
    end
  end
end

function mapEqSystemEquations(syst::BDAE.EQSYSTEM, traversalOperation::Function)
  syst = begin
    local eqs::Array{BDAE.Equation,1}
    @match syst begin
      BDAE.EQSYSTEM(orderedEqs = eqs) => begin
        for i in 1:arrayLength(eqs)
          eqs[i] = traversalOperation(eqs[i])
        end
        @assign syst.orderedEqs = eqs
        syst
      end
    end
  end
end

function mapEqSystemEquationsNoUpdate(syst::BDAE.EQSYSTEM, traversalOperation::Function, extArg)
  extArg = begin
    local eqs::Array{BDAE.Equation,1}
    @match syst begin
      BDAE.EQSYSTEM(orderedEqs = eqs) => begin
        for i in 1:arrayLength(eqs)
          extArg = traversalOperation(eqs[i], extArg)
        end
        extArg
      end
    end
  end
end

function mapEqSystemVariablesNoUpdate(syst::BDAE.EQSYSTEM, traversalOperation::Function, extArg)
  extArg = begin
    local varArr::Array{BDAE.Var,1}
    @match syst begin
      BDAE.EQSYSTEM(orderedVars =  varArr) => begin
        for i in 1:arrayLength(varArr)
          extArg = traversalOperation(varArr[i], extArg)
        end
        extArg
      end
    end
  end
  return extArg
end

"""
  Traverse a given equation using a traversalOperation.
  Mutates the given equation.
"""
Base.@nospecializeinfer function traverseEquationExpressions(@nospecialize(eq::BDAE.Equation),
                                                             traversalOperation::Function,
                                                             extArg::T)::Tuple{BDAE.Equation,T} where{T}
   (eq, extArg) = begin
     local lhs::DAE.Exp
     local rhs::DAE.Exp
     local cref::DAE.ComponentRef
     @match eq begin
       BDAE.EQUATION(lhs, rhs) => begin
         (lhs, extArg) = Util.traverseExpTopDown(lhs, traversalOperation, extArg)
         (rhs, extArg) = Util.traverseExpTopDown(rhs, traversalOperation, extArg)
         @assign begin
           eq.lhs = lhs
           eq.rhs = rhs
         end
         (eq, extArg)
       end
       BDAE.SOLVED_EQUATION(componentRef = cref, exp = rhs) => begin
         (rhs, extArg) = Util.traverseExpTopDown(rhs, traversalOperation, extArg)
         @assign eq.rhs = rhs;
         (eq, extArg)
       end
       BDAE.RESIDUAL_EQUATION(exp = rhs) => begin
         (rhs, extArg) = Util.traverseExpTopDown(rhs, traversalOperation, extArg)
         @assign eq.exp = rhs;
         (eq, extArg)
       end
       BDAE.IF_EQUATION(__) => begin
         local newConds = DAE.Exp[]
         local condChanged = false
         for exp in eq.conditions
           (resExp, extArg) = Util.traverseExpTopDown(exp, traversalOperation, extArg)
           push!(newConds, resExp)
           if !referenceEq(exp, resExp)
             condChanged = true
           end
         end
         if condChanged
           @assign eq.conditions = list(newConds...)
         end
         local newTrueBranches = List{BDAE.Equation}[]
         local trueChanged = false
         for eqLst in eq.eqnstrue
           local newEqs = BDAE.Equation[]
           for equation in eqLst
             (newEq, extArg) = traverseEquationExpressions(equation, traversalOperation, extArg)
             push!(newEqs, newEq)
             if !referenceEq(equation, newEq)
               trueChanged = true
             end
           end
           push!(newTrueBranches, list(newEqs...))
         end
         if trueChanged
           @assign eq.eqnstrue = list(newTrueBranches...)
         end
         local newFalseEqs = BDAE.Equation[]
         local falseChanged = false
         for equation in eq.eqnsfalse
           (newEq, extArg) = traverseEquationExpressions(equation, traversalOperation, extArg)
           push!(newFalseEqs, newEq)
           if !referenceEq(equation, newEq)
             falseChanged = true
           end
         end
         if falseChanged
           @assign eq.eqnsfalse = list(newFalseEqs...)
         end
         (eq, extArg)
       end
       BDAE.WHEN_EQUATION(__) => begin
         local whenEquation = eq.whenEquation
         (newCond, extArg) = Util.traverseExpTopDown(whenEquation.condition, traversalOperation, extArg)
         @assign eq.whenEquation.condition = newCond
         lst = traverseWhenEquation!(whenEquation, traversalOperation, extArg)
         @assign eq.whenEquation.whenStmtLst = lst
         #= TODO: Handle elsewhen =#
         (eq, extArg)
       end
       BDAE.ARRAY_EQUATION(left = lhs, right = rhs) => begin
         (newLhs, extArg) = Util.traverseExpTopDown(lhs, traversalOperation, extArg)
         (newRhs, extArg) = Util.traverseExpTopDown(rhs, traversalOperation, extArg)
         @assign begin
           eq.left = newLhs
           eq.right = newRhs
         end
         (eq, extArg)
       end
       _ => begin
         (eq, extArg)
       end
     end
   end
end

"""
  Traverses BDAE.WHEN_EQUATION equations.
  Note, currently only BDAE.REINIT is implemented.
"""
function traverseWhenEquation!(whenEq, traversalOperation, extArg)
  newWhenStmtLst = list()
  for stmt in whenEq.whenStmtLst
    @match stmt begin
      BDAE.REINIT(__) => begin
        (stateVar, extArg) = Util.traverseExpTopDown(stmt.stateVar, traversalOperation, extArg)
        (value, extArg) = Util.traverseExpTopDown(stmt.value, traversalOperation, extArg)
        newWhenStmtLst = BDAE.REINIT(stateVar, value, stmt.source) <| newWhenStmtLst
      end
      BDAE.ASSIGN(__) => begin
        (lhs, extArg) = Util.traverseExpTopDown(stmt.left, traversalOperation, extArg)
        (rhs, extArg) = Util.traverseExpTopDown(stmt.right, traversalOperation, extArg)
        newWhenStmtLst = BDAE.ASSIGN(lhs, rhs, stmt.source) <| newWhenStmtLst
      end
      BDAE.NORETCALL(__) => begin
        (exp, extArg) = Util.traverseExpTopDown(stmt.exp, traversalOperation, extArg)
        newStmt = exp === stmt.exp ? stmt : BDAE.NORETCALL(exp, stmt.source)
        newWhenStmtLst = newStmt <| newWhenStmtLst
      end
      #= Modelica `terminate("msg")` inside a when-clause. Traverse the
         message expression and rebuild. Surfaces on MSL
         Mechanics.MultiBody.Examples.Systems.RobotR3.{oneAxis,fullRobot}
         where PathPlanning calls terminate() when the motion profile
         finishes. =#
      BDAE.TERMINATE(__) => begin
        (msg, extArg) = Util.traverseExpTopDown(stmt.message, traversalOperation, extArg)
        newStmt = msg === stmt.message ? stmt : BDAE.TERMINATE(msg, stmt.source)
        newWhenStmtLst = newStmt <| newWhenStmtLst
      end
      BDAE.ASSERT(__) => begin
        (cond, extArg) = Util.traverseExpTopDown(stmt.condition, traversalOperation, extArg)
        (msg, extArg) = Util.traverseExpTopDown(stmt.message, traversalOperation, extArg)
        (lvl, extArg) = Util.traverseExpTopDown(stmt.level, traversalOperation, extArg)
        newStmt = (cond === stmt.condition && msg === stmt.message && lvl === stmt.level) ?
                    stmt : BDAE.ASSERT(cond, msg, lvl, stmt.source)
        newWhenStmtLst = newStmt <| newWhenStmtLst
      end
      _ => begin
        error(string(stmt) * " is not implemented yet!")
      end
    end
  end
  return listReverse(newWhenStmtLst)
end

"""
Directly maps the DAE type to the BDAE type.
Before causalization we do not know if variables are state or not.
"""
Base.@nospecializeinfer function DAE_VarKind_to_BDAE_VarKind(@nospecialize(kind::DAE.VarKind))::BDAE.VarKind
  @match kind begin
    DAE.VARIABLE(__) => BDAE.VARIABLE()
    DAE.DISCRETE(__) => BDAE.DISCRETE()
    DAE.PARAM(__) => BDAE.PARAM()
    DAE.CONST(__) => BDAE.CONST()
  end
end

function isStateOrVariable(var::BDAE.VAR)
  local kind = var.varKind
  return isStateOrVariable(kind)
end

function isStateOrAlgebraicOrDiscrete(var::BDAE.VAR)
  local kind = var.varKind
  return isStateOrAlgebraicOrDiscrete(kind)
end

function isStateOrVariable(kind::BDAE.VarKind)
  res = @match kind begin
    BDAE.VARIABLE(__) => true
    BDAE.STATE(__) => true
    _ => false
  end
  return res
end


function isStateOrAlgebraicOrDiscrete(kind::BDAE.VarKind)
  res = @match kind begin
    BDAE.VARIABLE(__) => true
    BDAE.STATE(__) => true
    BDAE.DISCRETE(__) => true
    _ => false
  end
  return res
end

function isVariable(kind::BDAE.VarKind)
  res = @match kind begin
    BDAE.VARIABLE(__) => true
    _ => false
  end
  return res
end

function isState(kind::BDAE.VarKind)
  res = @match kind begin
    BDAE.STATE(__) => true
    _ => false
  end
  return res
end

function isState(var::BDAE.VAR)
  isState(var.varKind)
end

function isDiscrete(kind::BDAE.VarKind)
  res = @match kind begin
    BDAE.DISCRETE(__) => true
    _ => false
  end
  return res
end

function isWhenEquation(eq::BDAE.Equation)
  @match eq begin
    BDAE.WHEN_EQUATION(__) => true
    _ => false
  end
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
        #= Adds a state with boolean value that does not matter,
           it is later  BDAE.BACKEND_DAE(eqs = eqs) checked if it exists at all =#
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

#=TODO. Did I do something stupid down below here.. ?=#

function countAllUniqueVariablesInSetOfEquations(eqs::Vector{RES_EQ}, vars::Vector{VAR}) where {RES_EQ, VAR}
  vars = Set()
  for eq in eqs
    varsForEq = getAllVariables(eq, vars)
    for v in varsForEq
      push!(vars, v)
    end
  end
  return length(vars)
end

"""
  Author: johti17
  input: Backend Equation, eq
  input: All existing variables
  output All variable in that specific equation
"""

function getAllVariables(eq::BDAE.RESIDUAL_EQUATION, vars::Vector{BDAE.VAR})::Vector{DAE.ComponentRef}
  local componentReferences::List = Util.getAllCrefs(eq.exp)
  local stateCrefs = Dict{DAE.ComponentRef, Bool}()
  (_, stateElements)  = traverseEquationExpressions(eq, detectStateExpression, stateCrefs)
  local stateElementArray = [string(cr) for cr in collect(keys(stateElements))]
  local componentReferencesNotStates = [string(cr) for cr in componentReferences]
  variablesInEq::Vector = []
  for var in vars
    local vn = string(var.varName)
    if vn in componentReferencesNotStates && isVariable(var.varKind)
      push!(variablesInEq, var.varName)
    elseif vn in stateElementArray
      push!(variablesInEq, var.varName)
    else
    end
  end
  return variablesInEq
end

"""
  input: Backend when assignment (BDAE.ASSIGN)
  input: All existing variables
  output All variable in that specific equation
"""
function getAllVariables(assignment::BDAE.ASSIGN, vars::Vector{BDAE.VAR})::Vector{DAE.ComponentRef}
  local leftCrefs = listArray(Util.getAllCrefs(assignment.left))
  local rightCrefs = listArray(Util.getAllCrefs(assignment.right))
  return vcat(leftCrefs, rightCrefs)
end

function getAllVariables(bdaeRenit::BDAE.REINIT, vars::Vector{BDAE.VAR})#::Vector{DAE.ComponentRef}
  variables = DAE.CREF[bdaeRenit.stateVar]
  crefs = Util.getAllCrefs(bdaeRenit.value)
  return vcat(variables, listArray(crefs))
end

"""
  ASSERT_EQUATION participates in the runtime check, not the continuous
  equation system. Return an empty set for matching/graph-building so
  the assert is a passive observer. The check is still emitted via the
  code-generation path. Variables referenced in the condition need not
  bind new equations here — any real equation that uses the same
  variables will already pull them into the graph.
"""
function getAllVariables(eq::BDAE.ASSERT_EQUATION, vars::Vector{BDAE.VAR})::Vector{DAE.ComponentRef}
  return DAE.ComponentRef[]
end

"""
  Returns CREFs from the right-hand side of a when-statement.
  Used for generating local variable bindings in callback affect functions.
  Excludes the target (stateVar/LHS) because the affect body accesses it
  via integrator.u[idx] directly.
"""
function getRHSVariables(assignment::BDAE.ASSIGN)::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(assignment.right))
end

function getRHSVariables(bdaeReinit::BDAE.REINIT)::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(bdaeReinit.value))
end

function getRHSVariables(call::BDAE.NORETCALL)::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(call.exp))
end

function getRHSVariables(assert::BDAE.ASSERT)::Vector{DAE.ComponentRef}
  return vcat(listArray(Util.getAllCrefs(assert.condition)),
              listArray(Util.getAllCrefs(assert.message)),
              listArray(Util.getAllCrefs(assert.level)))
end

function getRHSVariables(term::BDAE.TERMINATE)::Vector{DAE.ComponentRef}
  return listArray(Util.getAllCrefs(term.message))
end

"""
  Fetches all variables in if equations.
FIXME:
Ideally this function should also return a vector of component references
"""
function getAllVariables(eq::BDAE.IF_EQUATION, vars::Vector{BDAE.VAR})
  local condVars = map(c -> listArray(Util.getAllCrefs(c)), eq.conditions)
  condVars = map(x -> string(x), collect(Iterators.flatten(condVars)))
  local ifEqEqsTrue = collect(Iterators.flatten(listArray(eq.eqnstrue)))
  local ifEqEqsFalse = listArray(eq.eqnsfalse)
  local trueVars = map(eq -> getAllVariables(eq, vars), ifEqEqsTrue)
  local falseVars = map(eq -> getAllVariables(eq, vars), ifEqEqsFalse)
  trueVars = collect(Iterators.flatten(trueVars))
  falseVars = collect(Iterators.flatten(falseVars))
  local res = vcat(condVars, trueVars, falseVars)
  res = map(x ->string(x), res)
  return res
end

"""
  Author:johti17
  input: Backend Equation, eq
  input: All existing variables
  output All variable in that specific equation except the state variables
"""
function getAllVariablesExceptStates(eq::BDAE.IF_EQUATION, vars::Vector{BDAE.VAR})::Vector{DAE.ComponentRef}
  local allVarNames = getAllVariables(eq, vars)
  local stateNames = Set(string(v.varName) for v in vars if isState(v))
  return filter(v -> !(v in stateNames), allVarNames)
end

function isArray(cref::DAE.ComponentRef)::Bool
  @match cref begin
    DAE.OPTIMICA_ATTR_INST_CREF(__) || DAE.WILD(__) => false
    _ => begin
      typeof(cref.identType) == DAE.T_ARRAY
    end
  end
end

function getSubscriptAsIntArray(dims)::Array
  local dimIndices = []
  for d in dims
    if ! (typeof(d) == DAE.DIM_INTEGER)
      throw("Non integers dimensions for arrays are not supported by OMBackend. Variable was $(string(v))")
    else
      push!(dimIndices, d.integer)
    end
  end
  return dimIndices
end


"""
  Inverts a given DAE.Exp
"""
function invertCondition(cond::DAE.Exp)
  @match cond begin
    DAE.RELATION(exp1, DAE.GREATER(__), exp2, index, optionExpisASUB) => begin
      DAE.RELATION(exp1, DAE.LESS(cond.operator.ty), exp2, index, optionExpisASUB)
    end
    DAE.RELATION(exp1, DAE.LESS(__), exp2, index, optionExpisASUB) => begin
      DAE.RELATION(exp1, DAE.GREATER(cond.operator.ty), exp2, index, optionExpisASUB)
    end
    _ => throw("Tried to invert unsupported backend condition:" * string(cond) * "type was: " * string(typeof(cond)))
  end
end

"""
  Maps a backend equation to a backend when equation.
"""
function eqToWhenOperator(eq::BDAE.Equation)
  res = @match eq begin
    BDAE.EQUATION(lhs, rhs, source, attributes) => begin
      BDAE.ASSIGN(lhs, rhs, source)
    end
    _ => begin
      throw(string("Conversion from ", string(eq), " Not supported", "Type was: ", typeof(eq)))
    end
  end
  return res
end

function DAE_DimensionToIntVector(dims::Cons{<:DAE.Dimension})::Vector{Int}
  local dimIndices = []
  for d in dims
    if ! (typeof(d) == DAE.DIM_INTEGER)
      throw("Non integers dimensions for arrays are not supported by OMBackend. Dimension was $(string(dim))")
    else
      push!(dimIndices, d.integer)
    end
  end
  return dimIndices
end

function getDimensionFromComplexType(callTy::DAE.T_COMPLEX)
  Int[length(callTy.varLst)]
end

"""
  Extract the T_COMPLEX type from a DAE expression.
  Handles:
    - CREF with T_COMPLEX identType (the simple record-variable case)
    - CALL whose CALL_ATTR return type is T_COMPLEX (operator-record functions
      like `Modelica_SIunits_ComplexMagneticFlux_'+'` appearing as either the
      LHS or RHS of a COMPLEX_EQUATION)
    - RECORD constructor literal (the RHS `Complex[REC(0.0, 0.0)]` case —
      T_COMPLEX is reconstructible from the record path + field list)
  Returns nothing for other shapes (BINARY / IFEXP / ASUB / T_ARRAY of records).
"""
function getComplexType(exp::DAE.Exp)
  @match exp begin
    DAE.CREF(_, ty && DAE.T_COMPLEX(__)) => ty
    DAE.CALL(_, _, DAE.CALL_ATTR(ty = ty && DAE.T_COMPLEX(__))) => ty
    DAE.RECORD(ty = ty && DAE.T_COMPLEX(__)) => ty
    _ => nothing
  end
end

"""
  Append a field name to a DAE.CREF expression, producing a deeper CREF.
  E.g. CREF(a.b.R, T_COMPLEX) + "T" => CREF(a.b.R.T, fieldTy)
"""
function appendFieldToCref(exp::DAE.Exp, fieldName::String, fieldTy::DAE.Type)::DAE.Exp
  @match exp begin
    DAE.CREF(cref, _) => begin
      local newCref = appendFieldToComponentRef(cref, fieldName, fieldTy)
      DAE.CREF(newCref, fieldTy)
    end
    DAE.RECORD(_, exps, fieldNames, _) => begin
      local expVec = collect(exps)
      local nameVec = collect(fieldNames)
      local fieldIdx = findfirst(==(fieldName), nameVec)
      if fieldIdx === nothing
        @warn "appendFieldToCref: record constructor missing field $fieldName, returning original expression"
        exp
      else
        expVec[fieldIdx]
      end
    end
    #= Push field access into both branches of an if-expression.
       MSL ComplexMath.conj wrapped in `if cond then c else conj(c)` produces
       DAE.IFEXP as the LHS of a record equation; decomposition needs to
       descend into each branch. =#
    DAE.IFEXP(cond, thenExp, elseExp) => begin
      DAE.IFEXP(cond,
                appendFieldToCref(thenExp, fieldName, fieldTy),
                appendFieldToCref(elseExp, fieldName, fieldTy))
    end
    #= Inline Modelica.ComplexMath.conj: conj(c).re == c.re, conj(c).im == -c.im.
       Surfaces on SeriesBode where `conj(u2)` appears in an IFEXP branch. =#
    DAE.CALL(path, expLst, _) where _isComplexConjCall(path, expLst) => begin
      local innerArg = listHead(expLst)
      if fieldName == "re"
        appendFieldToCref(innerArg, "re", fieldTy)
      elseif fieldName == "im"
        local imExp = appendFieldToCref(innerArg, "im", fieldTy)
        DAE.UNARY(DAE.UMINUS(fieldTy), imExp)
      else
        error("appendFieldToCref: unexpected field \"$fieldName\" on conj(); " *
              "expected \"re\" or \"im\".")
      end
    end
    #= Inlined `Complex(re, im)` lowers to a call of the `fromReal` constructor.
       Field access on the constructor result is the matching positional arg. =#
    DAE.CALL(path, expLst, _) where _isComplexFromRealCall(path) => begin
      local argVec = collect(expLst)
      if fieldName == "re"
        isempty(argVec) ? exp : argVec[1]
      elseif fieldName == "im"
        length(argVec) >= 2 ? argVec[2] : DAE.RCONST(0.0)
      else
        error("appendFieldToCref: unexpected field \"$fieldName\" on Complex() " *
              "constructor; expected \"re\" or \"im\".")
      end
    end
    _ => begin
      error("appendFieldToCref: unexpected expression type $(typeof(exp)) " *
            "when appending field \"$fieldName\" of type $(string(fieldTy)). " *
            "Expression: $(first(string(exp), 200))")
    end
  end
end

#= Pattern match for `Modelica.ComplexMath.conj(arg)` calls. =#
function _isComplexConjCall(path::Absyn.Path, expLst)::Bool
  if listEmpty(expLst); return false; end
  @match path begin
    Absyn.QUALIFIED("Modelica",
      Absyn.QUALIFIED("ComplexMath", Absyn.IDENT("conj"))) => true
    Absyn.IDENT("Modelica_ComplexMath_conj") => true
    _ => false
  end
end

#= Pattern match for the `Complex.'constructor'.fromReal(re, im)` lowering
   produced after inlining `Complex(re, im)`. =#
function _isComplexFromRealCall(path::Absyn.Path)::Bool
  @match path begin
    Absyn.QUALIFIED("Complex",
      Absyn.QUALIFIED("'constructor'", Absyn.IDENT("fromReal"))) => true
    Absyn.IDENT("Complex_'constructor'_fromReal") => true
    Absyn.IDENT("Complex_constructor_fromReal") => true
    _ => false
  end
end

"""
  Append a field to a ComponentRef chain.
  The last CREF_IDENT becomes a CREF_QUAL, and the field becomes the new CREF_IDENT.
"""
function appendFieldToComponentRef(cr::DAE.ComponentRef, fieldName::String, fieldTy::DAE.Type)::DAE.ComponentRef
  @match cr begin
    DAE.CREF_IDENT(ident, identType, subs) => begin
      DAE.CREF_QUAL(ident, identType, subs,
                    DAE.CREF_IDENT(fieldName, fieldTy, nil))
    end
    DAE.CREF_QUAL(ident, identType, subs, innerCref) => begin
      DAE.CREF_QUAL(ident, identType, subs,
                    appendFieldToComponentRef(innerCref, fieldName, fieldTy))
    end
  end
end

include("backendDump.jl")
@exportAll()
end
