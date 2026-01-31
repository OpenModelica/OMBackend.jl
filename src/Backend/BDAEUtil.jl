#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortiurm (OSMC),
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
    local eqs::Vector{BDAE.BDAE.EQSYSTEM}
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
function traverseEquationExpressions(eq::BDAE.Equation,
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
         @assign eq.lhs = lhs
         @assign eq.rhs = rhs
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
         for exp in eq.conditions
           (resExp, extArg) = Util.traverseExpTopDown(exp, traversalOperation, extArg)
         end
         for eqLst in eq.eqnstrue
           for equation in eqLst
             traverseEquationExpressions(equation, traversalOperation, extArg)
           end
         end
         for equation in eq.eqnsfalse
           traverseEquationExpressions(equation, traversalOperation, extArg)
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
         @assign eq.left = newLhs
         @assign eq.right = newRhs
         @info "Processed ARRAY_EQUATION: lhs changed=$(lhs !== newLhs), rhs changed=$(rhs !== newRhs)"
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
      end
      _ => begin
        throw(string(stmt) * " is not implemented yet!")
      end
    end
  end
  return listReverse(newWhenStmtLst)
end

"""
Directly maps the DAE type to the BDAE type.
Before casualisation we do not know if variables are state or not.
"""
function DAE_VarKind_to_BDAE_VarKind(kind::DAE.VarKind)::BDAE.VarKind
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
  local assignmentAsEq = BDAE.RESIDUAL_EQUATION(DAE.BINARY(assignment.left,
                                                           DAE.SUB(assignment.left.ty),
                                                           assignment.right),
                                                assignment.source, BDAE.NO_ATTRIBUTES())
  local variablesInEq = getAllVariables(assignmentAsEq, vars)
  return variablesInEq
end

function getAllVariables(bdaeRenit::BDAE.REINIT, vars::Vector{BDAE.VAR})#::Vector{DAE.ComponentRef}
  variables = DAE.CREF[bdaeRenit.stateVar]
  crefs = Util.getAllCrefs(bdaeRenit.value)
  return vcat(variables, listArray(crefs))
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
  #=FIXME: Not pretty =#
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
  local componentReferences::List = Util.getAllCrefs(eq.exp)
  local componentReferencesArr::Array = [componentReferences...]
  local varNames = [v.varName for v in vars]
  variablesInEq::Vector = []
  for vn in varNames
    if vn in componentReferencesArr
      push!(variablesInEq, vn)
    end
  end
  return variablesInEq
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

function getSubscriptAsUnicodeString(subscriptLst)::String
  local subscripts = subscriptLst
  local subscriptStr = ""
  for s in subscripts
    @assert(typeof(s) == DAE.INDEX, "DAE.INDEX: is expected for $(s)")
    #= Here I assume integer index!=#
    local indexAsInt::Integer = s.exp.integer
    local result = 0
    local tmp = 0
    subscriptStr *= getIndexAsAUnicodeString(s)
  end
  return subscriptStr
end

function getIndexAsAUnicodeString(idx::DAE.INDEX)
  local indexAsInt::Integer = idx.exp.integer
  local subscriptStr = ""
  return getIntAsUnicodeSubscript(indexAsInt)
end

"""
input: 100
output \"₁₀₀\"
"""
function getIntAsUnicodeSubscript(i::Integer)
  local subscriptStr = ""
  while i > 0
    tmp = i % 10
    subscriptStr  *= OMBackend.LATEX_SYMBOLS["\\_" * string(tmp)]
    i =  i ÷ 10
  end
  #= Since the code above generates the string in the reverse order it needs to be re reversed=#
  return reverse(subscriptStr)
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

function DAE_DimensionToIntVector(dims::Cons{DAE.Dimension})::Vector{Int}
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
  length(callTy.varLst)
end


include("backendDump.jl")
@exportAll()
end
