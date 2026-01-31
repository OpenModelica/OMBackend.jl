module FrontendUtil

import Absyn
import DAE
import OMFrontend
using MetaModelica

include("AbsynUtil.jl")
include("Util.jl")
include("Prefix.jl")

export Util
export AbsynUtil
export Prefix
export removeSmoothFromStatements

"""
 This function handles certain builtin functions.
 For now it only removes the smooth function
"""
function handleBuiltin(fm::OMFrontend.Frontend.FLAT_MODEL)
  fm = removeSmoothOperator(fm)
  return fm
end

"""
  Removes the smooth operator from the set of equations.
  (From the specification the application of this function seems to be optional)
See:
https://build.openmodelica.org/Documentation/ModelicaReference.Operators.%27smooth()%27.html
"""
function removeSmoothOperator(fm::OMFrontend.Frontend.FLAT_MODEL)
  local equations = fm.equations
  equations = OMFrontend.Frontend.mapList(equations, removeSmooth)
  @assign fm.equations = equations
  return fm
end

"""
  Removes the smooth operator from an equation, returns the argument to smooth.
"""
function removeSmooth(eq::OMFrontend.Frontend.Equation)
  @match eq begin
    OMFrontend.Frontend.EQUATION_EQUALITY(lhs = e1, rhs = OMFrontend.Frontend.CALL_EXPRESSION(call)) where string(OMFrontend.Frontend.functionName(call)) == "smooth"  => begin
      local arguments = OMFrontend.Frontend.arguments(call)
      @match x <| y <| nil = arguments
      local newEq = eq
      @assign newEq.rhs = y
      newEq
    end
    _ => eq
  end
end

#=
  Remove smooth calls from DAE expressions.
  smooth(p, expr) -> expr
  Returns (newExp, continueTraversal, arg) as expected by traverseExpTopDown.
=#
function removeSmoothFromExp(exp::DAE.Exp, arg)
  @match exp begin
    DAE.CALL(path = Absyn.IDENT("smooth"), expLst = _ <| exprArg <| _) => begin
      #= smooth has two arguments: order and expression. Return the expression. =#
      return (exprArg, false, arg)
    end
    _ => (exp, true, arg)
  end
end

"""
  Remove smooth calls from a list of DAE statements.
"""
function removeSmoothFromStatements(stmts::Vector)::Vector{DAE.Statement}
  return DAE.Statement[removeSmoothFromStatement(stmt) for stmt in stmts]
end

"""
  Remove smooth calls from a single DAE statement by traversing its expressions.
"""
function removeSmoothFromStatement(stmt::DAE.Statement)::DAE.Statement
  @match stmt begin
    DAE.STMT_ASSIGN(type_, exp1, exp, source) => begin
      (newExp1, _) = Util.traverseExpTopDown(exp1, removeSmoothFromExp, nothing)
      (newExp, _) = Util.traverseExpTopDown(exp, removeSmoothFromExp, nothing)
      DAE.STMT_ASSIGN(type_, newExp1, newExp, source)
    end
    DAE.STMT_TUPLE_ASSIGN(type_, expExpLst, exp, source) => begin
      newExpLst = map(expExpLst) do e
        (newE, _) = Util.traverseExpTopDown(e, removeSmoothFromExp, nothing)
        newE
      end
      (newExp, _) = Util.traverseExpTopDown(exp, removeSmoothFromExp, nothing)
      DAE.STMT_TUPLE_ASSIGN(type_, newExpLst, newExp, source)
    end
    DAE.STMT_ASSIGN_ARR(type_, lhs, exp, source) => begin
      (newLhs, _) = Util.traverseExpTopDown(lhs, removeSmoothFromExp, nothing)
      (newExp, _) = Util.traverseExpTopDown(exp, removeSmoothFromExp, nothing)
      DAE.STMT_ASSIGN_ARR(type_, newLhs, newExp, source)
    end
    DAE.STMT_IF(exp, statementLst, else_, source) => begin
      (newExp, _) = Util.traverseExpTopDown(exp, removeSmoothFromExp, nothing)
      newStmts = removeSmoothFromStatements(collect(statementLst))
      DAE.STMT_IF(newExp, list(newStmts...), else_, source)
    end
    DAE.STMT_FOR(type_, iterIsArray, iter, index, range, statementLst, source) => begin
      (newRange, _) = Util.traverseExpTopDown(range, removeSmoothFromExp, nothing)
      newStmts = removeSmoothFromStatements(collect(statementLst))
      DAE.STMT_FOR(type_, iterIsArray, iter, index, newRange, list(newStmts...), source)
    end
    DAE.STMT_WHILE(exp, statementLst, source) => begin
      (newExp, _) = Util.traverseExpTopDown(exp, removeSmoothFromExp, nothing)
      newStmts = removeSmoothFromStatements(collect(statementLst))
      DAE.STMT_WHILE(newExp, list(newStmts...), source)
    end
    DAE.STMT_RETURN(__) => stmt
    DAE.STMT_NORETCALL(exp, source) => begin
      (newExp, _) = Util.traverseExpTopDown(exp, removeSmoothFromExp, nothing)
      DAE.STMT_NORETCALL(newExp, source)
    end
    _ => stmt
  end
end

end
