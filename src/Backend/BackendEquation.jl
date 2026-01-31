#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
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

module BackendEquation

using MetaModelica

using ExportAll

import DAE
import ..BDAE
import ..@BACKEND_LOGGING
import ..FrontendUtil.Util

"""
    kabdelhak:
    Create an empty equation array
"""
function emptyEqns()
  eqns::Array = []
  (eqns)
end

"""
    kabdelhak:
    Transform a single equation to residual form by subtracting the rhs from
    the lhs
"""
function makeResidualEquation(eqn::BDAE.Equation)
  eqn = begin
    local lhs::DAE.Exp
    local rhs::DAE.Exp
    local source::DAE.ElementSource
    local attr::BDAE.EquationAttributes
    @match eqn begin
      BDAE.EQUATION(lhs, rhs, source, attr) => begin
        BDAE.RESIDUAL_EQUATION(makeResidualExp(lhs, rhs), source, attr)
      end
      BDAE.IF_EQUATION(__) => begin
        makeResidualIfEquation(eqn)
      end
      _ => begin
        (eqn)
      end
    end
  end
end

"""
johti17:
  Transforms the sub-equations of an if-equation into residuals
"""
function makeResidualIfEquation(eqn::BDAE.IF_EQUATION)::BDAE.IF_EQUATION
  local trueEquations::List{List{BDAE.Equation}} = eqn.eqnstrue
  local falseEquations::List{BDAE.Equation} = eqn.eqnsfalse
  local trueEquations2::List{List{BDAE.Equation}} = nil
  local falseEquations2::List{BDAE.Equation} = nil
  #= Each true equation is one branch, so a double loop is needed. =#
  for eqBranch in trueEquations
    tmp::List{BDAE.Equation} = nil
    for eq in eqBranch
      tmp = makeResidualEquation(eq) <| tmp
    end
    trueEquations2 = tmp <| trueEquations2
  end
  for eq in falseEquations
    falseEquations2 = makeResidualEquation(eq) <| falseEquations2
  end
  return BDAE.IF_EQUATION(eqn.conditions, listReverse(trueEquations2), listReverse(falseEquations2), eqn.source, eqn.attr)
end

"""
   kabdelhak:
   Splits an if equation in multiple equations containing if expressions.
   INTENDED?: Currently requires correct ordering of branch equations
"""
function splitIfEquationResidual(eqn::BDAE.Equation)::List{BDAE.Equation}
  residualEqs::List{BDAE.Equation}=nil
  residualExps = splitIfEquationResidualTraverse(eqn.conditions, Util.transposeNestedList(eqn.eqnstrue), eqn.eqnsfalse, nil)
  for exp in residualExps
    residualEqs = BDAE.RESIDUAL_EQUATION(exp, eqn.source, eqn.attr) <| residualEqs
  end
  return residualEqs
end

function splitIfEquationResidualTraverse(lstCond::List{DAE.Exp}, lstlstTrue::List{List{BDAE.Equation}}, lstFalse::List{BDAE.Equation}, acc::List{DAE.Exp})::List{DAE.Exp}
  acc = begin
      local tmpTrue::List{BDAE.Equation}
      local restTrue::List{List{BDAE.Equation}}
      local tmpFalse::BDAE.Equation
      local restFalse::List{BDAE.Equation}
      @match (lstlstTrue, lstFalse) begin
      (nil, _) => begin
        (acc)
      end
      (tmpTrue <| restTrue, tmpFalse <| restFalse) => begin
         acc  = makeNestedIfExpressionResidual(lstCond, tmpTrue, tmpFalse) <| acc
         splitIfEquationResidualTraverse(lstCond, restTrue, restFalse, acc)
      end
    end
  end
  return acc
end

function makeNestedIfExpressionResidual(lstCond::List{DAE.Exp}, lstTrue::List{BDAE.Equation}, eqFalse::BDAE.Equation)
  exp = begin
    local cond::DAE.Exp
    local restCond::List{DAE.Exp}
    local eqTrue::DAE.Equation
    local restTrue::List{DAE.Equation}
    @match (lstCond, lstTrue) begin
      (nil, _) => begin
         #= fails for anything but BDAE.EQUATION() =#
         makeResidualExp(eqFalse.lhs, eqFalse,rhs)
      end
      (cond <| restCond, eqTrue <| restTrue) => begin
         #= fails for anything but BDAE.EQUATION() =#
         DAE.IFEXP(cond, makeResidualExp(eqTrue.lhs, eqTrue.rhs), makeNestedIfExpressionResidual(restCond, restTrue, eqFalse))
      end
    end
  end
end

function makeResidualExp(lhs::DAE.Exp, rhs::DAE.Exp)::DAE.Exp
  DAE.BINARY(lhs, DAE.SUB(DAE.T_REAL_DEFAULT), rhs)
end


@exportAll()
end
