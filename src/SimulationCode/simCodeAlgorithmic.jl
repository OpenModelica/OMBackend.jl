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

abstract type ModelicaFunction end

struct MODELICA_FUNCTION <: ModelicaFunction
  name::String
  #= Vector of integer, real string etc=#
  inputs::Vector
  outputs::Vector
  locals::Vector
  statements::Vector{DAE.Statement}
end

struct EXTERNAL_MODELICA_FUNCTION <: ModelicaFunction
  name::String
  inputs::Vector
  outputs::Vector
  libInfo::String
end

#= Lowered form of a Modelica `initial algorithm` or `algorithm when initial()`
   body. Fires once during initialization (Modelica spec §11.4). `statements`
   carries the SimCode-native `WhenOperator` form (control flow flattened
   by `_daeStmtsToWhenOps`); `daeStatements` carries the structured
   `DAE.Statement` list (preserving STMT_IF / STMT_FOR / STMT_WHILE) for the
   module-load-time codegen path. =#
struct INITIAL_ALGORITHM
  statements::Vector{WhenOperator}
  daeStatements::Vector{DAE.Statement}
end

INITIAL_ALGORITHM(stmts::Vector{WhenOperator}) = INITIAL_ALGORITHM(stmts, DAE.Statement[])

#= Boundary convenience: accept a `List{BDAE.WhenOperator}` or
   `Vector{BDAE.WhenOperator}` at construction time and project through
   `toWhenOperator` so the boundary callers do not need to thread the
   conversion themselves. =#
function INITIAL_ALGORITHM(stmts, daeStmts::Vector{DAE.Statement})
  local simStmts = WhenOperator[toWhenOperator(s) for s in stmts]
  return INITIAL_ALGORITHM(simStmts, daeStmts)
end

function INITIAL_ALGORITHM(stmts)
  local simStmts = WhenOperator[toWhenOperator(s) for s in stmts]
  return INITIAL_ALGORITHM(simStmts, DAE.Statement[])
end
