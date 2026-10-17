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
    SimCode

The simulation code of a model: variables with their categories and indices, residual
equations, when- and if-equations, initial equations and algorithms, the functions the model
calls; the code generators' input. It is lowered from OMJLIR (`fromOMJLIR`, lowerModel.jl) and transformed
by its passes; its types (types.jl) are its own and depend on no other representation.
"""
module SimCode

import DataStructures: OrderedDict, OrderedSet
import Graphs
import ..BackendUtil.GraphAlgorithms
import ..OMJLIR
import ..Traversal
import ..Traversal: traverseFields, traverseElements
import ..Passes: legacyName, isSampleTrigger, isBuiltinCallOf, tupleAssignElements, initialCall, builtinCall, edgeCall
import ..unsupported, .._fallback, ..isTunableParameter, ..TUNABLE_PARAMETERS, ..BACKEND_PERFLOG, ..envSwitch
const IR = OMJLIR

include("types.jl")
include("traverse.jl")
include("print.jl")
include("lowerExpressions.jl")
include("lowerVariables.jl")
include("lowerModel.jl")
include("lowerStructure.jl")
include("recordCallSites.jl")
include("lowerFunctions.jl")
include("queries.jl")
include("Passes/evaluation.jl")
include("Passes/trivialEquations.jl")
include("Passes/substitution.jl")
include("Passes/parameterClosure.jl")
include("Passes/constantPropagation.jl")
include("Passes/complexRecords.jl")
include("Passes/aliasElimination.jl")
include("Passes/parameterElimination.jl")
include("Passes/functionEvaluation.jl")
include("Passes/frozenStates.jl")
include("Passes/explicitFold.jl")
include("Passes/discretes.jl")
include("Passes/initialValues.jl")
include("Passes/structure.jl")
include("Passes/outputOnly.jl")
include("Passes/runner.jl")
include("Passes/pipeline.jl")
include("check.jl")

end # module SimCode
