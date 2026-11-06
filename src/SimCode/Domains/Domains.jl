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
The domain and causal levels of the simulation code (2026-11-01): what the array code generation generates from.

  - The domain level: a model's equations as classes over domains (`EqClass`, `WhenClass`, `AlgClass`, `AssertClass`:
    a template over iterator slots and the table of the slots' values per instance), in the expression form of the
    generated code (`IR`: literals, slots, element references, operations, numbered relations and event functions),
    the variables laid out (`VarInfo`, `Gen`). Built from the model level of a model with its arrays kept
    (`SimCode.lowerArraySystem`) by `domainSystem` (translate.jl), from the simulation code alone; its passes
    (`domainPasses!`): rerolling (`_reroll`), folding (`_folded`), alias elimination (`_eliminateAliases!`), semiLinear
    chains (`_simplifySemiLinear!`), the discrete variables equations compute as unknowns.
  - The causal level (`causalSchedule!`, `Schedule`; it reads the domain system, filling only its parameter tables):
    the instances' elements
    (`_elements`), the matching (`_match`), the schedule with its loops (`_schedule`, `LoopBlocks`), the large blocks
    torn (`_tearBlock`), the solutions of equations for their unknowns (`_solveFor`), the state dependencies and the
    Jacobian's pattern, the initialization's matching (`InitialMatching`), the rows the right-hand side and the
    relations need and the time-invariant ones.

CodeGeneration's ArrayODEGen emits the module from them.
"""
module Domains

import ..SimCode
import ...Passes

const SC = SimCode
const _OMBACKEND = parentmodule(parentmodule(@__MODULE__))

include("classes.jl")
include("causal.jl")
include("translate.jl")

end # module Domains
