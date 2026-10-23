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
    IndexReduction

Index reduction and state selection on OMJLIR's continuous partition (the switch
`OMBACKEND_INDEX_REDUCTION`, on by default; off: MTK's structural_simplify reduces the
index as before). The steps: the partition's structure (its scalar rows and the continuous
variables they read), Pantelides' algorithm (which equations to differentiate, how often),
the differentiated equations (`Differentiate`), dummy derivatives with Modelica's state
selection, and dynamic state sets where any fixed choice of states turns singular; after the
commit, the derivative unknowns solved for and substituted (`solveDerivatives!`, the switch
`OMBACKEND_SOLVE_DERIVATIVES`, on by default).

The analysis works on a structure of its own and changes the model only once every step has
succeeded: where it meets something it does not handle, the model is left as it was (the
MTK path reduces it), with the reason logged once.

The choice of states (`chooseDummies`, each block of differentiated equations): its dummy
derivatives make the block's Jacobian regular, the candidates ranked by

1. the StateSelect of the variable's alias set, then of the variable itself (never, avoid,
   default, prefer; always only where nothing else regularizes);
2. a variable whose derivative only the differentiated equations read before one whose
   derivative the model's own equations read (a natural state: a filter's output, a body's
   position);
3. a constant entry of the Jacobian before one that reads the unknowns (`x = L * sin(phi)`
   keeps phi the state);
4. the largest pivot of the Jacobian at the start values (where those are not regular, their
   pivots kept and the rest at a generic point that keeps the relations explicit equations
   give, `propagatedValues`, then at `genericValues`, then structurally);
5. among equals, as dummy first the derivative of a derivative variable (`\$DER.x`, which has
   no name of its own), then a variable without a fixed start, then the later declared: the
   first declared variables, and those the user fixed, stay states.

OpenModelica chooses the same states for most models; otherwise an alias of ours, another
coordinate of a rigid chain, or a dynamic state set (here behind `OMBACKEND_DYNAMIC_STATE_SETS`).

With arrays kept (`OMBACKEND_ARRAY_INDEX_REDUCTION`, Arrays.jl, ArrayCommit.jl, ArraySets.jl): the
plan is made on the model's element view (its arrays as their elements, its for-equations
unrolled: the scalar graph of Abdelhak, Casella and Bachmann's Pseudo Array Causalization) and
committed in array form where it treats the elements of each array alike: a for-equation
differentiated once with its iterator kept (checked against every iteration's own derivative),
one kind per array, the derivative arrays whole, the dynamic state sets of an array's elements
one family in a for-equation. Where it does not (a boundary element, some elements states:
slicing, not yet), NotHandled, and the frontend scalarizes.
"""
module IndexReduction

import ..OMJLIR
import ..substituteIterators, ..foldSubscripts, ..expressionLargerThan, ..foldInteger, ..mapRoots, ..rowMajor,
       ..unsubscriptedName, ..allSubscripts, ..literalIndices
import ...envSwitch
import ..Evaluation: arithmetic, asRecord, divide, elementOf, evaluate, foldScalar, genericOffset,
       genericValues, ifThenElse, inlineBody, isLiteral, isOne, isZero, isZeroExp, literalCondition,
       literalValue, minus, negate, ONE, plus, power, real, selectElement, StartValues, subscripted,
       times, withKnownValues, functionDefinedValues, ZERO, zeroOf
const IR = OMJLIR

include("Structure.jl")
include("LinearConstraints.jl")
include("Pantelides.jl")
include("Differentiate.jl")
include("Aliases.jl")
include("DummyDerivatives.jl")
include("SolveDerivatives.jl")
include("Commit.jl")
include("Arrays.jl")
include("ArrayCommit.jl")
include("ArraySets.jl")
include("Explain.jl")

end # IndexReduction
