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
    Passes

The backend's passes on OMJLIR, between the lowering of the flat model (LowerFlatModel) and the
lowering to SimCode. Each is a function of the model, documented with what it does and why;
`PIPELINE` runs them in order.
"""
module Passes

import ..OMJLIR
import ..unsupported, .._fallback, ..UnsupportedLowering
const IR = OMJLIR

include("util.jl")
include("records.jl")
include("algorithms.jl")
include("equations.jl")
include("discrete.jl")
include("clocks.jl")
include("clockedLowering.jl")
include("derivatives.jl")

"""
The passes, in order: name => pass. The first group is the previous backend's BDAECreate (the
lowering's transformations), then clock partitioning and the clocked lowering, then its
Causalize passes.
"""
const PIPELINE = Pair{String, Base.Function}[
  "removeSmooth" => removeSmooth!,
  "decomposeRecordEquations" => decomposeRecordEquations!,
  "liftAlgorithms" => liftAlgorithms!,
  "deduplicateEquations" => deduplicateEquations!,
  "addBindingEquations" => addBindingEquations!,
  "booleanizeWhenConditions" => booleanizeWhenConditions!,
  "liftDiscreteEquations" => liftDiscreteEquations!,
  "hoistIfEquationAsserts" => hoistIfEquationAsserts!,
  "partitionClocks" => partitionClocks!,
  "lowerClockedPartitions" => lowerClockedPartitions!,
  "introduceDerivatives" => introduceDerivatives!,
]

"""
    runPasses!(model)

`model` through the passes of `PIPELINE`, in order.
"""
function runPasses!(model::IR.Model)::IR.Model
  for (_, pass) in PIPELINE
    model = pass(model)
  end
  return model
end

end # module Passes
