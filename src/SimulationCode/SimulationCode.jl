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

"""
  File: SimulationCode.jl
  Data structures and algorithms used for simulation code.
"""
module SimulationCode

using MetaModelica
using DataStructures
using Setfield

using ..FrontendUtil
using ..Backend

import Absyn
import DAE
import Graphs
import ImmutableList
import MetaGraphs
import OMFrontend

import ..OMBackend
import ..Backend.BDAE
import ..@BACKEND_LOGGING
import ..BackendUtil.GraphAlgorithms
import ..FrontendUtil.Util

import ..OMBackend: TUNABLE_PARAMETERS, isTunableParameter

include("simCodeData.jl")
include("simCodeTraverse.jl")
include("simCodePasses.jl")
include("complexLowering.jl")
include("simVarQueries.jl")
include("simCodeGraphs.jl")
include("trivialEquations.jl")
include("outputOnlyElimination.jl")
include("aliasElimination.jl")
include("constantPropagation.jl")
include("discreteClassification.jl")
include("parameterClosure.jl")
include("canonicalNames.jl")
include("parameterElimination.jl")
include("frozenStates.jl")
include("explicitFold.jl")
include("simCodeDiagnostics.jl")
include("initialValues.jl")
include("simCodeDump.jl")
include("simulationCodeTransformation.jl")
include("simCodeFunctions.jl")
include("simCodeFunctionEval.jl")
include("simCodeCheck.jl")
include("simCodeExpBridge.jl")

end # module SimulationCode
