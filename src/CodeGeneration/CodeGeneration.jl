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

module CodeGeneration

import DataStructures
using DataStructures: OrderedDict, OrderedSet
import ..FrontendUtil.Util
using MetaModelica
#= This was also introduced in the data structure package. We need to be explicit in which one we use.=#
using MetaModelica: Cons
using Setfield
using DocStringExtensions
using ModelingToolkit
using LinearAlgebra
import DiffEqBase

using ..FrontendUtil
using ..Backend #Should maybe not be using here... since it can make certain overloads a bit tricky to follow.
using ..SimulationCode

import ..Backend.BDAE
import ..@BACKEND_LOGGING
import ..COMPONENT_SEPARATOR

import Absyn
import DAE
import MetaGraphs
import OMFrontend
import OMParser
import Symbolics
import Symbolics.RuntimeGeneratedFunctions
import SymbolicUtils
import OMRuntimeExternalC

#= Initialize RTG for this module to enable world-age-safe function generation =#
RuntimeGeneratedFunctions.init(@__MODULE__)

include("DAEInitSolve.jl")
include("mtkDump.jl")
include("mtkExternals.jl")
include("./exprRewrite.jl")
include("./arrayUtils.jl")
include("./AlgorithmicCodeGeneration.jl")
include("./CodeGenerationUtil.jl")
using .CodeGenerationUtil
include("./MTK_CodeGenerationUtil.jl")
using .MTK_CodeGenerationUtil
include("./structuralCallbacks.jl")
include("./DirectRHSGeneration.jl")
include("./MTK_CodeGeneration.jl")
include("./DiscreteDummyDemotion.jl")

#= Pure DifferentialEquations.jl code generation (legacy/donor) =#
include("./codeGen.jl")

#= Direct DifferentialEquations.jl code generation (DEMode, fresh emitter) =#
include("./DECodeGeneration.jl")

end #= End CodeGeneration=#
