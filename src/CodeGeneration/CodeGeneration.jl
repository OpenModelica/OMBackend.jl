#=
# This file is part of OpenModelica.
#
# Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
# c/o Linköpings universitet, Department of Computer and Information Science,
# SE-58183 Linköping, Sweden.
#
# All rights reserved.
#
# THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
# THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
# ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
# RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
# ACCORDING TO RECIPIENTS CHOICE.
#
# The OpenModelica software and the Open Source Modelica
# Consortium (OSMC) Public License (OSMC-PL) are obtained
# from OSMC, either from the above address,
# from the URLs: http:www.ida.liu.se/projects/OpenModelica or
# http:www.openmodelica.org, and in the OpenModelica distribution.
# GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
#
# This program is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of  MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
# IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
#
# See the full OSMC Public License conditions for more details.
#
Author: John Tinnerholm, john.tinnerholm@liu.se
=#

#= TODO make CodeGenerationUtil its own module.. =#
module CodeGeneration

import DataStructures
import ..Util
using MetaModelica
#= This was also introduced in the data structure package. We need to be explicit in which one we use.=#
using MetaModelica: Cons
using Setfield
using DocStringExtensions
using ModelingToolkit

using ..FrontendUtil
using ..Backend #Should maybe not be using here... since it can make certain overloads a bit tricky to follow.
using ..SimulationCode

import ..Backend.BDAE
import ..@BACKEND_LOGGING

import Absyn
import DAE
import MetaGraphs
import OMFrontend
import OMParser
import Symbolics

include("mtkExternals.jl")
include("./CodeGenerationUtil.jl")
include("./structuralCallbacks.jl")
include("./AlgorithmicCodeGeneration.jl")
include("./MTK_CodeGeneration.jl")

#= Pure DifferentialEquations.jl code generation =#
include("./codeGen.jl")

end #= End CodeGeneration=#
