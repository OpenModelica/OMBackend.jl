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
    OMBackendOMFrontendExt

OMFrontend.jl's flat model through OMBackend, loaded with OMFrontend: the lowering into OMJLIR (`LowerFlatModel`),
the backend's entry points on the FlatModel (`translate`, `translateReducedArrays`, `omjlirModel`,
`explainIndexReduction`, ...; translate.jl), the frontend's part of a recompilation at a structural change
(`Recompilation`), the URI resolution of external functions' include directories, and the translate precompile
workload. OMBackend alone takes OMJLIR (`translate(::OMJLIR.Model)`; omc's frontend through OMBackendOMCompilerExt).
"""
module OMBackendOMFrontendExt

#= The process that precompiles this extension loads the packages in the order of these imports. On Windows the
   order decides which gettext runtime the process gets: OMRuntimeExternalC (a dependency of OMBackend, and of
   OMFrontend) loads its bundled libintl-8.dll, Glib's libraries (Plots) cannot be loaded against it later, and the
   process waits behind Windows' "Entry Point Not Found" dialog (the precompilation never ends). This extension does not
   load Plots; a session that does must load Plots first (OM does, OMBackend.jl#101). =#
import OMBackend
import MetaModelica
import OMFrontend
import OMBackend: OMJLIR, Passes, SimCode, IMTKGen, DEFAULT_BACKEND_MODE, IMTK_MODE, WARN_MISSING_START_VALUES,
  MODEL_RUN_DIRS, ARRAY_ODE_MODELS, ARRAY_REDUCED_MODELS, ARRAY_PATH_REASONS, COMPILED_MODELS_MTK, canonicalName,
  createLogRunId, resetFallbacks!, withLogRunDir, debugWrite, logPath, envSwitch, isFatal, declineArrayPath!,
  generateTargetCode, scalarizedReducedModel, generateReducedArrayODETargetCode, simulateModel,
  defaultSolver, daeFallbackSolver, @BACKEND_PERFLOG, @BACKEND_LOGGING
import OMBackend: translate, explainIndexReduction, omjlirModel, arrayReducedModel, reducedArrayModel,
  translateReducedArrays, lowerFlatModel, resolveModelicaURI

const F = OMFrontend.Frontend

include("LowerFlatModel.jl")
import .LowerFlatModel: FunctionList
include("translate.jl")
include("recompilation.jl")

"The file of a `modelica://` URI, where OMFrontend loaded the library from (OMFrontend.resolveModelicaURI)."
resolveModelicaURI(uri::String)::String = OMFrontend.resolveModelicaURI(uri)

include("precompile.jl")

end # module OMBackendOMFrontendExt
