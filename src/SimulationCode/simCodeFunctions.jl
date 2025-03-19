#=
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

#=
  The code in this file is used to convert frontend functions to a definition that can be used by the backend, and the code generators there.
=#

const FRONTEND_FUNCTION = OMFrontend.Frontend.M_FUNCTION

"""
  Generates algorithmic simcode
TODO:
Handle concrete non variable arguments.
"""
function generateSimCodeFunctions(functionList::List{FRONTEND_FUNCTION})::Tuple{Vector{ModelicaFunction}, Bool}
  local functions = ModelicaFunction[]
  local externalFunctionsUsed = false
  for f in functionList
    local n = string(f.path)
    local inputs = map(f.inputs) do input
      OMFrontend.Frontend.convertFunctionParam(input)
    end
    local outputs = map(f.outputs) do output
      OMFrontend.Frontend.convertFunctionParam(output)
    end
    local locals = map(f.locals) do l
      OMFrontend.Frontend.convertFunctionParam(l)
    end
    if ! OMFrontend.Frontend.isExternal(f)
      local body::List{OMFrontend.Frontend.Statement} = OMFrontend.Frontend.getBody(f)
      local stmts = OMFrontend.Frontend.convertStatements(body)
      local mf = MODELICA_FUNCTION(n, inputs, outputs, locals, listArray(stmts))
      push!(functions, mf)
    else #= The function is a wrapper for some internal builtin Modelica Function =#
      externalFunctionsUsed = true
      s = OMFrontend.Frontend.IOStream_M.create(getInstanceName(), OMFrontend.Frontend.IOStream_M.LIST())
      s = OMFrontend.Frontend.toFlatStream(OMFrontend.Frontend.getSections(f.node), f.path, s)#"dummy"
      str = OMFrontend.Frontend.IOStream_M.string(s)
      #=This should really really not be done by string splitting magic... =#
      local libInfo = first(split(str, "annotation"))
      libInfo = replace(libInfo, "external \"C\"" => "")
      libInfo = replace(libInfo, "'" => "")
      push!(functions, EXTERNAL_MODELICA_FUNCTION(n, inputs, outputs, libInfo))
    end
  end
  return (functions, externalFunctionsUsed)
end
