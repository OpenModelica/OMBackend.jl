"""
The example models of Examples.mo as the backend takes them: flattened by OMFrontend.

    flatModel("HelloWorld") -> (flat model, function list)
"""
module ExampleModels

import OMFrontend

export flatModel

const EXAMPLES_FILE = joinpath(@__DIR__, "Examples.mo")
const EXAMPLES_SCODE = Ref{Any}(nothing)

"""
    flatModel(name) -> Tuple

The flat model of the example `name` (`Examples.<name>`) and its functions, the arguments of
`OMBackend.translate(fm; functionList)`.
"""
function flatModel(name::String)::Tuple
  EXAMPLES_SCODE[] === nothing && (EXAMPLES_SCODE[] = OMFrontend.translateToSCode(OMFrontend.parseFile(EXAMPLES_FILE)))
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("Examples." * name, EXAMPLES_SCODE[])
  return (fm, OMFrontend.cacheToFunctionList(cache))
end

end # module ExampleModels
