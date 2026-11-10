#=
  The formatter of the generated code, with JuliaFormatter.jl. Julia loads this extension when JuliaFormatter is
  loaded; until then OMBackend.formatText returns its text as it is.
=#
module OMBackendJuliaFormatterExt

import OMBackend
import JuliaFormatter

function __init__()
  OMBackend.FORMAT_TEXT[] = JuliaFormatter.format_text
end

end # module OMBackendJuliaFormatterExt
