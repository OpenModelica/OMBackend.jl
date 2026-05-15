#=
  The backend of OMBackend..
=#

module Backend

using ..FrontendUtil

import ..@BACKEND_LOGGING
import ..@BACKEND_PERFLOG

include("BDAE.jl")
include("BackendEquation.jl")
include("BDAEUtil.jl")
include("BDAECreate.jl")
include("Causalize.jl")

export BDAE, BDAECreate, BackendEquation, BDAEUtil, Causalize, BackendDAEExp

end
