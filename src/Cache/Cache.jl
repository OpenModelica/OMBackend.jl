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
    Cache

The models whose generated code and build the backend keeps for the session, oldest first, each with its size on
the heap (`sizeOf`). When their total passes the limit (`limitBytes`, 6 GiB unless `OMBACKEND_CACHE_LIMIT_GIB`
says otherwise), the oldest models are forgotten (OMBackend.forgetModel) until the rest fits; the model just stored
stays, however large. Storing a model again (a new translate or build) makes it the newest.

What the sizes do not count: the compiled code of a model's generated module, which Julia never frees, and the
memory of external objects (tables) outside the Julia heap, which forgetting a model frees by their destructors.
"""
module Cache

"""
The cached models' canonical names, oldest first.
"""
const MODELS = String[]

"""
The size in bytes of each cached model's code and build when it was stored.
"""
const SIZES = Dict{String, Int}()

"""
The default limit of the cached models' total size, in GiB.
"""
const DEFAULT_LIMIT_GIB = 6.0

"""
    limitBytes() -> Int

The limit of the cached models' total size: `OMBACKEND_CACHE_LIMIT_GIB` (GiB, read on each use), else 6 GiB.
"""
function limitBytes()::Int
  local gib = something(tryparse(Float64, get(ENV, "OMBACKEND_CACHE_LIMIT_GIB", "")), DEFAULT_LIMIT_GIB)
  return round(Int, max(gib, 0.0) * 2^30)
end

"""
What `sizeOf` does not follow: modules (a build's closures reach the generated model module, whose bindings would
count every model's), types, methods and compiled code. A RuntimeGeneratedFunction's body holds GlobalRefs, and
since Julia 1.12 a GlobalRef holds its binding, whose backedges lead into the compiler's code instances and all they
cache: every MSL model with an event iteration measured the session's 1.7 GiB, in 150 s.
"""
const NOT_COUNTED = Union{Module, DataType, Core.TypeName, Core.MethodInstance, Core.CodeInstance, Method, GlobalRef,
                          (isdefined(Core, :Binding) ? Core.Binding : Union{})}

"""
    sizeOf(x) -> Int

The bytes on the Julia heap that `x` reaches (`Base.summarysize`), without `NOT_COUNTED`.
"""
sizeOf(@nospecialize(x))::Int = Base.summarysize(x; exclude = NOT_COUNTED)

"""
    totalBytes() -> Int

The total size of the cached models.
"""
totalBytes()::Int = sum(values(SIZES); init = 0)

"""
    store!(name, bytes; forget = OMBackend.forgetModel) -> Vector{String}

`name` (a canonical model name) cached with `bytes`, as the newest model; then the oldest models forgotten
(`forget(name)`) while the total is over the limit, `name` kept. The models forgotten, oldest first.
"""
function store!(name::String, bytes::Int; forget::Function = _forgetModel)::Vector{String}
  remove!(name)
  push!(MODELS, name)
  SIZES[name] = bytes
  local forgotten = String[]
  local limit = limitBytes()
  while totalBytes() > limit && length(MODELS) > 1
    local oldest = popfirst!(MODELS)
    delete!(SIZES, oldest)
    push!(forgotten, oldest)
    forget(oldest)
  end
  isempty(forgotten) ||
    @info "[cache] over $(round(limit / 2^30; digits = 2)) GiB: forgot the oldest models" forgotten
  return forgotten
end

"""
    remove!(name) -> Nothing

`name` no longer cached (its model forgotten elsewhere). Nothing when it is not.
"""
function remove!(name::String)::Nothing
  local k = findfirst(==(name), MODELS)
  k === nothing || deleteat!(MODELS, k)
  delete!(SIZES, name)
  return nothing
end

"""
    clear!() -> Nothing

No model cached (OMBackend.clearCaches!).
"""
function clear!()::Nothing
  empty!(MODELS)
  empty!(SIZES)
  return nothing
end

"""
OMBackend.forgetModel, looked up when it is called: this module is loaded before OMBackend's API.
"""
_forgetModel(name::String) = getglobal(parentmodule(@__MODULE__), :forgetModel)(name)

end
