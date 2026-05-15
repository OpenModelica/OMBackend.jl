#=
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

#=
This file contains utility macros used by the backend.
=#

using Dates
using DataStructures: OrderedDict

import Absyn
import DAE


"""
Runtime-toggleable backend logging switch. Default is taken from
`ENV["ENABLE_BACKEND_LOGGING"]` at module load; flip at runtime via
`OMBackend.BACKEND_LOGGING[] = true / false` to avoid a Julia restart.
"""
const BACKEND_LOGGING = Ref{Bool}(get(ENV, "ENABLE_BACKEND_LOGGING", "false") == "true")
"""
    @BACKEND_LOGGING expr

Evaluates `expr` only when `BACKEND_LOGGING[]` is true; otherwise the
expression is dropped and a `nothing` literal is emitted. The runtime check
is a single boolean load and branch per call site.
"""
macro BACKEND_LOGGING(expr)
  return quote
    if $(BACKEND_LOGGING)[]
      $(esc(expr))
    else
      nothing
    end
  end
end


"""
Runtime-toggleable VSS debug switch. Same shape as `BACKEND_LOGGING`.
"""
const VSS_DEBUG = Ref{Bool}(get(ENV, "ENABLE_VSS_DEBUG", "false") == "true")
"""
    @VSS_DEBUG expr

Active only when `VSS_DEBUG[]` is true. Use for VSS-runtime debug logging.
"""
macro VSS_DEBUG(expr)
  return quote
    if $(VSS_DEBUG)[]
      $(esc(expr))
    else
      nothing
    end
  end
end

"""
Runtime-toggleable perf logging switch. Default is taken from
`ENV["ENABLE_BACKEND_PERFLOG"]` at module load; flip at runtime via
`OMBackend.BACKEND_PERFLOG[] = true / false` to avoid a Julia restart.
"""
const BACKEND_PERFLOG = Ref{Bool}(get(ENV, "ENABLE_BACKEND_PERFLOG", "false") == "true")

"""
    @BACKEND_PERFLOG "label" expr

Wraps `expr` in `@time` when `BACKEND_PERFLOG[]` is true; otherwise the
expression is evaluated with no instrumentation. The runtime check costs
one boolean load and one branch per call site, so use it on coarse pass
boundaries, not in tight inner loops.
"""
macro BACKEND_PERFLOG(label, expr)
  return quote
    if $(BACKEND_PERFLOG)[]
      @time $(esc(label)) $(esc(expr))
    else
      $(esc(expr))
    end
  end
end

const COMPONENT_SEPARATOR = "_"

function _joinCanonicalSegments(segs)::String
  return join((String(s) for s in segs), COMPONENT_SEPARATOR)
end

function _canonicalIdentSegments(ident::AbstractString)::Vector{String}
  local s = String(ident)
  if startswith(s, "'") && endswith(s, "'")
    return String[s]
  end
  if occursin('.', s)
    return String[String(seg) for seg in split(s, '.')]
  end
  return String[s]
end

function _subscriptSuffix(subscriptLst)::String
  local buf = IOBuffer()
  for s in subscriptLst
    print(buf, "[")
    print(buf, string(s))
    print(buf, "]")
  end
  return String(take!(buf))
end

function _canonicalPathSegments(path::Absyn.IDENT)
  return _canonicalIdentSegments(path.name)
end

function _canonicalPathSegments(path::Absyn.QUALIFIED)
  return vcat(_canonicalIdentSegments(path.name), _canonicalPathSegments(path.path))
end

function _canonicalPathSegments(path::Absyn.FULLYQUALIFIED)
  return _canonicalPathSegments(path.path)
end

function _canonicalCrefSegments(cr::DAE.CREF_IDENT)
  local segs = _canonicalIdentSegments(cr.ident)
  segs[end] = string(segs[end], _subscriptSuffix(cr.subscriptLst))
  return segs
end

function _canonicalCrefSegments(cr::DAE.CREF_QUAL)
  local segs = _canonicalIdentSegments(cr.ident)
  segs[end] = string(segs[end], _subscriptSuffix(cr.subscriptLst))
  return vcat(segs,
              _canonicalCrefSegments(cr.componentRef))
end

function _canonicalCrefSegments(cr::DAE.CREF_ITER)
  local segs = _canonicalIdentSegments(cr.ident)
  segs[end] = string(segs[end], _subscriptSuffix(cr.subscriptLst))
  return segs
end

function _canonicalCrefSegments(::DAE.WILD)
  return String["_"]
end

function canonicalName(name::AbstractString)::String
  local s = String(name)
  if startswith(s, "der(") && endswith(s, ")")
    return string("der(", canonicalName(s[5:end-1]), ")")
  end
  if startswith(s, "'") && endswith(s, "'")
    return s
  end
  if occursin('.', s)
    return _joinCanonicalSegments(split(s, '.'))
  end
  return s
end

canonicalName(path::Absyn.Path)::String = _joinCanonicalSegments(_canonicalPathSegments(path))
canonicalName(cr::DAE.ComponentRef)::String = _joinCanonicalSegments(_canonicalCrefSegments(cr))
canonicalName(exp::DAE.CREF)::String = canonicalName(exp.componentRef)

canonicalSymbol(x)::Symbol = Symbol(canonicalName(x))

#= NameRewriteMap is populated incrementally by `_recordNameRewrite!` while the
   canonicalize pass walks the SimCode tree. Use `canonicalToOriginal[canonical]`
   to recover the dotted Modelica name for diagnostics; the standalone
   "originalName" inverse cannot be reconstructed from a canonical string alone
   because `.` is irreversibly collapsed to `_`. =#
struct NameRewriteMap
  originalToCanonical::OrderedDict{String, String}
  canonicalToOriginal::OrderedDict{String, String}
end

NameRewriteMap() = NameRewriteMap(OrderedDict{String, String}(),
                                  OrderedDict{String, String}())


#= -----------------------------------------------------------------------------
   Log directory helpers.

   All per-stage log files go under a single OS-appropriate root so they do not
   pollute the working directory (repo root during test runs). The root is:

     * `ENV["OMJL_LOG_DIR"]` if set, or
     * `joinpath(tempdir(), "OMJL", OMJL_SESSION_ID)` otherwise.

   `OMJL_SESSION_ID` = "<pid>_<epoch_seconds>", which keeps concurrent Julia
   processes from stomping on each other while letting all dumps from a single
   session live in one directory. Override via env var if needed.

   Use:

     logPath("backend/bdae", "bdae_initial.log")   # returns absolute path and
                                                   # ensures the directory exists

----------------------------------------------------------------------------- =#
const OMJL_SESSION_ID = string(getpid(), "_", round(Int, time()))
const OMJL_LOG_RUN_DIR_STACK = String[]

function sanitizeLogRunName(name::AbstractString)::String
  sanitized = replace(strip(name), r"[\\/:*?\"<>|\s.]+" => "_")
  sanitized = replace(sanitized, r"_+" => "_")
  sanitized = strip(sanitized, '_')
  return isempty(sanitized) ? "run" : sanitized
end

function createLogRunId(modelName::AbstractString; suffix::Union{Nothing, AbstractString} = nothing,
                        timestamp::DateTime = Dates.now())::String
  local base = sanitizeLogRunName(modelName)
  local stamp = Dates.format(timestamp, "yyyy-mm-dd_HH-MM-SS")
  if suffix !== nothing
    base = string(base, "_", sanitizeLogRunName(suffix))
  end
  return string(base, "_", stamp)
end

function currentLogRunDir()
  return isempty(OMJL_LOG_RUN_DIR_STACK) ? nothing : OMJL_LOG_RUN_DIR_STACK[end]
end

function hasActiveLogRunDir()::Bool
  return !isempty(OMJL_LOG_RUN_DIR_STACK)
end

function pushLogRunDir(runDir::AbstractString)::String
  local sanitized = sanitizeLogRunName(runDir)
  local current = currentLogRunDir()
  local nextDir = current === nothing ? sanitized : joinpath(current, sanitized)
  push!(OMJL_LOG_RUN_DIR_STACK, nextDir)
  return nextDir
end

function popLogRunDir()
  return isempty(OMJL_LOG_RUN_DIR_STACK) ? nothing : pop!(OMJL_LOG_RUN_DIR_STACK)
end

function withLogRunDir(f::Function, runDir::AbstractString)
  pushLogRunDir(runDir)
  try
    return f()
  finally
    popLogRunDir()
  end
end

"""
    logDir() -> String

Root directory for OMJL log/dump files in the current session. Respects
`ENV["OMJL_LOG_DIR"]`; otherwise a per-session subdirectory under `tempdir()`.
"""
function logDir()
  local base = get(ENV, "OMJL_LOG_DIR", joinpath(tempdir(), "OMJL", OMJL_SESSION_ID))
  local activeRunDir = currentLogRunDir()
  return activeRunDir === nothing ? base : joinpath(base, activeRunDir)
end

"""
    logPath(stage::AbstractString, filename::AbstractString) -> String

Absolute path for a log file in the given pipeline stage (e.g. `"backend/bdae"`,
`"frontend/flat"`, `"backend/mtk"`). Creates the stage subdirectory on first use.
"""
function logPath(stage::AbstractString, filename::AbstractString)
  dir = joinpath(logDir(), stage)
  mkpath(dir)
  return joinpath(dir, filename)
end
