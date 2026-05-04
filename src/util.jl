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


"""
Enable logging by specifing the following environmental variable before compiling the backend.

```
ENABLE_BACKEND_LOGGING=true julia
```
"""
const ENABLE_BACKEND_LOGGING::Bool = get(ENV, "ENABLE_BACKEND_LOGGING", "false") == "true"
"""
If  ```ENABLE_BACKEND_LOGGING``` is true this macro is active. If not this macro represents a NOP.
"""
macro BACKEND_LOGGING(expr)
  if ENABLE_BACKEND_LOGGING
    return esc(expr)
  else
    return nothing
  end
end


const ENABLE_VSS_DEBUG::Bool = get(ENV, "ENABLE_VSS_DEBUG", "false") == "true"
"""
If ```ENABLE_VSS_DEBUG``` is true this macro is active. If not this macro represents a NOP.
Use for debug logging in the VSS (Variable Structure Systems) runtime.
"""
macro VSS_DEBUG(expr)
  if ENABLE_VSS_DEBUG
    return esc(expr)
  else
    return nothing
  end
end


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
