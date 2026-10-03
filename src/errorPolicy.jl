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

#= The error policy of fallbacks (consolidation item 7, 2026-09-29).

   A `catch` that takes a fallback calls `_fallback(e, :site)` first:
   - interrupts, out-of-memory and stack overflows always propagate;
   - an error that looks like a programming error of OMBackend (UndefVarError,
     FieldError, UndefKeywordError, a MethodError of one of our own functions,
     an AssertionError of our own code) propagates, unless the site names it
     in `expect`, or the OMBACKEND_FALLBACK_ON_BUG switch (observe mode,
     default off) is on, which only logs it loudly and takes the fallback;
   - a site that names what it `only` takes (a lowering: UnsupportedLowering)
     treats any other error as a programming error too;
   - every other error takes the fallback, logged once per site per translate:
     @debug when only speed is at stake (impact :perf), @info when the fallback
     changes the result (impact :result: a dropped constraint, event or assert,
     another simulation path).
   IMTKGen's build caught an UndefVarError of a typo and every model silently
   took another simulation path with different results: this is what the
   policy exists for. `fallbackSummary()` lists the fallbacks taken since the
   last translate. =#

using Base.CoreLogging: @logmsg, Debug, Info, Error

"""
    UnsupportedLowering(what, construct)

A construct that code generation does not support (an expression, a
condition, a statement or an equation form to lower to Julia/MTK code).
Thrown through `unsupported(what, construct)`; a catch around a lowering
narrows to it. Never a programming error of the policy's kind: a model
may contain such a construct.
"""
struct UnsupportedLowering <: Exception
  what::String
  construct::Any
end
Base.showerror(io::IO, e::UnsupportedLowering) =
  print(io, "UnsupportedLowering: ", e.what, ": ", first(string(e.construct), 300))
unsupported(what::AbstractString, @nospecialize(construct)) = throw(UnsupportedLowering(what, construct))

isFatal(@nospecialize(e))::Bool = e isa Union{InterruptException, OutOfMemoryError, StackOverflowError}

#= The modules of OMBackend's source (not the model modules evaluated into it
   later): a MethodError of their functions, or an assertion in them, is a
   programming error. The Modelica functions' implementations are evaluated
   into CodeGeneration, so a MethodError of one counts as ours too (the calls
   are generated). Filled at the end of loading (_collectSourceModules!). =#
const SOURCE_MODULES = Set{Module}()

function _collectSourceModules!(m::Module)
  push!(SOURCE_MODULES, m)
  for n in names(m; all = true)
    isdefined(m, n) || continue
    local v = getfield(m, n)
    v isa Module && v !== m && parentmodule(v) === m && !(v in SOURCE_MODULES) && _collectSourceModules!(v)
  end
  return SOURCE_MODULES
end

#= `bt` is the error's backtrace. An AssertionError is ours only when our code
   asserted: ModelingToolkit's and Symbolics' assertions depend on the model
   (a fallback). =#
function isBug(@nospecialize(e), bt)::Bool
  e isa Union{UndefVarError, UndefKeywordError, FieldError} && return true
  e isa AssertionError && return _thrownInSourceModules(bt)
  e isa MethodError && return _isOwnFunction(e.f)
  return false
end

#= A callable object's module is its type's; a type's (a constructor) its own. =#
_isOwnFunction(@nospecialize(f))::Bool = parentmodule(f isa Union{DataType, UnionAll} ? f : typeof(f)) in SOURCE_MODULES

#= The first frame outside Base and Core (its throw helpers) is in our source. =#
function _thrownInSourceModules(bt)::Bool
  for frame in Base.StackTraces.stacktrace(bt)
    local m = _frameModule(frame)
    (m === nothing || m === Base || m === Core) && continue
    return m in SOURCE_MODULES
  end
  return false
end

function _frameModule(frame)::Union{Module, Nothing}
  local x = frame.linfo
  while x isa Union{Core.MethodInstance, Core.CodeInstance}
    x = x.def
  end
  return x isa Method ? x.module : x isa Module ? x : nothing
end

mutable struct FallbackRecord
  count::Int
  exceptionType::String
  impact::Symbol
  bug::Bool
end

#= Per site, since the last translate (under a lock: run-time sites can fire
   from several threads, e.g. an ensemble). =#
const FALLBACKS = Dict{Symbol, FallbackRecord}()
const FALLBACKS_LOCK = ReentrantLock()
resetFallbacks!() = @lock FALLBACKS_LOCK empty!(FALLBACKS)
fallbackSummary() = sort!([(site, r.count, r.exceptionType, r.impact, r.bug) for (site, r) in FALLBACKS]; by = first)

"""
    _fallback(e, site; expect = Union{}, only = Any, impact = :perf)

Called first in a `catch` that takes a fallback (see the policy above): rethrows
`e` when it is fatal, not of the type the site `only` takes, or a programming
error the site does not `expect` (outside observe mode); otherwise records and
logs the fallback once per site and returns.
"""
function _fallback(@nospecialize(e), site::Symbol; expect::Type = Union{}, only::Type = Any, impact::Symbol = :perf)
  isFatal(e) && rethrow()
  local bt = catch_backtrace()
  local bug = !(e isa only) || (!(e isa expect) && isBug(e, bt))
  bug && !envSwitch("OMBACKEND_FALLBACK_ON_BUG") && rethrow()
  local first = @lock FALLBACKS_LOCK begin
    local r = get!(() -> FallbackRecord(0, string(typeof(e)), impact, bug), FALLBACKS, site)
    r.count += 1
    r.count == 1
  end
  if first
    local level = bug ? Error : impact === :result ? Info : Debug
    @logmsg level "[fallback] $(site)" impact exception = (e, bt)
  end
  return nothing
end

"""
    _tryOr(f, default, site; kw...)

`f()`, or `default` when it throws something `_fallback` lets pass.
"""
function _tryOr(f, default, site::Symbol; kw...)
  try
    return f()
  catch e
    _fallback(e, site; kw...)
    return default
  end
end
