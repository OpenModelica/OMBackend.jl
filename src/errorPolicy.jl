#= The error policy of fallbacks (consolidation item 7, 2026-09-29).

   A `catch` that takes a fallback calls `_fallback(e, :site)` first:
   - interrupts, out-of-memory and stack overflows always propagate;
   - an error that looks like a programming error of OMBackend (UndefVarError,
     FieldError, UndefKeywordError, AssertionError, a MethodError of one of our
     own functions) propagates, unless the site names it in `expect`, or the
     OMBACKEND_FALLBACK_ON_BUG switch (observe mode) is on, which only logs it
     loudly and takes the fallback;
   - every other error takes the fallback, logged once per site per translate:
     @debug when only speed is at stake (impact :perf), @info when the fallback
     changes the result (impact :result: a dropped constraint, event or assert,
     another simulation path).
   IMTKGen's build caught an UndefVarError of a typo and every model silently
   took another simulation path with different results: this is what the
   policy exists for. `fallbackSummary()` lists the fallbacks taken since the
   last translate. =#

using Base.CoreLogging: @logmsg, Debug, Info, Error

isFatal(@nospecialize(e))::Bool = e isa Union{InterruptException, OutOfMemoryError, StackOverflowError}

#= The modules of OMBackend's source (not the model modules evaluated into it
   later): a MethodError of their functions is a programming error. Filled at
   the end of loading (_collectSourceModules!). =#
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

function isBug(@nospecialize(e))::Bool
  e isa Union{UndefVarError, UndefKeywordError, FieldError, AssertionError} && return true
  e isa MethodError && return parentmodule(e.f) in SOURCE_MODULES
  return false
end

mutable struct FallbackRecord
  count::Int
  exceptionType::String
  impact::Symbol
  bug::Bool
end

#= Per site, since the last translate. =#
const FALLBACKS = Dict{Symbol, FallbackRecord}()
resetFallbacks!() = empty!(FALLBACKS)
fallbackSummary() = sort!([(site, r.count, r.exceptionType, r.impact, r.bug) for (site, r) in FALLBACKS]; by = first)

"""
    _fallback(e, site; expect = Union{}, impact = :perf)

Called first in a `catch` that takes a fallback (see the policy above): rethrows
`e` when it is fatal, or a programming error the site does not `expect` (outside
observe mode); otherwise records and logs the fallback once per site and returns.
"""
function _fallback(@nospecialize(e), site::Symbol; expect::Type = Union{}, impact::Symbol = :perf)
  isFatal(e) && rethrow()
  local bug = isBug(e) && !(e isa expect)
  bug && !envSwitch("OMBACKEND_FALLBACK_ON_BUG") && rethrow()
  local r = get!(() -> FallbackRecord(0, string(typeof(e)), impact, bug), FALLBACKS, site)
  r.count += 1
  if r.count == 1
    local level = bug ? Error : impact === :result ? Info : Debug
    @logmsg level "[fallback] $(site)" impact exception = (e, catch_backtrace())
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
