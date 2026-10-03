

#= Parameters kept as parameters instead of being folded into the equations at
   compile time, so a compiled model can be simulated again with other values
   (withTunableParameters). Canonical names. Read by the lowering
   (Causalize.resolveCrefBindings!), the SimCode passes (every compile-time
   evaluation of a parameter goes through _boundParameterExpression, so a
   parameter bound to a tunable one is not constant either) and the code
   generator. =#
const TUNABLE_PARAMETERS = Base.ScopedValues.ScopedValue(Set{String}())
#= An array parameter is tunable as a whole: its scalarized elements match
   the array's name, the trailing subscripts removed one at a time
   (`nn_W1[1][2]` → `nn_W1`; in an array of components `nn[1]_W1[2]` →
   `nn[1]_W1`, never `nn`). =#
isTunableParameter(name::AbstractString)::Bool = isTunableParameter(name, TUNABLE_PARAMETERS[])
function isTunableParameter(name::AbstractString, tun::AbstractSet{String})::Bool
  isempty(tun) && return false
  name in tun && return true
  local s = SubString(name)
  while endswith(s, ']')
    local b = findlast('[', s)
    (b === nothing || b == 1) && return false
    s = SubString(s, 1, b - 1)
    s in tun && return true
  end
  return false
end

#= `A[2,1]` → `A[2][1]`, the form OM.jl gives scalarized array elements. =#
_elementSubscripts(name::AbstractString)::String =
  replace(name, r"\[\s*\d+(\s*,\s*\d+)+\s*\]" => m -> join("[" * strip(i) * "]" for i in split(m[2:end-1], ',')))

#= Environment switches of the backend, read on each use: set one in a running
   session and rerun. Kill switches (default on) turn a mechanism off to compare
   with the behaviour before it; diagnostics (default off) print or keep what a
   run did. "true", "1" or "yes" (any case) is on; anything else set is off.
   The logging switches (ENABLE_BACKEND_LOGGING, ENABLE_BACKEND_PERFLOG,
   ENABLE_VSS_DEBUG) are read once at load into Refs (util.jl); OMJL_LOG_DIR
   is a directory. =#
const ENV_SWITCHES = Dict{String, Tuple{Bool, String}}(
  #= Kill switches =#
  "OMBACKEND_ALIAS_INDEX" => (true, "alias start values through an index of the alias equations (off: scan every equation's text)"),
  "OMBACKEND_DEMOTE_CONDTARGETS" => (true, "demote a discrete that an if-equation relay defines"),
  "OMBACKEND_DISCRETE_PRE_MEMORY" => (true, "discrete clusters with event iteration for every model (off: MTK affects where no mode FSM, table or switch cluster needs them)"),
  "OMBACKEND_INIT_ANCHOR" => (true, "initialization phase 2 anchored to the entry guesses (the nearest root)"),
  "OMBACKEND_INIT_COMPLETE" => (true, "complete an underdetermined initialization by null-space projection"),
  "OMBACKEND_INIT_DISCRETES" => (true, "initial fixpoint of the discrete clusters"),
  "OMBACKEND_INIT_RELATIONS" => (true, "initial values of the if-equations' relation literals"),
  "OMBACKEND_INIT_REPIN" => (true, "re-pin the pinned and latched variables after the free initialization phase"),
  "OMBACKEND_MODULE_COMPILE_MIN" => (true, "the generated model module compiles with optimize=0 compile=min infer=false (off: default compilation; longer build)"),
  "OMBACKEND_INIT_ROWSCALE" => (true, "row scaling of the initialization Newton Jacobian"),
  "OMBACKEND_INIT_SYMBOLIC_EQS" => (true, "symbolic initialization residuals of the reduced system"),
  "OMBACKEND_LIVE_IFCOND_AFFECT" => (true, "live affects for chains of if-equation relays"),
  "OMBACKEND_RELAY_T0_FIXEDPOINT" => (true, "initial if-equation branches by a fixed point with their targets' t0 values"),
  "OMBACKEND_TGRAD" => (true, "exact time derivative (tgrad) for DirectRHS problems"),
  "OMBACKEND_WHEN_STRING_SKIP" => (true, "String variables left out of the when callbacks' index bindings"),
  #= Diagnostics =#
  "OMBACKEND_EVENT_TRACE" => (false, "trace the event iteration"),
  "OMBACKEND_FALLBACK_ON_BUG" => (false, "observe mode of the error policy (errorPolicy.jl): a programming error in a fallback's try is logged at @error and the fallback taken instead of propagating"),
  "OMBACKEND_INDEX_DIAG" => (false, "log the equations the index analysis finds over-constraining"),
  "OMBACKEND_INIT_TRACE" => (false, "trace the initialization phases"),
  "OMBACKEND_RELAY_T0_TRACE" => (false, "trace the relays' t0 fixed point"),
  "OMBACKEND_NO_PRECOMPILE_WORKLOAD" => (false, "skip the precompile workload"),
  "OMJL_DUMP_IMTK_SRC" => (false, "write the generated iMTK source"),
  "OMJL_STASH_MODELCODE" => (false, "keep the last generated model code in LAST_MODELCODE"))

"""
    envSwitch(name) -> Bool

The environment switch `name` (see `ENV_SWITCHES`): its default when unset.
"""
function envSwitch(name::String)::Bool
  local value = get(ENV, name, nothing)
  value === nothing && return first(ENV_SWITCHES[name])
  return lowercase(value) in ("true", "1", "yes")
end
