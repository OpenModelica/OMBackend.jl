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



#= Parameters kept as parameters instead of being folded into the equations at
   compile time, so a compiled model can be simulated again with other values
   (withTunableParameters). Canonical names. Read by the lowering
   (Causalize.resolveCrefBindings!), the SimCode passes (every compile-time
   evaluation of a parameter goes through _boundParameterExpression, so a
   parameter bound to a tunable one is not constant either) and the code
   generator. =#
const TUNABLE_PARAMETERS = Base.ScopedValues.ScopedValue(Set{String}())
"""
An array parameter is tunable as a whole: its scalarized elements match
the array's name, the trailing subscripts removed one at a time
(`nn_W1[1][2]` → `nn_W1`; in an array of components `nn[1]_W1[2]` →
`nn[1]_W1`, never `nn`).
"""
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

"""
`A[2,1]` → `A[2][1]`, the form OM.jl gives scalarized array elements.
"""
_elementSubscripts(name::AbstractString)::String =
  replace(name, r"\[\s*\d+(\s*,\s*\d+)+\s*\]" => m -> join("[" * strip(i) * "]" for i in split(m[2:end-1], ',')))

#= Environment switches of the backend, read on each use: set one in a running
   session and rerun. Kill switches (default on) turn a mechanism off to compare
   with the behaviour before it; diagnostics (default off) print or keep what a
   run did; research features (default off) are new paths not yet the default. "true", "1" or
   "yes" (any case) is on; anything else set is off.
   The logging switches (ENABLE_BACKEND_LOGGING, ENABLE_BACKEND_PERFLOG,
   ENABLE_VSS_DEBUG) are read once at load into Refs (util.jl); OMJL_LOG_DIR
   is a directory. =#
const ENV_SWITCHES = Dict{String, Tuple{Bool, String}}(
  #= Kill switches =#
  "OMBACKEND_ALIAS_INDEX" => (true, "alias start values through an index of the alias equations (off: scan every equation's text)"),
  "OMBACKEND_DEMOTE_CONDTARGETS" => (true, "demote a discrete that an if-equation relay defines"),
  "OMBACKEND_DISCRETE_PRE_MEMORY" => (true, "discrete clusters with event iteration for every model (off: MTK affects where no mode FSM, table or switch cluster needs them)"),
  "OMBACKEND_INIT_ANCHOR" => (true, "initialization phase 2 anchored to the entry guesses (the nearest root)"),
  "OMBACKEND_INIT_FIX_STATES" => (true, "an underdetermined initialization completed as OpenModelica and Dymola do: the unfixed states a matching leaves free fixed at their start values, the square remainder solved (off: phase 2's least squares over all states)"),
  "OMBACKEND_INIT_COMPLETE" => (true, "complete an underdetermined initialization by null-space projection"),
  "OMBACKEND_INIT_DISCRETES" => (true, "initial fixpoint of the discrete clusters"),
  "OMBACKEND_INIT_RELATIONS" => (true, "initial values of the if-equations' relation literals"),
  "OMBACKEND_INIT_REPIN" => (true, "re-pin the pinned and latched variables after the free initialization phase"),
  "OMBACKEND_CALL_PARTIALS" => (true, "partials by central differences of the terms of a function's element extractors and flat calls in the symbolic Jacobian (off: no rule, the whole Jacobian by finite differences)"),
  "OMBACKEND_COMPILED_FUNCTIONS" => (true, "the model's Modelica functions compiled, as RuntimeGeneratedFunctions of CodeGeneration (off: closures of the model module, interpreted under OMBACKEND_MODULE_COMPILE_MIN)"),
  "OMBACKEND_MODULE_COMPILE_MIN" => (true, "the generated model module compiles with optimize=0 compile=min infer=false (off: default compilation; longer build)"),
  "OMBACKEND_INIT_ROWSCALE" => (true, "row scaling of the initialization Newton Jacobian"),
  "OMBACKEND_INIT_SYMBOLIC_EQS" => (true, "symbolic initialization residuals of the reduced system"),
  "OMBACKEND_LIVE_IFCOND_AFFECT" => (true, "live affects for chains of if-equation relays"),
  "OMBACKEND_RELAY_T0_FIXEDPOINT" => (true, "initial if-equation branches by a fixed point with their targets' t0 values"),
  "OMBACKEND_TGRAD" => (true, "exact time derivative (tgrad) for DirectRHS problems"),
  "OMBACKEND_WHEN_STRING_SKIP" => (true, "String variables left out of the when callbacks' index bindings"),
  "OMBACKEND_IF_READS_IRREDUCIBLE" => (false, "the variables an if-equation of several targets reads (its conditions' and equations' algebraic variables) kept unknowns as its targets are (off: only its targets and the states it differentiates; a condition variable an explicit equation defines stays explicit)"),
  "OMBACKEND_INDEX_REDUCTION" => (true, "OMBackend's own index reduction and state selection on OMJLIR (Passes.IndexReduction; off: MTK's structural_simplify reduces the index)"),
  "OMBACKEND_DYNAMIC_STATE_SETS" => (true, "with OMBACKEND_INDEX_REDUCTION: dynamic state sets (OpenModelica's form, x = A * V), switched while the simulation runs where the choice depends on the state (one row once the static dummies are eliminated: a quaternion's norm, a pendulum's positions with its velocities), and OpenModelica's ranking of variables without a StateSelect; off: the choice at the start values"),
  "OMBACKEND_ALIAS_SUBSTITUTION" => (true, "with OMBACKEND_INDEX_REDUCTION: the alias sets (a = ±b, a = c) read through one representative before index reduction (Passes.IndexReduction.substituteAliases!; the alias equations kept), undone where the model is not reduced; off: index reduction reads them as written"),
  "OMBACKEND_ARRAY_INDEX_REDUCTION" => (true, "with OMBACKEND_INDEX_REDUCTION: a model translated with its arrays kept (scalarize = false) whose index is above one reduced in array form (Passes.IndexReduction.reduceIndexArrays!: planned on its elements, for-equations differentiated with their iterators, arrays of one kind), then scalarized for the simulation code; off, or where the array form does not take the plan: scalarized in the frontend as before"),
  "OMBACKEND_LINEAR_CONSTRAINTS" => (true, "with OMBACKEND_INDEX_REDUCTION: the linear constraints among the states that the structure does not show (a combination of the linear equations of literal coefficients in which the other unknowns cancel: three chokes feeding a diode bridge whose DC side is not grounded give i1 + i2 + i3 = 0) written in place of one equation each before index reduction (Passes.IndexReduction.exposeLinearConstraints!), so Pantelides differentiates them; off: the equations as written"),
  "OMBACKEND_PREFER_STATES" => (true, "with OMBACKEND_INDEX_REDUCTION: a continuous variable of StateSelect.prefer that the model does not differentiate an artificial state, as OpenModelica's (Pantelides differentiates the equations that make it one; given up where they do not differentiate): MSL Fluid's medium states instead of a volume's m and U, omc's states for PumpingSystem, the Machines' idq currents; off: only StateSelect.always"),
  "OMBACKEND_SOLVE_DERIVATIVES" => (true, "with OMBACKEND_INDEX_REDUCTION: the derivative unknowns of a reduced model solved for and substituted (Passes.IndexReduction.solveDerivatives!), so the code generators compute them from the other unknowns; off: they stay algebraic unknowns, kept by MTK"),
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
