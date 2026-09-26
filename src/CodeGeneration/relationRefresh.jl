#= Relations with a hysteresis (if-equation branch conditions; MLS 8.5,
   OpenModelica's relationhysteresis). A relation's value is buffered in its
   branch's ifCond, changes only at events, and its crossing function is
   shifted by eps = H * scale relative to that value.

   Event iteration: DiffEq applies one event per step, so a relation whose
   crossing coincides with the event's (a complementary relation on the same
   zero set) or that the event moved past its threshold is not updated by its
   own callback. After any event, every relation is evaluated again from the
   post-event state with OpenModelica's rule: a TRUE relation stays TRUE while
   zc <= eps, a FALSE one becomes TRUE when zc <= -eps (single sweep). =#

#= A getter for a crossing function or scale; a constant (not symbolic) is a
   Real that is not a Symbolics.Num. Not SII.getu: it takes an expression over
   observed variables for a parameter-only one (GetParameterObserved), and the
   generated function then reads the unknowns as undefined globals (the V6
   cylinder rig, `2*v_rel` with v_rel observed). A single variable works, and
   so does building the observed function directly. =#
function _valueGetter(problem, @nospecialize(ex))
  local v = Symbolics.unwrap(ex)
  (v isa Real && !(v isa Symbolics.Num)) && return (integrator -> Float64(v))
  local f = ModelingToolkit.build_explicit_observed_function(problem.f.sys, v)
  return integrator -> f(integrator.u, integrator.p, integrator.t)
end

"""
    withRelationRefresh(callbacks, problem, hSym, entries) -> callbacks

Add the event iteration over `entries` = `(ifCond, crossing function,
scale)`: a DiscreteCallback that runs when an ifCond changed (an event) and
re-evaluates every relation with the hysteresis rule. `hSym` is the
hysteresis parameter H.
"""
function withRelationRefresh(callbacks, problem, hSym::Symbol, entries::Vector)
  isempty(entries) && return callbacks
  local SII = ModelingToolkit.SymbolicIndexingInterface
  local syms = Symbol[e[1] for e in entries]
  local getCond, getH, setters, getters
  try
    getCond = SII.getp(problem, syms)
    getH = SII.getp(problem, hSym)
    setters = [SII.setp(problem, s) for s in syms]
    getters = [(_valueGetter(problem, e[2]), _valueGetter(problem, e[3])) for e in entries]
  catch err
    @warn "[events] the relations cannot be read from the problem; events are not iterated" exception = err
    return callbacks
  end
  local current = integrator -> Float64[v for v in getCond(integrator)]
  local snapshot = Ref(Float64[])
  #= The values the initialization was solved with (from start attributes). =#
  local compiled = Float64[v for v in getCond(problem)]
  local cb = DiffEqBase.DiscreteCallback((u, t, integrator) -> current(integrator) != snapshot[],
                                         function (integrator)
                                           local H = getH(integrator)
                                           local changed = false
                                           for (k, old) in enumerate(current(integrator))
                                             local zc = getters[k][1](integrator)
                                             local eps = H * getters[k][2](integrator)
                                             local new = old > 0.5 ? zc <= eps : zc <= -eps
                                             if new != (old > 0.5)
                                               setters[k](integrator, new ? 1.0 : 0.0)
                                               changed = true
                                             end
                                           end
                                           snapshot[] = current(integrator)
                                           #= Only ever raise the flag: the event's own affect
                                              set it, and clearing it would skip the restart. =#
                                           changed && DiffEqBase.u_modified!(integrator, true)
                                         end;
                                         initialize = function (c, u, t, integrator)
                                           #= Runs after the event callbacks' initialize affects,
                                              which set every relation to its literal value at the
                                              solved initial state. If one differs from the value
                                              the initialization used, the algebraic unknowns are
                                              solved again with it (the states kept). =#
                                           snapshot[] = current(integrator)
                                           snapshot[] != compiled && _resolveAlgebraics!(integrator)
                                           nothing
                                         end,
                                         save_positions = (false, false))
  return DiffEqBase.CallbackSet(callbacks, cb)
end

#= Solve the algebraic unknowns again, the differential states kept (a
   pure ODE has none). =#
function _resolveAlgebraics!(integrator)
  local mm = integrator.f.mass_matrix
  (mm isa LinearAlgebra.UniformScaling || all(!iszero, LinearAlgebra.diag(mm))) && return nothing
  DiffEqBase.initialize_dae!(integrator, OMBackend.OrdinaryDiffEq.BrownFullBasicInit())
  DiffEqBase.u_modified!(integrator, true)
  return nothing
end

"""
    setZCHysteresis!(problem, hSym, reltol)

Set the hysteresis parameter H of the event crossing functions for a solve
with relative tolerance `reltol`: H = 1e-4 * reltol (OpenModelica's tolZC).
Nothing for a problem without it.
"""
function setZCHysteresis!(problem, hSym::Symbol, reltol)
  local SII = ModelingToolkit.SymbolicIndexingInterface
  local present = try
    SII.parameter_index(problem, hSym) !== nothing
  catch
    false
  end
  present || return problem
  SII.setp(problem, hSym)(problem, 1.0e-4 * max(Float64(first(reltol)), 1.0e-12))
  return problem
end

#= Callbacks that change parameters at initialization (the literal relation
   values) make the integrator save t0 twice; interpolating at t0 then uses
   the zero-length interval and returns NaN. The solution starts with the
   post-initialization values, as a Modelica result does. =#
function dropPreInitializationPoint!(sol)
  (length(sol.t) >= 2 && sol.t[1] == sol.t[2]) || return sol
  local n = length(sol.t)
  deleteat!(sol.t, 1)
  length(sol.u) == n && deleteat!(sol.u, 1)
  local k = hasproperty(sol, :k) ? sol.k : nothing
  (k isa AbstractVector && length(k) == n) && deleteat!(k, 1)
  local interp = hasproperty(sol, :interp) ? sol.interp : nothing
  if interp !== nothing && hasproperty(interp, :alg_choice)
    local ac = interp.alg_choice
    (ac isa AbstractVector && length(ac) == n) && deleteat!(ac, 1)
  end
  return sol
end
