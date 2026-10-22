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

#= The re-solve of the algebraic unknowns at an event: Newton with the
   problem's own Jacobian first (`_eventNewton!`), then damped, then Levenberg-Marquardt
   with it (`_eventLevenbergMarquardt!`), then BrownFullBasicInit,
   both at its own tolerance (1e-10 on the residual: relations are decided on the
   solved values; a looser one skips the solve where a switched diode's
   equations barely change the residual, and its s keeps the wrong sign).
   Where that is out of reach (large values, a switched Jacobian near
   singular: a thyristor bridge's commutation stops at 4e-9 with values of
   1e6), the result stands if it did not raise the residual and the
   residual is within the solve's abstol, as the initialization's
   (defaultInitializeKwargs), or at its round-off floor, or the unknowns are
   within the solve's tolerance (`_withinSolveTolerance`). Not raised: a
   second re-solve at the same event (after the move to the event time,
   after the affect) starts where the first stopped. A type, because a
   callback's reinitializealg is fixed when the code is generated. =#
struct EventReinit <: ModelingToolkit.SciMLBase.DAEInitializationAlgorithm end

"""
SciMLBase's fallback takes any algorithm for any integrator,
OrdinaryDiffEqCore's any for its own. The first re-solve at a new time
begins the instant (eventInstant.jl `beginInstant!`): a branch event's
callback re-solves where it moves the integrator to the event's time,
before its affect and before the event iteration runs.
"""
DiffEqBase.initialize_dae!(integrator::ModelingToolkit.SciMLBase.DEIntegrator, ::EventReinit) =
  (beginInstant!(integrator); _eventReinit!(integrator))
DiffEqBase.initialize_dae!(integrator::OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator, ::EventReinit) =
  (beginInstant!(integrator); _eventReinit!(integrator))

"""
The re-solve at the start of a solve (`defaultInitializeKwargs`), after the initialization:
the event re-solve's stages and acceptance (`_eventReinit!`) with `alg` (BrownFullBasicInit at
the solve's abstol, or the table-cluster models' `tableClusterInitAlg`). BrownFullBasicInit
alone held the raw residual to its abstol: MSL AST_BatchPlant TanksWithEmptyingPipe2's empty
port (zetaLarge = 1e5) stopped at 1.4e-4, a flow error of 1e-11, and the solve ended in
InitialFailure.
"""
struct StartReinit{A <: DiffEqBase.BrownFullBasicInit} <: ModelingToolkit.SciMLBase.DAEInitializationAlgorithm
  alg::A
end

DiffEqBase.initialize_dae!(integrator::ModelingToolkit.SciMLBase.DEIntegrator, s::StartReinit) =
  _eventReinit!(integrator, s.alg)
DiffEqBase.initialize_dae!(integrator::OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator, s::StartReinit) =
  _eventReinit!(integrator, s.alg)

"""
    _eventNewton!(integrator; tol, maxiters, damped = false) -> Bool

Newton on the algebraic rows of a mass-matrix problem with the problem's own Jacobian (its
algebraic block, factorized sparse when the Jacobian is), the differential unknowns kept, to
`tol` on the residual, or, once it stalls (the residual no longer halves), to each row's
round-off floor (`tol` plus 1e3 eps times the size of its terms, as `_atRoundoffFloor`: an
ideal switch's rows carry terms of 1e9, MSL CauerLowPassSC's re-solves stopped at 1-3e-10 and
took the polyalgorithm, 0.3 s each), or with its next correction within the solve's tolerance
(`_withinSolveTolerance`'s norm): the floor is an estimate, and a property function's own
rounding can exceed it (MSL Fluid MeasuringTemperature on the index-reduction path: a tank's
port density from IF97's `rho_ph`, 979, stalled at 3.3e-10 against a floor of 3.2e-10, a
correction of 1e-6, and the polyalgorithm took 70-95 s, four times at one event). Whether it
converged; `integrator.u` changes only then
(its algebraic unknowns, as BrownFullBasicInit's). At an event's crossing a relation without an
event (`noEvent`, a stream port's upstream side) can switch within a finite-difference step, and
BrownFullBasicInit's Jacobian of differences is wrong there: MSL Fluid MeasuringTemperature's
tank port flows at 1e-10 at their reversal took re-solves of 70-80 s (1352 nonlinear steps),
1 to 3 iterations with the problem's Jacobian. Float64 states only (a sparse LU of dual
numbers has no method). `damped`: each step's fraction from `_dampedFraction` (the stage
after undamped Newton; always damped, the largest residual of rows of different scales does
not fall monotonically where undamped Newton converges, and MSL WaterPump and HBridge_DC_Drive
failed).
"""
function _eventNewton!(integrator::ModelingToolkit.SciMLBase.DEIntegrator; tol::Float64, maxiters::Int = 10,
                       damped::Bool = false)::Bool
  local f = integrator.f
  (eltype(integrator.u) === Float64 && hasproperty(f, :jac) && f.jac !== nothing &&
   hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing && ModelingToolkit.SciMLBase.isinplace(f)) || return false
  local rows = _algebraicRows(f)
  isempty(rows) && return false
  local (p, t) = (integrator.p, integrator.t)
  local u = copy(integrator.u)
  local residual = similar(u)
  local jac = similar(f.jac_prototype, Float64)
  local previous = Inf
  for _ in 0:maxiters
    f(residual, u, p, t)
    local g = residual[rows]
    all(isfinite, g) || return false
    local size = maximum(abs, g)
    if size <= tol
      copyto!(integrator.u, u)
      return true
    end
    f.jac(jac, u, p, t)
    #= the round-off floor once Newton stalls: while it converges, a residual under the floor can
       still fall (a solution to round-off, not before it) =#
    local stalled = size > previous / 2
    previous = size
    local floor = (abs.(jac) * abs.(u))[rows]
    if stalled && all(k -> abs(g[k]) <= tol + 1e3 * eps() * floor[k], eachindex(g))
      copyto!(integrator.u, u)
      return true
    end
    local F = LinearAlgebra.lu(jac[rows, rows]; check = false)
    LinearAlgebra.issuccess(F) || return false
    local step = F \ g
    if stalled && _scaledNorm(rows, integrator, step, u) <= 1
      copyto!(integrator.u, u)
      return true
    end
    u[rows] .-= (damped ? _dampedFraction(f, residual, u, rows, step, p, t, size) : 1.0) .* step
  end
  return false
end

"""
The fraction of a Newton step that lowers the largest residual, halved from 1 up to 10 times;
1 where none does (the full step, as undamped Newton takes it); damped Newton's
(`_eventNewton!`'s `damped`, after undamped Newton failed). At a pipe's or a port's flow
characteristic near zero flow (a square root of the pressure difference) the full step
overshoots to the mirror point and Newton oscillates between them: MSL Fluid TanksWithOverflow
when the upper tank's level falls below its overflow port, the port's pressure jumping by
±35 kPa. `residual` is a work vector.
"""
function _dampedFraction(f, residual::Vector{Float64}, u::Vector{Float64}, rows::Vector{Int}, step::Vector{Float64},
                         p, t::Float64, size::Float64)::Float64
  local trial = copy(u)
  local λ = 1.0
  for _ in 0:10
    trial[rows] .= u[rows] .- λ .* step
    f(residual, trial, p, t)
    local r = maximum(k -> abs(residual[k]), rows)
    (isfinite(r) && r < size) && return λ
    λ /= 2
  end
  return 1.0
end

"""
    _eventLevenbergMarquardt!(integrator; tol, maxiters) -> Bool

The algebraic rows of a mass-matrix problem solved by Levenberg-Marquardt (NonlinearSolve) with the problem's own
Jacobian (its algebraic block, dense: up to `EVENT_LM_MAX_UNKNOWNS` unknowns), the differential unknowns kept, where
Newton did not converge (`_eventNewton!`). MSL 4 Fluid IncompressibleFluidNetwork at t = 50, valve1 closing and the
flow through pipe8 reversing, its pipes' temperatures the states: the network's pressure level weakly tied to its
boundaries (condition 1.2e10), Newton diverged along the pressures and BrownFullBasicInit stopped at 1.8e-4;
Levenberg-Marquardt reached 4.8e-9. Accepted at `tol`, or within the solve's abstol where it lowered the residual
(as the polyalgorithm's result is, `_eventReinit!`). Whether it was accepted; `integrator.u` changes only then.
"""
function _eventLevenbergMarquardt!(integrator::ModelingToolkit.SciMLBase.DEIntegrator; tol::Float64, maxiters::Int = 200)::Bool
  local f = integrator.f
  (eltype(integrator.u) === Float64 && hasproperty(f, :jac) && f.jac !== nothing &&
   hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing && ModelingToolkit.SciMLBase.isinplace(f)) || return false
  local rows = _algebraicRows(f)
  (isempty(rows) || length(rows) > EVENT_LM_MAX_UNKNOWNS) && return false
  local (p, t) = (integrator.p, integrator.t)
  local u0 = copy(integrator.u)
  local u = copy(u0)
  local residual = similar(u0)
  local jac = similar(f.jac_prototype, Float64)
  local rowsOf! = (out, x, _) -> begin
    u[rows] .= x
    f(residual, u, p, t)
    out .= view(residual, rows)
    return nothing
  end
  local jacOf! = (J, x, _) -> begin
    u[rows] .= x
    f.jac(jac, u, p, t)
    J .= jac[rows, rows]
    return nothing
  end
  local problem = NonlinearSolve.NonlinearProblem(NonlinearSolve.NonlinearFunction(rowsOf!; jac = jacOf!), u0[rows])
  local solution = NonlinearSolve.solve(problem, NonlinearSolve.LevenbergMarquardt(); abstol = tol, maxiters)
  local entry = similar(solution.u)
  rowsOf!(entry, u0[rows], nothing)
  local final = similar(solution.u)
  rowsOf!(final, solution.u, nothing)
  all(isfinite, final) || return false
  local size = maximum(abs, final)
  (size <= tol || (size <= _solveAbstol(integrator) && size <= maximum(abs, entry))) || return false
  integrator.u[rows] .= solution.u
  return true
end

"The largest algebraic block `_eventLevenbergMarquardt!` solves (dense)."
const EVENT_LM_MAX_UNKNOWNS = 400

"""
`alg`: BrownFullBasicInit at its own tolerance, or the table-cluster
models' (tableClusterInitAlg); the same fallback for both (the MSL QS
IMC_Transformer's switching event at t = 2 stopped at a residual within
the solve's abstol but above the table algorithm's 1e-8). Newton with the
problem's Jacobian first, to the same tolerance (`_eventNewton!`), then
damped Newton (`_dampedFraction`), Levenberg-Marquardt with it (`_eventLevenbergMarquardt!`).
"""
function _eventReinit!(integrator::ModelingToolkit.SciMLBase.DEIntegrator, alg::DiffEqBase.BrownFullBasicInit = DiffEqBase.BrownFullBasicInit())
  local solved = try
    _eventNewton!(integrator; tol = alg.abstol) || _eventNewton!(integrator; tol = alg.abstol, damped = true)
  catch err
    OMBackend._fallback(err, :eventNewton)
    false
  end
  solved && return nothing
  local fitted = try
    _eventLevenbergMarquardt!(integrator; tol = alg.abstol)
  catch err
    OMBackend._fallback(err, :eventLevenbergMarquardt)
    false
  end
  fitted && return nothing
  local InitialFailure = ModelingToolkit.SciMLBase.ReturnCode.InitialFailure
  local before = integrator.sol.retcode
  local u0 = copy(integrator.u)
  try
    DiffEqBase.initialize_dae!(integrator, alg)
  catch err
    OMBackend._fallback(err, :eventInitializeDae)
    #= A method of the default polyalgorithm can throw (NonlinearSolve's
       MoreTrustRegion descent, `restructure(x, nothing)`, on the MSL Spice3
       Inverter's switching MOSFETs): Newton with finite differences instead,
       from the state before; a failure if that throws too. =#
    copyto!(integrator.u, u0)
    try
      DiffEqBase.initialize_dae!(integrator, tableClusterInitAlg())
    catch err2
      OMBackend._fallback(err2, :eventInitializeDaeNewton; impact = :result)
      copyto!(integrator.u, u0)
      _restoreRetcode!(integrator, InitialFailure)
    end
  end
  (integrator.sol.retcode == InitialFailure && before != InitialFailure) || return nothing
  local residual = _maxAlgebraicResidual(integrator, integrator.u)
  ((residual <= _solveAbstol(integrator) || _atRoundoffFloor(integrator, integrator.u) ||
    _withinSolveTolerance(integrator, integrator.u)) &&
   residual <= _maxAlgebraicResidual(integrator, u0)) || return nothing
  @debug "[events] algebraic re-solve kept within the solve's tolerance" t = integrator.t residual
  return _restoreRetcode!(integrator, before)
end

"""
Whether every algebraic row's residual at u is within the solve's abstol
plus its round-off floor, 1e3 eps times the size of its terms
(sum_j |J_ij u_j|, from the problem's Jacobian; false without one). A
switched ideal element's rows carry terms of 1e11 (the MSL QS
IMC_Transformer's commuting switch: s = -9.1e5 against Goff = 1e-5), and
their residual stays near 1e-5 whatever the solver does.
"""
function _atRoundoffFloor(integrator, u)
  local f = integrator.f
  (hasproperty(f, :jac) && f.jac !== nothing && hasproperty(f, :jac_prototype) && f.jac_prototype !== nothing) ||
    return false
  local rows = _algebraicRows(f)
  (isempty(rows) || !ModelingToolkit.SciMLBase.isinplace(f)) && return false
  local J = similar(f.jac_prototype, eltype(u))
  local r = similar(u)
  try
    f.jac(J, u, integrator.p, integrator.t)
    f(r, u, integrator.p, integrator.t)
  catch e
    OMBackend._fallback(e, :roundoffFloorJacobian)
    return false
  end
  local scale = abs.(J) * abs.(u)
  local abstol = _solveAbstol(integrator)
  return all(k -> abs(r[k]) <= abstol + 1.0e3 * eps(Float64) * scale[k], rows)
end

"""
Whether one more Newton correction of the algebraic unknowns at u (the
differential ones kept) is within the solve's tolerance: in the solver's
error norm (algebraicStepControl.jl `_scaledNorm`, 1 is the tolerance), the
norm the integrator's own Newton iteration converges in. A Jacobian too
ill-conditioned for the re-solve's 1e-10 can leave the unknowns there all
the same: MSL DifferenceAmplifier at its ramp's end, the transistors'
der(vbc) at 1e9 (condition 7e26), the residual stalls at 1e-4 with a
correction of 0.08 (RMS, as the integrator's norm; 0.22 at most).
"""
function _withinSolveTolerance(integrator, u)
  local f = integrator.f
  local rows = _algebraicRows(f)
  (isempty(rows) || !ModelingToolkit.SciMLBase.isinplace(f)) && return false
  local n = length(rows)
  local w = AlgebraicWork(copy(u), similar(u), similar(u), similar(u, n), similar(u, n), similar(u, n, n),
                          nothing, nothing, Int[])
  local F = _algebraicJacobian!(w, rows, integrator, integrator.t)
  F === nothing && return false
  _algebraicResidual!(w, rows, integrator, integrator.t)
  LinearAlgebra.ldiv!(w.correction, F, w.g)
  return _scaledNorm(rows, integrator, w.correction, u) <= 1
end

"""
The solve's absolute tolerance, the smallest of a per-component one.
"""
_solveAbstol(integrator) = (local a = integrator.opts.abstol; a isa Number ? a : minimum(a))

"""
The largest residual of the algebraic equations of a mass-matrix ODE at
u (Inf for other problems).
"""
function _maxAlgebraicResidual(integrator, u)
  local f = integrator.f
  local rows = _algebraicRows(f)
  (isempty(rows) || !ModelingToolkit.SciMLBase.isinplace(f)) && return Inf
  local r = similar(u)
  f(r, u, integrator.p, integrator.t)
  return maximum(k -> abs(r[k]), rows)
end

"""
Solve the algebraic unknowns again with `alg` (nothing: EventReinit),
the differential states kept (a pure ODE has none). A DAEFunction (DFBDF,
IDA) has no mass matrix; its algebraic variables are the ones
`differential_vars` excludes. Whether it succeeded.
"""
function _resolveAlgebraics!(integrator, alg = nothing)
  local f = integrator.f
  if hasproperty(f, :mass_matrix)
    local mm = f.mass_matrix
    (mm isa LinearAlgebra.UniformScaling || all(!iszero, LinearAlgebra.diag(mm))) && return true
  end
  alg === nothing ? DiffEqBase.initialize_dae!(integrator, EventReinit()) : _eventReinit!(integrator, alg)
  _derivativeDiscontinuity!(integrator, true)
  return integrator.sol.retcode != ModelingToolkit.SciMLBase.ReturnCode.InitialFailure
end
