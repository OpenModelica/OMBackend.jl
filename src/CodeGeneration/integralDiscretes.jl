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

#= The held discretes: unknowns with der(v) = 0, written by the affects only.

   Integer, Boolean and enumeration variables hold integral values. A
   Rosenbrock or BDF step can leave round-off in them (pivoting in the linear
   solve mixes rows): 3 becomes 3.0000000000000004, and change(v) or v == 3
   would see it.

   An event inside a step interpolates every state to its time, and the
   interpolant of a constant is off by an ulp at some times: every continuous
   callback of the problem moves the integrator there with the held discretes
   put back to their values at the step's start (`HeldBack`). =#

"""
    HeldBack(alg)

The re-initialization of a continuous callback whose own is `alg` (nothing:
the integrator's), with the held discretes put back (`_heldBack!`) where the
callback moves the integrator to its event's time, before anything reads the
state there: the re-initialization `alg` of that move, the save, the affects.

OrdinaryDiffEqCore hands a callback's re-initialization to the move
(`change_t_via_interpolation!`), which runs it after the interpolation for a
DAE (a singular mass matrix) and not at all for an ODE, and again after the
affect. Hence two hooks: the method of the move marks the record as moving and
puts the discretes back after it if no re-initialization did; the
re-initialization puts them back first while the record is moving. After an
affect the record is not moving: what the affect wrote stays. A callback
carries it whatever its affect is (a function, or the compiled equations of
MTK's own events, which call nothing of OMBackend).
"""
struct HeldBack{A} <: ModelingToolkit.SciMLBase.DAEInitializationAlgorithm
  alg::A
end

_mergeableReinit(h::HeldBack)::Bool = _mergeableReinit(h.alg)

function ModelingToolkit.SciMLBase.change_t_via_interpolation!(integrator::OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator, t,
                                                               modifySaveEndpoint::Type{Val{T}}, h::HeldBack) where {T}
  local general = Tuple{OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator, Any, Type{Val{T}}, Any}
  local record::Union{Nothing, _InstantPre} = _heldRecord(integrator)
  if record === nothing || t == integrator.t
    invoke(ModelingToolkit.SciMLBase.change_t_via_interpolation!, general, integrator, t, modifySaveEndpoint, h)
    return nothing
  end
  record.moving = true
  try
    invoke(ModelingToolkit.SciMLBase.change_t_via_interpolation!, general, integrator, t, modifySaveEndpoint, h)
    #= Still moving: no re-initialization ran (an ODE). =#
    record.moving && _heldBack!(record, integrator)
  finally
    record.moving = false
  end
  return nothing
end

"""
SciMLBase's fallback takes any algorithm for any integrator,
OrdinaryDiffEqCore's any for its own (as for EventReinit).
"""
DiffEqBase.initialize_dae!(integrator::ModelingToolkit.SciMLBase.DEIntegrator, h::HeldBack) = _heldBackReinit!(integrator, h)
DiffEqBase.initialize_dae!(integrator::OrdinaryDiffEq.OrdinaryDiffEqCore.ODEIntegrator, h::HeldBack) =
  _heldBackReinit!(integrator, h)

function _heldBackReinit!(integrator, h::HeldBack)
  local record::Union{Nothing, _InstantPre} = _heldRecord(integrator)
  if record !== nothing && record.moving
    record.moving = false
    _heldBack!(record, integrator)
  end
  return DiffEqBase.initialize_dae!(integrator, h.alg === nothing ? integrator.initializealg : h.alg)
end

"""
    holdingBack(callbacks) -> CallbackSet

`callbacks` with the re-initialization of every continuous callback in a
`HeldBack`: the system's own events and the generated ones alike
(`withProblemCallbacks`).
"""
function holdingBack(set::DiffEqBase.CallbackSet)::DiffEqBase.CallbackSet
  local continuous::Vector{Any} = Any[_holdingBack(cb) for cb in set.continuous_callbacks]
  return DiffEqBase.CallbackSet(continuous, Any[set.discrete_callbacks...])
end
_holdingBack(@nospecialize(cb::ModelingToolkit.SciMLBase.AbstractContinuousCallback)) =
  cb.initializealg isa HeldBack ? cb :
  Setfield.set(cb, Setfield.PropertyLens{:initializealg}(), HeldBack(cb.initializealg))

"""
    withHeldDiscretes(callbacks, problem, names, integralNames) -> callbacks

Add a DiscreteCallback ahead of the other discrete callbacks (before the event
iteration reads them). At the start of a solve it registers the unknowns
`names` as the held discretes of the integrator (`holdDiscretes!`); after a
step that left round-off in the unknowns `integralNames` it rounds them back
to integers.
"""
function withHeldDiscretes(callbacks, problem, names::Vector{String}, integralNames::Vector{String})
  local states::Vector{Symbol} = getStatesAsSymbols(problem.f)
  local held::Vector{Int} = Int[k for k in indexin(Symbol.(names), states) if k !== nothing]
  isempty(held) && return callbacks
  local idx::Vector{Int} = Int[k for k in indexin(Symbol.(integralNames), states) if k !== nothing]
  local drifted = (u, t, integrator) -> any(k -> u[k] != round(u[k]), idx)
  local round! = function (integrator)
    for k::Int in idx
      integrator.u[k] = round(integrator.u[k])
    end
    return nothing
  end
  local register! = (c, u, t, integrator) -> holdDiscretes!(integrator, held)
  local cb = DiffEqBase.DiscreteCallback(drifted, round!; initialize = register!, save_positions = (false, false),
                                         initializealg = ModelingToolkit.SciMLBase.NoInit())
  return _joinCallbacks(cb, callbacks)
end
