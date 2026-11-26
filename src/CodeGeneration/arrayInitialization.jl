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

#= The initialization of the array models (ArrayPreserveDE): one method for every model, over its parameters `p` and the
   model's code they hold (`p.code`, ArrayModelCode): the initial system's solve, its redundant equations, the lifted
   clusters' passes, homotopy(). Generated into each model's module before, and compiled again with each
   (PID_Controller: 0.2 s of its first simulate). =#

"The loops' unknowns of the initial solve (`initLoopPlaces`): their guesses."
function loopGuesses(p::ArrayModelParameters)::Vector{Float64}
  return Float64[f > 0 ? p.loops[f][j, m] : p.blocks[-f].y[j] for (f::Int, j::Int, m::Int) in p.code.initLoopPlaces]
end

"The loops' guesses set to `x` (in the order of `initLoopPlaces`)."
function loopGuesses!(p::ArrayModelParameters, x::AbstractVector)::ArrayModelParameters
  for (k::Int, (f::Int, j::Int, m::Int)) in enumerate(p.code.initLoopPlaces)
    f > 0 ? (p.loops[f][j, m] = x[k]) : (p.blocks[-f].y[j] = x[k])
  end
  return p
end

"""
    initialize(u0, p, t, guesses = loopGuesses(p)) -> Vector

The initial state: the free states (and fixed = false parameters, into p.values) the initial equations determine,
solved for them with the unknowns of the loops the initial equations reach (`InitialSystem`, from `guesses`; the loops'
residuals rows of the solve), as OpenModelica's initial system; the loops solved again in the equations at the solution,
from it. MLS 8.6, as the MTK path: the relations at the guess, a solve with them, then at the solution again, until they
settle (a branch the solved state selects); relations that cycle (no consistent branch) keep the first solution. The
solve sorted into blocks as OpenModelica's initial system (arrayInitialBlocks.jl), each solved scaled in a trust region;
Newton where that fails. No solution: refused as the MTK path refuses it (OpenModelica fails there too).
"""
function initialize(u0::AbstractVector, p::ArrayModelParameters, t::Real,
                    guesses::Vector{Float64} = loopGuesses(p))::AbstractVector
  local c::ArrayModelCode = p.code
  local nS::Int = length(c.initUnknowns)
  local nR::Int = nS + length(c.initParams)
  local nAlg::Int = length(p.abuf)
  local u::AbstractVector = copy(u0)
  local du::AbstractVector = similar(u); local a::AbstractVector = similar(u, nAlg)
  local setUnknowns!::Function = function (uu::AbstractVector, z::AbstractVector)
    uu[c.initUnknowns] .= view(z, 1:nS)
    p.values[c.initParams] .= view(z, (nS + 1):nR)
    loopGuesses!(p, view(z, (nR + 1):length(z)))
    return uu
  end
  #= the relations keep their values during a solve =#
  local residual::Function = function (res::AbstractVector, z::AbstractVector, _)
    local uz::AbstractVector = similar(z, length(u0))
    uz .= u0
    setUnknowns!(uz, z)
    local duz::AbstractVector = similar(uz); local az::AbstractVector = similar(uz, nAlg)
    p.pass.x .= view(z, (nR + 1):length(z))
    p.pass.on = true
    try
      c.equations!(duz, az, uz, p, t, NO_CROSSINGS, false)
    finally
      p.pass.on = false
    end
    if isempty(c.initRedundant)
      c.initialResiduals!(view(res, 1:nR), duz, az, uz, p, t)
    else
      local every::AbstractVector = similar(res, c.nInit)
      c.initialResiduals!(every, duz, az, uz, p, t)
      view(res, 1:nR) .= view(every, c.initRows)
    end
    view(res, (nR + 1):length(res)) .= p.pass.r
    return nothing
  end
  local z::Vector{Float64} = vcat(u0[c.initUnknowns], p.values[c.initParams], guesses)
  local scale::Vector{Float64} = map(x::Float64 -> x == 0 ? 1.0 : abs(x), z)
  local whole::Function = (res::AbstractVector, w::AbstractVector) -> residual(res, w, nothing)
  local first::Union{Nothing, Vector{Float64}} = nothing
  local seen::Vector{Tuple{Vector{Bool}, Vector{Float64}}} = Tuple{Vector{Bool}, Vector{Float64}}[]
  for _ in 1:20
    c.equations!(du, a, setUnknowns!(u, z), p, t, NO_CROSSINGS, true)
    local rel0::Vector{Bool} = copy(p.rel); local e0::Vector{Float64} = copy(p.evv)
    if (rel0, e0) in seen
      c.equations!(du, a, setUnknowns!(u, first), p, t, NO_CROSSINGS, true)
      return u
    end
    push!(seen, (rel0, e0))
    local (ok::Bool, zs::Vector{Float64}, _) = sortedInitialSolve(whole, z, scale, c.initReads)
    if ok
      z = zs
      first === nothing && (first = copy(z))
      c.equations!(du, a, setUnknowns!(u, z), p, t, NO_CROSSINGS, true)
      (p.rel == rel0 && p.evv == e0) && return u
      continue
    end
    local sol = newtonInitialSolve(residual, z)
    DiffEqBase.SciMLBase.successful_retcode(sol) ||
      _OMBACKEND.unsupported("fixed start values or initial equations that the initialization cannot hold",
                             string("the initial system (", sol.retcode, ")"))
    z = copy(sol.u)
    first === nothing && (first = copy(z))
    c.equations!(du, a, setUnknowns!(u, z), p, t, NO_CROSSINGS, true)
    (p.rel == rel0 && p.evv == e0) && return u
  end
  c.equations!(du, a, setUnknowns!(u, first), p, t, NO_CROSSINGS, true)
  return u
end

"""
    checkRedundantInitial(u, p, t)

The initial equations beyond the free states (`initRedundant`) at the initial state: each satisfied by what the others
gave, as OpenModelica keeps a model whose redundant initial equations are consistent; an error for one that is not.
"""
function checkRedundantInitial(u::AbstractVector, p::ArrayModelParameters, t::Real)::Nothing
  local c::ArrayModelCode = p.code
  local du::AbstractVector = similar(u); local a::AbstractVector = similar(u, length(p.abuf))
  c.equations!(du, a, u, p, t, NO_CROSSINGS, false)
  local res::Vector{Float64} = zeros(c.nInit)
  c.initialResiduals!(res, du, a, u, p, t)
  for (k::Int, r::Int) in enumerate(c.initRedundant)
    abs(res[r]) <= 1.0e-6 ||
      _OMBACKEND.unsupported("fixed start values or initial equations that the initialization cannot hold",
                             string("the initialization is over-determined: ", c.initRedundantTexts[k],
                                    " (residual ", res[r], ")"))
  end
  return nothing
end

"""
    initialClusters!(uStart, u, p, t, preStart, used, guessStart) -> Vector

The start's mixed system (MLS 8.6: pre() the start values), as OpenModelica and the MTK path
(`_settleInitialDiscretes!`): the initial system was solved with the lifted clusters' variables (`clusterPre`) at
`used` (their start values); their initial arms (`initialAlgorithms!`) gave their values at that solution. Where those
differ, the initial system again from the start values `uStart` with them, pre() of them the start values `preStart`,
until they stay (a cycle or 10 passes: the last solution stands, the start's event iteration settles them). MSL
RobotR3's bearing friction: solved unlocked (`a = sa + tau0_max`), the steady start held the motor at the friction's
limit (La.i = -0.43, omc 0); locked, sa keeps its start in the initial system (`InitialSystem`).
"""
function initialClusters!(uStart::AbstractVector, u::AbstractVector, p::ArrayModelParameters, t::Real,
                          preStart::Vector{Float64}, used::Vector{Float64}, guessStart::Vector{Float64})::AbstractVector
  local c::ArrayModelCode = p.code
  length(c.initRows) > 0 || return u
  local seen::Vector{Vector{Float64}} = Vector{Float64}[used]
  for _ in 1:10
    local now::Vector{Float64} = p.d[c.clusterPre]
    now in seen && return u
    push!(seen, now)
    p.dpre[c.clusterPre] .= preStart
    u = initialize(copy(uStart), p, t, copy(guessStart))
    isempty(c.initRedundant) || checkRedundantInitial(u, p, t)
    c.initialAlgorithms!(u, p, t)
  end
  return u
end

"""
    homotopyStart!(u0, p, t) -> Vector

homotopy() (MLS 3.7.4.4), as OpenModelica's default and the scalarized route (`_homotopyContinuation`): the start
solved at λ = 0 (the simplified expressions), then from each solution at λ = 1/3 and 2/3, the relations and the loops'
guesses at each step's solution; the initialization at λ = 1 follows from the last. MSL's IdealizedOpAmpLimited
(SignalGenerator): from the start values the limiter's loop went to the unstable linear solution, all values 0
(OpenModelica: saturated). A step that fails: the initialization from the start values.
"""
function homotopyStart!(u0::AbstractVector, p::ArrayModelParameters, t::Real)::AbstractVector
  local u::AbstractVector = copy(u0)
  local before::Tuple{Vector{Float64}, Vector{Float64}, Vector{Bool}, Vector{Float64}} = _discreteState(p)
  local wcond::Vector{Bool} = copy(p.wcond)
  local guesses::Vector{Matrix{Float64}} = deepcopy(p.loops)
  try
    for λ::Float64 in (0.0, 1 / 3, 2 / 3)
      p.homotopy[1] = λ
      length(p.code.initRows) > 0 && (u = initialize(u, p, t))
      startSolution!(u, p, t, false)
    end
  catch e
    _OMBACKEND._fallback(e, :arrayHomotopyContinuation, impact = :result)
    u = copy(u0)
    p.d .= before[1]; p.dpre .= before[2]; p.rel .= before[3]; p.evv .= before[4]
    p.wcond .= wcond
    for (g::Matrix{Float64}, g0::Matrix{Float64}) in zip(p.loops, guesses)
      copyto!(g, g0)
    end
  finally
    p.homotopy[1] = 1.0
  end
  return u
end

"The names the model's module binds to the functions of this file."
const _INITIALIZATION_FUNCTIONS = (:loopGuesses, :loopGuesses!, :initialize, :checkRedundantInitial, :initialClusters!,
                                   :homotopyStart!)

"""
The model's module's bindings of the functions common to every array model (arrayEvents.jl, arrayAsserts.jl, this
file): its own code and the callbacks it builds call them by these names.
"""
_driverBindings()::Vector{Expr} =
  Expr[:(const $n = $(getfield(@__MODULE__, n)))
       for n in (_EVENT_FUNCTIONS..., _ASSERT_FUNCTIONS..., _INITIALIZATION_FUNCTIONS...)]

"""
    _enterMode!(u0, p, starts) -> Nothing

The start of a mode entered at a structural change (`simulateModes`), in the place of its initialization: the states
and discrete variables named in `starts` (the values at the change, by the mode's names) take them, the states' pre()
and reinit buffers and the discrete variables' pre() with them; the others keep their start values. The run starts
after the change's event (`ArrayRuntimeState.entered`).
"""
function _enterMode!(u0::AbstractVector{Float64}, p::ArrayModelParameters, starts::AbstractDict{String, Float64})::Nothing
  local stateIndex::Dict{String, Int} = p.model.STATE_INDEX
  local discreteIndex::Dict{String, Int} = p.model.DISCRETE_INDEX
  for (name::String, value::Float64) in starts
    local i::Int = get(stateIndex, name, 0)
    if i > 0
      u0[i] = value
      continue
    end
    local k::Int = get(discreteIndex, name, 0)
    k > 0 && (p.d[k] = value; p.dpre[k] = value)
  end
  p.upre .= u0; p.ureinit .= u0
  p.state.entered = true
  return nothing
end
