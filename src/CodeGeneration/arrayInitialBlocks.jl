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

#=
The initial system of an array model solved in its blocks, as OpenModelica sorts its initial system (BLT): each initial
equation matched to an unknown it reads, the strongly connected blocks of that matching solved one after another, each
with the unknowns of the blocks before it known. One system of all unknowns from their start values can read values
that are not defined there: MSL FluidHeatFlow TestOpenTank's T = T0 reads h = H / m, NaN at the starts m = H = 0, while
the level's equation alone gives m first. The incidence comes from finite differences at generic points; a system that
is not structurally regular, or that is one block, is solved whole.
=#

"""
    scaledTrustRegionSolve(residual!, z0, scale) -> (ok, z, retcode)

The solution of `residual!(res, z) = 0` from `z0`, as OpenModelica scales its nonlinear systems: each unknown by
`scale` (its start's magnitude, 1 for a zero start), so the finite differences step relative to it (MSL
LightningImpulse: T10 ~ 4e-7 and tau ~ 1e-5 beside eta ~ 1), each residual by its largest Jacobian entry at the guess;
solved in a trust region (OpenModelica's hybrid dogleg): undamped Newton ran to LightningImpulse's degenerate root
(tau1 = tau2, eta = 0).
"""
function scaledTrustRegionSolve(residual!::Function, z0::Vector{Float64},
                                scale::Vector{Float64})::Tuple{Bool, Vector{Float64}, Any}
  local r0::Vector{Float64} = similar(z0)
  local rp::Vector{Float64} = similar(z0)
  local rows::Vector{Float64} = zeros(length(z0))
  residual!(r0, z0)
  for k::Int in eachindex(z0)
    local zp::Vector{Float64} = copy(z0)
    local h::Float64 = 1.0e-7 * max(abs(z0[k]) / scale[k], 1.0)
    zp[k] += h * scale[k]
    residual!(rp, zp)
    rows .= max.(rows, abs.(rp .- r0) ./ h)
  end
  rows .= map(r -> (r > 0 && isfinite(r)) ? 1 / r : 1.0, rows)
  local scaledResidual::Function = function (res, w, _)
    residual!(res, w .* scale)
    res .*= rows
    return nothing
  end
  local prob = NonlinearSolve.NonlinearProblem(NonlinearSolve.NonlinearFunction(scaledResidual), z0 ./ scale)
  local sol = NonlinearSolve.solve(prob, NonlinearSolve.TrustRegion(autodiff = ADTypes.AutoFiniteDiff());
                                   abstol = 1e-10, reltol = 1e-10)
  return (NonlinearSolve.SciMLBase.successful_retcode(sol), sol.u .* scale, sol.retcode)
end

"""
    sortedInitialSolve(residual!, z0, scale, reads) -> (ok, z, retcode)

The initial system `residual!(res, z) = 0` solved in its blocks (`initialBlocks`; `reads`: each row's unknowns from the
structure, `nothing`: from finite differences), each by `scaledTrustRegionSolve` with the unknowns of the blocks
before it known; solved whole from `z0` where it has one block, is not structurally regular, or a block has no
solution.
"""
function sortedInitialSolve(residual!::Function, z0::Vector{Float64}, scale::Vector{Float64},
                            reads::Union{Nothing, Vector{Vector{Int}}})::Tuple{Bool, Vector{Float64}, Any}
  local blocks::Union{Nothing, Vector{Tuple{Vector{Int}, Vector{Int}}}} = initialBlocks(residual!, z0, scale, reads)
  if blocks !== nothing
    local z::Vector{Float64} = copy(z0)
    local full::Vector{Float64} = similar(z0)
    local solved::Bool = true
    for (rows::Vector{Int}, cols::Vector{Int}) in blocks
      local blockResidual::Function = function (res, w)
        local zz::Vector{Float64} = copy(z)
        zz[cols] .= w
        residual!(full, zz)
        res .= view(full, rows)
        return nothing
      end
      local (ok::Bool, w::Vector{Float64}, _) = scaledTrustRegionSolve(blockResidual, z[cols], scale[cols])
      ok || (solved = false; break)
      z[cols] .= w
    end
    solved && return (true, z, NonlinearSolve.SciMLBase.ReturnCode.Success)
  end
  return scaledTrustRegionSolve(residual!, z0, scale)
end

"""
    initialBlocks(residual!, z, scale, structure) -> Union{Nothing, Vector{Tuple{Vector{Int}, Vector{Int}}}}

The blocks of the initial system `residual!(res, z) = 0` in the order they are solved: (residual rows, unknowns), from
a perfect matching of the rows to the unknowns they read (`structure`, the code generation's `InitialSystem`: every
branch of an if-expression read, as OpenModelica's matching; where it has none `_initialIncidence`) and the strongly
connected components of the rows (Tarjan: a row's block after the blocks of the unknowns it reads). `nothing` for a
system without a perfect matching, without an incidence, or of one block.
"""
function initialBlocks(residual!::Function, z::Vector{Float64}, scale::Vector{Float64},
                       structure::Union{Nothing, Vector{Vector{Int}}}
                       )::Union{Nothing, Vector{Tuple{Vector{Int}, Vector{Int}}}}
  local n::Int = length(z)
  n > 1 || return nothing
  local reads::Union{Nothing, Vector{Vector{Int}}} =
    structure === nothing ? _initialIncidence(residual!, z, scale) : structure
  reads === nothing && return nothing
  local colOfRow::Vector{Int} = zeros(Int, n)
  local rowOfCol::Vector{Int} = zeros(Int, n)
  for i::Int in 1:n
    _augment!(i, reads, colOfRow, rowOfCol, falses(n)) || return nothing
  end
  local after::Vector{Vector{Int}} = [Int[rowOfCol[c] for c in reads[i] if c != colOfRow[i]] for i in 1:n]
  local components::Vector{Vector{Int}} = _stronglyConnected(after)
  length(components) > 1 || return nothing
  return Tuple{Vector{Int}, Vector{Int}}[(rows, colOfRow[rows]) for rows in components]
end

"""
    _initialIncidence(residual!, z, scale) -> Union{Nothing, Vector{Vector{Int}}}

Per residual row, the unknowns it reads: the rows that change when one unknown moves by 1e-4 of its scale, at `z` and
at two generic points beside it (a value that is not defined at the start values, a reading that cancels at one point).
A row counts at the points where it is finite; `nothing` where a row is finite at none of them.
"""
function _initialIncidence(residual!::Function, z::Vector{Float64},
                           scale::Vector{Float64})::Union{Nothing, Vector{Vector{Int}}}
  local n::Int = length(z)
  local depends::BitMatrix = falses(n, n)
  local seen::BitVector = falses(n)
  local r::Vector{Float64} = zeros(n)
  local rp::Vector{Float64} = zeros(n)
  for j::Int in 0:2
    local g::Vector{Float64} =
      j == 0 ? copy(z) : [z[k] + scale[k] * (0.1 + 0.07 * mod(k * (2j + 5), 11) / 11) for k in 1:n]
    try
      residual!(r, g)
      local finite::BitVector = isfinite.(r)
      any(finite) || continue
      for k::Int in 1:n
        local gp::Vector{Float64} = copy(g)
        gp[k] += 1.0e-4 * scale[k]
        residual!(rp, gp)
        for i::Int in 1:n
          (finite[i] && !(rp[i] == r[i])) && (depends[i, k] = true)
        end
      end
      seen .|= finite
    catch err
      err isa InterruptException && rethrow()
      continue
    end
  end
  all(seen) || return nothing
  return Vector{Int}[findall(view(depends, i, :)) for i in 1:n]
end

"An augmenting path from row `i` (Kuhn's matching): `true` where `i` got an unknown."
function _augment!(i::Int, reads::Vector{Vector{Int}}, colOfRow::Vector{Int}, rowOfCol::Vector{Int},
                   visited::BitVector)::Bool
  for c::Int in reads[i]
    visited[c] && continue
    visited[c] = true
    if rowOfCol[c] == 0 || _augment!(rowOfCol[c], reads, colOfRow, rowOfCol, visited)
      colOfRow[i] = c
      rowOfCol[c] = i
      return true
    end
  end
  return false
end

"""
    _stronglyConnected(after) -> Vector{Vector{Int}}

The strongly connected components of the graph `i -> after[i]` (Tarjan), each after the components it reaches: the
order the blocks are solved in.
"""
function _stronglyConnected(after::Vector{Vector{Int}})::Vector{Vector{Int}}
  local n::Int = length(after)
  local index::Vector{Int} = zeros(Int, n)
  local low::Vector{Int} = zeros(Int, n)
  local onStack::BitVector = falses(n)
  local stack::Vector{Int} = Int[]
  local components::Vector{Vector{Int}} = Vector{Int}[]
  #= Base's: this module's Ref is SimCode's reference =#
  local counter::Base.RefValue{Int} = Base.Ref(0)
  for v::Int in 1:n
    index[v] == 0 && _tarjan!(v, after, index, low, onStack, stack, components, counter)
  end
  return components
end

"One visit of Tarjan's algorithm from `v`."
function _tarjan!(v::Int, after::Vector{Vector{Int}}, index::Vector{Int}, low::Vector{Int}, onStack::BitVector,
                  stack::Vector{Int}, components::Vector{Vector{Int}}, counter::Base.RefValue{Int})::Nothing
  counter[] += 1
  index[v] = counter[]
  low[v] = counter[]
  push!(stack, v)
  onStack[v] = true
  for w::Int in after[v]
    if index[w] == 0
      _tarjan!(w, after, index, low, onStack, stack, components, counter)
      low[v] = min(low[v], low[w])
    elseif onStack[w]
      low[v] = min(low[v], index[w])
    end
  end
  if low[v] == index[v]
    local component::Vector{Int} = Int[]
    while true
      local w::Int = pop!(stack)
      onStack[w] = false
      push!(component, w)
      w == v && break
    end
    push!(components, sort!(component))
  end
  return nothing
end
