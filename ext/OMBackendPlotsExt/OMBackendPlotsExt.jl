#=
  The plot methods of OMBackend, with Plots.jl. Julia loads this extension when Plots is loaded.
=#
module OMBackendPlotsExt

#= Plots first: the process that precompiles this extension loads the packages in the order of these imports, and on
   Windows Plots (Glib) must be loaded before OMBackend's OMRuntimeExternalC (OMBackend.jl#101). =#
import Plots
import OMBackend
import OMBackend.Runtime

"""
`plot(sol::Runtime.OMSolution)`
  The default plot function of OMBackend.
  All labels of the variables and the name is given by default
"""
function OMBackend.plot(sol::Runtime.OMSolution)
  local nsolution = sol.diffEqSol
  local t = nsolution.t
  local rescols = collect(eachcol(transpose(hcat(nsolution.u...))))
  labels = permutedims(sol.idxToName.vals)
  Plots.plot(t, rescols; labels=labels)
end

"""
`function plot(sol)`
  An alternative plot function in OMBackend.
  All labels of the variables and the name is given by default
"""
function OMBackend.plot(sol)
  Plots.plot(sol)
end

"""
  Plot for an OMSolution that contains several sub solutions.
  Plots all part of the solution on the same graph.
"""
function OMBackend.plot(sol::Runtime.OMSolutions; legend = false, limX = 0.0, limY = 1.0)
  local sols = sol.diffEqSol
  local prevP = Plots.plot!(sols[1]; legend = legend, xlim=limX, ylim = limY)
  for sol in sols[2:end]
    p = Plots.plot!(prevP; legend = legend, xlim=limX, ylim = limY)
    prevP = p
  end
  return prevP
end

"""
  Plots a vector of solutions
"""
function OMBackend.plot(sol::Vector; legend = false, limX = 0.0, limY = 1.0, kwargs...)
  sols = sol
  local prevP = Plots.plot!(sols[1]; legend = legend, xlim=limX, ylim = limY, kwargs...)
  for sol in sols[2:end]
    p = Plots.plot!(prevP; legend = legend, xlim=limX, ylim = limY, kwargs...)
    prevP = p
  end
  return prevP
end

end # module OMBackendPlotsExt
