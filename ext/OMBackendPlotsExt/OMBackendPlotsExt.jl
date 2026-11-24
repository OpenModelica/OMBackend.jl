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
  return OMBackend.plot(sol.diffEqSol; legend, limX, limY)
end

"""
  Plots a vector of solutions, all on one new plot.
"""
function OMBackend.plot(sols::Vector; legend = false, limX = 0.0, limY = 1.0, kwargs...)
  local p = Plots.plot(first(sols); legend, xlim = limX, ylim = limY, kwargs...)
  for s in @view sols[2:end]
    Plots.plot!(p, s; legend, xlim = limX, ylim = limY, kwargs...)
  end
  return p
end

end # module OMBackendPlotsExt
