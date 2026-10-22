#= A when on a relation whose own crossing function located the event takes the change where the root
   finder left the event a hair before the shifted boundary (relationRefresh.jl, _whenRelationUpdate!). The
   hysteresis rule kept the relation there, nothing changed, and the root was found again at the next step: MSL
   Machines IMC_DCBraking with preferred states, a dynamic state set's switch at +2.3e-14 for 200 000 steps. =#
using Test
import OMBackend

@testset "A relation that located its event takes the change on the boundary" begin
  local CG = OMBackend.CodeGeneration
  #= a stub integrator: the when reads u[1] as its crossing function, scale 1 =#
  local integ = (f = (sys = nothing,), u = [0.0], p = Float64[], t = 1.0, opts = (reltol = 1e-6,),
                 sol = (prob = (tspan = (0.0, 1.0),),))
  local eps = CG._hysteresisFromTolerance(integ)
  local fresh() = CG.RelationWhenAffect(["x"], (t, x) -> (x, 1.0), true, (i, pre) -> nothing, nothing,
                                        Any[(u, p, t) -> u[1]], false, false, false)
  #= the root finder's point: the shifted function zc + eps at +1e-14, the rule sees it short of zc <= -eps =#
  local integAt(x) = merge(integ, (u = [x],))
  local short = -eps + 1e-14
  local a = fresh()
  @test !CG._whenRelationUpdate!(a, integAt(short))       # not located: the rule decides, it holds
  a.located = true
  @test CG._whenRelationUpdate!(a, integAt(short)) && a.rel # located: it changes
  @test !a.located
  #= located, but the function far from its boundary (another callback's event in the same step): the rule =#
  local b = fresh()
  b.located = true
  @test !CG._whenRelationUpdate!(b, integAt(0.5)) && !b.rel
  #= back: a true relation located at its shifted boundary zc - eps, just short of zc > eps =#
  local c = fresh()
  c.rel = true
  c.located = true
  @test CG._whenRelationUpdate!(c, integAt(eps - 1e-14)) && !c.rel
  @test CG._locatedOnBoundary(-eps + 1e-14, eps, false) && !CG._locatedOnBoundary(-eps + 1e-3, eps, false)
end
