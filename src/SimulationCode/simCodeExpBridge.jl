#= Phase 4 SimCode-Exp infrastructure (additive).

   The SimCode-native `Exp` hierarchy is defined in `simCodeData.jl`.
   This file holds the only public boundary helpers:

   - `Base.convert(::Type{Exp}, ::DAE.Exp)` so the auto-generated outer
     constructors of any equation struct whose `exp` field is still
     typed `::DAE.Exp` keep accepting bare DAE.Exp values from BDAE
     producers without explicit cast.

   - `Base.convert(::Type{DAE.Exp}, ::Exp)` so a SimCode-native `Exp`
     can be passed to legacy DAE.Exp consumers via explicit
     `convert(DAE.Exp, e)` (Julia does NOT auto-trip convert at
     function dispatch boundaries, only at struct-field assignment
     and explicit convert calls — that is the reason every codegen
     consumer that wants to keep working with SIM Exp gets a native
     `::SimulationCode.Exp` overload instead of relying on convert).

   The previous attempt at a per-helper SIM-Exp delegating overload
   per `::DAE.Exp` consumer (via `toDAEExp`) was abandoned: round-
   tripping every expression through `DAE.Exp` re-runs frontend-grade
   walks and slowed `OM.translate` to a crawl during precompile.
   The correct path is native SIM-Exp emit functions
   (`expToJulia*(::SimulationCode.Exp, …)` overloads with bodies that
   walk SIM Exp directly). Those overloads live in their respective
   codegen files; see `codeGen.jl`, `MTK_CodeGenerationUtil.jl`,
   `algorithmic.jl`, `DECodeGeneration.jl`. =#

Base.@nospecializeinfer function Base.convert(::Type{DAE.Exp}, @nospecialize(e::Exp))
  return toDAEExp(e)
end

# Util.* are DAE-only; SIM consumers pass SIM.Exp post-migration.
import ..FrontendUtil.Util
Util.getAllCrefs(e::Exp) = Util.getAllCrefs(toDAEExp(e))
Util.traverseExpBottomUp(e::Exp, visitor, ctx) = Util.traverseExpBottomUp(toDAEExp(e), visitor, ctx)
Util.traverseExpTopDown(e::Exp, visitor, ctx) = Util.traverseExpTopDown(toDAEExp(e), visitor, ctx)

Base.@nospecializeinfer function Base.convert(::Type{Exp}, @nospecialize(e::Exp))
  return e
end
# Intentionally not defining `convert(::Type{Exp}, ::DAE.Exp) = toSimExp(e)`.
# That convert would fire on any `::Exp`-typed slot (function param,
# struct field, Vector{Exp} push!) and silently coerce DAE.Exp into a
# SimCode.Exp — which surfaces as "unsupported DAE.Exp variant" warnings
# whenever a downstream check expects a DAE.* tag. Re-add ONLY when an
# equation field actually carries `::Exp`, never as a general bridge.
