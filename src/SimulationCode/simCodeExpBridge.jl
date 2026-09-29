#= Boundary helpers between SimCode's own `Exp` (simCodeData.jl) and
   DAE.Exp; the SimCode passes are part way through moving to `Exp`.

   - `convert(DAE.Exp, ::Exp)` (toDAEExp), for DAE.Exp consumers that
     convert explicitly. Julia converts only on field assignment and
     explicit calls, so each codegen consumer of `Exp` has its own
     `::SimulationCode.Exp` method instead (codeGen.jl,
     MTK_CodeGenerationUtil.jl, algorithmic.jl, DECodeGeneration.jl).
   - Util's DAE traversals on `Exp`, through toDAEExp.
   - No `convert(Exp, ::DAE.Exp)`: see below.

   Delegating every `Exp` consumer to its DAE.Exp twin through toDAEExp was
   tried and abandoned: the round trips slowed OM.translate to a crawl
   during precompile. =#

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
