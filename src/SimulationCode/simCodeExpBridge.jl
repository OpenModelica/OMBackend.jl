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

#= `Util.traverseExpTopDown(::DAE.Exp, func, ext_arg)` is the canonical
   recursive descent over a `DAE.Exp` tree used by alias substitution,
   constant folding, cref collection, etc. When the caller passes a
   SimCode-native `Exp`, route through `toDAEExp` and convert the
   returned expression back to `Exp` so the call site sees the same
   in/out type. =#
Base.@nospecializeinfer function Util.traverseExpTopDown(@nospecialize(inExp::Exp), func::Function, ext_arg)
  local (outDAE, outArg) = Util.traverseExpTopDown(toDAEExp(inExp), func, ext_arg)
  return (toSimExp(outDAE), outArg)
end
