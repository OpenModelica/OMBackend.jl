#=
This file contains slightly modified code from MTK.
This is used here according to the MIT license.
Details below.
=#

#= The ModelingToolkit.jl package is licensed under the MIT "Expat" License:
# Copyright (c) 2018-25: Christopher Rackauckas, Julia Computing.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  SOFTWARE.
=#


#=
* This file is part of OpenModelica.
*
* Copyright (c) 2021-CurrentYear, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
=#


#=
This file contains "hacks".
This is done in order to get the equations on a MTK compatible format before calling functions such as structurally simplify.
TODO:
!Adjust the uncessary string conversions!
=#


#= Global dictionary to store dynamically generated function implementations =#
#= The key is the function name, the value is the implementation function =#
const MODELICA_FUNCTION_IMPLS = Dict{Symbol, Function}()

#= Global dictionary to store RTG wrappers for each function =#
const MODELICA_FUNCTION_WRAPPERS = Dict{Symbol, Any}()

#= Cache for per-element extractor functions.
   Key: (funcName::Symbol, indices::Tuple{Vararg{Int}})
   Value: the created function object =#
const ELEM_FUNC_CACHE = Dict{Tuple{Symbol, Tuple}, Any}()

"""
Unwrap a value for use in symbolic Terms.
For arrays, unwraps each element. For scalars, unwraps directly.
Non-symbolic values (plain Float64, Int, etc.) pass through unchanged.
"""
function unwrapForSymbolic(x)
  if x isa AbstractArray
    return map(Symbolics.unwrap, x)
  else
    return Symbolics.unwrap(x)
  end
end

"""
Get or create a per-element extractor function for an array-returning Modelica function.
The extractor calls the implementation directly via MODELICA_FUNCTION_IMPLS and extracts
a single element by index. This avoids creating getindex(Term{Real}(...), i) symbolic
terms which crash Symbolics._linear_expansion (OffsetArrays.Origin error).
"""
function getOrCreateElemFunc(funcName::Symbol, indices::Tuple{Vararg{Int}}, nArgs::Int)
  local key = (funcName, indices)
  if haskey(ELEM_FUNC_CACHE, key)
    return ELEM_FUNC_CACHE[key]
  end
  local fnQuote = QuoteNode(funcName)
  local argNames = [Symbol("a", k) for k in 1:nArgs]
  local implCall = Expr(:call, :(Base.invokelatest), :impl, argNames...)
  local body
  if length(indices) == 1
    local idx = indices[1]
    body = Expr(:->, Expr(:tuple, argNames...), Expr(:block,
      :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]),
      :(result = $implCall),
      :(return result[$idx])
    ))
  elseif length(indices) == 2
    local i = indices[1]
    local j = indices[2]
    body = Expr(:->, Expr(:tuple, argNames...), Expr(:block,
      :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]),
      :(result = $implCall),
      :(row = result[$i]),
      Expr(:if, :(row isa AbstractVector),
        :(return row[$j]),
        :(return result[$i, $j])
      )
    ))
  end
  local f = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  ELEM_FUNC_CACHE[key] = f
  return f
end

"""
Create a symbolic array representation for an array-returning function call.
Returns a Vector{Num} for 1D outputs or Matrix{Num} for 2D outputs.
Each element is a scalar Term{Real} calling a per-element extractor function,
avoiding the getindex operation that triggers OffsetArrays.Origin errors in
Symbolics._linear_expansion.
"""
function createSymbolicArrayCall(funcRef, uwArgs::Vector{Any}, dims::Tuple{Vararg{Int}}; funcName::Symbol = Symbol())
  #= If funcName not provided, try to extract it from funcRef =#
  if funcName === Symbol()
    funcName = funcRef isa Symbol ? funcRef : nameof(funcRef)
  end
  local nArgs = length(uwArgs)
  if length(dims) == 1
    return [Symbolics.Num(SymbolicUtils.Term{Real}(getOrCreateElemFunc(funcName, (i,), nArgs), uwArgs)) for i in 1:dims[1]]
  elseif length(dims) == 2
    return [Symbolics.Num(SymbolicUtils.Term{Real}(getOrCreateElemFunc(funcName, (i, j), nArgs), uwArgs)) for i in 1:dims[1], j in 1:dims[2]]
  else
    #= 3D+ fallback: use linear indexing =#
    local totalLen = prod(dims)
    return [Symbolics.Num(SymbolicUtils.Term{Real}(getOrCreateElemFunc(funcName, (i,), nArgs), uwArgs)) for i in 1:totalLen]
  end
end

"""
Helper to check if a value is symbolic (Symbolics.Num or contains symbolic expressions).
Also recursively checks arrays, since record field arrays assembled from scalarized
parameters may be Vector{Symbolics.Num} or Vector{Vector{Symbolics.Num}}.
"""
isSymbolicArg(::Float64) = false
isSymbolicArg(::Int64) = false
isSymbolicArg(::Bool) = false
isSymbolicArg(::AbstractArray{Float64}) = false
isSymbolicArg(::AbstractArray{Int64}) = false
function isSymbolicArg(x)
  x isa Symbolics.Num && return true
  x isa Symbolics.Arr && return true
  x isa SymbolicUtils.BasicSymbolic && return true
  if x isa AbstractArray
    for el in x
      isSymbolicArg(el) && return true
    end
  end
  return false
end

"""
Helper to check if any argument in a tuple/collection is symbolic.
"""
function hasSymbolicArgs(args...)
  return any(isSymbolicArg, args)
end

"""
Build the body Expr for a wrapper function with the given arity and dispatch mode.
Returns a quoted `function(args...) ... end` expression suitable for RuntimeGeneratedFunctions.
"""
function _buildWrapperBody(funcName::Symbol, nArgs::Int, arrayFunction::Bool, outputDims::Tuple{Vararg{Int}})
  local fnQuote = QuoteNode(funcName)
  local hasArrayDims = !isempty(outputDims)

  #= Build argument names. RTG does not support varargs, so always use fixed arity. =#
  local argNames = [Symbol("arg", i) for i in 1:nArgs]

  #= Build the symbolic check =#
  local symbolicCheckExpr
  if nArgs == 0
    symbolicCheckExpr = nothing
  elseif nArgs == 1
    symbolicCheckExpr = :(isSymbolicArg(arg1))
  else
    symbolicCheckExpr = Expr(:call, :hasSymbolicArgs, argNames...)
  end

  #= Build the unwrapped args for symbolic dispatch =#
  local uwArgsExpr
  if nArgs > 0
    uwArgsExpr = :(Any[$([ :(unwrapForSymbolic($a)) for a in argNames ]...)])
  end

  #= Build the symbolic return expression =#
  local symbolicReturnExpr
  if nArgs == 0
    symbolicReturnExpr = nothing
  elseif arrayFunction && hasArrayDims
    symbolicReturnExpr = :(return createSymbolicArrayCall(
      MODELICA_FUNCTION_WRAPPERS[$fnQuote], $uwArgsExpr, $outputDims;
      funcName = $fnQuote))
  elseif !arrayFunction
    symbolicReturnExpr = :(return Symbolics.Num(SymbolicUtils.Term{Real}(
      MODELICA_FUNCTION_WRAPPERS[$fnQuote], $uwArgsExpr)))
  else
    symbolicReturnExpr = nothing
  end

  #= Build the impl call =#
  local implCallExpr
  if nArgs == 0
    implCallExpr = :(Base.invokelatest(impl))
  else
    implCallExpr = Expr(:call, :(Base.invokelatest), :impl, argNames...)
  end

  #= Assemble the body block =#
  local stmts = Expr[]

  #= Add symbolic dispatch if applicable =#
  if symbolicCheckExpr !== nothing && symbolicReturnExpr !== nothing
    push!(stmts, Expr(:if, symbolicCheckExpr, symbolicReturnExpr))
  end

  #= Add impl lookup and call =#
  push!(stmts, :(impl = MODELICA_FUNCTION_IMPLS[$fnQuote]))
  push!(stmts, implCallExpr)

  local bodyBlock = Expr(:block, stmts...)

  #= Build the arrow function expression: (args...) -> begin ... end
     RTG requires arrow form with fixed arity (no varargs). =#
  if nArgs == 0
    return Expr(:->, Expr(:tuple), bodyBlock)
  else
    return Expr(:->, Expr(:tuple, argNames...), bodyBlock)
  end
end

function createModelicaFunctionWrapper(funcName::Symbol, nArgs::Int, arrayFunction::Bool = false, outputDims::Tuple{Vararg{Int}} = ())
  #= Always (re-)create the wrapper so the correct arrayFunction flag is applied. =#

  #= Build the function body expression and create an RTG function.
     RTG functions are world-age safe: they can be called from any world age,
     unlike @eval-created functions which are "too new" when called from
     RuntimeGeneratedFunction context (e.g., MTK equation evaluation). =#
  local body = _buildWrapperBody(funcName, nArgs, arrayFunction, outputDims)
  local rtg = RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, body)
  MODELICA_FUNCTION_WRAPPERS[funcName] = rtg

  #= Create a module-level binding so equation expressions that reference
     the function by name (e.g., in rewritten equations) can find it.
     Only create the binding if it does not already exist. The binding is a
     closure that delegates to the MODELICA_FUNCTION_WRAPPERS dict, so it
     always resolves to the latest RTG wrapper even if re-created. =#
  if !isdefined(@__MODULE__, funcName)
    local fnQuote = QuoteNode(funcName)
    @eval $funcName = MODELICA_FUNCTION_WRAPPERS[$fnQuote]
  end
end

#=
So we know about t an der in the global scope.
This is needed for the rules below to match correctly.
=#

@independent_variables t
const D = Differential(t)
using DataStructures

"""
Rewrite equations for MTK: move derivatives to the LHS, rename der to D,
qualify Modelica function calls, and wrap dynamic calls with invokelatest.
"""
function rewriteEquations(edeqs, simCode)
  local funcNames = Set{Symbol}(Symbol(replace(f.name, "." => "_")) for f in simCode.functions)
  return rewriteEquationsExprLevel(edeqs isa Vector{Expr} ? edeqs : Expr[e for e in edeqs];
                                   modelicaFuncNames = funcNames)
end

"""
  Check if a symbol is a registered dynamic Modelica function.
  Uses the MODELICA_FUNCTION_WRAPPERS dictionary populated by createModelicaFunctionWrapper.
"""
function isDynamicModelicaFunction(sym::Symbol)
  return haskey(MODELICA_FUNCTION_WRAPPERS, sym)
end

"""
Check if an Expr represents a qualified call to OMBackend.CodeGeneration.X
by inspecting the Expr structure directly instead of stringifying.
"""
function _isOMBackendQualifiedCall(e::Expr)
  e.head == :. || return false
  length(e.args) >= 1 || return false
  local lhs = e.args[1]
  #= Check for nested dot: OMBackend.CodeGeneration =#
  if lhs isa Expr && lhs.head == :.
    return lhs == :(OMBackend.CodeGeneration)
  end
  return false
end
_isOMBackendQualifiedCall(_) = false

"""
  Wrap function calls to dynamically generated Modelica functions with Base.invokelatest
  to avoid world-age issues.
"""
function wrapWithInvokelatest(expr::Expr)
  if expr.head == :call
    func = expr.args[1]
    #= Check if the function is a qualified call to OMBackend.CodeGeneration =#
    if func isa Expr && _isOMBackendQualifiedCall(func)
      #= Wrap with Base.invokelatest =#
      local newArgs = Any[:(Base.invokelatest), func]
      for a in expr.args[2:end]
        push!(newArgs, wrapWithInvokelatest(a))
      end
      return Expr(:call, newArgs...)
    #= Check if the function is a bare symbol that is a registered dynamic function =#
    elseif func isa Symbol && isDynamicModelicaFunction(func)
      #= Wrap with Base.invokelatest =#
      local newArgs = Any[:(Base.invokelatest), func]
      for a in expr.args[2:end]
        push!(newArgs, wrapWithInvokelatest(a))
      end
      return Expr(:call, newArgs...)
    end
    #= Recursively process arguments; return original if nothing changed =#
    local changed = false
    local newArgs = copy(expr.args)
    for (i, a) in enumerate(expr.args)
      r = wrapWithInvokelatest(a)
      newArgs[i] = r
      changed |= r !== a
    end
    return changed ? Expr(:call, newArgs...) : expr
  end
  #= For all other Expr types; return original if nothing changed =#
  local changed = false
  local newArgs = copy(expr.args)
  for (i, a) in enumerate(expr.args)
    r = wrapWithInvokelatest(a)
    newArgs[i] = r
    changed |= r !== a
  end
  return changed ? Expr(expr.head, newArgs...) : expr
end

wrapWithInvokelatest(x) = x  #= For non-Expr types, return as-is =#


"""
  $(SIGNATURES)

  Structurally simplify algebraic equations in a system and compute the
  topological sort of the observed equations. When `simplify=true`, the `simplify`
  function will be applied during the tearing process. It also takes kwargs
  `allow_symbolic=false` and `allow_parameter=true` which limits the coefficient
  types during tearing.

  The optional argument `io` may take a tuple `(inputs, outputs)`.
  This will convert all `inputs` to parameters and allow them to be unconnected, i.e.,
  simplification will allow models where `n_states = n_equations - n_inputs`.
  """

"""
  Filter out equations that contain no symbolic variables (e.g. `0 ~ 0.0`, `0 ~ 255.0`).
  These arise when a variable is reclassified as a parameter but its defining equation is kept.
  Such equations are either tautologies or contradictions and confuse the initialization system.
"""
function filterConstantEquations(eqs::AbstractVector)
  filtered = filter(eqs) do eq
    lhs_vars = Symbolics.get_variables(eq.lhs)
    rhs_vars = Symbolics.get_variables(eq.rhs)
    !isempty(lhs_vars) || !isempty(rhs_vars)
  end
  local n_removed = length(eqs) - length(filtered)
  if n_removed > 0
    @info "Removed $n_removed constant-only equations (no unknowns)"
  end
  return filtered
end

"""
    resolveAliasInitialValue(diffState, fullEqs, ivMap)

  Resolve the initial value of a differential state that has no direct Modelica
  start value (typically an MTK-generated derivative variable like `Xˍt`).

  Scans `fullEqs` for a 2-variable linear equation where one variable is
  `diffState` and the other has a known value in `ivMap`. Uses symbolic
  variable analysis (`Symbolics.get_variables`, `isequal`, `substitute`).

  Returns the resolved value, or `nothing` if no alias equation was found.
"""
function resolveAliasInitialValue(diffState, fullEqs, ivMap::Dict)
  #= Substitute all known values into each equation, then check if the equation
     becomes a simple linear expression in diffState alone. This avoids type
     mismatches between get_variables output and ivMap keys. =#
  for eq in fullEqs
    local eqStr = string(eq)
    local diffStr = string(diffState)
    if !contains(eqStr, diffStr)
      continue
    end
    #= Skip differential equations (Differential(t)(X) ~ ...) as these are
       dynamic equations, not algebraic alias equations. =#
    if contains(string(eq.lhs), "Differential")
      continue
    end
    local expr = eq.lhs - eq.rhs
    #= Substitute all known initial values =#
    local exprSub = Symbolics.substitute(expr, ivMap)
    #= Check if what remains is linear in diffState =#
    local c = Symbolics.substitute(exprSub, Dict(diffState => 0))
    local a_plus_c = Symbolics.substitute(exprSub, Dict(diffState => 1))
    if !(c isa Number) || !(a_plus_c isa Number)
      continue
    end
    local a = a_plus_c - c
    if !iszero(a)
      return Float64(Symbolics.unwrap(-c / a))
    end
  end
  return nothing
end

"""
    splitInitialValues(reducedSystem, finalInitialValues)

  Split initial values into hard constraints and soft guesses based on the mass matrix.
  Differential states (mass matrix diagonal != 0) get hard u0 values.
  Algebraic states (mass matrix diagonal == 0) become guesses to avoid
  overdetermining the initialization system.
  For pure ODE systems (identity mass matrix), all values stay hard.

  Returns `(system, hardInitialValues)` where system may have updated guesses.
"""
function splitInitialValues(reducedSystem, finalInitialValues, allInitialValues = Pair[])
  local massMatrix = ModelingToolkit.calculate_massmatrix(reducedSystem)
  local reducedUnks = unknowns(reducedSystem)
  #= Identity mass matrix means pure ODE: all states are differential =#
  if massMatrix isa LinearAlgebra.UniformScaling
    @info "ODEProblem: pure ODE (identity mass matrix), $(length(finalInitialValues)) hard u0, $(length(reducedUnks)) unknowns"
    return (reducedSystem, finalInitialValues)
  end
  #= DAE system: classify states by mass matrix diagonal =#
  @BACKEND_LOGGING @info "[splitIV] finalInitialValues keys:" [string(p.first) for p in finalInitialValues]
  @BACKEND_LOGGING @info "[splitIV] reducedUnks:" [string(u) for u in reducedUnks]
  local diffStateSet = Set{Any}()
  local diffStateStrSet = Set{String}()
  for i in 1:min(size(massMatrix, 1), length(reducedUnks))
    if massMatrix[i, i] != 0
      push!(diffStateSet, reducedUnks[i])
      push!(diffStateStrSet, string(reducedUnks[i]))
    end
  end
  #= Use string comparison for hard/soft split because finalInitialValues keys
     are Num-wrapped while diffStateSet contains unwrapped BasicSymbolic values.
     isequal(Num(x), x) can fail depending on Symbolics version. =#
  local hardInitialValues = filter(finalInitialValues) do pair
    string(pair.first) in diffStateStrSet
  end
  local softInitialValues = filter(finalInitialValues) do pair
    !(string(pair.first) in diffStateStrSet)
  end
  #= If no differential state has an explicit IV but there are algebraic IVs,
     keep all IVs as hard constraints. This handles models where only algebraic
     variables have start values (e.g. VSS Pendulum: initial equation x=x0, y=y0
     but no phi start). MTK needs these as hard constraints to determine the
     differential states via algebraic equations during initialization.
     Only demote algebraic IVs to guesses when differential states also have IVs. =#
  if isempty(hardInitialValues) && !isempty(softInitialValues)
    @info "No differential states have explicit IVs; keeping all $(length(softInitialValues)) algebraic IVs as hard constraints"
    hardInitialValues = softInitialValues
    softInitialValues = Pair[]
  end
  if !isempty(softInitialValues)
    local currentGuesses = ModelingToolkit.guesses(reducedSystem)
    local newGuesses = merge(currentGuesses, Dict(softInitialValues))
    @set! reducedSystem.guesses = newGuesses
  end
  #= Ensure all differential states have hard initial values.
     MTK's order-lowering creates derivative variables (e.g. Inertia_phiˍt for
     der(Inertia_phi)) that have no Modelica start value.
     Use resolveAliasInitialValue to find alias equations in the reduced system.
     The allIVMap includes both reduced-system unknowns AND pre-simplification
     variables (allInitialValues) so we can resolve aliases to variables that
     were eliminated by structural_simplify. =#
  #= Use string-based set for checking existing hard IVs, because isequal between
     Num-wrapped keys (from finalInitialValues) and BasicSymbolic values (from
     reducedUnks/diffStateSet) can fail. =#
  local hardSymStrSet = Set(string(iv.first) for iv in hardInitialValues)
  local allIVMap = Dict{Any, Any}(iv.first => iv.second
                                  for iv in vcat(hardInitialValues, softInitialValues, allInitialValues))
  local fullEqs = ModelingToolkit.full_equations(reducedSystem)
  for diffState in diffStateSet
    if !(string(diffState) in hardSymStrSet)
      local diffStateStr = string(diffState)
      #= Only attempt alias resolution for MTK-generated derivative variables
         (e.g. Inertia_phiˍt created by order-lowering). These have the Unicode
         dot character ˍ in their name and no Modelica start value.
         For regular Modelica variables without explicit start values, do NOT add
         a hard u0 entry. Let MTK's initialization solver infer their values from
         algebraic constraints and guesses (e.g. phi inferred from x,y via
         x = L*sin(phi)). Adding hard 0.0 would override this inference. =#
      if contains(diffStateStr, "\u02cd")
        local ivMapForResolve = filter(p -> string(p.first) != diffStateStr, allIVMap)
        local resolved = resolveAliasInitialValue(diffState, fullEqs, ivMapForResolve)
        local finalVal = something(resolved, 0.0)
        push!(hardInitialValues, diffState => finalVal)
        if resolved !== nothing
          @info "Resolved differential state $(diffState) to $(finalVal) via equation alias"
        else
          @info "Defaulted MTK derivative $(diffState) to 0.0 (no alias equation found)"
        end
      else
        @info "Leaving differential state $(diffState) for MTK initialization (no explicit start value)"
      end
    end
  end
  @info "ODEProblem: DAE, $(length(hardInitialValues)) hard u0 (differential), $(length(softInitialValues)) as guesses (algebraic), $(length(reducedUnks)) unknowns ($(length(diffStateSet)) differential)"
  return (reducedSystem, hardInitialValues)
end

"""
    injectObservedEquations(sys, observedEqs)

Append `observedEqs` to the observed equations of a completed system.
Used to inject observed equations AFTER structural_simplify so they do not
interfere with AffectSystem tearing during callback compilation.
"""
function injectObservedEquations(sys, observedEqs::Vector)
  #= MTK's getproperty follows the parent chain: if a completed system has
     a parent, getvar is called on the PARENT, not on the child. So we must
     inject into the parent as well, otherwise variable lookups from the
     solution (e.g. sol[:absX]) will fail with "variable does not exist".
     The observed field is a mutable Vector{Equation}, so we append! in-place
     to avoid copying the entire immutable System struct. =#
  append!(ModelingToolkit.get_observed(sys), observedEqs)
  parent = ModelingToolkit.get_parent(sys)
  if parent !== nothing
    append!(ModelingToolkit.get_observed(parent), observedEqs)
  end
  return sys
end

"""
  TODO:
  Document why some parts here are outcommented
  The irreductable variables scheme does not work using plain simplify.

  It should be noted that for some models both running tearing and structurally simplify is needed.
  Report and issue for the MTK reporters giving an example of this behavior.

  One example is running tearing twice broke the system
"""
function structural_simplify(sys::ModelingToolkit.AbstractSystem,
                             io = nothing;
                             simplify = false,
                             allow_parameter = true,
                             kwargs...)
  local pre_eqs = length(equations(sys))
  local pre_unknowns = length(unknowns(sys))
  @info "Before structural_simplify: $pre_eqs equations, $pre_unknowns unknowns"
  @BACKEND_LOGGING begin
    open("mtk_preSimplify.log", "w") do io
      println(io, "############################################")
      println(io, "MTK system before structural_simplify")
      println(io, "############################################")
      println(io)
      println(io, "Unknowns ($pre_unknowns):")
      println(io, "---------------------------------------------")
      for (i, u) in enumerate(unknowns(sys))
        println(io, "  [$i] $u")
      end
      println(io)
      println(io, "Equations ($pre_eqs):")
      println(io, "---------------------------------------------")
      for (i, eq) in enumerate(equations(sys))
        println(io, "  [$i] $eq")
      end
      println(io)
      println(io, "Parameters ($(length(parameters(sys)))):")
      println(io, "---------------------------------------------")
      for (i, p) in enumerate(parameters(sys))
        println(io, "  [$i] $p")
      end
      local defs = ModelingToolkit.defaults(sys)
      println(io)
      println(io, "Defaults ($(length(defs))):")
      println(io, "---------------------------------------------")
      for (k, v) in defs
        println(io, "  $k => $v")
      end
    end
  end
  #= DirectRHS uses a plain Float64[] parameter vector, so the reduced system
     must use split=false (flat vector) rather than split=true (MTKParameters).
     This also makes generate_continuous_callbacks produce callbacks compatible
     with the flat format. The non-DirectRHS path (VSS, standard ODEProblem)
     needs split=true for SCCNonlinearProblem initialization to work.
     The split value is embedded at codegen time by performStructuralSimplify. =#
  local useSplit = get(kwargs, :split, true)
  if OMBackend.ENABLE_BACKEND_LOGGING
    local _ss_timed = @timed ModelingToolkit.structural_simplify(sys; simplify = simplify, split = useSplit)
    sys = _ss_timed.value
    @info "structural_simplify took $(_ss_timed.time)s, $(round(_ss_timed.bytes / 1e9, digits=2)) GiB"
  else
    sys = ModelingToolkit.structural_simplify(sys; simplify = simplify, split = useSplit)
  end
  local post_eqs = length(equations(sys))
  local post_full_eqs = length(ModelingToolkit.full_equations(sys))
  local post_unknowns = length(unknowns(sys))
  @info "After structural_simplify: equations=$(post_eqs), full_equations=$(post_full_eqs), unknowns=$(post_unknowns)"
  if post_eqs != post_unknowns
    @warn "equations(sys) != unknowns(sys): $post_eqs vs $post_unknowns"
    for (i, eq) in enumerate(equations(sys))
      @info "  eq[$i]: $eq"
    end
    for (i, u) in enumerate(unknowns(sys))
      @info "  unk[$i]: $u"
    end
  end
  if post_full_eqs != post_unknowns
    @warn "full_equations(sys) != unknowns(sys): $post_full_eqs vs $post_unknowns"
  end
  @BACKEND_LOGGING begin
    open("mtk_postSimplify.log", "w") do io
      println(io, "############################################")
      println(io, "MTK system after structural_simplify")
      println(io, "############################################")
      println(io)
      println(io, "Unknowns ($post_unknowns):")
      for (i, u) in enumerate(unknowns(sys))
        println(io, "  [$i] $u")
      end
      println(io)
      println(io, "Equations ($post_eqs):")
      for (i, eq) in enumerate(equations(sys))
        println(io, "  [$i] $eq")
      end
      println(io)
      println(io, "Guesses ($(length(ModelingToolkit.guesses(sys)))):")
      for (k, v) in ModelingToolkit.guesses(sys)
        println(io, "  $k => $v")
      end
      println(io)
      println(io, "Full equations ($post_full_eqs):")
      for (i, eq) in enumerate(ModelingToolkit.full_equations(sys))
        println(io, "  [$i] $eq")
      end
    end
  end
  #= Workaround: after structural_simplify, the tearing state's graph may have
     stale dimensions (more equations/variables than equations(sys)/unknowns(sys)).
     This causes a DimensionMismatch in W_sparsity when the graph-based Jacobian
     sparsity (nsrcs x nsrcs) is broadcast against the mass matrix (neqs x neqs).
     Clearing the tearing state forces jacobian_sparsity to fall back to symbolic
     computation, which uses the correct equation/unknown counts. =#
  try
    local ts = ModelingToolkit.get_tearing_state(sys)
    if ts !== nothing
      local g = ts.structure.graph
      local graph_eqs = ModelingToolkit.BipartiteGraphs.nsrcs(g)
      local sys_eqs = length(equations(sys))
      if graph_eqs != sys_eqs
        @warn "Tearing state graph has $graph_eqs equations but system has $sys_eqs; clearing tearing state"
        @set! sys.tearing_state = nothing
      end
    end
  catch ex
    @warn "Could not inspect tearing state" exception=(ex, catch_backtrace())
  end
  #= Filter guesses to only include variables that are unknowns of the reduced system.
     After structural_simplify, many original unknowns become observed (algebraically
     determined). Keeping guesses for those creates an overdetermined initialization. =#
  local reducedUnknowns = Set(unknowns(sys))
  local currentGuesses = ModelingToolkit.guesses(sys)
  local filteredGuesses = Dict(k => v for (k, v) in currentGuesses if k in reducedUnknowns)
  if length(filteredGuesses) != length(currentGuesses)
    @info "Filtered guesses: $(length(currentGuesses)) -> $(length(filteredGuesses)) (removed $(length(currentGuesses) - length(filteredGuesses)) non-unknown guesses)"
    @set! sys.guesses = filteredGuesses
  end
  return sys
end

"""
  $(TYPEDSIGNATURES)

  Takes a Nth order System and returns a new System written in first order
  form by defining new variables which represent the N-1 derivatives.
  """
function ode_order_lowering(sys::System)
  iv = ModelingToolkit.get_iv(sys)
  eqs_lowered, new_vars = ode_order_lowering(equations(sys), iv, unknowns(sys))
  @set! sys.eqs = eqs_lowered
  @set! sys.unknowns = new_vars
  return sys
end

function dae_order_lowering(sys::System)
  iv = get_iv(sys)
  eqs_lowered, new_vars = dae_order_lowering(equations(sys), iv, unknowns(sys))
  @set! sys.eqs = eqs_lowered
  @set! sys.unknowns = new_vars
  return sys
end

function ode_order_lowering(eqs, iv, unknown_vars)
  var_order = OrderedDict{Any, Int}()
  D = Differential(iv)
  diff_eqs = Equation[]
  diff_vars = []
  alge_eqs = Equation[]
  for (i, eq) in enumerate(eqs)
    if !isdiffeq(eq)
      push!(alge_eqs, eq)
    else
      var, maxorder = ModelingToolkit.var_from_nested_derivative(eq.lhs)
      maxorder > get(var_order, var, 1) && (var_order[var] = maxorder)
      var′ = ModelingToolkit.lower_varname(var, iv, maxorder - 1)
      if ! isreal(eq.rhs) #= Modification by me. =#
        rhs′ = ModelingToolkit.diff2term_with_unit(eq.rhs, iv)
      else
        rhs′ = eq.rhs
      end
      push!(diff_vars, var′)
      push!(diff_eqs, D(var′) ~ rhs′)
    end
  end
  for (var, order) in var_order
    for o in (order - 1):-1:1
      lvar = lower_varname(var, iv, o - 1)
      rvar = lower_varname(var, iv, o)
      push!(diff_vars, lvar)

      rhs = rvar
      eq = Differential(iv)(lvar) ~ rhs
      push!(diff_eqs, eq)
    end
  end
  # we want to order the equations and variables to be `(diff, alge)`
  return (vcat(diff_eqs, alge_eqs), vcat(diff_vars, setdiff(unknown_vars, diff_vars)))
end

function dae_order_lowering(eqs, iv, unknown_vars)
  var_order = OrderedDict{Any, Int}()
  D = Differential(iv)
  diff_eqs = Equation[]
  diff_vars = OrderedSet()
  alge_eqs = Equation[]
  vars = Set()
  subs = Dict()

  for (i, eq) in enumerate(eqs)
    vars!(vars, eq)
    n_diffvars = 0
    for vv in vars
      isdifferential(vv) || continue
      var, maxorder = var_from_nested_derivative(vv)
      isparameter(var) && continue
      n_diffvars += 1
      order = get(var_order, var, nothing)
      seen = order !== nothing
      if !seen
        order = 1
      end
      maxorder > order && (var_order[var] = maxorder)
      var′ = lower_varname(var, iv, maxorder - 1)
      subs[vv] = D(var′)
      if !seen
        push!(diff_vars, var′)
      end
    end
    n_diffvars == 0 && push!(alge_eqs, eq)
    empty!(vars)
  end

  for (var, order) in var_order
    for o in (order - 1):-1:1
      lvar = lower_varname(var, iv, o - 1)
      rvar = lower_varname(var, iv, o)
      push!(diff_vars, lvar)

      rhs = rvar
      eq = Differential(iv)(lvar) ~ rhs
      push!(diff_eqs, eq)
    end
  end

  return ([diff_eqs; substitute.(eqs, (subs,))],
          vcat(collect(diff_vars), setdiff(unknown_vars, diff_vars)))
end

function getStatesAsSymbolicVariables(odeFunc::ODEFunction)
  return ModelingToolkit.get_unknowns(odeFunc.sys)
end

function getStatesAsSymbols(odeFunc::ODEFunction)
  local states = ModelingToolkit.get_unknowns(odeFunc.sys)
  map(x->x.f.name, states)
end

function getParametersAsSymbols(odeFunc::ODEFunction)
  local states = ModelingToolkit.parameters(odeFunc.sys)
  map(x->x.name, states)
end

function getSymsAsStrings(odeFunc::ODEFunction)
  local unknowns = ModelingToolkit.parameters(odeFunc.sys)
  return map(string, unknowns)
end
