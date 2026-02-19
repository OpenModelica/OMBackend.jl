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

#= Global registry of dynamically generated Modelica function names =#
#= Used by wrapWithInvokelatest to detect which bare symbols need invokelatest wrapping =#
const DYNAMIC_MODELICA_FUNCTIONS = Set{Symbol}()

"""
  Register a dynamically generated Modelica function name.
  Called when functions are eval'd in MTK_CodeGeneration.
"""
function registerDynamicFunction!(funcName::Symbol)
  push!(DYNAMIC_MODELICA_FUNCTIONS, funcName)
end

#= Global dictionary to store dynamically generated function implementations =#
#= The key is the function name, the value is the implementation function =#
const MODELICA_FUNCTION_IMPLS = Dict{Symbol, Function}()

#= Global dictionary to store RTG wrappers for each function =#
const MODELICA_FUNCTION_WRAPPERS = Dict{Symbol, Any}()

#= Cache for per-element extractor functions.
   Key: (funcName::Symbol, indices::Tuple{Vararg{Int}})
   Value: the created function object =#
const ELEM_FUNC_CACHE = Dict{Tuple{Symbol, Tuple}, Function}()

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
function getOrCreateElemFunc(funcName::Symbol, indices::Tuple{Vararg{Int}})
  local key = (funcName, indices)
  if haskey(ELEM_FUNC_CACHE, key)
    return ELEM_FUNC_CACHE[key]
  end
  local idxStr = join(indices, "_")
  local elemName = Symbol(funcName, :__e, idxStr)
  local fnQuote = QuoteNode(funcName)
  if length(indices) == 1
    local idx = indices[1]
    @eval function $elemName(args...)
      local impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      local result = Base.invokelatest(impl, args...)
      return result[$idx]
    end
  elseif length(indices) == 2
    local i = indices[1]
    local j = indices[2]
    @eval function $elemName(args...)
      local impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      local result = Base.invokelatest(impl, args...)
      local row = result[$i]
      if row isa AbstractVector
        return row[$j]
      else
        return result[$i, $j]
      end
    end
  end
  local f = getfield(@__MODULE__, elemName)
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
function createSymbolicArrayCall(funcRef, uwArgs::Vector{Any}, dims::Tuple{Vararg{Int}})
  local funcName = nameof(funcRef)
  if length(dims) == 1
    return [Symbolics.Num(SymbolicUtils.Term{Real}(getOrCreateElemFunc(funcName, (i,)), uwArgs)) for i in 1:dims[1]]
  elseif length(dims) == 2
    return [Symbolics.Num(SymbolicUtils.Term{Real}(getOrCreateElemFunc(funcName, (i, j)), uwArgs)) for i in 1:dims[1], j in 1:dims[2]]
  else
    #= 3D+ fallback: use linear indexing =#
    local totalLen = prod(dims)
    return [Symbolics.Num(SymbolicUtils.Term{Real}(getOrCreateElemFunc(funcName, (i,)), uwArgs)) for i in 1:totalLen]
  end
end

"""
Helper to check if a value is symbolic (Symbolics.Num or contains symbolic expressions).
Also recursively checks arrays, since record field arrays assembled from scalarized
parameters may be Vector{Symbolics.Num} or Vector{Vector{Symbolics.Num}}.
"""
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
  Create a wrapper function that calls the implementation via Base.invokelatest.
  This handles world-age issues when the implementation is defined via eval at runtime.

  The wrapper also handles symbolic arguments: if any argument is symbolic (Symbolics.Num),
  it returns a symbolic term instead of calling the implementation. This replaces the need
  for @register_symbolic which does not work reliably when called at runtime.

TODO:
This error still persists in some generations.
"""
function createModelicaFunctionWrapper(funcName::Symbol, nArgs::Int, arrayFunction::Bool = false, outputDims::Tuple{Vararg{Int}} = ())
  #= Always (re-)create the wrapper so the correct arrayFunction flag is applied.
     The wrapper is cheap to create and the impl dict lookup is the same regardless. =#

  #= Define a function that:
     1. Checks if any argument is symbolic
     2. If so, returns a symbolic term (scalar) or vector of symbolic terms (array)
     3. Otherwise calls the implementation via invokelatest
     For array-returning functions with known outputLen, symbolic args produce a
     Vector{Num} of per-element Term nodes: [f(args...)[1], f(args...)[2], ...].
     This defers the actual computation (including any if-statements) to runtime.
  =#
  local fnQuote = QuoteNode(funcName)
  local hasArrayDims = !isempty(outputDims)
  if nArgs == 0
    @eval function $funcName()
      impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      Base.invokelatest(impl)
    end
  elseif nArgs == 1
    if arrayFunction && hasArrayDims
      @eval function $funcName(arg1)
        if isSymbolicArg(arg1)
          return createSymbolicArrayCall($funcName, Any[unwrapForSymbolic(arg1)], $outputDims)
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1)
      end
    elseif arrayFunction
      @eval function $funcName(arg1)
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1)
      end
    else
      @eval function $funcName(arg1)
        if isSymbolicArg(arg1)
          return Symbolics.Num(SymbolicUtils.Term{Real}($funcName, [unwrapForSymbolic(arg1)]))
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1)
      end
    end
  elseif nArgs == 2
    if arrayFunction && hasArrayDims
      @eval function $funcName(arg1, arg2)
        if hasSymbolicArgs(arg1, arg2)
          return createSymbolicArrayCall($funcName, Any[unwrapForSymbolic(arg1), unwrapForSymbolic(arg2)], $outputDims)
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2)
      end
    elseif arrayFunction
      @eval function $funcName(arg1, arg2)
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2)
      end
    else
      @eval function $funcName(arg1, arg2)
        if hasSymbolicArgs(arg1, arg2)
          return Symbolics.Num(SymbolicUtils.Term{Real}($funcName, [unwrapForSymbolic(arg1), unwrapForSymbolic(arg2)]))
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2)
      end
    end
  elseif nArgs == 3
    if arrayFunction && hasArrayDims
      @eval function $funcName(arg1, arg2, arg3)
        if hasSymbolicArgs(arg1, arg2, arg3)
          return createSymbolicArrayCall($funcName, Any[unwrapForSymbolic(arg1), unwrapForSymbolic(arg2), unwrapForSymbolic(arg3)], $outputDims)
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2, arg3)
      end
    elseif arrayFunction
      @eval function $funcName(arg1, arg2, arg3)
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2, arg3)
      end
    else
      @eval function $funcName(arg1, arg2, arg3)
        if hasSymbolicArgs(arg1, arg2, arg3)
          return Symbolics.Num(SymbolicUtils.Term{Real}($funcName, [unwrapForSymbolic(arg1), unwrapForSymbolic(arg2), unwrapForSymbolic(arg3)]))
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2, arg3)
      end
    end
  elseif nArgs == 4
    if arrayFunction && hasArrayDims
      @eval function $funcName(arg1, arg2, arg3, arg4)
        if hasSymbolicArgs(arg1, arg2, arg3, arg4)
          return createSymbolicArrayCall($funcName, Any[unwrapForSymbolic(arg1), unwrapForSymbolic(arg2), unwrapForSymbolic(arg3), unwrapForSymbolic(arg4)], $outputDims)
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2, arg3, arg4)
      end
    elseif arrayFunction
      @eval function $funcName(arg1, arg2, arg3, arg4)
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2, arg3, arg4)
      end
    else
      @eval function $funcName(arg1, arg2, arg3, arg4)
        if hasSymbolicArgs(arg1, arg2, arg3, arg4)
          return Symbolics.Num(SymbolicUtils.Term{Real}($funcName, [unwrapForSymbolic(arg1), unwrapForSymbolic(arg2), unwrapForSymbolic(arg3), unwrapForSymbolic(arg4)]))
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, arg1, arg2, arg3, arg4)
      end
    end
  else
    if arrayFunction && hasArrayDims
      @eval function $funcName(args...)
        if any(isSymbolicArg, args)
          return createSymbolicArrayCall($funcName, Any[unwrapForSymbolic(a) for a in args], $outputDims)
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, args...)
      end
    elseif arrayFunction
      @eval function $funcName(args...)
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, args...)
      end
    else
      @eval function $funcName(args...)
        if any(isSymbolicArg, args)
          return Symbolics.Num(SymbolicUtils.Term{Real}($funcName, [unwrapForSymbolic(a) for a in args]))
        end
        impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
        Base.invokelatest(impl, args...)
      end
    end
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
  Uses the global DYNAMIC_MODELICA_FUNCTIONS registry populated when functions are eval'd.
"""
function isDynamicModelicaFunction(sym::Symbol)
  return sym in DYNAMIC_MODELICA_FUNCTIONS
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
    #= Recursively process arguments =#
    local processedArgs = Any[]
    for a in expr.args
      push!(processedArgs, wrapWithInvokelatest(a))
    end
    return Expr(:call, processedArgs...)
  end
  #= For all other Expr types, recursively process arguments =#
  local processedArgs = Any[]
  for a in expr.args
    push!(processedArgs, wrapWithInvokelatest(a))
  end
  return Expr(expr.head, processedArgs...)
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
  #sys = ModelingToolkit.ode_order_lowering(sys)
  #sys = ModelingToolkit.dae_index_lowering(sys)
  #sys = ModelingToolkit.tearing(sys; simplify = simplify)
  if false #Note report this to the developers of modeling toolkit.
    sys = ode_order_lowering(sys)
    sys = dae_index_lowering(sys)
    sys = ModelingToolkit.tearing(sys; simplify = false, allow_parameter = true)
   # sys = mtkcompile(sys)
    #Note some system breaks if tearing is run twice.
    # Note2 In some cases we need to do index reduction before simplify
    # return complete(sys) #Addition. To be removed.
  end
  local pre_eqs = length(equations(sys))
  local pre_unknowns = length(unknowns(sys))
  @info "Before structural_simplify: $pre_eqs equations, $pre_unknowns unknowns"
  sys = ModelingToolkit.structural_simplify(sys, simplify = simplify)
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
