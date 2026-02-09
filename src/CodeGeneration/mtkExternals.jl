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

"""
Helper to check if a value is symbolic (Symbolics.Num or contains symbolic expressions).
"""
function isSymbolicArg(x)
  return x isa Symbolics.Num || x isa Symbolics.Arr || x isa SymbolicUtils.BasicSymbolic
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
function createModelicaFunctionWrapper(funcName::Symbol, nArgs::Int)
  #= Check if symbol already exists - if so, skip redefinition =#
  #= The implementation in MODELICA_FUNCTION_IMPLS is updated separately =#
  #= Any existing callable (function or RTG object) will use the new impl via invokelatest =#
  if isdefined(@__MODULE__, funcName)
    return nothing
  end

  #= Define a function that:
     1. Checks if any argument is symbolic
     2. If so, returns a symbolic term (like @register_symbolic would)
     3. Otherwise calls the implementation via invokelatest
  =#
  local fnQuote = QuoteNode(funcName)
  if nArgs == 0
    @eval function $funcName()
      impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      Base.invokelatest(impl)
    end
  elseif nArgs == 1
    @eval function $funcName(arg1)
      #= If argument is symbolic, return a symbolic term =#
      if isSymbolicArg(arg1)
        return Symbolics.Num(SymbolicUtils.Term($funcName, [Symbolics.unwrap(arg1)]))
      end
      impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      Base.invokelatest(impl, arg1)
    end
  elseif nArgs == 2
    @eval function $funcName(arg1, arg2)
      if hasSymbolicArgs(arg1, arg2)
        return Symbolics.Num(SymbolicUtils.Term($funcName, [Symbolics.unwrap(arg1), Symbolics.unwrap(arg2)]))
      end
      impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      Base.invokelatest(impl, arg1, arg2)
    end
  elseif nArgs == 3
    @eval function $funcName(arg1, arg2, arg3)
      if hasSymbolicArgs(arg1, arg2, arg3)
        return Symbolics.Num(SymbolicUtils.Term($funcName, [Symbolics.unwrap(arg1), Symbolics.unwrap(arg2), Symbolics.unwrap(arg3)]))
      end
      impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      Base.invokelatest(impl, arg1, arg2, arg3)
    end
  elseif nArgs == 4
    @eval function $funcName(arg1, arg2, arg3, arg4)
      if hasSymbolicArgs(arg1, arg2, arg3, arg4)
        return Symbolics.Num(SymbolicUtils.Term($funcName, [Symbolics.unwrap(arg1), Symbolics.unwrap(arg2), Symbolics.unwrap(arg3), Symbolics.unwrap(arg4)]))
      end
      impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      Base.invokelatest(impl, arg1, arg2, arg3, arg4)
    end
  else
    @eval function $funcName(args...)
      if any(isSymbolicArg, args)
        return Symbolics.Num(SymbolicUtils.Term($funcName, [Symbolics.unwrap(a) for a in args]))
      end
      impl = MODELICA_FUNCTION_IMPLS[$fnQuote]
      Base.invokelatest(impl, args...)
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
  sys = ModelingToolkit.structural_simplify(sys, simplify = simplify)
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
