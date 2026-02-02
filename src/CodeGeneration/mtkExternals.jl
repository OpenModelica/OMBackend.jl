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

@independent_variables t #ModelingToolkit.@variables t
const D = Differential(t)

using DataStructures

"""
    Temporary rewrite function. Not very pretty...
    Original code by Chris R. Expanded to fix terms of type X * D(Y).
    The solution to solve it is not pretty and is probably flaky.
  #istree returns true if x is a term. If true, operation, arguments must also be defined for x appropriately.
"""
function move_diffs(eq::Equation; rewrite)
  # Do not modify `D(x) ~ ...`, already correct
  res =
    if !(istree(eq.lhs) && operation(eq.lhs) isa Differential) && !(istree(eq.lhs) && operation(eq.lhs) isa Difference)
      local _eq
      _eq = eq.rhs-eq.lhs
      rhs = rewrite(_eq)
      if rhs === nothing
        return eq
      end
      lhs = _eq - rhs
      if !(lhs isa Number) && (operation(lhs) isa Differential)
        lhs ~ -rhs
      elseif !(lhs isa Number) && (operation(lhs) == *)
        #=
        This code is probably quite flaky, however, it should not be needed after similar things are introduced in MTK.
        TODO: Handle this more elegantly using rewrite rules instead.
        =#
        local newRhs
        local newLhs
        for arg in arguments(lhs)
          local isTermAndNotDifferential = istree(arg) && !(operation(arg) isa Differential)
          local argIsANumberOrSymbolButNotTerm = (arg isa Number || (arg isa SymbolicUtils.BasicSymbolic{Real} && !istree(arg)))
          if argIsANumberOrSymbolButNotTerm || isTermAndNotDifferential
            newRhs = substitute(rhs, rhs => rhs / arg)
            rhs = newRhs
            newLhs = substitute(lhs, lhs => lhs / arg)
            lhs = newLhs
            #@info "New equation in the for loop is $(lhs) = $(rhs)"
          end
        end
        tmp = ~(newLhs, -newRhs)
        tmp
      else
        -lhs ~ rhs
      end
    else
      eq
    end
  return res
end

"""
    Rewrite equations that do not conform to the requirements of MTK.
  Since MTK currently requires the derivative to be at the lhs.
  """
function rewriteEquations(edeqs, iv, eVars, ePars, simCode; arrayParameterExprs::Vector{Expr} = Expr[])
  #println("Recived #edeqs")
  #println(length(edeqs))
  #= TODO: Try to move der to the top level to avoid eval here. =#
  local der = ModelingToolkit.Differential(t)
  #= Remove the t's =#
  eVars = [Symbol(replace(string(v), "(t)" => "")) for v in eVars]
  #Temporary fix for the ESCIMO climate model: eVars = vcat(eVars, [Symbol("combi_Population_Lookup_bn_y")])
  #=
  TODO: This should ideally be done without using eval.
  =#
  preEval = quote
    vars = ModelingToolkit.@variables begin
      $(eVars...)
    end
    pars = ModelingToolkit.@parameters begin
      $(ePars...)
    end
  end
  #Hardcoded
  #= Make the derivative symbol known =#
  eval(preEval)
  eval(:(der = ModelingToolkit.Differential(t)))
  #= Make array parameters available as concrete Julia arrays =#
  for ap in arrayParameterExprs
    eval(ap)
  end
  #= Make the external runtime available if it is used. =#
  if simCode.externalRuntime
    eval(generateExternalRuntimeImport(simCode))
  end
  #= Note: @register_symbolic is now called earlier in ODE_MODE_MTK_MODEL_GENERATION =#
  #= immediately after function definitions are eval'd, before equations are processed =#
  local deqs = evalEDeqs(edeqs)
  #= Use invokelatest to access vars which was created via eval in a newer world age =#
  Base.invokelatest() do
    write("MTK_REWRITE.log", debugRewrite(deqs, iv, vars, parameters; separator="\n"))
  end
  #= Rewrite equations =#
  D = Differential(iv)
  local r1 = SymbolicUtils.@rule ~~a * D(~~b) * ~~c => 0
  local r2 = SymbolicUtils.@rule D(~~b) => 0
  local remove_diffs = SymbolicUtils.Postwalk(SymbolicUtils.Chain([r1,r2]))
  local usedStates = Set()
  local rewrittenDeqs = Symbolics.Equation[]
  local req
  for eq in deqs
    #= Only do the rewrite for the differentials. The others have already been rewritten.=#
    eqStr = string(eq)
    if (contains(eqStr, "Differential")) # TODO expensive comp! Needs to be optimized.
      req = move_diffs(eq, rewrite = remove_diffs)
      #@info "Left hand side of the equation" req.lhs
      if req.lhs isa Real
        push!(rewrittenDeqs, req)
      elseif !(req.lhs in usedStates)
        #@info "Not a duplicate" req.lhs
        #@info "Used equations are" usedStates
        push!(rewrittenDeqs, req)
        push!(usedStates, req.lhs)
      else
        #@info "Duplicate:" req.lhs
        push!(rewrittenDeqs, eq)
      end
    else
      push!(rewrittenDeqs, eq)
    end
  end
  @BACKEND_LOGGING OMBackend.debugWrite("codeAfterBackendPreprocessing.log", debugRewrite(rewrittenDeqs, iv, vars, parameters; separator="\n"))
  return rewrittenDeqs
end

"""
  This function evaluates the supplied equations.
  In the case we are unable to evaluate them, we currently hack it by some string conversions.
  TODO:
    Fix me do this the proper way.
    This routine is way way to slow currently...
  """
function evalEDeqs(edeqs)
  writeEqsToFile(edeqs, "beforeEqRewrite.log")
  local deqs = []
  for e in edeqs
    try
      eq = eval(e)
      if typeof(eq.lhs) == Int64 && eq.lhs == 0
        push!(deqs, eq)
      else
        local unSimplifiedString = string(e)
        unSimplifiedString = replace(unSimplifiedString, "&&" => "&")
        unSimplifiedString = replace(unSimplifiedString, r"\bbegin\b" => "(")
        unSimplifiedString = replace(unSimplifiedString, r"\bend\b" => ")")
        #println(unSimplifiedString)
        estrExp = Meta.parse(unSimplifiedString)
        estrExp2 = eval(estrExp)
        estrExp2LHS = estrExp2.lhs
        @assign estrExp2.lhs = 0
        @assign estrExp2.rhs = estrExp2.rhs - estrExp2LHS
        push!(deqs, estrExp2)
      end
    catch ex
      global TEST = e
      local unSimplifiedString = string(e)
      unSimplifiedString = replace(unSimplifiedString, "&&" => "&")
      unSimplifiedString = replace(unSimplifiedString, r"\bbegin\b" => "(")
      unSimplifiedString = replace(unSimplifiedString, r"\bend\b" => ")")
      estrExp = Meta.parse(unSimplifiedString)
      estrExp2 = eval(estrExp)
      estrExp2LHS = estrExp2.lhs
      @assign estrExp2.lhs = 0
      @assign estrExp2.rhs = estrExp2.rhs - estrExp2LHS
      push!(deqs, estrExp2)
    end
  end
  return deqs
end

function debugRewrite(deqs, t, vars, parameters; separator = ",")
  local buffer = IOBuffer()
  print(buffer, "@variables t;")
  print(buffer, "vars2 = @variables")
  print(buffer, "(")
  for v in vars
    print(buffer, v, separator)
  end
  print(buffer, ");")
  print(buffer, "eqs2 =")
  print(buffer, "[")
  for eq in deqs
    print(buffer, replace(string(eq), "(t)" => "", "Differential" => "D"), separator)
  end
  print(buffer, "];")
  return String(take!(buffer))
end

"""
  Check if a symbol is a registered dynamic Modelica function.
  Uses the global DYNAMIC_MODELICA_FUNCTIONS registry populated when functions are eval'd.
"""
function isDynamicModelicaFunction(sym::Symbol)
  return sym in DYNAMIC_MODELICA_FUNCTIONS
end

"""
  Wrap function calls to dynamically generated Modelica functions with Base.invokelatest
  to avoid world-age issues.
"""
function wrapWithInvokelatest(expr::Expr)
  if expr.head == :call
    func = expr.args[1]
    #= Check if the function is a qualified call to OMBackend.CodeGeneration =#
    if func isa Expr && func.head == :.
      funcStr = string(func)
      if startswith(funcStr, "OMBackend.CodeGeneration.")
        #= Wrap with Base.invokelatest =#
        local newArgs = Any[:(Base.invokelatest), func]
        for a in expr.args[2:end]
          push!(newArgs, wrapWithInvokelatest(a))
        end
        return Expr(:call, newArgs...)
      end
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
  elseif expr.head in (:(=), :block, :if, :elseif, :||, :&&, :comparison)
    local processedArgs = Any[]
    for a in expr.args
      push!(processedArgs, wrapWithInvokelatest(a))
    end
    return Expr(expr.head, processedArgs...)
  else
    local processedArgs = Any[]
    for a in expr.args
      push!(processedArgs, wrapWithInvokelatest(a))
    end
    return Expr(expr.head, processedArgs...)
  end
end

wrapWithInvokelatest(x) = x  #= For non-Expr types, return as-is =#

"""
Fix malformed function calls where stringification produces (FuncName())(args)
instead of FuncName(args). This happens with some Modelica functions containing
if-statements when processed through MTK symbolic machinery.
"""
function fixMalformedFunctionCalls(expr::Expr)
  if expr.head == :call
    local func = expr.args[1]
    #= Check for pattern: (SomeFunc())(args) - where first arg is a zero-arg call =#
    if func isa Expr && func.head == :call && length(func.args) == 1
      #= This is a malformed double-call: func is (SomeFunc()) =#
      #= Transform to: SomeFunc(args...) =#
      local actualFunc = func.args[1]
      local newArgs = Any[actualFunc]
      for a in expr.args[2:end]
        push!(newArgs, fixMalformedFunctionCalls(a))
      end
      return Expr(:call, newArgs...)
    end
  end
  #= Recursively process all arguments for relevant expression types =#
  if expr.head in (:call, :block, :if, :elseif, :(=), :||, :&&, :comparison)
    local processedArgs = Any[]
    for a in expr.args
      push!(processedArgs, fixMalformedFunctionCalls(a))
    end
    return Expr(expr.head, processedArgs...)
  end
  return expr
end

fixMalformedFunctionCalls(x) = x  #= For non-Expr types, return as-is =#

rewriteEq(eq) = begin
  local eqStr = string(eq)
  #= Remove typed array constructors - they cause conversion errors at runtime =#
  eqStr = replace(eqStr, r"SymbolicUtils\.BasicSymbolic\{Real\}\[" => "[")
  #= Fix gensym-style function names: var"#FuncName" -> FuncName =#
  #= MTK stringifies registered functions with # prefix that needs removal =#
  eqStr = replace(eqStr, r"var\"#([^\"]+)\"" => s"\1")
  res = Meta.parse(replace(eqStr, "Differential(t)" => "D"))
  #= Fix malformed function calls: (FuncName())(args) -> FuncName(args) =#
  res = fixMalformedFunctionCalls(res)
  #= Wrap dynamically generated function calls with Base.invokelatest =#
  #= This is needed because the wrapper functions are defined via @eval at runtime, =#
  #= creating them in a newer world age than the simulation code. =#
  res = wrapWithInvokelatest(res)
  return res
end

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
# function structural_simplify(sys::ModelingToolkit.AbstractSystem, io = nothing; simplify = false, kwargs...)
#   @info "Calling custom structural_simplify"
#   sys = expand_connections(sys)
#   state = TearingState(sys)
#   has_io = io !== nothing
#   has_io && markio!(state, io...)
#   state, input_idxs = ModelingToolkit.inputs_to_parameters!(state, io)
#   sys, ag = ModelingToolkit.alias_elimination!(state; kwargs...)
#   check_consistency(state, ag)
#   sys = dummy_derivative(sys, state, ag; simplify)
#   fullstates = [map(eq -> eq.lhs, observed(sys)); states(sys)]
#   @set! sys.observed = ModelingToolkit.topsort_equations(observed(sys), fullstates)
#   ModelingToolkit.invalidate_cache!(sys)
#   return has_io ? (sys, input_idxs) : sys
# end



# function structural_simplify(sys::ModelingToolkit.AbstractSystem, io = nothing; simplify = false, kwargs...)
#   @info "Calling custom structural_simplify"
#   #sys = ModelingToolkit.ode_order_lowering(sys)
#   #sys = ModelingToolkit.dae_index_lowering(sys)
#   #sys = ModelingToolkit.tearing(sys; simplify = simplify)
#   sys = ModelingToolkit.structural_simplify(sys, simplify = simplify)
#   return sys
# end

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
