#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
* c/o Linkoepings universitet, Department of Computer and Information Science,
* SE-58183 Linkoeping, Sweden.
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
  Direct RHS Generation for OM.jl

  Bypasses MTK's ODEProblem constructor by extracting the RHS function
  directly from the reduced system's symbolic equations. Uses
  Symbolics.build_function with CSE to generate compact index-based code
  wrapped in a RuntimeGeneratedFunction for world-age safety.

  This avoids the expensive LLVM compilation of deeply nested symbolic
  expressions that occurs with MTK's default pipeline that occured prior, reducing compilation
  time from 35+ minutes to seconds for large models.


  Author: John Tinnerholm
=#

"""
    buildDirectRHSProblem(reducedSystem, finalInitialValues, pars, tspan, callbacks)

Build an ODEProblem by extracting the RHS function directly from the reduced MTK
system's symbolic equations, bypassing MTK's ODEProblem constructor.

Uses `Symbolics.build_function` with CSE to generate compact index-based code,
wrapped in a `RuntimeGeneratedFunction` for world-age safety.

Returns an `ODEProblem` ready for `solve()`.
"""
function buildDirectRHSProblem(reducedSystem, finalInitialValues, pars, tspan, callbacks)
  local states = ModelingToolkit.unknowns(reducedSystem)
  local params = ModelingToolkit.parameters(reducedSystem)
  # Use full_equations to inline observed variable definitions.
  # equations() may reference observed variables by name, which would appear
  # as undefined symbols in the generated RHS function. full_equations()
  # substitutes observed definitions, so only states and params remain.
  local eqs = ModelingToolkit.full_equations(reducedSystem)
  local iv = ModelingToolkit.get_iv(reducedSystem)
  local nStates = length(states)
  local nParams = length(params)

  @debug "DirectRHS: $(nStates) states, $(nParams) params, $(length(eqs)) equations"

  # Handle empty systems (0 unknowns after structural_simplify).
  # Use a dummy 1-element state so the ODE solver does not reject an empty range.
  if nStates == 0
    @debug "DirectRHS: empty system (0 unknowns), building trivial dummy problem"
    local emptyRHS = (du, u, p, t) -> (du[1] = 0.0)
    local f0 = ModelingToolkit.ODEFunction{true}(emptyRHS)
    return ModelingToolkit.ODEProblem{true}(f0, [0.0], tspan, Float64[]; callback=callbacks)
  end

  # 1. Build the RHS function expression from symbolic equations
  local rhs_list = [eq.rhs for eq in eqs]
  local f_ip_expr = _buildRHSExpression(rhs_list, states, params, iv)

  # 2. Create world-age-safe function via RuntimeGeneratedFunction
  local rhsFunc = _exprToRTGFunction(f_ip_expr)

  # 3. Build u0 and parameter vectors in the correct ordering
  local u0 = _buildStateVector(states, finalInitialValues)
  local p_vec = _buildParamVector(params, pars)

  @debug "DirectRHS: u0 has $(count(!iszero, u0))/$(nStates) nonzero, p has $(count(!iszero, p_vec))/$(nParams) nonzero"

  # 4. Extract event callbacks from the reduced system and merge with custom callbacks.
  #    Our structural_simplify wrapper uses split=false, so the compiled event
  #    callbacks expect a flat parameter vector matching our p_vec format.
  local allCallbacks = _extractAndMergeEventCallbacks(reducedSystem, callbacks)

  # 5. Construct ODEProblem, handling mass matrix for DAE systems.
  #    Attach reducedSystem via sys= so callbacks can look up state/parameter
  #    names from integrator.f.sys (used by getStatesAsSymbols/getParametersAsSymbols).
  local massMatrix = ModelingToolkit.calculate_massmatrix(reducedSystem)
  local problem
  if massMatrix isa LinearAlgebra.UniformScaling
    @debug "DirectRHS: pure ODE (identity mass matrix)"
    local f = ModelingToolkit.ODEFunction{true}(rhsFunc; sys=reducedSystem)
    problem = ModelingToolkit.ODEProblem{true}(f, u0, tspan, p_vec; callback=allCallbacks)
  else
    @debug "DirectRHS: DAE with mass matrix"
    local mm = collect(massMatrix)
    local f = ModelingToolkit.ODEFunction{true}(rhsFunc; mass_matrix=mm, sys=reducedSystem)
    problem = ModelingToolkit.ODEProblem{true}(f, u0, tspan, p_vec; callback=allCallbacks)
  end

  @debug "DirectRHS: problem constructed successfully"
  return problem
end


"""
    _buildRHSExpression(rhs_list, states, params, iv)

Build the in-place RHS function expression using `Symbolics.build_function`.
Applies CSE (Common Subexpression Elimination) when available to decompose
deeply nested expressions into flat sequential assignments, which compile
much faster through LLVM.
"""
function _buildRHSExpression(rhs_list, states, params, iv)
  # Try with CSE first for better compilation performance
  try
    local result = Symbolics.build_function(rhs_list, states, params, iv;
                                             expression=Val{true}, cse=true)
    @debug "DirectRHS: generated RHS function with CSE"
    return result[2]  # in-place form
  catch e
    @warn "DirectRHS: CSE failed, using direct generation" exception=(e, catch_backtrace())
  end
  # Fallback without CSE
  local result = Symbolics.build_function(rhs_list, states, params, iv;
                                           expression=Val{true})
  return result[2]
end


"""
    _exprToRTGFunction(f_expr)

Convert a function expression (from `Symbolics.build_function`) to a
`RuntimeGeneratedFunction` for world-age safety. This allows the generated
RHS function to be called from any world age, avoiding the world-age
issues that would occur with a plain `eval`.
"""
function _exprToRTGFunction(f_expr)
  local arrow_expr = f_expr
  # Convert :(function (args...) body end) to :((args...) -> body) if needed
  if f_expr isa Expr && f_expr.head == :function
    local args_part = f_expr.args[1]
    local body_part = f_expr.args[2]
    # Handle named function: :(fname(a, b, c))
    if args_part isa Expr && args_part.head == :call
      args_part = Expr(:tuple, args_part.args[2:end]...)
    end
    arrow_expr = Expr(:->, args_part, body_part)
  end
  return RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
    @__MODULE__, @__MODULE__, arrow_expr)
end


"""
    _buildStateVector(states, finalInitialValues)

Build the initial state vector `u0`, ordered to match `unknowns(reducedSystem)`.
Uses string comparison for key matching to avoid `Num`/`BasicSymbolic` type
mismatch issues.
"""
function _buildStateVector(states, finalInitialValues)
  local nStates = length(states)
  local u0 = zeros(Float64, nStates)
  local stateStrToIdx = Dict{String, Int}()
  for (i, s) in enumerate(states)
    stateStrToIdx[string(s)] = i
  end
  local matched = 0
  for pair in finalInitialValues
    local keyStr = string(pair.first)
    if haskey(stateStrToIdx, keyStr)
      u0[stateStrToIdx[keyStr]] = _toFloat64(pair.second)
      matched += 1
    end
  end
  @debug "DirectRHS: matched $(matched)/$(length(finalInitialValues)) initial values to $(nStates) states"
  return u0
end


"""
    _buildParamVector(params, pars)

Build the parameter vector, ordered to match `parameters(reducedSystem)`.
Resolves parameter-to-parameter dependencies by iteratively substituting
known numeric values until all parameters are numeric (or until convergence).
"""
function _buildParamVector(params, pars)
  local nParams = length(params)
  local p_vec = zeros(Float64, nParams)

  # Resolve parameter values by iterative substitution
  local resolvedParams = _resolveParamValues(pars)

  local matched = 0
  for (i, p) in enumerate(params)
    local pStr = string(p)
    if haskey(resolvedParams, pStr)
      p_vec[i] = resolvedParams[pStr]
      matched += 1
    end
  end
  @debug "DirectRHS: resolved $(matched)/$(nParams) parameters to numeric values"
  return p_vec
end


"""
    _resolveParamValues(pars)

Resolve parameter values by iteratively substituting known numeric values
into symbolic parameter expressions. Returns a Dict{String, Float64}
mapping parameter names to their numeric values.
"""
function _resolveParamValues(pars)
  # Separate numeric and symbolic parameter values
  local numericByStr = Dict{String, Float64}()
  local symbolicByKey = Vector{Tuple{Any, Any, String}}()  # (unwrapped_key, unwrapped_val, str_key)

  for (k, v) in pars
    local kStr = string(k)
    local uv = v isa Symbolics.Num ? Symbolics.unwrap(v) : v
    if uv isa Number
      numericByStr[kStr] = Float64(uv)
    else
      local uk = k isa Symbolics.Num ? Symbolics.unwrap(k) : k
      push!(symbolicByKey, (uk, uv, kStr))
    end
  end

  @debug "DirectRHS: $(length(numericByStr)) numeric params, $(length(symbolicByKey)) symbolic params to resolve"

  # Build a substitution dict from numeric values (using unwrapped symbolic keys)
  local subDict = Dict{Any, Any}()
  for (k, v) in pars
    local kStr = string(k)
    if haskey(numericByStr, kStr)
      local uk = k isa Symbolics.Num ? Symbolics.unwrap(k) : k
      subDict[uk] = numericByStr[kStr]
    end
  end

  # Iteratively resolve symbolic parameters
  for iteration in 1:10
    local newlyResolved = 0
    local remaining = Vector{Tuple{Any, Any, String}}()
    for (uk, uv, kStr) in symbolicByKey
      local resolved = try
        Symbolics.substitute(uv, subDict)
      catch
        uv  # substitution failed, keep original
      end
      if resolved isa Number
        local fval = Float64(resolved)
        numericByStr[kStr] = fval
        subDict[uk] = fval
        newlyResolved += 1
      else
        push!(remaining, (uk, uv, kStr))
      end
    end
    symbolicByKey = remaining
    if newlyResolved == 0
      break
    end
    @debug "DirectRHS: resolved $(newlyResolved) more params in iteration $(iteration) ($(length(remaining)) remaining)"
  end

  if !isempty(symbolicByKey)
    @warn "DirectRHS: $(length(symbolicByKey)) parameters could not be resolved to numeric values, defaulting to 0.0"
    for (_, _, kStr) in symbolicByKey
      numericByStr[kStr] = 0.0
    end
  end

  return numericByStr
end


"""
    _extractAndMergeEventCallbacks(reducedSystem, customCallbacks)

Extract continuous and discrete event callbacks from the reduced MTK system
and merge them with custom callbacks (e.g. VSS structural change callbacks).

Requires the reduced system to have been compiled with `split=false` so that
the generated event callback functions expect a flat parameter vector.
"""
function _extractAndMergeEventCallbacks(reducedSystem, customCallbacks)
  local eventCBs = nothing
  try
    eventCBs = ModelingToolkit.process_events(reducedSystem; callback=customCallbacks)
  catch ex
    @warn "DirectRHS: failed to extract event callbacks, using custom callbacks only" exception=(ex, catch_backtrace())
    return customCallbacks
  end
  if eventCBs === nothing
    @debug "DirectRHS: no events in reduced system"
    return customCallbacks
  end
  @debug "DirectRHS: extracted event callbacks from reduced system"
  return eventCBs
end


"""
    _toFloat64(val)

Convert a value to Float64, handling Symbolics.Num wrappers and other numeric types.
"""
function _toFloat64(val)
  local unwrapped = val isa Symbolics.Num ? Symbolics.unwrap(val) : val
  if unwrapped isa Number
    return Float64(unwrapped)
  end
  @warn "DirectRHS: could not convert value to Float64, using 0.0" val=val type=typeof(val)
  return 0.0
end
