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
    _solveDAEInitialization!(u0, rhsFunc, p_vec, mm; maxiter=200, tol=1e-10)

Solve the algebraic constraints of a DAE system to find consistent initial conditions.
Uses a two-phase pseudoinverse Newton approach:

Phase 1: Fix differential states, solve only for algebraic unknowns. This preserves
the initial configuration (e.g. pendulum angles) while computing consistent
algebraic values (positions, forces). Works for most DAE systems.

Phase 2 (fallback): If phase 1 fails to converge, free ALL unknowns including
differential states. This handles rank-deficient kinematic loops (e.g. Engine1a)
where the differential state values are constrained by algebraic equations.
"""
function _solveDAEInitialization!(u0, rhsFunc, p_vec, mm; maxiter=200, tol=1e-10, failure_threshold=1e-1)
  local n = length(u0)
  #= Defensive clamp: the mass matrix may be smaller than u0 if MTK's
     structural_simplify produced an imbalanced reduced system (more unknowns
     than full_equations). buildDirectRHSProblem detects this case and errors
     out earlier, but we guard the loop bounds here as well. =#
  local nMM = size(mm, 1)
  local nSafe = min(n, nMM)
  local alg_idx = [i for i in 1:nSafe if mm[i,i] == 0]
  local diff_idx = [i for i in 1:nSafe if mm[i,i] != 0]
  if isempty(alg_idx)
    return u0
  end
  local du = similar(u0)
  rhsFunc(du, u0, p_vec, 0.0)
  local init_res = maximum(abs, du[alg_idx])
  if init_res < tol
    @debug "DirectRHS init: algebraic residuals already consistent (max=$init_res)"
    return u0
  end
  # Phase 1: Fix differential states, solve algebraic equations for algebraic unknowns only.
  local u0_phase1 = copy(u0)
  local phase1_ok = _solveDAEPhase!(u0_phase1, rhsFunc, p_vec, alg_idx, alg_idx,
                                     maxiter=min(maxiter, 50), tol=tol)
  if phase1_ok
    copyto!(u0, u0_phase1)
    return u0
  end
  # Phase 2: Free all unknowns (handles rank-deficient kinematic loops).
  @debug "DirectRHS init: phase 1 failed, trying phase 2 with all unknowns free"
  local all_idx = collect(1:n)
  local phase2_ok = _solveDAEPhase!(u0, rhsFunc, p_vec, alg_idx, all_idx,
                                     maxiter=maxiter, tol=tol)
  if !phase2_ok
    rhsFunc(du, u0, p_vec, 0.0)
    local final_res = maximum(abs, du[alg_idx])
    #= NaN/Inf means the residual function broke down (division by zero, sqrt of
       negative, etc.), not merely a large residual. `NaN >= failure_threshold` is
       `false`, so without an explicit isfinite check NaN silently routes into the
       "below threshold" warn branch. Surface it as @error so it is distinguishable
       from ordinary non-convergence; still return u0 so the integrator retains its
       chance to recover. =#
    if !isfinite(final_res)
      @error "DirectRHS init: residual is non-finite (value: $(final_res)); initial conditions could not be verified and the integrator may produce NaNs. Proceeding, but consider adding explicit start values."
    elseif final_res >= failure_threshold
      #= Residual is too large to proceed safely. Handing the solver inconsistent
         initial conditions produces silent numerical garbage. Escalate rather than
         warn. Callers that need a higher tolerance can pass `failure_threshold`. =#
      error("DirectRHS init: residual $(round(final_res, sigdigits=4)) exceeds failure_threshold=$(failure_threshold); refusing to proceed with inconsistent initial conditions.")
    else
      @warn "DirectRHS init: did not fully converge (residual: $(round(final_res, sigdigits=4))) but is below failure_threshold=$(failure_threshold); proceeding."
    end
  end
  return u0
end

"""
    _solveDAEPhase!(u0, rhsFunc, p_vec, eq_idx, var_idx; maxiter, tol)

Single-phase Newton solve: minimize residuals of equations `eq_idx` by adjusting
unknowns `var_idx`. Uses pseudoinverse for rank-deficient/underdetermined systems.
Returns true if converged.
"""
function _solveDAEPhase!(u0, rhsFunc, p_vec, eq_idx, var_idx; maxiter=50, tol=1e-10)
  local nEq = length(eq_idx)
  local nVar = length(var_idx)
  local du = similar(u0)
  local eps_fd = 1e-7
  for iter in 1:maxiter
    rhsFunc(du, u0, p_vec, 0.0)
    local res = du[eq_idx]
    local norm_res = maximum(abs, res)
    if !isfinite(norm_res)
      @debug "DirectRHS init: non-finite residual at iteration $iter, aborting"
      return false
    end
    if norm_res < tol
      @debug "DirectRHS init: converged in $iter iterations (residual: $norm_res)"
      return true
    end
    # Finite-difference Jacobian
    local J = zeros(nEq, nVar)
    local du_pert = similar(u0)
    for (jcol, jstate) in enumerate(var_idx)
      local u_pert = copy(u0)
      u_pert[jstate] += eps_fd
      rhsFunc(du_pert, u_pert, p_vec, 0.0)
      J[:, jcol] = (du_pert[eq_idx] .- res) ./ eps_fd
    end
    if any(!isfinite, J)
      @debug "DirectRHS init: non-finite Jacobian at iteration $iter, aborting"
      return false
    end
    local delta = LinearAlgebra.pinv(J) * res
    local alpha = min(1.0, 10.0 / max(1.0, LinearAlgebra.norm(delta)))
    for (jcol, jstate) in enumerate(var_idx)
      u0[jstate] -= alpha * delta[jcol]
    end
  end
  return false
end


"""
    buildDirectRHSProblem(reducedSystem, finalInitialValues, pars, tspan, callbacks;
                          allInitialValues=nothing)

Build an ODEProblem by extracting the RHS function directly from the reduced MTK
system's symbolic equations, bypassing MTK's ODEProblem constructor.

Uses `Symbolics.build_function` with CSE to generate compact index-based code,
wrapped in a `RuntimeGeneratedFunction` for world-age safety.

`allInitialValues` provides Modelica start values for algebraic variables that
`splitInitialValues` demoted to guesses. Without these, algebraic variables
default to 0.0, which may cause InitialFailure for DAE systems.

Returns an `ODEProblem` ready for `solve()`.
"""
function buildDirectRHSProblem(reducedSystem, finalInitialValues, pars, tspan, callbacks;
                               allInitialValues=nothing)  # allInitialValues kept for API compat but guesses from reducedSystem are preferred
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
  local nEqs = length(eqs)

  @debug "DirectRHS: $(nStates) states, $(nParams) params, $(nEqs) equations"

  #= Reject structurally imbalanced systems early. MTK's structural_simplify
     occasionally leaves reduced systems where full_equations(sys) != unknowns(sys)
     (observed in Rotational.Friction: 35 full_equations, 36 unknowns). Attempting
     to build an RHS from such a system produces nonsense results and eventually
     crashes with a BoundsError in the mass-matrix DAE initialization path. Fail
     here with a clear diagnostic instead. =#
  if nStates != 0 && nEqs != nStates
    error("DirectRHS: structural imbalance in reduced system: " *
          "$(nEqs) full_equations vs $(nStates) unknowns. " *
          "The model cannot be integrated as a well-posed DAE; " *
          "this is typically a residual issue from MTK structural_simplify.")
  end

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

  # 3. Build u0 and parameter vectors in the correct ordering.
  #    Resolve parameter values first so _buildStateVector can substitute
  #    symbolic parameter references in initial conditions.
  local resolvedParams = _resolveParamValues(pars)
  # Extract guesses from the reduced system. These are properly mapped to
  # post-simplification unknowns and provide Modelica start values for variables
  # that splitInitialValues could not map (pre-simplification names do not match).
  local systemGuesses = try
    ModelingToolkit.guesses(reducedSystem)
  catch
    Dict()
  end
  local u0 = _buildStateVector(states, finalInitialValues; resolvedParams=resolvedParams,
                                systemGuesses=systemGuesses)
  local p_vec = _buildParamVector(params, pars; resolvedParams=resolvedParams)

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
    # Solve algebraic constraints to find consistent initial conditions.
    # Two-phase approach: first tries algebraic-only solve (fixing differential
    # states), then falls back to full solve if needed (for kinematic loops).
    u0 = _solveDAEInitialization!(u0, rhsFunc, p_vec, mm)
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
function _buildStateVector(states, finalInitialValues;
                           resolvedParams::Union{Dict{String,Float64},Nothing}=nothing,
                           systemGuesses=nothing)
  local nStates = length(states)
  local u0 = zeros(Float64, nStates)
  local stateStrToIdx = Dict{String, Int}()
  for (i, s) in enumerate(states)
    stateStrToIdx[string(s)] = i
  end
  local matchedSet = Set{String}()
  for pair in finalInitialValues
    local keyStr = string(pair.first)
    if haskey(stateStrToIdx, keyStr)
      u0[stateStrToIdx[keyStr]] = _toFloat64(pair.second; resolvedParams=resolvedParams)
      push!(matchedSet, keyStr)
    end
  end
  # Fill unmatched states from system guesses (post-simplification variable space).
  # These provide Modelica start values for algebraic variables whose pre-simplification
  # names did not match the reduced system unknowns.
  local guessMatched = 0
  if systemGuesses !== nothing && !isempty(systemGuesses)
    for (gk, gv) in systemGuesses
      local keyStr = string(gk)
      if haskey(stateStrToIdx, keyStr) && !(keyStr in matchedSet)
        local val = _toFloat64(gv; resolvedParams=resolvedParams)
        u0[stateStrToIdx[keyStr]] = val
        push!(matchedSet, keyStr)
        guessMatched += 1
      end
    end
  end
  @debug "DirectRHS: matched $(length(matchedSet))/$(nStates) states ($(length(matchedSet) - guessMatched) hard, $(guessMatched) from guesses)"
  return u0
end


"""
    _buildParamVector(params, pars)

Build the parameter vector, ordered to match `parameters(reducedSystem)`.
Resolves parameter-to-parameter dependencies by iteratively substituting
known numeric values until all parameters are numeric (or until convergence).
"""
function _buildParamVector(params, pars; resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local nParams = length(params)
  local p_vec = zeros(Float64, nParams)

  # Resolve parameter values by iterative substitution (reuse if already done)
  if resolvedParams === nothing
    resolvedParams = _resolveParamValues(pars)
  end

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

  # Build a name-based lookup for substitution fallback
  local nameToNumeric = Dict{String, Float64}()
  for (kStr, fval) in numericByStr
    nameToNumeric[kStr] = fval
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
      # Unwrap Num if needed before checking for numeric
      local unwrapped = resolved isa Symbolics.Num ? Symbolics.unwrap(resolved) : resolved
      if unwrapped isa Number
        local fval = Float64(unwrapped)
        numericByStr[kStr] = fval
        nameToNumeric[kStr] = fval
        subDict[uk] = fval
        newlyResolved += 1
      else
        # Fallback: try name-based substitution by matching free variable names
        # to known numeric parameters. This handles cases where the symbolic
        # objects in the expression have different identity than the parameter keys.
        local nameDict = Dict{Any, Any}()
        local freeVars = try
          Symbolics.get_variables(uv)
        catch
          Any[]
        end
        for fv in freeVars
          local fvName = string(fv)
          if haskey(nameToNumeric, fvName)
            nameDict[fv] = nameToNumeric[fvName]
          end
        end
        if !isempty(nameDict)
          local resolved2 = try
            Symbolics.substitute(uv, nameDict)
          catch
            uv
          end
          # Unwrap Num if needed, then check for numeric result
          local unwrapped2 = resolved2 isa Symbolics.Num ? Symbolics.unwrap(resolved2) : resolved2
          if unwrapped2 isa Number
            local fval2 = Float64(unwrapped2)
            numericByStr[kStr] = fval2
            nameToNumeric[kStr] = fval2
            subDict[uk] = fval2
            newlyResolved += 1
            continue
          end
          # Last resort: try Symbolics.value on the substituted result
          local numVal = try
            Float64(Symbolics.value(resolved2))
          catch
            nothing
          end
          if numVal !== nothing && isfinite(numVal)
            numericByStr[kStr] = numVal
            nameToNumeric[kStr] = numVal
            subDict[uk] = numVal
            newlyResolved += 1
            continue
          end
        end
        # Final fallback: evaluate expression string with known numeric bindings
        local evalResult = try
          local evalExpr = Meta.parse(string(uv))
          local evalModule = Module()
          for (n, v) in nameToNumeric
            local sym = Symbol(n)
            Core.eval(evalModule, :($sym = $v))
          end
          Float64(Core.eval(evalModule, evalExpr))
        catch
          nothing
        end
        if evalResult !== nothing && isfinite(evalResult)
          numericByStr[kStr] = evalResult
          nameToNumeric[kStr] = evalResult
          subDict[uk] = evalResult
          newlyResolved += 1
          continue
        end
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
    local unresolvedNames = [kStr for (_, _, kStr) in symbolicByKey]
    local unresolvedVals = [string(uv) for (_, uv, _) in symbolicByKey]
    @warn "DirectRHS: $(length(symbolicByKey)) parameters could not be resolved to numeric values, defaulting to 0.0" unresolvedNames unresolvedVals
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
    _toFloat64(val; resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)

Convert a value to Float64, handling Symbolics.Num wrappers, constant symbolic
expressions, and parameter references (resolved via resolvedParams dict).
"""
function _toFloat64(val; resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local unwrapped = val isa Symbolics.Num ? Symbolics.unwrap(val) : val
  unwrapped isa Number && return Float64(unwrapped)
  # Constant symbolic expression (no free variables): parse its string repr
  local freeVars = Symbolics.get_variables(unwrapped)
  if isempty(freeVars)
    local f = tryparse(Float64, string(val))
    f !== nothing && return f
  end
  if resolvedParams !== nothing
    # Direct name lookup (handles bare parameter references)
    local key = string(val)
    haskey(resolvedParams, key) && return resolvedParams[key]
    # Substitute known parameters into the expression
    if !isempty(freeVars)
      local subDict = Dict{Any,Any}(fv => resolvedParams[string(fv)]
                                     for fv in freeVars
                                     if haskey(resolvedParams, string(fv)))
      if !isempty(subDict)
        local resolved = Symbolics.substitute(unwrapped, subDict)
        resolved isa Number && return Float64(resolved)
        local rv = resolved isa Symbolics.Num ? Symbolics.unwrap(resolved) : resolved
        rv isa Number && return Float64(rv)
      end
    end
  end
  @warn "DirectRHS: could not convert value to Float64, using 0.0" val=val type=typeof(val)
  return 0.0
end
