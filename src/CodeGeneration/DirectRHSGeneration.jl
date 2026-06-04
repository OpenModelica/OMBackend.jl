#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
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
  expressions that occur with MTK's default pipeline that occurred prior, reducing compilation
  time from 35+ minutes to seconds for large models.


  Author: John Tinnerholm
=#

# MTK-stage dump helpers (see CodeGeneration/mtkDump.jl).
import .MTKDump: dumpBuildDirectRHSInputs, dumpRHSExpression

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

  # Dump every Symbol/Num touching the post-MTK boundary. See MTKDump for
  # rationale and format.
  dumpBuildDirectRHSInputs(states, params, eqs, finalInitialValues, pars, reducedSystem, callbacks)

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

  # Dump the actual generated RHS expression — see MTKDump.
  dumpRHSExpression(rhs_list, f_ip_expr)

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
  local hardInitialValues = _collectHardInitializationValues(
    reducedSystem, finalInitialValues; resolvedParams=resolvedParams)
  local observedEquations = try
    ModelingToolkit.observed(reducedSystem)
  catch
    Symbolics.Equation[]
  end
  local u0 = _buildStateVector(states, finalInitialValues; resolvedParams=resolvedParams,
                                systemGuesses=systemGuesses,
                                hardInitialValues=hardInitialValues,
                                observedEquations=observedEquations)
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
    #= Pinned indices: vars whose u0 came from a fixed=true Modelica init eq
       (after splitInitialValues). The DAE init solver must NOT modify these,
       otherwise an algebraic var pinned by `start=1, fixed=true` (e.g.
       sd1.s_rel = 1) gets overwritten by the alg residual that depends on
       free vars (e.g. m1.s, m2.s) — collapsing to a different consistent
       root than the user requested. =#
    local pinnedKeyStrSet = OrderedSet(string(p.first) for p in finalInitialValues)
    local pinnedIdx = Int[i for (i, st) in enumerate(states)
                          if string(st) in pinnedKeyStrSet]
    local derivativeInitTargets = _derivativeInitializationTargets(
      reducedSystem, states; resolvedParams=resolvedParams)
    u0 = _solveDAEInitialization!(u0, rhsFunc, p_vec, mm;
                                  pinned=pinnedIdx,
                                  derivative_targets=derivativeInitTargets)
    problem = ModelingToolkit.ODEProblem{true}(f, u0, tspan, p_vec; callback=allCallbacks)
  end

  @debug "DirectRHS: problem constructed successfully"
  return problem
end


function _collectHardInitializationValues(reducedSystem, finalInitialValues;
                                          resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local values = Dict{Any, Float64}()
  for pair in finalInitialValues
    values[pair.first] = _toFloat64(pair.second; resolvedParams=resolvedParams)
  end
  local initEqs = try
    ModelingToolkit.initialization_equations(reducedSystem)
  catch
    return values
  end
  for eq in initEqs
    startswith(string(eq.lhs), "Differential(") && continue
    local rhsVal = _literalNumericValue(eq.rhs)
    rhsVal === nothing && continue
    values[eq.lhs] = rhsVal
  end
  return values
end


function _literalNumericValue(val)
  local raw = val
  raw = raw isa Symbolics.Num ? Symbolics.unwrap(raw) : raw
  raw isa Number && return Float64(raw)
  raw = try
    Symbolics.value(raw)
  catch
    return nothing
  end
  raw = raw isa Symbolics.Num ? Symbolics.unwrap(raw) : raw
  return raw isa Number ? Float64(raw) : nothing
end


function _derivativeInitializationTargets(reducedSystem, states;
                                          resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local stateStrToIdx = Dict{String, Int}(string(st) => i for (i, st) in enumerate(states))
  local targets = Pair{Int, Float64}[]
  local initEqs = try
    ModelingToolkit.initialization_equations(reducedSystem)
  catch
    return targets
  end
  for eq in initEqs
    local lhsStr = string(eq.lhs)
    startswith(lhsStr, "Differential(") || continue
    local matchedIdx = nothing
    for (stateStr, idx) in stateStrToIdx
      if endswith(lhsStr, "(" * stateStr * ")")
        matchedIdx = idx
        break
      end
    end
    matchedIdx === nothing && continue
    local target = try
      _toFloat64(eq.rhs; resolvedParams=resolvedParams)
    catch
      continue
    end
    push!(targets, matchedIdx => target)
  end
  return targets
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
                           systemGuesses=nothing,
                           hardInitialValues=nothing,
                           observedEquations=nothing)
  local nStates = length(states)
  local u0 = zeros(Float64, nStates)
  local stateStrToIdx = Dict{String, Int}()
  for (i, s) in enumerate(states)
    stateStrToIdx[string(s)] = i
  end
  local matchedSet = OrderedSet{String}()
  local hardValueMap = Dict{Any, Float64}()
  for pair in finalInitialValues
    local keyStr = string(pair.first)
    local val = _toFloat64(pair.second; resolvedParams=resolvedParams)
    hardValueMap[pair.first] = val
    if haskey(stateStrToIdx, keyStr)
      u0[stateStrToIdx[keyStr]] = val
      push!(matchedSet, keyStr)
    end
  end
  if hardInitialValues !== nothing
    for (key, val) in hardInitialValues
      hardValueMap[key] = val
      local keyStr = string(key)
      if haskey(stateStrToIdx, keyStr) && !(keyStr in matchedSet)
        u0[stateStrToIdx[keyStr]] = val
        push!(matchedSet, keyStr)
      end
    end
  end
  local aliasMatched = 0
  if observedEquations !== nothing && !isempty(observedEquations)
    aliasMatched = _propagateObservedAliasInitialValues!(
      u0, states, matchedSet, hardValueMap, observedEquations)
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
  @debug "DirectRHS: matched $(length(matchedSet))/$(nStates) states ($(length(matchedSet) - guessMatched) hard, $(aliasMatched) via observed aliases, $(guessMatched) from guesses)"
  return u0
end


function _propagateObservedAliasInitialValues!(u0, states, matchedSet::OrderedSet{String},
                                               hardValueMap::Dict{Any, Float64},
                                               observedEquations)
  local aliasMatched = 0
  local progressed = true
  while progressed
    progressed = false
    for (i, st) in enumerate(states)
      local stStr = string(st)
      stStr in matchedSet && continue
      local resolved = _resolveObservedAffineInitialValue(st, observedEquations, hardValueMap)
      resolved === nothing && continue
      u0[i] = resolved
      hardValueMap[st] = resolved
      push!(matchedSet, stStr)
      aliasMatched += 1
      progressed = true
    end
  end
  return aliasMatched
end


function _resolveObservedAffineInitialValue(target, observedEquations, hardValueMap::Dict{Any, Float64})
  local targetStr = string(target)
  for eq in observedEquations
    contains(string(eq), targetStr) || continue
    local knownValues = Dict{Any, Any}(k => v for (k, v) in hardValueMap
                                      if string(k) != targetStr)
    local resolved = _resolveAffineInitialValue(target, eq, knownValues)
    resolved === nothing || return resolved
  end
  return nothing
end


function _resolveAffineInitialValue(target, eq, knownValues::Dict)
  local expr = Symbolics.substitute(eq.lhs - eq.rhs, knownValues)
  local y0 = _substituteTargetNumeric(expr, target, 0.0)
  y0 === nothing && return nothing
  local y1 = _substituteTargetNumeric(expr, target, 1.0)
  y1 === nothing && return nothing
  local y2 = _substituteTargetNumeric(expr, target, 2.0)
  y2 === nothing && return nothing
  local slope1 = y1 - y0
  local slope2 = y2 - y1
  iszero(slope1) && return nothing
  isapprox(slope1, slope2; atol=1e-8, rtol=1e-8) || return nothing
  local value = -y0 / slope1
  return isfinite(value) ? Float64(value) : nothing
end


function _substituteTargetNumeric(expr, target, value::Float64)
  local substituted = Symbolics.substitute(expr, Dict(target => value))
  return _literalNumericValue(substituted)
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
    _evalSymbolicFunctionCall(expr, nameToNumeric)

Try to numerically evaluate a Symbolics call expression (e.g. a registered
Modelica function applied to parameter symbols) by walking down to leaf
arguments, substituting each leaf with its known numeric value from
`nameToNumeric`, and invoking the Julia function held by
`SymbolicUtils.operation(expr)` via `Base.invokelatest`.

Returns `Float64` on success or `nothing` if any leaf is unresolved or
the evaluation throws. The recursive design lets the resolver handle
nested calls like `arrayCtor(scaleFun(p1), p2)` without needing the
symbol-name lookup to find each function.

Tuple-returning Modelica functions show up as registered scalar wrappers
that return a `Tuple`; in that case we cannot map a single Float64 back,
so callers must treat `nothing` here as "not resolvable through this
path" and fall back to the next strategy.
"""
function _evalSymbolicFunctionCall(expr, nameToNumeric::Dict{String, Float64})
  if expr isa Number
    return Float64(expr)
  end
  if expr isa Symbolics.Num
    expr = Symbolics.unwrap(expr)
  end
  if !(expr isa SymbolicUtils.BasicSymbolic)
    return nothing
  end
  #= Symbolic numeric Const (e.g. literal 500.0 or pre-folded 0.0015) appears
     as a non-call, non-sym BasicSymbolic with a `Float64`/`Int` `symtype`. Pull
     the value out via Symbolics.value before falling through to the name-based
     leaf lookup, otherwise we treat literals as unknown free vars. =#
  if !SymbolicUtils.iscall(expr) && !SymbolicUtils.issym(expr)
    local v = try; Symbolics.value(expr); catch; nothing; end
    if v isa Number
      return Float64(v)
    end
  end
  if SymbolicUtils.iscall(expr)
    local f = SymbolicUtils.operation(expr)
    local rawArgs = SymbolicUtils.arguments(expr)
    local numArgs = Vector{Float64}(undef, length(rawArgs))
    for (i, a) in enumerate(rawArgs)
      local av = _evalSymbolicFunctionCall(a, nameToNumeric)
      av === nothing && return nothing
      numArgs[i] = av
    end
    local result = try
      Base.invokelatest(f, numArgs...)
    catch
      return nothing
    end
    if result isa Number
      return Float64(result)
    end
    return nothing
  end
  #= Leaf symbolic (free variable): look up by string name. =#
  local nm = string(expr)
  if haskey(nameToNumeric, nm)
    return nameToNumeric[nm]
  end
  return nothing
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
          # Pre-evaluate Modelica function calls whose args are all numeric.
          # The symbolic operation reference (`SymbolicUtils.operation`)
          # holds the registered Julia function, so we can invoke it
          # directly without resolving the function name through a fresh
          # Module's namespace. `invokelatest` covers the case where the
          # function was registered after this call site was compiled.
          local fnVal = try
            _evalSymbolicFunctionCall(unwrapped2, nameToNumeric)
          catch
            nothing
          end
          if fnVal !== nothing && isfinite(fnVal)
            numericByStr[kStr] = fnVal
            nameToNumeric[kStr] = fnVal
            subDict[uk] = fnVal
            newlyResolved += 1
            continue
          end
        end
        # Final fallback: evaluate expression string with known numeric bindings.
        # `invokelatest` lets us call freshly-registered functions defined in
        # later world ages without tripping `MethodError ... in world age`.
        local evalResult = try
          local evalExpr = Meta.parse(string(uv))
          local evalModule = Module()
          for (n, v) in nameToNumeric
            local sym = Symbol(n)
            Base.invokelatest(Core.eval, evalModule, :($sym = $v))
          end
          Float64(Base.invokelatest(Core.eval, evalModule, evalExpr))
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
        local vextract = try; Symbolics.value(rv); catch; nothing; end
        vextract isa Number && return Float64(vextract)
      end
    end
  end
  @warn "DirectRHS: could not convert value to Float64, using 0.0" val=val type=typeof(val)
  return 0.0
end
