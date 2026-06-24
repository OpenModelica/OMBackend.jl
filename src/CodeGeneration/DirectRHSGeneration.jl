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
    _buildDirectODEFunction(rhsFunc, u0, p_vec, t0; mass_matrix, sys, jacFunc, jacProto)

Build the `ODEFunction` for the direct-RHS problem. When
`OMBackend.DIRECT_RHS_TYPE_ERASE` is set the runtime-generated RHS (and the
symbolic Jacobian) are wrapped in `FunctionWrappers` so the resulting problem
type is constant across models; the solver then compiles its stepping / Newton
/ linear-solve machinery once instead of once per distinct model. `u0`, `p_vec`
and `t0` supply only the argument *types* the wrappers specialize on; their
values are immaterial. The mass matrix, Jacobian and `sys` are attached to the
function we build, so type erasure does not drop them.
"""
function _buildDirectODEFunction(rhsFunc, u0, p_vec, t0;
                                 mass_matrix=nothing, sys=nothing,
                                 jacFunc=nothing, jacProto=nothing)
  if !OMBackend.DIRECT_RHS_TYPE_ERASE[]
    local jacKw = jacFunc === nothing ? NamedTuple() : (; jac=jacFunc, jac_prototype=jacProto)
    return mass_matrix === nothing ?
      ModelingToolkit.ODEFunction{true}(rhsFunc; sys=sys, jacKw...) :
      ModelingToolkit.ODEFunction{true}(rhsFunc; mass_matrix=mass_matrix, sys=sys, jacKw...)
  end
  local FW = ModelingToolkit.SciMLBase.FunctionWrapperSpecialize
  #= Multi-variant wrapper (Float64 + ForwardDiff Dual signatures) so autodiff
     solvers stay correct; the Jacobian is never called with Duals, so a single
     variant suffices there. =#
  local wrappedRHS = DiffEqBase.wrapfun_iip(rhsFunc, (u0, u0, p_vec, t0))
  local erasedKw = jacFunc === nothing ? NamedTuple() :
    (; jac = DiffEqBase.wrapfun_jac_iip(jacFunc, (jacProto, u0, p_vec, t0)),
       jac_prototype = jacProto)
  return mass_matrix === nothing ?
    ModelingToolkit.ODEFunction{true, FW}(wrappedRHS; sys=sys, erasedKw...) :
    ModelingToolkit.ODEFunction{true, FW}(wrappedRHS; mass_matrix=mass_matrix, sys=sys, erasedKw...)
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
                               allInitialValues=nothing,  # kept for API compat but guesses from reducedSystem are preferred
                               preMem=nothing)
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
  local (hardInitialValues, initEqPinKeys) = _collectHardInitializationValues(
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

  #= Symbolic sparse Jacobian; nothing when not differentiable. Built after
     u0/p_vec so the generated function can be probed once: an unresolved
     symbolic derivative surfaces only when the function runs, not at build. =#
  local (jacFunc, jacProto) = _buildSparseJacobian(rhs_list, states, params, iv,
                                                   u0, p_vec, tspan[1])

  # 4. Extract event callbacks from the reduced system and merge with custom callbacks.
  #    Our structural_simplify wrapper uses split=false, so the compiled event
  #    callbacks expect a flat parameter vector matching our p_vec format.
  local allCallbacks = _extractAndMergeEventCallbacks(reducedSystem, callbacks)
  #= The callback is stored UN-collapsed; the erasure happens at SOLVE time
     (simulateIMTK), in a settled world, so the FunctionWrappers it builds dispatch
     correctly to RGF-backed MTK callbacks (build-time collapse captured a stale world
     -> wrong events). The solve-time collapse also yields the model-independent erased
     type whose `solve` is baked into the image. =#

  # 5. Construct ODEProblem, handling mass matrix for DAE systems.
  #    Attach reducedSystem via sys= so callbacks can look up state/parameter
  #    names from integrator.f.sys (used by getStatesAsSymbols/getParametersAsSymbols).
  local massMatrix = ModelingToolkit.calculate_massmatrix(reducedSystem)
  local problem
  if massMatrix isa LinearAlgebra.UniformScaling
    @debug "DirectRHS: pure ODE (identity mass matrix)"
    local f = _buildDirectODEFunction(rhsFunc, u0, p_vec, tspan[1];
                                      sys=reducedSystem, jacFunc=jacFunc, jacProto=jacProto)
    problem = ModelingToolkit.ODEProblem{true}(f, u0, tspan, p_vec; callback=allCallbacks)
  else
    @debug "DirectRHS: DAE with mass matrix"
    local mm = collect(massMatrix)
    #= A sparse Jacobian prototype needs a sparse mass matrix, otherwise the
       solver's W = M - gamma*J assembly densifies or mismatches. =#
    local mmForF = jacFunc === nothing ? mm : Symbolics.SparseArrays.sparse(mm)
    local f = _buildDirectODEFunction(rhsFunc, u0, p_vec, tspan[1];
                                      mass_matrix=mmForF, sys=reducedSystem,
                                      jacFunc=jacFunc, jacProto=jacProto)
    #= Pinned indices: vars whose u0 came from a fixed=true Modelica init eq
       (after splitInitialValues). The DAE init solver must NOT modify these,
       otherwise an algebraic var pinned by `start=1, fixed=true` (e.g.
       sd1.s_rel = 1) gets overwritten by the alg residual that depends on
       free vars (e.g. m1.s, m2.s) — collapsing to a different consistent
       root than the user requested. =#
    #= Hard initialization values also cover literal `x ~ v` initialization
       equations; those are user-requested constraints exactly like fixed=true
       starts and must survive the free phases of the init solve. The sidecar
       only knows splitInitialValues-level keys, so the literal init-eq keys
       are unioned in here. Lifted-discrete states are excluded everywhere:
       their initialization is owned by the t0 initialize affects, and pinning
       them couples Newton to relation-kink defining rows it cannot satisfy. =#
    local discreteNames = preMem === nothing ? OrderedSet{String}() :
                          OrderedSet{String}(string(k) for k in keys(preMem))
    local isDiscreteKey = k -> replace(k, "(t)" => "") in discreteNames
    local pinnedKeyStrSet = OrderedSet{String}(
      k for k in union(explicitPinnedInitialValueKeys(reducedSystem, hardInitialValues),
                       initEqPinKeys)
      if !isDiscreteKey(k))
    local pinnedIdx = Int[i for (i, st) in enumerate(states)
                          if string(st) in pinnedKeyStrSet]
    #= Discrete latches with literal init values: kept out of the Newton
       phases (their defining rows are relation cliffs) but re-imposed in the
       final constrained polish. =#
    local discretePinnedIdx = Int[i for (i, st) in enumerate(states)
                                  if isDiscreteKey(string(st)) && string(st) in initEqPinKeys]
    local derivativeInitTargets = _derivativeInitializationTargets(
      reducedSystem, states; resolvedParams=resolvedParams)
    local eqLabels = try
      ModelingToolkit.equations(reducedSystem)
    catch
      nothing
    end
    #= Signal-valued initialization equations become extra residual rows of
       the init solve. Validate the generated evaluator once on the entry
       guesses; a throwing or non-finite evaluator must not poison Newton. =#
    local symInit = _symbolicInitializationResiduals(reducedSystem, states, params,
                                                     ModelingToolkit.get_iv(reducedSystem), mm;
                                                     resolvedParams=resolvedParams,
                                                     excludeNames=discreteNames)
    local extraResiduals = nothing
    if symInit !== nothing
      local (gF, dIdxs, mmS) = symInit
      local candidate = (du, u) -> begin
        local g = gF(u, p_vec, 0.0)
        Float64[dIdxs[i] == 0 ? Float64(g[i]) : du[dIdxs[i]] - mmS[i] * Float64(g[i])
                for i in 1:length(dIdxs)]
      end
      local probeOk = try
        local duProbe = similar(u0)
        rhsFunc(duProbe, u0, p_vec, 0.0)
        all(isfinite, candidate(duProbe, u0))
      catch
        false
      end
      if probeOk
        extraResiduals = candidate
        @debug "DirectRHS: enforcing $(length(dIdxs)) symbolic initialization residual rows"
      else
        @debug "DirectRHS: symbolic initialization residuals failed probe, skipping"
      end
    end
    u0 = _solveDAEInitialization!(u0, rhsFunc, p_vec, mm;
                                  pinned=pinnedIdx,
                                  derivative_targets=derivativeInitTargets,
                                  eqLabels=eqLabels,
                                  extra_residuals=extraResiduals,
                                  discrete_pinned=discretePinnedIdx)
    problem = ModelingToolkit.ODEProblem{true}(f, u0, tspan, p_vec; callback=allCallbacks)
  end

  #= After initialization pre(x) equals the committed x(t0); refresh the
     lifted-discrete memory from the solved initial state. =#
  resetDiscretePreMem!(preMem, reducedSystem, u0)

  @debug "DirectRHS: problem constructed successfully"
  return problem
end

"""
    resetDiscretePreMem!(preMem, reducedSystem, u0)

Refresh the lifted-discrete pre() memory from an initial state vector so
pre(x) at the first event resolves to the committed x(t0). Must run at every
solve start: a cached build otherwise carries the previous run's final values.
"""
function resetDiscretePreMem!(preMem, reducedSystem, u0)
  (preMem === nothing || u0 === nothing) && return nothing
  local states = try
    ModelingToolkit.unknowns(reducedSystem)
  catch
    return nothing
  end
  for (i, st) in enumerate(states)
    i <= length(u0) || break
    local k = Symbol(replace(string(st), "(t)" => ""))
    haskey(preMem, k) && (preMem[k] = u0[i])
  end
  return nothing
end


#= Returns `(values, constraintKeys)`. `values` seeds u0; `constraintKeys`
   names the literal initialization-equation LHS variables: user-requested
   constraints that must stay pinned through the free phases of the init
   solve regardless of what splitInitialValues demoted to guesses. =#
function _collectHardInitializationValues(reducedSystem, finalInitialValues;
                                          resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local values = Dict{Any, Float64}()
  local constraintKeys = OrderedSet{String}()
  for pair in finalInitialValues
    local val = _tryToFloat64(pair.second; resolvedParams=resolvedParams)
    val === nothing && continue
    values[pair.first] = val
  end
  local initEqs = try
    ModelingToolkit.initialization_equations(reducedSystem)
  catch
    return (values, constraintKeys)
  end
  for eq in initEqs
    startswith(string(eq.lhs), "Differential(") && continue
    local rhsVal = _literalNumericValue(eq.rhs)
    if rhsVal === nothing
      rhsVal = _tryToFloat64(eq.rhs; resolvedParams=resolvedParams)
    end
    rhsVal === nothing && continue
    values[eq.lhs] = rhsVal
    push!(constraintKeys, string(eq.lhs))
  end
  return (values, constraintKeys)
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
    local target = _tryToFloat64(eq.rhs; resolvedParams=resolvedParams)
    target === nothing && continue
    push!(targets, matchedIdx => target)
  end
  return targets
end


#= Initialization equations with a signal-valued RHS (references unknowns or
   observed variables). Literal / parameter-resolvable rows are pinned via
   _collectHardInitializationValues and _derivativeInitializationTargets;
   the rows collected here become extra residual rows of the DAE init solve,
   so user initial equations like `x = signal` and `der(x) = signal` hold at
   t0. Returns `(gFunc, derIdxs, mmScales)` where `gFunc(u, p, t)` evaluates
   the row expressions, `derIdxs[i] == 0` marks an algebraic row with residual
   `g[i]`, and `derIdxs[i] > 0` marks a derivative row with residual
   `du[derIdxs[i]] - mmScales[i] * g[i]`. Returns nothing when no such rows
   exist or they cannot be reduced to states/params. =#
function _symbolicInitializationResiduals(reducedSystem, states, params, iv, mm;
                                          resolvedParams::Union{Dict{String,Float64},Nothing}=nothing,
                                          excludeNames::AbstractSet{String}=OrderedSet{String}())
  get(ENV, "OMBACKEND_INIT_SYMBOLIC_EQS", "true") == "true" || return nothing
  local initEqs = try
    ModelingToolkit.initialization_equations(reducedSystem)
  catch
    return nothing
  end
  isempty(initEqs) && return nothing
  local stateStrToIdx = OrderedDict{String, Int}(string(st) => i for (i, st) in enumerate(states))
  local exprs = Any[]
  local derIdxs = Int[]
  local mmScales = Float64[]
  for eq in initEqs
    local lhsStr = string(eq.lhs)
    if startswith(lhsStr, "Differential(")
      #= Literal derivative rows are handled as derivative_targets. =#
      _tryToFloat64(eq.rhs; resolvedParams=resolvedParams) === nothing || continue
      local matchedIdx = nothing
      for (stateStr, idx) in stateStrToIdx
        if endswith(lhsStr, "(" * stateStr * ")")
          matchedIdx = idx
          break
        end
      end
      matchedIdx === nothing && continue
      push!(exprs, eq.rhs)
      push!(derIdxs, matchedIdx)
      push!(mmScales, Float64(mm[matchedIdx, matchedIdx]))
    else
      #= Lifted-discrete rows belong to the t0 initialize affects. =#
      replace(lhsStr, "(t)" => "") in excludeNames && continue
      #= Literal algebraic rows are pinned hard values, but only a state can
         be pinned: a literal row on an observed variable (an acceleration-
         zero condition, for example) must be enforced as a residual row. =#
      local rhsVal = _literalNumericValue(eq.rhs)
      rhsVal === nothing && (rhsVal = _tryToFloat64(eq.rhs; resolvedParams=resolvedParams))
      if rhsVal !== nothing && haskey(stateStrToIdx, lhsStr)
        continue
      end
      push!(exprs, eq.lhs - eq.rhs)
      push!(derIdxs, 0)
      push!(mmScales, 1.0)
    end
  end
  isempty(exprs) && return nothing
  #= Inline observed definitions on demand so only states, params and the iv
     remain; the observed list is topologically ordered, so bounded repeated
     substitution terminates. =#
  local obsEqs = try
    ModelingToolkit.observed(reducedSystem)
  catch
    Symbolics.Equation[]
  end
  local obsByStr = OrderedDict{String, Any}(string(o.lhs) => o.rhs for o in obsEqs)
  local allowed = OrderedSet{String}(string(st) for st in states)
  for p in params
    push!(allowed, string(p))
  end
  push!(allowed, string(iv))
  for i in 1:length(exprs)
    for _pass in 1:(length(obsEqs) + 1)
      local pending = OrderedDict{Any, Any}()
      for v in Symbolics.get_variables(exprs[i])
        local vs = string(v)
        vs in allowed && continue
        haskey(obsByStr, vs) && (pending[v] = obsByStr[vs])
      end
      isempty(pending) && break
      exprs[i] = Symbolics.substitute(exprs[i], pending)
    end
  end
  #= Drop rows still referencing anything else (e.g. der() inside observed). =#
  local keep = Int[]
  for (i, ex) in enumerate(exprs)
    if all(v -> string(v) in allowed, Symbolics.get_variables(ex))
      push!(keep, i)
    end
  end
  if length(keep) < length(exprs)
    @debug "DirectRHS: dropped $(length(exprs) - length(keep)) symbolic initialization rows (unresolvable references)"
  end
  isempty(keep) && return nothing
  exprs = exprs[keep]
  derIdxs = derIdxs[keep]
  mmScales = mmScales[keep]
  local gFunc = try
    local fExpr = Symbolics.build_function(exprs, states, params, iv; expression = Val{true})
    _exprToRTGFunction(fExpr[1])
  catch e
    @debug "DirectRHS: could not build symbolic initialization residuals" exception = e
    return nothing
  end
  return (gFunc, derIdxs, mmScales)
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
    return _demoteWideNumericLiterals!(result[2])  # in-place form
  catch e
    @warn "DirectRHS: CSE failed, using direct generation" exception=(e, catch_backtrace())
  end
  # Fallback without CSE
  local result = Symbolics.build_function(rhs_list, states, params, iv;
                                           expression=Val{true})
  return _demoteWideNumericLiterals!(result[2])
end

"""
    _buildSparseJacobian(rhs_list, states, params, iv, u0, p_vec, t0)

Build an in-place sparse symbolic Jacobian for the RHS so implicit solvers do
not finite-difference one RHS column per state every step. The generated
function is probed once at `(u0, p_vec, t0)`: an unresolved symbolic
derivative (opaque external call) passes build_function silently and only
throws when the function runs. Returns `(jacFunc, jacPrototype)`, or
`(nothing, nothing)` when generation is disabled or the probe fails.
"""
function _buildSparseJacobian(rhs_list, states, params, iv, u0, p_vec, t0)
  OMBackend.DIRECT_JAC_GENERATION[] || return (nothing, nothing)
  try
    local jacSym = Symbolics.sparsejacobian(rhs_list, states)
    local result = Symbolics.build_function(jacSym, states, params, iv;
                                            expression=Val{true}, cse=true)
    local jacFunc = _exprToRTGFunction(_demoteWideNumericLiterals!(result[2]))
    local jacProto = similar(jacSym, Float64)
    jacProto.nzval .= 0.0
    local probe = copy(jacProto)
    jacFunc(probe, u0, p_vec, t0)
    @debug "DirectRHS: symbolic sparse Jacobian with $(length(jacProto.nzval)) structural nonzeros"
    return (jacFunc, jacProto)
  catch e
    @debug "DirectRHS: symbolic Jacobian generation failed; solver will finite-difference" exception=(e, catch_backtrace())
    return (nothing, nothing)
  end
end

#= Symbolic simplification can fold integer parameter products into exact
   Rational{BigInt} coefficients; one such literal promotes every downstream
   operation to BigFloat, allocating per RHS call. Demote at the generated-code
   boundary where Float64 semantics are already assumed. Exact integer
   literals (Int128/BigInt) stay untouched: RNG state constants are bit-exact
   and exceed Float64's 2^53 integer range. =#
_demoteWideNumericLiterals!(x) =
  x isa Union{Rational, BigFloat, Irrational} ? Float64(x) : x
function _demoteWideNumericLiterals!(ex::Expr)
  for (i, a) in pairs(ex.args)
    ex.args[i] = _demoteWideNumericLiterals!(a)
  end
  return ex
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


# Trivial reinit: no post-event DAE re-initialization, so the merged callback's
# default (nothing) is behaviour-preserving.
_isTrivialReinit(ia)::Bool = ia === nothing || ia isa ModelingToolkit.SciMLBase.NoInit

#= Typed callable structs for the merged continuous callback. Typed fields and a
   concrete struct type keep the merged condition/affect inferred (vs a closure
   boxing its captures); the FunctionWrapper in `_eraseContinuousCallbacks` erases
   the outer type for the image bake. =#
struct _MergedContinuousCondition{S}
  subs::S
  offsets::Vector{Int}
  nsub::Int
end

# `integrator` stays untyped: the FunctionWrapper declares it `Any` to keep the
# wrapped callback type model-independent.
# `out` / `u` are AbstractVector, NOT Vector: SciML's VectorContinuousCallback passes
# `out` as a SubArray view of the rootfind buffer. A concrete `Vector{Float64}` arg (here
# or in the FunctionWrapper signature) forces a convert/copy, so writes to `out` land in a
# discarded copy and no crossing is ever detected.
function (c::_MergedContinuousCondition)(out::AbstractVector{Float64}, u::AbstractVector{Float64},
                                         t::Float64, integrator)::Nothing
  local SB = ModelingToolkit.SciMLBase
  for k in 1:c.nsub
    local s = c.subs[k]
    if s isa SB.VectorContinuousCallback
      s.condition(view(out, (c.offsets[k] + 1):c.offsets[k + 1]), u, t, integrator)
    else
      out[c.offsets[k] + 1] = s.condition(u, t, integrator)
    end
  end
  return nothing
end

# One struct serves both affect! and affect_neg! (selected by `neg`). The vector
# affect signature is (integrator, componentIndex).
struct _MergedContinuousAffect{S}
  subs::S
  offsets::Vector{Int}
  lens::Vector{Int}
  nsub::Int
  neg::Bool
end

function (a::_MergedContinuousAffect)(integrator, gidx::Int)::Nothing
  local SB = ModelingToolkit.SciMLBase
  # Map the global 1-based component index to (subIndex, localIndex).
  local k::Int = a.nsub
  local li::Int = a.lens[a.nsub]
  for kk in 1:a.nsub
    if gidx <= a.offsets[kk + 1]
      k = kk
      li = gidx - a.offsets[kk]
      break
    end
  end
  local s = a.subs[k]
  local aff = a.neg ? s.affect_neg! : s.affect!
  aff === nothing && return nothing
  s isa SB.VectorContinuousCallback ? aff(integrator, li) : aff(integrator)
  return nothing
end

# Runs every sub-callback's `initialize` at integration start. Lets a sub carrying a
# custom initialize (e.g. chua's DAE event) collapse WITHOUT dropping it; the merge
# is FunctionWrapper-erased so the VCC's initialize param stays model-independent.
struct _MergedContinuousInitialize{S}
  subs::S
  nsub::Int
end

function (m::_MergedContinuousInitialize)(c, u, t, integrator)::Nothing
  for k in 1:m.nsub
    local s = m.subs[k]
    s.initialize(s, u, t, integrator)
  end
  return nothing
end

"""
    _eraseContinuousCallbacks(cbset)

Collapse the continuous callbacks of a `CallbackSet` into a single
`VectorContinuousCallback` whose combined condition / affect! / affect_neg! are
typed callable structs wrapped in `FunctionWrappers` (integrator typed `Any`)
that dispatch to the original per-component callbacks by index. This removes the
two model-specific axes of the callback type, the tuple arity (number of
continuous callbacks) and the per-event closure types, so the `CallbackSet` type
is constant across models and `solve` can be compiled once and baked into the
image. Per-component event semantics are preserved exactly: each component's own
condition, affect! and affect_neg! are called unchanged. Discrete callbacks are
passed through.

Called at SOLVE time (see `simulateIMTK`) in a settled world, so it collapses any
callable — OM-generated closures and MTK `process_events` `CompiledCondition` /
`FunctionalAffect` alike (the integrator never dispatches on the concrete type).
Returns `cbset` unchanged (no collapse) when there is no continuous callback, or on
any structural surprise: a non-`CallbackSet` argument, a continuous entry that is
neither a scalar `ContinuousCallback` nor a `VectorContinuousCallback`, non-uniform
`rootfind` / `save_positions` across components, or any component carrying event
metadata a flat merge cannot represent (a custom `initialize` / `finalize`, an
`idxs` slice, or a non-trivial reinitialization algorithm).
"""
function _eraseContinuousCallbacks(cbset)
  local SB = ModelingToolkit.SciMLBase
  cbset isa SB.CallbackSet || return cbset
  local subs = collect(cbset.continuous_callbacks)
  local dc = cbset.discrete_callbacks
  isempty(subs) && return cbset
  for s in subs
    (s isa SB.ContinuousCallback || s isa SB.VectorContinuousCallback) || return cbset
  end
  #= No parentmodule check: this runs at SOLVE time (see simulateIMTK), in a settled
     world, so MTK process_events callbacks (CompiledCondition / FunctionalAffect,
     RGF-backed) collapse correctly too. The structural guard below is the only safety
     bound the integrator needs (it never dispatches on the concrete callback type). =#
  #= A flat merge is faithful only when no sub-callback carries event metadata the
     merge cannot represent: a custom `finalize` (would be dropped), an `idxs` slice
     (the condition would read the wrong state), or a non-trivial reinitialization
     algorithm (post-event DAE consistency would change). A custom `initialize` IS
     allowed: it is preserved via the merged initialize below (chua's DAE event). =#
  for s in subs
    (s.finalize === SB.FINALIZE_DEFAULT &&
     s.idxs === nothing &&
     _isTrivialReinit(s.initializealg)) || return cbset
  end
  #= A single VectorContinuousCallback applies one rootfind / save_positions to
     every component, so only collapse when these already agree. =#
  local rootfind = subs[1].rootfind
  local savePos = subs[1].save_positions
  for s in subs
    (s.rootfind == rootfind && s.save_positions == savePos) || return cbset
  end
  local lens::Vector{Int} = Int[(s isa SB.VectorContinuousCallback) ? s.len : 1 for s in subs]
  local offsets::Vector{Int} = cumsum(vcat(0, lens))   # offsets[k] = #components before sub k
  local total::Int = offsets[end]
  local nsub::Int = length(subs)
  local condF = _MergedContinuousCondition(subs, offsets, nsub)
  local affF = _MergedContinuousAffect(subs, offsets, lens, nsub, false)
  local affNF = _MergedContinuousAffect(subs, offsets, lens, nsub, true)
  local initF = _MergedContinuousInitialize(subs, nsub)
  local FW = DiffEqBase.FunctionWrapper
  local condW = FW{Nothing, Tuple{AbstractVector{Float64}, AbstractVector{Float64}, Float64, Any}}(condF)
  local affW = FW{Nothing, Tuple{Any, Int}}(affF)
  local affNW = FW{Nothing, Tuple{Any, Int}}(affNF)
  #= Always FunctionWrapper-wrap the merged initialize (even when every sub uses the
     default) so the VCC's initialize param is the SAME model-independent type whether or
     not a sub carries a custom initialize -> chua and the synthetic bake share one type. =#
  local initW = FW{Nothing, Tuple{Any, Any, Any, Any}}(initF)
  local vcc = SB.VectorContinuousCallback(condW, affW, affNW, total;
                                          initialize = initW,
                                          rootfind = rootfind, save_positions = savePos)
  return SB.CallbackSet(vcc, dc...)
end


"""
    _toFloat64(val; resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)

Convert a value to Float64, handling Symbolics.Num wrappers, constant symbolic
expressions, and parameter references (resolved via resolvedParams dict).
"""
function _toFloat64(val; resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local resolved = _tryToFloat64(val; resolvedParams=resolvedParams)
  resolved === nothing || return resolved
  @warn "DirectRHS: could not convert value to Float64, using 0.0" val=val type=typeof(val)
  return 0.0
end

function _tryToFloat64(val; resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)::Union{Float64, Nothing}
  local unwrapped = val isa Symbolics.Num ? Symbolics.unwrap(val) : val
  unwrapped isa Number && return Float64(unwrapped)
  # Constant symbolic expression (no free variables): parse its string repr
  local freeVars = try
    Symbolics.get_variables(unwrapped)
  catch
    return nothing
  end
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
  return nothing
end
