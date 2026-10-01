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
values are immaterial. The mass matrix, Jacobian, time derivative `tgradFunc`
and `sys` are attached to the function we build, so type erasure does not drop
them.
"""
function _buildDirectODEFunction(rhsFunc, u0, p_vec, t0;
                                 mass_matrix=nothing, sys=nothing,
                                 jacFunc=nothing, jacProto=nothing, tgradFunc=nothing)
  if !OMBackend.DIRECT_RHS_TYPE_ERASE[]
    local jacKw = jacFunc === nothing ? NamedTuple() : (; jac=jacFunc, jac_prototype=jacProto)
    local tgradKw = tgradFunc === nothing ? NamedTuple() : (; tgrad=tgradFunc)
    return mass_matrix === nothing ?
      ModelingToolkit.ODEFunction{true}(rhsFunc; sys=sys, jacKw..., tgradKw...) :
      ModelingToolkit.ODEFunction{true}(rhsFunc; mass_matrix=mass_matrix, sys=sys, jacKw..., tgradKw...)
  end
  local FW = ModelingToolkit.SciMLBase.FunctionWrapperSpecialize
  #= Multi-variant wrapper (Float64 + ForwardDiff Dual signatures) so autodiff
     solvers stay correct; the Jacobian is never called with Duals, so a single
     variant suffices there. =#
  local wrappedRHS = DiffEqBase.wrapfun_iip(rhsFunc, (u0, u0, p_vec, t0))
  local erasedKw = jacFunc === nothing ? NamedTuple() :
    (; jac = DiffEqBase.wrapfun_jac_iip(jacFunc, (jacProto, u0, p_vec, t0)),
       jac_prototype = jacProto)
  #= Every direct-RHS problem has a tgrad (_buildTimeDerivative): one erased type. =#
  tgradFunc === nothing ||
    (erasedKw = (; erasedKw..., tgrad = DiffEqBase.wrapfun_jac_iip(tgradFunc, (u0, u0, p_vec, t0))))
  return mass_matrix === nothing ?
    ModelingToolkit.ODEFunction{true, FW}(wrappedRHS; sys=sys, erasedKw...) :
    ModelingToolkit.ODEFunction{true, FW}(wrappedRHS; mass_matrix=mass_matrix, sys=sys, erasedKw...)
end

#= Re-initialization per problem (keyed by its generated RHS function):
   parameter vector -> consistent initial state, the vector's
   initialization-defined parameters assigned. See buildDirectRHSProblem. =#
const DAE_REINIT = IdDict{Any, Function}()

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
                               liftedDiscretes=String[],
                               freeParameters=String[],
                               initRelations=Any[],
                               initClusters=Any[],
                               discreteStarts=Dict{String, Float64}())
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

  # Resolve parameter values first: _buildStateVector substitutes symbolic
  # parameter references in initial conditions, and an empty system needs
  # only them.
  local resolvedParams = _resolveParamValues(pars; used = Set{String}(string.(params)))
  #= Parameters an initialization equation defines (fixed = false): assigned
     from u at the initial state (in the DAE init solve, at each evaluation).
     A row may read another assigned parameter: evaluating again settles a
     chain. =#
  local paramAssign = _initialParameterAssignments(reducedSystem, states, params, iv)
  local assignedNames = paramAssign === nothing ? OrderedSet{String}() :
    OrderedSet{String}(string(params[k]) for k in last(paramAssign))
  #= Real parameters without a value that no initialization equation assigns
     (FREE_PARAMETERS: fixed = false; the MSL InitSpringConstant's spring.c,
     which a fixed rev.a = 0 determines): unknowns of the DAE init solve. =#
  local freeIdx = Int[k for (k, p) in enumerate(params)
                      if string(p) in freeParameters && !(string(p) in assignedNames)]
  #= Parameters bound to an assigned or free one (spring.spring.c = spring.c)
     follow it; _buildParamVector resolved them from its start value. =#
  local dependents = _parameterDependents(pars, params,
                                          union(assignedNames, OrderedSet{String}(string(params[k]) for k in freeIdx));
                                          resolvedParams=resolvedParams)
  isempty(setdiff(freeParameters, string.(params))) ||
    @debug "DirectRHS: free parameters not among the system's: $(setdiff(freeParameters, string.(params)))"
  local unknownParamNames = union(assignedNames, OrderedSet{String}(string(params[k]) for k in freeIdx),
                                  dependents === nothing ? OrderedSet{String}() : dependents[3])
  local follow! = dependents === nothing ? nothing : let (dF, dIdxs) = dependents
    pv -> begin
      local vals = dF(pv)
      for (i, k) in enumerate(dIdxs)
        pv[k] = Float64(vals[i])
      end
      nothing
    end
  end
  local assignParams! = paramAssign === nothing ? nothing : let (pF, pIdxs) = paramAssign
    (pv, u, t) -> begin
      for _ in 1:length(pIdxs)
        local vals = pF(u, pv, t)
        local changed = false
        for (i, k) in enumerate(pIdxs)
          local v = Float64(vals[i])
          changed |= !isequal(pv[k], v)
          pv[k] = v
        end
        follow! === nothing || follow!(pv)
        changed || break
      end
      nothing
    end
  end

  #= An empty system (0 unknowns after structural_simplify): a dummy 1-element
     state, so the ODE solver does not reject an empty range, with the system
     and its parameter values: its observed variables are the whole result (MSL
     Media Inverse_sine, IdealGasH2O, Utilities' readRealParameterModel), and
     without `sys` no signal could be read from the solution. Its events and
     initial parameter assignments as for any system; a free parameter is an
     unknown of the init solve, which it does not run. =#
  if nStates == 0
    @debug "DirectRHS: empty system (0 unknowns), building trivial dummy problem"
    isempty(freeIdx) || OMBackend.unsupported("free parameters (fixed = false) of a system without states",
                                              join((string(params[k]) for k in freeIdx), ", "))
    local emptyRHS = (du, u, p, t) -> (du[1] = 0.0)
    local f0 = ModelingToolkit.ODEFunction{true}(emptyRHS; sys = reducedSystem)
    local p0 = _buildParamVector(params, pars; resolvedParams=resolvedParams)
    if assignParams! !== nothing
      assignParams!(p0, Float64[], tspan[1])
      DAE_REINIT[emptyRHS] = pv -> (assignParams!(pv, Float64[], tspan[1]); [0.0])
    end
    return ModelingToolkit.ODEProblem{true}(f0, [0.0], tspan, p0;
                                            callback=_extractAndMergeEventCallbacks(reducedSystem, callbacks))
  end

  # 1. Build the RHS function expression from symbolic equations
  local rhs_list = [eq.rhs for eq in eqs]
  local f_ip_expr = _buildRHSExpression(rhs_list, states, params, iv)

  # Dump the actual generated RHS expression — see MTKDump.
  dumpRHSExpression(rhs_list, f_ip_expr)

  # 2. Create world-age-safe function via RuntimeGeneratedFunction
  local rhsFunc = _exprToRTGFunction(f_ip_expr)

  # 3. Build u0 and parameter vectors in the correct ordering.
  #= The initialization constraints read those as unknowns, not as the values
     resolved from their start values: `x = k` must hold for the solved k. =#
  local initResolved = isempty(unknownParamNames) ? resolvedParams :
    Dict{String, Float64}(k => v for (k, v) in resolvedParams if !(k in unknownParamNames))
  # Extract guesses from the reduced system. These are properly mapped to
  # post-simplification unknowns and provide Modelica start values for variables
  # that splitInitialValues could not map (pre-simplification names do not match).
  local systemGuesses = ModelingToolkit.guesses(reducedSystem)
  local (hardInitialValues, initEqPinKeys) = _collectHardInitializationValues(
    reducedSystem, finalInitialValues; resolvedParams=initResolved)
  local observedEquations = ModelingToolkit.observed(reducedSystem)
  local u0 = _buildStateVector(states, finalInitialValues; resolvedParams=resolvedParams,
                                systemGuesses=systemGuesses,
                                hardInitialValues=hardInitialValues,
                                observedEquations=observedEquations)
  local p_vec = _buildParamVector(params, pars; resolvedParams=resolvedParams)

  @debug "DirectRHS: u0 has $(count(!iszero, u0))/$(nStates) nonzero, p has $(count(!iszero, p_vec))/$(nParams) nonzero"
  OMBackend.envSwitch("OMBACKEND_INIT_TRACE") &&
    println("[initu0] states ", states, "\n[initu0] hard starts ", finalInitialValues, "\n[initu0] guesses ", systemGuesses,
            "\n[initu0] initialization equations ", ModelingToolkit.initialization_equations(reducedSystem),
            "\n[initu0] u0 ", u0)

  #= Symbolic sparse Jacobian; nothing when not differentiable. Built after
     u0/p_vec so the generated function can be probed once: an unresolved
     symbolic derivative surfaces only when the function runs, not at build. =#
  local (jacFunc, jacProto) = _buildSparseJacobian(rhs_list, states, params, iv,
                                                   u0, p_vec, tspan[1])
  local tgradFunc = _buildTimeDerivative(rhs_list, states, params, iv, rhsFunc, u0, p_vec, tspan[1])

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
  #= A pure ODE with free parameters, with derivative initial equations, or
     with ones whose value is not a number (`y = 2*x`), needs the init solve
     too: M = I. Without it they were left out (u0 the start values). =#
  if massMatrix isa LinearAlgebra.UniformScaling && isempty(freeIdx) &&
     isempty(first(_observedDerivativeInitEquations(reducedSystem, states; resolvedParams=initResolved))) &&
     !_hasSolvedInitializationRows(reducedSystem; resolvedParams=initResolved)
    @debug "DirectRHS: pure ODE (identity mass matrix)"
    local f = _buildDirectODEFunction(rhsFunc, u0, p_vec, tspan[1];
                                      sys=reducedSystem, jacFunc=jacFunc, jacProto=jacProto,
                                      tgradFunc=tgradFunc)
    if assignParams! !== nothing
      assignParams!(p_vec, u0, tspan[1])
      #= Other tunable parameter values may change them. =#
      local u0Start = copy(u0)
      DAE_REINIT[rhsFunc] = pv -> (assignParams!(pv, u0Start, tspan[1]); copy(u0Start))
    end
    problem = ModelingToolkit.ODEProblem{true}(f, u0, tspan, p_vec; callback=allCallbacks)
  else
    @debug "DirectRHS: DAE with mass matrix"
    local mm = massMatrix isa LinearAlgebra.UniformScaling ?
      Matrix{Float64}(LinearAlgebra.I, nStates, nStates) : collect(massMatrix)
    #= A sparse Jacobian prototype needs a sparse mass matrix, otherwise the
       solver's W = M - gamma*J assembly densifies or mismatches. =#
    local mmForF = jacFunc === nothing ? mm : Symbolics.SparseArrays.sparse(mm)
    local f = _buildDirectODEFunction(rhsFunc, u0, p_vec, tspan[1];
                                      mass_matrix=mmForF, sys=reducedSystem,
                                      jacFunc=jacFunc, jacProto=jacProto, tgradFunc=tgradFunc)
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
       the discrete clusters set them at the start (their start bodies), and pinning
       them couples Newton to relation-kink defining rows it cannot satisfy. =#
    local discreteNames = OrderedSet{String}(liftedDiscretes)
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
      reducedSystem, states; resolvedParams=initResolved)
    local eqLabels = ModelingToolkit.equations(reducedSystem)
    #= Signal-valued initialization equations become extra residual rows of
       the init solve. Validate the generated evaluator once on the entry
       guesses; a throwing or non-finite evaluator must not poison Newton. =#
    local symInit = _symbolicInitializationResiduals(reducedSystem, states, params,
                                                     ModelingToolkit.get_iv(reducedSystem), mm;
                                                     resolvedParams=initResolved,
                                                     excludeNames=union(discreteNames, assignedNames))
    local extraResiduals = nothing
    #= The residual rows for parameter values `pv` (a re-initialization for
       other tunable parameter values evaluates them at those). =#
    local symRowsAt = symInit === nothing ? nothing : let (gF, dIdxs, mmS) = symInit
      pv -> (du, u) -> begin
        local g = gF(u, pv, 0.0)
        Float64[dIdxs[i] == 0 ? Float64(g[i]) : du[dIdxs[i]] - mmS[i] * Float64(g[i])
                for i in 1:length(dIdxs)]
      end
    end
    #= Literal derivative targets on algebraic unknowns (a zero mass-matrix
       row): the init solve's derivative targets need a differential state, and
       dropped them. The MSL AIMC_Initialize's `der(aimc.idq_sr) = zeros(2)`, its
       steady state, on currents index reduction left algebraic. Their time
       derivatives come from the differentiated algebraic rows. =#
    local algIdx = Int[i for i in 1:size(mm, 1) if mm[i, i] == 0]
    local difIdx = Int[i for i in 1:size(mm, 1) if mm[i, i] != 0]
    local algDerTargets = Pair{Int, Float64}[t for t in derivativeInitTargets
                                             if 1 <= t.first <= size(mm, 1) && mm[t.first, t.first] == 0]
    #= Literal derivative targets on observed variables (not unknowns): the
       MSL FundamentalWave AIMC_Initialize's der(aimc.airGap.V_msr.re) = 0.
       der(w) = ∇w · u̇ + ∂w/∂t, u̇ the unknowns' derivatives (the algebraic
       ones as above). =#
    local obsDer = LinearAlgebra.isdiag(mm) ?
      _observedDerivativeTargets(reducedSystem, states, params, ModelingToolkit.get_iv(reducedSystem);
                                 resolvedParams=initResolved) : nothing
    #= ż_a with the symbolic Jacobian only (finite differences of finite
       differences are too noisy for the solve's tolerance, and n^2 RHS calls
       per iteration), and a diagonal mass matrix (row i the equation of
       unknown i). =#
    local needAlg = !isempty(algDerTargets) ||
      (obsDer !== nothing && any(j -> j <= obsDer[4] && mm[j, j] == 0, obsDer[3]))
    #= They were left out with a warning (MSL Fluid MomentumBalanceFittings,
       HeatExchangerSimulation, which then failed at the simulation anyway). =#
    needAlg && (jacFunc === nothing || !LinearAlgebra.isdiag(mm)) &&
      OMBackend.unsupported("derivative initial equations on algebraic or observed variables without a symbolic Jacobian or with a non-diagonal mass matrix",
                            length(algDerTargets))
    local ftAlg = needAlg ? _explicitTimeDerivative(rhs_list[algIdx], states, params, iv) : nothing
    local derRowsAt = (isempty(algDerTargets) && obsDer === nothing) ? nothing : let
      local pos = Dict(i => k for (k, i) in enumerate(algIdx))
      local tPos = Int[pos[t.first] for t in algDerTargets]
      local tVal = Float64[t.second for t in algDerTargets]
      pv -> begin
        local J = needAlg ? copy(jacProto) : nothing
        local nz = obsDer === nothing ? Float64[] : zeros(length(obsDer[2]))
        (du, u) -> begin
          local zdot = needAlg ?
            _algebraicDerivatives!(J, rhsFunc, jacFunc, ftAlg, mm, algIdx, difIdx, u, du, pv, 0.0) : Float64[]
          local rows = zdot[tPos] .- tVal
          obsDer === nothing && return rows
          local (nzF, I, Jc, n, tgts) = obsDer
          local udot = Float64[du[i] / (mm[i, i] == 0 ? 1.0 : mm[i, i]) for i in 1:n]
          needAlg && (udot[algIdx] = zdot)
          nzF(nz, u, pv, 0.0)
          local obsRows = -copy(tgts)
          for k in eachindex(I)
            obsRows[I[k]] += nz[k] * (Jc[k] <= n ? udot[Jc[k]] : 1.0)
          end
          vcat(rows, obsRows)
        end
      end
    end
    #= The kinds of rows are probed apart below: a part that fails at the
       entry guesses is left out on its own. =#
    local residualsAt = nothing
    #= The init solve's RHS assigns the initialization-defined parameters
       first. Their f may assert where the model's own equations only guard
       (the MSL selectBranch asserts a regular loop position): at a trial point
       (the entry guesses, a line-search step, a finite difference) that is a
       non-finite residual the solve rejects, not a failed build. =#
    local initRhs = assignParams! === nothing ? rhsFunc : (du, u, p, t) -> begin
      try
        assignParams!(p, u, t)
      catch e
        OMBackend._fallback(e, :buildDirectRHSProblem_5)
        fill!(du, NaN)
        return nothing
      end
      rhsFunc(du, u, p, t)
    end
    local probeRows = rowsAt -> try
      local duProbe = similar(u0)
      initRhs(duProbe, u0, p_vec, 0.0)
      all(isfinite, rowsAt(p_vec)(duProbe, u0))
    catch _e
      OMBackend._fallback(_e, :buildDirectRHSProblem_6, impact = :result)
      false
    end
    local parts = Any[]
    for (label, rowsAt) in (("symbolic initialization", symRowsAt), ("derivative", derRowsAt))
      rowsAt === nothing && continue
      #= A part not finite at the entry guesses was left out with a warning
         (MSL AIMC_withLosses, which then failed anyway). =#
      probeRows(rowsAt) || OMBackend.unsupported("$(label) initial-equation rows not finite at the entry guesses", label)
      push!(parts, rowsAt)
    end
    if !isempty(parts)
      residualsAt = length(parts) == 1 ? parts[1] :
        pv -> (local fs = [r(pv) for r in parts]; (du, u) -> reduce(vcat, [f(du, u) for f in fs]))
      extraResiduals = residualsAt(p_vec)
    end
    local initDiscretes = _initialDiscreteClusters(initClusters, reducedSystem, discreteStarts)
    local keptIdx = vcat(pinnedIdx, discretePinnedIdx)
    #= The relations' literal values select the branches the initialization
       holds for (MLS 8.6). Codegen takes them from the start attributes; at the
       entry point they come from the start state's observed values, which the
       initialization keeps where it can (the MSL V6 cylinder: `x > 0.933` with
       x = 1 - s_rel/L was true from s_rel's start 0, while the crank's
       kinematics puts x below it; the solve held the other branch's quartic far
       outside its range). Non-finite values keep the compiled literal. =#
    local initRels = _initialRelationLiterals(reducedSystem, params, initRelations)
    initRels === nothing || initRels.eval!(p_vec, u0, 0.0; finiteOnly = true)
    local uEntry = copy(u0)
    local firstErr = nothing
    local solveFree = (u, pv) -> _solveDAEInitializationFree!(u, initRhs, pv, mm, freeIdx, follow!;
                                                              pinned=pinnedIdx,
                                                              derivative_targets=derivativeInitTargets,
                                                              eqLabels=eqLabels,
                                                              extra_residuals=extraResiduals,
                                                              discrete_pinned=discretePinnedIdx)
    u0 = try
      solveFree(u0, p_vec)
    catch e
      OMBackend._fallback(e, :buildDirectRHSProblem_7, impact = :result)
      #= The relations at the failed solve's last point: where one differs, the
         initialization is solved again from the entry with it (the event
         iteration of the initialization, before a solution exists). =#
      local retried = initRels !== nothing && initRels.eval!(p_vec, u0, 0.0; finiteOnly = true) ?
        try
          solveFree(copy(uEntry), p_vec)
        catch e2
          OMBackend._fallback(e2, :buildDirectRHSProblem_8, impact = :result)
          nothing
        end : nothing
      if retried !== nothing
        retried
      else
        #= With discrete clusters the initialization is tried again from their start values. =#
        initDiscretes === nothing && rethrow()
        firstErr = e
        copy(uEntry)
      end
    end
    #= At the solved state, at the solve's time (_solveDAEInitialization! evaluates at 0.0). =#
    assignParams! === nothing || assignParams!(p_vec, u0, 0.0)
    #= The relations' literals at the solved state; solved again until they settle. =#
    local useExtra = extraResiduals !== nothing
    local resolveWith = (u, pv, ok) -> begin
      local kept = u[keptIdx]
      local un = _solveDAEInitializationFree!(u, initRhs, pv, mm, freeIdx, follow!;
                                              pinned=pinnedIdx,
                                              derivative_targets=derivativeInitTargets,
                                              eqLabels=eqLabels,
                                              extra_residuals=useExtra ? residualsAt(pv) : nothing,
                                              discrete_pinned=discretePinnedIdx,
                                              converged=ok)
      #= A solve that relaxed the fixed starts is not a settled initialization. =#
      isapprox(un[keptIdx], kept; rtol = 1e-8, atol = 1e-10) || (ok[] = false)
      assignParams! === nothing || assignParams!(pv, un, 0.0)
      un
    end
    #= The differential states other than the clusters' members. =#
    local diffKept = Int[i for i in 1:size(mm, 1) if mm[i, i] != 0 &&
                         !(initDiscretes !== nothing && any(c -> i in c.memberIndex, initDiscretes.clusters))]
    local (uSettled, settled) = _settleInitialDiscretes!(resolveWith, u0, uEntry, p_vec, initDiscretes, diffKept)
    firstErr === nothing || settled || throw(firstErr)
    u0 = uSettled
    u0 = _settleInitialRelations!(resolveWith, u0, p_vec, initRels)
    problem = ModelingToolkit.ODEProblem{true}(f, u0, tspan, p_vec; callback=allCallbacks)
    #= The same initialization for other parameter values (tunable parameters,
       OMBackend.withTunableParameters): a run with changed parameters needs
       the consistent initial state for them, not the one solved here. It
       starts from this one, which is close for nearby values. =#
    local u0Solved = copy(u0)
    DAE_REINIT[rhsFunc] = pv -> begin
      local u = _solveDAEInitializationFree!(copy(u0Solved), initRhs, pv, mm, freeIdx, follow!;
                                         pinned=pinnedIdx,
                                         derivative_targets=derivativeInitTargets,
                                         eqLabels=eqLabels,
                                         extra_residuals=useExtra ? residualsAt(pv) : nothing,
                                         discrete_pinned=discretePinnedIdx,
                                         warm=true)
      assignParams! === nothing || assignParams!(pv, u, 0.0)
      u = first(_settleInitialDiscretes!(resolveWith, u, u, pv, initDiscretes, diffKept; startPath = false))
      _settleInitialRelations!(resolveWith, u, pv, initRels)
    end
  end

  @debug "DirectRHS: problem constructed successfully"
  return problem
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
  local initEqs = ModelingToolkit.initialization_equations(reducedSystem)
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


#= Whether an initialization equation needs a solve: one that is not a start
   value `x = v`, a variable of the system and a number (a derivative row,
   `y = 2*x`, `2*x = 4`). =#
function _hasSolvedInitializationRows(reducedSystem; resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)::Bool
  local variables = Set{String}(string(v) for v in Iterators.flatten((ModelingToolkit.unknowns(reducedSystem),
                                                                      ModelingToolkit.parameters(reducedSystem),
                                                                      (o.lhs for o in ModelingToolkit.observed(reducedSystem)))))
  return any(ModelingToolkit.initialization_equations(reducedSystem)) do eq
    startswith(string(eq.lhs), "Differential(") || !(string(eq.lhs) in variables) ||
      _tryToFloat64(eq.rhs; resolvedParams=resolvedParams) === nothing
  end
end

function _literalNumericValue(val)
  local raw = val
  raw = raw isa Symbolics.Num ? Symbolics.unwrap(raw) : raw
  raw isa Number && return Float64(raw)
  raw = Symbolics.value(raw)
  raw = raw isa Symbolics.Num ? Symbolics.unwrap(raw) : raw
  return raw isa Number ? Float64(raw) : nothing
end


function _derivativeInitializationTargets(reducedSystem, states;
                                          resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local stateStrToIdx = Dict{String, Int}(string(st) => i for (i, st) in enumerate(states))
  local targets = Pair{Int, Float64}[]
  local initEqs = ModelingToolkit.initialization_equations(reducedSystem)
  for eq in initEqs
    local lhs = Symbolics.unwrap(eq.lhs)
    (SymbolicUtils.iscall(lhs) && SymbolicUtils.operation(lhs) isa Symbolics.Differential) || continue
    #= The argument itself, not a suffix of the string: D() can hold an expression. =#
    local matchedIdx = get(stateStrToIdx, string(SymbolicUtils.arguments(lhs)[1]), nothing)
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
  OMBackend.envSwitch("OMBACKEND_INIT_SYMBOLIC_EQS") || return nothing
  local initEqs = ModelingToolkit.initialization_equations(reducedSystem)
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
  local keep = _inlineObservedRows!(exprs, reducedSystem, states, params, iv)
  if length(keep) < length(exprs)
    @debug "DirectRHS: dropped $(length(exprs) - length(keep)) symbolic initialization rows (unresolvable references)"
  end
  isempty(keep) && return nothing
  exprs = exprs[keep]
  derIdxs = derIdxs[keep]
  mmScales = mmScales[keep]
  #= With CSE, as the RHS: the rows have the observed equations substituted,
     which a multibody model (MSL fullRobot: 119 unknowns, 1809 observed
     equations) expands to millions of terms as a tree; Julia never finishes
     lowering such a function. =#
  local gFunc = try
    local fExpr = Symbolics.build_function(exprs, states, params, iv; expression = Val{true}, cse = true)
    _exprToRTGFunction(fExpr[1])
  catch e
    OMBackend._fallback(e, :_symbolicInitializationResiduals_2, impact = :result)
    @debug "DirectRHS: could not build symbolic initialization residuals" exception = e
    return nothing
  end
  return (gFunc, derIdxs, mmScales)
end

#= Inline observed definitions into `exprs` on demand so only states, params
   and the iv remain; the observed list is topologically ordered, so bounded
   repeated substitution terminates. Returns the indices of the rows that
   reduced to those (a row still reading der() inside an observed does not). =#
function _inlineObservedRows!(exprs, reducedSystem, states, params, iv)::Vector{Int}
  local obsEqs = ModelingToolkit.observed(reducedSystem)
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
  return Int[i for (i, ex) in enumerate(exprs)
             if all(v -> string(v) in allowed, Symbolics.get_variables(ex))]
end

#= Initialization equations `p = f(...)` for a parameter p without a value
   (fixed = false; the MSL analytic loop joints' positiveBranch, the branch of
   the loop's solution that the initial positions select). No unknown of the
   init solve is p: as a residual row the solve can only meet it by moving u
   to where f is p's default, relaxing fixed starts (or not at all, for a
   Boolean f). The init solve assigns p = f at each evaluation instead, and p
   keeps its value at the initial state. Returns `(pFunc, pIdxs)`, where
   `pFunc(u, p, t)` evaluates the values of `p[pIdxs]`, or nothing. A row whose
   f reads its own p stays a residual row. =#
function _initialParameterAssignments(reducedSystem, states, params, iv)
  local initEqs = ModelingToolkit.initialization_equations(reducedSystem)
  local paramIdx = Dict{String, Int}(string(p) => i for (i, p) in enumerate(params))
  local exprs = Any[]
  local pIdxs = Int[]
  for eq in initEqs
    local k = get(paramIdx, string(eq.lhs), 0)
    k == 0 && continue
    push!(exprs, eq.rhs)
    push!(pIdxs, k)
  end
  isempty(exprs) && return nothing
  local keep = _inlineObservedRows!(exprs, reducedSystem, states, params, iv)
  filter!(keep) do i
    local own = string(params[pIdxs[i]])
    !any(v -> string(v) == own, Symbolics.get_variables(exprs[i]))
  end
  isempty(keep) && return nothing
  local pFunc = try
    local fExpr = Symbolics.build_function(exprs[keep], states, params, iv; expression = Val{true}, cse = true)
    _exprToRTGFunction(fExpr[1])
  catch e
    OMBackend._fallback(e, :_initialParameterAssignments_2, impact = :result)
    @debug "DirectRHS: could not build the initial parameter assignments" exception = e
    return nothing
  end
  return (pFunc, pIdxs[keep])
end


#= Parameters whose binding reads `roots` (directly or through other such
   parameters): `spring.spring.c = spring.c` for the free spring.c of the MSL
   InitSpringConstant. _buildParamVector resolved them from the roots' start
   values; a root the initialization changes must carry them along. Returns
   `(depFunc, depIdxs, depNames)` with `depFunc(p)` the values of `p[depIdxs]`
   from the roots in `p` and `depNames` every dependent (in `params` or not),
   or nothing. A dependent that still reads a name outside `params` after
   substitution is left out. =#
function _parameterDependents(pars, params, roots::AbstractSet{String};
                              resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  isempty(roots) && return nothing
  local valByName = OrderedDict{String, Any}()
  for (k, v) in pars
    valByName[string(k)] = v isa Symbolics.Num ? Symbolics.unwrap(v) : v
  end
  local reads = Dict{String, Vector{String}}(
    name => (v isa Number ? String[] : String[string(x) for x in Symbolics.get_variables(v)])
    for (name, v) in valByName)
  local deps = OrderedSet{String}()
  local grew = true
  while grew
    grew = false
    for name in keys(valByName)
      (name in deps || name in roots) && continue
      any(x -> x in roots || x in deps, reads[name]) || continue
      push!(deps, name)
      grew = true
    end
  end
  isempty(deps) && return nothing
  local paramIdx = Dict{String, Int}(string(p) => i for (i, p) in enumerate(params))
  #= Each dependent in terms of the roots and the parameters that stay: other
     dependents substituted by their bindings (bounded, bindings are acyclic),
     names outside `params` by their resolved values. =#
  local byVar = Dict{Any, Any}()
  for (k, v) in pars
    local nm = string(k)
    local uk = k isa Symbolics.Num ? Symbolics.unwrap(k) : k
    if nm in deps
      byVar[uk] = valByName[nm]
    elseif !haskey(paramIdx, nm) && resolvedParams !== nothing && haskey(resolvedParams, nm)
      byVar[uk] = resolvedParams[nm]
    end
  end
  local depIdxs = Int[]
  local exprs = Any[]
  for d in deps
    local k = get(paramIdx, d, 0)
    k == 0 && continue
    local ex = valByName[d]
    for _ in 1:(length(deps) + 1)
      local next = Symbolics.substitute(ex, byVar)
      isequal(next, ex) && break
      ex = next
    end
    if !all(v -> haskey(paramIdx, string(v)), Symbolics.get_variables(ex))
      @debug "DirectRHS: dependent parameter $(d) reads a name outside the parameters; left at its value"
      continue
    end
    push!(depIdxs, k)
    push!(exprs, ex)
  end
  isempty(depIdxs) && return nothing
  local depFunc = try
    local fExpr = Symbolics.build_function(exprs, params; expression = Val{true})
    local fn = _exprToRTGFunction(fExpr[1])
    #= Probed once: a failing binding must not break every init evaluation. =#
    fn(ones(length(params)))
    fn
  catch e
    OMBackend._fallback(e, :_parameterDependents)
    @debug "DirectRHS: could not build the dependent parameters" exception = e
    return nothing
  end
  return (depFunc, depIdxs, deps)
end

#= The literal derivative initial equations `der(w) = c` whose w is not an
   unknown: `(ws, cs)`. =#
function _observedDerivativeInitEquations(reducedSystem, states;
                                          resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local exprs = Any[]
  local tgts = Float64[]
  local initEqs = ModelingToolkit.initialization_equations(reducedSystem)
  local stateStrs = Set{String}(string(st) for st in states)
  for eq in initEqs
    local lhs = Symbolics.unwrap(eq.lhs)
    (SymbolicUtils.iscall(lhs) && SymbolicUtils.operation(lhs) isa Symbolics.Differential) || continue
    local w = SymbolicUtils.arguments(lhs)[1]
    string(w) in stateStrs && continue
    local c = _tryToFloat64(eq.rhs; resolvedParams=resolvedParams)
    c === nothing && continue
    push!(exprs, w)
    push!(tgts, c)
  end
  return (exprs, tgts)
end

#= Literal derivative initial equations `der(w) = c` on an observed w (not an
   unknown; those are derivative targets): w inlined to the unknowns, its
   gradient w.r.t. the unknowns and the independent variable (explicit time)
   by the DAG differentiation, compiled as the nonzeros. Returns
   `(nzFunc!, rows, cols, n, targets)`: row k's der(w) is
   Σ nz * (col <= n ? u̇[col] : 1). Nothing when there are none or they
   cannot be reduced or differentiated. =#
function _observedDerivativeTargets(reducedSystem, states, params, iv;
                                    resolvedParams::Union{Dict{String,Float64},Nothing}=nothing)
  local (exprs, tgts) = _observedDerivativeInitEquations(reducedSystem, states; resolvedParams=resolvedParams)
  isempty(exprs) && return nothing
  local keep = _inlineObservedRows!(exprs, reducedSystem, states, params, iv)
  length(keep) < length(exprs) &&
    @warn "DirectRHS: $(length(exprs) - length(keep)) derivative initial equations on observed variables not reduced to the unknowns; left out"
  isempty(keep) && return nothing
  exprs = exprs[keep]
  tgts = tgts[keep]
  try
    local Jw = _dagSparseJacobian(exprs, vcat(collect(states), [iv]))
    local (I, J, _) = Symbolics.SparseArrays.findnz(Jw)
    local nzSym = collect(Symbolics.SparseArrays.nonzeros(Jw))
    local fExpr = Symbolics.build_function(nzSym, states, params, iv; expression = Val{true}, cse = true)
    return (_exprToRTGFunction(_demoteWideNumericLiterals!(fExpr[2])), I, J, length(states), tgts)
  catch e
    OMBackend._fallback(e, :_observedDerivativeTargets, impact = :result)
    @warn "DirectRHS: derivative initial equations on observed variables not differentiated; left out" exception = e
    return nothing
  end
end

#= The RHS's explicit time derivative ∂F/∂t (states and parameters fixed),
   the tgrad of the problem: by the DAG differentiation when every row has a
   derivative rule and it evaluates finite at the entry, else by a central
   difference (as the solver's own). Rosenbrock methods need it in each step;
   their finite difference in t has a step growing with t, and its error in
   the MSL SMEE machines' 50 Hz sources (~0.2 V/s at t = 2.8) set a floor
   the algebraic step control chased until maxiters (SMEE_DOL stopped at
   2.835 s after 1e6 steps). =#
function _buildTimeDerivative(rhs_list, states, params, iv, rhsFunc, u0, p_vec, t0)
  #= Kill switch OMBACKEND_TGRAD=false: no tgrad (the solver differences in t itself). =#
  OMBackend.envSwitch("OMBACKEND_TGRAD") || return nothing
  local tg = _explicitTimeDerivative(rhs_list, states, params, iv)
  if tg !== nothing
    #= At the entry guesses (before the initialization): only a row the RHS
       evaluates finite may not come out non-finite. =#
    local ok = try
      local dT = similar(u0)
      local du = similar(u0)
      Base.invokelatest(tg, dT, u0, p_vec, t0)
      rhsFunc(du, u0, p_vec, t0)
      all(i -> isfinite(dT[i]) || !isfinite(du[i]), eachindex(dT))
    catch e
      OMBackend._fallback(e, :_buildTimeDerivative)
      false
    end
    ok && return tg
    @debug "DirectRHS: the symbolic time derivative does not evaluate at the entry; a finite difference instead"
  else
    @debug "DirectRHS: no symbolic time derivative; a finite difference instead"
  end
  return (dT, u, p, t) -> begin
    local h = cbrt(eps(Float64)) * max(1.0, abs(t))
    local f2 = similar(dT)
    rhsFunc(dT, u, p, t + h)
    rhsFunc(f2, u, p, t - h)
    @. dT = (dT - f2) / (2h)
    nothing
  end
end

#= The time derivative of rows that do not read t. =#
_zeroTimeDerivative(dT, u, p, t) = (fill!(dT, 0); nothing)

#= ∂F/∂t of the rows `exprs` (explicit time) by the DAG differentiation, as an
   in-place `f!(out, u, p, t)`; nothing when a row has no derivative rule (a
   time table). =#
function _explicitTimeDerivative(exprs, states, params, iv)
  try
    #= The states as variables (x(t) would otherwise be a function of t), only t's column. =#
    local n = length(states)
    local Jt = _dagSparseJacobian(exprs, vcat(collect(states), [iv]); columns = BitSet((n + 1,)))
    Symbolics.SparseArrays.nnz(Jt) == 0 && return _zeroTimeDerivative
    local col = Any[Jt[i, n + 1] for i in 1:length(exprs)]
    local fExpr = Symbolics.build_function(col, states, params, iv; expression = Val{true}, cse = true)
    return _exprToRTGFunction(_demoteWideNumericLiterals!(fExpr[2]))
  catch e
    OMBackend._fallback(e, :_explicitTimeDerivative)
    @debug "DirectRHS: explicit time derivative not symbolic; a finite difference instead" exception = e
    return nothing
  end
end

#= The time derivatives of the algebraic unknowns (zero mass-matrix rows of a
   diagonal mass matrix) at (u, p, t): differentiating the algebraic rows
   0 = F_a(u, t) gives J_aa ż_a = -(J_ad u̇_d + ∂F_a/∂t), with u̇_d = F_d / m_d
   from `du` = F(u). J from the symbolic sparse Jacobian (kept sparse, written
   into `J`), ∂F_a/∂t from `ft!` or, without it (or when it throws or is not
   finite), by a second-order one-sided difference (a time switch at t sees
   its right limit there). The raw RHS with p
   fixed: parameters are constant in time. A singular J_aa takes the
   least-squares solution; NaN if anything throws. =#
function _algebraicDerivatives!(J, rhs, jac, ft!, mm, algIdx, difIdx, u, du, p, t)
  try
    jac(J, u, p, t)
    local Ft = Vector{Float64}(undef, length(algIdx))
    local symbolic = ft! !== nothing && try
      ft!(Ft, u, p, t)
      all(isfinite, Ft)
    catch e
      OMBackend._fallback(e, :_algebraicDerivatives!_1)
      false
    end
    if !symbolic
      local h = cbrt(eps(Float64)) * max(1.0, abs(t))
      local d1 = similar(du)
      local d2 = similar(du)
      rhs(d1, u, p, t + h)
      rhs(d2, u, p, t + 2h)
      Ft .= (-3 .* du[algIdx] .+ 4 .* d1[algIdx] .- d2[algIdx]) ./ (2h)
    end
    local udot_d = Float64[du[i] / mm[i, i] for i in difIdx]
    local b = J[algIdx, difIdx] * udot_d .+ Ft
    local Jaa = J[algIdx, algIdx]
    local F = LinearAlgebra.lu(Jaa; check = false)
    return LinearAlgebra.issuccess(F) ? -(F \ b) : -(LinearAlgebra.qr(Matrix(Jaa), LinearAlgebra.ColumnNorm()) \ b)
  catch e
    OMBackend._fallback(e, :_algebraicDerivatives!_2)
    return fill(NaN, length(algIdx))
  end
end

#= The literal values of the if-equation relations at an initial state:
   `eval!(pv, u, t)` writes 1.0/0.0 into their ifCond parameters and returns
   whether one changed. From the codegen's entries (ifCond, observed names,
   crossing functions, observed -> Bool); an entry whose ifCond is not a
   parameter is left out, and when the crossing functions cannot be observed
   there are none. Nothing when none remain. =#
function _initialRelationLiterals(reducedSystem, params, entries)
  isempty(entries) && return nothing
  local paramIdx = Dict{String, Int}(string(p) => i for (i, p) in enumerate(params))
  local kept = Any[]
  local zcs = Any[]
  for (sym, names, fns, lit) in entries
    local k = get(paramIdx, string(sym), 0)
    k == 0 && continue
    push!(kept, (k, names, length(zcs) + 1, length(fns), lit))
    append!(zcs, fns)
  end
  isempty(kept) && return nothing
  local f = try
    _buildObservedFunction(reducedSystem, zcs)
  catch e
    OMBackend._fallback(e, :_initialRelationLiterals, impact = :result)
    @debug "DirectRHS: relation literals not observable; the initialization keeps the compiled ifConds" exception = e
    return nothing
  end
  local eval! = (pv, u, t; finiteOnly::Bool = false) -> begin
    #= The observed function and the literals come from the model's eval:
       called in the latest world (a re-initialization can run in an older). =#
    local vals = Base.invokelatest(f, u, pv, t)
    local changed = false
    for (k, names, off, n, lit) in kept
      local nt = NamedTuple{Tuple(names)}(Tuple(Float64(vals[off + i - 1]) for i in 1:n))
      finiteOnly && !all(isfinite, values(nt)) && continue
      local v = Base.invokelatest(lit, nt) ? 1.0 : 0.0
      changed |= pv[k] != v
      pv[k] = v
    end
    changed
  end
  return (eval! = eval!, idxs = Int[k for (k, _...) in kept])
end

#= The discrete clusters at initialization, for `_settleInitialDiscretes!`:
   the codegen's clusters (instances of their own) bound to the reduced
   system (the problem does not exist yet), their members' start values
   (index => value, those that evaluate) and per cluster the start values its
   pre() reads take (MLS 8.6: pre(v) = v.start). Nothing when there are none
   or they cannot be observed. =#
function _initialDiscreteClusters(clusters, reducedSystem, startOf::AbstractDict{String, Float64})
  isempty(clusters) && return nothing
  local starts = Pair{Int, Float64}[]
  local preStarts = Vector{Union{Nothing, Float64}}[]
  try
    for c in clusters
      _bindToSystem!(c, reducedSystem)
      for (n, k) in zip(c.names, c.memberIndex)
        k == 0 || !haskey(startOf, n) || push!(starts, k => startOf[n])
      end
      push!(preStarts, Union{Nothing, Float64}[get(startOf, string(r), nothing)
                                               for r in c.reads[(c.nOperands + 1):(c.nOperands + c.nPre)]])
    end
  catch e
    OMBackend._fallback(e, :_initialDiscreteClusters, impact = :result)
    @debug "DirectRHS: discrete clusters not observable at initialization; not settled" exception = e
    return nothing
  end
  return (clusters = clusters, starts = starts, preStarts = preStarts)
end

#= The members whose cluster body, at state u (relations literal, MLS 8.5;
   pre() the start values, else the values in u; initial() true), gives
   another value than u holds: index => value. Local buffers: a
   re-initialization may run while another one does. =#
function _initialDiscreteChanges(dc, u, pv)
  local point = (u = u, p = pv, t = 0.0)
  local changes = Pair{Int, Float64}[]
  for (c, preStart) in zip(dc.clusters, dc.preStarts)
    local v = Base.invokelatest(c.values, u, pv, 0.0)
    local zs = similar(c.zs)
    Base.invokelatest(c.crossings!, zs, u, pv, 0.0)
    local rel = Bool[_literal(zs, k, c.strict[k]) for k in eachindex(c.rel)]
    local pre = [something(s, x) for (s, x) in zip(preStart, _preValues(c, v))]
    local vals = Base.invokelatest(c.body, point, _operands(c, v), pre, rel, copy(rel), true)
    vals === nothing && continue
    for (i, k) in enumerate(c.memberIndex)
      (k == 0 || u[k] == vals[i]) && continue
      push!(changes, k => Float64(vals[i]))
    end
  end
  return changes
end

#= Initial fixpoint of the discrete clusters' mixed systems, as OpenModelica
   solves them (MLS 8.6: a discrete equation outside a when holds at the
   initial solution, its relations at their literal values, pre() at the
   start values). The initial algorithm set the members from their
   equations at the continuous start values before the solve (an ideal
   diode's `off = s < 0` at s = 0: conducting); with a fixed inductor
   current or capacitor voltage that entry can make the solve fail (MSL
   HBridge_RL: 340 kA) or reach another fixpoint of the mixed system
   (MultiPhase Rectifier: every diode conducting, 5.8e6 V).
   When an entry member differs from its start value (`startPath`), from the
   entry with the members at their start values: solved, the members take
   their bodies' values at the solution, solved again until they stay.
   Otherwise, or when that does not settle, from the first solution the same
   way. A result is taken only if the differential states `kept` stay where
   the solve it started from had them (OpenModelica keeps states without an
   initial equation at their start: a diode with a free capacitor voltage
   must not move it to reach the tie s = 0); else, and on a cycle, a pass
   limit, a throw or a failed solve, the first solution stands and the event
   iteration at the start settles the members as before.
   Returns `(u, settled)`. =#
function _settleInitialDiscretes!(resolve, u0, uEntry, pv, dc, kept::Vector{Int};
                                  startPath::Bool = true, maxPasses::Int = 10)
  dc === nothing && return (u0, false)
  local trace = OMBackend.envSwitch("OMBACKEND_INIT_TRACE")
  local pvFirst = copy(pv)
  local memberIdx = unique!(Int[k for c in dc.clusters for k in c.memberIndex if k != 0])
  local keptAt = (u, ref) -> isapprox(u[kept], ref[kept]; rtol = 1e-6, atol = 1e-9)
  #= From a solution `u` of the solve that started at `ref`. =#
  local iterate = function (u, ref)
    local changes = _initialDiscreteChanges(dc, u, pv)
    local seen = Set{Vector{Float64}}()
    for _ in 1:maxPasses
      trace && println("[initdiscretes] members to change: ", changes)
      isempty(changes) && return u
      for (k, v) in changes
        u[k] = v
      end
      local key = u[memberIdx]
      key in seen && return nothing
      push!(seen, key)
      local ok = Ref(true)
      local un = resolve(copy(u), pv, ok)
      (ok[] && keptAt(un, ref)) || return nothing
      u = un
      changes = _initialDiscreteChanges(dc, u, pv)
    end
    return nothing
  end
  local attempt = function (f)
    try
      return f()
    catch e
      OMBackend._fallback(e, :_settleInitialDiscretes!, impact = :result)
      @debug "DirectRHS: the discrete clusters did not settle at initialization" exception = e
      return nothing
    end
  end
  if startPath && any(kv -> uEntry[kv.first] != kv.second, dc.starts)
    local settled = attempt() do
      local u = copy(uEntry)
      for (k, v) in dc.starts
        u[k] = v
      end
      local ok = Ref(true)
      local un = resolve(copy(u), pv, ok)
      trace && println("[initdiscretes] from the start values: ", ok[] ? "solved" : "not solved",
                       ", states kept: ", keptAt(un, u))
      (ok[] && keptAt(un, u)) ? iterate(un, u) : nothing
    end
    settled === nothing || return (settled, true)
    copyto!(pv, pvFirst)
  end
  local settled = attempt(() -> iterate(copy(u0), u0))
  settled === nothing || return (settled, true)
  trace && println("[initdiscretes] not settled; the first solution stands")
  copyto!(pv, pvFirst)
  return (u0, false)
end

#= Initial event iteration for the if-equation relations: solved with their
   entry values, the relations take their literal values at the solution
   and, while one changes, the initialization is solved again (from the last
   solution) for them. A cycle, a pass limit, a throw or a solve that does
   not converge keeps the first solution and the entry parameters (MSL
   EngineV6_analytic: the gas force's `v_rel < 0` was false at the entry,
   the steady-state filter settled on the wrong torque). =#
function _settleInitialRelations!(resolve, u0, pv, rels; maxPasses::Int = 5)
  rels === nothing && return u0
  local pvFirst = copy(pv)
  local uFirst = copy(u0)
  local seen = Set{Vector{Float64}}([pv[rels.idxs]])
  local u = u0
  local err = nothing
  for pass in 1:(maxPasses + 1)
    try
      rels.eval!(pv, u, 0.0) || return u
      pass > maxPasses && break
      local key = pv[rels.idxs]
      key in seen && break
      push!(seen, key)
      local ok = Ref(true)
      local un = resolve(copy(u), pv, ok)
      ok[] || break
      u = un
    catch e
      OMBackend._fallback(e, :_settleInitialRelations!, impact = :result)
      err = e
      break
    end
  end
  @debug "DirectRHS: the relations did not settle at initialization; kept the entry branches" exception = err
  copyto!(pv, pvFirst)
  return uFirst
end

#= The DAE init solve with the free parameters p[freeIdx] as unknowns: they
   are appended to u as algebraic unknowns (zero mass-matrix rows whose
   residuals are identically zero), so each phase may move them like a free
   algebraic variable; `follow!(p)` updates the parameters bound to them.
   Returns the states; p[freeIdx] keeps the solved values. =#
function _solveDAEInitializationFree!(u0, rhs, pv, mm, freeIdx::Vector{Int}, follow!;
                                      extra_residuals=nothing, kwargs...)
  isempty(freeIdx) && return _solveDAEInitialization!(u0, rhs, pv, mm; extra_residuals=extra_residuals, kwargs...)
  local n = length(u0)
  local nq = length(freeIdx)
  local setFree! = (p, u) -> begin
    for (j, k) in enumerate(freeIdx)
      p[k] = u[n + j]
    end
    follow! === nothing || follow!(p)
    nothing
  end
  #= Copies, not views: the generated functions would compile again for
     SubArray arguments. =#
  local uN = similar(u0)
  local duN = similar(u0)
  local rhsExt = (du, u, p, t) -> begin
    setFree!(p, u)
    copyto!(uN, 1, u, 1, n)
    rhs(duN, uN, p, t)
    copyto!(du, 1, duN, 1, n)
    fill!(view(du, (n + 1):(n + nq)), 0.0)
    nothing
  end
  local extraExt = extra_residuals === nothing ? nothing : (du, u) -> begin
    copyto!(uN, 1, u, 1, n)
    copyto!(duN, 1, du, 1, n)
    extra_residuals(duN, uN)
  end
  local mmExt = zeros(eltype(mm), n + nq, n + nq)
  mmExt[1:n, 1:n] .= mm
  local uExt = _solveDAEInitialization!(vcat(u0, pv[freeIdx]), rhsExt, pv, mmExt;
                                        extra_residuals=extraExt, kwargs...)
  setFree!(pv, uExt)
  return uExt[1:n]
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
    OMBackend._fallback(e, :_buildRHSExpression)
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
throws when the function runs. The Jacobian is `_dagSparseJacobian`'s; where
that has no derivative for an array construct, Symbolics' sparsejacobian is
tried when no RHS equation's tree exceeds `DIRECT_JAC_TREE_NODE_LIMIT` nodes.
Returns `(jacFunc, jacPrototype)`, or `(nothing, nothing)` when generation is
disabled, no Jacobian is found, or the probe fails.
"""
function _buildSparseJacobian(rhs_list, states, params, iv, u0, p_vec, t0)
  OMBackend.DIRECT_JAC_GENERATION[] || return (nothing, nothing)
  try
    local jacSym = try
      _dagSparseJacobian(rhs_list, states)
    catch e
      e isa _NoDagDerivative || rethrow()
      if !e.retry
        @debug "DirectRHS: no derivative for $(e.what); solver will finite-difference"
        return (nothing, nothing)
      end
      #= Symbolics differentiates the trees: bounded by their size. =#
      if any(ex -> _exprLargerThan(ex, OMBackend.DIRECT_JAC_TREE_NODE_LIMIT[]), rhs_list)
        @debug "DirectRHS: no DAG derivative ($(e.what)) and an RHS equation has more than $(OMBackend.DIRECT_JAC_TREE_NODE_LIMIT[]) tree nodes; solver will finite-difference"
        return (nothing, nothing)
      end
      Symbolics.sparsejacobian(rhs_list, states)
    end
    #= The nonzeros as a vector: build_function applies no CSE to a sparse
       matrix (MSL Engine1b_analytic's 12 nonzeros: over 50 million Expr
       nodes and 10 GB; as a vector 14,123 nodes). The solver's Jacobian has
       the prototype's structure, so they are its nzval. =#
    local nzSym = collect(Symbolics.SparseArrays.nonzeros(jacSym))
    local result = Symbolics.build_function(nzSym, states, params, iv;
                                            expression=Val{true}, cse=true)
    local nzFunc = _exprToRTGFunction(_demoteWideNumericLiterals!(result[2]))
    local jacProto = similar(jacSym, Float64)
    jacProto.nzval .= 0.0
    local nnz0 = length(nzSym)
    local jacFunc = (J, u, p, t) -> begin
      local nz = Symbolics.SparseArrays.nonzeros(J)
      #= The generated code writes by position, without bounds checks. =#
      length(nz) == nnz0 || throw(DimensionMismatch("Jacobian with $(length(nz)) stored entries, not $(nnz0)"))
      nzFunc(nz, u, p, t)
      nothing
    end
    local probe = copy(jacProto)
    jacFunc(probe, u0, p_vec, t0)
    @debug "DirectRHS: symbolic sparse Jacobian with $(length(jacProto.nzval)) structural nonzeros"
    return (jacFunc, jacProto)
  catch e
    OMBackend._fallback(e, :_buildSparseJacobian)
    @debug "DirectRHS: symbolic Jacobian generation failed; solver will finite-difference" exception=(e, catch_backtrace())
    return (nothing, nothing)
  end
end

#= An expression _dagSparseJacobian has no derivative for. `retry`: whether
   Symbolics' sparsejacobian may have one (array constructs, Differential,
   Integral); a call without a derivative rule or a callable symbolic leaves
   Symbolics with an unresolved derivative too. =#
struct _NoDagDerivative <: Exception
  what::String
  retry::Bool
end

#= The sparse symbolic Jacobian of `rhs_list` with respect to `states`,
   differentiating each expression as the DAG it is: a subexpression shared
   by several uses (the observed equations full_equations inlines, a
   multibody chain of frames) is differentiated once per state, and the
   derivatives share their subexpressions the same way. Symbolics'
   sparsejacobian differentiates the trees: MSL EngineV6_analytic's 17
   equations are 1,943 DAG nodes but 14.6 million tree nodes, and its 17x17
   Jacobian took 48 s and 15 GB. The structure is Symbolics'
   (jacobian_sparsity: an entry for each state an equation reads), the rules
   are its derivative rules (derivative_idx); throws _NoDagDerivative where
   there is none. =#
function _dagSparseJacobian(rhs_list, states; columns::Union{Nothing, AbstractSet{Int}} = nothing)
  local T = Symbolics.VartypeT
  local stateIdx = Dict{Any, Int}(Symbolics.unwrap(s) => j for (j, s) in enumerate(states))
  #= Whole-array uses of scalarized states would read no state here. =#
  local arrParents = Set{Any}(SymbolicUtils.arguments(u)[1] for u in keys(stateIdx)
                              if SymbolicUtils.iscall(u) && SymbolicUtils.operation(u) === getindex)
  local depsMemo = IdDict{Any, BitSet}()
  local deps = x -> begin
    local hit = get(depsMemo, x, nothing)
    hit === nothing || return hit
    local j = get(stateIdx, x, 0)
    local d = BitSet()
    if j > 0
      push!(d, j)
    elseif x in arrParents
      throw(_NoDagDerivative("array use of a scalarized state", true))
    elseif SymbolicUtils.iscall(x)
      for a in SymbolicUtils.arguments(x)
        union!(d, deps(a))
      end
    end
    depsMemo[x] = d
    return d
  end
  #= Combined as plain terms, not with + and *: canonical sums and products
     merge their operands' terms, copying what the DAG shares (EngineV6_analytic
     grew to 16 GB that way too). Symbolic 0 and 1 (the rules of sign, floor,
     comparisons) fold like numbers. =#
  local isZero = x -> (local v = SymbolicUtils.unwrap_const(x); v isa Number && iszero(v))
  local isOne = x -> (local v = SymbolicUtils.unwrap_const(x); v isa Number && isone(v))
  local plus = (a, b) -> isZero(a) ? b : isZero(b) ? a : SymbolicUtils.term(+, a, b; vartype = T)
  local times = (a, b) -> (isZero(a) || isZero(b)) ? 0 : isOne(a) ? b : isOne(b) ? a :
                          SymbolicUtils.term(*, a, b; vartype = T)
  local sumOf = terms -> foldl(plus, terms; init = 0)
  local isSum = x -> SymbolicUtils.isadd(x) || (SymbolicUtils.isterm(x) && SymbolicUtils.operation(x) === (+))
  local isProduct = x -> SymbolicUtils.ismul(x) || (SymbolicUtils.isterm(x) && SymbolicUtils.operation(x) === (*))
  local derivative
  derivative = (x, j, memo) -> begin
    j in deps(x) || return 0
    local hit = get(memo, x, nothing)
    hit === nothing || return hit
    local r = if get(stateIdx, x, 0) == j
      1
    elseif isSum(x)
      sumOf(Any[derivative(a, j, memo) for a in SymbolicUtils.arguments(x) if j in deps(a)])
    elseif isProduct(x)
      local args = SymbolicUtils.arguments(x)
      sumOf(Any[times(foldl(times, (args[k] for k in eachindex(args) if k != i); init = 1),
                      derivative(args[i], j, memo))
                for i in eachindex(args) if j in deps(args[i])])
    elseif SymbolicUtils.isdiv(x)
      local (num, den) = SymbolicUtils.arguments(x)
      local dn = derivative(num, j, memo)
      local dd = derivative(den, j, memo)
      local a = isZero(dn) ? 0 : SymbolicUtils.term(/, dn, den; vartype = T)
      local b = isZero(dd) ? 0 : SymbolicUtils.term(/, times(num, dd), SymbolicUtils.term(^, den, 2; vartype = T); vartype = T)
      isZero(b) ? a : isZero(a) ? SymbolicUtils.term(-, b; vartype = T) : SymbolicUtils.term(-, a, b; vartype = T)
    elseif SymbolicUtils.ispow(x) && !(j in deps(SymbolicUtils.arguments(x)[2]))
      local (b, e) = SymbolicUtils.arguments(x)
      local ev = SymbolicUtils.unwrap_const(e)
      local bPow = ev isa Number ? (isone(ev - 1) ? b : SymbolicUtils.term(^, b, ev - 1; vartype = T)) :
                   SymbolicUtils.term(^, b, SymbolicUtils.term(-, e, 1; vartype = T); vartype = T)
      times(times(e, bPow), derivative(b, j, memo))
    elseif SymbolicUtils.isterm(x)
      local f = SymbolicUtils.operation(x)
      local args = SymbolicUtils.arguments(x)
      if f === ifelse || f === SymbolicUtils.ifelse_eager || f === SymbolicUtils.ifelse_branching
        local dt = derivative(args[2], j, memo)
        local df = derivative(args[3], j, memo)
        (isZero(dt) && isZero(df)) ? 0 : SymbolicUtils.term(f, args[1], dt, df; vartype = T)
      elseif f isa Symbolics.Differential || f isa Symbolics.Integral || f === getindex
        throw(_NoDagDerivative(string(f), true))
      elseif f isa SymbolicUtils.Operator
        #= Pre, Sample, Hold, Shift: new variables, as in Symbolics. =#
        0
      elseif f isa SymbolicUtils.BasicSymbolic
        throw(_NoDagDerivative(string(f), false))
      else
        local terms = Any[]
        for (i, a) in enumerate(args)
          j in deps(a) || continue
          local rule = Symbolics.derivative_idx(x, i)
          rule === nothing && throw(_NoDagDerivative(string(f), false))
          isZero(rule) && continue
          push!(terms, times(rule, derivative(a, j, memo)))
        end
        sumOf(terms)
      end
    else
      throw(_NoDagDerivative(string(typeof(x)), true))
    end
    memo[x] = r
    return r
  end
  local I = Int[]
  local J = Int[]
  local V = Symbolics.Num[]
  local memos = [IdDict{Any, Any}() for _ in states]
  for (i, ex) in enumerate(rhs_list)
    local x = Symbolics.unwrap(ex)
    for j in deps(x)
      columns === nothing || j in columns || continue
      push!(I, i)
      push!(J, j)
      push!(V, Symbolics.Num(derivative(x, j, memos[j])))
    end
  end
  return Symbolics.SparseArrays.sparse(I, J, V, length(rhs_list), length(states))
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
    #= A fixed start (finalInitialValues). One that reads other unknowns (`y = 2x`)
       is solved by the initialization (_hasSolvedInitializationRows): 0.0 is its
       placeholder. One that reads none and cannot be evaluated is refused, not
       0.0 (a guess may be). =#
    local val = _tryToFloat64(pair.second; resolvedParams=resolvedParams)
    if val === nothing
      _readsUnknowns(pair.second, resolvedParams) ||
        OMBackend.unsupported("a start value that cannot be evaluated", "$(keyStr) = $(pair.second)")
      @warn "DirectRHS: a start value reads other variables; 0.0 until the initialization solves it" key = keyStr val = pair.second
      val = 0.0
    end
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
    resolvedParams = _resolveParamValues(pars; used = Set{String}(string.(params)))
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
    local v = Symbolics.value(expr)
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
    catch _e
      OMBackend._fallback(_e, :_evalSymbolicFunctionCall_2)
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
    _resolveParamValues(pars; used = nothing)

Resolve parameter values by iteratively substituting known numeric values
into symbolic parameter expressions. Returns a Dict{String, Float64}
mapping parameter names to their numeric values. One of `used` (the system's
parameters, by name; all when nothing) that does not resolve is refused.
"""
function _resolveParamValues(pars; used::Union{Nothing, Set{String}} = nothing)
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
      catch _e
        OMBackend._fallback(_e, :_resolveParamValues_1)
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
        catch _e
          OMBackend._fallback(_e, :_resolveParamValues_2)
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
          catch _e
            OMBackend._fallback(_e, :_resolveParamValues_3)
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
          catch _e
            OMBackend._fallback(_e, :_resolveParamValues_4)
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
          catch _e
            OMBackend._fallback(_e, :_resolveParamValues_5)
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
        catch _e
          #= A name of the expression not resolved yet (another parameter,
             a model function) is undefined in the fresh module. =#
          OMBackend._fallback(_e, :_resolveParamValues_6; expect = UndefVarError)
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

  #= A parameter of the system (`used`) without a (finite) number would be 0.0
     in the simulation: refused. =#
  local unresolved = filter(e -> used === nothing || e[3] in used, symbolicByKey)
  isempty(unresolved) ||
    OMBackend.unsupported("parameters that do not resolve to finite numbers",
                          join(("$(kStr) = $(uv)" for (_, uv, kStr) in unresolved), ", "))

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
    #= Without the system's own events (if-equation relations, whens) the
       result is wrong (it went on with the custom callbacks only, a warning:
       MSL CauerLowPassSC on 2026-09-27; none in the 425 models now). =#
    isempty(ModelingToolkit.continuous_events(reducedSystem)) && isempty(ModelingToolkit.discrete_events(reducedSystem)) ||
      OMBackend.unsupported("events of the reduced system that process_events cannot build", sprint(showerror, ex))
    OMBackend._fallback(ex, :_extractAndMergeEventCallbacks)
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
    withProblemCallbacks(problem, buildCallbacks, callbacks) -> problem

`problem` with the model's final event callbacks `callbacks`, ahead of them the
events of the MTK system it was built with (its callbacks other than those of
`buildCallbacks`, the set it was built from). A solve uses the problem's alone:
solve() merges `problem.kwargs[:callback]` with the callbacks it is given, so a
callback in both would run twice. The structural (VSS) paths build their
problems from the build's callbacks, which stay the full set.
"""
function withProblemCallbacks(problem, buildCallbacks, callbacks)
  local built = Base.IdSet{Any}(_callbackList(buildCallbacks))
  local own = filter(c -> !(c in built), _callbackList(get(problem.kwargs, :callback, nothing)))
  #= Lazily: a remake of an MTK problem with trivial initialization would
     otherwise run it now, at build time, instead of in the solve. =#
  return ModelingToolkit.SciMLBase.remake(problem; callback = DiffEqBase.CallbackSet(own..., _callbackList(callbacks)...),
                                          lazy_initialization = true)
end
_callbackList(::Nothing) = Any[]
_callbackList(cb::DiffEqBase.CallbackSet) = Any[cb.continuous_callbacks..., cb.discrete_callbacks...]
_callbackList(cb::ModelingToolkit.SciMLBase.DECallback) = Any[cb]

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

#= SciMLBase 3's VectorContinuousCallback has no affect_neg!: its affect! is
   called once per event instant with the events of all components, 0 (none),
   +1 (upcrossing) or -1 (downcrossing). A scalar sub gets its affect! or
   affect_neg! by the sign; a vector sub gets its slice. =#
const _VCC_HAS_AFFECT_NEG = hasfield(ModelingToolkit.SciMLBase.VectorContinuousCallback, :affect_neg!)

struct _MergedContinuousEvents{S}
  subs::S
  offsets::Vector{Int}
  nsub::Int
end

function (m::_MergedContinuousEvents)(integrator, events::AbstractVector)::Nothing
  local SB = ModelingToolkit.SciMLBase
  for k in 1:m.nsub
    local s = m.subs[k]
    local slice = view(events, (m.offsets[k] + 1):m.offsets[k + 1])
    if s isa SB.VectorContinuousCallback
      any(!iszero, slice) && s.affect!(integrator, slice)
    else
      local e = slice[1]
      local aff = e > 0 ? s.affect! : e < 0 ? s.affect_neg! : nothing
      aff === nothing || aff(integrator)
    end
  end
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
  local initF = _MergedContinuousInitialize(subs, nsub)
  local FW = DiffEqBase.FunctionWrapper
  local condW = FW{Nothing, Tuple{AbstractVector{Float64}, AbstractVector{Float64}, Float64, Any}}(condF)
  #= Always FunctionWrapper-wrap the merged initialize (even when every sub uses the
     default) so the VCC's initialize param is the SAME model-independent type whether or
     not a sub carries a custom initialize -> chua and the synthetic bake share one type. =#
  local initW = FW{Nothing, Tuple{Any, Any, Any, Any}}(initF)
  local vcc = if _VCC_HAS_AFFECT_NEG
    local affW = FW{Nothing, Tuple{Any, Int}}(_MergedContinuousAffect(subs, offsets, lens, nsub, false))
    local affNW = FW{Nothing, Tuple{Any, Int}}(_MergedContinuousAffect(subs, offsets, lens, nsub, true))
    SB.VectorContinuousCallback(condW, affW, affNW, total;
                                initialize = initW, rootfind = rootfind, save_positions = savePos)
  else
    local evW = FW{Nothing, Tuple{Any, Any}}(_MergedContinuousEvents(subs, offsets, nsub))
    SB.VectorContinuousCallback(condW, evW, total;
                                initialize = initW, rootfind = rootfind, save_positions = savePos)
  end
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
  catch _e
    OMBackend._fallback(_e, :_tryToFloat64_1)
    return nothing
  end
  #= A start value reading time: the build's start time, 0 (u0 is built once;
     _startTimeGuard refuses another start time). It was 0.0 with a warning. =#
  local timeVars = filter(v -> string(v) == "t", freeVars)
  if !isempty(timeVars)
    return _tryToFloat64(Symbolics.substitute(unwrapped, Dict{Any, Any}(v => 0.0 for v in timeVars));
                         resolvedParams = resolvedParams)
  end
  if isempty(freeVars)
    local str = string(val)
    local f = tryparse(Float64, str)
    f !== nothing && return f
    #= A Boolean start value (MSL FluxTubes' asc(start = true)): a symbolic
       constant true that became 0.0, false. =#
    local b = tryparse(Bool, str)
    b !== nothing && return Float64(b)
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
        local vextract = Symbolics.value(rv)
        vextract isa Number && return Float64(vextract)
        #= A call the substitution does not fold (`floor(2.7)` of the start
           `integer(p27)`: the fixed state started at 0.0). =#
        local folded = _evalConstantTerm(rv)
        folded isa Number && return Float64(folded)
      end
    end
  end
  isempty(freeVars) && (local folded = _evalConstantTerm(unwrapped); folded isa Number) && return Float64(folded)
  return nothing
end

"""
    substituteDerivatives(expr, eqs)

`expr` with each derivative `D(x)` replaced by the right side of its explicit
equation `D(x) ~ f` among `eqs`; nothing when a derivative has none.
"""
function substituteDerivatives(expr, eqs)
  local SU = Symbolics.SymbolicUtils
  local subs = Dict{Any, Any}()
  for eq in eqs
    local l = Symbolics.unwrap(eq.lhs)
    SU.iscall(l) && SU.operation(l) isa Symbolics.Differential && (subs[l] = eq.rhs)
  end
  local r = isempty(subs) ? expr : Symbolics.substitute(expr, subs)
  return _hasDerivative(Symbolics.unwrap(r)) ? nothing : r
end

function _hasDerivative(@nospecialize(x))::Bool
  local SU = Symbolics.SymbolicUtils
  SU.iscall(x) || return false
  SU.operation(x) isa Symbolics.Differential && return true
  return any(_hasDerivative, SU.arguments(x))
end

#= Whether a value reads a variable that is neither time nor a resolved parameter. =#
function _readsUnknowns(@nospecialize(val), resolvedParams)::Bool
  local u = val isa Symbolics.Num ? Symbolics.unwrap(val) : val
  u isa Number && return false
  local vars = try
    Symbolics.get_variables(u)
  catch e
    OMBackend._fallback(e, :_readsUnknowns)
    return true
  end
  return any(v -> string(v) != "t" && (resolvedParams === nothing || !haskey(resolvedParams, string(v))), vars)
end

#= A symbolic term without variables, evaluated by applying its operations to
   its evaluated arguments; nothing where it has a variable. =#
function _evalConstantTerm(@nospecialize(x))
  local u = x isa Symbolics.Num ? Symbolics.unwrap(x) : x
  u isa Number && return u
  local v = Symbolics.value(u)
  v isa Number && return v
  Symbolics.SymbolicUtils.iscall(u) || return nothing
  local args = Any[_evalConstantTerm(a) for a in Symbolics.SymbolicUtils.arguments(u)]
  any(a -> a === nothing, args) && return nothing
  return try
    Symbolics.SymbolicUtils.operation(u)(args...)
  catch e
    OMBackend._fallback(e, :_evalConstantTerm)
    nothing
  end
end
