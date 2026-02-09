function reconfiguration()
  #=
  Rerun OCC algorithms.
  Find the root variables of the OCC graph + the variables they assign and the root sources.
  That is the reference variables for the roots.
  =#
  @time (rootIndices, variablesToSetIdx, rootSources, variablestoReset) = returnRootIndices(cb.name,
                                                                                            cb,
                                                                                            integrator.u,
                                                                                            tspan,
                                                                                            problem)
  @assert length(rootIndices) == length(keys(rootSources)) "Root sources and indices must have the same length. Length was $(length(rootIndices)) == $(length(keys(rootSources)))"
  local newU0::Vector{Float64} = Float64[v for v in integrator.u]
  local stateVars = states(OMBackend.LATEST_REDUCED_SYSTEM)
  #= This is bad, do not use strings this way. =#
  local stateVarsAsStr = [replace(string(s), "(t)" => "") for s in stateVars]
  local OM_NameToMTKIdx = Dict()
  local rootKeys = [k for k in keys(rootSources)]
  local rootValues = [v for v in values(rootSources)]
  local variablesToSet = collect(Iterators.flatten([v for v in values(variablestoReset)]))
  local rootKeysToMTKIdx = indexin(rootKeys, stateVarsAsStr)
  local rootValsToMTKIdx = indexin(rootValues, stateVarsAsStr)
  local variablesToResetMTKIdx = indexin(variablesToSet, stateVarsAsStr)
  local rootToEquationMap::Dict{String, Symbolics.Equation} = Dict()
  for (i, rk) in enumerate(rootKeys)
    OM_NameToMTKIdx[rk] = rootKeysToMTKIdx[i]
  end
  for (i, rv) in enumerate(rootValues)
    #= This case is true if there is a constant value at the end =#
    if rootValsToMTKIdx[i] != nothing
      OM_NameToMTKIdx[rv] = rootValsToMTKIdx[i]
    else
      OM_NameToMTKIdx[rv] = Meta.parse(rv)
    end
  end
  for (i, vr) in enumerate(variablesToSet)
    OM_NameToMTKIdx[vr] = variablesToResetMTKIdx[i]
  end
  #=
  Start by setting the root variables we got from returnRootIndices.
  Each of these variables have a reference variable.
  =#
  for k in rootKeys
    rootStart = OM_NameToMTKIdx[k]
    rootSource = OM_NameToMTKIdx[rootSources[k]]
    if ! (rootSource isa Float64)
      integrator.u[rootStart] = integrator.u[rootSource]
      rootToEquationMap[k] = ~(0, stateVars[rootStart] - stateVars[rootSource])
    else
      integrator.u[rootStart] = rootSource
      rootToEquationMap[k] = ~(0, stateVars[rootStart] - rootSource)
    end
  end
  #=
  Get the variables of the system
  =#
  local equationDeps = ModelingToolkit.equation_dependencies(OMBackend.LATEST_REDUCED_SYSTEM)
  #= TODO: Not good should strive to fix the internal representations. =#
  local allEquationsAsStr = map((x)-> begin
                                  if !isempty(x)
                                    [replace(string(y), "(t)" => "") for y in x]
                                  end
                                end, equationDeps)
  local rootKeys = [string(k) for k in keys(rootSources)]
  local equationToAddMap = Dict()
  #=
  Find the equations to replace
  =#
  local oldEquations = equations(OMBackend.LATEST_REDUCED_SYSTEM)
  local assignedRoots = String[]
  for (i, eqVariables) in enumerate(allEquationsAsStr)
    if eqVariables !== nothing && length(eqVariables) == 2
      firstV = first(eqVariables)
      secondV = last(eqVariables)
      check1 = firstV in rootKeys && secondV in rootValues
      check2 = firstV in rootValues && secondV in rootKeys
      check3 = length(ModelingToolkit.difference_vars(oldEquations[i])) == 2
      if (check1 || check2) && check3
        push!(assignedRoots, last(eqVariables))
      end
    end
  end
  for (i, eqVariables) in enumerate(allEquationsAsStr)
    if eqVariables !== nothing && length(eqVariables) == 2
      local cand = first(eqVariables)
      if cand in rootKeys && ! (cand in assignedRoots)
        equationToAddMap[cand] = i
      end
    end
  end
  #= The equation indices are the locations in which we are to insert our new equations =#
  #= One assignment need to be changed. =#
  for k in rootKeys
    varsToSet = variablestoReset[k]
    val = OM_NameToMTKIdx[k]
    for v in varsToSet
      vIdx = OM_NameToMTKIdx[v]
      integrator.u[vIdx] = integrator.u[val]
    end
  end
  push!(solutions, integrator.sol)
  push!(oldSols, (integrator.sol, getSyms(problem), activeModeName))
  newEquations = Symbolics.Equation[]
  for eq in oldEquations
    push!(newEquations, eq)
  end
  for eq in keys(rootToEquationMap)
    if eq in assignedRoots
      continue
    end
    local replacementIdx = equationToAddMap[eq]
    local newEquation = rootToEquationMap[eq]
    newEquations[replacementIdx] = newEquation
  end
  @time newSystem = ODESystem(newEquations,
                              independent_variable(OMBackend.LATEST_REDUCED_SYSTEM),
                              states(OMBackend.LATEST_REDUCED_SYSTEM),
                              parameters(OMBackend.LATEST_REDUCED_SYSTEM);
                              name = Symbol(cb.name),
                              discrete_events = ModelingToolkit.discrete_events(OMBackend.LATEST_REDUCED_SYSTEM),
                              )
  newSystem = OMBackend.CodeGeneration.structural_simplify(newSystem)
  local discrete_events = newSystem.discrete_events
  newU0 = integrator.u
  events = if length(discrete_events) > 0
    RuntimeUtil.evalDiscreteEvents(discrete_events, newU0, i.t, newSystem)
  else
    []
  end
  for e in events
    newU0[e[1]] = e[2]
  end
  @time newProblem = ModelingToolkit.ODEProblem(
    newSystem,
    newU0,
    tspan,
    problem.p,
    #=
    TODO currently only handles a single structural callback.
    =#
    callback = callbackConditions
  )
  @time integrator = init(newProblem,
                          alg;
                          kwargs...)
  @time reinit!(integrator,
                newU0;
                t0 = i.t,
                reset_dt = true)

end
