#= Discretes the causalization left unclassified (when-assigned targets). =#

"""
    _classifyAdditionalDiscreteVariables(simCode::SIM_CODE)::SIM_CODE

Reclassify any `ALG_VARIABLE` whose only definition lives inside a
`when`-equation as `DISCRETE`. This catches Real-valued variables that are
held between events (Modelica's classic `T_start := time` pattern inside a
`when`-clause) but were not picked up by the upstream Integer/enum discrete
classification, leaving them as algebraic unknowns with no defining residual.

Without this pass, the model has fewer equations than unknowns at
`structural_simplify` time and MTK raises `ExtraVariablesSystemException`.
After this pass the variable lands in `discreteVariables` during MTK
codegen, gets a `der(x) ~ 0` dummy, and the when-clause callback affect
has a state to update.

Detection: walk every `BDAE.WHEN_EQUATION` and collect the LHS variable name
of every `BDAE.ASSIGN` operator (and the `stateVar` of every `BDAE.REINIT`).
Any name in the set whose simvar is currently `ALG_VARIABLE` is
reclassified to `DISCRETE`. Variables that already have a non-algebraic
kind (state, parameter, discrete, occ, array, data structure) are left
alone.

No-op for VSS / submodel / metaModel variants where the
equation set is restructured at runtime.
"""
function _classifyAdditionalDiscreteVariables(simCode::SIM_CODE)::SIM_CODE
  if hasStructuralTransitions(simCode) || hasSubModels(simCode) ||
     hasMetaModel(simCode)
    @debug "[SIMCODE: $(simCode.name): classifyAdditionalDiscretes] skipped (VSS/multi-mode model)"
    return simCode
  end

  if isempty(simCode.whenEquations)
    return simCode
  end

  #= Step 1: collect every var name that appears as LHS of a when-ASSIGN
     or as the target of a when-REINIT. =#
  local whenLhsNames = OrderedSet{String}()
  for whenEq in simCode.whenEquations
    _collectWhenAssignTargets!(whenLhsNames, whenEq.whenEquation)
  end

  if isempty(whenLhsNames)
    return simCode
  end

  #= Step 2: reclassify ALG_VARIABLE -> DISCRETE for those names. =#
  local ht = simCode.stringToSimVarHT
  local reclassified = String[]
  for name in whenLhsNames
    haskey(ht, name) || continue
    local (idx, sv) = ht[name]
    if sv.varKind isa ALG_VARIABLE
      ht[name] = (idx, SIMVAR(sv.name, sv.index, DISCRETE(), sv.attributes))
      push!(reclassified, name)
    end
  end

  if !isempty(reclassified)
    @debug "[SIMCODE: $(simCode.name): classifyAdditionalDiscretes] reclassified $(length(reclassified)) algebraic variables to discrete (when-driven): $(reclassified)"
  end
  return simCode
end

#= Walk a BDAE.WhenEquation (WHEN_STMTS) tree, collecting every variable
   name that is assigned or reinit-ed inside. Recurses into elsewhen. =#
function _collectWhenAssignTargets!(names::OrderedSet{String}, whenEq)
  if whenEq isa WHEN_STMTS
    for stmt in whenEq.whenStmtLst
      if stmt isa ASSIGN
        local r = extractCrefName(stmt.left)
        if r !== nothing
          push!(names, r[1])
        end
      elseif stmt isa REINIT
        push!(names, DAE_identifierToString(stmt.stateVar))
      end
    end
    if whenEq.elsewhenPart !== nothing
      _collectWhenAssignTargets!(names, whenEq.elsewhenPart)
    end
    return nothing
  end
  @match whenEq begin
    BDAE.WHEN_STMTS(_, stmts, elsewhen) => begin
      for stmt in stmts
        @match stmt begin
          BDAE.ASSIGN(left = lhs) => begin
            local r = extractCrefName(lhs)
            if r !== nothing
              push!(names, r[1])
            end
          end
          BDAE.REINIT(stateVar = cr) => begin
            push!(names, DAE_identifierToString(cr))
          end
          _ => nothing
        end
      end
      if isSome(elsewhen)
        @match SOME(elseEq) = elsewhen
        _collectWhenAssignTargets!(names, elseEq)
      end
    end
    _ => nothing
  end
end
