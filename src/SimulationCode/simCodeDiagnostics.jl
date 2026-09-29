#= The index over-constraint diagnostic. =#

#= Collect cref names from a DAE exp, distinguishing a differentiated occurrence
   `der(x)` (recorded in `diff`) from a plain occurrence `x` (recorded in
   `plain`). This is the differential incidence the algebraic-level
   `collectCrefNames!` collapses. =#
function _collectDerAwareCrefs!(diff::OrderedSet{String}, plain::OrderedSet{String},
                                @nospecialize(exp))
  @match exp begin
    DAE.CALL(Absyn.IDENT("der"), args, _) => begin
      local inner = OrderedSet{String}()
      for a in args
        collectCrefNames!(inner, a)
      end
      union!(diff, inner)
    end
    DAE.CREF(cr, _) => push!(plain, DAE_identifierToString(cr))
    DAE.BINARY(exp1 = e1, exp2 = e2) => begin
      _collectDerAwareCrefs!(diff, plain, e1); _collectDerAwareCrefs!(diff, plain, e2)
    end
    DAE.UNARY(exp = e1) => _collectDerAwareCrefs!(diff, plain, e1)
    DAE.LUNARY(exp = e1) => _collectDerAwareCrefs!(diff, plain, e1)
    DAE.LBINARY(exp1 = e1, exp2 = e2) => begin
      _collectDerAwareCrefs!(diff, plain, e1); _collectDerAwareCrefs!(diff, plain, e2)
    end
    DAE.RELATION(exp1 = e1, exp2 = e2) => begin
      _collectDerAwareCrefs!(diff, plain, e1); _collectDerAwareCrefs!(diff, plain, e2)
    end
    DAE.IFEXP(expCond = c, expThen = t, expElse = e) => begin
      _collectDerAwareCrefs!(diff, plain, c); _collectDerAwareCrefs!(diff, plain, t)
      _collectDerAwareCrefs!(diff, plain, e)
    end
    DAE.CALL(expLst = as) => begin
      for a in as; _collectDerAwareCrefs!(diff, plain, a); end
    end
    DAE.CAST(exp = e) => _collectDerAwareCrefs!(diff, plain, e)
    DAE.ARRAY(array = lst) => begin
      for a in lst; _collectDerAwareCrefs!(diff, plain, a); end
    end
    DAE.ASUB(exp = e, sub = subs) => begin
      _collectDerAwareCrefs!(diff, plain, e)
      for s in subs; _collectDerAwareCrefs!(diff, plain, s); end
    end
    _ => ()
  end
  return nothing
end

"""
    _localizeOverconstraint(simCode) -> (unmatchedIdxs, nEqs, nHighestVars)

Differential-incidence localization. Builds the incidence with der(x) and x as
distinct columns and matches residual equations to their highest-order
derivative variables (`der_x` for a state x, `y` for an algebraic y) via
augmenting-path maximum matching. Returns indices of the structurally-unmatched
(over-constraining) equations. Pure; does not mutate the system.
"""
function _localizeOverconstraint(simCode::SIM_CODE)
  local ht = simCode.stringToSimVarHT
  local res = simCode.residualEquations
  local n_eqs = length(res)
  local stateNames = OrderedSet{String}()
  local algNames = OrderedSet{String}()
  for (k, (_, sv)) in pairs(ht)
    isUnknownVarKind(sv.varKind) || continue
    isState(sv) ? push!(stateNames, k) : push!(algNames, k)
  end
  local incidence = Vector{OrderedSet{String}}(undef, n_eqs)
  for (i, eq) in enumerate(res)
    local diff = OrderedSet{String}(); local plain = OrderedSet{String}()
    _collectDerAwareCrefs!(diff, plain, toDAEExp(eq.exp))
    local cols = OrderedSet{String}()
    for d in diff
      d in stateNames && push!(cols, "der_" * d)
      d in algNames && push!(cols, d)
    end
    for p in plain
      p in algNames && push!(cols, p)
    end
    incidence[i] = cols
  end
  local col_to_eq = Dict{String, Int}()
  local eq_to_col = fill("", n_eqs)
  function augmentOC!(eq_idx::Int, seen::OrderedSet{String})::Bool
    for col in incidence[eq_idx]
      col in seen && continue
      push!(seen, col)
      if !haskey(col_to_eq, col) || augmentOC!(col_to_eq[col], seen)
        col_to_eq[col] = eq_idx
        eq_to_col[eq_idx] = col
        return true
      end
    end
    return false
  end
  for i in 1:n_eqs
    augmentOC!(i, OrderedSet{String}())
  end
  local unmatched = Int[i for i in 1:n_eqs if isempty(eq_to_col[i])]
  return (unmatched, n_eqs, length(stateNames) + length(algNames))
end

"""
    indexOverconstraintDiagnostic(simCode) -> SIM_CODE

Standalone, gated diagnostic pass: when the differential-incidence localization
finds structurally-unmatched (over-constraining) equations, log them for
inspection. Returns simCode unchanged. Gated on `OMBACKEND_INDEX_DIAG`.
"""
function indexOverconstraintDiagnostic(simCode::SIM_CODE)::SIM_CODE
  OMBackend.envSwitch("OMBACKEND_INDEX_DIAG") || return simCode
  try
    local (unmatched, ne, nhv) = _localizeOverconstraint(simCode)
    @info "[SIMCODE: $(simCode.name): indexDiag] differential-incidence localization" n_eqs=ne nHighestVars=nhv nUnmatched=length(unmatched)
    for i in unmatched
      local diff = OrderedSet{String}(); local plain = OrderedSet{String}()
      _collectDerAwareCrefs!(diff, plain, toDAEExp(simCode.residualEquations[i].exp))
      local kindOf = v -> begin
        local e = get(simCode.stringToSimVarHT, v, nothing)
        e === nothing ? "?" : (isState(last(e)) ? "state" : (isAlgebraic(last(e)) ? "alg" : (isParameter(last(e)) ? "param" : "other")))
      end
      @info "[SIMCODE: $(simCode.name): indexDiag] unmatched [$i]" plain=[v * ":" * kindOf(v) for v in plain] der=collect(diff)
    end
  catch e
    @warn "[SIMCODE: $(simCode.name): indexDiag] threw" exception=(e, catch_backtrace())
  end
  return simCode
end
