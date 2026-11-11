#= A/B of the alias start value resolution (`CodeGeneration._resolveAliasFromExpr`, mtkExternals.jl): a reference
   implementation against the package's, on every candidate equation that model builds look at.

   Use it when the resolver changes: put the implementation to compare against in `_resolveAliasReference` below
   (as it is here: the one before the constant `ifelse` branches were taken without `simplify`), then, in a
   session with OM loaded:
     include("utilities/aliasStartValueAB.jl")
     aliasStartValueAB(["Modelica.Mechanics.Rotational.Examples.First"]; MSL_Version = "MSL:4.1.0")
     aliasStartValueDifferences()
   Each model is built anew and run to `stopTime`; a line per model gives the candidates and the differing ones.

   Session only: the file replaces the method of `resolveAliasInitialValue` in the loaded package.
   `aliasStartValueRestore()` puts the package's back. =#
import OMBackend

@eval OMBackend.CodeGeneration begin
  "Candidate equations seen by the A/B."
  const ALIAS_AB_CANDIDATES = Ref{Int}(0)
  "(state, reference result, package result, equation) of each candidate the two implementations differ on."
  const ALIAS_AB_DIFFERENCES = Tuple{String, Any, Any, String}[]

  "The implementation the package's is compared against."
  function _resolveAliasReference(diffState, @nospecialize(expr), varCount::Int, ivMap::Dict)
    local exprSub = Symbolics.substitute(expr, ivMap)
    local intercept = Symbolics.value(Symbolics.substitute(exprSub, Dict(diffState => 0)))
    local sumOnePoint = Symbolics.value(Symbolics.substitute(exprSub, Dict(diffState => 1)))
    if !(intercept isa Number && sumOnePoint isa Number)
      varCount <= 8 || return nothing
      _exprLargerThan(exprSub, 200) && return nothing
      local exprS = Symbolics.simplify(exprSub)
      any(v -> !isequal(v, diffState), Symbolics.get_variables(exprS)) && return nothing
      intercept = Symbolics.value(Symbolics.simplify(Symbolics.substitute(exprS, Dict(diffState => 0))))
      sumOnePoint = Symbolics.value(Symbolics.simplify(Symbolics.substitute(exprS, Dict(diffState => 1))))
    end
    (intercept isa Number && sumOnePoint isa Number) || return nothing
    local interceptF::Float64 = Float64(intercept)
    local slope::Float64 = Float64(sumOnePoint) - interceptF
    return iszero(slope) ? nothing : -interceptF / slope
  end

  function resolveAliasInitialValue(diffState, idx::AliasEqIndex, ivMap::Dict)
    for i in get(idx.byVar, variableName(diffState), Int[])
      ALIAS_AB_CANDIDATES[] += 1
      local resolved = try
        _resolveAliasFromExpr(diffState, idx.exprs[i], idx.varCounts[i], ivMap)
      catch e
        e isa Union{DivideError, OverflowError, InexactError} || rethrow()
        nothing
      end
      local reference = try
        _resolveAliasReference(diffState, idx.exprs[i], idx.varCounts[i], ivMap)
      catch e
        e isa Union{DivideError, OverflowError, InexactError} ? nothing : string("throws ", typeof(e))
      end
      isequal(reference, resolved) ||
        push!(ALIAS_AB_DIFFERENCES, (string(diffState), reference, resolved, string(idx.exprs[i])))
      resolved === nothing || return resolved
    end
    return nothing
  end
end

"Builds each model anew, runs it to `stopTime`, and prints its candidate equations and how many differ."
function aliasStartValueAB(models::Vector{String}; MSL_Version::String = "MSL:4.1.0", stopTime::Float64 = 1e-3)::Nothing
  local CG::Module = OMBackend.CodeGeneration
  for model::String in models
    local candidates::Int = CG.ALIAS_AB_CANDIDATES[]
    local differences::Int = length(CG.ALIAS_AB_DIFFERENCES)
    local t0::Float64 = time()
    local outcome::String = try
      local sol = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        Main.OM.simulate(model; MSL_Version = MSL_Version, stopTime = stopTime, overwriteCache = true)
      end
      string(sol.retcode)
    catch e
      string("FAIL ", first(sprint(showerror, e), 200))
    end
    println(rpad(model, 78), " ", outcome, ", candidates ", CG.ALIAS_AB_CANDIDATES[] - candidates, ", differ ",
            length(CG.ALIAS_AB_DIFFERENCES) - differences, " (", round(time() - t0; digits = 1), " s)")
  end
  return nothing
end

"Prints the candidates the two implementations differ on."
function aliasStartValueDifferences()::Nothing
  for (state::String, reference, resolved, equation::String) in OMBackend.CodeGeneration.ALIAS_AB_DIFFERENCES
    println(state, ": reference ", reference, ", package ", resolved, "\n  ", first(equation, 1500))
  end
  return nothing
end

"Puts the package's `resolveAliasInitialValue` back (as in mtkExternals.jl)."
function aliasStartValueRestore()::Nothing
  @eval OMBackend.CodeGeneration function resolveAliasInitialValue(diffState, idx::AliasEqIndex, ivMap::Dict)
    for i in get(idx.byVar, variableName(diffState), Int[])
      local resolved = try
        _resolveAliasFromExpr(diffState, idx.exprs[i], idx.varCounts[i], ivMap)
      catch e
        e isa Union{DivideError, OverflowError, InexactError} || rethrow()
        nothing
      end
      resolved === nothing || return resolved
    end
    return nothing
  end
  return nothing
end
