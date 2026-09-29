#= MTK code generation: start conditions, initial equations and algorithms, initial derivatives. =#

"""
  Generates the initial value for the equations.
  Algebraics without an explicit `start =` and without `fixed = true` are
  always skipped — MTK's init solver supplies defaults.
  States and OCC vars emit `0.0` defaults so MTK ODEProblem has a value for
  every unknown, unless `skipDefaultStateStarts` is true (used in the
  final-guess pass when an explicit user start is already pinned elsewhere).
"""
function createStartConditionsEquationsMTK(states::Vector,
                                        algebraics::Vector,
                                        simCode::SimulationCode.SIM_CODE;
                                        skipDefaultStateStarts::Bool = false)::Vector{Expr}
  local algInit = getStartConditionsMTK(algebraics, simCode; skipDefaultStarts = true)
  local stateInit = getStartConditionsMTK(states, simCode; skipDefaultStarts = skipDefaultStateStarts)
  local initialEquations = simCode.initialEquations
  local ieqInit = generateInitialEquations(initialEquations, simCode)
  #=
    Start with the start conditions above.
    Generate the equations in order afterwards
  =#
  #= Place the initial equations last =#
  return vcat(algInit, stateInit, ieqInit)
end

"""
  Generates initial equations as Symbolics `lhs ~ rhs` Equation forms suitable
  for passing to MTK's `initialization_eqs` kwarg of `System(...)`. Unlike the
  `=>` pair form (which acts as a guess only), `~` form is a real constraint
  that MTK's initialization solver must satisfy at t=0. Required for models
  with `InitialOutput` init mode (e.g. PID controllers' integrator state).
"""
function generateInitialEquationsAsConstraints(initialEqs, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local result = Expr[]
  for ieq in initialEqs
    if ieq isa BDAE.COMPLEX_EQUATION || ieq isa BDAE.ARRAY_EQUATION || ieq isa SimulationCode.ARRAY_EQUATION
      @debug "[MTK GEN: initialConstraints] skipping $(typeof(ieq)) (record/array constraints not yet lowered to scalar `~` form)"
      continue
    end
    if isParametricOnlyEquation(ieq, simCode)
      continue
    end
    local ieqLhsDAE = SimulationCode.toDAEExp(ieq.lhs)
    local ieqRhsDAE = SimulationCode.toDAEExp(ieq.rhs)
    local lhs = try
      expToJuliaExpMTK(ieqLhsDAE, simCode)
    catch err
      OMBackend._fallback(err, :initialConstraintLhs)
      @warn "[CODEGEN: initialConstraints] failed to lower LHS; constraint dropped" lhs=ieqLhsDAE err
      continue
    end
    local rhs = try
      @match ieqRhsDAE begin
        DAE.CREF(DAE.CREF_IDENT("time", _, _), _) => expToJuliaExpMTK(ieqRhsDAE, simCode)
        DAE.CREF(__) => begin
          local crefAsStr = string(ieqRhsDAE)
          if haskey(simCode.stringToSimVarHT, crefAsStr)
            local simCodeVar = last(simCode.stringToSimVarHT[crefAsStr])
            if SimulationCode.isStateOrAlgebraic(simCodeVar)
              expToJuliaExpMTK(ieqRhsDAE, simCode)
            elseif SimulationCode.hasBindingExp(simCodeVar)
              evalSimCodeParameter(simCodeVar, simCode)
            else
              expToJuliaExpMTK(ieqRhsDAE, simCode)
            end
          else
            expToJuliaExpMTK(ieqRhsDAE, simCode)
          end
        end
        _ => evalDAE_Expression(ieqRhsDAE, simCode)
      end
    catch err
      OMBackend._fallback(err, :initialConstraintRhs)
      @warn "[CODEGEN: initialConstraints] failed to lower RHS; constraint dropped" rhs=ieqRhsDAE err
      continue
    end
    push!(result, :($lhs ~ $rhs))
  end
  return result
end

"""
  Generates initial equations.
  Currently unsorted unless they are sorted before being passed to the simulation code phase.
"""
function generateInitialEquations(initialEqs, simCode::SimulationCode.SIM_CODE; parameterAssignment = true)::Vector{Expr}
  local initialEqsExps = Expr[]
  for ieq in initialEqs
    #= COMPLEX_EQUATION/ARRAY_EQUATION should have been expanded before this point =#
    if ieq isa BDAE.COMPLEX_EQUATION || ieq isa BDAE.ARRAY_EQUATION || ieq isa SimulationCode.ARRAY_EQUATION
      error("generateInitialEquations: unexpected unexpanded $(typeof(ieq)) in initial equations — this is a compiler bug upstream")
    end
    #= Skip parametric-only initial equations (already solved by solveParametricInitialEquations!) =#
    if isParametricOnlyEquation(ieq, simCode)
      continue
    end
    local ieqLhsDAE = SimulationCode.toDAEExp(ieq.lhs)
    local ieqRhsDAE = SimulationCode.toDAEExp(ieq.rhs)
    #= LHS will typically be a variable. Don't have to be though.. =#
    lhs = expToJuliaExpMTK(ieqLhsDAE, simCode)
    rhs = @match ieqRhsDAE begin
      #= `time` is the independent variable and never appears in
         stringToSimVarHT. Route it directly through expToJuliaExpMTK
         which emits the Julia symbol `t` for it. Without this guard
         the generic DAE.CREF arm below indexes the HT with key
         `"time"` and throws KeyError. Surfaced by models like
         Modelica.Fluid.Examples.ControlledTankSystem.ControlledTanks
         whose initial equations contain `<var> = time`. =#
      DAE.CREF(DAE.CREF_IDENT("time", _, _), _) => begin
        expToJuliaExpMTK(ieqRhsDAE, simCode)
      end
      DAE.CREF(__) => begin
        #= Evaluate the right hand side at this point =#
        local crefAsStr = string(ieqRhsDAE)
        local simCodeVar = last(simCode.stringToSimVarHT[crefAsStr])
        local res = if SimulationCode.isStateOrAlgebraic(simCodeVar)
          expToJuliaExpMTK(ieqRhsDAE, simCode)
        elseif SimulationCode.hasBindingExp(simCodeVar)
          evalSimCodeParameter(simCodeVar, simCode)
        else
          #= Parameter without binding (fixed=false): leave as symbol =#
          expToJuliaExpMTK(ieqRhsDAE, simCode)
        end
      end
      #= For more complicated expressions, we do local constant folding. =#
      _ => begin
        res = evalDAE_Expression(ieqRhsDAE, simCode)
        res
      end
    end
    if parameterAssignment
      push!(initialEqsExps,
            quote
              $lhs => $rhs
            end)
    else
      push!(initialEqsExps,
            quote
              $lhs = $rhs
            end)
    end
  end
  return initialEqsExps
end

"""
  Given a vector of variables and the simulation code
  extracts the start attributes to generate initial conditions.

If `skipDefaultStarts` is true, variables without explicit start values are skipped.
When false, variables without start values get default 0.0 initialization.
"""
function getStartConditionsMTK(vars::Vector, simCode::SimulationCode.SIM_CODE; skipDefaultStarts = false)::Vector{Expr}
  local startExprs::Vector{Expr} = Expr[]
  local residuals = simCode.residualEquations
  local ht::Dict = simCode.stringToSimVarHT
  local missingStartWarnings = OrderedSet{String}()
  if length(vars) == 0
    return Expr[]
  end
  for var in vars
    (index, simVar) = ht[var]
    varName = simVar.name
    local simVarType = simVar.varKind
    local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
    () = @match optAttributes begin
      SOME(attributes) => begin
        () = @match (attributes.start, attributes.fixed) begin
          (SOME(DAE.CREF(start)), SOME(__)) || (SOME(DAE.CREF(start)), _)  => begin
            #= Delegate to expToJuliaExpMTK so DATA_STRUCTURE / PARAMETER /
               subscripted CREFs are all handled uniformly. The previous
               two-branch split would emit `pars[name]` for non-subscripted
               CREFs, which only works when the referenced var is a
               PARAMETER (in `pars`). DATA_STRUCTURE constants and
               int/enum vars reclassified by Causalize are not in `pars`. =#
            push!(startExprs,
                  quote
                    $(Symbol("$varName")) => $(expToJuliaExpMTK(DAE.CREF(start, DAE.T_REAL(MetaModelica.Nil())), simCode))
                  end)
            continue
          end
          (SOME(start), SOME(fixed)) || (SOME(start), _)  => begin
            push!(startExprs,
                  quote
                    $(Symbol("$varName")) => $(expToJuliaExpMTK(start, simCode))
                  end)
            continue
          end
          (NONE(), SOME(fixed)) => begin
            #= `fixed = true` with no `start` pins the var at 0.0; honour even when
               default-skipping is on. `fixed = false` / non-Bool: MTK's init solver
               handles it, so skip emission in skip mode. =#
            local _fixedTrue = fixed isa DAE.BCONST && fixed.bool
            if skipDefaultStarts && !_fixedTrue
              continue
            end
            push!(startExprs, :($(Symbol(varName)) => 0.0))
            continue
          end
          (NONE(), NONE()) || (_, _) => begin
            #= No start value specified, default to 0.0 =#
            if !skipDefaultStarts
              push!(missingStartWarnings, varName)
              push!(startExprs, :($(Symbol(varName)) => 0.0))
            end
            continue
          end
        end
      end
      NONE() where {!skipDefaultStarts} => begin
        #=
        If no attribute. Let it default to zero.
        This branch should only be taken for compiler generated variables.
        =#
        push!(startExprs, :($(Symbol(varName)) => 0.0))
        continue
      end
      _ => begin
        continue
      end
    end
  end
  if OMBackend.WARN_MISSING_START_VALUES[] && !isempty(missingStartWarnings)
    local warningList = sort!(collect(missingStartWarnings))
    local maxShown = 20
    local shown = warningList[1:min(end, maxShown)]
    local omitted = length(warningList) - length(shown)
    local summary = "Assumed starting value of 0.0 for $(length(warningList)) variable(s): " * join(shown, ", ")
    if omitted > 0
      summary *= ", ... (+$(omitted) more)"
    end
    @warn summary
  end
  return startExprs
end

"""
  Emit `lhs ~ rhs` constraint Equations for state vars with `fixed=true` and an
  explicit `start`. Goes into `initialization_eqs` so MTK pins them at t=0
  rather than treating them as soft `guesses` the iteration may override.
"""
function getFixedStartConstraintsMTK(vars::Vector, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local result::Vector{Expr} = Expr[]
  if isempty(vars)
    return result
  end
  local ht::Dict = simCode.stringToSimVarHT
  for var in vars
    (index, simVar) = ht[var]
    local varName = simVar.name
    local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
    local startExp = @match optAttributes begin
      SOME(attributes) => @match (attributes.start, attributes.fixed) begin
        (SOME(s), SOME(DAE.BCONST(true))) => s
        _ => nothing
      end
      _ => nothing
    end
    if startExp === nothing
      continue
    end
    push!(result, :($(Symbol(varName)) ~ $(expToJuliaExpMTK(startExp, simCode))))
  end
  return result
end

"""
  Emit `Expr`s that push init-algorithm-derived (state => value) pairs into the
  `finalInitialValues` vector inside Model(). Mirrors `emitInitAlgConstraintAppends`
  but the push target is the u0-pair list (consumed by `ODEProblem(...; u0 = ...)`)
  rather than `initialization_eqs`. The merge is required because the pure-ODE
  branch of the ODEProblem build skips MTK's init solver, so the init-eq alone
  would not propagate the init-algorithm value into u0.
"""
#= The names an initial algorithm assigns and reads: from its DAE statements
   when it has them (an `initial algorithm` section), else from its when
   operators (the body of `when initial()` or `when {c, initial()}`). A model
   can have both kinds: the MSL SignalPWM's sawtooth and ZeroOrderHold. =#
function _collectInitAlgNames!(lhsNames, rhsNames, ia)
  if isempty(ia.daeStatements)
    for op in ia.statements
      _collectInitAlgLhsRhsCrefs!(lhsNames, rhsNames, op)
    end
  else
    for s in ia.daeStatements
      _collectInitAlgLhsRhsCrefsDAE!(lhsNames, rhsNames, s)
    end
  end
  return lhsNames
end

#= The initial algorithms the early pass (`__runInitialAlgorithmEarly!`)
   evaluates; its results become start values and initialization constraints
   of the system. In a model with an `initial algorithm` section only those:
   the bodies of `when initial()` / `when {c, initial()}` run in the runtime
   pass (`__runInitialAlgorithm!`) there, as before both kinds were
   supported together. In the early pass their values (switch controls of
   the MSL QS IMC_Transformer) became hard initial conditions of the reduced
   system and its switching event at t = 2 failed. =#
_earlyInitialAlgorithms(simCode) =
  any(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms) ?
    filter(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms) : simCode.initialAlgorithms

function emitInitAlgU0Appends(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local appends::Vector{Expr} = Expr[]
  isempty(simCode.initialAlgorithms) && return appends
  local ht::Dict = simCode.stringToSimVarHT
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  for ia in _earlyInitialAlgorithms(simCode)
    _collectInitAlgNames!(lhsNames, rhsNames, ia)
  end
  for name in lhsNames
    haskey(ht, name) || continue
    local (_, sv) = ht[name]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      continue
    end
    local qn = QuoteNode(Symbol(name))
    #= Replace (not append) any existing start-attribute entry so a lifted
       discrete with both a start value and an init-algorithm value does not
       leave a duplicate key in the u0 pair list (which drops other entries
       during splitInitialValues). =#
    push!(appends, :(if haskey(_algResults, $(qn))
                       filter!(_p -> !isequal(_p.first, $(Symbol(name))), fiv)
                       push!(fiv, $(Symbol(name)) => _algResults[$(qn)])
                     end))
  end
  return appends
end

"""
  Emit `Expr`s that conditionally push init-algorithm-derived constraints into
  the local `_eqs` vector inside `_buildInitialConstraintEqs`. Each emitted line
  looks like `haskey(_algResults, :T_start) && push!(_eqs, T_start ~ _algResults[:T_start])`.
"""
function emitInitAlgConstraintAppends(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local appends::Vector{Expr} = Expr[]
  isempty(simCode.initialAlgorithms) && return appends
  local ht::Dict = simCode.stringToSimVarHT
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  for ia in _earlyInitialAlgorithms(simCode)
    _collectInitAlgNames!(lhsNames, rhsNames, ia)
  end
  for name in lhsNames
    haskey(ht, name) || continue
    local (_, sv) = ht[name]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      continue
    end
    local qn = QuoteNode(Symbol(name))
    push!(appends, :(haskey(_algResults, $(qn)) &&
                     push!(_eqs, $(Symbol(name)) ~ _algResults[$(qn)])))
  end
  return appends
end

#= Lower a single BDAE.WhenOperator from an INITIAL_ALGORITHM body to a Julia
   expression suitable for module-top eval inside `__runInitialAlgorithm!`.
   Parameter CREFs are folded to their literal bindings via _substituteBoundParameters
   before lowering with the algorithmic (non-MTK) translator, so no Symbolics
   bindings are needed at init time.

   An ASSIGN to a parameter and a REINIT write `LATEST_PROBLEM`; an ASSIGN to
   another variable is collected in `_hard`, which simulate turns into hard
   start values of the unknowns (`remake(prob; u0 = …)`). The function runs
   after the ODEProblem is constructed (see `simulate(...)` in the generated
   module), so LATEST_PROBLEM is in scope. Without this, the LHS state stayed
   at its default (0) — e.g. `T_start := startTime + count*period` in the
   trapezoid signal source was silently dropped, breaking every model that
   relies on `initial algorithm` to seed states. =#
#= der(x) in a runtime initial algorithm (the initial() arm of a relation
   on a derivative, MSL FluxTubes' `asc = der(Hstat) > 0`) reads the
   derivative from the problem (_initialDerivative); there is no `der`
   function. =#
function _initialDerivativeReads(ex)
  ex isa Expr || return ex
  if ex.head === :call && length(ex.args) == 2 && ex.args[1] === :der
    local x = _blockValue(ex.args[2])
    x isa Symbol && return :(OMBackend.CodeGeneration._initialDerivative(LATEST_PROBLEM, $(QuoteNode(x))))
  end
  return Expr(ex.head, map(_initialDerivativeReads, ex.args)...)
end

_callsDer(ex) = ex isa Expr && ((ex.head === :call && ex.args[1] === :der) || any(_callsDer, ex.args))

#= The value of a `begin #= line =# x end` the algorithm lowering wraps
   expressions in. =#
function _blockValue(ex)
  (ex isa Expr && ex.head === :block) || return ex
  local body = filter(a -> !(a isa LineNumberNode), ex.args)
  return length(body) == 1 ? _blockValue(body[1]) : ex
end

#= The derivative of the variable `name` at `problem`'s start values: the
   runtime initial algorithm runs before the solve. A discrete cluster's
   start pass evaluates such a relation again on the solved state. =#
function _initialDerivative(problem, name::Symbol)
  local sys = problem.f.sys
  local d = ModelingToolkit.Differential(ModelingToolkit.get_iv(sys))(getproperty(sys, name))
  local f = _buildObservedFunction(sys, d)
  return Float64(f(problem.u0, problem.p, first(problem.tspan)))
end

function _initialWhenOpToJulia(wStmt, simCode::SimulationCode.SIM_CODE,
                               renamedNames::OrderedSet{String} = OrderedSet{String}())
  local sub = e -> _substituteBoundParameters(e, simCode)
  local lowerAlg = e -> _initialDerivativeReads(_renameAlgIdentifiers(
    _resolveModelicaCallTargets(AlgorithmicCodeGeneration.expToJuliaExpAlg(sub(e))),
    renamedNames,
    ""))
  local crefName = cr -> SimulationCode.DAE_identifierToString(cr)
  if wStmt isa BDAE.NORETCALL || wStmt isa SimulationCode.NORETCALL
    return :( $(lowerAlg(wStmt.exp)); nothing )
  elseif wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
    # SimulationCode.ASSIGN.left is ::Exp post-migration; convert to DAE for the @match.
    local leftDAE = wStmt isa SimulationCode.ASSIGN ? SimulationCode.toDAEExp(wStmt.left) : wStmt.left
    local name = @match leftDAE begin
      DAE.CREF(cr, _) => crefName(cr)
      _ => nothing
    end
    if name === nothing
      return :( $(lowerAlg(wStmt.right)); nothing )
    end
    local sym = Symbol(name)
    local isParam = haskey(simCode.stringToSimVarHT, name) &&
                    let (_, sv) = simCode.stringToSimVarHT[name]
                      sv.varKind isa SimulationCode.PARAMETER ||
                      sv.varKind isa SimulationCode.ARRAY_PARAMETER
                    end
    if isParam
      return :( $(sym) = $(lowerAlg(wStmt.right)); LATEST_PROBLEM.ps[$(QuoteNode(sym))] = $(sym); nothing )
    else
      return :( $(sym) = $(lowerAlg(wStmt.right));
         try
           _hard[getproperty(LATEST_REDUCED_SYSTEM, $(QuoteNode(sym)))] = $(sym)
         catch _e
           OMBackend._fallback(_e, :initialWhenHardStart)
           nothing
         end;
         nothing )
    end
  elseif wStmt isa BDAE.REINIT || wStmt isa SimulationCode.REINIT
    local name = crefName(wStmt.stateVar)
    local sym = Symbol(name)
    return :( $(sym) = $(lowerAlg(wStmt.value)); LATEST_PROBLEM[$(QuoteNode(sym))] = $(sym); nothing )
  elseif wStmt isa BDAE.ASSERT || wStmt isa SimulationCode.ASSERT
    local cond = lowerAlg(wStmt.condition)
    local msg = lowerAlg(wStmt.message)
    return :(if !($cond); @warn "Modelica assert() during init" message=$(msg); end)
  elseif wStmt isa BDAE.TERMINATE || wStmt isa SimulationCode.TERMINATE
    local msg = lowerAlg(wStmt.message)
    return :(@info "Modelica terminate() during init" message=$(msg))
  end
  throw(ErrorException("_initialWhenOpToJulia: unsupported variant $(typeof(wStmt))"))
end

#= Translate a single `BDAE.WhenOperator` from an init-algorithm body into a
   Julia statement suitable for the procedural body of
   `__runInitialAlgorithmEarly!`. ASSIGN emits `_alg_<lhs> = <rhs>` (with
   `local` on the first occurrence of that LHS); RHS identifiers are renamed
   via `_renameAlgIdentifiers` so they bind to the let-block locals rather
   than to module-level Symbolics bindings of the same name. =#
function _initialWhenOpToJuliaEarly(wStmt, simCode::SimulationCode.SIM_CODE,
                                    renamedNames::OrderedSet{String}, seenLHS::OrderedSet{String})
  local sub = e -> _substituteBoundParameters(e, simCode)
  local lowerAlg = e -> _renameAlgIdentifiers(
    _resolveModelicaCallTargets(AlgorithmicCodeGeneration.expToJuliaExpAlg(sub(e))),
    renamedNames)
  local crefName = cr -> SimulationCode.DAE_identifierToString(cr)
  if wStmt isa BDAE.NORETCALL || wStmt isa SimulationCode.NORETCALL
    return :( $(lowerAlg(wStmt.exp)); nothing )
  elseif wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
    # SimulationCode.ASSIGN.left is ::Exp post-migration; convert to DAE for the @match.
    local leftDAE = wStmt isa SimulationCode.ASSIGN ? SimulationCode.toDAEExp(wStmt.left) : wStmt.left
    local name = @match leftDAE begin
      DAE.CREF(cr, _) => crefName(cr)
      _ => nothing
    end
    if name === nothing
      return :( $(lowerAlg(wStmt.right)); nothing )
    end
    local algSym = Symbol("_alg_" * name)
    if name in seenLHS
      return :( $(algSym) = $(lowerAlg(wStmt.right)); nothing )
    end
    push!(seenLHS, name)
    return :( local $(algSym) = $(lowerAlg(wStmt.right)); nothing )
  elseif wStmt isa BDAE.ASSERT || wStmt isa SimulationCode.ASSERT
    local cond = lowerAlg(wStmt.condition)
    local msg = lowerAlg(wStmt.message)
    return :(if !($cond); @warn "Modelica assert() during init (early)" message=$(msg); end)
  elseif wStmt isa BDAE.TERMINATE || wStmt isa SimulationCode.TERMINATE
    local msg = lowerAlg(wStmt.message)
    return :(@info "Modelica terminate() during init (early)" message=$(msg))
  end
  return :( nothing )
end

#= Modelica function calls in algorithm code are `Base.invokelatest(Name, ...)`
   with Name bound in OMBackend.CodeGeneration (createModelicaFunctionWrapper),
   which a function body runs in. The early initial algorithm runs in the
   model's module, where Name is undefined: qualify it. (MSL
   WriteRealMatrixToFile: `when initial() then success1 := writeRealMatrix(...)`
   raised UndefVarError, swallowed, and success1..4 stayed false.) =#
function _qualifyInvokedFunctions(ex)
  ex isa Expr || return ex
  local args = map(_qualifyInvokedFunctions, ex.args)
  if ex.head === :call && length(args) >= 2 && args[1] == :(Base.invokelatest) && args[2] isa Symbol
    args[2] = Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(args[2]))
  end
  return Expr(ex.head, args...)
end

"""
    generateInitialAlgorithmEarlyFunction(simCode) -> Expr

Emit `function __runInitialAlgorithmEarly!() -> Dict{Symbol, Float64}` that
executes the `initial algorithm` bodies procedurally at module-load time
(Modelica §11.4: statements run sequentially, the LHS final value becomes the
variable's initial value).

When `simCode.initialAlgorithms[i].daeStatements` is non-empty for any body,
the procedural body is lowered via `AlgorithmicCodeGeneration.generateStatements`
— the same path used for regular Modelica algorithm sections and function
bodies, with full STMT_IF / STMT_FOR / STMT_WHILE / STMT_ASSERT / STMT_REINIT
support. The resulting Julia AST is then rewritten by `_renameAlgIdentifiers`
to prefix every cref name with `_alg_`, so the locals do not collide with the
Symbolics `Num` bindings of the same name living in the surrounding model
scope. When `daeStatements` is empty (e.g. older callers that only provide a
`Vector{BDAE.WhenOperator}`), the legacy flat-WhenOperator translator
`_initialWhenOpToJuliaEarly` is used as a fallback.

The body is wrapped in `let time = 0.0 ... end`. Non-LHS crefs read on the
RHS get a pre-seeded `_alg_<name>` from the SimVar's `start` attribute or
`0.0` (a parameter: its statically folded binding); every LHS starts at
`0.0`. After the body, each LHS final value is captured into the returned
`Dict{Symbol, Float64}` (an entry that is not a number is skipped).

A body that throws returns no results (a fallback); the runtime `remake`
path remains for state-cref-RHS reads whose post-init value differs from
the `start` attribute.
"""
function generateInitialAlgorithmEarlyFunction(simCode::SimulationCode.SIM_CODE)::Expr
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  for ia in _earlyInitialAlgorithms(simCode)
    _collectInitAlgNames!(lhsNames, rhsNames, ia)
  end
  local noEarlyPass = quote
    function __runInitialAlgorithmEarly!()
      return Dict{Symbol, Float64}()
    end
  end
  isempty(lhsNames) && isempty(rhsNames) && return noEarlyPass
  local ht = simCode.stringToSimVarHT
  local renamedNames = union(lhsNames, rhsNames)
  push!(renamedNames, "time")
  local prefetches = Expr[]
  local unfoldedParameter = false
  for name in setdiff(rhsNames, lhsNames)
    name == "time" && continue
    haskey(ht, name) || begin
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = 0.0))
      continue
    end
    local sv = ht[name][2]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      local paramLit = @match sv.varKind begin
        SimulationCode.PARAMETER(SOME(b)) => _foldParameterBindStatic(b, simCode)
        _ => nothing
      end
      if paramLit === nothing
        #= No static value (an array parameter, a binding that does not
           fold): no local, so a statement that reads it throws an
           UndefVarError, which the body's catch expects (below). The
           statements that do not read it still give their values. =#
        unfoldedParameter = true
        continue
      end
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(paramLit)))
      continue
    end
    local lit = _readStartAttributeAsLiteral(sv)
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(lit)))
  end
  for name in lhsNames
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = 0.0))
  end
  local stmts = Expr[]
  local seenLHS = copy(lhsNames)
  for ia in _earlyInitialAlgorithms(simCode)
    if isempty(ia.daeStatements)
      for op in ia.statements
        push!(stmts, _qualifyInvokedFunctions(_initialWhenOpToJuliaEarly(op, simCode, renamedNames, seenLHS)))
      end
    else
      for s in AlgorithmicCodeGeneration.generateStatements(ia.daeStatements)
        push!(stmts, _qualifyInvokedFunctions(_renameAlgIdentifiers(s, renamedNames)))
      end
    end
  end
  #= der() reads the problem, which the early pass runs before: a body that
     reads a derivative (a relation on one, MSL FluxTubes) is not run early.
     Running the other statements alone would turn the values after it into
     wrong initialization constraints. The runtime pass
     (`__runInitialAlgorithm!`, _initialDerivativeReads) runs the bodies of
     `when initial()`; the `initial algorithm` sections it does not run
     (_earlyInitialAlgorithms), so a derivative read in one has no effect,
     as before (the early body threw there). =#
  any(_callsDer, stmts) && return noEarlyPass
  local captures = Expr[]
  for name in lhsNames
    haskey(ht, name) || continue
    local sv = ht[name][2]
    if sv.varKind isa SimulationCode.PARAMETER ||
       sv.varKind isa SimulationCode.ARRAY_PARAMETER
      continue
    end
    local algSym = Symbol("_alg_" * name)
    local qn = QuoteNode(Symbol(name))
    push!(captures, :(try; _results[$(qn)] = Float64($(algSym)); catch _e; OMBackend._fallback(_e, :initAlgEarlyCapture); nothing; end))
  end
  return quote
    function __runInitialAlgorithmEarly!()
      local _results = Dict{Symbol, Float64}()
      try
        let time = 0.0
          $(prefetches...)
          $(stmts...)
          $(captures...)
        end
      catch _err
        OMBackend._fallback(_err, :initAlgEarlyBody; impact = :result,
                            expect = $(unfoldedParameter ? :UndefVarError : :(Union{})))
      end
      return _results
    end
  end
end

#= The Boolean-typed crefs an assignment's right-hand side reads, named as
   _collectInitAlgLhsRhsCrefs! names them. =#
function _collectBoolRhsCrefs!(names::OrderedSet{String}, op)
  (op isa BDAE.ASSIGN || op isa SimulationCode.ASSIGN) || return names
  local rhs = op.right isa SimulationCode.Exp ? SimulationCode.toDAEExp(op.right) : op.right
  Util.traverseExpBottomUp(rhs, (e, acc) -> begin
      if e isa DAE.CREF && e.ty isa DAE.T_BOOL
        push!(acc, SimulationCode.DAE_identifierToString(e.componentRef))
      end
      (e, acc)
    end, names)
  return names
end

"""
    generateInitialAlgorithmFunction(simCode) -> Expr

Emit a `function __runInitialAlgorithm!() ... end` whose body executes once
during initialization, lowered from `simCode.initialAlgorithms`. Parameter
literals are already baked into the body by `inlineParamsInInitialAlgorithms`
at SimCode construction time, so no module-scope parameter bindings are needed
here. Returns a no-op stub when the model has no `when initial()` clauses.
"""
function generateInitialAlgorithmFunction(simCode::SimulationCode.SIM_CODE)::Expr
  #= An algorithm with DAE statements: the early path emits control-flow-correct
     `initialization_eqs` for its LHS. The runtime `remake` here is built from
     the lossy WhenOperator flattening and would overwrite the init-eq result
     with the flat-first-branch value at simulate time; it is left out here. =#
  local whenAlgorithms = filter(ia -> isempty(ia.daeStatements), simCode.initialAlgorithms)
  if isempty(whenAlgorithms)
    return quote
      function __runInitialAlgorithm!()
        return Dict{Any, Any}()
      end
    end
  end
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  local boolNames = OrderedSet{String}()
  for ia in whenAlgorithms
    for op in ia.statements
      _collectInitAlgLhsRhsCrefs!(lhsNames, rhsNames, op)
      _collectBoolRhsCrefs!(boolNames, op)
    end
  end
  local renamedNames = union(lhsNames, rhsNames)
  push!(renamedNames, "time")
  local stmts = Expr[]
  for ia in whenAlgorithms
    for op in ia.statements
      push!(stmts, _initialWhenOpToJulia(op, simCode, renamedNames))
    end
  end
  if isempty(stmts)
    return quote
      function __runInitialAlgorithm!()
        return Dict{Any, Any}()
      end
    end
  end
  #= Pre-fetch every non-parameter RHS-referenced cref. Names that ALSO
     appear as LHS still need a fetch because Julia compiles `x = if c then
     v else x end` with `x` as a function-local: the else-branch reads `x`
     before the assignment completes and throws UndefVarError. Self-referential
     IFEXP shapes come from the algorithm lifter (algorithmSynthesis.jl) when a
     non-when algorithm contains an if/elseif chain whose else-branches
     preserve a discrete LHS's previous value. =#
  local fetches = Expr[]
  local ht = simCode.stringToSimVarHT
  for name in rhsNames
    name == "time" && continue
    local sv = nothing
    if haskey(ht, name)
      sv = ht[name][2]
      if sv.varKind isa SimulationCode.PARAMETER || sv.varKind isa SimulationCode.ARRAY_PARAMETER
        continue
      end
      if sv.varKind isa SimulationCode.DATA_STRUCTURE || sv.varKind isa SimulationCode.STRING
        local sym = Symbol(name)
        local boundSym = Symbol(sv.name)
        push!(fetches, Expr(:local, Expr(:(=), sym, :(getfield(@__MODULE__, $(QuoteNode(boundSym)))))))
        continue
      end
      #= DISCRETE vars (Logic/enum) are used as array indices. MTK
         initialisation may leave them at 0 which BoundsErrors on 1-based
         index vectors (e.g. INV3S's UX01Conv[iNV3S_enable]). Clamp to 1 as
         a band-aid until proper discrete-IC lowering lands. Not a Boolean:
         the clamp made false true (an ideal thyristor's `fire`, so it
         started conducting without a firing pulse). =#
      if sv.varKind isa SimulationCode.DISCRETE && !(name in boolNames) && !_isBoolDiscreteName(name, simCode)
        local sym = Symbol(name)
        push!(fetches, Expr(:local,
          Expr(:(=), sym,
            :(try
                let _g = ModelingToolkit.SciMLBase.getu(LATEST_PROBLEM, $(QuoteNode(sym)))
                  local _raw = _g(LATEST_PROBLEM)
                  local _v = if _raw isa Integer
                    Int(_raw)
                  elseif _raw isa Real
                    Int(round(Float64(_raw)))
                  else
                    1
                  end
                  _v < 1 ? 1 : _v
                end
              catch _e
                OMBackend._fallback(_e, :initAlgGetuFetch, impact = :result)
                1
              end))))
        continue
      end
    end
    #= Non-discrete SimVars (Real states, alg vars) and local algorithm
       temporaries not in the HT: fetch as Float64, no index clamp. =#
    local sym = Symbol(name)
    push!(fetches, Expr(:local,
      Expr(:(=), sym,
        :(try
            Float64(ModelingToolkit.SciMLBase.getu(LATEST_PROBLEM, $(QuoteNode(sym)))(LATEST_PROBLEM))
          catch _e
            OMBackend._fallback(_e, :initAlgGetuValue, impact = :result)
            0.0
          end))))
  end
  #= Shadow `Base.time` (a UNIX-time function) with the local Modelica `time`
     value, which is 0 at simulation init. Without this, init-algorithm bodies
     that reference `time` (e.g. trapezoid sources' `count := integer((time -
     startTime) / period)`) generate `time - <Float64>` and hit MethodError
     because `Base.time` is a function, not a number. =#
  return quote
    function __runInitialAlgorithm!()
      #= `_hard` collects (symbolic_var => value) pairs for each ASSIGN to a
         non-parameter variable. simulate() passes it to `remake(prob; u0=…,
         initializealg=NoInit())` so MTK treats the init-algorithm-computed
         values as hard initial conditions (Modelica §11.2), not guesses. =#
      local _hard = Dict{Any, Any}()
      let time = 0.0
        $(fetches...)
        $(stmts...)
      end
      return _hard
    end
  end
end
