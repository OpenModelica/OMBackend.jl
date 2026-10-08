#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF AGPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GNU AGPL
* VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the OSMC (Open Source Modelica Consortium)
* Public License (OSMC-PL) are obtained from OSMC, either from the above
* address, from the URLs:
* http://www.openmodelica.org or
* https://github.com/OpenModelica/ or
* http://www.ida.liu.se/projects/OpenModelica,
* and in the OpenModelica distribution.
*
* GNU AGPL version 3 is obtained from:
* https://www.gnu.org/licenses/licenses.html#GPL
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#

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
    #= Solved at the build (solveParametricInitialEquations!), unless a
       parameter of it is still unbound: then the initialization solves it. =#
    if isParametricOnlyEquation(ieq, simCode) && !_readsUnboundParameter(ieq, simCode)
      continue
    end
    local ieqLhsDAE = SimulationCode.toDAEExp(ieq.lhs)
    local ieqRhsDAE = SimulationCode.toDAEExp(ieq.rhs)
    local lhs = try
      expToJuliaExpMTK(ieqLhsDAE, simCode)
    catch err
      OMBackend._fallback(err, :initialConstraintLhs; only = UnsupportedLowering, impact = :result)
      @warn "[CODEGEN: initialConstraints] failed to lower LHS; constraint dropped" lhs=ieqLhsDAE err
      continue
    end
    local rhs = try
      _initialEquationRhs(ieqRhsDAE, simCode)
    catch err
      OMBackend._fallback(err, :initialConstraintRhs; only = UnsupportedLowering, impact = :result)
      @warn "[CODEGEN: initialConstraints] failed to lower RHS; constraint dropped" rhs=ieqRhsDAE err
      continue
    end
    push!(result, :($lhs ~ $rhs))
  end
  return result
end

#= The right side of an initial equation, lowered (both of its forms:
   generateInitialEquationsAsConstraints, generateInitialEquations). `time`
   (the independent variable, in no table: MSL ControlledTanks' `v = time`),
   an unknown and a parameter without a binding stay references; a bound
   parameter and an expression are evaluated at the build, except where they
   read a tunable parameter, directly or through a binding: `x = p` and
   `x = q` with q = 2 * p kept x(0) for every p. =#
function _initialEquationRhs(rhs, simCode::SimulationCode.SIM_CODE)
  return @match rhs begin
    DAE.CREF(DAE.CREF_IDENT("time", _, _), _) => expToJuliaExpMTK(rhs, simCode)
    DAE.CREF(__) => begin
      local entry = get(simCode.stringToSimVarHT, string(rhs), nothing)
      if entry === nothing || SimulationCode.isStateOrAlgebraic(last(entry)) || !SimulationCode.hasBindingExp(last(entry))
        expToJuliaExpMTK(rhs, simCode)
      elseif _readsTunableParameter(rhs, simCode)
        evalDAE_Expression(rhs, simCode; keepTunable = true)
      else
        evalSimCodeParameter(last(entry), simCode)
      end
    end
    _ => evalDAE_Expression(rhs, simCode; keepTunable = true)
  end
end

#= Whether an equation reads a parameter without a binding (a fixed=false one
   the build did not solve). =#
function _readsUnboundParameter(ieq, simCode::SimulationCode.SIM_CODE)::Bool
  local ht = simCode.stringToSimVarHT
  return any(equationSides(ieq)) do side
    any(Util.getAllCrefs(side)) do c
      local entry = get(ht, string(c), nothing)
      entry !== nothing && SimulationCode.isParameter(last(entry)) && !SimulationCode.hasBindingExp(last(entry))
    end
  end
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
    rhs = _initialEquationRhs(ieqRhsDAE, simCode)
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
  Emit `lhs ~ rhs` constraint Equations for state vars with `fixed=true` and a
  `start` (0 where it has none: the default start, MLS 4.9.1). Goes into
  `initialization_eqs` so MTK pins them at t=0 rather than treating them as soft
  `guesses` the iteration may override. A `v(fixed = true)` without a start was
  left out: v = xa + 1 started at 1, not 0 (OpenModelica: xa(0) = -1).
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
        (NONE(), SOME(DAE.BCONST(true))) => DAE.RCONST(0.0)
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
#= What the build evaluated at time 0, for a simulation from another start
   time. The parametric initial equations it solved that read `time` are
   refused (MSL Blocks.Math.Mean's t0 = time gave 0 with any start time); one
   reading an unbound parameter is solved by the initialization, at the start
   time. The early initial algorithms run again at the start time, refused
   only where a value differs (a Pulse's count is the same over its first
   period). Nothing when there is neither. =#
function _startTimeGuard(simCode::SimulationCode.SimCode)
  local what = String[]
  for ieq in simCode.initialEquations
    (isParametricOnlyEquation(ieq, simCode) && !_readsUnboundParameter(ieq, simCode)) || continue
    local sides = equationSides(ieq)
    any(_expMentionsTime, sides) && append!(what, (string(c) for side in sides for c in Util.getAllCrefs(side) if string(c) != "time"))
  end
  #= Start values reading time: the build's u0, for time 0. =#
  for (name, (_, sv)) in simCode.stringToSimVarHT
    local start = @match sv.attributes begin
      SOME(a) where hasproperty(a, :start) => a.start
      _ => nothing
    end
    start isa SOME && _expMentionsTime(start.data isa SimulationCode.Exp ? SimulationCode.toDAEExp(start.data) : start.data) &&
      push!(what, name * "'s start value")
  end
  local equationGuard = isempty(what) ? :() :
    :(tspan[1] == 0 || OMBackend.unsupported($(string("a start time other than 0 (", join(unique(what), ", "),
                                                      " evaluated at the build, for time 0)")), tspan[1]))
  #= The ticks of a sample() or a Pulse's period count from the start time
     (PeriodicCallback's phase): sample(0, 0.25) from 0.1 ticked at 0.1, 0.35. =#
  local periodicGuard = !_hasPeriodicWhen(simCode) ? :() :
    :(tspan[1] == 0 || OMBackend.unsupported("a start time other than 0 with sample() or a periodic source (ticks counted from the start time)", tspan[1]))
  local algorithmGuard = isempty(_earlyInitialAlgorithms(simCode)) ? :() :
    :(tspan[1] == 0 ||
      isequal(Base.invokelatest(__runInitialAlgorithmEarly!, 0.0), Base.invokelatest(__runInitialAlgorithmEarly!, Float64(tspan[1]))) ||
      OMBackend.unsupported("a start time other than 0 (initial algorithm values evaluated at the build, for time 0, that differ)", tspan[1]))
  return quote
    $(equationGuard)
    $(periodicGuard)
    $(algorithmGuard)
  end
end

#= Whether a when (an elsewhen arm too) is lowered to a PeriodicCallback. =#
function _hasPeriodicWhen(simCode)::Bool
  local periodic = function (arm)
    local cond = SimulationCode.toDAEExp(_elsewhenCondition(arm))
    local stmts = arm isa SimulationCode.WHEN_STMTS ? arm.whenStmtLst : arm.whenEquation.whenStmtLst
    _containsSampleCall(cond) || _pulsePeriodicSpec(cond, simCode, stmts) !== nothing
  end
  return any(simCode.whenEquations) do weq
    local arm = weq
    while arm !== nothing
      periodic(arm) && return true
      arm = _elsewhenInner(arm isa SimulationCode.WHEN_STMTS ? arm.elsewhenPart : arm.whenEquation.elsewhenPart)
    end
    false
  end
end

_expMentionsTime(@nospecialize(e)) = last(Util.traverseExpTopDown(e, (x, found) ->
  (x, !found, found || (x isa DAE.CREF && x.componentRef isa DAE.CREF_IDENT && x.componentRef.ident == "time")), false))

#= The initial algorithms the early pass runs: the sections with DAE
   statements (else the ops of `when initial()`), less those reading a
   derivative: der() reads the problem, which the early pass runs before. An
   `initial algorithm` section reading one is refused
   (generateInitialAlgorithmEarlyFunction); the ops of `when initial()` the
   runtime pass runs (_initialDerivativeReads). =#
function _earlyInitialAlgorithms(simCode)
  local sections = any(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms) ?
    filter(ia -> !isempty(ia.daeStatements), simCode.initialAlgorithms) : simCode.initialAlgorithms
  return any(_readsDerivative, sections) ? filter(!_readsDerivative, sections) : sections
end

function _readsDerivative(ia)::Bool
  local found = false
  local look = e -> begin
    local d = e isa SimulationCode.Exp ? SimulationCode.toDAEExp(e) : e
    d isa DAE.Exp && (found |= containsDerCall(d))
    e
  end
  foreach(s -> Util.mapDAEStatementExps(look, s), ia.daeStatements)
  for op in ia.statements, fld in (:right, :condition, :exp, :value, :message)
    hasproperty(op, fld) && look(getproperty(op, fld))
  end
  return found
end

function emitInitAlgU0Appends(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local appends::Vector{Expr} = Expr[]
  isempty(simCode.initialAlgorithms) && return appends
  local ht::Dict = simCode.stringToSimVarHT
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  for ia in _earlyInitialAlgorithms(simCode)
    _collectInitAlgNames!(lhsNames, rhsNames, ia)
  end
  for (_, el, _) in _initAlgTargets(lhsNames, ht, _initAlgIterators(simCode))
    local qn = QuoteNode(Symbol(el))
    #= Replace (not append) any existing start-attribute entry so a lifted
       discrete with both a start value and an init-algorithm value does not
       leave a duplicate key in the u0 pair list (which drops other entries
       during splitInitialValues). =#
    push!(appends, :(if haskey(_algResults, $(qn))
                       filter!(_p -> !isequal(_p.first, $(Symbol(el))), fiv)
                       push!(fiv, $(Symbol(el)) => _algResults[$(qn)])
                     end))
  end
  return appends
end

#= A variable's start value in the early pass: its start attribute, a literal
   or an expression that evaluates (`start = x0`, a parameter); else 0.0. =#
#= The value an initial algorithm section reads for the fixed variable `name`:
   its start (0 without one, MLS 4.9.1), evaluated at the build. A start that
   does not evaluate there (a Modelica function's call) is refused, and one
   that reads a tunable parameter is an error as a direct read is (the pass
   runs once, at the build): both were read as 0.0 (y0 := x + 1 gave 1 for
   `x(start = twice(p), fixed = true)`, OpenModelica 3). =#
function _fixedStartForInitialAlgorithm(name::String, sv, simCode)::Float64
  local attrs = sv.attributes
  (attrs isa SOME && hasproperty(attrs.data, :start) && attrs.data.start isa SOME) || return 0.0
  local start = attrs.data.start.data
  local startDAE = start isa SimulationCode.Exp ? SimulationCode.toDAEExp(start) : start
  _readsTunableParameter(startDAE, simCode) &&
    throw(ArgumentError("the start value of $(name), read by an initial algorithm, reads a tunable parameter; " *
                        "the initial algorithm is evaluated when the model is compiled, so it would keep its compiled value"))
  local v = OMBackend._tryOr(() -> SimulationCode.tryEvalNumeric(startDAE, simCode), nothing, :initAlgStart)
  v isa Real && return Float64(v)
  local folded = _foldParameterBindStatic(start, simCode)
  folded === nothing || return folded
  #= Calls and if-expressions (`sqrt(p) + exp(0 * p)`, `if p > 0 then 2 else 3`). =#
  local evaluated = OMBackend._tryOr(() -> evalDAE_Expression(startDAE, simCode), nothing, :initAlgStartEval)
  local parts = evaluated isa Expr && evaluated.head === :block ?
    filter(a -> !(a isa LineNumberNode), evaluated.args) : Any[evaluated]
  local value = isempty(parts) ? nothing : last(parts)
  value isa Real ||
    OMBackend.unsupported("an initial algorithm reading a fixed variable whose start does not evaluate at the build", name)
  return Float64(value)
end

function _initAlgStartValue(sv, simCode)::Float64
  local attrs = sv.attributes
  attrs isa SOME || return 0.0
  hasproperty(attrs.data, :start) && attrs.data.start isa SOME || return 0.0
  local v = OMBackend._tryOr(() -> SimulationCode.tryEvalNumeric(attrs.data.start.data, simCode), nothing, :initAlgStart)
  return v isa Real ? Float64(v) : _readStartAttributeAsLiteral(sv)
end

#= The non-parameter variables that the assignments to `lhsNames` set, as
   (assigned name, variable, index into the assigned value): the name itself
   (index nothing), or for an array its scalarized elements name[k] present
   in the table (`x := {1, 2, 3}`; 2-D names x[1][2] are not matched). A for
   loop's iterator (`iterators`, also among the assigned names) is no
   variable, even where an array of its name is. =#
function _initAlgTargets(lhsNames, ht, iterators = Set{String}(); parameters::Bool = false)::Vector{Tuple{String, String, Union{Nothing, Int}}}
  local out = Tuple{String, String, Union{Nothing, Int}}[]
  for name in lhsNames
    local elems = if haskey(ht, name)
      Tuple{String, Union{Nothing, Int}}[(name, nothing)]
    elseif name in iterators
      Tuple{String, Union{Nothing, Int}}[]
    else
      local prefix = string(name, "[")
      local idx = Int[]
      for key in keys(ht)
        (startswith(key, prefix) && endswith(key, "]")) || continue
        local k = tryparse(Int, SubString(key, ncodeunits(prefix) + 1, prevind(key, lastindex(key))))
        k === nothing || push!(idx, k)
      end
      Tuple{String, Union{Nothing, Int}}[(string(name, "[", k, "]"), k) for k in sort!(idx)]
    end
    for (el, k) in elems
      local sv = ht[el][2]
      parameters || !(sv.varKind isa SimulationCode.PARAMETER || sv.varKind isa SimulationCode.ARRAY_PARAMETER) || continue
      #= Captured as one number (none in MSL 3.2.3: 14 initial algorithms). =#
      sv.varKind isa SimulationCode.ARRAY_PARAMETER &&
        OMBackend.unsupported("an initial algorithm assigning an array parameter", name)
      push!(out, (name, el, k))
    end
  end
  return out
end

#= The iterators of the for loops of the early initial algorithms. =#
function _initAlgIterators(simCode)::Set{String}
  local out = Set{String}()
  local walk! = nothing
  walk! = stmts -> foreach(stmts) do s
    @match s begin
      DAE.STMT_FOR(_, _, iter, _, _, body, _) => (push!(out, iter); walk!(body))
      DAE.STMT_PARFOR(_, _, iter, _, _, body, _, _) => (push!(out, iter); walk!(body))
      DAE.STMT_IF(_, body, else_, _) => (walk!(body); _walkElse!(walk!, else_))
      DAE.STMT_WHILE(_, body, _) => walk!(body)
      _ => nothing
    end
  end
  for ia in _earlyInitialAlgorithms(simCode)
    walk!(ia.daeStatements)
  end
  return out
end

function _walkElse!(walk!, else_)
  @match else_ begin
    DAE.ELSEIF(_, stmts, rest) => (walk!(stmts); _walkElse!(walk!, rest))
    DAE.ELSE(stmts) => walk!(stmts)
    _ => nothing
  end
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
  for (_, el, _) in _initAlgTargets(lhsNames, ht, _initAlgIterators(simCode); parameters = true)
    local qn = QuoteNode(Symbol(el))
    push!(appends, :(haskey(_algResults, $(qn)) &&
                     push!(_eqs, $(Symbol(el)) ~ _algResults[$(qn)])))
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

#= pre(v) at the initialization: v's start value (MLS 3.7.3). The runtime pass
   runs a `when initial()` body after the early pass, which set v already:
   `n = pre(n) + 1` gave 2. =#
function _preAtInitialization(@nospecialize(e), simCode)
  local d = e isa SimulationCode.Exp ? SimulationCode.toDAEExp(e) : e
  d isa DAE.Exp || return e
  local (out, _) = Util.traverseExpBottomUp(d, (x, acc) -> begin
      if x isa DAE.CALL && x.path isa Absyn.IDENT && x.path.name == "pre" && listHead(x.expLst) isa DAE.CREF
        local entry = get(simCode.stringToSimVarHT, string(listHead(x.expLst).componentRef), nothing)
        entry === nothing || (x = DAE.RCONST(_readStartAttributeAsLiteral(last(entry))))
      end
      (x, acc)
    end, nothing)
  return out
end

function _initialWhenOpToJulia(wStmt, simCode::SimulationCode.SIM_CODE,
                               renamedNames::OrderedSet{String} = OrderedSet{String}())
  local sub = e -> _substituteBoundParameters(_preAtInitialization(e, simCode), simCode)
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
    _refuseInitialReinit(wStmt)
  elseif wStmt isa BDAE.ASSERT || wStmt isa SimulationCode.ASSERT
    #= On the initialization's values: AssertionLevel.error stops the
       simulation, as an equation's assert does (it only warned). =#
    local cond = lowerAlg(wStmt.condition)
    local msg = lowerAlg(wStmt.message)
    local level = wStmt.level isa SimulationCode.Exp ? SimulationCode.toDAEExp(wStmt.level) : wStmt.level
    local condText = string(wStmt.condition isa SimulationCode.Exp ? SimulationCode.toDAEExp(wStmt.condition) : wStmt.condition)
    local violated = AlgorithmicCodeGeneration.isWarningAssertionLevel(level) ?
      :(@warn string("Assertion violated at the initialization: ", $(msg))) :
      :(throw(OMBackend.CodeGeneration.ModelicaAssertionError(LATEST_PROBLEM.tspan[1], string($(msg)), $(condText))))
    return :(if !($cond); $(violated); end; nothing)
  elseif wStmt isa BDAE.TERMINATE || wStmt isa SimulationCode.TERMINATE
    local msg = lowerAlg(wStmt.message)
    return :(@info "Modelica terminate() during init" message=$(msg))
  end
  unsupported("initial when-statement variant", wStmt)
end

#= The early pass runs at initialization: initial() is true (a body that reads
   it, MSL Fluid PumpingSystem: UndefVarError `initial`). =#
_initialIsTrue(@nospecialize(e)) = first(Util.traverseExpBottomUp(e, (x, acc) -> begin
    (x isa DAE.CALL && x.path isa Absyn.IDENT && x.path.name == "initial") ? (DAE.BCONST(true), acc) : (x, acc)
  end, nothing))

#= Translate a single `BDAE.WhenOperator` from an init-algorithm body into a
   Julia statement suitable for the procedural body of
   `__runInitialAlgorithmEarly!`. ASSIGN emits `_alg_<lhs> = <rhs>` (with
   `local` on the first occurrence of that LHS); RHS identifiers are renamed
   via `_renameAlgIdentifiers` so they bind to the let-block locals rather
   than to module-level Symbolics bindings of the same name. =#
function _initialWhenOpToJuliaEarly(wStmt, simCode::SimulationCode.SIM_CODE,
                                    renamedNames::OrderedSet{String}, seenLHS::OrderedSet{String})
  local sub = e -> _substituteBoundParameters(_initialIsTrue(e), simCode)
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
  elseif wStmt isa BDAE.REINIT || wStmt isa SimulationCode.REINIT
    _refuseInitialReinit(wStmt)
  end
  #= A block, not the symbol `nothing` (the statements are an Expr vector:
     a reinit here was a MethodError). =#
  return Expr(:block, :nothing)
end

#= reinit in a `when initial()` body: OpenModelica ignores it (x keeps its
   start), the early pass failed, the runtime pass set it. None in MSL 3.2.3. =#
_refuseInitialReinit(wStmt) =
  OMBackend.unsupported("reinit in a when initial() body", SimulationCode.DAE_identifierToString(wStmt.stateVar))

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

Emit `function __runInitialAlgorithmEarly!(t0 = 0.0) -> Dict{Symbol, Float64}` that
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

The body is wrapped in `let time = t0 ... end` (the build runs it for time 0,
_startTimeGuard again for the start time). Non-LHS crefs read on the
RHS get a pre-seeded `_alg_<name>` from the SimVar's `start` attribute or
`0.0` (a parameter: its statically folded binding); every LHS starts at
its start value. After the body, each LHS final value is captured into the returned
`Dict{Symbol, Float64}` (an entry that is not a number is skipped).

A body that throws returns no results (a fallback); the runtime `remake`
path remains for state-cref-RHS reads whose post-init value differs from
the `start` attribute.
"""
function generateInitialAlgorithmEarlyFunction(simCode::SimulationCode.SIM_CODE)::Expr
  #= der() reads the problem, which the early pass runs before (none in MSL
     3.2.3; skipped, its targets kept their start values). =#
  local readsDer = count(ia -> !isempty(ia.daeStatements) && _readsDerivative(ia), simCode.initialAlgorithms)
  readsDer > 0 && OMBackend.unsupported("an initial algorithm reading der()", readsDer)
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  #= What the `initial algorithm` sections read (not the `when initial()`
     bodies, which the pass runs where a model has no such section). =#
  local sectionReads = OrderedSet{String}()
  for ia in _earlyInitialAlgorithms(simCode)
    _collectInitAlgNames!(lhsNames, rhsNames, ia)
    isempty(ia.daeStatements) || _collectInitAlgNames!(OrderedSet{String}(), sectionReads, ia)
  end
  local noEarlyPass = quote
    function __runInitialAlgorithmEarly!(t0::Float64 = 0.0)
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
    #= A name the backend eliminated (folded by constant propagation, an
       alias, an RHS-equivalent variable) was read as 0.0: `s := sum(v)` gave
       0 and `r := w2 + 1` 1 (OpenModelica 12 and 2). None in MSL 3.2.3, whose
       initial algorithms read parameters and time. =#
    local inSection = name in sectionReads
    haskey(ht, name) || begin
      inSection && OMBackend.unsupported("an initial algorithm reading a variable the backend eliminated", name)
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
        #= A parameter the initialization computes (fixed = false, no
           binding): the section's results were dropped with the expected
           UndefVarError below (`s := k + 1`, k = x: s = 0, OpenModelica 4). =#
        inSection && sv.varKind isa SimulationCode.PARAMETER && !SimulationCode.hasBindingExp(sv) &&
          OMBackend.unsupported("an initial algorithm reading a parameter the initialization computes", name)
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
    #= A String parameter (the MSL TraceSubstances sensors' `substanceName`;
       ReadRealMatrixFromFile's `file = loadResource(...)`): its literal, or
       the module-level value its binding gives (createStringParameterAssignments,
       as the runtime pass reads it). It was read as its start, 0.0. =#
    if sv.varKind isa SimulationCode.STRING
      local value = @match sv.varKind begin
        SimulationCode.STRING(bindExp = SOME(b)) =>
          (local d = SimulationCode.toDAEExp(b); d isa DAE.SCONST ? d.string : Symbol(sv.name))
        _ => nothing
      end
      value === nothing && OMBackend.unsupported("an initial algorithm reading a String without a value", name)
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(value)))
      continue
    end
    #= A variable the initialization determines: the pass runs before it, and
       its start value is no value (`q := z + 1` with `initial equation z = 5`
       gave 1, OpenModelica 6). A fixed one has its start value. =#
    inSection || begin
      #= Its start value, evaluated: a parameter (`x(start = x0)`) was read as 0.0. =#
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(_initAlgStartValue(sv, simCode))))
      continue
    end
    _isFixedStart(sv) ||
      OMBackend.unsupported("an initial algorithm reading a variable the initialization determines", name)
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(_fixedStartForInitialAlgorithm(name, sv, simCode))))
  end
  #= An assigned variable starts at its start value: read before it is
     assigned, and what an untaken branch leaves (an initial algorithm
     determines all its left-hand sides). An array, the vector of its
     elements' starts. =#
  local iterators = _initAlgIterators(simCode)
  for name in lhsNames
    local start = if haskey(ht, name)
      _initAlgStartValue(ht[name][2], simCode)
    else
      local elems = _initAlgTargets([name], ht, iterators)
      local starts = zeros(isempty(elems) ? 0 : maximum(last, elems))
      for (_, el, k) in elems
        starts[k] = _initAlgStartValue(ht[el][2], simCode)
      end
      isempty(elems) ? 0.0 : Expr(:vect, starts...)
    end
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(start)))
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
  #= A safety net: _earlyInitialAlgorithms leaves out the sections reading a
     derivative (der() reads the problem, which the early pass runs before;
     the other statements alone would make wrong initialization constraints). =#
  any(_callsDer, stmts) && return noEarlyPass
  local captures = Expr[]
  for (name, el, k) in _initAlgTargets(lhsNames, ht, iterators; parameters = true)
    local algSym = Symbol("_alg_" * name)
    local qn = QuoteNode(Symbol(el))
    push!(captures, :(try; _results[$(qn)] = Float64($(k === nothing ? algSym : :($(algSym)[$(k)]))); catch _e; OMBackend._fallback(_e, :initAlgEarlyCapture); nothing; end))
  end
  return quote
    function __runInitialAlgorithmEarly!(t0::Float64 = 0.0)
      local _results = Dict{Symbol, Float64}()
      try
        let time = t0
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

Emit a `function __runInitialAlgorithm!(t0 = 0.0) ... end` whose body executes once
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
      function __runInitialAlgorithm!(t0::Float64 = 0.0)
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
      function __runInitialAlgorithm!(t0::Float64 = 0.0)
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
      #= A discrete Real as it is, an Integer as an Int. An enumeration
         (Logic), or a discrete whose attributes do not say (the MSL Digital
         examples' Logic signals), is an array index: MTK's initialization may
         leave it at 0, which BoundsErrors on 1-based index vectors (INV3S's
         UX01Conv[iNV3S_enable]; NXFER), so it is clamped to 1. Every discrete
         was rounded and clamped (`n = m + 5`, m = -2: n = 6, OpenModelica 3;
         a discrete Real 0.3 read as 1). Not a Boolean: the clamp made false
         true (an ideal thyristor's `fire`). One the problem does not have
         stays unassigned, as below (it was 1). =#
      if sv.varKind isa SimulationCode.DISCRETE && !(name in boolNames) && !_isBoolDiscreteName(name, simCode)
        local sym = Symbol(name)
        local value = @match sv.attributes begin
          SOME(DAE.VAR_ATTR_REAL(__)) => :(Float64(_raw))
          SOME(DAE.VAR_ATTR_INT(__)) => :(Int(round(Float64(_raw))))
          _ => :(max(1, Int(round(Float64(_raw)))))
        end
        push!(fetches, quote
          local $(sym)
          try
            local _raw = ModelingToolkit.SciMLBase.getu(LATEST_PROBLEM, $(QuoteNode(sym)))(LATEST_PROBLEM)
            $(sym) = $(value)
          catch _e
            OMBackend._fallback(_e, :initAlgGetuFetch)
            push!(_unfetched, $(QuoteNode(sym)))
          end
        end)
        continue
      end
    end
    #= Non-discrete SimVars (Real states, alg vars) and local algorithm
       temporaries not in the HT: fetch as Float64, no index clamp. One the
       problem does not have stays unassigned: a temporary is assigned before
       it is read, and a read of a variable the backend eliminated is refused
       below (it was 0.0: `s = sum(v)` of a folded v gave 0, OpenModelica 12). =#
    local sym = Symbol(name)
    push!(fetches, quote
      local $(sym)
      try
        $(sym) = Float64(ModelingToolkit.SciMLBase.getu(LATEST_PROBLEM, $(QuoteNode(sym)))(LATEST_PROBLEM))
      catch _e
        OMBackend._fallback(_e, :initAlgGetuValue)
        push!(_unfetched, $(QuoteNode(sym)))
      end
    end)
  end
  #= Shadow `Base.time` (a UNIX-time function) with the local Modelica `time`
     value, the start time (t0). Without this, init-algorithm bodies
     that reference `time` (e.g. trapezoid sources' `count := integer((time -
     startTime) / period)`) generate `time - <Float64>` and hit MethodError
     because `Base.time` is a function, not a number. =#
  return quote
    function __runInitialAlgorithm!(t0::Float64 = 0.0)
      #= `_hard` collects (symbolic_var => value) pairs for each ASSIGN to a
         non-parameter variable. simulate() passes it to `remake(prob; u0=…,
         initializealg=NoInit())` so MTK treats the init-algorithm-computed
         values as hard initial conditions (Modelica §11.2), not guesses. =#
      local _hard = Dict{Any, Any}()
      local _unfetched = Symbol[]
      let time = t0
        $(fetches...)
        try
          $(stmts...)
        catch _e
          (_e isa UndefVarError && _e.var in _unfetched) &&
            OMBackend.unsupported("a when initial() body reading a variable that is not in the solved system", _e.var)
          rethrow()
        end
      end
      return _hard
    end
  end
end
