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
  The initial equations are placed last.
"""
function createStartConditionsEquationsMTK(states::Vector, algebraics::Vector, simCode::SC.SIM_CODE;
                                           skipDefaultStateStarts::Bool = false)::Vector{Expr}
  local algInit = getStartConditionsMTK(algebraics, simCode; skipDefaultStarts = true)
  local stateInit = getStartConditionsMTK(states, simCode; skipDefaultStarts = skipDefaultStateStarts)
  local ieqInit = generateInitialEquations(simCode.initialEquations, simCode)
  return vcat(algInit, stateInit, ieqInit)
end

"""
An initial equation `lhs = rhs`; another kind (an if-equation the lowering left: its
branches do not line up) is refused.
"""
function _initialEquation(ieq::SC.Equation)::SC.EQUATION
  ieq isa SC.EQUATION || unsupported("this initial equation (an if-equation whose branches differ)", ieq)
  return ieq
end

"""
  Generates initial equations as Symbolics `lhs ~ rhs` Equation forms suitable
  for passing to MTK's `initialization_eqs` kwarg of `System(...)`. Unlike the
  `=>` pair form (which acts as a guess only), `~` form is a real constraint
  that MTK's initialization solver must satisfy at t=0. Required for models
  with `InitialOutput` init mode (e.g. PID controllers' integrator state).
  The parametric ones the build solved are left out, unless a parameter of one is
  still unbound: then the initialization solves it.
"""
function generateInitialEquationsAsConstraints(initialEqs::Vector{SC.Equation}, simCode::SC.SIM_CODE)::Vector{Expr}
  local result = Expr[]
  for ieq in initialEqs
    if ieq isa SC.ARRAY_EQUATION
      @debug "[MTK GEN: initialConstraints] skipping an array equation (array constraints not yet lowered to scalar `~` form)"
      continue
    end
    (isParametricOnlyEquation(ieq, simCode) && !_readsUnboundParameter(ieq, simCode)) && continue
    local eq = _initialEquation(ieq)
    #= A tuple equation of a call the build did not solve: one constraint per variable of the
       left side, its output (MSL Filter's `({r[1], r[2]}, _, _, _) = roots.lowPass(cr, ...)`
       with cr computed by the initialization; dropped, r kept 0 and the filter's states did
       not move). =#
    local targets = MTK_CodeGenerationUtil._tupleEquationTargets(eq.lhs, eq.rhs)
    if targets !== nothing
      for (target, element, _) in targets
        try
          push!(result, :($(expToJuliaExpMTK(target, simCode)) ~ $(_initialEquationRhs(element, simCode))))
        catch err
          OMBackend._fallback(err, :initialConstraintTuple; only = UnsupportedLowering, impact = :result)
          @warn "[CODEGEN: initialConstraints] failed to lower a tuple element; constraint dropped" target err
        end
      end
      continue
    end
    local lhs = try
      expToJuliaExpMTK(eq.lhs, simCode)
    catch err
      OMBackend._fallback(err, :initialConstraintLhs; only = UnsupportedLowering, impact = :result)
      @warn "[CODEGEN: initialConstraints] failed to lower LHS; constraint dropped" lhs=eq.lhs err
      continue
    end
    local rhs = try
      _initialEquationRhs(eq.rhs, simCode)
    catch err
      OMBackend._fallback(err, :initialConstraintRhs; only = UnsupportedLowering, impact = :result)
      @warn "[CODEGEN: initialConstraints] failed to lower RHS; constraint dropped" rhs=eq.rhs err
      continue
    end
    push!(result, :($lhs ~ $rhs))
  end
  return result
end

"""
The right side of an initial equation, lowered (both of its forms:
generateInitialEquationsAsConstraints, generateInitialEquations). `time` (the independent
variable, in no table: MSL ControlledTanks' `v = time`), an unknown and a parameter without
a binding stay references; a bound parameter and an expression are evaluated at the build,
except where they read a tunable parameter, directly or through a binding: `x = p` and `x =
q` with q = 2 * p kept x(0) for every p.
"""
function _initialEquationRhs(rhs::SC.Exp, simCode::SC.SIM_CODE)
  rhs isa SC.EXP_CREF || return evalExpression(rhs, simCode; keepTunable = true)
  _isTimeCref(rhs) && return expToJuliaExpMTK(rhs, simCode)
  local entry = get(simCode.stringToSimVarHT, SC.crefName(rhs.cref), nothing)
  (entry === nothing || SC.isStateOrAlgebraic(last(entry)) || !SC.hasBindingExp(last(entry))) &&
    return expToJuliaExpMTK(rhs, simCode)
  _readsTunableParameter(rhs, simCode) && return evalExpression(rhs, simCode; keepTunable = true)
  return evalSimCodeParameter(last(entry), simCode)
end

"""
Whether an equation reads a parameter without a binding (a fixed=false one the build did
not solve).
"""
function _readsUnboundParameter(ieq::SC.Equation, simCode::SC.SIM_CODE)::Bool
  local ht = simCode.stringToSimVarHT
  return any(equationSides(ieq)) do side
    any(SC.allCrefNames(side)) do c
      local entry = get(ht, c, nothing)
      entry !== nothing && SC.isParameter(last(entry)) && !SC.hasBindingExp(last(entry))
    end
  end
end

"""
Whether `e` reads a state, an algebraic or a discrete variable, or one the passes eliminated
(in no table).
"""
function _readsUnknown(e::SC.Exp, simCode::SC.SIM_CODE)::Bool
  local ht = simCode.stringToSimVarHT
  return any(SC.allCrefNames(e)) do name
    local entry = get(ht, name, nothing)
    entry === nothing ? name != "time" : (SC.isStateOrAlgebraic(last(entry)) || SC.isDiscrete(last(entry)))
  end
end

"""
    generateInitialEquations(initialEqs, simCode; parameterAssignment = true)

The initial equations as guesses `lhs => rhs` (`lhs = rhs` without `parameterAssignment`), in
their order; the parametric ones the build solved left out. A guess only from what the build
knows: an expression reading an unknown is no value here (an eliminated one is not even
defined: the lowered initial if-equation of CDL SunRiseSet reads cosHou); the equation stays a
constraint. A plain unknown is its variable.
"""
function generateInitialEquations(initialEqs::Vector{SC.Equation}, simCode::SC.SIM_CODE; parameterAssignment::Bool = true)::Vector{Expr}
  local initialEqsExps = Expr[]
  for ieq in initialEqs
    ieq isa SC.ARRAY_EQUATION &&
      error("generateInitialEquations: unexpected unexpanded array equation in initial equations — this is a compiler bug upstream")
    isParametricOnlyEquation(ieq, simCode) && continue
    local eq = _initialEquation(ieq)
    !(eq.rhs isa SC.EXP_CREF) && _readsUnknown(eq.rhs, simCode) && continue
    local lhs = expToJuliaExpMTK(eq.lhs, simCode)
    local rhs = _initialEquationRhs(eq.rhs, simCode)
    push!(initialEqsExps, parameterAssignment ? quote $lhs => $rhs end : quote $lhs = $rhs end)
  end
  return initialEqsExps
end

"""
  Given a vector of variables and the simulation code
  extracts the start attributes to generate initial conditions.

If `skipDefaultStarts` is true, variables without explicit start values are skipped.
When false, variables without start values get default 0.0 initialization; with
`fixed = true` and no start, 0.0 in either case (it pins the variable there). A variable without
attributes of its own (attributesGiven: compiler-generated ones) defaults without a warning.
"""
function getStartConditionsMTK(vars::Vector, simCode::SC.SIM_CODE; skipDefaultStarts::Bool = false)::Vector{Expr}
  local startExprs = Expr[]
  isempty(vars) && return startExprs
  local ht = simCode.stringToSimVarHT
  local missingStartWarnings = OrderedSet{String}()
  for var in vars
    local simVar = last(ht[var])
    local varName = simVar.name
    local a = simVar.attributes
    if !SC.attributesGiven(simVar.elementType, a)
      skipDefaultStarts || push!(startExprs, :($(Symbol(varName)) => 0.0))
    elseif a.start isa SC.EXP_CREF
      #= Through expToJuliaExpMTK, so a data structure, a parameter or a subscripted reference
         is each read as such (`pars[name]` held only for a parameter). =#
      push!(startExprs, quote $(Symbol(varName)) => $(expToJuliaExpMTK(SC.EXP_CREF(a.start.cref, SC.TYPE_REAL()), simCode)) end)
    elseif a.start !== nothing
      push!(startExprs, quote $(Symbol(varName)) => $(expToJuliaExpMTK(a.start, simCode)) end)
    elseif a.fixed !== nothing
      (skipDefaultStarts && a.fixed != SC.BCONST(true)) && continue
      push!(startExprs, :($(Symbol(varName)) => 0.0))
    elseif !skipDefaultStarts
      push!(missingStartWarnings, varName)
      push!(startExprs, :($(Symbol(varName)) => 0.0))
    end
  end
  if OMBackend.WARN_MISSING_START_VALUES[] && !isempty(missingStartWarnings)
    local warningList = sort!(collect(missingStartWarnings))
    local maxShown = 20
    local shown = warningList[1:min(end, maxShown)]
    local omitted = length(warningList) - length(shown)
    local summary = "Assumed starting value of 0.0 for $(length(warningList)) variable(s): " * join(shown, ", ")
    omitted > 0 && (summary *= ", ... (+$(omitted) more)")
    @warn summary
  end
  return startExprs
end

"""
    equationDefinedDiscreteNames(simCode)

The discretes an equation defines, not a when-equation: a residual's top-level operand (`d -
rhs`, `rhs + d`, `-d`, `d`), or one lifted into a when-equation on the change of the relations
it reads (`liftedDiscreteNames`).
"""
function equationDefinedDiscreteNames(simCode::SC.SIM_CODE)::Set{String}
  local ht = simCode.stringToSimVarHT
  local discreteName = function (e::SC.Exp)
    e isa SC.EXP_CREF || return nothing
    local entry = get(ht, SC.crefName(e.cref), nothing)
    return entry !== nothing && last(entry).varKind isa SC.DISCRETE ? SC.crefName(e.cref) : nothing
  end
  local defined = Set{String}(liftedDiscreteNames(simCode))
  for eq in simCode.residualEquations
    local e = eq.exp
    local operands = e isa SC.BINARY && e.op in (SC.OP_SUB, SC.OP_ADD) ? (e.exp1, e.exp2) :
                     e isa SC.UNARY && e.op === SC.OP_UMINUS ? (e.exp,) : (e,)
    for o in operands
      local nm = discreteName(o)
      nm === nothing || push!(defined, nm)
    end
  end
  return defined
end

"""
  Emit `lhs ~ rhs` constraint Equations for state vars with `fixed=true` and a
  `start` (0 where it has none: the default start, MLS 4.9.1). Goes into
  `initialization_eqs` so MTK pins them at t=0 rather than treating them as soft
  `guesses` the iteration may override. A `v(fixed = true)` without a start was
  left out: v = xa + 1 started at 1, not 0 (OpenModelica: xa(0) = -1).
"""
function getFixedStartConstraintsMTK(vars::Vector, simCode::SC.SIM_CODE)::Vector{Expr}
  local result = Expr[]
  local ht = simCode.stringToSimVarHT
  #= a discrete an equation defines: its fixed start is its pre() at the initialization (MLS
     8.6), its value the equation's (MSL OpenTank's ports_m_flow_out(each start = true, each
     fixed = true) = false: v = true overdetermined the tank) =#
  local defined = equationDefinedDiscreteNames(simCode)
  for var in vars
    local simVar = last(ht[var])
    simVar.attributes.fixed == SC.BCONST(true) || continue
    simVar.name in defined && continue
    local startExp = something(simVar.attributes.start, SC.RCONST(0.0))
    push!(result, :($(Symbol(simVar.name)) ~ $(expToJuliaExpMTK(startExp, simCode))))
  end
  return result
end

#= ==================================================================================== =#
#= Initial algorithms                                                                   =#
#= ==================================================================================== =#

"""
The names an initial algorithm's statement assigns (`lhs`; a for loop's iterator too) and
reads (`rhs`).
"""
function _collectInitAlgNames!(lhs::OrderedSet{String}, rhs::OrderedSet{String}, s::SC.Statement)
  local reads = e::SC.Exp -> union!(rhs, SC.allCrefNames(e))
  local assigns = e::SC.Exp -> e isa SC.EXP_CREF && push!(lhs, SC.crefName(e.cref))
  if s isa SC.ASSIGN
    s.left isa SC.TUPLE ? foreach(assigns, s.left.elements) : assigns(s.left)
    reads(s.right)
  elseif s isa SC.REINIT
    push!(lhs, SC.crefName(s.stateVar))
    reads(s.value)
  elseif s isa SC.NORETCALL
    reads(s.exp)
  elseif s isa SC.ASSERT
    reads(s.condition); reads(s.message); reads(s.level)
  elseif s isa SC.TERMINATE
    reads(s.message)
  elseif s isa SC.IF
    for b in s.branches
      reads(b.condition)
      foreach(x -> _collectInitAlgNames!(lhs, rhs, x), b.body)
    end
    foreach(x -> _collectInitAlgNames!(lhs, rhs, x), s.elseBody)
  elseif s isa SC.FOR
    reads(s.iterator.range)
    push!(lhs, String(s.iterator.name))
    foreach(x -> _collectInitAlgNames!(lhs, rhs, x), s.body)
  elseif s isa SC.WHILE
    reads(s.condition)
    foreach(x -> _collectInitAlgNames!(lhs, rhs, x), s.body)
  end
  return lhs
end

function _collectInitAlgNames!(lhs::OrderedSet{String}, rhs::OrderedSet{String}, ia::SC.INITIAL_ALGORITHM)
  foreach(s -> _collectInitAlgNames!(lhs, rhs, s), ia.statements)
  return lhs
end

"""
What the build evaluated at time 0, for a simulation from another start time. The
parametric initial equations it solved that read `time` are refused (MSL Blocks.Math.Mean's
t0 = time gave 0 with any start time); one reading an unbound parameter is solved by the
initialization, at the start time. Start values reading time: the build's u0, for time 0.
The ticks of a sample() or a Pulse's period count from the start time (PeriodicCallback's
phase: sample(0, 0.25) from 0.1 ticked at 0.1, 0.35). The early initial algorithms run
again at the start time, refused only where a value differs (a Pulse's count is the same
over its first period). Nothing when there is neither.
"""
function _startTimeGuard(simCode::SC.SIM_CODE)::Expr
  local what = String[]
  for ieq in simCode.initialEquations
    (isParametricOnlyEquation(ieq, simCode) && !_readsUnboundParameter(ieq, simCode)) || continue
    local sides = equationSides(ieq)
    any(_expMentionsTime, sides) && append!(what, (c for side in sides for c in SC.allCrefNames(side) if c != "time"))
  end
  for (name, (_, sv)) in simCode.stringToSimVarHT
    local start = sv.attributes.start
    (start !== nothing && _expMentionsTime(start)) && push!(what, name * "'s start value")
  end
  local equationGuard = isempty(what) ? :() :
    :(tspan[1] == 0 || OMBackend.unsupported($(string("a start time other than 0 (", join(unique(what), ", "),
                                                      " evaluated at the build, for time 0)")), tspan[1]))
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

"""
Whether a when (an elsewhen arm too) is lowered to a PeriodicCallback.
"""
function _hasPeriodicWhen(simCode::SC.SIM_CODE)::Bool
  return any(simCode.whenEquations) do weq
    local arm = weq.whenEquation
    while arm !== nothing
      (_containsSampleCall(arm.condition) || _pulsePeriodicSpec(arm.condition, simCode, arm.body) !== nothing) && return true
      arm = arm.elsewhenPart
    end
    false
  end
end

_expMentionsTime(e::SC.Exp)::Bool = SC.anyExp(_isTimeCref, e)

"""
The initial algorithms the early pass runs: the sections (`initial algorithm`, an
algorithm's `when initial()`), else the other initial arms; less those reading a derivative:
der() reads the problem, which the early pass runs before. A section reading one is refused
(generateInitialAlgorithmEarlyFunction); the other initial arms the runtime pass runs
(_initialDerivativeReads). Not the sections the runtime pass runs (_lateSections).
"""
function _earlyInitialAlgorithms(simCode::SC.SIM_CODE)::Vector{SC.INITIAL_ALGORITHM}
  local late = _lateSections(simCode)
  local sections = any(ia -> ia.section && !(ia in late), simCode.initialAlgorithms) ?
    filter(ia -> ia.section && !(ia in late), simCode.initialAlgorithms) : filter(ia -> !(ia in late), simCode.initialAlgorithms)
  return any(_readsDerivative, sections) ? filter(!_readsDerivative, sections) : sections
end

"""
    _lateSections(simCode) -> Vector{SC.INITIAL_ALGORITHM}

The sections of assignments only that read a variable the initialization determines (not a parameter, a String,
a fixed start, time, or one the section assigns): the runtime pass runs them, after the initialization, with its
values (generateInitialAlgorithmFunction). The early pass runs before it and refused them: MSL MathBoolean's
OnDelay, `when initial() then delaySignal := u; t_next := time - 1`, its input u connected (BooleanNetwork1).
"""
function _lateSections(simCode::SC.SIM_CODE)::Vector{SC.INITIAL_ALGORITHM}
  local ht = simCode.stringToSimVarHT
  local late = SC.INITIAL_ALGORITHM[]
  for ia in simCode.initialAlgorithms
    (ia.section && all(st -> st isa SC.ASSIGN, ia.statements) && !_readsDerivative(ia)) || continue
    local (lhs, rhs) = (OrderedSet{String}(), OrderedSet{String}())
    _collectInitAlgNames!(lhs, rhs, ia)
    local determined = name -> begin
      (name == "time" || name in lhs || !haskey(ht, name)) && return false
      local sv = last(ht[name])
      return !(sv.varKind isa Union{SC.PARAMETER, SC.ARRAY_PARAMETER, SC.STRING, SC.DATA_STRUCTURE}) && !_isFixedStart(sv)
    end
    any(determined, rhs) && push!(late, ia)
  end
  return late
end

"""
Whether an initial algorithm reads der().
"""
function _readsDerivative(ia::SC.INITIAL_ALGORITHM)::Bool
  local found = false
  foreach(s -> SC.mapStatementExps(e -> (found |= containsDerCall(e); e), s), ia.statements)
  return found
end

"""
  Emit `Expr`s that push init-algorithm-derived (state => value) pairs into the
  `finalInitialValues` vector inside Model(). Mirrors `emitInitAlgConstraintAppends`
  but the push target is the u0-pair list (consumed by `ODEProblem(...; u0 = ...)`)
  rather than `initialization_eqs`. The merge is required because the pure-ODE
  branch of the ODEProblem build skips MTK's init solver, so the init-eq alone
  would not propagate the init-algorithm value into u0. An existing start entry is
  replaced (not appended): a lifted discrete with both a start value and an
  init-algorithm value left a duplicate key in the u0 pair list (which dropped other
  entries during splitInitialValues).
"""
function emitInitAlgU0Appends(simCode::SC.SIM_CODE)::Vector{Expr}
  local appends = Expr[]
  isempty(simCode.initialAlgorithms) && return appends
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  foreach(ia -> _collectInitAlgNames!(lhsNames, rhsNames, ia), _earlyInitialAlgorithms(simCode))
  for (_, el, _) in _initAlgTargets(lhsNames, simCode, _initAlgIterators(simCode))
    local qn = QuoteNode(Symbol(el))
    push!(appends, :(if haskey(_algResults, $(qn))
                       filter!(_p -> !isequal(_p.first, $(Symbol(el))), fiv)
                       push!(fiv, $(Symbol(el)) => _algResults[$(qn)])
                     end))
  end
  return appends
end

"""
The value an initial algorithm section reads for the fixed variable `name`: its start (0
without one, MLS 4.9.1), evaluated at the build. A start that does not evaluate there (a
Modelica function's call) is refused, and one that reads a tunable parameter is an error as
a direct read is (the pass runs once, at the build): both were read as 0.0 (y0 := x + 1
gave 1 for `x(start = twice(p), fixed = true)`, OpenModelica 3).
"""
function _fixedStartForInitialAlgorithm(name::String, sv::SC.SIMVAR, simCode::SC.SIM_CODE)::Float64
  local start = sv.attributes.start
  start === nothing && return 0.0
  _readsTunableParameter(start, simCode) &&
    throw(ArgumentError("the start value of $(name), read by an initial algorithm, reads a tunable parameter; " *
                        "the initial algorithm is evaluated when the model is compiled, so it would keep its compiled value"))
  local v = OMBackend._tryOr(() -> SC.tryEvalNumeric(start, simCode), nothing, :initAlgStart)
  v === nothing || return v
  local folded = _foldParameterBindStatic(start, simCode)
  folded === nothing || return folded
  #= Calls and if-expressions (`sqrt(p) + exp(0 * p)`, `if p > 0 then 2 else 3`). =#
  local evaluated = OMBackend._tryOr(() -> evalExpression(start, simCode), nothing, :initAlgStartEval)
  local parts = evaluated isa Expr && evaluated.head === :block ?
    filter(a -> !(a isa LineNumberNode), evaluated.args) : Any[evaluated]
  local value = isempty(parts) ? nothing : last(parts)
  value isa Real ||
    OMBackend.unsupported("an initial algorithm reading a fixed variable whose start does not evaluate at the build", name)
  return Float64(value)
end

"""
A variable's start value in the early pass: its start attribute, a literal or an expression
that evaluates (`start = x0`, a parameter); else its literal start (0.0 without one).
"""
function _initAlgStartValue(sv::SC.SIMVAR, simCode::SC.SIM_CODE)::Float64
  local start = sv.attributes.start
  start === nothing && return 0.0
  local v = OMBackend._tryOr(() -> SC.tryEvalNumeric(start, simCode), nothing, :initAlgStart)
  return v === nothing ? _readStartAttributeAsLiteral(sv) : v
end

"""
The non-parameter variables that the assignments to `lhsNames` set, as (assigned name,
variable, index into the assigned value): the name itself (index nothing), or for an array
its scalarized elements name[k] present in the table (`x := {1, 2, 3}`; 2-D names x[1][2]
are not matched). A for loop's iterator (`iterators`, also among the assigned names) is no
variable, even where an array of its name is. An array parameter is refused (captured as one
number; none in MSL 3.2.3: 14 initial algorithms).
"""
function _initAlgTargets(lhsNames, simCode::SC.SIM_CODE, iterators::Set{String} = Set{String}();
                         parameters::Bool = false)::Vector{Tuple{String, String, Union{Nothing, Int}}}
  local ht = simCode.stringToSimVarHT
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
      local sv = last(ht[el])
      parameters || !(sv.varKind isa Union{SC.PARAMETER, SC.ARRAY_PARAMETER}) || continue
      sv.varKind isa SC.ARRAY_PARAMETER && OMBackend.unsupported("an initial algorithm assigning an array parameter", name)
      push!(out, (name, el, k))
    end
  end
  return out
end

"""
The iterators of the for loops of the early initial algorithms.
"""
function _initAlgIterators(simCode::SC.SIM_CODE)::Set{String}
  local out = Set{String}()
  local walk! = nothing
  walk! = (body::Vector{SC.Statement}) -> foreach(body) do s
    if s isa SC.FOR
      push!(out, String(s.iterator.name))
      walk!(s.body)
    elseif s isa SC.IF
      foreach(b -> walk!(b.body), s.branches)
      walk!(s.elseBody)
    elseif s isa SC.WHILE
      walk!(s.body)
    end
  end
  foreach(ia -> walk!(ia.statements), _earlyInitialAlgorithms(simCode))
  return out
end

"""
  Emit `Expr`s that conditionally push init-algorithm-derived constraints into
  the local `_eqs` vector inside `_buildInitialConstraintEqs`. Each emitted line
  looks like `haskey(_algResults, :T_start) && push!(_eqs, T_start ~ _algResults[:T_start])`.
"""
function emitInitAlgConstraintAppends(simCode::SC.SIM_CODE)::Vector{Expr}
  local appends = Expr[]
  isempty(simCode.initialAlgorithms) && return appends
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  foreach(ia -> _collectInitAlgNames!(lhsNames, rhsNames, ia), _earlyInitialAlgorithms(simCode))
  for (_, el, _) in _initAlgTargets(lhsNames, simCode, _initAlgIterators(simCode); parameters = true)
    local qn = QuoteNode(Symbol(el))
    push!(appends, :(haskey(_algResults, $(qn)) &&
                     push!(_eqs, $(Symbol(el)) ~ _algResults[$(qn)])))
  end
  return appends
end

"""
der(x) as the builtin call der(x), which the algorithmic translator emits as a call (a
function has no derivatives): _initialDerivativeReads reads its value from the problem.
"""
_derAsCall(e::SC.Exp)::SC.Exp =
  SC.mapExp(x -> x isa SC.DER ? SC.CALL(:der, SC.Exp[SC.EXP_CREF(x.cref, SC.TYPE_REAL())], SC.TYPE_REAL(), SC.CALL_BUILTIN) : x, e)

"""
der(x) in a runtime initial algorithm (the initial() arm of a relation on a derivative,
MSL FluxTubes' `asc = der(Hstat) > 0`) reads the derivative from the problem
(_initialDerivative); there is no `der` function.
"""
function _initialDerivativeReads(@nospecialize(ex))
  ex isa Expr || return ex
  if ex.head === :call && length(ex.args) == 2 && ex.args[1] === :der
    local x = _blockValue(ex.args[2])
    x isa Symbol && return :(OMBackend.CodeGeneration._initialDerivative(LATEST_PROBLEM, $(QuoteNode(x))))
  end
  return Expr(ex.head, map(_initialDerivativeReads, ex.args)...)
end

_callsDer(@nospecialize(ex)) = ex isa Expr && ((ex.head === :call && ex.args[1] === :der) || any(_callsDer, ex.args))

#= The value of a `begin #= line =# x end` the algorithm lowering wraps expressions in. =#
function _blockValue(@nospecialize(ex))
  (ex isa Expr && ex.head === :block) || return ex
  local body = filter(a -> !(a isa LineNumberNode), ex.args)
  return length(body) == 1 ? _blockValue(body[1]) : ex
end

"""
The derivative of the variable `name` at `problem`'s start values: the runtime initial
algorithm runs before the solve. A discrete cluster's start pass evaluates such a relation
again on the solved state.
"""
function _initialDerivative(problem, name::Symbol)
  local sys = problem.f.sys
  local d = ModelingToolkit.Differential(ModelingToolkit.get_iv(sys))(getproperty(sys, name))
  local f = _buildObservedFunction(sys, d)
  return Float64(f(problem.u0, problem.p, first(problem.tspan)))
end

"""
pre(v) at the initialization: v's start value (MLS 3.7.3). The runtime pass runs a `when
initial()` body after the early pass, which set v already: `n = pre(n) + 1` gave 2.
"""
function _preAtInitialization(e::SC.Exp, simCode::SC.SIM_CODE)::SC.Exp
  return SC.mapExp(e) do x
    local name = _preReadName(x)
    local entry = name === nothing ? nothing : get(simCode.stringToSimVarHT, name, nothing)
    entry === nothing ? x : SC.RCONST(_readStartAttributeAsLiteral(last(entry)))
  end
end

"""
Both initialization passes run at the initialization: initial() is true (a body that reads
it, MSL Fluid PumpingSystem's TriggeredTrapezoid: UndefVarError `initial`).
"""
_initialIsTrue(e::SC.Exp)::SC.Exp = SC.mapExp(x -> SC.isBuiltin(x, :initial) ? SC.BCONST(true) : x, e)

"""
A statement of a runtime initial algorithm (the initial arms of when-equations) as Julia
for module-top eval inside `__runInitialAlgorithm!`: the parameters folded to their
bindings (_substituteBoundParameters) and lowered with the algorithmic translator, so no
Symbolics bindings are needed at the initialization; initial() true and pre(v) v's start.

An assignment to a parameter and a reinit write `LATEST_PROBLEM`; an assignment to another
variable is collected in `_hard`, which simulate turns into hard start values of the
unknowns (`remake(prob; u0 = …)`). The function runs after the ODEProblem is constructed
(`simulate(...)` in the generated module), so LATEST_PROBLEM is in scope. Without this, the
left side stayed at its default (0): `T_start := startTime + count*period` in the trapezoid
signal source was dropped, breaking every model that relies on `initial algorithm` to seed
states. An assert of level error stops the simulation, as an equation's does (on the
initialization's values).
"""
function _initialWhenOpToJulia(wStmt::SC.Statement, simCode::SC.SIM_CODE, renamedNames::OrderedSet{String} = OrderedSet{String}())
  local sub = e::SC.Exp -> _substituteBoundParameters(_preAtInitialization(_initialIsTrue(_derAsCall(e)), simCode), simCode)
  local lowerAlg = e::SC.Exp -> _initialDerivativeReads(_renameAlgIdentifiers(
    _resolveModelicaCallTargets(AlgorithmicCodeGeneration.expToJuliaExpAlg(sub(e))), renamedNames, ""))
  if wStmt isa SC.NORETCALL
    return :($(lowerAlg(wStmt.exp)); nothing)
  elseif wStmt isa SC.ASSIGN
    local name = SC.crefNameOf(wStmt.left)
    name === nothing && return :($(lowerAlg(wStmt.right)); nothing)
    local sym = Symbol(name)
    local entry = get(simCode.stringToSimVarHT, name, nothing)
    if entry !== nothing && last(entry).varKind isa Union{SC.PARAMETER, SC.ARRAY_PARAMETER}
      return :($(sym) = $(lowerAlg(wStmt.right)); LATEST_PROBLEM.ps[$(QuoteNode(sym))] = $(sym); nothing)
    end
    return :($(sym) = $(lowerAlg(wStmt.right));
             try
               _hard[getproperty(LATEST_REDUCED_SYSTEM, $(QuoteNode(sym)))] = $(sym)
             catch _e
               OMBackend._fallback(_e, :initialWhenHardStart)
               nothing
             end;
             nothing)
  elseif wStmt isa SC.REINIT
    _refuseInitialReinit(wStmt)
  elseif wStmt isa SC.ASSERT
    local cond = lowerAlg(wStmt.condition)
    local msg = lowerAlg(wStmt.message)
    local violated = AlgorithmicCodeGeneration.isWarningAssertionLevel(wStmt.level) ?
      :(@warn string("Assertion violated at the initialization: ", $(msg))) :
      :(throw(OMBackend.CodeGeneration.ModelicaAssertionError(LATEST_PROBLEM.tspan[1], string($(msg)), $(string(wStmt.condition)))))
    return :(if !($cond); $(violated); end; nothing)
  elseif wStmt isa SC.TERMINATE
    return :(@info "Modelica terminate() during init" message=$(lowerAlg(wStmt.message)))
  end
  unsupported("initial when-statement variant", wStmt)
end

"""
A statement of an initial arm in the procedural body of `__runInitialAlgorithmEarly!`
(where the model has no initial algorithm section): an assignment `_alg_<lhs> = <rhs>`
(`local` at the first one of its left side), the right side's names renamed
(`_renameAlgIdentifiers`) so they bind to the let-block's locals, not to the module-level
Symbolics bindings of the same name.
"""
function _initialWhenOpToJuliaEarly(wStmt::SC.Statement, simCode::SC.SIM_CODE,
                                    renamedNames::OrderedSet{String}, seenLHS::OrderedSet{String})::Expr
  local sub = e::SC.Exp -> _substituteBoundParameters(_initialIsTrue(e), simCode)
  local lowerAlg = e::SC.Exp -> _renameAlgIdentifiers(
    _resolveModelicaCallTargets(AlgorithmicCodeGeneration.expToJuliaExpAlg(sub(e))), renamedNames)
  if wStmt isa SC.NORETCALL
    return :($(lowerAlg(wStmt.exp)); nothing)
  elseif wStmt isa SC.ASSIGN
    local name = SC.crefNameOf(wStmt.left)
    name === nothing && return :($(lowerAlg(wStmt.right)); nothing)
    local algSym = Symbol("_alg_" * name)
    name in seenLHS && return :($(algSym) = $(lowerAlg(wStmt.right)); nothing)
    push!(seenLHS, name)
    return :(local $(algSym) = $(lowerAlg(wStmt.right)); nothing)
  elseif wStmt isa SC.ASSERT
    return :(if !($(lowerAlg(wStmt.condition))); @warn "Modelica assert() during init (early)" message=$(lowerAlg(wStmt.message)); end)
  elseif wStmt isa SC.TERMINATE
    return :(@info "Modelica terminate() during init (early)" message=$(lowerAlg(wStmt.message)))
  elseif wStmt isa SC.REINIT
    _refuseInitialReinit(wStmt)
  end
  #= A block, not the symbol `nothing` (the statements are an Expr vector). =#
  return Expr(:block, :nothing)
end

"""
reinit in a `when initial()` body: OpenModelica ignores it (x keeps its start), the early
pass failed, the runtime pass set it. None in MSL 3.2.3.
"""
_refuseInitialReinit(wStmt::SC.REINIT) = OMBackend.unsupported("reinit in a when initial() body", SC.crefName(wStmt.stateVar))

"""
Modelica function calls in algorithm code are `Base.invokelatest(Name, ...)` with Name bound
in OMBackend.CodeGeneration (createModelicaFunctionWrapper), which a function body runs in.
The early initial algorithm runs in the model's module, where Name is undefined: qualify it.
(MSL WriteRealMatrixToFile: `when initial() then success1 := writeRealMatrix(...)` raised
UndefVarError, swallowed, and success1..4 stayed false.)
"""
function _qualifyInvokedFunctions(@nospecialize(ex))
  ex isa Expr || return ex
  local args = map(_qualifyInvokedFunctions, ex.args)
  if ex.head === :call && length(args) >= 2 && args[1] == :(Base.invokelatest) && args[2] isa Symbol
    args[2] = Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(args[2]))
  end
  return Expr(ex.head, args...)
end

"""
    generateInitialAlgorithmEarlyFunction(simCode) -> Expr

Emit `function __runInitialAlgorithmEarly!(t0 = 0.0) -> Dict{Symbol, Float64}` that executes
the initial algorithms of `_earlyInitialAlgorithms` procedurally at module-load time
(Modelica §11.4: statements run sequentially, the left side's final value becomes the
variable's initial value).

A section is lowered with `AlgorithmicCodeGeneration.generateStatements`, the path of the
algorithm sections and function bodies (if, for, while, assert, reinit), another initial arm
statement by statement (`_initialWhenOpToJuliaEarly`); either renamed (`_renameAlgIdentifiers`)
to prefix every name with `_alg_`, so the locals do not collide with the Symbolics `Num`
bindings of the same name in the surrounding model scope.

The body is wrapped in `let time = t0 ... end` (the build runs it for time 0, _startTimeGuard
again for the start time). A name the right sides read gets a pre-seeded `_alg_<name>`: a
parameter its statically folded binding, a String its value, a fixed variable its start,
another its start or 0.0; every left side starts at its start value. After the body, each left
side's final value is captured into the returned `Dict{Symbol, Float64}` (an entry that is not
a number is skipped). A body that throws returns no results (a fallback); the runtime `remake`
path remains for reads whose post-init value differs from the start attribute.
"""
function generateInitialAlgorithmEarlyFunction(simCode::SC.SIM_CODE)::Expr
  #= der() reads the problem, which the early pass runs before (none in MSL 3.2.3; skipped, its
     targets kept their start values). =#
  local readsDer = count(ia -> ia.section && _readsDerivative(ia), simCode.initialAlgorithms)
  readsDer > 0 && OMBackend.unsupported("an initial algorithm reading der()", readsDer)
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  #= What the sections read (not the other initial arms, which the pass runs where a model has
     no section). =#
  local sectionReads = OrderedSet{String}()
  for ia in _earlyInitialAlgorithms(simCode)
    _collectInitAlgNames!(lhsNames, rhsNames, ia)
    ia.section && _collectInitAlgNames!(OrderedSet{String}(), sectionReads, ia)
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
    #= A name the backend eliminated (folded by constant propagation, an alias, an
       RHS-equivalent variable) was read as 0.0: `s := sum(v)` gave 0 and `r := w2 + 1` 1
       (OpenModelica 12 and 2). None in MSL 3.2.3, whose initial algorithms read parameters and
       time. =#
    local inSection = name in sectionReads
    if !haskey(ht, name)
      inSection && OMBackend.unsupported("an initial algorithm reading a variable the backend eliminated", name)
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = 0.0))
      continue
    end
    local sv = last(ht[name])
    if sv.varKind isa Union{SC.PARAMETER, SC.ARRAY_PARAMETER}
      local paramLit = sv.varKind isa SC.PARAMETER && sv.varKind.bindExp !== nothing ?
        _foldParameterBindStatic(sv.varKind.bindExp, simCode) : nothing
      if paramLit === nothing
        #= A parameter the initialization computes (fixed = false, no binding): the section's
           results were dropped with the expected UndefVarError below (`s := k + 1`, k = x: s =
           0, OpenModelica 4). =#
        inSection && sv.varKind isa SC.PARAMETER && !SC.hasBindingExp(sv) &&
          OMBackend.unsupported("an initial algorithm reading a parameter the initialization computes", name)
        #= No static value (an array parameter, a binding that does not fold): no local, so a
           statement that reads it throws an UndefVarError, which the body's catch expects
           (below). The statements that do not read it still give their values. =#
        unfoldedParameter = true
        continue
      end
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(paramLit)))
      continue
    end
    #= A String parameter (the MSL TraceSubstances sensors' `substanceName`;
       ReadRealMatrixFromFile's `file = loadResource(...)`): its literal, or the module-level
       value its binding gives (createStringParameterAssignments, as the runtime pass reads
       it). It was read as its start, 0.0. =#
    if sv.varKind isa SC.STRING
      local b = sv.varKind.bindExp
      b === nothing && OMBackend.unsupported("an initial algorithm reading a String without a value", name)
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(b isa SC.SCONST ? b.value : Symbol(sv.name))))
      continue
    end
    #= A variable the initialization determines: the pass runs before it, and its start value
       is no value (`q := z + 1` with `initial equation z = 5` gave 1, OpenModelica 6). A fixed
       one has its start value. Outside a section its start value, evaluated: a parameter
       (`x(start = x0)`) was read as 0.0. =#
    if !inSection
      push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(_initAlgStartValue(sv, simCode))))
      continue
    end
    _isFixedStart(sv) || OMBackend.unsupported("an initial algorithm reading a variable the initialization determines", name)
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(_fixedStartForInitialAlgorithm(name, sv, simCode))))
  end
  #= An assigned variable starts at its start value: read before it is assigned, and what an
     untaken branch leaves (an initial algorithm determines all its left-hand sides). An array,
     the vector of its elements' starts. =#
  local iterators = _initAlgIterators(simCode)
  for name in lhsNames
    local start = if haskey(ht, name)
      _initAlgStartValue(last(ht[name]), simCode)
    else
      local elems = _initAlgTargets([name], simCode, iterators)
      local starts = zeros(isempty(elems) ? 0 : maximum(last, elems))
      for (_, el, k) in elems
        starts[k] = _initAlgStartValue(last(ht[el]), simCode)
      end
      isempty(elems) ? 0.0 : Expr(:vect, starts...)
    end
    push!(prefetches, :(local $(Symbol("_alg_" * name)) = $(start)))
  end
  local stmts = Expr[]
  local seenLHS = copy(lhsNames)
  for ia in _earlyInitialAlgorithms(simCode)
    if ia.section
      for s in AlgorithmicCodeGeneration.generateStatements(ia.statements)
        push!(stmts, _qualifyInvokedFunctions(_renameAlgIdentifiers(s, renamedNames)))
      end
    else
      foreach(op -> push!(stmts, _qualifyInvokedFunctions(_initialWhenOpToJuliaEarly(op, simCode, renamedNames, seenLHS))),
              ia.statements)
    end
  end
  #= A safety net: _earlyInitialAlgorithms leaves out the sections reading a derivative (der()
     reads the problem, which the early pass runs before; the other statements alone would
     make wrong initialization constraints). =#
  any(_callsDer, stmts) && return noEarlyPass
  local captures = Expr[]
  for (name, el, k) in _initAlgTargets(lhsNames, simCode, iterators; parameters = true)
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

"""
The Boolean references an assignment's right side reads.
"""
function _collectBoolRhsCrefs!(names::OrderedSet{String}, op::SC.Statement)::OrderedSet{String}
  op isa SC.ASSIGN || return names
  SC.foreachExp(e -> e isa SC.EXP_CREF && e.ty isa SC.TYPE_BOOL && push!(names, SC.crefName(e.cref)), op.right)
  return names
end

"""
    generateInitialAlgorithmFunction(simCode) -> Expr

Emit a `function __runInitialAlgorithm!(t0 = 0.0) ... end` whose body executes once during
the initialization: the initial arms of the when-equations (not the sections: the early pass
emits control-flow-correct `initialization_eqs` for their left sides; but those of `_lateSections`). Parameter literals are
in the bodies already (the SimCode lowering inlines them), so no module-scope parameter
bindings are needed here. A no-op stub when the model has none.
"""
function generateInitialAlgorithmFunction(simCode::SC.SIM_CODE)::Expr
  local noRuntimePass = quote
    function __runInitialAlgorithm!(t0::Float64 = 0.0)
      return Dict{Any, Any}()
    end
  end
  local late = _lateSections(simCode)
  local whenAlgorithms = filter(ia -> !ia.section || ia in late, simCode.initialAlgorithms)
  isempty(whenAlgorithms) && return noRuntimePass
  local lhsNames = OrderedSet{String}()
  local rhsNames = OrderedSet{String}()
  local boolNames = OrderedSet{String}()
  for ia in whenAlgorithms, op in ia.statements
    _collectInitAlgNames!(lhsNames, rhsNames, op)
    _collectBoolRhsCrefs!(boolNames, op)
  end
  local renamedNames = union(lhsNames, rhsNames)
  push!(renamedNames, "time")
  local stmts = Expr[_initialWhenOpToJulia(op, simCode, renamedNames) for ia in whenAlgorithms for op in ia.statements]
  isempty(stmts) && return noRuntimePass
  #= Pre-fetch every non-parameter right-side reference. Names that ALSO appear as left sides
     still need a fetch because Julia compiles `x = if c then v else x end` with `x` as a
     function-local: the else-branch reads `x` before the assignment completes and throws
     UndefVarError. Self-referential if-expressions come from the algorithm lifter when a
     non-when algorithm contains an if/elseif chain whose else-branches keep a discrete left
     side's previous value. =#
  local fetches = Expr[]
  local ht = simCode.stringToSimVarHT
  for name in rhsNames
    name == "time" && continue
    if haskey(ht, name)
      local sv = last(ht[name])
      if sv.varKind isa Union{SC.PARAMETER, SC.ARRAY_PARAMETER}
        #= A parameter without a binding (fixed = false, set by the initialization: MSL 4
           DCPowerSupply's vLim from its initial algorithm) is no module global: its value is the
           problem's. =#
        if sv.varKind isa SC.PARAMETER && sv.varKind.bindExp === nothing
          local psym = Symbol(name)
          push!(fetches, quote
            local $(psym)
            try
              $(psym) = LATEST_PROBLEM.ps[$(QuoteNode(psym))]
            catch _e
              OMBackend._fallback(_e, :initAlgGetpValue)
              push!(_unfetched, $(QuoteNode(psym)))
            end
          end)
        end
        continue
      end
      if sv.varKind isa Union{SC.DATA_STRUCTURE, SC.STRING}
        push!(fetches, Expr(:local, Expr(:(=), Symbol(name), :(getfield(@__MODULE__, $(QuoteNode(Symbol(sv.name))))))))
        continue
      end
      #= A discrete Real as it is, an Integer as an Int. An enumeration (Logic) is an array
         index: MTK's initialization may leave it at 0, which BoundsErrors on 1-based index
         vectors (INV3S's UX01Conv[iNV3S_enable]; NXFER), so it is clamped to 1. Every discrete
         was rounded and clamped (`n = m + 5`, m = -2: n = 6, OpenModelica 3; a discrete Real 0.3
         read as 1). Not a Boolean: the clamp made false true (an ideal thyristor's `fire`).
         One the problem does not have stays unassigned, as below (it was 1). =#
      if sv.varKind isa SC.DISCRETE && !(name in boolNames) && !_isBoolDiscreteName(name, simCode)
        local sym = Symbol(name)
        local value = sv.elementType isa SC.TYPE_REAL ? :(Float64(_raw)) :
          sv.elementType isa SC.TYPE_INTEGER ? :(Int(round(Float64(_raw)))) : :(max(1, Int(round(Float64(_raw)))))
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
    #= Non-discrete variables (Real states, algebraic variables) and local algorithm
       temporaries not in the table: fetched as Float64, no index clamp. One the problem does
       not have stays unassigned: a temporary is assigned before it is read, and a read of a
       variable the backend eliminated is refused below (it was 0.0: `s = sum(v)` of a folded v
       gave 0, OpenModelica 12). =#
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
  #= `time` the start time (t0): it shadows `Base.time` (a UNIX-time function), which the
     bodies that read `time` (trapezoid sources' `count := integer((time - startTime) /
     period)`) subtracted from (a MethodError). `_hard` collects (symbolic_var => value) pairs
     for each assignment to a non-parameter variable: simulate() passes it to `remake(prob;
     u0=…, initializealg=NoInit())` so MTK treats the values as hard initial conditions
     (Modelica §11.2), not guesses. =#
  return quote
    function __runInitialAlgorithm!(t0::Float64 = 0.0)
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
