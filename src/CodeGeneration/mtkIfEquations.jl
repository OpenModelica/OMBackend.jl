#= MTK code generation: if-equations, their branch conditions and events, the time-event refresh. =#

#= Build one SymbolicContinuousCallback per deferred pure-time event. Callback K
   fires at event K's threshold, so its affect sets ITS OWN ifCond by the
   crossing's direction (rising zc: the condition becomes FALSE, falling: TRUE;
   a periodic condition such as sin(time) > 0.5 crosses both ways) and
   re-derives every OTHER pure-time ifCond from its zero-crossing sign
   (`zc < 0` <=> condition TRUE). The events are located on the left of their
   roots, so an event that coincides with K (the same threshold, e.g. the
   BooleanSteps of the phases of a switch sharing startTime) still shows its
   pre-crossing sign at t: read at t, each coincident affect reset the others
   and only the last one stayed switched. The other conditions are therefore
   read just after t (`_TIME_EVENT_AHEAD`): a coincident crossing is taken as
   happened, a condition away from its crossing keeps its sign, and one that
   only touches zero there keeps its value. A pure-time zc depends on t and
   parameters only, so shifting t needs no derivative (mod/floor included). =#
#= How far after an event, relative to 1 + |t|, the other pure-time conditions are read:
   crossings closer than this coincide with the event (1e-10 s near t = 1, 1e-5 s at
   t = 1e5). Each callback substitutes every other condition, n(n - 1) in all. =#
const _TIME_EVENT_AHEAD = 1.0e-10

function _buildTimeEventRefreshCallbacks(allPT::Vector, simCode)
  local n = length(allPT)
  local cbs = Expr[]
  for k in 1:n
    local mtkCondK = allPT[k][3]
    local obsKws = Expr[]
    local modKws = Expr[]
    local retKwsBy = Dict(0.0 => Expr[], 1.0 => Expr[])     # own value after a rising / falling crossing
    for j in 1:n
      local symJ = allPT[j][1]
      push!(modKws, Expr(:kw, symJ, symJ))
      if j == k
        push!(retKwsBy[0.0], Expr(:kw, symJ, 0.0))
        push!(retKwsBy[1.0], Expr(:kw, symJ, 1.0))
      else
        local zcName = Symbol("_zc", j)
        local ahead = :(t + $(_TIME_EVENT_AHEAD) * (1 + abs(t)))
        push!(obsKws, Expr(:kw, zcName, :(Symbolics.substitute($(allPT[j][2]), Dict(t => $(ahead))))))
        local refresh = Expr(:kw, symJ, :((observed.$(zcName) < 0) ? 1.0 : 0.0))
        push!(retKwsBy[0.0], refresh)
        push!(retKwsBy[1.0], refresh)
      end
    end
    local affectFor = function (ownVal::Float64)
      local retKws = retKwsBy[ownVal]
      local modNT = Expr(:tuple, Expr(:parameters, modKws...))
      local retNT = Expr(:tuple, Expr(:parameters, retKws...))
      local fExpr = :((modified, observed, ctx, integrator) -> $(retNT))
      if isempty(obsKws)
        return :(ModelingToolkit.ImperativeAffect($(fExpr), $(modNT); skip_checks = true))
      end
      local obsNT = Expr(:tuple, Expr(:parameters, obsKws...))
      return :(ModelingToolkit.ImperativeAffect($(fExpr), $(modNT);
                                                observed = $(obsNT), skip_checks = true))
    end
    push!(cbs, :(ModelingToolkit.SymbolicContinuousCallback(
      ($(mtkCondK)) => $(affectFor(0.0));
      affect_neg = $(affectFor(1.0)),
      reinitializealg = SciMLBase.NoInit()
    )))
  end
  return cbs
end

"""
  Creates the components of the If-Equations.
Each if equation is marked by the identifier.
So the first will have 1 and so on.
"""
function createIfEquations(stateVariables, algebraicVariables, simCode)
  local ifEquations = IfEquationComponent[]
  local identifier::Int
  local sortedIfEquations = sort(collect(simCode.ifEquations);
                                 by = ifEq -> _ifEquationSortKey(ifEq, simCode))
  #= Shared relay-t0 map: targets computed by earlier if-equations feed the
     condition initial values of later ones. =#
  local relayT0 = OrderedDict{Symbol, Float64}()
  #= The identifier is increased by 1 in each iteration. =#
  for (identifier, ifEq) in enumerate(sortedIfEquations)
    push!(ifEquations, createIfEquation(stateVariables, algebraicVariables, ifEq, identifier, simCode, relayT0))
  end
  #= Pure-time-event branches deferred their callbacks (see createIfEquation);
     build the model-level refresh callbacks now that every if-equation's
     pure-time conditions are known. =#
  local allPT = collect(Iterators.flatten(c.pureTimeEvents for c in ifEquations))
  if !isempty(allPT)
    local refreshCbs = _buildTimeEventRefreshCallbacks(allPT, simCode)
    push!(ifEquations, IfEquationComponent(refreshCbs, Expr[], Symbol[],
                                           Tuple{String, Bool}[], Tuple{Symbol, Any, Any, Float64}[],
                                           Expr[]))
  end
  return ifEquations
end

function _ifEquationSortKey(ifEq::SimulationCode.IF_EQUATION, simCode)::String
  local targets = String[]
  try
    for branch in ifEq.branches
      for resEq in branch.residualEquations
        push!(targets, string(last(deCausalize(resEq, simCode))))
      end
      isempty(targets) || break
    end
  catch _e
    OMBackend._fallback(_e, :ifEquationSortKeyTargets)
    empty!(targets)
  end
  if isempty(targets)
    try
      for branch in ifEq.branches
        push!(targets, string(branch.condition))
      end
    catch _e
      OMBackend._fallback(_e, :ifEquationSortKeyString)
      return ""
    end
  end
  sort!(targets)
  return join(targets, "|")
end

function _ifConditionDependsOnTime(@nospecialize(condition))::Bool
  local refs::OrderedSet{String} = OrderedSet{String}()
  try
    SimulationCode.collectCrefNames!(refs, condition)
  catch _e
    OMBackend._fallback(_e, :ifConditionDependsOnTime)
    return false
  end
  return "time" in refs
end

#= _ifConditionAllDiscreteOrParameter / _allBranchConditionsDiscrete live in the
   MTK_CodeGenerationUtil submodule (generateIfExpressions needs them); call them
   here as MTK_CodeGenerationUtil._allBranchConditionsDiscrete. =#

"""
    _ifConditionIsPureTimeEvent(condition, simCode) -> Bool

Return true when `condition` is a deterministic time event: it references
`time` and every other reference is a PARAMETER (no STATE / ALG / DISCRETE).
The transition instant is then fixed a priori, so coincident time events
(two sources transitioning at the same instant) must all be applied at once.
Conservative: any non-parameter reference returns false, keeping the default
per-branch continuous callback.
"""
function _ifConditionIsPureTimeEvent(@nospecialize(condition), simCode)::Bool
  local refs::OrderedSet{String} = OrderedSet{String}()
  try
    SimulationCode.collectCrefNames!(refs, condition)
  catch _e
    OMBackend._fallback(_e, :ifConditionIsPureTimeEvent)
    return false
  end
  ("time" in refs) || return false
  local ht = simCode.stringToSimVarHT
  for name in refs
    name == "time" && continue
    local entry = get(ht, name, nothing)
    entry === nothing && return false
    (entry[2].varKind isa SimulationCode.PARAMETER) || return false
  end
  return true
end

"""
This function creates symbolic if equations for use in MTK.
The function returns a tuple, where the first part of the tuple represent the conditions and the affect of the if-equation on the form:
  continuous_events = [
    <Condition> => <affect>
    <Condition> => <affect>
    ....
  ]
Each condition generates one variable with zero dynamics the variable being true or not depending on the branch.
  Example:
  if <condition> then
    <equations>
  elseif <condition> then
    <equations>
  else
    <equations>
  end if;
Would result in:
continuous_events = [
    <condition> => [ifCond1 ~ true, ifCond2 ~ false]
    <condition> => [ifCond1 ~ false, ifCond2 ~ true]
]
An if equation with a single condition would only generate one condition:
continuous_events = [
    <condition> => [ifCond1 ~ true]
]

The second value in the return tuple represent the if-equations itself:
<lhs> = IfElse.ifelse(<condition>, <value>, IfElse.ifelse(<condition>, <value>, <value>))
  lhs can be one or several variables. (TODO, fix the case for several variables in this kind of branch)

The third part of the tuple contains a set of zero dynamic equations (One for each if equation condition variable)
See the following issue: https://github.com/SciML/ModelingToolkit.jl/issues/1523

The forth part of the tuple contains a vector of symbolic variables.
One for each conditional variable created.
"""
#= True when a branch condition references a variable that is itself an
   if-equation relay target (key of the shared relay-t0 map): exactly the
   chained staged-trajectory case where a boundary crossing must re-evaluate
   sibling conditions live instead of applying static toggles. =#
function _conditionReferencesRelayTarget(@nospecialize(condition), rT0)::Bool
  isempty(rT0) && return false
  local refs::OrderedSet{String} = OrderedSet{String}()
  try
    SimulationCode.collectCrefNames!(refs, condition)
  catch _e
    OMBackend._fallback(_e, :conditionReferencesRelayTarget)
    return false
  end
  for name in refs
    haskey(rT0, Symbol(name)) && return true
  end
  return false
end

#= Initial branch-condition values consistent with the t0 values of the
   targets this if-equation defines. Round: evaluate every condition with the
   current relay-t0 map, select the branch, evaluate the selected branch's
   target RHS values at t0 and feed them back; stop when the condition vector
   is stable. Mutates `rT0` so later if-equations see earlier targets. =#
function _fixedPointInitialConditions(ifEq::SimulationCode.IF_EQUATION, simCode, rT0)::Vector{Bool}
  OMBackend.envSwitch("OMBACKEND_RELAY_T0_FIXEDPOINT") || return Bool[]
  local condBranches = [b for b in ifEq.branches if b.identifier != -1]
  local elseBranch = nothing
  for b in ifEq.branches
    b.identifier == -1 && (elseBranch = b)
  end
  local conds = Any[]
  local closed = Bool[]
  for b in condBranches
    push!(conds, transformToMTKContinuousConditionEquation(b.condition, simCode; atInitial = true))
    push!(closed, MTK_CodeGenerationUtil.condClosedAtBoundary(b.condition))
  end
  local valMap = nothing
  local explicit = Set{Symbol}()
  try
    (valMap, explicit) = MTK_CodeGenerationUtil._buildT0ValueMapAndExplicit(simCode)
  catch _e
    OMBackend._fallback(_e, :fixedPointT0ValueMap)
    valMap = nothing
  end
  local ivs = Bool[true for _ in condBranches]
  for _round in 1:8
    local newIvs = Bool[evalInitialCondition(conds[k], simCode; closedBoundary = closed[k], extraVals = rT0)
                        for k in 1:length(condBranches)]
    local sel = elseBranch
    for (k, b) in enumerate(condBranches)
      if !newIvs[k]
        sel = b
        break
      end
    end
    local rT0Changed = false
    if valMap !== nothing && sel !== nothing
      local mergedMap = copy(valMap)
      local mergedExplicit = copy(explicit)
      for (k, v) in rT0
        mergedMap[k] = v
        push!(mergedExplicit, k)
      end
      for r in sel.residualEquations
        local gv = try
          local (rhsE, lhsE) = deCausalize(r, simCode)
          local key = Symbol(MTK_CodeGenerationUtil._causalLhsKey(lhsE))
          local val = MTK_CodeGenerationUtil.evalCausalRHSAtT0(rhsE, mergedMap, mergedExplicit)
          val === nothing ? nothing : (key => Float64(val))
        catch _e
          OMBackend._fallback(_e, :fixedPointT0Eval)
          nothing
        end
        if gv !== nothing && (!haskey(rT0, gv.first) || rT0[gv.first] != gv.second)
          rT0[gv.first] = gv.second
          rT0Changed = true
          #= Later targets of the same round may depend on this one. =#
          mergedMap[gv.first] = gv.second
          push!(mergedExplicit, gv.first)
        end
      end
    end
    if OMBackend.envSwitch("OMBACKEND_RELAY_T0_TRACE")
      @info "[relayT0] round" _round newIvs rT0Changed nT0=length(rT0) rT0=collect(rT0)
    end
    if newIvs == ivs && !rT0Changed
      break
    end
    ivs = newIvs
  end
  return ivs
end

function createIfEquation(stateVariables::Vector,
                          algebraicVariables::Vector,
                          ifEq::SimulationCode.IF_EQUATION,
                          identifier::Int,
                          simCode,
                          relayT0 = nothing)::IfEquationComponent
  local i::Int = 0
  local nBranches::Int = length(ifEq.branches)
  local branchesWithConds::Int = nBranches - 1
  #= Fixed point between branch-condition initial values and the targets the
     selected branch defines: conditions may reference targets of this very
     if-equation (a hoisted staged trajectory), so a single static evaluation
     with those operands defaulted to 0 picks the wrong initial branch. =#
  local _rT0 = relayT0 === nothing ? OrderedDict{Symbol, Float64}() : relayT0
  local ivPre = _fixedPointInitialConditions(ifEq, simCode, _rT0)
  #= Zero crossings of every conditional branch, for the live sibling
     re-evaluation affect of multi-branch chains. =#
  local allZcs = Any[]
  local allClosed = Bool[]
  for b in ifEq.branches
    b.identifier == -1 && continue
    try
      local mc = transformToMTKContinuousConditionEquation(b.condition, simCode)
      push!(allZcs, _extractZeroCrossingLHS(mc))
      push!(allClosed, MTK_CodeGenerationUtil.condClosedAtBoundary(b.condition))
    catch _e
      OMBackend._fallback(_e, :ifEquationZeroCrossings, impact = :result)
      empty!(allZcs)
      empty!(allClosed)
      break
    end
  end
  #= Collect all ifCond symbols for this if-equation.
     These are parameters modified by imperative affects. =#
  local allIfCondSyms = [Symbol(string("ifCond", identifier, j)) for j in 1:branchesWithConds]
  local conditions = Expr[]
  local ivConditions = Bool[]
  local pureTimeEvents = Tuple{Symbol, Any, Any, Float64}[]
  #= ivPre is indexed over CONDITIONAL branches only; the loop counter `i`
     also advances over the else branch, so it must not index ivPre. =#
  local condIdx::Int = 0
  #= One callback per UNIQUE zero crossing for live-affect equations: staged
     chains repeat a boundary (two branches share t = Tvs), and two callbacks
     firing in succession leave a half-flipped relay between them - a torque
     slam. The live affect rewrites every sibling ifCond, so one suffices. =#
  local liveZcSeen = OrderedSet{String}()
  #= Live-affect qualification is per EQUATION, not per branch: in a staged
     chain some boundaries are plain parameters while others are relay
     targets. Mixing live and static toggles leaves the relay half-flipped
     at the static boundaries. =#
  local anyRelayCondBranch::Bool =
    any(b -> b.identifier != -1 && _conditionReferencesRelayTarget(b.condition, _rT0),
        ifEq.branches)
  local relations = Tuple{Symbol, Any, Any}[]
  local initLiterals = Expr[]
  for branch in ifEq.branches
    i += 1
    @match branch begin
      SimulationCode.BRANCH(condition, residuals, -1 #= Else =#, targets, _, _, _, _, _) => begin
      end
      SimulationCode.BRANCH(condition, residuals, _, targets, _, _, _, _, _) => begin
        condIdx += 1
        local mtkCond = transformToMTKContinuousConditionEquation(branch.condition, simCode)
        #= Evaluate the initial value condition; the original operator decides
           the zc == 0 boundary. Precomputed via the relay-aware fixed point. =#
        local _closedB = MTK_CodeGenerationUtil.condClosedAtBoundary(branch.condition)
        local ivCond = condIdx <= length(ivPre) ? ivPre[condIdx] :
                       evalInitialCondition(transformToMTKContinuousConditionEquation(branch.condition, simCode;
                                                                                      atInitial = true),
                                            simCode; closedBoundary = _closedB, extraVals = _rT0)
        local numVal = ivCond ? 1.0 : 0.0
        local invVal = ivCond ? 0.0 : 1.0
        #= Build ImperativeAffect: function returns a NamedTuple of new values.
           modified NamedTuple maps aliases to the symbolic parameter variables.
           Callback fires for every branch condition; ifCondN is the load-bearing
           branch switch the residual ifelse reads.

           Direction-aware toggle: the MTK convention here is `zcLhs < 0` <=>
           condition TRUE. The positive edge (`affect`, prev_sign < 0) is a
           true->false transition, the negative edge (`affect_neg`, prev_sign > 0)
           is false->true. So the firing branch's own ifCond is set false on the
           positive edge and true on the negative edge. A single constant value
           cannot toggle a condition that crosses repeatedly (e.g. a Pulse/periodic
           source waveform). Other branches' ifConds keep their current value:
           the branch conditions nest first-true-wins (generateIfExpressions), so
           each ifCond stands for its own condition only. =#
        local modifiedKws::Vector{Expr} = Expr[Expr(:kw, sym, sym) for sym in allIfCondSyms]
        local modifiedNT::Expr = Expr(:tuple, Expr(:parameters, modifiedKws...))
        local upKws::Vector{Expr}   = Expr[Expr(:kw, sym, (j == i) ? 0.0 : :(modified.$(sym))) for (j, sym) in enumerate(allIfCondSyms)]
        local downKws::Vector{Expr} = Expr[Expr(:kw, sym, (j == i) ? 1.0 : :(modified.$(sym))) for (j, sym) in enumerate(allIfCondSyms)]
        local upFExpr::Expr   = :((modified, observed, ctx, integrator) -> $(Expr(:tuple, Expr(:parameters, upKws...))))
        local downFExpr::Expr = :((modified, observed, ctx, integrator) -> $(Expr(:tuple, Expr(:parameters, downKws...))))
        local affectTuple::Expr     = :(($(upFExpr), $(modifiedNT)))
        local affectNegTuple::Expr  = :(($(downFExpr), $(modifiedNT)))
        #= Multi-branch chains whose conditions reference solved unknowns: a
           static toggle scrambles the selection when one boundary crossing
           hands over to a SIBLING branch (staged trajectories with computed
           phase times). Re-evaluate every sibling condition live from its own
           zero crossing on either edge; first-true-wins nesting keeps the
           relay consistent. Purely time/parameter-staged chains keep the
           static toggles: their transition instants are exact and the
           deferred pure-time refresh machinery owns them. =#
        local liveAffect = nothing
        if branchesWithConds > 1 && length(allZcs) == branchesWithConds &&
           anyRelayCondBranch &&
           OMBackend.envSwitch("OMBACKEND_LIVE_IFCOND_AFFECT")
          local liveKws = Expr[]
          local liveObsKws = Expr[]
          for (j, sym) in enumerate(allIfCondSyms)
            local zcName = Symbol("zc", j)
            local test = allClosed[j] ? :(observed.$(zcName) <= 0) : :(observed.$(zcName) < 0)
            push!(liveKws, Expr(:kw, sym, :($(test) ? 1.0 : 0.0)))
            push!(liveObsKws, Expr(:kw, zcName, allZcs[j]))
          end
          local liveFn = :((modified, observed, ctx, integrator) -> $(Expr(:tuple, Expr(:parameters, liveKws...))))
          local liveObsNT = Expr(:tuple, Expr(:parameters, liveObsKws...))
          liveAffect = :(ModelingToolkit.ImperativeAffect($(liveFn), $(modifiedNT);
                                                          observed = $(liveObsNT), skip_checks = true))
        end
        #= When the branch condition depends on a non-lifted algebraic variable
           (an operating-point value `evalInitialCondition` defaulted to 0, e.g.
           an op-amp input voltage), the static initial ifCond can be wrong with
           no zero-crossing to fire the affect. Add an `initialize` affect that
           re-evaluates the condition from the solved state (mirrors
           evalInitialCondition: zc < 0 means the condition is TRUE). Restricted
           to non-lifted algebraic zc so it does not observe other lifted ifEq_tmp
           values (which would form a circular init dependency). =#
        local zcLhs = _extractZeroCrossingLHS(mtkCond)
        local thisSym::Symbol = allIfCondSyms[i]
        if MTK_CodeGenerationUtil._ifConditionAllDiscreteOrParameter(branch.condition, simCode)
          #= A discrete or parameter condition: the residuals gate directly on it
             (generateIfExpressions), whose own update event localises the step.
             No crossing function, no callback. =#
          push!(ivConditions, ivCond)
        #= A condition with initial() is a relation (a constant crossing function
           of either sign), not a pure-time event: `initial() or time > T` held
           until T. =#
        elseif !MTK_CodeGenerationUtil._hasInitialCall(branch.condition) &&
               _ifConditionIsPureTimeEvent(branch.condition, simCode)
          #= Deterministic time event: defer to model-level refresh callbacks built
             in createIfEquations, so two sources whose transitions coincide cannot
             drop one another's affect. `numVal` is the post-crossing ifCond value
             (same value the per-branch toggle would set). The ifCond parameter is
             still declared and initialised below via ivConditions. =#
          push!(pureTimeEvents, (thisSym, zcLhs, mtkCond, numVal))
          push!(ivConditions, ivCond)
        else
          #= The crossing function with a hysteresis relative to the branch's
             buffered value (MLS 8.5; OpenModelica LessZC): a TRUE relation turns
             FALSE when zc > eps, a FALSE one TRUE when zc < -eps, eps = H*scale.
             g is never zero after initialization or an event, so a function
             starting at its threshold (a dead centre) or two coinciding roots
             cannot be missed or chatter. =#
          local scaleExpr = MTK_CodeGenerationUtil._conditionScaleExpr(branch.condition, simCode)
          local hystCond = :((($(zcLhs)) + $(ZC_HYSTERESIS) * ($(scaleExpr)) * (1 - 2 * $(thisSym))) ~ 0)
          push!(relations, (thisSym, zcLhs, scaleExpr))
          #= The literal value at initialization, from the solved initial state
             (the static value comes from start attributes). Not for conditions
             on other lifted if-expressions: their values are not ready then. =#
          local initAffect = nothing
          local litInit = nothing
          local litObsInit = Expr[]
          if !_exprMentionsPrefix(zcLhs, "ifEq_tmp")
            local litObs = Expr[]
            local lit = MTK_CodeGenerationUtil._literalConditionExpr(branch.condition, simCode, litObs)
            litInit = lit
            litObsInit = litObs
            if lit !== nothing
              local litObsNT = Expr(:tuple, Expr(:parameters, litObs...))
              local litModNT = Expr(:tuple, Expr(:parameters, Expr(:kw, thisSym, thisSym)))
              local litRetNT = Expr(:tuple, Expr(:parameters, Expr(:kw, thisSym, :(($(lit)) ? 1.0 : 0.0))))
              initAffect = :(ModelingToolkit.ImperativeAffect(((modified, observed, ctx, integrator) -> $(litRetNT)),
                                                              $(litModNT); observed = $(litObsNT), skip_checks = true))
            end
          end
          local cond::Expr
          local _dupLiveZc::Bool = false
          if liveAffect !== nothing
            local _zcKey = string(mtkCond)
            #= A sibling may already have registered this exact crossing; its
               live affect rewrites this branch's ifCond too. =#
            _dupLiveZc = _zcKey in liveZcSeen
            push!(liveZcSeen, _zcKey)
          end
          if _dupLiveZc
            cond = :(nothing)
          elseif liveAffect !== nothing
            #= Positional affect form: the pair form does not commit an
               ImperativeAffect's writes. =#
            cond = :(ModelingToolkit.SymbolicContinuousCallback(
              ($(hystCond)),
              $(liveAffect);
              affect_neg = $(liveAffect),
              initialize = $(liveAffect),
              rootfind = SciMLBase.RightRootFind,
              reinitializealg = SciMLBase.NoInit()
            ))
          elseif initAffect !== nothing
            #= Settled by the initialization too (MLS 8.6: a relation takes its
               literal value at the initial solution), unless it reads pre(),
               an event operator or time (the init solve is at t = 0). =#
            if !_conditionHasCall(branch.condition, ("pre", "initial", "terminal", "sample", "edge", "change", "delay")) &&
               !_ifConditionDependsOnTime(branch.condition)
              local obsNames = Symbol[kw.args[1] for kw in litObsInit]
              push!(initLiterals, :(($(QuoteNode(thisSym)), $(QuoteNode(obsNames)), Any[$([kw.args[2] for kw in litObsInit]...)],
                                     (observed -> $(litInit)))))
            end
            cond = :(ModelingToolkit.SymbolicContinuousCallback(
              ($(hystCond)) => $(affectTuple);
              affect_neg = $(affectNegTuple),
              initialize = $(initAffect),
              reinitializealg = $(_BRANCH_EVENT_REINIT)
            ))
          else
            cond = :(ModelingToolkit.SymbolicContinuousCallback(
              ($(hystCond)) => $(affectTuple);
              affect_neg = $(affectNegTuple),
              reinitializealg = $(_BRANCH_EVENT_REINIT)
            ))
          end
          _dupLiveZc || push!(conditions, cond)
          push!(ivConditions, ivCond)
        end
      end
    end
  end
  #= Create the equations themselves =#
  local target = 1
  local resEqs = ifEq.branches[target].residualEquations
  local ifExpressions = Expr[]
  #= The number of residuals is the same for both branches. =#
  local nResEqsInTarget = length(resEqs)
  #= t0-selected branch: first conditional branch whose condition is TRUE at
     t0 (ivCond stores the negation), else the else branch. Its causalized RHS
     evaluated at the t0 value map seeds soft guesses for the targets; zero
     default guesses put guarded denominators at 0/0 before the init solve. =#
  local condBranches = Any[]
  local elseBranch = nothing
  for branch in ifEq.branches
    if branch.identifier == -1
      elseBranch = branch
    else
      push!(condBranches, branch)
    end
  end
  local selBranch = elseBranch
  for (k, branch) in enumerate(condBranches)
    if k <= length(ivConditions) && !(ivConditions[k])
      selBranch = branch
      break
    end
  end
  local relayGuesses = Expr[]
  local _t0ValMap = nothing
  local _t0Explicit = Set{Symbol}()
  if selBranch !== nothing
    try
      (_t0ValMap, _t0Explicit) = MTK_CodeGenerationUtil._buildT0ValueMapAndExplicit(simCode)
    catch _e
      OMBackend._fallback(_e, :ifEquationT0ValueMap)
      _t0ValMap = nothing
    end
  end
  #= Branches that define different variables (a switch with an arc:
     i = Goff*v while quenched, v = Ron*i when closed) take the residual form,
     the whole if-equation (pairing by variable and by position must not
     mix); the causal one put each branch's right-hand side on the first
     branch's variable (i = Ron*i). =#
  local branchKeys = [Set{String}(MTK_CodeGenerationUtil._causalLhsKey(last(deCausalize(r, simCode)))
                                  for r in b.residualEquations) for b in ifEq.branches]
  local causal = all(r -> all(ks -> MTK_CodeGenerationUtil._causalLhsKey(last(deCausalize(r, simCode))) in ks,
                              branchKeys), resEqs)
  for resEqIdx in 1:nResEqsInTarget
    local resEq = resEqs[resEqIdx]
    local lhsExpr = last(deCausalize(resEq, simCode))
    local lhsKey = MTK_CodeGenerationUtil._causalLhsKey(lhsExpr)
    push!(ifExpressions,
          :($(causal ? lhsExpr : 0) ~ $(generateIfExpressions(ifEq.branches,
                                                              target,
                                                              resEqIdx,
                                                              identifier,
                                                              simCode;
                                                              subIdentifier = 1,
                                                              lhsKey = lhsKey,
                                                              residualForm = !causal))))
    if causal && _t0ValMap !== nothing && !isempty(selBranch.residualEquations)
      local gv = try
        local selEq = MTK_CodeGenerationUtil._branchResidualForLhs(selBranch, lhsKey, resEqIdx, simCode)
        MTK_CodeGenerationUtil.evalCausalRHSAtT0(
          first(deCausalize(selEq, simCode)), _t0ValMap, _t0Explicit)
      catch _e
        OMBackend._fallback(_e, :ifEquationT0Eval)
        nothing
      end
      gv === nothing || push!(relayGuesses, :($(string(_unwrapBlockExpr(lhsExpr))) => $(gv)))
    end
  end
  #= ifCond variables are discrete parameters (not ODE unknowns), so they do
     not need der() ~ 0 equations. Collect their names and initial values for
     parameter declaration. =#
  local conditionVariables = Symbol[]
  local conditionVariableNames = Tuple{String, Bool}[]
  for i in 1:length(ivConditions)
    push!(conditionVariables, Symbol(string("ifCond", identifier, i)))
    push!(conditionVariableNames, (string("ifCond", identifier, i), !(ivConditions[i])))
  end
  return IfEquationComponent(conditions, ifExpressions,
                             conditionVariables, conditionVariableNames, pureTimeEvents,
                             relayGuesses, relations, initLiterals)
end

#= Whether a condition calls one of `names`. =#
function _conditionHasCall(@nospecialize(cond), names)::Bool
  cond isa SimulationCode.Exp && (cond = SimulationCode.toDAEExp(cond))
  local found = Ref(false)
  Util.traverseExpBottomUp(cond, (x, acc) -> begin
    x isa DAE.CALL && x.path isa Absyn.IDENT && x.path.name in names && (found[] = true)
    (x, true, acc)
  end, 0)
  return found[]
end
