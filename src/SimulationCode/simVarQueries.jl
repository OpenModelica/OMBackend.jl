#= Variable predicates and lookups, index assignment, irreducible variables, cref collection. =#

"""
  Returns true if simvar is either a algebraic or a state variable.
"""
function isStateOrAlgebraic(simvar::SimVar)::Bool
  return isAlgebraic(simvar) || isState(simvar)
end

"""
  Returns true if the simulation code variable is discrete.
"""
function isDiscrete(simVar::SimVar)::Bool
  res = @match simVar.varKind begin
    DISCRETE(__) => true
    _ => false
  end
end

"""
  Returns true if simvar is an algebraic variable.
"""
function isAlgebraic(simvar::SimVar)::Bool
  res = @match simvar.varKind begin
    ALG_VARIABLE(__) => true
    _ => false
  end
end

"""
  Returns true if the variable is a parameter.
"""
function isParameter(simvar::SimVar)::Bool
  res = @match simvar.varKind begin
    PARAMETER(__) => true
    _ => false
  end
end

"""
Returns true if the parameter has a binding expression.
"""
function hasBindingExp(simvar::SimVar)::Bool
  @match simvar.varKind begin
    PARAMETER(SOME(_)) => true
    _ => false
  end
end


"""
 The identifier of a variable's name as a string: the name of an unqualified
 one, the part after the first qualifier of a qualified one.
"""
function getInnerIdentOfVar(var)::String
  res = @match var.varName begin
    DAE.CREF_IDENT(ident) => begin
      ident
    end
    DAE.CREF_QUAL(ident = ident, componentRef = componentRef) => begin
      componentRef
    end
  end
  return string(res)
end


"
Returns true if simvar is  an algebraic variable
"
function isState(simvar::SimVar)::Bool
  res = @match simvar.varKind begin
    STATE(__) => true
    _ => false
  end
end

"""
  Prints what equation involves which variable.
The ht maps a string to the simcode variable structure in simcode data.
"""
function dumpVariableEqMapping(mapping::OrderedDict, residualEquations, ifEquations, whenEquations, ht)::String
  local dump = IOBuffer()
  println(dump, "VARIABLES:")
  for v in keys(ht)
    println(dump, v * ":" * string(first(ht[v])))
  end
  println(dump, "EQUATION MAPPING:")
  local equations = keys(mapping)
  for e in equations
    variablesAtEq = "{"
    for v in mapping[e]
      variablesAtEq *= "$(v),"
    end
    variablesAtEq *= "}"
    println(dump, "Equation $e: involves: $(variablesAtEq)\n")
  end
  for (i, e) in enumerate(residualEquations)
    println(dump, string("Equation " * string(i) * ":" * string(e)))
  end
  for (i, e) in enumerate(ifEquations)
    println(dump, string("IF-Equation " * string(i) * ":" * string(e)))
  end
  for (i, e) in enumerate(whenEquations)
    println(dump, string("WHEN-Equation " * string(i) * ":" * string(e)))
  end
  return String(take!(dump))
end


"""
Author: John & Andreas
   This function creates and assigns indices for variables
   Thus Construct the table that maps variable name to the actual variable.
It executes the following steps:
1. Collect all variables
2. Search all states (e.g. x and y) and give them indices starting at 1 (so x=1, y=2). Then give the corresponding state derivatives (x' and y') the same indices.
3. Remaining algebraic variables will get indices starting with i+1, where i is the number of states.
4. Parameters will get own set of indices, starting at 1.
5. Discrete shares the index with the states and starts at #states + 1
6. Data structure variables are only allowed as parameters and/or constants. They share the index with the parameters.
The index of discretes is updated after the state index is calculated.
"""
function createIndices(simulationVars::Vector{SimulationCode.SIMVAR})::OrderedDict{String, Tuple{Int, SimulationCode.SimVar}}
  local ht::OrderedDict{String, Tuple{Int, SimulationCode.SimVar}} = OrderedDict()
  local stateCounter = 0
  local parameterCounter = 0
  local discretes = SimulationCode.SIMVAR[]
  local complexVariables = SimulationCode.SIMVAR[]
  local arrayParameters = SimulationCode.SIMVAR[]
  local numberOfStates = 0
  for var in simulationVars
    @match var.varKind begin
      SimulationCode.STATE(__) => begin
        stateCounter += 1
        @assign var.index = SOME(stateCounter)
        stVar = SimulationCode.SIMVAR(var.name, var.index, SimulationCode.STATE_DERIVATIVE(var.name), var.attributes)
        push!(ht, var.name => (stateCounter, var))
        #= Adding the state derivative as well =#
        push!(ht, "der($(var.name))" => (stateCounter, stVar))
      end
      SimulationCode.PARAMETER(__) => begin
        parameterCounter += 1
        push!(ht, var.name => (parameterCounter, var))
      end
      SimulationCode.DISCRETE(__) => begin
        push!(discretes, var)
      end
      SimulationCode.DATA_STRUCTURE(__) => begin
        parameterCounter += 1
        push!(ht, var.name => (parameterCounter, var))
      end
      SimulationCode.STRING(__) => begin
        #parameterCounter += 1
        push!(discretes, var)
      end
      SimulationCode.ARRAY_PARAMETER(__) => begin
        push!(arrayParameters, var)
      end
      _ => continue
    end
  end
  #= Assign indices to array parameters =#
  local arrayParamCounter = parameterCounter
  for var in arrayParameters
    arrayParamCounter += 1
    @assign var.index = SOME(arrayParamCounter)
    push!(ht, var.name => (arrayParamCounter, var))
  end
  local discreteCounter = stateCounter
  for var in discretes
    discreteCounter += 1
    push!(ht, var.name => (discreteCounter, var))
  end
  local algIndexCounter::Int = discreteCounter
  local algSortingIdx::Int = stateCounter #This idx is used by the backend sorting algorithms
  for var in simulationVars
    @match var.varKind begin
      SimulationCode.ALG_VARIABLE(__) => begin
        algIndexCounter += 1
        algSortingIdx += 1
        @assign begin
          var.index = SOME(algIndexCounter)
          var.varKind = ALG_VARIABLE(algSortingIdx)
        end
        push!(ht, var.name => (var.index.data, var))
      end
      SimulationCode.ARRAY(__) => begin
        algIndexCounter += 1
        algSortingIdx += 1
        @assign var.index = SOME(algIndexCounter)
        push!(ht, var.name => (var.index.data, var))
      end
      _ => continue
    end
  end
  return ht
end

#= The variable assigned by an if-equation whose branches all assign the same
   single variable (`v = expression`, or its residual `v - expression`), or
   nothing. Every if-expression lifted by Causalize's IfExpressionLifter has
   this form (v is its ifEq_tmp temporary). =#
function _liftedIfTarget(eq::BDAE.IF_EQUATION)
  local lhsNames = Set{String}()
  for eqs in Iterators.flatten((eq.eqnstrue, (eq.eqnsfalse,)))
    for e in eqs
      local lhs = @match e begin
        BDAE.EQUATION(lhs = l) => l
        BDAE.RESIDUAL_EQUATION(exp = DAE.BINARY(l, DAE.SUB(__), _)) => l
        _ => nothing
      end
      lhs isa DAE.CREF || return nothing
      push!(lhsNames, DAE_identifierToString(lhs))
    end
  end
  return length(lhsNames) == 1 ? only(lhsNames) : nothing
end

_whenOperatorExps(op) = @match op begin
  BDAE.ASSIGN(left = l, right = r) => DAE.Exp[l, r]
  BDAE.REINIT(stateVar = v, value = e) => DAE.Exp[v, e]
  BDAE.ASSERT(condition = c, message = m) => DAE.Exp[c, m]
  BDAE.TERMINATE(message = m) => DAE.Exp[m]
  BDAE.NORETCALL(exp = e) => DAE.Exp[e]
  _ => DAE.Exp[]
end

#= The states a when-equation reads or writes (conditions and bodies, the
   elsewhen parts included). Its callbacks index the state vector by name
   (`x[lookuptableStates[name]]`, reinit), so these must stay unknowns. =#
function _statesUsedByWhens(whenEqs::Vector{BDAE.WHEN_EQUATION}, stateNames::Set{String})::Vector{String}
  local used = OrderedSet{String}()
  local visit = function (e)
    for cr in Util.getAllCrefs(e)
      local nm = string(cr)
      nm in stateNames && push!(used, nm)
    end
  end
  for weq in whenEqs
    local stmts = weq.whenEquation
    while true
      visit(stmts.condition)
      for op in stmts.whenStmtLst
        foreach(visit, _whenOperatorExps(op))
      end
      local next = @match stmts.elsewhenPart begin
        SOME(ew) => ew.whenEquation
        _ => nothing
      end
      next === nothing && break
      stmts = next
    end
  end
  return collect(used)
end

"""
  Get all variables that should be marked as irreducible (MTK must keep them
  as unknowns).
OBS:
Parameters are never added to this list.
The states are not irreducible: as in OpenModelica, the compiler chooses the
states (index reduction, tearing), and a start value that is not fixed is a
guess. Irreducible are:
- for an if-equation whose branches assign one variable (every lifted
  if-expression), that variable; for other if-equations, all their variables;
- the states that when-equations read or write (their callbacks index the
  state vector by name);
- the discretes in when conditions, and THETA.
"""
function getIrreducibleVars(ifEquations::Vector{BDAE.IF_EQUATION},
                             whenEqs::Vector{BDAE.WHEN_EQUATION},
                             algebraicAndStateVariables::Vector{BDAE.VAR},
                             ht::OrderedDict{String, Tuple{Int, SimulationCode.SimVar}})
  local irreducibles::Vector{Any} = []
  for eq in ifEquations
    local target = _liftedIfTarget(eq)
    if target !== nothing
      push!(irreducibles, [target])
    else
      variablesForEq = Backend.BDAEUtil.getAllVariables(eq, algebraicAndStateVariables)
      push!(irreducibles, variablesForEq)
    end
  end
  local stateNames = String[BDAE_identifierToVarString(v) for v in algebraicAndStateVariables if BDAEUtil.isState(v)]
  push!(irreducibles, _statesUsedByWhens(whenEqs, Set{String}(stateNames)))
  #=
    Parameters should not be marked as irreducible
    Remove them from the list
  =#
  irreducibles = collect(Iterators.flatten(irreducibles))
  irreducibles = filter(irv -> irv == "time" ||
                                  (haskey(ht, irv) && !isParameter(last(ht[irv]))),
                          irreducibles)
  local irreduciblesAsStr = map(x -> string(x), irreducibles)
  #= Protect discretes referenced in a when-CONDITION: if elimination drops one to
     observed-only, the DiscreteCallback condition still reads it from the state
     vector and hits x[nothing]. Condition + discrete only, to stay narrow. =#
  for weq in whenEqs
    local stmts = weq.whenEquation
    while stmts isa BDAE.WhenEquation
      for cref in Util.getAllCrefs(stmts.condition)
        local nm = string(cref)
        if haskey(ht, nm) && isDiscrete(last(ht[nm]))
          push!(irreduciblesAsStr, nm)
        end
      end
      stmts = stmts.elsewhenPart
    end
  end
  #=
  If THETA exists, treat it as an irreducible variable
  Currently, theta is a variable with "_THETA" in the variable name.
  This is subject to change
  =#
  thetaVariables = findall([endswith(x, "THETA") for x in keys(ht)])
  @assert length(thetaVariables) < 2
  if !(isempty(thetaVariables))
    #= Hardcoded for now can be fixed with annotation in the frontend =#
    push!(irreduciblesAsStr, collect(keys(ht))[first(thetaVariables)])
  end
  irreduciblesAsStr = filter(x -> x != "time", irreduciblesAsStr)
  return irreduciblesAsStr
end

"""
TODO: the name of the theta variable is hardcoded for now
Note that this function must be called before sorting.
"""
function handleZimmerThetaConstant(resEqs, irreducibleVars::Vector{String}, ht)
  thetaVariables = findall([endswith(x, "THETA") for x in keys(ht)])
  if !(isempty(thetaVariables))
    #= Hardcoded for now can be fixed with annotation in the frontend =#
    thetaConstant = collect(keys(ht))[first(thetaVariables)]
    push!(irreducibleVars, thetaConstant)
    tmpResEq = DAE.BINARY(
      DAE.CREF(DAE.CREF_IDENT(thetaConstant, DAE.T_REAL_DEFAULT, MetaModelica.list()), DAE.T_REAL_DEFAULT),
      DAE.SUB(DAE.T_REAL_DEFAULT),
      DAE.RCONST(1.0))
    push!(resEqs,
          BDAE.RESIDUAL_EQUATION(tmpResEq, DAE.emptyElementSource, BDAE.EQ_ATTR_DEFAULT_DYNAMIC))
    (zimmerThetaIdx, simVar) = ht[thetaConstant]
    @assign simVar.varKind = ALG_VARIABLE(0)
    ht[thetaConstant] = (zimmerThetaIdx, simVar)
  end
  return(resEqs, irreducibleVars)
end

function getSimVarByName(name::String, ht::AbstractDict{String, Tuple{Int, SimVar}})
  return last(ht[name])
end

function makeDummyVariableName(equationSystemName::String; idx::Int = 1)
  return Base.string(equationSystemName, "__dummy", idx)
end

"""
  Creates a dummy residual.
  The dummy residual specifies that the derivative of a dummy variable is zero.
  0 = dx(<dummy_name><idx>)/dt - 0
"""
function makeDummyResidualEquation(equationSystemName::String, idx::Int = 1)
  local dummyName = makeDummyVariableName(equationSystemName; idx = idx)
  local crefIdent = DAE.CREF_IDENT(dummyName, DAE.T_REAL_DEFAULT, MetaModelica.list())
  local crefExpression = DAE.CREF(crefIdent, DAE.T_REAL_DEFAULT)
  return BDAE.RESIDUAL_EQUATION(
    DAE.BINARY(
      DAE.CALL(Absyn.IDENT("der"), crefExpression <| MetaModelica.list(), DAE.callAttrBuiltinReal),
      DAE.SUB(DAE.T_REAL_DEFAULT),
      DAE.RCONST(0.0)),
    DAE.emptyElementSource,
    BDAE.EQ_ATTR_DEFAULT_DYNAMIC,
  )
end

"""
    buildBaseNameIndex(ht::OrderedDict{String, Tuple{Int, SimVar}})

Build a reverse index from base variable names (without subscripts) to all
subscripted full names in the hash table. For example, if the HT contains
"world_x[1]" and "world_x[2]", the result maps "world_x" => ["world_x[1]", "world_x[2]"].
This handles the ASUB case where `getAllCrefs` extracts a base CREF without subscripts.
"""
function buildBaseNameIndex(ht::OrderedDict{String, Tuple{Int, SimVar}})::Dict{String, Vector{String}}
  local index = Dict{String, Vector{String}}()
  for (varName, _) in ht
    local bi = findfirst('[', varName)
    local bn = bi === nothing ? varName : varName[1:(bi - 1)]
    if bn != varName
      if !haskey(index, bn)
        index[bn] = String[]
      end
      push!(index[bn], varName)
    end
  end
  return index
end

"""
    collectEquationVarNames(exp::DAE.Exp,
                            ht::OrderedDict{String, Tuple{Int, SimVar}},
                            baseNameToFullNames::Dict{String, Vector{String}})

Extract all variable names referenced by a DAE expression, using the robust
`Util.getAllCrefs` traversal (via `traverseExpTopDown`). Falls back to base-name
matching for ASUB-wrapped CREFs where subscripts are separated from the CREF.

Returns a OrderedSet{String} of variable names that exist in the HT.
"""
function collectEquationVarNames(exp::DAE.Exp,
                                 ht::OrderedDict{String, Tuple{Int, SimVar}},
                                 baseNameToFullNames::Dict{String, Vector{String}})::OrderedSet{String}
  local crefs::List{DAE.ComponentRef} = Util.getAllCrefs(exp)
  local names = OrderedSet{String}()
  for cr in crefs
    local name = DAE_identifierToString(cr)
    if haskey(ht, name)
      push!(names, name)
    else
      #= Base name fallback: the CREF may come from inside an ASUB expression,
         missing its subscripts. Match all subscripted variants conservatively. =#
      local bi = findfirst('[', name)
      local bn = bi === nothing ? name : name[1:(bi - 1)]
      if bn != name && haskey(ht, bn)
        #= The CREF itself has partial subscripts; try the full name and base =#
        push!(names, bn)
      end
      local lookupKey = haskey(baseNameToFullNames, name) ? name : bn
      if haskey(baseNameToFullNames, lookupKey)
        for fullName in baseNameToFullNames[lookupKey]
          push!(names, fullName)
        end
      end
    end
  end
  return names
end

"""
    rebuildMatchOrder(simCode::SIM_CODE)

Rebuild a fresh bipartite matching from the current equations and variables.
This is needed when the original matchOrder is stale (e.g. after const-prop
and alias-elim have removed equations and variables).

Returns `(matchOrder::Vector{Int}, nameToMatchIdx::Dict{String,Int}, matchIdxToName::Dict{Int,String})`
where `matchOrder[varMatchIdx] = eqIdx` (0 = unmatched).
"""
function rebuildMatchOrder(simCode::SIM_CODE)
  local ht = simCode.stringToSimVarHT
  local resEqs = simCode.residualEquations
  local nEqs = length(resEqs)
  #= Collect unknown variables (those that participate in matching) =#
  local nameToMatchIdx = Dict{String, Int}()
  local matchIdxToName = Dict{Int, String}()
  local matchIdx = 0
  for (varName, (_idx, sv)) in ht
    local isUnknown = @match sv.varKind begin
      STATE(__) => true
      STATE_DERIVATIVE(__) => true
      ALG_VARIABLE(__) => true
      SimulationCode.ARRAY(__) => true
      DISCRETE(__) => true
      _ => false
    end
    if isUnknown
      matchIdx += 1
      nameToMatchIdx[varName] = matchIdx
      matchIdxToName[matchIdx] = varName
    end
  end
  local nVars = matchIdx
  #= Build the base name index for robust CREF extraction =#
  local baseNameToFullNames = buildBaseNameIndex(ht)
  #= Build bipartite adjacency: for each equation, which variable match indices does it reference? =#
  #= Int-keyed: GraphAlgorithms.matching consumes only `.vals` positionally, so the
     interpolated "e$(i)" string keys were pure allocation/hashing overhead. =#
  local eqVarMapping = DataStructures.OrderedDict{Int, Vector{Int}}()
  for eqI in 1:nEqs
    local refs = collectEquationVarNames(toDAEExp(resEqs[eqI].exp), ht, baseNameToFullNames)
    local indices = Int[]
    for refName in refs
      if haskey(nameToMatchIdx, refName)
        push!(indices, nameToMatchIdx[refName])
      end
    end
    eqVarMapping[eqI] = sort(unique(indices))
  end
  #= The matching algorithm requires a square system (n used for both eq loop
     and assign array). For over-determined systems (nVars > nEqs), pad with
     dummy empty equations so the algorithm sees a square system. The dummy
     equations will remain unmatched. For under-determined systems (nEqs > nVars),
     skip since we cannot produce a valid matching. =#
  if nEqs > nVars
    @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] under-determined system ($nEqs equations, $nVars unknowns), skipping"
    return (Int[], nameToMatchIdx, matchIdxToName)
  end
  local nMatch = nVars
  if nVars > nEqs
    for dummyI in (nEqs + 1):nVars
      eqVarMapping[dummyI] = Int[]
    end
  end
  local matchOrder::Vector{Int}
  try
    local (_isSingular, mo) = GraphAlgorithms.matching(eqVarMapping, nMatch)
    matchOrder = mo
  catch e
    @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] matching failed, skipping DCE" exception=(e, catch_backtrace())
    return (Int[], nameToMatchIdx, matchIdxToName)
  end
  local nMatched = count(>(0), matchOrder)
  @debug "[SIMCODE: $(simCode.name): rebuildMatchOrder] $nEqs equations, $nVars unknowns, $nMatched matched"
  return (matchOrder, nameToMatchIdx, matchIdxToName)
end

"""
    identifyOutputOnlyVariables(simCode::SIM_CODE)

Identify variables and equations that do not influence the dynamic states.
Performs a backward reachability analysis from state and state-derivative equations
through the causalized equation dependency graph.

Returns `(outputOnlyVarNames::OrderedSet{String}, outputOnlyEqIndices::OrderedSet{Int})`.
Variables in the returned set are purely "output" (they can be computed from states
but do not feed back into any state derivative).
"""
#= Pure read-only cref-name collector over the SIM Exp tree. Walks the tree and
   pushes referenced names without reconstructing any nodes (unlike
   traverseExpTopDown, which rebuilds the tree and allocates). The ASUB arm
   reconstructs the subscripted key (e.g. "R_T[1][1]") so the use-def chain
   matches the scalarized hash-table keys, then descends into the base (pushing
   the bare name) and the subscripts. =#
function collectCrefNames!(names::OrderedSet{String}, exp::Exp)
  @match exp begin
    EXP_CREF(__) => push!(names, DAE_identifierToString(toDAECref(exp.cref).componentRef))
    BINARY(__) => begin collectCrefNames!(names, exp.exp1); collectCrefNames!(names, exp.exp2) end
    LBINARY(__) => begin collectCrefNames!(names, exp.exp1); collectCrefNames!(names, exp.exp2) end
    RELATION(__) => begin collectCrefNames!(names, exp.exp1); collectCrefNames!(names, exp.exp2) end
    UNARY(__) => collectCrefNames!(names, exp.exp)
    LUNARY(__) => collectCrefNames!(names, exp.exp)
    CAST(__) => collectCrefNames!(names, exp.exp)
    TSUB(__) => collectCrefNames!(names, exp.exp)
    RSUB(__) => collectCrefNames!(names, exp.exp)
    IFEXP(__) => begin
      collectCrefNames!(names, exp.cond)
      collectCrefNames!(names, exp.thenExp)
      collectCrefNames!(names, exp.elseExp)
    end
    ARRAY_EXP(__) => begin for x in exp.elements; collectCrefNames!(names, x) end end
    CALL(__) => begin for x in exp.args; collectCrefNames!(names, x) end end
    RECORD(__) => begin for x in exp.exps; collectCrefNames!(names, x) end end
    TUPLE(__) => begin for x in exp.PR; collectCrefNames!(names, x) end end
    REDUCTION(__) => collectCrefNamesForReduction(names, exp)
    ASUB(__) => collectCrefNamesForAsub(names, exp)
    _ => ()
  end
  return names
end

function collectCrefNames!(names::OrderedSet{String}, @nospecialize(exp))
  @match exp begin
    DAE.CREF(cr, _) => begin
      push!(names, DAE_identifierToString(cr))
    end
    DAE.BINARY(exp1 = e1, exp2 = e2) => begin
      collectCrefNames!(names, e1)
      collectCrefNames!(names, e2)
    end
    DAE.UNARY(exp = e1) => collectCrefNames!(names, e1)
    DAE.LUNARY(exp = e1) => collectCrefNames!(names, e1)
    DAE.LBINARY(exp1 = e1, exp2 = e2) => begin
      collectCrefNames!(names, e1)
      collectCrefNames!(names, e2)
    end
    DAE.CALL(expLst = args) => begin
      for arg in args
        collectCrefNames!(names, arg)
      end
    end
    DAE.IFEXP(expCond = c, expThen = t, expElse = e) => begin
      collectCrefNames!(names, c)
      collectCrefNames!(names, t)
      collectCrefNames!(names, e)
    end
    DAE.ARRAY(array = lst) => begin
      for e in lst
        collectCrefNames!(names, e)
      end
    end
    DAE.ASUB(exp = e, sub = subs) => collectCrefNamesForDAEAsub(names, e, subs)
    DAE.RELATION(exp1 = e1, exp2 = e2) => begin
      collectCrefNames!(names, e1)
      collectCrefNames!(names, e2)
    end
    DAE.CAST(exp = e) => collectCrefNames!(names, e)
    DAE.TSUB(exp = e) => collectCrefNames!(names, e)
    DAE.RSUB(exp = e) => collectCrefNames!(names, e)
    DAE.REDUCTION(expr = e, iterators = iters) => begin
      collectCrefNames!(names, e)
      for it in iters
        @match it begin
          DAE.REDUCTIONITER(exp = guardExp) => collectCrefNames!(names, guardExp)
          _ => ()
        end
      end
    end
    _ => ()
  end
  return nothing
end

"""
    _simConstSubscriptSuffix(subs::Vector{Exp}) -> Union{String, Nothing}

Build the `"[i][j]..."` suffix for an all-constant integer SIM subscript list.
Returns `nothing` if any subscript is non-constant or the list is empty.
"""
function _simConstSubscriptSuffix(subs::Vector{Exp})::Union{String, Nothing}
  local suffix = ""
  for s in subs
    local piece = @match s begin
      ICONST(i) => Base.string("[", i, "]")
      _ => nothing
    end
    piece === nothing && return nothing
    suffix = Base.string(suffix, piece)
  end
  return isempty(suffix) ? nothing : suffix
end

"""
    _daeConstSubscriptSuffix(subs) -> Union{String, Nothing}

DAE-side counterpart of `_simConstSubscriptSuffix` over a `DAE.ICONST` subscript
list. Returns `nothing` if any subscript is non-constant or the list is empty.
"""
function _daeConstSubscriptSuffix(@nospecialize(subs))::Union{String, Nothing}
  local suffix = ""
  for s in subs
    #= ASUB.sub is List{Subscript}; constant index is INDEX(ICONST). =#
    local piece = @match s begin
      DAE.INDEX(DAE.ICONST(i)) => Base.string("[", i, "]")
      _ => nothing
    end
    piece === nothing && return nothing
    suffix = Base.string(suffix, piece)
  end
  return isempty(suffix) ? nothing : suffix
end

"Collect cref names from a SIM `REDUCTION` body and its iterator range/guard exps."
function collectCrefNamesForReduction(names::OrderedSet{String}, exp::REDUCTION)
  collectCrefNames!(names, exp.body)
  #= iterators carry DAE.ReductionIterator range/guard exps (passed through by
     toDAEExp); collect their crefs to match the DAE collector exactly. =#
  for it in exp.iterators
    @match it begin
      DAE.REDUCTIONITER(exp = rangeExp) => collectCrefNames!(names, rangeExp)
      _ => ()
    end
  end
  return names
end

"""
    collectCrefNamesForAsub(names::OrderedSet{String}, exp::ASUB) -> names

Collect cref names from a SIM `ASUB`, reconstructing the subscripted key
(e.g. `"R_T[1][1]"`) for all-constant subscripts so the use-def chain matches
the scalarized hash-table keys.
"""
function collectCrefNamesForAsub(names::OrderedSet{String}, exp::ASUB)
  if exp.exp isa EXP_CREF
    local suffix = _simConstSubscriptSuffix(exp.subs)
    if suffix !== nothing
      push!(names, Base.string(DAE_identifierToString(toDAECref(exp.exp.cref).componentRef), suffix))
    end
  end
  collectCrefNames!(names, exp.exp)
  for s in exp.subs
    collectCrefNames!(names, s)
  end
  return names
end

"""
    collectCrefNamesForDAEAsub(names::OrderedSet{String}, e, subs) -> nothing

DAE-side counterpart of `collectCrefNamesForAsub`: reconstructs the subscripted
key for a `DAE.CREF` base with all-constant subscripts, then descends into the
base and subscript expressions.
"""
function collectCrefNamesForDAEAsub(names::OrderedSet{String}, @nospecialize(e), @nospecialize(subs))
  local asubHandled = false
  @match e begin
    DAE.CREF(cr, _) => begin
      local baseName = DAE_identifierToString(cr)
      local suffix = _daeConstSubscriptSuffix(subs)
      suffix === nothing || push!(names, Base.string(baseName, suffix))
      push!(names, baseName)
      asubHandled = true
    end
    _ => ()
  end
  asubHandled || collectCrefNames!(names, e)
  for s in subs
    collectCrefNames!(names, s)
  end
  return nothing
end

function _hasUnknownCref(exp, ht)::Bool
  local names = OrderedSet{String}()
  collectCrefNames!(names, exp)
  for name in names
    local entry = get(ht, name, nothing)
    if entry !== nothing && isUnknownVarKind(last(entry).varKind)
      return true
    end
  end
  return false
end
