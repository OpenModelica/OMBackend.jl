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
    DAE.RANGE(start = s, step = st, stop = e) => begin
      collectCrefNames!(names, s)
      st isa SOME && collectCrefNames!(names, st.data)
      collectCrefNames!(names, e)
    end
    DAE.MATRIX(matrix = rows) => foreach(row -> foreach(x -> collectCrefNames!(names, x), row), rows)
    DAE.TUPLE(PR = lst) => foreach(x -> collectCrefNames!(names, x), lst)
    DAE.RECORD(exps = lst) => foreach(x -> collectCrefNames!(names, x), lst)
    DAE.SIZE(exp = e, sz = sz) => begin
      collectCrefNames!(names, e)
      sz isa SOME && collectCrefNames!(names, sz.data)
    end
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

"""
    extractCrefName(exp::DAE.Exp)

Extract the variable name from a CREF or ASUB(CREF, ...) expression.
Returns `(name::String, cref::DAE.ComponentRef, ty::DAE.Type)` or `nothing`
if the expression is not a simple variable reference.
"""
function extractCrefName(@nospecialize(exp))
  # SIM.EXP_CREF (post-Phase-4b when-ASSIGN LHS) → DAE.CREF so the match fires.
  if exp isa Exp
    exp = toDAEExp(exp)
  end
  @match exp begin
    DAE.CREF(cr, ty) => begin
      return (DAE_identifierToString(cr), cr, ty)
    end
    #= ASUB-wrapped CREFs are skipped for alias detection.
       The ASUB wraps a base CREF with subscripts, but the CREF itself does not
       carry the subscripts. Eliminating an ASUB alias would replace the base CREF
       in all equations (affecting all subscripts), breaking the equation balance.
       These equations are better handled by MTK structural_simplify. =#
    _ => return nothing
  end
end

"""
    isUnknownVarKind(varKind::SimVarType)::Bool

Check if a variable kind represents an unknown (not a parameter or constant).
Only unknowns participate in the equation-unknown balance.
"""
function isUnknownVarKind(@nospecialize(varKind::SimVarType))::Bool
  @match varKind begin
    STATE(__) => true
    STATE_DERIVATIVE(__) => true
    ALG_VARIABLE(__) => true
    ARRAY(__) => true
    DISCRETE(__) => true
    _ => false
  end
end

#= For every DAE.CREF with T_COMPLEX type in the given equations, append
   `<base>_<fieldname>` for each field of the complex record when that scalar
   name exists in the simvar hash table. Used to protect those scalar params
   from constant-elimination — codegen later flattens the complex CREF into
   the scalar field symbols, which must resolve at module eval time. =#
# Per-cref handler for complex-field protection: identical logic on a DAE.CREF
# leaf whether reached via the SIM walk or the DAE fallback.
function _complexCrefFields!(names::OrderedSet{String}, @nospecialize(dcref), ht)
  @match dcref begin
    DAE.CREF(cr, ty) => begin
      local baseName = DAE_identifierToString(cr)
      if ty isa DAE.T_COMPLEX
        for field in ty.varLst
          local fieldName = Base.string(baseName, "_", field.name)
          if haskey(ht, fieldName)
            push!(names, fieldName)
          end
        end
      else
        #= Fallback: any cref X whose X_re and X_im scalars exist in HT.
           Codegen will flatten X via flattenRecordCallArg into [X_re, X_im];
           protect both even when the cref's ty was downgraded from T_COMPLEX. =#
        local reName = Base.string(baseName, "_re")
        local imName = Base.string(baseName, "_im")
        if haskey(ht, reName) && haskey(ht, imName)
          push!(names, reName)
          push!(names, imName)
        end
      end
    end
    _ => nothing
  end
  return nothing
end

# Pure read-only walk over the SIM tree; convert only cref leaves to DAE
# (toDAEExp(EXP_CREF) gives the same DAE.CREF the whole-tree conversion would).
# Avoids building and rebuilding a parallel DAE tree per equation.
function _walkComplexSIM!(names::OrderedSet{String}, e::Exp, ht)
  if e isa EXP_CREF
    _complexCrefFields!(names, toDAEExp(e), ht)
  elseif e isa IFEXP
    _walkComplexSIM!(names, e.cond, ht)
    _walkComplexSIM!(names, e.thenExp, ht)
    _walkComplexSIM!(names, e.elseExp, ht)
  elseif e isa BINARY || e isa LBINARY || e isa RELATION
    _walkComplexSIM!(names, e.exp1, ht)
    _walkComplexSIM!(names, e.exp2, ht)
  elseif e isa UNARY || e isa LUNARY
    _walkComplexSIM!(names, e.exp, ht)
  elseif e isa CALL
    for a in e.args
      _walkComplexSIM!(names, a, ht)
    end
  elseif e isa ARRAY_EXP
    for x in e.elements
      _walkComplexSIM!(names, x, ht)
    end
  elseif e isa ASUB
    _walkComplexSIM!(names, e.exp, ht)
    for s in e.subs
      _walkComplexSIM!(names, s, ht)
    end
  elseif e isa TSUB || e isa RSUB || e isa CAST
    _walkComplexSIM!(names, e.exp, ht)
  elseif e isa RECORD
    for x in e.exps
      _walkComplexSIM!(names, x, ht)
    end
  elseif e isa TUPLE
    for x in e.PR
      _walkComplexSIM!(names, x, ht)
    end
  elseif e isa REDUCTION
    _walkComplexSIM!(names, e.body, ht)
  end
  return names
end

_complexCrefDAEVisitor(@nospecialize(exp), ctx) =
  (_complexCrefFields!(ctx[1], exp, ctx[2]); (exp, true, ctx))

function _collectComplexFieldNames!(names::OrderedSet{String}, eqs, ht)
  for eq in eqs
    if eq isa RESIDUAL_EQUATION
      _walkComplexSIM!(names, eq.exp, ht)
    elseif eq isa EQUATION
      _walkComplexSIM!(names, eq.lhs, ht)
      _walkComplexSIM!(names, eq.rhs, ht)
    elseif eq isa ARRAY_EQUATION
      _walkComplexSIM!(names, eq.left, ht)
      _walkComplexSIM!(names, eq.right, ht)
    elseif eq isa BDAE.RESIDUAL_EQUATION
      #= BDAE equations carry DAE.Exp fields; fall back to the DAE traversal. =#
      Util.traverseExpTopDown(eq.exp, _complexCrefDAEVisitor, (names, ht))
    elseif eq isa BDAE.EQUATION
      Util.traverseExpTopDown(eq.lhs, _complexCrefDAEVisitor, (names, ht))
      Util.traverseExpTopDown(eq.rhs, _complexCrefDAEVisitor, (names, ht))
    elseif eq isa BDAE.COMPLEX_EQUATION || eq isa BDAE.ARRAY_EQUATION
      Util.traverseExpTopDown(eq.left, _complexCrefDAEVisitor, (names, ht))
      Util.traverseExpTopDown(eq.right, _complexCrefDAEVisitor, (names, ht))
    end
  end
  return names
end

"""
Collect all CREF names from a WHEN_STMTS node (condition + statements + elsewhen).
"""
function _collectWhenCrefNames!(names::OrderedSet{String}, whenStmts::WHEN_STMTS)
  collectCrefNames!(names, whenStmts.condition)
  for stmt in whenStmts.whenStmtLst
    if stmt isa ASSIGN
      collectCrefNames!(names, stmt.left)
      collectCrefNames!(names, stmt.right)
    elseif stmt isa REINIT
      collectCrefNames!(names, stmt.stateVar)
      collectCrefNames!(names, stmt.value)
    elseif stmt isa NORETCALL
      collectCrefNames!(names, stmt.exp)
    elseif stmt isa ASSERT
      collectCrefNames!(names, stmt.condition)
      collectCrefNames!(names, stmt.message)
    end
  end
  if whenStmts.elsewhenPart !== nothing
    _collectWhenCrefNames!(names, whenStmts.elsewhenPart)
  end
  return names
end

function _collectWhenCrefNames!(names::OrderedSet{String}, whenStmts::BDAE.WHEN_STMTS)
  collectCrefNames!(names, whenStmts.condition)
  for stmt in whenStmts.whenStmtLst
    @match stmt begin
      BDAE.ASSIGN(__) => begin
        collectCrefNames!(names, stmt.left)
        collectCrefNames!(names, stmt.right)
      end
      BDAE.REINIT(__) => begin
        collectCrefNames!(names, stmt.stateVar)
        collectCrefNames!(names, stmt.value)
      end
      BDAE.NORETCALL(__) => collectCrefNames!(names, stmt.exp)
      BDAE.ASSERT(__) => begin
        collectCrefNames!(names, stmt.condition)
        collectCrefNames!(names, stmt.message)
      end
      _ => ()
    end
  end
  @match whenStmts.elsewhenPart begin
    SOME(elseWhenEq) => _collectWhenCrefNames!(names, elseWhenEq.whenEquation)
    NONE() => ()
  end
  return nothing
end
