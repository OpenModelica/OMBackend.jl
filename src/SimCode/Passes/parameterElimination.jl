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

#= Parameters of literal values folded; observation-only variables and dead parameters dropped. =#

#= ==================================================================================== =#
#= The names the passes must keep                                                       =#
#= ==================================================================================== =#

"The names the `start`, `fixed`, `min`, `max` and `nominal` attributes read: the code generators read them by name."
function attributeNames!(names::OrderedSet{String}, ht::HT; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  for (_, (_, v)) in ht
    local a = v.attributes
    for e in (a.start, a.fixed, a.min, a.max, a.nominal)
      e === nothing || crefNames!(names, e; dynamic)
    end
  end
  return names
end

"The names the conditions of the if-expressions in `e` read."
function ifConditionNames!(names::OrderedSet{String}, e::Exp; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  foreachExp(x -> x isa IFEXP && crefNames!(names, x.cond; dynamic), e)
  return names
end

"The names a when-clause's condition and its elsewhen conditions read."
function whenConditionNames!(names::OrderedSet{String}, w::WHEN_STMTS; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  crefNames!(names, w.condition; dynamic)
  w.elsewhenPart === nothing || whenConditionNames!(names, w.elsewhenPart; dynamic)
  return names
end

"The names the Modelica functions' bodies read (their locals' defaults too)."
function functionBodyNames!(names::OrderedSet{String}, functions::Vector{ModelicaFunction}; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  for f in functions
    f isa MODELICA_FUNCTION || continue
    for v in f.locals
      v.default === nothing || crefNames!(names, v.default; dynamic)
    end
    statementCrefNames!(names, f.statements; dynamic)
  end
  return names
end

"The names the initial equations read (an array equation's too)."
function initialEquationNames!(names::OrderedSet{String}, eqs::Vector{Equation}; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  for eq in eqs, e in equationExps(eq)
    crefNames!(names, e; dynamic)
  end
  return names
end

"The names the if-equations read: their conditions', their branches' residuals'."
function ifEquationNames!(names::OrderedSet{String}, ifEquations::Vector{IF_EQUATION}; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  for ifEq in ifEquations, b in ifEq.branches
    crefNames!(names, b.condition; dynamic)
    foreach(eq -> crefNames!(names, eq.exp; dynamic), b.residualEquations)
  end
  return names
end

"The names the asserts' conditions and messages read."
function assertNames!(names::OrderedSet{String}, asserts::Vector{ASSERT_EQUATION}; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  for a in asserts
    crefNames!(names, a.condition; dynamic)
    crefNames!(names, a.message; dynamic)
  end
  return names
end

"""
    dynamicallyIndexedElements!(names, simCode, dynamic)

Every element of the arrays `dynamic`, those read by a subscript that reads a variable (`crefNames!`'s, `dynamicArrays!`):
no element is named there, the code generators look it up at run time (CDL's TimeTable `y[:] = val[idx, :]`). A read
by a literal subscript (an Integer's, an enumeration's, a Boolean's: `p[E_b(2)]`) names its element, which stays a
variable or is folded like any other.
"""
function dynamicallyIndexedElements!(names::OrderedSet{String}, simCode::SIM_CODE, dynamic::Set{Symbol})::OrderedSet{String}
  isempty(dynamic) && return names
  for k::String in keys(simCode.stringToSimVarHT)
    local array::Union{Nothing, Symbol} = elementArray(simCode, k)
    (array !== nothing && array in dynamic) && push!(names, k)
  end
  return names
end

"""
    dataStructureArrayElements!(names, simCode)

The names an external object's constructor reads (`CombiTimeTable(..., table, ...)`), and the
elements of the arrays it reads whole (its references not an element's, `crefReads!`): the code
generators rebuild the array from its elements.
"""
function dataStructureArrayElements!(names::OrderedSet{String}, simCode::SIM_CODE; dynamic::Union{Nothing, Set{Symbol}} = nothing)::OrderedSet{String}
  local ht::HT = simCode.stringToSimVarHT
  local wholes::Set{Symbol} = Set{Symbol}()
  for (_, (_, v)) in ht
    (v.varKind isa DATA_STRUCTURE && v.varKind.bindExp isa CALL) || continue
    crefNames!(names, v.varKind.bindExp; dynamic)
    for read::Read in crefReads!(Read[], v.varKind.bindExp)
      local c::SimCref = read.cref
      (c.ref !== nothing && isempty(c.subs) && !isElementCref(c.ref)) && push!(wholes, c.ref.name)
    end
  end
  isempty(wholes) && return names
  for k::String in keys(ht)
    local array::Union{Nothing, Symbol} = elementArray(simCode, k)
    (array !== nothing && array in wholes) && push!(names, k)
  end
  return names
end

"""
    requestedNames(simCode)

The variables the observed filter asks for (a validation's reference signals): kept by the
parameter eliminations, so that they can be read from the solution.
"""
function requestedNames(simCode::SIM_CODE)::OrderedSet{String}
  local names = OrderedSet{String}()
  local patterns = simCode.observedFilter
  (patterns === nothing || isempty(patterns)) && return names
  local regexes = Regex[Regex(p) for p in patterns]
  #= the patterns are the user's, on the variables' identifiers (`codeName`: `a_b[1][2]`) =#
  for n in keys(simCode.stringToSimVarHT)
    local id::String = codeName(simCode, n)
    any(r -> occursin(r, id), regexes) && push!(names, n)
  end
  return names
end

#= ==================================================================================== =#
#= Parameters of literal values                                                         =#
#= ==================================================================================== =#

"""
    elementName(e)

The name of a reference, or of an element of one by literal subscripts (an integral Real
too); `nothing` for another expression.
"""
function elementName(e::Exp)::Union{Nothing, String}
  e isa EXP_CREF && return crefName(e.cref)
  local index::Vector{Int} = Int[]
  while e isa ASUB
    local outer::Vector{Int} = Int[]
    for s in e.subs
      local i::Union{Nothing, Int} = s isa ICONST ? s.value : (s isa RCONST && isinteger(s.value)) ? Int(s.value) : nothing
      i === nothing && return nothing
      push!(outer, i)
    end
    index = Int[outer; index]
    e = e.exp
  end
  return (e isa EXP_CREF && !isempty(index)) ? elementKey(e.cref, index) : nothing
end

"The value of a parameter of a literal value: a number, or an enumeration literal."
const ParameterValue = Union{Float64, ENUM_LITERAL}

"The literal of the value `v` read as type `ty`: an Integer's, a Boolean's, else a Real; an enumeration literal itself."
literalOf(v::Float64, ::TYPE_INTEGER)::Exp = ICONST(Int(round(v)))
literalOf(v::Float64, ::TYPE_BOOL)::Exp = BCONST(v != 0.0)
literalOf(v::Float64, ::SType)::Exp = RCONST(v)
literalOf(v::ENUM_LITERAL, ::SType)::Exp = v

"The literal of an array element of the value `v`, of the array's element type `ty` (an Integer array's element an Integer)."
elementLiteral(v::ParameterValue, ty::SType)::Exp = literalOf(v, scalarType(ty))

"""
    enumerationValue(e, simCode, seen)

The enumeration literal `e` is, through parameters' bindings; `nothing` if it is none.
"""
function enumerationValue(e::Exp, simCode::SIM_CODE, seen::OrderedSet{String})::Union{Nothing, ENUM_LITERAL}
  e isa ENUM_LITERAL && return e
  e isa EXP_CREF || return nothing
  return withParameterBinding(b -> enumerationValue(b, simCode, seen), e, simCode, seen)
end

"""
    substituteValues(e, values)

`e` with the parameters of `values` replaced by their literals, top-down: a reference by the
literal of its type, an array element by literal subscripts by a Real.
"""
function substituteValues(e::Exp, values::OrderedDict{String, ParameterValue})::Exp
  isempty(values) && return e
  return first(traverseExpTopDown(e, valueVisitor, values))
end

function valueVisitor(e::Exp, values::OrderedDict{String, ParameterValue})::Tuple{Exp, Bool, OrderedDict{String, ParameterValue}}
  if e isa EXP_CREF
    local v = get(values, crefName(e.cref), nothing)
    v === nothing || return (literalOf(v, e.ty), false, values)
  elseif e isa ASUB
    local name = elementName(e)
    local v = name === nothing ? nothing : get(values, name, nothing)
    v === nothing || return (elementLiteral(v, typeOf(e)), false, values)
  end
  return (e, true, values)
end

"""
    arrayElementValues!(values, name, binding, simCode, kept)

The values of the elements (by their keys, `elementKey`: `A[i]`, nested arrays `A[i,j]`) of the
array parameter `name`'s binding that evaluate: to a number, or to an element evaluated already;
not a kept one.
"""
function arrayElementValues!(values::OrderedDict{String, ParameterValue}, name::String, binding::Exp, simCode::SIM_CODE,
                             kept::OrderedSet{String}, index::Vector{Int} = Int[])::Nothing
  binding isa ARRAY_EXP || return nothing
  local array::SimCref = SimCref(Symbol(name), Int[], last(simCode.stringToSimVarHT[name]).ref)
  for (i, x) in enumerate(binding.elements)
    local at::Vector{Int} = Int[index; i]
    local elementName_::String = elementKey(array, at)
    elementName_ in kept && continue
    if x isa ARRAY_EXP
      arrayElementValues!(values, name, x, simCode, kept, at)
      continue
    end
    local v = something(enumerationValue(x, simCode, OrderedSet{String}()), Some(tryEvalNumeric(x, simCode)))
    if v === nothing
      local ref = elementName(x)
      (ref !== nothing && haskey(values, ref)) && (v = values[ref])
    end
    v === nothing || (values[elementName_] = v)
  end
  return nothing
end

"""
    eliminateConstantParameters(simCode)

The parameters whose bindings evaluate to a number (`tryEvalNumeric`; the elements of an
array parameter's binding too) replaced by their literals everywhere and dropped: MTK sees
fewer parameters. Kept: an alias's representative; one an attribute reads (the code
generators read start values by name), one an if-equation's, a when-clause's or an
if-expression's condition reads (a switch the user may flip), one an initial equation reads;
one the observed filter asks for; the names an external object's constructor reads and the
elements of its arrays; a Complex variable's parts; one a function body reads; the elements of
an array read by a subscript that is no literal; a tunable one. A parameter still read after
the substitution is kept. Nothing for a model with structural modes or translated again at
its structural changes. An enumeration parameter is replaced by its literal (not its index:
`String(e)`).
"""
function eliminateConstantParameters(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  #= the arrays read by a subscript that reads a variable, wherever the kept names are read =#
  local dynamic::Set{Symbol} = Set{Symbol}()
  local kept = OrderedSet{String}(e.representativeName for e in simCode.aliasMap)
  attributeNames!(kept, ht; dynamic)
  for ifEq in simCode.ifEquations, b in ifEq.branches
    crefNames!(kept, b.condition; dynamic)
  end
  foreach(w -> whenConditionNames!(kept, w.whenEquation; dynamic), simCode.whenEquations)
  initialEquationNames!(kept, simCode.initialEquations; dynamic)
  union!(kept, requestedNames(simCode))
  foreach(eq -> ifConditionNames!(kept, eq.exp; dynamic), simCode.residualEquations)
  for (_, (_, v)) in ht
    v.varKind isa PARAMETER && v.varKind.bindExp !== nothing && ifConditionNames!(kept, v.varKind.bindExp; dynamic)
  end
  dataStructureArrayElements!(kept, simCode; dynamic)
  complexFieldNames!(kept, simCode.residualEquations, simCode)
  complexFieldNames!(kept, simCode.initialEquations, simCode)
  functionBodyNames!(kept, simCode.functions; dynamic)
  #= the equations' dynamic reads: their arrays' elements stay variables (their other reads may be folded) =#
  foreach(eq -> dynamicArrays!(dynamic, eq.exp), simCode.residualEquations)
  for ifEq in simCode.ifEquations, b in ifEq.branches
    foreach(eq -> dynamicArrays!(dynamic, eq.exp), b.residualEquations)
  end
  dynamicallyIndexedElements!(kept, simCode, dynamic)
  if !isempty(TUNABLE_PARAMETERS[])
    union!(kept, TUNABLE_PARAMETERS[])
    foreach(k -> isTunableParameter(codeName(simCode, k)) && push!(kept, k), keys(ht))
  end
  local values = OrderedDict{String, ParameterValue}()
  for (name, (_, v)) in ht
    (name in kept || !isScalarParameter(v) || v.varKind.bindExp === nothing) && continue
    local x = v.elementType isa TYPE_ENUM ? enumerationValue(v.varKind.bindExp, simCode, OrderedSet{String}()) :
      tryEvalNumeric(v.varKind.bindExp, simCode)
    x === nothing || (values[name] = x)
  end
  #= the array parameters' elements, to a fixed point (an element may read another's) =#
  local before = -1
  while length(values) != before
    before = length(values)
    for (name, (_, v)) in ht
      (name in kept || !isArrayParameter(v) || v.varKind.bindExp === nothing) && continue
      arrayElementValues!(values, name, v.varKind.bindExp, simCode, kept)
    end
  end
  isempty(values) && return simCode
  local x = e::Exp -> substituteValues(e, values)
  local residuals = RESIDUAL_EQUATION[RESIDUAL_EQUATION(x(eq.exp), eq.source, eq.attr) for eq in simCode.residualEquations]
  local initialEquations = Equation[initialWithValues(eq, values) for eq in simCode.initialEquations]
  local ifEquations = IF_EQUATION[IF_EQUATION(BRANCH[BRANCH(x(b.condition), RESIDUAL_EQUATION[RESIDUAL_EQUATION(x(eq.exp), eq.source, eq.attr)
                                                                                           for eq in b.residualEquations],
                                                           b.identifier, b.targets) for b in ifEq.branches])
                                  for ifEq in simCode.ifEquations]
  local whenEquations = WHEN_EQUATION[WHEN_EQUATION(whenWithValues(w.whenEquation, x), w.source, w.attr) for w in simCode.whenEquations]
  #= the bindings too: a kept candidate's (still read), a String's (evaluated where an eliminated parameter is no name) =#
  local newHT = copy(ht)
  for (name, (index, v)) in ht
    local b = bindingOf(v.varKind)
    (v.varKind isa Union{PARAMETER, STRING} && b !== nothing) || continue
    local nb = x(b)
    nb == b || (newHT[name] = (index, withKind(v, withBinding(v.varKind, nb))))
  end
  local read = OrderedSet{String}()
  foreach(eq -> crefNames!(read, eq.exp), residuals)
  initialEquationNames!(read, initialEquations)
  ifEquationNames!(read, ifEquations)
  foreach(w -> whenCrefNames!(read, w.whenEquation), whenEquations)
  local stillRead = String[n for n in keys(values) if n in read]
  isempty(stillRead) || @debug "[SIMCODE: $(simCode.name): eliminateConstantParameters] parameters still read after the substitution, kept" stillRead
  local eliminated = String[n for n in keys(values) if !(n in read)]
  isempty(eliminated) && return simCode
  foreach(n -> delete!(newHT, n), eliminated)
  #= a model variable folded into a parameter stays observable: `v = <its value>` =#
  local (observed, observations) = foldedObservations(simCode, eliminated, n -> literalOf(values[n], last(ht[n]).elementType))
  return update(simCode; residualEquations = residuals, initialEquations, ifEquations, whenEquations, stringToSimVarHT = newHT,
                eliminatedVariables = vcat(simCode.eliminatedVariables, observed),
                eliminatedEquations = vcat(RESIDUAL_EQUATION[RESIDUAL_EQUATION(x(eq.exp), eq.source, eq.attr) for eq in simCode.eliminatedEquations],
                                           observations),
                asserts = ASSERT_EQUATION[ASSERT_EQUATION(x(a.condition), x(a.message), a.level, a.source) for a in simCode.asserts],
                initialAlgorithms = INITIAL_ALGORITHM[INITIAL_ALGORITHM(mapStatementExps(x, a.statements), a.section) for a in simCode.initialAlgorithms])
end

"An initial residual or equation with the parameters of `values` replaced; another kept."
initialWithValues(eq::RESIDUAL_EQUATION, values::OrderedDict{String, ParameterValue})::Equation =
  RESIDUAL_EQUATION(substituteValues(eq.exp, values), eq.source, eq.attr)
initialWithValues(eq::EQUATION, values::OrderedDict{String, ParameterValue})::Equation =
  EQUATION(substituteValues(eq.lhs, values), substituteValues(eq.rhs, values), eq.source, eq.attr)
initialWithValues(eq::Equation, ::OrderedDict{String, ParameterValue})::Equation = eq

"A when-clause with `x` applied to its condition, to its statements' expressions, to its elsewhen arms'."
function whenWithValues(w::WHEN_STMTS, x::F)::WHEN_STMTS where {F}
  return WHEN_STMTS(x(w.condition), mapStatementExps(x, w.body), w.elsewhenPart === nothing ? nothing : whenWithValues(w.elsewhenPart, x))
end

#= ==================================================================================== =#
#= Observation-only variables, dead parameters                                          =#
#= ==================================================================================== =#

"""
    dropObservationOnlyVariables(simCode)

The protected algebraic (and array) variables no one observes dropped with their defining
residual: a variable at most one residual reads, and nothing else (no initial equation, if-
or when-equation, assert, alias, attribute, if-expression condition, eliminated equation,
function body; not one the observed filter asks for). To a fixed point: a sink whose only reader was a dropped sink's residual
goes next. Nothing for a model with modes or translated again at its structural changes.
"""
function dropObservationOnlyVariables(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  local droppable = v::SIMVAR -> v.attributes.isProtected && v.varKind isa ALG_VARIABLE
  any(((_, (_, v)),) -> droppable(v), ht) || return simCode
  local untouchable = requestedNames(simCode)
  initialEquationNames!(untouchable, simCode.initialEquations)
  ifEquationNames!(untouchable, simCode.ifEquations)
  foreach(w -> whenCrefNames!(untouchable, w.whenEquation), simCode.whenEquations)
  assertNames!(untouchable, simCode.asserts)
  for e in simCode.aliasMap
    push!(untouchable, e.representativeName, e.eliminatedName)
  end
  attributeNames!(untouchable, ht)
  foreach(eq -> ifConditionNames!(untouchable, eq.exp), simCode.residualEquations)
  foreach(eq -> crefNames!(untouchable, eq.exp), simCode.eliminatedEquations)
  functionBodyNames!(untouchable, simCode.functions)
  local residuals = simCode.residualEquations
  local reads = OrderedSet{String}[crefNames!(OrderedSet{String}(), eq.exp) for eq in residuals]
  local readers = Dict{String, Int}()
  local readBy = Dict{String, Vector{Int}}()
  for (i, names) in enumerate(reads), n in names
    readers[n] = get(readers, n, 0) + 1
    push!(get!(readBy, n, Int[]), i)
  end
  local dropped = OrderedSet{String}()
  local droppedEquations = Set{Int}()
  local progressed = true
  while progressed
    progressed = false
    for (name, (_, v)) in ht
      (name in dropped || name in untouchable || !droppable(v)) && continue
      local n = get(readers, name, 0)
      n <= 1 || continue
      local defining = n == 1 ? findfirst(i -> !(i in droppedEquations), get(readBy, name, Int[])) : nothing
      if defining !== nothing
        local i = readBy[name][defining]
        foreach(m -> readers[m] -= 1, reads[i])
        push!(droppedEquations, i)
      end
      push!(dropped, name)
      progressed = true
    end
  end
  isempty(dropped) && return simCode
  @debug "[SIMCODE: $(simCode.name): dropObservationOnlyVariables] dropped $(length(dropped)) protected sinks and $(length(droppedEquations)) residuals"
  local newHT = copy(ht)
  foreach(n -> delete!(newHT, n), dropped)
  return update(simCode; stringToSimVarHT = newHT,
                residualEquations = RESIDUAL_EQUATION[eq for (i, eq) in enumerate(residuals) if !(i in droppedEquations)])
end

"""
    eliminateDeadParameters(simCode)

The parameters nothing reads dropped: no residual, initial equation, if- or when-equation,
assert, eliminated equation, alias, attribute, binding, function body, initial algorithm;
not one the observed filter asks for, a Complex variable's part, an element of an external
object's array or of an array read by a subscript that is no literal, a tunable one. Nothing
for a model with modes or translated again at its structural changes, nor for one without
continuous equations, whose parameters are its result (MSL readRealParameterModel: three
parameters read from a file and a when-equation that prints; its reference compares them, the
solution had none).
"""
function eliminateDeadParameters(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  (isempty(simCode.residualEquations) && isempty(simCode.ifEquations)) && return simCode
  local ht = simCode.stringToSimVarHT
  #= the arrays read by a subscript that reads a variable, wherever they are read =#
  local dynamic::Set{Symbol} = Set{Symbol}()
  local read = OrderedSet{String}()
  foreach(eq -> crefNames!(read, eq.exp; dynamic), simCode.residualEquations)
  initialEquationNames!(read, simCode.initialEquations; dynamic)
  ifEquationNames!(read, simCode.ifEquations; dynamic)
  foreach(w -> whenCrefNames!(read, w.whenEquation; dynamic), simCode.whenEquations)
  assertNames!(read, simCode.asserts; dynamic)
  foreach(eq -> crefNames!(read, eq.exp; dynamic), simCode.eliminatedEquations)
  for e in simCode.aliasMap
    push!(read, e.representativeName, e.eliminatedName)
  end
  attributeNames!(read, ht; dynamic)
  union!(read, requestedNames(simCode))
  functionBodyNames!(read, simCode.functions; dynamic)
  foreach(a -> statementCrefNames!(read, a.statements; dynamic), simCode.initialAlgorithms)
  complexFieldNames!(read, simCode.residualEquations, simCode)
  complexFieldNames!(read, simCode.initialEquations, simCode)
  for (_, (_, v)) in ht
    local b = bindingOf(v.varKind)
    (v.varKind isa Union{PARAMETER, DATA_STRUCTURE} && b !== nothing) && crefNames!(read, b; dynamic)
  end
  dataStructureArrayElements!(read, simCode; dynamic)
  dynamicallyIndexedElements!(read, simCode, dynamic)
  local dead = String[name for (name, (_, v)) in ht if isScalarParameter(v) && !(name in read) && !isTunableParameter(codeName(simCode, name))]
  isempty(dead) && return simCode
  @debug "[SIMCODE: $(simCode.name): eliminateDeadParameters] dropped $(length(dead)) unread parameters"
  local newHT = copy(ht)
  foreach(n -> delete!(newHT, n), dead)
  #= a folded model variable observed by its binding (whose parameters the bindings' reads keep) =#
  local (observed, observations) = foldedObservations(simCode, dead, n -> something(last(ht[n]).varKind.bindExp, RCONST(0.0)))
  return update(simCode; stringToSimVarHT = newHT, eliminatedVariables = vcat(simCode.eliminatedVariables, observed),
                eliminatedEquations = vcat(simCode.eliminatedEquations, observations))
end

"""
    foldedObservations(simCode, dropped, valueOf)

Of the parameters `dropped`, the model variables `foldParameterClosure` folded
(`foldedVariables`) with their observed equations `v - valueOf(v)`: eliminated variables, read
from the solution (dropped with no equation, a variable of the model was in none).
"""
function foldedObservations(simCode::SIM_CODE, dropped::Vector{String}, valueOf::F)::Tuple{Vector{String}, Vector{RESIDUAL_EQUATION}} where {F}
  local folded = Set{String}(simCode.foldedVariables)
  local names = String[n for n in dropped if n in folded]
  local ht = simCode.stringToSimVarHT
  local equations::Vector{RESIDUAL_EQUATION} = RESIDUAL_EQUATION[RESIDUAL_EQUATION(BINARY(EXP_CREF(SimCref(Symbol(n), Int[], last(ht[n]).ref), last(ht[n]).elementType), OP_SUB, valueOf(n)),
                                                        NO_SOURCE, EQ_ATTR_DEFAULT) for n in names]
  return (names, equations)
end
