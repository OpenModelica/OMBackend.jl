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

#= The simulation code of a model with its arrays kept (2026-11-01): the array route's model, in the array code's
   form (Passes.arrayCodeForm!), lowered one to one: its equations in their order, for-equations and all. =#

"""
    lowerArraySystem(model) -> SIM_CODE

The simulation code of a model with its arrays kept, in the array code's form (`Passes.arrayCodeForm!`; one continuous
partition, no clocks, no structural modes):

  - its variables as `lowerSystem`'s, an array one variable of its shape (`SIMVAR.dims`); a state's derivative is its
    link's (`\$STATEDER.x`, `IR.STATE_DER`), `der(x)` in the expressions;
  - its equations and initial equations in their order (`SIM_CODE.arrayEquations`, `arrayInitialEquations`): a scalar
    equation as written (`EQUATION`), a for-equation (`FOR_EQUATION`), an if-equation (`INLINE_IF_EQUATION`), an
    assert, a when-equation (`WHEN_EQUATION`; the initial kinds `INITIAL_WHEN_EQUATION`) whose branches' triggers are
    the vector of them (`when {c1, c2}`: each fires at its own rising edge), `initial()` the last of a branch active at
    the initialization;
  - a reference to an element of an array variable as the variable subscripted (`ASUB`, `elementReference`);
  - the bindings of the parameters the equations read (through other bindings too: `Passes.parametersRead`), the
    others' left out (the array code takes their values, evaluated); the start and fixed attributes;
  - the values the array code takes per element evaluated on the variables' element view (`IndexReduction.elements`,
    `scalarVariables`, OMJLIR's `Evaluation.StartValues`): each variable's start and fixed attributes (as constants
    where an element has one and every element's evaluates; a missing one the default, `0.0` and `false`, a
    parameter's fixed `true`), a parameter's value (`PARAMETER.value`), the array variables' element names
    (`ReferenceIndex.elementNames`).

An array equation as `ARRAY_EQUATION`, which the array code refuses where it reaches one; an equation of an
if-equation's branch that does not lower as `UNLOWERED_EQUATION` (the branch may never be taken). UnsupportedLowering
for another equation, a structural when-equation, or the model's functions where they do not lower.
"""
function lowerArraySystem(model::IR.Model)::SIM_CODE
  (length(model.continuous) == 1 && isempty(model.clocked) && isempty(model.modes)) ||
    unsupported("an OMJLIR model of several partitions, clocks or structural modes", nothing)
  local E::Passes.IndexReduction.Elements = Passes.IndexReduction.elements(model.variables)
  local evaluated::Dict{Symbol, ElementValues} = elementValues(model.variables, E, model.functions)
  local variables::IR.VariableSet = elementReferenceVariables(model.variables)
  local read::Set{Symbol} = Passes.parametersRead(model)
  for v in variables
    (v.kind isa Union{IR.PARAMETER, IR.CONSTANT} && !(v.name in read)) && (v.binding = nothing)
  end
  local partition::IR.Partition = model.continuous[1]
  local vars::Vector{IR.Variable} = IR.Variable[v for v in variables if !(v.kind isa IR.STATE_DER)]
  local names::Names = arraySystemNames(variables)
  local simVars::Vector{SIMVAR} = SIMVAR[evaluatedVariable(SIMVAR(simName(names, v.cref), nothing, simVarKind(v, names),
                                                                  arrayVarAttributes(v, names), simType(IR.elementType(v.ty)),
                                                                  arrayDimensions(v), simKind(v.kind, variables, names),
                                                                  v.origin, v.cref), evaluated[v.name], names) for v in vars]
  local elementNames::Dict{String, Vector{String}} =
    Dict{String, Vector{String}}(simName(names, v.cref) => String.(evaluated[v.name].names) for v in vars if v.ty isa IR.T_ARRAY)
  local indexed::HT = createIndices(simVars, derivativeReferences(vars, variables, names))
  #= the table in the model's order (the array code lays the variables out so), a state's derivative after it =#
  local derivativeKey::Dict{String, String} =
    Dict{String, String}(v.varKind.varName => k for (k, (_, v)) in indexed if v.varKind isa STATE_DERIVATIVE)
  local ht::HT = HT()
  for v in vars
    local key::String = simName(names, v.cref)
    ht[key] = indexed[key]
    haskey(derivativeKey, key) && haskey(indexed, derivativeKey[key]) && (ht[derivativeKey[key]] = indexed[derivativeKey[key]])
  end
  local lowered::Function = (eq::IR.Equation) ->
    arrayEquation(names, IR.mapEquation!((x::IR.Expression) -> elementReference(x, variables), deepcopy(eq)))
  local equations::Vector{Equation} = Equation[lowered(eq) for eq in partition.equations]
  local initialEquations::Vector{Equation} = Equation[lowered(eq) for eq in partition.initialEquations]
  local functions::Vector{ModelicaFunction} = try
    simFunctions(model.functions)
  catch err
    err isa UnsupportedLowering || rethrow()
    unsupported("the model's functions", first(sprint(showerror, err), 160))
  end
  return SIM_CODE(String(codeName(Symbol(model.name))), ht, RESIDUAL_EQUATION[], Equation[], WHEN_EQUATION[], IF_EQUATION[],
                  StructuralTransition[], SIM_CODE[], String[], String[], Equation[],
                  String(codeName(Symbol(something(model.initialMode, model.name)))), model.frontendModel, String[],
                  functions, any((f::ModelicaFunction) -> f isa EXTERNAL_MODELICA_FUNCTION, functions),
                  RESIDUAL_EQUATION[], String[], AliasEntry[], nothing, INITIAL_ALGORITHM[], ASSERT_EQUATION[], String[],
                  any((v::IR.Variable) -> v.kind isa Union{IR.DUMMY_STATE, IR.DUMMY_DER}, vars),
                  ReferenceIndex(ht, String[simName(names, v.cref) for v in vars], model.name; elementNames), equations,
                  initialEquations)
end

"""
    ElementValues(names, startGiven, starts, fixed, fixedEvaluates, values)

What the lowering evaluates of a variable per element (`elementValues`), row-major: its elements' names; whether an
element has a start; their starts (`0.0` for none; `nothing` where one does not evaluate); their fixed attributes
(`nothing` for none) and whether every one given evaluates; a parameter's values (typed as its elements; `nothing` where
one does not evaluate, or of another type).
"""
struct ElementValues
  names::Vector{Symbol}
  startGiven::Bool
  starts::Union{Nothing, Vector{Float64}}
  fixed::Vector{Union{Nothing, Bool}}
  fixedEvaluates::Bool
  values::Union{Nothing, Vector{Any}}
  startExps::Union{Nothing, Vector{IR.Expression}}   #= the element starts as written, where one reads a parameter =#
end

"""
    elementValues(variables, E, functions) -> Dict{Symbol, ElementValues}

Each variable's values per element (`ElementValues`), on the element view of `variables` (`E`, their elements;
`IndexReduction.scalarVariables`, the elements as variables): the elements of an array by the view's names (an array
parameter's whose elements are variables of their own: `modelElementCref`), their start and fixed attributes and a
parameter's binding evaluated (`Evaluation.StartValues`).
"""
function elementValues(variables::IR.VariableSet, E::Passes.IndexReduction.Elements,
                       functions::Vector{IR.ModelicaFunction})::Dict{Symbol, ElementValues}
  local view::IR.VariableSet = Passes.IndexReduction.scalarVariables(variables, E)
  local values::Passes.Evaluation.StartValues = Passes.Evaluation.StartValues(view, functions)
  local out::Dict{Symbol, ElementValues} = Dict{Symbol, ElementValues}()
  for v in variables
    v.kind isa IR.STATE_DER && continue
    local dims::Vector{Int} = v.ty isa IR.T_ARRAY ?
      Int[something(IR.dimensionSize(d), -1) for d in (v.ty::IR.T_ARRAY).dimensions] : Int[]
    any((d::Int) -> d < 0, dims) && continue
    local known::Bool = v.kind isa Union{IR.PARAMETER, IR.CONSTANT}
    local names::Vector{Symbol} = haskey(E.elements, v.name) ? E.elements[v.name] :
      (!isempty(dims) && known) ? Symbol[Passes.IndexReduction.modelElementCref(v, index).name for index in Passes.rowMajor(dims)] :
      Symbol[v.name]
    local attribute::Function = (x::Symbol, f::Symbol) -> begin
      local ev::Union{Nothing, IR.Variable} = get(view, x, nothing)
      ev === nothing ? nothing : getfield(ev.attributes, f)
    end
    local startOf::Function = (s::Union{Nothing, IR.Expression}) -> s === nothing ? 0.0 : Passes.Evaluation.evaluate(s, values)
    local startGiven::Bool = any((x::Symbol) -> attribute(x, :start) !== nothing, names)
    local starts::Vector{Union{Nothing, Float64}} = Union{Nothing, Float64}[startOf(attribute(x, :start)) for x in names]
    #= the starts that read a parameter as written too: an override of it reaches them (the array code); the references
       to elements of array variables as the variables subscripted (`c[2].s` of a rolled component's `c.s`) =#
    local startExprs::Vector{IR.Expression} =
      IR.Expression[IR.mapExp((y::IR.Expression) -> elementReference(y, variables), something(attribute(x, :start), IR.REAL(0.0)))
                    for x in names]
    local readsParameter::Bool = any((s::IR.Expression) -> IR.anyExp((y::IR.Expression) -> y isa IR.CREF, s), startExprs)
    local fixedGiven::Vector{Union{Nothing, IR.Expression}} = Union{Nothing, IR.Expression}[attribute(x, :fixed) for x in names]
    local fixed::Vector{Union{Nothing, Bool}} = Union{Nothing, Bool}[f === nothing ? nothing : Passes.Evaluation.evaluateCondition(f, values)
                                                                     for f in fixedGiven]
    local fixedEvaluates::Bool = all(k -> fixedGiven[k] === nothing || fixed[k] !== nothing, eachindex(fixed))
    local vtype::Symbol = valueType(simType(v.ty))
    local paramValues::Union{Nothing, Vector{Any}} = nothing
    if known && vtype != :other
      local vals::Vector{Union{Nothing, Float64}} = Union{Nothing, Float64}[values(x) for x in names]
      all((x::Union{Nothing, Float64}) -> x !== nothing, vals) &&
        (paramValues = Any[vtype == :int ? round(Int, x) : vtype == :bool ? x != 0 : x for x in vals])
    end
    out[v.name] = ElementValues(names, startGiven,
                                all((x::Union{Nothing, Float64}) -> x !== nothing, starts) ? Float64[x for x in starts] : nothing,
                                fixed, fixedEvaluates, paramValues, (readsParameter && !known) ? startExprs : nothing)
  end
  return out
end

"""
    evaluatedVariable(sv, e, names) -> SIMVAR

`sv` with what the lowering evaluated (`ElementValues`): its start and fixed attributes as constants where an element
has one and every element's evaluates (an array's the array literal of them; a missing start `0.0`, a missing fixed
the default, `true` of a parameter, else `false`), a parameter's value (`PARAMETER.value`); a numeric constant's too, a
`PARAMETER` of its value then (the array code takes a constant as a parameter). An attribute given on the elements that
does not evaluate, the variable's own being none (a rolled component's start): a String constant saying so, which the
array code refuses as no value. The element starts that read a parameter as written too (`startExps`).
"""
function evaluatedVariable(sv::SIMVAR, e::ElementValues, names::Names)::SIMVAR
  local a::SimVarAttributes = sv.attributes
  local notEvaluated::Function = (what::String, x::Union{Nothing, Exp}) -> x === nothing ? SCONST("the $(what) does not evaluate") : x
  local start::Union{Nothing, Exp} = !e.startGiven ? a.start : e.starts === nothing ? notEvaluated("start", a.start) :
    constantExp(Any[x for x in e.starts], sv.dims)
  local default::Bool = sv.varKind isa PARAMETER
  local fixed::Union{Nothing, Exp} = all((x::Union{Nothing, Bool}) -> x === nothing, e.fixed) && e.fixedEvaluates ? a.fixed :
    !e.fixedEvaluates ? notEvaluated("fixed attribute", a.fixed) : constantExp(Any[something(x, default) for x in e.fixed], sv.dims)
  local kind::SimVarType = sv.varKind
  #= a constant's value too: the array code takes it as a parameter's =#
  kind isa Union{PARAMETER, DATA_STRUCTURE} && e.values !== nothing && (kind = PARAMETER(kind.bindExp, constantExp(e.values, sv.dims)))
  local startExps::Union{Nothing, Vector{Exp}} = e.startExps === nothing ? nothing : Exp[simExp(names, x) for x in e.startExps]
  return withKind(withAttributes(sv, SimVarAttributes(start, fixed, a.nominal, a.min, a.max, a.unit, a.displayUnit, a.quantity,
                                                      a.stateSelect, a.isProtected, startExps)), kind)
end

"The constant of the values `vals` (row-major) of a variable of dimensions `dims`: a scalar's, an array's the array literal."
function constantExp(vals::Vector{Any}, dims::Vector{Int})::Exp
  isempty(dims) && return constantOf(vals[1])
  return constantRows(vals, dims, 0, 1, isempty(vals) ? TYPE_REAL() : typeOfConstant(vals[1]))
end

"The rows of dimension `k` of the constant `constantExp` makes, from the value after `offset` on, of elements `el`."
function constantRows(vals::Vector{Any}, dims::Vector{Int}, offset::Int, k::Int, el::SType)::Exp
  local stride::Int = prod(dims[k + 1:end]; init = 1)
  local elements::Vector{Exp} = Exp[k == length(dims) ? constantOf(vals[offset + i]) : constantRows(vals, dims, offset + (i - 1) * stride, k + 1, el)
                                    for i in 1:dims[k]]
  return ARRAY_EXP(TYPE_ARRAY(el, dims[k:end]), k == length(dims), elements)
end

"A value as a constant of the simulation code."
constantOf(x::Bool)::Exp = BCONST(x)
constantOf(x::Integer)::Exp = ICONST(Int(x))
constantOf(x::Real)::Exp = RCONST(Float64(x))
"The type of a value's constant (`constantOf`)."
typeOfConstant(x::Bool)::SType = TYPE_BOOL()
typeOfConstant(x::Integer)::SType = TYPE_INTEGER()
typeOfConstant(x::Real)::SType = TYPE_REAL()

"""
    arraySystemNames(variables) -> Names

The keys of a model's variables with its arrays kept (`systemNames`): a state's derivative is its link's
(`\$STATEDER.x`: `der(x)` in the expressions; the link no variable of the simulation code).
"""
function arraySystemNames(variables::IR.VariableSet)::Names
  local stateOf::Dict{Symbol, IR.ComponentRef} = Dict{Symbol, IR.ComponentRef}()
  for v in variables
    v.kind isa IR.STATE_DER && (stateOf[v.name] = variables[v.kind.state].cref)
  end
  local names::Names = Names(true, Set{String}(), stateOf, Set{String}())
  for v in variables
    local n::String = simName(names, v.cref)
    push!(names.known, n)
    v.ty isa IR.T_ARRAY || push!(names.scalars, n)
  end
  push!(names.known, "time", "pi", "e")
  return names
end

"""
    elementReferenceVariables(variables) -> VariableSet

A copy of `variables` with the references to elements of array variables in their bindings and attributes as the
variables subscripted (`elementReference`).
"""
function elementReferenceVariables(variables::IR.VariableSet)::IR.VariableSet
  local out::IR.VariableSet = deepcopy(variables)
  local refs::Function = (e::IR.Expression) -> IR.mapExp((x::IR.Expression) -> elementReference(x, out), e)
  local optional::Function = (e::Union{Nothing, IR.Expression}) -> e === nothing ? nothing : refs(e)
  for v in out
    v.binding = optional(v.binding)
    local a::IR.VariableAttributes = v.attributes
    v.attributes = IR.VariableAttributes(optional(a.start), optional(a.fixed), optional(a.nominal), optional(a.min),
                                         optional(a.max), a.unit, a.displayUnit, a.quantity, a.stateSelect)
  end
  return out
end

"""
    arrayVarAttributes(v, names) -> SimVarAttributes

The attributes of `v` the array code reads: its start and fixed attributes (where the lowering does not evaluate them
per element, `evaluatedVariable`: of a variable whose elements have none); stateSelect and whether it is protected.
"""
function arrayVarAttributes(v::IR.Variable, names::Names)::SimVarAttributes
  local a::IR.VariableAttributes = v.attributes
  local e::Function = (x::Union{Nothing, IR.Expression}) -> x === nothing ? nothing : simExp(names, x)
  return SimVarAttributes(e(a.start), e(a.fixed), nothing, nothing, nothing, nothing, nothing, nothing,
                          a.stateSelect === nothing ? nothing : STATE_SELECT_NAMES[a.stateSelect], v.isProtected)
end

"""
    elementReference(x, variables) -> Expression

A reference to an element of an array variable of `variables` (`x[\$i1]`, `c[2].T` of a rolled component's member
`c.T`, `\$STATEDER.x[i]`) as the variable subscripted by all its subscripts (`Passes.allSubscripts`); another
expression (a variable's own reference, an element that is a variable of its own: an array split for index reduction)
as it is.
"""
function elementReference(x::IR.Expression, variables::IR.VariableSet)::IR.Expression
  (x isa IR.CREF && IR.hasSubscripts(x.cref) && !haskey(variables, x.cref.name)) || return x
  local v::Union{Nothing, IR.Variable} = get(variables, unsubscriptedName(x.cref), nothing)
  v === nothing && return x
  return IR.SUBSCRIPTED(IR.CREF(v.cref, v.ty), allSubscripts(x.cref), x.ty)
end

"An equation of a model with its arrays kept in the simulation code (`lowerArraySystem`)."
arrayEquation(names::Names, eq::IR.SCALAR_EQUATION)::Equation =
  EQUATION(simExp(names, eq.lhs), simExp(names, eq.rhs), simSource(eq.source), EQ_ATTR(eq.attributes.differentiated, false))

arrayEquation(names::Names, eq::IR.FOR_EQUATION)::Equation =
  FOR_EQUATION(Iterator(eq.iterator.name, simExp(names, eq.iterator.range)), Equation[arrayEquation(names, b) for b in eq.body],
               simSource(eq.source), EQ_ATTR(eq.attributes.differentiated, false))

arrayEquation(names::Names, eq::IR.IF_EQUATION)::Equation =
  INLINE_IF_EQUATION(Exp[simExp(names, b.condition) for b in eq.branches],
                     Vector{Equation}[Equation[branchEquation(names, e) for e in b.equations] for b in eq.branches],
                     Equation[branchEquation(names, e) for e in eq.elseEquations], simSource(eq.source),
                     EQ_ATTR(eq.attributes.differentiated, false))

"""
    branchEquation(names, eq) -> Equation

An equation of an if-equation's branch (`arrayEquation`); one that does not lower (UnsupportedLowering) as
`UNLOWERED_EQUATION`: a branch of a parameter's condition may never be taken, and the array code refuses it only where
it takes the branch.
"""
function branchEquation(names::Names, eq::IR.Equation)::Equation
  try
    return arrayEquation(names, eq)
  catch err
    err isa UnsupportedLowering || rethrow()
    return UNLOWERED_EQUATION(err.what, first(string(err.construct), 200), simSource(eq.source))
  end
end

arrayEquation(names::Names, eq::IR.ASSERT_EQUATION)::Equation =
  ASSERT_EQUATION(simExp(names, eq.condition), simExp(names, eq.message), simExp(names, eq.level), simSource(eq.source))

function arrayEquation(names::Names, eq::IR.WHEN_EQUATION)::Equation
  local stmts::Union{Nothing, WHEN_STMTS} = nothing
  local regular::Bool = eq.kind == IR.REGULAR_WHEN
  for b in Iterators.reverse(eq.branches)
    stmts = WHEN_STMTS(arrayWhenCondition(names, b; atEvents = regular), arrayStatements(names, b.body), stmts)
  end
  eq.kind == IR.REGULAR_WHEN &&
    return WHEN_EQUATION(stmts::WHEN_STMTS, simSource(eq.source), EQ_ATTR(false, false, eq.attributes.kind === IR.EQ_DISCRETE))
  eq.kind in (IR.INITIAL_WHEN, IR.INITIAL_SECTION) &&
    return INITIAL_WHEN_EQUATION(stmts::WHEN_STMTS, simSource(eq.source), EQ_ATTR_DEFAULT)
  unsupported("a when-equation of kind $(eq.kind)", IR.toString(eq))
end

"""
    arrayStatements(names, body) -> Vector{Statement}

The statements of a when-branch of a model with its arrays kept (`simStatements`'), a reinit of an element of an array
state the reinit of the state variable at the element's subscripts (`REINIT.subscripts`; an iterator's, `x[i]`).
"""
arrayStatements(names::Names, body::Vector{IR.Statement})::Vector{Statement} = Statement[arrayStatement(names, s) for s in body]

"A statement of a when-branch (`arrayStatements`): an if-statement's branches by `arrayStatements` too."
arrayStatement(names::Names, s::IR.Statement)::Statement = simStatement(names, s)

arrayStatement(names::Names, s::IR.IF_STATEMENT)::Statement =
  IF(Branch[Branch(simExp(names, b.condition), arrayStatements(names, b.body)) for b in s.branches],
     arrayStatements(names, s.elseBody), simSource(s.source))

function arrayStatement(names::Names, s::IR.REINIT)::Statement
  local c::IR.ComponentRef = s.state
  (IR.hasSubscripts(c) && !(simName(names, c) in names.known)) || return simStatement(names, s)
  local base::IR.ComponentRef = IR.withoutSubscripts(c)
  simName(names, base) in names.known || unsupported("a reinit of an element of no variable", IR.toString(c))
  local subscripts::Vector{Exp} = Exp[]
  for x in allSubscripts(c)
    x isa IR.INDEX || unsupported("a reinit of a slice", IR.toString(c))
    push!(subscripts, simExp(names, (x::IR.INDEX).exp))
  end
  return REINIT(simCref(names, base), simExp(names, s.value), simSource(s.source), subscripts)
end

"An array equation as it is (`left = right` of arrays): the array code refuses it where it reaches one (not in a branch never taken)."
arrayEquation(names::Names, eq::IR.ARRAY_EQUATION)::Equation =
  ARRAY_EQUATION(Int[something(IR.dimensionSize(d), -1) for d in eq.ty.dimensions], simExp(names, eq.lhs), simExp(names, eq.rhs),
                 simSource(eq.source), EQ_ATTR(eq.attributes.differentiated, false))

arrayEquation(::Names, eq::IR.Equation)::Equation = unsupported("$(nameof(typeof(eq))) in an OMJLIR model", IR.toString(eq))

"""
    arrayWhenCondition(names, b; atEvents) -> Exp

A when-branch's condition in a model with its arrays kept: its triggers, and of a regular when-equation (`atEvents`)
`initial()` after them for a branch active at the initialization, `terminal()` last for one at the terminal event;
one alone itself, several their vector (`when {c1, c2}`); none the empty vector (an initial when-equation's).
"""
function arrayWhenCondition(names::Names, b::IR.WhenBranch; atEvents::Bool)::Exp
  local triggers::Vector{IR.Expression} = copy(b.triggers)
  atEvents && b.atInitial && push!(triggers, initialCall())
  atEvents && b.atTerminal && push!(triggers, builtinCall(:terminal, IR.Expression[], IR.T_BOOLEAN()))
  isempty(triggers) && return ARRAY_EXP(TYPE_ARRAY(TYPE_BOOL(), Int[0]), true, Exp[])
  length(triggers) == 1 && return simExp(names, only(triggers))
  return ARRAY_EXP(TYPE_ARRAY(TYPE_BOOL(), Int[length(triggers)]), true, Exp[simExp(names, t) for t in triggers])
end
