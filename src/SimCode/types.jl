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

#= The SimCode types: a closed set, lower-level than OMJLIR's. Everything above them (the
   structural kinds, der/pre as calls, records, clocks, the frontend's names) is resolved
   by the lowering from OMJLIR (`fromOMJLIR`, lowerModel.jl), so the code generators read only these. =#

#= ==================================================================================== =#
#= Values                                                                               =#
#= ==================================================================================== =#

"""
Root of SimCode's immutable values that compare structurally (`==`, `hash`): types,
expressions, statements, variables and their parts.
"""
abstract type Node end

"Fields compared one by one: the values are trees of vectors, strings and symbols."
function Base.:(==)(a::T, b::T)::Bool where {T <: Node}
  for f in fieldnames(T)
    getfield(a, f) == getfield(b, f) || return false
  end
  return true
end
Base.:(==)(::Node, ::Node)::Bool = false

"""
Fields compared one by one with `isequal`, as `hash` hashes them: the identity of a `Dict` or
`Set` key (`RCONST(0.0)` and `RCONST(-0.0)` apart, `RCONST(NaN)` equal to itself; `==` has
them the other way round, as numbers).
"""
function Base.isequal(a::T, b::T)::Bool where {T <: Node}
  for f in fieldnames(T)
    isequal(getfield(a, f), getfield(b, f)) || return false
  end
  return true
end
Base.isequal(::Node, ::Node)::Bool = false

function Base.hash(a::T, h::UInt)::UInt where {T <: Node}
  local r = hash(T, h)
  for f in fieldnames(T)
    r = hash(getfield(a, f), r)
  end
  return r
end

#= ==================================================================================== =#
#= Sources                                                                              =#
#= ==================================================================================== =#

"""
    Source(file, line)

Where an equation or a statement comes from in the Modelica source, for the messages of the
generated code. `NO_SOURCE` for one the backend creates.
"""
struct Source <: Node
  file::String
  line::Int
end

const NO_SOURCE = Source("", 0)

#= ==================================================================================== =#
#= Types                                                                                =#
#= ==================================================================================== =#

"""
    SType

The type of an expression: its category (`isa TYPE_REAL`, `isa TYPE_ARRAY`, ...), an array's
element type and integer dimensions, a record's fields.
"""
abstract type SType <: Node end

"Scalar `Real`."
struct TYPE_REAL <: SType end
"Scalar `Integer`."
struct TYPE_INTEGER <: SType end
"Scalar `Boolean`."
struct TYPE_BOOL <: SType end
"Scalar `String`."
struct TYPE_STRING <: SType end
"A type the code generators need not know (the inputs of generic built-in functions)."
struct TYPE_UNKNOWN <: SType end
"The type of a call of a function without outputs (a call for its effects)."
struct TYPE_NORETCALL <: SType end

"""
    TYPE_COMPLEX(isRecord, fieldNames, fieldTypes)

A record (`isRecord`) or an external object: its fields' names and types.
"""
struct TYPE_COMPLEX <: SType
  isRecord::Bool
  fieldNames::Vector{String}
  fieldTypes::Vector{SType}
end

"An enumeration type: its literals' names in order (a tuple: equal types are `===`)."
struct TYPE_ENUM <: SType
  names::Tuple{Vararg{String}}
end

"An array of `elementType` with dimensions `dims` (`-1` for one of unknown size)."
struct TYPE_ARRAY <: SType
  elementType::SType
  dims::Vector{Int}
end

"The type of a call to a function with several outputs."
struct TYPE_TUPLE <: SType
  types::Vector{SType}
end

"The type of a function value: a function's inputs' and outputs' types."
struct TYPE_FUNCTION <: SType
  inputs::Vector{SType}
  outputs::Vector{SType}
end

#= ==================================================================================== =#
#= Component references                                                                 =#
#= ==================================================================================== =#

"""
    SimCref(sym, subs = Int[])

A reference to a simulation variable: its flat name (`body_frame_a_r_0`, `x[2]`) and, for an
element of an array variable kept as an array, the integer subscripts. A single `===` on the
symbol compares names.
"""
struct SimCref
  sym::Symbol
  subs::Vector{Int}
end

SimCref(sym::Symbol) = SimCref(sym, Int[])
SimCref(name::AbstractString) = SimCref(Symbol(name), Int[])
SimCref(name::AbstractString, subs::Vector{Int}) = SimCref(Symbol(name), subs)

Base.:(==)(a::SimCref, b::SimCref)::Bool = a.sym === b.sym && a.subs == b.subs
Base.isequal(a::SimCref, b::SimCref)::Bool = a == b
Base.hash(c::SimCref, h::UInt)::UInt = hash(c.subs, hash(c.sym, h))

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

"""
    OpKind

The operators of SimCode expressions: arithmetic on scalars; element-wise on arrays of equal
dimensions (`_EW`); between a scalar and an array; the vector and matrix products; logical;
relational. Each code generator emits one operation per kind: `a .* b` (`OP_MUL_EW`) and
`A * B` (`OP_MATRIX_PRODUCT`) are apart.
"""
@enum OpKind begin
  OP_ADD; OP_SUB; OP_MUL; OP_DIV; OP_POW; OP_UMINUS
  OP_ADD_EW; OP_SUB_EW; OP_MUL_EW; OP_DIV_EW; OP_POW_EW
  OP_ADD_SCALAR_ARRAY; OP_ADD_ARRAY_SCALAR; OP_SUB_SCALAR_ARRAY; OP_SUB_ARRAY_SCALAR
  OP_MUL_SCALAR_ARRAY; OP_MUL_ARRAY_SCALAR; OP_MUL_VECTOR_MATRIX; OP_MUL_MATRIX_VECTOR
  OP_SCALAR_PRODUCT; OP_MATRIX_PRODUCT
  OP_DIV_SCALAR_ARRAY; OP_DIV_ARRAY_SCALAR; OP_POW_SCALAR_ARRAY; OP_POW_ARRAY_SCALAR; OP_POW_MATRIX
  OP_AND; OP_OR; OP_NOT
  OP_LESS; OP_LESSEQ; OP_GREATER; OP_GREATEREQ; OP_EQUAL; OP_NEQUAL
end

"""
    CallKind

What a call calls: a Modelica built-in function or operator (by its plain name: `sin`, `sum`,
`edge`, `sample`, ...), a function of the model (SIM_CODE.functions), an impure one (MLS 12.3,
not to be moved or evaluated at compile time), or a function value (a functional input).
"""
@enum CallKind CALL_BUILTIN CALL_FUNCTION CALL_IMPURE_FUNCTION CALL_FUNCTION_VALUE

"A SimCode expression."
abstract type Exp <: Node end

"An `Integer` literal."
struct ICONST <: Exp
  value::Int
end

"A `Real` literal."
struct RCONST <: Exp
  value::Float64
end

"A `Boolean` literal."
struct BCONST <: Exp
  value::Bool
end

"A `String` literal."
struct SCONST <: Exp
  value::String
end

"An enumeration literal: its name (`Type.literal`) and index (1-based)."
struct ENUM_LITERAL <: Exp
  name::Symbol
  index::Int
end

"A reference to a variable, of type `ty`."
struct EXP_CREF <: Exp
  cref::SimCref
  ty::SType
end

"The derivative of the state `cref`: `der(x)`."
struct DER <: Exp
  cref::SimCref
end

"The value of `cref` before the event: `pre(x)`."
struct PRE <: Exp
  cref::SimCref
  ty::SType
end

"The discarded output in a tuple assignment: `(a, _) = f(x)`."
struct WILD <: Exp end

"The subscript `:`: every element of its dimension (`A[1, :]`)."
struct WHOLE_DIMENSION <: Exp end

"`exp1 op exp2` for an arithmetic `op`."
struct BINARY <: Exp
  exp1::Exp
  op::OpKind
  exp2::Exp
end

"`op exp` for `OP_UMINUS`."
struct UNARY <: Exp
  op::OpKind
  exp::Exp
end

"`exp1 op exp2` for `OP_AND`, `OP_OR`."
struct LBINARY <: Exp
  exp1::Exp
  op::OpKind
  exp2::Exp
end

"`not exp`."
struct LUNARY <: Exp
  op::OpKind
  exp::Exp
end

"""
    RELATION(exp1, op, exp2, index)

`exp1 op exp2` for a relational `op`; `index` its zero-crossing slot for the event-aware code
generation, 0 for one that generates no event.
"""
struct RELATION <: Exp
  exp1::Exp
  op::OpKind
  exp2::Exp
  index::Int
end

"`if cond then thenExp else elseExp`."
struct IFEXP <: Exp
  cond::Exp
  thenExp::Exp
  elseExp::Exp
end

"An array of `elements`, of type `ty`; `scalar`: its elements are scalars."
struct ARRAY_EXP <: Exp
  ty::SType
  scalar::Bool
  elements::Vector{Exp}
end

"A range `start:step:stop`; `step` is `nothing` for `start:stop`."
struct RANGE <: Exp
  ty::SType
  start::Exp
  step::Union{Nothing, Exp}
  stop::Exp
end

"`exp[subs...]` for an expression that is not a variable."
struct ASUB <: Exp
  exp::Exp
  subs::Vector{Exp}
end

"Output `index` of the tuple `exp`."
struct TSUB <: Exp
  exp::Exp
  index::Int
  ty::SType
end

"Field `fieldName` (number `index`) of the record `exp`."
struct RSUB <: Exp
  exp::Exp
  index::Int
  fieldName::String
  ty::SType
end

"A call `name(args...)` of type `ty`."
struct CALL <: Exp
  name::Symbol
  args::Vector{Exp}
  ty::SType
  kind::CallKind
end

"`exp` converted to type `ty` (`Integer` to `Real`)."
struct CAST <: Exp
  ty::SType
  exp::Exp
end

"A record value: the record's name, its fields' values and names, its type."
struct RECORD <: Exp
  name::Symbol
  exps::Vector{Exp}
  fieldNames::Vector{String}
  ty::SType
end

"A tuple: the outputs of a call to a function with several outputs."
struct TUPLE <: Exp
  elements::Vector{Exp}
end

"An iterator of a reduction or an array constructor: `name in range`."
struct Iterator <: Node
  name::Symbol
  range::Exp
end

"""
    REDUCTION(reducer, body, iterators, ty)

`reducer(body for iterators)`: `reducer` is `:sum`, `:product`, `:min`, `:max`, or `:array`
for an array constructor `{body for iterators}`.
"""
struct REDUCTION <: Exp
  reducer::Symbol
  body::Exp
  iterators::Vector{Iterator}
  ty::SType
end

"""
    PARTEVALFUNCTION(name, args, argNames, inputs, ty)

`function name(argNames[i] = args[i], ...)`: a function value with some inputs bound, of type
`ty` (the inputs not bound); `inputs` are all the function's inputs as declared (records not
flattened), by name and type.
"""
struct PARTEVALFUNCTION <: Exp
  name::Symbol
  args::Vector{Exp}
  argNames::Vector{Symbol}
  inputs::Vector{Pair{Symbol, SType}}
  ty::TYPE_FUNCTION
end

#= ==================================================================================== =#
#= Statements                                                                           =#
#= ==================================================================================== =#

"""
A statement of an algorithm, a function body, a when-equation's body or an initial
algorithm.
"""
abstract type Statement <: Node end

"`left := right`; `left` a variable, a subscripted one, or a `TUPLE` of them."
struct ASSIGN <: Statement
  left::Exp
  right::Exp
  source::Source
end

"A condition and the statements executed when it holds."
struct Branch <: Node
  condition::Exp
  body::Vector{Statement}
end

"`if ... elseif ... else ... end if`: the branches in order, then the else-statements."
struct IF <: Statement
  branches::Vector{Branch}
  elseBody::Vector{Statement}
  source::Source
end

"`for iterator loop body end for`."
struct FOR <: Statement
  iterator::Iterator
  body::Vector{Statement}
  source::Source
end

"`while condition loop body end while`."
struct WHILE <: Statement
  condition::Exp
  body::Vector{Statement}
  source::Source
end

"`when ... elsewhen ... end when` in an algorithm: the branches in order."
struct WHEN <: Statement
  branches::Vector{Branch}
  source::Source
end

"`assert(condition, message, level)`."
struct ASSERT <: Statement
  condition::Exp
  message::Exp
  level::Exp
  source::Source
end

"`terminate(message)`."
struct TERMINATE <: Statement
  message::Exp
  source::Source
end

"`reinit(stateVar, value)`."
struct REINIT <: Statement
  stateVar::SimCref
  value::Exp
  source::Source
end

"A call without outputs as a statement."
struct NORETCALL <: Statement
  exp::Exp
  source::Source
end

"`return`."
struct RETURN <: Statement
  source::Source
end

"`break`."
struct BREAK <: Statement
  source::Source
end

"OM.jl's structural change: translate the model again with `componentToChange = newValue`."
struct RECOMPILATION <: Statement
  componentToChange::SimCref
  newValue::Exp
end

"OM.jl's structural change decided by an agent: the components it may change, its prompt."
struct AGENTIC_RECOMPILATION <: Statement
  componentsToChange::Vector{SimCref}
  prompt::Union{Nothing, String}
  initialEquations::Union{Nothing, String}
end

#= ==================================================================================== =#
#= Equations                                                                            =#
#= ==================================================================================== =#

"""
    EQ_ATTR(differentiated, alsoInitial)

`differentiated`: the equation is the derivative of another (index reduction). `alsoInitial`:
a when-equation's runtime arm whose condition had `initial()` too.
"""
struct EQ_ATTR <: Node
  differentiated::Bool
  alsoInitial::Bool
end

const EQ_ATTR_DEFAULT = EQ_ATTR(false, false)

"An equation of the simulation code."
abstract type Equation end

"`0 = exp`: the bulk of the system."
struct RESIDUAL_EQUATION <: Equation
  exp::Exp
  source::Source
  attr::EQ_ATTR
end

"`lhs = rhs`, not residualized: initial equations."
struct EQUATION <: Equation
  lhs::Exp
  rhs::Exp
  source::Source
  attr::EQ_ATTR
end

"`left = right` of arrays of dimensions `dimSize`."
struct ARRAY_EQUATION <: Equation
  dimSize::Vector{Int}
  left::Exp
  right::Exp
  source::Source
  attr::EQ_ATTR
end

"An algorithm section: its statements."
struct ALGORITHM <: Equation
  statements::Vector{Statement}
  source::Source
  attr::EQ_ATTR
end

"""
    WHEN_STMTS(condition, body, elsewhenPart)

A when-clause: its condition, the statements of its body, the `elsewhen` part (`nothing` for
none).
"""
struct WHEN_STMTS <: Node
  condition::Exp
  body::Vector{Statement}
  elsewhenPart::Union{Nothing, WHEN_STMTS}
end

"A when-equation evaluated at the events of its conditions."
struct WHEN_EQUATION <: Equation
  whenEquation::WHEN_STMTS
  source::Source
  attr::EQ_ATTR
end

"`when initial() then ... end when`: evaluated once, at the initialization."
struct INITIAL_WHEN_EQUATION <: Equation
  whenEquation::WHEN_STMTS
  source::Source
  attr::EQ_ATTR
end

"""
    INLINE_IF_EQUATION(conditions, branchesTrue, branchElse, source, attr)

An if-equation in a list of equations (initial equations), not lifted to IF_EQUATION's
branches: a condition per `if`/`elseif` arm, the arms' equations, the else-arm's.
"""
struct INLINE_IF_EQUATION <: Equation
  conditions::Vector{Exp}
  branchesTrue::Vector{Vector{Equation}}
  branchElse::Vector{Equation}
  source::Source
  attr::EQ_ATTR
end

"`assert(condition, message, level)` outside a when-clause: checked after each step."
struct ASSERT_EQUATION <: Equation
  condition::Exp
  message::Exp
  level::Exp
  source::Source
end

#= ==================================================================================== =#
#= Variables                                                                            =#
#= ==================================================================================== =#

"The category of a simulation variable."
abstract type SimVarType <: Node end

"A state."
struct STATE <: SimVarType end

"The derivative of the state `varName`."
struct STATE_DERIVATIVE <: SimVarType
  varName::String
end

"""
An algebraic variable; `sortIdx` its index for the backend's structural analysis (not the
index of the generated code).
"""
struct ALG_VARIABLE <: SimVarType
  sortIdx::Int
end

"A top-level input."
struct INPUT <: SimVarType end

"A String variable or parameter, with its binding."
struct STRING <: SimVarType
  bindExp::Union{Nothing, Exp}
end

"A constant, a record or external object parameter: a module-level binding."
struct DATA_STRUCTURE <: SimVarType
  bindExp::Union{Nothing, Exp}
end

"An array variable kept as an array: its dimensions and binding."
struct ARRAY <: SimVarType
  dimensions::Vector{Int}
  bindExp::Union{Nothing, Exp}
end

"A parameter, with its binding."
struct PARAMETER <: SimVarType
  bindExp::Union{Nothing, Exp}
end

"An array parameter: its dimensions and binding."
struct ARRAY_PARAMETER <: SimVarType
  dimensions::Vector{Int}
  bindExp::Union{Nothing, Exp}
end

"A discrete-time variable."
struct DISCRETE <: SimVarType end

"""
    SimVarAttributes(start, fixed, nominal, min, max, unit, displayUnit, quantity, stateSelect,
                     isProtected)

A variable's built-in attributes the code generators read: an expression for each one given,
`nothing` for one not; `isProtected` for a protected variable.
"""
struct SimVarAttributes <: Node
  start::Union{Nothing, Exp}
  fixed::Union{Nothing, Exp}
  nominal::Union{Nothing, Exp}
  min::Union{Nothing, Exp}
  max::Union{Nothing, Exp}
  unit::Union{Nothing, Exp}
  displayUnit::Union{Nothing, Exp}
  quantity::Union{Nothing, Exp}
  stateSelect::Union{Nothing, Symbol}
  isProtected::Bool
end

const NO_ATTRIBUTES = SimVarAttributes(nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                                       nothing, nothing, false)

"""
    SIMVAR(name, index, varKind, attributes, elementType)

A simulation variable: its name, index (1-based, per category; `nothing` before indexing),
category, attributes, and the type of its scalars (of an array variable, of its elements):
what the code generators read to declare an Integer, Boolean or enumeration variable.
"""
struct SIMVAR <: Node
  name::String
  index::Union{Nothing, Int}
  varKind::SimVarType
  attributes::SimVarAttributes
  elementType::SType
end

#= ==================================================================================== =#
#= Functions and initial algorithms                                                     =#
#= ==================================================================================== =#

"""
    FunctionVariable(name, ty, dimensions, default, fieldDefaults)

An input, output or protected variable of a function: its name; its type (an array's
TYPE_ARRAY, a size an expression gives -1); an array's dimensions as expressions, for its
allocation (`ICONST(n)`, an expression of the inputs `size(u, 1)`, `ICONST(0)` for `:`); its
default value (`nothing` for none); for a record instance its fields' defaults (the
instance's modifiers, the record's), by field.
"""
struct FunctionVariable <: Node
  name::String
  ty::SType
  dimensions::Vector{Exp}
  default::Union{Nothing, Exp}
  fieldDefaults::Vector{Pair{String, Exp}}
end

"""
    FunctionDerivative(name, excludedInputs, nInputs)

A function's derivative annotation of order 1 (MLS 12.7.1): the derivative function, the
inputs (by position) without a derivative input in it (zeroDerivative, noDerivative), the
function's number of inputs (as declared, before its records are flattened).
"""
struct FunctionDerivative <: Node
  name::String
  excludedInputs::Vector{Int}
  nInputs::Int
end

"A function the model calls."
abstract type ModelicaFunction end

"""
    MODELICA_FUNCTION(name, inputs, outputs, locals, statements, derivative)

A Modelica function: its variables (a record input or output as its fields'), its algorithm,
its derivative annotation (`nothing` for none).
"""
struct MODELICA_FUNCTION <: ModelicaFunction
  name::String
  inputs::Vector{FunctionVariable}
  outputs::Vector{FunctionVariable}
  locals::Vector{FunctionVariable}
  statements::Vector{Statement}
  derivative::Union{Nothing, FunctionDerivative}
end

"""
    ExternalCall(name, language, arguments, output, annotation)

The external clause of a function: the external function's name and language (`"C"`,
`"FORTRAN 77"`), its arguments (in the function's variables), the variable receiving its
result (`nothing` for a void function), the clause's annotation as the flat model writes it
(`Include`, `IncludeDirectory`, `Library`, `LibraryDirectory`; empty for none).
"""
struct ExternalCall <: Node
  name::String
  language::String
  arguments::Vector{Exp}
  output::Union{Nothing, Exp}
  annotation::String
end

"An external function: its variables, its external call, its derivative annotation."
struct EXTERNAL_MODELICA_FUNCTION <: ModelicaFunction
  name::String
  inputs::Vector{FunctionVariable}
  outputs::Vector{FunctionVariable}
  locals::Vector{FunctionVariable}
  external::ExternalCall
  derivative::Union{Nothing, FunctionDerivative}
end

"""
    INITIAL_ALGORITHM(statements, section)

Statements run once, at the initialization (MLS 11.4): an `initial algorithm` section or an
algorithm's `when initial()` body (`section`: the initialization's first pass runs them, their
results start values of the system), or the `initial()` arm of a when-equation, of a lifted
algorithm or discrete cluster.
"""
struct INITIAL_ALGORITHM
  statements::Vector{Statement}
  section::Bool
end

#= ==================================================================================== =#
#= If-equations, structural transitions, the simulation code                            =#
#= ==================================================================================== =#

"Control-flow constructs of the simulation code."
abstract type Construct end

"""
    BRANCH(condition, residualEquations, identifier, targets)

A branch of an if-equation: its condition, its residual equations, its identifier (`-1` for
the else-branch) and the identifier of the branch evaluated when its condition does not hold
(`-1` for the else-branch).
"""
struct BRANCH <: Construct
  condition::Exp
  residualEquations::Vector{RESIDUAL_EQUATION}
  identifier::Int
  targets::Int
end

"The identifier of an else-branch."
const ELSE_BRANCH = -1

"An if-equation of the simulation code: its branches."
struct IF_EQUATION <: Construct
  branches::Vector{BRANCH}
end

"A transition between structural modes."
abstract type StructuralTransition end

"An explicit transition from mode `fromState` to `toState` when `transitionCondition` holds."
struct EXPLICIT_STRUCTURAL_TRANSITION <: StructuralTransition
  fromState::String
  toState::String
  transitionCondition::Exp
end

"A transition in a when-clause: its mode is known when the body runs."
struct IMPLICIT_STRUCTURAL_TRANSITION <: StructuralTransition
  whenEquation::WHEN_STMTS
  source::Source
  attr::EQ_ATTR
end

"""
    EliminationOptions(; reachability = true, patterns = Regex[], keepPatterns = Regex[])

Options of the non-dynamic variable elimination (`eliminateNonDynamic`): the variables the
dynamics do not need (not reachable from the state derivatives) are eliminated, all of them with
`reachability`, else those matching `patterns`; never those matching `keepPatterns`.
"""
struct EliminationOptions
  reachability::Bool
  patterns::Vector{Regex}
  keepPatterns::Vector{Regex}
end

EliminationOptions(; reachability::Bool = true, patterns::Vector{Regex} = Regex[],
                   keepPatterns::Vector{Regex} = Regex[]) =
  EliminationOptions(reachability, patterns, keepPatterns)

"An eliminated alias: `eliminatedName = representativeName`, or its negation (`negated`)."
struct AliasEntry
  eliminatedName::String
  representativeName::String
  negated::Bool
end

"""
    SIM_CODE

The simulation code of a model: its variables by name (index, SimVar), its residual, initial,
when- and if-equations, its structural modes and transitions, the functions it calls, its
initial algorithms and asserts, and what the passes eliminated (for observed equations). A
model with structural modes has a simulation code per mode (`subModels`).
"""
struct SIM_CODE
  name::String
  stringToSimVarHT::OrderedDict{String, Tuple{Int, SIMVAR}}
  residualEquations::Vector{RESIDUAL_EQUATION}
  initialEquations::Vector{Equation}
  whenEquations::Vector{WHEN_EQUATION}
  ifEquations::Vector{IF_EQUATION}
  structuralTransitions::Vector{StructuralTransition}
  subModels::Vector{SIM_CODE}
  sharedVariables::Vector{String}
  topVariables::Vector{String}
  sharedEquations::Vector{Equation}
  activeModel::String
  "What the frontend needs to translate the model again at a structural change (opaque)."
  frontendModel::Any
  irreducibleVariables::Vector{String}
  functions::Vector{ModelicaFunction}
  externalRuntime::Bool
  eliminatedEquations::Vector{RESIDUAL_EQUATION}
  eliminatedVariables::Vector{String}
  aliasMap::Vector{AliasEntry}
  observedFilter::Union{Nothing, Vector{String}}
  initialAlgorithms::Vector{INITIAL_ALGORITHM}
  asserts::Vector{ASSERT_EQUATION}
  "The model's variables `foldParameterClosure` made parameters: observed when a pass drops them."
  foldedVariables::Vector{String}
end

"""
    update(simCode; fields...)

`simCode` with the fields named in `fields` replaced: `update(sc; residualEquations = eqs)`.
"""
function update(sc::SIM_CODE; fields...)::SIM_CODE
  for name in keys(fields)
    hasfield(SIM_CODE, name) || throw(ArgumentError("SIM_CODE has no field $name"))
  end
  return SIM_CODE((get(fields, f, getfield(sc, f)) for f in fieldnames(SIM_CODE))...)
end
