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

"Fields compared one by one: the values are trees of vectors, strings and symbols (unrolled per type, as OMJLIR's: `IR.fieldComparison`)."
@generated function Base.:(==)(a::T, b::T)::Bool where {T <: Node}
  return IR.fieldComparison(T, ==)
end
Base.:(==)(::Node, ::Node)::Bool = false

"""
Fields compared one by one with `isequal`, as `hash` hashes them: the identity of a `Dict` or
`Set` key (`RCONST(0.0)` and `RCONST(-0.0)` apart, `RCONST(NaN)` equal to itself; `==` has
them the other way round, as numbers).
"""
@generated function Base.isequal(a::T, b::T)::Bool where {T <: Node}
  return IR.fieldComparison(T, isequal)
end
Base.isequal(::Node, ::Node)::Bool = false

"Fields hashed one by one after the type's name, an enumeration value by its integer: the same in every build (unrolled, as OMJLIR's values: `IR.fieldHashing`, `IR.fieldHash`)."
@generated function Base.hash(a::T, h::UInt)::UInt where {T <: Node}
  return IR.fieldHashing(T, IR.fieldHash)
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
    SimCref(sym, subs, ref)

A reference to a simulation variable: its key (`body_frame_a_r_0`, `x[2]`), for an element of an
array variable kept as an array the integer subscripts, and the IR reference it was lowered from
(`ref`: the variable's, an element's with its subscripts, a record's field's, a function's local's), by
whose parts every structural question is answered (the array an element is of, a record's field). The
key is the reference's flat name (`refKey`; a function's local's its identifier, `derivedKey`); in the
code view (`codeView`) the identifier. Equal by key and subscripts: a single `===` on the symbol compares
names. `ref` is `nothing` only in tests.
"""
struct SimCref
  sym::Symbol
  subs::Vector{Int}
  ref::Union{Nothing, IR.ComponentRef}
end

"A reference by its key alone, without its IR reference: tests."
SimCref(sym::Symbol) = SimCref(sym, Int[], nothing)

"The IR reference of `c`; an error naming its key if it has none (a test's)."
irRef(c::SimCref)::IR.ComponentRef =
  c.ref === nothing ? error("SimCode reference $(c.sym) has no IR reference") : c.ref

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

"""
    TSUB(exp, index, ty)

Element `index` of the tuple a call `exp` returns: its position in the function's result as generated, a record
output's fields in place (`Passes.tupleAssignElements`: `(r, q) := f(x)` assigns `r.a`, `r.b`, `q` elements 1, 2, 3).
"""
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

"""
    REINIT(stateVar, value, source, subscripts)

`reinit(stateVar, value)`; of an element of an array state of a model with its arrays kept (`lowerArraySystem`) the
state variable and the element's subscripts (an iterator's, `x[i]`), none otherwise.
"""
struct REINIT <: Statement
  stateVar::SimCref
  value::Exp
  source::Source
  subscripts::Vector{Exp}
end

"`reinit(stateVar, value)` of a state, no subscripts."
REINIT(stateVar::SimCref, value::Exp, source::Source) = REINIT(stateVar, value, source, Exp[])

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
    EQ_ATTR(differentiated, alsoInitial, everyEvent = false)

`differentiated`: the equation is the derivative of another (index reduction). `alsoInitial`:
a when-equation's runtime arm whose condition had `initial()` too. `everyEvent`: a when-equation
of discrete equations (a lifted cluster, OMJLIR's `EQ_DISCRETE`) that the array code runs
where it runs the discrete equations in event mode, at every pass of every event and of the
start's iteration, its conditions the events' sources (OpenModelica's discrete equations: held
between events, pre() the initial values after the initialization).
"""
struct EQ_ATTR <: Node
  differentiated::Bool
  alsoInitial::Bool
  everyEvent::Bool
end

EQ_ATTR(differentiated::Bool, alsoInitial::Bool) = EQ_ATTR(differentiated, alsoInitial, false)

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

"""
    FOR_EQUATION(iterator, body, source, attr)

`for iterator loop body end for`: the equations of `body` for each value of the iterator's range, which they read
as an `EXP_CREF` of its name (of no IR reference); a model with its arrays kept (`SIM_CODE.arrayEquations`).
"""
struct FOR_EQUATION <: Equation
  iterator::Iterator
  body::Vector{Equation}
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

"""
    UNLOWERED_EQUATION(what, construct, source)

An equation of a branch of an if-equation the lowering of a model with its arrays kept (`lowerArraySystem`) could not
express (UnsupportedLowering: what, and the construct's text): the branch may never be taken (a parameter's condition),
so the code generators refuse it only where they reach it, with the reason.
"""
struct UNLOWERED_EQUATION <: Equation
  what::String
  construct::String
  source::Source
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

"""
    PARAMETER(bindExp, value = nothing)

A parameter (an array parameter, `SIMVAR.dims`, too), with its binding and its value where the lowering evaluated it (a
model with its arrays kept: `lowerArraySystem`, by OMJLIR's Evaluation; a constant, an array one an array literal);
`nothing` otherwise, and once its binding changes (`withBinding`).
"""
struct PARAMETER <: SimVarType
  bindExp::Union{Nothing, Exp}
  value::Union{Nothing, Exp}
end

PARAMETER(bindExp::Union{Nothing, Exp}) = PARAMETER(bindExp, nothing)

"A discrete-time variable."
struct DISCRETE <: SimVarType end

"""
    SimVarAttributes(start, fixed, nominal, min, max, unit, displayUnit, quantity, stateSelect,
                     isProtected, startExps = nothing)

A variable's built-in attributes the code generators read: an expression for each one given,
`nothing` for one not; `isProtected` for a protected variable. `startExps`: the start of each
element (row-major) as written, where one reads a parameter (the array code's `start` is the
evaluated constants; it computes these again from overridden parameters); `nothing` otherwise.
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
  startExps::Union{Nothing, Vector{Exp}}
end

SimVarAttributes(start, fixed, nominal, min, max, unit, displayUnit, quantity, stateSelect, isProtected::Bool) =
  SimVarAttributes(start, fixed, nominal, min, max, unit, displayUnit, quantity, stateSelect, isProtected, nothing)

const NO_ATTRIBUTES = SimVarAttributes(nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                                       nothing, nothing, false)

"""
    SIMVAR(name, index, varKind, attributes, elementType, dims, irKind, irOrigin, ref)
    SIMVAR(name, index, varKind, attributes, elementType)

A simulation variable: its name, index (1-based, per category; `nothing` before indexing),
category, attributes, the type of its scalars (of an array variable, of its elements):
what the code generators read to declare an Integer, Boolean or enumeration variable; its
dimensions (an array variable's sizes, outermost first; none for a scalar): its shape, apart
from its category (an array state is a `STATE`, an array parameter a `PARAMETER`); its kind in OMJLIR (`IR.VariableKind`: a state, a state's derivative as an algebraic
unknown, index reduction's dummy derivative or dummy state, ...), the variables it names
by their simulation names (`simKind`), its origin there (`:model` for one of the flat
model, else the pass that introduced it) and its reference there (`ref`: its parts, from which its
name is derived; a state's derivative the IR derivative variable's); `nothing` for a variable that is
not the IR's.
"""
struct SIMVAR <: Node
  name::String
  index::Union{Nothing, Int}
  varKind::SimVarType
  attributes::SimVarAttributes
  elementType::SType
  dims::Vector{Int}
  irKind::Union{Nothing, IR.VariableKind}
  irOrigin::Union{Nothing, Symbol}
  ref::Union{Nothing, IR.ComponentRef}
end

"A scalar variable of no IR kind, origin and reference: tests."
SIMVAR(name::String, index::Union{Nothing, Int}, varKind::SimVarType, attributes::SimVarAttributes, elementType::SType) =
  SIMVAR(name, index, varKind, attributes, elementType, Int[], nothing, nothing, nothing)

#= ==================================================================================== =#
#= Functions and initial algorithms                                                     =#
#= ==================================================================================== =#

"""
    FunctionVariable(name, ty, dimensions, default, fieldDefaults, cref)

An input, output or protected variable of a function: its name (its identifier in the
generated function); its type (an array's TYPE_ARRAY, a size an expression gives -1); an
array's dimensions as expressions, for its allocation (`ICONST(n)`, an expression of the
inputs `size(u, 1)`, `ICONST(0)` for `:`); its default value (`nothing` for none); for a record
instance its fields' defaults (the instance's modifiers, the record's), by field; its IR
reference (what the function body's references to it are, the name derived from it).
"""
struct FunctionVariable <: Node
  name::String
  ty::SType
  dimensions::Vector{Exp}
  default::Union{Nothing, Exp}
  fieldDefaults::Vector{Pair{String, Exp}}
  cref::IR.ComponentRef
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
    ReferenceIndex(references, keyByReference, arrayOf, codeNames, elementNames)
    ReferenceIndex(ht, order, modelName; elementNames)

The simulation code's variables by their IR references and the names derived from them, built together from the
variables' references once at lowering (`ReferenceIndex(ht, order, modelName)`) and renamed together (`codeView`):
- `references`: every variable's IR reference by its key (`SIMVAR.ref`), the eliminated ones' too (a state's
  derivative the IR derivative variable's): the structure of a name (`referenceOf`);
- `keyByReference`: the keys by the references' names, `references` inverted (`keyOf`);
- `arrayOf`: the array each element variable is an element of, by its key (`arrayBase` of its reference;
  `elementArray`);
- `codeNames`: each variable's identifier in the generated code by its key (`codeIdentifiers`; `codeName`): what the
  code generators and OM.jl's results know it by (`a_b[1][2]` of `a.b[1,2]`, `der(x)` of a state's derivative).
"""
struct ReferenceIndex
  references::Dict{String, IR.ComponentRef}
  keyByReference::Dict{Symbol, String}
  arrayOf::Dict{String, Symbol}
  codeNames::Dict{String, String}
  "An array variable's element names (row-major), by its key: a model with its arrays kept (`lowerArraySystem`)."
  elementNames::Dict{String, Vector{String}}
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
  "OMBackend reduced the model's index (Passes.IndexReduction, OMBACKEND_INDEX_REDUCTION): MTK only tears."
  indexReduced::Bool
  """
  The variables' IR references, the keys by them, the arrays of the elements and the identifiers in the generated
  code (`ReferenceIndex`). The code generators see the simulation code renamed to the identifiers (`codeView`); the
  passes match user-given names against them.
  """
  referenceIndex::ReferenceIndex
  """
  The equations of a model lowered with its arrays kept (`lowerArraySystem`), in their order: scalar ones as written
  (`EQUATION`, a tuple's too), for-equations (`FOR_EQUATION`), if-equations (`INLINE_IF_EQUATION`), asserts, when-
  equations (`WHEN_EQUATION`, the initial ones `INITIAL_WHEN_EQUATION`). Empty for a scalar model, whose equations
  are in the fields above; no pass of the scalar model's reads them.
  """
  arrayEquations::Vector{Equation}
  "The initial equations of a model lowered with its arrays kept, in their order (as `arrayEquations`)."
  arrayInitialEquations::Vector{Equation}
end

"""
The array the variable `key` is an element of (`arrayBase` of its reference: `x` of `x[2]`, `c` of `c[2].T`), from
`ReferenceIndex.arrayOf`; `nothing` for a variable that is no element.
"""
elementArray(sc::SIM_CODE, key::String)::Union{Nothing, Symbol} = get(sc.referenceIndex.arrayOf, key, nothing)

"The IR reference of the variable `key` (`ReferenceIndex.references`, an eliminated variable's too); `nothing` for a name of no variable."
referenceOf(sc::SIM_CODE, key::String)::Union{Nothing, IR.ComponentRef} = get(sc.referenceIndex.references, key, nothing)

"The arrays of the element variables of `references`, by their keys (`ReferenceIndex.arrayOf`)."
function elementArrays(references::Dict{String, IR.ComponentRef})::Dict{String, Symbol}
  local out::Dict{String, Symbol} = Dict{String, Symbol}()
  for (key::String, ref::IR.ComponentRef) in references
    local array::Symbol = arrayBase(ref)
    array === ref.name || (out[key] = array)
  end
  return out
end

"The identifier of the variable `key` in the generated code (`ReferenceIndex.codeNames`); `key` itself for a name of no variable (`time`, an iterator)."
codeName(sc::SIM_CODE, key::AbstractString)::String = get(sc.referenceIndex.codeNames, key, key)

"The key of the variable the IR reference `ref` (or of the reference name `name`) names; `nothing` for one that names none."
keyOf(sc::SIM_CODE, ref::IR.ComponentRef)::Union{Nothing, String} = keyOf(sc, ref.name)
keyOf(sc::SIM_CODE, name::Symbol)::Union{Nothing, String} = get(sc.referenceIndex.keyByReference, name, nothing)

"""
    identifierOf(sc, ref) -> String

The identifier of the IR reference `ref` in the code view `sc` (`codeView`): its variable's (`keyOf`), else derived
from the reference (`derivedKey`: a field of a record whose fields are the variables, an element of an array kept
whole).
"""
function identifierOf(sc::SIM_CODE, ref::IR.ComponentRef)::String
  local id::Union{Nothing, String} = keyOf(sc, ref)
  return id === nothing ? derivedKey(ref) : id
end

"The IR reference of the variable `v`; an error naming it if it has none (a test's mock)."
irRef(v::SIMVAR)::IR.ComponentRef = v.ref === nothing ? error("SimCode variable $(v.name) has no IR reference") : v.ref

"The keys by the IR references' names of `references`; an error naming both keys if two variables have one reference."
function referenceKeys(references::Dict{String, IR.ComponentRef})::Dict{Symbol, String}
  local byReference::Dict{Symbol, String} = Dict{Symbol, String}()
  for (key::String, ref::IR.ComponentRef) in references
    local other::Union{Nothing, String} = get(byReference, ref.name, nothing)
    other === nothing || error("SimCode variables $(join(sort([other, key]), " and ")) have one IR reference, $(ref.name)")
    byReference[ref.name] = key
  end
  return byReference
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
