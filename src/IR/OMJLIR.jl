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

"""
    OMJLIR

OMBackend's intermediate representation of a model, modelled on the new OpenModelica
backend's (NBackendDAE): variables with a structural kind (state, state derivative,
algebraic, discrete, parameter, ...), equations with a kind (continuous, discrete, clocked,
initial), and partitions of equations evaluated together (the continuous system, the clocked
partitions of MLS 16).

The frontend's flat model is lowered into OMJLIR, the backend passes transform it, and the
simulation code is generated from it. OMJLIR depends on no frontend package: names are
`Symbol`s, expressions, types and statements are its own.

Conventions:
- A variable's name is its flat Modelica name (`:x`, `Symbol("a.b")`,
  `Symbol("r[2].v")` for an element of a scalarized array of records). A component reference
  is that name and the subscripts selecting elements of an array variable (`x[i]`).
- The derivative of a state `x` is a variable of its own, `\$DER.x` (`derivativeName`), of
  kind `STATE_DER`; `der(x)` in an equation is a reference to it. A state and its derivative
  name each other (`STATE.derivative`, `STATE_DER.state`). With the derivatives as unknowns
  and the states as knowns, a partition's structure (adjacency, matching, strong components)
  is that of NB, so index reduction (Pantelides, dummy derivatives) can work on it: it adds
  derivative variables and differentiated equations, links an equation to its derivative
  (`EquationAttributes.derivative`) and marks dummy derivatives and states (`DUMMY_DER`,
  `DUMMY_STATE`).
- Expressions, types and component references are immutable values with structural equality
  (`==`, `hash`). Variables and equations are mutable entities with identity (NB's
  pointers): a pass changes a variable's kind or an equation's attributes in place, and every
  partition holding it sees the change.
- `nothing` for an absent optional value.
- NF's `Type` and `Function` are `ModelicaType` and `ModelicaFunction` here: in Julia, those
  names would shadow `Base.Type` and `Base.Function`.
"""
module OMJLIR

import ..Traversal
import ..Traversal: traverseFields, traverseElements

export Node, Expression, ModelicaType, Subscript, Dimension, ComponentRef, ClockKind
export Equation, Statement, Variable, VariableKind, VariableSet, ModelicaFunction, Partition, Model

#= ==================================================================================== =#
#= Abstract types: declared first, the concrete types refer to each other.              =#
#= ==================================================================================== =#

"Root of the immutable IR values that compare structurally (`==`, `hash`)."
abstract type Node end

"A Modelica expression."
abstract type Expression <: Node end

"The type of an expression or a variable."
abstract type ModelicaType <: Node end

"One subscript of a component reference or a subscripted expression."
abstract type Subscript <: Node end

"One dimension of an array type."
abstract type Dimension <: Node end

"The kind of a clock constructor (MLS 16.3)."
abstract type ClockKind <: Node end

"An equation of the model: a mutable entity, compared by identity."
abstract type Equation end

"A statement of an algorithm, a function body or a when-equation."
abstract type Statement <: Node end

"The structural kind of a variable."
abstract type VariableKind <: Node end

"How a partition is evaluated: continuously, or at the ticks of a clock."
abstract type Association <: Node end

"Fields compared one by one: the IR values are trees of vectors, strings and symbols."
function Base.:(==)(a::T, b::T)::Bool where {T <: Node}
  for f in fieldnames(T)
    getfield(a, f) == getfield(b, f) || return false
  end
  return true
end
Base.:(==)(::Node, ::Node)::Bool = false

"""
Fields compared one by one with `isequal`, as `hash` hashes them: the identity of a `Dict` or
`Set` key (`REAL(0.0)` and `REAL(-0.0)` apart, `REAL(NaN)` equal to itself; `==` has them the
other way round, as numbers).
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
#= Source locations                                                                     =#
#= ==================================================================================== =#

"""
    Source(file, line, column)

Where an equation, a statement or a variable comes from in the Modelica source, for the
messages. `NO_SOURCE` for one the backend creates.
"""
struct Source <: Node
  file::String
  line::Int
  column::Int
end

"The source of what the backend creates."
const NO_SOURCE = Source("", 0, 0)

#= ==================================================================================== =#
#= Types                                                                                =#
#= ==================================================================================== =#

"`Integer`."
struct T_INTEGER <: ModelicaType end
"`Real`."
struct T_REAL <: ModelicaType end
"`Boolean`."
struct T_BOOLEAN <: ModelicaType end
"`String`."
struct T_STRING <: ModelicaType end
"`Clock` (MLS 16)."
struct T_CLOCK <: ModelicaType end

"An enumeration type: its name and its literals in order (literal `i` has index `i`)."
struct T_ENUMERATION <: ModelicaType
  name::Symbol
  literals::Vector{Symbol}
end

"An array type: its element type and its dimensions, outermost first."
struct T_ARRAY <: ModelicaType
  elementType::ModelicaType
  dimensions::Vector{Dimension}
end

"A field of a record type."
struct Field <: Node
  name::Symbol
  ty::ModelicaType
end

"A record type: its name (also the name of its constructor function) and its fields."
struct T_RECORD <: ModelicaType
  name::Symbol
  fields::Vector{Field}
end

"An external object class (MLS 12.9.7): its name, also of its constructor function."
struct T_EXTERNAL_OBJECT <: ModelicaType
  name::Symbol
end

"""
    T_CLASS(name)

The type of a component of a model, block or connector class: of a part of a component
reference (`body.frame_a.r_0`'s `body` and `frame_a`), never of a value.
"""
struct T_CLASS <: ModelicaType
  name::Symbol
end

"""
    T_COMPONENTS(cls, members)

The type of a part that stands for separately declared components of the class `cls` (EngineV6's `cylinder1`, ...,
`cylinder6`) rolled into an array of `length(members)` (`IndexReduction.rollComponents!`): the part subscripted by a
literal `k` is named as the member `members[k]` (`flatName`), so an element keeps its flat name (`cylinder3.x`).
"""
struct T_COMPONENTS <: ModelicaType
  cls::Symbol
  members::Vector{Symbol}
end

"The type of a call to a function with several outputs."
struct T_TUPLE <: ModelicaType
  types::Vector{ModelicaType}
end

"The type of a function value (a functional input argument, a partial application)."
struct T_FUNCTION <: ModelicaType
  name::Symbol
  inputs::Vector{ModelicaType}
  outputs::Vector{ModelicaType}
end

"The type of a call to a function without outputs."
struct T_NORETCALL <: ModelicaType end

"A type the backend does not know (yet)."
struct T_UNKNOWN <: ModelicaType end

"A dimension of known size."
struct DIM_SIZE <: Dimension
  size::Int
end

"A dimension indexed by the literals of an enumeration: `Real x[E]`."
struct DIM_ENUM <: Dimension
  ty::T_ENUMERATION
end

"A dimension indexed by `false`, `true`: `Real x[Boolean]`."
struct DIM_BOOLEAN <: Dimension end

"A dimension whose size is an expression (a parameter not evaluated)."
struct DIM_EXP <: Dimension
  exp::Expression
end

"A dimension of unknown size (`:`)."
struct DIM_UNKNOWN <: Dimension end

#= ==================================================================================== =#
#= Component references and subscripts                                                  =#
#= ==================================================================================== =#

"""
    CrefPart(name, subscripts, ty)

One part of a component reference: a name, its subscripts (`b[2]` of `a.b[2].c`) and the type
of the component it names, before the subscripts (`Real[3]` of `b`).
"""
struct CrefPart <: Node
  name::Symbol
  subscripts::Vector{Subscript}
  ty::ModelicaType
end

"""
    ComponentRef(parts)
    ComponentRef(name)

A reference to a variable, `a.b[2].c`: its parts, the outermost first, and its flat `name`,
the parts joined with their subscripts (`x[1,2]`, `E.one` for an enumeration literal, an
expression as written: `x[i]`). A variable's name is the flat name of its reference (a
scalarized array's element `x[2]` is a variable of its own); a reference `x[i]` to an array
variable `x` is the variable subscripted (`variableName`).

Two references are equal when their names are: the parts are for the passes that rewrite
subscripts (an algorithm's loop over a scalarized array reads `x[i]`, then `x[2]`).
`ComponentRef(name)` is a reference of one part, for the names the backend makes (`\$DER.x`).
"""
struct ComponentRef <: Node
  parts::Vector{CrefPart}
  name::Symbol
  ComponentRef(parts::Vector{CrefPart}) = new(parts, flatName(parts))
end

ComponentRef(name::Symbol, ty::ModelicaType = T_UNKNOWN()) = ComponentRef(CrefPart[CrefPart(name, Subscript[], ty)])

Base.:(==)(a::ComponentRef, b::ComponentRef)::Bool = a.name === b.name
Base.isequal(a::ComponentRef, b::ComponentRef)::Bool = a.name === b.name
Base.hash(c::ComponentRef, h::UInt)::UInt = hash(c.name, h)

"The subscripts of the last part: those of an array variable's reference."
subscripts(c::ComponentRef)::Vector{Subscript} = c.parts[end].subscripts

"Whether `c` has a subscript in any part."
hasSubscripts(c::ComponentRef)::Bool = any(p -> !isempty(p.subscripts), c.parts)

"The name of the variable `c` references: its own, or without the last part's subscripts."
function variableName(c::ComponentRef, isVariable::F)::Symbol where {F}
  isVariable(c.name) && return c.name
  isempty(c.parts[end].subscripts) && return c.name
  return flatName(CrefPart[c.parts[1:end - 1]..., CrefPart(c.parts[end].name, Subscript[], c.parts[end].ty)])
end

"Subscript by an index expression: `x[i]`."
struct INDEX <: Subscript
  exp::Expression
end

"Subscript by an array of indices or a range: `x[2:3]`, `x[{1, 3}]`."
struct SLICE <: Subscript
  exp::Expression
end

"The whole dimension: `x[:]`."
struct WHOLE <: Subscript end

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

"""
Arithmetic, logical and relational operators (NF's Operator.Op); the `_EW` ones are
element-wise, on operands of equal dimensions.
"""
@enum Op begin
  ADD; SUB; MUL; DIV; POW
  ADD_EW; SUB_EW; MUL_EW; DIV_EW; POW_EW
  ADD_SCALAR_ARRAY; ADD_ARRAY_SCALAR; SUB_SCALAR_ARRAY; SUB_ARRAY_SCALAR
  MUL_SCALAR_ARRAY; MUL_ARRAY_SCALAR; MUL_VECTOR_MATRIX; MUL_MATRIX_VECTOR
  SCALAR_PRODUCT; MATRIX_PRODUCT
  DIV_SCALAR_ARRAY; DIV_ARRAY_SCALAR; POW_SCALAR_ARRAY; POW_ARRAY_SCALAR; POW_MATRIX
  UMINUS
  AND; OR; NOT
  LESS; LESSEQ; GREATER; GREATEREQ; EQUAL; NEQUAL
end

"An `Integer` literal."
struct INTEGER <: Expression
  value::Int
end

"A `Real` literal."
struct REAL <: Expression
  value::Float64
end

"A `Boolean` literal."
struct BOOLEAN <: Expression
  value::Bool
end

"A `String` literal."
struct STRING <: Expression
  value::String
end

"An enumeration literal: its type, its name and its index (1-based)."
struct ENUM_LITERAL <: Expression
  ty::T_ENUMERATION
  name::Symbol
  index::Int
end

"A clock constructor: `Clock(...)` (MLS 16.3)."
struct CLOCK <: Expression
  kind::ClockKind
end

"A reference to a variable, of type `ty`."
struct CREF <: Expression
  cref::ComponentRef
  ty::ModelicaType
end

"The discarded output in a tuple assignment: `(a, _) = f(x)`."
struct WILD <: Expression end

"An array of `elements`, of type `ty`."
struct ARRAY <: Expression
  elements::Vector{Expression}
  ty::T_ARRAY
end

"A range `start:step:stop`; `step` is `nothing` for `start:stop`."
struct RANGE <: Expression
  start::Expression
  step::Union{Nothing, Expression}
  stop::Expression
  ty::T_ARRAY
end

"A tuple: the outputs of a call to a function with several outputs."
struct TUPLE <: Expression
  elements::Vector{Expression}
  ty::T_TUPLE
end

"A record value, its fields in the order of the record type's."
struct RECORD <: Expression
  elements::Vector{Expression}
  ty::T_RECORD
end

"""
    CallAttributes(builtin, impure, inline)

`builtin`: a Modelica built-in function or operator (`der`, `pre`, `sin`, `sum`, ...), named
by its plain name; otherwise a function of the model's `functions`. `impure`: the function is
impure (MLS 12.3). `inline`: the function has the annotation Inline = true.
"""
struct CallAttributes <: Node
  builtin::Bool
  impure::Bool
  inline::Bool
end

"The attributes of a call to a pure built-in function."
const BUILTIN_CALL = CallAttributes(true, false, false)
"The attributes of a call to a pure function of the model."
const FUNCTION_CALL = CallAttributes(false, false, false)

"A call `name(arguments...)` of type `ty` (`T_TUPLE` for several outputs)."
struct CALL <: Expression
  name::Symbol
  arguments::Vector{Expression}
  ty::ModelicaType
  attributes::CallAttributes
end

"""
    Iterator(name, range)

The iterator of a for-loop, an array constructor or a reduction: `name in range`.
"""
struct Iterator <: Node
  name::Symbol
  range::Expression
end

"An array constructor `{body for iterators}`."
struct ARRAY_CONSTRUCTOR <: Expression
  body::Expression
  iterators::Vector{Iterator}
  ty::T_ARRAY
end

"""
A reduction `reducer(body for iterators)`: `reducer` is `:sum`, `:product`, `:min` or `:max`.
"""
struct REDUCTION <: Expression
  reducer::Symbol
  body::Expression
  iterators::Vector{Iterator}
  ty::ModelicaType
end

"""
    PARTIAL_APPLICATION(name, arguments, argumentNames, inputs, ty)

`function name(argumentNames[i] = arguments[i], ...)`: a function value with some inputs
bound (MLS 12.4.2.1), of type `ty` (the inputs not bound); `inputs` are all the function's
inputs as declared, by name and type (where the bound ones are).
"""
struct PARTIAL_APPLICATION <: Expression
  name::Symbol
  arguments::Vector{Expression}
  argumentNames::Vector{Symbol}
  inputs::Vector{Pair{Symbol, ModelicaType}}
  ty::T_FUNCTION
end

"`size(exp)`, or `size(exp, dimension)`."
struct SIZE <: Expression
  exp::Expression
  dimension::Union{Nothing, Expression}
end

"`exp1 op exp2` for an arithmetic `op`, of type `ty`."
struct BINARY <: Expression
  exp1::Expression
  op::Op
  exp2::Expression
  ty::ModelicaType
end

"`op exp` for an arithmetic `op` (`UMINUS`), of type `ty`."
struct UNARY <: Expression
  op::Op
  exp::Expression
  ty::ModelicaType
end

"`exp1 op exp2` for `AND` and `OR`."
struct LBINARY <: Expression
  exp1::Expression
  op::Op
  exp2::Expression
end

"`not exp`."
struct LUNARY <: Expression
  op::Op
  exp::Expression
end

"""
    RELATION(exp1, op, exp2, index)

`exp1 op exp2` for a relational `op`. `index` numbers the relations that generate events
(zero crossings), from 1; 0 for one that does not (in `noEvent`, of discrete operands).
"""
struct RELATION <: Expression
  exp1::Expression
  op::Op
  exp2::Expression
  index::Int
end

"`if condition then thenExp else elseExp`, of type `ty`."
struct IF_EXPRESSION <: Expression
  condition::Expression
  thenExp::Expression
  elseExp::Expression
  ty::ModelicaType
end

"`exp` converted to type `ty` (`Integer` to `Real`)."
struct CAST <: Expression
  exp::Expression
  ty::ModelicaType
end

"`exp[subscripts]` for an expression that is not a component reference."
struct SUBSCRIPTED <: Expression
  exp::Expression
  subscripts::Vector{Subscript}
  ty::ModelicaType
end

"Output `index` of the tuple `exp`."
struct TUPLE_ELEMENT <: Expression
  exp::Expression
  index::Int
  ty::ModelicaType
end

"Field `field` (number `index`) of the record `exp`."
struct RECORD_ELEMENT <: Expression
  exp::Expression
  index::Int
  field::Symbol
  ty::ModelicaType
end

#= Clocks (MLS 16.3) =#

"`Clock()`: a clock inferred from the clocks it is synchronized with."
struct INFERRED_CLOCK <: ClockKind end

"`Clock(intervalCounter, resolution)`: interval intervalCounter/resolution seconds."
struct INTEGER_CLOCK <: ClockKind
  intervalCounter::Expression
  resolution::Expression
end

"`Clock(interval)`: interval seconds."
struct REAL_CLOCK <: ClockKind
  interval::Expression
end

"`Clock(condition, startInterval)`: an event clock, ticking when `condition` becomes true."
struct BOOLEAN_CLOCK <: ClockKind
  condition::Expression
  startInterval::Expression
end

"`Clock(clock, solverMethod)`: `clock`, for discretizing a continuous-time partition."
struct SOLVER_CLOCK <: ClockKind
  clock::Expression
  solverMethod::Expression
end

#= ==================================================================================== =#
#= Statements: algorithms, function bodies and the bodies of when-equations             =#
#= ==================================================================================== =#

"`lhs := rhs`; `lhs` a component reference, a subscripted one or a `TUPLE` of them."
struct ASSIGNMENT <: Statement
  lhs::Expression
  rhs::Expression
  source::Source
end

"A condition and the statements executed when it holds."
struct StatementBranch <: Node
  condition::Expression
  body::Vector{Statement}
end

"""
    WhenBranch(triggers, atInitial, atTerminal, body)

A branch of a when-equation or a when-statement (MLS 8.3.5, 8.6): the conditions whose rising
edges fire it (`[c]` for `when c`, `[c1, c2]` for `when {c1, c2}`), whether it is also active
at the initialization (`when initial()`, `when {..., initial(), ...}`) and at the end of the
simulation (`terminal()`), and its statements. `when initial()` has no triggers.
"""
struct WhenBranch <: Node
  triggers::Vector{Expression}
  atInitial::Bool
  atTerminal::Bool
  body::Vector{Statement}
end

"`if ... elseif ... else ... end if`: the branches in order, then the else-statements."
struct IF_STATEMENT <: Statement
  branches::Vector{StatementBranch}
  elseBody::Vector{Statement}
  source::Source
end

"`for iterator loop body end for`."
struct FOR_STATEMENT <: Statement
  iterator::Iterator
  body::Vector{Statement}
  source::Source
end

"`while condition loop body end while`."
struct WHILE_STATEMENT <: Statement
  condition::Expression
  body::Vector{Statement}
  source::Source
end

"`when ... elsewhen ... end when` in an algorithm: the branches in order."
struct WHEN_STATEMENT <: Statement
  branches::Vector{WhenBranch}
  source::Source
end

"`assert(condition, message, level)`."
struct ASSERT <: Statement
  condition::Expression
  message::Expression
  level::Expression
  source::Source
end

"`terminate(message)`."
struct TERMINATE <: Statement
  message::Expression
  source::Source
end

"`reinit(state, value)`."
struct REINIT <: Statement
  state::ComponentRef
  value::Expression
  source::Source
end

"A call without outputs as a statement: `print(...)`, an external function's side effect."
struct NORETCALL <: Statement
  call::Expression
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

"""
    RECOMPILATION(component, newValue)

OM.jl's structural change at an event: the model is translated again with the parameter
`component` set to `newValue`.
"""
struct RECOMPILATION <: Statement
  component::ComponentRef
  newValue::Expression
end

"""
    AGENTIC_RECOMPILATION(components, prompt, initialEquations)

OM.jl's structural change at an event decided by an agent: the components it may change, the
prompt and the initial equations for the new model.
"""
struct AGENTIC_RECOMPILATION <: Statement
  components::Vector{ComponentRef}
  prompt::Union{Nothing, String}
  initialEquations::Union{Nothing, String}
end

#= ==================================================================================== =#
#= Equations                                                                            =#
#= ==================================================================================== =#

"""
When an equation is evaluated (NB's EquationKind): in the continuous-time system
(`EQ_CONTINUOUS`), at events only, its unknowns discrete-time (`EQ_DISCRETE`), at the ticks
of a clock (`EQ_CLOCKED`); `EQ_UNKNOWN` before classification.
"""
@enum EquationKind EQ_CONTINUOUS EQ_DISCRETE EQ_CLOCKED EQ_UNKNOWN

"""
    EquationAttributes(kind, differentiated, derivative)

`kind`: when the equation is evaluated. `differentiated`: the equation is the time
derivative of another (made by index reduction). `derivative`: the equation's time
derivative, once index reduction differentiated it; `nothing` before.
"""
struct EquationAttributes
  kind::EquationKind
  differentiated::Bool
  derivative::Union{Nothing, Equation}
end

"The attributes of an equation before classification."
const DEFAULT_ATTRIBUTES = EquationAttributes(EQ_UNKNOWN, false, nothing)

"`lhs = rhs` of scalar type `ty`."
mutable struct SCALAR_EQUATION <: Equation
  lhs::Expression
  rhs::Expression
  ty::ModelicaType
  source::Source
  attributes::EquationAttributes
end

"`lhs = rhs` of array type `ty`."
mutable struct ARRAY_EQUATION <: Equation
  lhs::Expression
  rhs::Expression
  ty::T_ARRAY
  source::Source
  attributes::EquationAttributes
end

"`lhs = rhs` of record type `ty`: `size` equations, one per field (a record or array field counts one)."
mutable struct RECORD_EQUATION <: Equation
  lhs::Expression
  rhs::Expression
  ty::ModelicaType
  size::Int
  source::Source
  attributes::EquationAttributes
end

"""
    ALGORITHM(statements, outputs, source, attributes)

An algorithm section: its statements and the variables it assigns (its unknowns, MLS 11.1.2).
"""
mutable struct ALGORITHM <: Equation
  statements::Vector{Statement}
  outputs::Vector{ComponentRef}
  source::Source
  attributes::EquationAttributes
end

"A condition and the equations that hold when it holds."
struct EquationBranch
  condition::Expression
  equations::Vector{Equation}
end

"`if ... elseif ... else ... end if` equations: the branches in order, then the else-equations."
mutable struct IF_EQUATION <: Equation
  branches::Vector{EquationBranch}
  elseEquations::Vector{Equation}
  source::Source
  attributes::EquationAttributes
end

"`for iterator loop body end for` equations."
mutable struct FOR_EQUATION <: Equation
  iterator::Iterator
  body::Vector{Equation}
  source::Source
  attributes::EquationAttributes
end

"""
A when-equation's role: a regular one; an initial one the passes made: an initial algorithm
section or an algorithm's `when initial()` body (`INITIAL_SECTION`: the initialization's first
pass runs them, before the system is solved), the initial arm of a lifted algorithm or discrete
cluster (`INITIAL_WHEN`); the initialization runs both before the model's own `when initial()`
equations; OM.jl's structural transition (its body changes the model's structure:
`recompilation`, `agentic_recompilation`).
"""
@enum WhenKind REGULAR_WHEN INITIAL_WHEN INITIAL_SECTION STRUCTURAL_WHEN

"""
    WHEN_EQUATION(branches, kind, source, attributes)

`when ... elsewhen ... end when`: the branches in order, each with its triggers and the
statements (assignments, `reinit`, `assert`, `terminate`, calls) executed at its events.
"""
mutable struct WHEN_EQUATION <: Equation
  branches::Vector{WhenBranch}
  kind::WhenKind
  source::Source
  attributes::EquationAttributes
end

"`assert(condition, message, level)` in an equation section."
mutable struct ASSERT_EQUATION <: Equation
  condition::Expression
  message::Expression
  level::Expression
  source::Source
end

#= ==================================================================================== =#
#= Variables                                                                            =#
#= ==================================================================================== =#

"`StateSelect` (MLS 4.8.8.1)."
@enum StateSelect NEVER AVOID DEFAULT PREFER ALWAYS

"The causality of a top-level variable."
@enum Direction NO_DIRECTION INPUT OUTPUT

"The connector prefix of a variable."
@enum ConnectorKind NOT_CONNECTOR POTENTIAL FLOW STREAM

"A continuous-time variable that is not differentiated."
struct ALGEBRAIC <: VariableKind end

"""
    STATE(index, derivative, natural)

A differentiated variable. `index`: how often it is differentiated. `derivative`: the
variable holding its derivative, once the backend introduces it. `natural`: it is a state
because it is differentiated, not forced by StateSelect.
"""
struct STATE <: VariableKind
  index::Int
  derivative::Union{Nothing, Symbol}
  natural::Bool
end

"The derivative `der(state)` as a variable of its own."
struct STATE_DER <: VariableKind
  state::Symbol
end

"A state derivative that index reduction made algebraic: the derivative of `dummyState`."
struct DUMMY_DER <: VariableKind
  dummyState::Symbol
end

"A state that index reduction made algebraic; `dummyDerivative` is its derivative."
struct DUMMY_STATE <: VariableKind
  dummyDerivative::Symbol
end

"A discrete-time variable: it changes at events only."
struct DISCRETE <: VariableKind end

"""
    CLOCKED_STATE(previous, isStartFixed)

A variable of a clocked partition whose `previous` value is used (MLS 16.4): `previous`
names the variable holding it.
"""
struct CLOCKED_STATE <: VariableKind
  previous::Symbol
  isStartFixed::Bool
end

"A variable of type Clock."
struct CLOCK_VARIABLE <: VariableKind end

"A parameter."
struct PARAMETER <: VariableKind end

"A constant."
struct CONSTANT <: VariableKind end

"An external object (MLS 12.9.7), of the class `className`."
struct EXTERNAL_OBJECT <: VariableKind
  className::Symbol
end

"""
    VariableAttributes(start, fixed, nominal, min, max, unit, displayUnit, quantity, stateSelect)

The built-in attributes of a variable (MLS 4.8): an expression for each one given,
`nothing` for one not.
"""
struct VariableAttributes <: Node
  start::Union{Nothing, Expression}
  fixed::Union{Nothing, Expression}
  nominal::Union{Nothing, Expression}
  min::Union{Nothing, Expression}
  max::Union{Nothing, Expression}
  unit::Union{Nothing, Expression}
  displayUnit::Union{Nothing, Expression}
  quantity::Union{Nothing, Expression}
  stateSelect::Union{Nothing, StateSelect}
end

"No attributes given."
const NO_ATTRIBUTES = VariableAttributes(nothing, nothing, nothing, nothing, nothing, nothing,
                                         nothing, nothing, nothing)

"""
    Variable(cref, ty, kind, binding, attributes, direction, isProtected, connector, source, origin)

A variable of the model: its reference (its parts, each with its subscripts and type, as the
flat model names it) and its flat name `name` (`cref.name`, by which it is looked up). Mutable:
the passes change a variable's kind, binding or attributes in place, and every reference to it
sees the change. `origin`: `:model` for one of the flat model, else the pass that introduced it
(NB's auxiliaries: `\$DER.x`, a when-condition's variable, an if-expression's temporary).
"""
mutable struct Variable
  const cref::ComponentRef
  const name::Symbol
  ty::ModelicaType
  kind::VariableKind
  binding::Union{Nothing, Expression}
  attributes::VariableAttributes
  direction::Direction
  isProtected::Bool
  connector::ConnectorKind
  source::Source
  const origin::Symbol
  Variable(cref::ComponentRef, ty::ModelicaType, kind::VariableKind, binding::Union{Nothing, Expression},
           attributes::VariableAttributes, direction::Direction, isProtected::Bool, connector::ConnectorKind,
           source::Source, origin::Symbol) =
    new(cref, cref.name, ty, kind, binding, attributes, direction, isProtected, connector, source, origin)
end

"A variable the backend makes, of one part `name`."
Variable(name::Symbol, ty::ModelicaType, kind::VariableKind, binding::Union{Nothing, Expression},
         attributes::VariableAttributes, direction::Direction, isProtected::Bool, connector::ConnectorKind,
         source::Source, origin::Symbol) =
  Variable(ComponentRef(name, ty), ty, kind, binding, attributes, direction, isProtected, connector, source, origin)

"""
    VariableSet()

Variables in order, looked up by name (NB's VariablePointers).
"""
struct VariableSet
  index::Dict{Symbol, Int}
  variables::Vector{Variable}
end

VariableSet() = VariableSet(Dict{Symbol, Int}(), Variable[])

function VariableSet(variables::Vector{Variable})
  local set = VariableSet()
  for v in variables
    push!(set, v)
  end
  return set
end

Base.length(set::VariableSet)::Int = length(set.variables)
Base.isempty(set::VariableSet)::Bool = isempty(set.variables)
Base.iterate(set::VariableSet) = iterate(set.variables)
Base.iterate(set::VariableSet, state::Int) = iterate(set.variables, state)
Base.eltype(::Type{VariableSet}) = Variable
Base.haskey(set::VariableSet, name::Symbol)::Bool = haskey(set.index, name)
Base.getindex(set::VariableSet, name::Symbol)::Variable = set.variables[set.index[name]]

function Base.get(set::VariableSet, name::Symbol, default::Nothing)::Union{Nothing, Variable}
  local i = get(set.index, name, 0)
  return i == 0 ? default : set.variables[i]
end

"Add `v`; a variable of the same name must not be in `set`."
function Base.push!(set::VariableSet, v::Variable)::VariableSet
  haskey(set.index, v.name) && error("OMJLIR: variable $(v.name) is already in the set")
  push!(set.variables, v)
  set.index[v.name] = length(set.variables)
  return set
end

"Remove all the variables."
function Base.empty!(set::VariableSet)::VariableSet
  empty!(set.variables)
  empty!(set.index)
  return set
end

"Keep the variables for which `f` holds, in their order."
function Base.filter!(f::F, set::VariableSet)::VariableSet where {F}
  filter!(f, set.variables)
  empty!(set.index)
  for (i, v) in enumerate(set.variables)
    set.index[v.name] = i
  end
  return set
end

"The variables of `set` of kind `K`."
function variablesOfKind(set::VariableSet, ::Type{K})::Vector{Variable} where {K <: VariableKind}
  return Variable[v for v in set.variables if v.kind isa K]
end

isState(v::Variable)::Bool = v.kind isa STATE
isParameter(v::Variable)::Bool = v.kind isa PARAMETER
isConstant(v::Variable)::Bool = v.kind isa CONSTANT
isDiscrete(v::Variable)::Bool = v.kind isa DISCRETE || v.kind isa CLOCKED_STATE

"A variable whose value is not computed from the equations: a parameter or a constant."
isKnown(v::Variable)::Bool = v.kind isa PARAMETER || v.kind isa CONSTANT

"The name of the derivative variable of the state `state`: `\$DER.state`."
derivativeName(state::Symbol)::Symbol = Symbol("\$DER.", state)

"The derivative variable of the state `state` in `set`; `nothing` before it is introduced."
function derivativeOf(set::VariableSet, state::Symbol)::Union{Nothing, Variable}
  local k = set[state].kind
  k isa STATE && k.derivative !== nothing || return nothing
  return set[k.derivative]
end

#= ==================================================================================== =#
#= Functions                                                                            =#
#= ==================================================================================== =#

"""
    ExternalDeclaration(name, language, arguments, output, annotation)

The external clause of a function (MLS 12.9): the external function's name and language
(`"C"`, `"FORTRAN 77"`), its arguments (in the function's variables), the variable receiving
its result (`nothing` for a void function), and the clause's annotation (`Include`,
`IncludeDirectory`, `Library`, ... as written, MLS 12.9.4; empty for none).
"""
struct ExternalDeclaration <: Node
  name::String
  language::String
  arguments::Vector{Expression}
  output::Union{Nothing, ComponentRef}
  annotation::String
end

"""
    FunctionVariable(name, ty, binding, fieldBindings)

An input, output or protected variable of a function: its name, its type (an array's
dimensions may be expressions of the inputs, `size(u, 1)`), its binding (an input's or
output's default, a protected variable's value), and for one of a record type the bindings
of its fields as the instance has them (its modifiers, the record's defaults; one reading
another field reads the variable's own: `w.a`).
"""
struct FunctionVariable <: Node
  name::Symbol
  ty::ModelicaType
  binding::Union{Nothing, Expression}
  fieldBindings::Vector{Pair{Symbol, Expression}}
end

"""
    FunctionDerivative(derivative, order, excludedInputs)

An annotation derivative (MLS 12.7.1): the function computing the derivative of order
`order`, and the inputs (by position) zeroDerivative or noDerivative exclude (the derivative
function has no derivative input for them).
"""
struct FunctionDerivative <: Node
  derivative::Symbol
  order::Int
  excludedInputs::Vector{Int}
end

"""
    ModelicaFunction(name, inputs, outputs, locals, body, external, derivatives, impure,
                     inline, source)

A function the model calls: its inputs, outputs and protected variables in declaration
order, its algorithm (empty for an external function), its external clause, its derivative
annotations and its prefixes (`inline`: Inline = true).
"""
struct ModelicaFunction
  name::Symbol
  inputs::Vector{FunctionVariable}
  outputs::Vector{FunctionVariable}
  locals::Vector{FunctionVariable}
  body::Vector{Statement}
  external::Union{Nothing, ExternalDeclaration}
  derivatives::Vector{FunctionDerivative}
  impure::Bool
  inline::Bool
  source::Source
end

#= ==================================================================================== =#
#= Partitions and the model                                                             =#
#= ==================================================================================== =#

"Evaluated in continuous time and at events."
struct CONTINUOUS_PARTITION <: Association end

"""
    SubClock(factor, shift, solverMethod)

A clock derived from a base clock (MLS 16.5.2): ticks at the base clock's ticks multiplied
by `factor` and shifted by `shift` (in base intervals); `solverMethod` for a solver clock.
"""
struct SubClock <: Node
  factor::Rational{Int}
  shift::Rational{Int}
  solverMethod::Union{Nothing, String}
end

"The clock that ticks with its base clock."
const IDENTITY_SUBCLOCK = SubClock(1 // 1, 0 // 1, nothing)

"Evaluated at the ticks of the sub-clock `subClock` of base clock `baseClock`."
struct CLOCKED_PARTITION <: Association
  baseClock::Int
  subClock::SubClock
end

"""
    Adjacency(equationUnknowns, unknownEquations)

The structure of a partition: `equationUnknowns[i]`, the unknowns (indices into the
partition's `unknowns`) occurring in equation `i`; `unknownEquations[j]`, the equations
unknown `j` occurs in.
"""
struct Adjacency
  equationUnknowns::Vector{Vector{Int}}
  unknownEquations::Vector{Vector{Int}}
end

"""
    Matching(unknownEquation, equationUnknown)

Which equation each unknown is solved from and which unknown each equation is solved for;
0 for one unmatched.
"""
struct Matching
  unknownEquation::Vector{Int}
  equationUnknown::Vector{Int}
end

"""
    StrongComponent(equations, unknowns)

Equations solved together for unknowns (indices into the partition's): one equation for one
unknown, or an algebraic loop.
"""
struct StrongComponent
  equations::Vector{Int}
  unknowns::Vector{Int}
end

"""
    Partition(index, association, unknowns, equations, initialEquations)

Equations evaluated together (NB's Partition): a continuous-time system or a clocked
partition (MLS 16.7), the variables they are solved for (in the continuous system: the state
derivatives and the algebraic and discrete variables) and its initial equations.

The structural slots are `nothing` until causalization fills them: `adjacency` over
`equations` and `unknowns`, a `matching`, and the strong `components` in evaluation order.
Index reduction changes `unknowns` and `equations` and computes them again.
"""
mutable struct Partition
  const index::Int
  association::Association
  unknowns::Vector{Symbol}
  equations::Vector{Equation}
  initialEquations::Vector{Equation}
  adjacency::Union{Nothing, Adjacency}
  matching::Union{Nothing, Matching}
  components::Union{Nothing, Vector{StrongComponent}}
end

Partition(index::Int, association::Association, unknowns::Vector{Symbol},
          equations::Vector{Equation}, initialEquations::Vector{Equation}) =
  Partition(index, association, unknowns, equations, initialEquations, nothing, nothing, nothing)

"""
    StructuralMode(name, variables, system)

A mode of an OM.jl model with structural modes: its variables and the system it switches to.
"""
struct StructuralMode
  name::String
  variables::VariableSet
  system::Partition
end

"""
    StructuralTransition(from, to, condition)

OM.jl's explicit structural transition: from mode `from` to mode `to` when `condition`
becomes true.
"""
struct StructuralTransition <: Node
  from::String
  to::String
  condition::Expression
end

"""
    Model(name, variables, continuous, clocked, baseClocks, functions, initialMode, modes,
          transitions, frontendModel)

A model in OMJLIR (NB's BackendDAE MAIN):

- `variables`: all of them; their kinds tell states, derivatives, parameters, ...
- `continuous`: the continuous-time partitions (NB's ode partitions): the equations of the
  continuous-time system and its events, with the initial equations. One partition until
  the system is split into independent subsystems.
- `clocked`: the clocked partitions (MLS 16.7), after clock partitioning.
- `baseClocks`: the base clocks the clocked partitions tick with (`baseClock` indexes it).
- `functions`: the functions the model calls, in the frontend's order.
- `initialMode`, `modes`, `transitions`: OM.jl's structural modes, if the model has them.
- `frontendModel`: what the frontend needs to translate the model again at a structural
  change; the backend does not look into it.
"""
struct Model
  name::String
  variables::VariableSet
  continuous::Vector{Partition}
  clocked::Vector{Partition}
  baseClocks::Vector{ClockKind}
  functions::Vector{ModelicaFunction}
  initialMode::Union{Nothing, String}
  modes::Vector{StructuralMode}
  transitions::Vector{StructuralTransition}
  frontendModel::Any
end

#= ==================================================================================== =#
#= Types of expressions                                                                 =#
#= ==================================================================================== =#

"The type of `e`."
function typeOf(e::Expression)::ModelicaType
  return _typeOf(e)
end
_typeOf(::INTEGER)::ModelicaType = T_INTEGER()
_typeOf(::REAL)::ModelicaType = T_REAL()
_typeOf(::BOOLEAN)::ModelicaType = T_BOOLEAN()
_typeOf(::STRING)::ModelicaType = T_STRING()
_typeOf(e::ENUM_LITERAL)::ModelicaType = e.ty
_typeOf(::CLOCK)::ModelicaType = T_CLOCK()
_typeOf(::WILD)::ModelicaType = T_UNKNOWN()
_typeOf(::LBINARY)::ModelicaType = T_BOOLEAN()
_typeOf(::LUNARY)::ModelicaType = T_BOOLEAN()
_typeOf(::RELATION)::ModelicaType = T_BOOLEAN()
_typeOf(e::SIZE)::ModelicaType =
  e.dimension === nothing ? T_ARRAY(T_INTEGER(), Dimension[DIM_UNKNOWN()]) : T_INTEGER()
_typeOf(e::Expression)::ModelicaType = e.ty

"The number of indices of a dimension; `nothing` for one of unknown size."
dimensionSize(d::DIM_SIZE)::Union{Nothing, Int} = d.size
dimensionSize(d::DIM_ENUM)::Union{Nothing, Int} = length(d.ty.literals)
dimensionSize(::DIM_BOOLEAN)::Union{Nothing, Int} = 2
dimensionSize(::Union{DIM_EXP, DIM_UNKNOWN})::Union{Nothing, Int} = nothing

"""
    dimensionIndex(d, k)

The `k`-th index of the dimension `d` as a subscript expression: `k`, the `k`-th literal of
an enumeration, `false`/`true`.
"""
function dimensionIndex(d::Dimension, k::Int)::Expression
  d isa DIM_ENUM && return ENUM_LITERAL(d.ty, d.ty.literals[k], k)
  d isa DIM_BOOLEAN && return BOOLEAN(k == 2)
  return INTEGER(k)
end

"The element type of an array type; a scalar type itself."
elementType(ty::T_ARRAY)::ModelicaType = ty.elementType
elementType(ty::T_COMPONENTS)::ModelicaType = T_CLASS(ty.cls)
elementType(ty::ModelicaType)::ModelicaType = ty

"The dimensions of an array type; none for a scalar type."
dimensions(ty::T_ARRAY)::Vector{Dimension} = ty.dimensions
dimensions(ty::T_COMPONENTS)::Vector{Dimension} = Dimension[DIM_SIZE(length(ty.members))]
dimensions(::ModelicaType)::Vector{Dimension} = Dimension[]

isArray(ty::ModelicaType)::Bool = ty isa T_ARRAY
isScalar(ty::ModelicaType)::Bool = !(ty isa T_ARRAY)

#= ==================================================================================== =#
#= Traversal                                                                            =#
#= ==================================================================================== =#

#= By `Traversal` (Traversal.jl): what a node's field holds is declared here, per type; every
   traversal follows from it. =#

"Expressions, statements and equations: a traversal's children, handed to its `recur`."
const TraversalChild = Union{Expression, Statement, Equation}

"Iterators, branches, subscripts, reference parts and clocks: they hold children, traversed field by field."
const TraversalContainer = Union{Iterator, StatementBranch, WhenBranch, EquationBranch, Subscript, CrefPart, ClockKind}

#= A component reference holds its subscripts (its own rule below). Anything else (types,
   attributes, an equation's derivative link) is a leaf, kept. =#
Traversal.traverseField(x::TraversalChild, recur::R, arg::A) where {R, A} = recur(x, arg)
Traversal.traverseField(x::TraversalContainer, recur::R, arg::A) where {R, A} = traverseFields(x, recur, arg)
Traversal.traverseField(xs::Vector{<:Union{TraversalChild, TraversalContainer, ComponentRef}}, recur::R, arg::A) where {R, A} =
  traverseElements(xs, recur, arg)

"A component reference: its parts' subscripts traversed, its flat name following them."
function Traversal.traverseField(cref::ComponentRef, recur::R, arg::A) where {R, A}
  hasSubscripts(cref) || return (cref, arg)
  local (parts, a) = traverseElements(cref.parts, recur, arg)
  return (parts === cref.parts ? cref : ComponentRef(parts), a)
end

"""
    mapChildren(g, e)

`e` with `g` applied to each of its direct subexpressions (the subscripts' expressions, the
iterators' ranges and the clocks' arguments included), one level; `e` itself if none changed.
"""
mapChildren(g::G, e::Expression) where {G} = first(traverseFields(e, _Apply(g), nothing))

"`g` as a traversal's `recur`: a child replaced by `g(child)`."
struct _Apply{G}
  g::G
end
(m::_Apply)(x::Expression, a::Nothing) = (m.g(x), a)

"`mapExp` of `f` as a callable, for `mapChildren`."
struct _BottomUp{F}
  f::F
end
(m::_BottomUp)(e::Expression) = mapExp(m.f, e)

"""
    mapExp(f, e)

`e` with every subexpression `s` replaced by `f(s)`, bottom-up: `f` sees a node whose
children are already mapped. The subscripts' expressions are mapped too.
"""
mapExp(f::F, e::Expression) where {F} = f(mapChildren(_BottomUp(f), e))

"`mapExpTopDown` of `f` as a callable, for `mapChildren`."
struct _TopDown{F}
  f::F
end
(m::_TopDown)(e::Expression) = mapExpTopDown(m.f, e)

"""
    mapExpTopDown(f, e)

`e` mapped top-down: `f(e)` is `(e2, descend)`; `e2` replaces `e`, and its children are mapped
in turn if `descend` (a replacement that is final, a value substituted for a variable, is not).
"""
function mapExpTopDown(f::F, e::Expression) where {F}
  local (r, descend) = f(e)
  return descend ? mapChildren(_TopDown(f), r) : r
end

"""
    foreachExp(f, e)

Call `f` on `e` and on every subexpression, top-down (a node before its children).
"""
function foreachExp(f::F, e::Expression)::Nothing where {F}
  f(e)
  traverseFields(e, _VisitChild(f), nothing)
  return nothing
end

"`foreachExp` of `f` as a traversal's `recur`."
struct _VisitChild{F}
  f::F
end
(v::_VisitChild)(x::Expression, a::Nothing) = (foreachExp(v.f, x); (x, a))

"""
    anyExp(p, e)

Whether `p` holds for `e` or for one of its subexpressions; the children after the first that
holds are not visited.
"""
function anyExp(p::F, e::Expression)::Bool where {F}
  p(e) && return true
  return last(traverseFields(e, _Any(p), false))
end

"`anyExp` of `p` as a traversal's `recur`."
struct _Any{F}
  p::F
end
(s::_Any)(x::Expression, found::Bool) = (x, found || anyExp(s.p, x))

"The direct subexpressions of `e`, the subscripts' expressions included, in field order."
children(e::Expression)::Vector{Expression} = last(traverseFields(e, _pushChild, Expression[]))

_pushChild(x::Expression, out::Vector{Expression})::Tuple{Expression, Vector{Expression}} = (push!(out, x); (x, out))

#= Statements and equations =#

"`f` mapped (as by `mapExp`) over a traversal's children: an expression, a statement, an equation (in place)."
struct _MapAll{F}
  f::F
end
(m::_MapAll)(e::Expression, a::Nothing) = (mapExp(m.f, e), a)
(m::_MapAll)(s::Statement, a::Nothing) = (mapStatement(m.f, s), a)
(m::_MapAll)(eq::Equation, a::Nothing) = (mapEquation!(m.f, eq), a)

"""
    mapStatement(f, s)

`s` with `f` mapped (as by `mapExp`) over all its expressions, its targets and the subscripts
of its references included.
"""
mapStatement(f::F, s::Statement) where {F} = first(traverseFields(s, _MapAll(f), nothing))

mapStatements(f::F, stmts::Vector{Statement}) where {F} = Statement[mapStatement(f, s) for s in stmts]

"`cref` with `f` mapped (as by `mapExp`) over its subscripts' expressions."
mapCref(f::F, cref::ComponentRef) where {F} = first(Traversal.traverseField(cref, _MapAll(f), nothing))

"""
    mapEquation!(f, eq)

`eq` with `f` mapped (as by `mapExp`) over all its expressions, in place (equations are
entities); the equations of its branches and bodies too. Returns `eq`.
"""
mapEquation!(f::F, eq::Equation) where {F} = first(traverseFields(eq, _MapAll(f), nothing))::Equation

"""
    foreachRootExp(f, x::Union{Statement, Equation})

Call `f` on each root expression of a statement or an equation (its sides, conditions,
triggers, ranges, arguments, the subscripts of its references) and of the equations and
statements in it, without descending into the expressions: for a visitor that decides itself
where to descend. A reinit's state is a root expression, as a reference.
"""
foreachRootExp(f::F, x::Union{Statement, Equation}) where {F} = (traverseFields(x, _RootVisit(f), nothing); nothing)
foreachRootExp(f::F, s::REINIT) where {F} = (f(CREF(s.state, T_REAL())); f(s.value); nothing)

"`foreachRootExp` of `f` as a traversal's `recur`."
struct _RootVisit{F}
  f::F
end
(v::_RootVisit)(e::Expression, a::Nothing) = (v.f(e); (e, a))
(v::_RootVisit)(x::Union{Statement, Equation}, a::Nothing) = (foreachRootExp(v.f, x); (x, a))

"`foreachExp` of `f` as a callable."
struct _Visit{F}
  f::F
end
(v::_Visit)(e::Expression) = foreachExp(v.f, e)

"""
    foreachExp(f, x::Union{Statement, Equation})

Call `f` on every expression of a statement or an equation (and of the equations and
statements in it), each top-down.
"""
foreachExp(f::F, x::Union{Statement, Equation}) where {F} = foreachRootExp(_Visit(f), x)

"The component references in `e`, in the order of a top-down traversal, with duplicates."
function crefsOf(e::Expression)::Vector{ComponentRef}
  local out = ComponentRef[]
  foreachExp(s -> s isa CREF && push!(out, s.cref), e)
  return out
end

"Whether `e` is a call of the built-in `name` (`:der`, `:pre`, ...)."
isBuiltinCall(e::Expression, name::Symbol)::Bool =
  e isa CALL && e.name === name && e.attributes.builtin

#= ==================================================================================== =#
#= Printing, in Modelica syntax                                                         =#
#= ==================================================================================== =#

"`x` in Modelica syntax."
toString(x::Union{Node, Variable, Equation, EquationBranch, EquationAttributes, Partition})::String = sprint(_print, x)

Base.show(io::IO, x::Node) = _print(io, x)
Base.show(io::IO, v::Variable) = _print(io, v)
Base.show(io::IO, eq::Equation) = _print(io, eq)

const _OP_STRINGS = Dict{Op, String}(
  ADD => "+", SUB => "-", MUL => "*", DIV => "/", POW => "^",
  ADD_EW => ".+", SUB_EW => ".-", MUL_EW => ".*", DIV_EW => "./", POW_EW => ".^",
  ADD_SCALAR_ARRAY => ".+", ADD_ARRAY_SCALAR => ".+", SUB_SCALAR_ARRAY => ".-",
  SUB_ARRAY_SCALAR => ".-", MUL_SCALAR_ARRAY => "*", MUL_ARRAY_SCALAR => "*",
  MUL_VECTOR_MATRIX => "*", MUL_MATRIX_VECTOR => "*", SCALAR_PRODUCT => "*",
  MATRIX_PRODUCT => "*", DIV_SCALAR_ARRAY => "./", DIV_ARRAY_SCALAR => "/",
  POW_SCALAR_ARRAY => ".^", POW_ARRAY_SCALAR => ".^", POW_MATRIX => "^",
  UMINUS => "-", AND => "and", OR => "or", NOT => "not",
  LESS => "<", LESSEQ => "<=", GREATER => ">", GREATEREQ => ">=", EQUAL => "==",
  NEQUAL => "<>")

"The operator `op` in Modelica syntax."
opString(op::Op)::String = _OP_STRINGS[op]

function _printJoined(io::IO, xs::AbstractVector, sep::String = ", ")
  for (i, x) in enumerate(xs)
    i > 1 && print(io, sep)
    _print(io, x)
  end
end

_print(io::IO, ::T_INTEGER) = print(io, "Integer")
_print(io::IO, ::T_REAL) = print(io, "Real")
_print(io::IO, ::T_BOOLEAN) = print(io, "Boolean")
_print(io::IO, ::T_STRING) = print(io, "String")
_print(io::IO, ::T_CLOCK) = print(io, "Clock")
_print(io::IO, t::T_ENUMERATION) = print(io, t.name)
_print(io::IO, t::T_ARRAY) = (_print(io, t.elementType); print(io, "["); _printJoined(io, t.dimensions); print(io, "]"))
_print(io::IO, f::Field) = (_print(io, f.ty); print(io, " ", f.name))
_print(io::IO, t::T_RECORD) = print(io, t.name)
_print(io::IO, t::Union{T_EXTERNAL_OBJECT, T_CLASS}) = print(io, t.name)
_print(io::IO, t::T_COMPONENTS) = print(io, t.cls, "[", length(t.members), "]")
_print(io::IO, t::T_TUPLE) = (print(io, "("); _printJoined(io, t.types); print(io, ")"))
_print(io::IO, t::T_FUNCTION) = print(io, "function ", t.name)
_print(io::IO, ::T_NORETCALL) = print(io, "()")
_print(io::IO, ::T_UNKNOWN) = print(io, "unknown")
_print(io::IO, d::DIM_SIZE) = print(io, d.size)
_print(io::IO, d::DIM_ENUM) = print(io, d.ty.name)
_print(io::IO, ::DIM_BOOLEAN) = print(io, "Boolean")
_print(io::IO, d::DIM_EXP) = _print(io, d.exp)
_print(io::IO, ::DIM_UNKNOWN) = print(io, ":")

_print(io::IO, s::INDEX) = _print(io, s.exp)
_print(io::IO, s::SLICE) = _print(io, s.exp)
_print(io::IO, ::WHOLE) = print(io, ":")

_print(io::IO, cref::ComponentRef) = print(io, cref.name)
_print(io::IO, p::CrefPart) = print(io, flatName(CrefPart[p]))

"""
    flatName(parts)

The flat name of a reference's parts: the names joined by `.`, each with its subscripts in
brackets separated by `,`; a literal index as its value (`2`, `true`, `E.one`), another
expression as written. A part of rolled components (`T_COMPONENTS`) subscripted by a literal
index is named as that member (`cylinder3`).
"""
function flatName(parts::Vector{CrefPart})::Symbol
  length(parts) == 1 && isempty(parts[1].subscripts) && return parts[1].name
  local io = IOBuffer()
  for (i, p) in enumerate(parts)
    i > 1 && print(io, '.')
    if p.ty isa T_COMPONENTS && length(p.subscripts) == 1 && p.subscripts[1] isa INDEX && (p.subscripts[1]::INDEX).exp isa INTEGER
      print(io, p.ty.members[((p.subscripts[1]::INDEX).exp::INTEGER).value])
      continue
    end
    print(io, p.name)
    isempty(p.subscripts) && continue
    print(io, '[')
    for (j, sub) in enumerate(p.subscripts)
      j > 1 && print(io, ',')
      _printSubscriptInName(io, sub)
    end
    print(io, ']')
  end
  return Symbol(String(take!(io)))
end

_printSubscriptInName(io::IO, s::INDEX) =
  s.exp isa INTEGER ? print(io, s.exp.value) : s.exp isa BOOLEAN ? print(io, s.exp.value) : _print(io, s.exp)
_printSubscriptInName(io::IO, s::Subscript) = _print(io, s)

_print(io::IO, e::INTEGER) = print(io, e.value)
_print(io::IO, e::REAL) = print(io, e.value)
_print(io::IO, e::BOOLEAN) = print(io, e.value)
_print(io::IO, e::STRING) = print(io, repr(e.value))
_print(io::IO, e::ENUM_LITERAL) = print(io, e.ty.name, ".", e.name)
_print(io::IO, e::CLOCK) = _print(io, e.kind)
_print(io::IO, e::CREF) = _print(io, e.cref)
_print(io::IO, ::WILD) = print(io, "_")
_print(io::IO, e::ARRAY) = (print(io, "{"); _printJoined(io, e.elements); print(io, "}"))
function _print(io::IO, e::RANGE)
  _print(io, e.start)
  e.step === nothing || (print(io, ":"); _print(io, e.step))
  print(io, ":")
  _print(io, e.stop)
end
_print(io::IO, e::TUPLE) = (print(io, "("); _printJoined(io, e.elements); print(io, ")"))
_print(io::IO, e::RECORD) = (print(io, e.ty.name, "("); _printJoined(io, e.elements); print(io, ")"))
_print(io::IO, e::CALL) = (print(io, e.name, "("); _printJoined(io, e.arguments); print(io, ")"))
_print(io::IO, it::Iterator) = (print(io, it.name, " in "); _print(io, it.range))
function _print(io::IO, e::ARRAY_CONSTRUCTOR)
  print(io, "{")
  _print(io, e.body)
  print(io, " for ")
  _printJoined(io, e.iterators)
  print(io, "}")
end
function _print(io::IO, e::REDUCTION)
  print(io, e.reducer, "(")
  _print(io, e.body)
  print(io, " for ")
  _printJoined(io, e.iterators)
  print(io, ")")
end
function _print(io::IO, e::PARTIAL_APPLICATION)
  print(io, "function ", e.name, "(")
  for (i, (n, a)) in enumerate(zip(e.argumentNames, e.arguments))
    i > 1 && print(io, ", ")
    print(io, n, " = ")
    _print(io, a)
  end
  print(io, ")")
end
function _print(io::IO, e::SIZE)
  print(io, "size(")
  _print(io, e.exp)
  e.dimension === nothing || (print(io, ", "); _print(io, e.dimension))
  print(io, ")")
end
_print(io::IO, e::Union{BINARY, LBINARY, RELATION}) =
  (print(io, "("); _print(io, e.exp1); print(io, " ", opString(e.op), " "); _print(io, e.exp2); print(io, ")"))
_print(io::IO, e::UNARY) = (print(io, "(", opString(e.op)); _print(io, e.exp); print(io, ")"))
_print(io::IO, e::LUNARY) = (print(io, "(not "); _print(io, e.exp); print(io, ")"))
function _print(io::IO, e::IF_EXPRESSION)
  print(io, "(if ")
  _print(io, e.condition)
  print(io, " then ")
  _print(io, e.thenExp)
  print(io, " else ")
  _print(io, e.elseExp)
  print(io, ")")
end
_print(io::IO, e::CAST) = (print(io, "/*"); _print(io, e.ty); print(io, "*/ "); _print(io, e.exp))
_print(io::IO, e::SUBSCRIPTED) = (_print(io, e.exp); print(io, "["); _printJoined(io, e.subscripts); print(io, "]"))
_print(io::IO, e::TUPLE_ELEMENT) = (_print(io, e.exp); print(io, "[", e.index, "]"))
_print(io::IO, e::RECORD_ELEMENT) = (_print(io, e.exp); print(io, ".", e.field))

_print(io::IO, ::INFERRED_CLOCK) = print(io, "Clock()")
_print(io::IO, k::INTEGER_CLOCK) = (print(io, "Clock("); _print(io, k.intervalCounter); print(io, ", "); _print(io, k.resolution); print(io, ")"))
_print(io::IO, k::REAL_CLOCK) = (print(io, "Clock("); _print(io, k.interval); print(io, ")"))
_print(io::IO, k::BOOLEAN_CLOCK) = (print(io, "Clock("); _print(io, k.condition); print(io, ", "); _print(io, k.startInterval); print(io, ")"))
_print(io::IO, k::SOLVER_CLOCK) = (print(io, "Clock("); _print(io, k.clock); print(io, ", "); _print(io, k.solverMethod); print(io, ")"))

_print(io::IO, s::Source) = s === NO_SOURCE ? print(io, "<backend>") : print(io, s.file, ":", s.line, ":", s.column)

function _printStatements(io::IO, body::Vector{Statement}, indent::String)
  for s in body
    print(io, indent)
    _print(io, s, indent)
    print(io, ";\n")
  end
end

_print(io::IO, s::Statement) = _print(io, s, "")
_print(io::IO, s::ASSIGNMENT, ::String) = (_print(io, s.lhs); print(io, " := "); _print(io, s.rhs))
function _printBranches(io::IO, keyword::String, elseKeyword::String, branches::Vector{StatementBranch}, indent::String)
  for (i, b) in enumerate(branches)
    print(io, i == 1 ? keyword : indent * elseKeyword, " ")
    _print(io, b.condition)
    print(io, " then\n")
    _printStatements(io, b.body, indent * "  ")
  end
end

"A when-branch's condition as written: its triggers, `initial()` and `terminal()` among them."
function _printWhenCondition(io::IO, b::WhenBranch)
  local members = Any[b.triggers...]
  b.atInitial && push!(members, "initial()")
  b.atTerminal && push!(members, "terminal()")
  length(members) == 1 && return (members[1] isa String ? print(io, members[1]) : _print(io, members[1]))
  print(io, "{")
  for (i, m) in enumerate(members)
    i > 1 && print(io, ", ")
    m isa String ? print(io, m) : _print(io, m)
  end
  print(io, "}")
end

function _printWhenBranches(io::IO, branches::Vector{WhenBranch}, indent::String)
  for (i, b) in enumerate(branches)
    print(io, i == 1 ? "when" : indent * "elsewhen", " ")
    _printWhenCondition(io, b)
    print(io, " then\n")
    _printStatements(io, b.body, indent * "  ")
  end
end
function _print(io::IO, s::IF_STATEMENT, indent::String)
  _printBranches(io, "if", "elseif", s.branches, indent)
  if !isempty(s.elseBody)
    print(io, indent, "else\n")
    _printStatements(io, s.elseBody, indent * "  ")
  end
  print(io, indent, "end if")
end
function _print(io::IO, s::FOR_STATEMENT, indent::String)
  print(io, "for ")
  _print(io, s.iterator)
  print(io, " loop\n")
  _printStatements(io, s.body, indent * "  ")
  print(io, indent, "end for")
end
function _print(io::IO, s::WHILE_STATEMENT, indent::String)
  print(io, "while ")
  _print(io, s.condition)
  print(io, " loop\n")
  _printStatements(io, s.body, indent * "  ")
  print(io, indent, "end while")
end
_print(io::IO, s::WHEN_STATEMENT, indent::String) =
  (_printWhenBranches(io, s.branches, indent); print(io, indent, "end when"))
_print(io::IO, s::ASSERT, ::String) =
  (print(io, "assert("); _printJoined(io, Expression[s.condition, s.message, s.level]); print(io, ")"))
_print(io::IO, s::TERMINATE, ::String) = (print(io, "terminate("); _print(io, s.message); print(io, ")"))
_print(io::IO, s::REINIT, ::String) = (print(io, "reinit("); _print(io, s.state); print(io, ", "); _print(io, s.value); print(io, ")"))
_print(io::IO, s::NORETCALL, ::String) = _print(io, s.call)
_print(io::IO, ::RETURN, ::String) = print(io, "return")
_print(io::IO, ::BREAK, ::String) = print(io, "break")
_print(io::IO, s::RECOMPILATION, ::String) = (print(io, "recompilation("); _print(io, s.component); print(io, ", "); _print(io, s.newValue); print(io, ")"))
_print(io::IO, s::AGENTIC_RECOMPILATION, ::String) =
  (print(io, "agenticRecompilation("); _printJoined(io, s.components); print(io, ")"))

_print(io::IO, b::StatementBranch) = _printBranches(io, "if", "elseif", [b], "")
_print(io::IO, b::WhenBranch) = _printWhenBranches(io, [b], "")
_print(io::IO, b::EquationBranch) = (print(io, "if "); _print(io, b.condition); print(io, " then ..."))

function _printEquations(io::IO, eqs::Vector{Equation}, indent::String)
  for eq in eqs
    print(io, indent)
    _print(io, eq, indent)
    print(io, ";\n")
  end
end

_print(io::IO, eq::Equation) = _print(io, eq, "")
_print(io::IO, eq::Union{SCALAR_EQUATION, ARRAY_EQUATION, RECORD_EQUATION}, ::String) =
  (_print(io, eq.lhs); print(io, " = "); _print(io, eq.rhs))
function _print(io::IO, eq::ALGORITHM, indent::String)
  print(io, "algorithm\n")
  _printStatements(io, eq.statements, indent * "  ")
  print(io, indent, "end algorithm")
end
function _print(io::IO, eq::IF_EQUATION, indent::String)
  for (i, b) in enumerate(eq.branches)
    print(io, i == 1 ? "if " : indent * "elseif ")
    _print(io, b.condition)
    print(io, " then\n")
    _printEquations(io, b.equations, indent * "  ")
  end
  if !isempty(eq.elseEquations)
    print(io, indent, "else\n")
    _printEquations(io, eq.elseEquations, indent * "  ")
  end
  print(io, indent, "end if")
end
function _print(io::IO, eq::FOR_EQUATION, indent::String)
  print(io, "for ")
  _print(io, eq.iterator)
  print(io, " loop\n")
  _printEquations(io, eq.body, indent * "  ")
  print(io, indent, "end for")
end
function _print(io::IO, eq::WHEN_EQUATION, indent::String)
  eq.kind === REGULAR_WHEN || print(io, "/*", eq.kind, "*/ ")
  _printWhenBranches(io, eq.branches, indent)
  print(io, indent, "end when")
end
_print(io::IO, eq::ASSERT_EQUATION, ::String) =
  (print(io, "assert("); _printJoined(io, Expression[eq.condition, eq.message, eq.level]); print(io, ")"))

_print(io::IO, a::EquationAttributes) =
  print(io, a.kind, a.differentiated ? " differentiated" : "", a.derivative === nothing ? "" : " (has derivative)")

_print(io::IO, ::ALGEBRAIC) = print(io, "algebraic")
_print(io::IO, k::STATE) = print(io, "state(", k.index, k.derivative === nothing ? "" : ", der = $(k.derivative)", ")")
_print(io::IO, k::STATE_DER) = print(io, "der(", k.state, ")")
_print(io::IO, k::DUMMY_DER) = print(io, "dummy der(", k.dummyState, ")")
_print(io::IO, k::DUMMY_STATE) = print(io, "dummy state, der = ", k.dummyDerivative)
_print(io::IO, ::DISCRETE) = print(io, "discrete")
_print(io::IO, k::CLOCKED_STATE) = print(io, "clocked state, previous = ", k.previous)
_print(io::IO, ::CLOCK_VARIABLE) = print(io, "clock")
_print(io::IO, ::PARAMETER) = print(io, "parameter")
_print(io::IO, ::CONSTANT) = print(io, "constant")
_print(io::IO, k::EXTERNAL_OBJECT) = print(io, "external object ", k.className)

function _print(io::IO, a::VariableAttributes)
  local first = true
  for f in fieldnames(VariableAttributes)
    local x = getfield(a, f)
    x === nothing && continue
    print(io, first ? "" : ", ", f, " = ")
    x isa Node ? _print(io, x) : print(io, x)
    first = false
  end
end

function _print(io::IO, v::Variable)
  v.direction === NO_DIRECTION || print(io, lowercase(string(v.direction)), " ")
  v.isProtected && print(io, "protected ")
  v.connector in (FLOW, STREAM) && print(io, lowercase(string(v.connector)), " ")
  print(io, "[")
  _print(io, v.kind)
  print(io, "] ")
  _print(io, v.ty)
  print(io, " ", v.name)
  v.origin === :model || print(io, " /*", v.origin, "*/")
  v.attributes == NO_ATTRIBUTES || (print(io, "("); _print(io, v.attributes); print(io, ")"))
  v.binding === nothing || (print(io, " = "); _print(io, v.binding))
end

_print(io::IO, c::SubClock) = print(io, "subClock(factor = ", c.factor, ", shift = ", c.shift,
                                   c.solverMethod === nothing ? "" : ", solverMethod = $(c.solverMethod)", ")")
_print(io::IO, ::CONTINUOUS_PARTITION) = print(io, "continuous")
_print(io::IO, a::CLOCKED_PARTITION) = (print(io, "clocked by base clock ", a.baseClock, ", "); _print(io, a.subClock))
_print(io::IO, t::StructuralTransition) = (print(io, t.from, " -> ", t.to, " when "); _print(io, t.condition))

"The partition as text: its association, unknowns, equations and initial equations."
function _print(io::IO, p::Partition)
  print(io, "// partition ", p.index, ": ")
  _print(io, p.association)
  println(io, "; unknowns ", join(p.unknowns, ", "))
  println(io, "equation")
  _printEquations(io, p.equations, "  ")
  if !isempty(p.initialEquations)
    println(io, "initial equation")
    _printEquations(io, p.initialEquations, "  ")
  end
end

Base.show(io::IO, p::Partition) = _print(io, p)

"""
    dump(io, model)

The model as text: its variables, its partitions and its structural transitions.
"""
function dump(io::IO, model::Model)
  println(io, "model ", model.name)
  println(io, "  // ", length(model.variables), " variables")
  for v in model.variables
    print(io, "  ")
    _print(io, v)
    println(io, ";")
  end
  for p in Iterators.flatten((model.continuous, model.clocked))
    _print(io, p)
  end
  for t in model.transitions
    print(io, "  // transition ")
    _print(io, t)
    println(io)
  end
  println(io, "end ", model.name, ";")
end

dump(model::Model)::String = sprint(dump, model)

end # module OMJLIR
