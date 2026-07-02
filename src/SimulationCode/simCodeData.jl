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

#=
TODO:
Remove DAE structure from this file s.t simcode can stand alone.

This file holds ONLY data definitions: structs, abstract types, enums, consts,
and trivial same-type constructors. All conversion / projection machinery and
the DAE-wrapping boundary constructors live in `simCodeStructureUtil.jl`,
included at the end.
=#

#= SimCode-native type representation. The Exp variants' `ty` fields hold `SType`;
   the DAE-wrapping boundary constructors (simCodeStructureUtil.jl) accept a
   `DAE.Type` and wrap via `toSimType`. =#
include("simCodeStructureTypes.jl")

#= ---- SimCref ----

   Flat component reference for SimCode and codegen.

   By the time SimCode is built, OMFrontend / BDAE has already flattened
   qualified names (`body.frame_a.r_0` → `body_frame_a_r_0`) and scalarized
   most arrays (each element is its own variable). Yet downstream code
   still wraps every cref in `DAE.CREF_IDENT(name, type, subscripts)`,
   carrying an empty subscript list and a `DAE.Type` that duplicates
   information already held by the `SimVar`. Codegen pattern matchers
   then peel the wrapper back off via `string(cref)` / `_refName` to get
   the flat name they actually wanted.

   `SimCref` is the post-flatten primitive: just a `Symbol` plus an
   optional `Vector{Int}` of integer subscripts for non-scalarized
   record-array fields. Scalar vars (the common case) carry an empty
   subscript vector and behave like a typed `Symbol`. Designed so a
   single `===` on the symbol is enough to compare names, no string
   round-trip needed.
=#

"""
    SimCref(sym, subs = Int[])

Flat component reference used inside SimCode and codegen. The symbol is
the underscore-joined flat name produced by the BDAE flatten pass; the
subscript vector is empty for scalar variables and holds integer
indices for non-scalarized record-array elements.

# Constructors
- `SimCref(sym::Symbol)`                — scalar, no subscripts
- `SimCref(sym::Symbol, subs::AbstractVector)` — explicit subscripts
- `SimCref(name::AbstractString[, subs])`     — convenience over `Symbol(name)`
- `SimCref(cref::DAE.ComponentRef)`            — convert from DAE (in simCodeStructureUtil.jl)
- `SimCref(cref::DAE.CREF)`                    — convert from a wrapping DAE.CREF Exp (idem)
"""
struct SimCref
  sym  :: Symbol
  subs :: Vector{Int}
end

SimCref(sym::Symbol) = SimCref(sym, Int[])
SimCref(name::AbstractString) = SimCref(Symbol(name), Int[])
SimCref(name::AbstractString, subs::AbstractVector{<:Int}) =
  SimCref(Symbol(name), Int[s for s in subs])
SimCref(sym::Symbol, subs::AbstractVector{<:Int}) =
  SimCref(sym, Int[s for s in subs])

#= ---- Expression hierarchy (additive, no migration yet) ----

   SimCode-native `Exp` AST. DAE.Exp carries a lot of frontend-only
   information that SimCode never reads (type-checked subscripts,
   call attributes, etc.); a SimCode-local subset gives each codegen
   target the same closed taxonomy and removes per-backend
   `@match DAE.X` clauses. =#

"""
    OpKind

Closed enum of arithmetic, logical, and relational operator atoms used
inside SIM `BINARY`, `UNARY`, `LBINARY`, `LUNARY`, and `RELATION`. The
operator-side `ty::DAE.Type` field carried by `DAE.Operator` is dropped —
each codegen target derives operator behavior from the operand types
directly, so the operator-side type was redundant.
"""
@enum OpKind begin
  OP_ADD; OP_SUB; OP_MUL; OP_DIV; OP_POW; OP_UMINUS
  OP_AND; OP_OR; OP_NOT
  OP_LESS; OP_LESSEQ; OP_GREATER; OP_GREATEREQ; OP_EQUAL; OP_NEQUAL
end

"""
    Exp

Abstract root for SimCode-native expressions. Closed taxonomy covering
the variants SimCode actually reaches (frequency-distilled from the
existing SIM-side codegen). Less-common DAE.Exp variants (REDUCTION,
MATRIX, SIZE, PARTEVALFUNCTION, …) are not part of this subset; if a
new pass needs them, add the variant here and the boundary converter
clause.
"""
abstract type Exp end

"Literal integer."
struct ICONST <: Exp
  value::Int
end

"Literal floating-point."
struct RCONST <: Exp
  value::Float64
end

"Literal Boolean."
struct BCONST <: Exp
  value::Bool
end

"Literal string."
struct SCONST <: Exp
  value::String
end

"Modelica enumeration literal `Path.Name(index = i)`."
struct ENUM_LITERAL <: Exp
  path::Absyn.Path
  index::Int
end

"Component reference `name[subs...]` with type tag."
struct EXP_CREF <: Exp
  cref::SimCref
  ty::SType
end

"Wildcard `_`: a discarded output slot in a tuple-LHS equation `(x, _) = f()`.
Singleton; codegen emits Julia's `_` discard."
struct WILD <: Exp end

"`e1 op e2` over numeric / array operands."
struct BINARY <: Exp
  exp1::Exp
  op::OpKind
  exp2::Exp
end

"`op e` unary numeric (UMINUS)."
struct UNARY <: Exp
  op::OpKind
  exp::Exp
end

"`e1 op e2` over Boolean operands (AND / OR)."
struct LBINARY <: Exp
  exp1::Exp
  op::OpKind
  exp2::Exp
end

"`op e` over Boolean operand (NOT)."
struct LUNARY <: Exp
  op::OpKind
  exp::Exp
end

"`e1 op e2` over numeric operands producing a Boolean (LESS, EQUAL, …).
`index` is the zero-crossing slot for event-aware codegen."
struct RELATION <: Exp
  exp1::Exp
  op::OpKind
  exp2::Exp
  index::Int
end

"`if cond then e1 else e2`."
struct IFEXP <: Exp
  cond::Exp
  thenExp::Exp
  elseExp::Exp
end

"`{e1, e2, ...}` array literal."
struct ARRAY_EXP <: Exp
  ty::SType
  scalar::Bool
  elements::Vector{Exp}
end

"Array subscript `e[s1, s2, …]`."
struct ASUB <: Exp
  exp::Exp
  subs::Vector{Exp}
end

"Tuple subscript `e.i` (1-based)."
struct TSUB <: Exp
  exp::Exp
  index::Int
  ty::SType
end

"Record field subscript `e.fieldName`."
struct RSUB <: Exp
  exp::Exp
  index::Int
  fieldName::String
  ty::SType
end

"Modelica function call `path(args...)`. Call attributes (builtin /
ext / etc.) are preserved via `attr` because the codegen targets care
about them; the args list is the SimCode-recursive part."
struct CALL <: Exp
  path::Absyn.Path
  args::Vector{Exp}
  attr::DAE.CallAttributes
end

"Type cast `(ty) e`."
struct CAST <: Exp
  ty::SType
  exp::Exp
end

"Record literal `Path(field = e, …)`."
struct RECORD <: Exp
  path::Absyn.Path
  exps::Vector{Exp}
  fieldNames::Vector{String}
  ty::SType
end

"`(e1, e2, …)` tuple, e.g. multi-return call LHS."
struct TUPLE <: Exp
  PR::Vector{Exp}
end

"Reduction / array comprehension `op(body for it in range, …)` (sum/product/
array/min/max). Only the `body` is the SimCode-recursive part; `info`
(DAE.ReductionInfo) and `iterators` (DAE.ReductionIterator list) are kept
opaque for faithful round-trip back to DAE.REDUCTION, which the codegen
consumes directly via toDAEExp."
struct REDUCTION <: Exp
  info::Any
  body::Exp
  iterators::Any
end

#= ---- Equation hierarchy ----

   SimCode-native equation types. The goal is to give SimCode its own
   closed sum type whose variants name the *semantic* categories codegen
   cares about, leaving syntactic shape to the upstream BDAE pass.

   Multi-target design: codegen for MTK, DifferentialEquations.jl, and
   plain C all need the SAME semantic categories — "this is a residual",
   "this is a when-equation", "this is an algorithm section". =#

"""
    EQ_ATTR(; differentiated = false)

Opaque metadata bundle carried by every `Equation`. Today the only
field codegen ever reads is `differentiated::Bool` (set during index
reduction).
"""
struct EQ_ATTR
  differentiated::Bool
end

EQ_ATTR(; differentiated::Bool = false) = EQ_ATTR(differentiated)

const EQ_ATTR_DEFAULT = EQ_ATTR(false)

"""
    Equation

Abstract root for SimCode-native equation kinds. Closed taxonomy:

- `RESIDUAL_EQUATION{rhs}`              — `0 = rhs`. The bulk of the residual system.
- `ALGORITHM{alg}`           — Modelica `algorithm` section.
- `WHEN_EQUATION{trigger, body}`      — runtime `when` clause; emits a callback.
- `INITIAL_WHEN_EQUATION{trigger, body}` — `when initial() then ... end when`.
- `EQUATION{lhs, rhs}`       — generic explicit `lhs = rhs`. Mostly initial.
- `ARRAY_EQUATION{dims, lhs, rhs}`    — array-form equation `lhs[..] = rhs[..]`.

`SimIfEq` is intentionally not a variant here: lifted Modelica
`if`-equations live in the parallel `Construct` hierarchy (`IF_EQUATION` /
`BRANCH`) because they carry their own matching/SCC structure.
"""
abstract type Equation end

"""
    RESIDUAL_EQUATION(exp; source = DAE.emptyElementSource, attr = EQ_ATTR_DEFAULT)

Residual equation `0 = exp`. `source` is the original Modelica source
location. `attr` is the opaque metadata bundle. The DAE.Exp-wrapping
boundary constructors live in `simCodeStructureUtil.jl`.
"""
struct RESIDUAL_EQUATION <: Equation
  exp::Exp
  source::DAE.ElementSource
  attr::EQ_ATTR
end

"""
    ALGORITHM(size, alg, source, expand, attr)

Modelica `algorithm` section. Side-effecting statements; emitted by
backends as a sequence of assignments inside a generated function.
"""
struct ALGORITHM <: Equation
  size::Int
  alg::DAE.Algorithm
  source::DAE.ElementSource
  expand::DAE.Expand
  attr::EQ_ATTR
end

#= ---- SIM-native when-operator hierarchy ----

   SimCode owns its own `when`-body taxonomy. BDAE.WhenOperator and
   BDAE.WHEN_STMTS legitimately exist on the producer side; SimCode
   converts at the boundary (simulationCodeTransformation.jl). The
   converters and DAE-wrapping constructors live in simCodeStructureUtil.jl. =#

"""
    WhenOperator

Abstract root for `when`-body statements after they enter SimCode.
Variants mirror `BDAE.WhenOperator` 1:1 so the boundary converter is a
straightforward field copy.
"""
abstract type WhenOperator end

struct ASSIGN <: WhenOperator
  left::Exp
  right::Exp
  source::DAE.ElementSource
end

struct REINIT <: WhenOperator
  stateVar::DAE.CREF
  value::Exp
  source::DAE.ElementSource
end

struct ASSERT <: WhenOperator
  condition::Exp
  message::Exp
  level::Exp
  source::DAE.ElementSource
end

struct TERMINATE <: WhenOperator
  message::Exp
  source::DAE.ElementSource
end

struct NORETCALL <: WhenOperator
  exp::Exp
  source::DAE.ElementSource
end

struct RECOMPILATION <: WhenOperator
  componentToChange::DAE.CREF
  newValue::Exp
end

struct AGENTIC_RECOMPILATION <: WhenOperator
  componentsToChange::Vector{DAE.CREF}
  prompt::Option
  initialEquations::Option
end

"""
    WHEN_STMTS(condition, whenStmtLst, elsewhenPart)

SimCode counterpart of `BDAE.WHEN_STMTS`. `elsewhenPart` is `nothing`
when there is no `elsewhen` arm, otherwise the nested `WHEN_STMTS`.
"""
struct WHEN_STMTS
  condition::Exp
  whenStmtLst::Vector{WhenOperator}
  elsewhenPart::Union{Nothing, WHEN_STMTS}
end

"""
    WHEN_EQUATION(size, whenEquation, source, attr)

Runtime `when` clause with a SimCode-native `WHEN_STMTS` body.
"""
struct WHEN_EQUATION <: Equation
  size::Int
  whenEquation::WHEN_STMTS
  source::DAE.ElementSource
  attr::EQ_ATTR
end

"""
    INITIAL_WHEN_EQUATION(size, whenEquation, source, attr)

`when initial() then ... end when` clause — runs once at init, never at
runtime. Same shape as `WHEN_EQUATION`, distinct type so init-time
and runtime when-bodies stay separated through codegen.
"""
struct INITIAL_WHEN_EQUATION <: Equation
  size::Int
  whenEquation::WHEN_STMTS
  source::DAE.ElementSource
  attr::EQ_ATTR
end

"""
    INLINE_IF_EQUATION(conditions, branchesTrue, branchElse, source, attr)

Raw Modelica `if`-equation appearing inline in a residual / initial /
shared equation list — NOT yet lifted to the event + matched-BRANCH
hierarchy used by `simCode.ifEquations`.

# Fields
- `conditions`   : `Vector{Exp}` — one per `elseif` arm (no trailing else).
- `branchesTrue` : `Vector{Vector{Equation}}` — equation body per arm.
- `branchElse`   : `Vector{Equation}` — `else` arm body (empty if no else).
- `source`       : `DAE.ElementSource` — source-location decoration.
- `attr`         : `EQ_ATTR` — opaque equation-attribute bundle.
"""
struct INLINE_IF_EQUATION <: Equation
  conditions   :: Vector{Exp}
  branchesTrue :: Vector{Vector{Equation}}
  branchElse   :: Vector{Equation}
  source       :: DAE.ElementSource
  attr         :: EQ_ATTR
end

"""
    EQUATION(lhs, rhs, source, attr)

Generic `lhs = rhs` equation. Mostly used for initial equations that
have not been residualized yet.
"""
struct EQUATION <: Equation
  lhs::Exp
  rhs::Exp
  source::DAE.ElementSource
  attr::EQ_ATTR
end

"""
    ARRAY_EQUATION(dimSize, left, right, source, attr)

Array-form equation `left[..] = right[..]`. Emitted by frontend for
non-scalarized array operations.
"""
struct ARRAY_EQUATION <: Equation
  dimSize::Vector{Int64}
  left::Exp
  right::Exp
  source::DAE.ElementSource
  attr::EQ_ATTR
end

"""
 Category of a simulation variable
"""
abstract type SimVarType end

"""
State variable
"""
struct  STATE <: SimVarType
end

"""
 State Derivative
"""
struct  STATE_DERIVATIVE <: SimVarType
  varName::String
end

"""
Algebraic variable.
Has a sort idx for backend algorithms,
this is different from the final idx used in code generation.
The purpose of the sort index is for initial structural analysis before code generation.
"""
struct ALG_VARIABLE <: SimVarType
  sortIdx::Int
end

"""
Input variable
"""
struct  INPUT <: SimVarType end

"
  A special state variable, used for dynamic overconstrained connectors.
  In pratice this variable is treated as state.
"
struct OCC_VARIABLE <: SimVarType end

struct STRING <: SimVarType
  bindExp::Option{Exp}
end

"""
  A Data structure type.
  Currently this type represents complex data structures such as matrices
  or pointers to data structures in memory.
"""
struct DATA_STRUCTURE <: SimVarType
  bindExp::Option{Exp}
end

"""
An array with N dimensions.
Contains optional binding expression for compile-time evaluation of subscripted access.
"""
struct ARRAY <: SimVarType
  dimensions::Vector{Int}
  bindExp::Option{Exp}
end

#= Backwards-compatible constructor =#
ARRAY(dimensions::Vector{Int}) = ARRAY(dimensions, NONE())

"""
Parameter variable
"""
struct PARAMETER <: SimVarType
  bindExp::Option{Exp}
end

"""
Array parameter variable. Has dimensions and an optional binding expression.
Distinct from ARRAY (which is a state array) to ensure correct MTK code generation.
"""
struct ARRAY_PARAMETER <: SimVarType
  dimensions::Vector{Int}
  bindExp::Option{Exp}
end

"""
  Discrete Variable
"""
struct DISCRETE <: SimVarType end

"""
Abstract type for a simulation variable
"""
abstract type SimVar end

const ELSE_BRANCH = -1

"""
Variable data type used for code generation
"""
struct SIMVAR{T0 <: String, T1 <: Option{Int}, T2 <: SimVarType, T3 <: Option{<:DAE.VariableAttributes}} <: SimVar
  "Readable name of variable"
  name::T0
  "Index of variable, 0 based, type based"
  index::T1
  "Kind of variable, one of SimulationCode.SimVarType"
  varKind::T2
  "Variable attributes. Same as in DAE"
  attributes::T3
end

""" Abstract type for different variants of simulation code """
abstract type SimCode end

"""
  Abstract type for control flow constructs for simulation code
"""
abstract type Construct end


include("simCodeAlgorithmic.jl")


"""
  Represents a branch in simulation code.
  Contains a single condition, a set of inner equations and a set of possible targets.
  Since each branch potentially contains a set of equations information exist so that the code in each branch can be
  matched (Similar to the larger system )
"""
struct BRANCH{T1 <: Exp,
              T2 <: Vector{RESIDUAL_EQUATION},
              T3 <: Int, #= Integer code Each branch has one target (next) The ID of one branch is target - 1=#
              T4 <: Bool,
              T5 <: Vector{Int},
              T6 <: Graphs.AbstractGraph,
              T7 <: Vector{Vector{Int}},
              T8 <: AbstractDict{String, Tuple{Int, SimVar}}} <: Construct

  condition::T1
  residualEquations::T2
  identifier::T3 #= A value of -1 indicates that this branch is an else branch =#
  targets::T3
  isSingular::T4
  matchOrder::T5
  equationGraph::T6
  sccs::T7
  stringToSimVarHT::T8
end

"""
  A representation of a simcode IF Equation.
  Similar to the main simcode module it contains information to make construct a causal representation easier.
"""
struct IF_EQUATION{Branches <: Vector{BRANCH}} <: Construct
  branches::Branches
end

abstract type StructuralTransition end

"""
    EXPLICIT_STRUCTURAL_TRANSITION(fromState, toState, transitionCondition)

Explicit transition between two named structural submodels. Fires the
`transitionCondition` and recompiles into `toState`.
"""
struct EXPLICIT_STRUCTURAL_TRANSITION <: StructuralTransition
  fromState::String
  toState::String
  transitionCondition::Exp
end

"""
    IMPLICIT_STRUCTURAL_TRANSITION(size, whenEquation, source, attr)

Implicit structural transition embedded in a `when` clause; the final
state is not known until the when-body executes. Body is a SimCode-native
`WHEN_STMTS`.
"""
struct IMPLICIT_STRUCTURAL_TRANSITION <: StructuralTransition
  size::Int
  whenEquation::WHEN_STMTS
  source::DAE.ElementSource
  attr::EQ_ATTR
end

"""
    DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION(ifEquation)

DOCC if-equation preserved as a structural switch over connector
topologies. The body is a Frontend `EQUATION_IF`, which describes the
connector branches before BDAE lowering.
"""
struct DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION <: StructuralTransition
  ifEquation::OMFrontend.Frontend.EQUATION_IF
end

"""
    EliminationOptions(; reachability=true, patterns=Regex[], keepPatterns=Regex[])

Options for non-dynamic variable elimination. Controls which variables and equations
are removed from the system before MTK code generation.

When passed to `OM.translate` or `OM.simulate` via the `eliminateNonDynamic` keyword,
variables that do not influence the dynamic states are removed, reducing ODEProblem
compilation time. The eliminated equations are tracked in the SIM_CODE for later
reconstruction (e.g., 3D visualization).

# Fields
- `reachability::Bool` -- eliminate variables not reachable from state derivatives
  via backward dependency analysis. Default: `true`.
- `patterns::Vector{Regex}` -- additionally eliminate variables whose names match
  any of these patterns (e.g., `r"^world_"`). Reserved for future use.
- `keepPatterns::Vector{Regex}` -- force-keep variables matching any of these patterns,
  overriding both reachability and pattern elimination. Reserved for future use.

# Examples
```julia
# Eliminate all non-dynamic variables (default when passing true)
OM.translate(...; eliminateNonDynamic=true)

# Fine-grained control
OM.translate(...; eliminateNonDynamic=EliminationOptions(keepPatterns=[r"revolute"]))
```
"""
struct EliminationOptions
  reachability::Bool
  patterns::Vector{Regex}
  keepPatterns::Vector{Regex}
end

EliminationOptions(; reachability::Bool = true,
                     patterns::Vector{Regex} = Regex[],
                     keepPatterns::Vector{Regex} = Regex[]) =
  EliminationOptions(reachability, patterns, keepPatterns)

"""
Represents an alias relationship discovered during alias elimination.
`eliminatedVar = representativeVar` when `negated` is false,
`eliminatedVar = -representativeVar` when `negated` is true.
"""
struct AliasEntry
  eliminatedName::String
  representativeName::String
  negated::Bool
end

"""
  Root data structure containing information required for code generation to
  generate simulation code for a Modelica model.

The topmost model of a system consisting of several sub models lacks:
  - matchorder
  - The graph of the equations
  - Descriptions of the SCC
  - Information if it is singular or not.
This information is instead contained for each of the structural submodels, where one model is active at the time.
"""
struct SIM_CODE{T0<:String,
                T1<:AbstractDict{String, Tuple{Int, SimVar}},
                T2<:Vector{RESIDUAL_EQUATION},
                T22,
                T4<:Vector{WHEN_EQUATION},
                #=
                  If equations are represented via a vector of possible branches in which the code can operate.
                  Similar to basic blocks
                =#
                T5<:Vector{IF_EQUATION},
                T6<:Bool,
                T7<:Vector{Int},
                T8<:Graphs.AbstractGraph,
                T9<:Vector,
                T10 <: Vector{StructuralTransition},
                T11 <: Vector,
                T12 <: Vector{String},
                T13 <: String} <: SimCode
  name::T0
  "Mapping of names to the corresponding variable"
  stringToSimVarHT::T1
  "Different equations stored within simulation code"
  residualEquations::T2
  "The Initial equations"
  initialEquations::T22
  "When equations"
  whenEquations::T4
  "If Equations (Simulation code branches). Each branch contains a condition a set of residual equations and a set of targets"
  ifEquations::T5
  "True if the system that we are solving is singular"
  isSingular::T6
  "
   The match order:
   Result of assign array, e.g array(j) = equation_i
  "
  matchOrder::T7
    "
    The merged graph. E.g digraph constructed from matching info.
    The indices are the same as above and they are shared.
    If the system is singular tearing is needed.
   "
  equationGraph::T8
  " The reverse topological sort of the equation-graph "
  stronglyConnectedComponents::T9
  "Contains all structural transitions"
  structuralTransitions::T10
  "Structural submodels"
  subModels::T11
  " Variables that different submodels have in common"
  sharedVariables::T12
  "Top variables"
  topVariables::T12
  "Shared equations. These are equations shared between structural submodels. These are required to be residuals."
  sharedEquations::Vector{Equation}
  "Initial model"
  activeModel::T13
  "The MetaModel. That is a reference from the model to a higher order representation of the model itself."
  metaModel::Option
  "An alternate flat model. Used by structural if equations to add or remove connector statements affecting the virtual connection graph."
  flatModel::Option
  "Irreductable variables. That is the names of variables that are involved in events such as discrete variables"
  irreducibleVariables::T12
  "Modelica functions"
  functions::Vector{ModelicaFunction}
  "Specify if an external Modelica runtime is needed or not. Used for build in functions"
  externalRuntime::Bool
  "Equations eliminated by non-dynamic variable elimination (tracked for later reconstruction)"
  eliminatedEquations::Vector{RESIDUAL_EQUATION}
  "Names of variables eliminated by non-dynamic variable elimination"
  eliminatedVariables::Vector{String}
  "Alias map: eliminated alias variables and their representative (for observed equation reconstruction)"
  aliasMap::Vector{AliasEntry}
  "Filter patterns for observed equations. Only alias/observed variables matching these patterns are kept. Nothing means keep all."
  observedFilter::Union{Nothing, Vector{String}}
  "Initial-algorithm bodies lowered from `when initial() then ... end when` clauses; run once during init, never as runtime callbacks."
  initialAlgorithms::Vector{INITIAL_ALGORITHM}
end

#= Conversion / projection machinery + DAE-wrapping boundary constructors. =#
include("simCodeStructureUtil.jl")
