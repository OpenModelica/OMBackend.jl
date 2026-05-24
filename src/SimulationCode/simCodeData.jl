#=
TODO:
Remove DAE structure from this file s.t simcode can stand alone.
=#

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

   This spike is additive — `SimCref` is not yet used by any pass.
   Migration of `stringToSimVarHT` and the codegen-internal sets to
   `SimCref` is a separate, larger step.
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
- `SimCref(cref::DAE.ComponentRef)`            — convert from DAE
- `SimCref(cref::DAE.CREF)`                    — convert from a wrapping DAE.CREF Exp
"""
struct SimCref
  sym  :: Symbol
  subs :: Vector{Int}
end

SimCref(sym::Symbol) = SimCref(sym, Int[])
SimCref(name::AbstractString) = SimCref(Symbol(name), Int[])
SimCref(name::AbstractString, subs::AbstractVector{<:Integer}) =
  SimCref(Symbol(name), Int[s for s in subs])
SimCref(sym::Symbol, subs::AbstractVector{<:Integer}) =
  SimCref(sym, Int[s for s in subs])

"""
    _extractIntSubscripts(subs) -> Vector{Int}

Convert a `DAE.Subscript` list (post-scalarization) to a vector of plain
integer indices. Walks `INDEX(ICONST(i))` and `INDEX(IConst-ish)` shapes.
`WHOLEDIM`, `SLICE`, and non-constant `INDEX(expr)` subscripts are
ignored (returned as no subscript), which matches the post-scalarization
invariant that everything reachable from SimCode has constant
subscripts.
"""
function _extractIntSubscripts(@nospecialize(subs))::Vector{Int}
  local out = Int[]
  for s in subs
    @match s begin
      DAE.INDEX(exp) => begin
        @match exp begin
          DAE.ICONST(i) => push!(out, Int(i))
          _ => nothing
        end
      end
      _ => nothing
    end
  end
  return out
end

"""
    _flattenQualifiedName(cref::DAE.ComponentRef) -> String

Walk a `CREF_QUAL` chain and produce the underscore-joined flat name.
Defensive helper: by the time SimCode is built, `Causalize.flattenArrayCrefs`
should have already collapsed every qualified cref to `CREF_IDENT`, so
this path is only taken if a qualified cref leaked through.
"""
function _flattenQualifiedName(@nospecialize(cref))::String
  local parts = String[]
  local cur = cref
  while true
    @match cur begin
      DAE.CREF_QUAL(ident = id, componentRef = child) => begin
        push!(parts, id)
        cur = child
      end
      DAE.CREF_IDENT(ident = id) => begin
        push!(parts, id)
        break
      end
      DAE.CREF_ITER(ident = id) => begin
        push!(parts, id)
        break
      end
      _ => break
    end
  end
  return join(parts, "_")
end

function SimCref(@nospecialize(cref::DAE.ComponentRef))::SimCref
  @match cref begin
    DAE.CREF_IDENT(ident = id, subscriptLst = subs) =>
      SimCref(Symbol(id), _extractIntSubscripts(subs))
    DAE.CREF_ITER(ident = id, subscriptLst = subs) =>
      SimCref(Symbol(id), _extractIntSubscripts(subs))
    DAE.CREF_QUAL(__) =>
      SimCref(Symbol(_flattenQualifiedName(cref)), Int[])
    DAE.WILD(__) =>
      error("SimCref(DAE.WILD): wildcards are not valid SimCode references")
    _ =>
      error("SimCref: unsupported ComponentRef variant $(typeof(cref))")
  end
end

SimCref(daeCref::DAE.CREF) = SimCref(daeCref.componentRef)

"""
    toDAECref(c::SimCref) -> DAE.CREF

Round-trip back to a wrapping `DAE.CREF` Exp. The original `DAE.Type`
information is lost in the SimCref direction (it lives on the SimVar
now), so this reconstruction uses `DAE.T_REAL_DEFAULT` as the placeholder
type. Subscripts are emitted as `DAE.INDEX(DAE.ICONST(i))`.

Use sparingly — once consumer migration is complete, the only place that
should still need this is the OMC interop boundary
(`OMFrontend.toFlatModelica` and friends). Not implemented as a method
on `DAE.CREF` itself because that would be type piracy (the constructor
lives in the DAE module, not here) and would break Julia precompilation.
"""
function toDAECref(c::SimCref)
  local subList = isempty(c.subs) ?
    MetaModelica.nil :
    MetaModelica.list((DAE.INDEX(DAE.ICONST(i)) for i in c.subs)...)
  return DAE.CREF(DAE.CREF_IDENT(string(c.sym), DAE.T_REAL_DEFAULT, subList),
                  DAE.T_REAL_DEFAULT)
end

#= Set / Dict / display integration. Equality is just symbol identity +
   subscript-vector equality; the symbol comparison is `===` (interned),
   so scalar SimCrefs are zero-allocation to hash and compare. =#
Base.:(==)(a::SimCref, b::SimCref) = a.sym === b.sym && a.subs == b.subs
Base.hash(c::SimCref, h::UInt) = hash(c.sym, hash(c.subs, h))
Base.show(io::IO, c::SimCref) =
  isempty(c.subs) ? print(io, c.sym) : print(io, c.sym, '[', join(c.subs, ", "), ']')
Base.string(c::SimCref) = sprint(show, c)

#= ---- Equation hierarchy ----

   SimCode-native equation types. Today every equation field on
   `SIM_CODE` is `Vector{BDAE.X}`, which leaks the BDAE layer's
   syntactic taxonomy into every downstream codegen. The goal is to
   give SimCode its own closed sum type whose variants name the
   *semantic* categories codegen cares about, leaving syntactic shape
   (operator nesting, line decoration, ...) to the upstream BDAE pass.

   The expression atom remains `DAE.Exp` for now — replacing it with a
   `SimExp` type is a separate, much larger refactor that touches every
   expression node. The boundary is: Equation owns the structure;
   DAE.Exp owns the leaf expressions. Once SimCode has migrated all its
   equation fields to Equation, the `import ..Backend.BDAE` line in
   `SimulationCode.jl` should disappear (only `import DAE` remains).

   Multi-target design: codegen for MTK, DifferentialEquations.jl, and
   plain C all need the SAME semantic categories — "this is a residual",
   "this is a when-equation", "this is an algorithm section". Lifting
   Equation here gives each future emitter the same input shape and
   removes per-backend `@match BDAE.X` clauses.
=#

"""
    EQ_ATTR(; differentiated = false)

Opaque metadata bundle carried by every `Equation`. Today the only
field codegen ever reads is `differentiated::Bool` (set during index
reduction). Replaces the larger `BDAE.EquationAttributes`, which also
carried `kind::EquationKind` and `evalStages::EvaluationStages` —
neither of those is observed by any codegen path so they are dropped.
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

Residual equation `0 = exp`. `exp` is a `DAE.Exp`. `source` is the
original Modelica source location (for diagnostics). `attr` is the
opaque metadata bundle.
"""
struct RESIDUAL_EQUATION <: Equation
  exp::DAE.Exp
  source::DAE.ElementSource
  attr::EQ_ATTR
end

RESIDUAL_EQUATION(exp::DAE.Exp;
            source::DAE.ElementSource = DAE.emptyElementSource,
            attr::EQ_ATTR = EQ_ATTR_DEFAULT) =
  RESIDUAL_EQUATION(exp, source, attr)

#= Legacy compatibility: a single `(exp, nothing, nothing)` form so the
   handful of `BDAE.RESIDUAL_EQUATION(exp, nothing, nothing)` call sites
   could rename to `RESIDUAL_EQUATION(exp, nothing, nothing)`.
   Do NOT add a fully-untyped `(::Any, ::Any, ::Any)` overload — Julia
   auto-generates one from the struct fields, and a duplicate raises
   "Method overwriting is not permitted during Module precompilation". =#
RESIDUAL_EQUATION(exp::DAE.Exp, src::Nothing, attr::Nothing) =
  RESIDUAL_EQUATION(exp, DAE.emptyElementSource, EQ_ATTR_DEFAULT)

#= Convert lets the auto-generated outer constructor accept legacy attr
   types (e.g. BDAE.EquationAttributes) without a per-call-site cast:
   `RESIDUAL_EQUATION(exp, src, bdaeAttr)` works because Julia tries
   `convert(EQ_ATTR, bdaeAttr)` when the field type doesn't match. =#
Base.convert(::Type{EQ_ATTR}, a::EQ_ATTR) = a
Base.convert(::Type{EQ_ATTR}, a) = toEqAttr(a)

"""
    ALGORITHM(size, alg, source, expand, attr)

Modelica `algorithm` section. Side-effecting statements; emitted by
backends as a sequence of assignments inside a generated function.
"""
struct ALGORITHM <: Equation
  size::Integer
  alg::DAE.Algorithm
  source::DAE.ElementSource
  expand::DAE.Expand
  attr::EQ_ATTR
end

#= ---- SIM-native when-operator hierarchy ----

   SimCode owns its own `when`-body taxonomy. BDAE.WhenOperator and
   BDAE.WHEN_STMTS legitimately exist on the producer side (BDAECreate.jl
   and the BDAE pass machinery); SimCode converts at the boundary
   (simulationCodeTransformation.jl). =#

"""
    WhenOperator

Abstract root for `when`-body statements after they enter SimCode.
Variants mirror `BDAE.WhenOperator` 1:1 so the boundary converter is a
straightforward field copy.
"""
abstract type WhenOperator end

struct ASSIGN <: WhenOperator
  left::DAE.Exp
  right::DAE.Exp
  source::DAE.ElementSource
end

struct REINIT <: WhenOperator
  stateVar::DAE.CREF
  value::DAE.Exp
  source::DAE.ElementSource
end

struct ASSERT <: WhenOperator
  condition::DAE.Exp
  message::DAE.Exp
  level::DAE.Exp
  source::DAE.ElementSource
end

struct TERMINATE <: WhenOperator
  message::DAE.Exp
  source::DAE.ElementSource
end

struct NORETCALL <: WhenOperator
  exp::DAE.Exp
  source::DAE.ElementSource
end

struct RECOMPILATION <: WhenOperator
  componentToChange::DAE.CREF
  newValue::DAE.Exp
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
  condition::DAE.Exp
  whenStmtLst::Vector{WhenOperator}
  elsewhenPart::Union{Nothing, WHEN_STMTS}
end

"""
    toWhenOperator(op) -> WhenOperator

Project a `BDAE.WhenOperator` to its SimCode counterpart. Idempotent.
"""
toWhenOperator(op::WhenOperator)::WhenOperator = op

function toWhenOperator(@nospecialize(op))::WhenOperator
  @match op begin
    BDAE.ASSIGN(left = l, right = r, source = s) => ASSIGN(l, r, s)
    BDAE.REINIT(stateVar = sv, value = v, source = s) => REINIT(sv, v, s)
    BDAE.ASSERT(condition = c, message = m, level = lv, source = s) => ASSERT(c, m, lv, s)
    BDAE.TERMINATE(message = m, source = s) => TERMINATE(m, s)
    BDAE.NORETCALL(exp = e, source = s) => NORETCALL(e, s)
    BDAE.RECOMPILATION(componentToChange = c, newValue = v) => RECOMPILATION(c, v)
    BDAE.AGENTIC_RECOMPILATION(componentsToChange = cs, prompt = p, initialEquations = ie) =>
      AGENTIC_RECOMPILATION(cs, p, ie)
    _ => error("toWhenOperator: unsupported variant $(typeof(op))")
  end
end

"""
    toWhenStmts(we) -> WHEN_STMTS

Project a `BDAE.WHEN_STMTS` (the sole `BDAE.WhenEquation` variant) to a
SimCode-native `WHEN_STMTS`. Recursively converts the `elsewhenPart`.
Idempotent on `WHEN_STMTS`.
"""
toWhenStmts(we::WHEN_STMTS)::WHEN_STMTS = we

function toWhenStmts(@nospecialize(we))::WHEN_STMTS
  #= BDAE wraps elsewhen as `SOME(WHEN_EQUATION(WHEN_STMTS(...)))` rather than
     `SOME(WHEN_STMTS(...))`. Unwrap the outer WHEN_EQUATION before recursing. =#
  if we isa BDAE.WHEN_EQUATION || we isa BDAE.INITIAL_WHEN_EQUATION ||
     we isa BDAE.STRUCTURAL_WHEN_EQUATION
    return toWhenStmts(we.whenEquation)
  end
  @match we begin
    BDAE.WHEN_STMTS(condition = cond, whenStmtLst = stmts, elsewhenPart = ewp) => begin
      local simStmts = WhenOperator[toWhenOperator(s) for s in stmts]
      local simElse = @match ewp begin
        SOME(inner) => toWhenStmts(inner)
        NONE() => nothing
        _ => nothing
      end
      WHEN_STMTS(cond, simStmts, simElse)
    end
    _ => error("toWhenStmts: unsupported variant $(typeof(we))")
  end
end

#= Convert lets the auto-generated WHEN_EQUATION / INITIAL_WHEN_EQUATION
   constructors accept legacy `BDAE.WHEN_STMTS` (or BDAE.WhenEquation) values
   without per-call-site casts during the migration. =#
Base.convert(::Type{WHEN_STMTS}, w::WHEN_STMTS) = w
Base.convert(::Type{WHEN_STMTS}, w) = toWhenStmts(w)

"""
    WHEN_EQUATION(size, whenEquation, source, attr)

Runtime `when` clause with a SimCode-native `WHEN_STMTS` body.
"""
struct WHEN_EQUATION <: Equation
  size::Integer
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
  size::Integer
  whenEquation::WHEN_STMTS
  source::DAE.ElementSource
  attr::EQ_ATTR
end

"""
    INLINE_IF_EQUATION(conditions, branchesTrue, branchElse, source, attr)

Raw Modelica `if`-equation appearing inline in a residual / initial /
shared equation list — NOT yet lifted to the event + matched-BRANCH
hierarchy used by `simCode.ifEquations`. The two forms are different:
- `IF_EQUATION` (the `Construct`-subtyped struct above) wraps matched
  `BRANCH`es with per-branch SCC / matchOrder metadata and is the form
  the codegen lifts into MTK callbacks.
- `INLINE_IF_EQUATION` (this struct) is the un-matched form preserved in
  initial / shared equation lists, where if-equations may legally appear
  but never go through matching.

# Fields
- `conditions`   : `Vector{DAE.Exp}` — one per `elseif` arm (no trailing else).
- `branchesTrue` : `Vector{Vector{Equation}}` — equation body per arm.
- `branchElse`   : `Vector{Equation}` — `else` arm body (empty if no else).
- `source`       : `DAE.ElementSource` — source-location decoration.
- `attr`         : `EQ_ATTR` — opaque equation-attribute bundle.
"""
struct INLINE_IF_EQUATION <: Equation
  conditions   :: Vector{DAE.Exp}
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
  lhs::DAE.Exp
  rhs::DAE.Exp
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
  left::DAE.Exp
  right::DAE.Exp
  source::DAE.ElementSource
  attr::EQ_ATTR
end

#= ---- BDAE → Equation converters ----

   Single boundary at simulationCodeTransformation.jl: when SimCode
   ingests a BDAE.EqSystem, every BDAE equation runs through `toSim` and
   becomes a Equation. Downstream sees only Equation. The reverse
   converter exists only for the codegen passes that still need to round-
   trip through BDAE constructors during the migration; those calls
   should disappear as the migration completes. =#

"""
    toEqAttr(bdaeAttr) -> EQ_ATTR

Project a `BDAE.EquationAttributes` to the bits SimCode keeps.
"""
function toEqAttr(@nospecialize(bdaeAttr))::EQ_ATTR
  @match bdaeAttr begin
    BDAE.EQUATION_ATTRIBUTES(differentiated = d) => EQ_ATTR(d)
    _ => EQ_ATTR_DEFAULT
  end
end

"""
    toSim(eq) -> Equation

Convert a single equation to its SimCode counterpart. Idempotent — if `eq`
is already a `Equation`, returns it unchanged.

Drops the BDAE attribute fields that no downstream pass reads.
"""
toSim(eq::Equation)::Equation = eq

function toSim(@nospecialize(eq))::Equation
  @match eq begin
    BDAE.RESIDUAL_EQUATION(exp = exp, source = src, attr = attr) =>
      RESIDUAL_EQUATION(exp, src, toEqAttr(attr))
    BDAE.EQUATION(lhs = lhs, rhs = rhs, source = src, attributes = attr) =>
      EQUATION(lhs, rhs, src, toEqAttr(attr))
    BDAE.WHEN_EQUATION(size = sz, whenEquation = wbody, source = src, attr = attr) =>
      WHEN_EQUATION(sz, toWhenStmts(wbody), src, toEqAttr(attr))
    BDAE.INITIAL_WHEN_EQUATION(size = sz, whenEquation = wbody, source = src, attr = attr) =>
      INITIAL_WHEN_EQUATION(sz, toWhenStmts(wbody), src, toEqAttr(attr))
    BDAE.ALGORITHM(size = sz, alg = alg, source = src, expand = ex, attr = attr) =>
      ALGORITHM(sz, alg, src, ex, toEqAttr(attr))
    BDAE.ARRAY_EQUATION(dimSize = ds, left = l, right = r, source = src, attr = attr) =>
      ARRAY_EQUATION(ds, l, r, src, toEqAttr(attr))
    BDAE.IF_EQUATION(conditions = conds, eqnstrue = trueBs, eqnsfalse = elseB, source = src, attr = attr) => begin
      local condVec = DAE.Exp[c for c in conds]
      local trueVec = Vector{Equation}[Equation[toSim(e) for e in br] for br in trueBs]
      local elseVec = Equation[toSim(e) for e in elseB]
      INLINE_IF_EQUATION(condVec, trueVec, elseVec, src, toEqAttr(attr))
    end
    _ =>
      error("toSim: unsupported BDAE.Equation variant $(typeof(eq))")
  end
end

#= Reverse converter — round-trip back to BDAE for the codegen sites
   still consuming the legacy types during the migration. Loses any
   `kind` / `evalStages` distinction (defaults to UNKNOWN); the
   migration target is to delete these call sites entirely. =#
function toBDAE(eq::RESIDUAL_EQUATION)
  return BDAE.RESIDUAL_EQUATION(eq.exp, eq.source,
    BDAE.EQUATION_ATTRIBUTES(eq.attr.differentiated,
                             BDAE.UNKNOWN_EQUATION_KIND(),
                             BDAE.defaultEvalStages))
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
  bindExp::Option{DAE.Exp}
end

"""
  A Data structure type.
  Currently this type represents complex data structures such as matrices
  or pointers to data structures in memory.
"""
struct DATA_STRUCTURE <: SimVarType
  bindExp::Option{DAE.Exp}
end

"""
An array with N dimensions.
Contains optional binding expression for compile-time evaluation of subscripted access.
"""
struct ARRAY <: SimVarType
  dimensions::Vector{Int}
  bindExp::Option{DAE.Exp}
end

#= Backwards-compatible constructor =#
ARRAY(dimensions::Vector{Int}) = ARRAY(dimensions, NONE())

"""
Parameter variable
"""
struct PARAMETER <: SimVarType
  bindExp::Option{DAE.Exp}
end

"""
Array parameter variable. Has dimensions and an optional binding expression.
Distinct from ARRAY (which is a state array) to ensure correct MTK code generation.
"""
struct ARRAY_PARAMETER <: SimVarType
  dimensions::Vector{Int}
  bindExp::Option{DAE.Exp}
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
struct BRANCH{T1 <: DAE.Exp,
              T2 <: Vector{RESIDUAL_EQUATION},
              T3 <: Int, #= Integer code Each branch has one target (next) The ID of one branch is target - 1=#
              T4 <: Bool,
              T5 <: Vector{Int},
              T6 <: Graphs.AbstractGraph,
              T7 <: Vector{Vector{Int}},
              T8 <: AbstractDict{String, Tuple{Integer, SimVar}}} <: Construct

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
    EXPLICIT_STRUCTURAL_TRANSISTION(fromState, toState, transistionCondition)

Explicit transition between two named structural submodels. Fires the
`transistionCondition` and recompiles into `toState`.
"""
struct EXPLICIT_STRUCTURAL_TRANSISTION <: StructuralTransition
  fromState::String
  toState::String
  transistionCondition::DAE.Exp
end

"""
    IMPLICIT_STRUCTURAL_TRANSISTION(size, whenEquation, source, attr)

Implicit structural transition embedded in a `when` clause; the final
state is not known until the when-body executes. Body is a SimCode-native
`WHEN_STMTS`.
"""
struct IMPLICIT_STRUCTURAL_TRANSISTION <: StructuralTransition
  size::Integer
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
                T1<:AbstractDict{String, Tuple{Integer, SimVar}},
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
