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
  #= Delegate to the same canonicaliser that names SimVars
     (BDAE_identifierToVarString → canonicalName), so a qualified cref's
     flattened form matches its variable-table key exactly — including
     per-subscript brackets on each segment (e.g.
     `myRecord.innerRecords[1].y` -> `myRecord_innerRecords[1]_y`). The
     previous hand-rolled `join(ids, "_")` dropped every segment's
     subscript, so nested record-array crefs failed cref-resolution. =#
  return OMBackend.canonicalName(cref)
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

#= ---- Expression hierarchy (additive, no migration yet) ----

   SimCode-native `Exp` AST. DAE.Exp carries a lot of frontend-only
   information that SimCode never reads (type-checked subscripts,
   call attributes, etc.); a SimCode-local subset gives each codegen
   target the same closed taxonomy and removes per-backend
   `@match DAE.X` clauses.

   This block is additive: nothing constructs `SimulationCode.Exp` yet,
   and the equation fields still hold `DAE.Exp`. Migration of each
   `.exp` / `.lhs` / `.rhs` field is a separate, larger step. =#

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
  ty::DAE.Type
end

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
  ty::DAE.Type
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
  ty::DAE.Type
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
  ty::DAE.Type
  exp::Exp
end

"Record literal `Path(field = e, …)`."
struct RECORD <: Exp
  path::Absyn.Path
  exps::Vector{Exp}
  fieldNames::Vector{String}
  ty::DAE.Type
end

"`(e1, e2, …)` tuple, e.g. multi-return call LHS."
struct TUPLE <: Exp
  PR::Vector{Exp}
end

"""
    toOpKind(op::DAE.Operator) -> OpKind

Project a `DAE.Operator` to its SimCode `OpKind`. Drops the operator-
side type tag. The few DAE-only variants (USERDEFINED, the `_ARR` /
`_ARRAY_SCALAR` / `_SCALAR_ARRAY` arithmetic forms used pre-
scalarization) collapse to the scalar operator — by the time SimCode
sees them everything is scalarized so the distinction is moot.
"""
Base.@nospecializeinfer function toOpKind(@nospecialize(op))::OpKind
  @match op begin
    DAE.ADD(__) || DAE.ADD_ARR(__) || DAE.ADD_ARRAY_SCALAR(__) => OP_ADD
    DAE.SUB(__) || DAE.SUB_ARR(__) || DAE.SUB_SCALAR_ARRAY(__) => OP_SUB
    DAE.MUL(__) || DAE.MUL_ARR(__) || DAE.MUL_ARRAY_SCALAR(__) ||
      DAE.MUL_SCALAR_PRODUCT(__) || DAE.MUL_MATRIX_PRODUCT(__) => OP_MUL
    DAE.DIV(__) || DAE.DIV_ARR(__) || DAE.DIV_ARRAY_SCALAR(__) ||
      DAE.DIV_SCALAR_ARRAY(__) => OP_DIV
    DAE.POW(__) || DAE.POW_ARR(__) || DAE.POW_ARR2(__) ||
      DAE.POW_ARRAY_SCALAR(__) || DAE.POW_SCALAR_ARRAY(__) => OP_POW
    DAE.UMINUS(__) || DAE.UMINUS_ARR(__) => OP_UMINUS
    DAE.AND(__) => OP_AND
    DAE.OR(__) => OP_OR
    DAE.NOT(__) => OP_NOT
    DAE.LESS(__) => OP_LESS
    DAE.LESSEQ(__) => OP_LESSEQ
    DAE.GREATER(__) => OP_GREATER
    DAE.GREATEREQ(__) => OP_GREATEREQ
    DAE.EQUAL(__) => OP_EQUAL
    DAE.NEQUAL(__) => OP_NEQUAL
    _ => error("toOpKind: unsupported operator $(typeof(op))")
  end
end

"""
    toSimExp(e) -> Exp

Project a `DAE.Exp` to its SimCode-native counterpart. Idempotent on
`SimulationCode.Exp`. Errors on DAE variants not yet covered by the
subset — extend `Exp` plus this function when a new variant is reached.
"""
Base.@nospecializeinfer function toSimExp(@nospecialize(e::Exp))::Exp
  return e
end

# Per-variant dispatch: tiny method bodies, JIT-cheap, inference-cheap.
toSimExp(e::DAE.ICONST)::Exp = ICONST(Int(e.integer))
toSimExp(e::DAE.RCONST)::Exp = RCONST(Float64(e.real))
toSimExp(e::DAE.BCONST)::Exp = BCONST(e.bool)
toSimExp(e::DAE.SCONST)::Exp = SCONST(e.string)
toSimExp(e::DAE.ENUM_LITERAL)::Exp = ENUM_LITERAL(e.name, Int(e.index))
toSimExp(e::DAE.CREF)::Exp = EXP_CREF(SimCref(e.componentRef), e.ty)
toSimExp(e::DAE.BINARY)::Exp =
  BINARY(toSimExp(e.exp1), toOpKind(e.operator), toSimExp(e.exp2))
toSimExp(e::DAE.UNARY)::Exp = UNARY(toOpKind(e.operator), toSimExp(e.exp))
toSimExp(e::DAE.LBINARY)::Exp =
  LBINARY(toSimExp(e.exp1), toOpKind(e.operator), toSimExp(e.exp2))
toSimExp(e::DAE.LUNARY)::Exp = LUNARY(toOpKind(e.operator), toSimExp(e.exp))
toSimExp(e::DAE.RELATION)::Exp =
  RELATION(toSimExp(e.exp1), toOpKind(e.operator), toSimExp(e.exp2), Int(e.index))
toSimExp(e::DAE.IFEXP)::Exp =
  IFEXP(toSimExp(e.expCond), toSimExp(e.expThen), toSimExp(e.expElse))
toSimExp(e::DAE.ARRAY)::Exp =
  ARRAY_EXP(e.ty, e.scalar, Exp[toSimExp(x) for x in e.array])
toSimExp(e::DAE.ASUB)::Exp =
  ASUB(toSimExp(e.exp), Exp[toSimExp(x) for x in e.sub])
toSimExp(e::DAE.TSUB)::Exp = TSUB(toSimExp(e.exp), Int(e.ix), e.ty)
toSimExp(e::DAE.CAST)::Exp = CAST(e.ty, toSimExp(e.exp))
toSimExp(e::DAE.CALL)::Exp =
  CALL(e.path, Exp[toSimExp(a) for a in e.expLst], e.attr)
toSimExp(e::DAE.RECORD)::Exp =
  RECORD(e.path, Exp[toSimExp(x) for x in e.exps],
         String[string(n) for n in e.comp], e.ty)
toSimExp(e::DAE.TUPLE)::Exp =
  TUPLE(Exp[toSimExp(x) for x in e.PR])

"""
    toDAEOperator(k::OpKind, ty=DAE.T_REAL_DEFAULT) -> DAE.Operator

Reverse projection of an `OpKind` back to a `DAE.Operator`. The operator
needs a `ty::DAE.Type` field which SimCode dropped; the default is
`T_REAL_DEFAULT` for arithmetic operators (Boolean ops still take a
type, by Modelica convention `T_BOOL_DEFAULT`). The DAE consumers that
read `op.ty` are tolerant of this default.
"""
Base.@nospecializeinfer function toDAEOperator(@nospecialize(k::OpKind), ty = DAE.T_REAL_DEFAULT)
  if k === OP_ADD;       DAE.ADD(ty)
  elseif k === OP_SUB;   DAE.SUB(ty)
  elseif k === OP_MUL;   DAE.MUL(ty)
  elseif k === OP_DIV;   DAE.DIV(ty)
  elseif k === OP_POW;   DAE.POW(ty)
  elseif k === OP_UMINUS; DAE.UMINUS(ty)
  elseif k === OP_AND;   DAE.AND(DAE.T_BOOL_DEFAULT)
  elseif k === OP_OR;    DAE.OR(DAE.T_BOOL_DEFAULT)
  elseif k === OP_NOT;   DAE.NOT(DAE.T_BOOL_DEFAULT)
  elseif k === OP_LESS;     DAE.LESS(DAE.T_BOOL_DEFAULT)
  elseif k === OP_LESSEQ;   DAE.LESSEQ(DAE.T_BOOL_DEFAULT)
  elseif k === OP_GREATER;  DAE.GREATER(DAE.T_BOOL_DEFAULT)
  elseif k === OP_GREATEREQ; DAE.GREATEREQ(DAE.T_BOOL_DEFAULT)
  elseif k === OP_EQUAL;    DAE.EQUAL(DAE.T_BOOL_DEFAULT)
  elseif k === OP_NEQUAL;   DAE.NEQUAL(DAE.T_BOOL_DEFAULT)
  else error("toDAEOperator: unhandled OpKind $k")
  end
end

"""
    toDAEExp(e::Exp) -> DAE.Exp

Reverse projection of a SimCode `Exp` back to a `DAE.Exp`. Lossy in the
operator type field (uses defaults — see `toDAEOperator`), but
round-trip-safe on variant shape. Used by codegen helpers during the
transition while the DAE.Exp-walking machinery is gradually rewritten
to walk SIM `Exp` natively.
"""
Base.@nospecializeinfer function toDAEExp(@nospecialize(e::DAE.Exp))::DAE.Exp
  return e
end

# Per-variant dispatch: keeps each method body tiny (JIT-cheap, inference-cheap).
toDAEExp(e::ICONST)::DAE.Exp = DAE.ICONST(e.value)
toDAEExp(e::RCONST)::DAE.Exp = DAE.RCONST(e.value)
toDAEExp(e::BCONST)::DAE.Exp = DAE.BCONST(e.value)
toDAEExp(e::SCONST)::DAE.Exp = DAE.SCONST(e.value)
toDAEExp(e::ENUM_LITERAL)::DAE.Exp = DAE.ENUM_LITERAL(e.path, e.index)
toDAEExp(e::EXP_CREF)::DAE.Exp = DAE.CREF(toDAECref(e.cref).componentRef, e.ty)
toDAEExp(e::BINARY)::DAE.Exp =
  DAE.BINARY(toDAEExp(e.exp1), toDAEOperator(e.op), toDAEExp(e.exp2))
toDAEExp(e::UNARY)::DAE.Exp =
  DAE.UNARY(toDAEOperator(e.op), toDAEExp(e.exp))
toDAEExp(e::LBINARY)::DAE.Exp =
  DAE.LBINARY(toDAEExp(e.exp1), toDAEOperator(e.op), toDAEExp(e.exp2))
toDAEExp(e::LUNARY)::DAE.Exp =
  DAE.LUNARY(toDAEOperator(e.op), toDAEExp(e.exp))
toDAEExp(e::RELATION)::DAE.Exp =
  DAE.RELATION(toDAEExp(e.exp1), toDAEOperator(e.op),
               toDAEExp(e.exp2), e.index, NONE())
toDAEExp(e::IFEXP)::DAE.Exp =
  DAE.IFEXP(toDAEExp(e.cond), toDAEExp(e.thenExp), toDAEExp(e.elseExp))
toDAEExp(e::ARRAY_EXP)::DAE.Exp =
  DAE.ARRAY(e.ty, e.scalar,
            MetaModelica.list((toDAEExp(x) for x in e.elements)...))
toDAEExp(e::ASUB)::DAE.Exp =
  DAE.ASUB(toDAEExp(e.exp),
           MetaModelica.list((toDAEExp(x) for x in e.subs)...))
toDAEExp(e::TSUB)::DAE.Exp = DAE.TSUB(toDAEExp(e.exp), e.index, e.ty)
toDAEExp(e::CAST)::DAE.Exp = DAE.CAST(e.ty, toDAEExp(e.exp))
toDAEExp(e::CALL)::DAE.Exp =
  DAE.CALL(e.path,
           MetaModelica.list((toDAEExp(x) for x in e.args)...),
           e.attr)
toDAEExp(e::RECORD)::DAE.Exp =
  DAE.RECORD(e.path,
             MetaModelica.list((toDAEExp(x) for x in e.exps)...),
             MetaModelica.list((String(n) for n in e.fieldNames)...),
             e.ty)
toDAEExp(e::TUPLE)::DAE.Exp =
  DAE.TUPLE(MetaModelica.list((toDAEExp(x) for x in e.PR)...))

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

Residual equation `0 = exp`. `exp` is a `DAE.Exp` today. The field
migration to `SimulationCode.Exp` is staged on a SIM-Exp-walking
codegen (`expToJulia*(::SimulationCode.Exp, …)` overloads, native
implementation) which is still being built out — see
`simCodeExpBridge.jl` for the Phase 4a infrastructure.

`source` is the original Modelica source location. `attr` is the
opaque metadata bundle.
"""
struct RESIDUAL_EQUATION <: Equation
  exp::Exp
  source::DAE.ElementSource
  attr::EQ_ATTR
end

#= Phase 4b migration: field type flipped DAE.Exp -> SimCode `Exp`.
   Producers on the BDAE boundary still hand in `DAE.Exp`; explicit
   outer constructors wrap with `toSimExp` so existing call sites
   (BDAECreate, BackendEquation, simulationCodeTransformation.toSim)
   continue to type-check without scattering `toSimExp(...)` calls. =#
Base.@nospecializeinfer function RESIDUAL_EQUATION(@nospecialize(exp::DAE.Exp),
                                                   @nospecialize(src::DAE.ElementSource),
                                                   @nospecialize(attr::EQ_ATTR))
  return RESIDUAL_EQUATION(toSimExp(exp), src, attr)
end

Base.@nospecializeinfer function RESIDUAL_EQUATION(@nospecialize(exp::Exp);
                                                   source::DAE.ElementSource = DAE.emptyElementSource,
                                                   attr::EQ_ATTR = EQ_ATTR_DEFAULT)
  return RESIDUAL_EQUATION(exp, source, attr)
end

Base.@nospecializeinfer function RESIDUAL_EQUATION(@nospecialize(exp::DAE.Exp);
                                                   source::DAE.ElementSource = DAE.emptyElementSource,
                                                   attr::EQ_ATTR = EQ_ATTR_DEFAULT)
  return RESIDUAL_EQUATION(toSimExp(exp), source, attr)
end

Base.@nospecializeinfer function RESIDUAL_EQUATION(@nospecialize(exp::Exp), src::Nothing, attr::Nothing)
  return RESIDUAL_EQUATION(exp, DAE.emptyElementSource, EQ_ATTR_DEFAULT)
end

Base.@nospecializeinfer function RESIDUAL_EQUATION(@nospecialize(exp::DAE.Exp), src::Nothing, attr::Nothing)
  return RESIDUAL_EQUATION(toSimExp(exp), DAE.emptyElementSource, EQ_ATTR_DEFAULT)
end

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
  left::Exp
  right::Exp
  source::DAE.ElementSource
end

Base.@nospecializeinfer function ASSIGN(@nospecialize(left::DAE.Exp),
                                        @nospecialize(right::DAE.Exp),
                                        @nospecialize(src::DAE.ElementSource))
  return ASSIGN(toSimExp(left), toSimExp(right), src)
end

struct REINIT <: WhenOperator
  stateVar::DAE.CREF
  value::Exp
  source::DAE.ElementSource
end

Base.@nospecializeinfer function REINIT(@nospecialize(stateVar::DAE.CREF),
                                        @nospecialize(value::DAE.Exp),
                                        @nospecialize(src::DAE.ElementSource))
  return REINIT(stateVar, toSimExp(value), src)
end

struct ASSERT <: WhenOperator
  condition::Exp
  message::Exp
  level::Exp
  source::DAE.ElementSource
end

Base.@nospecializeinfer function ASSERT(@nospecialize(condition::DAE.Exp),
                                        @nospecialize(message::DAE.Exp),
                                        @nospecialize(level::DAE.Exp),
                                        @nospecialize(src::DAE.ElementSource))
  return ASSERT(toSimExp(condition), toSimExp(message), toSimExp(level), src)
end

struct TERMINATE <: WhenOperator
  message::Exp
  source::DAE.ElementSource
end

Base.@nospecializeinfer function TERMINATE(@nospecialize(message::DAE.Exp),
                                           @nospecialize(src::DAE.ElementSource))
  return TERMINATE(toSimExp(message), src)
end

struct NORETCALL <: WhenOperator
  exp::Exp
  source::DAE.ElementSource
end

Base.@nospecializeinfer function NORETCALL(@nospecialize(exp::DAE.Exp),
                                           @nospecialize(src::DAE.ElementSource))
  return NORETCALL(toSimExp(exp), src)
end

struct RECOMPILATION <: WhenOperator
  componentToChange::DAE.CREF
  newValue::Exp
end

Base.@nospecializeinfer function RECOMPILATION(@nospecialize(componentToChange::DAE.CREF),
                                               @nospecialize(newValue::DAE.Exp))
  return RECOMPILATION(componentToChange, toSimExp(newValue))
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

Base.@nospecializeinfer function WHEN_STMTS(@nospecialize(condition::DAE.Exp),
                                            @nospecialize(whenStmtLst::Vector{WhenOperator}),
                                            @nospecialize(elsewhenPart::Union{Nothing, WHEN_STMTS}))
  return WHEN_STMTS(toSimExp(condition), whenStmtLst, elsewhenPart)
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
      #= BDAE side hands a `DAE.Exp` condition; the SIM `WHEN_STMTS.condition`
         field is `::Exp`, so wrap explicitly here at the boundary. =#
      WHEN_STMTS(toSimExp(cond), simStmts, simElse)
    end
    _ => error("toWhenStmts: unsupported variant $(typeof(we))")
  end
end

#= No `Base.convert` overloads for WHEN_STMTS. Producers (toSim, codeGen)
   wrap explicitly with `toWhenStmts`; defining `Base.convert` on a
   user-defined struct turned out to drive Julia 1.12 dispatch into a
   world-split recursion that segfaulted type inference. =#

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
  conditions   :: Vector{Exp}
  branchesTrue :: Vector{Vector{Equation}}
  branchElse   :: Vector{Equation}
  source       :: DAE.ElementSource
  attr         :: EQ_ATTR
end

Base.@nospecializeinfer function INLINE_IF_EQUATION(@nospecialize(conditions::Vector{DAE.Exp}),
                                                    @nospecialize(branchesTrue::Vector{Vector{Equation}}),
                                                    @nospecialize(branchElse::Vector{Equation}),
                                                    @nospecialize(src::DAE.ElementSource),
                                                    @nospecialize(attr::EQ_ATTR))
  return INLINE_IF_EQUATION(Exp[toSimExp(c) for c in conditions],
                            branchesTrue, branchElse, src, attr)
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

#= Boundary overloads accept BDAE-side `DAE.Exp` and wrap via `toSimExp`;
   producers and consumers continue to type-check without scattering
   conversion calls. =#
Base.@nospecializeinfer function EQUATION(@nospecialize(lhs::DAE.Exp),
                                          @nospecialize(rhs::DAE.Exp),
                                          @nospecialize(src::DAE.ElementSource),
                                          @nospecialize(attr::EQ_ATTR))
  return EQUATION(toSimExp(lhs), toSimExp(rhs), src, attr)
end
Base.@nospecializeinfer function EQUATION(@nospecialize(lhs::Exp),
                                          @nospecialize(rhs::DAE.Exp),
                                          @nospecialize(src::DAE.ElementSource),
                                          @nospecialize(attr::EQ_ATTR))
  return EQUATION(lhs, toSimExp(rhs), src, attr)
end
Base.@nospecializeinfer function EQUATION(@nospecialize(lhs::DAE.Exp),
                                          @nospecialize(rhs::Exp),
                                          @nospecialize(src::DAE.ElementSource),
                                          @nospecialize(attr::EQ_ATTR))
  return EQUATION(toSimExp(lhs), rhs, src, attr)
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

Base.@nospecializeinfer function ARRAY_EQUATION(@nospecialize(dims::Vector{Int64}),
                                                @nospecialize(l::DAE.Exp),
                                                @nospecialize(r::DAE.Exp),
                                                @nospecialize(src::DAE.ElementSource),
                                                @nospecialize(attr::EQ_ATTR))
  return ARRAY_EQUATION(dims, toSimExp(l), toSimExp(r), src, attr)
end
Base.@nospecializeinfer function ARRAY_EQUATION(@nospecialize(dims::Vector{Int64}),
                                                @nospecialize(l::Exp),
                                                @nospecialize(r::DAE.Exp),
                                                @nospecialize(src::DAE.ElementSource),
                                                @nospecialize(attr::EQ_ATTR))
  return ARRAY_EQUATION(dims, l, toSimExp(r), src, attr)
end
Base.@nospecializeinfer function ARRAY_EQUATION(@nospecialize(dims::Vector{Int64}),
                                                @nospecialize(l::DAE.Exp),
                                                @nospecialize(r::Exp),
                                                @nospecialize(src::DAE.ElementSource),
                                                @nospecialize(attr::EQ_ATTR))
  return ARRAY_EQUATION(dims, toSimExp(l), r, src, attr)
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
  return BDAE.RESIDUAL_EQUATION(toDAEExp(eq.exp), eq.source,
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
struct BRANCH{T1 <: Exp,
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
  transistionCondition::Exp
end

Base.@nospecializeinfer function EXPLICIT_STRUCTURAL_TRANSISTION(@nospecialize(fromState::String),
                                                                 @nospecialize(toState::String),
                                                                 @nospecialize(transistionCondition::DAE.Exp))
  return EXPLICIT_STRUCTURAL_TRANSISTION(fromState, toState, toSimExp(transistionCondition))
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
