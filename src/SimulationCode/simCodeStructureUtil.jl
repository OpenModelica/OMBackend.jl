#= simCodeStructureUtil.jl

   Conversion / projection machinery and the DAE-wrapping boundary constructors
   for the SimCode data structures defined in `simCodeData.jl`. Kept separate so
   `simCodeData.jl` holds only struct/enum/const definitions and trivial
   same-type constructors. Included at the end of `simCodeData.jl`, so every
   struct it references is already defined; the boundary constructors and
   converters resolve their callees at runtime, never at definition time. =#

# ============================================================================
#  SimCref ⇄ DAE component reference, plus Set/Dict/display interface
# ============================================================================

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

#= Enum-literal subscripts cannot be carried in the Int subscript vector;
   they are folded into the symbol using the same spelling the flat variable
   names use (e.g. `name[Logic.'U'(1)]`), so cref-to-name resolution at
   codegen time matches the variable-table keys. =#
function _hasEnumLiteralSubscript(@nospecialize(subs))::Bool
  for s in subs
    if s isa DAE.INDEX && s.exp isa DAE.ENUM_LITERAL
      return true
    end
  end
  return false
end

function SimCref(@nospecialize(cref::DAE.ComponentRef))::SimCref
  @match cref begin
    DAE.CREF_IDENT(ident = id, subscriptLst = subs) =>
      _hasEnumLiteralSubscript(subs) ?
        SimCref(Symbol(string(cref)), Int[]) :
        SimCref(Symbol(id), _extractIntSubscripts(subs))
    DAE.CREF_ITER(ident = id, subscriptLst = subs) =>
      SimCref(Symbol(id), _extractIntSubscripts(subs))
    DAE.CREF_QUAL(__) =>
      SimCref(Symbol(_flattenQualifiedName(cref)), Int[])
    #= WILD is intercepted in toSimExp(DAE.CREF) and becomes a SimCode.WILD()
       node, so SimCref should never see it. Defensive error if it does. =#
    DAE.WILD(__) =>
      error("SimCref(DAE.WILD): wildcard should be a SimCode.WILD() node, not a SimCref")
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

# ============================================================================
#  Operator projection: OpKind ⇄ DAE.Operator
# ============================================================================

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

# ============================================================================
#  Expression projection: Exp ⇄ DAE.Exp
# ============================================================================

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
toSimExp(e::DAE.CREF)::Exp =
  e.componentRef isa DAE.WILD ? WILD() : EXP_CREF(SimCref(e.componentRef), e.ty)
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
toSimExp(e::DAE.RSUB)::Exp = RSUB(toSimExp(e.exp), Int(e.ix), String(e.fieldName), e.ty)
toSimExp(e::DAE.CAST)::Exp = CAST(e.ty, toSimExp(e.exp))
toSimExp(e::DAE.CALL)::Exp =
  CALL(e.path, Exp[toSimExp(a) for a in e.expLst], e.attr)
toSimExp(e::DAE.RECORD)::Exp =
  RECORD(e.path, Exp[toSimExp(x) for x in e.exps],
         String[string(n) for n in e.comp], e.ty)
toSimExp(e::DAE.TUPLE)::Exp =
  TUPLE(Exp[toSimExp(x) for x in e.PR])
toSimExp(e::DAE.REDUCTION)::Exp =
  REDUCTION(e.reductionInfo, toSimExp(e.expr), e.iterators)

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
toDAEExp(e::EXP_CREF)::DAE.Exp = DAE.CREF(toDAECref(e.cref).componentRef, toDAEType(e.ty))
toDAEExp(e::WILD)::DAE.Exp = DAE.CREF(DAE.WILD(), DAE.T_REAL_DEFAULT)
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
  DAE.ARRAY(toDAEType(e.ty), e.scalar,
            MetaModelica.list((toDAEExp(x) for x in e.elements)...))
toDAEExp(e::ASUB)::DAE.Exp =
  DAE.ASUB(toDAEExp(e.exp),
           MetaModelica.list((toDAEExp(x) for x in e.subs)...))
toDAEExp(e::TSUB)::DAE.Exp = DAE.TSUB(toDAEExp(e.exp), e.index, toDAEType(e.ty))
toDAEExp(e::RSUB)::DAE.Exp = DAE.RSUB(toDAEExp(e.exp), e.index, e.fieldName, toDAEType(e.ty))
toDAEExp(e::CAST)::DAE.Exp = DAE.CAST(toDAEType(e.ty), toDAEExp(e.exp))
toDAEExp(e::CALL)::DAE.Exp =
  DAE.CALL(e.path,
           MetaModelica.list((toDAEExp(x) for x in e.args)...),
           e.attr)
toDAEExp(e::RECORD)::DAE.Exp =
  DAE.RECORD(e.path,
             MetaModelica.list((toDAEExp(x) for x in e.exps)...),
             MetaModelica.list((String(n) for n in e.fieldNames)...),
             toDAEType(e.ty))
toDAEExp(e::TUPLE)::DAE.Exp =
  DAE.TUPLE(MetaModelica.list((toDAEExp(x) for x in e.PR)...))
toDAEExp(e::REDUCTION)::DAE.Exp =
  DAE.REDUCTION(e.info, toDAEExp(e.body), e.iterators)

#= Exp ty-field boundary constructors: accept a DAE.Type and wrap via toSimType,
   so toSimExp and any call site handing in a DAE.Type keep working unchanged. =#
EXP_CREF(cref::SimCref, ty::DAE.Type) = EXP_CREF(cref, toSimType(ty))
ARRAY_EXP(ty::DAE.Type, scalar::Bool, elements::Vector{Exp}) = ARRAY_EXP(toSimType(ty), scalar, elements)
CAST(ty::DAE.Type, exp::Exp) = CAST(toSimType(ty), exp)
TSUB(exp::Exp, index::Int, ty::DAE.Type) = TSUB(exp, index, toSimType(ty))
RSUB(exp::Exp, index::Int, fieldName::String, ty::DAE.Type) =
  RSUB(exp, index, fieldName, toSimType(ty))
RECORD(path::Absyn.Path, exps::Vector{Exp}, fieldNames::Vector{String}, ty::DAE.Type) =
  RECORD(path, exps, fieldNames, toSimType(ty))

#= SimVarType `bindExp` boundary: an `Option{DAE.Exp}` binding (from BDAE / producer
   sites) into the SimCode-native `Option{Exp}` the SimVarType structs store. The
   inverse is `toDAEExp` applied to the unwrapped binding at the (DAE) consumer. =#
Base.@nospecializeinfer function _toSimBindExp(@nospecialize(o))::Option{Exp}
  @match o begin
    SOME(e) => SOME(toSimExp(e))
    _       => NONE()
  end
end

# ============================================================================
#  DAE.Exp-wrapping boundary constructors
#
#  Producers on the BDAE boundary hand in `DAE.Exp`; these outer constructors
#  wrap with `toSimExp` so existing call sites type-check without scattering
#  `toSimExp(...)` calls. The structs themselves live in simCodeData.jl.
# ============================================================================

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

#= BDAE residuals may carry a Nothing source with a real attr; preserve the attr
   and substitute the empty source (converting the exp when it is still DAE). =#
Base.@nospecializeinfer function RESIDUAL_EQUATION(@nospecialize(exp::Exp), src::Nothing, attr::EQ_ATTR)
  return RESIDUAL_EQUATION(exp, DAE.emptyElementSource, attr)
end

Base.@nospecializeinfer function RESIDUAL_EQUATION(@nospecialize(exp::DAE.Exp), src::Nothing, attr::EQ_ATTR)
  return RESIDUAL_EQUATION(toSimExp(exp), DAE.emptyElementSource, attr)
end

Base.convert(::Type{EQ_ATTR}, a::EQ_ATTR) = a
Base.convert(::Type{EQ_ATTR}, a) = toEqAttr(a)

Base.@nospecializeinfer function ASSIGN(@nospecialize(left::DAE.Exp),
                                        @nospecialize(right::DAE.Exp),
                                        @nospecialize(src::DAE.ElementSource))
  return ASSIGN(toSimExp(left), toSimExp(right), src)
end

Base.@nospecializeinfer function REINIT(@nospecialize(stateVar::DAE.CREF),
                                        @nospecialize(value::DAE.Exp),
                                        @nospecialize(src::DAE.ElementSource))
  return REINIT(stateVar, toSimExp(value), src)
end

Base.@nospecializeinfer function ASSERT(@nospecialize(condition::DAE.Exp),
                                        @nospecialize(message::DAE.Exp),
                                        @nospecialize(level::DAE.Exp),
                                        @nospecialize(src::DAE.ElementSource))
  return ASSERT(toSimExp(condition), toSimExp(message), toSimExp(level), src)
end

Base.@nospecializeinfer function TERMINATE(@nospecialize(message::DAE.Exp),
                                           @nospecialize(src::DAE.ElementSource))
  return TERMINATE(toSimExp(message), src)
end

Base.@nospecializeinfer function NORETCALL(@nospecialize(exp::DAE.Exp),
                                           @nospecialize(src::DAE.ElementSource))
  return NORETCALL(toSimExp(exp), src)
end

Base.@nospecializeinfer function RECOMPILATION(@nospecialize(componentToChange::DAE.CREF),
                                               @nospecialize(newValue::DAE.Exp))
  return RECOMPILATION(componentToChange, toSimExp(newValue))
end

Base.@nospecializeinfer function WHEN_STMTS(@nospecialize(condition::DAE.Exp),
                                            @nospecialize(whenStmtLst::Vector{WhenOperator}),
                                            @nospecialize(elsewhenPart::Union{Nothing, WHEN_STMTS}))
  return WHEN_STMTS(toSimExp(condition), whenStmtLst, elsewhenPart)
end

Base.@nospecializeinfer function INLINE_IF_EQUATION(@nospecialize(conditions::Vector{DAE.Exp}),
                                                    @nospecialize(branchesTrue::Vector{Vector{Equation}}),
                                                    @nospecialize(branchElse::Vector{Equation}),
                                                    @nospecialize(src::DAE.ElementSource),
                                                    @nospecialize(attr::EQ_ATTR))
  return INLINE_IF_EQUATION(Exp[toSimExp(c) for c in conditions],
                            branchesTrue, branchElse, src, attr)
end

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

Base.@nospecializeinfer function EXPLICIT_STRUCTURAL_TRANSITION(@nospecialize(fromState::String),
                                                                 @nospecialize(toState::String),
                                                                 @nospecialize(transitionCondition::DAE.Exp))
  return EXPLICIT_STRUCTURAL_TRANSITION(fromState, toState, toSimExp(transitionCondition))
end

# ============================================================================
#  BDAE → SimCode converters
#
#  Single boundary at simulationCodeTransformation.jl: when SimCode ingests a
#  BDAE.EqSystem, every BDAE equation/when-operator runs through these and
#  becomes a SimCode-native value. The reverse `toBDAE` exists only for the
#  codegen passes that still round-trip through BDAE during the migration.
# ============================================================================

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
