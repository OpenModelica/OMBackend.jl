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


#= MTK code generation: discrete affects and clusters, the event iteration, asserts, self-scheduling time whens, when statements. =#

#= ==================================================================================== =#
#= Equation-form affects                                                                =#
#= ==================================================================================== =#

"""
An operand inside an event affect. `pre(x)` becomes `ModelingToolkit.Pre(x)`, the
pre-event value, so an affect that reads a discrete's own previous value (`mode = …
pre(mode) …`) is solvable; a bare reference would make the written variable appear on its
own right side and MTK raises UnsolvableCallbackError. Continuous operands read their
current value.
"""
function _affectExpToReal(e::SC.Exp, simCode::SC.SIM_CODE)
  e isa SC.PRE && return :(ModelingToolkit.Pre($(expToJuliaExpMTK(SC.EXP_CREF(e.cref, e.ty), simCode))))
  SC.isBuiltin(e, :pre) && return :(ModelingToolkit.Pre($(expToJuliaExpMTK(first(e.args), simCode))))
  e isa SC.BINARY && return :($(opKindToJuliaOperator(e.op))($(_affectExpToReal(e.exp1, simCode)), $(_affectExpToReal(e.exp2, simCode))))
  e isa SC.UNARY && return :($(opKindToJuliaOperator(e.op))($(_affectExpToReal(e.exp, simCode))))
  return expToJuliaExpMTK(e, simCode)
end

"""
A Boolean expression as an MTK Real (0.0/1.0): each relation `ifelse(rel, 1.0, 0.0)`, so
and/or/not stay arithmetic on Reals (a bare relation is a Julia Bool, and `Bool + Bool` an
Int64, illegal in the Boolean context the affect feeds). pre() and discretes are 0/1 Reals
already.
"""
function _boolToReal(e::SC.Exp, simCode::SC.SIM_CODE)
  e isa SC.BCONST && return e.value ? :(1.0) : :(0.0)
  if e isa SC.RELATION
    return :(ModelingToolkit.ifelse(
      $(opKindToJuliaOperator(e.op))($(_affectExpToReal(e.exp1, simCode)), $(_affectExpToReal(e.exp2, simCode))), 1.0, 0.0))
  end
  e isa SC.LUNARY && return :(1.0 - $(_boolToReal(e.exp, simCode)))
  if e isa SC.LBINARY
    local a = _boolToReal(e.exp1, simCode)
    local b = _boolToReal(e.exp2, simCode)
    return e.op === SC.OP_AND ? :($(a) * $(b)) : :($(a) + $(b) - $(a) * $(b))
  end
  if e isa SC.IFEXP
    local c = _boolToReal(e.cond, simCode)
    return :($(c) * $(_boolToReal(e.thenExp, simCode)) + (1.0 - $(c)) * $(_boolToReal(e.elseExp, simCode)))
  end
  (e isa SC.PRE || SC.isBuiltin(e, :pre)) && return _affectExpToReal(e, simCode)
  return expToJuliaExpMTK(e, simCode)
end

"""
An Integer or enumeration discrete's right side as an MTK Real of its value (not a 0/1
Boolean): the conditions through the Boolean lowering, the branches their values, so a
five-valued FSM (PartialFriction's `mode`) is kept instead of clamped.
"""
function _discreteIntToReal(e::SC.Exp, simCode::SC.SIM_CODE)
  e isa Union{SC.ICONST, SC.RCONST} && return Float64(e.value)
  e isa SC.BCONST && return e.value ? 1.0 : 0.0
  e isa SC.ENUM_LITERAL && return Float64(e.index)
  if e isa SC.IFEXP
    return :(ModelingToolkit.ifelse(0.5 < $(_boolToReal(e.cond, simCode)),
                                    $(_discreteIntToReal(e.thenExp, simCode)), $(_discreteIntToReal(e.elseExp, simCode))))
  end
  e isa SC.BINARY && return :($(opKindToJuliaOperator(e.op))($(_discreteIntToReal(e.exp1, simCode)), $(_discreteIntToReal(e.exp2, simCode))))
  (e isa SC.PRE || SC.isBuiltin(e, :pre)) && return _affectExpToReal(e, simCode)
  return expToJuliaExpMTK(e, simCode)
end

"""
One affect equation `disc ~ <value>` for a lifted discrete: the triggering relations
pinned to their post-crossing values (`pins`, a zero-set group, see _zeroSetPins).
Integer/enum discretes keep their multi-valued result; Boolean discretes are 0/1-clamped.
"""
function _discreteAffectEq(discSym::Symbol, rhs::SC.Exp, pins::Dict{SC.Exp, Bool}, isInt::Bool, simCode::SC.SIM_CODE)::Expr
  local pinned = _substRelations(rhs, pins)
  isInt && return :($(discSym) ~ $(_discreteIntToReal(pinned, simCode)))
  return :($(discSym) ~ ModelingToolkit.ifelse(0.5 < $(_boolToReal(pinned, simCode)), 1.0, 0.0))
end

"""
The initialize affect for a lifted discrete: the FULL right side at t0 (no relation pinned).
A `time >= startTime` relation is already satisfied at t0, so no zero crossing fires
there; without this the discrete is stuck at its default start value. The `initial()` term
of the §17.4.4 condition.
"""
function _discreteAffectEqInit(discSym::Symbol, rhs::SC.Exp, isInt::Bool, simCode::SC.SIM_CODE)::Expr
  isInt && return :($(discSym) ~ $(_discreteIntToReal(rhs, simCode)))
  return :($(discSym) ~ ModelingToolkit.ifelse(0.5 < $(_boolToReal(rhs, simCode)), 1.0, 0.0))
end

"""
The discrete-cluster path (discreteClusters.jl): the whens over relations of coupled
discretes are evaluated by the event iteration (MLS 8.6) instead of MTK affects, which have
no event iteration: a pulse's sample event, the whens it enables and the algebraic re-solve
then happen in one event. The path for every model; OMBACKEND_DISCRETE_PRE_MEMORY=false
keeps the MTK affects for the models without mode FSMs, table or switch clusters (the
variable keeps its name from the pre-memory affects this replaced).
"""
_discreteClustersForced()::Bool = OMBackend.envSwitch("OMBACKEND_DISCRETE_PRE_MEMORY")

"""
Whether an expression reads a constant table by a subscript (an ASUB). Such clusters are
Newton-hostile: equation-form affects compile to an implicit solve whose residuals are
piecewise constant.
"""
function _expHasTableLookup(e::SC.Exp)::Bool
  e isa SC.ASUB && return true
  e isa Union{SC.BINARY, SC.LBINARY, SC.RELATION, SC.UNARY, SC.LUNARY, SC.CAST, SC.IFEXP, SC.CALL} || return false
  return any(_expHasTableLookup, SC.children(e))
end

"""
True if the model has lifted discrete-Boolean when clusters AND its residuals read
constant tables. The affect system MTK builds for an equation-form affect pulls in the
surrounding algebraic equations, so the model-level residual content decides
Newton-hostility, not the cluster bodies. Selects the imperative affect lowering and the
Newton-FD event re-initialization. A cluster without relations (it follows pre() values)
closes no loop through a table.
"""
function _modelHasTableClusters(simCode::SC.SIM_CODE)::Bool
  local hasCluster = any(simCode.whenEquations) do weq
    local rels = _extractChangeRelations(weq.whenEquation.condition, simCode)
    rels !== nothing && !isempty(rels)
  end
  return hasCluster && any(eq -> _expHasTableLookup(eq.exp), simCode.residualEquations)
end

"""
Unwrap nested block Exprs (and their annotation LineNumberNodes) down to the core
expression.
"""
function _unwrapBlockExpr(@nospecialize(e))
  while e isa Expr && e.head == :block
    local args = [a for a in e.args if !(a isa LineNumberNode)]
    isempty(args) && return e
    e = last(args)
  end
  return e
end

"""
The name a pre() reads: of `pre(x)`, x; nothing for another expression.
"""
function _preReadName(e::SC.Exp)::Union{Nothing, String}
  e isa SC.PRE && return SC.crefName(e.cref)
  (SC.isBuiltin(e, :pre) && first(e.args) isa SC.EXP_CREF) && return SC.crefName(first(e.args).cref)
  return nothing
end

"""
Whether an expression reads pre() of a name in `names`.
"""
_expHasPreOf(e::SC.Exp, names::Set{String})::Bool =
  SC.anyExp(x -> (local n = _preReadName(x); n !== nothing && n in names), e)

_isIntegerOrEnumeration(v::SC.SIMVAR)::Bool = v.elementType isa Union{SC.TYPE_INTEGER, SC.TYPE_ENUM}

"""
True if a lifted cluster assigns an Integer/enum discrete (a mode variable) whose right side
reads pre() of a discrete assigned in the same cluster. Such bodies are sequential FSM
transitions; the equation-form affect solves them simultaneously, which mis-latches. Scoped
to the mode-carrying shape; Boolean-only self-pre clusters (freewheel logic) stay on the
plain path, which iterates them to a fixpoint within the event (_eventIterAffectParts).
"""
function _modelHasModeFSMClusters(simCode::SC.SIM_CODE)::Bool
  for weq in simCode.whenEquations
    _extractChangeRelations(weq.whenEquation.condition, simCode) === nothing && continue
    local assigned = Set{String}()
    local intRhss = SC.Exp[]
    for st in weq.whenEquation.body
      st isa SC.ASSIGN || continue
      local leftName = SC.crefNameOf(st.left)
      leftName === nothing && continue
      push!(assigned, leftName)
      local entry = get(simCode.stringToSimVarHT, leftName, nothing)
      (entry !== nothing && _isIntegerOrEnumeration(last(entry))) && push!(intRhss, st.right)
    end
    any(rhs -> _expHasPreOf(rhs, assigned), intRhss) && return true
  end
  return false
end

"""
Newton with an FD Jacobian degenerates to the Gauss-Jacobi sweep that piecewise-constant
algebraic rows need; Broyden's secant update diverges on them. The event re-init default
for table-cluster models.
"""
function tableClusterInitAlg()
  return DiffEqBase.BrownFullBasicInit(1e-8, NonlinearSolve.NewtonRaphson(; autodiff = ADTypes.AutoFiniteDiff()))
end

"""
The discrete unknowns of Integer, Boolean or enumeration type, whose values must stay
integral (withIntegralDiscretes).
"""
function integralDiscreteNames(discreteSyms::AbstractVector, simCode::SC.SIM_CODE)::Vector{String}
  local out = String[]
  for s in discreteSyms
    local entry = get(simCode.stringToSimVarHT, string(s), nothing)
    (entry !== nothing && last(entry).elementType isa Union{SC.TYPE_INTEGER, SC.TYPE_BOOL, SC.TYPE_ENUM}) &&
      push!(out, string(s))
  end
  return out
end

"""
Whether `name` is a Boolean discrete (pre(name) read from the Float memory must become a
Bool before use in an and/or/not context).
"""
function _isBoolDiscreteName(name::String, simCode::SC.SIM_CODE)::Bool
  local entry = get(simCode.stringToSimVarHT, name, nothing)
  return entry !== nothing && last(entry).elementType isa SC.TYPE_BOOL
end

#= ==================================================================================== =#
#= Imperative affects                                                                   =#
#= ==================================================================================== =#

#= Names substituted statically inside an affect body (post-event branch values replacing
   stale observed reads). Default: no substitution. =#
const _EMPTY_MEM_SUBST = Dict{Symbol, Any}()
const _EMPTY_REL_PINS = Dict{SC.Exp, Bool}()

#= An expression that _affectValue cannot lower into an ImperativeAffect body. The
   event-iteration path catches it and keeps the equation-form affects. =#
struct _UnsupportedInAffect <: Exception
  exp::Any
end
Base.showerror(io::IO, e::_UnsupportedInAffect) =
  print(io, "_affectValue: unsupported in an ImperativeAffect body: ", string(e.exp))

#= What a lowering to an affect body or a run-time check throws for a construct it does not support. =#
const LoweringFailure = Union{UnsupportedLowering, _UnsupportedInAffect}

#= Modelica event operators with no meaning inside an affect body; the generic call lowering
   would emit an undefined Julia function for them. =#
const _AFFECT_UNSUPPORTED_BUILTINS = (:sample, :terminal, :delay, :der, :reinit, :cardinality)

"""
The pinned value of a relation inside an affect body (a Bool, or an expression that reads
it, a discrete cluster's buffer), or nothing.
"""
_relPinValue(e::SC.Exp, relPins::AbstractDict{SC.Exp}) = (!isempty(relPins) && e isa SC.RELATION) ? get(relPins, e, nothing) : nothing

"""
For edge()/change() of a relation, or of a Boolean defined by one, whose values `relPins`
and values before the pass `relPre` are given (a discrete cluster): the two expressions.
Otherwise nothing.
"""
function _relationAndPre(v::SC.Exp, relPins::AbstractDict{SC.Exp}, relPre::Union{Nothing, AbstractDict{SC.Exp}},
                         simCode::SC.SIM_CODE)
  relPre === nothing && return nothing
  local rel = v isa SC.RELATION ? v : v isa SC.EXP_CREF ? _condVarRelation(SC.crefName(v.cref), simCode) : nothing
  (rel !== nothing && haskey(relPins, rel) && haskey(relPre, rel)) || return nothing
  return (relPins[rel], relPre[rel])
end

"""
    _affectBool(e, obsAcc, simCode; initVal, subst, relPins, preSubst, relPre)

`e` at a Boolean position of an ImperativeAffect body: live reads come back as 0/1 Floats, `>
0.5` coerces both Bool and Float. `initVal` is what `initial()` lowers to (true in an
initialize affect, or the name of a Bool argument). `edge(v)` is `v and not pre(v)`,
`change(v)` `v <> pre(v)`, pre() lowered like any other (the event iteration's previous pass;
for a discrete cluster's relation, with `relPre`, its value before the pass). The frontend
inlines `b = x > 0.2` into edge(x > 0.2): in an event iteration a relation changes only at its
own crossing, i.e. when this callback pins it, and only in the first pass (later passes see
pre = the same value).
"""
function _affectBool(e::SC.Exp, obsAcc::Dict{Symbol, Symbol}, simCode::SC.SIM_CODE;
                     initVal::Union{Bool, Symbol} = false, subst::Dict{Symbol, Any} = _EMPTY_MEM_SUBST,
                     relPins::AbstractDict{SC.Exp} = _EMPTY_REL_PINS, preSubst::Union{Nothing, Dict{Symbol, Any}} = nothing,
                     relPre::Union{Nothing, AbstractDict{SC.Exp}} = nothing)
  local recb = x::SC.Exp -> _affectBool(x, obsAcc, simCode; initVal, subst, relPins, preSubst, relPre)
  local pin = _relPinValue(e, relPins)
  pin === nothing || return pin
  e isa SC.BCONST && return e.value
  e isa SC.LUNARY && return :(!$(recb(e.exp)))
  e isa SC.LBINARY && return e.op === SC.OP_AND ? :($(recb(e.exp1)) && $(recb(e.exp2))) : :($(recb(e.exp1)) || $(recb(e.exp2)))
  e isa SC.RELATION && return _affectValue(e, obsAcc, simCode; initVal, subst, relPins, preSubst, relPre)
  SC.isBuiltin(e, :initial) && return initVal
  if SC.isBuiltin(e, :edge) || SC.isBuiltin(e, :change)
    local v = first(e.args)
    local rp = _relationAndPre(v, relPins, relPre, simCode)
    rp === nothing || return e.name === :edge ? :($(rp[1]) && !$(rp[2])) : :($(rp[1]) != $(rp[2]))
    if v isa SC.RELATION && preSubst !== nothing
      e.name === :edge && return _relPinValue(v, relPins) === true ? :(_firstPass) : false
      return _relPinValue(v, relPins) === nothing ? false : :(_firstPass)
    end
    v isa SC.EXP_CREF || throw(_UnsupportedInAffect(e))
    local preV = SC.PRE(v.cref, v.ty)
    e.name === :edge && return :($(recb(v)) && !$(recb(preV)))
    _isBoolDiscreteName(SC.crefName(v.cref), simCode) && return :($(recb(v)) != $(recb(preV)))
    local recv = x::SC.Exp -> _affectValue(x, obsAcc, simCode; initVal, subst, relPins, preSubst, relPre)
    return :($(recv(v)) != $(recv(preV)))
  end
  e isa SC.IFEXP && return :($(recb(e.cond)) ? $(recb(e.thenExp)) : $(recb(e.elseExp)))
  local v = _affectValue(e, obsAcc, simCode; initVal, subst, preSubst, relPre)
  #= pre() of a Boolean discrete comes back as a Bool already (`_pre[1] > 0.5`) =#
  return v isa Bool || _isBoolPreRead(e, simCode) ? v : :($(v) > 0.5)
end

"Whether `e` is pre() of a Boolean discrete (`_affectValue` reads it as a Bool)."
function _isBoolPreRead(e::SC.Exp, simCode::SC.SIM_CODE)::Bool
  (e isa SC.PRE || SC.isBuiltin(e, :pre)) || return false
  local nm = _preReadName(e)
  return nm !== nothing && _isBoolDiscreteName(nm, simCode)
end

"""
    _affectValue(e, obsAcc, simCode; initVal, subst, relPins, preSubst, relPre)

`e` in an ImperativeAffect body (or a discrete cluster's body): `pre(x)` as `preSubst[x]`, else
`observed.x` (its value when the affect runs, before it changes anything); a continuous
variable or a parameter as `observed.<name>` (collected into `obsAcc`); relations, if, and, or,
not and arithmetic as Julia's. `initVal`: what `initial()` lowers to (see _affectBool).
Relations are replaced through `relPins`; `relPre`: see _affectBool.
"""
function _affectValue(e::SC.Exp, obsAcc::Dict{Symbol, Symbol}, simCode::SC.SIM_CODE;
                      initVal::Union{Bool, Symbol} = false, subst::Dict{Symbol, Any} = _EMPTY_MEM_SUBST,
                      relPins::AbstractDict{SC.Exp} = _EMPTY_REL_PINS, preSubst::Union{Nothing, Dict{Symbol, Any}} = nothing,
                      relPre::Union{Nothing, AbstractDict{SC.Exp}} = nothing)
  local rec = x::SC.Exp -> _affectValue(x, obsAcc, simCode; initVal, subst, relPins, preSubst, relPre)
  local recb = x::SC.Exp -> _affectBool(x, obsAcc, simCode; initVal, subst, relPins, preSubst, relPre)
  local pin = _relPinValue(e, relPins)
  pin === nothing || return pin
  e isa Union{SC.ICONST, SC.RCONST} && return Float64(e.value)
  e isa Union{SC.BCONST, SC.SCONST} && return e.value
  e isa SC.ENUM_LITERAL && return Float64(e.index)
  if e isa SC.PRE || SC.isBuiltin(e, :pre)
    local nm = _preReadName(e)
    nm === nothing && throw(_UnsupportedInAffect(e))
    local key = Symbol(nm)
    local rd = if preSubst !== nothing && haskey(preSubst, key)
      preSubst[key]
    else
      obsAcc[key] = key
      :(observed.$(key))
    end
    return _isBoolDiscreteName(nm, simCode) ? :($(rd) > 0.5) : rd
  end
  e isa SC.DER && throw(_UnsupportedInAffect(e))
  e isa SC.CALL && return _affectCall(e, obsAcc, simCode, rec, recb, initVal)
  e isa SC.CAST && return rec(e.exp)
  e isa Union{SC.RELATION, SC.BINARY} && return :($(opKindToJuliaOperator(e.op))($(rec(e.exp1)), $(rec(e.exp2))))
  e isa SC.UNARY && return :($(opKindToJuliaOperator(e.op))($(rec(e.exp))))
  e isa SC.LUNARY && return :(!$(recb(e.exp)))
  e isa SC.LBINARY && return e.op === SC.OP_AND ? :($(recb(e.exp1)) && $(recb(e.exp2))) : :($(recb(e.exp1)) || $(recb(e.exp2)))
  e isa SC.IFEXP && return :($(recb(e.cond)) ? $(rec(e.thenExp)) : $(rec(e.elseExp)))
  if e isa SC.EXP_CREF
    local name = SC.crefName(e.cref)
    local nm = Symbol(name)
    #= module-scope `time` is Base.time; the affect reads the integrator clock =#
    nm === :time && return :(integrator.t)
    haskey(subst, nm) && return subst[nm]
    #= a module-global table handle (DATA_STRUCTURE), referenced by bare name =#
    local entry = get(simCode.stringToSimVarHT, name, nothing)
    (entry !== nothing && last(entry).varKind isa SC.DATA_STRUCTURE) && return nm
    obsAcc[nm] = nm
    return :(observed.$(nm))
  end
  #= Constant-table lookup `table[idx...]`: the table a constant literal (the standard
     lowering), the subscripts gate inputs read through `observed` (constTableLookup handles
     numeric and rounded indices). One element: a slice or a whole dimension (`t[:, k]`,
     which read row 1) is the equation form's. =#
  if e isa SC.ASUB
    any(s -> s isa SC.WHOLE_DIMENSION, e.subs) && throw(_UnsupportedInAffect(e))
    return :(OMBackend.CodeGeneration.constTableLookup($(expToJuliaExpMTK(e.exp, simCode)), $((rec(s) for s in e.subs)...)))
  end
  e isa SC.ARRAY_EXP && return expToJuliaExpMTK(e, simCode)
  #= an element of a multi-output call (a tuple assignment split per target: MSL TimeTable's
     (a, b, ...) := getInterpolationCoefficients(...)) =#
  e isa SC.TSUB && return :(getindex($(rec(e.exp)), $(e.index)))
  throw(_UnsupportedInAffect(e))
end

"""
A call in an affect body: the event operators edge() and change() Boolean, initial() its
value, the others refused (they mean nothing there); the index helpers of Digital's gate
residuals (the lookup rounds its index: floor/integer only evaluate the inner value) and the
numeric functions as Julia's; noEvent() and smooth() semantic no-ops; String Modelica's
(Julia's String has no such method: an assert message, MSL Media); another function by its
name (an external one: CombiTimeTable's getNextTimeEvent), its arguments read like the
others.
"""
function _affectCall(e::SC.CALL, obsAcc::Dict{Symbol, Symbol}, simCode::SC.SIM_CODE, rec::R, recb::B,
                     initVal::Union{Bool, Symbol}) where {R, B}
  _isDelayCall(e) && throw(_UnsupportedInAffect(e))
  e.kind === SC.CALL_BUILTIN || return Expr(:call, e.name, map(rec, e.args)...)
  local name = e.name
  name === :initial && return initVal
  name in (:edge, :change) && return recb(e)
  name in _AFFECT_UNSUPPORTED_BUILTINS && throw(_UnsupportedInAffect(e))
  name === :floor && return :(floor($(rec(e.args[1]))))
  name === :integer && return :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.modelica_integer($(rec(e.args[1]))))
  name in (:abs, :sign, :sqrt) && return :($(name)($(rec(e.args[1]))))
  if name in (:min, :max)
    #= min/max of an array (one argument) is not lowered here =#
    length(e.args) == 2 || throw(_UnsupportedInAffect(e))
    return :($(name)($(rec(e.args[1])), $(rec(e.args[2]))))
  end
  name === :noEvent && return rec(e.args[1])
  name === :smooth && return rec(e.args[2])
  name === :String && return AlgorithmicCodeGeneration.modelicaStringCall(e.args, rec)
  return Expr(:call, name, map(rec, e.args)...)
end

"""
True if some member's right side reads pre() of a cluster member, so one pass of the body
can leave pre(x) <> x (see _eventIterAffectParts).
"""
function _clusterReadsOwnPre(assigns::Vector{Tuple{Symbol, SC.Exp, Bool}})::Bool
  local members = Set{String}(string(d) for (d, _, _) in assigns)
  return any(rhs -> _expHasPreOf(rhs, members), (rhs for (_, rhs, _) in assigns))
end

"""
True for the condition of a when synthesized from discrete equations: an or-chain of
`change(<relation>)`. User whens (`when change(b)`, `when x > 0`) run their body once per
event and must not be iterated.
"""
function _isSynthesizedChangeCondition(cond::SC.Exp)::Bool
  SC.isBuiltin(cond, :change) && return first(cond.args) isa SC.RELATION || _isChangeOfPre(cond)
  (cond isa SC.LBINARY && cond.op === SC.OP_OR) && return _isSynthesizedChangeCondition(cond.exp1) && _isSynthesizedChangeCondition(cond.exp2)
  return false
end

"""
Whether the ImperativeAffect can be generated for this cluster: every right side lowers
into its body. Otherwise the cluster keeps the equation-form affects, which lower more
expression kinds.
"""
function _eventIterLowerable(assigns::Vector{Tuple{Symbol, SC.Exp, Bool}}, simCode::SC.SIM_CODE)::Bool
  try
    _eventIterAffectParts(assigns, simCode)
    return true
  catch e
    e isa _UnsupportedInAffect || rethrow()
    @info "Discrete cluster $(join(first.(assigns), ", ")) keeps equation-form event affects" reason = sprint(showerror, e)
    return false
  end
end

#= Upper bound on the passes of one event iteration. =#
const _EVENT_ITERATION_MAX_PASSES = 20

"""
The ImperativeAffect for a plain-path cluster that reads its own pre() values: Modelica's
event iteration (§8.6). Each pass evaluates the body in order with pre(x) = x after the
previous pass (x at the event for the first pass) and member references = their latest
value, until a pass changes nothing. The pinned relations stay pinned; the other operands
are read once, at the event. An equation-form affect is one pass: in the OneWayClutch the
crossing to w_rel <= 0 sets stuck, but `locked = pre(stuck) and not startForward` needs the
second pass, so without it the freewheel never locks. `iterate = false`: one pass (a
cluster that does not read its own pre()). `initVal`: the value of initial() (true in the
t0 pass).
"""
function _eventIterAffectParts(assigns::Vector{Tuple{Symbol, SC.Exp, Bool}}, simCode::SC.SIM_CODE;
                               relPins::Dict{SC.Exp, Bool} = _EMPTY_REL_PINS, initVal::Bool = false, iterate::Bool = true)
  local obsAcc = Dict{Symbol, Symbol}()
  local preSyms = Dict{Symbol, Any}(d => Symbol("_pre_", d) for (d, _, _) in assigns)
  local subst = Dict{Symbol, Any}(d => Symbol("_v_", d) for (d, _, _) in assigns)
  local setup = Expr[]; local pass = Expr[]; local changed = Expr[]
  local shift = Expr[]; local retKws = Expr[]
  for (d, rhs, isInt) in assigns
    local vsym = subst[d]; local psym = preSyms[d]
    push!(setup, :(local $(psym) = Float64(modified.$(d))))
    push!(setup, :(local $(vsym) = $(psym)))
    local valExpr = isInt ?
      :(Float64($(_affectValue(rhs, obsAcc, simCode; initVal, subst, relPins, preSubst = preSyms)))) :
      :($(_affectBool(rhs, obsAcc, simCode; initVal, subst, relPins, preSubst = preSyms)) ? 1.0 : 0.0)
    push!(pass, :($(vsym) = $(valExpr)))
    push!(changed, :($(vsym) != $(psym)))
    push!(shift, :($(psym) = $(vsym)))
    push!(retKws, Expr(:kw, d, vsym))
  end
  local retNT = Expr(:tuple, Expr(:parameters, retKws...))
  local passes = if iterate
    local anyChanged = foldl((a, b) -> :($(a) || $(b)), changed)
    #= One non-convergence warning per cluster, not per generated source line. =#
    local warnId = QuoteNode(Symbol("eventIteration_", first(assigns)[1]))
    quote
      local _converged = false
      for _ in 1:$(_EVENT_ITERATION_MAX_PASSES)
        $(pass...)
        if !($(anyChanged))
          _converged = true
          break
        end
        $(shift...)
        _firstPass = false
      end
      _converged || @warn("Event iteration did not converge", t = integrator.t,
                          cluster = $(QuoteNode(first.(assigns))), _id = $(warnId), maxlog = 1)
    end
  else
    Expr(:block, pass...)
  end
  local fexpr = :((modified, observed, ctx, integrator) -> begin
                    $(setup...)
                    local _firstPass = true   # edge()/change() of a crossing relation
                    $(passes)
                    #= The integrator must see the change (Rosenbrock and multistep
                       methods would otherwise keep the old discrete in their history). =#
                    SciMLBase.u_modified!(integrator, true)
                    $(retNT)
                  end)
  local obsNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, k, v) for (k, v) in obsAcc]...))
  local modNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, d, d) for (d, _, _) in assigns]...))
  return (fexpr, obsNT, modNT)
end

"""
_eventIterAffectParts as an ImperativeAffect expression.
"""
function _imperativeClusterAffect(assigns::Vector{Tuple{Symbol, SC.Exp, Bool}}, simCode::SC.SIM_CODE; kwargs...)::Expr
  local (fn, obs, mod) = _eventIterAffectParts(assigns, simCode; kwargs...)
  return :(ModelingToolkit.ImperativeAffect($(fn), $(mod); observed = $(obs), skip_checks = true))
end

#= ==================================================================================== =#
#= Module-level declarations of the generated model                                    =#
#= ==================================================================================== =#

"""
A discrete's start attribute as a number: a literal, an enumeration literal (its index), or
a parameter's folded binding (the MSL thyristor bridges' off(start = offStart_p1)). Nothing
without a start or when it does not evaluate.
"""
function _discreteStartValue(sv::SC.SIMVAR, simCode::SC.SIM_CODE)::Union{Float64, Nothing}
  local start = sv.attributes.start
  start === nothing && return nothing
  start isa SC.EXP_CREF || return _foldParameterBindStatic(start, simCode)
  local entry = get(simCode.stringToSimVarHT, SC.crefName(start.cref), nothing)
  entry === nothing && return nothing
  local k = last(entry).varKind
  (k isa SC.PARAMETER && k.bindExp !== nothing) || return nothing
  return _foldParameterBindStatic(k.bindExp, simCode)
end

"""
    liftedDiscreteNames(simCode)

The discretes an equation defines at every instant (`b = x > 0.2`, `y = pre(y) and ...`), lifted
into when-equations on the change of the relations they read: the variables those
when-equations assign. Their values at the initialization are their equations' too.
"""
function liftedDiscreteNames(simCode::SC.SIM_CODE)::Vector{String}
  local names = String[]
  for weq in simCode.whenEquations
    _extractChangeRelations(weq.whenEquation.condition, simCode) === nothing && continue
    for st in weq.whenEquation.body
      st isa SC.ASSIGN || continue
      local leftName = SC.crefNameOf(st.left)
      local entry = leftName === nothing ? nothing : get(simCode.stringToSimVarHT, leftName, nothing)
      entry === nothing || push!(names, last(entry).name)
    end
  end
  return names
end

"""
Module-level list of the discretes the discrete clusters assign (emitDiscreteClusters): the
direct-RHS initialization leaves them to the clusters' start bodies; with their start values
and those of the other discretes (what pre() reads at the initialization, MLS 8.6), for the
initial fixpoint of the clusters. Empty when the model does not take that path.
"""
function liftedDiscretesDecl(simCode::SC.SIM_CODE)::Expr
  _usesDiscreteClusters(simCode) || return Expr(:block)
  local names = liftedDiscreteNames(simCode)
  isempty(names) && return Expr(:block)
  local starts = Dict{String, Float64}()
  for (_, (_, sv)) in simCode.stringToSimVarHT
    (sv.varKind isa SC.DISCRETE || sv.name in names) || continue
    local v = _discreteStartValue(sv, simCode)
    v === nothing || (starts[sv.name] = v)
  end
  return quote
    LIFTED_DISCRETES = $(names)
    LIFTED_DISCRETE_STARTS = $(starts)
  end
end

"""
Module-level list of the Real parameters without a value that the initialization computes
(fixed = false, no binding; the MSL InitSpringConstant's spring.c): the direct-RHS
initialization solves them next to the states. Names are the MTK parameters'
(simVar.name). A parameter alone on the left of an initial equation is assigned by it (at
codegen, or by the init solve's assignments, which take the left side), not free: the MSL
Mean block's `t0 = time` (as a free unknown it had no determining row, and its zero Jacobian
column disabled the scaled Newton step the ideal diodes of DiodeBridge2mPulse need); so are
the variables of a tuple equation, each equal to the call's output (MSL Filter's `({r[1],
r[2]}, _, _, _) = roots.lowPass(cr, ...)`). Those the initialization assigns are
ASSIGNED_PARAMETERS: only these, never a bound or tunable parameter alone on the left (`p =
2 * q`: p was overwritten with 2q, and q never solved).
"""
function freeParametersDecl(simCode::SC.SIM_CODE)::Expr
  local assigned = OrderedSet{String}()
  for ieq in simCode.initialEquations
    hasEquationSides(ieq) || continue
    local (lhs, rhs) = equationSides(ieq)
    if lhs isa SC.EXP_CREF
      push!(assigned, SC.crefName(lhs.cref))
    else
      local targets = MTK_CodeGenerationUtil._tupleEquationTargets(lhs, rhs)
      targets === nothing || foreach(t -> push!(assigned, SC.crefName(first(t).cref)), targets)
    end
  end
  local names = String[]
  local assignedNames = String[]
  for (key, (_, sv)) in simCode.stringToSimVarHT
    (SC.isParameter(sv) && !SC.hasBindingExp(sv)) || continue
    if key in assigned
      push!(assignedNames, sv.name)
    elseif sv.elementType isa SC.TYPE_REAL && sv.attributes.fixed == SC.BCONST(false)
      push!(names, sv.name)
    end
  end
  local decls = Expr[]
  isempty(names) || push!(decls, :(FREE_PARAMETERS = $(sort!(unique!(names)))))
  isempty(assignedNames) || push!(decls, :(ASSIGNED_PARAMETERS = $(sort!(unique!(assignedNames)))))
  return Expr(:block, decls...)
end

"""
Module-level list of the user's fixed values (USER_PINS): the continuous Real variables
with fixed = true (their start, or the default one) and the left sides of initial equations
`v = literal or parameter`, named as the MTK unknowns (simVar.name; a relayed variable by
its if-equation temporary). Not discretes: fixed = true there is `pre(v) = start` (MLS
8.6), which may move at the start. The direct-RHS initialization refuses a result that moved
one (buildDirectRHSProblem). Empty when there are none.
"""
function userPinsDecl(simCode::SC.SIM_CODE, relayAliases::AbstractDict)::Expr
  local names = OrderedSet{String}()
  local rename = n -> string(get(relayAliases, Symbol(n), Symbol(n)))
  #= The literal value a fixed start gives (0 without a start), by name. =#
  local fixedLiteral = Dict{String, Float64}()
  for (_, (_, sv)) in simCode.stringToSimVarHT
    (sv.varKind isa Union{SC.STATE, SC.ALG_VARIABLE} && sv.elementType isa SC.TYPE_REAL &&
     sv.attributes.fixed == SC.BCONST(true)) || continue
    push!(names, rename(sv.name))
    local start = sv.attributes.start
    start === nothing && (fixedLiteral[sv.name] = 0.0)
    start isa Union{SC.RCONST, SC.ICONST} && (fixedLiteral[sv.name] = Float64(start.value))
  end
  local plain = x -> begin
    while x isa Expr && x.head === :block
      local args = filter(a -> !(a isa LineNumberNode), x.args)
      length(args) == 1 || return x
      x = only(args)
    end
    x
  end
  for e in generateInitialEquationsAsConstraints(simCode.initialEquations, simCode)
    (e isa Expr && e.head === :call && length(e.args) == 3 && e.args[1] === :~) || continue
    local lhs = plain(e.args[2]); local rhs = plain(e.args[3])
    lhs isa Symbol || continue
    local entry = get(simCode.stringToSimVarHT, string(lhs), nothing)
    entry === nothing || last(entry).varKind isa Union{SC.STATE, SC.ALG_VARIABLE} || continue
    local rhsEntry = rhs isa Symbol ? get(simCode.stringToSimVarHT, string(rhs), nothing) : nothing
    #= A parameter right side (a tunable one stays a reference) is a pin too; a conflict with a
       fixed start is then checked at the build and per run (DirectRHSGeneration's
       _collectHardInitializationValues), here only a literal's. =#
    (rhs isa Number || (rhsEntry !== nothing && SC.isParameter(last(rhsEntry)))) || continue
    #= A fixed start and an initial equation that give one variable two values (`x(start = 1,
       fixed = true)` and `x = 2`): the initialization took the equation's without a word;
       OpenModelica refuses it. =#
    local startVal = get(fixedLiteral, string(lhs), nothing)
    (rhs isa Number && startVal !== nothing && !isapprox(Float64(rhs), startVal; rtol = 1e-12, atol = 1e-12)) &&
      CodeGeneration.unsupported("a fixed start and an initial equation that give a variable different values",
                                 "$(lhs): start $(startVal), initial equation $(rhs)")
    push!(names, rename(lhs))
  end
  isempty(names) && return Expr(:block)
  return :(USER_PINS = $(sort!(collect(names))))
end

"""
Module-level list of the discrete variables (DISCRETE_VARIABLES): their fixed start is
`pre(v) = start` (MLS 8.6), and an initialization equation `v ~ start` beside `v ~ value` (a
when initial() body) is no conflict (_collectHardInitializationValues). Empty when there are
none.
"""
function discreteVariablesDecl(simCode::SC.SIM_CODE)::Expr
  local names = sort!([sv.name for (_, (_, sv)) in simCode.stringToSimVarHT if SC.isDiscrete(sv)])
  isempty(names) && return Expr(:block)
  return :(DISCRETE_VARIABLES = $(names))
end

"""
Module-level list of the tunable parameters (TUNABLE_NAMES, withTunableParameters) the model
was compiled with, for the runs with other values of them (DirectRHSGeneration's
_runEntry). Empty when there are none.
"""
function tunableParametersDecl()::Expr
  local names = sort!(collect(OMBackend.TUNABLE_PARAMETERS[]))
  isempty(names) && return Expr(:block)
  return :(TUNABLE_NAMES = $(names))
end

"""
Whether a continuous or initialization equation of the model calls homotopy() (where it is
the blend, HOMOTOPY_BLEND): it then gets the homotopy parameter (HOMOTOPY_LAMBDA).
"""
function _usesHomotopy(simCode::SC.SIM_CODE)::Bool
  local has = e::SC.Exp -> SC.anyExp(x -> SC.isBuiltin(x, :homotopy), e)
  any(eq -> has(eq.exp), simCode.residualEquations) && return true
  return any(ieq -> hasEquationSides(ieq) && any(has, equationSides(ieq)), simCode.initialEquations)
end

"""
Whether the model takes the discrete-cluster path, shared by every site that must agree
(emitDiscreteClusters, the MTK events, LIFTED_DISCRETES).
"""
_usesDiscreteClusters(simCode::SC.SIM_CODE)::Bool =
  _discreteClustersForced() || _modelHasTableClusters(simCode) || _modelHasModeFSMClusters(simCode) ||
  _modelHasSwitchClusters(simCode) || _modelHasRelationlessClusters(simCode)

"""
True if a lifted cluster has no relation (it follows pre() values): only the event
iteration evaluates it, the MTK events have no crossing for it.
"""
_modelHasRelationlessClusters(simCode::SC.SIM_CODE)::Bool =
  any(simCode.whenEquations) do w
    local rels = _extractChangeRelations(w.whenEquation.condition, simCode)
    rels !== nothing && isempty(rels)
  end

"""
True if a lifted cluster closes an algebraic loop through its own equations, as the MSL's
ideal diodes and thyristors: one of its relations reads an algebraic unknown that only
equations reading a member, or a value a member selects, determine (s in `off = s < 0`, `v =
s*(if off then 1 else Ron)`, `i = s*(if off then Goff else 1)`). After the cluster changes,
s and the circuit must be solved again before its relation is read, within the event: the
discrete-cluster path's event iteration does that; the equation-form affect's implicit solve
fails on the piecewise equations, and at a thyristor's firing the observed relation is read
at the crossing itself. A switch with an arc model is not one: its relation reads the
circuit's own current and voltage.
"""
function _modelHasSwitchClusters(simCode::SC.SIM_CODE)::Bool
  local ht = simCode.stringToSimVarHT
  local names = (exps...) -> (local ns = OrderedSet{String}(); foreach(e -> SC.crefNames!(ns, e), exps); ns)
  local isAlgebraic = name -> haskey(ht, name) && last(ht[name]).varKind isa SC.ALG_VARIABLE
  local isUnknown = name -> haskey(ht, name) && !SC.isParameter(last(ht[name]))
  local ifConds = [names((b.condition for b in ifEq.branches)...) for ifEq in simCode.ifEquations]
  #= Per candidate operand: the names that make an equation the cluster's own. =#
  local own = Dict{String, Set{String}}()
  for weq in simCode.whenEquations
    local rels = _extractChangeRelations(weq.whenEquation.condition, simCode)
    rels === nothing && continue
    local operands = [c for r in rels for c in SC.allCrefNames(r) if isAlgebraic(c)]
    isempty(operands) && continue
    local assigns = _gatherClusterAssigns(weq, simCode)
    assigns === nothing && continue
    #= The members, and what they select: an if-equation on a member whose branches set one
       unknown each to parameter values (the lowered `if off then Goff else 1`). =#
    local ownNames = Set{String}(string(d) for (d, _, _) in assigns)
    for (k, ifEq) in enumerate(simCode.ifEquations)
      any(in(ownNames), ifConds[k]) || continue
      for branch in ifEq.branches, eq in branch.residualEquations
        local unknowns = filter(isUnknown, names(eq.exp))
        length(unknowns) == 1 && union!(ownNames, unknowns)
      end
    end
    foreach(op -> union!(get!(own, op, Set{String}()), ownNames), operands)
  end
  isempty(own) && return false
  #= One pass over the equations (an if-equation branch's with its condition). =#
  local read = Set{String}(); local foreign = Set{String}()
  local visit = ns -> for (op, ownNames) in own
    op in ns || continue
    push!(read, op)
    any(in(ns), ownNames) || push!(foreign, op)
  end
  foreach(eq -> visit(names(eq.exp)), simCode.residualEquations)
  for (k, ifEq) in enumerate(simCode.ifEquations), branch in ifEq.branches, eq in branch.residualEquations
    visit(union(ifConds[k], names(eq.exp)))
  end
  return any(op -> op in read && !(op in foreign), keys(own))
end

"""
The discrete-cluster and self-scheduling lowerings run only the assignments of a when body:
its asserts are not checked there (reported).
"""
_uncheckedWhenAssert(st::SC.ASSERT) =
  @warn "[MTK GEN: when] an assert in a when on a buffered relation or a self-scheduling time when is not checked" condition = string(st.condition)

"""
Whether an expression has an Integer or enumeration value.
"""
function _isIntegralValued(e::SC.Exp)::Bool
  e isa SC.EXP_CREF && return _isIntegralType(e.ty)
  e isa Union{SC.ICONST, SC.ENUM_LITERAL} && return true
  e isa SC.IFEXP && return _isIntegralValued(e.thenExp) || _isIntegralValued(e.elseExp)
  return false
end
_isIntegralType(ty::SC.SType)::Bool =
  ty isa Union{SC.TYPE_INTEGER, SC.TYPE_ENUM} || (ty isa SC.TYPE_ARRAY && _isIntegralType(ty.elementType))

"""
A synthesized when cluster's ordered (discrete, right side, numeric) assignments; `nothing`
when it assigns nothing. A single-member cluster has one entry; a coupled FSM cluster has the
body in topological order. The cluster paths are the only ones for such a when
(createCallbackCode), so what they cannot run is refused. Numeric (else Boolean) by the
variable's type, else by the type of the assigned reference or value: an alias elimination
can make a Boolean variable (a gate's Logic input) the member that a clock's `y = if ...
then '0' else '1'` sets.
"""
function _gatherClusterAssigns(weq::SC.WHEN_EQUATION, simCode::SC.SIM_CODE)::Union{Nothing, Vector{Tuple{Symbol, SC.Exp, Bool}}}
  weq.whenEquation.elsewhenPart === nothing ||
    unsupported("an elsewhen of a when on a buffered relation (a discrete cluster)", weq.whenEquation.condition)
  local assigns = Tuple{Symbol, SC.Exp, Bool}[]
  for st in weq.whenEquation.body
    st isa SC.ASSERT && (_uncheckedWhenAssert(st); continue)
    st isa SC.ASSIGN || unsupported("this statement in a when on a buffered relation (a discrete cluster)", st)
    local leftName = SC.crefNameOf(st.left)
    local entry = leftName === nothing ? nothing : get(simCode.stringToSimVarHT, leftName, nothing)
    entry === nothing && unsupported("an assignment to a non-variable in a when on a buffered relation (a discrete cluster)", st)
    local var = last(entry)
    local isInt = !(var.elementType isa SC.TYPE_BOOL) || _isIntegralValued(st.left) || _isIntegralValued(st.right)
    push!(assigns, (Symbol(var.name), st.right, isInt))
  end
  return isempty(assigns) ? nothing : assigns
end

#= ==================================================================================== =#
#= Self-scheduling time whens                                                           =#
#= ==================================================================================== =#

"""
The relations comparing `time` against a `pre()` value (a self-scheduling time event) of an
or-chain. False when a disjunct is neither such a relation nor initial(): its trigger would
be lost.
"""
function _collectSelfSchedRels!(rels::Vector{SC.RELATION}, e::SC.Exp)::Bool
  if e isa SC.RELATION && (_isTimeCref(e.exp1) || _isTimeCref(e.exp2)) && (_isPreCref(e.exp1) || _isPreCref(e.exp2))
    push!(rels, e)
    return true
  elseif e isa SC.LBINARY && e.op === SC.OP_OR
    local okA = _collectSelfSchedRels!(rels, e.exp1)
    local okB = _collectSelfSchedRels!(rels, e.exp2)
    return okA && okB
  end
  return SC.isBuiltin(e, :initial)
end

"""
The relations of a self-scheduling time when (the time tables' `when {time >=
pre(nextEvent), initial()}`, its initial() split off into an initial algorithm): the `time ⋚
pre(x)` relations of its condition. Another trigger or an elsewhen is refused: the general
path cannot evaluate such a condition either.
"""
function _selfSchedulingTimeRels(weq::SC.WHEN_EQUATION)::Vector{SC.RELATION}
  local cond = weq.whenEquation.condition
  local rels = SC.RELATION[]
  local onlyThese = _collectSelfSchedRels!(rels, cond)
  isempty(rels) && return rels
  onlyThese || unsupported("a self-scheduling time when with another trigger", cond)
  weq.whenEquation.elsewhenPart === nothing || unsupported("an elsewhen of a self-scheduling time when", cond)
  return rels
end

"""
(functionExpr, observedNT, modifiedNT) of the ImperativeAffect of a self-scheduling
time-event when. Each assignment `x := rhs` is recomputed imperatively (`_affectValue`:
time as integrator.t, a table handle the module global, an external call resolved). The
table residual's pre(x) lowers to x, omc's state between events: the table's left limit for
t >= x, the new segment after this affect. `atInit` lowers `initial()` to true for the run at
the start of a solve. The state before the instant is taken before the affect writes (pre()
of what it sets, read by a when it triggers at the same instant: `e = edge(b)` with b set
here stayed false).
"""
function _selfSchedAffectParts(weq::SC.WHEN_EQUATION, simCode::SC.SIM_CODE; atInit::Bool = false)
  local obsAcc = Dict{Symbol, Symbol}()
  local stmts = Expr[]
  local retKws = Expr[]
  local modNames = Symbol[]
  local subst = Dict{Symbol, Any}()
  for st in weq.whenEquation.body
    st isa SC.ASSERT && (atInit || _uncheckedWhenAssert(st); continue)
    st isa SC.ASSIGN || unsupported("this statement in a self-scheduling time when", st)
    st.left isa SC.EXP_CREF || unsupported("this assignment in a self-scheduling time when", st)
    local xn = SC.crefName(st.left.cref)
    local xsym = Symbol(xn)
    local vsym = Symbol("_v_", xn)
    push!(stmts, :(local $(vsym) = $(_affectValue(st.right, obsAcc, simCode; initVal = atInit, subst))))
    push!(retKws, Expr(:kw, xsym, vsym))
    push!(modNames, xsym)
    subst[xsym] = vsym
  end
  local retNT = Expr(:tuple, Expr(:parameters, retKws...))
  local fexpr = :((modified, observed, ctx, integrator) -> begin
                    $(atInit ? :() : :(OMBackend.CodeGeneration.instantPre(integrator)))
                    $(stmts...)
                    $(retNT)
                  end)
  local obsNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, k, v) for (k, v) in obsAcc]...))
  local modNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, d, d) for d in modNames]...))
  return (fexpr, obsNT, modNT)
end

"""
Whether a self-scheduling when's body reads no pre() of a variable it sets: run again at a
fixed time it gives the same values.
"""
function _selfSchedIdempotent(weq::SC.WHEN_EQUATION)::Bool
  local body = weq.whenEquation.body
  local assigned = Set{String}(SC.crefName(st.left.cref) for st in body if st isa SC.ASSIGN && st.left isa SC.EXP_CREF)
  return !any(st -> st isa SC.ASSIGN && _expHasPreOf(st.right, assigned), body)
end

"""
MTK SymbolicContinuousCallbacks for self-scheduling time-event whens. The crossing `time -
nextTimeEvent` fires when time reaches the held discrete; an ImperativeAffect runs the body
again (it updates only integrator.u, never an AffectSystem, so it does not pull the
table-fed continuous network into an unsolvable callback).
"""
function createSelfSchedulingTimeWhenEvents(simCode::SC.SIM_CODE)::Vector{Expr}
  local events = Expr[]
  for weq in simCode.whenEquations
    local rels = _selfSchedulingTimeRels(weq)
    isempty(rels) && continue
    local (fn, obs, modN) = _selfSchedAffectParts(weq, simCode)
    isempty(modN.args[1].args) && continue
    #= A when with initial() (the time tables' `{time >= pre(nextTimeEvent), initial()}`) runs
       in the initial algorithm; at the start of every solve again only where that is
       idempotent (its body reads no pre() of what it sets): a solve of the problem itself
       (remake) skips the initial algorithm, and the table's C object kept its end state (z(1) =
       2.77, OpenModelica 2.47). Not a counter (`n = pre(n) + 1` ran twice: n = 2, OpenModelica
       1), and not a when without initial() (it ran at the start; OpenModelica: first at its
       time). =#
    local initAffect = if weq.attr.alsoInitial && _selfSchedIdempotent(weq)
      local (fnI, obsI, modI) = _selfSchedAffectParts(weq, simCode; atInit = true)
      :(ModelingToolkit.ImperativeAffect($(fnI), $(modI); observed = $(obsI), skip_checks = true))
    else
      nothing
    end
    for (k, rel) in enumerate(rels)
      #= transformToMTKContinuousCondition emits `pre(nextTimeEvent) - time` for `time >=
         pre(nextTimeEvent)`, which falls through zero as time reaches the event. Negated so the
         crossing RISES through zero exactly when the Modelica condition becomes true (the
         when's rising edge), fired only on that positive edge (affect_neg = nothing). A
         monotone time event is one-directional, so a single edge is correct and avoids
         double-firing. =#
      local zc = transformToMTKContinuousCondition(rel, simCode)
      push!(events, :(ModelingToolkit.SymbolicContinuousCallback(
        (-($(zc)) ~ 0),
        ModelingToolkit.ImperativeAffect($(fn), $(modN); observed = $(obs), skip_checks = true);
        affect_neg = nothing,
        initialize = $(k == 1 ? initAffect : nothing),
        rootfind = SciMLBase.RightRootFind,
        reinitializealg = SciMLBase.NoInit())))
    end
  end
  return events
end

"""
The event iteration over the buffered relations (relationRefresh.jl): after every step, a
relation that disagrees with the state is flipped with the hysteresis rule, and so on until
nothing changes. Emitted with no if-equation relations too: whens on a relation among the
callbacks are iterated as well (none of either: a no-op). In the latest world, like the
event list: the symbolic variables are globals bound while the model is built.
"""
function emitRelationRefresh(relations::Vector)::Expr
  local entries = [:(($(QuoteNode(sym)), $(zc), $(scale))) for (sym, zc, scale) in relations]
  return :(callbacks = OMBackend.CodeGeneration.withRelationRefresh(callbacks, problem,
                                                                     $(QuoteNode(ZC_HYSTERESIS)),
                                                                     Base.invokelatest(() -> Any[$(entries...)])))
end

#= ==================================================================================== =#
#= Asserts                                                                              =#
#= ==================================================================================== =#

"""
The asserts of equation sections (simCode.asserts) as one callback that checks them after
the initialization and after each accepted step (asserts.jl). Conditions and messages are
lowered like event affects: every variable is read through `observed`, `time` from the
integrator. An assert whose expression cannot be lowered is reported and left out. Names,
not the symbolic variables, are observed: a variable the compiler eliminated has no binding
in the generated code, but may still be observed by name.
"""
function emitAssertCallback(simCode::SC.SIM_CODE)::Expr
  isempty(simCode.asserts) && return Expr(:block)
  local entries = Expr[]
  for a in simCode.asserts
    local obsAcc = Dict{Symbol, Symbol}()
    local (cond, crossings) = try
      (_assertConditionExpr(a.condition, obsAcc, simCode), _assertCrossingExprs(a.condition, obsAcc, simCode))
    catch err
      OMBackend._fallback(err, :assertCondition; only = LoweringFailure, impact = :result)
      @warn "[MTK GEN: asserts] an assert cannot be checked at run time; it is left out" condition = string(a.condition) exception = err
      continue
    end
    local msg = _assertMessageExpr(a.message, obsAcc, simCode)
    local timeVarying = _exprMentions(cond, :integrator) || any(keys(obsAcc)) do k
      local e = get(simCode.stringToSimVarHT, string(k), nothing)
      e === nothing || !(last(e).varKind isa Union{SC.PARAMETER, SC.ARRAY_PARAMETER})
    end
    local obsNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, k, QuoteNode(v)) for (k, v) in obsAcc]...))
    push!(entries, :(OMBackend.CodeGeneration.ModelicaAssert($(obsNT),
                                                             (observed, integrator) -> $(cond),
                                                             (observed, integrator) -> $(msg),
                                                             $(AlgorithmicCodeGeneration.isWarningAssertionLevel(a.level)), $(timeVarying),
                                                             $(string(a.condition)),
                                                             $(Expr(:tuple, (:((observed, integrator) -> $(c)) for c in crossings)...)))))
  end
  isempty(entries) && return Expr(:block)
  return :(callbacks = OMBackend.CodeGeneration.withAssertCallback(callbacks, problem, [$(entries...)]))
end

_isEffectCall(e::SC.Exp)::Bool = e isa SC.CALL && e.ty isa SC.TYPE_NORETCALL

"""
The crossings of an assert condition's ordering relations on numbers (`a - b`), where the
condition can change within a step (asserts.jl). One is 1.0 where the condition's
short-circuit does not reach its relation (`x <= 0 or sqrt(x) < 2`: sqrt(x) of a negative
threw). Not under noEvent(), nor inside another expression (a call's argument, a
reduction).
"""
function _assertCrossingExprs(cond::SC.Exp, obsAcc::Dict{Symbol, Symbol}, simCode::SC.SIM_CODE)::Vector{Any}
  local out = Any[]
  local both = (r, c) -> r === true ? c : :($(r) && $(c))
  local walk = nothing
  walk = function (e::SC.Exp, reached)
    if e isa SC.LBINARY
      walk(e.exp1, reached)
      local lhs = _affectBool(e.exp1, obsAcc, simCode)
      walk(e.exp2, both(reached, e.op === SC.OP_AND ? lhs : :(!$(lhs))))
    elseif e isa SC.LUNARY
      walk(e.exp, reached)
    elseif e isa SC.IFEXP
      walk(e.cond, reached)
      local c = _affectBool(e.cond, obsAcc, simCode)
      walk(e.thenExp, both(reached, c))
      walk(e.elseExp, both(reached, :(!$(c))))
    elseif e isa SC.RELATION && e.op in (SC.OP_LESS, SC.OP_LESSEQ, SC.OP_GREATER, SC.OP_GREATEREQ) &&
           !(SC.scalarType(SC.typeOf(e.exp1)) isa Union{SC.TYPE_STRING, SC.TYPE_BOOL})
      local zc = :(Float64($(_affectValue(e.exp1, obsAcc, simCode))) - Float64($(_affectValue(e.exp2, obsAcc, simCode))))
      push!(out, reached === true ? zc : :($(reached) ? $(zc) : 1.0))
    end
    nothing
  end
  walk(cond, true)
  return out
end

"""
An assert's condition; a call equation for its effects runs and holds, one in an
if-equation's branch, `if guard then call else true`, where the guard holds.
"""
function _assertConditionExpr(cond::SC.Exp, obsAcc::Dict{Symbol, Symbol}, simCode::SC.SIM_CODE)
  _isEffectCall(cond) && return :($(_effectCallExpr(cond, obsAcc, simCode)); true)
  if cond isa SC.IFEXP && _isEffectCall(cond.thenExp)
    return :((if $(_affectBool(cond.cond, obsAcc, simCode))
                $(_effectCallExpr(cond.thenExp, obsAcc, simCode))
              end); true)
  end
  return _affectBool(cond, obsAcc, simCode)
end

"""
A call for its effects, its arguments read as values: a String as it is, a Boolean decoded
from the arithmetic encoding, an array element by element (a literal one too: its elements
may be variables).
"""
function _effectCallExpr(call::SC.CALL, obsAcc::Dict{Symbol, Symbol}, simCode::SC.SIM_CODE)::Expr
  local arg = nothing
  arg = a::SC.Exp -> a isa SC.SCONST ? a.value :
    a isa SC.ARRAY_EXP ? Expr(:vect, (arg(e) for e in a.elements)...) :
    _isBooleanExp(a) ? _affectBool(a, obsAcc, simCode) : _affectValue(a, obsAcc, simCode)
  return Expr(:call, call.name, (arg(a) for a in call.args)...)
end

_isBooleanExp(e::SC.Exp)::Bool =
  e isa Union{SC.BCONST, SC.RELATION, SC.LBINARY, SC.LUNARY} || (e isa Union{SC.EXP_CREF, SC.CALL} && e.ty isa SC.TYPE_BOOL)

"""
An assert's message: string literals, `+` concatenation and String(x) of variables;
anything else is shown as the Modelica expression. A String parameter is a module-level
binding (createStringParameterAssignments), not a variable of the simulation (read through
`observed`, the assert was left out as reading a variable the simulation does not keep).
"""
function _assertMessageExpr(msg::SC.Exp, obsAcc::Dict{Symbol, Symbol}, simCode::SC.SIM_CODE)
  local part = nothing
  part = e::SC.Exp -> begin
    e isa SC.SCONST && return e.value
    (e isa SC.BINARY && e.op === SC.OP_ADD) && return :(string($(part(e.exp1)), $(part(e.exp2))))
    (e isa SC.EXP_CREF && _isStringBinding(SC.crefName(e.cref), simCode)) &&
      return Symbol(last(simCode.stringToSimVarHT[SC.crefName(e.cref)]).name)
    return :(string($(_affectValue(e, obsAcc, simCode))))
  end
  return try
    part(msg)
  catch err
    #= The assert is still checked; its message is the Modelica text. =#
    OMBackend._fallback(err, :assertMessage; only = LoweringFailure, impact = :result)
    string(msg)
  end
end

_isStringBinding(name::String, simCode::SC.SIM_CODE)::Bool =
  (local e = get(simCode.stringToSimVarHT, name, nothing); e !== nothing && last(e).varKind isa SC.STRING)

_exprMentionsPrefix(@nospecialize(e), prefix::String) =
  (e isa Symbol && startswith(string(e), prefix)) || (e isa Expr && any(a -> _exprMentionsPrefix(a, prefix), e.args))

_exprMentions(@nospecialize(e), s::Symbol) = e === s || (e isa Expr && any(a -> _exprMentions(a, s), e.args))

#= ==================================================================================== =#
#= Discrete clusters                                                                    =#
#= ==================================================================================== =#

"""
Whether a relation has no continuous-time operand: only time, parameters, constants and
discretes. Its crossing gets no hysteresis, so time events stay exact (as the if-equation
branches keep pure time events exact).
"""
function _withoutContinuousOperand(rel::SC.Exp, simCode::SC.SIM_CODE)::Bool
  for name in SC.allCrefNames(rel)
    name == "time" && continue
    local entry = get(simCode.stringToSimVarHT, name, nothing)
    entry === nothing && return false
    (SC.isDiscrete(last(entry)) || SC.isParameter(last(entry))) || return false
  end
  return true
end

"""
The variables a when reads through pre(), in first-occurrence order; `edge(b)` and
`change(b)` read pre(b).
"""
function _clusterPreNames(cond::SC.Exp, assigns::Vector{Tuple{Symbol, SC.Exp, Bool}})::Vector{String}
  local names = String[]
  local visit = function (e::SC.Exp)
    local n = _preReadName(e)
    if n === nothing && (SC.isBuiltin(e, :edge) || SC.isBuiltin(e, :change)) && first(e.args) isa SC.EXP_CREF
      n = SC.crefName(first(e.args).cref)
    end
    (n === nothing || n in names) || push!(names, n)
  end
  SC.foreachExp(visit, cond)
  for (_, rhs, _) in assigns
    SC.foreachExp(visit, rhs)
  end
  return names
end

#= The equations as a graph for _clusterCoupled: each equation's variable names (an
   if-equation: its conditions and all its branches' equations together), the equations each
   name occurs in, and the names a re-solve of the algebraic unknowns keeps: time, parameters
   and when-assigned discretes. Not the variables under der(): index reduction can make one
   algebraic (a dummy derivative), and a re-solve then moves it. =#
struct _EquationGraph
  names::Vector{OrderedSet{String}}
  occurs::Dict{String, Vector{Int}}
  fixed::Set{String}
end

function _equationGraph(simCode::SC.SIM_CODE)::_EquationGraph
  local names = OrderedSet{String}[SC.crefNames!(OrderedSet{String}(), eq.exp) for eq in simCode.residualEquations]
  for ifEq in simCode.ifEquations
    local ns = OrderedSet{String}()
    for b in ifEq.branches
      SC.crefNames!(ns, b.condition)
      foreach(eq -> SC.crefNames!(ns, eq.exp), b.residualEquations)
    end
    push!(names, ns)
  end
  local occurs = Dict{String, Vector{Int}}()
  for (k, ns) in enumerate(names), n in ns
    push!(get!(occurs, n, Int[]), k)
  end
  local fixed = Set{String}(["time"])
  for (n, (_, v)) in simCode.stringToSimVarHT
    SC.isParameter(v) && push!(fixed, n)
  end
  union!(fixed, _collectWhenAssignedNames(simCode))
  return _EquationGraph(names, occurs, fixed)
end

"""
Whether solving the algebraic unknowns again can change what a cluster reads: a read
reachable from its members through the equations over the names a re-solve moves. An ideal
diode's `s` in `off = s < 0` is; a switched-capacitor clock's `time >= pulseStart` and a
constant resistance are not, and such a cluster settles without a solve of its own: the
event iteration solves once after the sweep (CauerLowPassSC: 11 solves per clock event ->
1).
"""
function _clusterCoupled(members::Vector{String}, reads::OrderedSet{String}, g::_EquationGraph)::Bool
  local reached = Set{String}(members)
  local frontier = copy(members)
  local visited = falses(length(g.names))
  while !isempty(frontier)
    for k in get(g.occurs, pop!(frontier), Int[])
      visited[k] && continue
      visited[k] = true
      for n in g.names[k]
        (n in g.fixed || n in reached) && continue
        push!(reached, n)
        push!(frontier, n)
      end
    end
  end
  return any(n -> n in reached && !(n in members), reads)
end

"""
One DiscreteCluster (discreteClusters.jl) for a when on the relations `rels` that assigns
`assigns`. Every distinct relation gets a buffer; `==` and `<>` (a crossing function that
jumps between -0.5 and 0.5) and relations without a continuous-time operand have no
hysteresis. A when lifted from discrete equations holds at every pass of an event (its
equations do); another fires when its condition does. Later members read an earlier one's
new value (the body is in topological order).
"""
function _discreteClusterSpec(cond::SC.Exp, rels::Vector{SC.RELATION}, assigns::Vector{Tuple{Symbol, SC.Exp, Bool}},
                              simCode::SC.SIM_CODE, table::Bool, graph::_EquationGraph)::Expr
  local buffered = unique(rels)
  local relPins = Dict{SC.Exp, Any}(r => :(_rel[$k]) for (k, r) in enumerate(buffered))
  local relPre = Dict{SC.Exp, Any}(r => :(_relPre[$k]) for (k, r) in enumerate(buffered))
  local preNames = _clusterPreNames(cond, assigns)
  local preSubst = Dict{Symbol, Any}(Symbol(n) => :(_pre[$i]) for (i, n) in enumerate(preNames))
  local obsAcc = Dict{Symbol, Symbol}()
  local subst = Dict{Symbol, Any}()
  local stmts = Expr[]
  for (d, rhs, isInt) in assigns
    local vsym = Symbol("_v_", d)
    local valExpr = isInt ?
      :(Float64($(_affectValue(rhs, obsAcc, simCode; initVal = :_init, subst, relPins, preSubst, relPre)))) :
      :($(_affectBool(rhs, obsAcc, simCode; initVal = :_init, subst, relPins, preSubst, relPre)) ? 1.0 : 0.0)
    push!(stmts, :(local $(vsym) = $(valExpr)))
    subst[d] = vsym
  end
  local fires = _isSynthesizedChangeCondition(cond) ? true :
    _affectBool(cond, obsAcc, simCode; initVal = :_init, relPins, preSubst, relPre)
  local operands = sort!(collect(keys(obsAcc)))
  local observedNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, n, :(_o[$i])) for (i, n) in enumerate(operands)]...))
  local body = :((integrator, _o, _pre, _rel, _relPre, _init) -> begin
                   local observed = $(observedNT)
                   $(fires) || return nothing
                   $(stmts...)
                   ($([Symbol("_v_", d) for (d, _, _) in assigns]...),)
                 end)
  local reads = Any[operands..., Symbol.(preNames)...]
  for r in buffered
    push!(reads, transformToMTKContinuousCondition(r, simCode), _conditionScaleExpr(r, simCode))
  end
  local strict = Bool[r.op in (SC.OP_LESS, SC.OP_GREATER) for r in buffered]
  local exact = Bool[_relationZeroSet(r) === nothing || _withoutContinuousOperand(r, simCode) for r in buffered]
  local eventOnly = Bool[_withoutContinuousOperand(r, simCode) && !_expMentionsTime(r) for r in buffered]
  local atStart = !_condHasInitial(cond) ? 0 : (table ? 2 : 1)
  local memberNames = String[string(d) for (d, _, _) in assigns]
  local readNames = OrderedSet{String}(string(o) for o in operands)
  foreach(r -> SC.crefNames!(readNames, r), buffered)
  local coupled = _clusterCoupled(memberNames, readNames, graph)
  return :(OMBackend.CodeGeneration.DiscreteCluster($(memberNames),
                                                   Any[$([d for (d, _, _) in assigns]...)],
                                                   Any[$(reads...)], $(length(operands)), $(length(preNames)),
                                                   $(strict), $(exact), $(body), $(atStart), $(table), $(coupled),
                                                   $(eventOnly)))
end

"""
    emitDiscreteClusters(simCode) -> Expr

For a model with discrete clusters (`_usesDiscreteClusters`), the whens over relations of
coupled discretes are discrete clusters: a continuous callback each locates the crossings of
their relations, and the event iteration (emitted after this) evaluates them. In the latest
world, like the event list: the symbolic variables are globals bound while the model is built.
"""
function emitDiscreteClusters(simCode::SC.SIM_CODE)::Expr
  local specs = _discreteClusterSpecs(simCode)
  isempty(specs) && return Expr(:block)
  return :(callbacks = OMBackend.CodeGeneration.withDiscreteClusters(callbacks, problem,
                                                                      Base.invokelatest(() -> Any[$(specs...)])))
end

"""
The DiscreteCluster constructions of the model's discrete clusters (emitDiscreteClusters);
empty when it has none.
"""
function _discreteClusterSpecs(simCode::SC.SIM_CODE)::Vector{Expr}
  _usesDiscreteClusters(simCode) || return Expr[]
  local table = _modelHasTableClusters(simCode)
  local specs = Expr[]
  local relationOf = _condVarRelations(simCode)
  local graph = nothing
  for weq in simCode.whenEquations
    local rels = _extractChangeRelations(weq.whenEquation.condition, simCode)
    rels === nothing && continue
    local assigns = _gatherClusterAssigns(weq, simCode)
    assigns === nothing && continue
    if _isSynthesizedChangeCondition(weq.whenEquation.condition)
      (rels, assigns) = _inlineReadBooleanRelations(rels, assigns, relationOf, simCode)
    end
    graph === nothing && (graph = _equationGraph(simCode))
    push!(specs, _discreteClusterSpec(weq.whenEquation.condition, rels, assigns, simCode, table, graph))
  end
  return specs
end

"""
    createDiscreteBoolWhenEvents(simCode)

MTK SymbolicContinuousCallbacks for the synthesized discrete-Boolean whens of a model without
discrete clusters: one callback per relation zero set, whose affect rewrites the held discrete
from its defining expression, at the (post-rootfind) event point, so it is direction-correct.
Each callback rewrites the WHOLE ordered cluster with THIS group's relations pinned to their
post-crossing values (the others at their current value), so a coupled FSM recomputes
consistently on any member crossing, without the `f ≈ 0` ambiguity. transformToMTKContinuousCondition
normalises zc so relation-true ⟺ zc < 0 for the group's first relation: the `=>` affect is
the up-crossing (the relation becomes false), affect_neg the down-crossing (it becomes true).
"""
function createDiscreteBoolWhenEvents(simCode::SC.SIM_CODE)::Vector{Expr}
  local events = Expr[]
  local usesClusters = _usesDiscreteClusters(simCode)
  for weq in simCode.whenEquations
    local rels = _extractChangeRelations(weq.whenEquation.condition, simCode)
    rels === nothing && continue
    #= `edge(b)` fires on the rising transition only (relation false->true); `change` and the
       gate lift fire on both. =#
    local isEdge = _isEdgeWhenCondition(weq.whenEquation.condition)
    local assigns = _gatherClusterAssigns(weq, simCode)
    assigns === nothing && continue
    #= Discrete clusters are evaluated by the event iteration (emitDiscreteClusters). =#
    usesClusters && continue
    #= Initialize affect: every discrete from its full right side at t0. Only whens whose
       condition carries an `initial()` term (the synthesized lifter conditions) run their
       body at t0; user whens with plain relation conditions must not (Trapezoid `T_start =
       time` at t0 would destroy a negative-startTime phase, friction would mis-latch). =#
    local hasInit = _condHasInitial(weq.whenEquation.condition)
    #= Relations with the same zero set (`w_rel <= 0` and `w_rel > 0`) get ONE callback whose
       crossing pins all of them. With one callback each, both sat on the same root with
       opposite signs; where the integrator stopped exactly on it they fired alternately, each
       pinning its own relation and reading the other from the state at the root, and flipped
       the cluster back and forth without time advancing (MSL Rotational
       OneWayClutchDisengaged, maxiters). Rising-edge-only whens keep one callback per
       relation. =#
    local groups = isEdge ? Vector{SC.RELATION}[SC.RELATION[r] for r in rels] : _groupRelationsByZeroSet(rels)
    #= A synthesized cluster that reads its own pre() values iterates to a fixpoint within the
       event (_eventIterAffectParts); the others need a single pass. =#
    local selfPre = _isSynthesizedChangeCondition(weq.whenEquation.condition) && _clusterReadsOwnPre(assigns)
    local lowerable = (selfPre || hasInit) && _eventIterLowerable(assigns, simCode)
    local eventIteration = selfPre && lowerable
    #= The t0 pass is imperative wherever the body lowers. As equations, ModelingToolkit
       solves it together with the whole algebraic system, every held discrete an unknown of
       that solve, and at t0 that system sits on its singular points (a Mean block's sqrt(y) at
       y = 0, a zero current's angle): the solve fails (MTK's UnsolvableCallbackError). The
       crossings keep the equation form, whose solve also moves the algebraic unknowns after a
       switch (MSL CauerLowPassSC's switched capacitors). =#
    local affInit = !hasInit ? nothing : lowerable ?
      _imperativeClusterAffect(assigns, simCode; initVal = true, iterate = selfPre) :
      :([$([_discreteAffectEqInit(d, r, ii, simCode) for (d, r, ii) in assigns]...)])
    for (relIdx, grp) in enumerate(groups)
      local rel = grp[1]
      local zc = transformToMTKContinuousCondition(rel, simCode)
      local pinsFalse = _zeroSetPins(grp, rel, false)
      local pinsTrue = _zeroSetPins(grp, rel, true)
      #= The `initial()` term: only the first group's callback carries it, so the discretes
         are set once at t0 (no double-apply). =#
      local initKw = relIdx == 1 && affInit !== nothing ? Expr[Expr(:kw, :initialize, affInit)] : Expr[]
      if eventIteration
        local affUp = _imperativeClusterAffect(assigns, simCode; relPins = pinsFalse)
        local affDn = _imperativeClusterAffect(assigns, simCode; relPins = pinsTrue)
        push!(events, :(ModelingToolkit.SymbolicContinuousCallback(
          ($(zc) ~ 0), $(affUp);
          affect_neg = $(affDn),
          $(initKw...),
          reinitializealg = SciMLBase.NoInit())))
      else
        #= `edge(b)`: the body runs when the relation becomes TRUE (the down-crossing of zc,
           affect_neg); the up-crossing does nothing. =#
        local affFalse = isEdge ? Expr[] :
          Expr[_discreteAffectEq(d, r, pinsFalse, ii, simCode) for (d, r, ii) in assigns]
        local affTrue = Expr[_discreteAffectEq(d, r, pinsTrue, ii, simCode) for (d, r, ii) in assigns]
        push!(events, :(ModelingToolkit.SymbolicContinuousCallback(
          ($(zc) ~ 0) => $(isEdge ? :(Any[]) : :([$(affFalse...)]));
          affect_neg = [$(affTrue...)],
          $(initKw...),
          reinitializealg = SciMLBase.NoInit())))
      end
    end
  end
  return events
end

#= ==================================================================================== =#
#= When statements                                                                      =#
#= ==================================================================================== =#

"""
The target of field `f` of a record target of a tuple assignment: the field's variable
(`r_f`), or an omitted output's.
"""
function _recordFieldTarget(target::SC.Exp, f::String)::SC.Exp
  target isa SC.WILD && return target
  target isa SC.EXP_CREF || unsupported("this record target of a tuple assignment in a when", target)
  return SC.EXP_CREF(SC.SimCref(SC.crefName(target.cref) * COMPONENT_SEPARATOR * f), SC.TYPE_REAL())
end

"""
`integrator.u[i] = value` of a when's assignment to the variable `var`, and the body's
local of it (a later statement read the value from before: Buildings' occupant windows,
`(ran, state) = random(pre(state)); on = ran < p` decided on the previous sample's
number).
"""
function _stateAssignment(var::SC.SIMVAR, value)::Expr
  quote
    $(MTK_CodeGenerationUtil.STATE_INDEX) = lookuptableStates[Symbol($(var.name))]
    integrator.u[$(MTK_CodeGenerationUtil.STATE_INDEX)] = $(value)
    $(Symbol(var.name)) = integrator.u[$(MTK_CodeGenerationUtil.STATE_INDEX)]
  end
end

"""
A target of a tuple assignment in a when: a variable, an array of them, or `_`. A target
missing from the variable table would be bound as a local of the affect, its value lost.
"""
function _emitWhenTupleElementAssignMTK!(res::Vector{Expr}, lhs::SC.Exp, rhsAccess::Union{Symbol, Expr}, simCode::SC.SIM_CODE)::Vector{Expr}
  if lhs isa SC.WILD
    return res
  elseif lhs isa SC.EXP_CREF
    local entry = get(simCode.stringToSimVarHT, SC.crefName(lhs.cref), nothing)
    entry === nothing && unsupported("a tuple target in a when that is no variable", lhs)
    push!(res, _stateAssignment(last(entry), rhsAccess))
  elseif lhs isa SC.ARRAY_EXP
    for (i, elem) in enumerate(lhs.elements)
      _emitWhenTupleElementAssignMTK!(res, elem, :($rhsAccess[$i]), simCode)
    end
  else
    unsupported("tuple-LHS element in a when statement", lhs)
  end
  return res
end

"""
    createWhenStatementsMTK(statements, simCode)

The statements of a when body in a callback's affect: an assignment writes the state vector
(a tuple's targets each an output, a record output's fields in place: a generated function
returns them so, an omitted one skipped), a reinit too (noteReinit!), terminate() stops the
integration, an assert of level error throws (as an equation's: asserts.jl), of level warning
reports.
"""
function createWhenStatementsMTK(statements::Vector{SC.Statement}, simCode::SC.SIM_CODE)::Vector{Expr}
  local res = Expr[]
  @debug "[MTK GEN: when] createWhenStatementsMTK" statements=length(statements)
  for wStmt in statements
    if wStmt isa SC.ASSIGN && wStmt.left isa SC.TUPLE
      local tupSym = gensym(:tupResult)
      push!(res, :(local $tupSym = $(expToJuliaExpMTK(wStmt.right, simCode))))
      local outputTypes = wStmt.right isa SC.CALL && wStmt.right.ty isa SC.TYPE_TUPLE ? wStmt.right.ty.types : nothing
      local pos = 0
      for (k, elem) in enumerate(wStmt.left.elements)
        local fields = outputTypes === nothing ? String[] : AlgorithmicCodeGeneration._recordFieldNames(outputTypes[k])
        if isempty(fields)
          pos += 1
          _emitWhenTupleElementAssignMTK!(res, elem, :($tupSym[$pos]), simCode)
        else
          for f in fields
            pos += 1
            _emitWhenTupleElementAssignMTK!(res, _recordFieldTarget(elem, f), :($tupSym[$pos]), simCode)
          end
        end
      end
    elseif wStmt isa SC.ASSIGN
      local leftName = SC.crefNameOf(wStmt.left)
      leftName === nothing && unsupported("this assignment target in a when", wStmt.left)
      push!(res, _stateAssignment(last(simCode.stringToSimVarHT[leftName]), expToJuliaExpMTK(wStmt.right, simCode)))
    elseif wStmt isa SC.REINIT
      local var = last(simCode.stringToSimVarHT[SC.crefName(wStmt.stateVar)])
      push!(res, quote
              $(MTK_CodeGenerationUtil.STATE_INDEX) = lookuptableStates[Symbol($(var.name))]
              OMBackend.CodeGeneration.noteReinit!(integrator, $(MTK_CodeGenerationUtil.STATE_INDEX))
              integrator.u[$(MTK_CodeGenerationUtil.STATE_INDEX)] = $(expToJuliaExpMTK(wStmt.value, simCode))
            end)
    elseif wStmt isa SC.TERMINATE
      push!(res, quote
              @info "Modelica terminate() reached" message=$(expToJuliaExpMTK(wStmt.message, simCode))
              OMBackend.DifferentialEquations.terminate!(integrator)
            end)
    elseif wStmt isa SC.NORETCALL
      push!(res, quote $(expToJuliaExpMTK(wStmt.exp, simCode)) end)
    elseif wStmt isa SC.ASSERT
      local msgExpr = expToJuliaExpMTK(wStmt.message, simCode)
      local violated = if AlgorithmicCodeGeneration.isWarningAssertionLevel(wStmt.level)
        :(@warn string("Assertion violated at time ", integrator.t, ": ", $(msgExpr)))
      else
        :(throw(OMBackend.CodeGeneration.ModelicaAssertionError(integrator.t, string($(msgExpr)), $(string(wStmt.condition)))))
      end
      #= The lowering encodes and/or/not arithmetically: a number, not a Bool. =#
      push!(res, quote
              local _holds = $(expToJuliaExpMTK(wStmt.condition, simCode))
              if !(_holds isa Bool ? _holds : _holds != 0)
                $(violated)
              end
            end)
    else
      unsupported("when-statement variant", wStmt)
    end
  end
  return res
end

"""
Whether a when-equation's condition is Modelica's `terminal()`.
"""
_isTerminalWhen(eq::SC.WHEN_EQUATION)::Bool = SC.isBuiltin(eq.whenEquation.condition, :terminal)

"""
Post-solve runner for `when terminal()` bodies, or `nothing` when the model has none (so
models without a terminal event are unchanged). The bodies reuse `createWhenStatementsMTK`
by mocking `integrator` from the final solution point: writes land in `_sol.u[end]` using
the same state-index convention the discrete-callback affects rely on. Runs only on success.
String parameters are module-level constants: referenced directly, not shadowed by a
nonexistent state lookup. The external functions the body calls are bound to their
OMBackend.CodeGeneration RTG wrapper: the bare model-module name is the @register_symbolic
binding (symbolic only), which has no method for concrete runtime arguments.
"""
function createTerminalBodyRunner(simCode::SC.SIM_CODE)
  local terminalWhens = filter(_isTerminalWhen, simCode.whenEquations)
  isempty(terminalWhens) && return nothing
  local modelFns = OrderedSet(replace(f.name, "." => "_") for f in simCode.functions)
  local calledNames = OrderedSet{String}()
  local perWhen = Expr[]
  for eq in terminalWhens
    local body = eq.whenEquation.body
    foreach(s -> collectCalledFunctionNames!(calledNames, s), body)
    for c in _bodyReads(body)
      local entry = get(simCode.stringToSimVarHT, c, nothing)
      entry !== nothing && last(entry).varKind isa SC.STRING && continue
      push!(perWhen, Expr(:(=), Symbol(c), getIdxForLookupMTK(c, simCode)))
    end
    append!(perWhen, createWhenStatementsMTK(body, simCode))
  end
  local fnRebinds = Expr[]
  for n in calledNames
    local nn = replace(n, "." => "_")
    nn in modelFns && push!(fnRebinds, :(local $(Symbol(nn)) = OMBackend.CodeGeneration.$(Symbol(nn))))
  end
  return quote
    if _sol.retcode == ModelingToolkit.SciMLBase.ReturnCode.Success
      #= Best-effort: a terminal body runs after a completed, valid solution, so a body we
         cannot evaluate (an unsupported external call) warns rather than discarding the
         result. =#
      try
        let integrator = (u = _sol.u[end], t = _sol.t[end], f = _sol.prob.f, dt = 0.0, p = _sol.prob.p, ps = _sol.prob.ps),
            x = _sol.u[end],
            t = _sol.t[end],
            p = _sol.prob.p,
            lookuptableStates = Dict(sym => i for (i, sym) in enumerate(OMBackend.CodeGeneration.getStatesAsSymbols(_sol.prob.f))),
            lookuptableParams = Dict(sym => i for (i, sym) in enumerate(OMBackend.CodeGeneration.getParametersAsSymbols(_sol.prob.f)))
          local idx = 0
          $(fnRebinds...)
          $(perWhen...)
        end
      catch _terminalErr
        OMBackend._fallback(_terminalErr, :terminalBody)
        @warn "when terminal() body could not be evaluated; returning the completed solution unchanged" exception = _terminalErr
      end
    end
  end
end
