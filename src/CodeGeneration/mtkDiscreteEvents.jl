#= MTK code generation: discrete affects and clusters, the event iteration, asserts, self-scheduling time whens, when statements. =#

#= Lower a DAE boolean expression to an MTK Real (0.0/1.0). Each relation is
   wrapped `ifelse(rel, 1.0, 0.0)` so AND/OR/NOT stay arithmetic on Reals: a bare
   relation is a Julia Bool and `Bool + Bool` is an Int64, illegal in the boolean
   context the affect feeds. pre()/discrete crefs are already 0/1 Reals;
   constants and params fall through to the general emitter. =#
#= Lower an operand inside an event affect. `pre(x)` becomes
   `ModelingToolkit.Pre(x)` — the pre-event value — so an affect that reads a
   discrete's own previous value (e.g. `mode = … pre(mode) …`) is solvable; a
   bare reference would make the written variable appear on its own RHS and MTK
   raises UnsolvableCallbackError. Continuous operands read their current value. =#
Base.@nospecializeinfer function _affectExpToReal(@nospecialize(exp::DAE.Exp), simCode)
  @match exp begin
    DAE.CALL(Absyn.IDENT("pre"), args, _) =>
      :(ModelingToolkit.Pre($(expToJuliaExpMTK(listHead(args), simCode))))
    DAE.BINARY(e1, op, e2) =>
      :($(DAE_OP_toJuliaOperator(op))($(_affectExpToReal(e1, simCode)), $(_affectExpToReal(e2, simCode))))
    DAE.UNARY(op, e) => :($(DAE_OP_toJuliaOperator(op))($(_affectExpToReal(e, simCode))))
    _ => expToJuliaExpMTK(exp, simCode)
  end
end

Base.@nospecializeinfer function _boolDaeToReal(@nospecialize(exp::DAE.Exp), simCode)
  @match exp begin
    DAE.BCONST(b) => (b ? :(1.0) : :(0.0))
    DAE.RELATION(e1, op, e2) => :(ModelingToolkit.ifelse(
        $(DAE_OP_toJuliaOperator(op))($(_affectExpToReal(e1, simCode)), $(_affectExpToReal(e2, simCode))), 1.0, 0.0))
    DAE.LUNARY(DAE.NOT(__), e) => :(1.0 - $(_boolDaeToReal(e, simCode)))
    DAE.LBINARY(e1, DAE.AND(__), e2) =>
      :($(_boolDaeToReal(e1, simCode)) * $(_boolDaeToReal(e2, simCode)))
    DAE.LBINARY(e1, DAE.OR(__), e2) => begin
      local a = _boolDaeToReal(e1, simCode)
      local b = _boolDaeToReal(e2, simCode)
      :($(a) + $(b) - $(a) * $(b))
    end
    DAE.IFEXP(c, t, f) => begin
      local cr = _boolDaeToReal(c, simCode)
      local tr = _boolDaeToReal(t, simCode)
      local fr = _boolDaeToReal(f, simCode)
      :($(cr) * $(tr) + (1.0 - $(cr)) * $(fr))
    end
    DAE.CALL(Absyn.IDENT("pre"), _, _) => _affectExpToReal(exp, simCode)
    _ => expToJuliaExpMTK(exp, simCode)
  end
end

#= Lower an INTEGER/enum-valued discrete RHS to an MTK Real holding the actual
   numeric value (NOT a 0/1 Boolean). Conditions are evaluated through the
   Boolean lowering; integer branches keep their value, so a five-valued FSM
   (e.g. PartialFriction `mode`) is preserved instead of clamped. =#
Base.@nospecializeinfer function _discreteIntToReal(@nospecialize(exp::DAE.Exp), simCode)
  @match exp begin
    DAE.ICONST(i) => Float64(i)
    DAE.RCONST(r) => r
    DAE.BCONST(b) => (b ? 1.0 : 0.0)
    DAE.ENUM_LITERAL(_, idx) => Float64(idx)
    DAE.IFEXP(c, t, f) => :(ModelingToolkit.ifelse(
        0.5 < $(_boolDaeToReal(c, simCode)),
        $(_discreteIntToReal(t, simCode)),
        $(_discreteIntToReal(f, simCode))))
    DAE.BINARY(e1, op, e2) => begin
      local opSym = DAE_OP_toJuliaOperator(op)
      :($(opSym)($(_discreteIntToReal(e1, simCode)), $(_discreteIntToReal(e2, simCode))))
    end
    DAE.CALL(Absyn.IDENT("pre"), _, _) => _affectExpToReal(exp, simCode)
    _ => expToJuliaExpMTK(exp, simCode)
  end
end

#= One affect equation `disc ~ <value>` for a lifted discrete: the triggering
   relations are pinned to their post-crossing values (`pins`, a zero-set group,
   see _zeroSetPins). Integer/enum discretes keep their multi-valued result;
   Boolean discretes are 0/1-clamped. =#
Base.@nospecializeinfer function _discreteAffectEq(discSym::Symbol, @nospecialize(rhsDAE::DAE.Exp),
                                                   pins::Dict{String,Bool}, isInt::Bool, simCode)
  local pinned = _substRelations(rhsDAE, pins)
  if isInt
    return :($(discSym) ~ $(_discreteIntToReal(pinned, simCode)))
  end
  return :($(discSym) ~ ModelingToolkit.ifelse(0.5 < $(_boolDaeToReal(pinned, simCode)), 1.0, 0.0))
end

#= Initialize affect for a lifted discrete: evaluate the FULL rhs at t0 (no
   relation pinned). A `time >= startTime` relation is already satisfied at t0,
   so no zero-crossing fires there; without this the discrete is stuck at its
   default start value. Implements the `initial()` term of the §17.4.4 condition. =#
Base.@nospecializeinfer function _discreteAffectEqInit(discSym::Symbol, @nospecialize(rhsDAE::DAE.Exp),
                                                       isInt::Bool, simCode)
  if isInt
    return :($(discSym) ~ $(_discreteIntToReal(rhsDAE, simCode)))
  end
  return :($(discSym) ~ ModelingToolkit.ifelse(0.5 < $(_boolDaeToReal(rhsDAE, simCode)), 1.0, 0.0))
end

#= The discrete-cluster path (discreteClusters.jl): the whens over relations
   of coupled discretes are evaluated by the event iteration (MLS 8.6)
   instead of MTK affects, which have no event iteration: a pulse's sample
   event, the whens it enables and the algebraic re-solve then happen in one
   event. The path for every model; OMBACKEND_DISCRETE_PRE_MEMORY=false
   keeps the MTK affects for the models without mode FSMs, table or switch
   clusters (the variable keeps its name from the pre-memory affects this
   replaced). =#
_discreteClustersForced()::Bool = OMBackend.envSwitch("OMBACKEND_DISCRETE_PRE_MEMORY")

#= True if the expression contains a constant-table subscript (DAE.ASUB).
   Such clusters are Newton-hostile: equation-form affects compile to an
   implicit solve whose residuals are piecewise constant. =#
Base.@nospecializeinfer function _expHasTableLookup(@nospecialize(exp))::Bool
  @match exp begin
    DAE.ASUB(__) => true
    DAE.BINARY(e1, _, e2) => _expHasTableLookup(e1) || _expHasTableLookup(e2)
    DAE.LBINARY(e1, _, e2) => _expHasTableLookup(e1) || _expHasTableLookup(e2)
    DAE.RELATION(e1, _, e2) => _expHasTableLookup(e1) || _expHasTableLookup(e2)
    DAE.UNARY(_, e) => _expHasTableLookup(e)
    DAE.LUNARY(_, e) => _expHasTableLookup(e)
    DAE.CAST(_, e) => _expHasTableLookup(e)
    DAE.IFEXP(c, t, f) =>
      _expHasTableLookup(c) || _expHasTableLookup(t) || _expHasTableLookup(f)
    DAE.CALL(_, args, _) => begin
      local found = false
      for a in args
        found = found || _expHasTableLookup(a)
      end
      found
    end
    _ => false
  end
end

#= True if the model has lifted discrete-Boolean when clusters AND its
   residuals read constant tables. The affect system MTK builds for an
   equation-form affect pulls in the surrounding algebraic equations, so the
   model-level residual content decides Newton-hostility, not the cluster
   bodies. Selects the imperative affect lowering and the Newton-FD event
   re-initialization. =#
function _modelHasTableClusters(simCode)::Bool
  local hasCluster = false
  for weq in simCode.whenEquations
    #= A cluster without relations (it follows pre() values) closes no loop through a table. =#
    local rels = _extractChangeRelations(weq.whenEquation.condition, simCode)
    if rels !== nothing && !isempty(rels)
      hasCluster = true
      break
    end
  end
  hasCluster || return false
  for eq in simCode.residualEquations
    if _expHasTableLookup(SimulationCode.toDAEExp(eq.exp))
      return true
    end
  end
  return false
end

#= Unwrap nested block Exprs (and their annotation LineNumberNodes) down to
   the core expression. =#
function _unwrapBlockExpr(e)
  while e isa Expr && e.head == :block
    local args = [a for a in e.args if !(a isa LineNumberNode)]
    isempty(args) && return e
    e = last(args)
  end
  return e
end

#= True if the expression reads pre() of any name in `names`. =#
Base.@nospecializeinfer function _expHasPreOf(@nospecialize(exp), names::Set{String})::Bool
  @match exp begin
    DAE.CALL(Absyn.IDENT("pre"), args, _) => string(listHead(args).componentRef) in names
    DAE.BINARY(e1, _, e2) => _expHasPreOf(e1, names) || _expHasPreOf(e2, names)
    DAE.LBINARY(e1, _, e2) => _expHasPreOf(e1, names) || _expHasPreOf(e2, names)
    DAE.RELATION(e1, _, e2) => _expHasPreOf(e1, names) || _expHasPreOf(e2, names)
    DAE.UNARY(_, e) => _expHasPreOf(e, names)
    DAE.LUNARY(_, e) => _expHasPreOf(e, names)
    DAE.CAST(_, e) => _expHasPreOf(e, names)
    DAE.IFEXP(c, t, f) =>
      _expHasPreOf(c, names) || _expHasPreOf(t, names) || _expHasPreOf(f, names)
    DAE.CALL(_, args, _) => begin
      local found = false
      for a in args
        found = found || _expHasPreOf(a, names)
      end
      found
    end
    _ => false
  end
end

#= True if any lifted cluster assigns an Integer/enum discrete (a mode
   variable) whose rhs reads pre() of a discrete assigned in the same
   cluster. Such bodies are sequential FSM transitions; the equation-form
   affect solves them simultaneously, which mis-latches. Scoped to the
   mode-carrying shape; Boolean-only self-pre clusters (freewheel logic)
   stay on the plain path, which iterates them to a fixpoint within the
   event (_eventIterAffectParts). =#
function _modelHasModeFSMClusters(simCode)::Bool
  for weq in simCode.whenEquations
    _extractChangeRelations(weq.whenEquation.condition, simCode) === nothing && continue
    local assigned = Set{String}()
    local intRhss = Any[]
    for st in collect(weq.whenEquation.whenStmtLst)
      (st isa SimulationCode.ASSIGN || st isa BDAE.ASSIGN) || continue
      local leftStr = SimulationCode.string(SimulationCode.toDAEExp(st.left))
      push!(assigned, leftStr)
      haskey(simCode.stringToSimVarHT, leftStr) || continue
      local (_, var) = simCode.stringToSimVarHT[leftStr]
      local isInt = @match var.attributes begin
        SOME(DAE.VAR_ATTR_INT(__)) => true
        SOME(DAE.VAR_ATTR_ENUMERATION(__)) => true
        _ => false
      end
      isInt && push!(intRhss, SimulationCode.toDAEExp(st.right))
    end
    isempty(intRhss) && continue
    for rhs in intRhss
      _expHasPreOf(rhs, assigned) && return true
    end
  end
  return false
end

#= Newton with an FD Jacobian degenerates to the Gauss-Jacobi sweep that
   piecewise-constant algebraic rows need; Broyden's secant update diverges
   on them. Used as the event re-init default for table-cluster models. =#
function tableClusterInitAlg()
  return DiffEqBase.BrownFullBasicInit(1e-8, NonlinearSolve.NewtonRaphson(; autodiff = ADTypes.AutoFiniteDiff()))
end

#= The discrete unknowns of Integer, Boolean or enumeration type, whose
   values must stay integral (withIntegralDiscretes). =#
function integralDiscreteNames(discreteSyms, simCode)::Vector{String}
  local out = String[]
  for s in discreteSyms
    local name = string(s)
    haskey(simCode.stringToSimVarHT, name) || continue
    local (_, var) = simCode.stringToSimVarHT[name]
    local integral = @match var.attributes begin
      SOME(DAE.VAR_ATTR_INT(__)) => true
      SOME(DAE.VAR_ATTR_BOOL(__)) => true
      SOME(DAE.VAR_ATTR_ENUMERATION(__)) => true
      _ => false
    end
    integral && push!(out, name)
  end
  return out
end

#= True if `name` is a Boolean-typed discrete (so pre(name) read from the Float
   memory must become a Bool before use in an and/or/not context). =#
Base.@nospecializeinfer function _isBoolDiscreteName(name::String, simCode)::Bool
  haskey(simCode.stringToSimVarHT, name) || return false
  local (_, var) = simCode.stringToSimVarHT[name]
  return @match var.attributes begin
    SOME(DAE.VAR_ATTR_BOOL(__)) => true
    _ => false
  end
end

#= Names substituted statically inside an affect body (post-event branch values
   replacing stale observed reads). Default: no substitution. =#
const _EMPTY_MEM_SUBST = Dict{Symbol, Any}()
const _EMPTY_REL_PINS = Dict{String, Bool}()

#= An expression that _daeExpToJuliaMem cannot lower into an ImperativeAffect
   body. The event-iteration path catches it and keeps the equation-form affects. =#
struct _UnsupportedInAffect <: Exception
  exp::Any
end
Base.showerror(io::IO, e::_UnsupportedInAffect) =
  print(io, "_daeExpToJuliaMem: unsupported in an ImperativeAffect body: ", string(e.exp))

#= What a lowering to an affect body or a run-time check throws for a construct it does not support. =#
const LoweringFailure = Union{UnsupportedLowering, _UnsupportedInAffect}

#= Modelica event operators with no meaning inside an affect body; the generic
   call lowering would emit an undefined Julia function for them. =#
const _AFFECT_UNSUPPORTED_BUILTINS = ("sample", "terminal", "delay", "der", "reinit", "cardinality")

#= Pinned value of a relation inside an affect body (a Bool, or an
   expression that reads it, e.g. a discrete cluster's buffer), or nothing. =#
Base.@nospecializeinfer function _relPinValue(@nospecialize(exp::DAE.Exp), relPins::AbstractDict{String})
  (!isempty(relPins) && exp isa DAE.RELATION) || return nothing
  return get(relPins, string(exp), nothing)
end

#= For edge()/change() of a relation, or of a Boolean defined by one, whose
   values `relPins` and values before the pass `relPre` are given (a
   discrete cluster): the two expressions. Otherwise nothing. =#
Base.@nospecializeinfer function _relationAndPre(@nospecialize(v::DAE.Exp), relPins::AbstractDict{String},
                                                 relPre::Union{Nothing, AbstractDict{String}}, simCode)
  relPre === nothing && return nothing
  local rel = v isa DAE.RELATION ? v : (v isa DAE.CREF ? _condVarRelation(string(v.componentRef), simCode) : nothing)
  rel === nothing && return nothing
  local key = string(rel)
  (haskey(relPins, key) && haskey(relPre, key)) || return nothing
  return (relPins[key], relPre[key])
end

#= Lower a DAE exp at a boolean position of an ImperativeAffect body. Live
   reads come back as 0/1 Floats; `> 0.5` coerces both Bool and Float.
   `initVal` is what `initial()` lowers to (true in an initialize affect, or
   the name of a Bool argument). With `relPre` (a discrete cluster),
   edge()/change() of a relation compare its value with its value before the
   pass. =#
Base.@nospecializeinfer function _daeBoolMem(@nospecialize(exp::DAE.Exp), obsAcc::Dict{Symbol,Symbol}, simCode;
                                             initVal::Union{Bool, Symbol} = false,
                                             subst::Dict{Symbol,Any} = _EMPTY_MEM_SUBST,
                                             relPins::AbstractDict{String} = _EMPTY_REL_PINS,
                                             preSubst::Union{Nothing, Dict{Symbol,Any}} = nothing,
                                             relPre::Union{Nothing, AbstractDict{String}} = nothing)
  recb(@nospecialize e) = _daeBoolMem(e, obsAcc, simCode; initVal = initVal, subst = subst, relPins = relPins,
                                      preSubst = preSubst, relPre = relPre)
  local pin = _relPinValue(exp, relPins)
  pin === nothing || return pin
  @match exp begin
    DAE.BCONST(b) => b
    DAE.LUNARY(DAE.NOT(__), e) => :(!$(recb(e)))
    DAE.LBINARY(e1, DAE.AND(__), e2) => :($(recb(e1)) && $(recb(e2)))
    DAE.LBINARY(e1, DAE.OR(__), e2) => :($(recb(e1)) || $(recb(e2)))
    DAE.RELATION(__) => _daeExpToJuliaMem(exp, obsAcc, simCode; initVal = initVal, subst = subst, relPins = relPins,
                                          preSubst = preSubst, relPre = relPre)
    DAE.CALL(Absyn.IDENT("initial"), _, _) => initVal
    #= edge(v) = v and not pre(v); change(v) = v <> pre(v), with pre() lowered
       like any other (the event iteration's previous pass; for a discrete
       cluster's relation, its value before the pass, `relPre`).
       The frontend inlines `b = x > 0.2` into edge(x > 0.2): in an event
       iteration a relation changes only at its own crossing, i.e. when this
       callback pins it, and only in the first pass (later passes see
       pre = the same value). =#
    DAE.CALL(Absyn.IDENT("edge"), args, attr) => begin
      local v = listHead(args)
      local rp = _relationAndPre(v, relPins, relPre, simCode)
      if rp !== nothing
        :($(rp[1]) && !$(rp[2]))
      elseif v isa DAE.RELATION && preSubst !== nothing
        _relPinValue(v, relPins) === true ? :(_firstPass) : false
      else
        v isa DAE.CREF || throw(_UnsupportedInAffect(exp))
        :($(recb(v)) && !$(recb(DAE.CALL(Absyn.IDENT("pre"), args, attr))))
      end
    end
    DAE.CALL(Absyn.IDENT("change"), args, attr) => begin
      local v = listHead(args)
      local preV = DAE.CALL(Absyn.IDENT("pre"), args, attr)
      local rp = _relationAndPre(v, relPins, relPre, simCode)
      if rp !== nothing
        :($(rp[1]) != $(rp[2]))
      elseif v isa DAE.RELATION && preSubst !== nothing
        _relPinValue(v, relPins) === nothing ? false : :(_firstPass)
      elseif !(v isa DAE.CREF)
        throw(_UnsupportedInAffect(exp))
      elseif _isBoolDiscreteName(string(v.componentRef), simCode)
        :($(recb(v)) != $(recb(preV)))
      else
        local recv(@nospecialize e) = _daeExpToJuliaMem(e, obsAcc, simCode; initVal = initVal, subst = subst,
                                                        relPins = relPins, preSubst = preSubst, relPre = relPre)
        :($(recv(v)) != $(recv(preV)))
      end
    end
    DAE.IFEXP(c, t, f) => :($(recb(c)) ? $(recb(t)) : $(recb(f)))
    _ => begin
      local v = _daeExpToJuliaMem(exp, obsAcc, simCode; initVal = initVal, subst = subst, preSubst = preSubst,
                                  relPre = relPre)
      v isa Bool ? v : :($(v) > 0.5)
    end
  end
end

#= Lower a DAE exp to a Julia Expr for an ImperativeAffect body (or a
   discrete cluster's body):
     pre(x)                -> preSubst[x], else observed.x (its value when the
                              affect runs, before it changes anything)
     continuous/param cref -> observed.<name>        (collected into obsAcc)
     relation/ifelse/and/or/not/arith -> Julia control flow
   `initVal` is what `initial()` lowers to (see _daeBoolMem). Relations are
   replaced through `relPins`; `relPre`: see _daeBoolMem. =#
Base.@nospecializeinfer function _daeExpToJuliaMem(@nospecialize(exp::DAE.Exp), obsAcc::Dict{Symbol,Symbol}, simCode;
                                                   initVal::Union{Bool, Symbol} = false,
                                                   subst::Dict{Symbol,Any} = _EMPTY_MEM_SUBST,
                                                   relPins::AbstractDict{String} = _EMPTY_REL_PINS,
                                                   preSubst::Union{Nothing, Dict{Symbol,Any}} = nothing,
                                                   relPre::Union{Nothing, AbstractDict{String}} = nothing)
  rec(@nospecialize e) = _daeExpToJuliaMem(e, obsAcc, simCode; initVal = initVal, subst = subst, relPins = relPins,
                                           preSubst = preSubst, relPre = relPre)
  recb(@nospecialize e) = _daeBoolMem(e, obsAcc, simCode; initVal = initVal, subst = subst, relPins = relPins,
                                      preSubst = preSubst, relPre = relPre)
  local pin = _relPinValue(exp, relPins)
  pin === nothing || return pin
  @match exp begin
    DAE.ICONST(i) => Float64(i)
    DAE.RCONST(r) => r
    DAE.BCONST(b) => b
    DAE.ENUM_LITERAL(_, idx) => Float64(idx)
    DAE.CALL(Absyn.IDENT("pre"), args, _) => begin
      local nm = string(listHead(args).componentRef)
      local key = Symbol(nm)
      local rd = if preSubst !== nothing && haskey(preSubst, key)
        preSubst[key]
      else
        obsAcc[key] = key
        :(observed.$(key))
      end
      _isBoolDiscreteName(nm, simCode) ? :($(rd) > 0.5) : rd
    end
    DAE.CALL(Absyn.IDENT("initial"), _, _) => initVal
    DAE.CALL(Absyn.IDENT("edge"), _, _) || DAE.CALL(Absyn.IDENT("change"), _, _) => recb(exp)
    DAE.CALL(Absyn.IDENT(name), _, _) where name in _AFFECT_UNSUPPORTED_BUILTINS =>
      throw(_UnsupportedInAffect(exp))
    DAE.CALL(path, _, _) where _isDelayCall(path) => throw(_UnsupportedInAffect(exp))
    DAE.CAST(_, e) => rec(e)
    DAE.RELATION(e1, op, e2) => :($(DAE_OP_toJuliaOperator(op))($(rec(e1)), $(rec(e2))))
    DAE.LUNARY(DAE.NOT(__), e) => :(!$(recb(e)))
    DAE.LBINARY(e1, DAE.AND(__), e2) => :($(recb(e1)) && $(recb(e2)))
    DAE.LBINARY(e1, DAE.OR(__), e2) => :($(recb(e1)) || $(recb(e2)))
    DAE.IFEXP(c, t, f) => :($(recb(c)) ? $(rec(t)) : $(rec(f)))
    DAE.BINARY(e1, op, e2) => :($(DAE_OP_toJuliaOperator(op))($(rec(e1)), $(rec(e2))))
    DAE.UNARY(op, e) => :($(DAE_OP_toJuliaOperator(op))($(rec(e))))
    DAE.CREF(cr, _) => begin
      local nm = Symbol(string(cr))
      if nm === :time
        #= module-scope `time` is Base.time; the affect reads the integrator clock =#
        :(integrator.t)
      elseif haskey(subst, nm)
        subst[nm]
      elseif haskey(simCode.stringToSimVarHT, string(cr)) &&
             simCode.stringToSimVarHT[string(cr)][2].varKind isa SimulationCode.DATA_STRUCTURE
        #= module-global table handle (DATA_STRUCTURE), referenced by bare name =#
        Symbol(string(cr))
      else
        obsAcc[nm] = nm
        :(observed.$(nm))
      end
    end
    #= Modelica index helpers used by Digital gate residuals. The lookup rounds
       its index, so floor/integer only need to evaluate the inner value. =#
    DAE.CALL(Absyn.IDENT("floor"), args, _) => :(floor($(rec(listHead(args)))))
    DAE.CALL(Absyn.IDENT("integer"), args, _) =>
      :(OMBackend.CodeGeneration.AlgorithmicCodeGeneration.modelica_integer($(rec(listHead(args)))))
    #= Numeric calls with direct Julia equivalents. =#
    DAE.CALL(Absyn.IDENT("abs"), args, _) => :(abs($(rec(listHead(args)))))
    DAE.CALL(Absyn.IDENT("sign"), args, _) => :(sign($(rec(listHead(args)))))
    DAE.CALL(Absyn.IDENT("sqrt"), args, _) => :(sqrt($(rec(listHead(args)))))
    #= min/max of an array (one argument) is not lowered here. =#
    DAE.CALL(Absyn.IDENT("min"), args, _) where (listLength(args) == 2) =>
      :(min($(rec(listHead(args))), $(rec(listHead(listRest(args))))))
    DAE.CALL(Absyn.IDENT("max"), args, _) where (listLength(args) == 2) =>
      :(max($(rec(listHead(args))), $(rec(listHead(listRest(args))))))
    DAE.CALL(Absyn.IDENT("min"), _, _) => throw(_UnsupportedInAffect(exp))
    DAE.CALL(Absyn.IDENT("max"), _, _) => throw(_UnsupportedInAffect(exp))
    #= Event-control wrappers are semantic no-ops inside an affect body. =#
    DAE.CALL(Absyn.IDENT("noEvent"), args, _) => rec(listHead(args))
    DAE.CALL(Absyn.IDENT("smooth"), args, _) => rec(listHead(listRest(args)))
    #= Constant-table lookup `table[idx...]`: the table is a constant literal
       (lower via the standard expression path), the subscripts are gate inputs
       lowered through `rec` so they read observed values. Routes
       through `constTableLookup` (handles numeric + rounded indices). =#
    DAE.ASUB(exp = tableExp, sub = subs) => begin
      #= One element: each subscript an index. A slice or a whole dimension
         (`t[:, k]`, which read row 1) is the equation form's. =#
      local _subExp = s -> s isa DAE.INDEX ? s.exp : throw(_UnsupportedInAffect(exp))
      local subCodes = collect(rec(_subExp(s)) for s in subs)
      :(OMBackend.CodeGeneration.constTableLookup($(expToJuliaExpMTK(tableExp, simCode)), $(subCodes...)))
    end
    DAE.SCONST(str) => str
    DAE.ARRAY(__) => expToJuliaExpMTK(exp, simCode)
    #= Modelica's String: Julia's String has no such method (an assert
       message, MSL Media), and the values arrive as numbers. =#
    DAE.CALL(path = Absyn.IDENT("String"), expLst = cargs, attr = attr) where attr.builtin =>
      AlgorithmicCodeGeneration.modelicaStringCall(collect(cargs), rec)
    #= External / qualified Modelica function (e.g. CombiTimeTable
       Internal.getNextTimeEvent): mirror the residual's name resolution
       (canonicalName -> underscore form), args lowered imperatively. =#
    DAE.CALL(path = p, expLst = cargs) =>
      Expr(:call, Symbol(OMBackend.canonicalName(string(p))), (rec(a) for a in cargs)...)
    #= An element of a multi-output call (a tuple assignment split per target:
       MSL TimeTable's (a, b, ...) := getInterpolationCoefficients(...)). =#
    DAE.TSUB(e, ix, _) => :(getindex($(rec(e)), $(ix)))
    _ => throw(_UnsupportedInAffect(exp))
  end
end

#= True if some member's rhs reads pre() of a cluster member, so one pass of the
   body can leave pre(x) <> x (see _eventIterAffectParts). =#
function _clusterReadsOwnPre(assigns::Vector{Tuple{Symbol,Any,Bool}})::Bool
  local members = Set{String}(string(d) for (d, _, _) in assigns)
  local found = false
  function visit(e::DAE.Exp, arg)
    if e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name == "pre"
      local a = listHead(e.expLst)
      a isa DAE.CREF && string(a.componentRef) in members && (found = true)
    end
    return (e, arg)
  end
  for (_, rhs, _) in assigns
    Util.traverseExpBottomUp(rhs, visit, nothing)
  end
  return found
end

#= True for the condition of a when synthesized from discrete equations
   (synthesizeWhenEquationsFromDiscreteEquations): an OR-chain of
   `change(<relation>)`. User whens (`when change(b)`, `when x > 0`) run their
   body once per event and must not be iterated. =#
function _isSynthesizedChangeCondition(@nospecialize(cond))::Bool
  local d = cond isa SimulationCode.Exp ? SimulationCode.toDAEExp(cond) : cond
  @match d begin
    DAE.CALL(Absyn.IDENT("change"), args, _) => listHead(args) isa DAE.RELATION || _isChangeOfPre(d)
    DAE.LBINARY(e1, DAE.OR(__), e2) => _isSynthesizedChangeCondition(e1) && _isSynthesizedChangeCondition(e2)
    _ => false
  end
end

#= Whether the ImperativeAffect can be generated for this cluster: every rhs
   lowers into its body. Otherwise the cluster keeps the equation-form affects,
   which lower more expression kinds. =#
function _eventIterLowerable(assigns::Vector{Tuple{Symbol,Any,Bool}}, simCode)::Bool
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

#= ImperativeAffect for a plain-path cluster that reads its own pre() values:
   Modelica event iteration (§8.6). Each pass evaluates the body in order with
   pre(x) = x after the previous pass (x at the event for the first pass) and
   member crefs = their latest value, until a pass changes nothing. The pinned
   relations stay pinned; the other operands are read once, at the event. An
   equation-form affect is one pass: in the OneWayClutch the crossing to
   w_rel <= 0 sets stuck, but `locked = pre(stuck) and not startForward`
   needs the second pass, so without it the freewheel never locks.
   `iterate = false`: one pass (a cluster that does not read its own pre()).
   `initVal`: the value of initial() (true in the t0 pass). =#
Base.@nospecializeinfer function _eventIterAffectParts(assigns::Vector{Tuple{Symbol,Any,Bool}}, simCode;
                                                       relPins::Dict{String,Bool} = _EMPTY_REL_PINS,
                                                       initVal::Bool = false, iterate::Bool = true)
  local obsAcc = Dict{Symbol,Symbol}()
  local preSyms = Dict{Symbol,Any}(d => Symbol("_pre_", d) for (d, _, _) in assigns)
  local subst = Dict{Symbol,Any}(d => Symbol("_v_", d) for (d, _, _) in assigns)
  local setup = Expr[]; local pass = Expr[]; local changed = Expr[]
  local shift = Expr[]; local retKws = Expr[]
  for (d, rhs, isInt) in assigns
    local vsym = subst[d]; local psym = preSyms[d]
    push!(setup, :(local $(psym) = Float64(modified.$(d))))
    push!(setup, :(local $(vsym) = $(psym)))
    local valExpr = isInt ?
      :(Float64($(_daeExpToJuliaMem(rhs, obsAcc, simCode; initVal = initVal, subst = subst, relPins = relPins,
                                    preSubst = preSyms)))) :
      :($(_daeBoolMem(rhs, obsAcc, simCode; initVal = initVal, subst = subst, relPins = relPins,
                      preSubst = preSyms)) ? 1.0 : 0.0)
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

#= _eventIterAffectParts as an ImperativeAffect expression. =#
function _imperativeClusterAffect(assigns::Vector{Tuple{Symbol,Any,Bool}}, simCode; kwargs...)::Expr
  local (fn, obs, mod) = _eventIterAffectParts(assigns, simCode; kwargs...)
  return :(ModelingToolkit.ImperativeAffect($(fn), $(mod); observed = $(obs), skip_checks = true))
end

#= A discrete's start attribute as a number: a literal, an enumeration
   literal (its index), or a parameter's folded binding (the MSL thyristor
   bridges' off(start = offStart_p1)). Nothing without a start or when it
   does not evaluate. =#
function _discreteStartValue(sv, simCode)::Union{Float64, Nothing}
  local start = @match sv.attributes begin
    SOME(a) => (hasproperty(a, :start) ? a.start : NONE())
    _ => NONE()
  end
  return @match start begin
    SOME(e && DAE.CREF(__)) => begin
      local entry = get(simCode.stringToSimVarHT, SimulationCode.string(e), nothing)
      entry === nothing && return nothing
      @match entry[2].varKind begin
        SimulationCode.PARAMETER(bindExp = SOME(b)) => _foldParameterBindStatic(b, simCode)
        _ => nothing
      end
    end
    SOME(e) => _foldParameterBindStatic(e, simCode)
    _ => nothing
  end
end

#= Module-level list of the discretes the discrete clusters assign
   (emitDiscreteClusters): the direct-RHS initialization leaves them to the
   clusters' start bodies. Empty when the model does not take that path. =#
function liftedDiscretesDecl(simCode)::Expr
  _usesDiscreteClusters(simCode) || return Expr(:block)
  local names = String[]
  for weq in simCode.whenEquations
    _extractChangeRelations(weq.whenEquation.condition, simCode) === nothing && continue
    for st in collect(weq.whenEquation.whenStmtLst)
      (st isa SimulationCode.ASSIGN || st isa BDAE.ASSIGN) || continue
      local leftStr = SimulationCode.string(SimulationCode.toDAEExp(st.left))
      haskey(simCode.stringToSimVarHT, leftStr) || continue
      push!(names, string(last(simCode.stringToSimVarHT[leftStr]).name))
    end
  end
  isempty(names) && return Expr(:block)
  #= Their start values and those of the other discretes (what pre() reads at
     initialization, MLS 8.6), for the initial fixpoint of the clusters. =#
  local starts = Dict{String, Float64}()
  for (_, (_, sv)) in simCode.stringToSimVarHT
    (sv.varKind isa SimulationCode.DISCRETE || string(sv.name) in names) || continue
    local v = _discreteStartValue(sv, simCode)
    v === nothing || (starts[string(sv.name)] = v)
  end
  return quote
    LIFTED_DISCRETES = $(names)
    LIFTED_DISCRETE_STARTS = $(starts)
  end
end

#= Module-level list of the Real parameters without a value that the
   initialization computes (fixed = false, no binding; the MSL
   InitSpringConstant's spring.c): the direct-RHS initialization solves them
   next to the states. Names are the MTK parameters' (simVar.name, as in
   createParameterEquationsMTK). Empty when there are none. =#
function freeParametersDecl(simCode)::Expr
  #= A parameter alone on the left of an initial equation is assigned by it
     (at codegen, or by the init solve's assignments, which take the left
     side), not free: the MSL Mean block's `t0 = time`. As a free unknown it
     had no determining row, and its zero Jacobian column disabled the scaled
     Newton step the ideal diodes of DiodeBridge2mPulse need. Alone on the
     right (`x = q`) nothing assigns it: left out here too, q kept its start
     and both of its equations were broken. =#
  local assigned = OrderedSet{String}()
  for ieq in simCode.initialEquations
    hasEquationSides(ieq) || continue
    local lhs = first(equationSides(ieq))
    lhs isa DAE.CREF && push!(assigned, string(lhs))
  end
  local names = String[]
  #= The ones the initialization assigns (ASSIGNED_PARAMETERS): only these,
     never a bound or tunable parameter alone on the left (`p = 2 * q`: p
     was overwritten with 2q, and q never solved). =#
  local assignedNames = String[]
  for (key, (_, sv)) in simCode.stringToSimVarHT
    (SimulationCode.isParameter(sv) && !SimulationCode.hasBindingExp(sv)) || continue
    if key in assigned
      push!(assignedNames, string(sv.name))
      continue
    end
    local free = @match sv.attributes begin
      SOME(DAE.VAR_ATTR_REAL(fixed = SOME(DAE.BCONST(false)))) => true
      _ => false
    end
    free && push!(names, string(sv.name))
  end
  local decls = Expr[]
  isempty(names) || push!(decls, :(FREE_PARAMETERS = $(sort!(unique!(names)))))
  isempty(assignedNames) || push!(decls, :(ASSIGNED_PARAMETERS = $(sort!(unique!(assignedNames)))))
  return Expr(:block, decls...)
end

#= Module-level list of the user's fixed values (USER_PINS): the continuous
   Real variables with fixed = true (their start, or the default one) and the
   left sides of initial equations `v = literal or parameter`, named as the
   MTK unknowns (simVar.name; a relayed variable by its if-equation
   temporary). Not discretes: fixed = true there is `pre(v) = start` (MLS 8.6),
   which may move at the start. The direct-RHS initialization refuses a result
   that moved one (buildDirectRHSProblem). Empty when there are none. =#
function userPinsDecl(simCode, relayAliases::AbstractDict)::Expr
  local names = OrderedSet{String}()
  local rename = n -> string(get(relayAliases, Symbol(n), Symbol(n)))
  #= The literal value a fixed start gives (0 without a start), by name. =#
  local fixedLiteral = Dict{String, Float64}()
  for (_, (_, sv)) in simCode.stringToSimVarHT
    sv.varKind isa Union{SimulationCode.STATE, SimulationCode.ALG_VARIABLE} || continue
    local fixed = @match sv.attributes begin
      SOME(DAE.VAR_ATTR_REAL(fixed = SOME(DAE.BCONST(true)))) => true
      _ => false
    end
    fixed || continue
    push!(names, rename(sv.name))
    local startVal = @match sv.attributes begin
      SOME(DAE.VAR_ATTR_REAL(start = SOME(DAE.RCONST(r)))) => Float64(r)
      SOME(DAE.VAR_ATTR_REAL(start = SOME(DAE.ICONST(i)))) => Float64(i)
      SOME(DAE.VAR_ATTR_REAL(start = NONE())) => 0.0
      _ => nothing
    end
    startVal === nothing || (fixedLiteral[string(sv.name)] = startVal)
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
    entry === nothing || last(entry).varKind isa Union{SimulationCode.STATE, SimulationCode.ALG_VARIABLE} || continue
    local rhsEntry = rhs isa Symbol ? get(simCode.stringToSimVarHT, string(rhs), nothing) : nothing
    #= A parameter right side (a tunable one stays a reference) is a pin too;
       a conflict with a fixed start is then checked at the build and per run
       (DirectRHSGeneration's _collectHardInitializationValues), here only a literal's. =#
    (rhs isa Number || (rhsEntry !== nothing && SimulationCode.isParameter(last(rhsEntry)))) || continue
    #= A fixed start and an initial equation that give one variable two values
       (`x(start = 1, fixed = true)` and `x = 2`): the initialization took the
       equation's without a word; OpenModelica refuses it. =#
    local startVal = get(fixedLiteral, string(lhs), nothing)
    (rhs isa Number && startVal !== nothing && !isapprox(Float64(rhs), startVal; rtol = 1e-12, atol = 1e-12)) &&
      CodeGeneration.unsupported("a fixed start and an initial equation that give a variable different values",
                                 "$(lhs): start $(startVal), initial equation $(rhs)")
    push!(names, rename(lhs))
  end
  isempty(names) && return Expr(:block)
  return :(USER_PINS = $(sort!(collect(names))))
end

#= Module-level list of the discrete variables (DISCRETE_VARIABLES): their
   fixed start is `pre(v) = start` (MLS 8.6), and an initialization equation
   `v ~ start` beside `v ~ value` (a when initial() body) is no conflict
   (_collectHardInitializationValues). Empty when there are none. =#
function discreteVariablesDecl(simCode)::Expr
  local names = sort!([string(sv.name) for (_, (_, sv)) in simCode.stringToSimVarHT if SimulationCode.isDiscrete(sv)])
  isempty(names) && return Expr(:block)
  return :(DISCRETE_VARIABLES = $(names))
end

#= Module-level list of the tunable parameters (TUNABLE_NAMES,
   withTunableParameters) the model was compiled with, for the runs with
   other values of them (DirectRHSGeneration's _runEntry). Empty when there
   are none. =#
function tunableParametersDecl()::Expr
  local names = sort!(collect(OMBackend.TUNABLE_PARAMETERS[]))
  isempty(names) && return Expr(:block)
  return :(TUNABLE_NAMES = $(names))
end

#= Whether a continuous or initialization equation of the model calls
   homotopy() (where it is the blend, HOMOTOPY_BLEND): it then gets the
   homotopy parameter (HOMOTOPY_LAMBDA). By the printed equations (a call
   prints as `homotopy(`); a function of another name ending in it only adds
   an unused parameter and the continuation's steps. =#
function _usesHomotopy(simCode)::Bool
  local rx = r"(^|[^A-Za-z0-9_])homotopy\("
  local has = e -> occursin(rx, string(e isa SimulationCode.Exp ? SimulationCode.toDAEExp(e) : e))
  any(eq -> has(eq.exp), simCode.residualEquations) && return true
  for ieq in simCode.initialEquations
    hasEquationSides(ieq) && any(has, equationSides(ieq)) && return true
  end
  return false
end

#= Whether the model takes the discrete-cluster path, shared by every site
   that must agree (emitDiscreteClusters, the MTK events, LIFTED_DISCRETES). =#
_usesDiscreteClusters(simCode)::Bool =
  _discreteClustersForced() || _modelHasTableClusters(simCode) || _modelHasModeFSMClusters(simCode) ||
  _modelHasSwitchClusters(simCode) || _modelHasRelationlessClusters(simCode)

#= True if a lifted cluster has no relation (it follows pre() values): only
   the event iteration evaluates it, the MTK events have no crossing for it. =#
_modelHasRelationlessClusters(simCode)::Bool =
  any(simCode.whenEquations) do w
    local rels = _extractChangeRelations(w.whenEquation.condition, simCode)
    rels !== nothing && isempty(rels)
  end

#= True if a lifted cluster closes an algebraic loop through its own
   equations, as the MSL's ideal diodes and thyristors: one of its relations
   reads an algebraic unknown that only equations reading a member, or a value
   a member selects, determine (s in `off = s < 0`, `v = s*(if off then 1 else Ron)`,
   `i = s*(if off then Goff else 1)`). After the cluster changes, s and the
   circuit must be solved again before its relation is read, within the event:
   the discrete-cluster path's event iteration does that; the equation-form
   affect's implicit solve fails on the piecewise equations, and at a
   thyristor's firing the observed relation is read at the crossing itself.
   A switch with an arc model is not one: its relation reads the circuit's own
   current and voltage. =#
function _modelHasSwitchClusters(simCode)::Bool
  local ht = simCode.stringToSimVarHT
  names(exps...) = (local ns = OrderedSet{String}(); foreach(e -> SimulationCode.collectCrefNames!(ns, e), exps); ns)
  isAlgebraic(name) = haskey(ht, name) && last(ht[name]).varKind isa SimulationCode.ALG_VARIABLE
  isUnknown(name) = haskey(ht, name) && !SimulationCode.isParameter(last(ht[name]))
  local ifConds = [names((b.condition for b in ifEq.branches)...) for ifEq in simCode.ifEquations]
  #= Per candidate operand: the names that make an equation the cluster's own. =#
  local own = Dict{String, Set{String}}()
  for weq in simCode.whenEquations
    local rels = _extractChangeRelations(weq.whenEquation.condition, simCode)
    rels === nothing && continue
    local operands = [string(c) for r in rels for c in Util.getAllCrefs(r) if isAlgebraic(string(c))]
    isempty(operands) && continue
    local assigns = _gatherClusterAssigns(weq, simCode)
    assigns === nothing && continue
    #= The members, and what they select: an if-equation on a member whose
       branches set one unknown each to parameter values (the lowered
       `if off then Goff else 1`). =#
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
  visit(ns) = for (op, ownNames) in own
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

_isWhenAssert(@nospecialize(st)) = st isa BDAE.ASSERT || st isa SimulationCode.ASSERT

#= The discrete-cluster and self-scheduling lowerings run only the
   assignments of a when body: its asserts are not checked there (reported). =#
_uncheckedWhenAssert(@nospecialize(st)) =
  @warn "[MTK GEN: when] an assert in a when on a buffered relation or a self-scheduling time when is not checked" condition = string(SimulationCode.toDAEExp(st.condition))

#= Gather a synthesized when cluster's ordered (discreteSymbol, rhsDAE, isNumeric)
   assignments; `nothing` when it assigns nothing. A single-member cluster has
   one entry; a coupled FSM cluster has the body in topological order. The
   cluster paths are the only ones for such a when (createCallbackCode), so
   what they cannot run is refused. =#
function _gatherClusterAssigns(weq, simCode)
  weq.whenEquation.elsewhenPart === nothing ||
    unsupported("an elsewhen of a when on a buffered relation (a discrete cluster)", weq.whenEquation.condition)
  local assigns = Tuple{Symbol, Any, Bool}[]
  for st in collect(weq.whenEquation.whenStmtLst)
    _isWhenAssert(st) && (_uncheckedWhenAssert(st); continue)
    (st isa SimulationCode.ASSIGN || st isa BDAE.ASSIGN) ||
      unsupported("this statement in a when on a buffered relation (a discrete cluster)", st)
    local leftStr = SimulationCode.string(SimulationCode.toDAEExp(st.left))
    haskey(simCode.stringToSimVarHT, leftStr) ||
      unsupported("an assignment to a non-variable in a when on a buffered relation (a discrete cluster)", st)
    local (_, var) = simCode.stringToSimVarHT[leftStr]
    #= Numeric (else Boolean) by the attributes, else by the type of the
       assigned cref or value: an alias elimination can make an attribute-less
       variable (a gate's Logic input) the member that a clock's `y = if ...
       then '0' else '1'` sets. A Real target was written as 0/1. =#
    local leftDAE = SimulationCode.toDAEExp(st.left)
    local isInt = @match var.attributes begin
      SOME(DAE.VAR_ATTR_INT(__)) => true
      SOME(DAE.VAR_ATTR_ENUMERATION(__)) => true
      SOME(DAE.VAR_ATTR_REAL(__)) => true
      _ => _isIntegralValued(leftDAE) || _isIntegralValued(SimulationCode.toDAEExp(st.right)) ||
        (leftDAE isa DAE.CREF && leftDAE.ty isa DAE.T_REAL)
    end
    push!(assigns, (Symbol(string(var.name)), SimulationCode.toDAEExp(st.right), isInt))
  end
  return isempty(assigns) ? nothing : assigns
end

#= Whether an expression has an Integer or enumeration value. =#
Base.@nospecializeinfer function _isIntegralValued(@nospecialize(e))::Bool
  @match e begin
    DAE.CREF(_, ty) => _isIntegralType(ty)
    DAE.ICONST(__) => true
    DAE.ENUM_LITERAL(__) => true
    DAE.IFEXP(_, a, b) => _isIntegralValued(a) || _isIntegralValued(b)
    _ => false
  end
end
_isIntegralType(@nospecialize(ty))::Bool =
  ty isa DAE.T_INTEGER || ty isa DAE.T_ENUMERATION || (ty isa DAE.T_ARRAY && _isIntegralType(ty.ty))

#= Collect the relations comparing `time` against a `pre()` value (a
   self-scheduling time event) of an OR-chain. False when a disjunct is neither
   such a relation nor initial(): its trigger would be lost. =#
function _collectSelfSchedRels!(rels::Vector{DAE.Exp}, @nospecialize(e))::Bool
  @match e begin
    DAE.RELATION(exp1 = e1, exp2 = e2) where ((_isTimeCref(e1) || _isTimeCref(e2)) && (_isPreCref(e1) || _isPreCref(e2))) =>
      (push!(rels, e); true)
    DAE.LBINARY(exp1 = a, operator = DAE.OR(__), exp2 = b) => begin
      local okA = _collectSelfSchedRels!(rels, a)
      local okB = _collectSelfSchedRels!(rels, b)
      okA && okB
    end
    DAE.CALL(Absyn.IDENT("initial"), _, _) => true
    _ => false
  end
end

#= The relations of a self-scheduling time when (the time tables' `when {time
   >= pre(nextEvent), initial()}`, its initial() split off into an initial
   algorithm): the `time ⋚ pre(x)` relations of its condition. Another trigger
   (lost before: this lowering took the when) or an elsewhen is refused: the
   general path cannot evaluate such a condition either. =#
function _selfSchedulingTimeRels(weq)::Vector{DAE.Exp}
  local cond = weq.whenEquation.condition
  local d = cond isa SimulationCode.Exp ? SimulationCode.toDAEExp(cond) : cond
  local rels = DAE.Exp[]
  local onlyThese = _collectSelfSchedRels!(rels, d)
  isempty(rels) && return rels
  onlyThese || unsupported("a self-scheduling time when with another trigger", d)
  weq.whenEquation.elsewhenPart === nothing || unsupported("an elsewhen of a self-scheduling time when", d)
  return rels
end

#= Build (functionExpr, observedNT, modifiedNT) for the ImperativeAffect of a
   self-scheduling time-event when. Each ASSIGN `x := rhs` is recomputed
   imperatively (rhs lowered via `_daeExpToJuliaMem`: time->integrator.t,
   table handle->module global, external call->resolved). The table residual's
   pre(x) lowers to x, omc's state between events: the table's left limit for
   t >= x, the new segment after this affect. `atInit` lowers `initial()` to
   true for the run at the start of a solve. =#
Base.@nospecializeinfer function _selfSchedAffectParts(weq, simCode; atInit::Bool = false)
  local obsAcc = Dict{Symbol,Symbol}()
  local stmts = Expr[]
  local retKws = Expr[]
  local modNames = Symbol[]
  local subst = Dict{Symbol,Any}()
  for st in collect(weq.whenEquation.whenStmtLst)
    _isWhenAssert(st) && (atInit || _uncheckedWhenAssert(st); continue)
    (st isa SimulationCode.ASSIGN || st isa BDAE.ASSIGN) ||
      unsupported("this statement in a self-scheduling time when", st)
    local lhsDAE = SimulationCode.toDAEExp(st.left)
    lhsDAE isa DAE.CREF || unsupported("this assignment in a self-scheduling time when", st)
    local xn = string(lhsDAE.componentRef)
    local xsym = Symbol(xn)
    local vsym = Symbol("_v_", xn)
    local rhsJ = _daeExpToJuliaMem(SimulationCode.toDAEExp(st.right), obsAcc, simCode;
                                   initVal = atInit, subst = subst)
    push!(stmts, :(local $(vsym) = $(rhsJ)))
    push!(retKws, Expr(:kw, xsym, vsym))
    push!(modNames, xsym)
    subst[xsym] = vsym
  end
  local retNT = Expr(:tuple, Expr(:parameters, retKws...))
  #= The state before the instant, before the affect writes (pre() of what it
     sets, read by a when it triggers at the same instant: `e = edge(b)` with b
     set here stayed false). =#
  local fexpr = :((modified, observed, ctx, integrator) -> begin
                    $(atInit ? :() : :(OMBackend.CodeGeneration.instantPre(integrator)))
                    $(stmts...)
                    $(retNT)
                  end)
  local obsNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, k, v) for (k, v) in obsAcc]...))
  local modNT = Expr(:tuple, Expr(:parameters, [Expr(:kw, d, d) for d in modNames]...))
  return (fexpr, obsNT, modNT)
end

#= MTK SymbolicContinuousCallbacks for self-scheduling time-event whens. The
   crossing `time - nextTimeEvent` fires when time reaches the held discrete; an
   ImperativeAffect re-runs the body (updates only integrator.u, never an
   AffectSystem, so it does not pull the table-fed continuous network into an
   unsolvable callback). =#
#= Whether a self-scheduling when's body reads no pre() of a variable it sets:
   run again at a fixed time it gives the same values. =#
function _selfSchedIdempotent(weq)::Bool
  local assigned = Set{String}()
  local stmts = collect(weq.whenEquation.whenStmtLst)
  for st in stmts
    (st isa SimulationCode.ASSIGN || st isa BDAE.ASSIGN) || continue
    local lhs = SimulationCode.toDAEExp(st.left)
    lhs isa DAE.CREF && push!(assigned, string(lhs.componentRef))
  end
  for st in stmts
    (st isa SimulationCode.ASSIGN || st isa BDAE.ASSIGN) || continue
    local readsPre = false
    Util.traverseExpBottomUp(SimulationCode.toDAEExp(st.right), (e, arg) -> begin
      if e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name == "pre"
        local a = listHead(e.expLst)
        a isa DAE.CREF && string(a.componentRef) in assigned && (readsPre = true)
      end
      (e, arg)
    end, nothing)
    readsPre && return false
  end
  return true
end

function createSelfSchedulingTimeWhenEvents(simCode)::Vector{Expr}
  local events = Expr[]
  for weq in simCode.whenEquations
    local rels = _selfSchedulingTimeRels(weq)
    isempty(rels) && continue
    local (fn, obs, modN) = _selfSchedAffectParts(weq, simCode)
    isempty(modN.args[1].args) && continue
    #= A when with initial() (the time tables' `{time >= pre(nextTimeEvent),
       initial()}`) runs in the initial algorithm; at the start of every solve
       again only where that is idempotent (its body reads no pre() of what it
       sets): a solve of the problem itself (remake) skips the initial
       algorithm, and the table's C object kept its end state (z(1) = 2.77,
       OpenModelica 2.47). Not a counter (`n = pre(n) + 1` ran twice: n = 2,
       OpenModelica 1), and not a when without initial() (it ran at the
       start; OpenModelica: first at its time). =#
    local initAffect = if weq.attr.alsoInitial && _selfSchedIdempotent(weq)
      local (fnI, obsI, modI) = _selfSchedAffectParts(weq, simCode; atInit = true)
      :(ModelingToolkit.ImperativeAffect($(fnI), $(modI); observed = $(obsI), skip_checks = true))
    else
      nothing
    end
    for (k, rel) in enumerate(rels)
      #= transformToMTKContinuousCondition emits `pre(nextTimeEvent) - time` for
         `time >= pre(nextTimeEvent)`, which falls through zero as time reaches
         the event. Negate so the crossing RISES through zero exactly when the
         Modelica condition becomes true (the when's rising edge), and fire only
         on that positive edge (affect_neg = nothing). A monotone time event is
         one-directional, so a single edge is correct and avoids double-firing. =#
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

#= The event iteration over the buffered relations (relationRefresh.jl):
   after every step, a relation that disagrees with the state is flipped
   with the hysteresis rule, and so on until nothing changes. =#
function emitRelationRefresh(relations)::Expr
  #= Emitted with no if-equation relations too: whens on a relation among
     the callbacks are iterated as well (none of either: a no-op). =#
  local entries = [:(($(QuoteNode(sym)), $(zc), $(scale))) for (sym, zc, scale) in relations]
  #= In the latest world, like the event list: the symbolic variables are
     globals bound while the model is built. =#
  return :(callbacks = OMBackend.CodeGeneration.withRelationRefresh(callbacks, problem,
                                                                     $(QuoteNode(ZC_HYSTERESIS)),
                                                                     Base.invokelatest(() -> Any[$(entries...)])))
end

#= The asserts of equation sections (simCode.asserts) as one callback that
   checks them after initialization and after each accepted step (asserts.jl).
   Conditions and messages are lowered like event affects: every variable is
   read through `observed`, `time` from the integrator. An assert whose
   expression cannot be lowered is reported and left out. =#
function emitAssertCallback(simCode)::Expr
  isempty(simCode.asserts) && return Expr(:block)
  local entries = Expr[]
  for a in simCode.asserts
    local obsAcc = Dict{Symbol,Symbol}()
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
      e === nothing || !(last(e).varKind isa Union{SimulationCode.PARAMETER, SimulationCode.ARRAY_PARAMETER})
    end
    #= Names, not the symbolic variables: a variable the compiler eliminated has
       no binding in the generated code, but may still be observed by name. =#
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

_isEffectCall(@nospecialize(e))::Bool = e isa DAE.CALL && e.attr.ty isa DAE.T_NORETCALL

#= The crossings of an assert condition's ordering relations on numbers
   (`a - b`), where the condition can change within a step (asserts.jl). One
   is 1.0 where the condition's short-circuit does not reach its relation
   (`x <= 0 or sqrt(x) < 2`: sqrt(x) of a negative threw). Not under noEvent(),
   nor inside another expression (a call's argument, a reduction). =#
function _assertCrossingExprs(@nospecialize(cond), obsAcc::Dict{Symbol,Symbol}, simCode)::Vector{Any}
  local out = Any[]
  local both = (r, c) -> r === true ? c : :($(r) && $(c))
  local walk = nothing
  walk = function (@nospecialize(e), reached)
    if e isa DAE.LBINARY && e.operator isa Union{DAE.AND, DAE.OR}
      walk(e.exp1, reached)
      local lhs = _daeBoolMem(e.exp1, obsAcc, simCode)
      walk(e.exp2, both(reached, e.operator isa DAE.AND ? lhs : :(!$(lhs))))
    elseif e isa DAE.LUNARY
      walk(e.exp, reached)
    elseif e isa DAE.IFEXP
      walk(e.expCond, reached)
      local c = _daeBoolMem(e.expCond, obsAcc, simCode)
      walk(e.expThen, both(reached, c))
      walk(e.expElse, both(reached, :(!$(c))))
    elseif e isa DAE.RELATION && e.operator isa Union{DAE.LESS, DAE.LESSEQ, DAE.GREATER, DAE.GREATEREQ} &&
           !(e.operator.ty isa Union{DAE.T_STRING, DAE.T_BOOL})
      local zc = :(Float64($(_daeExpToJuliaMem(e.exp1, obsAcc, simCode))) - Float64($(_daeExpToJuliaMem(e.exp2, obsAcc, simCode))))
      push!(out, reached === true ? zc : :($(reached) ? $(zc) : 1.0))
    end
    nothing
  end
  walk(cond, true)
  return out
end

#= An assert's condition; a call equation for its effects (BDAECreate) runs and holds. =#
function _assertConditionExpr(@nospecialize(cond), obsAcc::Dict{Symbol,Symbol}, simCode)
  _isEffectCall(cond) && return :($(_effectCallExpr(cond, obsAcc, simCode)); true)
  return _daeBoolMem(cond, obsAcc, simCode)
end

#= A call for its effects, its arguments read as values: a String as it is,
   a Boolean decoded from the arithmetic encoding, an array element by
   element (a literal one too: its elements may be variables). =#
function _effectCallExpr(call::DAE.CALL, obsAcc::Dict{Symbol,Symbol}, simCode)
  local arg(@nospecialize a) = @match a begin
    DAE.SCONST(str) => str
    DAE.ARRAY(_, _, es) => Expr(:vect, (arg(e) for e in es)...)
    _ where _isBooleanExp(a) => _daeBoolMem(a, obsAcc, simCode)
    _ => _daeExpToJuliaMem(a, obsAcc, simCode)
  end
  return Expr(:call, Symbol(OMBackend.canonicalName(string(call.path))), (arg(a) for a in call.expLst)...)
end

Base.@nospecializeinfer function _isBooleanExp(@nospecialize(e))::Bool
  @match e begin
    DAE.BCONST(__) || DAE.RELATION(__) || DAE.LBINARY(__) || DAE.LUNARY(__) => true
    DAE.CREF(_, ty) => ty isa DAE.T_BOOL
    DAE.CALL(attr = attr) => attr.ty isa DAE.T_BOOL
    _ => false
  end
end

#= An assert's message: string literals, `+` concatenation and String(x) of
   variables; anything else is shown as the Modelica expression. =#
Base.@nospecializeinfer function _assertMessageExpr(@nospecialize(msg::DAE.Exp), obsAcc::Dict{Symbol,Symbol}, simCode)
  local part(@nospecialize e) = @match e begin
    DAE.SCONST(s) => s
    DAE.BINARY(e1, DAE.ADD(__), e2) => :(string($(part(e1)), $(part(e2))))
    #= A String parameter is a module-level binding (createStringParameterAssignments),
       not a variable of the simulation: read through `observed`, the assert was
       left out as reading a variable the simulation does not keep. =#
    DAE.CREF(cr, _) where _isStringBinding(string(cr), simCode) =>
      Symbol(last(simCode.stringToSimVarHT[string(cr)]).name)
    _ => :(string($(_daeExpToJuliaMem(e, obsAcc, simCode))))
  end
  return try
    part(msg)
  catch _e
    #= The assert is still checked; its message is the Modelica text. =#
    OMBackend._fallback(_e, :assertMessage; only = LoweringFailure, impact = :result)
    string(msg)
  end
end

_isStringBinding(name::String, simCode)::Bool =
  (local e = get(simCode.stringToSimVarHT, name, nothing); e !== nothing && last(e).varKind isa SimulationCode.STRING)

_exprMentionsPrefix(@nospecialize(e), prefix::String) =
  (e isa Symbol && startswith(string(e), prefix)) || (e isa Expr && any(a -> _exprMentionsPrefix(a, prefix), e.args))

_exprMentions(@nospecialize(e), s::Symbol) = e === s || (e isa Expr && any(a -> _exprMentions(a, s), e.args))

#= Whether a relation has no continuous-time operand: only time, parameters,
   constants and discretes. Its crossing gets no hysteresis, so time events
   stay exact (as the if-equation branches keep pure time events exact). =#
function _withoutContinuousOperand(@nospecialize(rel::DAE.Exp), simCode)::Bool
  for cref in Util.getAllCrefs(rel)
    local name = string(cref)
    name == "time" && continue
    local entry = get(simCode.stringToSimVarHT, name, nothing)
    entry === nothing && return false
    local var = last(entry)
    (SimulationCode.isDiscrete(var) || SimulationCode.isParameter(var)) || return false
  end
  return true
end

#= The variables a when reads through pre(), in first-occurrence order;
   `edge(b)` and `change(b)` read pre(b). =#
function _clusterPreNames(@nospecialize(cond::DAE.Exp), assigns::Vector{Tuple{Symbol,Any,Bool}})::Vector{String}
  local names = String[]
  local visit = function (e::DAE.Exp, arg)
    if e isa DAE.CALL && e.path isa Absyn.IDENT && e.path.name in ("pre", "edge", "change")
      local a = listHead(e.expLst)
      a isa DAE.CREF && !(string(a.componentRef) in names) && push!(names, string(a.componentRef))
    end
    return (e, arg)
  end
  Util.traverseExpBottomUp(cond, visit, nothing)
  for (_, rhs, _) in assigns
    Util.traverseExpBottomUp(rhs, visit, nothing)
  end
  return names
end

#= The equations as a graph for _clusterCoupled: each equation's variable
   names (an if-equation: its conditions and all its branches' equations
   together), the equations each name occurs in, and the names a re-solve of
   the algebraic unknowns keeps: time, parameters and when-assigned
   discretes. Not the variables under der(): index reduction can make one
   algebraic (a dummy derivative), and a re-solve then moves it. =#
struct _EquationGraph
  names::Vector{OrderedSet{String}}
  occurs::Dict{String, Vector{Int}}
  fixed::Set{String}
end

function _equationGraph(simCode)::_EquationGraph
  local names = OrderedSet{String}[]
  for eq in simCode.residualEquations
    local ns = OrderedSet{String}()
    SimulationCode.collectCrefNames!(ns, eq.exp)
    push!(names, ns)
  end
  for ifEq in simCode.ifEquations
    local ns = OrderedSet{String}()
    for b in ifEq.branches
      SimulationCode.collectCrefNames!(ns, b.condition)
      foreach(eq -> SimulationCode.collectCrefNames!(ns, eq.exp), b.residualEquations)
    end
    push!(names, ns)
  end
  local occurs = Dict{String, Vector{Int}}()
  for (k, ns) in enumerate(names), n in ns
    push!(get!(occurs, n, Int[]), k)
  end
  local fixed = Set{String}(["time"])
  for (n, (_, v)) in simCode.stringToSimVarHT
    SimulationCode.isParameter(v) && push!(fixed, n)
  end
  union!(fixed, _collectWhenAssignedNames(simCode))
  return _EquationGraph(names, occurs, fixed)
end

#= Whether solving the algebraic unknowns again can change what a cluster
   reads: a read reachable from its members through the equations over the
   names a re-solve moves. An ideal diode's `s` in `off = s < 0` is; a
   switched-capacitor clock's `time >= pulseStart` and a constant resistance
   are not, and such a cluster settles without a solve of its own: the event
   iteration solves once after the sweep (CauerLowPassSC: 11 solves per clock
   event -> 1). =#
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

#= One DiscreteCluster (discreteClusters.jl) for a when on the relations
   `rels` that assigns `assigns`. Every distinct relation gets a buffer;
   `==` and `<>` (a crossing function that jumps between -0.5 and 0.5) and
   relations without a continuous-time operand have no hysteresis. A when
   lifted from discrete equations holds at every pass of an event (its
   equations do); another fires when its condition does. =#
function _discreteClusterSpec(@nospecialize(cond), rels::Vector{DAE.Exp},
                              assigns::Vector{Tuple{Symbol,Any,Bool}}, simCode, table::Bool,
                              graph::_EquationGraph)::Expr
  local condDAE = cond isa SimulationCode.Exp ? SimulationCode.toDAEExp(cond) : cond
  local buffered = DAE.Exp[]
  for r in rels
    any(b -> string(b) == string(r), buffered) || push!(buffered, r)
  end
  local relPins = Dict{String,Any}(string(r) => :(_rel[$k]) for (k, r) in enumerate(buffered))
  local relPre = Dict{String,Any}(string(r) => :(_relPre[$k]) for (k, r) in enumerate(buffered))
  local preNames = _clusterPreNames(condDAE, assigns)
  local preSubst = Dict{Symbol,Any}(Symbol(n) => :(_pre[$i]) for (i, n) in enumerate(preNames))
  local obsAcc = Dict{Symbol,Symbol}()
  local subst = Dict{Symbol,Any}()
  local stmts = Expr[]
  for (d, rhs, isInt) in assigns
    local vsym = Symbol("_v_", d)
    local valExpr = isInt ?
      :(Float64($(_daeExpToJuliaMem(rhs, obsAcc, simCode; initVal = :_init, subst = subst, relPins = relPins,
                                    preSubst = preSubst, relPre = relPre)))) :
      :($(_daeBoolMem(rhs, obsAcc, simCode; initVal = :_init, subst = subst, relPins = relPins,
                      preSubst = preSubst, relPre = relPre)) ? 1.0 : 0.0)
    push!(stmts, :(local $(vsym) = $(valExpr)))
    #= Later members read this one's new value (the body is in topological order). =#
    subst[d] = vsym
  end
  local fires = _isSynthesizedChangeCondition(condDAE) ? true :
    _daeBoolMem(condDAE, obsAcc, simCode; initVal = :_init, relPins = relPins, preSubst = preSubst, relPre = relPre)
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
  local strict = Bool[r.operator isa DAE.LESS || r.operator isa DAE.GREATER for r in buffered]
  local exact = Bool[_relationZeroSet(r) === nothing || _withoutContinuousOperand(r, simCode) for r in buffered]
  local eventOnly = Bool[_withoutContinuousOperand(r, simCode) && !_expMentionsTime(r) for r in buffered]
  local atStart = !_condHasInitial(condDAE) ? 0 : (table ? 2 : 1)
  local memberNames = String[string(d) for (d, _, _) in assigns]
  local readNames = OrderedSet{String}(string(o) for o in operands)
  foreach(r -> SimulationCode.collectCrefNames!(readNames, r), buffered)
  local coupled = _clusterCoupled(memberNames, readNames, graph)
  return :(OMBackend.CodeGeneration.DiscreteCluster($(memberNames),
                                                   Any[$([d for (d, _, _) in assigns]...)],
                                                   Any[$(reads...)], $(length(operands)), $(length(preNames)),
                                                   $(strict), $(exact), $(body), $(atStart), $(table), $(coupled),
                                                   $(eventOnly)))
end

"""
    emitDiscreteClusters(simCode) -> Expr

For a model with discrete clusters (`_usesDiscreteClusters`), the whens over
relations of coupled discretes are discrete clusters: a continuous callback each locates the crossings of their
relations, and the event iteration (emitted after this) evaluates them.
"""
function emitDiscreteClusters(simCode)::Expr
  local specs = _discreteClusterSpecs(simCode)
  isempty(specs) && return Expr(:block)
  #= In the latest world, like the event list: the symbolic variables are
     globals bound while the model is built. =#
  return :(callbacks = OMBackend.CodeGeneration.withDiscreteClusters(callbacks, problem,
                                                                      Base.invokelatest(() -> Any[$(specs...)])))
end

#= The DiscreteCluster constructions of the model's discrete clusters
   (emitDiscreteClusters); empty when it has none. =#
function _discreteClusterSpecs(simCode)::Vector{Expr}
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

#= Build MTK SymbolicContinuousCallbacks for the synthesised discrete-Boolean
   whens: one callback per relation zero set, whose affect rewrites the held
   discrete unknown from its defining expression. The affect re-evaluates
   the Boolean RHS at the (post-rootfind) event point, so it is direction-correct. =#
function createDiscreteBoolWhenEvents(simCode)::Vector{Expr}
  local events = Expr[]
  local usesClusters = _usesDiscreteClusters(simCode)
  for weq in simCode.whenEquations
    local rels = _extractChangeRelations(weq.whenEquation.condition, simCode)
    rels === nothing && continue
    #= `edge(b)` fires on the rising transition only (relation false->true);
       `change`/the gate lift fire on both. =#
    local isEdge = _isEdgeWhenCondition(weq.whenEquation.condition)
    local assigns = _gatherClusterAssigns(weq, simCode)
    assigns === nothing && continue
    #= Discrete clusters are evaluated by the event iteration (emitDiscreteClusters). =#
    usesClusters && continue
    #= One callback per relation zero set in the cluster (below).
       transformToMTKContinuousCondition normalises zc so relation-TRUE ⟺ zc<0
       for the group's first relation: the `=>` affect is the up-crossing
       (relation becomes FALSE) and affect_neg the down-crossing (relation becomes
       TRUE). Each callback rewrites the WHOLE ordered cluster with THIS group's
       relations pinned to their post-crossing values (others at their current
       value) so a coupled FSM recomputes consistently and direction-correctly on
       any member crossing, without the `f≈0` ambiguity. =#
    #= Initialize affect: set every discrete from its full rhs at t0. Only
       whens whose condition carries an `initial()` term (the synthesized
       lifter conditions) run their body at t0; user whens with plain
       relation conditions must not (Trapezoid `T_start = time` at t0 would
       destroy a negative-startTime phase, friction would mis-latch). =#
    local hasInit = _condHasInitial(weq.whenEquation.condition)
    #= Relations with the same zero set (`w_rel <= 0` and `w_rel > 0`) get ONE
       callback whose crossing pins all of them. With one callback each, both sat
       on the same root with opposite signs; where the integrator stopped exactly
       on it they fired alternately, each pinning its own relation and reading the
       other from the state at the root, and flipped the cluster back and forth
       without time advancing (MSL Rotational OneWayClutchDisengaged, maxiters).
       Rising-edge-only whens keep one callback per relation. =#
    local groups = isEdge ? Vector{Any}[Any[r] for r in rels] : _groupRelationsByZeroSet(rels)
    #= A synthesized cluster that reads its own pre() values iterates to a
       fixpoint within the event (_eventIterAffectParts); the others need a
       single pass. =#
    local selfPre = _isSynthesizedChangeCondition(weq.whenEquation.condition) && _clusterReadsOwnPre(assigns)
    local lowerable = (selfPre || hasInit) && _eventIterLowerable(assigns, simCode)
    local eventIteration = selfPre && lowerable
    #= The t0 pass is imperative wherever the body lowers. As equations,
       ModelingToolkit solves it together with the whole algebraic system, every
       held discrete an unknown of that solve, and at t0 that system sits on its
       singular points (a Mean block's sqrt(y) at y = 0, a zero current's angle):
       the solve fails (MTK's UnsolvableCallbackError). The crossings keep the
       equation form, whose solve also moves the algebraic unknowns after a
       switch (MSL CauerLowPassSC's switched capacitors). =#
    local affInit = !hasInit ? nothing : lowerable ?
      _imperativeClusterAffect(assigns, simCode; initVal = true, iterate = selfPre) :
      :([$([_discreteAffectEqInit(d, r, ii, simCode) for (d, r, ii) in assigns]...)])
    for (relIdx, grp) in enumerate(groups)
      local rel = grp[1]
      local zc = transformToMTKContinuousCondition(rel, simCode)
      local pinsFalse = _zeroSetPins(grp, rel, false)
      local pinsTrue = _zeroSetPins(grp, rel, true)
      #= The `initial()` term: only the first group's callback carries it, so the
         discretes are set once at t0 (no double-apply). =#
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
        #= `edge(b)`: the body runs when the relation becomes TRUE (the
           down-crossing of zc, affect_neg); the up-crossing does nothing. =#
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

#= The target of field `f` of a record target of a tuple assignment: the
   field's variable (`r_f`), or an omitted output's. =#
function _recordFieldTarget(@nospecialize(target), f::String)
  local d = target isa SimulationCode.Exp ? SimulationCode.toDAEExp(target) : target
  (d isa DAE.CREF && d.componentRef isa DAE.WILD) && return d
  d isa DAE.CREF || unsupported("this record target of a tuple assignment in a when", target)
  local name = OMBackend.canonicalName(string(SimulationCode.string(d), ".", f))
  return DAE.CREF(DAE.CREF_IDENT(name, DAE.T_REAL_DEFAULT, MetaModelica.nil), DAE.T_REAL_DEFAULT)
end

#= A target missing from the variable table would be bound as a local of the
   affect, its value lost. =#
function _emitWhenTupleElementAssignMTK!(res::Vector{Expr}, lhs,
                                          rhsAccess, simCode::SimulationCode.SIM_CODE)
  @match lhs begin
    DAE.CREF(DAE.WILD(), _) => nothing
    DAE.CREF(__) => begin
      local name = SimulationCode.string(lhs)
      local entry = get(simCode.stringToSimVarHT, name, nothing)
      entry === nothing && unsupported("a tuple target in a when that is no variable", lhs)
      local (_, var) = entry
      push!(res, quote
              idx = lookuptableStates[Symbol($(string(var.name)))]
              integrator.u[idx] = $rhsAccess
            end)
    end
    DAE.ARRAY(_, _, elements) => begin
      local i = 0
      for elem in elements
        i += 1
        _emitWhenTupleElementAssignMTK!(res, elem, :($rhsAccess[$i]), simCode)
      end
    end
    SimulationCode.EXP_CREF(cref, _) => begin
      local name = string(cref)
      local entry = get(simCode.stringToSimVarHT, name, nothing)
      entry === nothing && unsupported("a tuple target in a when that is no variable", lhs)
      local (_, var) = entry
      push!(res, quote
              idx = lookuptableStates[Symbol($(string(var.name)))]
              integrator.u[idx] = $rhsAccess
            end)
    end
    SimulationCode.ARRAY_EXP(_, _, elements) => begin
      local i = 0
      for elem in elements
        i += 1
        _emitWhenTupleElementAssignMTK!(res, elem, :($rhsAccess[$i]), simCode)
      end
    end
    _ => unsupported("tuple-LHS element in a when statement", lhs)
  end
  return res
end

function createWhenStatementsMTK(whenStatements, simCode::SimulationCode.SIM_CODE; varPrefix = "", varSuffix = "")::Vector{Expr}
  local res::Array{Expr} = []
  local nWhenStatements = 0
  for _ in whenStatements
    nWhenStatements += 1
  end
  @debug "[MTK GEN: when] createWhenStatementsMTK" statements=nWhenStatements
  for wStmt in whenStatements
    if wStmt isa BDAE.ASSIGN || wStmt isa SimulationCode.ASSIGN
      if wStmt.left isa DAE.TUPLE || wStmt.left isa SimulationCode.TUPLE
        local tupSym = gensym(:tupResult)
        local rhsExpr = expToJuliaExpMTK(wStmt.right, simCode;
                                         varPrefix = varPrefix, varSuffix = varSuffix)
        push!(res, :(local $tupSym = $rhsExpr))
        #= A generated function returns a record output as its fields in place:
           a record target takes them into its fields' variables, an omitted
           one skips them (each target after a record took a field before). =#
        local rhsTy = SimulationCode.toDAEExp(wStmt.right)
        local outputTypes = rhsTy isa DAE.CALL && rhsTy.attr.ty isa DAE.T_TUPLE ? collect(rhsTy.attr.ty.types) : nothing
        local pos = 0
        for (k, elem) in enumerate(wStmt.left.PR)
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
      else
        # SimulationCode.ASSIGN.left is ::Exp post-migration; HT keys are DAE-stringified.
        local leftStr = SimulationCode.string(SimulationCode.toDAEExp(wStmt.left))
        (index, var) = simCode.stringToSimVarHT[leftStr]
        local lhsSym = Symbol(string(var.name))
        local rhsE = expToJuliaExpMTK(wStmt.right, simCode; varPrefix = varPrefix, varSuffix = varSuffix)
        push!(res, quote
                idx = lookuptableStates[Symbol($(string(var.name)))]
                integrator.u[idx] = $(rhsE)
                $(lhsSym) = integrator.u[idx]
              end)
      end
    elseif wStmt isa BDAE.REINIT || wStmt isa SimulationCode.REINIT
      (index, var) = simCode.stringToSimVarHT[SimulationCode.string(wStmt.stateVar)]
      push!(res, quote
              idx = lookuptableStates[Symbol($(string(var.name)))]
              OMBackend.CodeGeneration.noteReinit!(integrator, idx)
              integrator.u[idx] = $(expToJuliaExpMTK(wStmt.value,
                                                     simCode; varPrefix = varPrefix, varSuffix = varSuffix))
            end)
    elseif wStmt isa BDAE.TERMINATE || wStmt isa SimulationCode.TERMINATE
      local msgExpr = expToJuliaExpMTK(wStmt.message, simCode;
                                        varPrefix = varPrefix, varSuffix = varSuffix)
      push!(res, quote
              @info "Modelica terminate() reached" message=$(msgExpr)
              OMBackend.DifferentialEquations.terminate!(integrator)
            end)
    elseif wStmt isa BDAE.NORETCALL || wStmt isa SimulationCode.NORETCALL
      local callExpr = expToJuliaExpMTK(wStmt.exp, simCode;
                                         varPrefix = varPrefix, varSuffix = varSuffix)
      push!(res, quote
              $(callExpr)
            end)
    elseif wStmt isa BDAE.ASSERT || wStmt isa SimulationCode.ASSERT
      local condExpr = expToJuliaExpMTK(wStmt.condition, simCode;
                                         varPrefix = varPrefix, varSuffix = varSuffix)
      local msgExpr = expToJuliaExpMTK(wStmt.message, simCode;
                                        varPrefix = varPrefix, varSuffix = varSuffix)
      #= AssertionLevel.error stops the simulation, as an equation's assert
         does (asserts.jl); warning reports it. =#
      local violated = if AlgorithmicCodeGeneration.isWarningAssertionLevel(SimulationCode.toDAEExp(wStmt.level))
        :(@warn string("Assertion violated at time ", integrator.t, ": ", $(msgExpr)))
      else
        :(throw(OMBackend.CodeGeneration.ModelicaAssertionError(integrator.t, string($(msgExpr)),
                                                                 $(string(SimulationCode.toDAEExp(wStmt.condition))))))
      end
      #= The lowering encodes and/or/not arithmetically: a number, not a Bool. =#
      push!(res, quote
              local _holds = $(condExpr)
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

#= True when a when-equation's condition is the Modelica `terminal()` operator. =#
function _isTerminalWhen(@nospecialize(eq))::Bool
  (eq isa BDAE.WHEN_EQUATION || eq isa SimulationCode.WHEN_EQUATION) || return false
  return @match SimulationCode.toDAEExp(eq.whenEquation.condition) begin
    DAE.CALL(Absyn.IDENT("terminal"), _, _) => true
    _ => false
  end
end

#= Post-solve runner for `when terminal()` bodies, or `nothing` when the model
   has none (so models without a terminal event are unchanged). The bodies
   reuse `createWhenStatementsMTK` by mocking `integrator` from the final
   solution point — writes land in `_sol.u[end]` using the same state-index
   convention the discrete-callback affects rely on. Runs only on success. =#
function createTerminalBodyRunner(simCode::SimulationCode.SIM_CODE)
  local terminalWhens = filter(_isTerminalWhen, simCode.whenEquations)
  isempty(terminalWhens) && return nothing
  local modelFns = OrderedSet(replace(f.name, "." => "_") for f in simCode.functions)
  local calledNames = OrderedSet{String}()
  local perWhen = Expr[]
  for eq in terminalWhens
    local body = eq.whenEquation.whenStmtLst
    for s in body
      collectCalledFunctionNames!(calledNames, s)
    end
    for c in vcat(map(s -> getRHSVariables(s), body)...)
      local entry = get(simCode.stringToSimVarHT, string(c), nothing)
      #= String parameters are emitted as module-level constants; referencing them
         directly avoids shadowing that binding with a nonexistent state lookup. =#
      entry !== nothing && entry[2].varKind isa SimulationCode.STRING && continue
      push!(perWhen, Expr(:(=), Symbol(string(c)), getIdxForLookupMTK(c, simCode)))
    end
    append!(perWhen, createWhenStatementsMTK(body, simCode))
  end
  #= Bind external functions the body calls to their concrete OMBackend.CodeGeneration
     RTG wrapper: the bare model-module name is the @register_symbolic binding (symbolic
     only), which has no method for concrete runtime arguments. =#
  local fnRebinds = Expr[]
  for n in calledNames
    local nn = replace(n, "." => "_")
    nn in modelFns && push!(fnRebinds, :(local $(Symbol(nn)) = OMBackend.CodeGeneration.$(Symbol(nn))))
  end
  return quote
    if _sol.retcode == ModelingToolkit.SciMLBase.ReturnCode.Success
      #= Best-effort: a terminal body runs after a completed, valid solution, so a
         body we cannot evaluate (e.g. an unsupported external call) warns rather
         than discarding the result. =#
      try
        let integrator = (u = _sol.u[end], t = _sol.t[end], f = _sol.prob.f, dt = 0.0, ps = _sol.prob.ps),
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
