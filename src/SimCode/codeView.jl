#= The simulation code as the code generators see it: every key, reference and name renamed to its identifier
   in the generated code (`ReferenceIndex.codeNames`), derived from the references once at lowering. The passes key
   the simulation code by the references' flat names; the generated code, OM.jl's results and the runtime know
   the variables by these identifiers (`a_b[1][2]`, `der(x)`). =#

"""
    codeView(simCode) -> SIM_CODE

`simCode` with every variable's key, every reference to it (`SimCref.sym`: expressions, statements, `reinit` and
structural targets, the variables' attributes and bindings, their IR kinds' names) and every list of names (aliases,
eliminated, irreducible, shared, top, folded variables) as its identifier (`codeName`); the sub-models' too. A name of
no variable (`time`, an iterator, a function's local) as it is. `codeNames` then maps each identifier to itself.
"""
function codeView(sc::SIM_CODE)::SIM_CODE
  local rn::Renaming = Renaming(sc.referenceIndex.codeNames)
  local name::Function = (k::String) -> get(rn.names, k, k)
  local ht::OrderedDict{String, Tuple{Int, SIMVAR}} = OrderedDict{String, Tuple{Int, SIMVAR}}()
  for (k::String, entry::Tuple{Int, SIMVAR}) in sc.stringToSimVarHT
    ht[name(k)] = (first(entry), renamed(rn, last(entry)))
  end
  return update(sc; stringToSimVarHT = ht,
                residualEquations = RESIDUAL_EQUATION[renamed(rn, eq) for eq in sc.residualEquations],
                initialEquations = Equation[renamed(rn, eq) for eq in sc.initialEquations],
                whenEquations = WHEN_EQUATION[renamed(rn, eq) for eq in sc.whenEquations],
                ifEquations = IF_EQUATION[renamed(rn, eq) for eq in sc.ifEquations],
                structuralTransitions = StructuralTransition[renamed(rn, t) for t in sc.structuralTransitions],
                subModels = SIM_CODE[codeView(m) for m in sc.subModels],
                sharedVariables = String[name(k) for k in sc.sharedVariables],
                topVariables = String[name(k) for k in sc.topVariables],
                sharedEquations = Equation[renamed(rn, eq) for eq in sc.sharedEquations],
                irreducibleVariables = String[name(k) for k in sc.irreducibleVariables],
                eliminatedEquations = RESIDUAL_EQUATION[renamed(rn, eq) for eq in sc.eliminatedEquations],
                eliminatedVariables = String[name(k) for k in sc.eliminatedVariables],
                aliasMap = AliasEntry[AliasEntry(name(e.eliminatedName), name(e.representativeName), e.negated) for e in sc.aliasMap],
                initialAlgorithms = INITIAL_ALGORITHM[INITIAL_ALGORITHM(Statement[renamed(rn, s) for s in a.statements], a.section)
                                                      for a in sc.initialAlgorithms],
                asserts = ASSERT_EQUATION[renamed(rn, a) for a in sc.asserts],
                foldedVariables = String[name(k) for k in sc.foldedVariables],
                referenceIndex = renamed(rn, sc.referenceIndex))
end

"The identifiers of `codeView`, by the keys they rename; as a traversal's `recur`, an expression or statement renamed."
struct Renaming
  names::Dict{String, String}
end
(rn::Renaming)(e::Exp, a::Nothing) = (renamed(rn, e), a)
(rn::Renaming)(s::Statement, a::Nothing) = (renamed(rn, s), a)

"The reference index with its keys renamed to their identifiers (`codeView`); each identifier its own then."
function renamed(rn::Renaming, ri::ReferenceIndex)::ReferenceIndex
  local name::Function = (k::String) -> get(rn.names, k, k)
  return ReferenceIndex(Dict{String, IR.ComponentRef}(name(k) => r for (k, r) in ri.references),
                        Dict{Symbol, String}(r => name(k) for (r, k) in ri.keyByReference),
                        Dict{String, Symbol}(name(k) => a for (k, a) in ri.arrayOf),
                        Dict{String, String}(id => id for id in values(ri.codeNames)),
                        Dict{String, Vector{String}}(name(k) => n for (k, n) in ri.elementNames))
end

"""
A reference renamed: its key's identifier, its subscripts and IR reference kept. A reference to no variable of the
table with an IR reference (a record whose fields are the variables: `r` of `r.T`) by its identifier derived as a
variable's (`derivedKey`), an element of an array that is no variable (`subs`: a constant record's array field,
`Medium.data.alow[1]`) by its array's (the reference without its last subscripts, as the lowering keyed it,
`simCrefExp`); one without (`time`, an iterator, a function's local) as it is.
"""
function renamed(rn::Renaming, c::SimCref)::SimCref
  local id::Union{Nothing, String} = get(rn.names, String(c.sym), nothing)
  (id === nothing && c.ref !== nothing) && (id = derivedKey(isempty(c.subs) ? c.ref : IR.withoutLastSubscripts(c.ref)))
  return id === nothing ? c : SimCref(Symbol(id), c.subs, c.ref)
end

"An expression with its references renamed (`mapExp`)."
renamed(rn::Renaming, e::Exp)::Exp = mapExp(x::Exp -> _renamedNode(rn, x), e)
_renamedNode(rn::Renaming, e::EXP_CREF)::Exp = EXP_CREF(renamed(rn, e.cref), e.ty)
_renamedNode(rn::Renaming, e::DER)::Exp = DER(renamed(rn, e.cref))
_renamedNode(rn::Renaming, e::PRE)::Exp = PRE(renamed(rn, e.cref), e.ty)
_renamedNode(::Renaming, e::Exp)::Exp = e

"A statement with its expressions, nested statements and targets (`reinit`'s, the structural changes') renamed."
renamed(rn::Renaming, s::Statement)::Statement = first(traverseFields(s, rn, nothing))
renamed(rn::Renaming, s::REINIT)::Statement =
  REINIT(renamed(rn, s.stateVar), renamed(rn, s.value), s.source, Exp[renamed(rn, x) for x in s.subscripts])
renamed(rn::Renaming, s::RECOMPILATION)::Statement = RECOMPILATION(renamed(rn, s.componentToChange), renamed(rn, s.newValue))
renamed(rn::Renaming, s::AGENTIC_RECOMPILATION)::Statement =
  AGENTIC_RECOMPILATION(SimCref[renamed(rn, c) for c in s.componentsToChange], s.prompt, s.initialEquations)

"An equation, when-clause, branch or structural transition with its expressions and statements renamed."
function renamed(rn::Renaming, x::T)::T where {T <: Union{Equation, Construct, WHEN_STMTS, StructuralTransition}}
  return first(traverseFields(x, rn, nothing))
end

"A variable renamed: its key, its kind's names (a state's derivative's state), its binding and attributes, its IR kind's names."
function renamed(rn::Renaming, v::SIMVAR)::SIMVAR
  local name::Function = (k::String) -> get(rn.names, k, k)
  local a::SimVarAttributes = v.attributes
  local e::Function = (x::Union{Nothing, Exp}) -> x === nothing ? nothing : renamed(rn, x)
  local attributes::SimVarAttributes = SimVarAttributes(e(a.start), e(a.fixed), e(a.nominal), e(a.min), e(a.max), e(a.unit),
                                                        e(a.displayUnit), e(a.quantity), a.stateSelect, a.isProtected)
  local irKind::Union{Nothing, IR.VariableKind} = v.irKind === nothing ? nothing : IR.mapKindNames((x::Symbol) -> Symbol(name(String(x))), v.irKind)
  return SIMVAR(name(v.name), v.index, renamedKind(rn, v.varKind), attributes, v.elementType, v.dims, irKind, v.irOrigin, v.ref)
end

"A variable's category with the names and expressions it holds renamed."
renamedKind(rn::Renaming, k::STATE_DERIVATIVE)::SimVarType = STATE_DERIVATIVE(get(rn.names, k.varName, k.varName))
renamedKind(rn::Renaming, k::SimVarType)::SimVarType = withBinding(k, (local b = bindingOf(k); b === nothing ? nothing : renamed(rn, b)))
