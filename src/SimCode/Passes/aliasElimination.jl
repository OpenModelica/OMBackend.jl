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

#= Alias elimination: the unknowns equal to another unknown (or to its negation) replaced by
   one representative. =#

"""
    AliasEquation(first, second, negated, equation)

The residual number `equation`, `a - b` (`a + b`: `negated`, `a = -b`) between the unknowns
`first` and `second`.
"""
struct AliasEquation
  first::String
  second::String
  negated::Bool
  equation::Int
end

"The class of values an alias pair must share: `:real`, `:bool`, `:int`, `:enum`; `:other` for none."
aliasClass(::TYPE_REAL)::Symbol = :real
aliasClass(::TYPE_BOOL)::Symbol = :bool
aliasClass(::TYPE_INTEGER)::Symbol = :int
aliasClass(::TYPE_ENUM)::Symbol = :enum
aliasClass(t::TYPE_ARRAY)::Symbol = aliasClass(t.elementType)
aliasClass(::SType)::Symbol = :other

"`e` without trailing `- 0` and `+ 0` (a connection's `(a + b) - 0.0`)."
function withoutZeroTerms(e::Exp)::Exp
  while e isa BINARY && (e.op === OP_SUB || e.op === OP_ADD) && isZeroLiteral(e.exp2)
    e = e.exp1
  end
  return e
end

"A variable reference under unary minuses, and whether their number is odd; `nothing` for another expression."
function signedReference(e::Exp)::Union{Nothing, Tuple{EXP_CREF, Bool}}
  local negated = false
  while e isa UNARY && e.op === OP_UMINUS
    negated = !negated
    e = e.exp
  end
  return e isa EXP_CREF ? (e, negated) : nothing
end

"""
    aliasPair(e, ht)

The two unknowns of the residual `e` if it is `±a ± b` (a `- 0` or `+ 0` after it peeled):
their references and whether `a = -b`; both variables of the table, unknowns, of one class
(`aliasClass`). `nothing` for another residual.
"""
function aliasPair(e::Exp, ht::HT)::Union{Nothing, Tuple{EXP_CREF, EXP_CREF, Bool}}
  local p = withoutZeroTerms(e)
  (p isa BINARY && (p.op === OP_SUB || p.op === OP_ADD)) || return nothing
  local (r1, r2) = (signedReference(p.exp1), signedReference(p.exp2))
  (r1 === nothing || r2 === nothing) && return nothing
  local ((a, negA), (b, negB)) = (r1, r2)
  local (entryA, entryB) = (get(ht, crefName(a.cref), nothing), get(ht, crefName(b.cref), nothing))
  (entryA === nothing || entryB === nothing) && return nothing
  local class = aliasClass(a.ty)
  (class === :other || class !== aliasClass(b.ty)) && return nothing
  (isUnknownVarKind(last(entryA).varKind) && isUnknownVarKind(last(entryB).varKind)) || return nothing
  #= ±a ± b = 0 with the coefficient of a made 1: b's is positive for a = -b =#
  return (a, b, !xor(xor(negA, negB), p.op === OP_SUB))
end

"The preference of a variable category for the representative of its aliases."
categoryPriority(::STATE)::Int = 100
categoryPriority(::STATE_DERIVATIVE)::Int = 90
categoryPriority(::DISCRETE)::Int = 80
categoryPriority(::ALG_VARIABLE)::Int = 20
categoryPriority(::ARRAY)::Int = 10
categoryPriority(::SimVarType)::Int = 0

"""
    representativePriority(name, v, irreducible)

How much the variable `v` is preferred as the representative of its aliases: by its category,
more for one the events need (irreducible), a little more for one with a start value (its
start is kept).
"""
function representativePriority(name::String, v::SIMVAR, irreducible::Set{String})::Int
  local hasStart = v.attributes.start !== nothing && v.elementType isa Union{TYPE_REAL, TYPE_INTEGER, TYPE_BOOL}
  return categoryPriority(v.varKind) + (name in irreducible ? 60 : 0) + (hasStart ? 5 : 0)
end

"""
    aliasComponents(pairs)

The connected components of the alias graph, each its variables with their negation relative
to its first, in breadth-first order, and its equations (each once, as their edges are first
crossed).
"""
function aliasComponents(pairs::Vector{AliasEquation})::Vector{Tuple{Vector{Tuple{String, Bool}}, Vector{Int}}}
  local adjacent = OrderedDict{String, Vector{Tuple{String, Bool, Int}}}()
  for (k, p) in enumerate(pairs)
    push!(get!(adjacent, p.first, Tuple{String, Bool, Int}[]), (p.second, p.negated, k))
    push!(get!(adjacent, p.second, Tuple{String, Bool, Int}[]), (p.first, p.negated, k))
  end
  local negationOf = Dict{String, Bool}()
  local crossed = Set{Int}()
  local components = Tuple{Vector{Tuple{String, Bool}}, Vector{Int}}[]
  for start in keys(adjacent)
    haskey(negationOf, start) && continue
    local (members, equations) = (Tuple{String, Bool}[], Int[])
    local queue = [(start, false)]
    negationOf[start] = false
    while !isempty(queue)
      local (node, negated) = popfirst!(queue)
      push!(members, (node, negated))
      for (neighbor, edgeNegated, k) in adjacent[node]
        k in crossed || (push!(crossed, k); push!(equations, pairs[k].equation))
        haskey(negationOf, neighbor) && continue
        negationOf[neighbor] = xor(negated, edgeNegated)
        push!(queue, (neighbor, negationOf[neighbor]))
      end
    end
    push!(components, (members, equations))
  end
  return components
end

"The root of `x` in the union-find forest `parent` (paths halved)."
function findRoot!(parent::Dict{String, String}, x::String)::String
  haskey(parent, x) || (parent[x] = x)
  while parent[x] != x
    parent[x] = parent[parent[x]]
    x = parent[x]
  end
  return x
end

"Join the sets of `a` and `b`; whether they were apart."
function join!(parent::Dict{String, String}, a::String, b::String)::Bool
  local (ra, rb) = (findRoot!(parent, a), findRoot!(parent, b))
  ra == rb && return false
  parent[ra] = rb
  return true
end

"""
    eliminateAliasVariables(simCode)

The unknowns of the residuals `a - b` and `a + b` (`aliasPair`) replaced by one representative
per connected component of such residuals (`representativePriority`), negated where the path
to it negates; not a shared variable, nor one the events need (irreducible) unless both are
states. A component's residuals that, substituted, read one variable or join variables joined
already are removed; the others stay. The representative takes the eliminated variables'
attributes it lacks (`mergeAliasAttributes`). Every equation, when-clause, initial equation
and algorithm, assert is substituted; an eliminated variable still read is kept. Each
eliminated variable is paired with a removed residual of it (`observationPairs`) for its
observed equation. Nothing for a model with modes or translated again at its structural
changes.
"""
function eliminateAliasVariables(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  local residuals = simCode.residualEquations
  local shared = Set{String}(simCode.sharedVariables)
  local irreducible = Set{String}(simCode.irreducibleVariables)
  local pairs = AliasEquation[]
  local refs = Dict{String, EXP_CREF}()
  for (i, eq) in enumerate(residuals)
    local p = aliasPair(eq.exp, ht)
    p === nothing && continue
    local (a, b, negated) = p
    local (n1, n2) = (crefName(a.cref), crefName(b.cref))
    (n1 == n2 || n1 in shared || n2 in shared) && continue
    push!(pairs, AliasEquation(n1, n2, negated, i))
    refs[n1] = a
    refs[n2] = b
  end
  isempty(pairs) && return simCode
  local namesOf = Dict{Int, Tuple{String, String}}(p.equation => (p.first, p.second) for p in pairs)
  local replacements = Replacements()
  local representativeOf = Dict{String, String}()
  local entries = AliasEntry[]
  local mergedAttributes = OrderedDict{String, SimVarAttributes}()
  local removed = Set{Int}()
  for (members, equations) in aliasComponents(pairs)
    local (best, bestPriority, bestNegated) = ("", -1, false)
    for (name, negated) in members
      local priority = representativePriority(name, last(ht[name]), irreducible)
      priority > bestPriority && ((best, bestPriority, bestNegated) = (name, priority, negated))
    end
    local representative = last(ht[best])
    for (name, negated) in members
      name == best && continue
      local v = last(ht[name])
      (name in irreducible && !(representative.varKind isa STATE && v.varKind isa STATE)) && continue
      local n = xor(negated, bestNegated)
      replacements[name] = Replacement(refs[best], n, false)
      representativeOf[name] = best
      push!(entries, AliasEntry(name, best, n))
      mergedAttributes[best] = mergeAliasAttributes(get(mergedAttributes, best, representative.attributes),
                                                    representative.elementType, v, n)
    end
    #= the residuals that, substituted, join no two apart variables go =#
    local parent = Dict{String, String}()
    for i in equations
      local (r1, r2) = map(n -> get(representativeOf, n, n), namesOf[i])
      (r1 == r2 || !join!(parent, r1, r2)) && push!(removed, i)
    end
  end
  isempty(replacements) && return simCode
  local newResiduals = RESIDUAL_EQUATION[substitute(eq, replacements) for (i, eq) in enumerate(residuals) if !(i in removed)]
  local ifEquations = IF_EQUATION[substitute(ifEq, replacements) for ifEq in simCode.ifEquations]
  local whenEquations = WHEN_EQUATION[WHEN_EQUATION(substitute(w.whenEquation, replacements), w.source, w.attr)
                                      for w in simCode.whenEquations]
  local initialEquations = Equation[substituteInitial(eq, replacements) for eq in simCode.initialEquations]
  local initialAlgorithms = INITIAL_ALGORITHM[substitute(a, replacements) for a in simCode.initialAlgorithms]
  #= an eliminated variable the substitution missed is kept =#
  local read = OrderedSet{String}()
  foreach(eq -> crefNames!(read, eq.exp), newResiduals)
  for ifEq in ifEquations, b in ifEq.branches, eq in b.residualEquations
    crefNames!(read, eq.exp)
  end
  for eq in initialEquations, e in equationExps(eq)
    crefNames!(read, e)
  end
  foreach(a -> statementCrefNames!(read, a.statements), initialAlgorithms)
  foreach(w -> whenCrefNames!(read, w.whenEquation), whenEquations)
  local stillRead = Set{String}(n for n in keys(replacements) if n in read)
  isempty(stillRead) || @debug "[SIMCODE: $(simCode.name): eliminateAliasVariables] eliminated variables still read, kept" stillRead
  local newHT = copy(ht)
  local eliminated = String[n for n in keys(replacements) if !(n in stillRead)]
  foreach(n -> deleteEliminated!(newHT, ht, n, representativeOf[n]), eliminated)
  for (name, attributes) in mergedAttributes
    haskey(newHT, name) || continue
    local (index, v) = newHT[name]
    attributes == v.attributes || (newHT[name] = (index, SIMVAR(v.name, v.index, v.varKind, attributes, v.elementType)))
  end
  local (observedVariables, observedEquations) = observationPairs(eliminated, pairs, removed, residuals, replacements, refs)
  local aliasMap = withAliases(simCode.aliasMap, filter(e -> !(e.eliminatedName in stillRead), entries))
  local eliminatedSet = Set{String}(eliminated)
  return update(simCode; residualEquations = newResiduals, initialEquations, initialAlgorithms, stringToSimVarHT = newHT,
                ifEquations, whenEquations, aliasMap, asserts = ASSERT_EQUATION[substitute(a, replacements) for a in simCode.asserts],
                irreducibleVariables = filter(n -> !(n in eliminatedSet), simCode.irreducibleVariables),
                eliminatedEquations = vcat(RESIDUAL_EQUATION[substitute(eq, replacements) for eq in simCode.eliminatedEquations],
                                           observedEquations),
                eliminatedVariables = vcat(simCode.eliminatedVariables, observedVariables))
end

"""
    deleteEliminated!(newHT, ht, name, representative)

The eliminated variable `name` deleted from `newHT`; of a state aliased to a state, its
derivative too (the substitution reads the representative's).
"""
function deleteEliminated!(newHT::HT, ht::HT, name::String, representative::String)::Nothing
  delete!(newHT, name)
  last(ht[name]).varKind isa STATE && last(ht[representative]).varKind isa STATE && delete!(newHT, "der($(name))")
  return nothing
end

"""
    withAliases(aliasMap, entries)

The alias map `aliasMap` with the aliases `entries` of a pass added (a name aliased already
keeps its alias): an alias whose representative the pass eliminated points to that one's
representative instead, the negations composed, so that a representative stays a variable
(`checkAliasConsistency`; MSL SMPM_Braking: inertiaLoad.phi, an alias of the rotor's phi, which
the second alias elimination eliminates for the stray load's phi).
"""
function withAliases(aliasMap::Vector{AliasEntry}, entries::Vector{AliasEntry})::Vector{AliasEntry}
  local entryOf = Dict{String, AliasEntry}(e.eliminatedName => e for e in entries)
  local out = AliasEntry[]
  local known = Set{String}()
  for e in aliasMap
    local r = get(entryOf, e.representativeName, nothing)
    push!(out, r === nothing ? e : AliasEntry(e.eliminatedName, r.representativeName, xor(e.negated, r.negated)))
    push!(known, e.eliminatedName)
  end
  for e in entries
    e.eliminatedName in known && continue
    push!(out, e)
    push!(known, e.eliminatedName)
  end
  return out
end

"""
    observationPairs(eliminated, pairs, removed, residuals, replacements, refs)

The eliminated variables, each with a removed alias residual of it (a maximum matching:
augmenting paths, in the variables' order), for the observed equations; a variable without
one (its residual stayed: the other side was irreducible) with `x - rep` (`x + rep`).
"""
function observationPairs(eliminated::Vector{String}, pairs::Vector{AliasEquation}, removed::Set{Int},
                          residuals::Vector{RESIDUAL_EQUATION}, replacements::Replacements,
                          refs::Dict{String, EXP_CREF})::Tuple{Vector{String}, Vector{RESIDUAL_EQUATION}}
  local eliminatedSet = Set{String}(eliminated)
  local incidence = AliasEquation[p for p in pairs if p.equation in removed && (p.first in eliminatedSet || p.second in eliminatedSet)]
  local equationOf = Dict{String, Int}()
  local variableOf = Dict{Int, String}()
  foreach(n -> augment!(n, Set{Int}(), incidence, variableOf, equationOf), eliminated)
  local variables = String[n for n in eliminated if haskey(equationOf, n)]
  local equations = RESIDUAL_EQUATION[residuals[equationOf[n]] for n in variables]
  for n in eliminated
    (haskey(equationOf, n) || !haskey(refs, n)) && continue
    local r = replacements[n]
    push!(variables, n)
    push!(equations, RESIDUAL_EQUATION(BINARY(refs[n], r.negated ? OP_ADD : OP_SUB, r.ref), NO_SOURCE, EQ_ATTR_DEFAULT))
  end
  return (variables, equations)
end

"""
    augment!(name, seen, incidence, variableOf, equationOf)

Whether an augmenting path from the variable `name` pairs it with an equation of `incidence`
(one not in `seen`), re-pairing the variables on the path.
"""
function augment!(name::String, seen::Set{Int}, incidence::Vector{AliasEquation}, variableOf::Dict{Int, String},
                  equationOf::Dict{String, Int})::Bool
  for p in incidence
    (p.first == name || p.second == name) || continue
    p.equation in seen && continue
    push!(seen, p.equation)
    if !haskey(variableOf, p.equation) || augment!(variableOf[p.equation], seen, incidence, variableOf, equationOf)
      variableOf[p.equation] = name
      equationOf[name] = p.equation
      return true
    end
  end
  return false
end

#= ==================================================================================== =#
#= The attributes of the eliminated aliases                                             =#
#= ==================================================================================== =#

"""
    attributesGiven(elementType, a)

Whether a variable has attributes of its own: a Real one always; another if it gives one or
is protected (the frontend gives an Integer, Boolean or enumeration variable without them
none, and a merge leaves it so).
"""
function attributesGiven(elementType::SType, a::SimVarAttributes)::Bool
  elementType isa TYPE_REAL && return true
  return a.isProtected || any(x -> x !== nothing, (a.start, a.fixed, a.nominal, a.min, a.max, a.unit, a.displayUnit,
                                                   a.quantity, a.stateSelect))
end

"`a` with the given attributes replaced."
withAttributes(a::SimVarAttributes; start::Union{Nothing, Exp} = a.start, fixed::Union{Nothing, Exp} = a.fixed,
               nominal::Union{Nothing, Exp} = a.nominal, min::Union{Nothing, Exp} = a.min, max::Union{Nothing, Exp} = a.max,
               unit::Union{Nothing, Exp} = a.unit, displayUnit::Union{Nothing, Exp} = a.displayUnit,
               quantity::Union{Nothing, Exp} = a.quantity, stateSelect::Union{Nothing, Symbol} = a.stateSelect,
               isProtected::Bool = a.isProtected)::SimVarAttributes =
  SimVarAttributes(start, fixed, nominal, min, max, unit, displayUnit, quantity, stateSelect, isProtected)

"An attribute's value negated, the literals folded; `nothing` for none."
negatedValue(::Nothing)::Nothing = nothing
negatedValue(e::RCONST)::Exp = RCONST(-e.value)
negatedValue(e::ICONST)::Exp = ICONST(-e.value)
negatedValue(e::Exp)::Exp = e isa UNARY && e.op === OP_UMINUS ? e.exp : UNARY(OP_UMINUS, e)

"A Boolean attribute's value negated; `nothing` for none."
notValue(::Nothing)::Nothing = nothing
notValue(e::BCONST)::Exp = BCONST(!e.value)
notValue(e::Exp)::Exp = LUNARY(OP_NOT, e)

isTrueLiteral(e::Union{Nothing, Exp})::Bool = e isa BCONST && e.value

"An attribute's value `a`, or `b` for none."
orElse(a::Union{Nothing, Exp, Symbol}, b::Union{Nothing, Exp, Symbol}) = a === nothing ? b : a

"""
    mergeAliasAttributes(rep, repType, eliminated, negated)

The attributes `rep` of a representative (of element type `repType`) with those it lacks from
the `eliminated` alias (OpenModelica's mergeAliasVars), negated (`start`, `nominal`; `min` and
`max` swapped) for a negated alias. `start` and `fixed` go together: a fixed start of the alias
replaces a free one of the representative. Of an Integer, Boolean or enumeration alias:
`start`, `fixed`, `min`, `max` (`not start` of a negated Boolean; a negated enumeration gives
none); a representative without attributes of its own takes the alias's.
"""
function mergeAliasAttributes(rep::SimVarAttributes, repType::SType, eliminated::SIMVAR, negated::Bool)::SimVarAttributes
  local e = eliminated.attributes
  attributesGiven(eliminated.elementType, e) || return rep
  local t = eliminated.elementType
  if t isa TYPE_REAL
    local (start, nominal) = negated ? (negatedValue(e.start), negatedValue(e.nominal)) : (e.start, e.nominal)
    local (lo, hi) = negated ? (negatedValue(e.max), negatedValue(e.min)) : (e.min, e.max)
    local fixedStart = isTrueLiteral(e.fixed) && !isTrueLiteral(rep.fixed)
    return SimVarAttributes(fixedStart ? start : orElse(rep.start, start), fixedStart ? e.fixed : orElse(rep.fixed, e.fixed),
                            orElse(rep.nominal, nominal), orElse(rep.min, lo), orElse(rep.max, hi),
                            orElse(rep.unit, e.unit), orElse(rep.displayUnit, e.displayUnit),
                            orElse(rep.quantity, e.quantity), orElse(rep.stateSelect, e.stateSelect),
                            rep.isProtected || e.isProtected)
  end
  t isa Union{TYPE_INTEGER, TYPE_BOOL, TYPE_ENUM} || return rep
  (negated && t isa TYPE_ENUM) && return rep
  local hasRep = attributesGiven(repType, rep)
  local base = hasRep ? rep : e
  local start = !negated ? e.start : t isa TYPE_BOOL ? notValue(e.start) : negatedValue(e.start)
  local fixedStart = isTrueLiteral(e.fixed) && !(hasRep && isTrueLiteral(base.fixed))
  local newStart = fixedStart || !hasRep ? start : orElse(base.start, start)
  local newFixed = fixedStart || !hasRep ? e.fixed : orElse(base.fixed, e.fixed)
  t isa TYPE_BOOL && return withAttributes(base; start = newStart, fixed = newFixed)
  local (lo, hi) = negated ? (negatedValue(e.max), negatedValue(e.min)) : (e.min, e.max)
  return withAttributes(base; start = newStart, fixed = newFixed, min = hasRep ? orElse(base.min, lo) : lo,
                        max = hasRep ? orElse(base.max, hi) : hi)
end

#= ==================================================================================== =#
#= Unknowns equal to one expression                                                    =#
#= ==================================================================================== =#

"`e` without one unary minus, and whether it had one."
withoutMinus(e::Exp)::Tuple{Exp, Bool} = e isa UNARY && e.op === OP_UMINUS ? (e.exp, true) : (e, false)

"""
    definitionOf(e, ht)

The unknown `v` and the expression `x` of a residual `v - x` (`x - v`; either side under one
unary minus): `v`'s reference, `x` (without its minus), and whether `v = -x`. `nothing` for
another residual, one of two variables (an alias) or of none.
"""
function definitionOf(e::Exp, ht::HT)::Union{Nothing, Tuple{EXP_CREF, Exp, Bool}}
  (e isa BINARY && e.op === OP_SUB) || return nothing
  local ((x1, neg1), (x2, neg2)) = (withoutMinus(e.exp1), withoutMinus(e.exp2))
  (x1 isa EXP_CREF) == (x2 isa EXP_CREF) && return nothing
  local (v, x, negV, negX) = x1 isa EXP_CREF ? (x1, x2, neg1, neg2) : (x2, x1, neg2, neg1)
  local entry = get(ht, crefName(v.cref), nothing)
  (entry === nothing || !isUnknownVarKind(last(entry).varKind) || aliasClass(v.ty) === :other) && return nothing
  return (v, x, xor(negV, negX))
end

"Whether `v` is declared `stateSelect = StateSelect.always`: it must stay a state, with its own start."
isStateSelectAlways(v::SIMVAR)::Bool = v.attributes.stateSelect === :always

"""
    eliminateRHSEquivalentEquations(simCode)

The unknowns that residuals set to one expression (`x - der(z)`, `y - der(z)`; up to sign)
replaced by one of them (by its category, more for an irreducible one, most for one of
`stateSelect = always`), their residuals removed. A variable is eliminated once (a later
group keeps its residual, a constraint then) and never if it represents another group; not a
shared variable, a Complex part (`_re`, `_im`), one of `stateSelect = always`, or an
irreducible one unless both are states. The representative takes the eliminated variables'
attributes it lacks. Nothing for a model with modes or translated again at its structural
changes.
"""
function eliminateRHSEquivalentEquations(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  local residuals = simCode.residualEquations
  local irreducible = Set{String}(simCode.irreducibleVariables)
  local shared = Set{String}(simCode.sharedVariables)
  local groups = OrderedDict{Exp, Vector{Tuple{EXP_CREF, Int, Bool}}}()
  for (i, eq) in enumerate(residuals)
    local d = definitionOf(eq.exp, ht)
    d === nothing && continue
    local (v, x, negated) = d
    push!(get!(groups, x, Tuple{EXP_CREF, Int, Bool}[]), (v, i, negated))
  end
  local replacements = Replacements()
  local entries = AliasEntry[]
  local removed = Set{Int}()
  local (eliminated, eliminatedEquations) = (String[], RESIDUAL_EQUATION[])
  local claimed = Set{String}()
  local representatives = Set{String}()
  local mergedAttributes = OrderedDict{String, SimVarAttributes}()
  for members in values(groups)
    length(members) >= 2 || continue
    local (best, bestPriority) = (0, -1)
    for (j, (v, _, _)) in enumerate(members)
      local name = crefName(v.cref)
      name in claimed && continue
      local sv = last(ht[name])
      local priority = categoryPriority(sv.varKind) + (name in irreducible ? 60 : 0) + (isStateSelectAlways(sv) ? 200 : 0)
      priority > bestPriority && ((best, bestPriority) = (j, priority))
    end
    best == 0 && continue
    local (repRef, _, repNegated) = members[best]
    local repName = crefName(repRef.cref)
    local representative = last(ht[repName])
    push!(representatives, repName)
    for (j, (v, i, negated)) in enumerate(members)
      local name = crefName(v.cref)
      (j == best || name == repName || name in shared || name in claimed || name in representatives) && continue
      (endswith(name, "_re") || endswith(name, "_im")) && continue
      local sv = last(ht[name])
      isStateSelectAlways(sv) && continue
      (name in irreducible && !(representative.varKind isa STATE && sv.varKind isa STATE)) && continue
      local n = xor(negated, repNegated)
      replacements[name] = Replacement(repRef, n, false)
      mergedAttributes[repName] = mergeAliasAttributes(get(mergedAttributes, repName, representative.attributes),
                                                       representative.elementType, sv, n)
      push!(entries, AliasEntry(name, repName, n))
      push!(removed, i)
      push!(eliminated, name)
      push!(eliminatedEquations, residuals[i])
      push!(claimed, name)
    end
  end
  isempty(replacements) && return simCode
  @debug "[SIMCODE: $(simCode.name): eliminateRHSEquivalentEquations] $(length(replacements)) variables aliased by equal right-hand sides"
  local newHT = copy(ht)
  foreach(e -> deleteEliminated!(newHT, ht, e.eliminatedName, e.representativeName), entries)
  for (name, attributes) in mergedAttributes
    haskey(newHT, name) || continue
    local (index, v) = newHT[name]
    attributes == v.attributes || (newHT[name] = (index, SIMVAR(v.name, v.index, v.varKind, attributes, v.elementType)))
  end
  local eliminatedSet = Set{String}(eliminated)
  return update(simCode;
                residualEquations = RESIDUAL_EQUATION[substitute(eq, replacements) for (i, eq) in enumerate(residuals) if !(i in removed)],
                initialEquations = Equation[substituteInitial(eq, replacements) for eq in simCode.initialEquations],
                initialAlgorithms = INITIAL_ALGORITHM[substitute(a, replacements) for a in simCode.initialAlgorithms],
                ifEquations = IF_EQUATION[substitute(ifEq, replacements) for ifEq in simCode.ifEquations],
                whenEquations = WHEN_EQUATION[WHEN_EQUATION(substitute(w.whenEquation, replacements), w.source, w.attr)
                                              for w in simCode.whenEquations],
                asserts = ASSERT_EQUATION[substitute(a, replacements) for a in simCode.asserts],
                stringToSimVarHT = newHT,
                irreducibleVariables = filter(n -> !(n in eliminatedSet), simCode.irreducibleVariables),
                aliasMap = withAliases(simCode.aliasMap, entries),
                eliminatedVariables = vcat(simCode.eliminatedVariables, eliminated),
                eliminatedEquations = vcat(RESIDUAL_EQUATION[substitute(eq, replacements) for eq in simCode.eliminatedEquations],
                                           eliminatedEquations))
end
