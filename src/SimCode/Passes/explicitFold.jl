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

#= Explicitly assigned algebraic variables folded into their uses; reciprocal equations oriented. =#

"Whether `v` is a Real variable of `fixed = true` (its start is an initial equation)."
hasFixedStart(v::SIMVAR)::Bool = v.elementType isa TYPE_REAL && isTrueLiteral(v.attributes.fixed)

"""
    explicitDefinition(e, ht)

The variable `v` of the table and the expression `x` of a residual `v - x` (`x - v`), exactly
one side a variable reference; `nothing` for another residual.
"""
function explicitDefinition(e::Exp, ht::HT)::Union{Nothing, Tuple{String, Exp}}
  (e isa BINARY && e.op === OP_SUB) || return nothing
  (e.exp1 isa EXP_CREF) == (e.exp2 isa EXP_CREF) && return nothing
  local (v, x) = e.exp1 isa EXP_CREF ? (e.exp1, e.exp2) : (e.exp2, e.exp1)
  local name = crefName(v.cref)
  return haskey(ht, name) ? (name, x) : nothing
end

"""
    foldExplicitSingleAssign(simCode)

Every algebraic variable a single residual `v - x` defines (`x` without a derivative and
without `v`) replaced by `x` everywhere, its residual removed and recorded for the observed
equations; to a fixed point (at most 8 rounds). Not a scalarized array element (the code
generators rebuild the array from its elements), an alias's, an irreducible, shared or event
variable (`eventAndComplexNames`), one of a fixed start (its initial equation), one on a cycle of the
round's definitions (cyclicNames: InitialEquationTests IEQ54, `u = 2y + 0.1x` and `y = noEvent(if u > 1
...)`, folded together, were observed equations without an order, ModelingToolkit's "The equations have
at least one cycle"; they stay residuals, a loop). A round in which a folded name is still read somewhere
is undone; one that would remove every residual is not done. Nothing for a model with modes or translated
again at its structural changes.
"""
function foldExplicitSingleAssign(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  isempty(simCode.residualEquations) && return simCode
  local kept = eventAndComplexNames(simCode)
  union!(kept, simCode.irreducibleVariables, simCode.sharedVariables)
  for _ in 1:8
    local next = foldExplicitOnce(simCode, kept)
    next === nothing && break
    simCode = next
  end
  return simCode
end

"One round of `foldExplicitSingleAssign`; `nothing` if nothing folds."
function foldExplicitOnce(simCode::SIM_CODE, kept::OrderedSet{String})::Union{Nothing, SIM_CODE}
  local ht = simCode.stringToSimVarHT
  local residuals = simCode.residualEquations
  local aliases = Set{String}(Iterators.flatten((e.eliminatedName, e.representativeName) for e in simCode.aliasMap))
  local definitions = OrderedDict{String, Vector{Tuple{Int, Exp}}}()
  for (i, eq) in enumerate(residuals)
    local d = explicitDefinition(eq.exp, ht)
    d === nothing && continue
    local (name, x) = d
    (occursin('[', name) || occursin(']', name) || name in kept || name in aliases) && continue
    local v = last(ht[name])
    (v.varKind isa ALG_VARIABLE && !hasFixedStart(v)) || continue
    anyExp(y -> y isa DER || (y isa CALL && y.name === :der), x) && continue
    name in crefNames!(OrderedSet{String}(), x) && continue
    push!(get!(definitions, name, Tuple{Int, Exp}[]), (i, x))
  end
  local folded = OrderedDict{String, Tuple{Int, Exp}}(n => only(ds) for (n, ds) in definitions if length(ds) == 1)
  local cyclic = cyclicNames(OrderedDict{String, OrderedSet{String}}(n => crefNames!(OrderedSet{String}(), x) for (n, (_, x)) in folded))
  foreach(n -> delete!(folded, n), cyclic)
  isempty(folded) && return nothing
  length(residuals) <= length(folded) && return nothing
  local rs = Replacements(n => Replacement(x, false, false) for (n, (_, x)) in folded)
  local removed = Set{Int}(i for (i, _) in values(folded))
  local newResiduals = RESIDUAL_EQUATION[RESIDUAL_EQUATION(foldNumeric(substitute(eq.exp, rs)), eq.source, eq.attr)
                                         for (i, eq) in enumerate(residuals) if !(i in removed)]
  local initialEquations = Equation[frozenInitial(eq, rs) for eq in simCode.initialEquations]
  local ifEquations = IF_EQUATION[substitute(ifEq, rs) for ifEq in simCode.ifEquations]
  local eliminatedEquations = RESIDUAL_EQUATION[substitute(eq, rs) for eq in simCode.eliminatedEquations]
  local asserts = ASSERT_EQUATION[substitute(a, rs) for a in simCode.asserts]
  local initialAlgorithms = INITIAL_ALGORITHM[substitute(a, rs) for a in simCode.initialAlgorithms]
  #= a folded name still read (by a chain of folds of this round, a when-clause, a binding): the round is undone =#
  local read = OrderedSet{String}()
  foreach(eq -> crefNames!(read, eq.exp), newResiduals)
  for eq in initialEquations, e in equationExps(eq)
    eq isa ARRAY_EQUATION || crefNames!(read, e)
  end
  ifEquationNames!(read, ifEquations)
  foreach(eq -> crefNames!(read, eq.exp), eliminatedEquations)
  foreach(w -> whenCrefNames!(read, w.whenEquation), simCode.whenEquations)
  assertNames!(read, asserts)
  foreach(a -> statementCrefNames!(read, a.statements), initialAlgorithms)
  for (_, (_, v)) in ht
    local b = bindingOf(v.varKind)
    (v.varKind isa Union{PARAMETER, ARRAY_PARAMETER, DATA_STRUCTURE} && b !== nothing) && crefNames!(read, b)
  end
  any(n -> n in read, keys(folded)) && return nothing
  local newHT = copy(ht)
  foreach(n -> delete!(newHT, n), keys(folded))
  local names = sort(collect(keys(folded)))
  @debug "[SIMCODE: $(simCode.name): foldExplicitSingleAssign] folded $(length(names)) explicit assignments"
  return update(simCode; residualEquations = newResiduals, initialEquations, ifEquations, asserts, initialAlgorithms,
                stringToSimVarHT = newHT, irreducibleVariables = filter(n -> !haskey(folded, n), simCode.irreducibleVariables),
                eliminatedVariables = vcat(simCode.eliminatedVariables, names),
                eliminatedEquations = vcat(eliminatedEquations, RESIDUAL_EQUATION[residuals[first(folded[n])] for n in names]))
end

"""
    orientReciprocalEquations(simCode)

`x = 1/y`, `x` and `y` algebraic unknowns and `y` in no other equation, as `y = 1/x`: `y` is
the variable that equation determines, now explicitly (omc solves `1/y = x` for `y`). As
`x = 1/y`, foldExplicitSingleAssign put `1/y` into `x`'s other equations, which then determined
`y` nonlinearly, and the initialization's Newton for `y` diverged from its start (MSL
FluxTubes' LeakageWithCoefficient: `R_m = 1/G_m`). Not a `y` an event, an initial or an
eliminated equation reads. Nothing for a model with modes or translated again at its
structural changes.
"""
function orientReciprocalEquations(simCode::SIM_CODE)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  isempty(simCode.residualEquations) && return simCode
  local ht = simCode.stringToSimVarHT
  local excluded = eventAndComplexNames(simCode)
  for eq in simCode.initialEquations
    eq isa ARRAY_EQUATION && continue
    foreach(e -> crefNames!(excluded, e), equationExps(eq))
  end
  foreach(eq -> crefNames!(excluded, eq.exp), simCode.eliminatedEquations)
  local uses = Dict{String, Int}()
  for eq in simCode.residualEquations, n in crefNames!(OrderedSet{String}(), eq.exp)
    uses[n] = get(uses, n, 0) + 1
  end
  local oriented = 0
  local residuals = RESIDUAL_EQUATION[]
  for eq in simCode.residualEquations
    local e = reciprocalOrientation(eq.exp, ht, uses, excluded)
    e === nothing ? push!(residuals, eq) : (push!(residuals, RESIDUAL_EQUATION(e, eq.source, eq.attr)); oriented += 1)
  end
  oriented == 0 && return simCode
  @debug "[SIMCODE: $(simCode.name): orientReciprocalEquations] $(oriented) equations as y = 1/x"
  return update(simCode; residualEquations = residuals)
end

"The variable reference `y` of `1/y`; `nothing` for another expression."
function reciprocalOf(e::Exp)::Union{Nothing, EXP_CREF}
  (e isa BINARY && e.op === OP_DIV && isOneLiteral(e.exp1)) || return nothing
  return e.exp2 isa EXP_CREF ? e.exp2 : nothing
end

"`x - 1/y` (`1/y - x`) as `y - 1/x`, both scalar algebraic unknowns without a fixed start, `y` in this equation only and in no excluded one; `nothing` for another residual."
function reciprocalOrientation(e::Exp, ht::HT, uses::Dict{String, Int}, excluded::OrderedSet{String})::Union{Nothing, Exp}
  (e isa BINARY && e.op === OP_SUB) || return nothing
  local (y, x) = (reciprocalOf(e.exp2), e.exp1)
  y === nothing && ((y, x) = (reciprocalOf(e.exp1), e.exp2))
  (y === nothing || !(x isa EXP_CREF)) && return nothing
  local (xn, yn) = (crefName(x.cref), crefName(y.cref))
  xn == yn && return nothing
  for n in (xn, yn)
    (haskey(ht, n) && !occursin('[', n)) || return nothing
    local v = last(ht[n])
    (v.varKind isa ALG_VARIABLE && !hasFixedStart(v)) || return nothing
  end
  (get(uses, yn, 0) == 1 && !(yn in excluded)) || return nothing
  return BINARY(y, OP_SUB, BINARY(RCONST(1.0), OP_DIV, x))
end
