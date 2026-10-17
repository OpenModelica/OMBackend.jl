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

#= Elimination of the variables that only feed outputs (eliminateNonDynamic). =#

"The elements of the array variables by their names without subscripts (`x` => `x[1]`, `x[2]`)."
function elementsByBase(ht::HT)::Dict{String, Vector{String}}
  local index = Dict{String, Vector{String}}()
  for name in keys(ht)
    local i = findfirst('[', name)
    i === nothing || push!(get!(index, name[1:prevind(name, i)], String[]), name)
  end
  return index
end

"""
    equationVariableNames(e, ht, elements)

The variables of the table a residual reads: a name of the table; for another name, its
array's name (one with subscripts), and every element of the array it names (a read of the
whole array, or by a subscript that is no literal).
"""
function equationVariableNames(e::Exp, ht::HT, elements::Dict{String, Vector{String}})::OrderedSet{String}
  local names = OrderedSet{String}()
  for name in crefNames!(OrderedSet{String}(), e)
    if haskey(ht, name)
      push!(names, name)
      continue
    end
    local i = findfirst('[', name)
    local base = i === nothing ? name : name[1:prevind(name, i)]
    (base != name && haskey(ht, base)) && push!(names, base)
    union!(names, get(elements, haskey(elements, name) ? name : base, String[]))
  end
  return names
end

"""
    freshMatching(simCode)

A matching of the current residuals to the unknowns (those of the causalization are stale
after the passes): each unknown's residual (`0` for none), the unknowns by number and the
numbers by name; empty if there are more residuals than unknowns or the matching fails.
"""
function freshMatching(simCode::SIM_CODE)::Tuple{Vector{Int}, Dict{String, Int}, Dict{Int, String}}
  local ht = simCode.stringToSimVarHT
  local residuals = simCode.residualEquations
  local numberOf = Dict{String, Int}()
  local nameOf = Dict{Int, String}()
  for (name, (_, v)) in ht
    isUnknownVarKind(v.varKind) || continue
    numberOf[name] = length(numberOf) + 1
    nameOf[numberOf[name]] = name
  end
  local n = length(numberOf)
  length(residuals) > n && return (Int[], numberOf, nameOf)
  local elements = elementsByBase(ht)
  local rows = OrderedDict{Int, Vector{Int}}()
  for (i, eq) in enumerate(residuals)
    rows[i] = sort(unique(Int[numberOf[r] for r in equationVariableNames(eq.exp, ht, elements) if haskey(numberOf, r)]))
  end
  #= the matching takes a square system: empty rows for the variables over the equations =#
  for i in (length(residuals) + 1):n
    rows[i] = Int[]
  end
  local matchOrder = try
    last(GraphAlgorithms.matching(rows, n))
  catch err
    #= a large system can overflow the recursive matching's stack: no elimination =#
    err isa StackOverflowError || _fallback(err, :outputOnlyMatching)
    return (Int[], numberOf, nameOf)
  end
  return (matchOrder, numberOf, nameOf)
end

"""
    essentialEquations(simCode, matchOrder, nameOf, reads)

The residuals the dynamics need: from those of the states, their derivatives, the discrete,
irreducible variables, the aliases' representatives, what the asserts and the if-equations
read, and the unmatched residuals that read an unknown, backward along what they read to the
residuals that solve it (an array's elements' for its name).
"""
function essentialEquations(simCode::SIM_CODE, matchOrder::Vector{Int}, nameOf::Dict{Int, String},
                            reads::Vector{OrderedSet{String}})::OrderedSet{Int}
  local ht = simCode.stringToSimVarHT
  local solvedBy = Dict{String, Int}()
  local solvedByBase = Dict{String, Vector{Int}}()
  for (k, i) in enumerate(matchOrder)
    (i > 0 && haskey(nameOf, k)) || continue
    local name = nameOf[k]
    solvedBy[name] = i
    local b = findfirst('[', name)
    b === nothing || push!(get!(solvedByBase, name[1:prevind(name, b)], Int[]), i)
  end
  local seeds = OrderedSet{Int}()
  local seed! = n -> haskey(solvedBy, n) && push!(seeds, solvedBy[n])
  for (name, (_, v)) in ht
    v.varKind isa Union{STATE, STATE_DERIVATIVE, DISCRETE} && seed!(name)
  end
  foreach(seed!, simCode.irreducibleVariables)
  foreach(e -> seed!(e.representativeName), simCode.aliasMap)
  foreach(seed!, assertNames!(OrderedSet{String}(), simCode.asserts))
  foreach(seed!, ifEquationNames!(OrderedSet{String}(), simCode.ifEquations))
  local matched = Set{Int}(i for i in matchOrder if i > 0)
  for i in eachindex(reads)
    i in matched && continue
    any(n -> haskey(ht, n) && isUnknownVarKind(last(ht[n]).varKind), reads[i]) && push!(seeds, i)
  end
  local essential = OrderedSet{Int}()
  local queue = collect(seeds)
  while !isempty(queue)
    local i = popfirst!(queue)
    i in essential && continue
    push!(essential, i)
    1 <= i <= length(reads) || continue
    for n in reads[i]
      haskey(solvedBy, n) && !(solvedBy[n] in essential) && push!(queue, solvedBy[n])
      for j in get(solvedByBase, n, Int[])
        j in essential || push!(queue, j)
      end
    end
  end
  return essential
end

#= ==================================================================================== =#
#= Linearity in a variable                                                              =#
#= ==================================================================================== =#

"The name of a variable reference, or of an element by literal subscripts; `nothing` for another expression."
scalarReferenceName(e::EXP_CREF)::Union{Nothing, String} = crefName(e.cref)
function scalarReferenceName(e::ASUB)::Union{Nothing, String}
  (e.exp isa EXP_CREF && all(s -> s isa ICONST, e.subs)) || return nothing
  return string(crefName(e.exp.cref), join(("[$(s.value)]" for s in e.subs)))
end
scalarReferenceName(::Exp)::Union{Nothing, String} = nothing

"""
    occursLinearly(e, name)

Whether the variable `name` appears in `e`, and if so only affinely (never under a nonlinear
operator or in a function's argument): `(occurs, linear)`. Conservative: what is not known to
be linear is not.
"""
function occursLinearly(e::Exp, name::String)::Tuple{Bool, Bool}
  local n = scalarReferenceName(e)
  n === nothing || return (n == name, true)
  e isa Union{ICONST, RCONST, BCONST, SCONST, ENUM_LITERAL, WILD} && return (false, true)
  e isa Union{UNARY, CAST} && return occursLinearly(e.exp, name)
  if e isa BINARY
    local ((o1, l1), (o2, l2)) = (occursLinearly(e.exp1, name), occursLinearly(e.exp2, name))
    e.op in (OP_ADD, OP_SUB) && return (o1 || o2, l1 && l2)
    e.op in (OP_MUL, OP_SCALAR_PRODUCT) && return (o1 || o2, l1 && l2 && !(o1 && o2))
    e.op === OP_DIV && return (o1 || o2, l1 && !o2)
    if e.op === OP_POW
      o2 && return (true, false)
      o1 || return (false, true)
      return (true, l1 && isOneLiteral(e.exp2))
    end
    return (o1 || o2, !(o1 || o2))
  elseif e isa IFEXP
    first(occursLinearly(e.cond, name)) && return (true, false)
    local ((ot, lt), (oe, le)) = (occursLinearly(e.thenExp, name), occursLinearly(e.elseExp, name))
    return (ot || oe, lt && le)
  end
  local occurs = name in crefNames!(OrderedSet{String}(), e)
  return (occurs, !occurs)
end

"Whether `name` appears in the residual `e`, affinely: the observed equation solves it linearly (Symbolics.solve_for)."
isLinearlySolvableFor(e::Exp, name::String)::Bool = occursLinearly(e, name) == (true, true)

#= ==================================================================================== =#
#= The pass                                                                             =#
#= ==================================================================================== =#

"""
    eliminateOutputOnlyVariables(simCode, options)

The algebraic (and array) variables the dynamics do not need (`essentialEquations`, on a
fresh matching) removed with the residuals that define them, if those solve them linearly;
recorded for the observed equations. `options`: with `reachability` every such variable, else
only those a pattern of `patterns` matches; none a pattern of `keepPatterns` matches. A variable that a remaining residual (or one reading its
array's name), a when-clause, an initial equation reads, a Complex part, one of a fixed start,
is kept with its residual, to a fixed point (what a kept residual reads stays too). Never every
residual. Nothing for a model with modes or translated again at its structural changes.
"""
function eliminateOutputOnlyVariables(simCode::SIM_CODE, options::EliminationOptions)::SIM_CODE
  (hasSubModels(simCode) || hasFrontendModel(simCode)) && return simCode
  local (matchOrder, numberOf, nameOf) = freshMatching(simCode)
  isempty(matchOrder) && return simCode
  local ht = simCode.stringToSimVarHT
  local residuals = simCode.residualEquations
  local elements = elementsByBase(ht)
  local reads = OrderedSet{String}[equationVariableNames(eq.exp, ht, elements) for eq in residuals]
  local essential = essentialEquations(simCode, matchOrder, nameOf, reads)
  length(essential) == length(residuals) && return simCode
  local variableOf = Dict{Int, String}(i => nameOf[k] for (k, i) in enumerate(matchOrder) if i > 0 && haskey(nameOf, k))
  local removed = OrderedSet{Int}()
  local pairs = Tuple{String, Int}[]
  for i in 1:length(residuals)
    (i in essential || !haskey(variableOf, i)) && continue
    local name = variableOf[i]
    (haskey(ht, name) && last(ht[name]).varKind isa Union{ALG_VARIABLE, ARRAY}) || continue
    (options.reachability || any(p -> occursin(p, name), options.patterns)) || continue
    any(p -> occursin(p, name), options.keepPatterns) && continue
    isLinearlySolvableFor(residuals[i].exp, name) || continue
    push!(removed, i)
    push!(pairs, (name, i))
  end
  (isempty(removed) || length(removed) >= length(residuals)) && return simCode
  local readBy = Dict{String, OrderedSet{Int}}()
  for (i, names) in enumerate(reads), n in names
    push!(get!(readBy, n, OrderedSet{Int}()), i)
  end
  local whenReads = OrderedSet{String}()
  foreach(w -> whenCrefNames!(whenReads, w.whenEquation), simCode.whenEquations)
  local complexParts = OrderedSet{String}()
  complexFieldNames!(complexParts, residuals, ht)
  complexFieldNames!(complexParts, simCode.initialEquations, ht)
  for ifEq in simCode.ifEquations, b in ifEq.branches
    complexFieldNames!(complexParts, b.residualEquations, ht)
  end
  complexFieldNames!(complexParts, simCode.eliminatedEquations, ht)
  local initialReads = OrderedSet{String}()
  for eq in simCode.initialEquations
    eq isa ARRAY_EQUATION && continue
    foreach(e -> crefNames!(initialReads, e), equationExps(eq))
  end
  local stillRead = name::String -> begin
    local b = findfirst('[', name)
    local base = b === nothing ? name : name[1:prevind(name, b)]
    any(n -> any(i -> !(i in removed), get(readBy, n, OrderedSet{Int}())), base == name ? (name,) : (name, base)) ||
      name in whenReads || name in initialReads || name in complexParts
  end
  local candidates = OrderedSet{String}(first(p) for p in pairs)
  local kept = OrderedSet{String}()
  local more = true
  while more
    more = false
    for name in candidates
      name in kept && continue
      (stillRead(name) || (haskey(ht, name) && hasFixedStart(last(ht[name])))) || continue
      push!(kept, name)
      more = true
      if haskey(numberOf, name)
        local i = matchOrder[numberOf[name]]
        i > 0 && delete!(removed, i)
      end
    end
  end
  local eliminated = Tuple{String, Int}[p for p in pairs if !(first(p) in kept)]
  local newHT = copy(ht)
  foreach(n -> n in kept || delete!(newHT, n), candidates)
  @debug "[SIMCODE: $(simCode.name): eliminateNonDynamic] eliminated $(length(eliminated)) variables (kept $(length(kept)))"
  return update(simCode; stringToSimVarHT = newHT,
                residualEquations = RESIDUAL_EQUATION[eq for (i, eq) in enumerate(residuals) if !(i in removed)],
                eliminatedVariables = vcat(simCode.eliminatedVariables, String[first(p) for p in eliminated]),
                eliminatedEquations = vcat(simCode.eliminatedEquations, RESIDUAL_EQUATION[residuals[last(p)] for p in eliminated]))
end
