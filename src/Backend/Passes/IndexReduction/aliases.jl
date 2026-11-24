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
#= Aliases read through one representative before index reduction (the alias equations kept). =#

"""
    AliasSign(name, negated)

A variable of an alias set relative to the set's root: `name = ±root` (`negated`: `name = -root`).
"""
struct AliasSign
  name::Symbol
  negated::Bool
end

"""
    AliasSets()

Union-find of the alias equations with signs (`a = b`, `a = -b`): each variable's parent and its
sign relative to it; the constant a set equals (`a = c`, `c` a literal or a parameter), the
equations that joined it.
"""
struct AliasSets
  parent::Dict{Symbol, Symbol}
  negated::Dict{Symbol, Bool}
  constant::Dict{Symbol, IR.Expression}
  conflicting::Set{Symbol}
  equations::Dict{Symbol, Vector{IR.Equation}}
end

AliasSets() = AliasSets(Dict{Symbol, Symbol}(), Dict{Symbol, Bool}(), Dict{Symbol, IR.Expression}(), Set{Symbol}(),
                        Dict{Symbol, Vector{IR.Equation}}())

"The root of `x`'s set and whether `x = -root` (paths compressed)."
function aliasRoot!(sets::AliasSets, x::Symbol)::Tuple{Symbol, Bool}
  haskey(sets.parent, x) || (sets.parent[x] = x; sets.negated[x] = false; sets.equations[x] = IR.Equation[])
  local p = sets.parent[x]
  p == x && return (x, false)
  local (root, n) = aliasRoot!(sets, p)
  sets.parent[x] = root
  sets.negated[x] = xor(sets.negated[x], n)
  return (root, sets.negated[x])
end

"""
    joinAliases!(sets, a, b, negated, eq)

The sets of `a` and `b` joined by `eq`, `a = b` (`a = -b`: `negated`); a set that joins itself
with the other sign (`a = b` and `a = -b`: both zero) or two constants becomes `conflicting`.
"""
function joinAliases!(sets::AliasSets, a::Symbol, b::Symbol, negated::Bool, eq::IR.Equation)::Nothing
  local (ra, na) = aliasRoot!(sets, a)
  local (rb, nb) = aliasRoot!(sets, b)
  if ra == rb
    xor(na, nb) == negated || push!(sets.conflicting, ra)
    push!(sets.equations[ra], eq)
    return nothing
  end
  #= rb under ra: b = ±a, so rb = (nb) b = ... =#
  sets.parent[rb] = ra
  sets.negated[rb] = xor(xor(na, nb), negated)
  append!(sets.equations[ra], sets.equations[rb])
  push!(sets.equations[ra], eq)
  rb in sets.conflicting && push!(sets.conflicting, ra)
  if haskey(sets.constant, rb)
    local c = sets.negated[rb] ? negate(sets.constant[rb]) : sets.constant[rb]
    haskey(sets.constant, ra) ? push!(sets.conflicting, ra) : (sets.constant[ra] = c)
  end
  return nothing
end

"`a = c`: the set of `a` equal to the constant `c` (a second constant makes it `conflicting`)."
function constantAlias!(sets::AliasSets, a::Symbol, c::IR.Expression, eq::IR.Equation)::Nothing
  local (ra, na) = aliasRoot!(sets, a)
  push!(sets.equations[ra], eq)
  haskey(sets.constant, ra) ? push!(sets.conflicting, ra) : (sets.constant[ra] = na ? negate(c) : c)
  return nothing
end

"A reference under unary minuses: its name and whether their number is odd; `nothing` for another expression."
function signedName(e::IR.Expression)::Union{Nothing, Tuple{Symbol, Bool}}
  local negated = false
  while e isa IR.UNARY && e.op === IR.UMINUS
    negated = !negated
    e = e.exp
  end
  return e isa IR.CREF ? (e.cref.name, negated) : nothing
end

"""
    aliasEquation(eq, isCandidate, isConstant) -> Union{Nothing, Tuple}

An alias equation of continuous variables (`isCandidate`): `(a, b, negated)` for `±a = ±b`,
`±a ± b = 0`; `(a, c)` for `±a = c` with `c` a constant (`isConstant`: a literal, a parameter, a
negation of one). `nothing` for another equation.
"""
function aliasEquation(eq::IR.Equation, isCandidate::F, isConstant::G)::Union{Nothing, Tuple} where {F, G}
  (eq isa IR.SCALAR_EQUATION && eq.ty isa IR.T_REAL) || return nothing
  local (l, r) = (signedName(eq.lhs), signedName(eq.rhs))
  if l !== nothing && r !== nothing
    (isCandidate(l[1]) && isCandidate(r[1]) && l[1] != r[1]) || return nothing
    return (l[1], r[1], xor(l[2], r[2]))
  end
  for (side, other) in ((eq.lhs, eq.rhs), (eq.rhs, eq.lhs))
    local s = signedName(side)
    if s !== nothing && isCandidate(s[1]) && isConstant(other)
      return (s[1], s[2] ? negate(other) : other)
    end
    #= ±a ± b = 0 =#
    if isZero(other) && side isa IR.BINARY && side.op in (IR.ADD, IR.SUB)
      local (x, y) = (signedName(side.exp1), signedName(side.exp2))
      (x === nothing || y === nothing || !isCandidate(x[1]) || !isCandidate(y[1]) || x[1] == y[1]) && continue
      #= x ± y = 0: x = ∓y =#
      return (x[1], y[1], !xor(xor(x[2], y[2]), side.op === IR.SUB))
    end
  end
  return nothing
end

"""
    AliasUndo(equations, initialEquations, kinds, removed, substituted, attributes)

What `substituteAliases!` changed, to restore it (`undoAliases!`): the partition's equation
lists before, the variables whose kind it changed as they were, the derivative variables it
removed, the number of names substituted, and the attributes it merged (`mergeAliasStarts!`) as
they were.
"""
struct AliasUndo
  equations::Vector{IR.Equation}
  initialEquations::Vector{IR.Equation}
  kinds::Vector{Pair{Symbol, IR.VariableKind}}
  removed::Vector{IR.Variable}
  substituted::Int
  attributes::Vector{Pair{Symbol, IR.VariableAttributes}}
end

"The model as before `substituteAliases!`."
function undoAliases!(model::IR.Model, partition::IR.Partition, undo::AliasUndo)::Nothing
  partition.equations = undo.equations
  partition.initialEquations = undo.initialEquations
  foreach(v -> push!(model.variables, v), undo.removed)
  for (name, kind) in undo.kinds
    model.variables[name].kind = kind
  end
  for (name, attributes) in undo.attributes
    model.variables[name].attributes = attributes
  end
  return nothing
end

"""
    fixedStart(v) -> Bool

Whether `v` declares a fixed start: `fixed` given and not the literal `false` (an array's `each fixed = true`, an
expression).
"""
fixedStart(v::IR.Variable)::Bool = v.attributes.fixed !== nothing && !(v.attributes.fixed isa IR.BOOLEAN && !v.attributes.fixed.value)

"`-e` of a start value: an array literal element by element (its type kept), a scalar by `negate`."
function negatedStart(e::IR.Expression)::IR.Expression
  e isa IR.ARRAY && return IR.ARRAY(IR.Expression[negatedStart(x) for x in e.elements], e.ty)
  return negate(e)
end

"`a` with the start `start` and the fixed attribute `fixed`."
withStart(a::IR.VariableAttributes, start::Union{Nothing, IR.Expression}, fixed::Union{Nothing, IR.Expression})::IR.VariableAttributes =
  IR.VariableAttributes(start, fixed, a.nominal, a.min, a.max, a.unit, a.displayUnit, a.quantity, a.stateSelect)

"""
    mergeAliasStarts!(variables, best, members, changed) -> Nothing

The start values of an alias set's members read through its representative `best` (`members`: `AliasSign(m, n)`,
`m = -best` where `n`; by priority) merged onto `best`, as omc merges them when it removes aliases
(RemoveSimpleEquations.mergeStartFixedAttributes): a member's fixed start becomes `best`'s, the member no longer fixed
(it is computed from `best`; MSL ThreeSprings' `body1.r_0`, fixed and `avoid`, read through the state
`body1.frame_a.r_0`); else, where `best` has no start, the first member's start (not fixed). The set has at most one
fixed start (the callers leave a set of more as written). The attributes as they were go to `changed`.
"""
function mergeAliasStarts!(variables::IR.VariableSet, best::Symbol, members::Vector{AliasSign},
                           changed::Vector{Pair{Symbol, IR.VariableAttributes}})::Nothing
  local target::IR.Variable = variables[best]
  fixedStart(target) && return nothing
  local signed = (m::AliasSign, e::IR.Expression) -> m.negated ? negatedStart(e) : e
  local fixedMember::Union{Nothing, Int} = findfirst(m::AliasSign -> fixedStart(variables[m.name]), members)
  if fixedMember !== nothing
    local m::AliasSign = members[fixedMember]
    local source::IR.Variable = variables[m.name]
    source.attributes.start === nothing && return nothing
    push!(changed, best => target.attributes, m.name => source.attributes)
    target.attributes = withStart(target.attributes, signed(m, source.attributes.start), source.attributes.fixed)
    source.attributes = withStart(source.attributes, source.attributes.start, nothing)
    return nothing
  end
  target.attributes.start === nothing || return nothing
  local withValue::Union{Nothing, Int} = findfirst(m::AliasSign -> variables[m.name].attributes.start !== nothing, members)
  withValue === nothing && return nothing
  local free::AliasSign = members[withValue]
  push!(changed, best => target.attributes)
  target.attributes = withStart(target.attributes, signed(free, variables[free.name].attributes.start), target.attributes.fixed)
  return nothing
end

"The derivative chain of `name` by the states' kinds (`STATE.derivative`): `\$DER.name`, `\$DER.\$DER.name`, ... as far as they are variables of the model."
function derivativeChain(variables::IR.VariableSet, name::Symbol)::Vector{Symbol}
  local chain = Symbol[]
  local k::IR.VariableKind = variables[name].kind
  while k isa IR.STATE && k.derivative !== nothing && haskey(variables, k.derivative)
    push!(chain, k.derivative)
    k = variables[k.derivative].kind
  end
  return chain
end

"""
    antiderivatives(variables) -> Dict{Symbol, Symbol}

Each derivative variable's antiderivative, by the kinds: a state's `derivative` (a `\$DER.x` of a higher-order
chain is a state itself), a dummy state's `dummyDerivative`, a derivative variable's state (`STATE_DER`,
`ALGEBRAIC_DER`, `DUMMY_DER`).
"""
function antiderivatives(variables::IR.VariableSet)::Dict{Symbol, Symbol}
  local out = Dict{Symbol, Symbol}()
  for v in variables
    local k::IR.VariableKind = v.kind
    if k isa IR.STATE
      k.derivative === nothing || (out[k.derivative] = v.name)
    elseif k isa IR.DUMMY_STATE
      out[k.dummyDerivative] = v.name
    elseif k isa Union{IR.STATE_DER, IR.ALGEBRAIC_DER}
      out[v.name] = k.state
    elseif k isa IR.DUMMY_DER
      out[v.name] = k.dummyState
    end
  end
  return out
end

"""
    substituteAliases!(model, partition) -> AliasUndo

The continuous variables of alias sets read through one representative in the partition's
equations and initial equations, so that index reduction differentiates neither the alias
equations (a connection's `frame_a.r_0 = frame_b.r_0`) nor their derivatives: the sets of the
alias equations `a = ±b` (`±a ± b = 0`) and `a = c` (a literal or a parameter) among the
continuous variables. A set equal to a constant reads the constant, its derivatives 0. The
representative of another: a state with a fixed start, else a state, then by StateSelect, a
start value, the declaration order; the substituted members' start values merged onto it
(`mergeAliasStarts!`). The alias equations stay (each name is still computed;
SimCode's alias elimination takes them as before). A variable is left as it is where it is
written by a when-equation (`reinit`) or an algorithm, read as part of a whole record or array
(`frame_a.R` passed to a function) or by a when-equation's derivative, or is an input; a set
with two fixed starts, with the signs or constants in conflict, or with a member whose
derivatives the representative's do not match, is left too. The states substituted become
algebraic, their derivative variables removed. Returns what to restore (`undoAliases!`).
"""
function substituteAliases!(model::IR.Model, partition::IR.Partition)::AliasUndo
  local variables = model.variables
  local discretes = discreteTargets(partition)
  local isCandidate = name::Symbol -> begin
    local v = get(variables, name, nothing)
    v !== nothing && v.ty isa IR.T_REAL && v.kind isa Union{IR.STATE, IR.ALGEBRAIC, IR.ALGEBRAIC_DER} && !(name in discretes) &&
      v.direction !== IR.INPUT
  end
  local isConstant = (e::IR.Expression) -> begin
    local s = signedName(e)
    s === nothing && return isLiteral(e) || (e isa IR.UNARY && e.op === IR.UMINUS && isLiteral(e.exp))
    local v = get(variables, s[1], nothing)
    return v !== nothing && v.kind isa Union{IR.PARAMETER, IR.CONSTANT} && v.ty isa IR.T_REAL
  end
  local antiderivative::Dict{Symbol, Symbol} = antiderivatives(variables)
  #= the variables that must keep their equations (inputs are no candidates) =#
  local blocked::Set{Symbol} = aliasBlocked(variables, partition)
  #= the sets =#
  local sets = AliasSets()
  for eq in partition.equations
    local a = aliasEquation(eq, isCandidate, isConstant)
    a === nothing && continue
    length(a) == 3 ? joinAliases!(sets, a[1], a[2], a[3], eq) : constantAlias!(sets, a[1], a[2], eq)
  end
  local members = Dict{Symbol, Vector{AliasSign}}()
  for x in collect(keys(sets.parent))
    local (root, negated) = aliasRoot!(sets, x)
    push!(get!(() -> AliasSign[], members, root), AliasSign(x, negated))
  end
  local fixedState = (name::Symbol) -> begin
    local v = variables[name]
    v.kind isa IR.STATE && v.attributes.fixed isa IR.BOOLEAN && v.attributes.fixed.value
  end
  local position = Dict{Symbol, Int}(v.name => i for (i, v) in enumerate(variables))
  local priority = (name::Symbol) -> begin
    local v = variables[name]
    (name in blocked ? 10000 : 0) + (fixedState(name) ? 1000 : 0) + (v.kind isa IR.STATE ? 100 : 0) +
      10 * Int(something(v.attributes.stateSelect, IR.DEFAULT)) + (v.attributes.start === nothing ? 0 : 1)
  end
  local replacements = Dict{Symbol, IR.Expression}()
  local definitions = Base.IdSet{IR.Equation}()
  local states = Symbol[]
  #= the variable's own reference (a matrix element's subscripts as SimCode names it, `R.T[1][1]`) =#
  local ref = (name::Symbol, negated::Bool) -> begin
    local r = IR.CREF(variables[name].cref, IR.T_REAL())
    negated ? negate(r) : r
  end
  local changedAttributes::Vector{Pair{Symbol, IR.VariableAttributes}} = Pair{Symbol, IR.VariableAttributes}[]
  for (root, set) in members
    #= a set of two fixed starts (of states or not) is left as written =#
    (root in sets.conflicting || count(m -> fixedStart(variables[m.name]), set) > 1) && continue
    local constant = get(sets.constant, root, nothing)
    constant !== nothing && any(m -> fixedState(m.name) || m.name in blocked, set) && continue
    local best = constant === nothing ? argmax(m -> (priority(m.name), -position[m.name]), set) : nothing
    any(m -> fixedState(m.name) && (best === nothing || m.name != best.name), set) && continue
    local bestChain = best === nothing ? Symbol[] : derivativeChain(variables, best.name)
    local substituted = false
    local merged::Vector{AliasSign} = AliasSign[]
    for m in set
      (best !== nothing && m.name == best.name) && continue
      m.name in blocked && continue
      local chain = derivativeChain(variables, m.name)
      length(chain) <= length(bestChain) || best === nothing || continue
      #= m = ±root, best = ±root: m = (m.negated xor best.negated) best =#
      if best === nothing
        replacements[m.name] = m.negated ? negate(constant) : constant
        foreach(d -> replacements[d] = ZERO, chain)
      else
        local n = xor(m.negated, best.negated)
        replacements[m.name] = ref(best.name, n)
        foreach(((k, d),) -> replacements[d] = ref(bestChain[k], n), enumerate(chain))
        push!(merged, AliasSign(m.name, n))
      end
      isempty(chain) || push!(states, m.name)
      substituted = true
    end
    substituted && foreach(eq -> push!(definitions, eq), sets.equations[root])
    isempty(merged) ||
      mergeAliasStarts!(variables, best.name, sort!(merged; by = m::AliasSign -> (-priority(m.name), position[m.name])),
                        changedAttributes)
  end
  local undo = AliasUndo(partition.equations, partition.initialEquations,
                         Pair{Symbol, IR.VariableKind}[n => variables[n].kind for n in states],
                         IR.Variable[variables[d] for n in states for d in derivativeChain(variables, n)],
                         count(n -> !haskey(antiderivative, n), keys(replacements)), changedAttributes)
  isempty(replacements) && return undo
  local substitute = (@nospecialize(e::IR.Expression)) -> (e isa IR.CREF ? get(replacements, e.cref.name, e) : e)
  local reads = (eq::IR.Equation) ->
    IR.anyExp((@nospecialize(x::IR.Expression)) -> x isa IR.CREF && haskey(replacements, x.cref.name), eq)
  local rewrite = (@nospecialize(eq::IR.Equation)) ->
    (eq in definitions || eq isa Union{IR.WHEN_EQUATION, IR.ASSERT_EQUATION} || !reads(eq)) ? eq : IR.mapEquation!(substitute, deepcopy(eq))
  partition.equations = IR.Equation[rewrite(eq) for eq in partition.equations]
  partition.initialEquations = IR.Equation[rewrite(eq) for eq in partition.initialEquations]
  #= the states whose derivatives the substitution removed: algebraic =#
  local removed = Set{Symbol}(v.name for v in undo.removed)
  for n in states
    variables[n].kind = IR.ALGEBRAIC()
  end
  filter!(v -> !(v.name in removed), variables)
  return undo
end

"""
    aliasBlocked(variables, partition) -> Set{Symbol}

The variables of the partition whose equations alias substitution leaves as written (`substituteAliases!`): written
by a when-equation (`reinit`) or an algorithm, read as part of a whole record or array (`frame_a.R` passed to a
function: a name of no variable, the variables under it), or read through a derivative by a when-equation.
"""
function aliasBlocked(variables::IR.VariableSet, partition::IR.Partition)::Set{Symbol}
  local antiderivative::Dict{Symbol, Symbol} = antiderivatives(variables)
  local blocked::Set{Symbol} = Set{Symbol}()
  local resolve::Resolver = Resolver(Symbol[v.name for v in variables])
  local scanWhole = (@nospecialize(e::IR.Expression)) -> begin
    IR.foreachExp(e) do @nospecialize(x::IR.Expression)
      x isa IR.CREF || return nothing
      local n::Symbol = x.cref.name
      (haskey(variables, n) || !haskey(resolve.under, n)) && return nothing
      union!(blocked, resolve.under[n])
      nothing
    end
  end
  for eq in vcat(partition.equations, partition.initialEquations)
    if eq isa IR.ALGORITHM
      foreach(c -> union!(blocked, resolve(c.name)), eq.outputs)
    elseif eq isa IR.WHEN_EQUATION
      IR.foreachRootExp(eq) do @nospecialize(e::IR.Expression)
        IR.foreachExp(e) do @nospecialize(x::IR.Expression)
          (x isa IR.CREF && haskey(antiderivative, x.cref.name)) || return nothing
          push!(blocked, baseVariable(antiderivative, x.cref.name))
          nothing
        end
        nothing
      end
      for b in eq.branches, s in b.body
        s isa IR.REINIT && push!(blocked, s.state.name)
      end
    end
    IR.foreachRootExp(scanWhole, eq)
  end
  return blocked
end

"The variable a derivative variable is a derivative of, through its chain (`\$DER.\$DER.x` -> `x`; `antiderivatives`)."
function baseVariable(antiderivative::Dict{Symbol, Symbol}, name::Symbol)::Symbol
  local base::Symbol = name
  while haskey(antiderivative, base)
    base = antiderivative[base]
  end
  return base
end
