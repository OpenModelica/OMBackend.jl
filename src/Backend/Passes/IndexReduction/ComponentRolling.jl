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
#= Component rolling: separately declared components of one class (EngineV6's cylinder1, ..., cylinder6) as arrays
   over the instances, so that the array index reduction and the array code take them as a component array
   (`Pendulum p[3]`'s `p[i].x`). On the model the passes leave with its arrays kept (records decomposed, array
   equations rolled, if-expressions lifted: every instance alike):
   - every variable of the instances one array, its first dimension the instance's, on a part of type
     T_COMPONENTS that names an element by its member (`cylinder3.x`): the element view, the scalarized model and
     the results keep the names; an element variable of a record's array field (`R.T[1,2]`) an element of the array;
     a derivative (`\$DER.cylinder2.x`) an array named after the rolled one; an auxiliary variable only one instance
     reads (an if-expression's `ifEq_tmp<k>`) an array too;
   - an instance's equations rolled into for-equations over the instances where every instance has them (matched by
     a normalized form: the sides and a sum's terms in any order; a connection set's equalities as its classes, each
     instance's part of a class a chain), the others and those that read several instances as element equations.
   The model's meaning, and its scalarized form, are unchanged. =#

"""
    ComponentGroup(cls, members, part)

Separately declared top-level components of the class `cls`, `members` in order, and the part that stands for them in
the rolled variables' references (named by the members joined by `|`, of type `T_COMPONENTS`).
"""
struct ComponentGroup
  cls::Symbol
  members::Vector{Symbol}
  part::IR.CrefPart
end

"The rolled base variables' references by their group and the instances' rest (`MemberReference`), and the auxiliaries' parts."
struct RolledNames
  base::Dict{Tuple{Int, Symbol}, IR.ComponentRef}
  auxiliary::Dict{Symbol, Tuple{Int, Int, IR.CrefPart}}
end

"""
    componentGroups(variables) -> Vector{ComponentGroup}

The top-level components of one class declared separately, at least two (a part of type `T_CLASS`, not subscripted,
first in the references of its variables).
"""
function componentGroups(variables::IR.VariableSet)::Vector{ComponentGroup}
  local byClass::Dict{Symbol, Vector{Symbol}} = Dict{Symbol, Vector{Symbol}}()
  local order::Vector{Symbol} = Symbol[]
  local seen::Set{Symbol} = Set{Symbol}()
  for v in variables
    length(v.cref.parts) >= 2 || continue
    local first::IR.CrefPart = v.cref.parts[1]
    (isempty(first.subscripts) && first.ty isa IR.T_CLASS) || continue
    first.name in seen && continue
    push!(seen, first.name)
    local cls::Symbol = (first.ty::IR.T_CLASS).name
    haskey(byClass, cls) || push!(order, cls)
    push!(get!(() -> Symbol[], byClass, cls), first.name)
  end
  local groups::Vector{ComponentGroup} = ComponentGroup[]
  for cls in order
    local members::Vector{Symbol} = byClass[cls]
    length(members) >= 2 || continue
    push!(groups, ComponentGroup(cls, members, IR.CrefPart(Symbol(join(String.(members), "|")), IR.Subscript[],
                                                           IR.T_COMPONENTS(cls, members))))
  end
  return groups
end

"The placeholder for an instance's member in a reference compared across the instances (`memberRelative`, an auxiliary's role)."
const MEMBER_PLACEHOLDER = Symbol("§")

"The subscript of the instance `k` of the group `gi` in a reference outside a rolled equation: its literal index."
instanceLiteral(gi::Int, k::Int)::IR.Expression = IR.INTEGER(k)

"""
The share of the instances' equations rolled into for-equations below which a group is left as it is (`rollGroup!`):
little to gain.
"""
const MIN_ROLLED_SHARE = 0.5

"""
    MemberReference

A reference into an instance of a group taken apart by its parts (`memberReference`): its derivative parts (`\$DER`,
`IR.isDerivativeCref`), its group and instance (the member: the first part after them, not subscripted, with parts
after it), the flat name of the parts after the member without an element's literal subscripts on the last part
(`b1.R.T` of `cylinder2.b1.R.T[1,2]`), and those subscripts.
"""
struct MemberReference
  operators::Vector{IR.CrefPart}
  group::Int
  instance::Int
  rest::Symbol
  index::Vector{Int}
end

"""
    memberReference(cref, memberOf) -> Union{Nothing, MemberReference}

`cref` taken apart as a reference into an instance (`MemberReference`); `nothing` for a reference of no instance (a
rolled member's, `T_MEMBER`, reads a rolled array).
"""
function memberReference(cref::IR.ComponentRef, memberOf::Dict{Symbol, Tuple{Int, Int}})::Union{Nothing, MemberReference}
  local k::Int = IR.operandStart(cref)
  k < length(cref.parts) || return nothing
  local member::IR.CrefPart = cref.parts[k]
  (isempty(member.subscripts) && !IR.isMember(member)) || return nothing
  local at::Union{Nothing, Tuple{Int, Int}} = get(memberOf, member.name, nothing)
  at === nothing && return nothing
  local rest::Vector{IR.CrefPart} = cref.parts[k + 1:end]
  local last::IR.CrefPart = rest[end]
  local index::Union{Nothing, Vector{Int}} = isempty(last.subscripts) ? nothing : IR.literalIndices(last.subscripts)
  index === nothing && return MemberReference(cref.parts[1:k - 1], at[1], at[2], IR.flatName(rest), Int[])
  rest[end] = IR.CrefPart(last.name, IR.Subscript[], last.ty)
  return MemberReference(cref.parts[1:k - 1], at[1], at[2], IR.flatName(rest), index)
end

"""
    RolledVariable

A rolled variable being built: its group, the instances' derivative parts and rest (`MemberReference`), each
instance's variable whole (or `nothing`) and its element variables by their literal subscripts.
"""
struct RolledVariable
  group::Int
  operators::Vector{IR.CrefPart}
  rest::Symbol
  whole::Vector{Union{Nothing, IR.Variable}}
  elements::Vector{Dict{Vector{Int}, IR.Variable}}
end

"A rolled variable for messages: its derivative parts and rest joined by `.` (`\$DER.b1.R.T`)."
variableLabel(r::RolledVariable)::String = join(String[String[String(p.name) for p in r.operators]; String(r.rest)], ".")

"""
    rollComponents!(model) -> Int

The separately declared components of one class of `model` (with its arrays kept, through the passes) rolled into
arrays over the instances (see the file's comment), in place; the number of groups rolled. A group whose instances'
variables differ (names, types, kinds), or that rolls few of its equations, is left as it is; so is a model of
several partitions, clocks or modes.
"""
function rollComponents!(model::IR.Model)::Int
  (length(model.continuous) == 1 && isempty(model.clocked) && isempty(model.modes)) || return 0
  local rolled::Int = 0
  for g in componentGroups(model.variables)
    try
      rollGroup!(model, g) && (rolled += 1)
    catch err
      err isa NotHandled || rethrow()
      @info "[index reduction] components $(join(String.(g.members), ", ")) not rolled: $(err.reason)"
    end
  end
  return rolled
end

"""
    rollGroup!(model, g) -> Bool

The components of the group `g` rolled (`rollComponents!`); `false` where its equations roll too little to gain.
NotHandled where the instances differ: the model is then unchanged.
"""
function rollGroup!(model::IR.Model, g::ComponentGroup)::Bool
  local groups::Vector{ComponentGroup} = ComponentGroup[g]
  local memberOf::Dict{Symbol, Tuple{Int, Int}} = Dict{Symbol, Tuple{Int, Int}}(m => (1, k) for (k, m) in enumerate(g.members))
  local n::Int = length(g.members)
  local partition::IR.Partition = model.continuous[1]
  #= the instances' variables, by their derivative parts' names and rest =#
  local byKey::Dict{Tuple{Symbol, Symbol}, RolledVariable} = Dict{Tuple{Symbol, Symbol}, RolledVariable}()
  local keyOrder::Vector{Tuple{Symbol, Symbol}} = Tuple{Symbol, Symbol}[]
  local owned::Set{Symbol} = Set{Symbol}()
  #= the variables' references (the instances' are replaced below) =#
  local crefOf::Dict{Symbol, IR.ComponentRef} = Dict{Symbol, IR.ComponentRef}(v.name => v.cref for v in model.variables)
  for v in model.variables
    local m::Union{Nothing, MemberReference} = memberReference(v.cref, memberOf)
    m === nothing && continue
    (isempty(m.operators) && any(p::IR.CrefPart -> !isempty(p.subscripts), v.cref.parts[1:end - 1])) &&
      throw(NotHandled("the variable $(v.name), a component of its reference subscripted"))
    local key::Tuple{Symbol, Symbol} = (IR.flatName(m.operators), m.rest)
    haskey(byKey, key) || (push!(keyOrder, key); byKey[key] = RolledVariable(1, m.operators, m.rest, Union{Nothing, IR.Variable}[nothing for _ in 1:n],
                                                                               Dict{Vector{Int}, IR.Variable}[Dict{Vector{Int}, IR.Variable}() for _ in 1:n]))
    local r::RolledVariable = byKey[key]
    isempty(m.index) ? (r.whole[m.instance] = v) : (r.elements[m.instance][m.index] = v)
    push!(owned, v.name)
  end
  isempty(byKey) && return false
  for key in keyOrder
    local r::RolledVariable = byKey[key]
    local wholes::Int = count(x::Union{Nothing, IR.Variable} -> x !== nothing, r.whole)
    (wholes == 0 || wholes == n) || throw(NotHandled("the variable $(variableLabel(r)) of only some instances"))
    all(d::Dict{Vector{Int}, IR.Variable} -> Set(keys(d)) == Set(keys(r.elements[1])), r.elements) ||
      throw(NotHandled("the elements of $(variableLabel(r)) differ between the instances"))
    #= a record's array field of a parameter, its elements variables too (expandRecordFieldArrays): the array rolled
       from its elements, each instance's array kept beside them for a call's whole argument (the code and the
       evaluation read a record argument's fields by name) =#
    if wholes == n && !isempty(r.elements[1])
      foreach(x::Union{Nothing, IR.Variable} -> delete!(owned, (x::IR.Variable).name), r.whole)
      fill!(r.whole, nothing)
    end
  end
  #= auxiliaries (an if-expression's `ifEq_tmp<k>`): read only by the equations of one instance =#
  local auxiliaries::Vector{Vector{Symbol}} = auxiliaryVariables(model, partition, g, memberOf, owned)
  local names::RolledNames = RolledNames(Dict{Tuple{Int, Symbol}, IR.ComponentRef}(), Dict{Symbol, Tuple{Int, Int, IR.CrefPart}}())
  local newVariables::Vector{IR.Variable} = IR.Variable[]
  #= the rolled variables: the bases first (a derivative's array is named after its base's) =#
  for key in sort(keyOrder; by = k::Tuple{Symbol, Symbol} -> (length(byKey[k].operators), k[1]))
    push!(newVariables, rolledVariable(byKey[key], g, groups, memberOf, names, crefOf))
  end
  for (j, auxs) in enumerate(auxiliaries)
    push!(newVariables, rolledAuxiliary(auxs, j, g, model.variables, names))
  end
  #= the equations =#
  local rewrite::Function = (e::IR.Expression, index::Function) -> rolledExpression(e, groups, memberOf, names, index)
  local iterator::Symbol = Symbol("\$c")
  local equations::Vector{IR.Equation} = rolledEquations(partition.equations, groups, memberOf, names, n, iterator; aliases = true)
  local initial::Vector{IR.Equation} = rolledEquations(partition.initialEquations, groups, memberOf, names, n, iterator; aliases = false)
  local rolledCount::Int = count(eq::IR.Equation -> eq isa IR.FOR_EQUATION && eq.iterator.name === iterator, equations)
  #= little to gain: left as it is =#
  rolledCount * n >= MIN_ROLLED_SHARE * count(eq::IR.Equation -> !isempty(instancesOf(eq, memberOf, names)), partition.equations) || return false
  #= the variables: the instances' replaced by the rolled ones, in the place of the first instance's first =#
  local out::Vector{IR.Variable} = IR.Variable[]
  local placed::Bool = false
  local auxiliarySet::Set{Symbol} = Set{Symbol}(Iterators.flatten(auxiliaries))
  for v in model.variables
    if v.name in owned || v.name in auxiliarySet
      placed || append!(out, newVariables)
      placed = true
      continue
    end
    #= bindings and attributes of the other variables may read the instances' =#
    push!(out, withRolledReferences(v, (e::IR.Expression) -> rewrite(e, instanceLiteral)))
  end
  empty!(model.variables)
  foreach(v::IR.Variable -> push!(model.variables, v), out)
  partition.equations = equations
  partition.initialEquations = initial
  local unknowns::Vector{Symbol} = Symbol[]
  for u in partition.unknowns
    local name::Symbol = ((haskey(crefOf, u) && memberReference(crefOf[u], memberOf) !== nothing) || haskey(names.auxiliary, u)) ?
      rolledName(u, crefOf, memberOf, names) : u
    name in unknowns || push!(unknowns, name)
  end
  empty!(partition.unknowns); append!(partition.unknowns, unknowns)
  @info "[index reduction] components $(join(String.(g.members), ", ")) rolled: $(length(newVariables)) arrays, $(rolledCount) for-equations"
  return true
end

"""
A term's form (`termForm`): an expression as written (`LeafForm`, alike by `IR.sameForm`), a sum's signed terms
(`SumForm`) or a product's factors (`ProductForm`), each in a normal order (by their hashes, the same in every build).
"""
abstract type TermForm end

struct LeafForm <: TermForm
  e::IR.Expression
end

struct SumForm <: TermForm
  terms::Vector{Tuple{Bool, TermForm}}
end

struct ProductForm <: TermForm
  factors::Vector{TermForm}
end

"An equation other than a scalar one as written (`IR.sameForm`)."
struct EquationForm
  eq::IR.Equation
end

Base.:(==)(a::LeafForm, b::LeafForm)::Bool = IR.sameForm(a.e, b.e)
Base.:(==)(a::SumForm, b::SumForm)::Bool = a.terms == b.terms
Base.:(==)(a::ProductForm, b::ProductForm)::Bool = a.factors == b.factors
Base.:(==)(a::EquationForm, b::EquationForm)::Bool = IR.sameForm(a.eq, b.eq)
Base.hash(a::LeafForm, h::UInt)::UInt = IR.formHash(a.e, hash(:LeafForm, h))
Base.hash(a::SumForm, h::UInt)::UInt = hash(a.terms, hash(:SumForm, h))
Base.hash(a::ProductForm, h::UInt)::UInt = hash(a.factors, hash(:ProductForm, h))
Base.hash(a::EquationForm, h::UInt)::UInt = IR.formHash(a.eq, hash(:EquationForm, h))

"An equation's form (`equationForm`): a scalar equation's terms (`SumForm`), another equation as written."
const RolledForm = Union{SumForm, EquationForm}

"""
    equationForm(eq) -> RolledForm

An equation's form up to the order of its sides and of a sum's terms (and its sign: `a = b` is `b = a`, `a + b = c`
is `c - b = a`): a scalar equation by its signed terms in a normal order (`SumForm`; of the two signs the one whose
terms' hashes are the lesser); another as written (`EquationForm`). The instances' equations of one role have one form.
"""
function equationForm(eq::IR.Equation)::RolledForm
  eq isa IR.SCALAR_EQUATION || return EquationForm(eq)
  local terms::Vector{Tuple{Bool, TermForm}} =
    sort!(signedTerms!(signedTerms!(Tuple{Bool, TermForm}[], eq.lhs, true), eq.rhs, false); by = hash)
  local flipped::Vector{Tuple{Bool, TermForm}} = sort!(Tuple{Bool, TermForm}[(!s, f) for (s, f) in terms]; by = hash)
  return SumForm(UInt[hash(t) for t in terms] <= UInt[hash(t) for t in flipped] ? terms : flipped)
end

"The signed terms of a sum (`(true, a)`, `(false, b)`), zeros left out, each term's form (`termForm`)."
function signedTerms!(out::Vector{Tuple{Bool, TermForm}}, e::IR.Expression, positive::Bool)::Vector{Tuple{Bool, TermForm}}
  if e isa IR.BINARY && e.op in (IR.ADD, IR.SUB)
    signedTerms!(out, e.exp1, positive)
    signedTerms!(out, e.exp2, e.op == IR.ADD ? positive : !positive)
  elseif e isa IR.UNARY && e.op == IR.UMINUS
    signedTerms!(out, e.exp, !positive)
  elseif !(e isa Union{IR.REAL, IR.INTEGER} && iszero(e.value))
    push!(out, (positive, termForm(e)))
  end
  return out
end

"A term's form: its sums' terms and its products' factors in a normal order; another expression as written."
function termForm(e::IR.Expression)::TermForm
  if (e isa IR.BINARY && e.op in (IR.ADD, IR.SUB)) || (e isa IR.UNARY && e.op == IR.UMINUS)
    return SumForm(sort!(signedTerms!(Tuple{Bool, TermForm}[], e, true); by = hash))
  elseif e isa IR.BINARY && e.op == IR.MUL
    return ProductForm(sort!(TermForm[termForm(e.exp1), termForm(e.exp2)]; by = hash))
  end
  return LeafForm(e)
end

"""
    auxiliaryVariables(model, partition, g, memberOf, owned) -> Vector{Vector{Symbol}}

The variables the passes made outside any component (`ifEq_tmp<k>`: of one part, no derivative by the kinds; not parameters) that only the equations of one instance read, matched
across the instances: one list per role (instance by instance), the role the form of the equation that defines it
(the first that reads it, `equationForm`), in the order the equations first read the first instance's. NotHandled
where the instances' roles differ.
"""
function auxiliaryVariables(model::IR.Model, partition::IR.Partition, g::ComponentGroup, memberOf::Dict{Symbol, Tuple{Int, Int}},
                            owned::Set{Symbol})::Vector{Vector{Symbol}}
  local n::Int = length(g.members)
  local readers::Dict{Symbol, Set{Int}} = Dict{Symbol, Set{Int}}()
  local firstReader::Dict{Symbol, IR.Equation} = Dict{Symbol, IR.Equation}()
  local readOrder::Vector{Symbol} = Symbol[]
  local antiderivative::Dict{Symbol, Symbol} = antiderivatives(model.variables)
  local candidate::Function = (x::Symbol) -> begin
    local v::Union{Nothing, IR.Variable} = get(model.variables, x, nothing)
    v !== nothing && v.origin !== :model && length(v.cref.parts) == 1 && !haskey(antiderivative, x) && !(x in owned) &&
      !(v.kind isa Union{IR.PARAMETER, IR.CONSTANT})
  end
  for eq in Iterators.flatten((partition.equations, partition.initialEquations))
    local instances::Set{Int} = Set{Int}()
    local free::Vector{Symbol} = Symbol[]
    IR.foreachExp(eq) do x::IR.Expression
      x isa IR.CREF || return nothing
      local whole::IR.ComponentRef = IR.hasSubscripts(x.cref) ? IR.withoutSubscripts(x.cref) : x.cref
      local m::Union{Nothing, MemberReference} = memberReference(whole, memberOf)
      m === nothing || push!(instances, m.instance)
      local name::Symbol = whole.name
      candidate(name) && push!(free, name)
      nothing
    end
    for x in free
      union!(get!(() -> Set{Int}(), readers, x), isempty(instances) ? Set{Int}(0) : instances)
      haskey(firstReader, x) || (firstReader[x] = eq; push!(readOrder, x))
    end
  end
  local roles::Vector{Dict{RolledForm, Symbol}} = Dict{RolledForm, Symbol}[Dict{RolledForm, Symbol}() for _ in 1:n]
  for (x, ks) in readers
    (length(ks) == 1 && !(0 in ks)) || continue
    local k::Int = only(ks)
    local role::RolledForm = equationForm(IR.mapEquation!(e::IR.Expression -> roleExpression(e, x, memberOf), deepcopy(firstReader[x])))
    haskey(roles[k], role) && throw(NotHandled("two auxiliaries of one role in $(g.members[k])"))
    roles[k][role] = x
  end
  all(r::Dict{RolledForm, Symbol} -> Set(keys(r)) == Set(keys(roles[1])), roles) ||
    throw(NotHandled("the instances' auxiliary variables differ"))
  local position::Dict{Symbol, Int} = Dict{Symbol, Int}(x => i for (i, x) in enumerate(readOrder))
  local ordered::Vector{RolledForm} = sort!(collect(keys(roles[1])); by = role::RolledForm -> position[roles[1][role]])
  return Vector{Symbol}[Symbol[roles[k][role] for k in 1:n] for role in ordered]
end

"`e` for an auxiliary's role: the auxiliary `x` as `§` (`MEMBER_PLACEHOLDER`), an instance's references with their member as it (`memberRelative`)."
function roleExpression(e::IR.Expression, x::Symbol, memberOf::Dict{Symbol, Tuple{Int, Int}})::IR.Expression
  return IR.mapExp(e) do y::IR.Expression
    y isa IR.CREF || return y
    unsubscriptedName(y.cref) === x && return IR.CREF(IR.ComponentRef(MEMBER_PLACEHOLDER), y.ty)
    memberReference(y.cref, memberOf) === nothing && return y
    return IR.CREF(memberRelative(y.cref), y.ty)
  end
end

"""
    rolledVariable(r, g, groups, memberOf, names, crefOf) -> IR.Variable

The array over the instances of the variables `r`: of a base (no derivative parts) its reference the group's part and
the parts after the member (an element variable's last part the array of its elements), of a derivative the instances' derivative
parts before its base's rolled reference (`\$DER.a|b|c.x`); its kind the instances' (all alike, the variables it names
by `crefOf`), its binding and attributes the instances' (`rolledAttribute`). A base's reference recorded in `names`.
"""
function rolledVariable(r::RolledVariable, g::ComponentGroup, groups::Vector{ComponentGroup}, memberOf::Dict{Symbol, Tuple{Int, Int}},
                        names::RolledNames, crefOf::Dict{Symbol, IR.ComponentRef})::IR.Variable
  local n::Int = length(g.members)
  local whole::Bool = r.whole[1] !== nothing
  local first::IR.Variable = whole ? r.whole[1]::IR.Variable : r.elements[1][sort!(collect(keys(r.elements[1])))[1]]
  local innerSizes::Vector{Int} = whole ? knownSizes(first.ty, "the variable $(first.name)") : elementSizes(r)
  local elementTy::IR.ModelicaType = IR.elementType(first.ty)
  local ty::IR.ModelicaType = IR.T_ARRAY(elementTy, IR.Dimension[IR.DIM_SIZE(d) for d in [n; innerSizes]])
  #= every element of every instance of one kind and type =#
  local everyOne::Vector{IR.Variable} = whole ? IR.Variable[x::IR.Variable for x in r.whole] :
    IR.Variable[r.elements[k][index] for k in 1:n for index in rowMajor(innerSizes)]
  all(x::IR.Variable -> typeof(x.kind) == typeof(first.kind) && x.ty == first.ty, everyOne) ||
    throw(NotHandled("the variable $(variableLabel(r)): its instances of other kinds or types"))
  local cref::IR.ComponentRef = if isempty(r.operators)
    local tail::Vector{IR.CrefPart} = first.cref.parts[2:end]
    whole || (tail[end] = IR.CrefPart(tail[end].name, IR.Subscript[], IR.T_ARRAY(elementTy, IR.Dimension[IR.DIM_SIZE(d) for d in innerSizes])))
    IR.ComponentRef(IR.CrefPart[g.part; tail])
  else
    local base::Union{Nothing, IR.ComponentRef} = get(names.base, (1, r.rest), nothing)
    base === nothing && throw(NotHandled("the variable $(variableLabel(r)) without its base"))
    IR.withOperatorsOf(first.cref, base)
  end
  isempty(r.operators) && (names.base[(1, r.rest)] = cref)
  local rewrite::Function = (e::IR.Expression) -> rolledExpression(e, groups, memberOf, names, instanceLiteral)
  local attribute::Function = (field::Symbol) -> rolledAttribute(r, n, innerSizes, (v::IR.Variable) -> getfield(v.attributes, field), rewrite)
  local a::IR.VariableAttributes = first.attributes
  local stateSelects::Vector{Union{Nothing, IR.StateSelect}} = Union{Nothing, IR.StateSelect}[x.attributes.stateSelect for x in everyOne]
  all(s::Union{Nothing, IR.StateSelect} -> s == stateSelects[1], stateSelects) ||
    throw(NotHandled("the variable $(variableLabel(r)): its instances of other StateSelects"))
  local attributes::IR.VariableAttributes =
    IR.VariableAttributes(attribute(:start), attribute(:fixed), attribute(:nominal), attribute(:min), attribute(:max),
                          attribute(:unit), attribute(:displayUnit), attribute(:quantity), a.stateSelect)
  local binding::Union{Nothing, IR.Expression} = rolledAttribute(r, n, innerSizes, (v::IR.Variable) -> v.binding, rewrite)
  local kind::IR.VariableKind = rolledKind(first.kind, crefOf, memberOf, names)
  return IR.Variable(cref, ty, kind, binding, attributes, first.direction, first.isProtected, first.connector, first.source, first.origin)
end

"The dimensions of the array of an instance's element variables (`R.T[1,1]` ... `R.T[3,3]`: 3, 3); NotHandled where they are not all of it."
function elementSizes(r::RolledVariable)::Vector{Int}
  local indices::Vector{Vector{Int}} = collect(keys(r.elements[1]))
  local d::Int = length(indices[1])
  all(i::Vector{Int} -> length(i) == d, indices) || throw(NotHandled("the elements of $(variableLabel(r)) of different ranks"))
  local sizes::Vector{Int} = Int[maximum(i[j] for i in indices) for j in 1:d]
  length(indices) == prod(sizes) || throw(NotHandled("the elements of $(variableLabel(r)), not all of an array"))
  return sizes
end

"""
    rolledAttribute(r, n, innerSizes, get, rewrite) -> Union{Nothing, IR.Expression}

An attribute (or the binding, `get`) of the rolled variable `r`: none where no instance has one; the instances' one
where it is the same, reads no instance and is a scalar (each element's); otherwise the array literal of the
instances' (element by element where the instances have element variables: a record's array field of a parameter
has both, the elements' bindings evaluated as the scalarized model's), their references to instances rewritten
(`rewrite`). NotHandled where only some have one.
"""
function rolledAttribute(r::RolledVariable, n::Int, innerSizes::Vector{Int}, get::Function, rewrite::Function)::Union{Nothing, IR.Expression}
  local whole::Bool = isempty(r.elements[1])
  local values::Vector{Union{Nothing, IR.Expression}} = whole ? Union{Nothing, IR.Expression}[get(x::IR.Variable) for x in r.whole] :
    Union{Nothing, IR.Expression}[get(r.elements[k][index]) for k in 1:n for index in rowMajor(innerSizes)]
  all(x::Union{Nothing, IR.Expression} -> x === nothing, values) && return nothing
  any(x::Union{Nothing, IR.Expression} -> x === nothing, values) &&
    throw(NotHandled("an attribute of $(variableLabel(r)) of only some instances"))
  local rewritten::Vector{IR.Expression} = IR.Expression[rewrite(x::IR.Expression) for x in values]
  local same::Bool = all(x::IR.Expression -> x == rewritten[1], rewritten) && rewritten[1] == values[1]
  local scalar::Bool = !(IR.typeOf(rewritten[1]) isa IR.T_ARRAY)
  same && scalar && return rewritten[1]
  if whole && !isempty(innerSizes)
    scalar && throw(NotHandled("an attribute of the array $(variableLabel(r)) given by a scalar that differs"))
    local elementTy::IR.ModelicaType = IR.elementType(IR.typeOf(rewritten[1]))
    return IR.ARRAY(rewritten, IR.T_ARRAY(elementTy, IR.Dimension[IR.DIM_SIZE(d) for d in [n; innerSizes]]))
  end
  return nestedArray([n; innerSizes], IR.typeOf(rewritten[1]), rewritten)
end

"The kind of the rolled variable: the instances'; the variable it names (a state's derivative, a derivative's state) the rolled one."
function rolledKind(kind::IR.VariableKind, crefOf::Dict{Symbol, IR.ComponentRef}, memberOf::Dict{Symbol, Tuple{Int, Int}},
                    names::RolledNames)::IR.VariableKind
  kind isa Union{IR.DUMMY_DER, IR.DUMMY_STATE, IR.CLOCKED_STATE} && throw(NotHandled("an instance's variable of kind $(nameof(typeof(kind)))"))
  return IR.mapKindNames((x::Symbol) -> rolledName(x, crefOf, memberOf, names), kind)
end

"The rolled variable an instance's variable `name` (its reference by `crefOf`) is an element of (a base's, a derivative's, an auxiliary's); NotHandled for another."
function rolledName(name::Symbol, crefOf::Dict{Symbol, IR.ComponentRef}, memberOf::Dict{Symbol, Tuple{Int, Int}}, names::RolledNames)::Symbol
  local aux::Union{Nothing, Tuple{Int, Int, IR.CrefPart}} = get(names.auxiliary, name, nothing)
  aux === nothing || return aux[3].name
  local cref::Union{Nothing, IR.ComponentRef} = get(crefOf, name, nothing)
  local m::Union{Nothing, MemberReference} = cref === nothing ? nothing : memberReference(cref, memberOf)
  local base::Union{Nothing, IR.ComponentRef} = m === nothing ? nothing : get(names.base, (m.group, m.rest), nothing)
  base === nothing && throw(NotHandled("the variable $(name), of no rolled array"))
  return IR.withOperatorsOf(cref, base).name
end

"""
    rolledAuxiliary(auxs, j, g, variables, names) -> IR.Variable

The auxiliaries `auxs` (one per instance, of role `j`) as an array of one part of type `T_COMPONENTS` (an element
named as its auxiliary), recorded in `names`.
"""
function rolledAuxiliary(auxs::Vector{Symbol}, j::Int, g::ComponentGroup, variables::IR.VariableSet, names::RolledNames)::IR.Variable
  local first::IR.Variable = variables[auxs[1]]
  isempty(IR.dimensions(first.ty)) || throw(NotHandled("the auxiliary array $(first.name)"))
  all(x::Symbol -> typeof(variables[x].kind) == typeof(first.kind) && variables[x].ty == first.ty, auxs) ||
    throw(NotHandled("auxiliaries $(join(String.(auxs), ", ")) of other kinds or types"))
  local part::IR.CrefPart = IR.CrefPart(Symbol(join(String.(auxs), "|")), IR.Subscript[], IR.T_COMPONENTS(Symbol("auxiliary", j), auxs))
  for (k, x) in enumerate(auxs)
    names.auxiliary[x] = (1, k, part)
  end
  local starts::Vector{Union{Nothing, IR.Expression}} = Union{Nothing, IR.Expression}[variables[x].attributes.start for x in auxs]
  local start::Union{Nothing, IR.Expression} = all(s::Union{Nothing, IR.Expression} -> s == starts[1], starts) ? starts[1] :
    throw(NotHandled("auxiliaries $(join(String.(auxs), ", ")) of other start values"))
  local a::IR.VariableAttributes = first.attributes
  local attributes::IR.VariableAttributes = IR.VariableAttributes(start, a.fixed, a.nominal, a.min, a.max, a.unit, a.displayUnit,
                                                                 a.quantity, a.stateSelect)
  local ty::IR.ModelicaType = IR.T_ARRAY(first.ty, IR.Dimension[IR.DIM_SIZE(length(auxs))])
  return IR.Variable(IR.ComponentRef(IR.CrefPart[part]), ty, first.kind, first.binding, attributes, first.direction,
                     first.isProtected, first.connector, first.source, first.origin)
end

"""
    rolledReference(cref, ty, groups, memberOf, names, index) -> Union{Nothing, IR.ComponentRef}

A reference to an instance's variable, record or component as a reference to the rolled arrays, the instance's
subscript `index(group, instance)` (a literal, or the iterator of a rolled equation): a reference whose first part
is a member its group's part subscripted; a derivative's (`\$DER.cylinder2.x[2]`) the rolled derivative's, by the
instance's subscript, the element's and its own on its last part (`IR.withLastSubscripts`); an auxiliary's its array's
element. `nothing` for another (a rolled member's, `T_MEMBER`: it reads the rolled arrays).
"""
function rolledReference(cref::IR.ComponentRef, groups::Vector{ComponentGroup}, memberOf::Dict{Symbol, Tuple{Int, Int}},
                         names::RolledNames, index::Function)::Union{Nothing, IR.ComponentRef}
  local first::IR.CrefPart = cref.parts[1]
  IR.isMember(first) && return nothing
  local at::Union{Nothing, Tuple{Int, Int}} = isempty(first.subscripts) ? get(memberOf, first.name, nothing) : nothing
  if at !== nothing && length(cref.parts) >= 2
    local g::ComponentGroup = groups[at[1]]
    return IR.ComponentRef(IR.CrefPart[IR.CrefPart(g.part.name, IR.Subscript[IR.INDEX(index(at[1], at[2]))], g.part.ty); cref.parts[2:end]])
  end
  if IR.isDerivativeCref(cref)
    local m::Union{Nothing, MemberReference} = memberReference(IR.withoutLastSubscripts(cref), memberOf)
    m === nothing && return nothing
    local base::Union{Nothing, IR.ComponentRef} = get(names.base, (m.group, m.rest), nothing)
    base === nothing && return nothing
    local subscripts::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(index(m.group, m.instance)); IR.Subscript[IR.INDEX(IR.INTEGER(i)) for i in m.index];
                                                          cref.parts[end].subscripts]
    return IR.withLastSubscripts(IR.withOperatorsOf(cref, base), subscripts)
  end
  length(cref.parts) == 1 || return nothing
  local aux::Union{Nothing, Tuple{Int, Int, IR.CrefPart}} = get(names.auxiliary, first.name, nothing)
  aux === nothing && return nothing
  return IR.ComponentRef(IR.CrefPart[IR.CrefPart(aux[3].name, IR.Subscript[IR.INDEX(index(aux[1], aux[2])); first.subscripts], aux[3].ty)])
end

"""
    rolledExpression(e, groups, memberOf, names, index) -> IR.Expression

`e` reading the rolled arrays (`rolledReference`); an array of the rolled ones subscripted (`R.w[\$i1]` of an
element-only array) as the reference subscripted.
"""
function rolledExpression(e::IR.Expression, groups::Vector{ComponentGroup}, memberOf::Dict{Symbol, Tuple{Int, Int}},
                          names::RolledNames, index::Function)::IR.Expression
  return IR.mapExp(e) do x::IR.Expression
    if x isa IR.CREF
      local r::Union{Nothing, IR.ComponentRef} = rolledReference(x.cref, groups, memberOf, names, index)
      return r === nothing ? x : IR.CREF(r, x.ty)
    elseif x isa IR.SUBSCRIPTED && x.exp isa IR.CREF && IR.typeOf(x.exp) isa IR.T_ARRAY && isRolledReference(x.exp.cref) &&
           all(s::IR.Subscript -> s isa IR.INDEX, x.subscripts)
      return IR.CREF(IR.appendSubscripts(x.exp.cref, x.subscripts), x.ty)
    end
    return x
  end
end

"Whether a reference reads a rolled array: its first part after its derivative parts a group's or a member's (`IR.isRolled`)."
isRolledReference(cref::IR.ComponentRef)::Bool = IR.isRolled(cref.parts[IR.operandStart(cref)].ty)

"A variable with its binding and attributes reading the rolled arrays (`rewrite`)."
function withRolledReferences(v::IR.Variable, rewrite::Function)::IR.Variable
  local f::Function = (x::Union{Nothing, IR.Expression}) -> x === nothing ? nothing : rewrite(x)
  local a::IR.VariableAttributes = v.attributes
  v.binding = f(v.binding)
  v.attributes = IR.VariableAttributes(f(a.start), f(a.fixed), f(a.nominal), f(a.min), f(a.max), f(a.unit), f(a.displayUnit),
                                       f(a.quantity), a.stateSelect)
  return v
end

"""
    instancesOf(eq, memberOf, names) -> Set{Int}

The instances whose variables (records, components, derivatives, auxiliaries) the equation reads (`instanceOf`).
"""
function instancesOf(eq::IR.Equation, memberOf::Dict{Symbol, Tuple{Int, Int}}, names::RolledNames)::Set{Int}
  local out::Set{Int} = Set{Int}()
  IR.foreachExp(eq) do x::IR.Expression
    x isa IR.CREF || return nothing
    local k::Int = instanceOf(x.cref, memberOf, names)
    k == 0 || push!(out, k)
    nothing
  end
  return out
end

"""
    instanceOf(cref, memberOf, names) -> Int

The instance whose variable (record, component, derivative, auxiliary) the reference `cref` reads; 0 for another (a
rolled member's, `T_MEMBER`).
"""
function instanceOf(cref::IR.ComponentRef, memberOf::Dict{Symbol, Tuple{Int, Int}}, names::RolledNames)::Int
  local first::IR.CrefPart = cref.parts[1]
  IR.isMember(first) && return 0
  local at::Union{Nothing, Tuple{Int, Int}} = (isempty(first.subscripts) && length(cref.parts) >= 2) ? get(memberOf, first.name, nothing) : nothing
  at === nothing || return at[2]
  if IR.isDerivativeCref(cref)
    local m::Union{Nothing, MemberReference} = memberReference(IR.withoutLastSubscripts(cref), memberOf)
    return m === nothing ? 0 : m.instance
  end
  length(cref.parts) == 1 || return 0
  local aux::Union{Nothing, Tuple{Int, Int, IR.CrefPart}} = get(names.auxiliary, first.name, nothing)
  return aux === nothing ? 0 : aux[2]
end

"""
    rolledEquations(equations, groups, memberOf, names, n, iterator; aliases) -> Vector{IR.Equation}

The equations with the group's instances rolled: those of no instance as they are; with `aliases` the equalities of
two references of which one is an instance's (a connection set's) as their classes (`aliasClasses`); an equation of
one instance rolled into a for-equation over the instances (`iterator`) where each instance has it as often
(`equationForm` of its rolled form), once per occurrence, the first instance's as the body; the others element
equations (the instance's literal subscript).
"""
function rolledEquations(equations::Vector{IR.Equation}, groups::Vector{ComponentGroup}, memberOf::Dict{Symbol, Tuple{Int, Int}},
                         names::RolledNames, n::Int, iterator::Symbol; aliases::Bool)::Vector{IR.Equation}
  local loop::Function = (gi::Int, k::Int) -> IR.CREF(IR.ComponentRef(iterator, IR.T_INTEGER()), IR.T_INTEGER())
  local rewrite::Function = (eq::IR.Equation, index::Function) ->
    IR.mapEquation!(e::IR.Expression -> rolledExpression(e, groups, memberOf, names, index), deepcopy(eq))
  local (kept::Vector{IR.Equation}, equalities::Vector{IR.Equation}) = aliases ? aliasClasses(equations, memberOf, names) : (equations, IR.Equation[])
  #= each instance's equations by their rolled forms =#
  local keyed::Vector{Dict{RolledForm, Vector{IR.Equation}}} = Dict{RolledForm, Vector{IR.Equation}}[Dict{RolledForm, Vector{IR.Equation}}() for _ in 1:n]
  local keyOf::IdDict{IR.Equation, RolledForm} = IdDict{IR.Equation, RolledForm}()
  local ordered::Vector{IR.Equation} = IR.Equation[kept; equalities]
  for eq in ordered
    local instances::Set{Int} = instancesOf(eq, memberOf, names)
    length(instances) == 1 || continue
    local key::RolledForm = equationForm(rewrite(eq, loop))
    keyOf[eq] = key
    push!(get!(() -> IR.Equation[], keyed[only(instances)], key), eq)
  end
  local rolls::Dict{RolledForm, Int} = Dict{RolledForm, Int}(key => minimum(get(keyed[k], key, IR.Equation[]) |> length for k in 1:n)
                                                             for key in keys(keyed[1]))
  local out::Vector{IR.Equation} = IR.Equation[]
  local used::Dict{Tuple{Int, RolledForm}, Int} = Dict{Tuple{Int, RolledForm}, Int}()
  local range::IR.Expression = IR.RANGE(IR.INTEGER(1), nothing, IR.INTEGER(n), IR.T_ARRAY(IR.T_INTEGER(), IR.Dimension[IR.DIM_SIZE(n)]))
  for eq in ordered
    local instances::Set{Int} = instancesOf(eq, memberOf, names)
    if isempty(instances)
      push!(out, eq)
      continue
    end
    if length(instances) == 1
      local k::Int = only(instances)
      local key::RolledForm = keyOf[eq]
      local j::Int = get(used, (k, key), 0) + 1
      used[(k, key)] = j
      if j <= get(rolls, key, 0)
        #= the first instance's occurrence is the body; the others' are its iterations =#
        k == 1 && push!(out, IR.FOR_EQUATION(IR.Iterator(iterator, range), IR.Equation[rewrite(eq, loop)], eq.source,
                                             hasproperty(eq, :attributes) ? eq.attributes : IR.DEFAULT_ATTRIBUTES))
        continue
      end
    end
    push!(out, rewrite(eq, instanceLiteral))
  end
  return out
end

"""
    aliasClasses(equations, memberOf, names) -> Tuple{Vector{IR.Equation}, Vector{IR.Equation}}

The equations without the equalities of two references (scalars `a = b`; arrays `for i loop a[i] = b[i]`) of which one
is an instance's, and those equalities again by their classes (union-find): each instance's part of a class a chain
in the order of the names (alike in every instance), the parts linked by their first references (the other
components' part first). The number of equations is the same.
"""
function aliasClasses(equations::Vector{IR.Equation}, memberOf::Dict{Symbol, Tuple{Int, Int}}, names::RolledNames)::Tuple{Vector{IR.Equation}, Vector{IR.Equation}}
  #= a reference of an equality by its shape (0: a scalar's; `n`: the whole arrays of a for-equation over 1:n) and name =#
  local parent::Dict{Tuple{Int, Symbol}, Tuple{Int, Symbol}} = Dict{Tuple{Int, Symbol}, Tuple{Int, Symbol}}()
  local reference::Dict{Tuple{Int, Symbol}, Tuple{IR.CREF, IR.Equation}} = Dict{Tuple{Int, Symbol}, Tuple{IR.CREF, IR.Equation}}()
  local kept::Vector{IR.Equation} = IR.Equation[]
  for eq in equations
    local pair::Union{Nothing, Tuple{IR.CREF, IR.CREF, Int}} = equalityPair(eq)
    if pair === nothing || isempty(instancesOf(eq, memberOf, names))
      push!(kept, eq)
      continue
    end
    local (a::IR.CREF, b::IR.CREF, shape::Int) = pair
    local ka::Tuple{Int, Symbol} = (shape, a.cref.name)
    local kb::Tuple{Int, Symbol} = (shape, b.cref.name)
    haskey(reference, ka) || (reference[ka] = (a, eq))
    haskey(reference, kb) || (reference[kb] = (b, eq))
    local (ra::Tuple{Int, Symbol}, rb::Tuple{Int, Symbol}) = (findRoot!(parent, ka), findRoot!(parent, kb))
    ra == rb || (parent[ra] = rb)
  end
  local classes::Dict{Tuple{Int, Symbol}, Vector{Tuple{Int, Symbol}}} = Dict{Tuple{Int, Symbol}, Vector{Tuple{Int, Symbol}}}()
  for x in sort!(collect(keys(reference)); by = classOrder)
    push!(get!(() -> Tuple{Int, Symbol}[], classes, findRoot!(parent, x)), x)
  end
  local equalities::Vector{IR.Equation} = IR.Equation[]
  for root in sort!(collect(keys(classes)); by = classOrder)
    #= the class's references by instance (0: the other components'), each part in the order of its names past the member =#
    local byInstance::Dict{Int, Vector{Tuple{Int, Symbol}}} = Dict{Int, Vector{Tuple{Int, Symbol}}}()
    for x in classes[root]
      local k::Int = instanceOf(reference[x][1].cref, memberOf, names)
      push!(get!(() -> Tuple{Int, Symbol}[], byInstance, k), x)
    end
    local chain::Function = (a::Tuple{Int, Symbol}, b::Tuple{Int, Symbol}) -> equalityEquation(reference[a][1], reference[b][1], reference[a][2])
    local representatives::Vector{Tuple{Int, Symbol}} = Tuple{Int, Symbol}[]
    for k in sort!(collect(keys(byInstance)))
      local members::Vector{Tuple{Int, Symbol}} = sort(byInstance[k]; by = x::Tuple{Int, Symbol} -> memberOrder(x[1], reference[x][1].cref, memberOf))
      for j in 1:length(members) - 1
        push!(equalities, chain(members[j], members[j + 1]))
      end
      push!(representatives, members[1])
    end
    for j in 1:length(representatives) - 1
      push!(equalities, chain(representatives[j], representatives[j + 1]))
    end
  end
  return (kept, equalities)
end

"The root of `x`'s class in the union-find `parent` (the path compressed)."
function findRoot!(parent::Dict{K, K}, x::K)::K where {K}
  local p::K = get(parent, x, x)
  p == x && return x
  local root::K = findRoot!(parent, p)
  parent[x] = root
  return root
end

"The order of an equality's references (`aliasClasses`): by their flat names after their shape's (`[n]`), the order the equations were emitted in before the keys were (shape, name), kept as the output depends on it (an order by the keys would reorder the equalities)."
classOrder(key::Tuple{Int, Symbol})::String = string(key[1] == 0 ? "" : "[$(key[1])]", key[2])

"""
    memberOrder(shape, cref, memberOf) -> String

The order of a class's references in an instance: by the flat name of a reference with its instance's member (the
first part after its derivative parts) as `§` (`§.b.x` of `cylinder2.b.x`), after its equality's shape (`classOrder`),
alike in every instance.
"""
function memberOrder(shape::Int, cref::IR.ComponentRef, memberOf::Dict{Symbol, Tuple{Int, Int}})::String
  local member::IR.CrefPart = cref.parts[IR.operandStart(cref)]
  (haskey(memberOf, member.name) && !IR.isMember(member)) || return classOrder((shape, cref.name))
  return classOrder((shape, memberRelative(cref).name))
end

"A reference with its member (the first part after its derivative parts) as `§` (`\$DER.§.b.x` of `\$DER.cylinder2.b.x`)."
function memberRelative(cref::IR.ComponentRef)::IR.ComponentRef
  local k::Int = IR.operandStart(cref)
  local parts::Vector{IR.CrefPart} = copy(cref.parts)
  parts[k] = IR.CrefPart(MEMBER_PLACEHOLDER, parts[k].subscripts, parts[k].ty)
  return IR.ComponentRef(parts)
end

"""
    equalityPair(eq) -> Union{Nothing, Tuple{IR.CREF, IR.CREF, Int}}

The two references of an equality: a scalar equation of two references (shape 0); a for-equation of one body
`a[i] = b[i]` over its iterator, both subscripted by it alone (shape the range's size `n`), the references the
whole arrays. `nothing` for another equation.
"""
function equalityPair(eq::IR.Equation)::Union{Nothing, Tuple{IR.CREF, IR.CREF, Int}}
  if eq isa IR.SCALAR_EQUATION
    (eq.lhs isa IR.CREF && eq.rhs isa IR.CREF) || return nothing
    (IR.hasSubscripts(eq.lhs.cref) && !literalSubscripts(eq.lhs.cref)) && return nothing
    (IR.hasSubscripts(eq.rhs.cref) && !literalSubscripts(eq.rhs.cref)) && return nothing
    return (eq.lhs, eq.rhs, 0)
  elseif eq isa IR.FOR_EQUATION && length(eq.body) == 1 && eq.body[1] isa IR.SCALAR_EQUATION
    local body::IR.SCALAR_EQUATION = eq.body[1]
    (body.lhs isa IR.CREF && body.rhs isa IR.CREF) || return nothing
    local range::Union{Nothing, StepRange{Int, Int}} = literalRange(eq.iterator.range)
    (range !== nothing && first(range) == 1 && step(range) == 1) || return nothing
    local byIterator::Function = (x::IR.CREF) -> begin
      local s::Vector{IR.Subscript} = x.cref.parts[end].subscripts
      length(s) == 1 && s[1] isa IR.INDEX && (s[1]::IR.INDEX).exp isa IR.CREF && ((s[1]::IR.INDEX).exp::IR.CREF).cref.name === eq.iterator.name &&
        !any(IR.isSubscripted, @view x.cref.parts[1:end - 1])
    end
    (byIterator(body.lhs) && byIterator(body.rhs)) || return nothing
    local whole::Function = (x::IR.CREF) -> IR.CREF(IR.withoutLastSubscripts(x.cref), x.ty)
    return (whole(body.lhs), whole(body.rhs), last(range))
  end
  return nothing
end

"Whether every subscript of the reference is a literal index."
literalSubscripts(cref::IR.ComponentRef)::Bool =
  all(p::IR.CrefPart -> all(s::IR.Subscript -> IR.literalIndex(s) !== nothing, p.subscripts), cref.parts)

"""
    equalityEquation(a, b, like) -> IR.Equation

`a = b` as the equality `like` was (a scalar equation; for whole arrays, a for-equation over their elements with
`like`'s iterator).
"""
function equalityEquation(a::IR.CREF, b::IR.CREF, like::IR.Equation)::IR.Equation
  like isa IR.SCALAR_EQUATION && return IR.SCALAR_EQUATION(a, b, like.ty, like.source, like.attributes)
  local f::IR.FOR_EQUATION = like::IR.FOR_EQUATION
  local body::IR.SCALAR_EQUATION = f.body[1]::IR.SCALAR_EQUATION
  local i::IR.Expression = IR.CREF(IR.ComponentRef(f.iterator.name, IR.T_INTEGER()), IR.T_INTEGER())
  local element::Function = (x::IR.CREF) -> IR.CREF(IR.withLastSubscripts(x.cref, IR.Subscript[IR.INDEX(i)]), body.ty)
  return IR.FOR_EQUATION(f.iterator, IR.Equation[IR.SCALAR_EQUATION(element(a), element(b), body.ty, body.source, body.attributes)],
                         f.source, f.attributes)
end

