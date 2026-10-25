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

"The rolled variables' names by the instance's part (`\$DER.` and the base's suffix), and the auxiliaries' parts."
struct RolledNames
  base::Dict{Tuple{Int, String}, Symbol}
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

"""
    splitName(name, memberOf) -> Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}}

A variable name of an instance taken apart: its backend prefix (`\$DER.`), group, instance, the suffix after the
member without an element's literal subscripts (`b1.R.T` of `cylinder2.b1.R.T[1,2]`), and those subscripts;
`nothing` for a name of no instance.
"""
function splitName(name::String, memberOf::Dict{String, Tuple{Int, Int}})::Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}}
  local m::Union{Nothing, RegexMatch} = match(r"^((?:\$[A-Z]+\.)*)([^.\[]+)\.(.+)$", name)
  m === nothing && return nothing
  local at::Union{Nothing, Tuple{Int, Int}} = get(memberOf, String(m.captures[2]), nothing)
  at === nothing && return nothing
  local suffix::String = String(m.captures[3])
  local element::Union{Nothing, RegexMatch} = match(r"^(.*)\[([0-9]+(?:,[0-9]+)*)\]$", suffix)
  element === nothing && return (String(m.captures[1]), at[1], at[2], suffix, Int[])
  return (String(m.captures[1]), at[1], at[2], String(element.captures[1]), parse.(Int, split(element.captures[2], ',')))
end

"""
    RolledVariable

A rolled variable being built: its group, prefix and suffix, each instance's variable whole (or `nothing`) and its
element variables by their literal subscripts.
"""
struct RolledVariable
  group::Int
  prefix::String
  suffix::String
  whole::Vector{Union{Nothing, IR.Variable}}
  elements::Vector{Dict{Vector{Int}, IR.Variable}}
end

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
  local memberOf::Dict{String, Tuple{Int, Int}} = Dict{String, Tuple{Int, Int}}(String(m) => (1, k) for (k, m) in enumerate(g.members))
  local n::Int = length(g.members)
  local partition::IR.Partition = model.continuous[1]
  #= the instances' variables, by their prefix and suffix =#
  local byKey::Dict{Tuple{String, String}, RolledVariable} = Dict{Tuple{String, String}, RolledVariable}()
  local keyOrder::Vector{Tuple{String, String}} = Tuple{String, String}[]
  local owned::Set{Symbol} = Set{Symbol}()
  for v in model.variables
    local parts::Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}} = splitName(String(v.name), memberOf)
    parts === nothing && continue
    local (prefix::String, _, k::Int, suffix::String, index::Vector{Int}) = parts
    (isempty(prefix) && any(p::IR.CrefPart -> !isempty(p.subscripts), v.cref.parts[1:end - 1])) &&
      throw(NotHandled("the variable $(v.name), a component of its reference subscripted"))
    local key::Tuple{String, String} = (prefix, suffix)
    haskey(byKey, key) || (push!(keyOrder, key); byKey[key] = RolledVariable(1, prefix, suffix, Union{Nothing, IR.Variable}[nothing for _ in 1:n],
                                                                               Dict{Vector{Int}, IR.Variable}[Dict{Vector{Int}, IR.Variable}() for _ in 1:n]))
    local r::RolledVariable = byKey[key]
    isempty(index) ? (r.whole[k] = v) : (r.elements[k][index] = v)
    push!(owned, v.name)
  end
  isempty(byKey) && return false
  for key in keyOrder
    local r::RolledVariable = byKey[key]
    local wholes::Int = count(x::Union{Nothing, IR.Variable} -> x !== nothing, r.whole)
    (wholes == 0 || wholes == n) || throw(NotHandled("the variable $(key[1])$(key[2]) of only some instances"))
    all(d::Dict{Vector{Int}, IR.Variable} -> Set(keys(d)) == Set(keys(r.elements[1])), r.elements) ||
      throw(NotHandled("the elements of $(key[1])$(key[2]) differ between the instances"))
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
  local names::RolledNames = RolledNames(Dict{Tuple{Int, String}, Symbol}(), Dict{Symbol, Tuple{Int, Int, IR.CrefPart}}())
  local newVariables::Vector{IR.Variable} = IR.Variable[]
  #= the rolled variables: the bases first (a derivative's array is named after its base's) =#
  for key in sort(keyOrder; by = k::Tuple{String, String} -> (length(k[1]), k[1]))
    push!(newVariables, rolledVariable(byKey[key], g, groups, memberOf, names))
  end
  for (j, auxs) in enumerate(auxiliaries)
    push!(newVariables, rolledAuxiliary(auxs, j, g, model.variables, names))
  end
  #= the equations =#
  local rewrite::Function = (e::IR.Expression, index::Function) -> rolledExpression(e, groups, memberOf, names, index)
  local literal::Function = (gi::Int, k::Int) -> IR.INTEGER(k)
  local iterator::Symbol = Symbol("\$c")
  local equations::Vector{IR.Equation} = rolledEquations(partition.equations, groups, memberOf, names, n, iterator; aliases = true)
  local initial::Vector{IR.Equation} = rolledEquations(partition.initialEquations, groups, memberOf, names, n, iterator; aliases = false)
  local rolledCount::Int = count(eq::IR.Equation -> eq isa IR.FOR_EQUATION && eq.iterator.name === iterator, equations)
  #= little to gain: left as it is =#
  rolledCount * n * 2 >= count(eq::IR.Equation -> !isempty(instancesOf(eq, memberOf, names)), partition.equations) || return false
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
    push!(out, withRolledReferences(v, (e::IR.Expression) -> rewrite(e, literal)))
  end
  empty!(model.variables)
  foreach(v::IR.Variable -> push!(model.variables, v), out)
  partition.equations = equations
  partition.initialEquations = initial
  local unknowns::Vector{Symbol} = Symbol[]
  for u in partition.unknowns
    local name::Symbol = (splitName(String(u), memberOf) !== nothing || haskey(names.auxiliary, u)) ? rolledName(u, memberOf, names) : u
    name in unknowns || push!(unknowns, name)
  end
  empty!(partition.unknowns); append!(partition.unknowns, unknowns)
  @info "[index reduction] components $(join(String.(g.members), ", ")) rolled: $(length(newVariables)) arrays, $(rolledCount) for-equations"
  return true
end

"""
    auxiliaryVariables(model, partition, g, memberOf, owned) -> Vector{Vector{Symbol}}

The variables the passes made outside any component (`ifEq_tmp<k>`, not parameters) that only the equations of one instance read, matched
across the instances: one list per role (instance by instance), the role the normalized equation that defines it
(the first that reads it). NotHandled where the instances' roles differ.
"""
function auxiliaryVariables(model::IR.Model, partition::IR.Partition, g::ComponentGroup, memberOf::Dict{String, Tuple{Int, Int}},
                            owned::Set{Symbol})::Vector{Vector{Symbol}}
  local n::Int = length(g.members)
  local readers::Dict{Symbol, Set{Int}} = Dict{Symbol, Set{Int}}()
  local firstReader::Dict{Symbol, IR.Equation} = Dict{Symbol, IR.Equation}()
  local candidate::Function = (x::Symbol) -> begin
    local v::Union{Nothing, IR.Variable} = get(model.variables, x, nothing)
    v !== nothing && v.origin !== :model && !occursin('.', String(x)) && !(x in owned) && !(v.kind isa Union{IR.PARAMETER, IR.CONSTANT})
  end
  for eq in Iterators.flatten((partition.equations, partition.initialEquations))
    local instances::Set{Int} = Set{Int}()
    local free::Vector{Symbol} = Symbol[]
    IR.foreachExp(eq) do x::IR.Expression
      x isa IR.CREF || return nothing
      local parts::Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}} = splitName(String(variableNameOf(x.cref)), memberOf)
      parts === nothing || push!(instances, parts[3])
      local name::Symbol = variableNameOf(x.cref)
      candidate(name) && push!(free, name)
      nothing
    end
    for x in free
      union!(get!(() -> Set{Int}(), readers, x), isempty(instances) ? Set{Int}(0) : instances)
      haskey(firstReader, x) || (firstReader[x] = eq)
    end
  end
  local roles::Vector{Dict{String, Symbol}} = Dict{String, Symbol}[Dict{String, Symbol}() for _ in 1:n]
  for (x, ks) in readers
    (length(ks) == 1 && !(0 in ks)) || continue
    local k::Int = only(ks)
    local role::String = normalizedKey(IR.mapEquation!(e::IR.Expression -> roleExpression(e, x, memberOf), deepcopy(firstReader[x])))
    haskey(roles[k], role) && throw(NotHandled("two auxiliaries of one role in $(g.members[k])"))
    roles[k][role] = x
  end
  all(r::Dict{String, Symbol} -> Set(keys(r)) == Set(keys(roles[1])), roles) ||
    throw(NotHandled("the instances' auxiliary variables differ"))
  return Vector{Symbol}[Symbol[roles[k][role] for k in 1:n] for role in sort!(collect(keys(roles[1])))]
end

"The name of the variable a reference reads (its own, or without its subscripts)."
variableNameOf(cref::IR.ComponentRef)::Symbol = IR.hasSubscripts(cref) ? unsubscriptedName(cref) : cref.name

"`e` for an auxiliary's role: the auxiliary `x` as `§`, an instance's references by its suffix only."
function roleExpression(e::IR.Expression, x::Symbol, memberOf::Dict{String, Tuple{Int, Int}})::IR.Expression
  return IR.mapExp(e) do y::IR.Expression
    y isa IR.CREF || return y
    variableNameOf(y.cref) === x && return IR.CREF(IR.ComponentRef(Symbol("§")), y.ty)
    local parts::Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}} = splitName(String(y.cref.name), memberOf)
    parts === nothing && return y
    return IR.CREF(IR.ComponentRef(Symbol(parts[1], "§.", parts[4], isempty(parts[5]) ? "" : "[" * join(parts[5], ",") * "]")), y.ty)
  end
end

"""
    rolledVariable(r, g, groups, memberOf, names) -> IR.Variable

The array over the instances of the variables `r`: of a base (no prefix) its reference the group's part and the
suffix's parts (an element variable's last part the array of its elements), of a derivative one part named after its
base's rolled variable (`\$DER.` and that name); its kind the instances' (all alike), its binding and attributes the
instances' (`rolledAttribute`). Its name recorded in `names`.
"""
function rolledVariable(r::RolledVariable, g::ComponentGroup, groups::Vector{ComponentGroup}, memberOf::Dict{String, Tuple{Int, Int}},
                        names::RolledNames)::IR.Variable
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
    throw(NotHandled("the variable $(r.prefix)$(r.suffix): its instances of other kinds or types"))
  local cref::IR.ComponentRef = if isempty(r.prefix)
    local tail::Vector{IR.CrefPart} = IR.CrefPart[first.cref.parts[2:end]...]
    whole || (tail[end] = IR.CrefPart(tail[end].name, IR.Subscript[], IR.T_ARRAY(elementTy, IR.Dimension[IR.DIM_SIZE(d) for d in innerSizes])))
    IR.ComponentRef(IR.CrefPart[g.part; tail])
  else
    local base::Union{Nothing, Symbol} = get(names.base, (1, r.suffix), nothing)
    base === nothing && throw(NotHandled("the variable $(r.prefix)$(r.suffix) without its base"))
    IR.ComponentRef(Symbol(r.prefix, base), ty)
  end
  isempty(r.prefix) && (names.base[(1, r.suffix)] = cref.name)
  local literal::Function = (gi::Int, k::Int) -> IR.INTEGER(k)
  local rewrite::Function = (e::IR.Expression) -> rolledExpression(e, groups, memberOf, names, literal)
  local attribute::Function = (field::Symbol) -> rolledAttribute(r, n, innerSizes, (v::IR.Variable) -> getfield(v.attributes, field), rewrite)
  local a::IR.VariableAttributes = first.attributes
  local stateSelects::Vector{Union{Nothing, IR.StateSelect}} = Union{Nothing, IR.StateSelect}[x.attributes.stateSelect for x in everyOne]
  all(s::Union{Nothing, IR.StateSelect} -> s == stateSelects[1], stateSelects) ||
    throw(NotHandled("the variable $(r.prefix)$(r.suffix): its instances of other StateSelects"))
  local attributes::IR.VariableAttributes =
    IR.VariableAttributes(attribute(:start), attribute(:fixed), attribute(:nominal), attribute(:min), attribute(:max),
                          attribute(:unit), attribute(:displayUnit), attribute(:quantity), a.stateSelect)
  local binding::Union{Nothing, IR.Expression} = rolledAttribute(r, n, innerSizes, (v::IR.Variable) -> v.binding, rewrite)
  local kind::IR.VariableKind = rolledKind(first.kind, memberOf, names)
  return IR.Variable(cref, ty, kind, binding, attributes, first.direction, first.isProtected, first.connector, first.source, first.origin)
end

"The dimensions of the array of an instance's element variables (`R.T[1,1]` ... `R.T[3,3]`: 3, 3); NotHandled where they are not all of it."
function elementSizes(r::RolledVariable)::Vector{Int}
  local indices::Vector{Vector{Int}} = collect(keys(r.elements[1]))
  local d::Int = length(indices[1])
  all(i::Vector{Int} -> length(i) == d, indices) || throw(NotHandled("the elements of $(r.suffix) of different ranks"))
  local sizes::Vector{Int} = Int[maximum(i[j] for i in indices) for j in 1:d]
  length(indices) == prod(sizes) || throw(NotHandled("the elements of $(r.suffix), not all of an array"))
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
    throw(NotHandled("an attribute of $(r.prefix)$(r.suffix) of only some instances"))
  local rewritten::Vector{IR.Expression} = IR.Expression[rewrite(x::IR.Expression) for x in values]
  local same::Bool = all(x::IR.Expression -> x == rewritten[1], rewritten) && rewritten[1] == values[1]
  local scalar::Bool = !(IR.typeOf(rewritten[1]) isa IR.T_ARRAY)
  same && scalar && return rewritten[1]
  if whole && !isempty(innerSizes)
    scalar && throw(NotHandled("an attribute of the array $(r.prefix)$(r.suffix) given by a scalar that differs"))
    local elementTy::IR.ModelicaType = IR.elementType(IR.typeOf(rewritten[1]))
    return IR.ARRAY(rewritten, IR.T_ARRAY(elementTy, IR.Dimension[IR.DIM_SIZE(d) for d in [n; innerSizes]]))
  end
  return nestedArray([n; innerSizes], IR.typeOf(rewritten[1]), rewritten)
end

"The kind of the rolled variable: the instances'; the variable it names (a state's derivative, a derivative's state) the rolled one."
function rolledKind(kind::IR.VariableKind, memberOf::Dict{String, Tuple{Int, Int}}, names::RolledNames)::IR.VariableKind
  kind isa IR.STATE && kind.derivative !== nothing && return IR.STATE(kind.index, rolledName(kind.derivative, memberOf, names), kind.natural)
  kind isa IR.STATE_DER && return IR.STATE_DER(rolledName(kind.state, memberOf, names))
  kind isa Union{IR.DUMMY_DER, IR.DUMMY_STATE, IR.CLOCKED_STATE} && throw(NotHandled("an instance's variable of kind $(nameof(typeof(kind)))"))
  return kind
end

"The rolled variable an instance's variable `name` is an element of (a base's, a derivative's, an auxiliary's); NotHandled for another."
function rolledName(name::Symbol, memberOf::Dict{String, Tuple{Int, Int}}, names::RolledNames)::Symbol
  local aux::Union{Nothing, Tuple{Int, Int, IR.CrefPart}} = get(names.auxiliary, name, nothing)
  aux === nothing || return aux[3].name
  local parts::Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}} = splitName(String(name), memberOf)
  local base::Union{Nothing, Symbol} = parts === nothing ? nothing : get(names.base, (parts[2], parts[4]), nothing)
  base === nothing && throw(NotHandled("the variable $(name), of no rolled array"))
  return Symbol(parts[1], base)
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
is a member its group's part subscripted; a derivative's (`\$DER.cylinder2.x[2]`, one part) the rolled derivative's,
by the instance's subscript, the element's and its own; an auxiliary's its array's element. `nothing` for another.
"""
function rolledReference(cref::IR.ComponentRef, groups::Vector{ComponentGroup}, memberOf::Dict{String, Tuple{Int, Int}},
                         names::RolledNames, index::Function)::Union{Nothing, IR.ComponentRef}
  local first::IR.CrefPart = cref.parts[1]
  local at::Union{Nothing, Tuple{Int, Int}} = isempty(first.subscripts) ? get(memberOf, String(first.name), nothing) : nothing
  if at !== nothing && length(cref.parts) >= 2
    local g::ComponentGroup = groups[at[1]]
    return IR.ComponentRef(IR.CrefPart[IR.CrefPart(g.part.name, IR.Subscript[IR.INDEX(index(at[1], at[2]))], g.part.ty); cref.parts[2:end]])
  end
  length(cref.parts) == 1 || return nothing
  local aux::Union{Nothing, Tuple{Int, Int, IR.CrefPart}} = get(names.auxiliary, first.name, nothing)
  if aux !== nothing
    return IR.ComponentRef(IR.CrefPart[IR.CrefPart(aux[3].name, IR.Subscript[IR.INDEX(index(aux[1], aux[2])); first.subscripts], aux[3].ty)])
  end
  local parts::Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}} = splitName(String(first.name), memberOf)
  (parts === nothing || isempty(parts[1])) && return nothing
  local base::Union{Nothing, Symbol} = get(names.base, (parts[2], parts[4]), nothing)
  base === nothing && return nothing
  local subscripts::Vector{IR.Subscript} = IR.Subscript[IR.INDEX(index(parts[2], parts[3])); IR.Subscript[IR.INDEX(IR.INTEGER(i)) for i in parts[5]];
                                                        first.subscripts]
  return IR.ComponentRef(IR.CrefPart[IR.CrefPart(Symbol(parts[1], base), subscripts, first.ty)])
end

"""
    rolledExpression(e, groups, memberOf, names, index) -> IR.Expression

`e` reading the rolled arrays (`rolledReference`); an array of the rolled ones subscripted (`R.w[\$i1]` of an
element-only array) as the reference subscripted.
"""
function rolledExpression(e::IR.Expression, groups::Vector{ComponentGroup}, memberOf::Dict{String, Tuple{Int, Int}},
                          names::RolledNames, index::Function)::IR.Expression
  return IR.mapExp(e) do x::IR.Expression
    if x isa IR.CREF
      local r::Union{Nothing, IR.ComponentRef} = rolledReference(x.cref, groups, memberOf, names, index)
      return r === nothing ? x : IR.CREF(r, x.ty)
    elseif x isa IR.SUBSCRIPTED && x.exp isa IR.CREF && IR.typeOf(x.exp) isa IR.T_ARRAY && isRolledReference(x.exp.cref) &&
           all(s::IR.Subscript -> s isa IR.INDEX, x.subscripts)
      local parts::Vector{IR.CrefPart} = copy(x.exp.cref.parts)
      parts[end] = IR.CrefPart(parts[end].name, IR.Subscript[parts[end].subscripts; x.subscripts], parts[end].ty)
      return IR.CREF(IR.ComponentRef(parts), x.ty)
    end
    return x
  end
end

"Whether a reference reads a rolled array (its first part the group's, or a rolled derivative's one part)."
isRolledReference(cref::IR.ComponentRef)::Bool =
  cref.parts[1].ty isa IR.T_COMPONENTS || (length(cref.parts) == 1 && startswith(String(cref.parts[1].name), "\$") && occursin('|', String(cref.parts[1].name)))

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

The instances whose variables (records, components, derivatives, auxiliaries) the equation reads.
"""
function instancesOf(eq::IR.Equation, memberOf::Dict{String, Tuple{Int, Int}}, names::RolledNames)::Set{Int}
  local out::Set{Int} = Set{Int}()
  IR.foreachExp(eq) do x::IR.Expression
    x isa IR.CREF || return nothing
    local first::IR.CrefPart = x.cref.parts[1]
    local at::Union{Nothing, Tuple{Int, Int}} = (isempty(first.subscripts) && length(x.cref.parts) >= 2) ? get(memberOf, String(first.name), nothing) : nothing
    if at !== nothing
      push!(out, at[2])
    elseif length(x.cref.parts) == 1
      local aux::Union{Nothing, Tuple{Int, Int, IR.CrefPart}} = get(names.auxiliary, first.name, nothing)
      if aux !== nothing
        push!(out, aux[2])
      else
        local parts::Union{Nothing, Tuple{String, Int, Int, String, Vector{Int}}} = splitName(String(first.name), memberOf)
        parts === nothing || push!(out, parts[3])
      end
    end
    nothing
  end
  return out
end

"""
    rolledEquations(equations, groups, memberOf, names, n, iterator; aliases) -> Vector{IR.Equation}

The equations with the group's instances rolled: those of no instance as they are; with `aliases` the equalities of
two references of which one is an instance's (a connection set's) as their classes (`aliasClasses`); an equation of
one instance rolled into a for-equation over the instances (`iterator`) where each instance has it as often
(`normalizedKey` of its rolled form), once per occurrence, the first instance's as the body; the others element
equations (the instance's literal subscript).
"""
function rolledEquations(equations::Vector{IR.Equation}, groups::Vector{ComponentGroup}, memberOf::Dict{String, Tuple{Int, Int}},
                         names::RolledNames, n::Int, iterator::Symbol; aliases::Bool)::Vector{IR.Equation}
  local literal::Function = (gi::Int, k::Int) -> IR.INTEGER(k)
  local loop::Function = (gi::Int, k::Int) -> IR.CREF(IR.ComponentRef(iterator, IR.T_INTEGER()), IR.T_INTEGER())
  local rewrite::Function = (eq::IR.Equation, index::Function) ->
    IR.mapEquation!(e::IR.Expression -> rolledExpression(e, groups, memberOf, names, index), deepcopy(eq))
  local (kept::Vector{IR.Equation}, equalities::Vector{IR.Equation}) = aliases ? aliasClasses(equations, memberOf, names) : (equations, IR.Equation[])
  #= each instance's equations by their rolled forms' keys =#
  local keyed::Vector{Dict{String, Vector{IR.Equation}}} = Dict{String, Vector{IR.Equation}}[Dict{String, Vector{IR.Equation}}() for _ in 1:n]
  local keyOf::IdDict{IR.Equation, String} = IdDict{IR.Equation, String}()
  local ordered::Vector{IR.Equation} = IR.Equation[kept; equalities]
  for eq in ordered
    local instances::Set{Int} = instancesOf(eq, memberOf, names)
    length(instances) == 1 || continue
    local key::String = normalizedKey(rewrite(eq, loop))
    keyOf[eq] = key
    push!(get!(() -> IR.Equation[], keyed[only(instances)], key), eq)
  end
  local rolls::Dict{String, Int} = Dict{String, Int}(key => minimum(get(keyed[k], key, IR.Equation[]) |> length for k in 1:n)
                                                     for key in keys(keyed[1]))
  local out::Vector{IR.Equation} = IR.Equation[]
  local used::Dict{Tuple{Int, String}, Int} = Dict{Tuple{Int, String}, Int}()
  local range::IR.Expression = IR.RANGE(IR.INTEGER(1), nothing, IR.INTEGER(n), IR.T_ARRAY(IR.T_INTEGER(), IR.Dimension[IR.DIM_SIZE(n)]))
  for eq in ordered
    local instances::Set{Int} = instancesOf(eq, memberOf, names)
    if isempty(instances)
      push!(out, eq)
      continue
    end
    if length(instances) == 1
      local k::Int = only(instances)
      local key::String = keyOf[eq]
      local j::Int = get(used, (k, key), 0) + 1
      used[(k, key)] = j
      if j <= get(rolls, key, 0)
        #= the first instance's occurrence is the body; the others' are its iterations =#
        k == 1 && push!(out, IR.FOR_EQUATION(IR.Iterator(iterator, range), IR.Equation[rewrite(eq, loop)], eq.source,
                                             hasproperty(eq, :attributes) ? eq.attributes : IR.DEFAULT_ATTRIBUTES))
        continue
      end
    end
    push!(out, rewrite(eq, literal))
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
function aliasClasses(equations::Vector{IR.Equation}, memberOf::Dict{String, Tuple{Int, Int}}, names::RolledNames)::Tuple{Vector{IR.Equation}, Vector{IR.Equation}}
  local parent::Dict{String, String} = Dict{String, String}()
  local reference::Dict{String, Tuple{IR.CREF, IR.Equation}} = Dict{String, Tuple{IR.CREF, IR.Equation}}()
  local kept::Vector{IR.Equation} = IR.Equation[]
  for eq in equations
    local pair::Union{Nothing, Tuple{IR.CREF, IR.CREF, String}} = equalityPair(eq)
    if pair === nothing || isempty(instancesOf(eq, memberOf, names))
      push!(kept, eq)
      continue
    end
    local (a::IR.CREF, b::IR.CREF, shape::String) = pair
    local ka::String = shape * String(a.cref.name)
    local kb::String = shape * String(b.cref.name)
    haskey(reference, ka) || (reference[ka] = (a, eq))
    haskey(reference, kb) || (reference[kb] = (b, eq))
    local (ra::String, rb::String) = (findRoot!(parent, ka), findRoot!(parent, kb))
    ra == rb || (parent[ra] = rb)
  end
  local classes::Dict{String, Vector{String}} = Dict{String, Vector{String}}()
  for x in sort!(collect(keys(reference)))
    push!(get!(() -> String[], classes, findRoot!(parent, x)), x)
  end
  local equalities::Vector{IR.Equation} = IR.Equation[]
  for root in sort!(collect(keys(classes)))
    #= the class's references by instance (0: the other components'), each part in the order of its names past the member =#
    local byInstance::Dict{Int, Vector{String}} = Dict{Int, Vector{String}}()
    for x in classes[root]
      local k::Int = let s = instancesOf(IR.SCALAR_EQUATION(reference[x][1], reference[x][1], IR.T_REAL(), reference[x][2].source, reference[x][2].attributes), memberOf, names)
        isempty(s) ? 0 : only(s)
      end
      push!(get!(() -> String[], byInstance, k), x)
    end
    local chain::Function = (a::String, b::String) -> equalityEquation(reference[a][1], reference[b][1], reference[a][2])
    local representatives::Vector{String} = String[]
    for k in sort!(collect(keys(byInstance)))
      local members::Vector{String} = sort(byInstance[k]; by = x::String -> relativeName(x, memberOf))
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
function findRoot!(parent::Dict{String, String}, x::String)::String
  local p::String = get(parent, x, x)
  p == x && return x
  local root::String = findRoot!(parent, p)
  parent[x] = root
  return root
end

"A name with its instance's member replaced (`§.b.x` of `cylinder2.b.x`), to order a class's part alike in every instance."
function relativeName(x::String, memberOf::Dict{String, Tuple{Int, Int}})::String
  local m::Union{Nothing, RegexMatch} = match(r"^(\[[0-9,]*\]|)((?:\$[A-Z]+\.)*)([^.\[]+)(.*)$", x)
  m === nothing && return x
  haskey(memberOf, String(m.captures[3])) || return x
  return string(m.captures[1], m.captures[2], "§", m.captures[4])
end

"""
    equalityPair(eq) -> Union{Nothing, Tuple{IR.CREF, IR.CREF, String}}

The two references of an equality: a scalar equation of two references (shape ""); a for-equation of one body
`a[i] = b[i]` over its iterator, both subscripted by it alone (shape the range's size, `[n]`), the references the
whole arrays. `nothing` for another equation.
"""
function equalityPair(eq::IR.Equation)::Union{Nothing, Tuple{IR.CREF, IR.CREF, String}}
  if eq isa IR.SCALAR_EQUATION
    (eq.lhs isa IR.CREF && eq.rhs isa IR.CREF) || return nothing
    (IR.hasSubscripts(eq.lhs.cref) && !literalSubscripts(eq.lhs.cref)) && return nothing
    (IR.hasSubscripts(eq.rhs.cref) && !literalSubscripts(eq.rhs.cref)) && return nothing
    return (eq.lhs, eq.rhs, "")
  elseif eq isa IR.FOR_EQUATION && length(eq.body) == 1 && eq.body[1] isa IR.SCALAR_EQUATION
    local body::IR.SCALAR_EQUATION = eq.body[1]
    (body.lhs isa IR.CREF && body.rhs isa IR.CREF) || return nothing
    local range::Union{Nothing, StepRange{Int, Int}} = literalRange(eq.iterator.range)
    (range !== nothing && first(range) == 1 && step(range) == 1) || return nothing
    local byIterator::Function = (x::IR.CREF) -> begin
      local s::Vector{IR.Subscript} = x.cref.parts[end].subscripts
      length(s) == 1 && s[1] isa IR.INDEX && (s[1]::IR.INDEX).exp isa IR.CREF && ((s[1]::IR.INDEX).exp::IR.CREF).cref.name === eq.iterator.name &&
        !any(p::IR.CrefPart -> !isempty(p.subscripts), x.cref.parts[1:end - 1])
    end
    (byIterator(body.lhs) && byIterator(body.rhs)) || return nothing
    local whole::Function = (x::IR.CREF) -> IR.CREF(withoutLastSubscripts(x.cref), x.ty)
    return (whole(body.lhs), whole(body.rhs), "[$(last(range))]")
  end
  return nothing
end

"Whether every subscript of the reference is a literal index."
literalSubscripts(cref::IR.ComponentRef)::Bool =
  all(p::IR.CrefPart -> all(s::IR.Subscript -> s isa IR.INDEX && (s::IR.INDEX).exp isa IR.INTEGER, p.subscripts), cref.parts)

"The reference without its last part's subscripts."
withoutLastSubscripts(cref::IR.ComponentRef)::IR.ComponentRef =
  IR.ComponentRef(IR.CrefPart[cref.parts[1:end - 1]..., IR.CrefPart(cref.parts[end].name, IR.Subscript[], cref.parts[end].ty)])

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
  local element::Function = (x::IR.CREF) -> begin
    local parts::Vector{IR.CrefPart} = copy(x.cref.parts)
    parts[end] = IR.CrefPart(parts[end].name, IR.Subscript[IR.INDEX(i)], parts[end].ty)
    IR.CREF(IR.ComponentRef(parts), body.ty)
  end
  return IR.FOR_EQUATION(f.iterator, IR.Equation[IR.SCALAR_EQUATION(element(a), element(b), body.ty, body.source, body.attributes)],
                         f.source, f.attributes)
end

"""
    normalizedKey(eq) -> String

An equation's form up to the order of its sides and of a sum's terms (and its sign: `a = b` is `b = a`, `a + b = c`
is `c - b = a`): a scalar equation by its signed terms sorted; another as printed.
"""
function normalizedKey(eq::IR.Equation)::String
  eq isa IR.SCALAR_EQUATION || return replace(string(eq), r"\s+" => " ")
  local terms::Vector{String} = sort!(signedTerms!(signedTerms!(String[], eq.lhs, true), eq.rhs, false))
  local flipped::Vector{String} = sort!(String[(t[1] == '+' ? "-" : "+") * t[2:end] for t in terms])
  return min(join(terms, " "), join(flipped, " "))
end

"The signed terms of a sum (`+a`, `-b`), zeros left out, each term's own sums normalized."
function signedTerms!(out::Vector{String}, e::IR.Expression, positive::Bool)::Vector{String}
  if e isa IR.BINARY && e.op in (IR.ADD, IR.SUB)
    signedTerms!(out, e.exp1, positive)
    signedTerms!(out, e.exp2, e.op == IR.ADD ? positive : !positive)
  elseif e isa IR.UNARY && e.op == IR.UMINUS
    signedTerms!(out, e.exp, !positive)
  elseif !(e isa Union{IR.REAL, IR.INTEGER} && iszero(e.value))
    push!(out, (positive ? "+" : "-") * termString(e))
  end
  return out
end

"A term as printed, its sums and the factors of its products in a normal order."
function termString(e::IR.Expression)::String
  if (e isa IR.BINARY && e.op in (IR.ADD, IR.SUB)) || (e isa IR.UNARY && e.op == IR.UMINUS)
    return "(" * join(sort!(signedTerms!(String[], e, true)), " ") * ")"
  elseif e isa IR.BINARY && e.op == IR.MUL
    return "(" * join(sort!(String[termString(e.exp1), termString(e.exp2)]), " * ") * ")"
  end
  return replace(string(e), r"\s+" => " ")
end
