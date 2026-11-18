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
#= The derivative unknowns of a model reduced in array form solved for in array form, so that its arrays reach the
   code generation (ArrayODEGen): the decisions those of `solveDerivatives!` on the element view, committed per
   for-equation as the plan is (arrayCommit.jl). A body whose iterations determine the elements of a derivative array
   is solved once for it, its iterators kept, and its value put into the equations that read the array; the result
   scalarized is checked against the element view solved, equation by equation. Where the elements of one array are
   solved differently (a boundary element) or the array form is not the elements' (a coefficient that differs per
   element), NotHandled: the caller takes the scalarized path. =#

"""
    SolvedTemplate(cref, iterators, value)

A derivative unknown solved in array form: its reference in the body it is solved from (`\$DER.x[\$i1]`, or a
scalar's own), the iterators of the for-equations around that body (outermost first), and its value, over them.
"""
struct SolvedTemplate
  cref::IR.CREF
  iterators::Vector{Symbol}
  value::IR.Expression
end

"""
    withIteratorExpressions(e, values) -> Expression

`e` with each iterator of `values` (name => expression) replaced by its expression, in one pass, in the subscripts of
its references too (`withIterators` with expressions); inside a nested constructor or reduction, the iterators it
binds again are not.
"""
function withIteratorExpressions(e::IR.Expression, values::Dict{Symbol, IR.Expression})::IR.Expression
  isempty(values) && return e
  return IR.mapExpTopDown(e) do x::IR.Expression
    if x isa Union{IR.ARRAY_CONSTRUCTOR, IR.REDUCTION} && any(it::IR.Iterator -> haskey(values, it.name), x.iterators)
      local inner::Dict{Symbol, IR.Expression} =
        filter(((name, _)::Pair{Symbol, IR.Expression},) -> all(it::IR.Iterator -> it.name !== name, x.iterators), values)
      return (IR.mapChildren(c::IR.Expression -> withIteratorExpressions(c, inner), x), false)
    end
    x isa IR.CREF || return (x, true)
    (length(x.cref.parts) == 1 && !IR.hasSubscripts(x.cref) && haskey(values, x.cref.name)) && return (values[x.cref.name], false)
    IR.hasSubscripts(x.cref) || return (x, false)
    local parts::Vector{IR.CrefPart} = IR.CrefPart[IR.CrefPart(p.name, IR.Subscript[s isa IR.INDEX ? IR.INDEX(withIteratorExpressions(s.exp, values)) : s
                                                               for s in p.subscripts], p.ty) for p in x.cref.parts]
    return (IR.CREF(foldSubscripts(IR.ComponentRef(parts)), x.ty), false)
  end
end

"""
    solvedTarget(x, E, templates) -> Union{Nothing, Symbol}

The solved derivative a reference reads: the array of a reference to an element of a solved derivative array (by
any subscripts) or to the whole array (`R * \$DER.r`), a solved scalar by its name; `nothing` for another.
"""
function solvedTarget(x::IR.Expression, E::Elements, templates::Dict{Symbol, SolvedTemplate})::Union{Nothing, Symbol}
  x isa IR.CREF || return nothing
  #= a scalar of its own first (an element of an array split for index reduction, `x[2]`) =#
  (haskey(templates, x.cref.name) && !haskey(E.sizes, x.cref.name)) && return x.cref.name
  local array::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
  return (haskey(E.sizes, array) && haskey(templates, array)) ? array : nothing
end

"""
    instantiated(t, x, values) -> Expression

The value `values` (over the iterators of the template `t`) of the solved derivative at the reference `x`: its
iterators by `x`'s subscripts (`\$DER.x[\$i1]`'s value at `\$DER.x[j + 1]` with `\$i1` as `j + 1`); a scalar's as
it is; at a reference to the whole array, `wholeArray`. NotHandled for a reference by a slice.
"""
function instantiated(t::SolvedTemplate, x::IR.CREF, values::IR.Expression)::IR.Expression
  isempty(t.iterators) && return values
  local subscripts::Vector{IR.Subscript} = allSubscripts(x.cref)
  local own::Vector{IR.Subscript} = allSubscripts(t.cref.cref)
  isempty(subscripts) && return wholeArray(t, x, values)
  (length(subscripts) == length(own) && all(s::IR.Subscript -> s isa IR.INDEX, subscripts)) ||
    throw(NotHandled("a solved derivative read by a slice or partly subscripted ($(x.cref.name))"))
  local map::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}()
  for (s, o) in zip(subscripts, own)
    map[(o::IR.INDEX).exp.cref.name] = (s::IR.INDEX).exp
  end
  return withIteratorExpressions(values, map)
end

"The most elements of a solved derivative array whose value `wholeArray` writes out as an array literal."
const WHOLE_ARRAY_ELEMENTS = 27

"""
    wholeArray(t, x, values) -> Expression

The value `values` of the solved derivative array of the template `t` at `x`, a reference to the whole array
(`R * \$DER.r`, a vector or matrix expression): the array its value reads by the template's own subscripts in their
order (`v[\$i1]`: `v`, of the same sizes), else the array literal of its elements' values (`instantiated` at each),
up to `WHOLE_ARRAY_ELEMENTS` elements. NotHandled for a larger array, sizes not known, or a template whose own
subscripts are not its iterators.
"""
function wholeArray(t::SolvedTemplate, x::IR.CREF, values::IR.Expression)::IR.Expression
  local own::Vector{IR.Subscript} = allSubscripts(t.cref.cref)
  local iterators::Vector{Symbol} = Symbol[]
  for o in own
    (o isa IR.INDEX && o.exp isa IR.CREF && length(o.exp.cref.parts) == 1 && o.exp.cref.name in t.iterators) ||
      throw(NotHandled("a solved derivative read as a whole array, its template subscripted by other than its iterators ($(x.cref.name))"))
    push!(iterators, o.exp.cref.name)
  end
  local sizes::Vector{Union{Nothing, Int}} = x.ty isa IR.T_ARRAY ? Union{Nothing, Int}[IR.dimensionSize(d) for d in x.ty.dimensions] : Union{Nothing, Int}[]
  (length(sizes) == length(own) && all((s::Union{Nothing, Int}) -> s !== nothing, sizes)) ||
    throw(NotHandled("a solved derivative read as a whole array of sizes not known ($(x.cref.name))"))
  local known::Vector{Int} = Int[s for s in sizes]
  local same::Union{Nothing, IR.CREF} = sameSizedArray(values, iterators, known, x.ty)
  same === nothing || return same
  prod(known) <= WHOLE_ARRAY_ELEMENTS ||
    throw(NotHandled("a solved derivative read as a whole array of $(prod(known)) elements ($(x.cref.name))"))
  local leaves::Vector{IR.Expression} = IR.Expression[]
  #= row-major: the last subscript fastest =#
  for at in CartesianIndices(Tuple(reverse(known)))
    local index::Vector{Int} = reverse(collect(Tuple(at)))
    local map::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}(it => IR.INTEGER(k) for (it, k) in zip(iterators, index))
    push!(leaves, withIteratorExpressions(values, map))
  end
  return nestedArray(known, (x.ty::IR.T_ARRAY).elementType, leaves)
end

"""
    sameSizedArray(values, iterators, sizes, ty) -> Union{Nothing, CREF}

The array `values` reads by `iterators` in their order as its only subscripts, on its last part (`v[\$i1, \$i2]`: `v`),
where its dimensions are `sizes`; `nothing` for another value.
"""
function sameSizedArray(values::IR.Expression, iterators::Vector{Symbol}, sizes::Vector{Int}, ty::IR.ModelicaType)::Union{Nothing, IR.CREF}
  values isa IR.CREF || return nothing
  local parts::Vector{IR.CrefPart} = values.cref.parts
  (any(IR.isMember, parts) || any(p::IR.CrefPart -> !isempty(p.subscripts), @view(parts[1:end-1]))) && return nothing
  local subscripts::Vector{IR.Subscript} = last(parts).subscripts
  length(subscripts) == length(iterators) || return nothing
  for (s, it) in zip(subscripts, iterators)
    (s isa IR.INDEX && s.exp isa IR.CREF && length(s.exp.cref.parts) == 1 && s.exp.cref.name === it) || return nothing
  end
  local whole::IR.CrefPart = IR.unsubscriptedPart(last(parts))
  (whole.ty isa IR.T_ARRAY && Union{Nothing, Int}[IR.dimensionSize(d) for d in whole.ty.dimensions] == sizes) || return nothing
  local wholeParts::Vector{IR.CrefPart} = IR.CrefPart[parts[1:end-1]; whole]
  return IR.CREF(IR.ComponentRef(wholeParts), ty)
end

"""
    templateBody(partition, key) -> Tuple{SCALAR_EQUATION, Vector{Symbol}}

The scalar equation of the arrays-kept partition an element view's equation comes from (`key`: the origin's equation
and path), and the iterators around it; NotHandled for another kind of equation.
"""
function templateBody(partition::IR.Partition, key::Tuple{Int, Vector{Int}})::Tuple{IR.SCALAR_EQUATION, Vector{Symbol}}
  local (i::Int, path::Vector{Int}) = key
  local eq::IR.Equation = partition.equations[i]
  local (body::IR.Equation, iterators::Vector{Symbol}) = isempty(path) ? (eq, Symbol[]) :
    eq isa IR.FOR_EQUATION ? bodyAt(eq, path) : throw(NotHandled("an equation path through a $(nameof(typeof(eq)))"))
  body isa IR.SCALAR_EQUATION || throw(NotHandled("a derivative solved from a $(nameof(typeof(body))) in array form"))
  return (body, iterators)
end

"`eq` (of a deep copy) with its body equation at `path` (in nested for-equations) replaced by `new`."
function replaceBody!(eq::IR.FOR_EQUATION, path::Vector{Int}, new::IR.Equation)::Nothing
  local current::IR.FOR_EQUATION = eq
  for k in path[1:(end - 1)]
    current = current.body[k]::IR.FOR_EQUATION
  end
  current.body[path[end]] = new
  return nothing
end

"""
    solvedTemplate(body, iterators, rows, view, E, model, functions, parameters) -> SolvedTemplate

The body equation `body` solved in array form for the derivative its iterations `rows` (element view row => the
derivative solved from it) are solved for: the reference in the body that is each row's derivative at its iteration
(one for all of them, subscripted by the iterators, each once), the coefficient as the element view's (an expression
where it reads only parameters too: `literalCoefficients = false`; the literal it is at every iteration,
`rowLiteral`), the value `solvedValue`'s. NotHandled where the
rows solve for references the body does not have alike.
"""
function solvedTemplate(body::IR.SCALAR_EQUATION, iterators::Vector{Symbol}, rows::Vector{Pair{Int, Symbol}},
                        view::Scalarized, E::Elements, model::IR.Model, functions::Dict{Symbol, IR.ModelicaFunction},
                        parameters::StartValues)::SolvedTemplate
  local canonical::IR.SCALAR_EQUATION = IR.SCALAR_EQUATION(canonicalReferences(body.lhs, E, model.variables), canonicalReferences(body.rhs, E, model.variables),
                                       body.ty, body.source, body.attributes)
  local valuesAt::Function = (r::Int) -> Dict{Symbol, Int}(zip(iterators, view.origins[r].iteration))
  local isElement::Function = (x::IR.CREF, r::Int, d::Symbol) -> begin
    local scalar::IR.Expression = equationExpression(x, valuesAt(r), E)
    scalar isa IR.CREF && scalar.cref.name === d
  end
  local (r1::Int, d1::Symbol) = rows[1]
  local candidates::Dict{Symbol, IR.CREF} = Dict{Symbol, IR.CREF}()
  for side in (canonical.lhs, canonical.rhs)
    IR.foreachExp(x::IR.Expression -> ((x isa IR.CREF && isElement(x, r1, d1)) && (candidates[x.cref.name] = x); nothing), side)
  end
  length(candidates) == 1 ||
    throw(NotHandled("a body solved for $(d1) through $(length(candidates)) references ($(equationLine(body)))"))
  local cref::IR.CREF = only(values(candidates))
  all(((r, d)::Pair{Int, Symbol},) -> isElement(cref, r, d), rows) ||
    throw(NotHandled("a for-equation whose iterations are solved for different references ($(equationLine(body)))"))
  #= each iterator once, as a subscript of its own: the value is over the iterators, read at any subscripts; a variable
     of its own (an element of an array split for index reduction, `\$DER.x[2]`, as `solvedTarget` reads it) has none =#
  local own::Vector{IR.Subscript} = haskey(model.variables, cref.cref.name) ? IR.Subscript[] : allSubscripts(cref.cref)
  #= a scalar equation's reference to the one element of its array (`\$DER.star.pin.i[1]`, a plug of one phase: the
     zero-sequence star of MSL Machines' IMC_DOL_CommonLeakage) has literal subscripts and no iterator: as a scalar =#
  (isempty(iterators) && all(s::IR.Subscript -> IR.literalIndex(s) !== nothing, own)) && (own = IR.Subscript[])
  (length(own) == length(iterators) &&
   Set(Symbol[s isa IR.INDEX && s.exp isa IR.CREF ? s.exp.cref.name : Symbol("") for s in own]) == Set(iterators)) ||
    throw(NotHandled("a derivative solved through $(cref.cref.name), not subscripted by its iterators $(iterators)"))
  local c::Union{Nothing, IR.Expression} = partialDerivative(canonical, cref.cref.name, functions; derivativeCalls = true)
  c === nothing && throw(NotHandled("the coefficient of $(cref.cref.name) in $(equationLine(body))"))
  return SolvedTemplate(cref, iterators, solvedValue(canonical, cref.cref.name, rowLiteral(c, rows, valuesAt, E)))
end

"""
    rowLiteral(c, rows, valuesAt, E) -> Expression

The coefficient `c` of a body's derivative as the literal it is at every one of its iterations `rows`, where they
all give the same one; else `c`. A product rule's terms on a matrix read by the iterator (FreeMotion's
`w = 2 * (M(Q) * der(Q))` differentiated: `0.0 * der(Q[k])` per element) are zero only once the iterator has a value:
in array form the coefficient was an expression, the value divided by it, and the element view's value (the literal's
reciprocal) was another form.
"""
function rowLiteral(c::IR.Expression, rows::Vector{Pair{Int, Symbol}}, valuesAt::Function, E::Elements)::IR.Expression
  isLiteral(c) && return c
  local literal::Union{Nothing, IR.Expression} = nothing
  for (r, _) in rows
    local atRow::IR.Expression = foldTree(equationExpression(c, valuesAt(r), E))
    isLiteral(atRow) || return c
    (literal === nothing || literalValue(literal) == literalValue(atRow)) || return c
    literal = atRow
  end
  return something(literal, c)
end

"""
    resolvedTemplates(templates, E, small) -> Dict{Symbol, IR.Expression}

Each template's value with the solved derivatives it reads replaced by theirs, in dependency order (`resolvedValues`
in array form: a value read at its reference's subscripts, `instantiated`); those in a cycle, and those that read one,
left out. A value stays a reference where the element view's values are larger than `SUBSTITUTED_VALUE_NODE_LIMIT`
nodes (`small`: the solved derivatives whose values are put in): the template's size counts the iterators'
subscripts, which the elements do not have.
"""
function resolvedTemplates(templates::Dict{Symbol, SolvedTemplate}, E::Elements, small::Set{Symbol})::Dict{Symbol, IR.Expression}
  local reads::Dict{Symbol, Set{Symbol}} = Dict{Symbol, Set{Symbol}}()
  for (target, t) in templates
    local r::Set{Symbol} = Set{Symbol}()
    IR.foreachExp(x::IR.Expression -> (local s::Union{Nothing, Symbol} = solvedTarget(x, E, templates); s === nothing || push!(r, s); nothing),
                  t.value)
    reads[target] = r
  end
  local readers::Dict{Symbol, Vector{Symbol}} = Dict{Symbol, Vector{Symbol}}(t => Symbol[] for t in keys(templates))
  foreach(((t, r)::Pair{Symbol, Set{Symbol}},) -> foreach(x::Symbol -> push!(readers[x], t), r), reads)
  local pending::Dict{Symbol, Int} = Dict{Symbol, Int}(t => length(r) for (t, r) in reads)
  local ready::Vector{Symbol} = sort!(Symbol[t for (t, n) in pending if n == 0])
  local resolved::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}()
  while !isempty(ready)
    local t::Symbol = popfirst!(ready)
    resolved[t] = foldTree(IR.mapExp(templates[t].value) do x::IR.Expression
      local s::Union{Nothing, Symbol} = solvedTarget(x, E, templates)
      (s === nothing || !haskey(resolved, s)) && return x
      return s in small ? instantiated(templates[s], x, resolved[s]) : x
    end)
    for r in readers[t]
      pending[r] -= 1
      pending[r] == 0 && push!(ready, r)
    end
  end
  return resolved
end

"""
    solvedInArrays(model; known) -> Tuple{IR.Model, Vector{Symbol}}

The model reduced in array form (`reduceIndexArrays!`) with its derivative unknowns solved for in array form (the
switch OMBACKEND_SOLVE_DERIVATIVES): a new model (its variables shared, its partition's equations new) and the
derivatives so solved (arrays and scalars). `model` is changed in place where a derivative array is split
(`splitArrays!`: its meaning and its scalarized form unchanged, but a split that throws part-way leaves it half split):
a caller that goes on with `model` passes a copy. The decisions are `solveDerivatives!`'s on the element
view (`derivativeSolution`; `known` as there); each body solved once for the derivative its iterations determine
(`solvedTemplate`), the values resolved (`resolvedTemplates`) and put in. Checked: the result scalarized is the
element view solved. NotHandled where a derivative array's elements are solved from different bodies or only some of
them, a body's iterations for different references, from an element of an array equation, or where the check
fails. A derivative the element view solved with the values of others put into its equation (`solveThroughSolved!`:
a stray inductance in series with a main one, MSL Machines' air gaps) is solved from its body with the others'
values put in (`withTemplateValues`).
"""
function solvedInArrays(model::IR.Model;
                        known::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}())::Tuple{IR.Model, Vector{Symbol}}
  envSwitch("OMBACKEND_SOLVE_DERIVATIVES") || return (model, Symbol[])
  #= the element equations of the equations a split keeps (`ScalarCache`): the views of the split rounds share them;
     the view after the solve is new (the solve puts the derivatives into its view's) =#
  local cache::ScalarCache = ScalarCache()
  local view::Scalarized = scalarizeArrays(model; cache)
  #= the coefficients that read only parameters as expressions: the array form's differ per element =#
  local solution::Union{Nothing, DerivativeSolution} =
    derivativeSolution(view.model, view.model.continuous[1]; known, literalCoefficients = false)
  solution === nothing && return (model, Symbol[])
  #= a derivative array whose elements are solved differently (slicing): split into its elements (`splitArrays!`, in
     place: the model's meaning, and its scalarized form, unchanged), then grouped again. The splits follow the element
     view's matching, which stays (its equations and unknowns do), so its solution is kept and only the array form's
     elements and origins are new (a scalarization); the values put in are the final view's, solved once more (its
     variables' order shapes the resolved expressions, and so which are small). MSL EngineV6 splits along its frames'
     chain, a frame per round (29 rounds): solving the element view each round took 18 s. =#
  local stale::Bool = false
  for _ in 1:64
    try
      stale || return solvedInArraysOnce(model, view, solution)
      #= after a split only the splits asked for (`arrayTemplates`), then the view solved: the stale view's values
         and check were made and dropped (EngineV6: 0.8 s). `view` is the model's view already (made after the last
         split; neither call changes the model or the view): scalarized again, 0.22 s =#
      arrayTemplates(model, view, solution)
      solution = derivativeSolution(view.model, view.model.continuous[1]; known, literalCoefficients = false)
      solution === nothing && return (model, Symbol[])
      stale = false
      return solvedInArraysOnce(model, view, solution)
    catch err
      (err isa NotHandled && needsSplit(err)) || rethrow()
      splitArrays!(model, view, err.arrays, err.equations; cache)
      view = scalarizeArrays(model; cache)
      stale = true
    end
  end
  throw(NotHandled("derivatives solved in array form: more splits than 64"))
end

"""
    unsubstitutedNote(eq, solution, E, templates, small) -> String

For a refusal: the solved derivatives an element equation of the array form still reads, and why their values were
not put in (no template of the array form, its values not small in the element view, or not resolved).
"""
function unsubstitutedNote(eq::IR.Equation, solution::DerivativeSolution, E::Elements, templates::Dict{Symbol, SolvedTemplate},
                           small::Set{Symbol})::String
  local notes::Vector{String} = String[]
  IR.foreachExp(eq) do x::IR.Expression
    (x isa IR.CREF && haskey(solution.resolved, x.cref.name)) || return nothing
    local target::Symbol = haskey(E.array, x.cref.name) ? E.array[x.cref.name][1] : x.cref.name
    push!(notes, "$(x.cref.name): " * (!haskey(templates, target) ? "no template ($(target))" : !(target in small) ? "not small" :
                                       "not resolved, reads " * readChain(target, E, templates)))
    nothing
  end
  return isempty(notes) ? "" : " (still read: " * join(unique(notes), "; ") * ")"
end

"A cycle among the templates' reads reachable from `target` (depth first), or the reads of `target`: for a refusal."
function readChain(target::Symbol, E::Elements, templates::Dict{Symbol, SolvedTemplate})::String
  local readsOf::Function = (t::Symbol) -> begin
    local reads::Vector{Symbol} = Symbol[]
    IR.foreachExp(x::IR.Expression -> (local s::Union{Nothing, Symbol} = solvedTarget(x, E, templates); (s === nothing || s in reads) || push!(reads, s); nothing),
                  templates[t].value)
    reads
  end
  local stack::Vector{Symbol} = Symbol[]
  local done::Set{Symbol} = Set{Symbol}()
  local cycle::Vector{Symbol} = Symbol[]
  local visit::Function = function (t::Symbol)
    t in done && return false
    local at::Union{Nothing, Int} = findfirst(==(t), stack)
    if at !== nothing
      append!(cycle, stack[at:end]); push!(cycle, t)
      return true
    end
    push!(stack, t)
    for r in readsOf(t)
      visit(r) && return true
    end
    pop!(stack); push!(done, t)
    return false
  end
  visit(target) && return "cycle " * join(String.(cycle), " -> ")
  return join(String.(readsOf(target)), ", ")
end

"""
    throughPartners(solution, E, bodyOf) -> Dict{Symbol, Set{Symbol}}

The solved derivatives (arrays or scalars, the keys of `bodyOf`) the element view solved with the values of others
put into their equation (`DerivativeSolution.through`), each with those others (arrays or scalars). NotHandled,
naming the arrays to split, where only some elements of an array are solved so.
"""
function throughPartners(solution::DerivativeSolution, E::Elements,
                         bodyOf::Dict{Symbol, Tuple{Int, Vector{Int}}})::Dict{Symbol, Set{Symbol}}
  local partners::Dict{Symbol, Set{Symbol}} = Dict{Symbol, Set{Symbol}}()
  isempty(solution.through) && return partners
  local mixed::Vector{Symbol} = Symbol[]
  for target in keys(bodyOf)
    local names::Vector{Symbol} = haskey(E.elements, target) ? E.elements[target] : Symbol[target]
    local n::Int = count(d::Symbol -> haskey(solution.through, d), names)
    n == 0 && continue
    if n != length(names)
      push!(mixed, target)
      continue
    end
    local others::Set{Symbol} = Set{Symbol}()
    for d in names, o in solution.through[d]
      push!(others, haskey(E.array, o) ? E.array[o][1] : o)
    end
    partners[target] = others
  end
  #= of a cycle of two the element view solves the one first by name again, per element: an array whose elements
     meet partners of other names (a rotation, MSL Machines' `i_ms = i_ss + RotationMatrix * i_rr`) is split =#
  isempty(mixed) ||
    throw(NotHandled("derivative arrays solved through the values of others for some of their elements " *
                     "($(length(mixed)) arrays: slicing)", sort!(mixed), Int[]))
  return partners
end

"The body `body` with the values of the solved derivatives `others` put in at their references (`instantiated`)."
function withTemplateValues(body::IR.SCALAR_EQUATION, others::Set{Symbol}, E::Elements,
                            templates::Dict{Symbol, SolvedTemplate})::IR.SCALAR_EQUATION
  local insert::Function = (x::IR.Expression) -> begin
    local s::Union{Nothing, Symbol} = solvedTarget(x, E, templates)
    return (s !== nothing && s in others) ? instantiated(templates[s], x, templates[s].value) : x
  end
  return IR.SCALAR_EQUATION(IR.mapExp(insert, body.lhs), IR.mapExp(insert, body.rhs), body.ty, body.source,
                            body.attributes)
end

"""
    arrayTemplates(model, view, solution) -> Tuple{Dict{Symbol, SolvedTemplate}, Dict{Symbol, Tuple{Int, Vector{Int}}}}

The solved derivatives (arrays or scalars) of `model` in array form, of its element view `view` and its derivatives'
solution `solution`: each one's template (`solvedTemplate`) and its body (equation, path). NotHandled, naming the
arrays or for-equations to split, where solved differently: the only place `solvedInArraysOnce` asks for splits.
"""
function arrayTemplates(model::IR.Model, view::Scalarized,
                        solution::DerivativeSolution)::Tuple{Dict{Symbol, SolvedTemplate}, Dict{Symbol, Tuple{Int, Vector{Int}}}}
  local E::Elements = view.elements
  local partition::IR.Partition = model.continuous[1]
  local rowsOf::Dict{Tuple{Int, Vector{Int}}, Vector{Int}} = Dict{Tuple{Int, Vector{Int}}, Vector{Int}}()
  foreach(((r, o)::Tuple{Int, EquationOrigin},) -> push!(get!(() -> Int[], rowsOf, (o.equation, o.path)), r), enumerate(view.origins))
  #= the rows each body's iterations solve, by body; each solved derivative array (or scalar) from one body =#
  local solvedRows::Dict{Tuple{Int, Vector{Int}}, Vector{Pair{Int, Symbol}}} = Dict{Tuple{Int, Vector{Int}}, Vector{Pair{Int, Symbol}}}()
  local bodyOf::Dict{Symbol, Tuple{Int, Vector{Int}}} = Dict{Symbol, Tuple{Int, Vector{Int}}}()
  #= every array and for-equation solved differently (slicing), collected: split together =#
  local splitArrays::Set{Symbol} = Set{Symbol}()
  local splitEquations::Set{Int} = Set{Int}()
  for d in sort!(collect(keys(solution.resolved)))
    local r::Int = solution.equationOf[d]
    local origin::EquationOrigin = view.origins[r]
    isempty(origin.element) || throw(NotHandled("the derivative $(d) solved from an element of an array equation"))
    local key::Tuple{Int, Vector{Int}} = (origin.equation, origin.path)
    push!(get!(() -> Pair{Int, Symbol}[], solvedRows, key), r => d)
    local target::Symbol = haskey(E.array, d) ? E.array[d][1] : d
    get!(bodyOf, target, key) == key || push!(splitArrays, target)
  end
  for (target, key) in bodyOf
    target in splitArrays && continue
    length(solvedRows[key]) == length(rowsOf[key]) || push!(splitEquations, key[1])
    (haskey(E.elements, target) && length(solvedRows[key]) != length(E.elements[target])) && push!(splitArrays, target)
  end
  if !(isempty(splitArrays) && isempty(splitEquations))
    splitClosure!(splitArrays, splitEquations, model, E, bodyOf)
    throw(NotHandled("derivative arrays solved element by element differently ($(length(splitArrays)) arrays, " *
                     "$(length(splitEquations)) for-equations: slicing)", sort!(collect(splitArrays)), sort!(collect(splitEquations))))
  end
  local functions::Dict{Symbol, IR.ModelicaFunction} = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in model.functions)
  local parameters::StartValues = StartValues(model.variables, model.functions; parametersOnly = true)
  local templates::Dict{Symbol, SolvedTemplate} = Dict{Symbol, SolvedTemplate}()
  local partners::Dict{Symbol, Set{Symbol}} = throughPartners(solution, E, bodyOf)
  for (target, key) in bodyOf
    haskey(partners, target) && continue
    local rows::Vector{Pair{Int, Symbol}} = solvedRows[key]
    local (body::IR.SCALAR_EQUATION, iterators::Vector{Symbol}) = templateBody(partition, key)
    #= a body its iterations solve for different references, or not in array form: unrolled (slicing) =#
    try
      templates[target] = solvedTemplate(body, iterators, rows, view, E, model, functions, parameters)
    catch err
      (err isa NotHandled && !isempty(iterators)) || rethrow()
      push!(splitEquations, key[1])
    end
  end
  #= then those solved with the values of others put into their body (`solveThroughSolved!`): the body with the
     others' templates' values put in, as the element view puts their matched values in, solved the same way =#
  for target in sort!(collect(keys(partners)))
    local key::Tuple{Int, Vector{Int}} = bodyOf[target]
    key[1] in splitEquations && continue
    local others::Set{Symbol} = partners[target]
    all(o::Symbol -> haskey(templates, o), others) ||
      throw(NotHandled("derivatives solved through the values of others that have no value in array form ($(target))"))
    local (body::IR.SCALAR_EQUATION, iterators::Vector{Symbol}) = templateBody(partition, key)
    local substituted::IR.SCALAR_EQUATION = withTemplateValues(body, others, E, templates)
    try
      templates[target] = solvedTemplate(substituted, iterators, solvedRows[key], view, E, model, functions, parameters)
    catch err
      (err isa NotHandled && !isempty(iterators)) || rethrow()
      push!(splitEquations, key[1])
    end
  end
  if !isempty(splitEquations)
    local more::Set{Symbol} = Set{Symbol}()
    splitClosure!(more, splitEquations, model, E, bodyOf)
    throw(NotHandled("for-equations whose iterations are solved differently ($(length(splitEquations)): slicing)",
                     sort!(collect(more)), sort!(collect(splitEquations))))
  end
  return (templates, bodyOf)
end

"""
    splitClosure!(arrays, equations, model, E, bodyOf)

The splits that splitting `arrays` and unrolling `equations` (`splitArrays!`) asks for next, added to `arrays`: an
equation that reads a split array is unrolled, so a derivative array of more than one element solved from it has its
elements from different equations, and is split in the next round; its readers are unrolled in turn. MSL EngineV6
split the angular velocities along its frames' chain, a frame per round (rounds 3 to 7 of 7, 0.1 s each). Not where
an array is split along its inner dimensions only (`innerSplittable`: its slices are new arrays of the next view),
nor for such an array found: the next round asks for those.
"""
function splitClosure!(arrays::Set{Symbol}, equations::Set{Int}, model::IR.Model, E::Elements,
                       bodyOf::Dict{Symbol, Tuple{Int, Vector{Int}}})::Nothing
  local variables::IR.VariableSet = model.variables
  local bases::Set{Symbol} = Set{Symbol}(familyBase(a, variables) for a in arrays)
  any(b::Symbol -> innerSplittable(model, b), bases) && return nothing
  #= the element view's arrays by their family; the equations reading each name (`readsArrays`), a whole record or
     array apart (the containers of a split array) =#
  local members::Dict{Symbol, Vector{Symbol}} = Dict{Symbol, Vector{Symbol}}()
  for n in keys(E.elements)
    push!(get!(() -> Symbol[], members, familyBase(n, variables)), n)
  end
  local readers::Dict{Symbol, Vector{Int}} = Dict{Symbol, Vector{Int}}()
  local wholeReaders::Dict{Symbol, Vector{Int}} = Dict{Symbol, Vector{Int}}()
  for (i, eq) in enumerate(model.continuous[1].equations)
    IR.foreachExp(eq) do x::IR.Expression
      x isa IR.CREF || return nothing
      local name::Symbol = IR.hasSubscripts(x.cref) ? unsubscriptedName(x.cref) : x.cref.name
      push!(get!(() -> Int[], readers, name), i)
      x.ty isa Union{IR.T_RECORD, IR.T_ARRAY} && push!(get!(() -> Int[], wholeReaders, name), i)
      nothing
    end
  end
  #= the derivative arrays of more than one element, by the equation they are solved from =#
  local solvedFrom::Dict{Int, Vector{Symbol}} = Dict{Int, Vector{Symbol}}()
  for (target, key) in bodyOf
    (haskey(E.elements, target) && length(E.elements[target]) > 1) && push!(get!(() -> Symbol[], solvedFrom, key[1]), target)
  end
  local unrolled::Set{Int} = copy(equations)
  local newEquations::Vector{Int} = collect(equations)
  local queue::Vector{Symbol} = collect(bases)
  while !(isempty(queue) && isempty(newEquations))
    for b in queue, n in get(members, b, Symbol[])
      local reading::Vector{Int} = copy(get(readers, n, Int[]))
      foreach(c::Symbol -> append!(reading, get(wholeReaders, c, Int[])), arrayContainers(Set{Symbol}((n,))))
      for i in reading
        i in unrolled && continue
        push!(unrolled, i); push!(newEquations, i)
      end
    end
    empty!(queue)
    for i in newEquations, target in get(solvedFrom, i, Symbol[])
      local b::Symbol = familyBase(target, variables)
      (b in bases || innerSplittable(model, b)) && continue
      push!(bases, b); push!(arrays, target); push!(queue, b)
    end
    empty!(newEquations)
  end
  return nothing
end

"""
    solvedInArraysOnce(model, view, solution) -> Tuple{IR.Model, Vector{Symbol}}

`solvedInArrays` without splitting, of the model's element view `view` and its derivatives' solution `solution`:
NotHandled, naming the arrays or for-equations to split, where solved differently (`arrayTemplates`).
"""
function solvedInArraysOnce(model::IR.Model, view::Scalarized, solution::DerivativeSolution)::Tuple{IR.Model, Vector{Symbol}}
  local elementPartition::IR.Partition = view.model.continuous[1]
  local E::Elements = view.elements
  local partition::IR.Partition = model.continuous[1]
  local (templates::Dict{Symbol, SolvedTemplate}, bodyOf::Dict{Symbol, Tuple{Int, Vector{Int}}}) = arrayTemplates(model, view, solution)
  #= a solved derivative put in where the element view puts its elements' values in (all of them small) =#
  local small::Set{Symbol} = Set{Symbol}()
  for target in keys(templates)
    local elementNames::Vector{Symbol} = haskey(E.elements, target) ? E.elements[target] : Symbol[target]
    all(d::Symbol -> haskey(solution.resolved, d) && !expressionLargerThan(solution.resolved[d], SUBSTITUTED_VALUE_NODE_LIMIT), elementNames) &&
      push!(small, target)
  end
  local resolved::Dict{Symbol, IR.Expression} = resolvedTemplates(templates, E, small)
  #= the values put in (a copy: the model with its arrays stays the scalarized path's), the bodies their definitions =#
  local equations::Vector{IR.Equation} = IR.Equation[IR.copyEquation(eq) for eq in partition.equations]
  local substitute::Function = (x::IR.Expression) -> begin
    local s::Union{Nothing, Symbol} = solvedTarget(x, E, templates)
    (s === nothing || !haskey(resolved, s)) && return x
    return s in small ? instantiated(templates[s], x, resolved[s]) : x
  end
  for (i, eq) in enumerate(equations)
    equations[i] = IR.mapEquation!(substitute, eq)
  end
  for (target, value) in resolved
    local (i::Int, path::Vector{Int}) = bodyOf[target]
    local old::IR.SCALAR_EQUATION = templateBody(partition, (i, path))[1]
    local definition::IR.SCALAR_EQUATION = IR.SCALAR_EQUATION(templates[target].cref, value, IR.T_REAL(), old.source, old.attributes)
    isempty(path) ? (equations[i] = definition) : replaceBody!(equations[i]::IR.FOR_EQUATION, path, definition)
  end
  local solvedPartition::IR.Partition = IR.Partition(partition.index, partition.association, partition.unknowns, equations,
                                       partition.initialEquations)
  local solved::IR.Model = IR.Model(model.name, model.variables, IR.Partition[solvedPartition], model.clocked, model.baseClocks,
                          model.functions, model.initialMode, model.modes, model.transitions, model.frontendModel)
  #= the check: the array form's elements are the element view's solved equations =#
  substituteDerivatives!(elementPartition, solution, view.model.variables)
  local mine::Vector{IR.Equation} = scalarizeArrays(solved).model.continuous[1].equations
  length(mine) == length(elementPartition.equations) ||
    throw(NotHandled("derivatives solved in array form: $(length(mine)) element equations, the element view $(length(elementPartition.equations))"))
  #= scalar and if-equations folded (`sameEquation`: a whole array's value written out per element, `{0, 0, 0}[1] * r`) =#
  for (a, b) in zip(mine, elementPartition.equations)
    (a isa Union{IR.SCALAR_EQUATION, IR.IF_EQUATION} ? sameEquation(a, b) : string(a) == string(b)) ||
      throw(NotHandled("derivatives solved in array form: $(equationLine(a)) for the element view's $(equationLine(b))" *
                       unsubstitutedNote(a, solution, E, templates, small)))
  end
  return (solved, sort!(collect(keys(resolved))))
end
