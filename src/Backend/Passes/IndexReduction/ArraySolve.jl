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
   for-equation as the plan is (ArrayCommit.jl). A body whose iterations determine the elements of a derivative array
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
any subscripts), a solved scalar by its name; `nothing` for another.
"""
function solvedTarget(x::IR.Expression, E::Elements, templates::Dict{Symbol, SolvedTemplate})::Union{Nothing, Symbol}
  x isa IR.CREF || return nothing
  #= a scalar of its own first (an element of an array split for index reduction, `x[2]`) =#
  (haskey(templates, x.cref.name) && !haskey(E.sizes, x.cref.name)) && return x.cref.name
  IR.hasSubscripts(x.cref) || return nothing
  local array::Symbol = unsubscriptedName(x.cref)
  return (haskey(E.sizes, array) && haskey(templates, array)) ? array : nothing
end

"""
    instantiated(t, x, values) -> Expression

The value `values` (over the iterators of the template `t`) of the solved derivative at the reference `x`: its
iterators by `x`'s subscripts (`\$DER.x[\$i1]`'s value at `\$DER.x[j + 1]` with `\$i1` as `j + 1`); a scalar's as
it is. NotHandled for a reference by a slice.
"""
function instantiated(t::SolvedTemplate, x::IR.CREF, values::IR.Expression)::IR.Expression
  isempty(t.iterators) && return values
  local subscripts::Vector{IR.Subscript} = allSubscripts(x.cref)
  local own::Vector{IR.Subscript} = allSubscripts(t.cref.cref)
  (length(subscripts) == length(own) && all(s::IR.Subscript -> s isa IR.INDEX, subscripts)) ||
    throw(NotHandled("a solved derivative read by a slice or partly subscripted ($(x.cref.name))"))
  local map::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}()
  for (s, o) in zip(subscripts, own)
    map[(o::IR.INDEX).exp.cref.name] = (s::IR.INDEX).exp
  end
  return withIteratorExpressions(values, map)
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
where it reads only parameters too: `literalCoefficients = false`), the value `solvedValue`'s. NotHandled where the
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
  #= each iterator once, as a subscript of its own: the value is over the iterators, read at any subscripts =#
  local own::Vector{IR.Subscript} = allSubscripts(cref.cref)
  (length(own) == length(iterators) &&
   Set(Symbol[s isa IR.INDEX && s.exp isa IR.CREF ? s.exp.cref.name : Symbol("") for s in own]) == Set(iterators)) ||
    throw(NotHandled("a derivative solved through $(cref.cref.name), not subscripted by its iterators $(iterators)"))
  local c::Union{Nothing, IR.Expression} = partialDerivative(canonical, cref.cref.name, functions; derivativeCalls = true)
  c === nothing && throw(NotHandled("the coefficient of $(cref.cref.name) in $(equationLine(body))"))
  return SolvedTemplate(cref, iterators, solvedValue(canonical, cref.cref.name, c))
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
them, a body's iterations for different references, through others' values (`solveThroughSolved!`), from an element
of an array equation, or where the check fails.
"""
function solvedInArrays(model::IR.Model;
                        known::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}())::Tuple{IR.Model, Vector{Symbol}}
  envSwitch("OMBACKEND_SOLVE_DERIVATIVES") || return (model, Symbol[])
  local view::Scalarized = scalarizeArrays(model)
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
      local result::Tuple{IR.Model, Vector{Symbol}} = solvedInArraysOnce(model, view, solution)
      stale || return result
      #= a new view: the check put the derivatives into this one's (the model is not changed) =#
      view = scalarizeArrays(model)
      solution = derivativeSolution(view.model, view.model.continuous[1]; known, literalCoefficients = false)
      solution === nothing && return (model, Symbol[])
      stale = false
      return solvedInArraysOnce(model, view, solution)
    catch err
      (err isa NotHandled && needsSplit(err)) || rethrow()
      splitArrays!(model, view, err.arrays, err.equations)
      view = scalarizeArrays(model)
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
    solvedInArraysOnce(model, view, solution) -> Tuple{IR.Model, Vector{Symbol}}

`solvedInArrays` without splitting, of the model's element view `view` and its derivatives' solution `solution`:
NotHandled, naming the arrays or for-equations to split, where solved differently.
"""
function solvedInArraysOnce(model::IR.Model, view::Scalarized, solution::DerivativeSolution)::Tuple{IR.Model, Vector{Symbol}}
  local elementPartition::IR.Partition = view.model.continuous[1]
  isempty(solution.throughSolved) ||
    throw(NotHandled("derivatives solved through the values of others ($(first(sort!(collect(solution.throughSolved)))))"))
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
  (isempty(splitArrays) && isempty(splitEquations)) ||
    throw(NotHandled("derivative arrays solved element by element differently ($(length(splitArrays)) arrays, " *
                     "$(length(splitEquations)) for-equations: slicing)", sort!(collect(splitArrays)), sort!(collect(splitEquations))))
  local functions::Dict{Symbol, IR.ModelicaFunction} = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in model.functions)
  local parameters::StartValues = StartValues(model.variables, model.functions; parametersOnly = true)
  local templates::Dict{Symbol, SolvedTemplate} = Dict{Symbol, SolvedTemplate}()
  for (target, key) in bodyOf
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
  isempty(splitEquations) ||
    throw(NotHandled("for-equations whose iterations are solved differently ($(length(splitEquations)): slicing)", Symbol[],
                     sort!(collect(splitEquations))))
  #= a solved derivative put in where the element view puts its elements' values in (all of them small) =#
  local small::Set{Symbol} = Set{Symbol}()
  for target in keys(templates)
    local elementNames::Vector{Symbol} = haskey(E.elements, target) ? E.elements[target] : Symbol[target]
    all(d::Symbol -> haskey(solution.resolved, d) && !expressionLargerThan(solution.resolved[d], SUBSTITUTED_VALUE_NODE_LIMIT), elementNames) &&
      push!(small, target)
  end
  local resolved::Dict{Symbol, IR.Expression} = resolvedTemplates(templates, E, small)
  #= the values put in (a copy: the model with its arrays stays the scalarized path's), the bodies their definitions =#
  local equations::Vector{IR.Equation} = deepcopy(partition.equations)
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
  substituteDerivatives!(elementPartition, solution)
  local mine::Vector{IR.Equation} = scalarizeArrays(solved).model.continuous[1].equations
  length(mine) == length(elementPartition.equations) ||
    throw(NotHandled("derivatives solved in array form: $(length(mine)) element equations, the element view $(length(elementPartition.equations))"))
  for (a, b) in zip(mine, elementPartition.equations)
    (a isa IR.SCALAR_EQUATION ? sameEquation(a, b) : string(a) == string(b)) ||
      throw(NotHandled("derivatives solved in array form: $(equationLine(a)) for the element view's $(equationLine(b))" *
                       unsubstitutedNote(a, solution, E, templates, small)))
  end
  return (solved, sort!(collect(keys(resolved))))
end
