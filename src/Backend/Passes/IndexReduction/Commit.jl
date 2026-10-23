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

#= The plan of index reduction, made without changing the model, and its commit. =#

"""
    Plan(structure, differentiated, dummies, sets, known)

What index reduction changes in a partition: the structure Pantelides' algorithm grew (its new
derivative variables), the differentiated equations (by the index of the equation they are
derivatives of), the dummy derivatives (the dynamic sets' candidates among them), and the dynamic
state sets; the simulation code's values it was made with (`KNOWN_VALUES`; empty for a plan as
written), which the derivatives' solve reads too (`solveDerivatives!`).
"""
struct Plan
  structure::Structure
  differentiated::Dict{Int, Vector{IR.Equation}}
  dummies::Set{Int}
  sets::Vector{DynamicSet}
  known::Dict{Symbol, IR.Expression}
end

"""
    KNOWN_VALUES

The values the simulation code's passes find for the model's variables (OMBackend sets it once
the simulation code is loaded: `knownValues(model)`, SimCode's folding of the model without index
reduction, a record function's constant outputs among them: MSL Spice3's `capbx`, which takes a
capacitance's der() out of the system). `nothing` when not set.
"""
const KNOWN_VALUES = Ref{Union{Nothing, Function}}(nothing)

"""
    plan(model, partition) -> Union{Nothing, Plan}

The index reduction of `partition`, without changing the model; `nothing` where the partition
is of index 1 (nothing to differentiate). Throws NotHandled where a step does not handle it.
Made on the partition with the values known before the simulation as literals
(`withKnownValues`), the structure the code generators see, der() of an expression expanded
(`expandDerivatives!`).
"""
function plan(model::IR.Model, partition::IR.Partition;
              known::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}(),
              trace::Union{Nothing, Vector{BlockTrace}} = nothing)::Union{Nothing, Plan}
  any(v -> v.ty isa IR.T_ARRAY && !(v.kind isa Union{IR.PARAMETER, IR.CONSTANT}), model.variables) &&
    throw(NotHandled("array variables (the arrays-kept path; scalarized models only)"))
  #= the structure the code generators see: the known values substituted =#
  local folded = withKnownValues(model, partition; known)
  local functions = Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in model.functions)
  local derived = expandDerivatives!(folded.equations, continuousNames(model.variables, folded), functions)
  local s = Structure(model.variables, folded, model.functions; derived)
  local nRows = length(s.rows)
  nRows == unknownCount(s) ||
    throw(NotHandled("a continuous system of $(nRows) rows over $(unknownCount(s)) unknowns (not square as the structure reads it)"))
  local preferred = artificialStates!(s, folded, model.variables; prefer = envSwitch("OMBACKEND_PREFER_STATES"))
  preferred == 0 && return planStructure(s, nRows, folded, model; trace, known)
  try
    return planStructure(s, nRows, folded, model; trace, known)
  catch e
    e isa NotHandled || rethrow()
    #= the artificial states of StateSelect.prefer given up where their equations do not differentiate, as
       OpenModelica's sanityCheckArtificialStates (all of them here) =#
    local t = Structure(model.variables, folded, model.functions; derived)
    artificialStates!(t, folded, model.variables; prefer = false)
    return planStructure(t, nRows, folded, model; trace, known)
  end
end

"""
    planStructure(s, nRows, partition, model; trace, known) -> Union{Nothing, Plan}

The plan of `plan` from the structure `s` of `partition` (its first `nRows` rows the partition's, its artificial
states' derivatives added): Pantelides' algorithm, the differentiated equations, the dummy derivatives; `nothing`
where nothing is differentiated. `known`: the simulation code's values the plan is made with.
"""
function planStructure(s::Structure, nRows::Int, partition::IR.Partition, model::IR.Model;
                       trace::Union{Nothing, Vector{BlockTrace}} = nothing,
                       known::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}())::Union{Nothing, Plan}
  pantelides!(s)
  length(s.rows) == nRows && return nothing
  local differentiated = differentiatedEquations(s, partition, model.functions)
  local sets = DynamicSet[]
  trace === nothing || empty!(trace)
  local dummies = dummyDerivatives(s, partition, differentiated, model.variables, model.functions; sets, trace)
  foreach(set -> union!(dummies, set.candidates, set.linked...), sets)
  return Plan(s, differentiated, dummies, sets, known)
end

"The names of the continuous variables of `partition` (`isContinuousVariable`)."
function continuousNames(variables::IR.VariableSet, partition::IR.Partition)::Set{Symbol}
  local discretes = discreteTargets(partition)
  return Set{Symbol}(v.name for v in variables if isContinuousVariable(v, discretes))
end

"""
    commit!(model, partition, plan)

The plan applied: der() of an expression expanded in the partition's equations as the plan read
them (`expandDerivatives!`); the new derivative variables added to the model (kind by their role); each
continuous variable's kind by its derivative chain: a state where its derivative is not a dummy
(STATE, its derivative STATE_DER, a state too where its own derivative is no dummy: `der(x)` a
state of a second-order chain), else algebraic (DUMMY_STATE, its derivative DUMMY_DER); the
differentiated equations appended to the partition, each equation linked to its derivative
(`attributes.derivative`).
"""
function commit!(model::IR.Model, partition::IR.Partition, plan::Plan)::Nothing
  local s = plan.structure
  local D = plan.dummies
  #= der() of an expression by the chain rule, as the plan read it: a variable whose derivative the plan's folded
     equations do not read (no derivative variable in the structure: MSL Fourbar_analytic's rod1.v_CM_0, known 0) has
     the derivative 0 =#
  expandDerivatives!(partition.equations,
                     Set{Symbol}(n for n in continuousNames(model.variables, partition) if haskey(s.index, n) && s.derivative[s.index[n]] != 0),
                     Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in model.functions))
  for name in s.variables
    haskey(model.variables, name) && continue
    push!(model.variables, IR.Variable(name, IR.T_REAL(), IR.ALGEBRAIC(), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                                       false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
  end
  for v in eachindex(s.variables)
    local kind = plannedKind(s, D, v)
    kind === nothing || (model.variables[s.variables[v]].kind = kind)
  end
  for (eq, chain) in sort(collect(plan.differentiated); by = first)
    local previous = partition.equations[eq]
    for derivative in chain
      setDerivative!(previous, derivative)
      push!(partition.equations, derivative)
      previous = derivative
    end
  end
  for v in selectedStates(s, D)
    linkStateDerivative!(model, partition, s.variables[v], s.variables[s.derivative[v]])
  end
  foreach(((k, set),) -> dynamicSet!(model, partition, s, set, k), enumerate(plan.sets))
  return nothing
end

"""
    plannedKind(s, dummies, v) -> Union{Nothing, VariableKind}

The kind the plan gives the structure variable `v`: a derivative is a dummy (DUMMY_DER) where it is one, else a
state derivative where it is the highest or its own derivative is a dummy; a variable with a derivative a dummy
state (DUMMY_STATE) where that derivative is a dummy, else a state; `nothing` for a variable without derivative
links (its kind unchanged).
"""
function plannedKind(s::Structure, dummies::Set{Int}, v::Int)::Union{Nothing, IR.VariableKind}
  local a = s.antiderivative[v]
  local d = s.derivative[v]
  if a != 0 && (v in dummies || d == 0 || d in dummies)
    return v in dummies ? IR.DUMMY_DER(s.variables[a]) : IR.STATE_DER(s.variables[a])
  elseif d != 0
    return d in dummies ? IR.DUMMY_STATE(s.variables[d]) : IR.STATE(1, s.variables[d], true)
  end
  return nothing
end

"The variable holding a selected state's derivative (`linkStateDerivative!`): `\$STATEDER.x`."
stateDerivativeName(state::Symbol)::Symbol = Symbol("\$STATEDER.", state)

"""
    dynamicSet!(model, partition, s, set, k)

The dynamic state set `set` as the `k`-th state set of the model (`\$STATESETk`, OpenModelica's
form): its candidates are dummy derivatives (commit!), their variables `V` dummy states; the
set's states `\$STATESETk.x[j]`, one less than the candidates, select from them by the discrete
matrix `\$STATESETk.A[j,i]` (0 or 1, the start values' choice first): `x[j] = sum(A[j,i] * V[i])`,
`der(x[j]) = sum(A[j,i] * der(V[i]))`, linear in `V` whatever the selection. Each linked level
(`set.linked`, the velocities of a pendulum's positions) selects by the same matrix, its states
`\$STATESETk.dx[j]` (`ddx` the next). The dummy is the candidate no state selects; a
when-equation makes candidate `d` the dummy where its pivot (`jacobian`, one row) grows past
twice the dummy's: `A` set to the selection of the others, the states of every level
reinitialized to it.
"""
function dynamicSet!(model::IR.Model, partition::IR.Partition, s::Structure, set::DynamicSet, k::Int)::Nothing
  local (variables, equations) = dynamicSetParts(model, s, set, k)
  foreach(v -> push!(model.variables, v), variables)
  append!(partition.equations, equations)
  return nothing
end

"""
    dynamicSetParts(model, s, set, k) -> Tuple{Vector{IR.Variable}, Vector{IR.Equation}}

The variables and equations of the dynamic state set `set` as the `k`-th state set of the model (`dynamicSet!`),
the model not changed.
"""
function dynamicSetParts(model::IR.Model, s::Structure, set::DynamicSet, k::Int)::Tuple{Vector{IR.Variable}, Vector{IR.Equation}}
  size(set.jacobian, 1) == 1 || throw(NotHandled("a dynamic state set of $(size(set.jacobian, 1)) rows"))
  local variables = IR.Variable[]
  local equations = IR.Equation[]
  local n = length(set.candidates)
  #= a variable of the model by its own reference (a matrix element as SimCode names it), a new one by its name =#
  local realRef = name::Symbol -> IR.CREF(haskey(model.variables, name) ? model.variables[name].cref : IR.ComponentRef(name, IR.T_REAL()), IR.T_REAL())
  local integerRef = name::Symbol -> IR.CREF(IR.ComponentRef(name, IR.T_INTEGER()), IR.T_INTEGER())
  local A = Symbol[Symbol("\$STATESET$(k).A[$(j),$(i)]") for j in 1:n-1, i in 1:n]
  #= the candidate the j-th state selects where candidate d is the dummy =#
  local index = (d::Int, j::Int) -> j < d ? j : j + 1
  local start = findfirst(==(only(set.chosen)), set.candidates)
  for j in 1:n-1, i in 1:n
    push!(variables, IR.Variable(A[j, i], IR.T_INTEGER(), IR.DISCRETE(), nothing,
                                 IR.VariableAttributes(IR.INTEGER(i == index(start, j) ? 1 : 0), IR.BOOLEAN(true), nothing,
                                                       nothing, nothing, nothing, nothing, nothing, nothing),
                                 IR.NO_DIRECTION, true, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
  end
  local selection = (j::Int, names::Vector{Symbol}) -> foldl(plus, IR.Expression[times(integerRef(A[j, i]), realRef(names[i])) for i in 1:n])
  #= each level: its variables V and their states x =#
  local levels = Tuple{Vector{Symbol}, Vector{Symbol}}[]
  for (l, candidates) in enumerate(vcat([set.candidates], set.linked))
    local V = Symbol[s.variables[s.antiderivative[c]] for c in candidates]
    local D = Symbol[s.variables[c] for c in candidates]
    local x = Symbol[setStateName(k, l, j) for j in 1:n-1]
    push!(levels, (V, x))
    for j in 1:n-1
      local derivative = Symbol("\$DER.", x[j])
      local var = get(model.variables, V[index(start, j)], nothing)
      local guess = var === nothing ? nothing : var.attributes.start
      push!(variables, IR.Variable(x[j], IR.T_REAL(), IR.STATE(1, derivative, true), nothing,
                                   IR.VariableAttributes(guess, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing),
                                   IR.NO_DIRECTION, true, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
      push!(variables, IR.Variable(derivative, IR.T_REAL(), IR.STATE_DER(x[j]), nothing, IR.NO_ATTRIBUTES,
                                   IR.NO_DIRECTION, true, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
      push!(equations, IR.SCALAR_EQUATION(realRef(x[j]), selection(j, V), IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
      push!(equations, IR.SCALAR_EQUATION(realRef(derivative), selection(j, D), IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
    end
  end
  #= the dummy's pivot: each candidate's weighted by 1 - sum(A[:, i]), 1 for the candidate no state selects =#
  local magnitude = IR.Expression[builtin(:abs, set.jacobian[1, i]) for i in 1:n]
  local unselected = i::Int -> foldl(minus, IR.Expression[integerRef(A[j, i]) for j in 1:n-1]; init = ONE)
  local pivot = foldl(plus, IR.Expression[times(unselected(i), magnitude[i]) for i in 1:n])
  local branches = IR.WhenBranch[]
  for d in 1:n
    local trigger = IR.RELATION(magnitude[d], IR.GREATER, times(IR.REAL(2.0), pivot), 0)
    local body = IR.Statement[IR.ASSIGNMENT(integerRef(A[j, i]), IR.INTEGER(i == index(d, j) ? 1 : 0), IR.NO_SOURCE)
                              for j in 1:n-1 for i in 1:n]
    for (V, x) in levels, j in 1:n-1
      push!(body, IR.REINIT(IR.ComponentRef(x[j], IR.T_REAL()), realRef(V[index(d, j)]), IR.NO_SOURCE))
    end
    push!(branches, IR.WhenBranch(IR.Expression[trigger], false, false, body))
  end
  push!(equations, IR.WHEN_EQUATION(branches, IR.REGULAR_WHEN, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
  return (variables, equations)
end

"The `j`-th state of the `l`-th level of the `k`-th dynamic state set: `\$STATESETk.x[j]`, `\$STATESETk.dx[j]` (its derivative level), ..."
setStateName(k::Int, l::Int, j::Int)::Symbol = Symbol("\$STATESET$(k).", "d"^(l - 1), "x[$(j)]")

"""
    linkStateDerivative!(model, partition, state, derivative)

The state's derivative as an algebraic unknown: `derivative` (`\$DER.x`, read by the
differentiated equations) becomes algebraic, a new variable `\$STATEDER.x` the state's
derivative, linked by the equation `\$STATEDER.x = \$DER.x`, so `der(x)` occurs in this one
equation (the code generators solve an equation reading one derivative for it, and the
differentiated equations read `der(x)` too: MTK's own index reduction writes its dummy
derivatives `xˍt` the same way). A `derivative` that is a state itself (a second-order chain)
stays one: `der(x) = \$DER.x` is then the order reduction.
"""
function linkStateDerivative!(model::IR.Model, partition::IR.Partition, state::Symbol, derivative::Symbol)::Nothing
  local link = stateDerivativeName(state)
  push!(model.variables, IR.Variable(link, IR.T_REAL(), IR.STATE_DER(state), nothing, IR.NO_ATTRIBUTES, IR.NO_DIRECTION,
                                     false, IR.NOT_CONNECTOR, IR.NO_SOURCE, :indexReduction))
  model.variables[state].kind = IR.STATE(1, link, true)
  model.variables[derivative].kind isa IR.STATE || (model.variables[derivative].kind = IR.ALGEBRAIC())
  local ref = name -> IR.CREF(model.variables[name].cref, IR.T_REAL())
  push!(partition.equations, IR.SCALAR_EQUATION(ref(link), ref(derivative), IR.T_REAL(), IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES))
  return nothing
end

"`eq` linked to its derivative `derivative` (its attributes' `derivative`)."
function setDerivative!(eq::IR.Equation, derivative::IR.Equation)::Nothing
  hasfield(typeof(eq), :attributes) || return nothing
  local a = eq.attributes
  eq.attributes = IR.EquationAttributes(a.kind, a.differentiated, derivative)
  return nothing
end

"""
    knownPlan(model, partition; trace) -> Union{Nothing, Plan}

`plan` of the partition, and where reduced again with the values the simulation code's passes
fold (they can take a der() out: `KNOWN_VALUES`).
"""
function knownPlan(model::IR.Model, partition::IR.Partition;
                   trace::Union{Nothing, Vector{BlockTrace}} = nothing)::Union{Nothing, Plan}
  local firstPlan = plan(model, partition; trace)
  local known = (firstPlan === nothing || KNOWN_VALUES[] === nothing) ? Dict{Symbol, IR.Expression}() : KNOWN_VALUES[](model)
  return isempty(known) ? firstPlan : plan(model, partition; known, trace)
end

"""
    aliasPlan(model, partition, written; trace) -> Tuple{Union{Nothing, Plan}, Union{Nothing, AliasUndo}}

The plan of a partition that needs index reduction (`written`, its first plan as written): with
OMBACKEND_ALIAS_SUBSTITUTION planned again with the aliases read through their representatives
(`substituteAliases!`), which the model keeps; where that leaves nothing to reduce (the aliases
were the constraint: a capacitor loop's `v1 = v2`) or is not handled, the model as written and
its plan (`knownPlan`; `nothing` where the simulation code's values leave nothing to reduce).
"""
function aliasPlan(model::IR.Model, partition::IR.Partition, written::Plan;
                   trace::Union{Nothing, Vector{BlockTrace}} = nothing)::Tuple{Union{Nothing, Plan}, Union{Nothing, AliasUndo}}
  if envSwitch("OMBACKEND_ALIAS_SUBSTITUTION")
    local aliases = substituteAliases!(model, partition)
    if aliases.substituted > 0
      local p = try
        knownPlan(model, partition; trace)
      catch err
        err isa NotHandled || rethrow()
        nothing
      end
      p === nothing || return (p, aliases)
    end
    undoAliases!(model, partition, aliases)
  end
  local known = KNOWN_VALUES[] === nothing ? Dict{Symbol, IR.Expression}() : KNOWN_VALUES[](model)
  return (isempty(known) ? written : plan(model, partition; known, trace), nothing)
end

"""
    reduceIndex!(model) -> Model

Index reduction and state selection of the model's continuous partition (`OMBACKEND_INDEX_REDUCTION`,
on by default): the linear constraints the structure does not show written in first
(`exposeLinearConstraints!`, which keeps the equations' solutions); planned on the model as written; where reduced, again with the aliases read
through their representatives (`aliasPlan`, switch `OMBACKEND_ALIAS_SUBSTITUTION`), then
committed. A model of index 1 is left as it is. Where the plan does not handle the model (a
model with structural modes, a construct `NotHandled` names), the model is left as it is, MTK's
structural_simplify reduces its index, and the reason is logged.
"""
function reduceIndex!(model::IR.Model)::IR.Model
  if !isempty(model.modes) || length(model.continuous) != 1
    @info "[index reduction] $(model.name): a model with structural modes or several continuous partitions; MTK reduces the index"
    return model
  end
  local partition = model.continuous[1]
  local exposed = exposeLinearConstraints!(model, partition)
  local (p, aliases) = try
    local written = plan(model, partition)
    written === nothing ? (nothing, nothing) : aliasPlan(model, partition, written)
  catch err
    err isa NotHandled || rethrow()
    @info "[index reduction] $(model.name): $(sprint(showerror, err)); MTK reduces the index"
    return model
  end
  p === nothing && return model
  commit!(model, partition, p)
  local solved = envSwitch("OMBACKEND_SOLVE_DERIVATIVES") ? solveDerivatives!(model, partition; known = p.known) : Symbol[]
  local sets = isempty(p.sets) ? "" : "; dynamic state sets $(join(("{" * join(String.(p.structure.variables[[p.structure.antiderivative[c] for c in vcat(set.candidates, set.linked...)]]), ", ") * "}" for set in p.sets), ", "))"
  local aliasText = (aliases === nothing || aliases.substituted == 0) ? "" : "$(aliases.substituted) aliases substituted; "
  local exposedText = isempty(exposed) ? "" : "$(length(exposed)) hidden linear constraints exposed; "
  @info "[index reduction] $(model.name): $(exposedText)$(aliasText)$(length(p.dummies)) dummy derivatives, states $(join(first(String.(p.structure.variables[selectedStates(p.structure, p.dummies)]), 8), ", "))$(sets)$(isempty(solved) ? "" : "; $(length(solved)) derivatives solved for")"
  return model
end
