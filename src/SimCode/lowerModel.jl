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

#= The lowering of an OMJLIR model (after the backend's passes) to its simulation code. =#

"""
    fromOMJLIR(model)

The simulation code of `model` (after the passes): the variables by their simulation names
with their categories and indices, the equations of the continuous-time partition as
residuals `lhs - rhs`, its when-equations (initial ones and the initial arms of others as
initial algorithms), its if-equations, the initial equations, the asserts.
"""
function fromOMJLIR(model::IR.Model)::SIM_CODE
  isempty(model.modes) && return lowerSystem(model, model.variables, model.continuous[1])
  return lowerStructuralModes(model)
end

#= ==================================================================================== =#
#= Structural modes                                                                     =#
#= ==================================================================================== =#

"""
    lowerStructuralModes(model)

The simulation code of a model with structural modes: its top level (its variables, its
equations, every mode has them; its explicit transitions and the mode it starts in) with one
simulation code per mode, the mode's variables and system with the top level's. The variables
shared, whose values a transition carries over: those every mode has, by their names in the
mode (`x` of `firstMode.x` and `secondMode.x`), and the top level's.
"""
function lowerStructuralModes(model::IR.Model)::SIM_CODE
  local top = model.continuous[1]
  any(eq -> eq isa IR.IF_EQUATION, top.equations) &&
    unsupported("an if-equation at the top level of a model with structural modes", model.name)
  local functions = simFunctions(model.functions)
  local subModels = SIM_CODE[]
  for mode in model.modes
    local variables = IR.VariableSet(vcat(collect(mode.variables), collect(model.variables)))
    local system = IR.Partition(mode.system.index, mode.system.association, vcat(mode.system.unknowns, top.unknowns),
                                vcat(mode.system.equations, top.equations),
                                vcat(mode.system.initialEquations, top.initialEquations))
    push!(subModels, lowerSystem(model, variables, system; name = mode.name, functions))
  end
  local simCode = lowerSystem(model, model.variables, top; functions)
  local names = systemNames(IR.Variable[v for v in model.variables if !(v.kind isa IR.STATE_DER)], model.name)
  local transitions = StructuralTransition[EXPLICIT_STRUCTURAL_TRANSITION(t.from, t.to, simExp(names, t.condition))
                                           for t in model.transitions]
  local topVariables = String[simName(names, v.cref) for v in model.variables if !(v.kind isa IR.STATE_DER)]
  return update(simCode; subModels, structuralTransitions = vcat(transitions, simCode.structuralTransitions),
                sharedVariables = sharedModeVariables(model), topVariables,
                sharedEquations = Equation[simCode.residualEquations..., simCode.whenEquations...])
end

"""
    sharedModeVariables(model)

The variables of a model with structural modes a transition carries over: the names in their
mode (the reference without its first part, the mode's) of the states and algebraic variables
every mode has, then the top level's states, algebraic and discrete variables by theirs
(without the first part of a component's: `battery.U` as `U`).
"""
function sharedModeVariables(model::IR.Model)::Vector{String}
  local inMode = v::IR.Variable -> length(v.cref.parts) > 1 ? join((String(p.name) for p in v.cref.parts[2:end]), ".") : String(v.cref.parts[1].name)
  local perMode = [String[inMode(v) for v in mode.variables if v.kind isa Union{IR.STATE, IR.ALGEBRAIC}] for mode in model.modes]
  local common = isempty(perMode) ? String[] : intersect(perMode...)
  return vcat(common, String[inMode(v) for v in model.variables if v.kind isa Union{IR.STATE, IR.ALGEBRAIC, IR.DISCRETE}])
end

#= ==================================================================================== =#
#= The system                                                                           =#
#= ==================================================================================== =#

"""
    lowerSystem(model, variables, partition; name = model.name, functions)

The simulation code of the system `partition` of the variables `variables`, named `name` (a
structural mode's), with the functions `functions` (the model's).
"""
function lowerSystem(model::IR.Model, variables::IR.VariableSet, partition::IR.Partition; name::String = model.name,
                     functions::Vector{ModelicaFunction} = simFunctions(model.functions))::SIM_CODE
  local vars = IR.Variable[v for v in variables if !(v.kind isa IR.STATE_DER)]
  local names = systemNames(vars, model.name)
  local simVars = SIMVAR[SIMVAR(simName(names, v.cref), nothing, simVarKind(v, names), simVarAttributes(v, names),
                                simType(IR.elementType(v.ty))) for v in vars]
  local ht = createIndices(simVars)
  local eqs = classifyEquations(partition.equations)
  local initialAlgorithms = INITIAL_ALGORITHM[INITIAL_ALGORITHM(simStatements(names, w.branches[1].body), w.kind === IR.INITIAL_SECTION)
                                              for w in eqs.initialWhens]
  local whens = substituteSampleTriggers(eqs.whens, partition.equations)
  local (runtimeWhens, initialBodies) = extractInitialWhenAlgorithms(whens)
  append!(initialAlgorithms, INITIAL_ALGORITHM[INITIAL_ALGORITHM(simStatements(names, b), false) for b in initialBodies])
  local residuals = RESIDUAL_EQUATION[residual(names, eq) for eq in eqs.residuals]
  local (asserts, initialAssertAlgorithms) = lowerAsserts(names, ht, eqs.asserts, partition.initialEquations)
  append!(initialAlgorithms, initialAssertAlgorithms)
  initialAlgorithms = INITIAL_ALGORITHM[INITIAL_ALGORITHM(mapStatements(e -> inlineParameters(e, ht), a.statements), a.section)
                                        for a in initialAlgorithms]
  local whenEquations = WHEN_EQUATION[lowerWhen(names, w) for w in runtimeWhens]
  local irreducibles = irreducibleVariables(names, eqs.ifs, whenEquations, vars, ht)
  local simIfEquations = constructIfEquations(names, eqs.ifs)
  local initialEquations::Vector{Equation} = Equation[lowerInitialEquation(names, eq) for eq in partition.initialEquations
                                    if !(eq isa IR.ASSERT_EQUATION)]
  initialEquations = splitTupleInitialEquations(lowerInitialIfEquations(initialEquations), ht)
  local transitions = StructuralTransition[lowerStructuralTransition(names, w) for w in eqs.structural]
  return SIM_CODE(String(codeName(Symbol(name))), ht, residuals, initialEquations, whenEquations, simIfEquations,
                  transitions, SIM_CODE[], String[], String[], Equation[],
                  String(codeName(Symbol(something(model.initialMode, model.name)))), model.frontendModel, irreducibles,
                  functions, any(f -> f isa EXTERNAL_MODELICA_FUNCTION, functions),
                  RESIDUAL_EQUATION[], String[], AliasEntry[], nothing, initialAlgorithms, asserts, String[])
end

"""
    systemNames(vars, modelName)

The simulation names of the variables `vars` (collisions renamed; a variable named `t`, MTK's
independent variable, as `<model>V_t`) and the names known as variables: theirs, `der(x)` of
the states, `time`, `pi`, `e`; those of the variables that are not arrays.
"""
function systemNames(vars::Vector{IR.Variable}, modelName::String)::Names
  local stateOf = Dict{Symbol, IR.ComponentRef}()
  for v in vars
    (v.kind isa IR.STATE && v.kind.derivative !== nothing) && (stateOf[v.kind.derivative] = v.cref)
  end
  local renamed = collidingNames(vars)
  for v in vars
    if legacyName(v.cref) == "t" && !haskey(renamed, v.name)
      renamed[v.name] = string(codeName(Symbol(modelName)), "V_t")
      @warn "Variable name t clash with builtin symbol in MTK. Variable renamed $(renamed[v.name])"
    end
  end
  local names = Names(renamed, Set{String}(), stateOf, Set{String}())
  for v in vars
    local n = simName(names, v.cref)
    push!(names.known, n)
    v.ty isa IR.T_ARRAY || push!(names.scalars, n)
    v.kind isa IR.STATE && push!(names.known, "der($(n))")
  end
  push!(names.known, "time", "pi", "e")
  return names
end

#= ==================================================================================== =#
#= Equations                                                                            =#
#= ==================================================================================== =#

"The equations of a partition by what the simulation code makes of them."
struct ClassifiedEquations
  residuals::Vector{IR.SCALAR_EQUATION}
  whens::Vector{IR.WHEN_EQUATION}
  initialWhens::Vector{IR.WHEN_EQUATION}
  structural::Vector{IR.WHEN_EQUATION}
  ifs::Vector{IR.IF_EQUATION}
  asserts::Vector{IR.ASSERT_EQUATION}
end

"""
    classifyEquations(equations)

The equations of a partition by kind, in their order: the scalar equations (residuals), the
regular, initial and structural when-equations, the if-equations, the asserts. An array or
record equation (the expansion leaves none) is not supported.
"""
function classifyEquations(equations::Vector{IR.Equation})::ClassifiedEquations
  local c = ClassifiedEquations(IR.SCALAR_EQUATION[], IR.WHEN_EQUATION[], IR.WHEN_EQUATION[], IR.WHEN_EQUATION[],
                                IR.IF_EQUATION[], IR.ASSERT_EQUATION[])
  for eq in equations
    if eq isa IR.SCALAR_EQUATION
      push!(c.residuals, eq)
    elseif eq isa IR.WHEN_EQUATION
      push!(eq.kind === IR.REGULAR_WHEN ? c.whens : eq.kind in (IR.INITIAL_WHEN, IR.INITIAL_SECTION) ? c.initialWhens : c.structural, eq)
    elseif eq isa IR.IF_EQUATION
      push!(c.ifs, residualOrder(eq))
    elseif eq isa IR.ASSERT_EQUATION
      push!(c.asserts, eq)
    else
      unsupported("this equation in the simulation code", IR.toString(eq))
    end
  end
  return c
end

"""
    residualOrder(eq)

An if-equation with the equations of each branch but the else-branch in reverse (of nested
if-equations too): the previous backend's residualization reversed them, and the branches'
equations are matched in that order.
"""
residualOrder(eq::IR.IF_EQUATION)::IR.IF_EQUATION =
  IR.IF_EQUATION(IR.EquationBranch[IR.EquationBranch(b.condition, IR.Equation[residualOrder(e) for e in Iterators.reverse(b.equations)])
                                   for b in eq.branches],
                 IR.Equation[residualOrder(e) for e in eq.elseEquations], eq.source, eq.attributes)
residualOrder(eq::IR.Equation)::IR.Equation = eq

"The residual `lhs - rhs` of a scalar equation."
residual(names::Names, eq::IR.SCALAR_EQUATION)::RESIDUAL_EQUATION =
  RESIDUAL_EQUATION(BINARY(simExp(names, eq.lhs), OP_SUB, simExp(names, eq.rhs)), simSource(eq.source),
                    EQ_ATTR(eq.attributes.differentiated, false))

simSource(s::IR.Source)::Source = Source(s.file, s.line)

"""
    lowerInitialEquation(names, eq)

An initial equation in the simulation code: `lhs = rhs` (not a residual: the initialization
solves them as written), an array equation, an if-equation of them.
"""
function lowerInitialEquation(names::Names, eq::IR.Equation)::Equation
  local attr = EQ_ATTR(eq isa IR.ASSERT_EQUATION ? false : eq.attributes.differentiated, false)
  if eq isa IR.SCALAR_EQUATION
    return EQUATION(simExp(names, eq.lhs), simExp(names, eq.rhs), simSource(eq.source), attr)
  elseif eq isa IR.ARRAY_EQUATION
    local dims = Int[something(IR.dimensionSize(d), -1) for d in eq.ty.dimensions]
    return ARRAY_EQUATION(dims, simExp(names, eq.lhs), simExp(names, eq.rhs), simSource(eq.source), attr)
  elseif eq isa IR.IF_EQUATION
    return INLINE_IF_EQUATION(Exp[simExp(names, b.condition) for b in eq.branches],
                              Vector{Equation}[Equation[lowerInitialEquation(names, e) for e in b.equations] for b in eq.branches],
                              Equation[lowerInitialEquation(names, e) for e in eq.elseEquations], simSource(eq.source), attr)
  end
  unsupported("this initial equation in the simulation code", IR.toString(eq))
end

"""
    lowerInitialIfEquations(equations)

The if-equations among the initial equations (CDL SunRiseSet's `if cosHou < -1 then nextSunSet
= f(...) - 86400; else nextSunSet = f(...); end if`) as one equation per position of their
branches: `v = if c1 then e1 elseif ... else en` where every branch assigns the same `v` there,
else `0 = if c1 then l1 - r1 ...`; nested ones first (MSL 4 PowerConverters' SoftStarter). The
branches must have as many plain equations (MLS 8.3.4: in a non-parameter condition they must);
another if-equation is left as it is.
"""
function lowerInitialIfEquations(equations::Vector{Equation})::Vector{Equation}
  local out = Equation[]
  for eq in equations
    local lowered = eq isa INLINE_IF_EQUATION ? lowerInitialIf(eq) : nothing
    lowered === nothing ? push!(out, eq) : append!(out, lowered)
  end
  return out
end

function lowerInitialIf(eq::INLINE_IF_EQUATION)::Union{Nothing, Vector{Equation}}
  local branches = Vector{Equation}[]
  for b in vcat(eq.branchesTrue, [eq.branchElse])
    local flat = Equation[]
    for e in b
      if e isa INLINE_IF_EQUATION
        local inner = lowerInitialIf(e)
        inner === nothing && return nothing
        append!(flat, inner)
      else
        push!(flat, e)
      end
    end
    push!(branches, flat)
  end
  local n = length(first(branches))
  (n > 0 && all(b -> length(b) == n && all(e -> e isa EQUATION, b), branches)) || return nothing
  local lowered = Equation[]
  for k in 1:n
    local ks = EQUATION[b[k] for b in branches]
    local sameLhs = ks[1].lhs isa EXP_CREF && all(e -> e.lhs == ks[1].lhs, ks)
    local value = e -> sameLhs ? e.rhs : BINARY(e.lhs, OP_SUB, e.rhs)
    local exp = value(ks[end])
    for j in length(eq.conditions):-1:1
      exp = IFEXP(eq.conditions[j], value(ks[j]), exp)
    end
    push!(lowered, EQUATION(sameLhs ? ks[1].lhs : RCONST(0.0), exp, eq.source, eq.attr))
  end
  return lowered
end

"""
    splitTupleInitialEquations(equations, ht)

`(a, b) = f(...)` among the initial equations, its targets variables, as one equation per
element of each target, the call's output by TSUB (an array output's element by ASUB of it):
Buildings' occupant windows' `(ran, state) = Xorshift1024star.random(...)`. A pure call, once
per element; with a target that is a parameter the equation stays.
"""
function splitTupleInitialEquations(equations::Vector{Equation}, ht::HT)::Vector{Equation}
  local out = Equation[]
  for eq in equations
    local split = eq isa EQUATION ? splitTupleEquation(eq, ht) : nothing
    split === nothing ? push!(out, eq) : append!(out, split)
  end
  return out
end

function splitTupleEquation(eq::EQUATION, ht::HT)::Union{Nothing, Vector{Equation}}
  (eq.lhs isa TUPLE && eq.rhs isa CALL && eq.rhs.kind !== CALL_IMPURE_FUNCTION && eq.rhs.ty isa TYPE_TUPLE) || return nothing
  local types = eq.rhs.ty.types
  local targets = eq.lhs.elements
  length(types) >= length(targets) || return nothing
  local isVariable = e -> e isa EXP_CREF && (local entry = get(ht, crefName(e.cref), nothing); entry !== nothing && !(last(entry).varKind isa PARAMETER))
  local eqs = Equation[]
  for (k, target) in enumerate(targets)
    target isa WILD && continue
    local output = TSUB(eq.rhs, k, types[k])
    if target isa ARRAY_EXP
      all(isVariable, target.elements) || return nothing
      for (i, el) in enumerate(target.elements)
        push!(eqs, EQUATION(el, ASUB(output, Exp[ICONST(i)]), eq.source, eq.attr))
      end
    elseif isVariable(target)
      push!(eqs, EQUATION(target, output, eq.source, eq.attr))
    else
      return nothing
    end
  end
  return eqs
end

#= ==================================================================================== =#
#= Asserts                                                                              =#
#= ==================================================================================== =#

"""
    lowerAsserts(names, ht, asserts, initialEquations)

The asserts of the equations, and the initial algorithms of the asserts of the initial
equations: an initial assert on parameters joins the asserts (constant, it holds after), one
on variables runs once at the initialization; a call for its effects in an initial equation's
branch (`if guard then call else true`) is made once there. The parameters in the asserts as
their values (the passes may eliminate them); a tunable one stays a reference.
"""
function lowerAsserts(names::Names, ht::OrderedDict{String, Tuple{Int, SIMVAR}}, asserts::Vector{IR.ASSERT_EQUATION},
                      initialEquations::Vector{IR.Equation})::Tuple{Vector{ASSERT_EQUATION}, Vector{INITIAL_ALGORITHM}}
  local out = ASSERT_EQUATION[simAssert(names, a) for a in asserts]
  local initial = INITIAL_ALGORITHM[]
  for eq in initialEquations
    eq isa IR.ASSERT_EQUATION || continue
    local a = simAssert(names, eq)
    if a.condition isa IFEXP && a.condition.thenExp isa CALL && isNoReturnCall(eq.condition.thenExp)
      push!(initial, INITIAL_ALGORITHM(Statement[NORETCALL(a.condition, a.source)], false))
      continue
    end
    local onVariables = any(n -> (local entry = get(ht, n, nothing); entry !== nothing && !(last(entry).varKind isa PARAMETER)),
                            crefNames!(crefNames!(OrderedSet{String}(), a.condition), a.message))
    if !onVariables
      push!(out, a)
    elseif a.condition isa CALL && isNoReturnCall(eq.condition)
      push!(initial, INITIAL_ALGORITHM(Statement[NORETCALL(a.condition, a.source)], false))
    else
      push!(initial, INITIAL_ALGORITHM(Statement[ASSERT(a.condition, a.message, a.level, a.source)], false))
    end
  end
  local inline = e -> inlineParameters(e, ht; keepTunable = true, keepCalls = true)
  out = ASSERT_EQUATION[ASSERT_EQUATION(inline(a.condition), inline(a.message), a.level, a.source) for a in out]
  return (out, initial)
end

simAssert(names::Names, a::IR.ASSERT_EQUATION)::ASSERT_EQUATION =
  ASSERT_EQUATION(simExp(names, a.condition), simExp(names, a.message), simExp(names, a.level), simSource(a.source))

isNoReturnCall(e::IR.Expression)::Bool = e isa IR.CALL && e.ty isa IR.T_NORETCALL

"""
    inlineParameters(e, ht; keepTunable)

`e` with the parameters replaced by their bindings, to a fixed point (a parameter bound to
another): the initial algorithms run when the generated module is loaded, the asserts read
parameters the passes may eliminate. An element of an array parameter scalarized into its
elements is the array of the elements' bindings. A tunable parameter is an error in an initial
algorithm (it would keep its compiled value), a reference with `keepTunable`. With `keepCalls`
(the asserts), a parameter bound to a call of a function stays a reference, its value computed
once: inlined, a time-varying assert's crossing called it at every evaluation (Buildings'
weather data, `getTimeSpanTMY3(filNam, "tab1")` reading the weather file, 40 ms a call).
"""
function inlineParameters(e::Exp, ht::OrderedDict{String, Tuple{Int, SIMVAR}}; keepTunable::Bool = false,
                          keepCalls::Bool = false)::Exp
  local visit = x -> begin
    local name = x isa EXP_CREF ? crefName(x.cref) : nothing
    name === nothing && return x
    if isTunableParameter(name)
      keepTunable && return x
      throw(ArgumentError("tunable parameter $(name) is read in an initial algorithm, which is evaluated " *
                          "when the model is compiled, so it would keep its compiled value; leave it out of " *
                          "withTunableParameters"))
    end
    local entry = get(ht, name, nothing)
    if entry !== nothing
      local k = last(entry).varKind
      local bind = k isa Union{PARAMETER, ARRAY_PARAMETER} ? k.bindExp : nothing
      (bind === nothing || (keepCalls && anyExp(y -> y isa CALL && y.kind !== CALL_BUILTIN, bind))) && return x
      return bind
    end
    return something(scalarizedArrayParameter(name, ht), Some(x))
  end
  local visitElements = x -> x isa ASUB ? something(elementsBySubscript(x, ht), Some(x)) : visit(x)
  local current = e
  for _ in 1:100
    local next = mapExp(visitElements, current)
    next == current && return current
    current = next
  end
  return current
end

"""
    scalarizedArrayParameter(name, ht)

The array value of an array parameter `name` known by its elements' parameters (`A[1][2]`,
1 or 2 dimensions) with bindings, a missing element 0.0; `nothing` if there are none.
"""
function scalarizedArrayParameter(name::String, ht::OrderedDict{String, Tuple{Int, SIMVAR}})::Union{Nothing, Exp}
  haskey(ht, name) && return nothing
  local entries = Tuple{Vector{Int}, Exp}[]
  for (key, (_, sv)) in ht
    local parsed = parseElementName(key)
    (parsed === nothing || parsed[1] != name) && continue
    (sv.varKind isa PARAMETER && sv.varKind.bindExp !== nothing) || continue
    push!(entries, (parsed[2], sv.varKind.bindExp))
  end
  isempty(entries) && return nothing
  local element = idx -> (local k = findfirst(x -> x[1] == idx, entries); k === nothing ? RCONST(0.0) : entries[k][2])
  local nDims = length(entries[1][1])
  if nDims == 1
    local n = maximum(x -> x[1][1], entries)
    return ARRAY_EXP(TYPE_ARRAY(TYPE_REAL(), Int[n]), false, Exp[element([i]) for i in 1:n])
  elseif nDims == 2
    local (n, m) = (maximum(x -> x[1][1], entries), maximum(x -> x[1][2], entries))
    local rowTy = TYPE_ARRAY(TYPE_REAL(), Int[m])
    return ARRAY_EXP(TYPE_ARRAY(rowTy, Int[n]), false,
                     Exp[ARRAY_EXP(rowTy, false, Exp[element([i, j]) for j in 1:m]) for i in 1:n])
  end
  return nothing
end

"""
    elementsBySubscript(e, ht)

An array that is not a variable read by a subscript that is not a literal (a loop's iterator:
the plotters' `legend[i]`), its elements 1, 2, ... variables: the array of the elements'
references, subscripted; `nothing` for another expression.
"""
function elementsBySubscript(e::ASUB, ht::OrderedDict{String, Tuple{Int, SIMVAR}})::Union{Nothing, Exp}
  (e.exp isa EXP_CREF && length(e.subs) == 1 && !(e.subs[1] isa ICONST)) || return nothing
  local base = crefName(e.exp.cref)
  haskey(ht, base) && return nothing
  local elementTy = e.exp.ty isa TYPE_ARRAY ? e.exp.ty.elementType : e.exp.ty
  local elements = Exp[]
  while haskey(ht, string(base, "[", length(elements) + 1, "]"))
    push!(elements, EXP_CREF(SimCref(Symbol(base, "[", length(elements) + 1, "]")), elementTy))
  end
  isempty(elements) && return nothing
  return ASUB(ARRAY_EXP(TYPE_ARRAY(elementTy, Int[length(elements)]), true, elements), e.subs)
end

"An element's name `A[1][2]` as `(\"A\", [1, 2])`; `nothing` for another name."
function parseElementName(key::String)::Union{Nothing, Tuple{String, Vector{Int}}}
  local open = findfirst('[', key)
  open === nothing && return nothing
  local indices = Int[]
  local rest = key[open:end]
  while !isempty(rest)
    rest[1] == '[' || return nothing
    local close = findfirst(']', rest)
    close === nothing && return nothing
    local i = tryparse(Int, rest[2:close-1])
    i === nothing && return nothing
    push!(indices, i)
    rest = rest[close+1:end]
  end
  return (key[1:open-1], indices)
end

#= ==================================================================================== =#
#= When-equations                                                                       =#
#= ==================================================================================== =#

"""
    substituteSampleTriggers(whens, equations)

A when-condition on a Boolean defined by `b = sample(start, interval)` (MSL DiscreteBlock's
`sampleTrigger`, read by `when {sampleTrigger, initial()}`), or by a guard and-ed with one
(Buildings' plotters), has the definition in place of `b`: only a condition with a sample()
call is a periodic callback, `b` itself is false between the ticks. Only the condition's
Boolean structure (references, and, or) is rewritten; in the bodies, the reads of `b` outside
pre(), edge() and change() too (true at the ticks only).
"""
function substituteSampleTriggers(whens::Vector{IR.WHEN_EQUATION}, equations::Vector{IR.Equation})::Vector{IR.WHEN_EQUATION}
  local defs = Dict{Symbol, IR.Expression}()
  for eq in equations
    eq isa IR.SCALAR_EQUATION || continue
    for (a, b) in ((eq.lhs, eq.rhs), (eq.rhs, eq.lhs))
      (a isa IR.CREF && isSampleTrigger(b)) && (defs[a.cref.name] = b)
    end
  end
  isempty(defs) && return whens
  local subst
  subst = c -> c isa IR.CREF ? get(defs, c.cref.name, c) :
               c isa IR.LBINARY ? IR.LBINARY(subst(c.exp1), c.op, subst(c.exp2)) : c
  local inBody = e -> IR.mapExpTopDown(e) do x
    isBuiltinCallOf(x, (:pre, :edge, :change)) && return (x, false)
    x isa IR.CREF && return (get(defs, x.cref.name, x), false)
    return (x, true)
  end
  local bodyStatement = s -> s isa IR.ASSIGNMENT ? IR.ASSIGNMENT(s.lhs, inBody(s.rhs), s.source) :
                             s isa IR.REINIT ? IR.REINIT(s.state, inBody(s.value), s.source) :
                             s isa IR.ASSERT ? IR.ASSERT(inBody(s.condition), s.message, s.level, s.source) : s
  return IR.WHEN_EQUATION[IR.WHEN_EQUATION(IR.WhenBranch[IR.WhenBranch(IR.Expression[subst(t) for t in b.triggers],
                                                                      b.atInitial, b.atTerminal,
                                                                      IR.Statement[bodyStatement(s) for s in whenOperators(b.body)])
                                                          for b in w.branches], w.kind, w.source, w.attributes)
                          for w in whens]
end

"""
    extractInitialWhenAlgorithms(whens)

The runtime when-equations, and the bodies that run once at the initialization: a `when
initial()` is the latter only (its elsewhen arms the runtime when: initial() is false after the
initialization), a `when {c, ..., initial()}` both (its runtime arm on the other triggers,
marked to run at the start of a solve too). A tuple assignment in an initial body is one per
target.
"""
function extractInitialWhenAlgorithms(whens::Vector{IR.WHEN_EQUATION})::Tuple{Vector{RuntimeWhen}, Vector{Vector{IR.Statement}}}
  local runtime = RuntimeWhen[]
  local initial = Vector{IR.Statement}[]
  local visit
  visit = (branches::Vector{IR.WhenBranch}, w::IR.WHEN_EQUATION, afterInitial::Bool) -> begin
    local b = branches[1]
    if b.atInitial && isempty(b.triggers) && !b.atTerminal
      afterInitial || push!(initial, tupleAssignmentsSplit(whenOperators(b.body)))
      length(branches) > 1 && visit(branches[2:end], w, true)
    elseif b.atInitial
      afterInitial || push!(initial, tupleAssignmentsSplit(whenOperators(b.body)))
      #= its other triggers as one condition, a terminal() one among them (not split) =#
      local runtimeBranches = IR.WhenBranch[IR.WhenBranch(b.triggers, false, b.atTerminal, b.body), branches[2:end]...]
      push!(runtime, RuntimeWhen(runtimeBranches, w.source, true))
    else
      append!(runtime, splitTerminalTrigger(RuntimeWhen(branches, w.source, false)))
    end
  end
  foreach(w -> visit(w.branches, w, false), whens)
  return (runtime, initial)
end

"A runtime when-equation: its branches, source, and whether it runs at the start of a solve too (its condition had `initial()`)."
struct RuntimeWhen
  branches::Vector{IR.WhenBranch}
  source::IR.Source
  alsoInitial::Bool
end

"""
    splitTerminalTrigger(w)

`when {c1, ..., terminal()}` as `when {c1, ...}` and `when terminal()` with the same body: a
vector trigger fires at any element's rising edge, a `terminal()` when runs after the solve. A
when with elsewhen arms stays.
"""
function splitTerminalTrigger(w::RuntimeWhen)::Vector{RuntimeWhen}
  local b = w.branches[1]
  (length(w.branches) == 1 && b.atTerminal && !isempty(b.triggers)) || return RuntimeWhen[w]
  return RuntimeWhen[RuntimeWhen(IR.WhenBranch[IR.WhenBranch(b.triggers, false, false, b.body)], w.source, w.alsoInitial),
                     RuntimeWhen(IR.WhenBranch[IR.WhenBranch(IR.Expression[], false, true, b.body)], w.source, w.alsoInitial)]
end

"""
    tupleAssignmentsSplit(body)

`(a, b) := f(x)` as one assignment per target (as an algorithm has them): the initial
algorithms take a variable on the left only.
"""
function tupleAssignmentsSplit(body::Vector{IR.Statement})::Vector{IR.Statement}
  local out = IR.Statement[]
  for s in body
    if s isa IR.ASSIGNMENT && s.lhs isa IR.TUPLE
      for (target, value) in tupleAssignElements(s.lhs.elements, s.rhs)
        push!(out, IR.ASSIGNMENT(target, value, s.source))
      end
    else
      push!(out, s)
    end
  end
  return out
end

"""
    whenOperators(body)

A when-equation's body without if-statements (MLS 8.3.5: their branches assign the same
variables): one assignment per variable of `if c1 then e1 elseif ... else en`, a branch
without one keeping the value; a `reinit` likewise (to the state itself); an assert under its
branch's condition. A call for its effects in a branch is not made.
"""
function whenOperators(body::Vector{IR.Statement})::Vector{IR.Statement}
  any(s -> s isa IR.IF_STATEMENT, body) || return body
  local out = IR.Statement[]
  for s in body
    s isa IR.IF_STATEMENT ? append!(out, ifStatementOperators(s)) : push!(out, s)
  end
  return out
end

function ifStatementOperators(s::IR.IF_STATEMENT)::Vector{IR.Statement}
  local bodies = vcat([whenOperators(b.body) for b in s.branches], [whenOperators(s.elseBody)])
  local conditions = IR.Expression[b.condition for b in s.branches]
  local guards = IR.Expression[]
  local before = nothing
  for c in conditions
    push!(guards, before === nothing ? c : IR.LBINARY(before, IR.AND, c))
    before = before === nothing ? IR.LUNARY(IR.NOT, c) : IR.LBINARY(before, IR.AND, IR.LUNARY(IR.NOT, c))
  end
  push!(guards, before === nothing ? IR.BOOLEAN(true) : before)
  local targets = OrderedDict{Tuple{Symbol, IR.Expression}, IR.Expression}()
  local values = [Dict{Tuple{Symbol, IR.Expression}, IR.Expression}() for _ in bodies]
  local asserts = IR.Statement[]
  for (k, ops) in enumerate(bodies), op in ops
    if op isa IR.ASSIGNMENT
      targets[(:assign, op.lhs)] = op.lhs
      values[k][(:assign, op.lhs)] = op.rhs
    elseif op isa IR.REINIT
      local state = IR.CREF(op.state, IR.T_REAL())
      targets[(:reinit, state)] = state
      values[k][(:reinit, state)] = op.value
    elseif op isa IR.ASSERT
      push!(asserts, IR.ASSERT(IR.LBINARY(IR.LUNARY(IR.NOT, guards[k]), IR.OR, op.condition), op.message, op.level, op.source))
    elseif !(op isa IR.NORETCALL)
      unsupported("this statement in an if-equation in a when-equation", IR.toString(op))
    end
  end
  local out = IR.Statement[]
  for ((kind, _), target) in targets
    local key = (kind, target)
    local value = get(values[end], key, target)
    for k in length(bodies)-1:-1:1
      value = IR.IF_EXPRESSION(conditions[k], get(values[k], key, target), value, IR.typeOf(target))
    end
    push!(out, kind === :assign ? IR.ASSIGNMENT(target, value, s.source) : IR.REINIT(target.cref, value, s.source))
  end
  return vcat(out, asserts)
end

"""
    lowerWhen(names, w)

A runtime when-equation in the simulation code: each branch's condition its triggers as one
(`whenCondition`), its body.
"""
function lowerWhen(names::Names, w::RuntimeWhen)::WHEN_EQUATION
  local stmts = nothing
  for b in Iterators.reverse(w.branches)
    stmts = WHEN_STMTS(simExp(names, whenCondition(b)), simStatements(names, whenOperators(b.body)), stmts)
  end
  return WHEN_EQUATION(stmts, simSource(w.source), EQ_ATTR(false, w.alsoInitial))
end

"""
    whenCondition(b)

A branch's triggers as one condition: one is itself; with several, the when fires at each
one's rising edge (MLS 8.3.5), so each is `edge(c)` (an event call `sample()`, `change()`,
`edge()` as it is), and-or'ed: `edge(c1) or edge(c2)` (`c1 or c2` rises once while `c1` stays
true). `initial()` and `terminal()` (in an elsewhen arm) are triggers here, first and last.
"""
function whenCondition(b::IR.WhenBranch)::IR.Expression
  local triggers = copy(b.triggers)
  b.atInitial && pushfirst!(triggers, initialCall())
  b.atTerminal && push!(triggers, builtinCall(:terminal, IR.Expression[], IR.T_BOOLEAN()))
  length(triggers) == 1 && return only(triggers)
  local edgeOf = e -> isBuiltinCallOf(e, (:sample, :change, :edge)) ? e : edgeCall(e)
  return foldl((a, c) -> IR.LBINARY(a, IR.OR, c), map(edgeOf, triggers))
end

"""
    lowerStructuralTransition(names, w)

A structural when-equation (OM.jl's `recompilation`): the transition is known when its body
runs.
"""
function lowerStructuralTransition(names::Names, w::IR.WHEN_EQUATION)::StructuralTransition
  local stmts = nothing
  for b in Iterators.reverse(w.branches)
    stmts = WHEN_STMTS(simExp(names, whenCondition(b)), simStatements(names, whenOperators(b.body)), stmts)
  end
  return IMPLICIT_STRUCTURAL_TRANSITION(stmts, simSource(w.source), EQ_ATTR_DEFAULT)
end

#= ==================================================================================== =#
#= Statements                                                                           =#
#= ==================================================================================== =#

"The statements in the simulation code."
simStatements(names::Names, body::Vector{IR.Statement})::Vector{Statement} = Statement[simStatement(names, s) for s in body]

simStatement(names::Names, s::IR.ASSIGNMENT)::Statement = ASSIGN(simExp(names, s.lhs), simExp(names, s.rhs), simSource(s.source))
simStatement(names::Names, s::IR.IF_STATEMENT)::Statement =
  IF(Branch[Branch(simExp(names, b.condition), simStatements(names, b.body)) for b in s.branches],
     simStatements(names, s.elseBody), simSource(s.source))
simStatement(names::Names, s::IR.FOR_STATEMENT)::Statement =
  FOR(Iterator(s.iterator.name, simExp(names, s.iterator.range)), simStatements(names, s.body), simSource(s.source))
simStatement(names::Names, s::IR.WHILE_STATEMENT)::Statement =
  WHILE(simExp(names, s.condition), simStatements(names, s.body), simSource(s.source))
simStatement(names::Names, s::IR.WHEN_STATEMENT)::Statement =
  WHEN(Branch[Branch(simExp(names, whenCondition(b)), simStatements(names, b.body)) for b in s.branches], simSource(s.source))
simStatement(names::Names, s::IR.ASSERT)::Statement =
  ASSERT(simExp(names, s.condition), simExp(names, s.message), simExp(names, s.level), simSource(s.source))
simStatement(names::Names, s::IR.TERMINATE)::Statement = TERMINATE(simExp(names, s.message), simSource(s.source))
simStatement(names::Names, s::IR.REINIT)::Statement = REINIT(simCref(names, s.state), simExp(names, s.value), simSource(s.source))
simStatement(names::Names, s::IR.NORETCALL)::Statement = NORETCALL(simExp(names, s.call), simSource(s.source))
simStatement(::Names, s::IR.RETURN)::Statement = RETURN(simSource(s.source))
simStatement(::Names, s::IR.BREAK)::Statement = BREAK(simSource(s.source))
simStatement(names::Names, s::IR.RECOMPILATION)::Statement = RECOMPILATION(simCref(names, s.component), simExp(names, s.newValue))
simStatement(names::Names, s::IR.AGENTIC_RECOMPILATION)::Statement =
  AGENTIC_RECOMPILATION(SimCref[simCref(names, c) for c in s.components], s.prompt, s.initialEquations)
