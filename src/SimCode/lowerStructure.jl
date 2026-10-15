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

#= The structure of the simulation code: the irreducible variables, the equation-variable
   graph, the matching and the strongly connected components, the if-equations' branches. =#

#= ==================================================================================== =#
#= Irreducible variables                                                                =#
#= ==================================================================================== =#

"""
    irreducibleVariables(names, ifEquations, whens, vars, ht)

The variables the code generation must keep as unknowns (MTK must not eliminate them), never
parameters:

- of an if-equation whose branches assign one variable (every lifted if-expression), that
  variable; of another, the variables of its conditions and equations and what its branches
  assign (nested if-equations included: a state an if-equation defines, MSL 4 KinematicPTP's
  `s`, became an observed ifelse read back with the final parameter values);
- the states the when-equations read or write (their callbacks index the state vector by name);
- the discretes in the when-conditions (an eliminated one is read by the callback), of their
  first branches (the previous backend's);
- a variable named `*THETA` (Zimmer's homotopy parameter).
"""
function irreducibleVariables(names::Names, ifEquations::Vector{IR.IF_EQUATION}, whens::Vector{WHEN_EQUATION},
                              vars::Vector{IR.Variable}, ht::HT)::Vector{String}
  local out = String[]
  for eq in ifEquations
    local target = liftedIfTarget(names, eq)
    if target !== nothing
      push!(out, target)
    else
      append!(out, ifEquationVariables(names, eq, vars))
      append!(out, ifEquationTargets(names, eq))
    end
  end
  local stateNames = Set{String}(simName(names, v.cref) for v in vars if v.kind isa IR.STATE)
  append!(out, statesUsedByWhens(whens, stateNames))
  filter!(n -> n == "time" || (haskey(ht, n) && !(last(ht[n]).varKind isa PARAMETER)), out)
  for w in whens, n in allCrefNames(w.whenEquation.condition)
    (haskey(ht, n) && last(ht[n]).varKind isa DISCRETE) && push!(out, n)
  end
  local theta = thetaVariable(ht)
  theta === nothing || push!(out, theta)
  return filter(n -> n != "time", out)
end

"The variable named `*THETA` (at most one)."
function thetaVariable(ht::HT)::Union{Nothing, String}
  local thetas = [k for k in keys(ht) if endswith(k, "THETA")]
  @assert length(thetas) < 2
  return isempty(thetas) ? nothing : first(thetas)
end

"""
    addThetaResidual!(residuals, irreducibles, ht)

Zimmer's homotopy parameter `*THETA` as an algebraic variable with the equation `THETA = 1`
(irreducible).
"""
function addThetaResidual!(residuals::Vector{RESIDUAL_EQUATION}, irreducibles::Vector{String}, ht::HT)::Nothing
  local theta = thetaVariable(ht)
  theta === nothing && return nothing
  push!(irreducibles, theta)
  push!(residuals, RESIDUAL_EQUATION(BINARY(EXP_CREF(SimCref(Symbol(theta)), TYPE_REAL()), OP_SUB, RCONST(1.0)),
                                     NO_SOURCE, EQ_ATTR_DEFAULT))
  local (index, sv) = ht[theta]
  ht[theta] = (index, SIMVAR(sv.name, sv.index, ALG_VARIABLE(0), sv.attributes))
  return nothing
end

"The equations of an if-equation's branches (the else-branch's last)."
branchEquations(eq::IR.IF_EQUATION) = Iterators.flatten((Iterators.flatten(b.equations for b in eq.branches), eq.elseEquations))

"""
The variable an if-equation's branches all assign, with one equation each (`v = e`, not a
derivative `der(x) = e`); `nothing` for another.
"""
function liftedIfTarget(names::Names, eq::IR.IF_EQUATION)::Union{Nothing, String}
  local targets = Set{String}()
  for e in branchEquations(eq)
    (e isa IR.SCALAR_EQUATION && isVariableReference(names, e.lhs)) || return nothing
    push!(targets, simName(names, e.lhs.cref))
  end
  return length(targets) == 1 ? only(targets) : nothing
end

"Whether `e` references a variable (not a derivative)."
isVariableReference(names::Names, e::IR.Expression)::Bool = e isa IR.CREF && !haskey(names.stateOf, e.cref.name)

"The variables an if-equation's branches assign (their equations' left sides), of nested ones too."
function ifEquationTargets(names::Names, eq::IR.IF_EQUATION)::Vector{String}
  local out = String[]
  for e in branchEquations(eq)
    if e isa IR.IF_EQUATION
      append!(out, ifEquationTargets(names, e))
    elseif e isa IR.SCALAR_EQUATION && isVariableReference(names, e.lhs)
      push!(out, simName(names, e.lhs.cref))
    end
  end
  return out
end

"""
    ifEquationVariables(names, eq, vars)

The references of an if-equation's conditions, then for each of its equations the algebraic
variables it reads and the states it differentiates (in the variables' order); of nested
if-equations in turn.
"""
function ifEquationVariables(names::Names, eq::IR.IF_EQUATION, vars::Vector{IR.Variable})::Vector{String}
  local out = String[]
  for b in eq.branches
    append!(out, reverse(unique(referenceNames(names, b.condition))))
  end
  for e in branchEquations(eq)
    if e isa IR.IF_EQUATION
      append!(out, ifEquationVariables(names, e, vars))
    elseif e isa IR.SCALAR_EQUATION
      local read = Set{String}(vcat(referenceNames(names, e.lhs), referenceNames(names, e.rhs)))
      local differentiated = Set{String}(vcat(derivativeNames(names, e.lhs), derivativeNames(names, e.rhs)))
      for v in vars
        local n = simName(names, v.cref)
        ((v.kind isa IR.ALGEBRAIC && n in read) || n in differentiated) && push!(out, n)
      end
    end
  end
  return out
end

"The simulation names of the references in `e` (a derivative's state's too), in order."
function referenceNames(names::Names, e::IR.Expression)::Vector{String}
  local out = String[]
  IR.foreachExp(e) do x
    x isa IR.CREF || return
    local state = get(names.stateOf, x.cref.name, nothing)
    push!(out, simName(names, state === nothing ? x.cref : state))
  end
  return out
end

"The simulation names of the states whose derivatives `e` reads."
function derivativeNames(names::Names, e::IR.Expression)::Vector{String}
  local out = String[]
  IR.foreachExp(e) do x
    x isa IR.CREF || return
    local state = get(names.stateOf, x.cref.name, nothing)
    state === nothing || push!(out, simName(names, state))
  end
  return out
end

"""
    statesUsedByWhens(whens, stateNames)

The states the when-equations read or write: in the conditions and the bodies, the elsewhen
arms included.
"""
function statesUsedByWhens(whens::Vector{WHEN_EQUATION}, stateNames::Set{String})::Vector{String}
  local used = OrderedSet{String}()
  local visit = e -> foreach(n -> n in stateNames && push!(used, n), allCrefNames(e))
  for w in whens
    local stmts = w.whenEquation
    while stmts !== nothing
      visit(stmts.condition)
      for s in stmts.body
        foreach(visit, whenStatementExps(s))
      end
      stmts = stmts.elsewhenPart
    end
  end
  return collect(used)
end

"""
    allCrefNames(e)

The names of the references in `e`, each once, in the reverse order of their first occurrence
(the previous backend's order, a list built by consing).
"""
function allCrefNames(e::Exp)::Vector{String}
  local out = String[]
  foreachExp(x -> x isa Union{EXP_CREF, DER, PRE} && push!(out, crefName(x.cref)), e)
  return reverse(unique(out))
end

"The expressions of a when-body's statement: an assignment's sides, a reinit's state and value, ..."
whenStatementExps(s::ASSIGN)::Vector{Exp} = Exp[s.left, s.right]
whenStatementExps(s::REINIT)::Vector{Exp} = Exp[EXP_CREF(s.stateVar, TYPE_REAL()), s.value]
whenStatementExps(s::ASSERT)::Vector{Exp} = Exp[s.condition, s.message]
whenStatementExps(s::TERMINATE)::Vector{Exp} = Exp[s.message]
whenStatementExps(s::NORETCALL)::Vector{Exp} = Exp[s.exp]
whenStatementExps(s::Statement)::Vector{Exp} = Exp[]

#= ==================================================================================== =#
#= The equation-variable graph, matching, components                                   =#
#= ==================================================================================== =#

"""
    equationVariableGraph(names, residuals, ifEquations, whens, ht)

The bipartite graph of the residual system for the matching: each equation's unknowns by
their indices (an algebraic variable's sort index, a state's index): the residuals', the
first branch's equations of each if-equation (the branches are balanced), and the Real
assignments of the when-equations' first branches.
"""
function equationVariableGraph(names::Names, residuals::Vector{RESIDUAL_EQUATION}, ifEquations::Vector{IR.IF_EQUATION},
                               whens::Vector{WHEN_EQUATION}, ht::HT)::OrderedDict{Int, Vector{Int}}
  local graph = OrderedDict{Int, Vector{Int}}()
  for r in residuals
    graph[length(graph) + 1] = equationUnknowns(r.exp, ht)
  end
  addIfEquationRows!(graph, names, ifEquations, ht)
  addWhenRows!(graph, whens, ht)
  return graph
end

"""
    equationUnknowns(e, ht)

The unknowns of a residual `e` by their indices, sorted: the algebraic variables it reads (in
`pre()` too) and the states whose derivatives it reads.
"""
function equationUnknowns(e::Exp, ht::HT)::Vector{Int}
  local out = Int[]
  local seen = Set{String}()
  foreachExp(e) do x
    local (name, derivative) = x isa Union{EXP_CREF, PRE} ? (crefName(x.cref), false) : x isa DER ? (crefName(x.cref), true) : (nothing, false)
    (name === nothing || name in seen) && return
    local entry = get(ht, name, nothing)
    entry === nothing && return
    local (index, sv) = entry
    if !derivative && sv.varKind isa ALG_VARIABLE
      push!(seen, name)
      push!(out, sv.varKind.sortIdx)
    elseif derivative && sv.varKind isa STATE
      push!(seen, name)
      push!(out, index)
    end
  end
  return sort(out)
end

"The indices of the unknowns among `refs` (an algebraic variable's sort index, a state's index), sorted."
function unknownIndices(refs, ht::HT)::Vector{Int}
  local out = Int[]
  for name in refs
    local entry = get(ht, name, nothing)
    entry === nothing && continue
    local (index, sv) = entry
    sv.varKind isa ALG_VARIABLE ? push!(out, sv.varKind.sortIdx) : sv.varKind isa STATE && push!(out, index)
  end
  return sort(out)
end

function addIfEquationRows!(graph::OrderedDict{Int, Vector{Int}}, names::Names, ifEquations::Vector{IR.IF_EQUATION}, ht::HT)
  for eq in ifEquations
    isempty(eq.branches) && continue
    for e in eq.branches[1].equations
      if e isa IR.SCALAR_EQUATION
        graph[length(graph) + 1] = equationUnknowns(BINARY(simExp(names, e.lhs), OP_SUB, simExp(names, e.rhs)), ht)
      elseif e isa IR.IF_EQUATION
        graph[length(graph) + 1] = unknownIndices(nestedIfReferences(names, e, ht), ht)
      else
        graph[length(graph) + 1] = Int[]
      end
    end
  end
  return graph
end

"""
The references of a nested if-equation for the graph (the previous backend's): its
conditions' references (states too), its equations' algebraic variables and differentiated
states.
"""
function nestedIfReferences(names::Names, eq::IR.IF_EQUATION, ht::HT)::Vector{String}
  local out = String[]
  for b in eq.branches
    append!(out, referenceNames(names, b.condition))
  end
  for e in branchEquations(eq)
    if e isa IR.IF_EQUATION
      append!(out, nestedIfReferences(names, e, ht))
    elseif e isa IR.SCALAR_EQUATION
      foreachExp(BINARY(simExp(names, e.lhs), OP_SUB, simExp(names, e.rhs))) do x
        if x isa Union{EXP_CREF, PRE}
          local entry = get(ht, crefName(x.cref), nothing)
          (entry !== nothing && last(entry).varKind isa ALG_VARIABLE) && push!(out, crefName(x.cref))
        elseif x isa DER
          push!(out, crefName(x.cref))
        end
      end
    end
  end
  return out
end

function addWhenRows!(graph::OrderedDict{Int, Vector{Int}}, whens::Vector{WHEN_EQUATION}, ht::HT)
  for w in whens, s in w.whenEquation.body
    (s isa ASSIGN && s.left isa EXP_CREF && s.left.ty isa TYPE_REAL) || continue
    local refs = String[]
    local add = x -> x isa Union{EXP_CREF, DER, PRE} && push!(refs, crefName(x.cref))
    foreachExp(add, s.left)
    foreachExp(add, s.right)
    graph[length(graph) + 1] = unknownIndices(refs, ht)
  end
  return graph
end

"""
    matchAndSort(graph)

The matching of the equations to their unknowns and the strongly connected components of the
resulting graph; a structurally singular system is left to MTK (its structural_simplify does
index reduction), as one the matching fails on (the recursion overflowing on a large one).
"""
function matchAndSort(graph::OrderedDict{Int, Vector{Int}})::Tuple{Bool, Vector{Int}, MetaGraphs.MetaDiGraph, Vector{Vector{Int}}}
  local (isSingular, matchOrder) = try
    GraphAlgorithms.matching(graph, length(graph.keys))
  catch e
    e isa StackOverflowError || _fallback(e, :matchingDelegatedToMTK)
    return (true, Int[], MetaGraphs.MetaDiGraph(), Vector{Int}[])
  end
  local digraph = GraphAlgorithms.merge(matchOrder, graph)
  return (isSingular, matchOrder, digraph, GraphAlgorithms.stronglyConnectedComponents(digraph))
end

#= ==================================================================================== =#
#= If-equations                                                                         =#
#= ==================================================================================== =#

"""
    constructIfEquations(names, ifEquations, residuals, whens, ht)

The if-equations of the simulation code, each a small graph of branches: per branch its
condition, its residuals, and the structure (matching, components) of the system with the
branch's equations in place of the if-equation (the others' first branches as in the main
graph). Nested if-equations are flattened first (`flattenIfEquation`); an if-equation without
an else-branch holds no equations (the asserts were hoisted) and is left out.
"""
function constructIfEquations(names::Names, ifEquations::Vector{IR.IF_EQUATION}, residuals::Vector{RESIDUAL_EQUATION},
                              whens::Vector{WHEN_EQUATION}, ht::HT)::Vector{IF_EQUATION}
  local out = IF_EQUATION[]
  for (i, original) in enumerate(ifEquations)
    local eq = flattenIfEquation(original)
    local others = IR.IF_EQUATION[e for (j, e) in enumerate(ifEquations) if j != i]
    if isempty(eq.elseEquations)
      all(holdsNoEquations, b.equations for b in eq.branches) ||
        unsupported("an if-equation without else whose branches have equations", IR.toString(eq.branches[1].condition))
      continue
    end
    local branches = BRANCH[]
    local arms = vcat([(simExp(names, b.condition), b.equations, k) for (k, b) in enumerate(eq.branches)],
                      [(SCONST("ELSE_BRANCH"), eq.elseEquations, ELSE_BRANCH)])
    for (condition, equations, identifier) in arms
      local branchResiduals = RESIDUAL_EQUATION[]
      for e in equations
        if e isa IR.SCALAR_EQUATION
          push!(branchResiduals, residual(names, e))
        else
          @warn "Skipping unsupported $(nameof(typeof(e))) inside an if-equation's branch; the model may translate but lose this equation's effect"
        end
      end
      local graph = OrderedDict{Int, Vector{Int}}()
      for r in vcat(residuals, branchResiduals)
        graph[length(graph) + 1] = equationUnknowns(r.exp, ht)
      end
      addIfEquationRows!(graph, names, others, ht)
      addWhenRows!(graph, whens, ht)
      local (isSingular, matchOrder, digraph, sccs) = matchAndSort(graph)
      local target = identifier == ELSE_BRANCH ? ELSE_BRANCH : identifier + 1
      push!(branches, BRANCH(condition, branchResiduals, identifier, target, isSingular, matchOrder, digraph, sccs, ht))
    end
    push!(out, IF_EQUATION(branches))
  end
  return out
end

"A branch without equations: none, or if-equations without any (left by the assert hoisting)."
holdsNoEquations(body::Vector{IR.Equation})::Bool =
  all(e -> e isa IR.IF_EQUATION && all(b -> holdsNoEquations(b.equations), e.branches) && holdsNoEquations(e.elseEquations), body)

"""
    flattenIfEquation(eq)

An if-equation without nested ones: an else-branch that is one if-equation as further
branches (`else if` as `elseif`), and an if-equation in a branch as branches of the
conjoined conditions (`if a then (if b then E1 else E2)` as `if a and b then E1 elseif a and
not b then E2`), its equations in its place (the branches pair their equations by position).
"""
function flattenIfEquation(eq::IR.IF_EQUATION)::IR.IF_EQUATION
  return flattenNestedThenBranchIfs(flattenNestedElseIfChain(eq))
end

function flattenNestedElseIfChain(eq::IR.IF_EQUATION)::IR.IF_EQUATION
  (length(eq.elseEquations) == 1 && eq.elseEquations[1] isa IR.IF_EQUATION) || return eq
  local nested = flattenNestedElseIfChain(eq.elseEquations[1])
  return IR.IF_EQUATION(vcat(eq.branches, nested.branches), nested.elseEquations, eq.source, eq.attributes)
end

function flattenNestedThenBranchIfs(eq::IR.IF_EQUATION)::IR.IF_EQUATION
  local branches = IR.EquationBranch[]
  for b in eq.branches
    local expanded = expandBranchEquations(b.equations)
    if length(expanded) == 1 && expanded[1][1] === nothing
      push!(branches, b)
      continue
    end
    for (extra, eqs) in expanded
      push!(branches, IR.EquationBranch(extra === nothing ? b.condition : IR.LBINARY(b.condition, IR.AND, extra), eqs))
    end
  end
  local elseEquations = eq.elseEquations
  local elseExpanded = expandBranchEquations(eq.elseEquations)
  if !(length(elseExpanded) == 1 && elseExpanded[1][1] === nothing)
    local outerElse = negateAll(IR.Expression[b.condition for b in eq.branches])
    #= the last entry, every inner else, stays the else: the branches are evaluated in order =#
    for (extra, eqs) in elseExpanded[1:end-1]
      push!(branches, IR.EquationBranch(extra === nothing ? outerElse : IR.LBINARY(outerElse, IR.AND, extra), eqs))
    end
    elseEquations = last(elseExpanded)[2]
  end
  return IR.IF_EQUATION(branches, elseEquations, eq.source, eq.attributes)
end

"""
    expandBranchEquations(eqs)

A branch's equations over the if-equations in it: each nested one multiplies the branch into
one entry per inner branch, with the inner condition (the negation of all, for the inner
else) as the entry's extra guard, its equations spliced at its position. `nothing` as the
guard: no nested if-equation. The last entry is every nested one's else.
"""
function expandBranchEquations(eqs::Vector{IR.Equation})::Vector{Tuple{Union{Nothing, IR.Expression}, Vector{IR.Equation}}}
  local at = findfirst(e -> e isa IR.IF_EQUATION, eqs)
  at === nothing && return Tuple{Union{Nothing, IR.Expression}, Vector{IR.Equation}}[(nothing, eqs)]
  local inner = flattenIfEquation(eqs[at])
  local (before, after) = (eqs[1:at-1], eqs[at+1:end])
  local out = Tuple{Union{Nothing, IR.Expression}, Vector{IR.Equation}}[]
  for b in inner.branches
    for (extra, sub) in expandBranchEquations(vcat(before, b.equations, after))
      push!(out, (extra === nothing ? b.condition : IR.LBINARY(b.condition, IR.AND, extra), sub))
    end
  end
  local elseGuard = negateAll(IR.Expression[b.condition for b in inner.branches])
  for (extra, sub) in expandBranchEquations(vcat(before, inner.elseEquations, after))
    push!(out, (extra === nothing ? elseGuard : IR.LBINARY(elseGuard, IR.AND, extra), sub))
  end
  return out
end

"`not c1 and not c2 and ...` (left to right); `true` for none."
function negateAll(conditions::Vector{IR.Expression})::IR.Expression
  isempty(conditions) && return IR.BOOLEAN(true)
  local acc::IR.Expression = IR.LUNARY(IR.NOT, conditions[1])
  for c in conditions[2:end]
    acc = IR.LBINARY(acc, IR.AND, IR.LUNARY(IR.NOT, c))
  end
  return acc
end
