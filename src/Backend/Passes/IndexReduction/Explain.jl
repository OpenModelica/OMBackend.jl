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

#= What index reduction does to a model and why: the plan, reported (explain). =#

"A candidate of a block: a derivative, the variable it is the derivative of, its StateSelect (of the alias set, its own), whether that variable has a fixed start, and whether the derivative is a dummy."
struct ExplainedCandidate
  derivative::Symbol
  variable::Symbol
  setClass::IR.StateSelect
  ownClass::IR.StateSelect
  fixed::Bool
  dummy::Bool
end

"A block of the dummy derivatives' choice: its level (1 the highest derivatives), its equations (how often differentiated) and its candidates."
struct ExplainedBlock
  level::Int
  equations::Vector{Pair{String, Int}}
  candidates::Vector{ExplainedCandidate}
end

"""
    Explanation

What index reduction does to a model and why (`explain`): the `outcome` (`:index1` nothing to
reduce, `:reduced`, `:notHandled` with the `reason`: MTK reduces the index), the differentiated
equations and how often, the blocks of the dummy derivatives' choice that have one (more
candidates than equations), the states and the dynamic state sets: each set's candidates'
variables (its linked levels' after them) and the states the sets add (`setStates`,
`\$STATESET1.x[1]`).
"""
struct Explanation
  model::String
  outcome::Symbol
  reason::String
  differentiated::Vector{Pair{String, Int}}
  blocks::Vector{ExplainedBlock}
  states::Vector{Symbol}
  sets::Vector{Vector{Symbol}}
  setStates::Vector{Symbol}
end

Explanation(model::String, outcome::Symbol, reason::String)::Explanation =
  Explanation(model, outcome, reason, Pair{String, Int}[], ExplainedBlock[], Symbol[], Vector{Symbol}[], Symbol[])

"An equation as one line of text, at most `n` characters."
equationText(eq::IR.Equation; n::Int = 100)::String = first(replace(string(eq), r"\s*\n\s*" => " "), n)

"""
    explain(model) -> Explanation

What index reduction does to `model` (OMJLIR before index reduction: the passes run `through`
the last one; `OMBackend.explainIndexReduction` from a flat model) and why: planned as
`reduceIndex!` plans (its hidden linear constraints exposed, the aliases substituted, again with the simulation code's known values
where reduced), the model left as it is.
"""
function explain(model::IR.Model)::Explanation
  (!isempty(model.modes) || length(model.continuous) != 1) &&
    return Explanation(model.name, :notHandled, "a model with structural modes or several continuous partitions")
  local partition = model.continuous[1]
  local trace = BlockTrace[]
  local aliases = nothing
  local exposed = exposeLinearConstraints!(model, partition)
  try
    #= planned as `reduceIndex!` plans: as written, where reduced with the aliases substituted (`aliasPlan`) =#
    local p = try
      local written = plan(model, partition; trace)
      written === nothing ? nothing : ((q, aliases) = aliasPlan(model, partition, written; trace); q)
    catch err
      err isa NotHandled || rethrow()
      return Explanation(model.name, :notHandled, sprint(showerror, err))
    end
    p === nothing && return Explanation(model.name, :index1, "")
    local known = KNOWN_VALUES[] === nothing ? Dict{Symbol, IR.Expression}() : KNOWN_VALUES[](model)
    return explainPlan(model, partition, p, trace, known)
  finally
    #= the model restored =#
    aliases === nothing || undoAliases!(model, partition, aliases)
    undoLinearConstraints!(partition, exposed)
  end
end

"`explain`'s report of the plan `p` (its blocks `trace`, made with the values `known`)."
function explainPlan(model::IR.Model, partition::IR.Partition, p::Plan, trace::Vector{BlockTrace},
                     known::Dict{Symbol, IR.Expression})::Explanation
  local s = p.structure
  local folded = withKnownValues(model, partition; known)
  local natural = naturalDerivatives(s)
  local declared = declaredClasses(s, folded, model.variables, natural; dynamic = envSwitch("OMBACKEND_DYNAMIC_STATE_SETS"))
  local aliasClass = aliasClasses(s, folded, model.variables, declared)
  local differentiated = Pair{String, Int}[equationText(partition.equations[eq]) => length(chain)
                                           for (eq, chain) in sort(collect(p.differentiated); by = first)]
  local blocks = ExplainedBlock[]
  for b in trace
    length(b.candidates) > length(b.rows) || continue
    local equations = Pair{String, Int}[equationText(partition.equations[s.rows[r].equation]) => rowOrder(s, r) for r in b.rows]
    local candidates = ExplainedCandidate[]
    for c in b.candidates
      local a = s.antiderivative[c]
      push!(candidates, ExplainedCandidate(s.variables[c], a == 0 ? s.variables[c] : s.variables[a],
                                           a == 0 ? IR.DEFAULT : aliasClass[a], dummyClass(s, declared, c),
                                           fixedStart(s, model.variables, c), c in b.dummies || c in p.dummies))
    end
    push!(blocks, ExplainedBlock(b.level, equations, candidates))
  end
  local sets = Vector{Symbol}[Symbol[s.variables[s.antiderivative[c]] for c in vcat(set.candidates, set.linked...)] for set in p.sets]
  local setStates::Vector{Symbol} = Symbol[setStateCref(k, l, j).name for (k, set) in enumerate(p.sets) for l in 1:1+length(set.linked)
                           for j in 1:length(set.candidates)-size(set.jacobian, 1)]
  return Explanation(model.name, :reduced, "", differentiated, blocks, s.variables[selectedStates(s, p.dummies)], sets, setStates)
end

"`n` as how often: once, twice, `n` times."
timesText(n::Int)::String = n == 1 ? "once" : n == 2 ? "twice" : "$(n) times"

"A StateSelect in lower case (`default`)."
stateSelectText(c::IR.StateSelect)::String = lowercase(string(c))

function Base.show(io::IO, ::MIME"text/plain", e::Explanation)
  print(io, "Index reduction of ", e.model, ": ")
  if e.outcome === :index1
    println(io, "index 1, nothing to reduce")
    return
  elseif e.outcome === :notHandled
    println(io, "not handled (", e.reason, "): MTK's structural_simplify reduces the index")
    return
  end
  println(io, "index ", 1 + maximum(last, e.differentiated; init = 0), ", ", length(e.states) + length(e.setStates), " states")
  println(io, "  differentiated equations (", length(e.differentiated), "):")
  for (text, n) in first(e.differentiated, 20)
    println(io, "    ", rpad(text, 60), "  ", timesText(n))
  end
  length(e.differentiated) > 20 && println(io, "    ... ", length(e.differentiated) - 20, " more")
  for b in e.blocks
    local kept = filter(c -> !c.dummy, b.candidates)
    println(io, "  level ", b.level, ": ", length(b.equations), " equations, ", length(b.candidates), " candidates, ",
            length(kept), " kept")
    for (text, k) in first(b.equations, 4)
      println(io, "    eq ", k == 0 ? "" : "(differentiated $(timesText(k))) ", text)
    end
    length(b.equations) > 4 && println(io, "    ... ", length(b.equations) - 4, " more equations")
    #= the kept candidates (their variables states), then the dummies of their classes or higher: the competition =#
    local lowest = minimum(c -> (c.setClass, c.ownClass), kept; init = (IR.ALWAYS, IR.ALWAYS))
    local rivals = filter(c -> c.dummy && (c.setClass, c.ownClass) >= lowest, b.candidates)
    for c in vcat(kept, first(rivals, 8))
      println(io, "    ", c.dummy ? "dummy " : "state ", rpad(string(c.derivative), 44), " of ", rpad(string(c.variable), 36),
              " StateSelect ", stateSelectText(c.setClass), " (alias set), ", stateSelectText(c.ownClass), " (own)",
              c.fixed ? ", fixed start" : "")
    end
    length(rivals) > 8 && println(io, "    ... ", length(rivals) - 8, " more dummies of these classes")
  end
  println(io, "  states: ", join(String.(e.states), ", "))
  isempty(e.sets) || println(io, "  dynamic state sets: ", join(("{" * join(String.(set), ", ") * "}" for set in e.sets), ", "),
                              "; their states ", join(String.(e.setStates), ", "))
  return nothing
end
