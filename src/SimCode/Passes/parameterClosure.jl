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

#= Algebraic variables that parameters alone define, folded into parameters. =#

"""
    foldParameterClosure(simCode)

The algebraic variables whose one defining residual (`v - f` or `f - v`) has an `f` of
parameters, constants and variables so folded already, as parameters bound to `f`, their
residuals removed; to a fixed point (a folded variable unlocks the next: `sd_max =
1/abs(aux1[1])` once `aux1[1]` is folded). MSL KinematicPTP's seven parameter closures reached
MTK as unknowns with a zero guess (`sqrt(1/0)` in the first Newton step). Not a variable of an
if-equation's condition (the condition is evaluated at the module's scope, where a folded
parameter is no name). The folded variables are recorded (`foldedVariables`): a pass that drops
one of them as a parameter keeps it observable. Nothing for a model with structural modes, without
residuals, or if the fold would remove every residual (MTK takes no empty system).
"""
function foldParameterClosure(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode) || isempty(simCode.residualEquations)) && return simCode
  local ht = simCode.stringToSimVarHT
  local excluded = Set{String}()
  for ifEq in simCode.ifEquations, b in ifEq.branches
    b.identifier == ELSE_BRANCH && continue
    union!(excluded, allCrefNames(b.condition))
  end
  local algNames = OrderedSet{String}(n for (n, (_, sv)) in ht if !(n in excluded) && sv.varKind isa ALG_VARIABLE)
  isempty(algNames) && return simCode
  local definingEquation = Dict{String, Int}()
  local definitions = Dict{String, Int}()
  for (i, eq) in enumerate(simCode.residualEquations)
    local n = definedName(eq.exp, algNames)
    n === nothing && continue
    definitions[n] = get(definitions, n, 0) + 1
    haskey(definingEquation, n) || (definingEquation[n] = i)
  end
  local folded = OrderedDict{String, Exp}()
  local removed = Set{Int}()
  local progressed = true
  while progressed
    progressed = false
    for (n, i) in definingEquation
      (haskey(folded, n) || definitions[n] != 1) && continue
      local f = parameterClosure(simCode.residualEquations[i].exp, n, ht, folded)
      f === nothing && continue
      folded[n] = f
      push!(removed, i)
      progressed = true
    end
  end
  isempty(folded) && return simCode
  length(simCode.residualEquations) == length(removed) && return simCode
  local newHT = copy(ht)
  for (n, f) in folded
    local (index, sv) = newHT[n]
    newHT[n] = (index, withKind(sv, PARAMETER(f)))
  end
  #= the variables the codegen marks irreducible or shares are no parameters =#
  local unfolded = v -> !haskey(folded, v)
  return update(simCode; stringToSimVarHT = newHT,
                residualEquations = RESIDUAL_EQUATION[eq for (i, eq) in enumerate(simCode.residualEquations) if !(i in removed)],
                irreducibleVariables = filter(unfolded, simCode.irreducibleVariables),
                sharedVariables = filter(unfolded, simCode.sharedVariables),
                foldedVariables = vcat(simCode.foldedVariables, collect(keys(folded))))
end

"""
    crefLikeName(e)

The name a variable reference or an element of an array variable by literal subscripts
(`ASUB(x, 1)` as `x[1]`) is known by; `nothing` for another expression.
"""
function crefLikeName(e::Exp)::Union{Nothing, String}
  e isa EXP_CREF && return crefName(e.cref)
  (e isa ASUB && e.exp isa EXP_CREF && !isempty(e.subs) && all(s -> s isa ICONST, e.subs)) || return nothing
  return elementKey(e.exp.cref, Int[(s::ICONST).value for s in e.subs])
end

"The name of `candidates` a residual `v - f` (or `f - v`) has as `v`; `nothing` for another residual."
function definedName(e::Exp, candidates::OrderedSet{String})::Union{Nothing, String}
  (e isa BINARY && e.op === OP_SUB) || return nothing
  for x in (e.exp1, e.exp2)
    local n = crefLikeName(x)
    (n !== nothing && n in candidates) && return n
  end
  return nothing
end

"""
    parameterClosure(e, name, ht, folded)

`f` of the residual `name - f` (or `f - name`) where every reference of `f` is a parameter, a
constant or a folded variable (not `name` itself, not a name the table does not know);
`nothing` for another residual.
"""
function parameterClosure(e::Exp, name::String, ht::HT, folded::AbstractDict{String, Exp})::Union{Nothing, Exp}
  (e isa BINARY && e.op === OP_SUB) || return nothing
  crefLikeName(e.exp1) == name && isParameterClosure(e.exp2, ht, folded, name) && return e.exp2
  crefLikeName(e.exp2) == name && isParameterClosure(e.exp1, ht, folded, name) && return e.exp1
  return nothing
end

function isParameterClosure(e::Exp, ht::HT, folded::AbstractDict{String, Exp}, exclude::String)::Bool
  for n in crefNames!(OrderedSet{String}(), e)
    n == exclude && return false
    haskey(folded, n) && continue
    local entry = get(ht, n, nothing)
    (entry === nothing || isUnknownVarKind(last(entry).varKind)) && return false
  end
  return true
end
