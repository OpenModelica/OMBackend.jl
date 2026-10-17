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

#= pre() of parameters, and the unknowns equal to a parameter, propagated. =#

"""
    inlinePreOfConstantParameters(simCode)

The residuals with `pre(p)` (`previous(p)`) of a parameter `p` as `p`: a parameter's value
before an event is its value. `propagateConstants` then sees `x - pre(p)` as `x = p`.
"""
function inlinePreOfConstantParameters(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  local inline = e::Exp -> begin
    e isa PRE && isParameter(ht, e.cref) && return EXP_CREF(e.cref, e.ty)
    (e isa CALL && e.name in (:pre, :previous) && length(e.args) == 1 && e.args[1] isa EXP_CREF &&
      isParameter(ht, e.args[1].cref)) && return e.args[1]
    return e
  end
  return update(simCode; residualEquations = RESIDUAL_EQUATION[RESIDUAL_EQUATION(mapExp(inline, eq.exp), eq.source, eq.attr)
                                                               for eq in simCode.residualEquations])
end

"Whether `c` names a parameter of `ht`."
function isParameter(ht::HT, c::SimCref)::Bool
  local entry = get(ht, crefName(c), nothing)
  return entry !== nothing && isParameter(last(entry))
end

"""
    constantEquation(e, ht)

What the residual `e` says if it is `a - b` or `a + b` of two variables: `:trivial` if both
are parameters (or constants); `(:constant, unknown, parameter, negated)` if one is an unknown
and the other a parameter (`unknown = parameter`, `= -parameter` of `a + b`); `nothing` else
(of two unknowns: an alias).
"""
function constantEquation(e::Exp, ht::HT)::Union{Nothing, Symbol, Tuple{Symbol, String, EXP_CREF, Bool}}
  (e isa BINARY && (e.op === OP_SUB || e.op === OP_ADD) && e.exp1 isa EXP_CREF && e.exp2 isa EXP_CREF) || return nothing
  local (n1, n2) = (crefName(e.exp1.cref), crefName(e.exp2.cref))
  (haskey(ht, n1) && haskey(ht, n2)) || return nothing
  local unknown1 = isUnknownVarKind(last(ht[n1]).varKind)
  local unknown2 = isUnknownVarKind(last(ht[n2]).varKind)
  (unknown1 && unknown2) && return nothing
  (unknown1 || unknown2) || return :trivial
  return unknown1 ? (:constant, n1, e.exp2, e.op === OP_ADD) : (:constant, n2, e.exp1, e.op === OP_ADD)
end

"""
    propagateConstants(simCode)

The unknowns that a residual sets to a parameter (`u - p`, `u + p`) replaced by the parameter
(its negation) everywhere, to a fixed point (a replacement makes `v - u` read `v - p`), their
residuals removed; the residuals between parameters removed. Not a shared or irreducible
variable; not an element of an array whose whole is read (a call's argument: `resolve2(R_T)`).
When-clauses get their conditions substituted only (an assignment's target cannot be a
parameter). An eliminated unknown that is still read (by a when-clause's body) is kept, with
its residual substituted by the other replacements. The eliminated residuals and variables
are recorded for the observed equations. Nothing for a model with structural modes.
"""
function propagateConstants(simCode::SIM_CODE)::SIM_CODE
  (hasStructuralTransitions(simCode) || hasSubModels(simCode)) && return simCode
  local ht = simCode.stringToSimVarHT
  local excluded = Set{String}(Iterators.flatten((simCode.sharedVariables, simCode.irreducibleVariables)))
  local wholeArrays = Set{String}()
  for eq in simCode.residualEquations, n in crefNames!(OrderedSet{String}(), eq.exp)
    occursin('[', n) || push!(wholeArrays, n)
  end
  local replacements = Replacements()
  local eliminatedBy = OrderedDict{Int, String}()
  local trivial = Set{Int}()
  local residuals = simCode.residualEquations
  local changed = true
  while changed
    changed = false
    for (i, eq) in enumerate(residuals)
      (haskey(eliminatedBy, i) || i in trivial) && continue
      local c = constantEquation(eq.exp, ht)
      c === nothing && continue
      if c === :trivial
        push!(trivial, i)
        changed = true
        continue
      end
      local (_, unknown, parameter, negated) = c
      unknown in excluded && continue
      local bracket = findfirst('[', unknown)
      (bracket !== nothing && unknown[1:prevind(unknown, bracket)] in wholeArrays) && continue
      haskey(replacements, unknown) && continue
      replacements[unknown] = Replacement(parameter, negated, true)
      eliminatedBy[i] = unknown
      changed = true
    end
    changed && (residuals = RESIDUAL_EQUATION[substitute(eq, replacements) for eq in residuals])
  end
  (isempty(eliminatedBy) && isempty(trivial)) && return simCode
  #= the remaining residuals, if-equations, initial equations substituted; their references =#
  local read = OrderedSet{String}()
  local newResiduals = RESIDUAL_EQUATION[]
  local pairs = Tuple{String, RESIDUAL_EQUATION, RESIDUAL_EQUATION}[]
  for (i, eq) in enumerate(simCode.residualEquations)
    if haskey(eliminatedBy, i)
      push!(pairs, (eliminatedBy[i], eq, substitute(eq, replacements)))
    elseif !(i in trivial)
      local s = substitute(eq, replacements)
      crefNames!(read, s.exp)
      push!(newResiduals, s)
    end
  end
  local ifEquations = IF_EQUATION[substitute(ifEq, replacements) for ifEq in simCode.ifEquations]
  for ifEq in ifEquations, b in ifEq.branches, eq in b.residualEquations
    crefNames!(read, eq.exp)
  end
  local initialEquations = Equation[substituteInitial(eq, replacements) for eq in simCode.initialEquations]
  for eq in initialEquations, e in equationExps(eq)
    crefNames!(read, e)
  end
  local whenEquations = WHEN_EQUATION[WHEN_EQUATION(WHEN_STMTS(substitute(w.whenEquation.condition, replacements),
                                                               w.whenEquation.body, w.whenEquation.elsewhenPart),
                                                    w.source, w.attr) for w in simCode.whenEquations]
  foreach(w -> whenCrefNames!(read, w.whenEquation), whenEquations)
  #= an eliminated unknown still read is kept, with its residual (by the other replacements) =#
  local stillRead = Set{String}(n for n in keys(replacements) if n in read)
  isempty(stillRead) || @debug "[SIMCODE: $(simCode.name): propagateConstants] eliminated variables still read, kept" stillRead
  local others = isempty(stillRead) ? replacements : Replacements(n => r for (n, r) in replacements if !(n in stillRead))
  local newHT = copy(ht)
  local eliminatedVariables = String[]
  local eliminatedEquations = RESIDUAL_EQUATION[]
  for (name, original, substituted) in pairs
    if name in stillRead
      push!(newResiduals, substitute(original, others))
    elseif !haskey(newHT, name)
      push!(newResiduals, substituted)
    else
      delete!(newHT, name)
      push!(eliminatedVariables, name)
      push!(eliminatedEquations, original)
    end
  end
  return update(simCode; residualEquations = newResiduals, initialEquations, stringToSimVarHT = newHT, ifEquations,
                whenEquations, asserts = ASSERT_EQUATION[substitute(a, replacements) for a in simCode.asserts],
                initialAlgorithms = INITIAL_ALGORITHM[substitute(a, replacements) for a in simCode.initialAlgorithms],
                eliminatedEquations = vcat(simCode.eliminatedEquations, eliminatedEquations),
                eliminatedVariables = vcat(simCode.eliminatedVariables, eliminatedVariables))
end
