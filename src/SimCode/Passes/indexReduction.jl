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

#= After OMBackend's index reduction (OMBACKEND_INDEX_REDUCTION, SIM_CODE.indexReduced). =#

"""
    derivativesOfDummyStates(simCode) -> SIM_CODE

`der(v)` of a variable that is no state (an initial equation's `der(flange.phi) = w_start`, the
alias elimination having replaced `flange.phi` by a dummy state) as v's derivative variable
`\$DER_v`, which index reduction made an unknown, or that variable's alias representative
(negated where the alias is). A state's `der` stays; a `der(v)` with no derivative variable
stays too (the code generators report it).
"""
function derivativesOfDummyStates(simCode::SIM_CODE)::SIM_CODE
  local ht = simCode.stringToSimVarHT
  local aliases = Dict{String, Tuple{String, Bool}}(e.eliminatedName => (e.representativeName, e.negated) for e in simCode.aliasMap)
  local reference = name -> EXP_CREF(SimCref(Symbol(name)), TYPE_REAL())
  local derivativeOf = name -> begin
    (haskey(ht, name) && last(ht[name]).varKind isa STATE) && return nothing
    local d = "\$DER_" * name
    haskey(ht, d) && return reference(d)
    local alias = get(aliases, d, nothing)
    alias === nothing && return nothing
    local (representative, negated) = alias
    return negated ? UNARY(OP_UMINUS, reference(representative)) : reference(representative)
  end
  local replaced = Ref(0)
  local f = e -> begin
    e isa DER || return e
    local d = derivativeOf(crefName(e.cref))
    d === nothing && return e
    replaced[] += 1
    return d
  end
  local recur = (x, a) -> (x isa Exp ? mapExp(f, x) : x isa Statement ? mapStatement(f, x) : x, a)
  local initialEquations = Equation[first(traverseFields(eq, recur, nothing)) for eq in simCode.initialEquations]
  replaced[] == 0 && return simCode
  @debug "[SIMCODE: $(simCode.name): derivativesOfDummyStates] $(replaced[]) der() of non-states replaced"
  return update(simCode; initialEquations)
end

"""
    unnegatedInitialEquations(simCode) -> SIM_CODE

An initial equation `-x = e` (alias elimination substituted a negated representative:
MultiPhase Rectifier's `supplyL.i[2] = -IDC` as `-sineVoltage.plug_p.pin[2].i = -IDC`) as
`x = -e`, `-(-c)` as `c`: the code generators read an initial equation's value by its
left-hand side's name.
"""
function unnegatedInitialEquations(simCode::SIM_CODE)::SIM_CODE
  local negated = e::Exp -> e isa UNARY && e.op === OP_UMINUS
  any(eq -> eq isa EQUATION && negated(eq.lhs), simCode.initialEquations) || return simCode
  local initialEquations = Equation[(eq isa EQUATION && negated(eq.lhs)) ?
    EQUATION(eq.lhs.exp, negated(eq.rhs) ? eq.rhs.exp : UNARY(OP_UMINUS, eq.rhs), eq.source, eq.attr) : eq
    for eq in simCode.initialEquations]
  return update(simCode; initialEquations)
end

"""
    unlinkUnreadStateDerivatives(simCode) -> SIM_CODE

The link `der(x) = \$DER_x` (index reduction's, the state's only `der(x)`) of a state whose
derivative variable nothing else reads any more, a fold having removed its readers (MSL Spice3's
`icapbx = capbx * der(vbx)`, capbx folded to 0): dropped with `\$DER_x`, so that the state is
dead (`demoteDeadStates`), as without index reduction.
"""
function unlinkUnreadStateDerivatives(simCode::SIM_CODE)::SIM_CODE
  local ht = simCode.stringToSimVarHT
  local link = eq::RESIDUAL_EQUATION -> begin
    local e = eq.exp
    (e isa BINARY && e.op === OP_SUB && e.exp1 isa DER && e.exp2 isa EXP_CREF) || return nothing
    local d = crefName(e.exp2.cref)
    (startswith(d, "\$DER_") && haskey(ht, d) && last(ht[d]).varKind isa ALG_VARIABLE) || return nothing
    return d
  end
  local links = Dict{Int, String}()
  for (i, eq) in enumerate(simCode.residualEquations)
    local d = link(eq)
    d === nothing || (links[i] = d)
  end
  isempty(links) && return simCode
  local reads = Dict{String, Int}()
  local visit = e::Exp -> foreachExp(x -> (x isa Union{EXP_CREF, PRE} && (reads[crefName(x.cref)] = get(reads, crefName(x.cref), 0) + 1)), e)
  local visitStatement = s::Statement -> mapStatementExps(e -> (visit(e); e), s)
  foreach(((i, eq),) -> haskey(links, i) || visit(eq.exp), enumerate(simCode.residualEquations))
  foreach(eq -> visit(eq.exp), simCode.eliminatedEquations)
  foreach(eq -> foreach(visit, equationExps(eq)), simCode.initialEquations)
  for ifEq in simCode.ifEquations, b in ifEq.branches
    visit(b.condition)
    foreach(eq -> visit(eq.exp), b.residualEquations)
  end
  for w in simCode.whenEquations
    local stmts = w.whenEquation
    while stmts !== nothing
      visit(stmts.condition)
      foreach(visitStatement, stmts.body)
      stmts = stmts.elsewhenPart
    end
  end
  foreach(a -> foreach(visitStatement, a.statements), simCode.initialAlgorithms)
  foreach(a -> (visit(a.condition); visit(a.message)), simCode.asserts)
  foreach(e -> (reads[e.representativeName] = get(reads, e.representativeName, 0) + 1), simCode.aliasMap)
  local unread = Set{Int}(i for (i, d) in links if get(reads, d, 0) == 0)
  isempty(unread) && return simCode
  local newHT = copy(ht)
  foreach(i -> delete!(newHT, links[i]), unread)
  @debug "[SIMCODE: $(simCode.name): unlinkUnreadStateDerivatives] $(sort([links[i] for i in unread]))"
  return update(simCode; stringToSimVarHT = newHT,
                residualEquations = RESIDUAL_EQUATION[eq for (i, eq) in enumerate(simCode.residualEquations) if !(i in unread)])
end
