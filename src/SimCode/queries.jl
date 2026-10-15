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

#= Queries on the simulation code: its variables' categories, the references of its
   expressions and equations. =#

isState(v::SIMVAR)::Bool = v.varKind isa STATE
isAlgebraic(v::SIMVAR)::Bool = v.varKind isa ALG_VARIABLE
isStateOrAlgebraic(v::SIMVAR)::Bool = isAlgebraic(v) || isState(v)
isDiscrete(v::SIMVAR)::Bool = v.varKind isa DISCRETE
isParameter(v::SIMVAR)::Bool = v.varKind isa PARAMETER
"A parameter with a binding."
hasBindingExp(v::SIMVAR)::Bool = v.varKind isa PARAMETER && v.varKind.bindExp !== nothing

"An unknown: a state, a state derivative, an algebraic, an array or a discrete variable (not a parameter or a constant)."
isUnknownVarKind(k::SimVarType)::Bool = k isa Union{STATE, STATE_DERIVATIVE, ALG_VARIABLE, ARRAY, DISCRETE}

"The binding of a variable category that has one (a parameter's, a data structure's, an array's, a String's)."
bindingOf(k::SimVarType)::Union{Nothing, Exp} = hasproperty(k, :bindExp) ? k.bindExp : nothing

"A variable category with its binding replaced by `b` (of one that has a binding)."
withBinding(k::PARAMETER, b::Union{Nothing, Exp})::SimVarType = PARAMETER(b)
withBinding(k::DATA_STRUCTURE, b::Union{Nothing, Exp})::SimVarType = DATA_STRUCTURE(b)
withBinding(k::STRING, b::Union{Nothing, Exp})::SimVarType = STRING(b)
withBinding(k::ARRAY, b::Union{Nothing, Exp})::SimVarType = ARRAY(k.dimensions, b)
withBinding(k::ARRAY_PARAMETER, b::Union{Nothing, Exp})::SimVarType = ARRAY_PARAMETER(k.dimensions, b)
withBinding(k::SimVarType, ::Union{Nothing, Exp})::SimVarType = k

"A variable with another category."
withKind(v::SIMVAR, k::SimVarType)::SIMVAR = SIMVAR(v.name, v.index, k, v.attributes, v.elementType)
"A variable with the index `index` (and the category `k`)."
withIndex(v::SIMVAR, index::Int, k::SimVarType = v.varKind)::SIMVAR = SIMVAR(v.name, index, k, v.attributes, v.elementType)

hasStructuralTransitions(sc::SIM_CODE)::Bool = !isempty(sc.structuralTransitions)
hasSubModels(sc::SIM_CODE)::Bool = !isempty(sc.subModels)
hasFrontendModel(sc::SIM_CODE)::Bool = sc.frontendModel !== nothing

"Whether `e` reads an unknown (`isUnknownVarKind`)."
function hasUnknownCref(e::Exp, ht::HT)::Bool
  for n in crefNames!(OrderedSet{String}(), e)
    local entry = get(ht, n, nothing)
    (entry !== nothing && isUnknownVarKind(last(entry).varKind)) && return true
  end
  return false
end

"The name of a variable reference; `nothing` for another expression (an array element read by ASUB too)."
crefNameOf(e::Exp)::Union{Nothing, String} = e isa EXP_CREF ? crefName(e.cref) : nothing

"""
    complexFieldNames!(names, equations, ht)

For each reference of a record type in the equations, the names of its fields' variables
(`r_f`; of a reference whose `r_re` and `r_im` are variables, those): the code generation
flattens a record reference into its fields, which must stay.
"""
function complexFieldNames!(names::OrderedSet{String}, equations::AbstractVector{<:Equation}, ht::HT)::OrderedSet{String}
  local visit = x -> begin
    x isa EXP_CREF || return
    local base = crefName(x.cref)
    if x.ty isa TYPE_COMPLEX
      for f in x.ty.fieldNames
        local n = string(base, "_", f)
        haskey(ht, n) && push!(names, n)
      end
    else
      local (re, im) = (string(base, "_re"), string(base, "_im"))
      (haskey(ht, re) && haskey(ht, im)) && push!(names, re, im)
    end
  end
  for eq in equations
    for e in equationExps(eq)
      foreachExp(visit, e)
    end
  end
  return names
end

"The expressions of an equation (its sides; a residual's expression)."
equationExps(eq::RESIDUAL_EQUATION)::Vector{Exp} = Exp[eq.exp]
equationExps(eq::EQUATION)::Vector{Exp} = Exp[eq.lhs, eq.rhs]
equationExps(eq::ARRAY_EQUATION)::Vector{Exp} = Exp[eq.left, eq.right]
equationExps(eq::Equation)::Vector{Exp} = Exp[]

"""
    whenCrefNames!(names, w)

The references of a when-clause: its condition's, its statements' (an assignment's sides, a
reinit's state and value, a call's, an assert's condition and message), its elsewhen arms'.
"""
function whenCrefNames!(names::OrderedSet{String}, w::WHEN_STMTS)::OrderedSet{String}
  crefNames!(names, w.condition)
  for s in w.body
    if s isa ASSIGN
      crefNames!(names, s.left); crefNames!(names, s.right)
    elseif s isa REINIT
      push!(names, crefName(s.stateVar)); crefNames!(names, s.value)
    elseif s isa NORETCALL
      crefNames!(names, s.exp)
    elseif s isa ASSERT
      crefNames!(names, s.condition); crefNames!(names, s.message)
    end
  end
  w.elsewhenPart === nothing || whenCrefNames!(names, w.elsewhenPart)
  return names
end

"""
    statementCrefNames!(names, body)

The references of statements: their expressions' (an assert's level too), a reinit's state,
of the nested statements.
"""
function statementCrefNames!(names::OrderedSet{String}, body::Vector{Statement})::OrderedSet{String}
  for s in body
    statementCrefNames!(names, s)
  end
  return names
end

statementCrefNames!(names::OrderedSet{String}, s::ASSIGN) = (crefNames!(names, s.left); crefNames!(names, s.right))
statementCrefNames!(names::OrderedSet{String}, s::REINIT) = (push!(names, crefName(s.stateVar)); crefNames!(names, s.value))
statementCrefNames!(names::OrderedSet{String}, s::NORETCALL) = crefNames!(names, s.exp)
statementCrefNames!(names::OrderedSet{String}, s::ASSERT) =
  (crefNames!(names, s.condition); crefNames!(names, s.message); crefNames!(names, s.level))
statementCrefNames!(names::OrderedSet{String}, s::TERMINATE) = crefNames!(names, s.message)
statementCrefNames!(names::OrderedSet{String}, s::Union{IF, WHEN}) =
  (foreach(b -> (crefNames!(names, b.condition); statementCrefNames!(names, b.body)), s.branches);
   s isa IF && statementCrefNames!(names, s.elseBody); names)
statementCrefNames!(names::OrderedSet{String}, s::FOR) = (crefNames!(names, s.iterator.range); statementCrefNames!(names, s.body))
statementCrefNames!(names::OrderedSet{String}, s::WHILE) = (crefNames!(names, s.condition); statementCrefNames!(names, s.body))
statementCrefNames!(names::OrderedSet{String}, s::RECOMPILATION) = crefNames!(names, s.newValue)
statementCrefNames!(names::OrderedSet{String}, ::Statement) = names
