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

#=
  OMJL's structural modes for the arrays-kept route: a model of explicit structural modes as one model per mode, its
  explicit transitions as structural when-equations of their modes.

  Author: John Tinnerholm
=#

"""
    isStructural(model) -> Bool

Whether `model` changes its structure during a simulation (OMJL): it has explicit structural modes (`structuralmode`),
or a structural when-equation (`recompilation`, `agentic_recompilation`).
"""
function isStructural(model::IR.Model)::Bool
  isempty(model.modes) || return true
  for p in model.continuous, eq in p.equations
    (eq isa IR.WHEN_EQUATION && eq.kind === IR.STRUCTURAL_WHEN) && return true
  end
  return false
end

"""
    modeModels(model) -> Vector{IR.Model}

The models of the structural modes of `model` (OMJL's `structuralmode`), in the modes' order: each the mode's variables
and system with the top level's (copies: the passes change them in place), named `<model>.<mode>`, without modes.
The explicit transitions from the mode (`structuralTransition(mode, to, condition)`) are its structural when-equations
of the switch (`IR.TRANSITION`, `transitionWhen`). A model whose top level is not one continuous partition, or with a
recompilation besides its modes, is not taken (UnsupportedLowering).
"""
function modeModels(model::IR.Model)::Vector{IR.Model}
  isempty(model.modes) && throw(ArgumentError("$(model.name): no structural modes"))
  (length(model.continuous) == 1 && isempty(model.clocked)) ||
    unsupported("structural modes in a model of several partitions or clocks", model.name)
  model.frontendModel === nothing || unsupported("structural modes and a recompilation in one model", model.name)
  local top::IR.Partition = model.continuous[1]
  local names::Set{String} = Set{String}(m.name for m in model.modes)
  for t::IR.StructuralTransition in model.transitions
    (t.from in names && t.to in names) || unsupported("a structural transition between modes the model does not have", t)
  end
  local out::Vector{IR.Model} = IR.Model[]
  for mode::IR.StructuralMode in model.modes
    local variables::IR.VariableSet = IR.VariableSet(IR.Variable[deepcopy(collect(mode.variables)); deepcopy(collect(model.variables))])
    local equations::Vector{IR.Equation} = IR.Equation[deepcopy(mode.system.equations); deepcopy(top.equations)]
    for t::IR.StructuralTransition in model.transitions
      t.from == mode.name && push!(equations, transitionWhen(t, variables))
    end
    local system::IR.Partition = IR.Partition(1, top.association, Symbol[mode.system.unknowns; top.unknowns], equations,
                                              IR.Equation[deepcopy(mode.system.initialEquations); deepcopy(top.initialEquations)])
    push!(out, IR.Model(string(model.name, ".", mode.name), variables, IR.Partition[system], IR.Partition[], IR.ClockKind[],
                        deepcopy(model.functions), nothing, IR.StructuralMode[], IR.StructuralTransition[], nothing))
  end
  return out
end

"""
    transitionWhen(t, variables) -> IR.WHEN_EQUATION

The explicit transition `t` as a structural when-equation of its mode (of the variables `variables`): the switch to
`t.to` (`IR.TRANSITION`) where its condition changes, either way, for a condition of time or of continuous variables
(`change(condition)`: as the ModelingToolkit route's continuous callback, `time <= 5` switches at 5), at the event
where it becomes true for one of discrete-time variables (discrete, or of an Integer, Boolean or enumeration type) and
parameters.
"""
function transitionWhen(t::IR.StructuralTransition, variables::IR.VariableSet)::IR.WHEN_EQUATION
  local continuous::Bool = false
  for c::IR.ComponentRef in IR.crefsOf(t.condition)
    local v::Union{Nothing, IR.Variable} = get(variables, IR.withoutSubscripts(c).name, nothing)
    continuous |= c.name === :time || v === nothing ||
                  !(v.kind isa Union{IR.DISCRETE, IR.PARAMETER, IR.CONSTANT} || isDiscreteType(v.ty))
  end
  local trigger::IR.Expression = continuous ? changeCall(t.condition) : t.condition
  return IR.WHEN_EQUATION(IR.WhenBranch[IR.WhenBranch(IR.Expression[trigger], false, false, IR.Statement[IR.TRANSITION(t.to)])],
                          IR.STRUCTURAL_WHEN, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)
end
