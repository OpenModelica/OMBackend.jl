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

#= Numeric evaluation of scalar OMJLIR expressions at the start values: the Jacobians state
   selection pivots on. Best effort: `nothing` where a value is not known here. =#

"""
    StartValues(variables, functions; parametersOnly = false, generic = false, equations = Equation[])

A variable's value at the start (`values(name)`): a parameter's or constant's binding (or, of
one without, the equation of `equations` that defines it or the record it is a field of,
`R0 = planarFrame(a)`: a binding that reads a parameter becomes an initial equation; or its start), a continuous variable's start attribute, 0 where none is
given (Modelica's default start); each evaluated once, recursively, a call of a function that
does not fold into one expression by running its algorithm (`interpretScalar`). With
`parametersOnly`, only the values known before the simulation: a parameter's or constant's
binding, through other such bindings; not a tunable parameter's, not one of `fixed = false`.
With `generic`, each unknown's start moved by its `genericOffset`: a point where the entries a
special start makes vanish (currents and fluxes all 0: MSL SMEE_Generator's `psi × i`) are not 0.
"""
struct StartValues
  variables::IR.VariableSet
  functions::Dict{Symbol, IR.ModelicaFunction}
  cache::Dict{Symbol, Union{Nothing, Float64}}
  parametersOnly::Bool
  generic::Bool
  "The right-hand sides of the equations that define parameters (or records of them) by name."
  defined::Dict{Symbol, IR.Expression}
  "Pure function calls interpreted so far: (function, arguments) to their outputs (`callValue`)."
  calls::Dict{Tuple{Symbol, Vector{Dict{String, Float64}}}, Union{Nothing, Vector{Dict{String, Float64}}}}
end

function StartValues(variables::IR.VariableSet, functions::Vector{IR.ModelicaFunction}; parametersOnly::Bool = false,
                     generic::Bool = false, equations::Vector{IR.Equation} = IR.Equation[])
  local defined = Dict{Symbol, IR.Expression}()
  for eq in equations
    (eq isa IR.SCALAR_EQUATION && eq.lhs isa IR.CREF) || continue
    local v = get(variables, eq.lhs.cref.name, nothing)
    (v === nothing || v.kind isa Union{IR.PARAMETER, IR.CONSTANT}) && (defined[eq.lhs.cref.name] = eq.rhs)
  end
  return StartValues(variables, Dict{Symbol, IR.ModelicaFunction}(f.name => f for f in functions),
                     Dict{Symbol, Union{Nothing, Float64}}(), parametersOnly, generic, defined,
                     Dict{Tuple{Symbol, Vector{Dict{String, Float64}}}, Union{Nothing, Vector{Dict{String, Float64}}}}())
end

"`values` at a generic point (`StartValues`' `generic`), a cache of its own (the pure calls' shared: their outputs depend on their arguments only)."
genericValues(values::StartValues)::StartValues =
  StartValues(values.variables, values.functions, Dict{Symbol, Union{Nothing, Float64}}(), values.parametersOnly, true,
              values.defined, values.calls)

"""
    definedValue(values, name) -> Union{Nothing, Float64}

The value of the parameter `name` by the equation that defines it, or the record it is a field
of (`R0 = planarFrame(a)` for `R0.T[1,2]`); `nothing` where none does or it does not evaluate.
"""
function definedValue(values::StartValues, name::Symbol)::Union{Nothing, Float64}
  isempty(values.defined) && return nothing
  local own = get(values.defined, name, nothing)
  own === nothing || return evaluate(own, values)
  local str = String(name)
  for i in Iterators.reverse(collect(eachindex(str)))
    str[i] in ('.', '[') || continue
    local whole = get(values.defined, Symbol(str[1:prevind(str, i)]), nothing)
    whole === nothing && continue
    local v = valueOf(whole, Interpretation(values))
    return v === nothing ? nothing : get(v, str[i:end], nothing)
  end
  return nothing
end

"""
    genericOffset(name) -> Float64

A fixed offset in (0.1, 0.9) for the unknown `name`, from its name (the same in every run): its
start moved by it is a generic point.
"""
function genericOffset(name::Symbol)::Float64
  local h = UInt64(7)
  for c in String(name)
    h = (h * 31 + UInt64(c)) % 1000003
  end
  return 0.1 + 0.8 * h / 1000003
end

function (sv::StartValues)(name::Symbol)::Union{Nothing, Float64}
  haskey(sv.cache, name) && return sv.cache[name]
  sv.cache[name] = nothing             # a cycle of bindings: unknown
  local v = get(sv.variables, name, nothing)
  local value = if v === nothing
    nothing
  elseif sv.parametersOnly
    isKnownParameter(v) ? evaluate(v.binding, sv) : nothing
  elseif v.kind isa Union{IR.PARAMETER, IR.CONSTANT}
    local d = v.binding === nothing ? definedValue(sv, name) : nothing
    local b = v.binding !== nothing ? v.binding : v.attributes.start
    d !== nothing ? d : b === nothing ? 0.0 : evaluate(b, sv)
  else
    local start = v.attributes.start === nothing ? 0.0 : evaluate(v.attributes.start, sv)
    (sv.generic && start !== nothing) ? start + genericOffset(name) : start
  end
  sv.cache[name] = value
  return value
end

"Whether `v` is a parameter or constant whose value its binding gives before the simulation."
isKnownParameter(v::IR.Variable)::Bool =
  v.kind isa Union{IR.PARAMETER, IR.CONSTANT} && v.binding !== nothing && !isTunableParameter(String(v.name)) &&
  !(v.attributes.fixed isa IR.BOOLEAN && !v.attributes.fixed.value)

"""
    withKnownValues(model, partition) -> Partition

A copy of `partition` with the values known before the simulation as literals, its scalar
arithmetic folded, so index reduction reads the structure the code generators see (they
substitute these values and Symbolics drops the zero terms): each parameter whose binding
evaluates (`StartValues` with `parametersOnly`: `revolute.e = {0, 0, 1}`, the revolute's
`R_rel.w[1] = e[1] * w` then reads no `w`), and each algebraic variable an equation sets to a
literal (`world.frame_b.R.w[1] = 0.0`), through the equations that then set another
(`bar1.frame_a.R.w[1] = world.frame_b.R.w[1]`), in every equation but the one that sets it (its
row). `known`: more variables' values (SimCode's), taken for a variable a function call defines
(`v = f(...)` or `(c, v) = f(...)`, the only equation with v on its left: a record function's output
SimCode's interpretation finds constant, which this folding cannot evaluate, MSL Spice3's clamp diodes'
`(cc, m_dCap) = diodeNoBypassCode(...)` without a junction capacitance; not the values SimCode infers
from the unreduced, singular system), kept in that equation. The equations are copies, the indices the partition's.
"""
function withKnownValues(model::IR.Model, partition::IR.Partition;
                         known::Dict{Symbol, IR.Expression} = Dict{Symbol, IR.Expression}())::IR.Partition
  local values = StartValues(model.variables, model.functions; parametersOnly = true)
  #= the values known from elsewhere (SimCode's folding: `knownValues`) of variables a function call defines (what
     this folding cannot evaluate: a record function's output), each kept in its defining equation =#
  local constants = functionDefinedValues(partition, known)
  local substitute = (e::IR.Expression, i::Int) -> begin
    if e isa IR.CREF
      local c = get(constants, e.cref.name, nothing)
      c === nothing || return c[2] == i ? e : c[1]
      return parameterLiteral(e, model.variables, values)
    end
    return foldScalar(e)
  end
  local equations = IR.Equation[deepcopy(eq) for eq in partition.equations]
  local readers = Dict{Symbol, Vector{Int}}()
  for (i, eq) in enumerate(equations)
    IR.mapEquation!(e -> substitute(e, i), eq)
    IR.foreachRootExp(eq) do root
      IR.foreachExp(x -> (x isa IR.CREF && push!(get!(Vector{Int}, readers, x.cref.name), i); nothing), root)
    end
  end
  local queue = collect(eachindex(equations))
  while !isempty(queue)
    local i = pop!(queue)
    local set = setsConstant(equations[i], model.variables)
    (set === nothing || haskey(constants, first(set))) && continue
    constants[first(set)] = (last(set), i)
    for j in get(readers, first(set), Int[])
      j == i && continue
      IR.mapEquation!(e -> substitute(e, j), equations[j])
      push!(queue, j)
    end
  end
  return IR.Partition(partition.index, partition.association, partition.unknowns, equations, partition.initialEquations)
end

"""
    functionDefinedValues(partition, known) -> Dict{Symbol, Tuple{Expression, Int}}

The values of `known` (SimCode's: `knownValues`) of the variables a function call defines, each with the index of
its defining equation: `v = f(...)` or `(c, v) = f(...)`, the only equation of `partition` with v on its left (of
two, a connector's `f = 0` and a body's `f = m * resolve2(...)`, the function's is the equation of motion, not the
variable's definition).
"""
function functionDefinedValues(partition::IR.Partition, known::Dict{Symbol, IR.Expression})::Dict{Symbol, Tuple{IR.Expression, Int}}
  local out = Dict{Symbol, Tuple{IR.Expression, Int}}()
  isempty(known) && return out
  local lefts = Dict{Symbol, Vector{Int}}()
  for (i, eq) in enumerate(partition.equations)
    eq isa IR.SCALAR_EQUATION || continue
    for target in (eq.lhs isa IR.TUPLE ? eq.lhs.elements : IR.Expression[eq.lhs])
      (target isa IR.CREF && haskey(known, target.cref.name)) && push!(get!(Vector{Int}, lefts, target.cref.name), i)
    end
  end
  for (name, is) in lefts
    (length(is) == 1 && callsFunction(partition.equations[only(is)].rhs)) || continue
    out[name] = (known[name], only(is))
  end
  return out
end

"Whether `e` calls a function that is no built-in."
function callsFunction(e::IR.Expression)::Bool
  local found = false
  IR.foreachExp(e) do x
    (x isa IR.CALL && !x.attributes.builtin) && (found = true)
    nothing
  end
  return found
end

"A parameter's reference as its literal where `values` knows its value; `e` else."
function parameterLiteral(e::IR.CREF, variables::IR.VariableSet, values::StartValues)::IR.Expression
  local v = get(variables, e.cref.name, nothing)
  (v === nothing || !(v.kind isa Union{IR.PARAMETER, IR.CONSTANT})) && return e
  local x = values(e.cref.name)
  x === nothing && return e
  v.ty isa IR.T_REAL && return IR.REAL(x)
  (v.ty isa IR.T_INTEGER && isinteger(x)) && return IR.INTEGER(Int(x))
  v.ty isa IR.T_BOOLEAN && return IR.BOOLEAN(x != 0)
  return e
end

"""
    setsConstant(eq, variables) -> Union{Nothing, Tuple{Symbol, Expression}}

The algebraic Real variable a scalar equation sets to a literal (`x = 0.0`, `1.0 = x`) and the
literal; `nothing` for another equation.
"""
function setsConstant(eq::IR.Equation, variables::IR.VariableSet)::Union{Nothing, Tuple{Symbol, IR.Expression}}
  eq isa IR.SCALAR_EQUATION || return nothing
  local (x, c) = isLiteral(eq.rhs) ? (eq.lhs, eq.rhs) : isLiteral(eq.lhs) ? (eq.rhs, eq.lhs) : (nothing, nothing)
  x isa IR.CREF || return nothing
  local v = get(variables, x.cref.name, nothing)
  (v !== nothing && v.kind isa IR.ALGEBRAIC && v.ty isa IR.T_REAL) || return nothing
  return (x.cref.name, c)
end

"""
    evaluate(e, values) -> Union{Nothing, Float64}

The value of the scalar expression `e`, its references by `values(name)` (`time` is 0);
`nothing` where it does not evaluate here (an array, an unknown function, an unknown value).
"""
function evaluate(e::IR.Expression, values::StartValues)::Union{Nothing, Float64}
  e isa IR.REAL && return e.value
  e isa IR.INTEGER && return Float64(e.value)
  e isa IR.BOOLEAN && return e.value ? 1.0 : 0.0
  e isa IR.ENUM_LITERAL && return Float64(e.index)
  if e isa IR.CREF
    e.cref.name === :time && return 0.0
    return values(e.cref.name)
  end
  if e isa IR.BINARY
    local a = evaluate(e.exp1, values)
    local b = evaluate(e.exp2, values)
    (a === nothing || b === nothing) && return nothing
    e.op in (IR.ADD, IR.ADD_EW) && return a + b
    e.op in (IR.SUB, IR.SUB_EW) && return a - b
    e.op in (IR.MUL, IR.MUL_EW) && return a * b
    e.op in (IR.DIV, IR.DIV_EW) && return b == 0 ? nothing : a / b
    e.op in (IR.POW, IR.POW_EW) && return (a < 0 && !isinteger(b)) ? nothing : a ^ b
    return nothing
  end
  if e isa IR.UNARY
    local a = evaluate(e.exp, values)
    return (a === nothing || e.op !== IR.UMINUS) ? nothing : -a
  end
  if e isa IR.IF_EXPRESSION
    local c = evaluateCondition(e.condition, values)
    c === nothing && return nothing
    return evaluate(c ? e.thenExp : e.elseExp, values)
  end
  e isa IR.CAST && return evaluate(e.exp, values)
  e isa IR.CALL && return evaluateCall(e, values)
  if e isa IR.SUBSCRIPTED
    local picked = selectElement(e.exp, e.subscripts, e.ty)
    return picked isa IR.SUBSCRIPTED ? interpretScalar(e, values) : evaluate(picked, values)
  end
  if e isa IR.RECORD_ELEMENT
    local record = asRecord(e.exp)
    return record === nothing ? interpretScalar(e, values) : evaluate(record.elements[e.index], values)
  end
  e isa IR.TUPLE_ELEMENT && return interpretScalar(e, values)
  return nothing
end

const UNARY_MATH = Dict{Symbol, Function}(:sin => sin, :cos => cos, :tan => tan, :asin => asin, :acos => acos,
  :atan => atan, :sinh => sinh, :cosh => cosh, :tanh => tanh, :exp => exp, :log => log, :log10 => log10,
  :sqrt => sqrt, :abs => abs, :sign => sign, :integer => floor, :floor => floor, :ceil => ceil)

"""
    evaluateCall(e, values) -> Union{Nothing, Float64}

A call's value at the start values: a user function by its body where that runs into one
expression (`inlineBody`), a built-in by its Julia counterpart; `nothing` for another call or an
argument without a value (a math function outside its domain).
"""
function evaluateCall(e::IR.CALL, values::StartValues)::Union{Nothing, Float64}
  if !e.attributes.builtin
    local f = get(values.functions, e.name, nothing)
    local body = f === nothing ? nothing : inlineBody(f, e)
    return body === nothing ? interpretScalar(e, values) : evaluate(body, values)
  end
  local n = e.name
  local args = e.arguments
  if haskey(UNARY_MATH, n) && length(args) == 1
    local a = evaluate(args[1], values)
    a === nothing && return nothing
    return try
      UNARY_MATH[n](a)
    catch err
      #= outside the function's domain at the start values (sqrt of a negative) =#
      err isa DomainError || rethrow()
      nothing
    end
  end
  n in (:noEvent, :homotopy) && return evaluate(args[1], values)
  n === :smooth && return evaluate(args[2], values)
  if n in (:min, :max, :atan2, :div, :mod, :rem) && length(args) == 2
    local (a, b) = (evaluate(args[1], values), evaluate(args[2], values))
    (a === nothing || b === nothing) && return nothing
    n === :min && return min(a, b)
    n === :max && return max(a, b)
    n === :atan2 && return atan(a, b)
    b == 0 && return nothing
    n === :div && return trunc(a / b)
    n === :mod && return a - floor(a / b) * b
    return a - trunc(a / b) * b
  end
  return nothing
end

"The value of a Boolean condition at the start; `nothing` where it does not evaluate."
function evaluateCondition(e::IR.Expression, values::StartValues)::Union{Nothing, Bool}
  e isa IR.BOOLEAN && return e.value
  if e isa IR.RELATION
    local (a, b) = (evaluate(e.exp1, values), evaluate(e.exp2, values))
    (a === nothing || b === nothing) && return nothing
    e.op === IR.LESS && return a < b
    e.op === IR.LESSEQ && return a <= b
    e.op === IR.GREATER && return a > b
    e.op === IR.GREATEREQ && return a >= b
    e.op === IR.EQUAL && return a == b
    e.op === IR.NEQUAL && return a != b
    return nothing
  end
  if e isa IR.LBINARY
    local (a, b) = (evaluateCondition(e.exp1, values), evaluateCondition(e.exp2, values))
    (a === nothing || b === nothing) && return nothing
    return e.op === IR.AND ? (a && b) : (a || b)
  end
  e isa IR.LUNARY && e.op === IR.NOT && (local a = evaluateCondition(e.exp, values); return a === nothing ? nothing : !a)
  e isa IR.CALL && e.attributes.builtin && e.name === :noEvent && return evaluateCondition(e.arguments[1], values)
  if e isa IR.CREF
    local v = evaluate(e, values)
    return v === nothing ? nothing : v != 0
  end
  return nothing
end
