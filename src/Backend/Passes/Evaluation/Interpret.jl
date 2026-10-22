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

#= Numeric evaluation of what does not fold into one expression: a function's algorithm run on
   numbers (`callValue`), arrays and records. A value is flat: its scalars by the suffix of their
   name, as the scalarized model and the function bodies name them (`""` a scalar, `"[1,2]"` an
   array's element, `".T[1,1]"` an element of a record's field). Integers, Booleans and
   enumeration literals are numbers. Anything not handled gives `nothing`, never a throw: an
   external function, a String, a clock, an array of records in array arithmetic, a function as a
   function's output, a when-statement or reinit (not in a function). =#

"A flat value: its scalars by the suffix of their name (`\"\"` a scalar, `\"[i,j]\"` an array's element, `\".f\"` a record's field)."
const FlatValue = Dict{String, Float64}

"The flat value of a scalar."
scalarValue(x::Float64)::FlatValue = FlatValue("" => x)

"The scalar of a flat value; `nothing` for an array or a record."
scalarOf(v::FlatValue)::Union{Nothing, Float64} = (length(v) == 1 && haskey(v, "")) ? v[""] : nothing

"The statements one interpretation runs at most, and the calls it nests at most (a loop or a recursion that does not end)."
const MAX_INTERPRETED_STEPS = 100_000
const MAX_INTERPRETED_DEPTH = 32

"A function passed as an argument (`PARTIAL_APPLICATION`, MLS 12.4.2.1): the function and the inputs it binds by name."
struct PartialCall
  name::Symbol
  bound::Vector{Pair{Symbol, FlatValue}}
end

"""
    Interpretation(values, scalars, functionArguments, steps, depth, inFunction)

One numeric evaluation: `scalars` the values of the running function's variables by flat name
(`T[1,1]`, `R.T[2,3]`, `Q[4]`, a for-loop's iterator), `functionArguments` its inputs that are
functions; outside a function (`inFunction` false) other names are the model's, by `values`
(their start). `steps` the statements run so far, `depth` the calls nested.
"""
mutable struct Interpretation
  values::StartValues
  scalars::Dict{Symbol, Float64}
  functionArguments::Dict{Symbol, PartialCall}
  steps::Int
  depth::Int
  inFunction::Bool
end

"An interpretation of an expression of the model (not in a function), its names read at `values`."
Interpretation(values::StartValues) = Interpretation(values, Dict{Symbol, Float64}(), Dict{Symbol, PartialCall}(), 0, 0, false)

"""
    interpretScalar(e, values) -> Union{Nothing, Float64}

The value of the scalar expression `e` by numeric interpretation (`valueOf`): a call of a
function whose algorithm does not run into one expression (MSL Frames.to_Q), an element of an
array or a record such a call returns; `nothing` where it does not evaluate.
"""
function interpretScalar(e::IR.Expression, values::StartValues)::Union{Nothing, Float64}
  local v = valueOf(e, Interpretation(values))
  return v === nothing ? nothing : scalarOf(v)
end

"""
    valueOf(e, run) -> Union{Nothing, FlatValue}

The value of the expression `e` in `run`: literals, references (the running function's
variables, or outside a function the model's start values, a whole record or array element by
element, a slice of one), array and record literals, calls (built-ins, record constructors,
functions by their algorithm, a function input), tuple elements, subscripts (indices, slices,
whole dimensions; a record's field by position) and record fields, arithmetic on scalars and
arrays, relations, logic, if-expressions, ranges, `size`, array constructors and reductions
(sum, product, min, max) of any number of iterators.
"""
function valueOf(e::IR.Expression, run::Interpretation)::Union{Nothing, FlatValue}
  e isa IR.REAL && return scalarValue(e.value)
  e isa IR.INTEGER && return scalarValue(Float64(e.value))
  e isa IR.BOOLEAN && return scalarValue(e.value ? 1.0 : 0.0)
  e isa IR.ENUM_LITERAL && return scalarValue(Float64(e.index))
  e isa IR.CREF && return crefValue(e, run)
  e isa IR.CAST && return valueOf(e.exp, run)
  e isa IR.ARRAY && return elementsValue(e.elements, run)
  if e isa IR.RECORD
    local out = FlatValue()
    for (field, x) in zip(e.ty.fields, e.elements)
      local v = valueOf(x, run)
      v === nothing && return nothing
      for (s, y) in v
        out[string(".", field.name, s)] = y
      end
    end
    return out
  end
  if e isa IR.CALL
    local record = asRecord(e)
    record === nothing || return valueOf(record, run)
    e.attributes.builtin && return builtinValue(e, run)
    local outputs = callValue(e, run)
    return (outputs === nothing || isempty(outputs)) ? nothing : outputs[1]
  end
  if e isa IR.TUPLE_ELEMENT
    (e.exp isa IR.CALL && !e.exp.attributes.builtin) || return nothing
    local outputs = callValue(e.exp, run)
    return (outputs === nothing || e.index > length(outputs)) ? nothing : outputs[e.index]
  end
  if e isa IR.SUBSCRIPTED
    local v = valueOf(e.exp, run)
    v === nothing && return nothing
    local index = subscriptIndices(e.subscripts, run)
    index === nothing && return selectValue(v, e.subscripts, run)
    #= a record subscripted by position: its field (a record binding split by field, `planarFrame(a)[1][1, 2]`) =#
    local ty = IR.typeOf(e.exp)
    if ty isa IR.T_RECORD
      (length(index) == 1 && 1 <= index[1] <= length(ty.fields)) || return nothing
      return flatField(v, ty.fields[index[1]].name)
    end
    return elementOf(v, index)
  end
  if e isa IR.RECORD_ELEMENT
    local v = valueOf(e.exp, run)
    return v === nothing ? nothing : flatField(v, e.field)
  end
  if e isa IR.UNARY
    local v = valueOf(e.exp, run)
    (v === nothing || e.op !== IR.UMINUS) && return nothing
    return FlatValue(s => -y for (s, y) in v)
  end
  if e isa IR.BINARY
    local (a, b) = (valueOf(e.exp1, run), valueOf(e.exp2, run))
    (a === nothing || b === nothing) && return nothing
    return binaryValue(e.op, a, b)
  end
  if e isa Union{IR.RELATION, IR.LBINARY}
    local (a, b) = (valueOf(e.exp1, run), valueOf(e.exp2, run))
    (a === nothing || b === nothing) && return nothing
    local (x, y) = (scalarOf(a), scalarOf(b))
    (x === nothing || y === nothing) && return nothing
    local holds = comparison(e.op, x, y)
    return holds === nothing ? nothing : scalarValue(holds ? 1.0 : 0.0)
  end
  if e isa IR.LUNARY
    local v = valueOf(e.exp, run)
    local x = v === nothing ? nothing : scalarOf(v)
    return (x === nothing || e.op !== IR.NOT) ? nothing : scalarValue(x == 0 ? 1.0 : 0.0)
  end
  if e isa IR.IF_EXPRESSION
    local c = valueOf(e.condition, run)
    local x = c === nothing ? nothing : scalarOf(c)
    x === nothing && return nothing
    return valueOf(x != 0 ? e.thenExp : e.elseExp, run)
  end
  if e isa IR.RANGE
    local range = rangeValues(e, run)
    return range === nothing ? nothing : fromArray(range)
  end
  e isa IR.ARRAY_CONSTRUCTOR && return constructorValue(e, run)
  e isa IR.REDUCTION && return reductionValue(e, run)
  if e isa IR.SIZE
    local v = valueOf(e.exp, run)
    local A = v === nothing ? nothing : toArray(v)
    A === nothing && return nothing
    e.dimension === nothing && return fromArray(Float64[size(A)...])
    local d = valueOf(e.dimension, run)
    local k = d === nothing ? nothing : scalarOf(d)
    (k === nothing || !isinteger(k) || !(1 <= k <= ndims(A))) && return nothing
    return scalarValue(Float64(size(A, Int(k))))
  end
  return nothing
end

"The flat value of an array literal's elements (`[i]` before each element's suffix)."
function elementsValue(elements::Vector{IR.Expression}, run::Interpretation)::Union{Nothing, FlatValue}
  local out = FlatValue()
  for (i, x) in enumerate(elements)
    local v = valueOf(x, run)
    v === nothing && return nothing
    for (s, y) in v
      out[isempty(s) ? "[$i]" : startswith(s, "[") ? string("[", i, ",", SubString(s, 2)) : string("[", i, "]", s)] = y
    end
  end
  return out
end

"""
    crefValue(e, run) -> Union{Nothing, FlatValue}

A reference's value: its name with the subscripts evaluated (`c[k]`, `R.T[i,i]` in a loop); in a
function, its variables' scalars (a whole array or record by its elements); outside one, the
model's start value (`time` 0), a whole record or array of the model element by element.
"""
function crefValue(e::IR.CREF, run::Interpretation)::Union{Nothing, FlatValue}
  local name = resolvedName(e.cref, run)
  if name === nothing
    #= a slice or a whole dimension (`x[2:3]`, `A[:, 1]`) of the last part: the selection of the whole =#
    local last = e.cref.parts[end]
    (isempty(last.subscripts) || any(p -> !isempty(p.subscripts), e.cref.parts[1:end - 1])) && return nothing
    local whole = IR.CREF(IR.ComponentRef(IR.CrefPart[e.cref.parts[1:end - 1]..., IR.CrefPart(last.name, IR.Subscript[], last.ty)]), last.ty)
    local v = crefValue(whole, run)
    return v === nothing ? nothing : selectValue(v, last.subscripts, run)
  end
  local x = get(run.scalars, Symbol(name), nothing)
  x === nothing || return scalarValue(x)
  local whole = prefixValue(run.scalars, name)
  isempty(whole) || return whole
  run.inFunction && return nothing
  name == "time" && return scalarValue(0.0)
  e.ty isa Union{IR.T_RECORD, IR.T_ARRAY} && return wholeValue(name, e.ty, run.values)
  local v = run.values(Symbol(name))
  return v === nothing ? nothing : scalarValue(v)
end

"The flat name of a reference, its subscripts evaluated in `run`; `nothing` where one does not evaluate to an index."
function resolvedName(cref::IR.ComponentRef, run::Interpretation)::Union{Nothing, String}
  all(p -> isempty(p.subscripts), cref.parts) && return String(cref.name)
  local io = IOBuffer()
  for (i, p) in enumerate(cref.parts)
    i > 1 && print(io, '.')
    print(io, p.name)
    isempty(p.subscripts) && continue
    local index = subscriptIndices(p.subscripts, run)
    index === nothing && return nothing
    print(io, '[', join(index, ','), ']')
  end
  return String(take!(io))
end

"The indices of subscripts that are single indices; `nothing` for a slice, a whole dimension or an index that does not evaluate."
function subscriptIndices(subscripts::Vector{IR.Subscript}, run::Interpretation)::Union{Nothing, Vector{Int}}
  local out = Int[]
  for sub in subscripts
    sub isa IR.INDEX || return nothing
    local v = valueOf(sub.exp, run)
    local x = v === nothing ? nothing : scalarOf(v)
    (x === nothing || !isinteger(x)) && return nothing
    push!(out, Int(x))
  end
  return out
end

"""
    selections(subscripts, run) -> Union{Nothing, Vector{Union{Int, Vector{Int}, Colon}}}

Subscripts as array selections: an index, the indices of a slice (`2:3`, `{1, 3}`), all of a
dimension (`:`); `nothing` where one does not evaluate.
"""
function selections(subscripts::Vector{IR.Subscript}, run::Interpretation)::Union{Nothing, Vector{Union{Int, Vector{Int}, Colon}}}
  local out = Union{Int, Vector{Int}, Colon}[]
  for sub in subscripts
    if sub isa IR.WHOLE
      push!(out, Colon())
      continue
    end
    local v = valueOf(sub.exp, run)
    v === nothing && return nothing
    local x = scalarOf(v)
    if sub isa IR.INDEX && x !== nothing
      isinteger(x) || return nothing
      push!(out, Int(x))
    else
      local A = toArray(v)
      (A === nothing || ndims(A) != 1 || !all(isinteger, A)) && return nothing
      push!(out, Int.(A))
    end
  end
  return out
end

"The selection `subscripts` of an array's flat value (slices and whole dimensions; an index drops its dimension, as in Modelica)."
function selectValue(v::FlatValue, subscripts::Vector{IR.Subscript}, run::Interpretation)::Union{Nothing, FlatValue}
  local A = toArray(v)
  local sel = selections(subscripts, run)
  (A === nothing || sel === nothing || length(sel) > ndims(A)) && return nothing
  local full = vcat(sel, fill(Colon(), ndims(A) - length(sel)))
  all(((d, s),) -> s isa Colon || all(k -> 1 <= k <= size(A, d), s), enumerate(full)) || return nothing
  local r = A[full...]
  return r isa Number ? scalarValue(Float64(r)) : fromArray(r)
end

"The items an iterator runs over (a range, an array's elements)."
function iteratorItems(it::IR.Iterator, run::Interpretation)::Union{Nothing, Vector{Float64}}
  local v = valueOf(it.range, run)
  local A = v === nothing ? nothing : toArray(v)
  return (A === nothing || ndims(A) != 1) ? nothing : vec(A)
end

"""
    iterate!(visit, iterators, run) -> Bool

`visit(position)` for each combination of the iterators' items (the first iterator outermost),
each iterator's value in `run` meanwhile (restored after), `position` the items' indices;
whether every range evaluated and every visit returned true.
"""
function iterate!(visit::F, iterators::Vector{IR.Iterator}, run::Interpretation)::Bool where {F}
  local items = Vector{Float64}[]
  for it in iterators
    local xs = iteratorItems(it, run)
    xs === nothing && return false
    push!(items, xs)
  end
  local previous = Union{Nothing, Float64}[get(run.scalars, it.name, nothing) for it in iterators]
  local ok = true
  for I in CartesianIndices(Tuple(length.(items)))
    for (k, it) in enumerate(iterators)
      run.scalars[it.name] = items[k][I[k]]
    end
    visit(collect(Tuple(I))) || (ok = false; break)
  end
  for (it, x) in zip(iterators, previous)
    x === nothing ? delete!(run.scalars, it.name) : (run.scalars[it.name] = x)
  end
  return ok
end

"A reduction `sum`, `product`, `min` or `max` of its body over its iterators (of an empty range: 0, 1; min and max none)."
function reductionValue(e::IR.REDUCTION, run::Interpretation)::Union{Nothing, FlatValue}
  local op = e.reducer === :sum ? IR.ADD : e.reducer === :product ? IR.MUL_EW : nothing
  local acc = Ref{Union{Nothing, FlatValue}}(nothing)
  local ok = iterate!(e.iterators, run) do _
    local v = valueOf(e.body, run)
    v === nothing && return false
    if acc[] === nothing
      acc[] = v
    elseif op !== nothing
      acc[] = binaryValue(op, acc[], v)
    else
      local (x, y) = (scalarOf(acc[]), scalarOf(v))
      (x === nothing || y === nothing) && return false
      acc[] = scalarValue(e.reducer === :min ? min(x, y) : max(x, y))
    end
    return acc[] !== nothing
  end
  ok || return nothing
  acc[] === nothing && return e.reducer === :sum ? scalarValue(0.0) : e.reducer === :product ? scalarValue(1.0) : nothing
  return acc[]
end

"The scalars under `name` (`name[...]`, `name.field`) as a flat value of their suffixes."
function prefixValue(scalars::Dict{Symbol, Float64}, name::String)::FlatValue
  local out = FlatValue()
  for (k, x) in scalars
    local s = String(k)
    (length(s) > length(name) && startswith(s, name) && s[length(name) + 1] in ('[', '.')) || continue
    out[s[length(name) + 1:end]] = x
  end
  return out
end

"""
    wholeValue(name, ty, values) -> Union{Nothing, FlatValue}

A whole record or array of the model (a reference the scalarized model keeps as a function's
argument, `Frames.to_Q(body.R_start, ...)`) from the start values of its scalars.
"""
function wholeValue(name::String, ty::IR.ModelicaType, values::StartValues)::Union{Nothing, FlatValue}
  local out = FlatValue()
  return collectWhole!(out, name, "", ty, values) ? out : nothing
end

"`collectWhole!`'s step: the scalars of `name * suffix` of type `ty` into `out`; whether all evaluated."
function collectWhole!(out::FlatValue, name::String, suffix::String, ty::IR.ModelicaType, values::StartValues)::Bool
  if ty isa IR.T_RECORD
    return all(f -> collectWhole!(out, name, string(suffix, ".", f.name), f.ty, values), ty.fields)
  elseif ty isa IR.T_ARRAY
    local sizes = Int[]
    for d in IR.dimensions(ty)
      local n = IR.dimensionSize(d)
      n === nothing && return false
      push!(sizes, n)
    end
    return all(I -> collectWhole!(out, name, string(suffix, "[", join(Tuple(I), ','), "]"), IR.elementType(ty), values),
               CartesianIndices(Tuple(sizes)))
  end
  local x = values(Symbol(name, suffix))
  x === nothing && return false
  out[suffix] = x
  return true
end

"The element of a flat value at `index` (all its dimensions: a scalar; fewer: the rest of the array, or a record element's fields)."
function elementOf(v::FlatValue, index::Vector{Int})::Union{Nothing, FlatValue}
  local p = string("[", join(index, ','))
  local out = FlatValue()
  for (s, y) in v
    if s == p * "]"
      out[""] = y
    elseif startswith(s, p * ",")
      out[string("[", SubString(s, length(p) + 2))] = y
    elseif startswith(s, p * "]")
      out[s[length(p) + 2:end]] = y
    end
  end
  return isempty(out) ? nothing : out
end

"The field `field` of a record's flat value."
function flatField(v::FlatValue, field::Symbol)::Union{Nothing, FlatValue}
  local p = string(".", field)
  local out = FlatValue()
  for (s, y) in v
    if s == p
      out[""] = y
    elseif startswith(s, p) && s[length(p) + 1] in ('[', '.')
      out[s[length(p) + 1:end]] = y
    end
  end
  return isempty(out) ? nothing : out
end

"""
    toArray(v) -> Union{Nothing, Array{Float64}}

A flat value of array elements (`[i,j]` suffixes, every index present) as an array; `nothing`
for a scalar, a record or an incomplete array.
"""
function toArray(v::FlatValue)::Union{Nothing, Array{Float64}}
  isempty(v) && return nothing
  local entries = Tuple{Vector{Int}, Float64}[]
  for (s, y) in v
    (startswith(s, "[") && endswith(s, "]")) || return nothing
    local index = Int[]
    for p in split(SubString(s, 2, lastindex(s) - 1), ',')
      local k = tryparse(Int, p)
      (k === nothing || k < 1) && return nothing
      push!(index, k)
    end
    push!(entries, (index, y))
  end
  local n = length(first(first(entries)))
  all(e -> length(first(e)) == n, entries) || return nothing
  local sizes = Tuple(maximum(first(e)[d] for e in entries) for d in 1:n)
  prod(sizes) == length(entries) || return nothing
  local A = zeros(sizes)
  for (index, y) in entries
    A[index...] = y
  end
  return A
end

"An array as a flat value (`[i,j]` suffixes)."
function fromArray(A::AbstractArray{<:Real})::FlatValue
  return FlatValue(string("[", join(Tuple(I), ','), "]") => Float64(A[I]) for I in CartesianIndices(A))
end

"""
    binaryValue(op, a, b) -> Union{Nothing, FlatValue}

Arithmetic on flat values: of two scalars; a scalar with an array element by element; two
arrays element by element, or their matrix, matrix-vector, vector-matrix or scalar product.
"""
function binaryValue(op::IR.Op, a::FlatValue, b::FlatValue)::Union{Nothing, FlatValue}
  local (x, y) = (scalarOf(a), scalarOf(b))
  if x !== nothing && y !== nothing
    local z = arithmetic(op, x, y)
    return z === nothing ? nothing : scalarValue(z)
  end
  if op in (IR.MATRIX_PRODUCT, IR.MUL_MATRIX_VECTOR, IR.MUL_VECTOR_MATRIX, IR.SCALAR_PRODUCT) ||
     (op === IR.MUL && x === nothing && y === nothing)
    local (A, B) = (toArray(a), toArray(b))
    (A === nothing || B === nothing) && return nothing
    local (m, n) = (ndims(A), ndims(B))
    if m == 1 && n == 1
      length(A) == length(B) || return nothing
      return scalarValue(sum(A .* B))
    elseif m == 2 && n == 2
      size(A, 2) == size(B, 1) || return nothing
      return fromArray(A * B)
    elseif m == 2 && n == 1
      size(A, 2) == length(B) || return nothing
      return fromArray(A * B)
    elseif m == 1 && n == 2
      length(A) == size(B, 1) || return nothing
      return fromArray(vec(permutedims(B) * A))
    end
    return nothing
  end
  local out = FlatValue()
  if x !== nothing
    for (s, w) in b
      local z = arithmetic(op, x, w)
      z === nothing && return nothing
      out[s] = z
    end
  elseif y !== nothing
    for (s, w) in a
      local z = arithmetic(op, w, y)
      z === nothing && return nothing
      out[s] = z
    end
  else
    keys(a) == keys(b) || return nothing
    for (s, w) in a
      local z = arithmetic(op, w, b[s])
      z === nothing && return nothing
      out[s] = z
    end
  end
  return out
end

"A scalar operation by the operator of any shape (`ADD`, `ADD_EW`, `ADD_SCALAR_ARRAY`, ...); `nothing` outside its domain."
function arithmetic(op::IR.Op, x::Float64, y::Float64)::Union{Nothing, Float64}
  op in (IR.ADD, IR.ADD_EW, IR.ADD_SCALAR_ARRAY, IR.ADD_ARRAY_SCALAR) && return x + y
  op in (IR.SUB, IR.SUB_EW, IR.SUB_SCALAR_ARRAY, IR.SUB_ARRAY_SCALAR) && return x - y
  op in (IR.MUL, IR.MUL_EW, IR.MUL_SCALAR_ARRAY, IR.MUL_ARRAY_SCALAR, IR.SCALAR_PRODUCT) && return x * y
  op in (IR.DIV, IR.DIV_EW, IR.DIV_SCALAR_ARRAY, IR.DIV_ARRAY_SCALAR) && return y == 0 ? nothing : x / y
  op in (IR.POW, IR.POW_EW, IR.POW_SCALAR_ARRAY, IR.POW_ARRAY_SCALAR) && return (x < 0 && !isinteger(y)) ? nothing : x ^ y
  return nothing
end

"A relation or a logical operator on two scalars (Booleans as 1 and 0); `nothing` for another operator."
function comparison(op::IR.Op, x::Float64, y::Float64)::Union{Nothing, Bool}
  op === IR.LESS && return x < y
  op === IR.LESSEQ && return x <= y
  op === IR.GREATER && return x > y
  op === IR.GREATEREQ && return x >= y
  op === IR.EQUAL && return x == y
  op === IR.NEQUAL && return x != y
  op === IR.AND && return x != 0 && y != 0
  op === IR.OR && return x != 0 || y != 0
  return nothing
end

"The values of a range (`start:step:stop`)."
function rangeValues(e::IR.RANGE, run::Interpretation)::Union{Nothing, Vector{Float64}}
  local bound = x::IR.Expression -> (v = valueOf(x, run); v === nothing ? nothing : scalarOf(v))
  local (a, b) = (bound(e.start), bound(e.stop))
  local h = e.step === nothing ? 1.0 : bound(e.step)
  (a === nothing || b === nothing || h === nothing || h == 0) && return nothing
  return collect(a:h:b)
end

"An array constructor (`{f(i, j) for i in 1:n, j in 1:m}`): its body for each combination of the iterators' items, the first iterator the first dimension."
function constructorValue(e::IR.ARRAY_CONSTRUCTOR, run::Interpretation)::Union{Nothing, FlatValue}
  local out = FlatValue()
  local ok = iterate!(e.iterators, run) do position
    local v = valueOf(e.body, run)
    v === nothing && return false
    local index = join(position, ',')
    for (s, y) in v
      out[isempty(s) ? "[$index]" : startswith(s, "[") ? string("[", index, ",", SubString(s, 2)) : string("[", index, "]", s)] = y
    end
    return true
  end
  return ok ? out : nothing
end

"""
    builtinValue(e, run) -> Union{Nothing, FlatValue}

A built-in's value: the scalar functions of `evaluateCall` (sin, sqrt, min, max, atan2, ...)
and the array built-ins sum, product, transpose, identity, zeros, ones, fill, size, cross, skew,
vector, scalar and cat; `nothing` for another.
"""
function builtinValue(e::IR.CALL, run::Interpretation)::Union{Nothing, FlatValue}
  local n = e.name
  n in (:noEvent, :homotopy) && return valueOf(e.arguments[1], run)
  n === :smooth && return valueOf(e.arguments[2], run)
  local args = FlatValue[]
  for a in e.arguments
    local v = valueOf(a, run)
    v === nothing && return nothing
    push!(args, v)
  end
  local scalars = Union{Nothing, Float64}[scalarOf(v) for v in args]
  if haskey(UNARY_MATH, n) && length(args) == 1
    #= a scalar function of an array: element by element (MLS 12.4.6) =#
    return try
      scalars[1] !== nothing ? scalarValue(Float64(UNARY_MATH[n](scalars[1]))) :
        FlatValue(s => Float64(UNARY_MATH[n](y)) for (s, y) in args[1])
    catch err
      err isa DomainError || rethrow()
      nothing
    end
  end
  if n in (:min, :max) && length(args) == 1
    local A = toArray(args[1])
    return A === nothing ? nothing : scalarValue(n === :min ? minimum(A) : maximum(A))
  end
  if n in (:min, :max, :atan2, :div, :mod, :rem) && length(args) == 2
    local (x, y) = (scalars[1], scalars[2])
    (x === nothing || y === nothing) && return nothing
    n === :min && return scalarValue(min(x, y))
    n === :max && return scalarValue(max(x, y))
    n === :atan2 && return scalarValue(atan(x, y))
    y == 0 && return nothing
    n === :div && return scalarValue(trunc(x / y))
    n === :mod && return scalarValue(x - floor(x / y) * y)
    return scalarValue(x - trunc(x / y) * y)
  end
  local sizes = () -> begin
    local out = Int[]
    for x in scalars[2:end]
      (x === nothing || !isinteger(x) || x < 0) && return nothing
      push!(out, Int(x))
    end
    out
  end
  if n in (:zeros, :ones)
    local dims = Int[]
    for x in scalars
      (x === nothing || !isinteger(x) || x < 0) && return nothing
      push!(dims, Int(x))
    end
    return fromArray(n === :zeros ? zeros(Tuple(dims)) : ones(Tuple(dims)))
  end
  if n === :fill && length(args) >= 2 && scalars[1] !== nothing
    local dims = sizes()
    return dims === nothing ? nothing : fromArray(fill(scalars[1], Tuple(dims)))
  end
  if n === :identity && length(args) == 1 && scalars[1] !== nothing && isinteger(scalars[1])
    local k = Int(scalars[1])
    return fromArray(Float64[i == j ? 1.0 : 0.0 for i in 1:k, j in 1:k])
  end
  local arrays = [toArray(v) for v in args]
  if n in (:sum, :product) && length(args) == 1 && arrays[1] !== nothing
    return scalarValue(n === :sum ? sum(arrays[1]) : prod(arrays[1]))
  end
  if n === :transpose && length(args) == 1 && arrays[1] !== nothing && ndims(arrays[1]) == 2
    return fromArray(permutedims(arrays[1]))
  end
  if n === :size && arrays[1] !== nothing
    length(args) == 1 && return fromArray(Float64[size(arrays[1])...])
    (length(args) == 2 && scalars[2] !== nothing && isinteger(scalars[2])) || return nothing
    local d = Int(scalars[2])
    return 1 <= d <= ndims(arrays[1]) ? scalarValue(Float64(size(arrays[1], d))) : nothing
  end
  if n === :cross && length(args) == 2 && all(A -> A !== nothing && size(A) == (3,), arrays)
    local (a, b) = (arrays[1], arrays[2])
    return fromArray([a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]])
  end
  if n === :skew && length(args) == 1 && arrays[1] !== nothing && size(arrays[1]) == (3,)
    local x = arrays[1]
    return fromArray([0.0 -x[3] x[2]; x[3] 0.0 -x[1]; -x[2] x[1] 0.0])
  end
  if n === :vector && length(args) == 1
    scalars[1] === nothing || return fromArray([scalars[1]])
    return arrays[1] === nothing ? nothing : fromArray(vec(arrays[1]))
  end
  if n === :scalar && length(args) == 1
    scalars[1] === nothing || return args[1]
    return (arrays[1] === nothing || length(arrays[1]) != 1) ? nothing : scalarValue(arrays[1][1])
  end
  if n === :ndims && length(args) == 1
    scalars[1] === nothing || return scalarValue(0.0)
    return arrays[1] === nothing ? nothing : scalarValue(Float64(ndims(arrays[1])))
  end
  if n === :diagonal && length(args) == 1 && arrays[1] !== nothing && ndims(arrays[1]) == 1
    local d = arrays[1]
    return fromArray(Float64[i == j ? d[i] : 0.0 for i in eachindex(d), j in eachindex(d)])
  end
  if n === :outerProduct && length(args) == 2 && all(A -> A !== nothing && ndims(A) == 1, arrays)
    return fromArray(arrays[1] * permutedims(arrays[2]))
  end
  if n === :symmetric && length(args) == 1 && arrays[1] !== nothing && ndims(arrays[1]) == 2 && size(arrays[1], 1) == size(arrays[1], 2)
    local A = arrays[1]
    return fromArray(Float64[i <= j ? A[i, j] : A[j, i] for i in axes(A, 1), j in axes(A, 2)])
  end
  if n === :matrix && length(args) == 1
    scalars[1] === nothing || return fromArray(fill(scalars[1], 1, 1))
    local A = arrays[1]
    A === nothing && return nothing
    ndims(A) == 1 && return fromArray(reshape(A, :, 1))
    return (ndims(A) >= 2 && all(d -> size(A, d) == 1, 3:ndims(A))) ? fromArray(reshape(A, size(A, 1), size(A, 2))) : nothing
  end
  if n === :linspace && length(args) == 3 && all(x -> x !== nothing, scalars) && isinteger(scalars[3]) && scalars[3] >= 2
    return fromArray(collect(range(scalars[1], scalars[2]; length = Int(scalars[3]))))
  end
  if n === :cat && length(args) >= 2 && scalars[1] !== nothing && isinteger(scalars[1])
    any(A -> A === nothing, arrays[2:end]) && return nothing
    return try
      fromArray(cat(arrays[2:end]...; dims = Int(scalars[1])))
    catch err
      err isa DimensionMismatch || rethrow()
      nothing
    end
  end
  return nothing
end

"""
    callValue(call, run) -> Union{Nothing, Vector{FlatValue}}

The outputs of a call of a function by its algorithm run on numbers: its arguments (values; a
function for an input that is one: a partial application, a function's name, the caller's own
function input), an omitted input's default, its outputs' and locals' bindings (a record's
field defaults), then its statements (`runStatements!`). A call of a function input calls the
function it names with the inputs it binds. A pure call of values only is computed once
(`StartValues`' `calls`). `nothing` for an external function, a function not known, a call
nested too deep, or anything that does not evaluate.
"""
function callValue(call::IR.CALL, run::Interpretation)::Union{Nothing, Vector{FlatValue}}
  local partial = get(run.functionArguments, call.name, nothing)
  local f = get(run.values.functions, partial === nothing ? call.name : partial.name, nothing)
  (f === nothing || f.external !== nothing || run.depth >= MAX_INTERPRETED_DEPTH) && return nothing
  local given = Dict{Symbol, Union{FlatValue, PartialCall}}()
  partial === nothing || foreach(((n, v),) -> given[n] = v, partial.bound)
  #= the call's arguments fill the inputs the partial application leaves, in order =#
  local open = IR.FunctionVariable[v for v in f.inputs if !haskey(given, v.name)]
  length(call.arguments) <= length(open) || return nothing
  local pure = partial === nothing && !f.impure
  for (v, a) in zip(open, call.arguments)
    local value = v.ty isa IR.T_FUNCTION ? functionArgument(a, run) : valueOf(a, run)
    value === nothing && return nothing
    value isa PartialCall && (pure = false)
    given[v.name] = value
  end
  local key = pure ? (f.name, FlatValue[given[v.name] for v in open[1:length(call.arguments)]]) : nothing
  key !== nothing && haskey(run.values.calls, key) && return run.values.calls[key]
  local outputs = runFunction(f, given, run)
  key === nothing || (run.values.calls[key] = outputs)
  return outputs
end

"The function an argument passes to a function input: a partial application, a function's name, or the caller's own function input."
function functionArgument(a::IR.Expression, run::Interpretation)::Union{Nothing, PartialCall}
  if a isa IR.PARTIAL_APPLICATION
    local bound = Pair{Symbol, FlatValue}[]
    for (n, x) in zip(a.argumentNames, a.arguments)
      local v = valueOf(x, run)
      v === nothing && return nothing
      push!(bound, n => v)
    end
    return PartialCall(a.name, bound)
  elseif a isa IR.CREF
    local own = get(run.functionArguments, a.cref.name, nothing)
    own === nothing || return own
    haskey(run.values.functions, a.cref.name) && return PartialCall(a.cref.name, Pair{Symbol, FlatValue}[])
  end
  return nothing
end

"`callValue`'s run of `f`: its inputs `given` (by name; the others their defaults), its outputs' and locals' bindings, its algorithm; its outputs' values."
function runFunction(f::IR.ModelicaFunction, given::Dict{Symbol, Union{FlatValue, PartialCall}},
                     run::Interpretation)::Union{Nothing, Vector{FlatValue}}
  local inner = Interpretation(run.values, Dict{Symbol, Float64}(), Dict{Symbol, PartialCall}(), 0, run.depth + 1, true)
  for v in f.inputs
    local value = get(given, v.name, nothing)
    if value === nothing
      v.binding === nothing && return nothing
      value = v.ty isa IR.T_FUNCTION ? functionArgument(v.binding, inner) : valueOf(v.binding, inner)
      value === nothing && return nothing
    end
    value isa PartialCall ? (inner.functionArguments[v.name] = value) : bind!(inner, String(v.name), value)
  end
  #= a binding that does not evaluate leaves the variable to the algorithm; a record without one its fields' defaults =#
  for v in vcat(f.outputs, f.locals)
    if v.binding !== nothing
      local value = valueOf(v.binding, inner)
      value === nothing || bind!(inner, String(v.name), value)
    else
      for (field, x) in v.fieldBindings
        local value = valueOf(x, inner)
        value === nothing || bind!(inner, string(v.name, ".", field), value)
      end
    end
  end
  runStatements!(f.body, inner) === :fail && return nothing
  local outputs = FlatValue[]
  for v in f.outputs
    local x = get(inner.scalars, v.name, nothing)
    local value = x === nothing ? prefixValue(inner.scalars, String(v.name)) : scalarValue(x)
    isempty(value) && return nothing
    push!(outputs, value)
  end
  return outputs
end

"The variable `name` given the value `v` in `run`: its old scalars (a whole array assigned again) removed."
function bind!(run::Interpretation, name::String, v::FlatValue)::Nothing
  delete!(run.scalars, Symbol(name))
  for k in collect(keys(prefixValue(run.scalars, name)))
    delete!(run.scalars, Symbol(name, k))
  end
  for (s, y) in v
    run.scalars[Symbol(name, s)] = y
  end
  return nothing
end

"""
    runStatements!(body, run) -> Symbol

The statements run on numbers in `run`: assignments (to a variable, an element, a tuple of a
call's outputs), if, for (over a range or an array), while, return and break, an assert that
must hold; a call statement is skipped (its effects, `print`, are not values). `:ok`, `:return`,
`:break`, or `:fail` where something does not evaluate or the step bound is reached.
"""
function runStatements!(body::Vector{IR.Statement}, run::Interpretation)::Symbol
  for st in body
    run.steps += 1
    run.steps > MAX_INTERPRETED_STEPS && return :fail
    local status = runStatement!(st, run)
    status === :ok || return status
  end
  return :ok
end

function runStatement!(st::IR.ASSIGNMENT, run::Interpretation)::Symbol
  if st.lhs isa IR.CREF
    local name = resolvedName(st.lhs.cref, run)
    local v = valueOf(st.rhs, run)
    v === nothing && return :fail
    name === nothing && return assignSelection!(st.lhs.cref, v, run)
    bind!(run, name, v)
    return :ok
  elseif st.lhs isa IR.TUPLE && st.rhs isa IR.CALL && !st.rhs.attributes.builtin
    local outputs = callValue(st.rhs, run)
    (outputs === nothing || length(outputs) < length(st.lhs.elements)) && return :fail
    for (target, v) in zip(st.lhs.elements, outputs)
      target isa IR.WILD && continue
      target isa IR.CREF || return :fail
      local name = resolvedName(target.cref, run)
      name === nothing && return :fail
      bind!(run, name, v)
    end
    return :ok
  end
  return :fail
end

"""
    assignSelection!(cref, v, run) -> Symbol

An assignment to a slice or a whole dimension of an array variable's last part (`x[2:3] := v`,
`A[:, 1] := v`): the selected elements of its current value replaced by `v`'s.
"""
function assignSelection!(cref::IR.ComponentRef, v::FlatValue, run::Interpretation)::Symbol
  local last = cref.parts[end]
  (isempty(last.subscripts) || any(p -> !isempty(p.subscripts), cref.parts[1:end - 1])) && return :fail
  local base = resolvedName(IR.ComponentRef(IR.CrefPart[cref.parts[1:end - 1]..., IR.CrefPart(last.name, IR.Subscript[], last.ty)]), run)
  base === nothing && return :fail
  local A = toArray(prefixValue(run.scalars, base))
  local sel = selections(last.subscripts, run)
  (A === nothing || sel === nothing || length(sel) > ndims(A)) && return :fail
  local full = vcat(sel, fill(Colon(), ndims(A) - length(sel)))
  all(((d, s),) -> s isa Colon || all(k -> 1 <= k <= size(A, d), s), enumerate(full)) || return :fail
  local x = scalarOf(v)
  if x !== nothing
    A[full...] .= x
  else
    local B = toArray(v)
    (B === nothing || length(B) != length(view(A, full...))) && return :fail
    A[full...] = reshape(B, size(view(A, full...)))
  end
  bind!(run, base, fromArray(A))
  return :ok
end

function runStatement!(st::IR.IF_STATEMENT, run::Interpretation)::Symbol
  for branch in st.branches
    local c = valueOf(branch.condition, run)
    local x = c === nothing ? nothing : scalarOf(c)
    x === nothing && return :fail
    x != 0 && return runStatements!(branch.body, run)
  end
  return runStatements!(st.elseBody, run)
end

function runStatement!(st::IR.FOR_STATEMENT, run::Interpretation)::Symbol
  local range = valueOf(st.iterator.range, run)
  local items = range === nothing ? nothing : toArray(range)
  (items === nothing || ndims(items) != 1) && return :fail
  local name = st.iterator.name
  local previous = get(run.scalars, name, nothing)
  local status = :ok
  for k in items
    run.scalars[name] = k
    status = runStatements!(st.body, run)
    status === :break && (status = :ok; break)
    status === :ok || break
  end
  previous === nothing ? delete!(run.scalars, name) : (run.scalars[name] = previous)
  return status
end

function runStatement!(st::IR.WHILE_STATEMENT, run::Interpretation)::Symbol
  while true
    local c = valueOf(st.condition, run)
    local x = c === nothing ? nothing : scalarOf(c)
    x === nothing && return :fail
    x == 0 && return :ok
    local status = runStatements!(st.body, run)
    status === :break && return :ok
    status === :ok || return status
    run.steps > MAX_INTERPRETED_STEPS && return :fail
  end
end

runStatement!(::IR.RETURN, ::Interpretation)::Symbol = :return
runStatement!(::IR.BREAK, ::Interpretation)::Symbol = :break
runStatement!(::IR.NORETCALL, ::Interpretation)::Symbol = :ok

function runStatement!(st::IR.ASSERT, run::Interpretation)::Symbol
  local c = valueOf(st.condition, run)
  local x = c === nothing ? nothing : scalarOf(c)
  return (x !== nothing && x == 0) ? :fail : :ok
end

runStatement!(::IR.Statement, ::Interpretation)::Symbol = :fail
