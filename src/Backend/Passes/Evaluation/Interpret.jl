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
   numbers (`callValue`), arrays and records. A value is flat (`FlatValue`): its scalars by their paths,
   the steps from the value to them (`[]` a scalar, `[[1, 2]]` an array's element, `[:T, [1, 1]]` an element of a
   record's field); a function's variables by their names, a reference to one by its parts (`variablePath`), one
   of the model by its reference with the subscripts evaluated (`resolvedReference`). Integers, Booleans and
   enumeration literals are numbers. Anything not handled gives `nothing`, never a throw: an
   external function, a String, a clock, an array of records in array arithmetic, a function as a
   function's output, a when-statement or reinit (not in a function). =#

"The path of a scalar: no step."
const SCALAR_PATH = Path()

"The flat value of a scalar."
scalarValue(x::Float64)::FlatValue = FlatValue(SCALAR_PATH => x)

"The scalar of a flat value; `nothing` for an array or a record."
scalarOf(v::FlatValue)::Union{Nothing, Float64} = (length(v) == 1 && haskey(v, SCALAR_PATH)) ? v[SCALAR_PATH] : nothing

"`p` under the field `field`, or under the element `index` of an array (an array's element's indices after `index`: `[i, j]`)."
underStep(field::Symbol, p::Path)::Path = vcat(Step[field], p)
function underStep(index::Vector{Int}, p::Path)::Path
  (!isempty(p) && p[1] isa Vector{Int}) && return vcat(Step[Int[index; p[1]::Vector{Int}]], @view p[2:end])
  return vcat(Step[index], p)
end

"Whether the path `p` starts with the steps of `prefix`."
function startsWith(p::Path, prefix::Path)::Bool
  length(p) >= length(prefix) || return false
  for k in eachindex(prefix)
    p[k] == prefix[k] || return false
  end
  return true
end

"The part of the flat value `v` at `path` (its scalars under it, by the rest of their paths), a new value."
function selectPath(v::FlatValue, path::Path)::FlatValue
  local out::FlatValue = FlatValue()
  for (p::Path, y::Float64) in v
    startsWith(p, path) && (out[p[length(path) + 1:end]] = y)
  end
  return out
end

"The statements one interpretation runs at most, and the calls it nests at most (a loop or a recursion that does not end)."
const MAX_INTERPRETED_STEPS = 100_000
const MAX_INTERPRETED_DEPTH = 32

"A function passed as an argument (`PARTIAL_APPLICATION`, MLS 12.4.2.1): the function and the inputs it binds by name."
struct PartialCall
  name::Symbol
  bound::Vector{Pair{Symbol, FlatValue}}
end

"""
    Interpretation(values, variables, functionArguments, steps, depth, inFunction)

One numeric evaluation: `variables` the values of the running function's variables by their names (`T`, `R`, `Q`,
a for-loop's iterator), `functionArguments` its inputs that are functions; outside a function (`inFunction`
false) other references are the model's, by `values` (their start). `steps` the statements run so far, `depth`
the calls nested.
"""
mutable struct Interpretation
  values::StartValues
  variables::Dict{Symbol, FlatValue}
  functionArguments::Dict{Symbol, PartialCall}
  steps::Int
  depth::Int
  inFunction::Bool
end

"An interpretation of an expression of the model (not in a function), its names read at `values`."
Interpretation(values::StartValues) = Interpretation(values, Dict{Symbol, FlatValue}(), Dict{Symbol, PartialCall}(), 0, 0, false)

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
      for (s::Path, y::Float64) in v
        out[underStep(field.name, s)] = y
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

"The flat value of an array literal's elements (each element's paths under its index, `underStep`)."
function elementsValue(elements::Vector{IR.Expression}, run::Interpretation)::Union{Nothing, FlatValue}
  local out = FlatValue()
  for (i, x) in enumerate(elements)
    local v = valueOf(x, run)
    v === nothing && return nothing
    for (s::Path, y::Float64) in v
      out[underStep(Int[i], s)] = y
    end
  end
  return out
end

"""
    crefValue(e, run) -> Union{Nothing, FlatValue}

A reference's value, its subscripts evaluated (`c[k]`, `R.T[i,i]` in a loop): a variable of `run` (a function's,
an iterator) at the reference's path (`variablePath`: a whole array or record by its elements); outside a
function, else the model's start value of the reference (`resolvedReference`; `time` 0), a whole record or array
of the model element by element.
"""
function crefValue(e::IR.CREF, run::Interpretation)::Union{Nothing, FlatValue}
  local indices::Union{Nothing, Vector{Vector{Int}}} = partIndices(e.cref, run)
  if indices === nothing
    #= a slice or a whole dimension (`x[2:3]`, `A[:, 1]`) of the last part: the selection of the whole =#
    local last = e.cref.parts[end]
    (isempty(last.subscripts) || any(IR.isSubscripted, @view e.cref.parts[1:end - 1])) && return nothing
    local array::IR.CREF = IR.CREF(IR.withoutLastSubscripts(e.cref), last.ty)
    local arrayValue::Union{Nothing, FlatValue} = crefValue(array, run)
    return arrayValue === nothing ? nothing : selectValue(arrayValue, last.subscripts, run)
  end
  local own::Union{Nothing, FlatValue} = get(run.variables, e.cref.parts[1].name, nothing)
  if own !== nothing
    local v::FlatValue = selectPath(own, variablePath(e.cref, indices))
    isempty(v) || return v
  end
  run.inFunction && return nothing
  local ref::IR.ComponentRef = resolvedReference(e.cref, indices)
  ref.name === :time && return scalarValue(0.0)
  e.ty isa Union{IR.T_RECORD, IR.T_ARRAY} && return wholeValue(ref, e.ty, run.values)
  local x::Union{Nothing, Float64} = run.values(ref.name)
  return x === nothing ? nothing : scalarValue(x)
end

"The indices of the parts' subscripts, evaluated in `run` (none for a part without); `nothing` where one is no single index."
function partIndices(cref::IR.ComponentRef, run::Interpretation)::Union{Nothing, Vector{Vector{Int}}}
  local out::Vector{Vector{Int}} = Vector{Vector{Int}}(undef, length(cref.parts))
  for (k::Int, p::IR.CrefPart) in enumerate(cref.parts)
    if isempty(p.subscripts)
      out[k] = Int[]
      continue
    end
    local index::Union{Nothing, Vector{Int}} = subscriptIndices(p.subscripts, run)
    index === nothing && return nothing
    out[k] = index
  end
  return out
end

"""
    variablePath(cref, indices) -> Path

The path of the reference `cref` in the value of its first part's variable, its parts' subscripts at `indices`
(`partIndices`): its first part's indices, then each other part's field and indices (`[[2], :x]` of `c[2].x`).
"""
function variablePath(cref::IR.ComponentRef, indices::Vector{Vector{Int}})::Path
  local out::Path = Path()
  for (k::Int, p::IR.CrefPart) in enumerate(cref.parts)
    k > 1 && push!(out, p.name)
    isempty(indices[k]) || push!(out, indices[k])
  end
  return out
end

"The reference `cref` with its subscripts as the literal `indices` (`partIndices`): a variable's of the model (`c[2].x` of `c[k].x`; a part of rolled components the member, `IR.CrefPart`)."
function resolvedReference(cref::IR.ComponentRef, indices::Vector{Vector{Int}})::IR.ComponentRef
  all(isempty, indices) && return cref
  return IR.ComponentRef(IR.CrefPart[isempty(indices[k]) ? p :
                                     IR.CrefPart(p.name, IR.Subscript[IR.INDEX(IR.INTEGER(i)) for i in indices[k]], p.ty)
                                     for (k::Int, p::IR.CrefPart) in enumerate(cref.parts)])
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
  local previous = Union{Nothing, FlatValue}[get(run.variables, it.name, nothing) for it in iterators]
  local ok = true
  for I in CartesianIndices(Tuple(length.(items)))
    for (k, it) in enumerate(iterators)
      run.variables[it.name] = scalarValue(items[k][I[k]])
    end
    visit(collect(Tuple(I))) || (ok = false; break)
  end
  for (it, x) in zip(iterators, previous)
    x === nothing ? delete!(run.variables, it.name) : (run.variables[it.name] = x)
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

"""
    wholeValue(ref, ty, values) -> Union{Nothing, FlatValue}

A whole record or array of the model (a reference the scalarized model keeps as a function's
argument, `Frames.to_Q(body.R_start, ...)`) from the start values of its scalars (their references:
`IR.withField`, `IR.elementCref`).
"""
function wholeValue(ref::IR.ComponentRef, ty::IR.ModelicaType, values::StartValues)::Union{Nothing, FlatValue}
  local out = FlatValue()
  return collectWhole!(out, ref, Path(), ty, values) ? out : nothing
end

"`collectWhole!`'s step: the scalars of the reference `ref` (at `path` in the whole) of type `ty` into `out`; whether all evaluated."
function collectWhole!(out::FlatValue, ref::IR.ComponentRef, path::Path, ty::IR.ModelicaType, values::StartValues)::Bool
  if ty isa IR.T_RECORD
    return all(f -> collectWhole!(out, IR.withField(ref, f.name, f.ty), vcat(path, Step[f.name]), f.ty, values), ty.fields)
  elseif ty isa IR.T_ARRAY
    local sizes = Int[]
    for d in IR.dimensions(ty)
      local n = IR.dimensionSize(d)
      n === nothing && return false
      push!(sizes, n)
    end
    return all(I -> (local index::Vector{Int} = collect(Tuple(I));
                     collectWhole!(out, IR.elementCref(ref, index), vcat(path, Step[index]), IR.elementType(ty), values)),
               CartesianIndices(Tuple(sizes)))
  end
  local x = values(ref.name)
  x === nothing && return false
  out[path] = x
  return true
end

"The element of a flat value at `index` (all its dimensions: a scalar; fewer: the rest of the array, or a record element's fields)."
function elementOf(v::FlatValue, index::Vector{Int})::Union{Nothing, FlatValue}
  local n::Int = length(index)
  local out = FlatValue()
  for (p::Path, y::Float64) in v
    (!isempty(p) && p[1] isa Vector{Int}) || continue
    local first::Vector{Int} = p[1]::Vector{Int}
    if first == index
      out[p[2:end]] = y
    elseif length(first) > n && first[1:n] == index
      out[vcat(Step[first[n + 1:end]], @view p[2:end])] = y
    end
  end
  return isempty(out) ? nothing : out
end

"The field `field` of a record's flat value."
function flatField(v::FlatValue, field::Symbol)::Union{Nothing, FlatValue}
  local out = FlatValue()
  for (p::Path, y::Float64) in v
    (!isempty(p) && p[1] === field) && (out[p[2:end]] = y)
  end
  return isempty(out) ? nothing : out
end

"""
    toArray(v) -> Union{Nothing, Array{Float64}}

A flat value of array elements (each path one step of indices, every index present) as an array; `nothing`
for a scalar, a record or an incomplete array.
"""
function toArray(v::FlatValue)::Union{Nothing, Array{Float64}}
  isempty(v) && return nothing
  local entries = Tuple{Vector{Int}, Float64}[]
  for (p::Path, y::Float64) in v
    (length(p) == 1 && p[1] isa Vector{Int}) || return nothing
    local index::Vector{Int} = p[1]::Vector{Int}
    any(k::Int -> k < 1, index) && return nothing
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

"An array as a flat value (each element's path its indices)."
function fromArray(A::AbstractArray{<:Real})::FlatValue
  return FlatValue(Step[collect(Tuple(I))] => Float64(A[I]) for I in CartesianIndices(A))
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
    for (s::Path, y::Float64) in v
      out[underStep(position, s)] = y
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
  local inner = Interpretation(run.values, Dict{Symbol, FlatValue}(), Dict{Symbol, PartialCall}(), 0, run.depth + 1, true)
  for v in f.inputs
    local value = get(given, v.name, nothing)
    if value === nothing
      v.binding === nothing && return nothing
      value = v.ty isa IR.T_FUNCTION ? functionArgument(v.binding, inner) : valueOf(v.binding, inner)
      value === nothing && return nothing
    end
    value isa PartialCall ? (inner.functionArguments[v.name] = value) : bind!(inner, v.name, SCALAR_PATH, value)
  end
  #= a binding that does not evaluate leaves the variable to the algorithm; a record without one its fields' defaults =#
  for v in vcat(f.outputs, f.locals)
    if v.binding !== nothing
      local value = valueOf(v.binding, inner)
      value === nothing || bind!(inner, v.name, SCALAR_PATH, value)
    else
      for (field, x) in v.fieldBindings
        local value = valueOf(x, inner)
        value === nothing || bind!(inner, v.name, Step[field], value)
      end
    end
  end
  runStatements!(f.body, inner) === :fail && return nothing
  local outputs = FlatValue[]
  for v in f.outputs
    local value::Union{Nothing, FlatValue} = get(inner.variables, v.name, nothing)
    (value === nothing || isempty(value)) && return nothing
    push!(outputs, copy(value))
  end
  return outputs
end

"""
    bind!(run, name, path, v)

The variable `name` given the value `v` at `path` (`SCALAR_PATH`: the whole variable; an element, a field): its
old scalars there (a whole array assigned again) removed. A copy: values read stay apart (`selectPath`).
"""
function bind!(run::Interpretation, name::Symbol, path::Path, v::FlatValue)::Nothing
  if isempty(path)
    run.variables[name] = copy(v)
    return nothing
  end
  local own::FlatValue = get!(FlatValue, run.variables, name)
  for p in collect(keys(own))
    startsWith(p, path) && delete!(own, p)
  end
  for (s::Path, y::Float64) in v
    own[vcat(path, s)] = y
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
    local indices::Union{Nothing, Vector{Vector{Int}}} = partIndices(st.lhs.cref, run)
    local v = valueOf(st.rhs, run)
    v === nothing && return :fail
    indices === nothing && return assignSelection!(st.lhs.cref, v, run)
    bind!(run, st.lhs.cref.parts[1].name, variablePath(st.lhs.cref, indices), v)
    return :ok
  elseif st.lhs isa IR.TUPLE && st.rhs isa IR.CALL && !st.rhs.attributes.builtin
    local outputs = callValue(st.rhs, run)
    (outputs === nothing || length(outputs) < length(st.lhs.elements)) && return :fail
    for (target, v) in zip(st.lhs.elements, outputs)
      target isa IR.WILD && continue
      target isa IR.CREF || return :fail
      local indices::Union{Nothing, Vector{Vector{Int}}} = partIndices(target.cref, run)
      indices === nothing && return :fail
      bind!(run, target.cref.parts[1].name, variablePath(target.cref, indices), v)
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
  (isempty(last.subscripts) || any(IR.isSubscripted, @view cref.parts[1:end - 1])) && return :fail
  local base::IR.ComponentRef = IR.withoutLastSubscripts(cref)
  local indices::Union{Nothing, Vector{Vector{Int}}} = partIndices(base, run)
  indices === nothing && return :fail
  local name::Symbol = base.parts[1].name
  local path::Path = variablePath(base, indices)
  local own::Union{Nothing, FlatValue} = get(run.variables, name, nothing)
  local A = own === nothing ? nothing : toArray(selectPath(own, path))
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
  bind!(run, name, path, fromArray(A))
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
  local previous::Union{Nothing, FlatValue} = get(run.variables, name, nothing)
  local status = :ok
  for k in items
    run.variables[name] = scalarValue(k)
    status = runStatements!(st.body, run)
    status === :break && (status = :ok; break)
    status === :ok || break
  end
  previous === nothing ? delete!(run.variables, name) : (run.variables[name] = previous)
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
