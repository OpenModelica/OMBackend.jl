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

#= Symbolic differentiation of OMJLIR expressions and equations: d/dt for the equations
   Pantelides' algorithm differentiates, partial derivatives for the Jacobians of state
   selection. One set of rules, a seed telling what a variable reference and time become. The
   derivatives are built with Evaluation's simplifying constructors (Evaluation/Algebra.jl), a
   call without derivative annotation inlined (Evaluation/Inline.jl). =#

"""
    Seeds

What differentiation replaces the leaves by: `variable(name)` the derivative of a reference to
the variable `name` (an expression; `nothing`: the variable is time-invariant or discrete, its
derivative 0), `time` the derivative of `time` (1 for d/dt, 0 for a partial derivative). The
`functions` by name: a call's derivative annotation, or its body for inlining; `active` the
functions whose bodies are being differentiated (a recursive call is not inlined again);
`derivativeCalls`: a call of a derivative function differentiated by its linearity in its
derivative inputs (`derivativeFunctionCall`; solving for derivative unknowns only, not the choice
of states, whose Jacobian it would give entries it cannot tell from 0).
"""
struct Seeds{V}
  variable::V
  time::Float64
  functions::Dict{Symbol, IR.ModelicaFunction}
  active::Vector{Symbol}
  derivativeCalls::Bool
end

Seeds(variable::V, time::Float64, functions::Dict{Symbol, IR.ModelicaFunction}) where {V} = Seeds(variable, time, functions, Symbol[], false)
Seeds(variable::V, time::Float64, functions::Dict{Symbol, IR.ModelicaFunction}, active::Vector{Symbol}) where {V} =
  Seeds(variable, time, functions, active, false)

builtin(name::Symbol, args::IR.Expression...)::IR.Expression = IR.CALL(name, IR.Expression[args...], IR.T_REAL(), IR.BUILTIN_CALL)
"`e` with its scalar arithmetic folded throughout (`foldScalar`, bottom-up)."
foldTree(e::IR.Expression)::IR.Expression = IR.mapExp(foldScalar, e)

isArrayTyped(e::IR.Expression)::Bool = IR.typeOf(e) isa IR.T_ARRAY

"""
    sumOf(a, op, b, ty) -> Expression

`a op b` for an additive operator (`+`, `-` and their element-wise and scalar-array forms) of
result type `ty`, a zero term dropped where the other has the result's shape.
"""
function sumOf(a::IR.Expression, op::IR.Op, b::IR.Expression, ty::IR.ModelicaType)::IR.Expression
  local subtracting = op in (IR.SUB, IR.SUB_EW, IR.SUB_SCALAR_ARRAY, IR.SUB_ARRAY_SCALAR)
  local zeroA = isZeroExp(a)
  local zeroB = isZeroExp(b)
  if zeroA && zeroB
    local z = zeroOf(IR.BINARY(a, op, b, ty))
    z === nothing || return z
  end
  if zeroB && isArrayTyped(a) == (ty isa IR.T_ARRAY)
    return a
  end
  if zeroA && isArrayTyped(b) == (ty isa IR.T_ARRAY)
    return subtracting ? IR.UNARY(IR.UMINUS, b, IR.typeOf(b)) : b
  end
  return IR.BINARY(a, op, b, ty)
end

"`a op b` for a bilinear operator (products), of result type `ty`; a zero factor makes the zero of `ty`."
function productOf(a::IR.Expression, op::IR.Op, b::IR.Expression, ty::IR.ModelicaType)::IR.Expression
  local z = (isZeroExp(a) || isZeroExp(b)) ? zeroOf(IR.BINARY(a, op, b, ty)) : nothing
  return z === nothing ? IR.BINARY(a, op, b, ty) : z
end

"Whether `e` is of a type with no derivative (Integer, Boolean, String, an enumeration): piecewise constant."
isPiecewiseConstant(e::IR.Expression)::Bool = IR.typeOf(e) isa Union{IR.T_INTEGER, IR.T_BOOLEAN, IR.T_STRING, IR.T_ENUMERATION}

"""
    differentiate(e, seeds) -> Expression

The derivative of `e` by the rules of calculus, the leaves by `seeds` (see `Seeds`). Throws
NotHandled for an expression without rules here (a call without a derivative annotation and a
body to inline, an array expression, ...).
"""
function differentiate(e::IR.Expression, seeds::Seeds)::IR.Expression
  isPiecewiseConstant(e) && return ZERO
  return derivativeOf(e, seeds)
end

derivativeOf(::Union{IR.INTEGER, IR.REAL, IR.BOOLEAN, IR.STRING, IR.ENUM_LITERAL}, ::Seeds)::IR.Expression = ZERO

"""
    ArrayVariables(variable)

The `variable` of the seeds of an arrays-kept model (`arraySeeds`): it knows an array variable as a whole, so a
reference to an element it does not know by name (`x[i]`, `c[i].T`) is that element of the array's derivative
(`arrayElementDerivative`).
"""
struct ArrayVariables{F}
  variable::F
end

(a::ArrayVariables)(name::Symbol) = a.variable(name)

function derivativeOf(e::IR.CREF, seeds::Seeds)::IR.Expression
  e.cref.name === :time && return real(seeds.time)
  e.ty isa Union{IR.T_ARRAY, IR.T_RECORD} && return wholeDerivative(e, seeds)
  local d = seeds.variable(e.cref.name)
  return d === nothing ? arrayElementDerivative(e, seeds) : d
end

"The derivative of a whole array or record reference (`wholeDerivative` by its reference); with arrays kept (`ArrayVariables`) an array variable's derivative array where the seeds give one, of a part of one that part."
wholeDerivative(e::IR.CREF, seeds::Seeds)::IR.Expression = wholeDerivative(e.cref, e.ty, seeds)
function wholeDerivative(e::IR.CREF, seeds::Seeds{<:ArrayVariables})::IR.Expression
  e.ty isa IR.T_ARRAY || return wholeDerivative(e.cref, e.ty, seeds)
  if IR.hasSubscripts(e.cref)
    #= a part of an array (a rolled component's vector, `G[\$c].cylinder.e`): that part of its derivative array; of
       an array without one, by literal subscripts its elements' derivatives (a record's array field whose elements are
       variables, `frame_a.R.w`: theirs; a parameter's: 0), by others zeros where its elements have no derivative (a
       parameter's), NotHandled where they do =#
    local array = seeds.variable(unsubscriptedName(e.cref))
    if array === nothing
      local sizes::Vector{Int} = knownSizes(e.ty, "a part of the array $(unsubscriptedName(e.cref))")
      local subscripts::Vector{IR.Subscript} = allSubscripts(e.cref)
      local given::Union{Nothing, Vector{Int}} = IR.literalIndices(subscripts)
      if given !== nothing
        return elementsDerivative(IR.withoutSubscripts(e.cref), [given; sizes], given, IR.elementType(e.ty), seeds)
      end
      local probe::IR.ComponentRef = withIteratorsAsOne(e.cref)
      local probeElement::Symbol = IR.appendSubscripts(probe, IR.Subscript[IR.INDEX(IR.INTEGER(1)) for _ in sizes]).name
      seeds.variable(probeElement) === nothing ||
        throw(NotHandled("the derivative of a part of an array ($(e.cref.name)) whose elements are variables of their own"))
      local zeros::Union{Nothing, IR.Expression} = zeroOf(e)
      zeros === nothing && throw(NotHandled("the derivative of a part of an array ($(e.cref.name)) of a size not known"))
      return zeros
    end
    (array isa IR.CREF && array.ty isa IR.T_ARRAY && !IR.hasSubscripts(array.cref) && IR.isDerivativeCref(array.cref)) ||
      throw(NotHandled("the derivative of a part of an array ($(e.cref.name)), its array's derivative $(IR.toString(array))"))
    return IR.CREF(IR.withLastSubscripts(array.cref, allSubscripts(e.cref)), e.ty)
  end
  local d = seeds.variable(e.cref.name)
  return d === nothing ? wholeDerivative(e.cref, e.ty, seeds) : d
end

"A reference with every subscript that is not a literal index as 1 (a probe element of a part of an array)."
withIteratorsAsOne(cref::IR.ComponentRef)::IR.ComponentRef =
  IR.ComponentRef(IR.CrefPart[IR.CrefPart(p.name, IR.Subscript[IR.literalIndex(x) !== nothing ? x : IR.INDEX(IR.INTEGER(1))
                                                         for x in p.subscripts], p.ty) for p in cref.parts])

"""
    arrayElementDerivative(e, seeds) -> Expression

The derivative of an element of an array whose derivative the seeds (`ArrayVariables`) give as a whole: that
element of the derivative array (`\$DER.x[i]`, `\$DER.c.T[i]`, the subscripts of all parts on its last part,
`IR.withLastSubscripts`); of a
slice's element the slice's; of an element of an array that is no variable, its elements' derivatives at the
subscripts; 0 for another reference, and with other seeds (the variable has no derivative).
"""
arrayElementDerivative(::IR.CREF, ::Seeds)::IR.Expression = ZERO

function arrayElementDerivative(e::IR.CREF, seeds::Seeds{<:ArrayVariables})::IR.Expression
  IR.hasSubscripts(e.cref) || return ZERO
  local array = seeds.variable(unsubscriptedName(e.cref))
  if array === nothing
    #= an element of an array split along its inner dimensions: the slice's (`slicedCref`) =#
    local sliced::Union{Nothing, IR.ComponentRef} = slicedCref(e.cref, (n::Symbol) -> seeds.variable(n) !== nothing)
    sliced === nothing || return derivativeOf(IR.CREF(sliced, e.ty), seeds)
    #= an element by subscripts of an array that is no variable but whose elements are (a record's array field,
       `frame_a.R.w[\$i1]`): the array of its elements' derivatives (by name, 0 for a parameter's) at the subscripts =#
    local last::IR.CrefPart = e.cref.parts[end]
    if last.ty isa IR.T_ARRAY && length(IR.dimensions(last.ty)) == length(last.subscripts) &&
       !any(IR.isSubscripted, @view e.cref.parts[1:end - 1]) && all(x::IR.Subscript -> x isa IR.INDEX, last.subscripts)
      return foldTree(selectElement(wholeDerivative(IR.withoutSubscripts(e.cref), last.ty, seeds), last.subscripts, e.ty))
    end
    return ZERO
  end
  (array isa IR.CREF && array.ty isa IR.T_ARRAY && !IR.hasSubscripts(array.cref) && IR.isDerivativeCref(array.cref)) ||
    throw(NotHandled("the derivative of the element $(e.cref.name), its array's derivative $(IR.toString(array))"))
  return IR.CREF(IR.withLastSubscripts(array.cref, allSubscripts(e.cref)), e.ty)
end

"""
    wholeDerivative(cref, ty, seeds) -> Expression

The derivative of a whole array or record `cref` (a reference the scalarized model keeps as a
function argument): the array or record of its elements' derivatives (`x[1,2]`, `r.f`: their
references, by their flat names).
"""
function wholeDerivative(cref::IR.ComponentRef, ty::IR.ModelicaType, seeds::Seeds)::IR.Expression
  if ty isa IR.T_RECORD
    return IR.RECORD(IR.Expression[wholeDerivative(IR.withField(cref, f.name, f.ty), f.ty, seeds)
                                   for f in ty.fields], ty)
  elseif ty isa IR.T_ARRAY
    local sizes = Int[]
    for d in IR.dimensions(ty)
      local n = IR.dimensionSize(d)
      n === nothing && throw(NotHandled("the derivative of the array $(cref.name) of unknown size"))
      push!(sizes, n)
    end
    return elementsDerivative(cref, sizes, Int[], IR.elementType(ty), seeds)
  end
  ty isa IR.T_REAL || return ZERO
  local d::Union{Nothing, IR.Expression} = seeds.variable(cref.name)
  return d === nothing ? ZERO : d
end

"The derivative of the elements of the array `cref` (of `sizes`) under the subscripts `index` so far, on its last part: an array literal of them, nested by dimension."
function elementsDerivative(cref::IR.ComponentRef, sizes::Vector{Int}, index::Vector{Int}, element::IR.ModelicaType, seeds::Seeds)::IR.Expression
  if length(index) == length(sizes)
    return wholeDerivative(IR.appendSubscripts(cref, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in index]),
                           element, seeds)
  end
  local rest = sizes[length(index) + 1:end]
  local ty = IR.T_ARRAY(element, IR.Dimension[IR.DIM_SIZE(n) for n in rest])
  return IR.ARRAY(IR.Expression[elementsDerivative(cref, sizes, vcat(index, k), element, seeds) for k in 1:rest[1]], ty)
end

function derivativeOf(e::IR.BINARY, seeds::Seeds)::IR.Expression
  local (a, b) = (e.exp1, e.exp2)
  local (da, db) = (differentiate(a, seeds), differentiate(b, seeds))
  local op = e.op
  if op in (IR.ADD, IR.SUB) && !(e.ty isa IR.T_ARRAY)
    return op === IR.ADD ? plus(da, db) : minus(da, db)
  end
  op in ADDITIVE && return sumOf(da, op, db, e.ty)
  if op === IR.MUL && !(e.ty isa IR.T_ARRAY)
    return plus(times(da, b), times(a, db))
  end
  op in BILINEAR && return sumOf(productOf(da, op, b, e.ty), IR.ADD_EW, productOf(a, op, db, e.ty), e.ty)
  if op === IR.DIV && !(e.ty isa IR.T_ARRAY)
    return minus(divide(da, b), divide(times(a, db), power(b, real(2.0))))
  end
  if op in (IR.DIV_EW, IR.DIV_ARRAY_SCALAR)
    #= (a ./ b)' = da ./ b - a .* db ./ b.^2 (b an array or a scalar) =#
    local square = IR.BINARY(b, op === IR.DIV_EW ? IR.POW_EW : IR.POW, real(2.0), IR.typeOf(b))
    local first = productOf(da, op, b, e.ty)
    local second = productOf(IR.BINARY(a, op === IR.DIV_EW ? IR.MUL_EW : IR.MUL_ARRAY_SCALAR, db, e.ty), op, square, e.ty)
    return sumOf(first, IR.SUB_EW, second, e.ty)
  end
  if op === IR.POW && !(e.ty isa IR.T_ARRAY)
    #= a^b: b * a^(b - 1) * da where b is constant, else a^b * (db * log(a) + b * da / a) =#
    isZero(db) && return times(times(b, power(a, minus(b, ONE))), da)
    return times(e, plus(times(db, builtin(:log, a)), divide(times(b, da), a)))
  end
  throw(NotHandled("the derivative of the operator $(op)"))
end

"Additive operators: the derivative of `a op b` is `da op db`."
const ADDITIVE = (IR.ADD, IR.SUB, IR.ADD_EW, IR.SUB_EW, IR.ADD_SCALAR_ARRAY, IR.ADD_ARRAY_SCALAR,
                  IR.SUB_SCALAR_ARRAY, IR.SUB_ARRAY_SCALAR)

"Bilinear operators (products): the derivative of `a op b` is `da op b + a op db`."
const BILINEAR = (IR.MUL, IR.MUL_EW, IR.MUL_SCALAR_ARRAY, IR.MUL_ARRAY_SCALAR, IR.MUL_VECTOR_MATRIX,
                  IR.MUL_MATRIX_VECTOR, IR.SCALAR_PRODUCT, IR.MATRIX_PRODUCT)

function derivativeOf(e::IR.UNARY, seeds::Seeds)::IR.Expression
  e.op === IR.UMINUS || throw(NotHandled("the derivative of the operator $(e.op)"))
  return negate(differentiate(e.exp, seeds))
end

"The condition stays (the branch taken is decided at events); each branch differentiated."
derivativeOf(e::IR.IF_EXPRESSION, seeds::Seeds)::IR.Expression =
  ifThenElse(e.condition, differentiate(e.thenExp, seeds), differentiate(e.elseExp, seeds), e.ty)

derivativeOf(e::IR.CAST, seeds::Seeds)::IR.Expression = differentiate(e.exp, seeds)

function derivativeOf(e::IR.TUPLE_ELEMENT, seeds::Seeds)::IR.Expression
  e.exp isa IR.CALL && !e.exp.attributes.builtin || throw(NotHandled("the derivative of a tuple element of $(nameof(typeof(e.exp)))"))
  return IR.TUPLE_ELEMENT(functionDerivative(e.exp, seeds), e.index, e.ty)
end

derivativeOf(e::IR.ARRAY, seeds::Seeds)::IR.Expression =
  IR.ARRAY(IR.Expression[differentiate(x, seeds) for x in e.elements], e.ty)

"""
    derivativeOf(e::ARRAY_CONSTRUCTOR, seeds)

The derivative of `{body for i in 1:n}` (one iterator over `1:n`, `n` a literal: the known
values folded, `withKnownValues`): of the array of its elements, `i` replaced by its value, in
subscripts too (`withIterator`: MSL Media's mixtures, `{h_T(data[i], T) for i in 1:nX}`).
"""
function derivativeOf(e::IR.ARRAY_CONSTRUCTOR, seeds::Seeds)::IR.Expression
  local range = length(e.iterators) == 1 ? e.iterators[1].range : nothing
  (range isa IR.RANGE && isOne(range.start) && range.step === nothing && range.stop isa IR.INTEGER) ||
    throw(NotHandled("the derivative of an array constructor not over 1:n"))
  local n = range.stop.value
  local name = e.iterators[1].name
  local ty = IR.T_ARRAY(e.ty.elementType, IR.Dimension[IR.DIM_SIZE(n)])
  return derivativeOf(IR.ARRAY(IR.Expression[withIterator(e.body, name, k) for k in 1:n], ty), seeds)
end

"""
    withIterator(e, name, k) -> Expression

`e` with the iterator `name` replaced by the value `k`, in subscripts too (folded: `data[i]` is
`data[2]`); not inside a nested constructor or reduction that binds `name` again.
"""
withIterator(e::IR.Expression, name::Symbol, k::Int)::IR.Expression = withIterators(e, Dict(name => k))

"""
    withIterators(e, values) -> Expression

`e` with each iterator of `values` (name => value) replaced by its value, in one pass (`withIterator`);
inside a nested constructor or reduction, the iterators it binds again are not.
"""
function withIterators(e::IR.Expression, values::Dict{Symbol, Int})::IR.Expression
  isempty(values) && return e
  return IR.mapExpTopDown(e) do x
    if x isa Union{IR.ARRAY_CONSTRUCTOR, IR.REDUCTION} && any(it -> haskey(values, it.name), x.iterators)
      local inner = filter(((name, _),) -> all(it -> it.name !== name, x.iterators), values)
      return (IR.mapChildren(c -> withIterators(c, inner), x), false)
    end
    x isa IR.CREF || return (x, true)
    (length(x.cref.parts) == 1 && !IR.hasSubscripts(x.cref) && haskey(values, x.cref.name)) &&
      return (IR.INTEGER(values[x.cref.name]), false)
    IR.hasSubscripts(x.cref) || return (x, false)
    local parts = IR.CrefPart[IR.CrefPart(p.name, IR.Subscript[s isa IR.INDEX ? IR.INDEX(substituteIterators(s.exp, values)) : s
                                                               for s in p.subscripts], p.ty) for p in x.cref.parts]
    return (IR.CREF(foldSubscripts(IR.ComponentRef(parts)), x.ty), false)
  end
end

derivativeOf(e::IR.SUBSCRIPTED, seeds::Seeds)::IR.Expression = selectElement(differentiate(e.exp, seeds), e.subscripts, e.ty)

derivativeOf(e::IR.RECORD, seeds::Seeds)::IR.Expression =
  IR.RECORD(IR.Expression[differentiate(x, seeds) for x in e.elements], e.ty)

"""
A field of a record expression: of a record constructor, its element; of a call, the field of
the call's derivative (its body inlined where it runs into one expression, `inlineBody`, as
MSL's Frames.planarRotation).
"""
function derivativeOf(e::IR.RECORD_ELEMENT, seeds::Seeds)::IR.Expression
  local inner = e.exp
  local record = asRecord(inner)
  record === nothing || return differentiate(record.elements[e.index], seeds)
  if inner isa IR.CALL && !inner.attributes.builtin
    local f = get(seeds.functions, inner.name, nothing)
    local body = f === nothing ? nothing : inlineBody(f, inner)
    local inlined = body === nothing ? nothing : asRecord(body)
    inlined === nothing || return differentiate(inlined.elements[e.index], seeds)
    return IR.RECORD_ELEMENT(functionDerivative(inner, seeds), e.index, e.field, e.ty)
  end
  inner isa IR.CREF &&
    return differentiate(IR.CREF(IR.withField(inner.cref, e.field, e.ty), e.ty), seeds)
  throw(NotHandled("the derivative of a field of a $(nameof(typeof(inner)))"))
end

derivativeOf(e::IR.Expression, ::Seeds)::IR.Expression = throw(NotHandled("the derivative of a $(nameof(typeof(e)))"))

function derivativeOf(e::IR.CALL, seeds::Seeds)::IR.Expression
  e.attributes.builtin || return functionDerivative(e, seeds)
  local args = e.arguments
  local a = isempty(args) ? ZERO : args[1]
  local da() = differentiate(a, seeds)
  local n = e.name
  n === :sin && return times(builtin(:cos, a), da())
  n === :cos && return negate(times(builtin(:sin, a), da()))
  n === :tan && return divide(da(), power(builtin(:cos, a), real(2.0)))
  n === :asin && return divide(da(), builtin(:sqrt, minus(ONE, power(a, real(2.0)))))
  n === :acos && return negate(divide(da(), builtin(:sqrt, minus(ONE, power(a, real(2.0))))))
  n === :atan && return divide(da(), plus(ONE, power(a, real(2.0))))
  if n === :atan2
    local (y, x) = (args[1], args[2])
    return divide(minus(times(x, differentiate(y, seeds)), times(y, differentiate(x, seeds))),
                  plus(power(x, real(2.0)), power(y, real(2.0))))
  end
  n === :sinh && return times(builtin(:cosh, a), da())
  n === :cosh && return times(builtin(:sinh, a), da())
  n === :tanh && return times(minus(ONE, power(builtin(:tanh, a), real(2.0))), da())
  n === :exp && return times(e, da())
  n === :log && return divide(da(), a)
  n === :log10 && return divide(da(), times(a, real(log(10.0))))
  n === :sqrt && return divide(da(), times(real(2.0), e))
  n === :abs && return times(builtin(:noEvent, builtin(:sign, a)), da())
  n in (:sign, :integer, :floor, :ceil, :div, :pre, :previous, :edge, :change, :initial, :terminal, :sample) && return ZERO
  n === :noEvent && return builtin(:noEvent, da())
  n === :smooth && return differentiate(args[2], seeds)
  n === :homotopy && return da()
  if n in (:min, :max)
    #= the branch by noEvent: no new zero crossing, the derivative jumps where the operands cross =#
    local (b, db) = (args[2], differentiate(args[2], seeds))
    local cond = builtin(:noEvent, IR.RELATION(a, n === :min ? IR.LESS : IR.GREATER, b, 0))
    return ifThenElse(cond, da(), db, IR.T_REAL())
  end
  if n in (:mod, :rem)
    #= mod(x, y) = x - floor(x/y)*y, rem(x, y) = x - div(x, y)*y: piecewise x' - k y' =#
    local (x, y) = (args[1], args[2])
    local k = n === :mod ? builtin(:floor, divide(x, y)) : builtin(:div, x, y)
    return minus(differentiate(x, seeds), times(k, differentiate(y, seeds)))
  end
  if n === :semiLinear
    local (x, kp, km) = (args[1], args[2], args[3])
    local cond = builtin(:noEvent, IR.RELATION(x, IR.GREATEREQ, ZERO, 0))
    return ifThenElse(cond, plus(times(differentiate(x, seeds), kp), times(x, differentiate(kp, seeds))),
                            plus(times(differentiate(x, seeds), km), times(x, differentiate(km, seeds))), IR.T_REAL())
  end
  if n in LINEAR_BUILTINS
    #= linear in its first argument (the others are sizes or dimensions) =#
    local d = da()
    local call = IR.CALL(n, IR.Expression[d; args[2:end]...], e.ty, e.attributes)
    local z = isZeroExp(d) ? zeroOf(call) : nothing
    return z === nothing ? call : z
  end
  if n in CONSTANT_BUILTINS
    local z = zeroOf(e)
    #= of unknown size: the call times 0 =#
    return z === nothing ? IR.BINARY(e, e.ty isa IR.T_ARRAY ? IR.MUL_ARRAY_SCALAR : IR.MUL, ZERO, e.ty) : z
  end
  if n in BILINEAR_BUILTINS
    local (x, y) = (args[1], args[2])
    local first = IR.CALL(n, IR.Expression[differentiate(x, seeds), y], e.ty, e.attributes)
    local second = IR.CALL(n, IR.Expression[x, differentiate(y, seeds)], e.ty, e.attributes)
    return sumOf(first, e.ty isa IR.T_ARRAY ? IR.ADD_EW : IR.ADD, second, e.ty)
  end
  if n === :cat
    return IR.CALL(n, IR.Expression[args[1]; IR.Expression[differentiate(x, seeds) for x in args[2:end]]...], e.ty, e.attributes)
  end
  if n === :der
    #= der(der-free expression) of a call: d/dt of its argument's derivative =#
    return differentiate(differentiate(a, Seeds(seeds.variable, 1.0, seeds.functions, seeds.active)), seeds)
  end
  throw(NotHandled("the derivative of the built-in $(n)"))
end

"Built-ins linear in their first argument: the derivative is the call of the argument's derivative."
const LINEAR_BUILTINS = (:promote, :transpose, :vector, :matrix, :scalar, :sum, :diagonal, :skew, :fill, :symmetric)

"Built-ins of no continuous argument: a zero derivative."
const CONSTANT_BUILTINS = (:identity, :zeros, :ones, :size, :ndims)

"Built-ins bilinear in their two arguments: `f(da, b) + f(a, db)`."
const BILINEAR_BUILTINS = (:cross, :outerProduct)

"Whether a value of type `ty` contains a Real (a Real, an array of Reals, a record with one): an input a derivative function takes the derivative of."
containsReal(ty::IR.ModelicaType)::Bool =
  IR.elementType(ty) isa IR.T_REAL || (ty isa IR.T_RECORD && any(f -> containsReal(f.ty), ty.fields))

"""
    derivativeFunctionCall(call, seeds) -> Union{Nothing, Expression}

The derivative of a call of a derivative function (the function another's annotation of order 1 names) where the
seeds read none of its leading inputs, those of the function it is the derivative of: it is linear in its derivative
inputs (MLS 12.7.1), so its derivative is the call with each derivative input's derivative in its place. MSL R134a's
`rho_ph_der(p, h, derivs, p_der, h_der)` by `der(p)`: its body (an if-statement on a record another call computes)
does not inline, and the equation `der(d) = rho_ph_der(...)` determined no derivative unknown. `nothing` for another
call, or where the seeds read a leading input.
"""
function derivativeFunctionCall(call::IR.CALL, seeds::Seeds)::Union{Nothing, IR.Expression}
  local n = nothing
  for g in values(seeds.functions)
    any(d -> d.order == 1 && d.derivative === call.name, g.derivatives) || continue
    n = length(g.inputs)
    break
  end
  (n === nothing || n >= length(call.arguments)) && return nothing
  local leading = call.arguments[1:n]
  any(a -> readsSeeded(a, seeds), leading) && return nothing
  return IR.CALL(call.name, IR.Expression[leading..., (differentiate(a, seeds) for a in call.arguments[(n + 1):end])...],
                 call.ty, call.attributes)
end

"""
    readsSeeded(e, seeds) -> Bool

Whether `e` reads a variable the seeds differentiate: a reference they give a derivative (an array's element by the
array's name too), `time` with a time derivative, an array or record reference whose derivative is not 0 (or does
not evaluate).
"""
function readsSeeded(e::IR.Expression, seeds::Seeds)::Bool
  return IR.anyExp(e) do x
    x isa IR.CREF || return false
    x.cref.name === :time && return seeds.time != 0
    if x.ty isa Union{IR.T_ARRAY, IR.T_RECORD}
      return try
        !isZeroExp(foldTree(wholeDerivative(x, seeds)))
      catch err
        err isa NotHandled || rethrow()
        true
      end
    end
    seeds.variable(x.cref.name) !== nothing && return true
    return IR.hasSubscripts(x.cref) && seeds.variable(unsubscriptedName(x.cref)) !== nothing
  end
end

"""
    functionDerivative(call, seeds) -> Expression

The derivative of a call of a user function: by its derivative annotation of order 1 (MLS
12.7.1: the inputs, then the derivatives of those that contain a Real, `containsReal`, and the
annotation does not exclude), else by
its body where that runs symbolically into one expression (`inlineBody`: assignments and
if-statements, the inputs substituted), else NotHandled (so is a recursive call within its own
body, `seeds.active`).
"""
function functionDerivative(call::IR.CALL, seeds::Seeds)::IR.Expression
  local f = get(seeds.functions, call.name, nothing)
  f === nothing && throw(NotHandled("the derivative of $(call.name) (a function not in the model)"))
  local annotation = findfirst(d -> d.order == 1, f.derivatives)
  if annotation !== nothing
    local d = f.derivatives[annotation]
    local args = copy(call.arguments)
    for (i, a) in enumerate(call.arguments)
      (i in d.excludedInputs || !containsReal(f.inputs[i].ty)) && continue
      push!(args, differentiate(a, seeds))
    end
    local dty = length(f.outputs) == 1 ? f.outputs[1].ty : call.ty
    return IR.CALL(d.derivative, args, dty, call.attributes)
  end
  local linear = seeds.derivativeCalls ? derivativeFunctionCall(call, seeds) : nothing
  linear === nothing || return linear
  call.name in seeds.active && throw(NotHandled("the derivative of $(call.name): a recursive function without derivative annotation"))
  local body = inlineBody(f, call)
  body === nothing && throw(NotHandled("the derivative of $(call.name): no derivative annotation, and a body that does not run into one expression"))
  push!(seeds.active, call.name)
  try
    return differentiate(body, seeds)
  finally
    pop!(seeds.active)
  end
end

"""
    differentiateEquation(eq, seeds) -> Equation

The derivative of a continuous equation as a new equation (its source; attributes
`differentiated`): a scalar equation side by side, an if-equation branch by branch (its
conditions kept). Throws NotHandled for an algorithm and the other kinds.
"""
function differentiateEquation(eq::IR.SCALAR_EQUATION, seeds::Seeds)::IR.Equation
  return IR.SCALAR_EQUATION(differentiate(eq.lhs, seeds), differentiate(eq.rhs, seeds), eq.ty, eq.source,
                            IR.EquationAttributes(eq.attributes.kind, true, nothing, false))
end

function differentiateEquation(eq::IR.IF_EQUATION, seeds::Seeds)::IR.Equation
  local branches = IR.EquationBranch[IR.EquationBranch(b.condition, IR.Equation[differentiateEquation(e, seeds) for e in b.equations])
                                      for b in eq.branches]
  return IR.IF_EQUATION(branches, IR.Equation[differentiateEquation(e, seeds) for e in eq.elseEquations], eq.source,
                        IR.EquationAttributes(eq.attributes.kind, true, nothing, false))
end

differentiateEquation(eq::IR.Equation, ::Seeds)::IR.Equation =
  throw(NotHandled("differentiating a $(nameof(typeof(eq)))"))
