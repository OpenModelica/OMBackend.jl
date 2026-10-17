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

#= Complex operator records (QuasiStatic phasors) lowered to their real and imaginary parts. =#

"""
    lowerComplexOperatorRecords(simCode)

The residuals with their Complex values read by part: the frontend gives a Complex variable
`c` the variables `c_re` and `c_im` but leaves the operator records' calls (`'*'.multiply`,
`conj`, `fromReal`, `'-'.subtract`, `'+'.add`, `'/'.divide`, `exp`) and the projections
(`.re`, `.im`, `'abs'`, `arg`) of them; each projection becomes Real arithmetic on the parts,
so that the later passes and the code generators see Real scalars only.
"""
function lowerComplexOperatorRecords(simCode::SIM_CODE)::SIM_CODE
  return update(simCode; residualEquations = RESIDUAL_EQUATION[RESIDUAL_EQUATION(lowerComplex(eq.exp), eq.source, eq.attr)
                                                               for eq in simCode.residualEquations])
end

"`e` with its projections of Complex values (`complexProjection`) lowered, top-down."
lowerComplex(e::Exp)::Exp = first(traverseExpTopDown(e, complexVisitor, nothing))

function complexVisitor(e::Exp, ::Nothing)::Tuple{Exp, Bool, Nothing}
  local p = complexProjection(e)
  return p === nothing ? (e, true, nothing) : (p, false, nothing)
end

"""
    operatorName(name)

The operation of a call to an operator record's function: the last `_`-separated part of its
canonical name (`Complex_'*'_multiply` as `multiply`).
"""
operatorName(name::Symbol)::String = String(last(split(String(name), '_')))

"""
    isComplexFunction(name)

Whether `name` is one of Complex's operator functions (`Complex_'*'_multiply`,
`Complex_'constructor'_fromReal`) or of Modelica.ComplexMath (`Modelica_ComplexMath_conj`).
"""
isComplexFunction(name::Symbol)::Bool =
  (local s = String(name); startswith(s, "Complex_") || startswith(s, "Complex.") || startswith(s, "Modelica_ComplexMath_"))

"An operator record's function: its name has an operator's part (`ComplexVoltage_'*'_multiply`, `_'+'`, `_'constructor'_`)."
isOperatorRecordFunction(name::Symbol)::Bool = occursin(r"_'(\*|\+|-|/|constructor)'(_|$)", String(name))

"A record of the fields `re` and `im`: Complex, or a type that extends it (Modelica.SIunits.ComplexVoltage)."
isComplexRecordType(t::SType)::Bool = t isa TYPE_COMPLEX && t.isRecord && t.fieldNames == ["re", "im"]

"""
    isComplexOperation(e)

Whether the call `e` is one `operatorName` reads as a Complex operation: a function of Complex or
Modelica.ComplexMath (`isComplexFunction`), or an operator of a type extending Complex
(`Modelica_SIunits_ComplexVoltage_'*'_multiply`, `isOperatorRecordFunction`) that returns one. A
Real function named like an operation (a `Pkg.add` of four inputs) is none.
"""
isComplexOperation(e::CALL)::Bool =
  isComplexFunction(e.name) || (isOperatorRecordFunction(e.name) && isComplexRecordType(e.ty))

"The variables of the real and imaginary parts of the Complex variable `base`."
partVariables(base::String)::Tuple{Exp, Exp} = (EXP_CREF(SimCref(base * "_re"), TYPE_REAL()), EXP_CREF(SimCref(base * "_im"), TYPE_REAL()))
mul(a::Exp, b::Exp)::Exp = BINARY(a, OP_MUL, b)
add(a::Exp, b::Exp)::Exp = BINARY(a, OP_ADD, b)
sub(a::Exp, b::Exp)::Exp = BINARY(a, OP_SUB, b)
divide(a::Exp, b::Exp)::Exp = BINARY(a, OP_DIV, b)
neg(a::Exp)::Exp = UNARY(OP_UMINUS, a)
realCall(name::Symbol, args::Exp...)::Exp = CALL(name, Exp[args...], TYPE_REAL(), CALL_BUILTIN)

"The product of the Complex values `(ar, ai)` and `(br, bi)`."
complexProduct(ar::Exp, ai::Exp, br::Exp, bi::Exp)::Tuple{Exp, Exp} = (sub(mul(ar, br), mul(ai, bi)), add(mul(ar, bi), mul(ai, br)))

"The quotient of the Complex values `(nr, ni)` and `(dr, di)`."
function complexQuotient(nr::Exp, ni::Exp, dr::Exp, di::Exp)::Tuple{Exp, Exp}
  local den = add(mul(dr, dr), mul(di, di))
  return (divide(add(mul(nr, dr), mul(ni, di)), den), divide(sub(mul(ni, dr), mul(nr, di)), den))
end

"""
    complexParts(e)

The real and imaginary parts of the Complex value `e`: of a variable (an array element) of a
record type, its `_re` and `_im` variables; of an operator record's call, by its operation,
with the operands given whole or by part (the frontend splits a Complex input into its parts:
`multiply(re1, im1, re2, im2)`); `nothing` for another expression.
"""
function complexParts(e::Exp)::Union{Nothing, Tuple{Exp, Exp}}
  if e isa EXP_CREF
    e.ty isa TYPE_COMPLEX || return nothing
    return partVariables(crefName(e.cref))
  end
  (e isa CALL && isComplexOperation(e)) || return nothing
  local (op, args, n) = (operatorName(e.name), e.args, length(e.args))
  local x = lowerComplex
  if op == "fromReal" && n >= 1
    return (x(args[1]), n >= 2 ? x(args[2]) : RCONST(0.0))
  elseif op == "conj" && n == 2
    return (x(args[1]), neg(x(args[2])))
  elseif op == "exp" && n == 2
    local er = realCall(:exp, x(args[1]))
    return (mul(er, realCall(:cos, x(args[2]))), mul(er, realCall(:sin, x(args[2]))))
  elseif op == "conj" && n >= 1
    local p = complexParts(args[1])
    return p === nothing ? nothing : (p[1], neg(p[2]))
  elseif op in ("multiply", "'*'")
    n == 4 && return complexProduct(x(args[1]), x(args[2]), x(args[3]), x(args[4]))
    if n == 3
      local a = complexParts(args[1])
      return a === nothing ? nothing : complexProduct(a[1], a[2], x(args[2]), x(args[3]))
    end
    n >= 2 && return byParts(complexProduct, args)
  elseif op in ("subtract", "'-'") && n >= 2
    n == 4 && return (sub(x(args[1]), x(args[3])), sub(x(args[2]), x(args[4])))
    return byParts((ar, ai, br, bi) -> (sub(ar, br), sub(ai, bi)), args)
  elseif op in ("negate", "'-'") && n == 1
    local a = complexParts(args[1])
    return a === nothing ? nothing : (neg(a[1]), neg(a[2]))
  elseif op in ("add", "'+'") && n >= 2
    n == 4 && return (add(x(args[1]), x(args[3])), add(x(args[2]), x(args[4])))
    return byParts((ar, ai, br, bi) -> (add(ar, br), add(ai, bi)), args)
  elseif op in ("divide", "'/'") && n >= 2
    n == 4 && return complexQuotient(x(args[1]), x(args[2]), x(args[3]), x(args[4]))
    return byParts(complexQuotient, args)
  end
  return nothing
end

"`f` of the parts of the two Complex operands `args[1]` and `args[2]`; `nothing` if one is no Complex value."
function byParts(f::F, args::Vector{Exp})::Union{Nothing, Tuple{Exp, Exp}} where {F}
  local (a, b) = (complexParts(args[1]), complexParts(args[2]))
  (a === nothing || b === nothing) && return nothing
  return f(a[1], a[2], b[1], b[2])
end

"""
    complexProjection(e)

The Real value of a projection of a Complex value: its part (`.re`, `.im`; the first or second
field by index); its absolute value (`'abs'`, `abs`) or argument (`arg(c)`, `arg(c, phi0)` in
`(phi0 - pi, phi0 + pi]`), of the value given whole or by part; `nothing` for another
expression.
"""
function complexProjection(e::Exp)::Union{Nothing, Exp}
  if e isa RSUB
    e.fieldName in ("re", "im") || return nothing
    local p = complexParts(e.exp)
    return p === nothing ? nothing : e.fieldName == "re" ? p[1] : p[2]
  elseif e isa ASUB
    (length(e.subs) == 1 && e.subs[1] isa ICONST && e.subs[1].value in (1, 2)) || return nothing
    local p = complexParts(e.exp)
    return p === nothing ? nothing : p[e.subs[1].value]
  elseif e isa TSUB
    e.index in (1, 2) || return nothing
    local p = complexParts(e.exp)
    return p === nothing ? nothing : p[e.index]
  elseif e isa CALL
    isComplexFunction(e.name) || return nothing
    local op = operatorName(e.name)
    op in ("'abs'", "abs", "arg") || return nothing
    local args = e.args
    local (parts, rest) = if length(args) >= 2 && complexParts(args[1]) === nothing
      ((lowerComplex(args[1]), lowerComplex(args[2])), args[3:end])
    elseif length(args) >= 1
      (complexParts(args[1]), args[2:end])
    else
      (nothing, Exp[])
    end
    parts === nothing && return nothing
    local (r, i) = parts
    op == "arg" || return BINARY(add(BINARY(r, OP_POW, RCONST(2.0)), BINARY(i, OP_POW, RCONST(2.0))), OP_POW, RCONST(0.5))
    local w = realCall(:atan2, i, r)
    local phi0 = isempty(rest) ? nothing : lowerComplex(rest[1])
    (phi0 === nothing || (phi0 isa Union{RCONST, ICONST} && iszero(phi0.value))) && return w
    #= MSL Math.atan3: w + 2pi*integer((pi + phi0 - w)/(2pi)) =#
    local twoPi = RCONST(2pi)
    return add(w, mul(twoPi, CALL(:integer, Exp[divide(add(RCONST(Float64(pi)), sub(phi0, w)), twoPi)], TYPE_INTEGER(), CALL_BUILTIN)))
  end
  return nothing
end
