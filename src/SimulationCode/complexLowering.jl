#= Lowering Complex operator records (QuasiStatic phasors) to their real and imaginary parts. =#

#= ── Complex operator-record lowering ────────────────────────────────────
   QuasiStationary / QuasiStatic models carry Complex phasor records. The
   frontend scalarizes the components into `<name>_re` / `<name>_im` SimVars
   but leaves operator-record calls (`'*'.multiply`, `conj`, `fromReal`,
   `'-'.subtract`, `'+'.add`, `'/'.divide`, `arg`, `'abs'`) and bare complex
   crefs in the residuals. This pass rewrites them to scalar `_re`/`_im`
   arithmetic so later passes and codegen only ever see real scalars. =#

function _pathLastName(@nospecialize(p))::String
  p isa Absyn.IDENT && return p.name
  p isa Absyn.QUALIFIED && return _pathLastName(p.path)
  p isa Absyn.FULLYQUALIFIED && return _pathLastName(p.path)
  return ""
end

# Operator-record function token. canonicalizeCrefNames flattens qualified
# paths into one IDENT with `_` separators (e.g. ComplexVoltage_'*'_multiply),
# so the operation is the final underscore-delimited token.
_opToken(@nospecialize(p))::String = String(last(split(_pathLastName(p), '_')))

_cplxRe(base::AbstractString) = EXP_CREF(SimCref(base * "_re"), TYPE_REAL())
_cplxIm(base::AbstractString) = EXP_CREF(SimCref(base * "_im"), TYPE_REAL())
_mulE(a, b) = BINARY(a, OP_MUL, b)
_addE(a, b) = BINARY(a, OP_ADD, b)
_subE(a, b) = BINARY(a, OP_SUB, b)
_negE(a)    = UNARY(OP_UMINUS, a)

"(reExp, imExp) for a complex-valued expression, or nothing if not complex."
function _complexParts(@nospecialize(exp))::Union{Nothing, Tuple{Exp, Exp}}
  if exp isa EXP_CREF && exp.ty isa TYPE_COMPLEX
    #= Preserve cref subscripts on Complex array elements: a Complex array y[m]
       with `subs = [1]` must scalarize to `y[1]_re` / `y[1]_im`, not bare
       `y_re` / `y_im`. The SimVar table holds the bracketed names after BDAE's
       afterExpandComplex pass, so dropping subs leaves SimCodeCheck flagging
       unresolved refs (this was the UnsymmetricalLoad failure shape). =#
    local sc = exp.cref
    local base = isempty(sc.subs) ?
                   string(sc.sym) :
                   string(sc.sym) * "[" * join(sc.subs, "][") * "]"
    return (_cplxRe(base), _cplxIm(base))
  elseif exp isa CALL
    local fn = _opToken(exp.path)
    local args = exp.args
    if fn == "fromReal" && length(args) >= 1
      return (_lowerComplexExp(args[1]),
              length(args) >= 2 ? _lowerComplexExp(args[2]) : RCONST(0.0))
    elseif fn == "conj" && length(args) == 2
      #= pre-split conj(re, im) -> (re, -im) =#
      return (_lowerComplexExp(args[1]), _negE(_lowerComplexExp(args[2])))
    elseif fn == "exp" && length(args) == 2
      #= exp(re + i·im) = e^re·(cos(im) + i·sin(im)); args pre-split (re, im). =#
      local rev = _lowerComplexExp(args[1]); local imv = _lowerComplexExp(args[2])
      local er = CALL(Absyn.IDENT("exp"), Exp[rev], DAE.callAttrBuiltinReal)
      return (_mulE(er, CALL(Absyn.IDENT("cos"), Exp[imv], DAE.callAttrBuiltinReal)),
              _mulE(er, CALL(Absyn.IDENT("sin"), Exp[imv], DAE.callAttrBuiltinReal)))
    elseif fn == "conj" && length(args) >= 1
      local p = _complexParts(args[1]); p === nothing && return nothing
      return (p[1], _negE(p[2]))
    elseif (fn == "multiply" || fn == "'*'") && length(args) == 4
      #= pre-split multiply(re1, im1, re2, im2) -> complex product. =#
      local r1 = _lowerComplexExp(args[1]); local i1 = _lowerComplexExp(args[2])
      local r2 = _lowerComplexExp(args[3]); local i2 = _lowerComplexExp(args[4])
      return (_subE(_mulE(r1, r2), _mulE(i1, i2)),
              _addE(_mulE(r1, i2), _mulE(i1, r2)))
    elseif (fn == "multiply" || fn == "'*'") && length(args) == 3
      #= 3-arg multiply: `f(c1: Complex, c2_re: Real, c2_im: Real)` shape used by
         Modelica.ComplexBlocks.Interfaces.ComplexInput.'*'.multiply where the
         second Complex operand is already passed pre-split. =#
      local a = _complexParts(args[1])
      a === nothing && return nothing
      local br = _lowerComplexExp(args[2])
      local bi = _lowerComplexExp(args[3])
      return (_subE(_mulE(a[1], br), _mulE(a[2], bi)),
              _addE(_mulE(a[1], bi), _mulE(a[2], br)))
    elseif (fn == "multiply" || fn == "'*'") && length(args) >= 2
      local a = _complexParts(args[1]); local b = _complexParts(args[2])
      (a === nothing || b === nothing) && return nothing
      return (_subE(_mulE(a[1], b[1]), _mulE(a[2], b[2])),
              _addE(_mulE(a[1], b[2]), _mulE(a[2], b[1])))
    elseif (fn == "subtract" || fn == "'-'") && length(args) == 4
      #= pre-split subtract(re1, im1, re2, im2) -> (re1-re2, im1-im2). =#
      return (_subE(_lowerComplexExp(args[1]), _lowerComplexExp(args[3])),
              _subE(_lowerComplexExp(args[2]), _lowerComplexExp(args[4])))
    elseif (fn == "subtract" || fn == "'-'") && length(args) >= 2
      local a = _complexParts(args[1]); local b = _complexParts(args[2])
      (a === nothing || b === nothing) && return nothing
      return (_subE(a[1], b[1]), _subE(a[2], b[2]))
    elseif (fn == "negate" || fn == "'-'") && length(args) == 1
      local a = _complexParts(args[1]); a === nothing && return nothing
      return (_negE(a[1]), _negE(a[2]))
    elseif (fn == "add" || fn == "'+'") && length(args) == 4
      #= pre-split add(re1, im1, re2, im2) -> (re1+re2, im1+im2). =#
      return (_addE(_lowerComplexExp(args[1]), _lowerComplexExp(args[3])),
              _addE(_lowerComplexExp(args[2]), _lowerComplexExp(args[4])))
    elseif (fn == "add" || fn == "'+'") && length(args) >= 2
      local a = _complexParts(args[1]); local b = _complexParts(args[2])
      (a === nothing || b === nothing) && return nothing
      return (_addE(a[1], b[1]), _addE(a[2], b[2]))
    elseif (fn == "divide" || fn == "'/'") && length(args) == 4
      #= 4-arg divide: `f(nr, ni, dr, di) -> Complex` shape used by Complex_'/'
         where both operands are passed pre-split. =#
      local nr = _lowerComplexExp(args[1])
      local ni = _lowerComplexExp(args[2])
      local dr = _lowerComplexExp(args[3])
      local di = _lowerComplexExp(args[4])
      local den = _addE(_mulE(dr, dr), _mulE(di, di))
      return (BINARY(_addE(_mulE(nr, dr), _mulE(ni, di)), OP_DIV, den),
              BINARY(_subE(_mulE(ni, dr), _mulE(nr, di)), OP_DIV, den))
    elseif (fn == "divide" || fn == "'/'") && length(args) >= 2
      local a = _complexParts(args[1]); local b = _complexParts(args[2])
      (a === nothing || b === nothing) && return nothing
      local den = _addE(_mulE(b[1], b[1]), _mulE(b[2], b[2]))
      return (BINARY(_addE(_mulE(a[1], b[1]), _mulE(a[2], b[2])), OP_DIV, den),
              BINARY(_subE(_mulE(a[2], b[1]), _mulE(a[1], b[2])), OP_DIV, den))
    end
    return nothing
  end
  return nothing
end

"Real scalar for a complex projection (re / im / abs / arg), or nothing."
function _complexProjection(@nospecialize(exp))::Union{Nothing, Exp}
  if exp isa RSUB
    local p = _complexParts(exp.exp); p === nothing && return nothing
    exp.fieldName == "re" && return p[1]
    exp.fieldName == "im" && return p[2]
    return nothing
  elseif exp isa ASUB && length(exp.subs) == 1 && exp.subs[1] isa ICONST
    local p = _complexParts(exp.exp); p === nothing && return nothing
    exp.subs[1].value == 1 && return p[1]
    exp.subs[1].value == 2 && return p[2]
    return nothing
  elseif exp isa TSUB
    #= TSUB(complex_expr, idx) appears when a Modelica `'/'` / `'*'` overload
       on Complex returns a record whose .re/.im fields are accessed by index
       (1 = re, 2 = im) instead of by name. =#
    local p = _complexParts(exp.exp); p === nothing && return nothing
    exp.index == 1 && return p[1]
    exp.index == 2 && return p[2]
    return nothing
  elseif exp isa CALL
    local fn = _opToken(exp.path)
    #= abs/arg may arrive with a single Complex arg, or pre-split as scalar
       (re, im[, extra]) args. Resolve (re, im) from whichever form. =#
    if (fn == "'abs'" || fn == "abs" || fn == "arg")
      local re, im
      local phi0 = nothing
      if length(exp.args) >= 2 && _complexParts(exp.args[1]) === nothing
        re = _lowerComplexExp(exp.args[1]); im = _lowerComplexExp(exp.args[2])
        length(exp.args) >= 3 && (phi0 = _lowerComplexExp(exp.args[3]))
      elseif length(exp.args) >= 1
        local p = _complexParts(exp.args[1]); p === nothing && return nothing
        re = p[1]; im = p[2]
        length(exp.args) >= 2 && (phi0 = _lowerComplexExp(exp.args[2]))
      else
        return nothing
      end
      if fn == "arg"
        local w = CALL(Absyn.IDENT("atan2"), Exp[im, re], exp.attr)
        (phi0 === nothing || (phi0 isa RCONST && iszero(phi0.value)) || (phi0 isa ICONST && iszero(phi0.value))) && return w
        #= arg(c, phi0) in (phi0 - pi, phi0 + pi]: MSL Math.atan3, w + 2pi*integer((pi + phi0 - w)/(2pi)).
           phi0 was dropped (arg(c, 3) gave -2.36 for 3.93). =#
        local twoPi = RCONST(2pi)
        local n = CALL(Absyn.IDENT("integer"), Exp[BINARY(_addE(RCONST(Float64(pi)), _subE(phi0, w)), OP_DIV, twoPi)], exp.attr)
        return _addE(w, _mulE(twoPi, n))
      else
        return BINARY(_addE(BINARY(re, OP_POW, RCONST(2.0)),
                            BINARY(im, OP_POW, RCONST(2.0))), OP_POW, RCONST(0.5))
      end
    end
    return nothing
  end
  return nothing
end

function _lowerComplexVisitor(@nospecialize(exp), arg)
  local proj = _complexProjection(exp)
  proj === nothing ? (exp, true, arg) : (proj, false, arg)
end

_lowerComplexExp(@nospecialize(exp))::Exp = traverseExpTopDown(exp, _lowerComplexVisitor, nothing)[1]

"SimCode pass: scalarize Complex operator-record expressions in residuals."
function lowerComplexOperatorRecords(simCode::SIM_CODE)::SIM_CODE
  local newRes = RESIDUAL_EQUATION[]
  for eq in simCode.residualEquations
    push!(newRes, typeof(eq)(_lowerComplexExp(eq.exp), eq.source, eq.attr))
  end
  @assign simCode.residualEquations = newRes
  return simCode
end
