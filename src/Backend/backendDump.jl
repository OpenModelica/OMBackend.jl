#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
=#

#= This file contains functions to translate backend structures into a Julia string representation. =#


using MetaModelica

const HEAD_LINE = "############################################"
const DOUBLE_LINE = "============================================"
const LINE = "---------------------------------------------"
const _ARRAY_DUMP_LIMIT = 12

#= Abbreviate large literal arrays in dumps so constant lookup tables (e.g. the
   9x9 Logic enum tables) collapse to a `{<R×C table>}` / `{…<N elems>}` summary
   instead of flooding the log. Small arrays still print in full. =#
function _dumpArray(expl)::String
  local n = 0
  local first = nothing
  for e in expl
    n += 1
    n == 1 && (first = e)
  end
  n == 0 && return "{}"
  if first isa DAE.ARRAY
    local m = 0
    for _ in first.array
      m += 1
    end
    n * m > _ARRAY_DUMP_LIMIT && return "{<" * string(n) * "×" * string(m) * " table>}"
  elseif n > _ARRAY_DUMP_LIMIT
    local parts = String[]
    local i = 0
    for e in expl
      i += 1
      i <= 3 || break
      push!(parts, string(e))
    end
    return "{" * join(parts, ", ") * ", …<" * string(n) * " elems>}"
  end
  return "{" * lstString(expl, ", ") * "}"
end

function stringHeading1(i::Any, heading::String)::String
  str = heading1(heading) + "\n" + string(i)
end

function stringHeading2(i::Any, heading::String)::String
  str = heading2(heading) + "\n" + string(i)
end

function stringHeading3(i::Any, heading::String)::String
  str = heading3(heading) + string(i)
end

function heading1(heading::String)::String
  str = HEAD_LINE + "\n" + heading + "\n" + HEAD_LINE + "\n\n"
end

function heading2(heading::String)::String
  str = DOUBLE_LINE + "\n" + heading + "\n" + DOUBLE_LINE + "\n\n"
end

function heading3(heading::String)::String
  str = heading + ":\n" + LINE + "\n"
end

function Base.string(dae::BDAE.BACKEND_DAE)::String
  str::String = ""
  for i in 1:arrayLength(dae.eqs)
    str = str + stringHeading2(dae.eqs[i], "EqSystem " + Base.string(i)) + "\n"
  end
  return str
end

function Base.string(eq::BDAE.EQSYSTEM)::String
  str::String = ""
  #= One-line greppable summary mirroring the simCode dump, so a pass that shifts
     the BDAE variable/equation counts is visible via `grep SUMMARY`. The equation
     figure is a list-entry count: array/when/if entries are single elements that may
     each stand for several scalar equations, so it is not a scalar balance. =#
  local kindCounts = Dict{String, Int}()
  for v in eq.orderedVars
    local k = string(v.varKind)
    kindCounts[k] = get(kindCounts, k, 0) + 1
  end
  local kindStr = join([ck * "=" * string(kindCounts[ck]) for ck in sort(collect(keys(kindCounts)))], " ")
  str = str * "SUMMARY vars | " * kindStr * " total=" * string(length(eq.orderedVars)) * "\n"
  str = str * "SUMMARY eqs | listEntries=" * string(length(eq.orderedEqs)) * " note=array/when/if-entries-not-scalar-expanded\n\n"
  str = str + heading3("Variables") + BDAEUtil.mapEqSystemVariablesNoUpdate(eq, stringTraverse, "") + "\n"
  str = str * "#Total variables, parameters and constants = " * string(length(eq.orderedVars), "\n")
  str = str + heading3("Equations") + BDAEUtil.mapEqSystemEquationsNoUpdate(eq, stringTraverse, "") + "\n"
  str = str * "# Equations = " * string(length(eq.orderedEqs), "\n")
end

function stringTraverse(in, str)::String
  str = str + string(in)
end

function Base.string(var::BDAE.VAR)::String
  local bindStr = @match var.bindExp begin
    SOME(exp) => " | Binding: " * string(exp)
    _ => ""
  end
  return string(var.varType) * " " * string(var.varName) * " | " * string(var.varKind) *
         bindStr * _dumpVarAttributes(var.values) *
         " | Array-dimensions:" * string(var.arryDim) * "\n"
end

#= Render the start/fixed/stateSelect (and min/max/nominal) carried on the
   attribute bundle so a pass that drops an initial-condition attribute is
   visible in the variable dumps, not just the equation list. =#
_attrOptStr(label, opt)::String = @match opt begin
  SOME(v) => label * "=" * string(v) * ", "
  _ => ""
end

function _dumpVarAttributes(values)::String
  local s = @match values begin
    SOME(a) where a isa DAE.VAR_ATTR_REAL =>
      _attrOptStr("start", a.start) * _attrOptStr("fixed", a.fixed) *
      _attrOptStr("stateSelect", a.stateSelectOption) * _attrOptStr("min", a.min) *
      _attrOptStr("max", a.max) * _attrOptStr("nominal", a.nominal)
    SOME(a) where a isa DAE.VAR_ATTR_INT =>
      _attrOptStr("start", a.start) * _attrOptStr("fixed", a.fixed) *
      _attrOptStr("min", a.min) * _attrOptStr("max", a.max)
    SOME(a) where a isa DAE.VAR_ATTR_BOOL =>
      _attrOptStr("start", a.start) * _attrOptStr("fixed", a.fixed)
    SOME(a) where a isa DAE.VAR_ATTR_ENUMERATION =>
      _attrOptStr("start", a.start) * _attrOptStr("fixed", a.fixed) *
      _attrOptStr("min", a.min) * _attrOptStr("max", a.max)
    _ => ""
  end
  return isempty(s) ? "" : " | Attr: " * rstrip(c -> (c == ',' || c == ' '), s)
end

Base.@nospecializeinfer function Base.string(@nospecialize(varKind::BDAE.VarKind))::String
  str = begin
    @match varKind begin
      BDAE.VARIABLE() =>  "VARIABLE"
      BDAE.STATE() =>  "STATE"
      BDAE.STATE_DER() =>  "STATE_DER"
      BDAE.DUMMY_DER() =>  "DUMMY_DER"
      BDAE.DUMMY_STATE() =>  "DUMMY_STATE"
      BDAE.CLOCKED_STATE() =>  "CLOCKED_STATE"
      BDAE.DISCRETE() =>  "DISCRETE"
      BDAE.PARAM() =>  "PARAM"
      BDAE.CONST() =>  "CONST"
      BDAE.EXTOBJ() =>  "EXTOBJ"
      BDAE.JAC_VAR() =>  "JAC_VAR"
      BDAE.JAC_DIFF_VAR() =>  "JAC_DIFF_VAR"
      BDAE.SEED_VAR() =>  "SEED_VAR"
      BDAE.OPT_CONSTR() =>  "OPT_CONSTR"
      BDAE.OPT_FCONSTR() =>  "OPT_FCONSTR"
      BDAE.OPT_INPUT_WITH_DER() =>  "OPT_INPUT_WITH_DER"
      BDAE.OPT_INPUT_DER() =>  "OPT_INPUT_DER"
      BDAE.OPT_TGRID() =>  "OPT_TGRID"
      BDAE.OPT_LOOP_INPUT() =>  "OPT_LOOP_INPUT"
      BDAE.ALG_STATE() =>  "ALG_STATE"
      BDAE.ALG_STATE_OLD() =>  "ALG_STATE_OLD"
      BDAE.DAE_RESIDUAL_VAR() =>  "DAE_RESIDUAL_VAR"
      BDAE.DAE_AUX_VAR() =>  "DAE_AUX_VAR"
      BDAE.LOOP_ITERATION() =>  "LOOP_ITERATION"
      BDAE.LOOP_SOLVED() =>  "LOOP_SOLVED"
    end
  end
end

function Base.string(lst::List{BDAE.Equation})
  res = "List: {"
  for eq in lst
    res *= string(eq) * ","
  end
  res *= "}"
  return res
end

function Base.string(lstlst::List{List{BDAE.Equation}})
  res = "List: {"
  for lst in lstlst
    res *= string(lst) * ","
  end
  res *= "}"
  return res
end

function Base.string(exprs::List{DAE.Exp})
  res = "List: {"
  for exp in exprs
    res *= string(exp) * ","
  end
  res *= "}"
  return res
end

function Base.string(@nospecialize(eq::BDAE.Equation))
  str = begin
    local lhs::DAE.Exp
    local rhs::DAE.Exp
    local cref::DAE.ComponentRef
    local whenEquation::BDAE.WhenEquation
    @match eq begin
      BDAE.EQUATION(lhs = lhs, rhs = rhs) => begin
        (string(lhs) + " = " + string(rhs))
      end

      BDAE.ARRAY_EQUATION(dimSize, left, right, _, _, _) => begin
        "ARRAY_EQUATION[" * join(string.(dimSize), ",") * "]: " * string(left) * " = " * string(right)
      end

      BDAE.SOLVED_EQUATION(componentRef = cref, exp = rhs) => begin
        (string(cref) + " = " + string(rhs))
      end

      BDAE.RESIDUAL_EQUATION(exp = rhs) => begin
        ("0 = " + string(rhs))
      end

      BDAE.WHEN_EQUATION(whenEquation = whenEquation) => begin
        string(whenEquation)
      end

      BDAE.STRUCTURAL_WHEN_EQUATION(s, weq, source, attr) => begin
        string("STRUCTURAL|", string(weq))
      end

      BDAE.INITIAL_STRUCTURAL_STATE(__) => begin
        "INITIAL_STRUCTURAL_STATE(" * eq.initialState * ")"
      end

      BDAE.STRUCTURAL_TRANSITION(__) => begin
        "STRUCTURAL_TRANSITION " * eq.fromState * " -> " * eq.toState * "| if:" * string(eq.transitionCondition)
      end

      BDAE.DUMMY_EQUATION() => begin
        "DUMMY_EQUATION"
      end

      BDAE.ASSERT_EQUATION(condition, message, _, _) => begin
        "ASSERT_EQUATION(condition = $(string(condition)), message = \"$(string(message))\")"
      end

      BDAE.IF_EQUATION(__) => begin
        local strTmp::String
        local conditions::List{DAE.Exp}
        local trueEquation::BDAE.Equation
        local ifEq = eq
        local conditions = ifEq.conditions
        local trueEquations = ifEq.eqnstrue
        local strTmp = ""
        counter = 1
        for cond in conditions
          local condStr = string(cond)
          if counter == 1
            strTmp *= "if " + condStr + " then\n"
          else
            strTmp *= "else if " + condStr + " then\n"
          end
          for teq in listGet(trueEquations, counter)
            strTmp *= string(teq)
          end
          counter += 1
        end
        strTmp += "else\n"
        for feq in ifEq.eqnsfalse
          strTmp += string(feq)
        end
        strTmp += "end"
      end

      BDAE.COMPLEX_EQUATION(size, left, right, _, _) => begin
        "COMPLEX_EQUATION[size=$size]: " * string(left) * " = " * string(right)
      end

      BDAE.FOR_EQUATION(iter, start, stop, body, _, _) => begin
        "for " * string(iter) * " in " * string(start) * ":" * string(stop) * " loop\n  " * string(body) * "end for"
      end

      BDAE.ALGORITHM(size, alg, _, _, _) => begin
        "ALGORITHM[size=$size]"
      end

      _ => begin
        "UNKNOWN_EQUATION_TYPE: " * string(typeof(eq))
      end
    end

  end
  return str * "\n"
end

function Base.string(whenEq::BDAE.WhenEquation)::String
  local elseWhen
  str = "when " + string(whenEq.condition) + " then\n"
  for op in whenEq.whenStmtLst
    str = str + "  " + string(op) + "\n"
  end
  if isSome(whenEq.elsewhenPart)
    @match SOME(elseWhen) = whenEq.elsewhenPart
    str = str + "else" + string(elseWhen)
  end
  return str + "end;"
end

function Base.string(whenOp::BDAE.WhenOperator)::String
  str = begin
    local e1::DAE.Exp
    local e2::DAE.Exp
    local cref::DAE.CREF
    @match whenOp begin
      BDAE.ASSIGN(left = e1, right = e2) => begin
        string(e1) + " := " + string(e2)
      end
      BDAE.REINIT(stateVar = cref, value = e1) => begin
        "reinit(" + string(cref) + ", " + string(e1) + ")"
      end
      BDAE.ASSERT(condition = e1, message = e2) => begin
        "assert(" + string(e1) + ", " + string(e2) + ")"
      end
      BDAE.TERMINATE(message = e1) => begin
        "[TERMINATE]" + string(e1)
      end
      BDAE.NORETCALL(exp = e1) => begin
        "[NORET]" + string(e1)
      end
      BDAE.RECOMPILATION(componentToChange, newValue) => begin
        string("RECOMPILATION", "(", string(componentToChange), ",", string(newValue), ")")
      end
    end
  end
end

"""
  `Base.string(cr::DAE.ComponentRef; seprator=".")`
   Converts a `DAE.ComponentRef` to a Julia string.
   TODO: Discuss separators. A different one should maybe be used..
"""
Base.@nospecializeinfer function Base.string(@nospecialize(cr::DAE.ComponentRef); separator=OMBackend.COMPONENT_SEPARATOR, printType = false)
  buf = IOBuffer()
  _writeCref(buf, cr, separator)
  if printType
    @match cr begin
      DAE.CREF_QUAL(identType = ty) || DAE.CREF_IDENT(identType = ty) => begin
        print(buf, '|')
        print(buf, string(ty))
      end
      _ => nothing
    end
  end
  return String(take!(buf))
end

function _writeSubscripts(buf::IOBuffer, subscriptLst)
  if !listEmpty(subscriptLst)
    for s in subscriptLst
      print(buf, '[')
      print(buf, string(s))
      print(buf, ']')
    end
  end
end

Base.@nospecializeinfer function _writeCref(buf::IOBuffer, @nospecialize(cr::DAE.ComponentRef), sep::String)
  @match cr begin
    DAE.CREF_QUAL(ident = ident, subscriptLst = subscriptLst, componentRef = cref) => begin
      print(buf, ident)
      _writeSubscripts(buf, subscriptLst)
      print(buf, sep)
      _writeCref(buf, cref, sep)
    end
    DAE.CREF_IDENT(ident, _, subscriptLst) => begin
      print(buf, ident)
      _writeSubscripts(buf, subscriptLst)
    end
    DAE.CREF_ITER(ident = ident) => begin
      print(buf, ident)
    end
    #= Wildcard CREF: emitted for tuple-assign placeholders like `(a, _, b) := f(...)`. =#
    DAE.WILD() => print(buf, "_")
    #= Optimica attribute cref: fall back to the underlying componentRef. =#
    DAE.OPTIMICA_ATTR_INST_CREF(componentRef = inner) => _writeCref(buf, inner, sep)
  end
end

function Base.string(@nospecialize(op::DAE.Operator))::String
  str = begin
    @match op begin
      DAE.ADD() => "+"

      DAE.SUB() => "-"

      DAE.MUL() => "*"

      DAE.DIV() => "/"

      DAE.POW() => "^"

      DAE.UMINUS() =>  "-"

      DAE.UMINUS_ARR() => "-"

      DAE.ADD_ARR() => "+"

      DAE.SUB_ARR() => "-"

      DAE.MUL_ARR() => "*"

      DAE.DIV_ARR() => "/"

      DAE.MUL_ARRAY_SCALAR() => "*"

      DAE.ADD_ARRAY_SCALAR() => "+"

      DAE.SUB_SCALAR_ARRAY() =>  "-"

      DAE.MUL_SCALAR_PRODUCT() => "*"

      DAE.MUL_MATRIX_PRODUCT() => "*"

      DAE.DIV_ARRAY_SCALAR() => "/"

      DAE.DIV_SCALAR_ARRAY() => "/"

      DAE.POW_ARRAY_SCALAR() => "^"

      DAE.POW_SCALAR_ARRAY() => "^"

      DAE.POW_ARR() => "^"

      DAE.POW_ARR2() => "^"

      DAE.AND() => "&&"

      DAE.OR() => "||"

      DAE.NOT() => "!"

      DAE.LESS() => "<"

      DAE.LESSEQ() => "<="

      DAE.GREATER() => ">"

      DAE.GREATEREQ() => ">="

      DAE.EQUAL() => "="

      DAE.NEQUAL() => "!="

      DAE.USERDEFINED() => "[UNDEF OP]"

      _ => throw("Unknown operator")

    end
  end
end

function Base.string(@nospecialize(exp::DAE.Exp))::String
  str = begin
    local int::Int
    local real::ModelicaReal
    local bool::Bool
    local tmpStr::String
    local cr::DAE.ComponentRef
    local e1::DAE.Exp
    local e2::DAE.Exp
    local e3::DAE.Exp
    local expl::List{DAE.Exp}
    local lstexpl::List{List{DAE.Exp}}
    local ty::DAE.Type
    @match exp begin
      DAE.ICONST(int) => begin
        Base.string(int)
      end

      DAE.RCONST(real)  => begin
        Base.string(real)
      end

      DAE.SCONST(tmpStr)  => begin
        tmpStr
      end

      DAE.BCONST(bool)  => begin
        Base.string(bool)
      end

      DAE.ENUM_LITERAL(name = Absyn.IDENT(str), index = int) => begin
        str * "(" * Base.string(int) * ")"
      end

      DAE.ENUM_LITERAL(name = path, index = int) => begin
        #= For qualified enum paths like
           `Modelica.Electrical.Digital.Interfaces.Logic.'1'` keep only
           the last two segments (`Logic.'1'`) plus the literal index. =#
        local _segs = String[]
        local _walk = function(p)
          if p isa Absyn.IDENT
            push!(_segs, p.name)
          elseif p isa Absyn.QUALIFIED
            push!(_segs, p.name)
            _walk(p.path)
          elseif p isa Absyn.FULLYQUALIFIED
            _walk(p.path)
          end
        end
        _walk(path)
        local _short = if length(_segs) >= 2
          _segs[end-1] * "." * _segs[end]
        elseif length(_segs) == 1
          _segs[1]
        else
          Base.string(path)
        end
        _short * "(" * Base.string(int) * ")"
      end

      DAE.CREF(cr, _)  => begin
        string(cr)
      end

      DAE.UNARY(operator = op, exp = e1) => begin
        ("(" + string(op) + " " + string(e1) + ")")
      end

      DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        #= Parenthesize nested BINARY sub-expressions so the dump preserves
           operator precedence visually. Without this, `a - (b - c)` prints
           as `a - b - c`, which silently looks like `(a - b) - c` and has
           burned debugging time tracing what looked like a residual sign
           bug back into a correct AST. =#
        local s1 = (e1 isa DAE.BINARY || e1 isa DAE.LBINARY || e1 isa DAE.RELATION) ?
                   "(" + string(e1) + ")" : string(e1)
        local s2 = (e2 isa DAE.BINARY || e2 isa DAE.LBINARY || e2 isa DAE.RELATION) ?
                   "(" + string(e2) + ")" : string(e2)
        (s1 + " " + string(op) + " " + s2)
      end

      DAE.LUNARY(operator = op, exp = e1)  => begin
        ("(" + string(op) + " " + string(e1) + ")")
      end

      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local s1 = (e1 isa DAE.BINARY || e1 isa DAE.LBINARY || e1 isa DAE.RELATION) ?
                   "(" + string(e1) + ")" : string(e1)
        local s2 = (e2 isa DAE.BINARY || e2 isa DAE.LBINARY || e2 isa DAE.RELATION) ?
                   "(" + string(e2) + ")" : string(e2)
        (s1 + " " + string(op) + " " + s2)
      end

      DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
        (string(e1) + " " + string(op) + " " + string(e2))
      end

      DAE.IFEXP(expCond = e1, expThen = e2, expElse = e3) => begin
        ("IF_EXPRESSION: " + "($(string(e1)))" + " " + string(e2) + " ELSE " + string(e3))
      end

      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = expl)  => begin
        tmpStr = tmpStr + "(" + lstString(expl, ", ") + ")"
      end

      DAE.CALL(Absyn.QUALIFIED(__), _)  => begin
        tmpStr = string(exp.path; separator = OMBackend.COMPONENT_SEPARATOR) + "(" + lstString(exp.expLst, ", ") + ")"
      end

      DAE.RECORD(path = Absyn.IDENT(tmpStr), exps = expl)  => begin
        tmpStr = tmpStr + "[REC(" + lstString(expl, ", ") + ")"
      end

      DAE.PARTEVALFUNCTION(path = Absyn.IDENT(tmpStr), expList = expl)  => begin
        tmpStr = tmpStr + "[PARTEVAL](" + lstString(expl, ", ") + ")"
      end

      DAE.ARRAY(array = expl)  => begin
        _dumpArray(expl)
      end

      DAE.MATRIX(matrix = lstexpl)  => begin
        local nRows = 0
        local nCols = 0
        for lst in lstexpl
          nRows += 1
          if nRows == 1
            for _ in lst
              nCols += 1
            end
          end
        end
        if nRows * nCols > _ARRAY_DUMP_LIMIT
          "[MAT <" * string(nRows) * "×" * string(nCols) * " table>]"
        else
          str = "[MAT]"
          for lst in lstexpl
            str = str + "{" + lstString(lst, ", ") + "}"
          end
          str
        end
      end

      DAE.RANGE(start = e1, step = NONE(), stop = e2)  => begin
         string(e1) + ":" + string(e2)
      end

      DAE.RANGE(start = e1, step = SOME(e2), stop = e3)  => begin
         string(e1) + ":" + string(e2) + ":" + string(e3)
      end

      DAE.TUPLE(PR = expl) => begin
         "[TPL](" + lstString(expl, ", ") + ")"
      end

      DAE.CAST(ty = ty, exp = e1)  => begin
         string(ty) + string(e1)
      end

      DAE.ASUB(exp = e1, sub = expl)  => begin
         string(e1) + "[" + lstString(expl, ", ") + "]"
      end

      DAE.TSUB(exp = e1, ix = int) => begin
         "[TSUB]" + string(e1) + "(" + string(int) + ")"
      end

      DAE.RSUB(exp = e1)  => begin
        "[RSUB]" + string(e1)
      end

      DAE.SIZE(exp = e1, sz = NONE())  => begin
        "[SIZE]" + string(e1)
      end

      DAE.SIZE(exp = e1, sz = SOME(e2))  => begin
         "[SIZE]" + string(e1) + "(" + string(e2) + ")"
      end

     DAE.CODE(__) => begin
       "[CODE]"
     end

     DAE.REDUCTION(expr = e1) => begin
       "[REDUCTION]" + string(e1)
     end

     DAE.EMPTY(__)  => begin
       "[EMPTY]"
     end

     DAE.CONS(e1, e2)  => begin
       "[CONS]" + "{" + string(e1) + ", " + string(e2) + "}"
     end

     DAE.LIST(expl)  => begin
       "[LST]" + "{" + lstString(expl, ", ") + " }"
     end

    _ => begin
      str = string(repr(exp))
    end

    end
  end
end

function Base.string(ty::DAE.Type)::String
  @match ty begin
    DAE.T_INTEGER() => "(int) "
    DAE.T_REAL() => "(real) "
    DAE.T_STRING() => "(string) "
    DAE.T_BOOL() => "(bool) "
    DAE.T_ARRAY(__) => "(array of $(string(ty.ty))) "
    DAE.T_ENUMERATION(__) => "(enumeration) "
    DAE.T_COMPLEX(__) => "(complex type) "
    _ => "$(ty)"
  end
end

function Base.string(path::Absyn.QUALIFIED; separator = OMBackend.COMPONENT_SEPARATOR)
  return path.name + separator + string(path.path; separator = separator)
end

function Base.string(path::Absyn.IDENT; separator = OMBackend.COMPONENT_SEPARATOR)
  return path.name
end

function Base.string(path::Absyn.FULLYQUALIFIED; separator = OMBackend.COMPONENT_SEPARATOR)
  return separator + string(path.path; separator = separator)
end

function lstString(expLst::List{T}, separator::String)::String where{T}
  str = begin
    local e::T
    local rest::List{T}
    @match expLst begin
      (e <| rest) => begin
        str = string(e)
        for r in rest
          str = str + separator + string(r)
        end
        (str)
      end
      _ => begin
        str = ""
      end
    end
  end
end

function Base.string(ss::Cons{DAE.Subscript})::String
  local str = ""
  if length(ss) > 1
    for ix in ss
      str *= "_$(string(ix))"
    end
  else
    str *= string(ss.head)
  end
  return str
end

function Base.string(idx::DAE.INDEX)::String
  local str = string(idx.exp)
  return str
end

function Base.string(structuralIfEquation::BDAE.STRUCTURAL_IF_EQUATION)
  local str = "DYNAMIC_"
  local ifEqStr = replace(OMFrontend.Frontend.toString(structuralIfEquation.ifEquation), "\\n" => "\n")
  str *= ifEqStr * "\n"
end


function Base.string(stmt::DAE.STMT_ASSIGN)
  return string(stmt.exp1) * ":=" * string(stmt.exp) * "|" * string(stmt.type_)
end

function Base.string(stmt::DAE.STMT_TUPLE_ASSIGN)
  return string(stmt.expExpLst) * ":=" * string(stmt.exp) * "|" * string(stmt.type_)
end

function Base.string(stmt::DAE.STMT_ASSIGN_ARR)
  return string(stmt.lhs) * ":=" * string(stmt.exp) * "|" * string(stmt.type_)
end

function Base.string(stmt::DAE.STMT_WHILE)
  local buffer = IOBuffer()
  print(buffer, "WHILE ")
  println(buffer, string(stmt.exp))
  for s in stmt.statementLst
    println(buffer, string(s))
  end
  println(buffer, "END WHILE")
end

function Base.string(stmt::DAE.STMT_FOR)
  local buffer = IOBuffer()
  print(buffer, "FOR ")
  print(buffer, string(stmt.iter))
  println(buffer, " in " * string(stmt.range))
  for s in stmt.statementLst
    println(buffer, "  " * string(s))
  end
  println(buffer, "END FOR")
  return String(take!(buffer))
end

function Base.string(stmt::DAE.STMT_IF)
  local buffer = IOBuffer()
  print(buffer, "IF ")
  println(buffer, string(stmt.exp))
  for s in stmt.statementLst
    println(buffer, " " *string(s))
  end
  print(buffer, string(stmt.else_))
  return String(take!(buffer))
end

function Base.string(stmt::DAE.NOELSE)
  ""
end

function Base.string(stmt::DAE.ELSE)
  local buffer = IOBuffer()
  println(buffer, " ELSE")
  for s in stmt.statementLst
    println(buffer, " " *string(s))
  end
  println(buffer, " END IF")
  return String(take!(buffer))
end

function Base.string(stmt::DAE.ELSEIF)
  local buffer = IOBuffer()
  print(buffer, "ELSE IF")
  print(buffer, string(stmt.exp))
  for s in stmt.statementLst
    println(buffer, " " * string(s))
  end
  println(buffer, "END IF")
  print(buffer, string(stmt.else_))
  return String(take!(buffer))
end

function Base.string(v::DAE.VAR)
  string(v.componentRef)
end
