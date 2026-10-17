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

#= The lowering of OMJLIR's names, types and expressions to SimCode's. =#

#= ==================================================================================== =#
#= Names                                                                                =#
#= ==================================================================================== =#

"""
    Names

The simulation names of a model's references: a reference's flat name (`a.b[1,2]` is
`a_b[1][2]`, `legacyName`), and for the references whose flat names collide (`a.b` and a
variable `a_b`) a new one (`a_b_1`): the least qualified keeps the name. `known` are the
names of the simulation variables (with `der(x)` of the states and the built-in `time`,
`pi`, `e`): a reference to an element of an array known as a whole is the array's, subscripted.
`stateOf` maps a derivative variable `\$DER.x` to its state `x`. `scalars` are the known names
of variables that are not arrays.
"""
struct Names
  renamed::Dict{Symbol, String}
  known::Set{String}
  stateOf::Dict{Symbol, IR.ComponentRef}
  scalars::Set{String}
end

"""
    collidingNames(variables)

The new names of the variables whose flat names collide, by the variables' names: of each
group of references with one flat name, the least qualified (the first of those) keeps it, the
others are `<name>_1`, `<name>_2`, ... (names not taken). A reference with subscripts is not
renamed (with a warning).
"""
function collidingNames(variables::Vector{IR.Variable})::Dict{Symbol, String}
  local byFlatName = OrderedDict{String, OrderedDict{Symbol, IR.ComponentRef}}()
  for v in variables
    push!(get!(OrderedDict{Symbol, IR.ComponentRef}, byFlatName, legacyName(v.cref)), v.name => v.cref)
  end
  local taken = Set{String}(keys(byFlatName))
  local renamed = Dict{Symbol, String}()
  for (flat, claimants) in byFlatName
    length(claimants) < 2 && continue
    local keep = argmin(n -> length(claimants[n].parts), collect(keys(claimants)))
    for (name, cref) in claimants
      name === keep && continue
      if IR.hasSubscripts(cref)
        @warn "The flat name $(flat) of the subscripted variable $(name) collides; left unresolved"
        continue
      end
      local k = 1
      while string(flat, "_", k) in taken
        k += 1
      end
      local newName = string(flat, "_", k)
      push!(taken, newName)
      renamed[name] = newName
    end
  end
  return renamed
end

"""
    simName(names, cref)

The simulation name of `cref`: its flat name with every `.` as `_` (of an enumeration
literal's subscript: `x[Logic_'U'(1)]`), a quoted name as it is; or its new name.
"""
function simName(names::Names, cref::IR.ComponentRef)::String
  local r = isempty(names.renamed) || IR.hasSubscripts(cref) ? nothing : get(names.renamed, cref.name, nothing)
  r === nothing || return r
  local flat = legacyName(cref)
  (startswith(flat, '\'') && endswith(flat, '\'')) && return flat
  return replace(flat, '.' => '_')
end

"The name of the function, record or class `name` in the simulation code: its parts joined by `_`."
codeName(name::Symbol)::Symbol = Symbol(replace(String(name), '.' => '_'))

"An enumeration literal's name: its type's last name and the literal (`Logic.'U'`)."
enumLiteralName(ty::IR.T_ENUMERATION, literal::Symbol)::Symbol =
  Symbol(last(split(String(ty.name), '.')), ".", literal)

#= ==================================================================================== =#
#= Types                                                                                =#
#= ==================================================================================== =#

"The SimCode type of an OMJLIR type: an array's dimensions of unknown or non-Integer size are -1."
simType(::IR.T_REAL)::SType = TYPE_REAL()
simType(::IR.T_INTEGER)::SType = TYPE_INTEGER()
simType(::IR.T_BOOLEAN)::SType = TYPE_BOOL()
simType(::IR.T_STRING)::SType = TYPE_STRING()
simType(t::IR.T_ENUMERATION)::SType = TYPE_ENUM(Tuple(String[String(l) for l in t.literals]))
simType(t::IR.T_ARRAY)::SType =
  TYPE_ARRAY(simType(t.elementType), Int[d isa IR.DIM_SIZE ? d.size : -1 for d in t.dimensions])
simType(t::IR.T_RECORD)::SType =
  TYPE_COMPLEX(true, String[String(f.name) for f in t.fields], SType[simType(f.ty) for f in t.fields])
simType(::Union{IR.T_CLASS, IR.T_EXTERNAL_OBJECT})::SType = TYPE_COMPLEX(false, String[], SType[])
simType(t::IR.T_TUPLE)::SType = TYPE_TUPLE(SType[simType(x) for x in t.types])
simType(t::IR.T_FUNCTION)::SType = TYPE_FUNCTION(SType[simType(x) for x in t.inputs], SType[simType(x) for x in t.outputs])
simType(::IR.T_NORETCALL)::SType = TYPE_NORETCALL()
simType(::Union{IR.T_CLOCK, IR.T_UNKNOWN})::SType = TYPE_UNKNOWN()

#= ==================================================================================== =#
#= Operators                                                                            =#
#= ==================================================================================== =#

const OPERATORS = Dict{IR.Op, OpKind}(
  IR.ADD => OP_ADD, IR.SUB => OP_SUB, IR.MUL => OP_MUL, IR.DIV => OP_DIV, IR.POW => OP_POW,
  IR.ADD_EW => OP_ADD_EW, IR.SUB_EW => OP_SUB_EW, IR.MUL_EW => OP_MUL_EW, IR.DIV_EW => OP_DIV_EW,
  IR.POW_EW => OP_POW_EW, IR.ADD_SCALAR_ARRAY => OP_ADD_SCALAR_ARRAY, IR.ADD_ARRAY_SCALAR => OP_ADD_ARRAY_SCALAR,
  IR.SUB_SCALAR_ARRAY => OP_SUB_SCALAR_ARRAY, IR.SUB_ARRAY_SCALAR => OP_SUB_ARRAY_SCALAR,
  IR.MUL_SCALAR_ARRAY => OP_MUL_SCALAR_ARRAY, IR.MUL_ARRAY_SCALAR => OP_MUL_ARRAY_SCALAR,
  IR.MUL_VECTOR_MATRIX => OP_MUL_VECTOR_MATRIX, IR.MUL_MATRIX_VECTOR => OP_MUL_MATRIX_VECTOR,
  IR.SCALAR_PRODUCT => OP_SCALAR_PRODUCT, IR.MATRIX_PRODUCT => OP_MATRIX_PRODUCT,
  IR.DIV_SCALAR_ARRAY => OP_DIV_SCALAR_ARRAY, IR.DIV_ARRAY_SCALAR => OP_DIV_ARRAY_SCALAR,
  IR.POW_SCALAR_ARRAY => OP_POW_SCALAR_ARRAY, IR.POW_ARRAY_SCALAR => OP_POW_ARRAY_SCALAR,
  IR.POW_MATRIX => OP_POW_MATRIX, IR.UMINUS => OP_UMINUS, IR.AND => OP_AND, IR.OR => OP_OR, IR.NOT => OP_NOT,
  IR.LESS => OP_LESS, IR.LESSEQ => OP_LESSEQ, IR.GREATER => OP_GREATER, IR.GREATEREQ => OP_GREATEREQ,
  IR.EQUAL => OP_EQUAL, IR.NEQUAL => OP_NEQUAL)

simOp(op::IR.Op)::OpKind = OPERATORS[op]

"The element-wise kinds of the arithmetic operators, for array operands (the previous backend's `*_ARR`)."
const ELEMENTWISE = Dict{OpKind, OpKind}(OP_ADD => OP_ADD_EW, OP_SUB => OP_SUB_EW, OP_MUL => OP_MUL_EW,
                                         OP_DIV => OP_DIV_EW, OP_POW => OP_POW_EW)

"The kind of a binary operator of type `ty`: `+ - * / ^` of arrays element-wise."
function simOp(op::IR.Op, ty::IR.ModelicaType)::OpKind
  local k = simOp(op)
  return ty isa IR.T_ARRAY ? get(ELEMENTWISE, k, k) : k
end

#= ==================================================================================== =#
#= Expressions                                                                          =#
#= ==================================================================================== =#

"""
    simExp(names, e)

The SimCode expression of the OMJLIR expression `e`: references by their simulation names
(`simCrefExp`), a derivative variable `\$DER.x` as `der(x)`, `pre(x)` of a reference as PRE,
the size of a known dimension as its value, calls with their kind; BOX/UNBOX are gone.
"""
simExp(::Names, e::IR.INTEGER)::Exp = ICONST(e.value)
simExp(::Names, e::IR.REAL)::Exp = RCONST(e.value)
simExp(::Names, e::IR.BOOLEAN)::Exp = BCONST(e.value)
simExp(::Names, e::IR.STRING)::Exp = SCONST(e.value)
simExp(::Names, e::IR.ENUM_LITERAL)::Exp = ENUM_LITERAL(enumLiteralName(e.ty, e.name), e.index)
simExp(::Names, ::IR.WILD)::Exp = WILD()
simExp(names::Names, e::IR.CREF)::Exp = simCrefExp(names, e.cref, e.ty)
simExp(names::Names, e::IR.ARRAY)::Exp =
  ARRAY_EXP(simType(e.ty), length(e.ty.dimensions) == 1, simExps(names, e.elements))
simExp(names::Names, e::IR.RANGE)::Exp =
  RANGE(simType(e.ty), simExp(names, e.start), e.step === nothing ? nothing : simExp(names, e.step), simExp(names, e.stop))
simExp(names::Names, e::IR.TUPLE)::Exp = TUPLE(simExps(names, e.elements))
simExp(names::Names, e::IR.RECORD)::Exp =
  RECORD(codeName(e.ty.name), simExps(names, e.elements), String[String(f.name) for f in e.ty.fields], simType(e.ty))
simExp(names::Names, e::IR.ARRAY_CONSTRUCTOR)::Exp =
  REDUCTION(:array, simExp(names, e.body), simIterators(names, e.iterators), simType(e.ty))
simExp(names::Names, e::IR.REDUCTION)::Exp =
  REDUCTION(e.reducer, simExp(names, e.body), simIterators(names, e.iterators), simType(e.ty))
simExp(names::Names, e::IR.PARTIAL_APPLICATION)::Exp =
  PARTEVALFUNCTION(codeName(e.name), simExps(names, e.arguments), e.argumentNames,
                   Pair{Symbol, SType}[n => simType(t) for (n, t) in e.inputs], simType(e.ty)::TYPE_FUNCTION)
simExp(names::Names, e::IR.BINARY)::Exp = BINARY(simExp(names, e.exp1), simOp(e.op, e.ty), simExp(names, e.exp2))
simExp(names::Names, e::IR.UNARY)::Exp = UNARY(simOp(e.op), simExp(names, e.exp))
simExp(names::Names, e::IR.LBINARY)::Exp = LBINARY(simExp(names, e.exp1), simOp(e.op), simExp(names, e.exp2))
simExp(names::Names, e::IR.LUNARY)::Exp = LUNARY(simOp(e.op), simExp(names, e.exp))
simExp(names::Names, e::IR.RELATION)::Exp = RELATION(simExp(names, e.exp1), simOp(e.op), simExp(names, e.exp2), e.index)
simExp(names::Names, e::IR.IF_EXPRESSION)::Exp =
  IFEXP(simExp(names, e.condition), simExp(names, e.thenExp), simExp(names, e.elseExp))
simExp(names::Names, e::IR.CAST)::Exp = CAST(simType(e.ty), simExp(names, e.exp))
simExp(names::Names, e::IR.SUBSCRIPTED)::Exp = ASUB(simExp(names, e.exp), Exp[subscriptExp(names, s) for s in e.subscripts])
simExp(names::Names, e::IR.TUPLE_ELEMENT)::Exp = TSUB(simExp(names, e.exp), e.index, simType(e.ty))
simExp(names::Names, e::IR.RECORD_ELEMENT)::Exp = RSUB(simExp(names, e.exp), e.index, String(e.field), simType(e.ty))
simExp(::Names, e::IR.CLOCK)::Exp = error("SimCode lowering: a clock $(IR.toString(e)) after the clocked lowering")

function simExp(names::Names, e::IR.SIZE)::Exp
  local ty = IR.typeOf(e.exp)
  if e.dimension isa IR.INTEGER && ty isa IR.T_ARRAY && 1 <= e.dimension.value <= length(ty.dimensions)
    local d = ty.dimensions[e.dimension.value]
    d isa IR.DIM_SIZE && return ICONST(d.size)
  end
  local args = e.dimension === nothing ? Exp[simExp(names, e.exp)] : Exp[simExp(names, e.exp), simExp(names, e.dimension)]
  return CALL(:size, args, e.dimension === nothing ? TYPE_ARRAY(TYPE_INTEGER(), Int[-1]) : TYPE_INTEGER(), CALL_BUILTIN)
end

function simExp(names::Names, e::IR.CALL)::Exp
  if e.attributes.builtin && length(e.arguments) == 1 && e.arguments[1] isa IR.CREF &&
     !haskey(names.stateOf, e.arguments[1].cref.name)
    e.name === :der && return DER(simCref(names, e.arguments[1].cref))
    e.name === :pre && return PRE(simCref(names, e.arguments[1].cref), simType(e.ty))
  end
  return CALL(e.attributes.builtin ? e.name : codeName(e.name), simExps(names, e.arguments), simType(e.ty), callKind(e.attributes))
end

simExps(names::Names, es::Vector{IR.Expression})::Vector{Exp} = Exp[simExp(names, x) for x in es]

simIterators(names::Names, its::Vector{IR.Iterator})::Vector{Iterator} =
  Iterator[Iterator(it.name, simExp(names, it.range)) for it in its]

callKind(a::IR.CallAttributes)::CallKind = a.builtin ? CALL_BUILTIN : a.impure ? CALL_IMPURE_FUNCTION : CALL_FUNCTION

"A subscript's expression (`:` as WHOLE_DIMENSION)."
subscriptExp(names::Names, s::Union{IR.INDEX, IR.SLICE})::Exp = simExp(names, s.exp)
subscriptExp(::Names, ::IR.WHOLE)::Exp = WHOLE_DIMENSION()

#= ==================================================================================== =#
#= References                                                                           =#
#= ==================================================================================== =#

"""
    simCrefExp(names, cref, ty)

A reference of type `ty`: a simulation variable by its name; the derivative variable
`\$DER.x` as `der(x)`; an element of an array known as a whole as the array's reference with
the element's Integer subscripts (with a subscript that is an expression, a slice or `:`, as
the array subscripted: an iterator's element in a reduction, `p[2:n + 1]`); one whose single subscript that is not a
literal, on a part before the last, selects among variables (a loop's iterator: Buildings' DX
coils' `sta[i].nomVal`, `sta[iSta].perCur.capFunT[6]` in an initial algorithm) as the array of
their references, subscripted; a scalar subscripted by ones as the scalar; another reference
by its flat name.
"""
function simCrefExp(names::Names, cref::IR.ComponentRef, ty::IR.ModelicaType)::Exp
  local state = get(names.stateOf, cref.name, nothing)
  state === nothing || return DER(SimCref(Symbol(simName(names, state))))
  local full = simName(names, cref)
  full in names.known && return EXP_CREF(SimCref(Symbol(full)), simType(ty))
  if any(p -> any(s -> !(s isa IR.INDEX && s.exp isa Union{IR.INTEGER, IR.ENUM_LITERAL}), p.subscripts),
         @view cref.parts[1:end-1])
    local selected = variablesBySubscript(names, cref, ty)
    selected === nothing || return selected
  end
  local last = cref.parts[end]
  local base = simName(names, IR.ComponentRef(IR.CrefPart[cref.parts[1:end-1]..., IR.CrefPart(last.name, IR.Subscript[], last.ty)]))
  if isempty(last.subscripts) && !(base in names.known)
    return something(variablesBySubscript(names, cref, ty), Some(EXP_CREF(SimCref(Symbol(full)), simType(ty))))
  end
  #= a scalar subscripted by ones is the scalar: the frontend's equations of expandable
     connectors' scalar signals (`bus2.y1[1] = two.bus.y1[1]`, Buildings' Templates) =#
  (base in names.scalars && all(p -> isempty(p.subscripts), @view cref.parts[1:end-1]) &&
   all(s -> s isa IR.INDEX && s.exp isa IR.INTEGER && s.exp.value == 1, last.subscripts)) &&
    return EXP_CREF(SimCref(Symbol(base)), simType(ty))
  if all(s -> s isa IR.INDEX && s.exp isa IR.INTEGER, last.subscripts)
    return EXP_CREF(SimCref(Symbol(base), Int[s.exp.value for s in last.subscripts]), simType(ty))
  end
  all(s -> s isa IR.INDEX && s.exp isa Union{IR.INTEGER, IR.ENUM_LITERAL}, last.subscripts) &&
    return EXP_CREF(SimCref(Symbol(full)), simType(ty))
  return ASUB(EXP_CREF(SimCref(Symbol(base)), simType(last.ty)), Exp[subscriptExp(names, s) for s in last.subscripts])
end

"The SimCref of a reference (a variable's name)."
simCref(names::Names, cref::IR.ComponentRef)::SimCref = SimCref(Symbol(simName(names, cref)))

"""
    variablesBySubscript(names, cref, ty)

`{x[1], x[2], ...}[i]` for a reference `x[i]` whose one subscript `i` is not a literal (in any
part), its elements 1, 2, ... variables; `nothing` for another.
"""
function variablesBySubscript(names::Names, cref::IR.ComponentRef, ty::IR.ModelicaType)::Union{Nothing, Exp}
  local positions = Tuple{Int, Int}[(p, k) for (p, part) in enumerate(cref.parts) for (k, s) in enumerate(part.subscripts)
                                    if s isa IR.INDEX && !(s.exp isa Union{IR.INTEGER, IR.ENUM_LITERAL})]
  length(positions) == 1 || return nothing
  local (p, k) = only(positions)
  local index = cref.parts[p].subscripts[k].exp
  local withIndex = i -> IR.ComponentRef(IR.CrefPart[q == p ?
    IR.CrefPart(part.name, IR.Subscript[j == k ? IR.INDEX(IR.INTEGER(i)) : s for (j, s) in enumerate(part.subscripts)], part.ty) :
    part for (q, part) in enumerate(cref.parts)])
  local elements = Exp[]
  while simName(names, withIndex(length(elements) + 1)) in names.known
    push!(elements, EXP_CREF(SimCref(Symbol(simName(names, withIndex(length(elements) + 1)))), simType(ty)))
  end
  isempty(elements) && return nothing
  return ASUB(ARRAY_EXP(TYPE_ARRAY(simType(ty), Int[length(elements)]), true, elements), Exp[simExp(names, index)])
end
