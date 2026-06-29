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

#= simCodeStructureTypes.jl

   SimCode-native type representation. `DAE.Type` carries frontend/back-end baggage
   the backend never reads: per-field attribute lists, bindings, enum literal `Var`s,
   the nominal class path. `SType` keeps only what codegen queries: a type's CATEGORY
   (`isa T_REAL` / `isa T_ARRAY` ...), an array's element type + integer dimensions,
   and for record/Complex types the field directory (names + element types) plus a
   RECORD-vs-connector routing bit that record scalarization reads.

   `SType` keeps exactly that closed subset, in the same spirit as `OpKind`
   (which dropped the operator-side type) and `Exp` (which dropped DAE.Exp
   frontend detail).

   Conversions:
   - `toSimType(::DAE.Type) -> SType`  (drops attrs; idempotent on SType)
   - `toDAEType(::SType)    -> DAE.Type` (round-trip-safe on category/shape and on
     the record field directory; lossy only on never-read attrs/bindings/paths).
=#

"""
    SType

Abstract root for SimCode-native types. Closed taxonomy distilled from the
`DAE.T_*` variants the backend actually reads (`T_REAL`/`T_INTEGER`/`T_BOOL`/
`T_STRING`/`T_ARRAY`/`T_ENUMERATION`/`T_COMPLEX`/`T_TUPLE`/`T_UNKNOWN`).
"""
abstract type SType end

"Scalar `Real`."
struct TYPE_REAL    <: SType end
"Scalar `Integer`."
struct TYPE_INTEGER <: SType end
"Scalar `Boolean`."
struct TYPE_BOOL    <: SType end
"Scalar `String`."
struct TYPE_STRING  <: SType end
"`T_UNKNOWN` / `T_ANYTYPE` / any unmapped DAE type."
struct TYPE_UNKNOWN <: SType end
"Record / `Complex` / connector type. Holds the field directory (names + element
types) and `isRecord` (RECORD ClassInf vs connector/other), which is all the
record-scalarization codegen reads off the type. The nominal path is dropped."
struct TYPE_COMPLEX <: SType
  isRecord::Bool
  fieldNames::Vector{String}
  fieldTypes::Vector{SType}
end

"Enumeration type. The backend only checks `isa TYPE_ENUM` (the category) — an
enum type's path/literals are never read off the type (the literal identity lives
on the `ENUM_LITERAL` Exp) — so this is a marker."
struct TYPE_ENUM <: SType end

"Array of `elementType` with integer `dims` (`-1` = unknown/flexible dimension)."
struct TYPE_ARRAY <: SType
  elementType::SType
  dims::Vector{Int}
end

"Tuple type (multi-result function return, multi-assign LHS)."
struct TYPE_TUPLE <: SType
  types::Vector{SType}
end

# ---- DAE.Type -> SType ----

_dimToInt(@nospecialize(d))::Int = @match d begin
  DAE.DIM_INTEGER(i) => Int(i)
  _ => -1
end
_dimsToSim(@nospecialize(ds))::Vector{Int} = Int[_dimToInt(d) for d in ds]

# Field directory (names + element STypes) from a `T_COMPLEX` varLst.
function _complexFieldsToSim(@nospecialize(varLst))::Tuple{Vector{String}, Vector{SType}}
  local names = String[]
  local types = SType[]
  for v in varLst
    if v isa DAE.TYPES_VAR
      push!(names, v.name)
      push!(types, toSimType(v.ty))
    end
  end
  return (names, types)
end

"""
    toSimType(ty) -> SType

Project a `DAE.Type` to its SimCode category. Drops attribute lists / enum
literal Vars / the complex ClassInf node. Idempotent on `SType`. Unmapped
DAE variants (functions, subtypes, code, …) collapse to `TYPE_UNKNOWN`.
"""
Base.@nospecializeinfer function toSimType(@nospecialize(ty))::SType
  @match ty begin
    DAE.T_REAL(__)    => TYPE_REAL()
    DAE.T_INTEGER(__) => TYPE_INTEGER()
    DAE.T_BOOL(__)    => TYPE_BOOL()
    DAE.T_STRING(__)  => TYPE_STRING()
    DAE.T_ENUMERATION(__) => TYPE_ENUM()
    DAE.T_ARRAY(ty = et, dims = ds) => TYPE_ARRAY(toSimType(et), _dimsToSim(ds))
    DAE.T_COMPLEX(cc, vl, _) => begin
      local (ns, ts) = _complexFieldsToSim(vl)
      TYPE_COMPLEX(cc isa DAE.ClassInf.RECORD, ns, ts)
    end
    DAE.T_TUPLE(types = ts) => TYPE_TUPLE(SType[toSimType(t) for t in ts])
    _ => TYPE_UNKNOWN()
  end
end
toSimType(t::SType)::SType = t

# ---- SType -> DAE.Type ----

_dimsToDAE(dims::Vector{Int}) =
  MetaModelica.list((d < 0 ? DAE.DIM_UNKNOWN() : DAE.DIM_INTEGER(d) for d in dims)...)

"""
    toDAEType(t::SType) -> DAE.Type

Reverse projection. Lossy only on never-read attribute fields (uses the
`*_DEFAULT` parameterless DAE types), round-trip-safe on category, array/tuple
shape, and the record field directory. `TYPE_COMPLEX` rebuilds a `T_COMPLEX`
with a synthetic path and per-field `TYPES_VAR`s so record scalarization
(`flattenRecordCallArg` and friends) reads back the field names + types.
"""
toDAEType(::TYPE_REAL)::DAE.Type    = DAE.T_REAL_DEFAULT
toDAEType(::TYPE_INTEGER)::DAE.Type = DAE.T_INTEGER_DEFAULT
toDAEType(::TYPE_BOOL)::DAE.Type    = DAE.T_BOOL_DEFAULT
toDAEType(::TYPE_STRING)::DAE.Type  = DAE.T_STRING_DEFAULT
toDAEType(::TYPE_UNKNOWN)::DAE.Type = DAE.T_UNKNOWN()
toDAEType(t::TYPE_COMPLEX)::DAE.Type =
  DAE.T_COMPLEX(t.isRecord ? DAE.ClassInf.RECORD(Absyn.IDENT("COMPLEX"))
                           : DAE.ClassInf.CONNECTOR(Absyn.IDENT("COMPLEX"), false),
                MetaModelica.list((DAE.TYPES_VAR(t.fieldNames[i], DAE.dummyAttrVar,
                                                 toDAEType(t.fieldTypes[i]), DAE.UNBOUND(), NONE())
                                   for i in 1:length(t.fieldNames))...),
                NONE())
#= path/names are never read off an enum type (only `isa T_ENUMERATION` is),
   so reconstruct a placeholder that preserves the category for round-trips. =#
toDAEType(::TYPE_ENUM)::DAE.Type =
  DAE.T_ENUMERATION(NONE(), Absyn.IDENT("ENUM"), MetaModelica.nil, MetaModelica.nil, MetaModelica.nil)
toDAEType(t::TYPE_ARRAY)::DAE.Type =
  DAE.T_ARRAY(toDAEType(t.elementType), _dimsToDAE(t.dims))
toDAEType(t::TYPE_TUPLE)::DAE.Type =
  DAE.T_TUPLE(MetaModelica.list((toDAEType(x) for x in t.types)...), NONE())
toDAEType(t::DAE.Type)::DAE.Type = t
