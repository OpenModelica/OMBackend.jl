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

#= MTK code generation: parameter equations, arrays, assignments and data structures. =#

#= A parameter binding in Julia (createParameterEquationsMTK, createParameterAssignmentsMTK).
   An element of the result of a call with literal arguments, `p[i, j] = f(...)[i, j]` (an
   array parameter scalarized), reads one evaluation of the call: `_parameterCalls`, a local
   of the model function, by the call's code. Every element called it (Buildings'
   TemperatureResponseMatrix: 152 calls of an impure function that computes a g-function
   and writes a file, 15 s each); a binding is evaluated once. =#
"""
`bindExp`: a binding, or a start value (a parameter without a binding).
"""
function _parameterBindingExpr(bindExp::SC.Exp, simCode::SC.SIM_CODE)
  local e = expToJuliaExpMTK(bindExp, simCode)
  _isLiteralCallElement(bindExp) || return e
  #= the indexing, inside the quote blocks of the translation =#
  local ref = e
  while ref isa Expr && ref.head === :block
    local stmts = filter(a -> !(a isa LineNumberNode), ref.args)
    length(stmts) == 1 || return e
    ref = only(stmts)
  end
  (ref isa Expr && ref.head === :ref && ref.args[1] isa Expr) || return e
  local callExpr = ref.args[1]
  ref.args[1] = :(get!(() -> $callExpr, _parameterCalls, $(hash(string(callExpr)))))
  return e
end

"""
f(...)[i, ...] with literal arguments and subscripts.
"""
_isLiteralCallElement(e::SC.Exp)::Bool =
  e isa SC.ASUB && e.exp isa SC.CALL && all(_isLiteralExp, e.exp.args) && all(s -> s isa SC.ICONST, e.subs)

"""
A parameter's value: its binding, else its start (a literal or an expression of other
parameters: MSL 4.1.0's lightning examples), else 0.0.
"""
function _parameterValueExp(simVar::SC.SIMVAR)::SC.Exp
  local k = simVar.varKind::SC.PARAMETER
  k.bindExp === nothing || return k.bindExp
  return something(simVar.attributes.start, SC.RCONST(0.0))
end

"""
    createParameterEquationsMTK(parameters, simCode)

The parameters' values as `p => value` pairs (a String parameter skipped). Both sides of a
pair are wrapped with `Symbolics.wrap` to keep the pair element type at `Pair{Num, Num}`:
without the left wrap MTK failed `convert(Pair{Num}, Pair{BasicSymbolicImpl{SymReal},
Float64})` where the parameter symbol resolves to a bare `BasicSymbolic` (SpeedControlledDCPM);
`Symbolics.wrap` is a no-op on a `Num`. An Integer or Boolean value as a float.
"""
function createParameterEquationsMTK(parameters::Vector, simCode::SC.SIM_CODE)::Vector{Expr}
  local parameterEquations = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in parameters
    local simVar = last(ht[param])
    if simVar.varKind isa SC.STRING
      @warn "String parameter $(param) found in numeric parameter list; skipping."
      continue
    end
    simVar.varKind isa SC.PARAMETER ||
      throw(ErrorException("Unknown SimVarType for parameter: " * string(param) * " of type: " * string(simVar.varKind)))
    local bindExp = _parameterValueExp(simVar)
    push!(parameterEquations, if isIntOrBool(bindExp)
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        Symbolics.wrap($(Symbol(simVar.name))) => Symbolics.wrap(float($(_parameterBindingExpr(bindExp, simCode))))
      end
    else
      :(Symbolics.wrap($(Symbol(simVar.name))) => Symbolics.wrap($(_parameterBindingExpr(bindExp, simCode))))
    end)
  end
  return parameterEquations
end

"""
  Creates array parameter definitions for MTK.
  Array parameters (e.g. record fields like R_T::Real[3,3]) are created as
  concrete Julia arrays assigned to their symbol names, so that the generated
  algorithmic functions can subscript into them.
"""
function createArrayParametersMTK(arrayParameters::Vector, simCode::SC.SIM_CODE)::Vector{Expr}
  local exprs = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in arrayParameters
    local simVar = last(ht[param])
    local vk = simVar.varKind
    vk isa SC.ARRAY_PARAMETER || continue
    if vk.bindExp !== nothing
      push!(exprs, :($(Symbol(simVar.name)) = $(expToJuliaExpMTK(vk.bindExp, simCode))))
      continue
    end
    #= AUDIT (ombackend-bug-audit-2026-06-05 #12): no binding expression. Mirror the scalar
       parameter paths (createParameterEquationsMTK / createParameterArray) and consult the
       declared start attribute before defaulting to zeros, so an unbound array parameter
       carrying a non-zero array-literal start is not silently materialized as all zeros. Only
       an explicit array-literal start is emitted directly (a scalar/other start has ambiguous
       broadcast shape); anything else falls through to a warned zeros materialization so the
       gap is attributable. =#
    local start = simVar.attributes.start
    if start isa SC.ARRAY_EXP
      push!(exprs, :($(Symbol(simVar.name)) = $(expToJuliaExpMTK(start, simCode))))
    else
      @warn "[MTK GEN: createArrayParametersMTK] array parameter $(simVar.name): no binding and no array-literal start; materializing as zeros($(vk.dimensions)). Any function subscripting it computes with zeros."
      push!(exprs, :($(Symbol(simVar.name)) = zeros(Float64, $(vk.dimensions...))))
    end
  end
  return exprs
end

"""
The binding the model function assigns a parameter (createParameterAssignmentsMTK),
or nothing. A tunable parameter stays the symbolic @parameters variable: a parameter
bound to it (`rate = 3 * k`) then lowers to a symbolic expression, so changing k
changes rate too. One without a binding (a free parameter, fixed = false) is
computed by the initialization.
"""
function _assignedBinding(param::String, ht::AbstractDict)::Union{Nothing, SC.Exp}
  OMBackend.isTunableParameter(param) && return nothing
  local k = last(ht[param]).varKind
  return k isa SC.PARAMETER ? k.bindExp : nothing
end

"""
The parameters the model function assigns no value: the equations and the other
parameters' bindings refer to them by name, as the symbolic parameters.
"""
unassignedParameters(parameters::Vector, simCode::SC.SIM_CODE)::Set{Symbol} =
  Set{Symbol}(Symbol(p) for p in parameters if _assignedBinding(p, simCode.stringToSimVarHT) === nothing)

"""
  Creates parameters assignments *(:=) on a MTK parameters compatible format.
"""
function createParameterAssignmentsMTK(parameters::Vector, simCode::SC.SIM_CODE)::Vector{Expr}
  local parameterEquations = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in parameters
    local bindExp = _assignedBinding(param, ht)
    bindExp === nothing && continue
    local simVar = last(ht[param])
    #= Solution for https://github.com/SciML/ModelingToolkit.jl/issues/991 =#
    #TODO: Is this workaround still relevant? John 2023-02-22
    local expr = if isIntOrBool(bindExp)
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        $(Symbol(simVar.name)) = float($(_parameterBindingExpr(bindExp, simCode)))
      end
    else
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        $(Symbol(simVar.name)) = $(_parameterBindingExpr(bindExp, simCode))
      end
    end
    push!(parameterEquations, expr)
  end
  return parameterEquations
end


"""
  createStringParameterAssignments(simCode) -> Vector{Expr}
Emit one module-level Julia assignment per Modelica `String` parameter, e.g.
```julia
table2_combiTimeTable_fileName = "NoName"
lossTable_fileName = "NoName"
```
"""
function createStringParameterAssignments(simCode::SC.SIM_CODE)::Vector{Expr}
  local exprs::Vector{Expr} = Expr[]
  local emitted = Set{String}()
  #= Variables the SimCode passes eliminated (aliases, folded equations): no
     longer in the table, and not bound at module level either. =#
  local eliminated = union(OrderedSet{String}(simCode.eliminatedVariables),
                           OrderedSet{String}(a.eliminatedName for a in simCode.aliasMap))
  #= at module level the model's functions are CodeGeneration's (Buildings' ShaGFunction:
     a String parameter bound to a function's SHA) =#
  local funcNames = OrderedSet{Symbol}(Symbol(f.name) for f in simCode.functions)
  for varName in keys(simCode.stringToSimVarHT)
    OMBackend.isTunableParameter(varName) && continue
    local simVar = last(simCode.stringToSimVarHT[varName])
    local k = simVar.varKind
    local bindExp = k isa SC.STRING ? k.bindExp :
      (k isa SC.PARAMETER && k.bindExp !== nothing && _isLiteralBind(k.bindExp)) ? k.bindExp : nothing
    #= A String variable set by an equation (`s = if time > 0.5 then "ab"
       else "abc"`) was assigned nowhere: an UndefVarError at module load. =#
    bindExp === nothing && simVar.varKind isa SC.STRING &&
      OMBackend.unsupported("a String variable set by an equation", varName)
    bindExp === nothing && continue
    #= A String kind is a String parameter, but also a String variable: one
       reading time or a variable changes during the simulation, assigned here
       once (`t` undefined at module level). A String it reads is checked on
       its own. =#
    if simVar.varKind isa SC.STRING
      local reads = SC.allCrefNames(bindExp)
      local varying = _expMentionsTime(bindExp) || any(reads) do c
        local e = get(simCode.stringToSimVarHT, c, nothing)
        e !== nothing && last(e).varKind isa Union{SC.STATE, SC.STATE_DERIVATIVE, SC.ALG_VARIABLE, SC.DISCRETE, SC.INPUT, SC.ARRAY}
      end
      varying && OMBackend.unsupported("a String variable that changes during the simulation", varName)
      local gone = [c for c in reads if c in eliminated]
      isempty(gone) || OMBackend.unsupported("a String variable that reads an eliminated variable",
                                             varName * ": " * join(gone, ", "))
    end
    #= Only emit literal bindings at module level. Computed defaults / cross-
       parameter refs cannot be safely lowered before MTK builds `pars`. The
       DATA_STRUCTURE_ASSIGNMENTS at module top reference these names (e.g.
       CombiTimeTable's `startTime` / `shiftTime` / `fileName`); without an
       emission step, loading the module raises UndefVarError. =#
    local rhs = try
      expToJuliaExpMTK(bindExp, simCode)
    catch _e
      OMBackend._fallback(_e, :stringParameterBinding; only = UnsupportedLowering, impact = :result)
      continue
    end
    rhs isa Expr && !isempty(funcNames) && qualifyModelicaFunctions!(rhs, funcNames)
    push!(exprs, :( $(Symbol(simVar.name)) = $(rhs) ))
    push!(emitted, varName)
  end
  append!(exprs, _dataStructureParameterPrelude(simCode, emitted))
  return exprs
end

"""
Parameters with a computed binding that a DATA_STRUCTURE's binding reads (MSL 4.1
CombiTimeTable's isCsvExt from its file name, an argument of the table's constructor):
at module level as well, each after what it reads; a parameter that reads anything
else (a variable, an array or another data structure) is left as it was.
"""
function _dataStructureParameterPrelude(simCode::SC.SIM_CODE, emitted::Set{String})::Vector{Expr}
  local ht = simCode.stringToSimVarHT
  local out = Expr[]
  local visiting = Set{String}()
  #= a parameter that cannot be computed at module level (it reads a variable, or its binding
     does not lower): not tried again from each parameter that reads it =#
  local failed = Set{String}()
  #= at module level the model's functions are CodeGeneration's (qualifyModelicaFunctions!) =#
  local funcNames = OrderedSet{Symbol}(Symbol(f.name) for f in simCode.functions)
  local emit!
  emit! = function (nm::String)
    nm in emitted && return true
    (nm in visiting || nm in failed) && return false
    local e = get(ht, nm, nothing)
    e === nothing && return false
    local sv = last(e)
    (sv.varKind isa SC.PARAMETER && !OMBackend.isTunableParameter(nm)) || return false
    local b = sv.varKind.bindExp
    b === nothing && return false
    push!(visiting, nm)
    local ok = all(emit!, SC.allCrefNames(b))
    local rhs = ok ? OMBackend._tryOr(() -> expToJuliaExpMTK(b, simCode), nothing, :dataStructureParameterPrelude;
                                     only = UnsupportedLowering, impact = :result) : nothing
    delete!(visiting, nm)
    rhs === nothing && (push!(failed, nm); return false)
    rhs isa Expr && !isempty(funcNames) && qualifyModelicaFunctions!(rhs, funcNames)
    push!(out, :($(Symbol(sv.name)) = $(rhs)))
    push!(emitted, nm)
    return true
  end
  for (_, (_, sv)) in ht
    (sv.varKind isa SC.DATA_STRUCTURE && sv.varKind.bindExp !== nothing) || continue
    foreach(emit!, SC.allCrefNames(sv.varKind.bindExp))
  end
  return out
end

#= Emit ARRAY_PARAMETER bindings at module top so that DATA_STRUCTURE
   constructor calls (CombiTable / CombiTimeTable / ExternalObject) can
   reference them by their bare Julia name. Without this, the in-function
   emission via createArrayParametersMTK happens too late: it lives inside
   `function <Model>Model(tspan)`, while DATA_STRUCTURE_ASSIGNMENTS run at
   module load time. =#
"""
Module-level prelude for array parameters referenced by DATA_STRUCTURE
constructors. Two cases:

  1. The HT carries a single ARRAY_PARAMETER entry with a literal-array bind.
     Emit `name = <array-expr>` directly.
  2. The HT carries scalarized entries (e.g. `tableData[1][1]`,
     `tableData[1][2]`, ..., `tableData[3][2]`) and the parent name has no
     bind of its own. Reconstruct an N-dim Julia matrix from the scalar
     element bindings and emit `tableData = <reconstructed>`. Required
     because ExternalObject constructors (CombiTable, CombiTimeTable, ...)
     appear at module top and reference the parent array name.
"""
function createArrayParameterPrelude(simCode::SC.SIM_CODE)::Vector{Expr}
  local exprs::Vector{Expr} = Expr[]
  local ht = simCode.stringToSimVarHT

  #= Restrict the prelude to arrays actually referenced by DATA_STRUCTURE
     constructor calls. Emitting every ARRAY_PARAMETER at module top would
     shadow MTK's per-model parameter handling for arrays not needed at
     module-load time (e.g. body_r_CM in MultiBody models), perturbing the
     resulting integration trajectory. =#
  local neededBases = OrderedSet{String}()
  for (_, (_, simVar)) in ht
    (simVar.varKind isa SC.DATA_STRUCTURE && simVar.varKind.bindExp isa SC.CALL) &&
      SC.crefNames!(neededBases, simVar.varKind.bindExp)
  end
  #= Also include any ARRAY_PARAMETER whose subscripted form appears in
     residual equations. This rescues `world_gravityArrowHead_lengthDirection[2]`
     and the cluster of MultiBody examples where `eliminateDeadParameters` /
     `eliminateConstantParameters` did not substitute the subscripted CREF
     (parent was kept as ARRAY_PARAMETER but never emitted module-top), so
     MTK eval fails with `<name>[idx] not defined`. We collect base names of
     CREFs that appear in residuals and intersect with the set of
     ARRAY_PARAMETERs in HT so we only emit parents that actually exist. =#
  local _residualCrefs = OrderedSet{String}()
  for eq in simCode.residualEquations
    SC.crefNames!(_residualCrefs, eq.exp)
  end
  for _n in _residualCrefs
    local _bracket = findfirst('[', _n)
    if _bracket !== nothing
      local _base = _n[1:_bracket-1]
      local _entry = get(ht, _base, nothing)
      _entry === nothing && continue
      _entry[2].varKind isa SC.ARRAY_PARAMETER && push!(neededBases, _base)
    end
  end
  #= Collect orphan subscripted CREFs (referenced in residuals, base NOT in HT)
     before the early-return so the defensive fallback at the end of this
     function still emits even when no DATA_STRUCTURE/ARRAY_PARAMETER paths
     fire. Recorded here so the fallback loop downstream can consume them. =#
  local _orphanRefsEarly = OrderedSet{String}()
  for _n in _residualCrefs
    local _bracket = findfirst('[', _n)
    _bracket === nothing && continue
    local _base = _n[1:_bracket-1]
    haskey(ht, _base) && continue
    haskey(ht, _n) && continue
    push!(_orphanRefsEarly, _n)
  end
  if isempty(neededBases)
    for _ref in _orphanRefsEarly
      push!(exprs, :( $(Symbol(_ref)) = 0.0 ))
    end
    return exprs
  end

  local emitted = OrderedSet{String}()
  for (varName, (_, simVar)) in ht
    local bindExp = simVar.varKind isa SC.ARRAY_PARAMETER ? simVar.varKind.bindExp : nothing
    bindExp === nothing && continue
    varName ∈ neededBases || continue
    varName ∈ emitted && continue
    push!(emitted, varName)
    local rhs = try
      expToJuliaExpMTK(bindExp, simCode)
    catch _e
      OMBackend._fallback(_e, :arrayParameterBinding; only = UnsupportedLowering, impact = :result)
      continue
    end
    push!(exprs, :( $(Symbol(simVar.name)) = $(rhs) ))
  end

  local scalarGroups = Dict{String, Vector{Tuple{Vector{Int}, Any, Int}}}()
  for (varName, (_, simVar)) in ht
    local bracketIdx = findfirst('[', varName)
    bracketIdx === nothing && continue
    local baseName = varName[1:bracketIdx-1]
    baseName ∈ neededBases || continue
    baseName ∈ emitted && continue
    local idxStr = varName[bracketIdx:end]
    local indices = Int[]
    for m in eachmatch(r"\[(\d+)\]", idxStr)
      push!(indices, parse(Int, m.captures[1]))
    end
    isempty(indices) && continue
    #= A literal binding; one of an expression of parameters, its value at the build (MSL 4
       BearingFriction's ExternalCombiTable1D(table = tau_pos) with RobotR3's tau_pos = [0, Rv0
       / unitTorque; ...]: the table was undefined). =#
    local b = simVar.varKind isa SC.PARAMETER ? simVar.varKind.bindExp : nothing
    local val = b === nothing ? nothing : b isa Union{SC.RCONST, SC.ICONST, SC.BCONST} ? b.value : SC.valueAtBuildStart(b, simCode)
    val === nothing && continue
    push!(get!(scalarGroups, baseName, Tuple{Vector{Int}, Any, Int}[]),
          (indices, val, length(indices)))
  end

  for (baseName, entries) in scalarGroups
    baseName ∈ emitted && continue
    local nDims = entries[1][3]
    all(e -> e[3] == nDims, entries) || continue
    local maxIdx = zeros(Int, nDims)
    for (idxs, _, _) in entries
      for d in 1:nDims
        maxIdx[d] = max(maxIdx[d], idxs[d])
      end
    end
    local expectedCount = prod(maxIdx)
    length(entries) == expectedCount || continue
    local elemType = isa(entries[1][2], Bool) ? Bool :
                     isa(entries[1][2], Integer) ? Int : Float64
    local arr = Array{elemType}(undef, maxIdx...)
    local complete = true
    for (idxs, val, _) in entries
      try
        arr[idxs...] = val
      catch _e
        OMBackend._fallback(_e, :arrayParameterElement)
        complete = false
        break
      end
    end
    complete || continue
    push!(emitted, baseName)
    push!(exprs, :( $(Symbol(baseName)) = $(arr) ))
  end

  #= Defensive fallback: a residual references `<name>[i]` for some base
     that is NOT in HT (no ARRAY_PARAMETER, no scalarized PARAMETER) and was
     therefore not emitted by either of the two passes above. Observed for
     the `World` component's `gravityArrowHead.lengthDirection[2..3]` cluster
     on MultiBody examples (RollingWheel / Surfaces / RollingWheelSetDriving
     / ...): the frontend instantiates the parent record but never registers
     the per-index parameters as SimVars, so codegen leaks subscripted CREFs
     into the residual that point at nothing. Emit `var"<name>[i]" = 0.0`
     for every observed index — the codegen produces `Symbol("<name>[i]")`
     bindings, so the recovery variable has to be the bracketed name itself,
     not the parent. Default value 0.0 is wrong if the model actually uses
     the parameter dynamically, but for visualization-only constants
     (`gravityArrowHead`, axis arrows, ...) it is benign. =#
  local _orphanRefs = OrderedSet{String}()
  for _n in _residualCrefs
    local _bracket = findfirst('[', _n)
    _bracket === nothing && continue
    local _base = _n[1:_bracket-1]
    haskey(ht, _base) && continue
    haskey(ht, _n) && continue
    push!(_orphanRefs, _n)
  end
  for _ref in _orphanRefs
    push!(exprs, :( $(Symbol(_ref)) = 0.0 ))
  end

  return exprs
end

"""
    createDataStructureAssignments(dataStructureVariables, simCode; owner = simCode.name) -> Vector{Expr}

The module-level assignments of the data structures (records, external objects) named in
`dataStructureVariables`; an external object is registered under the model `owner`, whose build
frees it (a structural mode's are its model's).
"""
function createDataStructureAssignments(dataStructureVariables::Vector{String}, simCode::SC.SIM_CODE;
                                        owner::String = simCode.name)::Vector{Expr}
  local dsAssignments::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  #= Same Modelica-function name set used by rewriteEquations: needed to qualify
     bare calls (e.g. Modelica_Blocks_Types_ExternalCombiTimeTable_constructor)
     so they resolve to the OMBackend.CodeGeneration wrapper rather than failing
     with UndefVarError in the per-model module scope. Surfaces on every model
     using CombiTable / CombiTimeTable / ExternalObject constructors. =#
  local funcNames = OrderedSet{Symbol}(Symbol(f.name) for f in simCode.functions)
  for ds in dataStructureVariables
    local simVar = last(ht[ds])
    #= An unbound DATA_STRUCTURE is a dead/eliminated record field with no remaining reference
       (a Medium `data` field whose uses were constant-folded away); emit nothing rather than
       failing the model. =#
    local bindExp = simVar.varKind isa SC.DATA_STRUCTURE ? simVar.varKind.bindExp : nothing
    bindExp === nothing && continue
    local rhs = expToJuliaExpMTK(bindExp, simCode)
    if rhs isa Expr
      qualifyModelicaFunctions!(rhs, funcNames)
    end
    #= an external object by its name in the terms that read it (ExternalObjectRef), owned by
       this model's build and freed with it by its class's destructor =#
    local callee = bindExp isa SC.CALL ? String(bindExp.name) : ""
    local destructor = endswith(callee, "_constructor") ? Symbol(replace(callee, r"_constructor$" => "_destructor")) : Symbol("")
    local expr = quote
      $(LineNumberNode(@__LINE__, "$ds eq"))
      $(Symbol(simVar.name)) = OMBackend.CodeGeneration._externalObjectRef($(QuoteNode(Symbol(simVar.name))), $(rhs),
                                                                        $(QuoteNode(destructor)), $(OMBackend.canonicalName(owner)))
    end
    push!(dsAssignments, expr)
    #= A record's fields by their flattened names too (`Medium_data[1]_MM`): the equations
       read the record, the eliminated observed equations its fields (the MSL ideal-gas
       mixtures' molar masses: UndefVarError). =#
    if bindExp isa SC.RECORD && length(bindExp.fieldNames) == length(bindExp.exps)
      for (field, fieldExp) in zip(bindExp.fieldNames, bindExp.exps)
        local fieldRhs = try
          expToJuliaExpMTK(fieldExp, simCode)
        catch _e
          OMBackend._fallback(_e, :recordFieldBinding; only = UnsupportedLowering, impact = :result)
          continue
        end
        local fieldSym = Symbol(simVar.name * "_" * field)
        push!(dsAssignments, :($(fieldSym) = $(fieldRhs)))
        #= An array field's elements by their scalarized names too (`Medium_data_alow[1]`). =#
        if fieldExp isa SC.ARRAY_EXP
          for k in eachindex(fieldExp.elements)
            push!(dsAssignments, :($(Symbol("$(fieldSym)[$(k)]")) = $(fieldSym)[$(k)]))
          end
        end
      end
    end
  end
  append!(dsAssignments, _recordArrayFieldArrays(dataStructureVariables, simCode))
  return dsAssignments
end

"""
For an array of records `base[1..n]` among the data structures, each field as an array
`base_field = [base[1]_field, ...]`: a whole record array passed to a function goes field
by field (`h_T(Medium_data_name, Medium_data_MM, ...)`).
"""
function _recordArrayFieldArrays(dataStructureVariables::Vector{String}, simCode::SC.SIM_CODE)::Vector{Expr}
  local ht = simCode.stringToSimVarHT
  local elements = OrderedDict{String, Dict{Int, Vector{String}}}()
  for ds in dataStructureVariables
    local m = match(r"^(.*)\[(\d+)\]$", ds)
    m === nothing && continue
    local k = ht[ds][2].varKind
    local bindExp = k isa SC.DATA_STRUCTURE ? k.bindExp : nothing
    bindExp isa SC.RECORD || continue
    get!(elements, m.captures[1], Dict{Int, Vector{String}}())[parse(Int, m.captures[2])] = bindExp.fieldNames
  end
  local out = Expr[]
  for (base, byIndex) in elements
    local n = length(byIndex)
    (all(i -> haskey(byIndex, i), 1:n) && allequal(values(byIndex))) || continue
    for field in byIndex[1]
      local parts = [Symbol("$(base)[$(i)]_$(field)") for i in 1:n]
      push!(out, :($(Symbol(base * "_" * field)) = [$(parts...)]))
    end
  end
  return out
end

"""
    _foldParameterBindStatic(exp, simCode; depth = 0)

Statically fold a parameter bind expression to a Float64. Returns nothing
when the expression depends on anything that is not a parameter chain
grounded in literals.
"""
function _foldParameterBindStatic(exp::SC.Exp, simCode::SC.SIM_CODE; depth::Int = 0)::Union{Float64, Nothing}
  depth > 32 && return nothing
  exp isa Union{SC.RCONST, SC.ICONST} && return Float64(exp.value)
  exp isa SC.BCONST && return exp.value ? 1.0 : 0.0
  exp isa SC.ENUM_LITERAL && return Float64(exp.index)
  exp isa SC.CAST && return _foldParameterBindStatic(exp.exp, simCode; depth = depth + 1)
  if exp isa SC.UNARY
    local v = _foldParameterBindStatic(exp.exp, simCode; depth = depth + 1)
    return v === nothing ? nothing : -v
  elseif exp isa SC.BINARY
    local l = _foldParameterBindStatic(exp.exp1, simCode; depth = depth + 1)
    l === nothing && return nothing
    local r = _foldParameterBindStatic(exp.exp2, simCode; depth = depth + 1)
    r === nothing && return nothing
    exp.op === SC.OP_ADD && return l + r
    exp.op === SC.OP_SUB && return l - r
    exp.op === SC.OP_MUL && return l * r
    exp.op === SC.OP_DIV && return l / r
    exp.op === SC.OP_POW && return l ^ r
    return nothing
  elseif exp isa SC.EXP_CREF
    exp.cref.sym === :time && return nothing
    local entry = get(simCode.stringToSimVarHT, SC.crefName(exp.cref), nothing)
    entry === nothing && return nothing
    local k = last(entry).varKind
    (k isa SC.PARAMETER && k.bindExp !== nothing) || return nothing
    return _foldParameterBindStatic(k.bindExp, simCode; depth = depth + 1)
  end
  return nothing
end

"""
  Creates a parameter array.
  A parameter array is an array containing the values of the parameters sorted by index.
  The index here is the index assigned by the code generator earlier in the lowering
  of the hybrid DAE.
"""
function createParameterArray(parameters::Vector, parameterAssignments::Vector, simCode::SC.SIM_CODE)
  local paramArray = Union{Float64, Symbol}[]
  local hT = simCode.stringToSimVarHT
  for param in parameters
    local simVar = last(hT[param])
    simVar.varKind isa SC.PARAMETER ||
      throw(ErrorException("createParameterArray: parameter $(param) has no bound expression (got $(simVar.varKind))."))
    local hasBind = simVar.varKind.bindExp !== nothing
    local bindExp = _parameterValueExp(simVar)
    #= Fold statically; a codegen-time module-scope eval can read stale
       same-named symbols left behind by previously translated models. =#
    local folded = _foldParameterBindStatic(bindExp, simCode)
    local parValue
    if folded !== nothing
      parValue = :($(folded))
    elseif hasBind
      #= Non-foldable bind (cross-parameter chain through a call, string,
         array): read the in-scope parameter assignment at runtime. =#
      parValue = :($(Symbol(param)))
    else
      @warn "[MTK GEN: createParameterArray] parameter $(param): no bind and non-literal start; substituting 0 in the legacy parameter array. Any event-driven read of this parameter via the aux[1] mirror will be wrong (the modern MTK path is unaffected)."
      parValue = :(0.0)
    end
    push!(paramArray, parValue)
  end
  return paramArray
end
