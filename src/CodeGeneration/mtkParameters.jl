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

"""
  `createParameterEquationsMTK(parameters::Vector, type, simCode::SimulationCode.SIM_CODE)`
    The Type specifies what kind of parameter equation a call to this function should yield.
"""
function createParameterEquationsMTK(parameters::Vector, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local parameterEquations::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in parameters
    (index, simVar) = ht[param]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    bindExp = @match simVarType begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => begin
        exp
      end
      #= We have a parameter without a binding. Check if we have a start attribute...=#
      SimulationCode.PARAMETER(__) => begin
        local optAttributes::Option{DAE.VariableAttributes} = simVar.attributes
        @match optAttributes begin
          SOME(attr) where attr.start isa SOME => begin
            @assert !(attr.start.data isa DAE.CREF) "Non-numeric start attributes are not currently supported"
            @match SOME(startVal) = attr.start
            startVal
          end
          #= Either NONE() for missing attributes, or SOME(attr) whose start is
             NONE(). Both collapse to the default-float path. Without this
             catch-all the match fails on SOME{VariableAttributes} whose start
             is unset (e.g. several Blocks.Examples.Filter variants). =#
          _ => DAE.RCONST(0.0)
        end
      end
      SimulationCode.STRING(__) => begin
        @warn "String parameter $(param) found in numeric parameter list; skipping."
        continue
      end
      _ => begin
        throw(ErrorException("Unknown SimulationCode.SimVarType for parameter: " * string(param)  * " of type: " * string(simVarType)))
      end
    end
    #=
      Check if conversions are needed.
      Both sides of the Pair are wrapped with `Symbolics.wrap` to keep the
      pair element type at `Pair{Num, Num}`. Without the LHS wrap MTK fails
      `convert(Pair{Num}, Pair{BasicSymbolicImpl{SymReal}, Float64})` on
      models like SpeedControlledDCPM where the parameter symbol resolves
      to a bare `BasicSymbolic`. `Symbolics.wrap` is a no-op when the input
      is already a `Num`.
    =#
    expr = if isIntOrBool(bindExp)
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        Symbolics.wrap($(Symbol(simVar.name))) => Symbolics.wrap(float($((expToJuliaExpMTK(bindExp, simCode)))))
      end
    else
        :(Symbolics.wrap($(Symbol(simVar.name))) => Symbolics.wrap($(expToJuliaExpMTK(bindExp, simCode))))
    end
      # expr = quote
      #   $(LineNumberNode(@__LINE__, "$param eq"))
      #   $(Symbol(simVar.name)) => float($((expToJuliaExpMTK(bindExp, simCode))))
      # end
    push!(parameterEquations, expr)
  end #=For=#
  return parameterEquations
end

"""
  Creates array parameter definitions for MTK.
  Array parameters (e.g. record fields like R_T::Real[3,3]) are created as
  concrete Julia arrays assigned to their symbol names, so that the generated
  algorithmic functions can subscript into them.
"""
function createArrayParametersMTK(arrayParameters::Vector, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local exprs = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in arrayParameters
    (_, simVar) = ht[param]
    local vk = simVar.varKind
    @match vk begin
      SimulationCode.ARRAY_PARAMETER(dims, SOME(bindExp)) => begin
        local valExpr = expToJuliaExpMTK(bindExp, simCode)
        push!(exprs, :($(Symbol(simVar.name)) = $(valExpr)))
      end
      SimulationCode.ARRAY_PARAMETER(dims, NONE()) => begin
        #= AUDIT (ombackend-bug-audit-2026-06-05 #12): no binding expression.
           Mirror the scalar parameter paths (createParameterEquationsMTK /
           createParameterArray) and consult the declared start attribute before
           defaulting to zeros, so an unbound array parameter carrying a non-zero
           array-literal start is not silently materialized as all zeros. Only an
           explicit array-literal start is emitted directly (a scalar/other start
           has ambiguous broadcast shape); anything else falls through to a
           warned zeros materialization so the gap is attributable. =#
        local arrStart = @match simVar.attributes begin
          SOME(attr) where attr.start isa SOME => begin
            @match SOME(sv) = attr.start
            (sv isa DAE.ARRAY) ? expToJuliaExpMTK(sv, simCode) : nothing
          end
          _ => nothing
        end
        if arrStart === nothing
          @warn "[MTK GEN: createArrayParametersMTK] array parameter $(simVar.name): no binding and no array-literal start; materializing as zeros($(dims)). Any function subscripting it computes with zeros."
          push!(exprs, :($(Symbol(simVar.name)) = zeros(Float64, $(dims...))))
        else
          push!(exprs, :($(Symbol(simVar.name)) = $(arrStart)))
        end
      end
      _ => nothing
    end
  end
  return exprs
end

#= The binding the model function assigns a parameter (createParameterAssignmentsMTK),
   or nothing. A tunable parameter stays the symbolic @parameters variable: a parameter
   bound to it (`rate = 3 * k`) then lowers to a symbolic expression, so changing k
   changes rate too. One without a binding (a free parameter, fixed = false) is
   computed by the initialization. =#
function _assignedBinding(param, ht)
  SimulationCode.isTunableParameter(param) && return nothing
  return @match ht[param][2].varKind begin
    SimulationCode.PARAMETER(bindExp = SOME(exp)) => exp
    _ => nothing
  end
end

#= The parameters the model function assigns no value: the equations and the other
   parameters' bindings refer to them by name, as the symbolic parameters. =#
unassignedParameters(parameters::Vector, simCode::SimulationCode.SIM_CODE)::Set{Symbol} =
  Set{Symbol}(Symbol(p) for p in parameters if _assignedBinding(p, simCode.stringToSimVarHT) === nothing)

"""
  Creates parameters assignments *(:=) on a MTK parameters compatible format.
"""
function createParameterAssignmentsMTK(parameters::Vector,
                                       simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local parameterEquations::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  for param in parameters
    bindExp = _assignedBinding(param, ht)
    bindExp === nothing && continue
    (index, simVar) = ht[param]
    #= Solution for https://github.com/SciML/ModelingToolkit.jl/issues/991 =#
    #TODO: Is this workaround still relevant? John 2023-02-22
    expr =  if isIntOrBool(bindExp)
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        $(Symbol(simVar.name)) = float($((expToJuliaExpMTK(bindExp, simCode))))
      end
    else
      quote
        $(LineNumberNode(@__LINE__, "$param eq"))
        $(Symbol(simVar.name)) = $(expToJuliaExpMTK(bindExp, simCode))
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
function createStringParameterAssignments(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local exprs::Vector{Expr} = Expr[]
  local emitted = Set{String}()
  #= Variables the SimCode passes eliminated (aliases, folded equations): no
     longer in the table, and not bound at module level either. =#
  local eliminated = union(OrderedSet{String}(simCode.eliminatedVariables),
                           OrderedSet{String}(a.eliminatedName for a in simCode.aliasMap))
  for varName in keys(simCode.stringToSimVarHT)
    SimulationCode.isTunableParameter(varName) && continue
    (idx, simVar) = simCode.stringToSimVarHT[varName]
    local bindExp = @match simVar.varKind begin
      SimulationCode.STRING(bindExp = SOME(e)) => e
      SimulationCode.PARAMETER(bindExp = SOME(e)) where _isLiteralBind(e) => e
      _ => nothing
    end
    #= A String variable set by an equation (`s = if time > 0.5 then "ab"
       else "abc"`) was assigned nowhere: an UndefVarError at module load. =#
    bindExp === nothing && simVar.varKind isa SimulationCode.STRING &&
      OMBackend.unsupported("a String variable set by an equation", varName)
    bindExp === nothing && continue
    #= A String kind is a String parameter, but also a String variable: one
       reading time or a variable changes during the simulation, assigned here
       once (`t` undefined at module level). A String it reads is checked on
       its own. =#
    if simVar.varKind isa SimulationCode.STRING
      local d = SimulationCode.toDAEExp(bindExp)
      local varying = _expMentionsTime(d) || any(Util.getAllCrefs(d)) do c
        local e = get(simCode.stringToSimVarHT, string(c), nothing)
        e !== nothing && last(e).varKind isa Union{SimulationCode.STATE, SimulationCode.STATE_DERIVATIVE,
                                                   SimulationCode.ALG_VARIABLE, SimulationCode.DISCRETE,
                                                   SimulationCode.INPUT, SimulationCode.ARRAY}
      end
      varying && OMBackend.unsupported("a String variable that changes during the simulation", varName)
      local gone = [string(c) for c in Util.getAllCrefs(d) if string(c) in eliminated]
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
    push!(exprs, :( $(Symbol(simVar.name)) = $(rhs) ))
    push!(emitted, varName)
  end
  append!(exprs, _dataStructureParameterPrelude(simCode, emitted))
  return exprs
end

#= Parameters with a computed binding that a DATA_STRUCTURE's binding reads (MSL 4.1
   CombiTimeTable's isCsvExt from its file name, an argument of the table's constructor):
   at module level as well, each after what it reads; a parameter that reads anything
   else (a variable, an array or another data structure) is left as it was. =#
function _dataStructureParameterPrelude(simCode::SimulationCode.SIM_CODE, emitted::Set{String})::Vector{Expr}
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
    (sv.varKind isa SimulationCode.PARAMETER && !SimulationCode.isTunableParameter(nm)) || return false
    local b = sv.varKind.bindExp
    b isa SOME || return false
    push!(visiting, nm)
    local ok = all(c -> emit!(string(c)), Util.getAllCrefs(SimulationCode.toDAEExp(b.data)))
    local rhs = ok ? OMBackend._tryOr(() -> expToJuliaExpMTK(b.data, simCode), nothing, :dataStructureParameterPrelude;
                                     only = UnsupportedLowering, impact = :result) : nothing
    delete!(visiting, nm)
    rhs === nothing && (push!(failed, nm); return false)
    rhs isa Expr && !isempty(funcNames) && qualifyModelicaFunctions!(rhs, funcNames)
    push!(out, :($(Symbol(sv.name)) = $(rhs)))
    push!(emitted, nm)
    return true
  end
  for (_, (_, sv)) in ht
    sv.varKind isa SimulationCode.DATA_STRUCTURE || continue
    local b = sv.varKind.bindExp
    b isa SOME || continue
    foreach(c -> emit!(string(c)), Util.getAllCrefs(SimulationCode.toDAEExp(b.data)))
  end
  return out
end

#= Emit ARRAY_PARAMETER bindings at module top so that DATA_STRUCTURE
   constructor calls (CombiTable / CombiTimeTable / ExternalObject) can
   reference them by their bare Julia name. Without this, the in-function
   emission via createArrayParametersMTK happens too late: it lives inside
   `function <Model>Model(tspan)`, while DATA_STRUCTURE_ASSIGNMENTS run at
   module load time. =#
#= Module-level prelude for array parameters referenced by DATA_STRUCTURE
   constructors. Two cases:

     1. The HT carries a single ARRAY_PARAMETER entry with a literal-array bind.
        Emit `name = <array-expr>` directly.
     2. The HT carries scalarized entries (e.g. `tableData[1][1]`,
        `tableData[1][2]`, ..., `tableData[3][2]`) and the parent name has no
        bind of its own. Reconstruct an N-dim Julia matrix from the scalar
        element bindings and emit `tableData = <reconstructed>`. Required
        because ExternalObject constructors (CombiTable, CombiTimeTable, ...)
        appear at module top and reference the parent array name. =#
function createArrayParameterPrelude(simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local exprs::Vector{Expr} = Expr[]
  local ht = simCode.stringToSimVarHT

  #= Restrict the prelude to arrays actually referenced by DATA_STRUCTURE
     constructor calls. Emitting every ARRAY_PARAMETER at module top would
     shadow MTK's per-model parameter handling for arrays not needed at
     module-load time (e.g. body_r_CM in MultiBody models), perturbing the
     resulting integration trajectory. =#
  local neededBases = OrderedSet{String}()
  for (_, (_, simVar)) in ht
    @match simVar.varKind begin
      SimulationCode.DATA_STRUCTURE(SOME(b)) => begin
        @match b begin
          SimulationCode.CALL(__) => SimulationCode.collectCrefNames!(neededBases, b)
          _ => nothing
        end
      end
      _ => nothing
    end
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
    SimulationCode.collectCrefNames!(_residualCrefs, eq.exp)
  end
  for _n in _residualCrefs
    local _bracket = findfirst('[', _n)
    if _bracket !== nothing
      local _base = _n[1:_bracket-1]
      local _entry = get(ht, _base, nothing)
      _entry === nothing && continue
      local _isArr = @match _entry[2].varKind begin
        SimulationCode.ARRAY_PARAMETER(__) => true
        _ => false
      end
      _isArr && push!(neededBases, _base)
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
    local bindExp = @match simVar.varKind begin
      SimulationCode.ARRAY_PARAMETER(_, SOME(e)) => SimulationCode.toDAEExp(e)
      _ => nothing
    end
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
    local val = @match simVar.varKind begin
      SimulationCode.PARAMETER(SOME(SimulationCode.RCONST(r))) => r
      SimulationCode.PARAMETER(SOME(SimulationCode.ICONST(i))) => i
      SimulationCode.PARAMETER(SOME(SimulationCode.BCONST(b))) => b
      _ => nothing
    end
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

function createDataStructureAssignments(dataStructureVariables::Vector{String}, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local dsAssignments::Vector = Expr[]
  local ht = simCode.stringToSimVarHT
  #= Same Modelica-function name set used by rewriteEquations: needed to qualify
     bare calls (e.g. Modelica_Blocks_Types_ExternalCombiTimeTable_constructor)
     so they resolve to the OMBackend.CodeGeneration wrapper rather than failing
     with UndefVarError in the per-model module scope. Surfaces on every model
     using CombiTable / CombiTimeTable / ExternalObject constructors. =#
  local funcNames = OrderedSet{Symbol}(Symbol(f.name) for f in simCode.functions)
  for ds in dataStructureVariables
    (index, simVar) = ht[ds]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    #= An unbound DATA_STRUCTURE (bindExp NONE) is a dead/eliminated record
       field with no remaining reference (e.g. a Medium `data` field whose uses
       were constant-folded away); emit nothing rather than failing the model. =#
    bindExp = @match simVarType begin
      SimulationCode.DATA_STRUCTURE(bindExp = SOME(exp)) => exp
      _ => nothing
    end
    bindExp === nothing && continue
    local rhs = expToJuliaExpMTK(bindExp, simCode)
    if rhs isa Expr
      qualifyModelicaFunctions!(rhs, funcNames)
    end
    expr = quote
      $(LineNumberNode(@__LINE__, "$ds eq"))
      $(Symbol(simVar.name)) = $(rhs)
    end
    push!(dsAssignments, expr)
    #= A record's fields by their flattened names too (`Medium_data[1]_MM`): the equations
       read the record, the eliminated observed equations its fields (the MSL ideal-gas
       mixtures' molar masses: UndefVarError). =#
    if bindExp isa SimulationCode.RECORD && length(bindExp.fieldNames) == length(bindExp.exps)
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
        if fieldExp isa SimulationCode.ARRAY_EXP
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

#= For an array of records `base[1..n]` among the data structures, each field as an array
   `base_field = [base[1]_field, ...]`: a whole record array passed to a function goes field
   by field (`h_T(Medium_data_name, Medium_data_MM, ...)`). =#
function _recordArrayFieldArrays(dataStructureVariables::Vector{String}, simCode::SimulationCode.SIM_CODE)::Vector{Expr}
  local ht = simCode.stringToSimVarHT
  local elements = OrderedDict{String, Dict{Int, Vector{String}}}()
  for ds in dataStructureVariables
    local m = match(r"^(.*)\[(\d+)\]$", ds)
    m === nothing && continue
    local bindExp = @match ht[ds][2].varKind begin
      SimulationCode.DATA_STRUCTURE(bindExp = SOME(exp)) => exp
      _ => nothing
    end
    bindExp isa SimulationCode.RECORD || continue
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
function _foldParameterBindStatic(@nospecialize(exp), simCode::SimulationCode.SIM_CODE;
                                  depth::Int = 0)::Union{Float64, Nothing}
  depth > 32 && return nothing
  if exp isa SimulationCode.RCONST || exp isa SimulationCode.ICONST
    return Float64(exp.value)
  elseif exp isa SimulationCode.BCONST
    return exp.value ? 1.0 : 0.0
  elseif exp isa DAE.RCONST
    return Float64(exp.real)
  elseif exp isa DAE.ICONST
    return Float64(exp.integer)
  elseif exp isa DAE.BCONST
    return exp.bool ? 1.0 : 0.0
  elseif exp isa SimulationCode.ENUM_LITERAL || exp isa DAE.ENUM_LITERAL
    return Float64(exp.index)
  elseif exp isa SimulationCode.CAST
    return _foldParameterBindStatic(exp.exp, simCode; depth = depth + 1)
  elseif exp isa SimulationCode.UNARY
    local v = _foldParameterBindStatic(exp.exp, simCode; depth = depth + 1)
    v === nothing && return nothing
    local op = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(exp.op))
    return op === :- ? -v : (op === :+ ? v : nothing)
  elseif exp isa SimulationCode.BINARY
    local l = _foldParameterBindStatic(exp.exp1, simCode; depth = depth + 1)
    l === nothing && return nothing
    local r = _foldParameterBindStatic(exp.exp2, simCode; depth = depth + 1)
    r === nothing && return nothing
    local binop = DAE_OP_toJuliaOperator(SimulationCode.toDAEOperator(exp.op))
    binop === :+ && return l + r
    binop === :- && return l - r
    binop === :* && return l * r
    binop === :/ && return l / r
    binop === :^ && return l ^ r
    return nothing
  elseif exp isa SimulationCode.EXP_CREF
    exp.cref.sym === :time && return nothing
    local lookUpStr = isempty(exp.cref.subs) ?
      string(exp.cref.sym) :
      string(exp.cref.sym, "[", join(exp.cref.subs, ","), "]")
    local entry = get(simCode.stringToSimVarHT, lookUpStr, nothing)
    entry === nothing && return nothing
    local bind = @match entry[2].varKind begin
      SimulationCode.PARAMETER(bindExp = SOME(b)) => b
      _ => nothing
    end
    bind === nothing && return nothing
    return _foldParameterBindStatic(bind, simCode; depth = depth + 1)
  end
  return nothing
end

"""
  Creates a parameter array.
  A parameter array is an array containing the values of the parameters sorted by index.
  The index here is the index assigned by the code generator earlier in the lowering
  of the hybrid DAE.
"""
function createParameterArray(parameters::Vector{T1},
                              parameterAssignments::Vector{T2},
                              simCode::SIM_T) where {T1, T2, SIM_T}
  local paramArray = Union{Float64, Symbol}[]
  local hT = simCode.stringToSimVarHT
  for param in parameters
    (index, simVar) = hT[param]
    local simVarType::SimulationCode.SimVarType = simVar.varKind
    local hasBind::Bool = false
    bindExp = @match simVarType begin
      SimulationCode.PARAMETER(bindExp = SOME(exp)) => begin
        hasBind = true
        exp
      end
      SimulationCode.PARAMETER(__) => begin
        @match simVar.attributes begin
          SOME(attr) where attr.start isa SOME => begin
            @match SOME(startVal) = attr.start
            startVal
          end
          _ => DAE.RCONST(0.0)
        end
      end
      _ => throw(ErrorException("createParameterArray: parameter $(param) has no bound expression (got $(simVarType))."))
    end
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
