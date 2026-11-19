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

#= MTK code generation: variable, equation and parameter declarations in chunks; function registration. =#

"""
 Decomposes the continuous variables into chunked constructor functions.
 Returns (outer_defs, inner_refs) where:
  - outer_defs: function definitions to be placed at module level (before model function)
  - inner_refs: the variableConstructors array assignment (inside model function)
 Constructor functions are defined at module level to avoid JIT overhead from
 compiling nested closures inside the model function.
 The `modelPrefix` avoids name collisions when multiple models are translated in one session.
"""
function decomposeVariables(stateVariables::Vector{Symbol}, algebraicVariables::Vector{Symbol};
                            modelPrefix::String = "")
  local stateVectors = collect(Iterators.partition(stateVariables, CHUNK_SIZE[]))
  local algVectors = collect(Iterators.partition(algebraicVariables, CHUNK_SIZE[]))
  local outerDefs = Expr[]
  local constructorNames = Symbol[]
  local i = 1::Int
  for stateVector in stateVectors
    local fName = Symbol(modelPrefix, "generateStateVariables", i)
    push!(outerDefs, quote
      function $(fName)()
        $(Tuple([stateVector...]))
      end
    end)
    push!(constructorNames, fName)
    i += 1
  end
  i = 1
  for algVector in algVectors
    local fName = Symbol(modelPrefix, "generateAlgebraicVariables", i)
    push!(outerDefs, quote
      function $(fName)()
        $(Tuple([algVector...]))
      end
    end)
    push!(constructorNames, fName)
    i += 1
  end
  local outerExpr = quote
    $(outerDefs...)
  end
  local innerExpr = quote
    variableConstructors = Function[$(constructorNames...)]
  end
  return (outerExpr, innerExpr)
end

"""
    decomposeEquationsInline(equations, parameterAssignments, components; chunkSize) -> Expr

The model function's code that pushes `equations` to its vector `components`, `chunkSize` at a time, each chunk as
`push!(components, Base.invokelatest(() -> Symbolics.Equation[...]))` after `parameterAssignments`. The chunks read
the model function's locals (its `@parameters`); `hoistBuildClosures!` makes each a function of the module that takes
them as arguments, since Julia's lowering of a function is quadratic in its closures (a 4000-element chain: 480 chunk
closures in the model function, 10 s to lower it).
"""
function decomposeEquationsInline(equations::Vector, parameterAssignments::Vector, components::Symbol;
                                  chunkSize::Int = CHUNK_SIZE[])::Expr
  local exprs::Vector{Any} = Any[parameterAssignments...]
  for equationVector in Iterators.partition(equations, chunkSize)
    local (csEqs, csPreamble) = extractCommonHvcats(collect(equationVector))
    push!(exprs, :(push!($(components), Base.invokelatest(() -> begin
      $(csPreamble...)
      Symbolics.Equation[$(csEqs...)]
    end))))
  end
  return Expr(:block, exprs...)
end

"""
    decomposeStartEquationsInline(equations, components; chunkSize) -> Expr

The model function's code that pushes the start equations (`x => value`) `equations` to its vector `components`,
`chunkSize` at a time, as `decomposeEquationsInline` does.
"""
function decomposeStartEquationsInline(equations::Vector, components::Symbol; chunkSize::Int = CHUNK_SIZE[])::Expr
  local exprs::Vector{Any} = Any[]
  for equationVector in Iterators.partition(equations, chunkSize)
    local (csEqs, csPreamble) = extractCommonHvcats(collect(equationVector))
    push!(exprs, :(push!($(components), Base.invokelatest(() -> begin
      $(csPreamble...)
      [$(csEqs...)]
    end))))
  end
  return Expr(:block, exprs...)
end

"""
Generate code to declare ifCond variables as plain parameters (not time-dependent).
These parameters are modified by SymbolicContinuousCallback affects and are NOT
part of the ODE state vector, so the solver never perturbs them during Jacobian
computation. Using plain parameters (not `p(t)`) avoids MTK creating Shift operators.
Returns a no-op expression if there are no ifCond parameters. Those named in `conditions` (the if-equations'
conditions, not the hysteresis or the homotopy parameter declared with them) are marked (`isIfConditionParameter`).
"""
function generateDiscreteIfCondDeclaration(ifCondParamDecls::Vector{Expr}, ifCondNames::Vector{Symbol},
                                           conditions::Vector{Symbol})
  if isempty(ifCondParamDecls)
    return :()
  end
  local nameQuotes::Vector{QuoteNode} = [QuoteNode(s) for s in ifCondNames]
  local marked::Vector{Bool} = Bool[s in conditions for s in ifCondNames]
  quote
    local _ifCondParams = [m ? Symbolics.wrap(SymbolicUtils.setmetadata(Symbolics.unwrap(p), OMBackend.CodeGeneration.IfConditionParameter, true)) : p
                           for (m, p) in zip($(marked), ModelingToolkit.@parameters begin
                             $(ifCondParamDecls...)
                           end)]
    local _ifCondBindBlock = Expr(:block)
    for (name, p) in zip([$(nameQuotes...)], _ifCondParams)
      push!(_ifCondBindBlock.args, :($name = $p))
    end
    eval(_ifCondBindBlock)
    parameters = vcat(parameters, _ifCondParams)
  end
end

"""
Generate code to add ifCond discrete parameter values to the pars Dict.
Returns a no-op expression if there are no ifCond parameters.
"""
function generateIfCondParamAssignments(ifCondParamPairs::Vector{Expr})
  if isempty(ifCondParamPairs)
    return :()
  end
  quote
    for (k, v) in [$(ifCondParamPairs...)]
      pars[k] = v
    end
  end
end

"""
  Chunks the @parameters macro call into inner functions to reduce the model
  function body size. Each inner function calls @parameters with a subset of
  parameter names and returns the resulting vector. Results are concatenated.

  After chunking, parameter symbols are eval'd into module scope so that
  pars Dict closures and the array parameters' code can reference them by name.
  `unassigned` (unassignedParameters) are also bound as locals.
"""
function decomposeParametersDeclaration(parVariablesSym; unassigned::Set{Symbol}, chunkSize = CHUNK_SIZE[])
  if length(parVariablesSym) <= chunkSize
    return quote
      parameters = ModelingToolkit.@parameters begin
        ($(parVariablesSym...))
      end
    end
  end
  local chunks = collect(Iterators.partition(parVariablesSym, chunkSize))
  local exprs = Expr[]
  local constructorNames = Symbol[]
  for (i, chunk) in enumerate(chunks)
    local fName = Symbol("_createParams_", i - 1)
    local chunkSyms = collect(chunk)
    push!(exprs, quote
      function $(fName)()
        ModelingToolkit.@parameters begin
          ($(chunkSyms...))
        end
      end
    end)
    push!(constructorNames, fName)
  end
  local paramNameQuotes = [QuoteNode(s) for s in parVariablesSym]
  #= A parameter the model function assigns no value (createParameterAssignmentsMTK
     skips it: a tunable one, or one without a binding such as a free parameter the
     initialization computes) is referred to by name as the symbolic parameter, by
     the equations and by other parameters' bindings. Bind it as a local from
     `parameters`: the global made by the `eval` below is too new for this
     function's world (UndefVarError: the MSL InitSpringConstant's
     `spring_spring_c = spring_c`). =#
  local localBinds = Expr[:($(s) = parameters[$(i)]) for (i, s) in enumerate(parVariablesSym) if s in unassigned]
  return quote
    $(exprs...)
    local _allParamChunks = Any[]
    for _fn in [$(constructorNames...)]
      push!(_allParamChunks, Base.invokelatest(_fn))
    end
    parameters = vcat(_allParamChunks...)
    local _paramNames = [$(paramNameQuotes...)]
    local _paramBindBlock = Expr(:block)
    for (name, p) in zip(_paramNames, parameters)
      push!(_paramBindBlock.args, :($name = $p))
    end
    eval(_paramBindBlock)
    $(localBinds...)
  end
end

"""
    decomposeParameterEquationsInline(parameterEquations; chunkSize) -> Expr

The model function's code that sets `pars`, the parameter equations' (`p => value`) Dict: in one literal up to
`chunkSize` of them, else `chunkSize` at a time, each chunk as `merge!(pars, Base.invokelatest(() -> Dict(...)))`
(a function of the module, `decomposeEquationsInline`).
"""
function decomposeParameterEquationsInline(parameterEquations::Vector; chunkSize::Int = CHUNK_SIZE[])::Expr
  if length(parameterEquations) <= chunkSize
    return :(pars = Dict($(parameterEquations...)))
  end
  local exprs::Vector{Any} = Any[:(pars = Dict{Any, Any}())]
  for chunk in Iterators.partition(parameterEquations, chunkSize)
    push!(exprs, :(merge!(pars, Base.invokelatest(() -> Dict($(chunk...))))))
  end
  return Expr(:block, exprs...)
end

"""
  Optionally generate an import statement to OMRuntimeExternalC
"""
function generateExternalRuntimeImport()::Expr
  :(import OMRuntimeExternalC)
end
