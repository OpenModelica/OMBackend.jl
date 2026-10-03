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
  Decomposes equations into chunked constructor functions.
  Returns (outer_defs, inner_refs) where:
  - outer_defs: function definitions at module level
  - inner_refs: parameter assignments + equationConstructorCalls array (inside model function)
  The `modelPrefix` avoids name collisions when multiple models are translated in one session.
"""
function decomposeEquations(equations, parameterAssignments; modelPrefix::String = "", chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local outerDefs = Expr[]
  local functionNames = Symbol[]
  local i = 0
  for equationVector in equationVectors
    local eqv = collect(equationVector)
    local fName = Symbol(modelPrefix, "generateEquations", i)
    push!(outerDefs, quote
      function $(fName)()
        [$(eqv...)]
      end
    end)
    push!(functionNames, fName)
    i += 1
  end
  local outerExpr = quote
    $(outerDefs...)
  end
  local innerExpr = quote
    $(parameterAssignments...)
    local equationConstructors::Vector{Function}
    local equationConstructorCalls::Vector
    equationConstructorCalls = [$(functionNames...)]
  end
  return (outerExpr, innerExpr)
end

"""
  Decomposes start equations into chunked constructor functions.
  Returns (outer_defs, inner_refs) where:
  - outer_defs: function definitions at module level
  - inner_refs: startEquationConstructors array assignment (inside model function)
  The `modelPrefix` avoids name collisions, `functionSuffix` differentiates initial vs final start eqs.
"""
function decomposeStartEquations(equations; functionSuffix = "", modelPrefix::String = "", chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local outerDefs = Expr[]
  local constructorNames = Symbol[]
  local i = 0
  for equationVector in equationVectors
    local fName = Symbol(modelPrefix, "generateStartEquations", functionSuffix, i)
    push!(outerDefs, quote
      function $(fName)()
        [$(equationVector...)]
      end
    end)
    push!(constructorNames, fName)
    i += 1
  end
  local outerExpr = quote
    $(outerDefs...)
  end
  local innerExpr = quote
    startEquationConstructors = Function[$(constructorNames...)]
  end
  return (outerExpr, innerExpr)
end

"""
  Inline variant of decomposeEquations that keeps function definitions inside
  the model function body. Equation expressions reference parameter symbols that
  are local to the model function (created by @parameters), so they cannot be
  moved to module level.
"""
function decomposeEquationsInline(equations, parameterAssignments; chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local exprs = Expr[]
  local functionNames = Symbol[]
  local constructors = quote
    $(parameterAssignments...)
    local equationConstructors::Vector{Function}
    local equationConstructorCalls::Vector{Function}
  end
  push!(exprs, constructors)
  local i = 0
  for equationVector in equationVectors
    local eqv = collect(equationVector)
    local (csEqs, csPreamble) = extractCommonHvcats(eqv)
    local fName = Symbol("generateEquations", i)
    if isempty(csPreamble)
      push!(exprs, quote
        function $(fName)()
          Symbolics.Equation[$(eqv...)]
        end
      end)
    else
      push!(exprs, quote
        function $(fName)()
          $(csPreamble...)
          Symbolics.Equation[$(csEqs...)]
        end
      end)
    end
    push!(functionNames, fName)
    i += 1
  end
  return quote
    $(exprs...)
    equationConstructorCalls = [$(functionNames...)]
  end
end

"""
  Inline variant of decomposeStartEquations that keeps function definitions inside
  the model function body. Start equations reference symbols that may be local to
  the model function (created by @parameters or phase 3 eval), so they cannot be
  moved to module level.
"""
function decomposeStartEquationsInline(equations; functionSuffix = "", chunkSize::Int = CHUNK_SIZE[])
  local equationVectors = collect(Iterators.partition(equations, chunkSize))
  local exprs = Expr[]
  local constructorNames = Symbol[]
  local i = 0
  for equationVector in equationVectors
    local eqv = collect(equationVector)
    local (csEqs, csPreamble) = extractCommonHvcats(eqv)
    local fName = Symbol("generateStartEquations", functionSuffix, i)
    if isempty(csPreamble)
      push!(exprs, quote
        function $(fName)()
          [$(eqv...)]
        end
      end)
    else
      push!(exprs, quote
        function $(fName)()
          $(csPreamble...)
          [$(csEqs...)]
        end
      end)
    end
    push!(constructorNames, fName)
    i += 1
  end
  return quote
    $(exprs...)
    startEquationConstructors = Function[$(constructorNames...)]
  end
end

"""
Generate code to declare ifCond variables as plain parameters (not time-dependent).
These parameters are modified by SymbolicContinuousCallback affects and are NOT
part of the ODE state vector, so the solver never perturbs them during Jacobian
computation. Using plain parameters (not `p(t)`) avoids MTK creating Shift operators.
Returns a no-op expression if there are no ifCond parameters.
"""
function generateDiscreteIfCondDeclaration(ifCondParamDecls::Vector{Expr}, ifCondNames::Vector{Symbol})
  if isempty(ifCondParamDecls)
    return :()
  end
  local nameQuotes = [QuoteNode(s) for s in ifCondNames]
  quote
    local _ifCondParams = ModelingToolkit.@parameters begin
      $(ifCondParamDecls...)
    end
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
  pars Dict closures and ARRAY_PARAMETERS code can reference them by name.
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
  Chunks parameter equations (sym => value pairs) into small inner functions
  to reduce JIT overhead from compiling one massive Dict literal.
  Each chunk function returns a Dict of parameter pairs. Results are
  merged into the final pars Dict via invokelatest.

  Parameter equations reference @parameters symbols which are local to the
  model function, so chunk functions are defined inline as closures.
"""
function decomposeParameterEquationsInline(parameterEquations; chunkSize = CHUNK_SIZE[])
  if length(parameterEquations) <= chunkSize
    return :(pars = Dict($(parameterEquations...)))
  end
  local chunks = collect(Iterators.partition(parameterEquations, chunkSize))
  local exprs = Expr[]
  local constructorNames = Symbol[]
  for (i, chunk) in enumerate(chunks)
    local fName = Symbol("_generatePars_", i - 1)
    local chunkExprs = collect(chunk)
    push!(exprs, quote
      function $(fName)()
        Dict($(chunkExprs...))
      end
    end)
    push!(constructorNames, fName)
  end
  return quote
    $(exprs...)
    pars = Dict{Any,Any}()
    for _parFn in [$(constructorNames...)]
      merge!(pars, Base.invokelatest(_parFn))
    end
  end
end

"""
  Optionally generate an import statement to OMRuntimeExternalC
"""
function generateExternalRuntimeImport()::Expr
  :(import OMRuntimeExternalC)
end
