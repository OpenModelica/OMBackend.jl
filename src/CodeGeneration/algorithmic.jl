#=
Code generation for algorithmic Modelica.
author:johti17
=#

#= Check if a DAE.VAR is a multi-dimensional array (2+ dimensions). =#
#= Dimensions may be in v.ty (T_ARRAY) or in v.dims field. =#
function isMultiDimArray(v::DAE.VAR)::Bool
  #= First check if ty is T_ARRAY with 2+ dims =#
  tyHasMultiDims = @match v.ty begin
    DAE.T_ARRAY(dims = dims) => length(dims) >= 2
    _ => false
  end
  if tyHasMultiDims
    return true
  end
  #= Also check v.dims field (used for function parameters) =#
  try
    dimCount = 0
    for _ in v.dims
      dimCount += 1
    end
    return dimCount >= 2
  catch
    return false
  end
end

#= Check if a ModelicaFunction has any array-typed output.
   Used to decide whether the function wrapper should skip the symbolic short-circuit:
   array-returning functions must always execute their body so the result is indexable. =#
function hasArrayOutput(f::SimulationCode.ModelicaFunction)::Bool
  for v in f.outputs
    local crefType = @match v.componentRef begin
      DAE.CREF_IDENT(_, identType, _) => identType
      DAE.CREF_QUAL(_, identType, _, _) => identType
      _ => v.ty
    end
    @match crefType begin
      DAE.T_ARRAY(__) => return true
      _ => nothing
    end
  end
  return false
end

#= Check if a function body contains conditionals that would fail with symbolic args.
   This includes STMT_IF statements and IFEXP (ternary if-expressions) in assignment RHS.
   Only these functions need the symbolic Term dispatch; pure-arithmetic functions
   work fine when called directly with symbolic values.
   Checks recursively inside for-loops and other compound statements. =#
function hasIfStatements(func::SimulationCode.ModelicaFunction)::Bool
  @match func begin
    SimulationCode.MODELICA_FUNCTION(statements = stmts, locals = locals) => begin
      _statementsContainIf(stmts) && return true
      for v in locals
        @match v.binding begin
          SOME(bindingExp) => begin
            _expContainsIfExp(bindingExp) && return true
          end
          _ => nothing
        end
      end
      return false
    end
    _ => return false
  end
end

function _statementsContainIf(stmts)::Bool
  for s in stmts
    if s isa DAE.STMT_IF
      return true
    elseif s isa DAE.STMT_FOR
      _statementsContainIf(s.statementLst) && return true
    elseif s isa DAE.STMT_WHILE
      _statementsContainIf(s.statementLst) && return true
    elseif s isa DAE.STMT_ASSIGN
      _expContainsIfExp(s.exp) && return true
    end
  end
  return false
end

function _expContainsIfExp(exp::DAE.Exp)::Bool
  @match exp begin
    DAE.IFEXP(__) => return true
    DAE.BINARY(exp1 = e1, exp2 = e2) => begin
      return _expContainsIfExp(e1) || _expContainsIfExp(e2)
    end
    DAE.UNARY(exp = e1) => begin
      return _expContainsIfExp(e1)
    end
    DAE.CALL(expLst = args) => begin
      for a in args
        _expContainsIfExp(a) && return true
      end
      return false
    end
    DAE.ARRAY(array = elems) => begin
      for e in elems
        _expContainsIfExp(e) && return true
      end
      return false
    end
    DAE.ASUB(exp = inner) => begin
      return _expContainsIfExp(inner)
    end
    _ => return false
  end
end

#= Compute the output dimensions for a single-array-output function.
   Returns a tuple of ints, e.g. (4,) for a vector or (3,3) for a matrix.
   Returns () if not applicable (multiple outputs, unknown dimensions, etc.). =#
function computeArrayOutputDims(f::SimulationCode.ModelicaFunction)::Tuple{Vararg{Int}}
  if length(f.outputs) != 1
    return ()
  end
  local v = f.outputs[1]
  local crefType = @match v.componentRef begin
    DAE.CREF_IDENT(_, identType, _) => identType
    DAE.CREF_QUAL(_, identType, _, _) => identType
    _ => v.ty
  end
  @match crefType begin
    DAE.T_ARRAY(_, dims) => begin
      local dimVals = Int[]
      for d in dims
        @match d begin
          DAE.DIM_INTEGER(n) => push!(dimVals, n)
          _ => return ()
        end
      end
      return Tuple(dimVals)
    end
    _ => return ()
  end
end

#= Generate ensureArray conversion statements for multi-dimensional array parameters. =#
function generateArrayConversions(inputs::Vector)::Vector{Expr}
  conversions = Expr[]
  for v in inputs
    if isMultiDimArray(v)
      varName = Symbol(string(v.componentRef))
      push!(conversions, :($varName = OMBackend.CodeGeneration.ensureArray($varName)))
    end
  end
  return conversions
end

"""
  Generates algorithmic Modelica Code.
  Returns the generated Julia code + the names of the functions that has been generated.

  To avoid world-age issues when these functions are called from MTK's RuntimeGeneratedFunctions,
  we use a two-step approach:
  1. Generate the implementation as an anonymous function stored in MODELICA_FUNCTION_IMPLS dictionary
  2. Create a wrapper function at module load time that looks up the implementation

  The wrapper is created via createModelicaFunctionWrapper which must be called before
  the implementation is stored. This is handled in ODE_MODE_MTK_PROGRAM_GENERATION.
"""
function generateFunctions(functions::Vector{SimulationCode.ModelicaFunction})::Tuple{Vector{Expr}, Vector{String}}
  local jFuncs = Expr[]
  local names = String[]
  for func in functions
    local inputs = generateIOL(func.inputs)
    local outputs = generateIOL(func.outputs)
    local f
    #= Normalize function name: replace dots with underscores for valid Julia identifiers =#
    local normalizedName = replace(func.name, "." => "_")
    local nArgs = length(inputs)
    local isArrayFunc = hasArrayOutput(func)
    local needsSymbolicArrayDispatch = isArrayFunc && hasIfStatements(func)
    local outputDims = needsSymbolicArrayDispatch ? computeArrayOutputDims(func) : ()
    inputsJL = if nArgs > 1
      tuple(inputs...)
    elseif nArgs == 1
      inputs[1]
    else
      ()  #= Empty tuple for no inputs =#
    end
    @match func begin
      SimulationCode.MODELICA_FUNCTION(__) => begin
        local locals = generateLocals(func.locals)
        local arrayConversions = generateArrayConversions(func.inputs)
        local statements = generateStatements(func.statements)
        local returnExpr = if length(outputs) > 1
          Expr(:tuple, outputs...)
        elseif length(outputs) == 1
          outputs[1]
        else
          nothing
        end
        #= Build the anonymous function expression manually to avoid parsing issues =#
        local funcBody = Expr(:block, arrayConversions..., locals..., statements..., :(return $(returnExpr)))
        local anonFunc = if nArgs == 0
          Expr(:->, Expr(:tuple), funcBody)
        elseif inputsJL isa Tuple
          Expr(:->, Expr(:tuple, inputsJL...), funcBody)
        else
          Expr(:->, inputsJL, funcBody)
        end

        f = quote
          #= Create the wrapper function (always re-created to apply correct flags) =#
          OMBackend.CodeGeneration.createModelicaFunctionWrapper($(QuoteNode(Symbol(normalizedName))), $(nArgs), $(isArrayFunc), $(outputDims))
          #= Store the implementation in the dictionary =#
          OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[$(QuoteNode(Symbol(normalizedName)))] = $(anonFunc)
        end
      end
      SimulationCode.EXTERNAL_MODELICA_FUNCTION(__) => begin
        local extCall = Meta.parse(func.libInfo)
        extCall = namespaceifyExternalFunction(extCall)
        local returnExpr = if length(outputs) > 1
          Expr(:tuple, outputs...)
        elseif length(outputs) == 1
          outputs[1]
        else
          nothing
        end

        #= Build the anonymous function expression manually =#
        local funcBody = Expr(:block, extCall, returnExpr)
        local anonFunc = if nArgs == 0
          Expr(:->, Expr(:tuple), funcBody)
        elseif inputsJL isa Tuple
          Expr(:->, Expr(:tuple, inputsJL...), funcBody)
        else
          Expr(:->, inputsJL, funcBody)
        end

        f = quote
          #= Create the wrapper function (always re-created to apply correct flags) =#
          OMBackend.CodeGeneration.createModelicaFunctionWrapper($(QuoteNode(Symbol(normalizedName))), $(nArgs), $(isArrayFunc), $(outputDims))
          #= Store the implementation in the dictionary =#
          OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[$(QuoteNode(Symbol(normalizedName)))] = $(anonFunc)
        end
      end
    end
    push!(jFuncs, f)
    push!(names, normalizedName)
  end
  return jFuncs, names
end

function generateIOL(inputs::Vector)::Vector{Symbol}
  local jInputs::Vector{Symbol} = Symbol[]
  for i in inputs
    #= Check if this is a record type. If so, flatten into individual field parameters =#
    local flattenedInputs::Vector{Symbol} = flattenRecordInput(i)
    if !isempty(flattenedInputs)
      append!(jInputs, flattenedInputs)
    else
      local s = DAE_VAR_ToJulia(i)
      #= Complex type, prefixed with void* =#
      push!(jInputs, s)
    end
  end
  return jInputs
end

"""
  Check if a DAE.VAR is a record type and flatten it into individual field parameters.
  Returns a vector of Symbols for the flattened fields, or empty vector if not a record.
"""
function flattenRecordInput(v::DAE.VAR)::Vector{Symbol}
  local baseName::String = string(v.componentRef)
  @match v.ty begin
    DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), varLst, _) => begin
      local flattenedSymbols::Vector{Symbol} = Symbol[]
      for field in varLst
        @match field begin
          DAE.TYPES_VAR(fieldName, _, _, _, _) => begin
            local flatName::String = baseName * "_" * fieldName
            push!(flattenedSymbols, Symbol(flatName))
          end
          _ => nothing
        end
      end
      return flattenedSymbols
    end
    _ => return Symbol[]
  end
end

"""
`generateSignatureForRegistration(inputs::Vector{DAE.VAR})`
This function generates the input signature for calls to Symbolics.register.
Record inputs are flattened into individual field parameters to match the
generated function signature from generateIOL/flattenRecordInput.
"""
function generateSignatureForRegistration(inputs::Vector{DAE.VAR})
  local jInputs = Expr[]
  for i in inputs
    @match i.ty begin
      DAE.T_COMPLEX(DAE.ClassInf.RECORD(__), _, _) => begin
        #= Flatten record into individual field parameters =#
        local flattenedSymbols = flattenRecordInput(i)
        for s in flattenedSymbols
          push!(jInputs, Expr(:(::), s, :(Any)))
        end
      end
      _ => begin
        local s = DAE_VAR_ToJulia(i)
        push!(jInputs, Expr(:(::), s, :(Any)))
      end
    end
  end
  return jInputs
end

function generateLocals(inputs::Vector)
  local jInputs = Expr[]
  for i in inputs
    local s = DAE_VAR_ToJulia(i)
    #= If the variable has a binding expression (e.g., protected constants),
       generate local s = <bindingExpr> instead of just local s =#
    local hasBinding = @match i.binding begin
      SOME(bindingExp) => begin
        local bindExpr = expToJuliaExpAlg(bindingExp)
        push!(jInputs, :(local $s = $bindExpr))
        true
      end
      _ => false
    end
    if !hasBinding
      push!(jInputs, Expr(:local, s))
    end
  end
  return jInputs
end

function generateStatements(statements::Union{List{DAE.Statement}, Vector{DAE.Statement}})
  local jStmts = Expr[]
  for s in statements
    stmt = generateStatement(s)
    push!(jStmts, stmt)
  end
  return jStmts
end

function generateStatement(s::DAE.Statement)
  throw("Unsupported stmt:" * string(s))
end

function generateStatement(stmt::DAE.STMT_ASSIGN)
  local lhs = string(stmt.exp1)
  local rhs = expToJuliaExpAlg(stmt.exp)
  return :($(Symbol(lhs)) = $(rhs))
end

function generateStatement(stmt::DAE.STMT_TUPLE_ASSIGN)
  local lhs = string(stmt.expExpLst)
  local rhs = expToJuliaExpAlg(stmt.exp)
  return :($(Symbol(lhs)) = $(rhs))
end

function generateStatement(stmt::DAE.STMT_ASSIGN_ARR)
  local lhs = string(stmt.lhs)
  local rhs = expToJuliaExpAlg(stmt.exp)
  return :($(Symbol(lhs)) = $(rhs))
end

function generateStatement(stmt::DAE.STMT_WHILE, simCode)
  local buffer = IOBuffer()
  local cond = expToJuliaExpAlg(stmt.exp)
  local stmts = generateStatements(stmt.statementLst)
  quote
    while ($(cond))
      $(stmts...)
    end
  end
end

"""
  Generates for statements.
"""
function generateStatement(stmt::DAE.STMT_FOR)::Expr
  local iterVar = Symbol(stmt.iter)
  local rangeExpr = expToJuliaExpAlg(stmt.range)
  local bodyStmts = generateStatements(stmt.statementLst)
  local blck = Expr(:block)
  for s in bodyStmts
    push!(blck.args, s)
  end
  return Expr(:for, Expr(:(=), iterVar, rangeExpr), blck)
end

"""
  Generates If statements
"""
function generateStatement(stmt::DAE.STMT_IF)::Expr
  local cond = expToJuliaExpAlg(stmt.exp)
  local stmts = generateStatements(stmt.statementLst)
  local res = @match stmt.else_ begin
    DAE.NOELSE(__) => begin
      local expr = Expr(:if, cond)
      local blck = Expr(:block)
      for stmt in stmts
        push!(blck.args, stmt)
      end
      push!(expr.args, blck)
      expr
    end
    DAE.ELSE(__) => begin
      local expr = Expr(:if, cond)
      local blck = Expr(:block)
      for stmt in stmts
        push!(blck.args, stmt)
      end
      push!(expr.args, blck)
      local elseStmts = generateStatement(stmt.else_)
      push!(expr.args, elseStmts)
      expr
    end
    DAE.ELSEIF(__) => begin
      local expr = Expr(:if, cond)
      local blck = Expr(:block)
      for stmt in stmts
        push!(blck.args, stmt)
      end
      push!(expr.args, blck)
      local elseIfs = generateStatement(stmt.else_)
      push!(expr.args, elseIfs)
      expr
    end
  end
  return res
end

"""
For the else branch we generate a block and add the statements of the ELSE to this block.
Should never be called at the top level.
"""
function generateStatement(stmt::DAE.ELSE)::Expr
  local block = Expr(:block)
  stmts = generateStatements(stmt.statementLst)
  for stmt in stmts
    push!(block.args, stmt)
  end
  return block
end

"""
For the elseif branch we create an elseif expression.
Similar to the else this should never be called from the top level.
"""
function generateStatement(stmt::DAE.ELSEIF)::Expr
  local cond = expToJuliaExpAlg(stmt.exp)
  local stmts = generateStatements(stmt.statementLst)
  local blck = Expr(:block)
  for s in stmts
    push!(blck.args, s)
  end
  local elseExpr = @match stmt.else_ begin
    DAE.NOELSE(__) => nothing
    _ => generateStatement(stmt.else_)
  end
  if elseExpr === nothing
    Expr(:elseif, cond, blck)
  else
    Expr(:elseif, cond, blck, elseExpr)
  end
end

"""
  Generates Julia code for Modelica assert statements.
  If level is error (index 1), throws an error when condition is false.
  If level is warning (index 2), prints a warning when condition is false.
"""
function generateStatement(stmt::DAE.STMT_ASSERT)::Expr
  local condExpr = expToJuliaExpAlg(stmt.cond)
  local msgExpr = expToJuliaExpAlg(stmt.msg)
  #= Check assertion level: error (1) or warning (2) =#
  local level = @match stmt.level begin
    DAE.ENUM_LITERAL(_, idx) => idx
    _ => 1  #= Default to error =#
  end
  if level == 1
    #= AssertionLevel.error - throw an error =#
    quote
      if !($condExpr)
        error($msgExpr)
      end
    end
  else
    #= AssertionLevel.warning - print a warning =#
    quote
      if !($condExpr)
        @warn $msgExpr
      end
    end
  end
end

"""
  Maps a DAE expression to a Julia expression for algorithmic code in Modelica Functions(!).
  Since functions do not use the model HT the original name is preserved for algorithmic generation.
For algorithmic code outside Modelica functions do not call this function.
"""
function expToJuliaExpAlg(@nospecialize(exp::DAE.Exp))::Expr
  local expr::Expr = begin
    local int::Int64
    local real::Float64
    local bool::Bool
    local tmpStr::String
    local cr::DAE.ComponentRef
    local e1::DAE.Exp
    local e2::DAE.Exp
    local e3::DAE.Exp
    local expl::List{DAE.Exp}
    local lstexpl::List{List{DAE.Exp}}
    @match exp begin
      DAE.BCONST(bool) => quote $bool end
      DAE.ICONST(int) => quote $int end
      DAE.RCONST(real) => quote $real end
      DAE.SCONST(tmpStr) => quote $tmpStr end
      DAE.CREF(Absyn.IDENT("time"), _) => begin
        quote t end
      end
      #= Array accesses for simple CREF_IDENT =#
      DAE.CREF(DAE.CREF_IDENT(ident, identType, subscriptLst), _) where !isempty(subscriptLst)  => begin
        #= Extract base name if ident contains subscripts =#
        local baseName = if occursin('[', ident)
          ident[1:findfirst('[', ident)-1]
        else
          ident
        end
        #= Convert all subscripts to Julia index expressions =#
        local idxExprs = map(subscriptLst) do sub
          @match sub begin
            DAE.INDEX(e) => expToJuliaExpAlg(e)
            _ => throw("Unsupported subscript in algorithmic code: $sub")
          end
        end
        #= Construct proper Julia multi-dimensional indexing: arr[i, j, ...] =#
        expr = Expr(:ref, Symbol(baseName), idxExprs...)
      end
      #= Qualified CREF (record field access like R.T) =#
      DAE.CREF(DAE.CREF_QUAL(ident, identType, qualSubscriptLst, innerCref), _) => begin
        local varName::String = SimulationCode.string(exp.componentRef)
        #= Replace dots with underscores to match flattened parameter names =#
        local flatName::String = replace(varName, "." => "_")
        #= If the CREF has subscripts anywhere, parse it as an array access =#
        local allSubscripts = CodeGeneration.FrontendUtil.Util.getSubscriptsFromCref(exp.componentRef)
        if !isempty(allSubscripts)
          expr = Meta.parse(flatName)
        else
          quote
            $(Symbol(flatName))
          end
        end
      end
      DAE.CREF(cr, _)  => begin
        local varName::String = SimulationCode.string(cr)
        quote
          $(Symbol(varName))
        end
      end
      DAE.UNARY(operator = op, exp = e1) => begin
        o = CodeGeneration.DAE_OP_toJuliaOperator(op)
        :($(o)($(expToJuliaExpAlg(e1))))
      end
      DAE.BINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpAlg(e1)
        local rhs = expToJuliaExpAlg(e2)
        #= Special handling for vector dot product and matrix product =#
        @match op begin
          DAE.MUL_SCALAR_PRODUCT(__) => begin
            :(OMBackend.CodeGeneration.vectorDot($(lhs), $(rhs)))
          end
          DAE.MUL_MATRIX_PRODUCT(__) => begin
            :(OMBackend.CodeGeneration.ensureMatrix($(lhs)) * OMBackend.CodeGeneration.ensureMatrix($(rhs)))
          end
          _ => begin
            local opSym = CodeGeneration.DAE_OP_toJuliaOperator(op)
            :($opSym($(lhs), $(rhs)))
          end
        end
      end
      DAE.LUNARY(operator = op, exp = e1)  => begin
        local operand = expToJuliaExpAlg(e1)
        local op = CodeGeneration.DAE_OP_toJuliaOperator(op)
        :($op($(operand)))
      end
      DAE.LBINARY(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpAlg(e1)
        local rhs = expToJuliaExpAlg(e2)
        #= || and && are special forms in Julia, not regular functions.
           Must use Expr(:||, ...) / Expr(:&&, ...) instead of Expr(:call, :||, ...) =#
        @match op begin
          DAE.OR(__) => Expr(:||, lhs, rhs)
          DAE.AND(__) => Expr(:&&, lhs, rhs)
          _ => begin
            local opSym = CodeGeneration.DAE_OP_toJuliaOperator(op)
            :($opSym($(lhs), $(rhs)))
          end
        end
      end
      DAE.RELATION(exp1 = e1, operator = op, exp2 = e2) => begin
        local lhs = expToJuliaExpAlg(e1)
        local rhs = expToJuliaExpAlg(e2)
        local op = CodeGeneration.DAE_OP_toJuliaOperator(op)
        :($op($(lhs), $(rhs)))
      end
      DAE.IFEXP(expCond = e1, expThen = e2, expElse = e3) => begin
        local cond = expToJuliaExpAlg(e1)
        local thenExp = expToJuliaExpAlg(e2)
        local elseExp = expToJuliaExpAlg(e3)
        quote
          if $(cond)
            $(thenExp)
          else
            $(elseExp)
          end
        end
      end
      DAE.CALL(path = Absyn.IDENT(tmpStr), expLst = explst, attr = attr)  => begin
        local funcSym = Symbol(tmpStr)
        #= Use Base.invokelatest for non-builtin functions to avoid world-age issues =#
        local callTarget = if !(attr.builtin)
          :(Base.invokelatest)
        elseif haskey(MODELICA_BUILTIN_FUNCTIONS, tmpStr)
          Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(MODELICA_BUILTIN_FUNCTIONS[tmpStr]))
        else
          funcSym
        end
        local expr = Expr(:call, callTarget)
        if !(attr.builtin)
          push!(expr.args, funcSym)
        end
        local args = map(explst) do arg
          expToJuliaExpAlg(arg)
        end
        expr.args = vcat(expr.args, args)
        quote
          $(expr)
        end
      end
      DAE.CALL(path, expLst, attr) => begin
        local funcName = string(path)
        local funcSym = Symbol(funcName)
        #= Use Base.invokelatest for non-builtin functions to avoid world-age issues =#
        local callTarget = if !(attr.builtin)
          :(Base.invokelatest)
        elseif haskey(MODELICA_BUILTIN_FUNCTIONS, funcName)
          Expr(:., Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(:AlgorithmicCodeGeneration)), QuoteNode(MODELICA_BUILTIN_FUNCTIONS[funcName]))
        else
          funcSym
        end
        local expr = Expr(:call, callTarget)
        if !(attr.builtin)
          push!(expr.args, funcSym)
        end
        local args = map(expLst) do arg
          expToJuliaExpAlg(arg)
        end
        expr.args = vcat(expr.args, args)
        expr
      end
      DAE.CAST(ty, exp)  => begin
        #= Type cast expression =#
        local innerExpr = expToJuliaExpAlg(exp)
        @match ty begin
          DAE.T_REAL(__) => :(float($innerExpr))
          DAE.T_INTEGER(__) => :(Int(round($innerExpr)))
          DAE.T_BOOL(__) => :(Bool($innerExpr))
          _ => innerExpr  #= For other types, just return the inner expression =#
        end
      end
      DAE.ARRAY(ty, scalar, expl) => begin
        local elements = map(expl) do e
          expToJuliaExpAlg(e)
        end
        local arrExpr = Expr(:vect, elements...)
        #= If elements are themselves arrays (matrix literal), wrap with ensureArray
           to convert Vector{Vector{T}} to a proper Matrix{T} =#
        isNested = @match ty begin
          DAE.T_ARRAY(__) => true
          _ => false
        end
        if isNested
          :(OMBackend.CodeGeneration.ensureArray($arrExpr))
        else
          arrExpr
        end
      end
      DAE.RANGE(ty, startExp, stepExp, stopExp) => begin
        local startExpr = expToJuliaExpAlg(startExp)
        local stopExpr = expToJuliaExpAlg(stopExp)
        if stepExp === nothing
          :($startExpr:$stopExpr)
        else
          local stepExpr = expToJuliaExpAlg(stepExp)
          :($startExpr:$stepExpr:$stopExpr)
        end
      end
      DAE.SIZE(arrExp, SOME(dimExp)) => begin
        local arrExpr = expToJuliaExpAlg(arrExp)
        local dimExpr = expToJuliaExpAlg(dimExp)
        :(size($arrExpr, $dimExpr))
      end
      DAE.SIZE(arrExp, NONE()) => begin
        local arrExpr = expToJuliaExpAlg(arrExp)
        :(size($arrExpr))
      end
      DAE.ENUM_LITERAL(name, index) => begin
        #= Enum literals are represented by their integer index =#
        quote $index end
      end
      DAE.ASUB(arrExp, subLst) => begin
        #= Array subscript expression: arr[i, j, ...] =#
        local arrExpr = expToJuliaExpAlg(arrExp)
        local subs = map(expToJuliaExpAlg, subLst)
        Expr(:ref, arrExpr, subs...)
      end
      DAE.RECORD(path, exps, fieldNames, ty) => begin
        #= Record constructor: generate as a simple tuple =#
        local fieldExprs = map(expToJuliaExpAlg, exps)
        if length(fieldExprs) == 1
          #= Single element - return as-is or wrap =#
          first(fieldExprs)
        else
          Expr(:tuple, fieldExprs...)
        end
      end
      _ =>  throw(ErrorException("$exp not yet supported"))
    end
  end
  return expr
end

"""
  Converts a DAE var to an equvivalent Julia repr.
  Simple for now.
"""
function DAE_VAR_ToJulia(v::DAE.VAR)
  local vName = string(v.componentRef)
  Symbol(vName)
end

"""
  Adds OMRuntimeExternalC as a prefix to external calls
"""
function namespaceifyExternalFunction(expr::Expr)
  res = if expr.head == :(=)
    local callExpr = last(expr.args)
    @match Expr(:call, [funcName, y...,z]) = callExpr
    exp = Expr(:call, Expr(:(.), Symbol("OMRuntimeExternalC"), QuoteNode(funcName)), y...,z)
    #= Add the prefixes call to the right hand side of the expression. =#
    expr.args[2] = exp
    expr
  else #Otherwise a side effect call or a call that returns directly.
    @assert expr.head === :call "Invalid call passed to namespaceifyExternalFunction"
    @match Expr(:call, [funcName, y...,z]) = expr
    Expr(:call, Expr(:(.), Symbol("OMRuntimeExternalC"), QuoteNode(funcName)))
  end
  return res
end
