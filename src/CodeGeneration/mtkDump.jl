#=
  MTKDump — MTK-stage dump helpers.

  All `@BACKEND_LOGGING`-gated dumps that capture state at the OMBackend → MTK
  boundary live in this submodule, so the call sites in MTK_CodeGenerationUtil.jl,
  mtkExternals.jl, DirectRHSGeneration.jl, and MTK_CodeGeneration.jl reduce
  to a one-liner `MTKDump.dumpX(...)` invocation.

  When logging is off (`ENABLE_BACKEND_LOGGING != "true"` at module load), the
  `@BACKEND_LOGGING` macro compiles each function body to `nothing`, so these
  helpers cost nothing in production.

  Each helper:
    * is `@nospecialize`'d on heavy MTK types to keep compile latency down;
    * wraps its body in `try/catch + @warn` so a dump failure can never abort
      the simulation;
    * writes to `OMBackend.logPath("backend/mtk", "<filename>.log")`, which
      lands in the per-model run directory (the simulate-time
      `withLogRunDir` wrap in `simulateModel` pushes the translate-time
      runId looked up from `MODEL_RUN_DIRS`).
=#
module MTKDump

import ModelingToolkit
import ...OMBackend: @BACKEND_LOGGING, logPath

export dumpMTKPreSimplify, dumpMTKPostSimplify, dumpPreStructuralSimplifyExpr,
       dumpBuildDirectRHSInputs, dumpRHSExpression, dumpBatchBlock

"""
    dumpMTKPreSimplify(sys, pre_eqs, pre_unknowns)

Emit `backend/mtk/mtk_preSimplify.log`: the ODESystem `sys` before
`structural_simplify` — unknowns, equations, parameters, defaults.
"""
function dumpMTKPreSimplify(@nospecialize(sys), pre_eqs::Int, pre_unknowns::Int)
  @BACKEND_LOGGING begin
    try
      open(logPath("backend/mtk", "mtk_preSimplify.log"), "w") do io
        println(io, "############################################")
        println(io, "MTK system before structural_simplify")
        println(io, "############################################")
        println(io)
        println(io, "Unknowns ($pre_unknowns):")
        println(io, "---------------------------------------------")
        for (i, u) in enumerate(ModelingToolkit.unknowns(sys))
          println(io, "  [$i] $u")
        end
        println(io)
        println(io, "Equations ($pre_eqs):")
        println(io, "---------------------------------------------")
        for (i, eq) in enumerate(ModelingToolkit.equations(sys))
          println(io, "  [$i] $eq")
        end
        println(io)
        println(io, "Parameters ($(length(ModelingToolkit.parameters(sys)))):")
        println(io, "---------------------------------------------")
        for (i, p) in enumerate(ModelingToolkit.parameters(sys))
          println(io, "  [$i] $p")
        end
        try
          local defs = ModelingToolkit.defaults(sys)
          println(io)
          println(io, "Defaults ($(length(defs))):")
          println(io, "---------------------------------------------")
          for (k, v) in defs
            println(io, "  $k => $v")
          end
        catch
          println(io, "\n(defaults not available in this MTK version)")
        end
      end
    catch _err
      @warn "[mtk_preSimplify dump] failed" exception=_err
    end
  end
  return nothing
end

"""
    dumpMTKPostSimplify(sys)

Emit `backend/mtk/mtk_postSimplify.log`: the reduced ODESystem after
`structural_simplify` — unknowns, equations, observed, guesses.
"""
function dumpMTKPostSimplify(@nospecialize(sys))
  @BACKEND_LOGGING begin
    try
      open(logPath("backend/mtk", "mtk_postSimplify.log"), "w") do io
        local unks = ModelingToolkit.unknowns(sys)
        local eqs = ModelingToolkit.equations(sys)
        println(io, "############################################")
        println(io, "MTK system after structural_simplify")
        println(io, "############################################")
        println(io)
        println(io, "Unknowns ($(length(unks))):")
        println(io, "---------------------------------------------")
        for (i, u) in enumerate(unks); println(io, "  [$i] $u"); end
        println(io)
        println(io, "Equations ($(length(eqs))):")
        println(io, "---------------------------------------------")
        for (i, eq) in enumerate(eqs); println(io, "  [$i] $eq"); end
        try
          local obs = ModelingToolkit.observed(sys)
          println(io)
          println(io, "Observed ($(length(obs))):")
          println(io, "---------------------------------------------")
          for (i, eq) in enumerate(obs); println(io, "  [$i] $eq"); end
        catch
        end
        try
          local guesses = ModelingToolkit.guesses(sys)
          println(io)
          println(io, "Guesses ($(length(guesses))):")
          println(io, "---------------------------------------------")
          for (k, v) in guesses; println(io, "  $k => $v"); end
        catch
        end
      end
    catch _err
      @warn "[mtk_postSimplify dump] failed" exception=_err
    end
  end
  return nothing
end

"""
    dumpPreStructuralSimplifyExpr(logPath)

Build the runtime `try ... write(...) catch ... end` expression that dumps
the `firstOrderSystem` (equations + unknowns) just before
`structural_simplify` runs. `logPath` is captured at codegen time so the
dump lands in the per-model run directory even though the file is opened at
simulate time.

Used inside `performStructuralSimplify`'s quote; returns an `Expr` that is
inlined into the generated `simulate` body. Reading the equations/unknowns
inside the quote (instead of doing it from this Julia function) is essential
— the values are local to the generated `Model()` call.
"""
function dumpPreStructuralSimplifyExpr(logPathStr::AbstractString)
  return quote
    try
      local _buffer = IOBuffer()
      local _eqs = ModelingToolkit.equations(firstOrderSystem)
      local _unks = ModelingToolkit.unknowns(firstOrderSystem)
      println(_buffer, "Pre-structural-simplify dump")
      println(_buffer, "============================")
      println(_buffer, "equations: ", length(_eqs))
      println(_buffer, "unknowns:  ", length(_unks))
      println(_buffer, "")
      println(_buffer, "Equations:")
      println(_buffer, "----------")
      for (_i, _e) in enumerate(_eqs)
        println(_buffer, "[", _i, "] ", _e)
      end
      println(_buffer, "")
      println(_buffer, "Unknowns:")
      println(_buffer, "---------")
      for (_i, _u) in enumerate(_unks)
        println(_buffer, "[", _i, "] ", _u)
      end
      write($(logPathStr), String(take!(_buffer)))
    catch _err
      @warn "[preStructuralSimplify dump] failed" exception=_err
    end
  end
end

"""
    dumpBuildDirectRHSInputs(states, params, eqs, finalInitialValues, pars, reducedSystem, callbacks)

Emit `backend/mtk/buildDirectRHSInputs.log`: every Symbol/Num touching the
post-MTK boundary on entry to `buildDirectRHSProblem`. This is the codegen
surface where bare-base-name references to array variables would leak in if
any earlier pass forgot to subscript them.
"""
function dumpBuildDirectRHSInputs(@nospecialize(states), @nospecialize(params),
                                  @nospecialize(eqs), @nospecialize(finalInitialValues),
                                  @nospecialize(pars), @nospecialize(reducedSystem),
                                  @nospecialize(callbacks))
  @BACKEND_LOGGING begin
    try
      open(logPath("backend/mtk", "buildDirectRHSInputs.log"), "w") do io
        println(io, "############################################")
        println(io, "buildDirectRHSProblem inputs (post-structural_simplify)")
        println(io, "############################################")
        println(io)
        println(io, "States ($(length(states))):")
        println(io, "---------------------------------------------")
        for (i, s) in enumerate(states); println(io, "  [$i] $s"); end
        println(io)
        println(io, "Parameters ($(length(params))):")
        println(io, "---------------------------------------------")
        for (i, p) in enumerate(params); println(io, "  [$i] $p"); end
        println(io)
        println(io, "full_equations ($(length(eqs))):")
        println(io, "---------------------------------------------")
        for (i, e) in enumerate(eqs); println(io, "  [$i] $e"); end
        println(io)
        println(io, "finalInitialValues ($(length(finalInitialValues))):")
        println(io, "---------------------------------------------")
        for p in finalInitialValues; println(io, "  $(p.first)  =>  $(p.second)"); end
        println(io)
        println(io, "pars ($(length(pars))):")
        println(io, "---------------------------------------------")
        for (k, v) in pars; println(io, "  $k  =>  $v"); end
        println(io)
        println(io, "Observed equations:")
        println(io, "---------------------------------------------")
        local obs = try ModelingToolkit.observed(reducedSystem) catch; [] end
        for (i, e) in enumerate(obs); println(io, "  [$i] $e"); end
        println(io)
        println(io, "Guesses:")
        println(io, "---------------------------------------------")
        local guesses = try ModelingToolkit.guesses(reducedSystem) catch; Dict() end
        for (k, v) in guesses; println(io, "  $k  =>  $v"); end
        println(io)
        println(io, "Callbacks (raw show):")
        println(io, "---------------------------------------------")
        show(io, MIME"text/plain"(), callbacks)
        println(io)
      end
    catch _err
      @warn "[buildDirectRHSInputs dump] failed" exception=_err
    end
  end
  return nothing
end

"""
    dumpRHSExpression(rhs_list, f_ip_expr)

Emit `backend/mtk/rhsExpression.log`: the symbolic RHS list plus the actual
`Symbolics.build_function` body that becomes the integrator's RGF. Bare-base-
name references to subscripted state arrays would appear here if Symbolics'
`build_function` ever stripped the `[i]_field` part of a `var"..."` symbol.
"""
function dumpRHSExpression(@nospecialize(rhs_list), @nospecialize(f_ip_expr))
  @BACKEND_LOGGING begin
    try
      open(logPath("backend/mtk", "rhsExpression.log"), "w") do io
        println(io, "############################################")
        println(io, "Generated RHS expression (input to RuntimeGeneratedFunction)")
        println(io, "############################################")
        println(io)
        println(io, "rhs_list ($(length(rhs_list))):")
        println(io, "---------------------------------------------")
        for (i, r) in enumerate(rhs_list); println(io, "  [$i] $r"); end
        println(io)
        println(io, "build_function expression (in-place form):")
        println(io, "---------------------------------------------")
        println(io, f_ip_expr)
      end
    catch _err
      @warn "[rhsExpression dump] failed" exception=_err
    end
  end
  return nothing
end

"""
    dumpBatchBlock(vars, irreductableSyms, statePriorityPairs, batchBlock)

Emit `backend/mtk/batchBlock.log`: the resolved (`Symbol => Symbolics.Num`)
pairs and the raw `Expr` that is `eval`'d at simulate time to create the
model module's Symbolics-variable bindings. A bare base-name `\$sym = ...`
binding here (vs the expected `var"x[i]_y" = ...`) would point directly at
the construction site of any UndefVarError on an array-of-records state.
"""
function dumpBatchBlock(@nospecialize(vars), @nospecialize(irreductableSyms),
                        @nospecialize(statePriorityPairs), @nospecialize(batchBlock))
  @BACKEND_LOGGING begin
    try
      open(logPath("backend/mtk", "batchBlock.log"), "w") do io
        println(io, "############################################")
        println(io, "Resolved Symbolics-variable batch block (passed to eval)")
        println(io, "############################################")
        println(io)
        println(io, "vars ($(length(vars))):")
        println(io, "---------------------------------------------")
        for (_sym, _var) in vars
          println(io, "  ", _sym, "  =>  ", _var)
        end
        println(io)
        println(io, "irreductableSyms ($(length(irreductableSyms))):")
        println(io, "---------------------------------------------")
        for _s in irreductableSyms
          println(io, "  ", _s)
        end
        println(io)
        println(io, "statePriorityPairs ($(length(statePriorityPairs))):")
        println(io, "---------------------------------------------")
        for (_s, _p) in statePriorityPairs
          println(io, "  ", _s, " => ", _p)
        end
        println(io)
        println(io, "batchBlock raw Expr:")
        println(io, "---------------------------------------------")
        for (_i, _a) in enumerate(batchBlock.args)
          println(io, "  [$_i] ", _a)
        end
      end
    catch _err
      @warn "[batchBlock dump] failed" exception=_err
    end
  end
  return nothing
end

end # module MTKDump
