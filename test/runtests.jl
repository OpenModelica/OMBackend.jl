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
*/ =#

#= Author John Tinnerholm & Andreas Heureman =#

#=
  OMBackend unit tests.

  These tests exercise the backend pipeline on the example models of
  examples/Examples.mo, flattened by OMFrontend (ExampleModels), and on
  hand-built OMJLIR and SimCode values.

  For full integration tests (frontend + backend + simulation), see
  the top-level OM.jl test suite at /test/runtests.jl.
=#

# On Windows, a failed DLL load or a segfault in external Modelica C functions
# opens a modal Windows Error Reporting dialog. On a headless CI runner nothing
# clicks it away, so the test process hangs until the job timeout (a silent
# ~90 min stall). Disable those hard-error popups so such a failure crashes/errors
# visibly in the log instead of blocking. (SEM_FAILCRITICALERRORS |
# SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX = 0x8003.)
@static if Sys.iswindows()
    ccall((:SetErrorMode, "kernel32.dll"), UInt32, (UInt32,), 0x8003)
end

# Load Plots (and through it Glib's gettext runtime) FIRST in the test process,
# before OMBackend — which pulls in OMRuntimeExternalC and its bundled
# libintl-8.dll. On Windows, whichever libintl-8.dll loads first wins the name;
# if OMRuntimeExternalC's loads first, Glib's libgio-2.0-0.dll later fails with
# "the specified procedure could not be found". Preloading Plots here forces Glib's
# gettext to win regardless of the package __init__ order, which (unlike the
# in-module import in OMBackend) is not guaranteed when loading the precompiled
# package on CI.
import Plots

import OMFrontend
using OMBackend
using Test
using Dates
using DifferentialEquations: ReturnCode

include("../examples/ExampleModels.jl")
import .ExampleModels

@testset "OMBackend" begin

  function resetLogRunDirStack!()
    while OMBackend.hasActiveLogRunDir()
      OMBackend.popLogRunDir()
    end
  end

  #= ── 0. Logging helpers ──────────────────────────────────────── =#

  @testset "Log Scoping" begin
    local originalLogDir = get(ENV, "OMJL_LOG_DIR", nothing)
    resetLogRunDirStack!()
    try
      delete!(ENV, "OMJL_LOG_DIR")

      @test OMBackend.sanitizeLogRunName(" Modelica.Mechanics/Pendulum? ") == "Modelica_Mechanics_Pendulum"
      @test OMBackend.createLogRunId("Pendulum";
                                     timestamp = DateTime(2026, 4, 21, 14, 33, 8)) == "Pendulum_2026-04-21_14-33-08"

      local sessionRoot = joinpath(tempdir(), "OMJL", OMBackend.OMJL_SESSION_ID)
      @test OMBackend.logDir() == sessionRoot

      local firstRun = OMBackend.pushLogRunDir("Parent.Run")
      @test firstRun == "Parent_Run"
      @test OMBackend.logDir() == joinpath(sessionRoot, "Parent_Run")

      local nestedRun = OMBackend.pushLogRunDir("child/run")
      @test nestedRun == joinpath("Parent_Run", "child_run")
      @test OMBackend.logDir() == joinpath(sessionRoot, "Parent_Run", "child_run")

      @test OMBackend.popLogRunDir() == joinpath("Parent_Run", "child_run")
      @test OMBackend.logDir() == joinpath(sessionRoot, "Parent_Run")
      @test OMBackend.popLogRunDir() == "Parent_Run"
      @test OMBackend.logDir() == sessionRoot

      mktempdir() do tmp
        ENV["OMJL_LOG_DIR"] = tmp
        local runId = OMBackend.createLogRunId("Scoped.Model";
                                               timestamp = DateTime(2026, 4, 21, 16, 2, 3))
        local path = OMBackend.withLogRunDir(runId) do
          OMBackend.logPath("backend/bdae", "x.log")
        end
        @test path == joinpath(tmp, runId, "backend/bdae", "x.log")
        @test isdir(dirname(path))

        OMBackend.pushLogRunDir(runId)
        local nestedPath = OMBackend.withLogRunDir("recompile pass") do
          OMBackend.logPath("backend/runtime", "y.log")
        end
        @test nestedPath == joinpath(tmp, runId, "recompile_pass", "backend/runtime", "y.log")
        @test OMBackend.popLogRunDir() == runId
      end

      mktempdir() do tmp
        local repoRoot = abspath(joinpath(@__DIR__, ".."))
        local script = """
        ENV["ENABLE_BACKEND_LOGGING"] = "true"
        ENV["OMJL_LOG_DIR"] = $(repr(tmp))
        cd($(repr(repoRoot)))
        include("src/OMBackend.jl")
        include("examples/ExampleModels.jl")
        using .ExampleModels
        println("same_module=", OMBackend === OMBackend.CodeGeneration.OMBackend)
        (fm, functions) = ExampleModels.flatModel("HelloWorld")
        OMBackend.translate(fm; functionList = functions, BackendMode = OMBackend.MTK_MODE)
        sleep(1)
        OMBackend.translate(fm; functionList = functions, BackendMode = OMBackend.MTK_MODE)
        runDirs = sort(filter(name -> startswith(name, "Examples_HelloWorld_"), readdir(ENV["OMJL_LOG_DIR"])))
        println("run_count=", length(runDirs))
        println("root_codegen=", isfile(joinpath(ENV["OMJL_LOG_DIR"], "backend", "codeGen", "equationFirstStageCodeGen.log")))
        println("run_codegen=", all(dir -> isfile(joinpath(ENV["OMJL_LOG_DIR"], dir, "backend", "codeGen", "equationFirstStageCodeGen.log")), runDirs))
        """
        #= The test environment, which has OMBackend's dependencies: the package's own
           project is not instantiated where its siblings are developed into a
           temporary environment (OM.jl's ci/test-package.jl, since [sources] is gone). =#
        local output = read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`, String)
        @test occursin("same_module=true", output)
        @test occursin("run_count=2", output)
        @test occursin("root_codegen=false", output)
        @test occursin("run_codegen=true", output)
      end
    finally
      resetLogRunDirStack!()
      if originalLogDir === nothing
        delete!(ENV, "OMJL_LOG_DIR")
      else
        ENV["OMJL_LOG_DIR"] = originalLogDir
      end
    end
  end

  #= ── 1. Cache management ──────────────────────────────────────── =#

  @testset "clearCaches!" begin
    @testset "clears all caches" begin
      #= Populate each cache with dummy entries =#
      OMBackend.COMPILED_MODELS_MTK["TestModel"] = (Expr(:block, :nothing), false, UInt64(0))
      OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[:testFunc] = identity
      OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS[:testFunc] = identity
      OMBackend.CodeGeneration.ELEM_FUNC_CACHE[(:testFunc, (1,), 1)] = identity
      @test !isempty(OMBackend.COMPILED_MODELS_MTK)
      @test !isempty(OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS)
      @test !isempty(OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS)
      @test !isempty(OMBackend.CodeGeneration.ELEM_FUNC_CACHE)

      cleared = OMBackend.clearCaches!()
      @test Set(cleared) == Set(["models", "implementations", "wrappers", "extractors"])
      @test isempty(OMBackend.COMPILED_MODELS_MTK)
      @test isempty(OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS)
      @test isempty(OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS)
      @test isempty(OMBackend.CodeGeneration.ELEM_FUNC_CACHE)
    end

    @testset "selective clearing" begin
      #= Populate all caches =#
      OMBackend.COMPILED_MODELS_MTK["TestModel"] = (Expr(:block, :nothing), false, UInt64(0))
      OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[:testFunc] = identity
      OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS[:testFunc] = identity
      OMBackend.CodeGeneration.ELEM_FUNC_CACHE[(:testFunc, (1,), 1)] = identity

      #= Clear only models =#
      cleared = OMBackend.clearCaches!(models=true, implementations=false,
                                       wrappers=false, extractors=false)
      @test cleared == ["models"]
      @test isempty(OMBackend.COMPILED_MODELS_MTK)
      #= Others should still have entries =#
      @test !isempty(OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS)
      @test !isempty(OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS)
      @test !isempty(OMBackend.CodeGeneration.ELEM_FUNC_CACHE)

      #= Clean up =#
      OMBackend.clearCaches!()
    end
  end

  #= ── 2. Lowering (flat model -> OMJLIR) ─────────────────────────── =#

  #= The OMJLIR model of an example, through the passes; the flat model and functions. =#
  local lowered = name -> begin
    local (fm, functions) = ExampleModels.flatModel(name)
    OMBackend.omjlirModel(fm; functionList = functions)
  end

  @testset "Lowering" begin
    for (name, nEquations) in (("HelloWorld", 1), ("LotkaVolterra", 2), ("VanDerPol", 2),
                               ("SimpleMechanicalSystem", 4), ("SimpleCircuit", 6))
      @testset "$name" begin
        local model = lowered(name)
        @test length(only(model.continuous).equations) >= nEquations
        @test length(model.variables) >= 1
      end
    end
  end

  #= ── 3. Translation (flat model -> MTK code) ─────────────────────── =#

  @testset "Translation" begin
    OMBackend.clearCaches!()
    for name in ("HelloWorld", "LotkaVolterra", "VanDerPol")
      @testset "$name translates" begin
        local (fm, functions) = ExampleModels.flatModel(name)
        local (modelName, code) = OMBackend.translate(fm; functionList = functions, BackendMode = OMBackend.MTK_MODE)
        @test modelName == "Examples_" * name
        @test code isa Expr
      end
    end
  end

  #= ── 4. SimCode partition: when initial() → INITIAL_ALGORITHM ── =#

  @testset "extractInitialWhenAlgorithms" begin
    local IR = OMBackend.OMJLIR
    local SC = OMBackend.SimCode
    local trigger = IR.CREF(IR.ComponentRef(:trigger, IR.T_BOOLEAN()), IR.T_BOOLEAN())
    local body = () -> IR.Statement[IR.NORETCALL(IR.CALL(:print, IR.Expression[IR.STRING("x")], IR.T_NORETCALL(), IR.BUILTIN_CALL),
                                                 IR.NO_SOURCE)]
    #= a when-equation of one branch: its triggers, and whether initial() is among them =#
    local mkWhen = (triggers, atInitial) -> IR.WHEN_EQUATION([IR.WhenBranch(triggers, atInitial, false, body())],
                                                             IR.REGULAR_WHEN, IR.NO_SOURCE, IR.DEFAULT_ATTRIBUTES)

    @testset "bare initial() condition is extracted" begin
      local (kept, inits) = SC.extractInitialWhenAlgorithms(IR.WHEN_EQUATION[mkWhen(IR.Expression[], true)])
      @test isempty(kept)
      @test length(inits) == 1
      @test length(inits[1]) == 1
      @test inits[1][1] isa IR.NORETCALL
    end

    @testset "non-initial when is kept" begin
      local (kept, inits) = SC.extractInitialWhenAlgorithms(IR.WHEN_EQUATION[mkWhen(IR.Expression[trigger], false)])
      @test length(kept) == 1
      @test isempty(inits)
    end

    @testset "array {initial(), other}: an initial algorithm and a when on the other" begin
      #= The when fires at the initialization and whenever `other` becomes true
         (MLS 8.3.5; the MSL ZeroOrderHold's `when {sampleTrigger, initial()}`). =#
      local (kept, inits) = SC.extractInitialWhenAlgorithms(IR.WHEN_EQUATION[mkWhen(IR.Expression[trigger], true)])
      @test length(inits) == 1
      @test length(kept) == 1
      @test kept[1].branches[1].triggers == IR.Expression[trigger]
      @test kept[1].alsoInitial
    end

    @testset "mixed input partitions correctly" begin
      local input = IR.WHEN_EQUATION[mkWhen(IR.Expression[], true), mkWhen(IR.Expression[trigger], false),
                                     mkWhen(IR.Expression[], true)]
      local (kept, inits) = SC.extractInitialWhenAlgorithms(input)
      @test length(kept) == 1
      @test length(inits) == 2
    end

    @testset "empty input gives empty outputs" begin
      local (kept, inits) = SC.extractInitialWhenAlgorithms(IR.WHEN_EQUATION[])
      @test isempty(kept)
      @test isempty(inits)
    end
  end

  #= ── 5. Simulation with result verification ───────────────────── =#

  @testset "Simulation" begin
    OMBackend.clearCaches!()
    local simulated = name -> begin
      local (fm, functions) = ExampleModels.flatModel(name)
      OMBackend.translate(fm; functionList = functions, BackendMode = OMBackend.MTK_MODE)
      OMBackend.simulateModel("Examples_" * name; MODE = OMBackend.MTK_MODE, tspan = (0.0, 1.0))
    end

    @testset "HelloWorld: der(x) = -a*x" begin
      local sol = simulated("HelloWorld")
      @test sol.retcode == ReturnCode.Success
      # Analytical: x(t) = exp(-t), so x(1) = exp(-1)
      @test isapprox(last(sol.u)[1], exp(-1.0); atol=1e-4)
    end

    @testset "LotkaVolterra simulates" begin
      @test simulated("LotkaVolterra").retcode == ReturnCode.Success
    end

    @testset "VanDerPol simulates" begin
      @test simulated("VanDerPol").retcode == ReturnCode.Success
    end
  end

  #= ── 6. SimCode Exp traversal + alias substitution ────────────── =#
  include("simCodeTraverseTests.jl")

  #= ── 7. Discrete-dummy demotion planning ──────────────────────── =#
  include("discreteDummyDemotionTests.jl")

  #= ── 8. One event callback per relation zero set ─────────────── =#
  include("relationZeroSetTests.jl")

  #= ── 9. Event iteration within one event ─────────────────────── =#
  include("eventIterationTests.jl")

  #= ── 10. Step control for algebraic unknowns ─────────────────── =#
  include("algebraicStepControlTests.jl")

  #= ── 11. The error policy of fallbacks ───────────────────────── =#
  include("errorPolicyTests.jl")

  #= ── 12. The Modelica builtins of generated code ─────────────── =#
  include("modelicaBuiltinsTests.jl")

  #= ── 13. The names Modelica function statements read ──────────── =#
  include("functionBodyCrefsTests.jl")

  #= ── 14. Protected variables keep their attributes ────────────── =#
  include("protectedAttributesTests.jl")

  #= ── 15. Calls of generated functions have the function's arity ─ =#
  include("functionArityTests.jl")

  #= ── 16. Generated functions hash the same in every process ───── =#
  include("deterministicHashTests.jl")

  #= ── 17. Merged continuous callbacks keep their re-initialization =#
  include("eventCallbackMergeTests.jl")

  #= ── 18. Solve keyword arguments with an abstol per unknown ────── =#
  include("solveKwargsTests.jl")

  #= ── 19. Array-preserving code generation (no-scalarize models) ─ =#
  include("arrayPathTests.jl")

  #= ── 20. The initialization's one-unknown rows ──────────────────── =#
  include("initializationCausalRowsTests.jl")

  #= ── 20b. OMBackend's index reduction step by step ──────────────── =#
  include("indexReductionTests.jl")

  #= ── 20c. Zero flow and zero speed (semiLinear chains, noEvent) ─── =#
  include("zeroFlowTests.jl")

  #= ── 20d. Parameters an initial algorithm assigns ───────────────── =#
  include("initialAlgorithmParameterTests.jl")

  #= ── 20e. Impure calls on the array path ───────────────────────── =#
  include("impureCallTests.jl")

  #= ── 20f. A function's arrays of records bound by calls ────────── =#
  include("recordArrayLocalTests.jl")

  #= ── 20g. The model's functions compiled ─────────────────────────── =#
  include("compiledFunctionTests.jl")

  #= ── 20h. Extractor calls hoisted: one call per function and arguments ── =#
  include("extractorHoistTests.jl")

  #= ── 20i. Underdetermined initialization: states at their start values ── =#
  include("initFixStatesTests.jl")

  #= ── 20j. A relation that located its event takes the change ── =#
  include("relationLocatedTests.jl")

  #= ── 20k. Partials of a function's call terms (extractors, flat calls) ── =#
  include("callPartialTests.jl")

  #= ── 20l. Observed equations without a cycle ── =#
  include("observedCycleTests.jl")

  #= ── 20m. Parameters as a model's result; external String outputs ── =#
  include("parameterResultTests.jl")

  #= ── 20n. An if-expression's branch not taken is not evaluated ── =#
  include("lazyBranchTests.jl")

  #= ── 20o. Known values of a function's tuple output in index reduction ── =#
  include("tupleKnownValueTests.jl")

  #= ── 20p. Clocked discretized continuous-time partitions (solver clocks) ── =#
  include("solverClockTests.jl")

  #= ── 20q. sample() in a discrete cluster; an initial section after the initialization ── =#
  include("sampleClusterTests.jl")

  #= ── 20r. Aliases of held discretes: the held one the relay's representative ── =#
  include("discreteAliasTests.jl")

  #= ── 20s. Hidden linear constraints among the states; a table lookup's derivative ── =#
  include("hiddenConstraintTests.jl")

  #= ── 20t. The band's edge as crossed for if-equation relations that read time ── =#
  include("bandEdgeRuleTests.jl")

  #= ── 20u. A cluster relation exactly on its boundary; the step after an event ── =#
  include("exactZeroRelationTests.jl")

  #= ── 20v. The model cache within its memory limit ── =#
  include("cacheTests.jl")

  #= ── 21. The backend reads the frontend's flat model only ──────── =#
  @testset "No DAE in the backend" begin
    #= The frontend's DAE, the previous backend's BDAE and its SimulationCode are gone: the
       backend lowers the flat model to OMJLIR. No source reads them (prose may name them). =#
    local src = normpath(joinpath(@__DIR__, "..", "src"))
    local reads = r"\bDAE\.[A-Za-z]|(import|using)\s+(\.\.)?(DAE|BDAE|SimulationCode|FrontendUtil)\b|\b(BDAE|SimulationCode|FrontendUtil)\.[A-Za-z]"
    local found = String[]
    for (root, _, files) in walkdir(src), f in files
      endswith(f, ".jl") || continue
      for (i, line) in enumerate(eachline(joinpath(root, f)))
        occursin(reads, line) && push!(found, string(relpath(joinpath(root, f), src), ":", i, ": ", strip(line)))
      end
    end
    @test isempty(found)
    isempty(found) || foreach(println, found)
  end

end
