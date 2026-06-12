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

  These tests exercise the backend pipeline using hand-crafted DAE IR
  constructs. The ExampleDAE module provides the IR as a stable contract
  for the DAE.DAE_LIST input format.

  For full integration tests (frontend + backend + simulation), see
  the top-level OM.jl test suite at /test/runtests.jl.
=#

import Absyn
import SCode
import DAE
using MetaModelica
import OMParser
import OMFrontend
using OMBackend
using Test
using Dates
using DifferentialEquations: ReturnCode

include("./ExampleDAE/ExampleDAEs.jl")
import .ExampleDAEs

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
        include("test/ExampleDAE/ExampleDAEs.jl")
        using .ExampleDAEs
        println("same_module=", OMBackend === OMBackend.CodeGeneration.OMBackend)
        OMBackend.translate(ExampleDAEs.helloWorld_DAE; BackendMode = OMBackend.MTK_MODE)
        sleep(1)
        OMBackend.translate(ExampleDAEs.helloWorld_DAE; BackendMode = OMBackend.MTK_MODE)
        local runDirs = sort(filter(name -> startswith(name, "HelloWorld_"), readdir(ENV["OMJL_LOG_DIR"])))
        println("run_count=", length(runDirs))
        println("root_codegen=", isfile(joinpath(ENV["OMJL_LOG_DIR"], "backend", "codeGen", "equationFirstStageCodeGen.log")))
        println("run_codegen=", all(dir -> isfile(joinpath(ENV["OMJL_LOG_DIR"], dir, "backend", "codeGen", "equationFirstStageCodeGen.log")), runDirs))
        """
        local output = read(`$(Base.julia_cmd()) --startup-file=no --project=$(repoRoot) -e $script`, String)
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

  #= ── 2. Lowering (DAE IR -> BDAE IR) ──────────────────────────── =#

  @testset "Lowering" begin
    @testset "HelloWorld" begin
      bDAE = OMBackend.lower(ExampleDAEs.helloWorld_DAE)
      eqSys = first(bDAE.eqs)
      @test length(eqSys.orderedEqs) >= 1
      @test length(eqSys.orderedVars) >= 1
    end

    @testset "LotkaVolterra" begin
      bDAE = OMBackend.lower(ExampleDAEs.lotkaVolterra_DAE)
      eqSys = first(bDAE.eqs)
      @test length(eqSys.orderedEqs) >= 2
    end

    @testset "VanDerPol" begin
      bDAE = OMBackend.lower(ExampleDAEs.vanDerPol_DAE)
      eqSys = first(bDAE.eqs)
      @test length(eqSys.orderedEqs) >= 2
    end

    @testset "SimpleMech" begin
      bDAE = OMBackend.lower(ExampleDAEs.simpleMech_DAE)
      eqSys = first(bDAE.eqs)
      @test length(eqSys.orderedEqs) >= 4
    end

    @testset "SimpleCircuit" begin
      bDAE = OMBackend.lower(ExampleDAEs.simpleCircuit_DAE)
      eqSys = first(bDAE.eqs)
      @test length(eqSys.orderedEqs) >= 6
    end
  end

  #= ── 3. Translation (DAE IR -> MTK code) ──────────────────────── =#

  @testset "Translation" begin
    OMBackend.clearCaches!()

    @testset "HelloWorld translates" begin
      (name, code) = OMBackend.translate(ExampleDAEs.helloWorld_DAE;
                                          BackendMode=OMBackend.MTK_MODE)
      @test name == "HelloWorld"
      @test code isa Expr
    end

    @testset "LotkaVolterra translates" begin
      (name, code) = OMBackend.translate(ExampleDAEs.lotkaVolterra_DAE;
                                          BackendMode=OMBackend.MTK_MODE)
      @test name == "LotkaVolterra"
      @test code isa Expr
    end

    @testset "VanDerPol translates" begin
      (name, code) = OMBackend.translate(ExampleDAEs.vanDerPol_DAE;
                                          BackendMode=OMBackend.MTK_MODE)
      @test name == "VanDerPol"
      @test code isa Expr
    end
  end

  #= ── 4. SimCode partition: when initial() → INITIAL_ALGORITHM ── =#

  @testset "extractInitialWhenAlgorithms" begin
    local SC = OMBackend.SimulationCode
    local BDAE = OMBackend.Backend.BDAE
    local callAttrBool = DAE.CALL_ATTR(DAE.T_BOOL(nil), false, true, false, false,
                                       DAE.NO_INLINE(), DAE.NO_TAIL())
    local initialCall = DAE.CALL(Absyn.IDENT("initial"), nil, callAttrBool)
    local nonInitialCref = DAE.CREF(DAE.CREF_IDENT("trigger", DAE.T_BOOL(nil), nil),
                                    DAE.T_BOOL(nil))
    local mkNoretBody = () -> list(BDAE.NORETCALL(DAE.RCONST(0.0), DAE.emptyElementSource))
    local mkWhenEq = cond -> BDAE.WHEN_EQUATION(
      0,
      BDAE.WHEN_STMTS(cond, mkNoretBody(), NONE()),
      DAE.emptyElementSource,
      nothing,
    )

    @testset "bare initial() condition is extracted" begin
      local input = BDAE.WHEN_EQUATION[mkWhenEq(initialCall)]
      local (kept, inits) = SC.extractInitialWhenAlgorithms(input)
      @test isempty(kept)
      @test length(inits) == 1
      @test inits[1] isa SC.INITIAL_ALGORITHM
      @test length(inits[1].statements) == 1
      @test inits[1].statements[1] isa SC.NORETCALL
    end

    @testset "non-initial when is kept" begin
      local input = BDAE.WHEN_EQUATION[mkWhenEq(nonInitialCref)]
      local (kept, inits) = SC.extractInitialWhenAlgorithms(input)
      @test length(kept) == 1
      @test isempty(inits)
    end

    @testset "array {initial(), other} is extracted" begin
      local arrCond = DAE.ARRAY(DAE.T_BOOL(nil), false,
                                list(initialCall, nonInitialCref))
      local input = BDAE.WHEN_EQUATION[mkWhenEq(arrCond)]
      local (kept, inits) = SC.extractInitialWhenAlgorithms(input)
      @test isempty(kept)
      @test length(inits) == 1
    end

    @testset "mixed input partitions correctly" begin
      local input = BDAE.WHEN_EQUATION[
        mkWhenEq(initialCall),
        mkWhenEq(nonInitialCref),
        mkWhenEq(initialCall),
      ]
      local (kept, inits) = SC.extractInitialWhenAlgorithms(input)
      @test length(kept) == 1
      @test length(inits) == 2
    end

    @testset "empty input gives empty outputs" begin
      local (kept, inits) = SC.extractInitialWhenAlgorithms(BDAE.WHEN_EQUATION[])
      @test isempty(kept)
      @test isempty(inits)
    end
  end

  #= ── 5. Simulation with result verification ───────────────────── =#

  @testset "Simulation" begin
    OMBackend.clearCaches!()

    @testset "HelloWorld: der(x) = -a*x" begin
      OMBackend.translate(ExampleDAEs.helloWorld_DAE;
                          BackendMode=OMBackend.MTK_MODE)
      sol = OMBackend.simulateModel("HelloWorld";
                                     MODE=OMBackend.MTK_MODE,
                                     tspan=(0.0, 1.0))
      @test sol.retcode == ReturnCode.Success
      # Analytical: x(t) = exp(-t), so x(1) = exp(-1)
      @test isapprox(last(sol.u)[1], exp(-1.0); atol=1e-4)
    end

    @testset "LotkaVolterra simulates" begin
      OMBackend.translate(ExampleDAEs.lotkaVolterra_DAE;
                          BackendMode=OMBackend.MTK_MODE)
      sol = OMBackend.simulateModel("LotkaVolterra";
                                     MODE=OMBackend.MTK_MODE,
                                     tspan=(0.0, 1.0))
      @test sol.retcode == ReturnCode.Success
    end

    @testset "VanDerPol simulates" begin
      OMBackend.translate(ExampleDAEs.vanDerPol_DAE;
                          BackendMode=OMBackend.MTK_MODE)
      sol = OMBackend.simulateModel("VanDerPol";
                                     MODE=OMBackend.MTK_MODE,
                                     tspan=(0.0, 1.0))
      @test sol.retcode == ReturnCode.Success
    end
  end

  #= ── 6. SimCode Exp traversal + alias substitution ────────────── =#
  include("simCodeTraverseTests.jl")

  #= ── 7. Discrete-dummy demotion planning ──────────────────────── =#
  include("discreteDummyDemotionTests.jl")

end
