#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
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
using DifferentialEquations: ReturnCode

include("./ExampleDAE/ExampleDAEs.jl")
import .ExampleDAEs

@testset "OMBackend" begin

  #= ── 1. Cache management ──────────────────────────────────────── =#

  @testset "clearCaches!" begin
    @testset "clears all caches" begin
      #= Populate each cache with dummy entries =#
      OMBackend.COMPILED_MODELS_MTK["TestModel"] = (Expr(:block, :nothing), false, UInt64(0))
      OMBackend.CodeGeneration.MODELICA_FUNCTION_IMPLS[:testFunc] = identity
      OMBackend.CodeGeneration.MODELICA_FUNCTION_WRAPPERS[:testFunc] = identity
      OMBackend.CodeGeneration.ELEM_FUNC_CACHE[(:testFunc, (1,))] = identity
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
      OMBackend.CodeGeneration.ELEM_FUNC_CACHE[(:testFunc, (1,))] = identity

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

  #= ── 4. Simulation with result verification ───────────────────── =#

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

end
