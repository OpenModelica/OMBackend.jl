#= exportModel: a translated model's Julia source, loaded from its file and simulated without the translate (Author: John
   Tinnerholm). =#
using Test
import OMBackend
import OMFrontend
using DifferentialEquations: ReturnCode

#= x' = 1 until x = 0.5 (t = 0.5), then 0.25: x(1) = 0.625; one event, counted by n. =#
const EXPORT_PROGRAM = OMFrontend.OpenModelicaParser.Modelica"""
  model ExportModel
    Real x(start = 0, fixed = true);
    discrete Integer n(start = 0, fixed = true);
  equation
    der(x) = if x < 0.5 then 1 else 0.25;
    when x > 0.5 then
      n = pre(n) + 1;
    end when;
  end ExportModel;"""

#= Two structural modes (OMBackend's variable structure systems): x' = -x until 0.5, then x' = {1, 2}. =#
const EXPORT_VSS_PROGRAM = OMFrontend.OpenModelicaParser.Modelica"""
  model ExportVSS
    model A
      Real x[2](start = {1.0, 2.0});
    equation
      for i in 1:2 loop
        der(x[i]) = -x[i];
      end for;
    end A;
    model B
      Real x[2](start = {1.0, 2.0});
    equation
      der(x) = {1.0, 2.0};
    end B;
    structuralmode A a;
    structuralmode B b;
  equation
    initialStructuralState(a);
    structuralTransition(a, b, time >= 0.5);
  end ExportVSS;"""

"""
The models of Models/ArrayPath.mo exported on the arrays-kept route: one or two features each (what OMBackend
generates for them), together most of what the array code holds.
"""
const EXPORT_ARRAY_PATH_MODELS = Tuple{String, String}[
  ("BouncingBalls", "an array of components, when with reinit"),
  ("Rectifier", "if-equations switching at events"),
  ("AlgebraicLoop", "a nonlinear algebraic loop"),
  ("CoupledLoopEvents", "algebraic loops coupled through events"),
  ("SampleZOH", "sample(), a zero-order hold"),
  ("Functions", "Modelica functions"),
  ("EventFns", "event functions (div, mod, integer)"),
  ("RecordTupleWhen", "records, a tuple call in a when"),
  ("ReinitElements", "reinit of array elements"),
  ("EnumTable", "enumerations"),
  ("TerminateInBranch", "terminate()"),
  ("TableDrive", "external C functions (CombiTimeTable, OMRuntimeExternalC)"),
  ("ClockedSampleHold", "clocked partitions"),
  ("ComplexArrays", "operator records (Complex)"),
  ("SlidingMass", "index reduction"),
  ("StickSlip", "friction, a mixed system"),
  ("InitSteady", "steady-state initialization"),
  ("RolledBodies", "rolled component arrays"),
  ("StringParameterMessage", "string parameters"),
  ("IdealDiodeLoop", "ideal switches in a loop")]

const EXPORT_ARRAY_PATH_SCODE = OMFrontend.translateToSCode(OMFrontend.parseFile(joinpath(@__DIR__, "Models", "ArrayPath.mo")))

"The roots of the `using` and `import` paths at the top level of the module `m`."
function importRoots(m::Expr)::Vector{Symbol}
  local roots::Vector{Symbol} = Symbol[]
  for s in m.args[3].args
    (s isa Expr && (s.head === :using || s.head === :import)) || continue
    for p in s.args
      local path::Expr = p.head === :(:) || p.head === :as ? p.args[1] : p
      push!(roots, path.args[1])
    end
  end
  return roots
end

"The model `model` of the SCode `scode` translated on the route `scalarize`; its name."
function exportTranslate(model::String, scode; scalarize::Bool)::String
  local (fm, cache) = OMFrontend.instantiateSCodeToFM(model, scode; scalarize = scalarize)
  OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache), scalarize = scalarize)
  return OMBackend.canonicalName(model)
end

"""
    exportRoundTrip(name::String; stop::Float64 = 1.0) -> (expected, sol, m)

The translated model `name` simulated on (0, `stop`) (`expected`), exported to a file, forgotten, and the file loaded
into a module of its own and its `simulate` run (`sol`); `m` the file's module expression. What the file defines is in
a later world: read through `invokelatest`.
"""
function exportRoundTrip(name::String; stop::Float64 = 1.0)::Tuple{Any, Any, Expr}
  local path::String = joinpath(mktempdir(), "Exported.jl")
  local expected::Any = nothing
  try
    expected = OMBackend.simulate(name; tspan = (0.0, stop), reltol = 1e-8, abstol = 1e-8)
    @test OMBackend.exportModel(path, name) == path
  finally
    OMBackend.forgetModel(name)
  end
  local m::Expr = only(filter(e -> e isa Expr && e.head === :module, Meta.parseall(read(path, String)).args))
  local sandbox::Module = Module(:ExportSandbox)
  Base.include(sandbox, path)
  local sol::Any = Base.invokelatest() do
    getglobal(getglobal(sandbox, m.args[2]::Symbol), :simulate)((0.0, stop); reltol = 1e-8, abstol = 1e-8)
  end
  return (expected, sol, m)
end

"The file's packages through OMBackend (or relative, `using .ColdCode`): it needs only OMBackend in the environment."
throughOMBackend(m::Expr)::Bool = all(r -> r === :OMBackend || r === :LinearAlgebra || r === :., importRoots(m))

@testset "exportModel: the Julia source loads from its file and simulates ($route)" for (route::String, scalarize::Bool) in
    (("array code", false), ("ModelingToolkit", true))
  local name::String = exportTranslate("ExportModel", OMFrontend.translateToSCode(EXPORT_PROGRAM); scalarize = scalarize)
  @test (name in OMBackend.ARRAY_PRESERVE_DE_MODELS) == !scalarize
  #= the line comments only when asked for =#
  @test !occursin(r"#= .*\.jl:\d+ =#", sprint(OMBackend.exportModel, name))
  @test occursin(r"#= .*\.jl:\d+ =#", sprint(OMBackend.exportModel, name, OMBackend.JuliaSource(comments = true)))
  local (expected, sol, m) = exportRoundTrip(name)
  @test OMBackend.variableValues(expected, "x")[end] ≈ 0.625 atol = 1e-6
  @test throughOMBackend(m)
  @test sol.retcode == ReturnCode.Success
  @test sol.u[end] ≈ expected.u[end] rtol = 1e-8
end

@testset "exportModel: models of the array path ($feature)" for (model::String, feature::String) in EXPORT_ARRAY_PATH_MODELS
  local name::String = exportTranslate("ArrayPath." * model, EXPORT_ARRAY_PATH_SCODE; scalarize = false)
  @test name in OMBackend.ARRAY_PRESERVE_DE_MODELS
  local (expected, sol, m) = exportRoundTrip(name)
  @test throughOMBackend(m)
  @test sol.retcode == expected.retcode
  @test sol.t[end] == expected.t[end]
  @test sol.u[end] ≈ expected.u[end] rtol = 1e-8
end

#= on the arrays-kept route the backend scalarizes it (no array code): ModelingToolkit's code, one solution per mode =#
@testset "exportModel: a variable structure system" begin
  local name::String = exportTranslate("ExportVSS", OMFrontend.translateToSCode(EXPORT_VSS_PROGRAM); scalarize = false)
  @test name in OMBackend.BACKEND_SCALARIZED_MODELS
  local (expected, sols, m) = exportRoundTrip(name)
  @test throughOMBackend(m)
  @test length(sols) == length(expected) == 2
  for (s, e) in zip(sols, expected)
    @test s.retcode == ReturnCode.Success
    @test s.t[end] == e.t[end]
    @test s.u[end] ≈ e.u[end] rtol = 1e-8
  end
  #= x(0.5) = {1, 2} e^-0.5, then x' = {1, 2} =#
  @test sort(sols[end].u[end]) ≈ sort([exp(-0.5) + 0.5, 2 * exp(-0.5) + 1.0]) rtol = 1e-6
end

@testset "exportModel: a model that is not translated" begin
  local path::String = joinpath(mktempdir(), "NotTranslated.jl")
  @test_throws ErrorException OMBackend.exportModel(path, "NotTranslated")
  @test !isfile(path)
end
