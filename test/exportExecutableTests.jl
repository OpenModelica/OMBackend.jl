#= exportModel's Executable: an array model's package for juliac, its main run in this session and, for one model,
   built with juliac and run (Author: John Tinnerholm). Needs exportModelTests.jl's models and helpers. =#
using Test
import OMBackend
import JuliaC
import Pkg

"The time points and states of a CSV written by an executable (`time,<names>` and a row per point)."
function readResultCSV(path::String)::Tuple{Vector{String}, Vector{Float64}, Vector{Vector{Float64}}}
  local lines::Vector{String} = readlines(path)
  local names::Vector{String} = String[strip(n, '"') for n in split(lines[1], ',')[2:end]]
  local rows::Vector{Vector{Float64}} = Vector{Float64}[parse.(Float64, split(l, ',')) for l in lines[2:end]]
  return (names, Float64[r[1] for r in rows], Vector{Float64}[r[2:end] for r in rows])
end

"""
    exportExecutable(name::String; stop::Float64 = 1.0) -> (expected, dir, packageName)

The translated array model `name` simulated on (0, `stop`) (`expected`), exported as an `Executable` package into
`dir`, and forgotten.
"""
function exportExecutable(name::String; stop::Float64 = 1.0)::Tuple{Any, String, Symbol}
  local dir::String = joinpath(mktempdir(), "pkg")
  local expected::Any = nothing
  try
    expected = OMBackend.simulate(name; tspan = (0.0, stop), reltol = 1e-8, abstol = 1e-8)
    @test OMBackend.exportModel(dir, name, OMBackend.Executable(stopTime = stop, reltol = 1e-8, abstol = 1e-8)) == dir
  finally
    OMBackend.forgetModel(name)
  end
  local project::Dict{String, Any} = Pkg.TOML.parsefile(joinpath(dir, "Project.toml"))
  return (expected, dir, Symbol(project["name"]))
end

"The package `packageName` in `dir` loaded into a module of its own and its main run with `args`; the exit code."
function runExecutableHere(dir::String, packageName::Symbol, args::Vector{String})::Cint
  local sandbox::Module = Module(:ExecutableSandbox)
  Base.include(sandbox, joinpath(dir, "src", string(packageName, ".jl")))
  return Base.invokelatest(() -> getglobal(getglobal(sandbox, packageName), :main)(args))::Cint
end

@testset "exportModel: an Executable's package, its main run here" begin
  local name::String = exportTranslate("ExportModel", OMFrontend.translateToSCode(EXPORT_PROGRAM); scalarize = false)
  local (expected, dir, packageName) = exportExecutable(name)
  @test sort(readdir(dir)) == ["Project.toml", "src"]
  @test sort(readdir(joinpath(dir, "src"))) == sort([string(packageName, ".jl"), "ExportModel.jl"])
  @test Pkg.TOML.parsefile(joinpath(dir, "Project.toml"))["deps"]["OMBackend"] == string(Base.PkgId(OMBackend).uuid)
  local csv::String = joinpath(mktempdir(), "result.csv")
  @test runExecutableHere(dir, packageName, String[csv]) == 0
  local (names, ts, us) = readResultCSV(csv)
  @test names == ["x"]
  @test ts[end] == 1.0
  @test us[end] ≈ expected.u[end] rtol = 1e-8
  @test us[end][1] ≈ 0.625 atol = 1e-6
  #= more than one argument, or an option: the usage, exit code 2 =#
  @test runExecutableHere(dir, packageName, String["--help"]) == 2
end

@testset "exportModel: an Executable of a model of the array path ($model)" for model::String in
    ("BouncingBalls", "SampleZOH", "TableDrive", "StickSlip")
  local name::String = exportTranslate("ArrayPath." * model, EXPORT_ARRAY_PATH_SCODE; scalarize = false)
  local (expected, dir, packageName) = exportExecutable(name)
  local csv::String = joinpath(mktempdir(), "result.csv")
  @test runExecutableHere(dir, packageName, String[csv]) == (OMBackend.SciMLBase.successful_retcode(expected.retcode) ? 0 : 1)
  local (_, ts, us) = readResultCSV(csv)
  @test ts[end] == expected.t[end]
  @test us[end] ≈ expected.u[end] rtol = 1e-8
end

@testset "exportModel: what an Executable refuses" begin
  #= ModelingToolkit's code: no array code to specialize =#
  local name::String = exportTranslate("ExportModel", OMFrontend.translateToSCode(EXPORT_PROGRAM); scalarize = true)
  try
    @test_throws ErrorException OMBackend.exportModel(joinpath(mktempdir(), "pkg"), name, OMBackend.Executable())
  finally
    OMBackend.forgetModel(name)
  end
  name = exportTranslate("ExportModel", OMFrontend.translateToSCode(EXPORT_PROGRAM); scalarize = false)
  try
    #= a directory that is not empty, a stream, a stop time not after the start =#
    local full::String = mktempdir()
    write(joinpath(full, "keep.txt"), "kept")
    @test_throws ErrorException OMBackend.exportModel(full, name, OMBackend.Executable())
    @test readdir(full) == ["keep.txt"]
    @test_throws ArgumentError OMBackend.exportModel(IOBuffer(), name, OMBackend.Executable())
    @test_throws ErrorException OMBackend.exportModel(joinpath(mktempdir(), "pkg"), name,
                                                      OMBackend.Executable(startTime = 1.0, stopTime = 1.0))
  finally
    OMBackend.forgetModel(name)
  end
end

"""
    buildExecutable(dir::String) -> String

The executable of the package `dir` built with juliac (JuliaC.jl, the test environment's: OMBackend does not depend on
it) into `dir/build`, after its environment is instantiated; its path.
"""
function buildExecutable(dir::String)::String
  Pkg.activate(() -> Pkg.instantiate(), dir)
  cd(dir) do
    withenv("OMBACKEND_NO_PRECOMPILE_WORKLOAD" => "true") do
      JuliaC.main(String["--output-exe", "model", "--bundle", joinpath(dir, "build"), "--trim=unsafe-warn",
                         "--experimental", dir])
    end
  end
  return joinpath(dir, "build", "bin", "model")
end

@testset "exportModel: an Executable built with juliac" begin
  local name::String = exportTranslate("ExportModel", OMFrontend.translateToSCode(EXPORT_PROGRAM); scalarize = false)
  local (expected, dir, _) = exportExecutable(name)
  local exe::String = buildExecutable(dir)
  @test isfile(exe)
  local csv::String = joinpath(mktempdir(), "result.csv")
  @test success(`$exe $csv`)
  local (names, ts, us) = readResultCSV(csv)
  @test names == ["x"]
  @test ts[end] == 1.0
  @test us[end] ≈ expected.u[end] rtol = 1e-8
  #= without a file: standard output =#
  @test readchomp(`$exe`) == chomp(read(csv, String))
end
