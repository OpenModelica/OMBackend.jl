#= exportModel's Executable: an array model's package for juliac, its main run in this session; juliac itself is
   experimental and not run here (Author: John Tinnerholm). Needs exportModelTests.jl's models and helpers. =#
using Test
import OMBackend
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

@testset "exportModel: an Executable's directory (a trailing separator, its mode)" begin
  local name::String = exportTranslate("ExportModel", OMFrontend.translateToSCode(EXPORT_PROGRAM); scalarize = false)
  try
    local parent::String = mktempdir()
    local dir::String = joinpath(parent, "pkg") * "/"
    @test OMBackend.exportModel(dir, name, OMBackend.Executable()) == dir
    @test isfile(joinpath(dir, "Project.toml"))
    #= no staging directory left beside it; the mode any new directory gets (the umask's), not 700 =#
    mkdir(joinpath(parent, "reference"))
    @test sort(readdir(parent)) == ["pkg", "reference"]
    @test filemode(joinpath(parent, "pkg")) & 0o777 == filemode(joinpath(parent, "reference")) & 0o777
  finally
    OMBackend.forgetModel(name)
  end
end
