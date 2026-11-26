#= The export of translated models to other formats (Author: John Tinnerholm). =#

#= The packages the generated code loads, bound in OMBackend (the others are through backendAPI.jl): the exported
   module reaches them as `OMBackend.<package>`, so its file needs only OMBackend in the environment. =#
import Symbolics, NonlinearSolve, DiffEqCallbacks, DataStructures, Setfield, OMRuntimeExternalC

"""
    ExportFormat

A format [`exportModel`](@ref) writes a translated model in: one subtype per format, each with its method
`exportModel(io::IO, modelName::String, format)`. A format that is not one stream (a directory) has its own
`exportModel(path::String, modelName::String, format)` as well: [`JuliaSource`](@ref), a file; [`Executable`](@ref), a
package directory.
"""
abstract type ExportFormat end

"""
    JuliaSource(; comments::Bool = false)

The Julia source of a translated model: the module its translate generated (the array path's, or the
ModelingToolkit path's), loadable from a file with OMBackend in the environment, and simulated by its
`simulate(tspan = (0.0, 1.0), solver = OMBackend.defaultSolver(); kwargs...)`:

```julia
import OMBackend
include("HelloWorld.jl")
sol = HelloWorld.simulate((0.0, 2.0); reltol = 1e-6)
```

With `comments`, each line keeps its comment `#= <file>:<line> =#`, where in OMBackend's source it was generated.
The code is formatted when JuliaFormatter is loaded (the extension `OMBackendJuliaFormatterExt`).
"""
Base.@kwdef struct JuliaSource <: ExportFormat
  comments::Bool = false
end

"""
    Executable(; startTime = 0.0, stopTime = 1.0, reltol = 1e-6, abstol = 1e-10, comments = false)

A translated array model (the arrays-kept route: `translate(...; scalarize = false)`) as the package of an executable,
built with juliac: its directory holds `Project.toml`, `src/<Package>.jl` with the main function and `src/<Model>.jl`,
the model as [`JuliaSource`](@ref) (`comments`). The executable simulates the model over (`startTime`, `stopTime`)
with Rodas5P (a named linear solver, KLU or LU), tolerances `reltol` and `abstol`, and writes the time points and
states as CSV to the file given as its argument, or to standard output. Its problem and integrator are made when
juliac builds it. Exit code 0 for a successful solve, 1 for a failed one. OMBackend does not depend on juliac
(JuliaC.jl); with an environment that has it, after `Pkg.instantiate()` in the package's directory `pkg`:

```
julia --project=<env with JuliaC> -e 'using JuliaC; JuliaC.main(ARGS)' -- --output-exe model --bundle build --trim=unsafe-warn --experimental pkg
```
"""
Base.@kwdef struct Executable <: ExportFormat
  startTime::Float64 = 0.0
  stopTime::Float64 = 1.0
  reltol::Float64 = 1e-6
  abstol::Float64 = CodeGeneration.ArrayPreserveDE.DEFAULT_ABSTOL
  comments::Bool = false
end

"""
    exportModel(path::String, modelName::String, format::ExportFormat = JuliaSource()) -> String
    exportModel(io::IO, modelName::String, format::ExportFormat = JuliaSource()) -> Nothing

Writes the translated model `modelName` (either name form, [`canonicalName`](@ref)) in `format` to the file `path`,
returned, or to `io` (a directory `path` for [`Executable`](@ref)). The file is written once the whole export is done:
an export that fails leaves no file. Throws for a model that is not translated ([`availableModels`](@ref)).

```julia
OMBackend.exportModel("HelloWorld.jl", "HelloWorld")
OMBackend.exportModel(stdout, "HelloWorld")
text = sprint(OMBackend.exportModel, "HelloWorld")
```
"""
function exportModel(path::String, modelName::String, format::ExportFormat = JuliaSource())::String
  local buffer::IOBuffer = IOBuffer()
  exportModel(buffer, modelName, format)
  write(path, take!(buffer))
  return path
end

exportModel(io::IO, modelName::String)::Nothing = exportModel(io, modelName, JuliaSource())

exportModel(::IO, ::String, ::Executable)::Nothing =
  throw(ArgumentError("exportModel: an Executable is a package directory: give its path, not a stream"))

function exportModel(path::String, modelName::String, format::Executable)::String
  local name::String = canonicalName(modelName)
  (name in ARRAY_PRESERVE_DE_MODELS && haskey(COMPILED_MODELS_ARRAY_PRESERVE_DE, name)) ||
    error("exportModel: an Executable needs the array code (translate with scalarize = false); $(name) has none")
  (ispath(path) && !(isdir(path) && isempty(readdir(path)))) &&
    error("exportModel: $(path) exists; an Executable is written to a new or empty directory")
  format.startTime < format.stopTime || error("exportModel: the start time is not before the stop time")
  local source::IOBuffer = IOBuffer()
  exportModel(source, name, JuliaSource(comments = format.comments))
  local modelText::String = String(take!(source))
  local modelModule::Symbol = _generatedModule(name).args[2]::Symbol
  local packageName::Symbol = Symbol(modelModule, "Executable")
  local main::Expr = CodeGeneration.ArrayPreserveDE.generateExecutableMain(packageName, modelModule,
    string(modelModule, ".jl"), (format.startTime, format.stopTime); reltol = format.reltol, abstol = format.abstol)
  local mainText::String = _formattedJulia(sprint(print, _sourceCode(main, false); context = :module => _ExportPrint))
  #= written beside `path`, then moved there: a failed export leaves nothing =#
  local parent::String = dirname(abspath(path))
  mkpath(parent)
  local staging::String = mktempdir(parent; prefix = ".omb-export-")
  try
    mkpath(joinpath(staging, "src"))
    write(joinpath(staging, "src", string(modelModule, ".jl")), modelText)
    write(joinpath(staging, "src", string(packageName, ".jl")), OSMC_COPYRIGHT_HEADER,
          _executableUsage(name, packageName), mainText, endswith(mainText, "\n") ? "" : "\n")
    write(joinpath(staging, "Project.toml"), _executableProject(packageName, _directImports(modelText)))
    isdir(path) && rm(path)
    mv(staging, path)
  catch
    rm(staging; force = true, recursive = true)
    rethrow()
  end
  return path
end

function exportModel(io::IO, modelName::String, format::JuliaSource)::Nothing
  local name::String = canonicalName(modelName)
  local source::Expr = _sourceCode(_generatedModule(name), format.comments)
  local code::Expr = _standaloneModule(source, _valueRoots(source))
  #= the large data values (name tables) bypass the formatter, which is superlinear in the arguments of one call
     (EngineV6: a Dict of 1.47 MB took 131 s of 167): placeholders while it formats, one entry per line after =#
  local data::Dict{Symbol, Any} = Dict{Symbol, Any}()
  local withPlaceholders::Expr =
    CodeGeneration.CodeGenerationUtil._iterativePostwalk(e -> _dataPlaceholder!(data, e), code)
  local formatted::String = _formattedJulia(sprint(print, withPlaceholders; context = :module => _ExportPrint))
  local text::String = _withDataValues(formatted, data)
  local parsed::Expr = Meta.parseall(text)
  any(e -> e isa Expr && e.head in (:error, :incomplete), parsed.args) &&
    error("exportModel: the Julia source of $(name) does not parse")
  any(_unquotedInterpolation, parsed.args) &&
    error("exportModel: the Julia source of $(name) has an interpolation outside a quote")
  print(io, OSMC_COPYRIGHT_HEADER, _juliaSourceUsage(name, code.args[2]::Symbol), text)
  endswith(text, "\n") || println(io)
  return nothing
end

"""
    _generatedModule(name::String) -> Expr

The module the translate of the model `name` (canonical) generated: the array path's for a model it translated, else
the ModelingToolkit path's. Throws for a model that is not translated.
"""
function _generatedModule(name::String)::Expr
  local code::Expr = name in ARRAY_PRESERVE_DE_MODELS && haskey(COMPILED_MODELS_ARRAY_PRESERVE_DE, name) ?
    COMPILED_MODELS_ARRAY_PRESERVE_DE[name][1] : getCompiledModel(name)
  code.head === :module || error("exportModel: the generated code of $(name) is not a module")
  return code
end

"""
    _sourceCode(code::Expr, comments::Bool) -> Expr

The generated code `code` as Julia source prints it the way it evaluates. Base's `show` prints some evaluated forms as
something else:
- a `QuoteNode` of a value other than a symbol as `\$(QuoteNode(...))`, an interpolation outside a quote: the value
  itself (it prints as the code that makes it), or a quote of an expression;
- the `catch` or `finally` part of a `try` only when it is a block (`stripBeginBlocks` unblocks a part of one
  statement): the part as a block.
Without `comments`, the line numbers go: the places in OMBackend's source that generated the code (`#= <file>:<line> =#`).
`code` (the model's cached code) is not changed.
"""
_sourceCode(code::Expr, comments::Bool)::Expr =
  CodeGeneration.CodeGenerationUtil._iterativePostwalk(comments ? _sourceNodeWithLines : _sourceNode, code)

_sourceNodeWithLines(e::Any)::Any = _sourceNode(e, true)
_sourceNode(e::Any)::Any = _sourceNode(e, false)

"The node `e` of the generated code as source (`_sourceCode`), with its line numbers if `lines`."
function _sourceNode(e::Any, lines::Bool)::Any
  e isa QuoteNode && return _quotedSource(e.value, lines)
  e isa Expr || return e
  if e.head === :try
    return Expr(:try, Any[i == 2 || a === false || (a isa Expr && a.head === :block) ? a : Expr(:block, a)
                          for (i::Int, a) in enumerate(e.args)]...)
  elseif lines
    return e
  elseif e.head === :macrocall && length(e.args) >= 2 && e.args[2] isa LineNumberNode
    return Expr(:macrocall, e.args[1], nothing, e.args[3:end]...)
  elseif (e.head === :block || e.head === :quote) && any(a -> a isa LineNumberNode, e.args)
    return Expr(e.head, Any[a for a in e.args if !(a isa LineNumberNode)]...)
  end
  return e
end

"""
    _quotedSource(value::Any, lines::Bool) -> Any

The source of `QuoteNode(value)`: a symbol stays quoted, an expression is quoted (`:(...)`, evaluating to a copy; one
with an interpolation is refused, a quote would evaluate it) as source itself (`_sourceCode`, with its line numbers if
`lines`), another value is itself.
"""
function _quotedSource(value::Any, lines::Bool)::Any
  value isa Symbol && return QuoteNode(value)
  value isa Expr || return value
  _interpolates(value) && error("exportModel: the generated code quotes an expression with an interpolation: $(value)")
  return Expr(:quote, _sourceCode(value, lines))
end

"Whether `e` holds an interpolation (`\$`)."
_interpolates(e::Any)::Bool = e isa Expr && (e.head === :$ || any(_interpolates, e.args))

"Whether the parsed source `e` holds an interpolation outside a quote or a macro call: an error when the file loads."
function _unquotedInterpolation(e::Any)::Bool
  e isa Expr || return false
  e.head === :$ && return true
  (e.head === :quote || e.head === :macrocall) && return false
  return any(_unquotedInterpolation, e.args)
end

"The module whose names the source is printed for (`:module` of the IOContext): Base's alone, so a value from any
other module prints by its module's path, whatever the session has loaded into Main."
module _ExportPrint end

"""
    _valueRoots(code::Expr) -> Vector{Symbol}

The packages (root modules) the values in `code` name when printed (`_ExportPrint`): modules, functions and types, and
the values made of them, print by their modules' paths, so the file binds those packages. Base, Core, Main and
OMBackend (imported) are left out. An anonymous function is refused: the source has no name for it.
"""
function _valueRoots(code::Expr)::Vector{Symbol}
  local roots::Set{Symbol} = Set{Symbol}()
  local seen::Base.IdSet{Any} = Base.IdSet{Any}()
  CodeGeneration.CodeGenerationUtil._iterativePostwalk(code) do e
    e isa Expr || _valueRoots!(roots, seen, e)
    return e
  end
  return sort!(collect(setdiff(roots, (:Base, :Core, :Main, :OMBackend))))
end

"The packages the value `v` names when printed, into `roots` (`seen`: the mutable values visited)."
function _valueRoots!(roots::Set{Symbol}, seen::Base.IdSet{Any}, v::Any)::Nothing
  (v isa Symbol || v isa LineNumberNode || v isa QuoteNode || v isa Number || v isa AbstractString ||
   v isa AbstractChar || v === nothing) && return nothing
  if ismutable(v)
    v in seen && return nothing
    push!(seen, v)
  end
  if v isa Module
    push!(roots, nameof(Base.moduleroot(v)))
  elseif v isa GlobalRef
    push!(roots, nameof(Base.moduleroot(v.mod)))
  elseif v isa Type || v isa TypeVar
    _typeRoots!(roots, seen, v)
  elseif v isa Function && startswith(String(typeof(v).name.singletonname), '#')
    error("exportModel: the generated code holds the anonymous function $(v), which Julia source cannot name")
  elseif v isa Function && Base.issingletontype(typeof(v)) && startswith(String(typeof(v).name.name), '#')
    #= a named function (`typeof(sin)` is `#sin`): printed by its module's path =#
    push!(roots, nameof(Base.moduleroot(typeof(v).name.module)))
  else
    #= other values, callable ones (`Base.Fix1`, `∘`) too: printed by their type and fields =#
    _typeRoots!(roots, seen, typeof(v))
    if v isa AbstractDict
      for (k, x) in v
        _valueRoots!(roots, seen, k); _valueRoots!(roots, seen, x)
      end
    elseif v isa AbstractArray || v isa Tuple || v isa NamedTuple
      for x in v
        _valueRoots!(roots, seen, x)
      end
    else
      for i::Int in 1:nfields(v)
        isdefined(v, i) && _valueRoots!(roots, seen, getfield(v, i))
      end
    end
  end
  return nothing
end

"The packages the type `t` names when printed (its name's, its parameters'), into `roots`."
function _typeRoots!(roots::Set{Symbol}, seen::Base.IdSet{Any}, t::Any)::Nothing
  if t isa DataType
    push!(roots, nameof(Base.moduleroot(t.name.module)))
    for p in t.parameters
      p isa Type || p isa TypeVar ? _typeRoots!(roots, seen, p) : _valueRoots!(roots, seen, p)
    end
  elseif t isa UnionAll
    _typeRoots!(roots, seen, Base.unwrap_unionall(t))
  elseif t isa Union
    _typeRoots!(roots, seen, t.a); _typeRoots!(roots, seen, t.b)
  elseif t isa TypeVar
    _typeRoots!(roots, seen, t.lb); _typeRoots!(roots, seen, t.ub)
  elseif t isa Core.TypeofVararg
    isdefined(t, :T) && _typeRoots!(roots, seen, t.T)
  end
  return nothing
end

"""
    _standaloneModule(code::Expr, roots::Vector{Symbol}) -> Expr

The model module `code` as it loads from a file, in a session with OMBackend loaded. In OMBackend the module binds
OMBackend itself (`const OMBackend = <the module>`), is evaluated where OMBackend's packages load, and holds values
(functions, modules, types) that need no binding; from a file:
- the binding is `import OMBackend`, and each submodule imports OMBackend first (the code names its functions as
  `OMBackend.<module>.<name>`);
- a `using` or `import` of a package that OMBackend binds goes through OMBackend (`using OMBackend.OrdinaryDiffEq`);
- the packages `roots` that the printed values name (`_valueRoots`) are imported, those OMBackend binds through it.
`code` is not changed.
"""
function _standaloneModule(code::Expr, roots::Vector{Symbol})::Expr
  local body::Expr = _standaloneStatement(code.args[3], roots)::Expr
  local bound::Set{Symbol} = _importedNames(body)
  local imports::Vector{Any} = Any[_rootImport(r) for r in roots if !(r in bound)]
  return Expr(:module, code.args[1], code.args[2], Expr(:block, imports..., body.args...))
end

"A submodule of the model module, from a file: OMBackend and the packages `roots` imported first."
function _standaloneSubmodule(code::Expr, roots::Vector{Symbol})::Expr
  local body::Expr = _standaloneStatement(code.args[3], roots)::Expr
  return Expr(:module, code.args[1], code.args[2],
              Expr(:block, _IMPORT_OMBACKEND, Any[_rootImport(r) for r in roots]..., body.args...))
end

const _IMPORT_OMBACKEND = Expr(:import, Expr(:., :OMBackend))

"The import of the package `root`: through OMBackend where OMBackend binds it."
_rootImport(root::Symbol)::Expr =
  _boundPackage(root) ? Expr(:import, Expr(:., :OMBackend, root)) : Expr(:import, Expr(:., root))

"The names the top-level `using` and `import` statements of `body` bind (the last name of each path)."
function _importedNames(body::Expr)::Set{Symbol}
  local names::Set{Symbol} = Set{Symbol}()
  for s in body.args
    if s isa Expr && s.head === :block
      union!(names, _importedNames(s))
    elseif s isa Expr && (s.head === :using || s.head === :import)
      for p in s.args
        p isa Expr && p.head === :. && !isempty(p.args) && p.args[end] isa Symbol && push!(names, p.args[end])
      end
    end
  end
  return names
end

"The top-level statement `s` of the model module (or of a block of its top level), from a file (`roots`: the packages
its submodules import)."
function _standaloneStatement(s::Any, roots::Vector{Symbol})::Any
  s isa Expr || return s
  if s.head === :block
    return Expr(:block, Any[_standaloneStatement(a, roots) for a in s.args]...)
  elseif s.head === :module
    return _standaloneSubmodule(s, roots)
  elseif s.head === :const && _bindsOMBackend(s.args[1])
    return _IMPORT_OMBACKEND
  elseif s.head === :using || s.head === :import
    return Expr(s.head, Any[_throughOMBackend(p) for p in s.args]...)
  end
  return s
end

"Whether `e` is the binding `OMBackend = <the OMBackend module>`."
_bindsOMBackend(e::Any)::Bool = e isa Expr && e.head === :(=) && e.args[1] === :OMBackend && e.args[2] === OMBackend

"""
    _throughOMBackend(path::Any) -> Any

The path of a `using` or `import` (`A.B`, `A.B: x, y`, `A.B as C`) to a package OMBackend binds, through OMBackend
(`OMBackend.A.B`). Others (OMBackend, relative paths, packages OMBackend does not bind) are unchanged.
"""
function _throughOMBackend(path::Any)::Any
  path isa Expr || return path
  if path.head === :(:) || path.head === :as
    return Expr(path.head, _throughOMBackend(path.args[1]), path.args[2:end]...)
  elseif path.head === :. && !isempty(path.args) && _boundPackage(path.args[1])
    return Expr(:., :OMBackend, path.args...)
  end
  return path
end

"Whether `root` names a package (a root module) that OMBackend binds, other than OMBackend."
function _boundPackage(root::Any)::Bool
  (root isa Symbol && root !== :OMBackend && isdefined(OMBackend, root)) || return false
  local value::Any = getglobal(OMBackend, root)
  return value isa Module && parentmodule(value) === value
end

"""
The roots of the `using` and `import` paths at the top level of the module of the Julia source `text`, other than
OMBackend and relative ones: the packages it loads directly (a standard library: LinearAlgebra).
"""
function _directImports(text::String)::Vector{Symbol}
  local m::Expr = only(filter(e -> e isa Expr && e.head === :module, Meta.parseall(text).args))
  local roots::Set{Symbol} = Set{Symbol}()
  for s::Any in (m.args[3]::Expr).args
    (s isa Expr && s.head in (:using, :import)) || continue
    for p::Any in s.args
      local path::Expr = p.head in (:(:), :as) ? p.args[1] : p
      local root::Symbol = path.args[1]::Symbol
      root in (:OMBackend, :.) || push!(roots, root)
    end
  end
  return sort!(collect(roots))
end

"""
The `Project.toml` of the executable's package `packageName`: OMBackend and the packages `imports` as its
dependencies, OMBackend by its directory (`[sources]`) where it is not installed from a registry (developed).
"""
function _executableProject(packageName::Symbol, imports::Vector{Symbol})::String
  local ombackend::Base.PkgId = Base.PkgId(OMBackend)
  local deps::Vector{Base.PkgId} = Base.PkgId[ombackend]
  for root::Symbol in imports
    local ids::Vector{Base.PkgId} = Base.PkgId[id for id in keys(Base.loaded_modules) if id.name == String(root)]
    length(ids) == 1 || error("exportModel: the package $(root) the model loads is not loaded")
    push!(deps, only(ids))
  end
  local dir::String = pkgdir(OMBackend)
  local registered::Bool = any(d -> startswith(dir, joinpath(d, "packages") * "/"), DEPOT_PATH)
  local io::IOBuffer = IOBuffer()
  println(io, "name = \"", packageName, "\"")
  println(io, "uuid = \"", Base.uuid5(ombackend.uuid, string(packageName)), "\"")
  println(io, "version = \"0.1.0\"\n\n[deps]")
  for id::Base.PkgId in sort!(deps; by = id -> id.name)
    println(io, id.name, " = \"", id.uuid, "\"")
  end
  registered || println(io, "\n[sources]\nOMBackend = {path = ", repr(dir), "}")
  println(io, "\n[compat]\nOMBackend = \"", something(pkgversion(OMBackend), v"1.3.0"), "\"\njulia = \"", VERSION.major, ".",
          VERSION.minor, "\"")
  return String(take!(io))
end

"The comment at the top of the main file of the executable's package `packageName` of the model `name`."
function _executableUsage(name::String, packageName::Symbol)::String
  local version::String = string(something(pkgversion(OMBackend), "(no version)"))
  return """
    #= The executable of the model $(name), exported by OMBackend $(version) (OMBackend.exportModel, Executable).
       Built with juliac (JuliaC.jl) from this package's directory after `Pkg.instantiate()`; it writes the time points
       and states as CSV to its argument (a file) or to standard output. =#
    """
end

"The comment at the top of the Julia source of the model `name`, whose module is `moduleName`."
function _juliaSourceUsage(name::String, moduleName::Symbol)::String
  local version::String = string(something(pkgversion(OMBackend), "(no version)"))
  return """
    #= The model $(name), exported by OMBackend $(version) (OMBackend.exportModel). With OMBackend in the environment:
         import OMBackend
         include("<this file>")
         sol = $(moduleName).simulate((0.0, 1.0))  # tspan, solver = OMBackend.defaultSolver(); keywords go to solve
    =#
    """
end

"The number of entries above which a data value (a dict, vector or tuple) bypasses the formatter (`_dataPlaceholder!`)."
const _DATA_ENTRIES_UNFORMATTED = 32

"""
    _dataPlaceholder!(data::Dict{Symbol, Any}, e::Any) -> Any

`e`, or a placeholder (a name `_OMBackendData<k>_`, the value in `data`) for a `Dict`, `Vector` or tuple of more than
`_DATA_ENTRIES_UNFORMATTED` entries. [`_withDataValues`](@ref) writes the values in for their placeholders.
"""
function _dataPlaceholder!(data::Dict{Symbol, Any}, e::Any)::Any
  (e isa Dict || e isa Vector || e isa Tuple) && length(e) > _DATA_ENTRIES_UNFORMATTED || return e
  local placeholder::Symbol = Symbol("_OMBackendData", length(data) + 1, "_")
  data[placeholder] = e
  return placeholder
end

"""
    _withDataValues(text::String, data::Dict{Symbol, Any}) -> String

The formatted source `text` with the data values of `data` written for their placeholders: the value's type and one
entry per line, indented from the placeholder's line (`Dict{String, Int64}(` / `"x" => 1,` / `)`).
"""
function _withDataValues(text::String, data::Dict{Symbol, Any})::String
  isempty(data) && return text
  local pattern::Regex = r"_OMBackendData\d+_"
  local out::IOBuffer = IOBuffer()
  for line::SubString{String} in eachsplit(text, '\n')
    if occursin(pattern, line)
      local indent::String = String(match(r"^\s*", line).match)
      print(out, replace(line, pattern => p -> _dataValueText(data[Symbol(p)], indent)), '\n')
    else
      print(out, line, '\n')
    end
  end
  return String(chop(String(take!(out))))
end

"The Julia source of the data value `v` (`_dataPlaceholder!`), its entries on lines of their own, indented `indent`."
function _dataValueText(v::Any, indent::String)::String
  local io::IOContext{IOBuffer} = IOContext(IOBuffer(), :module => _ExportPrint)
  local entry::String = "\n" * indent * "    "
  if v isa Dict
    show(io, typeof(v)); print(io, "(")
    for (k, x) in v
      print(io, entry); show(io, k); print(io, " => "); show(io, x); print(io, ",")
    end
    print(io, "\n", indent, ")")
  elseif v isa Vector
    show(io, eltype(v)); print(io, "[")
    for x in v
      print(io, entry); show(io, x); print(io, ",")
    end
    print(io, "\n", indent, "]")
  else
    print(io, "(")
    for x in v::Tuple
      print(io, entry); show(io, x); print(io, ",")
    end
    print(io, "\n", indent, ")")
  end
  return String(take!(io.io))
end

"""
    _formattedJulia(text::String) -> String

The Julia source `text` formatted (JuliaFormatter, when loaded: [`formatText`](@ref)); `text` if the formatter fails.
"""
function _formattedJulia(text::String)::String
  try
    return formatText(text; remove_extra_newlines = true, indent = 4, margin = 200)
  catch e
    isFatal(e) && rethrow()
    @warn "exportModel: JuliaFormatter could not format the code; it is written as it is" exception = e
    return text
  end
end
