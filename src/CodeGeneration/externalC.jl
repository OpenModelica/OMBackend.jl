#= External "C" functions that OMRuntimeExternalC does not have, from the C code of their
   Include annotation (Buildings' getTimeSpan.c: #include <getTimeSpan.c> with
   IncludeDirectory = "modelica://Buildings/Resources/C-Sources"): the code compiled to a
   shared library once (by its content), and OMRuntimeExternalC given a function of the
   name that ccalls it (in externalCModule()), with the C types of the Modelica declaration. The generated call
   then goes the way of the other external functions (namespaceifyExternalFunction).
   ModelicaError and the other utility functions resolve at load time from
   OMRuntimeExternalC's libModelicaCallbacks (loaded RTLD_GLOBAL). A function that a library
   OMRuntimeExternalC ships defines (its Library annotation) is called there, not compiled. =#

#= The string values of every key = "..." (or key = {"...", ...}) in an annotation (the
   flat annotation has the default IncludeDirectory before the declared one). =#
function _annotationStrings(ann::AbstractString, key::AbstractString)::Vector{String}
  local out = String[]
  for m in eachmatch(Regex("\\b" * key * "\\s*=\\s*(\\{[^}]*\\}|\"(?:[^\"\\\\]|\\\\.)*\")"), ann)
    for s in eachmatch(r"\"((?:[^\"\\]|\\.)*)\"", m.captures[1])
      push!(out, unescape_string(s.captures[1]))
    end
  end
  return out
end

#= The declarations of ModelicaUtilities.h that the callbacks library defines. =#
const _MODELICA_UTILITIES_H = """
#ifndef MODELICA_UTILITIES_H
#define MODELICA_UTILITIES_H
#include <stddef.h>
#include <stdarg.h>
#if defined(__cplusplus)
extern "C" {
#endif
void ModelicaMessage(const char *string);
void ModelicaFormatMessage(const char *string, ...);
void ModelicaVFormatMessage(const char *string, va_list args);
void ModelicaWarning(const char *string);
void ModelicaFormatWarning(const char *string, ...);
void ModelicaVFormatWarning(const char *string, va_list args);
void ModelicaError(const char *string);
void ModelicaFormatError(const char *string, ...);
void ModelicaVFormatError(const char *string, va_list args);
char* ModelicaAllocateString(size_t len);
char* ModelicaAllocateStringWithErrorReturn(size_t len);
#if defined(__cplusplus)
}
#endif
#endif
"""

#= The shared library of the C code (cached by the code and its include directories). =#
function _compileIncludeLibrary(code::String, dirs::Vector{String})::String
  local dir = joinpath(first(DEPOT_PATH), "omjl", "external-c")
  mkpath(dir)
  local key = string(hash((code, dirs)); base = 16)
  local lib = joinpath(dir, "lib" * key * (Sys.isapple() ? ".dylib" : Sys.iswindows() ? ".dll" : ".so"))
  isfile(lib) && return lib
  local src = joinpath(dir, "src" * key * ".c")
  write(src, code * "\n")
  write(joinpath(dir, "ModelicaUtilities.h"), _MODELICA_UTILITIES_H)
  local incs = String["-I" * dir]
  append!(incs, ["-I" * d for d in dirs])
  local undefinedFlag = Sys.isapple() ? `-undefined dynamic_lookup` : ``
  #= a new file, then moved: a library the process loaded is not overwritten in place
     (macOS kills a process whose mapped library's signature no longer matches) =#
  local tmp = lib * ".tmp" * string(getpid())
  local cmd = `cc -shared -fPIC -O1 -w $undefinedFlag $incs -o $tmp $src`
  try
    run(pipeline(cmd; stdout = devnull, stderr = joinpath(dir, "src" * key * ".log")))
  catch e
    e isa ProcessFailedException || rethrow()
    CodeGeneration.unsupported("an external C function whose Include code does not compile (" *
                          joinpath(dir, "src" * key * ".log") * ")", first(code, 200))
  end
  mv(tmp, lib; force = true)
  return lib
end

#= The C type of a Modelica scalar type (DAE). =#
function _cScalarType(@nospecialize(ty::DAE.Type))
  ty isa DAE.T_REAL && return :Cdouble
  (ty isa DAE.T_INTEGER || ty isa DAE.T_BOOL || ty isa DAE.T_ENUMERATION) && return :Cint
  ty isa DAE.T_STRING && return :Cstring
  #= an external object (its constructor's result, the other functions' argument): the
     C pointer (Buildings' weeklyScheduleInit, initArray, fileWriterInit) =#
  ty isa DAE.T_COMPLEX && ty.complexClassType isa DAE.ClassInf.EXTERNAL_OBJ && return :(Ptr{Cvoid})
  return nothing
end

#= The library OMRuntimeExternalC ships under a name of the Library annotation, when it
   defines the function: OMRuntimeExternalC has no Julia function for every C function of
   its libraries (MSL's ModelicaInternal_mkdir, in libModelicaExternalC; its Include,
   ModelicaInternal.h, only declares it). =#
function _shippedLibraryDefining(ann::AbstractString, cname::Symbol)::Union{String, Nothing}
  local installed = OMRuntimeExternalC.installedLibPath
  installed === nothing && return nothing
  local ext = Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so"
  for name in _annotationStrings(ann, "Library")
    local lib = joinpath(dirname(installed), "lib" * name * ext)
    isfile(lib) || continue
    local handle = Base.Libc.Libdl.dlopen(lib, Base.Libc.Libdl.RTLD_GLOBAL | Base.Libc.Libdl.RTLD_LAZY; throw_error = false)
    handle === nothing && continue
    Base.Libc.Libdl.dlsym(handle, cname; throw_error = false) === nothing || return lib
  end
  return nothing
end

"""
    ensureExternalC!(func)

For an external "C" function (SimulationCode.EXTERNAL_MODELICA_FUNCTION) whose C function
OMRuntimeExternalC has no Julia function for: one that ccalls it in the library of its Library
annotation that OMRuntimeExternalC ships, or else compiled from the code of its Include.
"""
function ensureExternalC!(func::SimulationCode.EXTERNAL_MODELICA_FUNCTION)
  local call = Meta.parse(func.libInfo)
  call isa Expr && call.head === :toplevel && length(call.args) == 1 && (call = call.args[1])
  local (resultVar, callExpr) = call isa Expr && call.head === :(=) ? (call.args[1], call.args[2]) : (nothing, call)
  (callExpr isa Expr && callExpr.head === :call && callExpr.args[1] isa Symbol) || return nothing
  local cname = callExpr.args[1]
  (isdefined(OMRuntimeExternalC, cname) || isdefined(externalCModule(), cname)) && return nothing
  local ann = get(SimulationCode.EXTERNAL_C_ANNOTATIONS, cname, "")
  local lib = _shippedLibraryDefining(ann, cname)
  if lib === nothing
    local includes = _annotationStrings(ann, "Include")
    isempty(includes) && return nothing
    #= the default IncludeDirectory (modelica://Lib/Resources/Include) of a library not loaded,
       or one that does not exist, is no directory =#
    local dirs = String[]
    for d in _annotationStrings(ann, "IncludeDirectory")
      local path = CodeGeneration.OMBackend._tryOr(() -> SimulationCode.OMFrontend.resolveModelicaURI(d), nothing,
                                                   :externalCIncludeDirectory; only = ErrorException)
      path !== nothing && isdir(path) && push!(dirs, path)
    end
    lib = _compileIncludeLibrary(join(includes, "\n"), dirs)
  end
  local vars = Dict{Symbol, DAE.VAR}()
  for v in Iterators.flatten((func.inputs, func.outputs, func.locals))
    vars[Symbol(DAE_VAR_ToJulia(v))] = v
  end
  local outputs = Set{Symbol}(Symbol(DAE_VAR_ToJulia(v)) for v in func.outputs)
  local params = Symbol[]; local types = Any[]; local args = Any[]; local prep = Expr[]; local back = Expr[]
  for (k, a) in enumerate(callExpr.args[2:end])
    local p = Symbol("a", k)
    push!(params, p)
    if a isa Symbol && haskey(vars, a)
      local v = vars[a]
      if _funcParamIsArray(v)
        local ct = _cScalarType(_funcParamElemType(v))
        (ct === nothing || ct === :Cstring) && CodeGeneration.unsupported("an external C function's array argument of this type", a)
        local jt = ct === :Cdouble ? :Float64 : :Cint
        local c = Symbol("c", k)
        #= Modelica arrays are row-major in C =#
        push!(prep, :(local $c = Array{$jt}(ndims($p) > 1 ? permutedims($p, ndims($p):-1:1) : $p)))
        push!(types, :(Ptr{$jt})); push!(args, c)
        a in outputs && push!(back, :(ndims($p) > 1 ? ($p .= permutedims($c, ndims($c):-1:1)) : ($p .= $c)))
      else
        a in outputs && CodeGeneration.unsupported("an external C function's scalar output argument", a)
        local ct = _cScalarType(v.ty)
        ct === nothing && CodeGeneration.unsupported("an external C function's argument of this type", a)
        push!(types, ct); push!(args, ct === :Cdouble ? :(Float64($p)) : ct === :Cint ? :(Cint($p)) : p)
      end
    elseif a isa AbstractString
      push!(types, :Cstring); push!(args, p)
    elseif a isa AbstractFloat
      push!(types, :Cdouble); push!(args, :(Float64($p)))
    else
      #= an Integer literal or size(x, k) =#
      push!(types, :Cint); push!(args, :(Cint($p)))
    end
  end
  local rt = resultVar === nothing ? :Cvoid :
    (resultVar isa Symbol && haskey(vars, resultVar) ? _cScalarType(vars[resultVar].ty) : nothing)
  rt === nothing && CodeGeneration.unsupported("an external C function's result of this type", func.libInfo)
  local ccallExpr = Expr(:call, :ccall, Expr(:tuple, QuoteNode(cname), lib), rt === :Cstring ? :(Ptr{UInt8}) : rt,
                         Expr(:tuple, types...), args...)
  local body = quote
    $(prep...)
    local r = $ccallExpr
    $(back...)
    return $(rt === :Cstring ? :(unsafe_string(r)) : rt === :Cvoid ? :nothing : :r)
  end
  Core.eval(externalCModule(), Expr(:function, Expr(:call, cname, params...), body))
  return nothing
end

#= The functions ensureExternalC! defines: a module made at run time (OMRuntimeExternalC is
   a closed, precompiled module). =#
const _EXTERNAL_C_MODULE = Ref{Union{Module, Nothing}}(nothing)
function externalCModule()::Module
  local m = _EXTERNAL_C_MODULE[]
  m === nothing || return m
  m = Module(:OMJLExternalC)
  _EXTERNAL_C_MODULE[] = m
  return m
end
