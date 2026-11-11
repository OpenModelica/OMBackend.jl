#= Whether every definition of a source file lowers.

   Revise evaluates a changed file only if all of its definitions lower. When one does not (two typed locals of one
   name in a function, an anonymous function with a return type), it removes the changed methods and adds none, and
   the session has to be restarted. Run this on the edited files before `Revise.revise()`; it evaluates nothing.

   In a session with the package loaded (the file's macros expand in its module):
     include("utilities/lowerCheck.jl")
     lowerCheck("src/CodeGeneration/mtkExternals.jl", OMBackend.CodeGeneration)
   From the shell, for files without macros of their own module:
     julia utilities/lowerCheck.jl src/OMBackend.jl ... =#

"The error of a lowered expression, or `nothing`."
function loweringError(@nospecialize(lowered))::Union{Nothing, String}
  (lowered isa Expr && lowered.head in (:error, :incomplete)) || return nothing
  local what = lowered.args[1]
  return first(what isa Exception ? sprint(showerror, what) : string(what), 300)
end

"""
The number of definitions in `expressions` that do not lower in `mod`; each is printed with its line in `file`.
A module's body is checked in that module when `mod` has it, else in a new one.
"""
function lowerCheck(expressions::Vector{Any}, mod::Module, file::String)::Int
  local bad::Int = 0
  local line::Int = 0
  for expression in expressions
    if expression isa LineNumberNode
      line = expression.line
      continue
    end
    expression isa Expr || continue
    if expression.head === :module
      local name::Symbol = expression.args[2]
      local inner::Module = isdefined(mod, name) && getfield(mod, name) isa Module ? getfield(mod, name) : Module(name)
      bad += lowerCheck(expression.args[3].args, inner, file)
      continue
    end
    local problem::Union{Nothing, String} = try
      expression.head in (:error, :incomplete) ? loweringError(expression) : loweringError(Meta.lower(mod, expression))
    catch e
      #= a macro that `mod` does not have, or one that throws =#
      first(sprint(showerror, e), 300)
    end
    problem === nothing && continue
    bad += 1
    println("BAD ", file, ":", line, ": ", problem)
  end
  return bad
end

"The number of definitions of `file` that do not lower in `mod` (the module the file is included in)."
function lowerCheck(file::String, mod::Module = Module(:LowerCheck))::Int
  local bad::Int = lowerCheck(Meta.parseall(read(file, String); filename = file).args, mod, file)
  bad == 0 && println("ok  ", file)
  return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
  exit(sum(lowerCheck, ARGS; init = 0) == 0 ? 0 : 1)
end
