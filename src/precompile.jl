#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF AGPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GNU AGPL
* VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the OSMC (Open Source Modelica Consortium)
* Public License (OSMC-PL) are obtained from OSMC, either from the above
* address, from the URLs:
* http://www.openmodelica.org or
* https://github.com/OpenModelica/ or
* http://www.ida.liu.se/projects/OpenModelica,
* and in the OpenModelica distribution.
*
* GNU AGPL version 3 is obtained from:
* https://www.gnu.org/licenses/licenses.html#GPL
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#

#= The package image's workloads: the DirectRHS solve of a pure ODE (`_precompileWarmupDirectRHS`), the MTK
   System-form solve of the structural (VSS) paths (`_precompileWarmup`), and small OMJLIR models translated, built and
   simulated by the array code and by MTK (`_precompileWarmupIR`): OMBackend is open here, so the builds evaluate into it.
   OMJL's translate over larger models of every route is the OMFrontend extension's (OMBackendOMFrontendExt, its
   precompile.jl; OMBackend is closed there, its models are translated only). Type erasure (the right-hand side
   and Jacobian behind FunctionWrappers, the callback sets as vectors: `_eraseCallbackSet`) makes the problems of every
   model of a route one type, so the solves compiled here are the ones a session's models run; a process's first model
   compiles its own generated code. =#
using PrecompileTools: @setup_workload, @compile_workload

#= ---------------------------------------------------------------------------------------------------------------
   The precompile display, as OMFrontend's: a header, one line per workload step (a mark, the step, its time), the
   total; in color when Julia has color. Julia shows it live above its progress bar when `using OMBackend`
   precompiles the package, and after the bar when Pkg precompiles it as a dependency. The OMFrontend extension's
   workload prints its own with these functions.
   --------------------------------------------------------------------------------------------------------------- =#

"""The stream of the display: stdout, in color when Julia runs with color (`--color`, passed on to precompilation)."""
_precompileIO()::IOContext = IOContext(stdout, :color => Base.get_have_color())

"""`ns` nanoseconds as `850 ms` or `1.9 s`."""
function _precompileDuration(ns::UInt64)::String
  local s::Float64 = ns / 1e9
  return s < 1 ? string(round(Int, 1000 * s), " ms") : string(round(s; digits = 1), " s")
end

"""
The display's header: `name` with OMBackend's version, Julia's version, and the `pipeline` the workload compiles.
Returns the time it started (`time_ns`), for `_precompileDone`.
"""
function _precompileHeader(io::IOContext, name::String, pipeline::String)::UInt64
  local version::String = string(something(pkgversion(@__MODULE__), "dev"))
  println(io)
  printstyled(io, "  ", name, " ", version; bold = true, color = :blue)
  printstyled(io, "   precompiling with Julia ", VERSION, "\n"; color = :light_black)
  printstyled(io, "  ", pipeline, "\n"; color = :blue)
  return time_ns()
end

"""A bar of `width` cells, `done` of `total` filled, with the count: `━━━━────  2/5  `."""
function _precompileBar(done::Int, total::Int, width::Int = 16)::String
  local filled::Int = round(Int, width * done / total)
  return string("━"^filled, "─"^(width - filled), " ", lpad(done, ndigits(total)), "/", total, "  ")
end

"""`✓ <prefix><label>  <time>`, or `✗ <prefix><label>  failed`; `prefix` in blue (a progress bar)."""
function _precompileLine(io::IOContext, label::String, ns::UInt64, failed::Bool; prefix::String = "")::Nothing
  printstyled(io, failed ? "  ✗ " : "  ✓ "; color = failed ? :red : :green, bold = failed)
  printstyled(io, prefix; color = :blue)
  print(io, rpad(label, 60 - length(prefix)))
  printstyled(io, failed ? "failed" : _precompileDuration(ns), "\n"; color = failed ? :red : :light_black)
  return nothing
end

"""
Runs `f` as a step of the display: its line (`_precompileLine`), and when `f` throws, the error as a warning (the
workload goes on). What `f` prints to stdout and stderr is dropped (`[SIMCODE: ...: check]` lines), and of its log
messages only the warnings show; `quiet`: none (the workload models' expected warnings, as Guard's assert). Returns
whether `f` ran without an error.
"""
function _precompileStep(f::F, io::IOContext, label::String; prefix::String = "", quiet::Bool = false)::Bool where {F <: Function}
  local logger::Base.CoreLogging.AbstractLogger =
    quiet ? Base.CoreLogging.NullLogger() : Base.CoreLogging.SimpleLogger(stderr, Base.CoreLogging.Warn)
  local t0::UInt64 = time_ns()
  local err::Any = nothing
  try
    redirect_stdout(devnull) do
      redirect_stderr(() -> Base.CoreLogging.with_logger(f, logger), devnull)
    end
  catch e
    err = (e, catch_backtrace())
  end
  _precompileLine(io, label, time_ns() - t0, err !== nothing; prefix)
  err === nothing || @warn "[OMBackend] precompile workload: $(label) failed" exception = err
  return err === nothing
end

"""The display's last line: the time since `t0` (`_precompileHeader`)."""
function _precompileDone(io::IOContext, t0::UInt64)::Nothing
  printstyled(io, "  Workload done in ", _precompileDuration(time_ns() - t0), "\n"; bold = true, color = :blue)
  return nothing
end

"""
Small index-1 system whose algebraic row folds away, leaving a pure-ODE
reduced system (UniformScaling mass matrix) — the most common DirectRHS class.
"""
function _precompileBuildReduced()
  @independent_variables t
  D = Differential(t)
  @variables x(t) y(t) z(t)
  @parameters a
  eqs = [D(x) ~ -a * x + y,
         D(y) ~ x - y,
         0 ~ z - (x + y)]
  sys = System(eqs, t, [x, y, z], [a];
                  name = :OMBackendPrecompileWarmup,
                  guesses = [z => 0.0],
                  initialization_eqs = ModelingToolkit.Equation[])
  local reduced = CodeGeneration.structural_simplify(sys; simplify = true,
                                                      allow_parameter = true, split = false)
  return (reduced, [x => 1.0, y => 0.0], Dict(a => 1.0))
end

"""
Mirrors the generated XModel build for the standard MTK System-form path
(structural transitions / non-DirectRHS modes).
"""
function _precompileWarmup()
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local prob = ODEProblem(reduced, merge(Dict(ivs), pars), (0.0, 1.0);
                          warn_initialize_determined = false,
                          build_initializeprob = true, fully_determined = false)
  prob = ModelingToolkit.SciMLBase.remake(prob; tspan = (0.0, 2.0))
  solve(prob, defaultSolver())
  solve(prob, daeFallbackSolver())
  return nothing
end

"""
    _precompileWarmupDirectRHS(eventful::Bool) -> Nothing

The type-erased DirectRHS solve of a pure ODE (`_precompileBuildReduced`: a UniformScaling mass matrix), as a model's
build solves it (`_precompileSolveAsModels`), without callbacks or (`eventful`) with one continuous callback whose
condition crosses zero in the solve. None of the translate workload's scalarized models is a pure ODE: without these,
MSL Thermal Motor's first simulate compiled 1.4 s more (2026-11-07).
"""
function _precompileWarmupDirectRHS(eventful::Bool)::Nothing
  local (reduced, ivs, pars) = _precompileBuildReduced()
  local callbacks::ModelingToolkit.SciMLBase.CallbackSet = eventful ?
    CodeGeneration._joinCallbacks(ModelingToolkit.SciMLBase.ContinuousCallback(_precompileCBCondition, _precompileCBAffect!)) :
    CodeGeneration._joinCallbacks()
  _precompileSolveAsModels(CodeGeneration.buildDirectRHSProblem(reduced, ivs, pars, (0.0, 2.0), callbacks), callbacks)
  return nothing
end

"The condition of `_precompileWarmupDirectRHS`'s callback: it crosses zero in the solve."
_precompileCBCondition(u::Vector{Float64}, ::Float64, _)::Float64 = u[1] - 0.5

"The affect of `_precompileWarmupDirectRHS`'s callback."
_precompileCBAffect!(_)::Nothing = nothing

"""
    _precompileSolveAsModels(prob, buildCallbacks)

The solves of a model's build of `prob` from `buildCallbacks` (the generated build and `simulateFromBuild`, and
simulateIMTK): its step control for a DAE (withAlgebraicStepControl, withProblemCallbacks: the solve's
`isoutofdomain`), the callbacks erased (`_eraseCallbackSet`), the solver `defaultSolverFor` picks from the default solver
and from the DAE fallback, the default initialization and tolerances: the integrator types a model's solve has, compiled
into the image. Solved without them, every process's first model compiled `solve!` again (2026-11-04).
"""
function _precompileSolveAsModels(prob::ModelingToolkit.SciMLBase.AbstractODEProblem,
                                  buildCallbacks::ModelingToolkit.SciMLBase.CallbackSet)::Nothing
  local callbacks::ModelingToolkit.SciMLBase.CallbackSet = CodeGeneration.withAlgebraicStepControl(buildCallbacks, prob)
  prob = CodeGeneration.withProblemCallbacks(prob, buildCallbacks, callbacks)
  prob = ModelingToolkit.SciMLBase.remake(prob; callback = CodeGeneration._eraseCallbackSet(get(prob.kwargs, :callback, nothing)))
  local initKw = CodeGeneration.defaultInitializeKwargs(prob, (;), false)
  local tolKw = CodeGeneration.defaultToleranceKwargs(prob, (;))
  for solver in (defaultSolver(), daeFallbackSolver())
    solve(prob, CodeGeneration.defaultSolverFor(solver, prob, prob.f.sys, Set{Symbol}(), false); initKw..., tolKw...)
  end
  return nothing
end

"""
The registries of OMBackend's modules (their mutable Dict and Set constants and their Refs), with copies of their
contents: the workloads restore them (`_restoreRegistries!`), so that their models' builds, generated code, function
implementations, run directories and array-code records are not kept in the image (their models' modules are). Base's
`Set` is an immutable struct over a mutable Dict: a registry, though `ismutable` is false (ARRAY_PRESERVE_DE_MODELS and
ARRAY_REDUCED_MODELS kept the workload's models before, 2026-11-09).
"""
function _registries(mod::Module = @__MODULE__, found::Vector{Pair{Any, Any}} = Pair{Any, Any}[])::Vector{Pair{Any, Any}}
  for name in names(mod; all = true)
    (isdefined(mod, name) && isconst(mod, name)) || continue
    local value::Any = getglobal(mod, name)
    if value isa Module
      (value !== mod && parentmodule(value) === mod) && _registries(value, found)
    elseif value isa Set || (value isa Union{AbstractDict, AbstractSet} && ismutable(value))
      push!(found, value => copy(value))
    elseif value isa Base.RefValue && isassigned(value)
      push!(found, value => value[])
    end
  end
  return found
end

"The registries `_registries` copied, with the contents they had."
function _restoreRegistries!(registries::Vector{Pair{Any, Any}})::Nothing
  for (registry, contents) in registries
    if registry isa AbstractDict
      empty!(registry)
      merge!(registry, contents)
    elseif registry isa AbstractSet
      empty!(registry)
      union!(registry, contents)
    else
      registry[] = contents
    end
  end
  return nothing
end

"A Real variable's reference."
_irReal(name::Symbol)::OMJLIR.CREF = OMJLIR.CREF(OMJLIR.ComponentRef(name), OMJLIR.T_REAL())

"A call of a built-in of a Real result (`der`, `pre`)."
_irBuiltin(f::Symbol, argument::OMJLIR.Expression)::OMJLIR.CALL =
  OMJLIR.CALL(f, OMJLIR.Expression[argument], OMJLIR.T_REAL(), OMJLIR.BUILTIN_CALL)

"A binary operation of Real operands."
_irBinary(a::OMJLIR.Expression, op::OMJLIR.Op, b::OMJLIR.Expression)::OMJLIR.BINARY = OMJLIR.BINARY(a, op, b, OMJLIR.T_REAL())

"A Real variable of kind `kind`, with a binding, or a start value fixed."
function _irVariable(name::Symbol, kind::OMJLIR.VariableKind; binding::Union{Nothing, Float64} = nothing,
                     start::Union{Nothing, Float64} = nothing)::OMJLIR.Variable
  local attributes::OMJLIR.VariableAttributes = start === nothing ? OMJLIR.NO_ATTRIBUTES :
    OMJLIR.VariableAttributes(OMJLIR.REAL(start), OMJLIR.BOOLEAN(true), nothing, nothing, nothing, nothing, nothing,
                              nothing, nothing)
  return OMJLIR.Variable(OMJLIR.ComponentRef(name), OMJLIR.T_REAL(), kind, binding === nothing ? nothing : OMJLIR.REAL(binding),
                         attributes, OMJLIR.NO_DIRECTION, false, OMJLIR.NOT_CONNECTOR, OMJLIR.NO_SOURCE, :model)
end

"An equation `lhs = rhs` of Reals."
_irEquation(lhs::OMJLIR.Expression, rhs::OMJLIR.Expression)::OMJLIR.SCALAR_EQUATION =
  OMJLIR.SCALAR_EQUATION(lhs, rhs, OMJLIR.T_REAL(), OMJLIR.NO_SOURCE, OMJLIR.DEFAULT_ATTRIBUTES)

"A model of one continuous-time partition, as a frontend's lowering gives it (before the passes)."
_irModel(name::String, variables::Vector{OMJLIR.Variable}, equations::Vector{OMJLIR.Equation})::OMJLIR.Model =
  OMJLIR.Model(name, OMJLIR.VariableSet(variables),
               OMJLIR.Partition[OMJLIR.Partition(1, OMJLIR.CONTINUOUS_PARTITION(), Symbol[], equations, OMJLIR.Equation[])],
               OMJLIR.Partition[], OMJLIR.ClockKind[], OMJLIR.ModelicaFunction[], nothing, OMJLIR.StructuralMode[],
               OMJLIR.StructuralTransition[], nothing)

"""
`model Decay parameter Real a = 1; Real x(start = 1, fixed = true); Real y; equation der(x) = -a * x; y = 2 * x;
end Decay;` in OMJLIR, named `name`.
"""
_irDecay(name::String)::OMJLIR.Model =
  _irModel(name, OMJLIR.Variable[_irVariable(:a, OMJLIR.PARAMETER(); binding = 1.0),
                                 _irVariable(:x, OMJLIR.ALGEBRAIC(); start = 1.0), _irVariable(:y, OMJLIR.ALGEBRAIC())],
           OMJLIR.Equation[_irEquation(_irBuiltin(:der, _irReal(:x)),
                                       OMJLIR.UNARY(OMJLIR.UMINUS, _irBinary(_irReal(:a), OMJLIR.MUL, _irReal(:x)), OMJLIR.T_REAL())),
                           _irEquation(_irReal(:y), _irBinary(OMJLIR.REAL(2.0), OMJLIR.MUL, _irReal(:x)))])

"""
`model Counter Real x(start = 1, fixed = true); discrete Real n(start = 0, fixed = true); equation der(x) = -x;
when x < 0.5 then n = pre(n) + 1; end when; end Counter;` in OMJLIR, named `name`: an event.
"""
function _irCounter(name::String)::OMJLIR.Model
  local body::Vector{OMJLIR.Statement} =
    OMJLIR.Statement[OMJLIR.ASSIGNMENT(_irReal(:n), _irBinary(_irBuiltin(:pre, _irReal(:n)), OMJLIR.ADD, OMJLIR.REAL(1.0)),
                                       OMJLIR.NO_SOURCE)]
  local condition::OMJLIR.Expression = OMJLIR.RELATION(_irReal(:x), OMJLIR.LESS, OMJLIR.REAL(0.5), 0)
  local when::OMJLIR.WHEN_EQUATION =
    OMJLIR.WHEN_EQUATION(OMJLIR.WhenBranch[LowerUtil.whenBranch(condition, body)], OMJLIR.REGULAR_WHEN, OMJLIR.NO_SOURCE,
                         OMJLIR.DEFAULT_ATTRIBUTES)
  return _irModel(name, OMJLIR.Variable[_irVariable(:x, OMJLIR.ALGEBRAIC(); start = 1.0),
                                        _irVariable(:n, OMJLIR.DISCRETE(); start = 0.0)],
                  OMJLIR.Equation[_irEquation(_irBuiltin(:der, _irReal(:x)), OMJLIR.UNARY(OMJLIR.UMINUS, _irReal(:x), OMJLIR.T_REAL())),
                                  when])
end

"""
`model Guard parameter Real a = 1; Real x(start = 1, fixed = true); equation der(x) = -homotopy(a * x, x);
assert(x > 0.5, "x below 0.5", AssertionLevel.warning); end Guard;` in OMJLIR, named `name`: homotopy() and an assert
that a crossing violates (a warning at t = log(2)).
"""
function _irGuard(name::String)::OMJLIR.Model
  local level::OMJLIR.ENUM_LITERAL =
    OMJLIR.ENUM_LITERAL(OMJLIR.T_ENUMERATION(:AssertionLevel, Symbol[:warning, :error]), :warning, 1)
  local guard::OMJLIR.ASSERT_EQUATION =
    OMJLIR.ASSERT_EQUATION(OMJLIR.RELATION(_irReal(:x), OMJLIR.GREATER, OMJLIR.REAL(0.5), 0), OMJLIR.STRING("x below 0.5"),
                           level, OMJLIR.NO_SOURCE)
  local blend::OMJLIR.CALL =
    OMJLIR.CALL(:homotopy, OMJLIR.Expression[_irBinary(_irReal(:a), OMJLIR.MUL, _irReal(:x)), _irReal(:x)],
                OMJLIR.T_REAL(), OMJLIR.BUILTIN_CALL)
  return _irModel(name, OMJLIR.Variable[_irVariable(:a, OMJLIR.PARAMETER(); binding = 1.0),
                                        _irVariable(:x, OMJLIR.ALGEBRAIC(); start = 1.0)],
                  OMJLIR.Equation[_irEquation(_irBuiltin(:der, _irReal(:x)), OMJLIR.UNARY(OMJLIR.UMINUS, blend, OMJLIR.T_REAL())),
                                  guard])
end

"""
    _precompileIRModel(model::Function, name::String, arrayCode::Bool) -> Nothing

The OMJLIR model `model(name)` translated as OMJL's `translate` does after a frontend's lowering, by the array code
(`reducedArrayModel`, `generateReducedArrayPreserveDETargetCode`; `arrayCode`) or through the passes and `translate`, then
built and simulated as OMJL's `simulate` does, with the default solver and with the DAE fallback: the build and the
solves of both routes compile into OMBackend's image without a frontend.
"""
function _precompileIRModel(model::Function, name::String, arrayCode::Bool)::Nothing
  local mode::BackendMode = DEFAULT_BACKEND_MODE[]
  if arrayCode
    local reduced::Union{Nothing, OMJLIR.Model} = reducedArrayModel(() -> model(name), name)
    (reduced !== nothing && generateReducedArrayPreserveDETargetCode(name, reduced) !== nothing) ||
      error("$(name): no array code ($(get(ARRAY_PATH_REASONS, name, "-")))")
  else
    translate(Passes.runPasses!(model(name)); mode, warnMissingStartValues = nothing,
              eliminateNonDynamic = true, observedFilter = nothing, checkSimCode = true)
  end
  for solver in (defaultSolver(), daeFallbackSolver())
    simulate(name; mode, tspan = (0.0, 1.0), solver = solver, overwriteCache = false)
  end
  return nothing
end

"""
The models of the OMJLIR workload (`_precompileWarmupIR`): the model, its name, and whether by the array code (else
through the passes and `translate`). Guard by the array code only (its asserts' and homotopy()'s drivers:
arrayAsserts.jl, arrayInitialization.jl).
"""
const PRECOMPILE_IR_MODELS = ((_irDecay, "Decay", true), (_irDecay, "Decay", false), (_irCounter, "Counter", true),
                              (_irCounter, "Counter", false), (_irGuard, "Guard", true))

"""
The OMJLIR workload: each model of `PRECOMPILE_IR_MODELS` (`_precompileIRModel`) a step of the display on `io`, the
registries restored afterwards (`_registries`). Returns the number of models that failed. A frontend's translate
workload adds the lowering and the passes of larger models (OMBackendOMFrontendExt); the build and the solve compile
here, where OMBackend is open.
"""
function _precompileWarmupIR(io::IOContext = IOContext(devnull))::Int
  local registries::Vector{Pair{Any, Any}} = _registries()
  local failed::Int = 0
  try
    local n::Int = length(PRECOMPILE_IR_MODELS)
    for (i, (model, name, arrayCode)) in enumerate(PRECOMPILE_IR_MODELS)
      local label::String = string(name, arrayCode ? " (array code)" : " (passes, translate)")
      _precompileStep(io, label; prefix = _precompileBar(i, n), quiet = true) do
        _precompileIRModel(model, string("OMBackendIRWorkload", name, arrayCode ? "Arrays" : ""), arrayCode)
      end || (failed += 1)
    end
  finally
    _restoreRegistries!(registries)
  end
  return failed
end

"The residual of `_precompileWarmupArrayHelpers`'s initial system: z = (1, 2), two blocks."
function _precompileResidual!(res::AbstractVector, z::AbstractVector)::Nothing
  res[1] = z[1] - 1.0
  res[2] = z[2] - 2.0 * z[1]
  return nothing
end

"`_precompileResidual!` as NonlinearSolve calls it (`newtonInitialSolve`), with its parameters."
_precompileResidual!(res::AbstractVector, z::AbstractVector, ::Any)::Nothing = _precompileResidual!(res, z)

"""
    _precompileWarmupArrayHelpers() -> Nothing

The array code's helpers that every array model calls, into the image: the initial system's solve
(`sortedInitialSolve`, `scaledTrustRegionSolve`: NonlinearSolve's TrustRegion over `_ScaledResidual`;
`newtonInitialSolve`, with the model's residual function `@nospecialize`: compiled once, for `Function`) and the
relation search of an event iteration (on the parameters of every translate's model, `ErasedArrayModelCode`). Without
them, a process's first model with an initial system compiled them (PID_Controller 0.14 s, 2026-11-18).
"""
function _precompileWarmupArrayHelpers()::Nothing
  local AG::Module = CodeGeneration.ArrayPreserveDE
  for reads::Union{Nothing, Vector{Vector{Int}}} in (nothing, [[1], [1, 2]])
    AG.sortedInitialSolve(_precompileResidual!, [0.0, 0.0], [1.0, 1.0], reads)
  end
  AG.newtonInitialSolve(_precompileResidual!, [0.0, 0.0])
  local P::DataType = AG.ArrayModelParameters{AG.ErasedArrayModelCode}
  local V::DataType = Vector{Float64}
  precompile(AG._searchRelations!, (V, V, V, P, Float64, Bool, Vector{Bool}, V, BitVector, Vector{Int},
                                    BitVector))
  return nothing
end

@setup_workload begin
  @compile_workload begin
    #= Escape hatch for fast dev precompiles. A workload failure must never break loading, but must be seen
       (`_precompileStep`: a ✗ line and the error): a rename or a broken path otherwise leaves the workload silently
       empty (efficiency review 2026-09-29). =#
    if !envSwitch("OMBACKEND_NO_PRECOMPILE_WORKLOAD")
      local io::IOContext = _precompileIO()
      local t0::UInt64 = _precompileHeader(io, "OMBackend", "OMJLIR ▸ Passes ▸ SimCode ▸ Code generation ▸ Build ▸ Solve")
      _precompileStep(_precompileWarmup, io, "MTK System-form solve (structural modes)")
      _precompileStep(() -> _precompileWarmupDirectRHS(false), io, "DirectRHS solve of an ODE")
      _precompileStep(() -> _precompileWarmupDirectRHS(true), io, "DirectRHS solve with an event")
      _precompileStep(_precompileWarmupArrayHelpers, io, "Array code helpers (initial system, asserts)")
      try
        _precompileWarmupIR(io)
      catch err
        @warn "[OMBackend] precompile workload: the OMJLIR models failed" exception = (err, catch_backtrace())
      end
      _precompileDone(io, t0)
    end
  end
end
