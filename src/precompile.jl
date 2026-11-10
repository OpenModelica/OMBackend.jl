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
_precompileCBCondition(u::Vector{Float64}, t::Float64, integrator)::Float64 = u[1] - 0.5

"The affect of `_precompileWarmupDirectRHS`'s callback."
_precompileCBAffect!(integrator)::Nothing = nothing

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
`Set` is an immutable struct over a mutable Dict: a registry, though `ismutable` is false (ARRAY_ODE_MODELS and
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
    _precompileIRModel(model::Function, name::String, arrayCode::Bool) -> Nothing

The OMJLIR model `model(name)` translated as OMJL's `translate` does after a frontend's lowering, by the array code
(`reducedArrayModel`, `generateReducedArrayODETargetCode`; `arrayCode`) or through the passes and `translate`, then
built and simulated as OMJL's `simulate` does, with the default solver and with the DAE fallback: the build and the
solves of both routes compile into OMBackend's image without a frontend.
"""
function _precompileIRModel(model::Function, name::String, arrayCode::Bool)::Nothing
  local mode::BackendMode = DEFAULT_BACKEND_MODE[]
  if arrayCode
    local reduced::Union{Nothing, OMJLIR.Model} = reducedArrayModel(() -> model(name), name)
    (reduced !== nothing && generateReducedArrayODETargetCode(name, reduced) !== nothing) ||
      error("$(name): no array code ($(get(ARRAY_PATH_REASONS, name, "-")))")
  else
    translate(Passes.runPasses!(model(name)); BackendMode = mode, warnMissingStartValues = nothing,
              eliminateNonDynamic = true, observedFilter = nothing, checkSimCode = true)
  end
  for solver in (defaultSolver(), daeFallbackSolver())
    simulateModel(name; MODE = mode, tspan = (0.0, 1.0), solver = solver, overwriteCache = false)
  end
  return nothing
end

"""
The OMJLIR workload (`_precompileIRModel`): Decay and Counter by the array code and by the passes and `translate`, the
registries restored afterwards (`_registries`). A frontend's translate workload adds the lowering and the passes of
larger models (OMBackendOMFrontendExt); the build and the solve compile here, where OMBackend is open.
"""
function _precompileWarmupIR()::Nothing
  local registries::Vector{Pair{Any, Any}} = _registries()
  try
    for (model, name) in ((_irDecay, "OMBackendIRWorkloadDecay"), (_irCounter, "OMBackendIRWorkloadCounter"))
      _precompileIRModel(model, name * "Arrays", true)
      _precompileIRModel(model, name, false)
    end
  finally
    _restoreRegistries!(registries)
  end
  return nothing
end

@setup_workload begin
  @compile_workload begin
    #= Escape hatch for fast dev precompiles. A workload failure must never break
       loading, but must be seen: a rename or a broken path otherwise leaves the
       workload silently empty (efficiency review 2026-09-29). =#
    if !envSwitch("OMBACKEND_NO_PRECOMPILE_WORKLOAD")
      try
        _precompileWarmup()
      catch err
        @warn "[OMBackend] precompile warmup skipped" exception = (err, catch_backtrace())
      end
      for eventful in (false, true)
        try
          _precompileWarmupDirectRHS(eventful)
        catch err
          @warn "[OMBackend] DirectRHS precompile warmup skipped" eventful exception = (err, catch_backtrace())
        end
      end
      try
        Base.CoreLogging.with_logger(_precompileWarmupIR, Base.CoreLogging.NullLogger())
      catch err
        @warn "[OMBackend] OMJLIR precompile warmup skipped" exception = (err, catch_backtrace())
      end
    end
  end
end
