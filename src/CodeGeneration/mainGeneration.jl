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

#= The main function of an array model's executable (Author: John Tinnerholm): the model's problem, solver and
   integrator made when the executable is built (juliac), and at run time only a solve!, with no keyword calls: what
   juliac's trimming resolves at compile time (`OMBackend.Executable`, Export/exportModel.jl). Experimental. =#

"""
    executableSolver(sparse::Bool) -> Rodas5P

The solver of an array model's executable: `OMBackend.defaultSolver()`'s Rodas5P, its linear solver named: KLU for a
sparse Jacobian (`sparse`), LU for a dense one. LinearSolve's default chooses one at run time, which trimming cannot
resolve (most of a trimmed model's errors without it).
"""
function executableSolver(sparse::Bool)::OrdinaryDiffEq.Rodas5P
  local fd::ADTypes.AutoFiniteDiff = ADTypes.AutoFiniteDiff(fdtype = Val(:central))
  return sparse ? OrdinaryDiffEq.Rodas5P(autodiff = fd, linsolve = LinearSolve.KLUFactorization()) :
                  OrdinaryDiffEq.Rodas5P(autodiff = fd, linsolve = LinearSolve.LUFactorization())
end

"""
    executableIntegrator(problem, callback, nRelations::Int; reltol::Float64, abstol::Float64) -> ODEIntegrator

The integrator of an executable's solve of `problem` (a model's `specializedProblem`) with its event callbacks
`callback`, made when the executable is built: the options of the model's `simulate` (`_stepKwargs` from the
`nRelations` relations), the solver of `executableSolver`, no logging (SciMLLogging's levels dispatch at run time).
"""
function executableIntegrator(problem::SciMLBase.ODEProblem, callback::Union{Nothing, SciMLBase.CallbackSet},
                              nRelations::Int; reltol::Float64, abstol::Float64)::SciMLBase.DEIntegrator
  local sparse::Bool = problem.f.jac_prototype !== nothing
  local step::NamedTuple = _stepKwargs(pairs((;)), problem.tspan, nRelations)
  return SciMLBase.init(problem, executableSolver(sparse); callback = callback, reltol = reltol, abstol = abstol,
                        step..., verbose = DiffEqBase.DEVerbosity(SciMLLogging.None()))
end

"""
    runExecutable(integrator, stateNames::Vector{String}, args::Vector{String}) -> Cint

An executable's run: `integrator` (`executableIntegrator`, its callbacks initialized when it was made) solved, the
solution's time points and states written as CSV (a header `time,<state names>`) to the file `args[1]`, or to standard output without one. Returns
the exit code: 0 for a successful solve, 1 for a failed one (its return code and the right-hand side's last error on
standard error), 2 for wrong arguments.
"""
function runExecutable(integrator::SciMLBase.DEIntegrator, stateNames::Vector{String}, args::Vector{String})::Cint
  if length(args) > 1 || (length(args) == 1 && startswith(args[1], "-"))
    println(Core.stderr, "usage: <executable> [<result file>]: the time points and states as CSV, to standard output",
            " without a file")
    return Cint(2)
  end
  _resetRHSError!()
  #= no reinit!: its callbacks' initialization (DiffEqBase's recursion over the callback tuple) is inferred for
     `Any` beyond the first callbacks, which trimming leaves out (MissingCodeError at run time); the integrator is made
     for one solve =#
  SciMLBase.solve!(integrator)
  local sol::SciMLBase.AbstractODESolution = integrator.sol
  local out::IOBuffer = IOBuffer()
  _resultCSV(out, sol.t, sol.u, stateNames)
  if isempty(args)
    write(Core.stdout, take!(out))
  else
    write(args[1], take!(out))
  end
  SciMLBase.successful_retcode(sol.retcode) && return Cint(0)
  println(Core.stderr, "the solve failed: ", string(Symbol(sol.retcode)))
  local err::Any = LAST_RHS_ERROR[]
  err === nothing || println(Core.stderr, "the right-hand side's error in every smaller step: ", _rhsErrorText(err))
  return Cint(1)
end

"The solution's time points `ts` and states `us` as CSV into `io`, a header `time,<names>` first."
function _resultCSV(io::IOBuffer, ts::Vector{Float64}, us::Vector{Vector{Float64}}, names::Vector{String})::Nothing
  print(io, "time")
  for name::String in names
    print(io, ",\"", name, "\"")
  end
  print(io, '\n')
  for (t::Float64, u::Vector{Float64}) in zip(ts, us)
    print(io, t)
    for x::Float64 in u
      print(io, ',', x)
    end
    print(io, '\n')
  end
  return nothing
end

"The message of a right-hand side's error (`_RHS_STEP_ERRORS`), without `showerror`'s dispatch."
function _rhsErrorText(err::Any)::String
  err isa ErrorException && return err.msg
  err isa ModelicaAssertionError && return string(err.message, " (assert ", err.condition, " at t = ", err.time, ")")
  err isa DomainError && return string("a domain error: ", err.msg)
  return "an error"
end

"""
    generateExecutableMain(packageName::Symbol, modelModule::Symbol, modelFile::String, tspan::Tuple{Float64, Float64};
                           reltol::Float64, abstol::Float64) -> Expr

The package module `packageName` of an array model's executable: the model's module `modelModule` from its file
`modelFile` (the model exported as Julia source, beside it), its problem (`specializedProblem` over `tspan`) and
integrator (`executableIntegrator`) as constants, made when juliac builds the executable, and the main function
(`runExecutable`).
"""
function generateExecutableMain(packageName::Symbol, modelModule::Symbol, modelFile::String,
                                tspan::Tuple{Float64, Float64}; reltol::Float64, abstol::Float64)::Expr
  local G::Expr = :(OMBackend.CodeGeneration.ArrayPreserveDE)
  local body::Expr = quote
    import OMBackend
    include($modelFile)
    const PROBLEM = $(modelModule).specializedProblem($tspan)
    const INTEGRATOR = $(G).executableIntegrator(PROBLEM[1], PROBLEM[2], $(modelModule).N_RELATIONS;
                                                 reltol = $reltol, abstol = $abstol)
    function (@main)(args::Vector{String})::Cint
      return $(G).runExecutable(INTEGRATOR, $(modelModule).STATE_NAMES, args)
    end
  end
  return Expr(:module, true, packageName, stripBeginBlocks(body))
end
