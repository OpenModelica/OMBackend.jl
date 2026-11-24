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

"""
    InitialMatching(matchRow, nRows, freeColumns)

The causal level of the initialization (`causalSchedule!`): each instance of the initial equations (by class in their
order, then by column) matched to an unknown it reaches (a state element that is not fixed, directly or through the
algebraic unknowns it depends on, or a free parameter's element: unknown `nStates + k` the `k`-th of `freeColumns`), the
instances' number, the free parameter elements (name, column-major index). Free states no initial equation reaches
keep their start values (as omc). An instance no unknown is left for has 0: an initial equation beyond the free
states (two inductors in series, each with `i = 0`: one current is a state), out of the square system and checked
at the solution (as omc removes a redundant consistent initial equation).
"""
struct InitialMatching
  matchRow::Vector{Int}
  nRows::Int
  freeColumns::Vector{Tuple{String, Int}}
end

"""
    InitialSystem(blockBase, rowBase, nLoop, reads)

The initialization as OpenModelica's initial system (`_initialSystem`): besides the free states and parameters of the
square system (`InitialMatching`), the unknowns of the algebraic loops the initial equations reach are unknowns of the
initial solve and the loops' residuals its rows, solved from the start values in blocks sorted by the structure. Per
loop block the offset of its unknowns among the loops' unknowns (-1: solved in the equations), per instance solved
alone (nonlinear in its unknown) its offset, the loops' unknowns' number, and per row of the solve (the square
system's initial equations, then the loops' residuals in their unknowns' order) the unknowns it reads (the free
states, the free parameters, then the loops' unknowns; directly or through the unknowns solved explicitly; `nothing`
where that would be large). OpenModelica solves MSL RobotR3's locked bearing friction `a_relfric = if locked then 0
else ...` for `sa` alone, after the initial equations fixed the accelerations: its residual is zero at the start value
and stays there (sa = 0, the motor holds the load). A solve of the states alone has no unknown there: it kept the
motor's current at its start and the loop put the load into the friction (sa = 1304 > tau0_max: the bearing slid).
"""
struct InitialSystem
  blockBase::Vector{Int}
  rowBase::Dict{Int, Int}
  nLoop::Int
  reads::Union{Nothing, Vector{Vector{Int}}}
end

"""
    Schedule(elements, nUnknowns, matchRow, batches, pos, producer, loops, hasArrays, large, deps, pattern, solutions,
             torn, initial, initialSystem, needed, relNeeded, invariant)

The causal level of a domain system (`causalSchedule!`): the instances' elements and the unknowns' number (states,
algebraic variables, the discrete variables equations compute), each row's matched unknown, the schedule (batches of
instances of a class, each row's position and each unknown's producing row, the loop blocks and their families),
whether the model keeps arrays a loop may go through, the large blocks (solved by a Newton solve of their own), the
rows each state derivative depends on and the Jacobian's sparsity pattern (`nothing` where not computed); each
(class, access) a batch or a torn block solves for its unknown, solved (`_solveFor`: the target and its value;
`nothing` where the class is nonlinear in it), each large block torn (`_tearBlock`; `nothing` for another block), the
initialization's matching (`InitialMatching`) and its system with the loops it reaches (`InitialSystem`), and per row
whether the state derivatives need it (the right-hand side), whether the relations need it (the crossing functions),
whether it is time-invariant (an explicit solution from parameters and invariant unknowns only: computed once per
solve after the initialization).
"""
struct Schedule
  elements::InstanceTable
  nUnknowns::Int
  matchRow::Vector{Int}
  batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}}
  pos::Vector{Int}
  producer::Vector{Int}
  loops::LoopBlocks
  hasArrays::Bool
  large::BitVector
  deps::Union{Nothing, Vector{Vector{Int}}}
  pattern::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}}
  solutions::Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}}
  torn::Vector{Union{Nothing, TornBlock}}
  initial::InitialMatching
  initialSystem::InitialSystem
  needed::BitVector
  relNeeded::BitVector
  invariant::BitVector
end

"""
    domainPasses!(g) -> Gen

The domain level's passes on the domain system `g` (`domainSystem`): its classes rerolled and folded, the alias
variables read through their roots, semiLinear chains rewritten (MLS 3.7.2.5), the discrete variables equations
compute numbered as unknowns of the system (after the states and algebraic variables), and the delay() calls
numbered (`_numberDelays!`).
"""
function domainPasses!(g::Gen)::Gen
  g.classes = _reroll(g.classes)
  g.classes = EqClass[EqClass(_folded(c.lhs), _folded(c.rhs), c.nslots, c.domain, c.nrel, c.text, c.nsample) for c in g.classes]
  g.initClasses = EqClass[EqClass(_folded(c.lhs), _folded(c.rhs), c.nslots, c.domain, c.nrel, c.text, c.nsample)
                          for c in g.initClasses]
  local nAliases::Int = _eliminateAliases!(g)
  _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE") && println("[arraycode] ", nAliases, " alias variables read through their roots")
  local nChains::Int = _simplifySemiLinear!(g)
  (nChains > 0 && _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE")) && println("[arraycode] ", nChains, " semiLinear chain(s) rewritten (MLS 3.7.2.5)")
  #= Discrete variables not assigned in a when-equation or an initial algorithm (which they keep between events) are
     computed by equations: unknowns of the system, kept in the discrete buffer (they change only at events: their
     relations are fixed between events). =#
  local assigned::Set{String} = union(Set{String}(t.name for w in g.whens for (kind, t, _) in w.body if kind == :assign),
                                      _initAlgAssigned(g))
  local nU::Int = g.nStates + g.nAlg
  for (nm, info) in sort!(collect(g.vars); by = first)
    (info.kind == :discrete && !(nm in assigned)) || continue
    for k in 1:(isempty(info.dims) ? 1 : prod(info.dims))
      g.discUnknown[info.offset + k] = (nU += 1)
    end
  end
  _numberDelays!(g)
  return g
end

"""
    causalSchedule!(g; keepsArrays, full, fixedMask) -> Schedule

The causal level of the domain system `g` after its passes (`domainSystem`, `domainPasses!`): its classes, variables
and equations read only, its parameter tables filled with the parameters the subscripts' evaluation reads (their values
and symbols, `_paramValue!`; those read structurally, `g.structural`, which a free parameter may not be). The
instances' elements, the matching, the schedule with its loops, the large blocks torn, each batch's explicit solution,
the state dependencies and the Jacobian's pattern, the initialization's matching (`fixedMask`: the states whose start
is fixed), the rows the right-hand side and the relations need and the time-invariant ones. `keepsArrays`: the model
keeps arrays an algebraic loop may go through; `full`: events, asserts and `initial()` taken (`ArrayPreserveDE.FULL`).
NotSupported for what the array code does not take.
"""
function causalSchedule!(g::Gen; keepsArrays::Bool, full::Bool, fixedMask::Vector{Bool})::Schedule
  local nU::Int = g.nStates + g.nAlg + length(g.discUnknown)
  if !full
    isempty(g.whens) || ns("when-equations (OMBackend.ARRAY_PATH_FULL)")
    isempty(g.asserts) && isempty(g.assertTexts) || ns("asserts (OMBackend.ARRAY_PATH_FULL)")
    any(c -> c.nrel > 0, g.classes) && ns("relations on continuous variables: events (OMBackend.ARRAY_PATH_FULL)")
    any(a -> a.nrel > 0 || a.nwhen > 0, g.algs) && ns("events in an algorithm (OMBackend.ARRAY_PATH_FULL)")
    g.usesInitial && ns("initial() (OMBackend.ARRAY_PATH_FULL)")
  end
  local el::InstanceTable = _elements(g)
  local matchRow::Vector{Int} = _match(el, nU, String[c.text for c in g.classes])
  #= the discrete unknowns the event iteration settles: computed from the continuous variables through relations
     only (a Boolean's `off = s < 0`), so constant between events; a loop is cut there (`_scheduleWithLoops`) =#
  local settled::BitVector = falses(nU)
  for r in 1:el.nEq
    matchRow[r] > g.nStates + g.nAlg || continue
    local own::EqClass = g.classes[el.class[r]]
    settled[matchRow[r]] = !(_continuous(g, own.lhs) || _continuous(g, own.rhs))
  end
  local (batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}}, pos::Vector{Int}, producer::Vector{Int}, loops::LoopBlocks) =
    _schedule(el, matchRow, nU, settled)
  if _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE")
    local blockSizes::Dict{Int, Int} = Dict{Int, Int}()
    foreach(b::Vector{Int} -> (blockSizes[length(b)] = get(blockSizes, length(b), 0) + 1), loops.blocks)
    println("[arraycode] ", length(g.classes), " classes, ", length(el.class), " instances; loop blocks (instances => count): ",
            sort!(collect(blockSizes)), "; ", length(loops.families), " families")
  end
  #= whether the model keeps arrays (a class of more than one instance, and `keepsArrays`: not only rolled components
     of a model of index 1). Its algebraic loops are the array code's either way: a model of scalars is one of
     arrays of no dimensions (a loop that does not start declines the array code at the translate,
     `OMBackend._arrayStartFailure`) =#
  local hasArrays::Bool = keepsArrays && any(c::EqClass -> size(c.domain, 2) > 1, g.classes)
  local large::BitVector = _largeBlocks(g, loops, matchRow, el)
  local deps::Union{Nothing, Vector{Vector{Int}}} = _stateDeps(el, matchRow, batches, producer, loops)
  local pattern::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}} = _jacobianPattern(el, matchRow, deps, g.nStates)
  local initial::InitialMatching =
    _initialMatching(g, deps, producer, fixedMask, _freeParameterDeps(g, el, batches, producer, loops))
  #= initial algorithms that run after the initial system: what they assign, which initial equations may not read =#
  local initAlgOuts::Set{String} = _initAlgAssigned(g, .!_parameterAlgorithms(g))
  for c in g.initClasses, r in accesses!(accesses!(Ref[], c.lhs), c.rhs)
    r.name in initAlgOuts && ns("initial equation reads $(r.name), which an initial algorithm assigns")
  end
  local needed::BitVector = _neededRows(g, el, matchRow, producer, loops, Int[r for r in eachindex(el.class) if 0 < matchRow[r] <= g.nStates])
  local relNeeded::BitVector = _neededRows(g, el, matchRow, producer, loops,
                                           Int[r for r in eachindex(el.class)
                                               if el.class[r] > 0 ? g.classes[el.class[r]].nrel > 0 :
                                                  (g.algs[-el.class[r]].nrel > 0 || g.algs[-el.class[r]].nwhen > 0)])
  local solutions::Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}} = Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}}()
  local invariant::BitVector = _invariantRows!(solutions, g, el, matchRow, producer, loops, batches)
  local torn::Vector{Union{Nothing, TornBlock}} = Union{Nothing, TornBlock}[nothing for _ in loops.blocks]
  for ((ci, p), batch) in batches
    if ci == 0 && large[batch[1]]
      for b in batch
        local t::TornBlock = _tearBlock(g, el, loops.blocks[b], matchRow)
        torn[b] = t
        for ((tci, tp), _) in t.batches
          local c::EqClass = g.classes[tci]
          get!(() -> _solveFor(c.lhs, c.rhs, tp, c.text), solutions, (tci, tp))
        end
      end
    elseif ci > 0 && !any(r::Int -> (r, matchRow[r]) in el.aliased, batch)
      _solution!(solutions, g.classes[ci], ci, p)
    end
  end
  #= a class with sample() calls only solved explicitly, outside an algebraic loop (its ticks' flags per instance) =#
  for ((ci, p), batch) in batches
    if ci == 0
      any(b::Int -> any(r::Int -> el.class[r] > 0 && g.classes[el.class[r]].nsample > 0, loops.blocks[b]), batch) &&
        ns("sample() in an algebraic loop")
    elseif ci > 0 && g.classes[ci].nsample > 0 && get(solutions, (ci, p), nothing) === nothing
      ns("sample() in an equation not solved explicitly: $(g.classes[ci].text)")
    end
  end
  local initialSystem::InitialSystem =
    _initialSystem(g, el, matchRow, batches, producer, loops, large, torn, solutions, initial)
  return Schedule(el, nU, matchRow, batches, pos, producer, loops, hasArrays, large, deps, pattern, solutions, torn,
                  initial, initialSystem, needed, relNeeded, invariant)
end

"""
    _solution!(solutions, c, ci, p) -> Union{Nothing, Tuple{Ref, IR}}

The class `c` (the `ci`-th) solved for its access `p` (`_solveFor`), once (`solutions`): `nothing` where it is
nonlinear in that unknown (each instance solved by Newton's method on its residual then).
"""
function _solution!(solutions::Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}}, c::EqClass, ci::Int,
                    p::Int)::Union{Nothing, Tuple{Ref, IR}}
  return get!(solutions, (ci, p)) do
    try
      _solveFor(c.lhs, c.rhs, p, c.text)
    catch err
      (err isa NotSupported && startswith(err.msg, "cannot solve")) || rethrow(err)
      nothing
    end
  end
end

"""
    _neededRows(g, el, matchRow, producer, loops, roots) -> BitVector

The rows `roots` need (backwards from them, through the algebraic unknowns they read, a loop block whole; discrete
values are read from the discrete buffer, set at events): from the rows that solve for a state derivative, those the
right-hand side computes (the MTK path likewise leaves outputs to the observed functions: acos(u) of an output was
evaluated a hair outside its domain by the solver's finite-difference time gradient); from the rows with relations,
those the crossing functions compute.
"""
function _neededRows(g::Gen, el::InstanceTable, matchRow::Vector{Int}, producer::Vector{Int}, loops::LoopBlocks,
                     roots::Vector{Int})::BitVector
  local needed::BitVector = falses(length(el.class))
  local stack::Vector{Int} = copy(roots)
  foreach(r -> (needed[r] = true), stack)
  while !isempty(stack)
    local r::Int = pop!(stack)
    #= a block is solved whole =#
    if loops.blockOf[r] > 0
      for q in loops.blocks[loops.blockOf[r]]
        needed[q] && continue
        needed[q] = true
        push!(stack, q)
      end
    end
    for i in el.uptr[r]:(el.uptr[r + 1] - 1)
      local u::Int = el.uid[i]
      (u > g.nStates + g.nAlg || u == matchRow[r] || u in el.outs[r]) && continue
      local q::Int = producer[u]
      (q > 0 && !needed[q]) || continue
      needed[q] = true
      push!(stack, q)
    end
  end
  return needed
end

"""
    _assertRows(g, el, matchRow, producer, loops, invariant, functionAsserts) -> BitVector

The rows the asserts' checks during a solve compute (`assertRows!`: `checkAsserts`, `assertCrossings!`): the rows that
give what the asserts' conditions and messages read (algebraic values, state derivatives; `_assertReads`), the
algorithms with an assert, and where a function has an assert (`functionAsserts`) the rows that call a function; with
the rows they need (`_neededRows`), but not the time-invariant ones (computed once per solve, `invariants!`, which ran
the functions they call before the solve). MSL EngineV6: its 72 assert instances read 60 of the 15470 algebraic
values; each step's two checks computed all of the equations (33 us and a buffer of 124 KB each), 36 and 91 us, now
1.6 and 1.5 us; its solve 3.23 -> 2.73 s, 2.06 -> 0.87 GB allocated.
"""
function _assertRows(g::Gen, el::InstanceTable, matchRow::Vector{Int}, producer::Vector{Int}, loops::LoopBlocks,
                     invariant::BitVector, functionAsserts::Bool)::BitVector
  local reads::Set{Int} = Set{Int}()
  foreach(a::AssertClass -> _assertReads!(reads, g, a), g.asserts)
  local roots::Vector{Int} = Int[q for q in (producer[u] for u in reads) if q > 0]
  for r in eachindex(el.class)
    local ci::Int = el.class[r]
    local check::Bool = ci < 0 ? any(st::Stmt -> _checks(st, functionAsserts), g.algs[-ci].body) :
                                 functionAsserts && (_callsFunction(g.classes[ci].lhs) || _callsFunction(g.classes[ci].rhs))
    check && push!(roots, r)
  end
  return _neededRows(g, el, matchRow, producer, loops, unique!(roots)) .& .!invariant
end

"""
    _assertReads!(reads, g, a) -> Set{Int}

The unknowns the assert `a` reads in its condition and message, over its instances, into `reads`: algebraic values
and state derivatives (as `_elements` numbers them); every element of a variable read with a subscript not known here
(`_knownSubscript`).
"""
function _assertReads!(reads::Set{Int}, g::Gen, a::AssertClass)::Set{Int}
  local acc::Vector{Ref} = accesses!(accesses!(Ref[], a.cond), a.msg)
  local idx::Vector{Int} = Int[]
  for r::Ref in acc
    local info::VarInfo = g.vars[r.name]
    local base::Int = info.kind == :alg ? g.nStates + info.offset : (info.kind == :state && r.der) ? info.offset : -1
    base < 0 && continue
    local n::Int = isempty(info.dims) ? 1 : prod(info.dims)
    for j in 1:size(a.domain, 2)
      empty!(idx)
      for s::IR in r.subs
        local k::Union{Nothing, Int} = _knownSubscript(s, a.domain, j)
        k === nothing && break
        push!(idx, k)
      end
      if length(idx) == length(info.dims) && all(i::Int -> 1 <= idx[i] <= info.dims[i], eachindex(idx))
        push!(reads, base + _elementOffset(info, idx, r.name) + 1)
      else
        foreach(k::Int -> push!(reads, base + k), 1:n)
      end
    end
  end
  return reads
end

"""
    _knownSubscript(x, d, col) -> Union{Nothing, Int}

The value of the subscript `x` at the instance `col` of the domain `d` from literals, the domain's slots and integer
arithmetic; `nothing` for another (a loop's iterator, a parameter: `_evalInt` would make it structural).
"""
function _knownSubscript(x::IR, d::Matrix{Int}, col::Int)::Union{Nothing, Int}
  x isa Lit && return x.v isa Integer ? Int(x.v) : nothing
  x isa Slot && return 1 <= x.k <= size(d, 1) ? d[x.k, col] : nothing
  x isa Op || return nothing
  local args::Vector{Union{Nothing, Int}} = Union{Nothing, Int}[_knownSubscript(y, d, col) for y in x.args]
  any(isnothing, args) && return nothing
  x.op == :+ && length(args) == 2 && return args[1] + args[2]
  x.op == :- && length(args) == 2 && return args[1] - args[2]
  x.op == :* && length(args) == 2 && return args[1] * args[2]
  x.op == :neg && length(args) == 1 && return -args[1]
  return nothing
end

"Whether `x` calls a function (`:fcall`, also in a relation's or an event-generating function's operand)."
_callsFunction(::IR)::Bool = false
_callsFunction(x::Op)::Bool = x.op == :fcall || any(_callsFunction, x.args)
_callsFunction(x::Rel)::Bool = _callsFunction(x.lhs) || _callsFunction(x.rhs)
_callsFunction(x::EvFn)::Bool = _callsFunction(x.arg)

"""
    _checks(st, functionAsserts) -> Bool

Whether the algorithm statement `st` asserts (`SAssert`) or, where a function has an assert (`functionAsserts`), calls
a function (`_callsFunction`), at any depth.
"""
_checks(::Stmt, ::Bool)::Bool = false
_checks(::SAssert, ::Bool)::Bool = true
_checks(st::SAssign, functionAsserts::Bool)::Bool = functionAsserts && _callsFunction(st.value)
_checks(st::SCall, functionAsserts::Bool)::Bool = functionAsserts && _callsFunction(st.call)
_checks(st::Union{SIf, SWhen}, functionAsserts::Bool)::Bool =
  any(b::Tuple{IR, Vector{Stmt}} -> (functionAsserts && _callsFunction(b[1])) || any(s::Stmt -> _checks(s, functionAsserts), b[2]),
      st.branches)
_checks(st::SFor, functionAsserts::Bool)::Bool = any(s::Stmt -> _checks(s, functionAsserts), st.body)
_checks(st::SWhile, functionAsserts::Bool)::Bool =
  (functionAsserts && _callsFunction(st.cond)) || any(s::Stmt -> _checks(s, functionAsserts), st.body)

"""
    _invariantRows!(solutions, g, el, matchRow, producer, loops, batches) -> BitVector

The time-invariant rows: an equation solved explicitly (`_solution!`, into `solutions`) for an algebraic unknown from
parameters and invariant unknowns only (`_readsDynamic` neither side), computed once per solve after the
initialization (`invariants!`), not in the right-hand side (MSL MultiBody's normalized axes, world frames, fixed
rotations).
"""
function _invariantRows!(solutions::Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}}, g::Gen, el::InstanceTable,
                         matchRow::Vector{Int}, producer::Vector{Int}, loops::LoopBlocks,
                         batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}})::BitVector
  local invariant::BitVector = falses(length(el.class))
  local staticClass::Dict{Int, Bool} = Dict{Int, Bool}()
  for ((ci, p), batch) in batches
    ci > 0 || continue
    local c::EqClass = g.classes[ci]
    get!(() -> c.nrel == 0 && !_readsDynamic(g, c.lhs) && !_readsDynamic(g, c.rhs), staticClass, ci) || continue
    any(r::Int -> (r, matchRow[r]) in el.aliased, batch) && continue
    _solution!(solutions, c, ci, p) === nothing && continue
    for r in batch
      (loops.blockOf[r] == 0 && g.nStates < matchRow[r] <= g.nStates + g.nAlg && el.sptr[r] == el.sptr[r + 1]) || continue
      invariant[r] = all(i::Int -> (local u::Int = el.uid[i];
                                    u == matchRow[r] || (u <= g.nStates + g.nAlg && producer[u] > 0 && invariant[producer[u]])),
                         el.uptr[r]:(el.uptr[r + 1] - 1))
    end
  end
  return invariant
end

"""
    _initialMatching(g, deps, producer, fixedMask, freeDeps) -> InitialMatching

The initial equations as a square system (`InitialMatching`): each instance matched (augmenting paths) to a state
element that is not fixed (`fixedMask`) and that it depends on (directly or through the algebraic unknowns: `deps` of
their producing rows), or to an element of a free parameter it reads (directly or through the algebraic unknowns:
`freeDeps`, `_freeParameterDeps`; not one an initial algorithm computes before: `_initialFreeParams`). An instance
without one is matched to none (0: over-determined, held only if the others' solution satisfies it). NotSupported
for a model whose dependencies were not computed (a large pattern).
"""
function _initialMatching(g::Gen, deps::Union{Nothing, Vector{Vector{Int}}}, producer::Vector{Int},
                          fixedMask::Vector{Bool}, freeDeps::Vector{Set{Tuple{String, Int}}})::InitialMatching
  isempty(g.initClasses) && return InitialMatching(Int[], 0, Tuple{String, Int}[])
  #= a free parameter element (name, column-major index) is unknown g.nStates + k =#
  local freeCols::Vector{Tuple{String, Int}} = Tuple{String, Int}[]
  local freeCol::Dict{Tuple{String, Int}, Int} = Dict{Tuple{String, Int}, Int}()
  local freeUnknown::Function = (key::Tuple{String, Int}) -> begin
    haskey(freeCol, key) || (push!(freeCols, key); freeCol[key] = length(freeCols))
    return g.nStates + freeCol[key]
  end
  deps === nothing && ns("initialization of a model with a large dependency pattern")
  local free::Set{String} = _initialFreeParams(g)
  #= candidate free states per initial equation instance =#
  local rows::Vector{Vector{Int}} = Vector{Int}[]
  for c in g.initClasses
    local acc::Vector{Ref} = accesses!(accesses!(Ref[], c.lhs), c.rhs)
    for j in 1:size(c.domain, 2)
      local cand::Set{Int} = Set{Int}()
      for r in acc
        local info::VarInfo = g.vars[r.name]
        if info.kind == :param
          r.name in free || continue
          local pidx::Vector{Int} = Int[_evalInt(g, s, c.domain, j) for s in r.subs]
          local lin::Int = isempty(info.dims) ? 1 : LinearIndices(Tuple(info.dims))[pidx...]
          push!(cand, freeUnknown((r.name, lin)))
          continue
        end
        local idx::Vector{Int} = Int[_evalInt(g, s, c.domain, j) for s in r.subs]
        local off::Int = info.offset + _elementOffset(info, idx, r.name) + 1
        if info.kind == :state && !r.der
          push!(cand, off)
        elseif info.kind == :discrete
          local du::Int = get(g.discUnknown, off, 0)
          du > 0 && union!(cand, deps[producer[du]])
          du > 0 && foreach(key -> push!(cand, freeUnknown(key)), freeDeps[producer[du]])
        else
          local u::Int = info.kind == :alg ? g.nStates + off : off
          union!(cand, deps[producer[u]])
          foreach(key -> push!(cand, freeUnknown(key)), freeDeps[producer[u]])
          r.der && push!(cand, off)
        end
      end
      push!(rows, sort!([k for k in cand if k > g.nStates || !fixedMask[k]]))
    end
  end
  #= matching rows -> free states (augmenting paths) =#
  local matchCol::Dict{Int, Int} = Dict{Int, Int}()
  local matchRow::Vector{Int} = zeros(Int, length(rows))
  for r0 in eachindex(rows)
    local visited::Set{Int} = Set{Int}()
    _augment!(rows, matchCol, matchRow, visited, r0)
  end
  return InitialMatching(matchRow, length(rows), freeCols)
end

"An augmenting path from the row `r` (`_initialMatching`): `r` matched, the rows on the path rematched."
function _augment!(rows::Vector{Vector{Int}}, matchCol::Dict{Int, Int}, matchRow::Vector{Int}, visited::Set{Int},
                   r::Int)::Bool
  for c in rows[r]
    c in visited && continue
    push!(visited, c)
    if !haskey(matchCol, c) || _augment!(rows, matchCol, matchRow, visited, matchCol[c])
      matchCol[c] = r; matchRow[r] = c
      return true
    end
  end
  return false
end

"""
    _initialReads(g, c, j, free) -> Tuple{Vector{Int}, Vector{Int}, Vector{Tuple{String, Int}}}

What the instance `j` of the initial equation class `c` reads: the states it reads (not their derivatives), the
unknowns (a state's derivative, an algebraic element; not a discrete one: the relations hold their values while the
initial system is solved), the elements (name, column-major index) of the free parameters `free`; as
`_initialMatching` reads them.
"""
function _initialReads(g::Gen, c::EqClass, j::Int,
                       free::Set{String})::Tuple{Vector{Int}, Vector{Int}, Vector{Tuple{String, Int}}}
  local states::Vector{Int} = Int[]
  local unknowns::Vector{Int} = Int[]
  local params::Vector{Tuple{String, Int}} = Tuple{String, Int}[]
  for r::Ref in accesses!(accesses!(Ref[], c.lhs), c.rhs)
    local info::VarInfo = g.vars[r.name]
    info.kind == :discrete && continue
    local idx::Vector{Int} = Int[_evalInt(g, s, c.domain, j) for s in r.subs]
    if info.kind == :param
      r.name in free || continue
      push!(params, (r.name, isempty(info.dims) ? 1 : LinearIndices(Tuple(info.dims))[idx...]))
      continue
    end
    local off::Int = info.offset + _elementOffset(info, idx, r.name) + 1
    if info.kind == :state && !r.der
      push!(states, off)
    else
      push!(unknowns, info.kind == :alg ? g.nStates + off : off)
    end
  end
  return (states, unknowns, params)
end

"""
    _initialSystem(g, el, matchRow, batches, producer, loops, large, torn, solutions, initial) -> InitialSystem

The initial system with the algebraic loops the initial equations reach (`InitialSystem`): those loop blocks (a large
block by its iteration variables and residuals, `TornBlock`) and the instances solved alone, in schedule order; each
row's reads in the solve's unknowns, an unknown solved explicitly read through what its row reads (a large block's
non-iteration unknowns through all the block reads). Empty for a model without initial equations.
"""
function _initialSystem(g::Gen, el::InstanceTable, matchRow::Vector{Int},
                        batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}}, producer::Vector{Int}, loops::LoopBlocks,
                        large::BitVector, torn::Vector{Union{Nothing, TornBlock}},
                        solutions::Dict{Tuple{Int, Int}, Union{Nothing, Tuple{Ref, IR}}},
                        initial::InitialMatching)::InitialSystem
  local N::Int = length(el.class)
  local nSA::Int = g.nStates + g.nAlg
  local blockBase::Vector{Int} = fill(-1, length(loops.blocks))
  local rowBase::Dict{Int, Int} = Dict{Int, Int}()
  initial.nRows == 0 && return InitialSystem(blockBase, rowBase, 0, nothing)
  local free::Set{String} = _initialFreeParams(g)
  #= the initial equations' instances of the square system and what each reads =#
  local initReads::Vector{Tuple{Vector{Int}, Vector{Int}, Vector{Tuple{String, Int}}}} =
    Tuple{Vector{Int}, Vector{Int}, Vector{Tuple{String, Int}}}[]
  local i::Int = 0
  for c::EqClass in g.initClasses, j::Int in 1:size(c.domain, 2)
    i += 1
    initial.matchRow[i] == 0 || push!(initReads, _initialReads(g, c, j, free))
  end
  #= the instances solved implicitly alone: nonlinear in their unknown, or reading it through two references =#
  local implicit::BitVector = falses(N)
  for ((ci::Int, p::Int), batch::Vector{Int}) in batches
    ci > 0 || continue
    (any(r::Int -> (r, matchRow[r]) in el.aliased, batch) || get(solutions, (ci, p), nothing) === nothing) || continue
    foreach(r::Int -> (implicit[r] = true), batch)
  end
  #= the instances the initial equations reach, backwards through the unknowns read (a block whole) =#
  local reached::BitVector = falses(N)
  local stack::Vector{Int} = Int[]
  local reach!::Function = (u::Int) -> begin
    (0 < u <= nSA && producer[u] > 0 && !reached[producer[u]]) || return nothing
    reached[producer[u]] = true
    push!(stack, producer[u])
    return nothing
  end
  foreach(((_, us, _),) -> foreach(reach!, us), initReads)
  while !isempty(stack)
    local q::Int = pop!(stack)
    if loops.blockOf[q] > 0
      for r::Int in loops.blocks[loops.blockOf[q]]
        reached[r] && continue
        reached[r] = true
        push!(stack, r)
      end
    end
    foreach(k::Int -> reach!(el.uid[k]), el.uptr[q]:(el.uptr[q + 1] - 1))
  end
  #= the solve's unknowns: the free states, the free parameters (as the square system orders them), the loops' =#
  local stateCol::Dict{Int, Int} = Dict{Int, Int}()
  local paramCol::Dict{Tuple{String, Int}, Int} = Dict{Tuple{String, Int}, Int}()
  local nz::Int = 0
  for k::Int in initial.matchRow
    0 < k <= g.nStates && (stateCol[k] = (nz += 1))
  end
  for k::Int in initial.matchRow
    k > g.nStates && (paramCol[initial.freeColumns[k - g.nStates]] = (nz += 1))
  end
  local loopCol::Dict{Int, Int} = Dict{Int, Int}()
  local residualRows::Vector{Int} = Int[]
  local nLoop::Int = 0
  for ((ci::Int, _), batch::Vector{Int}) in batches
    if ci == 0
      for b::Int in batch
        reached[loops.blocks[b][1]] || continue
        blockBase[b] = nLoop
        local t::Union{Nothing, TornBlock} = torn[b]
        local ids::Vector{Int} = large[b] ? t.tears : Int[matchRow[r] for r in loops.blocks[b]]
        for u::Int in ids
          loopCol[u] = nz + (nLoop += 1)
        end
        append!(residualRows, large[b] ? Int[loops.blocks[b][k] for k in t.residuals] : loops.blocks[b])
      end
    elseif ci > 0
      for r::Int in batch
        (implicit[r] && reached[r]) || continue
        rowBase[r] = nLoop
        loopCol[matchRow[r]] = nz + (nLoop += 1)
        push!(residualRows, r)
      end
    end
  end
  #= each reached instance's reads, in schedule order (what it reads is read before) =#
  local reads::Vector{Vector{Int}} = Vector{Vector{Int}}(undef, N)
  local total::Int = 0
  local readsOf!::Function = (s::Set{Int}, us::Vector{Int}, skip::Function) -> begin
    for u::Int in us
      u > nSA && continue
      haskey(loopCol, u) && (push!(s, loopCol[u]); continue)
      local q::Int = producer[u]
      (q == 0 || skip(q) || !isassigned(reads, q)) && continue
      union!(s, reads[q])
    end
    return s
  end
  local own!::Function = (s::Set{Int}, r::Int) -> begin
    foreach(k::Int -> (haskey(stateCol, k) && push!(s, stateCol[k])), el.sid[el.sptr[r]:(el.sptr[r + 1] - 1)])
    el.class[r] > 0 || return s
    local c::EqClass = g.classes[el.class[r]]
    for a::Ref in accesses!(accesses!(Ref[], c.lhs), c.rhs)
      a.name in free || continue
      local info::VarInfo = g.vars[a.name]
      local idx::Vector{Int} = Int[_evalInt(g, x, c.domain, el.col[r]) for x in a.subs]
      local key::Tuple{String, Int} = (a.name, isempty(info.dims) ? 1 : LinearIndices(Tuple(info.dims))[idx...])
      haskey(paramCol, key) && push!(s, paramCol[key])
    end
    return s
  end
  for ((ci::Int, _), batch::Vector{Int}) in batches
    if ci == 0
      for b::Int in batch
        local rs::Vector{Int} = loops.blocks[b]
        reached[rs[1]] || continue
        if large[b]
          #= the iteration variables, and what the block reads from outside =#
          local s::Set{Int} = Set{Int}(loopCol[u] for u in torn[b].tears)
          for r::Int in rs
            own!(s, r)
            readsOf!(s, el.uid[el.uptr[r]:(el.uptr[r + 1] - 1)], (q::Int) -> loops.blockOf[q] == b)
          end
          local d::Vector{Int} = sort!(collect(s))
          foreach(r::Int -> (reads[r] = d), rs)
          total += length(d) * length(rs)
        else
          for r::Int in rs
            reads[r] = sort!(collect(readsOf!(own!(Set{Int}(), r), el.uid[el.uptr[r]:(el.uptr[r + 1] - 1)],
                                              (q::Int) -> loops.blockOf[q] == b)))
            total += length(reads[r])
          end
        end
        total > 20_000_000 && return InitialSystem(blockBase, rowBase, nLoop, nothing)
      end
      continue
    end
    for r::Int in batch
      reached[r] || continue
      reads[r] = sort!(collect(readsOf!(own!(Set{Int}(), r), el.uid[el.uptr[r]:(el.uptr[r + 1] - 1)],
                                        (q::Int) -> q == r)))
      total += length(reads[r])
      total > 20_000_000 && return InitialSystem(blockBase, rowBase, nLoop, nothing)
    end
  end
  local rows::Vector{Vector{Int}} = Vector{Int}[]
  for (states::Vector{Int}, us::Vector{Int}, params::Vector{Tuple{String, Int}}) in initReads
    local s::Set{Int} = Set{Int}(stateCol[k] for k in states if haskey(stateCol, k))
    foreach(key -> (haskey(paramCol, key) && push!(s, paramCol[key])), params)
    push!(rows, sort!(collect(readsOf!(s, us, (_::Int) -> false))))
  end
  append!(rows, Vector{Int}[reads[r] for r in residualRows])
  return InitialSystem(blockBase, rowBase, nLoop, rows)
end
