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
keep their start values (as omc).
"""
struct InitialMatching
  matchRow::Vector{Int}
  nRows::Int
  freeColumns::Vector{Tuple{String, Int}}
end

"""
    Schedule(elements, nUnknowns, matchRow, batches, pos, producer, loops, hasArrays, large, deps, pattern, solutions,
             torn, initial, needed, relNeeded, invariant)

The causal level of a domain system (`causalSchedule!`): the instances' elements and the unknowns' number (states,
algebraic variables, the discrete variables equations compute), each row's matched unknown, the schedule (batches of
instances of a class, each row's position and each unknown's producing row, the loop blocks and their families),
whether the model keeps arrays a loop may go through, the large blocks (solved by a Newton solve of their own), the
rows each state derivative depends on and the Jacobian's sparsity pattern (`nothing` where not computed); each
(class, access) a batch or a torn block solves for its unknown, solved (`_solveFor`: the target and its value;
`nothing` where the class is nonlinear in it), each large block torn (`_tearBlock`; `nothing` for another block), the
initialization's matching (`InitialMatching`), and per row whether the state derivatives need it (the right-hand
side), whether the relations need it (the crossing functions), whether it is time-invariant (an explicit solution from
parameters and invariant unknowns only: computed once per solve after the initialization).
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
  needed::BitVector
  relNeeded::BitVector
  invariant::BitVector
end

"""
    domainPasses!(g) -> Gen

The domain level's passes on the domain system `g` (`domainSystem`): its classes rerolled and folded, the alias
variables read through their roots, semiLinear chains rewritten (MLS 3.7.2.5), and the discrete variables equations
compute numbered as unknowns of the system (after the states and algebraic variables).
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
keeps arrays an algebraic loop may go through; `full`: events, asserts and `initial()` taken (`ArrayODEGen.FULL`).
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
  local matchRow::Vector{Int} = _match(el, nU)
  local (batches::Vector{Tuple{Tuple{Int, Int}, Vector{Int}}}, pos::Vector{Int}, producer::Vector{Int}, loops::LoopBlocks) =
    _schedule(el, matchRow, nU)
  if _OMBACKEND.envSwitch("OMBACKEND_ARRAY_TRACE")
    local blockSizes::Dict{Int, Int} = Dict{Int, Int}()
    foreach(b::Vector{Int} -> (blockSizes[length(b)] = get(blockSizes, length(b), 0) + 1), loops.blocks)
    println("[arraycode] ", length(g.classes), " classes, ", length(el.class), " instances; loop blocks (instances => count): ",
            sort!(collect(blockSizes)), "; ", length(loops.families), " families")
  end
  #= algebraic loops in the array code only where it keeps arrays (a class of more than one instance, and `keepsArrays`:
     not only rolled components of a model of index 1): a model without arrays gains nothing from it, and the scalar
     path solves its loops (tearing) with its initialization and events =#
  local hasArrays::Bool = keepsArrays && any(c::EqClass -> size(c.domain, 2) > 1, g.classes)
  (isempty(loops.blocks) || hasArrays) || ns("algebraic loop in a model without arrays (the scalar path solves it)")
  local large::BitVector = _largeBlocks(g, loops, matchRow)
  local deps::Union{Nothing, Vector{Vector{Int}}} = _stateDeps(el, matchRow, batches, producer, loops, g.nStates)
  local pattern::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}} = _jacobianPattern(el, matchRow, deps, g.nStates)
  local initial::InitialMatching = _initialMatching(g, deps, producer, fixedMask)
  #= initial algorithms: what they assign, which initial equations may not read =#
  local initAlgOuts::Set{String} = _initAlgAssigned(g)
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
  return Schedule(el, nU, matchRow, batches, pos, producer, loops, hasArrays, large, deps, pattern, solutions, torn,
                  initial, needed, relNeeded, invariant)
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
    _initialMatching(g, deps, producer, fixedMask) -> InitialMatching

The initial equations as a square system (`InitialMatching`): each instance matched (augmenting paths) to a state
element that is not fixed (`fixedMask`) and that it depends on (directly or through the algebraic unknowns: `deps` of
their producing rows), or to an element of a free parameter it reads. NotSupported for an instance without one
(over-determined) or a model whose dependencies were not computed (a large pattern).
"""
function _initialMatching(g::Gen, deps::Union{Nothing, Vector{Vector{Int}}}, producer::Vector{Int},
                          fixedMask::Vector{Bool})::InitialMatching
  isempty(g.initClasses) && return InitialMatching(Int[], 0, Tuple{String, Int}[])
  #= a free parameter element (name, column-major index) is unknown g.nStates + k =#
  local freeCols::Vector{Tuple{String, Int}} = Tuple{String, Int}[]
  local freeCol::Dict{Tuple{String, Int}, Int} = Dict{Tuple{String, Int}, Int}()
  deps === nothing && ns("initialization of a model with a large dependency pattern")
  #= candidate free states per initial equation instance =#
  local rows::Vector{Vector{Int}} = Vector{Int}[]
  for c in g.initClasses
    local acc::Vector{Ref} = accesses!(accesses!(Ref[], c.lhs), c.rhs)
    for j in 1:size(c.domain, 2)
      local cand::Set{Int} = Set{Int}()
      for r in acc
        local info::VarInfo = g.vars[r.name]
        if info.kind == :param
          r.name in g.freeParams || continue
          local pidx::Vector{Int} = Int[_evalInt(g, s, c.domain, j) for s in r.subs]
          local lin::Int = isempty(info.dims) ? 1 : LinearIndices(Tuple(info.dims))[pidx...]
          local key::Tuple{String, Int} = (r.name, lin)
          haskey(freeCol, key) || (push!(freeCols, key); freeCol[key] = length(freeCols))
          push!(cand, g.nStates + freeCol[key])
          continue
        end
        local idx::Vector{Int} = Int[_evalInt(g, s, c.domain, j) for s in r.subs]
        local off::Int = info.offset + _elementOffset(info, idx, r.name) + 1
        if info.kind == :state && !r.der
          push!(cand, off)
        elseif info.kind == :discrete
          local du::Int = get(g.discUnknown, off, 0)
          du > 0 && union!(cand, deps[producer[du]])
        else
          local u::Int = info.kind == :alg ? g.nStates + off : off
          union!(cand, deps[producer[u]])
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
    _augment!(rows, matchCol, matchRow, visited, r0) ||
      ns("initialization over-determined (an initial equation without a free state)")
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
