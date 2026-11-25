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

#= Author: John Tinnerholm

   Calls of the model's pure Modelica functions shared in a generated function of the array models (ArrayPreserveDE): a
   record-valued call is one call per field read (`waterBaseProp_ph(p, h, phase, region)[k]` of MSL Media's IF97, 988 of
   them in MSL BatchPlant_StandardWater's right-hand side with 10 different arguments), each evaluating the whole record.
   The scalar path calls such a function once per arguments too (`_hoistExtractorCalls!`). =#

"""
A read of a generated function's state by a shared call's argument: the buffer (`:u`, `:a`, `:du`, `:y`, `:res`; `:pd`
for the discrete values `p.d`, `:pv` for the parameter values `p.values`) and the element (its literal linear index;
0 for any element). `(:all, 0)`: everything (a write the pass does not resolve).
"""
const StateRead = Tuple{Symbol, Int}

"The buffers a generated function's statements read and write by name (`StateRead`)."
const _SHARED_BUFFERS = (:u, :a, :du, :y, :res)

"Operations allowed around a shared call's argument reads (`_plainArgument!`): no state written, no call of the model."
const _PLAIN_OPERATIONS = (:round, :Int, :Float64, :Bool, :float, :+, :-, :*, :/, :^, :abs, :!, :(==), :!=, :min, :max)

"""
    _bufferOf(x) -> Union{Nothing, Symbol}

The buffer `x` names (`StateRead`): one of `_SHARED_BUFFERS`, `p.d` as `:pd`, `p.values` as `:pv`; `nothing` for
anything else.
"""
function _bufferOf(@nospecialize(x::Any))::Union{Nothing, Symbol}
  x isa Symbol && return x in _SHARED_BUFFERS ? x : nothing
  (x isa Expr && x.head === :. && x.args[1] === :p && x.args[2] isa QuoteNode) || return nothing
  local field::Any = (x.args[2]::QuoteNode).value
  return field === :d ? :pd : field === :values ? :pv : nothing
end

"""
    _plainArgument!(reads, x) -> Bool

Whether `x` is an argument a shared call may take (`_isSharedCall`): literals, `t`, the parameter locals (`P_...`), the
shared values (`_call...`), elements of the buffers at literal subscripts, and `_PLAIN_OPERATIONS` of these; the
elements it reads onto `reads`.
"""
function _plainArgument!(reads::Vector{StateRead}, @nospecialize(x::Any))::Bool
  (x isa Number || x isa QuoteNode || x isa String) && return true
  if x isa Symbol
    x === :t && return true
    x in (:Int, :Float64, :Bool) && return true
    local name::String = String(x)
    return startswith(name, "P_") || startswith(name, "_call")
  end
  x isa Expr || return false
  if x.head === :ref
    local buffer::Union{Nothing, Symbol} = _bufferOf(x.args[1])
    buffer === nothing && return false
    all(i::Any -> i isa Int, @view x.args[2:end]) || return false
    push!(reads, (buffer, length(x.args) == 2 ? x.args[2]::Int : 0))
    return true
  end
  (x.head === :call && x.args[1] isa Symbol && (x.args[1]::Symbol) in _PLAIN_OPERATIONS) || return false
  return all(a::Any -> _plainArgument!(reads, a), @view x.args[2:end])
end

"Whether `x` is a call of one of the pure functions `pure` whose arguments are all plain (`_plainArgument!`)."
function _isSharedCall(@nospecialize(x::Any), pure::Set{Symbol})::Bool
  (x isa Expr && x.head === :call && x.args[1] isa Symbol && (x.args[1]::Symbol) in pure) || return false
  local reads::Vector{StateRead} = StateRead[]
  return all(a::Any -> _plainArgument!(reads, a), @view x.args[2:end])
end

"The elements the shared call `call` reads (`_plainArgument!`)."
function _callReads(call::Expr)::Vector{StateRead}
  local reads::Vector{StateRead} = StateRead[]
  for a in @view call.args[2:end]
    _plainArgument!(reads, a)
  end
  return reads
end

"The calls of the pure functions `pure` in `x`, at any position, counted by their expression (`_isSharedCall`)."
function _countSharedCalls!(counts::Dict{Expr, Int}, @nospecialize(x::Any), pure::Set{Symbol})::Nothing
  x isa Expr || return nothing
  if _isSharedCall(x, pure)
    counts[x] = get(counts, x, 0) + 1
    return nothing
  end
  for a in x.args
    _countSharedCalls!(counts, a, pure)
  end
  return nothing
end

"""
Whether the call head `f` may write what a shared call reads: a mutating function (`!`), except the relations', event
functions', delays' and asserts' helpers (they write the run state, `p.rel`, `p.evv`, the histories).
"""
function _mayWriteState(@nospecialize(f::Any), pure::Set{Symbol})::Bool
  if f isa Symbol
    f in pure && return false
    return endswith(String(f), "!")
  end
  f isa Function && return !(f === _rel || f === _timeRel || f === _evfn || f === _checkedIndex) &&
                           endswith(String(nameof(f)), "!")
  f isa Expr && f.head === :. && f.args[end] isa QuoteNode && return endswith(String((f.args[end]::QuoteNode).value), "!")
  return true
end

"The element an assignment to `lhs` writes onto `writes` (`StateRead`): `(:all, 0)` where it is not a buffer element."
function _assignmentWrites!(writes::Vector{StateRead}, @nospecialize(lhs::Any))::Nothing
  if lhs isa Expr && lhs.head === :ref
    local buffer::Union{Nothing, Symbol} = _bufferOf(lhs.args[1])
    buffer === nothing && (push!(writes, (:all, 0)); return nothing)
    push!(writes, (buffer, length(lhs.args) == 2 && lhs.args[2] isa Int ? lhs.args[2]::Int : 0))
  elseif lhs isa Expr && lhs.head === :tuple
    for x in lhs.args
      _assignmentWrites!(writes, x)
    end
  elseif lhs isa Symbol
    (lhs in _SHARED_BUFFERS || lhs === :p || startswith(String(lhs), "P_")) && push!(writes, (:all, 0))
  else
    push!(writes, (:all, 0))
  end
  return nothing
end

"What the statement `x` may write, at any position (`StateRead`): assignments and the calls of `_mayWriteState`."
function _stateWrites!(writes::Vector{StateRead}, @nospecialize(x::Any), pure::Set{Symbol})::Nothing
  x isa Expr || return nothing
  if x.head === :local || x.head === :const
    #= a local's definition writes the local, its value is walked below =#
  elseif x.head in (:(=), :(+=), :(-=), :(*=), :(/=), :(.=), :(.+=), :(.-=), :(.*=), :(./=))
    _assignmentWrites!(writes, x.args[1])
  elseif x.head === :call && _mayWriteState(x.args[1], pure)
    push!(writes, (:all, 0))
  end
  for a in x.args
    #= a local's definition: its name is no write =#
    (x.head === :local && a isa Expr && a.head === :(=) && a.args[1] isa Symbol) && (_stateWrites!(writes, a.args[2], pure); continue)
    _stateWrites!(writes, a, pure)
  end
  return nothing
end

"Whether the reads `reads` meet the writes `writes` (`StateRead`; element 0: any element)."
function _readsWritten(reads::Vector{StateRead}, writes::Vector{StateRead})::Bool
  for (wb::Symbol, wi::Int) in writes
    wb === :all && return true
    for (rb::Symbol, ri::Int) in reads
      rb === wb && (wi == 0 || ri == 0 || wi == ri) && return true
    end
  end
  return false
end

"Whether `x` leaves its statement early (`return`, `break`, `continue`, a `@goto`): then nothing of it is shared."
function _exitsEarly(@nospecialize(x::Any))::Bool
  x isa Expr || return false
  x.head in (:return, :break, :continue, :symbolicgoto) && return true
  x.head === :macrocall && return true
  return any(_exitsEarly, x.args)
end

"""
Whether the `let` block `x` binds a name a shared call's arguments read (`_plainArgument!`): a buffer, `p`, `t`, a
parameter local or a shared value.
"""
function _letShadows(x::Expr)::Bool
  local bindings::Vector{Any} = x.args[1] isa Expr && x.args[1].head === :block ? x.args[1].args : Any[x.args[1]]
  for b in bindings
    b isa LineNumberNode && continue
    local name::Any = b isa Expr && b.head === :(=) ? b.args[1] : b
    name isa Symbol || return true
    (name in _SHARED_BUFFERS || name === :p || name === :t || startswith(String(name), "P_") ||
     startswith(String(name), "_call")) && return true
  end
  return false
end

"""
The shared calls (`_isSharedCall`, counted more than once in `counts`) `x` evaluates whenever it runs, in the order it
evaluates them, onto `found`: the operands of calls, assignments' values, the conditions of `if`, `&&` and `||`, the
bindings and body of a `let` block (an instance's `_col`) that binds none of the names they read; not the branches,
loops or closures.
"""
function _unconditionalCalls!(found::Vector{Expr}, @nospecialize(x::Any), pure::Set{Symbol}, counts::Dict{Expr, Int})::Nothing
  x isa Expr || return nothing
  local h::Symbol = x.head
  if h === :call
    if _isSharedCall(x, pure)
      (get(counts, x, 0) > 1 && !(x in found)) && push!(found, x)
      return nothing
    end
    for a in @view x.args[2:end]
      _unconditionalCalls!(found, a, pure, counts)
    end
  elseif h in (:if, :elseif, :&&, :||)
    _unconditionalCalls!(found, x.args[1], pure, counts)
  elseif h in (:(=), :(+=), :(-=), :(*=), :(/=))
    _unconditionalCalls!(found, x.args[2], pure, counts)
  elseif h in (:block, :ref, :vect, :tuple, :local, :...) || (h === :let && !_letShadows(x))
    for a in x.args
      _unconditionalCalls!(found, a, pure, counts)
    end
  end
  return nothing
end

"`x` with the shared calls of `shared` replaced by their locals, at any position but in a `let` that shadows (`_letShadows`)."
function _replaceShared(@nospecialize(x::Any), shared::Dict{Expr, Tuple{Symbol, Vector{StateRead}}}, pure::Set{Symbol})::Any
  x isa Expr || return x
  x.head === :let && _letShadows(x) && return x
  if x.head === :call && x.args[1] isa Symbol && (x.args[1]::Symbol) in pure
    local entry::Union{Nothing, Tuple{Symbol, Vector{StateRead}}} = get(shared, x, nothing)
    entry === nothing || return entry[1]
  end
  return Expr(x.head, Any[_replaceShared(a, shared, pure) for a in x.args]...)
end

"""
    _shareCalls(stmts, pure, counter) -> Vector{Any}

The statements `stmts` of a generated function with each call of a pure function (`pure`: the model's functions not
called as impure, `Gen.impure`) on plain arguments (`_isSharedCall`) that they make more than once evaluated once: at
the first statement that evaluates it whenever it runs (`_unconditionalCalls!`) a local (`_callN`, `counter` numbering
them) takes its value before that statement, and the later statements read it, until a statement may write what its
arguments read (`_stateWrites!`): from that one on it is evaluated again. A statement that writes what a call reads
or leaves early shares none of its own. A pure function's value is the same for the same arguments (MLS 12.3).
"""
function _shareCalls(stmts::Vector{Any}, pure::Set{Symbol}, counter::Base.RefValue{Int})::Vector{Any}
  isempty(pure) && return stmts
  local counts::Dict{Expr, Int} = Dict{Expr, Int}()
  for st in stmts
    _countSharedCalls!(counts, st, pure)
  end
  any(n::Int -> n > 1, values(counts)) || return stmts
  local shared::Dict{Expr, Tuple{Symbol, Vector{StateRead}}} = Dict{Expr, Tuple{Symbol, Vector{StateRead}}}()
  local out::Vector{Any} = Any[]
  local writes::Vector{StateRead} = StateRead[]
  local found::Vector{Expr} = Expr[]
  for st in stmts
    empty!(writes)
    _stateWrites!(writes, st, pure)
    isempty(writes) || filter!(kv -> !_readsWritten(kv[2][2], writes), shared)
    local current::Any = isempty(shared) ? st : _replaceShared(st, shared, pure)
    empty!(found)
    _exitsEarly(current) || _unconditionalCalls!(found, current, pure, counts)
    local added::Bool = false
    for call in found
      local reads::Vector{StateRead} = _callReads(call)
      _readsWritten(reads, writes) && continue
      counter[] += 1
      local name::Symbol = Symbol("_call", counter[])
      push!(out, :(local $name = $call))
      shared[call] = (name, reads)
      added = true
    end
    push!(out, added ? _replaceShared(current, shared, pure) : current)
  end
  return out
end
