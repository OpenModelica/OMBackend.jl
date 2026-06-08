#=
Expr-level equation rewriting. Operates directly on Julia Expr trees
instead of the old Symbolics round-trip.
=#

"""
Find `:(der(x))` in an Expr tree. Returns the variable symbol or `nothing`.
"""
function findDerivativeVar(expr::Expr)
    # Match both :der (DAE-emit) and :D (SIM-Exp-emit) so the residual→explicit
    # rewrite recognises the derivative regardless of which codegen path produced it.
    if expr.head == :call && length(expr.args) >= 2 &&
       (expr.args[1] === :der || expr.args[1] === :D)
        return expr.args[2]
    end
    for a in expr.args
        result = findDerivativeVar(a)
        if result !== nothing
            return result
        end
    end
    return nothing
end

findDerivativeVar(x) = nothing

"""
Replace `:(der(var))` with `val` throughout an Expr tree. Returns a new Expr.
"""
function substituteDer(expr::Expr, var, val)
    if expr.head == :call && length(expr.args) >= 2 &&
       (expr.args[1] === :der || expr.args[1] === :D) && expr.args[2] == var
        return val
    end
    newargs = Any[substituteDer(a, var, val) for a in expr.args]
    return Expr(expr.head, newargs...)
end

substituteDer(x, var, val) = x

"""
Transform `:(0 ~ expr)` into `:(D(x) ~ -rest/coeff)` where the derivative
coefficient is extracted via linearity: substitute der(x)=0 and der(x)=1.
Algebraic equations pass through unchanged.
"""
function moveDerivativeToLHS(eq_expr::Expr)
    if !(eq_expr.head == :call && length(eq_expr.args) == 3 && eq_expr.args[1] === :~)
        return eq_expr
    end
    rhs = eq_expr.args[3]
    der_var = findDerivativeVar(rhs isa Expr ? rhs : Expr(:block, rhs))
    if der_var === nothing
        return eq_expr
    end
    rest  = rhs isa Expr ? substituteDer(rhs, der_var, 0) : rhs
    atOne = rhs isa Expr ? substituteDer(rhs, der_var, 1) : rhs
    coeff = Expr(:call, :-, atOne, rest)
    return Expr(:call, :~, Expr(:call, :D, der_var),
                Expr(:call, :/, Expr(:call, :-, rest), coeff))
end

"""
Rename `:der` to `:D` in call positions, in place.
For non-residual equations (e.g. `:(der(x) ~ 0)`) where `moveDerivativeToLHS`
does not apply.
"""
function renameDerToD!(expr::Expr)
    if expr.head == :call && length(expr.args) >= 1 && expr.args[1] === :der
        expr.args[1] = :D
    end
    for a in expr.args
        renameDerToD!(a)
    end
    return expr
end

renameDerToD!(x) = x

"""
Qualify bare Modelica function calls with `OMBackend.CodeGeneration.` prefix, in place.
"""
function qualifyModelicaFunctions!(expr::Expr, funcNames::OrderedSet{Symbol})
    #= Iterative traversal: deeply nested wrappers (e.g. the generated body for
       Modelica.Utilities.Strings.scanToken and its scan-family helpers) push
       the previous recursive walker past Julia's runtime stack guard, firing
       SIGSEGV recoveries that cost ~30ms each. A worklist keeps the call
       depth flat regardless of AST shape. =#
    #= Dedup by object identity: the generated model Expr is a DAG with heavy
       sub-Expr sharing, so a plain worklist re-expands shared nodes
       exponentially. Visiting each unique node once is sufficient (the rewrite
       mutates the shared object in place, visible through every reference). =#
    local worklist = Any[expr]
    local visited = Base.IdSet{Any}()
    while !isempty(worklist)
        local e = pop!(worklist)
        e isa Expr || continue
        e in visited && continue
        push!(visited, e)
        if e.head == :call && length(e.args) >= 1
            if e.args[1] isa Symbol && e.args[1] in funcNames
                e.args[1] = Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(e.args[1]))
            elseif e.args[1] == :(Base.invokelatest) && length(e.args) >= 2 && e.args[2] isa Symbol && e.args[2] in funcNames
                e.args[2] = Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(e.args[2]))
            end
        end
        for a in e.args
            a isa Expr && push!(worklist, a)
        end
    end
    return expr
end

qualifyModelicaFunctions!(x, funcNames::OrderedSet{Symbol}) = x

"""
Return true when `x` is a zero numeric literal.
"""
_isZeroLiteral(x) = (x isa Integer || x isa AbstractFloat) && x == 0

"""
Unwrap `begin ... end` blocks that wrap a single expression (with optional
LineNumberNodes). `createResidualEquationsMTK` and `expToJuliaExpMTK` emit
variables inside `:block` wrappers; the additive flattener and leaf detector
must look through them.
"""
function _unwrapBlock(x)
    while x isa Expr && x.head === :block
        local content = nothing
        local extra = false
        for a in x.args
            a isa LineNumberNode && continue
            if content === nothing
                content = a
            else
                extra = true
                break
            end
        end
        (extra || content === nothing) && return x
        x = content
    end
    return x
end

"""
Return true when the Expr tree rooted at `x` contains a `der(...)` call.
"""
function _containsDerivativeCall(x::Expr)
    if x.head === :call && !isempty(x.args)
        x.args[1] === :der && return true
    end
    return any(_containsDerivativeCall, x.args)
end
_containsDerivativeCall(x) = false

"""
Return true when `x` is a single resolvable leaf: a plain Symbol (but not `:t`)
or a single-argument call `f(t)` (MTK variable-at-time encoding). Blocks that
wrap a single such expression also qualify.
"""
function _isSimpleLeafExpr(x)
    x = _unwrapBlock(x)
    x isa Symbol && return x !== :t
    if x isa Expr && x.head === :call && length(x.args) == 2
        return x.args[1] isa Symbol && x.args[2] === :t
    end
    return false
end

"""
Flatten a sum/difference expression into `(sign, term)` pairs, in place into `acc`.
Looks through `:block` wrappers so equations from `createResidualEquationsMTK`
(which emits each sub-expression inside `begin ... end`) flatten correctly.
"""
function _flattenAdditiveTerms!(acc::Vector{Tuple{Int,Any}}, expr, sign::Int = 1)
    expr = _unwrapBlock(expr)
    if expr isa Expr && expr.head === :call && !isempty(expr.args)
        fn = expr.args[1]
        if fn === :+ && length(expr.args) == 3
            _flattenAdditiveTerms!(acc, expr.args[2], sign)
            _flattenAdditiveTerms!(acc, expr.args[3], sign)
            return acc
        elseif fn === :- && length(expr.args) == 3
            _flattenAdditiveTerms!(acc, expr.args[2], sign)
            _flattenAdditiveTerms!(acc, expr.args[3], -sign)
            return acc
        elseif fn === :- && length(expr.args) == 2
            _flattenAdditiveTerms!(acc, expr.args[2], -sign)
            return acc
        end
    end
    push!(acc, (sign, expr))
    return acc
end

"""
Reconstruct a sum expression from `(sign, term)` pairs (left-associative).
Returns 0 for an empty list.
"""
function _buildSumExpr(terms::Vector{Tuple{Int,Any}})
    isempty(terms) && return 0
    acc = Any[]
    for (i, (sign, term)) in enumerate(terms)
        piece = sign == 1 ? term : Expr(:call, :-, term)
        if i == 1
            push!(acc, piece)
        else
            prev = pop!(acc)
            push!(acc, Expr(:call, :+, prev, piece))
        end
    end
    return only(acc)
end

"""
Convert `0 ~ expr` to `lhs ~ rhs` when `expr` has exactly one simple-leaf term
with unit coefficient that appears only once and contains no derivative.

Conservative: does nothing when the pattern does not match.
"""
function residualToExplicit(eq_expr::Expr)
    if !(eq_expr.head === :call && length(eq_expr.args) == 3 && eq_expr.args[1] === :~)
        return eq_expr
    end
    _isZeroLiteral(eq_expr.args[2]) || return eq_expr

    rhs = eq_expr.args[3]
    terms = Tuple{Int,Any}[]
    _flattenAdditiveTerms!(terms, rhs)

    idx = findfirst(terms) do (sign, term)
        abs(sign) == 1 && _isSimpleLeafExpr(term) && !_containsDerivativeCall(term)
    end
    idx === nothing && return eq_expr

    cand_sign, cand = terms[idx]
    rest = deleteat!(copy(terms), idx)
    rest_expr = _buildSumExpr(rest)

    if isempty(rest)
        return Expr(:call, :~, cand, 0)
    elseif cand_sign == 1
        # 0 ~ cand + rest  →  cand ~ -rest
        neg = rest_expr isa Integer ? -rest_expr : Expr(:call, :-, rest_expr)
        return Expr(:call, :~, cand, neg)
    else
        # 0 ~ -cand + rest  →  cand ~ rest
        return Expr(:call, :~, cand, rest_expr)
    end
end

"""
Expr-level replacement for `rewriteEquations`. Returns Vector{Expr}.
"""
function rewriteEquationsExprLevel(edeqs::Vector{Expr}; modelicaFuncNames::OrderedSet{Symbol} = OrderedSet{Symbol}())
    result = Expr[]
    for eq in edeqs
        rewritten = moveDerivativeToLHS(eq)
        rewritten = residualToExplicit(rewritten)
        renameDerToD!(rewritten)
        if !isempty(modelicaFuncNames)
            qualifyModelicaFunctions!(rewritten, modelicaFuncNames)
        end
        rewritten = wrapWithInvokelatest(rewritten)
        push!(result, rewritten)
    end
    return result
end

"""
    eliminateIfEqRelays(equations) -> (Vector{Expr}, Dict{Symbol,Symbol})

Remove pure relay equations `A ~ B` where both sides are simple leaf
variables. Returns the filtered equation list and an alias map
`A => canonical` so callers can drop the aliased symbols from variable lists.

Chain A → B → C is resolved to A → C before substitution.

`preferKeep` names a set of symbols that must survive as the canonical
representative of their alias component (e.g. discretes referenced in an
event condition, whose name the generated callback reads from the state
vector). When a component contains such a symbol it is forced to be the
root so it is never substituted away.

`paramSyms` names symbols that are parameters (no module-global symbolic
binding). A relay touching one is kept as an equation rather than
eliminated: collapsing a bound variable onto a parameter strands the
canonical name at module scope, and `structural_simplify` aliases the
variable to the parameter correctly on its own.
"""
function eliminateIfEqRelays(equations::Vector{Expr};
                             preferKeep::OrderedSet{Symbol} = OrderedSet{Symbol}(),
                             paramSyms::OrderedSet{Symbol} = OrderedSet{Symbol}())
    #= Union-find (not a plain map): a variable that is the leaf-alias LHS of two
       relays (`A~B`, `A~C`) means A≡B≡C; a `raw[k]=v` map overwrites and drops one
       side, disconnecting it from its definer. Union-find keeps the component intact. =#
    local parent = Dict{Symbol,Symbol}()
    local getroot = function (x::Symbol)
        haskey(parent, x) || (parent[x] = x)
        while parent[x] != x
            x = parent[x]
        end
        return x
    end
    local non_relay = Expr[]
    for eq in equations
        if _isIfEqTmpRelay(eq)
            local (k, v) = _ifEqRelayPair(eq)
            if k === nothing || v === nothing
                push!(non_relay, eq)
                continue
            end
            #= A relay onto a parameter has no module-global binding for the
               canonical name; keep the equation so structural_simplify aliases
               the variable to the parameter itself. =#
            if k in paramSyms || v in paramSyms
                push!(non_relay, eq)
                continue
            end
            local rk = getroot(k)
            local rv = getroot(v)
            rk == rv || (parent[rk] = rv)
        else
            push!(non_relay, eq)
        end
    end
    isempty(parent) && return (equations, Dict{Symbol,Symbol}())
    #= Re-root any component that contains a preferKeep symbol onto that symbol,
       so the alias map substitutes the other members toward it. =#
    if !isempty(preferKeep)
        local rootOf = Dict{Symbol,Symbol}()
        for k in keys(parent)
            local r = getroot(k)
            if k in preferKeep && !haskey(rootOf, r)
                rootOf[r] = k
            end
        end
        for (r, keep) in rootOf
            r == keep && continue
            parent[r] = keep
            parent[keep] = keep
        end
    end
    local alias = Dict{Symbol,Symbol}()
    for k in keys(parent)
        local r = getroot(k)
        k == r || (alias[k] = r)
    end
    resolved = [_substSyms(eq, alias) for eq in non_relay]
    return (resolved, alias)
end

# Extract symbol from any simple leaf (plain Symbol or sym(t) call).
function _simpleLeafSymbol(x)
    x = _unwrapBlock(x)
    x isa Symbol && return x
    if x isa Expr && x.head === :call && length(x.args) == 2 && x.args[2] === :t
        x.args[1] isa Symbol && return x.args[1]
    end
    return nothing
end

# Detect pure leaf aliases in both post-rewrite (`A(t) ~ B(t)`) and
# pre-rewrite (`0 ~ A(t) - B(t)`) forms.  MTK's alias_elimination cannot
# subtract two SymReal leaf variables, so these are eliminated before MTK.
function _isIfEqTmpRelay(eq)
    eq isa Expr || return false
    eq.head === :call || return false
    length(eq.args) == 3 || return false
    eq.args[1] === :~ || return false
    local lhs2 = _unwrapBlock(eq.args[2])
    local rhs2 = _unwrapBlock(eq.args[3])
    # Post-rewrite: A(t) ~ B(t)
    if _isSimpleLeafExpr(lhs2) && _isSimpleLeafExpr(rhs2)
        return true
    end
    # Pre-rewrite: 0 ~ A - B
    _isZeroLiteral(lhs2) || return false
    rhs2 isa Expr || return false
    rhs2.head === :call || return false
    rhs2.args[1] === :- && length(rhs2.args) == 3 || return false
    local a = _unwrapBlock(rhs2.args[2])
    local b = _unwrapBlock(rhs2.args[3])
    return _isSimpleLeafExpr(a) && _isSimpleLeafExpr(b)
end

# Extract (alias_sym, canonical_sym) from a relay.
# alias_sym → canonical_sym (alias_sym gets substituted away).
function _ifEqRelayPair(eq)
    local lhs = _unwrapBlock(eq.args[2])
    local rhs = _unwrapBlock(eq.args[3])
    if _isZeroLiteral(lhs)
        # Pre-rewrite: 0 ~ A - B  →  A → B
        local a = _unwrapBlock(rhs.args[2])
        local b = _unwrapBlock(rhs.args[3])
        return (_simpleLeafSymbol(a), _simpleLeafSymbol(b))
    end
    # Post-rewrite: A(t) ~ B(t)  →  A → B
    return (_simpleLeafSymbol(lhs), _simpleLeafSymbol(rhs))
end

_substSyms(x, _) = x
_substSyms(sym::Symbol, alias::Dict{Symbol,Symbol}) = get(alias, sym, sym)
function _substSyms(ex::Expr, alias::Dict{Symbol,Symbol})
    #= Affect bodies read NamedTuple fields through QuoteNodes; rewrite those in
       step with the kw keys or an aliased read misses its observed entry. =#
    if ex.head === :. && length(ex.args) == 2 &&
       (ex.args[1] === :observed || ex.args[1] === :modified) &&
       ex.args[2] isa QuoteNode && ex.args[2].value isa Symbol
        local quoted = ex.args[2].value
        return Expr(:., ex.args[1], QuoteNode(get(alias, quoted, quoted)))
    end
    Expr(ex.head, Any[_substSyms(a, alias) for a in ex.args]...)
end
