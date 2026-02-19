#=
Expr-level equation rewriting. Operates directly on Julia Expr trees
instead of the old Symbolics round-trip.
=#

"""
Find `:(der(x))` in an Expr tree. Returns the variable symbol or `nothing`.
"""
function findDerivativeVar(expr::Expr)
    if expr.head == :call && length(expr.args) >= 2 && expr.args[1] === :der
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
    if expr.head == :call && length(expr.args) >= 2 && expr.args[1] === :der && expr.args[2] == var
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
function qualifyModelicaFunctions!(expr::Expr, funcNames::Set{Symbol})
    if expr.head == :call && length(expr.args) >= 1
        if expr.args[1] isa Symbol && expr.args[1] in funcNames
            #= Direct call: funcName(args...) =#
            expr.args[1] = Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(expr.args[1]))
        elseif expr.args[1] == :(Base.invokelatest) && length(expr.args) >= 2 && expr.args[2] isa Symbol && expr.args[2] in funcNames
            #= invokelatest call: Base.invokelatest(funcName, args...) =#
            expr.args[2] = Expr(:., Expr(:., :OMBackend, QuoteNode(:CodeGeneration)), QuoteNode(expr.args[2]))
        end
    end
    for a in expr.args
        qualifyModelicaFunctions!(a, funcNames)
    end
    return expr
end

qualifyModelicaFunctions!(x, funcNames::Set{Symbol}) = x

"""
Expr-level replacement for `rewriteEquations`. Returns Vector{Expr}.
"""
function rewriteEquationsExprLevel(edeqs::Vector{Expr}; modelicaFuncNames::Set{Symbol} = Set{Symbol}())
    result = Expr[]
    for eq in edeqs
        rewritten = moveDerivativeToLHS(eq)
        renameDerToD!(rewritten)
        if !isempty(modelicaFuncNames)
            qualifyModelicaFunctions!(rewritten, modelicaFuncNames)
        end
        rewritten = wrapWithInvokelatest(rewritten)
        push!(result, rewritten)
    end
    return result
end
