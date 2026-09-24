

#= Parameters kept as parameters instead of being folded into the equations at
   compile time, so a compiled model can be simulated again with other values
   (withTunableParameters). Canonical names. Read by the lowering
   (Causalize.resolveCrefBindings!), the SimCode passes (every compile-time
   evaluation of a parameter goes through _boundParameterExpression, so a
   parameter bound to a tunable one is not constant either) and the code
   generator. =#
const TUNABLE_PARAMETERS = Base.ScopedValues.ScopedValue(Set{String}())
isTunableParameter(name::AbstractString)::Bool = name in TUNABLE_PARAMETERS[]
