

#= Parameters kept as parameters instead of being folded into the equations at
   compile time, so a compiled model can be simulated again with other values
   (withTunableParameters). Canonical names. Read by the lowering
   (Causalize.resolveCrefBindings!), the SimCode passes (every compile-time
   evaluation of a parameter goes through _boundParameterExpression, so a
   parameter bound to a tunable one is not constant either) and the code
   generator. =#
const TUNABLE_PARAMETERS = Base.ScopedValues.ScopedValue(Set{String}())
#= An array parameter is tunable as a whole: its scalarized elements match
   the array's name, the trailing subscripts removed one at a time
   (`nn_W1[1][2]` → `nn_W1`; in an array of components `nn[1]_W1[2]` →
   `nn[1]_W1`, never `nn`). =#
isTunableParameter(name::AbstractString)::Bool = isTunableParameter(name, TUNABLE_PARAMETERS[])
function isTunableParameter(name::AbstractString, tun::AbstractSet{String})::Bool
  isempty(tun) && return false
  name in tun && return true
  local s = SubString(name)
  while endswith(s, ']')
    local b = findlast('[', s)
    (b === nothing || b == 1) && return false
    s = SubString(s, 1, b - 1)
    s in tun && return true
  end
  return false
end

#= `A[2,1]` → `A[2][1]`, the form OM.jl gives scalarized array elements. =#
_elementSubscripts(name::AbstractString)::String =
  replace(name, r"\[\s*\d+(\s*,\s*\d+)+\s*\]" => m -> join("[" * strip(i) * "]" for i in split(m[2:end-1], ',')))
