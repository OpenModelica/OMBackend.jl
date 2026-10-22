#= Shared fixtures for OMBackend unit tests: a simulation code without equations and the
   SimCode values of variables, so single backend functions (code generation passes,
   traversals, demotion planning) are tested without a translation. In a module, so the
   builders do not clash with a test file's own helpers. =#
module BackendTestMocks

import OMBackend
const SC = OMBackend.SimCode

export mockSimCode, simCref, simVar

"A reference to the variable `name` of type `ty`."
simCref(name::String, ty::SC.SType = SC.TYPE_REAL())::SC.EXP_CREF = SC.EXP_CREF(SC.SimCref(Symbol(name)), ty)

"The variable `name` of kind `kind` and element type `ty`, without attributes."
simVar(name::String, kind::SC.SimVarType, ty::SC.SType = SC.TYPE_REAL())::SC.SIMVAR =
  SC.SIMVAR(name, nothing, kind, SC.NO_ATTRIBUTES, ty)

"""
    mockSimCode(variables = SIMVAR[]; name = "mock") -> SC.SIM_CODE

A simulation code of the variables `variables` (indexed as the lowering indexes them: a
state's derivative too) without equations, enough for the passes that take their equations
as an argument and only look variables up.
"""
function mockSimCode(variables::Vector{SC.SIMVAR} = SC.SIMVAR[]; name::String = "mock")::SC.SIM_CODE
  return SC.SIM_CODE(name, SC.createIndices(variables), SC.RESIDUAL_EQUATION[], SC.Equation[], SC.WHEN_EQUATION[],
                     SC.IF_EQUATION[], SC.StructuralTransition[],
                     SC.SIM_CODE[], String[], String[], SC.Equation[], name, nothing, String[], SC.ModelicaFunction[], false,
                     SC.RESIDUAL_EQUATION[], String[], SC.AliasEntry[], nothing, SC.INITIAL_ALGORITHM[], SC.ASSERT_EQUATION[],
                     String[], false, String[])
end

end # module BackendTestMocks
