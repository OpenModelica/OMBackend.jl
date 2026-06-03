#= Shared mock/fixture utilities for OMBackend unit tests.

   Provides minimal hand-built SimulationCode constructs so individual backend
   functions (codegen passes, traversals, demotion planning) can be unit-tested
   without a full frontend+flatten translate. Isolated in a module so the
   builders do not clash with test-file-local helpers. =#
module BackendTestMocks

import OMBackend
const SC = OMBackend.SimulationCode
const CG = OMBackend.CodeGeneration
const DAEx = OMBackend.DAE
import MetaModelica
const NONE = MetaModelica.NONE
const nil = MetaModelica.nil

export mockSimCode, simCref, daeCref, daeCrefExp, T_R

const T_R = DAEx.T_REAL(nil)

simCref(name)    = SC.EXP_CREF(SC.SimCref(Symbol(name)), SC.TYPE_REAL())
daeCref(name)    = DAEx.CREF_IDENT(name, T_R, nil)
daeCrefExp(name) = DAEx.CREF(daeCref(name), T_R)

"""
    mockSimCode(ht) -> SC.SIM_CODE

Minimal SIM_CODE with an empty equation/when/if system and the given
`stringToSimVarHT`. Sufficient for unit-testing passes that take their
equations as a separate argument (e.g. `planDemotions`) and only consult
the simcode for when-assigned / cyclic-SCC / duplicate-residual accounting.
"""
function mockSimCode(ht = Dict{String, Tuple{Integer, SC.SimVar}}())
  return SC.SIM_CODE("mock", ht,
    SC.RESIDUAL_EQUATION[], SC.Equation[], SC.WHEN_EQUATION[], SC.IF_EQUATION[],
    false, Int[], SC.Graphs.SimpleDiGraph(0), [], SC.StructuralTransition[], [],
    String[], String[], SC.Equation[], "mock", NONE(), NONE(), String[],
    SC.ModelicaFunction[], false, SC.RESIDUAL_EQUATION[], String[], SC.AliasEntry[],
    nothing, SC.INITIAL_ALGORITHM[])
end

end # module BackendTestMocks
