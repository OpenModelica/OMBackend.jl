module AlgorithmicCodeGeneration
import ..SimulationCode
import ..Absyn
import ..CodeGeneration
import DAE
import OMRuntimeExternalC
using MetaModelica
include("modelicaBuiltins.jl")
include("algorithmic.jl")
end #AlgorithmicCodeGeneration
