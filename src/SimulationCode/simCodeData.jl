#=
TODO:
Remove DAE structure from this file s.t simcode can stand alone.
=#

"""
 Category of a simulation variable
"""
abstract type SimVarType end

"""
State variable
"""
struct  STATE <: SimVarType
end

"""
 State Derivative
"""
struct  STATE_DERIVATIVE <: SimVarType
  varName::String
end

"""
Algebraic variable.
Has a sort idx for backend algorithms,
this is different from the final idx used in code generation.
The purpose of the sort index is for initial structural analysis before code generation.
"""
struct ALG_VARIABLE <: SimVarType
  sortIdx::Int
end

"""
Input variable
"""
struct  INPUT <: SimVarType end

"
  A special state variable, used for dynamic overconstrained connectors.
  In pratice this variable is treated as state.
"
struct OCC_VARIABLE <: SimVarType end

struct STRING <: SimVarType end

"""
  A Data structure type.
  Currently this type represents complex datastrutures such as matrices
  or pointers to data structures in memory.
"""
struct DATA_STRUCTURE <: SimVarType
  bindExp::Option{DAE.Exp}
end

"""
An array with N dimensions.
Contains optional binding expression for compile-time evaluation of subscripted access.
"""
struct ARRAY <: SimVarType
  dimensions::Vector{Int}
  bindExp::Option{DAE.Exp}
end

#= Backwards-compatible constructor =#
ARRAY(dimensions::Vector{Int}) = ARRAY(dimensions, NONE())

"""
Parameter variable
"""
struct PARAMETER <: SimVarType
  bindExp::Option{DAE.Exp}
end

"""
Array parameter variable. Has dimensions and an optional binding expression.
Distinct from ARRAY (which is a state array) to ensure correct MTK code generation.
"""
struct ARRAY_PARAMETER <: SimVarType
  dimensions::Vector{Int}
  bindExp::Option{DAE.Exp}
end

"""
  Discrete Variable
"""
struct DISCRETE <: SimVarType end

"""
Abstract type for a simulation variable
"""
abstract type SimVar end

const ELSE_BRANCH = -1

"""
Variable data type used for code generation
"""
struct SIMVAR{T0 <: String, T1 <: Option{Int}, T2 <: SimVarType, T3 <: Option{<:DAE.VariableAttributes}} <: SimVar
  "Readable name of variable"
  name::T0
  "Index of variable, 0 based, type based"
  index::T1
  "Kind of variable, one of SimulationCode.SimVarType"
  varKind::T2
  "Variable attributes. Same as in DAE"
  attributes::T3
end

""" Abstract type for different variants of simulation code """
abstract type SimCode end

"""
  Abstract type for control flow constructs for simulation code
"""
abstract type Construct end


include("simCodeAlgorithmic.jl")


"""
  Represents a branch in simulation code.
  Contains a single condition, a set of inner equations and a set of possible targets.
  Since each branch potentially contains a set of equations information exist so that the code in each branch can be
  matched (Similar to the larger system )
"""
struct BRANCH{T1 <: DAE.Exp,
              T2 <: Vector{BDAE.RESIDUAL_EQUATION},
              T3 <: Int, #= Integer code Each branch has one target (next) The ID of one branch is target - 1=#
              T4 <: Bool,
              T5 <: Vector{Int},
              T6 <: Graphs.AbstractGraph,
              T7 <: Vector{Vector{Int}},
              T8 <: AbstractDict{String, Tuple{Integer, SimVar}}} <: Construct

  condition::T1
  residualEquations::T2
  identifier::T3 #= A value of -1 indicates that this branch is an else branch =#
  targets::T3
  isSingular::T4
  matchOrder::T5
  equationGraph::T6
  sccs::T7
  stringToSimVarHT::T8
end

"""
  A representation of a simcode IF Equation.
  Similar to the main simcode module it contains information to make construct a causal representation easier.
"""
struct IF_EQUATION{Branches <: Vector{BRANCH}} <: Construct
  branches::Branches
end

abstract type StructuralTransition end

"""
  An explicit transition from one system state to another.
"""
struct EXPLICIT_STRUCTURAL_TRANSISTION <: StructuralTransition
  structuralTransition::BDAE.STRUCTURAL_TRANSISTION
end

"""
  Represents a structural transistion where a change of a parameter in the metamodel results in a new model.
"""
struct IMPLICIT_STRUCTURAL_TRANSISTION <: StructuralTransition
  structuralWhenEquation::BDAE.STRUCTURAL_WHEN_EQUATION
end

struct DYNAMIC_OVERCONSTRAINED_CONNECTOR_EQUATION <: StructuralTransition
  structuralDOCC_equation::BDAE.STRUCTURAL_IF_EQUATION
end

"""
  Root data structure containing information required for code generation to
  generate simulation code for a Modelica model.

The topmost model of a system consisting of several sub models lacks:
  - matchorder
  - The graph of the equations
  - Descriptions of the SCC
  - Information if it is singular or not.
This information is instead contained for each of the structural submodels, where one model is active at the time.
"""
struct SIM_CODE{T0<:String,
                T1<:AbstractDict{String, Tuple{Integer, SimVar}},
                T2<:Vector{BDAE.RESIDUAL_EQUATION},
                T22,
                T4<:Vector{BDAE.WHEN_EQUATION},
                #=
                  If equations are represented via a vector of possible branches in which the code can operate.
                  Similar to basic blocks
                =#
                T5<:Vector{IF_EQUATION},
                T6<:Bool,
                T7<:Vector{Int},
                T8<:Graphs.AbstractGraph,
                T9<:Vector,
                T10 <: Vector{StructuralTransition},
                T11 <: Vector,
                T12 <: Vector{String},
                T13 <: String} <: SimCode
  name::T0
  "Mapping of names to the corresponding variable"
  stringToSimVarHT::T1
  "Different equations stored within simulation code"
  residualEquations::T2
  "The Initial equations"
  initialEquations::T22
  "When equations"
  whenEquations::T4
  "If Equations (Simulation code branches). Each branch contains a condition a set of residual equations and a set of targets"
  ifEquations::T5
  "True if the system that we are solving is singular"
  isSingular::T6
  "
   The match order:
   Result of assign array, e.g array(j) = equation_i
  "
  matchOrder::T7
    "
    The merged graph. E.g digraph constructed from matching info.
    The indicies are the same as above and they are shared.
    If the system is singular tearing is needed.
   "
  equationGraph::T8
  " The reverse topological sort of the equation-graph "
  stronglyConnectedComponents::T9
  "Contains all structural transistions"
  structuralTransitions::T10
  "Structural submodels"
  subModels::T11
  " Variables that different submodels have in common"
  sharedVariables::T12
  "Top variables"
  topVariables::T12
  "Shared equations. These are equations shared between structural submodels. These are required to be residuals."
  sharedEquations::Vector{BDAE.Equation}
  "Initial model"
  activeModel::T13
  "The MetaModel. That is a reference from the model to a higher order representation of the model itself."
  metaModel::Option
  "An alternate flat model. Used by structural if equations to add or remove connector statements affecting the virtual connection graph."
  flatModel::Option
  "Irreductable variables. That is the names of variables that are involved in events such as discrete variables"
  irreductableVariables::T12
  "Modelica functions"
  functions::Vector{ModelicaFunction}
  "Specify if an external Modelica runtime is needed or not. Used for build in functions"
  externalRuntime::Bool
end
