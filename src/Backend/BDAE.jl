#=
* This file is part of OpenModelica.
*
* Copyright (c) 1998-CurrentYear, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF GPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.2.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GPL VERSION 3,
* ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the Open Source Modelica
* Consortium (OSMC) Public License (OSMC-PL) are obtained
* from OSMC, either from the above address,
* from the URLs: http:www.ida.liu.se/projects/OpenModelica or
* http:www.openmodelica.org, and in the OpenModelica distribution.
* GNU version 3 is obtained from: http:www.gnu.org/copyleft/gpl.html.
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of  MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#

#= Author: John Tinnerholm, partially automatically translated =#

"""
  THE LOWERED DAE consist of variables and equations. The variables are split into
  two lists, one for unknown variables states and algebraic and one for known variables
  constants and parameters.
  The equations are also split into two lists, one with simple equations, a=b, a-b=0, etc., that
  are removed from the set of equations to speed up calculations.
"""
module BDAE

import OMFrontend

using MetaModelica
using ExportAll

#= Predeclaration of mutable recursive types =#
@UniontypeDecl EqSystem
@UniontypeDecl SubClock
@UniontypeDecl BaseClockPartitionKind
@UniontypeDecl Shared
@UniontypeDecl InlineData
@UniontypeDecl BasePartition
@UniontypeDecl SubPartition
@UniontypeDecl PartitionsInfo
@UniontypeDecl ExtraInfo
@UniontypeDecl BDAEType
@UniontypeDecl DataReconciliationData
@UniontypeDecl Variables
@UniontypeDecl CrefIndex
@UniontypeDecl Var
@UniontypeDecl VarKind
@UniontypeDecl TearingSelect
@UniontypeDecl EquationKind
@UniontypeDecl EvaluationStages
@UniontypeDecl EquationAttributes
@UniontypeDecl Equation
@UniontypeDecl WhenEquation
@UniontypeDecl WhenOperator
@UniontypeDecl ExternalObjectClass

@UniontypeDecl IndexReduction
@UniontypeDecl EquationConstraints
@UniontypeDecl StateOrder
@UniontypeDecl StrongComponent
@UniontypeDecl TearingSet
@UniontypeDecl InnerEquation
@UniontypeDecl StateSet
@UniontypeDecl EventInfo
@UniontypeDecl ZeroCrossingSet
@UniontypeDecl ZeroCrossing
@UniontypeDecl TimeEvent
@UniontypeDecl Solvability
@UniontypeDecl IndexType
@UniontypeDecl JacobianType
@UniontypeDecl Jacobian
@UniontypeDecl DifferentiateInputData
@UniontypeDecl DifferentiationType
@UniontypeDecl CompInfo
@UniontypeDecl BDAEModeData

import Absyn
import DAE
import DoubleEnded
import SCode

#=  AdjacencyMatrixes =#
const AdjacencyMatrixElementEntry = Integer
const AdjacencyMatrixElement = List
const AdjacencyMatrix = Array
const AdjacencyMatrixT = AdjacencyMatrix
const AdjacencyMatrixMapping = Tuple
const AdjacencyMatrixElementEnhancedEntry = Tuple
const AdjacencyMatrixElementEnhanced = List
const AdjacencyMatrixEnhanced = Array
const AdjacencyMatrixTEnhanced = AdjacencyMatrixEnhanced
const ExternalObjectClasses = List  #= classes of external objects stored in list =#
const StateSets = List  #= List of StateSets =#
const StrongComponents = List  #= Order of the equations the have to be solved =#
const SymbolicJacobians = List
const SymbolicJacobian = Tuple
const SparsePatternCref = Tuple
const SparsePatternCrefs = List
const SparsePattern = Tuple
const emptySparsePattern = (nil, nil, (nil, nil), 0)::SparsePattern
const SparseColoring = List
const LinearIntegerJacobianRow = List
const LinearIntegerJacobianRhs = Array
const LinearIntegerJacobianIndices = Array
const LinearIntegerJacobian = Tuple
const InnerEquations = List
const Constraints = List

#= Component Reference Index =#
struct CREFINDEX
  cref::DAE.ComponentRef
  index::Int
end

mutable struct VAR <: Var
  varName::DAE.ComponentRef #= variable name =#
  varKind::VarKind #= kind of variable =#
  varDirection::DAE.VarDirection #= input, output or bidirectional =#
  varType::DAE.Type #= built-in type or enumeration =#
  bindExp::Option #= Binding expression e.g. for parameters =#
  arryDim::List #= array dimensions of non-expanded var =#
  source::DAE.ElementSource #= origin of variable =#
  values::Option #= values on built-in attributes =#
  tearingSelectOption::Option #= value for TearingSelect =#
  connectorType::DAE.ConnectorType #= flow, stream, unspecified or not connector. =#
  unreplaceable::Bool #= indicates if it is allowed to replace this variable =#
end

#=Simplify construction of var=#
VAR(varName,varKind,varType) =
  let
    VAR(varName, varKind, DAE.BIDIR(), varType, NONE(), nil, DAE.emptyElementSource,
        NONE(), NONE(), DAE.NON_CONNECTOR(), true #= Unreplaceable =#)
  end

"""
An independent system of equations (and their corresponding variables)
Structural submodels gets converted into these.
"""
mutable struct EQSYSTEM
  name::String
  orderedVars #= ordered Variables, only states and alg. vars =#::Vector{VAR}
  orderedEqs #= ordered Equations =#::Vector{Equation}
  simpleEquations::Vector{Equation} #= List of simple equations=#
  initialEqs::Vector{Equation}
end

"""
  Data that is shared between equation systems.
  The flat model is a reference to the flat model being translated.
  the DOCC_equations are the equations used by the dynamic overconstrained
  connector option.
"""
struct SHARED
  globalKnownVars::Vector{VAR}
  localKnownVars::Vector{VAR}
  metaModel::Option
  #= The flat model of the system itself =#
  flatModel::Option
  #= Dynamic if equations =#
  DOCC_equations::Vector{Equation}
end

""" THE LOWERED DAE consist of variables and equations. The variables are split into
two lists, one for unknown variables states and algebraic and one for known variables
constants and parameters.
The equations are also split into two lists, one with simple equations, a=b, a-b=0, etc., that
are removed from the set of equations to speed up calculations. """
mutable struct BACKEND_DAE
  name::String
  eqs::Vector{EQSYSTEM}
  shared::SHARED
end

#=Clock removed for now=#
@Uniontype SubClock begin
  @Record SUBCLOCK begin
  end
end

const DEFAULT_SUBCLOCK = "TODO. NOT SUPPORTED"

@Uniontype BaseClockPartitionKind begin
  @Record UNKNOWN_PARTITION begin
  end
end

@Uniontype BasePartition begin
  @Record BASE_PARTITION begin
    clock::DAE.ClockKind
    nSubClocks::Integer
  end
end

@Uniontype SubPartition begin
  @Record SUB_PARTITION begin
    clock::SubClock
    holdEvents::Bool
    prevVars::List{DAE.ComponentRef}
  end
end

@Uniontype PartitionsInfo begin
  @Record PARTITIONS_INFO begin
    basePartitions::Array{BasePartition}
    subPartitions::Array{SubPartition}
  end
end

#= extra information that we should send around with the DAE =#
@Uniontype ExtraInfo begin
  @Record EXTRA_INFO begin
    description #= the model description string =#::String
    fileNamePrefix #= the model name to be used in the dumps =#::String
  end
end

@Uniontype DataReconciliationData begin
  @Record DATA_RECON begin
    symbolicJacobian #= SET_S w.r.t ... =#::Jacobian
    setcVars #= setc solved vars =#::Variables
    datareconinputs::Variables
    #=  ... maybe more DATA for the code generation
    =#
  end
end

#= variable kind =#
@Uniontype VarKind begin
  @Record VARIABLE begin
  end

  @Record STATE begin
    index #= how often this states was differentiated =#::Integer
    derName #= the name of the derivative =#::Option{DAE.ComponentRef}
    natural #= false if it was forced by StateSelect.always or StateSelect.prefer or generated by index reduction =#::Bool
  end

  @Record STATE_DER begin
  end

  @Record DUMMY_DER begin
  end

  @Record DUMMY_STATE begin
  end

  @Record CLOCKED_STATE begin
    previousName #= the name of the previous variable =#::DAE.ComponentRef
    isStartFixed #= is fixed at first clock tick =#::Bool
  end

  @Record DISCRETE begin
  end

  @Record PARAM begin
  end

  @Record CONST begin
  end

  @Record EXTOBJ begin
    fullClassName::Absyn.Path
  end

  @Record JAC_VAR begin
  end

  @Record JAC_DIFF_VAR begin
  end

  @Record SEED_VAR begin
  end

  @Record OPT_CONSTR begin
  end

  @Record OPT_FCONSTR begin
  end

  @Record OPT_INPUT_WITH_DER begin
  end

  @Record OPT_INPUT_DER begin
  end

  @Record OPT_TGRID begin
  end

  @Record OPT_LOOP_INPUT begin
    replaceExp::DAE.ComponentRef
  end

  @Record ALG_STATE begin
  end

  @Record ALG_STATE_OLD begin
  end

  @Record DAE_RESIDUAL_VAR begin
  end

  @Record DAE_AUX_VAR begin
  end

  @Record LOOP_ITERATION begin
  end

  @Record LOOP_SOLVED begin
  end
end

@Uniontype TearingSelect begin
  @Record NEVER begin

  end

  @Record AVOID begin

  end

  @Record DEFAULT begin

  end

  @Record PREFER begin

  end

  @Record ALWAYS begin

  end
end

STATE() = let
  STATE(0, NONE(), false)
end

const WHENCLK_PRREFIX = "whenclk"::String

#= equation kind =#
@Uniontype EquationKind begin
  @Record BINDING_EQUATION begin

  end

  @Record DYNAMIC_EQUATION begin

  end

  @Record INITIAL_EQUATION begin

  end

  @Record CLOCKED_EQUATION begin

    clk::Integer
  end

  @Record DISCRETE_EQUATION begin

  end

  @Record AUX_EQUATION begin

  end

  @Record UNKNOWN_EQUATION_KIND begin

  end
end

#= evaluation stages =#
@Uniontype EvaluationStages begin
  @Record EVALUATION_STAGES begin

    dynamicEval::Bool
    algebraicEval::Bool
    zerocrossEval::Bool
    discreteEval::Bool
  end
end

const defaultEvalStages = EVALUATION_STAGES(false, false, false, false)::EvaluationStages

@Uniontype EquationAttributes begin
  @Record EQUATION_ATTRIBUTES begin
    differentiated #= true if the equation was differentiated, and should not be differentiated again to avoid equal equations =#::Bool
    kind::EquationKind
    evalStages::EvaluationStages
  end
  @Record NO_ATTRIBUTES begin
  end
end

const EQ_ATTR_DEFAULT_DYNAMIC = EQUATION_ATTRIBUTES(false, DYNAMIC_EQUATION(), defaultEvalStages)::EquationAttributes
const EQ_ATTR_DEFAULT_BINDING = EQUATION_ATTRIBUTES(false, BINDING_EQUATION(), defaultEvalStages)::EquationAttributes
const EQ_ATTR_DEFAULT_INITIAL = EQUATION_ATTRIBUTES(false, INITIAL_EQUATION(), defaultEvalStages)::EquationAttributes
const EQ_ATTR_DEFAULT_DISCRETE = EQUATION_ATTRIBUTES(false, DISCRETE_EQUATION(), defaultEvalStages)::EquationAttributes
const EQ_ATTR_DEFAULT_AUX = EQUATION_ATTRIBUTES(false, AUX_EQUATION(), defaultEvalStages)::EquationAttributes
const EQ_ATTR_DEFAULT_UNKNOWN = EQUATION_ATTRIBUTES(false, UNKNOWN_EQUATION_KIND(), defaultEvalStages)::EquationAttributes

@Uniontype Equation begin
  @Record EQUATION begin
    lhs
    rhs
    source
    attributes::EquationAttributes
  end

  @Record ARRAY_EQUATION begin
    dimSize #= dimension sizes =#::Vector{Int64}
    left #= lhs =#::DAE.Exp
    right #= rhs =#::DAE.Exp
    source #= origin of equation =#::DAE.ElementSource
    attr::EquationAttributes
    recordSize #= NONE() if not a record =#::Option{Integer}
  end

  @Record SOLVED_EQUATION begin
    componentRef::DAE.ComponentRef
    exp::DAE.Exp
    source #= origin of equation =#::DAE.ElementSource
    attr::EquationAttributes
  end

  @Record RESIDUAL_EQUATION begin
    exp #= not present from FrontEnd =#
    source #= origin of equation =#
    attr
  end

  @Record ALGORITHM begin
    size #= size of equation =#::Integer
    alg::DAE.Algorithm
    source #= origin of algorithm =#::DAE.ElementSource
    expand #= this algorithm was translated from an equation. we should not expand array crefs! =#::DAE.Expand
    attr::EquationAttributes
  end

  @Record WHEN_EQUATION begin
    size #= size of equation =#::Integer
    whenEquation::WhenEquation
    source #= origin of equation =#::DAE.ElementSource
    attr#::EquationAttributes
  end
  #= Similar to a when equation but might trigger a structural change=#
  @Record STRUCTURAL_WHEN_EQUATION begin
    size #= size of equation =#::Integer
    whenEquation::WhenEquation
    source #= origin of equation =#::DAE.ElementSource
    attr::EquationAttributes
  end

  @Record COMPLEX_EQUATION begin
    size #= size of equation =#::Integer
    left #= lhs =#::DAE.Exp
    right #= rhs =#::DAE.Exp
    source #= origin of equation =#::DAE.ElementSource
    attr::EquationAttributes
  end
  #= If equation. Each condition correspond to one branch has it's own conditions =#
  @Record IF_EQUATION begin
    conditions #= Condition =#::List{DAE.Exp}
    eqnstrue #= Equations of true branch =#::List{List{Equation}}
    eqnsfalse #= Equations of false branch =#::List{Equation}
    source #= origin of equation =#::DAE.ElementSource
    attr::EquationAttributes
  end

  @Record FOR_EQUATION begin
    iter #= the iterator variable =#::DAE.Exp
    start #= start of iteration =#::DAE.Exp
    stop #= end of iteration =#::DAE.Exp
    body #= iterated equation =#::Equation
    source #= origin of equation =#::DAE.ElementSource
    attr::EquationAttributes
  end

  @Record DUMMY_EQUATION begin
  end

  @Record ASSERT_EQUATION begin
    condition::DAE.Exp
    message::DAE.Exp
    level::DAE.Exp
    source::DAE.ElementSource
  end

  @Record INITIAL_STRUCTURAL_STATE begin
    initialState::String
  end

  #= Represents a Connections.branch call =#
  @Record BRANCH begin
    ar::DAE.Exp
    br::DAE.Exp
  end

  #=
  An if equation affecting the structure.
  used as part of DOCC.
  =#
  @Record STRUCTURAL_IF_EQUATION begin
    ifEquation::OMFrontend.Frontend.EQUATION_IF
  end

  #=
    Structural transistion has three parts
    From the condition a structural callback is generated
  =#
  @Record STRUCTURAL_TRANSISTION begin
    fromState::String
    toState::String
    transistionCondition::DAE.Exp
  end
end

@Uniontype WhenEquation begin
  @Record WHEN_STMTS begin
    condition #= the when-condition =#::DAE.Exp
    whenStmtLst::List{WhenOperator}
    elsewhenPart #= elsewhen equation with the same cref on the left hand side. =##::Option{WHEN_STMTS}
  end
end

@Uniontype WhenOperator begin
  @Record ASSIGN begin
    left #= left hand side of equation =#::DAE.Exp
    right #= right hand side of equation =#::DAE.Exp
    source #= origin of equation =#::DAE.ElementSource
  end

  @Record REINIT begin
    stateVar #= State variable to reinit =#::DAE.CREF #=I changed this to cref to unify exp traversal -johti17 =#
    value #= Value after reinit =#::DAE.Exp
    source #= origin of equation =#::DAE.ElementSource
  end

  @Record ASSERT begin
    condition::DAE.Exp
    message::DAE.Exp
    level::DAE.Exp
    source #= the origin of the component/equation/algorithm =#::DAE.ElementSource
  end

  @Record TERMINATE begin
    message::DAE.Exp
    source #= the origin of the component/equation/algorithm =#::DAE.ElementSource
  end

  @Record NORETCALL begin
    exp::DAE.Exp
    source #= the origin of the component/equation/algorithm =#::DAE.ElementSource
  end
  #= I'd argue it only make sense for this construct to occur in when equations =#
  @Record RECOMPILATION begin
    componentToChange::DAE.CREF
    newValue::DAE.Exp
  end
end

#= class of external objects =#
@Uniontype ExternalObjectClass begin
  @Record EXTOBJCLASS begin

    path #= className of external object =#::Absyn.Path
    source #= origin of equation =#::DAE.ElementSource
  end
end

@Uniontype EquationConstraints begin
  @Record ALLOW_UNDERCONSTRAINED begin
  end

  @Record EXACT begin
  end
end

const MatchingOptions = Tuple
#= StateOrder,ConstraintEqns,Eqn->EqnsIndxes,EqnIndex->Eqns,NrOfEqnsbeforeIndexReduction =#
const StructurallySingularSystemHandlerArg = Tuple
const ConstraintEquations = Array

@Uniontype StateOrder begin
  @Record STATEORDER begin
    hashTable #= x -> dx =#
    invHashTable #= dx -> {x,y,z} =#
  end

  @Record NOSTATEORDER begin
  end
end

@Uniontype StrongComponent begin
  @Record SINGLEEQUATION begin
    eqn::Integer
    var::Integer
  end

  @Record EQUATIONSYSTEM begin
    eqns::List{Integer}
    vars #= be careful with states, this are solved for der(x) =#::List{Integer}
    jac::Jacobian
    jacType::JacobianType
    mixedSystem #= true for system that discrete dependencies to the iteration variables =#::Bool
  end

  @Record SINGLEARRAY begin

    eqn::Integer
    vars #= be careful with states, this are solved for der(x) =#::List{Integer}
  end

  @Record SINGLEALGORITHM begin

    eqn::Integer
    vars #= be careful with states, this are solved for der(x) =#::List{Integer}
  end

  @Record SINGLECOMPLEXEQUATION begin

    eqn::Integer
    vars #= be careful with states, this are solved for der(x) =#::List{Integer}
  end

  @Record SINGLEWHENEQUATION begin

    eqn::Integer
    vars #= be careful with states, this are solved for der(x) =#::List{Integer}
  end

  @Record SINGLEIFEQUATION begin

    eqn::Integer
    vars #= be careful with states, this are solved for der(x) =#::List{Integer}
  end

  @Record TORNSYSTEM begin

    strictTearingSet::TearingSet
    casualTearingSet::Option{TearingSet}
    linear::Bool
    mixedSystem #= true for system that discrete dependencies to the iteration variables =#::Bool
  end
end

@Uniontype InnerEquation begin
  @Record INNEREQUATION begin
    eqn::Integer
    vars::List{Integer}
  end

  @Record INNEREQUATIONCONSTRAINTS begin
    eqn::Integer
    vars::List{Integer}
    cons::Constraints
  end
end

@Uniontype StateSet begin
  @Record STATESET begin
    index::Integer
    rang::Integer
    #=  how many states are needed?
    =#
    state::List{DAE.ComponentRef}
    crA #= set.x=A*states =#::DAE.ComponentRef
    varA::List{Var}
    #= the jacobian matrix entries
    =#
    statescandidates::List{Var}
    #= all state candidates
    =#
    ovars::List{Var}
    #= other variables to solve the eqns
    =#
    eqns::List{Equation}
    #= the constraint equations
    =#
    oeqns::List{Equation}
    #= other equations to solve the eqns
    =#
    crJ::DAE.ComponentRef
    #=  the jac vector
    =#
    varJ::List{Var}
    jacobian::Jacobian
  end
end

#=  event info and stuff =#

@Uniontype EventInfo begin
  @Record EVENT_INFO begin
    timeEvents #= stores all information related to time events =#::List{TimeEvent}
    zeroCrossings #= list of zero crossing conditions =# #=TODO: Use something else..=#
    relations #= list of zero crossing function as before =#::DoubleEnded.MutableList
    numberMathEvents #= stores the number of math function that trigger events e.g. floor, ceil, integer, ... =#::Integer
  end
end

@Uniontype ZeroCrossingSet begin
  @Record ZERO_CROSSING_SET begin
    zc::DoubleEnded.MutableList
    tree::Array
  end
end

@Uniontype ZeroCrossing begin
  @Record ZERO_CROSSING begin
    relation_ #= function =#::DAE.Exp
    occurEquLst #= list of equations where the function occurs =#::List{Integer}
  end
end

@Uniontype TimeEvent begin
  @Record SIMPLE_TIME_EVENT begin
  end

  @Record SAMPLE_TIME_EVENT begin
    index #= unique sample index =#::Integer
    startExp::DAE.Exp
    intervalExp::DAE.Exp
  end
end


@Uniontype Solvability begin
  @Record SOLVABILITY_SOLVED begin
  end

  @Record SOLVABILITY_CONSTONE begin
  end

  @Record SOLVABILITY_CONST begin
    b #= false if the constant is almost zero (<1e-6) =#::Bool
  end

  @Record SOLVABILITY_PARAMETER begin
    b #= false if the partial derivative is zero =#::Bool
  end

  @Record SOLVABILITY_LINEAR begin
    b #= false if the partial derivative is zero =#::Bool
  end

  @Record SOLVABILITY_NONLINEAR begin
  end

  @Record SOLVABILITY_UNSOLVABLE begin
  end

  @Record SOLVABILITY_SOLVABLE begin
  end
end


@Uniontype IndexType begin
  @Record ABSOLUTE begin
  end

  @Record NORMAL begin
  end

  @Record SOLVABLE begin
  end

  @Record BASECLOCK_IDX begin
  end

  @Record SUBCLOCK_IDX begin
  end

  @Record SPARSE begin

  end
end

#=  Jacobian stuff =#

@Uniontype JacobianType begin
  @Record JAC_CONSTANT begin
  end

  @Record JAC_LINEAR begin
  end

  @Record JAC_NONLINEAR begin
  end

  @Record JAC_GENERIC begin
  end

  @Record JAC_NO_ANALYTIC begin
  end
end

const SymbolicJacobianAIndex = 1::Integer

const SymbolicJacobianBIndex = 2::Integer

const SymbolicJacobianCIndex = 3::Integer

const SymbolicJacobianDIndex = 4::Integer

const derivativeNamePrefix = "DERAlias"::String

const partialDerivativeNamePrefix = "pDER"::String

const functionDerivativeNamePrefix = "funDER"::String

const outputAliasPrefix = "outputAlias_"::String

const optimizationMayerTermName = "OMCobjectMayerTerm"::String

const optimizationLagrangeTermName = "OMCobjectLagrangeTerm"::String

const symSolverDT = "__OMC_DT"::String

const homotopyLambda = "__HOM_LAMBDA"::String
FullJacobian = Option

@Uniontype Jacobian begin
  @Record FULL_JACOBIAN begin
    jacobian::FullJacobian
  end

  @Record GENERIC_JACOBIAN begin
    jacobian::Option{SymbolicJacobian}
    sparsePattern::SparsePattern
    coloring::SparseColoring
  end

  @Record EMPTY_JACOBIAN begin
  end
end

@Uniontype DifferentiateInputData begin
  @Record DIFFINPUTDATA begin

    independenentVars::Option{Variables}
    #=  Independent variables
    =#
    dependenentVars::Option{Variables}
    #=  Dependent variables
    =#
    knownVars::Option{Variables}
    #=  known variables (e.g. parameter, constants, ...)
    =#
    allVars::Option{Variables}
    #=  all variables
    =#
    controlVars::List{Var}
    #=  variables to save control vars of for algorithm
    =#
    diffCrefs::List{DAE.ComponentRef}
    #=  all crefs to differentiate, needed for generic gradient
    =#
    matrixName::Option{String}
    #=  name to create temporary vars, needed for generic gradient
    =#
    diffedFunctions #=TODO: Use something sane here=#
    #=  current functions, to prevent recursive differentiation
    =#
  end
end

const emptyInputData = DIFFINPUTDATA(NONE(), NONE(), NONE(), NONE(), nil, nil, NONE(), nothing #=TODO: Fixme=#)::DifferentiateInputData

DifferentiateInputArguments = Tuple

#= Define the behaviour of differentiation method for (e.g. index reduction, ...) =#
@Uniontype DifferentiationType begin
  @Record DIFFERENTIATION_TIME begin
  end
end

#= types to count operations for the components =#
@Uniontype CompInfo begin
  @Record COUNTER begin
    #=  single equation =#
    comp::StrongComponent
    numAdds::Integer
    numMul::Integer
    numDiv::Integer
    numTrig::Integer
    numRelations::Integer
    numLog::Integer
    #=  logical operations =#
    numOth::Integer
    #=  pow,... =#
    funcCalls::Integer
  end

  @Record SYSTEM begin
    comp::StrongComponent
    allOperations::CompInfo
    size::Integer
    density::Real
  end

  @Record TORN_ANALYSE begin
    comp::StrongComponent
    tornEqs::CompInfo
    otherEqs::CompInfo
    tornSize::Integer
  end

  @Record NO_COMP begin
    numAdds::Integer
    numMul::Integer
    numDiv::Integer
    numTrig::Integer
    numRelations::Integer
    numLog::Integer
    numOth::Integer
    funcCalls::Integer
  end
end

@Uniontype BDAEModeData begin
  @Record BDAE_MODE_DATA begin
    stateVars::List{Var}
    algStateVars::List{Var}
    numResVars::Integer
    modelVars::Option{Variables}
  end
end

const emptyDAEModeData = BDAE_MODE_DATA(nil, nil, 0, NONE())::BDAEModeData

@exportAll()
end
