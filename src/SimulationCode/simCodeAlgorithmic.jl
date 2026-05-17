abstract type ModelicaFunction end

struct MODELICA_FUNCTION <: ModelicaFunction
  name::String
  #= Vector of integer, real string etc=#
  inputs::Vector
  outputs::Vector
  locals::Vector
  statements::Vector{DAE.Statement}
end

struct EXTERNAL_MODELICA_FUNCTION <: ModelicaFunction
  name::String
  inputs::Vector
  outputs::Vector
  libInfo::String
end

#= Lowered form of a Modelica `initial algorithm` or `algorithm when initial()`
   body. Fires once during initialization (Modelica spec §11.4).

   `statements` holds the BDAE.WhenOperator form (control flow flattened by
   `_daeStmtsToWhenOps`), kept for the runtime cycle-19 path that translates
   per-WhenOperator with `LATEST_PROBLEM` side effects.

   `daeStatements` holds the original DAE.Statement list converted by
   `OMFrontend.Frontend.convertStatements` from the flat-model algorithm body,
   preserving STMT_IF / STMT_FOR / STMT_WHILE control flow. This is the
   representation `AlgorithmicCodeGeneration.generateStatements` consumes, and
   is the path the module-load-time `__runInitialAlgorithmEarly!()` lowering
   uses to honour Modelica §11.4 sequential semantics on bodies with control
   flow. Empty when populated by sources that only know the WhenOperator form
   (the fallback). =#
struct INITIAL_ALGORITHM
  statements::Vector{BDAE.WhenOperator}
  daeStatements::Vector{DAE.Statement}
end

INITIAL_ALGORITHM(stmts::Vector{BDAE.WhenOperator}) = INITIAL_ALGORITHM(stmts, DAE.Statement[])
