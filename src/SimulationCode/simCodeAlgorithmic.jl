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
   body. Fires once during initialization (Modelica spec §11.4). `statements`
   carries the SimCode-native `WhenOperator` form (control flow flattened
   by `_daeStmtsToWhenOps`); `daeStatements` carries the structured
   `DAE.Statement` list (preserving STMT_IF / STMT_FOR / STMT_WHILE) for the
   module-load-time codegen path. =#
struct INITIAL_ALGORITHM
  statements::Vector{WhenOperator}
  daeStatements::Vector{DAE.Statement}
end

INITIAL_ALGORITHM(stmts::Vector{WhenOperator}) = INITIAL_ALGORITHM(stmts, DAE.Statement[])

#= Boundary convenience: accept a `List{BDAE.WhenOperator}` or
   `Vector{BDAE.WhenOperator}` at construction time and project through
   `toWhenOperator` so the boundary callers do not need to thread the
   conversion themselves. =#
function INITIAL_ALGORITHM(stmts, daeStmts::Vector{DAE.Statement})
  local simStmts = WhenOperator[toWhenOperator(s) for s in stmts]
  return INITIAL_ALGORITHM(simStmts, daeStmts)
end

function INITIAL_ALGORITHM(stmts)
  local simStmts = WhenOperator[toWhenOperator(s) for s in stmts]
  return INITIAL_ALGORITHM(simStmts, DAE.Statement[])
end
