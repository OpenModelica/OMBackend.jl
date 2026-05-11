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

#= Lowered form of a Modelica `when initial() then ... end when` clause body.
   The body fires once during initialization (Modelica spec §8/§11), not as a
   runtime callback. Holds BDAE.WhenOperator entries (ASSIGN/REINIT/ASSERT/
   TERMINATE/NORETCALL) — same shape as a regular when-equation body. =#
struct INITIAL_ALGORITHM
  statements::Vector{BDAE.WhenOperator}
end
