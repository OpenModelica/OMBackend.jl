@info("OMBackend: Starting build script")

push!(LOAD_PATH, "@v#.#", "@stdlib")
@info("Current loadpath: $LOAD_PATH")

using Pkg

function buildDeps()
  Pkg.add("ExportAll")
  Pkg.add("Sundials")
  Pkg.add("CSV")
  Pkg.add("DifferentialEquations")
  Pkg.add("Setfield")
  Pkg.add("DataStructures")
  Pkg.add("Plots")
  Pkg.add("LightGraphs")
  Pkg.add("MetaGraphs")
  Pkg.add("JuliaFormatter")
  Pkg.add("MacroTools")
  #= This packages are available using the OpenModelica Julia registry=#
  Pkg.add("ImmutableList")
  Pkg.add("MetaModelica")
  Pkg.add("Absyn")
  Pkg.add("SCode")
  Pkg.add("DAE")
  #= Add if we do not have it =#
  Pkg.add("OMParser")
  Pkg.build("OMParser") #= Build the parser =#
  Pkg.add("OMFrontend")
  Pkg.build("OMFrontend")
  @info("OMFrontend + OMParser built successfully")
  @info("All dependencies succesfull")
end

buildDeps()
@info("OMBackend: Finished build script")
