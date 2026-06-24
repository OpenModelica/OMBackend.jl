[![Github Action CI](https://github.com/OpenModelica/OMBackend.jl/workflows/CI/badge.svg)](https://github.com/OpenModelica/OMBackend.jl/actions) [![License: OSMC-PL](https://img.shields.io/badge/license-OSMC--PL-lightgrey.svg)](LICENSE.md)

# About OMBackend.jl

OMBackend.jl is one component of OMJL, the Modelica compiler infrastructure
for Julia. It transforms a Hybrid DAE produced by OMFrontend.jl into a system
that can be simulated with DifferentialEquations.jl and ModelingToolkit.jl.

# Requirements

* Julia 1.12
* Git
* A checkout of the sibling OM*.jl packages next to this repository (see below)

# Dependencies

The main runtime dependencies are pulled in by `Project.toml`. The most
relevant ones are:

* DifferentialEquations.jl, OrdinaryDiffEq.jl, DiffEqBase.jl, DiffEqCallbacks.jl, Sundials.jl
* ModelingToolkit.jl, Symbolics.jl, SymbolicUtils.jl
* Graphs.jl, MetaGraphs.jl, DataStructures.jl, MacroTools.jl, Setfield.jl
* Plots.jl, GR.jl, CSV.jl, Tables.jl
* ExportAll.jl, DocStringExtensions.jl, JuliaFormatter.jl

The OMJL ecosystem packages are consumed as in-tree siblings via
`Project.toml`'s `[sources]` block. They must be checked out as sibling
directories next to `OMBackend.jl`:

* Absyn.jl
* DAE.jl
* ImmutableList.jl
* ListUtil.jl
* MetaModelica.jl
* OMFrontend.jl
* OMParser.jl
* OMRuntimeExternalC.jl
* SCode.jl

Layout:

```
OMJL-workspace/
├── Absyn.jl/
├── DAE.jl/
├── ImmutableList.jl/
├── ListUtil.jl/
├── MetaModelica.jl/
├── OMBackend.jl/        <-- this repository
├── OMFrontend.jl/
├── OMParser.jl/
├── OMRuntimeExternalC.jl/
└── SCode.jl/
```

`OMRuntimeExternalC.jl` is not in the General registry. The others are served
by the OpenModelica Julia registry but the local `[sources]` paths take
precedence and are used for development.

# Installation

Add the OpenModelica Julia registry once per Julia depot:

```julia
julia> import Pkg
julia> Pkg.Registry.add("General")
julia> Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/OpenModelica/OpenModelicaRegistry.git"))
```

From the `OMBackend.jl` directory, activate the project and build:

```julia
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.build(verbose = true)
```

`deps/build.jl` resolves and builds the dependent OM*.jl packages (notably
`OMParser` and `OMFrontend`).

Then load the package:

```julia
julia> using OMBackend
```

# Running tests

From the `OMBackend.jl` directory:

```julia
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.test()
```

Or directly:

```julia
julia> include("test/runtests.jl")
```

# Example use

Given a Modelica model translated to Hybrid DAE by OMFrontend.jl, pass it to
`OMBackend.translate` and then simulate.

```modelica
model BouncingBallReals
  parameter Real e = 0.7;
  parameter Real g = 9.81;
  Real h(start = 1);
  Real v;
equation
  der(h) = v;
  der(v) = -g;
  when h <= 0 then
    reinit(v, -e * pre(v));
  end when;
end BouncingBallReals;
```

```julia
julia> OMBackend.translate(BouncingBallReals)
julia> OMBackend.simulate("BouncingBallReals", tspan = (0.0, 2.5))
```

![image](https://user-images.githubusercontent.com/8775827/99516636-b6914280-298e-11eb-85cf-c9041314e9b4.png)

# Citation

If you use OMBackend.jl in academic work, please cite the entries in
[CITATION.bib](CITATION.bib).

# License

OMBackend.jl is distributed under the OSMC Public License. See
[LICENSE.md](LICENSE.md).
