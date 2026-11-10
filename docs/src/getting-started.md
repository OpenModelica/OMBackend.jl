# Getting started

## Installation

The packages are in the OpenModelica registry. Add it once per Julia depot, then
install:

```julia
julia> import Pkg
julia> Pkg.Registry.add("General")
julia> Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/OpenModelica/OpenModelicaRegistry.git"))
julia> Pkg.add("OMBackend")
```

To translate Modelica source add OMFrontend as well (`Pkg.add("OMFrontend")`):
loading it loads the extension that connects the two.

## Example

Given a Modelica model flattened by OMFrontend.jl, pass the flat model and its
functions to `OMBackend.translate`, then simulate:

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
julia> import OMFrontend, OMBackend
julia> scode = OMFrontend.translateToSCode(OMFrontend.parseFile("BouncingBallReals.mo"))
julia> (fm, cache) = OMFrontend.instantiateSCodeToFM("BouncingBallReals", scode)
julia> OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
julia> OMBackend.simulateModel("BouncingBallReals"; tspan = (0.0, 2.5))
```

`examples/` in the repository has more models and `examples/simulate_examples.jl`
runs them.

## Documentation in the REPL

Every function of the [API reference](api.md) has its docstring in Julia's help
mode, for example `?OMBackend.simulateModel`.

## Tests

```julia
julia> import Pkg
julia> Pkg.test("OMBackend")
```
