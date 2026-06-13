@info("OMBackend: starting build script")

# Build scripts run in an isolated environment that does not include stdlibs
# on the load path, so reach into the user depot to get at Pkg.
push!(LOAD_PATH, "@v#.#", "@stdlib")

using Pkg

# The runtime dependency set is already declared in Project.toml and resolved
# by Pkg.instantiate, so we do not re-add packages here. The OM siblings live
# in [sources] as local paths. The only thing the build step still owns is
# triggering the native build steps of OMParser and OMFrontend so that their
# generated artifacts are present before the first `using OMBackend`.

for pkg in ("OMParser", "OMFrontend")
    @info "OMBackend: building $pkg"
    Pkg.build(pkg; verbose = true)
end

@info("OMBackend: finished build script")
