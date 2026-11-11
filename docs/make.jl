using Documenter
using OMBackend

# Documenter pulls the docstrings from the loaded module. The frontend's part of the entry points lives
# in the package extension OMBackendOMFrontendExt, which Julia loads together with OMFrontend: the pages
# name it, its methods are not part of this reference.
DocMeta.setdocmeta!(OMBackend, :DocTestSetup, :(using OMBackend); recursive = true)

makedocs(
    sitename = "OMBackend.jl",
    modules = [OMBackend],
    authors = "John Tinnerholm <johti17@liu.se>, Adrian Pop <adrian.pop@liu.se>",
    repo = Documenter.Remotes.GitHub("OpenModelica", "OMBackend.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://OpenModelica.github.io/OMBackend.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "API reference" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

# Only deploy from CI; local `julia --project=docs docs/make.jl` just builds.
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/OpenModelica/OMBackend.jl.git",
        devbranch = "master",
        push_preview = true,
    )
end
