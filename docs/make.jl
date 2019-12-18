using Documenter, DiskArrays

makedocs(
    modules = [DiskArrays],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Fabian Gans",
    sitename = "DiskArrays.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/meggart/DiskArrays.jl.git",
)
