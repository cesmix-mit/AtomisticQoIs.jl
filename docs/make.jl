using AtomisticQoIs
using Documenter

DocMeta.setdocmeta!(AtomisticQoIs, :DocTestSetup, :(using AtomisticQoIs); recursive=true)

makedocs(;
    modules=[AtomisticQoIs],
    authors="Joanna Zou (jjzou@mit.edu) and contributors",
    repo="https://github.com/cesmix-mit/AtomisticQoIs.jl/blob/{commit}{path}#{line}",
    sitename="AtomisticQoIs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cesmix-mit.github.io/AtomisticQoIs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cesmix-mit/AtomisticQoIs.jl",
    devbranch="main",
)
