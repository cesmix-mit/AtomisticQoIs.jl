""" 
    Divergence

    A struct of abstract type Divergence contains properties needed to compute a divergence metric between measures of the stochastic process.
"""
abstract type Divergence end

include("gibbs_divergences.jl")
include("path_divergences.jl")
include("norm_errors.jl")
include("error_bounds.jl")

export
    GibbsDivergence,
    Hellinger,
    KLDivergence,
    FisherDivergence,
    RelEntropyRate,
    GoalOrientedDivergence,
    PathDivergence,
    PathHellinger,
    PathKL,
    PathFisher,
    compute_divergence,
    compute_grad_divergence,
    RadonNikodym,
    NormError,
    LpNorm,
    MGFError,
    ErrorBound,
    CuiBound,
    StuartBound,
    GoalOrientedBound,
    compute_error_bound


