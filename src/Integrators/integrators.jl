abstract type Integrator end
""" 
    GibbsIntegrator

    A struct of abstract type GibbsIntegrator computes the expectation of a function h(x, θ) with respect to the invariant measure p(x, θ).
"""
abstract type GibbsIntegrator <: Integrator end

""" 
    PathIntegrator

    A struct of abstract type PathIntegrator computes the expectation of a functional h with respect to the path measure P. 
"""
abstract type PathIntegrator <: Integrator end

include("quadrature.jl")
include("monte_carlo.jl")
include("importance_sampling.jl")
include("path_integrators.jl")


export
    GibbsIntegrator, 
    MCIntegrator,
    MonteCarlo,
    MCMC,
    MCSamples,
    MCPaths,
    ISIntegrator,
    ISMC,
    ISMCMC,
    ISSamples,
    ISMixSamples,
    QuadIntegrator,
    GaussQuadrature,
    gaussquad,
    gaussquad_2D,
    PathIntegrator,
    RiemannIntegrator,
    ItoIntegrator,
    compute_integral