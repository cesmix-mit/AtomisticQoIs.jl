abstract type MCIntegrator <: GibbsIntegrator end

"""
struct MonteCarlo <: MCIntegrator

Contains parameters for Monte Carlo (MC) integration using MC samples.
This method may only be implemented when the distribution can be analytically sampled using rand().

# Arguments
- `n :: Int`   : number of samples
"""
struct MonteCarlo <: MCIntegrator
    n :: Int
end


"""
struct MCMC <: MCIntegrator
    
Contains parameters for Monte Carlo (MC) integration using MCMC samples.
This method is implemented when the distribution cannot be analytically sampled.

# Arguments
- `T :: Int`               : number of simulation steps
- `n :: Int`               : number of samples
- `sampler :: Sampler`     : type of sampler (see `Sampler`)
- `ρ0 :: Union{Distribution, Real, Vector{<:Real}}`     : initial state or prior distribution of the state
"""
struct MCMC <: MCIntegrator
    T :: Int 
    n :: Int
    sampler :: Sampler
    ρ0 :: Union{Distribution, Real, Vector{<:Real}}
end
MCMC(n::Int, sampler::Sampler, ρ0::Union{Distribution, Real, Vector{<:Real}}) = MCMC(n, n, sampler, ρ0)


"""
struct MCSamples <: MCIntegrator

Contains pre-defined Monte Carlo samples to evaluate in integration.
This method is implemented with the user providing samples from the distribution.

# Arguments
- `xsamp :: Vector`    : fixed set of samples
"""
struct MCSamples <: MCIntegrator
    xsamp :: Vector
end
