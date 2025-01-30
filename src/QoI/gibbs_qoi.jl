"""
@kwdef mutable struct GibbsQoI <: QoI

Defines quantities for computing one-point invariant statistics, in particular: 
mean and variance of the observable function h over the Gibbs measure p, e. g. E_p[h(θ)] and Var_p[h(θ)]

# Arguments
- `h :: Function`   : function to evaluate expectation, receiving inputs (x,θ)
- `p :: Gibbs`      : Gibbs (invariant) distribution with θ == nothing (undefined parameters)
- `∇h :: Union{Function, Nothing} = nothing`   : gradient of h (if nothing, gradient is computed using ForwardDiff)

"""
@kwdef mutable struct GibbsQoI <: QoI
    h :: Function                               # function to evaluate expectation, receiving inputs (x,θ)
    p :: Gibbs                                  # Gibbs (invariant) distribution, receiving inputs (x,θ)
    ∇h :: Union{Function, Nothing} = nothing    # gradient of h (if nothing, gradient is computed using AutoDiff)
    
    function GibbsQoI(h::Function, p::Gibbs, ∇h::Function)
        return new(h, p, ∇h)
    end

    function GibbsQoI(h::Function, p::Gibbs, ∇h::Nothing)
        return new(h, p, ∇h)
    end

end

"""
function assign_param(qoi::GibbsQoI, θ::Vector)

Helper function modifying a GibbsQoI for a specific value of the parameter θ

"""
function assign_param(qoi::GibbsQoI, θ::Union{Real, Vector{<:Real}})
    if qoi.∇h === nothing
        return GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))
    else
        return GibbsQoI(
            h = x -> qoi.h(x, θ),
            ∇h = x -> qoi.∇h(x, θ),
            p=Gibbs(qoi.p, θ=θ))
    end
end


"""
function compute_qoi(θ:: Union{Real,Vector{<:Real}}, qoi::QoI, integrator::GibbsIntegrator)

Evaluates the expectation-valued quantity of interest (QoI).

# Arguments
- `θ :: Union{Real, Vector{<:Real}}`    : parameters to evaluate expectation
- `qoi :: QoI`                          : QoI object containing summand and measure of the expectation
- `integrator :: GibbsIntegrator`            : struct specifying method of integration

# Outputs 
- `res :: NamedTuple`                   : contains mean and variance of the summand and optional fields (samples, importance sampling weights)

""" 
# quadrature integration
function compute_qoi( 
            qoi::GibbsQoI,
            integrator::GaussQuadrature,
)
    qoi2 = GibbsQoI(h= x -> qoi.h(x).^2,
        ∇h= x -> 2*qoi.h(x)*qoi.∇h(x),
        p = qoi.p)
    
    # norm const
    Z = normconst(qoi.p, integrator)

    h1(x) = qoi.h(x) .* updf(qoi.p, x) ./ Z
    h2(x) = qoi2.h(x) .* updf(qoi2.p, x) ./ Z

    EX = sum(integrator.w .* h1.(integrator.ξ))
    EX2 = sum(integrator.w .* h2.(integrator.ξ))

    res = (
        mean = EX,
        var = EX2 - EX.^2,
    )
    return res
end


# integration with MCMC samples
function compute_qoi(
            qoi::GibbsQoI,
            integrator::MCMC,
)
    # x samples
    # xsamp = rand(qoi.p, integrator.T, integrator.sampler, integrator.ρ0)
    burnin = 1000
    xsamp, _ = sample(x -> qoi.p.V(x), x -> qoi.p.∇xV(x), integrator.sampler, integrator.T+burnin, integrator.ρ0)
    xsamp = xsamp[burnin+1:end]

    # subsample from path
    if integrator.T != integrator.n
        xsamp = xsamp[StatsBase.sample(1:integrator.T, integrator.n; replace=false)]
    end

    res = (
        mean = mean(qoi.h.(xsamp)),
        var = var(qoi.h.(xsamp)),
        xsamp = xsamp,
    )
    return res
end


# integration with MC samples provided
function compute_qoi(
            qoi::GibbsQoI,
            integrator::MCSamples,
)
    res = (
        mean = mean(qoi.h.(integrator.xsamp)),
        var = var(qoi.h.(integrator.xsamp)),
        xsamp = integrator.xsamp,
    )
    return res
end


# importance sampling (MCMC sampling from biasing dist.)
function compute_qoi(
            qoi::GibbsQoI,
            integrator::ISMCMC)
    # x samples
    xsamp = rand(integrator.g, integrator.n, integrator.sampler, integrator.ρ0) 

    # compute mean
    EX, hsamp, iswts = expectation_is_stable(xsamp, qoi.h, qoi.p, integrator.g)

    # compute variance
    qoi2 = GibbsQoI(h= x -> qoi.h(x).^2,
        ∇h= x -> 2*qoi.h(x)*qoi.∇h(x),
        p = qoi.p)
    EX2, hsamp, iswts = expectation_is_stable(xsamp, qoi2.h, qoi.p, integrator.g)

    res = (
        mean = EX,
        var = EX2 - EX.^2,
        xsamp = xsamp,
        hsamp = hsamp,
        wts = iswts,
    )
    return res

end


# importance sampling (MC sampling from biasing dist.)
function compute_qoi(
            qoi::GibbsQoI,
            integrator::ISMC,
) 
    # x samples
    xsamp = rand(integrator.g, integrator.n)

    # compute mean
    EX, hsamp, iswts = expectation_is_stable(xsamp, qoi.h, qoi.p, integrator.g)

    # compute variance
    qoi2 = GibbsQoI(h= x -> qoi.h(x).^2,
        ∇h= x -> 2*qoi.h(x)*qoi.∇h(x),
        p = qoi.p)
    EX2, hsamp, iswts = expectation_is_stable(xsamp, qoi2.h, qoi.p, integrator.g)

    res = (
        mean = EX,
        var = EX2 - EX.^2,
        xsamp = xsamp,
        hsamp = hsamp,
        wts = iswts,
    )
    return res

end


# importance sampling (samples provided)
function compute_qoi(
            qoi::GibbsQoI,
            integrator::ISSamples,
)         
    # compute log IS weights
    EX, hsamp, iswts = expectation_is_stable(integrator.xsamp, qoi.h, qoi.p, integrator.g; normint = integrator.normint)

    # compute variance
    qoi2 = GibbsQoI(h= x -> qoi.h(x).^2,
        ∇h= x -> 2*qoi.h(x)*qoi.∇h(x),
        p = qoi.p)
    EX2, _, _ = expectation_is_stable(integrator.xsamp, qoi2.h, qoi.p, integrator.g; normint = integrator.normint)

    res = (
        mean = EX,
        var = EX2 - EX.^2,
        xsamp = integrator.xsamp,
        hsamp = hsamp,
        wts = iswts,
    )
    return res

end


# importance sampling from mixture distribution (samples provided)
function compute_qoi(
            qoi::GibbsQoI,
            integrator::ISMixSamples,
)        
    # compute mixture weights
    wts = [compute_kernel(θ, c, integrator.knl) for c in integrator.refs]
    wts = wts ./ sum(wts)

    # sample from mixture model
    mm = MixtureModel(integrator.g, wts)
    xmix, rat = rand(mm, integrator.n, integrator.xsamp)

    # compute log IS weights
    expec, hsamp, iswts = expectation_is_stable(xmix, qoi.h, qoi.p, integrator.g, normint=integrator.normint)
    
    res = (
        mean = expec,
        xsamp = xmix,
        hsamp = hsamp,
        wts = iswts,
    )
    return res
    
end


"""
function compute_grad_qoi(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::GibbsIntegrator; gradh::Union{Function, Nothing}=nothing)

Computes the gradient of the QoI with respect to the parameters θ, e. g. ∇θ E_p[h(θ)].

# Arguments
- `θ :: Union{Real, Vector{<:Real}}`    : parameters to evaluate expectation
- `qoi :: QoI`                          : QoI object containing summand h and Gibbs measure p
- `integrator :: GibbsIntegrator`            : struct specifying method of integration
- `gradh :: Union{Function, Nothing}`   : gradient of qoi.h; if nothing, compute with ForwardDiff

# Outputs
- `res :: NamedTuple`                   : contains output quantities (see `compute_qoi` for more info)


"""
# parameter already provided in qoi
function compute_grad_qoi(
                    qoi::GibbsQoI,
                    integrator::GibbsIntegrator)

    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV = compute_qoi(E_qoi, integrator).mean

    # compute outer expectation
    hh(x) = qoi.∇h(x) .- qoi.p.β * qoi.h(x) .* (qoi.p.∇θV(x) .- E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return compute_qoi(hh_qoi, integrator)
end

function compute_grad_qoi(θ::Union{Real, Vector{<:Real}},
                          qoi::GibbsQoI,
                          integrator::GibbsIntegrator)
    # compute gradient of h
    if qoi.∇h === nothing
        qoi.∇h = (x, γ) -> ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    end

    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV = compute_qoi(θ, E_qoi, integrator).mean
    
    # compute outer expectation
    hh(x, γ) = qoi.∇h(x, γ) .- qoi.p.β * qoi.h(x, γ) .* (qoi.p.∇θV(x, γ) .- E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return compute_qoi(θ, hh_qoi, integrator)
end



function expectation_is_stable(xsamp::Vector, ϕ::Function, f::Gibbs, g::Distribution; normint=nothing)
    logwt(xsamp) = if hasupdf(g) # Gibbs biasing dist
        Zg = normconst(g, normint)
        logupdf.((f,), xsamp) .- logupdf.((g,), xsamp) .- log(Zg)
    elseif hasapproxnormconst(g) # mixture biasing dist
        logupdf.((f,), xsamp) .- logpdf(g, xsamp, normint)
    else # other biasing dist from Distributions.jl
        logupdf.((f,), xsamp) .- logpdf.((g,), xsamp)
    end

    M = maximum(logwt(xsamp))
    return sum( ϕ.(xsamp) .* exp.(logwt(xsamp) .- M) ) / sum( exp.(logwt(xsamp) .- M) ), ϕ.(xsamp), exp.(logwt(xsamp))
end



