abstract type GibbsDivergence <: Divergence end

struct Hellinger <: GibbsDivergence end

struct KLDivergence <: GibbsDivergence end

struct FisherDivergence <: GibbsDivergence end

struct RelEntropyRate <: GibbsDivergence
end

struct GoalOrientedDivergence <: GibbsDivergence
    D :: Union{Hellinger, KLDivergence, FisherDivergence, RelEntropyRate}
end

"""
function compute_divergence

Computes f-divergence between distributions p and q. 

# Arguments
- `p :: Distribution`                   : first distribution
- `q :: Distribution`                   : second distribution
- `integrator :: GibbsIntegrator`       : integrator for calculating norm constants
- `D :: Divergence`                     : divergence metric
"""
## squared Hellinger
function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    integrator::GibbsIntegrator,
    D::Hellinger;
    normint=integrator,
)
    Zp = normconst(p, normint)
    Zq = normconst(q, normint)
    f(x) = (1 - sqrt(pdf(p, x, Zp) / pdf(q, x, Zq) ))^2 
    qoi = GibbsQoI(h=f, p=q)
    return compute_qoi(qoi, integrator).mean
end

function compute_divergence(
    p::Gibbs,
    q::Distribution,
    integrator::GibbsIntegrator,
    D::Hellinger;
    normint=integrator,
)
    Zp = normconst(p, normint)
    f(x) = (1 - sqrt(pdf(p, x, Zp) / pdf(q, x) ))^2 
    qoi = GeneralQoI(h=f, p=q)
    return compute_qoi(qoi, integrator).mean
end

"""
function compute_grad_divergence

Computes gradient of the f-divergence between distributions p and q. 

# Arguments
- `p :: Distribution`                   : first distribution
- `q :: Distribution`                   : second distribution
- `integrator :: GibbsIntegrator`       : integrator for calculating norm constants
- `D :: Divergence`                     : divergence metric
- `wrt :: Integer`                      : denotes gradient wrt parameters of argument 1 or 2
"""
function compute_grad_divergence(
    p::Gibbs,
    q::Gibbs,
    integrator::GibbsIntegrator,
    D::Hellinger;
    normint=integrator,
    wrt=1,
)
    # compute Hellinger
    H = sqrt(compute_divergence(p, q, integrator, D))

    # normalizing constants
    Zp = normconst(p, normint)
    Zq = normconst(q, normint)

    # compute inner expectation E_p[∇θV]
    if wrt == 1
        E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
        E_∇θV = compute_qoi(E_qoi, integrator).mean

        f(x) = p.β .* (sqrt(pdf(q,x,Zq)/pdf(p,x,Zp)) - 1) .* (p.∇θV(x) .- E_∇θV)
        qoi = GibbsQoI(h=f, p=p)
        ∇θH = sqrt(compute_qoi(qoi, integrator).mean)
    
        return 2 .* H .* ∇θH
    end
end


## KL divergence
function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    integrator::GibbsIntegrator,
    D::KLDivergence;
    normint=integrator,
)
    Zp = normconst(p, normint)
    Zq = normconst(q, normint)
    f(x) = (logupdf(p, x) - log(Zp)) - (logupdf(q, x) - log(Zq))
    qoi = GibbsQoI(h=f, p=p)
    return compute_qoi(qoi, integrator).mean
end

function compute_grad_divergence(
    p::Gibbs,
    q::Gibbs,
    integrator::GibbsIntegrator,
    D::KLDivergence,
    integrator2=integrator;
    wrt=1,
)
    if wrt == 1
        # compute inner expectation E_p[∇θV]
        E_qoi = GibbsQoI(h=p.∇θV, p=p)
        E_∇θV = compute_qoi(E_qoi, integrator).mean
        
        f(x) = - (logupdf(p, x) - logupdf(q, x)) .* (qoi.p.∇θV(x) .- E_∇θV)
        qoi = GibbsQoI(h=f, p=p)
        return compute_qoi(qoi, integrator).mean
    elseif wrt == 2
        qoi_p = GibbsQoI(h=q.∇θV, p=p)
        qoi_q = GibbsQoI(h=q.∇θV, p=q)
        return compute_qoi(qoi_p, integrator).mean .- compute_qoi(qoi_q, integrator2).mean
    else
        throw(ArgumentError("ERROR: argument `wrt` must be 1 or 2 (with respect to the distribution in argument 1 or 2)."))
    end
end


## Fisher divergence
function compute_divergence(
    p::Distribution,
    q::Distribution,
    integrator::GibbsIntegrator,
    D::FisherDivergence,
)
    f(x) = norm(gradlogpdf(p, x) - gradlogpdf(q, x))^2
    qoi = GibbsQoI(h=f, p=p)
    return compute_qoi(qoi, integrator).mean
end

function compute_grad_divergence(
    p::Gibbs,
    q::Gibbs,
    integrator::GibbsIntegrator,
    D::FisherDivergence,
    ∇θxV :: Function; # corresp. to potential of the (wrt)th argument
    wrt = 1,
)
    if wrt == 1
        f = x -> norm(gradlogpdf(p, x) - gradlogpdf(q, x))^2
        if typeof(p.θ) <: Real
            ∇f = x -> -2 * (gradlogpdf(p, x) - gradlogpdf(q, x)) * ∇θxV(x, p.θ)
            qoi = GibbsQoI(h=f, ∇h=∇f, p=p)
        else
            ∇f = x -> -2 * ((gradlogpdf(p, x) - gradlogpdf(q, x))' * ∇θxV(x, p.θ))[:]
            qoi = GibbsQoI(h=f, ∇h=∇f, p=p)
        end
        return compute_grad_qoi(qoi, integrator).mean
    elseif wrt == 2
        if typeof(q.θ) <: Real
            f = x -> 2 * ((gradlogpdf(p, x) - gradlogpdf(q, x)) * ∇θxV(x, q.θ))
            qoi = GibbsQoI(h=f, p=p)
        else
            f = x -> 2 * ((gradlogpdf(p, x) - gradlogpdf(q, x))' * ∇θxV(x, q.θ))[:]
            qoi = GibbsQoI(h=f, p=p)
        end
        return compute_qoi(qoi, integrator).mean
    else
        throw(ArgumentError("ERROR: argument `wrt` must be 1 or 2 (with respect to the distribution in argument 1 or 2)."))
    end
end

## Relative entropy rate (RER)
function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    integrator::GibbsIntegrator,
    D::RelEntropyRate,
)
    σ = sqrt(2/p.β)
    Σ = σ * σ'
    f(x) = 0.5 * (q.∇xV(x) - p.∇xV(x))' * pinv(Σ) * (q.∇xV(x) - p.∇xV(x))
    qoi = GibbsQoI(h=f, p=p)
    return compute_qoi(qoi, integrator).mean
end

function compute_grad_divergence(
    p::Gibbs,
    q::Gibbs,
    integrator::GibbsIntegrator,
    D::RelEntropyRate,
    ∇θxV::Function; # corresp. to potential of the (wrt)th argument
    wrt=1,
)
    σ = sqrt(2/p.β)
    Σ = σ * σ'
    Γ = pinv(Σ)
    if wrt == 1
        f = x -> 0.5 * (q.∇xV(x) - p.∇xV(x))' * Γ * (q.∇xV(x) - p.∇xV(x))
        if typeof(p.θ) <: Real
            ∇f = x -> - Γ * (q.∇xV(x) - p.∇xV(x)) * ∇θxV(x, p.θ)
            qoi = GibbsQoI(h=f, ∇h=∇f, p=p)
        else
            ∇f = x -> (- Γ * (q.∇xV(x) - p.∇xV(x))' * ∇θxV(x, p.θ))[:]
            qoi = GibbsQoI(h=f, ∇h=∇f, p=p)
        end
        return compute_grad_qoi(qoi, integrator).mean
    elseif wrt == 2
        if typeof(q.θ) <: Real
            f = x -> Γ * (q.∇xV(x) - p.∇xV(x)) * ∇θxV(x, q.θ)
            qoi = GibbsQoI(h=f, p=p)
        else
            f = x -> (Γ * (q.∇xV(x) - p.∇xV(x))' * ∇θxV(x, q.θ))[:]
            qoi = GibbsQoI(h=f, p=p)
        end
        return compute_qoi(qoi, integrator).mean
    else
        throw(ArgumentError("ERROR: argument `wrt` must be 1 or 2 (with respect to the distribution in argument 1 or 2)."))
    end
end


## goal-oriented divergence
function compute_divergence(
    g_p::GibbsQoI, # model
    g_q::GibbsQoI, # ref
    integrator::GibbsIntegrator,
    G::GoalOrientedDivergence;
    mq = nothing,
    kwargs...,
)
    # compute divergence metric
    H = compute_divergence(g_p.p, g_q.p, integrator, G.D; kwargs...)

    # compute second moment of observable
    res = compute_qoi(g_p, integrator)
    mp = res.var + res.mean^2

    if mq == nothing
        mq = mp
    end
        
    # return sqrt(2)/2 * (sqrt(mp)+sqrt(mq)) * sqrt(H)
    return 0.5*(mp + mq)*H
end

function compute_grad_divergence(
    g_p::GibbsQoI, # model
    g_q::GibbsQoI, # ref
    integrator::GibbsIntegrator,
    G::GoalOrientedDivergence,
    args...;
    mq = nothing,
    kwargs...,
)
    d = length(g_p.p.θ)
    
    # define second moment of observable and its gradient
    g2 = GibbsQoI(h= x -> g_p.h(x).^2,
        ∇h= x -> 2*g_p.h(x)*g_p.∇h(x),
        p = g_p.p)

    # compute expectations
    E_g2 = compute_qoi(g2, integrator).mean
    # E_d = compute_divergence(g_p.p, g_q.p, integrator, G.D)
    ∇E_d = compute_grad_divergence(g_p.p, g_q.p, integrator, G.D, args...; wrt=1)
    
    # skip next calculations if ∇h = 0
    xtest = rand(d)
    if g_p.∇h(xtest) == zeros(d) || g_p.∇h(xtest) == zeros(d)
        ∇E_g2 = 0
        E_d = 0
    else
        ∇E_g2 = compute_grad_qoi(g2, integrator).mean
        E_d = compute_divergence(g_p.p, g_q.p, integrator, G.D; kwargs...)
    end

    if mq == nothing
        mq = E_g2 
    end

    # return sqrt(2)/4 * ( ∇E_g2 .* sqrt(E_d) ./ sqrt(E_g2) .+ (sqrt(mq) + sqrt(E_g2)) * ∇E_d ./ sqrt(E_d) )
    return 0.5 * ((mq+E_g2)*∇E_d .+ ∇E_g2*E_d)
end



## methods with θ as argument
function compute_divergence(
    θ::Vector,
    p::Distribution,
    q::Distribution,
    integrator::GibbsIntegrator,
    D::GibbsDivergence;
    wrt=1,
    kwargs...,
)
    if wrt == 1
        pθ = Gibbs(p, θ=θ)
        return compute_divergence(pθ, q, integrator, D; kwargs...)
    elseif wrt == 2
        qθ = Gibbs(q, θ=θ)
        return compute_divergence(p, qθ, integrator, D; kwargs...)
    end
end

function compute_grad_divergence(
    θ::Vector,
    p::Distribution,
    q::Distribution,
    integrator::GibbsIntegrator,
    D::GibbsDivergence,
    args...;
    wrt=1,
    kwargs...,
)
    if wrt == 1
        pθ = Gibbs(p, θ=θ)
        return compute_grad_divergence(pθ, q, integrator, D, args...; wrt=1, kwargs...)
    elseif wrt == 2
        qθ = Gibbs(q, θ=θ)
        return compute_grad_divergence(p, qθ, integrator, D, args...; wrt=2, kwargs...)
    end
end

function compute_divergence(
    θ::Vector,
    g_p::GibbsQoI, # model
    g_q::GibbsQoI, # ref
    integrator::GibbsIntegrator,
    D::GoalOrientedDivergence;
    mq = nothing,
    kwargs...,
)
    g_pθ = assign_param(g_p, θ)
    return compute_divergence(g_pθ, g_q, integrator, D; mq=mq, kwargs...)
end


function compute_grad_divergence(
    θ::Vector,
    g_p::GibbsQoI, # model
    g_q::GibbsQoI, # ref
    integrator::GibbsIntegrator,
    D::GoalOrientedDivergence,
    args...;
    mq = nothing,
    kwargs...,
)
    g_pθ = assign_param(g_p, θ)
    return compute_grad_divergence(g_pθ, g_q, integrator, D, args...; mq=mq, kwargs...)
end