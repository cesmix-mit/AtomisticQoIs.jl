"""
@kwdef mutable struct AutocorrelationQoI <: QoI

Defines quantities for computing two-point invariant statistics, in particular: 
mean and variance of the autocorrelation of the function H over the Gibbs measure p with lag t, e. g. E_p[H(xt)H(x0)] and Var_p[H(xt)H(x0)]

# Arguments
- `H :: Function`   : function to evaluate autocorrelation
- `lag :: Real`     : autocorrelation lag
- `s :: Function`   : score (drift) function

"""
@kwdef mutable struct AutocorrelationQoI <: QoI
    H :: Function             # function of state
    lag :: Real               # autocorrelation lag
    s :: Function             # score (drift) function
end


"""
@kwdef mutable struct GreenKuboQoI <: QoI

Defines quantities for computing a Green Kubo integral, in particular: 
the integrated autocorrelation function up to time lag t, e. g. ∫₀ᵗ E_p[H(xs)H(x0)] ds

    # Arguments
- `H :: Function`   : function to evaluate autocorrelation
- `lag :: Real`     : autocorrelation lag
- `s :: Function`   : score (drift) function

"""
@kwdef mutable struct GreenKuboQoI <: QoI
    H :: Function             # function of state
    lag :: Real               # autocorrelation lag
    s :: Function             # score (drift) function
end


function assign_param(qoi::AutocorrelationQoI, θ::Vector)
    return AutocorrelationQoI(H=qoi.H, lag=qoi.lag, s = x -> qoi.s(x, θ))
end
function assign_param(qoi::GreenKuboQoI, θ::Vector)
    return GreenKuboQoI(H=qoi.H, lag=qoi.lag, s = x -> qoi.s(x, θ))
end


function compute_qoi(
    qoi::AutocorrelationQoI,
    integrator::MCPaths;
    burnin=1000,
)
    (; H, lag, s) = qoi
    (; M, n, sampler, ρ0) = integrator

    if typeof(ρ0) <: Distribution; x0 = rand(ρ0); else; x0=ρ0; end
    xsamp, _ = AtomisticQoIs.sample(s, sampler, n, x0)
    
    κ = [H(xsamp[k])'*H(xsamp[k+lag]) for k = burnin:(n - lag)]

    res = (
        mean = mean(κ),
        var = var(κ),
    )
    return res
end


function compute_qoi(
    qoi::GreenKuboQoI,
    integrator::MCPaths;
    burnin=1000,
)
    (; H, lag, s) = qoi
    (; M, n, sampler, ρ0) = integrator

    D = zeros(n)
    for i = 1:n
        xsamp, _ = sample(s, sampler, n, x0)
        D[i] = sum(autocov(H.(xsamp[1:burnin]), 1:lag, demean=false))
    end

    res = (
        mean = mean(D),
        var = var(D),
    )
    return res
end
