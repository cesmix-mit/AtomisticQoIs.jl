abstract type PathDivergence <: Divergence end

struct PathHellinger <: PathDivergence
    Xpaths :: Vector{<:Vector}
    Wpaths :: Vector{<:Vector}
    dt :: Real
end

struct PathKL <: PathDivergence
    Xpaths :: Vector{<:Vector}
    dt :: Real
end

struct PathFisher <: PathDivergence end


function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    D::PathDivergence;
    kwargs...
)
    if p.β != q.β
        throw(ArgumentError("β parameter must be identical between Gibbs objects p and q."))
    end
    σ = sqrt(2/p.β)
    return compute_divergence(x -> -p.∇xV(x), x -> -q.∇xV(x), σ, D; kwargs...)
end
function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    qoi::HittingTimeQoI,
    mcp::MCPaths,
    D::PathDivergence;
    kwargs...
)
    if p.β != q.β
        throw(ArgumentError("β parameter must be identical between Gibbs objects p and q."))
    end
    σ = sqrt(2/p.β)
    _, Xpaths, Wpaths = compute_hitting_time(mcp.ρ0, mcp.sampler, x -> p.∇xV(x), qoi.D, mcp.n, mcp.M)
    if typeof(D) <: PathHellinger
        D = PathHellinger(Xpaths, Wpaths, mcp.sampler.ϵ)
    elseif typeof(D) <: PathKL
        D = PathKL(Xpaths, mcp.sampler.ϵ)
    end
    return compute_divergence(x -> -p.∇xV(x), x -> -q.∇xV(x), σ, D; kwargs...)
end

function compute_divergence(
    b::Function,
    a::Function,
    σ::Union{Matrix, Real},
    D::PathHellinger;
    sgn=-1,
)
    Npaths = length(D.Xpaths)
    
    H = 0
    for n = 1:Npaths
        rn = RadonNikodym(b,a,σ,D.Xpaths[n],D.Wpaths[n],D.dt;sgn=sgn)
        H += (1 - sqrt(rn))^2
    end

    return sqrt(H / Npaths)
end


function compute_divergence(
    b::Function,
    a::Function,
    σ::Union{Matrix, Real},
    D::PathKL,
)
    Npaths = length(D.Xpaths)

    u(x) = inv(σ) * (b(x) - a(x))

    KL = 0
    for n = 1:Npaths
        riem = RiemannIntegrator(D.Xpaths[n], D.dt)
        KL += compute_integral(x -> u(x)' * u(x), riem) 
    end
    
    return KL / (2*Npaths)
end


function compute_divergence(
    p::Gibbs, # model
    q::Gibbs, # ref
    qoi::HittingTimeQoI,
    statint::GibbsIntegrator,
    pathint::MCPaths,
    G::GoalOrientedDivergence;
    mq = nothing,
)
    T = pathint.n * pathint.sampler.ϵ # total simulation time

    # compute divergence metric
    RER = compute_divergence(p,q,statint,RelEntropyRate())

    # compute second moment of observable
    res = compute_qoi(p.θ, qoi, pathint)
    mp = res.var + res.mean^2

    if mq == nothing
        mq = mp
    end
    # return sqrt(T)/2 * (sqrt(mp)+sqrt(mq)) * sqrt(RER)
    return T/2 * (mq + mp) * RER
end

function compute_divergence(
    p::Gibbs, # model
    q::Gibbs, # ref
    qoi::HittingTimeQoI,
    statint::GibbsIntegrator,
    pathint::Union{MCPathSamples, RDPathSamples},
    G::GoalOrientedDivergence;
    mq = nothing,
)
    T = pathint.N * pathint.dt # total simulation time

    # compute divergence metric
    RER = compute_divergence(p,q,statint,RelEntropyRate())

    # compute second moment of observable
    res = compute_qoi(p.θ, qoi, pathint)
    mp = res.var + res.mean^2

    if mq == nothing
        mq = mp
    end
    # return sqrt(T)/2 * (sqrt(mp)+sqrt(mq)) * sqrt(RER)
    return T/2 * (mq + mp) * RER
end


function compute_grad_divergence(
    p::Gibbs, # model
    q::Gibbs, # ref
    qoi::HittingTimeQoI,
    statint::GibbsIntegrator,
    pathint::MCPaths,
    G::GoalOrientedDivergence,
    args...;
    mq = nothing,
)
    T = pathint.n * pathint.sampler.ϵ # total simulation time

    # compute divergence metric and its gradient
    E_d = compute_divergence(p,q,statint,RelEntropyRate())
    ∇E_d = compute_grad_divergence(p,q,statint,RelEntropyRate(),args...;wrt=1)

    # compute second moment of observable and its gradient
    E_τ2, ∇E_τ2 = compute_grad_qoi2(p.θ, qoi, pathint, args...)

    if mq == nothing
        mq == E_τ2
    end

    # return sqrt(T)/4 * ( ∇E_τ2 .* sqrt(E_d) ./ sqrt(E_τ2) .+ (sqrt(mq) + sqrt(E_τ2)) * ∇E_d ./ sqrt(E_d) )
    return T/2 * (∇E_τ2 .* E_d .+ (mq .+ E_τ2) .* ∇E_d)
end

function compute_grad_divergence(
    p::Gibbs, # model
    q::Gibbs, # ref
    qoi::HittingTimeQoI,
    statint::GibbsIntegrator,
    pathint::MCPathSamples,
    G::GoalOrientedDivergence,
    args...;
    mq = nothing,
)
    T = pathint.N * pathint.dt # total simulation time

    # compute divergence metric and its gradient
    E_d = compute_divergence(p,q,statint,RelEntropyRate())
    ∇E_d = compute_grad_divergence(p,q,statint,RelEntropyRate(),args...;wrt=1)

    # compute second moment of observable and its gradient
    E_τ2, ∇E_τ2 = compute_grad_qoi2(p.θ, qoi, pathint, args...)

    if mq == nothing
        mq == E_τ2
    end

    # return sqrt(T)/4 * ( ∇E_τ2 .* sqrt(E_d) ./ sqrt(E_τ2) .+ (sqrt(mq) + sqrt(E_τ2)) * ∇E_d ./ sqrt(E_d) )
    return T/2 * (∇E_τ2 .* E_d .+ (mq .+ E_τ2) .* ∇E_d)
end

function compute_grad_divergence(
    p::Gibbs, # model
    q::Gibbs, # ref
    qoi::HittingTimeQoI,
    statint::GibbsIntegrator,
    pathint::RDPathSamples,
    G::GoalOrientedDivergence,
    args...;
    mq = nothing,
)
    T = pathint.N * pathint.dt # total simulation time

    # compute divergence metric and its gradient
    E_d = compute_divergence(q,p,statint,RelEntropyRate())
    ∇E_d = compute_grad_divergence(q,p,statint,RelEntropyRate(),args...;wrt=2) # reverse order

    # compute second moment of observable and its gradient
    E_τ2, ∇E_τ2 = compute_grad_qoi2(p.θ, qoi, pathint, args...)

    if mq == nothing
        mq == E_τ2
    end

    # return sqrt(T)/4 * ( ∇E_τ2 .* sqrt(E_d) ./ sqrt(E_τ2) .+ (sqrt(mq) + sqrt(E_τ2)) * ∇E_d ./ sqrt(E_d) )
    return T/2 * (∇E_τ2 .* E_d .+ (mq .+ E_τ2) .* ∇E_d)
end

    
function RadonNikodym(
    b::Function,
    a::Function,
    σ::Union{Matrix, Real},
    Xpath::Vector,
    Wpath::Vector,
    dt::Real;
    sgn=1,
)
    u(x) = inv(σ) * (b(x) .- a(x))

    ito = ItoIntegrator(Xpath, Wpath)
    int1 = compute_integral(u, ito)

    riem = RiemannIntegrator(Xpath, dt)
    int2 = compute_integral(x -> u(x)' * u(x), riem)

    return exp(sgn * (int1 + 0.5 * int2))
end
