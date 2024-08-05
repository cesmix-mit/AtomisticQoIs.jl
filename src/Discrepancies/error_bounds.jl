abstract type ErrorBound end


struct StuartBound <: ErrorBound
    H :: Union{Hellinger, PathHellinger}
    bd :: Function
end
function StuartBound(Hel::Union{Hellinger, PathHellinger})
    bd(m1, m2, H) = 2 * sqrt(m1 + m2) * H
    return StuartBound(Hel, bd)
end


struct CuiBound <: ErrorBound
    H :: Union{Hellinger, PathHellinger}
    bd :: Function
end
function CuiBound(Hel::Union{Hellinger, PathHellinger})
    bd(v1, v2, H) = sqrt(2)*H/(1-sqrt(2)*H) * (sqrt(v1) + sqrt(v2))
    return CuiBound(Hel, bd)
end


function compute_error_bound(
    g1::GibbsQoI,
    g2::GibbsQoI,
    eb::ErrorBound,
)
    # compute mean and variance of observable
    res1 = compute_qoi(g1, eb.H.integrator)
    res2 = compute_qoi(g2, eb.H.integrator)

    # compute Hellinger distance
    Hel = compute_divergence(g1.p, g2.p, eb.H)

    return compute_error_bound(res1, res2, Hel, eb)
end


function compute_error_bound(
    τ1::HittingTimeQoI,
    τ2::HittingTimeQoI,
    eb::ErrorBound,
)
    # compute mean and variance of observable
    res1 = compute_qoi(τ1, eb.H.integrator)
    res2 = compute_qoi(τ2, eb.H.integrator)

    # compute Hellinger distance
    σ = 2 / eb.H.integrator.sampler.β
    Hel = compute_divergence(τ1.s, τ2.s, σ, ebh.H)

    return compute_error_bound(res1, res2, Hel, eb)
end


function compute_error_bound(
    res1::NamedTuple,
    res2::NamedTuple,
    H::Real,
    eb::CuiBound,
)
    # compute absolute error
    abserr = abs.(res1.mean .- res2.mean)

    # compute upper bound
    ubd = eb.bd(res1.var, res2.var, H)

    return abserr, ubd
end


function compute_error_bound(
    res1::NamedTuple,
    res2::NamedTuple,
    H::Real,
    eb::StuartBound,
)
    # compute absolute error
    abserr = abs.(res1.mean .- res2.mean)

    # compute second moments
    m1 = res1.var + res1.mean^2
    m2 = res2.var + res2.mean^2

    # compute upper bound
    ubd = eb.bd(m1, m2, H)

    return abserr, ubd
end