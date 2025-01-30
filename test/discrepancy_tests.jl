using AtomisticQoIs
using FastGaussQuadrature
using Distributions, LinearAlgebra
using Test

@testset "computing discrepancies" begin

    # test case
    V(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    ∇xV(x, θ) = - θ[1] * x + θ[2] * x.^3
    ∇θV(x, θ) = [-x^2 / 2, x^4 / 4]

    # densities
    β = 0.5
    p = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=β, θ=[3.0,3.0])
    q = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=β, θ=[1.0,5.0])
    σ = sqrt(2/β)

    # integration points
    ξi, wi = gaussquad(100, gausslegendre, limits=[-10, 10])
    GQint = GaussQuadrature(ξi, wi)

    nsamp = 10000
    nuts = NUTS(1e-2)
    ρx0 = Uniform(-2, 2)
    xsamp = rand(p, nsamp, nuts, ρx0)
    MCint = MCSamples(xsamp)


    ## Gibbs divergences
    # Hellinger
    Hpq = compute_divergence(p, q, GQint, Hellinger())
    Hqp = compute_divergence(q, p, GQint, Hellinger())

    @test Hpq > 0.0
    @test Hqp > 0.0
    @test abs(Hpq - Hqp) ./ Hpq <= 1.0
    @test compute_divergence(p, p, GQint, Hellinger()) < eps()
    
    # KL divergence
    Kpq = compute_divergence(p, q, GQint, KLDivergence())
    Kqp = compute_divergence(q, p, GQint, KLDivergence())

    @test Kpq > 0.0
    @test Kqp > 0.0
    @test Kpq != Kqp
    @test compute_divergence(p, p, GQint, KLDivergence()) < eps()

    @test 2*Hpq^2 <= Kpq
    @test 2*Hqp^2 <= Kqp

    # Fisher divergence
    Fpq = compute_divergence(p, q, GQint, FisherDivergence())
    Fqp = compute_divergence(q, p, GQint, FisherDivergence())

    @test Fpq > 0.0
    @test Fqp > 0.0
    @test Fpq != Fqp
    @test compute_divergence(p, p, GQint, FisherDivergence()) < eps()

    @test Kpq <= 0.5 * Fpq
    @test Kqp <= 0.5 * Fqp

    # Relative entropy rate
    Rpq = compute_divergence(p, q, GQint, RelEntropyRate(σ))
    Rqp = compute_divergence(q, p, GQint, RelEntropyRate(σ))



    ## Path divergences
    N = 100
    xsim = Vector{Vector}(undef, N)
    ξsim = Vector{Vector}(undef, N)

    ula = ULA(0.01, β)
    
    for n = 1:N
        xsim[n], ξsim[n] = AtomisticQoIs.sample(p.∇xV, ula, 500, 0.0)
    end

    # Path Hellinger
    Hel = PathHellinger(xsim, ξsim, 0.01)
    Hpq = compute_divergence(p, q, Hel)
    Hab = compute_divergence(p.∇xV, q.∇xV, σ, Hel)

    @test Hqp > 0.0
    @test Hpq == Hab
    @test compute_divergence(p, p, Hel) < eps()
    
    # Path KL
    KL = PathKL(xsim, 0.01)
    Kpq = compute_divergence(p, q, KL)
    Kab = compute_divergence(p.∇xV, q.∇xV, σ, KL)

    @test Kpq > 0.0
    @test Kpq == Kab
    @test compute_divergence(p, p, KL) < eps()

    # Path Fisher
    

end