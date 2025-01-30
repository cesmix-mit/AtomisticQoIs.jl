using AtomisticQoIs
using Distributions
using LinearAlgebra
using FastGaussQuadrature
using Test

@testset "computing QoIs" begin

    # 1D test case
    V(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    ∇xV(x, θ) = - θ[1] * x + θ[2] * x.^3
    ∇θV(x, θ) = [-x^2 / 2, x^4 / 4]

    πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    q0 = GibbsQoI(h=V, p=πgibbs)
    q = GibbsQoI(h=V, p=πgibbs, ∇h=∇θV)
    ρx0 = Uniform(-2, 2)
    θtest = [2.0, 4.0]

    ## one-point invariant statistics
    # GibbsQoI with quadrature integration
    ξi, wi = gaussquad(100, gaussjacobi, 0.5, 0.5, limits=[-10, 10])
    ξi, wi = gaussquad(100, gausslegendre, limits=[-10, 10])

    GQint = GaussQuadrature(ξi, wi)
    Qquad = compute_qoi(θtest, q, GQint).mean
    ∇Qquad = compute_grad_qoi(θtest, q0, GQint).mean # using ForwardDiff
    ∇Qquad2 = compute_grad_qoi(θtest, q, GQint).mean # using pre-defined gradient
    @test ∇Qquad ≈ ∇Qquad2

    # GibbsQoI with MCMC sampling 
    nsamp = 10000
    nuts = NUTS(1e-2)
    MCint = MCMC(nsamp, nuts, ρx0)
    Qmc = compute_qoi(θtest, q, MCint).mean
    @time ∇Qmc = compute_grad_qoi(θtest, q, MCint).mean

    # GibbsQoI with importance sampling
    # draw MCMC samples
    g = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=0.2, θ=[3,3])
    ISint = ISMCMC(g, nsamp, nuts, ρx0)
    Qis1 = compute_qoi(θtest, q, ISint).mean
    @time ∇Qis1 = compute_grad_qoi(θtest, q, ISint).mean 

    # use fixed samples
    xsamp = rand(g, nsamp, nuts, ρx0)
    ISint2 = ISSamples(g, xsamp)
    Qis2 = compute_qoi(θtest, q, ISint2).mean
    @time ∇Qis2 = compute_grad_qoi(θtest, q, ISint2).mean

    # use uniform biasing distribution
    πu = Uniform(-5,5)
    ISUint = ISMC(πu, nsamp)
    Qis3 = compute_qoi(θtest, q, ISUint).mean
    @time ∇Qis3 = compute_grad_qoi(θtest, q, ISUint).mean

    # use fixed samples from biasing distribution
    xsamp = rand(πu, nsamp)
    ISUint2 = ISSamples(πu, xsamp)
    Qis4 = compute_qoi(θtest, q, ISUint2).mean
    @time ∇Qis4 = compute_grad_qoi(θtest, q, ISUint2).mean

    # test with magnitude of error 
    @test abs((Qquad - Qmc)/Qquad) <= 0.1
    @test abs((Qquad - Qis1)/Qquad) <= 0.1
    @test abs((Qquad - Qis2)/Qquad) <= 0.1
    @test abs((Qquad - Qis3)/Qquad) <= 0.1
    @test abs((Qquad - Qis4)/Qquad) <= 0.1

    @test norm((∇Qquad - ∇Qmc)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis1)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis2)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis3)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis4)./∇Qquad) <= 1.0


    ## first hitting time statistics
    D = IntervalDomain([0.5, Inf])
    q = HittingTimeQoI(x0=0.0, D=D, s=∇xV)
    β = 2.0
    
    # HittingTimeQoI by Feynman-Kac PDE
    fk = FeynmanKac1D(
        "l_reflecting_r_absorbing",
        [-3, D.bounds[1]], # domain
        0.001, # dx
        1e-4, # σ
        β, # β
    )

    τpde = compute_qoi(θtest, q, fk).mean

    # HittingTimeQoI by Monte Carlo
    ula = ULA(0.001, β)
    mc = MCPaths(
        1000, # number of paths
        1_000_000, # T
        ula,
        0.0, # x0
    )
    
    τmc = compute_qoi(θtest, q, mc).mean

    # test with magnitude of error 
    @test abs((τpde - τmc)/τpde) <= 0.1


    ## two-point invariant statistics
    Id(x) = x
    lag = 1000
    q = AutocorrelationQoI(H=Id, lag=lag, s=∇xV)

    κmc1 = compute_qoi(θtest, q, mc; burnin=1000).mean
    κmc2 = compute_qoi(θtest, q, mc; burnin=1000).mean

    @test abs((κmc1 - κmc2)/κmc2) <= 0.1
    
end

@testset "computing QoI with 2D quadrature" begin

    V(x, θ) = 1/6 * (4*(1-x[1]^2-x[2]^2)^2
            + 2*(x[1]^2-2)^2
            + ((x[1]+x[2])^2-1)^2
            + ((x[1]-x[2])^2-1)^2)
        
    ∇xV(x, θ) = [4/3 * x[1] *(4*x[1]^2 + 5*x[2]^2 - 5),
                4/3 * x[2] *(5*x[1]^2 + 3*x[2]^2 - 3)]

    ∇θV(x, θ) = 0 

    πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    q = GibbsQoI(h=V, p=πgibbs, ∇h=∇θV)
    ρx0 = MvNormal(zeros(2), I(2))
    θtest = [2.0, 4.0]

    ξi, wi = gaussquad_2D(25, gausslegendre, limits=[-10, 10])
    GQint = GaussQuadrature(ξi, wi)
    Qquad = compute_qoi(θtest, q, GQint).mean
    ∇Qquad = compute_grad_qoi(θtest, q, GQint).mean

    πg = Gibbs(πgibbs, θ=θtest)
    Z = normconst(πg, GQint)


end 


 