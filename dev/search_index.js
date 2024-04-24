var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = AtomisticQoIs","category":"page"},{"location":"#AtomisticQoIs","page":"Home","title":"AtomisticQoIs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for AtomisticQoIs.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [AtomisticQoIs]","category":"page"},{"location":"#AtomisticQoIs.AtomisticQoIs","page":"Home","title":"AtomisticQoIs.AtomisticQoIs","text":"AtomisticQoIs\n\nAuthor: Joanna Zou <jjzou@mit.edu> Version: 0.1.0 Year: 2024 Notes: A Julia library for computing expectation-valued quantities of interest (QoIs) from molecular dynamics simulation.\n\n\n\n\n\n","category":"module"},{"location":"#AtomisticQoIs.GaussQuadrature","page":"Home","title":"AtomisticQoIs.GaussQuadrature","text":"struct GaussQuadrature <: QuadIntegrator\n\nContains parameters for 1D or 2D Gauss quadrature integration.\n\nArguments\n\nξ :: Union{Vector{<:Real}, Vector{<:Vector}}  : quadrature points\nw :: Union{Vector{<:Real}, Vector{<:Vector}}  : quadrature weights\nd :: Real                                     : dimension\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.GibbsQoI","page":"Home","title":"AtomisticQoIs.GibbsQoI","text":"@kwdef mutable struct GibbsQoI <: QoI\n\nDefines the struct for computing the quantity of interest (QoI) as the expected value of a function h over Gibbs measure p given input parameters θ, e. g. E_p[h(θ)]\n\nArguments\n\nh :: Function   : function to evaluate expectation, receiving inputs (x,θ)\np :: Gibbs      : Gibbs (invariant) distribution with θ == nothing (undefined parameters)\n∇h :: Union{Function, Nothing} = nothing   : gradient of h (if nothing, gradient is computed using AutoDiff)\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.HMC","page":"Home","title":"AtomisticQoIs.HMC","text":"struct HMC <: Sampler\n\nDefines the struct with parameters of the Hamiltonian Monte Carlo algorithm for sampling.\n\nArguments\n\nL :: Int        : time length of integrating Hamiltonian's equations\nϵ :: Real       : step size\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.ISMC","page":"Home","title":"AtomisticQoIs.ISMC","text":"struct ISMC <: ISIntegrator\n\nContains parameters for Importance Sampling (IS) integration using MC samples from a biasing distribution. This method is implemented when the biasing distribution g can be analytically sampled using rand().\n\nArguments\n\ng :: Distribution      : biasing distribution\nn :: Int               : number of samples\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.ISMCMC","page":"Home","title":"AtomisticQoIs.ISMCMC","text":"struct ISMCMC <: ISIntegrator\n\nContains parameters for Importance Sampling (IS) integration using MCMC samples from a biasing distribution. This method is implemented when the biasing distribution g cannot be analytically sampled.\n\nArguments\n\ng :: Distribution      : biasing distribution\nn :: Int               : number of samples\nsampler :: Sampler     : type of sampler (see Sampler)\nρ0 :: Distribution     : prior distribution of the state\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.ISMixSamples","page":"Home","title":"AtomisticQoIs.ISMixSamples","text":"struct ISMixSamples <: ISIntegrator\n\nContains parameters for Importance Sampling (IS) integration using samples from a mixture biasing distribution. The mixture weights are computed based on a kernel distance metric of the parameter with respect to parameters of the mixture component distributions. This method is implemented with the user providing samples from each component mixture distribution. \n\nArguments\n\ng :: MixtureModel        : mixture biasing distribution\nn :: Int                 : number of samples\nknl :: Kernel            : kernel function to compute mixture weights\nxsamp :: Vector          : Vector of sample sets from each component distribution\nnormint :: Integrator    : Integrator for the approximating the normalizing constant of each component distribution\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.ISSamples","page":"Home","title":"AtomisticQoIs.ISSamples","text":"struct ISSamples <: ISIntegrator\n\nContains pre-defined Monte Carlo samples from the biasing distribution to evaluate in integration. This method is implemented with the user providing samples from the biasing distribution. \n\nArguments\n\ng :: Distribution                               : biasing distribution\nxsamp :: Vector                                 : fixed set of samples\nnormint :: Union{Integrator, Nothing}           : integrator for computing normalizing constant of biasing distribution (required for mixture models)\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.Integrator","page":"Home","title":"AtomisticQoIs.Integrator","text":"Integrator\n\nA struct of abstract type Integrator computes the expectation of a function h(x, θ) with respect to an invariant measure p(x, θ).\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.MALA","page":"Home","title":"AtomisticQoIs.MALA","text":"struct MALA <: Sampler\n\nDefines the struct with parameters of the Metropolis-Adjusted Langevin Algorithm (MALA) for sampling.\n\nArguments\n\nL :: Int        : time length between evaluating accept/reject criterion\nϵ :: Real       : step size\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.MCMC","page":"Home","title":"AtomisticQoIs.MCMC","text":"struct MCMC <: MCIntegrator\n\nContains parameters for Monte Carlo (MC) integration using MCMC samples. This method is implemented when the distribution cannot be analytically sampled.\n\nArguments\n\nn :: Int               : number of samples\nsampler :: Sampler     : type of sampler (see Sampler)\nρ0 :: Distribution     : prior distribution of the state\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.MCSamples","page":"Home","title":"AtomisticQoIs.MCSamples","text":"struct MCSamples <: MCIntegrator\n\nContains pre-defined Monte Carlo samples to evaluate in integration. This method is implemented with the user providing samples from the distribution.\n\nArguments\n\nxsamp :: Vector    : fixed set of samples\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.MonteCarlo","page":"Home","title":"AtomisticQoIs.MonteCarlo","text":"struct MonteCarlo <: MCIntegrator\n\nContains parameters for Monte Carlo (MC) integration using MC samples. This method may only be implemented when the distribution can be analytically sampled using rand().\n\nArguments\n\nn :: Int   : number of samples\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.NUTS","page":"Home","title":"AtomisticQoIs.NUTS","text":"struct NUTS <: Sampler\n\nDefines the struct with parameters of the No-U-Turn-Sampler algorithm for sampling.\n\nArguments\n\nϵ :: Union{<:Real, Nothing}        : step size\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.QoI","page":"Home","title":"AtomisticQoIs.QoI","text":"QoI\n\nA struct of abstract type QoI computes the expectation of a function h(x, θ) with respect to an invariant measure p(x, θ).\n\n\n\n\n\n","category":"type"},{"location":"#AtomisticQoIs.EffSampleSize-Tuple{GibbsQoI, Vector}","page":"Home","title":"AtomisticQoIs.EffSampleSize","text":"function EffSampleSize(q::GibbsQoI, xsamp::Vector)\n\nComputes the effective sample size of a set of nMC samples from an MCMC chain. \n\nArguments\n\nq :: GibbsQoI     : Gibbs QoI struct\nxsamp :: Vector   : nMC-vector of x-samples\n\nOutputs\n\nESS :: Float64    : effective sample size (where 1 < ESS < nMC)\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.EffSampleSize-Tuple{Vector}","page":"Home","title":"AtomisticQoIs.EffSampleSize","text":"function EffSampleSize(hsamp::Vector)\n\nComputes the effective sample size of a set of nMC samples from an MCMC chain. \n\nArguments\n\nhsamp :: Vector   : nMC-vector of d-dimensional function evaluation h\n\nOutputs\n\nESS :: Float64    : effective sample size (where 1 < ESS < nMC)\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.ISWeightDiagnostic-Tuple{Vector}","page":"Home","title":"AtomisticQoIs.ISWeightDiagnostic","text":"function ISWeightDiagnostic(wsamp::Vector)\n\nReturns a diagnostic value for importance sampling weights using an unnormalized biasing distribution. The importance sampling estimate is reliable when the diagnostic is less than 5. \n\nArguments\n\nwsamp :: Vector   : nMC-vector of 1-dim. weights\n\nOutputs\n\ndiagnostic :: Float64    : diagnostic value (want < 5)\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.ISWeightVariance-Tuple{Vector}","page":"Home","title":"AtomisticQoIs.ISWeightVariance","text":"function ISWeightVariance(wsamp::Vector)\n\nReturns the variance of importance sampling weights. \n\nArguments\n\nwsamp :: Vector   : nMC-vector of weights\n\nOutputs\n\nvar(wsamp) :: Float64    : variance of weights\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.MCSE-Tuple{GibbsQoI, Vector}","page":"Home","title":"AtomisticQoIs.MCSE","text":"function MCSE(q::GibbsQoI, xsamp::Vector)\n\nEstimates the Monte Carlo standard error (se = std(h(x))/√n) using an empirical estimate of standard deviation.\n\nArguments\n\nq :: GibbsQoI     : Gibbs QoI struct\nxsamp :: Vector   : vector of samples\n\nOutputs\n\nse :: Float64     : Monte Carlo standard error\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.MCSEbm-Tuple{GibbsQoI, Vector}","page":"Home","title":"AtomisticQoIs.MCSEbm","text":"function MCSEbm(q::GibbsQoI, xsamp::Vector)\n\nEstimates the Monte Carlo standard error (se = std(h(x))/√n) using the batch means (BM) method.\n\nArguments\n\nq :: GibbsQoI     : Gibbs QoI struct\nxsamp :: Vector   : vector of samples\n\nOutputs\n\nse :: Float64     : Monte Carlo standard error\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.MCSEobm-Tuple{GibbsQoI, Vector}","page":"Home","title":"AtomisticQoIs.MCSEobm","text":"function MCSEbm(q::GibbsQoI, xsamp::Vector)\n\nEstimates the Monte Carlo standard error (se = std(h(x))/√n) using the overlapping batch means (OBM) method.\n\nArguments\n\nq :: GibbsQoI     : Gibbs QoI struct\nxsamp :: Vector   : vector of samples\n\nOutputs\n\nse :: Float64     : Monte Carlo standard error\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.compute_metrics-Tuple{Vector, Vector, Vector}","page":"Home","title":"AtomisticQoIs.compute_metrics","text":"function compute_metrics(θ::Vector, h::Vector, w::Vector)\n\nComputes importance sampling diagnostic metrics, including variance of IS weights (\"wvar\"), ESS of IS weights (\"wESS\"), and diagnostic measure (\"wdiag\").\n\nArguments\n\nθ :: Vector                 : vector of parameters\nh :: Vector                 : evaluations of summand function h(x) at samples\nw :: Vector                 : evaluations of importance sampling weights at samples\n\nOutputs\n\nmetrics :: Dict{String, Vector}        : dictionary of importance sampling diagnostics\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.expectation_is_stable-Tuple{Vector, Function, Gibbs, Distributions.Distribution}","page":"Home","title":"AtomisticQoIs.expectation_is_stable","text":"function expectationisstable(xsamp::Vector, ϕ::Function, f::Gibbs, g::Distribution; normint=nothing)\n\nHelper function for computing stable importance sampling weights (using log formulation).\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.gaussquad-Tuple{Integer, Function, Vararg{Any}}","page":"Home","title":"AtomisticQoIs.gaussquad","text":"function gaussquad(nquad::Integer, gaussbasis::Function, args...; limits::Vector=[-1,1])\n\nComputes 1D Gauss quadrature points with a change of domain to limits=[ll, ul].\n\nArguments\n\nnquad::Integer          : number of quadrature points\ngaussbasis::Function    : polynomial basis function from FastGaussQuadrature (gausslegendre, gaussjacobi, etc.)\nargs...                 : additional arguments for the gaussbasis() function\nlimits::Vector          : defines lower and upper limits for rescaling\n\nOutputs\n\nξ :: Vector{<:Real}     : quadrature points ranging [ll, ul]\nw :: Vector{<:Real}     : quadrature weights\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.gaussquad_2D-Tuple{Integer, Function, Vararg{Any}}","page":"Home","title":"AtomisticQoIs.gaussquad_2D","text":"function gaussquad_2D(nquad::Integer, gaussbasis::Function, args...; limits::Vector=[-1,1])\n\nComputes 2D (symmetric) Gauss quadrature points with a change of domain to limits=[ll, ul].\n\nArguments\n\nnquad::Integer          : number of quadrature points\ngaussbasis::Function    : polynomial basis function from FastGaussQuadrature (gausslegendre, gaussjacobi, etc.)\nargs...                 : additional arguments for the gaussbasis() function\nlimits::Vector          : defines lower and upper limits for rescaling\n\nOutputs\n\nξ :: Vector{<:Real}     : quadrature points ranging [ll, ul]\nw :: Vector{<:Real}     : quadrature weights\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.grad_expectation-Tuple{Union{Real, Vector{<:Real}}, GibbsQoI, Integrator}","page":"Home","title":"AtomisticQoIs.grad_expectation","text":"function grad_expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::Integrator; gradh::Union{Function, Nothing}=nothing)\n\nComputes the gradient of the QoI with respect to the parameters θ, e. g. ∇θ E_p[h(θ)]. See expectation for more information.\n\nArguments\n\nθ :: Union{Real, Vector{<:Real}}    : parameters to evaluate expectation\nqoi :: QoI                          : QoI object containing h and p\nintegrator :: Integrator            : struct specifying method of integration\ngradh :: Union{Function, Nothing}   : gradient of qoi.h; if nothing, compute with ForwardDiff\n\nOutputs\n\nexpec_estimator :: Vector{<:Real}   : estimate of vector-valued gradient of the expectation\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.hasapproxnormconst-Tuple{Gibbs}","page":"Home","title":"AtomisticQoIs.hasapproxnormconst","text":"function hasapproxnormconst(d::Distribution)\n\nReturns 'true' if distribution d requires an approximation of its normalizing constant.\n\nArguments\n\nd :: Distribution    : distribution to check\n\nOutputs\n\nflag :: Bool        : true or false\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.hasupdf-Tuple{Gibbs}","page":"Home","title":"AtomisticQoIs.hasupdf","text":"function hasupdf(d::Distribution)\n\nReturns 'true' if distribution d has an unnormalized pdf function (updf(), logupdf()).\n\nArguments\n\nd :: Distribution    : distribution to check\n\nOutputs\n\nflag :: Bool        : true or false\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.sample-Tuple{Function, Function, NUTS, Int64, Real}","page":"Home","title":"AtomisticQoIs.sample","text":"function sample(lπ::Function, gradlπ::Function, sampler::NUTS, n::Int64, x0::Real)\n\nDraw samples from target distribution π(x) using the No-U-Turn Sampler (NUTS) for scalar-valued support (x::Real).\n\nArguments\n\nlπ :: Function        : log likelihood of target π\ngradlπ :: Function    : gradient of log likelihood of π\nsampler :: HMC        : Sampler struct specifying sampling algorithm\nn :: Int64            : number of samples\nx0 :: Real            : initial state\n\nOutputs\n\nsamples :: Vector     : vector of samples from π\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.sample-Tuple{Function, Function, NUTS, Int64, Vector{<:Real}}","page":"Home","title":"AtomisticQoIs.sample","text":"function sample(lπ::Function, gradlπ::Function, sampler::NUTS, n::Int64, x0::Vector{<:Real})\n\nDraw samples from target distribution π(x) using the No-U-Turn Sampler (NUTS) from the AdvancedHMC package. Assumes multi-dimensional support (x::Vector).\n\nArguments\n\nlπ :: Function              : log likelihood of target π\ngradlπ :: Function          : gradient of log likelihood of π\nsampler :: HMC              : Sampler struct specifying sampling algorithm\nn :: Int64                  : number of samples\nx0 :: Vector{<:Real}        : initial state\n\nOutputs\n\nsamples::Vector{Vector}     : vector of samples from π\n\n\n\n\n\n","category":"method"},{"location":"#AtomisticQoIs.sample-Tuple{Function, Function, Union{HMC, MALA}, Int64, Real}","page":"Home","title":"AtomisticQoIs.sample","text":"function sample(lπ::Function, gradlπ::Function, sampler::Union{HMC,MALA}, n::Int64, x0::Real)\n\nDraw samples from target distribution π(x) for scalar-valued support (x::Real).\n\nArguments\n\nlπ :: Function                      : log likelihood of target π\ngradlπ :: Function                  : gradient of log likelihood of π\nsampler :: Union{HMC,MALA}          : Sampler struct specifying sampling algorithm\nn :: Int64                          : number of samples\nx0 :: Real                          : initial state\n\nOutputs\n\nsamples :: Vector                   : vector of samples from π\n\n\n\n\n\n","category":"method"}]
}
