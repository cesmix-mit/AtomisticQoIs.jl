import Base: @kwdef

""" 
    QoI

    Abstract type defining observable quantities of interest (QoI) from a stochastic system.
"""
abstract type QoI end

include("general_qoi.jl")
include("gibbs_qoi.jl")
include("hittingtime_qoi.jl")
include("autocorrelation_qoi.jl")


function compute_qoi(
    θ::Union{Real, Vector{<:Real}}, 
    qoi::QoI,
    integrator::Integrator;
    kwargs...)

    q = assign_param(qoi, θ)
    return compute_qoi(q, integrator; kwargs...)
end


export
    GibbsQoI,
    GeneralQoI,
    compute_qoi,
    compute_grad_qoi,
    compute_grad_qoi2,
    assign_param,
    HittingDomain,
    IntervalDomain,
    CircularDomain,
    EllipseDomain,
    RectangularDomain,
    hits_domain,
    compute_hitting_time,
    FeynmanKac1D,
    HittingTimeQoI,
    AutocorrelationQoI,
    GreenKuboQoI
