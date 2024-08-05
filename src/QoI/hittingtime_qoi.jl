## hitting time domains -------------------------------------------------

"""
HittingDomain

Abstract type defining domain in state space for computing the first hitting time 
"""
abstract type HittingDomain end


"""
struct IntervalDomain <: HittingDomain 

Defines bounds of the interval domain for 1D state space. 

# Arguments
- `bounds :: Vector{<:Real}`      : lower and upper bounds [ll, ul]

"""
struct IntervalDomain <: HittingDomain 
    bounds :: Vector{<:Real}
end


"""
struct CircularDomain <: HittingDomain

Defines circular domain for N-D state space. 

# Arguments
- `center :: Vector`        : coordinate of center point (N-D vector)
- `radius :: Real`          : circle radius

"""
struct CircularDomain <: HittingDomain
    center :: Vector
    radius :: Real
end


"""
struct RectangularDomain <: HittingDomain

Defines rectangular domain for N-D state space. 

# Arguments
- `bounds :: Vector`        : 2-dim. vector of uniform lower and upper bounds, or N-dim. vector of lower and upper bounds in each dimension
- `dim :: Real`             : dimension of state space (N)

"""
struct RectangularDomain <: HittingDomain
    bounds :: Vector
    dim :: Real
end
RectangularDomain(bounds::Vector) = RectangularDomain(bounds, length(bounds))
RectangularDomain(bounds::Vector, dim::Real) = RectangularDomain([bounds for i=1:dim], dim)


"""
function hits_domain(x::Union{Real, Vector{<:Real}, D::HittingDomain)

Returns bool of whether the point x falls within the hitting domain.

# Arguments
- `x::Union{Real, Vector{<:Real}`        : current state in the simulation trajectory
- `D::HittingDomain`                     : hitting time domain

"""
hits_domain(x::Real, D::IntervalDomain) = D.bounds[1] <= x <= D.bounds[2]

hits_domain(x::Vector{<:Real}, D::CircularDomain) = norm(x - D.center) <= D.radius

function hits_domain(x::Vector{<:Real}, D::RectangularDomain)
    hit = Vector{Bool}(undef, D.dim)
    for d = 1:D.dim
        lb, ub = D.bounds[d]
        hit[d] = lb <= x[d] <= ub
    end
    return any(hit)
end

## compute first hitting time -------------------------------------------------

# for one path realization - 1D
function compute_hitting_time(
    x0::Real,                               # initial position
    sampler::ULA,                           # sampling algorithm
    s::Function,                            # score/drift function
    D::IntervalDomain,                      # hitting time domain
    T::Real                                 # simulation time length;
)
    # initialize trajectory
    xsim = zeros(T); xsim[1] = x0
    ξsim = zeros(T)
    for t = 2:T
        # simulate
        xsim[t], ξsim[t] = propose(xsim[t-1], s, sampler)
        # evaluate exit time criteria
        if hits_domain(xsim[t], D)
            return t, xsim[1:t], ξsim[1:t]
        elseif t == T
            return NaN, xsim, ξsim
        end
    end
end

# for one path realization - multi-D
function compute_hitting_time(
    x0::Vector{<:Real},                             # initial position
    sampler::ULA,                                   # sampling algorithm
    s::Function,                                    # score/drift function
    D::Union{CircularDomain, RectangularDomain},    # hitting time domain
    T::Real                                         # simulation time length;
)
    d = length(x0)
    # initialize trajectory
    xsim = [zeros(d) for t=1:T]; xsim[1] = x0
    ξsim = [zeros(d) for t=1:T]
    for t = 2:T
        # simulate
        xsim[t], ξsim[t] = propose(xsim[t-1], s, sampler)
        # evaluate exit time criteria
        if hits_domain(xsim[t], D)
            return t, xsim[1:t], ξsim[1:t]
        elseif t == T
            return NaN, xsim, ξsim
        end
    end
end


# for N path realizations
function compute_hitting_time(
    x0::Union{Real, Vector{<:Real}},        # initial position
    sampler::ULA,                           # sampling algorithm
    s::Function,                            # score/drift function
    D::HittingDomain,                       # hitting time domain
    T::Real,                                # simulation time length
    N::Real                                 # number of path realizations
)
    texit = zeros(N)
    xsim = Vector{Vector}(undef, N)
    ξsim = Vector{Vector}(undef, N)
    for n = 1:N
        # ξt = rand(Normal(0,sqrt(dt)), 1000)
        texit[n], xsim[n], ξsim[n] = compute_hitting_time(x0, sampler, s, D, T)
    end
    return texit, xsim, ξsim
end


## PDE solution to first hitting time -------------------------------------------------------
function feynmankac_1d(
    xgrid::Vector,      
    gradV::Vector,
    dead_ids::Vector,
    alive_ids::Vector,
    bc::String,
    costfcn::Function,
    β::Real,
    σ::Real,
)
    dx = xgrid[2] - xgrid[1]
    npts = length(xgrid)

    # create infinitesimal generator
    Δ = SymTridiagonal(-2 .* ones(npts), ones(npts-1)) ./ dx^2 
    ∇V_dot_∇ = Matrix((Tridiagonal(gradV[1:end-1], zeros(npts), -gradV[2:end]) ./ (2*dx))')

    # apply boundary conditions
    L = apply_boundary_cond(Δ, ∇V_dot_∇, bc, β)

    # initialize solution vector ψ
    ψ = ones(npts)
    deadset = xgrid[dead_ids]
    aliveset = xgrid[alive_ids]
    ψ[dead_ids] = exp.(-σ .* costfcn.(deadset))
    f = costfcn.(aliveset)

    # solve PDE over aliveset 
    L_alive = L[alive_ids, alive_ids]
    L_ad = L[alive_ids, dead_ids]
    ψ[alive_ids] = (L_alive - σ .* Diagonal(f)) \ (-L_ad * ψ[dead_ids])

    # compute free energy
    F = -log.(ψ) / σ

    return ψ, F
end



function apply_boundary_cond(
    laplacian::Matrix,
    dV_dot_grad::Matrix,
    bc::String,
    β::Real,
)
    ϵ = 1 / β # temperature

    if bc == "periodic"
        laplacian[1,end] = laplacian[1,2]
        laplacian[end,1] = laplacian[end,end-1]

        dV_dot_grad[1,end] = -dV_dot_grad[1,2]
        dV_dot_grad[end,1] = -dV_dot_grad[end,end-1]

        infgen = ϵ .* laplacian - dV_dot_grad

    elseif bc == "absorbing"
        infgen = ϵ .* laplacian - dV_dot_grad
        infgen[1,:] .= 0.0
        infgen[1,1] = 1.0
        infgen[end,:] .= 0.0
        infgen[end,end] = 1.0
    
    elseif bc == "reflecting"
        infgen = ϵ .* laplacian - dV_dot_grad
        infgen[1,1] = -infgen[1,2]
        infgen[end,:] .= 0.0
        infgen[end,end] = 1.0

    elseif bc == "l_reflecting_r_absorbing"
        infgen = ϵ .* laplacian - dV_dot_grad
        infgen[1,1] = -infgen[1,2]
        infgen[end,:] .= 0.0
        infgen[end,end] = 1.0

    elseif bc == "r_reflecting_l_absorbing"
        infgen = ϵ .* laplacian - dV_dot_grad
        infgen[1,:] .= 0.0
        infgen[1,1] = 1.0
        infgen[end,end] = -infgen[end,end-1]
    else
        println("ERROR: Specified boundary conditions not supported.")
    end

    return infgen
end


struct FeynmanKac1D
    bc :: String            # type of boundary condition
    domain :: Vector        # domain over which to compute PDE solution
    dx :: Real              # spatial discretization
    σ :: Real               # scale parameter
    β :: Real               # inverse temperature
end



## compute statistics of first hitting time -------------------------------------------------

"""
@kwdef mutable struct HittingTimeQoI <: QoI

Defines quantities for computing statistics of the first hitting time, in particular: 
mean and variance of the first hitting time τ over the transient path measure P up to time τ, e. g. E_P[τ(θ)] and Var_P[τ(θ)]
 

# Arguments
- `x0 :: Union{Real, Vector{<:Real}} `      : initial state of the stochastic process
- `D :: HittingDomain`                      : hitting domain
- `s :: Function`                           : score (drift) function of the stochastic process

"""
@kwdef mutable struct HittingTimeQoI <: QoI
    x0 :: Union{Real, Vector{<:Real}}           # initial state
    D :: HittingDomain                          # hitting domain
    s :: Function                               # score (drift) function
end

function assign_param(qoi::HittingTimeQoI, θ::Vector)
    return HittingTimeQoI(x0=qoi.x0, D=qoi.D, s = x -> qoi.s(x, θ))
end


function compute_qoi(
                qoi::HittingTimeQoI,
                integrator::MCPaths,
)
    (; x0, D, s) = qoi
    (; M, n, sampler, ρ0) = integrator

    τ, _, _ = compute_hitting_time(x0, sampler, s, D, n, M)
    
    res = (
        mean = mean(τ .* sampler.ϵ),
        var = var(τ .* sampler.ϵ),
    )
    return res
end


function compute_qoi(
    qoi::HittingTimeQoI,
    integrator::FeynmanKac1D,
)
    (; x0, D, s) = qoi
    (; bc, domain, dx, σ, β) = integrator
    if domain[2] != D.bounds[1]
        throw(ArgumentError("PDE domain not compatible with hitting time domain."))
    end
    lb, rb = domain
    
    xgrid = Vector(lb:dx:rb)
    dead_ids = [length(xgrid)]
    alive_ids = Vector(1:(length(xgrid)-1))

    costfcn(x) = integrator.β * (x < rb)
    gradV = s.(xgrid)

    ψplus, Fplus = feynmankac_1d(xgrid, gradV, dead_ids, alive_ids, bc, costfcn, β, σ)
    ψ0, F0 = feynmankac_1d(xgrid, gradV, dead_ids, alive_ids, bc, costfcn, β, 1e-16)
    ψminus, Fminus = feynmankac_1d(xgrid, gradV, dead_ids, alive_ids, bc, costfcn, β, -σ)

    # centered difference approximation of derivatives
    τmean = -(1/β) * (ψplus - ψminus) / (2 * σ) # first moment of τ
    τ2 = (1/β)^2 * (ψplus - 2*ψ0 + ψminus) / σ^2 # second moment of τ
    τvar = τ2 - τmean.^2

    x0id = findall(x -> x == x0, xgrid)[1]

    res = (
        mean = τmean[x0id],
        var = τvar[x0id],
        mean_vec = τmean,
        var_vec = τvar,
        x_vec = xgrid,
    )
    return res
end