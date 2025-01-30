"""
struct MCPaths <: PathIntegrator

Contains parameters for generating path realizations to compute Monte Carlo averages over path measures. 

# Arguments
- `M :: Int`               : number of paths
- `n :: Int`               : max number of samples per path
- `sampler :: Sampler`     : type of sampler (see `Sampler`)
- `ρ0 :: Union{Distribution, Real, Vector{<:Real}}`     : initial state or prior distribution of the state

"""
struct MCPaths <: PathIntegrator
    M :: Int
    n :: Int
    sampler :: Sampler
    ρ0 :: Union{Distribution, Real, Vector{<:Real}}
end

struct MCPathSamples <: PathIntegrator
    Xpaths :: Vector
    Wpaths :: Vector
    dt :: Real # time step size
    β :: Real # inverse temp.
    N :: Real # max number of time steps
end

struct RDPathSamples <: PathIntegrator
    s :: Function
    Xpaths :: Vector
    Wpaths :: Vector
    dt :: Real # time step size
    β :: Real # inverse temp.
    N :: Real # max number of time steps
end

struct RiemannIntegrator <: PathIntegrator
    Xpath :: Vector
    dt :: Union{Real, Vector{<:Real}}
end

struct ItoIntegrator <: PathIntegrator
    Xpath::Vector
    Wpath::Vector
end

compute_integral(f::Function, int::RiemannIntegrator) = sum(f.(int.Xpath) .* int.dt)

compute_integral(f::Function, int::ItoIntegrator) =sum([f(X)' * W for (X,W) in zip(int.Xpath,int.Wpath)])
