abstract type PathDivergence <: Divergence end

struct PathHellinger <: PathDivergence
    integrator :: MCPaths
    Xpaths :: Vector{<:Vector}
    Wpaths :: Vector{<:Vector}
    dt :: Real
end

struct PathKL <: PathDivergence
    integrator :: MCPaths
    Xpaths :: Vector{<:Vector}
    dt :: Real
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


function KLDivergence(
    b::Function,
    a::Function,
    σ::Union{Matrix, Real},
    D::PathKL,
)
    Npaths = length(D.Xpaths)

    u(x) = inv(σ) * (b(x) - a(x))
    riem = RiemannIntegrator(x -> u(x)' * u(x), D.dt)

    KL = 0
    for n = 1:Npaths
        KL += compute_integral(D.Xpaths[n], riem) 
    end

    return KL / (2*Npaths)
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
    u(x) = inv(σ) * (b(x) - a(x))

    ito = ItoIntegrator(Xpath, Wpath)
    int1 = compute_integral(u, ito)

    riem = RiemannIntegrator(Xpath, dt)
    int2 = compute_integral(x -> u(x)' * u(x), riem)

    return exp(sgn * (int1 + 0.5 * int2))
end
