struct ULA <: Sampler
    ϵ :: Real   # step size
    β :: Real   # inverse temperature
    seed :: Union{Integer, Nothing}
end
ULA(ϵ::Real, β::Real) = ULA(ϵ, β, nothing)

# performs a single Euler-Maruyama step
function propose(x0::Real, gradlπ::Function, ula::ULA)
    ξt = sqrt(ula.ϵ) * randn()
    xt = x0 - gradlπ(x0) * ula.ϵ + sqrt(2/ula.β) * ξt
    return xt, ξt
end


function propose(x0::Vector{<:Real}, gradlπ::Function, ula::ULA)
    d = length(x0)
    ξt = rand(MvNormal(zeros(d), ula.ϵ .* I(d)))
    xt = x0 .- gradlπ(x0) .* ula.ϵ + sqrt(2/ula.β) .* ξt
    return xt, ξt
end


# samples using ULA
function sample(gradlπ::Function, sampler::ULA, T::Int64, x0::Real)
    xsim = zeros(T+1); xsim[1] = x0
    ξsim = zeros(T)

    if sampler.seed != nothing
        Random.seed!(sampler.seed)
    end

    for t = 1:T
        # simulate
        xsim[t+1], ξsim[t] = propose(xsim[t], gradlπ, sampler)
    end
    return xsim[1:T], ξsim
end


function sample(gradlπ::Function, sampler::ULA, T::Int64, x0::Vector{<:Real})
    d = length(x0)
    xsim = [zeros(d) for t = 1:T+1]; xsim[1] = x0
    ξsim = [zeros(d) for t = 1:T]

    if sampler.seed != nothing
        Random.seed!(sampler.seed)
    end
    
    for t = 1:T
        # simulate
        xsim[t+1], ξsim[t] = propose(xsim[t], gradlπ, sampler)
    end
    return xsim[1:T], ξsim
end

sample(lπ::Function, gradlπ::Function, sampler::ULA, T::Int64, x0::Union{Real, Vector{<:Real}}) = sample(gradlπ, sampler, T, x0)