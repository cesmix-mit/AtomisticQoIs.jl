struct ULA <: Sampler
    ϵ :: Real   # step size
    β :: Real   # inverse temperature
end


# performs a single Euler-Maruyama step
function propose(x0::Real, gradlπ::Function, ula::ULA)
    ξt = rand(Normal(0,sqrt(ula.ϵ))) 
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
    xsim = zeros(T); xsim[1] = x0
    ξsim = zeros(T)

    for t = 2:T
        # simulate
        xsim[t], ξsim[t] = propose(xsim[t-1], gradlπ, sampler)
    end
    return xsim, ξsim
end


function sample(gradlπ::Function, sampler::ULA, T::Int64, x0::Vector{<:Real})
    d = length(x0)
    xsim = [zeros(d) for t = 1:T]; xsim[1] = x0
    ξsim = [zeros(d) for t = 1:T]
    for t = 2:T
        # simulate
        xsim[t], ξsim[t] = propose(xsim[t-1], gradlπ, sampler)
    end
    return xsim, ξsim
end

sample(lπ::Function, gradlπ::Function, sampler::ULA, T::Int64, x0::Union{Real, Vector{<:Real}}) = sample(gradlπ, sampler, T, x0)