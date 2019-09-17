__precompile__()

module simGLM
    using ForwardDiff
    using StaticArrays
    using LinearAlgebra: dot

    mutable struct PillowGLM
        M::Int
        N::Int
        dh::Int
        dt::Float64
        θ::Array
        n::Int
    end

    function PillowGLM(w::Array, h::Array, b::Array, data::Array, params::Dict)
        M = size(data)[2]
        N = params["numNeurons"]
        dh = params["hist_dim"]
        dt = params["dt"]
        θ = [w[:]; h[:]; b[:]]
        PillowGLM(M, N, dh, dt, θ, 0)
    end

    function ll(o::PillowGLM, data)
        ll(o.θ, o, data)
    end

    function ll(θ, o::PillowGLM, data)
        expo = ones(eltype(θ), o.N, o.M)

        for j in 1:o.M  # Time
            expo[:,j] = runModelStep(θ, o, data[:, 1:j])
        end

        r̂ = o.dt.*exp.(expo)

        return (sum(r̂) + sum(data .* log.(r̂)))  # Log-likelihood
    end

    function ll_grad(o::PillowGLM, data)
        o.n += 1
        return ForwardDiff.gradient(θ -> ll(θ, o, data), o.θ)
    end

    function runModelStep(θ, o::PillowGLM, data)
        # Since theta is lumped in, separate.
        w = reshape(θ[1: o.N*o.N], o.N, o.N)
        h = reshape(θ[o.N*o.N+1: o.N*(o.N+o.dh)], o.N, o.dh)
        b = reshape(θ[o.N*(o.N+o.dh)+1: end], o.N)

        t = size(data)[2] # data length in time

        output = zeros(eltype(θ), o.N)
        for i in 1:o.N  # Neuron
            if t < 1
                hist = 0
            elseif t < o.dh
                hist = sum(reverse(h,dims=2)[i, 1:t] .* data[i, 1: t])
            else
                hist = sum(reverse(h,dims=2)[i, :] .* data[i, t-o.dh+1: t])
            end

            if t < 2
                weights = 0
            else
                weights = dot(w[i,:], data[:, t-1])
            end
            output[i] = b[i] + hist + weights
        end

        return output
    end

    function fit!(o::PillowGLM, data, iter; rate=i -> 1/i)
        o.θ -= simGLM.ll_grad(o, data) * rate(o.n)
    end
end

using Random
Random.seed!(82)

N = 3
M = 20
dh = 3
p = Dict("numSamples" => M, "numNeurons" => N, "hist_dim" => dh, "dt" => 0.1)

weights = rand(N, N)
hist = rand(N, dh)
base = rand(N)
data = randn(N, M) ./ 10

model = simGLM.PillowGLM(weights, hist, base, data, p)

println(simGLM.ll(model, data))
for i in 1:100
    simGLM.fit!(model, data)
end
println(simGLM.ll(model, data))
