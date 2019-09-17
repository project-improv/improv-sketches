module simGLM
    using CatViews
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

    function copy!(from::Array, to::Union{SubArray, Base.ReshapedArray})
        if length(size(from)) > 1
            for i in size(from)[1]
                for j in size(from)[2]
                    to[i, j] = from[i, j]
                end
            end
        else
            for i in size(from)[1]
                to[i] = from[i]
            end
        end
    end

    function MakePillowGLM(w::Array, h::Array, b::Array, data::Array, params::Dict)
        M = size(data)[2]
        N = params["numNeurons"]
        dh = params["hist_dim"]
        dt = params["dt"]

        θ = [w[:]; h[:]; b[:]]

        # w = reshape(θ[1: N*N], N, N)
        # h = reshape(θ[N*N+1: N*(N+dh)], N, dh)
        # b = reshape(θ[N*(N+dh)+1: end], N)

        PillowGLM(M, N, dh, dt, θ, 0)
    end

    function ll(θ, model, data, params)
        # θ is w, h, b, data is y
        M = size(data)[2]  # params["numSamples"]
        N = params["numNeurons"]
        dt = params["dt"]

        expo = ones(eltype(θ), N, M)

        for j in 1:M  # Time
            expo[:,j] = runModelStep(θ, model, data[:, 1:j], params)
        end

        r̂ = dt.*exp.(expo)

        return (sum(r̂) + sum(data .* log.(r̂)))  # Log-likelihood
    end

    function runModelStep(θ, model, data, params)
        M = params["numSamples"]
        N = params["numNeurons"]
        dh = params["hist_dim"]

        # Since theta is lumped in, separate.
        w = reshape(θ[1: N*N], N, N)
        h = reshape(θ[N*N+1: N*(N+dh)], N, dh)
        b = reshape(θ[N*(N+dh)+1: end], N)

        t = size(data)[2] # data length in time

        output = zeros(eltype(θ), N)
        for i in 1:N  # Neuron
            if t < 1
                hist = 0
            elseif t < dh
                hist = sum(reverse(h,dims=2)[i, 1:t] .* data[i, 1: t])
            else
                hist = sum(reverse(h,dims=2)[i, :] .* data[i, t-dh+1: t])
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

    function ll_grad(x, model, data, params)
        function wrapper(x)
            return ll(x, model, data, params)
        end
        return ForwardDiff.gradient(n -> wrapper(n), x)
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
model = simGLM.MakePillowGLM(weights, hist, base, data, p)

# θ = [weights[:]; hist[:]; base[:]]
data = randn(N, M) ./ 10
println(simGLM.ll(model.θ, model, data, p))

for i in 1:100
    global model
    model.θ -= (1/i) * simGLM.ll_grad(model.θ, model, data, p)
end

println(simGLM.ll(model.θ, model, data, p))
