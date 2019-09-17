__precompile__()

module simGLM
    using StaticArrays
    using LinearAlgebra: dot
    using Flux
    using Flux.Tracker
    using Flux.Tracker: update!

    mutable struct PillowGLM
        M::Int
        N::Int
        dh::Int
        dt::Float64
        θ_b::TrackedArray
        θ_h::TrackedArray
        θ_w::TrackedArray
        n::Int  # Iterations
    end

    function PillowGLM(w::Array, h::Array, b::Array, data::Array, params::Dict)
        M = size(data)[2]
        N = params["numNeurons"]
        dh = params["hist_dim"]
        dt = params["dt"]
        θ_b, θ_h, θ_w = param(b), param(reverse(h, dims=2)), param(w)
        PillowGLM(M, N, dh, dt, θ_b, θ_h, θ_w, 0)
    end

    function ll(o::PillowGLM, data)
        expo = ones(eltype(o.θ_h), o.N, o.M)
        for j in 1:o.M
            expo[:,j] = runModelStep(o, data[:, 1:j])
        end

        r̂ = o.dt.*exp.(expo)

        return (sum(r̂) + sum(data .* log.(r̂)))  # Log-likelihood
    end

    function runModelStep(o::PillowGLM, data)
        t = size(data)[2] # data length in time

        output = zeros(eltype(o.θ_h), o.N)
        for i in 1:o.N  # Neuron
            if t < 1
                hist = 0
            elseif t < o.dh
                hist = sum(o.θ_h[i, 1:t] .* data[i, 1: t])
            else
                hist = sum(o.θ_h[i, :] .* data[i, t-o.dh+1: t])
            end

            if t < 2
                weights = 0
            else
                weights = dot(o.θ_w[i,:], data[:, t-1])
            end
            output[i] = o.θ_b[i] + hist + weights
        end

        return output
    end

    function fit!(o::PillowGLM, data; opt=Descent(0.1))
        o.n += 1
        grads = Tracker.gradient(() -> ll(o, data), Params([o.θ_b, o.θ_h, o.θ_w]))
        for p in (o.θ_b, o.θ_h, o.θ_w)
            update!(opt, p, grads[p])
        end
    end

end

using Flux
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
out = zeros(100)
for i in 1:100
    out[i] = simGLM.ll(model, data).data
    simGLM.fit!(model, data, opt = ADAM(0.01))
end
println(simGLM.ll(model, data))
