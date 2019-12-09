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

    function PillowGLM(w::Array, h::Array, b::Array, params::Dict)
        M = params["numSamples"]
        N = params["numNeurons"]
        dh = params["hist_dim"]
        dt = params["dt"]
        θ_b, θ_h, θ_w = param(b), param(reverse(h, dims=2)), param(w)
        PillowGLM(M, N, dh, dt, θ_b, θ_h, θ_w, 0)
    end

    function ll(o::PillowGLM, data)
        expo = ones(Tracker.TrackedReal{Float64}, o.N, o.M)

        Threads.@threads for j in 1:o.M-o.dh
            expo[:,j] = runModelStep(o, data[:, j:j+o.dh])
        end

        r̂ = o.dt.*exp.(expo)

        (sum(r̂) + sum(data .* log.(r̂))) / o.M # Log-likelihood
    end

    function runModelStep(o::PillowGLM, data)
        t = size(data)[2] # data length in time
        @assert t == 3

        [o.θ_b[i] + dot(o.θ_h[i, :], data[i, 2:end]) + dot(o.θ_w[i, :], data[:, end-1]) for i in 1:o.N]
        # for i in 1:o.N  # Neuron
        #     hist = dot(o.θ_h[i, :], data[i, 2:t])
        #     weights = dot(o.θ_w[i,:], data[:, t-1])
        #     output[i] = o.θ_b[i] + hist + weights
        # end
    end

    function fit!(o::PillowGLM, data; opt=Descent(1e-3))#, rate=x -> 1/x)
        o.n += 1
        grads = Tracker.gradient(() -> ll(o, data), Params([o.θ_b, o.θ_h, o.θ_w]))
        for p in (o.θ_b, o.θ_h, o.θ_w)
            update!(opt, p, grads[p])
        end
    end
end

using DelimitedFiles
using Flux
using Random
using Plots

cd("C:\\Users\\Chaichontat\\Documents\\GitHub\\improv-sketches")
Random.seed!(82)
data = parse.(Float64, readdlm("data_sample.txt", ' ', String, '\n'));

N = size(data)[1]
M = 100
dh = 2
p = Dict("numSamples" => M, "numNeurons" => N, "hist_dim" => dh, "dt" => 0.1)

weights = zeros(N, N)
hist = zeros(N, dh)
base = zeros(N)

model = simGLM.PillowGLM(weights, hist, base, p)

iter = 900

out = zeros(iter)
mean_sq = zeros(iter, 3)

for i in 1:iter
    out[i] = simGLM.ll(model, data[:, mod(i,300)+1:mod(i,300)+99+1]).data
    if i > 1
        println(i, out[i])
    end
    mean_sq[i, 1] = Flux.mse(model.θ_b.data, d["b"])
    mean_sq[i, 2] = Flux.mse(model.θ_h.data, d["h"])
    mean_sq[i, 3] = Flux.mse(model.θ_w.data, d["w"])

    simGLM.fit!(model, data[:, mod(i,300)+1:mod(i,300)+99+1])
end
# plot(mean_sq[:, 1])
# plot!(mean_sq[:, 2])
plot(mean_sq[:, 1])
plot!(mean_sq[:, 2])
# plot(out)
