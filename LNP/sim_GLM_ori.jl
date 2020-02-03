module simGLM

using ForwardDiff
using StaticArrays
using LinearAlgebra: dot

function ll(theta, data, params)
    # theta is w, h, b, data is y
    M = size(data)[2] #params["numSamples"]
    N = params["numNeurons"]
    dt = params["dt"]

    expo = ones(eltype(theta), N, M)

    for j in range(1,stop=M)
        expo[:,j] = runModelStep(theta, data[:, 1:j], params)
    end

    rhat = dt.*exp.(expo)

    return 1e-6 * (sum(rhat)+sum(data.*log.(rhat)))
end

function runModelStep(theta, data, params)
    M = params["numSamples"]
    N = params["numNeurons"]
    dh = params["hist_dim"]

    w = reshape(theta[1:N*N], N, N)
    h = reshape(theta[N*N+1:N*(N+dh)], N, dh)
    b = reshape(theta[N*(N+dh)+1:end], N)

    t = size(data)[2] # data length in time

    expo = zeros(eltype(theta), N)
    for i in range(1, stop=N)
        if t<1
            hist = 0
        elseif t<dh
            hist = sum(reverse(h,dims=2)[i,1:t].*data[i,1:t])
        else
            hist = sum(reverse(h,dims=2)[i,:].*data[i,t-dh+1:t])
        end

        if t<2
            weights = 0
        else
            weights = dot(w[i,:], data[:,t-1])
        end
        expo[i] = b[i]+hist+weights
    end

    return expo
end

function ll_grad(x, data, params)
    function wrapper(x)
        return ll(x, data, params)
    end
    return ForwardDiff.gradient(n->wrapper(n), x)
end

end
#
# using DelimitedFiles
# using Flux
# using Random
# using Plots
#
#
# cd("C:\\Users\\Chaichontat\\Documents\\GitHub\\improv-sketches")
# Random.seed!(82)
# data = parse.(Float64, readdlm("data_sample.txt", ' ', String, '\n'));
#
# N = size(data)[1]
# M = 100
# dh = 2
# p = Dict("numSamples" => M, "numNeurons" => N, "hist_dim" => dh, "dt" => 0.1)
#
# weights = zeros(N,N)
# hist = zeros(N,dh)
# base = zeros(N)
#
#
# theta = [weights[:]; hist[:]; base[:]]
#
# gt = [d["w"][:]; d["h"][:]; d["b"][:]]
#
# println(simGLM.ll(theta, data, p))
#
# iter = 500
#
# out = zeros(iter)
# mean_sq = zeros(iter, 3)
#
# for i in 1:iter
#     global theta
#     out[i] = simGLM.ll(theta, data[:, i:i+99], p)
#     if i > 1
#         println(i, out[i])
#     end
#     mean_sq[i, 1] = Flux.mse(theta, gt)
#     theta = theta .- 1e-6 * out[i]
# end
#
# plot(out)
# # println(simGLM.ll_grad(theta, data, p))
#
