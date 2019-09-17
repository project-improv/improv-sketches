"""
    PillowGLM(p, args...; rate=LearningRate())
Fit a model that is linear in the parameters.
The (offline) objective function that PillowGLM approximately minimizes is
``(1/n) ∑ᵢ f(yᵢ, xᵢ'β) + ∑ⱼ λⱼ g(βⱼ),``
where ``fᵢ`` are loss functions of a single response and linear predictor, ``λⱼ``s are
nonnegative regularization parameters, and ``g`` is a penalty function.
# Arguments
- `loss = .5 * L2DistLoss()`
- `penalty = NoPenalty()`
- `algorithm = SGD()`
- `rate = LearningRate(.6)` (keyword arg)
# Example
    x = randn(1000, 5)
    y = x * range(-1, stop=1, length=5) + randn(1000)
    o = fit!(PillowGLM(5, MSPI()), (x, y))
    coef(o)
"""

using RecipesBase, Reexport, Statistics, LinearAlgebra, Dates
@reexport using OnlineStatsBase

import OnlineStatsBase: name, _fit!, _merge!, bessel, pdf, probs, smooth, smooth!,
    smooth_syr!, eachcol, eachrow, nvars

import LearnBase: fit!, nobs, value, predict
import StatsBase: autocov, autocor, confint, skewness, kurtosis, entropy, midpoints,
    fweights, sample, coef, Histogram

using LossFunctions: LossFunctions, Loss, L2DistLoss, AggMode
using PenaltyFunctions: PenaltyFunctions, Penalty
using LearnBase: LearnBase, deriv, prox
import OnlineStats: SGD, SGAlgorithm, Algorithm, XY, VectorOb

mutable struct PillowGLM{A<:Algorithm, L<:Loss, P<:Penalty, W} <: OnlineStat{XY}
    β::Array{Float64}
    λ::Array{Float64}
    gx::Array{Float64}
    loss::L
    penalty::P
    alg::A
    rate::W
    n::Int
end

function PillowGLM(p::Int, args...; rate=LearningRate())
    λ, loss, pen, alg = zeros(p, 2), .5*L2DistLoss(), PenaltyFunctions.NoPenalty(), SGD()
    for a in args
        a isa AbstractVector && (λ = a)
        a isa Float64        && (λ = fill(a, 1))
        a isa Loss           && (loss = a)
        a isa Penalty        && (pen = a)
        a isa Algorithm      && (alg = a)
    end
    # init!(alg, p)
    PillowGLM(zeros(p, 2), λ, zeros(p, 2), loss, pen, alg, rate, 0)
end

function Base.show(io::IO, o::PillowGLM)
    print(io, "PillowGLM: ")
    print(io, name(o.alg, false, false))
    print(io, " | mean(λ)=", mean(o.λ))
    print(io, " | ", o.loss)
    print(io, " | ", o.penalty)
    print(io, " | nobs=", nobs(o))
    print(io, " | nvars=", length(o.β))
end
coef(o::PillowGLM) = value(o)

function gradient!(o::PillowGLM, x, y)
    output = predict(o, x)
    for i in 1:2
        d_dη = LearnBase.deriv(o.loss, y[i], output[i])

        for j in eachindex(o.gx[:, i])
            o.gx[j, i] = x[j, i] * d_dη
        end
    end
end
function _fit!(o::PillowGLM{<:SGAlgorithm}, xy)
    x, y = xy
    o.n += 1
    gradient!(o, x, y)
    updateβ!(o, o.rate(o.n))
end

function predict(o::PillowGLM, x::AbstractMatrix)
    [dot(x[:, i], o.β[:, i]) for i in 1:2]
end

# function predict(o::PillowGLM, x::VectorOb)
#     dot(x, o.β)
# end
# predict(o::PillowGLM, x::AbstractMatrix) = x * o.β

# function objective(o::PillowGLM, x::AbstractMatrix, y::VectorOb)
#     value(o.loss, y, predict(o, x), AggMode.Mean()) + value(o.penalty, o.β, o.λ)
# end

#-----------------------------------------------------------------------# updateβ!
function updateβ!(o::PillowGLM{SGD}, γ)
    for i in 1:2
        for j in eachindex(o.β[:, i])
            o.β[j, i] = prox(o.penalty, o.β[j, i] - γ * o.gx[j, i], γ * o.λ[j, i])
        end
    end
end

using OnlineStats, LossFunctions, PenaltyFunctions
using Random, Plots
using Juno

Random.seed!(82)

# Data Parameters
n, p = 1000, 2
β = [1 2; 3 4]
ϵ = 0.05

# Data Generation
X = randn(n, p, p)
Y = zeros(n, p)

for i in 1:p  # Each neuron
    global Y
    for j in 1:p
        Y[:, i] += X[:, j, i] * β[j, i]
    end
end

# Define model
LossFunc = L2DistLoss()
PenaltyFunc = NoPenalty()
updater = SGD()

model = PillowGLM(p, LossFunc, PenaltyFunc, updater, rate=EqualWeight())

# Outputs
inferred_β = zeros(n)

# Online fit
for i in 1:n
    Xᵢ = @view X[i, :, :]
    Yᵢ = Y[i, :]
    fit!(model, (Xᵢ, Yᵢ))
    inferred_β[i] = coef(model)[1]
end

print(model.β)
