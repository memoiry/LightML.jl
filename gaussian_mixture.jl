include("utils/utils.jl")

using Distributions

type GaussianMixture
    K::Integer
    max_iters::Integer
    method::AbstractString
    tolerance::Real
    weights::Vector
    means::Matrix
    cov_::Dict{Any,Matrix}
    responsibilities::Matrix
    likelihood::Vector
end

function GaussianMixture(; 
                         K::Integer = 4,
                         max_iters::Integer = 500,
                         tolerance::Real = 1e-3,
                         method::AbstractString = "random",
                         weights::Vector = zeros(4),
                         means::Vector = zeros(4),
                         cov_::Vector = zeros(4),
                         likelihood::Vector = zeros(4))
    return GaussianMixture(K, max_iters, method, tolerance, weights, means, cov_)
end


function train!(model::GaussianMixture, X::Matrix)
    n_sample = size(X,1)
    initialize_(model, X)
    for i = 1:model.max_iters
        E_step!(model,X)
        M_step!(model)

        if isconverge_(model)
    end
end

function initialize_(model, X)
    model.weights = zeros(model.K) * (1/model.K)
    model.means = X[randperm(size(X,1))[1:model.K],:]
    rand_cov = cov(X)
    for i = 1:model.K
        model.cov_[i] = rand_cov
    end
    model.likelihood = []
end

function E_step!(model, X)
    likelihood = zeros(size(X,1),model.K)
    for i = 1:model.K
        dis = MvNormal(model.means[i,:], model.cov_[i])
        likeli = pdf(dis, X)
        likelihood[:, i] = likeli
    end
    push!(model.likelihood, sum(likelihood))
    weighted_likelihood = repmat(model.weights',size(X,1),1) .* likelihood
    for i = 1:size(weighted_likelihood, 1)
        weighted_likelihood[i,:] = weighted_likelihood[i,:] / sum(weighted_likelihood[i,:])
    end
end 


function M_step!(model, X)
    for i = 1:model.K
        resp = model.responsibilities[:,i]
        model.means[i, :] = X * resp ./ sum(resp)
        model.cov_[i] = (X-model.means[i,:]).^2 * resp ./ sum(resp)
        


end


function isconverge_(model::GaussianMixture)


end
function predict(mode::GaussianMixture, 
                 x::Matrix)
    n = size(x,1)
    res = zeros(n)
    for i = 1:n 
        res[i] = predict(model, x[i,:])
    end
    return res
end

function predict(model::GaussianMixture,
                 x::Vector)
end



function test_GaussianMixture()
    X_train, X_test, y_train, y_test = make_cla()
    model = GaussianMixture()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))
end













