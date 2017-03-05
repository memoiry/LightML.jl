include("utils/utils.jl")

using Distributions

type GaussianMixture
    K::Integer
    max_iters::Integer
    method::String
    tolerance::Float64
    weights::Vector
    means::Matrix
    cov_::Dict{Integer,Matrix}
    responsibilities::Matrix
    likelihood::Vector
end

function GaussianMixture(; 
                         K::Integer = 4,
                         max_iters::Integer = 500,
                         tolerance::Float64 = 1e-3,
                         method::String = "random",
                         weights::Vector = zeros(4),
                         means::Matrix = zeros(4,4),
                         cov_::Dict{Integer,Matrix} = Dict{Integer,Matrix}(),
                         likelihood::Vector = zeros(4),
                         responsibilities::Matrix = zeros(4,4))
    return GaussianMixture(K, max_iters, method,
     tolerance, weights, means, cov_, responsibilities, likelihood)
end


function train!(model::GaussianMixture, X::Matrix)
    n_sample = size(X,1)
    initialize_(model, X)
    for i = 1:model.max_iters
        E_step!(model,X)
        M_step!(model,X)
        if isconverge_(model)
            break
        end
    end
end

function initialize_(model, X)
    model.weights = ones(model.K) * (1/model.K)
    model.means = X[randperm(size(X,1))[1:model.K],:]'
    rand_cov = cov(X)
    for i = 1:model.K
        model.cov_[i] = rand_cov
    end
    model.likelihood = Vector{Real}()
end

function E_step!(model, X)
    likelihood = zeros(size(X,1),model.K)
    for i = 1:model.K
        model.cov_[i] = Symmetric(model.cov_[i],:L)
        dis = MvNormal(model.means[:,i], model.cov_[i])
        likeli = pdf(dis, X')
        likelihood[:, i] = likeli
    end
    weighted_likelihood = repmat(model.weights',size(X,1),1) .* likelihood
    temp1 = sum(log(sum(weighted_likelihood,2)))
    push!(model.likelihood, temp1)
    for i = 1:size(weighted_likelihood, 1)
        weighted_likelihood[i,:] = weighted_likelihood[i,:] / sum(weighted_likelihood[i,:])
    end
    model.responsibilities = weighted_likelihood
end 


function M_step!(model, X)
    n_sample = size(X,1)
    n_feature = size(X,2)
    for i = 1:model.K
        resp = model.responsibilities[:,i]
        resp = resp'
        model.weights[i] = sum(resp)/n_sample
        for j = 1:n_feature
            temp =  resp * X[:,j]/sum(resp)
            model.means[j, i] =  temp[1]
        end
        cov_ = 0
        for j = 1:n_sample
            temp = (X[j,:] - model.means[:,i])
            cov_ += resp[j] .* temp * temp'
        end
        model.cov_[i] = cov_ ./ sum(resp)
    end
end


function isconverge_(model::GaussianMixture)
    if length(model.likelihood) > 1 && model.likelihood[end] - model.likelihood[end-1] <= model.tolerance
        return true
    end
    return false
end
function predict(model::GaussianMixture, 
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
    likelihood = zeros(model.K)
    for i = 1:model.K
        model.cov_[i] = Symmetric(model.cov_[i],:L)
        dis = MvNormal(model.means[:,i], model.cov_[i])
        likeli = pdf(dis, x)
        likelihood[i] = likeli
    end
    weighted_likelihood = model.weights .* likelihood
    return indmax(weighted_likelihood)
end



function test_GaussianMixture()
    X_train, y_test= make_blo()
    clu = size(X_train, 2)
    model = GaussianMixture(K = clu)
    train!(model,X_train)
    predictions = predict(model,X_train)
end













