include("utils/utils.jl")

import StatsBase: countmap

abstract Dist

type Euclidean <: Dist
end

abstract KNN 

type KnnClassifier <: KNN
    k::Integer
    dis_func::Dist 
    X::Matrix
    y::Vector
end


type KnnRegression <: KNN
    k::Integer
    dis_func::Dist 
    X::Matrix
    y::Vector
end


function KnnClassifier(;
                       k::Integer = 5,
                       dist_func::Dist = Euclidean(),
                       X::Matrix = zeros(2,2),
                       y::Vector = zeros(2))
    return KnnClassifier(k,dist_func,X,y)
end


function train!(model::KnnClassifier, X::Matrix, y::Vector)
    model.X = X
    model.y = y 
end


function predict(model::KnnClassifier,
                 x::Matrix)
    n = size(x,1)
    res = zeros(n)
    for i = 1:n
        res[i] = predict(model, x[i,:])
    end
    return res
end

function predict(model::KnnClassifier,
                 x::Vector)
    n = size(model.X,1)
    res = zeros(n)
    for i = 1:n
        res[i] = dist(model.X[i,:],x,model.dis_func)
    end
    ind = sortperm(res)
    y_cos = model.y[ind[1:model.k]]
    label = 0
    label_freq = 0
    for (key,value) in countmap(y_cos)
        if value > label_freq
            label_freq = value
            label = key 
        end
    end
    return label

end

function predict(model::KnnRegression,
                 x::Vector)
    n = size(model.X,1)
    res = zeros(n)
    for i = 1:n
        res[i] = dist(model.X[i,:],x,model.dis_func)
    end
    ind = sortperm(res)
    y_cos = model.y[ind[1:model.k]]
    return mean(y_cos)
end


function dist(x::Vector, y::Vector, dist_func::Euclidean)
    return norm(x-y)
end


function regression_test()
    # Generate a random regression problem
    X_train, X_test, y_train, y_test = make_reg()

    model = KnnRegression()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("regression msea", mean_squared_error(y_test, predictions))
end

function classification_test()
    X_train, X_test, y_train, y_test = make_cla()
    model = KnnClassifier()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))
end













