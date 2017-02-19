include("utils/utils.jl")


type NeuralNetwork
    layers::Vector
    act::AbstractString
    weights::Dict{Any, Matrix}
    max_iter::Integer
end

function NeuralNetwork(;
                       layers::Vector = [2,2,1],
                       act::AbstractString = "sigmoid",
                       weights::Dict{Any,Matrix} = Dict{Any,Matrix}(),
                       max_iter::Integer = 50000)
    return NeuralNetwork(layers, act, weights, max_iter)
end

function train!(model::NeuralNetwork, X::Matrix, y::Vector)
    init_weights(model)
    depth = size(model.layers,1)
    a::Dict{Any, Vector} = Dict()
    X = hcat(ones(size(X,1)),X)
    @show depth
    for i = 1:model.max_iter
        r = rand(1:size(X,1))
        a[1] = X[r,:]
        for j = 2:depth
            a[j] = vec(sigmoid(a[j-1]' * model.weights[j-1]))
        end
        delta = Dict{Any, Vector}()
        error_ = a[depth] - y[r]
        if i % 1000 == 0
            println("$(i) epochs: error $(error_)")
        end
        delta[depth] = error_ .* sigmoid_prime(a[depth])
        for j = depth-1:-1:2
            delta[j] = vec(delta[j+1]' * model.weights[j]') .* sigmoid_prime(a[j])
        end
        for j = 1:depth-1
            del = delta[j+1]
            a_temp = a[j]
            model.weights[j] = model.weights[j] - a_temp * del'
        end
    end


end

function init_weights(model::NeuralNetwork)
    depth_ = size(model.layers,1)
    for i = 1:(depth_-2)
        model.weights[i] = 2*rand(model.layers[i]+1,model.layers[i+1]+1)-1
    end
    model.weights[depth_-1] = 2*rand(model.layers[depth_-1]+1,model.layers[depth_])-1
end

function predict(model::NeuralNetwork, 
                 x::Matrix)
    n = size(x,1)
    res = Dict()
    for i = 1:n 
        res[i] = predict(model, x[i,:])
    end
    return res
end

function predict(model::NeuralNetwork,
                 x::Vector)
    x = x'
    x = hcat(x,[1])
    for i = 1:length(model.layers)-1
        x = sigmoid(x * model.weights[i])
    end
    return x
end

function sigmoid(x)
    return 1./(1+exp(-x))
end

function sigmoid_prime(x)
    return sigmoid(x).*(1-sigmoid(x))
end
function test_NeuralNetwork()
    X_train = [0.1 0.1; 0.1 0.9; 0.9 0.1; 0.9 0.9]
    y_train = [0.1,0.9,0.9,0.1]
    model = NeuralNetwork(layers=[2,2,1])
    train!(model,X_train, y_train)
    predictions = predict(model,X_train)

    #print("classification accuracy", accuracy(y_test, predictions))
end













