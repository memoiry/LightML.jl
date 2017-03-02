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
                       max_iter::Integer = 500000)
    return NeuralNetwork(layers, act, weights, max_iter)
end

function train!(model::NeuralNetwork, X::Matrix, y::Vector)
    init_weights(model)
    depth = size(model.layers,1)
    a::Dict{Any, Vector} = Dict()
    z::Dict{Any, Vector} = Dict()
    X = hcat(X,ones(size(X,1)))
    @show depth

    for i = 1:model.max_iter
        r = rand(1:size(X,1))
        a[1] = X[r,:]
        for j = 2:(depth)
            z[j] = vec(a[j-1]' * model.weights[j-1])
            a[j] = vec(sigmoid(z[j]))
        end
        delta = Dict{Any, Vector}()
        error_ = a[depth] - y[r]
        if i % 1000 == 0
            println("$(i) epochs: error $(error_)")
        end
        delta[depth] = error_ .* sigmoid_prime(z[depth])
        for j = depth-1:-1:2
            delta[j] = vec(delta[j+1]' * model.weights[j]') .* sigmoid_prime(z[j])  
        end
        for j = 1:depth-1
            del = delta[j+1]
            a_temp = a[j]
            model.weights[j] = model.weights[j] - 0.1 * a_temp * del'
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
    m = model.layers[end]
    res = zeros(n,m)
    for i = 1:n 
        res[i,:] = predict(model, x[i,:])
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


## fixed

function test_NeuralNetwork()
    X_train = [0.1 0.1; 0.1 0.9; 0.9 0.1; 0.9 0.9]
    y_train = [0.1,0.9,0.9,0.1]
    model = NeuralNetwork(layers=[2,6,2,3,1])
    train!(model,X_train, y_train)
    predictions = predict(model,X_train)
end













