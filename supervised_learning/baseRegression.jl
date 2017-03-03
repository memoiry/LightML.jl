include("utils/utils.jl")

abstract BaseRegression


type LinearRegression <: BaseRegression
    C::Float64
    reg::String
    lr::Float64
    tolerance::Float64
    max_iters::Integer
    params::Vector
    errors::Vector
end

type LogisticRegression <: BaseRegression
    C::Float64
    reg::String
    lr::Float64
    tolerance::Float64
    max_iters::Integer
    params::Vector
    errors::Vector
end


function LinearRegression(;
                          C=0.01,
                          tolerance=0.1,
                          max_iters=1000,
                          reg="None",
                          lr=0.001,
                          params=nothing,
                          errors=nothing)
    if params == nothing
        params = randn(1)
    end
    if errors == nothing
        errors = randn(1)
    end
    return LinearRegression(C,reg,lr,tolerance,max_iters,params,errors)
end


function LogisticRegression(;
                          C=0.01,
                          tolerance=0.1,
                          max_iters=1000,
                          reg="None",
                          lr=0.001,
                          params=nothing,
                          errors=nothing)
    if params == nothing
        params = randn(1)
    end
    if errors == nothing
        errors = randn(1)
    end
    return LogisticRegression(C,reg,lr,tolerance,max_iters,params,errors)
end


function train!(model1::LinearRegression,
                X::Matrix,
                y::Vector)
    println("LinearRegression!$(model1.max_iters)")
    global model= model1
    n_sample = size(X,1)
    n_feature = size(X,2)

    model.params = randn(n_feature+1)*10
    model.errors = randn(n_feature+1)*10

    global X_global = hcat(X,ones(n_sample))
    global y_global = y
    cost_d = x -> ForwardDiff.gradient(cost_linear,x)
    errors_norm = 1e10
    iter_count = 0
    while errors_norm > model.tolerance && iter_count < model.max_iters
        model.params -= model.C*cost_d(model.params)
        errors_norm = cost_linear(model.params)
        println("Epoch: $(iter_count): current MSE is $(errors_norm)")
        iter_count = iter_count + 1
    end
end

function train!(model1::LogisticRegression,
                X::Matrix,
                y::Vector)
    println("LogisticRegression!$(model1.max_iters)")
    global model = model1
    n_sample = size(X,1)
    n_feature = size(X,2)

    model.params = randn(n_feature+1)
    model.errors = randn(n_feature+1)

    global X_global = hcat(X,ones(n_sample))
    global y_global = y
    cost_d = x -> ForwardDiff.gradient(cost_logistic,x)
    errors_norm = 1e10
    iter_count = 0
    while errors_norm > model.tolerance && iter_count < model.max_iters
        model.params -= model.C*cost_d(model.params)
        errors_norm = cost_linear(model.params)
        println("Epoch: $(iter_count): current MSE is $(errors_norm)")
        iter_count = iter_count + 1
    end
end


function predict!(model::LogisticRegression,
                 x)
    n = size(x,1)
    b = ones(n)
    res = sigmoid(hcat(x,b)*model.params)
    n_ = size(res,1)
    for i = 1:n_
        if res[i] >= 0.5
            res[i] = 1
        else
            res[i] = 0
        end
    end
    return res
end

function predict!(model::LinearRegression ,
                 x)
    n = size(x,1)
    b = ones(n)
    return hcat(x,b)*model.params
end



function cost_linear(w::Vector)
    return add_reg(mean_squared_error(y_global, X_global*w),w)
end

function cost_logistic(w::Vector)
    return add_reg(binary_crossentropy(y_global, sigmoid(X_global*w)),w)
end

function add_reg(loss,w)
    if model.reg == "l1"
        return loss + model.C * sum(abs(w[1:(end-1)]))
    elseif model.reg == "l2"
        return loss + 0.5 * model.C * mean(abs2(w))
    end
    return loss 
end


function regression_test()
    # Generate a random regression problem
    X_train, X_test, y_train, y_test = make_reg()

    model = LinearRegression(lr=0.01, max_iters=200, reg="l1", C=0.03)
    train!(model,X_train, y_train)
    predictions = predict!(model,X_test)
    print("regression msea", mean_squared_error(y_test, predictions))

end

function classification_test()
    # Generate a random binary classification problem.
    X, y = dat.make_classification(n_samples=1000, n_features=100,
                               n_informative=75, random_state=1111,
                               n_classes=2, class_sep=2.5, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        rand_seed=1111)

    model = LogisticRegression(lr=0.01, max_iters=100, reg="l2", C=0.01)
    train!(model,X_train, y_train)
    predictions = predict!(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))

end


