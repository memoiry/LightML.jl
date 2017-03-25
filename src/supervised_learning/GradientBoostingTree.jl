

abstract GradientBoosting

type GradientBoostingClassifier <: GradientBoosting
    n_clf::Int64
    learning_rate::Float64
    max_depth::Int64
    min_sample_split::Int64
    min_gain::Float64
    init_estimate::Union{String, Real, Vector}
    trees::Array{ClassificationTree}
end

type GradientBoostingRegressor <: GradientBoosting
    n_clf::Int64
    learning_rate::Float64
    max_depth::Int64
    min_sample_split::Int64
    min_gain::Float64
    init_estimate::Union{String, Real, Vector}
    trees::Array{RegressionTree}
end

function GradientBoostingRegressor(;
                      n_clf::Int64 = 10,
                      learning_rate::Float64 = 0.5,
                      min_sample_split::Int64 = 20,
                      min_gain::Float64 = 1e-4,
                      max_depth = 4,
                      init_estimate::Union{String, Real, Vector} = "nothing"
                      )
    trees = []
    for i = 1:n_clf
        push!(trees, RegressionTree(min_samples_split = min_sample_split,
            min_gain = min_gain, max_depth = max_depth))
    end
    return GradientBoostingRegressor(n_clf, learning_rate, 
                        max_depth, min_sample_split,
                        min_gain, init_estimate, trees)
end

function GradientBoostingClassifier(;
                      n_clf::Int64 = 10,
                      learning_rate::Float64 = 0.5,
                      min_sample_split::Int64 = 20,
                      min_gain::Float64 = 1e-4,
                      max_depth = 4,
                      init_estimate::Union{String, Real, Vector} = "nothing"
                      )
    trees = []
    for i = 1:n_clf
        push!(trees, ClassificationTree(min_samples_split = min_sample_split,
            min_gain = min_gain, max_depth = max_depth))
    end
    return GradientBoostingClassifier(n_clf, learning_rate, 
                        max_depth, min_sample_split,
                        min_gain, init_estimate, trees)
end


function train!(model::GradientBoosting, X::Matrix, y::Vector)
    if typeof(model) <: GradientBoostingClassifier
        y = one_hot(y)
    end
    n_sample, n_feature = size(X)
    y_pred = mean(y, 1)
    if typeof(model) <: GradientBoostingClassifier
        y_pred = repmat(y_pred,n_sample,1)
    else
        y_pred = y_pred * ones(n_sample)
    end
    count = 1
    for tree in model.trees
        residual = -(y - y_pred)
        train!(tree, X, residual)
        residual_pred = predict(tree,X)
        y_pred -= model.learning_rate * residual_pred
        #println("$(count)th tree trained!")
        count += 1
    end
end


function predict(model::GradientBoosting,
                 x::Matrix)
    count = 1
    y_pred = 0
    for tree in model.trees
        if count == 1
            y_pred = - model.learning_rate * predict(tree,x)
            count = count + 1
        else
            y_pred -= model.learning_rate * predict(tree, x)
        end
    end

    if typeof(model) <: GradientBoostingClassifier
        return sign(softmax(y_pred)-1.5)
    end
    return y_pred
    
end



function test_GradientBoostingRegressor()
    X_train, X_test, y_train, y_test = make_reg(n_features = 1)
    model = GradientBoostingRegressor()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("regression msea", mean_squared_error(y_test, predictions))
    PyPlot.scatter(X_test, y_test, color = "black")
    PyPlot.scatter(X_test, predictions, color = "green")
    legend(loc="upper right",fancybox="true")
end


function test_GradientBoostingClassifier()
    X_train, X_test, y_train, y_test = make_cla()
    model = GradientBoostingClassifier()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    println("classification accuracy: ", accuracy(y_test, predictions))
end










