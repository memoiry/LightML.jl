

type BoostingTree
    n_clf::Int64
    learning_rate::Float64
    max_depth::Int64
    min_sample_split::Int64
    min_gain::Float64
    init_estimate::Features
    trees::Array{RegressionTree}
end

function BoostingTree(;
                      n_clf::Int64 = 20,
                      learning_rate::Float64 = 0.5,
                      min_sample_split::Int64 = 20,
                      min_gain::Float64 = 1e-4,
                      max_depth = 4,
                      init_estimate::Features = "nothing"
                      )
    trees = []
    for i = 1:n_clf
        push!(trees, RegressionTree(min_samples_split = min_sample_split,
            min_gain = min_gain, max_depth = max_depth))
    end
    return BoostingTree(n_clf, learning_rate, 
                        max_depth, min_sample_split,
                        min_gain, init_estimate, trees)
end


function train!(model::BoostingTree, X::Matrix, y::Vector)
    n_sample, n_feature = size(X)
    model.init_estimate = mean(y)
    y_pred = model.init_estimate * ones(n_sample)
    count = 1
    for tree in model.trees
        residual = -(y - y_pred)
        train!(tree, X, residual)
        residual_pred = predict(tree,X)
        y_pred -= model.learning_rate * residual_pred
        println("$(count)th tree trained!")
        count += 1
    end
end




function predict(model::BoostingTree,
                 x::Matrix)
    y_pred = model.init_estimate
    for tree in model.trees
        y_pred -= model.learning_rate * predict(tree, x)
    end
    return y_pred
    
end



function test_BoostingTree()
    X_train, X_test, y_train, y_test = make_reg(n_features = 1)
    model = BoostingTree()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("regression msea", mean_squared_error(y_test, predictions))
    PyPlot.scatter(X_test, y_test, color = "black")
    PyPlot.scatter(X_test, predictions, color = "green")
    legend(loc="upper right",fancybox="true")

end













