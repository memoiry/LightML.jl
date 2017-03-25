

type XGBoost <: DecisionTree
    n_estimators::Int64
    learning_rate::Float64
    min_samples_split::Int64
    min_impurity::Float64
    max_depth::Int64
    init_estimate::Union{String, Array}
    loss::LogisticLoss
    trees::Vector{XGBoostRegressionTree}
end

function XGBoost(;
                 n_estimators::Int64 = 2,
                 learning_rate::Float64 = 0.001,
                 min_samples_split::Int64 = 2,
                 min_impurity::Float64 = 0.00001,
                 max_depth::Int64 = 2,
                 init_estimate = "nothing",
                 loss::LogisticLoss = LogisticLoss())
    trees = []
    single_boost_tree = XGBoostRegressionTree(min_samples_split = min_samples_split,
            min_gain = min_impurity, max_depth = max_depth, loss = loss)
    for i in 1:n_estimators
        push!(trees, single_boost_tree)
    end
    return XGBoost(n_estimators, learning_rate,
                   min_samples_split, min_impurity,
                   max_depth, init_estimate, loss, trees)
end

function train!(model::XGBoost, X::Matrix, y::Vector)
    y = one_hot(y)
    n_sample, n_feature = size(X)
    model.init_estimate = mean(y, 1)
    y_pred = repmat(model.init_estimate, size(y,1),1)
    for i in 1:model.n_estimators
        y_and_pred = hcat(y, y_pred)
        print("$(y_and_pred)")
        train!(model.trees[i], X, y_and_pred)
        update_pred = predict(model.trees[i],X)
        y_pred += model.learning_rate * update_pred
        progress = 1/model.n_estimators * i * 100
        #println("Progress: $(progress) %")
    end
end

function predict(model::XGBoost, 
                 x::Matrix)
    n_sample = size(x)
    flag = 1
    y_pred = 0
    for tree in model.trees
        update_pred = predict(tree,x)
        update = model.learning_rate * update_pred
        if flag == 1
            y_pred = update
            flag = 0
        else
            y_pred += update
        end
    end
    res = zeros(size(y_pred,1))
    for i in 1:size(y_pred,1)
        exp_pred = exp(y_pred[i,:])
        y_pred[i,:] = exp_pred / sum(exp_pred)
        print("$(y_pred[i,:])")
        res[i] = indmax(y_pred[i,:])
    end
    return res-1
end


function test_xgboost()
    X_train, X_test, y_train, y_test = make_iris()
    model = XGBoost()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("$(predictions)")
    print("classification accuracy", accuracy(y_test, predictions))
end













