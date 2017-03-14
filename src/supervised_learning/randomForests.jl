
type randomForest
    min_split::Int64
    min_gain::Float64 
    max_depth::Integer
    n_estimators::Int64
    max_features::Union{Int64, String}
    feature_index::Dict{Int64, Vector}
    trees::Vector{ClassificationTree}
end

function randomForest(;
                      min_split::Int64 = 2,
                      min_gain::Float64 = 1e-7,
                      max_depth::Integer = 10000000,
                      n_estimators::Int64 = 200,
                      max_features::Union{Int64, String} = "nothing",
                      feature_index::Dict{Int64,Vector} = Dict{Int64,Vector}(),
                      trees::Vector{ClassificationTree} = Vector{ClassificationTree}()
                      )
    for i = 1:n_estimators
        push!(trees, ClassificationTree(min_gain = min_gain, min_samples_split = min_split, max_depth = max_depth))
    end
    return randomForest(min_split, min_gain, max_depth, n_estimators, max_features, feature_index, trees)
end


function train!(model::randomForest, X::Matrix, y::Vector)
    n_sample, n_feature = size(X)
    if model.max_features == "nothing"
        model.max_features = trunc(Int64, sqrt(n_feature))
    end
    sets = get_random_subsets(X, y, model.n_estimators)
    #println("Start training Random Forests")
    for i = 1:model.n_estimators
        #temp = i/model.n_estimators*100
        #println("Progress: $(temp) %")
        temp_data = sets[1,:,:]
        X = temp_data[:, 1:(end-1)]
        y = temp_data[:, end]
        idx = sample(1:n_feature, model.max_features, replace = false)
        model.feature_index[i] = idx
        X = X[:, idx]
        train!(model.trees[i], X, y)
    end
end

function predict(model::randomForest, 
                 x::Matrix)
    n_sample = size(x,1)
    res = zeros(n_sample, model.n_estimators)
    for i = 1:model.n_estimators
        idx = model.feature_index[i]
        res[:,i] = predict(model.trees[i], x[:, idx])
    end
    ans = zeros(n_sample)
    for k = 1:n_sample
        most = 0
        max_time = 0
        for i in unique(res[k,:])
            n_times = 0
            for j = 1:model.n_estimators
                if i == res[j]
                    n_times += 1
                end
            end
            if n_times > max_time
                most = i
            end
        end
        ans[k] = most
    end
    return ans
end


function test_randomForest()
    X_train, X_test, y_train, y_test = make_iris()
    model = randomForest()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    println("classification accuracy ", accuracy(y_test, predictions))
    #pca
    pcamodel = PCA()
    train!(pcamodel, X_test)
    plot_in_2d(pcamodel, X_test, predictions, "RadomForest")
end













