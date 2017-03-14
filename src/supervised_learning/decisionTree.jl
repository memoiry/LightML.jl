

type DecisionNode
    label::Union{Vector, Int64, String, Float64}
    feature_index::Integer
    threshold::Features
    true_branch::Union{DecisionNode,String}
    false_branch::Union{DecisionNode, String}
    y_num::Int64
end

function DecisionNode(;
                      label = "nothing",
                      feature_index = 0,
                      threshold = Inf,
                      true_branch = "nothing",
                      false_branch = "nothing",
                      y_num = 1)
    return DecisionNode(label, feature_index, threshold,true_branch, false_branch, y_num)
end

abstract DecisionTree

type RegressionTree <: DecisionTree
    root::Union{DecisionNode,String}
    max_depth::Integer 
    min_gain::Float64 
    min_samples_split::Integer
    current_depth::Integer
end  


type ClassificationTree <: DecisionTree
    root::Union{DecisionNode,String}
    max_depth::Integer 
    min_gain::Float64 
    min_samples_split::Integer
    current_depth::Integer
    y_num::Integer
end

function ClassificationTree(;
                      root = "nothing",
                      min_samples_split = 2,
                      min_gain = 1e-7,
                      max_depth = 1e7,
                      current_depth = 0,
                      y_num = 1)
    return ClassificationTree(root, max_depth, min_gain,
        min_samples_split, current_depth, y_num)
end

function RegressionTree(;
                      root = "nothing",
                      min_samples_split = 2,
                      min_gain = 1e-7,
                      max_depth = 1e7,
                      current_depth = 0)
    return RegressionTree(root, max_depth, min_gain,
        min_samples_split, current_depth)
end


function train!(model::DecisionTree, X::Matrix, y::Array)
    model.current_depth = 0
    #Normalize
    #X_y = [X y]
    #X_y = normalize_(X_y)
    #X = X_y[:, 1:(end-1)]
    #y = X_y[:, end]
    model.root = build_tree(model, X, y)
end


function build_tree(model::DecisionTree, X::Matrix, y::Array)
    model.y_num = size(y,2)
    entropy = calc_entropy(y)
    largest_impurity = 0
    best_criteria = 0
    best_sets = 0
    n_features = size(X, 2)
    X_y = hcat(X, y)
    n_samples = size(X_y,1)
    if n_samples >= model.min_samples_split
        for i = 1:n_features
            feature_values = X_y[:, i]
            unique_values = unique(feature_values)
            #For large regression problem
            #if typeof(model) == RegressionTree
            #    num = 8
            #    if length(unique_values) >= num
            #        num_ = length(unique_values)
            #        indd = randperm(num_)[1:num]
            #        unique_values = unique_values[indd]
            #    end
            #end
            for threshold in unique_values
                Xy_1, Xy_2 = split_at_feature(X_y, i, threshold)
                if size(Xy_1,1) > 0 && size(Xy_2,1) > 0
                    y1 = Xy_1[:, (n_features+1):end]
                    y2 = Xy_2[:, (n_features+1):end]
                    impurity = impurity_calc(model, y, y1, y2)
                    if impurity > largest_impurity
                        #println("$(impurity)")
                        largest_impurity = impurity
                        best_criteria = Dict("feature_i" => i, "threshold" => threshold)
                        best_sets = Dict("left_branch" => Xy_1, "right_branch" => Xy_2)
                    end
                end
            end
        end
    end
    if model.current_depth < model.max_depth && largest_impurity > model.min_gain
        leftX, leftY = best_sets["left_branch"][:, 1:n_features], best_sets["left_branch"][:, (n_features+1):end]
        rightX, rightY = best_sets["right_branch"][:, 1:n_features], best_sets["right_branch"][:, (n_features+1):end]
        true_branch = build_tree(model, leftX, leftY)
        false_branch = build_tree(model, rightX, rightY)
        model.current_depth += 1
        return DecisionNode(feature_index = best_criteria["feature_i"],
         threshold = best_criteria["threshold"], true_branch = true_branch, 
         false_branch = false_branch)
    end

    ## if not constructed

    leaf_value = leaf_value_calc(model, y)
    return DecisionNode(label = leaf_value)
end


function leaf_value_calc(model::RegressionTree, y::Array)
    return mean(y)
end

function leaf_value_calc(model::ClassificationTree, y::Array)
    labels = size(y,2)
    if labels > 1
        y = unhot(y)
    end
    feature = unique(y)
    most_common = nothing
    count_max = 0
    for i in feature 
        count = sum(y .== i)
        if count > count_max
            count_max = count
            most_common = i
        end
    end
    if labels > 1
        most_common = trunc(Int64, most_common[1])
        most_common = eye(labels)[most_common,:]
    end
    return most_common
end



function impurity_calc(model::RegressionTree, y, y1, y2)
    var_total = var(y)
    var_y1 = var(y1)
    var_y2 = var(y2)
    frac_1 = length(y1) / length(y)
    frac_2 = length(y2) / length(y)

    variance_reduction = var_total - (frac_1 * var_y1 + frac_2 * var_y2)
    return variance_reduction
end

function impurity_calc(model::ClassificationTree, y, y1, y2)
    p = size(y1,1)/size(y,1)
    entro = calc_entropy(y)
    #println("entro: $(entro), p: $(p), res: $(entro - (p * calc_entropy(y1) + (1-p) * calc_entropy(y2)))")
    return entro - (p * calc_entropy(y1) + (1-p) * calc_entropy(y2))
end


function predict(model::DecisionTree, 
                 x::Matrix)
    n = size(x,1)
    res = zeros(n)
    if model.y_num == 1
        for i = 1:n 
            res[i] = predict(model.root, x[i,:])
        end
    else
        res = zeros(n, model.y_num)
        for i = 1:n
            res[i,:] = predict(model.root, x[i,:])
        end
    end

    return res
end


function predict(model::DecisionNode,
                 x::Vector)
    if model.label != "nothing"
        return model.label
    end
    feature_current = x[model.feature_index] 
    if typeof(feature_current) <:String
        if feature_current == model.threshold
            return predict(model.true_branch, x)
        else
            return predict(model.false_branch, x)
        end
    elseif typeof(feature_current) <: Real 
        if feature_current <= model.threshold
            return predict(model.true_branch, x)
        else
            return predict(model.false_branch, x)
        end
    end
end

function print_tree(model::DecisionTree)
    if model.lable != "nothing"
        println($(model.label))
    else
        println("$(model.feature_index):$(model.threshold)?")
        println(" T->")
        print_tree(model.true_branch)
        println(" F->")
        print_tree(model.true_branch)
    end
end


function split_at_feature(X, feature_i, threshold)
    if typeof(threshold) <: Real
        ind1 = find(X[:,feature_i] .<= threshold)
        ind2 = find(X[:,feature_i] .> threshold)
    elseif typeof(threshhold) <: String
        ind1 = find(X[:,feature_i] .== threshold)
        ind2 = find(X[:,feature_i] .!= threshold)
    end
    X_1 = X[ind1,:]
    X_2 = X[ind2,:]
    return X_1, X_2
end



function test_ClassificationTree()
    X_train, X_test, y_train, y_test = make_iris()
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    model = ClassificationTree()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    y_test = unhot(y_test)
    predictions = unhot(predictions)
    print("classification accuracy", accuracy(y_test, predictions))


    #PCA

    #pca_model = PCA()
    #train!(pca_model, X_test)
    #plot_in_2d(pca_model, X_test, predictions, "ClassificationTree")
end


function test_RegressionTree()
    X_train, X_test, y_train, y_test = make_reg(n_features = 1)
    model = RegressionTree()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("regression msea", mean_squared_error(y_test, predictions))
    PyPlot.scatter(X_test, y_test, color = "black")
    PyPlot.scatter(X_test, predictions, color = "green")
    legend(loc="upper right",fancybox="true")
end














