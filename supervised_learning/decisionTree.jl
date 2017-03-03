include("utils/utils.jl")

typealias features Union{String, Real}

type DecisionNode
    label::features
    feature_index::Integer
    threshold::features
    true_branch::Union{DecisionNode,String}
    false_branch::Union{DecisionNode, String}
end

function DecisionNode(;
                      label = "nothing",
                      feature_index = 0,
                      threshold = Inf,
                      true_branch = "nothing",
                      false_branch = "nothing")
    return DecisionNode(label, feature_index, threshold,true_branch, false_branch)
end

type DecisionTree
    root::Union{DecisionNode,String}
    max_depth::Integer 
    min_gain::Float64 
    min_samples_split::Integer
    current_depth::Integer
end

function DecisionTree(;
                      root = "nothing",
                      min_samples_split = 2,
                      min_gain = 1e-7,
                      max_depth = Inf,
                      current_depth = 0)
    return DecisionTree(root, max_depth, min_gain,
        min_samples_split, current_depth)
end


function train!(model::DecisionTree, X::Matrix, y::Vector)
    model.current_depth = 0
    model.root = build_tree(model, X, y)
end


function build_tree(model::DecisionTree, X::Matrix, y::Vector)
    entropy = calc_entropy(y)
    highest_info_gain = 0
    best_criteria = 0
    best_sets = 0

    X_y = hcat(X, y)

    n_samples, n_features = size(X_y)
    n_features = n_features - 1
    if n_samples >= model.min_samples_split
        for i = 1:n_features
            feature_values = X_y[:, i]
            unique_values = unique(feature_values)
            for threshold in unique_values
                Xy_1, Xy_2 = split_at_feature(X_y, i, threshold)
                if size(Xy_1,1) > 0 && size(Xy_2,1) > 0
                    y1 = Xy_1[:, end]
                    y2 = Xy_2[:, end]
                    p = length(y1)/n_samples
                    info_gain = entropy - p * calc_entropy(y1) - (1-p) * calc_entropy(y2)
                    if info_gain > highest_info_gain
                        highest_info_gain = info_gain
                        best_criteria = Dict("feature_i" => i, "threshold" => threshold)
                        best_sets = [Xy_1, Xy_2]
                    end
                end
            end
        end
    end


    if model.current_depth < model.max_depth && highest_info_gain > model.min_gain
        X_1, y_1 = best_sets[1][:, 1:(end-1)], best_sets[1][:, end]
        X_2, y_2 = best_sets[1][:, 1:(end-1)], best_sets[1][:, end]
        true_branch = build_tree(model, X_1, y_1)
        false_branch = build_tree(model, X_2, y_2)
        model.current_depth += 1
        return DecisionNode(feature_index = best_criteria["feature_i"],
         threshold = best_criteria["threshold"], true_branch = true_branch, 
         false_branch = false_branch)
    end

    ## if not constructed
    most_common = -1
    max_count = 0
    for label in unique(y)
        count = sum(y.==label)
        if count > max_count 
            max_count = count
            most_common = label
        end
    end
    return DecisionNode(label = most_common)
end



function predict(model::DecisionTree, 
                 x::Matrix)
    n = size(x,1)
    res = zeros(n)
    for i = 1:n 
        res[i] = predict(model.root, x[i,:])
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


function calc_entropy(y)
    feature_unique = unique(y)
    num_sample = length(y)
    entro = 0
    for i in feature_unique
        num_feature = sum(y .== i)
        p = num_feature / num_sample
        entro += - p * log2(p)
    end
    return entro 
end


## seems there is some problem...

function test_DecisionTree()
    X_train, X_test, y_train, y_test = make_cla()
    model = DecisionTree()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))
end













