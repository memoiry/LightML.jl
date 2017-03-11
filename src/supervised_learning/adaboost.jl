
type Adaboost
    n_clf::Int64
    clf::Matrix
end

function Adaboost(;
                  n_clf::Int64 = 10
                  )
    clf = zeros(4, n_clf)
    return Adaboost(n_clf, clf)
end


function train!(model::Adaboost, X::Matrix, y::Vector)
    n_sample, n_feature = size(X)
    w = ones(n_sample) / n_sample
    threshold = 0
    polarity = 0
    feature_index = 0
    alpha = 0
    for i = 1:model.n_clf
        err_max = 1e10
        for feature_ind = 1:n_feature
            for threshold_ind = 1:n_sample
                polarity_ = 1
                err = 0
                threshold_ = X[threshold_ind, feature_ind]

                for sample_ind = 1:n_sample 
                    pred = 1
                    x = X[sample_ind, feature_ind]
                    if x < threshold_
                        pred = -1
                    end
                    err += w[sample_ind] * (y[sample_ind] != pred)
                end
                if err > 0.5
                    err = 1 - 0.5
                    polarity_ = -1
                end

                if err < err_max
                    err_max = err 
                    threshold = threshold_
                    polarity = polarity_
                    feature_index = feature_ind
                end
            end
        end

        alpha = 1/2 * log((1.000001-err_max)/(err_max+0.000001))

        for j = 1:n_sample 
            pred = 1
            x = X[j, feature_index]
            if polarity * x < polarity * threshold
                pred = -1
            end
            w[j] = w[j] * exp(-alpha * y[j] * pred)
        end
        model.clf[:, i] = [feature_index, threshold, polarity, alpha]
    end
end

function predict(model::Adaboost, 
                 x::Matrix)
    n = size(x,1)
    res = zeros(n)
    for i = 1:n 
        res[i] = predict(model, x[i,:])
    end
    return res
end

function predict(model::Adaboost,
                 x::Vector)
    s = 0
    for i = 1:model.n_clf
        pred = 1
        feature_index = trunc(Int64,model.clf[1,i])
        threshold = model.clf[2,i]
        polarity = model.clf[3,i]
        alpha = model.clf[4,i]     
        x_temp = x[feature_index]
        if polarity * x_temp < polarity * threshold
            pred = -1
        end
        s += alpha * pred
    end

    return sign(s)

end



function test_Adaboost()
    X_train, X_test, y_train, y_test = make_cla(n_features = 8, n_samples = 1000)

    #Adaboost
    model = Adaboost()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    println("The number of week classifiers ", 10)
    println("classification accuracy: ", accuracy(y_test, predictions))

    #PCA

    pca_model = PCA()
    train!(pca_model, X_test)
    plot_in_2d(pca_model, X_test, predictions, "Adaboost")

end









