

type NaiveBayes 
    n_class::Integer
    class_mean::Matrix
    class_var::Matrix
    class_priors::Vector
    class_::Vector
end

function NaiveBayes(;
                    class_mean=zeros(2,2),
                    class_var=zeros(2,2),
                    class_priors=zeros(2),
                    n_class = 2,
                    class_ = zeros(2))
    return NaiveBayes(n_class, class_mean, class_var, class_priors,class_)
end
    


function train!(model::NaiveBayes, X::Matrix, y::Vector)
    n_feature = size(X,2)
    n_sample = size(X,1)
    model.class_ = sort(unique(y))
    model.n_class = length(unique(y))
    model.class_mean = zeros(n_feature, model.n_class)
    model.class_var = zeros(n_feature, model.n_class)
    model.class_priors = zeros(model.n_class)

    for (i,class) in enumerate(model.class_)
        X_temp = X[vec(y.==class),:]
        model.class_mean[:,i] = mean(X_temp,1)
        model.class_var[:,i] = var(X_temp,1)
        model.class_priors[i] = size(X_temp,1)/n_sample
    end

end

function predict(model::NaiveBayes,
                 X::Matrix)
    n_sample = size(X,1)
    res = zeros(n_sample)
    for i = 1:n_sample
        res[i] = predict(model, vec(X[i,:]))
    end
    return res
end

function predict(model::NaiveBayes,
                 X::Vector)
    temp = zeros(model.n_class)
    for i = 1:model.n_class
        prior = log(model.class_priors[i])
        cond = sum(log_pdf(model, X, i))
        temp[i] = prior + cond 
    end
    res = softmax(temp)
    class = model.class_[res]
    return class
end


function log_pdf(model::NaiveBayes,
                 X::Vector,n::Integer)

    mean_ = vec(model.class_mean[:,n])
    var_ = vec(model.class_var[:, n])
    d = sqrt(2*pi*var_)
    n = exp(-((X-mean_).^2./(2*var_)))
    return log(n./d)
end




function test_naive()
    X_train, X_test, y_train, y_test = make_cla()
    model = NaiveBayes()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))

    #PCA

    pca_model = PCA()
    train!(pca_model, X_test)
    plot_in_2d(pca_model, X_test, predictions, "Naive Bayes")
end







