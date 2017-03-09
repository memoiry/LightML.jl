

type GDA 
    n_class::Integer
    class_mean::Matrix
    class_cov::Matrix
    class_priors::Vector
    class_::Vector
end

function GDA(;
            class_mean=zeros(2,2),
            class_cov=zeros(2,2),
            class_priors=zeros(2),
            n_class = 2,
            class_ = zeros(2))
    return GDA(n_class, class_mean, class_cov, class_priors,class_)
end
    


function train!(model::GDA, X::Matrix, y::Vector)
    n_feature = size(X,2)
    n_sample = size(X,1)
    model.class_ = sort(unique(y))
    model.n_class = length(unique(y))
    model.class_mean = zeros(n_feature, model.n_class)
    model.class_cov = cov(X)
    model.class_priors = zeros(model.n_class)
    for (i,class) in enumerate(model.class_)
        X_temp = X[vec(y.==class),:]
        model.class_mean[:,i] = mean(X_temp,1)
        model.class_priors[i] = size(X_temp,1)/n_sample
    end
end

function predict(model::GDA,
                 X::Matrix)
    n_sample = size(X,1)
    res = zeros(n_sample)
    for i = 1:n_sample
        res[i] = predict(model, vec(X[i,:]))
    end
    return res
end

function predict(model::GDA,
                 X::Vector)
    temp = zeros(model.n_class)
    for (i, class) in enumerate(model.n_class)
        p = log_pdf(model, X, i)
        temp[i] = p[1]
    end
    res = indmax(exp(temp - log(sum(exp(temp)))))
    @show res
    class = model.class_[res]
    return class
end


function log_pdf(model::GDA,
                 X::Vector, class::Integer)
    t = size(X,1)
    c = -0.5 * (X - model.class_mean[:, class])' * pinv(model.class_cov) * (X- model.class_mean[:, class])
    d = -0.5 * log(det(model.class_cov))
    p = -t/2*log(2*pi)
    pp = log(model.class_priors[class])
    return c + d + p + pp
end

# therer is some problem to be fixed
function test_GDA()
    X_train, X_test, y_train, y_test = make_cla()
    model = GDA()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))
end







