
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

type LeastAngleRegression <: BaseRegression
    normalize::Bool
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


function LeastAngleRegression(;
    normalize=false,
    max_iters=1000,
    params=nothing,
    errors=nothing)
    if params == nothing
        params = randn(1)
    end
    if errors == nothing
        errors = randn(1)
    end
    return LeastAngleRegression(normalize,max_iters,params,errors)
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
        println("Epoch: $(iter_count): current errors norm is $(errors_norm)")
        iter_count = iter_count + 1
    end
end

function train!(model::LogisticRegression,
                X::Matrix,
                y::Vector)
    #println("LogisticRegression!$(model.max_iters)")
    #global model = model1
    n_sample = size(X,1)
    n_feature = size(X,2)

    model.params = randn(n_feature+1)
    model.errors = randn(n_feature+1)
    X = hcat(X,ones(n_sample))
    #global X_global = hcat(X,ones(n_sample))
    #global y_global = y
    #@show y_global
    #cost_d = x -> ForwardDiff.gradient(cost_logistic,x)
    errors_norm = 1e10
    iter_count = 0
    while abs(errors_norm) > model.tolerance && iter_count < model.max_iters
        model.params -= model.C*(X' * (sigmoid(X * model.params) - y))
        errors_norm = norm(predict(model, X[:, 1:(end-1)]) - y)
        #println("Epoch: $(iter_count): current errors norm is $(errors_norm)")
        iter_count = iter_count + 1
    end
end

function train!(model1::LeastAngleRegression,
    X::Matrix,
    y::Vector)
    println("LeastAngleRegression!")
    global model= model1
    n_sample = size(X,1)
    n_feature = size(X,2)

    model.params = randn(n_feature+1)*10
    model.errors = randn(n_feature+1)*10

    if model.normalize
        for i = 1:n_feature
            X[:, i] = (X[:, i] - mean(X[:, i])) / norm(X[:, i])
        end

        y = y - mean(y)
    end

    var_y = var(y)

    global X_global = X
    global y_global = y

    coefs = zeros(1, n_feature)
    mu = zeros(n_sample, 1)

    risk = Inf

    for i = 1:n_feature
        if i > model.max_iters
            break
        end

        c = X' * (y - mu)

        # print("c ", c, '\n')

        cabs = abs(c)
        C = maximum(cabs)

        a = []
        for j = 1:n_feature
            if abs(cabs[j] - C) < 1e-10
                push!(a, j)
            end
        end

        s = sign(c[a])

        a_size = size(a)[1]

        Xa = zeros(n_sample, a_size)
        for j = 1:a_size
            Xa[:, j] = s[j] * X[:, a[j]]
        end

        Ga = Xa' * Xa

        o = ones(a_size, 1)
        Ga_inv = inv(Ga)
        Aa = sqrt(o' * Ga_inv * o)[1,1]

        wa = Aa * Ga_inv * o
        ua = Xa * wa

        A = X' * ua

        a_complement = []
        for j = 1:n_feature
            if !in(j, a)
                push!(a_complement, j)
            end
        end

        # print("C ", C, '\n')
        # print("Aa ", Aa, '\n')
        # print("c[a_c] ", c[a_complement], '\n')
        # print("A[a_c] ", A[a_complement], '\n')

        m1 = (C - c[a_complement]) ./ (Aa - A[a_complement])
        m2 = (C + c[a_complement]) ./ (Aa + A[a_complement])

        gamma = Inf

        for j = 1:size(m1)[1]
            if (m1[j] < gamma && m1[j] >= 0)
                gamma = m1[j]
            end
            if (m2[j] < gamma && m2[j] >= 0)
                gamma = m2[j]
            end
        end

        mu += gamma * ua

        riskI = norm(y - mu)^2 / var_y - n_sample + 2 * i
        if risk < riskI
            break
        end
        risk = riskI

        coefs[a] += s * gamma

        # print(coefs, '\n')
    end

    model.params = vec(coefs)
end


function predict(model::LogisticRegression,
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

function predict(model::LinearRegression,
                 x)
    n = size(x,1)
    b = ones(n)
    return hcat(x,b)*model.params
end

function predict(model::LeastAngleRegression,
                 x)
    return x*model.params
end



function cost_linear(w::Vector)
    return add_reg(mean_squared_error(y_global, X_global*w),w)
end

function cost_logistic(X, y, w::Vector,model)
   return add_reg(binary_crossentropy(y, sign(sigmoid(X*w))),w, model)
end

function add_reg(loss,w,model)
    if model.reg == "l1"
        return loss + model.C * sum(abs(w[1:(end-1)]))
    elseif model.reg == "l2"
        return loss + 0.5 * model.C * mean(abs2(w))
    end
    return loss
end


function test_LinearRegression(;reg = "l1")

    X_train, X_test, y_train, y_test = make_reg(n_features = 1)
    model = LinearRegression(lr=0.01, max_iters=200, reg=reg, C=0.03)
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("regression mse: ", mean_squared_error(y_test, predictions))
    PyPlot.scatter(X_test, y_test, color = "black")
    PyPlot.scatter(X_test, predictions, color = "green")
    legend(loc="upper right",fancybox="true")
end

function test_LogisticRegression(; reg = "l2")
    # Generate a random binary classification problem.
    X_train, X_test, y_train, y_test = make_cla()
    y_train = (y_train + 1)/2
    y_test = (y_test + 1)/2
    model = LogisticRegression(lr=0.1, max_iters=1000, reg=reg, C=0.01)
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    println("classification accuracy: ", accuracy(y_test, predictions))

    #PCA

    pca_model = PCA()
    train!(pca_model, X_test)
    plot_in_2d(pca_model, X_test, predictions, "LogisticRegression")
end

function test_LeastAngleRegression()

    n_features = 20

    X_train, X_test, y_train, y_test = make_reg(n_features = n_features)

    for i = 1:n_features
        X_train[:, i] = (X_train[:, i] - mean(X_train[:, i])) / norm(X_train[:, i])
        X_test[:, i] = (X_test[:, i] - mean(X_test[:, i])) / norm(X_test[:, i])
    end

    y_train = y_train - mean(y_train)
    y_test = y_test - mean(y_test)

    model = LeastAngleRegression(normalize=false)
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)

    print("regression mse: ", mean_squared_error(y_test, predictions))
end
