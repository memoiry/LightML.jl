
typealias Arr Union{Vector, Matrix} 

type SVM
    X::Matrix
    y::Vector
    C::Float64 
    tol::Float64 
    max_iter::Integer
    kernel::String
    degree::Integer
    gamma::Float64
    alpha::Vector
    b::Float64
    sv_indx::Vector
    K::Matrix
end

function svm(X::Matrix,
             y::Vector;
             C::Float64 = 1.0,
             kernel::String = "linear",
             max_iter::Integer = 100,
             tol::Float64 = 1e-3,
             degree::Integer = 2,
             gamma::Float64 = 0.1,
             alpha::Vector = zeros(10),
             b::Float64 = 0.0)
    n = size(X,1)
    alpha = zeros(n)
    K = zeros(n,n)
    sv_indx = collect(1:n)
    return SVM(X,y,C,tol,max_iter,kernel, degree,gamma,alpha,b,sv_indx,K)
end

function predict(model::SVM,
                 x::Arr)
    n = size(x,1)
    res = zeros(n)
    if n == 1
        res[1] = predict_row(x,model)
    else 
        for i = 1:n
            res[i] = predict_row(x[i,:],model)
        end
    end
    return res
end


function train!(model::SVM)
    n_sample = size(model.X,1)
    model.K = zeros(n_sample,n_sample)
    for i in 1:n_sample
        model.K[:,i] = kernel_c(model.X,model.X[i,:],model)
    end
    # start training

    iters = 0
    while iters < model.max_iter
        iters += 1
       # println("Processing $(iters)/$(model.max_iter)")
        alpha_prev = copy(model.alpha)
        for j = 1:n_sample
            i = rand(1:n_sample)
            eta = 2.0 * model.K[i, j] - model.K[i, i] - model.K[j, j]
            if eta >= 0
                continue
            end
            L, H = count_bounds(i, j,model)

            # Error for current examples
            e_i, e_j = error_(i,model), error_(j,model)

            # Save old alphas
            alpha_io, alpha_jo = model.alpha[i], model.alpha[j]

            # Update alpha
            model.alpha[j] -= (model.y[j] * (e_i - e_j)) / eta
            model.alpha[j] = clamp(model.alpha[j], L, H)

            model.alpha[i] = model.alpha[i] + model.y[i] * model.y[j] * (alpha_jo - model.alpha[j])

            # Find intercept
            b1 = model.b - e_i - model.y[i] * (model.alpha[i] - alpha_jo) * model.K[i, i] - 
                 model.y[j] * (model.alpha[j] - alpha_jo) * model.K[i, j]
            b2 = model.b - e_j - model.y[j] * (model.alpha[j] - alpha_jo) * model.K[j, j] - 
                 model.y[i] * (model.alpha[i] - alpha_io) * model.K[i, j]
            if 0 < model.alpha[i] < model.C
                model.b = b1
            elseif 0 < model.alpha[j] < model.C
                model.b = b2
            else
                model.b = 0.5 * (b1 + b2)
            end

            # Check convergence
            diff = norm(model.alpha - alpha_prev)
            if diff < model.tol
                break
            end
        end
    end

    println("Convergence has reached after $(iters). for $(model.kernel)")

    # Save support vectors index
    model.sv_indx = find(model.alpha .> 0)

end

function kernel_c(X::Matrix,
                y::Vector,
                model::SVM)
    if model.kernel == "linear"
        return X * y
    elseif model.kernel == "poly"
        return (X * y).^model.degree
    elseif model.kernel == "rbf"
        n = size(X,1)
        res = zeros(n)
        for i = 1:n
            res[i] = e^(-model.gamma*sumabs2(X[i,:]-y))
        end
        return res
    end
end

function count_bounds(i,j,model)
    if model.y[i] != model.y[j]
        L = max(0, model.alpha[j] - model.alpha[i])
        H = min(model.C, model.C - model.alpha[i] + model.alpha[j])
    else
        L = max(0, model.alpha[i] + model.alpha[j] - model.C)
        H = min(model.C, model.alpha[i] + model.alpha[j])
    end
    return L, H       
end

function predict_row(x,model)
    res = kernel_c(model.X,x,model)
    return sign(res' * (model.alpha .* model.y) + model.b)[1]
end

function error_(i,model)
    return predict_row(model.X[i,:],model) - model.y[i]
end



function test_svm()
    X_train, X_test, y_train, y_test = make_cla(n_features = 14)
    predictions = 0
    for kernel in ["linear", "rbf"]
        model = svm(X_train, y_train, max_iter=500, kernel=kernel, C=0.6)
        train!(model)
        predictions = predict(model,X_test)
        println("Classification accuracy $(kernel): $(accuracy(y_test, predictions))")
        
    end
    #PCA
    pca_model = PCA()
    train!(pca_model, X_test)
    plot_in_2d(pca_model, X_test, predictions, "svm")
end






