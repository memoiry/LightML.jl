


type PCA
    solver::String
    n_components::Integer
    components::Matrix
    mean_::Vector
end

function PCA(;
             solver::String = "svd",
             n_components::Integer = 2,
             components::Matrix = zeros(4,4),
             mean_::Vector = zeros(4))
    return PCA(solver, n_components, components, mean_)
end


function train!(model::PCA, X::Matrix)
    model.mean_ = vec(mean(X,1))
    n_sample = size(X, 1)
    X_de_mean = X - repmat(model.mean_', n_sample, 1)

    if model.solver == "svd"
        U,S,V = svd(X_de_mean)
    elseif model.solver == "eig"
        cov_ = cov(X_de_mean)
        D,V = eigs(cov_, nev = model.n_components)
    end
    model.components = V[:, 1:2]
end

function transform(model::PCA, 
                 x::Matrix)
    n = size(x,1)
    res = zeros(n, model.n_components)
    for i = 1:n 
        res[i, :] = transform(model, x[i,:])
    end
    return res
end

function transform(model::PCA,
                 x::Vector)
    x = x - model.mean_
    x = vec(x' * model.components)
    return x
end

function plot_(X::Matrix, y::Vector)
    println("computing finished, ploting....")
    x1 = X[:, 1]
    x2 = X[:, 2]
    df = DataFrame(x = x1, y = x2, clu = y)
    plot(df, x = "x", y = "y", color = "clu", Geom.point)
end


function test_PCA()
    X_train, X_test, y_train, y_test = make_cla()
    model = PCA()
    train!(model,X_train)
    X_reduced = transform(model, X_train)
    plot_(X_reduced, y_train)
end













