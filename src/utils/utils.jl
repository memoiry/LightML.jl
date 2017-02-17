using ForwardDiff
using PyCall

@pyimport sklearn.datasets as dat


function one_hot(y)
    n = maximun(y)
    return eye(n+1)[y,]
end


function train_test_split(X,y;train_size=0.75,rand_seed=2135)
    srand(rand_seed)
    rand_indx = shuffle(1:size(X,1))
    train_num = Int(floor(size(X,1) * train_size))
    X_train = X[rand_indx[1:train_num],:]
    X_test = X[rand_indx[(train_num+1):end],:]
    y_train = y[rand_indx[1:train_num]]
    y_test = y[rand_indx[(train_num+1):end]]
    return  X_train, X_test, y_train, y_test
end


function batch_iter(X, batch_size = 64)
    n_sample = size(X,1)
    n_batch = floor(n_sample / (batch_size))
    batch_end = 0

    batch = []
    for i = 1n_batch
        batch_begin = (i-1)*n_batch
        batch_end = i * n_batch
        if i < n_batch
            push!(batch,X[batch_beginbatch_end, ])
        else
            push!(batch,X[batch_beginend, ])
        end
    end
    return batch
end

function euc_dist(x,y)
    return norm(x-y)
end

function l2_dist(X)
    sum_X = sum(X .* X, 1)
    return (-2 * X * X' + sum_X)' + sum_X
end

function unhot(predicted)
    """Convert one-hot representation into one column."""
    actual = []
    for i = 1size(predicted, 1)
        predicted_data = predicted[i,]
        push!(actual,indmax(predicted_data)-1)
    end
    actual = reduce(vcat,actual)
end


function absolute_error(actual, predicted)
    return abs(actual - predicted)
end

function classification_error(actual, predicted)
    if size(actual,2) > 1 && length(actual) > 1 
        actual = unhot(actual)
        predicted = unhot(predicted)
    end
    return sum(actual .!= predicted) / size(actual,1)
end

function accuracy(actual, predicted)
    return 1.0 - classification_error(actual, predicted)
end

function mean_absolute_error(actual, predicted)
    return mean(absolute_error(actual, predicted))
end

function squared_error(actual, predicted)
    return (actual - predicted) .^ 2
end

function squared_log_error(actual, predicted)
    return (log(actual + 1) - log(predicted + 1)) .^ 2
end

function mean_squared_log_error(actual, predicted)
    return mean(squared_log_error(actual, predicted))
end

function mean_squared_error(actual, predicted)
    return mean(squared_error(actual, predicted))
end

function root_mean_squared_error(actual, predicted)
    return sqrt(mean_squared_error(actual, predicted))
end

function root_mean_squared_log_error(actual, predicted)
    return sqrt(mean_squared_log_error(actual, predicted))
end

function logloss(actual, predicted)
    predicted = clamp(predicted, 0, 1 - 1e-15)
    loss = -sum(actual .* log(predicted))
    return loss / size(actual,1)
end

function hinge(actual, predicted)
    return mean(max(1. - actual .* predicted, 0.))
end

function binary_crossentropy(actual, predicted)
    predicted = clamp(predicted, 1e-15, 1 - 1e-15)
    return mean(-sum(actual .* log(predicted) +
                           (1 - actual) .* log(1 - predicted)))
end


function sigmoid(x)
    return 0.5 * (tanh(x) + 1)
end

function make_cla()
    X, y = dat.make_classification(n_samples=1200, n_features=10, n_informative=5,
                               random_state=1111, n_classes=2, class_sep=1.75,)
    # Convert y to {-1, 1}
    y = (y * 2) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        rand_seed=1111)
    X_train, X_test, y_train, y_test
end

function make_reg()
    X, y = dat.make_regression(n_samples=10000, n_features=100,
                           n_informative=75, n_targets=1, noise=0.05,
                           random_state=1111, bias=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        rand_seed=1111)
    X_train, X_test, y_train, y_test
end

function make_blo()
    X, y = dat.make_blobs(centers=4, n_samples=500, n_features=2,
                          random_state=42)
    return  X, y
end





