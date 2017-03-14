


function one_hot(y)
    n = length(unique(y))
    if minimum(y) == 0
        return eye(n)[(y+1),:]
    elseif minimum(y) == -1 && n == 2
        y = trunc(Int64,(y + 1)/2+1)
        return eye(2)[y,:]

    end
    return eye(n)[y,:]
end

function softmax(x::Vector)
    x = exp(x)
    pos = x./sum(x)
    return indmax(pos)
end

function softmax(X::Matrix)
    n_sample = size(X,1)
    res = zeros(n_sample)
    for i = 1:n_sample 
        res[i] = softmax(X[i,:])
    end
    return res
end

function calc_variance(X)
    n_sample = size(X,1)
    mean_ = repmat(mean(X, 1),n_sample,1)
    de_mean = X - mean_ 
    return 1/n_sample * diag(de_mean' * de_mean)
end


function get_random_subsets(X, y , n_subsets;replacement = true)
    n_sample, n_feature = size(X)

    X_y = hcat(X, y)
    X_y = X_y[shuffle(1:n_sample), :]
    subsample_size = trunc(Int64,n_sample / 2)
    if replacement == true
        subsample_size = n_sample
    end
    sets = zeros(n_subsets,subsample_size, n_feature + 1)
    for i = 1:n_subsets
        idx = sample(1:n_sample, subsample_size, replace = replacement)
        temp = X_y[idx,:]
        sets[i,:,:] = temp
    end
    return sets
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

function calc_entropy(y)
    if size(y,2) > 1
        y = unhot(y)
    end
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
    actual = zeros(size(predicted,1))
    for i = 1:size(predicted, 1)
        predicted_data = predicted[i,:]
        actual[i] = indmax(predicted_data)
    end
    return actual
end

function normalize_(X::Matrix)
    std_ = std(X, 1)
    mean_ = mean(X, 1)
    for i = 1:size(X,2)
        if std_[i] != 0
            X[:,i] = (X[:, i] - mean_[i])/std_[i]
        end
    end
    return X
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
    return mean(-sum(actual .* log(predicted) -
                           (1 - actual) .* log(1 - predicted)))
end


function sigmoid_tanh(x)
    return 0.5 * (tanh(x) + 1)
end


function sigmoid(x)
    return 1./(1+exp(-x))
end

function sigmoid_prime(x)
    return sigmoid(x).*(1-sigmoid(x))
end


function make_cla(;n_samples = 1200, n_features = 10, n_classes = 2)
    X, y = dat.make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=1111, n_classes= n_classes)
    # Convert y to {-1, 1}
    y = (y * 2) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        rand_seed=1111)
    X_train, X_test, y_train, y_test
end

function make_reg(;n_samples = 200,
                   n_features = 10)
    X, y = dat.make_regression(n_samples=n_samples, n_features=n_features, n_targets=1, noise=0.05,
                           random_state=1111, bias=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        rand_seed=1111)
    X_train, X_test, y_train, y_test
end

function make_iris()
    data= dat.load_iris()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        rand_seed=1111)
    X_train, X_test, y_train, y_test
end

function make_blo()
    X, y = dat.make_blobs(centers=4, n_samples=500, n_features=2,
                          random_state=42)
    return  X, y
end

function make_digits()
    data = dat.load_digits()
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                         rand_seed=1111)
    X_train, X_test, y_train, y_test
end




