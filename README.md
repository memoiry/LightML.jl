# lightML.jl


lightML.jl is a collection of reimplementation of general machine learning algorithm in Julia. I will update it for learning purpose.

# Why?

This project is targeting people who want to learn internals of ml algorithms or implement them from scratch.
The code is much easier to follow than the optimized libraries and easier to play with.
All algorithms are implemented in Julia.

## To-do List

- [x] Support vector machine(SVM)
- [x] Linear regression
- [x] Logistic regression
- [x] Lasso regression
- [x] Ridge regression
- [x] spectral clustering
- [x] K-Nearst neighbors
- [x] K-Means 
- [x] Naive bayes
- [x] Neural network
- [x] Hidden Markov Model
- [x] Gaussian discriminant analysis
- [x] Gaussian mixture model
- [x] Decision tree
- [ ] Random Forests
- [ ] AdaBoost
- [ ] Gradient Boosting trees
- [x] PCA
- [x] SVD
- [x] LDA
- [ ] Factorization machines
- [ ] Restricted Boltzmann machine
- [ ] t-Distributed Stochastic Neighbor Embedding

## API

### SVM example

```julia

include("src/MLsvm.jl")
X, y = dat.make_classification(n_samples=1200, n_features=10,n_informative=5,random_state=1111,n_classes=2, class_sep=1.75,)

# Convert y to {-1, 1}
y = (y * 2) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        rand_seed=1111)

for kernel in ["linear", "rbf"]
    model = svm(X_train, y_train, max_iter=500, kernel=kernel, C=0.6)
    train(model)
    predictions = predict(model,X_test)
    println("Classification accuracy $(kernel): $(accuracy(y_test, predictions))")
end
```

![](https:\/\/ooo.0o0.ooo\/2017\/02\/11\/589ee68aaf56d.png)

### kmeans

```julia
function test_kmeans_random()
    X, y = make_blo()
    clu = length(unique(y))
    @show clu
    model = Kmeans(k=clu,init="random")
    train!(model,X)
    predict!(model)
    plot_!(model)
end
```

![](https:\/\/ooo.0o0.ooo\/2017\/02\/18\/58a8445e2114b.png)

### LDA

```julia
function test_LDA()
    X_train, X_test, y_train, y_test = make_cla()
    model = LDA()
    plot_in_2d(model, X_train, y_train)
end
```

![](https:\/\/ooo.0o0.ooo\/2017\/03\/02\/58b82861bade3.png)


