# lightML.jl


lightML.jl is a collection of reimplementation of general machine learning algorithm in Julia. 

The purpose of this project is purely self-educational.

# Why?

This project is targeting people who want to learn internals of ml algorithms or implement them from scratch.

The code is much easier to follow than the optimized libraries and easier to play with.

All algorithms are implemented in Julia.

## To-do List

### Supervised Learning

- [x] Support vector machine
- [x] Linear regression
- [x] Logistic regression
- [x] Lasso regression
- [x] Ridge regression
- [x] K-Nearst neighbors
- [x] Naive bayes
- [x] Neural network
- [x] Hidden Markov Model
- [x] Gaussian discriminant analysis
- [x] Decision tree
- [x] LDA
- [x] Label propagation
- [ ] Multi-class LDA
- [ ] Random Forests
- [ ] AdaBoost
- [ ] Gradient Boosting trees
- [ ] Factorization machines
- [ ] Restricted Boltzmann machine
- [ ] t-Distributed Stochastic Neighbor Embedding

### Unsupervised Learning:

- [x] K-Means 
- [x] spectral clustering
- [x] PCA
- [x] SVD
- [x] Gaussian mixture model
- [ ] Apriori
- [ ] Partitioning Around Medoids


## API

### SVM example

```julia
function test_test()
    X, y = dat.make_classification(n_samples=1200, n_features=10, n_informative=5,
                               random_state=1111, n_classes=2, class_sep=1.75,)
    # Convert y to {-1, 1}
    y = (y * 2) - 1
    X_train, X_test, y_train, y_test = make_cla()

    for kernel in ["linear", "rbf"]
        model = svm(X_train, y_train, max_iter=500, kernel=kernel, C=0.6)
        train(model)
        predictions = predict(model,X_test)
        println("Classification accuracy $(kernel): $(accuracy(y_test, predictions))")
    end
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

### PCA


```julia
function test_PCA()
    X_train, X_test, y_train, y_test = make_cla()
    model = PCA()
    train!(model,X_train)
    X_reduced = transform(model, X_train)
    plot_(X_reduced, y_train)
end
```

![](https:\/\/ooo.0o0.ooo\/2017\/03\/03\/58b8c8ddc195b.png)

### Label propagation

For label propagation you could just do as follow.

For more detail you can just access [Labelpropagation.jl](https://github.com/memoiry/labelPropagation.jl).

```julia
Pkg.clone("https://github.com/memoiry/labelPropagation.jl")
using labelPropagation
num_unlabel_samples = 800  
Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples) 
iter = round(linspace(1,70,5))
res = []
for i in iter
    unlabel_data_labels = label_propagation(Mat_Label, Mat_Unlabel, labels, kernel_type = "knn", knn_num_neighbors = 10, max_iter = i)
    push!(res, unlabel_data_labels)
end
res = reduce(hcat, res)
show_example(Mat_Label, labels, Mat_Unlabel, res)  
```
![](https:\/\/ooo.0o0.ooo\/2017\/02\/06\/58975f6f57770.png)


