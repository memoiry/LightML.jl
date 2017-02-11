# lightML.jl


lightML.jl is a collection of reimplementation of general machine learning algorithm in Julia

## To-do List

- [x] Support vector machine(SVM)
- [ ] Linear regression
- [ ] Logistic regression
- [ ] Neural network
- [ ] K-Nearst neighbors
- [ ] K-Means 
- [ ] Naive bayes
- [ ] Random Forests
- [ ] PCA
- [ ] Factorization machines
- [ ] Restricted Boltzmann machine
- [ ] Gradient Boosting trees
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



