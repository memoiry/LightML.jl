# LightML.jl

### About
LightML.jl is a collection of reimplementation of general machine learning algorithm in Julia. 

The purpose of this project is purely self-educational.

### Why?

This project is targeting people who want to learn internals of ml algorithms or implement them from scratch.

The code is much easier to follow than the optimized libraries and easier to play with.

All algorithms are implemented in Julia. 

You should access test function of every implementation for its usage in detail. Every model is actually constructed in a similar manner.

### Installation

```julia
Pkg.clone("https://github.com/memoiry/LightML.jl")
```

### Running Implementations

```julia
using LightML
test_label_propagation()
```

![](https:\/\/ooo.0o0.ooo\/2017\/02\/06\/58975f6f57770.png)



## Current Implementations

#### Supervised Learning:
- [Adaboost](src/supervised_learning/adaboost.jl)
- [Decision Tree](src/supervised_learning/decisionTree.jl)
- [Gradient Boosting](src/supervised_learning/GradientBoostingTree.jl)
- [K Nearest Neighbors](src/supervised_learning/kNearestNeighbors.jl)
- [Linear Discriminant Analysis](src/supervised_learning/linearDiscriminantAnalysis.jl)
- [Linear Regression](src/supervised_learning/baseRegression.jl)
- [Logistic Regression](src/supervised_learning/baseRegression.jl)
- [Multilayer Perceptron](src/supervised_learning/neuralNetwork_bp.jl)
- [Naive Bayes](src/supervised_learning/naivdBayes.jl)
- [Ridge Regression](src/supervised_learning/baseRegression.jl)
- [Lasso Regression](src/supervised_learning/baseRegression.jl)
- [Support Vector Machine](src/supervised_learning/support_vector_machine.jl)
- [Hidden Markov Model](src/supervised_learning/hiddenMarkovModel.jl)
- [Label propagation](src/supervised_learning/labelPropagation.jl)
- [ ] Random Forests	
- [ ] XGBoost

#### Unsupervised Learning:

- [Gaussian Mixture Model](src/unsupervised_learning/gaussianMixtureModel.jl)
- [K-Means](src/unsupervised_learning/kMeans.jl)
- [Principal Component Analysis](src/unsupervised_learning/principalComponentAnalysis.jl)
- [ ] Apriori

#### Test Example available 

- test_ClassificationTree()
- test_RegressionTree()
- test_label_propagation()
- test_LDA()
- test_naive()
- test_NeuralNetwork()
- test_svm()
- test_kmeans_random()
- test_PCA()
- test_Adaboost()
- test_BoostingTree()
- test_spec_cluster()
- test_LogisticRegression()
- test_LinearRegression()
- test_kneast_regression()
- test_kneast_classification()
- test_GaussianMixture() **(Fixing)**
- test_GDA() **(Fixing)**
- test_HMM() **(Fixing)**

## Example

### LinearRegression

![](https:\/\/ooo.0o0.ooo\/2017\/03\/11\/58c2cf6a8726e.png)

### Adaboost

![](https:\/\/ooo.0o0.ooo\/2017\/03\/11\/58c2cf69813bb.png)


### SVM

```julia
function test_svm()
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
    plot_in_2d(model)
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
    plot_in_2d(X_reduced, y_train)
end
```

![](https:\/\/ooo.0o0.ooo\/2017\/03\/03\/58b8c8ddc195b.png)




