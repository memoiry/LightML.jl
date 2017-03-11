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
test_PCA()
```

<p align="center">
    <img src="https:\/\/ooo.0o0.ooo\/2017\/03\/11\/58c36773da5da.png">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 1: The Digit Dataset using PCA
</p>




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

```julia
using LightML
test_LinearRegression()
```

<p align="center">
    <img src="https:\/\/ooo.0o0.ooo\/2017\/03\/11\/58c2cf6a8726e.png">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 2: The regression Dataset using LinearRegression
</p>

### Adaboost

```julia
using LightML
test_Adaboost()
```

<p align="center">
    <img src="https:\/\/ooo.0o0.ooo\/2017\/03\/11\/58c36970c58a8.png">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 3: The classification Dataset using Adaboost
</p>



### SVM

```julia
using LightML
test_svm()
```

<p align="center">
    <img src="https:\/\/ooo.0o0.ooo\/2017\/03\/11\/58c367760e76a.png">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 4: The classification Dataset using LinearRegression
</p>

### Classification Tree

```julia
using LightML
test_ClassificationTree()
```

<p align="center">
    <img src="https:\/\/ooo.0o0.ooo\/2017\/03\/11\/58c36775113e6.png">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 5: The digit Dataset using Classification Tree
</p>


### kmeans

```julia
using LightML
test_kmeans_random()
```

<p align="center">
    <img src="https:\/\/ooo.0o0.ooo\/2017\/02\/18\/58a8445e2114b.png">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 6: The blobs Dataset using k-means
</p>

### LDA

```julia
using LightML
test_LDA()
```

<p align="center">
    <img src="https:\/\/ooo.0o0.ooo\/2017\/03\/02\/58b82861bade3.png">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 7: The classification Dataset using LDA
</p>



