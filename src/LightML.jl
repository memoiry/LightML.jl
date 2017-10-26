module LightML

using Gadfly
using DataFrames
using ForwardDiff
using Distributions
using PyCall
using PyPlot
using DataStructures
using Distances
using Clustering



@pyimport sklearn.datasets as dat



export 

    test_LinearRegression, 
    test_LogisticRegression, 

    test_ClassificationTree,
    test_RegressionTree,

    test_GDA,
    test_HMM,

    test_kneast_regression,
    test_kneast_classification,


    test_label_propagation,

    test_LDA,
    test_LDA_reduction,

    test_naive,

    test_NeuralNetwork,

    test_svm,

    test_GaussianMixture,

    test_kmeans_random,
    test_kmeans_speed,

    test_PCA,

    test_spec_cluster,

    test_Adaboost,
    test_BoostingTree,
    test_randomForest,

    make_cla,
    make_reg,
    make_digits,
    make_blo,
    make_iris,

    train!,
    predict,

    RegressionTree,
    ClassificationTree,
    randomForest,
    Adaboost,
    LinearRegression,
    LogisticRegression,
    GDA,
    BoostingTree,
    KnnRegression,
    KnnClassifier,
    label_propagation,
    show_example,
    LDA,
    plot_in_2d,
    NaiveBayes,
    NeuralNetwork,
    svm,
    GaussianMixture,
    Kmeans,
    PCA,
    transform,
    spec_clustering




Features = Union{String, Real}



#Supervised_learning 

include("supervised_learning/baseRegression.jl")
include("supervised_learning/decisionTree.jl")
include("supervised_learning/gaussianDiscriminantAnalysis.jl")
include("supervised_learning/hiddenMarkovModel.jl")
include("supervised_learning/kNearestNeighbors.jl")
include("supervised_learning/labelPropagation.jl")
include("supervised_learning/linearDiscriminantAnalysis.jl")
include("supervised_learning/naivdBayes.jl")
include("supervised_learning/neuralNetwork_bp.jl")
include("supervised_learning/support_vector_machine.jl")
include("supervised_learning/adaboost.jl")
include("supervised_learning/randomForests.jl")
include("supervised_learning/xgboost.jl")
include("supervised_learning/GradientBoostingTree.jl")


#Unsupervised_learning 

include("unsupervised_learning/gaussianMixtureModel.jl")
include("unsupervised_learning/kMeans.jl")
include("unsupervised_learning/principalComponentAnalysis.jl")
include("unsupervised_learning/spectralCluster.jl")
include("unsupervised_learning/largeScaleSpectralClustering.jl")

#Utils

include("utils/utils.jl")
include("utils/demo.jl")





end