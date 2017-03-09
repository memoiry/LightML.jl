module LightML


using Gadfly
using DataFrames
using ForwardDiff
using Distributions
using PyCall

@pyimport sklearn.datasets as dat



export 

    test_classification, 
    test_regression, 

    test_decision_tree,

    test_GDA,
    test_HMM,

    test_kneast_regression,
    test_kneast_classification,

    test_label_propagation,

    test_LDA,

    test_naive,

    test_NeuralNetwork,

    test_svm,

    test_GaussianMixture,

    test_kmeans_random,
    test_kmeans_speed,

    test_PCA,

    test_spec_cluster






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

#Unsupervised_learning 

include("unsupervised_learning/gaussianMixtureModel.jl")
include("unsupervised_learning/kMeans.jl")
include("unsupervised_learning/principalComponentAnalysis.jl")
include("unsupervised_learning/spectralCluster.jl")

#Utils

include("utils/utils.jl")




end