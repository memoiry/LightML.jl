function demo()
    println("+-------------------------------------------+")
    println("|                                           |")
    println("|     LightML.jl demo for digits dataset    |")
    println("|                                           |")
    println("+-------------------------------------------+")
    # ...........
    #  LOAD DATA
    # ...........
    data = dat.load_digits()
    digit1 = 1
    digit2 = 8
    idx = vcat(find(data["target"] .== digit1), find(data["target"] .== digit2))
    y = data["target"][idx]
    # Change labels to {0, 1}
    y[y .== digit1] = -1
    y[y .== digit2] = 1


    X = data["data"][idx,:]
    X = normalize_(X)

    println("Dataset: The Digit Dataset (digits $(digit1) and $(digit2))" )
    # ..........................
    #  DIMENSIONALITY REDUCTION
    # ..........................
    pca = PCA(n_components=5)
    train!(pca, X) # Reduce to 5 dimensions
    X = transform(pca, X)

    # ..........................
    #  TRAIN / TEST SPLIT
    # ..........................
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    rescaled_y_train = (y_train + 1) ./ 2
    rescaled_y_test = (y_test + 1) ./2
    # .......
    #  SETUP
    # .......
    adaboost = Adaboost(n_clf = 10)
    naive_bayes = NaiveBayes()
    knn = KnnClassifier(k=4)
    logistic_regression = LogisticRegression()
    #mlp = NeuralNetwork(hidden = 3)
    decision_tree = ClassificationTree()
    random_forest = randomForest(n_estimators=20)
    support_vector_machine_rbf = svm(X_train, y_train, max_iter=500, kernel="rbf", C=0.6)
    support_vector_machine_linear = svm(X_train, y_train, max_iter=500, kernel="linear", C=0.6)
    #lda = LDA()
    gbc = GradientBoostingClassifier(n_clf=20, learning_rate=.9, max_depth=2)
    #xgboost = XGBoost(n_estimators=50, learning_rate=0.5)

    # ........
    #  TRAIN
    # ........

    println("\tk-nearst")
    train!(knn, X_train, y_train)
    println("\tAdaboost")
    train!(adaboost,X_train, y_train)
    println("\tDecision Tree")
    train!(decision_tree,X_train, y_train)
    println("\tGradient Boosting")
    train!(gbc,X_train, y_train)
    #println("\tLDA")
    #train!(lda,X_train, y_train)
    println("\tLogistic Regression")
    train!(logistic_regression,X_train, rescaled_y_train)
    #println("\tMultilayer Perceptron")
    #train!(mlp,X_train, y_train)
    println("\tNaive Bayes")
    train!(naive_bayes,X_train, y_train)
    #println("\tPerceptron")
    #train!(perceptron,X_train, y_train)
    println("\tRandom Forest")
    train!(random_forest,X_train, y_train)
    println("\tSupport Vector Machine with rbf kernel")
    train!(support_vector_machine_rbf)
    println("\tSupport Vector Machine with linear kernel")
    train!(support_vector_machine_linear)
    #println("\tXGBoost")
    #xgboost.fit(X_train, y_train)



    # .........
    #  PREDICT
    # .........
    y_pred = Dict()
    y_pred["Adaboost"] = predict(adaboost, X_test)
    y_pred["Gradient Boosting"] = predict(gbc,X_test)
    y_pred["Naive Bayes"] = predict(naive_bayes, X_test)
    y_pred["K Nearest Neighbors"] = predict(knn, X_test)
    y_pred["Logistic Regression"] = predict(logistic_regression, X_test)
    #y_pred["LDA"] = predict(lda, X_test)
    #y_pred["Multilayer Perceptron"] = sign(predict(mlp, X_test))
    #y_pred["Perceptron"] = perceptron.predict(X_test)
    y_pred["Decision Tree"] = predict(decision_tree, X_test)
    y_pred["Random Forest"] = predict(random_forest, X_test)
    y_pred["Support Vector Machine, rbf kernel"] = predict(support_vector_machine_rbf, X_test)
    y_pred["Support Vector Machine, linear kernel"] = predict(support_vector_machine_linear, X_test)

    #y_pred["XGBoost"] = xgboost.predict(X_test)

    # ..........
    #  ACCURACY
    # ..........
    println("Accuracy:")
    for clf in keys(y_pred)
        if clf == "Logistic Regression"
            temp = accuracy(rescaled_y_test,y_pred[clf])
            println("\t$(clf): $(temp)")
        else
            temp = accuracy(y_test,y_pred[clf])
            println("\t$(clf): $(temp)")
        end
    end




    # .......
    #  PLOT
    # .......

    x1 = X_test[:, 1]
    x2 = X_test[:, 2]
    df = DataFrame(x = x1, y = x2, clu = y_test)
    Gadfly.plot(df, x = "x", y = "y", color = "clu", Geom.point, Guide.title("PCA"))
end
