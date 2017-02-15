include("utils/utils.jl")


type NaiveByes
    
end

function NaiveBayes()

end


function train!(model::NaiveBayes, X::Matrix, y::Vector)

end

function predict(model::NaiveBayes,
                 x::Vector)
end




function test_naive()
    X_train, X_test, y_train, y_test = make_cla()
    model = NaiveBayes()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))
end













