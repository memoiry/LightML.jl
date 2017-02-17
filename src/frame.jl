include("utils/utils.jl")


type ModelName
    
end

function ModelName()

end


function train!(model::ModelName, X::Matrix, y::Vector)

end

function predict(mode::ModelName, 
                 x::Matrix)
    n = size(x,1)
    res = zeros(n)
    for i = 1:n 
        res[i] = predict(model, x[i,:])
    end
    return res
end

function predict(model::ModelName,
                 x::Vector)
end



function test_ModelName()
    X_train, X_test, y_train, y_test = make_cla()
    model = ModelName()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))
end













