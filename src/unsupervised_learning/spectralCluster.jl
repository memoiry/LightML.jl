


function spec_clustering(data,k)

    w = computing_similarity(data)
    d = diagm(vec(sum(w,1)))
    l = d-w
    temp = eig(l)
    temp = temp[2]
    e_map = temp[:,1:(k-1)] #seems that the largest k eigenvalue works? 
    #e_map = temp[:,(end-k+1):end]
    model = Kmeans(k=k)
    train!(model,e_map)
    predict!(model)
    return model
end


function computing_similarity(data)
    n_sample = size(data,1)
    w = zeros(n_sample,n_sample)
    for i = 1:n_sample
        for j = 1:n_sample
            w[i,j] = count_sim(data[i,:],data[j,:])
        end
    end
    return w
end

function count_sim(x::Vector,y::Vector;
                   types="Gaussian",
                    gamma = 1)
    if types == "Gaussian"
        return exp(-gamma*norm(x-y)^2)
    end        
end

function test_spec_cluster()

    X, y = make_blo()
    clu = length(unique(y))
    model = spec_clustering(X,clu)
    predictions = model.clusters
    plot_in_2d(model)


end
