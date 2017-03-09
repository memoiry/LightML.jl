
#Partition a dataset into K clusters.

#Finds clusters by repeatedly assigning each data point to the cluster with
#the nearest centroid and iterating until the assignments converge (meaning
#they don't change during an iteration) or the maximum number of iterations
#is reached.

#Parameters
#----------
#K : int
#    The number of clusters into which the dataset is partitioned.
#max_iters: int
#    The maximum iterations of assigning points to the nearest cluster.
#    Short-circuited by the assignments converging on their own.
#init: str, default 'random'
#    The name of the method used to initialize the first clustering.
#
#   'random' - Randomly select values from the dataset as the K centroids.
#    '++' - Select a random first centroid from the dataset, then select
#           K - 1 more centroids by choosing values from the dataset with a
#           probability distribution proportional to the squared distance
#           from each point's closest existing cluster. Attempts to create
#           larger distances between initial clusters to improve convergence
#           rates and avoid degenerate cases.


type Kmeans
    clusters::Dict{Integer,Matrix}
    clu_ind::Dict{Integer,Vector}
    k::Integer
    X::Matrix
    max_iter::Integer
    centroid::Matrix
    init::String
end

function Kmeans(;
                clusters::Dict{Integer,Matrix} = Dict{Integer,Matrix}(),
                k::Integer = 2,
                X::Matrix = zeros(2,2),
                max_iter::Integer = 150,
                centroid::Matrix = zeros(2,2),
                init::String = "++",
                clu_ind::Dict{Integer,Vector} = Dict{Integer,Vector}())

    return Kmeans(clusters, clu_ind, k, X, max_iter, centroid , init)
end


function train!(model::Kmeans, X::Matrix)
    model.X = X 
end

function predict!(model::Kmeans)

    initialize_centroid(model)
    for i in 1:model.max_iter
        centroid_old = copy(model.centroid)
        assign_clusters!(model)
        update_centroid!(model)
        if is_converged(centroid_old, model.centroid)
            break 
        end
    end
end

function update_centroid!(model)
    @show model.centroid
    for i = 1:model.k
        model.centroid[i,:] = mean(model.clusters[i],1)
    end
    @show model.centroid
end


function assign_clusters!(model::Kmeans)
    n = size(model.X,1)
    model.clusters = Dict{Integer,Matrix}()
    model.clu_ind = Dict{Integer,Vector}()
    for i = 1:n 
        dist = zeros(model.k)
        for j = 1:length(dist)
            dist[j] = norm(model.centroid[j,:]-model.X[i,:])
        end
        clu = indmin(dist)
        if haskey(model.clusters,clu)
            model.clusters[clu] = vcat(model.clusters[clu], model.X[i,:]')
            model.clu_ind[clu] = vcat(model.clu_ind[clu], i)
        else
            model.clusters[clu] = model.X[i,:]'
            model.clu_ind[clu] = [i]
        end

    end
end


function initialize_centroid(model::Kmeans)
    model.centroid = zeros(model.k, size(model.X,2))
    if model.init == "random"
        model.centroid = model.X[randperm(size(model.X,1))[1:model.k],:]
    elseif model.init == "++"
        model.centroid[1,:] = model.X[rand(1:size(model.X,1)),:]
        for i = 2:model.k
            model.centroid[i,:] = find_next_centroid(model,i-1)
        end
    else
        error("you must provide a initial type")
    end

end

function find_next_centroid(model, num)
    mean_cent = vec(mean(model.centroid[1:num,:],1))
    n_sample = size(model.X,1)
    res = zeros(n_sample)
    for i = 1:n_sample
        res[i] = norm(model.X[i,:]-mean_cent)
    end
    prob = res/sum(res)
    cum = cumsum(prob, 1)
    r = rand()
    x_sel = model.X[cum .> r,:]
    x_sel = x_sel[1,:]
    return x_sel
end



function is_converged(x::Matrix,
                      y::Matrix)
    return norm(x-y) == 0 
end



function plot_!(model::Kmeans)
    y_ = []
    for i = 1:model.k
        push!(y_,model.clu_ind[i])
    end
    y_ = reduce(vcat,y_)
    x_ = zeros(size(y_,1),2)
    for i = 1:size(x_,1)
        x_[i,:] = model.X[y_[i],:]
    end
    @show size(x_)
    x_sep = x_[:,1]
    y_sep = x_[:,2]
    num_ = zeros(model.k)
    for i = 1:model.k
        num_[i] = size(model.clusters[i],1)
    end
    num_ = convert(Array{Int,1},num_)
    rep = zeros(sum(num_))
    for i = 1:model.k
        for j = 1:num_[i]
            j = convert(Int,j)
            if i == 1
                rep[j] = i
            else
                kk = i-1
                rep[sum(num_[1:kk])+j] = i
            end
        end
    end
    df = DataFrame(x = x_sep,y = y_sep , cluster = rep)

    plot(df, x = "x", y = "y", color = "cluster",Geom.point)
end


function test_kmeans_speed()
    X, y = make_blo()
    clu = length(unique(y))
    @show clu
    model = Kmeans(k=clu)
    train!(model,X)
    predict!(model)
    plot_!(model)
end


function test_kmeans_random()
    X, y = make_blo()
    clu = length(unique(y))
    @show clu
    model = Kmeans(k=clu,init="random")
    train!(model,X)
    predict!(model)
    plot_!(model)
end










