
type LSC
    n_clusters::Int64
    n_landmarks::Int64
    method::Symbol
    non_zero_landmarks::Int64
    bandwidth::Float64
    cluster_result::Vector
end

function LSC(;
             n_clusters::Int64 = 2,
             n_landmarks::Int64 = 150,
             method::Symbol = :Kmeans,
             non_zero_landmarks::Int64 = 4,
             bandwidth::Float64 = 0.4,
             cluster_result::Vector = zeros(10))

    return LSC(n_clusters, n_landmarks, method, non_zero_landmarks,
               bandwidth, cluster_result)

end

function gaussianKernel(distance, bandwidth)
    exp(-distance / (2*bandwidth^2));
end

function get_landmarks(X, p;method=:Kmeans)
    if(method == :Random)
        numberOfPoints = size(X,2);
        landmarks = X[:,randperm(numberOfPoints)[1:p]];
        return landmarks;
    end

    if(method == :Kmeans)
        kmeansResult = kmeans(X,p)
        return kmeansResult.centers;
    end
    throw(ArgumentError("method can only be :Kmeans or :Random"));
end

function compose_sparse_Z_hat_matrix(X, landmarks, bandwidth, r)

    distances = pairwise(Distances.Euclidean(), landmarks, X);
    similarities = map(x -> gaussianKernel(x, bandwidth), distances);
    ZHat = zeros(size(similarities));

    for i in 1:(size(similarities,2))
        topLandMarksIndices = selectperm(similarities[:,i], 1:r, rev=true);
        topLandMarksCoefficients = similarities[topLandMarksIndices, i];
        topLandMarksCoefficients = topLandMarksCoefficients / sum(topLandMarksCoefficients);
        ZHat[topLandMarksIndices,i] = topLandMarksCoefficients;
    end
    return diagm(sum(ZHat,2)[:])^(-1/2) * ZHat;

end


function train!(model::LSC, X::Matrix)
    if size(X,1) > size(X,2)
        X = transpose(X)
    end
    landmarks = get_landmarks(X, model.n_landmarks, method = model.method)
    Z_hat = compose_sparse_Z_hat_matrix(X, landmarks, model.bandwidth,
                                        model.non_zero_landmarks)
    svd_result = svd(transpose(Z_hat))
    temp = transpose(svd_result[1][:,1:model.n_clusters])
    model.cluster_result = kmeans(temp, model.n_clusters).assignments
end

function plot_in_2d(model::LSC, X::Matrix)
    if size(X,1) > size(X,2)
        X = transpose(X)
    end
    Gadfly.plot(x = X[1,:], y = X[2, :], color = model.cluster_result)
end


function test_LSC()
    path_ = "../data"
    data_name = ["smiley", "spirals", "shapes", "cassini1"]
    datasets = copy(data_name)
    for i in 1:4
        datasets[i] = datasets[i] * ".csv"
    end
    for i in 1:4
        datasets[i] = joinpath(path_, datasets[i])
    end
    clusters = [4, 2, 4, 3]
    n_landmarks = [50, 150, 50, 50]
    bandwidth = [0.4, 0.04, 0.04, 0.4]
    count = 0
    df = DataFrame()
    for i in 1:4
        data = readdlm(datasets[i])
        data = transpose(convert(Array{Float64,2}, data[:,1:2]))
        model = LSC(n_clusters = clusters[i], n_landmarks = n_landmarks[i], bandwidth = bandwidth[i])
        train!(model,data)
        dataframe = DataFrame(x=data[1,:],y=data[2,:],group=model.cluster_result,datasets=data_name[i])
        if count == 0
            df = dataframe
            count += 1
        else
            df = vcat(df, dataframe)
        end
        println("Progress: $(i/4*100)%....")
        #println("$(df)")
    end
    println("$(df)")
    println("computing finied, drawing the plot......")
    set_default_plot_size(25cm, 14cm)
    Gadfly.plot(df, x="x", y = "y", color = "group",xgroup = "datasets", Geom.subplot_grid(Geom.point), Guide.title("Large Scale Spectral Clustering"))
end













