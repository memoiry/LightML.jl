
function show_example(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)      
    mat = vcat(Mat_Label, Mat_Unlabel)
    label = vcat(labels, unlabel_data_labels)
    df = DataFrame(x = mat[:,1], y = mat[:,2], class = label)
    println("Computing finished")
    println("Drawing the plot.....Please Wait(Actually Gadfly is quite slow in drawing the first plot)")
    Gadfly.plot(df, x = "x", y = "y", color = "class", Geom.point)
end


function show_example(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels :: Array{Float64,2})      
    iter_size = size(unlabel_data_labels,2)
    num_size = size(unlabel_data_labels, 1) + size(labels,1)
    mat = vcat(Mat_Label, Mat_Unlabel)
    mat = repmat(mat,iter_size,1)
    labels = repmat(labels,1, iter_size)
    label = vcat(labels, unlabel_data_labels)
    group = zeros(num_size * iter_size, 1)
    for i = 1:iter_size
        group[((i-1)*num_size+1):num_size*i] = i 
    end
    df = DataFrame(x = mat[:,1], y = mat[:,2], iteration = vec(group), class = label[:])
    println("Computing finished")
    println("drawing the plot....Please Wait")
    Gadfly.plot(df, x = "x", y = "y", xgroup = "iteration", color = "class", Geom.subplot_grid(Geom.point))
end


function loadCircleData(num_data)
    center = [5.0, 5.0]
    radiu_inner = 2  
    radiu_outer = 4  
    num_inner = floor(num_data / 3)
    num_outer = num_data - num_inner  
      
    data = []
    theta = 0.0  
    for i in 1:num_inner
        pho = (theta % 360) * pi / 180  
        tmp = zeros(1,2)  
        tmp[1] = radiu_inner * cos(pho) + rand() + center[1]  
        tmp[2] = radiu_inner * sin(pho) + rand() + center[2]  
        push!(data,tmp)  
        theta += 2  
    end

    theta = 0.0 
    for i in 1:num_outer
        pho = (theta % 360) * pi / 180  
        tmp = zeros(1,2)
        tmp[1] = radiu_outer * cos(pho) + rand() + center[1]  
        tmp[2] = radiu_outer * sin(pho) + rand() + center[2]  
        push!(data,tmp)  
        theta += 1  
    end
      
    Mat_Label = zeros(2, 2)
    Mat_Label[1,:] = center + [ 0.5 - radiu_inner , 0]
    Mat_Label[2,:] = center + [ 0.5 - radiu_outer , 0]
    labels = [1, 2]  
    Mat_Unlabel = reduce(vcat,data)  
    return Mat_Label, labels, Mat_Unlabel  
end
  
function loadBandData(num_unlabel_samples)
    #Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])  
    #labels = [0, 1]  
    #Mat_Unlabel = np.array([[5.1, 2.], [5.0, 8.1]])  
      
    Mat_Label = [5.0 2;5.0 8.0]
    labels = [1, 2]  
    num_dim = size(Mat_Label,2)  
    Mat_Unlabel = zeros(num_unlabel_samples, num_dim)
    Mat_Unlabel[num_unlabel_samples/2, :] = (rand(num_unlabel_samples/2, num_dim) - 0.5) * np.array([3, 1]) + Mat_Label[0]  
    Mat_Unlabel[num_unlabel_samples/2 : num_unlabel_samples, :] = (np.random.rand(num_unlabel_samples/2, num_dim) - 0.5) * np.array([3, 1]) + Mat_Label[1]  
    return Mat_Label, labels, Mat_Unlabel  
end


function navie_knn(dataSet, query, k)
    numSamples = size(dataSet,1)
    ## step 1: calculate Euclidean distance
    diff = repmat(query,1,numSamples)' - dataSet
    squaredDiff = diff.^2
    squaredDist = vec(sum(squaredDiff, 2))

    sortedDistIndices = sortperm(squaredDist)
    if k > length(sortedDistIndices)
        k = length(sortedDistIndices)
    end
    return sortedDistIndices[1:k]
end


# build a big graph (normalized weight matrix)
function buildGraph(MatX, kernel_type, rbf_sigma = nothing, knn_num_neighbors = nothing)
    num_samples = size(MatX,1)
    affinity_matrix = zeros(num_samples, num_samples)
    if kernel_type == "rbf"
        if rbf_sigma == nothing
          error("You should input a sigma of rbf kernel!")
        end
        for i in 1:num_samples
            row_sum = 0.0
            for j in 1:num_samples
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i,j] = exp(sum(diff.^2) / (-2.0 * rbf_sigma^2))
                row_sum += affinity_matrix[i,j]
            end
            affinity_matrix[i,:] /= row_sum
        end
    elseif kernel_type == "knn"
        if knn_num_neighbors == nothing
          error("You should input a k of knn kernel!")
        end
        for i in 1:num_samples
          k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors)
          affinity_matrix[i, k_neighbors] = 1.0 / knn_num_neighbors
        end
    else
        erro("Not support kernel type! You can use knn or rbf!")
    end

    return affinity_matrix
end



# label propagation
function label_propagation(Mat_Label, Mat_Unlabel, labels; kernel_type = "rbf", rbf_sigma = 1.5,
                    knn_num_neighbors = 10, max_iter = 500, tol = 1e-3)
    # initialize
    num_label_samples = size(Mat_Label,1)
    num_unlabel_samples = size(Mat_Unlabel,1)
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = unique(labels)
    num_classes = length(labels_list)

    MatX = vcat(Mat_Label, Mat_Unlabel)
    clamp_data_label = zeros(num_label_samples, num_classes)
    if any(labels == 0)
      for i in 1:num_label_samples
        clamp_data_label[i,labels[i]+1] = 1.0
      end
    else
      for i in 1:num_label_samples
        clamp_data_label[i,labels[i]] = 1.0
      end
    end
    label_function = zeros(num_samples, num_classes)
    label_function[1:num_label_samples,:] = clamp_data_label
    label_function[num_label_samples+1:num_samples,:] = -1

    # graph construction
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)

    # start to propagation
    iter = 0;
    pre_label_function = zeros(num_samples, num_classes)
    changed = sum(abs(pre_label_function - label_function))
    while iter < max_iter && changed > tol
      ##if iter % 1 == 0
      #  println("---> Iteration $(iter)/$(max_iter), changed: $changed")
      #end
      pre_label_function = label_function
      iter += 1

      # propagation
      label_function = affinity_matrix * label_function

      # clamp
      label_function[1 : num_label_samples, :] = clamp_data_label

      # check converge
      changed = sum(abs(pre_label_function - label_function))
    end
    # get terminate label of unlabeled data
    unlabel_data_labels = zeros(num_unlabel_samples)
    for i in 1:num_unlabel_samples
      unlabel_data_labels[i] = indmax(label_function[i+num_label_samples,:])
    end
    return unlabel_data_labels
  end



# if affinity_matrix is given 
function label_propagation(affinity_matrix, labels; kernel_type = "rbf", rbf_sigma = 1.5,
                    knn_num_neighbors = 10, max_iter = 500, tol = 1e-3)
    # initialize
    num_label_samples = size(labels,1)
    num_samples = size(affinity_matrix,1)
    num_unlabel_samples = num_samples - num_label_samples
    labels_list = unique(labels)
    num_classes = length(labels_list)

    clamp_data_label = zeros(num_label_samples, num_classes)
    if any(labels == 0)
      for i in 1:num_label_samples
        clamp_data_label[i,labels[i]+1] = 1.0
      end
    else
      for i in 1:num_label_samples
        clamp_data_label[i,labels[i]] = 1.0
      end
    end
    label_function = zeros(num_samples, num_classes)
    label_function[1:num_label_samples,:] = clamp_data_label
    label_function[num_label_samples+1:num_samples,:] = -1

    # start to propagation
    iter = 0;
    pre_label_function = zeros(num_samples, num_classes)
    changed = sum(abs(pre_label_function - label_function))
    while iter < max_iter && changed > tol
      ##if iter % 1 == 0
      #  println("---> Iteration $(iter)/$(max_iter), changed: $changed")
      #end
      pre_label_function = label_function
      iter += 1

      # propagation
      label_function = affinity_matrix * label_function

      # clamp
      label_function[1 : num_label_samples, :] = clamp_data_label

      # check converge
      changed = sum(abs(pre_label_function - label_function))
    end
    # get terminate label of unlabeled data
    unlabel_data_labels = zeros(num_unlabel_samples)
    for i in 1:num_unlabel_samples
      unlabel_data_labels[i] = indmax(label_function[i+num_label_samples,:])
    end
    return unlabel_data_labels
  end


function test_label_propagation()
  num_unlabel_samples = 800  
  Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples) 
  iter = round(linspace(1,70,5))
  res = []
  for i in iter
      unlabel_data_labels = label_propagation(Mat_Label, Mat_Unlabel, labels, kernel_type = "knn", knn_num_neighbors = 10, max_iter = i)
      push!(res, unlabel_data_labels)
  end
  res = reduce(hcat, res)
  show_example(Mat_Label, labels, Mat_Unlabel, res)  
end


