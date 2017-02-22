include("utils/utils.jl")


type Hmm
    e_to_i::Dict{AbstractString, Integer}
    i_to_e::Dict{Integer, AbstractString}
    s_to_i::Dict{AbstractString, Integer}
    i_to_s::Dict{Integer, AbstractString}
    pi_::Vector
    A::Matrix
    B::Matrix
    data::Matrix
end


function normalize(x)
    return x./sum(x)
end


# forward-backward algorithm for scoring and conditional probability 

# Smoothing.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Ouptut: The posterior probability distribution over each state given all
#         of the observations: P(X(k)|E(0), ..., E(T-1) for 0 <= k <= T-1.
#
#         These distributions should be returned as a list of lists. 

function scoring(model::Hmm)
    n_sample = size(model.data,1)
    n_state = size(model.A, 1)
    alpha = zeros(n_state, n_sample)
    alpha[:, 1] = vec(model.pi_) .* model.B[:,model.e_to_i[model.data[1,2]]]
    for i = 2:size(data,1)
        alpha[:, i] = vec(alpha[:, i-1]' * model.A) .* model.B[:,model.e_to_i[model.data[i, 2]]]
    end
    score_alpha = sum(alpha[:, end])
    beta = zeros(n_state, n_sample)
    beta[:, n_sample] =  ones(3) 
    for i = (n_sample-1):-1:1
        beta[:, i] = model.A * beta[:, i+1] .* model.B[:, model.e_to_i[model.data[i+1, 2]]]
    end
    score_beta = sum(beta[:, 1] .* model.pi_ .* model.B[:, model.e_to_i[model.data[1,2]]])

    phi = alpha .* beta
    for i = 1:size(phi,2)
        phi[:,i] = normalize(phi[:, i])
    end 
    @show phi[:,1:10]
    return phi
end

# vitorbi

function matching(mode::Hmm)
    n_sample = size(model.data,1)
    n_state = size(model.A, 1)
    phi = zeros(n_state, n_sample)
    phi[:, 1] = normalize(model.pi_ .* model.B[:, model.e_to_i[model.data[1,2]]])
    eta = zeros(n_state, n_sample)
    eta[:, 1] = zeros(3)
    for i = 2:n_sample
        for j = 1:n_state 
            temp = phi[:, i-1] .* model.A[:, j]
            eta[j,i] = indmax(temp)
            phi[j,i] = maximum(temp)
        end
        phi[:, i] = phi[:, i] .* model.B[:, model.e_to_i[model.data[i,2]]] 
        phi[:, i] = normalize(phi[:, i])
    end
    @show eta[:, 1:10]
    state_optim = zeros(n_sample)
    state_optim[n_sample] = indmax(phi[:, n_sample])
    for i = (n_sample-1):-1:1
        state_optim[i] = eta[convert(Int,state_optim[i+1]), i+1]
    end
    state_optim = map(x -> model.i_to_s[convert(Int,x)], state_optim)
    return state_optim
end

function learning(model::Hmm)
    
end


function load_data(filename)
    num_ = readline(filename)
    num = parse(Int64, num_)
    data = []
    count = 1
    open(filename, "r") do f
        for line in readlines(f)
            if count == 1
                count += 1
                continue
            end
            str = split(chomp(line), ",")
            push!(data,str')
        end
        data = reduce(vcat,data)
    end
    return data
end


function test_Hmm()
    X_train, X_test, y_train, y_test = make_cla()
    model = Hmm()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    print("classification accuracy", accuracy(y_test, predictions))
end



# state map
weatherStateMap = Dict([("sunny", 1), ("rainy", 2), ("foggy", 3)])
weatherStateIndex = Dict([(1, "sunny"), (2, "rainy"), (3, "foggy")])

# observation map
weatherObsMap = Dict([("no", 1), ("yes", 2)])
weatherObsIndex = Dict([(1, "no"), (2, "yes")])

# prior prob
weatherprob = [0.5,0.25,0.25]

# trasition probabilities

weather_trans = [0.8 0.05 0.15;
                 0.2 0.6 0.2;
                 0.2 0.3 0.5]

# obs 

weather_obs = [0.9 0.1 ;0.2 0.8 ;0.7 0.3]

data = load_data("data/weather-test1-1000.txt")
model = Hmm(weatherObsMap, weatherObsIndex, weatherStateMap, 
            weatherStateIndex, weatherprob,
            weather_trans, weather_obs, data)

# phi = scoring(model)

matching_state = matching(model)











