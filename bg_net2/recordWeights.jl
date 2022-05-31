include("experiment.jl")


function record_weights(net)
    weights = []
    for postPop = pop_order(net)
        for prePop = pop_order(net)
            if net[prePop] in keys(net[postPop].projections)
                for post=1:size(net[postPop])
                    for synapse=net[postPop].projections[net[prePop]][post]
                        push!(weights, synapse.weight)
                    end
                end
            end
        end
    end
    return weights
end

function annotate_weights(net)
    synapses = []
    i = 1
    for postPop = pop_order(net)
        for prePop = pop_order(net)
            if net[prePop] in keys(net[postPop].projections)
                for post=1:size(net[postPop])
                    for synapse=net[postPop].projections[net[prePop]][post]
                        if typeof(synapse) == StaticSynapse
                            push!(synapses, (i, postPop, post, prePop, Int64(synapse.pre),
                                             "Static", synapse.weight > 0 ? "E" : "I"))
                        elseif typeof(synapse) == EligabilitySynapse{1}
                            push!(synapses, (i, postPop, post, prePop, Int64(synapse.pre),
                                             "Plastic", "E"))
                        elseif typeof(synapse) == EligabilitySynapse{-1}
                            push!(synapses, (i, postPop, post, prePop, Int64(synapse.pre),
                                             "Plastic", "I"))
                        end
                        i += 1
                    end
                end
            end
        end
    end
    return [NamedTuple{(:synapse_id, :post_pop, :post_neuron_id, :pre_pop,
                        :pre_neuron_id, :plasticity, :sign)}(s) for s in synapses]
end



const DEFAULT_PARAMS = (net_size=200, trial_length=200,
                        target_dim=4, target_tau=20.0, lambda=0.1,
                        n_trials=5000, striatumUpdate=:dopamine,
                        learning_rate=1e-3, feedback_factor=25,
                        synapseType=EligabilitySynapse, n_varicosities=10,
                        repetition=1)

net_size = DEFAULT_PARAMS.net_size
trial_length = DEFAULT_PARAMS.trial_length
target_dim = DEFAULT_PARAMS.target_dim
target_tau = DEFAULT_PARAMS.target_tau
lambda = DEFAULT_PARAMS.lambda
n_trials = DEFAULT_PARAMS.n_trials
learning_rate = DEFAULT_PARAMS.learning_rate
feedback_factor = DEFAULT_PARAMS.feedback_factor
synapseType = DEFAULT_PARAMS.synapseType
n_varicosities = DEFAULT_PARAMS.n_varicosities

net = BgNet(net_size, target_dim, learning_rate, learning_rate*feedback_factor; lambda=lambda, SynapseType=synapseType, n_varicosities=n_varicosities)
input = create_input(size(net[:thal]), trial_length)
target = 0.5 .+ 0.15*gaussianProcessTarget(trial_length, target_dim, target_tau)
losses = Float64[]
weights = Vector{Vector{Float64}}()
@showprogress for trial_id=1:n_trials
    loss = run_trial(net, target, input, :dopamine)
    push!(losses, loss)
    push!(weights, record_weights(net))
end


weight_meta = DataFrame(annotate_weights(net))
CSV.write("data/weight_tracking/weight_meta.csv", weight_meta)

weights_as_mat = hcat(weights...)'
size(weights_as_mat)

weight_file = MAT.matopen("data/weight_tracking/weights.mat", "w")
write(weight_file, "weights", weights_as_mat)
close(weight_file)
