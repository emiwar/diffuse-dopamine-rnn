include("experiment.jl")

net = BgNet(200, 2, 1e-2, 1e-1)
input = create_input(size(net[:thal]), 200)
target = 0.5 .+ 0.15*gaussianProcessTarget(200, 2, 20)
input_fcn(t) = input[t, :]
target_fcn(t) = target[t, :]
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
#plotTraces(log, target=target_fcn)

losses = Float64[]
@showprogress for trial_id=1:1000
    loss = run_trial(net, target, input, :dopamine)
    push!(losses, loss)
end

log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))

h5open("data/trained_net_with_noise.h5", "cw") do fid
    for p in pop_order(net)
        fid["activity/$p"] = log[p]
    end
    fid["pos/dmsn_pos"] = net.str_dmsn_pos
    fid["pos/imsn_pos"] = net.str_imsn_pos
    for pre_pop=pop_order(net), post_pop=pop_order(net)
        projs = net[post_pop].projections
        if net[pre_pop] in keys(projs)
            synapses = Tuple{Int64, Int64, Float64}[]
            proj = projs[net[pre_pop]]
            for post=1:length(proj)
                for syn in proj[post]
                    push!(synapses, (syn.pre, post, syn.weight))
                end
            end
            connections = [getindex.(synapses, 1) getindex.(synapses, 2)]
            weights = getindex.(synapses, 3)
            fid["syn/$pre_pop/$post_pop/connections"] = connections
            fid["syn/$pre_pop/$post_pop/weights"] = weights
        end
    end
    
end
