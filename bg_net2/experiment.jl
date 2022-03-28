using LinearAlgebra
using ProgressMeter
using Statistics
using Distributions
using IterTools
using HDF5
include("bg_net_v2.jl")

function run_trial(net::BgNet, target::AbstractArray, input::AbstractArray,
                   updateRule::Symbol)
    loss = 0.0
    for t=1:size(target, 1)
        loss += step!(net, target[t, :], clamp=(thal=input[t, :],),
                      updateStriatum=updateRule)
    end
    return loss
end

function create_input(thal_size, trial_length)
    proj = randn(thal_size, 2)
    proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
    theta = 2*pi/trial_length * (1:trial_length)
    return phi.(proj*[cos.(theta) sin.(theta)]')'
end

const DEFAULT_PARAMS = (net_size=200, trial_length=200,
                        target_dim=2, target_tau=20.0, lambda=0.1,
                        n_trials=1000, striatumUpdate=:dopamine,
                        learning_rate=1e-3, feedback_factor=25,
                        synapseType=EligabilitySynapse, repetition=1)

function train_network(desc::String; net_size=200, trial_length=200,
                        target_dim=2, target_tau=20.0, lambda=0.1,
                        n_trials=1000, striatumUpdate=:dopamine,
                        learning_rate=1e-3, feedback_factor=25,
                        synapseType=EligabilitySynapse, repetition=1)
    net = BgNet(net_size, target_dim, learning_rate, learning_rate*feedback_factor, lambda=lambda, SynapseType=synapseType)
    input = create_input(size(net[:thal]), trial_length)
    target = 0.5 .+ 0.15*gaussianProcessTarget(trial_length, target_dim, target_tau)
    losses = Float64[]
    angles = Float64[]
    @showprogress desc for trial_id=1:n_trials
        loss = run_trial(net, target, input, striatumUpdate)
        push!(losses, loss)
        push!(angles, getAlignmentAngle(net))
    end
    return net, losses, angles
end

function run_experiment(fn; params...)
    param_ranges = NamedTuple(params)
    git_commit = readchomp(`git rev-parse --short HEAD`)
    for (i, values) in enumerate(product(param_ranges...))
        named_params = NamedTuple{keys(param_ranges)}(values)
        if named_params.striatumUpdate == :ideal
            named_params = merge(named_params, (;feedback_factor = named_params.feedback_factor*10.0))
        end
        all_params = merge(DEFAULT_PARAMS, named_params)
        net, losses, angles = train_network("$named_params "; all_params...)
        h5open(fn, "cw") do fid
            fid["run$i"] = losses
            for k in keys(all_params)
                v = all_params[k]
                if v isa Number
                    attributes(fid["run$i"])[string(k)] = v
                else
                    attributes(fid["run$i"])[string(k)] = string(v)
                end
            end
            attributes(fid["run$i"])["git"] = git_commit
        end
    end
end
