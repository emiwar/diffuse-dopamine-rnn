using Statistics
using Plots
using ProgressMeter
using HDF5
include("experiment.jl")

net_size=200
trial_length=200
target_dim=4
target_tau=20.0
lambda=0.1
n_trials=5000
striatumUpdate=:dopamine
learning_rate=1e-3
feedback_factor=25


net = BgNet(net_size, target_dim, learning_rate, learning_rate*feedback_factor, lambda=lambda)
input = create_input(size(net[:thal]), trial_length)
target = 0.5 .+ 0.15*gaussianProcessTarget(trial_length, target_dim, target_tau)
losses = Float64[]
@showprogress for trial_id=1:n_trials
    loss = run_trial(net, target, input, striatumUpdate)
    push!(losses, loss)
end
run_log = recordSampleRun(net, net_size, clamp=(thal=t->input[t, :],))

h5open("data/sample_run_with_positions.h5", "cw") do fid
    fid["net_size"] = net_size
    fid["lambda"] = lambda
    fid["n_trials"] = n_trials
    fid["reps"] = 1000
    fid["striatumUpdate"] = string(striatumUpdate)
    fid["target_dim"] = target_dim
    fid["str_dmsn_pos"] = net.str_dmsn_pos
    fid["str_imsn_pos"] = net.str_imsn_pos
    fid["losses"] = losses
    fid["input"] = collect(input)
    fid["target"] = target
    for k in keys(run_log)
        fid["run_log/$k"] = run_log[k]
    end
end