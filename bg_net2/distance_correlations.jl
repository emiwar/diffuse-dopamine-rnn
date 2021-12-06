using Statistics
using Plots
using ProgressMeter
using HDF5
include("experiment.jl")

net_size=200
trial_length=200
target_dim=2
target_tau=20.0
lambda=0.1
n_trials=1000
striatumUpdate=:dopamine
learning_rate=1e-3
feedback_factor=25
for rep = 1:100
    net = BgNet(net_size, target_dim, learning_rate, learning_rate*feedback_factor, lambda=lambda)
    input = create_input(size(net[:thal]), trial_length)
    target = 0.5 .+ 0.15*gaussianProcessTarget(trial_length, target_dim, target_tau)
    losses = Float64[]
    @showprogress "Rep $rep" for trial_id=1:n_trials
        loss = run_trial(net, target, input, striatumUpdate)
        push!(losses, loss)
    end
    run_log = recordSampleRun(net, 200, clamp=(thal=t->input[t, :],))

    distances_dmsn = [norm(net.str_dmsn_pos[i, :]-net.str_dmsn_pos[j,:]) for i=1:100, j=1:100]
    correlations_dmsn = cor(run_log.str_dmsn')'
    distances_imsn = [norm(net.str_imsn_pos[i, :]-net.str_imsn_pos[j,:]) for i=1:100, j=1:100]
    correlations_imsn = cor(run_log.str_imsn')'
    distances_cross = [norm(net.str_dmsn_pos[i, :]-net.str_imsn_pos[j,:]) for i=1:100, j=1:100]
    correlations_cross = cor(run_log.str_dmsn', run_log.str_imsn')'
    h5open("data/dist_vs_corr.h5", "cw") do fid
        fid["rep$(rep)/dmsn/distances"] = distances_dmsn[:]
        fid["rep$(rep)/dmsn/correlations"] = correlations_dmsn[:]
        fid["rep$(rep)/imsn/distances"] = distances_imsn[:]
        fid["rep$(rep)/imsn/correlations"] = correlations_imsn[:]
        fid["rep$(rep)/cross/distances"] = distances_cross[:]
        fid["rep$(rep)/cross/correlations"] = correlations_cross[:]
    end
end
bin_size = 0.1
bin = round.(distances_dmsn[:]/bin_size)
grouped = Vector{Float64}[]
for b=1:maximum(bin)
    push!(grouped, correlations_dmsn[:][bin .== b])
end

plot(bin_size*(1:maximum(bin)), mean.(grouped))