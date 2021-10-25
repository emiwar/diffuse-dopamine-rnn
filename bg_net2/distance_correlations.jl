using Statistics
using Plots
using ProgressMeter
using HDF5
include("bg_net_v2.jl")



T = 200
base_period = 200
target_fcn(t) = 0.25*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                 0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)] .+ .5
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])


for rep=1:100
    net = BgNet(500, 2, 1e-2)
    recordSampleRun(net, T, clamp=(thal=input_fcn,))

    T_train = 200000
    losses = Float64[]
    loss = 0.0
    @showprogress "Rep $rep " for t=1:T_train
        loss += step!(net, target_fcn(t), clamp=(thal=input_fcn(t),))
        #loss += sum((net[:snr].r .- target_fcn(t)).^2)
        #net[:snr].r = target_fcn(t)
        if mod(t, base_period) == 0
            push!(losses, loss)
            loss = 0.0
        end
    end

    run_log = recordSampleRun(net, T, clamp=(thal=input_fcn,))


    distances_dmsn = [norm(net.str_dmsn_pos[i, :]-net.str_dmsn_pos[j,:]) for i=1:250, j=1:250]
    correlations_dmsn = cor(run_log.str_dmsn')'
    distances_imsn = [norm(net.str_imsn_pos[i, :]-net.str_imsn_pos[j,:]) for i=1:250, j=1:250]
    correlations_imsn = cor(run_log.str_imsn')'
    distances_cross = [norm(net.str_dmsn_pos[i, :]-net.str_imsn_pos[j,:]) for i=1:250, j=1:250]
    correlations_cross = cor(run_log.str_dmsn', run_log.str_imsn')'
    h5open("dist_vs_corr.h5", "cw") do fid
        fid["rep$(rep)/dmsn/distances"] = distances_dmsn[:]
        fid["rep$(rep)/dmsn/correlations"] = correlations_dmsn[:]
        fid["rep$(rep)/imsn/distances"] = distances_imsn[:]
        fid["rep$(rep)/imsn/correlations"] = correlations_imsn[:]
        fid["rep$(rep)/cross/distances"] = distances_cross[:]
        fid["rep$(rep)/cross/correlations"] = correlations_cross[:]
    end
end

bin_size = 0.025
bin = round.(distances[:]/bin_size)
grouped = Vector{Float64}[]
for b=1:40
    push!(grouped, correlations[:][bin .== b])
end

plot(mean.(grouped))