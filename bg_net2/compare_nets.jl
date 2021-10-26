using LinearAlgebra
using Plots
using ProgressMeter
using Statistics
using HDF5
include("bg_net_v2.jl")

nets = (:dopamine, :ideal, :no_dopamine)
#(dopamine=BgNet(500, 2, 2e-2), ideal=BgNet(500, 2, 2e-2),
        #no_dopamine=BgNet(500, 2, 2e-2))

T = 200
base_period = 200
target_fcn(t) = 0.25*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                 0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)] .+ .5
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])

T_train = 250000
n_runs = 3

losses = Dict(key=>zeros(div(T_train, base_period), n_runs) for key in nets)#map(net->Float64[], nets)
for key in nets
    for r=1:n_runs
        net = BgNet(200, 2, 5e-3)
        loss = 0.0
        @showprogress "$(key), run $r: " for t=1:T_train
            loss += step!(net, target_fcn(t), clamp=(thal=input_fcn(t),), updateStriatum=key)
            if mod(t, base_period) == 0
                losses[key][div(t, base_period), r] = loss
                loss = 0.0
            end
        end
    end
end
h5open("comparison_runs_v2.h5", "w") do fid
    for key in nets
        fid[string(key)] = losses[key]
    end
end

plot()
for (i, key) in enumerate(keys(losses))
    m = mean(losses[key], dims=2)
    s = std(losses[key], dims=2)
    plot!(m, color=i, label=string(key), ribbon=s, fillalpha=.2)
end
plot!(yaxis=:log, xlabel="Episode (or trial)", ylabel="Error (MSE)")

#plot(losses.dopamine, label="Dopamine", yaxis=:log, xlabel="Repition (\"trial\")",
#     ylabel="Squared loss")
#plot!(losses.ideal, label="Ideal")
#plot!(losses.no_dopamine, label="No dopamine")
