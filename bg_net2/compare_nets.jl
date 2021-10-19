using LinearAlgebra
using Plots
using ProgressMeter
include("bg_net_v2.jl")

nets = (dopamine=BgNet(500, 2, 1e-2), ideal=BgNet(500, 2, 1e-2),
        no_dopamine=BgNet(500, 2, 1e-2))

T = 200
base_period = 200
target_fcn(t) = 0.25*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                 0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)] .+ .5
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])

T_train = 200000

losses = map(net->Float64[], nets)
for key in keys(nets)
    net = nets[key]
    loss = 0.0
    @showprogress for t=1:T_train
        loss += step!(net, target_fcn(t), clamp=(thal=input_fcn(t),), updateStriatum=key)
        if mod(t, base_period) == 0
            push!(losses[key], loss)
            loss = 0.0
        end
    end
end
plot(losses.dopamine, label="Dopamine", yaxis=:log, xlabel="Repition (\"trial\")",
     ylabel="Squared loss")
plot!(losses.ideal, label="Ideal")
plot!(losses.no_dopamine, label="No dopamine")
