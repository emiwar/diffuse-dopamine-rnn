using LinearAlgebra
using Plots
using ProgressMeter
using Statistics
using HDF5
include("bg_net_v2.jl")

T = 200
base_period = 200
target_fcn(t) = 0.25*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                 0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)] .+ .5
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])

T_train = 200000
n_runs = 15

lambdas = 10 .^ (-3:0.25:1)
losses = Dict(lambda=>zeros(div(T_train, base_period), n_runs) for lambda in lambdas)#map(net->Float64[], nets)
for lambda in lambdas
    for r=1:n_runs
        net = BgNet(500, 2, 1e-2; lambda=lambda)
        loss = 0.0
        @showprogress "lambda=$(lambda), run $r: " for t=1:T_train
            loss += step!(net, target_fcn(t), clamp=(thal=input_fcn(t),))
            if mod(t, base_period) == 0
                losses[lambda][div(t, base_period), r] = loss
                loss = 0.0
            end
        end
    end
end
h5open("different_lambdas_v4.h5", "w") do fid
    for lambda in lambdas
        fid["lam$(round(lambda, digits=6))"] = losses[lambda]
    end
end

plot()
for lambda in lambdas
    m = mean(losses[lambda], dims=2)
    s = std(losses[lambda], dims=2)
    plot!(m, label=string(lambda), ribbon=s, fillalpha=.2)
end
plot!(yaxis=:log, xlabel="Episode (or trial)", ylabel="Error (MSE)")
savefig("different_lambdas_v2.png")

#plot(losses.dopamine, label="Dopamine", yaxis=:log, xlabel="Repition (\"trial\")",
#     ylabel="Squared loss")
#plot!(losses.ideal, label="Ideal")
#plot!(losses.no_dopamine, label="No dopamine")
