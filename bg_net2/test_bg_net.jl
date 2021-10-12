using LinearAlgebra
using Plots
using ProgressBars
include("bg_net_v2.jl")

net = BgNet(200, 2, 1e-3)

#plotTraces(net)
#balanceWeights!(net[:ctx_exc])
#balanceWeights!(net[:ctx_inh])
#for pop in pop_order(net)
#    net[pop].v = rand(size(net[pop])) .- 1
#end


T = 200
base_period = 200
target_fcn(t) = 0.25*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                 0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)] .+ .5

#plotTraces(net, T, target=target_fcn)
proj = randn(20, 2)*4
input_fcn(t) = phi.(proj*(target_fcn(t) .- 0.5))
plotTraces(recordSampleRun(net, T, clamp=(thal=input_fcn,)), target=target_fcn)


T_train = 500000
losses = Float64[]
loss = 0.0
for t=ProgressBar(1:T_train)
    loss += step!(net, target_fcn(t), clamp=(thal=input_fcn(t),))
    #loss += sum((net[:snr].r .- target_fcn(t)).^2)
    #net[:snr].r = target_fcn(t)
    if mod(t, base_period) == 0
        push!(losses, loss)
        loss = 0.0
    end
end
plot(losses[1:end])#, ylim=(0, 150))