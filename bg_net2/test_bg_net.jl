using LinearAlgebra
using Plots
using ProgressMeter
include("bg_net_v2.jl")

net = BgNet(500, 2, 5e-3)

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
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])


plotTraces(recordSampleRun(net, T, clamp=(thal=input_fcn,)), target=target_fcn)


T_train = 50000
losses = Float64[]
loss = 0.0
@showprogress for t=1:T_train
    loss += step!(net, target_fcn(t), clamp=(thal=input_fcn(t),))
    #loss += sum((net[:snr].r .- target_fcn(t)).^2)
    #net[:snr].r = target_fcn(t)
    if mod(t, base_period) == 0
        push!(losses, loss)
        loss = 0.0
    end
end
plot(losses[1:end])#, ylim=(0, 150))


net = BgNet(500, 2, 1e-2)
recordSampleRun(net, T, clamp=(thal=input_fcn,))
p = Progress(100)
@gif for ep=1:100
    next!(p)
    plotTraces(recordSampleRun(net, T, target_fcn, clamp=(thal=input_fcn,)), target=target_fcn)
end


