using LinearAlgebra
using Plots
using ProgressMeter
include("experiment.jl")

net = BgNet(200, 2, 1e-3)
input = create_input(size(net[:thal]), 200)
target = 0.5 .+ 0.15*gaussianProcessTarget(200, 2, 10)
input_fcn(t) = input[t, :]
target_fcn(t) = target[t, :]
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
plotTraces(log, target=target_fcn)


T_train = 20000
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


net = BgNet(200, 2, 1e-2)
recordSampleRun(net, 200, clamp=(thal=input_fcn,))
p = Progress(200)
@gif for ep=1:200
    next!(p)
    log = recordSampleRun(net, 200, target_fcn, clamp=(thal=input_fcn,), updateStriatum=:ideal)
    plotTraces(log, target=target_fcn)
end

