using LinearAlgebra
using Plots
using ProgressMeter
using Distributions
include("bg_net_v2.jl")

net = BgNet(200, 2, 1e-2)

function createSignals(duration, ndim, tau; eps=1e-6)
    dists = [min(abs(i-j), abs(duration+j-i), abs(-duration+j-i)) 
             for i=1:duration, j=1:duration]
    cov = exp.(-(dists.^2)./(tau^2))
    return rand(MvNormal(zeros(duration), cov+eps*I), ndim)
end   
    
T = 200
base_period = 200
target = 0.5 .+ 0.15*createSignals(T, 2, 10)
target_fcn(t) = target[t, :]
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])

net = BgNet(200, 2, 2e-3)
recordSampleRun(net, T, clamp=(thal=input_fcn,))
p = Progress(100)
@gif for ep=1:100
    next!(p)
    plotTraces(recordSampleRun(net, T, target_fcn, clamp=(thal=input_fcn,)), target=target_fcn)
end



plotTraces(recordSampleRun(net, T, clamp=(thal=input_fcn,)), target=target_fcn)

net = BgNet(200, 2, 5e-3)
T_train = 200000
losses = Float64[]
loss = 0.0
@showprogress for t=1:T_train
    loss += step!(net, target_fcn((t-1)%T+1), clamp=(thal=input_fcn(t),), updateStriatum=:dopamine)
    #loss += sum((net[:snr].r .- target_fcn(t)).^2)
    #net[:snr].r = target_fcn(t)
    if mod(t, base_period) == 0
        push!(losses, loss)
        loss = 0.0
    end
end
plot(losses[1:end])#, ylim=(0, 150))

n_runs=3
for target_tau=8:2:20
    for r=1:n_runs
        net = BgNet(200, 2, 2e-3)
        target = 0.5 .+ 0.15*createSignals(T, 2, target_tau)
        loss = 0.0
        @showprogress "tau=$(key), run $r: " for t=1:T_train
            loss += step!(net, target_fcn(t), clamp=(thal=input_fcn(t),), updateStriatum=key)
            if mod(t, base_period) == 0
                losses[key][div(t, base_period), r] = loss
                loss = 0.0
            end
        end
    end
end

target = 0.5 .+ 0.15*createSignals(T, 2, 10)
target_fcn(t) = target[t, :]
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])

net = BgNet(200, 2, 2e-3)
recordSampleRun(net, T, clamp=(thal=input_fcn,))