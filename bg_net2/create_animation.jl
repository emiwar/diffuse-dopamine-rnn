using LinearAlgebra
using Plots
using ProgressMeter
include("bg_net_v2.jl")

T = 200
base_period = 200
target_fcn(t) = 0.25*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                 0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)] .+ .5
proj = randn(20, 2)
proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
input_fcn(t) = phi.(proj*[cos(2*pi*t/base_period), sin(2*pi*t/base_period)])

net = BgNet(500, 2, 2e-2)
recordSampleRun(net, T, clamp=(thal=input_fcn,))
p = Progress(1000)
anim = @animate for ep=1:1000
    next!(p)
    plotTraces(recordSampleRun(net, T, target_fcn, clamp=(thal=input_fcn,)), target=target_fcn)
    plot!(xlabel="Time (ms)", title="Trial $ep", size=(900, 900))
end
webm(anim, "test_anim3.webm")