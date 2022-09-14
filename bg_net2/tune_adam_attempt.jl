using LinearAlgebra
using Plots
using ProgressMeter
using Measures

include("experiment.jl")
net = BgNet(200, 2, 2.5e-5, 2.5e-5; SynapseType=AdamSynapse)
input = create_input(size(net[:thal]), 200)
target = 0.5 .+ 0.15*gaussianProcessTarget(200, 2, 20)
input_fcn(t) = input[t, :]
target_fcn(t) = target[t, :]
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
losses = [sum((log.snr - target').^2)]
n_trials = 1000
p = Progress(n_trials)
ProgressMeter.ijulia_behavior(:clear)
anim = Animation()
for ep=1:n_trials
    next!(p, showvalues=[(:loss, losses[end]), (:trial, ep)])
    log = recordSampleRun(net, 200, target_fcn, clamp=(thal=input_fcn,))
    push!(losses, sum((log.snr - target').^2))
    #, size=(900, 900))    
    
    #ls_plt = plot(losses, xlabel="Trial", ylabel="Squared error", yaxis=:log,
    #              ylim=(0.1, 50.0), yticks=[0.1,1.0,10.0], xlim=(1,1000), legend=false)
    #l = @layout [a{0.7w} b]
    #both_plt = plot(tr_plt, ls_plt, layout=l, size=(1200, 800),minorticks=true,
    #                minorgrid=true, gridalpha=.25, minorgridalpha=.125)
    #if ep%10==1
        tr_plt = plotTraces(log, target=target_fcn)
        plot!(tr_plt, xlabel="Time (ms)", title="Trial $ep", ylim=(-1.2*5, 1.2), size=(500,700), left_margin=10mm)
        frame(anim, tr_plt)
    #end
end
webm(anim)#, "anim_swebags.webm")
