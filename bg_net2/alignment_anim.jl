include("experiment.jl")
using Plots

net = BgNet(200, 4, 1e-3, 2.5e-2)
input = create_input(size(net[:thal]), 200)
target = 0.5 .+ 0.15*gaussianProcessTarget(200, 4, 20)
input_fcn(t) = input[t, :]
target_fcn(t) = target[t, :]
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
#save_log("data/exampleRun.h5", "warmup", log)
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
p = Progress(5000)
ProgressMeter.ijulia_behavior(:clear)
anim = Animation()
angles = []
for ep=1:5000
    push!(angles, getAlignmentAngle(net))
    next!(p, showvalues=[(:angle, angles[end]), (:trial, ep)])
    log = recordSampleRun(net, 200, target_fcn, clamp=(thal=input_fcn,))
    #, size=(900, 900))    
    
    #ls_plt = plot(losses, xlabel="Trial", ylabel="Squared error", yaxis=:log,
    #              ylim=(0.1, 50.0), yticks=[0.1,1.0,10.0], xlim=(1,1000), legend=false)
    #l = @layout [a{0.7w} b]
    #both_plt = plot(tr_plt, ls_plt, layout=l, size=(1200, 800),minorticks=true,
    #                minorgrid=true, gridalpha=.25, minorgridalpha=.125)
    if ep%5==1
        fb = vcat(-net.feedback_dmsn[:], net.feedback_imsn[:])
        wm_dmsn = getWeightMatrix(net[:str_dmsn], net[:snr])'[:]
        wm_imsn = getWeightMatrix(net[:str_imsn], net[:snr])'[:]
        wm = vcat(wm_dmsn, wm_imsn)
        sc = scatter(fb, wm, xlabel="Effective nigrostriatal weight",
                     ylabel="Striatonigral weight", legend=false, title="Trial $ep",
                     ylim=(-0.5, 0.5))
        frame(anim, sc)
    end
end
webm(anim, "anim_alignment.webm")