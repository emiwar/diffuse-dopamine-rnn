include("plot_experiment.jl")

fontsize=7

files =  "data/different_lambda_only_dopamine" .* [".h5", "_v2.h5"]
losses = vcat(readAsDataFrame.(files)...)
losses[:,:striatumUpdate] .= "Dopamine"
plotSeries(losses, :lambda, :loss, :striatumUpdate, axis=:log, minorticks=true, minorgrid=true, gridalpha=.25, minorgridalpha=.125, xticks=10.0 .^(-3:2), ylim=(0.008, 0.3), yticks=[0.01, 0.1], xlabel="Dopamine spatial\nscale (𝜆)", ylabel="Error (MSE)", size=(250, 150), dpi=300, labelfontsize=fontsize, legendfontsize=fontsize, tickfontsize=fontsize)
losses = readAsDataFrame("data/different_lambda_corrected_feedback.h5")
lo, me, hi = quantile(losses[losses[:, :striatumUpdate] .== "no_dopamine", :loss][1:100], [0.25, 0.5, 0.75])
plot!([0.001, 100.0], [me, me], ribbon=([me-lo, me-lo], [hi-me, hi-me]), label="Striatofugal only")
lo, me, hi = quantile(losses[losses[:, :striatumUpdate] .== "ideal", :loss][1:100], [0.25, 0.5, 0.75])
plot!([0.001, 100.0], [me, me], ribbon=([me-lo, me-lo], [hi-me, hi-me]), label="Non-local feedback")
savefig("vary_lambda.svg")


files =  "data/different_lambda_only_dopamine" .* [".h5", "_v2.h5"]
losses = vcat(readAsDataFrame.(files, false)...)
losses[:,:striatumUpdate] .= "Dopamine"
gpy = groupby(losses[losses[:, :lambda] .== 0.1, :], :trial)
combined = combine(gpy, :loss=>median=>:median,
                   :loss=>(q->quantile(q, 0.25))=>:low_q,
                   :loss=>(q->quantile(q, 0.75))=>:high_q)
@df combined plot(:trial, :median, ribbon=(:median-:low_q, :high_q-:median), yaxis=:log, xlabel="Trial", ylabel="Error (MSE)", ylim=(0.01, 10.0), minorticks=true, minorgrid=true, gridalpha=.25, minorgridalpha=.125, size=(250, 150), dpi=300, labelfontsize=fontsize, legendfontsize=fontsize, tickfontsize=fontsize, label="Dopamine")

losses = readAsDataFrame("data/different_lambda_corrected_feedback.h5", false)
gpy = groupby(losses[losses[:, :striatumUpdate] .== "no_dopamine", :], :trial)
combined = combine(gpy, :loss=>median=>:median,
                   :loss=>(q->quantile(q, 0.25))=>:low_q,
                   :loss=>(q->quantile(q, 0.75))=>:high_q)
@df combined plot!(:trial, :median, ribbon=(:median-:low_q, :high_q-:median), yaxis=:log, label="Striatofugal only")
gpy = groupby(losses[losses[:, :striatumUpdate] .== "ideal", :], :trial)
combined = combine(gpy, :loss=>median=>:median,
                   :loss=>(q->quantile(q, 0.25))=>:low_q,
                   :loss=>(q->quantile(q, 0.75))=>:high_q)
@df combined plot!(:trial, :median, ribbon=(:median-:low_q, :high_q-:median), yaxis=:log, label="Non-local feedback")
savefig("learning_convergence.svg")

include("experiment.jl")
net = BgNet(200, 2, 1e-2, 1e-1)
input = create_input(size(net[:thal]), 200)
target = 0.5 .+ 0.15*gaussianProcessTarget(200, 2, 20)
input_fcn(t) = input[t, :]
target_fcn(t) = target[t, :]
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
#plotTraces(log, target=target_fcn)

p1 = plot(log.snr', title="1st trial")
plot!(p1, target, color=[1 2], linestyle=:dash)

losses = Float64[]
@showprogress for trial_id=1:1000
    loss = run_trial(net, target, input, :dopamine)
    push!(losses, loss)
end

log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
p2 = plot(log.snr', title="1000th trial", label=["SNr1" "SNr2"])
plot!(p2, target, color=[1 2], linestyle=:dash, label=["Target1" "Target2"])
plot(p1, p2, size=(250, 100), tickfontsize=fontsize, dpi=300, legend=true, ylim=(0.0,1.0), yticks=[0,.5,1], xticks=[0, 100, 200], xlabel="Time (ms)", labelfontsize=fontsize, titlefontsize=fontsize, xlim=(0,200), minorticks=true, minorgrid=true, gridalpha=.25, minorgridalpha=.125)
savefig("learning_example.svg")