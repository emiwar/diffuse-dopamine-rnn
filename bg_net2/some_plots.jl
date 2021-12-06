include("plot_experiment.jl")

losses = readAsDataFrame("data/compare_lambdas.h5")
plotSeries(losses, :lambda, :loss, :striatumUpdate, xaxis=:log)


losses = readAsDataFrame("data/compare_lambdas_again.h5")
plotSeries(losses, :lambda, :loss, :striatumUpdate, xaxis=:log)

files =  "data/target_timescale" .* [".h5", "_no_dopamine.h5"]
losses = vcat(readAsDataFrame.(files)...)
plotSeries(losses, :target_tau, :loss, :striatumUpdate, yaxis=:log)

losses = readAsDataFrame("data/many_trials.h5", false)
plotSeries(losses, :trial, :loss, :striatumUpdate, yaxis=:log, xaxis=:log)

losses = readAsDataFrame("data/target_dim.h5")
plotSeries(losses, :target_dim, :loss, :striatumUpdate)#, yaxis=:log)

losses = readAsDataFrame("data/learning_rate.h5")
plotSeries(losses, :learning_rate, :loss, :striatumUpdate, xaxis=:log, yaxis=:log)


losses = readAsDataFrame("data/feedback_scale_different_eta.h5")
for lr=learning_rates
    pl = plotSeries(losses[losses.learning_rate .== lr, :], :feedback_factor, :loss, :striatumUpdate, title="Learning rate $lr", yaxis=:log)
    display(pl)
end

for ff=unique(losses.feedback_factor)
    pl = plotSeries(losses[losses.feedback_factor .== ff, :], :learning_rate, :loss, :striatumUpdate, title="Feedback factor $ff", yaxis=:log)
    display(pl)
end
#Lr=0.005, dopamine_scale=10, ideal_scale=100

losses = readAsDataFrame("different_lambdas_correct_for_feedback_scale.h5")
plotSeries(losses, :lambda, :loss, :striatumUpdate, axis=:log)

losses = readAsDataFrame("data/different_lambda_corrected_feedback.h5")
plotSeries(losses, :lambda, :loss, :striatumUpdate, axis=:log)

losses = readAsDataFrame("data/different_lambda_only_dopamine.h5")
plotSeries(losses, :lambda, :loss, :striatumUpdate, axis=:log)

losses = readAsDataFrame("data/long_run3.h5", false)
plotSeries(losses, :trial, :loss, :striatumUpdate, axis=:log, xlabel="Trial", ylabel="Error (MSE)", minorticks=true, minorgrid=true, gridalpha=.25, minorgridalpha=.125, dpi=300, xticks=10 .^ (0:5))