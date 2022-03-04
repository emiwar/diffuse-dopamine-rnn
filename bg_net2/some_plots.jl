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

losses = readAsDataFrame("data/with_noise.h5", false)
for lr=unique(losses.learning_rate), ff=unique(losses.feedback_factor)
    filtered = losses[(losses.learning_rate .== lr) .& (losses.feedback_factor .== ff), :]
    pl = plotSeries(filtered, :trial, :loss, :striatumUpdate, yaxis=:log, xlabel="Trial", ylabel="Squared error", minorticks=true, minorgrid=true, gridalpha=.25, minorgridalpha=.125, title="Lr: ($lr, $(lr*ff))")
    display(pl)
end


losses = readAsDataFrame("data/with_noise.h5")
for ff=unique(losses.feedback_factor)
    filtered = losses[losses.feedback_factor .== ff, :]
    pl = plotSeries(filtered, :learning_rate, :loss, :striatumUpdate, axis=:log, xlabel="Trial", ylabel="Squared error", minorticks=true, minorgrid=true, gridalpha=.25, minorgridalpha=.125, title="Feedback: $ff")
    display(pl)
end


losses = readAsDataFrame("data/test_flat_dopamine3.h5", false)
for target_dim = [1,2,5,10]
    subset = losses[losses.target_dim .== target_dim, :]
    if size(subset, 1) > 0
        pl = plotSeries(subset, :trial, :loss, :striatumUpdate, yaxis=:log, title="Target dim $target_dim")
        display(pl)
    end
end