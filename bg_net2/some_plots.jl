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

