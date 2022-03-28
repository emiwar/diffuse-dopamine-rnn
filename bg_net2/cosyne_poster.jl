include("plot_experiment.jl")

fontsize=5*2

colors = [colorant"#269d26" colorant"#1a54a6" colorant"#820655"]

losses = readAsDataFrame("data/flat_dopamine_dim_3_and_4.h5", false)
losses = losses[losses.target_dim .== 4, :]
labels = Dict("dopamine" => "Heterogenous dopamine",
              "flat_dopamine" => "Homogenous dopamine",
              "no_dopamine" => "No corticostrital plasticity")
colors_d = Dict("dopamine" => colors[1],
              "flat_dopamine" => colors[2],
              "no_dopamine" => colors[3])
p1 = plotSeries(losses, :trial, :loss, :striatumUpdate, yaxis=:log,
           xlabel="Trial", ylabel="Squared error",
           labels=labels, minorticks=true, minorgrid=true, gridalpha=.4,
           minorgridalpha=.2, ylim=(1e-2, 1e2), fg_legend=nothing, left_margin=10mm, colors=colors_d)

final_losses = losses[losses.trial .== 5000, [:striatumUpdate, :loss]]
#colors = [RGBA(c, 0.6) for c in get_color_palette(:auto, :white)[1:3]]
p2 = boxplot(final_losses.striatumUpdate, final_losses.loss,
             group=final_losses.striatumUpdate, yaxis=:log, ylim=(1e-2, 1e2),
             xticks=[], label=nothing, minorticks=true, minorgrid=true,
             gridalpha=.4, minorgridalpha=.2, showaxis=:x, formatter=(y)->"",
             color=colors,
             linecolor=colors,
             whisker_range=10.0)
l = @layout [a{0.94w} b{0.06w}]
dpi = 200
combined = plot(p1, p2,
    layout=l, format="png", dpi=dpi, size=(450, 225), labelfontsize=fontsize,
    legendfontsize=fontsize, tickfontsize=fontsize, titlefontsize=fontsize, bottom_margin=10mm)
savefig(combined, "poster_figs/convergence.svg")



losses = readAsDataFrame("data/vary_lambda_dim4.h5", true)
plotSeries(losses, :lambda, :loss, :striatumUpdate, axis=:log, labels=labels, colors=colors_d)
losses = readAsDataFrame("data/vary_lambda_dim4_flat_dopamine.h5", true)
plotSeries!(losses, :lambda, :loss, :striatumUpdate, axis=:log, labels=labels, colors=colors_d)
lambda_plot = plot!(xlabel="Dopamine spatial constant", ylabel="Squared error",
      minorticks=true, minorgrid=true, gridalpha=.4,
      minorgridalpha=.2,
      labelfontsize=fontsize, legendfontsize=fontsize, 
      tickfontsize=fontsize, titlefontsize=fontsize, dpi=200,
      xticks=10.0 .^ (-3:2), size=(400, 225), bottom_margin=10mm)

savefig(lambda_plot, "poster_figs/lambda.svg")





function read_log(fn, hdfKey)
    order = (:thal, :ctx_exc, :ctx_inh, :str_dmsn, :str_imsn, :snr)
    h5open(fn, "r") do fid
        grp = fid[hdfKey]
        NamedTuple(k=>read(grp[string(k)]) for k=order)
    end
end
function add_snc(log, target)
    merge(log, NamedTuple((:snc=>target'-log[:snr],)))
end
target = h5open(fid->read(fid["target"]), "data/exampleRun.h5", "r")
first_log = add_snc(read_log("data/exampleRun.h5", "first_trial"), target)
last_log = add_snc(read_log("data/exampleRun.h5", "last_trial"), target)


first_output = plot(first_log[:snr]', color=[1 2 3 4], legend=false,
                    label=["SNr1" "SNr2" "SNr3" "SNr4"], xticks=[0,100,200])
plot!(first_output, target, color=[1 2 3 4], linestyle=:dash, legend=false,
      title="Trial #1", ylim=(0, 1), yticks=[0,.5,1],
      label=["Target1" "Target2" "Target3" "Target4"], xticks=[0,100,200], xlim=(0,200), minorticks=true, minorgrid=true, gridalpha=.4,
      minorgridalpha=.2, xlabel="Time (ms)", size=(160, 140), dpi=200,
labelfontsize=fontsize, legendfontsize=fontsize, 
      tickfontsize=fontsize, titlefontsize=fontsize)
savefig(first_output, "poster_figs/first_output.svg")
last_output = plot(last_log[:snr]', color=[1 2 3 4], legend=false,
                    label=["SNr1" "SNr2" "SNr3" "SNr4"], xticks=[0,100,200])
plot!(last_output, target, color=[1 2 3 4], linestyle=:dash, legend=false,
      title="Trial #5000", ylim=(0, 1), yticks=[0,.5,1],
      label=["Target1" "Target2" "Target3" "Target4"], xticks=[0,100,200], xlim=(0,200), minorticks=true, minorgrid=true, gridalpha=.4,
      minorgridalpha=.2, xlabel="Time (ms)", size=(160, 140), dpi=200,
labelfontsize=fontsize, legendfontsize=fontsize, 
      tickfontsize=fontsize, titlefontsize=fontsize)
savefig(last_output, "poster_figs/last_output.svg")



first_dopamine = plot(first_log[:snc]', legend=false,
                xticks=[0,100,200], color=[1 2 3 4], ylim=(-0.5, 0.5), yticks=[-.5,0,.5], xlim=(0,200), minorticks=true, minorgrid=true, gridalpha=.4,
      minorgridalpha=.2, xlabel="Time (ms)", size=(200, 125), dpi=200,
labelfontsize=fontsize, legendfontsize=fontsize, 
      tickfontsize=fontsize, titlefontsize=fontsize)
savefig(first_dopamine, "poster_figs/first_dopamine.svg")