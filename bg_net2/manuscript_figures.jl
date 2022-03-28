include("plot_experiment.jl")

fontsize=5*2

losses = readAsDataFrame("data/flat_dopamine_dim_3_and_4.h5", false)
losses = losses[losses.target_dim .== 3, :]
labels = Dict("dopamine" => "Heterogenous dopamine",
              "flat_dopamine" => "Homogenous dopamine",
              "no_dopamine" => "No corticostrital plasticity")
p1 = plotSeries(losses, :trial, :loss, :striatumUpdate, yaxis=:log,
           xlabel="Trial", ylabel="Squared error",
           labels=labels, minorticks=true, minorgrid=true, gridalpha=.4,
           minorgridalpha=.2, ylim=(1e-2, 1e2), fg_legend=nothing, left_margin=10mm)

final_losses = losses[losses.trial .== 5000, [:striatumUpdate, :loss]]
colors = [RGBA(c, 0.6) for c in get_color_palette(:auto, :white)[1:3]]
p2 = boxplot(final_losses.striatumUpdate, final_losses.loss,
             group=final_losses.striatumUpdate, yaxis=:log, ylim=(1e-2, 1e2),
             xticks=[], label=nothing, minorticks=true, minorgrid=true,
             gridalpha=.4, minorgridalpha=.2, showaxis=:x, formatter=(y)->"",
             color=reshape(colors, 1, 3),
             linecolor=reshape(get_color_palette(:auto, :white)[1:3], 1, 3),
             whisker_range=10.0)

losses = readAsDataFrame("data/vary_lambda_dim4.h5", true)
plotSeries(losses, :lambda, :loss, :striatumUpdate, axis=:log, labels=labels)
losses = readAsDataFrame("data/vary_lambda_dim4_flat_dopamine.h5", true)
plotSeries!(losses, :lambda, :loss, :striatumUpdate, axis=:log, labels=labels)
lambda_plot = plot!(xlabel="Dopamine spatial constant", ylabel="Squared error",
      minorticks=true, minorgrid=true, gridalpha=.4,
      minorgridalpha=.2,
      labelfontsize=fontsize, legendfontsize=fontsize, 
      tickfontsize=fontsize, titlefontsize=fontsize, dpi=200,
      xticks=10.0 .^ (-3:2), size=(400, 250))
savefig(lambda_plot, "manuscript_figs/lambda.svg")

labels = ["Thalamus", "Cortex\n(exc.)", "Cortex\n(inh.)", "Striatum\n(dSPNs)",
              "Striatum\n(iSPNs)", "SNr", "SNc"]
function plotHeatmaps(log::NamedTuple; sort::Integer=0, kwargs...)
    y = 1
    yticks = Float64[]
    p = plot()
    for pop in reverse(keys(log))
        pop_act = log[pop]
        dy, T = size(pop_act)
        if sort != 0
            order = sortperm(sort*getindex.(argmax(log[pop], dims=2)[:, 1], 2))
            pop_act = pop_act[order, :]
        end
        heatmap!(p, 1:T, y:(y+dy-1), pop_act, clim=(0, 1); kwargs...)
        push!(yticks, y+0.5dy)
        y += dy+5
    end
    plot!(p, yticks=(yticks, reverse(labels)), ylim=(-5, y))#reverse(keys(log))))
end

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
difference = NamedTuple(pop => last_log[pop] - first_log[pop] for pop in keys(first_log))


hmBefore = plotHeatmaps(first_log)
hmAfter = plotHeatmaps(last_log)
hmDiff = plotHeatmaps(difference, c=:balance, clim=(-1, 1))
plot!(hmBefore, colorbar=false, xlim=(0,200), xlabel="Time (ms)", title="Trial #1", 
      xticks=0:100:200)
plot!(hmAfter, colorbar=false, xlim=(0,200), xlabel="Time (ms)", yticks=[],
      title="Trial #5000", xticks=0:100:200)
plot!(hmDiff, colorbar=false, xlim=(0,200), xlabel="Time (ms)", yticks=[],
      title="Difference", xticks=0:100:200)

first_output = plot(first_log[:snr]', color=[1 2 3 4], legend=false,
                    label=["SNr1" "SNr2" "SNr3" "SNr4"], xticks=[0,100,200])
plot!(first_output, target, color=[1 2 3 4], linestyle=:dash, legend=true,
      title="Trial #1", ylim=(0, 1), xformatter=x->"", yticks=[0,.5,1],
      label=["Target1" "Target2" "Target3" "Target4"], xticks=[0,100,200], xlim=(0,200))
last_output = plot(last_log[:snr]', color=[1 2 3 4], legend=false, title="Trial #5000")
plot!(last_output, target, color=[1 2 3 4], linestyle=:dash, legend=false,
      title="Trial #5000", ylim=(0, 1), yticks=[0,.5,1],
      xticks=[0,100,200], xlim=(0,200), xformatter=x->"", yformatter=y->"")

first_dopamine = plot(first_log[:snc]', color=[1 2 3 4], title="Trial #1", ylim=(-.5, .5), legend=false, yticks=[-.5,0,.5], xlim=(0,200), xticks=[0,100,200], xlabel="Time (ms)")
last_dopamine = plot(last_log[:snc]', color=[1 2 3 4], title="Trial #5000", xlabel="Time (ms)", ylim=(-.5, .5), legend=false, yticks=[-.5,0,.5], xticks=[0,100,200], xlim=(0,200), yformatter=y->"")

l = @layout [a{0.4w} b{0.03w} c{0.1w} d{0.1w} e{0.1w} [f{0.35h, 0.8w}; g{0.35h}] [h{0.35h, 0.8w}; i{0.35h}]]
dpi = 200
combined = plot(p1, p2, hmBefore, hmAfter, hmDiff, first_output, first_dopamine, last_output, last_dopamine,
    layout=l, format="png", dpi=dpi, size=(7.5*dpi, 1.8*dpi), labelfontsize=fontsize,
    legendfontsize=fontsize, tickfontsize=fontsize, titlefontsize=fontsize, bottom_margin=10mm)
combined
savefig(combined, "manuscript_figs/example_run.svg")


dopamine_falloff = plot(d->exp(-d/0.1), xlim=(0, 1), legend=false, xlabel="Distance (cube sides)", ylabel="Relative\ndopamine", size=(250, 150), dpi=200, xticks=[0, 0.5, 1.0], yticks=[0, 0.5, 1.0], labelfontsize=fontsize, legendfontsize=fontsize, tickfontsize=fontsize, linewidth=2, color=:darkgreen, minorgrid=true)
savefig(dopamine_falloff, "manuscript_figs/dopamine_falloff.svg")

transfer_fcn = plot(V->phi(V), xlim=(-8, 8), legend=false, xlabel="V(t)", ylabel="r(t)", size=(250, 150), dpi=200, xticks=-8:4:8, yticks=[0, 0.5, 1.0], labelfontsize=fontsize, legendfontsize=fontsize, tickfontsize=fontsize, linewidth=2, minorgrid=true)
savefig(transfer_fcn, "manuscript_figs/transfer_fnc.svg")