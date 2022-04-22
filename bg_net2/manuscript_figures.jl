include("plot_experiment.jl")

fontsize=5*2

losses = readAsDataFrame("data/flat_dopamine_dim_3_and_4.h5", false)
losses = losses[losses.target_dim .== 4, :]
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

transfer_fcn = plot(V->phi(V), xlim=(-8, 8), legend=false, xlabel="V(t)", ylabel="r(t)",
                    size=(200, 120), dpi=200, xticks=-8:4:8, yticks=[0, 0.5, 1.0],
                    labelfontsize=fontsize, legendfontsize=fontsize, tickfontsize=fontsize,
                    linewidth=2, gridalpha=.4,
                    minorticks=true, minorgrid=true, minorgridalpha=.2)
savefig(transfer_fcn, "manuscript_figs/transfer_fnc.svg")

conns = [("Thal → Ctx_e", 0.2),
        ("Thal → Ctx_i", 0.2),
        ("Ctx_e → Ctx_e", 0.1),
        ("Ctx_e → Ctx_i", 0.1),
        ("Ctx_i → Ctx_e", 0.4),
        ("Ctx_i → Ctx_e", 0.4),
        ("Ctx_e → Str_d1", 0.2),
        ("Ctx_e → Str_d2", 0.2),
        ("Thal → Str_d1", 0.25),
        ("Thal → Str_d2", 0.25),
        ("Str_d1 → Str_d1", 0.2),
        ("Str_d1 → Str_d2", 0.2),
        ("Str_d2 → Str_d1", 0.2),
        ("Str_d2 → Str_d2", 0.2),
        ("Str_d1 → SNr", 1.0),
        ("Str_d2 → SNr", 1.0)]
#yticks = [(i, conns[i][1]) for i=1:length(conns)]
conn_bars = bar(1:16, getindex.(conns, 2), orientation=:h,
    yticks=(1:length(conns), getindex.(conns, 1)),
    xticks=[0.0,0.5,1.0], ylim=(0.3,16.7), xlim=(0,1),
    yflip=true, xlabel="P(connection)", legend=false,
    gridalpha=.4, #minorticks=:x, minorgrid=:x, minorgridalpha=.2,
    labelfontsize=fontsize, tickfontsize=fontsize, legendfontsize=fontsize,
    size=(1.3*dpi, 1.4*dpi), dpi=dpi, left_margin=10mm)
savefig(conn_bars, "manuscript_figs/conn_bars.svg")

losses = readAsDataFrame("data/constraint_exp_full.h5")
losses[!, :x_coord] .= -1

types = ["plastic", "static", "absent"]

legend = zeros(4, 3^4)
i = 1
for str_snr=1:3, str_str=1:3, thal_str=1:3, ctx_str=1:3
    legend[1, i] = str_snr
    legend[2, i] = str_str
    legend[3, i] = thal_str
    legend[4, i] = ctx_str
    filter = (losses.str_snr .== types[str_snr]) .&
             (losses.str_str .== types[str_str]) .&
             (losses.thal_str .== types[thal_str]) .&
             (losses.ctx_str .== types[ctx_str])
    losses[filter, :x_coord] .= i
    i += 1
end

colors = [:green, :blue, :darkred]
p2 = boxplot(losses.x_coord, losses.loss,
             group=losses.x_coord, yaxis=:log, yticks=10.0 .^ (-2:2),
             xticks=[], label=nothing, minorticks=true, minorgrid=true,
             gridalpha=.4, minorgridalpha=.2, color=:blue,
             ylabel="Squared loss", whisker_range=10.0)
hm = heatmap(legend, colorbar=false, c=colors)
for i=0:(3^4)+1
    plot!(hm, [i, i] .- 0.5, [0.5,4.5], color=:white, legend=false, lw=3)
end
for i=0:4
    plot!(hm, [0.5, 3^4+0.5], [i, i] .+ 0.5, color=:white, legend=false, lw=3)
end
yticks = ["Str → SNr", "Str → Str", "Thal → Str", "Ctx → Str"]
plot!(hm, showaxis=false, xticks=[], yticks=(1:4, yticks))
l =  @layout [a; b{0.25h}]
full_constr = plot(p2, hm, layout=l, xlim=(0, 3^4+1), size=(6.5*200, 1.5*200), dpi=200,
     labelfontsize=fontsize, tickfontsize=fontsize, legendfontsize=fontsize)

savefig(full_constr, "manuscript_figs/full_constraints_25samples.svg")

#heatmap(legend', c=colors, size=(100, 1000), colorbar=false, yticks=1:(3^4), left_margin=10mm)
subselection = [1, 2, 4, 10, 28, 13, 14, 41, 3, 7, 19, 55, 9, 81]
sublosses = losses[[(x in subselection) for x in losses.x_coord], :]
sublosses[!, :x_coord] .= [findfirst(i->(i==x), subselection) for x in sublosses.x_coord]
p3 = boxplot(sublosses.x_coord, sublosses.loss,
             group=sublosses.x_coord, yaxis=:log, yticks=10.0 .^ (-2:2),
             xticks=[], label=nothing, minorticks=true, minorgrid=true,
             gridalpha=.4, minorgridalpha=.2, color=RGBA(0, 0, 0, 0.2),
             ylabel="Squared error", whisker_range=10.0,
             xlim=(0, length(subselection)+1))
hm = heatmap(legend[:, subselection], colorbar=false, c=colors,
             showaxis=false, xticks=[], yticks=(1:4, yticks),
             xlim=(0, length(subselection)+1))
for i=0:length(subselection)+1
    plot!(hm, [i, i] .- 0.5, [0.5,4.5], color=:white, legend=false, lw=3)
end
for i=0:4
    plot!(hm, [0.5, length(subselection)+0.5], [i, i] .+ 0.5, color=:white, legend=false, lw=3)
end

losses = readAsDataFrame("data/random_feedback.h5", false)[end:-1:1, :]
labels = Dict("dopamine" => "Dopamine feedback",
              "random" => "Random feedback",
              "ideal" => "Ideal feedback",
              "no_dopamine" => "No feedback")
colors_d = Dict("dopamine" => colorant"#269d26",
                "random" => colorant"#1a54a6",
                "ideal" => colorant"orange",
                "no_dopamine" => colorant"#820655")
order = ["dopamine", "ideal", "no_dopamine", "random"]
p1 = plotSeries(losses, :trial, :loss, :striatumUpdate, yaxis=:log,
           xlabel="Trial", ylabel="Squared error",
           labels=labels, minorticks=true, minorgrid=true, gridalpha=.4,
           minorgridalpha=.2, ylim=(1e-2, 1e2), fg_legend=nothing, left_margin=10mm,
           colors=colors_d)
final_losses = losses[losses.trial .== 5000, [:striatumUpdate, :loss]]
colors = [RGBA(colors_d[l], 0.6) for l in order]
linecolors = [colors_d[l] for l in order]
p2 = boxplot(final_losses.striatumUpdate, final_losses.loss,
             group=final_losses.striatumUpdate, yaxis=:log, ylim=(1e-2, 1e2),
             xticks=[], label=nothing, minorticks=true, minorgrid=true,
             gridalpha=.4, minorgridalpha=.2, showaxis=:x, formatter=(y)->"",
             color=reshape(colors, 1, 4),
             linecolor=reshape(linecolors, 1, 4),
             whisker_range=10.0)

labels = Dict("dopamine" => "Heterogenous dopamine",
              "flat_dopamine" => "Homogenous dopamine")
colors_d = Dict("dopamine" => colorant"#269d26",
                "flat_dopamine" => colorant"#1a54a6")
losses = readAsDataFrame("data/vary_lambda_dim4.h5", true)
plotSeries(losses, :lambda, :loss, :striatumUpdate, axis=:log, labels=labels, colors=colors_d)
losses = readAsDataFrame("data/vary_lambda_dim4_flat_dopamine.h5", true)
plotSeries!(losses, :lambda, :loss, :striatumUpdate, axis=:log, labels=labels, colors=colors_d)
lambda_plot = plot!(ylabel="Squared error",
      minorticks=true, minorgrid=true, gridalpha=.4,
      minorgridalpha=.2,
      xticks=10.0 .^ (-3:2)) #, size=(400, 250))
#savefig(lambda_plot, "manuscript_figs/lambda.svg")

losses = readAsDataFrame("data/test_n_varicosities.h5", true)
gby = groupby(losses, [:n_varicosities, :lambda]) 
cmb = combine(gby, :loss => median)
ust = unstack(cmb, :n_varicosities, :lambda, :loss_median)
xlabels = [parse(Float64, c) for c in names(ust)[2:end]]
ylabels = ust.n_varicosities
values = Matrix{Float64}(ust[:, Not(:n_varicosities)])
hm2 = heatmap(xlabels, ylabels, -log10.(values), axis=:log,
              caxis=:log, xlabel="Dopamine  ",
              ylabel="Varicosities per SNc cell",
              yticks=10 .^ (0:3), xticks=10.0 .^ (-3:2),
              cticks=[-1,0,1,2], clim=(-1,2), colorbar=false)

first_eig, lambdas = h5open("data/first_eig_per_lambda.h5", "r") do fid
    read(fid["first_eig"]), read(fid["lambdas"])
end
m = median(first_eig, dims=2)[:,1]
low_q = [quantile(first_eig[i,:], 0.25) for i=1:size(first_eig,1)][:,1]
high_q = [quantile(first_eig[i,:], 0.75) for i=1:size(first_eig,1)][:,1]
eig_plot = plot(lambdas, m, ribbon=(m-low_q, high_q-m),
     xaxis=:log, legend=false, xticks=10.0 .^ (-3:2),
     minorticks=true, minorgrid=true, gridalpha=.4,
     minorgridalpha=.2, ylim=(-10, 400), color=colors_d["dopamine"])

l =  @layout [a{0.3w} b{0.03w} [c{0.95h, 0.9w}; d{0.18h, 0.9w}] [e{0.8w}; f] [g{1.0w, 0.8h};]]
fig2 = plot(p1, p2, p3, hm, lambda_plot, eig_plot, hm2, layout=l, size=(7.5*200, 1.8*200), dpi=200,
     labelfontsize=fontsize, tickfontsize=fontsize, legendfontsize=fontsize, bottom_margin=10mm)
savefig(fig2, "manuscript_figs/fig2_draft.svg")


losses = readAsDataFrame("data/lin_ind_corr.h5")
scatter(losses.feedback_eig1, losses.loss)
corspearman(losses.feedback_eig1, losses.loss)


losses = readAsDataFrame("data/adam_dim4.h5", false)
losses = losses[(losses.synapseType .== "AdamSynapse") .&
                (losses.striatumUpdate .== "dopamine"), :]
plotSeries(losses, :trial, :loss, :striatumUpdate, yaxis=:log,
           labels=Dict("dopamine" => "ADAM-RFLO"))
losses = readAsDataFrame("data/flat_dopamine_dim_3_and_4.h5", false)
losses = losses[(losses.target_dim .== 4) .& (losses.striatumUpdate .== "dopamine"), :]
p1 = plotSeries!(losses, :trial, :loss, :striatumUpdate, yaxis=:log,
           xlabel="Trial", ylabel="Squared error",
           labels=Dict("dopamine" => "RFLO"), minorticks=true,
           minorgrid=true, gridalpha=.4, colors=colors_d,
           minorgridalpha=.2, ylim=(1e-2, 1e2), fg_legend=nothing, left_margin=10mm)