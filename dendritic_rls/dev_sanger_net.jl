using Plots
using MultivariateStats
using Statistics
using StatsPlots
include("reference_force.jl")

function episode!(net::SANGER_NET, target::Array)
    reset_state!(net)
    t_max = size(target)[1]
    middle_out = zeros(t_max, size(net.W_middle)[1])
    r_out = zeros(t_max, length(net.v))
    for t=1:t_max
        middle_out[t, :] = step!(net, target[t, :])
        r_out[t, :] = r(net)
    end
    return middle_out, r_out
end

t_max = 100
theta = range(0,2pi;length=t_max)
y_target = @. sin(theta)+0.5sin(2theta)+0.25sin(4theta)
net = SANGER_NET(1, 50, 4, 1; g_rec=0.5, g_in=0.5, clamp_target=true)
explained_var = []
@gif for i=1:400
    y, h = episode!(net, y_target)
    pca = fit(PCA, h', maxoutdim=4, mean=0)
    pca_target = transform(pca, h')'
    for i=1:size(pca_target, 2)
        pca_target[:,i] .*= sign(pca_target[:, i]'*y[:,i])
    end

    l = @layout [[a b; c d; e] f]
    outer_p = heatmap(y'*y/t_max, yflip=true, clim=(-5, 5), color=:balance,
                      colorbar=false)
    corrs = heatmap(cor(y), yflip=true, clim=(-1, 1), color=:balance,
                    colorbar=false)
    y_sub = (sum(y, dims=2) .- y)
    means = groupedbar([sum(y, dims=1); sum(y_sub, dims=1); sum(y_sub.*y, dims=1)]'/t_max,
                label=["y" "y_sub" "y*y_sub"], legend=false, ylim=(-10, 10))
    mean_sqrs = groupedbar([sum(y.^2, dims=1); sum(y_sub.^2, dims=1); sum((y_sub.*y), dims=1)]'/t_max,
                label=["y" "y_sub" "y*y_sub"], legend=false, ylim=(-30, 30))
    lines = plot(y, label=["S1" "S2" "S3" "S4"], ylim=(-10, 10), flip=false)
    plot!(lines, pca_target, color=[1 2 3 4], linestyle=:dash)
    y_ex = [y ones(t_max)]
    push!(explained_var, 1-sum((h-y_ex*(y_ex\h)).^2)/sum(h.^2))
    ev = plot(explained_var, xlim=(1, 300), ylim=(0,1))
    plot(outer_p, corrs, means, mean_sqrs, ev, lines, layout=l)
    
    #plot!(p[2], pca_target, label=["PC1" "PC2" "PC3" "PC4"], color=[1 2 3 4], linestyle=:dash)
end