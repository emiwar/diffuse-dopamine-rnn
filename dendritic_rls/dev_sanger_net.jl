using Plots
using MultivariateStats
using Statistics
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

@gif for i=1:200
    y, h = episode!(net, y_target)
    pca = fit(PCA, h', maxoutdim=4, mean=0)
    pca_target = transform(pca, h')'
    for i=1:size(pca_target, 2)
        pca_target[:,i] .*= sign(pca_target[:, i]'*y[:,i])
    end

    l = @layout [a;b;c{0.5h}]
    p = plot(y_target, label="input", ylim=(-1.5, 1.5), layout=l)
    #plot!(p[1], y_target, label="y_target")
    plot!(p[2], h, legend=false)
    plot!(p[3], y, label=["S1" "S2" "S3" "S4"], ylim=(-6, 6))
    plot!(p[3], pca_target, label=["PC1" "PC2" "PC3" "PC4"], color=[1 2 3 4], linestyle=:dash)
end