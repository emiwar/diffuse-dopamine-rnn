using Plots
using Statistics

include("bg_net_v2.jl")
lambdas = 10 .^ (-3:0.125:2)
res = zeros(length(lambdas), 50)
for i=1:length(lambdas), j=1:50
    lambda = lambdas[i]
    net = BgNet(200, 4, 1e-3, 2.5e-2; lambda=lambda)
    fb_mat = vcat(-net.feedback_dmsn, net.feedback_imsn)
    res[i, j] = eigvals(fb_mat' * fb_mat)[1]
    #norm(fb_mat)
    #norm(triu(fb_mat * fb_mat', 1))
    #eigvals(fb_mat' * fb_mat)[1]
    #push!(norms, norm(triu(fb_mat * fb_mat', 1)))
    #push!(evs, )
end
m = median(res, dims=2)[:,1]
low_q = [quantile(res[i,:], 0.25) for i=1:size(res,1)][:,1]
high_q = [quantile(res[i,:], 0.75) for i=1:size(res,1)][:,1]
plot(lambdas, m, ribbon=(m-low_q, high_q-m),
     xaxis=:log, legend=false, xticks=10.0 .^ (-3:2))


n_vars = Int64.(round.(10 .^ (0:0.2:3)))
res2 = zeros(length(lambdas), length(n_vars))
for i=1:length(lambdas), j=1:length(n_vars)
    lambda = lambdas[i]
    n_var = n_vars[j]
    net = BgNet(200, 4, 1e-3, 2.5e-2; lambda=lambda, n_varicosities=n_var)
    fb_mat = vcat(-net.feedback_dmsn, net.feedback_imsn)
    res2[i, j] = eigvals(fb_mat' * fb_mat)[1]
    #norm(fb_mat)
    #norm(triu(fb_mat * fb_mat', 1))
    #eigvals(fb_mat' * fb_mat)[1]
    #push!(norms, norm(triu(fb_mat * fb_mat', 1)))
    #push!(evs, )
end

heatmap(lambdas, n_vars, log10.(res2' .+ 1e-10), axis=:log)