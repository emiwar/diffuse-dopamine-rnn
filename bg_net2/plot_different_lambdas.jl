using Plots
using HDF5
using Statistics
using LatexStrings

lambdas = 10 .^ (-3:0.25:1.0)
losses = Dict{Float64, Matrix{Float64}}()
final_loss = zeros(length(lambdas), 10)
h5open("different_lambdas_v3.h5", "r") do fid
    for (i, lambda) in enumerate(lambdas)
        losses[lambda] = read(fid["lam$(round(lambda, digits=6))"])
        final_loss[i, :] = losses[lambda][end, :]
    end
end

m = mean(final_loss, dims=2)
s = std(final_loss, dims=2)


plot(lambdas, m, ribbon=s, fillalpha=0.2, xaxis=:log, ylim=(0, .8), legend=false,
     minorticks=true, minorgrid=true, gridalpha=.25, minorgridalpha=.125, xticks=10.0 .^(-3:1),
     xlabel="Dopamine spatial scale (a.u.)", ylabel="Error (MSE)", size=(400, 200), dpi=300)
savefig("different_lambdas_log_scale_v3.png")


plots = [plot(0:0.001:1, x->exp(-abs(x-0.5)/(10.0^lam)),
         color=:green, title=L"\lambda=10^{%$lam}") for lam=-3:1]
plot(plots..., layout=(1, 5), legend=false, size=(400, 85),
     ylim=(0, 1.2), xlim=(0, 1), ticks=false, framestyle = :box,
     titlefontsize=10, fillrange=0, fillalpha=0.05, dpi=300)
savefig("lambda_illustrations.png")
