using HDF5
using Statistics
using Plots

n_reps = 1000#h5open(fid->read(fid, "reps"), "data/dist_vs_corr.h5")

#fn = "data/dist_vs_corr_v7.h5"
fn = "data/dist_vs_corr_lam_0_01.h5"

correlations = h5open(fn, "r") do fid
    [read(fid, "rep$(rep)/dmsn/correlations") for rep=1:n_reps]
end
distances = h5open(fn, "r") do fid
    [read(fid, "rep$(rep)/dmsn/distances") for rep=1:n_reps]
end
correlations = vcat(correlations...)
distances = vcat(distances...)

#correlations = Random.shuffle(correlations)

bin_size = 0.02
bin = floor.(distances[:]/bin_size)
grouped = Vector{Float64}[]
for b=1:maximum(bin)
    push!(grouped, correlations[:][bin .== b])
end

p1 = plot(bin_size .* (1:maximum(bin)), mean.(grouped),
          ylim=(-0.05, 0.3), legend=false, ribbon=2*StatsBase.sem.(grouped))
p2 = plot(bin_size .* (1:maximum(bin)), length.(grouped))#, yaxis=:log)
l = @layout [a;b]
plot(p1, p2, layout=l)
 

