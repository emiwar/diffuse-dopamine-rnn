using HDF5
using Statistics
using Plots

correlations = h5open("data/dist_vs_corr.h5") do fid
    [read(fid, "rep$(rep)/dmsn/correlations") for rep=1:49]
end
distances = h5open("data/dist_vs_corr.h5") do fid
    [read(fid, "rep$(rep)/dmsn/distances") for rep=1:49]
end
correlations = vcat(correlations...)
distances = vcat(distances...)

bin_size = 0.01
bin = round.(distances[:]/bin_size)
grouped = Vector{Float64}[]
for b=1:maximum(bin)
    push!(grouped, correlations[:][bin .== b])
end

p1 = plot(bin_size .* (1:maximum(bin)), mean.(grouped))
p2 = plot(bin_size .* (1:maximum(bin)), length.(grouped))#, yaxis=:log)
l = @layout [a;b]
plot(p1, p2, layout=l)