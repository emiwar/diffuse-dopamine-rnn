using HDF5
using Statistics
using Plots

correlations = h5open("dist_vs_corr.h5") do fid
    [read(fid, "rep$(rep)/dmsn/correlations") for rep=1:100]
end
distances = h5open("dist_vs_corr.h5") do fid
    [read(fid, "rep$(rep)/dmsn/distances") for rep=1:100]
end
correlations = vcat(correlations...)
distances = vcat(distances...)

bin_size = 0.05
bin = round.(distances[:]/bin_size)
grouped = Vector{Float64}[]
for b=1:floor(Int64, 1/bin_size)
    push!(grouped, correlations[:][bin .== b])
end

plot(mean.(grouped))