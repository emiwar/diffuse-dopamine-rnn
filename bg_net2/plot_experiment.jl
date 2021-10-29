using HDF5
using Plots
using Statistics
using DataFrames
using StatsPlots

function readAsDataFrame(fn, onlyFinal=true)
    data = DataFrame[]
    h5open(fn, "r") do fid
        i = 1
        while true
            if !haskey(fid, "run$i")
                break
            end
            losses = fid["run$i"]
            attrs = attributes(fid["run$i"])
            d = Dict(k=>read(attrs, k) for k in keys(attrs))
            if onlyFinal
                d["loss"] = losses[end]
            else
                d["loss"] = read(losses)
                d["trial"] = 1:length(d["loss"])
            end
            push!(data, DataFrame(d))
            i += 1
        end
    end
    return vcat(data...)
end

function plotSeries!(df::DataFrame, x::Symbol, y::Symbol, group::Symbol;
                    kwargs...)
    
    for g in unique(df[:, group])
        filtered = df[df[:, group] .== g, :]
        gpy = DataFrames.groupby(filtered, x)
        combined = combine(gpy, :loss=>median=>:mu, :loss=>std=>:sig,
                           :loss=>(q->quantile(q, 0.25))=>:low_q,
                           :loss=>(q->quantile(q, 0.75))=>:high_q)
        ribbon = (combined[:, :mu]-combined[:, :low_q],
                  combined[:, :high_q]-combined[:, :mu])
        plot!(combined[:, x], combined[:, :mu],
              ribbon=ribbon, label=g)
    end
    plot!(;kwargs...)
end

function plotSeries(df::DataFrame, x::Symbol, y::Symbol, group::Symbol;
                    kwargs...)
    plot()
    plotSeries!(df, x, y, group; kwargs...)
end

#runs = readAsDataFrame("data/comparison_dopamine.h5")
#@df runs boxplot(string.(:striatumUpdate), :loss)