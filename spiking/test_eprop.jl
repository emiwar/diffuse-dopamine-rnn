using Plots

include("eprop.jl")

striatum = SpikingLayer(:striatum, 100, 1; eta=5e-4)
bgNet = StriatumOnly(striatum, randn(100)/20, 0, 5e-4)


f = 2*pi/200
targets = []
outputs = []
spike_times = Float64[]
spike_ids = Int64[]
for t=1:2000
    target = sin(f*t)+0.5sin(2*f*t)+0.25sin(4*f*t)+2
    output = step!(bgNet, target)
    push!(targets, target)
    push!(outputs, output)
    spiking = findall(bgNet.striatum.last_spike.==0)
    spike_ids = vcat(spike_ids, spiking)
    spike_times = vcat(spike_times, t*ones(length(spiking)))
end

p = plot(outputs, label="output")
plot!(p, targets, label="target")
s = scatter(spike_times, spike_ids, markersize=1)
l = @layout [a; b]
plot(s, p, layout=l)