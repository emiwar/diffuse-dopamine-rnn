using LinearAlgebra
using Plots
include("bg_net_v2.jl")

net = BgNet(200, 2, 0.0)

for pop in pop_order(net)
    net[pop].v = rand(size(net[pop])) .- 1
end
T = 500
log = NamedTuple(pop=>zeros(size(net[pop]), T) for pop in pop_order(net))

for t=1:T
    step!(net)
    for pop in pop_order(net)
        log[pop][:, t] = net[pop].r
    end
end

y = 1
yticks = Float64[]
p = plot()
for pop in pop_order(net)
    dy = size(net[pop])
    heatmap!(p, 1:T, y:(y+dy-1),  log[pop], clim=(-1, 1))
    push!(yticks, y+0.5dy)
    y += dy+5
end
plot!(p, yticks=(yticks, pop_order(net)))
