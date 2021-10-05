using LinearAlgebra
using Plots
using ProgressBars
include("bg_net_v2.jl")

net = BgNet(200, 2, 0.0)
for pop in pop_order(net)
    net[pop].v = rand(size(net[pop])) .- 1
end


T = 200
log = NamedTuple(pop=>zeros(size(net[pop]), T) for pop in pop_order(net))
for t=1:T
    step!(net)
    for pop in pop_order(net)
        log[pop][:, t] = net[pop].r
    end
    #net[:snr].r = target_fcn(t)
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


net = BgNet(200, 2, 2e-3)
for pop in pop_order(net)
    net[pop].v = rand(size(net[pop])) .- 1
end


##
base_period = 200
target_fcn(t) = 0.5*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                 0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)]


T_train = 20000
losses = Float64[]
loss = 0.0
for t=ProgressBar(1:T_train)
    step!(net, target_fcn(t))
    loss += sum((net[:snr].r .- target_fcn(t)).^2)
    #net[:snr].r = target_fcn(t)
    if mod(t, base_period) == 0
        push!(losses, loss)
        loss = 0.0
    end
end
