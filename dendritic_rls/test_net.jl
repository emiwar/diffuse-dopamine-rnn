using Plots
include("reference_force.jl")

t_max = 100
theta = range(0,2pi;length=t_max)
y_target = @. sin(theta)+0.5sin(2theta)+0.25sin(4theta)
net = ONLINE_ECHO_NET(1, 50, 1; eta=5e-3, g_rec=0.5, g_in=0.5)
y, h = episode!(net, t_max)#, y_target)
l = @layout [a;b]
p = plot(y, label="y", ylim=(-1.5, 1.5), title="Untrained", layout=l)
plot!(p[1], y_target, label="y_target")
plot!(p[2], h, legend=false)


#net = ONLINE_ECHO_NET(1, 50, 1; eta=1e-3, g_rec=0.5, g_in=0.5, clamp_target=true)
net = FORCE_NET(1, 50, 1; g_rec=0.5, g_in=0.5, alpha=0.5, clamp_target=true)
loss = Float64[]
n_episodes = 100
@gif for ep=1:n_episodes
    y, h = episode!(net, y_target)
    push!(loss, sum((y.-y_target).^2)/t_max)
    l = @layout [[a; b] c{0.2w}]
    p = plot(y, label="y", ylim=(-1.5, 1.5), title="Episode $ep", layout=l)
    plot!(p[1], y_target, label="y_target")
    plot!(p[2], h, legend=false)
    plot!(p[3], loss, xlim=(0, n_episodes), ylim=(1e-5, 1e0),
          xticks=[0,div(n_episodes, 2),n_episodes], legend=false, yaxis=:log)
end

