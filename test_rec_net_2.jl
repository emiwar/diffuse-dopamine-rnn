using Plots
include("rec_net_2.jl")

t_max = 100
theta = range(0,2pi;length=t_max)
y_target = @. sin(theta)+0.5sin(2theta)+0.25sin(4theta)
inp = ones(t_max)
net = REC_NET(1, 50, 1; eta_in=2e-4, eta_rec=2e-4, eta_out=4e-4)
loss = Float64[]
n_episodes = 2000
@gif for ep=1:n_episodes
    y, h = episode!(net, inp, y_target)
    push!(loss, sum((y.-y_target).^2)/t_max)
    l = @layout [[a; b] c{0.2w}]
    p = plot(y, label="y", ylim=(-1.5, 1.5), title="Episode $ep", layout=l)
    plot!(p[1], y_target, label="y_target")
    plot!(p[2], h, legend=false)
    plot!(p[3], loss, xlim=(0, n_episodes), ylim=(1e-5, 1e0),
          xticks=[0,div(n_episodes, 2),n_episodes], legend=false, yaxis=:log)
end
