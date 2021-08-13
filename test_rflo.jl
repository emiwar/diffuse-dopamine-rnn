using Plots
include("rflo.jl")

t_max = 100
theta = range(0,2pi;length=t_max)
y_target = @. sin(theta)+0.5sin(2theta)+0.25sin(4theta)
inp = ones(t_max)
net = RFLO_NET(1, 50, 1; eta_in=5e-3, eta_rec=5e-3, eta_out=5e-3)
loss = Float64[]
@gif for ep=1:2000
    y, h = episode!(net, inp, y_target)
    push!(loss, sum((y.-y_target).^2)/t_max)
    l = @layout [[a; b] c{0.2w}]
    p = plot(y, label="y", ylim=(-1.5, 1.5), title="Episode $ep", layout=l)
    plot!(p[1], y_target, label="y_target")
    plot!(p[2], h, legend=false)
    plot!(p[3], loss, xlim=(0, 500), ylim=(1e-6, 2e0), xticks=[0,1000,2000], legend=false, yaxis=:log)
end
